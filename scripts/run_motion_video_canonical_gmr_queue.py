#!/usr/bin/env python3
"""Run the no-clobber CPU-only canonical-beta GVHMR -> A3 GMR queue.

This lane is deliberately separate from ``run_motion_video_gmr_queue.py``.
The legacy queue hard-codes ``diagnostic_video_betas``; reusing it for the
materialized cohort-median PTs would silently create false body-shape lineage.
This queue instead binds the materialization completion manifest, every input
PT, the exact GMR loader and its row-zero beta semantics, the neutral SMPL-X
body model, the clean GMR source/bundle, and the structural auditor.

A pass remains diagnostic and ``formal_eligible=false``.  It authorizes only
the next offline grounding/collision/dynamics gates, never robot execution.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_motion_video_gmr_queue import (
    QueueError,
    atomic_json,
    build_command,
    build_environment,
    git_clean_head,
    python_fingerprint,
    sha256_file,
    utc_now,
    verify_source_bundle,
)


BODY_SHAPE_CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a JSON mapping: {path}")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise QueueError(f"{label} must be a lowercase SHA-256")
    return value


def require_file_binding(binding: Any, label: str) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise QueueError(f"{label} must be a mapping")
    if not isinstance(binding.get("path"), str):
        raise QueueError(f"{label}.path must be a string")
    require_sha(binding.get("sha256"), f"{label}.sha256")
    if not isinstance(binding.get("bytes"), int) or binding["bytes"] <= 0:
        raise QueueError(f"{label}.bytes must be a positive integer")
    return binding


def validate_plan(plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    require_sha(expected_plan_sha256, "--expected-plan-sha256")
    actual = sha256_file(plan_path)
    if actual != expected_plan_sha256:
        raise QueueError(f"plan sha256 {actual} != expected {expected_plan_sha256}")
    plan = read_json(plan_path, "canonical GMR plan")
    if plan.get("schema_version") != 1:
        raise QueueError("canonical GMR plan schema_version must be 1")
    if plan.get("status") != "preregistered_not_executed":
        raise QueueError("canonical GMR plan must remain preregistered_not_executed")
    if plan.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise QueueError(f"body_shape_contract must be {BODY_SHAPE_CONTRACT}")
    if plan.get("formal_eligible") is not False:
        raise QueueError("canonical GMR plan must explicitly remain formal_eligible=false")
    if plan.get("a3_calibrated") is not False or plan.get("measured_height_m") is not None:
        raise QueueError("canonical GMR plan requires a3_calibrated=false and measured_height_m=null")
    blockers = plan.get("formal_blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or not all(isinstance(item, str) and item.strip() for item in blockers)
    ):
        raise QueueError("formal_blockers must be a non-empty string list")

    execution = plan.get("execution_contract")
    if not isinstance(execution, dict):
        raise QueueError("execution_contract must be a mapping")
    if execution.get("cpu_only") is not True or execution.get("CUDA_VISIBLE_DEVICES") != "":
        raise QueueError("canonical GMR execution must be CPU-only with CUDA_VISIBLE_DEVICES empty")
    for field in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if not isinstance(execution.get(field), int) or execution[field] <= 0:
            raise QueueError(f"execution_contract.{field} must be a positive integer")
    if execution.get("warmup_threshold_strict_lt") != 0.0001:
        raise QueueError("warmup_threshold_strict_lt must be exactly 0.0001")
    if execution.get("warmup_max_rounds") != 200:
        raise QueueError("warmup_max_rounds must be exactly 200")
    if execution.get("robot") != "agibot_a3" or execution.get("target_fps") != 30:
        raise QueueError("canonical GMR plan must bind agibot_a3 at 30 Hz")

    source = plan.get("source_materialization")
    if not isinstance(source, dict):
        raise QueueError("source_materialization must be a mapping")
    require_file_binding(source.get("completion_manifest"), "source_materialization.completion_manifest")
    require_file_binding(source.get("canonical_betas_artifact"), "source_materialization.canonical_betas_artifact")
    require_sha(source.get("canonical_vector_sha256"), "source_materialization.canonical_vector_sha256")
    if source.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise QueueError("source materialization body-shape contract mismatch")

    gmr = plan.get("gmr_source_contract")
    if not isinstance(gmr, dict):
        raise QueueError("gmr_source_contract must be a mapping")
    if not isinstance(gmr.get("root"), str):
        raise QueueError("gmr_source_contract.root must be a string")
    if not isinstance(gmr.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", gmr["commit"]):
        raise QueueError("gmr_source_contract.commit must be a lowercase git commit")
    for field in ("bundle", "entrypoint", "loader", "neutral_smplx_model"):
        require_file_binding(gmr.get(field), f"gmr_source_contract.{field}")
    semantics = gmr.get("loader_semantics")
    required_semantics = {
        "field": "smpl_params_global.betas",
        "selection": "betas[0].detach().cpu().numpy()[:10]",
        "selected_components": 10,
        "zero_padding": False,
        "height_formula_not_calibration": "1.66 + 0.1 * selected_betas[0]",
    }
    if semantics != required_semantics:
        raise QueueError("gmr_source_contract.loader_semantics does not match the audited loader")

    tools = plan.get("tool_contract")
    if not isinstance(tools, dict):
        raise QueueError("tool_contract must be a mapping")
    for field in ("queue", "legacy_queue_dependency", "result_auditor"):
        require_file_binding(tools.get(field), f"tool_contract.{field}")

    output = plan.get("output_contract")
    if not isinstance(output, dict):
        raise QueueError("output_contract must be a mapping")
    if not isinstance(output.get("output_root"), str) or not isinstance(output.get("state_dir"), str):
        raise QueueError("output_contract output_root/state_dir must be strings")
    required_output = {
        "result_suffix": ".diagnostic_cohort_median_betas.gmr.pkl",
        "output_root_must_not_exist": True,
        "state_dir_must_not_exist": True,
        "no_clobber": True,
        "stop_on_first_failure": True,
    }
    for field, expected in required_output.items():
        if output.get(field) != expected:
            raise QueueError(f"output_contract.{field} must be {expected!r}")

    order = plan.get("processing_order")
    rows = plan.get("inputs")
    if not isinstance(order, list) or not isinstance(rows, list) or len(rows) != 10:
        raise QueueError("canonical GMR plan requires ten ordered inputs")
    ids: list[str] = []
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise QueueError(f"inputs[{index}] must be a mapping")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
            raise QueueError(f"inputs[{index}].asset_id is unsafe: {asset_id!r}")
        if asset_id in ids:
            raise QueueError(f"duplicate input asset_id: {asset_id}")
        ids.append(asset_id)
        binding = require_file_binding(row.get("input"), f"inputs[{index}].input")
        if binding["path"] in seen_paths:
            raise QueueError(f"duplicate input path: {binding['path']}")
        seen_paths.add(binding["path"])
        if not isinstance(row.get("frames"), int) or row["frames"] <= 1:
            raise QueueError(f"inputs[{index}].frames must be an integer > 1")
        if row.get("canonical_vector_sha256") != source.get("canonical_vector_sha256"):
            raise QueueError(f"inputs[{index}] canonical vector SHA mismatch")
    if order != ids:
        raise QueueError("processing_order must exactly equal the ordered input asset ids")
    return plan


def verify_bound_file(binding: dict[str, Any], label: str) -> Path:
    path = Path(binding["path"]).resolve()
    if not path.is_file() or path.stat().st_size != binding["bytes"]:
        raise QueueError(
            f"{label} missing or bytes {path.stat().st_size if path.is_file() else 0} "
            f"!= {binding['bytes']}: {path}"
        )
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise QueueError(f"{label} sha256 {actual} != {binding['sha256']}: {path}")
    return path


def verify_materialization(plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = plan["source_materialization"]
    completion_path = verify_bound_file(source["completion_manifest"], "materialization manifest")
    canonical_path = verify_bound_file(source["canonical_betas_artifact"], "canonical beta artifact")
    completion = read_json(completion_path, "materialization completion manifest")
    if completion.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise QueueError("materialization completion body-shape contract mismatch")
    if completion.get("formal_eligible") is not False or completion.get("a3_calibrated") is not False:
        raise QueueError("materialization completion must remain formal-ineligible and uncalibrated")
    artifact = completion.get("canonical_betas_artifact")
    if not isinstance(artifact, dict):
        raise QueueError("materialization completion lacks canonical_betas_artifact")
    if (
        Path(str(artifact.get("path"))).resolve() != canonical_path
        or artifact.get("sha256") != source["canonical_betas_artifact"]["sha256"]
        or artifact.get("vector_sha256") != source["canonical_vector_sha256"]
    ):
        raise QueueError("canonical beta artifact binding mismatch")
    completion_rows = completion.get("results")
    if not isinstance(completion_rows, list):
        raise QueueError("materialization completion results must be a list")
    by_id = {
        row.get("asset_id"): row for row in completion_rows if isinstance(row, dict)
    }
    if len(by_id) != len(completion_rows):
        raise QueueError("materialization completion has malformed/duplicate asset ids")

    verified: list[dict[str, Any]] = []
    for row in plan["inputs"]:
        asset_id = row["asset_id"]
        remote = by_id.get(asset_id)
        if not isinstance(remote, dict):
            raise QueueError(f"materialization completion lacks {asset_id}")
        binding = row["input"]
        expected = {
            "output_path": binding["path"],
            "output_sha256": binding["sha256"],
            "output_bytes": binding["bytes"],
            "frames": row["frames"],
            "output_canonical_vector_sha256": row["canonical_vector_sha256"],
            "frame_beta_max_abs_deviation_from_video_median": 0.0,
            "non_beta_bit_exact": True,
        }
        for field, value in expected.items():
            if remote.get(field) != value:
                raise QueueError(f"materialization {asset_id}.{field} mismatch")
        beta_contract = remote.get("output_beta_contract")
        if not isinstance(beta_contract, dict) or beta_contract.get("shape") != [row["frames"], 10]:
            raise QueueError(f"materialization {asset_id} beta shape mismatch")
        if beta_contract.get("dtype") != "torch.float32" or beta_contract.get("shape_contract") != "frames_by_10":
            raise QueueError(f"materialization {asset_id} beta dtype/shape contract mismatch")
        input_path = verify_bound_file(binding, f"canonical GVHMR input {asset_id}")
        verified.append({**row, "input_path": str(input_path)})
    if set(by_id) != {row["asset_id"] for row in verified}:
        raise QueueError("materialization completion contains an unexpected asset set")
    return completion, verified


def select_rows(rows: list[dict[str, Any]], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return rows
    if len(requested) != len(set(requested)):
        raise QueueError("--asset-id values must be unique")
    by_id = {row["asset_id"]: row for row in rows}
    missing = [asset_id for asset_id in requested if asset_id not in by_id]
    if missing:
        raise QueueError(f"requested asset ids are absent from the plan: {missing}")
    return [by_id[asset_id] for asset_id in requested]


def verify_tool_contract(plan: dict[str, Any]) -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    expected_paths = {
        "queue": here,
        "legacy_queue_dependency": here.parent / "run_motion_video_gmr_queue.py",
        "result_auditor": here.parent / "audit_gmr_result.py",
    }
    for name, path in expected_paths.items():
        binding = plan["tool_contract"][name]
        if Path(binding["path"]).name != path.name:
            raise QueueError(f"tool_contract.{name}.path basename mismatch")
        if not path.is_file() or path.stat().st_size != binding["bytes"]:
            raise QueueError(f"tool_contract.{name} local bytes mismatch: {path}")
        if sha256_file(path) != binding["sha256"]:
            raise QueueError(f"tool_contract.{name} local SHA mismatch: {path}")
    return expected_paths["legacy_queue_dependency"], expected_paths["result_auditor"]


def verify_gmr_source(plan: dict[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    gmr = plan["gmr_source_contract"]
    root = Path(gmr["root"]).resolve()
    if git_clean_head(root) != gmr["commit"]:
        raise QueueError(f"clean GMR HEAD must be {gmr['commit']}")
    bundle_path = verify_bound_file(gmr["bundle"], "GMR recovery bundle")
    bundle = verify_source_bundle(root, bundle_path, gmr["commit"])
    if bundle["sha256"] != gmr["bundle"]["sha256"] or bundle["bytes"] != gmr["bundle"]["bytes"]:
        raise QueueError("verified GMR recovery bundle binding mismatch")
    entrypoint = verify_bound_file(gmr["entrypoint"], "GMR entrypoint")
    loader = verify_bound_file(gmr["loader"], "GMR SMPL loader")
    neutral_model = verify_bound_file(gmr["neutral_smplx_model"], "neutral SMPL-X model")
    if entrypoint != root / "scripts" / "gvhmr_to_robot.py":
        raise QueueError("GMR entrypoint is not under the bound clean root")
    if loader != root / "general_motion_retargeting" / "utils" / "smpl.py":
        raise QueueError("GMR loader is not under the bound clean root")
    expected_model = root / "assets" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if neutral_model != expected_model:
        raise QueueError("neutral SMPL-X model path is not the loader default")
    return root, entrypoint, loader, bundle


def verify_python(plan: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    expected = plan["execution_contract"].get("python_environment")
    if not isinstance(expected, dict) or not isinstance(expected.get("executable"), str):
        raise QueueError("execution_contract.python_environment is malformed")
    require_sha(expected.get("pip_freeze_sha256"), "python_environment.pip_freeze_sha256")
    python = Path(expected["executable"]).resolve()
    if not python.is_file():
        raise QueueError(f"motion Python is missing: {python}")
    actual = python_fingerprint(python)
    if actual != expected:
        raise QueueError(f"motion Python fingerprint mismatch: actual={actual}, expected={expected}")
    return python, actual


def run_one(
    row: dict[str, Any],
    *,
    gmr_root: Path,
    entrypoint: Path,
    python: Path,
    output_root: Path,
    state_dir: Path,
    result_auditor: Path,
    processing_contract: dict[str, Any],
    timeout_seconds: float,
) -> bool:
    asset_id = row["asset_id"]
    source = Path(row["input_path"])
    suffix = ".diagnostic_cohort_median_betas.gmr.pkl"
    output = output_root / f"{asset_id}{suffix}"
    binding_path = state_dir / "bindings" / f"{asset_id}.json"
    log_path = state_dir / "logs" / f"{asset_id}.log"
    audit_path = state_dir / "audits" / f"{asset_id}.json"
    for path in (output, binding_path, log_path, audit_path):
        if path.exists():
            raise QueueError(f"no-clobber target already exists for {asset_id}: {path}")

    command = build_command(python, entrypoint, source, output)
    execution = processing_contract["execution_contract"]
    env = build_environment(
        gmr_root,
        execution["OMP_NUM_THREADS"],
        execution["MKL_NUM_THREADS"],
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "asset_id": asset_id,
        "source_path": str(source),
        "source_sha256": row["input"]["sha256"],
        "source_bytes": row["input"]["bytes"],
        "source_frames": row["frames"],
        "canonical_vector_sha256": row["canonical_vector_sha256"],
        "body_shape_contract": BODY_SHAPE_CONTRACT,
        "a3_calibrated": False,
        "measured_height_m": None,
        "formal_eligible": False,
        "formal_blockers": processing_contract["formal_blockers"],
        "processing_contract": processing_contract,
        "command": command,
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(gmr_root),
            "OMP_NUM_THREADS": str(execution["OMP_NUM_THREADS"]),
            "MKL_NUM_THREADS": str(execution["MKL_NUM_THREADS"]),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        },
        "started_utc": utc_now(),
        "output_path": str(output),
        "run_log_path": str(log_path),
        "structural_audit_path": str(audit_path),
    }
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(binding_path, payload)
    print(f"[canonical-gmr] START {asset_id} source={source.name}", flush=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=gmr_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload.update(status="failed", completed_utc=utc_now(), failure=f"GMR command failed: {exc}")
        atomic_json(binding_path, payload)
        return False
    payload.update(returncode=result.returncode, completed_utc=utc_now())
    if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        payload.update(
            status="failed",
            failure=(
                f"returncode={result.returncode}, output_exists={output.is_file()}, "
                f"output_bytes={output.stat().st_size if output.is_file() else 0}"
            ),
        )
        atomic_json(binding_path, payload)
        return False

    audit_command = [
        str(python),
        str(result_auditor),
        "--result",
        str(output),
        "--expected-frames",
        str(row["frames"]),
        "--run-log",
        str(log_path),
        "--warmup-threshold",
        str(execution["warmup_threshold_strict_lt"]),
        "--warmup-max-rounds",
        str(execution["warmup_max_rounds"]),
        "--body-shape-contract",
        BODY_SHAPE_CONTRACT,
        "--json-out",
        str(audit_path),
    ]
    payload["audit_command"] = audit_command
    try:
        audit_result = subprocess.run(
            audit_command,
            cwd=gmr_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=min(timeout_seconds, 120.0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload.update(status="failed", failure=f"GMR audit failed: {exc}")
        atomic_json(binding_path, payload)
        return False
    payload.update(
        audit_returncode=audit_result.returncode,
        audit_stdout=audit_result.stdout[-4000:],
        audit_stderr=audit_result.stderr[-4000:],
    )
    if audit_result.returncode != 0 or not audit_path.is_file():
        payload.update(status="failed", failure="GMR structural audit did not pass")
        atomic_json(binding_path, payload)
        return False
    audit = read_json(audit_path, f"GMR audit for {asset_id}")
    output_sha = sha256_file(output)
    log_sha = sha256_file(log_path)
    if (
        audit.get("status") != "pass"
        or audit.get("body_shape_contract") != BODY_SHAPE_CONTRACT
        or audit.get("formal_eligible") is not False
        or audit.get("result_sha256") != output_sha
        or audit.get("run_log_sha256") != log_sha
        or audit.get("actual_frames") != row["frames"]
    ):
        payload.update(status="failed", failure="GMR audit content/SHA/lineage mismatch")
        atomic_json(binding_path, payload)
        return False
    payload.update(
        status="complete",
        output_bytes=output.stat().st_size,
        output_sha256=output_sha,
        run_log_sha256=log_sha,
        structural_audit_sha256=sha256_file(audit_path),
        warmup=audit["warmup"],
    )
    atomic_json(binding_path, payload)
    print(f"[canonical-gmr] COMPLETE {asset_id} sha256={output_sha[:12]}...", flush=True)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--asset-id", action="append", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_state: dict[str, Any] | None = None
    queue_state_path: Path | None = None
    try:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
            raise QueueError("timeout-seconds must be finite and positive")
        plan_path = args.plan.resolve()
        plan = validate_plan(plan_path, args.expected_plan_sha256)
        _, result_auditor = verify_tool_contract(plan)
        completion, rows = verify_materialization(plan)
        rows = select_rows(rows, args.asset_id)
        gmr_root, entrypoint, loader, bundle = verify_gmr_source(plan)
        python, python_contract = verify_python(plan)
        output_root = Path(plan["output_contract"]["output_root"]).resolve()
        state_dir = Path(plan["output_contract"]["state_dir"]).resolve()
        if output_root.exists() or state_dir.exists():
            raise QueueError(
                f"no-clobber roots must not exist: output={output_root.exists()} "
                f"state={state_dir.exists()}"
            )
        output_root.mkdir(parents=True)
        state_dir.mkdir(parents=True)
        lock_handle = (state_dir / "queue.lock").open("x+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        processing_contract = {
            "plan_path": str(plan_path),
            "plan_sha256": args.expected_plan_sha256,
            "queue_tool_sha256": plan["tool_contract"]["queue"]["sha256"],
            "legacy_queue_dependency_sha256": plan["tool_contract"]["legacy_queue_dependency"]["sha256"],
            "result_auditor_sha256": plan["tool_contract"]["result_auditor"]["sha256"],
            "materialization_manifest_path": plan["source_materialization"]["completion_manifest"]["path"],
            "materialization_manifest_sha256": plan["source_materialization"]["completion_manifest"]["sha256"],
            "canonical_betas_artifact_sha256": plan["source_materialization"]["canonical_betas_artifact"]["sha256"],
            "canonical_vector_sha256": plan["source_materialization"]["canonical_vector_sha256"],
            "body_shape_contract": BODY_SHAPE_CONTRACT,
            "gmr_commit": plan["gmr_source_contract"]["commit"],
            "gmr_worktree_clean": True,
            "gmr_source_bundle": bundle,
            "gmr_entrypoint_path": str(entrypoint),
            "gmr_entrypoint_sha256": plan["gmr_source_contract"]["entrypoint"]["sha256"],
            "gmr_loader_path": str(loader),
            "gmr_loader_sha256": plan["gmr_source_contract"]["loader"]["sha256"],
            "gmr_loader_semantics": plan["gmr_source_contract"]["loader_semantics"],
            "neutral_smplx_model": plan["gmr_source_contract"]["neutral_smplx_model"],
            "python_environment": python_contract,
            "execution_contract": plan["execution_contract"],
            "formal_eligible": False,
            "formal_blockers": plan["formal_blockers"],
        }
        queue_state = {
            "schema_version": 1,
            "status": "running",
            "plan_path": str(plan_path),
            "plan_sha256": args.expected_plan_sha256,
            "asset_ids": [row["asset_id"] for row in rows],
            "body_shape_contract": BODY_SHAPE_CONTRACT,
            "formal_eligible": False,
            "processing_contract": processing_contract,
            "materialization_completed_utc": completion.get("completed_utc"),
            "started_utc": utc_now(),
        }
        queue_state_path = state_dir / "queue_state.json"
        atomic_json(queue_state_path, queue_state)
        for row in rows:
            if not run_one(
                row,
                gmr_root=gmr_root,
                entrypoint=entrypoint,
                python=python,
                output_root=output_root,
                state_dir=state_dir,
                result_auditor=result_auditor,
                processing_contract=processing_contract,
                timeout_seconds=args.timeout_seconds,
            ):
                queue_state.update(
                    status="failed",
                    failed_asset_id=row["asset_id"],
                    completed_utc=utc_now(),
                )
                atomic_json(queue_state_path, queue_state)
                return 1
        queue_state.update(status="complete", completed_utc=utc_now())
        atomic_json(queue_state_path, queue_state)
        print(f"[canonical-gmr] PASS: {len(rows)} canonical-beta diagnostic assets", flush=True)
        return 0
    except (QueueError, OSError, subprocess.SubprocessError) as exc:
        if queue_state is not None and queue_state_path is not None:
            queue_state.update(status="fatal", failure=str(exc), completed_utc=utc_now())
            atomic_json(queue_state_path, queue_state)
        print(f"[canonical-gmr] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
