#!/usr/bin/env python3
"""Inspect or consume the versioned S0/M0 exact-GMR attempt v2.

Attempt v1 is permanently blocked because its whole-environment pip-freeze
digest was recorded without the normalized bytes needed to reproduce it.  This
consumer keeps v1 immutable, reuses its reviewed geometry/result machinery,
and adds an auditable pip snapshot plus exact direct-import metadata bindings.
It never authorizes schema-2, simulation, training, TOPP, Gate3, hardware, or a
strike claim.
"""

from __future__ import annotations

import argparse
import base64
import copy
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterator


def _load_v1_module() -> Any:
    path = Path(__file__).with_name("run_motion_s0_m0_exact_gmr.py")
    spec = importlib.util.spec_from_file_location("_motion_s0_m0_exact_gmr_v1_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed v1 base consumer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_v1_module()

ATTEMPT_VERSION = 2
DIRECT_IMPORTS = ("numpy", "torch", "mujoco", "smplx", "scipy")
SNAPSHOT_NORMALIZATION = "strip nonempty lines; bytewise sort; join LF; append one LF"
SNAPSHOT_PATH = "configs/motion_s0_m0_exact_gmr_pip_freeze_56b0f8af_v2.txt"
RUNTIME_PATH = "configs/motion_s0_m0_exact_gmr_runtime_20260714_v2.json"
EXPECTED_PLAN_IDS = {
    "s0_static_high_press": "motion-exact-gmr-static-high-press-s0-20260714-v2",
    "m0_lateral_teachers": "motion-exact-gmr-lateral-teachers-m0-20260714-v2",
}
EXPECTED_OUTPUT_ROOTS = {
    "s0_static_high_press": "/workspace/codexschema/motion_video_intake_20260713_s0/exact_gmr_v2",
    "m0_lateral_teachers": "/workspace/codexschema/motion_video_intake_20260713_m0/exact_gmr_v2",
}
SERIALIZATION_LOCK_PATH = "/workspace/codexschema/motion_s0_m0_exact_gmr_v2.consume.lock"
SERIALIZATION_LOCK_PAYLOAD = b"motion-s0-m0-exact-gmr-v2-serialized-consume-lock\n"


def _path_key(path: Path) -> str:
    return os.path.abspath(path)


def _read_json_strict(path: Path, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise base.ContractError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except base.ContractError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise base.ContractError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise base.ContractError(f"{label} must be a JSON mapping: {path}")
    return value


def _validate_snapshot_contract(python: dict[str, Any], repo_root: Path) -> None:
    snapshot = python.get("pip_freeze_snapshot")
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "path",
        "bytes",
        "sha256",
        "line_count",
        "normalization",
    }:
        raise base.ContractError("v2 pip-freeze snapshot field closure changed")
    base.require_binding(snapshot, "python_environment.pip_freeze_snapshot")
    if snapshot["path"] != SNAPSHOT_PATH or Path(snapshot["path"]).is_absolute():
        raise base.ContractError("v2 pip-freeze snapshot must use its exact tracked path")
    if snapshot["normalization"] != SNAPSHOT_NORMALIZATION:
        raise base.ContractError("v2 pip-freeze snapshot normalization changed")
    if not isinstance(snapshot["line_count"], int) or snapshot["line_count"] <= 0:
        raise base.ContractError("v2 pip-freeze snapshot line_count must be positive")
    path = base.verify_regular_file(snapshot, "v2 pip-freeze snapshot", root=repo_root)
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise base.ContractError(f"v2 pip-freeze snapshot is not UTF-8: {exc}") from None
    if base.normalized_pip_freeze_bytes(text) != payload:
        raise base.ContractError("v2 pip-freeze snapshot bytes are not canonical bytewise-sorted LF")
    if len(payload.splitlines()) != snapshot["line_count"]:
        raise base.ContractError("v2 pip-freeze snapshot line_count mismatch")
    if snapshot["sha256"] != python.get("pip_freeze_sha256"):
        raise base.ContractError("v2 pip-freeze snapshot and Python fingerprint SHA differ")


def _validate_direct_import_contract(python: dict[str, Any]) -> None:
    direct = python.get("direct_imports")
    if not isinstance(direct, dict) or set(direct) != set(DIRECT_IMPORTS):
        raise base.ContractError("v2 direct-import closure must contain the exact five modules")
    for name in DIRECT_IMPORTS:
        row = direct[name]
        if not isinstance(row, dict) or set(row) != {
            "distribution_name",
            "version",
            "module_origin",
            "dist_info_root",
            "metadata",
            "record",
        }:
            raise base.ContractError(f"v2 direct-import {name} field closure changed")
        if row["distribution_name"] != name:
            raise base.ContractError(f"v2 direct-import {name} distribution name changed")
        if not isinstance(row["version"], str) or not row["version"]:
            raise base.ContractError(f"v2 direct-import {name} version must be non-empty")
        if not isinstance(row["dist_info_root"], str) or not Path(row["dist_info_root"]).is_absolute():
            raise base.ContractError(f"v2 direct-import {name} dist-info root must be absolute")
        for field in ("module_origin", "metadata", "record"):
            binding = base.require_binding(row[field], f"direct_imports.{name}.{field}")
            if not Path(binding["path"]).is_absolute():
                raise base.ContractError(f"v2 direct-import {name}.{field} path must be absolute")
        root = Path(row["dist_info_root"])
        if Path(row["metadata"]["path"]).parent != root or Path(row["record"]["path"]).parent != root:
            raise base.ContractError(f"v2 direct-import {name} metadata/RECORD escaped dist-info root")


def _validate_v2_runtime(runtime: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    if runtime.get("schema_version") != 1 or runtime.get("attempt_version") != ATTEMPT_VERSION:
        raise base.ContractError("v2 runtime must be schema 1, attempt 2")
    expected_top = {
        "schema_version",
        "attempt_version",
        "status",
        "scope",
        "tool_contract",
        "batch_serialization_contract",
        "ignored_gmr_source",
        "a3_robot_contract",
        "execution_contract",
    }
    if set(runtime) != expected_top:
        raise base.ContractError("v2 shared runtime field closure changed")
    execution = runtime.get("execution_contract")
    if not isinstance(execution, dict):
        raise base.ContractError("v2 execution_contract must be a mapping")
    python = execution.get("python_environment")
    if not isinstance(python, dict):
        raise base.ContractError("v2 Python environment must be a mapping")
    base_python_keys = {
        "executable",
        "executable_bytes",
        "executable_sha256",
        "version",
        "pip_version",
        "pip_freeze_command",
        "pip_freeze_normalization",
        "pip_freeze_sha256",
        "xrobot_utils_resolution",
    }
    if set(python) != base_python_keys | {"pip_freeze_snapshot", "direct_imports"}:
        raise base.ContractError("v2 Python environment field closure changed")
    sanitized = copy.deepcopy(runtime)
    sanitized.pop("attempt_version")
    sanitized.pop("batch_serialization_contract")
    sanitized["tool_contract"].pop("base_consumer_v1")
    sanitized_python = sanitized["execution_contract"]["python_environment"]
    sanitized_python.pop("pip_freeze_snapshot")
    sanitized_python.pop("direct_imports")
    base._validate_robot_contract(sanitized)
    base._validate_source_contract(sanitized)
    base._validate_execution_contract(sanitized)
    _validate_snapshot_contract(python, repo_root)
    _validate_direct_import_contract(python)
    _validate_serialization_contract(runtime["batch_serialization_contract"])
    return sanitized


def _verify_v2_tool_contract(runtime: dict[str, Any], repo_root: Path) -> None:
    tool = runtime.get("tool_contract")
    if not isinstance(tool, dict) or set(tool) != {
        "consumer",
        "base_consumer_v1",
        "result_auditor",
    }:
        raise base.ContractError("v2 tool contract field closure changed")
    paths = {
        "consumer": Path(__file__).resolve(),
        "base_consumer_v1": Path(base.__file__).resolve(),
        "result_auditor": (repo_root / "scripts" / "audit_gmr_result.py").resolve(),
    }
    for name, path in paths.items():
        binding = base.require_binding(tool.get(name), f"v2 tool_contract.{name}")
        if Path(binding["path"]).name != path.name:
            raise base.ContractError(f"v2 tool_contract.{name} basename mismatch")
        base.verify_regular_file(binding, f"v2 tool_contract.{name}", root=repo_root)


def _validate_serialization_contract(contract: Any) -> None:
    expected = {
        "mode": "advisory_flock_exclusive_across_s0_m0_consume",
        "lock_path": SERIALIZATION_LOCK_PATH,
        "lock_payload_utf8": SERIALIZATION_LOCK_PAYLOAD.decode("utf-8").rstrip("\n"),
        "inspect_writes_lock": False,
        "consume_batches": ["s0_static_high_press", "m0_lateral_teachers"],
        "batch_order_dependency": False,
    }
    if contract != expected:
        raise base.ContractError("v2 S0/M0 serialization contract changed")


def _verify_existing_serialization_lock(plan: dict[str, Any]) -> None:
    """Read an existing shared lock without creating it; absence is valid."""

    _validate_serialization_contract(plan["batch_serialization_contract"])
    path = Path(SERIALIZATION_LOCK_PATH)
    if not os.path.lexists(path):
        return
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise base.ContractError(f"v2 serialization lock must be a regular file: {path}")
    if path.read_bytes() != SERIALIZATION_LOCK_PAYLOAD:
        raise base.ContractError("v2 serialization lock payload drifted")


@contextmanager
def _exclusive_batch_lock(plan: dict[str, Any]) -> Iterator[None]:
    """Serialize S0 and M0 consume without making either depend on the other."""

    _validate_serialization_contract(plan["batch_serialization_contract"])
    path = Path(SERIALIZATION_LOCK_PATH)
    parent = base.verify_real_directory(path.parent, "v2 serialization lock parent")
    if path.parent.resolve() != parent:
        raise base.ContractError("v2 serialization lock parent resolution changed")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as exc:
        raise base.ContractError(f"cannot open v2 serialization lock: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise base.ContractError("v2 serialization lock must be one regular inode")
        fcntl.flock(fd, fcntl.LOCK_EX)
        payload = os.pread(fd, len(SERIALIZATION_LOCK_PAYLOAD) + 1, 0)
        if payload == b"":
            written = os.pwrite(fd, SERIALIZATION_LOCK_PAYLOAD, 0)
            if written != len(SERIALIZATION_LOCK_PAYLOAD):
                raise base.ContractError("short write while initializing v2 serialization lock")
            os.ftruncate(fd, len(SERIALIZATION_LOCK_PAYLOAD))
            os.fsync(fd)
            base.fsync_dir(parent)
        elif payload != SERIALIZATION_LOCK_PAYLOAD:
            raise base.ContractError("v2 serialization lock payload drifted")
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def validate_plan(plan_path: Path, expected_plan_sha256: str, repo_root: Path) -> dict[str, Any]:
    """Validate v2 while reusing v1's reviewed scientific invariants."""

    base.require_sha(expected_plan_sha256, "--expected-plan-sha256")
    if base.sha256_file(plan_path) != expected_plan_sha256:
        raise base.ContractError("v2 plan SHA does not match --expected-plan-sha256")
    raw_plan = _read_json_strict(plan_path, "v2 exact-GMR preregistration")
    if raw_plan.get("schema_version") != 1 or raw_plan.get("attempt_version") != ATTEMPT_VERSION:
        raise base.ContractError("v2 plan must be schema 1, attempt 2")
    if "attempt_version" not in raw_plan:
        raise base.ContractError("v2 plan lacks attempt_version")
    runtime_binding = base.require_binding(raw_plan.get("runtime_contract"), "v2 runtime_contract")
    if runtime_binding["path"] != RUNTIME_PATH:
        raise base.ContractError("v2 plan must bind the exact versioned runtime path")
    runtime_path = base.verify_regular_file(
        runtime_binding, "v2 shared exact-GMR runtime contract", root=repo_root
    )
    raw_runtime = _read_json_strict(runtime_path, "v2 shared exact-GMR runtime contract")
    sanitized_runtime = _validate_v2_runtime(raw_runtime, repo_root)
    sanitized_plan = copy.deepcopy(raw_plan)
    sanitized_plan.pop("attempt_version")

    original_read_json = base.read_json
    original_verify_tool = base.verify_tool_contract

    def read_json_override(path: Path, label: str) -> dict[str, Any]:
        key = _path_key(path)
        if key == _path_key(plan_path):
            return copy.deepcopy(sanitized_plan)
        if key == _path_key(runtime_path):
            return copy.deepcopy(sanitized_runtime)
        return original_read_json(path, label)

    base.read_json = read_json_override
    base.verify_tool_contract = lambda *_args, **_kwargs: None
    try:
        plan = base.validate_plan(plan_path, expected_plan_sha256, repo_root)
    finally:
        base.read_json = original_read_json
        base.verify_tool_contract = original_verify_tool

    _verify_v2_tool_contract(raw_runtime, repo_root)
    batch = plan["batch_kind"]
    if raw_plan.get("plan_id") != EXPECTED_PLAN_IDS[batch]:
        raise base.ContractError(f"v2 {batch} plan_id changed")
    if raw_plan["output_contract"].get("output_root") != EXPECTED_OUTPUT_ROOTS[batch]:
        raise base.ContractError(f"v2 {batch} must use its new no-clobber output root")
    if "/exact_gmr_v1" in raw_plan["output_contract"]["output_root"]:
        raise base.ContractError("v2 may never reuse the v1 output root")
    plan["attempt_version"] = ATTEMPT_VERSION
    plan["tool_contract"] = raw_runtime["tool_contract"]
    plan["execution_contract"] = raw_runtime["execution_contract"]
    plan["batch_serialization_contract"] = raw_runtime["batch_serialization_contract"]
    return plan


def _record_origin_entry_matches(record: Path, origin: Path, dist_root: Path) -> bool:
    try:
        relative = origin.relative_to(dist_root.parent).as_posix()
    except ValueError:
        return False
    with record.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row and row[0] == relative]
    if len(rows) != 1 or len(rows[0]) < 3:
        return False
    digest = hashlib.sha256(origin.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return rows[0][1] == f"sha256={encoded}" and rows[0][2] == str(origin.stat().st_size)


def _probe_direct_imports(python: Path, env: dict[str, str]) -> dict[str, Any]:
    code = r'''
import importlib, importlib.metadata as md, json
from pathlib import Path
names = ["numpy", "torch", "mujoco", "smplx", "scipy"]
rows = {}
for name in names:
    module = importlib.import_module(name)
    dist = md.distribution(name)
    root = Path(dist._path).resolve()
    rows[name] = {
        "distribution_name": name,
        "version": dist.version,
        "module_origin_path": str(Path(module.__file__).resolve()),
        "dist_info_root": str(root),
        "metadata_path": str((root / "METADATA").resolve()),
        "record_path": str((root / "RECORD").resolve()),
    }
print(json.dumps(rows, sort_keys=True))
'''
    result = subprocess.run(
        [str(python), "-c", code], env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise base.ContractError(f"v2 direct-import probe failed: {result.stderr[-2000:]}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise base.ContractError(f"v2 direct-import probe returned invalid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise base.ContractError("v2 direct-import probe result must be a mapping")
    return value


def verify_python_v2(plan: dict[str, Any], gmr_root: Path, repo_root: Path) -> Path:
    expected_python = plan["execution_contract"]["python_environment"]
    sanitized_plan = copy.deepcopy(plan)
    sanitized_python = sanitized_plan["execution_contract"]["python_environment"]
    sanitized_python.pop("pip_freeze_snapshot")
    sanitized_python.pop("direct_imports")
    python = base.verify_python(sanitized_plan, gmr_root)
    snapshot_binding = expected_python["pip_freeze_snapshot"]
    snapshot = base.verify_regular_file(snapshot_binding, "v2 pip-freeze snapshot", root=repo_root)
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze.returncode != 0:
        raise base.ContractError(f"v2 pip freeze failed: {freeze.stderr.strip()}")
    actual = base.normalized_pip_freeze_bytes(freeze.stdout)
    if actual != snapshot.read_bytes():
        raise base.ContractError("v2 normalized pip freeze differs from tracked snapshot bytes")

    env = base.build_environment(plan, gmr_root)
    actual_imports = _probe_direct_imports(python, env)
    expected_imports = expected_python["direct_imports"]
    if set(actual_imports) != set(DIRECT_IMPORTS):
        raise base.ContractError("v2 direct-import probe module set changed")
    for name in DIRECT_IMPORTS:
        expected = expected_imports[name]
        actual_row = actual_imports[name]
        exact_scalars = {
            "distribution_name": expected["distribution_name"],
            "version": expected["version"],
            "module_origin_path": expected["module_origin"]["path"],
            "dist_info_root": expected["dist_info_root"],
            "metadata_path": expected["metadata"]["path"],
            "record_path": expected["record"]["path"],
        }
        if actual_row != exact_scalars:
            raise base.ContractError(
                f"v2 direct-import {name} identity drift: actual={actual_row}, expected={exact_scalars}"
            )
        root = base.verify_real_directory(Path(expected["dist_info_root"]), f"{name} dist-info")
        origin = base.verify_regular_file(expected["module_origin"], f"{name} module origin")
        base.verify_regular_file(expected["metadata"], f"{name} METADATA")
        record = base.verify_regular_file(expected["record"], f"{name} RECORD")
        if not _record_origin_entry_matches(record, origin, root):
            raise base.ContractError(f"v2 direct-import {name} origin does not match bound RECORD")
    return python


def inspect_plan(plan: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    canonical_mjcf = base.verify_tree_contract(plan, repo_root)
    base.verify_a3_orders_and_sites(plan, repo_root, canonical_mjcf)
    gmr = base.verify_gmr_source(plan)
    python = verify_python_v2(plan, gmr["root"], repo_root)
    rows = base.verify_materialization(plan, repo_root)
    output_root = Path(plan["output_contract"]["output_root"])
    if output_root.exists() or output_root.is_symlink():
        raise base.ContractError(f"v2 no-clobber output root already exists: {output_root}")
    _verify_existing_serialization_lock(plan)
    return {
        "canonical_mjcf": canonical_mjcf,
        "gmr": gmr,
        "python": python,
        "rows": rows,
        "output_root": output_root,
    }


def _consume_locked(
    plan: dict[str, Any], plan_path: Path, expected_sha: str, repo_root: Path
) -> Path:
    inspected = inspect_plan(plan, repo_root)
    root: Path = inspected["output_root"]
    root.mkdir(parents=True, exist_ok=False)
    for name in ("outputs", "logs", "audits", "bindings"):
        (root / name).mkdir()
    base.fsync_dir(root)
    gmr = inspected["gmr"]
    python: Path = inspected["python"]
    auditor = (repo_root / "scripts" / "audit_gmr_result.py").resolve()
    env = base.build_environment(plan, gmr["root"])
    rows_out: list[dict[str, Any]] = []
    for row in inspected["rows"]:
        asset_id = row["asset_id"]
        source = Path(row["input_path"])
        output = root / "outputs" / f"{asset_id}{plan['output_contract']['result_suffix']}"
        log = root / "logs" / f"{asset_id}.log"
        audit = root / "audits" / f"{asset_id}.json"
        binding_path = root / "bindings" / f"{asset_id}.json"
        command = base.build_converter_command(plan, python, gmr["converter"], source, output)
        try:
            with log.open("x", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=gmr["root"],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=float(plan["execution_contract"]["timeout_seconds_per_asset"]),
                )
            if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
                raise base.ContractError(
                    f"converter failed rc={result.returncode}, output_exists={output.is_file()}"
                )
            structural = base.run_auditor(
                plan, python, auditor, output, log, audit, row["frames"], env
            )
            payload = base.load_gmr_payload(output, row["frames"])
            stance = None
            if plan["batch_kind"] == "m0_lateral_teachers":
                stance = base.compute_m0_stance(
                    plan, row, payload, inspected["canonical_mjcf"]
                )
            result_row = {
                "asset_id": asset_id,
                "status": "complete_exact_gmr_diagnostic",
                "attempt_version": ATTEMPT_VERSION,
                "input": row["input"],
                "frames": row["frames"],
                "output": {
                    "path": str(output),
                    "bytes": output.stat().st_size,
                    "sha256": base.sha256_file(output),
                },
                "run_log": {
                    "path": str(log),
                    "bytes": log.stat().st_size,
                    "sha256": base.sha256_file(log),
                },
                "structural_audit": {
                    "path": str(audit),
                    "bytes": audit.stat().st_size,
                    "sha256": base.sha256_file(audit),
                    "warmup": structural["warmup"],
                },
                "m0_stance": stance,
                "observed_ball_contact": None,
                "strike_effectiveness": None,
                "formal_eligible": False,
                "schema2_authorized": False,
                "training_authorized": False,
                "hardware_authorized": False,
                "command": command,
            }
            base.write_json_exclusive(binding_path, result_row)
            for durable in (output, log, audit, binding_path):
                base.fsync_file(durable)
            rows_out.append(result_row)
        except (OSError, subprocess.SubprocessError, base.ContractError) as exc:
            if not binding_path.exists():
                base.write_json_exclusive(
                    binding_path,
                    {
                        "schema_version": 1,
                        "attempt_version": ATTEMPT_VERSION,
                        "asset_id": asset_id,
                        "status": "failed_preserved_no_completion_manifest",
                        "error": str(exc),
                        "command": command,
                        "formal_eligible": False,
                        "training_authorized": False,
                        "hardware_authorized": False,
                    },
                )
            base.fsync_dir(root / "bindings")
            raise
    for name in ("outputs", "logs", "audits", "bindings"):
        base.fsync_dir(root / name)
    base.fsync_dir(root)

    # Revalidate mutable/private inputs and the v2 Python closure after every
    # converter child exits, before the report-last completion marker exists.
    base.verify_gmr_source(plan)
    base.verify_materialization(plan, repo_root)
    canonical_mjcf = base.verify_tree_contract(plan, repo_root)
    base.verify_a3_orders_and_sites(plan, repo_root, canonical_mjcf)
    verify_python_v2(plan, gmr["root"], repo_root)

    expected_files = {
        f"outputs/{row['asset_id']}{plan['output_contract']['result_suffix']}"
        for row in inspected["rows"]
    }
    expected_files |= {f"logs/{row['asset_id']}.log" for row in inspected["rows"]}
    expected_files |= {f"audits/{row['asset_id']}.json" for row in inspected["rows"]}
    expected_files |= {f"bindings/{row['asset_id']}.json" for row in inspected["rows"]}
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise base.ContractError(
            f"unexpected v2 pre-completion output file set: actual={sorted(actual_files)}, "
            f"expected={sorted(expected_files)}"
        )
    completion = {
        "schema_version": 1,
        "attempt_version": ATTEMPT_VERSION,
        "status": "complete_exact_gmr_diagnostic",
        "batch_kind": plan["batch_kind"],
        "scope": "v2 exact canonical-beta to A3 GMR plus structural audit; M0 includes preregistered foot-stance measurement only",
        "completed_utc": base.utc_now(),
        "plan": {"path": str(plan_path), "sha256": expected_sha},
        "source_materialization": plan["source_materialization"],
        "runtime_contract": plan["_runtime_contract_binding"],
        "batch_serialization_contract": plan["batch_serialization_contract"],
        "ignored_gmr_source": plan["ignored_gmr_source"],
        "execution_contract": plan["execution_contract"],
        "a3_robot_contract": plan["a3_robot_contract"],
        "results": rows_out,
        "s0_semantic_guard": plan.get("s0_semantic_guard"),
        "m0_stance_contract": plan.get("m0_stance_contract"),
        "formal_eligible": False,
        "schema2_authorized": False,
        "training_authorized": False,
        "hardware_authorized": False,
        "next_gate": "separate_schema2_L0_L1_table_net_and_dynamics_preregistration_required",
    }
    completion_path = root / plan["output_contract"]["completion_manifest_filename"]
    if any(path.name == completion_path.name for path in root.iterdir()):
        raise base.ContractError("v2 completion manifest target exists before report-last publish")
    base.write_json_exclusive(completion_path, completion)
    base.fsync_dir(root)
    return completion_path


def consume(plan: dict[str, Any], plan_path: Path, expected_sha: str, repo_root: Path) -> Path:
    with _exclusive_batch_lock(plan):
        return _consume_locked(plan, plan_path, expected_sha, repo_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("command", choices=("static", "inspect", "consume"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        plan_path = args.plan.resolve()
        plan = validate_plan(plan_path, args.expected_plan_sha256, repo_root)
        if args.command == "static":
            print(
                f"PASS static-v2 {plan['batch_kind']} plan_sha256={args.expected_plan_sha256}"
            )
            return 0
        if args.command == "inspect":
            inspected = inspect_plan(plan, repo_root)
            print(
                f"PASS inspect-v2 {plan['batch_kind']} inputs={len(inspected['rows'])} "
                f"output_root_absent={not os.path.lexists(inspected['output_root'])}"
            )
            return 0
        completion = consume(plan, plan_path, args.expected_plan_sha256, repo_root)
        print(f"PASS consume-v2 {plan['batch_kind']} completion={completion}")
        return 0
    except (base.ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL exact-gmr-v2: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
