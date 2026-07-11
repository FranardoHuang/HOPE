#!/usr/bin/env python3
"""Bind and run the Isaac companion for the frozen M3 terminal causal q50 pair.

This wrapper never creates a paper.  ``prepare`` first revalidates a completed MuJoCo pair and
writes a separate, content-addressed Isaac runtime contract that points at the exact same schedule
file.  ``run`` requires that runtime-contract file SHA, launches the two Isaac cells sequentially
in independent process groups, and never sends a signal.  Every accepted result remains
causal/inexact and non-formal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ARM_ORDER = ("M3_old", "M3_S1")
SCORECARD_SCHEMA = "hope.isaac-bank-exam.v1"
HOLD_SEMANTICS = "stand-policy-actions-then-raw-frame0-v1"


class ContractError(RuntimeError):
    """A frozen companion, MuJoCo, schedule, or result binding no longer matches."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def atomic_json_no_clobber(path: Path, value: Any) -> None:
    if path.exists():
        raise ContractError(f"no-clobber: output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ContractError(f"no-clobber: preserved temporary output exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def atomic_json_replace(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ContractError(f"non-finite JSON constant {value!r} in {path}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load strict JSON {path}: {exc}") from exc


def exact_keys(value: Mapping[str, Any], expected: set[str], owner: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{owner} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{owner} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def require_sha(name: str, value: Any, *, length: int = 64) -> str:
    pattern = HEX64 if length == 64 else HEX40
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-{length * 4}: {value!r}")
    return value


def require_absolute(name: str, value: Any) -> Path:
    if not isinstance(value, str) or not os.path.isabs(value) or "\0" in value:
        raise ContractError(f"{name} must be an absolute path: {value!r}")
    return Path(value)


def require_under(name: str, path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"{name} escapes {root}: {path}") from exc


def require_file_sha(name: str, path: Path, expected_sha: str) -> None:
    require_sha(f"{name}.sha256", expected_sha)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ContractError(f"{name} file is missing or changed: {path}")


def require_arm_map(value: Any, owner: str) -> dict[str, Any]:
    """Require the frozen arm key set without assigning semantics to JSON map order."""
    if not isinstance(value, dict) or set(value) != set(ARM_ORDER):
        raise ContractError(f"{owner} arm keys must be exactly {sorted(ARM_ORDER)!r}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    data = load_json(path)
    exact_keys(
        data,
        {
            "schema_version",
            "contract_id",
            "status",
            "auto_start",
            "preregistration_sha256",
            "semantics",
            "checkouts",
            "mujoco_binding",
            "tools",
            "paper",
            "command",
            "arm_order",
        },
        "Isaac companion config",
    )
    if data["schema_version"] != 1:
        raise ContractError("Isaac companion config schema_version must be 1")
    if data["status"] != "offline_runtime_contract_required" or data["auto_start"] is not False:
        raise ContractError("Isaac companion config must remain offline with auto_start=false")
    require_sha("preregistration_sha256", data["preregistration_sha256"])
    if data["arm_order"] != list(ARM_ORDER):
        raise ContractError(f"arm_order must be exactly {list(ARM_ORDER)!r}")

    semantics = data["semantics"]
    exact_keys(
        semantics,
        {"causal", "evaluation_contract_exact", "formal_target", "deploy_gate"},
        "semantics",
    )
    if semantics != {
        "causal": True,
        "evaluation_contract_exact": False,
        "formal_target": False,
        "deploy_gate": False,
    }:
        raise ContractError("Isaac companion is restricted to causal/inexact diagnostics")

    checkouts = data["checkouts"]
    exact_keys(checkouts, {"training", "evaluation"}, "checkouts")
    for name, spec in checkouts.items():
        exact_keys(spec, {"path", "commit"}, f"checkouts.{name}")
        require_absolute(f"checkouts.{name}.path", spec["path"])
        require_sha(f"checkouts.{name}.commit", spec["commit"], length=40)

    mujoco = data["mujoco_binding"]
    exact_keys(
        mujoco,
        {"runner_sha256", "execution_config_sha256", "execution_contract_id"},
        "mujoco_binding",
    )
    require_sha("mujoco_binding.runner_sha256", mujoco["runner_sha256"])
    require_sha("mujoco_binding.execution_config_sha256", mujoco["execution_config_sha256"])
    if not isinstance(mujoco["execution_contract_id"], str) or not mujoco["execution_contract_id"]:
        raise ContractError("mujoco_binding.execution_contract_id must be non-empty")

    tools = data["tools"]
    exact_keys(tools, {"runner_sha256", "evaluation"}, "tools")
    require_sha("tools.runner_sha256", tools["runner_sha256"])
    expected_tool_names = {
        "isaac_evaluator",
        "isaac_adapter",
        "schedule_module",
        "isaac_scorer",
        "ball_physics_yaml",
        "setup_train_env",
    }
    exact_keys(tools["evaluation"], expected_tool_names, "tools.evaluation")
    for name, spec in tools["evaluation"].items():
        exact_keys(spec, {"path", "sha256"}, f"tools.evaluation.{name}")
        relative = Path(spec["path"])
        if not isinstance(spec["path"], str) or relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"tools.evaluation.{name}.path must be safe repo-relative")
        require_sha(f"tools.evaluation.{name}.sha256", spec["sha256"])

    paper = data["paper"]
    expected_paper = {
        "schema_version": 3,
        "schedule_k": 100,
        "attempts_per_side": 50,
        "per_clip_quota": 50,
        "schedule_seed": 0,
        "hold_range": [0, 100],
        "noise_scale": 0.0,
        "same_schedule_file_as_mujoco": True,
        "no_wrap": True,
        "allow_inexact_contract": True,
    }
    if paper != expected_paper:
        raise ContractError(f"paper must be the frozen shared clean q50: {expected_paper}")

    command = data["command"]
    exact_keys(
        command,
        {"isaac_python", "task", "headless", "output_stem", "max_parallel"},
        "command",
    )
    require_absolute("command.isaac_python", command["isaac_python"])
    if command != {
        "isaac_python": command["isaac_python"],
        "task": "HOPEPingPongVirtualBall",
        "headless": True,
        "output_stem": "isaac_clean_k100",
        "max_parallel": 1,
    }:
        raise ContractError("Isaac command semantics changed from the preregistered companion")
    return data


def load_mujoco_helper(path: Path, expected_sha: str) -> ModuleType:
    require_file_sha("MuJoCo paired runner", path, expected_sha)
    spec = importlib.util.spec_from_file_location(
        f"paired_mujoco_q50_{expected_sha[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import MuJoCo paired runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "load_execution_config",
        "validate_preregistration",
        "validate_runtime_contract",
        "validate_runtime_inputs",
        "validate_schedule_document",
        "validate_exam_result",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ContractError(f"MuJoCo paired runner lacks strict helper API: {missing}")
    return module


def validate_static_bindings(
    *,
    config_path: Path,
    expected_config_sha: str,
    prereg_path: Path,
    mujoco_runner_path: Path,
    mujoco_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], ModuleType]:
    require_sha("expected config SHA", expected_config_sha)
    require_file_sha("Isaac companion config", config_path, expected_config_sha)
    config = load_config(config_path)
    require_file_sha("Isaac companion runner", Path(__file__).resolve(), config["tools"]["runner_sha256"])
    require_file_sha("q50 preregistration", prereg_path, config["preregistration_sha256"])
    require_file_sha(
        "MuJoCo execution config",
        mujoco_config_path,
        config["mujoco_binding"]["execution_config_sha256"],
    )
    helper = load_mujoco_helper(
        mujoco_runner_path, config["mujoco_binding"]["runner_sha256"]
    )
    try:
        prereg = helper.load_json(prereg_path)
        mujoco_config = helper.load_execution_config(mujoco_config_path)
        helper.validate_preregistration(prereg, mujoco_config)
    except Exception as exc:
        raise ContractError(f"MuJoCo prereg/config binding failed: {exc}") from exc
    if mujoco_config.get("contract_id") != config["mujoco_binding"]["execution_contract_id"]:
        raise ContractError("MuJoCo execution contract ID changed")
    if mujoco_config.get("checkouts") != config["checkouts"]:
        raise ContractError("MuJoCo and Isaac companion checkout pins differ")
    return config, prereg, mujoco_config, helper


def validate_eval_tools(config: dict[str, Any], eval_root: Path) -> dict[str, Path]:
    paths = {}
    for name, spec in config["tools"]["evaluation"].items():
        path = (eval_root / spec["path"]).resolve()
        require_under(f"tools.evaluation.{name}", path, eval_root)
        require_file_sha(f"tools.evaluation.{name}", path, spec["sha256"])
        paths[name] = path
    return paths


def validate_mujoco_result(
    *,
    result_path: Path,
    expected_result_sha: str,
    runtime_path: Path,
    expected_runtime_sha: str,
    config: dict[str, Any],
    prereg: dict[str, Any],
    mujoco_config: dict[str, Any],
    helper: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require_file_sha("MuJoCo runtime contract", runtime_path, expected_runtime_sha)
    require_file_sha("MuJoCo paired result", result_path, expected_result_sha)
    try:
        runtime = helper.validate_runtime_contract(
            runtime_path, expected_runtime_sha, mujoco_config, prereg
        )
        helper.validate_runtime_inputs(mujoco_config, prereg)
    except Exception as exc:
        raise ContractError(f"MuJoCo runtime preflight no longer passes: {exc}") from exc
    result = load_json(result_path)
    if (
        result.get("schema_version") != 1
        or result.get("pair_id") != config["mujoco_binding"]["execution_contract_id"]
        or result.get("status") != "complete"
        or result.get("causal") is not True
        or result.get("evaluation_contract_exact") is not False
        or result.get("formal_target") is not False
        or result.get("deploy_gate") is not False
        or result.get("runtime_contract")
        != {"path": str(runtime_path), "sha256": expected_runtime_sha}
        or result.get("shared_schedule_sha256")
        != runtime["shared_schedule"]["schedule_sha256"]
        or result.get("question_id_order")
        != runtime["shared_schedule"]["question_id_order"]
    ):
        raise ContractError("MuJoCo paired result is not the completed frozen causal q50")
    result_arms = require_arm_map(result.get("arms"), "MuJoCo paired result")
    validated_arms = {}
    for name in ARM_ORDER:
        arm_result = result_arms[name]
        prereg_arm = prereg["arms"][name]
        report = Path(arm_result.get("report", {}).get("path", ""))
        if (
            arm_result.get("run_name") != prereg_arm["run_name"]
            or arm_result.get("checkpoint_sha256") != prereg_arm["checkpoint_sha256"]
            or arm_result.get("schedule_sha256")
            != runtime["shared_schedule"]["schedule_sha256"]
            or arm_result.get("question_id_order")
            != runtime["shared_schedule"]["question_id_order"]
            or arm_result.get("evaluation_contract_exact") is not False
            or arm_result.get("causal") is not True
            or arm_result.get("formal_target") is not False
        ):
            raise ContractError(f"MuJoCo paired result arm {name} changed semantics/provenance")
        for artifact_name in ("report", "summary", "attempt_ledger"):
            artifact = arm_result.get(artifact_name, {})
            artifact_path = require_absolute(
                f"MuJoCo {name}.{artifact_name}.path", artifact.get("path")
            )
            require_file_sha(
                f"MuJoCo {name}.{artifact_name}", artifact_path, artifact.get("sha256")
            )
        try:
            observed = helper.validate_exam_result(
                report=report,
                arm=prereg_arm,
                prereg=prereg,
                runtime_contract=runtime,
            )
        except Exception as exc:
            raise ContractError(f"MuJoCo raw artifacts for {name} no longer validate: {exc}") from exc
        for key in (
            "checkpoint_sha256",
            "schedule_sha256",
            "question_id_order",
            "mjcf_sha256",
            "execution_contract_sha256",
            "ready_state_sha256",
            "evaluation_contract_exact",
            "causal",
            "formal_target",
        ):
            if arm_result.get(key) != observed.get(key):
                raise ContractError(f"MuJoCo pair ledger disagrees with raw {name}.{key}")
        validated_arms[name] = observed
    return runtime, result, validated_arms


def prepare_runtime_contract(
    *,
    config_path: Path,
    prereg_path: Path,
    mujoco_runner_path: Path,
    mujoco_config_path: Path,
    mujoco_runtime_path: Path,
    expected_mujoco_runtime_sha: str,
    mujoco_result_path: Path,
    expected_mujoco_result_sha: str,
    state_dir: Path,
    output_path: Path,
    config: dict[str, Any],
    prereg: dict[str, Any],
    mujoco_config: dict[str, Any],
    helper: ModuleType,
) -> int:
    if state_dir.exists():
        raise ContractError(f"no-clobber: requested Isaac state directory exists: {state_dir}")
    if not state_dir.is_absolute():
        raise ContractError("Isaac state directory must be absolute")
    runtime, _, validated_mujoco_arms = validate_mujoco_result(
        result_path=mujoco_result_path,
        expected_result_sha=expected_mujoco_result_sha,
        runtime_path=mujoco_runtime_path,
        expected_runtime_sha=expected_mujoco_runtime_sha,
        config=config,
        prereg=prereg,
        mujoco_config=mujoco_config,
        helper=helper,
    )
    eval_root = Path(config["checkouts"]["evaluation"]["path"])
    tools = validate_eval_tools(config, eval_root)
    isaac_python = Path(config["command"]["isaac_python"])
    if not isaac_python.is_file() or not os.access(isaac_python, os.X_OK):
        raise ContractError(f"Isaac Python is missing/not executable: {isaac_python}")
    schedule_meta = runtime["shared_schedule"]
    schedule_path = Path(schedule_meta["path"])
    schedule = helper.validate_schedule_document(
        schedule_path, expected_bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    output_stem = config["command"]["output_stem"]
    prepared = {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "status": "prepared_not_started",
        "auto_start": False,
        "jobs_started": 0,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "causal": True,
        "evaluation_contract_exact": False,
        "formal_target": False,
        "deploy_gate": False,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preregistration": {
            "path": str(prereg_path),
            "sha256": sha256_file(prereg_path),
        },
        "checkouts": config["checkouts"],
        "mujoco": {
            "runner": {
                "path": str(mujoco_runner_path),
                "sha256": sha256_file(mujoco_runner_path),
            },
            "execution_config": {
                "path": str(mujoco_config_path),
                "sha256": sha256_file(mujoco_config_path),
            },
            "runtime_contract": {
                "path": str(mujoco_runtime_path),
                "sha256": expected_mujoco_runtime_sha,
            },
            "paired_result": {
                "path": str(mujoco_result_path),
                "sha256": expected_mujoco_result_sha,
            },
            "arms": {
                name: {
                    "summary": validated_mujoco_arms[name]["summary"],
                    "attempt_ledger": validated_mujoco_arms[name]["attempt_ledger"],
                }
                for name in ARM_ORDER
            },
        },
        "tools": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "evaluation": config["tools"]["evaluation"],
        },
        "exam_bank": prereg["paper"]["exam_bank"],
        "shared_schedule": {
            **schedule_meta,
            "path": str(schedule_path),
            "file_sha256": sha256_file(schedule_path),
            "question_id_order": [item["question_id"] for item in schedule["items"]],
        },
        "state_dir": str(state_dir),
        "output_stem": output_stem,
        "arm_order": list(ARM_ORDER),
        "arms": {
            name: {
                "run_name": prereg["arms"][name]["run_name"],
                "run_dir": str(Path(prereg["arms"][name]["checkpoint_path"]).parent),
                "checkpoint_path": prereg["arms"][name]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
                "training_contract_path": prereg["arms"][name]["training_contract_path"],
                "training_contract_sha256": prereg["arms"][name]["training_contract_sha256"],
                "face_command_pairing": prereg["arms"][name]["face_command_pairing"],
                "output_dir": str(state_dir / name),
                "output_json": str(state_dir / name / f"{output_stem}.json"),
                "output_csv": str(state_dir / name / f"{output_stem}.csv"),
                "job_status": "not_started",
            }
            for name in ARM_ORDER
        },
        "paired_result_path": str(state_dir / "isaac_paired_result.json"),
    }
    atomic_json_no_clobber(output_path, prepared)
    print("[paired-isaac-q50] prepared only; no Isaac process started")
    print(f"[paired-isaac-q50] runtime_contract={output_path} sha256={sha256_file(output_path)}")
    print(
        "[paired-isaac-q50] shared_schedule="
        f"{schedule_path} file_sha256={sha256_file(schedule_path)} "
        f"schedule_sha256={schedule['schedule_sha256']}"
    )
    return 0


def validate_prepared_runtime(
    *,
    path: Path,
    expected_sha: str,
    config: dict[str, Any],
    prereg: dict[str, Any],
    mujoco_config: dict[str, Any],
    helper: ModuleType,
) -> tuple[dict[str, Any], dict[str, Path]]:
    require_file_sha("Isaac companion runtime contract", path, expected_sha)
    runtime = load_json(path)
    exact_keys(
        runtime,
        {
            "schema_version",
            "contract_id",
            "status",
            "auto_start",
            "jobs_started",
            "prepared_utc",
            "causal",
            "evaluation_contract_exact",
            "formal_target",
            "deploy_gate",
            "config",
            "preregistration",
            "checkouts",
            "mujoco",
            "tools",
            "exam_bank",
            "shared_schedule",
            "state_dir",
            "output_stem",
            "arm_order",
            "arms",
            "paired_result_path",
        },
        "Isaac runtime contract",
    )
    if (
        runtime.get("schema_version") != 1
        or runtime.get("contract_id") != config["contract_id"]
        or runtime.get("status") != "prepared_not_started"
        or runtime.get("auto_start") is not False
        or runtime.get("jobs_started") != 0
        or runtime.get("causal") is not True
        or runtime.get("evaluation_contract_exact") is not False
        or runtime.get("formal_target") is not False
        or runtime.get("deploy_gate") is not False
        or runtime.get("checkouts") != config["checkouts"]
        or runtime.get("arm_order") != list(ARM_ORDER)
    ):
        raise ContractError("Isaac runtime contract is not pristine causal/inexact prepared state")
    bindings = (
        ("config", config),
        ("preregistration", prereg),
    )
    for name, _ in bindings:
        artifact = runtime.get(name, {})
        artifact_path = require_absolute(f"runtime.{name}.path", artifact.get("path"))
        require_file_sha(f"runtime.{name}", artifact_path, artifact.get("sha256"))
    if runtime["config"]["sha256"] != sha256_file(Path(runtime["config"]["path"])):
        raise ContractError("runtime config bytes changed")
    if runtime["preregistration"]["sha256"] != config["preregistration_sha256"]:
        raise ContractError("runtime preregistration SHA changed")
    if runtime.get("exam_bank") != prereg["paper"]["exam_bank"]:
        raise ContractError("runtime exam bank changed from preregistration")
    if runtime.get("tools") != {
        "runner_sha256": config["tools"]["runner_sha256"],
        "evaluation": config["tools"]["evaluation"],
    }:
        raise ContractError("runtime tool contract changed")
    mujoco = runtime.get("mujoco", {})
    for name in ("runner", "execution_config", "runtime_contract", "paired_result"):
        artifact = mujoco.get(name, {})
        artifact_path = require_absolute(f"runtime.mujoco.{name}.path", artifact.get("path"))
        require_file_sha(f"runtime.mujoco.{name}", artifact_path, artifact.get("sha256"))
    if mujoco["runner"]["sha256"] != config["mujoco_binding"]["runner_sha256"]:
        raise ContractError("runtime MuJoCo runner SHA differs from companion config")
    if (
        mujoco["execution_config"]["sha256"]
        != config["mujoco_binding"]["execution_config_sha256"]
    ):
        raise ContractError("runtime MuJoCo execution config SHA differs from companion config")
    validated_mujoco_runtime, _, _ = validate_mujoco_result(
        result_path=Path(mujoco["paired_result"]["path"]),
        expected_result_sha=mujoco["paired_result"]["sha256"],
        runtime_path=Path(mujoco["runtime_contract"]["path"]),
        expected_runtime_sha=mujoco["runtime_contract"]["sha256"],
        config=config,
        prereg=prereg,
        mujoco_config=mujoco_config,
        helper=helper,
    )
    paired_mujoco = load_json(Path(mujoco["paired_result"]["path"]))
    expected_mujoco_arms = {
        name: {
            "summary": paired_mujoco["arms"][name]["summary"],
            "attempt_ledger": paired_mujoco["arms"][name]["attempt_ledger"],
        }
        for name in ARM_ORDER
    }
    if mujoco.get("arms") != expected_mujoco_arms:
        raise ContractError("runtime cached MuJoCo arm bindings changed")
    eval_root = Path(config["checkouts"]["evaluation"]["path"])
    tools = validate_eval_tools(config, eval_root)
    schedule_meta = runtime.get("shared_schedule", {})
    if schedule_meta != validated_mujoco_runtime.get("shared_schedule"):
        raise ContractError("Isaac runtime must reuse the exact MuJoCo schedule path/file/SHA")
    schedule_path = require_absolute("runtime.shared_schedule.path", schedule_meta.get("path"))
    require_file_sha(
        "runtime.shared_schedule", schedule_path, schedule_meta.get("file_sha256")
    )
    schedule = helper.validate_schedule_document(
        schedule_path, expected_bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    if (
        schedule_meta.get("schema_version") != 3
        or schedule_meta.get("schedule_k") != 100
        or schedule_meta.get("attempts_per_side") != 50
        or schedule_meta.get("seed") != 0
        or schedule_meta.get("hold_range") != [0, 100]
        or schedule_meta.get("schedule_sha256") != schedule["schedule_sha256"]
        or schedule_meta.get("question_id_order")
        != [item["question_id"] for item in schedule["items"]]
    ):
        raise ContractError("Isaac runtime no longer binds the exact shared MuJoCo K=100 paper")
    state_dir = require_absolute("runtime.state_dir", runtime.get("state_dir"))
    if state_dir.exists():
        raise ContractError(f"no-clobber: Isaac state directory already exists: {state_dir}")
    paired_result = require_absolute(
        "runtime.paired_result_path", runtime.get("paired_result_path")
    )
    require_under("runtime.paired_result_path", paired_result, state_dir)
    if paired_result != state_dir / "isaac_paired_result.json":
        raise ContractError("runtime paired result path changed")
    if runtime.get("output_stem") != config["command"]["output_stem"]:
        raise ContractError("runtime output stem changed")
    runtime_arms = require_arm_map(runtime.get("arms"), "Isaac runtime contract")
    for name in ARM_ORDER:
        arm = runtime_arms[name]
        prereg_arm = prereg["arms"][name]
        expected_output = state_dir / name
        if (
            arm.get("run_name") != prereg_arm["run_name"]
            or arm.get("run_dir") != str(Path(prereg_arm["checkpoint_path"]).parent)
            or arm.get("checkpoint_path") != prereg_arm["checkpoint_path"]
            or arm.get("checkpoint_sha256") != prereg_arm["checkpoint_sha256"]
            or arm.get("training_contract_path") != prereg_arm["training_contract_path"]
            or arm.get("training_contract_sha256") != prereg_arm["training_contract_sha256"]
            or arm.get("face_command_pairing") != prereg_arm["face_command_pairing"]
            or arm.get("output_dir") != str(expected_output)
            or arm.get("output_json")
            != str(expected_output / f"{runtime['output_stem']}.json")
            or arm.get("output_csv")
            != str(expected_output / f"{runtime['output_stem']}.csv")
            or arm.get("job_status") != "not_started"
        ):
            raise ContractError(f"runtime arm {name} changed from its preregistered job")
    return runtime, tools


def build_command(
    *,
    config: dict[str, Any],
    runtime: dict[str, Any],
    tools: dict[str, Path],
    arm_name: str,
    gpu: int,
) -> list[str]:
    if arm_name not in ARM_ORDER:
        raise ContractError(f"unknown arm {arm_name!r}")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ContractError("GPU must be a non-negative integer")
    arm = runtime["arms"][arm_name]
    bank = runtime["exam_bank"]
    schedule = runtime["shared_schedule"]
    return [
        config["command"]["isaac_python"],
        str(tools["isaac_evaluator"]),
        f"task={config['command']['task']}",
        "headless=true",
        f"device=cuda:{gpu}",
        f"+run_dir={arm['run_dir']}",
        f"checkpoint={arm['checkpoint_path']}",
        f"+exam_bank={bank['path']}",
        f"+schedule_json={schedule['path']}",
        "+per_clip_quota=50",
        "+schedule_seed=0",
        "+noise_scale=0.0",
        "+allow_inexact_contract=true",
        f"+output_dir={arm['output_dir']}",
        f"+output_stem={runtime['output_stem']}",
    ]


def setup_environment(setup_script: Path) -> dict[str, str]:
    try:
        raw = subprocess.check_output(
            [
                "bash",
                "-c",
                'source "$1" >/dev/null && env -0',
                "paired-isaac-q50",
                str(setup_script),
            ],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"failed to source setup_train_env.sh: {exc.output!r}") from exc
    env = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        key, separator, value = entry.partition(b"=")
        if not separator:
            raise ContractError("setup environment emitted a malformed entry")
        env[key.decode("utf-8")] = value.decode("utf-8")
    env.update(
        HYDRA_FULL_ERROR="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return env


def _strict_bool(value: Any, expected: bool) -> bool:
    return value is expected


def validate_scorecard(
    *,
    json_path: Path,
    csv_path: Path,
    arm_name: str,
    config: dict[str, Any],
    prereg: dict[str, Any],
    runtime: dict[str, Any],
    tools: dict[str, Path],
) -> dict[str, Any]:
    if arm_name not in ARM_ORDER:
        raise ContractError(f"unknown scorecard arm {arm_name!r}")
    if not json_path.is_file() or not csv_path.is_file():
        raise ContractError(f"Isaac {arm_name} result JSON/CSV is missing")
    document = load_json(json_path)
    schedule_document = load_json(Path(runtime["shared_schedule"]["path"]))
    arm = runtime["arms"][arm_name]
    prereg_arm = prereg["arms"][arm_name]
    bank = runtime["exam_bank"]
    sources = document.get("sources", {})
    exam_bank = document.get("exam_bank", {})
    attempts = document.get("attempts")
    summary = document.get("summary", {})
    nominal_profile = document.get("nominal_eval_profile", {})
    expected_source_shas = {
        "evaluator_sha256": config["tools"]["evaluation"]["isaac_evaluator"]["sha256"],
        "adapter_sha256": config["tools"]["evaluation"]["isaac_adapter"]["sha256"],
        "schedule_module_sha256": config["tools"]["evaluation"]["schedule_module"]["sha256"],
        "isaac_scorer_sha256": config["tools"]["evaluation"]["isaac_scorer"]["sha256"],
        "ball_physics_yaml_sha256": config["tools"]["evaluation"]["ball_physics_yaml"]["sha256"],
    }
    if (
        document.get("schema") != SCORECARD_SCHEMA
        or document.get("status") != "valid"
        or document.get("evaluation_contract_exact") is not False
        or not isinstance(document.get("inexact_reasons"), list)
        or not document["inexact_reasons"]
        or document.get("simulator") != "isaac"
        or document.get("protocol") != "single"
        or document.get("noise_scale") != 0.0
        or document.get("schedule") != schedule_document
        or document.get("schedule_sha256")
        != runtime["shared_schedule"]["schedule_sha256"]
        or document.get("hold_semantics") != HOLD_SEMANTICS
        or exam_bank.get("path") != bank["path"]
        or exam_bank.get("sha256") != bank["sha256"]
        or exam_bank.get("schema_version") != 3
        or exam_bank.get("split") != "exam"
        or not HEX64.fullmatch(str(exam_bank.get("source_family_sha256", "")))
        or document.get("checkpoint")
        != {"path": arm["checkpoint_path"], "sha256": arm["checkpoint_sha256"]}
        or document.get("training_contract_sha256") != arm["training_contract_sha256"]
        or not HEX64.fullmatch(str(document.get("ready_state_sha256", "")))
        or not isinstance(document.get("termination_contract_id"), str)
        or not document["termination_contract_id"]
        or not isinstance(nominal_profile, dict)
        or not HEX64.fullmatch(str(nominal_profile.get("sha256", "")))
        or sources.get("git_head") != config["checkouts"]["evaluation"]["commit"]
        or any(sources.get(key) != value for key, value in expected_source_shas.items())
        or summary.get("n_attempts") != 100
        or summary.get("per_clip", {}).get("forehand", {}).get("n_attempts") != 50
        or summary.get("per_clip", {}).get("backhand", {}).get("n_attempts") != 50
        or not isinstance(attempts, list)
        or len(attempts) != 100
    ):
        raise ContractError(f"Isaac {arm_name} scorecard header/summary contract mismatch")
    expected_items = schedule_document["items"]
    ready_sha = document["ready_state_sha256"]
    for index, (row, item) in enumerate(zip(attempts, expected_items)):
        projection = {
            key: row.get(key)
            for key in (
                "schedule_index",
                "clip",
                "bank_row",
                "question_id",
                "repeat",
                "hold_steps",
                "attempt_seed",
            )
        }
        expected = {key: item[key] for key in projection}
        expected_side = "forehand" if item["clip"] == 0 else "backhand"
        if (
            projection != expected
            or row.get("env_id") != index
            or row.get("side") != expected_side
            or row.get("ready_state_sha256") != ready_sha
            or row.get("start_step") != 0
            or isinstance(row.get("end_step"), bool)
            or not isinstance(row.get("end_step"), int)
            or row["end_step"] <= 0
            or row.get("finalize_reason") not in (
                "clip_complete",
                "physical_fall",
                "guard_reset",
                "episode_timeout",
            )
            or not _strict_bool(row.get("finalized"), True)
            or not _strict_bool(row.get("censored"), False)
            or not isinstance(row.get("physical_fall"), bool)
            or not isinstance(row.get("guard_reset"), bool)
            or not isinstance(row.get("reached_exact"), bool)
            or not isinstance(row.get("hit"), bool)
            or not isinstance(row.get("returned"), bool)
            or not isinstance(row.get("net_clear"), bool)
        ):
            raise ContractError(f"Isaac {arm_name} attempt {index} is censored/reordered/malformed")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 100:
        raise ContractError(f"Isaac {arm_name} CSV does not have 100 attempts")
    try:
        for index, (row, item) in enumerate(zip(csv_rows, expected_items)):
            if (
                int(row["schedule_index"]) != item["schedule_index"]
                or int(row["env_id"]) != index
                or int(row["clip"]) != item["clip"]
                or int(row["bank_row"]) != item["bank_row"]
                or row["question_id"] != item["question_id"]
                or int(row["repeat"]) != item["repeat"]
                or int(row["hold_steps"]) != item["hold_steps"]
                or int(row["attempt_seed"]) != item["attempt_seed"]
                or row["ready_state_sha256"] != ready_sha
                or row["finalized"] != "True"
                or row["censored"] != "False"
            ):
                raise ContractError(f"Isaac {arm_name} CSV row {index} differs from JSON/paper")
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"Isaac {arm_name} CSV schema is invalid: {exc}") from exc
    if prereg_arm["checkpoint_sha256"] != arm["checkpoint_sha256"]:
        raise ContractError(f"Isaac {arm_name} checkpoint lost preregistration binding")
    return {
        "run_name": arm["run_name"],
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "training_contract_sha256": arm["training_contract_sha256"],
        "scorecard": {"path": str(json_path), "sha256": sha256_file(json_path)},
        "attempt_ledger": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "schedule_sha256": runtime["shared_schedule"]["schedule_sha256"],
        "question_id_order": runtime["shared_schedule"]["question_id_order"],
        "ready_state_sha256": ready_sha,
        "nominal_eval_profile_sha256": nominal_profile["sha256"],
        "evaluation_contract_exact": False,
        "causal": True,
        "formal_target": False,
        "summary": summary,
    }


def run_pair(
    *,
    runtime_path: Path,
    expected_runtime_sha: str,
    gpus: Sequence[int],
    config: dict[str, Any],
    prereg: dict[str, Any],
    mujoco_config: dict[str, Any],
    helper: ModuleType,
) -> int:
    runtime, tools = validate_prepared_runtime(
        path=runtime_path,
        expected_sha=expected_runtime_sha,
        config=config,
        prereg=prereg,
        mujoco_config=mujoco_config,
        helper=helper,
    )
    if len(gpus) != 2 or any(isinstance(value, bool) or value < 0 for value in gpus):
        raise ContractError("run requires two non-negative GPUs in M3_old, M3_S1 order")
    state_dir = Path(runtime["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=False)
    env = setup_environment(tools["setup_train_env"])
    results = {}
    for index, name in enumerate(ARM_ORDER):
        try:
            helper.validate_runtime_inputs(mujoco_config, prereg)
        except Exception as exc:
            raise ContractError(f"checkout/checkpoint preflight changed before {name}: {exc}") from exc
        validate_eval_tools(config, Path(config["checkouts"]["evaluation"]["path"]))
        require_file_sha(
            "shared schedule",
            Path(runtime["shared_schedule"]["path"]),
            runtime["shared_schedule"]["file_sha256"],
        )
        state_path = state_dir / f"{name}.state.json"
        log_path = state_dir / f"{name}.runner.log"
        output_json = Path(runtime["arms"][name]["output_json"])
        output_csv = Path(runtime["arms"][name]["output_csv"])
        if any(path.exists() for path in (state_path, log_path, output_json, output_csv)):
            raise ContractError(f"no-clobber: preserved Isaac artifacts exist for {name}")
        command = build_command(
            config=config,
            runtime=runtime,
            tools=tools,
            arm_name=name,
            gpu=int(gpus[index]),
        )
        with log_path.open("xb", buffering=0) as log:
            proc = subprocess.Popen(
                command,
                cwd=tools["isaac_evaluator"].parent.parent,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state = {
                "schema_version": 1,
                "arm": name,
                "status": "running",
                "pid": proc.pid,
                "pgid": proc.pid,
                "command": command,
                "runtime_contract_sha256": expected_runtime_sha,
                "schedule_file_sha256": runtime["shared_schedule"]["file_sha256"],
                "schedule_sha256": runtime["shared_schedule"]["schedule_sha256"],
                "checkpoint_sha256": runtime["arms"][name]["checkpoint_sha256"],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_json_replace(state_path, state)
            rc = proc.wait()
        state.update(
            status="complete" if rc == 0 else "failed",
            returncode=rc,
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            log_sha256=sha256_file(log_path),
        )
        atomic_json_replace(state_path, state)
        if rc != 0:
            raise ContractError(f"Isaac {name} failed rc={rc}; preserved {log_path}")
        try:
            helper.validate_runtime_inputs(mujoco_config, prereg)
        except Exception as exc:
            raise ContractError(f"checkout/checkpoint changed during Isaac {name}: {exc}") from exc
        validate_eval_tools(config, Path(config["checkouts"]["evaluation"]["path"]))
        require_file_sha(
            "shared schedule",
            Path(runtime["shared_schedule"]["path"]),
            runtime["shared_schedule"]["file_sha256"],
        )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        expected_json_line = f"[isaac-bank-exam] JSON {output_json}"
        expected_csv_line = f"[isaac-bank-exam] CSV  {output_csv}"
        if text.count(expected_json_line) != 1 or text.count(expected_csv_line) != 1:
            raise ContractError(f"Isaac {name} success handshake is missing/ambiguous")
        results[name] = validate_scorecard(
            json_path=output_json,
            csv_path=output_csv,
            arm_name=name,
            config=config,
            prereg=prereg,
            runtime=runtime,
            tools=tools,
        )

    old, s1 = results["M3_old"], results["M3_S1"]
    for key in (
        "schedule_sha256",
        "question_id_order",
        "ready_state_sha256",
        "nominal_eval_profile_sha256",
    ):
        if old[key] != s1[key]:
            raise ContractError(f"paired Isaac q50 cells disagree on {key}")
    pair_result_path = Path(runtime["paired_result_path"])
    pair_result = {
        "schema_version": 1,
        "pair_id": config["contract_id"],
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_contract": {"path": str(runtime_path), "sha256": expected_runtime_sha},
        "mujoco_paired_result": runtime["mujoco"]["paired_result"],
        "causal": True,
        "evaluation_contract_exact": False,
        "formal_target": False,
        "deploy_gate": False,
        "shared_schedule_file_sha256": runtime["shared_schedule"]["file_sha256"],
        "shared_schedule_sha256": runtime["shared_schedule"]["schedule_sha256"],
        "question_id_order": runtime["shared_schedule"]["question_id_order"],
        "arms": results,
    }
    atomic_json_no_clobber(pair_result_path, pair_result)
    print(f"[paired-isaac-q50] complete causal/inexact pair: {pair_result_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--mujoco-runner", required=True, type=Path)
    parser.add_argument("--mujoco-execution-config", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract-check")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--mujoco-runtime-contract", required=True, type=Path)
    prepare_parser.add_argument("--expected-mujoco-runtime-sha256", required=True)
    prepare_parser.add_argument("--mujoco-paired-result", required=True, type=Path)
    prepare_parser.add_argument("--expected-mujoco-result-sha256", required=True)
    prepare_parser.add_argument("--state-dir", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    runtime_parser = sub.add_parser("runtime-check")
    runtime_parser.add_argument("--runtime-contract", required=True, type=Path)
    runtime_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--runtime-contract", required=True, type=Path)
    run_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser.add_argument("--gpus", nargs=2, required=True, type=int, metavar=("OLD", "S1"))
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    prereg_path = args.preregistration.resolve()
    mujoco_runner_path = args.mujoco_runner.resolve()
    mujoco_config_path = args.mujoco_execution_config.resolve()
    config, prereg, mujoco_config, helper = validate_static_bindings(
        config_path=config_path,
        expected_config_sha=args.expected_config_sha256,
        prereg_path=prereg_path,
        mujoco_runner_path=mujoco_runner_path,
        mujoco_config_path=mujoco_config_path,
    )
    if args.command == "contract-check":
        print("[paired-isaac-q50] offline contract check PASS; no process started")
        return 0
    if args.command == "prepare":
        return prepare_runtime_contract(
            config_path=config_path,
            prereg_path=prereg_path,
            mujoco_runner_path=mujoco_runner_path,
            mujoco_config_path=mujoco_config_path,
            mujoco_runtime_path=args.mujoco_runtime_contract.resolve(),
            expected_mujoco_runtime_sha=args.expected_mujoco_runtime_sha256,
            mujoco_result_path=args.mujoco_paired_result.resolve(),
            expected_mujoco_result_sha=args.expected_mujoco_result_sha256,
            state_dir=args.state_dir.resolve(),
            output_path=args.output.resolve(),
            config=config,
            prereg=prereg,
            mujoco_config=mujoco_config,
            helper=helper,
        )
    if args.command == "runtime-check":
        validate_prepared_runtime(
            path=args.runtime_contract.resolve(),
            expected_sha=args.expected_runtime_contract_sha256,
            config=config,
            prereg=prereg,
            mujoco_config=mujoco_config,
            helper=helper,
        )
        print("[paired-isaac-q50] runtime check PASS; no process started")
        return 0
    return run_pair(
        runtime_path=args.runtime_contract.resolve(),
        expected_runtime_sha=args.expected_runtime_contract_sha256,
        gpus=args.gpus,
        config=config,
        prereg=prereg,
        mujoco_config=mujoco_config,
        helper=helper,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[paired-isaac-q50][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
