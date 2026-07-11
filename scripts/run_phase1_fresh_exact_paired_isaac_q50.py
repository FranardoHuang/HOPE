#!/usr/bin/env python3
"""Run the fresh-SZ model_2000/model_4000 Isaac companion on the MuJoCo q50 paper.

The companion never materializes a paper. ``prepare`` revalidates the completed fresh/exact
MuJoCo pair and writes a content-addressed Isaac runtime contract that points at the very same
schema-v3 K=100 schedule. ``run`` launches the two Isaac cells sequentially in independent
process groups. It never passes the inexact-contract escape, never stops/promotes the live arm,
and never authorizes deployment or a real robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


SCRIPT_DIR = Path(__file__).resolve().parent
FRESH_MUJOCO_RUNNER = SCRIPT_DIR / "run_phase1_fresh_exact_paired_bank_q50.py"
FRESH_MUJOCO_RUNNER_SHA256 = (
    "3528250777a170791f39d8dd17716c2a7f8ca91416a3ffa8433ec5eb691ed9e0"
)
SHARED_MUJOCO_RUNNER = SCRIPT_DIR / "run_phase1_paired_bank_q50.py"
SHARED_MUJOCO_RUNNER_SHA256 = (
    "095e476fd36fb68d500cb39ea7f71f6fee9b729209187d51599582c72c22198b"
)
ISAAC_UTILITY_RUNNER = SCRIPT_DIR / "run_phase1_paired_isaac_q50.py"
ISAAC_UTILITY_RUNNER_SHA256 = (
    "f5d0dce4a5f650981838779dfa472c881d63c28341437167e0eca9217f57e04e"
)


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_module(name: str, path: Path, expected_sha256: str) -> ModuleType:
    if not path.is_file() or _bootstrap_sha256(path) != expected_sha256:
        raise RuntimeError(f"refusing changed/missing bound dependency: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import bound dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if not SHARED_MUJOCO_RUNNER.is_file() or _bootstrap_sha256(
    SHARED_MUJOCO_RUNNER
) != SHARED_MUJOCO_RUNNER_SHA256:
    raise RuntimeError("fresh MuJoCo helper's content-addressed sibling is missing/changed")

fresh = _load_bound_module(
    "phase1_fresh_exact_mujoco_bound", FRESH_MUJOCO_RUNNER, FRESH_MUJOCO_RUNNER_SHA256
)
isaac = _load_bound_module(
    "phase1_causal_isaac_utilities_bound", ISAAC_UTILITY_RUNNER, ISAAC_UTILITY_RUNNER_SHA256
)

ContractError = fresh.ContractError
sha256_file = fresh.sha256_file
canonical_bytes = fresh.canonical_bytes
load_json = fresh.load_json
exact_keys = fresh.exact_keys
require_sha = fresh.require_sha
require_absolute = fresh.require_absolute
require_under = fresh.require_under

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARM_ORDER = ("model_2000", "model_4000")
ITERATIONS = {"model_2000": 2000, "model_4000": 4000}
SCORECARD_SCHEMA = "hope.isaac-bank-exam.v1"
HOLD_SEMANTICS = "stand-policy-actions-then-raw-frame0-v1"
EXPECTED_SEMANTICS = {
    "fresh_lineage": True,
    "evaluation_contract_exact": True,
    "formal_target": True,
    "checkpoint_selection_screen": True,
    "whole_arm_stop_allowed": False,
    "whole_arm_promote_allowed": False,
    "deploy_gate": False,
    "real_robot_authorized": False,
}
EXPECTED_SELECTION_POLICY = {
    "scope": "choose_only_between_model_2000_and_model_4000",
    "primary": "higher_aggregate_all_attempt_return_count_out_of_100",
    "tie_break_1": "higher_min_side_all_attempt_return_count_out_of_50",
    "tie_break_2": "fewer_physical_falls_out_of_100",
    "tie_break_3": "earlier_model_2000",
    "may_select_checkpoint": True,
    "may_stop_whole_arm": False,
    "may_promote_whole_arm": False,
}


def atomic_json_no_clobber(path: Path, value: Any) -> None:
    if path.exists():
        raise ContractError(f"no-clobber: output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ContractError(f"no-clobber: temporary output exists: {temporary}")
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


def require_file_sha(name: str, path: Path, expected_sha256: str) -> None:
    require_sha(f"{name}.sha256", expected_sha256)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ContractError(f"{name} file is missing or changed: {path}")


def require_arm_map(value: Any, owner: str) -> dict[str, Any]:
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
            "semantics",
            "checkouts",
            "mujoco_binding",
            "tools",
            "paper",
            "command",
            "runtime",
            "arm_order",
            "selection_policy",
        },
        "fresh exact Isaac companion config",
    )
    if data["schema_version"] != 1:
        raise ContractError("companion config schema_version must be 1")
    if data["status"] != "offline_runtime_contract_required" or data["auto_start"] is not False:
        raise ContractError("companion must remain offline with auto_start=false")
    if data["semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("companion semantics changed from the fresh exact selection screen")
    if data["arm_order"] != list(ARM_ORDER):
        raise ContractError(f"arm_order must be exactly {list(ARM_ORDER)!r}")
    if data["selection_policy"] != EXPECTED_SELECTION_POLICY:
        raise ContractError("selection policy changed")

    checkouts = data["checkouts"]
    exact_keys(checkouts, {"training", "evaluation"}, "checkouts")
    for name, spec in checkouts.items():
        exact_keys(spec, {"path", "commit"}, f"checkouts.{name}")
        require_absolute(f"checkouts.{name}.path", spec["path"])
        require_sha(f"checkouts.{name}.commit", spec["commit"], length=40)

    binding = data["mujoco_binding"]
    exact_keys(
        binding,
        {
            "execution_contract_id",
            "execution_config",
            "preregistration",
            "runtime_contract",
            "paired_result",
        },
        "mujoco_binding",
    )
    if not isinstance(binding["execution_contract_id"], str) or not binding[
        "execution_contract_id"
    ]:
        raise ContractError("MuJoCo execution contract ID must be non-empty")
    for name in ("execution_config", "preregistration", "runtime_contract", "paired_result"):
        artifact = binding[name]
        exact_keys(artifact, {"path", "sha256"}, f"mujoco_binding.{name}")
        require_absolute(f"mujoco_binding.{name}.path", artifact["path"])
        require_sha(f"mujoco_binding.{name}.sha256", artifact["sha256"])

    tools = data["tools"]
    exact_keys(
        tools,
        {"runner_sha256", "fresh_mujoco_runner", "shared_mujoco_runner", "isaac_utility_runner", "evaluation"},
        "tools",
    )
    require_sha("tools.runner_sha256", tools["runner_sha256"])
    expected_dependencies = {
        "fresh_mujoco_runner": (
            "run_phase1_fresh_exact_paired_bank_q50.py",
            FRESH_MUJOCO_RUNNER_SHA256,
        ),
        "shared_mujoco_runner": (
            "run_phase1_paired_bank_q50.py",
            SHARED_MUJOCO_RUNNER_SHA256,
        ),
        "isaac_utility_runner": (
            "run_phase1_paired_isaac_q50.py",
            ISAAC_UTILITY_RUNNER_SHA256,
        ),
    }
    for name, (expected_path, expected_sha) in expected_dependencies.items():
        spec = tools[name]
        exact_keys(spec, {"path", "sha256"}, f"tools.{name}")
        if spec != {"path": expected_path, "sha256": expected_sha}:
            raise ContractError(f"bound dependency changed: {name}")
    expected_eval_tools = {
        "isaac_evaluator",
        "isaac_adapter",
        "schedule_module",
        "isaac_scorer",
        "ball_physics_yaml",
        "setup_train_env",
    }
    exact_keys(tools["evaluation"], expected_eval_tools, "tools.evaluation")
    for name, spec in tools["evaluation"].items():
        exact_keys(spec, {"path", "sha256"}, f"tools.evaluation.{name}")
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"tools.evaluation.{name}.path must be safe repo-relative")
        require_sha(f"tools.evaluation.{name}.sha256", spec["sha256"])

    expected_paper = {
        "schema_version": 3,
        "schedule_k": 100,
        "attempts_per_side": 50,
        "per_clip_quota": 50,
        "schedule_seed": 0,
        "hold_range": [0, 100],
        "noise_scale": 0.0,
        "same_schedule_file_as_mujoco": True,
        "schedule_file_sha256": "66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3",
        "schedule_semantic_sha256": "7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e",
        "one_question_reset": True,
        "no_wrap": True,
        "allow_inexact_contract": False,
    }
    if data["paper"] != expected_paper:
        raise ContractError("paper changed from the exact shared MuJoCo K=100 schedule")
    command = data["command"]
    exact_keys(command, {"isaac_python", "task", "headless", "output_stem", "max_parallel"}, "command")
    require_absolute("command.isaac_python", command["isaac_python"])
    if command != {
        "isaac_python": command["isaac_python"],
        "task": "HOPEPingPongVirtualBall",
        "headless": True,
        "output_stem": "isaac_clean_k100",
        "max_parallel": 1,
    }:
        raise ContractError("Isaac command semantics changed")
    runtime = data["runtime"]
    exact_keys(runtime, {"state_dir", "runtime_contract_filename", "paired_result_filename"}, "runtime")
    require_absolute("runtime.state_dir", runtime["state_dir"])
    if runtime["runtime_contract_filename"] != "runtime_contract.prepared.json":
        raise ContractError("runtime-contract filename changed")
    if runtime["paired_result_filename"] != "isaac_paired_result.json":
        raise ContractError("paired-result filename changed")
    return data


def validate_eval_tools(config: dict[str, Any], eval_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["tools"]["evaluation"].items():
        path = (eval_root / spec["path"]).resolve()
        require_under(f"tools.evaluation.{name}", path, eval_root)
        require_file_sha(f"tools.evaluation.{name}", path, spec["sha256"])
        paths[name] = path
    return paths


def validate_mujoco_binding(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = config["mujoco_binding"]
    for name in ("execution_config", "preregistration", "runtime_contract", "paired_result"):
        artifact = binding[name]
        require_file_sha(name, Path(artifact["path"]), artifact["sha256"])
    require_file_sha(
        "fresh MuJoCo runner", FRESH_MUJOCO_RUNNER, config["tools"]["fresh_mujoco_runner"]["sha256"]
    )
    require_file_sha(
        "shared MuJoCo runner", SHARED_MUJOCO_RUNNER, config["tools"]["shared_mujoco_runner"]["sha256"]
    )
    require_file_sha(
        "Isaac utility runner", ISAAC_UTILITY_RUNNER, config["tools"]["isaac_utility_runner"]["sha256"]
    )
    execution = fresh.load_execution_config(Path(binding["execution_config"]["path"]))
    prereg = load_json(Path(binding["preregistration"]["path"]))
    fresh.validate_preregistration(prereg, execution)
    if execution["contract_id"] != binding["execution_contract_id"]:
        raise ContractError("MuJoCo execution contract ID changed")
    if execution["checkouts"] != config["checkouts"]:
        raise ContractError("MuJoCo and Isaac checkout pins differ")
    runtime = fresh.validate_runtime_contract(
        Path(binding["runtime_contract"]["path"]),
        binding["runtime_contract"]["sha256"],
        execution,
        prereg,
    )
    fresh.validate_runtime_inputs(execution, prereg)
    result = load_json(Path(binding["paired_result"]["path"]))
    if (
        result.get("schema_version") != 1
        or result.get("pair_id") != binding["execution_contract_id"]
        or result.get("status") != "complete"
        or result.get("runtime_contract") != binding["runtime_contract"]
        or any(result.get(key) != value for key, value in fresh.EXPECTED_SEMANTICS.items())
        or result.get("shared_schedule_sha256") != runtime["shared_schedule"]["schedule_sha256"]
        or result.get("question_id_order") != runtime["shared_schedule"]["question_id_order"]
        or result.get("selection_policy") != fresh.EXPECTED_SELECTION_POLICY
        or result.get("selected_checkpoint", {}).get("arm") != "model_2000"
    ):
        raise ContractError("completed MuJoCo pair lost its fresh/exact selection semantics")
    result_arms = require_arm_map(result.get("arms"), "MuJoCo pair")
    for name in ARM_ORDER:
        report = require_absolute(
            f"MuJoCo {name}.report.path", result_arms[name].get("report", {}).get("path")
        )
        for artifact_name in ("report", "summary", "attempt_ledger"):
            artifact = result_arms[name].get(artifact_name, {})
            artifact_path = require_absolute(
                f"MuJoCo {name}.{artifact_name}.path", artifact.get("path")
            )
            require_file_sha(
                f"MuJoCo {name}.{artifact_name}", artifact_path, artifact.get("sha256")
            )
        observed = fresh.validate_exam_result(
            report=report,
            arm_name=name,
            arm=prereg["arms"][name],
            prereg=prereg,
            runtime_contract=runtime,
        )
        if result_arms[name] != observed:
            raise ContractError(f"MuJoCo pair ledger disagrees with raw exact artifacts: {name}")
    schedule_path = Path(runtime["shared_schedule"]["path"])
    if (
        sha256_file(schedule_path) != config["paper"]["schedule_file_sha256"]
        or runtime["shared_schedule"]["schedule_sha256"]
        != config["paper"]["schedule_semantic_sha256"]
    ):
        raise ContractError("companion does not bind the completed MuJoCo schedule bytes/semantics")
    return execution, prereg, runtime, result, validate_eval_tools(
        config, Path(config["checkouts"]["evaluation"]["path"])
    )


def prepare_runtime_contract(
    *, config_path: Path, output_path: Path, config: dict[str, Any]
) -> int:
    _, prereg, mujoco_runtime, mujoco_result, tools = validate_mujoco_binding(config)
    state_dir = Path(config["runtime"]["state_dir"])
    if state_dir.exists():
        raise ContractError(f"no-clobber: requested Isaac state directory exists: {state_dir}")
    isaac_python = Path(config["command"]["isaac_python"])
    if not isaac_python.is_file() or not os.access(isaac_python, os.X_OK):
        raise ContractError(f"Isaac Python is missing/not executable: {isaac_python}")
    schedule = mujoco_runtime["shared_schedule"]
    output_stem = config["command"]["output_stem"]
    prepared = {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "status": "prepared_not_started",
        "auto_start": False,
        "jobs_started": 0,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **EXPECTED_SEMANTICS,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "checkouts": config["checkouts"],
        "mujoco_binding": config["mujoco_binding"],
        "mujoco_selection": mujoco_result["selection"],
        "tools": config["tools"],
        "exam_bank": prereg["paper"]["exam_bank"],
        "shared_schedule": schedule,
        "state_dir": str(state_dir),
        "output_stem": output_stem,
        "arm_order": list(ARM_ORDER),
        "arms": {
            name: {
                "run_name": prereg["arms"][name]["run_name"],
                "checkpoint_iteration": ITERATIONS[name],
                "run_dir": str(Path(prereg["arms"][name]["checkpoint_path"]).parent),
                "checkpoint_path": prereg["arms"][name]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
                "training_contract_path": prereg["arms"][name]["training_contract_path"],
                "training_contract_sha256": prereg["arms"][name]["training_contract_sha256"],
                "output_dir": str(state_dir / name),
                "output_json": str(state_dir / name / f"{output_stem}.json"),
                "output_csv": str(state_dir / name / f"{output_stem}.csv"),
                "job_status": "not_started",
            }
            for name in ARM_ORDER
        },
        "selection_policy": EXPECTED_SELECTION_POLICY,
        "paired_result_path": str(state_dir / config["runtime"]["paired_result_filename"]),
    }
    del tools  # tools were validated above; bytes are bound in the prepared object.
    atomic_json_no_clobber(output_path, prepared)
    print("[fresh-exact-paired-isaac-q50] prepared only; no Isaac process started")
    print(
        f"[fresh-exact-paired-isaac-q50] runtime_contract={output_path} "
        f"sha256={sha256_file(output_path)}"
    )
    print(
        "[fresh-exact-paired-isaac-q50] shared_schedule="
        f"{schedule['path']} file_sha256={schedule['file_sha256']} "
        f"semantic_sha256={schedule['schedule_sha256']}"
    )
    return 0


def validate_prepared_runtime(
    path: Path, expected_sha256: str, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    require_file_sha("Isaac runtime contract", path, expected_sha256)
    runtime = load_json(path)
    _, prereg, mujoco_runtime, mujoco_result, tools = validate_mujoco_binding(config)
    if (
        runtime.get("schema_version") != 1
        or runtime.get("contract_id") != config["contract_id"]
        or runtime.get("status") != "prepared_not_started"
        or runtime.get("auto_start") is not False
        or runtime.get("jobs_started") != 0
        or any(runtime.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
        or runtime.get("checkouts") != config["checkouts"]
        or runtime.get("mujoco_binding") != config["mujoco_binding"]
        or runtime.get("mujoco_selection") != mujoco_result["selection"]
        or runtime.get("tools") != config["tools"]
        or runtime.get("exam_bank") != prereg["paper"]["exam_bank"]
        or runtime.get("shared_schedule") != mujoco_runtime["shared_schedule"]
        or runtime.get("arm_order") != list(ARM_ORDER)
        or runtime.get("selection_policy") != EXPECTED_SELECTION_POLICY
    ):
        raise ContractError("Isaac runtime contract is not pristine fresh/exact prepared state")
    config_meta = runtime.get("config", {})
    config_artifact = require_absolute("runtime.config.path", config_meta.get("path"))
    require_file_sha("runtime config", config_artifact, config_meta.get("sha256"))
    state_dir = require_absolute("runtime.state_dir", runtime.get("state_dir"))
    if state_dir.exists():
        raise ContractError(f"no-clobber: Isaac state directory already exists: {state_dir}")
    schedule_path = require_absolute(
        "runtime.shared_schedule.path", runtime["shared_schedule"].get("path")
    )
    require_file_sha(
        "runtime shared schedule", schedule_path, config["paper"]["schedule_file_sha256"]
    )
    paired_result = require_absolute(
        "runtime.paired_result_path", runtime.get("paired_result_path")
    )
    if paired_result != state_dir / "isaac_paired_result.json":
        raise ContractError("runtime paired result path changed")
    arms = require_arm_map(runtime.get("arms"), "Isaac runtime")
    for name in ARM_ORDER:
        source = prereg["arms"][name]
        arm = arms[name]
        expected_output = state_dir / name
        if (
            arm.get("run_name") != source["run_name"]
            or arm.get("checkpoint_iteration") != ITERATIONS[name]
            or arm.get("run_dir") != str(Path(source["checkpoint_path"]).parent)
            or arm.get("checkpoint_path") != source["checkpoint_path"]
            or arm.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or arm.get("training_contract_path") != source["training_contract_path"]
            or arm.get("training_contract_sha256") != source["training_contract_sha256"]
            or arm.get("output_dir") != str(expected_output)
            or arm.get("output_json")
            != str(expected_output / f"{runtime['output_stem']}.json")
            or arm.get("output_csv") != str(expected_output / f"{runtime['output_stem']}.csv")
            or arm.get("job_status") != "not_started"
        ):
            raise ContractError(f"runtime arm {name} changed from fresh preregistration")
    return runtime, prereg, tools


def build_command(
    *, config: dict[str, Any], runtime: dict[str, Any], tools: dict[str, Path], arm_name: str, gpu: int
) -> list[str]:
    if arm_name not in ARM_ORDER:
        raise ContractError(f"unknown arm {arm_name!r}")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ContractError("GPU must be a non-negative integer")
    arm = runtime["arms"][arm_name]
    bank = runtime["exam_bank"]
    schedule = runtime["shared_schedule"]
    command = [
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
        f"+output_dir={arm['output_dir']}",
        f"+output_stem={runtime['output_stem']}",
    ]
    if any("allow_inexact_contract" in value for value in command):
        raise ContractError("fresh exact Isaac command contains the inexact escape")
    return command


def summarize_attempts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def group(group_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = lambda key: sum(row.get(key) is True for row in group_rows)
        n = len(group_rows)
        no_fall = n - count("physical_fall")
        return {
            "n_attempts": n,
            "n_no_fall": no_fall,
            "n_reached_exact": count("reached_exact"),
            "n_hit": count("hit"),
            "n_returned": count("returned"),
            "n_guard_reset": count("guard_reset"),
            "no_fall_rate": (no_fall / n) if n else None,
            "exact_reach_rate": (count("reached_exact") / n) if n else None,
            "hit_rate": (count("hit") / n) if n else None,
            "return_rate": (count("returned") / n) if n else None,
            "guard_reset_rate": (count("guard_reset") / n) if n else None,
            "finalize_reason_counts": dict(Counter(str(row.get("finalize_reason")) for row in group_rows)),
        }

    result = group(rows)
    result["per_clip"] = {
        "forehand": group([row for row in rows if row.get("clip") == 0]),
        "backhand": group([row for row in rows if row.get("clip") == 1]),
    }
    return result


def validate_scorecard(
    *, json_path: Path, csv_path: Path, arm_name: str, config: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    if arm_name not in ARM_ORDER:
        raise ContractError(f"unknown scorecard arm {arm_name!r}")
    if not json_path.is_file() or not csv_path.is_file():
        raise ContractError(f"Isaac {arm_name} JSON/CSV is missing")
    document = load_json(json_path)
    schedule_document = load_json(Path(runtime["shared_schedule"]["path"]))
    arm = runtime["arms"][arm_name]
    bank = runtime["exam_bank"]
    sources = document.get("sources", {})
    exam_bank = document.get("exam_bank", {})
    attempts = document.get("attempts")
    summary = document.get("summary", {})
    profile = document.get("nominal_eval_profile", {})
    recomputed = summarize_attempts(attempts) if isinstance(attempts, list) and len(attempts) == 100 else None
    expected_sources = {
        "evaluator_sha256": config["tools"]["evaluation"]["isaac_evaluator"]["sha256"],
        "adapter_sha256": config["tools"]["evaluation"]["isaac_adapter"]["sha256"],
        "schedule_module_sha256": config["tools"]["evaluation"]["schedule_module"]["sha256"],
        "isaac_scorer_sha256": config["tools"]["evaluation"]["isaac_scorer"]["sha256"],
        "ball_physics_yaml_sha256": config["tools"]["evaluation"]["ball_physics_yaml"]["sha256"],
    }
    if (
        document.get("schema") != SCORECARD_SCHEMA
        or document.get("status") != "valid"
        or document.get("evaluation_contract_exact") is not True
        or document.get("inexact_reasons") != []
        or document.get("simulator") != "isaac"
        or document.get("protocol") != "single"
        or document.get("noise_scale") != 0.0
        or document.get("schedule") != schedule_document
        or document.get("schedule_sha256") != runtime["shared_schedule"]["schedule_sha256"]
        or document.get("hold_semantics") != HOLD_SEMANTICS
        or exam_bank.get("path") != bank["path"]
        or exam_bank.get("sha256") != bank["sha256"]
        or exam_bank.get("source_family_sha256") != bank["source_family_sha256"]
        or exam_bank.get("schema_version") != 3
        or exam_bank.get("split") != "exam"
        or document.get("checkpoint") != {"path": arm["checkpoint_path"], "sha256": arm["checkpoint_sha256"]}
        or document.get("training_contract_sha256") != arm["training_contract_sha256"]
        or not HEX64.fullmatch(str(document.get("ready_state_sha256", "")))
        or not isinstance(document.get("termination_contract_id"), str)
        or not document["termination_contract_id"]
        or not isinstance(profile, dict)
        or not HEX64.fullmatch(str(profile.get("sha256", "")))
        or sources.get("git_head") != config["checkouts"]["evaluation"]["commit"]
        or any(sources.get(key) != value for key, value in expected_sources.items())
        or recomputed is None
        or summary != recomputed
        or summary.get("n_attempts") != 100
        or summary.get("per_clip", {}).get("forehand", {}).get("n_attempts") != 50
        or summary.get("per_clip", {}).get("backhand", {}).get("n_attempts") != 50
    ):
        raise ContractError(f"Isaac {arm_name} scorecard header/summary exact contract mismatch")
    expected_items = schedule_document["items"]
    ready_sha = document["ready_state_sha256"]
    for index, (row, item) in enumerate(zip(attempts, expected_items)):
        projection = {
            key: row.get(key)
            for key in ("schedule_index", "clip", "bank_row", "question_id", "repeat", "hold_steps", "attempt_seed")
        }
        expected = {key: item[key] for key in projection}
        if (
            projection != expected
            or row.get("env_id") != index
            or row.get("side") != ("forehand" if item["clip"] == 0 else "backhand")
            or row.get("ready_state_sha256") != ready_sha
            or row.get("start_step") != 0
            or isinstance(row.get("end_step"), bool)
            or not isinstance(row.get("end_step"), int)
            or row["end_step"] <= 0
            or row.get("finalize_reason") not in ("clip_complete", "physical_fall", "guard_reset", "episode_timeout")
            or row.get("finalized") is not True
            or row.get("censored") is not False
            or any(not isinstance(row.get(key), bool) for key in ("physical_fall", "guard_reset", "reached_exact", "hit", "returned", "net_clear"))
        ):
            raise ContractError(f"Isaac {arm_name} attempt {index} is censored/reordered/malformed")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 100:
        raise ContractError(f"Isaac {arm_name} CSV does not have 100 attempts")
    bool_columns = ("finalized", "censored", "physical_fall", "guard_reset", "reached_exact", "hit", "returned", "net_clear")
    try:
        for index, (csv_row, json_row, item) in enumerate(zip(csv_rows, attempts, expected_items)):
            if (
                int(csv_row["schedule_index"]) != item["schedule_index"]
                or int(csv_row["env_id"]) != index
                or int(csv_row["clip"]) != item["clip"]
                or int(csv_row["bank_row"]) != item["bank_row"]
                or csv_row["question_id"] != item["question_id"]
                or int(csv_row["repeat"]) != item["repeat"]
                or int(csv_row["hold_steps"]) != item["hold_steps"]
                or int(csv_row["attempt_seed"]) != item["attempt_seed"]
                or csv_row["ready_state_sha256"] != ready_sha
                or any(csv_row[key] != str(json_row[key]) for key in bool_columns)
            ):
                raise ContractError(f"Isaac {arm_name} CSV row {index} differs from JSON/paper")
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"Isaac {arm_name} CSV schema is invalid: {exc}") from exc
    returned = {
        "aggregate": summary["n_returned"],
        "forehand": summary["per_clip"]["forehand"]["n_returned"],
        "backhand": summary["per_clip"]["backhand"]["n_returned"],
        "physical_falls": 100 - summary["n_no_fall"],
    }
    return {
        "run_name": arm["run_name"],
        "checkpoint_iteration": arm["checkpoint_iteration"],
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "training_contract_sha256": arm["training_contract_sha256"],
        "scorecard": {"path": str(json_path), "sha256": sha256_file(json_path)},
        "attempt_ledger": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "schedule_file_sha256": runtime["shared_schedule"]["file_sha256"],
        "schedule_sha256": runtime["shared_schedule"]["schedule_sha256"],
        "question_id_order": runtime["shared_schedule"]["question_id_order"],
        "ready_state_sha256": ready_sha,
        "nominal_eval_profile_sha256": profile["sha256"],
        "evaluation_contract_exact": True,
        "fresh_lineage": True,
        "formal_target": True,
        "summary": summary,
        "returned_counts": returned,
        "returned_rates": {
            "aggregate": returned["aggregate"] / 100,
            "forehand": returned["forehand"] / 50,
            "backhand": returned["backhand"] / 50,
        },
    }


def select_checkpoint(results: Mapping[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    def rank(name: str) -> tuple[int, int, int, int]:
        counts = results[name]["returned_counts"]
        return (
            counts["aggregate"],
            min(counts["forehand"], counts["backhand"]),
            -counts["physical_falls"],
            1 if name == "model_2000" else 0,
        )

    selected = max(ARM_ORDER, key=rank)
    return selected, {
        "selected_arm": selected,
        "selected_checkpoint_iteration": ITERATIONS[selected],
        "selected_checkpoint_sha256": results[selected]["checkpoint_sha256"],
        "rank_vector": {
            name: {
                "aggregate_return_count": results[name]["returned_counts"]["aggregate"],
                "min_side_return_count": min(results[name]["returned_counts"]["forehand"], results[name]["returned_counts"]["backhand"]),
                "physical_falls": results[name]["returned_counts"]["physical_falls"],
                "earlier_checkpoint": name == "model_2000",
            }
            for name in ARM_ORDER
        },
        "scope": EXPECTED_SELECTION_POLICY["scope"],
        "whole_arm_action": "continue_unmodified",
        "whole_arm_stop_allowed": False,
        "whole_arm_promote_allowed": False,
        "deploy_gate": False,
        "real_robot_authorized": False,
    }


def run_pair(
    *, runtime_path: Path, expected_runtime_sha256: str, gpus: Sequence[int], config: dict[str, Any]
) -> int:
    runtime, _, tools = validate_prepared_runtime(runtime_path, expected_runtime_sha256, config)
    if len(gpus) != 2 or any(isinstance(value, bool) or value < 0 for value in gpus):
        raise ContractError("run requires two non-negative GPUs in model_2000/model_4000 order")
    state_dir = Path(runtime["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=False)
    eval_root = Path(config["checkouts"]["evaluation"]["path"])
    workdir = isaac.isaac_workdir(config, tools)
    env = isaac.setup_environment(tools["setup_train_env"], eval_root=eval_root)
    results: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(ARM_ORDER):
        validate_mujoco_binding(config)
        validate_eval_tools(config, eval_root)
        require_file_sha("shared schedule", Path(runtime["shared_schedule"]["path"]), runtime["shared_schedule"]["file_sha256"])
        state_path = state_dir / f"{name}.state.json"
        log_path = state_dir / f"{name}.runner.log"
        output_json = Path(runtime["arms"][name]["output_json"])
        output_csv = Path(runtime["arms"][name]["output_csv"])
        if any(path.exists() for path in (state_path, log_path, output_json, output_csv)):
            raise ContractError(f"no-clobber: preserved Isaac artifacts exist for {name}")
        command = build_command(config=config, runtime=runtime, tools=tools, arm_name=name, gpu=int(gpus[index]))
        with log_path.open("xb", buffering=0) as log:
            proc = subprocess.Popen(
                command,
                cwd=workdir,
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
                "runtime_contract_sha256": expected_runtime_sha256,
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
        validate_mujoco_binding(config)
        validate_eval_tools(config, eval_root)
        require_file_sha("shared schedule", Path(runtime["shared_schedule"]["path"]), runtime["shared_schedule"]["file_sha256"])
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        isaac.require_success_handshake(log_text, output_json, output_csv)
        results[name] = validate_scorecard(
            json_path=output_json,
            csv_path=output_csv,
            arm_name=name,
            config=config,
            runtime=runtime,
        )
    first, second = (results[name] for name in ARM_ORDER)
    for key in ("schedule_file_sha256", "schedule_sha256", "question_id_order", "ready_state_sha256", "nominal_eval_profile_sha256"):
        if first[key] != second[key]:
            raise ContractError(f"paired Isaac cells disagree on shared runtime field {key}")
    selected_name, selection = select_checkpoint(results)
    mujoco_result = load_json(Path(config["mujoco_binding"]["paired_result"]["path"]))
    mujoco_delta = (
        mujoco_result["arms"]["model_2000"]["returned_rates"]["aggregate"]
        - mujoco_result["arms"]["model_4000"]["returned_rates"]["aggregate"]
    )
    isaac_delta = (
        results["model_2000"]["returned_rates"]["aggregate"]
        - results["model_4000"]["returned_rates"]["aggregate"]
    )
    strict_reproduction = isaac_delta > 0.0 and mujoco_delta > 0.0
    paired_result = {
        "schema_version": 1,
        "pair_id": config["contract_id"],
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_contract": {"path": str(runtime_path), "sha256": expected_runtime_sha256},
        "mujoco_paired_result": config["mujoco_binding"]["paired_result"],
        **EXPECTED_SEMANTICS,
        "shared_schedule_file_sha256": runtime["shared_schedule"]["file_sha256"],
        "shared_schedule_sha256": runtime["shared_schedule"]["schedule_sha256"],
        "question_id_order": runtime["shared_schedule"]["question_id_order"],
        "arms": results,
        "selection_policy": EXPECTED_SELECTION_POLICY,
        "isaac_selection": selection,
        "cross_engine_comparison": {
            "mujoco_selected_arm": mujoco_result["selected_checkpoint"]["arm"],
            "isaac_selected_arm": selected_name,
            "mujoco_model_2000_minus_model_4000_aggregate_return_rate": mujoco_delta,
            "isaac_model_2000_minus_model_4000_aggregate_return_rate": isaac_delta,
            "strict_checkpoint_ranking_reproduced": strict_reproduction,
            "status": "ranking_reproduced" if strict_reproduction else "ranking_not_reproduced",
            "whole_arm_action": "continue_unmodified",
            "formal_or_deployment_promotion": False,
            "deploy_gate": False,
            "real_robot_authorized": False,
        },
    }
    atomic_json_no_clobber(Path(runtime["paired_result_path"]), paired_result)
    print(
        f"[fresh-exact-paired-isaac-q50] complete; Isaac selected {selected_name} within frozen pair only"
    )
    print(
        "[fresh-exact-paired-isaac-q50] "
        f"cross-engine strict ranking reproduced={strict_reproduction}; whole arm continues"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract-check")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True, type=Path)
    runtime_parser = sub.add_parser("runtime-check")
    runtime_parser.add_argument("--runtime-contract", required=True, type=Path)
    runtime_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--runtime-contract", required=True, type=Path)
    run_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser.add_argument("--gpus", nargs=2, required=True, type=int, metavar=("MODEL_2000", "MODEL_4000"))
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    require_sha("expected config SHA", args.expected_config_sha256)
    require_file_sha("companion config", config_path, args.expected_config_sha256)
    config = load_config(config_path)
    require_file_sha("companion runner", Path(__file__).resolve(), config["tools"]["runner_sha256"])
    if args.command == "contract-check":
        validate_mujoco_binding(config)
        print("[fresh-exact-paired-isaac-q50] offline contract check PASS; no process started")
        return 0
    if args.command == "prepare":
        return prepare_runtime_contract(config_path=config_path, output_path=args.output.resolve(), config=config)
    if args.command == "runtime-check":
        validate_prepared_runtime(
            args.runtime_contract.resolve(), args.expected_runtime_contract_sha256, config
        )
        print("[fresh-exact-paired-isaac-q50] runtime check PASS; no process started")
        return 0
    return run_pair(
        runtime_path=args.runtime_contract.resolve(),
        expected_runtime_sha256=args.expected_runtime_contract_sha256,
        gpus=args.gpus,
        config=config,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[fresh-exact-paired-isaac-q50][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
