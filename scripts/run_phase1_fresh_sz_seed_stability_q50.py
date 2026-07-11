#!/usr/bin/env python3
"""Run the four-seed fresh-SZ model_2000 q50 stability paper safely.

The runner has no SSH and no signal path.  Deploy it as an immutable external-control
artifact on each Pod.  ``contract-check`` is read-only, ``prepare`` copies one already
materialized content-addressed K100 schedule into a new no-clobber state directory without
starting a judge, and ``run`` evaluates that Pod's two preregistered arms sequentially.
Seed 1 may reuse its earlier accepted q50 only after the complete prior runtime/report/ledger
chain is revalidated against the identical paper; otherwise pass ``--rerun-seed1``.

``aggregate`` is dependency-light and combines copied Pod results.  Its gate is evidence about
seed stability at model_2000 only.  It never authorizes stopping training, promotion,
deployment, a checkout mutation, a process signal, or a real-robot command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
FRESH_RUNNER_PATH = SCRIPT_DIR / "run_phase1_fresh_exact_paired_bank_q50.py"
FRESH_RUNNER_SHA256 = "3528250777a170791f39d8dd17716c2a7f8ca91416a3ffa8433ec5eb691ed9e0"


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not FRESH_RUNNER_PATH.is_file() or _bootstrap_sha256(FRESH_RUNNER_PATH) != FRESH_RUNNER_SHA256:
    raise RuntimeError(
        "refusing to import unbound fresh exact q50 validator; deploy the preregistered "
        "35282507... dependency beside this runner"
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_phase1_fresh_exact_paired_bank_q50 as fresh  # noqa: E402


ContractError = fresh.ContractError
sha256_file = fresh.sha256_file
canonical_bytes = fresh.canonical_bytes
canonical_sha256 = fresh.canonical_sha256
atomic_json = fresh.atomic_json
load_json = fresh.load_json
exact_keys = fresh.exact_keys
require_sha = fresh.require_sha
require_absolute = fresh.require_absolute
require_under = fresh.require_under

SEED_ORDER = ("seed1", "seed2", "seed3", "seed4")
POD_ARM_ORDER = {"pod1": ("seed1", "seed3"), "pod2": ("seed2", "seed4")}
EXPECTED_SEMANTICS = {
    "fresh_lineage": True,
    "evaluation_contract_exact": True,
    "formal_target": True,
    "purpose": "four_seed_model_2000_stability_checkpoint_evidence",
    "training_mutation_allowed": False,
    "trainer_or_worker_signal_allowed": False,
    "whole_arm_stop_allowed": False,
    "whole_arm_promote_allowed": False,
    "deploy_gate": False,
    "real_robot_authorized": False,
}
EXPECTED_SCHEDULE = {
    "schema_version": 3,
    "file_sha256": "66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3",
    "semantic_sha256": "7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e",
    "question_id_order_sha256": "b87e81a34ff2d31766e17345f0a8c9d77665b78874093e26bdae257e8ed21f91",
    "per_clip_quota": 50,
    "schedule_k": 100,
    "attempts_per_side": 50,
    "schedule_seed": 0,
    "hold_range": [0, 100],
    "noise_scales": [0.0],
    "one_question_reset": True,
    "no_wrap": True,
    "materialize_new_schedule_allowed": False,
    "copy_identical_bytes_to_each_pod": True,
    "allow_inexact_contract": False,
}
EXPECTED_GATE_RULE = {
    "aggregate_rate_median_min": 0.75,
    "aggregate_rate_min_seed_min": 0.65,
    "aggregate_rate_max_minus_min_max": 0.20,
    "every_seed_every_side_rate_min": 0.50,
    "gate_scope": "seed_stability_checkpoint_evidence_only",
    "failure_action": "continue_all_arms_unmodified_and_keep_seed_stability_open",
    "pass_action": "record_model_2000_seed_stability_only_continue_all_arms_unmodified",
    "stop_or_promote_authorized": False,
    "deploy_or_real_robot_authorized": False,
}


def _strict_json(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def load_execution_config(path: Path) -> dict[str, Any]:
    data = _strict_json(path)
    exact_keys(
        data,
        {
            "schema_version", "contract_id", "status", "auto_start",
            "preregistration_sha256", "semantics", "checkouts", "tools", "schedule",
            "runtime", "pod_arm_order", "gate_rule", "seed1_reuse",
        },
        "seed-stability execution config",
    )
    if data["schema_version"] != 1:
        raise ContractError("execution config schema_version must be 1")
    if data["status"] != "offline_preregistered_not_prepared" or data["auto_start"] is not False:
        raise ContractError("execution config must remain offline/not-prepared")
    require_sha("preregistration_sha256", data["preregistration_sha256"])
    if data["semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("execution semantics changed")
    if data["schedule"] != EXPECTED_SCHEDULE:
        raise ContractError("execution schedule is not the frozen reused K100 paper")
    if data["pod_arm_order"] != {key: list(value) for key, value in POD_ARM_ORDER.items()}:
        raise ContractError("distributed Pod/arm order changed")
    if data["gate_rule"] != EXPECTED_GATE_RULE:
        raise ContractError("seed-stability decision rule changed")

    checkouts = data["checkouts"]
    exact_keys(checkouts, {"training", "evaluation"}, "checkouts")
    for name, spec in checkouts.items():
        exact_keys(spec, {"path", "commit"}, f"checkouts.{name}")
        require_absolute(f"checkouts.{name}.path", spec["path"])
        require_sha(f"checkouts.{name}.commit", spec["commit"], length=40)

    tools = data["tools"]
    exact_keys(tools, {"runner_sha256", "fresh_exact_validator", "evaluation"}, "tools")
    require_sha("tools.runner_sha256", tools["runner_sha256"])
    exact_keys(
        tools["fresh_exact_validator"], {"path", "sha256"},
        "tools.fresh_exact_validator",
    )
    if tools["fresh_exact_validator"] != {
        "path": "run_phase1_fresh_exact_paired_bank_q50.py",
        "sha256": FRESH_RUNNER_SHA256,
    }:
        raise ContractError("fresh exact validator dependency changed")
    expected_eval = {"judge", "materialize_schedule", "schedule_module", "mujoco_evaluator"}
    exact_keys(tools["evaluation"], expected_eval, "tools.evaluation")
    for name, spec in tools["evaluation"].items():
        exact_keys(spec, {"path", "sha256"}, f"tools.evaluation.{name}")
        if (
            not isinstance(spec["path"], str)
            or os.path.isabs(spec["path"])
            or ".." in Path(spec["path"]).parts
        ):
            raise ContractError(f"tools.evaluation.{name}.path must be repo-relative")
        require_sha(f"tools.evaluation.{name}.sha256", spec["sha256"])

    runtime = data["runtime"]
    exact_keys(
        runtime,
        {
            "checkpoint_python", "pod_state_dirs", "schedule_filename",
            "runtime_contract_filename", "pod_result_filename", "aggregate_output_dir",
        },
        "runtime",
    )
    require_absolute("runtime.checkpoint_python", runtime["checkpoint_python"])
    if set(runtime["pod_state_dirs"]) != set(POD_ARM_ORDER):
        raise ContractError("runtime.pod_state_dirs must name pod1 and pod2")
    for pod, value in runtime["pod_state_dirs"].items():
        require_absolute(f"runtime.pod_state_dirs.{pod}", value)
    require_absolute("runtime.aggregate_output_dir", runtime["aggregate_output_dir"])
    for key in ("schedule_filename", "runtime_contract_filename", "pod_result_filename"):
        value = runtime[key]
        if not isinstance(value, str) or Path(value).name != value or not value.endswith(".json"):
            raise ContractError(f"runtime.{key} must be a simple .json filename")

    reuse = data["seed1_reuse"]
    exact_keys(
        reuse,
        {
            "allowed_only_after_full_revalidation", "fallback", "prior_runtime_contract",
            "prior_paired_result", "prior_checked_result",
        },
        "seed1_reuse",
    )
    if reuse["allowed_only_after_full_revalidation"] is not True or reuse["fallback"] != (
        "rerun_seed1_on_identical_schedule"
    ):
        raise ContractError("seed1 reuse fence changed")
    for key in ("prior_runtime_contract", "prior_paired_result", "prior_checked_result"):
        exact_keys(reuse[key], {"path", "sha256"}, f"seed1_reuse.{key}")
        require_absolute(f"seed1_reuse.{key}.path", reuse[key]["path"])
        require_sha(f"seed1_reuse.{key}.sha256", reuse[key]["sha256"])
    return data


def validate_preregistration(data: dict[str, Any], config: dict[str, Any]) -> None:
    exact_keys(
        data,
        {
            "schema_version", "preregistration_id", "created_utc", "status",
            "auto_activate", "jobs_started", "runtime_state", "scope", "q10_trigger",
            "family", "training_commit", "eval_commit", "tools", "paper", "arms",
            "gate_rule", "formal_semantics", "activation", "preflight_requirements",
        },
        "seed-stability preregistration",
    )
    if data["schema_version"] != 1 or data["status"] != "preregistered_not_started":
        raise ContractError("preregistration must be pristine schema-version 1")
    if data["auto_activate"] is not False or data["jobs_started"] != 0 or data[
        "runtime_state"
    ] is not None:
        raise ContractError("preregistration already started or auto-activates")
    if data["training_commit"] != config["checkouts"]["training"]["commit"] or data[
        "eval_commit"
    ] != config["checkouts"]["evaluation"]["commit"]:
        raise ContractError("preregistration checkout commits disagree with execution config")
    if data["family"] != {
        "cell": "SZ",
        "name": "v4rg_runtime_order_v3",
        "source_family_sha256": "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5",
        "face_command_pairing": "shared_plus_y",
        "zero_joint_friction": True,
        "checkpoint_iteration": 2000,
    }:
        raise ContractError("fresh SZ family/checkpoint binding changed")
    if data["tools"] != {
        "runner_sha256": config["tools"]["runner_sha256"],
        "fresh_exact_validator_sha256": FRESH_RUNNER_SHA256,
        "judge_sha256": config["tools"]["evaluation"]["judge"]["sha256"],
        "mujoco_evaluator_sha256": config["tools"]["evaluation"]["mujoco_evaluator"]["sha256"],
    }:
        raise ContractError("preregistered tool bytes changed")
    for name, value in data["tools"].items():
        require_sha(f"tools.{name}", value)

    trigger = data["q10_trigger"]
    exact_keys(
        trigger,
        {
            "role", "screen_only", "stop_or_promote_allowed", "same_semantic_schedule_sha256",
            "observed_aggregate_return_rate", "evidence",
        },
        "q10_trigger",
    )
    if (
        trigger["role"] != "direction_only_trigger_for_same_paper_q50"
        or trigger["screen_only"] is not True
        or trigger["stop_or_promote_allowed"] is not False
        or trigger["same_semantic_schedule_sha256"]
        != "1335858971689fed5caf7c70947a0741dcf35b51253f08e10d63e8d64ed8866a"
        or trigger["observed_aggregate_return_rate"]
        != {"seed1": 0.9, "seed2": 1.0, "seed3": 1.0, "seed4": 0.25}
    ):
        raise ContractError("q10 trigger was changed or promoted into a decision")
    if not isinstance(trigger["evidence"], list) or len(trigger["evidence"]) != 4:
        raise ContractError("q10 trigger must bind one evidence record per seed")
    seen_trigger = set()
    for row in trigger["evidence"]:
        exact_keys(row, {"seed", "pod", "state", "log"}, "q10_trigger.evidence[]")
        seed = row["seed"]
        if seed not in SEED_ORDER or seed in seen_trigger:
            raise ContractError("q10 trigger has duplicate/unknown seed")
        seen_trigger.add(seed)
        for artifact in (row["state"], row["log"]):
            exact_keys(artifact, {"path", "sha256"}, "q10 trigger artifact")
            require_absolute("q10 trigger artifact path", artifact["path"])
            require_sha("q10 trigger artifact sha256", artifact["sha256"])

    paper = data["paper"]
    exact_keys(
        paper,
        {
            "role", "exam_bank", "schedule", "mjcf_sha256", "seed", "noise_scales",
            "schedule_k", "attempts_per_side", "hold_ref", "hold_steps_range", "no_wrap",
            "one_question_reset", "allow_inexact_contract_required",
            "expected_evaluation_contract_exact", "reuse_independence",
        },
        "paper",
    )
    if (
        paper["role"] != "four-seed fresh SZ model_2000 stability q50"
        or paper["seed"] != 0
        or paper["noise_scales"] != [0.0]
        or paper["schedule_k"] != 100
        or paper["attempts_per_side"] != 50
        or paper["hold_ref"] != "auto"
        or paper["hold_steps_range"] != [0, 100]
        or paper["no_wrap"] is not True
        or paper["one_question_reset"] is not True
        or paper["allow_inexact_contract_required"] is not False
        or paper["expected_evaluation_contract_exact"] is not True
        or paper["schedule"] != EXPECTED_SCHEDULE
    ):
        raise ContractError("formal q50 paper changed")
    bank = paper["exam_bank"]
    exact_keys(
        bank, {"path", "bytes", "sha256", "schema_version", "source_family_sha256"},
        "paper.exam_bank",
    )
    require_absolute("paper.exam_bank.path", bank["path"])
    require_sha("paper.exam_bank.sha256", bank["sha256"])
    if (
        bank["bytes"] != 63968
        or bank["schema_version"] != 3
        or bank["source_family_sha256"] != data["family"]["source_family_sha256"]
    ):
        raise ContractError("exam bank differs from the exact family")
    require_sha("paper.mjcf_sha256", paper["mjcf_sha256"])
    independence = paper["reuse_independence"]
    exact_keys(
        independence,
        {
            "source_preregistration", "source_result", "materialized_before_any_q50_outcome",
            "selected_without_seed2_seed3_seed4_policy_or_outcome", "new_materialization_forbidden",
        },
        "paper.reuse_independence",
    )
    if (
        independence["materialized_before_any_q50_outcome"] is not True
        or independence["selected_without_seed2_seed3_seed4_policy_or_outcome"] is not True
        or independence["new_materialization_forbidden"] is not True
    ):
        raise ContractError("reused schedule independence is not explicit")
    for key in ("source_preregistration", "source_result"):
        exact_keys(independence[key], {"path", "sha256"}, f"reuse_independence.{key}")
        require_sha(f"reuse_independence.{key}.sha256", independence[key]["sha256"])

    if list(data["arms"]) != list(SEED_ORDER):
        raise ContractError(f"arms must be ordered {list(SEED_ORDER)}")
    expected_pods = {"seed1": "pod1", "seed2": "pod2", "seed3": "pod1", "seed4": "pod2"}
    expected_modes = {"seed1": "reuse_candidate_or_identical_rerun", "seed2": "judge_required", "seed3": "judge_required", "seed4": "judge_required"}
    hard_sha = None
    for ordinal, seed in enumerate(SEED_ORDER, start=1):
        arm = data["arms"][seed]
        exact_keys(
            arm,
            {
                "training_seed", "pod", "gpu", "run_name", "checkpoint_iteration",
                "checkpoint_path", "checkpoint_sha256", "training_contract_path",
                "training_contract_sha256", "checkpoint_embedded_training_contract_sha256",
                "cell", "face_command_pairing", "zero_joint_friction", "lineage_exact",
                "execution_mode", "job_status", "pid", "pgid", "result",
            },
            f"arms.{seed}",
        )
        if (
            arm["training_seed"] != ordinal
            or arm["pod"] != expected_pods[seed]
            or isinstance(arm["gpu"], bool)
            or not isinstance(arm["gpu"], int)
            or arm["gpu"] < 0
            or arm["checkpoint_iteration"] != 2000
            or arm["cell"] != "SZ"
            or arm["face_command_pairing"] != "shared_plus_y"
            or arm["zero_joint_friction"] is not True
            or arm["lineage_exact"] is not True
            or arm["execution_mode"] != expected_modes[seed]
            or arm["job_status"] != "not_started"
            or arm["pid"] is not None
            or arm["pgid"] is not None
            or arm["result"] is not None
        ):
            raise ContractError(f"arm {seed} is not the frozen SZ model_2000 target")
        checkpoint = require_absolute(f"arms.{seed}.checkpoint_path", arm["checkpoint_path"])
        contract = require_absolute(
            f"arms.{seed}.training_contract_path", arm["training_contract_path"]
        )
        training_root = Path(config["checkouts"]["training"]["path"])
        require_under(f"arms.{seed}.checkpoint_path", checkpoint, training_root)
        require_under(f"arms.{seed}.training_contract_path", contract, training_root)
        if checkpoint.name != "model_2000.pt" or contract != checkpoint.parent / "params" / (
            "training_contract.json"
        ):
            raise ContractError(f"arm {seed} checkpoint/adjacent contract path changed")
        require_sha(f"arms.{seed}.checkpoint_sha256", arm["checkpoint_sha256"])
        current_hard = require_sha(
            f"arms.{seed}.training_contract_sha256", arm["training_contract_sha256"]
        )
        if arm["checkpoint_embedded_training_contract_sha256"] != current_hard:
            raise ContractError(f"arm {seed} loses embedded hard-contract binding")
        hard_sha = current_hard if hard_sha is None else hard_sha
        if current_hard != hard_sha:
            raise ContractError("four SZ seeds do not share byte-identical hard contract")
    if data["gate_rule"] != EXPECTED_GATE_RULE or data["formal_semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("gate/formal semantics changed")
    if data["activation"] != {
        "preregistered": True,
        "authorized_to_start_only_after_committed_prereg_and_all_preflights": True,
        "started_at_creation": False,
    }:
        raise ContractError("activation fence changed")


def validate_eval_tools(config: dict[str, Any], eval_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["tools"]["evaluation"].items():
        path = (eval_root / spec["path"]).resolve()
        require_under(f"tools.evaluation.{name}", path, eval_root)
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ContractError(f"evaluation tool bytes changed: {name} {path}")
        paths[name] = path
    if sha256_file(FRESH_RUNNER_PATH) != FRESH_RUNNER_SHA256:
        raise ContractError("fresh exact validator dependency changed")
    return paths


def validate_schedule(path: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    expected = prereg["paper"]["schedule"]
    if not path.is_file() or sha256_file(path) != expected["file_sha256"]:
        raise ContractError("shared K100 schedule file SHA mismatch")
    schedule = fresh.validate_schedule_document(
        path, expected_bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    order = [item["question_id"] for item in schedule["items"]]
    if (
        schedule["schedule_sha256"] != expected["semantic_sha256"]
        or canonical_sha256(order) != expected["question_id_order_sha256"]
        or len(schedule["items"]) != 100
    ):
        raise ContractError("shared K100 schedule semantics/order changed")
    return schedule


def validate_hard_contract(hard: dict[str, Any], prereg: dict[str, Any]) -> None:
    fresh.validate_hard_contract(hard, prereg)


def validate_q10_trigger_sources(prereg: dict[str, Any], *, pod: str) -> None:
    """Re-hash this Pod's preserved q10 trigger evidence without promoting it."""

    expected_seeds = set(POD_ARM_ORDER[pod])
    seen = set()
    for row in prereg["q10_trigger"]["evidence"]:
        if row["pod"] != pod:
            continue
        seed = row["seed"]
        if seed not in expected_seeds or seed in seen:
            raise ContractError(f"{pod} q10 trigger seed coverage changed")
        seen.add(seed)
        for key in ("state", "log"):
            artifact = row[key]
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ContractError(f"{pod}/{seed} q10 trigger {key} changed/missing")
    if seen != expected_seeds:
        raise ContractError(f"{pod} q10 trigger evidence is incomplete")


def validate_runtime_inputs(
    config: dict[str, Any], prereg: dict[str, Any], *, pod: str, schedule_path: Path,
) -> tuple[Path, Path, dict[str, Path], dict[str, Any], dict[str, Any]]:
    if pod not in POD_ARM_ORDER:
        raise ContractError(f"unknown pod {pod!r}")
    training_root = fresh.validate_checkout("training", config["checkouts"]["training"])
    eval_root = fresh.validate_checkout("evaluation", config["checkouts"]["evaluation"])
    tools = validate_eval_tools(config, eval_root)
    schedule = validate_schedule(schedule_path, prereg)
    validate_q10_trigger_sources(prereg, pod=pod)
    checkpoint_python = Path(config["runtime"]["checkpoint_python"])
    if not checkpoint_python.is_file() or not os.access(checkpoint_python, os.X_OK):
        raise ContractError(f"checkpoint Python missing/not executable: {checkpoint_python}")
    bank = prereg["paper"]["exam_bank"]
    bank_path = Path(bank["path"])
    if (
        not bank_path.is_file()
        or bank_path.stat().st_size != bank["bytes"]
        or sha256_file(bank_path) != bank["sha256"]
    ):
        raise ContractError("exact exam-bank bytes changed")
    audits = {}
    for seed in POD_ARM_ORDER[pod]:
        arm = prereg["arms"][seed]
        checkpoint = Path(arm["checkpoint_path"])
        hard_path = Path(arm["training_contract_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != arm["checkpoint_sha256"]:
            raise ContractError(f"{seed} checkpoint bytes changed")
        if not hard_path.is_file() or sha256_file(hard_path) != arm["training_contract_sha256"]:
            raise ContractError(f"{seed} hard-contract bytes changed")
        hard = _strict_json(hard_path)
        validate_hard_contract(hard, prereg)
        audits[seed] = fresh.checkpoint_audit(
            checkpoint_python,
            checkpoint,
            expected_iteration=2000,
            expected_contract_sha=arm["training_contract_sha256"],
        )
    return training_root, eval_root, tools, audits, schedule


def _runtime_document(
    *, config_path: Path, config: dict[str, Any], prereg_path: Path,
    prereg: dict[str, Any], pod: str, schedule_path: Path, schedule: dict[str, Any],
    audits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "pod": pod,
        "status": "prepared_not_started",
        "auto_start": False,
        "jobs_started": 0,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **EXPECTED_SEMANTICS,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "checkouts": config["checkouts"],
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "question_id_order": [item["question_id"] for item in schedule["items"]],
            "question_id_order_sha256": canonical_sha256(
                [item["question_id"] for item in schedule["items"]]
            ),
            "schedule_k": 100,
            "attempts_per_side": 50,
            "seed": 0,
            "hold_range": [0, 100],
        },
        "arm_order": list(POD_ARM_ORDER[pod]),
        "arms": {
            seed: {
                "checkpoint_path": prereg["arms"][seed]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][seed]["checkpoint_sha256"],
                "training_contract_sha256": prereg["arms"][seed]["training_contract_sha256"],
                "checkpoint_audit": audits[seed],
                "execution_mode": prereg["arms"][seed]["execution_mode"],
                "job_status": "not_started",
            }
            for seed in POD_ARM_ORDER[pod]
        },
        "gate_rule": EXPECTED_GATE_RULE,
    }


def prepare(
    config_path: Path, config: dict[str, Any], prereg_path: Path, prereg: dict[str, Any],
    *, pod: str, schedule_source: Path,
) -> int:
    _, _, _, audits, schedule = validate_runtime_inputs(
        config, prereg, pod=pod, schedule_path=schedule_source
    )
    state_dir = Path(config["runtime"]["pod_state_dirs"][pod])
    if state_dir.exists():
        raise ContractError(f"no-clobber: state directory exists: {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=False)
    copied_schedule = state_dir / config["runtime"]["schedule_filename"]
    shutil.copyfile(schedule_source, copied_schedule)
    copied = validate_schedule(copied_schedule, prereg)
    if copied != schedule:
        raise ContractError("copied schedule differs semantically from source")
    runtime_path = state_dir / config["runtime"]["runtime_contract_filename"]
    atomic_json(
        runtime_path,
        _runtime_document(
            config_path=config_path,
            config=config,
            prereg_path=prereg_path,
            prereg=prereg,
            pod=pod,
            schedule_path=copied_schedule,
            schedule=copied,
            audits=audits,
        ),
    )
    print(f"[seed-stability-q50] {pod} prepared only; no judge started")
    print(
        f"[seed-stability-q50] schedule={copied_schedule} "
        f"file_sha256={sha256_file(copied_schedule)} semantic_sha256={copied['schedule_sha256']}"
    )
    print(f"[seed-stability-q50] runtime_contract={runtime_path} sha256={sha256_file(runtime_path)}")
    return 0


def validate_runtime_contract(
    path: Path, expected_sha: str, config: dict[str, Any], prereg: dict[str, Any], *, pod: str,
) -> dict[str, Any]:
    require_sha("expected runtime-contract SHA", expected_sha)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ContractError("runtime-contract SHA mismatch")
    contract = _strict_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_id") != config["contract_id"]
        or contract.get("pod") != pod
        or contract.get("status") != "prepared_not_started"
        or contract.get("auto_start") is not False
        or contract.get("jobs_started") != 0
        or contract.get("arm_order") != list(POD_ARM_ORDER[pod])
        or contract.get("gate_rule") != EXPECTED_GATE_RULE
        or any(contract.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError("runtime contract is not a pristine prepared Pod paper")
    if contract.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise ContractError("runtime contract runner bytes changed")
    for key, expected_path in (
        ("config", None),
        ("preregistration", None),
    ):
        meta = contract.get(key, {})
        candidate = Path(str(meta.get("path", "")))
        if (
            not candidate.is_absolute()
            or not candidate.is_file()
            or meta.get("sha256") != sha256_file(candidate)
        ):
            raise ContractError(f"runtime contract {key} bytes changed")
    if load_execution_config(Path(contract["config"]["path"])) != config:
        raise ContractError("runtime contract points at a different execution config")
    if _strict_json(Path(contract["preregistration"]["path"])) != prereg:
        raise ContractError("runtime contract points at a different preregistration")
    if contract["preregistration"]["sha256"] != config["preregistration_sha256"]:
        raise ContractError("runtime contract preregistration differs from config binding")
    if contract["config"]["sha256"] != sha256_file(Path(contract["config"]["path"])):
        raise ContractError("runtime config hash mismatch")
    if contract["preregistration"]["sha256"] != sha256_file(
        Path(contract["preregistration"]["path"])
    ):
        raise ContractError("runtime preregistration hash mismatch")
    schedule_meta = contract.get("shared_schedule", {})
    schedule_path = Path(str(schedule_meta.get("path", "")))
    schedule = validate_schedule(schedule_path, prereg)
    if (
        schedule_meta.get("file_sha256") != sha256_file(schedule_path)
        or schedule_meta.get("schedule_sha256") != schedule["schedule_sha256"]
        or schedule_meta.get("question_id_order")
        != [item["question_id"] for item in schedule["items"]]
        or schedule_meta.get("question_id_order_sha256")
        != canonical_sha256([item["question_id"] for item in schedule["items"]])
    ):
        raise ContractError("runtime contract shared schedule changed")
    return contract


def _compat_runtime(schedule_path: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    schedule = validate_schedule(schedule_path, prereg)
    return {
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "question_id_order": [item["question_id"] for item in schedule["items"]],
        }
    }


def validate_seed1_reuse(
    config: dict[str, Any], prereg: dict[str, Any], current_runtime: dict[str, Any]
) -> dict[str, Any]:
    reuse = config["seed1_reuse"]
    checked = reuse["prior_checked_result"]
    checked_path = Path(checked["path"])
    if not checked_path.is_file() or sha256_file(checked_path) != checked["sha256"]:
        raise ContractError("seed1 checked reuse result changed/missing")
    checked_result = _strict_json(checked_path)
    prior_runtime_meta = reuse["prior_runtime_contract"]
    prior_runtime_path = Path(prior_runtime_meta["path"])
    if (
        not prior_runtime_path.is_file()
        or sha256_file(prior_runtime_path) != prior_runtime_meta["sha256"]
    ):
        raise ContractError("seed1 prior runtime contract changed/missing")
    prior_runtime = _strict_json(prior_runtime_path)
    prior_schedule_path = Path(prior_runtime.get("shared_schedule", {}).get("path", ""))
    prior_schedule = validate_schedule(prior_schedule_path, prereg)
    current_schedule_path = Path(current_runtime["shared_schedule"]["path"])
    current_schedule = validate_schedule(current_schedule_path, prereg)
    if prior_schedule != current_schedule or sha256_file(prior_schedule_path) != sha256_file(
        current_schedule_path
    ):
        raise ContractError("seed1 reuse paper differs from current Pod schedule bytes")
    paired_meta = reuse["prior_paired_result"]
    paired_path = Path(paired_meta["path"])
    if not paired_path.is_file() or sha256_file(paired_path) != paired_meta["sha256"]:
        raise ContractError("seed1 prior paired result changed/missing")
    paired = _strict_json(paired_path)
    if (
        paired.get("status") != "complete"
        or paired.get("shared_schedule_sha256") != EXPECTED_SCHEDULE["semantic_sha256"]
        or paired.get("evaluation_contract_exact") is not True
        or paired.get("fresh_lineage") is not True
        or paired.get("formal_target") is not True
    ):
        raise ContractError("seed1 prior paired result is not the accepted exact paper")
    source = paired.get("arms", {}).get("model_2000")
    if not isinstance(source, dict):
        raise ContractError("seed1 prior paired result lacks model_2000")
    arm = prereg["arms"]["seed1"]
    report_path = Path(source.get("report", {}).get("path", ""))
    if (
        source.get("checkpoint_iteration") != 2000
        or source.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or source.get("training_contract_sha256") != arm["training_contract_sha256"]
        or not report_path.is_file()
        or source.get("report", {}).get("sha256") != sha256_file(report_path)
    ):
        raise ContractError("seed1 prior result checkpoint/report binding changed")
    validated = fresh.validate_exam_result(
        report=report_path,
        arm_name="model_2000",
        arm=arm,
        prereg=prereg,
        runtime_contract=_compat_runtime(prior_schedule_path, prereg),
    )
    for key in ("summary", "attempt_ledger"):
        if validated[key] != source[key]:
            raise ContractError(f"seed1 prior {key} bytes changed")
    if validated["returned_counts"] != source["returned_counts"]:
        raise ContractError("seed1 prior headline counts no longer reproduce raw ledger")
    checked_arm = checked_result.get("arms", {}).get("model_2000", {})
    if (
        checked_arm.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or checked_arm.get("exam_summary_sha256") != validated["summary"]["sha256"]
        or checked_arm.get("attempt_ledger_sha256") != validated["attempt_ledger"]["sha256"]
    ):
        raise ContractError("checked seed1 result no longer binds the revalidated raw artifacts")
    return validated


def check_seed1_reuse(
    config: dict[str, Any], prereg: dict[str, Any], runtime_path: Path, runtime_sha: str,
) -> int:
    """Revalidate seed1's complete prior q50 chain without starting or writing a judge."""

    contract = validate_runtime_contract(
        runtime_path, runtime_sha, config, prereg, pod="pod1"
    )
    validate_runtime_inputs(
        config,
        prereg,
        pod="pod1",
        schedule_path=Path(contract["shared_schedule"]["path"]),
    )
    validated = validate_seed1_reuse(config, prereg, contract)
    print(
        "[seed-stability-q50] seed1 reuse check PASS; "
        f"report_sha256={validated['report']['sha256']} "
        f"summary_sha256={validated['summary']['sha256']} "
        f"attempt_ledger_sha256={validated['attempt_ledger']['sha256']}"
    )
    print("[seed-stability-q50] no write/judge/signal; identical rerun remains the fallback")
    return 0


def _run_judge(
    *, seed: str, arm: dict[str, Any], tools: dict[str, Path], schedule_path: Path,
    state_dir: Path, runtime_sha: str, gpu: int,
) -> dict[str, Any]:
    state_path = state_dir / f"{seed}.state.json"
    log_path = state_dir / f"{seed}.runner.log"
    if state_path.exists() or log_path.exists():
        raise ContractError(f"no-clobber: preserved state/log exists for {seed}")
    command = fresh.build_judge_command(
        judge=tools["judge"], arm=arm, schedule_path=schedule_path, gpu=gpu
    )
    if any("--allow-inexact-contract" in value for value in command):
        raise ContractError("formal seed-stability command contains diagnostic escape")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    with log_path.open("wb", buffering=0) as log:
        proc = subprocess.Popen(
            command,
            cwd=tools["judge"].parent.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state = {
            "schema_version": 1,
            "seed": seed,
            "status": "running",
            "pid": proc.pid,
            "pgid": proc.pid,
            "command": command,
            "runtime_contract_sha256": runtime_sha,
            "schedule_sha256": EXPECTED_SCHEDULE["semantic_sha256"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(state_path, state)
        rc = proc.wait()
    state.update(
        status="complete" if rc == 0 else "failed",
        returncode=rc,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        log_sha256=sha256_file(log_path),
    )
    atomic_json(state_path, state)
    if rc != 0:
        raise ContractError(f"judge {seed} failed rc={rc}; preserved {state_path} and {log_path}")
    report = fresh.base.find_report(log_path.read_text(encoding="utf-8", errors="replace"))
    return {"state": state, "state_path": state_path, "log_path": log_path, "report": report}


def _content_document(kind: str, content: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "content_sha256": canonical_sha256(content),
        "content": content,
    }


def run_pod(
    config: dict[str, Any], prereg: dict[str, Any], runtime_path: Path, runtime_sha: str,
    *, pod: str, rerun_seed1: bool,
) -> int:
    contract = validate_runtime_contract(
        runtime_path, runtime_sha, config, prereg, pod=pod
    )
    _, _, tools, audits, _ = validate_runtime_inputs(
        config,
        prereg,
        pod=pod,
        schedule_path=Path(contract["shared_schedule"]["path"]),
    )
    state_dir = runtime_path.parent
    result_path = state_dir / config["runtime"]["pod_result_filename"]
    if result_path.exists():
        raise ContractError(f"no-clobber: Pod result exists: {result_path}")
    results: dict[str, dict[str, Any]] = {}
    execution: dict[str, dict[str, Any]] = {}
    for seed in POD_ARM_ORDER[pod]:
        arm = prereg["arms"][seed]
        if seed == "seed1" and not rerun_seed1:
            validated = validate_seed1_reuse(config, prereg, contract)
            state = {
                "schema_version": 1,
                "seed": seed,
                "status": "reused_after_full_revalidation",
                "pid": None,
                "pgid": None,
                "runtime_contract_sha256": runtime_sha,
                "prior_paired_result_sha256": config["seed1_reuse"]["prior_paired_result"][
                    "sha256"
                ],
                "report_sha256": validated["report"]["sha256"],
                "summary_sha256": validated["summary"]["sha256"],
                "attempt_ledger_sha256": validated["attempt_ledger"]["sha256"],
            }
            state_path = state_dir / f"{seed}.state.json"
            if state_path.exists():
                raise ContractError(f"no-clobber: seed1 reuse state exists: {state_path}")
            atomic_json(state_path, state)
            execution[seed] = {
                "mode": "reuse_validated_no_judge_started",
                "state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            }
        else:
            launched = _run_judge(
                seed=seed,
                arm=arm,
                tools=tools,
                schedule_path=Path(contract["shared_schedule"]["path"]),
                state_dir=state_dir,
                runtime_sha=runtime_sha,
                gpu=arm["gpu"],
            )
            validated = fresh.validate_exam_result(
                report=launched["report"],
                arm_name="model_2000",
                arm=arm,
                prereg=prereg,
                runtime_contract=contract,
            )
            execution[seed] = {
                "mode": "fresh_identical_paper_judge",
                "state": {
                    "path": str(launched["state_path"]),
                    "sha256": sha256_file(launched["state_path"]),
                },
                "runner_log": {
                    "path": str(launched["log_path"]),
                    "sha256": sha256_file(launched["log_path"]),
                },
            }
        results[seed] = validated
        results[seed]["checkpoint_audit"] = audits[seed]
    first, second = (results[seed] for seed in POD_ARM_ORDER[pod])
    for key in (
        "schedule_sha256", "question_id_order", "mjcf_sha256", "execution_contract_sha256",
        "ready_state_sha256",
    ):
        if first[key] != second[key]:
            raise ContractError(f"{pod} q50 arms disagree on shared runtime field {key}")
    content = {
        "contract_id": config["contract_id"],
        "pod": pod,
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_contract": {"path": str(runtime_path), "sha256": runtime_sha},
        "config_sha256": sha256_file(Path(contract["config"]["path"])),
        "preregistration_sha256": sha256_file(Path(contract["preregistration"]["path"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        **EXPECTED_SEMANTICS,
        "shared_schedule": contract["shared_schedule"],
        "arm_order": list(POD_ARM_ORDER[pod]),
        "execution": execution,
        "arms": results,
        "gate_rule": EXPECTED_GATE_RULE,
        "actions": {
            "training": "continue_all_arms_unmodified",
            "trainer_or_worker_signals": [],
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
        },
    }
    atomic_json(result_path, _content_document("phase1_fresh_sz_seed_stability_q50_pod", content))
    print(f"[seed-stability-q50] {pod} complete: {result_path} sha256={sha256_file(result_path)}")
    print("[seed-stability-q50] all trainers/workers untouched; all arms continue")
    return 0


def validate_pod_result(
    path: Path, expected_sha: str, config: dict[str, Any], prereg_sha: str, *, pod: str,
) -> dict[str, Any]:
    require_sha(f"{pod} result SHA", expected_sha)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ContractError(f"{pod} result bytes changed")
    document = _strict_json(path)
    exact_keys(
        document, {"schema_version", "artifact_kind", "content_sha256", "content"},
        f"{pod} result",
    )
    content = document["content"]
    if (
        document["schema_version"] != 1
        or document["artifact_kind"] != "phase1_fresh_sz_seed_stability_q50_pod"
        or not isinstance(content, dict)
        or document["content_sha256"] != canonical_sha256(content)
        or content.get("contract_id") != config["contract_id"]
        or content.get("pod") != pod
        or content.get("status") != "complete"
        or content.get("preregistration_sha256") != prereg_sha
        or content.get("runner_sha256") != config["tools"]["runner_sha256"]
        or content.get("arm_order") != list(POD_ARM_ORDER[pod])
        or content.get("gate_rule") != EXPECTED_GATE_RULE
        or any(content.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError(f"{pod} result is not the frozen complete Pod artifact")
    actions = content.get("actions", {})
    if actions != {
        "training": "continue_all_arms_unmodified",
        "trainer_or_worker_signals": [],
        "stop_or_promote_authorized": False,
        "deploy_or_real_robot_authorized": False,
    }:
        raise ContractError(f"{pod} result contains an unauthorized action")
    if set(content.get("arms", {})) != set(POD_ARM_ORDER[pod]):
        raise ContractError(f"{pod} result arm coverage is incomplete")
    for seed in POD_ARM_ORDER[pod]:
        arm = content["arms"][seed]
        expected = config.get("_prereg_arms", {}).get(seed)
        if not isinstance(arm, dict):
            raise ContractError(f"{pod}/{seed} result missing")
        if (
            arm.get("checkpoint_iteration") != 2000
            or arm.get("evaluation_contract_exact") is not True
            or arm.get("formal_target") is not True
            or arm.get("fresh_lineage") is not True
            or arm.get("denominators") != {"aggregate": 100, "forehand": 50, "backhand": 50}
            or arm.get("schedule_sha256") != EXPECTED_SCHEDULE["semantic_sha256"]
        ):
            raise ContractError(f"{pod}/{seed} result violates the exact K100 contract")
        counts = arm.get("returned_counts", {})
        if any(
            isinstance(counts.get(key), bool)
            or not isinstance(counts.get(key), int)
            or counts[key] < 0
            for key in ("aggregate", "forehand", "backhand", "physical_falls")
        ) or counts["aggregate"] != counts["forehand"] + counts["backhand"]:
            raise ContractError(f"{pod}/{seed} returned counts are invalid")
        if expected is not None and (
            arm.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or arm.get("training_contract_sha256") != expected["training_contract_sha256"]
        ):
            raise ContractError(f"{pod}/{seed} checkpoint/hard-contract changed")
    return content


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ContractError("cannot take median of empty seed result")
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def aggregate(
    config_path: Path, config: dict[str, Any], prereg_path: Path, prereg: dict[str, Any],
    *, pod1_result: Path, pod1_sha: str, pod2_result: Path, pod2_sha: str,
    output_dir: Path,
) -> Path:
    config_for_results = dict(config)
    config_for_results["_prereg_arms"] = prereg["arms"]
    pod1 = validate_pod_result(
        pod1_result, pod1_sha, config_for_results, sha256_file(prereg_path), pod="pod1"
    )
    pod2 = validate_pod_result(
        pod2_result, pod2_sha, config_for_results, sha256_file(prereg_path), pod="pod2"
    )
    shared_fields = (
        "schedule_sha256", "question_id_order", "mjcf_sha256", "execution_contract_sha256",
        "ready_state_sha256",
    )
    arms = {**pod1["arms"], **pod2["arms"]}
    if list(sorted(arms, key=lambda value: int(value[4:]))) != list(SEED_ORDER):
        raise ContractError("distributed results do not cover seed1..seed4 exactly")
    reference = arms["seed1"]
    for seed, arm in arms.items():
        for key in shared_fields:
            if arm[key] != reference[key]:
                raise ContractError(f"{seed} differs on shared paper/runtime field {key}")
    aggregate_rates = {
        seed: arm["returned_counts"]["aggregate"] / 100.0 for seed, arm in arms.items()
    }
    side_rates = {
        seed: {
            "forehand": arm["returned_counts"]["forehand"] / 50.0,
            "backhand": arm["returned_counts"]["backhand"] / 50.0,
        }
        for seed, arm in arms.items()
    }
    values = list(aggregate_rates.values())
    median = _median(values)
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    min_side = min(value for sides in side_rates.values() for value in sides.values())
    checks = {
        "aggregate_rate_median": {
            "observed": median,
            "required_min": EXPECTED_GATE_RULE["aggregate_rate_median_min"],
            "pass": median >= EXPECTED_GATE_RULE["aggregate_rate_median_min"],
        },
        "aggregate_rate_min_seed": {
            "observed": minimum,
            "required_min": EXPECTED_GATE_RULE["aggregate_rate_min_seed_min"],
            "pass": minimum >= EXPECTED_GATE_RULE["aggregate_rate_min_seed_min"],
        },
        "aggregate_rate_max_minus_min": {
            "observed": spread,
            "required_max": EXPECTED_GATE_RULE["aggregate_rate_max_minus_min_max"],
            "pass": spread <= EXPECTED_GATE_RULE["aggregate_rate_max_minus_min_max"],
        },
        "every_seed_every_side_rate": {
            "observed_min": min_side,
            "required_min": EXPECTED_GATE_RULE["every_seed_every_side_rate_min"],
            "pass": min_side >= EXPECTED_GATE_RULE["every_seed_every_side_rate_min"],
        },
    }
    gate_pass = all(check["pass"] for check in checks.values())
    content = {
        "contract_id": config["contract_id"],
        "status": (
            "pass_seed_stability_checkpoint_evidence"
            if gate_pass
            else "fail_seed_stability_checkpoint_evidence"
        ),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "pod_results": {
            "pod1": {"path": str(pod1_result), "sha256": pod1_sha},
            "pod2": {"path": str(pod2_result), "sha256": pod2_sha},
        },
        **EXPECTED_SEMANTICS,
        "shared_paper": {
            "schedule_file_sha256": EXPECTED_SCHEDULE["file_sha256"],
            "schedule_sha256": reference["schedule_sha256"],
            "question_id_order_sha256": canonical_sha256(reference["question_id_order"]),
            "mjcf_sha256": reference["mjcf_sha256"],
            "execution_contract_sha256": reference["execution_contract_sha256"],
            "ready_state_sha256": reference["ready_state_sha256"],
            "attempts_per_seed": 100,
            "attempts_per_side_per_seed": 50,
        },
        "arms": arms,
        "aggregate_return_rates": aggregate_rates,
        "side_return_rates": side_rates,
        "gate_rule": EXPECTED_GATE_RULE,
        "gate_checks": checks,
        "gate_pass": gate_pass,
        "actions": {
            "training": "continue_all_arms_unmodified",
            "seed_stability_gate": "close" if gate_pass else "keep_open",
            "trainer_or_worker_signals": [],
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
            "cross_instrument_plant_recovery_gates": "remain_open",
        },
    }
    document = _content_document("phase1_fresh_sz_seed_stability_q50_aggregate", content)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"phase1_fresh_SZ_model2000_seed_stability_q50_{document['content_sha256']}.json"
    rendered = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise ContractError(f"content-addressed output collision: {output}")
    else:
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(rendered, encoding="utf-8")
        os.replace(temp, output)
    print(f"[seed-stability-q50] aggregate={output} gate_pass={str(gate_pass).lower()}")
    print("[seed-stability-q50] all arms continue; no stop/promotion/deploy/real-robot authority")
    return output


def _load_bound_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    config_path = args.config.resolve()
    prereg_path = args.preregistration.resolve()
    require_sha("expected config SHA", args.expected_config_sha256)
    if not config_path.is_file() or sha256_file(config_path) != args.expected_config_sha256:
        raise ContractError("execution config file SHA mismatch")
    config = load_execution_config(config_path)
    if sha256_file(Path(__file__).resolve()) != config["tools"]["runner_sha256"]:
        raise ContractError("seed-stability runner bytes differ from execution config")
    if not prereg_path.is_file() or sha256_file(prereg_path) != config[
        "preregistration_sha256"
    ]:
        raise ContractError("preregistration file SHA mismatch")
    prereg = _strict_json(prereg_path)
    validate_preregistration(prereg, config)
    return config_path, config, prereg_path, prereg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--preregistration", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--pod", choices=tuple(POD_ARM_ORDER), required=True)
    check.add_argument("--schedule-source", required=True, type=Path)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--pod", choices=tuple(POD_ARM_ORDER), required=True)
    prepare_parser.add_argument("--schedule-source", required=True, type=Path)
    reuse_parser = sub.add_parser("reuse-check")
    reuse_parser.add_argument("--runtime-contract", required=True, type=Path)
    reuse_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--pod", choices=tuple(POD_ARM_ORDER), required=True)
    run_parser.add_argument("--runtime-contract", required=True, type=Path)
    run_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser.add_argument("--rerun-seed1", action="store_true")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--pod1-result", required=True, type=Path)
    aggregate_parser.add_argument("--pod1-result-sha256", required=True)
    aggregate_parser.add_argument("--pod2-result", required=True, type=Path)
    aggregate_parser.add_argument("--pod2-result-sha256", required=True)
    aggregate_parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config_path, config, prereg_path, prereg = _load_bound_inputs(args)
    if args.command == "contract-check":
        validate_runtime_inputs(
            config, prereg, pod=args.pod, schedule_path=args.schedule_source.resolve()
        )
        print(f"[seed-stability-q50] {args.pod} contract check PASS; no write/judge/signal")
        return 0
    if args.command == "prepare":
        return prepare(
            config_path,
            config,
            prereg_path,
            prereg,
            pod=args.pod,
            schedule_source=args.schedule_source.resolve(),
        )
    if args.command == "reuse-check":
        return check_seed1_reuse(
            config,
            prereg,
            args.runtime_contract.resolve(),
            args.expected_runtime_contract_sha256,
        )
    if args.command == "run":
        if args.rerun_seed1 and args.pod != "pod1":
            raise ContractError("--rerun-seed1 is valid only on pod1")
        return run_pod(
            config,
            prereg,
            args.runtime_contract.resolve(),
            args.expected_runtime_contract_sha256,
            pod=args.pod,
            rerun_seed1=args.rerun_seed1,
        )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(config["runtime"]["aggregate_output_dir"])
    )
    aggregate(
        config_path,
        config,
        prereg_path,
        prereg,
        pod1_result=args.pod1_result.resolve(),
        pod1_sha=args.pod1_result_sha256,
        pod2_result=args.pod2_result.resolve(),
        pod2_sha=args.pod2_result_sha256,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[seed-stability-q50][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
