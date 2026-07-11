#!/usr/bin/env python3
"""Prepare and explicitly run the fresh-SZ exact model_2000/model_4000 q50 pair.

This is a formal-execution-contract checkpoint-selection screen, not an arm-stopping or
promotion rule.  ``contract-check`` is read-only.  ``prepare`` materializes one shared schema-v3
K=100 paper but starts no judge.  ``run`` requires the prepared contract's SHA and preserves one
process-group-scoped state/log per checkpoint.  The causal M3 runner remains byte-for-byte frozen;
this wrapper imports only its content-addressed strict JSON and schedule primitives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_CAUSAL_RUNNER_PATH = SCRIPT_DIR / "run_phase1_paired_bank_q50.py"
SHARED_CAUSAL_RUNNER_SHA256 = "095e476fd36fb68d500cb39ea7f71f6fee9b729209187d51599582c72c22198b"


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if (
    not SHARED_CAUSAL_RUNNER_PATH.is_file()
    or _bootstrap_sha256(SHARED_CAUSAL_RUNNER_PATH) != SHARED_CAUSAL_RUNNER_SHA256
):
    raise RuntimeError(
        "refusing to import unbound run_phase1_paired_bank_q50.py; deploy the preregistered "
        "095e476f... dependency beside this wrapper"
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_phase1_paired_bank_q50 as base  # noqa: E402


ContractError = base.ContractError
sha256_file = base.sha256_file
canonical_bytes = base.canonical_bytes
canonical_sha256 = base.canonical_sha256
atomic_json = base.atomic_json
load_json = base.load_json
exact_keys = base.exact_keys
require_sha = base.require_sha
require_absolute = base.require_absolute
require_under = base.require_under

ARM_ORDER = ("model_2000", "model_4000")
RUN_NAME = "phase1_fresh_v3_S1_seed1"
ITERATIONS = {"model_2000": 2000, "model_4000": 4000}
EXPECTED_SEMANTICS = {
    "fresh_lineage": True,
    "evaluation_contract_exact": True,
    "formal_target": True,
    "checkpoint_selection_screen": True,
    "whole_arm_stop_allowed": False,
    "whole_arm_promote_allowed": False,
    "deploy_gate": False,
}
EXPECTED_SCHEDULE = {
    "schema_version": 3,
    "per_clip_quota": 50,
    "schedule_k": 100,
    "attempts_per_side": 50,
    "schedule_seed": 0,
    "hold_range": [0, 100],
    "noise_scales": [0.0],
    "one_question_reset": True,
    "no_wrap": True,
    "same_artifact_for_both_checkpoints": True,
    "allow_inexact_contract": False,
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


def load_execution_config(path: Path) -> dict[str, Any]:
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
            "tools",
            "schedule",
            "runtime",
            "arm_order",
            "selection_policy",
        },
        "execution config",
    )
    if data["schema_version"] != 1:
        raise ContractError("execution config schema_version must be 1")
    if data["status"] != "offline_preregistered_not_prepared" or data["auto_start"] is not False:
        raise ContractError("execution config must remain offline/not-prepared with auto_start=false")
    require_sha("preregistration_sha256", data["preregistration_sha256"])
    if data["semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("execution semantics are not the frozen fresh/formal selection screen")
    if data["schedule"] != EXPECTED_SCHEDULE:
        raise ContractError("execution schedule is not the frozen exact clean q50 paper")
    if data["arm_order"] != list(ARM_ORDER):
        raise ContractError(f"arm_order must be exactly {list(ARM_ORDER)!r}")
    if data["selection_policy"] != EXPECTED_SELECTION_POLICY:
        raise ContractError("checkpoint-selection policy changed")

    checkouts = data["checkouts"]
    exact_keys(checkouts, {"training", "evaluation"}, "checkouts")
    for name, spec in checkouts.items():
        exact_keys(spec, {"path", "commit"}, f"checkouts.{name}")
        require_absolute(f"checkouts.{name}.path", spec["path"])
        require_sha(f"checkouts.{name}.commit", spec["commit"], length=40)

    tools = data["tools"]
    exact_keys(tools, {"runner_sha256", "shared_causal_runner", "evaluation"}, "tools")
    require_sha("tools.runner_sha256", tools["runner_sha256"])
    exact_keys(tools["shared_causal_runner"], {"path", "sha256"}, "tools.shared_causal_runner")
    if tools["shared_causal_runner"]["path"] != "run_phase1_paired_bank_q50.py":
        raise ContractError("shared causal runner dependency path changed")
    require_sha("tools.shared_causal_runner.sha256", tools["shared_causal_runner"]["sha256"])
    expected_eval = {"judge", "materialize_schedule", "schedule_module", "mujoco_evaluator"}
    exact_keys(tools["evaluation"], expected_eval, "tools.evaluation")
    for name, spec in tools["evaluation"].items():
        exact_keys(spec, {"path", "sha256"}, f"tools.evaluation.{name}")
        path = spec["path"]
        if not isinstance(path, str) or os.path.isabs(path) or Path(path).parts[0] == "..":
            raise ContractError(f"tools.evaluation.{name}.path must be repo-relative")
        require_sha(f"tools.evaluation.{name}.sha256", spec["sha256"])

    runtime = data["runtime"]
    exact_keys(
        runtime,
        {
            "state_dir",
            "checkpoint_python",
            "schedule_filename",
            "runtime_contract_filename",
            "paired_result_filename",
        },
        "runtime",
    )
    require_absolute("runtime.state_dir", runtime["state_dir"])
    require_absolute("runtime.checkpoint_python", runtime["checkpoint_python"])
    for key in ("schedule_filename", "runtime_contract_filename", "paired_result_filename"):
        value = runtime[key]
        if not isinstance(value, str) or Path(value).name != value or not value.endswith(".json"):
            raise ContractError(f"runtime.{key} must be a simple .json filename")
    return data


def validate_preregistration(data: dict[str, Any], config: dict[str, Any]) -> None:
    exact_keys(
        data,
        {
            "schema_version",
            "preregistration_id",
            "created_utc",
            "status",
            "auto_activate",
            "jobs_started",
            "runtime_state",
            "source_trigger",
            "family",
            "comparison",
            "training_commit",
            "eval_commit",
            "eval_root",
            "tools",
            "paper",
            "arms",
            "selection_policy",
            "formal_semantics",
            "activation",
            "preflight_requirements",
        },
        "preregistration",
    )
    if data["schema_version"] != 1:
        raise ContractError("q50 preregistration schema_version must be 1")
    if (
        data["status"] != "preregistered_not_started"
        or data["auto_activate"] is not False
        or data["jobs_started"] != 0
        or data["runtime_state"] is not None
    ):
        raise ContractError("q50 preregistration is no longer pristine/not-started")
    if data["training_commit"] != config["checkouts"]["training"]["commit"]:
        raise ContractError("preregistration training commit disagrees with execution config")
    if data["eval_commit"] != config["checkouts"]["evaluation"]["commit"]:
        raise ContractError("preregistration eval commit disagrees with execution config")
    if data["eval_root"] != config["checkouts"]["evaluation"]["path"]:
        raise ContractError("preregistration eval root disagrees with execution config")
    if data["family"] != {
        "cell": "SZ",
        "name": "v4rg_runtime_order_v3",
        "source_family_sha256": "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5",
        "training_seed": 1,
        "run_name": RUN_NAME,
    }:
        raise ContractError("formal fresh-SZ family binding changed")

    tools = data["tools"]
    exact_keys(tools, {"runner_sha256", "shared_causal_runner_sha256", "judge_sha256"}, "tools")
    if tools["runner_sha256"] != config["tools"]["runner_sha256"]:
        raise ContractError("preregistration runner SHA disagrees with execution config")
    if tools["shared_causal_runner_sha256"] != config["tools"]["shared_causal_runner"]["sha256"]:
        raise ContractError("preregistration shared-runner SHA disagrees with execution config")
    if tools["judge_sha256"] != config["tools"]["evaluation"]["judge"]["sha256"]:
        raise ContractError("preregistration judge SHA disagrees with execution config")
    for key, value in tools.items():
        require_sha(f"tools.{key}", value)

    trigger = data["source_trigger"]
    exact_keys(
        trigger,
        {
            "kind",
            "screen_only",
            "stop_or_promote_allowed",
            "observed",
            "sources",
            "trigger_met",
        },
        "source_trigger",
    )
    if (
        trigger["kind"] != "same_run_q10_peak_to_later_regression"
        or trigger["screen_only"] is not True
        or trigger["stop_or_promote_allowed"] is not False
        or trigger["trigger_met"] is not True
    ):
        raise ContractError("q10 trigger policy changed")
    observed = trigger["observed"]
    exact_keys(observed, set(ARM_ORDER), "source_trigger.observed")
    if observed != {
        "model_2000": {"attempts": 20, "attempts_per_side": 10, "aggregate_return_rate": 0.9},
        "model_4000": {"attempts": 20, "attempts_per_side": 10, "aggregate_return_rate": 0.5},
    }:
        raise ContractError("q10 0.90->0.50 regression trigger changed")
    if not isinstance(trigger["sources"], list) or len(trigger["sources"]) < 2:
        raise ContractError("q10 trigger must bind at least two source artifacts")
    for index, source in enumerate(trigger["sources"]):
        exact_keys(source, {"role", "path", "sha256"}, f"source_trigger.sources[{index}]")
        if not isinstance(source["role"], str) or not source["role"]:
            raise ContractError("q10 source role must be non-empty")
        if not isinstance(source["path"], str) or not source["path"]:
            raise ContractError("q10 source path must be non-empty")
        require_sha(f"source_trigger.sources[{index}].sha256", source["sha256"])

    paper = data["paper"]
    schedule = config["schedule"]
    required_paper = {
        "role": "same-run paired fresh exact q50 checkpoint-selection screen",
        "seed": schedule["schedule_seed"],
        "noise_scales": schedule["noise_scales"],
        "schedule_k": schedule["schedule_k"],
        "attempts_per_side": schedule["attempts_per_side"],
        "hold_steps_range": schedule["hold_range"],
        "no_wrap": True,
        "one_question_reset": True,
        "same_immutable_schedule_required_for_both_checkpoints": True,
        "allow_inexact_contract_required": False,
        "expected_evaluation_contract_exact": True,
    }
    for key, expected in required_paper.items():
        if paper.get(key) != expected:
            raise ContractError(f"preregistration paper field {key} changed")
    materialization = paper.get("schedule_materialization", {})
    if materialization.get("status") != "not_materialized" or materialization.get("sha256") is not None:
        raise ContractError("preregistration already claims a schedule materialization")
    bank = paper.get("exam_bank", {})
    require_absolute("paper.exam_bank.path", bank.get("path"))
    require_sha("paper.exam_bank.sha256", bank.get("sha256"))
    if bank.get("bytes") != 63968 or bank.get("schema_version") != 3:
        raise ContractError("formal exam bank size/schema changed")
    if bank.get("source_family_sha256") != data["family"]["source_family_sha256"]:
        raise ContractError("exam bank source family differs from the trained family")
    require_sha("paper.mjcf_sha256", paper.get("mjcf_sha256"))

    if tuple(data["arms"]) != ARM_ORDER:
        raise ContractError(f"preregistration arms must be ordered exactly as {ARM_ORDER}")
    training_root = Path(config["checkouts"]["training"]["path"])
    common_run_dir = None
    common_contract_sha = None
    for name in ARM_ORDER:
        arm = data["arms"][name]
        expected_iteration = ITERATIONS[name]
        if (
            arm.get("checkpoint_iteration") != expected_iteration
            or arm.get("run_name") != RUN_NAME
            or arm.get("cell") != "SZ"
            or arm.get("training_seed") != 1
            or arm.get("lineage_exact") is not True
            or arm.get("job_status") != "not_started"
            or arm.get("pid") is not None
            or arm.get("pgid") is not None
            or arm.get("result") is not None
        ):
            raise ContractError(f"arm {name} is not the frozen fresh exact checkpoint")
        checkpoint = require_absolute(f"arms.{name}.checkpoint_path", arm.get("checkpoint_path"))
        hard_path = require_absolute(
            f"arms.{name}.training_contract_path", arm.get("training_contract_path")
        )
        require_under(f"arms.{name}.checkpoint_path", checkpoint, training_root)
        require_under(f"arms.{name}.training_contract_path", hard_path, training_root)
        if checkpoint.name != f"model_{expected_iteration}.pt" or not checkpoint.parent.name.endswith(
            RUN_NAME
        ):
            raise ContractError(f"arm {name} checkpoint path/iteration changed")
        if hard_path != checkpoint.parent / "params" / "training_contract.json":
            raise ContractError(f"arm {name} hard contract is not checkpoint-adjacent")
        require_sha(f"arms.{name}.checkpoint_sha256", arm.get("checkpoint_sha256"))
        hard_sha = require_sha(
            f"arms.{name}.training_contract_sha256", arm.get("training_contract_sha256")
        )
        if arm.get("checkpoint_embedded_training_contract_sha256") != hard_sha:
            raise ContractError(f"arm {name} loses checkpoint/hard-contract binding")
        if arm.get("face_command_pairing") != "shared_plus_y" or arm.get("zero_joint_friction") is not True:
            raise ContractError(f"arm {name} is no longer the SZ cell")
        common_run_dir = checkpoint.parent if common_run_dir is None else common_run_dir
        common_contract_sha = hard_sha if common_contract_sha is None else common_contract_sha
        if checkpoint.parent != common_run_dir or hard_sha != common_contract_sha:
            raise ContractError("paired checkpoints must share one run and one hard contract")

    if data["selection_policy"] != EXPECTED_SELECTION_POLICY:
        raise ContractError("preregistered checkpoint-selection policy changed")
    if data["formal_semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("preregistration is not the exact formal-target screen")
    activation = data["activation"]
    if activation != {
        "preregistered": True,
        "authorized_to_start_by_this_file": False,
        "started_at_creation": False,
    }:
        raise ContractError("preregistration activation fence changed")


def validate_checkout(name: str, spec: dict[str, str]) -> Path:
    return base.validate_checkout(name, spec)


def validate_eval_tools(config: dict[str, Any], eval_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["tools"]["evaluation"].items():
        path = (eval_root / spec["path"]).resolve()
        require_under(f"tools.evaluation.{name}", path, eval_root)
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ContractError(f"evaluation tool bytes changed: {name} {path}")
        paths[name] = path
    shared = SCRIPT_DIR / config["tools"]["shared_causal_runner"]["path"]
    if not shared.is_file() or sha256_file(shared) != config["tools"]["shared_causal_runner"]["sha256"]:
        raise ContractError("shared causal runner dependency bytes changed")
    return paths


def validate_trigger_sources(prereg: dict[str, Any]) -> None:
    """Re-hash preserved q10 evidence before the q50 paper is materialized.

    The embedded schedule hashes are semantic identifiers rather than files; every absolute source
    is a preserved byte artifact and must still exist.  q10 remains trigger-only even when all
    source bytes pass this check.
    """

    for index, source in enumerate(prereg["source_trigger"]["sources"]):
        raw_path = source["path"]
        if not os.path.isabs(raw_path):
            continue
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ContractError(
                f"q10 trigger source bytes changed/missing at index {index}: {path}"
            )


CHECKPOINT_AUDIT_CODE = r"""
import json, sys, torch
p=sys.argv[1]
o=torch.load(p,map_location='cpu',weights_only=False)
infos=o.get('infos') if isinstance(o,dict) else None
if not isinstance(infos,dict): infos={}
bad=0; floating=0; tensors=0; elements=0
def walk(v):
 global bad,floating,tensors,elements
 if torch.is_tensor(v):
  tensors+=1
  if v.is_floating_point() or v.is_complex():
   floating+=1; elements+=v.numel(); bad+=int((~torch.isfinite(v)).sum().item())
 elif isinstance(v,dict):
  for x in v.values(): walk(x)
 elif isinstance(v,(list,tuple)):
  for x in v: walk(x)
walk(o)
print(json.dumps({'iter':o.get('iter') if isinstance(o,dict) else None,
 'training_contract_sha256':infos.get('training_contract_sha256'),
 'training_contract_schema_version':infos.get('training_contract_schema_version'),
 'training_contract_lineage_exact':infos.get('training_contract_lineage_exact'),
 'tensor_count':tensors,'floating_tensor_count':floating,'floating_elements':elements,
 'nonfinite':bad},sort_keys=True))
"""


def checkpoint_audit(
    python: Path, checkpoint: Path, *, expected_iteration: int, expected_contract_sha: str
) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_AUDIT_CODE, str(checkpoint)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"checkpoint audit failed for {checkpoint}: {exc.output}") from exc
    try:
        audit = load_json_from_text(output.strip().splitlines()[-1], "checkpoint audit")
    except IndexError as exc:
        raise ContractError(f"checkpoint audit returned no JSON: {output!r}") from exc
    if (
        audit.get("iter") != expected_iteration
        or audit.get("training_contract_sha256") != expected_contract_sha
        or audit.get("training_contract_schema_version") != 3
        or audit.get("training_contract_lineage_exact") not in (1, True)
        or not isinstance(audit.get("tensor_count"), int)
        or audit["tensor_count"] <= 0
        or not isinstance(audit.get("floating_elements"), int)
        or audit["floating_elements"] <= 0
        or audit.get("nonfinite") != 0
    ):
        raise ContractError(f"checkpoint audit violates fresh exact contract: {audit}")
    return audit


def load_json_from_text(text: str, owner: str) -> Any:
    import json

    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {owner}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ContractError(f"non-finite JSON constant {value!r} in {owner}")

    try:
        return json.loads(text, object_pairs_hook=reject_duplicate, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ContractError(f"cannot parse strict JSON from {owner}: {exc}") from exc


def validate_hard_contract(hard: dict[str, Any], prereg: dict[str, Any]) -> None:
    friction = hard.get("joint_friction_coefficients")
    bank = hard.get("question_bank")
    expected_bank = {
        "sha256": "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700",
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": prereg["family"]["source_family_sha256"],
        "exact": True,
    }
    if (
        hard.get("schema_version") != 3
        or hard.get("motion_kinematics_exact") is not True
        or hard.get("face_command_pairing") != "shared_plus_y"
        or not isinstance(friction, list)
        or len(friction) != 31
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in friction)
        or any(not math.isfinite(float(value)) or float(value) != 0.0 for value in friction)
        or bank != expected_bank
    ):
        raise ContractError("hard contract is not the frozen fresh SZ exact family")


def validate_runtime_inputs(
    config: dict[str, Any], prereg: dict[str, Any]
) -> tuple[Path, Path, dict[str, Path], dict[str, Any]]:
    training_root = validate_checkout("training", config["checkouts"]["training"])
    eval_root = validate_checkout("evaluation", config["checkouts"]["evaluation"])
    tools = validate_eval_tools(config, eval_root)
    validate_trigger_sources(prereg)
    checkpoint_python = Path(config["runtime"]["checkpoint_python"])
    if not checkpoint_python.is_file() or not os.access(checkpoint_python, os.X_OK):
        raise ContractError(f"checkpoint Python is missing/not executable: {checkpoint_python}")
    exam_bank = prereg["paper"]["exam_bank"]
    bank_path = Path(exam_bank["path"])
    if (
        not bank_path.is_file()
        or bank_path.stat().st_size != exam_bank["bytes"]
        or sha256_file(bank_path) != exam_bank["sha256"]
    ):
        raise ContractError("exam-bank bytes no longer match preregistration")

    audits = {}
    hard_seen = None
    for name in ARM_ORDER:
        arm = prereg["arms"][name]
        checkpoint = Path(arm["checkpoint_path"])
        hard_path = Path(arm["training_contract_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != arm["checkpoint_sha256"]:
            raise ContractError(f"{name} checkpoint bytes changed")
        if not hard_path.is_file() or sha256_file(hard_path) != arm["training_contract_sha256"]:
            raise ContractError(f"{name} hard-contract bytes changed")
        hard = load_json(hard_path)
        validate_hard_contract(hard, prereg)
        if hard_seen is None:
            hard_seen = hard
        elif hard != hard_seen:
            raise ContractError("paired checkpoints no longer share byte-identical hard semantics")
        audits[name] = checkpoint_audit(
            checkpoint_python,
            checkpoint,
            expected_iteration=ITERATIONS[name],
            expected_contract_sha=arm["training_contract_sha256"],
        )
    return training_root, eval_root, tools, audits


def validate_schedule_document(path: Path, *, expected_bank_sha256: str) -> dict[str, Any]:
    return base.validate_schedule_document(path, expected_bank_sha256=expected_bank_sha256)


def semantic_judge_args(schedule_path: Path) -> list[str]:
    if "\n" in str(schedule_path):
        raise ContractError("schedule path contains a newline")
    exam_extra = "--exam-schedule-json " + shlex.quote(str(schedule_path))
    if "--allow-inexact-contract" in exam_extra:
        raise ContractError("formal exact q50 must never request the diagnostic escape")
    return [
        "--seed",
        "0",
        "--noise-scales",
        "0.0",
        "--steps",
        "0",
        "--hold-ref",
        "auto",
        "--exam-extra",
        exam_extra,
    ]


def build_judge_command(
    *, judge: Path, arm: dict[str, Any], schedule_path: Path, gpu: int
) -> list[str]:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ContractError("GPU must be a non-negative integer")
    command = [
        "bash",
        str(judge),
        str(Path(arm["checkpoint_path"]).parent),
        arm["checkpoint_path"],
        "--gpu",
        str(gpu),
        *semantic_judge_args(schedule_path),
    ]
    if any("--allow-inexact-contract" in value for value in command):
        raise ContractError("formal exact q50 command contains diagnostic escape")
    return command


def prepare(
    config_path: Path,
    config: dict[str, Any],
    prereg_path: Path,
    prereg: dict[str, Any],
) -> int:
    _, _, tools, arm_audits = validate_runtime_inputs(config, prereg)
    runtime = config["runtime"]
    state_dir = Path(runtime["state_dir"])
    if state_dir.exists():
        raise ContractError(f"no-clobber: q50 state directory already exists: {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = state_dir / runtime["schedule_filename"]
    contract_path = state_dir / runtime["runtime_contract_filename"]
    bank = prereg["paper"]["exam_bank"]
    command = [
        runtime["checkpoint_python"],
        str(tools["materialize_schedule"]),
        "--exam-bank",
        bank["path"],
        "--per-clip-quota",
        "50",
        "--schedule-seed",
        "0",
        "--hold-range",
        "0",
        "100",
        "--output",
        str(schedule_path),
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode != 0:
        raise ContractError(
            f"schedule materialization failed rc={completed.returncode}: {completed.stdout}"
        )
    schedule = validate_schedule_document(schedule_path, expected_bank_sha256=bank["sha256"])
    runtime_contract = {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "status": "prepared_not_started",
        "auto_start": False,
        "jobs_started": 0,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **EXPECTED_SEMANTICS,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "checkouts": config["checkouts"],
        "tools": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "shared_causal_runner": config["tools"]["shared_causal_runner"],
            "evaluation": config["tools"]["evaluation"],
        },
        "exam_bank": bank,
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "schema_version": 3,
            "schedule_k": 100,
            "attempts_per_side": 50,
            "seed": 0,
            "hold_range": [0, 100],
            "question_id_order": [item["question_id"] for item in schedule["items"]],
        },
        "semantic_judge_args": semantic_judge_args(schedule_path),
        "arm_order": list(ARM_ORDER),
        "arms": {
            name: {
                "run_name": prereg["arms"][name]["run_name"],
                "checkpoint_iteration": ITERATIONS[name],
                "checkpoint_path": prereg["arms"][name]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
                "training_contract_path": prereg["arms"][name]["training_contract_path"],
                "training_contract_sha256": prereg["arms"][name]["training_contract_sha256"],
                "checkpoint_audit": arm_audits[name],
                "job_status": "not_started",
            }
            for name in ARM_ORDER
        },
        "selection_policy": EXPECTED_SELECTION_POLICY,
        "paired_result_path": str(state_dir / runtime["paired_result_filename"]),
    }
    atomic_json(contract_path, runtime_contract)
    print("[fresh-exact-paired-q50] prepared only; no judges started")
    print(
        f"[fresh-exact-paired-q50] schedule={schedule_path} "
        f"semantic_sha256={schedule['schedule_sha256']} file_sha256={sha256_file(schedule_path)}"
    )
    print(
        f"[fresh-exact-paired-q50] runtime_contract={contract_path} "
        f"sha256={sha256_file(contract_path)}"
    )
    return 0


def validate_runtime_contract(
    path: Path,
    expected_sha: str,
    config: dict[str, Any],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    require_sha("expected runtime-contract SHA", expected_sha)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ContractError("runtime-contract file SHA mismatch")
    contract = load_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_id") != config["contract_id"]
        or contract.get("status") != "prepared_not_started"
        or contract.get("auto_start") is not False
        or contract.get("jobs_started") != 0
        or contract.get("arm_order") != list(ARM_ORDER)
        or contract.get("selection_policy") != EXPECTED_SELECTION_POLICY
        or any(contract.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError("runtime contract is not a pristine fresh/formal prepared pair")
    config_meta = contract.get("config", {})
    prereg_meta = contract.get("preregistration", {})
    if (
        config_meta.get("path") != str(Path(config_meta.get("path", "")).resolve())
        or not Path(config_meta["path"]).is_file()
        or config_meta.get("sha256") != sha256_file(Path(config_meta["path"]))
    ):
        raise ContractError("runtime contract's execution config changed")
    if (
        prereg_meta.get("path") != str(Path(prereg_meta.get("path", "")).resolve())
        or not Path(prereg_meta["path"]).is_file()
        or prereg_meta.get("sha256") != sha256_file(Path(prereg_meta["path"]))
    ):
        raise ContractError("runtime contract's preregistration changed")
    schedule_meta = contract.get("shared_schedule", {})
    schedule_path = Path(schedule_meta.get("path", ""))
    if not schedule_path.is_file() or sha256_file(schedule_path) != schedule_meta.get("file_sha256"):
        raise ContractError("shared schedule file changed after preparation")
    schedule = validate_schedule_document(
        schedule_path, expected_bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    if (
        schedule["schedule_sha256"] != schedule_meta.get("schedule_sha256")
        or [item["question_id"] for item in schedule["items"]]
        != schedule_meta.get("question_id_order")
        or contract.get("semantic_judge_args") != semantic_judge_args(schedule_path)
    ):
        raise ContractError("runtime contract no longer binds the shared schedule exactly")
    return contract


def _strict_bool(value: Any, owner: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", "False", "false", "0"):
        return str(value).lower() in ("true", "1")
    raise ContractError(f"{owner} must be an exact boolean, got {value!r}")


def _finite_rate(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{owner} must be a finite numeric rate")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractError(f"{owner} is not a finite [0,1] rate: {value!r}")
    return result


def _integer(value: Any, owner: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{owner} must be an integer >= {minimum}")
    return value


def validate_exam_result(
    *,
    report: Path,
    arm_name: str,
    arm: dict[str, Any],
    prereg: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    expected_iteration = ITERATIONS[arm_name]
    match = re.fullmatch(r"judge_report_(model_\d+)_(\d{8}_\d{6})\.md", report.name)
    if not match or match.group(1) != f"model_{expected_iteration}":
        raise ContractError(f"unexpected judge report name for {arm_name}: {report}")
    expected_report_parent = Path(arm["checkpoint_path"]).parent / "judge"
    if report.parent.resolve() != expected_report_parent.resolve():
        raise ContractError(
            f"judge report escaped the checkpoint run: {report.parent} != {expected_report_parent}"
        )
    judge_dir = report.parent / f"{match.group(1)}_{match.group(2)}"
    summary_path = judge_dir / "exam" / "mujoco_sim2sim_summary.json"
    attempts_path = judge_dir / "exam" / "mujoco_sim2sim_attempts.csv"
    if not summary_path.is_file() or not attempts_path.is_file():
        raise ContractError("judge completed without summary/attempt ledger")
    summary = load_json(summary_path)
    schedule_meta = runtime_contract["shared_schedule"]
    bank = prereg["paper"]["exam_bank"]
    schedule_document = validate_schedule_document(
        Path(schedule_meta["path"]), expected_bank_sha256=bank["sha256"]
    )
    expected_items = schedule_document["items"]
    arguments = summary.get("arguments", {})
    input_artifacts = summary.get("input_artifacts", {})
    schedule_artifact = input_artifacts.get("exam_schedule_artifact", {})
    exam_schedule = summary.get("exam_schedule", {})
    results = summary.get("results")
    result = results[0] if isinstance(results, list) and len(results) == 1 else {}
    summary_items = exam_schedule.get("items", [])
    result_items = result.get("exam_schedule", {}).get("items", [])
    keys = tuple(expected_items[0])
    summary_projection = (
        [{key: item.get(key) for key in keys} for item in summary_items]
        if isinstance(summary_items, list) and all(isinstance(item, dict) for item in summary_items)
        else []
    )
    result_projection = (
        [{key: item.get(key) for key in keys} for item in result_items]
        if isinstance(result_items, list) and all(isinstance(item, dict) for item in result_items)
        else []
    )
    runtime_items_exact = bool(result_items) and all(
        item.get("eligible") is True
        and item.get("censored") is False
        and item.get("ready_state_mode") == "mjcf_named_keyframe:stand:v1"
        and item.get("ready_state_sha256") == summary.get("ready_state_sha256")
        and item.get("mjcf_sha256") == summary.get("mjcf_sha256")
        and item.get("execution_contract_sha256") == summary.get("execution_contract_sha256")
        and isinstance(item.get("physical_fall"), bool)
        and isinstance(item.get("guard_reset"), bool)
        and isinstance(item.get("hit"), bool)
        and isinstance(item.get("returned"), bool)
        and isinstance(item.get("finalize_reason"), str)
        and bool(item.get("finalize_reason"))
        and item.get("question_sequence_index") == item.get("schedule_index")
        for item in result_items
    )
    execution_contract = summary.get("execution_contract", {})
    execution_payload = dict(execution_contract) if isinstance(execution_contract, dict) else {}
    declared_execution_sha = execution_payload.pop("sha256", None)
    velocity = summary.get("joint_velocity_limit_diagnostics", {})
    if (
        summary.get("schema_version") != 3
        or summary.get("evaluation_contract_exact") is not True
        or arguments.get("allow_inexact_contract") is not False
        or arguments.get("target_source") != "bank"
        or arguments.get("seed") != 0
        or arguments.get("noise_scales") != [0.0]
        or arguments.get("qdes_clamp") is not True
        or arguments.get("hold_ref") != "auto"
        or arguments.get("ready_state") != "auto"
        or arguments.get("exam_continuity_diagnostic") is not False
        or arguments.get("exam_schedule_k") is not None
        or os.path.abspath(str(arguments.get("exam_schedule_json"))) != schedule_meta["path"]
        or input_artifacts.get("exam_bank", {}).get("sha256") != bank["sha256"]
        or schedule_artifact.get("sha256") != schedule_meta["file_sha256"]
        or schedule_artifact.get("schedule_sha256") != schedule_meta["schedule_sha256"]
        or schedule_artifact.get("schema_version") != 3
        or exam_schedule.get("sha256") != schedule_meta["schedule_sha256"]
        or exam_schedule.get("bank_sha256") != bank["sha256"]
        or exam_schedule.get("seed") != 0
        or exam_schedule.get("size") != 100
        or exam_schedule.get("one_question_reset") is not True
        or exam_schedule.get("ready_state_mode") != "mjcf_named_keyframe:stand:v1"
        or exam_schedule.get("shared_artifact") != schedule_document
        or summary_projection != expected_items
        or not all(
            item.get("question_sequence_index") == item.get("schedule_index")
            for item in summary_items
        )
        or result_projection != expected_items
        or not runtime_items_exact
        or result.get("evaluation_contract_exact") is not True
        or result.get("noise_scale") != 0.0
        or result.get("exam_schedule", {}).get("question_id_order")
        != schedule_meta["question_id_order"]
        or summary.get("mjcf_sha256") != prereg["paper"]["mjcf_sha256"]
        or declared_execution_sha != summary.get("execution_contract_sha256")
        or not isinstance(declared_execution_sha, str)
        or canonical_sha256(execution_payload) != declared_execution_sha
        or velocity.get("hit_count") != 0
        or velocity.get("proxy_clamp_applied") is not False
    ):
        raise ContractError("q50 result does not reproduce the frozen fresh exact paper")

    with attempts_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ledger_items = []
    ledger_runtime = []
    try:
        for index, row in enumerate(rows):
            clip_name = row["clip_name"]
            if clip_name not in ("forehand", "backhand"):
                raise ValueError(f"unknown clip_name={clip_name!r}")
            ledger_items.append(
                {
                    "schedule_index": int(row["schedule_index"]),
                    "clip": 0 if clip_name == "forehand" else 1,
                    "bank_row": int(row["bank_row"]),
                    "question_id": row["question_id"],
                    "repeat": int(row["repeat"]),
                    "hold_steps": int(row["hold_steps"]),
                    "attempt_seed": int(row["attempt_seed"]),
                }
            )
            ledger_runtime.append(
                {
                    "question_sequence_index": int(row["question_sequence_index"]),
                    "ready_state_mode": row["ready_state_mode"],
                    "ready_state_sha256": row["ready_state_sha256"],
                    "mjcf_sha256": row["mjcf_sha256"],
                    "execution_contract_sha256": row["execution_contract_sha256"],
                    "eligible": _strict_bool(row["eligible"], f"ledger[{index}].eligible"),
                    "censored": _strict_bool(row["censored"], f"ledger[{index}].censored"),
                    "physical_fall": _strict_bool(
                        row["physical_fall"], f"ledger[{index}].physical_fall"
                    ),
                    "guard_reset": _strict_bool(row["guard_reset"], f"ledger[{index}].guard_reset"),
                    "hit": _strict_bool(row["hit"], f"ledger[{index}].hit"),
                    "returned": _strict_bool(row["returned"], f"ledger[{index}].returned"),
                    "reached_exact": _strict_bool(
                        row["reached_exact"], f"ledger[{index}].reached_exact"
                    ),
                    "exact_composite": _strict_bool(
                        row["exact_composite"], f"ledger[{index}].exact_composite"
                    ),
                    "finalize_reason": row["finalize_reason"],
                }
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"q50 attempt ledger lacks exact raw fields: {exc}") from exc
    if (
        len(rows) != 100
        or ledger_items != expected_items
        or any(row.get("schedule_sha256") != schedule_meta["schedule_sha256"] for row in rows)
        or any(
            not item["eligible"]
            or item["censored"]
            or item["question_sequence_index"] != index
            or item["ready_state_mode"] != "mjcf_named_keyframe:stand:v1"
            or item["ready_state_sha256"] != summary["ready_state_sha256"]
            or item["mjcf_sha256"] != summary["mjcf_sha256"]
            or item["execution_contract_sha256"] != summary["execution_contract_sha256"]
            or not item["finalize_reason"]
            for index, item in enumerate(ledger_runtime)
        )
    ):
        raise ContractError("q50 attempt ledger is incomplete, reordered, censored, or re-papered")
    result_raw_keys = (
        "eligible",
        "censored",
        "physical_fall",
        "guard_reset",
        "hit",
        "returned",
        "finalize_reason",
    )
    for index, (item, raw) in enumerate(zip(result_items, ledger_runtime)):
        if any(item.get(key) != raw[key] for key in result_raw_keys):
            raise ContractError(f"summary/CSV raw attempt disagreement at row {index}")

    attempts = result.get("attempts", {})
    per_clip = attempts.get("per_clip", {})
    venue = result.get("venue", {})
    grouped_raw = {
        "all": ledger_runtime,
        "forehand": [
            raw for row, raw in zip(rows, ledger_runtime) if row["clip_name"] == "forehand"
        ],
        "backhand": [
            raw for row, raw in zip(rows, ledger_runtime) if row["clip_name"] == "backhand"
        ],
    }
    attempt_groups = {
        "all": attempts,
        "forehand": per_clip.get("forehand", {}),
        "backhand": per_clip.get("backhand", {}),
    }
    expected_denominators = {"all": 100, "forehand": 50, "backhand": 50}
    for owner, denominator in expected_denominators.items():
        raw_group = grouped_raw[owner]
        attempt_group = attempt_groups[owner]
        reached = sum(item["reached_exact"] for item in raw_group)
        composite = sum(item["exact_composite"] for item in raw_group)
        contacted = sum(item["hit"] for item in raw_group)
        returned = sum(item["returned"] for item in raw_group)
        if (
            len(raw_group) != denominator
            or attempt_group.get("n_attempts") != denominator
            or attempt_group.get("n_reached_exact") != reached
            or attempt_group.get("n_composite") != composite
            or not math.isclose(
                _finite_rate(
                    attempt_group.get("exact_reach_rate"),
                    f"attempts.{owner}.exact_reach_rate",
                ),
                reached / denominator,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                _finite_rate(
                    attempt_group.get("composite_rate_per_attempt"),
                    f"attempts.{owner}.composite_rate_per_attempt",
                ),
                composite / denominator,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ContractError(f"attempts.{owner} disagrees with its raw denominator ledger")
        metrics = venue.get(owner, {})
        if metrics.get("n_attempts") != denominator:
            raise ContractError(f"venue.{owner} denominator is not {denominator}")
        for key in (
            "exact_reach_rate_per_attempt",
            "contact_rate_per_attempt",
            "return_success_rate_per_attempt",
        ):
            _finite_rate(metrics.get(key), f"venue.{owner}.{key}")
        if (
            _integer(metrics.get("n_strikes"), f"venue.{owner}.n_strikes") != reached
            or _integer(metrics.get("contacted"), f"venue.{owner}.contacted") != contacted
            or _integer(metrics.get("landed_ok"), f"venue.{owner}.landed_ok") != returned
        ):
            raise ContractError(f"venue.{owner} raw counts disagree with the attempt ledger")

    counts = {
        "aggregate": sum(item["returned"] for item in ledger_runtime),
        "forehand": sum(
            item["returned"] for row, item in zip(rows, ledger_runtime)
            if row["clip_name"] == "forehand"
        ),
        "backhand": sum(
            item["returned"] for row, item in zip(rows, ledger_runtime)
            if row["clip_name"] == "backhand"
        ),
        "physical_falls": sum(item["physical_fall"] for item in ledger_runtime),
    }
    if (
        venue["all"]["landed_ok"] != counts["aggregate"]
        or venue["forehand"]["landed_ok"] != counts["forehand"]
        or venue["backhand"]["landed_ok"] != counts["backhand"]
        or not math.isclose(
            venue["all"]["return_success_rate_per_attempt"],
            counts["aggregate"] / 100,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            venue["forehand"]["return_success_rate_per_attempt"],
            counts["forehand"] / 50,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            venue["backhand"]["return_success_rate_per_attempt"],
            counts["backhand"] / 50,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ContractError("headline result does not equal raw all-attempt counts")
    return {
        "run_name": arm["run_name"],
        "checkpoint_iteration": expected_iteration,
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "training_contract_sha256": arm["training_contract_sha256"],
        "report": {"path": str(report), "sha256": sha256_file(report)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "attempt_ledger": {"path": str(attempts_path), "sha256": sha256_file(attempts_path)},
        "schedule_sha256": schedule_meta["schedule_sha256"],
        "question_id_order": schedule_meta["question_id_order"],
        "mjcf_sha256": summary["mjcf_sha256"],
        "execution_contract_sha256": summary["execution_contract_sha256"],
        "ready_state_sha256": summary["ready_state_sha256"],
        "evaluation_contract_exact": True,
        "formal_target": True,
        "fresh_lineage": True,
        "denominators": {"aggregate": 100, "forehand": 50, "backhand": 50},
        "returned_counts": counts,
        "returned_rates": {
            "aggregate": counts["aggregate"] / 100,
            "forehand": counts["forehand"] / 50,
            "backhand": counts["backhand"] / 50,
        },
        "raw_result": result,
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
                "min_side_return_count": min(
                    results[name]["returned_counts"]["forehand"],
                    results[name]["returned_counts"]["backhand"],
                ),
                "physical_falls": results[name]["returned_counts"]["physical_falls"],
                "earlier_checkpoint": name == "model_2000",
            }
            for name in ARM_ORDER
        },
        "scope": EXPECTED_SELECTION_POLICY["scope"],
        "whole_arm_action": "continue_unmodified",
        "whole_arm_stop_allowed": False,
        "whole_arm_promote_allowed": False,
    }


def run_pair(
    config: dict[str, Any],
    prereg: dict[str, Any],
    runtime_path: Path,
    runtime_sha: str,
    gpus: Sequence[int],
) -> int:
    contract = validate_runtime_contract(runtime_path, runtime_sha, config, prereg)
    _, _, tools, _ = validate_runtime_inputs(config, prereg)
    if len(gpus) != 2:
        raise ContractError("run requires exactly two GPUs in model_2000, model_4000 order")
    state_dir = runtime_path.parent
    pair_result_path = Path(contract["paired_result_path"])
    if pair_result_path.exists():
        raise ContractError(f"no-clobber: paired result already exists: {pair_result_path}")
    schedule_path = Path(contract["shared_schedule"]["path"])
    results = {}
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    for index, name in enumerate(ARM_ORDER):
        state_path = state_dir / f"{name}.state.json"
        log_path = state_dir / f"{name}.runner.log"
        if state_path.exists() or log_path.exists():
            raise ContractError(f"no-clobber: preserved state/log already exists for {name}")
        arm = prereg["arms"][name]
        command = build_judge_command(
            judge=tools["judge"], arm=arm, schedule_path=schedule_path, gpu=int(gpus[index])
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
                "arm": name,
                "status": "running",
                "pid": proc.pid,
                "pgid": proc.pid,
                "command": command,
                "runtime_contract_sha256": runtime_sha,
                "schedule_sha256": contract["shared_schedule"]["schedule_sha256"],
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
            raise ContractError(f"judge {name} failed rc={rc}; preserved {log_path}")
        report = base.find_report(log_path.read_text(encoding="utf-8", errors="replace"))
        results[name] = validate_exam_result(
            report=report,
            arm_name=name,
            arm=arm,
            prereg=prereg,
            runtime_contract=contract,
        )

    first, second = (results[name] for name in ARM_ORDER)
    for key in (
        "schedule_sha256",
        "question_id_order",
        "mjcf_sha256",
        "execution_contract_sha256",
        "ready_state_sha256",
    ):
        if first[key] != second[key]:
            raise ContractError(f"paired q50 checkpoints disagree on shared runtime field {key}")
    selected_name, selection = select_checkpoint(results)
    pair_result = {
        "schema_version": 1,
        "pair_id": config["contract_id"],
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_contract": {"path": str(runtime_path), "sha256": runtime_sha},
        **EXPECTED_SEMANTICS,
        "shared_schedule_sha256": first["schedule_sha256"],
        "question_id_order": first["question_id_order"],
        "arms": results,
        "selection_policy": EXPECTED_SELECTION_POLICY,
        "selection": selection,
        "selected_checkpoint": {
            "arm": selected_name,
            "iteration": ITERATIONS[selected_name],
            "sha256": results[selected_name]["checkpoint_sha256"],
        },
    }
    atomic_json(pair_result_path, pair_result)
    print(
        f"[fresh-exact-paired-q50] complete; selected {selected_name} within frozen pair only: "
        f"{pair_result_path}"
    )
    print("[fresh-exact-paired-q50] whole arm continues; no stop/promotion rule was declared")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--preregistration", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract-check")
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--runtime-contract", required=True, type=Path)
    run_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    run_parser.add_argument(
        "--gpus", nargs=2, required=True, type=int, metavar=("MODEL_2000", "MODEL_4000")
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    prereg_path = args.preregistration.resolve()
    require_sha("expected config SHA", args.expected_config_sha256)
    if not config_path.is_file() or sha256_file(config_path) != args.expected_config_sha256:
        raise ContractError("execution config file SHA mismatch")
    config = load_execution_config(config_path)
    if sha256_file(Path(__file__).resolve()) != config["tools"]["runner_sha256"]:
        raise ContractError("fresh exact q50 runner bytes do not match the execution config")
    shared = SCRIPT_DIR / config["tools"]["shared_causal_runner"]["path"]
    if not shared.is_file() or sha256_file(shared) != config["tools"]["shared_causal_runner"]["sha256"]:
        raise ContractError("shared causal runner dependency bytes changed")
    if not prereg_path.is_file() or sha256_file(prereg_path) != config["preregistration_sha256"]:
        raise ContractError("q50 preregistration file SHA mismatch")
    prereg = load_json(prereg_path)
    validate_preregistration(prereg, config)
    if args.command == "contract-check":
        print("[fresh-exact-paired-q50] offline contract check PASS; no schedule or judge started")
        return 0
    if args.command == "prepare":
        return prepare(config_path, config, prereg_path, prereg)
    return run_pair(
        config,
        prereg,
        args.runtime_contract.resolve(),
        args.expected_runtime_contract_sha256,
        args.gpus,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[fresh-exact-paired-q50][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
