#!/usr/bin/env python3
"""Prepare and explicitly run one fail-closed paired causal BankExam q50.

The script deliberately separates ``prepare`` from ``run``.  Preparation freezes one shared
schema-v3 K=100 schedule and writes an immutable runtime contract, but cannot start a judge.
Running requires the caller to supply the runtime-contract file SHA and records each judge in its
own process group.  This tool is for causal/inexact diagnostics only; it rejects formal-target or
exact-evaluation declarations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_ARM = re.compile(r"^[A-Za-z0-9_.-]+$")
ARM_ORDER = ("M3_old", "M3_S1")


class ContractError(RuntimeError):
    """A frozen input or causal-q50 invariant no longer matches."""


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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
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
        raise ContractError(f"cannot read strict JSON {path}: {exc}") from exc


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
        raise ContractError(f"{name} escapes frozen root {root}: {path}") from exc


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
        },
        "execution config",
    )
    if data["schema_version"] != 1:
        raise ContractError("execution config schema_version must be 1")
    if data["status"] != "offline_preregistered_not_prepared" or data["auto_start"] is not False:
        raise ContractError("execution config must remain offline/not-prepared with auto_start=false")
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
        raise ContractError("paired q50 runner is restricted to causal/inexact diagnostics")

    checkouts = data["checkouts"]
    exact_keys(checkouts, {"training", "evaluation"}, "checkouts")
    for name, spec in checkouts.items():
        exact_keys(spec, {"path", "commit"}, f"checkouts.{name}")
        require_absolute(f"checkouts.{name}.path", spec["path"])
        require_sha(f"checkouts.{name}.commit", spec["commit"], length=40)

    tools = data["tools"]
    exact_keys(tools, {"runner_sha256", "evaluation"}, "tools")
    require_sha("tools.runner_sha256", tools["runner_sha256"])
    expected_tools = {
        "judge",
        "materialize_schedule",
        "schedule_module",
        "mujoco_evaluator",
    }
    exact_keys(tools["evaluation"], expected_tools, "tools.evaluation")
    for name, spec in tools["evaluation"].items():
        exact_keys(spec, {"path", "sha256"}, f"tools.evaluation.{name}")
        if (
            not isinstance(spec["path"], str)
            or os.path.isabs(spec["path"])
            or Path(spec["path"]).parts[0] == ".."
        ):
            raise ContractError(f"tools.evaluation.{name}.path must be repo-relative")
        require_sha(f"tools.evaluation.{name}.sha256", spec["sha256"])

    schedule = data["schedule"]
    exact_keys(
        schedule,
        {
            "schema_version",
            "per_clip_quota",
            "schedule_k",
            "attempts_per_side",
            "schedule_seed",
            "hold_range",
            "noise_scales",
            "one_question_reset",
            "no_wrap",
            "same_artifact_for_both_arms",
            "allow_inexact_contract",
        },
        "schedule",
    )
    expected_schedule = {
        "schema_version": 3,
        "per_clip_quota": 50,
        "schedule_k": 100,
        "attempts_per_side": 50,
        "schedule_seed": 0,
        "hold_range": [0, 100],
        "noise_scales": [0.0],
        "one_question_reset": True,
        "no_wrap": True,
        "same_artifact_for_both_arms": True,
        "allow_inexact_contract": True,
    }
    if schedule != expected_schedule:
        raise ContractError(f"schedule must be the frozen clean q50 contract: {expected_schedule}")

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
    if data.get("schema_version") != 1:
        raise ContractError("q50 preregistration schema_version must be 1")
    if (
        data.get("status") != "preregistered_not_started"
        or data.get("auto_activate") is not False
        or data.get("jobs_started") != 0
        or data.get("runtime_state") is not None
    ):
        raise ContractError("q50 preregistration is no longer pristine/not-started")
    if data.get("training_commit") != config["checkouts"]["training"]["commit"]:
        raise ContractError("preregistration training commit disagrees with execution config")
    if data.get("eval_commit") != config["checkouts"]["evaluation"]["commit"]:
        raise ContractError("preregistration eval commit disagrees with execution config")
    if data.get("eval_root") != config["checkouts"]["evaluation"]["path"]:
        raise ContractError("preregistration eval root disagrees with execution config")
    judge_sha = config["tools"]["evaluation"]["judge"]["sha256"]
    if data.get("judge_script_sha256") != judge_sha:
        raise ContractError("preregistration judge SHA disagrees with execution config")

    trigger = data.get("source_trigger", {})
    if trigger.get("trigger_met") is not True or trigger.get("q10_pair_sha256") is None:
        raise ContractError("paired q50 trigger is not bound/met")
    require_sha("source_trigger.q10_pair_sha256", trigger["q10_pair_sha256"])

    paper = data.get("paper", {})
    schedule = config["schedule"]
    if (
        paper.get("seed") != schedule["schedule_seed"]
        or paper.get("noise_scales") != schedule["noise_scales"]
        or paper.get("schedule_k") != schedule["schedule_k"]
        or paper.get("attempts_per_side") != schedule["attempts_per_side"]
        or paper.get("hold_steps_range") != schedule["hold_range"]
        or paper.get("no_wrap") is not True
        or paper.get("one_question_reset") is not True
        or paper.get("same_immutable_schedule_required_for_both_arms") is not True
        or paper.get("allow_inexact_contract_required") is not True
        or paper.get("expected_evaluation_contract_exact") is not False
    ):
        raise ContractError("preregistration paper disagrees with the clean shared q50 contract")
    materialization = paper.get("schedule_materialization", {})
    if materialization.get("status") != "not_materialized" or materialization.get("sha256") is not None:
        raise ContractError("preregistration already claims a schedule materialization")
    bank = paper.get("exam_bank", {})
    require_absolute("paper.exam_bank.path", bank.get("path"))
    require_sha("paper.exam_bank.sha256", bank.get("sha256"))
    if not isinstance(bank.get("bytes"), int) or isinstance(bank.get("bytes"), bool) or bank["bytes"] <= 0:
        raise ContractError("paper.exam_bank.bytes must be a positive integer")
    require_sha("paper.mjcf_sha256", paper.get("mjcf_sha256"))

    if tuple(data.get("arms", {})) != ARM_ORDER:
        raise ContractError(f"preregistration arms must be ordered exactly as {ARM_ORDER}")
    training_root = Path(config["checkouts"]["training"]["path"])
    for name in ARM_ORDER:
        arm = data["arms"][name]
        if not SAFE_ARM.fullmatch(name):
            raise ContractError(f"unsafe arm name {name!r}")
        if (
            arm.get("checkpoint_iteration") != 20998
            or arm.get("lineage_exact") is not False
            or arm.get("job_status") != "not_started"
            or arm.get("pid") is not None
            or arm.get("pgid") is not None
            or arm.get("result") is not None
        ):
            raise ContractError(f"arm {name} is not the frozen causal terminal checkpoint")
        checkpoint = require_absolute(f"arms.{name}.checkpoint_path", arm.get("checkpoint_path"))
        hard_contract = require_absolute(
            f"arms.{name}.training_contract_path", arm.get("training_contract_path")
        )
        require_under(f"arms.{name}.checkpoint_path", checkpoint, training_root)
        require_under(f"arms.{name}.training_contract_path", hard_contract, training_root)
        if checkpoint.name != "model_20998.pt" or not checkpoint.parent.name.endswith(
            str(arm.get("run_name"))
        ):
            raise ContractError(f"arm {name} checkpoint is not inside its declared terminal run")
        if hard_contract != checkpoint.parent / "params" / "training_contract.json":
            raise ContractError(f"arm {name} hard contract is not checkpoint-adjacent")
        require_sha(f"arms.{name}.checkpoint_sha256", arm.get("checkpoint_sha256"))
        hard_sha = require_sha(
            f"arms.{name}.training_contract_sha256", arm.get("training_contract_sha256")
        )
        if arm.get("checkpoint_embedded_training_contract_sha256") != hard_sha:
            raise ContractError(f"arm {name} preregistration loses checkpoint/hard-contract binding")
    if data["arms"]["M3_old"].get("face_command_pairing") != "legacy_signed_vs_A":
        raise ContractError("M3_old pairing changed")
    if data["arms"]["M3_S1"].get("face_command_pairing") != "shared_plus_y":
        raise ContractError("M3_S1 pairing changed")

    semantics = data.get("diagnostic_semantics", {})
    if (
        semantics.get("causal") is not True
        or semantics.get("evaluation_contract_exact") is not False
        or semantics.get("formal_target") is not False
        or semantics.get("deploy_gate") is not False
    ):
        raise ContractError("preregistration is not explicitly causal/inexact/non-formal")
    activation = data.get("activation", {})
    if (
        activation.get("preregistered") is not True
        or activation.get("authorized_to_start_by_this_file") is not False
        or activation.get("started_at_creation") is not False
    ):
        raise ContractError("preregistration activation fence changed")


def git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"git preflight failed for {root}: {exc.output.strip()}") from exc


def validate_checkout(name: str, spec: dict[str, str]) -> Path:
    root = Path(spec["path"])
    if not root.is_dir():
        raise ContractError(f"{name} checkout does not exist: {root}")
    actual_root = Path(git_output(root, "rev-parse", "--show-toplevel"))
    if actual_root.resolve() != root.resolve():
        raise ContractError(f"{name} checkout root mismatch: {actual_root} != {root}")
    if git_output(root, "rev-parse", "HEAD") != spec["commit"]:
        raise ContractError(f"{name} checkout commit changed")
    if git_output(root, "status", "--porcelain"):
        raise ContractError(f"refusing dirty {name} checkout: {root}")
    return root


def validate_eval_tools(config: dict[str, Any], eval_root: Path) -> dict[str, Path]:
    paths = {}
    for name, spec in config["tools"]["evaluation"].items():
        path = (eval_root / spec["path"]).resolve()
        require_under(f"tools.evaluation.{name}", path, eval_root)
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ContractError(f"evaluation tool bytes changed: {name} {path}")
        paths[name] = path
    return paths


CHECKPOINT_AUDIT_CODE = r"""
import json, math, sys, torch
p=sys.argv[1]
o=torch.load(p,map_location='cpu',weights_only=False)
infos=o.get('infos') if isinstance(o,dict) else None
if not isinstance(infos,dict): infos={}
bad=0; floating=0; tensors=0
def walk(v):
 global bad,floating,tensors
 if torch.is_tensor(v):
  tensors+=1
  if v.is_floating_point() or v.is_complex():
   floating+=1; bad+=int((~torch.isfinite(v)).sum().item())
 elif isinstance(v,dict):
  for x in v.values(): walk(x)
 elif isinstance(v,(list,tuple)):
  for x in v: walk(x)
walk(o)
print(json.dumps({'iter':o.get('iter') if isinstance(o,dict) else None,
 'training_contract_sha256':infos.get('training_contract_sha256'),
 'training_contract_schema_version':infos.get('training_contract_schema_version'),
 'training_contract_lineage_exact':infos.get('training_contract_lineage_exact'),
 'tensor_count':tensors,'floating_tensor_count':floating,'nonfinite':bad},sort_keys=True))
"""


def checkpoint_audit(python: Path, checkpoint: Path, expected_contract_sha: str) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_AUDIT_CODE, str(checkpoint)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"checkpoint audit failed for {checkpoint}: {exc.output}") from exc
    try:
        audit = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint audit returned invalid JSON: {output!r}") from exc
    if (
        audit.get("iter") != 20998
        or audit.get("training_contract_sha256") != expected_contract_sha
        or audit.get("training_contract_schema_version") != 3
        or audit.get("training_contract_lineage_exact") not in (0, False)
        or not isinstance(audit.get("tensor_count"), int)
        or audit["tensor_count"] <= 0
        or audit.get("nonfinite") != 0
    ):
        raise ContractError(f"checkpoint audit violates terminal causal contract: {audit}")
    return audit


def validate_runtime_inputs(
    config: dict[str, Any], prereg: dict[str, Any]
) -> tuple[Path, Path, dict[str, Path], dict[str, Any]]:
    training_root = validate_checkout("training", config["checkouts"]["training"])
    eval_root = validate_checkout("evaluation", config["checkouts"]["evaluation"])
    tools = validate_eval_tools(config, eval_root)
    checkpoint_python = Path(config["runtime"]["checkpoint_python"])
    if not checkpoint_python.is_file() or not os.access(checkpoint_python, os.X_OK):
        raise ContractError(f"checkpoint Python is missing/not executable: {checkpoint_python}")
    bank = prereg["paper"]["exam_bank"]
    bank_path = Path(bank["path"])
    if (
        not bank_path.is_file()
        or bank_path.stat().st_size != bank["bytes"]
        or sha256_file(bank_path) != bank["sha256"]
    ):
        raise ContractError("exam-bank bytes no longer match preregistration")

    arm_audits = {}
    for name in ARM_ORDER:
        arm = prereg["arms"][name]
        checkpoint = Path(arm["checkpoint_path"])
        hard_path = Path(arm["training_contract_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != arm["checkpoint_sha256"]:
            raise ContractError(f"{name} checkpoint bytes changed")
        if not hard_path.is_file() or sha256_file(hard_path) != arm["training_contract_sha256"]:
            raise ContractError(f"{name} hard-contract bytes changed")
        hard = load_json(hard_path)
        if (
            hard.get("schema_version") != 3
            or hard.get("face_command_pairing") != arm["face_command_pairing"]
            # The adjacent schema-3 hard contract records whether the bound
            # motion kinematics are exact.  Overall checkpoint lineage is a
            # checkpoint ``infos`` field and is verified by checkpoint_audit;
            # it is intentionally not duplicated as ``lineage_exact`` here.
            or hard.get("motion_kinematics_exact") is not False
        ):
            raise ContractError(f"{name} hard contract semantics changed")
        audit = checkpoint_audit(
            checkpoint_python, checkpoint, arm["training_contract_sha256"]
        )
        arm_audits[name] = {
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "training_contract_sha256": arm["training_contract_sha256"],
            "checkpoint_audit": audit,
        }
    return training_root, eval_root, tools, arm_audits


def validate_schedule_document(
    path: Path, *, expected_bank_sha256: str
) -> dict[str, Any]:
    document = load_json(path)
    expected_keys = {
        "artifact_type",
        "bank_schema_version",
        "bank_sha256",
        "clip_order",
        "hold_range",
        "hold_semantics",
        "items",
        "no_wrap",
        "per_clip_quota",
        "question_counts",
        "schedule_seed",
        "schedule_sha256",
        "schema_version",
    }
    exact_keys(document, expected_keys, "schedule document")
    declared = require_sha("schedule.schedule_sha256", document["schedule_sha256"])
    payload = dict(document)
    payload.pop("schedule_sha256")
    if canonical_sha256(payload) != declared:
        raise ContractError("schedule semantic SHA does not match its canonical payload")
    if path.read_bytes() != canonical_bytes(document) + b"\n":
        raise ContractError("schedule artifact is not in canonical byte form")
    if (
        document["schema_version"] != 3
        or document["bank_schema_version"] != 3
        or document["artifact_type"] != "bank-exam-schedule"
        or document["bank_sha256"] != expected_bank_sha256
        or document["clip_order"] != ["forehand", "backhand"]
        or document["hold_range"] != [0, 100]
        or document["per_clip_quota"] != 50
        or document["schedule_seed"] != 0
        or document["no_wrap"] is not True
        or len(document["items"]) != 100
    ):
        raise ContractError("materialized schedule disagrees with the frozen q50 paper")
    seen_rows = set()
    seen_ids = set()
    for index, item in enumerate(document["items"]):
        exact_keys(
            item,
            {
                "attempt_seed",
                "bank_row",
                "clip",
                "hold_steps",
                "question_id",
                "repeat",
                "schedule_index",
            },
            f"schedule.items[{index}]",
        )
        clip = index % 2
        if (
            item["schedule_index"] != index
            or item["clip"] != clip
            or item["repeat"] != 0
            or not isinstance(item["hold_steps"], int)
            or isinstance(item["hold_steps"], bool)
            or not 0 <= item["hold_steps"] <= 100
            or not isinstance(item["bank_row"], int)
            or isinstance(item["bank_row"], bool)
            or item["bank_row"] < 0
        ):
            raise ContractError(f"schedule item {index} violates balanced no-wrap semantics")
        prefix = "forehand:" if clip == 0 else "backhand:"
        if not isinstance(item["question_id"], str) or not item["question_id"].startswith(prefix):
            raise ContractError(f"schedule item {index} has the wrong qualified question ID")
        require_sha(
            f"schedule.items[{index}].question_id",
            item["question_id"][len(prefix):],
        )
        key = (clip, item["bank_row"])
        if key in seen_rows or item["question_id"] in seen_ids:
            raise ContractError("schedule contains duplicate rows/questions")
        seen_rows.add(key)
        seen_ids.add(item["question_id"])
    return document


def semantic_judge_args(schedule_path: Path) -> list[str]:
    if "\n" in str(schedule_path):
        raise ContractError("schedule path contains a newline")
    exam_extra = (
        "--exam-schedule-json " + shlex.quote(str(schedule_path)) + " --allow-inexact-contract"
    )
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
    return [
        "bash",
        str(judge),
        str(Path(arm["checkpoint_path"]).parent),
        arm["checkpoint_path"],
        "--gpu",
        str(gpu),
        *semantic_judge_args(schedule_path),
    ]


def prepare(config_path: Path, config: dict[str, Any], prereg_path: Path, prereg: dict[str, Any]) -> int:
    _, eval_root, tools, arm_audits = validate_runtime_inputs(config, prereg)
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
        "tools": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
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
                "checkpoint_path": prereg["arms"][name]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
                "training_contract_path": prereg["arms"][name]["training_contract_path"],
                "training_contract_sha256": prereg["arms"][name]["training_contract_sha256"],
                "face_command_pairing": prereg["arms"][name]["face_command_pairing"],
                "checkpoint_audit": arm_audits[name]["checkpoint_audit"],
                "job_status": "not_started",
            }
            for name in ARM_ORDER
        },
        "paired_result_path": str(state_dir / runtime["paired_result_filename"]),
    }
    atomic_json(contract_path, runtime_contract)
    print(f"[paired-q50] prepared only; no judges started")
    print(f"[paired-q50] schedule={schedule_path} sha256={schedule['schedule_sha256']}")
    print(f"[paired-q50] runtime_contract={contract_path} sha256={sha256_file(contract_path)}")
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
        or contract.get("causal") is not True
        or contract.get("evaluation_contract_exact") is not False
        or contract.get("formal_target") is not False
        or contract.get("deploy_gate") is not False
        or contract.get("arm_order") != list(ARM_ORDER)
    ):
        raise ContractError("runtime contract is not a pristine causal/inexact prepared pair")
    if contract.get("config", {}).get("sha256") != sha256_file(Path(contract["config"]["path"])):
        raise ContractError("runtime contract's execution config changed")
    if contract.get("preregistration", {}).get("sha256") != sha256_file(
        Path(contract["preregistration"]["path"])
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


def find_report(log_text: str) -> Path:
    matches = re.findall(r"^\[judge\] \u62a5\u544a:\s*(.+)$", log_text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ContractError(f"judge log must name exactly one report, found {matches}")
    report = Path(matches[0].strip())
    if not report.is_file():
        raise ContractError(f"judge reported a missing report: {report}")
    return report


def validate_exam_result(
    *,
    report: Path,
    arm: dict[str, Any],
    prereg: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    match = re.fullmatch(r"judge_report_(model_\d+)_(\d{8}_\d{6})\.md", report.name)
    if not match:
        raise ContractError(f"unexpected judge report name: {report}")
    judge_dir = report.parent / f"{match.group(1)}_{match.group(2)}"
    summary_path = judge_dir / "exam" / "mujoco_sim2sim_summary.json"
    attempts_path = judge_dir / "exam" / "mujoco_sim2sim_attempts.csv"
    if not summary_path.is_file() or not attempts_path.is_file():
        raise ContractError("judge completed without summary/attempt ledger")
    summary = load_json(summary_path)
    schedule = runtime_contract["shared_schedule"]
    bank = prereg["paper"]["exam_bank"]
    schedule_document = validate_schedule_document(
        Path(schedule["path"]), expected_bank_sha256=bank["sha256"]
    )
    expected_items = schedule_document["items"]
    arguments = summary.get("arguments", {})
    input_artifacts = summary.get("input_artifacts", {})
    schedule_artifact = input_artifacts.get("exam_schedule_artifact", {})
    exam_schedule = summary.get("exam_schedule", {})
    results = summary.get("results")
    schedule_item_keys = tuple(expected_items[0])
    summary_items = exam_schedule.get("items", [])
    summary_item_projection = [
        {key: item.get(key) for key in schedule_item_keys} for item in summary_items
    ] if isinstance(summary_items, list) and all(isinstance(item, dict) for item in summary_items) else []
    summary_indices_exact = bool(summary_items) and all(
        item.get("question_sequence_index") == item.get("schedule_index")
        for item in summary_items
    )
    result_items = (
        results[0].get("exam_schedule", {}).get("items", [])
        if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict)
        else []
    )
    result_item_projection = [
        {key: item.get(key) for key in schedule_item_keys} for item in result_items
    ] if isinstance(result_items, list) and all(isinstance(item, dict) for item in result_items) else []
    result_runtime_exact = bool(result_items) and all(
        item.get("eligible") is True
        and item.get("censored") is False
        and item.get("ready_state_mode") == "mjcf_named_keyframe:stand:v1"
        and item.get("ready_state_sha256") == summary.get("ready_state_sha256")
        and item.get("mjcf_sha256") == summary.get("mjcf_sha256")
        and item.get("execution_contract_sha256")
        == summary.get("execution_contract_sha256")
        and isinstance(item.get("physical_fall"), bool)
        and isinstance(item.get("guard_reset"), bool)
        and isinstance(item.get("hit"), bool)
        and isinstance(item.get("returned"), bool)
        and item.get("question_sequence_index") == item.get("schedule_index")
        for item in result_items
    )
    if (
        summary.get("schema_version") != 3
        or summary.get("evaluation_contract_exact") is not False
        or arguments.get("allow_inexact_contract") is not True
        or arguments.get("target_source") != "bank"
        or arguments.get("seed") != 0
        or arguments.get("noise_scales") != [0.0]
        or arguments.get("qdes_clamp") is not True
        or arguments.get("hold_ref") != "auto"
        or arguments.get("ready_state") != "auto"
        or arguments.get("exam_continuity_diagnostic") is not False
        or arguments.get("exam_schedule_k") is not None
        or os.path.abspath(str(arguments.get("exam_schedule_json"))) != schedule["path"]
        or input_artifacts.get("exam_bank", {}).get("sha256") != bank["sha256"]
        or schedule_artifact.get("sha256") != schedule["file_sha256"]
        or schedule_artifact.get("schedule_sha256") != schedule["schedule_sha256"]
        or schedule_artifact.get("schema_version") != 3
        or exam_schedule.get("sha256") != schedule["schedule_sha256"]
        or exam_schedule.get("bank_sha256") != bank["sha256"]
        or exam_schedule.get("seed") != 0
        or exam_schedule.get("size") != 100
        or exam_schedule.get("one_question_reset") is not True
        or exam_schedule.get("ready_state_mode") != "mjcf_named_keyframe:stand:v1"
        or exam_schedule.get("shared_artifact") != schedule_document
        or summary_item_projection != expected_items
        or not summary_indices_exact
        or result_item_projection != expected_items
        or not result_runtime_exact
        or not isinstance(results, list)
        or len(results) != 1
        or results[0].get("evaluation_contract_exact") is not False
        or results[0].get("noise_scale") != 0.0
        or results[0].get("exam_schedule", {}).get("question_id_order")
        != schedule["question_id_order"]
    ):
        raise ContractError("q50 result does not reproduce the frozen causal/inexact paper")
    if summary.get("mjcf_sha256") != prereg["paper"]["mjcf_sha256"]:
        raise ContractError("q50 result MJCF differs from preregistration")

    with attempts_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ledger_items = []
    try:
        for row in rows:
            if row["clip_name"] not in ("forehand", "backhand"):
                raise ValueError(f"unknown clip_name={row['clip_name']!r}")
            ledger_items.append(
                {
                    "schedule_index": int(row["schedule_index"]),
                    "clip": 0 if row["clip_name"] == "forehand" else 1,
                    "bank_row": int(row["bank_row"]),
                    "question_id": row["question_id"],
                    "repeat": int(row["repeat"]),
                    "hold_steps": int(row["hold_steps"]),
                    "attempt_seed": int(row["attempt_seed"]),
                }
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"q50 attempt ledger lacks exact schedule fields: {exc}") from exc
    if (
        len(rows) != 100
        or ledger_items != expected_items
        or any(row.get("schedule_sha256") != schedule["schedule_sha256"] for row in rows)
        or any(row.get("censored") not in ("0", "False", "false") for row in rows)
    ):
        raise ContractError("q50 attempt ledger is incomplete, reordered, censored, or re-papered")
    return {
        "run_name": arm["run_name"],
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "report": {"path": str(report), "sha256": sha256_file(report)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "attempt_ledger": {"path": str(attempts_path), "sha256": sha256_file(attempts_path)},
        "schedule_sha256": schedule["schedule_sha256"],
        "question_id_order": schedule["question_id_order"],
        "mjcf_sha256": summary["mjcf_sha256"],
        "execution_contract_sha256": summary["execution_contract_sha256"],
        "ready_state_sha256": summary["ready_state_sha256"],
        "evaluation_contract_exact": False,
        "causal": True,
        "formal_target": False,
        "result": results[0],
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
        raise ContractError("run requires exactly two GPUs in M3_old, M3_S1 order")
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
        report = find_report(log_path.read_text(encoding="utf-8", errors="replace"))
        results[name] = validate_exam_result(
            report=report,
            arm=arm,
            prereg=prereg,
            runtime_contract=contract,
        )

    old, s1 = results["M3_old"], results["M3_S1"]
    for key in (
        "schedule_sha256",
        "question_id_order",
        "mjcf_sha256",
        "execution_contract_sha256",
        "ready_state_sha256",
    ):
        if old[key] != s1[key]:
            raise ContractError(f"paired q50 cells disagree on shared runtime field {key}")
    pair_result = {
        "schema_version": 1,
        "pair_id": config["contract_id"],
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_contract": {"path": str(runtime_path), "sha256": runtime_sha},
        "causal": True,
        "evaluation_contract_exact": False,
        "formal_target": False,
        "deploy_gate": False,
        "shared_schedule_sha256": old["schedule_sha256"],
        "question_id_order": old["question_id_order"],
        "arms": results,
    }
    atomic_json(pair_result_path, pair_result)
    print(f"[paired-q50] complete causal/inexact pair: {pair_result_path}")
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
    run_parser.add_argument("--gpus", nargs=2, required=True, type=int, metavar=("OLD", "S1"))
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    prereg_path = args.preregistration.resolve()
    require_sha("expected config SHA", args.expected_config_sha256)
    if not config_path.is_file() or sha256_file(config_path) != args.expected_config_sha256:
        raise ContractError("execution config file SHA mismatch")
    config = load_execution_config(config_path)
    if sha256_file(Path(__file__).resolve()) != config["tools"]["runner_sha256"]:
        raise ContractError("paired q50 runner bytes do not match the execution config")
    if (
        not prereg_path.is_file()
        or sha256_file(prereg_path) != config["preregistration_sha256"]
    ):
        raise ContractError("q50 preregistration file SHA mismatch")
    prereg = load_json(prereg_path)
    validate_preregistration(prereg, config)
    if args.command == "contract-check":
        print("[paired-q50] offline contract check PASS; no schedule or judge started")
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
        print(f"[paired-q50][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
