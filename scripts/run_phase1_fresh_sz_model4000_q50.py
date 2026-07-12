#!/usr/bin/env python3
"""Consume the all-four activation and run the matched fresh-SZ model_4000 q50.

This runner has no SSH or signal surface.  Every command requires the exact
content-addressed all-four activation produced by
``validate_phase1_fresh_sz_model4000_q50_queue.py``.  ``contract-check`` is
read-only, ``prepare`` creates one no-clobber Pod state directory and an
activation-bound runtime contract, ``run`` evaluates that Pod's two arms
serially, and ``aggregate`` combines the two content-bound Pod results.

Seed 1 is deliberately rerun on the same already-materialized K100 bytes.  No
prior result is reused, no new paper is generated, and no result authorizes a
trainer/worker signal, stop, promotion, deployment, or real-robot action.
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
QUEUE_VALIDATOR_PATH = SCRIPT_DIR / "validate_phase1_fresh_sz_model4000_q50_queue.py"
QUEUE_VALIDATOR_SHA256 = "e763ecb9a822f7e1c2e9338749701fcd4bfea9f26f9b6fe5b4b189f8ca5a6cd3"
FRESH_HELPER_PATH = SCRIPT_DIR / "run_phase1_fresh_exact_paired_bank_q50.py"
FRESH_HELPER_SHA256 = "3528250777a170791f39d8dd17716c2a7f8ca91416a3ffa8433ec5eb691ed9e0"


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for _dependency, _expected in (
    (QUEUE_VALIDATOR_PATH, QUEUE_VALIDATOR_SHA256),
    (FRESH_HELPER_PATH, FRESH_HELPER_SHA256),
):
    if not _dependency.is_file() or _bootstrap_sha256(_dependency) != _expected:
        raise RuntimeError(f"refusing changed q50 dependency: {_dependency}")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import validate_phase1_fresh_sz_model4000_q50_queue as barrier  # noqa: E402
import run_phase1_fresh_exact_paired_bank_q50 as fresh  # noqa: E402


ContractError = fresh.ContractError
sha256_file = fresh.sha256_file
canonical_sha256 = fresh.canonical_sha256
atomic_json = fresh.atomic_json
load_json = fresh.load_json
exact_keys = fresh.exact_keys
require_sha = fresh.require_sha
require_absolute = fresh.require_absolute
require_under = fresh.require_under

SEED_ORDER = barrier.SEED_ORDER
POD_ARM_ORDER = barrier.POD_ARM_ORDER
EXPECTED_QUEUE_SHA256 = "d4e69d91adfe7a42aee897c11b1b6d6bf7e5eaa7fb81d856b66cab7b3f7d3909"
EXPECTED_PREREG_SHA256 = "ca5ea90f8420ef4c96ee05881b25d062cc437faa97510babca45299afcabbff0"
EXPECTED_SEMANTICS = dict(barrier.EXPECTED_SEMANTICS)
EXPECTED_SCHEDULE = dict(barrier.EXPECTED_SCHEDULE)
EXPECTED_GATE_RULE = dict(barrier.EXPECTED_GATE_RULE)
EXPECTED_INTERPRETATION = {
    "known_seed1_model4000": {
        "aggregate_return_rate": 0.50,
        "forehand_return_rate": 0.00,
        "backhand_return_rate": 1.00,
        "family_gate_pass_possible": False,
    },
    "seed4_delayed_learning_supported_only_if": {
        "aggregate_rate_min": 0.65,
        "every_side_rate_min": 0.50,
        "threshold_source": "unchanged_model2000_stability_gate",
    },
    "family_stable_claim_allowed": False,
    "reason_family_claim_forbidden": "known_seed1_model4000_rate_0.50_is_below_0.65",
}
EXPECTED_EVAL_TOOLS = {
    "judge": {
        "path": "hope_training/whole_body_tracking/scripts/judge.sh",
        "sha256": "1a00702935096b063435c3f0bd23e75f76f13e1298c87310d1cec3c26cca8529",
    },
    "materialize_schedule": {
        "path": "hope_training/whole_body_tracking/scripts/materialize_bank_exam_schedule.py",
        "sha256": "fe2a4f694f8a1c5a69f42fafeeadaf848b440903385acfba7468073eef010937",
    },
    "schedule_module": {
        "path": "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py",
        "sha256": "32721f018f6a35a42aa12ff0a7e48c0d9bc513d238988d953241ee625744b23b",
    },
    "mujoco_evaluator": {
        "path": "hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py",
        "sha256": "e4a9fa42ff0f7e68cebdf16f2e0c61299507496c6edc13546baeb3d576ecb20a",
    },
}


def _strict_json(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _content_document(kind: str, content: Mapping[str, Any]) -> dict[str, Any]:
    material = dict(content)
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "content_sha256": canonical_sha256(material),
        "content": material,
    }


def _write_no_clobber(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise ContractError(f"no-clobber: output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, dict(document))


def load_execution_config(path: Path) -> dict[str, Any]:
    data = _strict_json(path)
    exact_keys(
        data,
        {
            "schema_version", "contract_id", "status", "auto_start", "source_bindings",
            "semantics", "checkouts", "evaluation_tools", "schedule", "activation",
            "runtime", "pod_arm_order", "gate_rule", "interpretation_rule",
        },
        "model4000 execution config",
    )
    if (
        data["schema_version"] != 1
        or data["status"] != "offline_activation_required_not_prepared"
        or data["auto_start"] is not False
    ):
        raise ContractError("execution config must stay offline and activation-gated")
    if data["semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("execution semantics changed")
    if data["checkouts"] != {
        "training": {"path": "/workspace/codexschema/nohope", "commit": barrier.EXPECTED_TRAIN_COMMIT},
        "evaluation": {
            "path": "/workspace/codexschema/nohope_eval_08e438e",
            "commit": barrier.EXPECTED_EVAL_COMMIT,
        },
    }:
        raise ContractError("execution checkout bindings changed")
    if data["schedule"] != EXPECTED_SCHEDULE:
        raise ContractError("execution config changed or rematerialized the K100 paper")
    if data["pod_arm_order"] != {key: list(value) for key, value in POD_ARM_ORDER.items()}:
        raise ContractError("Pod/seed serial order changed")
    if data["gate_rule"] != EXPECTED_GATE_RULE:
        raise ContractError("unchanged model2000 stability thresholds changed")
    if data["interpretation_rule"] != EXPECTED_INTERPRETATION:
        raise ContractError("known seed1 or delayed-vs-persistent interpretation changed")

    sources = data["source_bindings"]
    exact_keys(
        sources,
        {"queue", "preregistration", "queue_validator", "fresh_helper", "runner"},
        "source_bindings",
    )
    expected_sources = {
        "queue": {
            "path": "configs/phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json",
            "sha256": EXPECTED_QUEUE_SHA256,
        },
        "preregistration": {
            "path": "configs/phase1_fresh_SZ_model4000_seed_stability_q50_prereg_20260712.json",
            "sha256": EXPECTED_PREREG_SHA256,
        },
        "queue_validator": {
            "path": "scripts/validate_phase1_fresh_sz_model4000_q50_queue.py",
            "sha256": QUEUE_VALIDATOR_SHA256,
        },
        "fresh_helper": {
            "path": "scripts/run_phase1_fresh_exact_paired_bank_q50.py",
            "sha256": FRESH_HELPER_SHA256,
        },
    }
    for name, expected in expected_sources.items():
        if sources[name] != expected:
            raise ContractError(f"source binding changed: {name}")
    exact_keys(sources["runner"], {"path", "sha256"}, "source_bindings.runner")
    if sources["runner"]["path"] != "scripts/run_phase1_fresh_sz_model4000_q50.py":
        raise ContractError("runner path changed")
    require_sha("source_bindings.runner.sha256", sources["runner"]["sha256"])
    if sources["runner"]["sha256"] != sha256_file(Path(__file__).resolve()):
        raise ContractError("runner bytes differ from source binding")

    if data["evaluation_tools"] != EXPECTED_EVAL_TOOLS:
        raise ContractError("evaluation tool closure changed")
    activation = data["activation"]
    if activation != {
        "required": True,
        "artifact_kind": "phase1_fresh_sz_model4000_q50_all_four_activation",
        "barrier_id": "fresh_SZ_model4000_all_four_ready_v1",
        "status": "all_four_checkpoints_ready_judge_not_started",
        "seed_order": list(SEED_ORDER),
        "activation_sha_bound_at_runtime": True,
    }:
        raise ContractError("activation fence changed")

    runtime = data["runtime"]
    exact_keys(
        runtime,
        {
            "checkpoint_python", "pod_state_dirs", "schedule_filename",
            "runtime_contract_filename", "pod_result_filename", "aggregate_output_dir",
            "kit_boot_lock", "seed1_execution",
        },
        "runtime",
    )
    checkpoint_python = require_absolute("runtime.checkpoint_python", runtime["checkpoint_python"])
    if checkpoint_python.name != "python":
        raise ContractError("checkpoint interpreter changed")
    if set(runtime["pod_state_dirs"]) != set(POD_ARM_ORDER):
        raise ContractError("runtime state dirs must cover pod1 and pod2")
    for pod, raw in runtime["pod_state_dirs"].items():
        require_absolute(f"runtime.pod_state_dirs.{pod}", raw)
    require_absolute("runtime.aggregate_output_dir", runtime["aggregate_output_dir"])
    if runtime["kit_boot_lock"] != "/workspace/.kit_boot.lock":
        raise ContractError("shared Kit boot lock changed")
    if runtime["seed1_execution"] != "fresh_rerun_same_k100_no_reuse":
        raise ContractError("seed1 must be conservatively rerun")
    for key in ("schedule_filename", "runtime_contract_filename", "pod_result_filename"):
        value = runtime[key]
        if not isinstance(value, str) or value != Path(value).name or not value.endswith(".json"):
            raise ContractError(f"runtime.{key} must be a simple .json filename")
    return data


def _resolve_bound_sources(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    root = config_path.parent.parent.resolve()
    resolved: dict[str, Path] = {}
    for name, spec in config["source_bindings"].items():
        candidate = (root / spec["path"]).resolve()
        require_under(f"source_bindings.{name}", candidate, root)
        if not candidate.is_file() or sha256_file(candidate) != spec["sha256"]:
            raise ContractError(f"bound source bytes changed/missing: {name}")
        resolved[name] = candidate
    if resolved["runner"] != Path(__file__).resolve():
        raise ContractError("execution config does not bind this runner")
    if resolved["queue_validator"] != QUEUE_VALIDATOR_PATH.resolve():
        raise ContractError("queue validator dependency path changed")
    if resolved["fresh_helper"] != FRESH_HELPER_PATH.resolve():
        raise ContractError("fresh helper dependency path changed")
    queue = _strict_json(resolved["queue"])
    prereg = _strict_json(resolved["preregistration"])
    barrier.validate_queue(queue)
    barrier.validate_preregistration(prereg, queue)
    if queue["paper"] != config["schedule"] or prereg["gate_rule"] != config["gate_rule"]:
        raise ContractError("execution config disagrees with frozen queue/preregistration")
    return resolved["queue"], queue, resolved["preregistration"], prereg


def _validate_activation_document(
    activation_path: Path,
    expected_sha: str,
    queue_path: Path,
    queue: dict[str, Any],
    prereg_path: Path,
) -> dict[str, Any]:
    require_sha("expected activation SHA", expected_sha)
    if not activation_path.is_file() or sha256_file(activation_path) != expected_sha:
        raise ContractError("activation artifact bytes changed/missing")
    document = _strict_json(activation_path)
    exact_keys(
        document,
        {"schema_version", "artifact_kind", "content_sha256", "content"},
        "activation artifact",
    )
    content = document["content"]
    if (
        document["schema_version"] != 1
        or document["artifact_kind"]
        != "phase1_fresh_sz_model4000_q50_all_four_activation"
        or not isinstance(content, dict)
        or document["content_sha256"] != canonical_sha256(content)
        or content.get("queue_id") != queue["queue_id"]
        or content.get("barrier_id") != queue["barrier"]["id"]
        or content.get("status") != "all_four_checkpoints_ready_judge_not_started"
        or content.get("validator_sha256") != QUEUE_VALIDATOR_SHA256
        or content.get("seed_order") != list(SEED_ORDER)
        or content.get("gate_rule") != EXPECTED_GATE_RULE
        or any(content.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError("activation is not the exact all-four barrier artifact")
    if content.get("actions") != {
        "judges_started": 0,
        "trainer_or_worker_signals": [],
        "future_q50_runner_may_prepare_only_with_this_exact_artifact": True,
        "stop_or_promote_authorized": False,
        "deploy_or_real_robot_authorized": False,
    }:
        raise ContractError("activation contains an unauthorized action")
    if content.get("queue", {}).get("sha256") != EXPECTED_QUEUE_SHA256:
        raise ContractError("activation queue SHA changed")
    if content.get("preregistration", {}).get("sha256") != EXPECTED_PREREG_SHA256:
        raise ContractError("activation preregistration SHA changed")
    activation_queue_path = Path(str(content["queue"].get("path", "")))
    activation_prereg_path = Path(str(content["preregistration"].get("path", "")))
    if (
        not activation_queue_path.is_absolute()
        or not activation_queue_path.is_file()
        or sha256_file(activation_queue_path) != EXPECTED_QUEUE_SHA256
        or _strict_json(activation_queue_path) != queue
        or not activation_prereg_path.is_absolute()
        or not activation_prereg_path.is_file()
        or sha256_file(activation_prereg_path) != EXPECTED_PREREG_SHA256
        or _strict_json(activation_prereg_path) != _strict_json(prereg_path)
    ):
        raise ContractError("activation-bound queue/preregistration bytes are unavailable or changed")

    audits = content.get("pod_audits")
    if not isinstance(audits, dict) or set(audits) != set(POD_ARM_ORDER):
        raise ContractError("activation must bind exactly two Pod audits")
    pod_contents: dict[str, dict[str, Any]] = {}
    for pod in POD_ARM_ORDER:
        meta = audits[pod]
        if not isinstance(meta, dict) or set(meta) != {"path", "sha256"}:
            raise ContractError(f"activation {pod} audit metadata malformed")
        audit_path = Path(str(meta["path"]))
        pod_contents[pod] = barrier.validate_pod_audit(
            audit_path,
            meta["sha256"],
            activation_queue_path,
            queue,
            activation_prereg_path,
            pod=pod,
        )
    if pod_contents["pod1"]["shared_schedule"] != pod_contents["pod2"]["shared_schedule"]:
        raise ContractError("activation Pod audits disagree on K100 bytes/path")
    if content.get("shared_schedule") != pod_contents["pod1"]["shared_schedule"]:
        raise ContractError("activation shared schedule differs from Pod audits")

    audited_arms = {**pod_contents["pod1"]["arms"], **pod_contents["pod2"]["arms"]}
    if list(content.get("arms", {})) != list(SEED_ORDER):
        raise ContractError("activation arm order/coverage changed")
    if any(content["arms"][seed] != audited_arms[seed] for seed in SEED_ORDER):
        raise ContractError("activation arms differ from their two Pod audits")
    for seed in SEED_ORDER:
        arm = content["arms"][seed]
        require_sha(f"activation.{seed}.checkpoint_sha256", arm.get("checkpoint_sha256"))
        if arm.get("training_contract_sha256") != barrier.EXPECTED_HARD_CONTRACT_SHA256:
            raise ContractError(f"activation {seed} hard-contract SHA changed")
        audit = arm.get("checkpoint_audit")
        if not isinstance(audit, dict):
            raise ContractError(f"activation {seed} lacks checkpoint audit")
        barrier._validate_checkpoint_audit(audit)
    if (
        content["arms"]["seed1"]["checkpoint_sha256"]
        != queue["seed1_reuse"]["expected_checkpoint_sha256"]
    ):
        raise ContractError("activation changed the known seed1 checkpoint bytes")
    return {
        "path": activation_path,
        "sha256": expected_sha,
        "content_sha256": document["content_sha256"],
        "content": content,
        "pod_contents": pod_contents,
        "activation_queue_path": activation_queue_path,
        "activation_prereg_path": activation_prereg_path,
        "local_queue_path": queue_path,
    }


def _validate_schedule(path: Path, prereg: dict[str, Any]) -> dict[str, Any]:
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
        or len(order) != 100
    ):
        raise ContractError("shared K100 schedule semantics/order changed")
    return schedule


def _validate_eval_tools(config: dict[str, Any], eval_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["evaluation_tools"].items():
        path = (eval_root / spec["path"]).resolve()
        require_under(f"evaluation_tools.{name}", path, eval_root)
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ContractError(f"evaluation tool bytes changed: {name}")
        paths[name] = path
    source = paths["judge"].read_text(encoding="utf-8")
    if "JUDGE_KIT_BOOT_LOCK" not in source or "flock -x 8" not in source:
        raise ContractError("bound judge no longer uses the shared Kit boot lock")
    return paths


def _bound_arm(prereg: dict[str, Any], activation: dict[str, Any], seed: str) -> dict[str, Any]:
    arm = dict(prereg["arms"][seed])
    frozen = activation["content"]["arms"][seed]
    arm["checkpoint_sha256"] = frozen["checkpoint_sha256"]
    arm["training_contract_sha256"] = frozen["training_contract_sha256"]
    arm["checkpoint_embedded_training_contract_sha256"] = frozen[
        "training_contract_sha256"
    ]
    arm["execution_mode"] = "fresh_rerun_same_k100"
    return arm


def _validate_runtime_inputs(
    config: dict[str, Any],
    prereg: dict[str, Any],
    activation: dict[str, Any],
    *,
    pod: str,
    schedule_path: Path,
) -> tuple[Path, Path, dict[str, Path], dict[str, Any], dict[str, Any]]:
    if pod not in POD_ARM_ORDER:
        raise ContractError(f"unknown Pod {pod!r}")
    training_root = fresh.validate_checkout("training", config["checkouts"]["training"])
    eval_root = fresh.validate_checkout("evaluation", config["checkouts"]["evaluation"])
    tools = _validate_eval_tools(config, eval_root)
    schedule_path = schedule_path.resolve()
    activation_schedule = Path(activation["content"]["shared_schedule"]["path"]).resolve()
    if schedule_path != activation_schedule:
        raise ContractError("runner must consume the activation-audited materialized schedule path")
    schedule = _validate_schedule(schedule_path, prereg)
    bank = prereg["paper"]["exam_bank"]
    bank_path = Path(bank["path"])
    if (
        not bank_path.is_file()
        or bank_path.stat().st_size != bank["bytes"]
        or sha256_file(bank_path) != bank["sha256"]
    ):
        raise ContractError("exact exam-bank bytes changed")
    checkpoint_python = Path(config["runtime"]["checkpoint_python"])
    if not checkpoint_python.is_file() or not os.access(checkpoint_python, os.X_OK):
        raise ContractError("checkpoint audit interpreter missing/not executable")

    audits: dict[str, Any] = {}
    for seed in POD_ARM_ORDER[pod]:
        frozen = activation["content"]["arms"][seed]
        arm = _bound_arm(prereg, activation, seed)
        checkpoint = Path(arm["checkpoint_path"])
        hard_path = Path(arm["training_contract_path"])
        require_under(f"{seed}.checkpoint", checkpoint, training_root)
        require_under(f"{seed}.hard_contract", hard_path, training_root)
        if (
            checkpoint.name != "model_4000.pt"
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != frozen["checkpoint_sha256"]
        ):
            raise ContractError(f"{seed} checkpoint differs from all-four activation")
        if (
            not hard_path.is_file()
            or sha256_file(hard_path) != barrier.EXPECTED_HARD_CONTRACT_SHA256
        ):
            raise ContractError(f"{seed} adjacent hard-contract bytes changed")
        fresh.validate_hard_contract(_strict_json(hard_path), prereg)
        audit = fresh.checkpoint_audit(
            checkpoint_python,
            checkpoint,
            expected_iteration=4000,
            expected_contract_sha=barrier.EXPECTED_HARD_CONTRACT_SHA256,
        )
        barrier._validate_checkpoint_audit(audit)
        if audit != frozen["checkpoint_audit"]:
            raise ContractError(f"{seed} live checkpoint audit differs from activation audit")
        audits[seed] = audit
    return training_root, eval_root, tools, audits, schedule


def _runtime_document(
    *,
    config_path: Path,
    config: dict[str, Any],
    queue_path: Path,
    prereg_path: Path,
    prereg: dict[str, Any],
    activation: dict[str, Any],
    pod: str,
    schedule_path: Path,
    schedule: dict[str, Any],
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
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "queue_validator_sha256": QUEUE_VALIDATOR_SHA256,
        "fresh_helper_sha256": FRESH_HELPER_SHA256,
        "activation": {
            "path": str(activation["path"]),
            "sha256": activation["sha256"],
            "content_sha256": activation["content_sha256"],
            "barrier_id": activation["content"]["barrier_id"],
            "pod_audit": activation["content"]["pod_audits"][pod],
        },
        "checkouts": config["checkouts"],
        "evaluation_tools": config["evaluation_tools"],
        "kit_boot_lock": config["runtime"]["kit_boot_lock"],
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
                "checkpoint_sha256": activation["content"]["arms"][seed][
                    "checkpoint_sha256"
                ],
                "training_contract_sha256": barrier.EXPECTED_HARD_CONTRACT_SHA256,
                "checkpoint_audit": audits[seed],
                "execution_mode": "fresh_rerun_same_k100",
                "job_status": "not_started",
            }
            for seed in POD_ARM_ORDER[pod]
        },
        "gate_rule": EXPECTED_GATE_RULE,
        "interpretation_rule": EXPECTED_INTERPRETATION,
    }


def prepare(
    config_path: Path,
    config: dict[str, Any],
    queue_path: Path,
    prereg_path: Path,
    prereg: dict[str, Any],
    activation: dict[str, Any],
    *,
    pod: str,
    schedule_source: Path,
) -> int:
    _, _, _, audits, schedule = _validate_runtime_inputs(
        config, prereg, activation, pod=pod, schedule_path=schedule_source
    )
    state_dir = Path(config["runtime"]["pod_state_dirs"][pod])
    if state_dir.exists():
        raise ContractError(f"no-clobber: state directory exists: {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=False)
    copied_schedule = state_dir / config["runtime"]["schedule_filename"]
    if copied_schedule.exists():
        raise ContractError(f"no-clobber: copied schedule exists: {copied_schedule}")
    shutil.copyfile(schedule_source, copied_schedule)
    copied = _validate_schedule(copied_schedule, prereg)
    if copied != schedule:
        raise ContractError("copied K100 schedule differs from activation-audited source")
    runtime_path = state_dir / config["runtime"]["runtime_contract_filename"]
    _write_no_clobber(
        runtime_path,
        _runtime_document(
            config_path=config_path,
            config=config,
            queue_path=queue_path,
            prereg_path=prereg_path,
            prereg=prereg,
            activation=activation,
            pod=pod,
            schedule_path=copied_schedule,
            schedule=copied,
            audits=audits,
        ),
    )
    print(f"[model4000-q50] {pod} prepared only; activation={activation['sha256']}")
    print(f"[model4000-q50] runtime_contract={runtime_path} sha256={sha256_file(runtime_path)}")
    print("[model4000-q50] no judge/signal; schedule bytes copied, never materialized")
    return 0


def _validate_runtime_contract(
    path: Path,
    expected_sha: str,
    config: dict[str, Any],
    queue_path: Path,
    queue: dict[str, Any],
    prereg_path: Path,
    prereg: dict[str, Any],
    activation: dict[str, Any],
    *,
    pod: str,
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
        or contract.get("interpretation_rule") != EXPECTED_INTERPRETATION
        or any(contract.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError("runtime contract is not a pristine activation-bound Pod paper")
    if (
        contract.get("runner_sha256") != sha256_file(Path(__file__).resolve())
        or contract.get("queue_validator_sha256") != QUEUE_VALIDATOR_SHA256
        or contract.get("fresh_helper_sha256") != FRESH_HELPER_SHA256
        or contract.get("checkouts") != config["checkouts"]
        or contract.get("evaluation_tools") != config["evaluation_tools"]
        or contract.get("kit_boot_lock") != config["runtime"]["kit_boot_lock"]
    ):
        raise ContractError("runtime source/tool/checkout binding changed")
    for name, source_path, expected in (
        ("config", None, config),
        ("queue", queue_path, queue),
        ("preregistration", prereg_path, prereg),
    ):
        meta = contract.get(name, {})
        candidate = Path(str(meta.get("path", "")))
        if (
            not candidate.is_absolute()
            or not candidate.is_file()
            or meta.get("sha256") != sha256_file(candidate)
        ):
            raise ContractError(f"runtime {name} bytes changed")
        if source_path is not None and candidate.resolve() != source_path.resolve():
            raise ContractError(f"runtime {name} path changed")
        if _strict_json(candidate) != expected:
            raise ContractError(f"runtime {name} semantics changed")
    activation_meta = contract.get("activation", {})
    if activation_meta != {
        "path": str(activation["path"]),
        "sha256": activation["sha256"],
        "content_sha256": activation["content_sha256"],
        "barrier_id": activation["content"]["barrier_id"],
        "pod_audit": activation["content"]["pod_audits"][pod],
    }:
        raise ContractError("runtime contract activation binding changed")
    schedule_meta = contract.get("shared_schedule", {})
    schedule_path = Path(str(schedule_meta.get("path", "")))
    schedule = _validate_schedule(schedule_path, prereg)
    order = [item["question_id"] for item in schedule["items"]]
    if (
        schedule_meta.get("file_sha256") != sha256_file(schedule_path)
        or schedule_meta.get("schedule_sha256") != schedule["schedule_sha256"]
        or schedule_meta.get("question_id_order") != order
        or schedule_meta.get("question_id_order_sha256") != canonical_sha256(order)
        or schedule_meta.get("schedule_k") != 100
        or schedule_meta.get("attempts_per_side") != 50
    ):
        raise ContractError("runtime K100 paper changed")
    _, _, _, audits, _ = _validate_runtime_inputs(
        config,
        prereg,
        activation,
        pod=pod,
        schedule_path=Path(activation["content"]["shared_schedule"]["path"]),
    )
    for seed in POD_ARM_ORDER[pod]:
        if contract.get("arms", {}).get(seed, {}).get("checkpoint_audit") != audits[seed]:
            raise ContractError(f"runtime {seed} checkpoint audit changed")
    return contract


def _run_judge(
    *,
    seed: str,
    arm: dict[str, Any],
    tools: dict[str, Path],
    schedule_path: Path,
    state_dir: Path,
    runtime_sha: str,
    activation_sha: str,
    kit_boot_lock: str,
    gpu: int,
) -> dict[str, Any]:
    state_path = state_dir / f"{seed}.state.json"
    log_path = state_dir / f"{seed}.runner.log"
    if state_path.exists() or log_path.exists():
        raise ContractError(f"no-clobber: preserved state/log exists for {seed}")
    command = fresh.build_judge_command(
        judge=tools["judge"], arm=arm, schedule_path=schedule_path, gpu=gpu
    )
    if any("--allow-inexact-contract" in value for value in command):
        raise ContractError("formal model4000 command contains diagnostic escape")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JUDGE_KIT_BOOT_LOCK=kit_boot_lock,
    )
    state: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "status": "launching",
        "pid": None,
        "pgid": None,
        "command": command,
        "runtime_contract_sha256": runtime_sha,
        "activation_sha256": activation_sha,
        "schedule_sha256": EXPECTED_SCHEDULE["semantic_sha256"],
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "kit_boot_lock": kit_boot_lock,
        "start_new_session": True,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with log_path.open("xb", buffering=0) as log:
        try:
            proc = subprocess.Popen(
                command,
                cwd=tools["judge"].parent.parent,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            state.update(status="launch_failed", launch_error=f"{type(exc).__name__}: {exc}")
            _write_no_clobber(state_path, state)
            raise ContractError(f"judge {seed} launch failed; preserved state/log") from exc
        try:
            observed_pgid = os.getpgid(proc.pid)
        except ProcessLookupError as exc:
            observed_pgid = None
            state.update(pid=proc.pid, pgid=None, status="pgid_observation_failed")
            _write_no_clobber(state_path, state)
            rc = proc.wait()
            state.update(returncode=rc, log_sha256=sha256_file(log_path))
            atomic_json(state_path, state)
            raise ContractError(f"judge {seed} exited before exact PGID ownership was observed") from exc
        state.update(pid=proc.pid, pgid=observed_pgid, status="running")
        _write_no_clobber(state_path, state)
        if observed_pgid != proc.pid:
            rc = proc.wait()
            state.update(
                status="failed_pid_pgid_mismatch",
                returncode=rc,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                log_sha256=sha256_file(log_path),
            )
            atomic_json(state_path, state)
            raise ContractError(f"judge {seed} PID/PGID ownership mismatch; no signal sent")
        rc = proc.wait()
    state.update(
        status="process_complete_unvalidated" if rc == 0 else "failed",
        returncode=rc,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        log_sha256=sha256_file(log_path),
    )
    atomic_json(state_path, state)
    if rc != 0:
        raise ContractError(f"judge {seed} failed rc={rc}; preserved {state_path} and {log_path}")
    try:
        report = fresh.base.find_report(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    except ContractError as exc:
        state.update(status="report_discovery_failed", report_error=str(exc))
        atomic_json(state_path, state)
        raise ContractError(
            f"judge {seed} exited zero without one bound report; preserved state/log"
        ) from exc
    return {
        "state": state,
        "state_path": state_path,
        "log_path": log_path,
        "report": report,
    }


def _validate_exam_binding(
    validated: dict[str, Any],
    arm: dict[str, Any],
    prereg: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    if (
        validated.get("checkpoint_iteration") != 4000
        or validated.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or validated.get("training_contract_sha256") != arm["training_contract_sha256"]
        or validated.get("evaluation_contract_exact") is not True
        or validated.get("formal_target") is not True
        or validated.get("fresh_lineage") is not True
        or validated.get("denominators")
        != {"aggregate": 100, "forehand": 50, "backhand": 50}
        or validated.get("schedule_sha256") != EXPECTED_SCHEDULE["semantic_sha256"]
        or validated.get("question_id_order")
        != runtime["shared_schedule"]["question_id_order"]
        or canonical_sha256(validated["question_id_order"])
        != EXPECTED_SCHEDULE["question_id_order_sha256"]
        or validated.get("mjcf_sha256") != prereg["paper"]["mjcf_sha256"]
    ):
        raise ContractError("result violates checkpoint/fresh/exact/K100/MJCF binding")
    for key in ("execution_contract_sha256", "ready_state_sha256"):
        require_sha(f"result.{key}", validated.get(key))
    for key in ("report", "summary", "attempt_ledger"):
        meta = validated.get(key)
        if not isinstance(meta, dict) or set(meta) != {"path", "sha256"}:
            raise ContractError(f"result {key} binding missing")
        require_sha(f"result.{key}.sha256", meta["sha256"])
        path = Path(meta["path"])
        if not path.is_file() or sha256_file(path) != meta["sha256"]:
            raise ContractError(f"result {key} bytes changed/missing")
        require_under(f"result.{key}", path, Path(arm["checkpoint_path"]).parent / "judge")


def run_pod(
    config: dict[str, Any],
    queue_path: Path,
    queue: dict[str, Any],
    prereg_path: Path,
    prereg: dict[str, Any],
    activation: dict[str, Any],
    runtime_path: Path,
    runtime_sha: str,
    *,
    pod: str,
) -> int:
    runtime = _validate_runtime_contract(
        runtime_path,
        runtime_sha,
        config,
        queue_path,
        queue,
        prereg_path,
        prereg,
        activation,
        pod=pod,
    )
    _, _, tools, audits, _ = _validate_runtime_inputs(
        config,
        prereg,
        activation,
        pod=pod,
        schedule_path=Path(activation["content"]["shared_schedule"]["path"]),
    )
    state_dir = runtime_path.parent
    result_path = state_dir / config["runtime"]["pod_result_filename"]
    if result_path.exists():
        raise ContractError(f"no-clobber: Pod result exists: {result_path}")
    results: dict[str, Any] = {}
    execution: dict[str, Any] = {}
    for seed in POD_ARM_ORDER[pod]:
        arm = _bound_arm(prereg, activation, seed)
        launched = _run_judge(
            seed=seed,
            arm=arm,
            tools=tools,
            schedule_path=Path(runtime["shared_schedule"]["path"]),
            state_dir=state_dir,
            runtime_sha=runtime_sha,
            activation_sha=activation["sha256"],
            kit_boot_lock=config["runtime"]["kit_boot_lock"],
            gpu=arm["gpu"],
        )
        try:
            validated = fresh.validate_exam_result(
                report=launched["report"],
                arm_name="model_4000",
                arm=arm,
                prereg=prereg,
                runtime_contract=runtime,
            )
            _validate_exam_binding(validated, arm, prereg, runtime)
        except ContractError as exc:
            failed_state = dict(launched["state"])
            failed_state.update(
                status="validation_failed",
                validation_error=str(exc),
                log_sha256=sha256_file(launched["log_path"]),
            )
            atomic_json(launched["state_path"], failed_state)
            raise
        final_state = dict(launched["state"])
        final_state.update(
            status="validated_complete",
            report_sha256=validated["report"]["sha256"],
            summary_sha256=validated["summary"]["sha256"],
            attempt_ledger_sha256=validated["attempt_ledger"]["sha256"],
        )
        atomic_json(launched["state_path"], final_state)
        validated["checkpoint_audit"] = audits[seed]
        validated["raw_chain_revalidated_at_pod_run"] = True
        results[seed] = validated
        execution[seed] = {
            "mode": "fresh_identical_paper_judge",
            "seed1_reused": False,
            "state": {
                "path": str(launched["state_path"]),
                "sha256": sha256_file(launched["state_path"]),
            },
            "runner_log": {
                "path": str(launched["log_path"]),
                "sha256": sha256_file(launched["log_path"]),
            },
        }
    first, second = (results[seed] for seed in POD_ARM_ORDER[pod])
    for key in (
        "schedule_sha256", "question_id_order", "mjcf_sha256",
        "execution_contract_sha256", "ready_state_sha256",
    ):
        if first[key] != second[key]:
            raise ContractError(f"{pod} arms disagree on matched field {key}")
    content = {
        "contract_id": config["contract_id"],
        "pod": pod,
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_contract": {"path": str(runtime_path), "sha256": runtime_sha},
        "activation": {
            "path": str(activation["path"]),
            "sha256": activation["sha256"],
            "content_sha256": activation["content_sha256"],
            "barrier_id": activation["content"]["barrier_id"],
        },
        "config_sha256": sha256_file(Path(runtime["config"]["path"])),
        "queue_sha256": EXPECTED_QUEUE_SHA256,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "queue_validator_sha256": QUEUE_VALIDATOR_SHA256,
        "fresh_helper_sha256": FRESH_HELPER_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        **EXPECTED_SEMANTICS,
        "shared_schedule": runtime["shared_schedule"],
        "kit_boot_lock": config["runtime"]["kit_boot_lock"],
        "arm_order": list(POD_ARM_ORDER[pod]),
        "execution": execution,
        "arms": results,
        "gate_rule": EXPECTED_GATE_RULE,
        "interpretation_rule": EXPECTED_INTERPRETATION,
        "actions": {
            "training": "continue_all_arms_unmodified",
            "trainer_or_worker_signals": [],
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
        },
    }
    _write_no_clobber(
        result_path,
        _content_document("phase1_fresh_sz_model4000_q50_pod", content),
    )
    print(f"[model4000-q50] {pod} complete result={result_path} sha256={sha256_file(result_path)}")
    print("[model4000-q50] serial judges used shared Kit lock; no trainer/worker signal")
    return 0


def _validate_pod_result(
    path: Path,
    expected_sha: str,
    config: dict[str, Any],
    prereg: dict[str, Any],
    activation: dict[str, Any],
    expected_config_sha: str,
    *,
    pod: str,
) -> dict[str, Any]:
    require_sha(f"{pod} result SHA", expected_sha)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ContractError(f"{pod} result bytes changed/missing")
    document = _strict_json(path)
    exact_keys(
        document,
        {"schema_version", "artifact_kind", "content_sha256", "content"},
        f"{pod} result",
    )
    content = document["content"]
    expected_activation = {
        "path": str(activation["path"]),
        "sha256": activation["sha256"],
        "content_sha256": activation["content_sha256"],
        "barrier_id": activation["content"]["barrier_id"],
    }
    if (
        document["schema_version"] != 1
        or document["artifact_kind"] != "phase1_fresh_sz_model4000_q50_pod"
        or not isinstance(content, dict)
        or document["content_sha256"] != canonical_sha256(content)
        or content.get("contract_id") != config["contract_id"]
        or content.get("pod") != pod
        or content.get("status") != "complete"
        or content.get("activation") != expected_activation
        or content.get("config_sha256") != expected_config_sha
        or content.get("queue_sha256") != EXPECTED_QUEUE_SHA256
        or content.get("preregistration_sha256") != EXPECTED_PREREG_SHA256
        or content.get("queue_validator_sha256") != QUEUE_VALIDATOR_SHA256
        or content.get("fresh_helper_sha256") != FRESH_HELPER_SHA256
        or content.get("runner_sha256") != config["source_bindings"]["runner"]["sha256"]
        or content.get("arm_order") != list(POD_ARM_ORDER[pod])
        or content.get("gate_rule") != EXPECTED_GATE_RULE
        or content.get("interpretation_rule") != EXPECTED_INTERPRETATION
        or content.get("kit_boot_lock") != config["runtime"]["kit_boot_lock"]
        or any(content.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError(f"{pod} result is not the exact activation-bound artifact")
    if content.get("actions") != {
        "training": "continue_all_arms_unmodified",
        "trainer_or_worker_signals": [],
        "stop_or_promote_authorized": False,
        "deploy_or_real_robot_authorized": False,
    }:
        raise ContractError(f"{pod} result contains unauthorized actions")
    runtime_meta = content.get("runtime_contract")
    if not isinstance(runtime_meta, dict) or set(runtime_meta) != {"path", "sha256"}:
        raise ContractError(f"{pod} runtime-contract binding missing")
    require_absolute(f"{pod}.runtime_contract.path", runtime_meta["path"])
    require_sha(f"{pod}.runtime_contract.sha256", runtime_meta["sha256"])
    schedule_meta = content.get("shared_schedule")
    if (
        not isinstance(schedule_meta, dict)
        or schedule_meta.get("file_sha256") != EXPECTED_SCHEDULE["file_sha256"]
        or schedule_meta.get("schedule_sha256") != EXPECTED_SCHEDULE["semantic_sha256"]
        or canonical_sha256(schedule_meta.get("question_id_order", []))
        != EXPECTED_SCHEDULE["question_id_order_sha256"]
        or schedule_meta.get("question_id_order_sha256")
        != EXPECTED_SCHEDULE["question_id_order_sha256"]
        or schedule_meta.get("schedule_k") != 100
        or schedule_meta.get("attempts_per_side") != 50
        or schedule_meta.get("seed") != 0
        or schedule_meta.get("hold_range") != [0, 100]
    ):
        raise ContractError(f"{pod} result shared K100 runtime binding changed")
    if list(content.get("arms", {})) != list(POD_ARM_ORDER[pod]):
        raise ContractError(f"{pod} result seed coverage/order changed")
    for seed in POD_ARM_ORDER[pod]:
        arm = content["arms"][seed]
        frozen = activation["content"]["arms"][seed]
        if (
            arm.get("checkpoint_iteration") != 4000
            or arm.get("checkpoint_sha256") != frozen["checkpoint_sha256"]
            or arm.get("training_contract_sha256") != barrier.EXPECTED_HARD_CONTRACT_SHA256
            or arm.get("evaluation_contract_exact") is not True
            or arm.get("formal_target") is not True
            or arm.get("fresh_lineage") is not True
            or arm.get("denominators")
            != {"aggregate": 100, "forehand": 50, "backhand": 50}
            or arm.get("schedule_sha256") != EXPECTED_SCHEDULE["semantic_sha256"]
            or canonical_sha256(arm.get("question_id_order", []))
            != EXPECTED_SCHEDULE["question_id_order_sha256"]
            or arm.get("mjcf_sha256") != prereg["paper"]["mjcf_sha256"]
            or arm.get("checkpoint_audit") != frozen["checkpoint_audit"]
            or arm.get("raw_chain_revalidated_at_pod_run") is not True
        ):
            raise ContractError(f"{pod}/{seed} violates exact result binding")
        for key in ("execution_contract_sha256", "ready_state_sha256"):
            require_sha(f"{pod}/{seed}.{key}", arm.get(key))
        for key in ("report", "summary", "attempt_ledger"):
            meta = arm.get(key)
            if not isinstance(meta, dict) or set(meta) != {"path", "sha256"}:
                raise ContractError(f"{pod}/{seed} lacks {key} SHA binding")
            require_absolute(f"{pod}/{seed}.{key}.path", meta["path"])
            require_sha(f"{pod}/{seed}.{key}.sha256", meta["sha256"])
        counts = arm.get("returned_counts", {})
        if (
            any(
                isinstance(counts.get(key), bool)
                or not isinstance(counts.get(key), int)
                or counts[key] < 0
                for key in ("aggregate", "forehand", "backhand", "physical_falls")
            )
            or counts["aggregate"] != counts["forehand"] + counts["backhand"]
            or counts["forehand"] > 50
            or counts["backhand"] > 50
        ):
            raise ContractError(f"{pod}/{seed} returned counts invalid")
        if arm.get("returned_rates") != {
            "aggregate": counts["aggregate"] / 100.0,
            "forehand": counts["forehand"] / 50.0,
            "backhand": counts["backhand"] / 50.0,
        }:
            raise ContractError(f"{pod}/{seed} returned rates disagree with raw counts")
        execution = content.get("execution", {}).get(seed, {})
        if execution.get("mode") != "fresh_identical_paper_judge" or execution.get(
            "seed1_reused"
        ) is not False:
            raise ContractError(f"{pod}/{seed} was not freshly judged")
        for key in ("state", "runner_log"):
            meta = execution.get(key)
            if not isinstance(meta, dict) or set(meta) != {"path", "sha256"}:
                raise ContractError(f"{pod}/{seed} lacks execution {key}")
            require_absolute(f"{pod}/{seed}.execution.{key}.path", meta["path"])
            require_sha(f"{pod}/{seed}.execution.{key}.sha256", meta["sha256"])
    return content


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ContractError("cannot take median of empty results")
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def aggregate(
    config_path: Path,
    config: dict[str, Any],
    prereg_path: Path,
    prereg: dict[str, Any],
    activation: dict[str, Any],
    *,
    pod1_result: Path,
    pod1_sha: str,
    pod2_result: Path,
    pod2_sha: str,
    output_dir: Path,
) -> Path:
    expected_output_dir = Path(config["runtime"]["aggregate_output_dir"])
    if output_dir != expected_output_dir:
        raise ContractError(f"aggregate must use configured output dir {expected_output_dir}")
    pod1 = _validate_pod_result(
        pod1_result,
        pod1_sha,
        config,
        prereg,
        activation,
        sha256_file(config_path),
        pod="pod1",
    )
    pod2 = _validate_pod_result(
        pod2_result,
        pod2_sha,
        config,
        prereg,
        activation,
        sha256_file(config_path),
        pod="pod2",
    )
    arms = {**pod1["arms"], **pod2["arms"]}
    if list(sorted(arms, key=lambda value: int(value[4:]))) != list(SEED_ORDER):
        raise ContractError("Pod results do not cover seed1..seed4 exactly")
    reference = arms["seed1"]
    for seed, arm in arms.items():
        for key in (
            "schedule_sha256", "question_id_order", "mjcf_sha256",
            "execution_contract_sha256", "ready_state_sha256",
        ):
            if arm[key] != reference[key]:
                raise ContractError(f"{seed} differs on matched paper/runtime field {key}")
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
    spread = max(values) - minimum
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
    mechanical_thresholds_pass = all(check["pass"] for check in checks.values())
    seed4_aggregate = aggregate_rates["seed4"]
    seed4_min_side = min(side_rates["seed4"].values())
    seed4_delayed = (
        seed4_aggregate
        >= EXPECTED_INTERPRETATION["seed4_delayed_learning_supported_only_if"][
            "aggregate_rate_min"
        ]
        and seed4_min_side
        >= EXPECTED_INTERPRETATION["seed4_delayed_learning_supported_only_if"][
            "every_side_rate_min"
        ]
    )
    gate_pass = mechanical_thresholds_pass and EXPECTED_INTERPRETATION[
        "family_stable_claim_allowed"
    ]
    if gate_pass:
        raise ContractError("known seed1 evidence forbids a model4000 family-stable PASS")
    content = {
        "contract_id": config["contract_id"],
        "status": "fail_seed_stability_checkpoint_evidence_known_seed1_model4000",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "activation": {
            "path": str(activation["path"]),
            "sha256": activation["sha256"],
            "content_sha256": activation["content_sha256"],
            "barrier_id": activation["content"]["barrier_id"],
        },
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
        "arms": {seed: arms[seed] for seed in SEED_ORDER},
        "aggregate_return_rates": aggregate_rates,
        "side_return_rates": side_rates,
        "gate_rule": EXPECTED_GATE_RULE,
        "gate_checks": checks,
        "mechanical_thresholds_pass_on_rerun": mechanical_thresholds_pass,
        "known_before_prereg": EXPECTED_INTERPRETATION["known_seed1_model4000"],
        "family_stable_claim_allowed": False,
        "gate_pass": False,
        "seed4_interpretation": {
            "aggregate_return_rate": seed4_aggregate,
            "minimum_side_return_rate": seed4_min_side,
            "classification": (
                "delayed_learning_supported_at_model4000"
                if seed4_delayed
                else "persistent_weakness_through_model4000"
            ),
            "thresholds_unchanged": True,
            "family_stability_implication": "none_known_seed1_forbids_family_pass",
        },
        "actions": {
            "training": "continue_all_arms_unmodified",
            "seed_stability_gate": "keep_open",
            "trainer_or_worker_signals": [],
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
            "vendor_gate3_gate3b": "remain_open",
        },
    }
    document = _content_document("phase1_fresh_sz_model4000_q50_aggregate", content)
    output = output_dir / (
        f"phase1_fresh_SZ_model4000_seed_stability_q50_{document['content_sha256']}.json"
    )
    _write_no_clobber(output, document)
    print(f"[model4000-q50] aggregate={output} gate_pass=false")
    print(
        "[model4000-q50] seed4="
        f"{content['seed4_interpretation']['classification']}; all arms continue"
    )
    return output


def _load_bound_inputs(
    args: argparse.Namespace,
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    config_path = args.config.resolve()
    require_sha("expected config SHA", args.expected_config_sha256)
    if not config_path.is_file() or sha256_file(config_path) != args.expected_config_sha256:
        raise ContractError("execution config file SHA mismatch")
    config = load_execution_config(config_path)
    if sha256_file(Path(__file__).resolve()) != config["source_bindings"]["runner"]["sha256"]:
        raise ContractError("runner bytes differ from execution config")
    queue_path, queue, prereg_path, prereg = _resolve_bound_sources(config_path, config)
    activation = _validate_activation_document(
        args.activation.resolve(),
        args.expected_activation_sha256,
        queue_path,
        queue,
        prereg_path,
    )
    return config_path, config, queue_path, queue, prereg_path, prereg, activation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--expected-activation-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--pod", choices=tuple(POD_ARM_ORDER), required=True)
    check.add_argument("--schedule-source", required=True, type=Path)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--pod", choices=tuple(POD_ARM_ORDER), required=True)
    prepare_parser.add_argument("--schedule-source", required=True, type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--pod", choices=tuple(POD_ARM_ORDER), required=True)
    run_parser.add_argument("--runtime-contract", required=True, type=Path)
    run_parser.add_argument("--expected-runtime-contract-sha256", required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--pod1-result", required=True, type=Path)
    aggregate_parser.add_argument("--pod1-result-sha256", required=True)
    aggregate_parser.add_argument("--pod2-result", required=True, type=Path)
    aggregate_parser.add_argument("--pod2-result-sha256", required=True)
    aggregate_parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    (
        config_path,
        config,
        queue_path,
        queue,
        prereg_path,
        prereg,
        activation,
    ) = _load_bound_inputs(args)
    if args.command == "contract-check":
        _validate_runtime_inputs(
            config,
            prereg,
            activation,
            pod=args.pod,
            schedule_path=args.schedule_source.resolve(),
        )
        print(f"[model4000-q50] {args.pod} contract check PASS; no write/judge/signal")
        return 0
    if args.command == "prepare":
        return prepare(
            config_path,
            config,
            queue_path,
            prereg_path,
            prereg,
            activation,
            pod=args.pod,
            schedule_source=args.schedule_source.resolve(),
        )
    if args.command == "run":
        return run_pod(
            config,
            queue_path,
            queue,
            prereg_path,
            prereg,
            activation,
            args.runtime_contract.resolve(),
            args.expected_runtime_contract_sha256,
            pod=args.pod,
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
        activation,
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
        print(f"[model4000-q50][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
