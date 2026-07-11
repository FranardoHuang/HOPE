#!/usr/bin/env python3
"""Fail-closed queue barrier for the fresh-SZ model_4000 four-seed q50 paper.

This program deliberately has no judge, SSH, process-launch, or signal command.  It can
validate the committed paper, create one read-only readiness audit per Pod, and combine the
two audits into a content-addressed all-four-ready activation artifact.  A later q50 runner
must consume that activation artifact; the queue itself cannot start evaluation runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
FRESH_VALIDATOR_PATH = SCRIPT_DIR / "run_phase1_fresh_exact_paired_bank_q50.py"
FRESH_VALIDATOR_SHA256 = "3528250777a170791f39d8dd17716c2a7f8ca91416a3ffa8433ec5eb691ed9e0"


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if (
    not FRESH_VALIDATOR_PATH.is_file()
    or _bootstrap_sha256(FRESH_VALIDATOR_PATH) != FRESH_VALIDATOR_SHA256
):
    raise RuntimeError("refusing to import changed fresh exact q50 validator")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
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

SEED_ORDER = ("seed1", "seed2", "seed3", "seed4")
POD_ARM_ORDER = {"pod1": ("seed1", "seed3"), "pod2": ("seed2", "seed4")}
EXPECTED_TRAIN_COMMIT = "6d93bcb16c422a2f42748c2dc99432559653480b"
EXPECTED_EVAL_COMMIT = "46a0ce24524fdb843e55fe82ba4c045f2adc090f"
EXPECTED_HARD_CONTRACT_SHA256 = (
    "3a3b3d956e19d47f7e6f0a157159dc96c8f09d8345c436a776c8c7e99c0b9972"
)
EXPECTED_FAMILY = {
    "cell": "SZ",
    "name": "v4rg_runtime_order_v3",
    "source_family_sha256": (
        "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5"
    ),
    "face_command_pairing": "shared_plus_y",
    "zero_joint_friction": True,
    "checkpoint_iteration": 4000,
}
EXPECTED_SCHEDULE = {
    "schema_version": 3,
    "file_sha256": "66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3",
    "semantic_sha256": "7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e",
    "question_id_order_sha256": (
        "b87e81a34ff2d31766e17345f0a8c9d77665b78874093e26bdae257e8ed21f91"
    ),
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
    "pass_action": "record_model_4000_seed_stability_only_continue_all_arms_unmodified",
    "stop_or_promote_authorized": False,
    "deploy_or_real_robot_authorized": False,
}
EXPECTED_SEMANTICS = {
    "fresh_lineage": True,
    "evaluation_contract_exact": True,
    "formal_target": True,
    "purpose": "four_seed_model_4000_stability_checkpoint_evidence",
    "training_mutation_allowed": False,
    "trainer_or_worker_signal_allowed": False,
    "whole_arm_stop_allowed": False,
    "whole_arm_promote_allowed": False,
    "deploy_gate": False,
    "real_robot_authorized": False,
}
EXPECTED_ARM_PATHS = {
    "seed1": (
        "/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/"
        "agibot_a3_hope_virtualball/2026-07-11_00-52-16_phase1_fresh_v3_S1_seed1"
    ),
    "seed2": (
        "/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/"
        "agibot_a3_hope_virtualball/2026-07-11_00-54-53_phase1_fresh_v3_S1_seed2"
    ),
    "seed3": (
        "/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/"
        "agibot_a3_hope_virtualball/2026-07-11_05-39-31_phase1_fresh_v3_SZ_seed3"
    ),
    "seed4": (
        "/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/"
        "agibot_a3_hope_virtualball/2026-07-11_05-41-44_phase1_fresh_v3_SZ_seed4"
    ),
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


def validate_queue(data: dict[str, Any]) -> None:
    exact_keys(
        data,
        {
            "schema_version", "queue_id", "created_utc", "status", "auto_start",
            "runtime_entrypoint", "preregistration", "validator", "checkouts",
            "checkpoint_python", "paper", "pod_arm_order", "pod_audit_outputs",
            "activation_output_dir", "barrier", "seed1_reuse", "formal_semantics",
        },
        "model4000 q50 queue",
    )
    if (
        data["schema_version"] != 1
        or data["status"] != "offline_preregistered_waiting_all_four_checkpoint_audits"
        or data["auto_start"] is not False
        or data["runtime_entrypoint"] is not None
    ):
        raise ContractError("queue must remain offline, non-starting, and runtime-free")
    exact_keys(data["preregistration"], {"path", "sha256"}, "preregistration")
    exact_keys(data["validator"], {"path", "sha256"}, "validator")
    require_sha("preregistration.sha256", data["preregistration"]["sha256"])
    require_sha("validator.sha256", data["validator"]["sha256"])
    if data["validator"]["path"] != "scripts/validate_phase1_fresh_sz_model4000_q50_queue.py":
        raise ContractError("queue validator path changed")
    if data["paper"] != EXPECTED_SCHEDULE:
        raise ContractError("queue does not bind the immutable 2k K100 bytes and semantics")
    if data["pod_arm_order"] != {key: list(value) for key, value in POD_ARM_ORDER.items()}:
        raise ContractError("Pod/seed partition changed")
    if data["formal_semantics"] != EXPECTED_SEMANTICS:
        raise ContractError("queue formal semantics changed")
    checkouts = data["checkouts"]
    if checkouts != {
        "training": {"path": "/workspace/codexschema/nohope", "commit": EXPECTED_TRAIN_COMMIT},
        "evaluation": {
            "path": "/workspace/codexschema/nohope_eval_08e438e",
            "commit": EXPECTED_EVAL_COMMIT,
        },
    }:
        raise ContractError("queue checkout binding changed")
    checkpoint_python = require_absolute("checkpoint_python", data["checkpoint_python"])
    if checkpoint_python.name != "python":
        raise ContractError("checkpoint audit interpreter path changed")
    if data["pod_audit_outputs"] != {
        "pod1": (
            "/workspace/codexschema/phase1_fresh_20260711/q50/"
            "fresh_SZ_model4000_seed_stability_q50_barrier_v1/pod1_ready_audit.json"
        ),
        "pod2": (
            "/workspace/codexschema/phase1_fresh_20260711/q50/"
            "fresh_SZ_model4000_seed_stability_q50_barrier_v1/pod2_ready_audit.json"
        ),
    }:
        raise ContractError("Pod audit output paths changed")
    require_absolute("activation_output_dir", data["activation_output_dir"])
    barrier = data["barrier"]
    exact_keys(
        barrier,
        {
            "id", "required_seed_coverage", "checkpoint_iteration", "all_finite",
            "filename_matches_embedded_iteration", "same_hard_contract_sha256",
            "fresh_lineage_exact", "schedule_bytes_identical", "runtime_before_pass_allowed",
            "trainer_or_worker_signal_allowed", "activation_artifact_required_by_future_runner",
        },
        "barrier",
    )
    if barrier != {
        "id": "fresh_SZ_model4000_all_four_ready_v1",
        "required_seed_coverage": list(SEED_ORDER),
        "checkpoint_iteration": 4000,
        "all_finite": True,
        "filename_matches_embedded_iteration": True,
        "same_hard_contract_sha256": EXPECTED_HARD_CONTRACT_SHA256,
        "fresh_lineage_exact": True,
        "schedule_bytes_identical": True,
        "runtime_before_pass_allowed": False,
        "trainer_or_worker_signal_allowed": False,
        "activation_artifact_required_by_future_runner": True,
    }:
        raise ContractError("all-four barrier was weakened")
    reuse = data["seed1_reuse"]
    exact_keys(reuse, {"role", "source_result", "expected_checkpoint_sha256"}, "seed1_reuse")
    if (
        reuse["role"] != "candidate_only_after_full_raw_chain_revalidation"
        or reuse["source_result"] != {
            "path": "configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json",
            "sha256": "19d43fd0c507dd0aff985930b3c375cf1b72b1d8eff56f0d4a9934bdf09beeba",
        }
        or reuse["expected_checkpoint_sha256"]
        != "1a8fcf3de81775fdd003a8bbdcfd371c05b7168353daf19a8290a6d7c85e9071"
    ):
        raise ContractError("seed1 reuse chain changed")


def validate_preregistration(data: dict[str, Any], queue: dict[str, Any]) -> None:
    exact_keys(
        data,
        {
            "schema_version", "preregistration_id", "created_utc", "status",
            "auto_activate", "jobs_started", "runtime_state", "scope", "known_before_prereg",
            "question", "family", "training_commit", "eval_commit", "tools", "paper",
            "arms", "gate_rule", "interpretation_rule", "formal_semantics", "activation",
            "preflight_requirements",
        },
        "model4000 q50 preregistration",
    )
    if (
        data["schema_version"] != 1
        or data["status"] != "preregistered_waiting_all_four_checkpoint_audits"
        or data["auto_activate"] is not False
        or data["jobs_started"] != 0
        or data["runtime_state"] is not None
    ):
        raise ContractError("preregistration is not pristine and offline")
    if data["family"] != EXPECTED_FAMILY:
        raise ContractError("fresh SZ model_4000 family binding changed")
    if data["training_commit"] != EXPECTED_TRAIN_COMMIT or data["eval_commit"] != EXPECTED_EVAL_COMMIT:
        raise ContractError("preregistration checkout commits changed")
    if data["formal_semantics"] != EXPECTED_SEMANTICS or data["gate_rule"] != EXPECTED_GATE_RULE:
        raise ContractError("formal semantics or 2k stability thresholds changed")
    tools = data["tools"]
    if tools != {
        "queue_validator_sha256": queue["validator"]["sha256"],
        "fresh_exact_validator_sha256": FRESH_VALIDATOR_SHA256,
        "judge_sha256": "1a00702935096b063435c3f0bd23e75f76f13e1298c87310d1cec3c26cca8529",
        "mujoco_evaluator_sha256": (
            "e4a9fa42ff0f7e68cebdf16f2e0c61299507496c6edc13546baeb3d576ecb20a"
        ),
    }:
        raise ContractError("preregistered tool bytes changed")
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
        paper["role"] != "four-seed fresh SZ model_4000 matched stability q50"
        or paper["schedule"] != EXPECTED_SCHEDULE
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
        or paper["mjcf_sha256"]
        != "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
    ):
        raise ContractError("formal matched K100 paper changed")
    bank = paper["exam_bank"]
    if bank != {
        "path": (
            "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/"
            "s1_v4rg_runtime_order_schema3_exam.npz"
        ),
        "bytes": 63968,
        "sha256": "d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096",
        "schema_version": 3,
        "source_family_sha256": EXPECTED_FAMILY["source_family_sha256"],
    }:
        raise ContractError("exam bank bytes/family changed")
    independence = paper["reuse_independence"]
    if independence != {
        "source_model2000_preregistration": {
            "path": "configs/phase1_fresh_SZ_model2000_seed_stability_q50_prereg_20260711.json",
            "sha256": "cf3ce857b1b1a688e808e4a85ea662b252e84d2ca32240c29ee1b08009aaeabd",
        },
        "source_model2000_result": {
            "path": (
                "configs/phase1_fresh_SZ_model2000_seed_stability_q50_"
                "a756bf1d0e76d1016992ae241b935cf92b3c84ffd55fe503e7c199626d9c8ffd.json"
            ),
            "file_sha256": "d856468fb93461be52498a24655b25993ce28f530f5989e61110e33421736e43",
            "content_sha256": "a756bf1d0e76d1016992ae241b935cf92b3c84ffd55fe503e7c199626d9c8ffd",
        },
        "materialized_before_model2000_or_model4000_four_seed_outcomes": True,
        "new_materialization_forbidden": True,
        "same_exact_bytes_required": True,
    }:
        raise ContractError("paper independence or 2k result chain changed")
    if list(data["arms"]) != list(SEED_ORDER):
        raise ContractError("arms must remain seed1..seed4 ordered")
    for ordinal, seed in enumerate(SEED_ORDER, start=1):
        arm = data["arms"][seed]
        exact_keys(
            arm,
            {
                "training_seed", "pod", "gpu", "run_name", "checkpoint_iteration",
                "checkpoint_path", "checkpoint_sha256", "checkpoint_binding_stage",
                "training_contract_path", "training_contract_sha256",
                "checkpoint_embedded_training_contract_sha256", "cell",
                "face_command_pairing", "zero_joint_friction", "lineage_exact",
                "execution_mode", "job_status", "pid", "pgid", "result",
            },
            f"arms.{seed}",
        )
        expected_pod = "pod1" if seed in POD_ARM_ORDER["pod1"] else "pod2"
        run_dir = Path(EXPECTED_ARM_PATHS[seed])
        if (
            arm["training_seed"] != ordinal
            or arm["pod"] != expected_pod
            or isinstance(arm["gpu"], bool)
            or not isinstance(arm["gpu"], int)
            or arm["gpu"] < 0
            or arm["checkpoint_iteration"] != 4000
            or arm["checkpoint_path"] != str(run_dir / "model_4000.pt")
            or arm["checkpoint_sha256"] is not None
            or arm["checkpoint_binding_stage"]
            != "discover_and_freeze_only_in_all_four_activation_artifact"
            or arm["training_contract_path"] != str(run_dir / "params/training_contract.json")
            or arm["training_contract_sha256"] != EXPECTED_HARD_CONTRACT_SHA256
            or arm["checkpoint_embedded_training_contract_sha256"]
            != EXPECTED_HARD_CONTRACT_SHA256
            or arm["cell"] != "SZ"
            or arm["face_command_pairing"] != "shared_plus_y"
            or arm["zero_joint_friction"] is not True
            or arm["lineage_exact"] is not True
            or arm["job_status"] != "not_started"
            or arm["pid"] is not None
            or arm["pgid"] is not None
            or arm["result"] is not None
        ):
            raise ContractError(f"{seed} changed or was prematurely activated")
        expected_mode = "reuse_candidate_after_full_revalidation" if seed == "seed1" else "judge_required"
        if arm["execution_mode"] != expected_mode:
            raise ContractError(f"{seed} execution mode changed")
    known = data["known_before_prereg"]
    exact_keys(known, {"model2000_four_seed", "model4000_seed1"}, "known_before_prereg")
    if known["model2000_four_seed"] != {
        "aggregate_return_rates": {"seed1": 0.83, "seed2": 1.0, "seed3": 1.0, "seed4": 0.2},
        "gate_pass": False,
        "status": "fail_seed_stability_checkpoint_evidence",
    }:
        raise ContractError("2k trigger evidence changed")
    if known["model4000_seed1"] != {
        "aggregate_return_rate": 0.5,
        "forehand_return_rate": 0.0,
        "backhand_return_rate": 1.0,
        "consequence": "four_seed_overall_stability_gate_is_already_mathematically_unpassable",
    }:
        raise ContractError("known seed1 model_4000 evidence was hidden or changed")
    interpretation = data["interpretation_rule"]
    if interpretation != {
        "seed4_delayed_learning_supported_only_if": {
            "aggregate_rate_min": 0.65,
            "every_side_rate_min": 0.50,
            "threshold_source": "unchanged_model2000_stability_gate",
        },
        "seed4_persistent_weakness_if": "either_unchanged_seed_level_threshold_fails",
        "family_stable_claim_allowed": False,
        "reason_family_claim_forbidden": "known_seed1_model4000_rate_0.50_is_below_0.65",
    }:
        raise ContractError("late-learning interpretation or thresholds changed")
    if data["activation"] != {
        "preregistered": True,
        "all_four_ready_artifact_required": True,
        "runtime_authorized_at_creation": False,
        "started_at_creation": False,
    }:
        raise ContractError("activation fence changed")


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
        or len(order) != 100
    ):
        raise ContractError("shared K100 schedule semantics/order changed")
    return schedule


def _validate_checkout_and_static_inputs(
    queue: dict[str, Any], prereg: dict[str, Any], *, schedule_path: Path
) -> tuple[Path, dict[str, Any]]:
    training_root = fresh.validate_checkout("training", queue["checkouts"]["training"])
    fresh.validate_checkout("evaluation", queue["checkouts"]["evaluation"])
    schedule = validate_schedule(schedule_path, prereg)
    bank = prereg["paper"]["exam_bank"]
    bank_path = Path(bank["path"])
    if (
        not bank_path.is_file()
        or bank_path.stat().st_size != bank["bytes"]
        or sha256_file(bank_path) != bank["sha256"]
    ):
        raise ContractError("exact exam-bank bytes changed")
    python = Path(queue["checkpoint_python"])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ContractError("checkpoint audit Python is missing/not executable")
    return training_root, schedule


def _validate_checkpoint_audit(audit: Mapping[str, Any]) -> None:
    if (
        audit.get("iter") != 4000
        or audit.get("training_contract_sha256") != EXPECTED_HARD_CONTRACT_SHA256
        or audit.get("training_contract_schema_version") != 3
        or audit.get("training_contract_lineage_exact") not in (1, True)
        or isinstance(audit.get("tensor_count"), bool)
        or not isinstance(audit.get("tensor_count"), int)
        or audit["tensor_count"] <= 0
        or isinstance(audit.get("floating_elements"), bool)
        or not isinstance(audit.get("floating_elements"), int)
        or audit["floating_elements"] <= 0
        or audit.get("nonfinite") != 0
    ):
        raise ContractError(f"model_4000 checkpoint audit violates finite exact lineage: {audit}")


def audit_pod(
    queue_path: Path,
    queue: dict[str, Any],
    prereg_path: Path,
    prereg: dict[str, Any],
    *,
    pod: str,
    schedule_path: Path,
    output: Path,
) -> Path:
    if pod not in POD_ARM_ORDER:
        raise ContractError(f"unknown Pod {pod!r}")
    expected_output = Path(queue["pod_audit_outputs"][pod])
    if output != expected_output:
        raise ContractError(f"{pod} audit must use preregistered output path {expected_output}")
    training_root, schedule = _validate_checkout_and_static_inputs(
        queue, prereg, schedule_path=schedule_path
    )
    arms: dict[str, Any] = {}
    hard_seen: dict[str, Any] | None = None
    checkpoint_python = Path(queue["checkpoint_python"])
    for seed in POD_ARM_ORDER[pod]:
        arm = prereg["arms"][seed]
        checkpoint = Path(arm["checkpoint_path"])
        hard_path = Path(arm["training_contract_path"])
        require_under(f"{seed}.checkpoint", checkpoint, training_root)
        require_under(f"{seed}.hard_contract", hard_path, training_root)
        if checkpoint.name != "model_4000.pt" or not checkpoint.is_file():
            raise ContractError(f"{seed} model_4000 checkpoint is not ready")
        checkpoint_sha = sha256_file(checkpoint)
        require_sha(f"{seed}.checkpoint_sha256", checkpoint_sha)
        if (
            seed == "seed1"
            and checkpoint_sha != queue["seed1_reuse"]["expected_checkpoint_sha256"]
        ):
            raise ContractError("seed1 model_4000 bytes differ from the known reusable checkpoint")
        if not hard_path.is_file() or sha256_file(hard_path) != EXPECTED_HARD_CONTRACT_SHA256:
            raise ContractError(f"{seed} adjacent hard-contract bytes changed/missing")
        hard = _strict_json(hard_path)
        fresh.validate_hard_contract(hard, prereg)
        if hard_seen is None:
            hard_seen = hard
        elif hard != hard_seen:
            raise ContractError(f"{pod} arms do not share byte-identical hard semantics")
        checkpoint_audit = fresh.checkpoint_audit(
            checkpoint_python,
            checkpoint,
            expected_iteration=4000,
            expected_contract_sha=EXPECTED_HARD_CONTRACT_SHA256,
        )
        _validate_checkpoint_audit(checkpoint_audit)
        arms[seed] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "training_contract_path": str(hard_path),
            "training_contract_sha256": EXPECTED_HARD_CONTRACT_SHA256,
            "checkpoint_audit": checkpoint_audit,
        }
    content = {
        "queue_id": queue["queue_id"],
        "status": "pod_checkpoints_ready_judge_not_started",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pod": pod,
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        **EXPECTED_SEMANTICS,
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": sha256_file(schedule_path),
            "semantic_sha256": schedule["schedule_sha256"],
            "question_id_order_sha256": canonical_sha256(
                [item["question_id"] for item in schedule["items"]]
            ),
        },
        "arm_order": list(POD_ARM_ORDER[pod]),
        "arms": arms,
        "actions": {
            "judges_started": 0,
            "trainer_or_worker_signals": [],
            "runtime_authorized_by_this_pod_audit": False,
            "real_robot_authorized": False,
        },
    }
    document = _content_document("phase1_fresh_sz_model4000_q50_pod_ready_audit", content)
    _write_no_clobber(output, document)
    print(f"[model4000-q50-barrier] {pod} ready audit={output} sha256={sha256_file(output)}")
    print("[model4000-q50-barrier] no judge/runtime/signal; both Pod audits still required")
    return output


def validate_pod_audit(
    path: Path,
    expected_sha: str,
    queue_path: Path,
    queue: dict[str, Any],
    prereg_path: Path,
    *,
    pod: str,
) -> dict[str, Any]:
    require_sha(f"{pod} audit SHA", expected_sha)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ContractError(f"{pod} audit bytes changed/missing")
    document = _strict_json(path)
    exact_keys(document, {"schema_version", "artifact_kind", "content_sha256", "content"}, pod)
    content = document["content"]
    if (
        document["schema_version"] != 1
        or document["artifact_kind"] != "phase1_fresh_sz_model4000_q50_pod_ready_audit"
        or not isinstance(content, dict)
        or document["content_sha256"] != canonical_sha256(content)
        or content.get("queue_id") != queue["queue_id"]
        or content.get("status") != "pod_checkpoints_ready_judge_not_started"
        or content.get("pod") != pod
        or content.get("queue")
        != {"path": str(queue_path), "sha256": sha256_file(queue_path)}
        or content.get("preregistration")
        != {"path": str(prereg_path), "sha256": sha256_file(prereg_path)}
        or content.get("validator_sha256") != sha256_file(Path(__file__).resolve())
        or content.get("arm_order") != list(POD_ARM_ORDER[pod])
        or any(content.get(key) != value for key, value in EXPECTED_SEMANTICS.items())
    ):
        raise ContractError(f"{pod} audit is not a bound ready artifact")
    if content.get("shared_schedule", {}).get("file_sha256") != EXPECTED_SCHEDULE["file_sha256"]:
        raise ContractError(f"{pod} audit changed the immutable schedule bytes")
    if (
        content["shared_schedule"].get("semantic_sha256")
        != EXPECTED_SCHEDULE["semantic_sha256"]
        or content["shared_schedule"].get("question_id_order_sha256")
        != EXPECTED_SCHEDULE["question_id_order_sha256"]
    ):
        raise ContractError(f"{pod} audit changed schedule semantics/order")
    if content.get("actions") != {
        "judges_started": 0,
        "trainer_or_worker_signals": [],
        "runtime_authorized_by_this_pod_audit": False,
        "real_robot_authorized": False,
    }:
        raise ContractError(f"{pod} audit contains an unauthorized action")
    arms = content.get("arms")
    if not isinstance(arms, dict) or list(arms) != list(POD_ARM_ORDER[pod]):
        raise ContractError(f"{pod} audit seed coverage is incomplete/reordered")
    for seed in POD_ARM_ORDER[pod]:
        arm = arms[seed]
        prereg_arm = _strict_json(prereg_path)["arms"][seed]
        if (
            arm.get("checkpoint_path") != prereg_arm["checkpoint_path"]
            or arm.get("training_contract_path") != prereg_arm["training_contract_path"]
            or arm.get("training_contract_sha256") != EXPECTED_HARD_CONTRACT_SHA256
        ):
            raise ContractError(f"{pod}/{seed} path or hard-contract binding changed")
        require_sha(f"{pod}/{seed}.checkpoint_sha256", arm.get("checkpoint_sha256"))
        audit = arm.get("checkpoint_audit")
        if not isinstance(audit, dict):
            raise ContractError(f"{pod}/{seed} checkpoint audit missing")
        _validate_checkpoint_audit(audit)
    return content


def activate(
    queue_path: Path,
    queue: dict[str, Any],
    prereg_path: Path,
    *,
    pod1_audit: Path,
    pod1_sha: str,
    pod2_audit: Path,
    pod2_sha: str,
    output_dir: Path,
) -> Path:
    if output_dir != Path(queue["activation_output_dir"]):
        raise ContractError("activation must use the preregistered output directory")
    pod1 = validate_pod_audit(
        pod1_audit, pod1_sha, queue_path, queue, prereg_path, pod="pod1"
    )
    pod2 = validate_pod_audit(
        pod2_audit, pod2_sha, queue_path, queue, prereg_path, pod="pod2"
    )
    if pod1["shared_schedule"] != pod2["shared_schedule"]:
        raise ContractError("Pod audits do not bind the same schedule path and bytes")
    arms = {**pod1["arms"], **pod2["arms"]}
    if list(sorted(arms, key=lambda value: int(value[4:]))) != list(SEED_ORDER):
        raise ContractError("all-four barrier does not cover seed1..seed4 exactly")
    content = {
        "queue_id": queue["queue_id"],
        "barrier_id": queue["barrier"]["id"],
        "status": "all_four_checkpoints_ready_judge_not_started",
        "activated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "pod_audits": {
            "pod1": {"path": str(pod1_audit), "sha256": pod1_sha},
            "pod2": {"path": str(pod2_audit), "sha256": pod2_sha},
        },
        **EXPECTED_SEMANTICS,
        "shared_schedule": pod1["shared_schedule"],
        "seed_order": list(SEED_ORDER),
        "arms": {seed: arms[seed] for seed in SEED_ORDER},
        "gate_rule": EXPECTED_GATE_RULE,
        "actions": {
            "judges_started": 0,
            "trainer_or_worker_signals": [],
            "future_q50_runner_may_prepare_only_with_this_exact_artifact": True,
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
        },
    }
    document = _content_document("phase1_fresh_sz_model4000_q50_all_four_activation", content)
    output = output_dir / f"activation_{document['content_sha256']}.json"
    _write_no_clobber(output, document)
    print(f"[model4000-q50-barrier] all-four PASS activation={output}")
    print("[model4000-q50-barrier] judges_started=0; activation is mandatory for a future runner")
    return output


def _load_bound_inputs(
    queue_path: Path, expected_queue_sha: str
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    queue_path = queue_path.resolve()
    require_sha("expected queue SHA", expected_queue_sha)
    if not queue_path.is_file() or sha256_file(queue_path) != expected_queue_sha:
        raise ContractError("queue file SHA mismatch")
    queue = _strict_json(queue_path)
    validate_queue(queue)
    validator_path = (queue_path.parent.parent / queue["validator"]["path"]).resolve()
    if validator_path != Path(__file__).resolve() or sha256_file(validator_path) != queue["validator"]["sha256"]:
        raise ContractError("queue validator bytes changed")
    prereg_path = (queue_path.parent.parent / queue["preregistration"]["path"]).resolve()
    if not prereg_path.is_file() or sha256_file(prereg_path) != queue["preregistration"]["sha256"]:
        raise ContractError("preregistration bytes changed")
    prereg = _strict_json(prereg_path)
    validate_preregistration(prereg, queue)
    return queue_path, queue, prereg_path, prereg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--expected-queue-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    audit = sub.add_parser("audit-pod")
    audit.add_argument("--pod", required=True, choices=tuple(POD_ARM_ORDER))
    audit.add_argument("--schedule-source", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    activation = sub.add_parser("activate")
    activation.add_argument("--pod1-audit", required=True, type=Path)
    activation.add_argument("--pod1-audit-sha256", required=True)
    activation.add_argument("--pod2-audit", required=True, type=Path)
    activation.add_argument("--pod2-audit-sha256", required=True)
    activation.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    queue_path, queue, prereg_path, prereg = _load_bound_inputs(
        args.queue, args.expected_queue_sha256
    )
    if args.command == "validate-config":
        print("[model4000-q50-barrier] committed queue/preregistration PASS; no runtime")
        return 0
    if args.command == "audit-pod":
        audit_pod(
            queue_path,
            queue,
            prereg_path,
            prereg,
            pod=args.pod,
            schedule_path=args.schedule_source.resolve(),
            output=args.output.resolve(),
        )
        return 0
    activate(
        queue_path,
        queue,
        prereg_path,
        pod1_audit=args.pod1_audit.resolve(),
        pod1_sha=args.pod1_audit_sha256,
        pod2_audit=args.pod2_audit.resolve(),
        pod2_sha=args.pod2_audit_sha256,
        output_dir=args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[model4000-q50-barrier][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
