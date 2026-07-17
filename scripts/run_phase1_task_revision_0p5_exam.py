#!/usr/bin/env python3
"""Launch or inspect one activation-bound 0.5-second Isaac K100 exam.

The exam is deliberately diagnostic-only.  ``launch`` performs a bounded two-party
handshake and returns after a detached, content-bound supervisor acknowledges the
commit.  ``inspect`` is read-only.  The supervisor owns the sole evaluator process,
writes periodic heartbeats, applies a total deadline, and can signal only that exact
PID/PGID after revalidating its Linux process identity.  It never signals a trainer,
worker, robot, or an unbound process, and it never retries.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping
import zlib


QUEUE = Path("configs/phase1_task_revision_supercombo_20260716.yaml")
HISTORICAL_ACTIVATION_V1 = Path(
    "configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json"
)
HISTORICAL_ACTIVATION_V2 = Path(
    "configs/phase1_task_revision_0p5_exam_activation_v2_20260717.json"
)
ACTIVATION = Path("configs/phase1_task_revision_0p5_exam_activation_v3_20260717.json")
V1_FAILURE_RECEIPT = Path(
    "configs/phase1_task_revision_0p5_exam_v1_failure_20260717.json"
)
V1_ACTIVATION_SHA256 = (
    "996775d6c64a75d4c626d60da20fc52ec27ca86548008aeac900c380de87cfb6"
)
V1_HARNESS_SHA256 = (
    "c2ce27845cb26a1ff2474a547556364f72235bcbd830ae5a7768d85fe8141b63"
)
V2_ACTIVATION_SHA256 = (
    "2b91248b8ec7f626ea5281ea9abdf4512662d6e88ddb9ff399d637340bfc0626"
)
V2_HARNESS_SHA256 = (
    "be17289c6dbe917d42d505818ed40e8bf82b8d8ebd19767820504df74724cc59"
)
V2_FAILURE_LOG_SHA256 = (
    "f8c3be8b54e57d452847254e2a184dee9014361a621de5957996123fd77a9e28"
)
V2_FAILURE_REASON = (
    "timing rider requires a native-clock command before activation"
)
V2_SUPERVISOR_PID = 502505
V2_SUPERVISOR_START_TICKS = 573485617
V2_EVALUATOR_PID = 502542
V2_STOP_CONFIRM = "SIM_ONLY_STOP_EXACT_FAILED_TASKREV_0P5_K100_V2"
V3_ACTIVATION_ID = (
    "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_native_clock_v3"
)
CONFIRM = "SIM_ONLY_LAUNCH_ONE_PERSISTENT_TASKREV_0P5_K100"
RESULT_MARKER = "TASKREV_0P5_RESULT_JSON="
ISAAC_PYTHON = "/workspace/hope_isaac_venv/bin/python"
KIT_LOCK = "/workspace/.kit_boot.lock"
SCHEDULE_PATH = (
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/"
    "signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json"
)
EXAM_BANK_PATH = (
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/"
    "schema3_exam_bank_rebind_v1/s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz"
)
EXAM_BANK_BYTES = 63643
EXAM_BANK_SHA256 = "60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca"
EXAM_REBIND_REPORT_PATH = (
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/"
    "schema3_exam_bank_rebind_v1/rebind_report.json"
)
EXAM_REBIND_REPORT_BYTES = 18795
EXAM_REBIND_REPORT_SHA256 = (
    "dd4332edb47f1fb1f4d51ca00ceed612dbcadf9e395eb536c9b73bef9de69ad0"
)
PAPER_FILE_SHA256 = "6f5f152652acd0eb3a80bb5d903f617a1272e665c62c4ce3edc3fdba712f672d"
PAPER_SEMANTIC_SHA256 = "fa7e3c21d0427c4509359297596ee071ecbb06f6cfd5a8d3a252a350c6393b66"
MINIMUM_SELECTED_GPU_FREE_MIB = 6000
MINIMUM_OTHER_GPU_FREE_MIB = 3000


class ExamError(RuntimeError):
    """The requested diagnostic is not exactly bound or cannot be run safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise ExamError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    def nonfinite(value: str) -> None:
        raise ExamError(f"non-finite JSON value in {path}: {value}")

    try:
        value = json.loads(
            path.read_text(), object_pairs_hook=pairs, parse_constant=nonfinite
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExamError(f"cannot read strict activation {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExamError("activation root must be an object")
    return value


def _load_v1_failure_receipt(path: Path) -> dict[str, Any]:
    receipt = _strict_json(path)
    expected_argv = [
        "python3",
        "scripts/run_phase1_task_revision_0p5_exam.py",
        "--queue",
        "configs/phase1_task_revision_supercombo_20260716.yaml",
        "--activation",
        "configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json",
        "--eval-gpu",
        "0",
        "launch",
        "--execute",
        "--confirm",
        CONFIRM,
    ]
    if set(receipt) != {
        "schema_version", "artifact_kind", "receipt_id", "status", "attempt",
        "failure", "side_effects", "authority",
    }:
        raise ExamError("v1 failure receipt top-level keys differ")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_kind")
        != "phase1-task-revision-0p5-k100-launch-failure-receipt"
        or receipt.get("receipt_id")
        != "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1_failed_no_retry"
        or receipt.get("status") != "failed_no_retry"
    ):
        raise ExamError("v1 failure receipt identity differs")
    attempt = receipt.get("attempt")
    if not isinstance(attempt, dict) or attempt != {
        "activation_id": "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1",
        "activation_path": str(HISTORICAL_ACTIVATION_V1),
        "activation_sha256": V1_ACTIVATION_SHA256,
        "harness_path": "scripts/run_phase1_task_revision_0p5_exam.py",
        "harness_sha256": V1_HARNESS_SHA256,
        "source_commit": "3455e2f4c08b04533476f595d70288129308649b",
        "job_id": "taskrev_p2_equal_reward",
        "milestone": 5700,
        "pod": "pod2",
        "physical_eval_gpu": 0,
        "argv": expected_argv,
        "argv_evidence": "reconstructed_from_tracked_operation_and_observed_physical_gpu0",
        "exit_code": 2,
        "observed_rounded_wall_seconds": 5.779,
    }:
        raise ExamError("v1 failure receipt attempt facts differ")
    if receipt.get("failure") != {
        "stage": "validate_inputs",
        "exception_type": "FileNotFoundError",
        "errno": 2,
        "missing_path": EXAM_BANK_PATH,
    }:
        raise ExamError("v1 failure receipt failure facts differ")
    if receipt.get("side_effects") != {
        "runtime_materialized": False,
        "supervisor_created": False,
        "delegated_cgroup_created": False,
        "commit_ack_created": False,
        "evaluator_created": False,
        "trainer_signalled": False,
        "robot_command_sent": False,
    }:
        raise ExamError("v1 failure receipt side-effect boundary differs")
    if receipt.get("authority") != {
        "automatic_retry": False,
        "retry_authorized": False,
        "v1_launch_consumed": True,
        "v1_launch_authorized": False,
    }:
        raise ExamError("v1 failure receipt retry authority differs")
    return receipt


def _load_queue_module(root: Path):
    path = root / "scripts/run_phase1_task_revision_supercombo_queue.py"
    spec = importlib.util.spec_from_file_location("task_revision_queue_for_0p5", path)
    if spec is None or spec.loader is None:
        raise ExamError(f"cannot import task-revision queue: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_closure(root: Path) -> dict[str, str]:
    paths = {
        "spec": "configs/phase1_timing_exam_0p5_k100_20260716.json",
        "converter": "scripts/materialize_phase1_timing_exam_0p5.py",
        "evaluator": "hope_training/whole_body_tracking/scripts/isaac_bank_exam.py",
        "adapter": "hope_training/whole_body_tracking/scripts/isaac_bank_exam_adapter.py",
        "schedule_module": "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py",
        "timing_adapter": "hope_training/whole_body_tracking/scripts/isaac_timing_exam_adapter.py",
        "isaac_scorer": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/virtual_ball.py"
        ),
        "ball_physics": "configs/ball_physics_venue.yaml",
        "runtime": "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py",
        "setup": "hope_training/whole_body_tracking/setup_train_env.sh",
    }
    result: dict[str, str] = {}
    for key, relative in paths.items():
        path = root / relative
        if not path.is_file():
            raise ExamError(f"source closure file is missing: {path}")
        result[key] = sha256_file(path)
    return result


def build_v2_exact_stop_plan(
    queue_path: Path,
    *,
    activation_path: Path,
    eval_gpu: int,
) -> dict[str, Any]:
    """Build the sole authority to stop the already-consumed failed v2 run.

    This deliberately does not route through :func:`load_activation`: v2 binds
    the old harness bytes and may never be made launchable by the fixed v3
    harness.  The stop consumer reads the immutable v2 activation only to bind
    the extant process namespace and never authorizes a launch or retry.
    """

    queue_path = queue_path.resolve()
    root = queue_path.parent.parent.resolve()
    activation_path = activation_path.resolve()
    expected_activation_path = (root / HISTORICAL_ACTIVATION_V2).resolve()
    if activation_path != expected_activation_path:
        raise ExamError("v2 exact-stop requires the tracked historical v2 activation")
    if sha256_file(activation_path) != V2_ACTIVATION_SHA256:
        raise ExamError("historical v2 activation bytes differ")
    activation = _strict_json(activation_path)
    if (
        activation.get("schema_version") != 2
        or activation.get("activation_id")
        != "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_asset_restored_v2"
        or activation.get("harness")
        != {
            "path": "scripts/run_phase1_task_revision_0p5_exam.py",
            "sha256": V2_HARNESS_SHA256,
        }
        or activation.get("authority", {}).get("automatic_retry") is not False
        or activation.get("authority", {}).get("maximum_launches") != 1
    ):
        raise ExamError("historical v2 activation identity/authority differs")
    queue = activation.get("queue")
    if (
        not isinstance(queue, dict)
        or queue.get("path") != str(QUEUE)
        or queue.get("sha256") != sha256_file(queue_path)
    ):
        raise ExamError("historical v2 queue binding differs")
    if eval_gpu != 0:
        raise ExamError("historical v2 exact-stop is bound to physical evaluator GPU 0")
    selection = activation.get("selection")
    if not isinstance(selection, dict):
        raise ExamError("historical v2 selection is missing")
    expected_root = Path(
        "/workspace/codexschema/phase1_task_revision_supercombo_20260716/"
        "runs/p2_equal_reward"
    )
    expected_state = str(
        expected_root / "timing_exam_0p5_supervisor_asset_restored_v2" / "model_5700"
    )
    expected_output = str(
        expected_root / "timing_exam_0p5_asset_restored_v2" / "model_5700"
    )
    if (
        selection.get("job_id") != "taskrev_p2_equal_reward"
        or selection.get("milestone") != 5700
        or selection.get("pod") != "pod2"
        or selection.get("state_dir") != expected_state
        or selection.get("output_dir") != expected_output
    ):
        raise ExamError("historical v2 selection namespace differs")
    return {
        "schema_version": 1,
        "operation": "exact_stop_consumed_v2_native_clock_failure",
        "activation": {
            "path": str(HISTORICAL_ACTIVATION_V2),
            "sha256": V2_ACTIVATION_SHA256,
            "activation_id": activation["activation_id"],
        },
        "queue": {"path": str(QUEUE), "sha256": queue["sha256"]},
        "consumer_harness": {
            "path": "scripts/run_phase1_task_revision_0p5_exam.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "host": "162.43.172.181",
        "port": 13146,
        "ssh_key": str(Path("~/.ssh/id_ed25519_runpod").expanduser()),
        "job_id": "taskrev_p2_equal_reward",
        "milestone": 5700,
        "eval_gpu": 0,
        "state_dir": expected_state,
        "output_dir": expected_output,
        "failure_log": {
            "path": str(Path(expected_output) / "evaluator.log"),
            "sha256": V2_FAILURE_LOG_SHA256,
            "exact_reason": V2_FAILURE_REASON,
        },
        "supervisor": {
            "pid": V2_SUPERVISOR_PID,
            "pgid": V2_SUPERVISOR_PID,
            "sid": V2_SUPERVISOR_PID,
            "start_ticks": V2_SUPERVISOR_START_TICKS,
        },
        "evaluator": {
            "pid": V2_EVALUATOR_PID,
            "pgid": V2_EVALUATOR_PID,
            "sid": V2_EVALUATOR_PID,
        },
        "catastrophic_cleanup": activation["catastrophic_cleanup"],
        "stop_intent_name": "exact_stop_native_clock_failure_v1.json",
        "stop_result_name": "exact_stop_native_clock_failure_result_v1.json",
        "wait_timeout_seconds": 120.0,
        "ssh_timeout_seconds": 150.0,
        "automatic_retry": False,
        "retry_authorized": False,
        "launch_authorized": False,
        "trainer_signal": False,
        "robot_command": False,
        "evaluator_direct_signal": False,
        "cgroup_kill_by_consumer": False,
        "sigkill_by_consumer": False,
    }


def _positive_number(where: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExamError(f"{where} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ExamError(f"{where} must be in [{minimum}, {maximum}]")
    return result


def load_activation(path: Path, *, root: Path) -> dict[str, Any]:
    path = path.resolve()
    activation = _strict_json(path)
    if activation.get("activation_id") == (
        "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1"
    ):
        if sha256_file(path) != V1_ACTIVATION_SHA256:
            raise ExamError("historical v1 activation bytes differ")
        raise ExamError(
            "historical v1 activation is consumed failed_no_retry; current HEAD only "
            "authorizes the fresh native-clock v3 activation"
        )
    if activation.get("activation_id") == (
        "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_asset_restored_v2"
    ):
        if sha256_file(path) != V2_ACTIVATION_SHA256:
            raise ExamError("historical v2 activation bytes differ")
        raise ExamError(
            "historical v2 activation is consumed by its native-clock failure; "
            "only the fresh v3 activation may launch"
        )
    expected = {
        "schema_version",
        "activation_id",
        "created_utc",
        "selection",
        "prior_attempt",
        "required_assets",
        "consumption",
        "queue",
        "harness",
        "resource_gate",
        "catastrophic_cleanup",
        "supervision",
        "authority",
        "v2_failed_attempt",
    }
    if set(activation) != expected:
        raise ExamError("activation top-level keys differ")
    if (
        activation.get("schema_version") != 3
        or activation.get("activation_id") != V3_ACTIVATION_ID
    ):
        raise ExamError("activation is not the sole fresh native-clock v3 authority")
    selection = activation.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "job_id",
        "milestone",
        "pod",
        "output_dir",
        "state_dir",
        "milestone_receipt",
        "behavior_receipt",
        "rationale",
    }:
        raise ExamError("activation selection keys differ")
    if (
        selection.get("job_id") != "taskrev_p2_equal_reward"
        or selection.get("milestone") != 5700
        or selection.get("pod") != "pod2"
    ):
        raise ExamError("activation does not select the fixed equal-reward model_5700 cell")
    for key in ("output_dir", "state_dir", "milestone_receipt", "behavior_receipt"):
        value = selection.get(key)
        if not isinstance(value, str) or not value.startswith("/workspace/"):
            raise ExamError(f"activation selection.{key} must be absolute /workspace")
    prior = activation.get("prior_attempt")
    try:
        actual_failure_receipt_sha256 = sha256_file(root / V1_FAILURE_RECEIPT)
    except OSError as exc:
        raise ExamError("v1 failure receipt is missing or unreadable") from exc
    if not isinstance(prior, dict) or prior != {
        "failure_receipt_path": str(V1_FAILURE_RECEIPT),
        "failure_receipt_sha256": actual_failure_receipt_sha256,
        "activation_id": "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1",
        "status": "failed_no_retry",
        "retry_authorized": False,
        "v1_launch_consumed": True,
    }:
        raise ExamError("activation prior-attempt binding differs")
    if prior["failure_receipt_sha256"] != actual_failure_receipt_sha256:
        raise ExamError("v1 failure receipt bytes differ")
    _load_v1_failure_receipt(root / prior["failure_receipt_path"])
    required_assets = activation.get("required_assets")
    if not isinstance(required_assets, dict) or required_assets != {
        "exam_bank": {
            "path": EXAM_BANK_PATH,
            "bytes": EXAM_BANK_BYTES,
            "sha256": EXAM_BANK_SHA256,
        },
        "rebind_report": {
            "path": EXAM_REBIND_REPORT_PATH,
            "bytes": EXAM_REBIND_REPORT_BYTES,
            "sha256": EXAM_REBIND_REPORT_SHA256,
        },
    }:
        raise ExamError("activation restored-asset binding differs")
    expected_run_root = Path(
        "/workspace/codexschema/phase1_task_revision_supercombo_20260716/"
        "runs/p2_equal_reward"
    )
    v2_failed_attempt = activation.get("v2_failed_attempt")
    expected_v2_state = str(
        expected_run_root
        / "timing_exam_0p5_supervisor_asset_restored_v2"
        / "model_5700"
    )
    if not isinstance(v2_failed_attempt, dict) or v2_failed_attempt != {
        "activation": {
            "path": str(HISTORICAL_ACTIVATION_V2),
            "sha256": V2_ACTIVATION_SHA256,
            "activation_id": (
                "phase1_task_revision_0p5_k100_p2_equal_reward_model5700_asset_restored_v2"
            ),
        },
        "state_dir": expected_v2_state,
        "stop_result": {
            "basename": "exact_stop_native_clock_failure_result_v1.json",
            "artifact_kind": "taskrev_0p5_v2_exact_stop_result",
            "status": "v2_failed_no_retry_stopped_exact",
            "failure_log_sha256": V2_FAILURE_LOG_SHA256,
            "failure_reason": V2_FAILURE_REASON,
            "signal": "SIGTERM_supervisor_once",
            "retry_authorized": False,
            "evaluator_direct_signal": False,
            "cgroup_kill_by_consumer": False,
            "sigkill_by_consumer": False,
        },
    }:
        raise ExamError("activation v2 failed-attempt binding differs")
    consumption = activation.get("consumption")
    if not isinstance(consumption, dict) or consumption != {
        "v3_attempt_dir": str(
            expected_run_root
            / "timing_exam_0p5_attempt_native_clock_v3"
            / "model_5700"
        ),
        "v2_stop_result_required_before_any_consumption_write": True,
        "no_clobber": True,
        "retry_authorized": False,
    }:
        raise ExamError("activation v3 one-shot consumption binding differs")
    queue = activation.get("queue")
    harness = activation.get("harness")
    if not isinstance(queue, dict) or set(queue) != {"path", "sha256"}:
        raise ExamError("activation queue binding differs")
    if not isinstance(harness, dict) or set(harness) != {"path", "sha256"}:
        raise ExamError("activation harness binding differs")
    if queue["path"] != str(QUEUE) or sha256_file(root / queue["path"]) != queue["sha256"]:
        raise ExamError("activation queue bytes differ")
    if harness["path"] != "scripts/run_phase1_task_revision_0p5_exam.py":
        raise ExamError("activation harness path differs")
    if sha256_file(root / harness["path"]) != harness["sha256"]:
        raise ExamError("activation harness bytes differ")
    resources = activation.get("resource_gate")
    if not isinstance(resources, dict) or set(resources) != {
        "allowed_eval_gpus",
        "minimum_selected_gpu_free_mib",
        "minimum_other_gpu_free_mib",
        "headroom_rationale",
        "stable_samples",
        "sample_interval_seconds",
    }:
        raise ExamError("activation resource gate differs")
    if resources.get("allowed_eval_gpus") != [0, 1, 2]:
        raise ExamError("activation must allow exactly Pod2 GPUs 0,1,2")
    if resources.get("stable_samples") != 2:
        raise ExamError("activation requires exactly two resource samples")
    if resources.get("minimum_selected_gpu_free_mib") != MINIMUM_SELECTED_GPU_FREE_MIB:
        raise ExamError("activation selected-GPU free-memory floor differs")
    if resources.get("minimum_other_gpu_free_mib") != MINIMUM_OTHER_GPU_FREE_MIB:
        raise ExamError("activation cross-GPU free-memory floor differs")
    if not isinstance(resources.get("headroom_rationale"), str) or "all three Pod2 GPUs" not in resources["headroom_rationale"]:
        raise ExamError("activation cross-GPU headroom rationale differs")
    _positive_number(
        "resource sample interval", resources.get("sample_interval_seconds"), 0.1, 10
    )
    catastrophic = activation.get("catastrophic_cleanup")
    if not isinstance(catastrophic, dict) or catastrophic != {
        "requires_delegated_cgroup_v2": True,
        "guardian_required": True,
        "cgroup_kill_required": True,
        "trusted_single_operator_filesystem": True,
        "unsupported_result": "NO_LAUNCH",
    }:
        raise ExamError("activation catastrophic-cleanup contract differs")
    supervision = activation.get("supervision")
    required_supervision = {
        "hello_timeout_seconds",
        "commit_timeout_seconds",
        "ack_observation_seconds",
        "ssh_timeout_seconds",
        "heartbeat_seconds",
        "evaluator_total_timeout_seconds",
        "completed_artifact_teardown_grace_seconds",
        "exact_term_grace_seconds",
        "converter_timeout_seconds",
        "guardian_ready_timeout_seconds",
        "guardian_finish_timeout_seconds",
    }
    if not isinstance(supervision, dict) or set(supervision) != required_supervision:
        raise ExamError("activation supervision keys differ")
    bounds = {
        "hello_timeout_seconds": (1, 60),
        "commit_timeout_seconds": (2, 120),
        "ack_observation_seconds": (0.1, 30),
        "ssh_timeout_seconds": (5, 120),
        "heartbeat_seconds": (5, 120),
        "evaluator_total_timeout_seconds": (60, 7200),
        "completed_artifact_teardown_grace_seconds": (10, 600),
        "exact_term_grace_seconds": (1, 120),
        "converter_timeout_seconds": (10, 900),
        "guardian_ready_timeout_seconds": (1, 60),
        "guardian_finish_timeout_seconds": (5, 300),
    }
    for key, (minimum, maximum) in bounds.items():
        _positive_number(f"supervision.{key}", supervision.get(key), minimum, maximum)
    authority = activation.get("authority")
    if not isinstance(authority, dict) or authority != {
        "launch_authorized": True,
        "automatic_retry": False,
        "fresh_activation_after_v2_native_clock_failure": True,
        "maximum_launches": 1,
        "formal_evidence_eligible": False,
        "evaluation_contract_exact": False,
        "trainer_signal": False,
        "robot_command": False,
        "broad_signal": False,
    }:
        raise ExamError("activation authority differs")
    try:
        activation["_repo_path"] = str(path.relative_to(root.resolve()))
    except ValueError as exc:
        raise ExamError("activation must be a tracked file inside the repository") from exc
    if activation["_repo_path"] != str(ACTIVATION):
        raise ExamError("current HEAD authorizes only the tracked v3 activation path")
    activation["_path"] = str(path)
    activation["_sha256"] = sha256_file(path)
    return activation


def build_plan(
    queue_path: Path,
    *,
    activation_path: Path = ACTIVATION,
    eval_gpu: int,
) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    root = queue_path.parent.parent
    activation = load_activation(activation_path, root=root)
    approved_harness = (root / activation["harness"]["path"]).resolve()
    current_harness = Path(__file__).resolve()
    if current_harness != approved_harness:
        raise ExamError(
            "current executable is not the activation-approved harness path")
    if sha256_file(current_harness) != activation["harness"]["sha256"]:
        raise ExamError("current executable bytes are not activation-approved")
    selection = activation["selection"]
    module = _load_queue_module(root)
    queue = module.load_queue(queue_path)
    job = module._job(queue, selection["job_id"])
    module._require_launchable_job(job)
    training_slot = module.continuation._slots(queue)[job["resource"]["required_slot"]]
    absolute = module.continuation._absolute_schedule(
        job, module.continuation._parent_records_from_job_context(job)
    )
    milestone = int(selection["milestone"])
    if milestone not in absolute["milestones"]:
        raise ExamError("fixed milestone is not registered in the queue")
    pod = queue["pods"][training_slot.pod]
    if training_slot.pod != selection["pod"]:
        raise ExamError("fixed activation pod differs from queue slot")
    if type(eval_gpu) is not int or eval_gpu not in activation["resource_gate"]["allowed_eval_gpus"]:
        raise ExamError(f"eval GPU {eval_gpu!r} is not activation-authorized")
    if eval_gpu not in pod["gpus"]:
        raise ExamError(f"eval GPU {eval_gpu!r} is not on {training_slot.pod}")
    claim = module.continuation._attestor_claim_spec(queue, job, training_slot)
    paper_path = str(job["exam"]["path"])
    if paper_path != (
        "/workspace/codexschema/phase1_task_revision_supercombo_20260716/"
        "papers/timing_exam_0p5_k100.schedule.json"
    ):
        raise ExamError("task-revision queue points to an unexpected timing paper")
    expected_output = str(
        Path(job["run_dir"]) / "timing_exam_0p5_native_clock_v3" / f"model_{milestone}"
    )
    expected_state = str(
        Path(job["run_dir"])
        / "timing_exam_0p5_supervisor_native_clock_v3"
        / f"model_{milestone}"
    )
    expected_milestone = str(Path(job["run_dir"]) / "milestones" / f"model_{milestone}.json")
    expected_behavior = str(Path(job["run_dir"]) / "behavior_milestones" / f"model_{milestone}.json")
    if selection["output_dir"] != expected_output or selection["state_dir"] != expected_state:
        raise ExamError("activation output/state namespace differs from fixed queue cell")
    if selection["milestone_receipt"] != expected_milestone or selection["behavior_receipt"] != expected_behavior:
        raise ExamError("activation receipt paths differ from fixed queue cell")
    source = job["source"]
    return {
        "schema_version": 2,
        "activation": {
            "path": activation["_repo_path"],
            "sha256": activation["_sha256"],
            "activation_id": activation["activation_id"],
        },
        "prior_attempt": activation["prior_attempt"],
        "v2_failed_attempt": activation["v2_failed_attempt"],
        "consumption": activation["consumption"],
        "job_id": job["id"],
        "milestone": milestone,
        "milestone_offset_from_parent": milestone - absolute["parent_iteration"],
        "pod": training_slot.pod,
        "training_gpu": training_slot.gpu,
        "eval_gpu": eval_gpu,
        "host": str(pod["host"]),
        "port": int(pod["port"]),
        "ssh_key": str(Path(queue["ssh"]["key"]).expanduser()),
        "run_dir": str(job["run_dir"]),
        "output_dir": selection["output_dir"],
        "state_dir": selection["state_dir"],
        "milestone_receipt": selection["milestone_receipt"],
        "behavior_receipt": selection["behavior_receipt"],
        "binding_path": str(claim["binding_path"]),
        "expected_claim_content_sha256": str(claim["content_sha256"]),
        "source_checkout": str(source["checkout"]),
        "source_commit": str(source["commit"]),
        "queue": {"path": activation["queue"]["path"], "sha256": sha256_file(queue_path)},
        "harness": {
            "path": activation["harness"]["path"],
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_closure": _source_closure(root),
        "paper": {
            "path": paper_path,
            "file_sha256": PAPER_FILE_SHA256,
            "semantic_sha256": PAPER_SEMANTIC_SHA256,
        },
        "schedule": {"path": SCHEDULE_PATH},
        "exam_bank": activation["required_assets"]["exam_bank"],
        "exam_rebind_report": activation["required_assets"]["rebind_report"],
        "kit_lock": KIT_LOCK,
        "resource_gate": activation["resource_gate"],
        "catastrophic_cleanup": activation["catastrophic_cleanup"],
        "supervision": activation["supervision"],
        "formal_evidence_eligible": False,
        "evaluation_contract_exact": False,
        "automatic_retry": False,
    }


REMOTE_PROGRAM = r'''
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import select
import signal
import stat
import subprocess
import sys
import time

MARKER = "TASKREV_0P5_RESULT_JSON="
REL = {
    "spec": "configs/phase1_timing_exam_0p5_k100_20260716.json",
    "converter": "scripts/materialize_phase1_timing_exam_0p5.py",
    "evaluator": "hope_training/whole_body_tracking/scripts/isaac_bank_exam.py",
    "adapter": "hope_training/whole_body_tracking/scripts/isaac_bank_exam_adapter.py",
    "schedule_module": "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py",
    "timing_adapter": "hope_training/whole_body_tracking/scripts/isaac_timing_exam_adapter.py",
    "isaac_scorer": "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/virtual_ball.py",
    "ball_physics": "configs/ball_physics_venue.yaml",
    "runtime": "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py",
    "setup": "hope_training/whole_body_tracking/setup_train_env.sh",
}
PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
try:
    _PRCTL = _LIBC.prctl
except AttributeError:
    _PRCTL = None
if _PRCTL is not None:
    _PRCTL.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                       ctypes.c_ulong, ctypes.c_ulong]
    _PRCTL.restype = ctypes.c_int
try:
    _RENAMEAT2 = _LIBC.renameat2
except AttributeError:
    _RENAMEAT2 = None
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint]
    _RENAMEAT2.restype = ctypes.c_int
RENAME_NOREPLACE = 1

def cbytes(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True,
                      separators=(",", ":"), sort_keys=True).encode("utf-8")

def canonical(value):
    return hashlib.sha256(cbytes(value)).hexdigest()

def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"digest input is not a single-link regular file: {path}")
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
        after = os.fstat(fd)
        entry = path.lstat()
        signature = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns)
        if (signature(before) != signature(after) or
                signature(after) != signature(entry) or
                after.st_nlink != 1 or total != after.st_size):
            raise RuntimeError(f"digest input changed while reading: {path}")
        return h.hexdigest()
    finally:
        os.close(fd)

def validate_exact_asset(binding, label):
    if (not isinstance(binding, dict) or
            set(binding) != {"path", "bytes", "sha256"} or
            not isinstance(binding.get("path"), str) or
            not Path(binding["path"]).is_absolute() or
            isinstance(binding.get("bytes"), bool) or
            not isinstance(binding.get("bytes"), int) or binding["bytes"] <= 0 or
            not isinstance(binding.get("sha256"), str) or
            len(binding["sha256"]) != 64):
        raise RuntimeError(f"{label} binding shape differs")
    path = Path(binding["path"])
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
            info.st_size != binding["bytes"]):
        raise RuntimeError(f"{label} file type/link/size differs: {path}")
    if sha(path) != binding["sha256"]:
        raise RuntimeError(f"{label} SHA-256 differs: {path}")
    return path

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def strict_loads(raw, label):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    def nonfinite(value):
        raise RuntimeError(f"non-finite JSON in {label}: {value}")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root is not an object")
    return value

def stable_bytes(path, label, *, pseudo=False):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not regular non-symlink: {path}")
        if not pseudo and before.st_nlink != 1:
            raise RuntimeError(f"{label} is not a single-link regular file: {path}")
        chunks = []
        while True:
            row = os.read(fd, 1024 * 1024)
            if not row:
                break
            chunks.append(row)
        after = os.fstat(fd)
        entry = path.lstat()
        core = lambda value: (value.st_dev, value.st_ino, value.st_mode)
        if core(before) != core(after) or core(after) != core(entry):
            raise RuntimeError(f"{label} identity changed while reading: {path}")
        if not pseudo:
            signature = lambda value: (
                value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                value.st_size, value.st_mtime_ns)
            if (signature(before) != signature(after) or
                    signature(after) != signature(entry) or
                    after.st_nlink != 1 or
                    sum(len(row) for row in chunks) != after.st_size):
                raise RuntimeError(f"{label} changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)

def stable_json(path, label):
    raw = stable_bytes(path, label)
    return strict_loads(raw, label), raw

def _directory_signature(info):
    return (info.st_dev, info.st_ino, info.st_mode)

def _absolute_directory_parts(path):
    path = Path(path)
    if not path.is_absolute():
        raise RuntimeError(f"trusted directory must be absolute: {path}")
    parts = list(path.parts[1:])
    if any(not part or part in (".", "..") or "/" in part for part in parts):
        raise RuntimeError(f"trusted directory has unsafe component: {path}")
    return path, parts

def open_directory_guard(path, *, create_missing=False):
    path, parts = _absolute_directory_parts(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise RuntimeError("Linux O_NOFOLLOW/O_DIRECTORY directory guard is unavailable")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    current = os.open("/", flags)
    chain = [("/",) + _directory_signature(os.fstat(current))]
    current_path = Path("/")
    try:
        for part in parts:
            if create_missing:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=current)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise RuntimeError(f"trusted path component is not a directory: {current_path / part}")
            current_path /= part
            chain.append((str(current_path),) + _directory_signature(info))
            os.close(current)
            current = child
        return {"path": str(path), "fd": current, "chain": tuple(chain)}
    except BaseException:
        os.close(current)
        raise

def close_directory_guard(guard):
    if guard is None:
        return
    fd = guard.get("fd", -1)
    if fd >= 0:
        os.close(fd)
        guard["fd"] = -1

def revalidate_directory_guard(guard):
    if guard is None or guard.get("fd", -1) < 0:
        raise RuntimeError("trusted directory guard is closed")
    held = _directory_signature(os.fstat(guard["fd"]))
    expected = tuple(guard["chain"])
    if held != tuple(expected[-1][1:]):
        raise RuntimeError(f"held trusted directory identity changed: {guard['path']}")
    current = open_directory_guard(guard["path"], create_missing=False)
    try:
        if tuple(current["chain"]) != expected:
            raise RuntimeError(f"trusted directory path was replaced: {guard['path']}")
    finally:
        close_directory_guard(current)

def create_child_directory_guard(parent_guard, name):
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name:
        raise RuntimeError("unsafe child directory name")
    revalidate_directory_guard(parent_guard)
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_guard["fd"])
    except FileExistsError as exc:
        raise RuntimeError(
            f"no-clobber directory namespace exists: {Path(parent_guard['path']) / name}"
        ) from exc
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    child_fd = os.open(name, flags, dir_fd=parent_guard["fd"])
    child_info = os.fstat(child_fd)
    child_path = Path(parent_guard["path"]) / name
    guard = {
        "path": str(child_path),
        "fd": child_fd,
        "chain": tuple(parent_guard["chain"]) + (
            (str(child_path),) + _directory_signature(child_info),
        ),
    }
    try:
        revalidate_directory_guard(parent_guard)
        revalidate_directory_guard(guard)
    except BaseException:
        close_directory_guard(guard)
        raise
    return guard

def create_directory_guard(path):
    path, _ = _absolute_directory_parts(path)
    parent = open_directory_guard(path.parent, create_missing=True)
    try:
        return create_child_directory_guard(parent, path.name)
    finally:
        close_directory_guard(parent)

def guarded_child_exists(guard, name):
    revalidate_directory_guard(guard)
    try:
        os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False

def guarded_open(guard, name, flags, mode=0o444):
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name:
        raise RuntimeError("unsafe guarded file name")
    revalidate_directory_guard(guard)
    fd = os.open(name, flags | getattr(os, "O_NOFOLLOW", 0), mode,
                 dir_fd=guard["fd"])
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(
                f"guarded file is not single-link regular: {guard['path']}/{name}")
        revalidate_directory_guard(guard)
        return fd
    except BaseException:
        os.close(fd)
        raise

def guarded_file_stat(guard, name):
    revalidate_directory_guard(guard)
    info = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(
            f"guarded file is not single-link regular: {guard['path']}/{name}")
    revalidate_directory_guard(guard)
    return info

def guarded_publish_bytes(guard, name, raw, mode=0o444):
    if _RENAMEAT2 is None:
        raise RuntimeError("Linux renameat2(RENAME_NOREPLACE) is unavailable")
    if guarded_child_exists(guard, name):
        raise FileExistsError(
            f"no-clobber guarded entry exists: {guard['path']}/{name}")
    temporary = f".{name}.publish.{os.getpid()}.{secrets.token_hex(16)}"
    fd = guarded_open(
        guard, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        created = os.fstat(fd)
        created_identity = (created.st_dev, created.st_ino, created.st_mode)
        if created.st_nlink != 1 or created.st_size != 0:
            raise RuntimeError(
                f"new guarded temporary has unexpected link/size: {guard['path']}/{temporary}")
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
        held = os.fstat(fd)
        expected = created_identity + (1, len(raw))
        temporary_entry = os.stat(
            temporary, dir_fd=guard["fd"], follow_symlinks=False)
        if ((held.st_dev, held.st_ino, held.st_mode, held.st_nlink, held.st_size) != expected or
            (temporary_entry.st_dev, temporary_entry.st_ino, temporary_entry.st_mode,
             temporary_entry.st_nlink, temporary_entry.st_size) != expected or
            not stat.S_ISREG(temporary_entry.st_mode)):
            raise RuntimeError(
                f"guarded temporary entry was replaced: {guard['path']}/{temporary}")
        # renameat2 is the single atomic no-replace publication point.  The
        # final name is never visible until every byte is on the fsynced inode.
        ctypes.set_errno(0)
        rc = _RENAMEAT2(
            guard["fd"], os.fsencode(temporary), guard["fd"], os.fsencode(name),
            RENAME_NOREPLACE)
        if rc != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), name)
            raise OSError(error, os.strerror(error), name)
        os.fsync(guard["fd"])
        revalidate_directory_guard(guard)
        held_after = os.fstat(fd)
        entry_after = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        if ((held_after.st_dev, held_after.st_ino, held_after.st_mode,
             held_after.st_nlink, held_after.st_size) != expected or
            (entry_after.st_dev, entry_after.st_ino, entry_after.st_mode,
             entry_after.st_nlink, entry_after.st_size) != expected):
            raise RuntimeError(f"guarded published entry changed after fsync: {guard['path']}/{name}")
    finally:
        os.close(fd)
        try:
            os.unlink(temporary, dir_fd=guard["fd"])
        except FileNotFoundError:
            pass
    revalidate_directory_guard(guard)

def guarded_publish_json(guard, name, value, mode=0o444):
    raw = cbytes(value) + b"\n"
    guarded_publish_bytes(guard, name, raw, mode)
    if guarded_stable_bytes(guard, name, f"published {name}") != raw:
        raise RuntimeError(f"guarded published bytes differ after reopen: {guard['path']}/{name}")
    return hashlib.sha256(raw).hexdigest()

def publish_or_read_launch_decision(state_guard, candidate):
    """Atomically choose exactly one commit/abort decision for this namespace."""
    try:
        guarded_publish_json(state_guard, "launch_decision.json", candidate)
        won = True
    except FileExistsError:
        won = False
    decision, raw = guarded_stable_json(
        state_guard, "launch_decision.json", "launch decision")
    allowed = {"commit", "abort_deadline"}
    if (set(decision) != {"schema_version", "artifact_kind", "decision",
                         "plan_sha256", "pid", "pgid", "proc_start_ticks",
                         "hello_sha256", "ledger_sha256", "token_sha256",
                         "commit_deadline_monotonic_ns", "decided_monotonic_ns",
                         "decided_utc", "retry_authorized"} or
            decision.get("schema_version") != 1 or
            decision.get("artifact_kind") != "taskrev_0p5_launch_decision" or
            decision.get("decision") not in allowed or
            decision.get("retry_authorized") is not False):
        raise RuntimeError("existing launch decision has an invalid schema")
    return decision, raw, won

def validate_launch_decision_document(decision, raw, hello, spec):
    keys = {"schema_version", "artifact_kind", "decision", "plan_sha256",
            "pid", "pgid", "proc_start_ticks", "hello_sha256",
            "ledger_sha256", "token_sha256", "commit_deadline_monotonic_ns",
            "decided_monotonic_ns", "decided_utc", "retry_authorized"}
    if (set(decision) != keys or decision.get("schema_version") != 1 or
            decision.get("artifact_kind") != "taskrev_0p5_launch_decision" or
            decision.get("decision") not in {"commit", "abort_deadline"} or
            decision.get("plan_sha256") != canonical(spec) or
            decision.get("pid") != hello.get("pid") or
            decision.get("pgid") != hello.get("pgid") or
            decision.get("proc_start_ticks") != hello.get("proc_start_ticks") or
            decision.get("hello_sha256") != canonical_file_bytes(hello) or
            decision.get("commit_deadline_monotonic_ns") !=
                hello.get("commit_deadline_monotonic_ns") or
            type(decision.get("decided_monotonic_ns")) is not int or
            decision.get("retry_authorized") is not False):
        raise RuntimeError("launch decision does not bind the exact hello/plan")
    if decision["decision"] == "commit":
        if (not isinstance(decision.get("ledger_sha256"), str) or
                not isinstance(decision.get("token_sha256"), str) or
                decision["decided_monotonic_ns"] >=
                    decision["commit_deadline_monotonic_ns"]):
            raise RuntimeError("commit decision is late or lacks token bindings")
    elif (decision.get("ledger_sha256") is not None or
          decision.get("token_sha256") is not None or
          decision["decided_monotonic_ns"] <
              decision["commit_deadline_monotonic_ns"]):
        raise RuntimeError("deadline-abort decision is early or carries commit bindings")
    return hashlib.sha256(raw).hexdigest()

def canonical_file_bytes(value):
    return hashlib.sha256(cbytes(value) + b"\n").hexdigest()

def validate_committed_chain(state_guard, spec, *, require_ack):
    hello, hello_raw = guarded_stable_json(
        state_guard, "child_hello.json", "supervisor hello")
    ledger, ledger_raw = guarded_stable_json(
        state_guard, "launch_ledger.json", "launch ledger")
    token, token_raw = guarded_stable_json(
        state_guard, "commit_token.json", "commit token")
    decision, decision_raw = guarded_stable_json(
        state_guard, "launch_decision.json", "launch decision")
    hello_sha = hashlib.sha256(hello_raw).hexdigest()
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
    token_sha = hashlib.sha256(token_raw).hexdigest()
    decision_sha = hashlib.sha256(decision_raw).hexdigest()
    validate_launch_decision_document(decision, decision_raw, hello, spec)
    pid = hello.get("pid")
    exact = (
        set(hello) == {
            "schema_version", "artifact_kind", "plan_sha256", "activation",
            "job_id", "milestone", "pid", "pgid", "proc_start_ticks",
            "argv_sha256", "commit_deadline_monotonic_ns", "automatic_retry"} and
        hello.get("schema_version") == 1 and
        hello.get("artifact_kind") == "taskrev_0p5_supervisor_hello" and
        hello.get("plan_sha256") == canonical(spec) and
        hello.get("activation") == spec["activation"] and
        hello.get("job_id") == spec["job_id"] and
        hello.get("milestone") == spec["milestone"] and
        type(pid) is int and pid > 1 and hello.get("pgid") == pid and
        type(hello.get("proc_start_ticks")) is int and
        hello.get("automatic_retry") is False and
        set(ledger) == {
            "schema_version", "artifact_kind", "plan_sha256", "activation",
            "job_id", "milestone", "pid", "pgid", "proc_start_ticks",
            "hello_sha256", "output_dir", "state_dir",
            "resource_samples_before_fork", "committed_utc", "automatic_retry"} and
        ledger.get("schema_version") == 1 and
        ledger.get("artifact_kind") == "taskrev_0p5_launch_ledger" and
        ledger.get("plan_sha256") == canonical(spec) and
        ledger.get("activation") == spec["activation"] and
        ledger.get("job_id") == spec["job_id"] and
        ledger.get("milestone") == spec["milestone"] and
        ledger.get("pid") == pid and ledger.get("pgid") == pid and
        ledger.get("proc_start_ticks") == hello.get("proc_start_ticks") and
        ledger.get("hello_sha256") == hello_sha and
        ledger.get("output_dir") == spec["output_dir"] and
        ledger.get("state_dir") == spec["state_dir"] and
        ledger.get("automatic_retry") is False and
        set(token) == {
            "schema_version", "artifact_kind", "pid", "pgid", "proc_start_ticks",
            "hello_sha256", "ledger_sha256", "nonce", "published_utc",
            "retry_authorized"} and
        token.get("schema_version") == 1 and
        token.get("artifact_kind") == "taskrev_0p5_commit_token" and
        token.get("pid") == pid and token.get("pgid") == pid and
        token.get("proc_start_ticks") == hello.get("proc_start_ticks") and
        token.get("hello_sha256") == hello_sha and
        token.get("ledger_sha256") == ledger_sha and
        isinstance(token.get("nonce"), str) and len(token["nonce"]) == 64 and
        token.get("retry_authorized") is False and
        set(decision) == {
            "schema_version", "artifact_kind", "decision", "plan_sha256",
            "pid", "pgid", "proc_start_ticks", "hello_sha256",
            "ledger_sha256", "token_sha256", "commit_deadline_monotonic_ns",
            "decided_monotonic_ns", "decided_utc", "retry_authorized"} and
        decision.get("schema_version") == 1 and
        decision.get("artifact_kind") == "taskrev_0p5_launch_decision" and
        decision.get("decision") == "commit" and
        decision.get("plan_sha256") == canonical(spec) and
        decision.get("pid") == pid and decision.get("pgid") == pid and
        decision.get("proc_start_ticks") == hello.get("proc_start_ticks") and
        decision.get("hello_sha256") == hello_sha and
        decision.get("ledger_sha256") == ledger_sha and
        decision.get("token_sha256") == token_sha and
        decision.get("commit_deadline_monotonic_ns") ==
            hello.get("commit_deadline_monotonic_ns") and
        type(decision.get("decided_monotonic_ns")) is int and
        decision.get("decided_monotonic_ns") <
            hello.get("commit_deadline_monotonic_ns") and
        decision.get("retry_authorized") is False)
    if not exact:
        raise RuntimeError("committed launch chain does not bind the exact plan")
    result = {
        "hello": hello, "hello_raw": hello_raw,
        "ledger": ledger, "ledger_raw": ledger_raw,
        "token": token, "token_raw": token_raw,
        "decision": decision, "decision_raw": decision_raw,
        "hello_sha256": hello_sha, "ledger_sha256": ledger_sha,
        "token_sha256": token_sha, "decision_sha256": decision_sha,
    }
    if not require_ack:
        return result
    ack, ack_raw = guarded_stable_json(
        state_guard, "commit_ack.json", "commit acknowledgment")
    catastrophic = ack.get("catastrophic_cleanup")
    if (set(ack) != {
            "schema_version", "artifact_kind", "plan_sha256", "pid", "pgid",
            "proc_start_ticks", "hello_sha256", "ledger_sha256", "token_sha256",
            "decision_sha256", "acknowledged_utc", "kit_lock_held",
            "resource_samples", "catastrophic_cleanup", "automatic_retry"} or
        ack.get("schema_version") != 1 or
        ack.get("artifact_kind") != "taskrev_0p5_supervisor_commit_ack" or
        ack.get("plan_sha256") != canonical(spec) or
        ack.get("pid") != pid or ack.get("pgid") != pid or
        ack.get("proc_start_ticks") != hello.get("proc_start_ticks") or
        ack.get("hello_sha256") != hello_sha or
        ack.get("ledger_sha256") != ledger_sha or
        ack.get("token_sha256") != token_sha or
        ack.get("decision_sha256") != decision_sha or
        ack.get("kit_lock_held") is not True or
        not isinstance(ack.get("resource_samples"), list) or
        ack.get("automatic_retry") is not False or
        not isinstance(catastrophic, dict) or
        catastrophic.get("contract") != spec["catastrophic_cleanup"] or
        catastrophic.get("cgroup_exact_members") != [pid] or
        catastrophic.get("supervisor_contained") is not True or
        catastrophic.get("guardian_live_exact") is not True or
        not isinstance(catastrophic.get("guardian"), dict)):
        raise RuntimeError("commit acknowledgment does not bind the exact committed chain")
    result.update(ack=ack, ack_raw=ack_raw,
                  ack_sha256=hashlib.sha256(ack_raw).hexdigest())
    return result

def guarded_stable_bytes(guard, name, label):
    fd = guarded_open(guard, name, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        chunks = []
        while True:
            row = os.read(fd, 1024 * 1024)
            if not row:
                break
            chunks.append(row)
        after = os.fstat(fd)
        if (_directory_signature(before), before.st_nlink,
            before.st_size, before.st_mtime_ns) != (
            _directory_signature(after), after.st_nlink,
            after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"{label} changed while reading")
        if after.st_nlink != 1 or sum(len(row) for row in chunks) != after.st_size:
            raise RuntimeError(f"{label} is not stable single-link content")
        entry = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        expected = (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size)
        if ((entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink, entry.st_size) !=
                expected or not stat.S_ISREG(entry.st_mode)):
            raise RuntimeError(f"{label} directory entry was replaced while reading")
        revalidate_directory_guard(guard)
        entry_after = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        if ((entry_after.st_dev, entry_after.st_ino, entry_after.st_mode,
             entry_after.st_nlink, entry_after.st_size) != expected):
            raise RuntimeError(f"{label} directory entry changed after read validation")
        return b"".join(chunks)
    finally:
        os.close(fd)
        revalidate_directory_guard(guard)

def guarded_append_prefix_bytes(guard, name, label, *, max_bytes=64 * 1024 * 1024):
    """Read one frozen append prefix without chasing a concurrently growing EOF."""
    fd = guarded_open(guard, name, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        if before.st_size > max_bytes:
            raise RuntimeError(
                f"{label} frozen prefix exceeds {max_bytes} bytes")
        remaining = before.st_size
        chunks = []
        while remaining:
            row = os.read(fd, min(remaining, 1024 * 1024))
            if not row:
                raise RuntimeError(f"{label} ended before its frozen prefix")
            chunks.append(row)
            remaining -= len(row)
        after = os.fstat(fd)
        entry = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink)
        if (identity(before) != identity(after) or
                identity(after) != identity(entry) or
                after.st_nlink != 1 or after.st_size < before.st_size):
            raise RuntimeError(f"{label} identity changed or shrank")
        revalidate_directory_guard(guard)
        entry_after = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        if identity(entry_after) != identity(after) or entry_after.st_size < before.st_size:
            raise RuntimeError(f"{label} entry changed after prefix validation")
        return b"".join(chunks)
    finally:
        os.close(fd)
        revalidate_directory_guard(guard)

def guarded_stable_json(guard, name, label):
    raw = guarded_stable_bytes(guard, name, label)
    return strict_loads(raw, label), raw

def guarded_sha(guard, name, label):
    return hashlib.sha256(guarded_stable_bytes(guard, name, label)).hexdigest()

def _cgroup_file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode)

def cgroup_read(guard, name):
    revalidate_directory_guard(guard)
    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                 dir_fd=guard["fd"])
    try:
        before = os.fstat(fd)
        chunks = []
        while True:
            row = os.read(fd, 65536)
            if not row:
                break
            chunks.append(row)
        after = os.fstat(fd)
        entry = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        if (_cgroup_file_identity(before) != _cgroup_file_identity(after) or
            _cgroup_file_identity(after) != _cgroup_file_identity(entry) or
            not stat.S_ISREG(entry.st_mode)):
            raise RuntimeError(f"cgroup control file identity changed: {name}")
        return b"".join(chunks)
    finally:
        os.close(fd)
        revalidate_directory_guard(guard)

def cgroup_write(guard, name, raw):
    revalidate_directory_guard(guard)
    fd = os.open(name, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                 dir_fd=guard["fd"])
    try:
        before = os.fstat(fd)
        # cgroup control writes are commands, not append streams.  Retrying a
        # short write could submit a second, different command after a partial
        # migration, so treat anything except one complete write as fatal.
        if os.write(fd, raw) != len(raw):
            raise RuntimeError(f"cgroup control write was incomplete: {name}")
        after = os.fstat(fd)
        entry = os.stat(name, dir_fd=guard["fd"], follow_symlinks=False)
        if (_cgroup_file_identity(before) != _cgroup_file_identity(after) or
            _cgroup_file_identity(after) != _cgroup_file_identity(entry) or
            not stat.S_ISREG(entry.st_mode)):
            raise RuntimeError(f"cgroup control file identity changed: {name}")
    finally:
        os.close(fd)
        revalidate_directory_guard(guard)

def current_cgroup_parent_guard():
    raw = stable_bytes(
        "/proc/self/cgroup", "current cgroup membership", pseudo=True)
    rows = [row for row in raw.decode("utf-8", errors="strict").splitlines() if row]
    unified = [row[3:] for row in rows if row.startswith("0::")]
    if len(unified) != 1:
        raise RuntimeError("delegated cgroup v2 unified membership is unavailable")
    relative = unified[0].lstrip("/")
    path = Path("/sys/fs/cgroup") / relative
    guard = open_directory_guard(path, create_missing=False)
    try:
        for name in ("cgroup.procs", "cgroup.controllers"):
            cgroup_read(guard, name)
        return guard
    except BaseException:
        close_directory_guard(guard)
        raise

def prepare_owned_cgroup(spec):
    parent = current_cgroup_parent_guard()
    suffix = spec["activation"]["sha256"][:16]
    job = "".join(ch if ch.isalnum() else "_" for ch in spec["job_id"])
    name = f"taskrev0p5_{job}_{spec['milestone']}_{suffix}"
    child = None
    try:
        child = create_child_directory_guard(parent, name)
        for control in ("cgroup.procs", "cgroup.events"):
            cgroup_read(child, control)
        if cgroup_populated(child):
            raise RuntimeError("new owned cgroup is unexpectedly populated")
        result = {"parent": parent, "child": child, "name": name,
                  "path": str(Path(parent["path"]) / name)}
        probe_owned_cgroup_kill(result)
        if cgroup_populated(child) or cgroup_processes(child):
            raise RuntimeError("owned cgroup is not empty after delegated kill probe")
        return result
    except BaseException:
        if child is not None:
            close_directory_guard(child)
            try:
                os.rmdir(name, dir_fd=parent["fd"])
            except OSError:
                pass
        close_directory_guard(parent)
        raise

def cgroup_populated(child_guard):
    values = {}
    for row in cgroup_read(child_guard, "cgroup.events").decode("ascii").splitlines():
        key, sep, value = row.partition(" ")
        if sep:
            values[key] = value
    if values.get("populated") not in ("0", "1"):
        raise RuntimeError("cgroup.events lacks exact populated state")
    return values["populated"] == "1"

def cgroup_processes(child_guard):
    rows = cgroup_read(child_guard, "cgroup.procs").decode("ascii").splitlines()
    try:
        values = [int(row) for row in rows if row]
    except ValueError as exc:
        raise RuntimeError("cgroup.procs contains a non-integer PID") from exc
    if len(values) != len(set(values)) or any(value <= 0 for value in values):
        raise RuntimeError("cgroup.procs contains invalid or duplicate PIDs")
    return sorted(values)

def move_current_to_owned_cgroup(cgroup):
    pid = os.getpid()
    cgroup_write(cgroup["child"], "cgroup.procs", b"0\n")
    if cgroup_processes(cgroup["child"]) != [pid]:
        raise RuntimeError("supervisor did not become the sole owned-cgroup member")

def move_current_to_parent_cgroup(cgroup):
    pid = os.getpid()
    cgroup_write(cgroup["parent"], "cgroup.procs", b"0\n")
    if pid in cgroup_processes(cgroup["child"]):
        raise RuntimeError("supervisor remained in the owned cgroup after migration")

def require_owned_cgroup_members(cgroup, expected, label):
    actual = cgroup_processes(cgroup["child"])
    if actual != sorted(expected):
        raise RuntimeError(
            f"{label} owned-cgroup members differ: expected={sorted(expected)} actual={actual}")

def probe_owned_cgroup_kill(cgroup, *, timeout=5.0):
    """Prove cgroup.procs, populated 0->1, cgroup.kill, and populated 1->0."""
    ready_read, ready_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        try:
            os.setsid()
            if (_PRCTL is None or
                    _PRCTL(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0):
                os._exit(126)
            move_fd = os.open(
                "cgroup.procs", os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=cgroup["child"]["fd"])
            try:
                if os.write(move_fd, b"0\n") != 2:
                    os._exit(126)
            finally:
                os.close(move_fd)
            if os.write(ready_write, b"R") != 1:
                os._exit(126)
            while True:
                signal.pause()
        except BaseException:
            os._exit(126)
    os.close(ready_write)
    kill_fd = -1
    cleanup_error = None
    try:
        ready, _, _ = select.select([ready_read], [], [], timeout)
        if not ready or os.read(ready_read, 1) != b"R":
            raise RuntimeError("delegated cgroup child migration probe timed out")
        if cgroup_processes(cgroup["child"]) != [pid] or not cgroup_populated(cgroup["child"]):
            raise RuntimeError("delegated cgroup did not observe probe membership")
        kill_fd = os.open(
            "cgroup.kill", os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=cgroup["child"]["fd"])
        if not stat.S_ISREG(os.fstat(kill_fd).st_mode):
            raise RuntimeError("cgroup.kill is not a regular control file")
        if os.write(kill_fd, b"1\n") != 2:
            raise RuntimeError("delegated cgroup.kill probe write was incomplete")
        deadline = time.monotonic() + timeout
        while cgroup_populated(cgroup["child"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        if cgroup_populated(cgroup["child"]):
            raise RuntimeError("delegated cgroup.kill probe did not reach populated=0")
        waited = 0
        status = 0
        while time.monotonic() < deadline:
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
                break
            time.sleep(0.05)
        if waited != pid or not os.WIFSIGNALED(status):
            raise RuntimeError("delegated cgroup.kill probe child did not exit by signal")
    except BaseException as exc:
        cleanup_error = exc
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    finally:
        os.close(ready_read)
        if kill_fd >= 0:
            os.close(kill_fd)
    if cleanup_error is not None:
        raise cleanup_error

def guardian_kill_cgroup_and_wait(kill_fd, child_guard, *, poll_seconds=0.1):
    """Guardian-only fail-safe for descendants that escape the leader PGID."""
    revalidate_directory_guard(child_guard)
    if cgroup_populated(child_guard):
        before = os.fstat(kill_fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("guardian cgroup.kill lease is not a regular control file")
        if os.write(kill_fd, b"1\n") != 2:
            raise RuntimeError("guardian cgroup.kill write was incomplete")
        after = os.fstat(kill_fd)
        if _cgroup_file_identity(before) != _cgroup_file_identity(after):
            raise RuntimeError("guardian cgroup.kill lease identity changed")
    while cgroup_populated(child_guard):
        time.sleep(poll_seconds)
    revalidate_directory_guard(child_guard)

def remove_owned_cgroup(cgroup):
    if cgroup_populated(cgroup["child"]):
        raise RuntimeError("cannot remove populated owned cgroup")
    close_directory_guard(cgroup["child"])
    cgroup["child"] = None
    revalidate_directory_guard(cgroup["parent"])
    os.rmdir(cgroup["name"], dir_fd=cgroup["parent"]["fd"])
    revalidate_directory_guard(cgroup["parent"])

def close_owned_cgroup_guards(cgroup):
    if cgroup is None:
        return
    close_directory_guard(cgroup.get("child"))
    close_directory_guard(cgroup.get("parent"))

def append_prefix_bytes(path, label):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"{label} is not a single-link regular append stream")
        chunks = []
        remaining = before.st_size
        while remaining:
            row = os.read(fd, min(remaining, 1024 * 1024))
            if not row:
                raise RuntimeError(f"{label} ended before its frozen append prefix")
            chunks.append(row)
            remaining -= len(row)
        after = os.fstat(fd)
        entry = path.lstat()
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink)
        if (identity(before) != identity(after) or
            identity(after) != identity(entry) or
            after.st_nlink != 1 or after.st_size < before.st_size):
            raise RuntimeError(f"{label} identity changed or shrank")
        return b"".join(chunks)
    finally:
        os.close(fd)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def publish_bytes(path, raw, mode=0o444):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(path.parent)

def publish_json(path, value, mode=0o444):
    publish_bytes(path, cbytes(value) + b"\n", mode)

def _read_proc_file(proc_fd, name):
    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=proc_fd)
    try:
        chunks = []
        while True:
            row = os.read(fd, 1024 * 1024)
            if not row:
                return b"".join(chunks)
            chunks.append(row)
    finally:
        os.close(fd)

def _parse_proc_stat(raw, pid):
    text = raw.decode("utf-8", errors="strict")
    close = text.rfind(")")
    if close < 0:
        raise ValueError("/proc stat comm terminator is absent")
    fields = text[close + 2:].split()
    if len(fields) < 20:
        raise ValueError("/proc stat has too few fields")
    return {
        "pid": pid, "state": fields[0], "ppid": int(fields[1]),
        "pgid": int(fields[2]), "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
    }

def proc_identity(pid):
    try:
        proc_fd = os.open(
            f"/proc/{pid}", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0))
        try:
            first = _parse_proc_stat(_read_proc_file(proc_fd, "stat"), pid)
            cmdline_raw = _read_proc_file(proc_fd, "cmdline")
            second = _parse_proc_stat(_read_proc_file(proc_fd, "stat"), pid)
        finally:
            os.close(proc_fd)
        stable_keys = ("pid", "ppid", "pgid", "sid", "start_ticks")
        if any(first[key] != second[key] for key in stable_keys):
            return None
        cmdline = cmdline_raw.split(b"\0")
        argv = [row.decode("utf-8", errors="surrogateescape") for row in cmdline if row]
        second["argv"] = argv
        return second
    except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError,
            ValueError, OSError):
        return None

def exact_live(expected):
    current = proc_identity(int(expected["pid"]))
    if current is None:
        return None
    keys = ("pid", "ppid", "pgid", "sid", "start_ticks", "argv")
    return current if all(current.get(key) == expected.get(key) for key in keys) else None

def parse_gpu(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip().split(",")
    if len(row) != 4:
        raise RuntimeError("nvidia-smi GPU row changed")
    total, used, free, util = [int(value.strip()) for value in row]
    apps = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10).splitlines()
    return {"total_mib": total, "used_mib": used, "free_mib": free,
            "utilization_percent": util,
            "compute_apps": sorted(set(line.strip() for line in apps if line.strip()))}

def other_evaluators():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace")
        except (OSError, PermissionError):
            continue
        if "isaac_bank_exam.py" in raw or "run_phase1_task_revision_0p5_exam.py supervisor" in raw:
            found.append({"pid": int(proc.name), "cmdline_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    return sorted(found, key=lambda row: row["pid"])

def stable_resource_gate(spec):
    gate = spec["resource_gate"]
    if other_evaluators():
        raise RuntimeError("another evaluator/supervisor is already present")
    samples = []
    for index in range(gate["stable_samples"]):
        by_gpu = {}
        for gpu in gate["allowed_eval_gpus"]:
            sample = parse_gpu(gpu)
            minimum = (gate["minimum_selected_gpu_free_mib"] if gpu == spec["eval_gpu"]
                       else gate["minimum_other_gpu_free_mib"])
            if sample["free_mib"] < minimum:
                role = "selected eval" if gpu == spec["eval_gpu"] else "cross-GPU reserve"
                raise RuntimeError(
                    f"{role} GPU {gpu} has only {sample['free_mib']} MiB free; requires {minimum}")
            by_gpu[str(gpu)] = sample
        samples.append({
            "sample_index": index,
            "selected_eval_gpu": spec["eval_gpu"],
            "minimum_selected_gpu_free_mib": gate["minimum_selected_gpu_free_mib"],
            "minimum_other_gpu_free_mib": gate["minimum_other_gpu_free_mib"],
            "gpus": by_gpu,
        })
        if index + 1 < gate["stable_samples"]:
            time.sleep(gate["sample_interval_seconds"])
    if other_evaluators():
        raise RuntimeError("another evaluator/supervisor appeared during resource gate")
    return samples

def source_environment(source):
    setup = source / REL["setup"]
    raw = subprocess.check_output(
        ["bash", "-c", 'source "$1" >/dev/null && env -0', "taskrev-0p5", str(setup)],
        timeout=30,
    )
    env = {}
    for row in raw.split(b"\0"):
        if not row:
            continue
        key, sep, value = row.partition(b"=")
        if not sep:
            raise RuntimeError("setup environment emitted malformed row")
        env[key.decode()] = value.decode()
    expected = (source / "hope_training/whole_body_tracking/source/whole_body_tracking").resolve()
    entries = env.get("HOPE_WBT_PYTHONPATH", "").split(os.pathsep)
    if not entries or Path(entries[0]).resolve() != expected:
        raise RuntimeError("pinned source is not first in HOPE_WBT_PYTHONPATH")
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible and [row.strip() for row in visible.split(",")] != ["0", "1", "2"]:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must preserve physical Pod GPU order 0,1,2")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    env["PYTHONPATH"] = env["HOPE_WBT_PYTHONPATH"]
    env.update(HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return env

def validate_restored_assets(spec):
    # Asset restoration is the sole new causal condition of activation v2.  Bind
    # both no-clobber restored outputs before any runtime materialization,
    # namespace creation, cgroup creation, supervisor, ACK, or evaluator.
    bank = validate_exact_asset(spec["exam_bank"], "restored exam bank")
    rebind_report = validate_exact_asset(
        spec["exam_rebind_report"], "restored exam-bank rebind report")
    return {"bank": bank, "rebind_report": rebind_report}

def validate_inputs(spec, *, validate_process=True):
    restored_assets = validate_restored_assets(spec)
    bank = restored_assets["bank"]
    rebind_report = restored_assets["rebind_report"]
    source = Path(spec["source_checkout"])
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, timeout=10
    ).strip() != spec["source_commit"]:
        raise RuntimeError("evaluation source HEAD differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source, text=True, timeout=20,
    ):
        raise RuntimeError("evaluation source is dirty")
    for key, relative in REL.items():
        if sha(source / relative) != spec["source_closure"][key]:
            raise RuntimeError(f"source closure mismatch: {key}")
    runtime = load_module(source / REL["runtime"], "taskrev_0p5_runtime")
    binding, bound, claim, claim_content = runtime._load_binding(Path(spec["binding_path"]))
    if (bound.get("job_id") != spec["job_id"] or bound.get("pod") != spec["pod"] or
        bound.get("gpu") != spec["training_gpu"] or bound.get("run_dir") != spec["run_dir"] or
        bound.get("claim_content_sha256") != spec["expected_claim_content_sha256"]):
        raise RuntimeError("actual binding differs from registered queue cell")
    if canonical(claim_content) != spec["expected_claim_content_sha256"]:
        raise RuntimeError("claim canonical digest differs")
    process_state = None
    if validate_process:
        process_state = runtime._verify_bound_process(bound, proc_root=Path("/proc"), getpgid=os.getpgid)
    if spec["milestone"] not in bound.get("milestones", []):
        raise RuntimeError("milestone is not registered in immutable binding")
    rsl = Path(bound["rsl_log_dir"])
    checkpoint = rsl / f"model_{spec['milestone']}.pt"
    if not checkpoint.is_file():
        raise RuntimeError("fixed checkpoint is not ready")
    milestone_path = Path(spec["milestone_receipt"])
    if not milestone_path.is_file():
        raise RuntimeError("pre-existing checkpoint milestone receipt is required; no dynamic attestation")
    milestone, milestone_raw = stable_json(milestone_path, "checkpoint milestone receipt")
    milestone_content = milestone.get("content", {})
    if (canonical(milestone_content) != milestone.get("content_sha256") or
        milestone_content.get("job_id") != spec["job_id"] or
        milestone_content.get("milestone") != spec["milestone"] or
        milestone_content.get("claim_content_sha256") != spec["expected_claim_content_sha256"]):
        raise RuntimeError("checkpoint milestone receipt differs")
    checkpoint_sha = milestone_content["checkpoint"]["sha256"]
    hard_path = Path(milestone_content["hard_contract"]["path"])
    hard_sha = milestone_content["hard_contract"]["sha256"]
    if (milestone_content["checkpoint"]["path"] != str(checkpoint) or
        sha(checkpoint) != checkpoint_sha or sha(hard_path) != hard_sha):
        raise RuntimeError("checkpoint/hard bytes changed after attestation")
    behavior_path = Path(spec["behavior_receipt"])
    if not behavior_path.is_file():
        raise RuntimeError("pre-existing behavior receipt is required")
    behavior, behavior_raw = stable_json(behavior_path, "behavior receipt")
    behavior_content = behavior.get("content", {})
    behavior_analysis = behavior_content.get("behavior", {})
    windows = behavior_analysis.get("windows")
    trailing_exact = None
    if isinstance(windows, list) and len(windows) == 2 and isinstance(windows[1], dict):
        counters = windows[1].get("counters", {})
        if isinstance(counters, dict):
            trailing_exact = counters.get("planner_initial_tts_exact_0p5_count")
    behavior_milestone = behavior_content.get("milestone_receipt", {})
    if (canonical(behavior_content) != behavior.get("content_sha256") or
        behavior_content.get("job_id") != spec["job_id"] or
        behavior_content.get("formal_evidence_eligible") is not False or
        behavior_content.get("automatic_retry") is not False or
        behavior_content.get("binding_path") != spec["binding_path"] or
        behavior_content.get("binding_content_sha256") != binding["content_sha256"] or
        behavior_content.get("claim_content_sha256") != spec["expected_claim_content_sha256"] or
        not isinstance(behavior_milestone, dict) or
        behavior_milestone.get("path") != str(milestone_path) or
        behavior_milestone.get("file_sha256") != hashlib.sha256(milestone_raw).hexdigest() or
        behavior_milestone.get("content_sha256") != milestone.get("content_sha256") or
        behavior_analysis.get("milestone") != spec["milestone"] or
        behavior_analysis.get("milestone_offset_from_parent") != spec["milestone_offset_from_parent"] or
        not isinstance(windows, list) or len(windows) != 2 or
        any(not isinstance(row, dict) or row.get("update_count") != 100 for row in windows) or
        isinstance(trailing_exact, bool) or not isinstance(trailing_exact, int) or trailing_exact <= 0):
        raise RuntimeError("behavior receipt is not exact two-window evidence with 0.5-second exposure")
    paper, paper_raw = stable_json(spec["paper"]["path"], "timing paper")
    if (hashlib.sha256(paper_raw).hexdigest() != spec["paper"]["file_sha256"] or
        paper.get("paper_semantic_sha256") != spec["paper"]["semantic_sha256"] or
        len(paper.get("rows", [])) != 100 or
        any(row.get("tts_ticks") != 25 or row.get("tts_seconds") != 0.5 for row in paper["rows"])):
        raise RuntimeError("timing paper is not the fixed exact-25-tick K100")
    schedule = Path(spec["schedule"]["path"])
    if (sha(schedule) != paper["source_schedule"]["file_sha256"] or
            sha(bank) != paper["source_schedule"]["bank_sha256"] or
            paper["source_schedule"]["bank_sha256"] != spec["exam_bank"]["sha256"]):
        raise RuntimeError("schedule/bank differs from paper")
    return {
        "source": source, "runtime": runtime, "binding": binding, "bound": bound,
        "process_state": process_state, "rsl": rsl, "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha, "hard_path": hard_path, "hard_sha": hard_sha,
        "milestone": milestone, "milestone_raw": milestone_raw,
        "behavior": behavior, "behavior_raw": behavior_raw,
        "paper": paper, "paper_raw": paper_raw, "schedule": schedule, "bank": bank,
        "rebind_report": rebind_report,
    }

def evaluator_command(spec, context):
    output = Path(spec["output_dir"])
    return [
        "/workspace/hope_isaac_venv/bin/python", str(context["source"] / REL["evaluator"]),
        "task=HOPEPingPongVirtualBall", "headless=true", f"device=cuda:{spec['eval_gpu']}",
        f"+run_dir={context['rsl']}", f"checkpoint={context['checkpoint']}",
        f"+exam_bank={context['bank']}", f"+schedule_json={context['schedule']}",
        "+per_clip_quota=50", "+schedule_seed=0", "+noise_scale=0.0",
        "+allow_inexact_contract=true", f"+timing_paper={spec['paper']['path']}",
        f"+expected_timing_paper_sha256={spec['paper']['file_sha256']}",
        f"+expected_timing_paper_semantic_sha256={spec['paper']['semantic_sha256']}",
        f"+output_dir={output}", "+output_stem=isaac_timing_0p5",
    ]

def close_fds_except(keep):
    for row in list(Path("/proc/self/fd").iterdir()):
        try:
            fd = int(row.name)
        except ValueError:
            continue
        if fd > 2 and fd not in keep:
            try:
                os.close(fd)
            except OSError:
                pass

def redirect_detached(log_fd, lock_fd, keep_fds=()):
    os.setsid()
    null = os.open("/dev/null", os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    if null > 2:
        os.close(null)
    if log_fd > 2:
        os.close(log_fd)
    close_fds_except({lock_fd, *keep_fds})

def wait_for(path, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(path).is_file():
            return True
        time.sleep(0.05)
    return Path(path).is_file()

def guarded_wait_for(guard, name, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if guarded_child_exists(guard, name):
            return True
        time.sleep(0.05)
    return guarded_child_exists(guard, name)

def heartbeat(fd, value):
    raw = cbytes(value) + b"\n"
    # One O_APPEND write is one complete JSONL record.  A short write is an
    # integrity failure; readers may ignore only a crash-truncated final row.
    if os.write(fd, raw) != len(raw):
        raise RuntimeError("heartbeat append write was incomplete")
    os.fsync(fd)

def artifact_snapshot(output, guard=None):
    if guard is not None:
        revalidate_directory_guard(guard)
    result = {}
    for name in ("isaac_timing_0p5.json", "isaac_timing_0p5.csv", "result_ledger.json", "final_receipt.json"):
        try:
            if guard is not None:
                info = guarded_file_stat(guard, name)
            else:
                info = (output / name).lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    continue
            result[name] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        except FileNotFoundError:
            pass
    if guard is not None:
        revalidate_directory_guard(guard)
    return result

def complete_handshake(output, evaluator_log, guard=None):
    if guard is not None:
        revalidate_directory_guard(guard)
    scorecard = output / "isaac_timing_0p5.json"
    scorecard_csv = output / "isaac_timing_0p5.csv"
    if guard is not None:
        try:
            for name in ("isaac_timing_0p5.json", "isaac_timing_0p5.csv", "evaluator.log"):
                guarded_file_stat(guard, name)
            raw = guarded_append_prefix_bytes(
                guard, "evaluator.log", "evaluator log")
        except (FileNotFoundError, OSError, RuntimeError):
            return False
    else:
        if not scorecard.is_file() or not scorecard_csv.is_file() or not evaluator_log.is_file():
            return False
        try:
            raw = append_prefix_bytes(evaluator_log, "evaluator log")
        except (OSError, RuntimeError):
            return False
    text = raw.decode("utf-8", errors="replace")
    result = (text.count(f"[isaac-bank-exam] JSON {scorecard}") == 1 and
              text.count(f"[isaac-bank-exam] CSV  {scorecard_csv}") == 1)
    if guard is not None:
        revalidate_directory_guard(guard)
    return result

def _owned_fingerprint(identity):
    return {
        "pid": identity["pid"], "pgid": identity["pgid"], "sid": identity["sid"],
        "start_ticks": identity["start_ticks"], "argv": identity["argv"],
    }

def _scan_process_group(pgid):
    members = []
    for row in Path("/proc").iterdir():
        if not row.name.isdigit():
            continue
        current = proc_identity(int(row.name))
        if current is not None and current["pgid"] == pgid:
            members.append(current)
    return sorted(members, key=lambda value: value["pid"])

def refresh_owned_group(leader, owned):
    if (leader["pid"] != leader["pgid"] or leader["pid"] != leader["sid"] or
        not isinstance(leader.get("argv"), list)):
        raise RuntimeError("owned process-group leader is not an isolated session leader")
    members = _scan_process_group(leader["pgid"])
    by_pid = {row["pid"]: row for row in members}
    for row in members:
        if row["sid"] != leader["pid"]:
            raise RuntimeError("process group contains a foreign session member")
        fingerprint = _owned_fingerprint(row)
        previous = owned.get(row["pid"])
        if previous is not None:
            stable_keys = ("pid", "pgid", "sid", "start_ticks")
            core_changed = any(previous[key] != fingerprint[key] for key in stable_keys)
            argv_changed_while_live = (
                row.get("state") != "Z" and previous["argv"] != fingerprint["argv"])
            if core_changed or argv_changed_while_live:
                raise RuntimeError("owned process group PID/start_ticks/argv identity drifted")
    pending = {row["pid"] for row in members if row["pid"] not in owned}
    while pending:
        progressed = False
        for pid in list(pending):
            row = by_pid[pid]
            if (pid == leader["pid"] or row["ppid"] == leader["pid"] or
                row["ppid"] in owned or
                (row["ppid"] in by_pid and row["ppid"] not in pending)):
                owned[pid] = _owned_fingerprint(row)
                pending.remove(pid)
                progressed = True
        if not progressed:
            raise RuntimeError("process-group member cannot be proven to belong to owned tree")
    return members

def leader_exited_unreaped(proc):
    if getattr(proc, "returncode", None) is not None:
        return True
    required = (getattr(os, "P_PID", None), getattr(os, "WEXITED", None),
                getattr(os, "WNOHANG", None), getattr(os, "WNOWAIT", None))
    if any(value is None for value in required) or not hasattr(os, "waitid"):
        raise RuntimeError("waitid(WNOWAIT) is required for owned process supervision")
    try:
        info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        if proc.returncode is None:
            raise RuntimeError("owned child was reaped outside its supervisor")
        return True
    return info is not None

def exact_signal_evaluator(identity, proc, grace, *, owned=None,
                           target="owned_evaluator_pgid"):
    owned = {} if owned is None else owned
    current = exact_live(identity)
    exited = leader_exited_unreaped(proc)
    if not exited and current is None:
        raise RuntimeError("owned child identity changed; refusing signal")
    if current is not None:
        owned.setdefault(current["pid"], _owned_fingerprint(current))
    before_term = refresh_owned_group(identity, owned)
    if not before_term:
        if leader_exited_unreaped(proc):
            proc.wait(timeout=max(1.0, grace))
        else:
            raise RuntimeError("owned leader is live but absent from its process group")
        return {"signal": "none_group_already_empty", "target": target,
                "pid": identity["pid"], "pgid": identity["pgid"],
                "start_ticks": identity["start_ticks"], "members_before_term": [],
                "members_before_kill": [], "group_empty_confirmed": True}
    signalable_before_term = [row for row in before_term if row.get("state") != "Z"]
    if signalable_before_term:
        os.killpg(identity["pgid"], signal.SIGTERM)
    deadline = time.monotonic() + grace
    members = before_term
    while time.monotonic() < deadline:
        members = refresh_owned_group(identity, owned)
        signalable = [row for row in members if row.get("state") != "Z"]
        if not signalable and leader_exited_unreaped(proc):
            proc.wait(timeout=max(1.0, grace))
            members = refresh_owned_group(identity, owned)
        if not members:
            return {"signal": "SIGTERM" if signalable_before_term else "none_zombies_reaped",
                    "target": target,
                    "pid": identity["pid"], "pgid": identity["pgid"],
                    "start_ticks": identity["start_ticks"],
                    "members_before_term": [_owned_fingerprint(row) for row in before_term],
                    "members_before_kill": [], "group_empty_confirmed": True}
        time.sleep(0.1)
    before_kill = refresh_owned_group(identity, owned)
    signalable_before_kill = [row for row in before_kill if row.get("state") != "Z"]
    if signalable_before_kill:
        os.killpg(identity["pgid"], signal.SIGKILL)
    kill_deadline = time.monotonic() + max(1.0, grace)
    while time.monotonic() < kill_deadline:
        members = refresh_owned_group(identity, owned)
        signalable = [row for row in members if row.get("state") != "Z"]
        if not signalable and leader_exited_unreaped(proc):
            proc.wait(timeout=max(1.0, grace))
            members = refresh_owned_group(identity, owned)
        if not members:
            return {"signal": "SIGTERM_then_SIGKILL", "target": target,
                    "pid": identity["pid"], "pgid": identity["pgid"],
                    "start_ticks": identity["start_ticks"],
                    "members_before_term": [_owned_fingerprint(row) for row in before_term],
                    "members_before_kill": [_owned_fingerprint(row) for row in before_kill],
                    "group_empty_confirmed": True}
        time.sleep(0.1)
    raise RuntimeError("owned process group remained non-empty after SIGKILL")

def bind_owned_leader(proc, command, expected_parent_pid, seconds=5.0, identity_sink=None):
    deadline = time.monotonic() + seconds
    candidate = None
    while time.monotonic() < deadline:
        current = proc_identity(proc.pid)
        if current is not None:
            candidate = current
            if (current["pid"] != current["pgid"] or current["pid"] != current["sid"] or
                current["ppid"] != expected_parent_pid):
                raise RuntimeError("spawned child is not the expected isolated child session")
            if identity_sink is not None:
                identity_sink["identity"] = current
            if current["argv"] != command:
                raise RuntimeError("spawned child argv differs from requested command")
            return current
        if leader_exited_unreaped(proc):
            break
        time.sleep(0.02)
    if candidate is not None:
        raise RuntimeError("spawned child identity could not be bound exactly")
    if not leader_exited_unreaped(proc):
        raise RuntimeError("spawned child is live but /proc identity is unavailable")
    proc.wait()
    raise RuntimeError(f"spawned child exited before identity binding rc={proc.returncode}")

def child_preexec(expected_parent_pid, cgroup_dir_fd):
    def configure_child():
        try:
            os.setsid()
            if _PRCTL is None:
                os._exit(126)
            if _PRCTL(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
                os._exit(126)
        except BaseException:
            os._exit(126)
        if os.getppid() != expected_parent_pid:
            os._exit(125)
        try:
            move_fd = os.open(
                "cgroup.procs", os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=cgroup_dir_fd)
            try:
                if os.write(move_fd, b"0\n") != 2:
                    os._exit(126)
            finally:
                os.close(move_fd)
            os.close(cgroup_dir_fd)
        except BaseException:
            os._exit(126)
    return configure_child

def guardian_main(expected_parent_pid, control_fd, ack_fd, cgroup, lock_fd):
    parent_dead = {"value": False}
    def parent_death(_number, _frame):
        parent_dead["value"] = True
    kill_fd = -1
    try:
        os.setsid()
        for number in (signal.SIGUSR1, signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(number, parent_death)
        if _PRCTL is None or _PRCTL(PR_SET_PDEATHSIG, signal.SIGUSR1, 0, 0, 0) != 0:
            return 126
        if os.getppid() != expected_parent_pid:
            parent_dead["value"] = True
        close_fds_except({control_fd, ack_fd, cgroup["parent"]["fd"],
                          cgroup["child"]["fd"], lock_fd})
        kill_fd = os.open(
            "cgroup.kill", os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=cgroup["child"]["fd"])
        if not stat.S_ISREG(os.fstat(kill_fd).st_mode):
            raise RuntimeError("guardian cgroup.kill lease is unavailable")
        os.write(ack_fd, b"R")
        normal_finish = False
        while not parent_dead["value"]:
            ready, _, _ = select.select([control_fd], [], [], 1.0)
            if not ready:
                continue
            row = os.read(control_fd, 1)
            if row == b"F":
                normal_finish = True
                break
            parent_dead["value"] = True
        had_to_kill = cgroup_populated(cgroup["child"])
        if parent_dead["value"] or had_to_kill:
            guardian_kill_cgroup_and_wait(kill_fd, cgroup["child"])
        else:
            while cgroup_populated(cgroup["child"]):
                time.sleep(0.1)
        if cgroup_populated(cgroup["child"]):
            raise RuntimeError("guardian cannot acknowledge a populated cgroup")
        os.close(kill_fd)
        kill_fd = -1
        remove_owned_cgroup(cgroup)
        cleanup_ack = b"D0" if normal_finish and not had_to_kill else b"K0"
        try:
            if os.write(ack_fd, cleanup_ack) != len(cleanup_ack):
                raise RuntimeError("guardian cleanup acknowledgment was incomplete")
        except BrokenPipeError:
            if not parent_dead["value"]:
                raise
        return 0
    except BaseException as exc:
        print(f"[taskrev-0p5-guardian] fail-safe cleanup failed: {exc}", file=sys.stderr)
        while True:
            time.sleep(60.0)
    finally:
        if kill_fd >= 0:
            try:
                os.close(kill_fd)
            except OSError:
                pass

def start_cgroup_guardian(cgroup, lock_fd, ready_timeout):
    control_read, control_write = os.pipe()
    ack_read, ack_write = os.pipe()
    parent_pid = os.getpid()
    pid = os.fork()
    if pid == 0:
        os.close(control_write)
        os.close(ack_read)
        code = guardian_main(parent_pid, control_read, ack_write, cgroup, lock_fd)
        os._exit(code)
    os.close(control_read)
    os.close(ack_write)
    try:
        ready, _, _ = select.select([ack_read], [], [], ready_timeout)
        if not ready or os.read(ack_read, 1) != b"R":
            raise RuntimeError("cgroup guardian did not publish its ready lease")
        identity = proc_identity(pid)
        if (identity is None or identity["pid"] != identity["pgid"] or
            identity["pid"] != identity["sid"] or identity["ppid"] != parent_pid):
            raise RuntimeError("cgroup guardian identity is not exact")
        return {"pid": pid, "identity": identity, "control_fd": control_write,
                "ack_fd": ack_read, "finished": False}
    except BaseException as exc:
        # Closing the only control writer makes a live guardian take its
        # fail-safe cleanup path.  Never return (and thereby release the Kit
        # lock) while that guardian's exit is unproven.
        os.close(control_write)
        os.close(ack_read)
        while True:
            waited, _status = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
                raise RuntimeError(
                    f"cgroup guardian could not establish its ready lease: {exc}"
                ) from exc
            time.sleep(0.1)

def guardian_live_exact(guardian):
    current = exact_live(guardian["identity"])
    if current is None:
        raise RuntimeError("cgroup guardian lease is not live exact")
    return current

def finish_cgroup_guardian(guardian, cgroup, timeout, *, allow_fail_safe_kill=False):
    if guardian is None or guardian.get("finished"):
        return "already_finished"
    guardian_live_exact(guardian)
    populated_before_finish = cgroup_populated(cgroup["child"])
    if populated_before_finish and not allow_fail_safe_kill:
        raise RuntimeError("owned cgroup remains populated before normal guardian finish")
    finish_deadline = time.monotonic() + timeout
    os.write(guardian["control_fd"], b"F")
    os.close(guardian["control_fd"])
    guardian["control_fd"] = -1
    ready, _, _ = select.select(
        [guardian["ack_fd"]], [], [], max(0.0, finish_deadline - time.monotonic()))
    if not ready:
        raise RuntimeError("cgroup guardian finish acknowledgment timed out")
    result = os.read(guardian["ack_fd"], 2)
    os.close(guardian["ack_fd"])
    guardian["ack_fd"] = -1
    if result not in (b"D0", b"K0"):
        raise RuntimeError("cgroup guardian returned an invalid cleanup acknowledgment")
    if result == b"K0" and not allow_fail_safe_kill:
        raise RuntimeError("cgroup guardian required fail-safe kill during normal finish")
    waited = 0
    status = 0
    while time.monotonic() < finish_deadline:
        waited, status = os.waitpid(guardian["pid"], os.WNOHANG)
        if waited == guardian["pid"]:
            break
        time.sleep(0.05)
    if waited != guardian["pid"]:
        raise RuntimeError("cgroup guardian exit timed out after cleanup acknowledgment")
    if status != 0:
        raise RuntimeError("cgroup guardian did not exit cleanly")
    guardian["finished"] = True
    guardian["finish_result"] = result.decode("ascii")
    guardian["cgroup_populated_zero_acknowledged"] = True
    return guardian["finish_result"]

def reap_failed_guardian(guardian):
    if guardian is None or guardian.get("finished"):
        return True
    if exact_live(guardian["identity"]) is not None:
        return False
    waited, status = os.waitpid(guardian["pid"], os.WNOHANG)
    if waited == 0:
        return False
    if waited != guardian["pid"]:
        raise RuntimeError("unexpected guardian waitpid result")
    guardian["finished"] = True
    guardian["finish_result"] = f"failed_status_{status}"
    return True

def close_guardian_fds(guardian):
    if guardian is None:
        return
    for key in ("control_fd", "ack_fd"):
        fd = guardian.get(key, -1)
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            guardian[key] = -1

class OwnedCleanupUnproven(RuntimeError):
    pass

def close_owned_child(proc, identity, owned, signals, grace, *, target):
    if proc is None:
        return
    if identity is None:
        if not leader_exited_unreaped(proc):
            raise OwnedCleanupUnproven(
                f"{target} remains live without a signal-safe bound identity")
        proc.wait()
        if _scan_process_group(proc.pid):
            raise OwnedCleanupUnproven(
                f"{target} left an unbound process group after leader exit")
        return
    members = refresh_owned_group(identity, owned)
    if members:
        signals.append(exact_signal_evaluator(
            identity, proc, grace, owned=owned, target=target))
    else:
        proc.wait(timeout=max(1.0, grace))
    if refresh_owned_group(identity, owned):
        raise OwnedCleanupUnproven(f"{target} process group is not empty after cleanup")

def validate_and_convert(spec, context, *, evaluator_state, signals, gpu_samples, started,
                         output_guard, stop_requested, cgroup, guardian):
    output = Path(spec["output_dir"])
    scorecard = output / "isaac_timing_0p5.json"
    scorecard_csv = output / "isaac_timing_0p5.csv"
    revalidate_directory_guard(output_guard)
    if not complete_handshake(output, output / "evaluator.log", output_guard):
        raise RuntimeError("Isaac evaluator success handshake missing or ambiguous")
    scorecard_raw = guarded_stable_bytes(
        output_guard, "isaac_timing_0p5.json", "Isaac scorecard")
    scorecard_csv_raw = guarded_stable_bytes(
        output_guard, "isaac_timing_0p5.csv", "Isaac scorecard CSV")
    scorecard_sha = hashlib.sha256(scorecard_raw).hexdigest()
    converter = context["source"] / REL["converter"]
    result_path = output / "result_ledger.json"
    convert = [
        "/workspace/hope_isaac_venv/bin/python", str(converter), "convert-isaac-scorecard",
        "--spec", str(context["source"] / REL["spec"]),
        "--expected-spec-file-sha256", spec["source_closure"]["spec"],
        "--source-schedule", str(context["schedule"]), "--paper", spec["paper"]["path"],
        "--expected-paper-file-sha256", spec["paper"]["file_sha256"],
        "--scorecard", str(scorecard), "--expected-scorecard-file-sha256", scorecard_sha,
        "--checkpoint", str(context["checkpoint"]),
        "--expected-checkpoint-file-sha256", context["checkpoint_sha"],
        "--checkpoint-hard-contract", str(context["hard_path"]),
        "--expected-checkpoint-hard-contract-file-sha256", context["hard_sha"],
        "--output", str(result_path), "--confirm", "SIM_ONLY_CONVERT_ONE_ISAAC_TIMING_SCORECARD",
    ]
    converter_log = output / "converter.log"
    converter_fd = guarded_open(
        output_guard, "converter.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    converted = None
    converter_identity = None
    converter_owned = {}
    identity_sink = {}
    parent_pid = os.getpid()
    try:
        guardian_live_exact(guardian)
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "pre-converter launch")
        try:
            converted = subprocess.Popen(
                convert, cwd=context["source"], env=source_environment(context["source"]),
                stdin=subprocess.DEVNULL, stdout=converter_fd, stderr=subprocess.STDOUT,
                close_fds=True, pass_fds=(cgroup["child"]["fd"],),
                preexec_fn=child_preexec(parent_pid, cgroup["child"]["fd"]),
            )
        finally:
            os.close(converter_fd)
            converter_fd = -1
        converter_identity = bind_owned_leader(
            converted, convert, parent_pid, identity_sink=identity_sink)
        converter_owned[converter_identity["pid"]] = _owned_fingerprint(converter_identity)
        refresh_owned_group(converter_identity, converter_owned)
        converter_deadline = time.monotonic() + spec["supervision"]["converter_timeout_seconds"]
        while not leader_exited_unreaped(converted) and time.monotonic() < converter_deadline:
            guardian_live_exact(guardian)
            revalidate_directory_guard(output_guard)
            refresh_owned_group(converter_identity, converter_owned)
            if stop_requested["signal"] is not None:
                signals.append(exact_signal_evaluator(
                    converter_identity, converted,
                    spec["supervision"]["exact_term_grace_seconds"],
                    owned=converter_owned, target="owned_converter_pgid"))
                raise RuntimeError(
                    f"supervisor received signal {stop_requested['signal']}; converter ended exactly")
            time.sleep(0.1)
        if not leader_exited_unreaped(converted):
            signals.append(exact_signal_evaluator(
                converter_identity, converted, spec["supervision"]["exact_term_grace_seconds"],
                owned=converter_owned, target="owned_converter_pgid"))
            raise RuntimeError("timing scorecard converter timed out and was ended exactly")
        remaining = refresh_owned_group(converter_identity, converter_owned)
        if any(row["pid"] != converter_identity["pid"] for row in remaining):
            raise RuntimeError("timing scorecard converter left an owned descendant alive")
        converted.wait()
        if refresh_owned_group(converter_identity, converter_owned):
            raise RuntimeError("timing scorecard converter group was not empty after leader reap")
        if converted.returncode != 0:
            raise RuntimeError(f"timing scorecard converter failed rc={converted.returncode}")
        guardian_live_exact(guardian)
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "post-converter exit")
        revalidate_directory_guard(output_guard)
    finally:
        if converter_fd >= 0:
            os.close(converter_fd)
        if converter_identity is None:
            converter_identity = identity_sink.get("identity")
            if converter_identity is not None:
                converter_owned.setdefault(
                    converter_identity["pid"], _owned_fingerprint(converter_identity))
        try:
            close_owned_child(
                converted, converter_identity, converter_owned, signals,
                spec["supervision"]["exact_term_grace_seconds"],
                target="owned_converter_pgid")
        except OwnedCleanupUnproven:
            raise
        except BaseException as cleanup_exc:
            raise OwnedCleanupUnproven(
                f"converter cleanup could not prove an empty owned process group: {cleanup_exc}"
            ) from cleanup_exc
        revalidate_directory_guard(output_guard)
    guardian_live_exact(guardian)
    require_owned_cgroup_members(
        cgroup, {os.getpid()}, "pre-result validation")
    result, result_raw = guarded_stable_json(
        output_guard, "result_ledger.json", "converted timing result")
    if (result.get("engine") != "Isaac" or result.get("evaluation_contract_exact") is not False or
        result.get("checkpoint_sha256") != context["checkpoint_sha"] or
        len(result.get("attempts", [])) != 100 or
        any(row.get("tts_ticks") != 25 for row in result["attempts"])):
        raise RuntimeError("converted result is not the bound inexact exact-25-tick Isaac K100")
    materializer = load_module(converter, "taskrev_0p5_materializer")
    spec_doc = materializer.load_spec(
        context["source"] / REL["spec"], root=context["source"],
        expected_file_sha256=spec["source_closure"]["spec"])
    source_schedule = materializer.load_source_schedule(
        context["schedule"], source_contract=spec_doc["source_schedule"])
    paper_doc = materializer.validate_paper_document(
        context["paper"], spec=spec_doc,
        spec_file_sha256=spec["source_closure"]["spec"], source_schedule=source_schedule)
    validated = materializer.validate_result_document(
        result, paper=paper_doc, paper_file_sha256=spec["paper"]["file_sha256"])
    summary = materializer.score_result(validated, paper=paper_doc)
    if (summary["evaluation_contract_exact"] is not False or
        summary["formal_gate_pass"] is not False or
        summary["time_laws_dynamics_certified"] is not False):
        raise RuntimeError("diagnostic Isaac lane attempted to claim formal evidence")
    content = {
        "schema_version": 2,
        "artifact_type": "phase1-task-revision-0p5-k100-supervised-receipt",
        "job_id": spec["job_id"], "pod": spec["pod"],
        "training_gpu": spec["training_gpu"], "eval_gpu": spec["eval_gpu"],
        "milestone": spec["milestone"],
        "milestone_offset_from_parent": spec["milestone_offset_from_parent"],
        "activation": spec["activation"], "queue": spec["queue"], "harness": spec["harness"],
        "claim_content_sha256": spec["expected_claim_content_sha256"],
        "binding_path": spec["binding_path"],
        "binding_content_sha256": context["binding"]["content_sha256"],
        "process_state_at_launch": context["process_state"],
        "milestone_receipt": {"path": spec["milestone_receipt"],
            "file_sha256": hashlib.sha256(context["milestone_raw"]).hexdigest(),
            "content_sha256": context["milestone"]["content_sha256"]},
        "behavior_receipt": {"path": spec["behavior_receipt"],
            "file_sha256": hashlib.sha256(context["behavior_raw"]).hexdigest(),
            "content_sha256": context["behavior"]["content_sha256"],
            "trailing_exact_0p5_exposure_positive": True},
        "checkpoint": {"path": str(context["checkpoint"]), "sha256": context["checkpoint_sha"]},
        "hard_contract": {"path": str(context["hard_path"]), "sha256": context["hard_sha"]},
        "paper": spec["paper"],
        "schedule": {"path": str(context["schedule"]), "sha256": sha(context["schedule"])},
        "exam_bank": {"path": str(context["bank"]), "bytes": spec["exam_bank"]["bytes"],
                      "sha256": sha(context["bank"])},
        "exam_rebind_report": {
            "path": str(context["rebind_report"]),
            "bytes": spec["exam_rebind_report"]["bytes"],
            "sha256": sha(context["rebind_report"]),
        },
        "source_commit": spec["source_commit"], "source_closure": spec["source_closure"],
        "kit_lock": spec["kit_lock"], "gpu_gate_samples": gpu_samples,
        "catastrophic_cleanup": {
            "contract": spec["catastrophic_cleanup"],
            "cgroup_path": cgroup["path"],
            "guardian": guardian["identity"],
            "guardian_live_exact_before_receipt": True,
            "cgroup_members_before_receipt": [os.getpid()],
            "supervisor_contained_before_receipt": True,
        },
        "evaluator": evaluator_state,
        "scorecard": {"path": str(scorecard), "sha256": scorecard_sha},
        "scorecard_csv": {"path": str(scorecard_csv),
                          "sha256": hashlib.sha256(scorecard_csv_raw).hexdigest()},
        "result": {"path": str(result_path), "sha256": hashlib.sha256(result_raw).hexdigest()},
        "summary": summary, "started_utc": started, "finished_utc": utc(),
        "formal_evidence_eligible": False, "evaluation_contract_exact": False,
        "engine": "Isaac", "diagnostic_only": True,
        "natural_completion": evaluator_state["exit_kind"] == "natural_rc0",
        "signals": signals, "owned_process_groups_empty_before_publish": True,
        "trainer_or_robot_signals": [], "automatic_retry": False,
        "limitations": ["Isaac-only diagnostic", "planner feasibility unobserved",
                        "self-hit/table-net safety incomplete", "time laws not TOPP/dynamics certified"],
    }
    receipt = {"schema_version": 1, "content": content, "content_sha256": canonical(content)}
    if (guarded_sha(output_guard, "isaac_timing_0p5.json", "Isaac scorecard") !=
            scorecard_sha or
        guarded_sha(output_guard, "isaac_timing_0p5.csv", "Isaac scorecard CSV") !=
            hashlib.sha256(scorecard_csv_raw).hexdigest()):
        raise RuntimeError("evaluator scorecard bytes changed before final receipt publish")
    guarded_publish_json(output_guard, "final_receipt.json", receipt)
    guarded_stable_json(
        output_guard, "final_receipt.json", "0.5-second supervised K100 final receipt")
    return receipt, summary

def publish_terminal(state_guard, content):
    terminal = {"schema_version": 1, "content": content, "content_sha256": canonical(content)}
    guarded_publish_json(state_guard, "terminal.json", terminal)
    return terminal

def supervisor_child(spec, lock_fd, supervisor_log_fd, state_guard, output_parent_guard,
                     cgroup):
    state_dir = Path(spec["state_dir"])
    output_guard = None
    heartbeat_fd = None
    proc = None
    evaluator_identity = None
    evaluator_owned = {}
    signals = []
    cleanup_confirmed = True
    identity_sink = {}
    guardian = None
    guardian_finish_result = None
    cgroup_removed = False
    commit_token_observed = False
    supervisor_in_owned_cgroup = False
    stop_requested = {"signal": None}
    try:
        redirect_detached(
            supervisor_log_fd, lock_fd,
            keep_fds=(state_guard["fd"], output_parent_guard["fd"],
                      cgroup["parent"]["fd"], cgroup["child"]["fd"]))
        revalidate_directory_guard(state_guard)
        revalidate_directory_guard(output_parent_guard)
        identity = proc_identity(os.getpid())
        if (identity is None or identity["pid"] != identity["pgid"] or
            identity["pid"] != identity["sid"]):
            raise RuntimeError("supervisor did not establish PID=PGID")
        def request_stop(number, _frame):
            stop_requested["signal"] = int(number)
        for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(number, request_stop)
        commit_deadline = time.monotonic_ns() + int(
            spec["supervision"]["commit_timeout_seconds"] * 1_000_000_000)
        hello = {
            "schema_version": 1,
            "artifact_kind": "taskrev_0p5_supervisor_hello",
            "plan_sha256": canonical(spec),
            "activation": spec["activation"],
            "job_id": spec["job_id"], "milestone": spec["milestone"],
            "pid": identity["pid"], "pgid": identity["pgid"],
            "proc_start_ticks": identity["start_ticks"],
            "argv_sha256": canonical(identity["argv"]),
            "commit_deadline_monotonic_ns": commit_deadline,
            "automatic_retry": False,
        }
        hello_path = state_dir / "child_hello.json"
        ledger_path = state_dir / "launch_ledger.json"
        token_path = state_dir / "commit_token.json"
        ack_path = state_dir / "commit_ack.json"
        guarded_publish_json(state_guard, "child_hello.json", hello)
        while (not guarded_child_exists(state_guard, "commit_token.json") and
               time.monotonic_ns() < commit_deadline):
            time.sleep(0.05)
        hello_sha = guarded_sha(state_guard, "child_hello.json", "supervisor hello")
        decision_time = time.monotonic_ns()
        if (guarded_child_exists(state_guard, "commit_token.json") and
                decision_time < commit_deadline):
            ledger, ledger_raw = guarded_stable_json(
                state_guard, "launch_ledger.json", "launch ledger")
            token, token_raw = guarded_stable_json(
                state_guard, "commit_token.json", "commit token")
            decision_candidate = {
                "schema_version": 1,
                "artifact_kind": "taskrev_0p5_launch_decision",
                "decision": "commit", "plan_sha256": canonical(spec),
                "pid": identity["pid"], "pgid": identity["pgid"],
                "proc_start_ticks": identity["start_ticks"],
                "hello_sha256": hello_sha,
                "ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
                "token_sha256": hashlib.sha256(token_raw).hexdigest(),
                "commit_deadline_monotonic_ns": commit_deadline,
                "decided_monotonic_ns": decision_time,
                "decided_utc": utc(), "retry_authorized": False,
            }
        else:
            decision_candidate = {
                "schema_version": 1,
                "artifact_kind": "taskrev_0p5_launch_decision",
                "decision": "abort_deadline", "plan_sha256": canonical(spec),
                "pid": identity["pid"], "pgid": identity["pgid"],
                "proc_start_ticks": identity["start_ticks"],
                "hello_sha256": hello_sha,
                "ledger_sha256": None, "token_sha256": None,
                "commit_deadline_monotonic_ns": commit_deadline,
                "decided_monotonic_ns": decision_time,
                "decided_utc": utc(), "retry_authorized": False,
            }
        decision, decision_raw, decision_won = publish_or_read_launch_decision(
            state_guard, decision_candidate)
        if not decision_won or decision != decision_candidate:
            raise RuntimeError("supervisor is not the unique launch-decision writer")
        if (decision.get("plan_sha256") != canonical(spec) or
                decision.get("pid") != identity["pid"] or
                decision.get("pgid") != identity["pgid"] or
                decision.get("proc_start_ticks") != identity["start_ticks"] or
                decision.get("hello_sha256") != hello_sha):
            raise RuntimeError("launch decision does not bind this exact supervisor")
        if decision.get("decision") == "abort_deadline":
            if cgroup_populated(cgroup["child"]):
                raise RuntimeError("uncommitted supervisor cgroup unexpectedly populated")
            remove_owned_cgroup(cgroup)
            cgroup_removed = True
            publish_terminal(state_guard, {
                "status": "uncommitted_deadline_abort", "retry_authorized": False,
                "job_id": spec["job_id"], "milestone": spec["milestone"],
                "launch_decision_sha256": hashlib.sha256(decision_raw).hexdigest(),
                "finished_utc": utc(), "signals": [], "trainer_or_robot_signals": [],
            })
            return 3
        if decision.get("decision") != "commit":
            raise RuntimeError("launch decision is neither commit nor deadline abort")
        commit_token_observed = True
        committed = validate_committed_chain(state_guard, spec, require_ack=False)
        guardian = start_cgroup_guardian(
            cgroup, lock_fd, spec["supervision"]["guardian_ready_timeout_seconds"])
        guardian_live_exact(guardian)
        if cgroup_populated(cgroup["child"]):
            raise RuntimeError("owned cgroup is unexpectedly populated before token validation")
        move_current_to_owned_cgroup(cgroup)
        supervisor_in_owned_cgroup = True
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "post-commit supervisor migration")
        current = exact_live(identity)
        if current is None or committed["hello"]["proc_start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("commit token/ledger does not bind this exact supervisor")
        context = validate_inputs(spec, validate_process=True)
        gpu_samples = stable_resource_gate(spec)
        output = Path(spec["output_dir"])
        if (str(output.parent) != output_parent_guard["path"] or
            guarded_child_exists(output_parent_guard, output.name)):
            raise RuntimeError("output namespace appeared after commit")
        guardian_live_exact(guardian)
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "pre-acknowledgment")
        ack = {
            "schema_version": 1, "artifact_kind": "taskrev_0p5_supervisor_commit_ack",
            "plan_sha256": canonical(spec), "pid": identity["pid"], "pgid": identity["pgid"],
            "proc_start_ticks": identity["start_ticks"],
            "hello_sha256": guarded_sha(state_guard, "child_hello.json", "supervisor hello"),
            "ledger_sha256": guarded_sha(state_guard, "launch_ledger.json", "launch ledger"),
            "token_sha256": guarded_sha(state_guard, "commit_token.json", "commit token"),
            "decision_sha256": hashlib.sha256(decision_raw).hexdigest(),
            "acknowledged_utc": utc(),
            "kit_lock_held": True, "resource_samples": gpu_samples,
            "catastrophic_cleanup": {
                "contract": spec["catastrophic_cleanup"],
                "cgroup_path": cgroup["path"],
                "cgroup_exact_members": [os.getpid()],
                "supervisor_contained": True,
                "guardian": guardian["identity"],
                "guardian_live_exact": True,
            },
            "automatic_retry": False,
        }
        guarded_publish_json(state_guard, "commit_ack.json", ack)
        validate_committed_chain(state_guard, spec, require_ack=True)
        if (exact_live(identity) is None or
            guarded_sha(state_guard, "commit_token.json", "commit token") != ack["token_sha256"]):
            raise RuntimeError("supervisor identity/token changed after acknowledgment")
        output_guard = create_child_directory_guard(output_parent_guard, output.name)
        heartbeat_path = state_dir / "heartbeat.jsonl"
        heartbeat_fd = guarded_open(
            state_guard, "heartbeat.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o444)
        evaluator_log = output / "evaluator.log"
        evaluator_fd = guarded_open(
            output_guard, "evaluator.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        command = evaluator_command(spec, context)
        started = utc()
        parent_pid = os.getpid()
        guardian_live_exact(guardian)
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "pre-evaluator launch")
        try:
            proc = subprocess.Popen(
                command, cwd=context["source"] / "hope_training/whole_body_tracking",
                env=source_environment(context["source"]), stdin=subprocess.DEVNULL,
                stdout=evaluator_fd, stderr=subprocess.STDOUT, close_fds=True,
                pass_fds=(cgroup["child"]["fd"],),
                preexec_fn=child_preexec(parent_pid, cgroup["child"]["fd"]),
            )
        finally:
            os.close(evaluator_fd)
        evaluator_identity = bind_owned_leader(
            proc, command, parent_pid, identity_sink=identity_sink)
        evaluator_owned[evaluator_identity["pid"]] = _owned_fingerprint(evaluator_identity)
        refresh_owned_group(evaluator_identity, evaluator_owned)
        launched = {
            "schema_version": 1, "artifact_kind": "taskrev_0p5_owned_evaluator",
            "pid": evaluator_identity["pid"], "pgid": evaluator_identity["pgid"],
            "proc_start_ticks": evaluator_identity["start_ticks"],
            "argv_sha256": canonical(command), "started_utc": started,
        }
        guarded_publish_json(state_guard, "evaluator_identity.json", launched)
        start_mono = time.monotonic()
        last_heartbeat = 0.0
        complete_seen = None
        exit_kind = None
        while not leader_exited_unreaped(proc):
            now = time.monotonic()
            guardian_live_exact(guardian)
            revalidate_directory_guard(state_guard)
            revalidate_directory_guard(output_guard)
            refresh_owned_group(evaluator_identity, evaluator_owned)
            if now - last_heartbeat >= spec["supervision"]["heartbeat_seconds"]:
                info = guarded_file_stat(output_guard, "evaluator.log")
                heartbeat(heartbeat_fd, {
                    "schema_version": 1, "phase": "evaluator_running",
                    "utc": utc(), "elapsed_seconds": round(now - start_mono, 3),
                    "supervisor_pid": identity["pid"],
                    "evaluator": evaluator_identity,
                    "evaluator_identity_exact": exact_live(evaluator_identity) is not None,
                    "evaluator_log_bytes": info.st_size,
                    "artifacts": artifact_snapshot(output, output_guard),
                })
                last_heartbeat = now
            if stop_requested["signal"] is not None:
                signals.append(exact_signal_evaluator(
                    evaluator_identity, proc, spec["supervision"]["exact_term_grace_seconds"],
                    owned=evaluator_owned))
                raise RuntimeError(f"supervisor received signal {stop_requested['signal']}; evaluator ended exactly")
            complete = complete_handshake(output, evaluator_log, output_guard)
            if complete and complete_seen is None:
                complete_seen = now
            if complete_seen is not None and now - complete_seen >= spec["supervision"]["completed_artifact_teardown_grace_seconds"]:
                signals.append(exact_signal_evaluator(
                    evaluator_identity, proc, spec["supervision"]["exact_term_grace_seconds"],
                    owned=evaluator_owned))
                exit_kind = "forced_after_complete_handshake_teardown_timeout"
                break
            if now - start_mono >= spec["supervision"]["evaluator_total_timeout_seconds"]:
                was_complete = complete_handshake(output, evaluator_log, output_guard)
                signals.append(exact_signal_evaluator(
                    evaluator_identity, proc, spec["supervision"]["exact_term_grace_seconds"],
                    owned=evaluator_owned))
                if not was_complete:
                    raise RuntimeError("evaluator total timeout before complete scorecard handshake")
                exit_kind = "forced_after_complete_handshake_total_timeout"
                break
            time.sleep(0.5)
        remaining = refresh_owned_group(evaluator_identity, evaluator_owned)
        if any(row["pid"] != evaluator_identity["pid"] for row in remaining):
            signals.append(exact_signal_evaluator(
                evaluator_identity, proc, spec["supervision"]["exact_term_grace_seconds"],
                owned=evaluator_owned))
            raise RuntimeError("evaluator leader exited while owned descendants remained")
        rc = proc.wait()
        if refresh_owned_group(evaluator_identity, evaluator_owned):
            raise RuntimeError("evaluator process group was not empty after leader reap")
        guardian_live_exact(guardian)
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "post-evaluator exit")
        if exit_kind is None:
            if rc != 0:
                raise RuntimeError(f"Isaac timing evaluator failed rc={rc}")
            exit_kind = "natural_rc0"
        evaluator_state = {
            "identity": evaluator_identity, "returncode": rc, "exit_kind": exit_kind,
            "teardown_natural": exit_kind == "natural_rc0",
            "owned_group_empty_confirmed": True,
            "complete_handshake_before_conversion": complete_handshake(
                output, evaluator_log, output_guard),
        }
        heartbeat(heartbeat_fd, {
            "schema_version": 1, "phase": "validating_and_converting", "utc": utc(),
            "elapsed_seconds": round(time.monotonic() - start_mono, 3),
            "evaluator": evaluator_state,
            "evaluator_log_bytes": guarded_file_stat(output_guard, "evaluator.log").st_size,
            "artifacts": artifact_snapshot(output, output_guard),
        })
        receipt, summary = validate_and_convert(
            spec, context, evaluator_state=evaluator_state, signals=signals,
            gpu_samples=gpu_samples, started=started, output_guard=output_guard,
            stop_requested=stop_requested, cgroup=cgroup, guardian=guardian)
        final_receipt_raw = cbytes(receipt) + b"\n"
        if guarded_stable_bytes(
                output_guard, "final_receipt.json", "final receipt") != final_receipt_raw:
            raise RuntimeError("final receipt path bytes differ from in-memory receipt")
        final_receipt_sha = hashlib.sha256(final_receipt_raw).hexdigest()
        require_owned_cgroup_members(
            cgroup, {os.getpid()}, "pre-guardian success handoff")
        move_current_to_parent_cgroup(cgroup)
        supervisor_in_owned_cgroup = False
        if cgroup_populated(cgroup["child"]):
            raise RuntimeError("owned cgroup is populated after supervisor success handoff")
        guardian_finish_result = finish_cgroup_guardian(
            guardian, cgroup,
            spec["supervision"]["guardian_finish_timeout_seconds"])
        if guardian_finish_result != "D0":
            raise RuntimeError("normal completion did not receive clean guardian acknowledgment")
        cgroup_removed = True
        terminal_content = {
            "status": "complete_inexact_isaac_k100",
            "retry_authorized": False, "job_id": spec["job_id"],
            "milestone": spec["milestone"], "finished_utc": utc(),
            "final_receipt": {"path": str(output / "final_receipt.json"),
                              "sha256": final_receipt_sha,
                              "content_sha256": receipt["content_sha256"]},
            "summary": summary, "evaluator": evaluator_state,
            "signals": signals, "owned_process_groups_empty": True,
            "catastrophic_cleanup": {
                "contract": spec["catastrophic_cleanup"],
                "cgroup_path": cgroup["path"],
                "guardian": guardian["identity"],
                "guardian_finish_result": guardian_finish_result,
                "cgroup_populated_zero_acknowledged": True,
                "cgroup_removed_after_populated_zero": True,
                "kit_lock_retained_by_supervisor_until_terminal": True,
            },
            "trainer_or_robot_signals": [],
        }
        revalidate_directory_guard(state_guard)
        revalidate_directory_guard(output_guard)
        publish_terminal(state_guard, terminal_content)
        heartbeat(heartbeat_fd, {
            "schema_version": 1, "phase": "terminal_complete", "utc": utc(),
            "elapsed_seconds": round(time.monotonic() - start_mono, 3),
            "terminal_content_sha256": canonical(terminal_content),
            "artifacts": artifact_snapshot(output, output_guard),
        })
        os.close(heartbeat_fd)
        heartbeat_fd = None
        return 0
    except BaseException as exc:
        cleanup_confirmed = True
        self_migration_confirmed = True
        if supervisor_in_owned_cgroup:
            try:
                move_current_to_parent_cgroup(cgroup)
                supervisor_in_owned_cgroup = False
            except BaseException as migration_exc:
                self_migration_confirmed = False
                cleanup_confirmed = False
                print(
                    f"[taskrev-0p5-supervisor] self cgroup migration failed: {migration_exc}",
                    file=sys.stderr)
        if self_migration_confirmed:
            try:
                if evaluator_identity is None:
                    evaluator_identity = identity_sink.get("identity")
                    if evaluator_identity is not None:
                        evaluator_owned.setdefault(
                            evaluator_identity["pid"], _owned_fingerprint(evaluator_identity))
                close_owned_child(
                    proc, evaluator_identity, evaluator_owned, signals,
                    spec["supervision"]["exact_term_grace_seconds"],
                    target="owned_evaluator_pgid")
            except BaseException as cleanup_exc:
                cleanup_confirmed = False
                print(f"[taskrev-0p5-supervisor] exact evaluator cleanup failed: {cleanup_exc}",
                      file=sys.stderr)
        if self_migration_confirmed:
            try:
                if guardian is not None and not guardian.get("finished"):
                    guardian_finish_result = finish_cgroup_guardian(
                        guardian, cgroup,
                        spec["supervision"]["guardian_finish_timeout_seconds"],
                        allow_fail_safe_kill=True)
                    cgroup_removed = True
                    cleanup_confirmed = bool(
                        guardian.get("cgroup_populated_zero_acknowledged") and
                        guardian_finish_result in ("D0", "K0"))
                elif guardian is not None and guardian.get("finished"):
                    guardian_finish_result = guardian.get("finish_result")
                    cleanup_confirmed = bool(
                        guardian.get("cgroup_populated_zero_acknowledged") and
                        guardian_finish_result in ("D0", "K0"))
                    cgroup_removed = cleanup_confirmed
                elif not cgroup_removed:
                    if commit_token_observed:
                        raise RuntimeError(
                            "committed launch lacks a live guardian cleanup acknowledgment")
                    if cgroup_populated(cgroup["child"]):
                        raise RuntimeError("pre-guardian owned cgroup is unexpectedly populated")
                    remove_owned_cgroup(cgroup)
                    cgroup_removed = True
                    cleanup_confirmed = True
            except BaseException as guardian_cleanup_exc:
                print(
                    f"[taskrev-0p5-supervisor] guardian cleanup failed: {guardian_cleanup_exc}",
                    file=sys.stderr)
                try:
                    if guardian is not None:
                        reap_failed_guardian(guardian)
                        guardian_finish_result = guardian.get("finish_result")
                    # Only the independent guardian owns cgroup.kill.  Missing
                    # its D0/K0 ACK is permanent quarantine, never fallback.
                    cleanup_confirmed = False
                except BaseException as fallback_cleanup_exc:
                    cleanup_confirmed = False
                    print(
                        "[taskrev-0p5-supervisor] guardian failure audit failed: "
                        f"{fallback_cleanup_exc}", file=sys.stderr)
        else:
            cleanup_confirmed = False
        if cleanup_confirmed:
            try:
                if not guarded_child_exists(state_guard, "terminal.json"):
                    publish_terminal(state_guard, {
                        "status": ("failed_no_retry" if commit_token_observed else
                                   "uncommitted_failed_no_retry"),
                        "commit_token_observed": commit_token_observed,
                        "job_id": spec.get("job_id"), "milestone": spec.get("milestone"),
                        "finished_utc": utc(), "error": f"{type(exc).__name__}: {exc}",
                        "signals": signals, "owned_process_groups_empty": True,
                        "catastrophic_cleanup": {
                            "contract": spec["catastrophic_cleanup"],
                            "cgroup_path": cgroup["path"],
                            "cgroup_removed_after_populated_zero": cgroup_removed,
                            "guardian_finish_result": guardian_finish_result,
                            "cgroup_populated_zero_acknowledged": bool(
                                guardian is not None and guardian.get(
                                    "cgroup_populated_zero_acknowledged")),
                        },
                        "trainer_or_robot_signals": [],
                    })
            except BaseException as terminal_exc:
                print(f"[taskrev-0p5-supervisor] terminal receipt publish failed: {terminal_exc}", file=sys.stderr)
        else:
            print("[taskrev-0p5-supervisor] cleanup unproven; terminal deliberately withheld",
                  file=sys.stderr)
        print(f"[taskrev-0p5-supervisor] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not cleanup_confirmed:
            quarantine = {
                "schema_version": 1,
                "artifact_kind": "taskrev_0p5_cleanup_unproven_quarantine",
                "status": "cleanup_unproven_quarantine",
                "restart_authorized": False,
                "retry_authorized": False,
                "kit_lock_held": True,
                "job_id": spec.get("job_id"),
                "milestone": spec.get("milestone"),
                "entered_utc": utc(),
                "error": f"{type(exc).__name__}: {exc}",
                "signals": signals,
                "catastrophic_cleanup": {
                    "contract": spec["catastrophic_cleanup"],
                    "cgroup_path": cgroup["path"],
                    "guardian": guardian["identity"] if guardian is not None else None,
                    "cleanup_unproven": True,
                },
                "trainer_or_robot_signals": [],
            }
            try:
                if not guarded_child_exists(state_guard, "cleanup_quarantine.json"):
                    guarded_publish_json(state_guard, "cleanup_quarantine.json", quarantine)
            except BaseException as quarantine_exc:
                print(f"[taskrev-0p5-supervisor] quarantine receipt unavailable: {quarantine_exc}",
                      file=sys.stderr)
            while True:
                try:
                    if heartbeat_fd is not None:
                        heartbeat(heartbeat_fd, {
                            "schema_version": 1,
                            "phase": "cleanup_unproven_quarantine",
                            "utc": utc(),
                            "kit_lock_held": True,
                            "restart_authorized": False,
                            "signals": signals,
                        })
                except BaseException as heartbeat_exc:
                    print(f"[taskrev-0p5-supervisor] quarantine heartbeat failed: {heartbeat_exc}",
                          file=sys.stderr)
                time.sleep(max(5.0, spec["supervision"].get("heartbeat_seconds", 30.0)))
        return 3
    finally:
        if heartbeat_fd is not None:
            try:
                os.close(heartbeat_fd)
            except OSError:
                pass
        close_directory_guard(output_guard)
        close_directory_guard(output_parent_guard)
        close_directory_guard(state_guard)
        close_guardian_fds(guardian)
        close_owned_cgroup_guards(cgroup)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(lock_fd)
        except OSError:
            pass

def validate_v2_stop_before_v3(spec):
    """Require the exact v2 failure/cleanup result before v3 consumes anything."""
    predecessor = spec.get("v2_failed_attempt")
    if not isinstance(predecessor, dict):
        raise RuntimeError("v3 launch lacks a bound v2 failed attempt")
    state = Path(predecessor.get("state_dir", ""))
    expected = predecessor.get("stop_result")
    if not state.is_absolute() or not isinstance(expected, dict):
        raise RuntimeError("v3 v2-stop binding shape differs")
    basename = expected.get("basename")
    if (not isinstance(basename, str) or Path(basename).name != basename or
            basename in {"", ".", ".."}):
        raise RuntimeError("v3 v2-stop result basename differs")
    state_guard = open_directory_guard(state, create_missing=False)
    try:
        result, raw = guarded_stable_json(
            state_guard, basename, "v2 exact-stop result")
    finally:
        close_directory_guard(state_guard)
    if (result.get("schema_version") != 1 or
            result.get("artifact_kind") != expected.get("artifact_kind") or
            result.get("status") != expected.get("status") or
            result.get("activation") != predecessor.get("activation") or
            result.get("job_id") != spec.get("job_id") or
            result.get("milestone") != spec.get("milestone") or
            result.get("failure_log_sha256") != expected.get("failure_log_sha256") or
            result.get("failure_reason") != expected.get("failure_reason") or
            result.get("signal") != expected.get("signal") or
            result.get("retry_authorized") is not expected.get("retry_authorized") or
            result.get("evaluator_direct_signal") is not expected.get("evaluator_direct_signal") or
            result.get("cgroup_kill_by_consumer") is not expected.get("cgroup_kill_by_consumer") or
            result.get("sigkill_by_consumer") is not expected.get("sigkill_by_consumer") or
            result.get("cgroup_removed") is not True or
            result.get("guardian_finish_result") not in {"D0", "K0"} or
            not _sha_text(result.get("stop_intent_sha256")) or
            not _sha_text(result.get("terminal_sha256"))):
        raise RuntimeError("v3 requires the exact completed v2 stop result")
    return {
        "path": str(state / basename),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "stop_intent_sha256": result["stop_intent_sha256"],
        "terminal_sha256": result["terminal_sha256"],
        "guardian_finish_result": result["guardian_finish_result"],
        "content": result,
    }


def claim_activation_once(spec, v2_stop_binding):
    """No-clobber v3 consumption after its v2 failure was exactly closed."""
    consumption = spec["consumption"]
    attempt = Path(consumption["v3_attempt_dir"])
    attempt_parent = None
    attempt_guard = None
    try:
        attempt_parent = open_directory_guard(attempt.parent, create_missing=True)
        if guarded_child_exists(attempt_parent, attempt.name):
            raise RuntimeError(f"native-clock v3 activation attempt is already consumed: {attempt}")
        attempt_guard = create_child_directory_guard(attempt_parent, attempt.name)
        if validate_v2_stop_before_v3(spec) != v2_stop_binding:
            raise RuntimeError("v2 exact-stop result changed before v3 attempt publication")
        guarded_publish_json(attempt_guard, "attempt.json", {
            "schema_version": 1,
            "artifact_kind": "taskrev_0p5_native_clock_v3_attempt",
            "plan_sha256": canonical(spec),
            "activation": spec["activation"],
            "prior_attempt": spec["prior_attempt"],
            "v2_failed_attempt": spec["v2_failed_attempt"],
            "v2_exact_stop_binding": v2_stop_binding,
            "exam_bank": spec["exam_bank"],
            "exam_rebind_report": spec["exam_rebind_report"],
            "status": "consumed",
            "automatic_retry": False,
            "retry_authorized": False,
            "published_utc": utc(),
        })
        return attempt_guard
    except BaseException:
        close_directory_guard(attempt_guard)
        raise
    finally:
        close_directory_guard(attempt_parent)

def launch(spec):
    # No write is permitted until both restored assets have exact size and SHA.
    validate_restored_assets(spec)
    v2_stop_binding = validate_v2_stop_before_v3(spec)
    attempt_guard = claim_activation_once(spec, v2_stop_binding)
    try:
        context = validate_inputs(spec, validate_process=True)
    except BaseException as exc:
        try:
            guarded_publish_json(attempt_guard, "preflight_failure.json", {
                "schema_version": 1,
                "artifact_kind": "taskrev_0p5_native_clock_v3_preflight_failure",
                "activation": spec["activation"],
                "status": "failed_no_retry",
                "error": f"{type(exc).__name__}: {exc}",
                "automatic_retry": False,
                "retry_authorized": False,
                "failed_utc": utc(),
            })
        finally:
            close_directory_guard(attempt_guard)
        raise
    close_directory_guard(attempt_guard)
    output = Path(spec["output_dir"])
    state_dir = Path(spec["state_dir"])
    current_process = proc_identity(os.getpid())
    if (current_process is None or not hasattr(os, "waitid") or
        getattr(os, "WNOWAIT", None) is None or _PRCTL is None):
        raise RuntimeError("Linux /proc and waitid(WNOWAIT) ownership guard is unavailable")
    output_parent_guard = None
    state_parent_guard = None
    state_guard = None
    log_fd = None
    lock_fd = None
    cgroup = None
    cgroup_transferred = False
    try:
        gpu_samples = stable_resource_gate(spec)
        output_parent_guard = open_directory_guard(output.parent, create_missing=True)
        state_parent_guard = open_directory_guard(state_dir.parent, create_missing=True)
        if guarded_child_exists(output_parent_guard, output.name):
            raise RuntimeError(f"no-clobber output namespace exists: {output}")
        if guarded_child_exists(state_parent_guard, state_dir.name):
            raise RuntimeError(f"no-clobber supervisor state exists: {state_dir}")
        revalidate_directory_guard(output_parent_guard)
        revalidate_directory_guard(state_parent_guard)
        # This is the launch preflight, not a post-launch best effort.  An
        # environment without a writable delegated cgroup-v2 child and
        # cgroup.kill fails before state/output namespaces or evaluator
        # processes are created.
        cgroup = prepare_owned_cgroup(spec)
        lock_path = Path(spec["kit_lock"])
        lock_fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise RuntimeError("Kit lock is not regular")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Kit lock busy; no supervisor namespace created") from exc
        state_guard = create_child_directory_guard(state_parent_guard, state_dir.name)
        log_fd = guarded_open(
            state_guard, "supervisor.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        pid = os.fork()
        if pid == 0:
            close_directory_guard(state_parent_guard)
            code = supervisor_child(
                spec, lock_fd, log_fd, state_guard, output_parent_guard, cgroup)
            os._exit(code)
        cgroup_transferred = True
        close_owned_cgroup_guards(cgroup)
        cgroup = None
        os.close(log_fd)
        log_fd = None
        os.close(lock_fd)
        lock_fd = None
        hello_path = state_dir / "child_hello.json"
        if not guarded_wait_for(
                state_guard, "child_hello.json", spec["supervision"]["hello_timeout_seconds"]):
            raise RuntimeError("supervisor hello timed out; no token was written")
        hello, hello_raw = guarded_stable_json(
            state_guard, "child_hello.json", "supervisor hello")
        if (hello.get("plan_sha256") != canonical(spec) or
                hello.get("activation") != spec["activation"] or
                hello.get("job_id") != spec["job_id"] or
                hello.get("milestone") != spec["milestone"] or
                hello.get("automatic_retry") is not False):
            raise RuntimeError("supervisor hello does not bind the current exact plan")
        child_identity = proc_identity(pid)
        if (child_identity is None or child_identity["pid"] != child_identity["pgid"] or
            child_identity["pid"] != child_identity["sid"] or
            hello.get("pid") != pid or hello.get("pgid") != pid or
            hello.get("proc_start_ticks") != child_identity["start_ticks"] or
            hello.get("argv_sha256") != canonical(child_identity["argv"]) or
            hello.get("plan_sha256") != canonical(spec)):
            raise RuntimeError("supervisor hello/live identity differs; no token was written")
        if time.monotonic_ns() >= hello["commit_deadline_monotonic_ns"]:
            raise RuntimeError("supervisor commit deadline expired before ledger")
        ledger = {
            "schema_version": 1, "artifact_kind": "taskrev_0p5_launch_ledger",
            "plan_sha256": canonical(spec), "activation": spec["activation"],
            "job_id": spec["job_id"], "milestone": spec["milestone"],
            "pid": pid, "pgid": pid, "proc_start_ticks": child_identity["start_ticks"],
            "hello_sha256": hashlib.sha256(hello_raw).hexdigest(),
            "output_dir": spec["output_dir"], "state_dir": spec["state_dir"],
            "resource_samples_before_fork": gpu_samples,
            "committed_utc": utc(), "automatic_retry": False,
        }
        guarded_publish_json(state_guard, "launch_ledger.json", ledger)
        ledger_raw = guarded_stable_bytes(state_guard, "launch_ledger.json", "launch ledger")
        token = {
            "schema_version": 1, "artifact_kind": "taskrev_0p5_commit_token",
            "pid": pid, "pgid": pid, "proc_start_ticks": child_identity["start_ticks"],
            "hello_sha256": hashlib.sha256(hello_raw).hexdigest(),
            "ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
            "nonce": secrets.token_hex(32), "published_utc": utc(),
            "retry_authorized": False,
        }
        guarded_publish_json(state_guard, "commit_token.json", token)
        if not guarded_wait_for(
                state_guard, "launch_decision.json",
                spec["supervision"]["commit_timeout_seconds"] + 1.0):
            raise RuntimeError("supervisor launch decision timed out after token proposal")
        decision, decision_raw = guarded_stable_json(
            state_guard, "launch_decision.json", "launch decision")
        if decision.get("decision") != "commit":
            return {
                "status": "uncommitted_deadline_abort_won",
                "retry_authorized": False, "job_id": spec["job_id"],
                "milestone": spec["milestone"], "state_dir": str(state_dir),
                "supervisor_pid": pid,
                "proc_start_ticks": child_identity["start_ticks"],
                "launch_decision_sha256": hashlib.sha256(decision_raw).hexdigest(),
            }
        validate_committed_chain(state_guard, spec, require_ack=False)
        if guarded_wait_for(
                state_guard, "commit_ack.json", spec["supervision"]["ack_observation_seconds"]):
            validate_committed_chain(state_guard, spec, require_ack=True)
            status = "running_or_committed_exact"
        else:
            status = "token_published_pending_ack"
        return {"status": status, "retry_authorized": False, "job_id": spec["job_id"],
                "milestone": spec["milestone"], "state_dir": str(state_dir),
                "supervisor_pid": pid, "proc_start_ticks": child_identity["start_ticks"],
                "launch_decision_sha256": hashlib.sha256(decision_raw).hexdigest(),
                "token_sha256": guarded_sha(
                    state_guard, "commit_token.json", "commit token")}
    finally:
        if log_fd is not None:
            os.close(log_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
        if cgroup is not None:
            if not cgroup_transferred and cgroup.get("child") is not None:
                try:
                    if not cgroup_populated(cgroup["child"]):
                        remove_owned_cgroup(cgroup)
                except BaseException:
                    pass
            close_owned_cgroup_guards(cgroup)
        close_directory_guard(state_guard)
        close_directory_guard(state_parent_guard)
        close_directory_guard(output_parent_guard)

def last_heartbeat(path, guard=None):
    if guard is not None:
        if not guarded_child_exists(guard, Path(path).name):
            return None
        raw = guarded_append_prefix_bytes(
            guard, Path(path).name, "heartbeat stream", max_bytes=16 * 1024 * 1024)
    elif not path.is_file():
        return None
    else:
        raw = append_prefix_bytes(path, "heartbeat stream")
    result = None
    rows = raw.splitlines(keepends=True)
    for index, row in enumerate(rows):
        payload = row[:-1] if row.endswith(b"\n") else row
        if not payload:
            continue
        try:
            result = strict_loads(payload, f"heartbeat row {index}")
        except Exception:
            if index == len(rows) - 1 and not row.endswith(b"\n"):
                break
            raise
    return result

def _sha_text(value):
    return isinstance(value, str) and len(value) == 64 and all(
        row in "0123456789abcdef" for row in value)

def validate_v2_exact_stop_inputs(spec):
    """Read and bind the consumed v2 process chain without mutating it."""
    if (spec.get("operation") != "exact_stop_consumed_v2_native_clock_failure" or
            spec.get("automatic_retry") is not False or
            spec.get("retry_authorized") is not False or
            spec.get("launch_authorized") is not False or
            spec.get("evaluator_direct_signal") is not False or
            spec.get("cgroup_kill_by_consumer") is not False or
            spec.get("sigkill_by_consumer") is not False):
        raise RuntimeError("v2 exact-stop authority differs")
    state = Path(spec["state_dir"])
    output = Path(spec["output_dir"])
    state_guard = open_directory_guard(state, create_missing=False)
    output_guard = None
    cgroup_guard = None
    try:
        output_guard = open_directory_guard(output, create_missing=False)
        hello, hello_raw = guarded_stable_json(
            state_guard, "child_hello.json", "v2 supervisor hello")
        ledger, ledger_raw = guarded_stable_json(
            state_guard, "launch_ledger.json", "v2 launch ledger")
        token, token_raw = guarded_stable_json(
            state_guard, "commit_token.json", "v2 commit token")
        decision, decision_raw = guarded_stable_json(
            state_guard, "launch_decision.json", "v2 launch decision")
        ack, ack_raw = guarded_stable_json(
            state_guard, "commit_ack.json", "v2 commit acknowledgment")
        hello_sha = hashlib.sha256(hello_raw).hexdigest()
        ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
        token_sha = hashlib.sha256(token_raw).hexdigest()
        decision_sha = hashlib.sha256(decision_raw).hexdigest()
        expected_supervisor = spec["supervisor"]
        plan_sha = hello.get("plan_sha256")
        if (set(hello) != {
                "schema_version", "artifact_kind", "plan_sha256", "activation",
                "job_id", "milestone", "pid", "pgid", "proc_start_ticks",
                "argv_sha256", "commit_deadline_monotonic_ns", "automatic_retry"} or
                hello.get("schema_version") != 1 or
                hello.get("artifact_kind") != "taskrev_0p5_supervisor_hello" or
                not _sha_text(plan_sha) or
                hello.get("activation") != spec["activation"] or
                hello.get("job_id") != spec["job_id"] or
                hello.get("milestone") != spec["milestone"] or
                hello.get("pid") != expected_supervisor["pid"] or
                hello.get("pgid") != expected_supervisor["pgid"] or
                hello.get("proc_start_ticks") != expected_supervisor["start_ticks"] or
                not _sha_text(hello.get("argv_sha256")) or
                hello.get("automatic_retry") is not False):
            raise RuntimeError("v2 supervisor hello differs from frozen stop target")
        if (ledger.get("artifact_kind") != "taskrev_0p5_launch_ledger" or
                ledger.get("plan_sha256") != plan_sha or
                ledger.get("activation") != spec["activation"] or
                ledger.get("job_id") != spec["job_id"] or
                ledger.get("milestone") != spec["milestone"] or
                ledger.get("pid") != expected_supervisor["pid"] or
                ledger.get("pgid") != expected_supervisor["pgid"] or
                ledger.get("proc_start_ticks") != expected_supervisor["start_ticks"] or
                ledger.get("hello_sha256") != hello_sha or
                ledger.get("state_dir") != spec["state_dir"] or
                ledger.get("output_dir") != spec["output_dir"] or
                ledger.get("automatic_retry") is not False):
            raise RuntimeError("v2 launch ledger does not bind supervisor hello")
        if (token.get("artifact_kind") != "taskrev_0p5_commit_token" or
                token.get("pid") != expected_supervisor["pid"] or
                token.get("pgid") != expected_supervisor["pgid"] or
                token.get("proc_start_ticks") != expected_supervisor["start_ticks"] or
                token.get("hello_sha256") != hello_sha or
                token.get("ledger_sha256") != ledger_sha or
                token.get("retry_authorized") is not False):
            raise RuntimeError("v2 commit token does not bind launch ledger")
        if (decision.get("artifact_kind") != "taskrev_0p5_launch_decision" or
                decision.get("decision") != "commit" or
                decision.get("plan_sha256") != plan_sha or
                decision.get("pid") != expected_supervisor["pid"] or
                decision.get("pgid") != expected_supervisor["pgid"] or
                decision.get("proc_start_ticks") != expected_supervisor["start_ticks"] or
                decision.get("hello_sha256") != hello_sha or
                decision.get("ledger_sha256") != ledger_sha or
                decision.get("token_sha256") != token_sha or
                decision.get("retry_authorized") is not False):
            raise RuntimeError("v2 commit decision does not bind token")
        catastrophic = ack.get("catastrophic_cleanup")
        if (ack.get("artifact_kind") != "taskrev_0p5_supervisor_commit_ack" or
                ack.get("plan_sha256") != plan_sha or
                ack.get("pid") != expected_supervisor["pid"] or
                ack.get("pgid") != expected_supervisor["pgid"] or
                ack.get("proc_start_ticks") != expected_supervisor["start_ticks"] or
                ack.get("hello_sha256") != hello_sha or
                ack.get("ledger_sha256") != ledger_sha or
                ack.get("token_sha256") != token_sha or
                ack.get("decision_sha256") != decision_sha or
                ack.get("kit_lock_held") is not True or
                ack.get("automatic_retry") is not False or
                not isinstance(catastrophic, dict) or
                catastrophic.get("contract") != spec["catastrophic_cleanup"] or
                catastrophic.get("cgroup_exact_members") != [expected_supervisor["pid"]] or
                catastrophic.get("supervisor_contained") is not True or
                catastrophic.get("guardian_live_exact") is not True or
                not isinstance(catastrophic.get("guardian"), dict)):
            raise RuntimeError("v2 acknowledgment does not bind cleanup chain")

        supervisor = proc_identity(expected_supervisor["pid"])
        if (supervisor is None or supervisor["pgid"] != expected_supervisor["pgid"] or
                supervisor["sid"] != expected_supervisor["sid"] or
                supervisor["start_ticks"] != expected_supervisor["start_ticks"] or
                canonical(supervisor["argv"]) != hello["argv_sha256"]):
            raise RuntimeError("v2 supervisor live identity/argv differs")
        guardian = catastrophic["guardian"]
        guardian_current = exact_live(guardian)
        if guardian_current is None:
            raise RuntimeError("v2 cleanup guardian is not live exact")
        evaluator_doc, evaluator_raw = guarded_stable_json(
            state_guard, "evaluator_identity.json", "v2 evaluator identity")
        expected_evaluator = spec["evaluator"]
        if (set(evaluator_doc) != {
                "schema_version", "artifact_kind", "pid", "pgid",
                "proc_start_ticks", "argv_sha256", "started_utc"} or
                evaluator_doc.get("schema_version") != 1 or
                evaluator_doc.get("artifact_kind") != "taskrev_0p5_owned_evaluator" or
                evaluator_doc.get("pid") != expected_evaluator["pid"] or
                evaluator_doc.get("pgid") != expected_evaluator["pgid"] or
                not _sha_text(evaluator_doc.get("argv_sha256"))):
            raise RuntimeError("v2 evaluator identity receipt differs")
        evaluator = proc_identity(expected_evaluator["pid"])
        if (evaluator is None or evaluator["pgid"] != expected_evaluator["pgid"] or
                evaluator["sid"] != expected_evaluator["sid"] or
                evaluator["start_ticks"] != evaluator_doc["proc_start_ticks"] or
                canonical(evaluator["argv"]) != evaluator_doc["argv_sha256"]):
            raise RuntimeError("v2 evaluator live identity/argv differs")

        failure_raw = guarded_stable_bytes(
            output_guard, "evaluator.log", "v2 evaluator failure log")
        if hashlib.sha256(failure_raw).hexdigest() != spec["failure_log"]["sha256"]:
            raise RuntimeError("v2 evaluator failure log SHA differs")
        reason = spec["failure_log"]["exact_reason"].encode("utf-8")
        if failure_raw.count(reason) != 1:
            raise RuntimeError("v2 evaluator failure reason is missing or ambiguous")

        cgroup_path = Path(catastrophic.get("cgroup_path", ""))
        if not cgroup_path.is_absolute():
            raise RuntimeError("v2 acknowledgment cgroup path is not absolute")
        cgroup_guard = open_directory_guard(cgroup_path, create_missing=False)
        members = cgroup_processes(cgroup_guard)
        if expected_supervisor["pid"] not in members or expected_evaluator["pid"] not in members:
            raise RuntimeError("v2 owned cgroup lacks supervisor/evaluator")
        for pid in members:
            if pid == expected_supervisor["pid"]:
                continue
            row = proc_identity(pid)
            if row is None or row["pgid"] != expected_evaluator["pgid"]:
                raise RuntimeError("v2 owned cgroup contains an unbound process")
        return {
            "plan_sha256": plan_sha,
            "chain": {
                "hello_sha256": hello_sha, "ledger_sha256": ledger_sha,
                "token_sha256": token_sha, "decision_sha256": decision_sha,
                "ack_sha256": hashlib.sha256(ack_raw).hexdigest(),
                "evaluator_identity_sha256": hashlib.sha256(evaluator_raw).hexdigest(),
            },
            "supervisor": supervisor,
            "guardian": guardian,
            "evaluator": evaluator,
            "cgroup_path": str(cgroup_path),
            "cgroup_members": members,
            "failure_log_sha256": hashlib.sha256(failure_raw).hexdigest(),
        }
    finally:
        close_directory_guard(cgroup_guard)
        close_directory_guard(output_guard)
        close_directory_guard(state_guard)

def stop_v2_exact(spec):
    """Signal only the frozen v2 supervisor once, then verify guardian cleanup."""
    first = validate_v2_exact_stop_inputs(spec)
    state_guard = open_directory_guard(Path(spec["state_dir"]), create_missing=False)
    try:
        intent = {
            "schema_version": 1,
            "artifact_kind": "taskrev_0p5_v2_exact_stop_intent",
            "operation": spec["operation"],
            "activation": spec["activation"],
            "job_id": spec["job_id"],
            "milestone": spec["milestone"],
            "supervisor": spec["supervisor"],
            "evaluator": spec["evaluator"],
            "failure_log": spec["failure_log"],
            "validated_chain": first["chain"],
            "plan_sha256": first["plan_sha256"],
            "created_utc": utc(),
            "signal": "SIGTERM_supervisor_once",
            "retry_authorized": False,
            "evaluator_direct_signal": False,
            "cgroup_kill_by_consumer": False,
            "sigkill_by_consumer": False,
        }
        guarded_publish_json(state_guard, spec["stop_intent_name"], intent)
        intent_sha = guarded_sha(
            state_guard, spec["stop_intent_name"], "v2 exact-stop intent")
    finally:
        close_directory_guard(state_guard)

    second = validate_v2_exact_stop_inputs(spec)
    if (second["chain"] != first["chain"] or
            second["plan_sha256"] != first["plan_sha256"] or
            second["failure_log_sha256"] != first["failure_log_sha256"] or
            second["supervisor"] != first["supervisor"] or
            second["evaluator"] != first["evaluator"] or
            second["guardian"] != first["guardian"] or
            second["cgroup_path"] != first["cgroup_path"]):
        raise RuntimeError("v2 exact-stop inputs changed after stop-intent publication")
    current = proc_identity(spec["supervisor"]["pid"])
    if current != second["supervisor"]:
        raise RuntimeError("v2 supervisor changed immediately before SIGTERM")
    os.kill(spec["supervisor"]["pid"], signal.SIGTERM)

    deadline = time.monotonic() + float(spec["wait_timeout_seconds"])
    state_guard = None
    terminal = None
    terminal_raw = None
    while time.monotonic() < deadline:
        if proc_identity(spec["supervisor"]["pid"]) is None:
            try:
                state_guard = open_directory_guard(
                    Path(spec["state_dir"]), create_missing=False)
                if guarded_child_exists(state_guard, "terminal.json"):
                    terminal, terminal_raw = guarded_stable_json(
                        state_guard, "terminal.json", "v2 stopped terminal")
                    break
            finally:
                close_directory_guard(state_guard)
                state_guard = None
        time.sleep(0.25)
    if terminal is None:
        raise RuntimeError("v2 exact-stop timed out; no additional signal is authorized")
    content = terminal.get("content", {})
    catastrophic = content.get("catastrophic_cleanup")
    if (canonical(content) != terminal.get("content_sha256") or
            content.get("status") != "failed_no_retry" or
            content.get("retry_authorized") is not False or
            content.get("job_id") != spec["job_id"] or
            content.get("milestone") != spec["milestone"] or
            content.get("trainer_or_robot_signals") != [] or
            content.get("owned_process_groups_empty") is not True or
            not isinstance(catastrophic, dict) or
            catastrophic.get("contract") != spec["catastrophic_cleanup"] or
            catastrophic.get("guardian_finish_result") not in {"D0", "K0"} or
            catastrophic.get("cgroup_populated_zero_acknowledged") is not True or
            catastrophic.get("cgroup_removed_after_populated_zero") is not True):
        raise RuntimeError("v2 stopped terminal lacks exact failed-no-retry cleanup proof")
    if proc_identity(spec["supervisor"]["pid"]) is not None:
        raise RuntimeError("v2 supervisor remains live after terminal")
    if exact_live(first["evaluator"]) is not None:
        raise RuntimeError("v2 evaluator remains live exact after terminal")
    if exact_live(first["guardian"]) is not None:
        raise RuntimeError("v2 guardian remains live exact after terminal")
    if Path(first["cgroup_path"]).exists():
        raise RuntimeError("v2 owned cgroup still exists after terminal")

    result = {
        "schema_version": 1,
        "artifact_kind": "taskrev_0p5_v2_exact_stop_result",
        "status": "v2_failed_no_retry_stopped_exact",
        "activation": spec["activation"],
        "job_id": spec["job_id"],
        "milestone": spec["milestone"],
        "stop_intent_sha256": intent_sha,
        "terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        "failure_log_sha256": spec["failure_log"]["sha256"],
        "failure_reason": spec["failure_log"]["exact_reason"],
        "signal": "SIGTERM_supervisor_once",
        "guardian_finish_result": catastrophic["guardian_finish_result"],
        "cgroup_removed": True,
        "evaluator_direct_signal": False,
        "cgroup_kill_by_consumer": False,
        "sigkill_by_consumer": False,
        "retry_authorized": False,
        "finished_utc": utc(),
    }
    state_guard = open_directory_guard(Path(spec["state_dir"]), create_missing=False)
    try:
        guarded_publish_json(state_guard, spec["stop_result_name"], result)
    finally:
        close_directory_guard(state_guard)
    return result

def inspect(spec):
    state_dir = Path(spec["state_dir"])
    output = Path(spec["output_dir"])
    try:
        state_guard = open_directory_guard(state_dir, create_missing=False)
    except FileNotFoundError:
        return {"status": "not_launched", "read_only": True, "retry_authorized": False,
                "job_id": spec["job_id"], "milestone": spec["milestone"]}
    output_guard = None
    try:
        if not guarded_child_exists(state_guard, "child_hello.json"):
            return {"status": "uncommitted_no_hello", "read_only": True,
                    "retry_authorized": False, "state_dir": str(state_dir)}
        hello, hello_raw = guarded_stable_json(
            state_guard, "child_hello.json", "supervisor hello")
        identity = {"pid": hello["pid"], "pgid": hello["pgid"],
                    "start_ticks": hello["proc_start_ticks"], "argv": []}
        current = proc_identity(identity["pid"])
        live_exact = bool(current and current["pgid"] == identity["pgid"] and
                          current["sid"] == identity["pid"] and
                          current["start_ticks"] == identity["start_ticks"] and
                          canonical(current["argv"]) == hello["argv_sha256"])
        try:
            output_guard = open_directory_guard(output, create_missing=False)
        except FileNotFoundError:
            output_guard = None
        log_info = guarded_file_stat(state_guard, "supervisor.log")
        heartbeat_value = last_heartbeat(state_dir / "heartbeat.jsonl", state_guard)
        heartbeat_age = None
        heartbeat_fresh = False
        if guarded_child_exists(state_guard, "heartbeat.jsonl"):
            heartbeat_info = guarded_file_stat(state_guard, "heartbeat.jsonl")
            heartbeat_age = max(0.0, (time.time_ns() - heartbeat_info.st_mtime_ns) / 1e9)
            heartbeat_fresh = heartbeat_age <= max(
                10.0, 3.0 * spec["supervision"]["heartbeat_seconds"])
        common = {
            "read_only": True, "retry_authorized": False, "state_dir": str(state_dir),
            "output_dir": str(output), "job_id": spec["job_id"], "milestone": spec["milestone"],
            "supervisor": {"pid": identity["pid"], "pgid": identity["pgid"],
                           "proc_start_ticks": identity["start_ticks"], "live_exact": live_exact},
            "heartbeat": heartbeat_value,
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_fresh": heartbeat_fresh,
            "supervisor_log_bytes": log_info.st_size,
            "artifacts": artifact_snapshot(output, output_guard) if output_guard else {},
        }
        decision = None
        decision_raw = None
        if guarded_child_exists(state_guard, "launch_decision.json"):
            decision, decision_raw = guarded_stable_json(
                state_guard, "launch_decision.json", "launch decision")
            validate_launch_decision_document(decision, decision_raw, hello, spec)
        if guarded_child_exists(state_guard, "cleanup_quarantine.json"):
            if decision is None or decision.get("decision") != "commit":
                raise RuntimeError("cleanup quarantine lacks a bound commit decision")
            validate_committed_chain(state_guard, spec, require_ack=False)
            quarantine, quarantine_raw = guarded_stable_json(
                state_guard, "cleanup_quarantine.json", "cleanup quarantine")
            if (quarantine.get("status") != "cleanup_unproven_quarantine" or
                quarantine.get("restart_authorized") is not False or
                quarantine.get("retry_authorized") is not False or
                quarantine.get("kit_lock_held") is not True):
                raise RuntimeError("cleanup quarantine receipt differs")
            common.update(
                status="cleanup_unproven_quarantine", quarantine=quarantine,
                quarantine_file_sha256=hashlib.sha256(quarantine_raw).hexdigest())
            return common
        if guarded_child_exists(state_guard, "terminal.json"):
            terminal, terminal_raw = guarded_stable_json(
                state_guard, "terminal.json", "terminal receipt")
            content = terminal.get("content", {})
            status = content.get("status")
            if (canonical(content) != terminal.get("content_sha256") or
                    content.get("retry_authorized") is not False or
                    content.get("job_id") != spec["job_id"] or
                    content.get("milestone") != spec["milestone"] or
                    content.get("trainer_or_robot_signals") != []):
                raise RuntimeError("terminal receipt canonical digest differs")
            if decision is not None and decision.get("decision") == "commit":
                validate_committed_chain(state_guard, spec, require_ack=True)
                catastrophic = content.get("catastrophic_cleanup")
                if (status not in {"complete_inexact_isaac_k100", "failed_no_retry"} or
                        not isinstance(catastrophic, dict) or
                        catastrophic.get("contract") != spec["catastrophic_cleanup"] or
                        catastrophic.get("guardian_finish_result") not in {"D0", "K0"} or
                        catastrophic.get("cgroup_populated_zero_acknowledged") is not True or
                        catastrophic.get("cgroup_removed_after_populated_zero") is not True or
                        (status == "complete_inexact_isaac_k100" and
                         catastrophic.get("guardian_finish_result") != "D0")):
                    raise RuntimeError("committed terminal lacks exact guardian cleanup proof")
            elif (decision is None or decision.get("decision") != "abort_deadline" or
                  status not in {"uncommitted_deadline_abort", "uncommitted_failed_no_retry"} or
                  (status == "uncommitted_deadline_abort" and
                   content.get("launch_decision_sha256") !=
                       hashlib.sha256(decision_raw).hexdigest())):
                raise RuntimeError("uncommitted terminal does not bind the abort decision")
            common.update(status=status, terminal=content,
                          terminal_file_sha256=hashlib.sha256(terminal_raw).hexdigest())
            return common
        if decision is None:
            common["status"] = "uncommitted_pending_decision" if live_exact else "uncommitted_child_failed"
            return common
        if decision.get("decision") == "abort_deadline":
            common["status"] = (
                "uncommitted_abort_pending_terminal" if live_exact else
                "uncommitted_abort_child_failed")
            return common
        committed = validate_committed_chain(
            state_guard, spec,
            require_ack=guarded_child_exists(state_guard, "commit_ack.json"))
        if guarded_child_exists(state_guard, "commit_ack.json"):
            guardian_identity = committed["ack"]["catastrophic_cleanup"]["guardian"]
            guardian_live = exact_live(guardian_identity) is not None
            common["guardian_live_exact"] = guardian_live
            if live_exact and guardian_live and heartbeat_fresh:
                common["status"] = "running_exact"
            elif live_exact and guardian_live:
                common["status"] = "running_stale_heartbeat"
            else:
                common["status"] = "committed_cleanup_unproven_pending_quarantine"
        else:
            common["status"] = "token_published_pending_ack" if live_exact else "committed_child_failed_no_ack"
        return common
    finally:
        close_directory_guard(output_guard)
        close_directory_guard(state_guard)

if __name__ == "__main__":
    request = strict_loads(sys.argv[1].encode(), "remote request")
    action = request.pop("_action")
    try:
        if action == "launch":
            value = launch(request)
        elif action == "inspect":
            value = inspect(request)
        elif action == "stop-v2":
            value = stop_v2_exact(request)
        else:
            raise RuntimeError("remote action must be launch, inspect, or stop-v2")
        print(MARKER + json.dumps(value, allow_nan=False, sort_keys=True), flush=True)
    except BaseException as exc:
        print(MARKER + json.dumps({"status": "failed_no_retry", "retry_authorized": False,
              "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), flush=True)
        raise
'''


def _remote_command(plan: Mapping[str, Any], *, action: str) -> str:
    if action not in {"launch", "inspect", "stop-v2"}:
        raise ExamError("remote action must be launch, inspect, or stop-v2")
    program = zlib.compress(REMOTE_PROGRAM.encode("utf-8"), level=9)
    encoded_program = base64.b64encode(program).decode("ascii")
    request = dict(plan)
    request["_action"] = action
    encoded_request = base64.b64encode(
        json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).decode("ascii")
    program_sha = hashlib.sha256(REMOTE_PROGRAM.encode("utf-8")).hexdigest()
    launcher = (
        "import base64,hashlib,sys,zlib;"
        "raw=zlib.decompress(base64.b64decode(sys.argv[1],validate=True));"
        f"assert hashlib.sha256(raw).hexdigest()=={program_sha!r};"
        "ns={'__name__':'__main__','__file__':'embedded_task_revision_0p5_supervisor.py'};"
        "sys.argv=['embedded_task_revision_0p5_supervisor.py',"
        "base64.b64decode(sys.argv[2],validate=True).decode()];"
        "exec(compile(raw,ns['__file__'],'exec'),ns)"
    )
    return shlex.join([ISAAC_PYTHON, "-B", "-c", launcher, encoded_program, encoded_request])


def remote_action(plan: Mapping[str, Any], *, action: str) -> dict[str, Any]:
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
        "-i", str(plan["ssh_key"]), "-p", str(plan["port"]),
        f"root@{plan['host']}",
        f"bash -lc {shlex.quote(_remote_command(plan, action=action))}",
    ]
    try:
        ssh_timeout = (
            plan.get("supervision", {}).get("ssh_timeout_seconds")
            if isinstance(plan.get("supervision"), Mapping)
            else None
        )
        if ssh_timeout is None:
            ssh_timeout = plan.get("ssh_timeout_seconds")
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=float(ssh_timeout), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if action == "launch":
            raise ExamError(
                "bounded SSH launch observation timed out; launch state is UNKNOWN and must be "
                "resolved with inspect, never replayed"
            ) from exc
        if action == "stop-v2":
            raise ExamError(
                "bounded SSH exact-stop timed out; stop state is UNKNOWN and no second signal "
                "is authorized"
            ) from exc
        raise ExamError("bounded read-only SSH inspect timed out; remote state is UNKNOWN") from exc
    rows = [
        line[len(RESULT_MARKER):]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(rows) != 1:
        raise ExamError(
            f"remote result marker missing/ambiguous rc={completed.returncode}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    try:
        result = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        raise ExamError("remote result marker contains malformed JSON") from exc
    if completed.returncode != 0 or result.get("status") == "failed_no_retry":
        raise ExamError(
            f"remote {action} failed; automatic replay is forbidden: {result}; "
            f"stderr={completed.stderr!r}"
        )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--queue", type=Path, default=QUEUE)
    value.add_argument("--activation", type=Path, default=ACTIVATION)
    value.add_argument("--eval-gpu", type=int, required=True)
    commands = value.add_subparsers(dest="action", required=True)
    commands.add_parser("plan", help="print the exact fixed plan without SSH")
    launch = commands.add_parser("launch", help="start one persistent supervisor")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--confirm")
    inspect = commands.add_parser("inspect", help="read persistent supervisor state")
    inspect.add_argument("--execute", action="store_true")
    stop_v2 = commands.add_parser(
        "stop-v2", help="exactly stop the consumed failed v2 supervisor"
    )
    stop_v2.add_argument("--execute", action="store_true")
    stop_v2.add_argument("--confirm")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "stop-v2":
            plan = build_v2_exact_stop_plan(
                args.queue, activation_path=args.activation, eval_gpu=args.eval_gpu
            )
        else:
            plan = build_plan(
                args.queue, activation_path=args.activation, eval_gpu=args.eval_gpu
            )
        if args.action == "plan" or not getattr(args, "execute", False):
            print(
                json.dumps(
                    {"mode": args.action, "dry_run": True, "plan": plan},
                    indent=2, sort_keys=True, allow_nan=False,
                )
            )
            return 0
        if args.action == "launch" and args.confirm != CONFIRM:
            raise ExamError("launch confirmation token mismatch")
        if args.action == "stop-v2" and args.confirm != V2_STOP_CONFIRM:
            raise ExamError("v2 exact-stop confirmation token mismatch")
        if sha256_file(args.activation.resolve()) != plan["activation"]["sha256"]:
            raise ExamError("activation changed after plan construction")
        if sha256_file(args.queue.resolve()) != plan["queue"]["sha256"]:
            raise ExamError("queue changed after plan construction")
        harness_binding = (
            plan["consumer_harness"] if args.action == "stop-v2" else plan["harness"]
        )
        if sha256_file(Path(__file__).resolve()) != harness_binding["sha256"]:
            raise ExamError("exam harness changed after plan construction")
        print(
            json.dumps(
                remote_action(plan, action=args.action),
                indent=2, sort_keys=True, allow_nan=False,
            )
        )
        return 0
    except ExamError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
