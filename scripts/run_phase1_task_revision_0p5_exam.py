#!/usr/bin/env python3
"""Run one checkpoint-bound, diagnostic 0.5-second Isaac K100 exam.

The default is a local dry-run.  ``--execute`` performs exactly one SSH call to the
checkpoint's Pod.  The remote consumer verifies the immutable queue claim/binding,
checkpoint receipt, hard contract, paper, schedule, exam bank and evaluator source
closure before it takes the shared Kit lock.  It never signals a trainer or hardware.

The resulting score is deliberately Isaac-only and inexact: the current timing rider
lacks complete safety/planner observations and its uniform phase laws are not certified
by TOPP/dynamics.
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
CONFIRM = "SIM_ONLY_RUN_ONE_TASK_REVISION_0P5_K100"
RESULT_MARKER = "TASKREV_0P5_RESULT_JSON="
ISAAC_PYTHON = "/workspace/hope_isaac_venv/bin/python"
KIT_LOCK = "/workspace/.kit_boot.lock"
MIN_FREE_GPU_MIB = 6000
SCHEDULE_PATH = (
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/"
    "signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json"
)
EXAM_BANK_PATH = (
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/"
    "schema3_exam_bank_rebind_v1/s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz"
)
PAPER_FILE_SHA256 = "6f5f152652acd0eb3a80bb5d903f617a1272e665c62c4ce3edc3fdba712f672d"
PAPER_SEMANTIC_SHA256 = "fa7e3c21d0427c4509359297596ee071ecbb06f6cfd5a8d3a252a350c6393b66"


class ExamError(RuntimeError):
    """The requested diagnostic is not exactly bound or cannot be run safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_plan(
    queue_path: Path, *, job_id: str, milestone: int, eval_gpu: int
) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    root = queue_path.parent.parent
    module = _load_queue_module(root)
    queue = module.load_queue(queue_path)
    job = module._job(queue, job_id)
    module._require_launchable_job(job)
    training_slot = module.continuation._slots(queue)[job["resource"]["required_slot"]]
    absolute = module.continuation._absolute_schedule(
        job, module.continuation._parent_records_from_job_context(job)
    )
    if milestone not in absolute["milestones"]:
        raise ExamError(
            f"{milestone} is not a registered absolute milestone for {job_id}: "
            f"{absolute['milestones']}"
        )
    pod = queue["pods"][training_slot.pod]
    if type(eval_gpu) is not int or eval_gpu not in pod["gpus"]:
        raise ExamError(f"eval GPU {eval_gpu!r} is not on {training_slot.pod}")
    claim = module.continuation._attestor_claim_spec(queue, job, training_slot)
    paper_path = str(job["exam"]["path"])
    if paper_path != (
        "/workspace/codexschema/phase1_task_revision_supercombo_20260716/"
        "papers/timing_exam_0p5_k100.schedule.json"
    ):
        raise ExamError("task-revision queue points to an unexpected timing paper")
    source = job["source"]
    closure = _source_closure(root)
    harness_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "job_id": job_id,
        "milestone": milestone,
        "milestone_offset_from_parent": milestone - absolute["parent_iteration"],
        "pod": training_slot.pod,
        "training_gpu": training_slot.gpu,
        "eval_gpu": eval_gpu,
        "host": str(pod["host"]),
        "port": int(pod["port"]),
        "ssh_key": str(Path(queue["ssh"]["key"]).expanduser()),
        "run_dir": str(job["run_dir"]),
        "binding_path": str(claim["binding_path"]),
        "expected_claim_content_sha256": str(claim["content_sha256"]),
        "source_checkout": str(source["checkout"]),
        "source_commit": str(source["commit"]),
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "harness": {"path": str(harness_path), "sha256": sha256_file(harness_path)},
        "source_closure": closure,
        "paper": {
            "path": paper_path,
            "file_sha256": PAPER_FILE_SHA256,
            "semantic_sha256": PAPER_SEMANTIC_SHA256,
        },
        "schedule": {"path": SCHEDULE_PATH},
        "exam_bank": {"path": EXAM_BANK_PATH},
        "kit_lock": KIT_LOCK,
        "min_free_gpu_mib": MIN_FREE_GPU_MIB,
        "formal_evidence_eligible": False,
        "evaluation_contract_exact": False,
    }


REMOTE_PROGRAM = r'''
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

MARKER = "TASKREV_0P5_RESULT_JSON="

def canonical(value):
    raw = json.dumps(value, allow_nan=False, ensure_ascii=False,
                     separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_json(path, label):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    sig = lambda x: (x.st_dev, x.st_ino, x.st_mode, x.st_size, x.st_mtime_ns)
    if sig(before) != sig(after):
        raise RuntimeError(f"{label} changed while reading: {path}")
    return json.loads(raw), raw

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def source_environment(source):
    setup = source / "hope_training/whole_body_tracking/setup_train_env.sh"
    raw = subprocess.check_output(
        ["bash", "-c", 'source "$1" >/dev/null && env -0', "taskrev-0p5", str(setup)]
    )
    env = {}
    for row in raw.split(b"\0"):
        if not row:
            continue
        key, sep, value = row.partition(b"=")
        if not sep:
            raise RuntimeError("setup environment emitted a malformed row")
        env[key.decode()] = value.decode()
    expected = (source / "hope_training/whole_body_tracking/source/whole_body_tracking").resolve()
    pythonpath = env.get("HOPE_WBT_PYTHONPATH", "")
    entries = pythonpath.split(os.pathsep) if pythonpath else []
    if not entries or Path(entries[0]).resolve() != expected:
        raise RuntimeError("pinned source is not first in HOPE_WBT_PYTHONPATH")
    env["PYTHONPATH"] = pythonpath
    env.update(HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return env

def publish_text(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        raw = text.encode("utf-8", errors="replace")
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)

def parse_gpu(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if len(row) != 4:
        raise RuntimeError("nvidia-smi GPU row changed")
    total, used, free, util = [int(value.strip()) for value in row]
    apps = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    return {"total_mib": total, "used_mib": used, "free_mib": free,
            "utilization_percent": util, "compute_apps": sorted(set(line.strip() for line in apps if line.strip()))}

def run(spec):
    source = Path(spec["source_checkout"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != spec["source_commit"]:
        raise RuntimeError("evaluation source HEAD differs")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=source, text=True):
        raise RuntimeError("evaluation source is dirty")
    relpaths = {
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
    for key, relative in relpaths.items():
        if sha(source / relative) != spec["source_closure"][key]:
            raise RuntimeError(f"source closure mismatch: {key}")
    runtime_path = source / relpaths["runtime"]
    runtime = load_module(runtime_path, "taskrev_0p5_runtime")
    binding, bound, claim, claim_content = runtime._load_binding(Path(spec["binding_path"]))
    if (bound.get("job_id") != spec["job_id"] or bound.get("pod") != spec["pod"] or
        bound.get("gpu") != spec["training_gpu"] or bound.get("run_dir") != spec["run_dir"] or
        bound.get("claim_content_sha256") != spec["expected_claim_content_sha256"] or
        bound.get("source") != {"checkout": spec["source_checkout"], "commit": spec["source_commit"],
                                 "ignored_runtime_asset": bound.get("source", {}).get("ignored_runtime_asset")}):
        raise RuntimeError("actual binding differs from registered queue cell")
    if canonical(claim_content) != spec["expected_claim_content_sha256"]:
        raise RuntimeError("claim canonical digest differs")
    process_state = runtime._verify_bound_process(bound, proc_root=Path("/proc"), getpgid=os.getpgid)
    if spec["milestone"] not in bound.get("milestones", []):
        raise RuntimeError("milestone is not registered in the immutable binding")
    rsl = Path(bound["rsl_log_dir"])
    checkpoint = rsl / f"model_{spec['milestone']}.pt"
    output = Path(spec["run_dir"]) / "timing_exam_0p5" / f"model_{spec['milestone']}"
    if not checkpoint.is_file():
        if output.exists() or output.is_symlink():
            raise RuntimeError("not-ready checkpoint already has an output namespace")
        print(MARKER + json.dumps({"status": "not_ready", "job_id": spec["job_id"],
              "milestone": spec["milestone"], "checkpoint": str(checkpoint)}, sort_keys=True))
        return 0
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"no-clobber output namespace already exists: {output}")
    milestone_path = Path(spec["run_dir"]) / "milestones" / f"model_{spec['milestone']}.json"
    if not milestone_path.exists():
        runtime.attest_milestone(
            spec["binding_path"], spec["milestone"],
            expected_claim_content_sha256=spec["expected_claim_content_sha256"],
            expected_job_id=spec["job_id"], expected_runtime_sha256=spec["source_closure"]["runtime"],
        )
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
    if milestone_content["checkpoint"]["path"] != str(checkpoint) or sha(checkpoint) != checkpoint_sha or sha(hard_path) != hard_sha:
        raise RuntimeError("checkpoint/hard bytes changed after attestation")
    paper, paper_raw = stable_json(spec["paper"]["path"], "timing paper")
    if (hashlib.sha256(paper_raw).hexdigest() != spec["paper"]["file_sha256"] or
        paper.get("paper_semantic_sha256") != spec["paper"]["semantic_sha256"] or
        len(paper.get("rows", [])) != 100):
        raise RuntimeError("timing paper binding differs")
    schedule_path = Path(spec["schedule"]["path"])
    exam_bank_path = Path(spec["exam_bank"]["path"])
    if sha(schedule_path) != paper["source_schedule"]["file_sha256"]:
        raise RuntimeError("source schedule differs from paper")
    if sha(exam_bank_path) != paper["source_schedule"]["bank_sha256"]:
        raise RuntimeError("exam bank differs from paper")
    if any(row.get("tts_ticks") != 25 or row.get("tts_seconds") != 0.5 for row in paper["rows"]):
        raise RuntimeError("paper is not exact 25-tick K100")
    gpu = parse_gpu(spec["eval_gpu"])
    if gpu["free_mib"] < spec["min_free_gpu_mib"]:
        raise RuntimeError(f"eval GPU has only {gpu['free_mib']} MiB free")
    lock_path = Path(spec["kit_lock"])
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise RuntimeError("Kit lock is not a regular file")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Kit lock is busy; no evaluator namespace was created") from exc
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        output.mkdir(mode=0o700)
        env = source_environment(source)
        scorecard = output / "isaac_timing_0p5.json"
        scorecard_csv = output / "isaac_timing_0p5.csv"
        evaluator = source / relpaths["evaluator"]
        command = [
            "/workspace/hope_isaac_venv/bin/python", str(evaluator),
            "task=HOPEPingPongVirtualBall", "headless=true", f"device=cuda:{spec['eval_gpu']}",
            f"+run_dir={rsl}", f"checkpoint={checkpoint}", f"+exam_bank={exam_bank_path}",
            f"+schedule_json={schedule_path}", "+per_clip_quota=50", "+schedule_seed=0",
            "+noise_scale=0.0", "+allow_inexact_contract=true",
            f"+timing_paper={spec['paper']['path']}",
            f"+expected_timing_paper_sha256={spec['paper']['file_sha256']}",
            f"+expected_timing_paper_semantic_sha256={spec['paper']['semantic_sha256']}",
            f"+output_dir={output}", "+output_stem=isaac_timing_0p5",
        ]
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        completed = subprocess.run(command, cwd=source / "hope_training/whole_body_tracking",
                                   env=env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        publish_text(output / "evaluator.log", completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"Isaac timing evaluator failed rc={completed.returncode}; namespace preserved")
        if (completed.stdout.count(f"[isaac-bank-exam] JSON {scorecard}") != 1 or
            completed.stdout.count(f"[isaac-bank-exam] CSV  {scorecard_csv}") != 1):
            raise RuntimeError("Isaac evaluator success handshake missing or ambiguous")
        scorecard_sha = sha(scorecard)
        result_path = output / "result_ledger.json"
        converter = source / relpaths["converter"]
        convert = [
            "/workspace/hope_isaac_venv/bin/python", str(converter), "convert-isaac-scorecard",
            "--spec", str(source / relpaths["spec"]), "--expected-spec-file-sha256", spec["source_closure"]["spec"],
            "--source-schedule", str(schedule_path), "--paper", spec["paper"]["path"],
            "--expected-paper-file-sha256", spec["paper"]["file_sha256"],
            "--scorecard", str(scorecard), "--expected-scorecard-file-sha256", scorecard_sha,
            "--checkpoint", str(checkpoint), "--expected-checkpoint-file-sha256", checkpoint_sha,
            "--checkpoint-hard-contract", str(hard_path),
            "--expected-checkpoint-hard-contract-file-sha256", hard_sha,
            "--output", str(result_path), "--confirm", "SIM_ONLY_CONVERT_ONE_ISAAC_TIMING_SCORECARD",
        ]
        converted = subprocess.run(convert, cwd=source, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        publish_text(output / "converter.log", converted.stdout)
        if converted.returncode != 0:
            raise RuntimeError(f"timing scorecard converter failed rc={converted.returncode}")
        result, result_raw = stable_json(result_path, "converted timing result")
        if (result.get("engine") != "Isaac" or result.get("evaluation_contract_exact") is not False or
            result.get("checkpoint_sha256") != checkpoint_sha or len(result.get("attempts", [])) != 100):
            raise RuntimeError("converted result is not the bound inexact Isaac K100")
        if any(row.get("tts_ticks") != 25 for row in result["attempts"]):
            raise RuntimeError("converted result contains a non-25-tick attempt")
        materializer = load_module(converter, "taskrev_0p5_materializer")
        spec_doc = materializer.load_spec(source / relpaths["spec"], root=source,
                                          expected_file_sha256=spec["source_closure"]["spec"])
        source_schedule = materializer.load_source_schedule(schedule_path,
                                                             source_contract=spec_doc["source_schedule"])
        paper_doc = materializer.validate_paper_document(paper, spec=spec_doc,
                                                          spec_file_sha256=spec["source_closure"]["spec"],
                                                          source_schedule=source_schedule)
        validated_result = materializer.validate_result_document(result, paper=paper_doc,
                                                                   paper_file_sha256=spec["paper"]["file_sha256"])
        summary = materializer.score_result(validated_result, paper=paper_doc)
        if (summary["evaluation_contract_exact"] is not False or summary["formal_gate_pass"] is not False or
            summary["time_laws_dynamics_certified"] is not False):
            raise RuntimeError("inexact Isaac timing lane attempted to claim a formal pass")
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        content = {
            "schema_version": 1, "artifact_type": "phase1-task-revision-0p5-k100-receipt",
            "job_id": spec["job_id"], "pod": spec["pod"], "training_gpu": spec["training_gpu"],
            "eval_gpu": spec["eval_gpu"], "milestone": spec["milestone"],
            "milestone_offset_from_parent": spec["milestone_offset_from_parent"],
            "formal_evidence_eligible": False, "evaluation_contract_exact": False,
            "engine": "Isaac", "diagnostic_only": True,
            "queue": spec["queue"], "harness": spec["harness"],
            "claim_content_sha256": spec["expected_claim_content_sha256"],
            "binding_path": spec["binding_path"], "binding_content_sha256": binding["content_sha256"],
            "process_state_at_exam": process_state,
            "milestone_receipt": {"path": str(milestone_path),
                                   "file_sha256": hashlib.sha256(milestone_raw).hexdigest(),
                                   "content_sha256": milestone["content_sha256"]},
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "hard_contract": {"path": str(hard_path), "sha256": hard_sha},
            "paper": spec["paper"],
            "schedule": {"path": str(schedule_path), "sha256": sha(schedule_path)},
            "exam_bank": {"path": str(exam_bank_path), "sha256": sha(exam_bank_path)},
            "source_commit": spec["source_commit"], "source_closure": spec["source_closure"],
            "kit_lock": spec["kit_lock"], "gpu_before_exam": gpu,
            "scorecard": {"path": str(scorecard), "sha256": scorecard_sha},
            "scorecard_csv": {"path": str(scorecard_csv), "sha256": sha(scorecard_csv)},
            "result": {"path": str(result_path), "sha256": hashlib.sha256(result_raw).hexdigest()},
            "summary": summary, "started_utc": started, "finished_utc": finished,
            "limitations": ["Isaac-only diagnostic", "planner feasibility unobserved",
                            "self-hit/table-net safety incomplete", "time laws not TOPP/dynamics certified"],
            "trainer_or_robot_signals": [],
        }
        receipt = {"schema_version": 1, "content": content, "content_sha256": canonical(content)}
        receipt_path = output / "final_receipt.json"
        runtime._atomic_publish_json(receipt_path, receipt, "0.5-second K100 final receipt")
        print(MARKER + json.dumps({"status": "complete_inexact_isaac_k100", "receipt_path": str(receipt_path),
              "receipt_file_sha256": sha(receipt_path), "receipt_content_sha256": receipt["content_sha256"],
              "summary": summary}, allow_nan=False, sort_keys=True))
        return 0
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    try:
        raise SystemExit(run(request))
    except Exception as exc:
        print(MARKER + json.dumps({"status": "failed_no_retry", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise
'''


def _remote_command(plan: Mapping[str, Any]) -> str:
    program = zlib.compress(REMOTE_PROGRAM.encode("utf-8"), level=9)
    encoded_program = base64.b64encode(program).decode("ascii")
    encoded_plan = base64.b64encode(
        json.dumps(plan, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).decode("ascii")
    program_sha = hashlib.sha256(REMOTE_PROGRAM.encode("utf-8")).hexdigest()
    launcher = (
        "import base64,hashlib,json,sys,zlib;"
        "raw=zlib.decompress(base64.b64decode(sys.argv[1],validate=True));"
        f"assert hashlib.sha256(raw).hexdigest()=={program_sha!r};"
        "ns={'__name__':'__main__','__file__':'embedded_task_revision_0p5_exam.py'};"
        "sys.argv=['embedded_task_revision_0p5_exam.py',base64.b64decode(sys.argv[2],validate=True).decode()];"
        "exec(compile(raw,ns['__file__'],'exec'),ns)"
    )
    return shlex.join([ISAAC_PYTHON, "-B", "-c", launcher, encoded_program, encoded_plan])


def execute(plan: Mapping[str, Any]) -> dict[str, Any]:
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
        "-i", str(plan["ssh_key"]), "-p", str(plan["port"]),
        f"root@{plan['host']}", f"bash -lc {shlex.quote(_remote_command(plan))}",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    rows = [line[len(RESULT_MARKER):] for line in completed.stdout.splitlines()
            if line.startswith(RESULT_MARKER)]
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
            f"remote K100 failed and must not be replayed: {result}; stderr={completed.stderr!r}"
        )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--queue", type=Path, default=QUEUE)
    value.add_argument("--job-id", required=True)
    value.add_argument("--milestone", type=int, required=True)
    value.add_argument("--eval-gpu", type=int, required=True)
    value.add_argument("--execute", action="store_true")
    value.add_argument("--confirm")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = build_plan(
            args.queue, job_id=args.job_id, milestone=args.milestone, eval_gpu=args.eval_gpu
        )
        if not args.execute:
            print(json.dumps({"mode": "inspect", "dry_run": True, "plan": plan},
                             indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.confirm != CONFIRM:
            raise ExamError("execute confirmation token mismatch")
        queue_path = args.queue.resolve()
        if sha256_file(queue_path) != plan["queue"]["sha256"]:
            raise ExamError("queue changed after plan construction")
        if sha256_file(Path(__file__).resolve()) != plan["harness"]["sha256"]:
            raise ExamError("exam harness changed after plan construction")
        print(json.dumps(execute(plan), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except ExamError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
