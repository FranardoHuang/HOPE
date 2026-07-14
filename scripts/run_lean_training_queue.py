#!/usr/bin/env python3
"""Small YAML-driven RunPod training queue.

The queue is intentionally an exploratory-run tool, not a formal evidence
attestor.  One YAML row binds the motion/action, its train bank and exam, the
source checkout, base recipe, causal delta, seed, budget, checkpoint cadence,
and resource policy.  ``plan`` and ``launch-next`` are dry-run by default.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys
from typing import Any

import yaml


class QueueError(RuntimeError):
    """The lightweight queue contract or a launch preflight failed."""


@dataclass(frozen=True)
class Slot:
    pod: str
    gpu: int
    ordinal: int
    capacity: int

    @property
    def name(self) -> str:
        return f"{self.pod}/gpu{self.gpu}"


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
READY = "ready"
BLOCKED = "blocked"
TERMINAL = {"complete", "rejected"}
CONFIRM = "SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB"
ZERO_COMMIT = "0" * 40
GLOBAL_SCHEDULER_LOCK = Path("/tmp/hope_lean_training_queue.global.lock")
ISAAC_PYTHON = "/workspace/hope_isaac_venv/bin/python"
WBT_RELATIVE = "hope_training/whole_body_tracking"
SETUP_RELATIVE = "setup_train_env.sh"
ENTRYPOINT_RELATIVE = "scripts/train.py"
KIT_LAUNCHER_RELATIVE = "scripts/launch_kit_training_locked.sh"
KIT_BOOT_MARKER = "[train.py] hard training contract:"
KIT_BOOT_TIMEOUT_SECONDS = 900
UNIQUE_NUMERIC_PID_AWK = (
    r'{gsub(/^[ \t]+|[ \t]+$/, "", $0); '
    r'if ($0 ~ /^[0-9]+$/) seen[$0]=1} END {print length(seen)}'
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QueueError(f"{label} must be a list")
    return value


def _text(value: Any, label: str, *, safe_id: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueueError(f"{label} must be a non-empty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise QueueError(f"{label} must be one line")
    if safe_id and not SAFE_ID.fullmatch(value):
        raise QueueError(f"{label} is not a safe identifier")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise QueueError(f"{label} must be a positive integer")
    return value


def _ready_workspace_path(value: Any, label: str) -> str:
    path = _text(value, label)
    parsed = PurePosixPath(path)
    lowered = path.lower()
    if not parsed.is_absolute() or not path.startswith("/workspace/"):
        raise QueueError(f"{label} for a ready job must be an absolute /workspace path")
    if ".." in parsed.parts:
        raise QueueError(f"{label} for a ready job must not contain ..")
    if any(token in lowered for token in ("placeholder", "/path/to/", "<", ">")):
        raise QueueError(f"{label} for a ready job is still a placeholder")
    return path


def load_queue(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise QueueError(f"cannot read queue YAML {path}: {exc}") from exc
    queue = _mapping(raw, "queue")
    if queue.get("schema_version") != 1:
        raise QueueError("schema_version must be 1")
    if queue.get("simulation_only") is not True:
        raise QueueError("simulation_only must be true")

    ssh = _mapping(queue.get("ssh"), "ssh")
    _text(ssh.get("key"), "ssh.key")
    pods = _mapping(queue.get("pods"), "pods")
    if list(pods) != ["pod1", "pod2"]:
        raise QueueError("pods must be ordered exactly pod1, pod2")
    for pod_name, expected_capacity in (("pod1", 4), ("pod2", 3)):
        pod = _mapping(pods[pod_name], pod_name)
        _text(pod.get("host"), f"{pod_name}.host")
        _positive_int(pod.get("port"), f"{pod_name}.port")
        if pod.get("gpus") != [0, 1, 2]:
            raise QueueError(f"{pod_name}.gpus must be [0, 1, 2]")
        if pod.get("max_trainers_per_gpu") != expected_capacity:
            raise QueueError(
                f"{pod_name}.max_trainers_per_gpu must be {expected_capacity}"
            )

    if "runner" in queue:
        raise QueueError(
            "runner paths are source-pinned; queue YAML must not override setup/train/launcher"
        )

    jobs = _list(queue.get("jobs"), "jobs")
    if not jobs:
        raise QueueError("jobs must not be empty")
    seen: set[str] = set()
    run_names: set[str] = set()
    for index, value in enumerate(jobs):
        job = _mapping(value, f"jobs[{index}]")
        job_id = _text(job.get("id"), f"jobs[{index}].id", safe_id=True)
        if job_id in seen:
            raise QueueError(f"duplicate job id: {job_id}")
        seen.add(job_id)
        _text(job.get("human_name"), f"{job_id}.human_name")
        _text(job.get("action"), f"{job_id}.action", safe_id=True)
        status = job.get("status")
        if status not in {READY, BLOCKED, *TERMINAL}:
            raise QueueError(f"{job_id}.status must be ready/blocked/complete/rejected")
        blocker = job.get("blocker")
        if status == BLOCKED:
            _text(blocker, f"{job_id}.blocker")
        elif blocker not in (None, ""):
            raise QueueError(f"{job_id}.blocker must be empty unless status=blocked")

        action = job["action"]
        motion = _mapping(job.get("motion"), f"{job_id}.motion")
        if motion.get("action") != action:
            raise QueueError(f"{job_id}.motion.action must equal job action")
        bindings = _mapping(motion.get("bindings"), f"{job_id}.motion.bindings")
        if not bindings:
            raise QueueError(f"{job_id}.motion.bindings must not be empty")
        for arg, asset_path in bindings.items():
            _text(arg, f"{job_id}.motion arg", safe_id=True)
            _text(asset_path, f"{job_id}.motion.{arg}")

        bank = _mapping(job.get("bank"), f"{job_id}.bank")
        if bank.get("action") != action:
            raise QueueError(f"{job_id}.bank.action must equal job action")
        _text(bank.get("train_path"), f"{job_id}.bank.train_path")
        _text(bank.get("train_arg"), f"{job_id}.bank.train_arg")
        exam = _mapping(job.get("exam"), f"{job_id}.exam")
        if exam.get("action") != action:
            raise QueueError(f"{job_id}.exam.action must equal job action")
        _text(exam.get("path"), f"{job_id}.exam.path")
        _text(exam.get("family"), f"{job_id}.exam.family", safe_id=True)

        source = _mapping(job.get("source"), f"{job_id}.source")
        _text(source.get("checkout"), f"{job_id}.source.checkout")
        commit = _text(source.get("commit"), f"{job_id}.source.commit")
        if not COMMIT.fullmatch(commit):
            raise QueueError(f"{job_id}.source.commit must be a full Git commit")

        recipe = _mapping(job.get("recipe"), f"{job_id}.recipe")
        base = _list(recipe.get("base"), f"{job_id}.recipe.base")
        delta = _list(recipe.get("delta"), f"{job_id}.recipe.delta")
        if not base:
            raise QueueError(f"{job_id}.recipe.base must not be empty")
        for number, argument in enumerate([*base, *delta]):
            _text(argument, f"{job_id}.recipe argument {number}")

        if type(job.get("seed")) is not int or job["seed"] < 0:
            raise QueueError(f"{job_id}.seed must be a non-negative integer")
        budget = _mapping(job.get("budget"), f"{job_id}.budget")
        _positive_int(budget.get("num_envs"), f"{job_id}.budget.num_envs")
        iterations = _positive_int(
            budget.get("max_iterations"), f"{job_id}.budget.max_iterations"
        )
        save_interval = _positive_int(
            budget.get("save_interval"), f"{job_id}.budget.save_interval"
        )
        milestones = _list(job.get("milestones"), f"{job_id}.milestones")
        if not milestones or any(type(x) is not int or x <= 0 for x in milestones):
            raise QueueError(f"{job_id}.milestones must contain positive integers")
        if milestones != sorted(set(milestones)) or milestones[-1] > iterations:
            raise QueueError(
                f"{job_id}.milestones must be unique, sorted, and within max_iterations"
            )
        if any(x % save_interval for x in milestones):
            raise QueueError(f"{job_id}.milestones must align with save_interval")

        resource = _mapping(job.get("resource"), f"{job_id}.resource")
        if resource != {"policy": "six_gpu_round_robin"}:
            raise QueueError(
                f"{job_id}.resource must bind policy six_gpu_round_robin"
            )
        run_name = _text(job.get("run_name"), f"{job_id}.run_name", safe_id=True)
        if run_name in run_names:
            raise QueueError(f"duplicate run_name: {run_name}")
        run_names.add(run_name)
        _text(job.get("run_dir"), f"{job_id}.run_dir")
        if status == READY:
            if commit == ZERO_COMMIT:
                raise QueueError(f"{job_id}.source.commit is an all-zero placeholder")
            ready_paths = {
                "source.checkout": source["checkout"],
                **{
                    f"motion.{arg}": asset_path
                    for arg, asset_path in bindings.items()
                },
                "bank.train_path": bank["train_path"],
                "exam.path": exam["path"],
                "run_dir": job["run_dir"],
            }
            normalized = [
                _ready_workspace_path(path_value, f"{job_id}.{path_label}")
                for path_label, path_value in ready_paths.items()
            ]
            input_paths = normalized[1:-1]
            if len(set(input_paths)) != len(input_paths):
                raise QueueError(f"{job_id} has duplicate motion/bank/exam identities")
    return queue


def slots(queue: dict[str, Any]) -> list[Slot]:
    result: list[Slot] = []
    ordinal = 0
    # One full six-GPU round before any GPU receives its next trainer.
    for pod_name in ("pod1", "pod2"):
        pod = queue["pods"][pod_name]
        for gpu in pod["gpus"]:
            result.append(
                Slot(pod_name, gpu, ordinal, pod["max_trainers_per_gpu"])
            )
            ordinal += 1
    return result


def _ssh_prefix(queue: dict[str, Any], pod_name: str) -> list[str]:
    pod = queue["pods"][pod_name]
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
        "-i", str(Path(queue["ssh"]["key"]).expanduser()), "-p", str(pod["port"]),
        f"root@{pod['host']}",
    ]


def _run_ssh(
    queue: dict[str, Any], pod_name: str, remote: str, *, timeout: int = 30
) -> str:
    try:
        completed = subprocess.run(
            [*_ssh_prefix(queue, pod_name), f"bash -lc {shlex.quote(remote)}"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QueueError(f"{pod_name} SSH failed: {exc}") from exc
    return completed.stdout


def live_snapshot(queue: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Read GPU occupancy and this queue's claims in one SSH per Pod."""

    occupancy: dict[str, int] = {}
    claims: dict[str, dict[str, Any]] = {}
    job_dirs = {job["id"]: job["run_dir"] for job in queue["jobs"]}
    program = f"""import json
from pathlib import Path
import subprocess

jobs = json.loads({json.dumps(json.dumps(job_dirs))})
def lines(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()

compute_rows = lines(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
gpu_rows = lines(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
states = {{}}
for job_id, directory in jobs.items():
    root = Path(directory)
    claim = root / "queue_claim.json"
    if claim.is_file():
        state = "claimed"
        if (root / "run.log.launch").is_file():
            state = "launched"
        if (root / "terminal_result.json").is_file():
            state = "terminal"
        states[job_id] = {{"state": state, "claim_path": str(claim)}}
print(json.dumps({{"compute_rows": compute_rows, "gpu_rows": gpu_rows, "jobs": states}}, sort_keys=True))
"""
    command = f"python3 -c {shlex.quote(program)}"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = dict(
            zip(
                ("pod1", "pod2"),
                pool.map(lambda pod: _run_ssh(queue, pod, command), ("pod1", "pod2")),
            )
        )
    for pod_name in ("pod1", "pod2"):
        try:
            snapshot = json.loads(outputs[pod_name])
        except json.JSONDecodeError as exc:
            raise QueueError(f"{pod_name} returned malformed live snapshot") from exc
        occupancy.update(_parse_gpu_occupancy(pod_name, snapshot))
        for job_id, state in _mapping(snapshot.get("jobs"), f"{pod_name}.jobs").items():
            if job_id in claims:
                raise QueueError(f"job {job_id} is claimed on both Pods")
            claims[job_id] = {"pod": pod_name, **_mapping(state, f"{job_id}.state")}
    expected = {slot.name for slot in slots(queue)}
    if set(occupancy) != expected:
        raise QueueError(f"GPU inventory mismatch: expected={sorted(expected)} got={sorted(occupancy)}")
    return occupancy, claims


def live_occupancy(queue: dict[str, Any]) -> dict[str, int]:
    return live_snapshot(queue)[0]


def _parse_gpu_occupancy(pod_name: str, snapshot: dict[str, Any]) -> dict[str, int]:
    compute_rows = _list(snapshot.get("compute_rows"), f"{pod_name}.compute_rows")
    gpu_rows = _list(snapshot.get("gpu_rows"), f"{pod_name}.gpu_rows")
    pids_by_uuid: dict[str, set[int]] = {}
    for index, row in enumerate(compute_rows):
        row = _text(row, f"{pod_name}.compute_rows[{index}]")
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise QueueError(f"{pod_name} returned malformed compute-app row: {row!r}")
        pids_by_uuid.setdefault(fields[0], set()).add(int(fields[1]))
    result: dict[str, int] = {}
    for index, row in enumerate(gpu_rows):
        row = _text(row, f"{pod_name}.gpu_rows[{index}]")
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
            raise QueueError(f"{pod_name} returned malformed GPU row: {row!r}")
        result[f"{pod_name}/gpu{fields[0]}"] = len(pids_by_uuid.get(fields[1], set()))
    return result


def _assign(
    queue: dict[str, Any], occupancy: dict[str, int], claimed: set[str] | None = None
) -> list[tuple[dict[str, Any], Slot]]:
    claimed = set() if claimed is None else claimed
    current = dict(occupancy)
    assignments: list[tuple[dict[str, Any], Slot]] = []
    all_slots = slots(queue)
    for job in queue["jobs"]:
        if job["status"] != READY or job["id"] in claimed:
            continue
        available = [slot for slot in all_slots if current[slot.name] < slot.capacity]
        if not available:
            break
        chosen = min(available, key=lambda slot: (current[slot.name], slot.ordinal))
        assignments.append((job, chosen))
        current[chosen.name] += 1
    return assignments


def _training_argv(queue: dict[str, Any], job: dict[str, Any], gpu: int) -> list[str]:
    source = job["source"]["checkout"]
    workdir = f"{source.rstrip('/')}/{WBT_RELATIVE}"
    argv = [
        ISAAC_PYTHON, f"{workdir}/{ENTRYPOINT_RELATIVE}",
        *job["recipe"]["base"], *job["recipe"]["delta"],
    ]
    for arg, path in job["motion"]["bindings"].items():
        argv.append(f"{arg}={path}")
    argv.extend(
        [
            f"{job['bank']['train_arg']}={job['bank']['train_path']}",
            f"seed={job['seed']}",
            f"num_envs={job['budget']['num_envs']}",
            f"max_iterations={job['budget']['max_iterations']}",
            f"algo.runner.save_interval={job['budget']['save_interval']}",
            f"run_name={job['run_name']}",
            # CUDA_VISIBLE_DEVICES maps the chosen physical GPU to logical cuda:0.
            "device=cuda:0",
        ]
    )
    return argv


def _launch_script(queue: dict[str, Any], job: dict[str, Any], slot: Slot) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    run_dir = job["run_dir"].rstrip("/")
    required = [
        *job["motion"]["bindings"].values(),
        job["bank"]["train_path"], job["exam"]["path"],
    ]
    checks = "\n".join(f"test -f {shlex.quote(path)}" for path in required)
    claim = json.dumps(
        {
            "schema_version": 1,
            "job_id": job["id"],
            "action": job["action"],
            "exam": job["exam"]["path"],
            "seed": job["seed"],
            "pod": slot.pod,
            "gpu": slot.gpu,
        },
        sort_keys=True,
    ) + "\n"
    train_argv = _training_argv(queue, job, slot.gpu)
    launcher = f"{workdir}/{KIT_LAUNCHER_RELATIVE}"
    launch = [
        launcher, f"{run_dir}/run.log", "env",
        f"CUDA_VISIBLE_DEVICES={slot.gpu}", *train_argv,
    ]
    # The per-GPU flock covers the last capacity check, claim, and spawn.
    body = f"""set -euo pipefail
test \"$(git -C {shlex.quote(source)} rev-parse HEAD)\" = {shlex.quote(job['source']['commit'])}
test -z \"$(git -C {shlex.quote(source)} status --porcelain)\"
{checks}
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(UNIQUE_NUMERIC_PID_AWK)})
test \"$count\" -lt {slot.capacity}
mkdir -p {shlex.quote(run_dir)}
( set -o noclobber; printf %s {shlex.quote(claim)} > {shlex.quote(run_dir + '/queue_claim.json')} )
cd {shlex.quote(workdir)}
source {shlex.quote(workdir + '/' + SETUP_RELATIVE)}
export KIT_BOOT_MARKER={shlex.quote(KIT_BOOT_MARKER)}
export KIT_BOOT_TIMEOUT_S={KIT_BOOT_TIMEOUT_SECONDS}
{shlex.join(launch)}
"""
    return f"flock -n /tmp/hope_lean_queue_gpu{slot.gpu}.lock bash -lc {shlex.quote(body)}"


def cmd_plan(queue: dict[str, Any], *, live: bool) -> dict[str, Any]:
    if live:
        occupancy, claims = live_snapshot(queue)
    else:
        occupancy, claims = {slot.name: 0 for slot in slots(queue)}, {}
    assignments = _assign(queue, occupancy, set(claims))
    return {
        "mode": "plan",
        "dry_run": True,
        "occupancy_source": "live" if live else "assumed_empty",
        "occupancy": occupancy,
        "claims": claims,
        "assignments": [
            {
                "job_id": job["id"], "action": job["action"],
                "resource": slot.name,
                "milestones": job["milestones"],
            }
            for job, slot in assignments
        ],
        "blocked": [
            {"job_id": job["id"], "reason": job["blocker"]}
            for job in queue["jobs"] if job["status"] == BLOCKED
        ],
    }


def cmd_status(queue: dict[str, Any], *, live: bool) -> dict[str, Any]:
    plan = cmd_plan(queue, live=live)
    plan["mode"] = "status"
    plan["jobs"] = [
        {"job_id": job["id"], "action": job["action"], "status": job["status"]}
        for job in queue["jobs"]
    ]
    return plan


def cmd_launch_next(
    queue: dict[str, Any], *, execute: bool, confirm: str | None
) -> dict[str, Any]:
    if execute and confirm != CONFIRM:
        raise QueueError(f"--execute requires --confirm {CONFIRM}")
    if not execute:
        occupancy, claims = {slot.name: 0 for slot in slots(queue)}, {}
        assignments = _assign(queue, occupancy, set(claims))
        if not assignments:
            raise QueueError("no ready job fits an available GPU slot")
        job, slot = assignments[0]
        remote = _launch_script(queue, job, slot)
        return {
            "mode": "launch-next", "dry_run": True,
            "job_id": job["id"], "action": job["action"], "resource": slot.name,
            "ssh_argv": [
                *_ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
            ],
        }

    GLOBAL_SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_SCHEDULER_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Re-sample all six GPUs only after the global scheduler lock is held.
        occupancy, claims = live_snapshot(queue)
        assignments = _assign(queue, occupancy, set(claims))
        if not assignments:
            raise QueueError("no ready job fits an available GPU slot")
        job, slot = assignments[0]
        remote = _launch_script(queue, job, slot)
        output = _run_ssh(
            queue,
            slot.pod,
            remote,
            timeout=KIT_BOOT_TIMEOUT_SECONDS + 60,
        )
        return {
            "mode": "launch-next", "dry_run": False,
            "job_id": job["id"], "action": job["action"], "resource": slot.name,
            "scheduler_lock": str(GLOBAL_SCHEDULER_LOCK),
            "remote_output": output,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "status"):
        command = sub.add_parser(mode)
        command.add_argument("--live", action="store_true", help="read GPU occupancy over SSH")
    launch = sub.add_parser("launch-next")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--confirm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue.resolve())
        if args.mode == "plan":
            result = cmd_plan(queue, live=args.live)
        elif args.mode == "status":
            result = cmd_status(queue, live=args.live)
        else:
            result = cmd_launch_next(queue, execute=args.execute, confirm=args.confirm)
    except QueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
