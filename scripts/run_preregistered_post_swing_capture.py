#!/usr/bin/env python3
"""Fail-closed one-shot launcher for a preregistered post-swing capture.

This program runs *on the simulation host*.  It never opens SSH, sends a
signal, starts a trainer, or retries a spent namespace.  ``plan`` is read-only;
``launch`` performs one Hydra compose before creating the capture directory,
then starts exactly one inference process in a new numeric process group.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


class CaptureContractError(RuntimeError):
    """The frozen plan or current runtime does not authorize a launch."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CaptureContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptureContractError(f"{label} must be an object")
    return value


def _require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CaptureContractError(f"{label} must be an integer >= {minimum}")
    return value


def _normal_key(argument: str) -> str:
    if "=" not in argument:
        raise CaptureContractError(f"non-Hydra argument in frozen recipe: {argument}")
    key = argument.split("=", 1)[0].lstrip("+")
    if not key:
        raise CaptureContractError(f"empty Hydra key in frozen recipe: {argument}")
    return key


def _load_json_document(path: Path, expected_file_sha256: str, label: str) -> Mapping[str, Any]:
    result = path.lstat()
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
        raise CaptureContractError(f"{label} must be a regular non-symlink file: {path}")
    expected = _require_sha256(expected_file_sha256, f"{label}.file_sha256")
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise CaptureContractError(f"{label} file SHA mismatch: {actual} != {expected}")
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(f"{label} is not canonical JSON: {exc}") from exc
    return _require_mapping(result, label)


def _verify_content_document(row: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    path = Path(str(row.get("path", "")))
    document = _load_json_document(path, str(row.get("file_sha256", "")), label)
    content = _require_mapping(document.get("content"), f"{label}.content")
    declared = _require_sha256(row.get("content_sha256"), f"{label}.content_sha256")
    embedded = _require_sha256(document.get("content_sha256"), f"{label}.embedded_content_sha256")
    actual = _sha256_bytes(_canonical_bytes(content))
    if actual != declared or embedded != declared:
        raise CaptureContractError(
            f"{label} content binding mismatch: actual={actual} embedded={embedded} expected={declared}"
        )
    return document


def _inventory(root: Path) -> dict[str, Any]:
    if os.path.lexists(root) is False:
        raise CaptureContractError(f"ignored runtime asset is missing: {root}")
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise CaptureContractError(f"ignored runtime asset is not a real directory: {root}")
    rows: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(dirnames):
            path = current_path / name
            result = path.lstat()
            if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
                raise CaptureContractError(f"asset tree contains invalid directory entry: {path}")
        for name in sorted(filenames):
            path = current_path / name
            result = path.lstat()
            if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
                raise CaptureContractError(f"asset tree contains invalid file entry: {path}")
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": result.st_size,
                    "sha256": _sha256_file(path),
                }
            )
    rows.sort(key=lambda row: row["relative_path"])
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": _sha256_bytes(_canonical_bytes({"files": rows})),
    }


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *arguments], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise CaptureContractError(f"git check failed for {checkout}: {exc.output}") from exc


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != 2:
        raise CaptureContractError("only post-swing capture plan schema_version=2 is supported")
    if plan.get("status") != "preregistered_capture_not_started":
        raise CaptureContractError("plan is not in preregistered_capture_not_started state")
    if plan.get("simulation_only") is not True:
        raise CaptureContractError("capture must be simulation_only")
    contract = _require_mapping(plan.get("capture_contract"), "capture_contract")
    if contract.get("pod") != "pod2" or contract.get("gpu") != 1:
        raise CaptureContractError("this plan must remain Pod2 GPU1 only")
    if contract.get("cuda_visible_devices") != "1" or contract.get("runtime_device") != "cuda:0":
        raise CaptureContractError("CUDA remapping contract changed")
    for key in ("num_envs", "target_count", "max_inference_steps", "seed"):
        _require_plain_int(contract.get(key), f"capture_contract.{key}", minimum=1 if key != "seed" else 0)
    if contract.get("num_envs") != 4096 or contract.get("target_count") != 4096:
        raise CaptureContractError("first formal capture must remain 4096 environments/states")
    if contract.get("max_inference_steps") != 20000:
        raise CaptureContractError("first formal capture must retain the 20000-step ceiling")
    if type(contract.get("post_swing_start_prob")) not in (int, float) or not math.isclose(
        float(contract["post_swing_start_prob"]), 0.25, rel_tol=0.0, abs_tol=0.0
    ):
        raise CaptureContractError("post_swing_start_prob must remain exactly 0.25")
    limits = (
        ("root_linear_velocity_limit_mps", 2.0),
        ("root_angular_velocity_limit_radps", 4.0),
    )
    for key, expected in limits:
        value = contract.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) != expected:
            raise CaptureContractError(f"capture_contract.{key} must remain {expected}")
    if contract.get("capture_is_inference_only") is not True or contract.get("ppo_updates") != 0:
        raise CaptureContractError("capture may not perform PPO updates")
    if contract.get("natural_wrap_only") is not True or contract.get("wrap_teleport") is not False:
        raise CaptureContractError("capture must remain natural-wrap only")
    if contract.get("output_must_be_absent_before_one_shot") is not True:
        raise CaptureContractError("capture output must be no-clobber")
    namespace_root = Path("/workspace/codexschema/phase1_post_swing_teacher_20260715")
    output = Path(str(contract.get("output_directory", "")))
    launch_root = Path(str(contract.get("launch_root", "")))
    if not output.is_absolute() or not launch_root.is_absolute() or output == launch_root:
        raise CaptureContractError("launch_root/output_directory must be distinct absolute paths")
    for label, path in (("output_directory", output), ("launch_root", launch_root)):
        try:
            path.relative_to(namespace_root)
        except ValueError as exc:
            raise CaptureContractError(f"capture_contract.{label} escapes the frozen artifact root") from exc
    failure = _require_mapping(plan.get("failure_policy"), "failure_policy")
    for key in ("same_namespace_retry_forbidden", "automatic_retry_forbidden", "pod1_and_pod2_gpu0_forbidden"):
        if failure.get(key) is not True:
            raise CaptureContractError(f"failure_policy.{key} must remain true")
    authorization = _require_mapping(plan.get("authorization"), "authorization")
    if authorization.get("capture_authorized") is not True:
        raise CaptureContractError("capture is not authorized")
    for key in ("attestation_authorized_only_after_complete_capture",):
        if authorization.get(key) is not True:
            raise CaptureContractError(f"authorization.{key} must remain true")
    for key in ("first_reset_probe_authorized", "scientific_training_authorized", "second_seed_authorized", "judge_authorized", "hardware_authorized"):
        if authorization.get(key) is not False:
            raise CaptureContractError(f"authorization.{key} must remain false")
    derivation = _require_mapping(plan.get("runtime_recipe_derivation"), "runtime_recipe_derivation")
    required_removed = {
        "logger", "video", "checkpoint_path", "checkpoint_tolerant",
        "checkpoint_allow_missing_contract", "checkpoint_allow_contract_mismatch",
        "max_iterations", "algo.runner.save_interval", "run_name",
        "training_queue_claim_path", "training_run_binding_path", "training_launch_claim_sha256",
    }
    removed = set(derivation.get("remove_keys", []))
    if not required_removed <= removed:
        raise CaptureContractError(
            f"runtime derivation retains train-only keys: {sorted(required_removed - removed)}"
        )
    if derivation.get("seed_must_be_applied_by_play") is not True:
        raise CaptureContractError("play seed parity is not bound")


def _derive_argv(plan: Mapping[str, Any], binding: Mapping[str, Any]) -> list[str]:
    _validate_plan(plan)
    content = _require_mapping(binding.get("content"), "run_binding.content")
    base = content.get("training_argv")
    if not isinstance(base, list) or len(base) < 3 or not all(type(value) is str for value in base):
        raise CaptureContractError("run binding lacks a string training_argv")
    derivation = _require_mapping(plan["runtime_recipe_derivation"], "runtime_recipe_derivation")
    removed = set(derivation["remove_keys"])
    seen: dict[str, str] = {}
    retained: list[str] = []
    for argument in base[2:]:
        key = _normal_key(argument)
        value = argument.split("=", 1)[1]
        if key in removed:
            continue
        if key in seen:
            if seen[key] != value:
                raise CaptureContractError(f"conflicting duplicate Hydra key: {key}")
            continue
        seen[key] = value
        retained.append(argument)
    teacher = _require_mapping(plan["teacher_checkpoint"], "teacher_checkpoint")
    contract = _require_mapping(plan["capture_contract"], "capture_contract")
    motions = plan.get("ordered_motion_inputs")
    if not isinstance(motions, list) or len(motions) != 2:
        raise CaptureContractError("exactly two ordered motion inputs are required")
    bank = _require_mapping(plan["question_bank"], "question_bank")
    required = {
        "task": "HOPEPingPongVirtualBall",
        "algo": "ppo",
        "headless": "true",
        "device": str(contract["runtime_device"]),
        "num_envs": str(contract["num_envs"]),
        "seed": str(contract["seed"]),
        "task.motion.wrap_teleport": "false",
        "task.motion.post_swing_start_prob": str(contract["post_swing_start_prob"]),
        "motion_file": str(motions[0]["path"]),
        "motion_file_2": str(motions[1]["path"]),
        "task.racket.question_bank": str(bank["path"]),
    }
    for key, expected in required.items():
        if seen.get(key) != expected:
            raise CaptureContractError(
                f"training recipe mismatch for {key}: {seen.get(key)!r} != {expected!r}"
            )
    output = Path(str(contract["output_directory"]))
    additions = [
        f"checkpoint={teacher['path']}",
        f"+task.motion.post_swing_capture_output_dir={output}",
        f"+task.motion.post_swing_capture_target_count={contract['target_count']}",
        f"post_swing_capture_max_steps={contract['max_inference_steps']}",
    ]
    for argument in additions:
        key = _normal_key(argument)
        if key in seen:
            raise CaptureContractError(f"capture addition already exists in training recipe: {key}")
        seen[key] = argument.split("=", 1)[1]
    source = _require_mapping(plan["capture_source"], "capture_source")
    argv = [
        base[0],
        str(Path(str(source["checkout"])) / "hope_training/whole_body_tracking/scripts/play.py"),
        *retained,
        *additions,
    ]
    keys = [_normal_key(argument) for argument in argv[2:]]
    if len(keys) != len(set(keys)):
        raise CaptureContractError("derived capture argv contains duplicate Hydra keys")
    return argv


def _gpu_apps() -> list[dict[str, int]]:
    try:
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        app_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CaptureContractError(f"cannot inventory GPUs: {exc}") from exc
    indices = {
        row.split(",", 1)[1].strip(): int(row.split(",", 1)[0])
        for row in gpu_rows if row.strip()
    }
    result = []
    for row in app_rows:
        if not row.strip():
            continue
        uuid, pid = (value.strip() for value in row.split(",", 1))
        if uuid not in indices:
            raise CaptureContractError(f"compute app reports unknown GPU UUID: {uuid}")
        result.append({"gpu": indices[uuid], "pid": int(pid)})
    return result


def _verify_runtime(plan: Mapping[str, Any], current_script: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    _validate_plan(plan)
    source = _require_mapping(plan["capture_source"], "capture_source")
    checkout = Path(str(source["checkout"]))
    if _git_output(checkout, "rev-parse", "HEAD") != source.get("commit"):
        raise CaptureContractError("capture source commit mismatch")
    if _git_output(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise CaptureContractError("capture source has tracked changes")
    files = _require_mapping(source.get("files"), "capture_source.files")
    launcher_matched = False
    for label, raw_row in files.items():
        row = _require_mapping(raw_row, f"capture_source.files.{label}")
        path = checkout / str(row.get("path", ""))
        path_stat = path.lstat()
        if (
            stat.S_ISLNK(path_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_size != row.get("bytes")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise CaptureContractError(f"capture source file mismatch: {path}")
        if path.resolve() == current_script.resolve():
            launcher_matched = True
    if not launcher_matched:
        raise CaptureContractError("running launcher is not bound in capture_source.files")
    asset = _require_mapping(source.get("ignored_runtime_asset"), "ignored_runtime_asset")
    proof = _inventory(checkout / str(asset["relative_path"]))
    expected = {key: asset[key] for key in ("file_count", "total_file_bytes", "tree_content_sha256")}
    if proof != expected:
        raise CaptureContractError(f"ignored runtime asset mismatch: {proof} != {expected}")
    teacher = _require_mapping(plan["teacher_checkpoint"], "teacher_checkpoint")
    direct_rows = [
        ("checkpoint", teacher, "sha256"),
        ("hard_contract", _require_mapping(teacher["hard_contract"], "hard_contract"), "sha256"),
        ("question_bank", _require_mapping(plan["question_bank"], "question_bank"), "sha256"),
    ]
    motions = plan.get("ordered_motion_inputs")
    if not isinstance(motions, list):
        raise CaptureContractError("ordered_motion_inputs must be a list")
    direct_rows.extend((f"motion_{index}", _require_mapping(row, "motion"), "sha256") for index, row in enumerate(motions))
    verified: list[dict[str, str]] = []
    for label, row, sha_key in direct_rows:
        path = Path(str(row["path"])); expected_sha = _require_sha256(row[sha_key], f"{label}.sha256")
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise CaptureContractError(f"{label} must be a regular non-symlink file: {path}")
        actual = _sha256_file(path)
        if actual != expected_sha:
            raise CaptureContractError(f"{label} SHA mismatch: {actual} != {expected_sha}")
        verified.append({"label": label, "path": str(path), "sha256": actual})
    claim = _verify_content_document(_require_mapping(teacher["launch_claim"], "launch_claim"), "launch_claim")
    binding = _verify_content_document(_require_mapping(teacher["run_binding"], "run_binding"), "run_binding")
    _verify_content_document(_require_mapping(teacher["milestone_receipt"], "milestone_receipt"), "milestone_receipt")
    del claim
    return binding, {"source_commit": source["commit"], "asset_inventory": proof, "verified_inputs": verified}


def _exclusive_write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CaptureContractError(f"cannot write no-clobber artifact: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_real_parents(path: Path) -> None:
    """Create parents and reject any symlinked component before an O_EXCL leaf."""
    path.mkdir(parents=True, exist_ok=True)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        result = current.lstat()
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
            raise CaptureContractError(f"artifact parent is not a real directory: {current}")


def _environment(plan: Mapping[str, Any]) -> dict[str, str]:
    source = Path(str(plan["capture_source"]["checkout"]))
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(plan["capture_contract"]["cuda_visible_devices"]),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ":".join(
                [
                    str(source / "hope_training/whole_body_tracking/source/whole_body_tracking"),
                    "/workspace/IsaacLab/source/isaaclab",
                    "/workspace/IsaacLab/source/isaaclab_tasks",
                    "/workspace/IsaacLab/source/isaaclab_assets",
                    "/workspace/IsaacLab/source/isaaclab_rl",
                ]
            ),
        }
    )
    return environment


def _load_plan(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], bytes]:
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    expected = _require_sha256(expected_sha256, "expected plan SHA-256")
    if actual != expected:
        raise CaptureContractError(f"plan SHA mismatch: {actual} != {expected}")
    try:
        plan = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(f"plan is not JSON: {exc}") from exc
    plan = _require_mapping(plan, "plan")
    _validate_plan(plan)
    return plan, raw


def _plan_summary(plan: Mapping[str, Any], plan_raw: bytes, script: Path) -> dict[str, Any]:
    binding, proof = _verify_runtime(plan, script)
    argv = _derive_argv(plan, binding)
    launch_root = Path(str(plan["capture_contract"]["launch_root"]))
    output = Path(str(plan["capture_contract"]["output_directory"]))
    return {
        "plan_sha256": _sha256_bytes(plan_raw),
        "source_commit": proof["source_commit"],
        "argv_sha256": _sha256_bytes(_canonical_bytes(argv)),
        "launch_root_lexists": os.path.lexists(launch_root),
        "capture_output_lexists": os.path.lexists(output),
        "gpu_apps": _gpu_apps(),
        "asset_inventory": proof["asset_inventory"],
    }


def _launch(plan: Mapping[str, Any], plan_raw: bytes, script: Path) -> dict[str, Any]:
    binding, runtime_proof = _verify_runtime(plan, script)
    argv = _derive_argv(plan, binding)
    contract = plan["capture_contract"]
    launch_root = Path(str(contract["launch_root"]))
    output = Path(str(contract["output_directory"]))
    if os.path.lexists(launch_root) or os.path.lexists(output):
        raise CaptureContractError("launch or capture namespace is already spent")
    apps = _gpu_apps()
    if any(row["gpu"] == int(contract["gpu"]) for row in apps):
        raise CaptureContractError(f"requested GPU is occupied: {apps}")
    _mkdir_real_parents(launch_root.parent)
    os.mkdir(launch_root, 0o700)
    _exclusive_write(launch_root / "prereg.json", plan_raw)
    _exclusive_write(launch_root / "runtime_argv.json", _canonical_bytes({"argv": argv}) + b"\n")
    argv_sha = _sha256_bytes(_canonical_bytes(argv))
    prelaunch = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256": _sha256_bytes(plan_raw),
        "argv_sha256": argv_sha,
        "gpu_apps_before": apps,
        **runtime_proof,
    }
    _exclusive_write(launch_root / "prelaunch_receipt.json", _canonical_bytes(prelaunch) + b"\n")
    source = Path(str(plan["capture_source"]["checkout"]))
    cwd = source / "hope_training/whole_body_tracking"
    environment = _environment(plan)
    compose = subprocess.run(
        [*argv, "--cfg", "job"],
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _exclusive_write(launch_root / "hydra_compose.log", compose.stdout)
    if compose.returncode != 0:
        failure = {"schema_version": 1, "stage": "hydra_compose", "returncode": compose.returncode}
        _exclusive_write(launch_root / "failure.json", _canonical_bytes(failure) + b"\n")
        raise CaptureContractError(f"Hydra compose failed with rc={compose.returncode}; plan is spent")
    binding_after, runtime_after = _verify_runtime(plan, script)
    if _derive_argv(plan, binding_after) != argv or runtime_after != runtime_proof:
        failure = {"schema_version": 1, "stage": "post_compose_runtime_drift"}
        _exclusive_write(launch_root / "failure.json", _canonical_bytes(failure) + b"\n")
        raise CaptureContractError("source/input/runtime drifted during compose; plan is spent")
    apps_after = _gpu_apps()
    if any(row["gpu"] == int(contract["gpu"]) for row in apps_after):
        failure = {"schema_version": 1, "stage": "gpu_recheck", "gpu_apps": apps_after}
        _exclusive_write(launch_root / "failure.json", _canonical_bytes(failure) + b"\n")
        raise CaptureContractError("requested GPU became occupied after compose; plan is spent")
    if os.path.lexists(output):
        failure = {"schema_version": 1, "stage": "capture_namespace_appeared"}
        _exclusive_write(launch_root / "failure.json", _canonical_bytes(failure) + b"\n")
        raise CaptureContractError("capture namespace appeared during compose; plan is spent")
    _mkdir_real_parents(output.parent)
    os.mkdir(output, 0o700)
    log_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    log_descriptor = os.open(launch_root / "run.log", log_flags, 0o600)
    try:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_descriptor,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            failure = {
                "schema_version": 1,
                "stage": "process_start",
                "error_type": type(exc).__name__,
                "errno": exc.errno,
            }
            _exclusive_write(launch_root / "failure.json", _canonical_bytes(failure) + b"\n")
            raise CaptureContractError(f"cannot start capture process; plan is spent: {exc}") from exc
    finally:
        os.close(log_descriptor)
    time.sleep(2.0)
    returncode = process.poll()
    process_group = None
    starttime_ticks = None
    try:
        process_group = os.getpgid(process.pid)
        starttime_ticks = int(Path(f"/proc/{process.pid}/stat").read_text().split()[21])
    except (ProcessLookupError, FileNotFoundError):
        pass
    receipt = {
        "schema_version": 1,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": process.pid,
        "pgid": process_group,
        "leader_starttime_ticks": starttime_ticks,
        "returncode_after_2s": returncode,
        "plan_sha256": _sha256_bytes(plan_raw),
        "argv_sha256": argv_sha,
        "source_commit": runtime_proof["source_commit"],
        "capture_output": str(output),
        "run_log": str(launch_root / "run.log"),
    }
    _exclusive_write(launch_root / "launch.json", _canonical_bytes(receipt) + b"\n")
    if returncode is not None or process_group != process.pid or starttime_ticks is None:
        failure = {
            "schema_version": 1,
            "stage": "two_second_identity_gate",
            "pid": process.pid,
            "pgid": process_group,
            "leader_starttime_ticks": starttime_ticks,
            "returncode": returncode,
        }
        _exclusive_write(launch_root / "failure.json", _canonical_bytes(failure) + b"\n")
        raise CaptureContractError("capture failed the two-second process identity gate; plan is spent")
    return receipt


def _status(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan["capture_contract"]
    launch_root = Path(str(contract["launch_root"]))
    output = Path(str(contract["output_directory"]))
    receipt_path = launch_root / "launch.json"
    receipt = None
    alive = False
    identity_exact = False
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        pid = receipt.get("pid")
        stat_path = Path(f"/proc/{pid}/stat") if type(pid) is int else None
        if stat_path is not None and stat_path.is_file():
            fields = stat_path.read_text().split()
            alive = True
            identity_exact = (
                int(fields[4]) == receipt.get("pgid")
                and int(fields[21]) == receipt.get("leader_starttime_ticks")
            )
    fixed = {}
    for name in (
        "natural_wrap_capture.claim.json",
        "natural_wrap_states.npz",
        "natural_wrap_capture.json",
        "teacher_receipt.json",
    ):
        path = output / name
        fixed[name] = {
            "lexists": os.path.lexists(path),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
    return {
        "launch_root_lexists": os.path.lexists(launch_root),
        "capture_output_lexists": os.path.lexists(output),
        "launch_receipt": receipt,
        "leader_alive": alive,
        "leader_identity_exact": identity_exact,
        "artifacts": fixed,
        "gpu_apps": _gpu_apps(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("mode", choices=("plan", "launch", "status"))
    args = parser.parse_args(argv)
    try:
        plan, raw = _load_plan(args.plan, args.expected_plan_sha256)
        if args.mode == "plan":
            result = _plan_summary(plan, raw, Path(__file__))
        elif args.mode == "launch":
            result = _launch(plan, raw, Path(__file__))
        else:
            result = _status(plan)
    except (CaptureContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
