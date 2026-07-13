#!/usr/bin/env python3
"""One-shot B/C schema-2/FK consume runner.

``preflight`` is read-only. ``run`` serializes B/C under one flock, publishes an irreversible
per-asset claim before starting the exact historical materializer, captures the child result, and
publishes either a permanent failure ledger or a success ledger last. ``validate-result`` accepts
only an output with the complete claim -> runner -> activation -> receipt lineage. Direct invocation
of the historical materializer can therefore create bytes, but those bytes are never an acceptable
result.

This runner never starts a simulator, trainer, judge, deployment process, or hardware command.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import validate_motion_schema2_fk_consume_activation as gate


class OneShotRunnerError(RuntimeError):
    """The one-shot contract is incomplete, drifted, already spent, or failed."""


@dataclass(frozen=True)
class AttemptPaths:
    control_root: Path
    lock: Path
    claim: Path
    failure: Path
    success: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": gate.sha256_file(path)}


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink_components(path: Path, label: str) -> None:
    probe = path
    while probe != probe.parent:
        if _lexists(probe) and probe.is_symlink():
            raise OneShotRunnerError(f"{label} contains symlink component: {probe}")
        probe = probe.parent


def _ensure_real_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        info = path.stat()
    except OSError as exc:
        raise OneShotRunnerError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise OneShotRunnerError(f"{label} is not a real directory: {path}")


def _ensure_regular_file(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        info = path.stat()
    except OSError as exc:
        raise OneShotRunnerError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise OneShotRunnerError(f"{label} is not a regular file: {path}")


def _mkdir_control_root(path: Path) -> None:
    _reject_symlink_components(path.parent, "control parent")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _ensure_real_directory(path, "control root")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o600) -> dict[str, Any]:
    """Publish complete bytes with a hard-link no-replace commit."""

    _ensure_real_directory(path.parent, "ledger parent")
    if _lexists(path):
        raise OneShotRunnerError(f"no-replace path already exists: {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    linked = False
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise OneShotRunnerError(f"no-replace path raced into existence: {path}") from exc
        linked = True
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(path.parent)
    _ensure_regular_file(path, "published ledger")
    return binding(path)


class ExclusiveFlock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self):
        _ensure_real_directory(self.path.parent, "lock parent")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise OneShotRunnerError(f"shared lock is not a regular file: {self.path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except Exception:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


def _captured_record(run: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    return {
        "returncode": run.returncode,
        "stdout": run.stdout.decode("utf-8", errors="replace"),
        "stdout_bytes": len(run.stdout),
        "stdout_sha256": hashlib.sha256(run.stdout).hexdigest(),
        "stderr": run.stderr.decode("utf-8", errors="replace"),
        "stderr_bytes": len(run.stderr),
        "stderr_sha256": hashlib.sha256(run.stderr).hexdigest(),
    }


def _run_captured(
    argv: Sequence[str], *, cwd: Path, environment: Mapping[str, str], start_new_session: bool
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    run = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        start_new_session=start_new_session,
    )
    return run, _captured_record(run)


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(source), *args],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_detached_clean_checkout(source: Path, expected_commit: str) -> dict[str, Any]:
    _ensure_real_directory(source, "source checkout")
    head = _git(source, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.decode().strip() != expected_commit:
        raise OneShotRunnerError("source checkout HEAD changed")
    symbolic = _git(source, "symbolic-ref", "--quiet", "HEAD")
    if symbolic.returncode == 0:
        raise OneShotRunnerError("source checkout is attached, not detached")
    if symbolic.returncode not in (1,):
        raise OneShotRunnerError("cannot prove detached source checkout")
    status_run = _git(source, "status", "--porcelain=v1", "--untracked-files=normal")
    if status_run.returncode != 0 or status_run.stdout:
        raise OneShotRunnerError("source checkout is not clean")
    return {"path": str(source), "commit": expected_commit, "detached": True, "clean": True}


def _verify_binding_at_root(value: Mapping[str, Any], root: Path, label: str) -> dict[str, Any]:
    raw = value.get("path")
    if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise OneShotRunnerError(f"{label} path is not safe repo-relative")
    path = root / raw
    _ensure_regular_file(path, label)
    actual = binding(path)
    if actual["bytes"] != value.get("bytes") or actual["sha256"] != value.get("sha256"):
        raise OneShotRunnerError(f"{label} binding changed")
    return actual


def _verify_absolute_binding(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(value.get("path", "")))
    if not path.is_absolute():
        raise OneShotRunnerError(f"{label} path is not absolute")
    _ensure_regular_file(path, label)
    actual = binding(path)
    if actual["bytes"] != value.get("bytes") or actual["sha256"] != value.get("sha256"):
        raise OneShotRunnerError(f"{label} binding changed")
    return actual


RUNTIME_PROBE = r"""
import importlib.metadata as metadata
import json
import os
import platform
import sys
import numpy
import onnxruntime
import mujoco

def origin(module):
    return os.path.realpath(module.__file__)

print(json.dumps({
    "sys_executable": sys.executable,
    "resolved_executable": os.path.realpath(sys.executable),
    "python_version": platform.python_version(),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "packages": {
        "numpy": metadata.version("numpy"),
        "onnxruntime": metadata.version("onnxruntime"),
        "mujoco": metadata.version("mujoco"),
    },
    "module_origins": {
        "numpy": origin(numpy),
        "onnxruntime": origin(onnxruntime),
        "mujoco": origin(mujoco),
    },
}, sort_keys=True))
"""


def runtime_environment(activation: Mapping[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    for key, value in activation["runtime"]["environment_overrides"].items():
        environment[key] = value
    return environment


def validate_runtime_probe(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    wanted = {
        "sys_executable": expected["executable"],
        "resolved_executable": expected["resolved_executable"],
        "python_version": expected["python_version"],
        "prefix": expected["prefix"],
        "base_prefix": expected["base_prefix"],
        "packages": expected["packages"],
        "module_origins": expected["module_origins"],
    }
    if observed != wanted:
        raise OneShotRunnerError("runtime interpreter/package/module origin drift")


def _probe_runtime(activation: Mapping[str, Any]) -> dict[str, Any]:
    expected = activation["runtime"]
    executable = Path(expected["executable"])
    resolved = executable.resolve(strict=True)
    _ensure_regular_file(resolved, "resolved runtime executable")
    actual_executable = binding(resolved)
    if (
        str(resolved) != expected["resolved_executable"]
        or actual_executable["bytes"] != expected["executable_bytes"]
        or actual_executable["sha256"] != expected["executable_sha256"]
    ):
        raise OneShotRunnerError("runtime executable binding changed")
    run, record = _run_captured(
        [str(executable), "-I", "-c", RUNTIME_PROBE],
        cwd=Path("/"),
        environment=runtime_environment(activation),
        start_new_session=False,
    )
    if run.returncode != 0:
        raise OneShotRunnerError(f"runtime probe failed rc={run.returncode}: {record['stderr']}")
    try:
        observed = gate.strict_json_bytes(run.stdout, "runtime probe")
    except Exception as exc:
        raise OneShotRunnerError(f"runtime probe output invalid: {exc}") from exc
    validate_runtime_probe(expected, observed)
    return {"identity": observed, "executable": actual_executable, "probe": record}


def _check_attempt_paths_absent(paths: AttemptPaths) -> None:
    for label, path in (("claim", paths.claim), ("failure", paths.failure), ("success", paths.success)):
        if _lexists(path):
            raise OneShotRunnerError(f"{label} path already exists; attempt is spent or state is foreign: {path}")


def _attempt_paths(activation: Mapping[str, Any], asset: str) -> AttemptPaths:
    control = activation["control"]
    row = activation["assets"][asset]
    return AttemptPaths(
        control_root=Path(control["root"]),
        lock=Path(control["shared_flock_path"]),
        claim=Path(row["claim_path"]),
        failure=Path(row["failure_ledger_path"]),
        success=Path(row["success_ledger_path"]),
    )


def _inspect_argv(activation: Mapping[str, Any], asset: str) -> list[str]:
    argv = list(activation["commands"][asset]["child_argv"])
    if not argv or argv[-1] != "consume":
        raise OneShotRunnerError("bound child command is not consume")
    argv[-1] = "inspect"
    return argv


def _revalidate_current_contract_files(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], activation_meta: Mapping[str, Any]
) -> dict[str, Any]:
    activation_path = Path(str(activation_meta["path"]))
    _ensure_regular_file(activation_path, "consume activation")
    activation_actual = binding(activation_path)
    if activation_actual != activation_meta:
        raise OneShotRunnerError("consume activation changed before claim")
    if gate.strict_json_bytes(activation_path.read_bytes(), "consume activation") != activation:
        raise OneShotRunnerError("consume activation bytes no longer match loaded contract")
    receipt_actual = _verify_binding_at_root(
        activation["inspection_receipt"], gate.REPO_ROOT, "inspection receipt"
    )
    receipt_path = Path(receipt_actual["path"])
    if gate.strict_json_bytes(receipt_path.read_bytes(), "inspection receipt") != receipt:
        raise OneShotRunnerError("inspection receipt changed before claim")
    runner_actual = _verify_binding_at_root(activation["runner"], gate.REPO_ROOT, "one-shot runner")
    validator_actual = _verify_binding_at_root(
        activation["source_gate_validator"], gate.REPO_ROOT, "source gate validator"
    )
    return {
        "activation": activation_actual,
        "inspection_receipt": receipt_actual,
        "runner": runner_actual,
        "source_gate_validator": validator_actual,
    }


def runtime_preflight(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], asset: str,
    activation_meta: Mapping[str, Any]
) -> dict[str, Any]:
    source = Path(activation["source_checkout"]["path"])
    checkout = validate_detached_clean_checkout(source, activation["source_checkout"]["commit"])
    tracked = receipt["inspection_checkout"]["tracked_files"]
    tracked_evidence = {
        name: _verify_binding_at_root(value, source, f"source {name}")
        for name, value in tracked.items()
    }
    row = activation["assets"][asset]
    private_evidence = {
        "source_motion": _verify_absolute_binding(row["source_motion"], f"{asset} source motion"),
        "source_materialization_report": _verify_absolute_binding(
            row["source_materialization_report"], f"{asset} source report"
        ),
        "donor_onnx": _verify_absolute_binding(activation["donor_onnx"], "donor ONNX"),
    }
    output_root = Path(row["output_root"])
    if _lexists(output_root):
        raise OneShotRunnerError(f"output root already exists before claim: {output_root}")
    runtime = _probe_runtime(activation)
    environment = runtime_environment(activation)
    inspect_run, inspect_record = _run_captured(
        _inspect_argv(activation, asset),
        cwd=source,
        environment=environment,
        start_new_session=True,
    )
    expected_stdout = activation["commands"][asset]["expected_inspect_stdout"] + "\n"
    if inspect_run.returncode != 0 or inspect_record["stdout"] != expected_stdout or inspect_record["stderr"]:
        raise OneShotRunnerError(
            f"runtime inspect drift rc={inspect_run.returncode} stdout={inspect_record['stdout']!r} "
            f"stderr={inspect_record['stderr']!r}"
        )
    if _lexists(output_root):
        raise OneShotRunnerError("inspect wrote or raced with the output root")
    checkout_after = validate_detached_clean_checkout(source, activation["source_checkout"]["commit"])
    current_contract_files = _revalidate_current_contract_files(
        activation, receipt, activation_meta
    )
    return {
        "asset_id": asset,
        "checkout_before": checkout,
        "checkout_after": checkout_after,
        "tracked_files": tracked_evidence,
        "private_files": private_evidence,
        "runtime": runtime,
        "inspect": inspect_record,
        "current_contract_files": current_contract_files,
        "output_root_absent": True,
        "dynamics_steps": 0,
        "writes": 0,
    }


def _read_strict_file(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _ensure_regular_file(path, label)
    data = path.read_bytes()
    return gate.strict_json_bytes(data, label), {
        "path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
    }


def _validate_utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OneShotRunnerError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OneShotRunnerError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise OneShotRunnerError(f"{label} is not UTC")


def _validate_capture(value: Any, label: str) -> Mapping[str, Any]:
    record = gate.exact_keys(
        value,
        {
            "returncode", "stdout", "stdout_bytes", "stdout_sha256",
            "stderr", "stderr_bytes", "stderr_sha256",
        },
        label,
    )
    if isinstance(record["returncode"], bool) or not isinstance(record["returncode"], int):
        raise OneShotRunnerError(f"{label}.returncode must be an integer")
    for stream in ("stdout", "stderr"):
        text = record[stream]
        size = record[f"{stream}_bytes"]
        digest = record[f"{stream}_sha256"]
        if not isinstance(text, str):
            raise OneShotRunnerError(f"{label}.{stream} must be text")
        payload = text.encode("utf-8")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size != len(payload)
            or digest != hashlib.sha256(payload).hexdigest()
        ):
            raise OneShotRunnerError(f"{label}.{stream} capture binding changed")
    return record


EXPECTED_OUTPUT_FRAMES = {
    "franco_backhand_loop_b": 151,
    "franco_backhand_loop_c": 163,
}
BODY_ORDER_RELATIVE_PATH = "configs/a3_runtime_body_order.txt"
BODY_ORDER_BYTES = 629
BODY_ORDER_SHA256 = "1cdae4ba7c8d604428ee69ed4a3059e67fb195b22e1d0e294d509c4325809a3a"
NPZ_FIELDS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "kinematics_schema_version",
    "body_pos_point",
    "body_lin_vel_point",
    "body_names",
}


def _expected_body_names(activation: Mapping[str, Any]) -> tuple[str, ...]:
    path = Path(activation["source_checkout"]["path"]) / BODY_ORDER_RELATIVE_PATH
    _ensure_regular_file(path, "bound runtime body order")
    actual = binding(path)
    if actual["bytes"] != BODY_ORDER_BYTES or actual["sha256"] != BODY_ORDER_SHA256:
        raise OneShotRunnerError("runtime body-order binding changed")
    try:
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except UnicodeError as exc:
        raise OneShotRunnerError("runtime body-order file is not UTF-8") from exc
    if len(names) != 32 or len(set(names)) != 32:
        raise OneShotRunnerError("runtime body order is not a 32-name bijection")
    return names


def _scalar_text(value: Any, label: str) -> str:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exact Pod runtime is preregistered
        raise OneShotRunnerError("NumPy is required to validate schema-2 lineage") from exc
    raw = np.asarray(value)
    if raw.size != 1 or raw.dtype.hasobject:
        raise OneShotRunnerError(f"{label} must be one non-object scalar")
    item = raw.reshape(-1)[0]
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeError as exc:
            raise OneShotRunnerError(f"{label} is not UTF-8") from exc
    return str(item)


def _validate_schema2_npz(
    activation: Mapping[str, Any], asset: str, path: Path
) -> dict[str, Any]:
    """Validate the NPZ contents, not merely the report's file hash."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exact Pod runtime is preregistered
        raise OneShotRunnerError("NumPy is required to validate schema-2 lineage") from exc

    _ensure_regular_file(path, "schema2 NPZ")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise OneShotRunnerError(f"schema2 NPZ is not a valid ZIP archive: {exc}") from exc
    expected_members = {f"{name}.npy" for name in NPZ_FIELDS}
    if len(members) != len(set(members)) or set(members) != expected_members:
        raise OneShotRunnerError("schema2 NPZ members are missing, duplicated, or unexpected")

    frames = EXPECTED_OUTPUT_FRAMES[asset]
    expected_shapes = {
        "joint_pos": (frames, 31),
        "joint_vel": (frames, 31),
        "body_pos_w": (frames, 32, 3),
        "body_quat_w": (frames, 32, 4),
        "body_lin_vel_w": (frames, 32, 3),
        "body_ang_vel_w": (frames, 32, 3),
    }
    expected_names = _expected_body_names(activation)
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != NPZ_FIELDS or len(data.files) != len(NPZ_FIELDS):
                raise OneShotRunnerError("schema2 NPZ field set changed")
            fps = np.asarray(data["fps"])
            schema = np.asarray(data["kinematics_schema_version"])
            if fps.shape != (1,) or fps.dtype != np.int64 or int(fps[0]) != 50:
                raise OneShotRunnerError("schema2 NPZ fps is not exact int64 [50]")
            if schema.shape != (1,) or schema.dtype != np.int64 or int(schema[0]) != 2:
                raise OneShotRunnerError("schema2 NPZ schema version is not exact int64 [2]")
            if _scalar_text(data["body_pos_point"], "body_pos_point") != "link_origin":
                raise OneShotRunnerError("schema2 body_pos_point is not link_origin")
            if _scalar_text(data["body_lin_vel_point"], "body_lin_vel_point") != "center_of_mass":
                raise OneShotRunnerError("schema2 body_lin_vel_point is not center_of_mass")
            names_raw = np.asarray(data["body_names"])
            if names_raw.shape != (32,) or names_raw.dtype.hasobject:
                raise OneShotRunnerError("schema2 body_names shape/dtype changed")
            names = []
            for item in names_raw.tolist():
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                names.append(str(item))
            if tuple(names) != expected_names:
                raise OneShotRunnerError("schema2 body_names differ from bound runtime order")
            for name, shape in expected_shapes.items():
                array = np.asarray(data[name])
                if array.shape != shape or array.dtype != np.float32:
                    raise OneShotRunnerError(
                        f"schema2 {name} shape/dtype {array.shape}/{array.dtype} != {shape}/float32"
                    )
                if not np.isfinite(array).all():
                    raise OneShotRunnerError(f"schema2 {name} contains NaN/Inf")
            quaternions = np.asarray(data["body_quat_w"], dtype=np.float64)
            if not np.allclose(
                np.linalg.norm(quaternions, axis=-1), 1.0, atol=1.0e-5, rtol=0.0
            ):
                raise OneShotRunnerError("schema2 body_quat_w is not normalized")
    except OneShotRunnerError:
        raise
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as exc:
        raise OneShotRunnerError(f"cannot validate schema2 NPZ contents: {exc}") from exc
    return {
        "field_names": sorted(NPZ_FIELDS),
        "frames": frames,
        "fps": 50,
        "joint_count": 31,
        "body_count": 32,
        "time_series_dtype": "float32",
        "finite": True,
        "kinematics_schema_version": 2,
        "body_pos_point": "link_origin",
        "body_lin_vel_point": "center_of_mass",
        "body_names_sha256": BODY_ORDER_SHA256,
    }


def validate_materialized_output(
    activation: Mapping[str, Any], asset: str
) -> dict[str, Any]:
    row = activation["assets"][asset]
    output_root = Path(row["output_root"])
    _ensure_real_directory(output_root, "schema2 output root")
    children = list(output_root.iterdir())
    if {child.name for child in children} != {
        row["output_motion_filename"], row["report_filename"]
    } or len(children) != 2:
        raise OneShotRunnerError("schema2 output root contains missing or unexpected entries")
    motion_path = output_root / row["output_motion_filename"]
    report_path = output_root / row["report_filename"]
    _ensure_regular_file(motion_path, "schema2 NPZ")
    report, report_binding = _read_strict_file(report_path, "schema2 report")
    motion_binding = binding(motion_path)
    npz = _validate_schema2_npz(activation, asset, motion_path)
    expected_keys = {
        "schema_version", "status", "completed_utc", "scope", "asset_id", "preregistration",
        "shared_runtime", "source_motion", "source_materialization_report", "donor",
        "vendor_mjcf_closure", "output_motion", "structure", "authorization", "next_gate",
    }
    if set(report) != expected_keys:
        raise OneShotRunnerError("schema2 report keys changed")
    _validate_utc_timestamp(report["completed_utc"], "schema2 report completed_utc")
    expected_frames = EXPECTED_OUTPUT_FRAMES[asset]
    expected_input_frames = 91 if asset == "franco_backhand_loop_b" else 98
    expected_closure = {
        "algorithm": "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1",
        "file_count": 75,
        "total_bytes": 14127373,
        "manifest_sha256": "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de",
        "xml_file_count": 1,
        "include_reference_count": 0,
        "external_file_reference_count": 74,
        "unique_external_file_count": 74,
        "mesh_reference_count": 74,
    }
    expected_authorization = {
        "schema2_materialized": True,
        "l0_authorized": True,
        "vendor_l1_authorized": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }
    if (
        report["schema_version"] != 1
        or report["status"] != "complete_exact_schema2_fk_materialization_certificate_blocked"
        or report["scope"] != (
            "exact schema-2 MuJoCo FK materialization only; no L0/L1, table/net, dynamics, "
            "simulator, training, formal-motion or hardware claim"
        )
        or report["asset_id"] != asset
        or report["preregistration"] != {
            "path": activation["commands"][asset]["child_argv"][3],
            "sha256": row["preregistration"]["sha256"],
        }
        or report["source_motion"] != row["source_motion"]
        or report["source_materialization_report"] != row["source_materialization_report"]
        or report["donor"] != {
            "path": activation["donor_onnx"]["path"],
            "bytes": activation["donor_onnx"]["bytes"],
            "sha256": activation["donor_onnx"]["sha256"],
            "required_metadata_subset_exact": True,
        }
        or report["shared_runtime"] != {
            "path": "configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json",
            "bytes": 5503,
            "sha256": "3d32b146e72029960ebf9cb2777f484804dafc87097e9cd3d0513dc277eed6e8",
        }
        or report["vendor_mjcf_closure"] != expected_closure
        or report["output_motion"] != motion_binding
        or report["structure"] != {
            "input_frames": expected_input_frames,
            "input_fps": 30,
            "output_frames": expected_frames,
            "output_fps": 50,
            "hope_frame": "off",
            "kinematics_schema_version": 2,
            "body_pos_point": "link_origin",
            "body_lin_vel_point": "center_of_mass",
            "joint_count": 31,
            "body_count": 32,
            "finite": True,
        }
        or report["authorization"] != expected_authorization
        or report["next_gate"]
        != "independent_L0_static_schema2_audit_then_vendor_L1_self_collision"
    ):
        raise OneShotRunnerError("schema2 report lineage or output binding changed")
    return {
        "output_root": str(output_root),
        "motion": motion_binding,
        "report": report_binding,
        "report_status": report["status"],
        "npz": npz,
        "npz_lineage_bound_by_report_and_content": True,
    }


def _claim_payload(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], asset: str,
    activation_meta: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    attempt_id = hashlib.sha256(
        f"{activation_meta['sha256']}:{asset}:one-shot-v2".encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "attempt_claimed_irreversible_before_child",
        "asset_id": asset,
        "attempt_id": attempt_id,
        "claimed_utc": utc_now(),
        "activation": dict(activation_meta),
        "inspection_receipt": activation["inspection_receipt"],
        "runner": activation["runner"],
        "source_checkout": activation["source_checkout"],
        "output_root": activation["assets"][asset]["output_root"],
        "runtime_preflight_sha256": canonical_sha256(preflight),
        "runtime_preflight": preflight,
        "child_argv": activation["commands"][asset]["child_argv"],
        "attempt_spent": True,
        "automatic_retry_authorized": False,
    }


def _failure_payload(
    activation: Mapping[str, Any], asset: str, claim_binding: Mapping[str, Any],
    claim: Mapping[str, Any], phase: str, error: str, child: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed_attempt_consumed_permanently",
        "asset_id": asset,
        "attempt_id": claim["attempt_id"],
        "failed_utc": utc_now(),
        "activation": claim["activation"],
        "inspection_receipt": activation["inspection_receipt"],
        "runner": activation["runner"],
        "claim": dict(claim_binding),
        "phase": phase,
        "error": error,
        "child": child,
        "attempt_spent": True,
        "automatic_retry_authorized": False,
        "completion_authorized": False,
    }


def _success_payload(
    activation: Mapping[str, Any], asset: str, claim_binding: Mapping[str, Any],
    claim: Mapping[str, Any], child: Mapping[str, Any], output: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete_exact_schema2_fk_consume_runner_v2",
        "asset_id": asset,
        "attempt_id": claim["attempt_id"],
        "completed_utc": utc_now(),
        "activation": claim["activation"],
        "inspection_receipt": activation["inspection_receipt"],
        "runner": activation["runner"],
        "claim": dict(claim_binding),
        "runtime_preflight": claim["runtime_preflight"],
        "child": child,
        "output": dict(output),
        "completion_published_last": True,
        "direct_materializer_output_accepted": False,
        "authorization": {
            "schema2_materialized_with_runner_lineage": True,
            "l0_authorized": False,
            "vendor_l1_authorized": False,
            "table_net_authorized": False,
            "dynamics_authorized": False,
            "simulator_authorized": False,
            "training_authorized": False,
            "formal_motion_authorized": False,
            "hardware_authorized": False,
        },
    }


def _validate_recorded_preflight(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], asset: str,
    activation_meta: Mapping[str, Any], value: Any
) -> Mapping[str, Any]:
    evidence = gate.exact_keys(
        value,
        {
            "asset_id", "checkout_before", "checkout_after", "tracked_files",
            "private_files", "runtime", "inspect", "current_contract_files", "output_root_absent",
            "dynamics_steps", "writes",
        },
        "recorded runtime preflight",
    )
    expected_checkout = {
        "path": activation["source_checkout"]["path"],
        "commit": activation["source_checkout"]["commit"],
        "detached": True,
        "clean": True,
    }
    if (
        evidence["asset_id"] != asset
        or evidence["checkout_before"] != expected_checkout
        or evidence["checkout_after"] != expected_checkout
        or evidence["output_root_absent"] is not True
        or evidence["dynamics_steps"] != 0
        or evidence["writes"] != 0
    ):
        raise OneShotRunnerError("recorded preflight identity/write boundary changed")
    source = Path(activation["source_checkout"]["path"])
    expected_tracked = {
        name: {
            "path": str(source / binding_row["path"]),
            "bytes": binding_row["bytes"],
            "sha256": binding_row["sha256"],
        }
        for name, binding_row in receipt["inspection_checkout"]["tracked_files"].items()
    }
    if evidence["tracked_files"] != expected_tracked:
        raise OneShotRunnerError("recorded preflight tracked-source lineage changed")
    row = activation["assets"][asset]
    expected_private = {
        "source_motion": row["source_motion"],
        "source_materialization_report": row["source_materialization_report"],
        "donor_onnx": {
            key: activation["donor_onnx"][key] for key in ("path", "bytes", "sha256")
        },
    }
    if evidence["private_files"] != expected_private:
        raise OneShotRunnerError("recorded preflight private-input lineage changed")
    def current_binding(contract_binding: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "path": str(gate.REPO_ROOT / str(contract_binding["path"])),
            "bytes": contract_binding["bytes"],
            "sha256": contract_binding["sha256"],
        }

    expected_contract_files = {
        "activation": dict(activation_meta),
        "inspection_receipt": current_binding(activation["inspection_receipt"]),
        "runner": current_binding(activation["runner"]),
        "source_gate_validator": current_binding(activation["source_gate_validator"]),
    }
    if evidence["current_contract_files"] != expected_contract_files:
        raise OneShotRunnerError("recorded activation/receipt/runner revalidation changed")

    runtime = gate.exact_keys(
        evidence["runtime"], {"identity", "executable", "probe"}, "recorded runtime"
    )
    expected_identity = {
        "sys_executable": activation["runtime"]["executable"],
        "resolved_executable": activation["runtime"]["resolved_executable"],
        "python_version": activation["runtime"]["python_version"],
        "prefix": activation["runtime"]["prefix"],
        "base_prefix": activation["runtime"]["base_prefix"],
        "packages": activation["runtime"]["packages"],
        "module_origins": activation["runtime"]["module_origins"],
    }
    if runtime["identity"] != expected_identity or runtime["executable"] != {
        "path": activation["runtime"]["resolved_executable"],
        "bytes": activation["runtime"]["executable_bytes"],
        "sha256": activation["runtime"]["executable_sha256"],
    }:
        raise OneShotRunnerError("recorded runtime identity/executable binding changed")
    probe = _validate_capture(runtime["probe"], "recorded runtime probe")
    if probe["returncode"] != 0 or probe["stderr"] != "":
        raise OneShotRunnerError("recorded runtime probe was not a clean pass")
    try:
        probed_identity = gate.strict_json_bytes(probe["stdout"].encode("utf-8"), "probe stdout")
    except Exception as exc:
        raise OneShotRunnerError(f"recorded runtime probe stdout is invalid: {exc}") from exc
    if probed_identity != expected_identity:
        raise OneShotRunnerError("recorded runtime probe stdout differs from its identity")

    inspect = _validate_capture(evidence["inspect"], "recorded inspect")
    if (
        inspect["returncode"] != 0
        or inspect["stdout"] != activation["commands"][asset]["expected_inspect_stdout"] + "\n"
        or inspect["stderr"] != ""
    ):
        raise OneShotRunnerError("recorded inspect was not the exact no-write pass")
    return evidence


def _validate_claim(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], asset: str,
    activation_meta: Mapping[str, Any], value: Any
) -> Mapping[str, Any]:
    claim = gate.exact_keys(
        value,
        {
            "schema_version", "status", "asset_id", "attempt_id", "claimed_utc",
            "activation", "inspection_receipt", "runner", "source_checkout", "output_root",
            "runtime_preflight_sha256", "runtime_preflight", "child_argv", "attempt_spent",
            "automatic_retry_authorized",
        },
        "attempt claim",
    )
    expected_attempt = hashlib.sha256(
        f"{activation_meta['sha256']}:{asset}:one-shot-v2".encode("utf-8")
    ).hexdigest()
    if (
        claim["schema_version"] != 1
        or claim["status"] != "attempt_claimed_irreversible_before_child"
        or claim["asset_id"] != asset
        or claim["attempt_id"] != expected_attempt
        or claim["activation"] != activation_meta
        or claim["inspection_receipt"] != activation["inspection_receipt"]
        or claim["runner"] != activation["runner"]
        or claim["source_checkout"] != activation["source_checkout"]
        or claim["output_root"] != activation["assets"][asset]["output_root"]
        or claim["child_argv"] != activation["commands"][asset]["child_argv"]
        or claim["attempt_spent"] is not True
        or claim["automatic_retry_authorized"] is not False
    ):
        raise OneShotRunnerError("claim identity/lineage/retry semantics changed")
    _validate_utc_timestamp(claim["claimed_utc"], "claim claimed_utc")
    preflight = _validate_recorded_preflight(
        activation, receipt, asset, activation_meta, claim["runtime_preflight"]
    )
    if claim["runtime_preflight_sha256"] != canonical_sha256(preflight):
        raise OneShotRunnerError("claim runtime-preflight SHA changed")
    return claim


def _validate_success_child(
    activation: Mapping[str, Any], asset: str, value: Any
) -> Mapping[str, Any]:
    child = _validate_capture(value, "success child")
    expected_report = (
        Path(activation["assets"][asset]["output_root"])
        / activation["assets"][asset]["report_filename"]
    )
    if (
        child["returncode"] != 0
        or child["stdout"] != f"[schema2-fk] PASS consume report={expected_report}\n"
        or child["stderr"] != ""
    ):
        raise OneShotRunnerError("success ledger child record is not the exact consume pass")
    return child


def execute_once(
    *,
    paths: AttemptPaths,
    build_preflight: Callable[[], Mapping[str, Any]],
    build_claim: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    run_child: Callable[[], Mapping[str, Any]],
    validate_output: Callable[[], Mapping[str, Any]],
    build_failure: Callable[
        [Mapping[str, Any], Mapping[str, Any], str, str, Mapping[str, Any] | None],
        Mapping[str, Any],
    ],
    build_success: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """Core transaction used by production and synthetic attack tests."""

    _mkdir_control_root(paths.control_root)
    with ExclusiveFlock(paths.lock):
        _check_attempt_paths_absent(paths)
        preflight = dict(build_preflight())
        claim = dict(build_claim(preflight))
        claim_binding = atomic_write_once(paths.claim, json_bytes(claim))
        child: Mapping[str, Any] | None = None
        try:
            child = dict(run_child())
            if child.get("returncode") != 0:
                raise OneShotRunnerError(f"child returned rc={child.get('returncode')}")
            output = dict(validate_output())
            success = dict(build_success(claim_binding, claim, child, output))
            atomic_write_once(paths.success, json_bytes(success))
            return success
        except Exception as exc:
            phase = "child" if child is None or child.get("returncode") != 0 else "post_child_validation"
            failure = dict(build_failure(claim_binding, claim, phase, str(exc), child))
            if not _lexists(paths.failure):
                atomic_write_once(paths.failure, json_bytes(failure))
            raise OneShotRunnerError(
                f"attempt permanently spent for {claim.get('asset_id')}: {exc}"
            ) from exc


def _load_contract(path: Path, expected_sha: str):
    return gate.load_validated_contract(path.resolve(), expected_sha)


def run_asset(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], asset: str,
    activation_meta: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _attempt_paths(activation, asset)

    def preflight():
        return runtime_preflight(activation, receipt, asset, activation_meta)

    def claim(evidence):
        return _claim_payload(activation, receipt, asset, activation_meta, evidence)

    def child():
        source = Path(activation["source_checkout"]["path"])
        _run, record = _run_captured(
            activation["commands"][asset]["child_argv"],
            cwd=source,
            environment=runtime_environment(activation),
            start_new_session=True,
        )
        return record

    def output():
        source = Path(activation["source_checkout"]["path"])
        validate_detached_clean_checkout(source, activation["source_checkout"]["commit"])
        return validate_materialized_output(activation, asset)

    return execute_once(
        paths=paths,
        build_preflight=preflight,
        build_claim=claim,
        run_child=child,
        validate_output=output,
        build_failure=lambda cb, c, p, e, child_record: _failure_payload(
            activation, asset, cb, c, p, e, child_record
        ),
        build_success=lambda cb, c, child_record, output: _success_payload(
            activation, asset, cb, c, child_record, output
        ),
    )


def validate_formal_result(
    activation: Mapping[str, Any], receipt: Mapping[str, Any], asset: str,
    activation_meta: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _attempt_paths(activation, asset)
    if _lexists(paths.failure):
        raise OneShotRunnerError("failure ledger exists; result is permanently rejected")
    if not _lexists(paths.claim):
        raise OneShotRunnerError("missing irreversible claim; direct consume output is forbidden")
    if not _lexists(paths.success):
        raise OneShotRunnerError("missing completion-last success ledger; NPZ lineage is incomplete")
    claim, claim_binding = _read_strict_file(paths.claim, "attempt claim")
    success, success_binding = _read_strict_file(paths.success, "success ledger")
    claim = dict(_validate_claim(activation, receipt, asset, activation_meta, claim))
    expected_success_keys = {
        "schema_version", "status", "asset_id", "attempt_id", "completed_utc", "activation",
        "inspection_receipt", "runner", "claim", "runtime_preflight", "child", "output",
        "completion_published_last", "direct_materializer_output_accepted", "authorization",
    }
    if set(success) != expected_success_keys:
        raise OneShotRunnerError("success ledger keys changed")
    _validate_utc_timestamp(success["completed_utc"], "success completed_utc")
    child = _validate_success_child(activation, asset, success["child"])
    if (
        success["schema_version"] != 1
        or success["status"] != "complete_exact_schema2_fk_consume_runner_v2"
        or success["asset_id"] != asset
        or success["attempt_id"] != claim["attempt_id"]
        or success["activation"] != activation_meta
        or success["inspection_receipt"] != activation["inspection_receipt"]
        or success["runner"] != activation["runner"]
        or success["claim"] != claim_binding
        or success["runtime_preflight"] != claim["runtime_preflight"]
        or success["child"] != child
        or success["completion_published_last"] is not True
        or success["direct_materializer_output_accepted"] is not False
    ):
        raise OneShotRunnerError("success ledger lineage changed")
    expected_authorization = {
        "schema2_materialized_with_runner_lineage": True,
        "l0_authorized": False,
        "vendor_l1_authorized": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }
    if success["authorization"] != expected_authorization:
        raise OneShotRunnerError("success ledger over-authorizes a downstream gate")
    output = validate_materialized_output(activation, asset)
    if success["output"] != output:
        raise OneShotRunnerError("success ledger does not bind the current NPZ/report")
    return {"claim": claim_binding, "success": success_binding, "output": output}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--expected-activation-sha256", required=True)
    parser.add_argument("--asset", choices=gate.ASSET_IDS, required=True)
    parser.add_argument("command", choices=("preflight", "run", "validate-result"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        activation, receipt, activation_meta = _load_contract(
            args.activation, args.expected_activation_sha256
        )
        if args.command == "preflight":
            evidence = runtime_preflight(activation, receipt, args.asset, activation_meta)
            print(
                f"[schema2-fk-once] PASS preflight asset={args.asset} "
                f"runtime_exact=true inspect_exact=true no_write=true evidence={canonical_sha256(evidence)}"
            )
            return 0
        if args.command == "validate-result":
            evidence = validate_formal_result(activation, receipt, args.asset, activation_meta)
            print(
                f"[schema2-fk-once] PASS result asset={args.asset} runner_lineage=true "
                f"npz_bound=true success_sha256={evidence['success']['sha256']}"
            )
            return 0
        result = run_asset(activation, receipt, args.asset, activation_meta)
        print(
            f"[schema2-fk-once] PASS run asset={args.asset} attempt_spent=true "
            f"completion_last=true output_sha256={result['output']['motion']['sha256']}"
        )
        return 0
    except (gate.ActivationContractError, OneShotRunnerError, OSError, TypeError, ValueError) as exc:
        print(f"[schema2-fk-once] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
