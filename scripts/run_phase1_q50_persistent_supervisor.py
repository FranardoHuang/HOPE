#!/usr/bin/env python3
"""Reliably launch or read back one activation-bound model-4000 q50 runner.

The public command surface is deliberately limited to ``launch`` and ``inspect``.
``launch`` uses a two-party, no-clobber file handshake before replacing a detached
child with the already-bound q50 runner.  ``inspect`` never mutates state.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import secrets
import sys
import time
from typing import Any


class SupervisorError(RuntimeError):
    """The launch contract or preserved process identity is invalid."""


IdentityReader = Callable[[int], dict[str, Any] | None]
BootIdReader = Callable[[], str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SupervisorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(where: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SupervisorError(f"{where} must be a lowercase SHA-256")
    return value


def _strict_object_bytes(raw: bytes, where: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SupervisorError(f"cannot parse strict JSON object {where}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"JSON root must be an object: {where}")
    return value


def _strict_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SupervisorError(f"cannot read strict JSON object {path}: {exc}") from exc
    return _strict_object_bytes(raw, str(path))


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SupervisorError(
            f"{where} keys differ: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_no_clobber(
    path: Path,
    value: Mapping[str, Any],
    *,
    before_link_hook: Callable[[], None] | None = None,
) -> None:
    """Publish canonical JSON atomically without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(16)
    temporary = path.parent / f".{path.name}.{nonce}.tmp"
    payload = _canonical_bytes(dict(value)) + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if before_link_hook is not None:
            before_link_hook()
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SupervisorError(f"no-clobber output already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolve_source(config_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SupervisorError("supervisor source path is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = config_path.parent.parent / path
    return path.resolve()


def _absolute_file_binding(where: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SupervisorError(f"{where} must be an object")
    _exact_keys(value, {"path", "sha256"}, where)
    path = Path(str(value["path"]))
    if not path.is_absolute():
        raise SupervisorError(f"{where}.path must be absolute")
    sha = _require_sha(f"{where}.sha256", value["sha256"])
    return {"path": str(path), "sha256": sha}


def _validate_bound_file(
    where: str,
    binding: Mapping[str, Any],
    *,
    validation_path: Path | None = None,
) -> Path:
    bound_path = Path(str(binding["path"]))
    path = bound_path if validation_path is None else validation_path
    if not path.is_file() or _sha256_file(path) != binding["sha256"]:
        raise SupervisorError(f"{where} bytes changed or are missing: {bound_path}")
    return bound_path


def _executable_binding(where: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SupervisorError(f"{where} must be an object")
    _exact_keys(value, {"path", "resolved_path", "sha256"}, where)
    path = Path(str(value["path"]))
    resolved = Path(str(value["resolved_path"]))
    if not path.is_absolute() or not resolved.is_absolute():
        raise SupervisorError(f"{where} paths must be absolute")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": _require_sha(f"{where}.sha256", value["sha256"]),
    }


def _validate_executable_binding(where: str, binding: Mapping[str, Any]) -> Path:
    path = Path(str(binding["path"]))
    expected_resolved = Path(str(binding["resolved_path"]))
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SupervisorError(f"{where} cannot resolve: {path}: {exc}") from exc
    if (
        resolved != expected_resolved
        or not path.is_file()
        or not os.access(path, os.X_OK)
        or _sha256_file(path) != binding["sha256"]
    ):
        raise SupervisorError(f"{where} executable identity changed: {path}")
    return path


def load_supervisor_config(
    config_path: Path,
    expected_sha256: str,
    *,
    validation_path_resolver: Callable[[Path], Path] | None = None,
) -> dict[str, Any]:
    """Load a caller-pinned supervisor config and re-hash its complete closure."""

    config_path = config_path.resolve()
    _require_sha("expected supervisor config SHA", expected_sha256)
    if not config_path.is_file() or _sha256_file(config_path) != expected_sha256:
        raise SupervisorError("supervisor config SHA mismatch")
    data = _strict_object(config_path)
    _exact_keys(
        data,
        {
            "schema_version",
            "contract_id",
            "status",
            "auto_start",
            "real_robot_authorized",
            "supervisor_source",
            "environment",
            "runner",
            "execution_config",
            "activation",
            "handshake",
            "pods",
        },
        "supervisor config",
    )
    if (
        data["schema_version"] != 1
        or data["status"] != "manual_launch_only"
        or data["auto_start"] is not False
        or data["real_robot_authorized"] is not False
        or not isinstance(data["contract_id"], str)
        or not data["contract_id"]
    ):
        raise SupervisorError("supervisor config authority fields changed")

    source = data["supervisor_source"]
    if not isinstance(source, dict):
        raise SupervisorError("supervisor_source must be an object")
    _exact_keys(source, {"path", "sha256"}, "supervisor_source")
    source_path = _resolve_source(config_path, source["path"])
    source_sha = _require_sha("supervisor_source.sha256", source["sha256"])
    if source_path != Path(__file__).resolve() or _sha256_file(source_path) != source_sha:
        raise SupervisorError("supervisor source bytes/path differ from config")

    environment = data["environment"]
    if (
        not isinstance(environment, dict)
        or not environment
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or "\x00" in key
            or "=" in key
            or "\x00" in value
            for key, value in environment.items()
        )
    ):
        raise SupervisorError("environment must be a non-empty fixed string map")
    forbidden_environment = {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "LD_PRELOAD",
        "BASH_ENV",
        "ENV",
    }
    if forbidden_environment.intersection(environment) or any(
        key.startswith("DYLD_") for key in environment
    ):
        raise SupervisorError("environment contains an interpreter/shell injection variable")
    if environment.get("PYTHONUNBUFFERED") != "1":
        raise SupervisorError("fixed environment must keep Python output unbuffered")

    common: dict[str, dict[str, Any]] = {}
    for name in ("runner", "execution_config", "activation"):
        common[name] = _absolute_file_binding(name, data[name])
        bound_path = Path(common[name]["path"])
        validation_path = (
            bound_path
            if validation_path_resolver is None
            else validation_path_resolver(bound_path)
        )
        _validate_bound_file(name, common[name], validation_path=validation_path)
    if Path(common["runner"]["path"]).name != "run_phase1_fresh_sz_model4000_q50.py":
        raise SupervisorError("runner path does not name the frozen model-4000 consumer")

    handshake = data["handshake"]
    if not isinstance(handshake, dict):
        raise SupervisorError("handshake must be an object")
    _exact_keys(
        handshake,
        {
            "hello_timeout_seconds",
            "commit_timeout_seconds",
            "ack_observation_seconds",
            "exec_observation_seconds",
            "poll_seconds",
        },
        "handshake",
    )
    hello_timeout = handshake["hello_timeout_seconds"]
    commit_timeout = handshake["commit_timeout_seconds"]
    ack_observation = handshake["ack_observation_seconds"]
    exec_observation = handshake["exec_observation_seconds"]
    poll = handshake["poll_seconds"]
    if (
        isinstance(hello_timeout, bool)
        or not isinstance(hello_timeout, (int, float))
        or not 1 <= hello_timeout <= 60
        or isinstance(commit_timeout, bool)
        or not isinstance(commit_timeout, (int, float))
        or not 1 <= commit_timeout <= 120
        or isinstance(ack_observation, bool)
        or not isinstance(ack_observation, (int, float))
        or not 0.05 <= ack_observation <= 30
        or isinstance(exec_observation, bool)
        or not isinstance(exec_observation, (int, float))
        or not 0.05 <= exec_observation <= 30
        or isinstance(poll, bool)
        or not isinstance(poll, (int, float))
        or not 0.01 <= poll <= 0.5
    ):
        raise SupervisorError("handshake timing is outside the fixed safe range")

    pods = data["pods"]
    if not isinstance(pods, dict) or set(pods) != {"pod1", "pod2"}:
        raise SupervisorError("supervisor must bind exactly pod1 and pod2")
    normalized_pods: dict[str, dict[str, Any]] = {}
    state_dirs: set[Path] = set()
    for pod, raw in pods.items():
        if not isinstance(raw, dict):
            raise SupervisorError(f"{pod} binding must be an object")
        _exact_keys(
            raw,
            {
                "launch_authorized",
                "blocker",
                "python",
                "runtime_contract",
                "state_dir",
                "result_path",
                "arm_order",
            },
            pod,
        )
        launch_authorized = raw["launch_authorized"]
        blocker = raw["blocker"]
        if not isinstance(launch_authorized, bool) or not isinstance(blocker, str):
            raise SupervisorError(f"{pod} launch authority fields malformed")
        if launch_authorized:
            if blocker:
                raise SupervisorError(f"{pod} cannot be authorized with a blocker")
            python = _executable_binding(f"{pod}.python", raw["python"])
        else:
            if raw["python"] is not None or not blocker:
                raise SupervisorError(f"{pod} blocked entry needs null Python and a reason")
            python = None
        runtime = _absolute_file_binding(f"{pod}.runtime_contract", raw["runtime_contract"])
        state_dir = Path(str(raw["state_dir"]))
        result_path = Path(str(raw["result_path"]))
        if not state_dir.is_absolute() or not result_path.is_absolute():
            raise SupervisorError(f"{pod} state/result paths must be absolute")
        if state_dir in state_dirs:
            raise SupervisorError("Pod supervisor state directories must be distinct")
        state_dirs.add(state_dir)
        expected_order = ["seed1", "seed3"] if pod == "pod1" else ["seed2", "seed4"]
        if raw["arm_order"] != expected_order:
            raise SupervisorError(f"{pod} arm order changed")
        normalized_pods[pod] = {
            "launch_authorized": launch_authorized,
            "blocker": blocker,
            "python": python,
            "runtime_contract": runtime,
            "state_dir": str(state_dir),
            "result_path": str(result_path),
            "arm_order": expected_order,
        }

    normalized = dict(data)
    normalized["_config_path"] = str(config_path)
    normalized["_config_sha256"] = expected_sha256
    normalized["_supervisor_source_path"] = str(source_path)
    normalized["_supervisor_source_sha256"] = source_sha
    normalized["runner"] = common["runner"]
    normalized["execution_config"] = common["execution_config"]
    normalized["activation"] = common["activation"]
    normalized["pods"] = normalized_pods
    return normalized


def _runner_argv(config: Mapping[str, Any], pod: str) -> list[str]:
    binding = config["pods"][pod]
    return [
        str(binding["python"]["path"]),
        str(config["runner"]["path"]),
        "--config",
        str(config["execution_config"]["path"]),
        "--expected-config-sha256",
        str(config["execution_config"]["sha256"]),
        "--activation",
        str(config["activation"]["path"]),
        "--expected-activation-sha256",
        str(config["activation"]["sha256"]),
        "run",
        "--pod",
        pod,
        "--runtime-contract",
        str(binding["runtime_contract"]["path"]),
        "--expected-runtime-contract-sha256",
        str(binding["runtime_contract"]["sha256"]),
    ]


def _runner_environment(config: Mapping[str, Any]) -> dict[str, str]:
    return dict(config["environment"])


def _require_invoking_environment(config: Mapping[str, Any]) -> None:
    expected = dict(config["environment"])
    observed = dict(os.environ)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            key for key in set(expected).intersection(observed) if expected[key] != observed[key]
        )
        raise SupervisorError(
            "invoking environment differs from fixed config: "
            f"missing={missing} extra={extra} changed={changed}"
        )


def _require_pod_launch_ready(config: Mapping[str, Any], pod: str) -> None:
    binding = config["pods"][pod]
    if binding["launch_authorized"] is not True:
        raise SupervisorError(f"{pod} launch is blocked: {binding['blocker']}")
    python = binding["python"]
    if python is None:
        raise SupervisorError(f"{pod} has no frozen Python executable")
    _validate_executable_binding(f"{pod}.python", python)
    _validate_bound_file(f"{pod}.runtime_contract", binding["runtime_contract"])
    current_executable = Path(sys.executable)
    try:
        current_resolved = current_executable.resolve(strict=True)
    except OSError as exc:
        raise SupervisorError(f"cannot resolve current Python executable: {exc}") from exc
    if (
        current_resolved != Path(python["resolved_path"])
        or _sha256_file(current_executable) != python["sha256"]
    ):
        raise SupervisorError("supervisor must itself run under the Pod-bound Python executable")


def _read_proc_identity(pid: int) -> dict[str, Any] | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    executable_path = Path("/proc") / str(pid) / "exe"
    environment_path = Path("/proc") / str(pid) / "environ"
    try:
        stat = stat_path.read_text(encoding="utf-8")
        close = stat.rfind(")")
        if close < 0:
            raise SupervisorError(f"malformed proc stat for PID {pid}")
        head = stat[:close]
        tail = stat[close + 2 :].split()
        observed_pid = int(head.split(" ", 1)[0])
        if len(tail) < 20:
            raise SupervisorError(f"short proc stat for PID {pid}")
        raw_cmdline = cmdline_path.read_bytes()
        executable_realpath = os.readlink(executable_path)
        if executable_realpath.endswith(" (deleted)"):
            raise SupervisorError(f"proc executable was deleted for PID {pid}")
        executable_sha256 = _sha256_file(executable_path)
        raw_environment = environment_path.read_bytes()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError) as exc:
        raise SupervisorError(f"cannot read proc identity for PID {pid}: {exc}") from exc
    environment: dict[str, str] = {}
    for item in raw_environment.split(b"\0"):
        if not item:
            continue
        if b"=" not in item:
            raise SupervisorError(f"malformed proc environment for PID {pid}")
        key, value = item.split(b"=", 1)
        decoded_key = os.fsdecode(key)
        if decoded_key in environment:
            raise SupervisorError(f"duplicate proc environment key for PID {pid}")
        environment[decoded_key] = os.fsdecode(value)
    return {
        "pid": observed_pid,
        "pgid": int(tail[2]),
        "state": tail[0],
        "start_ticks": int(tail[19]),
        "cmdline": [
            os.fsdecode(item) for item in raw_cmdline.split(b"\0") if item
        ],
        "executable_realpath": executable_realpath,
        "executable_sha256": executable_sha256,
        "environment_sha256": _canonical_sha256(environment),
    }


def _read_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise SupervisorError(f"cannot read Linux boot id: {exc}") from exc
    parts = value.split("-")
    if [len(part) for part in parts] != [8, 4, 4, 4, 12] or any(
        character not in "0123456789abcdef" for character in value.replace("-", "")
    ):
        raise SupervisorError("Linux boot id has an invalid format")
    return value


def _close_inherited_descriptors() -> None:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        soft = int(os.sysconf("SC_OPEN_MAX"))
    os.closerange(3, int(soft))


def _descriptor_at_least_three(descriptor: int) -> int:
    if descriptor >= 3:
        return descriptor
    duplicate = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
    os.close(descriptor)
    return duplicate


def _diagnostic_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    try:
        _atomic_json_no_clobber(path, value)
    except Exception:
        pass


def _child_wait_and_exec(
    *,
    state_dir: Path,
    log_descriptor: int,
    nonce: str,
    pod: str,
    argv: list[str],
    environment: dict[str, str],
    cwd: Path,
    bindings: dict[str, Any],
    result_path: Path,
    commit_timeout: float,
    poll: float,
    identity_reader: IdentityReader,
    boot_id_reader: BootIdReader,
    after_rehash_hook: Callable[[], None] | None = None,
    ack_before_link_hook: Callable[[], None] | None = None,
    after_ack_hook: Callable[[], None] | None = None,
) -> None:
    try:
        os.setsid()
        null_descriptor = os.open("/dev/null", os.O_RDONLY)
        os.dup2(null_descriptor, 0)
        os.dup2(log_descriptor, 1)
        os.dup2(log_descriptor, 2)
        _close_inherited_descriptors()
        pid = os.getpid()
        identity = identity_reader(pid)
        if (
            identity is None
            or identity.get("pid") != pid
            or identity.get("pgid") != pid
            or not isinstance(identity.get("start_ticks"), int)
        ):
            raise SupervisorError("detached child identity is not PID=PGID with start ticks")
        boot_id = boot_id_reader()
        if result_path.exists():
            raise SupervisorError("terminal result already exists before child hello")
        commit_deadline_monotonic_ns = time.monotonic_ns() + int(
            commit_timeout * 1_000_000_000
        )
        hello_path = state_dir / "child_hello.json"
        ledger_path = state_dir / "launch_ledger.json"
        token_path = state_dir / "commit_token.json"
        ack_path = state_dir / "commit_ack.json"
        hello = {
            "schema_version": 1,
            "artifact_kind": "phase1_q50_supervisor_child_hello",
            "nonce": nonce,
            "pod": pod,
            "pid": pid,
            "pgid": pid,
            "proc_start_ticks": identity["start_ticks"],
            "boot_id": boot_id,
            "executable_realpath": identity.get("executable_realpath"),
            "executable_sha256": identity.get("executable_sha256"),
            "runner_argv": argv,
            "runner_argv_sha256": _canonical_sha256(argv),
            "environment_sha256": _canonical_sha256(environment),
            "cwd": str(cwd),
            "bindings": bindings,
            "result_path": str(result_path),
            "commit_deadline_monotonic_ns": commit_deadline_monotonic_ns,
            "created_unix_ns": time.time_ns(),
        }
        _atomic_json_no_clobber(hello_path, hello)
        while (
            not token_path.is_file()
            and time.monotonic_ns() < commit_deadline_monotonic_ns
        ):
            time.sleep(poll)
        if not token_path.is_file():
            _diagnostic_no_clobber(
                state_dir / "child_exit.json",
                {
                    "schema_version": 1,
                    "artifact_kind": "phase1_q50_supervisor_child_exit",
                    "status": "commit_token_timeout",
                    "pid": pid,
                    "pgid": pid,
                    "proc_start_ticks": identity["start_ticks"],
                    "boot_id": boot_id,
                    "hello_sha256": _sha256_file(hello_path),
                    "finished_unix_ns": time.time_ns(),
                },
            )
            os._exit(75)
        token = _strict_object(token_path)
        ledger = _strict_object(ledger_path)
        expected_token = {
            "schema_version": 1,
            "artifact_kind": "phase1_q50_supervisor_commit_token",
            "nonce": nonce,
            "pod": pod,
            "pid": pid,
            "pgid": pid,
            "proc_start_ticks": identity["start_ticks"],
            "boot_id": boot_id,
            "commit_deadline_monotonic_ns": commit_deadline_monotonic_ns,
            "hello_sha256": _sha256_file(hello_path),
            "ledger_sha256": _sha256_file(ledger_path),
        }
        if token != expected_token:
            raise SupervisorError("commit token does not bind this child and ledger")
        if (
            ledger.get("status") != "launch_prepared"
            or ledger.get("pid") != pid
            or ledger.get("pgid") != pid
            or ledger.get("proc_start_ticks") != identity["start_ticks"]
            or ledger.get("boot_id") != boot_id
            or ledger.get("commit_deadline_monotonic_ns")
            != commit_deadline_monotonic_ns
            or ledger.get("runner_argv") != argv
            or ledger.get("environment_sha256") != _canonical_sha256(environment)
            or ledger.get("bindings") != bindings
        ):
            raise SupervisorError("launch ledger does not bind this child execution")

        def require_commit_still_valid(stage: str) -> None:
            current_token = _strict_object(token_path)
            current = identity_reader(pid)
            if (
                current_token != expected_token
                or _sha256_file(ledger_path) != expected_token["ledger_sha256"]
                or current is None
                or current.get("pid") != pid
                or current.get("pgid") != pid
                or current.get("state") == "Z"
                or current.get("start_ticks") != identity["start_ticks"]
                or current.get("executable_realpath")
                != identity.get("executable_realpath")
                or current.get("executable_sha256")
                != identity.get("executable_sha256")
                or boot_id_reader() != boot_id
                or result_path.exists()
            ):
                raise SupervisorError(f"child commit invalid {stage}; refusing exec")

        require_commit_still_valid("before rehash")
        for name, binding in bindings.items():
            if name == "supervisor_config":
                continue
            if name == "python":
                _validate_executable_binding(name, binding)
            else:
                _validate_bound_file(name, binding)
        if _sha256_file(Path(bindings["supervisor_config"]["path"])) != bindings[
            "supervisor_config"
        ]["sha256"]:
            raise SupervisorError("supervisor config changed before exec")
        if after_rehash_hook is not None:
            after_rehash_hook()
        require_commit_still_valid("after rehash before acknowledgment")
        accepted_monotonic_ns = time.monotonic_ns()
        ack = {
            "schema_version": 1,
            "artifact_kind": "phase1_q50_supervisor_commit_ack",
            "pod": pod,
            "pid": pid,
            "pgid": pid,
            "proc_start_ticks": identity["start_ticks"],
            "boot_id": boot_id,
            "token_sha256": _sha256_file(token_path),
            "accepted_monotonic_ns": accepted_monotonic_ns,
        }
        _atomic_json_no_clobber(
            ack_path,
            ack,
            before_link_hook=ack_before_link_hook,
        )
        if after_ack_hook is not None:
            after_ack_hook()
        require_commit_still_valid("after acknowledgment before exec")
        os.chdir(cwd)
        os.write(1, b"[q50-supervisor] commit accepted; exec exact bound runner\n")
        os.execve(argv[0], argv, environment)
    except BaseException as exc:
        _diagnostic_no_clobber(
            state_dir / "child_exit.json",
            {
                "schema_version": 1,
                "artifact_kind": "phase1_q50_supervisor_child_exit",
                "status": "child_setup_or_exec_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_unix_ns": time.time_ns(),
            },
        )
        try:
            os.write(2, f"[q50-supervisor][FATAL] {type(exc).__name__}: {exc}\n".encode())
        except OSError:
            pass
        os._exit(74)


def _wait_for_file(path: Path, timeout: float, poll: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return True
        time.sleep(poll)
    return path.is_file()


def _require_live_identity(
    *,
    pid: int,
    pgid: int,
    start_ticks: int,
    boot_id: str,
    python: Mapping[str, Any],
    identity_reader: IdentityReader,
    boot_id_reader: BootIdReader,
    expected_argv: Sequence[str] | None = None,
    expected_environment_sha256: str | None = None,
) -> dict[str, Any]:
    identity = identity_reader(pid)
    if (
        identity is None
        or identity.get("pid") != pid
        or identity.get("pgid") != pgid
        or identity.get("state") == "Z"
        or identity.get("start_ticks") != start_ticks
        or identity.get("executable_realpath") != python["resolved_path"]
        or identity.get("executable_sha256") != python["sha256"]
        or boot_id_reader() != boot_id
    ):
        raise SupervisorError("live child process identity differs from the frozen contract")
    if expected_argv is not None and identity.get("cmdline") != list(expected_argv):
        raise SupervisorError("live child command line differs from the frozen runner argv")
    if (
        expected_environment_sha256 is not None
        and identity.get("environment_sha256") != expected_environment_sha256
    ):
        raise SupervisorError("live child environment differs from the frozen environment")
    return identity


def _launch_loaded(
    config: Mapping[str, Any],
    pod: str,
    *,
    identity_reader: IdentityReader = _read_proc_identity,
    boot_id_reader: BootIdReader = _read_boot_id,
    after_hello_hook: Callable[[], None] | None = None,
    child_after_rehash_hook: Callable[[], None] | None = None,
    child_ack_before_link_hook: Callable[[], None] | None = None,
    child_after_ack_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if pod not in config["pods"]:
        raise SupervisorError(f"unknown Pod {pod!r}")
    _require_pod_launch_ready(config, pod)
    binding = config["pods"][pod]
    state_dir = Path(binding["state_dir"])
    result_path = Path(binding["result_path"])
    if result_path.exists():
        raise SupervisorError(f"pre-existing terminal result forbids launch: {result_path}")
    state_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        state_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise SupervisorError(f"no-clobber supervisor state already exists: {state_dir}") from exc
    _fsync_directory(state_dir.parent)
    log_path = state_dir / "runner.stdout_stderr.log"
    log_descriptor = _descriptor_at_least_three(
        os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    )
    nonce = secrets.token_hex(32)
    argv = _runner_argv(config, pod)
    environment = _runner_environment(config)
    cwd = Path(config["runner"]["path"]).parent.parent
    bindings = {
        "supervisor_source": {
            "path": config["_supervisor_source_path"],
            "sha256": config["_supervisor_source_sha256"],
        },
        "supervisor_config": {
            "path": config["_config_path"],
            "sha256": config["_config_sha256"],
        },
        "python": dict(binding["python"]),
        "runner": dict(config["runner"]),
        "execution_config": dict(config["execution_config"]),
        "activation": dict(config["activation"]),
        "runtime_contract": dict(binding["runtime_contract"]),
    }
    try:
        pid = os.fork()
    except BaseException:
        os.close(log_descriptor)
        raise
    if pid == 0:
        _child_wait_and_exec(
            state_dir=state_dir,
            log_descriptor=log_descriptor,
            nonce=nonce,
            pod=pod,
            argv=argv,
            environment=environment,
            cwd=cwd,
            bindings=bindings,
            result_path=result_path,
            commit_timeout=float(config["handshake"]["commit_timeout_seconds"]),
            poll=float(config["handshake"]["poll_seconds"]),
            identity_reader=identity_reader,
            boot_id_reader=boot_id_reader,
            after_rehash_hook=child_after_rehash_hook,
            ack_before_link_hook=child_ack_before_link_hook,
            after_ack_hook=child_after_ack_hook,
        )
        os._exit(73)
    os.close(log_descriptor)
    hello_path = state_dir / "child_hello.json"
    if not _wait_for_file(
        hello_path,
        float(config["handshake"]["hello_timeout_seconds"]),
        float(config["handshake"]["poll_seconds"]),
    ):
        _diagnostic_no_clobber(
            state_dir / "parent_failure.json",
            {
                "schema_version": 1,
                "artifact_kind": "phase1_q50_supervisor_parent_failure",
                "status": "child_hello_timeout",
                "expected_pid": pid,
                "finished_unix_ns": time.time_ns(),
            },
        )
        raise SupervisorError("child hello timed out; no commit token was written")
    hello = _strict_object(hello_path)
    _exact_keys(
        hello,
        {
            "schema_version",
            "artifact_kind",
            "nonce",
            "pod",
            "pid",
            "pgid",
            "proc_start_ticks",
            "boot_id",
            "executable_realpath",
            "executable_sha256",
            "runner_argv",
            "runner_argv_sha256",
            "environment_sha256",
            "cwd",
            "bindings",
            "result_path",
            "commit_deadline_monotonic_ns",
            "created_unix_ns",
        },
        "child hello",
    )
    boot_id = boot_id_reader()
    python_binding = binding["python"]
    expected_hello_fields = {
        "schema_version": 1,
        "artifact_kind": "phase1_q50_supervisor_child_hello",
        "nonce": nonce,
        "pod": pod,
        "pid": pid,
        "pgid": pid,
        "boot_id": boot_id,
        "executable_realpath": python_binding["resolved_path"],
        "executable_sha256": python_binding["sha256"],
        "runner_argv": argv,
        "runner_argv_sha256": _canonical_sha256(argv),
        "environment_sha256": _canonical_sha256(environment),
        "cwd": str(cwd),
        "bindings": bindings,
        "result_path": str(result_path),
    }
    start_ticks = hello.get("proc_start_ticks")
    deadline_monotonic_ns = hello.get("commit_deadline_monotonic_ns")
    if (
        not isinstance(start_ticks, int)
        or not isinstance(deadline_monotonic_ns, int)
        or any(hello.get(key) != value for key, value in expected_hello_fields.items())
    ):
        _diagnostic_no_clobber(
            state_dir / "parent_failure.json",
            {
                "schema_version": 1,
                "artifact_kind": "phase1_q50_supervisor_parent_failure",
                "status": "child_identity_or_hello_mismatch",
                "expected_pid": pid,
                "finished_unix_ns": time.time_ns(),
            },
        )
        raise SupervisorError("child identity/hello validation failed; no commit token was written")
    _require_live_identity(
        pid=pid,
        pgid=pid,
        start_ticks=start_ticks,
        boot_id=boot_id,
        python=python_binding,
        identity_reader=identity_reader,
        boot_id_reader=boot_id_reader,
    )
    if after_hello_hook is not None:
        after_hello_hook()
    safety_margin_ns = int(float(config["handshake"]["poll_seconds"]) * 2e9)
    _require_live_identity(
        pid=pid,
        pgid=pid,
        start_ticks=start_ticks,
        boot_id=boot_id,
        python=python_binding,
        identity_reader=identity_reader,
        boot_id_reader=boot_id_reader,
    )
    if time.monotonic_ns() + safety_margin_ns >= deadline_monotonic_ns:
        raise SupervisorError("child commit deadline expired before ledger publication")
    if result_path.exists():
        raise SupervisorError("terminal result appeared before ledger publication")
    ledger_path = state_dir / "launch_ledger.json"
    token_path = state_dir / "commit_token.json"
    ack_path = state_dir / "commit_ack.json"
    observation_path = state_dir / "exec_observation.json"
    ledger = {
        "schema_version": 1,
        "artifact_kind": "phase1_q50_supervisor_launch_ledger",
        "status": "launch_prepared",
        "contract_id": config["contract_id"],
        "pod": pod,
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": start_ticks,
        "boot_id": boot_id,
        "commit_deadline_monotonic_ns": deadline_monotonic_ns,
        "runner_argv": argv,
        "runner_argv_sha256": _canonical_sha256(argv),
        "environment_sha256": _canonical_sha256(environment),
        "cwd": str(cwd),
        "bindings": bindings,
        "hello": {"path": str(hello_path), "sha256": _sha256_file(hello_path)},
        "commit_token_path": str(token_path),
        "commit_ack_path": str(ack_path),
        "exec_observation_path": str(observation_path),
        "log_path": str(log_path),
        "result_path": binding["result_path"],
        "arm_order": binding["arm_order"],
        "committed_unix_ns": time.time_ns(),
        "committed_monotonic_ns": time.monotonic_ns(),
    }
    _atomic_json_no_clobber(ledger_path, ledger)
    _require_live_identity(
        pid=pid,
        pgid=pid,
        start_ticks=start_ticks,
        boot_id=boot_id,
        python=python_binding,
        identity_reader=identity_reader,
        boot_id_reader=boot_id_reader,
    )
    if time.monotonic_ns() + safety_margin_ns >= deadline_monotonic_ns:
        raise SupervisorError("child commit deadline expired before token publication")
    if result_path.exists():
        raise SupervisorError("terminal result appeared before token publication")
    token = {
        "schema_version": 1,
        "artifact_kind": "phase1_q50_supervisor_commit_token",
        "nonce": nonce,
        "pod": pod,
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": start_ticks,
        "boot_id": boot_id,
        "commit_deadline_monotonic_ns": deadline_monotonic_ns,
        "hello_sha256": _sha256_file(hello_path),
        "ledger_sha256": _sha256_file(ledger_path),
    }
    _atomic_json_no_clobber(token_path, token)

    def committed_observation(
        status: str,
        *,
        observed_identity: Mapping[str, Any] | None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        commit_ack_sha256 = _sha256_file(ack_path) if ack_path.is_file() else None
        child_exit_path = state_dir / "child_exit.json"
        observation = {
            "schema_version": 1,
            "artifact_kind": "phase1_q50_supervisor_exec_observation",
            "status": status,
            "pod": pod,
            "pid": pid,
            "pgid": pid,
            "proc_start_ticks": start_ticks,
            "boot_id": boot_id,
            "commit_token_sha256": _sha256_file(token_path),
            "commit_ack_sha256": commit_ack_sha256,
            "observed_monotonic_ns": time.monotonic_ns(),
            "process_present": observed_identity is not None,
            "observed_state": (
                None if observed_identity is None else observed_identity.get("state")
            ),
            "observed_executable_realpath": (
                None
                if observed_identity is None
                else observed_identity.get("executable_realpath")
            ),
            "observed_executable_sha256": (
                None
                if observed_identity is None
                else observed_identity.get("executable_sha256")
            ),
            "observed_cmdline": (
                None if observed_identity is None else observed_identity.get("cmdline")
            ),
            "observed_environment_sha256": (
                None
                if observed_identity is None
                else observed_identity.get("environment_sha256")
            ),
            "result_present": result_path.is_file(),
            "child_exit_present": child_exit_path.is_file(),
            "failure_reason": failure_reason,
        }
        _atomic_json_no_clobber(observation_path, observation)
        return {
            "status": status,
            "pod": pod,
            "pid": pid,
            "pgid": pid,
            "proc_start_ticks": start_ticks,
            "boot_id": boot_id,
            "ledger_path": str(ledger_path),
            "ledger_sha256": _sha256_file(ledger_path),
            "commit_token_sha256": _sha256_file(token_path),
            "commit_ack_sha256": commit_ack_sha256,
            "exec_observation_path": str(observation_path),
            "exec_observation_sha256": _sha256_file(observation_path),
            "log_path": str(log_path),
            "result_path": binding["result_path"],
            "failure_reason": failure_reason,
        }

    ack: dict[str, Any] | None = None
    ack_observation_deadline = time.monotonic() + float(
        config["handshake"]["ack_observation_seconds"]
    )
    while time.monotonic() < ack_observation_deadline:
        if ack_path.is_file():
            ack = _strict_object(ack_path)
            break
        time.sleep(float(config["handshake"]["poll_seconds"]))
    if ack is None and ack_path.is_file():
        ack = _strict_object(ack_path)
    if ack is None:
        observed_identity = identity_reader(pid)
        status = "token_published_pending_ack"
        failure_reason = None
        if observed_identity is None or observed_identity.get("state") == "Z":
            status = "committed_child_failed"
            failure_reason = "committed child exited before acknowledgment"
        else:
            try:
                _require_live_identity(
                    pid=pid,
                    pgid=pid,
                    start_ticks=start_ticks,
                    boot_id=boot_id,
                    python=python_binding,
                    identity_reader=identity_reader,
                    boot_id_reader=boot_id_reader,
                )
            except SupervisorError as exc:
                status = "committed_child_failed"
                failure_reason = str(exc)
        return committed_observation(
            status,
            observed_identity=observed_identity,
            failure_reason=failure_reason,
        )
    expected_ack = {
        "schema_version": 1,
        "artifact_kind": "phase1_q50_supervisor_commit_ack",
        "pod": pod,
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": start_ticks,
        "boot_id": boot_id,
        "token_sha256": _sha256_file(token_path),
    }
    if (
        set(ack) != set(expected_ack) | {"accepted_monotonic_ns"}
        or any(ack.get(key) != value for key, value in expected_ack.items())
        or not isinstance(ack.get("accepted_monotonic_ns"), int)
    ):
        return committed_observation(
            "committed_child_failed",
            observed_identity=identity_reader(pid),
            failure_reason="child commit acknowledgment is invalid",
        )

    observation_deadline = time.monotonic() + float(
        config["handshake"]["exec_observation_seconds"]
    )
    last_identity: dict[str, Any] | None = None
    while time.monotonic() < observation_deadline:
        last_identity = identity_reader(pid)
        if last_identity is None or last_identity.get("state") == "Z":
            break
        if last_identity is not None:
            try:
                _require_live_identity(
                    pid=pid,
                    pgid=pid,
                    start_ticks=start_ticks,
                    boot_id=boot_id,
                    python=python_binding,
                    identity_reader=identity_reader,
                    boot_id_reader=boot_id_reader,
                    expected_argv=argv,
                    expected_environment_sha256=_canonical_sha256(environment),
                )
            except SupervisorError:
                pass
            else:
                return {
                    "status": "running_exact",
                    "pod": pod,
                    "pid": pid,
                    "pgid": pid,
                    "proc_start_ticks": start_ticks,
                    "boot_id": boot_id,
                    "ledger_path": str(ledger_path),
                    "ledger_sha256": _sha256_file(ledger_path),
                    "commit_ack_sha256": _sha256_file(ack_path),
                    "log_path": str(log_path),
                    "result_path": binding["result_path"],
                }
        time.sleep(float(config["handshake"]["poll_seconds"]))
    if last_identity is None or last_identity.get("state") == "Z":
        if result_path.is_file():
            terminal = _validate_terminal_result(result_path, config, pod)
            return {
                "status": "terminal_result_validated",
                "pod": pod,
                "pid": pid,
                "pgid": pid,
                "proc_start_ticks": start_ticks,
                "boot_id": boot_id,
                "ledger_path": str(ledger_path),
                "ledger_sha256": _sha256_file(ledger_path),
                "commit_token_sha256": _sha256_file(token_path),
                "commit_ack_sha256": _sha256_file(ack_path),
                "log_path": str(log_path),
                "result": terminal,
            }
        return committed_observation(
            "committed_child_failed",
            observed_identity=last_identity,
            failure_reason="committed child exited before exact runner exec was observed",
        )
    try:
        _require_live_identity(
            pid=pid,
            pgid=pid,
            start_ticks=start_ticks,
            boot_id=boot_id,
            python=python_binding,
            identity_reader=identity_reader,
            boot_id_reader=boot_id_reader,
        )
    except SupervisorError as exc:
        return committed_observation(
            "committed_child_failed",
            observed_identity=last_identity,
            failure_reason=str(exc),
        )
    if (
        last_identity.get("cmdline") == argv
        and last_identity.get("environment_sha256") != _canonical_sha256(environment)
    ):
        return committed_observation(
            "committed_child_failed",
            observed_identity=last_identity,
            failure_reason="exact runner argv has a different environment",
        )
    return committed_observation(
        "committed_pending_exec",
        observed_identity=last_identity,
    )


def launch(config_path: Path, expected_config_sha256: str, pod: str) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise SupervisorError("launch requires Linux procfs and detached-session semantics")
    config = load_supervisor_config(config_path, expected_config_sha256)
    _require_invoking_environment(config)
    return _launch_loaded(config, pod)


def _validate_preserved_ledger(
    config: Mapping[str, Any], pod: str, state_dir: Path
) -> dict[str, Any]:
    hello_path = state_dir / "child_hello.json"
    ledger_path = state_dir / "launch_ledger.json"
    token_path = state_dir / "commit_token.json"
    ack_path = state_dir / "commit_ack.json"
    observation_path = state_dir / "exec_observation.json"
    hello = _strict_object(hello_path)
    ledger = _strict_object(ledger_path)
    token = _strict_object(token_path)
    _exact_keys(
        hello,
        {
            "schema_version",
            "artifact_kind",
            "nonce",
            "pod",
            "pid",
            "pgid",
            "proc_start_ticks",
            "boot_id",
            "executable_realpath",
            "executable_sha256",
            "runner_argv",
            "runner_argv_sha256",
            "environment_sha256",
            "cwd",
            "bindings",
            "result_path",
            "commit_deadline_monotonic_ns",
            "created_unix_ns",
        },
        "preserved child hello",
    )
    _exact_keys(
        ledger,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "contract_id",
            "pod",
            "pid",
            "pgid",
            "proc_start_ticks",
            "boot_id",
            "commit_deadline_monotonic_ns",
            "runner_argv",
            "runner_argv_sha256",
            "environment_sha256",
            "cwd",
            "bindings",
            "hello",
            "commit_token_path",
            "commit_ack_path",
            "exec_observation_path",
            "log_path",
            "result_path",
            "arm_order",
            "committed_unix_ns",
            "committed_monotonic_ns",
        },
        "launch ledger",
    )
    _exact_keys(
        token,
        {
            "schema_version",
            "artifact_kind",
            "nonce",
            "pod",
            "pid",
            "pgid",
            "proc_start_ticks",
            "boot_id",
            "commit_deadline_monotonic_ns",
            "hello_sha256",
            "ledger_sha256",
        },
        "commit token",
    )
    binding = config["pods"][pod]
    argv = _runner_argv(config, pod)
    expected_bindings = {
        "supervisor_source": {
            "path": config["_supervisor_source_path"],
            "sha256": config["_supervisor_source_sha256"],
        },
        "supervisor_config": {
            "path": config["_config_path"],
            "sha256": config["_config_sha256"],
        },
        "python": dict(binding["python"]),
        "runner": dict(config["runner"]),
        "execution_config": dict(config["execution_config"]),
        "activation": dict(config["activation"]),
        "runtime_contract": dict(binding["runtime_contract"]),
    }
    expected_ledger = {
        "status": "launch_prepared",
        "contract_id": config["contract_id"],
        "pod": pod,
        "runner_argv": argv,
        "runner_argv_sha256": _canonical_sha256(argv),
        "cwd": str(Path(config["runner"]["path"]).parent.parent),
        "bindings": expected_bindings,
        "hello": {"path": str(hello_path), "sha256": _sha256_file(hello_path)},
        "commit_token_path": str(token_path),
        "commit_ack_path": str(ack_path),
        "exec_observation_path": str(observation_path),
        "log_path": str(state_dir / "runner.stdout_stderr.log"),
        "result_path": binding["result_path"],
        "arm_order": binding["arm_order"],
    }
    if (
        ledger.get("schema_version") != 1
        or ledger.get("artifact_kind") != "phase1_q50_supervisor_launch_ledger"
        or any(ledger.get(key) != value for key, value in expected_ledger.items())
        or not isinstance(ledger.get("pid"), int)
        or ledger.get("pid") <= 1
        or ledger.get("pgid") != ledger.get("pid")
        or not isinstance(ledger.get("proc_start_ticks"), int)
        or ledger.get("boot_id") != hello.get("boot_id")
        or ledger.get("environment_sha256") != _canonical_sha256(config["environment"])
        or not isinstance(ledger.get("committed_unix_ns"), int)
        or not isinstance(ledger.get("committed_monotonic_ns"), int)
        or not isinstance(ledger.get("commit_deadline_monotonic_ns"), int)
    ):
        raise SupervisorError("preserved launch ledger differs from bound config")
    expected_token = {
        "schema_version": 1,
        "artifact_kind": "phase1_q50_supervisor_commit_token",
        "nonce": hello.get("nonce"),
        "pod": pod,
        "pid": ledger["pid"],
        "pgid": ledger["pgid"],
        "proc_start_ticks": ledger["proc_start_ticks"],
        "boot_id": ledger["boot_id"],
        "commit_deadline_monotonic_ns": ledger["commit_deadline_monotonic_ns"],
        "hello_sha256": _sha256_file(hello_path),
        "ledger_sha256": _sha256_file(ledger_path),
    }
    if token != expected_token:
        raise SupervisorError("commit token does not bind preserved hello/ledger")
    if (
        hello.get("pid") != ledger["pid"]
        or hello.get("pgid") != ledger["pgid"]
        or hello.get("proc_start_ticks") != ledger["proc_start_ticks"]
        or hello.get("boot_id") != ledger["boot_id"]
        or hello.get("executable_realpath") != binding["python"]["resolved_path"]
        or hello.get("executable_sha256") != binding["python"]["sha256"]
        or hello.get("runner_argv") != argv
        or hello.get("environment_sha256") != ledger["environment_sha256"]
        or hello.get("bindings") != expected_bindings
        or hello.get("result_path") != binding["result_path"]
        or hello.get("commit_deadline_monotonic_ns")
        != ledger["commit_deadline_monotonic_ns"]
    ):
        raise SupervisorError("preserved child hello differs from committed ledger")
    preserved = dict(ledger)
    if ack_path.is_file():
        ack = _strict_object(ack_path)
        expected_ack = {
            "schema_version": 1,
            "artifact_kind": "phase1_q50_supervisor_commit_ack",
            "pod": pod,
            "pid": ledger["pid"],
            "pgid": ledger["pgid"],
            "proc_start_ticks": ledger["proc_start_ticks"],
            "boot_id": ledger["boot_id"],
            "token_sha256": _sha256_file(token_path),
        }
        if (
            set(ack) != set(expected_ack) | {"accepted_monotonic_ns"}
            or any(ack.get(key) != value for key, value in expected_ack.items())
            or not isinstance(ack.get("accepted_monotonic_ns"), int)
        ):
            raise SupervisorError("preserved commit acknowledgment is invalid")
        preserved["_commit_ack"] = ack
    else:
        preserved["_commit_ack"] = None
    return preserved


def _validate_terminal_result(
    path: Path, config: Mapping[str, Any], pod: str
) -> dict[str, Any]:
    try:
        frozen_bytes = path.read_bytes()
    except OSError as exc:
        raise SupervisorError(f"cannot freeze terminal result bytes: {exc}") from exc
    frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    frozen_document = _strict_object_bytes(frozen_bytes, f"frozen terminal result {path}")
    _exact_keys(
        frozen_document,
        {"schema_version", "artifact_kind", "content_sha256", "content"},
        "frozen terminal result",
    )
    frozen_content = frozen_document["content"]
    if (
        not isinstance(frozen_content, dict)
        or frozen_document["content_sha256"] != _canonical_sha256(frozen_content)
    ):
        raise SupervisorError("frozen terminal result canonical content hash is invalid")
    runner_path = Path(config["runner"]["path"])
    module_name = f"_q50_bound_result_validator_{config['runner']['sha256'][:16]}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, runner_path)
        if spec is None or spec.loader is None:
            raise SupervisorError("cannot construct exact bound-runner validator import")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        execution_path = Path(config["execution_config"]["path"])
        execution = module.load_execution_config(execution_path)
        queue_path, queue, prereg_path, prereg = module._resolve_bound_sources(
            execution_path, execution
        )
        activation = module._validate_activation_document(
            Path(config["activation"]["path"]),
            config["activation"]["sha256"],
            queue_path,
            queue,
            prereg_path,
        )
        validated_content = module._validate_pod_result(
            path,
            frozen_sha256,
            execution,
            prereg,
            activation,
            config["execution_config"]["sha256"],
            pod=pod,
        )
        if (
            not isinstance(validated_content, dict)
            or validated_content.get("runtime_contract")
            != config["pods"][pod]["runtime_contract"]
        ):
            raise SupervisorError("bound runner accepted a different runtime contract")
    except SupervisorError:
        raise
    except BaseException as exc:
        raise SupervisorError(
            f"exact bound-runner terminal validation failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        post_bytes = path.read_bytes()
    except OSError as exc:
        raise SupervisorError(f"cannot reread terminal result after validation: {exc}") from exc
    post_sha256 = hashlib.sha256(post_bytes).hexdigest()
    if post_sha256 != frozen_sha256 or post_bytes != frozen_bytes:
        raise SupervisorError("terminal result bytes changed during bound validation")
    post_document = _strict_object_bytes(post_bytes, f"post-validation terminal result {path}")
    post_content = post_document.get("content")
    if (
        post_document != frozen_document
        or not isinstance(post_content, dict)
        or post_document.get("content_sha256") != _canonical_sha256(post_content)
        or post_content != validated_content
        or _canonical_sha256(post_content) != _canonical_sha256(validated_content)
    ):
        raise SupervisorError(
            "terminal result document differs from exact bound-validator content"
        )
    return {
        "path": str(path),
        "sha256": frozen_sha256,
        "content_sha256": frozen_document["content_sha256"],
    }


def _inspect_loaded(
    config: Mapping[str, Any],
    pod: str,
    *,
    identity_reader: IdentityReader = _read_proc_identity,
    boot_id_reader: BootIdReader = _read_boot_id,
) -> dict[str, Any]:
    if pod not in config["pods"]:
        raise SupervisorError(f"unknown Pod {pod!r}")
    binding = config["pods"][pod]
    state_dir = Path(binding["state_dir"])
    if not state_dir.exists():
        if binding["launch_authorized"] is not True:
            return {
                "status": "blocked_not_launched",
                "pod": pod,
                "state_dir": str(state_dir),
                "blocker": binding["blocker"],
            }
        _require_pod_launch_ready(config, pod)
        if Path(binding["result_path"]).exists():
            raise SupervisorError("terminal result exists without supervisor state")
        return {"status": "not_launched", "pod": pod, "state_dir": str(state_dir)}
    _require_pod_launch_ready(config, pod)
    if not state_dir.is_dir():
        raise SupervisorError("configured supervisor state path is not a directory")
    ledger_path = state_dir / "launch_ledger.json"
    if not ledger_path.is_file():
        child_exit = state_dir / "child_exit.json"
        return {
            "status": "handshake_not_committed",
            "pod": pod,
            "state_dir": str(state_dir),
            "child_exit": (
                {"path": str(child_exit), "sha256": _sha256_file(child_exit)}
                if child_exit.is_file()
                else None
            ),
        }
    token_path = state_dir / "commit_token.json"
    if not token_path.is_file():
        child_exit = state_dir / "child_exit.json"
        return {
            "status": "handshake_not_committed",
            "pod": pod,
            "state_dir": str(state_dir),
            "ledger": {"path": str(ledger_path), "sha256": _sha256_file(ledger_path)},
            "child_exit": (
                {"path": str(child_exit), "sha256": _sha256_file(child_exit)}
                if child_exit.is_file()
                else None
            ),
        }
    ledger = _validate_preserved_ledger(config, pod, state_dir)
    identity = identity_reader(ledger["pid"])
    result_path = Path(binding["result_path"])

    def committed_child_failed(reason: str) -> dict[str, Any]:
        child_exit = state_dir / "child_exit.json"
        return {
            "status": "committed_child_failed",
            "pod": pod,
            "pid": ledger["pid"],
            "pgid": ledger["pgid"],
            "proc_start_ticks": ledger["proc_start_ticks"],
            "boot_id": ledger["boot_id"],
            "failure_reason": reason,
            "ledger": {"path": str(ledger_path), "sha256": _sha256_file(ledger_path)},
            "log": {
                "path": ledger["log_path"],
                "sha256": _sha256_file(Path(ledger["log_path"])),
            },
            "child_exit": (
                {"path": str(child_exit), "sha256": _sha256_file(child_exit)}
                if child_exit.is_file()
                else None
            ),
            "result_present": result_path.is_file(),
        }

    if identity is not None and identity.get("state") != "Z":
        try:
            _require_live_identity(
                pid=ledger["pid"],
                pgid=ledger["pgid"],
                start_ticks=ledger["proc_start_ticks"],
                boot_id=ledger["boot_id"],
                python=binding["python"],
                identity_reader=identity_reader,
                boot_id_reader=boot_id_reader,
            )
        except SupervisorError as exc:
            return committed_child_failed(str(exc))
        exact_exec = (
            ledger["_commit_ack"] is not None
            and identity.get("cmdline") == ledger["runner_argv"]
            and identity.get("environment_sha256") == ledger["environment_sha256"]
        )
        if (
            ledger["_commit_ack"] is not None
            and identity.get("cmdline") == ledger["runner_argv"]
            and identity.get("environment_sha256") != ledger["environment_sha256"]
        ):
            return committed_child_failed("exact runner argv has a different environment")
        if ledger["_commit_ack"] is None and identity.get("cmdline") == ledger["runner_argv"]:
            return committed_child_failed("exact runner exec is visible without its durable ack")
        status = (
            "running_exact"
            if exact_exec
            else (
                "committed_pending_exec"
                if ledger["_commit_ack"] is not None
                else "token_published_pending_ack"
            )
        )
        return {
            "status": status,
            "pod": pod,
            "pid": ledger["pid"],
            "pgid": ledger["pgid"],
            "proc_start_ticks": ledger["proc_start_ticks"],
            "boot_id": ledger["boot_id"],
            "runner_argv_sha256": ledger["runner_argv_sha256"],
            "environment_sha256": ledger["environment_sha256"],
            "observed_state": identity.get("state"),
            "observed_executable_realpath": identity.get("executable_realpath"),
            "observed_executable_sha256": identity.get("executable_sha256"),
            "observed_cmdline": identity.get("cmdline"),
            "observed_environment_sha256": identity.get("environment_sha256"),
            "ledger": {"path": str(ledger_path), "sha256": _sha256_file(ledger_path)},
            "log": {
                "path": ledger["log_path"],
                "sha256": _sha256_file(Path(ledger["log_path"])),
            },
            "result_present": result_path.is_file(),
        }
    if result_path.is_file():
        if ledger["_commit_ack"] is None:
            return committed_child_failed(
                "terminal result exists without the committed child's durable ack"
            )
        terminal = _validate_terminal_result(result_path, config, pod)
        return {
            "status": "terminal_result_validated",
            "pod": pod,
            "pid": ledger["pid"],
            "pgid": ledger["pgid"],
            "proc_start_ticks": ledger["proc_start_ticks"],
            "boot_id": ledger["boot_id"],
            "ledger": {"path": str(ledger_path), "sha256": _sha256_file(ledger_path)},
            "log": {
                "path": ledger["log_path"],
                "sha256": _sha256_file(Path(ledger["log_path"])),
            },
            "result": terminal,
        }
    return committed_child_failed("committed child exited without a terminal result")


def inspect(config_path: Path, expected_config_sha256: str, pod: str) -> dict[str, Any]:
    config = load_supervisor_config(config_path, expected_config_sha256)
    _require_invoking_environment(config)
    return _inspect_loaded(config, pod)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-config", required=True, type=Path)
    parser.add_argument("--expected-supervisor-config-sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--pod", choices=("pod1", "pod2"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "launch":
            result = launch(
                args.supervisor_config,
                args.expected_supervisor_config_sha256,
                args.pod,
            )
        else:
            result = inspect(
                args.supervisor_config,
                args.expected_supervisor_config_sha256,
                args.pod,
            )
    except SupervisorError as exc:
        print(f"[q50-supervisor][FATAL] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return (
        3
        if result["status"] in {"handshake_not_committed", "committed_child_failed"}
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
