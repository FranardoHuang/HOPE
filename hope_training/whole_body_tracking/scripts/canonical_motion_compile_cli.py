#!/usr/bin/env python3
"""Strict command-line adapter for the canonical motion compiler.

This module owns no motion-generation logic.  It validates and content-binds a
single compiler job, constructs :class:`canonical_motion_compiler.CompilerOptions`,
calls the compiler, and writes a final run receipt after the compiler's atomic
candidate publication.

The adapter can only publish ``compiler_candidate`` artifacts.  Passing this
CLI does not authorize training, deployment, or hardware use.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import multiprocessing
import multiprocessing.util as multiprocessing_util
import numbers
import os
import secrets
import signal
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

import canonical_motion_compiler as cmc
from canonical_motion_recipe import (
    CanonicalMotionRecipe,
    load_canonical_motion_recipe,
)
from mujoco_motion_player import RUNTIME_JOINT_NAMES


RUN_RECEIPT_NAME = "RUN_RECEIPT.json"
PUBLICATION_CLASS = "compiler_candidate"
_SHA256_HEX_LENGTH = 64
_MODEL_KEYS = ("mjcf", "urdf", "body_order")
_SCHEMA2_HASH_KEYS = frozenset(
    {
        "input_sha256",
        "ready_sha256",
        "mjcf_sha256",
        "body_order_sha256",
        "tool_sha256",
        "output_npz_sha256",
    }
)
_ACCELERATION_RECEIPT_KEYS = frozenset(
    {
        "acceleration_rad_s2",
        "all_positive",
        "joint_names",
        "limiting_source_frame",
        "method",
        "minimum_effort_margin_nm",
    }
)
_CANONICAL_BASE_COUNT = 5


class CanonicalMotionCompileCliError(RuntimeError):
    """The requested compiler job is incomplete, ambiguous, or unsafe."""


class CanonicalMotionCompileTerminated(BaseException):
    """A catchable parent termination request that must bypass compiler catches."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        try:
            name = signal.Signals(self.signum).name
        except ValueError:
            name = str(self.signum)
        super().__init__(f"canonical motion compile terminated by {name}")


class _FailClosedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CanonicalMotionCompileCliError(message)


@dataclass(frozen=True)
class _AccelerationReceipt:
    path: Path
    sha256: str
    acceleration_rad_s2: np.ndarray
    method: str


@dataclass(frozen=True)
class _AppendOnlyBase:
    recipe_path: Path
    recipe_sha256: str
    recipe: CanonicalMotionRecipe
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    output_directory: Path
    published_outputs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _ValidatedJob:
    recipe_path: Path
    repo_root: Path
    output_directory: Path
    recipe_sha256: str
    recipe: CanonicalMotionRecipe
    acceleration_receipt: _AccelerationReceipt
    options: cmc.CompilerOptions
    normalized_args: Mapping[str, Any]
    model_bindings: Mapping[str, Mapping[str, str]]
    cli_sha256: str
    append_only_base: _AppendOnlyBase | None
    station_center_shift_xy_m: tuple[float, float] | None


@dataclass
class _TerminationState:
    owner_pid: int
    received_signal: int | None = None
    output_directory: Path | None = None
    expected_manifest: Mapping[str, Any] | None = None
    output_identity: tuple[int, int] | None = None


@dataclass
class _ObservedProcess:
    pid: int
    linux_starttime: int | None
    pidfd: int | None = None


class _AfterForkGuard:
    """Weak-referenceable token for multiprocessing's after-fork registry."""


_AFTER_FORK_GUARD = _AfterForkGuard()
_ACTIVE_TERMINATION_STATE: _TerminationState | None = None
_PR_SET_PDEATHSIG = 1
_DESCENDANT_TERM_GRACE_S = 1.0
_DESCENDANT_KILL_GRACE_S = 0.5
_DESCENDANT_STABLE_EMPTY_S = 0.1
_LINUX_PIDFD_SEND_SIGNAL_SYSCALL = 424
_LINUX_PIDFD_OPEN_SYSCALL = 434


def _handled_termination_signals() -> list[int]:
    handled = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled.append(signal.SIGHUP)
    return handled


def _signal_contract() -> Mapping[str, Any]:
    handled = [
        signal.Signals(signum).name
        for signum in _handled_termination_signals()
    ]
    start_method = multiprocessing.get_start_method(allow_none=True)
    if start_method is None:
        start_method = multiprocessing.get_context().get_start_method()
    publication_mask = (
        "pthread_sigmask_closes_atomic_publish_to_inode_bind_window"
        if hasattr(signal, "pthread_sigmask")
        else (
            "unavailable; cleanup refuses to remove an output not already "
            "inode-bound when the signal arrived"
        )
    )
    return {
        "parent_signals_handled": handled,
        "catchable_signal_cleanup": (
            "continuously_rescan_descendants_then_TERM_and_stable_handle_KILL"
        ),
        "linux_fork_worker_parent_death_signal": (
            "PR_SET_PDEATHSIG_SIGTERM"
            if sys.platform.startswith("linux") and start_method == "fork"
            else "not_available_for_this_platform_or_start_method"
        ),
        "multiprocessing_start_method": start_method,
        "publication_signal_mask": publication_mask,
        "portable_boundary": (
            "without pthread_sigmask the publish-to-identity race fails closed "
            "by retaining an unproven output; descendant TERM without a "
            "stable process handle is best-effort only, and abrupt SIGKILL or "
            "host death has no portable cleanup guarantee"
        ),
        "publication_on_listed_catchable_termination": (
            "remove the requested output path only by inode-and-manifest-bound "
            "rename to a retained sibling quarantine; never recursively delete"
        ),
        "termination_quarantine_retention": (
            "the proven owned publication is retained for manual audit under "
            ".<output>.terminated-<pid>-<token>; it is not training-authorized"
        ),
        "abrupt_parent_death_boundary": (
            "Linux fork workers receive SIGTERM, but SIGKILL or host death "
            "cannot run parent-side output or staging cleanup"
        ),
    }


def _reset_worker_signals_and_set_parent_death(_token: object) -> None:
    """Run in forked multiprocessing workers before their task loop.

    Linux ``PR_SET_PDEATHSIG`` closes the hard-parent-death hole for the
    default ``fork`` ProcessPool path.  Spawn/forkserver and non-Linux workers
    rely on the parent's catchable-signal cleanup; there is deliberately no
    claim that portable code can survive parent ``SIGKILL`` or host death.
    """

    state = _ACTIVE_TERMINATION_STATE
    if state is None or os.getpid() == state.owner_pid:
        return
    for name, disposition in (
        ("SIGINT", signal.SIG_IGN),
        ("SIGTERM", signal.SIG_DFL),
        ("SIGHUP", signal.SIG_DFL),
    ):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), disposition)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    if not sys.platform.startswith("linux"):
        return
    expected_parent = state.owner_pid
    if os.getppid() != expected_parent:
        os._exit(125)
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = getattr(libc, "prctl", None)
    if prctl is None:
        os._exit(125)
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(125)
    if os.getppid() != expected_parent:
        os._exit(125)


multiprocessing_util.register_after_fork(
    _AFTER_FORK_GUARD, _reset_worker_signals_and_set_parent_death
)


def _linux_process_record(pid: int) -> tuple[str, int, int] | None:
    try:
        value = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8", errors="strict"
        )
    except (OSError, UnicodeError):
        return None
    right_parenthesis = value.rfind(")")
    if right_parenthesis < 0:
        return None
    fields = value[right_parenthesis + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return fields[0], int(fields[1]), int(fields[19])
    except ValueError:
        return None


def _linux_descendant_records(root_pid: int) -> dict[int, tuple[str, int, int]]:
    if not sys.platform.startswith("linux"):
        return {}
    records: dict[int, tuple[str, int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        record = _linux_process_record(pid)
        if record is not None:
            records[pid] = record
    descendants: set[int] = set()
    frontier = {int(root_pid)}
    while frontier:
        children = {
            pid
            for pid, (_, parent, _) in records.items()
            if parent in frontier and pid not in descendants
        }
        descendants.update(children)
        frontier = children
    descendants.discard(os.getpid())
    return {pid: records[pid] for pid in descendants}


def _linux_pidfd_open(pid: int) -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    if hasattr(os, "pidfd_open"):
        try:
            return os.pidfd_open(pid, 0)
        except (OSError, AttributeError):
            return None
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd_open = getattr(libc, "pidfd_open", None)
    if pidfd_open is not None:
        pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        pidfd_open.restype = ctypes.c_int
        descriptor = pidfd_open(pid, 0)
    else:
        syscall = getattr(libc, "syscall", None)
        if syscall is None:
            return None
        syscall.restype = ctypes.c_long
        descriptor = syscall(
            ctypes.c_long(_LINUX_PIDFD_OPEN_SYSCALL),
            ctypes.c_int(pid),
            ctypes.c_uint(0),
        )
    if descriptor < 0:
        return None
    return int(descriptor)


def _linux_pidfd_send_signal(
    pidfd: int, signum: int
) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if hasattr(signal, "pidfd_send_signal"):
        try:
            signal.pidfd_send_signal(pidfd, signum, None, 0)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        return True
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd_send_signal = getattr(libc, "pidfd_send_signal", None)
    if pidfd_send_signal is not None:
        pidfd_send_signal.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        pidfd_send_signal.restype = ctypes.c_int
        result = pidfd_send_signal(pidfd, signum, None, 0)
    else:
        syscall = getattr(libc, "syscall", None)
        if syscall is None:
            return False
        syscall.restype = ctypes.c_long
        result = syscall(
            ctypes.c_long(_LINUX_PIDFD_SEND_SIGNAL_SYSCALL),
            ctypes.c_int(pidfd),
            ctypes.c_int(signum),
            ctypes.c_void_p(),
            ctypes.c_uint(0),
        )
    if result == 0:
        return True
    # ESRCH means the stable process handle already refers to a dead process.
    if ctypes.get_errno() == errno.ESRCH:
        return True
    return False


def _open_pidfd(
    pid: int,
    expected_starttime: int,
    expected_parent: int,
) -> int | None:
    pidfd = _linux_pidfd_open(pid)
    if pidfd is None:
        return None
    record = _linux_process_record(pid)
    if (
        record is None
        or record[1] != expected_parent
        or record[2] != expected_starttime
    ):
        os.close(pidfd)
        return None
    return pidfd


def _observed_process_running(process: _ObservedProcess) -> bool:
    if process.pid <= 0 or process.pid == os.getpid():
        return False
    if process.linux_starttime is not None:
        record = _linux_process_record(process.pid)
        return (
            record is not None
            and record[0] != "Z"
            and record[2] == process.linux_starttime
        )
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_observed_signal(
    process: _ObservedProcess, signum: int
) -> None:
    if process.pidfd is not None:
        _linux_pidfd_send_signal(process.pidfd, signum)
        return
    # Linux must never fall back from a stable pidfd to check-then-kill: the
    # PID can be reused between those two operations.  On other POSIX systems
    # there is no pidfd equivalent; a single best-effort SIGTERM is the
    # explicitly documented portable boundary, and there is no raw-PID KILL.
    if sys.platform.startswith("linux") or signum != signal.SIGTERM:
        return
    try:
        os.kill(process.pid, signum)
    except ProcessLookupError:
        pass


def _observed_descendants(owner_pid: int) -> dict[int, _ObservedProcess]:
    records = _linux_descendant_records(owner_pid)
    pids = set(records)
    try:
        pids.update(
            int(child.pid)
            for child in multiprocessing.active_children()
            if child.pid is not None
        )
    except Exception:
        # Linux /proc remains the independent fallback.  On other platforms,
        # an empty result is handled by ordinary executor shutdown.
        pass
    pids.discard(owner_pid)
    pids.discard(os.getpid())
    observed: dict[int, _ObservedProcess] = {}
    for pid in pids:
        if pid <= 1:
            continue
        record = records.get(pid)
        if record is None and sys.platform.startswith("linux"):
            record = _linux_process_record(pid)
            # A PID supplied by active_children() but absent from the
            # descendant snapshot may have exited and been reused.  It is
            # eligible only while it is still our direct child.
            if record is None or record[1] != owner_pid:
                continue
        starttime = None if record is None else record[2]
        pidfd = (
            None
            if starttime is None
            else _open_pidfd(pid, starttime, record[1])
        )
        observed[pid] = _ObservedProcess(
            pid=pid,
            linux_starttime=starttime,
            pidfd=pidfd,
        )
    return observed


def _close_observed_process(process: _ObservedProcess) -> None:
    if process.pidfd is not None:
        os.close(process.pidfd)
        process.pidfd = None


def _merge_observed_descendants(
    observed: dict[int, _ObservedProcess],
    candidates: Mapping[int, _ObservedProcess],
) -> list[_ObservedProcess]:
    added: list[_ObservedProcess] = []
    for pid, candidate in candidates.items():
        existing = observed.get(pid)
        if existing is None:
            observed[pid] = candidate
            added.append(candidate)
            continue
        if (
            existing.linux_starttime is None
            and candidate.linux_starttime is not None
        ):
            # Upgrade a transient active_children-only observation to the
            # stable /proc identity and pidfd obtained by a later rescan.
            _close_observed_process(existing)
            observed[pid] = candidate
            added.append(candidate)
            continue
        if existing.linux_starttime == candidate.linux_starttime:
            if existing.pidfd is None and candidate.pidfd is not None:
                existing.pidfd = candidate.pidfd
                candidate.pidfd = None
                # The earlier raw observation was deliberately not signaled
                # on Linux.  Treat acquiring the pidfd as newly actionable.
                added.append(existing)
            _close_observed_process(candidate)
            continue
        if candidate.linux_starttime is None:
            _close_observed_process(candidate)
            continue
        if _observed_process_running(existing):
            _close_observed_process(candidate)
            continue
        _close_observed_process(existing)
        observed[pid] = candidate
        added.append(candidate)
    return added


def _signal_new_descendants(
    owner_pid: int,
    observed: dict[int, _ObservedProcess],
) -> bool:
    candidates = _observed_descendants(owner_pid)
    added = _merge_observed_descendants(observed, candidates)
    for process in added:
        _send_observed_signal(process, signal.SIGTERM)
    return bool(added)


def _running_observed(
    observed: Mapping[int, _ObservedProcess],
) -> list[_ObservedProcess]:
    return [
        process
        for process in observed.values()
        if _observed_process_running(process)
    ]


def _terminate_observed_descendants(owner_pid: int) -> None:
    observed: dict[int, _ObservedProcess] = {}
    try:
        deadline = time.monotonic() + _DESCENDANT_TERM_GRACE_S
        stable_empty_since: float | None = None
        while time.monotonic() < deadline:
            added = _signal_new_descendants(owner_pid, observed)
            running = _running_observed(observed)
            now = time.monotonic()
            if running or added:
                stable_empty_since = None
            elif stable_empty_since is None:
                stable_empty_since = now
            elif now - stable_empty_since >= _DESCENDANT_STABLE_EMPTY_S:
                break
            time.sleep(0.01)

        # Rescan during the hard-stop grace as well.  ProcessPool management
        # may still be finishing a concurrent worker launch when the first
        # scan runs.
        hard_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        hard_deadline = time.monotonic() + _DESCENDANT_KILL_GRACE_S
        stable_empty_since = None
        while time.monotonic() < hard_deadline:
            _signal_new_descendants(owner_pid, observed)
            running = _running_observed(observed)
            for process in running:
                if process.pidfd is not None:
                    _send_observed_signal(process, hard_signal)
            now = time.monotonic()
            if running:
                stable_empty_since = None
            elif stable_empty_since is None:
                stable_empty_since = now
            elif now - stable_empty_since >= _DESCENDANT_STABLE_EMPTY_S:
                break
            time.sleep(0.01)

        # One last observation closes the boundary at unwind.  Stable handles
        # can still be killed; processes without one are deliberately not
        # escalated through a reusable raw PID.
        _signal_new_descendants(owner_pid, observed)
        for process in _running_observed(observed):
            if process.pidfd is not None:
                _send_observed_signal(process, hard_signal)
    finally:
        for process in observed.values():
            _close_observed_process(process)


def _termination_handler(signum: int, _frame: Any) -> None:
    state = _ACTIVE_TERMINATION_STATE
    if state is None:
        raise CanonicalMotionCompileTerminated(signum)
    if os.getpid() != state.owner_pid:
        os._exit(128 + int(signum))
    if state.received_signal is not None:
        return
    state.received_signal = int(signum)
    for handled_signal in _handled_termination_signals():
        signal.signal(handled_signal, signal.SIG_IGN)
    _terminate_observed_descendants(state.owner_pid)
    raise CanonicalMotionCompileTerminated(signum)


@contextmanager
def _termination_guard():
    global _ACTIVE_TERMINATION_STATE
    if _ACTIVE_TERMINATION_STATE is not None:
        raise CanonicalMotionCompileCliError(
            "nested canonical compiler termination guards are forbidden"
        )
    state = _TerminationState(owner_pid=os.getpid())
    _ACTIVE_TERMINATION_STATE = state
    handled = _handled_termination_signals()
    previous = {signum: signal.getsignal(signum) for signum in handled}
    try:
        for signum in handled:
            signal.signal(signum, _termination_handler)
        yield state
    finally:
        if os.getpid() == state.owner_pid:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            _ACTIVE_TERMINATION_STATE = None


@contextmanager
def _defer_termination_signals():
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    handled = set(_handled_termination_signals())
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _bind_termination_publication(
    output_directory: Path, manifest: Mapping[str, Any]
) -> None:
    state = _ACTIVE_TERMINATION_STATE
    if state is None or os.getpid() != state.owner_pid:
        return
    state.output_directory = _absolute_path(output_directory)
    state.expected_manifest = manifest


def _mark_termination_publication_published(
    output_directory: Path,
) -> None:
    state = _ACTIVE_TERMINATION_STATE
    if state is None or os.getpid() != state.owner_pid:
        return
    output = _absolute_path(output_directory)
    if state.output_directory != output:
        raise CanonicalMotionCompileCliError(
            "published output differs from the termination cleanup binding"
        )
    state.output_identity = _directory_identity(output)


def _cleanup_terminated_publication(state: _TerminationState) -> None:
    output = state.output_directory
    expected = state.expected_manifest
    if output is None or expected is None or not os.path.lexists(output):
        return
    if output.is_symlink() or not output.is_dir():
        raise CanonicalMotionCompileCliError(
            "termination cleanup refused an unproven non-directory output"
        )
    if state.output_identity is None:
        raise CanonicalMotionCompileCliError(
            "termination cleanup has no published directory identity"
        )
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(output, directory_flags)
    try:
        metadata = os.fstat(directory_fd)
        if (metadata.st_dev, metadata.st_ino) != state.output_identity:
            raise CanonicalMotionCompileCliError(
                "termination cleanup refused a replaced output directory"
            )
        manifest_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        manifest_fd = os.open(
            cmc.BUILD_MANIFEST_NAME,
            manifest_flags,
            dir_fd=directory_fd,
        )
        with os.fdopen(manifest_fd, "rb") as stream:
            manifest_metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(manifest_metadata.st_mode):
                raise CanonicalMotionCompileCliError(
                    "terminated compiler build manifest is not a regular file"
                )
            actual = _strict_json_bytes(
                stream.read(), "terminated compiler build manifest"
            )
    finally:
        os.close(directory_fd)
    if actual != expected:
        raise CanonicalMotionCompileCliError(
            "termination cleanup refused output whose manifest does not "
            "match this job"
        )
    if _directory_identity(output) != state.output_identity:
        raise CanonicalMotionCompileCliError(
            "termination cleanup output changed before quarantine"
        )
    quarantine = output.parent / (
        f".{output.name}.terminated-{os.getpid()}-{secrets.token_hex(8)}"
    )
    if os.path.lexists(quarantine):
        raise CanonicalMotionCompileCliError(
            "termination cleanup quarantine unexpectedly exists"
        )
    os.rename(output, quarantine)
    if _directory_identity(quarantine) != state.output_identity:
        if not os.path.lexists(output):
            os.rename(quarantine, output)
        raise CanonicalMotionCompileCliError(
            "termination cleanup quarantined a replaced directory"
        )
    # Deliberately retain the inode-bound quarantine.  Any path-based
    # recursive deletion after the identity check would reopen a namespace
    # race in which another same-user process could replace ``quarantine`` and
    # make us delete unrelated data.  The requested output path is gone and
    # therefore cannot be consumed accidentally; the retained candidate
    # remains explicitly unauthorized and auditable.


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stable_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise CanonicalMotionCompileCliError(
            f"{label} must be an existing regular file: {path}"
        )
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise CanonicalMotionCompileCliError(
            f"{label} changed while it was being read: {path}"
        )
    return payload


def _read_stable_regular_nofollow(path: Path, label: str) -> bytes:
    """Read one exact regular-file inode without ever following a symlink."""

    try:
        path_before = path.lstat()
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot inspect {label} {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(path_before.st_mode):
        raise CanonicalMotionCompileCliError(
            f"{label} must be a real regular file, not a symlink: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot open {label} without following symlinks {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise CanonicalMotionCompileCliError(
                f"{label} changed before it could be read: {path}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"{label} changed after it was read {path}: {exc}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if (
        before_identity != after_identity
        or after_identity != path_identity
        or len(payload) != after.st_size
    ):
        raise CanonicalMotionCompileCliError(
            f"{label} changed while it was being read: {path}"
        )
    return payload


def _reject_json_constant(value: str) -> None:
    raise CanonicalMotionCompileCliError(
        f"JSON contains forbidden non-finite constant {value!r}"
    )


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalMotionCompileCliError(
                f"JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_json_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CanonicalMotionCompileCliError(
            f"{label} contains a non-finite number"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{label}[{index}]")


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except CanonicalMotionCompileCliError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot parse {label} as strict UTF-8 JSON: {exc}"
        ) from exc
    _assert_json_finite(parsed, label)
    return parsed


def _strict_json_file(path: Path, label: str) -> tuple[Any, str]:
    payload = _read_stable_file(path, label)
    return _strict_json_bytes(payload, label), _sha256_bytes(payload)


def _strict_json_regular_nofollow(
    path: Path, label: str
) -> tuple[Any, str]:
    payload = _read_stable_regular_nofollow(path, label)
    return _strict_json_bytes(payload, label), _sha256_bytes(payload)


def _expected_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CanonicalMotionCompileCliError(
            f"{label} must be a 64-character lowercase SHA-256"
        )
    return value


def _finite_vector(
    value: Any,
    length: int,
    label: str,
    *,
    strictly_positive: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    scalar_values = np.asarray(value, dtype=object).reshape(-1)
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, numbers.Real)
        for item in scalar_values
    ):
        raise CanonicalMotionCompileCliError(
            f"{label} must contain only JSON numbers"
        )
    if np.iscomplexobj(raw):
        raise CanonicalMotionCompileCliError(f"{label} must be real-valued")
    try:
        result = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalMotionCompileCliError(
            f"{label} must be a numeric vector"
        ) from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise CanonicalMotionCompileCliError(
            f"{label} must be finite shape ({length},), got {result.shape}"
        )
    if strictly_positive and np.any(result <= 0.0):
        raise CanonicalMotionCompileCliError(
            f"{label} must contain strictly positive values"
        )
    return np.ascontiguousarray(result)


def _finite_scalar(
    value: Any,
    label: str,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise CanonicalMotionCompileCliError(f"{label} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalMotionCompileCliError(
            f"{label} must be a real number"
        ) from exc
    if not math.isfinite(result):
        raise CanonicalMotionCompileCliError(f"{label} must be finite")
    if strictly_positive and result <= 0.0:
        raise CanonicalMotionCompileCliError(f"{label} must be positive")
    if nonnegative and result < 0.0:
        raise CanonicalMotionCompileCliError(f"{label} must be nonnegative")
    return result


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CanonicalMotionCompileCliError(
            f"{label} must be an integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise CanonicalMotionCompileCliError(
            f"{label} must be an integer <= {maximum}"
        )
    return value


def _load_acceleration_receipt(
    path: Path, expected_sha256: str
) -> _AccelerationReceipt:
    raw, actual_sha256 = _strict_json_file(
        path, "joint acceleration receipt"
    )
    expected = _expected_sha256(
        expected_sha256, "joint acceleration receipt SHA-256"
    )
    if actual_sha256 != expected:
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt SHA-256 mismatch: "
            f"expected {expected}, got {actual_sha256}"
        )
    if not isinstance(raw, Mapping):
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt must be a JSON object"
        )
    actual_keys = frozenset(str(key) for key in raw)
    if actual_keys != _ACCELERATION_RECEIPT_KEYS:
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt keys changed; "
            f"missing={sorted(_ACCELERATION_RECEIPT_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _ACCELERATION_RECEIPT_KEYS)}"
        )
    names = raw["joint_names"]
    if (
        not isinstance(names, list)
        or tuple(names) != tuple(RUNTIME_JOINT_NAMES)
    ):
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt joint_names must equal the exact "
            "31-joint runtime order"
        )
    acceleration = _finite_vector(
        raw["acceleration_rad_s2"],
        31,
        "joint acceleration receipt acceleration_rad_s2",
        strictly_positive=True,
    )
    if raw["all_positive"] is not True:
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt all_positive must be exactly true"
        )
    limiting = raw["limiting_source_frame"]
    if (
        not isinstance(limiting, list)
        or len(limiting) != 31
        or any(not isinstance(item, str) or not item.strip() for item in limiting)
    ):
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt limiting_source_frame must contain "
            "31 non-empty strings"
        )
    _finite_vector(
        raw["minimum_effort_margin_nm"],
        31,
        "joint acceleration receipt minimum_effort_margin_nm",
        strictly_positive=True,
    )
    method = raw["method"]
    if not isinstance(method, str) or not method.strip():
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt method must be a non-empty string"
        )
    return _AccelerationReceipt(
        path=path,
        sha256=actual_sha256,
        acceleration_rad_s2=acceleration,
        method=method,
    )


def _model_bindings(
    recipe: CanonicalMotionRecipe,
) -> Mapping[str, Mapping[str, str]]:
    if (
        frozenset(recipe.model_paths) != frozenset(_MODEL_KEYS)
        or frozenset(recipe.model_hashes) != frozenset(_MODEL_KEYS)
    ):
        raise CanonicalMotionCompileCliError(
            "recipe model binding keys must be exactly mjcf, urdf, body_order"
        )
    bindings: dict[str, Mapping[str, str]] = {}
    for name in _MODEL_KEYS:
        path = _absolute_path(recipe.model_paths[name])
        expected = _expected_sha256(
            recipe.model_hashes[name], f"{name} model SHA-256"
        )
        actual = _sha256_bytes(_read_stable_file(path, f"{name} model"))
        if actual != expected:
            raise CanonicalMotionCompileCliError(
                f"{name} model SHA-256 mismatch: expected {expected}, got {actual}"
            )
        bindings[name] = {"path": str(path), "sha256": actual}
    return bindings


def _verify_unchanged_job_inputs(job: _ValidatedJob) -> None:
    if _sha256_file(job.recipe_path) != job.recipe_sha256:
        raise CanonicalMotionCompileCliError(
            "recipe changed after validation"
        )
    if (
        _sha256_file(job.acceleration_receipt.path)
        != job.acceleration_receipt.sha256
    ):
        raise CanonicalMotionCompileCliError(
            "joint acceleration receipt changed after validation"
        )
    for name, binding in job.model_bindings.items():
        if _sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise CanonicalMotionCompileCliError(
                f"{name} model changed after validation"
            )
    if job.append_only_base is not None:
        base = job.append_only_base
        if _sha256_file(base.recipe_path) != base.recipe_sha256:
            raise CanonicalMotionCompileCliError(
                "append-base recipe changed after validation"
            )
        manifest, manifest_sha = _strict_json_regular_nofollow(
            base.manifest_path, "append-base build manifest"
        )
        if (
            manifest_sha != base.manifest_sha256
            or manifest != base.manifest
        ):
            raise CanonicalMotionCompileCliError(
                "append-base build manifest changed after validation"
            )
        current_outputs = tuple(
            _validate_published_outputs(
                base.output_directory,
                manifest,
                optional_names=frozenset({RUN_RECEIPT_NAME}),
            )
        )
        if current_outputs != base.published_outputs:
            raise CanonicalMotionCompileCliError(
                "append-base output bundle changed after validation"
            )
    cli_path = Path(__file__).resolve()
    if _sha256_file(cli_path) != job.cli_sha256:
        raise CanonicalMotionCompileCliError(
            "compiler CLI changed while the job was running"
        )


def _parser() -> argparse.ArgumentParser:
    parser = _FailClosedArgumentParser(
        description=(
            "Compile one canonical candidate matrix. Recipes that append after "
            "the canonical five must bind an exact existing 5x2 base bundle; "
            "only the appended motions are compiled."
        )
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append-base-recipe")
    parser.add_argument("--append-base-recipe-sha256")
    parser.add_argument("--append-base-manifest")
    parser.add_argument("--append-base-manifest-sha256")
    parser.add_argument(
        "--station-center-shift-xy-m",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
    )
    parser.add_argument("--joint-acceleration-receipt", required=True)
    parser.add_argument(
        "--joint-acceleration-receipt-sha256", required=True
    )
    parser.add_argument(
        "--full-root-position-lower",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    parser.add_argument(
        "--full-root-position-upper",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    parser.add_argument(
        "--full-root-velocity",
        type=float,
        nargs=6,
        required=True,
        metavar=("VX", "VY", "VZ", "VRX", "VRY", "VRZ"),
    )
    parser.add_argument(
        "--full-root-acceleration",
        type=float,
        nargs=6,
        required=True,
        metavar=("AX", "AY", "AZ", "ARX", "ARY", "ARZ"),
    )
    parser.add_argument(
        "--s0-full-grounding-offset-m", type=float, required=True
    )
    parser.add_argument(
        "--samples-per-scaled-unit", type=float, required=True
    )
    parser.add_argument(
        "--min-connector-intervals", type=int, required=True
    )
    parser.add_argument("--min-core-intervals", type=int, required=True)
    parser.add_argument("--grid-subdivisions", type=int, required=True)
    parser.add_argument("--search-workers", type=int, required=True)
    parser.add_argument(
        "--search-parallel-backend",
        choices=("thread", "process"),
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every input and hash without compiling or creating output",
    )
    return parser


def _append_base_values(args: argparse.Namespace) -> tuple[Any, ...]:
    return (
        args.append_base_recipe,
        args.append_base_recipe_sha256,
        args.append_base_manifest,
        args.append_base_manifest_sha256,
    )


def _source_identity(source: Any) -> tuple[Any, ...]:
    return (
        source.motion_id,
        _absolute_path(source.path),
        source.sha256,
        source.human_role,
        source.face_manifold,
        source.scope_overrides,
    )


def _load_append_only_base(
    args: argparse.Namespace,
    *,
    recipe: CanonicalMotionRecipe,
    repo_root: Path,
) -> _AppendOnlyBase | None:
    values = _append_base_values(args)
    supplied = tuple(value is not None for value in values)
    is_append = len(recipe.sources) > _CANONICAL_BASE_COUNT
    if not is_append:
        if any(supplied):
            raise CanonicalMotionCompileCliError(
                "append-base inputs are forbidden for a non-append recipe"
            )
        return None
    if not all(supplied):
        raise CanonicalMotionCompileCliError(
            "an append recipe must bind all four append-base inputs: recipe "
            "path/SHA-256 and BUILD_MANIFEST path/SHA-256"
        )

    base_recipe_path = _absolute_path(args.append_base_recipe)
    expected_base_recipe_sha = _expected_sha256(
        args.append_base_recipe_sha256,
        "append-base recipe SHA-256",
    )
    actual_base_recipe_sha = _sha256_bytes(
        _read_stable_regular_nofollow(
            base_recipe_path, "append-base recipe"
        )
    )
    if actual_base_recipe_sha != expected_base_recipe_sha:
        raise CanonicalMotionCompileCliError(
            "append-base recipe SHA-256 mismatch: expected "
            f"{expected_base_recipe_sha}, got {actual_base_recipe_sha}"
        )
    try:
        base_recipe = load_canonical_motion_recipe(
            base_recipe_path, repo_root=repo_root
        )
    except Exception as exc:
        raise CanonicalMotionCompileCliError(
            f"append-base recipe validation failed: {exc}"
        ) from exc
    if _sha256_file(base_recipe_path) != actual_base_recipe_sha:
        raise CanonicalMotionCompileCliError(
            "append-base recipe changed during validation"
        )
    if len(base_recipe.sources) != _CANONICAL_BASE_COUNT:
        raise CanonicalMotionCompileCliError(
            "append-base recipe must contain exactly the canonical five"
        )

    base_identities = tuple(_source_identity(row) for row in base_recipe.sources)
    append_prefix = tuple(
        _source_identity(row)
        for row in recipe.sources[:_CANONICAL_BASE_COUNT]
    )
    if append_prefix != base_identities:
        raise CanonicalMotionCompileCliError(
            "append recipe canonical-five source identities differ from the "
            "bound base recipe"
        )
    for source in base_recipe.sources:
        motion_id = str(source.motion_id)
        if (
            recipe.marker_semantics.row(motion_id)
            != base_recipe.marker_semantics.row(motion_id)
        ):
            raise CanonicalMotionCompileCliError(
                f"append recipe marker row {motion_id!r} differs from the "
                "bound base recipe"
            )
    if (
        recipe.ready.sha256 != base_recipe.ready.sha256
        or dict(recipe.model_hashes) != dict(base_recipe.model_hashes)
    ):
        raise CanonicalMotionCompileCliError(
            "append recipe ready/model identities differ from the bound base"
        )
    for key in (
        "frame_id",
        "canonical_ready",
        "model_contract",
        "scope_contract",
        "time_law",
        "entry_exit_search",
        "post_build_gates",
    ):
        if recipe.raw.get(key) != base_recipe.raw.get(key):
            raise CanonicalMotionCompileCliError(
                f"append recipe shared compiler contract {key!r} differs "
                "from the bound base recipe"
            )

    manifest_path = _absolute_path(args.append_base_manifest)
    if manifest_path.name != cmc.BUILD_MANIFEST_NAME:
        raise CanonicalMotionCompileCliError(
            "append-base manifest must be named "
            f"{cmc.BUILD_MANIFEST_NAME}"
        )
    manifest, actual_manifest_sha = _strict_json_regular_nofollow(
        manifest_path, "append-base build manifest"
    )
    expected_manifest_sha = _expected_sha256(
        args.append_base_manifest_sha256,
        "append-base build manifest SHA-256",
    )
    if actual_manifest_sha != expected_manifest_sha:
        raise CanonicalMotionCompileCliError(
            "append-base build manifest SHA-256 mismatch: expected "
            f"{expected_manifest_sha}, got {actual_manifest_sha}"
        )
    compiler = (
        manifest.get("compiler")
        if isinstance(manifest, Mapping)
        else None
    )
    if not isinstance(compiler, Mapping):
        raise CanonicalMotionCompileCliError(
            "append-base build manifest is missing its compiler binding"
        )
    base_compiler_sha = _expected_sha256(
        compiler.get("sha256"),
        "append-base compiler SHA-256",
    )
    _validate_candidate_manifest(
        manifest,
        recipe=base_recipe,
        recipe_sha256=actual_base_recipe_sha,
        compiler_sha256=base_compiler_sha,
    )
    output_directory = manifest_path.parent
    published_outputs = tuple(
        _validate_published_outputs(
            output_directory,
            manifest,
            optional_names=frozenset({RUN_RECEIPT_NAME}),
        )
    )
    return _AppendOnlyBase(
        recipe_path=base_recipe_path,
        recipe_sha256=actual_base_recipe_sha,
        recipe=base_recipe,
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha,
        manifest=manifest,
        output_directory=output_directory,
        published_outputs=published_outputs,
    )


def _validate_job(args: argparse.Namespace) -> _ValidatedJob:
    recipe_path = _absolute_path(args.recipe)
    repo_root = _absolute_path(args.repo_root)
    output = _absolute_path(args.output)
    acceleration_path = _absolute_path(args.joint_acceleration_receipt)
    if not repo_root.is_dir():
        raise CanonicalMotionCompileCliError(
            f"repo-root must be an existing directory: {repo_root}"
        )
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output}"
        )

    # Reject duplicate keys and every form of non-finite JSON before delegating
    # the complete recipe contract to the authoritative loader.
    raw_recipe, recipe_sha256 = _strict_json_file(
        recipe_path, "canonical motion recipe"
    )
    if not isinstance(raw_recipe, Mapping):
        raise CanonicalMotionCompileCliError(
            "canonical motion recipe must be a JSON object"
        )
    try:
        recipe = load_canonical_motion_recipe(
            recipe_path, repo_root=repo_root
        )
    except Exception as exc:
        raise CanonicalMotionCompileCliError(
            f"canonical motion recipe validation failed: {exc}"
        ) from exc
    if _sha256_file(recipe_path) != recipe_sha256:
        raise CanonicalMotionCompileCliError(
            "canonical motion recipe changed during validation"
        )
    append_only_base = _load_append_only_base(
        args,
        recipe=recipe,
        repo_root=repo_root,
    )
    station_shift: tuple[float, float] | None = None
    if args.station_center_shift_xy_m is not None:
        station_raw = _finite_vector(
            args.station_center_shift_xy_m,
            2,
            "station center shift xy",
        )
        station_shift = (float(station_raw[0]), float(station_raw[1]))
    if append_only_base is not None and station_shift is None:
        raise CanonicalMotionCompileCliError(
            "an append-only compile must explicitly bind "
            "--station-center-shift-xy-m"
        )
    if append_only_base is None and station_shift is not None:
        raise CanonicalMotionCompileCliError(
            "station center shift is only valid for an append-only compile"
        )
    acceleration_receipt = _load_acceleration_receipt(
        acceleration_path,
        args.joint_acceleration_receipt_sha256,
    )

    lower = _finite_vector(
        args.full_root_position_lower, 6, "full-root position lower"
    )
    upper = _finite_vector(
        args.full_root_position_upper, 6, "full-root position upper"
    )
    if np.any(lower >= upper):
        raise CanonicalMotionCompileCliError(
            "every full-root position lower bound must be below its upper bound"
        )
    velocity = _finite_vector(
        args.full_root_velocity,
        6,
        "full-root velocity",
        strictly_positive=True,
    )
    root_acceleration = _finite_vector(
        args.full_root_acceleration,
        6,
        "full-root acceleration",
        strictly_positive=True,
    )
    grounding = _finite_scalar(
        args.s0_full_grounding_offset_m,
        "S0 full grounding offset",
        nonnegative=True,
    )
    samples = _finite_scalar(
        args.samples_per_scaled_unit,
        "samples per scaled unit",
        strictly_positive=True,
    )
    connector_intervals = _integer(
        args.min_connector_intervals,
        "minimum connector intervals",
        minimum=5,
    )
    core_intervals = _integer(
        args.min_core_intervals,
        "minimum core intervals",
        minimum=5,
    )
    grid_subdivisions = _integer(
        args.grid_subdivisions, "grid subdivisions", minimum=2
    )
    search_workers = _integer(
        args.search_workers, "search workers", minimum=1, maximum=64
    )
    backend = args.search_parallel_backend
    if backend not in ("thread", "process"):
        raise CanonicalMotionCompileCliError(
            "search parallel backend must be thread or process"
        )
    option_fields = getattr(cmc.CompilerOptions, "__dataclass_fields__", {})
    # Fields this formal CLI maps from its own arguments.
    mapped_option_fields = frozenset(
        {
            "joint_acceleration_limits_rad_s2",
            "full_root_limits",
            "s0_full_grounding_offset_m",
            "samples_per_scaled_unit",
            "min_connector_intervals",
            "min_core_intervals",
            "grid_subdivisions",
            "search_workers",
            "search_parallel_backend",
            "face_config",
            "face_active_candidate_seeds",
        }
    )
    # Probe-grade knobs the formal CLI deliberately does NOT expose.  They stay
    # at their dataclass defaults here, and that is asserted below rather than
    # assumed: a probe knob silently reaching a formal build is exactly the
    # class of error this contract check exists to stop.
    probe_only_option_fields = frozenset(
        {
            "probe_entry_band",
            "probe_exit_band",
            "probe_exact_pointwise_caps",
            "probe_source_smoothing_tolerance_rad",
            "synthetic_face_solve_span_extension",
        }
    )
    expected_option_fields = mapped_option_fields | probe_only_option_fields
    if frozenset(option_fields) != expected_option_fields:
        raise CanonicalMotionCompileCliError(
            "canonical_motion_compiler CompilerOptions contract changed; "
            f"expected={sorted(expected_option_fields)}, "
            f"actual={sorted(option_fields)}"
        )
    root_limits = cmc.FullRootPathLimits(
        position_lower=lower,
        position_upper=upper,
        velocity=velocity,
        acceleration=root_acceleration,
    )
    options = cmc.CompilerOptions(
        joint_acceleration_limits_rad_s2=(
            acceleration_receipt.acceleration_rad_s2
        ),
        full_root_limits=root_limits,
        s0_full_grounding_offset_m=grounding,
        samples_per_scaled_unit=samples,
        min_connector_intervals=connector_intervals,
        min_core_intervals=core_intervals,
        grid_subdivisions=grid_subdivisions,
        search_workers=search_workers,
        search_parallel_backend=backend,
        face_config=None,
        face_active_candidate_seeds=(),
    )
    for name in sorted(probe_only_option_fields):
        default = cmc.CompilerOptions.__dataclass_fields__[name].default
        if getattr(options, name) != default:
            raise CanonicalMotionCompileCliError(
                f"formal compile left probe-grade option {name!r} at "
                f"{getattr(options, name)!r} instead of its default {default!r}"
            )
    model_bindings = _model_bindings(recipe)
    normalized_args = {
        "recipe": str(recipe_path),
        "repo_root": str(repo_root),
        "output": str(output),
        "append_base_recipe": (
            str(append_only_base.recipe_path)
            if append_only_base is not None
            else None
        ),
        "append_base_recipe_sha256": (
            append_only_base.recipe_sha256
            if append_only_base is not None
            else None
        ),
        "append_base_manifest": (
            str(append_only_base.manifest_path)
            if append_only_base is not None
            else None
        ),
        "append_base_manifest_sha256": (
            append_only_base.manifest_sha256
            if append_only_base is not None
            else None
        ),
        "station_center_shift_xy_m": (
            list(station_shift) if station_shift is not None else None
        ),
        "joint_acceleration_receipt": str(acceleration_path),
        "joint_acceleration_receipt_sha256": (
            acceleration_receipt.sha256
        ),
        "full_root_position_lower": lower.tolist(),
        "full_root_position_upper": upper.tolist(),
        "full_root_velocity": velocity.tolist(),
        "full_root_acceleration": root_acceleration.tolist(),
        "s0_full_grounding_offset_m": grounding,
        "samples_per_scaled_unit": samples,
        "min_connector_intervals": connector_intervals,
        "min_core_intervals": core_intervals,
        "grid_subdivisions": grid_subdivisions,
        "search_workers": search_workers,
        "search_parallel_backend": backend,
        "face_config": None,
        "face_active_candidate_seeds": [],
        "dry_run": bool(args.dry_run),
    }
    cli_path = Path(__file__).resolve()
    return _ValidatedJob(
        recipe_path=recipe_path,
        repo_root=repo_root,
        output_directory=output,
        recipe_sha256=recipe_sha256,
        recipe=recipe,
        acceleration_receipt=acceleration_receipt,
        options=options,
        normalized_args=normalized_args,
        model_bindings=model_bindings,
        cli_sha256=_sha256_file(cli_path),
        append_only_base=append_only_base,
        station_center_shift_xy_m=station_shift,
    )


def _assert_candidate_authorizations(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).endswith("_authorized") and item is not False:
                raise CanonicalMotionCompileCliError(
                    f"{label} authorization {key!r} must be false"
                )
            _assert_candidate_authorizations(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_candidate_authorizations(
                item, f"{label}[{index}]"
            )


def _validate_schema2_receipt_pair(
    schema: Any,
    report: Any,
    *,
    output_sha256: str,
    label: str,
) -> None:
    if (
        not isinstance(schema, Mapping)
        or schema.get("publication_class") != PUBLICATION_CLASS
        or schema.get("training_authorized") is not False
        or schema.get("build_verdict")
        != "PASS_COMPILER_CANDIDATE_ONLY"
        or schema.get("tool_id") != "canonical_schema2_builder_v1"
    ):
        raise CanonicalMotionCompileCliError(
            f"{label} manifest weakened candidate-only publication"
        )
    if (
        not isinstance(report, Mapping)
        or report.get("publication_class") != PUBLICATION_CLASS
        or report.get("training_authorized") is not False
        or report.get("status") != "PASS"
        or report.get("tool_id") != schema.get("tool_id")
    ):
        raise CanonicalMotionCompileCliError(
            f"{label} report weakened candidate-only publication"
        )
    _assert_candidate_authorizations(schema, f"{label} manifest")
    _assert_candidate_authorizations(report, f"{label} report")

    hashes = schema.get("hashes")
    report_hashes = report.get("hashes")
    if (
        not isinstance(hashes, Mapping)
        or frozenset(str(key) for key in hashes) != _SCHEMA2_HASH_KEYS
        or not isinstance(report_hashes, Mapping)
        or frozenset(str(key) for key in report_hashes)
        != _SCHEMA2_HASH_KEYS
    ):
        raise CanonicalMotionCompileCliError(
            f"{label} schema-2 hash receipt keys changed"
        )
    normalized_hashes = {
        key: _expected_sha256(
            hashes.get(key), f"{label} manifest hash {key}"
        )
        for key in sorted(_SCHEMA2_HASH_KEYS)
    }
    normalized_report_hashes = {
        key: _expected_sha256(
            report_hashes.get(key), f"{label} report hash {key}"
        )
        for key in sorted(_SCHEMA2_HASH_KEYS)
    }
    if normalized_report_hashes != normalized_hashes:
        raise CanonicalMotionCompileCliError(
            f"{label} manifest/report hash receipts disagree"
        )
    if normalized_hashes["output_npz_sha256"] != output_sha256:
        raise CanonicalMotionCompileCliError(
            f"{label} output NPZ SHA-256 binding changed"
        )

    files = schema.get("files")
    tool_value = (
        files.get("tool_path")
        if isinstance(files, Mapping)
        else None
    )
    if not isinstance(tool_value, str) or not tool_value:
        raise CanonicalMotionCompileCliError(
            f"{label} manifest is missing the schema-2 builder tool path"
        )
    tool_path = Path(tool_value)
    expected_tool_path = Path(
        cmc.build_schema2_candidate.__code__.co_filename
    ).resolve()
    if not tool_path.is_absolute() or tool_path != expected_tool_path:
        raise CanonicalMotionCompileCliError(
            f"{label} schema-2 builder tool path changed"
        )
    actual_tool_sha256 = _sha256_bytes(
        _read_stable_regular_nofollow(
            tool_path, f"{label} schema-2 builder tool"
        )
    )
    if actual_tool_sha256 != normalized_hashes["tool_sha256"]:
        raise CanonicalMotionCompileCliError(
            f"{label} schema-2 builder tool SHA-256 does not bind loaded code"
        )


def _validate_candidate_manifest(
    manifest: Any,
    *,
    recipe: CanonicalMotionRecipe,
    recipe_sha256: str,
    compiler_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise CanonicalMotionCompileCliError(
            "compiler build manifest must be a JSON object"
        )
    if manifest.get("publication_class") != PUBLICATION_CLASS:
        raise CanonicalMotionCompileCliError(
            "compiler build manifest publication class was weakened"
        )
    for key, value in manifest.items():
        if str(key).endswith("_authorized") and value is not False:
            raise CanonicalMotionCompileCliError(
                f"compiler build manifest authorization {key!r} must be false"
            )
    if (
        manifest.get("training_authorized") is not False
        or manifest.get("hardware_authorized") is not False
    ):
        raise CanonicalMotionCompileCliError(
            "compiler build manifest must keep training and hardware false"
        )
    if manifest.get("build_verdict") != "PASS_COMPILER_CANDIDATE_ONLY":
        raise CanonicalMotionCompileCliError(
            "compiler build manifest verdict is not candidate-only PASS"
        )
    manifest_recipe = manifest.get("recipe")
    if (
        not isinstance(manifest_recipe, Mapping)
        or manifest_recipe.get("sha256") != recipe_sha256
    ):
        raise CanonicalMotionCompileCliError(
            "compiler build manifest recipe SHA-256 does not match this job"
        )
    compiler = manifest.get("compiler")
    if (
        not isinstance(compiler, Mapping)
        or compiler.get("sha256") != compiler_sha256
    ):
        raise CanonicalMotionCompileCliError(
            "compiler build manifest compiler SHA-256 does not match loaded code"
        )
    matrix = manifest.get("output_matrix")
    outputs = manifest.get("outputs")
    expected_motion_ids = tuple(
        str(source.motion_id) for source in recipe.sources
    )
    expected_scopes = ("upper", "full")
    expected_pairs = tuple(
        (motion_id, scope)
        for motion_id in expected_motion_ids
        for scope in expected_scopes
    )
    expected_count = len(expected_pairs)
    if (
        not isinstance(matrix, Mapping)
        or tuple(matrix.get("motion_ids", ())) != expected_motion_ids
        or tuple(matrix.get("scopes", ())) != expected_scopes
        or matrix.get("candidate_count") != expected_count
        or not isinstance(outputs, list)
        or len(outputs) != expected_count
    ):
        raise CanonicalMotionCompileCliError(
            "compiler build manifest must contain the complete declared "
            f"{len(expected_motion_ids)}x2 matrix"
        )
    actual_pairs: list[tuple[str, str]] = []
    filenames: set[str] = set()
    for index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            raise CanonicalMotionCompileCliError(
                f"compiler output {index} must be an object"
            )
        actual_pairs.append(
            (str(output.get("motion_id")), str(output.get("scope")))
        )
        filename = output.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in filenames
        ):
            raise CanonicalMotionCompileCliError(
                f"compiler output {index} has an unsafe or duplicate filename"
            )
        filenames.add(filename)
        output_sha = _expected_sha256(
            output.get("output_npz_sha256"),
            f"compiler output {index} NPZ SHA-256",
        )
        schema = output.get("schema2_manifest")
        report = output.get("schema2_report")
        _validate_schema2_receipt_pair(
            schema,
            report,
            output_sha256=output_sha,
            label=f"compiler output {index}",
        )
    if tuple(actual_pairs) != expected_pairs:
        raise CanonicalMotionCompileCliError(
            "compiler outputs are not the exact ordered declared matrix: "
            f"{actual_pairs}"
        )
    gates = manifest.get("post_build_gates")
    expected_gates = tuple(recipe.raw["post_build_gates"])
    if (
        not isinstance(gates, list)
        or tuple(
            gate.get("name") if isinstance(gate, Mapping) else None
            for gate in gates
        )
        != expected_gates
        or any(
            frozenset(gate) != frozenset({"name", "status"})
            or gate.get("status") != "pending"
            for gate in gates
            if isinstance(gate, Mapping)
        )
    ):
        raise CanonicalMotionCompileCliError(
            "compiler post-build gates changed or are not all pending"
        )
    return manifest


def _append_only_composition(
    job: _ValidatedJob,
) -> Mapping[str, Any] | None:
    base = job.append_only_base
    if base is None:
        return None
    appended_ids = [
        str(source.motion_id)
        for source in job.recipe.sources[_CANONICAL_BASE_COUNT:]
    ]
    base_outputs = [
        {
            "motion_id": row["motion_id"],
            "scope": row["scope"],
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in base.published_outputs
    ]
    return {
        "mode": "reuse_exact_base_outputs_compile_appended_only",
        "base_outputs_rebuilt": False,
        "base_recipe": {
            "path": str(base.recipe_path),
            "sha256": base.recipe_sha256,
        },
        "base_build_manifest": {
            "path": str(base.manifest_path),
            "sha256": base.manifest_sha256,
        },
        "base_output_matrix": dict(base.manifest["output_matrix"]),
        "base_outputs": base_outputs,
        "appended_motion_ids": appended_ids,
        "appended_scopes": ["upper", "full"],
        "station_center_shift_xy_m": list(
            job.station_center_shift_xy_m or ()
        ),
        "composed_candidate_count": (
            len(base_outputs) + 2 * len(appended_ids)
        ),
    }


def _compile_recipe_for_job(job: _ValidatedJob) -> CanonicalMotionRecipe:
    if job.append_only_base is None:
        return job.recipe
    appended_sources = tuple(
        job.recipe.sources[_CANONICAL_BASE_COUNT:]
    )
    if not appended_sources:
        raise CanonicalMotionCompileCliError(
            "append-only compile has no appended motions"
        )
    raw = dict(job.recipe.raw)
    raw["required_output_matrix"] = {
        "motion_ids": [
            str(source.motion_id) for source in appended_sources
        ],
        "scopes": ["upper", "full"],
        "candidate_count": 2 * len(appended_sources),
    }
    if isinstance(job.recipe, CanonicalMotionRecipe):
        return replace(
            job.recipe,
            raw=MappingProxyType(raw),
            sources=appended_sources,
        )
    # Test doubles used by this module's hermetic CLI tests deliberately avoid
    # loading motion arrays. Production loader output is always the dataclass.
    return type(job.recipe)(
        **{
            **vars(job.recipe),
            "raw": MappingProxyType(raw),
            "sources": appended_sources,
        }
    )


def _attach_and_validate_composition(
    library: cmc.CompiledLibrary,
    *,
    job: _ValidatedJob,
    build_recipe: CanonicalMotionRecipe,
) -> cmc.CompiledLibrary:
    expected = _append_only_composition(job)
    actual = library.manifest.get("append_only_composition")
    if expected is None:
        if actual is not None:
            raise CanonicalMotionCompileCliError(
                "non-append compiler manifest unexpectedly declares an "
                "append-only composition"
            )
        return library
    if actual is not None:
        raise CanonicalMotionCompileCliError(
            "compiler, rather than the strict CLI, injected an append-only "
            "composition binding"
        )
    manifest = dict(library.manifest)
    manifest["append_only_composition"] = dict(expected)
    manifest["station_center_shift_xy_m"] = list(
        job.station_center_shift_xy_m or ()
    )
    return cmc.CompiledLibrary(
        recipe=build_recipe,
        motions=tuple(library.motions),
        manifest=MappingProxyType(manifest),
    )


def _validate_manifest_composition(
    manifest: Mapping[str, Any],
    *,
    job: _ValidatedJob,
) -> None:
    expected = _append_only_composition(job)
    actual = manifest.get("append_only_composition")
    if expected is None:
        if (
            actual is not None
            or manifest.get("station_center_shift_xy_m") is not None
        ):
            raise CanonicalMotionCompileCliError(
                "non-append build manifest declares append-only composition"
            )
        return
    if actual != expected:
        raise CanonicalMotionCompileCliError(
            "append-only build manifest does not bind the exact reused base "
            "bundle and appended motion set"
        )
    if manifest.get("station_center_shift_xy_m") != list(
        job.station_center_shift_xy_m or ()
    ):
        raise CanonicalMotionCompileCliError(
            "append-only build manifest station_center_shift_xy_m changed"
        )


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot inspect published output directory {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CanonicalMotionCompileCliError(
            f"published output must be a real directory, not a symlink: {path}"
        )
    return metadata.st_dev, metadata.st_ino


def _validate_published_outputs(
    output_directory: Path,
    manifest: Mapping[str, Any],
    *,
    optional_names: frozenset[str] = frozenset(),
) -> list[Mapping[str, Any]]:
    outputs = manifest["outputs"]
    expected_names = {cmc.BUILD_MANIFEST_NAME}
    for output in outputs:
        filename = str(output["filename"])
        expected_names.update(
            {
                filename,
                f"{filename}.manifest.json",
                f"{filename}.report.json",
            }
        )
    expected_count = 1 + 3 * len(outputs)
    if len(expected_names) != expected_count:
        raise CanonicalMotionCompileCliError(
            "published output filenames collide with schema-2 sidecars"
        )
    try:
        children = list(output_directory.iterdir())
    except OSError as exc:
        raise CanonicalMotionCompileCliError(
            f"cannot enumerate published output directory: {exc}"
        ) from exc
    actual_names = [child.name for child in children]
    actual_name_set = frozenset(actual_names)
    if (
        len(actual_names) != len(actual_name_set)
        or not frozenset(expected_names).issubset(actual_name_set)
        or not actual_name_set.issubset(
            frozenset(expected_names) | optional_names
        )
    ):
        raise CanonicalMotionCompileCliError(
            "published output directory contains missing, duplicate, or "
            f"unexpected files: {sorted(actual_names)}"
        )
    verified: list[Mapping[str, Any]] = []
    for index, output in enumerate(outputs):
        path = output_directory / output["filename"]
        actual_sha = _sha256_bytes(
            _read_stable_regular_nofollow(
                path, f"published output {index} NPZ"
            )
        )
        expected_sha = _expected_sha256(
            output["output_npz_sha256"],
            f"published output {index} NPZ SHA-256",
        )
        if actual_sha != expected_sha:
            raise CanonicalMotionCompileCliError(
                f"published output {index} SHA-256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        manifest_sidecar_path = path.with_suffix(
            path.suffix + ".manifest.json"
        )
        report_sidecar_path = path.with_suffix(
            path.suffix + ".report.json"
        )
        schema_manifest, schema_manifest_sha = (
            _strict_json_regular_nofollow(
                manifest_sidecar_path,
                f"published output {index} schema-2 manifest sidecar",
            )
        )
        schema_report, schema_report_sha = (
            _strict_json_regular_nofollow(
                report_sidecar_path,
                f"published output {index} schema-2 report sidecar",
            )
        )
        if schema_manifest != output.get("schema2_manifest"):
            raise CanonicalMotionCompileCliError(
                f"published output {index} manifest sidecar disagrees "
                "with BUILD_MANIFEST"
            )
        if schema_report != output.get("schema2_report"):
            raise CanonicalMotionCompileCliError(
                f"published output {index} report sidecar disagrees "
                "with BUILD_MANIFEST"
            )
        _validate_schema2_receipt_pair(
            schema_manifest,
            schema_report,
            output_sha256=actual_sha,
            label=f"published output {index} sidecars",
        )
        verified.append(
            {
                "motion_id": str(output["motion_id"]),
                "scope": str(output["scope"]),
                "path": str(path),
                "sha256": actual_sha,
                "manifest_sidecar": {
                    "path": str(manifest_sidecar_path),
                    "sha256": schema_manifest_sha,
                },
                "report_sidecar": {
                    "path": str(report_sidecar_path),
                    "sha256": schema_report_sha,
                },
            }
        )
    return verified


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalMotionCompileCliError(
            f"run receipt is not strict JSON: {exc}"
        ) from exc


def _write_run_receipt_atomic(
    output_directory: Path,
    receipt: Mapping[str, Any],
    *,
    expected_directory_identity: tuple[int, int],
) -> Path:
    final = output_directory / RUN_RECEIPT_NAME
    if os.path.lexists(final):
        raise FileExistsError(f"refusing to overwrite run receipt: {final}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(output_directory, directory_flags)
    metadata = os.fstat(directory_fd)
    actual_identity = (metadata.st_dev, metadata.st_ino)
    if actual_identity != expected_directory_identity:
        os.close(directory_fd)
        raise CanonicalMotionCompileCliError(
            "published output directory identity changed before receipt write"
        )
    temporary_name = (
        f".{RUN_RECEIPT_NAME}.{os.getpid()}."
        f"{secrets.token_hex(8)}.tmp"
    )
    temporary_fd: int | None = None
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        payload = _json_bytes(receipt)
        with os.fdopen(temporary_fd, "wb") as stream:
            temporary_fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary_name,
                RUN_RECEIPT_NAME,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite run receipt: {final}"
            ) from None
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except Exception:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)
    if _directory_identity(output_directory) != expected_directory_identity:
        raise CanonicalMotionCompileCliError(
            "published output directory identity changed after receipt write"
        )
    return final


def _dry_run_receipt(job: _ValidatedJob) -> Mapping[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "PASS_INPUT_VALIDATION_ONLY",
        "dry_run": True,
        "publication_class": PUBLICATION_CLASS,
        "training_authorized": False,
        "hardware_authorized": False,
        "output_created": False,
        "recipe": {
            "path": str(job.recipe_path),
            "sha256": job.recipe_sha256,
        },
        "joint_acceleration_receipt": {
            "path": str(job.acceleration_receipt.path),
            "sha256": job.acceleration_receipt.sha256,
            "method": job.acceleration_receipt.method,
        },
        "models": dict(job.model_bindings),
        "job_args": dict(job.normalized_args),
        "termination_contract": dict(_signal_contract()),
    }
    composition = _append_only_composition(job)
    if composition is not None:
        payload["append_only_composition"] = dict(composition)
    return payload


def _run_build(job: _ValidatedJob) -> Mapping[str, Any]:
    compiler_path = Path(cmc.__file__).resolve()
    compiler_sha256 = _sha256_file(compiler_path)
    _verify_unchanged_job_inputs(job)
    build_recipe = _compile_recipe_for_job(job)
    library = cmc.compile_loaded_canonical_motion_library(
        build_recipe,
        options=job.options,
    )
    library = _attach_and_validate_composition(
        library,
        job=job,
        build_recipe=build_recipe,
    )
    _verify_unchanged_job_inputs(job)
    _validate_candidate_manifest(
        library.manifest,
        recipe=build_recipe,
        recipe_sha256=job.recipe_sha256,
        compiler_sha256=compiler_sha256,
    )
    _validate_manifest_composition(library.manifest, job=job)
    _bind_termination_publication(
        job.output_directory,
        library.manifest,
    )
    # A listed termination signal must not land after the writer's atomic
    # publication but before cleanup has pinned that directory's identity.
    # Pending signals are delivered as soon as this short critical section
    # restores the caller's mask.
    with _defer_termination_signals():
        published = cmc.write_compiled_canonical_motion_library(
            library, job.output_directory
        )
        if _absolute_path(published) != job.output_directory:
            raise CanonicalMotionCompileCliError(
                "compiler writer returned a different output directory"
            )
        _mark_termination_publication_published(published)
    published_identity = _directory_identity(published)
    manifest_path = published / cmc.BUILD_MANIFEST_NAME
    if manifest_path.is_symlink():
        raise CanonicalMotionCompileCliError(
            "published build manifest may not be a symlink"
        )
    manifest, manifest_sha256 = _strict_json_file(
        manifest_path, "compiler build manifest"
    )
    _verify_unchanged_job_inputs(job)
    _validate_candidate_manifest(
        manifest,
        recipe=build_recipe,
        recipe_sha256=job.recipe_sha256,
        compiler_sha256=compiler_sha256,
    )
    _validate_manifest_composition(manifest, job=job)
    if manifest != library.manifest:
        raise CanonicalMotionCompileCliError(
            "published build manifest differs from the in-memory manifest"
        )
    published_outputs = _validate_published_outputs(published, manifest)
    if _directory_identity(published) != published_identity:
        raise CanonicalMotionCompileCliError(
            "published output directory identity changed during verification"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_COMPILER_CANDIDATE_ONLY",
        "publication_class": PUBLICATION_CLASS,
        "training_authorized": False,
        "hardware_authorized": False,
        "output_directory": str(published),
        "cli": {
            "path": str(Path(__file__).resolve()),
            "sha256": job.cli_sha256,
        },
        "compiler": {
            "path": str(compiler_path),
            "sha256": compiler_sha256,
        },
        "job_args": dict(job.normalized_args),
        "joint_acceleration_receipt": {
            "path": str(job.acceleration_receipt.path),
            "sha256": job.acceleration_receipt.sha256,
            "method": job.acceleration_receipt.method,
        },
        "recipe": {
            "path": str(job.recipe_path),
            "sha256": job.recipe_sha256,
        },
        "models": dict(job.model_bindings),
        "build_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "published_outputs": published_outputs,
        "termination_contract": dict(_signal_contract()),
    }
    composition = _append_only_composition(job)
    if composition is not None:
        payload["append_only_composition"] = dict(composition)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["run_receipt_payload_sha256"] = _sha256_bytes(canonical)
    _write_run_receipt_atomic(
        published,
        payload,
        expected_directory_identity=published_identity,
    )
    return payload


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    """Validate one CLI invocation, then optionally compile and publish it."""

    with _termination_guard() as termination_state:
        job: _ValidatedJob | None = None
        try:
            args = _parser().parse_args(argv)
            job = _validate_job(args)
            _verify_unchanged_job_inputs(job)
            if args.dry_run:
                return _dry_run_receipt(job)
            return _run_build(job)
        except CanonicalMotionCompileTerminated:
            _cleanup_terminated_publication(termination_state)
            raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(argv)
    except CanonicalMotionCompileTerminated as exc:
        print(f"TERMINATED: {exc}", file=sys.stderr, flush=True)
        return 128 + int(exc.signum)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(result, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CanonicalMotionCompileCliError",
    "CanonicalMotionCompileTerminated",
    "PUBLICATION_CLASS",
    "RUN_RECEIPT_NAME",
    "main",
    "run",
]
