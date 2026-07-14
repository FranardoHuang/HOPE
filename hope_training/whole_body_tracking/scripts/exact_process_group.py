#!/usr/bin/env python3
"""Bind and signal one Linux process group without trusting a stale PGID.

The launcher records a stable ``/proc`` identity before TERM.  Later checks
only accept the exact members present in that snapshot; a reused PID, changed
start time, or newly joined member fails closed before KILL.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import stat
import sys
from typing import Callable


class IdentityError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class Identity:
    pid: int
    pgid: int
    starttime_ticks: int


def _parse_stat(text: str, expected_pid: int) -> Identity:
    close = text.rfind(") ")
    if close < 0:
        raise IdentityError("proc stat lacks final command delimiter")
    head = text[:close]
    if not head.startswith(f"{expected_pid} ("):
        raise IdentityError("proc stat PID differs from requested PID")
    fields = text[close + 2 :].split()
    if len(fields) <= 19:
        raise IdentityError("proc stat is missing pgrp/starttime fields")
    try:
        pgid = int(fields[2])
        starttime = int(fields[19])
    except ValueError as exc:
        raise IdentityError("proc stat pgrp/starttime is not numeric") from exc
    if pgid <= 0 or starttime <= 0:
        raise IdentityError("proc stat pgrp/starttime must be positive")
    return Identity(expected_pid, pgid, starttime)


def stable_identity(
    proc_root: Path,
    pid: int,
    *,
    getpgid: Callable[[int], int] = os.getpgid,
) -> Identity:
    path = proc_root / str(pid) / "stat"
    try:
        first = _parse_stat(path.read_text(encoding="utf-8"), pid)
        observed_pgid = getpgid(pid)
        second = _parse_stat(path.read_text(encoding="utf-8"), pid)
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise IdentityError(f"process {pid} is absent") from exc
    if first != second:
        raise IdentityError(f"process {pid} identity changed while reading")
    if observed_pgid != first.pgid:
        raise IdentityError(f"process {pid} getpgid/proc pgrp mismatch")
    return first


def _identity_once(proc_root: Path, pid: int) -> Identity:
    try:
        return _parse_stat(
            (proc_root / str(pid) / "stat").read_text(encoding="utf-8"), pid
        )
    except FileNotFoundError as exc:
        raise IdentityError(f"process {pid} is absent") from exc


def group_snapshot(
    proc_root: Path,
    pgid: int,
    *,
    getpgid: Callable[[int], int] = os.getpgid,
) -> list[Identity]:
    def scan() -> list[Identity]:
        result: list[Identity] = []
        try:
            entries = list(proc_root.iterdir())
        except OSError as exc:
            raise IdentityError(f"cannot enumerate {proc_root}") from exc
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                candidate = _identity_once(proc_root, pid)
            except IdentityError as exc:
                if "is absent" in str(exc):
                    continue
                raise
            if candidate.pgid != pgid:
                continue
            try:
                identity = stable_identity(proc_root, pid, getpgid=getpgid)
            except IdentityError as exc:
                if "is absent" in str(exc):
                    continue
                raise
            if identity.pgid != pgid:
                raise IdentityError(
                    f"process {pid} changed group while snapshotting PGID {pgid}"
                )
            result.append(identity)
        return sorted(result)

    first = scan()
    second = scan()
    if first != second:
        raise IdentityError("process-group membership changed while reading")
    return first


def _read_regular_json(path: Path) -> dict:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise IdentityError(f"evidence must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        raw = b""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            raw += chunk
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise IdentityError(f"evidence changed while reading: {path}")
    try:
        after_path = path.lstat()
    except FileNotFoundError as exc:
        raise IdentityError(f"evidence vanished while reading: {path}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    ):
        raise IdentityError(f"evidence path changed while reading: {path}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityError(f"invalid evidence JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IdentityError("evidence root must be an object")
    return value


def _publish(path: Path, value: dict) -> None:
    payload = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def bind_leader(
    proc_root: Path,
    pid: int,
    pgid: int,
    output: Path,
    *,
    getpgid: Callable[[int], int] = os.getpgid,
) -> dict:
    identity = stable_identity(proc_root, pid, getpgid=getpgid)
    if identity.pid != pgid or identity.pgid != pgid:
        raise IdentityError("leader must have PID=PGID")
    value = {"schema_version": 1, "kind": "leader_identity", "leader": asdict(identity)}
    _publish(output, value)
    return value


def _leader_from(value: dict) -> Identity:
    try:
        leader = value["leader"]
        identity = Identity(
            pid=int(leader["pid"]),
            pgid=int(leader["pgid"]),
            starttime_ticks=int(leader["starttime_ticks"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IdentityError("leader evidence is malformed") from exc
    if identity.pid != identity.pgid:
        raise IdentityError("leader evidence does not bind PID=PGID")
    return identity


def term_group(
    proc_root: Path,
    leader_evidence: Path,
    output: Path,
    *,
    getpgid: Callable[[int], int] = os.getpgid,
    killpg: Callable[[int, int], None] = os.killpg,
) -> dict:
    leader = _leader_from(_read_regular_json(leader_evidence))
    current = stable_identity(proc_root, leader.pid, getpgid=getpgid)
    if current != leader:
        raise IdentityError("leader identity drifted before TERM")
    members = group_snapshot(proc_root, leader.pgid, getpgid=getpgid)
    if leader not in members:
        raise IdentityError("leader is absent from its process group before TERM")
    if stable_identity(proc_root, leader.pid, getpgid=getpgid) != leader:
        raise IdentityError("leader identity drifted after member snapshot")
    value = {
        "schema_version": 1,
        "kind": "pre_term_group_identity",
        "leader": asdict(leader),
        "members": [asdict(item) for item in members],
    }
    _publish(output, value)
    killpg(leader.pgid, signal.SIGTERM)
    return value


def _bound_members(value: dict) -> tuple[Identity, dict[int, Identity]]:
    leader = _leader_from(value)
    try:
        members = {
            int(item["pid"]): Identity(
                int(item["pid"]), int(item["pgid"]), int(item["starttime_ticks"])
            )
            for item in value["members"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise IdentityError("group evidence members are malformed") from exc
    if not members or leader.pid not in members or members[leader.pid] != leader:
        raise IdentityError("group evidence does not contain its exact leader")
    if any(item.pgid != leader.pgid for item in members.values()):
        raise IdentityError("group evidence contains a foreign PGID")
    return leader, members


def verify_residual(
    proc_root: Path,
    group_evidence: Path,
    *,
    getpgid: Callable[[int], int] = os.getpgid,
) -> list[Identity]:
    leader, bound = _bound_members(_read_regular_json(group_evidence))
    current = group_snapshot(proc_root, leader.pgid, getpgid=getpgid)
    for item in current:
        if bound.get(item.pid) != item:
            raise IdentityError(
                f"unbound or reused process {item.pid} joined PGID {leader.pgid}"
            )
    return current


def kill_residual(
    proc_root: Path,
    term_evidence: Path,
    output: Path,
    *,
    getpgid: Callable[[int], int] = os.getpgid,
    killpg: Callable[[int, int], None] = os.killpg,
) -> dict:
    evidence = _read_regular_json(term_evidence)
    leader, _bound = _bound_members(evidence)
    current = verify_residual(proc_root, term_evidence, getpgid=getpgid)
    value = {
        "schema_version": 1,
        "kind": "pre_kill_group_identity",
        "leader": asdict(leader),
        "members": [asdict(item) for item in current],
    }
    _publish(output, value)
    if current:
        killpg(leader.pgid, signal.SIGKILL)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="mode", required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--pid", type=int, required=True)
    bind.add_argument("--pgid", type=int, required=True)
    bind.add_argument("--output", type=Path, required=True)
    term = sub.add_parser("term")
    term.add_argument("--leader-evidence", type=Path, required=True)
    term.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--group-evidence", type=Path, required=True)
    kill = sub.add_parser("kill")
    kill.add_argument("--term-evidence", type=Path, required=True)
    kill.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "bind":
            value = bind_leader(args.proc_root, args.pid, args.pgid, args.output)
            leader = _leader_from(value)
            print(f"{leader.pid} {leader.pgid} {leader.starttime_ticks}")
        elif args.mode == "term":
            value = term_group(args.proc_root, args.leader_evidence, args.output)
            print(len(value["members"]))
        elif args.mode == "check":
            members = verify_residual(args.proc_root, args.group_evidence)
            print(len(members))
        else:
            value = kill_residual(args.proc_root, args.term_evidence, args.output)
            print(len(value["members"]))
    except (IdentityError, OSError) as exc:
        print(f"EXACT_GROUP_IDENTITY_REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
