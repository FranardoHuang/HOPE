from __future__ import annotations

import importlib.util
from pathlib import Path
import signal
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/exact_process_group.py"
SPEC = importlib.util.spec_from_file_location("exact_process_group", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def _write_stat(root: Path, pid: int, pgid: int, starttime: int) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    # fields after comm begin at field 3; pgrp is index 2 and starttime index 19.
    fields = ["S", "1", str(pgid), "1", *("0" for _ in range(15)), str(starttime)]
    (directory / "stat").write_text(
        f"{pid} (worker name with ) paren) " + " ".join(fields) + "\n",
        encoding="utf-8",
    )


def _getpgid(root: Path):
    def getpgid(pid: int) -> int:
        return M._identity_once(root, pid).pgid

    return getpgid


def test_bind_double_reads_exact_leader_and_publishes_starttime(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_stat(proc, 101, 101, 7001)
    output = tmp_path / "leader.json"
    value = M.bind_leader(proc, 101, 101, output, getpgid=_getpgid(proc))
    assert value["leader"] == {"pid": 101, "pgid": 101, "starttime_ticks": 7001}
    assert output.is_file()


def test_pid_reuse_before_term_refuses_without_signal(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_stat(proc, 101, 101, 7001)
    leader = tmp_path / "leader.json"
    M.bind_leader(proc, 101, 101, leader, getpgid=_getpgid(proc))
    _write_stat(proc, 101, 101, 9009)
    signals = []
    with pytest.raises(M.IdentityError, match="drifted before TERM"):
        M.term_group(
            proc,
            leader,
            tmp_path / "term.json",
            getpgid=_getpgid(proc),
            killpg=lambda pgid, sig: signals.append((pgid, sig)),
        )
    assert signals == []
    assert not (tmp_path / "term.json").exists()


def test_leader_exit_with_bound_child_uses_group_members_not_leader_liveness(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_stat(proc, 101, 101, 7001)
    _write_stat(proc, 202, 101, 7002)
    leader = tmp_path / "leader.json"
    term = tmp_path / "term.json"
    kill = tmp_path / "kill.json"
    signals = []
    M.bind_leader(proc, 101, 101, leader, getpgid=_getpgid(proc))
    M.term_group(
        proc,
        leader,
        term,
        getpgid=_getpgid(proc),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    )
    (proc / "101" / "stat").unlink()
    assert M.verify_residual(proc, term, getpgid=_getpgid(proc)) == [
        M.Identity(202, 101, 7002)
    ]
    M.kill_residual(
        proc,
        term,
        kill,
        getpgid=_getpgid(proc),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    )
    assert signals == [(101, signal.SIGTERM), (101, signal.SIGKILL)]


def test_member_join_after_term_refuses_kill(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_stat(proc, 101, 101, 7001)
    _write_stat(proc, 202, 101, 7002)
    leader = tmp_path / "leader.json"
    term = tmp_path / "term.json"
    signals = []
    M.bind_leader(proc, 101, 101, leader, getpgid=_getpgid(proc))
    M.term_group(
        proc,
        leader,
        term,
        getpgid=_getpgid(proc),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    )
    _write_stat(proc, 303, 101, 7003)
    with pytest.raises(M.IdentityError, match="unbound or reused process 303"):
        M.kill_residual(
            proc,
            term,
            tmp_path / "kill.json",
            getpgid=_getpgid(proc),
            killpg=lambda pgid, sig: signals.append((pgid, sig)),
        )
    assert signals == [(101, signal.SIGTERM)]


def test_identity_drift_between_proc_reads_refuses_bind(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_stat(proc, 101, 101, 7001)

    def mutating_getpgid(pid: int) -> int:
        _write_stat(proc, pid, 101, 7002)
        return 101

    with pytest.raises(M.IdentityError, match="changed while reading"):
        M.bind_leader(
            proc,
            101,
            101,
            tmp_path / "leader.json",
            getpgid=mutating_getpgid,
        )


def test_empty_exact_group_needs_no_kill_signal(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_stat(proc, 101, 101, 7001)
    leader = tmp_path / "leader.json"
    term = tmp_path / "term.json"
    kill = tmp_path / "kill.json"
    signals = []
    M.bind_leader(proc, 101, 101, leader, getpgid=_getpgid(proc))
    M.term_group(
        proc,
        leader,
        term,
        getpgid=_getpgid(proc),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    )
    (proc / "101" / "stat").unlink()
    M.kill_residual(
        proc,
        term,
        kill,
        getpgid=_getpgid(proc),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    )
    assert signals == [(101, signal.SIGTERM)]
