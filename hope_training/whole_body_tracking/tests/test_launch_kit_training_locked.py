from __future__ import annotations

import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = WBT_ROOT / "scripts" / "launch_kit_training_locked.sh"


@pytest.fixture
def portable_launch_tools(tmp_path: Path) -> Path:
    """Supply Linux-like flock/setsid semantics to the macOS developer test host."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    flock = bin_dir / "flock"
    flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    setsid = bin_dir / "setsid"
    setsid.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "os.setsid()\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n",
        encoding="utf-8",
    )
    # Production GNU setsid normally wins this scheduling race.  The Python
    # portability shim needs a matching ps shim that waits for os.setsid().
    ps = bin_dir / "ps"
    ps.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "pid = int(sys.argv[-1])\n"
        "for _ in range(200):\n"
        "    try:\n"
        "        pgid = os.getpgid(pid)\n"
        "    except ProcessLookupError:\n"
        "        raise SystemExit(1)\n"
        "    if pgid == pid:\n"
        "        print(pgid)\n"
        "        raise SystemExit(0)\n"
        "    time.sleep(0.01)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    # The production helper reads Linux /proc.  This fixture only substitutes
    # that helper on the macOS test host; its own procfs semantics have focused
    # fake-/proc tests in test_exact_process_group.py.
    python3 = bin_dir / "python3"
    python3.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, signal, subprocess, sys\n"
        "if len(sys.argv) < 3 or not sys.argv[1].endswith('exact_process_group.py'):\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        "argv = sys.argv[2:]\n"
        "mode = argv[0]\n"
        "def arg(name): return argv[argv.index(name) + 1]\n"
        "def write(path, value): pathlib.Path(path).write_text(json.dumps(value, sort_keys=True) + '\\n')\n"
        "if mode == 'bind':\n"
        "    pid, pgid = int(arg('--pid')), int(arg('--pgid'))\n"
        "    leader = {'pid': pid, 'pgid': pgid, 'starttime_ticks': pid + 1000}\n"
        "    write(arg('--output'), {'schema_version': 1, 'kind': 'leader_identity', 'leader': leader})\n"
        "    print(pid, pgid, pid + 1000)\n"
        "elif mode == 'term':\n"
        "    leader = json.loads(pathlib.Path(arg('--leader-evidence')).read_text())['leader']\n"
        "    write(arg('--output'), {'schema_version': 1, 'kind': 'pre_term_group_identity', 'leader': leader, 'members': [leader]})\n"
        "    os.killpg(leader['pgid'], signal.SIGTERM); print(1)\n"
        "elif mode == 'check':\n"
        "    leader = json.loads(pathlib.Path(arg('--group-evidence')).read_text())['leader']\n"
        "    p = subprocess.run(['/bin/ps', '-o', 'stat=', '-p', str(leader['pid'])], text=True, capture_output=True)\n"
        "    state = p.stdout.strip(); print(0 if p.returncode or not state or state.startswith('Z') else 1)\n"
        "else:\n"
        "    source = json.loads(pathlib.Path(arg('--term-evidence')).read_text())\n"
        "    write(arg('--output'), {'schema_version': 1, 'kind': 'pre_kill_group_identity', 'leader': source['leader'], 'members': source['members']})\n"
        "    os.killpg(source['leader']['pgid'], signal.SIGKILL); print(len(source['members']))\n",
        encoding="utf-8",
    )
    flock.chmod(0o755)
    setsid.chmod(0o755)
    ps.chmod(0o755)
    python3.chmod(0o755)
    return bin_dir


def _write_child(tmp_path: Path, body: str) -> Path:
    child = tmp_path / "child.sh"
    child.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    child.chmod(0o755)
    return child


def _run_launcher(
    tmp_path: Path,
    portable_launch_tools: Path,
    child: Path,
    *,
    timeout_s: int = 12,
    stale_timeout_s: str = "2",
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, float]:
    log = tmp_path / "run.log"
    state = tmp_path / "run.log.launch"
    env = {
        **os.environ,
        "PATH": f"{portable_launch_tools}{os.pathsep}{os.environ['PATH']}",
        "KIT_BOOT_LOCK": str(tmp_path / "kit.lock"),
        "KIT_BOOT_MARKER": "KIT_READY",
        "KIT_BOOT_TIMEOUT_S": str(timeout_s),
        "KIT_BOOT_STALE_TIMEOUT_S": stale_timeout_s,
        "KIT_BOOT_POLL_S": "1",
        "KIT_BOOT_STATE_FILE": str(state),
    }
    if extra_env:
        env.update(extra_env)
    started = time.monotonic()
    proc = subprocess.run(
        [str(LAUNCHER), str(log), str(child)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s + 12,
        check=False,
    )
    return proc, log, state, time.monotonic() - started


def _state_fields(state: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in state.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def _terminate_group_from_state(state: Path) -> None:
    if not state.exists():
        return
    fields = _state_fields(state)
    try:
        os.killpg(int(fields["pgid"]), signal.SIGTERM)
    except (KeyError, ProcessLookupError):
        return


def test_source_contract_has_180s_default_and_no_broad_signal() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "stale_timeout_s=${KIT_BOOT_STALE_TIMEOUT_S:-180}" in source
    assert '"$identity_helper" term' in source
    assert '"$identity_helper" kill' in source
    assert "leader_starttime_ticks" in source
    assert "kill -TERM" not in source and "kill -KILL" not in source
    assert source.index('grep -Fq -- "$marker"') < source.index("KIT_BOOT_STALE pid=")
    assert "pkill" not in source
    assert "killall" not in source


@pytest.mark.parametrize("invalid", ["0", "-1", "1.5", "true"])
def test_stale_timeout_env_must_be_a_positive_integer(
    tmp_path: Path, portable_launch_tools: Path, invalid: str
) -> None:
    child = _write_child(tmp_path, "exit 99\n")
    proc, log, state, _ = _run_launcher(
        tmp_path, portable_launch_tools, child, stale_timeout_s=invalid
    )
    assert proc.returncode == 2
    assert "KIT_BOOT_STALE_TIMEOUT_S" in proc.stderr
    assert not log.exists()
    assert not state.exists()


def test_log_growth_resets_stale_clock_until_marker_wins(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    child = _write_child(
        tmp_path,
        "i=0\n"
        "while [ \"$i\" -lt 4 ]; do\n"
        "  printf 'import chunk %s\\n' \"$i\"\n"
        "  i=$((i + 1))\n"
        "  sleep 1\n"
        "done\n"
        "printf 'KIT_READY\\n'\n"
        "while :; do sleep 1; done\n",
    )
    proc, _log, state, elapsed = _run_launcher(
        tmp_path, portable_launch_tools, child, timeout_s=12, stale_timeout_s="2"
    )
    try:
        assert proc.returncode == 0, proc.stderr
        assert elapsed >= 3.0
        fields = _state_fields(state)
        assert "ready_utc" in fields
        assert "boot_stale_timeout_s" not in fields
    finally:
        _terminate_group_from_state(state)


def test_marker_check_has_priority_over_watchdogs(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    # Inject the marker on the second marker probe.  That is deterministically
    # either the same-poll watchdog recheck or the next poll's leading check,
    # so the test exercises marker priority without racing a sub-second child
    # sleep against Python/process startup on a loaded developer host.
    real_grep = shutil.which("grep")
    assert real_grep is not None
    grep_count = tmp_path / "marker-grep.count"
    marker_log = tmp_path / "run.log"
    grep = portable_launch_tools / "grep"
    grep.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "counter = pathlib.Path(os.environ['MARKER_PRIORITY_GREP_COUNT'])\n"
        "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        "if count == 2:\n"
        "    with pathlib.Path(os.environ['MARKER_PRIORITY_LOG']).open('a') as stream:\n"
        "        stream.write('KIT_READY\\n')\n"
        f"os.execv({real_grep!r}, [{real_grep!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    grep.chmod(0o755)
    child = _write_child(
        tmp_path,
        "printf 'content before marker\\n'\n"
        "while :; do sleep 1; done\n",
    )
    proc, _log, state, _ = _run_launcher(
        tmp_path,
        portable_launch_tools,
        child,
        timeout_s=1,
        stale_timeout_s="1",
        extra_env={
            "MARKER_PRIORITY_GREP_COUNT": str(grep_count),
            "MARKER_PRIORITY_LOG": str(marker_log),
        },
    )
    try:
        assert proc.returncode == 0, proc.stderr
        assert grep_count.read_text(encoding="utf-8") == "2"
        fields = _state_fields(state)
        assert "ready_utc" in fields
        assert "boot_timeout_s" not in fields
        assert "boot_stale_timeout_s" not in fields
    finally:
        _terminate_group_from_state(state)


def test_content_bearing_stale_log_kills_only_recorded_group(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    unrelated_term = tmp_path / "unrelated.term"
    unrelated = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            f"trap 'printf term > {unrelated_term!s}; exit 0' TERM; while :; do sleep 1; done",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    target_term = tmp_path / "target.term"
    child = _write_child(
        tmp_path,
        "trap 'printf term > \"$TARGET_TERM_FILE\"; exit 0' TERM\n"
        "printf 'URDF import started\\n'\n"
        "while :; do sleep 1; done\n",
    )
    try:
        proc, log, state, _ = _run_launcher(
            tmp_path,
            portable_launch_tools,
            child,
            timeout_s=12,
            stale_timeout_s="2",
            extra_env={"TARGET_TERM_FILE": str(target_term)},
        )
        assert proc.returncode == 125, proc.stderr
        assert "KIT_BOOT_STALE" in proc.stderr
        fields = _state_fields(state)
        assert fields["pid"] == fields["pgid"]
        assert fields["boot_stale_timeout_s"] == "2"
        # The shell may append "Terminated" while handling TERM; the sidecar
        # intentionally preserves the last size observed before signalling.
        assert int(fields["boot_stale_last_size_bytes"]) == len(b"URDF import started\n")
        assert log.stat().st_size >= int(fields["boot_stale_last_size_bytes"])
        assert int(fields["boot_stale_last_mtime_epoch"]) > 0
        assert "boot_timeout_s" not in fields
        assert target_term.exists()
        assert unrelated.poll() is None
        assert not unrelated_term.exists()

    finally:
        _terminate_group_from_state(state)
        try:
            os.killpg(unrelated.pid, signal.SIGTERM)
            unrelated.wait(timeout=3)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(unrelated.pid, signal.SIGKILL)
            unrelated.wait(timeout=3)


def test_empty_log_remains_owned_by_hard_timeout(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    child = _write_child(tmp_path, "while :; do sleep 1; done\n")
    proc, log, state, _ = _run_launcher(
        tmp_path, portable_launch_tools, child, timeout_s=2, stale_timeout_s="1"
    )
    try:
        assert proc.returncode == 124, proc.stderr
        assert log.stat().st_size == 0
        fields = _state_fields(state)
        assert fields["boot_timeout_s"] == "2"
        assert "boot_stale_timeout_s" not in fields
    finally:
        _terminate_group_from_state(state)
