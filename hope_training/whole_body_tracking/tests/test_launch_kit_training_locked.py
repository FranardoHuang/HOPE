from __future__ import annotations

import os
from pathlib import Path
import signal
import shutil
import shlex
import subprocess
import sys
import time

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = WBT_ROOT / "scripts" / "launch_kit_training_locked.sh"
IDENTITY_HELPER = WBT_ROOT / "scripts" / "exact_process_group.py"


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
        "import os, subprocess, sys, time\n"
        "pid = int(sys.argv[-1])\n"
        "if any('stat=' in value for value in sys.argv):\n"
        "    raise SystemExit(subprocess.run(['/bin/ps', '-o', 'stat=', '-p', str(pid)]).returncode)\n"
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
        "def group_members(pgid):\n"
        "    result = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid='], capture_output=True, text=True, check=True)\n"
        "    return sorted(int(row.split()[0]) for row in result.stdout.splitlines() if len(row.split()) == 2 and int(row.split()[1]) == pgid)\n"
        "if mode == 'bind':\n"
        "    if os.environ.get('FAIL_EXACT_BIND') == '1': raise SystemExit(2)\n"
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
        "    try: os.getpgid(leader['pid'])\n"
        "    except ProcessLookupError: print(0)\n"
        "    else: print(1)\n"
        "elif mode == 'empty':\n"
        "    leader = json.loads(pathlib.Path(arg('--leader-evidence')).read_text())['leader']\n"
        "    try: os.getpgid(leader['pid'])\n"
        "    except ProcessLookupError: print(0)\n"
        "    else: raise SystemExit(2)\n"
        "elif mode == 'completed-term':\n"
        "    leader = json.loads(pathlib.Path(arg('--leader-evidence')).read_text())['leader']\n"
        "    pids = [pid for pid in group_members(leader['pgid']) if pid != leader['pid']]\n"
        "    members = [{'pid': pid, 'pgid': leader['pgid'], 'starttime_ticks': pid + 1000} for pid in pids]\n"
        "    write(arg('--output'), {'schema_version': 1, 'kind': 'pre_term_completed_group_identity', 'leader': leader, 'members': members})\n"
        "    if members: os.killpg(leader['pgid'], signal.SIGTERM)\n"
        "    print(len(members))\n"
        "elif mode == 'completed-check':\n"
        "    source = json.loads(pathlib.Path(arg('--group-evidence')).read_text())\n"
        "    current = group_members(source['leader']['pgid'])\n"
        "    if not set(current).issubset({row['pid'] for row in source['members']}): raise SystemExit(2)\n"
        "    print(len(current))\n"
        "elif mode == 'completed-kill':\n"
        "    source = json.loads(pathlib.Path(arg('--term-evidence')).read_text())\n"
        "    current = group_members(source['leader']['pgid'])\n"
        "    members = [row for row in source['members'] if row['pid'] in current]\n"
        "    write(arg('--output'), {'schema_version': 1, 'kind': 'pre_kill_completed_group_identity', 'leader': source['leader'], 'members': members})\n"
        "    if members: os.killpg(source['leader']['pgid'], signal.SIGKILL)\n"
        "    print(len(members))\n"
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


def _copy_test_launcher(
    tmp_path: Path, portable_launch_tools: Path, fixed_lock: Path
) -> Path:
    test_scripts = tmp_path / "test-scripts"
    test_scripts.mkdir(exist_ok=True)
    launcher = test_scripts / LAUNCHER.name
    grep_path = portable_launch_tools / "grep"
    if not grep_path.exists():
        resolved_grep = shutil.which("grep", path=os.defpath)
        assert resolved_grep is not None
        grep_path = Path(resolved_grep)
    resolved_stat = shutil.which("stat", path=os.defpath)
    resolved_mkfifo = shutil.which("mkfifo", path=os.defpath)
    assert resolved_stat is not None and resolved_mkfifo is not None
    replacements = {
        "readonly TRUSTED_PATH=/usr/bin:/bin": (
            "readonly TRUSTED_PATH=" + shlex.quote(os.defpath)
        ),
        "readonly FLOCK_BIN=/usr/bin/flock": (
            "readonly FLOCK_BIN="
            + shlex.quote(str(portable_launch_tools / "flock"))
        ),
        "readonly SETSID_BIN=/usr/bin/setsid": (
            "readonly SETSID_BIN="
            + shlex.quote(str(portable_launch_tools / "setsid"))
        ),
        "readonly PS_BIN=/usr/bin/ps": (
            "readonly PS_BIN="
            + shlex.quote(str(portable_launch_tools / "ps"))
        ),
        "readonly GREP_BIN=/usr/bin/grep": (
            "readonly GREP_BIN=" + shlex.quote(str(grep_path))
        ),
        "readonly STAT_BIN=/usr/bin/stat": (
            "readonly STAT_BIN=" + shlex.quote(resolved_stat)
        ),
        "readonly MKFIFO_BIN=/usr/bin/mkfifo": (
            "readonly MKFIFO_BIN=" + shlex.quote(resolved_mkfifo)
        ),
        "readonly PYTHON_BIN=/usr/bin/python3.10": (
            "readonly PYTHON_BIN="
            + shlex.quote(str(portable_launch_tools / "python3"))
        ),
        "lock_file=/workspace/.kit_boot.lock": (
            "lock_file=" + shlex.quote(str(fixed_lock))
        ),
    }
    source = LAUNCHER.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new, 1)
    launcher.write_text(source, encoding="utf-8")
    launcher.chmod(0o755)
    shutil.copyfile(IDENTITY_HELPER, test_scripts / IDENTITY_HELPER.name)
    return launcher


def _run_launcher(
    tmp_path: Path,
    portable_launch_tools: Path,
    child: Path,
    *,
    timeout_s: int = 12,
    stale_timeout_s: str = "2",
    extra_env: dict[str, str] | None = None,
    boot_lock_kind: str = "regular",
) -> tuple[subprocess.CompletedProcess[str], Path, Path, float]:
    log = tmp_path / "run.log"
    state = tmp_path / "run.log.launch"
    fixed_lock = tmp_path / "kit.lock"
    if boot_lock_kind == "regular":
        fixed_lock.touch()
    elif boot_lock_kind == "symlink":
        target = tmp_path / "kit.lock.target"
        target.touch()
        fixed_lock.symlink_to(target)
    elif boot_lock_kind != "missing":
        raise AssertionError(f"unknown boot lock kind: {boot_lock_kind}")
    launcher = _copy_test_launcher(
        tmp_path, portable_launch_tools, fixed_lock
    )
    env = {
        **os.environ,
        "PATH": f"{portable_launch_tools}{os.pathsep}{os.environ['PATH']}",
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
        [str(launcher), str(log), str(child)],
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
    assert source.index('"$GREP_BIN" -Fq -- "$marker"') < source.index(
        "KIT_BOOT_STALE pid="
    )
    assert "pkill" not in source
    assert "killall" not in source
    assert "lock_file=/workspace/.kit_boot.lock" in source
    assert "KIT_BOOT_LOCK must be the pod-wide" in source
    assert "caller-supplied HOPE_KIT_BOOT_FDS is forbidden" in source
    assert 'environment["HOPE_KIT_BOOT_FDS"]' not in source
    assert '"$MKFIFO_BIN" -m 600' in source
    assert source.startswith("#!/bin/bash -p\n")
    for assignment in (
        "readonly FLOCK_BIN=/usr/bin/flock",
        "readonly SETSID_BIN=/usr/bin/setsid",
        "readonly PS_BIN=/usr/bin/ps",
        "readonly GREP_BIN=/usr/bin/grep",
        "readonly STAT_BIN=/usr/bin/stat",
        "readonly MKFIFO_BIN=/usr/bin/mkfifo",
        "readonly PYTHON_BIN=/usr/bin/python3.10",
    ):
        assert assignment in source
    assert source.index('"$identity_helper" bind') < source.index("printf 'G' >&6")
    assert "trap '[[ -n $stop_signal ]] || stop_signal=TERM' TERM" in source
    for terminal_kind in (
        "pre_marker_exit",
        "watchdog_error",
        "stale_timeout",
        "boot_timeout",
        "signal_stop",
    ):
        assert f"terminal_kind={terminal_kind}" in source


def test_caller_fd_shortcut_is_refused_before_any_artifact_or_workload(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    started = tmp_path / "started"
    child = _write_child(tmp_path, f"printf started > {started!s}\n")
    proc, log, state, _ = _run_launcher(
        tmp_path,
        portable_launch_tools,
        child,
        extra_env={"HOPE_KIT_BOOT_FDS": "7,8,9"},
    )
    assert proc.returncode == 2
    assert "caller-supplied HOPE_KIT_BOOT_FDS is forbidden" in proc.stderr
    assert not log.exists()
    assert not state.exists()
    assert not started.exists()


def test_caller_path_and_loader_python_shell_hooks_do_not_reach_workload(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    fake_bin = tmp_path / "attacker-bin"
    fake_bin.mkdir()
    marker = tmp_path / "attacker-tool-ran"
    for name in ("python3", "flock", "setsid", "ps", "grep", "stat", "mkfifo"):
        tool = fake_bin / name
        tool.write_text(
            f"#!/bin/sh\nprintf injected >> {marker!s}\nexit 93\n",
            encoding="utf-8",
        )
        tool.chmod(0o755)
    leaked = tmp_path / "leaked-env"
    child = _write_child(
        tmp_path,
        "for name in PYTHONPATH PYTHONHOME BASH_ENV ENV GIT_DIR "
        "LD_PRELOAD DYLD_INSERT_LIBRARIES XDG_CONFIG_HOME; do\n"
        "  eval \"value=\\${$name-}\"\n"
        f"  [ -z \"$value\" ] || printf '%s=%s\\n' \"$name\" \"$value\" >> {leaked!s}\n"
        "done\n"
        "printf 'KIT_READY\\n'\n"
        "while :; do sleep 1; done\n",
    )
    proc, _log, state, _ = _run_launcher(
        tmp_path,
        portable_launch_tools,
        child,
        timeout_s=8,
        extra_env={
            "PATH": str(fake_bin),
            "PYTHONPATH": "/attacker/python",
            "PYTHONHOME": "/attacker/home",
            "BASH_ENV": "/attacker/bash-env",
            "ENV": "/attacker/sh-env",
            "GIT_DIR": "/attacker/git",
            "LD_PRELOAD": "/attacker/lib.so",
            "DYLD_INSERT_LIBRARIES": "/attacker/lib.dylib",
            "XDG_CONFIG_HOME": "/attacker/xdg",
        },
    )
    try:
        assert proc.returncode == 0, proc.stderr
        assert not marker.exists()
        assert not leaked.exists()
        fields = _state_fields(state)
        assert len(fields["bootstrap_handoff_token_sha256"]) == 64
        assert any(key.startswith("trusted_tool_") for key in fields)
    finally:
        _terminate_group_from_state(state)


def test_missing_or_symlink_boot_lock_fails_before_creating_run_artifacts(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    child = _write_child(tmp_path, "exit 99\n")
    for kind in ("missing", "symlink"):
        attempt = tmp_path / kind
        attempt.mkdir()
        attempt_child = _write_child(attempt, "exit 99\n")
        proc, log, state, _ = _run_launcher(
            attempt,
            portable_launch_tools,
            attempt_child,
            boot_lock_kind=kind,
        )
        assert proc.returncode != 0
        assert not log.exists()
        assert not state.exists()


def test_existing_log_is_no_clobber_and_workload_never_starts(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    started = tmp_path / "started"
    child = _write_child(tmp_path, f"printf started > {started!s}\n")
    log = tmp_path / "run.log"
    log.write_text("earlier owner\n", encoding="utf-8")
    proc, returned_log, state, _ = _run_launcher(
        tmp_path, portable_launch_tools, child
    )
    assert proc.returncode != 0
    assert returned_log == log
    assert log.read_text(encoding="utf-8") == "earlier owner\n"
    assert not state.exists()
    assert not started.exists()


def test_bind_failure_releases_gate_with_refusal_not_workload(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    started = tmp_path / "started"
    child = _write_child(tmp_path, f"printf started > {started!s}\n")
    proc, log, state, _ = _run_launcher(
        tmp_path,
        portable_launch_tools,
        child,
        extra_env={"FAIL_EXACT_BIND": "1"},
    )
    assert proc.returncode == 121
    assert log.read_bytes() == b""
    fields = _state_fields(state)
    assert fields["identity_bind_refused"] == "proc_identity_unverified"
    assert fields["still_gated_reaped"] == "identity_bind_refused"
    assert not started.exists()


def test_sigterm_after_gate_release_exactly_stops_bound_group(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    fixed_lock = tmp_path / "kit.lock"
    fixed_lock.touch()
    launcher = _copy_test_launcher(
        tmp_path, portable_launch_tools, fixed_lock
    )
    target_term = tmp_path / "target.term"
    child = _write_child(
        tmp_path,
        "trap 'printf term > \"$TARGET_TERM_FILE\"; exit 0' TERM\n"
        "printf 'booting\\n'\n"
        "while :; do sleep 1; done\n",
    )
    log = tmp_path / "run.log"
    state = tmp_path / "run.log.launch"
    env = {
        **os.environ,
        "PATH": f"{portable_launch_tools}{os.pathsep}{os.environ['PATH']}",
        "KIT_BOOT_MARKER": "NEVER_READY",
        "KIT_BOOT_TIMEOUT_S": "20",
        "KIT_BOOT_STALE_TIMEOUT_S": "10",
        "KIT_BOOT_POLL_S": "1",
        "KIT_BOOT_STATE_FILE": str(state),
        "TARGET_TERM_FILE": str(target_term),
    }
    proc = subprocess.Popen(
        [str(launcher), str(log), str(child)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if state.is_file() and "workload_gate_released_utc=" in state.read_text(
            encoding="utf-8"
        ):
            break
        time.sleep(0.02)
    else:
        proc.kill()
        proc.wait(timeout=3)
        pytest.fail("workload gate was not released")
    proc.send_signal(signal.SIGTERM)
    _stdout, stderr = proc.communicate(timeout=12)
    assert proc.returncode == 143, stderr
    fields = _state_fields(state)
    assert fields["stop_signal"] == "TERM"
    assert fields["terminal_kind"] == "signal_stop"
    assert fields["terminal_exit_code"] == "143"
    assert target_term.is_file()


def test_pre_marker_exit_publishes_terminal_classification(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    child = _write_child(tmp_path, "exit 17\n")
    proc, _log, state, _ = _run_launcher(
        tmp_path, portable_launch_tools, child, timeout_s=8
    )
    assert proc.returncode == 17
    fields = _state_fields(state)
    assert fields["terminal_kind"] == "pre_marker_exit"
    assert fields["terminal_exit_code"] == "17"


def test_completion_mode_waits_for_clean_exit_and_empty_group(
    tmp_path: Path, portable_launch_tools: Path
) -> None:
    child = _write_child(
        tmp_path,
        "printf 'KIT_READY\\n'\n"
        "sleep 1\n"
        "exit 0\n",
    )
    proc, _log, state, elapsed = _run_launcher(
        tmp_path,
        portable_launch_tools,
        child,
        timeout_s=8,
        extra_env={
            "KIT_WAIT_FOR_COMPLETION": "1",
            "KIT_COMPLETION_TIMEOUT_S": "5",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert elapsed >= 1.0
    fields = _state_fields(state)
    assert fields["completion_exit_code"] == "0"
    assert fields["terminal_kind"] == "clean_completion"
    assert fields["terminal_exit_code"] == "0"


@pytest.mark.parametrize(
    ("leader_exit", "expected_exit", "terminal_kind"),
    ((0, 123, "completion_residual_group"), (17, 17, "completion_nonzero_exit")),
)
def test_completion_mode_cleans_descendant_before_refusing_terminal_state(
    tmp_path: Path,
    portable_launch_tools: Path,
    leader_exit: int,
    expected_exit: int,
    terminal_kind: str,
) -> None:
    descendant_pid = tmp_path / "descendant.pid"
    child = _write_child(
        tmp_path,
        "sleep 30 &\n"
        f"printf '%s\\n' \"$!\" > {descendant_pid!s}\n"
        "printf 'KIT_READY\\n'\n"
        f"exit {leader_exit}\n",
    )
    proc, _log, state, _ = _run_launcher(
        tmp_path,
        portable_launch_tools,
        child,
        timeout_s=8,
        extra_env={
            "KIT_WAIT_FOR_COMPLETION": "1",
            "KIT_COMPLETION_TIMEOUT_S": "5",
        },
    )
    assert proc.returncode == expected_exit, proc.stderr
    fields = _state_fields(state)
    assert fields["completion_exit_code"] == str(leader_exit)
    assert fields["completion_cleanup_completed"] == "true"
    assert fields["terminal_kind"] == terminal_kind
    assert fields["terminal_exit_code"] == str(expected_exit)
    pid = int(descendant_pid.read_text(encoding="ascii"))
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("completed oracle descendant survived exact cleanup")


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
        assert fields["terminal_kind"] == "stale_timeout"
        assert fields["terminal_exit_code"] == "125"
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
        assert fields["terminal_kind"] == "boot_timeout"
        assert fields["terminal_exit_code"] == "124"
        assert "boot_stale_timeout_s" not in fields
    finally:
        _terminate_group_from_state(state)
