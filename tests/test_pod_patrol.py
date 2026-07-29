from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pod_patrol.sh"


def _write_fake_ssh(tmp_path: Path) -> Path:
    script = tmp_path / "fake-ssh"
    script.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >"$POD_PATROL_TEST_SSH_ARGS"
cat >/dev/null
if [ "${POD_PATROL_TEST_SSH_FAIL:-0}" = 1 ]; then
  echo "synthetic connection failure" >&2
  exit 255
fi
printf '%b\\n' "$POD_PATROL_TEST_SNAPSHOT"
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _write_exec_ssh(tmp_path: Path, *, ps_output: str = "") -> Path:
    remote_bin = tmp_path / "remote-bin"
    remote_bin.mkdir()
    (remote_bin / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (remote_bin / "ps").write_text(
        "#!/bin/sh\nprintf '%s' \"$POD_PATROL_REMOTE_PS_OUTPUT\"\n",
        encoding="utf-8",
    )
    for command in (remote_bin / "nvidia-smi", remote_bin / "ps"):
        command.chmod(command.stat().st_mode | stat.S_IXUSR)
    script = tmp_path / "exec-ssh"
    script.write_text(
        """#!/bin/sh
while [ "$#" -gt 0 ] && [ "$1" != bash ]; do shift; done
[ "$#" -gt 0 ] || exit 64
shift
PATH="$POD_PATROL_REMOTE_TEST_BIN:/usr/bin:/bin"
export PATH
POD_PATROL_PROC_ROOT="$POD_PATROL_REMOTE_PROC_ROOT"
export POD_PATROL_PROC_ROOT
exec /bin/bash "$@"
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _run(
    tmp_path: Path,
    *,
    snapshot: str = "",
    now: int = 1_000,
    extra: list[str] | None = None,
    fail: bool = False,
) -> subprocess.CompletedProcess[str]:
    fake_ssh = _write_fake_ssh(tmp_path)
    args_file = tmp_path / "ssh.args"
    env = os.environ.copy()
    env.update(
        {
            "POD_PATROL_SSH_BIN": str(fake_ssh),
            "POD_PATROL_TEST_SSH_ARGS": str(args_file),
            "POD_PATROL_TEST_SNAPSHOT": snapshot,
            "POD_PATROL_TEST_SSH_FAIL": "1" if fail else "0",
            "POD_PATROL_NOW_EPOCH": str(now),
        }
    )
    args = [
        "bash",
        str(SCRIPT),
        "--pod",
        "pod1=root@203.0.113.7:23456",
        "--root",
        "pod1=/workspace/franco/current_wave",
        "--state-dir",
        str(tmp_path / "state"),
    ]
    if extra:
        args.extend(extra)
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_row(
    iteration: int,
    checkpoint: int = -1,
    *,
    timing: tuple[int, str, str, str, str, int, str, str] | None = None,
) -> str:
    timing_fields = timing or ("?", "?", "?", "?", "?", "?", "?", "?")
    run_row = (
        "RUN\\tpod1\\t4321\\t2000\\tbh_block_upper\\t"
        "/workspace/franco/current_wave/bh_block_upper\\t"
        f"{iteration}\\t{checkpoint}\\t0\\t0.40\\t0.01\\t"
        "/workspace/franco/current_wave/bh_block_upper/run.log\\t"
        + "\\t".join(str(value) for value in timing_fields)
    )
    if timing is None:
        return run_row
    timing_row = (
        "TIMING\\tpod1\\tbh_block_upper\\t"
        + "\\t".join(str(value) for value in timing)
    )
    return timing_row + "\\n" + run_row


def _parse_log(tmp_path: Path, text: str) -> subprocess.CompletedProcess[str]:
    log = tmp_path / "run.log"
    log.write_text(text, encoding="utf-8")
    return subprocess.run(
        ["bash", str(SCRIPT), "_parse-rsl-log", str(log)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_syntax_and_no_legacy_endpoint_or_log_mtime_stale() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    subprocess.run(["/bin/bash", "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "162.43.172." not in source
    assert "/workspace/codexschema" not in source
    assert 'stat -c %Y "$log"' not in source
    assert "kill " not in source
    assert "Learning iteration/checkpoint" in source


def test_latest_complete_iteration_timing_ignores_ansi_and_newer_partial(
    tmp_path: Path,
) -> None:
    result = _parse_log(
        tmp_path,
        """
\x1b[1m Learning iteration 7/1000 \x1b[0m
Computation: 2227 steps/s (collection: 44.045s, learning 0.089s)
Total timesteps: 786432
Iteration time: 44.13s
\x1b[1m Learning iteration 8/1000 \x1b[0m
Computation: 2500 steps/s (collection: 39.000s, learning 0.080s)
Total timesteps: 884736
""",
    )
    assert result.returncode == 0
    assert result.stdout == "7\t44.13\t44.045\t0.089\t2227\t786432\n"


def test_all_complete_iteration_blocks_are_emitted(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        """
Learning iteration 7/1000
Computation: 2227 steps/s (collection: 44.045s, learning 0.089s)
Total timesteps: 786432
Iteration time: 44.13s
Learning iteration 8/1000
Computation: 2500 steps/s (collection: 39.000s, learning 0.080s)
Total timesteps: 884736
Iteration time: 39.08s
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "_parse-rsl-log-all", str(log)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "7\t44.13\t44.045\t0.089\t2227\t786432",
        "8\t39.08\t39.000\t0.080\t2500\t884736",
    ]


def test_timing_parser_rejects_missing_or_incomplete_block(tmp_path: Path) -> None:
    for missing_line in (
        "Computation: 2227 steps/s (collection: 44.045s, learning 0.089s)",
        "Total timesteps: 786432",
        "Iteration time: 44.13s",
    ):
        lines = [
            "Learning iteration 7/1000",
            "Computation: 2227 steps/s (collection: 44.045s, learning 0.089s)",
            "Total timesteps: 786432",
            "Iteration time: 44.13s",
        ]
        lines.remove(missing_line)
        result = _parse_log(tmp_path, "\n".join(lines) + "\n")
        assert result.returncode == 1
        assert result.stdout == ""


def test_unreachable_endpoint_is_a_warn(tmp_path: Path) -> None:
    result = _run(tmp_path, fail=True)
    assert result.returncode == 1
    assert "WARN pod1 unreachable endpoint=root@203.0.113.7:23456" in result.stdout
    assert "synthetic connection failure" in result.stdout


def test_expected_namespace_without_process_warns_missing_via_remote_body(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = tmp_path / "expected_run"
    namespace.mkdir()
    (namespace / "run.log").write_text("booting\n", encoding="utf-8")
    exec_ssh = _write_exec_ssh(tmp_path)
    monkeypatch.setenv("POD_PATROL_SSH_BIN", str(exec_ssh))
    monkeypatch.setenv(
        "POD_PATROL_REMOTE_TEST_BIN", str(tmp_path / "remote-bin")
    )
    monkeypatch.setenv("POD_PATROL_REMOTE_PS_OUTPUT", "")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--pod",
            "pod1=root@203.0.113.7:23456",
            "--namespace",
            f"pod1={namespace}",
            "--state-dir",
            str(tmp_path / "state"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"expected namespace {namespace} trainer MISSING" in result.stdout


def test_remote_body_parses_active_exact_namespace(tmp_path: Path, monkeypatch) -> None:
    namespace = tmp_path / "active_run"
    namespace.mkdir()
    (namespace / "run.log").write_text(
        """
Learning iteration 4/1000
Computation: 2048 steps/s (collection: 47.900s, learning 0.100s)
Total timesteps: 491520
Iteration time: 48.00s
""",
        encoding="utf-8",
    )
    ps_output = (
        "123 60 python /checkout/scripts/train.py "
        "run_name=active_run\n"
    )
    proc_root = tmp_path / "proc"
    (proc_root / "123").mkdir(parents=True)
    stat_fields = ["123", "(python)", "S", *(["0"] * 18), "777"]
    (proc_root / "123" / "stat").write_text(
        " ".join(stat_fields) + "\n", encoding="utf-8"
    )
    (proc_root / "123" / "cmdline").write_bytes(
        b"python\x00/checkout/scripts/train.py\x00run_name=active_run\x00"
    )
    (namespace / "run.log.launch.leader.json").write_text(
        '{"leader":{"pid":123,"starttime_ticks":777}}\n',
        encoding="utf-8",
    )
    (namespace / "launch_spec.json").write_text(
        '{"num_envs":4096,"num_steps_per_env":24}\n',
        encoding="utf-8",
    )
    exec_ssh = _write_exec_ssh(tmp_path, ps_output=ps_output)
    monkeypatch.setenv("POD_PATROL_SSH_BIN", str(exec_ssh))
    monkeypatch.setenv(
        "POD_PATROL_REMOTE_TEST_BIN", str(tmp_path / "remote-bin")
    )
    monkeypatch.setenv("POD_PATROL_REMOTE_PS_OUTPUT", ps_output)
    monkeypatch.setenv("POD_PATROL_REMOTE_PROC_ROOT", str(proc_root))
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--full",
            "--pod",
            "pod1=root@203.0.113.7:23456",
            "--namespace",
            f"pod1={namespace}",
            "--state-dir",
            str(tmp_path / "state"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "iteration=4" in result.stdout
    assert "update_s=48.00" in result.stdout
    assert "num_steps_per_env=24" in result.stdout
    assert "num_envs=4096" in result.stdout
    assert "vector_policy_step_s=2.0000" in result.stdout
    assert "collection_vector_step_wall_s=1.995833" in result.stdout
    assert "amortized_e2e_vector_step_wall_s=2.000000" in result.stdout
    assert "collection_environment_step_us=487.264" in result.stdout
    assert "collection_environment_steps_per_s=2052.276" in result.stdout
    assert "trainer MISSING" not in result.stdout


def test_same_iteration_becomes_stale_but_progress_clears_it(tmp_path: Path) -> None:
    first = _run(tmp_path, snapshot=_run_row(12), now=1_000)
    assert first.returncode == 0
    assert "progress stale" not in first.stdout

    stale = _run(tmp_path, snapshot=_run_row(12), now=2_001)
    assert stale.returncode == 0
    assert "progress stale 1001s iteration=12 checkpoint=-1" in stale.stdout

    advanced = _run(tmp_path, snapshot=_run_row(13), now=2_002, extra=["--full"])
    assert advanced.returncode == 0
    assert "progress stale" not in advanced.stdout
    assert "iteration=13" in advanced.stdout
    assert "progress_age=0s" in advanced.stdout


def test_startup_stale_does_not_use_log_mtime(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        snapshot=_run_row(-1),
        now=1_000,
        extra=["--startup-stale-after", "1800"],
    )
    assert "has no Learning iteration/checkpoint after 2000s" in result.stdout


def test_timing_output_and_cursor_samples_new_complete_blocks_once(
    tmp_path: Path,
) -> None:
    timing_12 = (
        12,
        "44.13",
        "44.045",
        "0.089",
        "2227",
        786432,
        "24",
        "4096",
    )
    first = _run(
        tmp_path,
        snapshot=_run_row(12, timing=timing_12),
        now=1_000,
        extra=["--full"],
    )
    assert first.returncode == 0
    assert "update_s=44.13" in first.stdout
    assert "collection_s=44.045" in first.stdout
    assert "learning_s=0.089" in first.stdout
    assert "env_steps_per_s=2227" in first.stdout
    assert "vector_policy_step_s=1.8388" in first.stdout
    assert "env_steps_per_s_compat=legacy_rsl_reported_rate" in first.stdout
    assert (
        "vector_policy_step_s_compat="
        "legacy_alias_of_amortized_e2e_vector_step_wall_s"
    ) in first.stdout
    assert "collection_vector_step_wall_s=1.835208" in first.stdout
    assert "amortized_e2e_vector_step_wall_s=1.838750" in first.stdout
    assert "collection_environment_step_us=448.049" in first.stdout
    assert "collection_environment_steps_per_s=2231.899" in first.stdout

    repeated = _run(
        tmp_path,
        snapshot=_run_row(12, timing=timing_12),
        now=1_001,
        extra=["--full"],
    )
    assert repeated.returncode == 0

    timing_13 = (
        13,
        "40.8",
        "40.7",
        "0.1",
        "2409",
        884736,
        "24",
        "4096",
    )
    advanced = _run(
        tmp_path,
        snapshot=_run_row(13, timing=timing_13),
        now=1_002,
        extra=["--full"],
    )
    assert "vector_policy_step_s=1.7000" in advanced.stdout

    timing_files = list((tmp_path / "state").glob("*.timing.tsv"))
    assert len(timing_files) == 1
    rows = timing_files[0].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert rows[2].split("\t")[1:] == [
        "12",
        "44.13",
        "44.045",
        "0.089",
        "2227",
        "786432",
        "24",
        "1.8388",
        "4096",
        "1.835208",
        "1.838750",
        "448.049",
        "2231.899",
        "legacy_rsl_reported_rate",
        "legacy_alias_of_amortized_e2e_vector_step_wall_s",
    ]
    assert rows[3].split("\t")[1] == "13"
    state_files = list((tmp_path / "state").glob("*.state"))
    assert len(state_files) == 1
    assert state_files[0].read_text(encoding="utf-8").split("\t")[:2] == [
        "pod1",
        "bh_block_upper",
    ]
    assert not list((tmp_path / "state").glob("*.tmp.*"))


def test_missing_num_steps_does_not_assume_rollout_24(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        snapshot=_run_row(
            12,
            timing=(
                12,
                "44.13",
                "44.045",
                "0.089",
                "2227",
                786432,
                "?",
                "4096",
            ),
        ),
        now=1_000,
        extra=["--full"],
    )
    assert result.returncode == 0
    assert "num_steps_per_env=?" in result.stdout
    assert "vector_policy_step_s=?" in result.stdout
    assert "collection_vector_step_wall_s=?" in result.stdout
    assert "amortized_e2e_vector_step_wall_s=?" in result.stdout
    assert "collection_environment_step_us=?" in result.stdout
    assert "collection_environment_steps_per_s=?" in result.stdout


def test_missing_num_envs_preserves_vector_step_metrics_only(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        snapshot=_run_row(
            12,
            timing=(
                12,
                "44.13",
                "44.045",
                "0.089",
                "2227",
                786432,
                "24",
                "?",
            ),
        ),
        now=1_000,
        extra=["--full"],
    )
    assert result.returncode == 0
    assert "num_envs=?" in result.stdout
    assert "collection_vector_step_wall_s=1.835208" in result.stdout
    assert "amortized_e2e_vector_step_wall_s=1.838750" in result.stdout
    assert "collection_environment_step_us=?" in result.stdout
    assert "collection_environment_steps_per_s=?" in result.stdout


def test_no_argument_does_not_auto_load_stale_wave(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("POD_PATROL_SPEC", None)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--state-dir",
            str(tmp_path / "state"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "no bound pod target" in result.stderr


def test_unsafe_target_components_are_rejected(tmp_path: Path) -> None:
    for args in (
        ["--pod", "../escape=root@host:22"],
        ["--pod", "pod1=-V:22"],
        [
            "--pod",
            "pod1=root@host:22",
            "--root",
            "pod1=/workspace/franco/*",
        ],
    ):
        result = subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0


def test_spec_only_requires_concrete_active_namespace(
    tmp_path: Path, monkeypatch
) -> None:
    exec_ssh = _write_exec_ssh(tmp_path)
    monkeypatch.setenv("POD_PATROL_SSH_BIN", str(exec_ssh))
    monkeypatch.setenv(
        "POD_PATROL_REMOTE_TEST_BIN", str(tmp_path / "remote-bin")
    )
    monkeypatch.setenv("POD_PATROL_REMOTE_PS_OUTPUT", "")
    for status, expected_warning in (
        ("queued_pending_materialization", False),
        ("booted_pending_first_iteration", True),
    ):
        namespace = tmp_path / f"run_{status}"
        namespace.mkdir()
        (namespace / "run.log").write_text("booting\n", encoding="utf-8")
        spec = tmp_path / f"{status}.json"
        spec.write_text(
            json.dumps(
                {
                    "source": {
                        "pod_endpoints": {
                            "pod1": {"host": "host", "port": 22}
                        }
                    },
                    "matrix": [
                        {
                            "pod_id": "pod1",
                            "run_root": str(tmp_path),
                            "upper_runs": [
                                {
                                    "namespace": str(namespace),
                                    "log_path": str(namespace / "run.log"),
                                    "status": status,
                                }
                            ],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--spec",
                str(spec),
                "--state-dir",
                str(tmp_path / f"state_{status}"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert ("trainer MISSING" in result.stdout) is expected_warning


def test_existing_state_dir_permissions_are_not_mutated(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir(mode=0o755)
    state.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--pod",
            "pod1=root@host:22",
            "--state-dir",
            str(state),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert stat.S_IMODE(state.stat().st_mode) == 0o755
    assert "must already be mode 700" in result.stderr


def test_dead_local_cursor_lock_is_recovered(tmp_path: Path) -> None:
    state = tmp_path / "state"
    lock = state / ".pod_patrol.lock"
    lock.mkdir(parents=True, mode=0o700)
    state.chmod(0o700)
    (lock / "owner_pid").write_text("99999999\n", encoding="utf-8")
    result = _run(tmp_path, snapshot="", now=1_000)
    assert result.returncode == 0
    assert not lock.exists()


def test_wave_spec_supplies_endpoint_checkout_and_exact_namespace(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "wave.json"
    spec.write_text(
        json.dumps(
            {
                "source": {
                    "pod_endpoints": {
                        "pod1": {
                            "host": "198.51.100.9",
                            "port": 2222,
                            "user": "root",
                        }
                    }
                },
                "matrix": [
                    {
                        "pod_id": "pod1",
                        "checkout": "/workspace/franco/exact_checkout",
                        "run_root": "/workspace/franco/exact_root",
                        "upper_runs": [
                            {
                                "namespace": (
                                    "/workspace/franco/exact_root/"
                                    "bh_loop_c_upper_current"
                                ),
                                "log_path": (
                                    "/workspace/franco/exact_root/"
                                    "bh_loop_c_upper_current/run.log"
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake_ssh = _write_fake_ssh(tmp_path)
    args_file = tmp_path / "ssh.args"
    env = os.environ.copy()
    env.update(
        {
            "POD_PATROL_SSH_BIN": str(fake_ssh),
            "POD_PATROL_TEST_SSH_ARGS": str(args_file),
            "POD_PATROL_TEST_SNAPSHOT": "",
            "POD_PATROL_NOW_EPOCH": "1000",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--spec",
            str(spec),
            "--state-dir",
            str(tmp_path / "state"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    ssh_args = args_file.read_text(encoding="utf-8")
    assert "-p 2222 root@198.51.100.9" in ssh_args
    assert "/workspace/franco/exact_root" in ssh_args
    assert "/workspace/franco/exact_root/bh_loop_c_upper_current" in ssh_args
    assert "/workspace/franco/exact_checkout" in ssh_args
