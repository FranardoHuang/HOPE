from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "launch_mujoco_full_mdp_successor.py"
RUNNER = Path("hope_training/whole_body_tracking/mjlab_lane/"
              "mujoco_gpu_ac_full_mdp_wait_rsl3.py")
UUID = "GPU-exact-0002"
NAMESPACE = "mujoco-full-a-h48-test-0001"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return result.stdout.strip()


def _executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"launch_successor_{path.parent.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    return module


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path.resolve()
    repo = base / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    runner = repo / RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("# tracked runner placeholder\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Launcher Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    module = _load(repo / "scripts" / SCRIPT.name)

    tools = base / "tools"
    tools.mkdir()
    python = _executable(tools / "fake-python", """#!/usr/bin/env python3
import fcntl, json, os, pathlib, sys, time
fd = os.open(os.environ["FAKE_LOCK_FILE"], os.O_RDWR)
held = False
try:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held = True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
finally:
    os.close(fd)
names = ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
         "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"]
payload = {"argv": sys.argv[1:], "cwd": os.getcwd(), "lock_held": held,
           "env": {name: os.environ.get(name) for name in names}}
pathlib.Path(os.environ["FAKE_CHILD_RECORD"]).write_text(json.dumps(payload))
pathlib.Path(os.environ["FAKE_CHILD_STARTED"]).write_text("started")
print("fake child stdout", flush=True)
print("fake child stderr", file=sys.stderr, flush=True)
release = os.environ.get("FAKE_CHILD_RELEASE")
deadline = time.monotonic() + 5
while release and not pathlib.Path(release).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
sys.exit(int(os.environ.get("FAKE_CHILD_RC", "0")))
""")
    nvidia = _executable(tools / "nvidia-smi", """#!/bin/sh
if [ "$#" -eq 2 ] && [ "$1" = "--query-gpu=index,uuid" ] && [ "$2" = "--format=csv,noheader,nounits" ]; then
    printf '%s' "$FAKE_GPU_ROWS"
elif [ "$#" -eq 2 ] && [ "$1" = "--query-compute-apps=gpu_uuid,pid" ] && [ "$2" = "--format=csv,noheader,nounits" ]; then
    printf '%s' "$FAKE_APP_ROWS"
else
    exit 91
fi
""")
    workspace = base / "workspace"
    workspace.mkdir()
    run_root = workspace / "runs" / NAMESPACE
    lock = base / "gpu.lock"
    lock.write_text("", encoding="utf-8")
    record, started, release = (base / name for name in ("record.json", "started", "release"))
    monkeypatch.setattr(module, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(module, "NVIDIA_SMI", nvidia)
    monkeypatch.setenv("FAKE_GPU_ROWS", f"0, GPU-other-0000\n2, {UUID}\n")
    monkeypatch.setenv("FAKE_APP_ROWS", "")
    monkeypatch.setenv("FAKE_LOCK_FILE", str(lock))
    monkeypatch.setenv("FAKE_CHILD_RECORD", str(record))
    monkeypatch.setenv("FAKE_CHILD_STARTED", str(started))
    monkeypatch.setenv("FAKE_CHILD_RELEASE", "")
    for name in module.ENV_UNSET:
        monkeypatch.setenv(name, "ambient-must-not-leak")

    def argv(*, dry: bool = False, root: Path = run_root,
             executable: Path = python, expected: str = UUID) -> list[str]:
        result = [
            "--python", str(executable), "--run-root", str(root),
            "--namespace", NAMESPACE, "--gpu-index", "2",
            "--expected-gpu-uuid", expected, "--lock-file", str(lock),
        ]
        return result + (["--dry-run"] if dry else [])

    return SimpleNamespace(
        module=module, repo=repo, runner=runner, python=python, nvidia=nvidia,
        workspace=workspace, root=run_root, lock=lock, record=record,
        started=started, release=release, argv=argv,
    )


def _expected_child(rig, commit: str) -> list[str]:
    root = rig.root
    return [
        str(rig.python), str(rig.runner), "--full-a",
        "--num-envs", "4096", "--num-updates", "12500",
        "--evidence-jsonl", str(root / "evidence.jsonl"),
        "--snapshot-dir", str(root / "snapshots"),
        "--completion-json", str(root / "completion.json"),
        "--source-commit", commit, "--run-namespace", NAMESPACE,
        "--mujoco-warp-runtime-site", str(root / "runtime_site"),
        "--save-interval", "500",
    ]


def test_dry_run_reports_actual_head_and_exact_contract_without_resources(
    rig, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.module.NVIDIA_SMI = rig.nvidia.with_name("must-not-run")
    descriptor = os.open(rig.lock, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert rig.module.main(rig.argv(dry=True)) == 0
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    output = capsys.readouterr()
    commit = _git(rig.repo, "rev-parse", "HEAD")
    assert json.loads(output.out) == {
        "argv": _expected_child(rig, commit),
        "env": {
            "set": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1"},
            "unset": ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"],
        },
    }
    assert output.err == ""
    assert not rig.root.parent.exists()


@pytest.mark.parametrize("dirty", ["tracked", "untracked"])
def test_dirty_source_fails_before_run_root(rig, dirty: str, capsys) -> None:
    if dirty == "tracked":
        rig.runner.write_text("# changed\n", encoding="utf-8")
    else:
        (rig.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert "tracked or untracked" in capsys.readouterr().err
    assert not rig.root.parent.exists()


@pytest.mark.parametrize("failure", ["uuid", "apps"])
def test_gpu_uuid_mapping_and_selected_compute_apps_fail_before_root(
    rig, failure: str, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    expected = UUID
    if failure == "uuid":
        expected = "GPU-wrong-0000"
    else:
        monkeypatch.setenv("FAKE_APP_ROWS", f"{UUID}, 9123\n")
    assert rig.module.main(rig.argv(expected=expected)) == 2
    assert "GPU" in capsys.readouterr().err
    assert not rig.root.parent.exists()


def test_busy_lock_fails_nonblocking_before_gpu_and_root(rig, monkeypatch, capsys) -> None:
    monkeypatch.setattr(rig.module, "_gpu_is_free", lambda *_: pytest.fail("GPU queried"))
    descriptor = os.open(rig.lock, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert rig.module.main(rig.argv()) == 2
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert "already held" in capsys.readouterr().err
    assert not rig.root.parent.exists()


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_existing_or_symlink_run_root_is_not_clobbered(rig, kind: str, monkeypatch) -> None:
    rig.root.parent.mkdir()
    if kind == "directory":
        target = rig.root
        target.mkdir()
    else:
        target = rig.workspace / "existing-target"
        target.mkdir()
        rig.root.symlink_to(target, target_is_directory=True)
    marker = target / "keep"
    marker.write_text("user data", encoding="utf-8")
    monkeypatch.setattr(rig.module, "_gpu_is_free", lambda *_: pytest.fail("GPU queried"))
    assert rig.module.main(rig.argv()) == 2
    assert marker.read_text(encoding="utf-8") == "user data"


def test_python_symlink_is_rejected_before_root(rig, capsys) -> None:
    linked = rig.python.with_name("linked-python")
    linked.symlink_to(rig.python)
    assert rig.module.main(rig.argv(dry=True, executable=linked)) == 2
    assert "canonical regular executable" in capsys.readouterr().err
    assert not rig.root.parent.exists()


def test_child_rc_logs_exact_env_argv_and_lock_lifetime(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHILD_RELEASE", str(rig.release))
    monkeypatch.setenv("FAKE_CHILD_RC", "7")
    results: list[int] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(rig.module.launch(rig.module.parse_args(rig.argv())))
        except BaseException as exc:  # make a background failure visible to pytest
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while not rig.started.exists() and time.monotonic() < deadline and not errors:
        time.sleep(0.01)
    assert rig.started.exists(), errors
    assert (rig.root / "snapshots").is_dir()
    assert not (rig.root / "runtime_site").exists()
    assert not (rig.root / "evidence.jsonl").exists()
    assert not (rig.root / "completion.json").exists()
    contender = os.open(rig.lock, os.O_RDWR)
    with pytest.raises(BlockingIOError):
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(contender)
    rig.release.write_text("release", encoding="utf-8")
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == [] and results == [7]
    assert (rig.root / "stdout.log").read_text() == "fake child stdout\n"
    assert (rig.root / "stderr.log").read_text() == "fake child stderr\n"
    record = json.loads(rig.record.read_text())
    assert record["argv"] == _expected_child(rig, _git(rig.repo, "rev-parse", "HEAD"))[1:]
    assert record["cwd"] == str(rig.repo)
    assert record["lock_held"] is True
    assert record["env"] == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": None,
        "PYTHONHOME": None, "VIRTUAL_ENV": None, "CONDA_PREFIX": None,
    }
    descriptor = os.open(rig.lock, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(descriptor)


def test_child_start_failure_leaves_spent_root(rig, capsys) -> None:
    bad_python = _executable(rig.python.with_name("bad-python"), "not an image\n")
    assert rig.module.main(rig.argv(executable=bad_python)) == 2
    assert "cannot start" in capsys.readouterr().err
    assert rig.root.is_dir() and (rig.root / "snapshots").is_dir()
    assert (rig.root / "stdout.log").is_file()
    assert (rig.root / "stderr.log").is_file()
    assert not (rig.root / "runtime_site").exists()
