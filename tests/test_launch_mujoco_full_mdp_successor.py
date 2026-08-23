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
READY_POSE = PROJECT / (
    "configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/"
    "take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json"
)
PLANT_XML = PROJECT / (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
RUNNER = Path("hope_training/whole_body_tracking/mjlab_lane/"
              "mujoco_gpu_ac_full_mdp_wait_rsl3.py")
PLANT_CONTRACT = Path("hope_training/whole_body_tracking/mjlab_lane/"
                      "mujoco_full_mdp_plant_contract.py")
PLANT_MANIFEST = Path("configs/a3_mujoco_identity_v2_20260803.json")
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
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    base = tmp_path.resolve()
    repo = base / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    runner = repo / RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("# tracked runner placeholder\n", encoding="utf-8")
    shutil.copy2(PROJECT / PLANT_CONTRACT, repo / PLANT_CONTRACT)
    manifest = repo / PLANT_MANIFEST
    manifest.parent.mkdir(parents=True)
    shutil.copy2(PROJECT / PLANT_MANIFEST, manifest)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Launcher Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    sys.modules.pop("_hope_mujoco_full_mdp_plant_contract", None)
    request.addfinalizer(
        lambda: sys.modules.pop("_hope_mujoco_full_mdp_plant_contract", None)
    )
    module = _load(repo / "scripts" / SCRIPT.name)

    tools = base / "tools"
    tools.mkdir()
    python = _executable(tools / "fake-python", """#!/usr/bin/env python3
import fcntl, json, os, pathlib, sys, time
root = pathlib.Path.cwd()
base = root.parents[2]
fd = os.open(base / "gpu.lock", os.O_RDWR)
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
         "ACTIONBALL_READY_POSE", "A3_PINGPONG_XML", "WARP_CACHE_PATH", "CUDA_CACHE_PATH",
         "TMPDIR", "PYTHONPYCACHEPREFIX", "PATH", "HOME", "XDG_CACHE_HOME",
         "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "LANG", "LC_ALL", "LC_CTYPE",
         "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "HOPE_GEOMETRY_PY",
         "LD_PRELOAD", "ACTIONBALL_AUDIT_AMBIENT", "OMP_NUM_THREADS", "CUDA_LAUNCH_BLOCKING"]
warp_cache = pathlib.Path(os.environ["WARP_CACHE_PATH"])
cuda_cache = pathlib.Path(os.environ["CUDA_CACHE_PATH"])
payload = {"argv": sys.argv[1:], "cwd": os.getcwd(), "lock_held": held,
           "env": {name: os.environ.get(name) for name in names},
           "warp_cache_is_dir": warp_cache.is_dir(),
           "cuda_cache_is_dir": cuda_cache.is_dir()}
root.joinpath("record.json").write_text(json.dumps(payload))
root.joinpath("started").write_text("started")
pathlib.Path("MUJOCO_LOG.TXT").write_text("process-local runtime log")
warp_cache.joinpath("fake-compiled-kernel").write_text("run-owned cache")
print("fake child stdout", flush=True)
print("fake child stderr", file=sys.stderr, flush=True)
release = root / "release"
deadline = time.monotonic() + 5
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
sys.exit(7)
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
    ready_pose = base / "ready-pose.json"
    shutil.copy2(READY_POSE, ready_pose)
    plant_xml = base / "a3_pingpong.xml"
    shutil.copy2(PLANT_XML, plant_xml)
    record, started, release = (
        run_root / name for name in ("record.json", "started", "release")
    )
    monkeypatch.setattr(module, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(module, "NVIDIA_SMI", nvidia)
    monkeypatch.setenv("FAKE_GPU_ROWS", f"0, GPU-other-0000\n2, {UUID}\n")
    monkeypatch.setenv("FAKE_APP_ROWS", "")
    monkeypatch.setenv("WARP_CACHE_PATH", str(base / "ambient-warp-cache"))
    monkeypatch.setenv("A3_PINGPONG_XML", str(base / "ambient-wrong-plant.xml"))
    for name in (
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
        "HOPE_GEOMETRY_PY", "LD_PRELOAD", "ACTIONBALL_AUDIT_AMBIENT",
        "OMP_NUM_THREADS", "CUDA_LAUNCH_BLOCKING",
    ):
        monkeypatch.setenv(name, "ambient-must-not-leak")

    def argv(*, dry: bool = False, root: Path = run_root,
             executable: Path = python, expected: str = UUID,
             ready: Path = ready_pose, plant: Path = plant_xml) -> list[str]:
        result = [
            "--python", str(executable), "--run-root", str(root),
            "--namespace", NAMESPACE, "--gpu-index", "2",
            "--expected-gpu-uuid", expected, "--lock-file", str(lock),
            "--ready-pose", str(ready), "--plant-xml", str(plant),
        ]
        return result + (["--dry-run"] if dry else [])

    return SimpleNamespace(
        module=module, repo=repo, runner=runner, python=python, nvidia=nvidia,
        workspace=workspace, root=run_root, lock=lock, record=record,
        ready_pose=ready_pose, plant_xml=plant_xml, started=started,
        release=release, argv=argv,
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


def test_plant_contract_loads_under_stdlib_only_python() -> None:
    script = """
import importlib.util, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('pure_plant_contract', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'numpy' not in sys.modules and 'mujoco' not in sys.modules
print(json.dumps(module.expected_plant_model_identity(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script, str(PROJECT / PLANT_CONTRACT)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    identity = json.loads(completed.stdout)
    assert identity["source_plant"]["portable_identity_sha256"] == (
        "472219ae346d9217b7d1af860d462a18d6ed8507c5cbb9c0f1ddcd6f964dfd7a"
    )


def test_good_ready_pose_dry_run_reports_exact_binding_without_resources(
    rig, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
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
            "set": {"PATH": rig.module.CHILD_PATH,
                    "HOME": str(rig.root / "home"),
                    "XDG_CACHE_HOME": str(rig.root / "xdg_cache"),
                    "XDG_CONFIG_HOME": str(rig.root / "xdg_config"),
                    "XDG_DATA_HOME": str(rig.root / "xdg_data"),
                    "XDG_STATE_HOME": str(rig.root / "xdg_state"),
                    "LANG": rig.module.CHILD_LOCALE,
                    "LC_ALL": rig.module.CHILD_LOCALE,
                    "LC_CTYPE": rig.module.CHILD_LOCALE,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "ACTIONBALL_READY_POSE": str(rig.ready_pose),
                    "A3_PINGPONG_XML": str(rig.plant_xml),
                    "CUDA_CACHE_PATH": str(rig.root / "cuda_cache"),
                    "TMPDIR": str(rig.root / "tmp"),
                    "PYTHONPYCACHEPREFIX": str(rig.root / "pycache"),
                    "WARP_CACHE_PATH": str(rig.root / "warp_cache")},
            "inherit": [],
        },
        "plant_xml": {
            "path": str(rig.plant_xml),
            "expected_identity": (
                rig.module._plant_contract_module().expected_plant_model_identity()
            ),
        },
    }
    assert output.err == ""
    assert not rig.root.parent.exists()


def test_child_env_is_closed_and_rejects_ambient_inheritance(rig) -> None:
    contract = rig.module._env_contract(
        2,
        rig.ready_pose,
        rig.plant_xml,
        rig.module._paths(rig.root),
    )
    child = rig.module._child_env(contract)
    assert child == contract["set"]
    assert contract["inherit"] == []
    for name in (
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
        "HOPE_GEOMETRY_PY", "LD_PRELOAD", "ACTIONBALL_AUDIT_AMBIENT",
        "OMP_NUM_THREADS", "CUDA_LAUNCH_BLOCKING",
    ):
        assert name not in child

    poisoned = {"set": dict(contract["set"]), "inherit": ["LD_PRELOAD"]}
    with pytest.raises(rig.module.LaunchError, match="environment contract"):
        rig.module._child_env(poisoned)


def test_ready_pose_argument_is_required(rig) -> None:
    argv = rig.argv(dry=True)
    offset = argv.index("--ready-pose")
    del argv[offset:offset + 2]
    with pytest.raises(SystemExit, match="2"):
        rig.module.parse_args(argv)


def test_plant_xml_argument_is_required(rig) -> None:
    argv = rig.argv(dry=True)
    offset = argv.index("--plant-xml")
    del argv[offset:offset + 2]
    with pytest.raises(SystemExit, match="2"):
        rig.module.parse_args(argv)


@pytest.mark.parametrize("kind", ["missing", "wrong", "symlink"])
def test_ready_pose_fails_closed_before_resources(rig, kind: str, capsys) -> None:
    candidate = rig.ready_pose.with_name(kind + ".json")
    if kind == "wrong":
        candidate.write_text("{}\n", encoding="utf-8")
    elif kind == "symlink":
        candidate.symlink_to(rig.ready_pose)
    assert rig.module.main(rig.argv(dry=True, ready=candidate)) == 2
    assert "ready-pose" in capsys.readouterr().err
    assert not rig.root.parent.exists()


@pytest.mark.parametrize("kind", ["missing", "wrong", "symlink"])
def test_plant_xml_fails_closed_before_resources(rig, kind: str, capsys) -> None:
    candidate = rig.plant_xml.with_name(kind + ".xml")
    if kind == "wrong":
        candidate.write_text("<mujoco/>\n", encoding="utf-8")
    elif kind == "symlink":
        candidate.symlink_to(rig.plant_xml)
    assert rig.module.main(rig.argv(dry=True, plant=candidate)) == 2
    assert "plant-xml" in capsys.readouterr().err
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
    rig, failure: str, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
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


def test_python_symlink_requires_a_canonical_venv(rig) -> None:
    linked = rig.python.with_name("linked-python")
    linked.symlink_to(rig.python)
    with pytest.raises(rig.module.LaunchError, match="canonical venv bin entry"):
        rig.module._python_entry(linked)
    venv = rig.workspace / "exact-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /exact\n", encoding="utf-8")
    entry = venv / "bin" / "python"
    entry.symlink_to(rig.python)
    assert rig.module._python_entry(entry) == entry


def test_child_rc_logs_exact_env_argv_and_lock_lifetime(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert (rig.root / "warp_cache").is_dir()
    assert (rig.root / "cuda_cache").is_dir()
    for directory in (
        "home", "xdg_cache", "xdg_config", "xdg_data", "xdg_state",
    ):
        assert (rig.root / directory).is_dir()
    assert (rig.root / "tmp").is_dir()
    assert (rig.root / "pycache").is_dir()
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
    assert record["cwd"] == str(rig.root)
    assert (rig.root / "MUJOCO_LOG.TXT").read_text() == "process-local runtime log"
    assert record["warp_cache_is_dir"] is True
    assert record["cuda_cache_is_dir"] is True
    assert (rig.root / "warp_cache" / "fake-compiled-kernel").read_text() == "run-owned cache"
    assert not (rig.repo / "MUJOCO_LOG.TXT").exists()
    assert record["lock_held"] is True
    assert record["env"] == {
        "PATH": rig.module.CHILD_PATH,
        "HOME": str(rig.root / "home"),
        "XDG_CACHE_HOME": str(rig.root / "xdg_cache"),
        "XDG_CONFIG_HOME": str(rig.root / "xdg_config"),
        "XDG_DATA_HOME": str(rig.root / "xdg_data"),
        "XDG_STATE_HOME": str(rig.root / "xdg_state"),
        "LANG": rig.module.CHILD_LOCALE,
        "LC_ALL": rig.module.CHILD_LOCALE,
        "LC_CTYPE": rig.module.CHILD_LOCALE,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "ACTIONBALL_READY_POSE": str(rig.ready_pose),
        "A3_PINGPONG_XML": str(rig.plant_xml),
        "WARP_CACHE_PATH": str(rig.root / "warp_cache"),
        "CUDA_CACHE_PATH": str(rig.root / "cuda_cache"),
        "TMPDIR": str(rig.root / "tmp"),
        "PYTHONPYCACHEPREFIX": str(rig.root / "pycache"),
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "VIRTUAL_ENV": None,
        "CONDA_PREFIX": None,
        "HOPE_GEOMETRY_PY": None,
        "LD_PRELOAD": None,
        "ACTIONBALL_AUDIT_AMBIENT": None,
        "OMP_NUM_THREADS": None,
        "CUDA_LAUNCH_BLOCKING": None,
    }
    descriptor = os.open(rig.lock, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(descriptor)


def test_child_start_failure_leaves_spent_root(rig, capsys) -> None:
    bad_python = _executable(rig.python.with_name("bad-python"), "not an image\n")
    assert rig.module.main(rig.argv(executable=bad_python)) == 2
    assert "cannot start" in capsys.readouterr().err
    assert rig.root.is_dir() and (rig.root / "snapshots").is_dir()
    assert (rig.root / "warp_cache").is_dir()
    assert (rig.root / "cuda_cache").is_dir()
    for directory in (
        "home", "xdg_cache", "xdg_config", "xdg_data", "xdg_state",
    ):
        assert (rig.root / directory).is_dir()
    assert (rig.root / "tmp").is_dir()
    assert (rig.root / "pycache").is_dir()
    assert (rig.root / "stdout.log").is_file()
    assert (rig.root / "stderr.log").is_file()
    assert not (rig.root / "runtime_site").exists()
