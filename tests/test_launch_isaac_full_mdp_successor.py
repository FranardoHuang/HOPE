from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "launch_isaac_full_mdp_successor.py"
TRAIN = Path("hope_training/whole_body_tracking/scripts/train.py")
KIT_LAUNCHER = Path(
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
NAMESPACE = "isaac-full-a-h48-test-0001"
UUID = "GPU-exact-0000"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"launch_isaac_successor_{path.parent.parent.name}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    return module


def _executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path.resolve()
    repo = base / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    train = repo / TRAIN
    train.parent.mkdir(parents=True)
    train.write_text("# train placeholder\n", encoding="utf-8")
    record = base / "kit-record.json"
    kit = repo / KIT_LAUNCHER
    kit.parent.mkdir(parents=True, exist_ok=True)
    _executable(
        kit,
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps({{'argv': sys.argv[1:], 'env': dict(os.environ)}}))\n",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Launcher Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    module = _load(repo / "scripts" / SCRIPT.name)

    isaaclab = base / "IsaacLab"
    for relative in (
        "source/isaaclab",
        "source/isaaclab_tasks",
        "source/isaaclab_assets",
        "source/isaaclab_rl",
    ):
        (isaaclab / relative).mkdir(parents=True)
        (isaaclab / relative / ".keep").write_text("pinned\n", encoding="utf-8")
    _git(isaaclab, "init", "-q")
    _git(isaaclab, "config", "user.email", "test@example.invalid")
    _git(isaaclab, "config", "user.name", "IsaacLab Test")
    _git(isaaclab, "add", ".")
    _git(isaaclab, "commit", "-qm", "fixture")

    tools = base / "tools"
    tools.mkdir()
    isaac_python = _executable(tools / "python.sh", "#!/bin/sh\nexit 0\n")
    kit_python = _executable(tools / "kit-python", "#!/bin/sh\nexit 0\n")
    nvidia = _executable(
        tools / "nvidia-smi",
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--query-gpu=index,uuid\" ]; then printf '0, %s\\n'; "
        "elif [ \"$1\" = \"--query-compute-apps=gpu_uuid,pid\" ]; then :; else exit 91; fi\n"
        % UUID,
    )
    venv = base / "venv-site"
    venv.mkdir()
    asset = base / "model.usd"
    asset.write_bytes(b"exact-usd")
    wheel = base / "rsl.whl"
    wheel.write_bytes(b"exact-rsl-wheel")
    workspace = base / "workspace"
    workspace.mkdir()
    root = workspace / "runs" / NAMESPACE
    lock = base / "gpu.lock"
    lock.write_bytes(b"")

    monkeypatch.setattr(module, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(module, "NVIDIA_SMI", nvidia)
    monkeypatch.setattr(module, "PINNED_ISAACLAB_COMMIT", _git(isaaclab, "rev-parse", "HEAD"))
    monkeypatch.setattr(module, "PINNED_KIT_PYTHON_SHA256", _sha(kit_python))
    monkeypatch.setattr(module, "PINNED_A3_USD_SHA256", _sha(asset))
    monkeypatch.setattr(module, "PINNED_RSL_WHEEL_SHA256", _sha(wheel))

    def argv(*, dry: bool = False, expected: str = UUID) -> list[str]:
        values = [
            "--isaac-python", str(isaac_python),
            "--kit-python", str(kit_python),
            "--isaaclab-root", str(isaaclab),
            "--venv-site", str(venv),
            "--asset-usd", str(asset),
            "--rsl-wheel", str(wheel),
            "--run-root", str(root),
            "--namespace", NAMESPACE,
            "--gpu-index", "0",
            "--expected-gpu-uuid", expected,
            "--lock-file", str(lock),
        ]
        return values + (["--dry-run"] if dry else [])

    return SimpleNamespace(
        module=module,
        repo=repo,
        isaaclab=isaaclab,
        isaac_python=isaac_python,
        kit_python=kit_python,
        venv=venv,
        asset=asset,
        wheel=wheel,
        root=root,
        lock=lock,
        record=record,
        argv=argv,
    )


def test_dry_run_is_h48_typed_longrun_without_rate_or_recipe_overrides(
    rig, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rig.module, "NVIDIA_SMI", Path("/must/not/run"))
    assert rig.module.main(rig.argv(dry=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_commit"] == _git(rig.repo, "rev-parse", "HEAD")
    joined = " ".join(payload["argv"])
    assert "task=HOPEPingPongActionBallFullMdpA" in joined
    assert "num_envs=4096" in joined
    assert "DIAGNOSTIC_UNAUTHORIZED" in joined
    assert "max_iterations=" not in joined
    assert "save_interval=" not in joined
    assert "action_ball_full_mdp_rate_probe" not in joined
    assert payload["runtime_env"]["PYTHONPATH"].startswith("/proc/self/fd/18:")
    assert payload["launcher_env"]["KIT_WAIT_FOR_COMPLETION"] == "0"
    assert not rig.root.exists()


@pytest.mark.parametrize("dirty", ["source", "isaaclab"])
def test_dirty_checkout_fails_before_gpu_or_root(rig, dirty: str, monkeypatch, capsys) -> None:
    target = rig.repo if dirty == "source" else rig.isaaclab
    (target / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(rig.module, "NVIDIA_SMI", Path("/must/not/run"))
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert "tracked or untracked" in capsys.readouterr().err
    assert not rig.root.exists()


@pytest.mark.parametrize("failure", ["uuid", "apps"])
def test_gpu_identity_or_live_app_refuses_before_root(rig, failure: str, monkeypatch, capsys) -> None:
    if failure == "uuid":
        argv = rig.argv(expected="GPU-wrong-0000")
    else:
        argv = rig.argv()
        def occupied_gpu(index: int, expected_uuid: str) -> None:
            assert index == 0
            assert expected_uuid == UUID
            raise rig.module.LaunchError(
                "selected GPU already has a compute application"
            )

        monkeypatch.setattr(rig.module, "_gpu_is_free", occupied_gpu)
    assert rig.module.main(argv) == 2
    assert "GPU" in capsys.readouterr().err
    assert not rig.root.exists()


def test_real_path_uses_clean_env_wrapper_and_spends_fresh_root(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptors: list[int] = []

    def fake_descriptors(paths, wheel):
        assert wheel == rig.wheel
        for target in (16, 18):
            descriptor = os.open(os.devnull, os.O_RDONLY)
            os.dup2(descriptor, target, inheritable=True)
            os.close(descriptor)
            descriptors.append(target)
        return 16, 18

    monkeypatch.setattr(rig.module, "_open_exact_runtime_descriptors", fake_descriptors)
    monkeypatch.setattr(rig.module, "_verify_started", lambda paths: (1234, 1234))
    assert rig.module.main(rig.argv()) == 0
    assert rig.root.is_dir()
    assert (rig.root / "asset" / "model.usd").read_bytes() == b"exact-usd"
    record = json.loads(rig.record.read_text())
    assert record["argv"][0] == str(rig.root / "run.log")
    child = record["argv"][1:]
    assert child[:2] == ["/usr/bin/env", "-i"]
    assert str(rig.isaac_python) in child
    assert "task=HOPEPingPongActionBallFullMdpA" in child
    assert record["env"]["KIT_BOOT_MARKER"] == "Learning iteration"
    assert "PYTHONPATH" not in record["env"]
    assert "LD_LIBRARY_PATH" not in record["env"]


def test_wrong_pinned_asset_or_rsl_bytes_fail_before_root(rig, capsys) -> None:
    rig.asset.write_bytes(b"changed")
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert "USD digest differs" in capsys.readouterr().err
    assert not rig.root.exists()
