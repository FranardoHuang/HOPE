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
    taskset = _executable(tools / "taskset", "#!/bin/sh\nexit 0\n")
    nvidia = _executable(
        tools / "nvidia-smi",
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--query-gpu=index,uuid\" ]; then printf '0, %s\\n'; "
        "elif [ \"$1\" = \"--query-compute-apps=gpu_uuid,pid\" ]; then :; else exit 91; fi\n"
        % UUID,
    )
    venv = base / "venv-site"
    venv.mkdir()
    asset = base / "asset" / "model.usd"
    asset.parent.mkdir()
    asset.write_bytes(b"exact-usd")
    (asset.parent / "source_bundle").mkdir()
    (asset.parent / "source_bundle" / "config.yaml").write_text(
        "fixture: exact\n", encoding="utf-8"
    )
    wheel = base / "rsl.whl"
    wheel.write_bytes(b"exact-rsl-wheel")
    workspace = base / "workspace"
    workspace.mkdir()
    root = workspace / "runs" / NAMESPACE
    lock = base / "gpu.lock"
    lock.write_bytes(b"")

    monkeypatch.setattr(module, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(module, "NVIDIA_SMI", nvidia)
    monkeypatch.setattr(module, "TASKSET", taskset)
    monkeypatch.setattr(module, "_available_cpu_ids", lambda: {2, 3})
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
    assert payload["argv"] == [
        str(rig.isaac_python),
        "-P",
        "-B",
        str(rig.repo / TRAIN),
        "task=HOPEPingPongActionBallFullMdpA",
        "algo=ppo",
        "headless=true",
        "video=false",
        "logger=tensorboard",
        "device=cuda:0",
        "seed=0",
        "num_envs=4096",
        f"run_name={NAMESPACE}-DIAGNOSTIC_UNAUTHORIZED",
        "checkpoint_path=null",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false",
        f"hydra.run.dir={rig.root / 'training' / 'hydra'}",
    ]
    joined = " ".join(payload["argv"])
    assert "task=HOPEPingPongActionBallFullMdpA" in joined
    assert "num_envs=4096" in joined
    assert "DIAGNOSTIC_UNAUTHORIZED" in joined
    assert "max_iterations=" not in joined
    assert "save_interval=" not in joined
    assert "action_ball_full_mdp_rate_probe" not in joined
    assert payload["runtime_env"]["PYTHONPATH"].startswith("/proc/self/fd/18:")
    assert payload["runtime_env"]["HOPE_ACTION_BALL_FULL_MDP_LOG_ROOT"] == str(
        rig.root / "training"
    )
    assert payload["runtime_env"]["HOME"] == str(rig.root / "home")
    assert payload["runtime_env"]["CUDA_CACHE_PATH"] == str(
        rig.root / "cuda_cache"
    )
    for name, directory in (
        ("XDG_CACHE_HOME", "xdg_cache"),
        ("XDG_CONFIG_HOME", "xdg_config"),
        ("XDG_DATA_HOME", "xdg_data"),
        ("XDG_STATE_HOME", "xdg_state"),
    ):
        assert payload["runtime_env"][name] == str(rig.root / directory)
        assert payload["launcher_env"][name] == str(rig.root / directory)
    assert payload["launcher_env"]["HOME"] == str(rig.root / "home")
    assert f"hydra.run.dir={rig.root / 'training' / 'hydra'}" in payload["argv"]
    assert payload["launcher_env"]["KIT_WAIT_FOR_COMPLETION"] == "0"
    assert "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES" not in payload["runtime_env"]
    assert not rig.root.exists()


def test_dry_run_can_bind_bounded_full_mdp_profiler(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(rig.argv(dry=True) + ["--profile-updates", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime_env"][
        "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES"
    ] == "5"
    assert not rig.root.exists()


def test_dry_run_rate_probe_is_explicit_completion_mode_and_default_is_unchanged(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(
        rig.argv(dry=True) + ["--diagnostic-rate-probe"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"][-1] == (
        "task.action_ball_full_mdp_rate_probe=true"
    )
    assert payload["launcher_env"]["KIT_WAIT_FOR_COMPLETION"] == "1"
    assert payload["launcher_env"]["KIT_COMPLETION_TIMEOUT_S"] == "7200"
    assert "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES" not in (
        payload["runtime_env"]
    )

    assert rig.module.main(rig.argv(dry=True)) == 0
    default = json.loads(capsys.readouterr().out)
    assert default["launcher_env"]["KIT_WAIT_FOR_COMPLETION"] == "0"
    assert "KIT_COMPLETION_TIMEOUT_S" not in default["launcher_env"]
    assert "task.action_ball_full_mdp_rate_probe=true" not in default["argv"]


def test_rate_probe_and_profiler_are_mutually_exclusive_before_root(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(
        rig.argv(dry=True)
        + ["--diagnostic-rate-probe", "--profile-updates", "1"]
    ) == 2
    assert "mutually exclusive" in capsys.readouterr().err
    assert not rig.root.exists()


def test_rate_probe_receipt_parses_exact_10_plus_50_and_is_no_clobber(
    rig, tmp_path: Path
) -> None:
    recipe = "a" * 64
    log = tmp_path / "run.log"
    rows = [
        "[train.py] FULLMDP_H48_RATE_PROBE: diagnostic_unauthorized=true "
        "updates=61 warmup=10 measured=50 tail=1 profiler=off",
        "[train.py] full-MDP PPO recipe: "
        f"kind=v4 learning_recipe_sha256={recipe}"
    ]
    for update in range(61):
        rows.extend(
            (
                f"\x1b[1m Learning iteration {update}/61 \x1b[0m",
                f"Iteration time: {1.0 + update / 100.0:.2f}s",
            )
        )
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = rig.module._rate_probe_payload(
        run_log=log,
        source_commit="b" * 40,
        namespace=NAMESPACE,
        gpu_index=0,
        gpu_uuid=UUID,
    )
    assert payload["diagnostic_unauthorized"] is True
    assert payload["formal_evidence"] is False
    assert payload["safety_gate"] is False
    assert payload["source_commit"] == "b" * 40
    assert payload["namespace"] == NAMESPACE
    assert payload["gpu"] == {"index": 0, "uuid": UUID}
    assert payload["action_ball_full_mdp_ppo_recipe_sha256"] == recipe
    assert len(payload["raw_update_seconds"]) == 61
    assert payload["measured_update_seconds"] == pytest.approx(
        [1.0 + update / 100.0 for update in range(10, 60)]
    )
    assert payload["update_seconds_p50"] == pytest.approx(1.345)
    assert payload["update_seconds_p90"] == pytest.approx(1.541)

    receipt = tmp_path / "rate.json"
    rig.module._write_rate_probe_receipt(receipt, payload)
    assert json.loads(receipt.read_text()) == payload
    with pytest.raises(rig.module.LaunchError, match="no-clobber"):
        rig.module._write_rate_probe_receipt(receipt, payload)


def test_rate_probe_parser_rejects_missing_update_or_recipe(rig, tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "[train.py] FULLMDP_H48_RATE_PROBE: diagnostic_unauthorized=true "
        "updates=61 warmup=10 measured=50 tail=1 profiler=off\n"
        + "\n".join(
            f"Learning iteration {update}/61\nIteration time: 1.00s"
            for update in range(60)
        ),
        encoding="utf-8",
    )
    with pytest.raises(rig.module.LaunchError, match="61-update"):
        rig.module._rate_probe_payload(
            run_log=log,
            source_commit="b" * 40,
            namespace=NAMESPACE,
            gpu_index=0,
            gpu_uuid=UUID,
        )


def test_rate_probe_completion_requires_natural_exit_without_stop_path(
    rig, tmp_path: Path
) -> None:
    state = tmp_path / "launch.state"
    runtime_receipt = tmp_path / "runtime.receipt"
    runtime_receipt.write_bytes(b"trainer_runtime_attested_v2\n")
    clean = (
        "pid=1234\npgid=1234\nready_utc=now\ncompletion_utc=later\n"
        "completion_exit_code=0\nterminal_kind=clean_completion\n"
        "terminal_exit_code=0\n"
    )
    state.write_text(clean, encoding="utf-8")
    paths = {"launch_state": state, "runtime_receipt": runtime_receipt}
    assert rig.module._verify_completed(paths) == (1234, 1234)

    state.write_text(clean + "stop_signal=TERM\n", encoding="utf-8")
    with pytest.raises(rig.module.LaunchError, match="used a stop path"):
        rig.module._verify_completed(paths)


def test_dry_run_can_bind_explicit_cpu_affinity(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    cpu = 2
    assert rig.module.main(
        rig.argv(dry=True) + ["--cpu-affinity", str(cpu)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu_affinity"] == str(cpu)
    assert payload["cpu_ids"] == [cpu]
    assert not rig.root.exists()


@pytest.mark.parametrize("value", ["2-1", "1,1", "cpu1"])
def test_invalid_cpu_affinity_fails_before_root(rig, value, capsys) -> None:
    assert rig.module.main(
        rig.argv(dry=True) + ["--cpu-affinity", value]
    ) == 2
    assert "cpu-affinity" in capsys.readouterr().err
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
        descriptors.extend(
            os.open(os.devnull, os.O_RDONLY) for _ in range(2)
        )
        return tuple(descriptors)

    monkeypatch.setattr(rig.module, "_open_exact_runtime_descriptors", fake_descriptors)
    monkeypatch.setattr(
        rig.module,
        "_verify_started",
        lambda paths, *, gpu_lock, lock_file: (1234, 1234),
    )
    assert rig.module.main(rig.argv()) == 0
    assert rig.root.is_dir()
    assert (rig.root / "asset" / "model.usd").read_bytes() == b"exact-usd"
    assert (rig.root / "training").is_dir()
    for directory in (
        "home",
        "cuda_cache",
        "xdg_cache",
        "xdg_config",
        "xdg_data",
        "xdg_state",
    ):
        assert (rig.root / directory).is_dir()
    assert (rig.root / "asset" / "source_bundle" / "config.yaml").read_text() == (
        "fixture: exact\n"
    )
    record = json.loads(rig.record.read_text())
    assert record["argv"][0] == str(rig.root / "run.log")
    child = record["argv"][1:]
    assert child[:2] == ["/usr/bin/env", "-i"]
    assert str(rig.isaac_python) in child
    assert "task=HOPEPingPongActionBallFullMdpA" in child
    assert f"hydra.run.dir={rig.root / 'training' / 'hydra'}" in child
    for item in (
        f"HOME={rig.root / 'home'}",
        f"CUDA_CACHE_PATH={rig.root / 'cuda_cache'}",
        f"XDG_CACHE_HOME={rig.root / 'xdg_cache'}",
        f"XDG_CONFIG_HOME={rig.root / 'xdg_config'}",
        f"XDG_DATA_HOME={rig.root / 'xdg_data'}",
        f"XDG_STATE_HOME={rig.root / 'xdg_state'}",
    ):
        assert item in child
    assert record["env"]["KIT_BOOT_MARKER"] == "Learning iteration"
    assert "PYTHONPATH" not in record["env"]
    assert "LD_LIBRARY_PATH" not in record["env"]


def test_real_path_wraps_child_in_explicit_cpu_affinity(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_descriptors(paths, wheel):
        assert wheel == rig.wheel
        return tuple(
            os.open(os.devnull, os.O_RDONLY) for _ in range(2)
        )

    monkeypatch.setattr(
        rig.module, "_open_exact_runtime_descriptors", fake_descriptors
    )
    monkeypatch.setattr(
        rig.module,
        "_verify_started",
        lambda paths, *, gpu_lock, lock_file: (1234, 1234),
    )
    assert rig.module.main(
        rig.argv() + ["--cpu-affinity", "2"]
    ) == 0
    record = json.loads(rig.record.read_text())
    child = record["argv"][1:]
    assert child[:5] == [
        str(rig.module.TASKSET),
        "-c",
        "2",
        "/usr/bin/env",
        "-i",
    ]


def test_wrong_pinned_asset_or_rsl_bytes_fail_before_root(rig, capsys) -> None:
    rig.asset.write_bytes(b"changed")
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert "USD digest differs" in capsys.readouterr().err
    assert not rig.root.exists()


def test_asset_package_symlink_fails_before_root(rig, capsys) -> None:
    (rig.asset.parent / "external-link").symlink_to(rig.wheel)
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert "non-regular entry" in capsys.readouterr().err
    assert not rig.root.exists()


def test_ready_workload_must_retain_exact_gpu_lifetime_flock(
    rig, tmp_path: Path
) -> None:
    proc = tmp_path / "proc" / "1234"
    (proc / "fd").mkdir(parents=True)
    (proc / "fdinfo").mkdir()
    descriptor = 3
    (proc / "fd" / str(descriptor)).symlink_to(rig.lock)
    info = proc / "fdinfo" / str(descriptor)
    info.write_text(
        "pos:\t0\nflags:\t02100002\n"
        "lock:\t1: FLOCK  ADVISORY  WRITE 0 00:3f:123 0 EOF\n",
        encoding="utf-8",
    )

    rig.module._verify_inherited_gpu_lock(
        proc=proc, descriptor=descriptor, lock_file=rig.lock
    )
    info.write_text("pos:\t0\nflags:\t02100002\n", encoding="utf-8")
    with pytest.raises(rig.module.LaunchError, match="lifetime flock is absent"):
        rig.module._verify_inherited_gpu_lock(
            proc=proc, descriptor=descriptor, lock_file=rig.lock
        )
