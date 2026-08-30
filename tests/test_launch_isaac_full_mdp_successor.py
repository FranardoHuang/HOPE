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
PPO_RECIPE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "action_ball_full_mdp_ppo_recipe.py"
)
KIT_LAUNCHER = Path(
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
BALL_PHYSICS = Path("configs/ball_physics_optitrack_20260730.yaml")
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
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    base = tmp_path.resolve()
    monkeypatch.setenv("HOPE_ISAAC_ACCEPT_EULA", "Y")
    monkeypatch.setenv("HOPE_ISAAC_PRIVACY_CONSENT", "Y")
    repo = base / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    train = repo / TRAIN
    train.parent.mkdir(parents=True)
    train.write_text("# train placeholder\n", encoding="utf-8")
    recipe = repo / PPO_RECIPE
    recipe.parent.mkdir(parents=True)
    shutil.copy2(PROJECT / PPO_RECIPE, recipe)
    ball_physics = repo / BALL_PHYSICS
    ball_physics.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / BALL_PHYSICS, ball_physics)
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
    sys.modules.pop("_hope_isaac_full_mdp_ppo_recipe", None)
    request.addfinalizer(
        lambda: sys.modules.pop("_hope_isaac_full_mdp_ppo_recipe", None)
    )
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
    opengl = base / "opengl"
    opengl.mkdir()
    (opengl / "libOpenGL.so.0.0.0").write_bytes(b"exact-opengl")
    (opengl / "libOpenGL.so.0").symlink_to("libOpenGL.so.0.0.0")
    glu = base / "glu"
    glu.mkdir()
    (glu / "libGLU.so.1.3.1").write_bytes(b"exact-glu")
    (glu / "libGLU.so.1").symlink_to("libGLU.so.1.3.1")
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
            "--opengl-lib-dir", str(opengl),
            "--glu-lib-dir", str(glu),
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
        opengl=opengl,
        glu=glu,
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
        "num_envs=512",
        f"run_name={NAMESPACE}-DIAGNOSTIC_UNAUTHORIZED",
        "checkpoint_path=null",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false",
        "action_ball_dynamic_ready_bootstrap=true",
        (
            "action_ball_dynamic_ready_artifact_path="
            f"{(rig.repo / rig.module.DYNAMIC_READY_ARTIFACT_RELATIVE).resolve()}"
        ),
        (
            "action_ball_dynamic_ready_artifact_sha256="
            f"{rig.module.DYNAMIC_READY_ARTIFACT_SHA256}"
        ),
        (
            "action_ball_dynamic_ready_nominal_receipt_path="
            f"{(rig.repo / rig.module.DYNAMIC_READY_RECEIPT_RELATIVE).resolve()}"
        ),
        (
            "action_ball_dynamic_ready_nominal_receipt_sha256="
            f"{rig.module.DYNAMIC_READY_RECEIPT_SHA256}"
        ),
        f"hydra.run.dir={rig.root / 'training' / 'hydra'}",
    ]
    joined = " ".join(payload["argv"])
    assert "task=HOPEPingPongActionBallFullMdpA" in joined
    assert "num_envs=512" in joined
    assert "DIAGNOSTIC_UNAUTHORIZED" in joined
    assert "max_iterations=" not in joined
    assert "save_interval=" not in joined
    assert "action_ball_full_mdp_rate_probe" not in joined
    assert payload["runtime_env"]["PYTHONPATH"].startswith("/proc/self/fd/18:")
    assert payload["runtime_env"]["LD_LIBRARY_PATH"] == (
        f"{rig.opengl}:{rig.glu}"
    )
    assert payload["gl_runtime"] == {
        "ld_library_path": f"{rig.opengl}:{rig.glu}",
        "libraries": {
            "opengl": {
                "real_path": str(rig.opengl / "libOpenGL.so.0.0.0"),
                "sha256": _sha(rig.opengl / "libOpenGL.so.0.0.0"),
                "soname_path": str(rig.opengl / "libOpenGL.so.0"),
                "soname_target": "libOpenGL.so.0.0.0",
            },
            "glu": {
                "real_path": str(rig.glu / "libGLU.so.1.3.1"),
                "sha256": _sha(rig.glu / "libGLU.so.1.3.1"),
                "soname_path": str(rig.glu / "libGLU.so.1"),
                "soname_target": "libGLU.so.1.3.1",
            },
        },
    }
    assert payload["runtime_env"]["HOPE_ACTION_BALL_FULL_MDP_LOG_ROOT"] == str(
        rig.root / "training"
    )
    assert payload["runtime_env"]["HOPE_BALL_PHYSICS_YAML"] == str(
        rig.repo / BALL_PHYSICS
    )
    assert payload["runtime_env"]["HOME"] == str(rig.root / "home")
    assert payload["runtime_env"]["CUDA_CACHE_PATH"] == str(
        rig.root / "cuda_cache"
    )
    assert payload["ppo_recipe"] == (
        rig.module.FULL_MDP_PPO_RECIPE.execution_recipe()
    )
    assert payload["ppo_recipe_sha256"] == (
        rig.module.FULL_MDP_PPO_RECIPE.recipe_sha256()
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


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("HOPE_ISAAC_ACCEPT_EULA", None, "machine provisioning"),
        ("HOPE_ISAAC_ACCEPT_EULA", "N", "machine provisioning"),
        ("HOPE_ISAAC_PRIVACY_CONSENT", None, "must be Y or N"),
        ("HOPE_ISAAC_PRIVACY_CONSENT", "maybe", "must be Y or N"),
    ),
)
def test_operator_runtime_choices_must_be_machine_provisioned_before_root(
    rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    name: str, value: str | None, message: str,
) -> None:
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert message in capsys.readouterr().err
    assert not rig.root.exists()


def test_machine_provisioned_privacy_opt_out_reaches_only_the_child(
    rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOPE_ISAAC_PRIVACY_CONSENT", "N")
    assert rig.module.main(rig.argv(dry=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime_env"]["ACCEPT_EULA"] == "Y"
    assert payload["runtime_env"]["PRIVACY_CONSENT"] == "N"
    assert "ACCEPT_EULA" not in payload["launcher_env"]
    assert "PRIVACY_CONSENT" not in payload["launcher_env"]
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


def test_dry_run_profile_probe_is_explicit_finite_completion_mode(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(
        rig.argv(dry=True)
        + ["--diagnostic-profile-probe", "--profile-updates", "12"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"][-1] == (
        "task.action_ball_full_mdp_profile_probe=true"
    )
    assert payload["runtime_env"][
        "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES"
    ] == "12"
    assert payload["launcher_env"]["KIT_WAIT_FOR_COMPLETION"] == "1"
    assert payload["launcher_env"]["KIT_COMPLETION_TIMEOUT_S"] == "7200"
    assert not rig.root.exists()


def test_dry_run_fixed_action_probe_uses_its_own_boot_marker(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(
        rig.argv(dry=True) + ["--diagnostic-fixed-action-probe"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"][-1] == (
        "task.action_ball_full_mdp_fixed_action_probe_output_path="
        f"{rig.root / 'fixed-action-probe'}"
    )
    assert payload["launcher_env"]["KIT_BOOT_MARKER"] == (
        rig.module.FIXED_ACTION_PROBE_BOOT_MARKER
    )
    assert payload["launcher_env"]["KIT_WAIT_FOR_COMPLETION"] == "1"
    assert payload["launcher_env"]["KIT_COMPLETION_TIMEOUT_S"] == "7200"
    assert not rig.root.exists()


def test_profile_probe_requires_budget_and_is_rate_exclusive_before_root(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(
        rig.argv(dry=True) + ["--diagnostic-profile-probe"]
    ) == 2
    assert "requires positive" in capsys.readouterr().err
    assert rig.module.main(
        rig.argv(dry=True)
        + [
            "--diagnostic-rate-probe",
            "--diagnostic-profile-probe",
            "--profile-updates",
            "1",
        ]
    ) == 2
    assert "mutually exclusive" in capsys.readouterr().err
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
    candidate = rig.module.FULL_MDP_PPO_RECIPE.execution_recipe()
    rate_execution = rig.module._rate_execution_recipe()
    assert payload["candidate_production_execution_recipe"] == candidate
    assert payload["candidate_production_execution_recipe_sha256"] == (
        rig.module.FULL_MDP_PPO_RECIPE.recipe_sha256()
    )
    assert payload["rate_execution_recipe"] == rate_execution
    assert payload["rate_execution_recipe_sha256"] == (
        rig.module._canonical_payload_sha256(rate_execution)
    )
    assert payload["rate_execution_recipe_sha256"] != (
        payload["candidate_production_execution_recipe_sha256"]
    )
    assert "ppo_recipe_sha256" not in payload

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
    recipe = rig.module.FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
    log = tmp_path / "run.log"
    rows = [
        rig.module._expected_rate_probe_marker(),
        rig.module._expected_ppo_recipe_marker(),
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
    assert payload["learning_recipe_sha256"] == recipe
    assert payload["kind"] == "action_ball_isaac_full_mdp_h48_rate_probe_v2"
    assert payload["schema_version"] == 2
    assert payload["shape"] == {
        "num_envs": 512,
        "num_steps_per_env": 48,
        "updates": 61,
    }
    candidate = rig.module.FULL_MDP_PPO_RECIPE.execution_recipe()
    rate_execution = rig.module._rate_execution_recipe()
    assert payload["candidate_production_execution_recipe"] == candidate
    assert payload["candidate_production_execution_recipe_sha256"] == (
        rig.module.FULL_MDP_PPO_RECIPE.recipe_sha256()
    )
    assert payload["rate_execution_recipe"] == rate_execution
    assert rate_execution["effective_runner"] == {
        "num_envs": 512,
        "num_steps_per_env": 48,
        "max_iterations": 61,
        "save_interval": 2_000,
    }
    assert rate_execution["runner_overrides"]["max_iterations"] == {
        "candidate_production": 100_000,
        "rate_execution": 61,
    }
    assert rate_execution["diagnostic_overrides"] == {
        "warmup_updates": 10,
        "measured_updates": 50,
        "tail_updates": 1,
        "profiler_enabled": False,
        "diagnostic_unauthorized": True,
        "formal_evidence": False,
        "checkpoint_authority": False,
        "resume_authority": False,
    }
    assert payload["rate_execution_recipe_sha256"] == (
        rig.module._canonical_payload_sha256(rate_execution)
    )
    assert payload["rate_execution_recipe_sha256"] != (
        payload["candidate_production_execution_recipe_sha256"]
    )
    assert "action_ball_full_mdp_ppo_execution_sha256" not in payload
    assert len(payload["raw_update_seconds"]) == 61
    assert payload["measured_update_seconds"] == pytest.approx(
        [1.0 + update / 100.0 for update in range(10, 60)]
    )
    assert payload["update_seconds_p50"] == pytest.approx(1.345)
    assert payload["update_seconds_p90"] == pytest.approx(1.541)

    receipt = tmp_path / "rate.json"
    rig.module._write_diagnostic_receipt(receipt, payload)
    assert json.loads(receipt.read_text()) == payload
    with pytest.raises(rig.module.LaunchError, match="no-clobber"):
        rig.module._write_diagnostic_receipt(receipt, payload)


def test_profile_probe_receipt_requires_exact_rows_and_marker(
    rig, tmp_path: Path
) -> None:
    rows = [
        rig.module._expected_profile_probe_marker(2),
        rig.module._expected_ppo_recipe_marker(),
    ]
    profiles = []
    for update in range(2):
        profile = {
            "schema_version": 2,
            "update": update,
            "profile_update_ordinal": update + 1,
            "requested_profile_updates": 2,
            "rollout_call_count_exact": True,
            "speed_evidence_eligible": False,
            "segments": {"env_step_total": {"calls": 48}},
        }
        profiles.append(profile)
        rows.append(
            rig.module.PROFILE_JSON_PREFIX
            + json.dumps(profile, separators=(",", ":"), sort_keys=True)
        )
    log = tmp_path / "profile.log"
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = rig.module._profile_probe_payload(
        run_log=log,
        source_commit="c" * 40,
        namespace=NAMESPACE,
        gpu_index=0,
        gpu_uuid=UUID,
        requested_updates=2,
    )
    assert payload["kind"] == "action_ball_isaac_full_mdp_h48_profile_probe_v1"
    assert payload["diagnostic_unauthorized"] is True
    assert payload["formal_evidence"] is False
    assert payload["speed_evidence"] is False
    assert payload["profiles"] == profiles

    rows.pop()
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(rig.module.LaunchError, match="rows differ"):
        rig.module._profile_probe_payload(
            run_log=log,
            source_commit="c" * 40,
            namespace=NAMESPACE,
            gpu_index=0,
            gpu_uuid=UUID,
            requested_updates=2,
        )

    rows.append(
        rig.module.PROFILE_JSON_PREFIX
        + json.dumps(profiles[-1], separators=(",", ":"), sort_keys=True)
        .replace('"schema_version":2', '"bad":NaN,"schema_version":2')
    )
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(rig.module.LaunchError, match="not finite JSON"):
        rig.module._profile_probe_payload(
            run_log=log,
            source_commit="c" * 40,
            namespace=NAMESPACE,
            gpu_index=0,
            gpu_uuid=UUID,
            requested_updates=2,
        )


@pytest.mark.parametrize("missing", ("update", "recipe"))
def test_rate_probe_parser_rejects_missing_update_or_recipe(
    rig, tmp_path: Path, missing: str
) -> None:
    log = tmp_path / "run.log"
    rows = [rig.module._expected_rate_probe_marker()]
    if missing != "recipe":
        rows.append(rig.module._expected_ppo_recipe_marker())
    update_count = 60 if missing == "update" else 61
    for update in range(update_count):
        rows.extend((
            f"Learning iteration {update}/61",
            "Iteration time: 1.00s",
        ))
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    expected_error = "61-update" if missing == "update" else "recipe identity"
    with pytest.raises(rig.module.LaunchError, match=expected_error):
        rig.module._rate_probe_payload(
            run_log=log,
            source_commit="b" * 40,
            namespace=NAMESPACE,
            gpu_index=0,
            gpu_uuid=UUID,
        )


@pytest.mark.parametrize(
    ("updates", "warmup", "measured", "tail"),
    (
        (61, 9, 51, 1),
        (61, 10, 49, 2),
        (61, 11, 49, 1),
        (62, 10, 50, 2),
    ),
)
def test_rate_probe_parser_rejects_noncanonical_equal_sum_windows(
    rig, tmp_path: Path, updates: int, warmup: int, measured: int, tail: int
) -> None:
    log = tmp_path / "run.log"
    rows = [
        "[train.py] FULLMDP_H48_RATE_PROBE: diagnostic_unauthorized=true "
        f"updates={updates} warmup={warmup} measured={measured} tail={tail} "
        "profiler=off formal_evidence=false checkpoint_authority=false",
        rig.module._expected_ppo_recipe_marker(),
    ]
    for update in range(updates):
        rows.extend((
            f"Learning iteration {update}/{updates}",
            "Iteration time: 1.00s",
        ))
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(
        rig.module.LaunchError, match="exact 61/10/50/1"
    ):
        rig.module._rate_probe_payload(
            run_log=log,
            source_commit="b" * 40,
            namespace=NAMESPACE,
            gpu_index=0,
            gpu_uuid=UUID,
        )


@pytest.mark.parametrize(
    "mutation",
    ("bad_authority", "duplicate_budget", "wrong_recipe_shape", "duplicate_recipe"),
)
def test_rate_probe_parser_requires_one_exact_child_marker(
    rig, tmp_path: Path, mutation: str
) -> None:
    budget = rig.module._expected_rate_probe_marker()
    recipe = rig.module._expected_ppo_recipe_marker()
    if mutation == "bad_authority":
        budget = budget.replace(
            "diagnostic_unauthorized=true", "diagnostic_unauthorized=false"
        ).replace("profiler=off", "profiler=on")
    elif mutation == "wrong_recipe_shape":
        recipe = recipe.replace(
            f"N={rig.module.FULL_MDP_PPO_RECIPE.num_envs}", "N=4096"
        )
    rows = [budget, recipe]
    if mutation == "duplicate_budget":
        rows.append(budget)
    elif mutation == "duplicate_recipe":
        rows.append(recipe)
    for update in range(61):
        rows.extend((
            f"Learning iteration {update}/61",
            "Iteration time: 1.00s",
        ))
    log = tmp_path / "run.log"
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(rig.module.LaunchError, match="budget|recipe identity"):
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
        "ACCEPT_EULA=Y",
        "PRIVACY_CONSENT=Y",
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


@pytest.mark.parametrize("failure", ("opengl_file", "glu_link"))
def test_wrong_gl_runtime_fails_before_root(rig, failure: str, capsys) -> None:
    if failure == "opengl_file":
        (rig.opengl / "libOpenGL.so.0.0.0").unlink()
        (rig.opengl / "libOpenGL.so.0.0.0").mkdir()
        expected = "OpenGL runtime must be one canonical regular file"
    else:
        (rig.glu / "libGLU.so.1").unlink()
        (rig.glu / "libGLU.so.1").symlink_to("wrong")
        expected = "GLU SONAME link differs"
    assert rig.module.main(rig.argv(dry=True)) == 2
    assert expected in capsys.readouterr().err
    assert not rig.root.exists()


def test_portable_gl_bytes_are_accepted_and_reported(rig, capsys) -> None:
    runtime = rig.opengl / "libOpenGL.so.0.0.0"
    runtime.write_bytes(b"different-supported-platform-build")
    assert rig.module.main(rig.argv(dry=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gl_runtime"]["libraries"]["opengl"]["sha256"] == _sha(runtime)
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
