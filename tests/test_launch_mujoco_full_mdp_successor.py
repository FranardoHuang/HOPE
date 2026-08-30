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
    "configs/action_ball_n1_measured_a3p0807_20260828/"
    "take061.local_closest_robust_feasible.dynamic_ready.v2.json"
)
PLANT_XML = PROJECT / (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3p_pingpong_0807/a3p_pingpong_0807.xml"
)
RUNNER = Path("hope_training/whole_body_tracking/mjlab_lane/"
              "mujoco_gpu_ac_full_mdp_wait_rsl3.py")
PPO_RECIPE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "action_ball_full_mdp_ppo_recipe.py"
)
PLANT_CONTRACT = Path("hope_training/whole_body_tracking/mjlab_lane/"
                      "mujoco_full_mdp_plant_contract.py")
PLANT_MANIFEST = Path(
    "configs/a3p_p1_0807_mujoco_identity_v1_20260828.json"
)
BALL_PHYSICS = Path("configs/ball_physics_optitrack_20260730.yaml")
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
    recipe = repo / PPO_RECIPE
    recipe.parent.mkdir(parents=True)
    shutil.copy2(PROJECT / PPO_RECIPE, recipe)
    shutil.copy2(PROJECT / PLANT_CONTRACT, repo / PLANT_CONTRACT)
    manifest = repo / PLANT_MANIFEST
    manifest.parent.mkdir(parents=True)
    shutil.copy2(PROJECT / PLANT_MANIFEST, manifest)
    shutil.copy2(PROJECT / BALL_PHYSICS, repo / BALL_PHYSICS)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Launcher Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    sys.modules.pop("_hope_mujoco_full_mdp_plant_contract", None)
    sys.modules.pop("_hope_mujoco_full_mdp_ppo_recipe_launcher", None)
    request.addfinalizer(
        lambda: sys.modules.pop("_hope_mujoco_full_mdp_plant_contract", None)
    )
    request.addfinalizer(
        lambda: sys.modules.pop(
            "_hope_mujoco_full_mdp_ppo_recipe_launcher", None
        )
    )
    module = _load(repo / "scripts" / SCRIPT.name)
    monkeypatch.setattr(module, "_available_cpu_ids", lambda: set(range(128)))

    tools = base / "tools"
    tools.mkdir()
    python = _executable(tools / "fake-python", """#!/usr/bin/env python3
import fcntl, json, os, pathlib, sys, time
root = pathlib.Path.cwd()
base = root.parents[2]
lock_stat = os.stat(base / "gpu.lock", follow_symlinks=False)
inherited_lock_fds = []
for candidate in range(3, 256):
    try:
        row = os.fstat(candidate)
    except OSError:
        continue
    if candidate > 2 and (row.st_dev, row.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
        inherited_lock_fds.append(candidate)
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
         "HOPE_BALL_PHYSICS_YAML",
         "TMPDIR", "PYTHONPYCACHEPREFIX", "PATH", "HOME", "XDG_CACHE_HOME",
         "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "LANG", "LC_ALL", "LC_CTYPE",
         "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "HOPE_GEOMETRY_PY",
         "LD_PRELOAD", "ACTIONBALL_AUDIT_AMBIENT", "OMP_NUM_THREADS", "CUDA_LAUNCH_BLOCKING"]
warp_cache = pathlib.Path(os.environ["WARP_CACHE_PATH"])
cuda_cache = pathlib.Path(os.environ["CUDA_CACHE_PATH"])
payload = {"argv": sys.argv[1:], "cwd": os.getcwd(), "lock_held": held,
           "inherited_lock_fds": inherited_lock_fds,
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
    taskset = _executable(tools / "taskset", """#!/usr/bin/env python3
import os, sys
if len(sys.argv) < 4 or sys.argv[1] != "-c":
    sys.exit(90)
try:
    os.execv(sys.argv[3], sys.argv[3:])
except OSError as exc:
    print(f"taskset child exec failed: {exc}", file=sys.stderr)
    sys.exit(126)
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
    plant_xml = base / "a3p_pingpong_0807.xml"
    shutil.copy2(PLANT_XML, plant_xml)
    record, started, release = (
        run_root / name for name in ("record.json", "started", "release")
    )
    monkeypatch.setattr(module, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(module, "NVIDIA_SMI", nvidia)
    monkeypatch.setattr(module, "TASKSET", taskset)
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
             ready: Path = ready_pose, plant: Path = plant_xml,
             cpu_affinity: str | None = None) -> list[str]:
        result = [
            "--python", str(executable), "--run-root", str(root),
            "--namespace", NAMESPACE, "--gpu-index", "2",
            "--expected-gpu-uuid", expected, "--lock-file", str(lock),
            "--ready-pose", str(ready), "--plant-xml", str(plant),
        ]
        if cpu_affinity is not None:
            result.extend(("--cpu-affinity", cpu_affinity))
        return result + (["--dry-run"] if dry else [])

    return SimpleNamespace(
        module=module, repo=repo, runner=runner, python=python, nvidia=nvidia,
        taskset=taskset,
        workspace=workspace, root=run_root, lock=lock, record=record,
        ready_pose=ready_pose, plant_xml=plant_xml, started=started,
        release=release, argv=argv,
    )


def _expected_trainer(rig, commit: str) -> list[str]:
    root = rig.root
    recipe = rig.module.FULL_MDP_PPO_RECIPE
    return [
        str(rig.python), str(rig.runner), "--full-a",
        "--num-envs", str(recipe.num_envs),
        "--num-updates", str(recipe.max_iterations),
        "--evidence-jsonl", str(root / "evidence.jsonl"),
        "--snapshot-dir", str(root / "snapshots"),
        "--completion-json", str(root / "completion.json"),
        "--source-commit", commit, "--run-namespace", NAMESPACE,
        "--mujoco-warp-runtime-site", str(root / "runtime_site"),
        "--save-interval", str(recipe.save_interval),
    ]


def _expected_rate_trainer(rig, commit: str) -> list[str]:
    root = rig.root
    recipe = rig.module.FULL_MDP_PPO_RECIPE
    return [
        str(rig.python), str(rig.runner), "--full-a",
        "--num-envs", str(recipe.num_envs), "--num-updates", "61",
        "--evidence-jsonl", str(root / "evidence.jsonl"),
        "--source-commit", commit, "--run-namespace", NAMESPACE,
        "--mujoco-warp-runtime-site", str(root / "runtime_site"),
        "--save-interval", str(recipe.save_interval),
        "--diagnostic-rate-probe",
    ]


def _runner_rate_payload(rig, commit: str) -> dict[str, object]:
    recipe = rig.module.FULL_MDP_PPO_RECIPE
    rate_execution = rig.module._rate_execution_recipe()
    measured = [1.0 + index / 100.0 for index in range(50)]
    measured_wall = sum(measured)
    measured_transitions = 50 * recipe.num_steps_per_env * recipe.num_envs
    return {
        "kind": "action_ball_mujoco_full_mdp_h48_rate_probe_v1",
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "formal_evidence": False,
        "safety_gate": False,
        "source_commit": commit,
        "namespace": NAMESPACE,
        "rsl_rl_version": "3.1.2",
        "ppo_update_calls": 61,
        "environment_steps": 61 * recipe.num_steps_per_env,
        "transitions": 61 * recipe.num_steps_per_env * recipe.num_envs,
        "policy_width": 215,
        "critic_width": 231,
        "learning_recipe_sha256": recipe.learning_recipe_sha256(),
        "task_lifecycle": "full_a_diagnostic_rate_probe",
        "candidate_production_execution_recipe": recipe.execution_recipe(),
        "candidate_production_execution_recipe_sha256": recipe.recipe_sha256(),
        "rate_execution_recipe": rate_execution,
        "rate_execution_recipe_sha256": (
            rig.module._canonical_payload_sha256(rate_execution)
        ),
        "rate_probe": {
            "warmup_updates": 10,
            "measured_updates": 50,
            "tail_updates": 1,
            "total_wall_seconds": measured_wall + 11.0,
            "measured_wall_seconds": measured_wall,
            "measured_update_seconds": measured,
            "measured_transitions": measured_transitions,
            "measured_transitions_per_second": (
                measured_transitions / measured_wall
            ),
            "update_seconds_p50": rig.module.statistics.median(measured),
            "update_seconds_p90": rig.module.statistics.quantiles(
                measured, n=10, method="inclusive"
            )[8],
        },
    }


def _expected_child(
    rig, commit: str, *, cpu_affinity: str | None = None
) -> list[str]:
    trainer = _expected_trainer(rig, commit)
    if cpu_affinity is None:
        return trainer
    return [str(rig.taskset), "-c", cpu_affinity, *trainer]


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
        "03e2590916f781e581c4a0ff6dbe305ab3a2471685b816cc32a015400816deba"
    )


def test_rate_execution_identity_is_exact_across_both_launchers_and_mu_runner() -> None:
    script = r"""
import importlib.util, json, pathlib, sys
modules = []
for index, raw in enumerate(sys.argv[1:]):
    path = pathlib.Path(raw)
    spec = importlib.util.spec_from_file_location(f'rate_identity_{index}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules.append(module)
payloads = [module._rate_execution_recipe() for module in modules]
hashes = [module._canonical_payload_sha256(payload) for module, payload in zip(modules, payloads)]
print(json.dumps({'payloads': payloads, 'hashes': hashes,
                  'candidate': modules[0].FULL_MDP_PPO_RECIPE.recipe_sha256()},
                 sort_keys=True))
"""
    runner = PROJECT / RUNNER
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(PROJECT / "scripts" / "launch_isaac_full_mdp_successor.py"),
            str(SCRIPT),
            str(runner),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["payloads"][0] == result["payloads"][1] == result["payloads"][2]
    assert len(set(result["hashes"])) == 1
    assert result["hashes"][0] != result["candidate"]
    assert result["payloads"][0]["runner_overrides"] == {
        "max_iterations": {
            "candidate_production": 100_000,
            "rate_execution": 61,
        }
    }


def test_good_ready_pose_dry_run_reports_exact_binding_without_resources(
    rig, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rig.module, "_gpu_is_free", lambda *_: pytest.fail("GPU occupancy queried")
    )
    monkeypatch.setattr(
        rig.module,
        "_available_cpu_ids",
        lambda: pytest.fail("CPU affinity queried without an explicit flag"),
    )
    monkeypatch.setattr(rig.module, "TASKSET", rig.taskset.with_name("missing"))
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
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "ACTIONBALL_READY_POSE": str(rig.ready_pose),
                    "HOPE_BALL_PHYSICS_YAML": str(rig.repo / BALL_PHYSICS),
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
        "ppo_recipe": rig.module.FULL_MDP_PPO_RECIPE.execution_recipe(),
        "ppo_recipe_sha256": (
            rig.module.FULL_MDP_PPO_RECIPE.recipe_sha256()
        ),
        "cpu_affinity": None,
        "cpu_affinity_source": None,
        "cpu_ids": [],
    }
    assert output.err == ""
    assert not rig.root.parent.exists()


def test_diagnostic_rate_dry_run_binds_actual_61_update_identity_only(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rig.module.main(
        rig.argv(dry=True) + ["--diagnostic-rate-probe"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    commit = _git(rig.repo, "rev-parse", "HEAD")
    assert payload["argv"] == _expected_rate_trainer(rig, commit)
    joined = " ".join(payload["argv"])
    assert "--snapshot-dir" not in joined
    assert "--completion-json" not in joined
    assert payload["candidate_production_execution_recipe"] == (
        rig.module.FULL_MDP_PPO_RECIPE.execution_recipe()
    )
    assert payload["candidate_production_execution_recipe_sha256"] == (
        rig.module.FULL_MDP_PPO_RECIPE.recipe_sha256()
    )
    rate_execution = rig.module._rate_execution_recipe()
    assert payload["rate_execution_recipe"] == rate_execution
    assert payload["rate_execution_recipe_sha256"] == (
        rig.module._canonical_payload_sha256(rate_execution)
    )
    assert payload["rate_execution_recipe_sha256"] != (
        payload["candidate_production_execution_recipe_sha256"]
    )
    assert rate_execution["runner_overrides"] == {
        "max_iterations": {
            "candidate_production": 100_000,
            "rate_execution": 61,
        }
    }
    assert "ppo_recipe_sha256" not in payload
    assert not rig.root.parent.exists()


def test_cpu_affinity_is_explicit_and_exact(rig) -> None:
    assert rig.module._cpu_affinity(None) == (None, ())
    assert rig.module._cpu_affinity("48-63,112-127") == (
        "48-63,112-127",
        (*range(48, 64), *range(112, 128)),
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "comma-separated"),
        ("1-", "comma-separated"),
        ("4-2", "reversed"),
        ("1,1", "duplicate"),
        ("0-1,1-2", "duplicate"),
        ("4096", "invalid"),
    ],
)
def test_malformed_explicit_cpu_affinity_fails(
    rig, value: str, message: str
) -> None:
    with pytest.raises(rig.module.LaunchError, match=message):
        rig.module._cpu_affinity(value)


def test_explicit_cpu_set_outside_launcher_cpuset_fails_before_root(
    rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rig.module, "_available_cpu_ids", lambda: {0, 1})
    assert rig.module.main(rig.argv(dry=True, cpu_affinity="2")) == 2
    assert "allowed CPU set" in capsys.readouterr().err
    assert not rig.root.parent.exists()


def test_explicit_cpu_affinity_dry_run_records_explicit_source(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    affinity = "48-63,112-127"
    assert rig.module.main(
        rig.argv(dry=True, cpu_affinity=affinity)
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    commit = _git(rig.repo, "rev-parse", "HEAD")
    assert payload["argv"] == _expected_child(
        rig, commit, cpu_affinity=affinity
    )
    assert payload["cpu_affinity"] == affinity
    assert payload["cpu_affinity_source"] == "explicit_cli"
    assert payload["cpu_ids"] == [*range(48, 64), *range(112, 128)]
    assert "cpu_affinity_gpu_index" not in payload


def test_explicit_cpu_affinity_wraps_exact_child_without_preexec(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    commit = _git(rig.repo, "rev-parse", "HEAD")

    class Child:
        @staticmethod
        def wait() -> int:
            return 13

    def popen(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return Child()

    monkeypatch.setattr(rig.module, "_source_commit", lambda: commit)
    monkeypatch.setattr(rig.module, "_gpu_is_free", lambda *_: None)
    monkeypatch.setattr(rig.module.subprocess, "Popen", popen)
    affinity = "48-63,112-127"
    args = rig.module.parse_args(rig.argv(cpu_affinity=affinity))
    assert rig.module.launch(args) == 13
    assert observed["argv"] == _expected_child(
        rig, commit, cpu_affinity=affinity
    )
    assert "preexec_fn" not in observed["kwargs"]


@pytest.mark.parametrize("kind", ["missing", "symlink", "nonexecutable"])
def test_invalid_taskset_fails_before_lock_gpu_or_root(
    rig, kind: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = rig.taskset.with_name(f"taskset-{kind}")
    if kind == "symlink":
        candidate.symlink_to(rig.taskset)
    elif kind == "nonexecutable":
        candidate.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(rig.module, "TASKSET", candidate)
    monkeypatch.setattr(
        rig.module, "_gpu_is_free", lambda *_: pytest.fail("GPU occupancy queried")
    )
    assert rig.module.main(
        rig.argv(cpu_affinity="48-63,112-127")
    ) == 2
    assert "taskset" in capsys.readouterr().err
    assert not rig.root.parent.exists()


def test_child_env_is_closed_and_rejects_ambient_inheritance(rig) -> None:
    contract = rig.module._env_contract(
        UUID,
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


def test_rate_receipt_parser_binds_runner_payload_and_is_no_clobber(
    rig, tmp_path: Path
) -> None:
    commit = _git(rig.repo, "rev-parse", "HEAD")
    child_payload = _runner_rate_payload(rig, commit)
    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        rig.module.RATE_PROBE_STDOUT_MARKER
        + json.dumps(child_payload, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    receipt_payload = rig.module._rate_probe_receipt_payload(
        stdout_log=stdout,
        source_commit=commit,
        namespace=NAMESPACE,
        gpu_index=2,
        gpu_uuid=UUID,
    )
    assert receipt_payload["gpu"] == {"index": 2, "uuid": UUID}
    assert receipt_payload["kind"] == (
        "action_ball_mujoco_full_mdp_h48_rate_receipt_v1"
    )
    assert receipt_payload["runner_marker"] == {
        "kind": "action_ball_mujoco_full_mdp_h48_rate_probe_v1",
        "schema_version": 1,
    }
    assert receipt_payload["stdout_log"] == {
        "path": str(stdout),
        "sha256": rig.module._sha256(stdout),
    }
    assert receipt_payload["rate_execution_recipe_sha256"] != (
        receipt_payload["candidate_production_execution_recipe_sha256"]
    )
    assert "action_ball_full_mdp_ppo_recipe_sha256" not in receipt_payload

    receipt = tmp_path / "diagnostic-rate-probe.json"
    rig.module._write_rate_probe_receipt(receipt, receipt_payload)
    assert json.loads(receipt.read_text()) == receipt_payload
    with pytest.raises(rig.module.LaunchError, match="no-clobber"):
        rig.module._write_rate_probe_receipt(receipt, receipt_payload)

    bad = json.loads(json.dumps(child_payload))
    bad["rate_execution_recipe_sha256"] = (
        bad["candidate_production_execution_recipe_sha256"]
    )
    stdout.write_text(
        rig.module.RATE_PROBE_STDOUT_MARKER + json.dumps(bad) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rig.module.LaunchError, match="identity differs"):
        rig.module._rate_probe_receipt_payload(
            stdout_log=stdout,
            source_commit=commit,
            namespace=NAMESPACE,
            gpu_index=2,
            gpu_uuid=UUID,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_authority", True),
        ("schema_version", True),
        ("diagnostic_unauthorized", 1),
        ("rsl_rl_version", "WRONG"),
        ("policy_width", 999),
        ("policy_width", 215.0),
        ("ppo_update_calls", 61.0),
        ("rate_probe.warmup_updates", 10.0),
        ("rate_execution_recipe.effective_runner.max_iterations", 61.0),
        ("rate_probe.measured_wall_seconds", 999.0),
        ("rate_probe.measured_transitions_per_second", -1.0),
        ("rate_probe.update_seconds_p50", 999.0),
        ("rate_probe.update_seconds_p90", -1.0),
    ),
)
def test_rate_receipt_parser_rejects_extra_authority_and_forged_metrics(
    rig, tmp_path: Path, field: str, value
) -> None:
    commit = _git(rig.repo, "rev-parse", "HEAD")
    child_payload = _runner_rate_payload(rig, commit)
    target = child_payload
    parts = field.split(".")
    for name in parts[:-1]:
        target = target[name]
    target[parts[-1]] = value
    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        rig.module.RATE_PROBE_STDOUT_MARKER
        + json.dumps(child_payload)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rig.module.LaunchError):
        rig.module._rate_probe_receipt_payload(
            stdout_log=stdout,
            source_commit=commit,
            namespace=NAMESPACE,
            gpu_index=2,
            gpu_uuid=UUID,
        )


def test_rate_receipt_parser_rejects_duplicate_json_keys(rig, tmp_path: Path) -> None:
    commit = _git(rig.repo, "rev-parse", "HEAD")
    child_payload = _runner_rate_payload(rig, commit)
    body = json.dumps(child_payload, separators=(",", ":"))
    body = body[:-1] + ',"kind":"duplicate"}'
    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        rig.module.RATE_PROBE_STDOUT_MARKER + body + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rig.module.LaunchError, match="finite JSON"):
        rig.module._rate_probe_receipt_payload(
            stdout_log=stdout,
            source_commit=commit,
            namespace=NAMESPACE,
            gpu_index=2,
            gpu_uuid=UUID,
        )


def test_real_rate_launch_writes_actual_execution_receipt(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    commit = _git(rig.repo, "rev-parse", "HEAD")
    child_payload = _runner_rate_payload(rig, commit)
    marker = (
        rig.module.RATE_PROBE_STDOUT_MARKER
        + json.dumps(child_payload, sort_keys=True, separators=(",", ":"))
    )
    rate_python = _executable(
        rig.python.with_name("rate-python"),
        "#!/usr/bin/env python3\n"
        f"print({marker!r}, flush=True)\n",
    )
    assert rig.module.main(
        rig.argv(executable=rate_python) + ["--diagnostic-rate-probe"]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "DIAGNOSTIC_RATE_PROBE_COMPLETE"
    receipt_path = rig.root / "diagnostic-rate-probe.json"
    assert summary["rate_receipt"] == str(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["kind"] == (
        "action_ball_mujoco_full_mdp_h48_rate_receipt_v1"
    )
    assert receipt["runner_marker"] == {
        "kind": "action_ball_mujoco_full_mdp_h48_rate_probe_v1",
        "schema_version": 1,
    }
    assert receipt["candidate_production_execution_recipe_sha256"] == (
        rig.module.FULL_MDP_PPO_RECIPE.recipe_sha256()
    )
    assert receipt["rate_execution_recipe_sha256"] == (
        rig.module._canonical_payload_sha256(
            rig.module._rate_execution_recipe()
        )
    )
    assert receipt["rate_execution_recipe_sha256"] != (
        receipt["candidate_production_execution_recipe_sha256"]
    )
    assert receipt["gpu"] == {"index": 2, "uuid": UUID}
    assert not (rig.root / "completion.json").exists()


def test_real_rate_launch_rejects_production_hash_as_execution_hash(
    rig, capsys: pytest.CaptureFixture[str]
) -> None:
    commit = _git(rig.repo, "rev-parse", "HEAD")
    child_payload = _runner_rate_payload(rig, commit)
    child_payload["rate_execution_recipe_sha256"] = (
        child_payload["candidate_production_execution_recipe_sha256"]
    )
    marker = rig.module.RATE_PROBE_STDOUT_MARKER + json.dumps(child_payload)
    rate_python = _executable(
        rig.python.with_name("bad-rate-python"),
        "#!/usr/bin/env python3\n"
        f"print({marker!r}, flush=True)\n",
    )
    assert rig.module.main(
        rig.argv(executable=rate_python) + ["--diagnostic-rate-probe"]
    ) == 2
    assert "rate receipt identity differs" in capsys.readouterr().err
    assert not (rig.root / "diagnostic-rate-probe.json").exists()


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
    assert record["argv"] == _expected_trainer(
        rig, _git(rig.repo, "rev-parse", "HEAD")
    )[1:]
    assert record["cwd"] == str(rig.root)
    assert (rig.root / "MUJOCO_LOG.TXT").read_text() == "process-local runtime log"
    assert record["warp_cache_is_dir"] is True
    assert record["cuda_cache_is_dir"] is True
    assert (rig.root / "warp_cache" / "fake-compiled-kernel").read_text() == "run-owned cache"
    assert not (rig.repo / "MUJOCO_LOG.TXT").exists()
    assert record["lock_held"] is True
    assert len(record["inherited_lock_fds"]) == 1
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
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "ACTIONBALL_READY_POSE": str(rig.ready_pose),
        "HOPE_BALL_PHYSICS_YAML": str(rig.repo / BALL_PHYSICS),
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
