#!/usr/bin/env python3
"""Launch one fresh Isaac Full-A H48 run through the existing Kit boot owner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path("/workspace")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
TASKSET = Path("/usr/bin/taskset")
TRAIN_RELATIVE = Path("hope_training/whole_body_tracking/scripts/train.py")
PPO_RECIPE_RELATIVE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "action_ball_full_mdp_ppo_recipe.py"
)
KIT_LAUNCHER_RELATIVE = Path(
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
PINNED_ISAACLAB_COMMIT = "8320e0be5c0f2def58d5b19d308c6d2539d47cb2"
PINNED_RSL_WHEEL_SHA256 = (
    "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
)
PINNED_KIT_PYTHON_SHA256 = (
    "5ab9c6fa43fc97154473ba58c9feaf22a4d6134fd6b4dee7b6a4f2b4c3c2ae8f"
)
PINNED_A3_USD_SHA256 = (
    "a3cd382943ff9f70beecf88c729a6cc1c052a3c0a0cbffe91003ec319ab78140"
)
PINNED_OPENGL_SHA256 = (
    "9a0a6024499300f918ef1b42d581427cdb20bbc17a7d8239a4b7434833a98d4a"
)
PINNED_GLU_SHA256 = (
    "af791d1ee2acf25417f612290e634248fd716cf5da0374ba21160fb264eaeab4"
)
DYNAMIC_READY_ARTIFACT_RELATIVE = Path(
    "configs/action_ball_n1_measured_a3p0807_20260828/"
    "take061.local_closest_robust_feasible.dynamic_ready.v2.json"
)
DYNAMIC_READY_ARTIFACT_SHA256 = (
    "b88d93c311b439bd61296b3b3a84198200d9c6938980471071992ec52d8df18f"
)
DYNAMIC_READY_RECEIPT_RELATIVE = Path(
    "configs/action_ball_n1_measured_a3p0807_20260828/"
    "take061.local_closest_robust_feasible.nominal_hold.v1.json"
)
DYNAMIC_READY_RECEIPT_SHA256 = (
    "b861e09db8482ecec2ceb5cea2c794d1c1afb23d92295b414078ce50e9b14c6c"
)
RATE_PROBE_COMPLETION_TIMEOUT_S = 7200
FIXED_ACTION_PROBE_BOOT_MARKER = "FULLMDP_ISAAC_FIXED_ACTION_PROBE_STARTED"
RATE_PROBE_WARMUP_UPDATES = 10
RATE_PROBE_MEASURED_UPDATES = 50
RATE_PROBE_TAIL_UPDATES = 1
RATE_PROBE_NUM_UPDATES = (
    RATE_PROBE_WARMUP_UPDATES
    + RATE_PROBE_MEASURED_UPDATES
    + RATE_PROBE_TAIL_UPDATES
)
RATE_PROBE_BUDGET = (
    RATE_PROBE_NUM_UPDATES,
    RATE_PROBE_WARMUP_UPDATES,
    RATE_PROBE_MEASURED_UPDATES,
    RATE_PROBE_TAIL_UPDATES,
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RATE_PROBE_MARKER_TOKEN = "FULLMDP_H48_RATE_PROBE:"
PROFILE_PROBE_MARKER_TOKEN = "FULLMDP_H48_PROFILE_PROBE:"
PROFILE_JSON_PREFIX = "HOPE_ACTION_BALL_FULL_MDP_PROFILE_JSON="
PPO_RECIPE_MARKER_TOKEN = "full-MDP PPO recipe:"
LEARNING_ITERATION_RE = re.compile(
    r"^\s*Learning iteration\s+([0-9]+)/([0-9]+)\s*$"
)
ITERATION_TIME_RE = re.compile(
    r"^\s*Iteration time:\s*([0-9]+(?:\.[0-9]+)?)s\s*$"
)


class LaunchError(RuntimeError):
    pass


def _ppo_recipe_module():
    """Load the dependency-free typed recipe from this exact checkout."""

    source = (REPO_ROOT / PPO_RECIPE_RELATIVE).resolve()
    name = "_hope_isaac_full_mdp_ppo_recipe"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise LaunchError("cached FullMDP PPO recipe origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise LaunchError("cannot load FullMDP PPO recipe")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


FULL_MDP_PPO_RECIPE = (
    _ppo_recipe_module().ACTION_BALL_FULL_MDP_PPO_RECIPE
)


def _canonical_payload_sha256(payload: dict[str, object]) -> str:
    body = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _rate_execution_recipe(
    *,
    num_envs: int = FULL_MDP_PPO_RECIPE.num_envs,
    num_steps_per_env: int = FULL_MDP_PPO_RECIPE.num_steps_per_env,
    max_iterations: int = RATE_PROBE_NUM_UPDATES,
    save_interval: int = FULL_MDP_PPO_RECIPE.save_interval,
) -> dict[str, object]:
    """Return the backend-neutral, actual finite rate execution identity."""

    effective_runner = {
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
    }
    candidate_runner = {
        "num_envs": FULL_MDP_PPO_RECIPE.num_envs,
        "num_steps_per_env": FULL_MDP_PPO_RECIPE.num_steps_per_env,
        "max_iterations": FULL_MDP_PPO_RECIPE.max_iterations,
        "save_interval": FULL_MDP_PPO_RECIPE.save_interval,
    }
    return {
        "schema_version": 1,
        "kind": "action_ball_full_mdp_h48_rate_execution_v1",
        "candidate_production_execution_recipe_sha256": (
            FULL_MDP_PPO_RECIPE.recipe_sha256()
        ),
        "learning_recipe_sha256": (
            FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
        ),
        "effective_runner": effective_runner,
        "runner_overrides": {
            name: {
                "candidate_production": candidate_runner[name],
                "rate_execution": effective_runner[name],
            }
            for name in candidate_runner
            if candidate_runner[name] != effective_runner[name]
        },
        "diagnostic_overrides": {
            "warmup_updates": RATE_PROBE_WARMUP_UPDATES,
            "measured_updates": RATE_PROBE_MEASURED_UPDATES,
            "tail_updates": RATE_PROBE_TAIL_UPDATES,
            "profiler_enabled": False,
            "diagnostic_unauthorized": True,
            "formal_evidence": False,
            "checkpoint_authority": False,
            "resume_authority": False,
        },
    }


def _expected_rate_probe_marker() -> str:
    return (
        "[train.py] FULLMDP_H48_RATE_PROBE: diagnostic_unauthorized=true "
        f"updates={RATE_PROBE_NUM_UPDATES} "
        f"warmup={RATE_PROBE_WARMUP_UPDATES} "
        f"measured={RATE_PROBE_MEASURED_UPDATES} "
        f"tail={RATE_PROBE_TAIL_UPDATES} "
        "profiler=off formal_evidence=false checkpoint_authority=false"
    )


def _expected_profile_probe_marker(updates: int) -> str:
    return (
        "[train.py] FULLMDP_H48_PROFILE_PROBE: "
        "diagnostic_unauthorized=true "
        f"updates={updates} "
        "profiler=host-inclusive-no-cuda-sync "
        "speed_evidence=false formal_evidence=false "
        "checkpoint_authority=false"
    )


def _expected_ppo_recipe_marker() -> str:
    recipe = FULL_MDP_PPO_RECIPE
    return (
        "[train.py] full-MDP PPO recipe: "
        f"kind={recipe.kind} "
        f"N={recipe.num_envs} "
        f"H={recipe.num_steps_per_env} "
        f"updates={recipe.max_iterations} "
        f"save={recipe.save_interval} "
        f"epochs={recipe.num_learning_epochs} "
        f"minibatches={recipe.num_mini_batches} "
        f"gamma={recipe.gamma} "
        f"lambda={recipe.lam} "
        "effective_normalization=false "
        f"learning_recipe_sha256={recipe.learning_recipe_sha256()}"
    )


def _run_text(argv: list[str], label: str) -> str:
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LaunchError(f"cannot run {label}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LaunchError(f"{label} failed rc={result.returncode}: {detail}")
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise LaunchError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _source_commit() -> str:
    prefix = ["git", "--no-optional-locks", "-C", str(REPO_ROOT)]
    commit = _run_text(prefix + ["rev-parse", "HEAD"], "source git rev-parse").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise LaunchError("source HEAD is not one exact commit")
    status = _run_text(
        prefix + ["status", "--porcelain=v1", "--untracked-files=all"],
        "source git clean check",
    )
    if status:
        raise LaunchError("source checkout has tracked or untracked changes")
    return commit


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise LaunchError(f"{label} must be absolute")
    try:
        row = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LaunchError(f"{label} is missing") from exc
    if (
        not stat.S_ISREG(row.st_mode)
        or stat.S_ISLNK(row.st_mode)
        or resolved != path
        or (executable and not os.access(path, os.X_OK))
    ):
        raise LaunchError(f"{label} must be one canonical regular file")
    return path


def _canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise LaunchError(f"{label} must be absolute")
    try:
        row = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LaunchError(f"{label} is missing") from exc
    if not stat.S_ISDIR(row.st_mode) or stat.S_ISLNK(row.st_mode) or resolved != path:
        raise LaunchError(f"{label} must be one canonical directory")
    return path


def _validate_isaaclab(path: Path) -> Path:
    root = _canonical_directory(path, "IsaacLab root")
    prefix = ["git", "--no-optional-locks", "-C", str(root)]
    commit = _run_text(prefix + ["rev-parse", "HEAD"], "IsaacLab git rev-parse").strip()
    if commit != PINNED_ISAACLAB_COMMIT:
        raise LaunchError("IsaacLab commit differs from the pinned runtime")
    status = _run_text(
        prefix + ["status", "--porcelain=v1", "--untracked-files=all"],
        "IsaacLab git clean check",
    )
    if status:
        raise LaunchError("IsaacLab checkout has tracked or untracked changes")
    for relative in (
        "source/isaaclab",
        "source/isaaclab_tasks",
        "source/isaaclab_assets",
        "source/isaaclab_rl",
    ):
        _canonical_directory(root / relative, f"IsaacLab {relative}")
    return root


def _nearest_existing(path: Path) -> Path:
    current = path
    while not os.path.lexists(current):
        if current.parent == current:
            raise LaunchError("run-root has no existing ancestor")
        current = current.parent
    return _canonical_directory(current, "run-root ancestor")


def _validate_run_root(root: Path, namespace: str) -> Path:
    if (
        not root.is_absolute()
        or root == WORKSPACE_ROOT
        or WORKSPACE_ROOT not in root.parents
        or ".." in root.parts
        or os.path.lexists(root)
        or root.name != namespace
    ):
        raise LaunchError("run-root must be absent /workspace/.../<namespace>")
    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise LaunchError("run-root must be outside the source checkout")
    _nearest_existing(root.parent)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}", namespace) is None:
        raise LaunchError("namespace format differs")
    return root


def _gpu_is_free(index: int, expected_uuid: str) -> None:
    gpu_rows = _run_text(
        [str(NVIDIA_SMI), "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        "GPU identity query",
    ).splitlines()
    app_rows = _run_text(
        [
            str(NVIDIA_SMI),
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        "GPU compute-app query",
    ).splitlines()
    observed: dict[int, str] = {}
    for raw in gpu_rows:
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise LaunchError("malformed GPU identity row")
        gpu_index = int(fields[0])
        if gpu_index in observed or fields[1] in observed.values():
            raise LaunchError("duplicate GPU identity row")
        observed[gpu_index] = fields[1]
    if observed.get(index) != expected_uuid:
        raise LaunchError("selected GPU UUID mapping differs")
    for raw in app_rows:
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 2 or not fields[0].startswith("GPU-") or not fields[1].isdigit():
            raise LaunchError("malformed GPU compute-app row")
        if fields[0] == expected_uuid:
            raise LaunchError("selected GPU already has a compute application")


def _available_cpu_ids() -> set[int]:
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise LaunchError("cpu-affinity requires Linux sched_getaffinity") from exc


def _cpu_affinity(value: str | None) -> tuple[str | None, tuple[int, ...]]:
    """Parse one explicit Linux CPU list and reject unavailable processors."""

    if value is None:
        return None, ()
    if re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value) is None:
        raise LaunchError("cpu-affinity must be a comma-separated CPU/range list")
    selected: set[int] = set()
    for field in value.split(","):
        if "-" in field:
            lower_text, upper_text = field.split("-", 1)
            lower, upper = int(lower_text), int(upper_text)
            if upper < lower:
                raise LaunchError("cpu-affinity range is reversed")
            values = range(lower, upper + 1)
        else:
            values = (int(field),)
        for cpu in values:
            if cpu > 4095 or cpu in selected:
                raise LaunchError("cpu-affinity contains an invalid or duplicate CPU")
            selected.add(cpu)
    available = _available_cpu_ids()
    if not selected or not selected.issubset(available):
        raise LaunchError("cpu-affinity is outside the launcher's allowed CPU set")
    return value, tuple(sorted(selected))


def _paths(root: Path) -> dict[str, Path]:
    return {
        "asset_dir": root / "asset",
        "asset": root / "asset" / "model.usd",
        "tmp": root / "tmp",
        "pycache": root / "pycache",
        "home": root / "home",
        "cuda_cache": root / "cuda_cache",
        "xdg_cache": root / "xdg_cache",
        "xdg_config": root / "xdg_config",
        "xdg_data": root / "xdg_data",
        "xdg_state": root / "xdg_state",
        "runtime_receipt": root / "train-runtime.receipt",
        "rate_receipt": root / "diagnostic-rate-probe.json",
        "profile_receipt": root / "diagnostic-profile-probe.json",
        "fixed_action_probe": root / "fixed-action-probe",
        "run_log": root / "run.log",
        "launch_state": root / "kit_boot.launch",
        "training": root / "training",
        "hydra": root / "training" / "hydra",
    }


def _asset_package(asset_source: Path) -> Path:
    package = _canonical_directory(asset_source.parent, "A3 asset package")
    for directory, names, files in os.walk(package, followlinks=False):
        for name in (*names, *files):
            item = Path(directory) / name
            try:
                row = item.lstat()
            except OSError as exc:
                raise LaunchError("cannot inspect A3 asset package") from exc
            if stat.S_ISLNK(row.st_mode) or not (
                stat.S_ISDIR(row.st_mode) or stat.S_ISREG(row.st_mode)
            ):
                raise LaunchError("A3 asset package contains a non-regular entry")
    return package


def _validate_gl_runtime(opengl_dir: Path, glu_dir: Path) -> str:
    """Bind exact headless OpenGL/GLU bytes before spending a run root."""

    opengl = _canonical_directory(opengl_dir, "OpenGL library directory")
    glu = _canonical_directory(glu_dir, "GLU library directory")
    rows = (
        (
            opengl,
            "libOpenGL.so.0.0.0",
            "libOpenGL.so.0",
            PINNED_OPENGL_SHA256,
            "OpenGL",
        ),
        (
            glu,
            "libGLU.so.1.3.1",
            "libGLU.so.1",
            PINNED_GLU_SHA256,
            "GLU",
        ),
    )
    for directory, real_name, link_name, expected_sha256, label in rows:
        real = _canonical_regular(directory / real_name, f"{label} runtime")
        if _sha256(real) != expected_sha256:
            raise LaunchError(f"{label} runtime digest differs")
        link = directory / link_name
        try:
            link_stat = link.lstat()
            target = os.readlink(link)
        except OSError as exc:
            raise LaunchError(f"{label} SONAME link is absent") from exc
        if not stat.S_ISLNK(link_stat.st_mode) or target != real_name:
            raise LaunchError(f"{label} SONAME link differs")
    return f"{opengl}:{glu}"


def _create_root(root: Path, paths: dict[str, Path], asset_package: Path) -> None:
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.parent.resolve(strict=True) != root.parent or os.path.lexists(root):
        raise LaunchError("run-root changed before creation")
    try:
        os.mkdir(root, 0o700)
        for name in (
            "tmp",
            "pycache",
            "home",
            "cuda_cache",
            "xdg_cache",
            "xdg_config",
            "xdg_data",
            "xdg_state",
            "training",
        ):
            os.mkdir(paths[name], 0o700)
        shutil.copytree(asset_package, paths["asset_dir"], symlinks=False)
        os.chmod(paths["asset"], 0o400)
    except OSError as exc:
        raise LaunchError("cannot create fresh Isaac run-root") from exc
    if _sha256(paths["asset"]) != PINNED_A3_USD_SHA256:
        raise LaunchError("copied A3 USD digest differs")


def _child_argv(
    isaac_python: Path,
    train: Path,
    namespace: str,
    hydra_run_dir: Path,
    *,
    diagnostic_rate_probe: bool = False,
    diagnostic_profile_probe: bool = False,
    diagnostic_fixed_action_probe: bool = False,
) -> list[str]:
    argv = [
        str(isaac_python),
        "-P",
        "-B",
        str(train),
        "task=HOPEPingPongActionBallFullMdpA",
        "algo=ppo",
        "headless=true",
        "video=false",
        "logger=tensorboard",
        "device=cuda:0",
        "seed=0",
        f"num_envs={FULL_MDP_PPO_RECIPE.num_envs}",
        f"run_name={namespace}-DIAGNOSTIC_UNAUTHORIZED",
        "checkpoint_path=null",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false",
        "action_ball_dynamic_ready_bootstrap=true",
        (
            "action_ball_dynamic_ready_artifact_path="
            f"{(REPO_ROOT / DYNAMIC_READY_ARTIFACT_RELATIVE).resolve()}"
        ),
        (
            "action_ball_dynamic_ready_artifact_sha256="
            f"{DYNAMIC_READY_ARTIFACT_SHA256}"
        ),
        (
            "action_ball_dynamic_ready_nominal_receipt_path="
            f"{(REPO_ROOT / DYNAMIC_READY_RECEIPT_RELATIVE).resolve()}"
        ),
        (
            "action_ball_dynamic_ready_nominal_receipt_sha256="
            f"{DYNAMIC_READY_RECEIPT_SHA256}"
        ),
        f"hydra.run.dir={hydra_run_dir}",
    ]
    if diagnostic_rate_probe:
        argv.append("task.action_ball_full_mdp_rate_probe=true")
    if diagnostic_profile_probe:
        argv.append("task.action_ball_full_mdp_profile_probe=true")
    if diagnostic_fixed_action_probe:
        argv.append(
            "task.action_ball_full_mdp_fixed_action_probe_output_path="
            f"{hydra_run_dir.parents[1] / 'fixed-action-probe'}"
        )
    return argv


def _pythonpath(isaaclab: Path, venv_site: Path) -> str:
    rows = [
        "/proc/self/fd/18",
        str(REPO_ROOT / "hope_training/whole_body_tracking/scripts"),
        str(REPO_ROOT / "hope_training/whole_body_tracking/source/whole_body_tracking"),
        *(str(isaaclab / relative) for relative in (
            "source/isaaclab",
            "source/isaaclab_tasks",
            "source/isaaclab_assets",
            "source/isaaclab_rl",
        )),
        str(venv_site),
    ]
    return ":".join(rows)


def _runtime_env(
    *,
    gpu_uuid: str,
    paths: dict[str, Path],
    isaaclab: Path,
    venv_site: Path,
    rsl_sha256: str,
    ld_library_path: str,
    profile_updates: int,
) -> dict[str, str]:
    result = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        # Kit/Omniverse, NVIDIA JIT and Python caches must be owned by this
        # fresh namespace.  The Pod root overlay is small and shared; using
        # /root would both risk ENOSPC and mix evidence across runs.
        "HOME": str(paths["home"]),
        "XDG_CACHE_HOME": str(paths["xdg_cache"]),
        "XDG_CONFIG_HOME": str(paths["xdg_config"]),
        "XDG_DATA_HOME": str(paths["xdg_data"]),
        "XDG_STATE_HOME": str(paths["xdg_state"]),
        "CUDA_CACHE_PATH": str(paths["cuda_cache"]),
        "TMPDIR": str(paths["tmp"]),
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "HOPE_AGIBOT_A3_USD_PATH": str(paths["asset"]),
        "HOPE_URDF_IMPORTER_NO_UI": "1",
        "HOPE_ACTION_BALL_RUNTIME_ATTESTATION": "sealed_rsl_v1",
        "HOPE_ACTION_BALL_RUNTIME_RECEIPT_PATH": str(paths["runtime_receipt"]),
        "HOPE_ACTION_BALL_RUNTIME_KIT_PYTHON_SHA256": PINNED_KIT_PYTHON_SHA256,
        "HOPE_ACTION_BALL_RUNTIME_RSL_ZIP_SHA256": rsl_sha256,
        "HOPE_ACTION_BALL_RUNTIME_VENV_SITE": str(venv_site),
        "HOPE_ACTION_BALL_FULL_MDP_LOG_ROOT": str(paths["training"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
        "PYTHONPATH": _pythonpath(isaaclab, venv_site),
        "LD_LIBRARY_PATH": ld_library_path,
    }
    if profile_updates:
        result["HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES"] = str(
            profile_updates
        )
    return result


def _launcher_env(
    paths: dict[str, Path], *, finite_probe: bool = False,
    boot_marker: str = "Learning iteration"
) -> dict[str, str]:
    result = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(paths["home"]),
        "XDG_CACHE_HOME": str(paths["xdg_cache"]),
        "XDG_CONFIG_HOME": str(paths["xdg_config"]),
        "XDG_DATA_HOME": str(paths["xdg_data"]),
        "XDG_STATE_HOME": str(paths["xdg_state"]),
        "KIT_BOOT_MARKER": boot_marker,
        "KIT_BOOT_TIMEOUT_S": "900",
        "KIT_BOOT_STALE_TIMEOUT_S": "180",
        "KIT_BOOT_POLL_S": "5",
        "KIT_BOOT_STATE_FILE": str(paths["launch_state"]),
        "KIT_WAIT_FOR_COMPLETION": "1" if finite_probe else "0",
    }
    if finite_probe:
        result["KIT_COMPLETION_TIMEOUT_S"] = str(
            RATE_PROBE_COMPLETION_TIMEOUT_S
        )
    return result


def _acquire_lock(path: Path) -> int:
    _canonical_regular(path, "GPU lock-file")
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise LaunchError("GPU launch lock is already held") from exc
    except OSError as exc:
        raise LaunchError("cannot acquire GPU launch lock") from exc
    os.set_inheritable(descriptor, True)
    return descriptor


def _open_exact_runtime_descriptors(paths: dict[str, Path], rsl_wheel: Path) -> tuple[int, int]:
    for target in (16, 18):
        try:
            os.fstat(target)
        except OSError:
            continue
        raise LaunchError(f"reserved runtime descriptor {target} is already open")
    try:
        receipt = os.open(
            paths["runtime_receipt"],
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        archive = os.memfd_create("hope-rsl-rl-3.1.2", os.MFD_ALLOW_SEALING)
        with rsl_wheel.open("rb", buffering=0) as source:
            while chunk := source.read(1024 * 1024):
                if os.write(archive, chunk) != len(chunk):
                    raise LaunchError("short write to sealed RSL archive")
        os.lseek(archive, 0, os.SEEK_SET)
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(archive, fcntl.F_ADD_SEALS, seals)
        os.dup2(receipt, 16, inheritable=True)
        os.dup2(archive, 18, inheritable=True)
    except (AttributeError, OSError) as exc:
        raise LaunchError("cannot create exact sealed RSL runtime descriptors") from exc
    finally:
        if "receipt" in locals() and receipt != 16:
            os.close(receipt)
        if "archive" in locals() and archive != 18:
            os.close(archive)
    return 16, 18


def _state_fields(path: Path) -> dict[str, str]:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LaunchError("Kit launcher did not publish its state") from exc
    return dict(row.split("=", 1) for row in rows if "=" in row)


def _verify_inherited_gpu_lock(
    *, proc: Path, descriptor: int, lock_file: Path
) -> None:
    """Require the ready workload to retain the exact locked open description."""

    try:
        descriptor_stat = os.stat(proc / "fd" / str(descriptor))
        pathname_stat = os.stat(lock_file, follow_symlinks=False)
        fdinfo = (proc / "fdinfo" / str(descriptor)).read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise LaunchError("Isaac child did not inherit the GPU lifetime lock") from exc
    if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
        pathname_stat.st_dev,
        pathname_stat.st_ino,
    ):
        raise LaunchError("Isaac child GPU lock descriptor identity differs")
    lock_rows = tuple(
        row for row in fdinfo.splitlines() if row.startswith("lock:")
    )
    lock_fields = lock_rows[0].split() if len(lock_rows) == 1 else ()
    if (
        len(lock_rows) != 1
        or not any(
            lock_fields[index : index + 3] == ["FLOCK", "ADVISORY", "WRITE"]
            for index in range(max(0, len(lock_fields) - 2))
        )
    ):
        raise LaunchError("Isaac child GPU lifetime flock is absent")


def _verify_started(
    paths: dict[str, Path], *, gpu_lock: int, lock_file: Path
) -> tuple[int, int]:
    fields = _state_fields(paths["launch_state"])
    try:
        pid = int(fields["pid"])
        pgid = int(fields["pgid"])
    except (KeyError, ValueError) as exc:
        raise LaunchError("Kit launch state has no exact PID/PGID") from exc
    if pid <= 1 or pgid != pid or fields.get("ready_utc") is None:
        raise LaunchError("Kit launch did not reach its ready marker")
    try:
        runtime_receipt = paths["runtime_receipt"].read_bytes()
    except OSError as exc:
        raise LaunchError("cannot read runtime attestation receipt") from exc
    if runtime_receipt != b"trainer_runtime_attested_v2\n":
        raise LaunchError("runtime attestation receipt differs")
    proc = Path(f"/proc/{pid}")
    try:
        state = (proc / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1][0]
    except (OSError, IndexError) as exc:
        raise LaunchError("Isaac child disappeared after ready marker") from exc
    if state == "Z":
        raise LaunchError("Isaac child is a zombie after ready marker")
    _verify_inherited_gpu_lock(
        proc=proc, descriptor=gpu_lock, lock_file=lock_file
    )
    return pid, pgid


def _verify_completed(paths: dict[str, Path]) -> tuple[int, int]:
    """Require one natural, clean completion after a finite diagnostic."""

    fields = _state_fields(paths["launch_state"])
    try:
        pid = int(fields["pid"])
        pgid = int(fields["pgid"])
    except (KeyError, ValueError) as exc:
        raise LaunchError("Kit launch state has no exact PID/PGID") from exc
    if (
        pid <= 1
        or pgid != pid
        or fields.get("ready_utc") is None
        or fields.get("completion_utc") is None
        or fields.get("completion_exit_code") != "0"
        or fields.get("terminal_kind") != "clean_completion"
        or fields.get("terminal_exit_code") != "0"
    ):
        raise LaunchError("finite diagnostic probe did not complete naturally")
    if any(
        key in fields
        for key in (
            "stop_signal",
            "completion_timeout_s",
            "term_identity_evidence",
            "kill_identity_evidence",
        )
    ):
        raise LaunchError("finite diagnostic probe completion used a stop path")
    try:
        runtime_receipt = paths["runtime_receipt"].read_bytes()
    except OSError as exc:
        raise LaunchError("cannot read runtime attestation receipt") from exc
    if runtime_receipt != b"trainer_runtime_attested_v2\n":
        raise LaunchError("runtime attestation receipt differs")
    return pid, pgid


def _verify_fixed_action_probe(
    paths: dict[str, Path], *, source_commit: str
) -> dict[str, object]:
    """Require one complete portable-state record from the finite Isaac child."""

    root = paths["fixed_action_probe"]
    summary_path = _canonical_regular(root / "summary.json", "fixed-action summary")
    arrays_path = _canonical_regular(
        root / "portable_state.npz", "fixed-action arrays"
    )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError("fixed-action summary is not JSON") from exc
    if (
        summary.get("schema_version") != 3
        or summary.get("record_type")
        != "action_ball_full_mdp_cross_engine_tape_probe_v3"
        or summary.get("backend") != "isaac"
        or summary.get("diagnostic_unauthorized") is not True
        or summary.get("training_authorized") is not False
        or summary.get("promotion_authority") is not False
        or summary.get("physics_parity_authority") is not False
        or summary.get("source")
        != {"commit": source_commit, "dirty": False}
        or summary.get("shape")
        != {"num_envs": 512, "num_ticks": 48, "action_width": 31}
        or summary.get("arrays_sha256") != _sha256(arrays_path)
    ):
        raise LaunchError("fixed-action probe identity differs")
    return summary


def _rate_probe_payload(
    *,
    run_log: Path,
    source_commit: str,
    namespace: str,
    gpu_index: int,
    gpu_uuid: str,
) -> dict[str, object]:
    """Parse RSL-RL's own update walls; do not redefine PPO timing."""

    try:
        raw = run_log.read_text(encoding="utf-8")
    except OSError as exc:
        raise LaunchError("cannot read completed diagnostic rate log") from exc
    lines = [ANSI_RE.sub("", line) for line in raw.splitlines()]
    iteration_ids: list[int] = []
    iteration_totals: list[int] = []
    update_seconds: list[float] = []
    for line in lines:
        match = LEARNING_ITERATION_RE.fullmatch(line)
        if match is not None:
            iteration_ids.append(int(match.group(1)))
            iteration_totals.append(int(match.group(2)))
        match = ITERATION_TIME_RE.fullmatch(line)
        if match is not None:
            seconds = float(match.group(1))
            if not 0.0 < seconds < float("inf"):
                raise LaunchError("diagnostic rate probe update wall differs")
            update_seconds.append(seconds)
    budget_markers = [
        line for line in lines if RATE_PROBE_MARKER_TOKEN in line
    ]
    if budget_markers != [_expected_rate_probe_marker()]:
        raise LaunchError(
            "diagnostic rate probe budget must be exact 61/10/50/1"
        )
    recipe_markers = [
        line for line in lines if PPO_RECIPE_MARKER_TOKEN in line
    ]
    if recipe_markers != [_expected_ppo_recipe_marker()]:
        raise LaunchError("diagnostic rate probe recipe identity differs")
    updates, warmup_updates, measured_updates, tail_updates = RATE_PROBE_BUDGET
    expected_ids = list(range(updates))
    if (
        iteration_ids != expected_ids
        or iteration_totals != [updates] * updates
        or len(update_seconds) != updates
    ):
        raise LaunchError("diagnostic rate probe 61-update timing window differs")
    child_learning_recipe_sha256 = FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
    measured = update_seconds[
        warmup_updates:warmup_updates + measured_updates
    ]
    tail = update_seconds[-tail_updates:]
    if len(measured) != measured_updates or len(tail) != tail_updates:
        raise LaunchError("diagnostic rate probe measured timing window differs")
    candidate_production_recipe = FULL_MDP_PPO_RECIPE.execution_recipe()
    candidate_production_sha256 = FULL_MDP_PPO_RECIPE.recipe_sha256()
    rate_execution_recipe = _rate_execution_recipe()
    rate_execution_sha256 = _canonical_payload_sha256(rate_execution_recipe)
    return {
        "kind": "action_ball_isaac_full_mdp_h48_rate_probe_v2",
        "schema_version": 2,
        "diagnostic_unauthorized": True,
        "formal_evidence": False,
        "safety_gate": False,
        "source_commit": source_commit,
        "namespace": namespace,
        "gpu": {"index": gpu_index, "uuid": gpu_uuid},
        "learning_recipe_sha256": child_learning_recipe_sha256,
        "candidate_production_execution_recipe": candidate_production_recipe,
        "candidate_production_execution_recipe_sha256": (
            candidate_production_sha256
        ),
        "rate_execution_recipe": rate_execution_recipe,
        "rate_execution_recipe_sha256": rate_execution_sha256,
        "shape": {
            "num_envs": FULL_MDP_PPO_RECIPE.num_envs,
            "num_steps_per_env": FULL_MDP_PPO_RECIPE.num_steps_per_env,
            "updates": updates,
        },
        "timing_source": (
            "RSL-RL Iteration time stdout; profiler off; values retain "
            "the runtime's printed precision"
        ),
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "tail_updates": tail_updates,
        "raw_update_seconds": update_seconds,
        "measured_update_seconds": measured,
        "update_seconds_p50": statistics.median(measured),
        "update_seconds_p90": statistics.quantiles(
            measured, n=10, method="inclusive"
        )[8],
        "run_log": {
            "path": str(run_log),
            "sha256": _sha256(run_log),
        },
    }


def _profile_probe_payload(
    *,
    run_log: Path,
    source_commit: str,
    namespace: str,
    gpu_index: int,
    gpu_uuid: str,
    requested_updates: int,
) -> dict[str, object]:
    """Validate and retain exact bounded inclusive host-wall profile rows."""

    try:
        raw = run_log.read_text(encoding="utf-8")
    except OSError as exc:
        raise LaunchError("cannot read completed diagnostic profile log") from exc
    lines = [ANSI_RE.sub("", line) for line in raw.splitlines()]
    markers = [line for line in lines if PROFILE_PROBE_MARKER_TOKEN in line]
    if markers != [_expected_profile_probe_marker(requested_updates)]:
        raise LaunchError("diagnostic profile probe budget marker differs")
    recipe_markers = [line for line in lines if PPO_RECIPE_MARKER_TOKEN in line]
    if recipe_markers != [_expected_ppo_recipe_marker()]:
        raise LaunchError("diagnostic profile probe recipe identity differs")
    profiles: list[dict[str, object]] = []
    for line in lines:
        if not line.startswith(PROFILE_JSON_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(PROFILE_JSON_PREFIX))
            json.dumps(payload, allow_nan=False)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LaunchError("diagnostic profile row is not finite JSON") from exc
        except ValueError as exc:
            raise LaunchError("diagnostic profile row is not finite JSON") from exc
        if not isinstance(payload, dict):
            raise LaunchError("diagnostic profile row is not a mapping")
        profiles.append(payload)
    expected_ordinals = list(range(1, requested_updates + 1))
    if (
        len(profiles) != requested_updates
        or [row.get("profile_update_ordinal") for row in profiles]
        != expected_ordinals
        or [row.get("update") for row in profiles]
        != list(range(requested_updates))
        or any(
            row.get("schema_version") != 2
            or row.get("requested_profile_updates") != requested_updates
            or row.get("rollout_call_count_exact") is not True
            or row.get("speed_evidence_eligible") is not False
            or not isinstance(row.get("segments"), dict)
            for row in profiles
        )
    ):
        raise LaunchError("diagnostic profile probe rows differ")
    return {
        "kind": "action_ball_isaac_full_mdp_h48_profile_probe_v1",
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "formal_evidence": False,
        "speed_evidence": False,
        "safety_gate": False,
        "source_commit": source_commit,
        "namespace": namespace,
        "gpu": {"index": gpu_index, "uuid": gpu_uuid},
        "learning_recipe_sha256": (
            FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
        ),
        "shape": {
            "num_envs": FULL_MDP_PPO_RECIPE.num_envs,
            "num_steps_per_env": FULL_MDP_PPO_RECIPE.num_steps_per_env,
            "updates": requested_updates,
        },
        "timing_source": (
            "inclusive nested host perf_counter spans without CUDA sync; "
            "profile rows are attribution-only"
        ),
        "profiles": profiles,
        "run_log": {"path": str(run_log), "sha256": _sha256(run_log)},
    }


def _write_diagnostic_receipt(path: Path, payload: dict[str, object]) -> None:
    try:
        body = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LaunchError("diagnostic receipt is not finite JSON") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(body):
            written = os.write(descriptor, body[offset:])
            if written <= 0:
                raise OSError("short diagnostic receipt write")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise LaunchError("cannot no-clobber diagnostic receipt") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)


def launch(args: argparse.Namespace) -> int:
    commit = _source_commit()
    train = _canonical_regular(REPO_ROOT / TRAIN_RELATIVE, "Isaac train entry")
    kit_launcher = _canonical_regular(
        REPO_ROOT / KIT_LAUNCHER_RELATIVE, "Kit boot launcher", executable=True
    )
    isaac_python = _canonical_regular(args.isaac_python, "Isaac python.sh", executable=True)
    kit_python = _canonical_regular(args.kit_python, "Kit Python", executable=True)
    if _sha256(kit_python) != PINNED_KIT_PYTHON_SHA256:
        raise LaunchError("Kit Python digest differs")
    isaaclab = _validate_isaaclab(args.isaaclab_root)
    venv_site = _canonical_directory(args.venv_site, "RSL/TensorDict venv site")
    asset_source = _canonical_regular(args.asset_usd, "A3 USD")
    if _sha256(asset_source) != PINNED_A3_USD_SHA256:
        raise LaunchError("A3 USD digest differs")
    asset_package = _asset_package(asset_source)
    rsl_wheel = _canonical_regular(args.rsl_wheel, "RSL-RL wheel")
    rsl_sha256 = _sha256(rsl_wheel)
    if rsl_sha256 != PINNED_RSL_WHEEL_SHA256:
        raise LaunchError("RSL-RL wheel digest differs")
    ld_library_path = _validate_gl_runtime(
        args.opengl_lib_dir,
        args.glu_lib_dir,
    )
    root = _validate_run_root(args.run_root, args.namespace)
    if args.gpu_index < 0:
        raise LaunchError("gpu-index must be nonnegative")
    if re.fullmatch(r"GPU-[A-Za-z0-9-]{8,}", args.expected_gpu_uuid) is None:
        raise LaunchError("expected-gpu-uuid format differs")
    if not 0 <= args.profile_updates <= 50:
        raise LaunchError("profile-updates must be between 0 and 50")
    diagnostic_count = sum(
        bool(value)
        for value in (
            args.diagnostic_rate_probe,
            args.diagnostic_profile_probe,
            args.diagnostic_fixed_action_probe,
        )
    )
    if diagnostic_count > 1:
        raise LaunchError("diagnostic probes are mutually exclusive")
    if args.diagnostic_rate_probe and args.profile_updates:
        raise LaunchError(
            "diagnostic-rate-probe and profile-updates are mutually exclusive"
        )
    if args.diagnostic_profile_probe and args.profile_updates == 0:
        raise LaunchError(
            "diagnostic-profile-probe requires positive profile-updates"
        )
    if args.diagnostic_fixed_action_probe and args.profile_updates:
        raise LaunchError(
            "diagnostic-fixed-action-probe and profile-updates are mutually exclusive"
        )
    cpu_affinity, cpu_ids = _cpu_affinity(args.cpu_affinity)
    if cpu_affinity is not None:
        _canonical_regular(TASKSET, "taskset", executable=True)
    paths = _paths(root)
    child_argv = _child_argv(
        isaac_python,
        train,
        args.namespace,
        paths["hydra"],
        diagnostic_rate_probe=args.diagnostic_rate_probe,
        diagnostic_profile_probe=args.diagnostic_profile_probe,
        diagnostic_fixed_action_probe=args.diagnostic_fixed_action_probe,
    )
    runtime_env = _runtime_env(
        gpu_uuid=args.expected_gpu_uuid,
        paths=paths,
        isaaclab=isaaclab,
        venv_site=venv_site,
        rsl_sha256=rsl_sha256,
        ld_library_path=ld_library_path,
        profile_updates=args.profile_updates,
    )
    boot_marker = (
        FIXED_ACTION_PROBE_BOOT_MARKER
        if args.diagnostic_fixed_action_probe
        else "Learning iteration"
    )
    if args.dry_run:
        dry_run_payload = {
            "source_commit": commit,
            "argv": child_argv,
            "launcher_env": _launcher_env(
                paths,
                finite_probe=(
                    args.diagnostic_rate_probe
                    or args.diagnostic_profile_probe
                    or args.diagnostic_fixed_action_probe
                ),
                boot_marker=boot_marker,
            ),
            "runtime_env": runtime_env,
            "run_root": str(root),
            "cpu_affinity": cpu_affinity,
            "cpu_ids": cpu_ids,
        }
        if args.diagnostic_rate_probe:
            rate_execution_recipe = _rate_execution_recipe()
            dry_run_payload.update({
                "candidate_production_execution_recipe": (
                    FULL_MDP_PPO_RECIPE.execution_recipe()
                ),
                "candidate_production_execution_recipe_sha256": (
                    FULL_MDP_PPO_RECIPE.recipe_sha256()
                ),
                "rate_execution_recipe": rate_execution_recipe,
                "rate_execution_recipe_sha256": (
                    _canonical_payload_sha256(rate_execution_recipe)
                ),
            })
        else:
            dry_run_payload.update({
                "ppo_recipe": FULL_MDP_PPO_RECIPE.execution_recipe(),
                "ppo_recipe_sha256": FULL_MDP_PPO_RECIPE.recipe_sha256(),
            })
        print(json.dumps(
            dry_run_payload, sort_keys=True, separators=(",", ":")
        ))
        return 0

    gpu_lock = _acquire_lock(args.lock_file)
    runtime_descriptors: tuple[int, int] | None = None
    try:
        _gpu_is_free(args.gpu_index, args.expected_gpu_uuid)
        _create_root(root, paths, asset_package)
        runtime_descriptors = _open_exact_runtime_descriptors(paths, rsl_wheel)
        clean_child = [
            "/usr/bin/env",
            "-i",
            *(f"{name}={value}" for name, value in runtime_env.items()),
            *child_argv,
        ]
        if cpu_affinity is not None:
            clean_child = [str(TASKSET), "-c", cpu_affinity, *clean_child]
        command = [str(kit_launcher), str(paths["run_log"]), *clean_child]
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_launcher_env(
                    paths,
                    finite_probe=(
                        args.diagnostic_rate_probe
                        or args.diagnostic_profile_probe
                        or args.diagnostic_fixed_action_probe
                    ),
                    boot_marker=boot_marker,
                ),
                stdin=subprocess.DEVNULL,
                check=False,
                pass_fds=(gpu_lock, *runtime_descriptors),
            )
        except OSError as exc:
            raise LaunchError("cannot start the Kit boot launcher") from exc
        if result.returncode != 0:
            raise LaunchError(f"Kit boot launcher failed rc={result.returncode}")
        if (
            args.diagnostic_rate_probe
            or args.diagnostic_profile_probe
            or args.diagnostic_fixed_action_probe
        ):
            pid, pgid = _verify_completed(paths)
        fixed_action_payload = None
        if args.diagnostic_rate_probe:
            rate_payload = _rate_probe_payload(
                run_log=paths["run_log"],
                source_commit=commit,
                namespace=args.namespace,
                gpu_index=args.gpu_index,
                gpu_uuid=args.expected_gpu_uuid,
            )
            _write_diagnostic_receipt(paths["rate_receipt"], rate_payload)
            profile_payload = None
            status = "DIAGNOSTIC_RATE_PROBE_COMPLETE"
        elif args.diagnostic_profile_probe:
            profile_payload = _profile_probe_payload(
                run_log=paths["run_log"],
                source_commit=commit,
                namespace=args.namespace,
                gpu_index=args.gpu_index,
                gpu_uuid=args.expected_gpu_uuid,
                requested_updates=args.profile_updates,
            )
            _write_diagnostic_receipt(
                paths["profile_receipt"], profile_payload
            )
            rate_payload = None
            status = "DIAGNOSTIC_PROFILE_PROBE_COMPLETE"
        elif args.diagnostic_fixed_action_probe:
            fixed_action_payload = _verify_fixed_action_probe(
                paths, source_commit=commit
            )
            rate_payload = None
            profile_payload = None
            status = "DIAGNOSTIC_FIXED_ACTION_PROBE_COMPLETE"
        else:
            pid, pgid = _verify_started(
                paths, gpu_lock=gpu_lock, lock_file=args.lock_file
            )
            rate_payload = None
            profile_payload = None
            fixed_action_payload = None
            status = "RUNNING"
        print(json.dumps({
            "diagnostic_unauthorized": True,
            "gpu_uuid": args.expected_gpu_uuid,
            "cpu_affinity": cpu_affinity,
            "namespace": args.namespace,
            "pid": pid,
            "pgid": pgid,
            "gpu_lock_fd": gpu_lock,
            "run_root": str(root),
            "source_commit": commit,
            "status": status,
            **(
                {
                    "rate_receipt": str(paths["rate_receipt"]),
                    "update_seconds_p50": rate_payload["update_seconds_p50"],
                    "update_seconds_p90": rate_payload["update_seconds_p90"],
                }
                if rate_payload is not None
                else {}
            ),
            **(
                {
                    "profile_receipt": str(paths["profile_receipt"]),
                    "profile_updates": args.profile_updates,
                }
                if profile_payload is not None
                else {}
            ),
            **(
                {
                    "fixed_action_probe": str(paths["fixed_action_probe"]),
                    "fixed_action_done_rows": fixed_action_payload["done_rows"],
                }
                if fixed_action_payload is not None
                else {}
            ),
        }, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        if runtime_descriptors is not None:
            for descriptor in runtime_descriptors:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        # Do not call LOCK_UN: the running child inherited this same open file
        # description.  Closing only our duplicate leaves the lock owned until
        # the child naturally exits.
        os.close(gpu_lock)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--kit-python", type=Path, required=True)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument("--venv-site", type=Path, required=True)
    parser.add_argument("--asset-usd", type=Path, required=True)
    parser.add_argument("--rsl-wheel", type=Path, required=True)
    parser.add_argument("--opengl-lib-dir", type=Path, required=True)
    parser.add_argument("--glu-lib-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--diagnostic-rate-probe", action="store_true")
    parser.add_argument("--diagnostic-profile-probe", action="store_true")
    parser.add_argument("--diagnostic-fixed-action-probe", action="store_true")
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return launch(parse_args(argv))
    except LaunchError as exc:
        print(f"launch_isaac_full_mdp_successor: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
