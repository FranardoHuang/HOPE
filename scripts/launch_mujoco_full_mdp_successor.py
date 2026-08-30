#!/usr/bin/env python3
"""Launch one fresh MuJoCo Full-A H48 run and wait for its natural exit."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path("/workspace")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
TASKSET = Path("/usr/bin/taskset")
RUNNER_RELATIVE = Path(
    "hope_training/whole_body_tracking/mjlab_lane/"
    "mujoco_gpu_ac_full_mdp_wait_rsl3.py"
)
PPO_RECIPE_RELATIVE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "action_ball_full_mdp_ppo_recipe.py"
)
PLANT_CONTRACT_RELATIVE = Path(
    "hope_training/whole_body_tracking/mjlab_lane/"
    "mujoco_full_mdp_plant_contract.py"
)
CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CHILD_LOCALE = "C.UTF-8"
READY_POSE_SHA256 = "b88d93c311b439bd61296b3b3a84198200d9c6938980471071992ec52d8df18f"
BALL_PHYSICS_RELATIVE = Path("configs/ball_physics_optitrack_20260730.yaml")
BALL_PHYSICS_SHA256 = (
    "3afb1c9a00f975d924169503d7dafab92ea6c0b96263336e27edcd1d6257ea14"
)
RATE_PROBE_WARMUP_UPDATES = 10
RATE_PROBE_MEASURED_UPDATES = 50
RATE_PROBE_TAIL_UPDATES = 1
RATE_PROBE_NUM_UPDATES = (
    RATE_PROBE_WARMUP_UPDATES
    + RATE_PROBE_MEASURED_UPDATES
    + RATE_PROBE_TAIL_UPDATES
)
RATE_PROBE_STDOUT_MARKER = "ACTION_BALL_MUJOCO_WAIT_RSL3_JSON="


class LaunchError(RuntimeError):
    pass


def _plant_contract_module():
    """Load the dependency-free plant contract from this exact checkout."""
    source = (REPO_ROOT / PLANT_CONTRACT_RELATIVE).resolve()
    name = "_hope_mujoco_full_mdp_plant_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise LaunchError("cached MuJoCo plant contract origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise LaunchError("cannot load MuJoCo plant contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _ppo_recipe_module():
    """Load the dependency-free typed recipe from this exact checkout."""

    source = (REPO_ROOT / PPO_RECIPE_RELATIVE).resolve()
    name = "_hope_mujoco_full_mdp_ppo_recipe_launcher"
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


def _same_exact(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _same_exact(left[name], right[name]) for name in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_exact(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


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


def _run_text(argv: list[str], label: str) -> str:
    try:
        result = subprocess.run(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, check=False,
        )
    except OSError as exc:
        raise LaunchError(f"cannot run {label}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LaunchError(f"{label} failed rc={result.returncode}: {detail}")
    return result.stdout


def _source_commit() -> str:
    prefix = ["git", "--no-optional-locks", "-C", str(REPO_ROOT)]
    commit = _run_text(prefix + ["rev-parse", "HEAD"], "git rev-parse").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise LaunchError("git HEAD is not one exact commit")
    status = _run_text(
        prefix + ["status", "--porcelain=v1", "--untracked-files=all"],
        "git clean check",
    )
    if status:
        raise LaunchError("source checkout has tracked or untracked changes")
    return commit


def _canonical_regular(path: Path, label: str) -> Path:
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
    ):
        raise LaunchError(f"{label} must be one canonical regular file")
    return path


def _ready_pose(path: Path) -> Path:
    path = _canonical_regular(path, "ready-pose")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LaunchError("cannot read ready-pose") from exc
    if digest != READY_POSE_SHA256:
        raise LaunchError("ready-pose SHA-256 differs")
    return path


def _plant_xml(path: Path) -> Path:
    path = _canonical_regular(path, "plant-xml")
    try:
        expected = _plant_contract_module().expected_plant_model_identity()
    except Exception as exc:
        raise LaunchError("cannot load pinned MuJoCo plant contract") from exc
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LaunchError("cannot read plant-xml") from exc
    if (
        path.name != expected["source_plant"]["root_filename"]
        or digest != expected["source_plant"]["root_mjcf_sha256"]
    ):
        raise LaunchError("plant-xml SHA-256 differs")
    return path


def _python_entry(path: Path) -> Path:
    """Validate a Python entry without resolving away its venv semantics."""
    if not path.is_absolute():
        raise LaunchError("python must be absolute")
    try:
        entry = path.lstat()
        resolved = path.resolve(strict=True)
        target = resolved.lstat()
    except OSError as exc:
        raise LaunchError("python is missing") from exc
    if not stat.S_ISREG(target.st_mode) or not os.access(path, os.X_OK):
        raise LaunchError("python target must be one canonical regular executable")
    if stat.S_ISREG(entry.st_mode):
        if resolved != path:
            raise LaunchError("python regular entry has a symlinked ancestor")
        return path
    if not stat.S_ISLNK(entry.st_mode) or path.parent.name != "bin":
        raise LaunchError("python symlink must be one canonical venv bin entry")
    _canonical_regular(path.parent.parent / "pyvenv.cfg", "python venv pyvenv.cfg")
    if path.parent.resolve(strict=True) != path.parent:
        raise LaunchError("python venv entry has no canonical pyvenv.cfg")
    return path


def _nearest_existing(path: Path) -> Path:
    current = path
    while not os.path.lexists(current):
        if current.parent == current:
            raise LaunchError("run-root has no existing ancestor")
        current = current.parent
    try:
        row = current.lstat()
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise LaunchError("run-root ancestor cannot be resolved") from exc
    if not stat.S_ISDIR(row.st_mode) or stat.S_ISLNK(row.st_mode) or resolved != current:
        raise LaunchError("run-root ancestor must be one canonical directory")
    return current


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    python = _python_entry(args.python)
    runner = _canonical_regular(REPO_ROOT / RUNNER_RELATIVE, "Full-A runner")
    ready_pose = _ready_pose(args.ready_pose)
    plant_xml = _plant_xml(args.plant_xml)
    root = args.run_root
    if (
        not root.is_absolute()
        or root == WORKSPACE_ROOT
        or WORKSPACE_ROOT not in root.parents
        or ".." in root.parts
        or os.path.lexists(root)
        or root.name != args.namespace
    ):
        raise LaunchError("run-root must be absent /workspace/.../<namespace>")
    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise LaunchError("run-root must be outside the source checkout")
    _nearest_existing(root.parent)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}", args.namespace) is None:
        raise LaunchError("namespace format differs")
    if args.gpu_index < 0:
        raise LaunchError("gpu-index must be nonnegative")
    if re.fullmatch(r"GPU-[A-Za-z0-9-]{8,}", args.expected_gpu_uuid) is None:
        raise LaunchError("expected-gpu-uuid format differs")
    if not args.lock_file.is_absolute():
        raise LaunchError("lock-file must be absolute")
    return python, runner, ready_pose, plant_xml, root


def _paths(root: Path) -> dict[str, Path]:
    return {
        "evidence": root / "evidence.jsonl",
        "snapshots": root / "snapshots",
        "completion": root / "completion.json",
        "runtime_site": root / "runtime_site",
        "warp_cache": root / "warp_cache",
        "cuda_cache": root / "cuda_cache",
        "home": root / "home",
        "xdg_cache": root / "xdg_cache",
        "xdg_config": root / "xdg_config",
        "xdg_data": root / "xdg_data",
        "xdg_state": root / "xdg_state",
        "tmp": root / "tmp",
        "pycache": root / "pycache",
        "stdout": root / "stdout.log",
        "stderr": root / "stderr.log",
        "rate_receipt": root / "diagnostic-rate-probe.json",
    }


def _child_argv(
    python: Path,
    runner: Path,
    paths: dict[str, Path],
    commit: str,
    namespace: str,
    *,
    diagnostic_rate_probe: bool = False,
) -> list[str]:
    argv = [
        str(python), str(runner), "--full-a",
        "--num-envs", str(FULL_MDP_PPO_RECIPE.num_envs),
        "--num-updates", str(
            RATE_PROBE_NUM_UPDATES
            if diagnostic_rate_probe
            else FULL_MDP_PPO_RECIPE.max_iterations
        ),
        "--evidence-jsonl", str(paths["evidence"]),
    ]
    if not diagnostic_rate_probe:
        argv.extend((
            "--snapshot-dir", str(paths["snapshots"]),
            "--completion-json", str(paths["completion"]),
        ))
    argv.extend((
        "--source-commit", commit, "--run-namespace", namespace,
        "--mujoco-warp-runtime-site", str(paths["runtime_site"]),
        "--save-interval", str(FULL_MDP_PPO_RECIPE.save_interval),
    ))
    if diagnostic_rate_probe:
        argv.append("--diagnostic-rate-probe")
    return argv


def _env_contract(
    gpu_uuid: str, ready_pose: Path, plant_xml: Path, paths: dict[str, Path]
) -> dict[str, object]:
    ball_physics = _canonical_regular(
        (REPO_ROOT / BALL_PHYSICS_RELATIVE).resolve(), "OptiTrack ball physics"
    )
    if _sha256(ball_physics) != BALL_PHYSICS_SHA256:
        raise LaunchError("OptiTrack ball-physics digest differs")
    return {
        "set": {
            # Construct the child environment from this closed set.  In
            # particular, dynamic-loader, Python, CUDA and thread-control
            # variables from the launch shell must not silently change the
            # execution represented by one durable run identity.
            "PATH": CHILD_PATH,
            "HOME": str(paths["home"]),
            "XDG_CACHE_HOME": str(paths["xdg_cache"]),
            "XDG_CONFIG_HOME": str(paths["xdg_config"]),
            "XDG_DATA_HOME": str(paths["xdg_data"]),
            "XDG_STATE_HOME": str(paths["xdg_state"]),
            "LANG": CHILD_LOCALE,
            "LC_ALL": CHILD_LOCALE,
            "LC_CTYPE": CHILD_LOCALE,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # CUDA accepts a GPU UUID directly.  Bind the child to the exact
            # identity that the launcher just checked with nvidia-smi instead
            # of assuming its numeric index shares CUDA's enumeration order.
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "A3_PINGPONG_XML": str(plant_xml),
            # Warp's compiler emits large PCH temporaries through TMPDIR.
            # Keep them off the small shared root overlay.
            "TMPDIR": str(paths["tmp"]),
            "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
            "ACTIONBALL_READY_POSE": str(ready_pose),
            "HOPE_BALL_PHYSICS_YAML": str(ball_physics),
            # NVIDIA's PTX/JIT cache otherwise defaults to
            # ~/.nv/ComputeCache on the small root overlay.
            "CUDA_CACHE_PATH": str(paths["cuda_cache"]),
            # Warp reads this before creating its versioned kernel cache in
            # warp.init().  Changed-source kernels belong to the fresh run,
            # not the launch user's ~/.cache/warp directory.
            "WARP_CACHE_PATH": str(paths["warp_cache"]),
        },
        "inherit": [],
    }


def _child_env(contract: dict[str, object]) -> dict[str, str]:
    if set(contract) != {"set", "inherit"} or contract["inherit"] != []:
        raise LaunchError("child environment contract differs")
    values = contract["set"]
    if (
        type(values) is not dict
        or not values
        or any(type(name) is not str or type(value) is not str
               for name, value in values.items())
    ):
        raise LaunchError("child environment contract differs")
    return dict(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise LaunchError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _rate_probe_receipt_payload(
    *,
    stdout_log: Path,
    source_commit: str,
    namespace: str,
    gpu_index: int,
    gpu_uuid: str,
) -> dict[str, object]:
    """Validate the runner's natural-exit marker and bind launcher identity."""

    try:
        rows = stdout_log.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LaunchError("cannot read completed MuJoCo rate log") from exc
    markers = [
        row[len(RATE_PROBE_STDOUT_MARKER):]
        for row in rows
        if row.startswith(RATE_PROBE_STDOUT_MARKER)
    ]
    if len(markers) != 1:
        raise LaunchError("MuJoCo diagnostic rate marker identity differs")
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result

    try:
        payload = json.loads(
            markers[0],
            object_pairs_hook=unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite JSON constant")
            ),
        )
        if type(payload) is not dict:
            raise TypeError("rate marker is not an object")
        _canonical_payload_sha256(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchError("MuJoCo diagnostic rate marker is not finite JSON") from exc

    candidate = FULL_MDP_PPO_RECIPE.execution_recipe()
    candidate_sha256 = FULL_MDP_PPO_RECIPE.recipe_sha256()
    rate_execution = _rate_execution_recipe()
    rate_execution_sha256 = _canonical_payload_sha256(rate_execution)
    expected_scalars = {
        "kind": "action_ball_mujoco_full_mdp_h48_rate_probe_v1",
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "formal_evidence": False,
        "safety_gate": False,
        "source_commit": source_commit,
        "namespace": namespace,
        "task_lifecycle": "full_a_diagnostic_rate_probe",
        "rsl_rl_version": "3.1.2",
        "policy_width": 215,
        "critic_width": 231,
        "learning_recipe_sha256": (
            FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
        ),
        "ppo_update_calls": RATE_PROBE_NUM_UPDATES,
        "environment_steps": (
            RATE_PROBE_NUM_UPDATES * FULL_MDP_PPO_RECIPE.num_steps_per_env
        ),
        "transitions": (
            RATE_PROBE_NUM_UPDATES
            * FULL_MDP_PPO_RECIPE.num_steps_per_env
            * FULL_MDP_PPO_RECIPE.num_envs
        ),
        "candidate_production_execution_recipe_sha256": candidate_sha256,
        "rate_execution_recipe_sha256": rate_execution_sha256,
    }
    outer_keys = set(expected_scalars) | {
        "candidate_production_execution_recipe",
        "rate_execution_recipe",
        "rate_probe",
    }
    if set(payload) != outer_keys or any(
        not _same_exact(payload.get(name), value)
        for name, value in expected_scalars.items()
    ):
        raise LaunchError("MuJoCo diagnostic rate receipt identity differs")
    if (
        not _same_exact(
            payload.get("candidate_production_execution_recipe"), candidate
        )
        or not _same_exact(payload.get("rate_execution_recipe"), rate_execution)
        or _canonical_payload_sha256(
            payload["candidate_production_execution_recipe"]
        ) != candidate_sha256
        or _canonical_payload_sha256(
            payload["rate_execution_recipe"]
        ) != rate_execution_sha256
    ):
        raise LaunchError("MuJoCo diagnostic rate recipe identity differs")
    rate = payload.get("rate_probe")
    rate_keys = {
        "warmup_updates",
        "measured_updates",
        "tail_updates",
        "total_wall_seconds",
        "measured_wall_seconds",
        "measured_update_seconds",
        "measured_transitions",
        "measured_transitions_per_second",
        "update_seconds_p50",
        "update_seconds_p90",
    }
    if type(rate) is not dict or set(rate) not in (
        rate_keys,
        rate_keys | {"torch_cuda_peak_allocated_bytes"},
    ) or any(
        not _same_exact(rate.get(name), value)
        for name, value in {
            "warmup_updates": RATE_PROBE_WARMUP_UPDATES,
            "measured_updates": RATE_PROBE_MEASURED_UPDATES,
            "tail_updates": RATE_PROBE_TAIL_UPDATES,
            "measured_transitions": (
                RATE_PROBE_MEASURED_UPDATES
                * FULL_MDP_PPO_RECIPE.num_steps_per_env
                * FULL_MDP_PPO_RECIPE.num_envs
            ),
        }.items()
    ):
        raise LaunchError("MuJoCo diagnostic rate timing window differs")
    measured = rate.get("measured_update_seconds")
    if (
        type(measured) is not list
        or len(measured) != RATE_PROBE_MEASURED_UPDATES
        or any(
            type(value) is not float
            or not 0.0 < value < float("inf")
            for value in measured
        )
    ):
        raise LaunchError("MuJoCo diagnostic rate timing samples differ")
    measured_wall = sum(measured)
    measured_transitions = (
        RATE_PROBE_MEASURED_UPDATES
        * FULL_MDP_PPO_RECIPE.num_steps_per_env
        * FULL_MDP_PPO_RECIPE.num_envs
    )
    expected_metrics = {
        "measured_wall_seconds": measured_wall,
        "measured_transitions_per_second": (
            measured_transitions / measured_wall
        ),
        "update_seconds_p50": statistics.median(measured),
        "update_seconds_p90": statistics.quantiles(
            measured, n=10, method="inclusive"
        )[8],
    }
    total_wall = rate.get("total_wall_seconds")
    if (
        type(total_wall) is not float
        or not measured_wall <= total_wall < float("inf")
        or any(
            not _same_exact(rate.get(name), value)
            for name, value in expected_metrics.items()
        )
        or (
            "torch_cuda_peak_allocated_bytes" in rate
            and (
                type(rate["torch_cuda_peak_allocated_bytes"]) is not int
                or rate["torch_cuda_peak_allocated_bytes"] < 0
            )
        )
    ):
        raise LaunchError("MuJoCo diagnostic rate aggregate metrics differ")

    receipt = dict(payload)
    receipt["runner_marker"] = {
        "kind": payload["kind"],
        "schema_version": payload["schema_version"],
    }
    receipt["kind"] = "action_ball_mujoco_full_mdp_h48_rate_receipt_v1"
    receipt["schema_version"] = 1
    receipt["gpu"] = {"index": gpu_index, "uuid": gpu_uuid}
    receipt["stdout_log"] = {
        "path": str(stdout_log),
        "sha256": _sha256(stdout_log),
    }
    return receipt


def _write_rate_probe_receipt(path: Path, payload: dict[str, object]) -> None:
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
        raise LaunchError("MuJoCo diagnostic rate receipt is not finite JSON") from exc
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
        raise LaunchError("cannot no-clobber MuJoCo diagnostic rate receipt") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)


def _gpu_is_free(index: int, expected_uuid: str) -> None:
    gpu_rows = _run_text(
        [str(NVIDIA_SMI), "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        "GPU identity query",
    ).splitlines()
    app_rows = _run_text(
        [str(NVIDIA_SMI), "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"],
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
    if re.fullmatch(
        r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value
    ) is None:
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
                raise LaunchError(
                    "cpu-affinity contains an invalid or duplicate CPU"
                )
            selected.add(cpu)
    available = _available_cpu_ids()
    if not selected or not selected.issubset(available):
        raise LaunchError("cpu-affinity is outside the launcher's allowed CPU set")
    return value, tuple(sorted(selected))


@contextmanager
def _exclusive_lock(path: Path):
    _canonical_regular(path, "lock-file")
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise LaunchError("cannot open lock-file") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise LaunchError("GPU launch lock is already held") from exc
    except OSError as exc:
        os.close(descriptor)
        raise LaunchError("cannot acquire GPU launch lock") from exc
    try:
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _create_root(root: Path, paths: dict[str, Path]) -> None:
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.parent.resolve(strict=True) != root.parent or os.path.lexists(root):
        raise LaunchError("run-root changed before creation")
    try:
        os.mkdir(root, 0o700)
        os.mkdir(paths["snapshots"], 0o700)
        os.mkdir(paths["warp_cache"], 0o700)
        os.mkdir(paths["cuda_cache"], 0o700)
        os.mkdir(paths["home"], 0o700)
        os.mkdir(paths["xdg_cache"], 0o700)
        os.mkdir(paths["xdg_config"], 0o700)
        os.mkdir(paths["xdg_data"], 0o700)
        os.mkdir(paths["xdg_state"], 0o700)
        os.mkdir(paths["tmp"], 0o700)
        os.mkdir(paths["pycache"], 0o700)
    except OSError as exc:
        raise LaunchError("cannot create fresh run-root") from exc


def launch(args: argparse.Namespace) -> int:
    commit = _source_commit()
    python, runner, ready_pose, plant_xml, root = _validate_inputs(args)
    cpu_affinity, cpu_ids = _cpu_affinity(args.cpu_affinity)
    taskset = None
    if cpu_affinity is not None:
        taskset = _canonical_regular(TASKSET, "taskset")
        if not os.access(taskset, os.X_OK):
            raise LaunchError("taskset must be executable")
    paths = _paths(root)
    argv = _child_argv(
        python,
        runner,
        paths,
        commit,
        args.namespace,
        diagnostic_rate_probe=args.diagnostic_rate_probe,
    )
    if cpu_affinity is not None:
        argv = [str(taskset), "-c", cpu_affinity, *argv]
    contract = _env_contract(
        args.expected_gpu_uuid, ready_pose, plant_xml, paths
    )
    if args.dry_run:
        plant_identity = _plant_contract_module().expected_plant_model_identity()
        dry_run_payload = {
            "argv": argv,
            "env": contract,
            "plant_xml": {
                "path": str(plant_xml), "expected_identity": plant_identity,
            },
            "cpu_affinity": cpu_affinity,
            "cpu_affinity_source": (
                None if cpu_affinity is None else "explicit_cli"
            ),
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
    with _exclusive_lock(args.lock_file) as gpu_lock:
        _gpu_is_free(args.gpu_index, args.expected_gpu_uuid)
        _create_root(root, paths)
        try:
            stdout = paths["stdout"].open("xb", buffering=0)
        except OSError as exc:
            raise LaunchError("cannot create child logs") from exc
        try:
            stderr = paths["stderr"].open("xb", buffering=0)
        except OSError as exc:
            stdout.close()
            raise LaunchError("cannot create child logs") from exc
        with stdout, stderr:
            try:
                child = subprocess.Popen(
                    # MuJoCo may create MUJOCO_LOG.TXT without an explicit
                    # path.  The fresh run root, not the immutable checkout,
                    # owns every such process-local fallback artifact.
                    argv, cwd=root, env=_child_env(contract),
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True, pass_fds=(gpu_lock,),
                )
            except OSError as exc:
                raise LaunchError("cannot start Full-A child") from exc
            returncode = child.wait()
        if not args.diagnostic_rate_probe or returncode != 0:
            return returncode
        rate_payload = _rate_probe_receipt_payload(
            stdout_log=paths["stdout"],
            source_commit=commit,
            namespace=args.namespace,
            gpu_index=args.gpu_index,
            gpu_uuid=args.expected_gpu_uuid,
        )
        _write_rate_probe_receipt(paths["rate_receipt"], rate_payload)
        print(json.dumps({
            "diagnostic_unauthorized": True,
            "formal_evidence": False,
            "namespace": args.namespace,
            "gpu_uuid": args.expected_gpu_uuid,
            "status": "DIAGNOSTIC_RATE_PROBE_COMPLETE",
            "rate_receipt": str(paths["rate_receipt"]),
            "update_seconds_p50": rate_payload["rate_probe"][
                "update_seconds_p50"
            ],
            "update_seconds_p90": rate_payload["rate_probe"][
                "update_seconds_p90"
            ],
        }, sort_keys=True, separators=(",", ":")))
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--ready-pose", type=Path, required=True)
    parser.add_argument("--plant-xml", type=Path, required=True)
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--diagnostic-rate-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return launch(parse_args(argv))
    except LaunchError as exc:
        print(f"launch_mujoco_full_mdp_successor: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
