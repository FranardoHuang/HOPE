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
PLANT_CONTRACT_RELATIVE = Path(
    "hope_training/whole_body_tracking/mjlab_lane/"
    "mujoco_full_mdp_plant_contract.py"
)
CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CHILD_LOCALE = "C.UTF-8"
READY_POSE_SHA256 = "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069"


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
    }


def _child_argv(
    python: Path, runner: Path, paths: dict[str, Path], commit: str, namespace: str
) -> list[str]:
    return [
        str(python), str(runner), "--full-a",
        "--num-envs", "4096", "--num-updates", "12500",
        "--evidence-jsonl", str(paths["evidence"]),
        "--snapshot-dir", str(paths["snapshots"]),
        "--completion-json", str(paths["completion"]),
        "--source-commit", commit, "--run-namespace", namespace,
        "--mujoco-warp-runtime-site", str(paths["runtime_site"]),
        "--save-interval", "500",
    ]


def _env_contract(
    gpu_uuid: str, ready_pose: Path, plant_xml: Path, paths: dict[str, Path]
) -> dict[str, object]:
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
    argv = _child_argv(python, runner, paths, commit, args.namespace)
    if cpu_affinity is not None:
        argv = [str(taskset), "-c", cpu_affinity, *argv]
    contract = _env_contract(
        args.expected_gpu_uuid, ready_pose, plant_xml, paths
    )
    if args.dry_run:
        plant_identity = _plant_contract_module().expected_plant_model_identity()
        print(json.dumps({
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
        }, sort_keys=True, separators=(",", ":")))
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
            return child.wait()


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
