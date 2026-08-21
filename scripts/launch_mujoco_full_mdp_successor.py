#!/usr/bin/env python3
"""Launch one fresh MuJoCo Full-A H48 run and wait for its natural exit."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
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
RUNNER_RELATIVE = Path(
    "hope_training/whole_body_tracking/mjlab_lane/"
    "mujoco_gpu_ac_full_mdp_wait_rsl3.py"
)
ENV_UNSET = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX")


class LaunchError(RuntimeError):
    pass


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


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    python = _python_entry(args.python)
    runner = _canonical_regular(REPO_ROOT / RUNNER_RELATIVE, "Full-A runner")
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
    return python, runner, root


def _paths(root: Path) -> dict[str, Path]:
    return {
        "evidence": root / "evidence.jsonl",
        "snapshots": root / "snapshots",
        "completion": root / "completion.json",
        "runtime_site": root / "runtime_site",
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


def _env_contract(gpu_index: int) -> dict[str, object]:
    return {
        "set": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "unset": list(ENV_UNSET),
    }


def _child_env(contract: dict[str, object]) -> dict[str, str]:
    result = dict(os.environ)
    for name in contract["unset"]:
        result.pop(name, None)
    result.update(contract["set"])
    return result


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
        yield
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
    except OSError as exc:
        raise LaunchError("cannot create fresh run-root") from exc


def launch(args: argparse.Namespace) -> int:
    commit = _source_commit()
    python, runner, root = _validate_inputs(args)
    paths = _paths(root)
    argv = _child_argv(python, runner, paths, commit, args.namespace)
    contract = _env_contract(args.gpu_index)
    if args.dry_run:
        print(json.dumps({"argv": argv, "env": contract}, sort_keys=True,
                         separators=(",", ":")))
        return 0
    with _exclusive_lock(args.lock_file):
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
                    argv, cwd=REPO_ROOT, env=_child_env(contract),
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True,
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
