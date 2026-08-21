#!/usr/bin/env python3
"""Launch one fresh Isaac Full-A H48 run through the existing Kit boot owner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path("/workspace")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
TRAIN_RELATIVE = Path("hope_training/whole_body_tracking/scripts/train.py")
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
LD_LIBRARY_PATH = (
    "/workspace/franco/runtime_assets/libopengl_noble_1_7_0/usr/lib/"
    "x86_64-linux-gnu:/workspace/franco/runtime_assets/libglu_af791d1e"
)


class LaunchError(RuntimeError):
    pass


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


def _paths(root: Path) -> dict[str, Path]:
    return {
        "asset_dir": root / "asset",
        "asset": root / "asset" / "model.usd",
        "tmp": root / "tmp",
        "pycache": root / "pycache",
        "runtime_receipt": root / "train-runtime.receipt",
        "run_log": root / "run.log",
        "launch_state": root / "kit_boot.launch",
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


def _create_root(root: Path, paths: dict[str, Path], asset_package: Path) -> None:
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.parent.resolve(strict=True) != root.parent or os.path.lexists(root):
        raise LaunchError("run-root changed before creation")
    try:
        os.mkdir(root, 0o700)
        for name in ("tmp", "pycache"):
            os.mkdir(paths[name], 0o700)
        shutil.copytree(asset_package, paths["asset_dir"], symlinks=False)
        os.chmod(paths["asset"], 0o400)
    except OSError as exc:
        raise LaunchError("cannot create fresh Isaac run-root") from exc
    if _sha256(paths["asset"]) != PINNED_A3_USD_SHA256:
        raise LaunchError("copied A3 USD digest differs")


def _child_argv(isaac_python: Path, train: Path, namespace: str) -> list[str]:
    return [
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
        "num_envs=4096",
        f"run_name={namespace}-DIAGNOSTIC_UNAUTHORIZED",
        "checkpoint_path=null",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false",
    ]


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
) -> dict[str, str]:
    return {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
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
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
        "PYTHONPATH": _pythonpath(isaaclab, venv_site),
        "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
    }


def _launcher_env(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
        "KIT_BOOT_MARKER": "Learning iteration",
        "KIT_BOOT_TIMEOUT_S": "900",
        "KIT_BOOT_STALE_TIMEOUT_S": "180",
        "KIT_BOOT_POLL_S": "5",
        "KIT_BOOT_STATE_FILE": str(paths["launch_state"]),
        "KIT_WAIT_FOR_COMPLETION": "0",
    }


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


def _verify_started(paths: dict[str, Path]) -> tuple[int, int]:
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
    return pid, pgid


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
    root = _validate_run_root(args.run_root, args.namespace)
    if args.gpu_index < 0:
        raise LaunchError("gpu-index must be nonnegative")
    if re.fullmatch(r"GPU-[A-Za-z0-9-]{8,}", args.expected_gpu_uuid) is None:
        raise LaunchError("expected-gpu-uuid format differs")
    paths = _paths(root)
    child_argv = _child_argv(isaac_python, train, args.namespace)
    runtime_env = _runtime_env(
        gpu_uuid=args.expected_gpu_uuid,
        paths=paths,
        isaaclab=isaaclab,
        venv_site=venv_site,
        rsl_sha256=rsl_sha256,
    )
    if args.dry_run:
        print(json.dumps({
            "source_commit": commit,
            "argv": child_argv,
            "launcher_env": _launcher_env(paths),
            "runtime_env": runtime_env,
            "run_root": str(root),
        }, sort_keys=True, separators=(",", ":")))
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
        command = [str(kit_launcher), str(paths["run_log"]), *clean_child]
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_launcher_env(paths),
                stdin=subprocess.DEVNULL,
                check=False,
                pass_fds=(gpu_lock, *runtime_descriptors),
            )
        except OSError as exc:
            raise LaunchError("cannot start the Kit boot launcher") from exc
        if result.returncode != 0:
            raise LaunchError(f"Kit boot launcher failed rc={result.returncode}")
        pid, pgid = _verify_started(paths)
        print(json.dumps({
            "diagnostic_unauthorized": True,
            "gpu_uuid": args.expected_gpu_uuid,
            "namespace": args.namespace,
            "pid": pid,
            "pgid": pgid,
            "run_root": str(root),
            "source_commit": commit,
            "status": "RUNNING",
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
        print(f"launch_isaac_full_mdp_successor: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
