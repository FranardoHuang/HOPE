#!/usr/bin/env python3
"""Small YAML-driven RunPod training queue.

The queue is intentionally an exploratory-run tool, not a formal evidence
attestor.  One YAML row binds the motion/action, its train bank and exam, the
source checkout, base recipe, causal delta, seed, budget, checkpoint cadence,
and resource policy.  ``plan`` and ``launch-next`` are dry-run by default.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
from typing import Any

import yaml


class QueueError(RuntimeError):
    """The lightweight queue contract or a launch preflight failed."""


@dataclass(frozen=True)
class Slot:
    pod: str
    gpu: int
    ordinal: int
    capacity: int

    @property
    def name(self) -> str:
        return f"{self.pod}/gpu{self.gpu}"


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
READY = "ready"
BLOCKED = "blocked"
TERMINAL = {"complete", "rejected"}
CONFIRM = "SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB"
WARMUP_CONFIRM = "SIM_ONLY_LAUNCH_ONE_BOOT_WARMUP"
FULL_SCENE_PROBE_CONFIRM = "SIM_ONLY_LAUNCH_ONE_FULL_SCENE_PROBE"
FINALIZE_FULL_SCENE_PROBE_CONFIRM = "SIM_ONLY_FINALIZE_ONE_FULL_SCENE_PROBE"
ATTEST_CONFIRM = "SIM_ONLY_ATTEST_ONE_LEAN_QUEUE_MILESTONE"
PREPARE_SOURCE_ASSET_CONFIRM = "SIM_ONLY_PREPARE_ONE_LEAN_QUEUE_SOURCE_ASSET"
ZERO_COMMIT = "0" * 40
GLOBAL_SCHEDULER_LOCK = Path("/tmp/hope_lean_training_queue.global.lock")
ISAAC_PYTHON = "/workspace/hope_isaac_venv/bin/python"
WBT_RELATIVE = "hope_training/whole_body_tracking"
SETUP_RELATIVE = "setup_train_env.sh"
ENTRYPOINT_RELATIVE = "scripts/train.py"
QUEUE_RUNTIME_RELATIVE = "scripts/lean_queue_runtime.py"
ATTESTOR_RUNTIME_ROOT = "/workspace/codexschema/lean_queue_attestor_runtime"
FULL_SCENE_PROBE_RUNTIME_RELATIVE = "scripts/full_scene_probe_runtime.py"
KIT_LAUNCHER_RELATIVE = "scripts/launch_kit_training_locked.sh"
KIT_BOOT_MARKER = "Learning iteration"
KIT_BOOT_TIMEOUT_SECONDS = 900
WARMUP_BOOT_TIMEOUT_SECONDS = 180
WARMUP_NUM_ENVS = 1
WARMUP_MAX_ITERATIONS = 2
WARMUP_SAVE_INTERVAL = 1
FULL_SCENE_PROBE_MAX_ITERATIONS = 2
FULL_SCENE_PROBE_SAVE_INTERVAL = 1
FULL_SCENE_PROBE_STALE_TIMEOUT_SECONDS = 180
SOURCE_ASSET_RECEIPT_ROOT = PurePosixPath(
    "/workspace/codexschema/lean_training_source_asset_receipts"
)
SOURCE_ASSET_STAGING_ROOT = PurePosixPath(
    "/workspace/codexschema/lean_training_source_asset_staging"
)
SOURCE_ASSET_URDF_RELATIVE_PATH = "urdf/model.urdf"
SOURCE_ASSET_UNIQUE_MESH_REFERENCES = 43
GPU_LAUNCH_LOCK_FD = 8
HARNESS_OWNED_OVERRIDE_KEYS = {
    "seed",
    "num_envs",
    "max_iterations",
    "algo.runner.save_interval",
    "run_name",
    "device",
    "training_launch_claim_sha256",
    "training_queue_claim_path",
    "training_run_binding_path",
}
UNIQUE_NUMERIC_PID_AWK = (
    r'{gsub(/^[ \t]+|[ \t]+$/, "", $0); '
    r'if ($0 ~ /^[0-9]+$/) seen[$0]=1} END {print length(seen)}'
)

_RUNTIME_SNAPSHOT_PROGRAM = r'''import base64
import hashlib
import json
import os
import stat
import sys

raw = base64.b64decode(sys.argv[1], validate=True)
expected = sys.argv[2]
root = sys.argv[3]
if len(expected) != 64 or hashlib.sha256(raw).hexdigest() != expected:
    raise RuntimeError("runtime payload SHA mismatch")
base, root_name = os.path.split(root)
if not base or not root_name or os.path.normpath(root) != root:
    raise RuntimeError("runtime snapshot root must be normalized absolute")
directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
base_fd = os.open(base, directory_flags)
try:
    try:
        os.mkdir(root_name, 0o755, dir_fd=base_fd)
    except FileExistsError:
        pass
    root_fd = os.open(root_name, directory_flags, dir_fd=base_fd)
finally:
    os.close(base_fd)
try:
    try:
        os.mkdir(expected, 0o755, dir_fd=root_fd)
    except FileExistsError:
        pass
    sha_fd = os.open(expected, directory_flags, dir_fd=root_fd)
finally:
    os.close(root_fd)

target = "lean_queue_runtime.py"
signature = lambda value: (
    value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns
)

def verify_target():
    before = os.stat(target, dir_fd=sha_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222:
        raise RuntimeError("runtime snapshot must be a read-only regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, dir_fd=sha_fd)
    try:
        opened = os.fstat(fd)
        if signature(opened) != signature(before):
            raise RuntimeError("runtime snapshot changed while opening")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    outside = os.stat(target, dir_fd=sha_fd, follow_symlinks=False)
    if signature(before) != signature(after) or signature(before) != signature(outside):
        raise RuntimeError("runtime snapshot changed while reading")
    observed = b"".join(chunks)
    if observed != raw or hashlib.sha256(observed).hexdigest() != expected:
        raise RuntimeError("existing runtime snapshot bytes mismatch")

try:
    verify_target()
    state = "existing_exact"
except FileNotFoundError:
    temporary = f".{target}.tmp.{os.getpid()}"
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(temporary, flags, 0o400, dir_fd=sha_fd)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        try:
            os.link(
                temporary, target, src_dir_fd=sha_fd, dst_dir_fd=sha_fd,
                follow_symlinks=False,
            )
            state = "created_no_replace"
            os.fsync(sha_fd)
        except FileExistsError:
            state = "race_existing_exact"
    finally:
        os.unlink(temporary, dir_fd=sha_fd)
    verify_target()
finally:
    os.close(sha_fd)

path = root + "/" + expected + "/" + target
print(json.dumps({"path": path, "sha256": expected, "state": state}, sort_keys=True))
'''


# This program is sent as one quoted ``python3 -c`` argument to exactly one
# selected Pod.  Keeping the mutation in one process makes the source-specific
# flock, no-clobber staging, atomic publish, and receipt publication one
# auditable state machine.  It deliberately has no retry loop.
SOURCE_ASSET_PROGRAM = r'''import base64
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET


class AssetError(RuntimeError):
    pass


def canonical_sha256(value):
    data = json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(checkout, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AssetError(f"Git validation failed for {checkout}: {exc}") from exc


def safe_relative(value, label):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AssetError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in (".", "..") or ".." in path.parts:
        raise AssetError(f"unsafe {label}: {value!r}")
    return path


def require_checkout(path, commit, label):
    if path.is_symlink() or not path.is_dir():
        raise AssetError(f"{label} checkout is missing or a symlink: {path}")
    if git(path, "rev-parse", "HEAD") != commit:
        raise AssetError(f"{label} checkout is at the wrong commit")
    if git(path, "status", "--porcelain"):
        raise AssetError(f"{label} checkout is dirty")


def checked_join(root, relative, label, *, leaf_may_be_missing):
    root = root.resolve()
    current = root
    parts = safe_relative(relative, label).parts
    for index, part in enumerate(parts):
        current = current / part
        exists = os.path.lexists(current)
        if not exists:
            if leaf_may_be_missing and index == len(parts) - 1:
                return current
            raise AssetError(f"{label} component is missing: {current}")
        result = current.lstat()
        if stat.S_ISLNK(result.st_mode):
            raise AssetError(f"{label} contains a symlink component: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(result.st_mode):
            raise AssetError(f"{label} parent is not a directory: {current}")
    try:
        current.resolve().relative_to(root)
    except ValueError as exc:
        raise AssetError(f"{label} escapes checkout") from exc
    return current


def inventory(root):
    if not os.path.lexists(root):
        raise AssetError(f"asset root is missing: {root}")
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise AssetError(f"asset root is not a real directory: {root}")
    rows = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(dirnames):
            path = current_path / name
            result = path.lstat()
            if stat.S_ISLNK(result.st_mode):
                raise AssetError(f"asset tree contains symlink: {path}")
            if not stat.S_ISDIR(result.st_mode):
                raise AssetError(f"asset tree contains special directory entry: {path}")
        for name in sorted(filenames):
            path = current_path / name
            result = path.lstat()
            if stat.S_ISLNK(result.st_mode):
                raise AssetError(f"asset tree contains symlink: {path}")
            if not stat.S_ISREG(result.st_mode):
                raise AssetError(f"asset tree contains special file: {path}")
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": result.st_size,
                "sha256": sha256_file(path),
            })
    rows.sort(key=lambda row: row["relative_path"])
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": canonical_sha256({"files": rows}),
    }


def verify_urdf(root, relative, expected_unique):
    urdf = checked_join(
        root, relative, "asset URDF", leaf_may_be_missing=False
    )
    result = urdf.lstat()
    if not stat.S_ISREG(result.st_mode):
        raise AssetError("asset URDF is not a regular file")
    try:
        xml_root = ET.parse(urdf).getroot()
    except (OSError, ET.ParseError) as exc:
        raise AssetError(f"cannot parse asset URDF: {exc}") from exc
    refs = []
    for element in xml_root.iter():
        if element.tag.rsplit("}", 1)[-1] != "mesh":
            continue
        filename = element.attrib.get("filename")
        if filename:
            refs.append(filename)
    unique = sorted(set(refs))
    if len(unique) != expected_unique:
        raise AssetError(
            f"asset URDF unique mesh references changed: {len(unique)} != {expected_unique}"
        )
    resolved = 0
    root_resolved = root.resolve()
    for reference in unique:
        if reference.startswith("package://") or "\x00" in reference:
            raise AssetError(f"unsupported asset mesh reference: {reference!r}")
        candidate = (urdf.parent / reference).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise AssetError(f"mesh reference escapes asset root: {reference!r}") from exc
        if not os.path.lexists(candidate):
            raise AssetError(f"mesh reference is missing: {reference!r}")
        item = candidate.lstat()
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
            raise AssetError(f"mesh reference is not a regular non-symlink file: {reference!r}")
        resolved += 1
    return {
        "mesh_reference_occurrences": len(refs),
        "unique_mesh_references": len(unique),
        "resolved_regular_meshes": resolved,
    }


def verify_tree(root, expected, urdf_relative, expected_unique):
    actual = inventory(root)
    if actual != expected:
        raise AssetError(f"asset tree inventory drift: actual={actual} expected={expected}")
    urdf = verify_urdf(root, urdf_relative, expected_unique)
    if urdf["resolved_regular_meshes"] != expected_unique:
        raise AssetError("asset URDF reference closure is incomplete")
    return actual, urdf


def require_gitignored(checkout, relative):
    completed = subprocess.run(
        ["git", "-C", str(checkout), "check-ignore", "-q", "--", relative],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if completed.returncode != 0:
        raise AssetError("target runtime asset is not Git-ignored")


def exact_source_trainers(source, entrypoint_relative, proc_root):
    expected_entrypoint = os.path.normpath(str(source / entrypoint_relative))
    expected_cwd = os.path.normpath(str((source / entrypoint_relative).parent.parent))
    found = []
    proc_root = Path(proc_root)
    if not proc_root.is_dir() or proc_root.is_symlink():
        raise AssetError("Linux procfs root is missing or a symlink")
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [
                item.decode("utf-8", "surrogateescape")
                for item in (proc / "cmdline").read_bytes().split(b"\0") if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        matched = any(
            os.path.normpath(item) == expected_entrypoint for item in argv
        )
        if not matched and "scripts/train.py" in argv:
            try:
                matched = os.path.normpath(os.readlink(proc / "cwd")) == expected_cwd
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                matched = False
        if matched:
            found.append(int(proc.name))
    return sorted(set(found))


def rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise AssetError("renameat2(RENAME_NOREPLACE) is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    rc = function(
        -100, os.fsencode(source), -100, os.fsencode(target), 1
    )
    if rc != 0:
        error = ctypes.get_errno()
        raise AssetError(
            f"atomic no-replace asset publish failed: {os.strerror(error)}"
        )


def publish_receipt(path, document, *, allow_create):
    encoded = (
        json.dumps(
            document, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ) + "\n"
    ).encode("utf-8")
    if os.path.lexists(path):
        result = path.lstat()
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
            raise AssetError("source asset receipt is not a regular non-symlink file")
        if path.read_bytes() != encoded:
            raise AssetError("source asset receipt exists with different bytes")
        return "existing_exact"
    if not allow_create:
        raise AssetError(
            "exact source asset receipt is missing; run explicit prepare first"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return "created"


def main():
    if len(sys.argv) != 2:
        raise AssetError("one base64 JSON runtime specification is required")
    try:
        spec = json.loads(base64.b64decode(sys.argv[1], validate=True))
    except Exception as exc:
        raise AssetError(f"invalid runtime specification: {exc}") from exc
    mode = spec.get("mode")
    if mode not in ("prepare", "doctor"):
        raise AssetError("source asset mode must be prepare or doctor")
    source_spec = spec["source"]
    contract = spec["contract"]
    donor_spec = contract["donor"]
    source = Path(source_spec["checkout"])
    donor_checkout = Path(donor_spec["checkout"])
    receipt = Path(spec["receipt_path"])
    staging = Path(spec["staging_path"])
    lock_path = Path(spec["lock_path"])
    require_checkout(source, source_spec["commit"], "target source")
    require_checkout(donor_checkout, donor_spec["commit"], "donor")
    target = checked_join(
        source, contract["target_relative_path"], "target runtime asset",
        leaf_may_be_missing=True,
    )
    donor = checked_join(
        donor_checkout, donor_spec["relative_path"], "donor runtime asset",
        leaf_may_be_missing=False,
    )
    source_resolved = source.resolve()
    try:
        receipt.resolve(strict=False).relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise AssetError("source asset receipt must remain outside source checkout")
    expected = {
        "file_count": contract["file_count"],
        "total_file_bytes": contract["total_file_bytes"],
        "tree_content_sha256": contract["tree_content_sha256"],
    }
    donor_inventory, donor_urdf = verify_tree(
        donor, expected, spec["urdf_relative_path"],
        spec["expected_unique_mesh_references"],
    )
    require_gitignored(source, contract["target_relative_path"])
    contract_sha256 = canonical_sha256(contract)
    receipt_content = {
        "schema_version": 1,
        "pod": spec["pod"],
        "source": source_spec,
        "ignored_runtime_asset": contract,
        "ignored_runtime_asset_sha256": contract_sha256,
        "target_path": str(target),
        "inventory": donor_inventory,
        "urdf_reference_closure": donor_urdf,
        "target_gitignored": True,
        "symlinks_present": False,
    }
    receipt_document = {
        "schema_version": 1,
        "content": receipt_content,
        "content_sha256": canonical_sha256(receipt_content),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AssetError("source-specific asset lock is already held") from exc
        materialized = False
        if mode == "prepare":
            trainers = exact_source_trainers(
                source, spec["entrypoint_relative_path"], spec["proc_root"]
            )
            if trainers:
                raise AssetError(
                    f"exact source has live trainers; refusing asset mutation: {trainers}"
                )
        if os.path.lexists(target):
            target_inventory, target_urdf = verify_tree(
                target, expected, spec["urdf_relative_path"],
                spec["expected_unique_mesh_references"],
            )
        else:
            if mode != "prepare":
                raise AssetError("target runtime asset is missing; run explicit prepare first")
            if os.path.lexists(staging):
                raise AssetError(
                    f"no-clobber staging path already exists; preserve and diagnose: {staging}"
                )
            staging.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(donor, staging, symlinks=False)
            staged_inventory, staged_urdf = verify_tree(
                staging, expected, spec["urdf_relative_path"],
                spec["expected_unique_mesh_references"],
            )
            # Re-read the ignored donor after the copy.  A concurrent mutation
            # cannot be hidden by a byte-exact staging tree.
            verify_tree(
                donor, expected, spec["urdf_relative_path"],
                spec["expected_unique_mesh_references"],
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if staging.stat().st_dev != target.parent.stat().st_dev:
                raise AssetError("staging and target are on different filesystems")
            rename_noreplace(staging, target)
            materialized = True
            target_inventory, target_urdf = verify_tree(
                target, expected, spec["urdf_relative_path"],
                spec["expected_unique_mesh_references"],
            )
            if target_inventory != staged_inventory or target_urdf != staged_urdf:
                raise AssetError("published target differs from verified staging tree")
        if target_inventory != donor_inventory or target_urdf != donor_urdf:
            raise AssetError("target and donor asset closures differ")
        require_gitignored(source, contract["target_relative_path"])
        receipt_state = publish_receipt(
            receipt, receipt_document, allow_create=(mode == "prepare")
        )
    print(json.dumps({
        "status": "SOURCE_ASSET_OK",
        "mode": mode,
        "materialized": materialized,
        "receipt_state": receipt_state,
        "receipt_path": str(receipt),
        "contract_sha256": contract_sha256,
        "inventory": target_inventory,
        "urdf_reference_closure": target_urdf,
    }, sort_keys=True))


try:
    main()
except AssetError as exc:
    print(f"SOURCE_ASSET_ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
'''


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QueueError(f"{label} must be a list")
    return value


def _text(value: Any, label: str, *, safe_id: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueueError(f"{label} must be a non-empty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise QueueError(f"{label} must be one line")
    if safe_id and not SAFE_ID.fullmatch(value):
        raise QueueError(f"{label} is not a safe identifier")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise QueueError(f"{label} must be a positive integer")
    return value


def _safe_relative_contract_path(value: Any, label: str) -> str:
    path = _text(value, label)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or path in {".", ".."} or ".." in parsed.parts:
        raise QueueError(f"{label} must be a safe relative path")
    return path


def _validate_ignored_runtime_asset(
    source: dict[str, Any], label: str
) -> dict[str, Any] | None:
    """Validate the optional ignored asset closure bound by a source row."""

    raw = source.get("ignored_runtime_asset")
    if raw is None:
        return None
    asset = _mapping(raw, f"{label}.ignored_runtime_asset")
    expected_keys = {
        "target_relative_path",
        "donor",
        "file_count",
        "total_file_bytes",
        "tree_content_sha256",
        "symlinks_forbidden",
        "target_must_be_gitignored",
    }
    if set(asset) != expected_keys:
        raise QueueError(
            f"{label}.ignored_runtime_asset keys changed: "
            f"expected={sorted(expected_keys)} got={sorted(asset)}"
        )
    _safe_relative_contract_path(
        asset.get("target_relative_path"),
        f"{label}.ignored_runtime_asset.target_relative_path",
    )
    donor = _mapping(asset.get("donor"), f"{label}.ignored_runtime_asset.donor")
    if set(donor) != {"checkout", "commit", "relative_path"}:
        raise QueueError(
            f"{label}.ignored_runtime_asset.donor must bind checkout/commit/relative_path"
        )
    _ready_workspace_path(
        donor.get("checkout"), f"{label}.ignored_runtime_asset.donor.checkout"
    )
    donor_commit = _text(
        donor.get("commit"), f"{label}.ignored_runtime_asset.donor.commit"
    )
    if not COMMIT.fullmatch(donor_commit):
        raise QueueError(
            f"{label}.ignored_runtime_asset.donor.commit must be a full Git commit"
        )
    _safe_relative_contract_path(
        donor.get("relative_path"),
        f"{label}.ignored_runtime_asset.donor.relative_path",
    )
    _positive_int(
        asset.get("file_count"), f"{label}.ignored_runtime_asset.file_count"
    )
    _positive_int(
        asset.get("total_file_bytes"),
        f"{label}.ignored_runtime_asset.total_file_bytes",
    )
    digest = _text(
        asset.get("tree_content_sha256"),
        f"{label}.ignored_runtime_asset.tree_content_sha256",
    )
    if not SHA256.fullmatch(digest):
        raise QueueError(
            f"{label}.ignored_runtime_asset.tree_content_sha256 must be SHA-256"
        )
    if asset.get("symlinks_forbidden") is not True:
        raise QueueError(f"{label}.ignored_runtime_asset must forbid symlinks")
    if asset.get("target_must_be_gitignored") is not True:
        raise QueueError(f"{label}.ignored_runtime_asset target must remain Git-ignored")
    return asset


def _ready_workspace_path(value: Any, label: str) -> str:
    path = _text(value, label)
    parsed = PurePosixPath(path)
    lowered = path.lower()
    if not parsed.is_absolute() or not path.startswith("/workspace/"):
        raise QueueError(f"{label} for a ready job must be an absolute /workspace path")
    if ".." in parsed.parts:
        raise QueueError(f"{label} for a ready job must not contain ..")
    if any(token in lowered for token in ("placeholder", "/path/to/", "<", ">")):
        raise QueueError(f"{label} for a ready job is still a placeholder")
    return path


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _override_key(argument: str, label: str) -> str:
    """Return one normalized Hydra key or fail before any SSH.

    Queue recipes are scalar override lists, not Hydra control surfaces.  In
    particular, a job may not smuggle in multirun/config flags, deletion,
    environment-dependent interpolation, or a second value for the same key.
    """

    if argument.startswith("-"):
        raise QueueError(f"{label} must not contain a Hydra control flag")
    if argument.startswith("~"):
        raise QueueError(f"{label} must not contain Hydra deletion syntax")
    if "${" in argument:
        raise QueueError(f"{label} must not contain Hydra interpolation/resolvers")
    raw_key, separator, _value = argument.partition("=")
    if not separator:
        raise QueueError(f"{label} must be one Hydra key=value override")
    if raw_key.startswith("++"):
        raw_key = raw_key[2:]
    elif raw_key.startswith("+"):
        raw_key = raw_key[1:]
    if raw_key.startswith("~") or not HYDRA_KEY.fullmatch(raw_key):
        raise QueueError(f"{label} has an unsupported Hydra key")
    if raw_key == "hydra" or raw_key.startswith("hydra."):
        raise QueueError(f"{label} must not modify Hydra runtime configuration")
    return raw_key


def _generated_override_key(raw_key: Any, label: str) -> str:
    key = _text(raw_key, label)
    if "=" in key:
        raise QueueError(f"{label} must be one Hydra key without a value")
    return _override_key(f"{key}=__harness_value__", label)


def _compile_recipe_override_keys(job: dict[str, Any], label: str) -> set[str]:
    recipe = _mapping(job.get("recipe"), f"{label}.recipe")
    base = _list(recipe.get("base"), f"{label}.recipe.base")
    delta = _list(recipe.get("delta"), f"{label}.recipe.delta")
    if not base:
        raise QueueError(f"{label}.recipe.base must not be empty")

    owned = set(HARNESS_OWNED_OVERRIDE_KEYS)
    motion = _mapping(job.get("motion"), f"{label}.motion")
    bindings = _mapping(motion.get("bindings"), f"{label}.motion.bindings")
    owned.update(
        _generated_override_key(key, f"{label}.motion binding") for key in bindings
    )
    bank = _mapping(job.get("bank"), f"{label}.bank")
    owned.add(
        _generated_override_key(
            bank.get("train_arg"), f"{label}.bank.train_arg"
        )
    )

    compiled: set[str] = set()
    for number, raw in enumerate([*base, *delta]):
        argument = _text(raw, f"{label}.recipe argument {number}")
        key = _override_key(argument, f"{label}.recipe argument {number}")
        if key in compiled:
            raise QueueError(f"{label}.recipe sets Hydra key {key!r} more than once")
        if key in owned:
            raise QueueError(f"{label}.recipe may not set harness-owned key {key!r}")
        compiled.add(key)
    return compiled


def load_queue(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise QueueError(f"cannot read queue YAML {path}: {exc}") from exc
    queue = _mapping(raw, "queue")
    if queue.get("schema_version") != 1:
        raise QueueError("schema_version must be 1")
    if queue.get("simulation_only") is not True:
        raise QueueError("simulation_only must be true")
    launch_authorized = queue.get("launch_authorized")
    if type(launch_authorized) is not bool:
        raise QueueError("launch_authorized must be an explicit true or false")

    ssh = _mapping(queue.get("ssh"), "ssh")
    _text(ssh.get("key"), "ssh.key")
    pods = _mapping(queue.get("pods"), "pods")
    if list(pods) != ["pod1", "pod2"]:
        raise QueueError("pods must be ordered exactly pod1, pod2")
    for pod_name, expected_capacity in (("pod1", 4), ("pod2", 3)):
        pod = _mapping(pods[pod_name], pod_name)
        _text(pod.get("host"), f"{pod_name}.host")
        _positive_int(pod.get("port"), f"{pod_name}.port")
        if pod.get("gpus") != [0, 1, 2]:
            raise QueueError(f"{pod_name}.gpus must be [0, 1, 2]")
        if pod.get("max_trainers_per_gpu") != expected_capacity:
            raise QueueError(
                f"{pod_name}.max_trainers_per_gpu must be {expected_capacity}"
            )

    dispatch_pods_value = queue.get("dispatch_pods", list(pods))
    dispatch_pods = _list(dispatch_pods_value, "dispatch_pods")
    if not dispatch_pods:
        raise QueueError("dispatch_pods must contain at least one Pod")
    if any(not isinstance(name, str) or name not in pods for name in dispatch_pods):
        raise QueueError("dispatch_pods may only name configured Pods")
    if len(dispatch_pods) != len(set(dispatch_pods)):
        raise QueueError("dispatch_pods must not contain duplicates")
    queue["dispatch_pods"] = list(dispatch_pods)

    if "runner" in queue:
        raise QueueError(
            "runner paths are source-pinned; queue YAML must not override setup/train/launcher"
        )

    jobs = _list(queue.get("jobs"), "jobs")
    if not jobs:
        raise QueueError("jobs must not be empty")
    seen: set[str] = set()
    run_names: set[str] = set()
    run_dirs: set[PurePosixPath] = set()
    ready_layouts: list[tuple[str, PurePosixPath, PurePosixPath]] = []
    for index, value in enumerate(jobs):
        job = _mapping(value, f"jobs[{index}]")
        job_id = _text(job.get("id"), f"jobs[{index}].id", safe_id=True)
        if job_id in seen:
            raise QueueError(f"duplicate job id: {job_id}")
        seen.add(job_id)
        _text(job.get("human_name"), f"{job_id}.human_name")
        _text(job.get("action"), f"{job_id}.action", safe_id=True)
        status = job.get("status")
        if status not in {READY, BLOCKED, *TERMINAL}:
            raise QueueError(f"{job_id}.status must be ready/blocked/complete/rejected")
        blocker = job.get("blocker")
        if status in {BLOCKED, "rejected"}:
            _text(blocker, f"{job_id}.blocker")
        elif blocker not in (None, ""):
            raise QueueError(
                f"{job_id}.blocker must be empty unless status=blocked/rejected"
            )

        action = job["action"]
        motion = _mapping(job.get("motion"), f"{job_id}.motion")
        if motion.get("action") != action:
            raise QueueError(f"{job_id}.motion.action must equal job action")
        bindings = _mapping(motion.get("bindings"), f"{job_id}.motion.bindings")
        if not bindings:
            raise QueueError(f"{job_id}.motion.bindings must not be empty")
        for arg, asset_path in bindings.items():
            _text(arg, f"{job_id}.motion arg", safe_id=True)
            _text(asset_path, f"{job_id}.motion.{arg}")

        bank = _mapping(job.get("bank"), f"{job_id}.bank")
        if bank.get("action") != action:
            raise QueueError(f"{job_id}.bank.action must equal job action")
        _text(bank.get("train_path"), f"{job_id}.bank.train_path")
        _text(bank.get("train_arg"), f"{job_id}.bank.train_arg")
        exam = _mapping(job.get("exam"), f"{job_id}.exam")
        if exam.get("action") != action:
            raise QueueError(f"{job_id}.exam.action must equal job action")
        _text(exam.get("path"), f"{job_id}.exam.path")
        _text(exam.get("family"), f"{job_id}.exam.family", safe_id=True)

        source = _mapping(job.get("source"), f"{job_id}.source")
        unknown_source_keys = set(source) - {
            "checkout", "commit", "ignored_runtime_asset"
        }
        if unknown_source_keys:
            raise QueueError(
                f"{job_id}.source has unsupported keys: {sorted(unknown_source_keys)}"
            )
        _text(source.get("checkout"), f"{job_id}.source.checkout")
        commit = _text(source.get("commit"), f"{job_id}.source.commit")
        if not COMMIT.fullmatch(commit):
            raise QueueError(f"{job_id}.source.commit must be a full Git commit")
        _validate_ignored_runtime_asset(source, f"{job_id}.source")
        runtime_binding = job.get("runtime_binding", False)
        if type(runtime_binding) is not bool:
            raise QueueError(f"{job_id}.runtime_binding must be true or false")

        _compile_recipe_override_keys(job, job_id)
        if runtime_binding:
            for raw in [*job["recipe"]["base"], *job["recipe"]["delta"]]:
                if _override_key(raw, f"{job_id}.recipe") != "checkpoint_path":
                    continue
                value = raw.partition("=")[2].strip().lower()
                if value not in {"null", "none"}:
                    raise QueueError(
                        f"{job_id}.runtime_binding currently supports fresh runs only; "
                        "checkpoint_path must be null"
                    )

        if type(job.get("seed")) is not int or job["seed"] < 0:
            raise QueueError(f"{job_id}.seed must be a non-negative integer")
        budget = _mapping(job.get("budget"), f"{job_id}.budget")
        _positive_int(budget.get("num_envs"), f"{job_id}.budget.num_envs")
        iterations = _positive_int(
            budget.get("max_iterations"), f"{job_id}.budget.max_iterations"
        )
        save_interval = _positive_int(
            budget.get("save_interval"), f"{job_id}.budget.save_interval"
        )
        milestones = _list(job.get("milestones"), f"{job_id}.milestones")
        if not milestones or any(type(x) is not int or x <= 0 for x in milestones):
            raise QueueError(f"{job_id}.milestones must contain positive integers")
        if milestones != sorted(set(milestones)) or milestones[-1] >= iterations:
            raise QueueError(
                f"{job_id}.milestones must be unique, sorted, and strictly below "
                "max_iterations (fresh RSL checkpoints end at max_iterations-1)"
            )
        if any(x % save_interval for x in milestones):
            raise QueueError(f"{job_id}.milestones must align with save_interval")

        resource = _mapping(job.get("resource"), f"{job_id}.resource")
        unknown_resource_keys = set(resource) - {
            "policy", "preferred_slot", "required_slot"
        }
        if unknown_resource_keys:
            raise QueueError(
                f"{job_id}.resource has unsupported keys: {sorted(unknown_resource_keys)}"
            )
        policy = resource.get("policy")
        if policy not in {"six_gpu_round_robin", "dispatch_gpu_round_robin"}:
            raise QueueError(
                f"{job_id}.resource must bind a supported round-robin policy"
            )
        if policy == "six_gpu_round_robin" and dispatch_pods != list(pods):
            raise QueueError(
                f"{job_id}.resource six_gpu_round_robin requires both configured Pods"
            )
        preferred_slot = resource.get("preferred_slot")
        required_slot = resource.get("required_slot")
        if preferred_slot is not None and required_slot is not None:
            raise QueueError(
                f"{job_id}.resource cannot set both preferred_slot and required_slot"
            )
        dispatch_slot_names = {
            f"{pod_name}/gpu{gpu}"
            for pod_name in dispatch_pods
            for gpu in pods[pod_name]["gpus"]
        }
        for slot_key, slot_value in (
            ("preferred_slot", preferred_slot),
            ("required_slot", required_slot),
        ):
            if slot_value is None:
                continue
            slot_value = _text(
                slot_value, f"{job_id}.resource.{slot_key}"
            )
            if slot_value not in dispatch_slot_names:
                raise QueueError(
                    f"{job_id}.resource.{slot_key} is not dispatch-enabled"
                )
        run_name = _text(job.get("run_name"), f"{job_id}.run_name", safe_id=True)
        if run_name in run_names:
            raise QueueError(f"duplicate run_name: {run_name}")
        run_names.add(run_name)
        run_dir = PurePosixPath(_text(job.get("run_dir"), f"{job_id}.run_dir"))
        if run_dir in run_dirs:
            raise QueueError(f"duplicate run_dir: {run_dir}")
        run_dirs.add(run_dir)
        source_path = PurePosixPath(source["checkout"])
        if run_dir == source_path or source_path in run_dir.parents:
            raise QueueError(f"{job_id}.run_dir must not equal or be inside its source")
        if status == READY:
            if commit == ZERO_COMMIT:
                raise QueueError(f"{job_id}.source.commit is an all-zero placeholder")
            ready_paths = {
                "source.checkout": source["checkout"],
                **{
                    f"motion.{arg}": asset_path
                    for arg, asset_path in bindings.items()
                },
                "bank.train_path": bank["train_path"],
                "exam.path": exam["path"],
                "run_dir": job["run_dir"],
            }
            normalized = [
                _ready_workspace_path(path_value, f"{job_id}.{path_label}")
                for path_label, path_value in ready_paths.items()
            ]
            input_paths = normalized[1:-1]
            if len(set(input_paths)) != len(input_paths):
                raise QueueError(f"{job_id} has duplicate motion/bank/exam identities")
            ready_layouts.append(
                (job_id, PurePosixPath(normalized[0]), PurePosixPath(normalized[-1]))
            )
    for run_job_id, _run_source, run_path in ready_layouts:
        for source_job_id, source_path, _source_run in ready_layouts:
            if run_path == source_path or source_path in run_path.parents:
                raise QueueError(
                    f"{run_job_id}.run_dir must not equal or be inside ready source "
                    f"checkout for {source_job_id}"
                )
    return queue


def slots(queue: dict[str, Any]) -> list[Slot]:
    result: list[Slot] = []
    ordinal = 0
    # One full round over every dispatch-enabled GPU before any GPU receives
    # its next trainer.  Disabled Pods remain claim-observable but can never be
    # selected for a new launch.
    for pod_name in queue.get("dispatch_pods", list(queue["pods"])):
        pod = queue["pods"][pod_name]
        for gpu in pod["gpus"]:
            result.append(
                Slot(pod_name, gpu, ordinal, pod["max_trainers_per_gpu"])
            )
            ordinal += 1
    return result


def _ssh_prefix(queue: dict[str, Any], pod_name: str) -> list[str]:
    pod = queue["pods"][pod_name]
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
        "-i", str(Path(queue["ssh"]["key"]).expanduser()), "-p", str(pod["port"]),
        f"root@{pod['host']}",
    ]


def _run_ssh(
    queue: dict[str, Any], pod_name: str, remote: str, *, timeout: int = 30,
    phase: str = "remote-command",
) -> str:
    try:
        completed = subprocess.run(
            [*_ssh_prefix(queue, pod_name), f"bash -lc {shlex.quote(remote)}"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise QueueError(
            f"{pod_name} {phase} failed rc={exc.returncode}; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise QueueError(
            f"{pod_name} {phase} timed out after {timeout}s; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc
    except OSError as exc:
        raise QueueError(f"{pod_name} {phase} SSH failed: {exc}") from exc
    return completed.stdout


def live_snapshot(queue: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Read occupancy and claims only from explicitly dispatch-enabled Pods."""

    occupancy: dict[str, int] = {}
    claims: dict[str, dict[str, Any]] = {}
    job_dirs = {job["id"]: job["run_dir"] for job in queue["jobs"]}
    program = f"""import hashlib
import json
from pathlib import Path
import subprocess

jobs = json.loads({json.dumps(json.dumps(job_dirs))})
def lines(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()

def canonical_sha256(value):
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def claim_identity(payload, claim):
    schema = payload.get("schema_version")
    if schema == 1:
        return payload
    if schema != 2:
        raise RuntimeError(f"unsupported claim schema: {{claim}}")
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"claim content is not a mapping: {{claim}}")
    digest = payload.get("content_sha256")
    if canonical_sha256(content) != digest:
        raise RuntimeError(f"claim content digest mismatch: {{claim}}")
    argv_without_claim = content.get("training_argv_without_claim")
    full_argv = payload.get("training_argv")
    if not isinstance(argv_without_claim, list) or not isinstance(full_argv, list):
        raise RuntimeError(f"claim argv is not a list: {{claim}}")
    expected_argv = [*argv_without_claim, f"++training_launch_claim_sha256={{digest}}"]
    if full_argv != expected_argv:
        raise RuntimeError(f"claim full argv does not bind its digest: {{claim}}")
    return content

compute_rows = lines(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
gpu_rows = lines(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
states = {{}}
for job_id, directory in jobs.items():
    root = Path(directory)
    claim = root / "queue_claim.json"
    if claim.is_file():
        payload = json.loads(claim.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"claim is not a mapping: {{claim}}")
        identity = claim_identity(payload, claim)
        state = "claimed"
        if (root / "run.log.launch").is_file():
            state = "launched"
        if (root / "terminal_result.json").is_file():
            state = "terminal"
        states[job_id] = {{
            "state": state,
            "claim_path": str(claim),
            "claim_schema_version": payload.get("schema_version"),
            "claim_content_sha256": payload.get("content_sha256"),
            "claim_job_id": identity.get("job_id"),
            "pod": identity.get("pod"),
            "gpu": identity.get("gpu"),
        }}
print(json.dumps({{"compute_rows": compute_rows, "gpu_rows": gpu_rows, "jobs": states}}, sort_keys=True))
"""
    command = f"python3 -c {shlex.quote(program)}"
    pod_names = tuple(queue["dispatch_pods"])
    with ThreadPoolExecutor(max_workers=len(pod_names)) as pool:
        outputs = dict(
            zip(
                pod_names,
                pool.map(lambda pod: _run_ssh(queue, pod, command), pod_names),
            )
        )
    for pod_name in pod_names:
        try:
            snapshot = json.loads(outputs[pod_name])
        except json.JSONDecodeError as exc:
            raise QueueError(f"{pod_name} returned malformed live snapshot") from exc
        occupancy.update(_parse_gpu_occupancy(pod_name, snapshot))
        for job_id, state_value in _mapping(
            snapshot.get("jobs"), f"{pod_name}.jobs"
        ).items():
            if job_id in claims:
                raise QueueError(f"job {job_id} is claimed on both Pods")
            state = _mapping(state_value, f"{job_id}.state")
            if state.get("claim_job_id") != job_id:
                raise QueueError(f"{job_id} claim binds a different job id")
            claim_pod = _text(state.get("pod"), f"{job_id}.claim.pod")
            if claim_pod != pod_name:
                raise QueueError(
                    f"{job_id} claim says pod={claim_pod}, found on {pod_name}"
                )
            gpu = state.get("gpu")
            if type(gpu) is not int or gpu not in queue["pods"][pod_name]["gpus"]:
                raise QueueError(f"{job_id} claim has invalid gpu={gpu!r}")
            claim_state = _text(state.get("state"), f"{job_id}.claim.state")
            if claim_state not in {"claimed", "launched", "terminal"}:
                raise QueueError(f"{job_id} claim has invalid state={claim_state!r}")
            claim_schema = state.get("claim_schema_version")
            if type(claim_schema) is not int or claim_schema not in (1, 2):
                raise QueueError(f"{job_id} claim has invalid schema={claim_schema!r}")
            claim_digest = state.get("claim_content_sha256")
            if claim_schema == 2:
                if type(claim_digest) is not str or not SHA256.fullmatch(claim_digest):
                    raise QueueError(f"{job_id} schema-2 claim has invalid content digest")
            elif claim_digest is not None:
                raise QueueError(f"{job_id} legacy claim unexpectedly has a content digest")
            claims[job_id] = {
                "pod": pod_name,
                "gpu": gpu,
                "state": claim_state,
                "claim_schema_version": claim_schema,
                "claim_content_sha256": claim_digest,
                "claim_path": _text(
                    state.get("claim_path"), f"{job_id}.claim.claim_path"
                ),
            }
    expected = {slot.name for slot in slots(queue)}
    if set(occupancy) != expected:
        raise QueueError(f"GPU inventory mismatch: expected={sorted(expected)} got={sorted(occupancy)}")
    return occupancy, claims


def live_occupancy(queue: dict[str, Any]) -> dict[str, int]:
    return live_snapshot(queue)[0]


def live_slot_occupancy(queue: dict[str, Any], slot: Slot) -> int:
    """Read only the explicitly selected dispatch slot for a non-science probe."""

    remote = (
        f"nvidia-smi -i {slot.gpu} --query-compute-apps=pid "
        "--format=csv,noheader,nounits | awk "
        + shlex.quote(UNIQUE_NUMERIC_PID_AWK)
    )
    output = _run_ssh(
        queue,
        slot.pod,
        remote,
        phase=f"slot-occupancy:{slot.name}",
    ).strip()
    if not output.isdigit():
        raise QueueError(f"{slot.name} returned invalid compute occupancy {output!r}")
    return int(output)


def _effective_occupancy(
    queue: dict[str, Any],
    occupancy: dict[str, int],
    claims: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Reserve a slot while a non-terminal claim is not yet visible in NVML.

    A claim with no ``run.log.launch`` is the narrow claim-to-NVML window (or a
    fail-closed launch that still needs explicit disposition).  It counts as a
    reservation only while its queue row is non-terminal.  Marking an audited
    infrastructure-only attempt ``rejected`` releases that stale reservation;
    creating a new namespace is the only permitted retry.
    """

    effective = dict(occupancy)
    jobs = {job["id"]: job for job in queue["jobs"]}
    for job_id, claim in claims.items():
        job = jobs.get(job_id)
        if job is None:
            raise QueueError(f"claim references unknown job: {job_id}")
        if job["status"] in TERMINAL or claim["state"] != "claimed":
            continue
        slot_name = f"{claim['pod']}/gpu{claim['gpu']}"
        if slot_name not in effective:
            if claim["pod"] not in queue.get("dispatch_pods", list(queue["pods"])):
                continue
            raise QueueError(f"claim references unknown GPU slot: {slot_name}")
        effective[slot_name] += 1
    return effective


def _parse_gpu_occupancy(pod_name: str, snapshot: dict[str, Any]) -> dict[str, int]:
    compute_rows = _list(snapshot.get("compute_rows"), f"{pod_name}.compute_rows")
    gpu_rows = _list(snapshot.get("gpu_rows"), f"{pod_name}.gpu_rows")
    pids_by_uuid: dict[str, set[int]] = {}
    for index, row in enumerate(compute_rows):
        row = _text(row, f"{pod_name}.compute_rows[{index}]")
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise QueueError(f"{pod_name} returned malformed compute-app row: {row!r}")
        pids_by_uuid.setdefault(fields[0], set()).add(int(fields[1]))
    result: dict[str, int] = {}
    for index, row in enumerate(gpu_rows):
        row = _text(row, f"{pod_name}.gpu_rows[{index}]")
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
            raise QueueError(f"{pod_name} returned malformed GPU row: {row!r}")
        result[f"{pod_name}/gpu{fields[0]}"] = len(pids_by_uuid.get(fields[1], set()))
    return result


def _assign(
    queue: dict[str, Any], occupancy: dict[str, int], claimed: set[str] | None = None
) -> list[tuple[dict[str, Any], Slot]]:
    claimed = set() if claimed is None else claimed
    current = dict(occupancy)
    assignments: list[tuple[dict[str, Any], Slot]] = []
    all_slots = slots(queue)
    for job in queue["jobs"]:
        if job["status"] != READY or job["id"] in claimed:
            continue
        available = [slot for slot in all_slots if current[slot.name] < slot.capacity]
        if not available:
            break
        required_slot = job["resource"].get("required_slot")
        if required_slot is not None:
            required = [slot for slot in available if slot.name == required_slot]
            # A hard-bound job never falls through to another GPU.  It does not,
            # however, stall unrelated jobs that can use other dispatch slots;
            # matched-arm atomicity needs its own explicit launch-group contract.
            if not required:
                continue
            chosen = required[0]
        else:
            preferred_slot = job["resource"].get("preferred_slot")
            preferred = [slot for slot in available if slot.name == preferred_slot]
            chosen = preferred[0] if preferred else min(
                available, key=lambda slot: (current[slot.name], slot.ordinal)
            )
        assignments.append((job, chosen))
        current[chosen.name] += 1
    return assignments


def _require_bound_slot(
    job: dict[str, Any],
    slot: Slot,
    *,
    phase: str,
    include_preferred: bool,
) -> None:
    """Fail before SSH/claim construction when an explicit slot binding drifts.

    ``required_slot`` is a hard execution constraint and therefore applies to
    every path.  ``preferred_slot`` deliberately permits science fallback when
    its preferred GPU is full, but explicit probe/warmup/finalizer commands must
    still use the selected source's preferred GPU so their evidence stays bound.
    """

    required_slot = job["resource"].get("required_slot")
    if required_slot is not None and required_slot != slot.name:
        raise QueueError(
            f"{phase} must use required_slot {required_slot}, got {slot.name}"
        )
    preferred_slot = job["resource"].get("preferred_slot")
    if include_preferred and preferred_slot is not None and preferred_slot != slot.name:
        raise QueueError(
            f"{phase} must use preferred_slot {preferred_slot}, got {slot.name}"
        )


def _training_argv(
    queue: dict[str, Any],
    job: dict[str, Any],
    gpu: int,
    *,
    include_run_binding: bool = True,
) -> list[str]:
    source = job["source"]["checkout"]
    workdir = f"{source.rstrip('/')}/{WBT_RELATIVE}"
    run_dir = job["run_dir"].rstrip("/")
    argv = [
        ISAAC_PYTHON, f"{workdir}/{ENTRYPOINT_RELATIVE}",
        *job["recipe"]["base"], *job["recipe"]["delta"],
    ]
    for arg, path in job["motion"]["bindings"].items():
        argv.append(f"{arg}={path}")
    argv.extend(
        [
            f"{job['bank']['train_arg']}={job['bank']['train_path']}",
            f"seed={job['seed']}",
            f"num_envs={job['budget']['num_envs']}",
            f"max_iterations={job['budget']['max_iterations']}",
            f"algo.runner.save_interval={job['budget']['save_interval']}",
            f"run_name={job['run_name']}",
            # CUDA_VISIBLE_DEVICES maps the chosen physical GPU to logical cuda:0.
            "device=cuda:0",
        ]
    )
    if include_run_binding and job.get("runtime_binding", False):
        argv.extend(
            [
                f"++training_queue_claim_path={run_dir}/queue_claim.json",
                f"++training_run_binding_path={run_dir}/run_binding.json",
            ]
        )
    return argv


def _launch_contract(
    queue: dict[str, Any], job: dict[str, Any], slot: Slot
) -> tuple[dict[str, Any], list[str]]:
    """Build one canonical claim and its self-binding execution argv.

    The digest covers every caller-controlled argument and input identity.  The
    one derived claim argument is appended afterward and stored in the claim
    envelope, avoiding an impossible self-referential hash while keeping the
    full executed argv independently reconstructible.
    """

    _require_bound_slot(
        job, slot, phase="science launch contract", include_preferred=False
    )

    argv_without_claim = _training_argv(queue, job, slot.gpu)
    content = {
        "schema_version": 1,
        "job_id": job["id"],
        "action": job["action"],
        "pod": slot.pod,
        "gpu": slot.gpu,
        # The complete source mapping intentionally includes an optional
        # ignored_runtime_asset closure.  The schema-2 claim therefore binds
        # the exact hydrated tree contract without a parallel side channel.
        "source": dict(job["source"]),
        "run_name": job["run_name"],
        "run_dir": job["run_dir"],
        "runtime_binding": bool(job.get("runtime_binding", False)),
        "seed": job["seed"],
        "budget": {
            "num_envs": job["budget"]["num_envs"],
            "max_iterations": job["budget"]["max_iterations"],
            "save_interval": job["budget"]["save_interval"],
            "milestones": list(job["milestones"]),
        },
        "inputs": {
            "motion": {
                "action": job["motion"]["action"],
                "bindings": dict(job["motion"]["bindings"]),
            },
            "bank": {
                "action": job["bank"]["action"],
                "train_path": job["bank"]["train_path"],
                "train_arg": job["bank"]["train_arg"],
            },
            "exam": {
                "action": job["exam"]["action"],
                "path": job["exam"]["path"],
                "family": job["exam"]["family"],
            },
        },
        "training_argv_without_claim": argv_without_claim,
    }
    digest = _canonical_sha256(content)
    execution_argv = [
        *argv_without_claim,
        f"++training_launch_claim_sha256={digest}",
    ]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": execution_argv,
    }
    return claim, execution_argv


def _source_asset_runtime_paths(
    job: dict[str, Any], pod: str
) -> tuple[str, str, str]:
    asset = job["source"].get("ignored_runtime_asset")
    if not isinstance(asset, dict):
        raise QueueError(f"{job['id']} does not declare ignored_runtime_asset")
    contract_digest = _canonical_sha256(asset)
    identity = PurePosixPath(
        job["source"]["commit"], contract_digest, pod
    )
    receipt = SOURCE_ASSET_RECEIPT_ROOT / identity / "receipt.json"
    staging = SOURCE_ASSET_STAGING_ROOT / identity / "asset.stage"
    lock_identity = hashlib.sha256(
        (
            job["source"]["checkout"]
            + "\0"
            + job["source"]["commit"]
            + "\0"
            + contract_digest
        ).encode("utf-8")
    ).hexdigest()
    lock = f"/tmp/hope_lean_source_asset_{lock_identity}.lock"
    return str(receipt), str(staging), lock


def _source_asset_runtime_spec(
    job: dict[str, Any], pod: str, *, mode: str
) -> dict[str, Any]:
    if mode not in {"prepare", "doctor"}:
        raise QueueError("source asset runtime mode must be prepare or doctor")
    receipt, staging, lock = _source_asset_runtime_paths(job, pod)
    return {
        "mode": mode,
        "pod": pod,
        "source": {
            "checkout": job["source"]["checkout"],
            "commit": job["source"]["commit"],
        },
        "contract": job["source"]["ignored_runtime_asset"],
        "receipt_path": receipt,
        "staging_path": staging,
        "lock_path": lock,
        "entrypoint_relative_path": (
            f"{WBT_RELATIVE}/{ENTRYPOINT_RELATIVE}"
        ),
        "proc_root": "/proc",
        "urdf_relative_path": SOURCE_ASSET_URDF_RELATIVE_PATH,
        "expected_unique_mesh_references": SOURCE_ASSET_UNIQUE_MESH_REFERENCES,
    }


def _source_asset_remote_command(
    job: dict[str, Any], pod: str, *, mode: str
) -> str:
    spec = _source_asset_runtime_spec(job, pod, mode=mode)
    encoded = base64.b64encode(
        json.dumps(
            spec, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    ).decode("ascii")
    return shlex.join(["python3", "-c", SOURCE_ASSET_PROGRAM, encoded])


def _hydra_compose_argv(training_argv: list[str]) -> list[str]:
    if len(training_argv) < 2:
        raise QueueError("training argv is missing the Python executable or train.py")
    return [
        training_argv[0],
        training_argv[1],
        "--cfg",
        "job",
        "--resolve",
        *training_argv[2:],
    ]


def _child_env_command(argv: list[str], gpu: int) -> str:
    """Render the one child environment shared by doctor and trainer."""

    return (
        f"{shlex.join(['env', f'CUDA_VISIBLE_DEVICES={gpu}'])} "
        f"PYTHONPATH=\"${{HOPE_WBT_PYTHONPATH}}\" {shlex.join(argv)}"
    )


def _doctor_body(
    queue: dict[str, Any],
    job: dict[str, Any],
    slot: Slot,
    *,
    training_argv: list[str] | None = None,
) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    required = [
        *job["motion"]["bindings"].values(),
        job["bank"]["train_path"], job["exam"]["path"],
    ]
    if job.get("runtime_binding", False):
        required.append(f"{workdir}/{QUEUE_RUNTIME_RELATIVE}")
    checks = "\n".join(f"test -f {shlex.quote(path)}" for path in required)
    source_asset_check = ""
    if job["source"].get("ignored_runtime_asset") is not None:
        source_asset_check = (
            "# source-asset-doctor: exact target/donor/receipt before Hydra or Kit\n"
            + _source_asset_remote_command(job, slot.pod, mode="doctor")
            + "\n"
        )
    expected_module_root = f"{workdir}/source/whole_body_tracking/whole_body_tracking"
    module_probe = (
        "import importlib.util,pathlib;"
        "s=importlib.util.find_spec('whole_body_tracking');"
        "assert s is not None and s.origin is not None;"
        "print(pathlib.Path(s.origin).resolve().parent)"
    )
    child_probe = _child_env_command([ISAAC_PYTHON, "-c", module_probe], slot.gpu)
    if training_argv is None:
        _claim, training_argv = _launch_contract(queue, job, slot)
    compose_probe = _child_env_command(_hydra_compose_argv(training_argv), slot.gpu)
    return f"""set -euo pipefail
test \"$(git -C {shlex.quote(source)} rev-parse HEAD)\" = {shlex.quote(job['source']['commit'])}
test -z \"$(git -C {shlex.quote(source)} status --porcelain)\"
{source_asset_check}{checks}
cd {shlex.quote(workdir)}
source {shlex.quote(workdir + '/' + SETUP_RELATIVE)}
resolved_module_root=$({child_probe})
test \"$resolved_module_root\" = {shlex.quote(expected_module_root)}
{compose_probe} >/dev/null
"""


def _replace_generated_overrides(argv: list[str], replacements: dict[str, str]) -> list[str]:
    """Replace harness-generated scalar overrides exactly once each."""

    result = list(argv)
    seen: set[str] = set()
    for index in range(2, len(result)):
        argument = result[index]
        key = _override_key(argument, f"generated argv[{index}]")
        if key not in replacements:
            continue
        if key in seen:
            raise QueueError(f"generated argv sets {key!r} more than once")
        result[index] = f"{key}={replacements[key]}"
        seen.add(key)
    missing = set(replacements) - seen
    if missing:
        raise QueueError(f"generated argv is missing harness keys: {sorted(missing)}")
    return result


def _require_exact_probe_job(job: dict[str, Any]) -> None:
    """Raise unless a ready/blocked row has ready-grade exact identities."""

    job_id = job["id"]
    if job["source"]["commit"] == ZERO_COMMIT:
        raise QueueError(f"{job_id}.source.commit is an all-zero placeholder")
    if job.get("runtime_binding") is not True:
        raise QueueError(
            f"{job_id}.runtime_binding=true is required for a full-scene probe"
        )
    if job["source"].get("ignored_runtime_asset") is None:
        raise QueueError(
            f"{job_id}.source.ignored_runtime_asset is required for a full-scene probe"
        )
    required_runtime_intent = {
        "task.actor_obs_contract": "deploy_parity_face179",
        "task.plant.zero_joint_friction": "true",
        "task.physical_ball": "true",
    }
    observed_runtime_intent: dict[str, str] = {}
    for raw in [*job["recipe"]["base"], *job["recipe"]["delta"]]:
        key = _override_key(raw, f"{job_id}.probe recipe")
        if key not in required_runtime_intent:
            continue
        observed_runtime_intent[key] = raw.split("=", 1)[1]
    for key, expected in required_runtime_intent.items():
        if observed_runtime_intent.get(key) != expected:
            raise QueueError(
                f"{job_id} full-scene probe requires {key}={expected}"
            )
    paths = {
        "source.checkout": job["source"]["checkout"],
        **{
            f"motion.{argument}": asset_path
            for argument, asset_path in job["motion"]["bindings"].items()
        },
        "bank.train_path": job["bank"]["train_path"],
        "exam.path": job["exam"]["path"],
        "run_dir": job["run_dir"],
    }
    normalized = [
        _ready_workspace_path(value, f"{job_id}.{label}")
        for label, value in paths.items()
    ]
    if len(set(normalized[1:-1])) != len(normalized[1:-1]):
        raise QueueError(f"{job_id} has duplicate motion/bank/exam identities")


def _boot_warmup_contract(
    queue: dict[str, Any], job: dict[str, Any], slot: Slot, attempt_id: str
) -> tuple[dict[str, Any], list[str], str]:
    """Build a tiny non-scientific scene-import warmup in its own namespace."""

    _require_bound_slot(
        job, slot, phase="boot warmup", include_preferred=True
    )
    attempt_id = _text(attempt_id, "attempt_id", safe_id=True)
    warmup_name = (
        f"boot_warmup_{job['source']['commit'][:8]}_{slot.pod}_gpu{slot.gpu}_{attempt_id}"
    )
    base_argv = _training_argv(
        queue, job, slot.gpu, include_run_binding=False
    )
    argv_without_claim = _replace_generated_overrides(
        base_argv,
        {
            "num_envs": str(WARMUP_NUM_ENVS),
            "max_iterations": str(WARMUP_MAX_ITERATIONS),
            "algo.runner.save_interval": str(WARMUP_SAVE_INTERVAL),
            "run_name": warmup_name,
        },
    )
    run_dir = str(
        PurePosixPath(job["run_dir"]).parent
        / "_boot_warmups"
        / job["source"]["commit"]
        / slot.pod
        / f"gpu{slot.gpu}"
        / attempt_id
    )
    content = {
        "schema_version": 1,
        "purpose": "boot_warmup_not_science",
        "job_id": job["id"],
        "pod": slot.pod,
        "gpu": slot.gpu,
        "attempt_id": attempt_id,
        "source": dict(job["source"]),
        "run_dir": run_dir,
        "budget": {
            "num_envs": WARMUP_NUM_ENVS,
            "max_iterations": WARMUP_MAX_ITERATIONS,
            "save_interval": WARMUP_SAVE_INTERVAL,
        },
        "inputs": {
            "motion": dict(job["motion"]),
            "bank": dict(job["bank"]),
            "exam": dict(job["exam"]),
        },
        "training_argv_without_claim": argv_without_claim,
    }
    digest = _canonical_sha256(content)
    execution_argv = [
        *argv_without_claim,
        f"++training_launch_claim_sha256={digest}",
    ]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": execution_argv,
    }
    return claim, execution_argv, run_dir


def _full_scene_probe_contract(
    queue: dict[str, Any], job: dict[str, Any], slot: Slot, attempt_id: str
) -> tuple[dict[str, Any], list[str], str]:
    """Derive a two-update non-science run without changing scene scale."""

    _require_bound_slot(
        job, slot, phase="full-scene probe", include_preferred=True
    )
    _require_exact_probe_job(job)
    attempt_id = _text(attempt_id, "attempt_id", safe_id=True)
    num_envs = job["budget"]["num_envs"]
    probe_name = (
        f"full_scene_probe_not_science_{job['id']}_"
        f"{job['source']['commit'][:8]}_"
        f"{slot.pod}_gpu{slot.gpu}_{attempt_id}"
    )
    run_dir = str(
        PurePosixPath(job["run_dir"]).parent
        / "_full_scene_probes"
        / job["id"]
        / job["source"]["commit"]
        / slot.pod
        / f"gpu{slot.gpu}"
        / attempt_id
    )
    base_argv = _training_argv(
        queue, job, slot.gpu, include_run_binding=False
    )
    argv_without_claim = _replace_generated_overrides(
        base_argv,
        {
            "max_iterations": str(FULL_SCENE_PROBE_MAX_ITERATIONS),
            "algo.runner.save_interval": str(FULL_SCENE_PROBE_SAVE_INTERVAL),
            "run_name": probe_name,
        },
    )
    argv_without_claim.extend(
        [
            f"++training_queue_claim_path={run_dir}/full_scene_probe_claim.json",
            f"++training_run_binding_path={run_dir}/full_scene_probe_binding.json",
        ]
    )
    expected_num_envs = f"num_envs={num_envs}"
    if argv_without_claim.count(expected_num_envs) != 1:
        raise QueueError(
            "full-scene probe must preserve the source job num_envs exactly"
        )
    source_path = PurePosixPath(job["source"]["checkout"])
    probe_path = PurePosixPath(run_dir)
    if probe_path == source_path or source_path in probe_path.parents:
        raise QueueError("full-scene probe namespace must stay outside source checkout")
    workdir = f"{job['source']['checkout'].rstrip('/')}/{WBT_RELATIVE}"
    supervisor_argv_prefix = [
        ISAAC_PYTHON,
        f"{workdir}/{FULL_SCENE_PROBE_RUNTIME_RELATIVE}",
        "supervise",
        "--run-dir",
        run_dir,
        "--log",
        f"{run_dir}/run.log",
        "--",
    ]
    source_asset_receipt, _staging, _lock = _source_asset_runtime_paths(
        job, slot.pod
    )
    content = {
        "schema_version": 1,
        "purpose": "full_scene_probe_not_science",
        "not_science": True,
        "attestable": False,
        "promotable": False,
        "job_id": job["id"],
        "pod": slot.pod,
        "gpu": slot.gpu,
        "attempt_id": attempt_id,
        "source": dict(job["source"]),
        "source_asset_receipt_path": source_asset_receipt,
        "supervisor_argv_prefix": supervisor_argv_prefix,
        "expected_training_contract_lineage_exact": 1,
        "run_name": probe_name,
        "run_dir": run_dir,
        "source_job_budget": dict(job["budget"]),
        "budget": {
            "num_envs": num_envs,
            "max_iterations": FULL_SCENE_PROBE_MAX_ITERATIONS,
            "save_interval": FULL_SCENE_PROBE_SAVE_INTERVAL,
            "milestones": [1],
        },
        "inputs": {
            "motion": dict(job["motion"]),
            "bank": dict(job["bank"]),
            "exam": dict(job["exam"]),
        },
        "training_argv_without_claim": argv_without_claim,
    }
    digest = _canonical_sha256(content)
    execution_argv = [
        *argv_without_claim,
        f"++training_launch_claim_sha256={digest}",
    ]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": execution_argv,
    }
    return claim, execution_argv, run_dir


def _boot_warmup_script(
    queue: dict[str, Any], job: dict[str, Any], slot: Slot, attempt_id: str
) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    claim_document, train_argv, run_dir = _boot_warmup_contract(
        queue, job, slot, attempt_id
    )
    run_parent = str(PurePosixPath(run_dir).parent)
    claim = json.dumps(
        claim_document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    launcher = f"{workdir}/{KIT_LAUNCHER_RELATIVE}"
    launch = shlex.join([launcher, f"{run_dir}/run.log"]) + " " + (
        _child_env_command(train_argv, slot.gpu)
    ) + f" {GPU_LAUNCH_LOCK_FD}>&-"
    doctor = _doctor_body(queue, job, slot, training_argv=train_argv)
    body = doctor + f"""
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(UNIQUE_NUMERIC_PID_AWK)})
test "$count" -lt {slot.capacity}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
( set -o noclobber; printf %s {shlex.quote(claim)} > {shlex.quote(run_dir + '/warmup_claim.json')} )
export KIT_BOOT_MARKER={shlex.quote(KIT_BOOT_MARKER)}
export KIT_BOOT_TIMEOUT_S={WARMUP_BOOT_TIMEOUT_SECONDS}
{launch}
"""
    return _gpu_launch_lock_script(slot, body)


def _full_scene_probe_script(
    queue: dict[str, Any], job: dict[str, Any], slot: Slot, attempt_id: str
) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    claim_document, train_argv, run_dir = _full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    run_parent = str(PurePosixPath(run_dir).parent)
    claim = json.dumps(
        claim_document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    launcher = f"{workdir}/{KIT_LAUNCHER_RELATIVE}"
    probe_runtime = f"{workdir}/{FULL_SCENE_PROBE_RUNTIME_RELATIVE}"
    queue_runtime = f"{workdir}/{QUEUE_RUNTIME_RELATIVE}"
    train_entry = f"{workdir}/{ENTRYPOINT_RELATIVE}"
    supervisor_argv = [
        *claim_document["content"]["supervisor_argv_prefix"],
        *train_argv,
    ]
    launch = shlex.join([launcher, f"{run_dir}/run.log"]) + " " + (
        _child_env_command(supervisor_argv, slot.gpu)
    ) + f" {GPU_LAUNCH_LOCK_FD}>&-"
    doctor = _doctor_body(queue, job, slot, training_argv=train_argv)
    body = doctor + f"""
test -f {shlex.quote(probe_runtime)}
test -f {shlex.quote(queue_runtime)}
test -f {shlex.quote(train_entry)}
grep -Fq -- 'def _publish_lean_queue_binding_if_requested' {shlex.quote(train_entry)}
grep -Fq -- '_publish_lean_queue_binding_if_requested(cfg, log_dir)' {shlex.quote(train_entry)}
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(UNIQUE_NUMERIC_PID_AWK)})
test "$count" -lt {slot.capacity}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
( set -o noclobber; printf %s {shlex.quote(claim)} > {shlex.quote(run_dir + '/full_scene_probe_claim.json')} )
export KIT_BOOT_MARKER={shlex.quote(KIT_BOOT_MARKER)}
export KIT_BOOT_TIMEOUT_S={KIT_BOOT_TIMEOUT_SECONDS}
export KIT_BOOT_STALE_TIMEOUT_S={FULL_SCENE_PROBE_STALE_TIMEOUT_SECONDS}
{launch}
printf '%s\n' 'phase=first_iter not_science=true' >> {shlex.quote(run_dir + '/run.log.launch')}
"""
    return _gpu_launch_lock_script(slot, body)


def _finalize_full_scene_probe_script(
    job: dict[str, Any], pod: str, run_dir: str, expected_claim_sha256: str
) -> str:
    """Render one selected-Pod, read-only terminal finalizer."""

    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    runtime = f"{workdir}/{FULL_SCENE_PROBE_RUNTIME_RELATIVE}"
    source_asset_check = _source_asset_remote_command(job, pod, mode="doctor")
    command = shlex.join(
        [
            ISAAC_PYTHON,
            runtime,
            "finalize",
            "--run-dir",
            run_dir,
            "--expected-claim-sha256",
            expected_claim_sha256,
        ]
    )
    return f"""set -euo pipefail
test "$(git -C {shlex.quote(source)} rev-parse HEAD)" = {shlex.quote(job['source']['commit'])}
test -z "$(git -C {shlex.quote(source)} status --porcelain --untracked-files=all)"
test -f {shlex.quote(runtime)}
test -d {shlex.quote(run_dir)}
{source_asset_check} >&2
cd {shlex.quote(workdir)}
source {shlex.quote(workdir + '/' + SETUP_RELATIVE)} >&2
PYTHONPATH="${{HOPE_WBT_PYTHONPATH}}" {command}
"""


def _job_by_id(queue: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [job for job in queue["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise QueueError(f"unknown or duplicate job id: {job_id}")
    return matches[0]


def _slot_by_identity(queue: dict[str, Any], pod: str, gpu: int) -> Slot:
    matches = [slot for slot in slots(queue) if slot.pod == pod and slot.gpu == gpu]
    if len(matches) != 1:
        raise QueueError(f"slot is not dispatch-enabled: {pod}/gpu{gpu}")
    return matches[0]


def _gpu_launch_lock_script(slot: Slot, body: str) -> str:
    """Hold the short launch lock in fd8, then close it in the long-lived child.

    Using ``flock FILE command`` leaves flock's private descriptor inherited by
    grandchildren.  A detached trainer then retains the lock for its full run,
    silently reducing a multi-trainer GPU to capacity one.  The controller
    shell owns an explicit fd8; the launcher command receives ``8>&-`` so its
    trainer cannot inherit that descriptor.
    """

    lock_path = f"/tmp/hope_lean_queue_gpu{slot.gpu}.lock"
    locked_body = (
        f"exec {GPU_LAUNCH_LOCK_FD}>{shlex.quote(lock_path)}\n"
        f"flock -n {GPU_LAUNCH_LOCK_FD}\n"
        f"{body}"
    )
    return f"bash -lc {shlex.quote(locked_body)}"


def _doctor_script(queue: dict[str, Any], job: dict[str, Any], slot: Slot) -> str:
    return _doctor_body(queue, job, slot) + (
        "printf '%s\\n' "
        + shlex.quote(
            "DOCTOR_OK scope=source-clean,assets,module-exact "
            "hydra=exact-no-kit-compose"
        )
    )


def _launch_script(queue: dict[str, Any], job: dict[str, Any], slot: Slot) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    run_dir = job["run_dir"].rstrip("/")
    run_parent = str(PurePosixPath(run_dir).parent)
    claim_document, train_argv = _launch_contract(queue, job, slot)
    claim = json.dumps(
        claim_document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    launcher = f"{workdir}/{KIT_LAUNCHER_RELATIVE}"
    launch = shlex.join([launcher, f"{run_dir}/run.log"]) + " " + (
        _child_env_command(train_argv, slot.gpu)
    ) + f" {GPU_LAUNCH_LOCK_FD}>&-"
    # The per-GPU flock covers the last capacity check, claim, and spawn.
    body = _doctor_body(queue, job, slot) + f"""
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(UNIQUE_NUMERIC_PID_AWK)})
test \"$count\" -lt {slot.capacity}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
mkdir {shlex.quote(run_dir + '/milestones')}
( set -o noclobber; printf %s {shlex.quote(claim)} > {shlex.quote(run_dir + '/queue_claim.json')} )
export KIT_BOOT_MARKER={shlex.quote(KIT_BOOT_MARKER)}
export KIT_BOOT_TIMEOUT_S={KIT_BOOT_TIMEOUT_SECONDS}
{launch}
printf '%s\\n' phase=first_iter >> {shlex.quote(run_dir + '/run.log.launch')}
"""
    return _gpu_launch_lock_script(slot, body)


def _queue_runtime_payload() -> tuple[bytes, str]:
    path = (
        Path(__file__).resolve().parents[1]
        / WBT_RELATIVE
        / QUEUE_RUNTIME_RELATIVE
    )
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise QueueError("lean queue runtime must be a regular non-symlink file")
    raw = path.read_bytes()
    after = path.lstat()
    signature = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns
    )
    if signature(before) != signature(after):
        raise QueueError("lean queue runtime changed while hashing")
    return raw, hashlib.sha256(raw).hexdigest()


def _runtime_snapshot_materializer(
    runtime_raw: bytes,
    runtime_sha256: str,
    *,
    root: str = ATTESTOR_RUNTIME_ROOT,
) -> tuple[str, str]:
    if hashlib.sha256(runtime_raw).hexdigest() != runtime_sha256:
        raise QueueError("reviewed runtime payload differs from its SHA")
    snapshot = f"{root.rstrip('/')}/{runtime_sha256}/lean_queue_runtime.py"
    encoded = base64.b64encode(runtime_raw).decode("ascii")
    command = shlex.join(
        [ISAAC_PYTHON, "-c", _RUNTIME_SNAPSHOT_PROGRAM, encoded, runtime_sha256, root]
    )
    return snapshot, command


def _milestone_attestor_script(
    job: dict[str, Any],
    milestone: int,
    *,
    expected_claim_content_sha256: str,
    expected_job_id: str,
    expected_runtime_sha256: str,
    runtime_raw: bytes,
) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{WBT_RELATIVE}"
    runtime, materialize_runtime = _runtime_snapshot_materializer(
        runtime_raw, expected_runtime_sha256
    )
    binding = f"{job['run_dir'].rstrip('/')}/run_binding.json"
    command = shlex.join(
        [
            ISAAC_PYTHON,
            runtime,
            "attest",
            "--binding",
            binding,
            "--milestone",
            str(milestone),
            "--expected-claim-content-sha256",
            expected_claim_content_sha256,
            "--expected-job-id",
            expected_job_id,
            "--expected-runtime-sha256",
            expected_runtime_sha256,
        ]
    )
    return f"""set -euo pipefail
test "$(git -C {shlex.quote(source)} rev-parse HEAD)" = {shlex.quote(job['source']['commit'])}
test -z "$(git -C {shlex.quote(source)} status --porcelain)"
{materialize_runtime}
cd {shlex.quote(workdir)}
source {shlex.quote(workdir + '/' + SETUP_RELATIVE)}
PYTHONPATH="${{HOPE_WBT_PYTHONPATH}}" {command}
"""


def _require_attestor_claim_matches_current_job(
    queue: dict[str, Any], job: dict[str, Any], claim: dict[str, Any]
) -> None:
    """Prevent a mutable queue row from selecting a verifier for another claim."""

    if claim.get("claim_schema_version") != 2:
        raise QueueError(f"{job['id']} milestone attestation requires a schema-2 claim")
    pod = claim["pod"]
    gpu = claim["gpu"]
    pod_cfg = queue["pods"][pod]
    slot = Slot(pod, gpu, ordinal=0, capacity=pod_cfg["max_trainers_per_gpu"])
    expected_claim, _argv = _launch_contract(queue, job, slot)
    immutable_digest = claim.get("claim_content_sha256")
    if immutable_digest != expected_claim["content_sha256"]:
        raise QueueError(
            f"{job['id']} current queue row differs from its immutable launch claim; "
            "refusing verifier source drift"
        )
    expected_path = f"{job['run_dir'].rstrip('/')}/queue_claim.json"
    if claim.get("claim_path") != expected_path:
        raise QueueError(f"{job['id']} immutable claim path differs from current run_dir")


def cmd_attest_milestone(
    queue: dict[str, Any],
    *,
    job_id: str,
    milestone: int,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    jobs = {job["id"]: job for job in queue["jobs"]}
    if job_id not in jobs:
        raise QueueError(f"unknown queue job: {job_id}")
    job = jobs[job_id]
    if not job.get("runtime_binding", False):
        raise QueueError(
            f"{job_id} did not preregister runtime_binding=true; no binding may be inferred"
        )
    if type(milestone) is not int or milestone not in job["milestones"]:
        raise QueueError(
            f"{job_id} milestone must be one of {job['milestones']}"
        )
    if execute and confirm != ATTEST_CONFIRM:
        raise QueueError(f"--execute requires --confirm {ATTEST_CONFIRM}")
    base = {
        "mode": "attest-milestone",
        "dry_run": not execute,
        "job_id": job_id,
        "milestone": milestone,
        "binding_path": f"{job['run_dir'].rstrip('/')}/run_binding.json",
        "receipt_path": (
            f"{job['run_dir'].rstrip('/')}/milestones/model_{milestone}.json"
        ),
    }
    runtime_raw, runtime_sha256 = _queue_runtime_payload()
    if not execute:
        remote = _milestone_attestor_script(
            job,
            milestone,
            expected_claim_content_sha256="<immutable-claim-content-sha256>",
            expected_job_id=job_id,
            expected_runtime_sha256=runtime_sha256,
            runtime_raw=runtime_raw,
        )
        return {
            **base,
            "pod_resolution": "immutable queue claim at execute time",
            "lean_queue_runtime_sha256": runtime_sha256,
            "remote_script": remote,
        }
    _occupancy, claims = live_snapshot(queue)
    claim = claims.get(job_id)
    if claim is None:
        raise QueueError(f"{job_id} has no immutable queue claim on either Pod")
    _require_attestor_claim_matches_current_job(queue, job, claim)
    pod = claim["pod"]
    expected_claim_content_sha256 = claim["claim_content_sha256"]
    remote = _milestone_attestor_script(
        job,
        milestone,
        expected_claim_content_sha256=expected_claim_content_sha256,
        expected_job_id=job_id,
        expected_runtime_sha256=runtime_sha256,
        runtime_raw=runtime_raw,
    )
    output = _run_ssh(
        queue,
        pod,
        remote,
        timeout=120,
        phase=f"attest-milestone:{job_id}:{milestone}",
    )
    return {
        **base,
        "pod": pod,
        "expected_launch_claim_content_sha256": expected_claim_content_sha256,
        "lean_queue_runtime_sha256": runtime_sha256,
        "remote_output": output,
    }


def cmd_plan(queue: dict[str, Any], *, live: bool) -> dict[str, Any]:
    if live:
        occupancy, claims = live_snapshot(queue)
    else:
        occupancy, claims = {slot.name: 0 for slot in slots(queue)}, {}
    effective_occupancy = _effective_occupancy(queue, occupancy, claims)
    assignments = _assign(queue, effective_occupancy, set(claims))
    return {
        "mode": "plan",
        "dry_run": True,
        "occupancy_source": "live" if live else "assumed_empty",
        "occupancy": occupancy,
        "effective_occupancy": effective_occupancy,
        "claims": claims,
        "assignments": [
            {
                "job_id": job["id"], "action": job["action"],
                "resource": slot.name,
                "milestones": job["milestones"],
            }
            for job, slot in assignments
        ],
        "blocked": [
            {"job_id": job["id"], "reason": job["blocker"]}
            for job in queue["jobs"] if job["status"] == BLOCKED
        ],
    }


def cmd_doctor(queue: dict[str, Any], *, live: bool) -> dict[str, Any]:
    if live:
        occupancy, claims = live_snapshot(queue)
    else:
        occupancy, claims = {slot.name: 0 for slot in slots(queue)}, {}
    effective_occupancy = _effective_occupancy(queue, occupancy, claims)
    assignments = _assign(queue, effective_occupancy, set(claims))
    results: list[dict[str, Any]] = []
    for job, slot in assignments:
        remote = _doctor_script(queue, job, slot)
        record: dict[str, Any] = {
            "job_id": job["id"],
            "resource": slot.name,
            "scope": "source-clean,assets,module-exact,hydra-resolved",
            "hydra": "exact-no-kit-compose",
        }
        if live:
            record["remote_output"] = _run_ssh(
                queue, slot.pod, remote, phase=f"doctor:{job['id']}"
            )
        else:
            record["ssh_argv"] = [
                *_ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
            ]
        results.append(record)
    return {
        "mode": "doctor",
        "dry_run": not live,
        "occupancy": occupancy,
        "effective_occupancy": effective_occupancy,
        "claims": claims,
        "results": results,
    }


def cmd_status(queue: dict[str, Any], *, live: bool) -> dict[str, Any]:
    plan = cmd_plan(queue, live=live)
    plan["mode"] = "status"
    plan["jobs"] = [
        {"job_id": job["id"], "action": job["action"], "status": job["status"]}
        for job in queue["jobs"]
    ]
    return plan


def cmd_launch_next(
    queue: dict[str, Any], *, execute: bool, confirm: str | None
) -> dict[str, Any]:
    if queue["launch_authorized"] is not True:
        raise QueueError("launch_authorized is false; launch-next is blocked")
    if execute and confirm != CONFIRM:
        raise QueueError(f"--execute requires --confirm {CONFIRM}")
    if not execute:
        occupancy, claims = {slot.name: 0 for slot in slots(queue)}, {}
        effective_occupancy = _effective_occupancy(queue, occupancy, claims)
        assignments = _assign(queue, effective_occupancy, set(claims))
        if not assignments:
            raise QueueError("no ready job fits an available GPU slot")
        job, slot = assignments[0]
        remote = _launch_script(queue, job, slot)
        return {
            "mode": "launch-next", "dry_run": True,
            "job_id": job["id"], "action": job["action"], "resource": slot.name,
            "ssh_argv": [
                *_ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
            ],
        }

    GLOBAL_SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_SCHEDULER_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Re-sample all six GPUs only after the global scheduler lock is held.
        occupancy, claims = live_snapshot(queue)
        effective_occupancy = _effective_occupancy(queue, occupancy, claims)
        assignments = _assign(queue, effective_occupancy, set(claims))
        if not assignments:
            raise QueueError("no ready job fits an available GPU slot")
        job, slot = assignments[0]
        remote = _launch_script(queue, job, slot)
        output = _run_ssh(
            queue,
            slot.pod,
            remote,
            timeout=KIT_BOOT_TIMEOUT_SECONDS + 60,
            phase=f"launch:{job['id']}",
        )
        return {
            "mode": "launch-next", "dry_run": False,
            "job_id": job["id"], "action": job["action"], "resource": slot.name,
            "scheduler_lock": str(GLOBAL_SCHEDULER_LOCK),
            "remote_output": output,
        }


def cmd_fill(
    queue: dict[str, Any], *, execute: bool, confirm: str | None, count: int
) -> dict[str, Any]:
    if queue["launch_authorized"] is not True:
        raise QueueError("launch_authorized is false; fill is blocked")
    if count <= 0:
        raise QueueError("fill --count must be a positive integer")
    if execute and confirm != CONFIRM:
        raise QueueError(f"--execute requires --confirm {CONFIRM}")
    if not execute:
        occupancy = {slot.name: 0 for slot in slots(queue)}
        assignments = _assign(queue, occupancy)[:count]
        return {
            "mode": "fill",
            "dry_run": True,
            "count_limit": count,
            "jobs": [
                {
                    "job_id": job["id"],
                    "resource": slot.name,
                    "doctor_ssh_argv": [
                        *_ssh_prefix(queue, slot.pod),
                        f"bash -lc {shlex.quote(_doctor_script(queue, job, slot))}",
                    ],
                    "launch_ssh_argv": [
                        *_ssh_prefix(queue, slot.pod),
                        f"bash -lc {shlex.quote(_launch_script(queue, job, slot))}",
                    ],
                }
                for job, slot in assignments
            ],
        }

    launched: list[dict[str, Any]] = []
    GLOBAL_SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_SCHEDULER_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for _ in range(count):
            occupancy, claims = live_snapshot(queue)
            effective_occupancy = _effective_occupancy(queue, occupancy, claims)
            assignments = _assign(queue, effective_occupancy, set(claims))
            if not assignments:
                break
            job, slot = assignments[0]
            # _launch_script begins with the same fail-closed doctor body and
            # keeps it under the remote per-GPU lock before capacity, mkdir,
            # claim, or Kit spawn.  A separate SSH doctor here was therefore
            # only a duplicate, non-authoritative preflight: it added another
            # network/config-compose failure surface without closing a TOCTOU
            # window.  Keep standalone ``doctor`` for explicit diagnostics,
            # but make each fill attempt one atomic remote transaction.
            launch_output = _run_ssh(
                queue,
                slot.pod,
                _launch_script(queue, job, slot),
                timeout=KIT_BOOT_TIMEOUT_SECONDS + 60,
                phase=f"launch-first-iteration:{job['id']}",
            )
            launched.append(
                {
                    "job_id": job["id"],
                    "resource": slot.name,
                    "preflight_mode": "embedded_in_atomic_launch",
                    "launch_output": launch_output,
                }
            )
    if not launched:
        raise QueueError("no ready job fits an available GPU slot")
    return {
        "mode": "fill",
        "result_schema_version": 2,
        "dry_run": False,
        "count_limit": count,
        "scheduler_lock": str(GLOBAL_SCHEDULER_LOCK),
        "launched": launched,
    }


def cmd_boot_warmup(
    queue: dict[str, Any], *, job_id: str, pod: str, gpu: int,
    attempt_id: str, execute: bool, confirm: str | None,
) -> dict[str, Any]:
    """Run one tiny scene-import warmup without consuming a science namespace."""

    if execute and confirm != WARMUP_CONFIRM:
        raise QueueError(f"--execute requires --confirm {WARMUP_CONFIRM}")
    job = _job_by_id(queue, _text(job_id, "job_id", safe_id=True))
    if job["status"] not in {READY, BLOCKED}:
        raise QueueError("boot warmup requires a ready or blocked source job")
    slot = _slot_by_identity(queue, _text(pod, "pod", safe_id=True), gpu)
    _require_bound_slot(
        job, slot, phase="boot warmup", include_preferred=True
    )
    claim, _argv, run_dir = _boot_warmup_contract(queue, job, slot, attempt_id)
    remote = _boot_warmup_script(queue, job, slot, attempt_id)
    result: dict[str, Any] = {
        "mode": "boot-warmup",
        "dry_run": not execute,
        "not_science": True,
        "job_id": job["id"],
        "resource": slot.name,
        "run_dir": run_dir,
        "claim_sha256": claim["content_sha256"],
        "budget": claim["content"]["budget"],
    }
    if not execute:
        result["ssh_argv"] = [
            *_ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
        ]
        return result

    occupancy, _claims = live_snapshot(queue)
    if occupancy[slot.name] >= slot.capacity:
        raise QueueError(f"warmup slot is at capacity: {slot.name}")
    result["remote_output"] = _run_ssh(
        queue,
        slot.pod,
        remote,
        timeout=WARMUP_BOOT_TIMEOUT_SECONDS + 60,
        phase=f"boot-warmup:{job['id']}:{attempt_id}",
    )
    return result


def cmd_full_scene_probe(
    queue: dict[str, Any], *, job_id: str, pod: str, gpu: int,
    attempt_id: str, execute: bool, confirm: str | None,
) -> dict[str, Any]:
    """Run one representative-scale scene boot in a non-science namespace."""

    if execute and confirm != FULL_SCENE_PROBE_CONFIRM:
        raise QueueError(f"--execute requires --confirm {FULL_SCENE_PROBE_CONFIRM}")
    job = _job_by_id(queue, _text(job_id, "job_id", safe_id=True))
    if job["status"] not in {READY, BLOCKED}:
        raise QueueError("full-scene probe requires a ready or blocked exact source job")
    slot = _slot_by_identity(queue, _text(pod, "pod", safe_id=True), gpu)
    _require_bound_slot(
        job, slot, phase="full-scene probe", include_preferred=True
    )
    claim, _argv, run_dir = _full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    remote = _full_scene_probe_script(queue, job, slot, attempt_id)
    result: dict[str, Any] = {
        "mode": "full-scene-probe",
        "dry_run": not execute,
        "purpose": "full_scene_probe_not_science",
        "not_science": True,
        "attestable": False,
        "promotable": False,
        "job_id": job["id"],
        "resource": slot.name,
        "run_dir": run_dir,
        "claim_path": f"{run_dir}/full_scene_probe_claim.json",
        "claim_sha256": claim["content_sha256"],
        "budget": claim["content"]["budget"],
        "first_iteration_observed": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *_ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
        ]
        return result

    if live_slot_occupancy(queue, slot) >= slot.capacity:
        raise QueueError(f"full-scene probe slot is at capacity: {slot.name}")
    result["remote_output"] = _run_ssh(
        queue,
        slot.pod,
        remote,
        timeout=KIT_BOOT_TIMEOUT_SECONDS + 60,
        phase=f"full-scene-probe:{job['id']}:{attempt_id}",
    )
    result["first_iteration_observed"] = True
    return result


def cmd_prepare_source_assets(
    queue: dict[str, Any], *, job_id: str, pod: str,
    execute: bool, confirm: str | None,
) -> dict[str, Any]:
    """Hydrate one declared ignored source asset on one selected Pod only."""

    if execute and confirm != PREPARE_SOURCE_ASSET_CONFIRM:
        raise QueueError(
            f"--execute requires --confirm {PREPARE_SOURCE_ASSET_CONFIRM}"
        )
    job = _job_by_id(queue, _text(job_id, "job_id", safe_id=True))
    if job["status"] not in {READY, BLOCKED}:
        raise QueueError("source asset preparation requires a ready or blocked job")
    pod = _text(pod, "pod", safe_id=True)
    if pod not in queue.get("dispatch_pods", list(queue["pods"])):
        raise QueueError(f"source asset preparation Pod is not dispatch-enabled: {pod}")
    if job["source"].get("ignored_runtime_asset") is None:
        raise QueueError(f"{job['id']} does not declare ignored_runtime_asset")
    receipt, staging, lock = _source_asset_runtime_paths(job, pod)
    remote = _source_asset_remote_command(job, pod, mode="prepare")
    result: dict[str, Any] = {
        "mode": "prepare-source-assets",
        "dry_run": not execute,
        "simulation_only": True,
        "job_id": job["id"],
        "pod": pod,
        "source": dict(job["source"]),
        "receipt_path": receipt,
        "staging_path": staging,
        "source_lock": lock,
        "automatic_retry": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *_ssh_prefix(queue, pod), f"bash -lc {shlex.quote(remote)}"
        ]
        return result
    # Exactly one selected-Pod SSH call.  A timeout is UNKNOWN and is returned
    # to the operator; this command never replays itself.
    result["remote_output"] = _run_ssh(
        queue, pod, remote, timeout=600,
        phase=f"prepare-source-assets:{job['id']}",
    )
    return result


def cmd_finalize_full_scene_probe(
    queue: dict[str, Any], *, job_id: str, pod: str, gpu: int,
    attempt_id: str, execute: bool, confirm: str | None,
) -> dict[str, Any]:
    """Finalize one exact probe on only its explicitly selected dispatch Pod."""

    if execute and confirm != FINALIZE_FULL_SCENE_PROBE_CONFIRM:
        raise QueueError(
            f"--execute requires --confirm {FINALIZE_FULL_SCENE_PROBE_CONFIRM}"
        )
    job = _job_by_id(queue, _text(job_id, "job_id", safe_id=True))
    slot = _slot_by_identity(queue, _text(pod, "pod", safe_id=True), gpu)
    _require_bound_slot(
        job, slot, phase="probe finalization", include_preferred=True
    )
    claim, _argv, run_dir = _full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    remote = _finalize_full_scene_probe_script(
        job, slot.pod, run_dir, claim["content_sha256"]
    )
    result: dict[str, Any] = {
        "mode": "finalize-full-scene-probe",
        "dry_run": not execute,
        "job_id": job["id"],
        "resource": slot.name,
        "run_dir": run_dir,
        "claim_path": f"{run_dir}/full_scene_probe_claim.json",
        "claim_sha256": claim["content_sha256"],
        "binding_path": f"{run_dir}/full_scene_probe_binding.json",
        "exit_receipt_path": f"{run_dir}/full_scene_probe_exit.json",
        "result_path": f"{run_dir}/probe_result.json",
        "automatic_retry_authorized": False,
        "queue_status_mutated": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *_ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
        ]
        return result
    remote_output = _run_ssh(
        queue,
        slot.pod,
        remote,
        timeout=180,
        phase=f"finalize-full-scene-probe:{job['id']}:{attempt_id}",
    )
    try:
        terminal = json.loads(remote_output)
        terminal_content = _mapping(
            _mapping(terminal.get("result"), "probe finalizer result").get("content"),
            "probe finalizer result content",
        )
    except json.JSONDecodeError as exc:
        raise QueueError("probe finalizer returned malformed JSON") from exc
    status = terminal_content.get("status")
    unlock = terminal_content.get("unlock_authorized")
    if status not in {"passed", "failed"} or type(unlock) is not bool:
        raise QueueError("probe finalizer returned an invalid terminal classification")
    if unlock is not (status == "passed"):
        raise QueueError("probe finalizer status/unlock classification is inconsistent")
    result["terminal_status"] = status
    result["unlock_authorized"] = unlock
    result["terminal_result"] = terminal
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "status", "doctor"):
        command = sub.add_parser(mode)
        command.add_argument("--live", action="store_true", help="read GPU occupancy over SSH")
    launch = sub.add_parser("launch-next")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--confirm")
    fill = sub.add_parser("fill")
    fill.add_argument("--count", type=int, default=1)
    fill.add_argument("--execute", action="store_true")
    fill.add_argument("--confirm")
    warmup = sub.add_parser("boot-warmup")
    warmup.add_argument("--job-id", required=True)
    warmup.add_argument("--pod", required=True)
    warmup.add_argument("--gpu", required=True, type=int)
    warmup.add_argument("--attempt-id", required=True)
    warmup.add_argument("--execute", action="store_true")
    warmup.add_argument("--confirm")
    full_scene = sub.add_parser("full-scene-probe")
    full_scene.add_argument("--job-id", required=True)
    full_scene.add_argument("--pod", required=True)
    full_scene.add_argument("--gpu", required=True, type=int)
    full_scene.add_argument("--attempt-id", required=True)
    full_scene.add_argument("--execute", action="store_true")
    full_scene.add_argument("--confirm")
    prepare_asset = sub.add_parser("prepare-source-assets")
    prepare_asset.add_argument("--job-id", required=True)
    prepare_asset.add_argument("--pod", required=True)
    prepare_asset.add_argument("--execute", action="store_true")
    prepare_asset.add_argument("--confirm")
    finalize_full_scene = sub.add_parser("finalize-full-scene-probe")
    finalize_full_scene.add_argument("--job-id", required=True)
    finalize_full_scene.add_argument("--pod", required=True)
    finalize_full_scene.add_argument("--gpu", required=True, type=int)
    finalize_full_scene.add_argument("--attempt-id", required=True)
    finalize_full_scene.add_argument("--execute", action="store_true")
    finalize_full_scene.add_argument("--confirm")
    attest = sub.add_parser("attest-milestone")
    attest.add_argument("--job-id", required=True)
    attest.add_argument("--milestone", type=int, required=True)
    attest.add_argument("--execute", action="store_true")
    attest.add_argument("--confirm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue.resolve())
        if args.mode == "plan":
            result = cmd_plan(queue, live=args.live)
        elif args.mode == "status":
            result = cmd_status(queue, live=args.live)
        elif args.mode == "doctor":
            result = cmd_doctor(queue, live=args.live)
        elif args.mode == "fill":
            result = cmd_fill(
                queue,
                execute=args.execute,
                confirm=args.confirm,
                count=args.count,
            )
        elif args.mode == "boot-warmup":
            result = cmd_boot_warmup(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                gpu=args.gpu,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "full-scene-probe":
            result = cmd_full_scene_probe(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                gpu=args.gpu,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "prepare-source-assets":
            result = cmd_prepare_source_assets(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "finalize-full-scene-probe":
            result = cmd_finalize_full_scene_probe(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                gpu=args.gpu,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "attest-milestone":
            result = cmd_attest_milestone(
                queue,
                job_id=args.job_id,
                milestone=args.milestone,
                execute=args.execute,
                confirm=args.confirm,
            )
        else:
            result = cmd_launch_next(queue, execute=args.execute, confirm=args.confirm)
    except QueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
