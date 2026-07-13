#!/usr/bin/env python3
"""Strict no-clobber rebind for one schema-3 Stage-1 question bank.

This is not a legacy-load escape hatch and it never regenerates or edits a
question.  It permits one metadata-only rebind when Git and AST evidence prove
that every previously existing physics-contract module is executable-AST
identical and the only addition is one frozen, unused top-level helper.  All
non-metadata arrays must remain byte-identical.  The current runtime loader must
accept the rebound bank before the completion report is published.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_PHYSICS_FILES = (
    "configs/ball_physics_venue.yaml",
    "hope_ws/src/hope_planner/hope_planner/constants.py",
    "hope_ws/src/hope_planner/hope_planner/ball_contact.py",
    "hope_ws/src/hope_planner/hope_planner/ball_trajectory_predictor.py",
    "hope_ws/src/hope_planner/hope_planner/strike_spec_planner.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
    "tasks/tracking/mdp/virtual_ball.py",
    "hope_training/whole_body_tracking/scripts/venue_ball_sampler.py",
)


class RebindError(RuntimeError):
    """A bank/source/publication invariant failed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise RebindError(f"{label} must be one lowercase SHA-256")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebindError(f"cannot read rebind manifest: {exc}") from exc
    if value.get("schema_version") != 1:
        raise RebindError("rebind manifest schema_version must be 1")
    if value.get("manifest_id") != (
        "phase1-signed-face-schema3-bank-additive-physics-rebind-20260713-v1"
    ):
        raise RebindError("unexpected rebind manifest_id")
    if value.get("simulation_only") is not True or value.get("real_robot_commands_forbidden") is not True:
        raise RebindError("rebind must remain simulation-only and forbid robot commands")
    if value.get("legacy_load_forbidden") is not True:
        raise RebindError("legacy-load escape hatch must remain forbidden")
    for label in ("base_commit", "target_commit"):
        if not COMMIT_RE.fullmatch(str(value.get(label, ""))):
            raise RebindError(f"{label} must be one full lowercase Git commit")
    source_bank = value.get("source_bank")
    physics = value.get("physics_contract")
    support = value.get("invariant_support_files")
    output = value.get("output")
    if not all(isinstance(item, dict) for item in (source_bank, physics, support, output)):
        raise RebindError("source_bank/physics_contract/invariant_support_files/output must be objects")
    require_sha(source_bank.get("sha256"), "source bank")
    require_sha(source_bank.get("physics_contract_sha256"), "source physics contract")
    for label in (
        "base_changed_file_sha256",
        "target_changed_file_sha256",
        "git_diff_sha256",
        "allowed_added_function_ast_sha256",
        "target_physics_contract_sha256",
    ):
        require_sha(physics.get(label), label)
    require_sha(value.get("target_source_family_sha256"), "target source-family")
    if tuple(physics.get("files") or ()) != EXPECTED_PHYSICS_FILES:
        raise RebindError("physics-contract file set/order changed")
    if physics.get("contract_name") != "stage1-physics-runtime-v1":
        raise RebindError("physics contract name changed")
    if physics.get("only_changed_file") not in EXPECTED_PHYSICS_FILES:
        raise RebindError("only_changed_file is not a physics-contract file")
    if physics.get("allowed_added_top_level_function") != "signed_face_hemisphere":
        raise RebindError("only signed_face_hemisphere may be added")
    expected_support = {
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/stage1_question_bank.py",
        "hope_training/whole_body_tracking/scripts/gen_stage1_questions.py",
    }
    if set(support) != expected_support:
        raise RebindError("invariant support-file set changed")
    for relative, digest in support.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RebindError(f"unsafe invariant support path: {relative}")
        require_sha(digest, f"invariant support file {relative}")
    if output.get("root_must_not_exist") is not True or output.get("completion_report_written_last") is not True:
        raise RebindError("no-clobber/completion-last publication was weakened")
    for key in ("source_repo", "runtime_python"):
        if not isinstance(value.get(key), str) or not Path(value[key]).is_absolute():
            raise RebindError(f"{key} must be an absolute path")
    for key in ("path",):
        if not isinstance(source_bank.get(key), str) or not Path(source_bank[key]).is_absolute():
            raise RebindError("source bank path must be absolute")
    motion_runtime = source_bank.get("motion_runtime")
    if not isinstance(motion_runtime, dict) or set(motion_runtime) != set(source_bank.get("clip_order") or []):
        raise RebindError("motion_runtime must bind every clip exactly once")
    for clip, runtime in motion_runtime.items():
        if not isinstance(runtime, dict) or set(runtime) != {"path", "n_frames", "strike_phase"}:
            raise RebindError(f"{clip} motion_runtime contract changed")
        if not isinstance(runtime["path"], str) or not Path(runtime["path"]).is_absolute():
            raise RebindError(f"{clip} motion path must be absolute")
        if int(runtime["n_frames"]) <= 1 or not 0.0 <= float(runtime["strike_phase"]) <= 1.0:
            raise RebindError(f"{clip} motion frame/phase contract is invalid")
    if not isinstance(output.get("root"), str) or not Path(output["root"]).is_absolute():
        raise RebindError("output root must be absolute")
    for key in ("bank_basename", "report_basename"):
        candidate = Path(str(output.get(key, "")))
        if not candidate.name or candidate.name != str(output.get(key)) or candidate.is_absolute():
            raise RebindError(f"output {key} must be one safe basename")
    return value


def git_bytes(repo: Path, args: list[str]) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        message = exc.output.decode("utf-8", errors="replace").strip()
        raise RebindError(f"git {' '.join(args)} failed: {message}") from exc


def git_text(repo: Path, args: list[str]) -> str:
    return git_bytes(repo, args).decode("utf-8", errors="strict").strip()


def git_show(repo: Path, commit: str, relative: str) -> bytes:
    return git_bytes(repo, ["show", f"{commit}:{relative}"])


def physics_contract_from_files(files: Mapping[str, str], contract_name: str) -> dict[str, Any]:
    return {"contract": contract_name, "files": dict(files)}


def prove_additive_source_change(manifest: Mapping[str, Any]) -> dict[str, Any]:
    repo = Path(manifest["source_repo"])
    if not repo.is_dir():
        raise RebindError(f"source repo is missing: {repo}")
    head = git_text(repo, ["rev-parse", "HEAD"])
    if head != manifest["target_commit"]:
        raise RebindError(f"source repo HEAD {head} != target {manifest['target_commit']}")
    if git_text(repo, ["status", "--porcelain"]):
        raise RebindError("source repo must be clean")

    base_hashes: dict[str, str] = {}
    target_hashes: dict[str, str] = {}
    base_bytes: dict[str, bytes] = {}
    target_bytes: dict[str, bytes] = {}
    for relative in EXPECTED_PHYSICS_FILES:
        old = git_show(repo, manifest["base_commit"], relative)
        new = git_show(repo, manifest["target_commit"], relative)
        current_path = repo / relative
        if not current_path.is_file() or current_path.is_symlink():
            raise RebindError(f"current physics file missing or symlinked: {relative}")
        if current_path.read_bytes() != new:
            raise RebindError(f"clean worktree bytes disagree with target commit: {relative}")
        base_bytes[relative] = old
        target_bytes[relative] = new
        base_hashes[relative] = sha256_bytes(old)
        target_hashes[relative] = sha256_bytes(new)

    support_hashes: dict[str, str] = {}
    for relative, expected_sha in manifest["invariant_support_files"].items():
        old = git_show(repo, manifest["base_commit"], relative)
        new = git_show(repo, manifest["target_commit"], relative)
        if old != new or sha256_bytes(new) != expected_sha:
            raise RebindError(f"invariant bank support file changed: {relative}")
        support_hashes[relative] = expected_sha

    changed = [name for name in EXPECTED_PHYSICS_FILES if base_bytes[name] != target_bytes[name]]
    expected_changed = manifest["physics_contract"]["only_changed_file"]
    if changed != [expected_changed]:
        raise RebindError(f"physics file delta must be exactly {[expected_changed]}, got {changed}")
    physics = manifest["physics_contract"]
    if base_hashes[expected_changed] != physics["base_changed_file_sha256"]:
        raise RebindError("base changed-file SHA mismatch")
    if target_hashes[expected_changed] != physics["target_changed_file_sha256"]:
        raise RebindError("target changed-file SHA mismatch")
    diff = git_bytes(
        repo,
        [
            "diff",
            "--no-ext-diff",
            "--full-index",
            manifest["base_commit"],
            manifest["target_commit"],
            "--",
            expected_changed,
        ],
    )
    if sha256_bytes(diff) != physics["git_diff_sha256"]:
        raise RebindError("frozen additive source diff SHA mismatch")

    try:
        old_tree = ast.parse(base_bytes[expected_changed].decode("utf-8"))
        new_tree = ast.parse(target_bytes[expected_changed].decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RebindError(f"changed physics module is not valid UTF-8 Python: {exc}") from exc
    name = physics["allowed_added_top_level_function"]
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name for node in old_tree.body):
        raise RebindError(f"base module already defines {name}")
    additions = [
        node for node in new_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(additions) != 1 or additions[0].decorator_list:
        raise RebindError(f"target must add exactly one undecorated top-level {name} function")
    added_ast_sha = sha256_bytes(ast.dump(additions[0], include_attributes=False).encode("utf-8"))
    if added_ast_sha != physics["allowed_added_function_ast_sha256"]:
        raise RebindError("added function AST SHA mismatch")
    new_tree.body = [node for node in new_tree.body if node is not additions[0]]
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise RebindError("pre-existing virtual-ball executable AST changed")

    current_contract = physics_contract_from_files(target_hashes, physics["contract_name"])
    if canonical_sha256(current_contract) != physics["target_physics_contract_sha256"]:
        raise RebindError("target physics-contract SHA mismatch")
    return {
        "base_commit": manifest["base_commit"],
        "target_commit": manifest["target_commit"],
        "base_file_sha256": base_hashes,
        "target_file_sha256": target_hashes,
        "only_changed_file": expected_changed,
        "git_diff_sha256": sha256_bytes(diff),
        "added_top_level_function": name,
        "added_function_ast_sha256": added_ast_sha,
        "preexisting_executable_ast_equal": True,
        "target_physics_contract": current_contract,
        "target_physics_contract_sha256": canonical_sha256(current_contract),
        "invariant_support_file_sha256": support_hashes,
    }


def decode_metadata(array: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(array)
    if raw.dtype != np.dtype("uint8") or raw.ndim != 1:
        raise RebindError("meta_json must be a one-dimensional uint8 array")
    try:
        value = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RebindError(f"invalid meta_json: {exc}") from exc
    if not isinstance(value, dict):
        raise RebindError("meta_json must decode to an object")
    return value


def array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise RebindError("object arrays are forbidden")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "c_order_bytes": int(contiguous.nbytes),
        "c_order_sha256": sha256_bytes(contiguous.tobytes(order="C")),
    }


def load_npz_stable(path: Path, expected_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read one regular no-symlink NPZ through a stable descriptor."""
    try:
        if path.resolve(strict=True) != path:
            raise RebindError(f"input path contains a symlink: {path}")
    except FileNotFoundError:
        raise RebindError(f"input file is missing: {path}") from None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RebindError(f"input is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RebindError(f"input SHA mismatch: {path}")
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as stream:
            with np.load(stream, allow_pickle=False) as loaded:
                arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RebindError(f"input changed while it was being read: {path}")
        receipt = {field: int(getattr(after, field)) for field in stable_fields}
        receipt["sha256"] = expected_sha256
        return arrays, receipt
    finally:
        os.close(fd)


def stable_regular_file_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Hash a regular input through O_NOFOLLOW and prove it stayed stable."""
    try:
        if path.resolve(strict=True) != path:
            raise RebindError(f"input path contains a symlink: {path}")
    except FileNotFoundError:
        raise RebindError(f"input file is missing: {path}") from None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RebindError(f"input is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RebindError(f"input changed while it was being hashed: {path}")
        if digest.hexdigest() != expected_sha256:
            raise RebindError(f"input SHA mismatch: {path}")
        receipt = {field: int(getattr(after, field)) for field in stable_fields}
        receipt["sha256"] = expected_sha256
        receipt["path"] = str(path)
        return receipt
    finally:
        os.close(fd)


def metadata_leaf_differences(old: Any, new: Any, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        paths: list[str] = []
        for key in sorted(set(old) | set(new)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                paths.append(child)
            else:
                paths.extend(metadata_leaf_differences(old[key], new[key], child))
        return paths
    if old != new:
        return [prefix]
    return []


def prepare_rebound_arrays(
    manifest: Mapping[str, Any], source_proof: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_cfg = manifest["source_bank"]
    try:
        arrays, source_receipt = load_npz_stable(
            Path(source_cfg["path"]), source_cfg["sha256"]
        )
    except Exception as exc:
        if isinstance(exc, RebindError):
            raise
        raise RebindError(f"cannot load source bank without pickle: {exc}") from exc
    if "meta_json" not in arrays:
        raise RebindError("source bank has no meta_json")
    old_meta = decode_metadata(arrays["meta_json"])
    if old_meta.get("schema_version") != source_cfg["schema_version"]:
        raise RebindError("source bank schema mismatch")
    if old_meta.get("split") != source_cfg["split"]:
        raise RebindError("source bank split mismatch")
    if list(old_meta.get("clip_order") or []) != list(source_cfg["clip_order"]):
        raise RebindError("source bank clip order mismatch")
    old_contract = old_meta.get("physics_contract")
    old_contract_sha = old_meta.get("physics_contract_sha256")
    if not isinstance(old_contract, dict) or canonical_sha256(old_contract) != old_contract_sha:
        raise RebindError("source physics contract/hash is internally inconsistent")
    if old_contract_sha != source_cfg["physics_contract_sha256"]:
        raise RebindError("source physics contract SHA is not the preregistered value")
    if old_contract != physics_contract_from_files(
        source_proof["base_file_sha256"], manifest["physics_contract"]["contract_name"]
    ):
        raise RebindError("source bank physics contract is not the exact base-commit contract")
    family = old_meta.get("source_family_contract")
    if not isinstance(family, dict) or canonical_sha256(family) != old_meta.get("source_family_sha256"):
        raise RebindError("source family contract/hash is internally inconsistent")
    if family.get("physics_contract_sha256") != old_contract_sha:
        raise RebindError("source family does not bind the source physics contract")

    for clip in source_cfg["clip_order"]:
        info = (old_meta.get("clips") or {}).get(clip) or {}
        if info.get("motion_sha256") != source_cfg["motion_sha256"][clip]:
            raise RebindError(f"{clip} motion SHA mismatch")
        expected_count = int(source_cfg["question_counts"][clip])
        if int(info.get("question_count", -1)) != expected_count:
            raise RebindError(f"{clip} metadata question count mismatch")
        for suffix in ("incoming_vel", "incoming_spin", "demanded_vel", "demanded_normal"):
            key = f"{clip}/{suffix}"
            if key not in arrays or np.asarray(arrays[key]).shape != (expected_count, 3):
                raise RebindError(f"{key} shape/count mismatch")
        difficulty = f"{clip}/difficulty_deg"
        if difficulty not in arrays or np.asarray(arrays[difficulty]).shape != (expected_count,):
            raise RebindError(f"{difficulty} shape/count mismatch")

    fingerprints = {
        key: array_fingerprint(value)
        for key, value in arrays.items()
        if key != "meta_json"
    }
    for key, value in arrays.items():
        if key == "meta_json":
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise RebindError(f"source question array contains non-finite values: {key}")

    motion_receipts = {}
    for clip in source_cfg["clip_order"]:
        motion_receipts[clip] = stable_regular_file_receipt(
            Path(source_cfg["motion_runtime"][clip]["path"]),
            source_cfg["motion_sha256"][clip],
        )

    new_meta = copy.deepcopy(old_meta)
    new_contract = copy.deepcopy(source_proof["target_physics_contract"])
    new_contract_sha = source_proof["target_physics_contract_sha256"]
    new_meta["physics_contract"] = new_contract
    new_meta["physics_contract_sha256"] = new_contract_sha
    new_meta["source_family_contract"]["physics_contract_sha256"] = new_contract_sha
    new_meta["source_family_sha256"] = canonical_sha256(new_meta["source_family_contract"])
    if new_meta["source_family_sha256"] != manifest["target_source_family_sha256"]:
        raise RebindError("target source-family SHA is not the preregistered value")
    differences = metadata_leaf_differences(old_meta, new_meta)
    allowed = sorted(manifest["allowed_metadata_leaf_changes"])
    if differences != allowed:
        raise RebindError(f"metadata delta is not exact: expected={allowed}, actual={differences}")
    arrays["meta_json"] = np.frombuffer(canonical_json_bytes(new_meta), dtype=np.uint8).copy()
    return arrays, {
        "old_metadata_sha256": canonical_sha256(old_meta),
        "new_metadata_sha256": canonical_sha256(new_meta),
        "old_source_family_sha256": old_meta["source_family_sha256"],
        "new_source_family_sha256": new_meta["source_family_sha256"],
        "allowed_metadata_leaf_changes": differences,
        "non_metadata_arrays": fingerprints,
        "all_non_metadata_arrays_finite": True,
        "source_file_receipt": source_receipt,
        "motion_file_receipts": motion_receipts,
        "source_npz_key_order": list(arrays),
    }


def write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def verify_written_bank_identity(
    path: Path, manifest: Mapping[str, Any], bank_proof: Mapping[str, Any]
) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        if list(loaded.files) != list(bank_proof["source_npz_key_order"]):
            raise RebindError("written bank changed NPZ key order")
        actual = {
            key: array_fingerprint(loaded[key])
            for key in loaded.files
            if key != "meta_json"
        }
        written_meta = decode_metadata(loaded["meta_json"])
    if actual != bank_proof["non_metadata_arrays"]:
        raise RebindError("written bank changed one or more non-metadata arrays")
    if canonical_sha256(written_meta) != bank_proof["new_metadata_sha256"]:
        raise RebindError("written bank metadata SHA differs from the audited new metadata")
    source_arrays, _ = load_npz_stable(
        Path(manifest["source_bank"]["path"]), manifest["source_bank"]["sha256"]
    )
    source_meta = decode_metadata(source_arrays["meta_json"])
    differences = metadata_leaf_differences(source_meta, written_meta)
    if differences != bank_proof["allowed_metadata_leaf_changes"]:
        raise RebindError(
            f"written metadata delta changed: expected={bank_proof['allowed_metadata_leaf_changes']}, "
            f"actual={differences}"
        )


def validate_with_target_runtime(
    manifest: Mapping[str, Any], bank_path: Path
) -> dict[str, Any]:
    python = Path(manifest["runtime_python"])
    # A normal venv ``bin/python`` is commonly a symlink to its versioned
    # interpreter.  The absolute configured entrypoint is allowed, but it must
    # resolve to a regular executable; source/bank inputs remain no-symlink.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RebindError(f"runtime Python is missing or not executable: {python}")
    repo = Path(manifest["source_repo"])
    module = repo / (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/stage1_question_bank.py"
    )
    changed_file = manifest["physics_contract"]["only_changed_file"]
    base_source = git_show(repo, manifest["base_commit"], changed_file).decode("utf-8")
    request = {
        "base_virtual_ball_source": base_source,
        "target_virtual_ball_path": str(repo / changed_file),
        "venue_path": str(repo / "configs/ball_physics_venue.yaml"),
        "bank_path": str(bank_path),
        "stage1_bank_module_path": str(module),
        "split": manifest["source_bank"]["split"],
        "clip_order": manifest["source_bank"]["clip_order"],
        "motion_files": [
            manifest["source_bank"]["motion_runtime"][clip]["path"]
            for clip in manifest["source_bank"]["clip_order"]
        ],
        "motion_frames": [
            manifest["source_bank"]["motion_runtime"][clip]["n_frames"]
            for clip in manifest["source_bank"]["clip_order"]
        ],
        "strike_phases": [
            manifest["source_bank"]["motion_runtime"][clip]["strike_phase"]
            for clip in manifest["source_bank"]["clip_order"]
        ],
    }
    code = r'''
import hashlib, importlib.util, json, platform, sys, types
import numpy as np
import torch
request = json.load(sys.stdin)
module_path = request["stage1_bank_module_path"]
bank_path = request["bank_path"]
split = request["split"]
spec = importlib.util.spec_from_file_location("rebind_runtime_stage1_question_bank", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bank = module.load_question_bank(bank_path, device="cpu", expected_split=split, allow_legacy=False)
module.validate_runtime_motion_contract(
    bank.metadata,
    request["motion_files"],
    request["motion_frames"],
    request["strike_phases"],
)

def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded

base_vb = types.ModuleType("rebind_base_virtual_ball")
base_vb.__file__ = "git-show-base-virtual_ball.py"
sys.modules[base_vb.__name__] = base_vb
exec(compile(request["base_virtual_ball_source"], base_vb.__file__, "exec"), base_vb.__dict__)
target_vb = load_path("rebind_target_virtual_ball", request["target_virtual_ball_path"])
base_params = base_vb.load_venue_params(request["venue_path"])
target_params = target_vb.load_venue_params(request["venue_path"])
torch.set_default_dtype(torch.float64)

def raw_fingerprint(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }

replay = {}
with np.load(bank_path, allow_pickle=False) as data:
    meta = json.loads(np.asarray(data["meta_json"], dtype=np.uint8).tobytes().decode("utf-8"))
    for clip in request["clip_order"]:
        incoming = torch.as_tensor(np.asarray(data[f"{clip}/incoming_vel"]), dtype=torch.float64)
        spin = torch.as_tensor(np.asarray(data[f"{clip}/incoming_spin"]), dtype=torch.float64)
        demanded_vel = torch.as_tensor(np.asarray(data[f"{clip}/demanded_vel"]), dtype=torch.float64)
        demanded_normal = torch.as_tensor(np.asarray(data[f"{clip}/demanded_normal"]), dtype=torch.float64)
        count = int(incoming.shape[0])
        p0 = torch.as_tensor(
            np.tile(np.asarray(data[f"{clip}/contact_pos_env"]), (count, 1)),
            dtype=torch.float64,
        )
        old_v, old_w = base_vb.predict_paddle_contact(
            incoming, demanded_vel, demanded_normal, spin, base_params
        )
        new_v, new_w = target_vb.predict_paddle_contact(
            incoming, demanded_vel, demanded_normal, spin, target_params
        )
        kwargs = {
            "surface_z": float(meta["table_surface_z"]) + float(target_params.ball_radius),
            "net_x": float(meta["near_x"]) + 1.37,
            "h": 0.01,
            "n_steps": 200,
        }
        old_land = base_vb.coarse_landing(p0, old_v, old_w, base_params, **kwargs)
        new_land = target_vb.coarse_landing(p0, new_v, new_w, target_params, **kwargs)
        old_outputs = {"v_plus": old_v, "w_plus": old_w, **old_land}
        new_outputs = {"v_plus": new_v, "w_plus": new_w, **new_land}
        if set(old_outputs) != set(new_outputs):
            raise RuntimeError(f"{clip}: old/new replay output keys changed")
        output_sha = {}
        for key in sorted(old_outputs):
            old_fp = raw_fingerprint(old_outputs[key])
            new_fp = raw_fingerprint(new_outputs[key])
            if old_fp != new_fp:
                raise RuntimeError(f"{clip}: old/new replay differs at {key}: {old_fp} != {new_fp}")
            output_sha[key] = old_fp
        target = torch.as_tensor(meta["landing_env"], dtype=torch.float64)
        error = torch.linalg.norm(new_land["land_xy"] - target, dim=-1)
        net_top = float(meta["table_surface_z"]) + 0.1525
        net_margin = new_land["net_z"] - (net_top + float(target_params.ball_radius))
        net_ok = new_land["net_valid"] & (net_margin > 0.0)
        if not bool(torch.isfinite(error).all()) or not bool(new_land["land_valid"].all()):
            raise RuntimeError(f"{clip}: rebound replay has invalid/non-finite landing")
        if not bool(net_ok.all()) or float(error.max()) > 0.10:
            raise RuntimeError(f"{clip}: rebound replay fails bank landing/net gate")
        replay[clip] = {
            "question_count": count,
            "old_new_all_output_bytes_equal": True,
            "output_fingerprints": output_sha,
            "max_landing_error_m": float(error.max()),
            "min_net_margin_m": float(net_margin.min()),
            "landing_valid_count": int(new_land["land_valid"].sum()),
            "net_clear_count": int(net_ok.sum()),
        }
value = {
    "clip_order": bank.metadata["clip_order"],
    "counts": [int(item) for item in bank.counts.cpu().tolist()],
    "schema_version": int(bank.metadata["schema_version"]),
    "split": bank.metadata["split"],
    "physics_contract_sha256": bank.metadata["physics_contract_sha256"],
    "source_family_sha256": bank.metadata["source_family_sha256"],
    "runtime_motion_contract_valid": True,
    "base_target_behavior_replay": replay,
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    },
}
print("REBIND_RUNTIME_JSON=" + json.dumps(value, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=repo,
        input=json.dumps(request, sort_keys=True),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    marker = "REBIND_RUNTIME_JSON="
    payload = next((line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)), None)
    if completed.returncode != 0 or payload is None:
        raise RebindError(
            "target runtime rejected rebound bank: "
            f"rc={completed.returncode}, output={completed.stdout[-4000:]}"
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RebindError(f"target runtime returned invalid validation JSON: {exc}") from exc
    expected_counts = [
        int(manifest["source_bank"]["question_counts"][clip])
        for clip in manifest["source_bank"]["clip_order"]
    ]
    if value.get("counts") != expected_counts:
        raise RebindError(f"runtime counts changed: {value.get('counts')} != {expected_counts}")
    if value.get("clip_order") != manifest["source_bank"]["clip_order"]:
        raise RebindError("runtime observed wrong clip order")
    if value.get("schema_version") != manifest["source_bank"]["schema_version"]:
        raise RebindError("runtime observed wrong bank schema")
    if value.get("split") != manifest["source_bank"]["split"]:
        raise RebindError("runtime observed wrong bank split")
    if value.get("physics_contract_sha256") != manifest["physics_contract"]["target_physics_contract_sha256"]:
        raise RebindError("runtime observed wrong target physics contract")
    if value.get("source_family_sha256") != manifest["target_source_family_sha256"]:
        raise RebindError("runtime observed wrong target source-family contract")
    if value.get("runtime_motion_contract_valid") is not True:
        raise RebindError("runtime motion contract was not validated")
    replay = value.get("base_target_behavior_replay")
    if not isinstance(replay, dict) or set(replay) != set(manifest["source_bank"]["clip_order"]):
        raise RebindError("runtime behavior replay is incomplete")
    for clip, expected_count in manifest["source_bank"]["question_counts"].items():
        result = replay.get(clip) or {}
        if result.get("question_count") != expected_count or result.get("old_new_all_output_bytes_equal") is not True:
            raise RebindError(f"{clip} runtime behavior replay did not prove exact equality")
    return value


def claim_output_root(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path, 0o755)
    except FileExistsError:
        raise RebindError(f"no-clobber output root already exists: {path}") from None


def publish(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    script_path: Path,
    source_proof: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    bank_proof: Mapping[str, Any],
    expected_manifest_sha256: str,
    expected_script_sha256: str,
) -> dict[str, Any]:
    output = manifest["output"]
    root = Path(output["root"])
    claim_output_root(root)
    partial = root / (output["bank_basename"] + ".partial")
    final_bank = root / output["bank_basename"]
    report_path = root / output["report_basename"]
    write_npz_exclusive(partial, arrays)
    verify_written_bank_identity(partial, manifest, bank_proof)
    runtime = validate_with_target_runtime(manifest, partial)
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RebindError("rebind manifest changed during execution")
    if sha256_file(script_path) != expected_script_sha256:
        raise RebindError("rebind script changed during execution")
    repeated_source_proof = prove_additive_source_change(manifest)
    if canonical_sha256(repeated_source_proof) != canonical_sha256(source_proof):
        raise RebindError("target source proof changed during execution")
    try:
        os.link(partial, final_bank)
    except FileExistsError:
        raise RebindError(f"no-clobber final bank already exists: {final_bank}") from None
    partial.unlink()
    dir_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    content = {
        "manifest_id": manifest["manifest_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "script_path": str(script_path),
        "script_sha256": expected_script_sha256,
        "source_bank": str(manifest["source_bank"]["path"]),
        "source_bank_sha256": manifest["source_bank"]["sha256"],
        "output_bank": str(final_bank),
        "output_bank_sha256": sha256_file(final_bank),
        "source_proof": source_proof,
        "bank_proof": bank_proof,
        "runtime_validation": runtime,
        "question_arrays_changed": False,
        "legacy_load_used": False,
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
    }
    report = {
        "artifact_kind": "stage1_schema3_additive_physics_contract_rebind",
        "schema_version": 1,
        "content": content,
        "content_sha256": canonical_sha256(content),
    }
    write_json_exclusive(report_path, report)
    dir_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return {
        "status": "published",
        "bank": str(final_bank),
        "bank_sha256": content["output_bank_sha256"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("action", choices=("validate", "run"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = args.config.resolve()
    script = Path(__file__).resolve()
    if sha256_file(config) != require_sha(args.expected_config_sha256, "expected config"):
        raise RebindError("rebind manifest SHA mismatch")
    if sha256_file(script) != require_sha(args.expected_script_sha256, "expected script"):
        raise RebindError("rebind script SHA mismatch")
    manifest = load_manifest(config)
    source_proof = prove_additive_source_change(manifest)
    arrays, bank_proof = prepare_rebound_arrays(manifest, source_proof)
    if args.action == "validate":
        print(json.dumps({
            "status": "validated_no_writes",
            "manifest_id": manifest["manifest_id"],
            "source_bank_sha256": manifest["source_bank"]["sha256"],
            "target_physics_contract_sha256": source_proof["target_physics_contract_sha256"],
            "question_array_count": len(bank_proof["non_metadata_arrays"]),
            "question_arrays_changed": False,
        }, sort_keys=True))
        return 0
    print(json.dumps(publish(
        manifest, config, script, source_proof, arrays, bank_proof,
        expected_manifest_sha256=args.expected_config_sha256,
        expected_script_sha256=args.expected_script_sha256,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RebindError as exc:
        print(f"[stage1-bank-rebind] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
