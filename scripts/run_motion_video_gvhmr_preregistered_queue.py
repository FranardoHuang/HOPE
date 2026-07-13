#!/usr/bin/env python3
"""Run one exact no-clobber 2026-07-13 S0/M0 offline-GVHMR batch.

This launcher accepts only a committed preregistration and its one-shot Pod
execution record.  All runtime arguments, including GPU placement,
``nvidia-smi``, polling cadence, source/output/state roots and tool hashes, are
read from that record and revalidated.  It atomically reserves every global
GVHMR output namespace, holds cross-state claim locks through binding
publication, and hashes each source immediately before and after its child.

The launcher authorizes no GMR, robot motion, simulator, TOPP, RL, deployment
or hardware work.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_motion_video_gvhmr_queue as legacy  # noqa: E402
from audit_motion_video_intake import sha256_file  # noqa: E402
from validate_motion_video_gvhmr_prereg import (  # noqa: E402
    PreregError,
    ensure_no_symlink_components,
    fsync_directory,
    validate_execution_record_for_launch,
    validate_runtime_after_execution,
    write_json_exclusive,
)


class QueueError(RuntimeError):
    """The preregistered queue cannot safely start or continue."""


def _sha256_fd(descriptor: int) -> str:
    import hashlib

    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def open_bound_source(path: Path, expected_bytes: int, expected_sha256: str) -> tuple[int, dict[str, Any]]:
    ensure_no_symlink_components(path, "asset source")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QueueError(f"cannot open bound source {path}: {exc}") from None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise QueueError(f"asset source must be a regular file: {path}")
        fingerprint = {
            "device": info.st_dev,
            "inode": info.st_ino,
            "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
            "sha256": _sha256_fd(descriptor),
        }
        if fingerprint["bytes"] != expected_bytes or fingerprint["sha256"] != expected_sha256:
            raise QueueError(
                f"asset source binding mismatch before launch: {path} "
                f"expected_bytes={expected_bytes} actual_bytes={fingerprint['bytes']} "
                f"expected_sha256={expected_sha256} actual_sha256={fingerprint['sha256']}"
            )
        return descriptor, fingerprint
    except BaseException:
        os.close(descriptor)
        raise


def verify_bound_source_after(
    descriptor: int,
    path: Path,
    before: dict[str, Any],
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    info = os.fstat(descriptor)
    after = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "sha256": _sha256_fd(descriptor),
    }
    try:
        path_info = path.lstat()
    except OSError as exc:
        raise QueueError(f"asset source disappeared during execution: {path}: {exc}") from None
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        raise QueueError(f"asset source identity changed during execution: {path}")
    if (path_info.st_dev, path_info.st_ino) != (after["device"], after["inode"]):
        raise QueueError(f"asset source path was replaced during execution: {path}")
    if after != before:
        raise QueueError(f"asset source bytes or identity changed during execution: {path}")
    if after["bytes"] != expected_bytes or after["sha256"] != expected_sha256:
        raise QueueError(f"asset source no longer matches its manifest after execution: {path}")
    return after


def _copy_fd_exact(source_fd: int, destination_fd: int) -> None:
    os.lseek(source_fd, 0, os.SEEK_SET)
    while True:
        block = os.read(source_fd, 1024 * 1024)
        if not block:
            break
        view = memoryview(block)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise QueueError("short write while creating bound source snapshot")
            view = view[written:]
    os.fsync(destination_fd)
    os.lseek(source_fd, 0, os.SEEK_SET)


def materialize_bound_source_snapshots(
    manifest: dict[str, Any],
    asset_ids: list[str],
    *,
    source_root: Path,
    state_dir: Path,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, tuple[int, dict[str, Any], Path]]]:
    """Copy exact source fds into a new private, read-only child input tree.

    GVHMR consumes this tree, never the mutable staging pathname.  The source is
    verified around the copy; the snapshot is content-verified, held open and
    reverified around the child.  Directory/file modes remove ordinary write
    paths after the complete batch snapshot has been materialized.
    """

    snapshot_root = state_dir / "bound_sources"
    ensure_no_symlink_components(state_dir, "state root for bound snapshots")
    try:
        snapshot_root.mkdir(mode=0o700)
    except FileExistsError:
        raise QueueError(f"bound source snapshot root already exists: {snapshot_root}") from None
    fsync_directory(state_dir)

    assets = {str(asset["id"]): asset for asset in manifest["assets"]}
    records: dict[str, dict[str, Any]] = {}
    held: dict[str, tuple[int, dict[str, Any], Path]] = {}
    try:
        for asset_id in asset_ids:
            asset = assets[asset_id]
            relative = str(asset["source_relpath"])
            ensure_no_symlink_components(
                source_root / relative, f"staged source path for {asset_id}"
            )
            source = legacy.resolve_asset_path(source_root, relative)
            snapshot = legacy.resolve_asset_path(snapshot_root, relative)
            snapshot.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            ensure_no_symlink_components(snapshot.parent, f"snapshot parent for {asset_id}")

            source_fd, source_before = open_bound_source(
                source, int(asset["bytes"]), str(asset["sha256"])
            )
            destination_fd: int | None = None
            snapshot_fd: int | None = None
            try:
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                destination_fd = os.open(snapshot, flags, 0o400)
                _copy_fd_exact(source_fd, destination_fd)
                os.fchmod(destination_fd, 0o400)
                os.close(destination_fd)
                destination_fd = None
                fsync_directory(snapshot.parent)

                verify_bound_source_after(
                    source_fd,
                    source,
                    source_before,
                    int(asset["bytes"]),
                    str(asset["sha256"]),
                )
                snapshot_fd, snapshot_before = open_bound_source(
                    snapshot, int(asset["bytes"]), str(asset["sha256"])
                )
                records[asset_id] = {
                    "source_path": str(source),
                    "source_before": source_before,
                    "snapshot_path": str(snapshot),
                    "snapshot_before": snapshot_before,
                    "consumption": "GVHMR child pathname resolves inside this private snapshot tree",
                }
                held[asset_id] = (snapshot_fd, snapshot_before, snapshot)
                snapshot_fd = None
            finally:
                if destination_fd is not None:
                    os.close(destination_fd)
                if snapshot_fd is not None:
                    os.close(snapshot_fd)
                os.close(source_fd)

        directories = sorted(
            (path for path in snapshot_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            os.chmod(directory, 0o500)
        os.chmod(snapshot_root, 0o500)
        fsync_directory(state_dir)
        return snapshot_root, records, held
    except BaseException:
        for descriptor, _before, _path in held.values():
            os.close(descriptor)
        raise


def reserve_output_namespaces(
    outputs: list[dict[str, Any]],
    *,
    prereg_sha256: str,
    execution_record_sha256: str,
) -> tuple[list[int], list[str]]:
    """Atomically claim each shared output stem and hold its lock fd."""

    locks: list[int] = []
    namespaces: list[str] = []
    for row in outputs:
        asset_id = str(row["asset_id"])
        output = Path(str(row["path"]))
        namespace = output.parent
        ensure_no_symlink_components(namespace.parent, f"output parent for {asset_id}")
        ensure_no_symlink_components(namespace, f"output namespace for {asset_id}", require_leaf=False)
        try:
            namespace.mkdir(mode=0o700)
        except FileExistsError:
            raise QueueError(f"no-clobber output namespace already exists: {namespace}") from None
        fsync_directory(namespace.parent)
        claim_path = namespace / ".hope_gvhmr_claim.json"
        write_json_exclusive(
            claim_path,
            {
                "schema_version": 1,
                "status": "claimed",
                "asset_id": asset_id,
                "output_path": str(output),
                "prereg_sha256": prereg_sha256,
                "execution_record_sha256": execution_record_sha256,
                "scope": "offline_gvhmr_only",
            },
        )
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(claim_path, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
        if output.exists():
            os.close(descriptor)
            raise QueueError(f"output appeared during namespace reservation: {output}")
        locks.append(descriptor)
        namespaces.append(str(namespace))
    return locks, namespaces


def mark_source_drift(binding_path: Path, message: str) -> None:
    payload: dict[str, Any]
    if binding_path.is_file():
        try:
            payload = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"schema_version": 1}
    else:
        payload = {"schema_version": 1}
    payload.update(
        status="failed",
        failure=message,
        completed_utc=legacy.utc_now(),
    )
    legacy.atomic_json(binding_path, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--execution-record", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_state: dict[str, Any] | None = None
    queue_state_path: Path | None = None
    lock_descriptors: list[int] = []
    try:
        binding = validate_execution_record_for_launch(
            args.prereg,
            args.execution_record,
            repo_root=SCRIPT_DIR.parent,
        )
        static = binding["static"]
        prereg = static["prereg"]
        record = binding["record"]
        manifest = static["intake"]
        if manifest.get("intake_id") != "motion-video-intake-20260713-v12-static-lateral":
            raise QueueError("secure queue only accepts the exact 2026-07-13 intake id")
        if manifest.get("processing_order") != prereg["processing_order"]:
            raise QueueError("intake processing order changed")

        state_dir = Path(record["state_root"])
        ensure_no_symlink_components(state_dir, "state root", require_leaf=False)
        try:
            state_dir.mkdir(mode=0o700)
        except FileExistsError:
            raise QueueError(f"no-clobber state root already exists: {state_dir}") from None
        fsync_directory(state_dir.parent)
        queue_lock_path = state_dir / "queue.lock"
        queue_lock_fd = os.open(
            queue_lock_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        fcntl.flock(queue_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_descriptors.append(queue_lock_fd)
        fsync_directory(state_dir)

        queue_state_path = state_dir / "queue_state.json"
        queue_state = {
            "schema_version": 2,
            "status": "reserving_outputs",
            "authorization_scope": record["authorization_scope"],
            "prereg_path": str(args.prereg.resolve()),
            "prereg_sha256": binding["prereg_sha256"],
            "execution_record_path": str(args.execution_record.resolve()),
            "execution_record_sha256": binding["execution_record_sha256"],
            "asset_ids": prereg["execution_batch"]["asset_ids"],
            "batch_id": prereg["execution_batch"]["batch_id"],
            "gpu_physical_index": record["gpu_physical_index"],
            "started_utc": legacy.utc_now(),
        }
        legacy.atomic_json(queue_state_path, queue_state)

        claim_locks, namespaces = reserve_output_namespaces(
            prereg["output_contract"]["outputs"],
            prereg_sha256=binding["prereg_sha256"],
            execution_record_sha256=binding["execution_record_sha256"],
        )
        lock_descriptors.extend(claim_locks)
        queue_state.update(status="running", claimed_output_namespaces=namespaces)
        legacy.atomic_json(queue_state_path, queue_state)

        runtime = prereg["gvhmr_runtime"]
        source_root = Path(record["source_root"])
        gvhmr_root = Path(record["gvhmr_root"])
        python = Path(record["python"])
        result_auditor = static["tool_paths"]["result_auditor"]
        assets = {str(asset["id"]): asset for asset in manifest["assets"]}
        snapshot_root, snapshot_records, snapshot_handles = materialize_bound_source_snapshots(
            manifest,
            prereg["execution_batch"]["asset_ids"],
            source_root=source_root,
            state_dir=state_dir,
        )
        lock_descriptors.extend(
            descriptor for descriptor, _before, _path in snapshot_handles.values()
        )
        queue_state.update(
            source_consumption="private_read_only_no_clobber_snapshot",
            bound_source_snapshots=snapshot_records,
        )
        legacy.atomic_json(queue_state_path, queue_state)
        processing_contract = {
            "manifest_sha256": prereg["intake"]["sha256"],
            "manual_event_review_sha256": prereg["manual_event_review"]["sha256"],
            "franco_priority_review_sha256": prereg["franco_priority_context"]["sha256"],
            "prereg_sha256": binding["prereg_sha256"],
            "execution_record_sha256": binding["execution_record_sha256"],
            "queue_tool_sha256": sha256_file(Path(__file__).resolve()),
            "legacy_queue_library_sha256": sha256_file(static["tool_paths"]["legacy_intake_guard"]),
            "result_auditor_sha256": sha256_file(result_auditor),
            "execution_validator_sha256": sha256_file(
                SCRIPT_DIR / "validate_motion_video_gvhmr_prereg.py"
            ),
            "gvhmr_commit": runtime["commit"],
            "gvhmr_worktree_clean": True,
            "gvhmr_dependency_tree": record["live_dependency_binding"]["checkpoint_tree"],
            "python_environment": record["live_dependency_binding"]["python_environment"],
            "nvidia_smi": record["nvidia_smi"],
            "static_camera": True,
            "authorization_scope": record["authorization_scope"],
            "source_consumption": "private_read_only_no_clobber_snapshot",
            "bound_source_snapshot_sha256": {
                asset_id: snapshot_records[asset_id]["snapshot_before"]["sha256"]
                for asset_id in prereg["execution_batch"]["asset_ids"]
            },
        }

        for asset_id in prereg["execution_batch"]["asset_ids"]:
            asset = assets[asset_id]
            snapshot_fd, snapshot_before, snapshot = snapshot_handles[asset_id]
            run_ok = False
            drift_error: QueueError | None = None
            try:
                run_ok = legacy.run_asset(
                    asset,
                    source_root=snapshot_root,
                    gvhmr_root=gvhmr_root,
                    python=python,
                    state_dir=state_dir,
                    gpu=record["gpu_physical_index"],
                    max_used_mib=record["max_used_mib"],
                    poll_seconds=record["poll_seconds"],
                    wait_timeout_seconds=record["wait_timeout_seconds"],
                    nvidia_smi=record["nvidia_smi"]["realpath"],
                    static_camera=True,
                    processing_contract=processing_contract,
                    result_auditor=result_auditor,
                )
            finally:
                try:
                    verify_bound_source_after(
                        snapshot_fd,
                        snapshot,
                        snapshot_before,
                        int(asset["bytes"]),
                        str(asset["sha256"]),
                    )
                except QueueError as exc:
                    drift_error = exc
            if drift_error is not None:
                binding_path = state_dir / "bindings" / f"{asset_id}.json"
                mark_source_drift(binding_path, str(drift_error))
                queue_state.update(
                    status="failed",
                    failed_asset_id=asset_id,
                    failure=str(drift_error),
                    completed_utc=legacy.utc_now(),
                )
                legacy.atomic_json(queue_state_path, queue_state)
                return 1
            if not run_ok:
                queue_state.update(
                    status="failed",
                    failed_asset_id=asset_id,
                    completed_utc=legacy.utc_now(),
                )
                legacy.atomic_json(queue_state_path, queue_state)
                return 1

        validate_runtime_after_execution(binding)
        for asset_id in prereg["execution_batch"]["asset_ids"]:
            asset = assets[asset_id]
            output = gvhmr_root / "outputs" / "demo" / Path(
                str(asset["source_relpath"])
            ).stem / "hmr4d_results.pt"
            binding_path = state_dir / "bindings" / f"{asset_id}.json"
            try:
                completed = json.loads(binding_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise QueueError(f"cannot reread final binding for {asset_id}: {exc}") from None
            if not legacy.completed_binding_matches(
                completed, asset, output, processing_contract
            ):
                raise QueueError(f"final output/binding revalidation failed for {asset_id}")
        queue_state.update(status="complete", completed_utc=legacy.utc_now())
        legacy.atomic_json(queue_state_path, queue_state)
        print(
            "[gvhmr-preregistered-queue] PASS: "
            f"{prereg['execution_batch']['batch_id']} "
            f"({len(prereg['execution_batch']['asset_ids'])} assets)",
            flush=True,
        )
        return 0
    except (PreregError, QueueError, legacy.IntakeError, OSError, subprocess.SubprocessError) as exc:
        if queue_state is not None and queue_state_path is not None:
            queue_state.update(status="fatal", failure=str(exc), completed_utc=legacy.utc_now())
            legacy.atomic_json(queue_state_path, queue_state)
        print(f"[gvhmr-preregistered-queue] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        for descriptor in reversed(lock_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
