#!/usr/bin/env python3
"""Run a content-addressed GVHMR intake serially behind a GPU-memory gate.

This is a preprocessing queue, not a training launcher.  It verifies the raw
video manifest before doing work, records one immutable binding JSON per
successful output, stops at the first failed clip, and never edits a training
checkout.  Existing GVHMR output without a matching completed binding is
rejected instead of silently reused.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_motion_video_intake import (  # noqa: E402
    IntakeError,
    audit_assets,
    load_manifest,
    resolve_asset_path,
    resolve_source_root,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_clean_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise IntakeError(f"cannot resolve GVHMR git HEAD at {path}: {result.stderr.strip()}")
    commit = result.stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise IntakeError(f"cannot inspect GVHMR worktree at {path}: {status.stderr.strip()}")
    if status.stdout.strip():
        changed = [line for line in status.stdout.splitlines() if line.strip()]
        raise IntakeError(f"GVHMR worktree must be clean; changed entries={changed[:10]}")
    return commit


def tree_fingerprint(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise IntakeError(f"dependency tree is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise IntakeError(f"dependency tree contains no files: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha = sha256_file(path)
        total_bytes += size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(file_sha.encode("ascii") + b"\n")
    return {
        "root": str(root),
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def python_fingerprint(python: Path) -> dict[str, str]:
    version = subprocess.run(
        [str(python), "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode != 0:
        raise IntakeError(f"motion Python --version failed: {version.stderr.strip()}")
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze.returncode != 0:
        raise IntakeError(f"motion Python pip freeze failed: {freeze.stderr.strip()}")
    normalized = "\n".join(sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip()))
    return {
        "executable": str(python),
        "version": (version.stdout or version.stderr).strip(),
        "pip_freeze_sha256": hashlib.sha256((normalized + "\n").encode()).hexdigest(),
    }


def select_assets(manifest: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    assets = {str(asset["id"]): asset for asset in manifest["assets"]}
    if not requested:
        selected = [assets[asset_id] for asset_id in manifest["processing_order"]]
    else:
        if len(set(requested)) != len(requested):
            raise IntakeError("--asset-id values must be unique")
        missing = [asset_id for asset_id in requested if asset_id not in assets]
        if missing:
            raise IntakeError(f"requested asset ids are absent from manifest: {missing}")
        selected = [assets[asset_id] for asset_id in requested]

    # GVHMR writes outputs/demo/<source stem>/hmr4d_results.pt, so two source
    # paths with the same basename would alias even when their manifest ids are
    # different.  Reject that ambiguity before any GPU work starts.
    output_keys: dict[str, str] = {}
    for asset in selected:
        key = Path(str(asset["source_relpath"])).stem
        previous = output_keys.setdefault(key, str(asset["id"]))
        if previous != str(asset["id"]):
            raise IntakeError(
                f"GVHMR output stem {key!r} aliases assets {previous!r} and {asset['id']!r}"
            )
    return selected


def gpu_used_mib(gpu: int, nvidia_smi: str = "nvidia-smi") -> int:
    executable = shutil.which(nvidia_smi) if os.sep not in nvidia_smi else nvidia_smi
    if not executable:
        raise IntakeError(f"nvidia-smi executable not found: {nvidia_smi!r}")
    result = subprocess.run(
        [
            executable,
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise IntakeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise IntakeError(f"expected one GPU memory row for index {gpu}, got {rows}")
    try:
        return int(rows[0])
    except ValueError:
        raise IntakeError(f"invalid nvidia-smi memory value: {rows[0]!r}") from None


def wait_for_memory(
    gpu: int,
    max_used_mib: int,
    poll_seconds: float,
    timeout_seconds: float,
    nvidia_smi: str,
) -> int:
    started = time.monotonic()
    while True:
        used = gpu_used_mib(gpu, nvidia_smi=nvidia_smi)
        if used <= max_used_mib:
            return used
        elapsed = time.monotonic() - started
        if timeout_seconds > 0.0 and elapsed >= timeout_seconds:
            raise IntakeError(
                f"GPU {gpu} stayed above {max_used_mib} MiB for {elapsed:.0f} s "
                f"(last={used} MiB)"
            )
        print(
            f"[gvhmr-queue] GPU {gpu} used={used} MiB > gate={max_used_mib}; waiting",
            flush=True,
        )
        time.sleep(poll_seconds)


def completed_binding_matches(
    binding: dict[str, Any],
    asset: dict[str, Any],
    output: Path,
    processing_contract: dict[str, Any],
) -> bool:
    if binding.get("status") != "complete" or binding.get("source_sha256") != asset["sha256"]:
        return False
    if binding.get("processing_contract") != processing_contract or not output.is_file():
        return False
    if binding.get("output_sha256") != sha256_file(output):
        return False
    audit_raw = binding.get("structural_audit_path")
    if not isinstance(audit_raw, str):
        return False
    audit_path = Path(audit_raw)
    if not audit_path.is_file() or binding.get("structural_audit_sha256") != sha256_file(audit_path):
        return False
    try:
        audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        audit_report.get("status") == "pass"
        and audit_report.get("result_sha256") == binding.get("output_sha256")
    )


def run_asset(
    asset: dict[str, Any],
    *,
    source_root: Path,
    gvhmr_root: Path,
    python: Path,
    state_dir: Path,
    gpu: int,
    max_used_mib: int,
    poll_seconds: float,
    wait_timeout_seconds: float,
    nvidia_smi: str,
    static_camera: bool,
    processing_contract: dict[str, Any],
    result_auditor: Path,
) -> bool:
    asset_id = str(asset["id"])
    source = resolve_asset_path(source_root, str(asset["source_relpath"]))
    output = gvhmr_root / "outputs" / "demo" / source.stem / "hmr4d_results.pt"
    binding_path = state_dir / "bindings" / f"{asset_id}.json"
    log_path = state_dir / "logs" / f"{asset_id}.log"
    audit_path = state_dir / "audits" / f"{asset_id}.json"
    if binding_path.is_file():
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IntakeError(f"invalid existing binding {binding_path}: {exc}") from None
        if completed_binding_matches(binding, asset, output, processing_contract):
            print(f"[gvhmr-queue] SKIP verified complete {asset_id}", flush=True)
            return True
        raise IntakeError(f"stale or mismatched existing binding for {asset_id}: {binding_path}")
    if output.exists():
        raise IntakeError(
            f"GVHMR output already exists without a matching completed binding: {output}"
        )

    used = wait_for_memory(
        gpu, max_used_mib, poll_seconds, wait_timeout_seconds, nvidia_smi
    )
    command = [str(python), "tools/demo/demo.py", f"--video={source}"]
    if static_camera:
        command.append("-s")
    started = utc_now()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "asset_id": asset_id,
        "source_path": str(source),
        "source_sha256": asset["sha256"],
        "processing_contract": processing_contract,
        "gvhmr_root": str(gvhmr_root),
        "gvhmr_commit": processing_contract["gvhmr_commit"],
        "gpu_physical_index": gpu,
        "gpu_used_mib_before": used,
        "memory_gate_mib": max_used_mib,
        "static_camera": static_camera,
        "command": command,
        "started_utc": started,
        "output_path": str(output),
        "log_path": str(log_path),
        "structural_audit_path": str(audit_path),
    }
    atomic_json(binding_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print(
        f"[gvhmr-queue] START {asset_id} gpu={gpu} used={used} MiB source={source.name}",
        flush=True,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=gvhmr_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            audit_returncode: int | None = None
            if result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
                audit_command = [
                    str(python),
                    str(result_auditor),
                    "--result", str(output),
                    "--expected-frames", str(asset["media"]["frames"]),
                    "--json-out", str(audit_path),
                ]
                log.write("\n[gvhmr-queue] structural output audit\n")
                log.flush()
                audit_result = subprocess.run(
                    audit_command,
                    cwd=gvhmr_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                audit_returncode = audit_result.returncode
                payload["audit_command"] = audit_command
            payload["audit_returncode"] = audit_returncode
    except OSError as exc:
        payload.update(
            status="failed",
            completed_utc=utc_now(),
            failure=f"cannot execute GVHMR/auditor: {exc}",
        )
        atomic_json(binding_path, payload)
        print(f"[gvhmr-queue] FAIL {asset_id}; see {log_path}", file=sys.stderr, flush=True)
        return False
    payload["completed_utc"] = utc_now()
    payload["returncode"] = result.returncode
    if (
        result.returncode != 0
        or not output.is_file()
        or output.stat().st_size <= 0
        or payload.get("audit_returncode") != 0
        or not audit_path.is_file()
    ):
        payload["status"] = "failed"
        payload["failure"] = (
            f"returncode={result.returncode}, output_exists={output.is_file()}, "
            f"output_bytes={output.stat().st_size if output.is_file() else 0}, "
            f"audit_returncode={payload.get('audit_returncode')}"
        )
        atomic_json(binding_path, payload)
        print(f"[gvhmr-queue] FAIL {asset_id}; see {log_path}", file=sys.stderr, flush=True)
        return False
    try:
        audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload.update(status="failed", failure=f"invalid structural audit JSON: {exc}")
        atomic_json(binding_path, payload)
        print(f"[gvhmr-queue] FAIL {asset_id}; see {log_path}", file=sys.stderr, flush=True)
        return False
    output_sha256 = sha256_file(output)
    if audit_report.get("status") != "pass" or audit_report.get("result_sha256") != output_sha256:
        payload.update(status="failed", failure="structural audit status/result SHA mismatch")
        atomic_json(binding_path, payload)
        print(f"[gvhmr-queue] FAIL {asset_id}; see {log_path}", file=sys.stderr, flush=True)
        return False
    payload["status"] = "complete"
    payload["output_bytes"] = output.stat().st_size
    payload["output_sha256"] = output_sha256
    payload["structural_audit_sha256"] = sha256_file(audit_path)
    atomic_json(binding_path, payload)
    print(
        f"[gvhmr-queue] COMPLETE {asset_id} output_sha256={payload['output_sha256'][:12]}...",
        flush=True,
    )
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gvhmr-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--asset-id", action="append", default=None)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-used-mib", type=int, default=19000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--static-camera", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_state: dict[str, Any] | None = None
    queue_state_path: Path | None = None
    try:
        if args.gpu < 0 or args.max_used_mib <= 0 or args.poll_seconds <= 0.0:
            raise IntakeError("gpu must be non-negative; max-used-mib and poll-seconds must be positive")
        manifest = load_manifest(args.manifest.resolve())
        source_root = resolve_source_root(manifest, args.source_root)
        failures = audit_assets(manifest, source_root)
        if failures:
            raise IntakeError("raw intake audit failed: " + "; ".join(failures))
        gvhmr_root = args.gvhmr_root.resolve()
        python = args.python.resolve()
        state_dir = args.state_dir.resolve()
        if not (gvhmr_root / "tools" / "demo" / "demo.py").is_file():
            raise IntakeError(f"GVHMR demo entrypoint missing under {gvhmr_root}")
        if not python.is_file():
            raise IntakeError(f"motion Python is missing: {python}")
        result_auditor = SCRIPT_DIR / "audit_gvhmr_result.py"
        if not result_auditor.is_file():
            raise IntakeError(f"GVHMR result auditor is missing: {result_auditor}")
        selected = select_assets(manifest, args.asset_id)
        state_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (state_dir / "queue.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise IntakeError(f"another GVHMR queue owns {state_dir / 'queue.lock'}") from None
        commit = git_clean_head(gvhmr_root)
        manifest_sha256 = sha256_file(args.manifest.resolve())
        queue_tool_sha256 = sha256_file(Path(__file__).resolve())
        dependency_tree = tree_fingerprint(gvhmr_root / "inputs" / "checkpoints")
        python_environment = python_fingerprint(python)
        processing_contract = {
            "manifest_sha256": manifest_sha256,
            "queue_tool_sha256": queue_tool_sha256,
            "result_auditor_sha256": sha256_file(result_auditor),
            "gvhmr_commit": commit,
            "gvhmr_worktree_clean": True,
            "gvhmr_dependency_tree": dependency_tree,
            "python_environment": python_environment,
            "static_camera": bool(args.static_camera),
        }
        queue_state = {
            "schema_version": 1,
            "status": "running",
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha256,
            "queue_tool_sha256": queue_tool_sha256,
            "gvhmr_commit": commit,
            "processing_contract": processing_contract,
            "asset_ids": [asset["id"] for asset in selected],
            "gpu_physical_index": args.gpu,
            "started_utc": utc_now(),
        }
        queue_state_path = state_dir / "queue_state.json"
        atomic_json(queue_state_path, queue_state)
        for asset in selected:
            if not run_asset(
                asset,
                source_root=source_root,
                gvhmr_root=gvhmr_root,
                python=python,
                state_dir=state_dir,
                gpu=args.gpu,
                max_used_mib=args.max_used_mib,
                poll_seconds=args.poll_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
                nvidia_smi=args.nvidia_smi,
                static_camera=bool(args.static_camera),
                processing_contract=processing_contract,
                result_auditor=result_auditor,
            ):
                queue_state.update(status="failed", failed_asset_id=asset["id"], completed_utc=utc_now())
                atomic_json(queue_state_path, queue_state)
                return 1
        queue_state.update(status="complete", completed_utc=utc_now())
        atomic_json(queue_state_path, queue_state)
        print(f"[gvhmr-queue] PASS: {len(selected)} assets", flush=True)
        return 0
    except (IntakeError, OSError, subprocess.SubprocessError) as exc:
        if queue_state is not None and queue_state_path is not None:
            queue_state.update(
                status="fatal",
                failure=str(exc),
                completed_utc=utc_now(),
            )
            atomic_json(queue_state_path, queue_state)
        print(f"[gvhmr-queue] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
