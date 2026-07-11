#!/usr/bin/env python3
"""Serially retarget tracked GVHMR results to diagnostic A3 GMR pickles.

Every input and output is content-addressed.  The queue requires a clean GMR
checkout, records the GMR/tool/Python/environment contract, audits frame-zero
warm-up convergence plus the output structure, and stops at the first failure.
Outputs deliberately retain the per-video GVHMR betas and are therefore marked
``formal_eligible=false`` regardless of structural audit success.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class QueueError(ValueError):
    """The GMR diagnostic queue contract cannot be satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_clean_head(path: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        raise QueueError(f"cannot resolve GMR git HEAD at {path}: {head.stderr.strip()}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise QueueError(f"cannot inspect GMR worktree at {path}: {status.stderr.strip()}")
    changed = [line for line in status.stdout.splitlines() if line.strip()]
    if changed:
        raise QueueError(f"GMR worktree must be clean; changed entries={changed[:10]}")
    return head.stdout.strip()


def verify_source_bundle(gmr_root: Path, bundle: Path, commit: str) -> dict[str, Any]:
    if not bundle.is_file() or bundle.stat().st_size <= 0:
        raise QueueError(f"GMR source bundle is missing or empty: {bundle}")
    verify = subprocess.run(
        ["git", "-C", str(gmr_root), "bundle", "verify", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise QueueError(f"GMR source bundle verification failed: {verify.stderr.strip()}")
    heads = subprocess.run(
        ["git", "-C", str(gmr_root), "bundle", "list-heads", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    if heads.returncode != 0 or not any(
        line.split(maxsplit=1)[0] == commit for line in heads.stdout.splitlines() if line.strip()
    ):
        raise QueueError(f"GMR source bundle does not advertise required commit {commit}")
    return {
        "path": str(bundle),
        "bytes": bundle.stat().st_size,
        "sha256": sha256_file(bundle),
        "verified_commit": commit,
    }


def python_fingerprint(python: Path) -> dict[str, str]:
    version = subprocess.run(
        [str(python), "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode != 0:
        raise QueueError(f"motion Python --version failed: {version.stderr.strip()}")
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze.returncode != 0:
        raise QueueError(f"motion Python pip freeze failed: {freeze.stderr.strip()}")
    normalized = "\n".join(
        sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip())
    )
    return {
        "executable": str(python),
        "version": (version.stdout or version.stderr).strip(),
        "pip_freeze_sha256": hashlib.sha256((normalized + "\n").encode()).hexdigest(),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"cannot read GVHMR result manifest {path}: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise QueueError("GVHMR result manifest must be a complete mapping")
    if manifest.get("formal_eligible") is not False:
        raise QueueError("GVHMR result manifest must explicitly remain formal_eligible=false")
    rows = manifest.get("results")
    if not isinstance(rows, list) or not rows:
        raise QueueError("GVHMR result manifest requires a non-empty results list")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise QueueError(f"results[{index}] must be a mapping")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
            raise QueueError(f"results[{index}].asset_id is unsafe: {asset_id!r}")
        if asset_id in seen:
            raise QueueError(f"duplicate result asset_id: {asset_id}")
        seen.add(asset_id)
        if not isinstance(row.get("result_path"), str):
            raise QueueError(f"results[{index}].result_path must be a string")
        sha = row.get("result_sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise QueueError(f"results[{index}].result_sha256 is invalid")
        if not isinstance(row.get("frames"), int) or row["frames"] <= 1:
            raise QueueError(f"results[{index}].frames must be an integer > 1")
        if not isinstance(row.get("result_bytes"), int) or row["result_bytes"] <= 0:
            raise QueueError(f"results[{index}].result_bytes must be a positive integer")
        if row.get("structural_status") != "pass":
            raise QueueError(f"results[{index}] lacks a passing GVHMR structural audit")
    return manifest


def select_results(
    manifest: dict[str, Any], requested: list[str] | None
) -> list[dict[str, Any]]:
    rows = list(manifest["results"])
    by_id = {str(row["asset_id"]): row for row in rows}
    if not requested:
        return rows
    if len(requested) != len(set(requested)):
        raise QueueError("--asset-id values must be unique")
    missing = [asset_id for asset_id in requested if asset_id not in by_id]
    if missing:
        raise QueueError(f"requested asset ids are absent from manifest: {missing}")
    return [by_id[asset_id] for asset_id in requested]


def verify_inputs(rows: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    for row in rows:
        source = Path(str(row["result_path"])).resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            failures.append(f"{row['asset_id']}: missing/empty {source}")
            continue
        expected_bytes = row.get("result_bytes")
        if isinstance(expected_bytes, int) and source.stat().st_size != expected_bytes:
            failures.append(
                f"{row['asset_id']}: bytes {source.stat().st_size} != {expected_bytes}"
            )
            continue
        actual_sha = sha256_file(source)
        if actual_sha != row["result_sha256"]:
            failures.append(
                f"{row['asset_id']}: sha256 {actual_sha} != {row['result_sha256']}"
            )
    if failures:
        raise QueueError("GVHMR input verification failed: " + "; ".join(failures))


def build_command(
    python: Path, entrypoint: Path, source: Path, output: Path
) -> list[str]:
    # Deliberately do not expose --no-warmup or --velocity-limit: this lane
    # requires the source warm-up fix and preserves the original legal swing.
    return [
        str(python),
        str(entrypoint),
        "--gvhmr_pred_file",
        str(source),
        "--robot",
        "agibot_a3",
        "--save_path",
        str(output),
    ]


def build_environment(
    gmr_root: Path, omp_threads: int, mkl_threads: int
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(gmr_root),
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(mkl_threads),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
    )
    return env


def completed_binding_matches(
    binding: dict[str, Any],
    row: dict[str, Any],
    output: Path,
    processing_contract: dict[str, Any],
) -> bool:
    if (
        binding.get("status") != "complete"
        or binding.get("source_sha256") != row["result_sha256"]
        or binding.get("processing_contract") != processing_contract
        or binding.get("body_shape_contract") != "diagnostic_video_betas"
        or binding.get("formal_eligible") is not False
        or not output.is_file()
        or binding.get("output_sha256") != sha256_file(output)
    ):
        return False
    audit_raw = binding.get("structural_audit_path")
    log_raw = binding.get("run_log_path")
    if not isinstance(audit_raw, str) or not isinstance(log_raw, str):
        return False
    audit_path, log_path = Path(audit_raw), Path(log_raw)
    if not audit_path.is_file() or not log_path.is_file():
        return False
    if binding.get("structural_audit_sha256") != sha256_file(audit_path):
        return False
    if binding.get("run_log_sha256") != sha256_file(log_path):
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        audit.get("status") == "pass"
        and audit.get("result_sha256") == binding.get("output_sha256")
        and audit.get("run_log_sha256") == binding.get("run_log_sha256")
        and audit.get("actual_frames") == row["frames"]
        and audit.get("body_shape_contract") == "diagnostic_video_betas"
        and audit.get("formal_eligible") is False
    )


def run_result(
    row: dict[str, Any],
    *,
    gmr_root: Path,
    python: Path,
    output_root: Path,
    state_dir: Path,
    processing_contract: dict[str, Any],
    result_auditor: Path,
    timeout_seconds: float,
    omp_threads: int,
    mkl_threads: int,
    warmup_threshold: float,
    warmup_max_rounds: int,
    warmup_regex: str | None,
) -> bool:
    asset_id = str(row["asset_id"])
    source = Path(str(row["result_path"])).resolve()
    output = output_root / f"{asset_id}.diagnostic_video_betas.pkl"
    binding_path = state_dir / "bindings" / f"{asset_id}.json"
    log_path = state_dir / "logs" / f"{asset_id}.log"
    audit_path = state_dir / "audits" / f"{asset_id}.json"
    if binding_path.is_file():
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QueueError(f"invalid existing binding {binding_path}: {exc}") from None
        if completed_binding_matches(binding, row, output, processing_contract):
            print(f"[gmr-queue] SKIP verified complete {asset_id}", flush=True)
            return True
        raise QueueError(f"stale or mismatched existing binding for {asset_id}: {binding_path}")
    if output.exists():
        raise QueueError(f"GMR output exists without a matching completed binding: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        python, gmr_root / "scripts" / "gvhmr_to_robot.py", source, output
    )
    env = build_environment(gmr_root, omp_threads, mkl_threads)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "asset_id": asset_id,
        "source_path": str(source),
        "source_sha256": row["result_sha256"],
        "source_frames": row["frames"],
        "processing_contract": processing_contract,
        "gmr_root": str(gmr_root),
        "gmr_commit": processing_contract["gmr_commit"],
        "body_shape_contract": "diagnostic_video_betas",
        "formal_eligible": False,
        "formal_blockers": [
            "retarget uses the per-video GVHMR body betas rather than canonical A3-bound body shape",
            "structural and warm-up audits do not prove collision, table/net clearance, strike feasibility, or robot safety",
        ],
        "command": command,
        "environment_overrides": processing_contract["environment_overrides"],
        "timeout_seconds": timeout_seconds,
        "started_utc": utc_now(),
        "output_path": str(output),
        "run_log_path": str(log_path),
        "structural_audit_path": str(audit_path),
    }
    atomic_json(binding_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gmr-queue] START {asset_id} source={source.name}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=gmr_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload.update(
            status="failed",
            completed_utc=utc_now(),
            failure=f"cannot complete GMR command: {exc}",
        )
        atomic_json(binding_path, payload)
        print(f"[gmr-queue] FAIL {asset_id}; see {log_path}", file=sys.stderr, flush=True)
        return False

    payload["returncode"] = result.returncode
    payload["completed_utc"] = utc_now()
    if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        payload.update(
            status="failed",
            failure=(
                f"returncode={result.returncode}, output_exists={output.is_file()}, "
                f"output_bytes={output.stat().st_size if output.is_file() else 0}"
            ),
        )
        atomic_json(binding_path, payload)
        print(f"[gmr-queue] FAIL {asset_id}; see {log_path}", file=sys.stderr, flush=True)
        return False

    audit_command = [
        str(python),
        str(result_auditor),
        "--result",
        str(output),
        "--expected-frames",
        str(row["frames"]),
        "--run-log",
        str(log_path),
        "--warmup-threshold",
        str(warmup_threshold),
        "--warmup-max-rounds",
        str(warmup_max_rounds),
        "--json-out",
        str(audit_path),
    ]
    if warmup_regex is not None:
        audit_command.extend(["--warmup-regex", warmup_regex])
    payload["audit_command"] = audit_command
    try:
        audit_result = subprocess.run(
            audit_command,
            cwd=gmr_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=min(timeout_seconds, 120.0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload.update(status="failed", failure=f"cannot complete GMR audit: {exc}")
        atomic_json(binding_path, payload)
        print(f"[gmr-queue] FAIL {asset_id}; auditor failed", file=sys.stderr, flush=True)
        return False
    payload["audit_returncode"] = audit_result.returncode
    payload["audit_stdout"] = audit_result.stdout[-4000:]
    payload["audit_stderr"] = audit_result.stderr[-4000:]
    if audit_result.returncode != 0 or not audit_path.is_file():
        payload.update(
            status="failed",
            failure=f"audit_returncode={audit_result.returncode}, audit_exists={audit_path.is_file()}",
        )
        atomic_json(binding_path, payload)
        print(
            f"[gmr-queue] FAIL {asset_id}; see audit output in {binding_path}",
            file=sys.stderr,
            flush=True,
        )
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload.update(status="failed", failure=f"invalid GMR structural audit JSON: {exc}")
        atomic_json(binding_path, payload)
        return False
    output_sha = sha256_file(output)
    log_sha = sha256_file(log_path)
    if (
        audit.get("status") != "pass"
        or audit.get("result_sha256") != output_sha
        or audit.get("run_log_sha256") != log_sha
        or audit.get("body_shape_contract") != "diagnostic_video_betas"
        or audit.get("formal_eligible") is not False
    ):
        payload.update(status="failed", failure="GMR audit status/content SHA mismatch")
        atomic_json(binding_path, payload)
        return False
    payload.update(
        status="complete",
        output_bytes=output.stat().st_size,
        output_sha256=output_sha,
        run_log_sha256=log_sha,
        structural_audit_sha256=sha256_file(audit_path),
        warmup=audit["warmup"],
    )
    atomic_json(binding_path, payload)
    print(f"[gmr-queue] COMPLETE {asset_id} output_sha256={output_sha[:12]}...", flush=True)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument(
        "--gmr-bundle",
        type=Path,
        required=True,
        help="verified recovery bundle that advertises the clean GMR HEAD",
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--asset-id", action="append", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--mkl-threads", type=int, default=8)
    parser.add_argument("--warmup-threshold", type=float, default=1e-4)
    parser.add_argument("--warmup-max-rounds", type=int, default=200)
    parser.add_argument(
        "--warmup-regex",
        help="optional Python regex with named groups 'rounds' and 'max_dq'",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_state: dict[str, Any] | None = None
    queue_state_path: Path | None = None
    try:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
            raise QueueError("timeout-seconds must be finite and positive")
        if args.omp_threads <= 0 or args.mkl_threads <= 0:
            raise QueueError("OMP/MKL thread counts must be positive")
        if not math.isfinite(args.warmup_threshold) or args.warmup_threshold <= 0.0:
            raise QueueError("warm-up threshold must be finite and positive")
        if args.warmup_max_rounds <= 0:
            raise QueueError("warm-up max rounds must be positive")
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        selected = select_results(manifest, args.asset_id)
        verify_inputs(selected)
        gmr_root = args.gmr_root.resolve()
        gmr_bundle = args.gmr_bundle.resolve()
        python = args.python.resolve()
        output_root = args.output_root.resolve()
        state_dir = args.state_dir.resolve()
        entrypoint = gmr_root / "scripts" / "gvhmr_to_robot.py"
        result_auditor = SCRIPT_DIR / "audit_gmr_result.py"
        if not entrypoint.is_file():
            raise QueueError(f"GMR entrypoint is missing: {entrypoint}")
        if not result_auditor.is_file():
            raise QueueError(f"GMR result auditor is missing: {result_auditor}")
        if not python.is_file():
            raise QueueError(f"motion Python is missing: {python}")

        state_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (state_dir / "queue.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise QueueError(f"another GMR queue owns {state_dir / 'queue.lock'}") from None

        commit = git_clean_head(gmr_root)
        source_bundle = verify_source_bundle(gmr_root, gmr_bundle, commit)
        environment_overrides = {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(gmr_root),
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.mkl_threads),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
        manifest_sha = sha256_file(manifest_path)
        queue_tool_sha = sha256_file(Path(__file__).resolve())
        processing_contract = {
            "manifest_sha256": manifest_sha,
            "queue_tool_sha256": queue_tool_sha,
            "result_auditor_sha256": sha256_file(result_auditor),
            "gmr_commit": commit,
            "gmr_worktree_clean": True,
            "gmr_source_bundle": source_bundle,
            "gmr_entrypoint_sha256": sha256_file(entrypoint),
            "python_environment": python_fingerprint(python),
            "environment_overrides": environment_overrides,
            "robot": "agibot_a3",
            "body_shape_contract": "diagnostic_video_betas",
            "formal_eligible": False,
            "warmup_threshold_strict_lt": args.warmup_threshold,
            "warmup_max_rounds": args.warmup_max_rounds,
            "warmup_regex": args.warmup_regex,
        }
        queue_state = {
            "schema_version": 1,
            "status": "running",
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "queue_tool_sha256": queue_tool_sha,
            "gmr_commit": commit,
            "processing_contract": processing_contract,
            "asset_ids": [row["asset_id"] for row in selected],
            "body_shape_contract": "diagnostic_video_betas",
            "formal_eligible": False,
            "started_utc": utc_now(),
        }
        queue_state_path = state_dir / "queue_state.json"
        atomic_json(queue_state_path, queue_state)
        for row in selected:
            if not run_result(
                row,
                gmr_root=gmr_root,
                python=python,
                output_root=output_root,
                state_dir=state_dir,
                processing_contract=processing_contract,
                result_auditor=result_auditor,
                timeout_seconds=args.timeout_seconds,
                omp_threads=args.omp_threads,
                mkl_threads=args.mkl_threads,
                warmup_threshold=args.warmup_threshold,
                warmup_max_rounds=args.warmup_max_rounds,
                warmup_regex=args.warmup_regex,
            ):
                queue_state.update(
                    status="failed", failed_asset_id=row["asset_id"], completed_utc=utc_now()
                )
                atomic_json(queue_state_path, queue_state)
                return 1
        queue_state.update(status="complete", completed_utc=utc_now())
        atomic_json(queue_state_path, queue_state)
        print(f"[gmr-queue] PASS: {len(selected)} diagnostic assets", flush=True)
        return 0
    except (QueueError, OSError, subprocess.SubprocessError) as exc:
        if queue_state is not None and queue_state_path is not None:
            queue_state.update(status="fatal", failure=str(exc), completed_utc=utc_now())
            atomic_json(queue_state_path, queue_state)
        print(f"[gmr-queue] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
