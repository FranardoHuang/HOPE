#!/usr/bin/env python3
"""Exactly stop the bound rolling pool before the task-revision cutover.

The local entry point reads the tracked queue and sends this same source to one
Pod as a remote worker over stdin.  The worker validates every registered
binding twice before signalling only its numeric PID=PGID.  It writes an
O_EXCL intent before the first TERM and an O_EXCL receipt after every exact
group is absent.  It never discovers signal targets with pgrep/pkill.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any

import yaml


QUEUE = Path("configs/phase1_rolling_timing_supercombo_20260716.yaml")
CONFIRM = "SIM_ONLY_STOP_BOUND_ROLLING_TASK_REVISION_CUTOVER"
CONTROL_ROOT = Path(
    "/workspace/codexschema/phase1_rolling_timing_supercombo_20260716/"
    "control/task_revision_cutover_stop_v1"
)
EXPECTED_COUNTS = {
    "pod1": {"registered": 12, "live": 11, "absent": 1},
    "pod2": {"registered": 12, "live": 11, "absent": 1},
}


class StopContractError(RuntimeError):
    """The exact-stop transaction cannot safely proceed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise StopContractError(f"not a non-empty regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    outside = os.lstat(path)
    signature = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if signature(before) != signature(after) or signature(after) != signature(outside):
        raise StopContractError(f"file changed during stable read: {path}")
    raw = b"".join(chunks)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise StopContractError(f"JSON root is not a mapping: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _stable_file(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise StopContractError(f"not a non-empty regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    outside = os.lstat(path)
    signature = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if signature(before) != signature(after) or signature(after) != signature(outside):
        raise StopContractError(f"file changed during stable hash: {path}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def _process_identity(pid: int) -> dict[str, Any] | None:
    root = Path(f"/proc/{pid}")
    if not root.exists():
        return None
    raw = (root / "stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "pgid": os.getpgid(pid),
        "starttime_ticks": int(fields[19]),
        "cwd": os.readlink(root / "cwd"),
        "argv": [
            item.decode("utf-8", "replace")
            for item in (root / "cmdline").read_bytes().split(b"\0")
            if item
        ],
    }


def _group_members(pgid: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[2]) == pgid:
                rows.append(
                    {"pid": int(entry.name), "starttime_ticks": int(fields[19])}
                )
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return sorted(rows, key=lambda row: row["pid"])


def _validate_envelope(
    value: dict[str, Any], *, label: str, keys: set[str], schema: int
) -> dict[str, Any]:
    if set(value) != keys or value.get("schema_version") != schema:
        raise StopContractError(f"{label} envelope/schema changed")
    content = value.get("content")
    if not isinstance(content, dict) or _canonical_sha(content) != value.get("content_sha256"):
        raise StopContractError(f"{label} canonical digest mismatch")
    return content


def _latest_checkpoint(rsl_dir: Path, *, required: bool) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in rsl_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file() and time.time() - path.stat().st_mtime > 5.0:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        if required:
            raise StopContractError(f"no stable checkpoint in live run {rsl_dir}")
        return None
    iteration, path = max(candidates)
    return {"iteration": iteration, **_stable_file(path)}


def _audit_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pod = payload["pod"]
    source = Path(payload["source_checkout"])
    source_head = payload["source_commit"]
    expected_cwd = str(source / "hope_training/whole_body_tracking")
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if actual_head != source_head or dirty:
        raise StopContractError("training source is not clean at the bound commit")

    rows: list[dict[str, Any]] = []
    for job in payload["jobs"]:
        root = Path(job["run_dir"])
        binding_path = root / "run_binding.json"
        claim_path = root / "queue_claim.json"
        binding, binding_file_sha = _stable_json(binding_path)
        claim, claim_file_sha = _stable_json(claim_path)
        binding_content = _validate_envelope(
            binding,
            label=str(binding_path),
            keys={"schema_version", "content", "content_sha256"},
            schema=1,
        )
        claim_content = _validate_envelope(
            claim,
            label=str(claim_path),
            keys={"schema_version", "content", "content_sha256", "training_argv"},
            schema=2,
        )
        exact_pairs = {
            "job_id": job["id"],
            "pod": pod,
            "run_dir": job["run_dir"],
        }
        for key, expected in exact_pairs.items():
            if binding_content.get(key) != expected or claim_content.get(key) != expected:
                raise StopContractError(f"{job['id']} {key} binding mismatch")
        if binding_content.get("claim_content_sha256") != claim["content_sha256"]:
            raise StopContractError(f"{job['id']} binding does not name exact claim")
        source_binding = binding_content.get("source", {})
        if source_binding.get("checkout") != str(source) or source_binding.get("commit") != source_head:
            raise StopContractError(f"{job['id']} source binding mismatch")
        if binding_content.get("source_state_at_binding") != {
            "clean": True,
            "head": source_head,
        }:
            raise StopContractError(f"{job['id']} source state at binding changed")

        process = binding_content.get("process")
        if not isinstance(process, dict):
            raise StopContractError(f"{job['id']} process binding missing")
        pid = process.get("pid")
        if (
            type(pid) is not int
            or pid < 1
            or process.get("pgid") != pid
            or type(process.get("starttime_ticks")) is not int
            or not isinstance(process.get("argv"), list)
        ):
            raise StopContractError(f"{job['id']} process identity malformed")
        observed = _process_identity(pid)
        live = observed is not None
        if live and (
            observed["pgid"] != pid
            or observed["starttime_ticks"] != process["starttime_ticks"]
            or observed["cwd"] != expected_cwd
            or observed["argv"] != process["argv"]
        ):
            raise StopContractError(f"{job['id']} live process identity changed")
        rsl_dir = Path(binding_content["rsl_log_dir"])
        rows.append(
            {
                "job_id": job["id"],
                "live": live,
                "process": process,
                "observed": observed,
                "group_members": _group_members(pid),
                "binding": {
                    "path": str(binding_path),
                    "file_sha256": binding_file_sha,
                    "content_sha256": binding["content_sha256"],
                },
                "claim": {
                    "path": str(claim_path),
                    "file_sha256": claim_file_sha,
                    "content_sha256": claim["content_sha256"],
                },
                "checkpoint": _latest_checkpoint(rsl_dir, required=live),
                "hard_contract": (
                    _stable_file(rsl_dir / "params" / "training_contract.json")
                    if (rsl_dir / "params" / "training_contract.json").is_file()
                    else None
                ),
            }
        )
    counts = EXPECTED_COUNTS[pod]
    live_count = sum(row["live"] for row in rows)
    if (
        len(rows) != counts["registered"]
        or live_count != counts["live"]
        or len(rows) - live_count != counts["absent"]
    ):
        raise StopContractError(
            f"{pod} state changed: registered={len(rows)} live={live_count}"
        )
    return rows


def _write_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def _remote_worker(payload: dict[str, Any], *, execute: bool) -> dict[str, Any]:
    pod = payload["pod"]
    rows = _audit_rows(payload)
    live = [row for row in rows if row["live"]]
    if not execute:
        return {
            "pod": pod,
            "status": "dry_run_passed",
            "registered": len(rows),
            "live": len(live),
            "latest_iterations": {
                row["job_id"]: row["checkpoint"]["iteration"]
                if row["checkpoint"] is not None
                else None
                for row in rows
            },
        }

    # Complete second identity pass before the first signal.
    expected_cwd = str(Path(payload["source_checkout"]) / "hope_training/whole_body_tracking")
    for row in live:
        observed = _process_identity(row["process"]["pid"])
        if observed is None or (
            observed["pgid"] != row["process"]["pgid"]
            or observed["starttime_ticks"] != row["process"]["starttime_ticks"]
            or observed["cwd"] != expected_cwd
            or observed["argv"] != row["process"]["argv"]
        ):
            raise StopContractError(
                f"{row['job_id']} identity changed before signal; zero signals sent"
            )

    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    intent_path = CONTROL_ROOT / f"{pod}.intent.json"
    receipt_path = CONTROL_ROOT / f"{pod}.receipt.json"
    if intent_path.exists() or receipt_path.exists():
        raise StopContractError(f"{pod} stop transaction already exists")
    intent = {
        "schema_version": 1,
        "purpose": "stop_instrumentation_incomplete_pool_before_task_revision_timing_exam_cutover",
        "pod": pod,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "checkout": payload["source_checkout"],
            "commit": payload["source_commit"],
        },
        "rows": rows,
    }
    intent_raw = json.dumps(
        intent,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _write_exclusive(intent_path, intent_raw)

    signals: list[dict[str, Any]] = []
    for row in live:
        pgid = row["process"]["pgid"]
        try:
            os.killpg(pgid, signal.SIGTERM)
            signals.append({"job_id": row["job_id"], "pgid": pgid, "signal": "SIGTERM"})
        except ProcessLookupError:
            signals.append(
                {
                    "job_id": row["job_id"],
                    "pgid": pgid,
                    "signal": "natural_exit_before_TERM",
                }
            )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and any(
        _group_members(row["process"]["pgid"]) for row in live
    ):
        time.sleep(0.5)
    for row in live:
        pgid = row["process"]["pgid"]
        remaining = _group_members(pgid)
        if remaining:
            os.killpg(pgid, signal.SIGKILL)
            signals.append(
                {
                    "job_id": row["job_id"],
                    "pgid": pgid,
                    "signal": "SIGKILL_after_20s",
                    "members_before": remaining,
                }
            )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(
        _group_members(row["process"]["pgid"]) for row in live
    ):
        time.sleep(0.5)
    remaining = {
        row["job_id"]: _group_members(row["process"]["pgid"])
        for row in live
        if _group_members(row["process"]["pgid"])
    }
    if remaining:
        raise StopContractError(f"exact process groups remain after stop: {remaining}")

    nvml = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        timeout=15,
    ).stdout.splitlines()
    nvml_pids = {int(item.strip()) for item in nvml if item.strip().isdigit()}
    stopped_pids = {row["process"]["pid"] for row in live}
    if stopped_pids & nvml_pids:
        raise StopContractError("stopped trainer PID remains in NVML")
    all_absent = all(_process_identity(row["process"]["pid"]) is None for row in rows)
    if not all_absent:
        raise StopContractError("a registered trainer leader remains after stop")

    receipt = {
        "schema_version": 1,
        "purpose": intent["purpose"],
        "pod": pod,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "intent": {
            "path": str(intent_path),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
        },
        "signals": signals,
        "postcondition": {
            "all_registered_leaders_absent": True,
            "all_live_exact_groups_absent": True,
            "stopped_pids_absent_from_nvml": True,
            "remaining_nvml_pids": sorted(nvml_pids),
        },
        "rows": rows,
    }
    receipt_raw = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _write_exclusive(receipt_path, receipt_raw)
    return {
        "pod": pod,
        "status": "stopped",
        "live_stopped": len(live),
        "already_absent": len(rows) - len(live),
        "sigkill_count": sum(
            item["signal"].startswith("SIGKILL") for item in signals
        ),
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "latest_iterations": {
            row["job_id"]: row["checkpoint"]["iteration"]
            if row["checkpoint"] is not None
            else None
            for row in rows
        },
    }


def _finalize_existing(payload: dict[str, Any]) -> dict[str, Any]:
    """Publish a receipt after a signalled worker exited before final publish.

    This recovery path is read-only until the final O_EXCL receipt.  It never
    sends another signal and therefore cannot replay a partially completed stop.
    """

    pod = payload["pod"]
    intent_path = CONTROL_ROOT / f"{pod}.intent.json"
    receipt_path = CONTROL_ROOT / f"{pod}.receipt.json"
    if receipt_path.exists():
        raise StopContractError(f"{pod} receipt already exists")
    intent, intent_file_sha = _stable_json(intent_path)
    expected_keys = {
        "schema_version",
        "purpose",
        "pod",
        "created_utc",
        "source",
        "rows",
    }
    if (
        set(intent) != expected_keys
        or intent["schema_version"] != 1
        or intent["pod"] != pod
        or intent["purpose"]
        != "stop_instrumentation_incomplete_pool_before_task_revision_timing_exam_cutover"
        or intent["source"]
        != {
            "checkout": payload["source_checkout"],
            "commit": payload["source_commit"],
        }
        or not isinstance(intent["rows"], list)
        or len(intent["rows"]) != EXPECTED_COUNTS[pod]["registered"]
    ):
        raise StopContractError(f"{pod} existing intent contract changed")
    rows = intent["rows"]
    remaining_groups: dict[str, Any] = {}
    remaining_leaders: dict[str, Any] = {}
    stopped_pids: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("process"), dict):
            raise StopContractError(f"{pod} intent row/process malformed")
        process = row["process"]
        pid = process.get("pid")
        pgid = process.get("pgid")
        if type(pid) is not int or pid < 1 or pgid != pid:
            raise StopContractError(f"{pod} intent process identity malformed")
        stopped_pids.add(pid)
        observed = _process_identity(pid)
        if observed is not None:
            remaining_leaders[row.get("job_id", str(pid))] = observed
        members = _group_members(pgid)
        if members:
            remaining_groups[row.get("job_id", str(pid))] = members
    if remaining_leaders or remaining_groups:
        raise StopContractError(
            f"{pod} cannot finalize while processes remain: "
            f"leaders={remaining_leaders} groups={remaining_groups}"
        )
    nvml = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        timeout=15,
    ).stdout.splitlines()
    nvml_pids = {int(item.strip()) for item in nvml if item.strip().isdigit()}
    if stopped_pids & nvml_pids:
        raise StopContractError(f"{pod} stopped trainer PID remains in NVML")
    receipt = {
        "schema_version": 1,
        "purpose": intent["purpose"],
        "pod": pod,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "intent": {
            "path": str(intent_path),
            "file_sha256": intent_file_sha,
            "content_sha256": _canonical_sha(intent),
        },
        "recovery_finalization": {
            "signals_sent_by_prior_worker": True,
            "signal_journal_available": False,
            "reason": (
                "worker_v1 persisted intent before signals but failed closed while "
                "waiting for final process reaping; this finalizer sent no signal"
            ),
        },
        "postcondition": {
            "all_registered_leaders_absent": True,
            "all_exact_groups_absent": True,
            "stopped_pids_absent_from_nvml": True,
            "remaining_nvml_pids": sorted(nvml_pids),
        },
        "rows": rows,
    }
    raw = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _write_exclusive(receipt_path, raw)
    return {
        "pod": pod,
        "status": "recovery_receipt_published_no_signal",
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "registered_absent": len(rows),
        "latest_iterations": {
            row["job_id"]: row["checkpoint"]["iteration"]
            if row.get("checkpoint") is not None
            else None
            for row in rows
        },
    }


def _payload(queue: dict[str, Any], pod: str) -> dict[str, Any]:
    jobs = [
        {"id": job["id"], "run_dir": job["run_dir"]}
        for job in queue["jobs"]
        if job["resource"]["required_slot"].split("/", 1)[0] == pod
    ]
    return {
        "pod": pod,
        "source_checkout": queue["blocking_contract"]["source_checkout"],
        "source_commit": queue["blocking_contract"]["source_commit"],
        "jobs": jobs,
    }


def _run_remote(
    queue: dict[str, Any], pod: str, *, execute: bool, finalize_existing: bool
) -> dict[str, Any]:
    payload = base64.b64encode(
        json.dumps(_payload(queue, pod), separators=(",", ":"), sort_keys=True).encode()
    ).decode("ascii")
    pod_cfg = queue["pods"][pod]
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=4",
        "-i",
        str(Path(queue["ssh"]["key"]).expanduser()),
        "-p",
        str(pod_cfg["port"]),
        f"root@{pod_cfg['host']}",
        "python3",
        "-",
        "remote-worker",
        payload,
        "finalize" if finalize_existing else ("execute" if execute else "dry-run"),
        CONFIRM,
    ]
    completed = subprocess.run(
        argv,
        input=Path(__file__).read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    if completed.returncode != 0:
        raise StopContractError(
            f"{pod} worker failed rc={completed.returncode}; "
            f"stdout={completed.stdout.decode(errors='replace')!r}; "
            f"stderr={completed.stderr.decode(errors='replace')!r}"
        )
    return json.loads(completed.stdout)


def _load_queue(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StopContractError("queue root is not a mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "remote-worker":
        if len(args) != 4 or args[3] != CONFIRM:
            raise StopContractError("remote worker arguments/confirmation changed")
        payload = json.loads(base64.b64decode(args[1], validate=True))
        if args[2] == "finalize":
            result = _finalize_existing(payload)
        elif args[2] in {"execute", "dry-run"}:
            result = _remote_worker(payload, execute=args[2] == "execute")
        else:
            raise StopContractError("unknown remote worker mode")
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--pod", choices=("pod1", "pod2"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--confirm")
    ns = parser.parse_args(args)
    if ns.execute and ns.finalize_existing:
        parser.error("--execute and --finalize-existing are mutually exclusive")
    if (ns.execute or ns.finalize_existing) and ns.confirm != CONFIRM:
        parser.error(f"mutation/finalization requires --confirm {CONFIRM}")
    result = _run_remote(
        _load_queue(ns.queue),
        ns.pod,
        execute=ns.execute,
        finalize_existing=ns.finalize_existing,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
