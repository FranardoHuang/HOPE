#!/usr/bin/env python3
"""Fail closed on Phase-1 q10 queue governance before a worker is launched.

This validator is deliberately independent of checkpoint availability and of
either simulator.  It checks the paper, not the runtime: q10 remains a screen,
q50 cannot be smuggled through the curve worker, and milestone barriers/order
cannot be weakened by reordering JSON jobs.  With no ``--manifest`` arguments
it also audits the complete checked-in 2026-07-11 scale-out/cadence set, the
deterministically generated scale-out bytes, the concurrent runbook example,
and the default GVHMR intake order.

It never starts or signals a process and contains no real-robot path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")
Q10_SCHEDULE_K = 20
Q10_ATTEMPTS_PER_SIDE = 10
Q50_FOLLOWUP = "same immutable schedule with at least 50 attempts per side"

SCALEOUT_MANIFESTS = {
    "scaleout_causal_pod1": (
        "configs/phase1_checkpoint_curve_scaleout_causal_pod1_20260711.json",
        8,
    ),
    "scaleout_fresh_pod1": (
        "configs/phase1_checkpoint_curve_scaleout_fresh_pod1_20260711.json",
        63,
    ),
    "scaleout_causal_pod2": (
        "configs/phase1_checkpoint_curve_scaleout_causal_pod2_20260711.json",
        8,
    ),
    "scaleout_fresh_pod2": (
        "configs/phase1_checkpoint_curve_scaleout_fresh_pod2_20260711.json",
        63,
    ),
}

CADENCE_MANIFESTS = {
    "cadence_causal_pod1": (
        "configs/phase1_checkpoint_curve_cadence_pod1_20260711.json",
        4,
    ),
    "cadence_fresh_pod1": (
        "configs/phase1_checkpoint_curve_cadence_fresh_pod1_20260711.json",
        8,
    ),
    "cadence_causal_pod2": (
        "configs/phase1_checkpoint_curve_cadence_pod2_20260711.json",
        4,
    ),
    "cadence_fresh_pod2": (
        "configs/phase1_checkpoint_curve_cadence_fresh_pod2_20260711.json",
        7,
    ),
}

EXPECTED_MOTION_ORDER = [
    "franco_forehand_block",
    "franco_backhand_block",
    "franco_forehand_loop",
    "franco_backhand_loop_a",
    "franco_backhand_loop_b",
    "franco_backhand_loop_c",
    "v6_forehand_block",
    "v6_backhand_block",
    "v7_forehand_block",
    "v7_backhand_block",
]


class ContractError(ValueError):
    """A queue paper can bypass the reviewed screen/decision discipline."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def checkpoint_iteration(job: dict[str, Any]) -> int:
    checkpoint = Path(str(job.get("checkpoint", "")))
    match = CHECKPOINT.fullmatch(checkpoint.name)
    if not match:
        raise ContractError(
            f"job {job.get('id')}: checkpoint must be model_<iteration>.pt"
        )
    return int(match.group(1))


def validate_manifest(
    manifest: dict[str, Any],
    *,
    label: str,
    require_readiness_barrier: bool = False,
) -> dict[str, Any]:
    """Validate one worker manifest without touching any referenced artifact."""

    policy = manifest.get("screen_policy")
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != 1:
        raise ContractError(f"{label}: schema_version must be 1")
    if not isinstance(policy, dict):
        raise ContractError(f"{label}: screen_policy is required")
    if policy.get("screen_only") is not True:
        raise ContractError(f"{label}: screen_only must be true")
    if policy.get("stop_or_promote_allowed") is not False:
        raise ContractError(f"{label}: stop_or_promote_allowed must be false")
    if policy.get("schedule_k") != Q10_SCHEDULE_K:
        raise ContractError(
            f"{label}: curve workers are q10-only (schedule_k={Q10_SCHEDULE_K}); "
            "q50 requires its separate immutable paired runner"
        )
    if policy.get("attempts_per_side") != Q10_ATTEMPTS_PER_SIDE:
        raise ContractError(
            f"{label}: q10 attempts_per_side must be {Q10_ATTEMPTS_PER_SIDE}"
        )
    if policy.get("decision_followup") != Q50_FOLLOWUP:
        raise ContractError(f"{label}: q50 decision follow-up text is missing or changed")
    if not isinstance(jobs, list) or not jobs:
        raise ContractError(
            f"{label}: a curve-worker manifest requires non-empty q10 jobs; "
            "an inactive/empty q50 template is not executable"
        )

    ids: set[str] = set()
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for index, raw in enumerate(jobs):
        if not isinstance(raw, dict):
            raise ContractError(f"{label}: jobs[{index}] must be an object")
        job_id = raw.get("id")
        if (
            not isinstance(job_id, str)
            or not SAFE_ID.fullmatch(job_id)
            or job_id in ids
        ):
            raise ContractError(f"{label}: unsafe or duplicate job id {job_id!r}")
        ids.add(job_id)

        iteration = checkpoint_iteration(raw)
        if not job_id.endswith(f"_{iteration}_clean_q10"):
            raise ContractError(f"{label}/{job_id}: id/checkpoint milestone mismatch")
        run_dir = Path(str(raw.get("run_dir", "")))
        checkpoint = Path(str(raw.get("checkpoint", "")))
        if (
            not run_dir.is_absolute()
            or not checkpoint.is_absolute()
            or checkpoint.parent != run_dir
        ):
            raise ContractError(
                f"{label}/{job_id}: run/checkpoint must be absolute and directly paired"
            )
        gpu = raw.get("gpu")
        if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0:
            raise ContractError(f"{label}/{job_id}: gpu must be a non-negative integer")
        if raw.get("screen_only") is not True:
            raise ContractError(f"{label}/{job_id}: job screen_only must be true")

        exact = raw.get("expected_evaluation_contract_exact")
        formal = raw.get("formal_target")
        role = raw.get("evaluation_role")
        if not isinstance(exact, bool) or not isinstance(formal, bool):
            raise ContractError(f"{label}/{job_id}: exact/formal flags must be booleans")
        if formal and not exact:
            raise ContractError(f"{label}/{job_id}: an inexact job cannot be formal")
        if not isinstance(role, str) or not role:
            raise ContractError(f"{label}/{job_id}: evaluation_role is required")
        expected_args = ["--schedule-k", str(Q10_SCHEDULE_K)]
        if not exact:
            expected_args += ["--exam-extra", "--allow-inexact-contract"]
        if raw.get("extra_args") != expected_args:
            raise ContractError(
                f"{label}/{job_id}: judge args must be exactly {expected_args!r}"
            )
        if "seed" in policy and raw.get("seed") != policy["seed"]:
            raise ContractError(f"{label}/{job_id}: exam seed contradicts screen_policy")
        if "noise_scales" in policy and raw.get("noise_scales") != policy["noise_scales"]:
            raise ContractError(
                f"{label}/{job_id}: noise_scales contradict screen_policy"
            )

        if current is None or current["iteration"] != iteration:
            if current is not None and iteration <= current["iteration"]:
                raise ContractError(
                    f"{label}: milestone groups must be contiguous and strictly increasing"
                )
            current = {
                "iteration": iteration,
                "jobs": [],
                "runs": set(),
                "barriers": set(),
            }
            groups.append(current)
        current["jobs"].append(job_id)
        current["runs"].add(str(run_dir))
        current["barriers"].add(raw.get("barrier_id"))

    expected_runs = groups[0]["runs"]
    readiness = manifest.get("checkpoint_readiness_barrier")
    if require_readiness_barrier and readiness is None:
        raise ContractError(f"{label}: reviewed scale-out queue requires readiness metadata")
    if readiness is not None and (
        not isinstance(readiness, dict)
        or readiness.get("ordering") != "milestone_major"
    ):
        raise ContractError(f"{label}: readiness ordering must be milestone_major")

    seen_barriers: set[str] = set()
    group_summaries: list[dict[str, Any]] = []
    for group in groups:
        if group["runs"] != expected_runs:
            raise ContractError(f"{label}: run membership changes across milestones")
        barriers = group["barriers"]
        if None in barriers and len(barriers) > 1:
            raise ContractError(f"{label}: barrier covers only part of a milestone")
        if None not in barriers and len(barriers) != 1:
            raise ContractError(f"{label}: milestone has contradictory barrier ids")
        barrier = None if barriers == {None} else next(iter(barriers))
        if barrier is not None:
            if not isinstance(barrier, str) or not SAFE_ID.fullmatch(barrier):
                raise ContractError(f"{label}: unsafe barrier id {barrier!r}")
            if not barrier.endswith(f"_{group['iteration']}"):
                raise ContractError(
                    f"{label}: barrier {barrier!r} disagrees with milestone "
                    f"{group['iteration']}"
                )
            if barrier in seen_barriers:
                raise ContractError(f"{label}: barrier is discontinuous/reused: {barrier}")
            seen_barriers.add(barrier)
        if readiness is not None and barrier is None:
            raise ContractError(f"{label}: readiness milestone lacks a barrier id")
        group_summaries.append(
            {
                "iteration": group["iteration"],
                "job_count": len(group["jobs"]),
                "barrier_id": barrier,
            }
        )

    return {
        "job_count": len(jobs),
        "run_count": len(expected_runs),
        "milestone_groups": group_summaries,
    }


def _validate_deterministic_scaleout(repo_root: Path) -> None:
    generator_path = (
        repo_root
        / "hope_training/whole_body_tracking/scripts/"
        "generate_phase1_scaleout_curve_manifests.py"
    )
    result = subprocess.run(
        [sys.executable, str(generator_path), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ContractError(f"deterministic scale-out generation check failed: {detail}")


def _validate_concurrent_runbook(repo_root: Path) -> None:
    path = repo_root / "docs/operations/run_training.md"
    text = path.read_text(encoding="utf-8")
    required = (
        "for queue in causal fresh; do",
        "nohup setsid python3",
        ">\"$state/worker.log\" 2>&1 </dev/null &",
        "Do not add `wait` to that loop",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ContractError(
            "run_training dual-queue example is no longer explicitly concurrent; "
            f"missing={missing}"
        )


def _validate_motion_default_order(repo_root: Path) -> dict[str, Any]:
    intake_path = repo_root / "configs/motion_video_intake_20260711.json"
    result_path = repo_root / "configs/motion_video_gvhmr_results_20260711.json"
    intake = load_json(intake_path)
    order = intake.get("processing_order")
    if order != EXPECTED_MOTION_ORDER:
        raise ContractError(
            "GVHMR default processing_order changed; default must start with the "
            "documented Franco forehand-block pilot and retain the frozen ten-asset order"
        )
    assets = intake.get("assets")
    if not isinstance(assets, list):
        raise ContractError("motion intake assets must be a list")
    ids = [row.get("id") for row in assets if isinstance(row, dict)]
    if len(ids) != len(assets) or len(set(ids)) != len(ids) or set(ids) != set(order):
        raise ContractError("motion intake processing_order must be an exact asset-id permutation")
    result = load_json(result_path)
    rows = result.get("results")
    result_order = [row.get("asset_id") for row in rows] if isinstance(rows, list) else None
    if result_order != order:
        raise ContractError("GVHMR result order no longer reproduces the frozen default queue")
    return {"asset_count": len(order), "first_asset_id": order[0]}


def validate_repository(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _validate_deterministic_scaleout(repo_root)

    scaleout: dict[str, Any] = {}
    scaleout_ids: set[str] = set()
    for label, (relative, expected_jobs) in SCALEOUT_MANIFESTS.items():
        manifest = load_json(repo_root / relative)
        summary = validate_manifest(
            manifest, label=label, require_readiness_barrier=True
        )
        if summary["job_count"] != expected_jobs:
            raise ContractError(
                f"{label}: expected {expected_jobs} jobs, got {summary['job_count']}"
            )
        ids = {job["id"] for job in manifest["jobs"]}
        overlap = scaleout_ids & ids
        if overlap:
            raise ContractError(f"scale-out job ids overlap across queues: {sorted(overlap)}")
        scaleout_ids |= ids
        queue, pod = label[len("scaleout_") :].rsplit("_", 1)
        if manifest.get("queue") != queue or manifest.get("pod") != pod:
            raise ContractError(f"{label}: filename/queue/pod metadata disagree")
        scaleout[label] = summary
    if len(scaleout_ids) != 142:
        raise ContractError(f"scale-out paper must contain exactly 142 unique jobs, got {len(scaleout_ids)}")

    for queue in ("causal", "fresh"):
        left = load_json(repo_root / SCALEOUT_MANIFESTS[f"scaleout_{queue}_pod1"][0])
        right = load_json(repo_root / SCALEOUT_MANIFESTS[f"scaleout_{queue}_pod2"][0])
        expected_left = (
            f"configs/phase1_checkpoint_curve_scaleout_{queue}_pod2_20260711.json"
        )
        expected_right = (
            f"configs/phase1_checkpoint_curve_scaleout_{queue}_pod1_20260711.json"
        )
        if left["checkpoint_readiness_barrier"].get("companion_cross_pod_manifest") != expected_left:
            raise ContractError(f"scaleout_{queue}_pod1: wrong cross-Pod companion")
        if right["checkpoint_readiness_barrier"].get("companion_cross_pod_manifest") != expected_right:
            raise ContractError(f"scaleout_{queue}_pod2: wrong cross-Pod companion")
        left_iterations = [group["iteration"] for group in scaleout[f"scaleout_{queue}_pod1"]["milestone_groups"]]
        right_iterations = [group["iteration"] for group in scaleout[f"scaleout_{queue}_pod2"]["milestone_groups"]]
        if left_iterations != right_iterations:
            raise ContractError(f"scaleout {queue}: cross-Pod milestone order differs")

    cadence: dict[str, Any] = {}
    cadence_ids: set[str] = set()
    for label, (relative, expected_jobs) in CADENCE_MANIFESTS.items():
        manifest = load_json(repo_root / relative)
        summary = validate_manifest(manifest, label=label)
        if summary["job_count"] != expected_jobs:
            raise ContractError(
                f"{label}: expected {expected_jobs} active-tail jobs, got {summary['job_count']}"
            )
        ids = {job["id"] for job in manifest["jobs"]}
        overlap = cadence_ids & ids
        if overlap:
            raise ContractError(f"cadence job ids overlap across queues: {sorted(overlap)}")
        cadence_ids |= ids
        cadence[label] = summary
    if len(cadence_ids) != 23:
        raise ContractError(
            f"cadence manifests must contain 23 active-tail jobs, got {len(cadence_ids)}"
        )
    pod2_fresh = load_json(
        repo_root / CADENCE_MANIFESTS["cadence_fresh_pod2"][0]
    )
    if "model_4000 were handled by earlier immutable workers" not in str(
        pod2_fresh.get("prior_milestones", "")
    ):
        raise ContractError(
            "cadence 24-slot ledger is incomplete: the one prior-completed Pod2 model_4000 "
            "slot is not declared"
        )

    _validate_concurrent_runbook(repo_root)
    motion = _validate_motion_default_order(repo_root)
    return {
        "status": "pass",
        "scaleout": {
            "manifest_jobs": len(scaleout_ids),
            "expected_jobs": 142,
            "queues": scaleout,
        },
        "cadence": {
            "planned_slots": 24,
            "manifest_active_tail_jobs": len(cadence_ids),
            "prior_completed_slots": 1,
            "queues": cadence,
        },
        "q50": {
            "curve_worker_jobs_allowed": False,
            "required_followup": Q50_FOLLOWUP,
        },
        "runbook_dual_queues_concurrent": True,
        "motion_default_order": motion,
        "real_robot_commands": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help="validate one manifest only; repeatable (default: audit checked-in repository set)",
    )
    parser.add_argument(
        "--require-readiness-barrier",
        action="store_true",
        help="require milestone-major readiness metadata on each --manifest",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.manifest:
            results = {}
            for path in args.manifest:
                resolved = path.resolve()
                results[str(resolved)] = validate_manifest(
                    load_json(resolved),
                    label=str(resolved),
                    require_readiness_barrier=args.require_readiness_barrier,
                )
            payload: dict[str, Any] = {"status": "pass", "manifests": results}
        else:
            payload = validate_repository(args.repo_root)
    except ContractError as exc:
        print(f"[phase1-queue-governance][FATAL] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
