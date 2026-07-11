#!/usr/bin/env python3
"""Generate milestone-major q10 manifests for the 18 Phase-1 scale-out arms.

The causal seed-2 pairs and the fresh factorial are deliberately emitted as
separate queues.  A slow causal terminal checkpoint therefore cannot prevent a
ready fresh milestone from being judged.  Within a queue, jobs are ordered by
milestone first: ``phase1_checkpoint_curve_worker.py --wait-for-checkpoints``
cannot reach milestone N+1 until every new arm on that Pod has presented the
checkpoint for milestone N and its export has entered the CPU phase.

The generated q10 exams are direction screens only.  Nothing in these
manifests authorizes stopping or promotion; those decisions require the
separate immutable q50 follow-up described by the checked-in binding spec.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINDINGS = REPO_ROOT / "configs" / "phase1_scaleout_curve_bindings_20260711.json"
CELL_ORDER = {"SZ": 0, "SP": 1, "LZ": 2, "LP": 3}
PAIRING_ORDER = {"legacy_signed_vs_A": 0, "shared_plus_y": 1}
OUTPUT_NAME = "phase1_checkpoint_curve_scaleout_{queue}_{pod}_20260711.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_int_list(value: Any, *, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{name} must be a non-empty increasing list of unique integers")
    return value


def _validate_design(
    matrix: dict[str, Any], bindings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if matrix.get("schema_version") != 1 or not isinstance(matrix.get("arms"), list):
        raise ValueError("scale-out matrix must be schema_version=1 with an arms list")
    if bindings.get("schema_version") != 1:
        raise ValueError("curve bindings must use schema_version=1")

    raw_run_dirs = bindings.get("run_dir_basenames")
    if not isinstance(raw_run_dirs, dict) or not raw_run_dirs:
        raise ValueError("run_dir_basenames must be a non-empty object")
    run_dirs: dict[str, str] = {}
    for run_name, basename in raw_run_dirs.items():
        if not isinstance(run_name, str) or not isinstance(basename, str):
            raise ValueError("run_dir_basenames keys and values must be strings")
        if Path(basename).name != basename or not basename.endswith("_" + run_name):
            raise ValueError(f"run basename does not bind unambiguously to {run_name}: {basename}")
        run_dirs[run_name] = basename

    arms = matrix["arms"]
    by_name: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("run_name"), str):
            raise ValueError("every matrix arm must have a string run_name")
        if arm["run_name"] in by_name:
            raise ValueError(f"duplicate matrix run_name: {arm['run_name']}")
        by_name[arm["run_name"]] = arm

    excluded_raw = bindings.get("excluded_existing_run_names")
    if not isinstance(excluded_raw, list) or not all(isinstance(v, str) for v in excluded_raw):
        raise ValueError("excluded_existing_run_names must be a string list")
    excluded = set(excluded_raw)
    selected = set(run_dirs)
    all_names = set(by_name)
    if selected & excluded:
        raise ValueError("selected and existing arm sets overlap")
    if selected | excluded != all_names:
        missing = sorted(all_names - selected - excluded)
        extra = sorted((selected | excluded) - all_names)
        raise ValueError(f"bindings do not partition the matrix; missing={missing}, extra={extra}")
    if len(selected) != 18:
        raise ValueError(f"expected exactly 18 new scale-out arms, got {len(selected)}")

    chosen = [by_name[name] for name in selected]
    causal = [arm for arm in chosen if arm.get("kind") == "continuation"]
    fresh = [arm for arm in chosen if arm.get("kind") == "fresh"]
    if len(causal) != 4 or len(fresh) != 14:
        raise ValueError(f"expected 4 causal and 14 fresh arms, got {len(causal)} and {len(fresh)}")
    if any(arm.get("seed") != 2 for arm in causal):
        raise ValueError("every scale-out causal arm must be continuation seed 2")

    for pod in ("pod1", "pod2"):
        pod_causal = [arm for arm in causal if arm.get("pod") == pod]
        if len(pod_causal) != 2 or {arm.get("pairing") for arm in pod_causal} != set(PAIRING_ORDER):
            raise ValueError(f"{pod} must contain one old/S1 causal seed-2 pair")
        pod_fresh = [arm for arm in fresh if arm.get("pod") == pod]
        if len(pod_fresh) != 7:
            raise ValueError(f"{pod} must contain seven new fresh arms")

    expected_fresh = {
        ("SZ", 3), ("SZ", 4),
        ("SP", 1), ("SP", 2), ("SP", 3), ("SP", 4),
        ("LZ", 1), ("LZ", 2), ("LZ", 3), ("LZ", 4),
        ("LP", 1), ("LP", 2), ("LP", 3), ("LP", 4),
    }
    actual_fresh = {(arm.get("cell"), arm.get("seed")) for arm in fresh}
    if actual_fresh != expected_fresh:
        raise ValueError(f"unexpected new fresh factorial cells/seeds: {sorted(actual_fresh)}")

    return chosen, run_dirs


def _arm_prefix(arm: dict[str, Any]) -> str:
    if arm["kind"] == "fresh":
        return f"fresh_{arm['cell']}_seed{arm['seed']}"
    name = arm["run_name"]
    if name.startswith("phase1_"):
        name = name[len("phase1_") :]
    return name.replace("_pairing_seed2", "_seed2")


def _evaluation_metadata(arm: dict[str, Any]) -> tuple[str, bool, bool]:
    if arm["kind"] == "continuation":
        return "causal_continuation_diagnostic", False, False
    cell = arm["cell"]
    if cell == "SZ":
        return "formal_target", True, True
    if cell == "SP":
        return "plant_diagnostic_non_target", True, False
    if cell in {"LZ", "LP"}:
        return "legacy_pairing_diagnostic_inexact", False, False
    raise ValueError(f"unknown fresh cell: {cell}")


def _job(
    arm: dict[str, Any],
    *,
    milestone: int,
    queue: str,
    run_dir: Path,
    exam: dict[str, Any],
) -> dict[str, Any]:
    role, expected_exact, formal_target = _evaluation_metadata(arm)
    prefix = _arm_prefix(arm)
    result: dict[str, Any] = {
        "id": f"{prefix}_{milestone}_clean_q10",
        "barrier_id": f"{queue}_{milestone}",
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / f"model_{milestone}.pt"),
        "gpu": arm["gpu"],
        # This is the immutable exam seed, not the PPO training seed.
        "seed": exam["seed"],
        "noise_scales": exam["noise_scales"],
        "extra_args": ["--schedule-k", str(exam["schedule_k"])],
        "training_seed": arm["seed"],
        "training_kind": arm["kind"],
        "training_family": arm["family"],
        "face_command_pairing": arm["pairing"],
        "zero_joint_friction": arm["zero_joint_friction"],
        "evaluation_role": role,
        "expected_evaluation_contract_exact": expected_exact,
        "formal_target": formal_target,
        "screen_only": True,
    }
    if arm["kind"] == "fresh":
        result["cell"] = arm["cell"]
    return result


def build_manifests(
    matrix: dict[str, Any], bindings: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return output filename -> worker-compatible manifest."""
    chosen, run_basenames = _validate_design(matrix, bindings)
    log_root = Path(bindings.get("training_log_root", ""))
    if not log_root.is_absolute():
        raise ValueError("training_log_root must be absolute")
    exam = bindings.get("exam")
    if not isinstance(exam, dict):
        raise ValueError("exam must be an object")
    if (
        not isinstance(exam.get("seed"), int)
        or not isinstance(exam.get("noise_scales"), str)
        or not isinstance(exam.get("schedule_k"), int)
        or exam.get("schedule_k") != 2 * exam.get("attempts_per_side", -1)
        or exam.get("screen_only") is not True
        or exam.get("stop_or_promote_allowed") is not False
    ):
        raise ValueError("exam must freeze a direction-only balanced q10 schedule")
    milestones = bindings.get("milestones")
    if not isinstance(milestones, dict):
        raise ValueError("milestones must be an object")
    causal_milestones = _require_int_list(milestones.get("continuation"), name="continuation milestones")
    fresh_milestones = _require_int_list(milestones.get("fresh"), name="fresh milestones")

    manifests: dict[str, dict[str, Any]] = {}
    for pod in ("pod1", "pod2"):
        for queue, kind, points in (
            ("causal", "continuation", causal_milestones),
            ("fresh", "fresh", fresh_milestones),
        ):
            queue_arms = [arm for arm in chosen if arm["pod"] == pod and arm["kind"] == kind]
            if kind == "continuation":
                queue_arms.sort(key=lambda arm: PAIRING_ORDER[arm["pairing"]])
            else:
                queue_arms.sort(key=lambda arm: (arm["seed"], CELL_ORDER[arm["cell"]]))

            jobs = [
                _job(
                    arm,
                    milestone=point,
                    queue=queue,
                    run_dir=log_root / run_basenames[arm["run_name"]],
                    exam=exam,
                )
                for point in points
                for arm in queue_arms
            ]
            filename = OUTPUT_NAME.format(queue=queue, pod=pod)
            companion = OUTPUT_NAME.format(queue=queue, pod="pod2" if pod == "pod1" else "pod1")
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "purpose": (
                    f"scale-out {queue} clean q10-per-side growth cadence; "
                    "direction screen only; stop/promotion requires immutable q50"
                ),
                "queue": queue,
                "pod": pod,
                "checkpoint_readiness_barrier": {
                    "ordering": "milestone_major",
                    "scope": "all new arms in this queue on this pod",
                    "companion_cross_pod_manifest": f"configs/{companion}",
                    "semantics": (
                        "worker cannot encounter milestone N+1 until every listed arm at N "
                        "has a stable checkpoint and its export reaches the CPU exam phase"
                    ),
                },
                "screen_policy": exam,
                "judge_script_sha256": bindings["judge_script_sha256"],
                "training_checkout": bindings["training_checkout"],
                "expected_training_commit": bindings["expected_training_commit"],
                "jobs": jobs,
            }
            if queue == "fresh":
                manifest["existing_sz_companion_manifest"] = bindings[
                    "companion_existing_fresh_manifests"
                ][pod]
            manifests[filename] = manifest
    return manifests


def render_manifest(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in manifests differ from deterministic generation",
    )
    args = parser.parse_args()

    bindings_path = args.bindings.resolve()
    bindings = _read_json(bindings_path)
    matrix_ref = bindings.get("scaleout_matrix")
    if not isinstance(matrix_ref, str):
        raise ValueError("scaleout_matrix must be a repository-relative path")
    matrix_path = (REPO_ROOT / matrix_ref).resolve()
    matrix = _read_json(matrix_path)
    manifests = build_manifests(matrix, bindings)

    mismatches: list[str] = []
    for filename, value in manifests.items():
        path = args.output_dir.resolve() / filename
        rendered = render_manifest(value)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            print(path)
    if mismatches:
        parser.error("generated manifests are stale: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
