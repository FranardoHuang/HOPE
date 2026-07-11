#!/usr/bin/env python3
"""Validate the fail-closed Phase-1 T0/T1 event-timing preregistration.

``design-check`` verifies the immutable design, aggregate venue evidence,
engineering schedule specification, exact baseline git blobs, and local Hitter
reference commits without starting a process or touching an external checkout.

``launch-check`` deliberately fails while the preregistration records missing
event scheduler, hard-contract, schedule materializer, Isaac judge, MuJoCo
judge, and self-hit instrumentation.  A future implementation must create a
new preregistration with content-addressed bindings; this file is immutable.

``reproduce-venue`` additionally accepts a restored external ``strikes.json``,
checks its bound SHA-256, reruns the repository A-B-A audit in memory, and
compares only the tracked aggregate.  It never writes raw rows or take ids.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_SOURCE_SHA256 = "6ad3c45959c94b6fdd4033130403c32e0f1b612a138738c12afa43a58f752841"
EXPECTED_TIMING_CELLS = (
    ("E0", 35, 70),
    ("E1", 50, 90),
    ("E2", 50, 110),
    ("E3", 65, 140),
    ("E4", 80, 180),
    ("E5", 100, 220),
)
REQUIRED_IMPLEMENTATION_BINDINGS = (
    "event_scheduler_source",
    "training_hard_contract_schema",
    "schedule_materializer",
    "immutable_question_schedule",
    "immutable_event_screen_schedule",
    "immutable_event_decision_schedule",
    "isaac_continuous_judge",
    "mujoco_continuous_judge",
    "self_hit_instrumentation",
    "fresh_exact_baseline_checkpoint",
    "semantics_correct_plant_contract",
)


class ContractError(ValueError):
    """A T0/T1 design or launch invariant is missing or inconsistent."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON mapping")
    return value


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_repo_binding(binding: Any, *, label: str, root: Path) -> Path:
    if not isinstance(binding, dict):
        raise ContractError(f"{label} must be a mapping")
    repo_path = binding.get("repo_path")
    if not isinstance(repo_path, str) or not repo_path or Path(repo_path).is_absolute():
        raise ContractError(f"{label}.repo_path must be a non-empty relative path")
    require_sha(binding.get("sha256"), f"{label}.sha256")
    if not isinstance(binding.get("bytes"), int) or binding["bytes"] <= 0:
        raise ContractError(f"{label}.bytes must be a positive integer")
    path = (root / repo_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ContractError(f"{label} escapes the repository: {repo_path}") from None
    if not path.is_file():
        raise ContractError(f"{label} is missing: {path}")
    if path.stat().st_size != binding["bytes"]:
        raise ContractError(f"{label} byte mismatch: {path.stat().st_size} != {binding['bytes']}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise ContractError(f"{label} SHA mismatch: {actual} != {binding['sha256']}")
    return path


def git_output(root: Path, *args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=not binary,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise ContractError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout


def verify_git_blob(binding: Any, *, label: str, root: Path) -> None:
    if not isinstance(binding, dict):
        raise ContractError(f"{label} must be a mapping")
    commit = binding.get("commit")
    repo_path = binding.get("repo_path")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise ContractError(f"{label}.commit must be a full lowercase git commit")
    if not isinstance(repo_path, str) or not repo_path or Path(repo_path).is_absolute():
        raise ContractError(f"{label}.repo_path must be a relative path")
    require_sha(binding.get("sha256"), f"{label}.sha256")
    if not isinstance(binding.get("bytes"), int) or binding["bytes"] <= 0:
        raise ContractError(f"{label}.bytes must be a positive integer")
    content = git_output(root, "show", f"{commit}:{repo_path}", binary=True)
    assert isinstance(content, bytes)
    if len(content) != binding["bytes"]:
        raise ContractError(f"{label} git blob byte mismatch")
    actual = sha256_bytes(content)
    if actual != binding["sha256"]:
        raise ContractError(f"{label} git blob SHA mismatch: {actual} != {binding['sha256']}")


def verify_reference_commits(prereg: dict[str, Any], root: Path) -> None:
    refs = prereg.get("local_reference_commits")
    if not isinstance(refs, list) or len(refs) < 5:
        raise ContractError("local_reference_commits must preserve at least five exact references")
    seen: set[str] = set()
    for index, item in enumerate(refs):
        if not isinstance(item, dict):
            raise ContractError(f"local_reference_commits[{index}] must be a mapping")
        commit = item.get("commit")
        if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            raise ContractError(f"local_reference_commits[{index}].commit is invalid")
        if commit in seen:
            raise ContractError(f"duplicate local reference commit {commit}")
        seen.add(commit)
        git_output(root, "cat-file", "-e", f"{commit}^{{commit}}")
        if not isinstance(item.get("role"), str) or not item["role"].strip():
            raise ContractError(f"local_reference_commits[{index}].role is missing")
        if item.get("causal_conclusion_reusable") is not False:
            raise ContractError("legacy Hitter references may provide mechanisms, not causal conclusions")


def validate_venue_report(report: dict[str, Any], *, root: Path) -> None:
    if report.get("schema_version") != 1 or report.get("status") != "completed_prior_audit_aggregate_only":
        raise ContractError("venue aggregate status/schema mismatch")
    source = report.get("source")
    if not isinstance(source, dict) or source.get("sha256") != EXPECTED_SOURCE_SHA256:
        raise ContractError("venue aggregate source SHA mismatch")
    if source.get("tracked_in_git") is not False or source.get("raw_rows_or_take_ids_in_this_report") is not False:
        raise ContractError("venue aggregate must not embed or claim to track raw rows/take ids")
    if source.get("source_rows") != 154 or source.get("bytes") is not None:
        raise ContractError("venue aggregate must bind 154 rows and unknown external byte count")

    tool = report.get("tool")
    if not isinstance(tool, dict):
        raise ContractError("venue aggregate tool binding is missing")
    verify_repo_binding(
        {"repo_path": tool.get("repo_path"), "bytes": tool.get("bytes"), "sha256": tool.get("sha256")},
        label="venue audit tool",
        root=root,
    )
    verify_repo_binding(
        {
            "repo_path": tool.get("test_repo_path"),
            "bytes": tool.get("test_bytes"),
            "sha256": tool.get("test_sha256"),
        },
        label="venue audit tool test",
        root=root,
    )
    if tool.get("command", [])[-3:] != ["--max-leg-s", "2.5", "--summary-only"]:
        raise ContractError("venue audit command must remain max-leg 2.5 s and summary-only")

    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict) or aggregate.get("accepted_samples") != 21:
        raise ContractError("venue aggregate must bind the 21 accepted overlapping windows")
    if aggregate.get("coarse_take_category_counts") != {"gaoqiu": 16, "tantiao": 4, "zhengchang": 1}:
        raise ContractError("venue aggregate coarse category counts changed")
    expected_quantiles = {
        "same_player_interval_s": {"q10": 1.757, "q50": 1.903, "q90": 3.356},
        "self_to_opponent_s": {"q10": 0.833, "q50": 0.951, "q90": 1.493},
        "opponent_to_self_s": {"q10": 0.823, "q50": 0.948, "q90": 1.683},
    }
    for field, expected in expected_quantiles.items():
        if aggregate.get(field) != expected:
            raise ContractError(f"venue aggregate {field} changed")

    biases = report.get("known_biases")
    required_biases = {
        "overlapping_samples": True,
        "samples_independent": False,
        "effective_sample_size": "unknown_but_less_than_21",
        "high_ball_dominated": True,
        "max_leg_filter_right_censors_slow_legs": True,
        "right_censor_threshold_s": 2.5,
        "not_match_play_representative": True,
    }
    if not isinstance(biases, dict) or any(biases.get(key) != value for key, value in required_biases.items()):
        raise ContractError("venue aggregate bias disclosures are incomplete")
    use = report.get("use_policy")
    if not isinstance(use, dict) or use.get("median_1_903_s_is_target") is not False:
        raise ContractError("venue median must never become the T1 target")
    prohibited = " ".join(str(item) for item in use.get("prohibited", []))
    if "1.903" not in prohibited or "sampling weights" not in prohibited:
        raise ContractError("venue use policy must prohibit target/weight fitting")


def validate_schedule_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1 or spec.get("status") != "preregistered_spec_not_materialized":
        raise ContractError("timing schedule spec status/schema mismatch")
    if spec.get("policy_rate_hz") != 50:
        raise ContractError("timing schedule must use the 50 Hz policy clock")
    cells = spec.get("timing_cells")
    actual = tuple(
        (cell.get("cell_id"), cell.get("reveal_ticks_after_prior_strike"), cell.get("next_strike_ticks_after_prior_strike"))
        for cell in cells
    ) if isinstance(cells, list) else ()
    if actual != EXPECTED_TIMING_CELLS:
        raise ContractError("timing cells differ from the frozen engineering grid")
    if any(reveal <= 0 or strike <= reveal for _, reveal, strike in actual):
        raise ContractError("each timing cell must reveal after strike zero and before its next deadline")
    construction = spec.get("cell_construction")
    if not isinstance(construction, dict):
        raise ContractError("cell_construction is missing")
    if construction.get("fit_to_venue_data") is not False or construction.get("venue_quantiles_used_as_targets_or_weights") is not False:
        raise ContractError("engineering timing grid must not be venue-fitted")
    if construction.get("contains_1_903_s_target") is not False:
        raise ContractError("1.903 s must not be frozen as a target")
    seconds_rows = spec.get("timing_cell_seconds_derived_at_50_hz")
    if not isinstance(seconds_rows, list) or len(seconds_rows) != len(actual):
        raise ContractError("derived timing-cell seconds must match every tick cell")
    for (cell_id, reveal_ticks, strike_ticks), derived in zip(actual, seconds_rows):
        if not isinstance(derived, dict) or derived.get("cell_id") != cell_id:
            raise ContractError("derived timing-cell order/id mismatch")
        expected = (reveal_ticks / 50.0, strike_ticks / 50.0, (strike_ticks - reveal_ticks) / 50.0)
        observed = (
            derived.get("reveal_s"),
            derived.get("next_strike_s"),
            derived.get("remaining_notice_s"),
        )
        if observed != expected:
            raise ContractError(f"derived timing seconds disagree with 50 Hz ticks for {cell_id}")
    seconds = [float(cell["next_strike_s"]) for cell in seconds_rows]
    if any(math.isclose(value, 1.903, rel_tol=0.0, abs_tol=1e-12) for value in seconds):
        raise ContractError("timing grid contains the prohibited 1.903 s target")
    if spec.get("transition_cells") != [
        "forehand_to_forehand",
        "forehand_to_backhand",
        "backhand_to_forehand",
        "backhand_to_backhand",
    ]:
        raise ContractError("timing schedule must balance all four side transitions")

    materialization = spec.get("materialization")
    if not isinstance(materialization, dict):
        raise ContractError("schedule materialization contract is missing")
    q10 = materialization.get("screen_q10")
    q50 = materialization.get("decision_q50")
    if not isinstance(q10, dict) or q10.get("questions_per_side") != 10 or q10.get("role") != "screen_only" or q10.get("may_stop_or_promote") is not False:
        raise ContractError("q10 must remain fixed screen-only and unable to stop/promote")
    if not isinstance(q50, dict) or q50.get("questions_per_side") != 50 or q50.get("role") != "decision" or q50.get("may_stop_or_promote") is not True:
        raise ContractError("q50 must remain the frozen decision paper")
    if any(materialization.get(name) is not None for name in ("question_schedule_sha256", "screen_schedule_sha256", "decision_schedule_sha256")):
        raise ContractError("unimplemented schedule artifacts must remain null in this immutable prereg")

    event = spec.get("event_semantics")
    if not isinstance(event, dict):
        raise ContractError("event_semantics is missing")
    required_event = {
        "carry_state": True,
        "reset_robot_or_last_action_on_install": False,
        "reset_history_or_noise_state_on_install": False,
        "mid_sequence_teleport_count_required": 0,
        "deadline_shift_allowed": False,
    }
    if any(event.get(key) != value for key, value in required_event.items()):
        raise ContractError("event carry-state/deadline/reset semantics changed")
    if "natural full-clip wrap" not in str(event.get("T0_install")) or "original" not in str(event.get("T0_install")):
        raise ContractError("T0 must defer to natural wrap without moving the original deadline")
    if "atomically install" not in str(event.get("T1_install")) or "immutable" not in str(event.get("T1_install")):
        raise ContractError("T1 must atomically install the immutable event row")
    if "continue" not in str(event.get("miss_policy")) or "do not delay" not in str(event.get("infeasible_policy")):
        raise ContractError("miss/infeasible rows must stay on the denominator without delay")
    engine = spec.get("immutable_engine_contract")
    if not isinstance(engine, dict) or engine.get("engines") != ["Isaac", "MuJoCo"]:
        raise ContractError("both immutable Isaac and MuJoCo exams are required")
    if engine.get("fresh_exact_only") is not True or engine.get("allow_inexact_contract") is not False:
        raise ContractError("timing axis must remain fresh exact only")
    safety = spec.get("safety")
    if not isinstance(safety, dict) or safety.get("racket_or_handle_self_contact_allowed") is not False:
        raise ContractError("racket/handle self-hit must be prohibited")
    if safety.get("any_self_hit_closes_cell") is not True or safety.get("real_robot_authorized") is not False:
        raise ContractError("self-hit must close a cell and robot execution must remain unauthorized")
    if set(safety.get("protected_robot_regions", [])) != {
        "head", "neck", "torso", "opposite_arm", "waist", "legs"
    }:
        raise ContractError("self-hit contract must cover every protected robot region")
    horizon = spec.get("horizon")
    if not isinstance(horizon, dict) or horizon.get("training_episode_length_s_shared") != 30.0:
        raise ContractError("T0/T1 must share the preregistered 30 s training horizon")
    if horizon.get("decision_sequence_length_opportunities") != 10 or horizon.get("evaluation_sequence_timeout_s") != 60.0:
        raise ContractError("continuous decision horizon must stay 10 opportunities / 60 s")


def validate_prereg(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = repository_root()
    require_sha(expected_sha256, "--expected-prereg-sha256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ContractError(f"prereg SHA mismatch: {actual} != {expected_sha256}")
    prereg = read_json(path, "T0/T1 preregistration")
    if prereg.get("schema_version") != 1 or prereg.get("status") != "preregistered_blocked_on_implementation":
        raise ContractError("T0/T1 prereg must remain schema 1 and blocked on implementation")
    if prereg.get("launch_authorized") is not False or prereg.get("real_robot_authorized") is not False:
        raise ContractError("this preregistration cannot authorize launch or real robot")

    validator = prereg.get("validator")
    validator_path = verify_repo_binding(validator, label="prereg validator", root=root)
    if validator_path != Path(__file__).resolve():
        raise ContractError("prereg validator path does not identify this script")
    evidence = prereg.get("evidence")
    if not isinstance(evidence, dict):
        raise ContractError("evidence bindings are missing")
    venue_path = verify_repo_binding(evidence.get("venue_timing_aggregate"), label="venue aggregate", root=root)
    spec_path = verify_repo_binding(evidence.get("engineering_schedule_spec"), label="schedule spec", root=root)
    venue = read_json(venue_path, "venue aggregate")
    spec = read_json(spec_path, "schedule spec")
    validate_venue_report(venue, root=root)
    validate_schedule_spec(spec)

    baseline = prereg.get("baseline_source")
    if not isinstance(baseline, dict) or baseline.get("training_commit") != "6d93bcb16c422a2f42748c2dc99432559653480b":
        raise ContractError("T0 baseline must bind the live-pool training source commit 6d93bcb")
    files = baseline.get("git_blobs")
    if not isinstance(files, list) or len(files) < 5:
        raise ContractError("baseline source must bind at least five exact git blobs")
    for index, binding in enumerate(files):
        verify_git_blob(binding, label=f"baseline_source.git_blobs[{index}]", root=root)
    verify_reference_commits(prereg, root)

    axis = prereg.get("causal_axis")
    if not isinstance(axis, dict) or axis.get("only_treatment") != "next_task_timing_semantics":
        raise ContractError("T0/T1 only treatment must be next-task timing semantics")
    arms = axis.get("arms")
    if not isinstance(arms, list) or [arm.get("id") for arm in arms] != ["T0", "T1"]:
        raise ContractError("causal axis must contain ordered T0/T1 arms")
    if arms[0].get("install_trigger") != "natural_full_clip_wrap" or arms[1].get("install_trigger") != "post_strike_event_schedule":
        raise ContractError("T0/T1 install triggers changed")
    if arms[0].get("carry_state") is not True or arms[1].get("carry_state") is not True:
        raise ContractError("both T0 and T1 must preserve carry state")
    frozen = axis.get("frozen_non_timing_axes")
    required_frozen = {
        "motion_path_and_time_law": "same_native_v4rg_no_TOPP",
        "plant": "same_semantics_correct_plant",
        "face_pairing": "same_shared_plus_y",
        "reward": "byte_identical_no_new_recovery_income",
        "network": "same_architecture",
        "question_family": "same_fresh_exact_immutable_family",
    }
    if not isinstance(frozen, dict) or any(frozen.get(key) != value for key, value in required_frozen.items()):
        raise ContractError("motion/TOPP/plant/face/reward/network/question axes are not frozen")
    required_runtime = {
        "episode_horizon": "same_30_seconds",
        "wrap_teleport": False,
        "stand_start_prob": 0.25,
        "speed_scale_range": [1.0, 1.0],
        "clip_switch_prob": 0.0,
        "midswing_resample_prob": 0.0,
        "stagger_initial_clock": False,
        "post_swing_start_prob": 0.25,
        "hold_steps_range": [0, 100],
        "T1_event_install_draws_extra_random_hold": False,
    }
    if any(frozen.get(key) != value for key, value in required_runtime.items()):
        raise ContractError("shared episode/reset/hold/clock runtime contract changed")

    selection = prereg.get("lineage_and_checkpoint_selection")
    if not isinstance(selection, dict) or selection.get("fresh_exact_only") is not True:
        raise ContractError("checkpoint selection must remain fresh exact only")
    if selection.get("allow_inexact_contract") is not False or selection.get("timing_outcomes_may_select_baseline") is not False:
        raise ContractError("baseline selection cannot use inexact escape or timing outcomes")
    if selection.get("same_checkpoint_for_T0_T1") is not True or selection.get("independent_seeds_required") != 2:
        raise ContractError("T0/T1 must share checkpoints and require two independent seeds")

    cadence = prereg.get("checkpoint_and_decision_policy")
    if not isinstance(cadence, dict):
        raise ContractError("checkpoint decision policy is missing")
    if cadence.get("q10_role") != "screen_only" or cadence.get("q10_may_stop_or_promote") is not False:
        raise ContractError("q10 must be screen-only")
    if cadence.get("q50_role") != "decision" or cadence.get("q50_required") is not True:
        raise ContractError("q50 is required for decisions")
    if cadence.get("terminal_only_selection") is not False or cadence.get("preserve_checkpoint_peaks") is not True:
        raise ContractError("checkpoint peaks must be retained without waiting for terminal")

    evaluation = prereg.get("evaluation_contract")
    if not isinstance(evaluation, dict) or evaluation.get("engines") != ["Isaac", "MuJoCo"]:
        raise ContractError("evaluation must use immutable Isaac and MuJoCo")
    if evaluation.get("same_schedule_bytes_across_arms_and_engines") is not True:
        raise ContractError("all arms and engines must share exact schedule bytes")
    if evaluation.get("all_scheduled_opportunity_denominator") is not True:
        raise ContractError("evaluation must use every scheduled opportunity as denominator")
    if evaluation.get("self_hit_hard_failure") is not True:
        raise ContractError("self-hit must be a hard failure")

    implementation = prereg.get("implementation_bindings")
    if not isinstance(implementation, dict) or set(implementation) != set(REQUIRED_IMPLEMENTATION_BINDINGS):
        raise ContractError("implementation_bindings must enumerate the exact required capability set")
    if any(value is not None for value in implementation.values()):
        raise ContractError("immutable blocked prereg must keep all unimplemented bindings null")
    gaps = prereg.get("observed_code_capability_gap")
    if not isinstance(gaps, dict) or gaps.get("event_driven_T1_supported") is not False:
        raise ContractError("prereg must state that T1 is not implemented")
    required_gap_flags = {
        "post_strike_only_event_trigger": False,
        "atomic_next_question_install": False,
        "external_deadline_independent_of_clip_phase": False,
        "continuous_immutable_Isaac_judge": False,
        "continuous_immutable_MuJoCo_judge": False,
        "racket_handle_self_hit_gate": False,
    }
    if any(gaps.get(key) != value for key, value in required_gap_flags.items()):
        raise ContractError("observed code capability gaps are incomplete")
    queue = prereg.get("queue")
    if not isinstance(queue, dict) or queue.get("status") != "queued_blocked_no_launch":
        raise ContractError("T0/T1 queue must remain blocked and not launched")
    if queue.get("pod_accessed") is not False or queue.get("training_started") is not False:
        raise ContractError("this prereg turn must not access pods or start training")
    return prereg, venue, spec


def reproduce_venue(report: dict[str, Any], strikes_path: Path) -> dict[str, Any]:
    root = repository_root()
    path = strikes_path.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"restored strikes JSON is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_SOURCE_SHA256:
        raise ContractError(f"restored strikes SHA mismatch: {actual_sha} != {EXPECTED_SOURCE_SHA256}")
    tool_path = root / report["tool"]["repo_path"]
    spec = importlib.util.spec_from_file_location("phase1_rally_interval_audit", tool_path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot import the bound rally interval audit tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.load_strikes(path)
    samples = module.extract_aba(rows, max_leg_s=2.5, excluded_prefixes=("dianqiu",))
    summary = module.summarize(path, rows, samples, max_leg_s=2.5, excluded_prefixes=("dianqiu",))
    expected = report["aggregate"]
    if len(rows) != report["source"]["source_rows"] or len(samples) != expected["accepted_samples"]:
        raise ContractError("reproduced venue row/sample counts disagree")
    category_counts = Counter()
    for sample in samples:
        take = str(sample["take"])
        category = next((name for name in ("gaoqiu", "tantiao", "zhengchang") if take.startswith(name)), "other")
        category_counts[category] += 1
    if dict(category_counts) != expected["coarse_take_category_counts"]:
        raise ContractError(f"reproduced coarse take counts disagree: {dict(category_counts)}")
    for field in ("same_player_interval_s", "self_to_opponent_s", "opponent_to_self_s"):
        actual = summary[field]
        for key, rounded_expected in expected[field].items():
            source_key = "q" + key[1:].zfill(2)
            if abs(float(actual[source_key]) - float(rounded_expected)) > 0.0005 + 1e-12:
                raise ContractError(f"reproduced {field}.{key} differs beyond 3-decimal rounding")
    return {
        "status": "pass",
        "source_sha256": actual_sha,
        "source_rows": len(rows),
        "accepted_samples": len(samples),
        "raw_rows_emitted": False,
        "take_ids_emitted": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument(
        "--mode",
        choices=("design-check", "launch-check", "reproduce-venue"),
        default="design-check",
    )
    parser.add_argument("--strikes-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prereg, venue, spec = validate_prereg(args.prereg.resolve(), args.expected_prereg_sha256)
        if args.mode == "launch-check":
            missing = [name for name, value in prereg["implementation_bindings"].items() if value is None]
            schedule_missing = [
                name
                for name in ("question_schedule_sha256", "screen_schedule_sha256", "decision_schedule_sha256")
                if spec["materialization"].get(name) is None
            ]
            raise ContractError(
                "LAUNCH BLOCKED: event-driven T1 is not implemented; missing implementation bindings="
                f"{missing}, missing immutable schedules={schedule_missing}. Create a new content-addressed "
                "preregistration after implementation and review; do not mutate this one."
            )
        if args.mode == "reproduce-venue":
            if args.strikes_json is None:
                raise ContractError("--strikes-json is required for reproduce-venue")
            result = reproduce_venue(venue, args.strikes_json)
        else:
            if args.strikes_json is not None:
                raise ContractError("--strikes-json is accepted only with --mode reproduce-venue")
            result = {
                "status": "pass_design_only",
                "launch_authorized": False,
                "event_driven_T1_supported": False,
                "venue_source_sha256": venue["source"]["sha256"],
                "venue_median_is_target": False,
                "timing_cells": [cell[0] for cell in EXPECTED_TIMING_CELLS],
                "q10_role": "screen_only",
                "q50_role": "decision",
                "real_robot_authorized": False,
            }
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ContractError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"[phase1-event-timing] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
