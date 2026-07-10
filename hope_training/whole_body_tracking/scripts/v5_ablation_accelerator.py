#!/usr/bin/env python3
"""Build and conservatively halve the V5 professional-transfer ablation paper.

This is an orchestration/decision tool, not a trainer.  It has two deliberately small
jobs:

``manifest``
    Materialize the semantic teacher × timing × path candidates, fail closed against an
    offline feasibility report, and bind every surviving recipe to one immutable exam
    schedule.  It never launches Isaac or a remote job.

``halve``
    Read paired, all-attempt scorecards for one manifest.  Contract/safety failures are
    removed first.  A score-lower recipe is removed only when its paired discordant
    outcomes conservatively establish dominance; uncertain small samples stay alive.
    Mechanism controls are protected unless they themselves fail safety or contract.

The JSON schemas are intentionally explicit and dependency-free.  Run either subcommand
with ``--help`` for an example.  Design source:
``docs/research/v5_professional_transfer_audit_2026-07-10.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
TEACHERS = ("task_only", "v4", "v5")
TIMINGS = ("robot", "human")

# A combined path must carry its backswing dose in the semantic name.  Calling two
# physically different clips merely "combined" would make candidate IDs lie.
PATHS = (
    "original",
    "backswing_20",
    "backswing_40",
    "followthrough",
    "combined_20",
    "combined_40",
)
EXTENSION_PATHS = frozenset(
    ("backswing_20", "backswing_40", "combined_20", "combined_40")
)
FOLLOWTHROUGH_PATHS = frozenset(("followthrough", "combined_20", "combined_40"))

BASE_HARD_GUARDS = (
    "strike_window_locked",
    "qdes_limits_pass",
    "self_collision_pass",
    "balance_pass",
    "friction_pass",
    "torque_pass",
    "velocity_pass",
    "timing_pass",
)
MORPH_HARD_GUARD = "strike_invariant_pass"
EXTENSION_EVIDENCE = (
    "original_path_limited",
    "stroke_length_gain_m",
    "a_min_reduction_mps2",
    "min_torque_margin_gain",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORMAL_READY_STATE_MODE = "mjcf_named_keyframe:stand:v1"


def canonical_json(value: Any) -> str:
    """Canonical JSON used by every content identity in this file."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(payload + "\n", encoding="utf-8")


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _strict_bool(value: Any) -> bool:
    return isinstance(value, bool) and value


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def validate_candidate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized semantic recipe, rejecting meaningless empty controls."""

    teacher = spec.get("teacher")
    timing = spec.get("timing")
    path = spec.get("path")
    if teacher not in TEACHERS:
        raise ValueError(f"unknown teacher {teacher!r}; expected one of {TEACHERS}")
    if timing not in TIMINGS:
        raise ValueError(f"unknown timing {timing!r}; expected one of {TIMINGS}")
    if path not in PATHS:
        raise ValueError(f"unknown path {path!r}; expected one of {PATHS}")
    if teacher == "task_only" and (timing != "robot" or path != "original"):
        raise ValueError(
            "task_only has no teacher clock or reference path; only robot/original is semantic"
        )

    extension_fraction = 0.0
    if path.endswith("_20"):
        extension_fraction = 0.20
    elif path.endswith("_40"):
        extension_fraction = 0.40
    out = {
        "teacher": teacher,
        "timing": timing,
        "path": path,
        "reference_mode": "none" if teacher == "task_only" else "soft_prior",
        "path_parameters": {
            "backswing_extension_fraction": extension_fraction,
            "rewrite_followthrough": path in FOLLOWTHROUGH_PATHS,
        },
    }
    return out


def candidate_id(spec: Mapping[str, Any]) -> str:
    normalized = validate_candidate_spec(spec)
    digest = content_sha256(
        {"candidate_schema_version": SCHEMA_VERSION, "recipe": normalized}
    )[:16]
    return (
        f"cand-{normalized['teacher']}-{normalized['timing']}-"
        f"{normalized['path']}-{digest}"
    )


def candidate_slots() -> tuple[dict[str, Any], ...]:
    """Finite, deterministic candidate paper; no task-only human/path pseudo-controls."""

    slots = [validate_candidate_spec(
        {"teacher": "task_only", "timing": "robot", "path": "original"}
    )]
    for teacher in ("v4", "v5"):
        for timing in TIMINGS:
            for path in PATHS:
                slots.append(validate_candidate_spec(
                    {"teacher": teacher, "timing": timing, "path": path}
                ))
    return tuple(slots)


def _record_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    normalized = validate_candidate_spec(value)
    return normalized["teacher"], normalized["timing"], normalized["path"]


def _feasibility_reasons(record: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    """Return every fail-closed reason.  An empty list is the only admission state."""

    reasons: list[str] = []
    if not _strict_bool(record.get("feasible")):
        reasons.append("feasible_not_explicitly_true")

    hard_failures = record.get("hard_failures", [])
    if not isinstance(hard_failures, list):
        reasons.append("hard_failures_not_a_list")
    elif hard_failures:
        reasons.append("reported_hard_failure")

    guards = record.get("hard_guards")
    if not isinstance(guards, Mapping):
        reasons.append("hard_guards_missing")
        guards = {}
    required = list(BASE_HARD_GUARDS)
    if spec["path"] != "original":
        required.append(MORPH_HARD_GUARD)
    for guard in required:
        if not _strict_bool(guards.get(guard)):
            reasons.append(f"guard_not_true:{guard}")

    if spec["path"] in EXTENSION_PATHS:
        evidence = record.get("extension_evidence")
        if not isinstance(evidence, Mapping):
            reasons.append("extension_evidence_missing")
            evidence = {}
        if not _strict_bool(evidence.get("original_path_limited")):
            reasons.append("original_path_limit_not_proven")
        for field in EXTENSION_EVIDENCE[1:]:
            if not _positive_number(evidence.get(field)):
                reasons.append(f"extension_gain_not_positive:{field}")
    return reasons


def build_manifest(
    feasibility: Mapping[str, Any],
    *,
    schedule_sha256: str,
    schedule_seed: int,
    ready_state_sha256: str,
    mjcf_sha256: str,
    execution_contract_sha256: str,
    training_seed_base: int = 20260710,
    m4_iterations: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, content-addressed accelerator manifest."""

    _require_sha256(schedule_sha256, "schedule_sha256")
    _require_sha256(ready_state_sha256, "ready_state_sha256")
    _require_sha256(mjcf_sha256, "mjcf_sha256")
    _require_sha256(execution_contract_sha256, "execution_contract_sha256")
    if not isinstance(schedule_seed, int) or isinstance(schedule_seed, bool):
        raise ValueError("schedule_seed must be an integer")
    if not isinstance(training_seed_base, int) or isinstance(training_seed_base, bool):
        raise ValueError("training_seed_base must be an integer")
    if m4_iterations is not None and (
        not isinstance(m4_iterations, int)
        or isinstance(m4_iterations, bool)
        or m4_iterations <= 0
    ):
        raise ValueError("m4_iterations must be a positive integer when supplied")
    if feasibility.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"feasibility schema_version must be {SCHEMA_VERSION}")
    records = feasibility.get("records")
    if not isinstance(records, list):
        raise ValueError("feasibility records must be a list")

    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"feasibility records[{index}] must be an object")
        key = _record_key(record)
        if key[0] == "task_only":
            raise ValueError("task_only must not have fake motion-feasibility evidence")
        if key in by_key:
            raise ValueError(f"duplicate feasibility record for {key}")
        by_key[key] = record

    # A JSON list's presentation order is not part of this scientific paper.  Bind the
    # same record set to the same SHA even when an upstream parallel oracle finishes in a
    # different order; every individual record remains content-addressed below.
    normalized_feasibility = dict(feasibility)
    normalized_feasibility["records"] = [by_key[key] for key in sorted(by_key)]

    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for spec in candidate_slots():
        cid = candidate_id(spec)
        key = (spec["teacher"], spec["timing"], spec["path"])
        if spec["teacher"] == "task_only":
            admitted.append({
                "candidate_id": cid,
                "recipe": spec,
                "feasibility": {"status": "not_applicable_task_only"},
            })
            continue
        record = by_key.get(key)
        if record is None:
            rejected.append({
                "candidate_id": cid,
                "recipe": spec,
                "reasons": ["missing_feasibility_record"],
            })
            continue
        reasons = _feasibility_reasons(record, spec)
        if reasons:
            rejected.append({
                "candidate_id": cid,
                "recipe": spec,
                "reasons": sorted(set(reasons)),
                "feasibility_record_sha256": content_sha256(record),
            })
            continue
        admitted.append({
            "candidate_id": cid,
            "recipe": spec,
            "feasibility": {
                "status": "passed",
                "record_sha256": content_sha256(record),
            },
        })

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v5_professional_transfer_ablation_manifest",
        "protocol": {
            "same_question_exam": {
                "schedule_sha256": schedule_sha256,
                "schedule_seed": schedule_seed,
                "denominator": "all_attempts",
                "one_question_one_reset": True,
                "ready_state_mode": FORMAL_READY_STATE_MODE,
                "ready_state_sha256": ready_state_sha256,
                "mjcf_sha256": mjcf_sha256,
                "execution_contract_sha256": execution_contract_sha256,
            },
            "selection_rule": (
                "hard safety/contract first; paired conservative dominance; preserve controls"
            ),
        },
        "budgets": {
            "M2": {
                "num_envs": 512,
                "iterations": 25,
                "purpose": "mechanism_smoke_not_absolute_ranking",
            },
            "M3": {
                "num_envs": 4096,
                "iterations": 2000,
                "selection": "paired_conservative_successive_halving",
            },
            "M4": {
                "candidate_limit": 2,
                "seed_count": 3,
                "training_seeds": [training_seed_base + i for i in range(3)],
                "iterations": m4_iterations,
                "launch_ready": m4_iterations is not None,
            },
        },
        "feasibility_input_sha256": content_sha256(normalized_feasibility),
        "candidates": admitted,
        "rejected_candidates": rejected,
    }
    body["manifest_sha256"] = content_sha256(body)
    return body


def wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval without pretending n=0 contains information."""

    if attempts < 0 or successes < 0 or successes > attempts:
        raise ValueError("need 0 <= successes <= attempts")
    if attempts == 0:
        return 0.0, 1.0
    n = float(attempts)
    p = float(successes) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _candidate_control_roles(recipe: Mapping[str, Any]) -> tuple[str, ...]:
    roles: list[str] = []
    if recipe["teacher"] == "task_only":
        roles.append("task_only_teacher_baseline")
    if recipe["teacher"] == "v4":
        roles.append("v4_nonprofessional_teacher_control")
    if recipe["timing"] == "human":
        roles.append("human_timing_control")
    if recipe["path"] == "original":
        roles.append("original_path_control")
    return tuple(roles)


def _validate_attempts(
    card: Mapping[str, Any], expected_ids: Sequence[str]
) -> tuple[list[bool] | None, list[str]]:
    reasons: list[str] = []
    attempts = card.get("attempts")
    if not isinstance(attempts, list):
        return None, ["attempts_missing"]
    actual_ids: list[str] = []
    outcomes: list[bool] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            reasons.append(f"attempt_not_object:{index}")
            continue
        qid = attempt.get("question_id")
        if not isinstance(qid, str) or not qid:
            reasons.append(f"question_id_missing:{index}")
        else:
            actual_ids.append(qid)
        success = attempt.get("return_success", attempt.get("returned"))
        if (
            "return_success" in attempt
            and "returned" in attempt
            and attempt.get("return_success") != attempt.get("returned")
        ):
            reasons.append(f"return_outcome_fields_disagree:{index}")
        if not isinstance(success, bool):
            reasons.append(f"return_success_not_bool:{index}")
        else:
            # Intentionally do not filter on eligible.  Falls, guard resets and other
            # scheduled attempts remain denominator failures.
            outcomes.append(success)
        if attempt.get("censored") is True:
            reasons.append(f"censored_attempt:{index}")
        for field in ("physical_fall", "hard_limit_violation", "nan_detected"):
            if attempt.get(field) is True:
                reasons.append(f"hard_safety:{field}:{index}")
        if success is True and (
            attempt.get("physical_fall") is True
            or attempt.get("hard_limit_violation") is True
            or attempt.get("nan_detected") is True
        ):
            reasons.append(f"success_with_hard_safety_failure:{index}")
    if tuple(actual_ids) != tuple(expected_ids):
        reasons.append("question_order_or_count_mismatch")
    if len(outcomes) != len(expected_ids):
        reasons.append("outcome_count_mismatch")
    return (outcomes if not reasons else None), sorted(set(reasons))


def _paired_comparison(
    candidate: Sequence[bool], best: Sequence[bool], *, min_discordant: int
) -> dict[str, Any]:
    wins = sum(a and not b for a, b in zip(candidate, best))
    losses = sum(b and not a for a, b in zip(candidate, best))
    discordant = wins + losses
    low, high = wilson_interval(wins, discordant)
    stable_dominated = discordant >= min_discordant and high < 0.5
    return {
        "wins": wins,
        "losses": losses,
        "discordant": discordant,
        "paired_delta_all_attempts": (wins - losses) / len(candidate),
        "conditional_win_wilson_95": [low, high],
        "min_discordant": min_discordant,
        "stable_dominated": stable_dominated,
    }


def halve_scorecards(
    manifest: Mapping[str, Any],
    scorecards: Mapping[str, Any],
    *,
    min_discordant: int = 10,
) -> dict[str, Any]:
    """Perform one conservative halving decision over a paired scorecard paper."""

    if not isinstance(min_discordant, int) or isinstance(min_discordant, bool) or min_discordant < 1:
        raise ValueError("min_discordant must be a positive integer")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    manifest_copy = dict(manifest)
    claimed_manifest_sha = manifest_copy.pop("manifest_sha256", None)
    if claimed_manifest_sha != content_sha256(manifest_copy):
        raise ValueError("manifest_sha256 does not match manifest content")
    if scorecards.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"scorecards schema_version must be {SCHEMA_VERSION}")

    schedule = manifest["protocol"]["same_question_exam"]
    expected_sha = _require_sha256(schedule.get("schedule_sha256"), "manifest schedule SHA")
    expected_ready_sha = _require_sha256(
        schedule.get("ready_state_sha256"), "manifest ready-state SHA"
    )
    expected_mjcf_sha = _require_sha256(schedule.get("mjcf_sha256"), "manifest MJCF SHA")
    expected_execution_sha = _require_sha256(
        schedule.get("execution_contract_sha256"), "manifest execution-contract SHA"
    )
    if schedule.get("ready_state_mode") != FORMAL_READY_STATE_MODE:
        raise ValueError(
            "manifest formal exam must use the shared MJCF named stand keyframe ready state"
        )
    expected_seed = schedule.get("schedule_seed")
    if scorecards.get("manifest_sha256") != claimed_manifest_sha:
        raise ValueError("scorecards manifest_sha256 does not match manifest")
    if scorecards.get("schedule_sha256") != expected_sha:
        raise ValueError("scorecards schedule_sha256 does not match manifest")
    if scorecards.get("schedule_seed") != expected_seed:
        raise ValueError("scorecards schedule_seed does not match manifest")
    for field, expected in (
        ("ready_state_sha256", expected_ready_sha),
        ("mjcf_sha256", expected_mjcf_sha),
        ("execution_contract_sha256", expected_execution_sha),
    ):
        if scorecards.get(field) != expected:
            raise ValueError(f"scorecards {field} does not match manifest")
    if scorecards.get("ready_state_mode") != FORMAL_READY_STATE_MODE:
        raise ValueError("scorecards ready_state_mode is not the formal shared stand keyframe")
    question_ids = scorecards.get("question_ids")
    if (
        not isinstance(question_ids, list)
        or not question_ids
        or not all(isinstance(v, str) and v for v in question_ids)
        or len(set(question_ids)) != len(question_ids)
    ):
        raise ValueError("scorecards question_ids must be a non-empty unique string list")
    cards = scorecards.get("candidates")
    if not isinstance(cards, list):
        raise ValueError("scorecards candidates must be a list")

    recipes = {
        row["candidate_id"]: row["recipe"] for row in manifest.get("candidates", [])
    }
    seen: set[str] = set()
    valid: dict[str, dict[str, Any]] = {}
    hard_eliminated: list[dict[str, Any]] = []
    for index, card in enumerate(cards):
        if not isinstance(card, Mapping):
            raise ValueError(f"scorecards candidates[{index}] must be an object")
        cid = card.get("candidate_id")
        if not isinstance(cid, str) or cid not in recipes:
            raise ValueError(f"unknown candidate_id {cid!r} in scorecards")
        if cid in seen:
            raise ValueError(f"duplicate scorecard for {cid}")
        seen.add(cid)
        # These identities define the plant and initial condition.  Treating a mismatch as one
        # candidate's ordinary hard failure would still permit a cross-candidate ranking over
        # different experiments, so reject the entire halving round instead.
        for field, expected in (
            ("ready_state_sha256", expected_ready_sha),
            ("mjcf_sha256", expected_mjcf_sha),
            ("execution_contract_sha256", expected_execution_sha),
        ):
            if card.get(field) != expected:
                raise ValueError(
                    f"candidate {cid} {field} does not match the common manifest contract"
                )
        if card.get("ready_state_mode") != FORMAL_READY_STATE_MODE:
            raise ValueError(
                f"candidate {cid} ready_state_mode is not the common formal stand keyframe"
            )
        reasons: list[str] = []
        if card.get("schedule_sha256") != expected_sha:
            reasons.append("candidate_schedule_sha_mismatch")
        if not _strict_bool(card.get("contract_exact")):
            reasons.append("contract_not_exact")
        if not _strict_bool(card.get("hard_safety_pass")):
            reasons.append("hard_safety_not_explicitly_true")
        reported = card.get("hard_failures", [])
        if not isinstance(reported, list):
            reasons.append("hard_failures_not_a_list")
        elif reported:
            reasons.append("reported_hard_failure")
        outcomes, attempt_reasons = _validate_attempts(card, question_ids)
        reasons.extend(attempt_reasons)
        if reasons:
            hard_eliminated.append({
                "candidate_id": cid,
                "phase": "hard_safety_or_contract",
                "reasons": sorted(set(reasons)),
            })
            continue
        assert outcomes is not None
        successes = sum(outcomes)
        low, high = wilson_interval(successes, len(outcomes))
        valid[cid] = {
            "candidate_id": cid,
            "recipe": recipes[cid],
            "outcomes": outcomes,
            "successes": successes,
            "attempts": len(outcomes),
            "return_rate_all_attempts": successes / len(outcomes),
            "return_rate_wilson_95": [low, high],
        }

    # A submitted round is a complete paper.  Silently absent candidates are contract
    # failures rather than a way to make a weak arm disappear from the denominator.
    for cid in sorted(set(recipes) - seen):
        hard_eliminated.append({
            "candidate_id": cid,
            "phase": "hard_safety_or_contract",
            "reasons": ["scorecard_missing"],
        })

    ranking = sorted(
        valid,
        key=lambda cid: (-valid[cid]["return_rate_all_attempts"], cid),
    )
    best_id = ranking[0] if ranking else None
    comparisons: dict[str, Any] = {}
    if best_id is not None:
        for cid in ranking:
            if cid == best_id:
                comparisons[cid] = {
                    "wins": 0,
                    "losses": 0,
                    "discordant": 0,
                    "paired_delta_all_attempts": 0.0,
                    "conditional_win_wilson_95": [0.0, 1.0],
                    "min_discordant": min_discordant,
                    "stable_dominated": False,
                }
            else:
                comparisons[cid] = _paired_comparison(
                    valid[cid]["outcomes"],
                    valid[best_id]["outcomes"],
                    min_discordant=min_discordant,
                )

    # Preserve the strongest valid instance of each explanatory control.  Controls do
    # not override a hard failure; those candidates were already removed above.
    protected_by_role: dict[str, str] = {}
    for cid in ranking:
        for role in _candidate_control_roles(valid[cid]["recipe"]):
            protected_by_role.setdefault(role, cid)
    protected = set(protected_by_role.values())

    requested_target = math.ceil(len(valid) / 2.0)
    # The round winner is never removed merely to make room for controls.  Count it in
    # the effective floor when it is not already one of the protected controls.
    non_droppable = protected | ({best_id} if best_id is not None else set())
    target = max(requested_target, len(non_droppable))
    worst_first = list(reversed(ranking))
    droppable = [
        cid for cid in worst_first
        if cid not in protected and comparisons[cid]["stable_dominated"]
    ]
    drop_count = min(max(0, len(valid) - target), len(droppable))
    score_eliminated_ids = set(droppable[:drop_count])
    survivors = [cid for cid in ranking if cid not in score_eliminated_ids]

    score_eliminated = [{
        "candidate_id": cid,
        "phase": "paired_dominance",
        "against": best_id,
        "comparison": comparisons[cid],
    } for cid in droppable[:drop_count]]

    rows = []
    for rank, cid in enumerate(ranking, start=1):
        row = dict(valid[cid])
        row.pop("outcomes")
        row["rank"] = rank
        row["paired_vs_best"] = comparisons[cid]
        row["protected_control_roles"] = sorted(
            role for role, protected_id in protected_by_role.items() if protected_id == cid
        )
        rows.append(row)

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v5_professional_transfer_halving_decision",
        "manifest_sha256": claimed_manifest_sha,
        "scorecards_input_sha256": content_sha256(scorecards),
        "schedule_sha256": expected_sha,
        "ready_state_mode": FORMAL_READY_STATE_MODE,
        "ready_state_sha256": expected_ready_sha,
        "mjcf_sha256": expected_mjcf_sha,
        "execution_contract_sha256": expected_execution_sha,
        "primary_metric": "return_success/all_attempts",
        "best_candidate_id": best_id,
        "ranked_valid_candidates": rows,
        "protected_controls": protected_by_role,
        "survivor_candidate_ids": survivors,
        "eliminated": hard_eliminated + score_eliminated,
        "round_summary": {
            "submitted_candidates": len(cards),
            "valid_candidates": len(valid),
            "requested_half_target": requested_target,
            "effective_target_after_controls": target,
            "survivors": len(survivors),
            "conservative_hold": len(survivors) > target,
            "hold_reason": (
                "insufficient_paired_evidence"
                if len(survivors) > target else None
            ),
        },
    }
    body["decision_sha256"] = content_sha256(body)
    return body


MANIFEST_EXAMPLE = """feasibility JSON (one record per motion recipe; omitted = rejected):
  {"schema_version":1,"records":[{
    "teacher":"v5","timing":"robot","path":"backswing_20","feasible":true,
    "hard_failures":[],"hard_guards":{
      "strike_window_locked":true,"qdes_limits_pass":true,
      "self_collision_pass":true,"balance_pass":true,"friction_pass":true,
      "torque_pass":true,"velocity_pass":true,"timing_pass":true,
      "strike_invariant_pass":true},
    "extension_evidence":{"original_path_limited":true,
      "stroke_length_gain_m":0.08,"a_min_reduction_mps2":3.0,
      "min_torque_margin_gain":0.05}}]}"""

HALVE_EXAMPLE = """paired scorecard JSON:
  {"schema_version":1,"manifest_sha256":"<manifest SHA>",
   "schedule_sha256":"<64 hex>","schedule_seed":17,
   "ready_state_mode":"mjcf_named_keyframe:stand:v1",
   "ready_state_sha256":"<64 hex>","mjcf_sha256":"<64 hex>",
   "execution_contract_sha256":"<64 hex>","question_ids":["q0","q1"],
   "candidates":[{"candidate_id":"cand-...","schedule_sha256":"<same>",
     "ready_state_mode":"mjcf_named_keyframe:stand:v1",
     "ready_state_sha256":"<same>","mjcf_sha256":"<same>",
     "execution_contract_sha256":"<same>",
     "contract_exact":true,"hard_safety_pass":true,"hard_failures":[],
     "attempts":[{"question_id":"q0","return_success":true},
                 {"question_id":"q1","return_success":false}]}]}"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic V5/Phase ablation accelerator (never launches training).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pm = sub.add_parser(
        "manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Build the semantic task-only/V4/V5 × human/robot × path paper. "
            "Task-only is only robot/original. Extension arms require explicit positive "
            "stroke-length, a_min and torque-margin evidence.\n\n" + MANIFEST_EXAMPLE
        ),
    )
    pm.add_argument("--feasibility", required=True, help="offline feasibility JSON")
    pm.add_argument("--schedule-sha256", required=True, help="fixed BankExam schedule SHA")
    pm.add_argument("--schedule-seed", required=True, type=int, help="fixed BankExam seed")
    pm.add_argument("--ready-state-sha256", required=True,
                    help="formal evaluator shared stand-keyframe state SHA")
    pm.add_argument("--mjcf-sha256", required=True, help="formal evaluator MJCF artifact SHA")
    pm.add_argument("--execution-contract-sha256", required=True,
                    help="formal evaluator execution-contract SHA")
    pm.add_argument("--training-seed-base", type=int, default=20260710)
    pm.add_argument(
        "--m4-iterations", type=int,
        help="explicit full-training budget; omission keeps M4 launch_ready=false",
    )
    pm.add_argument("--output", required=True, help="deterministic manifest JSON")

    ph = sub.add_parser(
        "halve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Apply one conservative paired halving round. Every scheduled attempt stays "
            "in the denominator; censored/missing/misaligned papers fail contract.\n\n"
            + HALVE_EXAMPLE
        ),
    )
    ph.add_argument("--manifest", required=True)
    ph.add_argument("--scorecards", required=True)
    ph.add_argument("--min-discordant", type=int, default=10)
    ph.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        result = build_manifest(
            _read_json(args.feasibility),
            schedule_sha256=args.schedule_sha256,
            schedule_seed=args.schedule_seed,
            ready_state_sha256=args.ready_state_sha256,
            mjcf_sha256=args.mjcf_sha256,
            execution_contract_sha256=args.execution_contract_sha256,
            training_seed_base=args.training_seed_base,
            m4_iterations=args.m4_iterations,
        )
    else:
        result = halve_scorecards(
            _read_json(args.manifest),
            _read_json(args.scorecards),
            min_discordant=args.min_discordant,
        )
    _write_json(args.output, result)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
