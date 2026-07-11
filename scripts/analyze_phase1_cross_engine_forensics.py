#!/usr/bin/env python3
"""Fail-closed, question-aligned Isaac/MuJoCo Phase-1 forensic analysis.

This tool is intentionally read-only with respect to the evaluator artifacts.  It verifies every
input SHA, joins attempts by immutable question ID, and writes one deterministic JSON report.  It
does not rescore, change thresholds, or run either simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping


class ForensicError(RuntimeError):
    """Raised when an input violates the frozen forensic contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForensicError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForensicError(f"JSON root must be an object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise ForensicError(f"cannot read CSV {path}: {exc}") from exc


def require_artifact(root: Path, spec: Mapping[str, Any]) -> Path:
    rel = spec.get("path")
    expected = spec.get("sha256")
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ForensicError(f"artifact path must be a safe relative path: {rel!r}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ForensicError(f"artifact {rel!r} has no SHA-256")
    path = root / rel
    if not path.is_file():
        raise ForensicError(f"required artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ForensicError(f"artifact SHA mismatch for {path}: expected {expected}, got {actual}")
    return path


def parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("1", "True", "true"):
        return True
    if value in ("0", "False", "false"):
        return False
    raise ForensicError(f"{field} is not a strict boolean: {value!r}")


def finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForensicError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ForensicError(f"{field} is not finite: {value!r}")
    return number


def count(rows: Iterable[Any], predicate: Callable[[Any], bool]) -> int:
    return sum(bool(predicate(row)) for row in rows)


def side_rows(rows: list[Any], side: str) -> list[Any]:
    if side == "aggregate":
        return rows
    return [row for row in rows if row[0]["clip_name"] == side]


def outcome_counts(rows: list[tuple[dict[str, str], dict[str, str] | None, dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("forehand", "backhand", "aggregate"):
        selected = side_rows(rows, side)
        result[side] = {
            "n_questions": len(selected),
            "mujoco": {
                "n_reached_strike": count(selected, lambda row: row[1] is not None),
                "n_hit": count(selected, lambda row: parse_bool(row[0]["hit"], field="mujoco.hit")),
                "n_returned": count(
                    selected, lambda row: parse_bool(row[0]["returned"], field="mujoco.returned")
                ),
                "n_physical_fall": count(
                    selected,
                    lambda row: parse_bool(row[0]["physical_fall"], field="mujoco.physical_fall"),
                ),
                "n_guard_reset": count(
                    selected,
                    lambda row: parse_bool(row[0]["guard_reset"], field="mujoco.guard_reset"),
                ),
                "n_counterfactual_contact": count(
                    selected,
                    lambda row: row[1] is not None
                    and parse_bool(row[1]["cf_contacted"], field="mujoco.cf_contacted"),
                ),
                "n_counterfactual_legal_landing": count(
                    selected,
                    lambda row: row[1] is not None
                    and parse_bool(row[1]["cf_landed_ok"], field="mujoco.cf_landed_ok"),
                ),
            },
            "isaac": {
                "n_reached_strike": count(selected, lambda row: bool(row[2]["reached_exact"])),
                "n_virtual_hit": count(selected, lambda row: bool(row[2]["hit"])),
                "n_virtual_returned": count(selected, lambda row: bool(row[2]["returned"])),
                "n_physical_fall": count(selected, lambda row: bool(row[2]["physical_fall"])),
                "n_guard_reset": count(selected, lambda row: bool(row[2]["guard_reset"])),
            },
        }
    return result


def continuous_metrics(
    rows: list[tuple[dict[str, str], dict[str, str] | None, dict[str, Any]]]
) -> dict[str, Any]:
    mapping = {
        "racket_position_error_m": ("pos_err", "pos_error_m"),
        "racket_velocity_error_mps": ("vel_err", "vel_error_mps"),
        "racket_normal_error_deg": ("normal_err_deg", "normal_error_deg"),
    }
    result: dict[str, Any] = {}
    for side in ("forehand", "backhand"):
        selected = [
            row
            for row in side_rows(rows, side)
            if row[1] is not None and row[2].get("pos_error_m") is not None
        ]
        result[side] = {}
        for label, (mujoco_key, isaac_key) in mapping.items():
            mujoco = [finite_float(row[1][mujoco_key], field=mujoco_key) for row in selected]
            isaac = [finite_float(row[2][isaac_key], field=isaac_key) for row in selected]
            if not mujoco or len(mujoco) != len(isaac):
                raise ForensicError(f"no paired finite {label} samples for {side}")
            result[side][label] = {
                "n_paired": len(mujoco),
                "mujoco_mean": fmean(mujoco),
                "mujoco_min": min(mujoco),
                "mujoco_max": max(mujoco),
                "isaac_mean": fmean(isaac),
                "isaac_min": min(isaac),
                "isaac_max": max(isaac),
                "paired_mean_absolute_difference": fmean(
                    abs(left - right) for left, right in zip(mujoco, isaac)
                ),
            }
    return result


def agreement_and_causes(
    rows: list[tuple[dict[str, str], dict[str, str] | None, dict[str, Any]]]
) -> dict[str, Any]:
    matrix = Counter()
    causes = Counter()
    for attempt, strike, isaac in rows:
        mujoco_return = parse_bool(attempt["returned"], field="mujoco.returned")
        isaac_return = bool(isaac["returned"])
        matrix[f"mujoco_{int(mujoco_return)}_isaac_{int(isaac_return)}"] += 1
        if not mujoco_return and isaac_return:
            if strike is None:
                causes["mujoco_no_strike"] += 1
            elif not parse_bool(strike["contacted"], field="mujoco.contacted"):
                causes["mujoco_no_physical_contact"] += 1
            elif not parse_bool(strike["net_clear"], field="mujoco.net_clear"):
                causes["mujoco_physical_contact_but_net_fail"] += 1
            elif not parse_bool(strike["landed_ok"], field="mujoco.landed_ok"):
                causes["mujoco_physical_contact_but_landing_fail"] += 1
            else:
                causes["unclassified"] += 1
    return {
        "return_agreement_matrix": dict(sorted(matrix.items())),
        "mujoco_0_isaac_1_question_causes": dict(sorted(causes.items())),
    }


def base_and_termination(
    rows: list[tuple[dict[str, str], dict[str, str] | None, dict[str, Any]]],
    mujoco_summary: Mapping[str, Any],
) -> dict[str, Any]:
    strikes = [row[1] for row in rows if row[1] is not None]
    result0 = mujoco_summary.get("results", [None])[0]
    if not isinstance(result0, dict):
        raise ForensicError("MuJoCo summary has no results[0]")
    base = {
        axis: fmean(finite_float(row[f"base_pos_w_{axis}"], field=f"base_pos_w_{axis}") for row in strikes)
        for axis in ("x", "y", "z")
    }
    physical = count(rows, lambda row: parse_bool(row[0]["physical_fall"], field="physical_fall"))
    guards = count(rows, lambda row: parse_bool(row[0]["guard_reset"], field="guard_reset"))
    return {
        "mujoco_at_strike_base_position_mean_m": base,
        "mujoco_run_level": {
            "base_roll_deg": result0.get("base_roll_deg"),
            "base_pitch_deg": result0.get("base_pitch_deg"),
            "foot_contact_fraction": result0.get("foot_contact_frac"),
        },
        "isaac_base_state_observability": "not_recorded_in_accepted_scorecard_or_attempt_csv",
        "termination": {
            "mujoco_attempt_physical_fall_count": physical,
            "mujoco_attempt_guard_reset_count": guards,
            "mujoco_summary_fell_union_count": result0.get("fell"),
            "mujoco_summary_term_breakdown": result0.get("term_breakdown", {}),
            "summary_fell_is_physical_plus_guard_union": result0.get("fell") == physical + guards,
            "isaac_attempt_physical_fall_count": count(rows, lambda row: bool(row[2]["physical_fall"])),
            "isaac_attempt_guard_reset_count": count(rows, lambda row: bool(row[2]["guard_reset"])),
        },
    }


def analyze_arm(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    files = spec.get("artifacts")
    if not isinstance(files, dict):
        raise ForensicError("arm has no artifact mapping")
    resolved = {name: require_artifact(root, item) for name, item in files.items()}
    required = {"mujoco_attempts", "mujoco_strikes", "mujoco_summary", "isaac_json", "isaac_csv"}
    if set(resolved) != required:
        raise ForensicError(f"arm artifacts must be exactly {sorted(required)}")

    mujoco_attempts = load_csv(resolved["mujoco_attempts"])
    mujoco_strikes = load_csv(resolved["mujoco_strikes"])
    mujoco_summary = load_json(resolved["mujoco_summary"])
    isaac = load_json(resolved["isaac_json"])
    isaac_csv = load_csv(resolved["isaac_csv"])
    isaac_attempts = isaac.get("attempts")
    if not isinstance(isaac_attempts, list):
        raise ForensicError("Isaac scorecard has no attempts list")
    if len(mujoco_attempts) != 100 or len(isaac_attempts) != 100 or len(isaac_csv) != 100:
        raise ForensicError("each engine must expose exactly 100 uncensored attempts")
    mujoco_ids = [row.get("question_id") for row in mujoco_attempts]
    isaac_ids = [row.get("question_id") for row in isaac_attempts]
    isaac_csv_ids = [row.get("question_id") for row in isaac_csv]
    if mujoco_ids != isaac_ids or mujoco_ids != isaac_csv_ids or len(set(mujoco_ids)) != 100:
        raise ForensicError("question ID/order differs across engine ledgers")
    if any(parse_bool(row["censored"], field="mujoco.censored") for row in mujoco_attempts):
        raise ForensicError("MuJoCo ledger contains censored attempts")
    if any(bool(row.get("censored")) for row in isaac_attempts):
        raise ForensicError("Isaac ledger contains censored attempts")
    for csv_row, json_row in zip(isaac_csv, isaac_attempts):
        for key in ("hit", "returned", "physical_fall", "guard_reset", "reached_exact"):
            if parse_bool(csv_row[key], field=f"isaac_csv.{key}") is not bool(json_row[key]):
                raise ForensicError(f"Isaac JSON/CSV disagree on {key}")

    strike_by_id = {row.get("question_id"): row for row in mujoco_strikes}
    if len(strike_by_id) != len(mujoco_strikes) or not set(strike_by_id).issubset(set(mujoco_ids)):
        raise ForensicError("MuJoCo strike ledger has duplicate or foreign question IDs")
    joined = [
        (mujoco_row, strike_by_id.get(question_id), isaac_row)
        for question_id, mujoco_row, isaac_row in zip(mujoco_ids, mujoco_attempts, isaac_attempts)
    ]
    mujoco_ready = {
        "mode": mujoco_attempts[0].get("ready_state_mode"),
        "sha256": mujoco_attempts[0].get("ready_state_sha256"),
    }
    isaac_ready = {
        "mode": isaac.get("nominal_eval_profile", {}).get("ready_state"),
        "sha256": isaac.get("ready_state_sha256"),
    }
    if any(row.get("ready_state_sha256") != mujoco_ready["sha256"] for row in mujoco_attempts):
        raise ForensicError("MuJoCo attempts disagree on ready-state SHA")
    if any(row.get("ready_state_sha256") != isaac_ready["sha256"] for row in isaac_attempts):
        raise ForensicError("Isaac attempts disagree on ready-state SHA")
    execution = mujoco_summary.get("execution_contract", {})
    profile = isaac.get("nominal_eval_profile", {})
    return {
        "name": spec.get("name"),
        "input_artifacts": {
            name: {"path": str(files[name]["path"]), "sha256": files[name]["sha256"]}
            for name in sorted(files)
        },
        "question_id_order_sha256": canonical_sha256(mujoco_ids),
        "question_order_equal": True,
        "outcomes": outcome_counts(joined),
        "continuous_at_strike": continuous_metrics(joined),
        "cross_engine_disagreement": agreement_and_causes(joined),
        "ready_state": {
            "mujoco": mujoco_ready,
            "isaac": isaac_ready,
            "byte_contract_equal": mujoco_ready["sha256"] == isaac_ready["sha256"],
        },
        "plant_and_scorer": {
            "mujoco_frictionloss_mode": execution.get("frictionloss_mode"),
            "mujoco_frictionloss_min": min(execution.get("mujoco_actuated_dof_frictionloss", [math.nan])),
            "mujoco_frictionloss_max": max(execution.get("mujoco_actuated_dof_frictionloss", [math.nan])),
            "mujoco_physical_ball_outcome_recorded": True,
            "mujoco_counterfactual_virtual_scorer_sha256": mujoco_summary.get(
                "virtual_return_scorer_contract_sha256"
            ),
            "isaac_physical_ball": profile.get("changes", {}).get(
                "commands.racket_target.physical_ball"
            ),
            "isaac_virtual_ball_outcome_recorded": True,
            "isaac_evaluation_contract_exact": isaac.get("evaluation_contract_exact"),
        },
        "base_and_termination": base_and_termination(joined, mujoco_summary),
    }


def arm_return_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def returns(arm: Mapping[str, Any], engine: str, side: str) -> int:
        key = "n_returned" if engine == "mujoco" else "n_virtual_returned"
        return int(arm["outcomes"][side][engine][key])

    return {
        engine: {
            side: returns(right, engine, side) - returns(left, engine, side)
            for side in ("forehand", "backhand", "aggregate")
        }
        for engine in ("mujoco", "isaac")
    }


def diagnosis(output_pairs: Mapping[str, Any]) -> dict[str, Any]:
    fresh = output_pairs["fresh_SZ_seed1_model2000_vs_model4000"]["arms"]
    m3 = output_pairs["causal_M3_old_vs_S1"]["arms"]
    f2 = fresh["model_2000"]
    f4 = fresh["model_4000"]
    old = m3["M3_old"]
    s1 = m3["M3_S1"]

    def fh_pos(arm: Mapping[str, Any], engine: str) -> float:
        return float(
            arm["continuous_at_strike"]["forehand"]["racket_position_error_m"][
                f"{engine}_mean"
            ]
        )

    return {
        "fresh_SZ": {
            "classification": "engine_execution_divergence_at_capture_margin_plus_isaac_virtual_metric_ceiling",
            "primary_evidence": {
                "model_4000_forehand_mujoco_hit_count": f4["outcomes"]["forehand"]["mujoco"]["n_hit"],
                "model_4000_forehand_isaac_virtual_hit_count": f4["outcomes"]["forehand"]["isaac"][
                    "n_virtual_hit"
                ],
                "model_4000_forehand_mujoco_mean_position_error_m": fh_pos(f4, "mujoco"),
                "model_4000_forehand_isaac_mean_position_error_m": fh_pos(f4, "isaac"),
                "model_2000_forehand_mujoco_mean_position_error_m": fh_pos(f2, "mujoco"),
                "model_2000_forehand_isaac_mean_position_error_m": fh_pos(f2, "isaac"),
                "frozen_virtual_capture_radius_m": 0.095,
            },
            "inference": "The same policy checkpoints reach materially different forehand strike states: model_4000 is wholly outside the MuJoCo 9.5 cm capture margin but comfortably inside it in Isaac. Isaac then scores an analytic virtual return rather than a physical collision. This is not repairable by changing a threshold and cannot be assigned to ball physics alone.",
            "unresolved_causal_axes": [
                "engine-specific ready-state mapping",
                "articulation/actuator/contact dynamics between the two simulators",
                "Isaac scorecard does not export per-question base state or physical-ball truth",
            ],
        },
        "causal_M3": {
            "classification": "signed_face_information_erased_by_isaac_analytic_scorer_plus_physical_contact_outcome_gap",
            "primary_evidence": {
                "M3_old_backhand_isaac_mean_normal_error_deg": old["continuous_at_strike"][
                    "backhand"
                ]["racket_normal_error_deg"]["isaac_mean"],
                "M3_S1_backhand_isaac_mean_normal_error_deg": s1["continuous_at_strike"][
                    "backhand"
                ]["racket_normal_error_deg"]["isaac_mean"],
                "M3_old_mujoco_counterfactual_legal_landing_count": old["outcomes"]["aggregate"][
                    "mujoco"
                ]["n_counterfactual_legal_landing"],
                "M3_old_mujoco_physical_return_count": old["outcomes"]["aggregate"]["mujoco"][
                    "n_returned"
                ],
                "M3_old_mujoco_contact_but_net_fail_count": old["cross_engine_disagreement"][
                    "mujoco_0_isaac_1_question_causes"
                ].get("mujoco_physical_contact_but_net_fail", 0),
            },
            "inference": "M3-old exposes the face-pairing defect in its signed normal metric, but the Isaac virtual contact path re-orients the normal toward the incoming ball before contact and therefore makes opposite face signs equivalent. MuJoCo physical contact preserves the distinction. The current Isaac return metric is structurally unable to reproduce this legacy face-pairing ranking.",
        },
        "common": {
            "classification": "same_question_order_but_not_same_outcome_instrument",
            "facts": [
                "Both papers have exact question_id/order alignment and no censored attempts.",
                "MuJoCo returned is based on physical contact/flight; Isaac returned is based on a capture gate plus analytic virtual contact/flight with physical_ball=false.",
                "MuJoCo and Isaac use different engine-specific ready-state modes and hashes.",
                "The fresh SZ paper holds the current zero-friction protocol exact, so plant friction cannot by itself explain its split.",
            ],
            "gate_action": "Keep both cross-engine gates open. Do not tune a score threshold to force agreement. Add a same-instrument diagnostic (Isaac physical-ball truth and/or matched analytic counterfactual in both engines), export comparable numeric ready/base state, and preserve the physical MuJoCo result as the checkpoint selector until the causal axis is isolated.",
        },
    }


def analyze(config_path: Path, artifact_root: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema_version") != 1:
        raise ForensicError("unsupported forensic input schema")
    pairs = config.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ForensicError("contract must contain exactly the fresh and causal paired papers")
    output_pairs: dict[str, Any] = {}
    for pair in pairs:
        pair_id = pair.get("id")
        arms_spec = pair.get("arms")
        if not isinstance(pair_id, str) or not isinstance(arms_spec, list) or len(arms_spec) != 2:
            raise ForensicError("each pair requires an ID and exactly two arms")
        arms = [analyze_arm(artifact_root, arm) for arm in arms_spec]
        if arms[0]["question_id_order_sha256"] != arms[1]["question_id_order_sha256"]:
            raise ForensicError(f"arms in pair {pair_id} do not share one question order")
        output_pairs[pair_id] = {
            "semantics": pair.get("semantics"),
            "question_id_order_sha256": arms[0]["question_id_order_sha256"],
            "arms": {arm["name"]: arm for arm in arms},
            f"delta_{arms[1]['name']}_minus_{arms[0]['name']}_return_count": arm_return_delta(
                arms[0], arms[1]
            ),
        }
    return {
        "schema_version": 1,
        "status": "complete_read_only_question_aligned_forensic",
        "scope": config.get("scope"),
        "input_contract": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "analyzer": {
            "path": "scripts/analyze_phase1_cross_engine_forensics.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_bindings": config.get("source_bindings"),
        "pairs": output_pairs,
        "diagnosis": diagnosis(output_pairs),
        "limitations": {
            "no_rescoring_or_threshold_change": True,
            "no_simulator_or_training_process_started": True,
            "isaac_base_state_not_exported": True,
            "ready_state_hash_schemas_are_engine_specific": True,
            "causal_M3_paper_remains_evaluation_contract_inexact": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.config.resolve(), args.artifact_root.resolve())
    if args.output.exists():
        raise ForensicError(f"no-clobber: output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[cross-engine-forensic] wrote {args.output}")
    print(f"[cross-engine-forensic] sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
