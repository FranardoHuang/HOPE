#!/usr/bin/env python3
"""Fail-closed CPU-only spatial-retarget proposal screen for air-swing motions.

The accepted v5 counterfactual screen established a canonical HOPE frame and
found zero fixed-position coverage.  This tool answers the narrower next
question: for each immutable question and source motion, is there a source
frame plus one *atomic, ground-preserving SE(2)* placement of the entire motion
whose racket state can return that ball?

This is deliberately not a capture-camera extrinsic solver.  It never changes
z, scale, handedness, joint values, or individual frames.  A proposal is not
accepted unless a content-bound candidate certificate proves the materialized
schema-2 clip, L0, L1, and full-path table/net swept-clearance gates.  Missing
certificates therefore produce proposals only and can never silently authorize
TOPP, RL, Gate3, or a robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WBT_SCRIPTS = REPO_ROOT / "hope_training" / "whole_body_tracking" / "scripts"
import sys

sys.path.insert(0, str(WBT_SCRIPTS))
from virtual_return_scorer import (  # noqa: E402
    VirtualReturnScorer,
    VirtualReturnSpec,
    load_venue_params,
)


PLAN_ID = "motion-video-spatial-retarget-20260712-v1"
EXPECTED_ASSETS = (
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
)
REQUIRED_CERTIFICATE_GATES = (
    "schema2_materialization",
    "l0_static_motion",
    "l1_vendor_mjcf_self_collision",
    "table_net_swept_clearance",
)


class RetargetError(RuntimeError):
    """Raised when a contract or runtime input is not exact enough to score."""


@dataclass(frozen=True)
class Question:
    question_id: str
    side: str
    ball_pos_w: np.ndarray
    ball_vel_w: np.ndarray
    ball_spin_w: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetargetError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise RetargetError(f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str, *, low: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetargetError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (low is not None and result < low):
        raise RetargetError(f"{label} is outside its finite range")
    return result


def _vec3(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise RetargetError(f"{label} must be a finite 3-vector")
    return result


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RetargetError(f"{label} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError:
        raise RetargetError(f"{label} must be hexadecimal") from None
    return value


def _binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RetargetError(f"{label} binding must be a mapping")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise RetargetError(f"{label}.path must be non-empty")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise RetargetError(f"{label}.bytes must be positive")
    _sha(value.get("sha256"), f"{label}.sha256")
    return value


def _verify_binding(value: dict[str, Any], label: str, override: Path | None = None) -> Path:
    raw = override or Path(value["path"])
    if override is None and not raw.is_absolute():
        raw = REPO_ROOT / raw
    path = raw.expanduser().resolve()
    if not path.is_file():
        raise RetargetError(f"{label} file is missing: {path}")
    if path.stat().st_size != value["bytes"]:
        raise RetargetError(f"{label} byte count changed")
    if sha256_file(path) != value["sha256"]:
        raise RetargetError(f"{label} SHA changed")
    return path


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _sha(expected_sha256, "expected manifest SHA")
    if sha256_file(path) != expected_sha256:
        raise RetargetError("manifest SHA does not match --expected-manifest-sha256")
    plan = _read_json(path, "manifest")
    if plan.get("schema_version") != 1 or plan.get("plan_id") != PLAN_ID:
        raise RetargetError("unknown spatial-retarget manifest schema/plan")
    if plan.get("status") != "preregistered_proposal_ready_promotion_blocked":
        raise RetargetError("manifest status must preserve the promotion blocker")
    if plan.get("cpu_only") is not True or plan.get("CUDA_VISIBLE_DEVICES") != "":
        raise RetargetError("spatial-retarget screen must remain CPU-only")
    if plan.get("real_robot_commands_authorized") is not False:
        raise RetargetError("real-robot authorization must remain false")
    if plan.get("contact_phase_truth") is not None:
        raise RetargetError("air-swing contact truth must remain null")
    if plan.get("capture_table_pose_observed") is not False:
        raise RetargetError("capture-table pose must remain explicitly unobserved")

    tool = _binding(plan.get("tool_contract"), "tool_contract")
    _verify_binding(tool, "tool_contract")

    predecessor = plan.get("predecessor")
    if not isinstance(predecessor, dict):
        raise RetargetError("predecessor binding is missing")
    _binding(predecessor.get("full_v5_result"), "predecessor.full_v5_result")
    compact = _binding(predecessor.get("compact_ledger"), "predecessor.compact_ledger")
    _verify_binding(compact, "predecessor.compact_ledger")
    _sha(predecessor.get("question_semantic_sha256"), "question semantic SHA")
    _sha(predecessor.get("frame_evidence_sha256"), "frame evidence SHA")
    if predecessor.get("exact_zero_retarget_coverage_all_assets") is not True:
        raise RetargetError("predecessor must retain exact zero-retarget evidence")

    assets = plan.get("asset_ids")
    if tuple(assets or ()) != EXPECTED_ASSETS:
        raise RetargetError("asset_ids must contain all ten motions in frozen order")
    if plan.get("candidate_priority_is_ranking_only") is not True:
        raise RetargetError("B/C priority must not remove any asset from the paper")

    transform = plan.get("transform_contract")
    if not isinstance(transform, dict):
        raise RetargetError("transform_contract is missing")
    exact_values = {
        "group": "SE2_ground_preserving_proper_rigid",
        "scope": "one_atomic_transform_for_entire_motion_question_candidate",
        "z_translation_m": 0.0,
        "scale": 1.0,
        "reflection_allowed": False,
        "per_frame_transform_allowed": False,
        "joint_edit_allowed": False,
        "capture_extrinsic_claim": False,
    }
    for key, expected in exact_values.items():
        if transform.get(key) != expected:
            raise RetargetError(f"transform_contract.{key} changed")
    yaw_grid = transform.get("yaw_grid_deg")
    if yaw_grid != [-10.0, -5.0, 0.0, 5.0, 10.0]:
        raise RetargetError("yaw grid changed")
    bounds = transform.get("station_bounds_m")
    if not isinstance(bounds, dict):
        raise RetargetError("station bounds are missing")
    for key in ("max_translation_norm", "max_abs_x", "max_abs_y"):
        _finite(bounds.get(key), f"station_bounds_m.{key}", low=0.0)
    if bounds.get("max_translation_norm") != 0.30:
        raise RetargetError("translation-norm bound changed")
    if bounds.get("max_abs_x") != 0.20 or bounds.get("max_abs_y") != 0.30:
        raise RetargetError("axis station bounds changed")

    search = plan.get("search_contract")
    if not isinstance(search, dict):
        raise RetargetError("search_contract is missing")
    if search.get("tiers") != {
        "R0": {"yaw_grid_deg": [0.0], "semantics": "translation_only"},
        "R1": {
            "yaw_grid_deg": [-10.0, -5.0, 0.0, 5.0, 10.0],
            "semantics": "bounded_yaw_plus_translation",
        },
    }:
        raise RetargetError("R0/R1 search tiers changed")
    if search.get("phase_source") != "every_v5_hard_safe_speed_eligible_source_frame":
        raise RetargetError("phase search source changed")
    if search.get("xy_alignment") != "exact_question_xy_minimum_station_translation":
        raise RetargetError("XY alignment semantics changed")
    if search.get("z_alignment") != "no_z_edit_strict_capture_radius_only":
        raise RetargetError("Z alignment semantics changed")
    if search.get("max_reported_proposals_per_motion_question_tier") != 16:
        raise RetargetError("proposal report budget changed")

    scorer = plan.get("virtual_return_contract")
    if not isinstance(scorer, dict):
        raise RetargetError("virtual_return_contract is missing")
    physics = _binding(scorer.get("physics"), "virtual return physics")
    # This tracked dependency is cheap enough to verify even when runtime assets are absent.
    _verify_binding(physics, "virtual return physics")
    scorer_dependency = _binding(
        scorer.get("scorer_dependency"), "virtual return scorer dependency"
    )
    _verify_binding(scorer_dependency, "virtual return scorer dependency")
    table = scorer.get("table_geometry")
    if not isinstance(table, dict):
        raise RetargetError("table geometry is missing")
    for key in ("surface_z_m", "net_x_m", "far_x_m", "half_width_m", "net_height_m"):
        _finite(table.get(key), f"table_geometry.{key}", low=0.0)
    if scorer.get("ball_start_for_flight") != "immutable_question_ball_position":
        raise RetargetError("flight must start at the immutable ball position")
    expected_scorer_values = {
        "capture_radius_m": 0.095,
        "minimum_approach_speed_mps": 0.3,
        "rollout_h_s": 0.01,
        "rollout_steps": 100,
    }
    for key, expected in expected_scorer_values.items():
        if scorer.get(key) != expected:
            raise RetargetError(f"virtual_return_contract.{key} changed")

    promotion = plan.get("promotion_contract")
    if not isinstance(promotion, dict):
        raise RetargetError("promotion_contract is missing")
    if tuple(promotion.get("required_candidate_certificates") or ()) != REQUIRED_CERTIFICATE_GATES:
        raise RetargetError("candidate certificate gates changed")
    if promotion.get("certificate_bundle_preregistered") is not False:
        raise RetargetError("this manifest must keep candidate-certificate promotion blocked")
    if any(promotion.get(flag) is not False for flag in ("topp", "rl", "real_robot")):
        raise RetargetError("no proposal may authorize TOPP/RL/robot")
    if promotion.get("final_arbiter") != "agibot_vendor_mujoco_gate3_gate3b_no_reset":
        raise RetargetError("vendor Gate3/Gate3B must remain the final arbiter")
    return plan


def rotation_z(yaw_deg: float) -> np.ndarray:
    angle = math.radians(float(yaw_deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _rounded_vector(value: np.ndarray) -> list[float]:
    """Canonicalize sub-picometre/libm noise before candidate content addressing."""

    return [float(round(float(item), 12)) for item in np.asarray(value).reshape(-1)]


def solve_atomic_se2(
    racket_pos: np.ndarray,
    racket_vel: np.ndarray,
    racket_normal: np.ndarray,
    question_pos: np.ndarray,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Rotate the whole motion about HOPE origin, then minimally align contact XY."""

    rotation = rotation_z(yaw_deg)
    rotated_pos = rotation @ _vec3(racket_pos, "racket_pos")
    rotated_vel = rotation @ _vec3(racket_vel, "racket_vel")
    rotated_normal = rotation @ _vec3(racket_normal, "racket_normal")
    normal_norm = float(np.linalg.norm(rotated_normal))
    if normal_norm <= 1e-12:
        raise RetargetError("racket normal is degenerate")
    rotated_normal /= normal_norm
    question = _vec3(question_pos, "question_pos")
    translation = np.asarray(
        (question[0] - rotated_pos[0], question[1] - rotated_pos[1], 0.0),
        dtype=np.float64,
    )
    mapped_pos = rotated_pos + translation
    pos_err = float(np.linalg.norm(mapped_pos - question))
    return mapped_pos, rotated_vel, rotated_normal, translation, pos_err


def _question_rows(full_result: dict[str, Any], expected_sha: str) -> list[Question]:
    schedule = full_result.get("question_schedule")
    if not isinstance(schedule, dict) or schedule.get("consumed_for_returnability") is not True:
        raise RetargetError("v5 result did not consume the question paper")
    if schedule.get("semantic_sha256") != expected_sha:
        raise RetargetError("v5 question semantic SHA changed")
    rows = schedule.get("questions")
    if not isinstance(rows, list) or len(rows) != 64:
        raise RetargetError("v5 result must contain the immutable 64 questions")
    result: list[Question] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RetargetError(f"question {index} is not a mapping")
        question_id = _sha(row.get("question_id"), f"question {index} id")
        side = row.get("side")
        if side not in ("forehand", "backhand") or question_id in seen:
            raise RetargetError(f"question {index} side/id is invalid")
        seen.add(question_id)
        result.append(
            Question(
                question_id=question_id,
                side=side,
                ball_pos_w=_vec3(row.get("ball_pos_w_m"), f"question {index} pos"),
                ball_vel_w=_vec3(row.get("ball_vel_w_mps"), f"question {index} vel"),
                ball_spin_w=_vec3(row.get("ball_spin_w_radps"), f"question {index} spin"),
            )
        )
    return result


def validate_predecessor_result(full_result: dict[str, Any], plan: dict[str, Any]) -> tuple[list[Question], list[dict[str, Any]]]:
    if full_result.get("formal_eligible") is not False or full_result.get("robot_approved") is not False:
        raise RetargetError("v5 predecessor must remain diagnostic/non-robot")
    if full_result.get("contact_phase_truth") is not None:
        raise RetargetError("v5 predecessor gained impossible contact truth")
    frame = full_result.get("frame_contract_evidence")
    if not isinstance(frame, dict) or frame.get("sha256") != plan["predecessor"]["frame_evidence_sha256"]:
        raise RetargetError("v5 frame evidence binding changed")
    if frame.get("capture_table_pose_observed") is not False:
        raise RetargetError("v5 predecessor must not claim capture-table pose")
    questions = _question_rows(full_result, plan["predecessor"]["question_semantic_sha256"])
    assets = full_result.get("assets")
    if not isinstance(assets, list) or tuple(row.get("asset_id") for row in assets) != EXPECTED_ASSETS:
        raise RetargetError("v5 full result does not contain all ten ordered assets")
    for asset in assets:
        if asset.get("selection_status") != "no_nonzero_exact_reference_coverage":
            raise RetargetError(f"{asset.get('asset_id')} predecessor zero-coverage status changed")
        frames = asset.get("per_source_frame")
        if not isinstance(frames, list) or len(frames) != asset.get("frames"):
            raise RetargetError(f"{asset.get('asset_id')} lacks complete per-frame evidence")
        _binding(asset.get("input"), f"{asset.get('asset_id')} source motion")
    return questions, assets


def _station_within_bounds(translation: np.ndarray, bounds: dict[str, Any]) -> bool:
    x, y = float(translation[0]), float(translation[1])
    return bool(
        math.hypot(x, y) <= float(bounds["max_translation_norm"]) + 1e-12
        and abs(x) <= float(bounds["max_abs_x"]) + 1e-12
        and abs(y) <= float(bounds["max_abs_y"]) + 1e-12
    )


def _return_margin(outcome: Any, scorer: VirtualReturnScorer, pos_err: float) -> float:
    if not outcome.landed_ok:
        return -math.inf
    x, y = float(outcome.landing_xy[0]), float(outcome.landing_xy[1])
    return float(
        min(
            scorer.spec.capture_radius - pos_err,
            float(outcome.net_z) - scorer.net_clear_center_z,
            x - scorer.spec.net_x,
            scorer.spec.far_x - x,
            scorer.spec.half_width - abs(y),
        )
    )


def candidate_id(candidate: dict[str, Any]) -> str:
    identity = {
        key: candidate[key]
        for key in (
            "asset_id",
            "source_motion_sha256",
            "predecessor_full_result_sha256",
            "question_id",
            "tier",
            "frame",
            "yaw_deg",
            "translation_w_m",
        )
    }
    return canonical_sha256({"kind": "motion-spatial-retarget-candidate-v1", **identity})


def search_motion_question(
    asset: dict[str, Any],
    question: Question,
    tier: str,
    plan: dict[str, Any],
    scorer: VirtualReturnScorer,
) -> list[dict[str, Any]]:
    if asset.get("effective_side_after_verified_mirror", asset.get("side")) != question.side:
        return []
    yaw_grid = plan["search_contract"]["tiers"][tier]["yaw_grid_deg"]
    bounds = plan["transform_contract"]["station_bounds_m"]
    proposals: list[dict[str, Any]] = []
    for frame_row in asset["per_source_frame"]:
        if frame_row.get("hard_safe") is not True or frame_row.get("candidate_eligible") is not True:
            continue
        for yaw_deg in yaw_grid:
            mapped_pos, velocity, normal, translation, pos_err = solve_atomic_se2(
                _vec3(frame_row.get("racket_site_pos_w_m"), "racket site position"),
                _vec3(frame_row.get("racket_site_vel_w_mps"), "racket site velocity"),
                _vec3(frame_row.get("racket_face_normal_w"), "racket face normal"),
                question.ball_pos_w,
                yaw_deg,
            )
            if not _station_within_bounds(translation, bounds):
                continue
            # Flight starts at the immutable ball, not at an air-swing/capture-camera point.
            outcome = scorer.score(
                ball_vel=question.ball_vel_w,
                ball_spin=question.ball_spin_w,
                racket_pos=question.ball_pos_w,
                racket_vel=velocity,
                racket_normal=normal,
                pos_err=pos_err,
            )
            if not outcome.landed_ok:
                continue
            proposal = {
                "asset_id": asset["asset_id"],
                "source_motion_sha256": asset["input"]["sha256"],
                "predecessor_full_result_sha256": plan["predecessor"]["full_v5_result"][
                    "sha256"
                ],
                "question_id": question.question_id,
                "side": question.side,
                "tier": tier,
                "frame": int(frame_row["frame"]),
                "phase": float(frame_row["phase"]),
                "yaw_deg": float(yaw_deg),
                "translation_w_m": _rounded_vector(translation),
                "mapped_racket_pos_w_m": _rounded_vector(mapped_pos),
                "vertical_position_error_m": float(round(pos_err, 12)),
                "translation_norm_m": float(round(float(np.linalg.norm(translation[:2])), 12)),
                "return_margin_m": float(round(_return_margin(outcome, scorer, pos_err), 12)),
                "landing_xy_m": _rounded_vector(outcome.landing_xy),
                "net_center_z_m": float(round(float(outcome.net_z), 12)),
                "source_dense_racket_body_clearance_m": float(
                    frame_row["dense_racket_body_clearance_m"]
                ),
                "transform_semantics": "atomic_whole_motion_ground_preserving_SE2",
                "capture_extrinsic_claim": False,
            }
            proposal["candidate_id"] = candidate_id(proposal)
            proposals.append(proposal)
    proposals.sort(
        key=lambda row: (
            row["translation_norm_m"],
            abs(row["yaw_deg"]),
            -row["return_margin_m"],
            -row["source_dense_racket_body_clearance_m"],
            row["frame"],
            row["candidate_id"],
        )
    )
    return proposals


def validate_certificate(candidate: dict[str, Any], value: Any) -> tuple[bool, str]:
    """Return acceptance only for an exact candidate-specific four-gate certificate."""

    if not isinstance(value, dict):
        return False, "missing_candidate_certificate"
    if value.get("candidate_id") != candidate["candidate_id"]:
        return False, "certificate_candidate_id_mismatch"
    if value.get("atomic_transform_applied_to_entire_motion") is not True:
        return False, "certificate_did_not_apply_atomic_whole_motion_transform"
    if value.get("capture_extrinsic_claim") is not False:
        return False, "certificate_claimed_capture_extrinsic"
    gates = value.get("gates")
    if not isinstance(gates, dict):
        return False, "certificate_gates_missing"
    for name in REQUIRED_CERTIFICATE_GATES:
        gate = gates.get(name)
        if not isinstance(gate, dict):
            return False, f"certificate_gate_missing:{name}"
        if gate.get("verdict") != "PASS":
            return False, f"certificate_gate_not_pass:{name}"
        binding = gate.get("report")
        if not isinstance(binding, dict):
            return False, f"certificate_report_missing:{name}"
        try:
            _binding(binding, f"certificate {name} report")
        except RetargetError as exc:
            return False, str(exc)
    table_gate = gates["table_net_swept_clearance"]
    if table_gate.get("zero_hard_failures") is not True:
        return False, "table_net_certificate_has_hard_failure"
    minimum = table_gate.get("minimum_clearance_m")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(float(minimum)):
        return False, "table_net_minimum_clearance_missing"
    if float(minimum) < 0.005:
        return False, "table_net_minimum_clearance_below_5mm"
    return True, "accepted_candidate_certificate"


def _certificate_index(bundle: dict[str, Any] | None) -> dict[str, Any]:
    if bundle is None:
        return {}
    if bundle.get("schema_version") != 1 or bundle.get("status") != "candidate_certificates_complete":
        raise RetargetError("certificate bundle schema/status is invalid")
    rows = bundle.get("certificates")
    if not isinstance(rows, list):
        raise RetargetError("certificate bundle certificates must be a list")
    result: dict[str, Any] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RetargetError(f"certificate {index} is not a mapping")
        key = _sha(row.get("candidate_id"), f"certificate {index} candidate_id")
        if key in result:
            raise RetargetError(f"duplicate certificate for {key}")
        result[key] = row
    return result


def build_scorer(plan: dict[str, Any]) -> VirtualReturnScorer:
    contract = plan["virtual_return_contract"]
    physics_path = _verify_binding(contract["physics"], "virtual return physics")
    geometry = contract["table_geometry"]
    return VirtualReturnScorer(
        load_venue_params(str(physics_path)),
        VirtualReturnSpec(
            table_surface_z=float(geometry["surface_z_m"]),
            net_x=float(geometry["net_x_m"]),
            far_x=float(geometry["far_x_m"]),
            half_width=float(geometry["half_width_m"]),
            net_height=float(geometry["net_height_m"]),
            capture_radius=float(contract["capture_radius_m"]),
            min_approach_speed=float(contract["minimum_approach_speed_mps"]),
            rollout_h=float(contract["rollout_h_s"]),
            rollout_steps=int(contract["rollout_steps"]),
        ),
    )


def run_screen(
    plan_path: Path,
    expected_manifest_sha256: str,
    *,
    predecessor_override: Path | None,
    certificate_bundle_path: Path | None,
) -> dict[str, Any]:
    plan = validate_manifest(plan_path, expected_manifest_sha256)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    predecessor_path = _verify_binding(
        plan["predecessor"]["full_v5_result"],
        "predecessor full v5 result",
        predecessor_override,
    )
    full_result = _read_json(predecessor_path, "predecessor full v5 result")
    questions, assets = validate_predecessor_result(full_result, plan)
    certificate_bundle = (
        _read_json(certificate_bundle_path, "certificate bundle")
        if certificate_bundle_path is not None
        else None
    )
    if certificate_bundle is not None and plan["promotion_contract"]["certificate_bundle_preregistered"] is not True:
        raise RetargetError(
            "candidate certificates were not preregistered in this manifest; "
            "amend and content-bind a new manifest before promotion"
        )
    certificates = _certificate_index(certificate_bundle)
    scorer = build_scorer(plan)
    report_limit = int(
        plan["search_contract"]["max_reported_proposals_per_motion_question_tier"]
    )
    cells: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    proposal_count = 0
    for asset in assets:
        for question in questions:
            if asset.get("effective_side_after_verified_mirror", asset.get("side")) != question.side:
                continue
            for tier in ("R0", "R1"):
                proposals = search_motion_question(asset, question, tier, plan, scorer)
                proposal_count += len(proposals)
                accepted_rows: list[dict[str, Any]] = []
                for candidate in proposals:
                    ok, reason = validate_certificate(
                        candidate, certificates.get(candidate["candidate_id"])
                    )
                    if ok:
                        accepted_row = {**candidate, "certificate_status": reason}
                        accepted_rows.append(accepted_row)
                        accepted.append(accepted_row)
                cells.append(
                    {
                        "asset_id": asset["asset_id"],
                        "question_id": question.question_id,
                        "side": question.side,
                        "tier": tier,
                        "proposal_count_before_certificate": len(proposals),
                        "reported_proposals": [
                            {
                                **row,
                                "promotion_status": validate_certificate(
                                    row, certificates.get(row["candidate_id"])
                                )[1],
                            }
                            for row in proposals[:report_limit]
                        ],
                        "accepted_candidate_ids": [row["candidate_id"] for row in accepted_rows],
                    }
                )
    accepted_ids = {row["candidate_id"] for row in accepted}
    return {
        "schema_version": 1,
        "status": (
            "complete_with_certified_candidates"
            if accepted
            else "complete_proposals_only_promotion_blocked"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "formal_eligible": False,
        "robot_approved": False,
        "contact_phase_truth": None,
        "capture_table_pose_observed": False,
        "manifest": {
            "path": str(plan_path.resolve()),
            "bytes": plan_path.stat().st_size,
            "sha256": expected_manifest_sha256,
        },
        "predecessor": {
            "path": str(predecessor_path),
            "bytes": predecessor_path.stat().st_size,
            "sha256": sha256_file(predecessor_path),
        },
        "question_semantic_sha256": plan["predecessor"]["question_semantic_sha256"],
        "asset_count": len(assets),
        "question_count": len(questions),
        "proposal_count_before_certificate": proposal_count,
        "certified_candidate_count": len(accepted_ids),
        "cells": cells,
        "accepted_candidates": accepted,
        "claims": {
            "spatial_retarget": "bounded atomic whole-motion SE2 proposal screen",
            "capture_extrinsic": None,
            "schema2_L0_L1_table_net": (
                "candidate_specific_certificates_bound"
                if accepted
                else "missing_or_no_matching_pass_certificates"
            ),
            "topp_authorized": False,
            "rl_authorized": False,
            "real_robot_authorized": False,
            "final_arbiter": "agibot_vendor_mujoco_gate3_gate3b_no_reset",
        },
    }


def atomic_write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RetargetError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise RetargetError(f"refusing concurrent overwrite of result: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the tracked preregistration only")
    screen = subparsers.add_parser("screen", help="run the proposal screen without GPU")
    screen.add_argument("--predecessor-result", type=Path)
    screen.add_argument("--candidate-certificates", type=Path)
    screen.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_manifest(args.manifest, args.expected_manifest_sha256)
        if args.command == "validate":
            print("spatial-retarget preregistration: PASS (proposal ready, promotion blocked)")
            return 0
        result = run_screen(
            args.manifest,
            args.expected_manifest_sha256,
            predecessor_override=args.predecessor_result,
            certificate_bundle_path=args.candidate_certificates,
        )
        atomic_write_new(args.output, result)
        print(
            f"spatial-retarget screen: {result['status']}; "
            f"proposals={result['proposal_count_before_certificate']} "
            f"certified={result['certified_candidate_count']} output={args.output}"
        )
        return 0
    except RetargetError as exc:
        print(f"spatial-retarget screen: FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
