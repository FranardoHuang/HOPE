#!/usr/bin/env python3
"""Content-addressed CPU-only phase and robot-safety screen for one or more A3 GMR clips.

The input contract is deliberately narrower than a general motion converter:

* inputs are explicit files listed in one manifest (there is no directory scan);
* every input, grounding report, MJCF, physics file and code dependency is SHA-256 bound;
* the input is a *grounded* GMR pickle (30 Hz, ``root_rot`` xyzw, 31 A3 joints);
* MuJoCo FK at the official ``right_racket`` site supplies paddle centre/orientation;
* ``mj_differentiatePos`` plus ``mj_objectVelocity`` supplies the site velocity;
* safety is sampled densely between GMR rows before any returnability score is considered;
* any robot self-interpenetration, ground penetration, or critical racket/handle clearance
  below the manifest's hard threshold excludes both endpoints of that interval;
* virtual-return scores use the repository's NumPy Isaac-metric specification and a frozen,
  deterministic venue-box question schedule.

Air swings contain no observed ball contact.  Consequently this tool always writes
``contact_phase_truth=null``.  Its top frames are offline *training phase candidates*, not a
claim about where the performer contacted a ball.  Exact-position coverage is a conservative
zero-retarget reference-path screen; it is not policy coverage, planner reachability, table/net
swept-volume clearance, dynamics, balance, TOPP, schema-2 approval, or robot approval.

The manifest may be either a blocked preregistration (``execution_ready=false``), which
``validate`` accepts but ``run`` rejects, or a ready runtime manifest containing exact grounded
input/report bindings.  This makes the next lane queueable before canonical grounding exists
without inventing the future output SHA values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[1]
WBT_SCRIPTS = REPO / "hope_training" / "whole_body_tracking" / "scripts"


def _runtime_dependency(filename: str, source_path: Path) -> Path:
    """Prefer a sibling control-bundle dependency, else the tracked source-tree path."""

    sibling = SCRIPT_DIR / filename
    return sibling if sibling.is_file() else source_path


GROUNDING_DEPENDENCY = _runtime_dependency("ground_gmr_pkl.py", REPO / "scripts/ground_gmr_pkl.py")
SELF_COLLISION_DEPENDENCY = _runtime_dependency(
    "audit_self_collision.py", WBT_SCRIPTS / "audit_self_collision.py"
)
MOTION_AUDIT_DEPENDENCY = _runtime_dependency("audit_motion_npz.py", WBT_SCRIPTS / "audit_motion_npz.py")
VIRTUAL_RETURN_DEPENDENCY = _runtime_dependency(
    "virtual_return_scorer.py", WBT_SCRIPTS / "virtual_return_scorer.py"
)
for dependency_dir in {
    GROUNDING_DEPENDENCY.parent,
    SELF_COLLISION_DEPENDENCY.parent,
    MOTION_AUDIT_DEPENDENCY.parent,
    VIRTUAL_RETURN_DEPENDENCY.parent,
}:
    sys.path.insert(0, str(dependency_dir))

from ground_gmr_pkl import (  # noqa: E402
    A3_GMR_JOINT_NAMES,
    bind_model,
    frame_clearances,
    load_pickle,
    validate_joint_ranges,
    validate_payload,
)
from virtual_return_scorer import (  # noqa: E402
    VirtualReturnScorer,
    VirtualReturnSpec,
    load_venue_params,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
ALGORITHM = "sha256-counter-balanced-side-box-isotropic-spin-rounded12-v1"
BODY_SHAPE_CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"
RACKET_SITE = "right_racket"
RACKET_FACE_AXIS = 1  # vendor site local +Y = red/forehand face normal
RACKET_GEOMS = ("right_racket_collision", "right_racket_handle_collision")
MOUNT_NORMAL_SIGN_PER_SIDE = {"forehand": 1.0, "backhand": -1.0}


class ScreenError(ValueError):
    """Fail-closed contract or screening error."""


@dataclass(frozen=True)
class Question:
    question_id: str
    side: str
    ball_pos_w: np.ndarray
    ball_vel_w: np.ndarray
    ball_spin_w: np.ndarray


@dataclass(frozen=True)
class ModelContract:
    mujoco: Any
    binding: Any
    site_id: int
    racket_geom_ids: tuple[int, ...]
    clearance_groups: dict[str, tuple[int, ...]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ScreenError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _require_number(value: Any, label: str, *, low: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreenError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (low is not None and result < low):
        raise ScreenError(f"{label} must be finite" + (f" and >= {low}" if low is not None else ""))
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreenError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ScreenError(f"{label} must be a JSON mapping")
    return value


def _binding(value: Any, label: str, *, require_exists: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ScreenError(f"{label} must contain a string path")
    _require_sha(value.get("sha256"), f"{label}.sha256")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise ScreenError(f"{label}.bytes must be a positive integer")
    if require_exists:
        _verify_file(value, label)
    return value


def _verify_file(binding: dict[str, Any], label: str) -> Path:
    path = Path(binding["path"]).expanduser().resolve()
    if not path.is_file():
        raise ScreenError(f"{label} is missing: {path}")
    if path.stat().st_size != binding["bytes"]:
        raise ScreenError(
            f"{label} bytes {path.stat().st_size} != bound {binding['bytes']}: {path}"
        )
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise ScreenError(f"{label} sha256 {actual} != bound {binding['sha256']}: {path}")
    return path


def _validate_box(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ScreenError(f"{label} must be a two-number list")
    lo = _require_number(value[0], f"{label}[0]")
    hi = _require_number(value[1], f"{label}[1]")
    if not lo < hi:
        raise ScreenError(f"{label} must be strictly increasing")
    return lo, hi


def _validate_proper_rigid_matrix(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ScreenError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ScreenError(f"{label} bottom row must be [0,0,0,1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
        raise ScreenError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9, rel_tol=0.0):
        raise ScreenError(f"{label} must be a proper rigid rotation (det=+1)")
    return transform


def _crosscheck_frame_contract_evidence(
    plan: dict[str, Any], evidence_path: Path
) -> None:
    evidence = _read_json(evidence_path, "frame-contract evidence")
    if evidence.get("status") != "complete_verified_per_asset_hope_frame_and_not_mirrored":
        raise ScreenError("frame-contract evidence is not a completed verified result")
    frame = evidence.get("frame_contract")
    mirror = evidence.get("mirror_contract")
    if not isinstance(frame, dict) or not isinstance(mirror, dict):
        raise ScreenError("frame-contract evidence lacks frame/mirror records")
    if (
        frame.get("status") != "verified"
        or frame.get("transform_scope") != "per_asset"
        or frame.get("gmr_world_to_hope_table_transform_verified") is not True
    ):
        raise ScreenError("frame-contract evidence does not verify per-asset transforms")
    if (
        frame.get("capture_table_pose_observed") is not False
        or frame.get("target_table_pose_semantics")
        != "canonical_counterfactual_HOPE_virtual_table_relative_to_robot_origin_not_capture_extrinsic"
    ):
        raise ScreenError("frame evidence must retain counterfactual-table semantics")
    if mirror.get("status") != "verified_not_mirrored" or mirror.get("side_swap_required") is not False:
        raise ScreenError("frame evidence does not verify no-mirror/no-side-swap semantics")
    semantics = evidence.get("returnability_semantics")
    if not isinstance(semantics, dict) or semantics.get("real_capture_table", {}).get("coverage") is not None:
        raise ScreenError("frame evidence must keep real-capture returnability null")
    eligibility = evidence.get("eligibility")
    if not isinstance(eligibility, dict) or eligibility.get(
        "immutable_64_question_returnability_phase_screen"
    ) is not True:
        raise ScreenError("frame evidence does not authorize the diagnostic question paper")

    expected_ids = plan["expected_asset_ids"]
    rows = evidence.get("assets")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows] != expected_ids:
        raise ScreenError("frame evidence must list every expected asset in exact order")
    by_id = {row["asset_id"]: row for row in rows}
    transforms = plan["frame_contract"]["per_asset_gmr_world_to_hope_matrix_4x4"]
    input_by_id = {row["asset_id"]: row for row in plan["inputs"]}
    for asset_id in expected_ids:
        row = by_id[asset_id]
        if row.get("mirror_status") != "verified_not_mirrored" or row.get("side_swap_required") is not False:
            raise ScreenError(f"frame evidence mirror status changed for {asset_id}")
        if row.get("transform", {}).get("matrix_4x4") != transforms[asset_id]:
            raise ScreenError(f"frame evidence transform changed for {asset_id}")
        if row.get("grounded_gmr", {}).get("sha256") != input_by_id[asset_id]["input"]["sha256"]:
            raise ScreenError(f"frame evidence GMR SHA changed for {asset_id}")

    table = evidence.get("source_semantics", {}).get("target_table")
    schedule_table = plan["question_schedule"]["table_geometry"]
    if not isinstance(table, dict):
        raise ScreenError("frame evidence lacks target table geometry")
    expected_table = {
        "surface_z_m": schedule_table["surface_z_m"],
        "net_x_m": schedule_table["net_x_m"],
        "far_edge_x_m": schedule_table["far_x_m"],
        "half_width_m": schedule_table["half_width_m"],
        "net_height_m": schedule_table["net_height_m"],
    }
    for key, expected in expected_table.items():
        if not math.isclose(float(table.get(key, math.nan)), float(expected), abs_tol=1e-12, rel_tol=0.0):
            raise ScreenError(f"frame evidence table {key} changed")


def validate_manifest(
    path: Path,
    expected_sha256: str,
    *,
    require_ready: bool,
    verify_files: bool,
) -> dict[str, Any]:
    _require_sha(expected_sha256, "--expected-manifest-sha256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ScreenError(f"manifest sha256 {actual} != expected {expected_sha256}")
    plan = _read_json(path, "phase+safety manifest")
    if plan.get("schema_version") != 1:
        raise ScreenError("manifest schema_version must be 1")
    if plan.get("plan_id") != "motion-video-gmr-phase-safety-20260711-v1":
        raise ScreenError("unexpected plan_id")
    ready = plan.get("execution_ready")
    if not isinstance(ready, bool):
        raise ScreenError("execution_ready must be boolean")
    if require_ready and not ready:
        raise ScreenError(
            "canonical grounding is still a prerequisite: this preregistration has "
            "execution_ready=false and cannot be run"
        )
    if plan.get("cpu_only") is not True or plan.get("CUDA_VISIBLE_DEVICES") != "":
        raise ScreenError("screen must be CPU-only with CUDA_VISIBLE_DEVICES empty")
    if plan.get("real_robot_commands_authorized") is not False:
        raise ScreenError("real_robot_commands_authorized must be false")
    if plan.get("input_mode") != "explicit_manifest_only_no_directory_scan":
        raise ScreenError("input_mode must forbid directory scanning")
    if plan.get("contact_phase_truth") is not None:
        raise ScreenError("air-swing contact_phase_truth must remain null")
    if plan.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise ScreenError(f"body_shape_contract must be {BODY_SHAPE_CONTRACT}")
    grounding_manifest_binding = _binding(
        plan.get("canonical_grounding_result_manifest"),
        "canonical_grounding_result_manifest",
        require_exists=verify_files,
    )

    tool_contract = plan.get("tool_contract")
    if not isinstance(tool_contract, dict):
        raise ScreenError("tool_contract must be a mapping")
    expected_tools = {
        "screen": Path(__file__).resolve(),
        "grounding_dependency": GROUNDING_DEPENDENCY,
        "self_collision_dependency": SELF_COLLISION_DEPENDENCY,
        "motion_audit_dependency": MOTION_AUDIT_DEPENDENCY,
        "virtual_return_dependency": VIRTUAL_RETURN_DEPENDENCY,
    }
    for name, expected_path in expected_tools.items():
        item = _binding(tool_contract.get(name), f"tool_contract.{name}", require_exists=False)
        if Path(item["path"]).name != expected_path.name:
            raise ScreenError(f"tool_contract.{name} basename mismatch")
        if verify_files:
            if not expected_path.is_file() or expected_path.stat().st_size != item["bytes"]:
                raise ScreenError(f"tool_contract.{name} local bytes mismatch")
            if sha256_file(expected_path) != item["sha256"]:
                raise ScreenError(f"tool_contract.{name} local SHA mismatch")

    mjcf = _binding(plan.get("mjcf"), "mjcf", require_exists=verify_files)
    physics = _binding(plan.get("physics"), "physics", require_exists=verify_files)
    venue_distribution = _binding(
        plan.get("venue_distribution"), "venue_distribution", require_exists=verify_files
    )
    if Path(mjcf["path"]).name != "a3_pingpong.xml":
        raise ScreenError("mjcf must bind a3_pingpong.xml")
    if Path(physics["path"]).name != "ball_physics_venue.yaml":
        raise ScreenError("physics must bind ball_physics_venue.yaml")
    if Path(venue_distribution["path"]).name != "incoming_ball_venue.yaml":
        raise ScreenError("venue_distribution must bind incoming_ball_venue.yaml")

    dense = plan.get("dense_safety_contract")
    if not isinstance(dense, dict):
        raise ScreenError("dense_safety_contract must be a mapping")
    if not isinstance(dense.get("substeps_per_source_interval"), int) or dense["substeps_per_source_interval"] < 2:
        raise ScreenError("dense substeps must be an integer >=2")
    for field in (
        "ground_penetration_tolerance_m",
        "self_collision_penetration_tolerance_m",
        "hard_racket_body_clearance_m",
        "warning_racket_body_clearance_m",
    ):
        _require_number(dense.get(field), f"dense_safety_contract.{field}", low=0.0)
    if dense["warning_racket_body_clearance_m"] < dense["hard_racket_body_clearance_m"]:
        raise ScreenError("warning clearance must be >= hard clearance")
    groups = dense.get("racket_body_clearance_groups")
    if not isinstance(groups, dict) or not groups:
        raise ScreenError("racket_body_clearance_groups must be a non-empty mapping")
    seen_geoms: set[str] = set()
    for name, geom_names in groups.items():
        if not SAFE_ID_RE.fullmatch(str(name)) or not isinstance(geom_names, list) or not geom_names:
            raise ScreenError(f"invalid clearance group {name!r}")
        if not all(isinstance(value, str) and value for value in geom_names):
            raise ScreenError(f"clearance group {name!r} contains an invalid geom name")
        duplicates = seen_geoms.intersection(geom_names)
        if duplicates:
            raise ScreenError(f"clearance geoms occur in multiple groups: {sorted(duplicates)}")
        seen_geoms.update(geom_names)

    schedule = plan.get("question_schedule")
    if not isinstance(schedule, dict) or schedule.get("algorithm") != ALGORITHM:
        raise ScreenError(f"question_schedule.algorithm must be {ALGORITHM}")
    if not isinstance(schedule.get("seed"), str) or not schedule["seed"]:
        raise ScreenError("question_schedule.seed must be a non-empty string")
    if not isinstance(schedule.get("count_per_side"), int) or schedule["count_per_side"] <= 0:
        raise ScreenError("question_schedule.count_per_side must be positive")
    if schedule.get("sides") != ["forehand", "backhand"]:
        raise ScreenError("question_schedule.sides must be ['forehand','backhand']")
    for field in (
        "venue_contact_x_from_net_m",
        "venue_contact_z_above_table_m",
        "incoming_vx_mps",
        "incoming_vy_mps",
        "incoming_vz_mps",
        "spin_magnitude_radps",
        "forehand_contact_y_m",
        "backhand_contact_y_m",
    ):
        _validate_box(schedule.get(field), f"question_schedule.{field}")
    table = schedule.get("table_geometry")
    if not isinstance(table, dict):
        raise ScreenError("question_schedule.table_geometry must be a mapping")
    for field in ("surface_z_m", "net_x_m", "far_x_m", "half_width_m", "net_height_m"):
        _require_number(table.get(field), f"table_geometry.{field}")
    if not table["net_x_m"] < table["far_x_m"] or table["half_width_m"] <= 0:
        raise ScreenError("table geometry is invalid")
    questions = build_questions(schedule)
    question_rows = [
        {
            "question_id": question.question_id,
            "side": question.side,
            "ball_pos_w_m": question.ball_pos_w.tolist(),
            "ball_vel_w_mps": question.ball_vel_w.tolist(),
            "ball_spin_w_radps": question.ball_spin_w.tolist(),
        }
        for question in questions
    ]
    expected_question_sha = _require_sha(
        schedule.get("expected_semantic_sha256"),
        "question_schedule.expected_semantic_sha256",
    )
    actual_question_sha = canonical_sha256(question_rows)
    if actual_question_sha != expected_question_sha:
        raise ScreenError(
            f"question schedule semantic SHA {actual_question_sha} != bound "
            f"{expected_question_sha}"
        )
    if verify_files:
        _crosscheck_venue_schedule(schedule, Path(venue_distribution["path"]).expanduser().resolve())

    phase = plan.get("phase_selection_contract")
    if not isinstance(phase, dict):
        raise ScreenError("phase_selection_contract must be a mapping")
    _require_number(phase.get("minimum_racket_speed_mps"), "minimum_racket_speed_mps", low=0.0)
    if phase.get("rank_order") != [
        "exact_return_count",
        "intrinsic_return_count",
        "median_exact_return_margin_m",
        "dense_racket_body_clearance_m",
        "earlier_frame",
    ]:
        raise ScreenError("phase_selection_contract.rank_order changed")
    frame_contract = plan.get("frame_contract")
    if not isinstance(frame_contract, dict) or not isinstance(
        frame_contract.get("returnability_enabled"), bool
    ):
        raise ScreenError("frame_contract.returnability_enabled must be boolean")
    if frame_contract["returnability_enabled"]:
        if frame_contract.get("gmr_world_to_hope_table_transform_verified") is not True:
            raise ScreenError("returnability requires a verified GMR-world -> HOPE table transform")
        if frame_contract.get("mirror_status") not in ("verified_not_mirrored", "verified_mirrored"):
            raise ScreenError("returnability requires verified mirror status")
        transform_scope = frame_contract.get("transform_scope", "global")
        if transform_scope == "global":
            _validate_proper_rigid_matrix(
                frame_contract.get("gmr_world_to_hope_matrix_4x4"), "frame transform"
            )
        elif transform_scope == "per_asset":
            if frame_contract.get("gmr_world_to_hope_matrix_4x4") is not None:
                raise ScreenError("per-asset transform scope must not also provide a global matrix")
            transforms = frame_contract.get("per_asset_gmr_world_to_hope_matrix_4x4")
            if not isinstance(transforms, dict) or not transforms:
                raise ScreenError("per-asset transform scope requires a non-empty transform mapping")
            _binding(
                plan.get("frame_contract_evidence"),
                "frame_contract_evidence",
                require_exists=verify_files,
            )
        else:
            raise ScreenError(f"unsupported frame transform_scope {transform_scope!r}")
    else:
        if frame_contract.get("gmr_world_to_hope_table_transform_verified") is not False:
            raise ScreenError("blocked returnability must explicitly mark frame transform unverified")
        if frame_contract.get("gmr_world_to_hope_matrix_4x4") is not None:
            raise ScreenError("blocked returnability must not invent a frame transform")
        blockers = frame_contract.get("blockers")
        if not isinstance(blockers, list) or not blockers or not all(
            isinstance(value, str) and value.strip() for value in blockers
        ):
            raise ScreenError("blocked returnability requires explicit frame-contract blockers")

    expected_ids = plan.get("expected_asset_ids")
    if (
        not isinstance(expected_ids, list)
        or len(expected_ids) != 10
        or len(set(expected_ids)) != len(expected_ids)
        or not all(isinstance(value, str) and SAFE_ID_RE.fullmatch(value) for value in expected_ids)
    ):
        raise ScreenError("expected_asset_ids must be ten unique safe ids")
    if frame_contract["returnability_enabled"] and frame_contract.get("transform_scope") == "per_asset":
        transforms = frame_contract["per_asset_gmr_world_to_hope_matrix_4x4"]
        if list(transforms) != expected_ids:
            raise ScreenError("per-asset frame transforms must list every expected asset in order")
        for asset_id in expected_ids:
            _validate_proper_rigid_matrix(transforms[asset_id], f"frame transform {asset_id}")

    inputs = plan.get("inputs")
    if not isinstance(inputs, list):
        raise ScreenError("inputs must be a list")
    if not ready:
        if inputs:
            raise ScreenError("blocked preregistration must not guess future input SHA bindings")
    else:
        if [row.get("asset_id") for row in inputs if isinstance(row, dict)] != expected_ids:
            raise ScreenError("ready inputs must exactly match expected_asset_ids in order")
        input_paths: set[str] = set()
        report_paths: set[str] = set()
        for index, row in enumerate(inputs):
            _validate_input_row(row, index, verify_files=verify_files)
            input_path = str(Path(row["input"]["path"]).expanduser().resolve())
            report_path = str(Path(row["grounding_report"]["path"]).expanduser().resolve())
            if input_path in input_paths or report_path in report_paths:
                raise ScreenError("ready inputs contain duplicate input/report paths")
            input_paths.add(input_path)
            report_paths.add(report_path)
        if verify_files:
            grounding_manifest_path = _verify_file(
                grounding_manifest_binding, "canonical_grounding_result_manifest"
            )
            _crosscheck_canonical_grounding_result(plan, grounding_manifest_path)
            if frame_contract["returnability_enabled"] and frame_contract.get("transform_scope") == "per_asset":
                evidence_path = _verify_file(plan["frame_contract_evidence"], "frame_contract_evidence")
                _crosscheck_frame_contract_evidence(plan, evidence_path)

    libraries = plan.get("libraries")
    if not isinstance(libraries, dict) or not libraries:
        raise ScreenError("libraries must be a non-empty mapping")
    for name, members in libraries.items():
        if not SAFE_ID_RE.fullmatch(str(name)) or not isinstance(members, list) or not members:
            raise ScreenError(f"invalid library {name!r}")
        if len(set(members)) != len(members) or any(member not in expected_ids for member in members):
            raise ScreenError(f"library {name!r} contains duplicate/unknown assets")
    comparisons = plan.get("library_comparisons")
    if not isinstance(comparisons, list):
        raise ScreenError("library_comparisons must be a list")
    for item in comparisons:
        if (
            not isinstance(item, dict)
            or item.get("baseline") not in libraries
            or item.get("candidate") not in libraries
        ):
            raise ScreenError(f"invalid library comparison {item!r}")

    output = plan.get("output_contract")
    if not isinstance(output, dict) or not isinstance(output.get("result"), str):
        raise ScreenError("output_contract.result must be an explicit path")
    if output.get("no_clobber") is not True:
        raise ScreenError("output contract must be no-clobber")
    if require_ready and (Path(output["result"]).exists() or Path(output["result"]).is_symlink()):
        raise ScreenError(f"no-clobber result already exists: {output['result']}")
    return plan


def _crosscheck_canonical_grounding_result(plan: dict[str, Any], path: Path) -> None:
    """Prove every ready input/report row came from the bound canonical-ground result."""

    result = _read_json(path, "canonical grounding result manifest")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "complete_diagnostic_canonical_grounding"
        or result.get("body_shape_contract") != BODY_SHAPE_CONTRACT
    ):
        raise ScreenError("canonical grounding result manifest status/lineage mismatch")
    rows = result.get("results")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows] != plan["expected_asset_ids"]:
        raise ScreenError("canonical grounding result asset order mismatch")
    by_id = {row["asset_id"]: row for row in rows}
    for runtime in plan["inputs"]:
        asset_id = runtime["asset_id"]
        source = by_id[asset_id]
        if source.get("status") != "complete_diagnostic_canonical_grounding":
            raise ScreenError(f"canonical grounding source {asset_id} is not complete")
        if source.get("formal_eligible") is not False:
            raise ScreenError(f"canonical grounding source {asset_id} must remain formal-ineligible")
        if source.get("output") != runtime["input"]:
            raise ScreenError(f"ready input binding for {asset_id} differs from grounding result")
        if source.get("report") != runtime["grounding_report"]:
            raise ScreenError(f"ready grounding-report binding for {asset_id} differs from result")
        if source.get("structure", {}).get("frames") != runtime["frames"]:
            raise ScreenError(f"ready frame count for {asset_id} differs from grounding result")
    processing = result.get("processing_contract", {})
    model = processing.get("mjcf", {})
    compiled = processing.get("compiled_collision_contract", {})
    if model.get("sha256") != plan["mjcf"]["sha256"]:
        raise ScreenError("canonical grounding result MJCF SHA mismatch")
    if compiled.get("expected_sha256") != plan["mjcf"]["compiled_kinematic_collision_sha256"]:
        raise ScreenError("canonical grounding result compiled collision SHA mismatch")


def _crosscheck_venue_schedule(schedule: dict[str, Any], path: Path) -> None:
    """Bind the diagnostic box paper to the exact existing venue-yaml fields it mirrors."""

    try:
        import yaml

        source = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ScreenError(f"cannot parse venue distribution {path}: {exc}") from None
    try:
        contact = source["at_strike"]["contact_pos_q10_q90"]
        matchlike = source["vb_boxes_matchlike"]
        expected = {
            "venue_contact_x_from_net_m": list(contact["x"]),
            "venue_contact_z_above_table_m": list(contact["z"]),
            "incoming_vx_mps": list(matchlike["vb_vel_x_range"]),
            "incoming_vy_mps": list(matchlike["vb_vel_y_range"]),
            "incoming_vz_mps": list(matchlike["vb_vel_z_range"]),
            "spin_magnitude_radps": [0.0, float(matchlike["vb_spin_abs_max"])],
            "forehand_contact_y_m": [float(contact["y"][0]), -0.155],
            "backhand_contact_y_m": [-0.155, float(contact["y"][1])],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenError(f"venue distribution lacks required existing boxes: {exc}") from None
    for field, value in expected.items():
        if [float(item) for item in schedule[field]] != [float(item) for item in value]:
            raise ScreenError(f"question_schedule.{field} no longer mirrors venue distribution")


def _validate_input_row(row: Any, index: int, *, verify_files: bool) -> None:
    if not isinstance(row, dict):
        raise ScreenError(f"inputs[{index}] must be a mapping")
    asset_id = row.get("asset_id")
    if not isinstance(asset_id, str) or not SAFE_ID_RE.fullmatch(asset_id):
        raise ScreenError(f"inputs[{index}].asset_id is invalid")
    if row.get("side") not in ("forehand", "backhand"):
        raise ScreenError(f"inputs[{index}].side must be forehand/backhand")
    for field in ("collection", "stroke", "action_slot"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ScreenError(f"inputs[{index}].{field} must be a non-empty string")
    if row.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise ScreenError(f"inputs[{index}] body-shape contract mismatch")
    if not isinstance(row.get("frames"), int) or row["frames"] < 2:
        raise ScreenError(f"inputs[{index}].frames must be >=2")
    _binding(row.get("input"), f"inputs[{index}].input", require_exists=verify_files)
    _binding(
        row.get("grounding_report"),
        f"inputs[{index}].grounding_report",
        require_exists=verify_files,
    )


def _counter_unit(seed: str, index: int, field: str) -> float:
    payload = f"{ALGORITHM}\0{seed}\0{index}\0{field}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / float(1 << 64)


def _sample_box(box: Sequence[float], unit: float) -> float:
    return float(box[0]) + unit * (float(box[1]) - float(box[0]))


def _canonical_question_vector(value: np.ndarray) -> np.ndarray:
    """Remove sub-picometre/libm drift before content addressing a question."""

    result = np.round(np.asarray(value, dtype=np.float64), decimals=12)
    result[result == 0.0] = 0.0
    return result


def build_questions(schedule: dict[str, Any]) -> list[Question]:
    """Materialize a stable, side-balanced venue-box paper without NumPy RNG state."""

    seed = schedule["seed"]
    count = int(schedule["count_per_side"])
    table = schedule["table_geometry"]
    questions: list[Question] = []
    for side_index, side in enumerate(schedule["sides"]):
        y_box = schedule[f"{side}_contact_y_m"]
        for local_index in range(count):
            index = side_index * count + local_index
            x_venue = _sample_box(
                schedule["venue_contact_x_from_net_m"], _counter_unit(seed, index, "px")
            )
            y = _sample_box(y_box, _counter_unit(seed, index, "py"))
            z_table = _sample_box(
                schedule["venue_contact_z_above_table_m"], _counter_unit(seed, index, "pz")
            )
            pos = np.array(
                [table["net_x_m"] + x_venue, y, table["surface_z_m"] + z_table],
                dtype=np.float64,
            )
            vel = np.array(
                [
                    _sample_box(schedule["incoming_vx_mps"], _counter_unit(seed, index, "vx")),
                    _sample_box(schedule["incoming_vy_mps"], _counter_unit(seed, index, "vy")),
                    _sample_box(schedule["incoming_vz_mps"], _counter_unit(seed, index, "vz")),
                ],
                dtype=np.float64,
            )
            magnitude = _sample_box(
                schedule["spin_magnitude_radps"], _counter_unit(seed, index, "spin_mag")
            )
            cos_theta = 2.0 * _counter_unit(seed, index, "spin_cos") - 1.0
            phi = 2.0 * math.pi * _counter_unit(seed, index, "spin_phi")
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            spin = magnitude * np.array(
                [sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta],
                dtype=np.float64,
            )
            pos = _canonical_question_vector(pos)
            vel = _canonical_question_vector(vel)
            spin = _canonical_question_vector(spin)
            atomic = {
                "side": side,
                "ball_pos_w": pos.tolist(),
                "ball_vel_w": vel.tolist(),
                "ball_spin_w": spin.tolist(),
            }
            questions.append(
                Question(
                    question_id=canonical_sha256({"kind": "motion-phase-question-v1", **atomic}),
                    side=side,
                    ball_pos_w=pos,
                    ball_vel_w=vel,
                    ball_spin_w=spin,
                )
            )
    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise ScreenError("deterministic question schedule produced duplicate question ids")
    return questions


def slerp_xyzw(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc, normalized xyzw quaternion interpolation."""

    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    u = float(fraction)
    if dot > 0.9995:
        value = qa + u * (qb - qa)
        return value / np.linalg.norm(value)
    angle = math.acos(dot)
    value = (math.sin((1.0 - u) * angle) * qa + math.sin(u * angle) * qb) / math.sin(angle)
    return value / np.linalg.norm(value)


def densify_payload(payload: dict[str, Any], substeps: int) -> tuple[dict[str, Any], np.ndarray]:
    """Densify a GMR path and return source-time coordinates for every dense row."""

    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float64)
    dof = np.asarray(payload["dof_pos"], dtype=np.float64)
    frames = root_pos.shape[0]
    dense_count = (frames - 1) * int(substeps) + 1
    dense_root = np.empty((dense_count, 3), dtype=np.float64)
    dense_rot = np.empty((dense_count, 4), dtype=np.float64)
    dense_dof = np.empty((dense_count, dof.shape[1]), dtype=np.float64)
    source_time = np.empty(dense_count, dtype=np.float64)
    cursor = 0
    for frame in range(frames - 1):
        for step in range(substeps):
            u = step / float(substeps)
            dense_root[cursor] = (1.0 - u) * root_pos[frame] + u * root_pos[frame + 1]
            dense_rot[cursor] = slerp_xyzw(root_rot[frame], root_rot[frame + 1], u)
            dense_dof[cursor] = (1.0 - u) * dof[frame] + u * dof[frame + 1]
            source_time[cursor] = frame + u
            cursor += 1
    dense_root[cursor] = root_pos[-1]
    dense_rot[cursor] = root_rot[-1] / np.linalg.norm(root_rot[-1])
    dense_dof[cursor] = dof[-1]
    source_time[cursor] = frames - 1
    dense = dict(payload)
    dense["root_pos"] = dense_root
    dense["root_rot"] = dense_rot
    dense["dof_pos"] = dense_dof
    dense["fps"] = float(np.asarray(payload["fps"]).reshape(-1)[0]) * int(substeps)
    return dense, source_time


def unsafe_source_mask(
    source_frames: int,
    source_time: np.ndarray,
    dangerous_dense: np.ndarray,
) -> np.ndarray:
    """Conservatively mark both source endpoints touched by a dangerous dense sample."""

    result = np.zeros(int(source_frames), dtype=bool)
    for coordinate in np.asarray(source_time)[np.asarray(dangerous_dense, dtype=bool)]:
        lo = int(math.floor(float(coordinate) + 1e-12))
        hi = int(math.ceil(float(coordinate) - 1e-12))
        result[max(0, lo)] = True
        result[min(source_frames - 1, hi)] = True
    return result


def wilson_lcb(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise ScreenError(f"invalid binomial counts {successes}/{total}")
    p = successes / float(total)
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (centre - radius) / denom)


def _qpos_from_payload(binding: Any, payload: dict[str, Any]) -> np.ndarray:
    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(payload["root_rot"], dtype=np.float64)
    dof = np.asarray(payload["dof_pos"], dtype=np.float64)
    result = np.repeat(binding.model.qpos0[None, :], root_pos.shape[0], axis=0)
    root = int(binding.root_qpos_address)
    result[:, root : root + 3] = root_pos
    normalized = root_xyzw / np.linalg.norm(root_xyzw, axis=1, keepdims=True)
    result[:, root + 3 : root + 7] = normalized[:, [3, 0, 1, 2]]
    result[:, list(binding.joint_qpos_addresses)] = dof
    return result


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return value if value is not None else f"geom{geom_id}"


def _body_name(mujoco: Any, model: Any, body_id: int) -> str:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    return value if value is not None else f"body{body_id}"


def load_model_contract(plan: dict[str, Any]) -> ModelContract:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - host dependent
        raise ScreenError("mujoco is required for the phase+safety runtime") from exc
    mjcf_path = _verify_file(plan["mjcf"], "mjcf")
    binding = bind_model(mujoco, mjcf_path, ground_geom_name="floor")
    expected_compiled = plan["mjcf"].get("compiled_kinematic_collision_sha256")
    if expected_compiled != binding.collision_contract_sha256:
        raise ScreenError(
            f"compiled MJCF collision SHA {binding.collision_contract_sha256} != bound "
            f"{expected_compiled}"
        )
    site_id = int(mujoco.mj_name2id(binding.model, mujoco.mjtObj.mjOBJ_SITE, RACKET_SITE))
    if site_id < 0:
        raise ScreenError(f"MJCF lacks official racket site {RACKET_SITE!r}")
    geom_by_name: dict[str, int] = {}
    for geom_id in binding.collision_geom_ids:
        name = _geom_name(mujoco, binding.model, geom_id)
        geom_by_name[name] = int(geom_id)
    missing_racket = [name for name in RACKET_GEOMS if name not in geom_by_name]
    if missing_racket:
        raise ScreenError(f"MJCF lacks racket collision geoms {missing_racket}")
    groups: dict[str, tuple[int, ...]] = {}
    for group, names in plan["dense_safety_contract"]["racket_body_clearance_groups"].items():
        missing = [name for name in names if name not in geom_by_name]
        if missing:
            raise ScreenError(f"MJCF lacks {group} clearance geoms {missing}")
        groups[group] = tuple(geom_by_name[name] for name in names)
    return ModelContract(
        mujoco=mujoco,
        binding=binding,
        site_id=site_id,
        racket_geom_ids=tuple(geom_by_name[name] for name in RACKET_GEOMS),
        clearance_groups=groups,
    )


def _group_clearance(contract: ModelContract, data: Any, group: Sequence[int]) -> float:
    # Import the repository's measured MuJoCo-3.10-safe saturation-bisection implementation.
    from audit_self_collision import geom_clearance

    best = float("inf")
    for racket in contract.racket_geom_ids:
        for body in group:
            distance, _ = geom_clearance(contract.binding.model, data, racket, body)
            best = min(best, float(distance))
    return best


def _validate_grounding_report(
    row: dict[str, Any],
    *,
    input_path: Path,
    report_path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    report = _read_json(report_path, f"grounding report {row['asset_id']}")
    if report.get("status") != "pass" or report.get("formal_eligible") is not False:
        raise ScreenError(f"{row['asset_id']} grounding report is not a diagnostic pass")
    if report.get("scope") != "diagnostic_gmr_root_z_grounding_only":
        raise ScreenError(f"{row['asset_id']} grounding report scope mismatch")
    output = report.get("output")
    if not isinstance(output, dict):
        raise ScreenError(f"{row['asset_id']} grounding report lacks output binding")
    if (
        Path(str(output.get("path"))).resolve() != input_path
        or output.get("sha256") != row["input"]["sha256"]
        or output.get("bytes") != row["input"]["bytes"]
    ):
        raise ScreenError(f"{row['asset_id']} grounding output binding mismatch")
    model = report.get("mjcf")
    if not isinstance(model, dict):
        raise ScreenError(f"{row['asset_id']} grounding report lacks MJCF binding")
    if (
        model.get("sha256") != plan["mjcf"]["sha256"]
        or model.get("compiled_kinematic_collision_sha256")
        != plan["mjcf"]["compiled_kinematic_collision_sha256"]
    ):
        raise ScreenError(f"{row['asset_id']} grounding MJCF binding mismatch")
    structure = report.get("structure")
    if (
        not isinstance(structure, dict)
        or structure.get("frames") != row["frames"]
        or structure.get("fps") != 30.0
    ):
        raise ScreenError(f"{row['asset_id']} grounding structure mismatch")
    invariants = report.get("invariants")
    if not isinstance(invariants, dict) or any(
        invariants.get(name) is not True
        for name in (
            "root_xy_exact",
            "root_rotation_exact",
            "dof_position_exact",
            "root_pos_dtype_preserved",
            "all_other_payload_fields_shallow_preserved",
        )
    ):
        raise ScreenError(f"{row['asset_id']} grounding invariants did not all pass")
    after = report.get("grounding", {}).get("after", {}).get("minimum_clearance_m")
    if not isinstance(after, (int, float)) or not math.isfinite(after) or after < -5e-7:
        raise ScreenError(f"{row['asset_id']} discrete grounding clearance is invalid")
    return report


def extract_source_racket_state(
    contract: ModelContract,
    qpos: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official site position/+Y face/site velocity for every source frame."""

    mujoco = contract.mujoco
    model = contract.binding.model
    data = mujoco.MjData(model)
    frames = qpos.shape[0]
    positions = np.empty((frames, 3), dtype=np.float64)
    normals = np.empty((frames, 3), dtype=np.float64)
    velocities = np.empty((frames, 3), dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        if frame == 0:
            dt, lo, hi = 1.0 / fps, 0, 1
        elif frame == frames - 1:
            dt, lo, hi = 1.0 / fps, frames - 2, frames - 1
        else:
            dt, lo, hi = 2.0 / fps, frame - 1, frame + 1
        mujoco.mj_differentiatePos(model, data.qvel, dt, qpos[lo], qpos[hi])
        mujoco.mj_forward(model, data)
        positions[frame] = data.site_xpos[contract.site_id]
        rotation = np.asarray(data.site_xmat[contract.site_id], dtype=np.float64).reshape(3, 3)
        normal = rotation[:, RACKET_FACE_AXIS]
        normals[frame] = normal / np.linalg.norm(normal)
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_SITE,
            contract.site_id,
            velocity,
            0,
        )
        velocities[frame] = velocity[3:6]
    if not (np.isfinite(positions).all() and np.isfinite(normals).all() and np.isfinite(velocities).all()):
        raise ScreenError("MuJoCo racket FK produced non-finite state")
    return positions, normals, velocities


def audit_dense_safety(
    contract: ModelContract,
    payload: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    dense_contract = plan["dense_safety_contract"]
    substeps = int(dense_contract["substeps_per_source_interval"])
    dense, source_time = densify_payload(payload, substeps)
    validate_joint_ranges(dense, contract.binding, tolerance_rad=1e-5)
    ground_clearance, ground_geom = frame_clearances(contract.mujoco, contract.binding, dense)
    ground_bad = ground_clearance < -float(dense_contract["ground_penetration_tolerance_m"])

    model = contract.binding.model
    data = contract.binding.data
    # Floor is checked analytically above.  Disable it before self-contact enumeration.
    model.geom_contype[contract.binding.ground_geom_id] = 0
    model.geom_conaffinity[contract.binding.ground_geom_id] = 0
    qpos = _qpos_from_payload(contract.binding, dense)
    collision_bad = np.zeros(qpos.shape[0], dtype=bool)
    clearance_bad = np.zeros(qpos.shape[0], dtype=bool)
    clearance_warn = np.zeros(qpos.shape[0], dtype=bool)
    min_clearance = np.full(qpos.shape[0], np.inf, dtype=np.float64)
    min_group = np.empty(qpos.shape[0], dtype=object)
    collision_events: list[dict[str, Any]] = []
    hard = float(dense_contract["hard_racket_body_clearance_m"])
    warning = float(dense_contract["warning_racket_body_clearance_m"])
    penetration_tolerance = float(dense_contract["self_collision_penetration_tolerance_m"])
    robot_geom_ids = set(int(value) for value in contract.binding.collision_geom_ids)
    for dense_frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[dense_frame]
        contract.mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) >= -penetration_tolerance:
                continue
            if int(contact.geom1) not in robot_geom_ids or int(contact.geom2) not in robot_geom_ids:
                continue
            collision_bad[dense_frame] = True
            if len(collision_events) < 512:
                body1 = int(model.geom_bodyid[contact.geom1])
                body2 = int(model.geom_bodyid[contact.geom2])
                collision_events.append(
                    {
                        "dense_frame": dense_frame,
                        "source_time_frames": float(source_time[dense_frame]),
                        "penetration_m": float(-contact.dist),
                        "geom_pair": [
                            _geom_name(contract.mujoco, model, contact.geom1),
                            _geom_name(contract.mujoco, model, contact.geom2),
                        ],
                        "body_pair": [
                            _body_name(contract.mujoco, model, body1),
                            _body_name(contract.mujoco, model, body2),
                        ],
                    }
                )
        for group_name, geom_ids in contract.clearance_groups.items():
            distance = _group_clearance(contract, data, geom_ids)
            if distance < min_clearance[dense_frame]:
                min_clearance[dense_frame] = distance
                min_group[dense_frame] = group_name
        clearance_bad[dense_frame] = min_clearance[dense_frame] < hard
        clearance_warn[dense_frame] = min_clearance[dense_frame] < warning

    dangerous = ground_bad | collision_bad | clearance_bad
    unsafe_source = unsafe_source_mask(payload["root_pos"].shape[0], source_time, dangerous)
    warning_source = unsafe_source_mask(payload["root_pos"].shape[0], source_time, clearance_warn)
    local_clearance = np.full(payload["root_pos"].shape[0], np.inf, dtype=np.float64)
    for dense_index, coordinate in enumerate(source_time):
        lo = int(math.floor(float(coordinate) + 1e-12))
        hi = int(math.ceil(float(coordinate) - 1e-12))
        local_clearance[lo] = min(local_clearance[lo], min_clearance[dense_index])
        local_clearance[hi] = min(local_clearance[hi], min_clearance[dense_index])
    lowest_dense = int(np.argmin(ground_clearance))
    closest_dense = int(np.argmin(min_clearance))
    return {
        "dense_frames": int(qpos.shape[0]),
        "substeps_per_source_interval": substeps,
        "effective_safety_hz": float(np.asarray(payload["fps"]).reshape(-1)[0]) * substeps,
        "unsafe_source_mask": unsafe_source,
        "warning_source_mask": warning_source,
        "source_local_min_clearance_m": local_clearance,
        "ground": {
            "dangerous_dense_samples": int(np.count_nonzero(ground_bad)),
            "minimum_clearance_m": float(ground_clearance[lowest_dense]),
            "minimum_source_time_frames": float(source_time[lowest_dense]),
            "minimum_geom_id": int(ground_geom[lowest_dense]),
            "minimum_geom_name": _geom_name(contract.mujoco, model, int(ground_geom[lowest_dense])),
        },
        "self_collision": {
            "dangerous_dense_samples": int(np.count_nonzero(collision_bad)),
            "events_truncated": len(collision_events) >= 512,
            "events": collision_events,
        },
        "racket_body_clearance": {
            "hard_threshold_m": hard,
            "warning_threshold_m": warning,
            "dangerous_dense_samples": int(np.count_nonzero(clearance_bad)),
            "warning_dense_samples": int(np.count_nonzero(clearance_warn)),
            "minimum_clearance_m": float(min_clearance[closest_dense]),
            "minimum_group": str(min_group[closest_dense]),
            "minimum_source_time_frames": float(source_time[closest_dense]),
        },
        "safe_source_frames": int(np.count_nonzero(~unsafe_source)),
        "unsafe_source_frames": int(np.count_nonzero(unsafe_source)),
    }


def _return_margin(outcome: Any, scorer: VirtualReturnScorer, pos_err: float) -> float:
    if not outcome.landed_ok:
        return float("-inf")
    x, y = (float(outcome.landing_xy[0]), float(outcome.landing_xy[1]))
    margins = (
        scorer.spec.capture_radius - float(pos_err),
        float(outcome.net_z) - scorer.net_clear_center_z,
        x - scorer.spec.net_x,
        scorer.spec.far_x - x,
        scorer.spec.half_width - abs(y),
    )
    return float(min(margins))


def _json_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def apply_verified_frame_contract(
    positions: np.ndarray,
    normals: np.ndarray,
    velocities: np.ndarray,
    frame_contract: dict[str, Any],
    *,
    matrix_4x4: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map GMR-world racket state into the explicitly verified HOPE table frame."""

    if frame_contract.get("returnability_enabled") is not True:
        raise ScreenError("cannot apply a disabled returnability frame contract")
    transform = _validate_proper_rigid_matrix(
        frame_contract["gmr_world_to_hope_matrix_4x4"]
        if matrix_4x4 is None
        else matrix_4x4,
        "applied frame transform",
    )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    mapped_positions = np.asarray(positions, dtype=np.float64) @ rotation.T + translation
    mapped_normals = np.asarray(normals, dtype=np.float64) @ rotation.T
    mapped_velocities = np.asarray(velocities, dtype=np.float64) @ rotation.T
    mapped_normals /= np.linalg.norm(mapped_normals, axis=1, keepdims=True)
    if not (
        np.isfinite(mapped_positions).all()
        and np.isfinite(mapped_normals).all()
        and np.isfinite(mapped_velocities).all()
    ):
        raise ScreenError("verified frame transform produced non-finite racket state")
    return mapped_positions, mapped_normals, mapped_velocities


def score_asset(
    row: dict[str, Any],
    *,
    payload: dict[str, Any],
    contract: ModelContract,
    safety: dict[str, Any],
    questions: list[Question],
    scorer: VirtualReturnScorer,
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_qpos = _qpos_from_payload(contract.binding, payload)
    fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
    positions, normals, velocities = extract_source_racket_state(contract, source_qpos, fps)
    matrix = None
    if plan["frame_contract"].get("transform_scope") == "per_asset":
        matrix = plan["frame_contract"]["per_asset_gmr_world_to_hope_matrix_4x4"][
            row["asset_id"]
        ]
    positions, normals, velocities = apply_verified_frame_contract(
        positions,
        normals,
        velocities,
        plan["frame_contract"],
        matrix_4x4=matrix,
    )
    speeds = np.linalg.norm(velocities, axis=1)
    effective_side = row["side"]
    if plan["frame_contract"]["mirror_status"] == "verified_mirrored":
        effective_side = "backhand" if effective_side == "forehand" else "forehand"
    side_questions = [question for question in questions if question.side == effective_side]
    clip_id = 0 if effective_side == "forehand" else 1
    target_normal_raw_a = np.array(
        [MOUNT_NORMAL_SIGN_PER_SIDE[effective_side], 0.0, 0.0], dtype=np.float64
    )
    unsafe = np.asarray(safety["unsafe_source_mask"], dtype=bool)
    local_clearance = np.asarray(safety["source_local_min_clearance_m"], dtype=np.float64)
    minimum_speed = float(plan["phase_selection_contract"]["minimum_racket_speed_mps"])
    frame_rows: list[dict[str, Any]] = []
    coverage_best: dict[str, dict[str, Any]] = {}
    for frame in range(positions.shape[0]):
        eligible = bool(not unsafe[frame] and speeds[frame] >= minimum_speed)
        exact_ids: list[str] = []
        intrinsic_ids: list[str] = []
        exact_margins: list[float] = []
        if eligible:
            for question in side_questions:
                pos_err = float(np.linalg.norm(positions[frame] - question.ball_pos_w))
                exact = scorer.score(
                    ball_vel=question.ball_vel_w,
                    ball_spin=question.ball_spin_w,
                    racket_pos=positions[frame],
                    racket_vel=velocities[frame],
                    racket_normal_raw_a=normals[frame],
                    target_normal_raw_a=target_normal_raw_a,
                    clip_id=clip_id,
                    pos_err=pos_err,
                )
                exact_margin = _return_margin(exact, scorer, pos_err)
                if exact.landed_ok:
                    exact_ids.append(question.question_id)
                    exact_margins.append(exact_margin)
                    previous = coverage_best.get(question.question_id)
                    candidate = {
                        "frame": frame,
                        "phase": frame / max(positions.shape[0] - 1, 1),
                        "return_margin_m": exact_margin,
                        "racket_body_clearance_m": float(local_clearance[frame]),
                    }
                    if previous is None or (
                        candidate["return_margin_m"],
                        candidate["racket_body_clearance_m"],
                        -candidate["frame"],
                    ) > (
                        previous["return_margin_m"],
                        previous["racket_body_clearance_m"],
                        -previous["frame"],
                    ):
                        coverage_best[question.question_id] = candidate
                intrinsic = scorer.score(
                    ball_vel=question.ball_vel_w,
                    ball_spin=question.ball_spin_w,
                    racket_pos=positions[frame],
                    racket_vel=velocities[frame],
                    racket_normal_raw_a=normals[frame],
                    target_normal_raw_a=target_normal_raw_a,
                    clip_id=clip_id,
                    pos_err=0.0,
                )
                if intrinsic.landed_ok:
                    intrinsic_ids.append(question.question_id)
        exact_count = len(exact_ids)
        intrinsic_count = len(intrinsic_ids)
        frame_rows.append(
            {
                "frame": frame,
                "phase": frame / max(positions.shape[0] - 1, 1),
                "time_s": frame / fps,
                "hard_safe": bool(not unsafe[frame]),
                "candidate_eligible": eligible,
                "racket_site_pos_w_m": positions[frame].tolist(),
                "racket_face_normal_w": normals[frame].tolist(),
                "racket_site_vel_w_mps": velocities[frame].tolist(),
                "racket_site_speed_mps": float(speeds[frame]),
                "dense_racket_body_clearance_m": float(local_clearance[frame]),
                "exact_return_count": exact_count,
                "exact_return_rate": exact_count / len(side_questions),
                "exact_return_wilson_lcb95": wilson_lcb(exact_count, len(side_questions)),
                "intrinsic_return_count": intrinsic_count,
                "intrinsic_return_rate": intrinsic_count / len(side_questions),
                "median_exact_return_margin_m": (
                    float(np.median(exact_margins)) if exact_margins else None
                ),
                "exact_return_question_ids": exact_ids,
                "intrinsic_return_question_ids": intrinsic_ids,
            }
        )

    def rank(frame_row: dict[str, Any]) -> tuple[float, ...]:
        return (
            float(frame_row["exact_return_count"]),
            float(frame_row["intrinsic_return_count"]),
            float(frame_row["median_exact_return_margin_m"] or -math.inf),
            float(frame_row["dense_racket_body_clearance_m"]),
            -float(frame_row["frame"]),
        )

    ranked = sorted((value for value in frame_rows if value["candidate_eligible"]), key=rank, reverse=True)
    top = [
        {
            "frame": value["frame"],
            "phase": value["phase"],
            "exact_return_count": value["exact_return_count"],
            "exact_return_rate": value["exact_return_rate"],
            "exact_return_wilson_lcb95": value["exact_return_wilson_lcb95"],
            "intrinsic_return_count": value["intrinsic_return_count"],
            "intrinsic_return_rate": value["intrinsic_return_rate"],
            "median_exact_return_margin_m": value["median_exact_return_margin_m"],
            "dense_racket_body_clearance_m": value["dense_racket_body_clearance_m"],
        }
        for value in ranked[:3]
    ]
    return {
        "asset_id": row["asset_id"],
        "collection": row["collection"],
        "side": row["side"],
        "effective_side_after_verified_mirror": effective_side,
        "stroke": row["stroke"],
        "action_slot": row["action_slot"],
        "body_shape_contract": row["body_shape_contract"],
        "contact_phase_truth": None,
        "frame_transform_scope": plan["frame_contract"].get("transform_scope", "global"),
        "frame_transform_matrix_4x4": (
            matrix
            if matrix is not None
            else plan["frame_contract"]["gmr_world_to_hope_matrix_4x4"]
        ),
        "frames": positions.shape[0],
        "fps": fps,
        "question_count_for_side": len(side_questions),
        "safety": {
            key: value
            for key, value in safety.items()
            if key not in ("unsafe_source_mask", "warning_source_mask", "source_local_min_clearance_m")
        },
        "top_training_phase_candidates": top,
        "selection_status": (
            "nonzero_exact_reference_coverage"
            if top and top[0]["exact_return_count"] > 0
            else "no_nonzero_exact_reference_coverage"
        ),
        "question_coverage": coverage_best,
        "per_source_frame": frame_rows,
    }


def report_asset_with_returnability_blocked(
    row: dict[str, Any],
    *,
    payload: dict[str, Any],
    contract: ModelContract,
    safety: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Keep valid safety/FK evidence while refusing scores in an unverified table frame."""

    source_qpos = _qpos_from_payload(contract.binding, payload)
    fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
    positions, normals, velocities = extract_source_racket_state(contract, source_qpos, fps)
    speeds = np.linalg.norm(velocities, axis=1)
    unsafe = np.asarray(safety["unsafe_source_mask"], dtype=bool)
    warning = np.asarray(safety["warning_source_mask"], dtype=bool)
    local_clearance = np.asarray(safety["source_local_min_clearance_m"], dtype=np.float64)
    frame_rows = [
        {
            "frame": frame,
            "phase": frame / max(positions.shape[0] - 1, 1),
            "time_s": frame / fps,
            "hard_safe": bool(not unsafe[frame]),
            "clearance_warning": bool(warning[frame]),
            "racket_site_pos_gmr_world_m": positions[frame].tolist(),
            "racket_face_normal_gmr_world": normals[frame].tolist(),
            "racket_site_vel_gmr_world_mps": velocities[frame].tolist(),
            "racket_site_speed_mps": float(speeds[frame]),
            "dense_racket_body_clearance_m": float(local_clearance[frame]),
        }
        for frame in range(positions.shape[0])
    ]
    safe_speed_order = sorted(
        (frame for frame in range(positions.shape[0]) if not unsafe[frame]),
        key=lambda frame: (float(speeds[frame]), float(local_clearance[frame]), -frame),
        reverse=True,
    )
    top_speed = [
        {
            "frame": frame,
            "phase": frame / max(positions.shape[0] - 1, 1),
            "racket_site_speed_mps": float(speeds[frame]),
            "dense_racket_body_clearance_m": float(local_clearance[frame]),
            "semantics": "safe_kinematic_speed_peak_not_strike_phase_candidate",
        }
        for frame in safe_speed_order[:3]
    ]
    return {
        "asset_id": row["asset_id"],
        "collection": row["collection"],
        "side_label_unverified_mirror": row["side"],
        "stroke": row["stroke"],
        "action_slot": row["action_slot"],
        "body_shape_contract": row["body_shape_contract"],
        "contact_phase_truth": None,
        "frames": positions.shape[0],
        "fps": fps,
        "safety": {
            key: value
            for key, value in safety.items()
            if key not in ("unsafe_source_mask", "warning_source_mask", "source_local_min_clearance_m")
        },
        "top_safe_kinematic_speed_frames": top_speed,
        "top_training_phase_candidates": None,
        "question_coverage": None,
        "selection_status": "blocked_unverified_gmr_world_to_hope_table_frame",
        "selection_blockers": list(plan["frame_contract"]["blockers"]),
        "per_source_frame": frame_rows,
    }


def aggregate_libraries(
    plan: dict[str, Any],
    assets: list[dict[str, Any]],
    questions: list[Question],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {asset["asset_id"]: asset for asset in assets}
    summaries: dict[str, Any] = {}
    for library_name, member_ids in plan["libraries"].items():
        selector: dict[str, Any] = {}
        for question in questions:
            candidates: list[dict[str, Any]] = []
            for asset_id in member_ids:
                asset = by_id[asset_id]
                if asset["side"] != question.side:
                    continue
                hit = asset["question_coverage"].get(question.question_id)
                if hit is not None:
                    candidates.append({"asset_id": asset_id, **hit})
            if candidates:
                best = max(
                    candidates,
                    key=lambda item: (
                        item["return_margin_m"],
                        item["racket_body_clearance_m"],
                        -item["frame"],
                        item["asset_id"],
                    ),
                )
                selector[question.question_id] = best
        count = len(selector)
        total = len(questions)
        per_side = {}
        for side in ("forehand", "backhand"):
            side_ids = {question.question_id for question in questions if question.side == side}
            side_count = len(side_ids.intersection(selector))
            per_side[side] = {
                "returned": side_count,
                "total": len(side_ids),
                "rate": side_count / len(side_ids),
                "wilson_lcb95": wilson_lcb(side_count, len(side_ids)),
            }
        summaries[library_name] = {
            "members": member_ids,
            "coverage_returned": count,
            "coverage_total": total,
            "coverage_rate": count / total,
            "coverage_wilson_lcb95": wilson_lcb(count, total),
            "per_side": per_side,
            "selector": selector,
        }
    comparisons: list[dict[str, Any]] = []
    for item in plan["library_comparisons"]:
        baseline = summaries[item["baseline"]]
        candidate = summaries[item["candidate"]]
        base_ids = set(baseline["selector"])
        candidate_ids = set(candidate["selector"])
        comparisons.append(
            {
                "baseline": item["baseline"],
                "candidate": item["candidate"],
                "common_support_count": len(base_ids & candidate_ids),
                "baseline_only_count": len(base_ids - candidate_ids),
                "candidate_only_count": len(candidate_ids - base_ids),
                "coverage_delta_count": len(candidate_ids) - len(base_ids),
                "coverage_delta_rate": candidate["coverage_rate"] - baseline["coverage_rate"],
                "decision_semantics": "offline_zero_retarget_reference_path_screen_only",
            }
        )
    return summaries, comparisons


def blocked_library_reports(plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    libraries = {
        name: {
            "members": members,
            "status": "blocked_unverified_gmr_world_to_hope_table_frame",
            "coverage_returned": None,
            "coverage_total": None,
            "coverage_rate": None,
            "selector": None,
        }
        for name, members in plan["libraries"].items()
    }
    comparisons = [
        {
            "baseline": item["baseline"],
            "candidate": item["candidate"],
            "status": "blocked_unverified_gmr_world_to_hope_table_frame",
            "coverage_delta_count": None,
            "coverage_delta_rate": None,
        }
        for item in plan["library_comparisons"]
    ]
    return libraries, comparisons


def _atomic_write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ScreenError(f"refusing to overwrite result: {path}")
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
            raise ScreenError(f"refusing concurrent overwrite of result: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def run_screen(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    plan = validate_manifest(
        plan_path,
        expected_sha256,
        require_ready=True,
        verify_files=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    questions = build_questions(plan["question_schedule"])
    contract = load_model_contract(plan)
    params = load_venue_params(str(_verify_file(plan["physics"], "physics")))
    geometry = plan["question_schedule"]["table_geometry"]
    scorer = VirtualReturnScorer(
        params,
        VirtualReturnSpec(
            table_surface_z=float(geometry["surface_z_m"]),
            net_x=float(geometry["net_x_m"]),
            far_x=float(geometry["far_x_m"]),
            half_width=float(geometry["half_width_m"]),
            net_height=float(geometry["net_height_m"]),
        ),
        mount_normal_sign_per_clip=(1.0, -1.0),
    )
    results: list[dict[str, Any]] = []
    for row in plan["inputs"]:
        input_path = _verify_file(row["input"], f"input {row['asset_id']}")
        report_path = _verify_file(
            row["grounding_report"], f"grounding report {row['asset_id']}"
        )
        grounding_report = _validate_grounding_report(
            row,
            input_path=input_path,
            report_path=report_path,
            plan=plan,
        )
        try:
            payload = load_pickle(input_path)
        except (OSError, pickle.PickleError) as exc:
            raise ScreenError(f"cannot load {row['asset_id']}: {exc}") from None
        validate_payload(
            payload,
            expected_frames=row["frames"],
            expected_fps=30.0,
            quaternion_norm_tolerance=1e-6,
        )
        validate_joint_ranges(payload, contract.binding, tolerance_rad=1e-5)
        safety = audit_dense_safety(contract, payload, plan)
        if plan["frame_contract"]["returnability_enabled"]:
            asset = score_asset(
                row,
                payload=payload,
                contract=contract,
                safety=safety,
                questions=questions,
                scorer=scorer,
                plan=plan,
            )
        else:
            asset = report_asset_with_returnability_blocked(
                row,
                payload=payload,
                contract=contract,
                safety=safety,
                plan=plan,
            )
        asset["input"] = row["input"]
        asset["grounding_report"] = row["grounding_report"]
        asset["grounding_discrete_minimum_clearance_m"] = float(
            grounding_report["grounding"]["after"]["minimum_clearance_m"]
        )
        results.append(asset)
    if plan["frame_contract"]["returnability_enabled"]:
        libraries, comparisons = aggregate_libraries(plan, results, questions)
    else:
        libraries, comparisons = blocked_library_reports(plan)
    question_rows = [
        {
            "question_id": question.question_id,
            "side": question.side,
            "ball_pos_w_m": question.ball_pos_w.tolist(),
            "ball_vel_w_mps": question.ball_vel_w.tolist(),
            "ball_spin_w_radps": question.ball_spin_w.tolist(),
        }
        for question in questions
    ]
    output = {
        "schema_version": 1,
        "status": (
            "complete_diagnostic_phase_safety_screen"
            if plan["frame_contract"]["returnability_enabled"]
            else "complete_dense_safety_returnability_blocked"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "formal_eligible": False,
        "robot_approved": False,
        "contact_phase_truth": None,
        "manifest": {
            "path": str(plan_path.resolve()),
            "bytes": plan_path.stat().st_size,
            "sha256": expected_sha256,
        },
        "body_shape_contract": BODY_SHAPE_CONTRACT,
        "question_schedule": {
            "algorithm": ALGORITHM,
            "count": len(question_rows),
            "semantic_sha256": canonical_sha256(question_rows),
            "questions": question_rows,
            "consumed_for_returnability": bool(plan["frame_contract"]["returnability_enabled"]),
        },
        "frame_contract": plan["frame_contract"],
        "frame_contract_evidence": plan.get("frame_contract_evidence"),
        "assets": results,
        "libraries": libraries,
        "library_comparisons": comparisons,
        "claims": {
            "racket_state": "vendor_MJCF_official_site_FK_and_generalized_velocity",
            "safety": "dense_sampled_not_mathematical_continuous_time_certificate",
            "exact_coverage": (
                "canonical_counterfactual_HOPE_table_zero_retarget_reference_path_lower_bound"
                if plan["frame_contract"]["returnability_enabled"]
                else None
            ),
            "real_capture_returnability": None,
            "intrinsic_returnability": (
                "ball_relocated_to_each_candidate_site_diagnostic"
                if plan["frame_contract"]["returnability_enabled"]
                else None
            ),
            "phase_candidates": (
                "offline_training_candidates_not_observed_contact_truth"
                if plan["frame_contract"]["returnability_enabled"]
                else None
            ),
        },
        "remaining_blockers": [
            "monocular air swings have no observed ball contact or paddle calibration truth",
            "the scored table is the canonical counterfactual HOPE table; real-capture table returnability remains unsupported",
            "dense sampling is not a continuous-time collision certificate",
            "table/net swept-volume clearance and dynamics/balance feasibility are separate gates",
            "coverage does not include policy retargeting, planner reachability, uncertainty, or immutable exam",
            "schema-2 conversion, TOPP re-audit, RL training and robot safety approval remain absent",
        ],
    }
    result_path = Path(plan["output_contract"]["result"]).expanduser().resolve()
    output["output_path"] = str(result_path)
    _atomic_write_new(result_path, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "validate",
        help="validate a blocked preregistration or ready runtime contract without screening",
    )
    subparsers.add_parser("run", help="run an execution_ready=true manifest, no-clobber")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            plan = validate_manifest(
                args.manifest.resolve(),
                args.expected_manifest_sha256,
                require_ready=False,
                verify_files=True,
            )
            print(
                "[motion-phase-safety] VALID "
                f"execution_ready={str(plan['execution_ready']).lower()} "
                f"assets={len(plan['inputs'])}"
            )
            return 0
        result = run_screen(args.manifest.resolve(), args.expected_manifest_sha256)
        print(
            "[motion-phase-safety] COMPLETE "
            f"assets={len(result['assets'])} questions={result['question_schedule']['count']} "
            f"result={result['output_path']}"
        )
        return 0
    except (ScreenError, OSError, ValueError) as exc:
        print(f"[motion-phase-safety] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
