#!/usr/bin/env python3
"""Apply one exact canonical-beta donor to an S0/M0 post-GVHMR handoff.

This is the narrow consumer between the immutable post-GVHMR handoff and GMR.
It deliberately reuses the audited PT loading, beta replacement, semantic
digest, save/reload verification and no-clobber publication primitives from
``materialize_canonical_gvhmr_betas.py``.  Unlike that historical cohort tool,
it does not recompute a median: the exact previously accepted same-performer
donor vector is injected into every newly reconstructed clip.

``static`` reads repository files only.  ``inspect`` reads all private runtime
inputs but writes nothing.  ``consume`` publishes a new directory once, with
the completion manifest last.  None of the modes runs GMR, simulation,
training, or hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import consume_motion_post_gvhmr_exact as post  # noqa: E402
import materialize_canonical_gvhmr_betas as base  # noqa: E402


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
HANDOFF_STATUS = "complete_exact_post_gvhmr_handoff_downstream_blocked"
PLAN_STATUS = "preregistered_not_executed"
RESULT_STATUS = "complete_exact_donor_beta_materialization"
RESULT_SUFFIX = ".diagnostic_franco_donor_betas.pt"
BASE_TOOL = "scripts/materialize_canonical_gvhmr_betas.py"
BODY_SHAPE_CONTRACT = base.BODY_SHAPE_CONTRACT


class MotionBetaError(ValueError):
    """The exact motion canonical-beta contract cannot be satisfied."""


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise MotionBetaError(f"{label} must be a lowercase SHA-256")
    return value


def require_repo_binding(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise MotionBetaError(f"{label} must contain exactly path/sha256")
    text = value.get("path")
    if not isinstance(text, str) or not text or Path(text).is_absolute() or ".." in Path(text).parts:
        raise MotionBetaError(f"{label}.path must be repository-relative")
    path = (REPO_ROOT / text).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise MotionBetaError(f"{label}.path escapes the repository") from None
    if not path.is_file():
        raise MotionBetaError(f"{label}.path is missing: {path}")
    expected = require_sha(value.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise MotionBetaError(f"{label} sha256 {actual} != {expected}")
    return path


def require_absolute_binding(value: Any, label: str, *, with_bytes: bool) -> dict[str, Any]:
    fields = {"path", "sha256", "bytes"} if with_bytes else {"path", "sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise MotionBetaError(f"{label} must contain exactly {sorted(fields)}")
    path = value.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise MotionBetaError(f"{label}.path must be absolute")
    require_sha(value.get("sha256"), f"{label}.sha256")
    if with_bytes and (
        isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] <= 0
    ):
        raise MotionBetaError(f"{label}.bytes must be positive")
    return value


def verify_absolute_binding(value: dict[str, Any], label: str, *, with_bytes: bool) -> Path:
    binding = require_absolute_binding(value, label, with_bytes=with_bytes)
    path = Path(binding["path"])
    post.ensure_no_symlink_components(path, label)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise MotionBetaError(f"{label} is not a regular file: {path}")
    if with_bytes and info.st_size != binding["bytes"]:
        raise MotionBetaError(f"{label} bytes {info.st_size} != {binding['bytes']}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise MotionBetaError(f"{label} sha256 {actual} != {binding['sha256']}")
    return path


def require_window(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
        or not (0.0 <= float(value[0]) < float(value[1]))
    ):
        raise MotionBetaError(f"{label} must be an increasing finite non-negative pair")
    return [float(value[0]), float(value[1])]


def validate_stance_contract(plan: Mapping[str, Any]) -> None:
    stance = plan.get("m0_stance_handoff")
    if plan.get("batch_kind") == "s0_static_high_press":
        if stance is not None:
            raise MotionBetaError("S0 must not carry an M0 stance contract")
        return
    if not isinstance(stance, dict):
        raise MotionBetaError("M0 requires m0_stance_handoff")
    exact = {
        "measurement_stage": "exact_robot_coordinate_gmr_before_schema2_promotion",
        "normalization": "remove_common_root_xy_and_align_heading_to_candidate_initial_ready_frame",
        "vector_definition": "d_xy=right_foot_xy-left_foot_xy",
        "window_estimator": "coordinatewise_robust_median_over_exact_ready_windows",
        "must_preserve_components": ["lateral_separation", "fore_aft_stagger"],
        "feet_together_or_narrower_substitute_allowed": False,
        "foot_site_mapping": None,
        "robot_coordinate_numeric_reference_m": None,
        "initial_d_xy_m": None,
        "terminal_d_xy_m": None,
        "component_tolerance_m": None,
        "stance_passed": None,
        "producer": "future_separately_preregistered_exact_gmr",
        "numeric_reference_status": "blocked_until_exact_gmr_materializes_a3_foot_sites",
        "missing_numeric_reference_acceptance": "fail_closed",
    }
    if stance != exact:
        raise MotionBetaError(
            "m0_stance_handoff must equal the frozen fail-closed pre-GMR contract"
        )


def validate_semantic_guard(plan: Mapping[str, Any]) -> None:
    guard = plan.get("semantic_guard")
    if not isinstance(guard, dict):
        raise MotionBetaError("semantic_guard must be an object")
    if plan.get("batch_kind") == "s0_static_high_press":
        exact = {
            "motion_role": "fifth_action_backhand_high_ball_forward_downward_press",
            "question_family": "separate_high_ball_high_press_paper_required_not_yet_preregistered",
            "pull_or_loop_question_paper_allowed": False,
            "observed_ball_contact": None,
            "strike_effectiveness": None,
            "safety_result": None,
        }
    else:
        exact = {
            "motion_role": "signed_lateral_displacement_conditioned_lower_body_teacher",
            "strike_effectiveness": None,
            "safety_result": None,
            "teacher_selection_result": None,
        }
    if guard != exact:
        raise MotionBetaError("semantic_guard changed or overclaims this offline stage")


def validate_plan(path: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    expected_sha = require_sha(expected_sha, "--expected-prereg-sha256")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise MotionBetaError(f"prereg sha256 {actual} != {expected_sha}")
    try:
        plan = post.read_json(path, "motion canonical-beta preregistration")
    except post.HandoffError as exc:
        raise MotionBetaError(str(exc)) from None
    if plan.get("schema_version") != 1 or plan.get("status") != PLAN_STATUS:
        raise MotionBetaError("plan must be schema 1 and preregistered_not_executed")
    if plan.get("batch_kind") not in {"s0_static_high_press", "m0_lateral_teachers"}:
        raise MotionBetaError("unsupported batch_kind")
    if plan.get("formal_eligible") is not False or plan.get("training_authorized") is not False:
        raise MotionBetaError("formal/training authorization must remain false")
    if plan.get("hardware_authorized") is not False:
        raise MotionBetaError("hardware_authorized must remain false")
    if plan.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise MotionBetaError("body_shape_contract mismatch")

    consumer = require_repo_binding(plan.get("consumer"), "consumer")
    if consumer != Path(__file__).resolve():
        raise MotionBetaError("consumer path must name this script")
    materializer = require_repo_binding(plan.get("base_materializer"), "base_materializer")
    if materializer != (REPO_ROOT / BASE_TOOL).resolve():
        raise MotionBetaError("base_materializer path mismatch")
    post_consumer = require_repo_binding(
        plan.get("post_handoff_consumer"), "post_handoff_consumer"
    )
    if post_consumer != Path(post.__file__).resolve():
        raise MotionBetaError("post_handoff_consumer path must name the imported helper")
    handoff = require_absolute_binding(plan.get("handoff"), "handoff", with_bytes=True)
    upstream = plan.get("upstream_post_gvhmr")
    if not isinstance(upstream, dict):
        raise MotionBetaError("upstream_post_gvhmr must be an object")
    require_repo_binding(upstream.get("preregistration"), "upstream_post_gvhmr.preregistration")
    require_sha(upstream.get("consumer_sha256"), "upstream_post_gvhmr.consumer_sha256")

    donor = plan.get("canonical_beta_donor")
    if not isinstance(donor, dict):
        raise MotionBetaError("canonical_beta_donor must be an object")
    if donor.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise MotionBetaError("canonical donor contract mismatch")
    require_absolute_binding(donor.get("artifact"), "canonical_beta_donor.artifact", with_bytes=True)
    require_absolute_binding(
        donor.get("completion_manifest"),
        "canonical_beta_donor.completion_manifest",
        with_bytes=True,
    )
    require_sha(donor.get("vector_sha256"), "canonical_beta_donor.vector_sha256")
    if donor.get("reuse") != "exact_existing_vector_no_reaggregation":
        raise MotionBetaError("canonical donor must forbid reaggregation")

    expected_count = 1 if plan["batch_kind"] == "s0_static_high_press" else 4
    rows = plan.get("inputs")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise MotionBetaError(f"inputs must contain exactly {expected_count} rows")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MotionBetaError(f"inputs[{index}] must be an object")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id) or asset_id in seen_ids:
            raise MotionBetaError(f"inputs[{index}].asset_id is unsafe or duplicate")
        seen_ids.add(asset_id)
        source = require_absolute_binding(row.get("source_pt"), f"{asset_id}.source_pt", with_bytes=True)
        if source["path"] in seen_paths:
            raise MotionBetaError("duplicate source PT path")
        seen_paths.add(source["path"])
        if isinstance(row.get("frames"), bool) or not isinstance(row.get("frames"), int) or row["frames"] <= 1:
            raise MotionBetaError(f"{asset_id}.frames must be >1")
        before = require_window(row.get("ready_before_window_s"), f"{asset_id}.ready_before_window_s")
        after = require_window(row.get("ready_after_window_s"), f"{asset_id}.ready_after_window_s")
        if before[1] >= after[0]:
            raise MotionBetaError(f"{asset_id} ready windows overlap or reverse")
    expected_ids = (
        ["static_backhand_high_press"]
        if expected_count == 1
        else [
            "lateral_step_left_1",
            "lateral_step_left_2",
            "lateral_step_right_1",
            "lateral_step_right_2",
        ]
    )
    if [row["asset_id"] for row in rows] != expected_ids:
        raise MotionBetaError("input order/asset set mismatch")
    if upstream.get("batch_id") != (
        "static_high_press_s0_v1" if expected_count == 1 else "lateral_teachers_m0_v1"
    ):
        raise MotionBetaError("upstream batch_id mismatch")
    if upstream.get("asset_ids") != expected_ids:
        raise MotionBetaError("upstream asset_ids mismatch")

    output = plan.get("output_contract")
    if not isinstance(output, dict):
        raise MotionBetaError("output_contract must be an object")
    required_output = {
        "result_suffix": RESULT_SUFFIX,
        "canonical_betas_filename": "canonical_betas.json",
        "completion_manifest_filename": "materialization_manifest.json",
        "output_root_must_not_exist": True,
        "no_clobber": True,
        "completion_manifest_published_last": True,
    }
    for field, value in required_output.items():
        if output.get(field) != value:
            raise MotionBetaError(f"output_contract.{field} must be {value!r}")
    if not isinstance(output.get("output_root"), str) or not Path(output["output_root"]).is_absolute():
        raise MotionBetaError("output_contract.output_root must be absolute")
    if output["output_root"] == str(Path(handoff["path"]).parent):
        raise MotionBetaError("output root must be disjoint from the handoff root")
    changed = plan.get("changed_field_allowlist")
    if changed != ["smpl_params_global.betas"]:
        raise MotionBetaError("only smpl_params_global.betas may change")
    validate_stance_contract(plan)
    validate_semantic_guard(plan)
    return plan, actual


def validate_handoff(plan: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    handoff_path = verify_absolute_binding(plan["handoff"], "handoff", with_bytes=True)
    try:
        handoff = post.read_json(handoff_path, "post-GVHMR handoff")
    except post.HandoffError as exc:
        raise MotionBetaError(str(exc)) from None
    upstream = plan["upstream_post_gvhmr"]
    expected = {
        "schema_version": 1,
        "status": HANDOFF_STATUS,
        "plan_sha256": upstream["preregistration"]["sha256"],
        "consumer_sha256": upstream["consumer_sha256"],
        "batch_kind": plan["batch_kind"],
        "formal_eligible": False,
        "training_authorized": False,
        "hardware_authorized": False,
    }
    for field, value in expected.items():
        if handoff.get(field) != value:
            raise MotionBetaError(f"handoff.{field} mismatch")
    evidence = handoff.get("runtime_evidence")
    if not isinstance(evidence, dict):
        raise MotionBetaError("handoff.runtime_evidence must be an object")
    if evidence.get("batch_id") != upstream["batch_id"] or evidence.get("asset_ids") != upstream["asset_ids"]:
        raise MotionBetaError("handoff batch/asset order mismatch")
    rows = evidence.get("results")
    if not isinstance(rows, list) or len(rows) != len(plan["inputs"]):
        raise MotionBetaError("handoff result count mismatch")
    for expected_row, runtime_row in zip(plan["inputs"], rows, strict=True):
        if not isinstance(runtime_row, dict):
            raise MotionBetaError("handoff result row must be an object")
        expected_pt = expected_row["source_pt"]
        exact = {
            "asset_id": expected_row["asset_id"],
            "frames": expected_row["frames"],
            "gvhmr_output": expected_pt,
            "ready_before_window_s": expected_row["ready_before_window_s"],
            "ready_after_window_s": expected_row["ready_after_window_s"],
        }
        for field, value in exact.items():
            if runtime_row.get(field) != value:
                raise MotionBetaError(f"handoff {expected_row['asset_id']}.{field} mismatch")
        finite = runtime_row.get("finite_elements")
        if isinstance(finite, bool) or not isinstance(finite, int) or finite <= 0:
            raise MotionBetaError(f"handoff {expected_row['asset_id']} finite count invalid")
    runtime_donor = evidence.get("canonical_beta_donor")
    if not isinstance(runtime_donor, dict):
        raise MotionBetaError("handoff lacks canonical donor")
    for field in ("artifact", "completion_manifest", "vector_sha256"):
        if runtime_donor.get(field) != plan["canonical_beta_donor"][field]:
            raise MotionBetaError(f"handoff canonical donor {field} mismatch")
    if runtime_donor.get("new_batch_materialization_status") != "not_run":
        raise MotionBetaError("handoff already claims new-batch materialization")
    gate = handoff.get("downstream_gate")
    if not isinstance(gate, dict) or gate.get("next_authorized_stage") != "canonical_beta_materialization_only":
        raise MotionBetaError("handoff does not authorize only canonical-beta materialization")
    statuses = gate.get("statuses")
    if not isinstance(statuses, dict) or statuses.get("canonical_beta_materialization") != "not_run":
        raise MotionBetaError("handoff canonical-beta stage is not not_run")
    return handoff, handoff_path


def load_donor(plan: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    donor = plan["canonical_beta_donor"]
    artifact_path = verify_absolute_binding(donor["artifact"], "canonical donor artifact", with_bytes=True)
    completion_path = verify_absolute_binding(
        donor["completion_manifest"], "canonical donor completion", with_bytes=True
    )
    try:
        artifact = post.read_json(artifact_path, "canonical donor artifact")
        completion = post.read_json(completion_path, "canonical donor completion")
    except post.HandoffError as exc:
        raise MotionBetaError(str(exc)) from None
    if artifact.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise MotionBetaError("donor artifact body-shape contract mismatch")
    if artifact.get("vector_sha256") != donor["vector_sha256"]:
        raise MotionBetaError("donor artifact vector SHA mismatch")
    components = artifact.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 10
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in components)
    ):
        raise MotionBetaError("donor components must be ten finite numbers")
    if artifact.get("measured_height_m") is not None or artifact.get("a3_calibrated") is not False:
        raise MotionBetaError("donor must remain explicitly uncalibrated")
    if completion.get("status") != "complete":
        raise MotionBetaError("donor completion manifest is not complete")
    bound = completion.get("canonical_betas_artifact")
    if not isinstance(bound, dict):
        raise MotionBetaError("donor completion lacks artifact binding")
    for field in ("path", "sha256", "vector_sha256"):
        expected = donor["artifact"].get(field) if field != "vector_sha256" else donor["vector_sha256"]
        if bound.get(field) != expected:
            raise MotionBetaError(f"donor completion {field} mismatch")
    return artifact, artifact_path.read_bytes()


def inspect_inputs(plan: Mapping[str, Any]) -> dict[str, Any]:
    handoff, handoff_path = validate_handoff(plan)
    donor, donor_bytes = load_donor(plan)
    torch = base._load_torch()
    loaded: list[dict[str, Any]] = []
    storage_contract: tuple[str, str, str] | None = None
    for row in plan["inputs"]:
        source_path = verify_absolute_binding(row["source_pt"], f"{row['asset_id']} source PT", with_bytes=True)
        payload = base.load_torch_result(source_path, torch)
        matrix, metadata = base.beta_array_and_metadata(payload, row["frames"], torch)
        contract = (metadata["storage_kind"], metadata["dtype"], metadata["numpy_dtype"])
        if storage_contract is None:
            storage_contract = contract
        elif storage_contract != contract:
            raise MotionBetaError(f"mixed beta storage/dtype contract {contract} != {storage_contract}")
        non_beta_sha, non_beta_leaves = base.semantic_digest(payload, torch)
        if sha256_file(source_path) != row["source_pt"]["sha256"]:
            raise MotionBetaError(f"{row['asset_id']} source changed during load")
        loaded.append(
            {
                "row": row,
                "source_path": source_path,
                "payload": payload,
                "source_beta_matrix": matrix,
                "beta_metadata": metadata,
                "non_beta_sha": non_beta_sha,
                "non_beta_leaves": non_beta_leaves,
            }
        )
    assert storage_contract is not None
    canonical = base.quantize_canonical_vector(
        np.asarray(donor["components"], dtype=np.float64), storage_contract[2]
    )
    if base._vector_sha256(canonical) != plan["canonical_beta_donor"]["vector_sha256"]:
        raise MotionBetaError("donor vector does not reproduce exact SHA in source beta dtype")
    return {
        "handoff": handoff,
        "handoff_path": handoff_path,
        "donor": donor,
        "donor_bytes": donor_bytes,
        "canonical": canonical,
        "torch": torch,
        "loaded": loaded,
        "storage_contract": storage_contract,
    }


def publish_staged_durable(staging: Path, output_root: Path, manifest_name: str) -> None:
    """Publish no-clobber, fsync artifacts, then link and fsync completion last."""

    if output_root.exists():
        raise MotionBetaError(f"output root already exists: {output_root}")
    try:
        output_root.mkdir()
    except FileExistsError:
        raise MotionBetaError(f"output root appeared during publication: {output_root}") from None
    post.fsync_directory(output_root.parent)
    names = sorted(path.name for path in staging.iterdir())
    if manifest_name not in names:
        raise MotionBetaError("staging is missing the completion manifest")
    names.remove(manifest_name)
    for name in names:
        try:
            os.link(staging / name, output_root / name)
        except FileExistsError:
            raise MotionBetaError(
                f"refusing to overwrite artifact that appeared during publication: {output_root / name}"
            ) from None
    post.fsync_directory(output_root)
    try:
        os.link(staging / manifest_name, output_root / manifest_name)
    except FileExistsError:
        raise MotionBetaError(
            f"refusing to overwrite completion manifest: {output_root / manifest_name}"
        ) from None
    post.fsync_directory(output_root)
    for source in staging.iterdir():
        source.unlink()
    staging.rmdir()


def materialize(plan: Mapping[str, Any], plan_path: Path, plan_sha: str, evidence: Mapping[str, Any]) -> Path:
    output_root = Path(plan["output_contract"]["output_root"])
    if output_root.exists():
        raise MotionBetaError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    post.ensure_no_symlink_components(output_root.parent, "output root parent")
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    try:
        staging.mkdir()
    except FileExistsError:
        raise MotionBetaError(f"staging path already exists: {staging}") from None
    rows: list[dict[str, Any]] = []
    try:
        for item in evidence["loaded"]:
            row = item["row"]
            output_name = row["asset_id"] + RESULT_SUFFIX
            temporary = staging / output_name
            output_payload = base.replace_betas(item["payload"], evidence["canonical"], evidence["torch"])
            if base.semantic_digest(item["payload"], evidence["torch"]) != (
                item["non_beta_sha"],
                item["non_beta_leaves"],
            ):
                raise MotionBetaError(f"{row['asset_id']} source payload mutated in memory")
            base.verify_materialized_betas(
                output_payload,
                evidence["canonical"],
                row["frames"],
                item["beta_metadata"],
                evidence["torch"],
            )
            output_non_beta_sha, output_leaves, output_sha, output_bytes = base.save_and_reload_verified(
                output_payload,
                item["non_beta_sha"],
                evidence["canonical"],
                row["frames"],
                item["beta_metadata"],
                temporary,
                evidence["torch"],
            )
            rows.append(
                {
                    "asset_id": row["asset_id"],
                    "frames": row["frames"],
                    "ready_before_window_s": row["ready_before_window_s"],
                    "ready_after_window_s": row["ready_after_window_s"],
                    "source_pt": row["source_pt"],
                    "source_beta_contract": item["beta_metadata"],
                    "source_beta_values_sha256": hashlib.sha256(
                        item["source_beta_matrix"].tobytes(order="C")
                    ).hexdigest(),
                    "output_path": str(output_root / output_name),
                    "output_sha256": output_sha,
                    "output_bytes": output_bytes,
                    "output_canonical_vector_sha256": plan["canonical_beta_donor"]["vector_sha256"],
                    "non_beta_bit_exact": True,
                    "source_non_beta_semantic_sha256": item["non_beta_sha"],
                    "output_non_beta_semantic_sha256": output_non_beta_sha,
                    "source_non_beta_leaf_count": item["non_beta_leaves"],
                    "output_non_beta_leaf_count": output_leaves,
                }
            )
        base._write_exclusive(staging / "canonical_betas.json", evidence["donor_bytes"])
        if sha256_file(staging / "canonical_betas.json") != plan["canonical_beta_donor"]["artifact"]["sha256"]:
            raise MotionBetaError("copied donor artifact SHA changed")
        result = {
            "schema_version": 1,
            "status": RESULT_STATUS,
            "completed_utc": base.utc_now(),
            "scope": "exact donor beta materialization only; no GMR, schema-2, safety, effectiveness, training, or hardware claim",
            "plan": {"path": str(plan_path), "sha256": plan_sha},
            "consumer": plan["consumer"],
            "base_materializer": plan["base_materializer"],
            "post_handoff_consumer": plan["post_handoff_consumer"],
            "execution_fingerprint": evidence["execution_fingerprint"],
            "handoff": plan["handoff"],
            "body_shape_contract": BODY_SHAPE_CONTRACT,
            "canonical_beta_donor": plan["canonical_beta_donor"],
            "canonical_beta_artifact_semantics": (
                "byte-exact donor copy only; it is not a new S0/M0 cohort aggregation result"
            ),
            "changed_field_allowlist": ["smpl_params_global.betas"],
            "formal_eligible": False,
            "training_authorized": False,
            "hardware_authorized": False,
            "results": rows,
            "m0_stance_handoff": plan.get("m0_stance_handoff"),
            "next_gate": {
                "authorized": "separate_exact_gmr_preregistration_only",
                "status": "blocked_until_exact_gmr_plan_and_runtime_are_bound",
            },
        }
        base._write_exclusive(staging / "materialization_manifest.json", base._json_bytes(result))
        publish_staged_durable(staging, output_root, "materialization_manifest.json")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_root / "materialization_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("static", help="validate tracked contract only")
    subparsers.add_parser("inspect", help="validate handoff, donor and PTs without writing")
    subparsers.add_parser("consume", help="materialize once and publish completion last")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan_path = args.prereg.resolve()
        plan, plan_sha = validate_plan(plan_path, args.expected_prereg_sha256)
        if args.command == "static":
            print(f"[motion-canonical-betas] PASS static batch={plan['batch_kind']} prereg_sha256={plan_sha}")
            return 0
        execution_fingerprint = base.validate_execution_contract(plan)
        evidence = inspect_inputs(plan)
        evidence["execution_fingerprint"] = execution_fingerprint
        if args.command == "inspect":
            print(
                "[motion-canonical-betas] PASS inspect "
                f"batch={plan['batch_kind']} assets={len(evidence['loaded'])} "
                f"vector_sha256={plan['canonical_beta_donor']['vector_sha256']}"
            )
            return 0
        manifest = materialize(plan, plan_path, plan_sha, evidence)
        print(f"[motion-canonical-betas] PASS consume manifest={manifest}")
        return 0
    except (MotionBetaError, base.MaterializationError, post.HandoffError, OSError, TypeError, ValueError) as exc:
        print(f"[motion-canonical-betas] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
