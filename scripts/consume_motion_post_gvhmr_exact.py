#!/usr/bin/env python3
"""Validate and publish an exact S0/M0 post-GVHMR handoff.

This consumer is intentionally narrow.  It proves that one already-completed
GVHMR batch is byte-for-byte the preregistered batch and publishes an immutable
handoff for the *next* offline stage.  It does not run canonical-beta
materialization, GMR, schema-2 conversion, simulation, training, or hardware.

The runtime chain is checked in both directions:

* preregistration -> execution record -> queue state;
* queue state -> per-asset binding -> structural audit -> GVHMR PT.

Publication is no-clobber.  All validation happens before the output directory
is claimed, and the handoff JSON is written with O_CREAT|O_EXCL and fsync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_motion_video_gvhmr_prereg import (  # noqa: E402
    PreregError,
    validate_static_prereg,
)


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
BODY_SHAPE_CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"
AUTHORIZATION_SCOPE = "offline_gvhmr_video_to_smplx_and_structural_audit_only"
SOURCE_CONSUMPTION = "private_read_only_no_clobber_snapshot"
HANDOFF_STATUS = "complete_exact_post_gvhmr_handoff_downstream_blocked"


class HandoffError(ValueError):
    """The exact post-GVHMR handoff contract cannot be satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise HandoffError(f"{label} must be a lowercase SHA-256")
    return value


def _reject_constant(value: str) -> None:
    raise HandoffError(f"JSON contains forbidden non-finite token {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise HandoffError(f"{label} root must be an object")
    return payload


def ensure_no_symlink_components(path: Path, label: str, *, leaf_may_be_missing: bool = False) -> None:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if leaf_may_be_missing and is_leaf:
                return
            raise HandoffError(f"{label} path component is missing: {current}") from None
        if stat.S_ISLNK(info.st_mode):
            raise HandoffError(f"{label} path contains a symlink: {current}")


def require_file_binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise HandoffError(f"{label} must contain exactly path/bytes/sha256")
    path = value.get("path")
    size = value.get("bytes")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise HandoffError(f"{label}.path must be absolute")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HandoffError(f"{label}.bytes must be a positive integer")
    require_sha(value.get("sha256"), f"{label}.sha256")
    return value


def verify_bound_file(value: dict[str, Any], label: str) -> Path:
    binding = require_file_binding(value, label)
    path = Path(binding["path"])
    ensure_no_symlink_components(path, label)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise HandoffError(f"{label} is not a regular file: {path}")
    if info.st_size != binding["bytes"]:
        raise HandoffError(
            f"{label} bytes {info.st_size} != {binding['bytes']}: {path}"
        )
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise HandoffError(
            f"{label} sha256 {actual} != {binding['sha256']}: {path}"
        )
    return path


def resolve_repo_file(path_text: Any, label: str, repo_root: Path) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise HandoffError(f"{label} must be a non-empty repository-relative path")
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise HandoffError(f"{label} must stay inside the repository")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        raise HandoffError(f"{label} escapes the repository") from None
    if not resolved.is_file():
        raise HandoffError(f"{label} is missing: {resolved}")
    return resolved


def verify_repo_binding(value: Any, label: str, repo_root: Path) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise HandoffError(f"{label} must contain exactly path/sha256")
    path = resolve_repo_file(value.get("path"), f"{label}.path", repo_root)
    expected = require_sha(value.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise HandoffError(f"{label} sha256 {actual} != {expected}")
    return path


def _require_string_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise HandoffError(f"{label} must be a non-empty unique string list")
    return value


def _validate_canonical_donor(plan: dict[str, Any], repo_root: Path) -> None:
    donor = plan.get("canonical_beta_donor")
    if not isinstance(donor, dict):
        raise HandoffError("canonical_beta_donor must be an object")
    if donor.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise HandoffError("canonical beta donor body-shape contract mismatch")
    if donor.get("same_performer_asserted_by_human_owner") is not True:
        raise HandoffError("canonical beta donor requires an explicit same-performer assertion")
    if donor.get("reuse_semantics") != (
        "exact donor vector only; each new GVHMR PT still requires a separate no-clobber "
        "beta materialization and bit-exact non-beta audit"
    ):
        raise HandoffError("canonical beta donor reuse semantics changed")
    result_path = verify_repo_binding(
        donor.get("tracked_result_manifest"), "canonical_beta_donor.tracked_result_manifest", repo_root
    )
    result = read_json(result_path, "canonical-beta tracked result")
    if result.get("status") != "complete_diagnostic_canonical_betas_materialization":
        raise HandoffError("canonical-beta donor result is not complete diagnostic evidence")
    if result.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise HandoffError("canonical-beta donor result contract mismatch")
    canonical = result.get("canonical_betas")
    if not isinstance(canonical, dict):
        raise HandoffError("canonical-beta donor result lacks canonical_betas")
    expected_artifact = require_file_binding(donor.get("artifact"), "canonical_beta_donor.artifact")
    if canonical.get("artifact") != expected_artifact:
        raise HandoffError("canonical-beta donor artifact binding disagrees with tracked result")
    vector_sha = require_sha(donor.get("vector_sha256"), "canonical_beta_donor.vector_sha256")
    if canonical.get("vector_sha256") != vector_sha:
        raise HandoffError("canonical-beta donor vector SHA disagrees with tracked result")
    if canonical.get("measured_height_m") is not None or canonical.get("a3_calibrated") is not False:
        raise HandoffError("canonical-beta donor must remain uncalibrated")
    remote = require_file_binding(
        donor.get("completion_manifest"), "canonical_beta_donor.completion_manifest"
    )
    if result.get("remote_completion_manifest") != remote:
        raise HandoffError("canonical-beta completion binding disagrees with tracked result")
    if donor.get("new_batch_materialization_status") != "not_run":
        raise HandoffError("new-batch canonical-beta materialization must remain not_run")


def _validate_downstream_order(plan: dict[str, Any]) -> None:
    gate = plan.get("downstream_gate")
    if not isinstance(gate, dict):
        raise HandoffError("downstream_gate must be an object")
    required_order = [
        "canonical_beta_materialization",
        "gmr_robot_retarget",
        "schema2_robot_motion_export",
        "l0_static_motion_audit",
        "vendor_l1_and_dynamics",
    ]
    if gate.get("ordered_stages") != required_order:
        raise HandoffError("downstream stage order changed")
    if gate.get("next_authorized_stage") != "canonical_beta_materialization_only":
        raise HandoffError("only canonical-beta materialization may be next")
    if gate.get("separate_exact_prereg_per_stage") is not True:
        raise HandoffError("every downstream stage requires a separate exact preregistration")
    statuses = gate.get("statuses")
    expected_statuses = {
        "canonical_beta_materialization": "not_run",
        "gmr_robot_retarget": "blocked_on_exact_canonical_beta_output",
        "schema2_robot_motion_export": "blocked_on_exact_gmr_output_and_runtime_body_order",
        "l0_static_motion_audit": "blocked_on_exact_schema2_npz",
        "vendor_l1_and_dynamics": "blocked_on_l0_and_vendor_runtime_binding",
    }
    if statuses != expected_statuses:
        raise HandoffError("downstream stage statuses changed")
    if gate.get("gmr_runtime_availability") != "external_private_dependency_not_verified_by_static_check":
        raise HandoffError("GMR runtime availability boundary changed")
    if gate.get("schema2_required_metadata") != {
        "kinematics_schema_version": 2,
        "body_pos_point": "link_origin",
        "body_lin_vel_point": "center_of_mass",
        "body_names": "exact_runtime_articulation_order_required",
    }:
        raise HandoffError("schema-2 prerequisite metadata changed")


def _validate_tracked_result_summary(
    plan: dict[str, Any], upstream: dict[str, Any], repo_root: Path
) -> None:
    summary_path = verify_repo_binding(
        upstream.get("tracked_result_summary"),
        "upstream_gvhmr.tracked_result_summary",
        repo_root,
    )
    summary = read_json(summary_path, "tracked S0/M0 GVHMR result summary")
    if summary.get("status") != "complete_structural_only":
        raise HandoffError("tracked S0/M0 summary is not complete_structural_only")
    if summary.get("formal_eligible") is not False:
        raise HandoffError("tracked S0/M0 summary must remain formal-ineligible")
    batches = summary.get("batches")
    if not isinstance(batches, list):
        raise HandoffError("tracked S0/M0 summary lacks batches")
    matches = [row for row in batches if isinstance(row, dict) and row.get("batch_id") == upstream["batch_id"]]
    if len(matches) != 1:
        raise HandoffError("tracked S0/M0 summary must contain the exact batch once")
    batch = matches[0]
    if batch.get("status") != "complete":
        raise HandoffError("tracked S0/M0 batch is not complete")
    if batch.get("preregistration") != upstream["preregistration"]:
        raise HandoffError("tracked S0/M0 preregistration binding mismatch")
    evidence = batch.get("external_evidence")
    if not isinstance(evidence, dict):
        raise HandoffError("tracked S0/M0 batch lacks external evidence")
    if evidence.get("execution_record") != upstream["execution_record"]:
        raise HandoffError("tracked S0/M0 execution-record binding mismatch")
    if evidence.get("queue_state") != upstream["queue_state"]:
        raise HandoffError("tracked S0/M0 queue-state binding mismatch")
    summary_assets = batch.get("assets")
    if not isinstance(summary_assets, list):
        raise HandoffError("tracked S0/M0 batch lacks assets")
    by_id = {
        row.get("asset_id"): row for row in summary_assets if isinstance(row, dict)
    }
    if set(by_id) != set(upstream["asset_ids"]) or len(by_id) != len(summary_assets):
        raise HandoffError("tracked S0/M0 asset set mismatch")
    for row in upstream["assets"]:
        asset_id = row["asset_id"]
        tracked = by_id[asset_id]
        expected = {
            "source_sha256": row["source_sha256"],
            "expected_frames": row["frames"],
            "actual_frames": row["frames"],
            "output": row["output"],
            "binding": row["binding"],
        }
        for field, value in expected.items():
            if tracked.get(field) != value:
                raise HandoffError(f"tracked S0/M0 {asset_id}.{field} mismatch")
        tracked_audit = tracked.get("structural_audit")
        if not isinstance(tracked_audit, dict) or tracked_audit.get("status") != "pass":
            raise HandoffError(f"tracked S0/M0 {asset_id} audit is not pass")
        for field in ("path", "bytes", "sha256"):
            if tracked_audit.get(field) != row["structural_audit"][field]:
                raise HandoffError(f"tracked S0/M0 {asset_id} audit {field} mismatch")


def _validate_batch_semantics(plan: dict[str, Any], review_by_id: dict[str, Any]) -> None:
    kind = plan.get("batch_kind")
    semantics = plan.get("semantic_guard")
    if not isinstance(semantics, dict):
        raise HandoffError("semantic_guard must be an object")
    assets = plan["upstream_gvhmr"]["assets"]
    ids = [row["asset_id"] for row in assets]
    if kind == "s0_static_high_press":
        if ids != ["static_backhand_high_press"]:
            raise HandoffError("S0 must contain only static_backhand_high_press")
        expected = {
            "motion_role": "fifth_action_backhand_high_ball_forward_downward_press",
            "question_family": "separate_high_ball_high_press_paper_required_not_yet_preregistered",
            "pull_or_loop_question_paper_allowed": False,
            "observed_ball_contact": None,
            "strike_effectiveness": None,
            "claim_boundary": "GVHMR structural reconstruction only; no strike, return, safety, or behavior claim",
        }
        if semantics != expected:
            raise HandoffError("S0 semantic guard changed or borrowed a pull/loop claim")
    elif kind == "m0_lateral_teachers":
        expected_ids = [
            "lateral_step_left_1",
            "lateral_step_left_2",
            "lateral_step_right_1",
            "lateral_step_right_2",
        ]
        if ids != expected_ids:
            raise HandoffError("M0 asset order changed")
        stance = semantics.get("terminal_stance_contract")
        expected_stance = {
            "measurement_stage": "exact_robot_coordinate_gmr_before_schema2_promotion",
            "heading_and_root_normalization": "remove common root XY translation and rotate heading to the candidate initial-ready frame",
            "vector_definition": "d_xy = right_foot_xy - left_foot_xy",
            "initial_reference": "coordinatewise robust median d_xy over the exact ready_before window",
            "terminal_measurement": "coordinatewise robust median d_xy over the exact ready_after window",
            "must_preserve_components": ["lateral_separation", "fore_aft_stagger"],
            "feet_together_or_narrower_substitute_allowed": False,
            "absolute_foot_pose_equality_required": False,
            "numeric_tolerance_status": "must_be_preregistered_before_gmr_result_acceptance",
            "failure_semantics": "missing foot-site mapping, missing numeric tolerance, or narrower terminal stance fails closed",
        }
        if stance != expected_stance:
            raise HandoffError("M0 terminal stance vector contract changed")
        if semantics.get("motion_role") != "signed_lateral_displacement_conditioned_lower_body_teacher":
            raise HandoffError("M0 motion role changed")
        if semantics.get("gvhmr_stance_result") is not None:
            raise HandoffError("GVHMR cannot claim a robot stance result")
    else:
        raise HandoffError(f"unsupported batch_kind {kind!r}")

    for asset_id in ids:
        review = review_by_id.get(asset_id)
        if not isinstance(review, dict):
            raise HandoffError(f"manual event review lacks {asset_id}")
        expected = next(row for row in assets if row["asset_id"] == asset_id)
        if expected.get("ready_before_window_s") != review.get("ready_before_window_s"):
            raise HandoffError(f"{asset_id} ready-before window mismatch")
        if expected.get("ready_after_window_s") != review.get("ready_after_window_s"):
            raise HandoffError(f"{asset_id} ready-after window mismatch")


def validate_plan(
    plan_path: Path,
    expected_plan_sha256: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    expected = require_sha(expected_plan_sha256, "--expected-prereg-sha256")
    actual = sha256_file(plan_path)
    if actual != expected:
        raise HandoffError(f"post-GVHMR prereg sha256 {actual} != {expected}")
    plan = read_json(plan_path, "post-GVHMR preregistration")
    if plan.get("schema_version") != 1:
        raise HandoffError("post-GVHMR prereg schema_version must be 1")
    if plan.get("status") != "preregistered_not_executed":
        raise HandoffError("post-GVHMR prereg must remain preregistered_not_executed")
    if plan.get("formal_eligible") is not False or plan.get("training_authorized") is not False:
        raise HandoffError("post-GVHMR handoff must remain formal-ineligible and training-blocked")
    if plan.get("hardware_authorized") is not False:
        raise HandoffError("hardware must remain unauthorized")

    tool_path = verify_repo_binding(plan.get("consumer"), "consumer", repo_root)
    if tool_path != Path(__file__).resolve():
        raise HandoffError("consumer binding does not name this exact tool")

    upstream = plan.get("upstream_gvhmr")
    if not isinstance(upstream, dict):
        raise HandoffError("upstream_gvhmr must be an object")
    upstream_path = verify_repo_binding(
        upstream.get("preregistration"), "upstream_gvhmr.preregistration", repo_root
    )
    try:
        static = validate_static_prereg(upstream_path, repo_root=repo_root)
    except PreregError as exc:
        raise HandoffError(f"upstream GVHMR prereg failed static validation: {exc}") from None
    prereg = static["prereg"]
    if upstream.get("batch_id") != prereg["execution_batch"]["batch_id"]:
        raise HandoffError("upstream batch id mismatch")
    asset_ids = _require_string_list(upstream.get("asset_ids"), "upstream_gvhmr.asset_ids")
    if asset_ids != prereg["execution_batch"]["asset_ids"]:
        raise HandoffError("upstream asset order mismatch")
    if upstream.get("state_root") != prereg["pod_contract"]["state_root"]:
        raise HandoffError("upstream state root mismatch")

    _validate_tracked_result_summary(plan, upstream, repo_root)

    for label in ("execution_record", "queue_state"):
        require_file_binding(upstream.get(label), f"upstream_gvhmr.{label}")
    if upstream["execution_record"]["path"] != prereg["pod_contract"]["execution_record_path"]:
        raise HandoffError("execution record path mismatch")
    if upstream["queue_state"]["path"] != str(Path(upstream["state_root"]) / "queue_state.json"):
        raise HandoffError("queue state path mismatch")

    intake_by_id = {row["id"]: row for row in static["intake"]["assets"]}
    outputs_by_id = {
        row["asset_id"]: row for row in prereg["output_contract"]["outputs"]
    }
    rows = upstream.get("assets")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows if isinstance(row, dict)] != asset_ids:
        raise HandoffError("upstream asset bindings must exactly match the ordered batch")
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HandoffError(f"upstream_gvhmr.assets[{index}] must be an object")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
            raise HandoffError(f"unsafe asset id {asset_id!r}")
        source = intake_by_id[asset_id]
        if row.get("source_sha256") != source["sha256"]:
            raise HandoffError(f"{asset_id} source SHA mismatch")
        if row.get("frames") != source["media"]["frames"]:
            raise HandoffError(f"{asset_id} frame count mismatch")
        for field in ("output", "binding", "structural_audit"):
            binding = require_file_binding(row.get(field), f"{asset_id}.{field}")
            if binding["path"] in seen_paths:
                raise HandoffError(f"duplicate external evidence path: {binding['path']}")
            seen_paths.add(binding["path"])
        if row["output"]["path"] != outputs_by_id[asset_id]["path"]:
            raise HandoffError(f"{asset_id} output path mismatch")
        if row["binding"]["path"] != str(Path(upstream["state_root"]) / "bindings" / f"{asset_id}.json"):
            raise HandoffError(f"{asset_id} binding path mismatch")
        if row["structural_audit"]["path"] != str(Path(upstream["state_root"]) / "audits" / f"{asset_id}.json"):
            raise HandoffError(f"{asset_id} audit path mismatch")

    manual_review = read_json(static["review_path"], "manual event review")
    review_rows = manual_review.get("assets")
    if not isinstance(review_rows, list):
        raise HandoffError("manual event review assets are missing")
    review_by_id = {row.get("asset_id"): row for row in review_rows if isinstance(row, dict)}
    _validate_batch_semantics(plan, review_by_id)
    _validate_canonical_donor(plan, repo_root)
    _validate_downstream_order(plan)

    output = plan.get("handoff_output")
    if not isinstance(output, dict):
        raise HandoffError("handoff_output must be an object")
    root = output.get("root")
    path = output.get("path")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise HandoffError("handoff_output.root must be absolute")
    if path != str(Path(root) / "handoff.json"):
        raise HandoffError("handoff_output.path must be <root>/handoff.json")
    required_output = {
        "root_must_not_exist": True,
        "no_clobber": True,
        "write": "O_CREAT|O_EXCL_then_fsync",
        "publish_after_all_validation": True,
    }
    for field, value in required_output.items():
        if output.get(field) != value:
            raise HandoffError(f"handoff_output.{field} must be {value!r}")
    return plan, static, actual


def _check_binding_contract(
    plan: dict[str, Any],
    row: dict[str, Any],
    binding: dict[str, Any],
    *,
    record_sha: str,
) -> None:
    upstream = plan["upstream_gvhmr"]
    asset_id = row["asset_id"]
    expected_scalars = {
        "status": "complete",
        "asset_id": asset_id,
        "source_sha256": row["source_sha256"],
        "output_path": row["output"]["path"],
        "output_bytes": row["output"]["bytes"],
        "output_sha256": row["output"]["sha256"],
        "returncode": 0,
        "audit_returncode": 0,
        "gvhmr_commit": "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
        "structural_audit_path": row["structural_audit"]["path"],
        "structural_audit_sha256": row["structural_audit"]["sha256"],
    }
    for field, expected in expected_scalars.items():
        if binding.get(field) != expected:
            raise HandoffError(
                f"{asset_id} binding {field} mismatch: {binding.get(field)!r} != {expected!r}"
            )
    processing = binding.get("processing_contract")
    if not isinstance(processing, dict):
        raise HandoffError(f"{asset_id} binding lacks processing_contract")
    expected_processing = {
        "manifest_sha256": plan["intake_sha256"],
        "manual_event_review_sha256": plan["manual_event_review_sha256"],
        "prereg_sha256": upstream["preregistration"]["sha256"],
        "execution_record_sha256": record_sha,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "source_consumption": SOURCE_CONSUMPTION,
        "gvhmr_commit": "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
    }
    for field, expected in expected_processing.items():
        if processing.get(field) != expected:
            raise HandoffError(f"{asset_id} processing contract {field} mismatch")
    snapshots = processing.get("bound_source_snapshot_sha256")
    if not isinstance(snapshots, dict) or snapshots.get(asset_id) != row["source_sha256"]:
        raise HandoffError(f"{asset_id} bound source snapshot SHA mismatch")


def _verify_canonical_runtime(plan: dict[str, Any]) -> dict[str, Any]:
    donor = plan["canonical_beta_donor"]
    artifact_path = verify_bound_file(donor["artifact"], "canonical beta donor artifact")
    artifact = read_json(artifact_path, "canonical beta donor artifact")
    if artifact.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise HandoffError("canonical beta artifact contract mismatch")
    if artifact.get("vector_sha256") != donor["vector_sha256"]:
        raise HandoffError("canonical beta artifact vector SHA mismatch")
    components = artifact.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 10
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in components)
    ):
        raise HandoffError("canonical beta artifact must contain ten finite components")
    if artifact.get("measured_height_m") is not None or artifact.get("a3_calibrated") is not False:
        raise HandoffError("canonical beta artifact must remain uncalibrated")
    completion_path = verify_bound_file(
        donor["completion_manifest"], "canonical beta completion manifest"
    )
    completion = read_json(completion_path, "canonical beta completion manifest")
    if completion.get("status") != "complete":
        raise HandoffError("canonical beta completion manifest is not complete")
    bound = completion.get("canonical_betas_artifact")
    if not isinstance(bound, dict):
        raise HandoffError("canonical beta completion lacks artifact binding")
    if (
        bound.get("path") != donor["artifact"]["path"]
        or bound.get("sha256") != donor["artifact"]["sha256"]
        or bound.get("vector_sha256") != donor["vector_sha256"]
    ):
        raise HandoffError("canonical beta completion/artifact binding mismatch")
    return {
        "artifact": donor["artifact"],
        "completion_manifest": donor["completion_manifest"],
        "vector_sha256": donor["vector_sha256"],
        "new_batch_materialization_status": "not_run",
    }


def inspect_runtime_evidence(plan: dict[str, Any]) -> dict[str, Any]:
    upstream = plan["upstream_gvhmr"]
    record_path = verify_bound_file(upstream["execution_record"], "GVHMR execution record")
    record_sha = upstream["execution_record"]["sha256"]
    record = read_json(record_path, "GVHMR execution record")
    expected_record = {
        "status": "ready_for_exact_offline_gvhmr",
        "authorization_scope": AUTHORIZATION_SCOPE,
        "prereg_sha256": upstream["preregistration"]["sha256"],
        "manifest_sha256": plan["intake_sha256"],
        "manual_event_review_sha256": plan["manual_event_review_sha256"],
        "state_root": upstream["state_root"],
        "processing_order": upstream["asset_ids"],
        "downstream_consumer": "none_beyond_structural_auditor",
    }
    for field, expected in expected_record.items():
        if record.get(field) != expected:
            raise HandoffError(f"execution record {field} mismatch")

    queue_path = verify_bound_file(upstream["queue_state"], "GVHMR queue state")
    queue = read_json(queue_path, "GVHMR queue state")
    expected_queue = {
        "schema_version": 2,
        "status": "complete",
        "authorization_scope": AUTHORIZATION_SCOPE,
        "prereg_sha256": upstream["preregistration"]["sha256"],
        "execution_record_sha256": record_sha,
        "asset_ids": upstream["asset_ids"],
        "batch_id": upstream["batch_id"],
        "source_consumption": SOURCE_CONSUMPTION,
    }
    for field, expected in expected_queue.items():
        if queue.get(field) != expected:
            raise HandoffError(f"queue state {field} mismatch")
    if "failed_asset_id" in queue or "failure" in queue:
        raise HandoffError("complete queue state contains a failure marker")
    snapshots = queue.get("bound_source_snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(upstream["asset_ids"]):
        raise HandoffError("queue source snapshot set mismatch")

    expected_binding_names = {f"{asset_id}.json" for asset_id in upstream["asset_ids"]}
    binding_dir = Path(upstream["state_root"]) / "bindings"
    audit_dir = Path(upstream["state_root"]) / "audits"
    if {path.name for path in binding_dir.glob("*.json")} != expected_binding_names:
        raise HandoffError("binding directory contains a missing or unexpected JSON binding")
    if {path.name for path in audit_dir.glob("*.json")} != expected_binding_names:
        raise HandoffError("audit directory contains a missing or unexpected JSON report")

    result_rows: list[dict[str, Any]] = []
    for row in upstream["assets"]:
        asset_id = row["asset_id"]
        output_path = verify_bound_file(row["output"], f"{asset_id} GVHMR output")
        binding_path = verify_bound_file(row["binding"], f"{asset_id} queue binding")
        audit_path = verify_bound_file(row["structural_audit"], f"{asset_id} structural audit")
        binding = read_json(binding_path, f"{asset_id} queue binding")
        _check_binding_contract(plan, row, binding, record_sha=record_sha)
        audit = read_json(audit_path, f"{asset_id} structural audit")
        expected_audit = {
            "status": "pass",
            "result_path": str(output_path),
            "result_bytes": row["output"]["bytes"],
            "result_sha256": row["output"]["sha256"],
            "expected_frames": row["frames"],
            "actual_frames": row["frames"],
        }
        for field, expected in expected_audit.items():
            if audit.get(field) != expected:
                raise HandoffError(f"{asset_id} structural audit {field} mismatch")
        finite = audit.get("finite_elements")
        if isinstance(finite, bool) or not isinstance(finite, int) or finite <= 0:
            raise HandoffError(f"{asset_id} structural audit finite_elements is invalid")
        snapshot = snapshots[asset_id]
        if not isinstance(snapshot, dict):
            raise HandoffError(f"{asset_id} queue snapshot record is malformed")
        before = snapshot.get("snapshot_before")
        if not isinstance(before, dict) or before.get("sha256") != row["source_sha256"]:
            raise HandoffError(f"{asset_id} queue snapshot SHA mismatch")
        result_rows.append(
            {
                "asset_id": asset_id,
                "source_sha256": row["source_sha256"],
                "frames": row["frames"],
                "gvhmr_output": row["output"],
                "queue_binding": row["binding"],
                "structural_audit": row["structural_audit"],
                "finite_elements": finite,
                "ready_before_window_s": row["ready_before_window_s"],
                "ready_after_window_s": row["ready_after_window_s"],
            }
        )

    canonical = _verify_canonical_runtime(plan)
    return {
        "execution_record": upstream["execution_record"],
        "queue_state": upstream["queue_state"],
        "batch_id": upstream["batch_id"],
        "asset_ids": upstream["asset_ids"],
        "results": result_rows,
        "canonical_beta_donor": canonical,
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    ensure_no_symlink_components(path.parent, "handoff output parent")
    ensure_no_symlink_components(path, "handoff output", leaf_may_be_missing=True)
    data = _json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise HandoffError(f"refusing to overwrite handoff: {path}") from None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def publish_handoff(
    plan: dict[str, Any],
    plan_sha256: str,
    evidence: dict[str, Any],
) -> Path:
    output = plan["handoff_output"]
    root = Path(output["root"])
    parent = root.parent
    ensure_no_symlink_components(parent, "handoff root parent")
    ensure_no_symlink_components(root, "handoff root", leaf_may_be_missing=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        raise HandoffError(f"no-clobber handoff root already exists: {root}") from None
    fsync_directory(parent)
    handoff = {
        "schema_version": 1,
        "status": HANDOFF_STATUS,
        "scope": "exact post-GVHMR lineage handoff only; no downstream stage or behavior accepted",
        "plan_sha256": plan_sha256,
        "consumer_sha256": plan["consumer"]["sha256"],
        "batch_kind": plan["batch_kind"],
        "formal_eligible": False,
        "training_authorized": False,
        "hardware_authorized": False,
        "runtime_evidence": evidence,
        "semantic_guard": plan["semantic_guard"],
        "downstream_gate": plan["downstream_gate"],
        "accepted_claims": [
            "the exact preregistered GVHMR outputs are structurally finite and fully bound to queue evidence",
            "the exact canonical-beta donor artifact is available for a separate no-clobber materialization",
        ],
        "not_claimed": [
            "canonical betas have been materialized into this new batch",
            "GMR or schema-2 output exists",
            "robot foot stance, contact, balance, safety, strike, return, or Gate3 behavior passed",
        ],
    }
    path = Path(output["path"])
    try:
        write_exclusive(path, handoff)
    except BaseException:
        # Preserve a claimed root as evidence of the interrupted publication.
        raise
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("static", help="validate only the committed machine contract")
    subparsers.add_parser("inspect", help="validate runtime evidence without writing")
    subparsers.add_parser("consume", help="validate runtime evidence and publish once")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan, _static, plan_sha = validate_plan(
            args.prereg.resolve(), args.expected_prereg_sha256
        )
        if args.command == "static":
            print(
                "[post-gvhmr] PASS static "
                f"batch={plan['upstream_gvhmr']['batch_id']} prereg_sha256={plan_sha}"
            )
            return 0
        evidence = inspect_runtime_evidence(plan)
        if args.command == "inspect":
            print(
                "[post-gvhmr] PASS inspect "
                f"batch={evidence['batch_id']} assets={len(evidence['results'])}"
            )
            return 0
        path = publish_handoff(plan, plan_sha, evidence)
        print(f"[post-gvhmr] PASS consume handoff={path}")
        return 0
    except (HandoffError, OSError, PreregError, TypeError, ValueError) as exc:
        print(f"[post-gvhmr] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
