#!/usr/bin/env python3
"""Validate and attest the exact static/lateral offline-GVHMR batches.

The static check is host-only.  ``attest`` is intended to run on the selected
Pod after the private videos have been staged.  It repeats the byte/media
audit, binds the clean GVHMR source, checkpoint tree and Python environment,
checks every output and state target is unused, then creates exactly one
execution record with O_EXCL.  The GVHMR queue re-runs the same live checks
before it creates the state root.

This contract authorizes only video-to-SMPL-X reconstruction and its structural
auditor.  It does not authorize GMR, schema-2 robot motion, simulation, TOPP,
RL, deployment or hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_motion_video_intake import (  # noqa: E402
    audit_assets,
    load_manifest,
    resolve_asset_path,
    sha256_file,
)


class PreregError(RuntimeError):
    """A static or live preregistration contract failed."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ALLOWED_STAGES = [
    "pod_stage_content_and_media_audit",
    "offline_gvhmr_video_to_smplx",
    "gvhmr_structural_output_audit",
]
REQUIRED_FORBIDDEN_STAGES = {
    "canonical_beta_materialization",
    "gmr_robot_retarget",
    "schema2_robot_motion_export",
    "simulator_replay",
    "topp_or_other_retime",
    "rl_training_or_evaluation",
    "deployment_or_hardware",
}
ACCEPTED_INTAKE_SHA256 = "44b00b3c46c837d797990bc6f6255055c0ff83c1bb8643ca81f9707033ca304c"
ACCEPTED_FRANCO_REVIEW_SHA256 = (
    "9e2a7a51c443d53d7b8ed5c39d02ca0a523f59eb195b1cfc2a335041126498f1"
)
ALLOWED_PREREGISTRATIONS = {
    "motion-video-gvhmr-static-high-press-20260713-s0-v1": {
        "filename": "motion_video_gvhmr_prereg_20260713.json",
        "sha256": "c610366e7e382b20f9b64b01a9c57b2722b72be501ada2aa16c24e350207f1ba",
        "batch_id": "static_high_press_s0_v1",
        "asset_ids": ["static_backhand_high_press"],
        "peer_filename": "motion_video_gvhmr_motion_prereg_20260713.json",
        "peer_id": "motion-video-gvhmr-lateral-teachers-20260713-m0-v1",
    },
    "motion-video-gvhmr-lateral-teachers-20260713-m0-v1": {
        "filename": "motion_video_gvhmr_motion_prereg_20260713.json",
        "sha256": "19794d62446335c2d125564d9ea7ee77e59e1aed39f7aeef4b4039843dce0f08",
        "batch_id": "lateral_teachers_m0_v1",
        "asset_ids": [
            "lateral_step_left_1",
            "lateral_step_left_2",
            "lateral_step_right_1",
            "lateral_step_right_2",
        ],
        "peer_filename": "motion_video_gvhmr_prereg_20260713.json",
        "peer_id": "motion-video-gvhmr-static-high-press-20260713-s0-v1",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_constant(value: str) -> None:
    raise PreregError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreregError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PreregError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise PreregError(f"{label} must be a JSON object: {path}")
    return payload


def resolve_repo_file(repo_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreregError(f"{label}.path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PreregError(f"{label}.path must be a safe repository-relative path")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PreregError(f"{label}.path escapes repository root: {relative}") from None
    if not resolved.is_file():
        raise PreregError(f"{label} is missing: {resolved}")
    return resolved


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PreregError(f"{label} must be a lowercase SHA-256")
    return value


def require_exact_file(binding: Any, repo_root: Path, label: str) -> Path:
    if not isinstance(binding, dict):
        raise PreregError(f"{label} binding must be an object")
    path = resolve_repo_file(repo_root, binding.get("path"), label)
    expected = require_sha(binding.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise PreregError(f"{label} SHA mismatch: expected={expected} actual={actual}")
    return path


def _require_window(row: dict[str, Any], field: str, duration: float) -> tuple[float, float]:
    raw = row.get(field)
    if not isinstance(raw, list) or len(raw) != 2:
        raise PreregError(f"{row.get('asset_id')}.{field} must have two numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise PreregError(f"{row.get('asset_id')}.{field} must have two numbers")
    start, end = float(raw[0]), float(raw[1])
    if not all(math.isfinite(value) for value in (start, end)):
        raise PreregError(f"{row.get('asset_id')}.{field} must be finite")
    if not (0.0 <= start < end <= duration + 1e-6):
        raise PreregError(
            f"{row.get('asset_id')}.{field} must satisfy 0 <= start < end <= duration"
        )
    return start, end


def validate_manual_review(
    review: dict[str, Any], intake: dict[str, Any], expected_order: list[str]
) -> None:
    if review.get("schema_version") != 1:
        raise PreregError("manual event review schema_version must be 1")
    if review.get("processing_order") != expected_order:
        raise PreregError("manual event review processing_order mismatch")
    intake_binding = review.get("intake")
    if not isinstance(intake_binding, dict):
        raise PreregError("manual event review intake binding must be an object")
    if intake_binding.get("sha256") != "44b00b3c46c837d797990bc6f6255055c0ff83c1bb8643ca81f9707033ca304c":
        raise PreregError("manual event review does not bind the accepted intake SHA")
    method = review.get("review_method")
    if not isinstance(method, dict) or "never an observed ball contact" not in str(
        method.get("anchor_semantics", "")
    ):
        raise PreregError("manual review must explicitly deny ball-contact truth")

    intake_by_id = {str(row["id"]): row for row in intake["assets"]}
    rows = review.get("assets")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows] != expected_order:
        raise PreregError("manual review assets must exactly follow processing_order")
    for row in rows:
        asset_id = str(row.get("asset_id"))
        source = intake_by_id[asset_id]
        if row.get("source_sha256") != source["sha256"]:
            raise PreregError(f"manual review source SHA mismatch for {asset_id}")
        duration = float(source["media"]["duration_s"])
        if abs(float(row.get("duration_s", -1.0)) - duration) > 1e-6:
            raise PreregError(f"manual review duration mismatch for {asset_id}")
        before = _require_window(row, "ready_before_window_s", duration)
        event = _require_window(row, "nominal_event_window_s", duration)
        after = _require_window(row, "ready_after_window_s", duration)
        anchor = row.get("nominal_event_anchor_s")
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
            raise PreregError(f"{asset_id}.nominal_event_anchor_s must be numeric")
        anchor = float(anchor)
        if not math.isfinite(anchor) or not event[0] <= anchor <= event[1]:
            raise PreregError(f"{asset_id} nominal anchor must lie inside its event window")
        if not before[1] < event[0] or not event[1] < after[0]:
            raise PreregError(f"{asset_id} ready/event windows must be ordered and disjoint")
        if row.get("review_status") != "nominal_air_swing_visual_review_complete":
            raise PreregError(f"{asset_id} manual review status is not complete")

    stance = review.get("foot_stance_contract")
    if not isinstance(stance, dict) or "initial ready window" not in str(stance.get("rule", "")):
        raise PreregError("manual review must preserve the initial ready-window foot separation")
    expected_lateral = [
        "lateral_step_left_1",
        "lateral_step_left_2",
        "lateral_step_right_1",
        "lateral_step_right_2",
    ]
    if stance.get("applies_to") != expected_lateral:
        raise PreregError("foot-stance contract must bind exactly the four lateral candidates")
    if review.get("downstream_policy", {}).get("training_or_simulator_authorized") is not False:
        raise PreregError("manual review must not authorize simulation or training")
    if review.get("downstream_policy", {}).get("hardware_authorized") is not False:
        raise PreregError("manual review must not authorize hardware")


def validate_franco_priority_review(
    review: dict[str, Any], repo_root: Path = REPO_ROOT
) -> None:
    """Bind the retained Franco B/C nominal anchors without inventing contact truth."""

    if review.get("schema_version") != 1:
        raise PreregError("Franco priority review schema_version must be 1")
    if review.get("status") != "nominal_visual_review_complete_contact_truth_null":
        raise PreregError("Franco priority review status changed")
    if review.get("human_owner") != "Franco" or review.get("executor") != "Codex":
        raise PreregError("Franco priority review owner/executor changed")
    intake_path = require_exact_file(review.get("intake"), repo_root, "Franco review intake")
    results_path = require_exact_file(
        review.get("gvhmr_results"), repo_root, "Franco review GVHMR results"
    )
    require_exact_file(
        review.get("counterfactual_results"),
        repo_root,
        "Franco review counterfactual results",
    )
    intake = load_manifest(intake_path)
    results = read_json(results_path, "Franco GVHMR results")
    method = review.get("review_method")
    expected_ids = ["franco_backhand_loop_b", "franco_backhand_loop_c"]
    if not isinstance(method, dict) or method.get("candidate_order") != expected_ids:
        raise PreregError("Franco priority review must bind B then C")
    if "contact_truth remains null" not in str(method.get("anchor_semantics", "")):
        raise PreregError("Franco priority review must keep contact truth null")
    if method.get("frame_index_semantics") != "zero-based decoded source-video frame index at 30 fps":
        raise PreregError("Franco priority review frame semantics changed")

    intake_by_id = {str(row["id"]): row for row in intake["assets"]}
    result_by_id = {str(row["asset_id"]): row for row in results.get("results", [])}
    expected_anchors = {
        "franco_backhand_loop_b": ([45, 53], 49, 1.6333333333333333),
        "franco_backhand_loop_c": ([46, 54], 50, 1.6666666666666667),
    }
    rows = review.get("assets")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows] != expected_ids:
        raise PreregError("Franco priority review assets must be exactly B then C")
    for row in rows:
        asset_id = str(row["asset_id"])
        source = intake_by_id[asset_id]
        result = result_by_id[asset_id]
        if row.get("source_relpath") != source["source_relpath"]:
            raise PreregError(f"Franco review source path mismatch for {asset_id}")
        if row.get("source_bytes") != source["bytes"] or row.get("source_sha256") != source["sha256"]:
            raise PreregError(f"Franco review source binding mismatch for {asset_id}")
        if row.get("source_frames") != source["media"]["frames"] or row.get("source_fps") != 30.0:
            raise PreregError(f"Franco review media binding mismatch for {asset_id}")
        if row.get("gvhmr_result_sha256") != result["result_sha256"]:
            raise PreregError(f"Franco review GVHMR result mismatch for {asset_id}")
        window, anchor, anchor_s = expected_anchors[asset_id]
        if (
            row.get("nominal_event_frame_window") != window
            or row.get("nominal_event_anchor_frame") != anchor
            or row.get("nominal_event_anchor_s") != anchor_s
        ):
            raise PreregError(f"Franco review nominal anchor changed for {asset_id}")
        if row.get("contact_truth", "not-null") is not None:
            raise PreregError(f"Franco review contact truth must remain null for {asset_id}")
    boundary = review.get("existing_evidence_boundary")
    if not isinstance(boundary, dict):
        raise PreregError("Franco review evidence boundary must be an object")
    for field in (
        "gvhmr_rerun_authorized",
        "spatial_retarget_or_schema2_authorized",
        "training_or_simulator_authorized",
        "hardware_authorized",
    ):
        if boundary.get(field) is not False:
            raise PreregError(f"Franco review must keep {field}=false")


def validate_static_prereg(prereg_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    prereg_path = prereg_path.resolve()
    prereg = read_json(prereg_path, "GVHMR preregistration")
    if prereg.get("schema_version") != 1:
        raise PreregError("prereg schema_version must be 1")
    if prereg.get("status") != "offline_gvhmr_authorized_subject_to_exact_pod_attestation":
        raise PreregError("prereg status does not authorize the bounded offline preflight")
    if prereg.get("human_owner") != "Franco" or prereg.get("executor") != "Codex":
        raise PreregError("prereg human owner/executor fields changed")
    prereg_id = prereg.get("prereg_id")
    batch_spec = ALLOWED_PREREGISTRATIONS.get(prereg_id)
    if batch_spec is None:
        raise PreregError("prereg id is not one of the exact S0/M0 contracts")
    if prereg_path.name != batch_spec["filename"]:
        raise PreregError("prereg filename does not match its exact S0/M0 identity")

    intake_path = require_exact_file(prereg.get("intake"), repo_root, "intake")
    intake_binding = prereg["intake"]
    if intake_binding.get("sha256") != ACCEPTED_INTAKE_SHA256:
        raise PreregError("prereg must bind the accepted 2026-07-13 intake SHA")
    intake = load_manifest(intake_path)
    expected_order = [str(value) for value in prereg.get("processing_order", [])]
    if expected_order != intake.get("processing_order"):
        raise PreregError("prereg processing_order must exactly match intake processing_order")
    if len(expected_order) != 7 or len(set(expected_order)) != 7:
        raise PreregError("prereg must bind exactly seven unique assets")
    batch = prereg.get("execution_batch")
    if not isinstance(batch, dict):
        raise PreregError("execution_batch must be an object")
    batch_order = batch.get("asset_ids")
    if batch.get("batch_id") != batch_spec["batch_id"] or batch_order != batch_spec["asset_ids"]:
        raise PreregError("execution batch does not match its exact S0/M0 contract")
    if batch.get("v12_authorized") is not False:
        raise PreregError("v12 must remain unauthorized in S0/M0")
    if batch.get("franco_gvhmr_rerun_authorized") is not False:
        raise PreregError("Franco GVHMR rerun must remain unauthorized")
    if batch.get("candidate_seed_repeats") != 0:
        raise PreregError("offline motion candidates must not repeat seed-style executions")
    if prereg_id == "motion-video-gvhmr-lateral-teachers-20260713-m0-v1":
        stance = prereg.get("future_robot_stance_contract")
        if not isinstance(stance, dict) or stance.get("status") != "not_evaluated_by_gvhmr":
            raise PreregError("M0 stance must remain an explicitly unevaluated future contract")
        if stance.get("applies_to") != batch_order:
            raise PreregError("M0 stance contract must bind exactly the four execution assets")
        rule = str(stance.get("rule", ""))
        if "initial-ready-window vector" not in rule or "fore-aft offset" not in rule:
            raise PreregError("M0 stance must preserve the aligned initial foot-separation vector")
        if stance.get("narrower_feet_together_substitute_allowed") is not False:
            raise PreregError("M0 must reject a narrower feet-together terminal substitute")

    review_path = require_exact_file(
        prereg.get("manual_event_review"), repo_root, "manual_event_review"
    )
    review = read_json(review_path, "manual event review")
    validate_manual_review(review, intake, expected_order)

    franco_review_path = require_exact_file(
        prereg.get("franco_priority_context"), repo_root, "franco_priority_context"
    )
    if prereg["franco_priority_context"].get("sha256") != ACCEPTED_FRANCO_REVIEW_SHA256:
        raise PreregError("prereg must bind the accepted Franco B/C visual review")
    franco_review = read_json(franco_review_path, "Franco priority visual review")
    validate_franco_priority_review(franco_review, repo_root=repo_root)

    authorization = prereg.get("authorization")
    if not isinstance(authorization, dict):
        raise PreregError("authorization must be an object")
    if authorization.get("allowed_stages") != REQUIRED_ALLOWED_STAGES:
        raise PreregError("allowed stages must be the exact offline GVHMR closure")
    forbidden = authorization.get("forbidden_stages")
    if (
        not isinstance(forbidden, list)
        or len(forbidden) != len(set(forbidden))
        or set(forbidden) != REQUIRED_FORBIDDEN_STAGES
    ):
        raise PreregError("forbidden stages do not close all downstream execution")
    if authorization.get("formal_eligible") is not False:
        raise PreregError("GVHMR preregistration must remain formal_eligible=false")

    tools = prereg.get("tool_closure")
    if not isinstance(tools, dict):
        raise PreregError("tool_closure must be an object")
    tool_paths = {
        name: require_exact_file(tools.get(name), repo_root, f"tool_closure.{name}")
        for name in (
            "intake_auditor",
            "queue",
            "result_auditor",
            "legacy_intake_guard",
            "historical_20260711_queue_source_archive",
        )
    }
    validator_binding = tools.get("execution_validator")
    if not isinstance(validator_binding, dict):
        raise PreregError("execution_validator binding must be an object")
    validator_path = resolve_repo_file(
        repo_root, validator_binding.get("path"), "tool_closure.execution_validator"
    )
    if validator_path.resolve() != Path(__file__).resolve():
        raise PreregError("execution validator path does not identify this program")

    runtime = prereg.get("gvhmr_runtime")
    if not isinstance(runtime, dict):
        raise PreregError("gvhmr_runtime must be an object")
    if not isinstance(runtime.get("root"), str) or not Path(runtime["root"]).is_absolute():
        raise PreregError("GVHMR root must be absolute")
    if not isinstance(runtime.get("commit"), str) or not GIT_SHA_RE.fullmatch(runtime["commit"]):
        raise PreregError("GVHMR commit must be an exact 40-character SHA")
    entrypoint_relpath = runtime.get("entrypoint_relpath")
    if (
        entrypoint_relpath != "tools/demo/demo.py"
        or Path(str(entrypoint_relpath)).is_absolute()
        or ".." in Path(str(entrypoint_relpath)).parts
    ):
        raise PreregError("GVHMR entrypoint must remain the safe tools/demo/demo.py path")
    if runtime.get("clean_worktree_required") is not True or runtime.get("static_camera") is not True:
        raise PreregError("GVHMR clean-worktree and static-camera requirements must stay enabled")
    checkpoints = runtime.get("checkpoint_tree")
    if not isinstance(checkpoints, dict):
        raise PreregError("checkpoint_tree must be an object")
    if checkpoints.get("relpath") != "inputs/checkpoints":
        raise PreregError("checkpoint tree relpath changed")
    for field in ("files", "bytes"):
        if isinstance(checkpoints.get(field), bool) or not isinstance(checkpoints.get(field), int):
            raise PreregError(f"checkpoint_tree.{field} must be an integer")
    require_sha(checkpoints.get("sha256"), "checkpoint_tree.sha256")
    python = runtime.get("python")
    if not isinstance(python, dict) or not Path(str(python.get("executable", ""))).is_absolute():
        raise PreregError("motion Python executable must be absolute")
    require_sha(python.get("pip_freeze_sha256"), "python.pip_freeze_sha256")
    gpu = runtime.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("allowed_physical_indices") != [0, 1, 2]:
        raise PreregError("GPU placement contract changed")
    if gpu.get("max_used_mib_before_each_asset") != 19000:
        raise PreregError("GPU memory gate changed")
    if gpu.get("poll_seconds") != 30.0 or gpu.get("wait_timeout_seconds") != 0.0:
        raise PreregError("GPU polling/timeout contract changed")
    nvidia_smi = gpu.get("nvidia_smi")
    if not isinstance(nvidia_smi, dict) or nvidia_smi.get("absolute_path") != "/usr/bin/nvidia-smi":
        raise PreregError("nvidia-smi must remain bound to the absolute /usr/bin/nvidia-smi path")

    pod = prereg.get("pod_contract")
    if not isinstance(pod, dict) or pod.get("allowed_pod_ids") != ["pod1", "pod2"]:
        raise PreregError("Pod contract changed")
    for field in ("staged_source_root", "execution_record_path", "state_root"):
        value = pod.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise PreregError(f"pod_contract.{field} must be absolute")
    if len({pod["staged_source_root"], pod["execution_record_path"], pod["state_root"]}) != 3:
        raise PreregError("Pod source, record and state paths must be distinct")

    output = prereg.get("output_contract")
    if not isinstance(output, dict) or output.get("no_clobber") is not True:
        raise PreregError("output contract must be no-clobber")
    output_root = Path(str(output.get("root", "")))
    gvhmr_root = Path(runtime["root"])
    if output_root != gvhmr_root / "outputs" / "demo":
        raise PreregError("output root must be derived from the exact GVHMR root")
    rows = output.get("outputs")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows] != batch_order:
        raise PreregError("output rows must exactly follow the execution batch")
    intake_by_id = {str(row["id"]): row for row in intake["assets"]}
    output_paths: list[Path] = []
    for row in rows:
        asset_id = str(row["asset_id"])
        expected = output_root / Path(str(intake_by_id[asset_id]["source_relpath"])).stem / "hmr4d_results.pt"
        actual = Path(str(row.get("path", "")))
        if actual != expected:
            raise PreregError(f"derived output path mismatch for {asset_id}")
        output_paths.append(actual)
    if len(set(output_paths)) != len(output_paths):
        raise PreregError("GVHMR output paths alias")

    downstream = prereg.get("downstream_consumer_closure")
    if not isinstance(downstream, dict):
        raise PreregError("downstream consumer closure must be an object")
    if downstream.get("authorized_consumers_in_this_prereg") != [
        "scripts/audit_gvhmr_result.py"
    ]:
        raise PreregError("only the structural auditor may consume outputs in this prereg")
    if downstream.get("post_gvhmr_consumer") != "none":
        raise PreregError("post-GVHMR consumer must remain none")
    if downstream.get("implicit_consumption_forbidden") is not True:
        raise PreregError("implicit downstream consumption must stay forbidden")

    no_clobber = prereg.get("no_clobber_contract")
    if not isinstance(no_clobber, dict) or any(
        no_clobber.get(field) is not True
        for field in (
            "execution_record_must_not_exist",
            "state_root_must_not_exist",
            "every_output_namespace_must_not_exist",
        )
    ):
        raise PreregError("no-clobber contract was weakened")
    if no_clobber.get("subset_or_reordered_execution_allowed") is not False:
        raise PreregError("subset/reordered execution must stay forbidden")

    independence = prereg.get("independence_contract")
    if not isinstance(independence, dict):
        raise PreregError("batch independence contract must be an object")
    if (
        independence.get("peer_prereg_path") != f"configs/{batch_spec['peer_filename']}"
        or independence.get("peer_prereg_id") != batch_spec["peer_id"]
    ):
        raise PreregError("batch peer identity changed")
    for field in (
        "record_state_and_output_namespaces_must_be_disjoint",
        "same_or_different_pod_allowed_after_each_exact_attestation",
    ):
        if independence.get(field) is not True:
            raise PreregError(f"batch independence field {field} must remain true")
    if independence.get("peer_failure_blocks_this_batch") is not False:
        raise PreregError("one exact batch failure must not block its peer")

    peer_path = resolve_repo_file(
        repo_root, independence["peer_prereg_path"], "independence peer prereg"
    )
    peer = read_json(peer_path, "independence peer prereg")
    if peer.get("prereg_id") != batch_spec["peer_id"]:
        raise PreregError("independence peer prereg id mismatch")
    peer_spec = ALLOWED_PREREGISTRATIONS[batch_spec["peer_id"]]
    if sha256_file(peer_path) != peer_spec["sha256"]:
        raise PreregError("independence peer is not the exact committed preregistration")
    peer_pod = peer.get("pod_contract")
    peer_output = peer.get("output_contract")
    if not isinstance(peer_pod, dict) or not isinstance(peer_output, dict):
        raise PreregError("independence peer contracts are malformed")
    own_paths = {pod["execution_record_path"], pod["state_root"]}
    peer_paths = {peer_pod.get("execution_record_path"), peer_pod.get("state_root")}
    if own_paths & peer_paths:
        raise PreregError("S0/M0 execution record or state roots collide")
    own_outputs = {str(row["path"]) for row in rows}
    peer_outputs = {str(row.get("path")) for row in peer_output.get("outputs", [])}
    if own_outputs & peer_outputs:
        raise PreregError("S0/M0 output namespaces collide")

    prereg_sha256 = sha256_file(prereg_path)
    if prereg_sha256 != batch_spec["sha256"]:
        raise PreregError("preregistration is not the exact committed S0/M0 file")

    return {
        "prereg": prereg,
        "prereg_path": prereg_path,
        "prereg_sha256": prereg_sha256,
        "intake": intake,
        "intake_path": intake_path,
        "review_path": review_path,
        "franco_review_path": franco_review_path,
        "tool_paths": tool_paths,
        "validator_path": validator_path,
    }


def git_clean_head(path: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        raise PreregError(f"cannot resolve GVHMR HEAD: {head.stderr.strip()}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise PreregError(f"cannot inspect GVHMR worktree: {status.stderr.strip()}")
    if status.stdout.strip():
        raise PreregError("GVHMR worktree is not clean")
    return head.stdout.strip()


def ensure_no_symlink_components(path: Path, label: str, *, require_leaf: bool = True) -> None:
    """Reject symlinks in every existing component of an absolute path."""

    if not path.is_absolute():
        raise PreregError(f"{label} must be absolute: {path}")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if require_leaf or index < len(parts) - 1:
                raise PreregError(f"{label} component is missing: {current}") from None
            return
        if stat.S_ISLNK(info.st_mode):
            raise PreregError(f"{label} contains a symlink component: {current}")


def regular_file_fingerprint(path: Path, label: str) -> dict[str, Any]:
    ensure_no_symlink_components(path, label)
    try:
        info = path.stat()
    except OSError as exc:
        raise PreregError(f"cannot stat {label} {path}: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        raise PreregError(f"{label} must be a regular file: {path}")
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "bytes": info.st_size,
        "sha256": sha256_file(path),
    }


def host_identity() -> dict[str, str]:
    def optional_file_sha(path: Path) -> str:
        if not path.is_file():
            return "unavailable"
        return sha256_file(path)

    return {
        "hostname": socket.gethostname(),
        "boot_id_sha256": optional_file_sha(Path("/proc/sys/kernel/random/boot_id")),
        "machine_id_sha256": optional_file_sha(Path("/etc/machine-id")),
        "semantics": "live execution identity; pod_id remains a caller routing assertion",
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def tree_fingerprint(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise PreregError(f"checkpoint tree is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise PreregError(f"checkpoint tree is empty: {root}")
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def python_fingerprint(python: Path) -> dict[str, str]:
    version = subprocess.run(
        [str(python), "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode != 0:
        raise PreregError(f"motion Python --version failed: {version.stderr.strip()}")
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze.returncode != 0:
        raise PreregError(f"motion Python pip freeze failed: {freeze.stderr.strip()}")
    normalized = "\n".join(
        sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip())
    )
    return {
        "version": (version.stdout or version.stderr).strip(),
        "pip_freeze_sha256": hashlib.sha256((normalized + "\n").encode()).hexdigest(),
    }


def execution_intake(static: dict[str, Any]) -> dict[str, Any]:
    """Return a batch-only audit view without weakening the committed full intake."""

    prereg = static["prereg"]
    all_assets = {str(row["id"]): row for row in static["intake"]["assets"]}
    batch_ids = prereg["execution_batch"]["asset_ids"]
    batch_intake = dict(static["intake"])
    batch_intake["processing_order"] = list(batch_ids)
    batch_intake["assets"] = [all_assets[asset_id] for asset_id in batch_ids]
    return batch_intake


def verify_live_contract(
    static: dict[str, Any],
    *,
    source_root: Path,
    gvhmr_root: Path,
    python: Path,
    state_root: Path,
    gpu: int,
    max_used_mib: int,
    require_unused_state: bool = True,
    require_unused_outputs: bool = True,
) -> dict[str, Any]:
    prereg = static["prereg"]
    pod = prereg["pod_contract"]
    runtime = prereg["gvhmr_runtime"]
    output = prereg["output_contract"]
    source_root = Path(os.path.abspath(source_root))
    gvhmr_root = Path(os.path.abspath(gvhmr_root))
    python = Path(os.path.abspath(python))
    state_root = Path(os.path.abspath(state_root))
    if source_root != Path(pod["staged_source_root"]):
        raise PreregError("staged source root does not match preregistration")
    if gvhmr_root != Path(runtime["root"]):
        raise PreregError("GVHMR root does not match preregistration")
    if python != Path(runtime["python"]["executable"]):
        raise PreregError("motion Python does not match preregistration")
    if state_root != Path(pod["state_root"]):
        raise PreregError("state root does not match preregistration")
    if gpu not in runtime["gpu"]["allowed_physical_indices"]:
        raise PreregError("GPU index is outside the preregistered set")
    if max_used_mib != runtime["gpu"]["max_used_mib_before_each_asset"]:
        raise PreregError("GPU memory gate does not match preregistration")

    ensure_no_symlink_components(source_root, "staged source root")
    ensure_no_symlink_components(gvhmr_root, "GVHMR root")
    ensure_no_symlink_components(python, "motion Python")
    ensure_no_symlink_components(
        state_root, "state root", require_leaf=not require_unused_state
    )
    ensure_no_symlink_components(Path(output["root"]), "GVHMR output root")

    all_assets = {str(row["id"]): row for row in static["intake"]["assets"]}
    batch_ids = prereg["execution_batch"]["asset_ids"]
    batch_intake = execution_intake(static)
    failures = audit_assets(batch_intake, source_root)
    if failures:
        raise PreregError("staged intake byte/media audit failed: " + "; ".join(failures))
    head = git_clean_head(gvhmr_root)
    if head != runtime["commit"]:
        raise PreregError(f"GVHMR commit mismatch: expected={runtime['commit']} actual={head}")
    entrypoint = gvhmr_root / runtime["entrypoint_relpath"]
    if not entrypoint.is_file():
        raise PreregError(f"GVHMR entrypoint is missing: {entrypoint}")
    checkpoint = tree_fingerprint(gvhmr_root / runtime["checkpoint_tree"]["relpath"])
    expected_checkpoint = {
        field: runtime["checkpoint_tree"][field] for field in ("files", "bytes", "sha256")
    }
    if checkpoint != expected_checkpoint:
        raise PreregError(
            f"GVHMR checkpoint tree mismatch: expected={expected_checkpoint} actual={checkpoint}"
        )
    if not python.is_file():
        raise PreregError(f"motion Python is missing: {python}")
    python_env = python_fingerprint(python)
    expected_python = {
        field: runtime["python"][field] for field in ("version", "pip_freeze_sha256")
    }
    if python_env != expected_python:
        raise PreregError(
            f"motion Python fingerprint mismatch: expected={expected_python} actual={python_env}"
        )
    if require_unused_state and state_root.exists():
        raise PreregError(f"no-clobber state root already exists: {state_root}")
    if not require_unused_state and not state_root.is_dir():
        raise PreregError(f"expected live state root is missing: {state_root}")
    if require_unused_outputs:
        for row in output["outputs"]:
            namespace = Path(row["path"]).parent
            ensure_no_symlink_components(namespace, "GVHMR output namespace", require_leaf=False)
            if namespace.exists():
                raise PreregError(f"no-clobber GVHMR output namespace already exists: {namespace}")

    nvidia_path = Path(runtime["gpu"]["nvidia_smi"]["absolute_path"])
    nvidia_binding = regular_file_fingerprint(nvidia_path, "nvidia-smi")

    inputs: list[dict[str, Any]] = []
    outputs = {str(row["asset_id"]): row for row in output["outputs"]}
    for asset_id in batch_ids:
        asset = all_assets[asset_id]
        ensure_no_symlink_components(
            source_root / asset["source_relpath"], f"staged source path for {asset_id}"
        )
        source = resolve_asset_path(source_root, asset["source_relpath"])
        output_binding = outputs.get(asset_id)
        inputs.append(
            {
                "asset_id": asset_id,
                "source_path": str(source),
                "source_sha256": asset["sha256"],
                "source_bytes": asset["bytes"],
                "media": asset["media"],
                "output_path": output_binding["path"] if output_binding is not None else None,
                "source_file": regular_file_fingerprint(source, f"staged source {asset_id}"),
            }
        )
    return {
        "gvhmr_head": head,
        "gvhmr_entrypoint_sha256": sha256_file(entrypoint),
        "checkpoint_tree": checkpoint,
        "python_environment": python_env,
        "nvidia_smi": nvidia_binding,
        "inputs": inputs,
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_components(path.parent, "execution-record parent")
    ensure_no_symlink_components(path, "execution record", require_leaf=False)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise PreregError(f"no-clobber execution record already exists: {path}") from None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def attest(
    prereg_path: Path,
    *,
    pod_id: str,
    source_root: Path,
    gvhmr_root: Path,
    python: Path,
    record_path: Path,
    gpu: int,
    max_used_mib: int,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    static = validate_static_prereg(prereg_path, repo_root=repo_root)
    prereg = static["prereg"]
    if pod_id not in prereg["pod_contract"]["allowed_pod_ids"]:
        raise PreregError(f"pod id is not allowed: {pod_id}")
    record_path = Path(os.path.abspath(record_path))
    if record_path != Path(prereg["pod_contract"]["execution_record_path"]):
        raise PreregError("execution record path does not match preregistration")
    if record_path.exists():
        raise PreregError(f"no-clobber execution record already exists: {record_path}")
    live = verify_live_contract(
        static,
        source_root=source_root,
        gvhmr_root=gvhmr_root,
        python=python,
        state_root=Path(prereg["pod_contract"]["state_root"]),
        gpu=gpu,
        max_used_mib=max_used_mib,
    )
    queue_path = static["tool_paths"]["queue"]
    host_python = regular_file_fingerprint(Path(sys.executable).resolve(), "host Python")
    record = {
        "schema_version": 1,
        "status": "ready_for_exact_offline_gvhmr",
        "authorization_scope": "offline_gvhmr_video_to_smplx_and_structural_audit_only",
        "pod_id": pod_id,
        "pod_id_semantics": prereg["pod_contract"]["pod_id_semantics"],
        "host_identity": host_identity(),
        "created_utc": utc_now(),
        "prereg_path": str(static["prereg_path"]),
        "prereg_sha256": static["prereg_sha256"],
        "manifest_path": str(static["intake_path"]),
        "manifest_sha256": prereg["intake"]["sha256"],
        "manual_event_review_path": str(static["review_path"]),
        "manual_event_review_sha256": prereg["manual_event_review"]["sha256"],
        "franco_priority_review_path": str(static["franco_review_path"]),
        "franco_priority_review_sha256": prereg["franco_priority_context"]["sha256"],
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "host_python": host_python,
        "queue_path": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "result_auditor_sha256": sha256_file(static["tool_paths"]["result_auditor"]),
        "source_root": str(source_root.resolve()),
        "gvhmr_root": str(gvhmr_root.resolve()),
        "python": str(python.resolve()),
        "state_root": prereg["pod_contract"]["state_root"],
        "gpu_physical_index": gpu,
        "max_used_mib": max_used_mib,
        "poll_seconds": prereg["gvhmr_runtime"]["gpu"]["poll_seconds"],
        "wait_timeout_seconds": prereg["gvhmr_runtime"]["gpu"]["wait_timeout_seconds"],
        "nvidia_smi": live["nvidia_smi"],
        "static_camera": True,
        "processing_order": prereg["execution_batch"]["asset_ids"],
        "staged_content_media_audit": {
            "status": "pass",
            "registered_asset_count": len(prereg["processing_order"]),
            "asset_count": len(prereg["execution_batch"]["asset_ids"]),
            "execution_asset_count": len(prereg["execution_batch"]["asset_ids"]),
        },
        "live_dependency_binding": live,
        "downstream_consumer": "none_beyond_structural_auditor",
    }
    record["launch_argv"] = [
        host_python["realpath"],
        str(queue_path),
        "--prereg",
        str(static["prereg_path"]),
        "--execution-record",
        str(record_path),
    ]
    write_json_exclusive(record_path, record)
    return record


def validate_execution_record_for_launch(
    prereg_path: Path,
    record_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    static = validate_static_prereg(prereg_path, repo_root=repo_root)
    prereg = static["prereg"]
    record_path = Path(os.path.abspath(record_path))
    if record_path != Path(prereg["pod_contract"]["execution_record_path"]):
        raise PreregError("execution record path does not match preregistration")
    ensure_no_symlink_components(record_path, "execution record")
    record = read_json(record_path, "GVHMR execution record")
    if record.get("status") != "ready_for_exact_offline_gvhmr":
        raise PreregError("execution record is not ready")
    if record.get("authorization_scope") != "offline_gvhmr_video_to_smplx_and_structural_audit_only":
        raise PreregError("execution record authorization scope changed")
    if record.get("prereg_sha256") != static["prereg_sha256"]:
        raise PreregError("execution record prereg SHA mismatch")
    if record.get("manifest_sha256") != prereg["intake"]["sha256"]:
        raise PreregError("execution record intake SHA mismatch")
    if record.get("manual_event_review_sha256") != prereg["manual_event_review"]["sha256"]:
        raise PreregError("execution record manual-review SHA mismatch")
    if record.get("franco_priority_review_sha256") != prereg["franco_priority_context"]["sha256"]:
        raise PreregError("execution record Franco-priority-review SHA mismatch")
    if record.get("validator_path") != str(Path(__file__).resolve()):
        raise PreregError("execution record validator path mismatch")
    if record.get("validator_sha256") != sha256_file(Path(__file__).resolve()):
        raise PreregError("execution record validator SHA mismatch")
    if record.get("queue_sha256") != sha256_file(static["tool_paths"]["queue"]):
        raise PreregError("execution record queue SHA mismatch")
    if record.get("result_auditor_sha256") != sha256_file(static["tool_paths"]["result_auditor"]):
        raise PreregError("execution record result-auditor SHA mismatch")
    if record.get("processing_order") != prereg["execution_batch"]["asset_ids"]:
        raise PreregError("execution record processing_order mismatch")
    current_host_python = regular_file_fingerprint(Path(sys.executable).resolve(), "host Python")
    if record.get("host_python") != current_host_python:
        raise PreregError("execution record host Python fingerprint mismatch")
    expected_argv = [
        current_host_python["realpath"],
        str(static["tool_paths"]["queue"]),
        "--prereg",
        str(static["prereg_path"]),
        "--execution-record",
        str(record_path),
    ]
    if record.get("launch_argv") != expected_argv:
        raise PreregError("execution record launch_argv mismatch")
    if record.get("host_identity") != host_identity():
        raise PreregError("execution record host identity changed")
    expected_arguments = {
        "manifest_path": str(static["intake_path"]),
        "manual_event_review_path": str(static["review_path"]),
        "franco_priority_review_path": str(static["franco_review_path"]),
        "source_root": prereg["pod_contract"]["staged_source_root"],
        "gvhmr_root": prereg["gvhmr_runtime"]["root"],
        "python": prereg["gvhmr_runtime"]["python"]["executable"],
        "state_root": prereg["pod_contract"]["state_root"],
        "max_used_mib": prereg["gvhmr_runtime"]["gpu"]["max_used_mib_before_each_asset"],
        "poll_seconds": prereg["gvhmr_runtime"]["gpu"]["poll_seconds"],
        "wait_timeout_seconds": prereg["gvhmr_runtime"]["gpu"]["wait_timeout_seconds"],
        "static_camera": True,
        "pod_id_semantics": prereg["pod_contract"]["pod_id_semantics"],
    }
    for field, expected in expected_arguments.items():
        if record.get(field) != expected:
            raise PreregError(
                f"execution record {field} mismatch: expected={expected!r} actual={record.get(field)!r}"
            )
    if record.get("pod_id") not in prereg["pod_contract"]["allowed_pod_ids"]:
        raise PreregError("execution record pod id is not allowed")
    gpu = record.get("gpu_physical_index")
    if isinstance(gpu, bool) or not isinstance(gpu, int):
        raise PreregError("execution record GPU index must be an integer")
    if gpu not in prereg["gvhmr_runtime"]["gpu"]["allowed_physical_indices"]:
        raise PreregError("execution record GPU index is outside the preregistered set")
    if record.get("downstream_consumer") != "none_beyond_structural_auditor":
        raise PreregError("execution record downstream closure changed")
    live = verify_live_contract(
        static,
        source_root=Path(record["source_root"]),
        gvhmr_root=Path(record["gvhmr_root"]),
        python=Path(record["python"]),
        state_root=Path(record["state_root"]),
        gpu=gpu,
        max_used_mib=record["max_used_mib"],
    )
    if live != record.get("live_dependency_binding"):
        raise PreregError("live dependency/input binding changed after attestation")
    if record.get("nvidia_smi") != live["nvidia_smi"]:
        raise PreregError("nvidia-smi path/SHA binding changed after attestation")
    return {
        "static": static,
        "record": record,
        "live": live,
        "prereg_sha256": static["prereg_sha256"],
        "execution_record_sha256": sha256_file(record_path),
        "manual_event_review_sha256": prereg["manual_event_review"]["sha256"],
        "franco_priority_review_sha256": prereg["franco_priority_context"]["sha256"],
        "authorization_scope": record["authorization_scope"],
    }


def validate_runtime_after_execution(binding: dict[str, Any]) -> None:
    """Re-hash all runtime/input dependencies after the queue finishes."""

    static = binding["static"]
    record = binding["record"]
    live = verify_live_contract(
        static,
        source_root=Path(record["source_root"]),
        gvhmr_root=Path(record["gvhmr_root"]),
        python=Path(record["python"]),
        state_root=Path(record["state_root"]),
        gpu=record["gpu_physical_index"],
        max_used_mib=record["max_used_mib"],
        require_unused_state=False,
        require_unused_outputs=False,
    )
    if live != record.get("live_dependency_binding"):
        raise PreregError("runtime/input binding changed during queue execution")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static", help="validate committed preregistration closure")
    static.add_argument("--prereg", type=Path, required=True)

    attest_parser = subparsers.add_parser(
        "attest", help="repeat live Pod audits and create one no-clobber execution record"
    )
    attest_parser.add_argument("--prereg", type=Path, required=True)
    attest_parser.add_argument("--pod-id", required=True)
    attest_parser.add_argument("--source-root", type=Path, required=True)
    attest_parser.add_argument("--gvhmr-root", type=Path, required=True)
    attest_parser.add_argument("--python", type=Path, required=True)
    attest_parser.add_argument("--record", type=Path, required=True)
    attest_parser.add_argument("--gpu", type=int, required=True)
    attest_parser.add_argument("--max-used-mib", type=int, default=19000)

    inspect = subparsers.add_parser(
        "inspect", help="live-revalidate an existing execution record without writing"
    )
    inspect.add_argument("--prereg", type=Path, required=True)
    inspect.add_argument("--record", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "static":
            static = validate_static_prereg(args.prereg)
            execution_batch = static["prereg"]["execution_batch"]
            print(
                "[motion-gvhmr-prereg] PASS static "
                f"prereg_sha256={static['prereg_sha256']} "
                f"batch_id={execution_batch['batch_id']} "
                f"batch_assets={len(execution_batch['asset_ids'])}"
            )
            return 0
        if args.command == "attest":
            record = attest(
                args.prereg,
                pod_id=args.pod_id,
                source_root=args.source_root,
                gvhmr_root=args.gvhmr_root,
                python=args.python,
                record_path=args.record,
                gpu=args.gpu,
                max_used_mib=args.max_used_mib,
            )
            print(
                "[motion-gvhmr-prereg] PASS attested "
                f"pod={record['pod_id']} gpu={record['gpu_physical_index']} record={args.record}"
            )
            return 0
        validate_execution_record_for_launch(
            args.prereg,
            args.record,
        )
        print(f"[motion-gvhmr-prereg] PASS live record={args.record}")
        return 0
    except (PreregError, OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        print(f"[motion-gvhmr-prereg] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
