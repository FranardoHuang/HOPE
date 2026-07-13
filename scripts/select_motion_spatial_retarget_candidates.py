#!/usr/bin/env python3
"""Select one deterministic B/C spatial-retarget candidate per motion.

This is the narrow, CPU-only Step-A consumer between the signed proposal
screen and candidate materialization.  It validates one exact proposal
artifact, removes only byte-semantically identical yaw-zero R0/R1 aliases,
sorts the remaining candidates by the frozen scientific key, and publishes a
no-clobber selection ledger.  It does not load a motion, run GMR, create a
schema-2 artifact, start a simulator, train a policy, or authorize hardware.

The ``resolve`` mode makes the fallback rule executable.  Only a table/net
external-geometry clearance failure may advance to the next frozen candidate.
Schema-2, L0, vendor-L1, and internal dynamics/balance failures stop that
asset.  The resolution ledger is diagnostic and never grants promotion.
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
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PLAN_STATUS = "frozen_post_screen_selection_contract"
INPUT_STATUS = "complete_proposals_only_promotion_blocked"
RESULT_STATUS = "complete_primary_pair_selected_certification_blocked"
TRANSFORM_SEMANTICS = "atomic_whole_motion_ground_preserving_SE2"
TARGET_ASSETS = ("franco_backhand_loop_b", "franco_backhand_loop_c")
EXPECTED_ASSET_SIDES = {
    "franco_forehand_block": "forehand",
    "franco_backhand_block": "backhand",
    "franco_forehand_loop": "forehand",
    "franco_backhand_loop_a": "backhand",
    "franco_backhand_loop_b": "backhand",
    "franco_backhand_loop_c": "backhand",
    "v6_forehand_block": "forehand",
    "v6_backhand_block": "backhand",
    "v7_forehand_block": "forehand",
    "v7_backhand_block": "backhand",
}
TARGET_WINDOWS = {
    "franco_backhand_loop_b": (45, 53),
    "franco_backhand_loop_c": (46, 54),
}
EXPECTED_RAW_COUNTS = {
    "franco_backhand_loop_b": 19,
    "franco_backhand_loop_c": 3,
}
EXPECTED_UNIQUE_COUNTS = {
    "franco_backhand_loop_b": 16,
    "franco_backhand_loop_c": 3,
}
SORT_FIELDS = (
    "translation_norm_m",
    "abs_yaw_deg",
    "negative_return_margin_m",
    "negative_source_dense_racket_body_clearance_m",
    "frame",
    "candidate_id",
)
ADVANCE_CODE = "external_geometry_table_or_net_clearance_failure"
STOP_CODES = (
    "schema2_materialization_failure",
    "l0_static_audit_failure",
    "l1_vendor_self_collision_failure",
    "internal_dynamics_or_balance_failure",
)

TOP_LEVEL_KEYS = {
    "accepted_candidates",
    "asset_count",
    "capture_table_pose_observed",
    "cells",
    "certified_candidate_count",
    "claims",
    "contact_phase_truth",
    "formal_eligible",
    "generated_utc",
    "manifest",
    "predecessor",
    "proposal_count_before_certificate",
    "question_count",
    "question_semantic_sha256",
    "robot_approved",
    "schema_version",
    "status",
}
CELL_KEYS = {
    "accepted_candidate_ids",
    "asset_id",
    "proposal_count_before_certificate",
    "question_id",
    "reported_proposals",
    "side",
    "tier",
}
PROPOSAL_KEYS = {
    "asset_id",
    "candidate_id",
    "capture_extrinsic_claim",
    "frame",
    "landing_xy_m",
    "mapped_racket_pos_w_m",
    "net_center_z_m",
    "phase",
    "predecessor_full_result_sha256",
    "promotion_status",
    "question_id",
    "return_margin_m",
    "side",
    "source_dense_racket_body_clearance_m",
    "source_motion_sha256",
    "tier",
    "transform_semantics",
    "translation_norm_m",
    "translation_w_m",
    "vertical_position_error_m",
    "yaw_deg",
}


class SelectionError(ValueError):
    """The frozen selection contract cannot be satisfied."""


def _reject_constant(token: str) -> None:
    raise SelectionError(f"non-finite JSON constant is forbidden: {token}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SelectionError(f"{label} is not UTF-8: {exc}") from None
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except SelectionError:
        raise
    except json.JSONDecodeError as exc:
        raise SelectionError(f"{label} is not strict JSON: {exc}") from None


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"payload is not finite canonical JSON: {exc}") from None


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise SelectionError(f"{label} must be a lowercase SHA-256")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SelectionError(f"{label} keys {actual} != {sorted(expected)}")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionError(f"{label} must be finite")
    return result


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SelectionError(f"{label} must be a positive integer")
    return value


def require_repo_binding(value: Any, label: str) -> Path:
    row = exact_keys(value, {"path", "sha256"}, label)
    text = row["path"]
    if not isinstance(text, str) or not text or Path(text).is_absolute() or ".." in Path(text).parts:
        raise SelectionError(f"{label}.path must be a safe repository-relative path")
    path = (REPO_ROOT / text).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise SelectionError(f"{label}.path escapes repository") from None
    if not path.is_file():
        raise SelectionError(f"{label}.path is missing: {path}")
    expected = require_sha(row["sha256"], f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise SelectionError(f"{label} sha256 {actual} != {expected}")
    return path


def _ensure_no_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and current == absolute:
            return
        try:
            info = current.lstat()
        except FileNotFoundError:
            raise SelectionError(f"{label} path component is missing: {current}") from None
        if stat.S_ISLNK(info.st_mode):
            raise SelectionError(f"{label} path contains symlink: {current}")


def read_stable_regular_file(path: Path, label: str) -> bytes:
    if not path.is_absolute():
        raise SelectionError(f"{label} path must be absolute")
    _ensure_no_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SelectionError(f"cannot open {label}: {exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SelectionError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after:
            raise SelectionError(f"{label} changed while being read")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise SelectionError(f"{label} short read {len(raw)} != {before.st_size}")
        return raw
    finally:
        os.close(fd)


def publish_no_clobber(path: Path, payload: Any) -> tuple[int, str]:
    if not path.is_absolute():
        raise SelectionError("output path must be absolute")
    if not path.parent.is_dir():
        raise SelectionError(f"output parent must already exist: {path.parent}")
    _ensure_no_symlink_components(path.parent, "output parent")
    raw = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise SelectionError(f"output already exists: {path}") from None
    except OSError as exc:
        raise SelectionError(f"cannot claim output: {exc}") from None
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    return len(raw), sha256_bytes(raw)


def validate_plan(path: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    expected_sha = require_sha(expected_sha, "--expected-prereg-sha256")
    raw = path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_sha:
        raise SelectionError(f"prereg sha256 {actual_sha} != {expected_sha}")
    plan = read_json_bytes(raw, "selection preregistration")
    exact_keys(
        plan,
        {
            "schema_version",
            "plan_id",
            "status",
            "human_owner",
            "executor",
            "scope",
            "formal_eligible",
            "training_authorized",
            "topp_authorized",
            "hardware_authorized",
            "consumer",
            "proposal_input",
            "tracked_screen_summary",
            "target_assets",
            "deduplication",
            "ranking",
            "fallback_policy",
            "output_contract",
            "not_claimed",
        },
        "plan",
    )
    if plan["schema_version"] != 1 or plan["status"] != PLAN_STATUS:
        raise SelectionError("plan schema/status mismatch")
    if plan["plan_id"] != "motion-backhand-loop-bc-proposal-selection-20260713-v1":
        raise SelectionError("plan_id changed")
    if plan["human_owner"] != "Franco" or plan["executor"] != "Codex":
        raise SelectionError("human owner/executor changed")
    if plan["scope"] != (
        "CPU-only deterministic selection of exactly one primary and one frozen fallback ladder "
        "for each Franco backhand-loop B/C signed whole-motion proposal set; no GMR, schema-2 "
        "materialization, simulator, training, TOPP, Gate3 or hardware"
    ):
        raise SelectionError("scope changed")
    for field in ("formal_eligible", "training_authorized", "topp_authorized", "hardware_authorized"):
        if plan[field] is not False:
            raise SelectionError(f"{field} must remain false")
    consumer = require_repo_binding(plan["consumer"], "consumer")
    if consumer != Path(__file__).resolve():
        raise SelectionError("consumer path must name this script")

    proposal = exact_keys(
        plan["proposal_input"],
        {"bytes", "sha256", "accepted_producer_path"},
        "proposal_input",
    )
    if positive_int(proposal["bytes"], "proposal_input.bytes") != 225920:
        raise SelectionError("proposal input byte count changed")
    if require_sha(proposal["sha256"], "proposal_input.sha256") != (
        "69c3db16fa78f526aef49f20eeafe0d7e5e3004c4ed27f5e2823bb3574e2465c"
    ):
        raise SelectionError("proposal input SHA changed")
    if proposal["accepted_producer_path"] != (
        "/workspace/codexschema/motion_spatial_retarget_signed_a4bbbaa_v1/proposals.json"
    ):
        raise SelectionError("accepted producer path changed")
    summary = require_repo_binding(plan["tracked_screen_summary"], "tracked_screen_summary")
    if summary != (
        REPO_ROOT / "configs/motion_video_spatial_retarget_signed_results_20260713.json"
    ).resolve():
        raise SelectionError("tracked screen summary path changed")

    targets = plan["target_assets"]
    if not isinstance(targets, list) or len(targets) != 2:
        raise SelectionError("target_assets must contain exactly B and C")
    if [row.get("asset_id") if isinstance(row, dict) else None for row in targets] != list(TARGET_ASSETS):
        raise SelectionError("target asset order/set changed")
    for row in targets:
        exact_keys(row, {"asset_id", "human_name", "inclusive_frame_window", "expected_raw_proposals", "expected_unique_after_dedup"}, f"target {row.get('asset_id')}")
        asset_id = row["asset_id"]
        if row["inclusive_frame_window"] != list(TARGET_WINDOWS[asset_id]):
            raise SelectionError(f"{asset_id} frame window changed")
        if row["expected_raw_proposals"] != EXPECTED_RAW_COUNTS[asset_id]:
            raise SelectionError(f"{asset_id} raw count changed")
        if row["expected_unique_after_dedup"] != EXPECTED_UNIQUE_COUNTS[asset_id]:
            raise SelectionError(f"{asset_id} unique count changed")
        if not isinstance(row["human_name"], str) or not row["human_name"]:
            raise SelectionError(f"{asset_id} human_name missing")

    if plan["deduplication"] != {
        "scope": "yaw_deg==0 only",
        "identity": "all proposal fields except candidate_id and tier must be identical",
        "allowed_tier_pair": ["R0", "R1"],
        "survivor": "R0",
        "any_other_duplicate": "fail_closed",
    }:
        raise SelectionError("deduplication contract changed")
    if plan["ranking"] != {
        "ascending_lexicographic_fields": list(SORT_FIELDS),
        "primary": "rank_0_exactly_one_per_asset",
        "remaining": "immutable_fallback_ladder",
        "numeric_rounding_before_sort": None,
    }:
        raise SelectionError("ranking contract changed")
    if plan["fallback_policy"] != {
        "advance_only_on": [ADVANCE_CODE],
        "stop_asset_no_fallback_on": list(STOP_CODES),
        "unknown_outcome": "fail_closed",
        "success": "continue_same_candidate_to_next_certificate_stage",
        "cross_asset_substitution": False,
    }:
        raise SelectionError("fallback policy changed")
    if plan["output_contract"] != {
        "selection_status": RESULT_STATUS,
        "output_file_must_not_exist": True,
        "no_clobber": True,
        "primary_count": 2,
        "materialization_authorized": False,
        "promotion_authorized": False,
    }:
        raise SelectionError("output contract changed")
    if plan["not_claimed"] != [
        "observed physical ball contact or true contact frame",
        "schema-2 materialization, L0, vendor L1, table/net clearance or dynamics pass",
        "action-family effectiveness, training eligibility, Gate3 behavior or hardware safety",
    ]:
        raise SelectionError("not_claimed contract changed")
    return plan, actual_sha


def _validate_vector(value: Any, size: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != size:
        raise SelectionError(f"{label} must have {size} elements")
    for index, item in enumerate(value):
        finite_number(item, f"{label}[{index}]")


def _validate_proposal(row: Any, cell: Mapping[str, Any], index: int) -> dict[str, Any]:
    label = f"{cell['asset_id']}/{cell['question_id']}/{cell['tier']} proposal[{index}]"
    proposal = dict(exact_keys(row, PROPOSAL_KEYS, label))
    for field in ("asset_id", "question_id", "side", "tier"):
        if proposal[field] != cell[field]:
            raise SelectionError(f"{label}.{field} disagrees with its cell")
    require_sha(proposal["candidate_id"], f"{label}.candidate_id")
    require_sha(proposal["predecessor_full_result_sha256"], f"{label}.predecessor_full_result_sha256")
    require_sha(proposal["source_motion_sha256"], f"{label}.source_motion_sha256")
    if proposal["capture_extrinsic_claim"] is not False:
        raise SelectionError(f"{label} claims a capture extrinsic")
    if proposal["promotion_status"] != "missing_candidate_certificate":
        raise SelectionError(f"{label} promotion status changed")
    if proposal["transform_semantics"] != TRANSFORM_SEMANTICS:
        raise SelectionError(f"{label} transform semantics changed")
    if isinstance(proposal["frame"], bool) or not isinstance(proposal["frame"], int) or proposal["frame"] < 0:
        raise SelectionError(f"{label}.frame must be a non-negative integer")
    _validate_vector(proposal["landing_xy_m"], 2, f"{label}.landing_xy_m")
    _validate_vector(proposal["mapped_racket_pos_w_m"], 3, f"{label}.mapped_racket_pos_w_m")
    _validate_vector(proposal["translation_w_m"], 3, f"{label}.translation_w_m")
    if finite_number(proposal["translation_w_m"][2], f"{label}.translation_w_m[2]") != 0.0:
        raise SelectionError(f"{label} changes vertical translation")
    for field in (
        "net_center_z_m",
        "phase",
        "return_margin_m",
        "source_dense_racket_body_clearance_m",
        "translation_norm_m",
        "vertical_position_error_m",
        "yaw_deg",
    ):
        finite_number(proposal[field], f"{label}.{field}")
    if proposal["translation_norm_m"] < 0.0 or proposal["return_margin_m"] < 0.0:
        raise SelectionError(f"{label} has negative distance/margin")
    if proposal["side"] != "backhand" or proposal["tier"] not in {"R0", "R1"}:
        raise SelectionError(f"{label} has unsupported side/tier")
    return proposal


def validate_proposal_artifact(plan: Mapping[str, Any], raw: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = plan["proposal_input"]
    if len(raw) != binding["bytes"]:
        raise SelectionError(f"proposal bytes {len(raw)} != {binding['bytes']}")
    actual_sha = sha256_bytes(raw)
    if actual_sha != binding["sha256"]:
        raise SelectionError(f"proposal sha256 {actual_sha} != {binding['sha256']}")
    data = read_json_bytes(raw, "signed proposal artifact")
    exact_keys(data, TOP_LEVEL_KEYS, "signed proposal artifact")
    if data["schema_version"] != 1 or data["status"] != INPUT_STATUS:
        raise SelectionError("proposal schema/status mismatch")
    if data["asset_count"] != 10 or data["question_count"] != 64 or len(data["cells"]) != 640:
        raise SelectionError("proposal grid must remain 10 assets x 64 questions")
    if data["proposal_count_before_certificate"] != 22:
        raise SelectionError("proposal count must remain 22")
    if data["accepted_candidates"] != [] or data["certified_candidate_count"] != 0:
        raise SelectionError("input unexpectedly claims accepted/certified candidates")
    if data["capture_table_pose_observed"] is not False or data["formal_eligible"] is not False or data["robot_approved"] is not False:
        raise SelectionError("input authorization/pose claims changed")
    if data["contact_phase_truth"] is not None:
        raise SelectionError("air-swing contact_phase_truth must remain null")
    require_sha(data["question_semantic_sha256"], "question_semantic_sha256")
    manifest = exact_keys(data["manifest"], {"path", "bytes", "sha256"}, "manifest")
    if manifest["sha256"] != "0f757c8c4abfc9bf5070b7db79f494fa1d97a45ddb222609898662eff63af66a":
        raise SelectionError("upstream signed-screen manifest changed")
    predecessor = exact_keys(data["predecessor"], {"path", "bytes", "sha256"}, "predecessor")
    if predecessor["sha256"] != "c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53":
        raise SelectionError("upstream v5 predecessor changed")
    claims = exact_keys(
        data["claims"],
        {"capture_extrinsic", "final_arbiter", "real_robot_authorized", "rl_authorized", "schema2_L0_L1_table_net", "spatial_retarget", "topp_authorized"},
        "claims",
    )
    if claims != {
        "capture_extrinsic": None,
        "final_arbiter": "agibot_vendor_mujoco_gate3_gate3b_no_reset",
        "real_robot_authorized": False,
        "rl_authorized": False,
        "schema2_L0_L1_table_net": "missing_or_no_matching_pass_certificates",
        "spatial_retarget": "bounded atomic whole-motion SE2 proposal screen",
        "topp_authorized": False,
    }:
        raise SelectionError("input claims changed")

    flattened: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    cell_keys: set[tuple[str, str, str]] = set()
    asset_question_ids: dict[str, set[str]] = {
        asset_id: set() for asset_id in EXPECTED_ASSET_SIDES
    }
    asset_tier_counts: dict[tuple[str, str], int] = {}
    side_question_ids: dict[str, set[str]] = {"forehand": set(), "backhand": set()}
    for cell_index, raw_cell in enumerate(data["cells"]):
        cell = exact_keys(raw_cell, CELL_KEYS, f"cells[{cell_index}]")
        if not isinstance(cell["reported_proposals"], list):
            raise SelectionError(f"cells[{cell_index}].reported_proposals must be a list")
        if cell["accepted_candidate_ids"] != []:
            raise SelectionError(f"cells[{cell_index}] unexpectedly accepts a candidate")
        if cell["proposal_count_before_certificate"] != len(cell["reported_proposals"]):
            raise SelectionError(f"cells[{cell_index}] proposal count mismatch")
        require_sha(cell["question_id"], f"cells[{cell_index}].question_id")
        if cell["asset_id"] not in EXPECTED_ASSET_SIDES:
            raise SelectionError(f"cells[{cell_index}] unknown asset_id")
        if cell["side"] != EXPECTED_ASSET_SIDES[cell["asset_id"]]:
            raise SelectionError(f"cells[{cell_index}] side disagrees with asset")
        if cell["tier"] not in {"R0", "R1"}:
            raise SelectionError(f"cells[{cell_index}] tier changed")
        identity = (cell["asset_id"], cell["question_id"], cell["tier"])
        if identity in cell_keys:
            raise SelectionError(f"cells[{cell_index}] duplicates asset/question/tier")
        cell_keys.add(identity)
        asset_question_ids[cell["asset_id"]].add(cell["question_id"])
        side_question_ids[cell["side"]].add(cell["question_id"])
        count_key = (cell["asset_id"], cell["tier"])
        asset_tier_counts[count_key] = asset_tier_counts.get(count_key, 0) + 1
        for proposal_index, raw_proposal in enumerate(cell["reported_proposals"]):
            proposal = _validate_proposal(raw_proposal, cell, proposal_index)
            if proposal["candidate_id"] in candidate_ids:
                raise SelectionError(f"duplicate candidate_id {proposal['candidate_id']}")
            candidate_ids.add(proposal["candidate_id"])
            flattened.append(proposal)
    if len(flattened) != 22:
        raise SelectionError(f"flattened proposal count {len(flattened)} != 22")
    for asset_id in EXPECTED_ASSET_SIDES:
        if len(asset_question_ids[asset_id]) != 32:
            raise SelectionError(f"{asset_id} must bind exactly 32 same-side questions")
        for tier in ("R0", "R1"):
            if asset_tier_counts.get((asset_id, tier)) != 32:
                raise SelectionError(f"{asset_id}/{tier} must contain exactly 32 cells")
    if len(side_question_ids["forehand"]) != 32 or len(side_question_ids["backhand"]) != 32:
        raise SelectionError("each side must bind exactly 32 immutable questions")
    if side_question_ids["forehand"] & side_question_ids["backhand"]:
        raise SelectionError("forehand/backhand question IDs must be disjoint")
    if {row["asset_id"] for row in flattened} != set(TARGET_ASSETS):
        raise SelectionError("only B/C may have proposals in this exact artifact")
    return data, flattened


def _semantic_alias_key(proposal: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in proposal.items() if key not in {"candidate_id", "tier"}}
    return canonical_json_bytes(payload)


def deduplicate_candidates(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[bytes, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_semantic_alias_key(row), []).append(row)
    kept: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        if len(group) != 2 or finite_number(group[0]["yaw_deg"], "duplicate yaw") != 0.0:
            raise SelectionError("only one identical yaw-zero R0/R1 pair may be deduplicated")
        tiers = {row["tier"] for row in group}
        if tiers != {"R0", "R1"}:
            raise SelectionError("yaw-zero duplicate must be exactly an R0/R1 pair")
        survivor = next(row for row in group if row["tier"] == "R0")
        removed = next(row for row in group if row["tier"] == "R1")
        kept.append(survivor)
        aliases.append(
            {
                "semantic_survivor_candidate_id": survivor["candidate_id"],
                "removed_alias_candidate_id": removed["candidate_id"],
                "survivor_tier": "R0",
                "removed_tier": "R1",
            }
        )
    return kept, sorted(aliases, key=lambda row: row["semantic_survivor_candidate_id"])


def candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        finite_number(row["translation_norm_m"], "translation_norm_m"),
        abs(finite_number(row["yaw_deg"], "yaw_deg")),
        -finite_number(row["return_margin_m"], "return_margin_m"),
        -finite_number(row["source_dense_racket_body_clearance_m"], "source_dense_racket_body_clearance_m"),
        row["frame"],
        row["candidate_id"],
    )


def _candidate_ledger(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": row["candidate_id"],
        "question_id": row["question_id"],
        "tier": row["tier"],
        "frame": row["frame"],
        "translation_w_m": row["translation_w_m"],
        "translation_norm_m": row["translation_norm_m"],
        "yaw_deg": row["yaw_deg"],
        "return_margin_m": row["return_margin_m"],
        "source_dense_racket_body_clearance_m": row["source_dense_racket_body_clearance_m"],
        "source_motion_sha256": row["source_motion_sha256"],
        "predecessor_full_result_sha256": row["predecessor_full_result_sha256"],
        "transform_semantics": row["transform_semantics"],
    }


def build_selection(plan: Mapping[str, Any], prereg_sha: str, flattened: Sequence[dict[str, Any]]) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    for plan_row in plan["target_assets"]:
        asset_id = plan_row["asset_id"]
        low, high = TARGET_WINDOWS[asset_id]
        raw_rows = [row for row in flattened if row["asset_id"] == asset_id]
        if len(raw_rows) != EXPECTED_RAW_COUNTS[asset_id]:
            raise SelectionError(f"{asset_id} raw count {len(raw_rows)} != expected")
        outside = [row for row in raw_rows if not low <= row["frame"] <= high]
        if outside:
            raise SelectionError(f"{asset_id} contains proposals outside frozen frame window")
        unique, aliases = deduplicate_candidates(raw_rows)
        if len(unique) != EXPECTED_UNIQUE_COUNTS[asset_id]:
            raise SelectionError(f"{asset_id} unique count {len(unique)} != expected")
        ordered = sorted(unique, key=candidate_sort_key)
        ledger = [_candidate_ledger(row, index) for index, row in enumerate(ordered)]
        assets.append(
            {
                "asset_id": asset_id,
                "human_name": plan_row["human_name"],
                "inclusive_frame_window": [low, high],
                "raw_proposal_count": len(raw_rows),
                "deduplicated_proposal_count": len(ordered),
                "yaw_zero_r0_r1_aliases_removed": aliases,
                "selected_primary": ledger[0],
                "fallback_ladder": ledger[1:],
            }
        )
    result = {
        "schema_version": 1,
        "status": RESULT_STATUS,
        "formal_eligible": False,
        "materialization_authorized": False,
        "training_authorized": False,
        "topp_authorized": False,
        "hardware_authorized": False,
        "preregistration": {
            "path": "configs/motion_backhand_loop_bc_proposal_selection_prereg_20260713.json",
            "sha256": prereg_sha,
        },
        "consumer": {
            "path": "scripts/select_motion_spatial_retarget_candidates.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "proposal_input": {
            "bytes": plan["proposal_input"]["bytes"],
            "sha256": plan["proposal_input"]["sha256"],
        },
        "ranking": plan["ranking"],
        "fallback_policy": plan["fallback_policy"],
        "primary_count": 2,
        "assets": assets,
        "next_required_stage": "separately_preregistered_exact_candidate_materialization_and_certificates",
        "not_claimed": plan["not_claimed"],
    }
    if sum(1 for asset in assets if asset["selected_primary"]) != 2:
        raise SelectionError("selection must produce exactly one primary per B/C asset")
    return result


def validate_selection(plan: Mapping[str, Any], prereg_sha: str, selection: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "status", "formal_eligible", "materialization_authorized",
        "training_authorized", "topp_authorized", "hardware_authorized", "preregistration",
        "consumer", "proposal_input", "ranking", "fallback_policy", "primary_count", "assets",
        "next_required_stage", "not_claimed",
    }
    exact_keys(selection, expected_keys, "selection result")
    if selection["schema_version"] != 1 or selection["status"] != RESULT_STATUS:
        raise SelectionError("selection result schema/status mismatch")
    for field in ("formal_eligible", "materialization_authorized", "training_authorized", "topp_authorized", "hardware_authorized"):
        if selection[field] is not False:
            raise SelectionError(f"selection.{field} must remain false")
    if selection["preregistration"] != {
        "path": "configs/motion_backhand_loop_bc_proposal_selection_prereg_20260713.json",
        "sha256": prereg_sha,
    }:
        raise SelectionError("selection preregistration binding mismatch")
    if selection["consumer"] != plan["consumer"]:
        raise SelectionError("selection consumer binding mismatch")
    if selection["proposal_input"] != {key: plan["proposal_input"][key] for key in ("bytes", "sha256")}:
        raise SelectionError("selection proposal binding mismatch")
    if selection["ranking"] != plan["ranking"] or selection["fallback_policy"] != plan["fallback_policy"]:
        raise SelectionError("selection science contract mismatch")
    if selection["primary_count"] != 2 or selection["not_claimed"] != plan["not_claimed"]:
        raise SelectionError("selection count/claims mismatch")
    if selection["next_required_stage"] != "separately_preregistered_exact_candidate_materialization_and_certificates":
        raise SelectionError("selection next stage changed")
    assets = selection["assets"]
    if not isinstance(assets, list) or [row.get("asset_id") for row in assets if isinstance(row, dict)] != list(TARGET_ASSETS):
        raise SelectionError("selection assets must be exactly B/C")
    for asset in assets:
        expected_asset_keys = {
            "asset_id", "human_name", "inclusive_frame_window", "raw_proposal_count",
            "deduplicated_proposal_count", "yaw_zero_r0_r1_aliases_removed",
            "selected_primary", "fallback_ladder",
        }
        exact_keys(asset, expected_asset_keys, f"selection asset {asset.get('asset_id')}")
        asset_id = asset["asset_id"]
        if asset["inclusive_frame_window"] != list(TARGET_WINDOWS[asset_id]):
            raise SelectionError(f"{asset_id} result frame window changed")
        if asset["raw_proposal_count"] != EXPECTED_RAW_COUNTS[asset_id] or asset["deduplicated_proposal_count"] != EXPECTED_UNIQUE_COUNTS[asset_id]:
            raise SelectionError(f"{asset_id} result counts changed")
        ladder = [asset["selected_primary"], *asset["fallback_ladder"]]
        if len(ladder) != EXPECTED_UNIQUE_COUNTS[asset_id]:
            raise SelectionError(f"{asset_id} result ladder length changed")
        if [row.get("rank") for row in ladder if isinstance(row, dict)] != list(range(len(ladder))):
            raise SelectionError(f"{asset_id} ladder ranks are not contiguous")
        ids = [row.get("candidate_id") for row in ladder]
        if len(set(ids)) != len(ids) or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in ids):
            raise SelectionError(f"{asset_id} ladder candidate IDs invalid")
    return selection


def resolve_outcome(selection: Mapping[str, Any], asset_id: str, candidate_id: str, outcome_code: str) -> dict[str, Any]:
    if asset_id not in TARGET_ASSETS:
        raise SelectionError(f"unsupported asset_id: {asset_id}")
    asset = next(row for row in selection["assets"] if row["asset_id"] == asset_id)
    ladder = [asset["selected_primary"], *asset["fallback_ladder"]]
    matches = [index for index, row in enumerate(ladder) if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise SelectionError("candidate_id is not exactly one frozen candidate for asset")
    rank = matches[0]
    if outcome_code == ADVANCE_CODE:
        if rank + 1 < len(ladder):
            status = "next_fallback_selected_certification_blocked"
            next_candidate = ladder[rank + 1]
        else:
            status = "fallback_exhausted_asset_stopped"
            next_candidate = None
    elif outcome_code in STOP_CODES:
        status = "asset_stopped_no_fallback"
        next_candidate = None
    else:
        raise SelectionError(f"unknown outcome_code: {outcome_code}")
    return {
        "schema_version": 1,
        "status": status,
        "formal_eligible": False,
        "materialization_authorized": False,
        "training_authorized": False,
        "hardware_authorized": False,
        "asset_id": asset_id,
        "failed_candidate_id": candidate_id,
        "failed_candidate_rank": rank,
        "outcome_code": outcome_code,
        "next_candidate": next_candidate,
        "policy": selection["fallback_policy"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--expected-prereg-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the tracked selection contract only")
    select = subparsers.add_parser("select", help="consume the exact signed proposal artifact")
    select.add_argument("--proposals", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    resolve = subparsers.add_parser("resolve", help="apply the frozen fallback policy to one failure")
    resolve.add_argument("--selection", required=True, type=Path)
    resolve.add_argument("--expected-selection-sha256", required=True)
    resolve.add_argument("--asset-id", required=True)
    resolve.add_argument("--candidate-id", required=True)
    resolve.add_argument("--outcome-code", required=True)
    resolve.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan, prereg_sha = validate_plan(args.prereg, args.expected_prereg_sha256)
        if args.command == "validate":
            print(f"PASS selection contract {prereg_sha} (CPU-only; no proposal consumed)")
            return 0
        if args.command == "select":
            proposal_raw = read_stable_regular_file(args.proposals, "proposal input")
            _, flattened = validate_proposal_artifact(plan, proposal_raw)
            result = build_selection(plan, prereg_sha, flattened)
            size, digest = publish_no_clobber(args.output, result)
            selected = {row["asset_id"]: row["selected_primary"]["candidate_id"] for row in result["assets"]}
            print(f"PASS selected exactly B/C {selected}; output bytes={size} sha256={digest}; promotion blocked")
            return 0
        selection_sha = require_sha(args.expected_selection_sha256, "--expected-selection-sha256")
        selection_raw = read_stable_regular_file(args.selection, "selection input")
        actual_selection_sha = sha256_bytes(selection_raw)
        if actual_selection_sha != selection_sha:
            raise SelectionError(f"selection sha256 {actual_selection_sha} != {selection_sha}")
        selection = validate_selection(plan, prereg_sha, read_json_bytes(selection_raw, "selection input"))
        decision = resolve_outcome(selection, args.asset_id, args.candidate_id, args.outcome_code)
        size, digest = publish_no_clobber(args.output, decision)
        print(f"PASS fallback decision status={decision['status']} bytes={size} sha256={digest}; promotion blocked")
        return 0
    except (OSError, SelectionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
