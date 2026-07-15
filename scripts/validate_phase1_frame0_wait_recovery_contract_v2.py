#!/usr/bin/env python3
"""Validate the Phase-1 selected-action frame-0 wait/recovery v2 design.

This validator is CPU-only and side-effect free.  It proves exact design bytes,
the unchanged v1 parent, the read-only source audit, and the fail-closed launch
boundary.  It never imports Isaac, opens a simulator, contacts a Pod, or issues
a robot command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_CONTRACT_SHA256 = "cc05d63fa4e4ffd9515f369f176ba032ca2a46d8996431a7b1e7d34e2b1bf28e"
EXPECTED_CONTRACT_ID = "phase1-selected-action-frame0-wait-recovery-v2"
EXPECTED_CREATED_UTC = "2026-07-15T05:20:00Z"
EXPECTED_SCOPE = (
    "freeze the no-teleport continuous-episode waiting reference: the currently public "
    "action owns its exact frame-0 pose, every reference velocity is zero, XY is captured "
    "from the live station at phase entry, and the next action remains hidden until one atomic reveal"
)
EXPECTED_TOP_KEYS = {
    "schema_version",
    "contract_id",
    "created_utc",
    "status",
    "scope",
    "launch_authorized",
    "real_robot_authorized",
    "validator",
    "immutable_parent",
    "audited_source",
    "phase_contract",
    "reference_contract",
    "continuous_episode_carry_contract",
    "nonleakage_contract",
    "ready_contract",
    "source_audit",
    "implementation_bindings",
}

EXPECTED_PARENT = {
    "repo_path": "configs/phase1_recovery_tuple_abc_prereg_20260712.json",
    "bytes": 17008,
    "sha256": "ca7806df83b650546cf4406963bb231622a248c8e04e944991a371e44d810616",
    "bytes_may_be_rewritten": False,
    "relationship": (
        "v2 narrows waiting and recovery reference semantics without editing or "
        "reinterpreting the v1 preregistration bytes"
    ),
}
EXPECTED_AUDIT_COMMIT = "6c3e47d1f6305688450353de3dfa847fb9a65d2e"
EXPECTED_SOURCE_BLOBS = {
    "motion_command": {
        "repo_path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/commands.py"
        ),
        "bytes": 117787,
        "sha256": "3428dc1d77b12cc9965a862acd2cd2febbcbb6a6cbf3c3b2d556ec59c57aceaa",
    },
    "racket_target_command": {
        "repo_path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
        ),
        "bytes": 257190,
        "sha256": "b43e85da20cf0030618dddb1493955764ce8bd636b42c450950df910be03576b",
    },
}

EXPECTED_PHASE_CONTRACT = {
    "phase_order": [
        "post_swing_pre_reveal_recovery",
        "atomic_reveal",
        "revealed_wait",
        "swing",
    ],
    "post_swing_pre_reveal_recovery": {
        "public_action_identity": "just_completed_action",
        "reference_pose": "just_completed_action_exact_frame0",
        "reference_velocity": "all_zero",
        "future_action_fields_visible": False,
    },
    "atomic_reveal": {
        "public_action_identity": "newly_revealed_action",
        "reference_install": "one_atomic_generation",
        "physical_state_write": False,
        "deadline_shift_allowed": False,
    },
    "revealed_wait": {
        "public_action_identity": "newly_revealed_action",
        "reference_pose": "newly_revealed_action_exact_frame0",
        "reference_velocity": "all_zero",
        "hold_until_native_clip_release": True,
    },
    "swing": {
        "reference_pose": "selected_action_native_clip",
        "reference_velocity": "selected_action_native_clip",
        "starts_from_carried_physical_and_policy_state": True,
    },
}

EXPECTED_REFERENCE_CONTRACT = {
    "pose_source": "selected_public_action_exact_frame0",
    "default_joint_pos_substitution_allowed": False,
    "root_pose": "selected_action_frame0_z_and_orientation_plus_phase_entry_station_xy",
    "joint_pose": "selected_action_frame0_joint_position",
    "body_pose": "selected_action_frame0_body_position_and_orientation_with_same_xy_translation",
    "xy_anchor": {
        "source": "live_robot_station_xy_at_reference_phase_entry",
        "capture_count": "exactly_once_per_reference_phase_entry",
        "immutable_until": "next_atomic_reveal_or_true_episode_boundary",
        "live_per_tick_reanchor_allowed": False,
    },
    "velocity_fields_exact_zero": [
        "root_linear_velocity",
        "root_angular_velocity",
        "joint_velocity",
        "body_linear_velocity",
        "body_angular_velocity",
    ],
}

EXPECTED_CARRY_CONTRACT = {
    "reference_switch_only": True,
    "simulator_root_write_allowed": False,
    "simulator_joint_write_allowed": False,
    "teleport_allowed": False,
    "episode_reset_allowed": False,
    "observation_history_clear_allowed": False,
    "last_action_clear_or_replace_allowed": False,
    "action_delay_ring_clear_allowed": False,
    "target_delay_ring_clear_allowed": False,
    "noise_or_dropout_state_clear_allowed": False,
    "per_swing_bias_clear_allowed": False,
    "carried_state_fields": [
        "root_pose_and_velocity",
        "joint_pose_and_velocity",
        "observation_history",
        "executed_last_action",
        "action_delay_ring",
        "target_delay_ring",
        "noise_dropout_and_hold_last_state",
        "per_swing_bias_state",
    ],
}

EXPECTED_NONLEAKAGE = {
    "before_reveal_future_action_identity_visible": False,
    "before_reveal_future_clip_id_visible": False,
    "before_reveal_future_frame0_visible": False,
    "before_reveal_future_target_visible": False,
    "before_reveal_future_deadline_visible": False,
    "before_reveal_allowed_reference": "just_completed_public_action_frame0_zero_velocity",
    "reveal_updates": "action_identity_clip_frame0_target_and_deadline_one_atomic_generation",
    "partial_or_mixed_generation_visible": False,
}

EXPECTED_READY_CONJUNCTS = [
    "station_xy_and_yaw",
    "upright_height_and_gravity_projection",
    "root_joint_body_and_racket_low_velocity",
    "bilateral_support_contact_and_slip",
    "joint_torque_qdes_and_thermal_margin",
    "self_robot_table_net_and_ground_clearance",
    "enabled_next_action_and_deadline_reachability",
]

EXPECTED_SOURCE_AUDIT = {
    "observed_conflicts": [
        (
            "commands.py substitutes robot.data.default_joint_pos while held instead of "
            "the selected clip frame0 joint pose"
        ),
        (
            "commands.py zeroes joint and body velocities while held but leaves anchor root "
            "linear and angular reference velocities ungated"
        ),
        (
            "commands.py reanchors body XY to the live robot anchor every update instead of "
            "capturing one immutable phase-entry station XY"
        ),
    ],
    "adapter_implemented": False,
    "adapter_default_enabled": False,
    "safe_adapter_verdict": (
        "no_launch_until_phase_entry_anchor_atomicity_and_carry_state_receipts_are_implemented_"
        "and_runtime_verified"
    ),
    "reason_not_patched_in_this_contract": (
        "changing only joint_pos would create a mixed frame0/default/root reference; a safe "
        "adapter must update pose, root and body velocity, immutable XY anchoring, reveal "
        "atomicity and carry-state observation together"
    ),
}

EXPECTED_BINDINGS = {
    "selected_action_frame0_source_adapter": None,
    "phase_entry_xy_anchor_snapshot": None,
    "atomic_reveal_nonleakage_assertion": None,
    "continuous_carry_state_runtime_receipt": None,
    "ready_numeric_tolerances": None,
    "isaac_full_scene_probe": None,
    "vendor_mujoco_continuous_gate": None,
}


class ContractError(ValueError):
    """Raised when checked bytes or semantics differ from the frozen design."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(raw: str) -> None:
    raise ContractError(f"non-finite JSON constant: {raw}")


def _walk_finite(value: Any, where: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite JSON number at {where}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{where}.{key}")
        return
    raise ContractError(f"unsupported JSON value at {where}: {type(value).__name__}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs_no_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse contract JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("contract root must be an object")
    _walk_finite(value)
    return value


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _repo_root(contract_path: Path) -> Path:
    resolved = contract_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".git").exists() or (candidate / "docs/START_HERE.md").is_file():
            return candidate
    fallback = Path(__file__).resolve().parents[1]
    if (fallback / "docs/START_HERE.md").is_file():
        return fallback
    raise ContractError("cannot locate repository root from contract path")


def _verify_file_binding(root: Path, binding: dict[str, Any], label: str) -> None:
    expected_keys = {"repo_path", "bytes", "sha256"}
    if set(binding) != expected_keys:
        raise ContractError(f"{label} binding keyset changed")
    path = root / binding["repo_path"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if len(raw) != binding["bytes"] or sha256_bytes(raw) != binding["sha256"]:
        raise ContractError(f"{label} bytes changed")


def _verify_git_blob(root: Path, commit: str, binding: dict[str, Any], label: str) -> None:
    if not COMMIT_RE.fullmatch(commit):
        raise ContractError("audited source commit is not a full lowercase git SHA")
    if set(binding) != {"repo_path", "bytes", "sha256"}:
        raise ContractError(f"{label} binding keyset changed")
    if type(binding["bytes"]) is not int or binding["bytes"] <= 0:
        raise ContractError(f"{label} byte count is invalid")
    if not isinstance(binding["sha256"], str) or not SHA256_RE.fullmatch(binding["sha256"]):
        raise ContractError(f"{label} SHA-256 is invalid")
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{commit}:{binding['repo_path']}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"cannot read audited git blob {label}") from exc
    if len(raw) != binding["bytes"] or sha256_bytes(raw) != binding["sha256"]:
        raise ContractError(f"audited git blob changed for {label}")


def _validate_ready_contract(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "semantics",
        "all_conjuncts_required",
        "positive_reward_can_offset_failure",
        "numeric_thresholds_bound",
        "conjuncts",
        "missing_or_nonfinite_measurement",
    }:
        raise ContractError("ready contract keyset changed")
    if value["semantics"] != "all_tolerance_conjuncts_not_weighted_score":
        raise ContractError("ready semantics changed")
    if value["all_conjuncts_required"] is not True:
        raise ContractError("ready must require every tolerance conjunct")
    if value["positive_reward_can_offset_failure"] is not False:
        raise ContractError("ready failure cannot be offset by reward")
    if value["numeric_thresholds_bound"] is not False:
        raise ContractError("this design contract cannot claim numeric readiness thresholds")
    conjuncts = value["conjuncts"]
    expected = [
        {"id": item, "threshold_binding": None} for item in EXPECTED_READY_CONJUNCTS
    ]
    if not _strict_equal(conjuncts, expected):
        raise ContractError("ready tolerance conjunctions changed")
    if value["missing_or_nonfinite_measurement"] != "not_ready_fail_closed":
        raise ContractError("ready measurement failure policy changed")


def validate_contract(path: Path, expected_sha256: str = EXPECTED_CONTRACT_SHA256) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ContractError("expected contract SHA-256 must be 64 lowercase hex characters")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ContractError(
            f"contract SHA mismatch: actual={actual_sha} expected={expected_sha256}"
        )
    value = load_json_strict(path)
    if set(value) != EXPECTED_TOP_KEYS:
        raise ContractError("top-level keyset changed")
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise ContractError("schema_version must remain exact integer 2")
    if value["contract_id"] != EXPECTED_CONTRACT_ID:
        raise ContractError("contract_id changed")
    if value["created_utc"] != EXPECTED_CREATED_UTC:
        raise ContractError("created_utc changed")
    if value["scope"] != EXPECTED_SCOPE:
        raise ContractError("scope changed")
    if value["status"] != "design_validated_no_launch":
        raise ContractError("status changed")
    if value["launch_authorized"] is not False or value["real_robot_authorized"] is not False:
        raise ContractError("design contract must not authorize launch or a robot")
    if not _strict_equal(value["validator"], {
        "repo_path": "scripts/validate_phase1_frame0_wait_recovery_contract_v2.py"
    }):
        raise ContractError("validator binding changed")

    if not _strict_equal(value["immutable_parent"], EXPECTED_PARENT):
        raise ContractError("immutable v1 parent binding changed")
    root = _repo_root(path)
    _verify_file_binding(root, {
        "repo_path": EXPECTED_PARENT["repo_path"],
        "bytes": EXPECTED_PARENT["bytes"],
        "sha256": EXPECTED_PARENT["sha256"],
    }, "immutable v1 parent")

    audited = value["audited_source"]
    if not isinstance(audited, dict) or set(audited) != {"commit", "git_blobs"}:
        raise ContractError("audited source keyset changed")
    if audited["commit"] != EXPECTED_AUDIT_COMMIT:
        raise ContractError("audited source commit changed")
    if not _strict_equal(audited["git_blobs"], EXPECTED_SOURCE_BLOBS):
        raise ContractError("audited source blob bindings changed")
    for name, binding in EXPECTED_SOURCE_BLOBS.items():
        _verify_git_blob(root, EXPECTED_AUDIT_COMMIT, binding, name)

    if not _strict_equal(value["phase_contract"], EXPECTED_PHASE_CONTRACT):
        raise ContractError("phase and reveal contract changed")
    if not _strict_equal(value["reference_contract"], EXPECTED_REFERENCE_CONTRACT):
        raise ContractError("frame0 reference contract changed")
    if not _strict_equal(value["continuous_episode_carry_contract"], EXPECTED_CARRY_CONTRACT):
        raise ContractError("continuous carry-state contract changed")
    if not _strict_equal(value["nonleakage_contract"], EXPECTED_NONLEAKAGE):
        raise ContractError("future-action nonleakage contract changed")
    _validate_ready_contract(value["ready_contract"])
    if not _strict_equal(value["source_audit"], EXPECTED_SOURCE_AUDIT):
        raise ContractError("commands.py source audit changed")
    if not _strict_equal(value["implementation_bindings"], EXPECTED_BINDINGS):
        raise ContractError("implementation bindings changed")
    return value


def _launch_blockers(value: dict[str, Any]) -> list[str]:
    blockers = [name for name, binding in value["implementation_bindings"].items() if binding is None]
    if value["ready_contract"]["numeric_thresholds_bound"] is not True:
        blockers.append("ready_contract.numeric_thresholds_bound")
    if value["source_audit"]["adapter_implemented"] is not True:
        blockers.append("source_audit.adapter_implemented")
    return blockers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/phase1_frame0_wait_recovery_contract_v2_20260715.json"),
    )
    parser.add_argument(
        "--expected-contract-sha256",
        default=EXPECTED_CONTRACT_SHA256,
    )
    parser.add_argument("--mode", choices=("design-check", "launch-check"), default="design-check")
    args = parser.parse_args(argv)
    try:
        value = validate_contract(args.contract, args.expected_contract_sha256)
    except ContractError as exc:
        print(f"CONTRACT INVALID: {exc}", file=sys.stderr)
        return 2

    blockers = _launch_blockers(value)
    if args.mode == "launch-check":
        print(
            "LAUNCH BLOCKED: " + ", ".join(blockers),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({
        "adapter_implemented": value["source_audit"]["adapter_implemented"],
        "contract_id": value["contract_id"],
        "future_action_hidden_before_reveal": True,
        "launch_authorized": value["launch_authorized"],
        "old_prereg_unchanged": True,
        "ready_numeric_thresholds_bound": value["ready_contract"]["numeric_thresholds_bound"],
        "status": "pass_design_only",
        "velocity_reference": "root_joint_body_all_zero",
        "wait_pose": "selected_public_action_exact_frame0",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
