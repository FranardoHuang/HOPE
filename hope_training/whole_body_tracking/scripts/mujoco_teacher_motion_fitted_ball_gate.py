#!/usr/bin/env python3
"""Formal teacher-motion physical-return gate using the frozen venue fit.

Authority
---------
* MuJoCo owns the free ball state, gravity, robot FK, table/net geometry, and
  robot/table/self contacts.
* The frozen venue YAML owns drag, Magnus, table impulse, and paddle impulse.
* Ball native collision is disabled (``contype=conaffinity=0``) so a fitted
  impulse can never be double-counted.
* A paddle impulse is armed exactly once and can fire only at an actual swept
  intersection of the ball sphere with the selected red/black finite rubber
  surface from the official STL.  The event uses the physical face-center
  offset, ball radius, and point velocity ``v_site + omega x r``.
* A table impulse can fire only at a swept descending sphere/surface crossing
  inside the table footprint.  Net/post intersection is geometry failure.

The robot follows the exact schema-2 teacher at every physics substep.  This is
a teacher-motion physical compatibility Gate, not policy or PD-plant evidence.
Every action is replayed at both 1.0 ms and 0.5 ms; both must pass and their
contact/net/landing outputs must satisfy frozen convergence tolerances.

Fail-closed inputs
------------------
The action manifest must be exact N (default N=5), bind the versioned
face-center geometry migration, and carry a ``physical_contact_contract`` v2
plus a per-action ``physical_ball_launch`` receipt.  The checked-in July-28
"N5" manifest is N=4, contains retired ``fh_loop``, has WORKTREE solver pins,
and lacks all physical-contact-v2 fields, so preflight rejects it.  This file
never silently constructs a launch from the old analytic center.

Per-action motion evidence is deliberately a non-authorizing
``compiler_candidate_pre_admission_v1`` identity (registry, compiler manifest,
and independent bank PASS).  The final training promotion is downstream: it
may consume this Gate's PASS together with the Isaac PASS, so requiring that
promotion here would be a circular self-authorization.

Formal execution must enter through the isolated pinned-byte bootstrap and
must supply an independently preregistered launch-evidence trust root binding
each raw recording/solver input and its Git-pinned validator.  Direct core
execution can only produce a blocked preflight receipt.

Exit 0 means the teacher physical Gate passed.  Exit 4 means preflight passed
but no action ran.  Exit 3 means a complete scientific/preflight failure
receipt.  Exit 2 means malformed input or runtime infrastructure failure.  No
result authorizes deployment or hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
FIT_ROOT = REPO_ROOT / "hope_training/ball_physics_fit"
if __name__ == "__main__" and "--preflight-only" not in sys.argv:
    print(
        "[fitted-ball-gate][FATAL] formal execution must use "
        "mujoco_teacher_motion_fitted_ball_gate_bootstrap.py under "
        "python -I; direct core execution cannot produce PASS",
        file=sys.stderr,
    )
    raise SystemExit(2)
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(FIT_ROOT))

import action_ball_action_set_contract as action_set_contract  # noqa: E402
import contact_model  # noqa: E402
import mujoco_motion_player as motion_player  # noqa: E402
import mujoco_table_scene as table_scene  # noqa: E402
import mujoco_teacher_motion_native_ball_diagnostic as native_diag  # noqa: E402
import racket_geometry_contract as racket_geometry  # noqa: E402


SCHEMA_VERSION = 1
CONTACT_CONTRACT_VERSION = 2
CONTACT_AUTHORITY = "venue_fitted_swept_selected_face_v2"
BALL_BODY_NAME = "fitted_ball_body"
BALL_JOINT_NAME = "fitted_ball_freejoint"
BALL_GEOM_NAME = "fitted_ball_visual_no_native_contact"
TABLE_GEOM_NAME = "motion_table_top"
NET_GEOM_NAMES = (
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)
CONTACT_MODEL_PATH = FIT_ROOT / "contact_model.py"
ACTION_SET_CONTRACT_SOURCE_PATH = (
    HERE / "action_ball_action_set_contract.py"
)
CONTACT_MODEL_SHA256 = (
    "b022bfbd2fa62520759721436a526d2f46036b4b3f929ecae81b99c82afbde2a"
)
CANONICAL_MJCF = native_diag.DEFAULT_MJCF
CANONICAL_IDENTITY_MANIFEST = native_diag.DEFAULT_IDENTITY_MANIFEST
CANONICAL_IDENTITY_MANIFEST_SHA256 = (
    native_diag.DEFAULT_IDENTITY_MANIFEST_SHA256
)
RUNTIME_SOURCE_PATHS = {
    "action_ball_action_set_contract.py": (
        HERE / "action_ball_action_set_contract.py"
    ),
    "mujoco_teacher_motion_fitted_ball_gate.py": Path(__file__).resolve(),
    "mujoco_teacher_motion_native_ball_diagnostic.py": (
        HERE / "mujoco_teacher_motion_native_ball_diagnostic.py"
    ),
    "mujoco_motion_player.py": HERE / "mujoco_motion_player.py",
    "racket_geometry_contract.py": HERE / "racket_geometry_contract.py",
    "canonical_mujoco_identity.py": HERE / "canonical_mujoco_identity.py",
}
RUNTIME_EXECUTION_SOURCE_PATHS = {
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py": (
        HERE / "action_ball_action_set_contract.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate_bootstrap.py": (
        HERE / "mujoco_teacher_motion_fitted_ball_gate_bootstrap.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate.py": Path(__file__).resolve(),
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_native_ball_diagnostic.py": (
        HERE / "mujoco_teacher_motion_native_ball_diagnostic.py"
    ),
    "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py": (
        HERE / "mujoco_motion_player.py"
    ),
    "hope_training/whole_body_tracking/scripts/audit_motion_npz.py": (
        HERE / "audit_motion_npz.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "motion_kinematics_contract.py": (
        HERE / "motion_kinematics_contract.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "racket_geometry_contract.py": (
        HERE / "racket_geometry_contract.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "canonical_mujoco_identity.py": (
        HERE / "canonical_mujoco_identity.py"
    ),
    "hope_training/ball_physics_fit/contact_model.py": CONTACT_MODEL_PATH,
    "scripts/mujoco_table_scene.py": REPO_ROOT
    / "scripts/mujoco_table_scene.py",
    "scripts/audit_motion_schema2_table_net_clearance.py": REPO_ROOT
    / "scripts/audit_motion_schema2_table_net_clearance.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/geometry.py"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/table_frame.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/table_frame.py"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/"
        "racket_contact_geometry.py"
    ),
}
RUNTIME_EXECUTION_DATA_PATHS = {
    "configs/a3_runtime_body_order.txt": (
        REPO_ROOT / "configs/a3_runtime_body_order.txt"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ),
}
BOOTSTRAP_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate_bootstrap.py"
)
REQUIRED_EXTERNAL_DISTRIBUTIONS = ("mujoco", "numpy")
FACE_MESH_PATHS = {
    1: (
        REPO_ROOT
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/meshes/pingpang_red_Link.STL"
    ),
    -1: (
        REPO_ROOT
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/meshes/pingpang_black_Link.STL"
    ),
}
FACE_MESH_PIN_KEYS = {
    "red": 1,
    "black": -1,
}
FACE_OUTER_Y_M = {
    1: float(racket_geometry.RED_OUTER_Y_FROM_SITE_M),
    -1: float(racket_geometry.BLACK_OUTER_Y_FROM_SITE_M),
}
DEFAULT_DT_S = (0.001, 0.0005)
FRESH_N5_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
FRESH_N5_FAMILY = {
    "bh_loop_c": "backhand",
    "v12_forehand_block": "forehand",
    "bh_block": "backhand",
    "s0_highpress": "backhand",
    "fh_loop_high": "forehand",
}
FORMAL_CONTACT_TIME_TOL_S = 0.010
FORMAL_CONTACT_POSITION_TOL_M = 0.005
FORMAL_REFERENCE_SITE_SPEED_TOL_MPS = 0.05
FORMAL_TASK_FACE_NORMAL_ANGLE_TOL_RAD = 0.01
FORMAL_TASK_LANDING_TOL_M = 0.10
FORMAL_SOLVER_RESIDUAL_MAX_M = 0.02
FORMAL_INCOMING_VELOCITY_TOL_MPS = 0.10
FORMAL_INCOMING_SPIN_TOL_RADPS = 2.0
FORMAL_TASK_TIME_IDENTITY_TOL_S = 1.0e-9
FORMAL_TASK_VECTOR_IDENTITY_TOL = 1.0e-9
FORMAL_MIN_LAUNCH_LEAD_S = 0.10
FORMAL_HOLDOUT_PER_ACTION_MIN = 768
FORMAL_POST_CONTACT_S = 1.5
FORMAL_SHARED_READY_JOINT_TOL_RAD = 1.0e-6
FORMAL_SHARED_READY_ROOT_POSITION_TOL_M = 1.0e-6
FORMAL_SHARED_READY_ROOT_ORIENTATION_TOL_RAD = 1.0e-6
FORMAL_RECOVERY_JOINT_TOL_RAD = 1.0e-6
FORMAL_RECOVERY_ROOT_POSITION_TOL_M = 1.0e-6
FORMAL_RECOVERY_ROOT_ORIENTATION_TOL_RAD = 1.0e-6
FORMAL_ENDPOINT_VELOCITY_TOL = 1.0e-9
FORMAL_EVENT_TIME_GUARD_S = 1.0e-6
FORMAL_LANDING_DEPTH_GUARD_M = 0.005
FORMAL_MIN_RETURN_NORMAL_X = 1.0e-6
FORMAL_FACE_EDGE_GUARD_M = 0.0005
FORMAL_SHADOW_MAX_DT_S = 0.00005
FORMAL_SHADOW_MAX_BALL_STEP_M = 0.00025
FORMAL_SHADOW_CLEARANCE_GUARD_M = 0.0005
FORMAL_SHADOW_MAX_ROBOT_SURFACE_STEP_M = 0.000125
FORMAL_ROBOT_OBSTACLE_GUARD_M = 0.0005
FORMAL_SHADOW_EVENT_MATCH_S = 0.00025
FORMAL_SHADOW_EVENT_POSITION_MATCH_M = (
    2.0 * FORMAL_SHADOW_MAX_BALL_STEP_M
    + FORMAL_SHADOW_CLEARANCE_GUARD_M
)
TABLE_CONTACT_FORCE_THRESHOLD_N = 1.0e-6
FLOOR_GEOM_NAME = "floor"
LEGAL_FOOT_BODY_NAMES = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
)
FOOT_FLOOR_PENETRATION_TOLERANCE_M = 0.002
NONFOOT_FLOOR_PENETRATION_TOLERANCE_M = 0.0001
FORMAL_NONFOOT_GROUND_CLEARANCE_GUARD_M = 0.0005
FORMAL_GROUND_DISTANCE_QUERY_CAP_M = 0.01
FORMAL_GROUND_DISTANCE_NUMERIC_TOL_M = 1.0e-9
FIVE_SOLID_SWEEP_METHOD = (
    "linear_geom_center_segment_plus_rotation_invariant_mujoco_rbound_v1"
)
ROOT_Z_FALL_M = 0.55
ROOT_TILT_FALL_RAD = 0.70
EPS = 1.0e-12
PHYSICAL_TASK_BINDING_SCHEMA_VERSION = 1
PHYSICAL_TASK_BINDING_AUTHORITY = (
    "pre_registered_frozen_action_ball_solver_receipt_v1"
)
PHYSICAL_TASK_CASE_ROLES = (
    "center_positive_seed_0",
    "center_positive_seed_1",
    "support_positive",
    "negative_t_hit_offset",
    "negative_face_sign",
    "negative_ball_state_mismatch",
)
PHYSICAL_TASK_POSITIVE_ROLES = frozenset(
    PHYSICAL_TASK_CASE_ROLES[:3]
)
PHYSICAL_TASK_NEGATIVE_EXPECTED_REASON = {
    "negative_t_hit_offset": "teacher_task_contact_time_mismatch",
    "negative_face_sign": "teacher_task_face_sign_mismatch",
    "negative_ball_state_mismatch": (
        "teacher_task_ball_state_mismatch"
    ),
}
PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND = (
    "fresh_n5_disposable_physical_gate_manifest_materialization_v1"
)
GENERIC_PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND = (
    "action_ball_disposable_physical_gate_manifest_materialization_v2"
)
MATERIALIZATION_PROFILE_CENTER_KEYS = (
    "contact_offset_center_b_yaw_m",
    "time_to_contact_center_s",
    "incoming_direction_center_b_yaw",
    "incoming_speed_center_mps",
    "spin_direction_center_b_yaw",
    "spin_magnitude_center_radps",
    "base_spawn_center_w_xy_m",
    "base_travel_center_b_yaw_xy_m",
)
MATERIALIZATION_ACTION_IDENTITY_KEYS = frozenset(
    {
        "action_id",
        "action_uid",
        "family",
        "motion_path",
        "motion_sha256",
        "scope",
        "profile_center",
        "profile_center_sha256",
    }
)
RECORDED_POSITION_VENUE_FIT_ZERO_SPIN_SOURCE = (
    "recorded_position_plus_venue_fit_velocity_zero_spin_v1"
)
PHYSICAL_GATE_TOP_LEVEL_FIELDS = (
    "racket_geometry_contract",
    "physical_contact_contract",
)
PHYSICAL_GATE_ACTION_FIELDS = (
    "physical_ball_launch",
    "physical_task_binding",
    "admission",
)


class FittedGateError(ValueError):
    """Formal contact, identity, or runtime invariant failed."""


def _repo_pin(path: Path, sha256: str) -> Dict[str, str]:
    try:
        relative = path.expanduser().resolve().relative_to(
            REPO_ROOT.resolve()
        ).as_posix()
    except ValueError as exc:
        raise FittedGateError(
            f"formal physical input is outside repository: {path}"
        ) from exc
    return {"path": relative, "sha256": sha256}


def _logical_repo_path(value: Any, label: str) -> str:
    raw = native_diag._nonempty_string(value, label)
    if Path(raw).is_absolute():
        raise FittedGateError(f"{label} must be repository-relative")
    candidate = (REPO_ROOT / raw).resolve()
    try:
        relative = candidate.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FittedGateError(
            f"{label} escapes the repository root: {raw!r}"
        ) from exc
    if relative != raw:
        raise FittedGateError(
            f"{label} must be a normalized repository path: {raw!r}"
        )
    return relative


def materialization_action_identity_matrix(
    strict_manifest: Mapping[str, Any],
    trusted_action_set: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Derive the reviewable exact action/profile-center closure.

    The strict manifest SHA already commits these bytes, but schema 2 also
    exposes the safety-critical identities directly in its receipt so a
    consumer never has to infer family, motion bytes, or the sampled center.
    """

    action_order = strict_manifest.get("action_order")
    actions = strict_manifest.get("actions")
    expected_ids = list(trusted_action_set["ordered_action_ids"])
    expected_uids = list(trusted_action_set["ordered_action_uids"])
    if (
        type(action_order) is not list
        or action_order != expected_ids
        or type(actions) is not list
        or len(actions) != len(expected_ids)
    ):
        raise FittedGateError(
            "strict manifest action matrix is disconnected from the "
            "trusted action-set contract"
        )
    scope = native_diag._nonempty_string(
        trusted_action_set.get("scope"),
        "trusted action-set scope",
    )
    output: List[Dict[str, Any]] = []
    for index, (action_id, action_uid, raw_row) in enumerate(
        zip(expected_ids, expected_uids, actions)
    ):
        row = native_diag._mapping(
            raw_row, f"strict manifest action[{index}]"
        )
        if (
            row.get("action_id") != action_id
            or row.get("action_uid") != action_uid
        ):
            raise FittedGateError(
                f"strict manifest action[{index}] ID/UID is disconnected"
            )
        family = native_diag._nonempty_string(
            row.get("family"), f"{action_id}.family"
        )
        motion_path = _logical_repo_path(
            row.get("motion_path"), f"{action_id}.motion_path"
        )
        motion_sha = native_diag._require_sha(
            row.get("motion_sha256"), f"{action_id}.motion_sha256"
        )
        ball_profile = native_diag._mapping(
            row.get("ball_profile"), f"{action_id}.ball_profile"
        )
        missing = [
            key
            for key in MATERIALIZATION_PROFILE_CENTER_KEYS
            if key not in ball_profile
        ]
        if missing:
            raise FittedGateError(
                f"{action_id}: ball profile lacks center fields {missing}"
            )
        center = {
            key: ball_profile[key]
            for key in MATERIALIZATION_PROFILE_CENTER_KEYS
        }
        center_sha = native_diag.sha256_bytes(
            native_diag.canonical_json_bytes(center)
        )
        output.append(
            {
                "action_id": action_id,
                "action_uid": action_uid,
                "family": family,
                "motion_path": motion_path,
                "motion_sha256": motion_sha,
                "scope": scope,
                "profile_center": center,
                "profile_center_sha256": center_sha,
            }
        )
    return output


def validate_staged_registry_overrides(
    repo_file_overrides: Optional[Mapping[str, Path]],
) -> Dict[str, Path]:
    """Validate narrowly scoped pre-publication registry-entry overrides.

    Keys remain normalized repository-relative identities.  Values may point
    at a staging directory, but must already be regular non-symlink files.
    Compiler manifests and bank reports intentionally have no override path.
    """

    if repo_file_overrides is None:
        return {}
    if not isinstance(repo_file_overrides, Mapping):
        raise FittedGateError("repo_file_overrides must be a mapping")
    validated: Dict[str, Path] = {}
    for raw_key, raw_path in repo_file_overrides.items():
        key = _logical_repo_path(raw_key, "registry override key")
        if not isinstance(raw_path, Path):
            raise FittedGateError(
                f"registry override {key!r} value must be pathlib.Path"
            )
        unresolved = raw_path.expanduser()
        if unresolved.is_symlink() or not unresolved.is_file():
            raise FittedGateError(
                f"registry override {key!r} must be a regular "
                f"non-symlink file: {unresolved}"
            )
        resolved = unresolved.resolve()
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise FittedGateError(
                f"registry override {key!r} is not a regular file: "
                f"{resolved}"
            )
        validated[key] = resolved
    return validated


def _resolve_registry_entry(
    logical_path: Any,
    label: str,
    *,
    repo_file_overrides: Mapping[str, Path],
) -> Path:
    logical = _logical_repo_path(logical_path, label)
    override = repo_file_overrides.get(logical)
    if override is not None:
        return override
    return native_diag._resolve_repo_file(logical, label)


def validate_physical_materialization_receipt(
    receipt: Mapping[str, Any],
    *,
    strict_manifest_pin: Mapping[str, str],
    physical_manifest_pin: Mapping[str, str],
    trusted_action_set: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the pure receipt schema before following any file edges.

    Schema 1 is deliberately restricted to the code-owned fresh N5 profile.
    Schema 2 is the arbitrary-N contract and binds the complete normalized
    action-set contract in addition to the strict and disposable manifests.
    """

    common_keys = {
        "schema_version",
        "kind",
        "strict_training_manifest",
        "physical_task_bundle",
        "physical_gate_manifest",
        "candidate_entries",
        "compiler_manifests",
        "bank_gate_reports",
        "action_order",
        "strict_training_manifest_preserved",
        "inline_manifest_gate_only",
        "selector_executed",
        "authorization_granted",
    }
    schema_version = receipt.get("schema_version")
    kind = receipt.get("kind")
    if (
        schema_version == 1
        and kind == PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND
    ):
        if set(receipt) != common_keys:
            raise FittedGateError(
                "N5 physical-gate materialization receipt keys are not exact"
            )
        if (
            trusted_action_set.get("profile_id")
            != "fresh_upper_nomove_n5_v3"
            or trusted_action_set.get("expected_n") != 5
        ):
            raise FittedGateError(
                "schema-1 fresh-N5 materialization receipt cannot authorize "
                "a non-N5 action set"
            )
    elif (
        schema_version == 2
        and kind == GENERIC_PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND
    ):
        expected_keys = common_keys | {
            "action_set_contract",
            "action_identity_matrix",
        }
        if set(receipt) != expected_keys:
            raise FittedGateError(
                "arbitrary-N physical-gate materialization receipt keys "
                "are not exact"
            )
        if receipt.get("action_set_contract") != dict(trusted_action_set):
            raise FittedGateError(
                "arbitrary-N materialization receipt action-set contract "
                "is disconnected"
            )
        identity_rows = receipt.get("action_identity_matrix")
        if (
            type(identity_rows) is not list
            or len(identity_rows)
            != int(trusted_action_set["expected_n"])
            or [
                row.get("action_id")
                for row in identity_rows
                if type(row) is dict
            ]
            != list(trusted_action_set["ordered_action_ids"])
            or [
                row.get("action_uid")
                for row in identity_rows
                if type(row) is dict
            ]
            != list(trusted_action_set["ordered_action_uids"])
            or any(
                type(row) is not dict
                or set(row) != MATERIALIZATION_ACTION_IDENTITY_KEYS
                or row.get("scope") != trusted_action_set["scope"]
                for row in identity_rows
            )
        ):
            raise FittedGateError(
                "arbitrary-N materialization action identity matrix "
                "is not exact"
            )
    else:
        raise FittedGateError(
            "unsupported physical-gate materialization receipt schema/kind"
        )
    if (
        receipt.get("strict_training_manifest")
        != dict(strict_manifest_pin)
        or receipt.get("physical_gate_manifest")
        != dict(physical_manifest_pin)
        or receipt.get("action_order")
        != list(trusted_action_set["ordered_action_ids"])
        or receipt.get("strict_training_manifest_preserved") is not True
        or receipt.get("inline_manifest_gate_only") is not True
        or receipt.get("selector_executed") is not False
        or receipt.get("authorization_granted") is not False
    ):
        raise FittedGateError(
            "physical-gate materialization receipt crossbinding drifted"
        )
    return receipt


def validate_physical_materialization_closure(
    *,
    strict_manifest: Mapping[str, Any],
    strict_manifest_path: Path,
    strict_manifest_sha256: str,
    physical_manifest: Mapping[str, Any],
    physical_manifest_path: Path,
    physical_manifest_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
    trusted_action_set: Mapping[str, Any],
    repo_file_overrides: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    """Prove a disposable physical overlay is exactly one strict manifest.

    The strict manifest remains the actor/training identity.  The disposable
    manifest may add only two top-level physical fields and exactly three
    per-action evidence fields.  The separately pinned materialization receipt
    and its physical-task bundle must cross-bind both byte identities.
    """

    staged_registry = validate_staged_registry_overrides(
        repo_file_overrides
    )
    receipt, receipt_file = native_diag.read_json_exact(
        receipt_path,
        "physical-gate materialization receipt",
        expected_sha256=receipt_sha256,
    )
    strict_pin = _repo_pin(
        strict_manifest_path, strict_manifest_sha256
    )
    physical_pin = _repo_pin(
        physical_manifest_path, physical_manifest_sha256
    )
    validate_physical_materialization_receipt(
        receipt,
        strict_manifest_pin=strict_pin,
        physical_manifest_pin=physical_pin,
        trusted_action_set=trusted_action_set,
    )
    if receipt.get("schema_version") == 2:
        expected_identity_matrix = materialization_action_identity_matrix(
            strict_manifest, trusted_action_set
        )
        if (
            receipt.get("action_identity_matrix")
            != expected_identity_matrix
        ):
            raise FittedGateError(
                "schema-2 materialization action family/motion/profile "
                "center matrix drifted"
            )
    bundle_pin = native_diag._mapping(
        receipt.get("physical_task_bundle"),
        "physical materialization receipt bundle pin",
    )
    if set(bundle_pin) != {"path", "sha256"}:
        raise FittedGateError(
            "physical task bundle pin keys are not exact"
        )
    bundle_path = native_diag._resolve_repo_file(
        bundle_pin.get("path"), "physical task bundle path"
    )
    bundle_sha = native_diag._require_sha(
        bundle_pin.get("sha256"), "physical task bundle SHA"
    )
    bundle, bundle_file = native_diag.read_json_exact(
        bundle_path,
        "physical task bundle",
        expected_sha256=bundle_sha,
    )
    schema2 = receipt.get("schema_version") == 2
    base = native_diag._mapping(
        bundle.get("base_manifest"),
        "physical task bundle base manifest",
    )
    if (
        base
        != {
            "path": strict_pin["path"],
            "raw_sha256": strict_pin["sha256"],
            "schema_version": 3,
            "strict_training_input": True,
        }
        or bundle.get("action_order")
        != list(trusted_action_set["ordered_action_ids"])
        or bundle.get("mobility_mode")
        != trusted_action_set["mobility_mode"]
        or bundle.get("selector_executed") is not False
        or bundle.get("action_identity_frozen") is not True
        or bundle.get("action_switching_allowed") is not False
    ):
        raise FittedGateError(
            "physical task bundle is disconnected from the trusted action set"
        )
    if schema2:
        expected_bundle_keys = {
            "base_manifest",
            "action_order",
            "mobility_mode",
            "selector_executed",
            "action_identity_frozen",
            "action_switching_allowed",
            "gate_materialization_fields",
            "action_identity_matrix",
            "actions",
        }
        if (
            set(bundle) != expected_bundle_keys
            or bundle.get("action_identity_matrix")
            != receipt.get("action_identity_matrix")
        ):
            raise FittedGateError(
                "schema-2 physical task bundle identity matrix/key set "
                "is not exact"
            )
    gate_fields = native_diag._mapping(
        bundle.get("gate_materialization_fields"),
        "physical task bundle gate fields",
    )
    if set(gate_fields) != set(PHYSICAL_GATE_TOP_LEVEL_FIELDS):
        raise FittedGateError(
            "physical task bundle gate fields are not exact"
        )
    stripped_top = dict(physical_manifest)
    physical_actions = stripped_top.pop("actions", None)
    gate_top = {
        name: stripped_top.pop(name, None)
        for name in PHYSICAL_GATE_TOP_LEVEL_FIELDS
    }
    strict_top = dict(strict_manifest)
    strict_actions = strict_top.pop("actions", None)
    if (
        stripped_top != strict_top
        or gate_top != dict(gate_fields)
        or type(physical_actions) is not list
        or type(strict_actions) is not list
        or len(physical_actions)
        != int(trusted_action_set["expected_n"])
        or len(strict_actions)
        != int(trusted_action_set["expected_n"])
    ):
        raise FittedGateError(
            "disposable physical manifest modified strict top-level fields"
        )
    bundle_actions = bundle.get("actions")
    if type(bundle_actions) is not list or len(bundle_actions) != len(
        physical_actions
    ):
        raise FittedGateError(
            "physical task bundle action matrix is not exact N"
        )
    bundle_action_order = [
        row.get("action_id") if type(row) is dict else None
        for row in bundle_actions
    ]
    if bundle_action_order != list(
        trusted_action_set["ordered_action_ids"]
    ):
        raise FittedGateError(
            "physical task bundle action rows are not in exact trusted order"
        )
    bundle_by_id = {
        row.get("action_id"): row
        for row in bundle_actions
        if type(row) is dict
    }
    candidate_rows = receipt.get("candidate_entries")
    if type(candidate_rows) is not list or len(candidate_rows) != len(
        physical_actions
    ):
        raise FittedGateError(
            "physical receipt candidate entries are not exact N"
        )
    for index, (physical_row, strict_row, action_id, action_uid) in enumerate(
        zip(
            physical_actions,
            strict_actions,
            trusted_action_set["ordered_action_ids"],
            trusted_action_set["ordered_action_uids"],
        )
    ):
        if type(physical_row) is not dict or type(strict_row) is not dict:
            raise FittedGateError(
                f"physical manifest action row {index} is malformed"
            )
        stripped = dict(physical_row)
        overlay = {
            name: stripped.pop(name, None)
            for name in PHYSICAL_GATE_ACTION_FIELDS
        }
        if stripped != strict_row:
            raise FittedGateError(
                f"physical manifest action {action_id} modified strict fields"
            )
        bundle_row = bundle_by_id.get(action_id)
        if (
            type(bundle_row) is not dict
            or (
                schema2
                and set(bundle_row)
                != {
                    "action_id",
                    "action_uid",
                    "physical_ball_launch",
                    "physical_task_binding",
                }
            )
            or bundle_row.get("action_uid") != action_uid
            or bundle_row.get("physical_ball_launch")
            != overlay["physical_ball_launch"]
            or bundle_row.get("physical_task_binding")
            != overlay["physical_task_binding"]
        ):
            raise FittedGateError(
                f"physical manifest action {action_id} differs from bundle"
            )
        candidate = candidate_rows[index]
        admission = overlay["admission"]
        if (
            type(candidate) is not dict
            or set(candidate) != {"action_id", "path", "sha256"}
            or candidate.get("action_id") != action_id
            or type(admission) is not dict
            or admission.get("registry_entry_path")
            != candidate.get("path")
            or admission.get("registry_entry_sha256")
            != candidate.get("sha256")
        ):
            raise FittedGateError(
                f"physical receipt candidate {action_id} is disconnected"
            )
        candidate_path = _resolve_registry_entry(
            candidate["path"],
            f"{action_id} physical candidate path",
            repo_file_overrides=staged_registry,
        )
        candidate_sha = native_diag._require_sha(
            candidate["sha256"],
            f"{action_id} physical candidate SHA",
        )
        if native_diag.sha256_file(candidate_path) != candidate_sha:
            raise FittedGateError(
                f"{action_id}: physical candidate bytes drifted"
            )
        if schema2:
            for group_name, path_key, sha_key in (
                (
                    "compiler_manifests",
                    "compiler_manifest_path",
                    "compiler_manifest_sha256",
                ),
                (
                    "bank_gate_reports",
                    "bank_gate_report_path",
                    "bank_gate_report_sha256",
                ),
            ):
                group = receipt.get(group_name)
                if type(group) is not list or len(group) != len(
                    physical_actions
                ):
                    raise FittedGateError(
                        f"schema-2 receipt {group_name} is not exact N"
                    )
                evidence_row = group[index]
                if (
                    type(evidence_row) is not dict
                    or set(evidence_row)
                    != {"action_id", "path", "sha256"}
                    or evidence_row.get("action_id") != action_id
                    or admission.get(path_key)
                    != evidence_row.get("path")
                    or admission.get(sha_key)
                    != evidence_row.get("sha256")
                ):
                    raise FittedGateError(
                        f"{action_id}: schema-2 {group_name} pin is "
                        "disconnected from admission"
                    )
                evidence_path = native_diag._resolve_repo_file(
                    evidence_row["path"],
                    f"{action_id} {group_name} path",
                )
                evidence_sha = native_diag._require_sha(
                    evidence_row["sha256"],
                    f"{action_id} {group_name} SHA",
                )
                if native_diag.sha256_file(evidence_path) != evidence_sha:
                    raise FittedGateError(
                        f"{action_id}: schema-2 {group_name} bytes drifted"
                    )
    expected_registry_paths = {
        native_diag._nonempty_string(
            row.get("path"), "physical receipt candidate path"
        )
        for row in candidate_rows
        if type(row) is dict
    }
    if set(staged_registry) - expected_registry_paths:
        raise FittedGateError(
            "repo_file_overrides contains unused or non-registry paths"
        )
    return {
        "receipt": receipt_file,
        "schema_version": receipt["schema_version"],
        "kind": receipt["kind"],
        "bundle": bundle_file,
        "strict_manifest": strict_pin,
        "physical_gate_manifest": physical_pin,
        "physical_task_bundle": {
            "path": str(bundle_path),
            "sha256": bundle_sha,
        },
        "action_set_contract_sha256": trusted_action_set[
            "contract_sha256"
        ],
        "action_identity_matrix_sha256": (
            None
            if receipt.get("schema_version") != 2
            else native_diag.sha256_bytes(
                native_diag.canonical_json_bytes(
                    receipt["action_identity_matrix"]
                )
            )
        ),
    }


@dataclass(frozen=True)
class VenueParams:
    path: Path
    sha256: str
    ball_mass: float
    ball_radius: float
    inertia_coeff: float
    gravity: float
    k_d: float
    k_m: float
    table_e: float
    table_a_t: float
    table_b_t: float
    table_mu: float
    paddle_g1: float
    paddle_g2: float
    paddle_a_t: float
    paddle_b_t: float
    paddle_mu: float


@dataclass(frozen=True)
class LaunchState:
    source: str
    activation_time_s: float
    position_w_m: np.ndarray
    velocity_w_mps: np.ndarray
    spin_w_radps: np.ndarray
    state_sha256: str
    source_artifact_path: Path
    source_artifact_sha256: str


@dataclass(frozen=True)
class CaseLaunchState:
    """Manifest-sealed initial state for one physical task-control case."""

    activation_time_s: float
    position_w_m: np.ndarray
    velocity_w_mps: np.ndarray
    spin_w_radps: np.ndarray
    required_incoming_table_bounces: int
    state_sha256: str


@dataclass(frozen=True)
class PhysicalTaskCase:
    """One preregistered solver-bound physical positive/negative control."""

    case_id: str
    case_role: str
    sample_seed: int
    expected_physical_verdict: str
    expected_failure_reason: Optional[str]
    launch: CaseLaunchState
    ball_contact_w_m: np.ndarray
    incoming_velocity_w_mps: np.ndarray
    incoming_spin_w_radps: np.ndarray
    time_to_contact_s: float
    base_spawn_w_m: np.ndarray
    base_goal_w_m: np.ndarray
    landing_aim_w_xy_m: np.ndarray
    mount_normal_sign: int
    racket_site_target_w_m: np.ndarray
    racket_normal_w: np.ndarray
    reference_racket_quat_wxyz: np.ndarray
    reference_racket_angular_velocity_w_radps: np.ndarray
    racket_command_quat_wxyz: np.ndarray
    racket_face_center_velocity_w_mps: np.ndarray
    racket_site_velocity_w_mps: np.ndarray
    racket_command_angular_velocity_w_radps: np.ndarray
    geometry_source_sha256: str
    teacher_rate_min: float
    teacher_rate_max: float
    teacher_rate: float
    scaled_t_hit_s: float
    scaled_t_cycle_s: float
    pre_swing_wait_s: float
    solver_residual_m: float
    ball_proposal_sha256: str
    task_payload_sha256: str
    case_binding_sha256: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class PhysicalTaskBinding:
    """Exact action identity, solver identity, and control-case closure."""

    ball_profile_sha256: str
    solver_profile_sha256: str
    physics_profile_sha256: str
    solver_source_sha256: Mapping[str, str]
    solver_execution_receipt_path: Path
    solver_execution_receipt_sha256: str
    solver_execution_receipt_payload_sha256: str
    cases_sha256: str
    cases: Tuple[PhysicalTaskCase, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class PhysicalManifest:
    base: native_diag.ManifestContract
    raw: Mapping[str, Any]
    contract: Mapping[str, Any]
    launches: Mapping[str, LaunchState]
    launch_source_receipts: Mapping[str, Mapping[str, Any]]
    task_bindings: Mapping[str, PhysicalTaskBinding] = field(
        default_factory=dict
    )
    action_set_contract: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceMesh:
    sign: int
    path: Path
    sha256: str
    outer_y_m: float
    triangles_xz_m: np.ndarray
    boundary_edges_xz_m: np.ndarray


@dataclass(frozen=True)
class FaceState:
    site_position_m: np.ndarray
    rotation_w_from_local: np.ndarray
    site_linear_velocity_mps: np.ndarray
    angular_velocity_radps: np.ndarray
    center_position_m: np.ndarray
    normal_w: np.ndarray


@dataclass(frozen=True)
class SweptFaceHit:
    alpha: float
    ball_center_m: np.ndarray
    face_point_m: np.ndarray
    face_point_local_m: np.ndarray
    normal_w: np.ndarray
    face_point_velocity_mps: np.ndarray
    relative_normal_speed_mps: float
    triangle_index: int
    edge_clearance_m: float


@dataclass(frozen=True)
class ShadowGeomMotionBound:
    geom_id: int
    root_rotation_radius_m: float
    hinge_terms: Tuple[Tuple[int, float], ...]
    slide_indices: Tuple[int, ...]


@dataclass(frozen=True)
class GroundContactContract:
    floor_geom_id: int
    foot_body_ids: Tuple[int, int]
    foot_geom_ids: Tuple[int, ...]
    nonfoot_robot_geom_ids: Tuple[int, ...]


@dataclass
class FittedEvents:
    activation_time_s: Optional[float] = None
    paddle_impulse_count: int = 0
    paddle_contact: Optional[Dict[str, Any]] = None
    incoming_table_bounces: int = 0
    return_table_bounces: int = 0
    incoming_table_bounce_times_s: List[float] = field(default_factory=list)
    return_table_bounce_times_s: List[float] = field(default_factory=list)
    table_contacts: List[Dict[str, Any]] = field(default_factory=list)
    net_crossing: Optional[Dict[str, Any]] = None
    first_landing: Optional[Dict[str, Any]] = None
    ball_net_collision: Optional[Dict[str, Any]] = None
    robot_obstacle_contacts: List[Dict[str, Any]] = field(default_factory=list)
    robot_obstacle_contact_count: int = 0
    robot_obstacle_contact_per_obstacle: Dict[str, int] = field(
        default_factory=lambda: {
            name: 0
            for name in table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
        }
    )
    robot_obstacle_swept_steps: int = 0
    robot_obstacle_swept_hit_count: int = 0
    robot_obstacle_swept_per_obstacle: Dict[str, int] = field(
        default_factory=lambda: {
            name: 0
            for name in table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
        }
    )
    robot_obstacle_swept_first_hit: Optional[Dict[str, Any]] = None
    ground_contact_count: int = 0
    legal_foot_support_contact_count: int = 0
    foot_floor_penetration_violation_count: int = 0
    nonfoot_ground_contact_violation_count: int = 0
    ground_contact_violations: List[Dict[str, Any]] = field(
        default_factory=list
    )
    ground_max_foot_penetration_m: float = 0.0
    ground_max_nonfoot_penetration_m: float = 0.0
    self_contacts: List[Dict[str, Any]] = field(default_factory=list)
    joint_limit_violation: Optional[Dict[str, Any]] = None
    fall: Optional[Dict[str, Any]] = None
    native_ball_contact_count: int = 0
    event_order_violations: List[str] = field(default_factory=list)
    ball_forbidden_contacts: List[Dict[str, Any]] = field(default_factory=list)
    shadow_probe_samples: int = 0
    shadow_robot_obstacle_near_contacts: List[Dict[str, Any]] = field(
        default_factory=list
    )
    shadow_self_near_contacts: List[Dict[str, Any]] = field(
        default_factory=list
    )
    shadow_nonfoot_ground_near_contacts: List[Dict[str, Any]] = field(
        default_factory=list
    )
    shadow_foot_floor_penetration_violations: List[
        Dict[str, Any]
    ] = field(default_factory=list)
    ground_shadow_certificate_intervals: int = 0
    ground_shadow_probe_samples: int = 0
    ground_shadow_covered_duration_s: float = 0.0
    ground_shadow_min_nonfoot_lower_bound_m: Optional[float] = None
    ground_shadow_min_foot_lower_bound_m: Optional[float] = None
    ground_shadow_certificate_failure: Optional[str] = None
    shadow_certificate_intervals: int = 0
    shadow_covered_duration_s: float = 0.0
    shadow_max_ball_path_bound_m: float = 0.0
    shadow_max_robot_surface_path_bound_m: float = 0.0


def _finite(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        return native_diag._number(
            value, label, positive=positive, nonnegative=nonnegative
        )
    except native_diag.GateError as exc:
        raise FittedGateError(str(exc)) from exc


def read_pinned_regular_file(
    path: Path, expected_sha256: str, label: str
) -> Tuple[Path, bytes]:
    """Read one regular file through a no-follow descriptor and verify its bytes."""

    try:
        expected = native_diag._require_sha(
            expected_sha256, f"{label} expected SHA"
        )
    except native_diag.GateError as exc:
        raise FittedGateError(str(exc)) from exc
    lexical = path.expanduser()
    if lexical.is_symlink():
        raise FittedGateError(f"{label} must not be a symlink: {lexical}")
    resolved = lexical.resolve()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(resolved), flags)
    except OSError as exc:
        raise FittedGateError(f"cannot open pinned {label}: {resolved}: {exc}") from exc
    chunks: List[bytes] = []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FittedGateError(
                f"{label} must be a regular file: {resolved}"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    actual = native_diag.sha256_bytes(raw)
    if actual != expected:
        raise FittedGateError(
            f"{label} SHA mismatch: expected {expected}, got {actual}"
        )
    return resolved, raw


def load_venue_yaml(path: Path, expected_sha256: str) -> VenueParams:
    path, raw_bytes = read_pinned_regular_file(
        path, expected_sha256, "venue YAML"
    )
    try:
        import yaml
    except ImportError as exc:
        raise FittedGateError("PyYAML is required to read the venue fit") from exc
    try:
        raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise FittedGateError("venue YAML is not UTF-8") from exc
    try:
        ball = raw["ball"]
        flight = raw["flight"]
        table = raw["contact"]["table"]
        paddle = raw["contact"]["paddle"]
    except (KeyError, TypeError) as exc:
        raise FittedGateError(f"venue YAML schema is incomplete: {exc}") from exc
    return VenueParams(
        path=path,
        sha256=expected_sha256,
        ball_mass=_finite(ball["mass"], "ball mass", positive=True),
        ball_radius=_finite(ball["radius"], "ball radius", positive=True),
        inertia_coeff=_finite(
            ball["inertia_coeff"], "ball inertia coefficient", positive=True
        ),
        gravity=_finite(flight["g"], "gravity", positive=True),
        k_d=_finite(flight["k_d"], "drag", nonnegative=True),
        k_m=_finite(flight["k_m"], "Magnus", nonnegative=True),
        table_e=_finite(table["e_eff"], "table restitution", nonnegative=True),
        table_a_t=_finite(table["a_t"], "table a_t", nonnegative=True),
        table_b_t=_finite(table["b_t"], "table b_t"),
        table_mu=_finite(table["mu_safety"], "table mu", nonnegative=True),
        paddle_g1=_finite(paddle["e_exp_g1"], "paddle g1", nonnegative=True),
        paddle_g2=_finite(paddle["e_exp_g2"], "paddle g2"),
        paddle_a_t=_finite(paddle["a_t"], "paddle a_t", nonnegative=True),
        paddle_b_t=_finite(paddle["b_t"], "paddle b_t"),
        paddle_mu=_finite(paddle["mu_safety"], "paddle mu", nonnegative=True),
    )


def validate_clean_checkout(expected_commit: str) -> Dict[str, Any]:
    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or any(char not in "0123456789abcdef" for char in expected_commit)
    ):
        raise FittedGateError(
            "expected clean commit must be an exact lowercase 40-digit Git SHA"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise FittedGateError(
            f"checkout commit mismatch: expected {expected_commit}, got {head}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise FittedGateError("formal fitted Gate requires an exact clean checkout")
    return {"commit": head, "clean": True}


def validate_runtime_execution_attestation(
    attestation: Any,
    *,
    manifest: PhysicalManifest,
    expected_commit: str,
    expected_input_bindings: Mapping[str, Tuple[Path, str]],
    require_all_modules_loaded: bool,
) -> Dict[str, Any]:
    """Validate the stdlib bootstrap's exact raw-byte execution capsule."""

    if not isinstance(attestation, dict):
        raise FittedGateError(
            "formal fitted Gate requires a pinned-byte bootstrap attestation"
        )
    if (
        attestation.get("schema_version") != 2
        or attestation.get("loader")
        != "external_preexec_capsule_pinned_bytes_v1"
        or attestation.get("repository_pyc_used") is not False
        or attestation.get("code_commit") != expected_commit
    ):
        raise FittedGateError(
            "runtime execution attestation header/commit is invalid"
        )
    payload = dict(attestation)
    observed_seal = payload.pop("capsule_sha256", None)
    expected_seal = native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(payload)
    )
    if observed_seal != expected_seal:
        raise FittedGateError("runtime execution capsule seal mismatch")
    external_preexec = attestation.get("external_preexec")
    trust_spec = attestation.get("committed_trust_spec")
    if (
        not isinstance(external_preexec, dict)
        or external_preexec.get("artifact_type")
        != "external_preexec_immutable_launch_capsule_v1"
        or external_preexec.get("capsule_layout")
        != "formal_fitted_ball_retained_capsule_v1"
        or external_preexec.get("code_commit") != expected_commit
        or external_preexec.get("materializer_source")
        != "external_git_show_stdin"
        or external_preexec.get("fresh_detached_worktree") is not True
        or external_preexec.get("checkout_read_only_before_exec") is not True
        or not isinstance(external_preexec.get("source_repo"), str)
        or not isinstance(
            external_preexec.get("capsule_staging_root"), str
        )
        or not isinstance(external_preexec.get("checkout_root"), str)
        or not isinstance(external_preexec.get("artifacts_root"), str)
        or Path(str(external_preexec.get("checkout_root"))).resolve()
        != REPO_ROOT.resolve()
        or Path(str(external_preexec.get("checkout_root"))).parent
        != Path(
            str(external_preexec.get("capsule_staging_root"))
        )
        or Path(str(external_preexec.get("artifacts_root"))).parent
        != Path(
            str(external_preexec.get("capsule_staging_root"))
        )
        or Path(str(external_preexec.get("checkout_root"))).name
        != "checkout"
        or Path(str(external_preexec.get("artifacts_root"))).name
        != "artifacts"
        or not isinstance(external_preexec.get("capsule_sha256"), str)
        or not isinstance(trust_spec, dict)
        or trust_spec.get("repo_path")
        != (
            "configs/"
            "mujoco_fitted_ball_pre_registered_launch_v2.json"
        )
        or not isinstance(trust_spec.get("sha256"), str)
    ):
        raise FittedGateError(
            "external pre-exec capsule/committed trust authority is invalid"
        )
    authorization = trust_spec.get("authorization")
    bindings = trust_spec.get("bindings")
    runtime_environment = trust_spec.get("runtime_environment")
    if (
        not isinstance(authorization, dict)
        or authorization.get("formal_simulation_authorized") is not True
        or authorization.get("hardware_authorized") is not False
        or authorization.get("registered_before_gate_run") is not True
        or not authorization.get("decision_id")
        or not authorization.get("human_dri")
    ):
        raise FittedGateError(
            "committed trust spec lacks simulation-only preregistration"
        )
    if (
        not isinstance(bindings, dict)
        or set(bindings)
        != set(expected_input_bindings) | {"bootstrap"}
    ):
        raise FittedGateError("committed trust input binding set is not exact")
    for key, (expected_path, expected_sha256) in expected_input_bindings.items():
        row = bindings.get(key)
        if (
            not isinstance(row, dict)
            or set(row) != {"repo_path", "sha256"}
            or row.get("sha256") != expected_sha256
            or (REPO_ROOT / str(row.get("repo_path", ""))).resolve()
            != expected_path.resolve()
        ):
            raise FittedGateError(
                f"committed trust input binding drifted: {key}"
            )
    expected_execution_sources = native_diag._mapping(
        manifest.contract["runtime_execution_source_sha256"],
        "runtime execution source pins",
    )
    bootstrap_binding = bindings.get("bootstrap")
    if (
        not isinstance(bootstrap_binding, dict)
        or set(bootstrap_binding) != {"repo_path", "sha256"}
        or bootstrap_binding.get("repo_path") != BOOTSTRAP_REPO_PATH
        or bootstrap_binding.get("sha256")
        != expected_execution_sources[BOOTSTRAP_REPO_PATH]
        or external_preexec.get("bootstrap_sha256")
        != bootstrap_binding.get("sha256")
        or (REPO_ROOT / BOOTSTRAP_REPO_PATH).resolve()
        != RUNTIME_EXECUTION_SOURCE_PATHS[BOOTSTRAP_REPO_PATH]
    ):
        raise FittedGateError(
            "committed trust bootstrap binding is not the manifest-pinned "
            "external launcher"
        )
    if (
        not isinstance(runtime_environment, dict)
        or set(runtime_environment)
        != {
            "python_executable_sha256",
            "git_executable_sha256",
            "python_version",
            "python_cache_tag",
            "python_import_roots",
            "required_distributions",
        }
    ):
        raise FittedGateError(
            "committed trust runtime environment binding is not exact"
        )

    expected_sources = expected_execution_sources
    expected_data = native_diag._mapping(
        manifest.contract["runtime_execution_data_sha256"],
        "runtime execution data pins",
    )
    source_rows = attestation.get("sources")
    data_rows = attestation.get("data")
    if not isinstance(source_rows, list) or not isinstance(data_rows, list):
        raise FittedGateError("runtime execution capsule rows are malformed")

    def validate_rows(
        rows: Sequence[Any],
        expected: Mapping[str, Any],
        label: str,
    ) -> Dict[str, Dict[str, Any]]:
        by_path: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise FittedGateError(
                    f"runtime execution {label} row is not an object"
                )
            path = row.get("repo_path")
            if not isinstance(path, str) or path in by_path:
                raise FittedGateError(
                    f"runtime execution {label} path is invalid/duplicate"
                )
            by_path[path] = row
        if set(by_path) != set(expected):
            raise FittedGateError(
                f"runtime execution {label} closure is not exact"
            )
        for path, row in by_path.items():
            pinned = native_diag._require_sha(
                expected[path], f"{label} pin {path}"
            )
            if (
                row.get("expected_sha256") != pinned
                or row.get("executed_sha256") != pinned
                or row.get("git_blob_sha256") != pinned
                or row.get("symlink_free") is not True
                or not isinstance(row.get("size_bytes"), int)
                or int(row["size_bytes"]) <= 0
            ):
                raise FittedGateError(
                    f"runtime execution {label} bytes are not triply bound: "
                    f"{path}"
                )
        return by_path

    source_by_path = validate_rows(
        source_rows, expected_sources, "source"
    )
    validate_rows(data_rows, expected_data, "data")
    python = attestation.get("python")
    git = attestation.get("git")
    if (
        not isinstance(python, dict)
        or python.get("isolated") is not True
        or python.get("ignore_environment") is not True
        or python.get("no_user_site") is not True
        or python.get("no_site") is not True
        or python.get("safe_path") is not True
        or python.get("dont_write_bytecode") is not True
        or not isinstance(python.get("executable"), str)
        or not isinstance(python.get("executable_sha256"), str)
        or len(python["executable_sha256"]) != 64
        or python.get("executable_sha256")
        != runtime_environment["python_executable_sha256"]
        or python.get("version") != runtime_environment["python_version"]
        or python.get("cache_tag")
        != runtime_environment["python_cache_tag"]
        or not isinstance(python.get("initial_sys_path"), list)
        or not all(
            isinstance(path, str) for path in python["initial_sys_path"]
        )
        or not isinstance(git, dict)
        or git.get("executable_sha256")
        != runtime_environment["git_executable_sha256"]
    ):
        raise FittedGateError(
            "bootstrap Python executable/environment is not isolated and bound"
        )
    dependencies = attestation.get("external_python_dependencies")
    if (
        not isinstance(dependencies, dict)
        or dependencies.get("authority")
        != "committed_symlink_free_dependency_tree_v1"
        or dependencies.get("site_module_executed") is not False
        or dependencies.get("pth_files_executed") is not False
        or dependencies.get("installed_directly_on_sys_path") is not True
    ):
        raise FittedGateError(
            "external NumPy/MuJoCo dependency authority is invalid"
        )
    import_roots = dependencies.get("import_roots")
    expected_import_roots = runtime_environment.get(
        "python_import_roots"
    )
    if (
        not isinstance(import_roots, list)
        or not isinstance(expected_import_roots, list)
        or len(import_roots) != len(expected_import_roots)
    ):
        raise FittedGateError(
            "preregistered Python import-root closure is malformed"
        )
    for actual, expected in zip(import_roots, expected_import_roots):
        if (
            not isinstance(actual, dict)
            or not isinstance(expected, dict)
            or actual.get("path") != expected.get("path")
            or actual.get("tree_sha256") != expected.get("tree_sha256")
            or actual.get("symlink_free") is not True
            or not isinstance(actual.get("file_count"), int)
            or actual["file_count"] <= 0
            or not isinstance(actual.get("total_size_bytes"), int)
            or actual["total_size_bytes"] <= 0
        ):
            raise FittedGateError(
                "preregistered Python import-root bytes are not exact"
            )
    expected_sys_path = list(python["initial_sys_path"]) + [
        str(row["path"]) for row in import_roots
    ]
    if list(sys.path) != expected_sys_path:
        raise FittedGateError(
            "Python sys.path differs from stdlib plus preregistered roots"
        )
    distribution_rows = dependencies.get("required_distributions")
    expected_distributions = runtime_environment.get(
        "required_distributions"
    )
    if (
        not isinstance(distribution_rows, dict)
        or not isinstance(expected_distributions, dict)
        or set(distribution_rows) != set(REQUIRED_EXTERNAL_DISTRIBUTIONS)
        or set(expected_distributions)
        != set(REQUIRED_EXTERNAL_DISTRIBUTIONS)
    ):
        raise FittedGateError(
            "required external distribution set is not exact"
        )
    for name in REQUIRED_EXTERNAL_DISTRIBUTIONS:
        actual = distribution_rows[name]
        expected = expected_distributions[name]
        expected_package_path = (
            Path(str(expected["import_root"]))
            / str(expected["package_subpath"])
        ).resolve()
        if (
            not isinstance(actual, dict)
            or actual.get("import_name") != name
            or actual.get("expected_version") != expected.get("version")
            or actual.get("import_root") != expected.get("import_root")
            or actual.get("package_subpath")
            != expected.get("package_subpath")
            or actual.get("tree_sha256") != expected.get("tree_sha256")
            or Path(str(actual.get("path", ""))).resolve()
            != expected_package_path
            or actual.get("symlink_free") is not True
        ):
            raise FittedGateError(
                f"required distribution bytes are not preregistered: {name}"
            )
        module = sys.modules.get(name)
        if module is None:
            if require_all_modules_loaded:
                raise FittedGateError(
                    f"required preregistered distribution was never loaded: "
                    f"{name}"
                )
            continue
        module_file = getattr(module, "__file__", None)
        try:
            if not isinstance(module_file, str):
                raise ValueError
            Path(module_file).resolve().relative_to(
                expected_package_path
            )
        except ValueError as exc:
            raise FittedGateError(
                f"required distribution loaded outside preregistered tree: "
                f"{name}"
            ) from exc
        if str(getattr(module, "__version__", "")) != str(
            expected["version"]
        ):
            raise FittedGateError(
                f"required distribution version drifted: {name}"
            )
    module_bindings = attestation.get("module_bindings")
    if not isinstance(module_bindings, dict) or not module_bindings:
        raise FittedGateError("runtime module binding map is missing")
    capsule_id = str(observed_seal)
    for module_name, repo_path in module_bindings.items():
        if (
            not isinstance(module_name, str)
            or repo_path not in source_by_path
        ):
            raise FittedGateError(
                "runtime module binding references an unpinned source"
            )
        module = sys.modules.get(module_name)
        if module is None:
            if require_all_modules_loaded:
                raise FittedGateError(
                    f"pinned runtime module was never loaded: {module_name}"
                )
            continue
        if (
            getattr(module, "__pinned_capsule_id__", None) != capsule_id
            or getattr(module, "__pinned_executed_sha256__", None)
            != source_by_path[repo_path]["executed_sha256"]
            or getattr(module, "__cached__", None) is not None
            or getattr(getattr(module, "__spec__", None), "origin", None)
            != str(RUNTIME_EXECUTION_SOURCE_PATHS[repo_path])
        ):
            raise FittedGateError(
                f"runtime module execution identity drifted: {module_name}"
            )
    if require_all_modules_loaded:
        consumed_data = globals().get(
            "RUNTIME_EXECUTION_DATA_CONSUMPTION"
        )
        if (
            not isinstance(consumed_data, set)
            or consumed_data != set(expected_data)
        ):
            raise FittedGateError(
                "runtime did not consume the exact pinned data closure"
            )
    return {
        "capsule_sha256": observed_seal,
        "source_count": len(source_rows),
        "data_count": len(data_rows),
        "module_binding_count": len(module_bindings),
        "all_modules_loaded": require_all_modules_loaded,
        "python": python,
        "git": git,
        "external_python_dependencies": dependencies,
        "external_preexec": external_preexec,
        "committed_trust_spec": trust_spec,
    }


def validate_profile_sources_at_commit(
    profile: Mapping[str, Any], expected_commit: str
) -> Dict[str, Any]:
    source_map = native_diag._mapping(
        profile.get("solver_implementation_source_sha256"),
        "profile solver implementation source map",
    )
    source_authority = native_diag._mapping(
        profile.get("source_authority"), "profile source authority"
    )
    source_blob_map_sha256 = native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(dict(source_map))
    )
    if (
        source_authority.get("authority")
        != "external_exact_commit_subset_blob_map_v1"
        or source_authority.get("commit_binding")
        != "external_preexec_immutable_launch_capsule_v1"
        or source_authority.get("embedded_commit") is not False
        or source_authority.get("source_blob_map_sha256")
        != source_blob_map_sha256
    ):
        raise FittedGateError(
            "profile source authority is not the exact external-commit "
            "subset blob map"
        )
    base = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    verified: Dict[str, str] = {}
    for name, expected_sha in source_map.items():
        repo_path = f"{base}/{name}"
        raw = subprocess.run(
            ["git", "show", f"{expected_commit}:{repo_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha:
            raise FittedGateError(
                f"profile source pin does not match commit blob: {name}"
            )
        verified[repo_path] = actual
    return {
        "authority": source_authority["authority"],
        "external_code_commit": expected_commit,
        "source_blob_map_sha256": source_blob_map_sha256,
        "verified_blobs": dict(sorted(verified.items())),
    }


def validate_recorded_position_venue_fit_raw_input(
    raw_input: Mapping[str, Any],
    *,
    action: native_diag.ActionSpec,
    source_receipt: Mapping[str, Any],
    expected_launch_state: Mapping[str, Any],
) -> None:
    """Reject claims beyond recorded position, fitted speed, and zero spin."""

    expected_keys = {
        "schema_version",
        "artifact_type",
        "action_id",
        "action_uid",
        "motion_sha256",
        "coordinate_frame",
        "units",
        "recorded_sample",
        "venue_fit",
        "birth_solution",
        "spin_assumption",
        "provenance",
    }
    provenance = {
        "position": "recorded",
        "velocity": "venue_fit_not_measured",
        "spin": "assumed_zero_not_measured",
        "measured_velocity_used": False,
        "measured_spin_used": False,
    }
    if (
        set(raw_input) != expected_keys
        or raw_input.get("schema_version") != 1
        or raw_input.get("artifact_type")
        != "recorded_position_venue_fit_input_v1"
        or raw_input.get("action_id") != action.action_id
        or raw_input.get("action_uid") != action.action_uid
        or raw_input.get("motion_sha256") != action.motion_sha256
        or raw_input.get("coordinate_frame") != "mujoco_world"
        or raw_input.get("units")
        != {
            "position": "m",
            "velocity": "m/s",
            "spin": "rad/s",
            "time": "s",
        }
        or raw_input.get("recorded_sample")
        != source_receipt.get("recorded_sample")
        or raw_input.get("venue_fit")
        != source_receipt.get("venue_fit")
        or raw_input.get("birth_solution")
        != source_receipt.get("birth_solution")
        or raw_input.get("spin_assumption")
        != source_receipt.get("spin_assumption")
        or raw_input.get("provenance") != provenance
    ):
        raise FittedGateError(
            f"{action.action_id}: recorded-position venue-fit raw input "
            "schema/provenance is not exact"
        )
    birth = native_diag._mapping(
        raw_input["birth_solution"],
        f"{action.action_id}.raw birth solution",
    )
    spin = native_diag._mapping(
        raw_input["spin_assumption"],
        f"{action.action_id}.raw spin assumption",
    )
    if (
        birth.get("activation_time_s")
        != expected_launch_state["activation_time_s"]
        or birth.get("position_w_m")
        != expected_launch_state["position_w_m"]
        or birth.get("velocity_w_mps")
        != expected_launch_state["velocity_w_mps"]
        or spin.get("spin_w_radps")
        != expected_launch_state["spin_w_radps"]
    ):
        raise FittedGateError(
            f"{action.action_id}: venue-fit raw input birth/spin differs "
            "from the manifest launch"
        )


def validate_launch_evidence_trust_root(
    *,
    path: Path,
    expected_sha256: str,
    manifest_sha256: str,
    expected_commit: str,
    manifest: PhysicalManifest,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Validate a separately pinned, Git-bound launch-evidence authority.

    A semantic upstream receipt is still self-authored unless an independent
    preregistration ledger binds it.  This root binds every outer artifact,
    upstream receipt, raw recording/solver input, and validator source to the
    exact manifest and clean commit.
    """

    try:
        root, root_receipt = native_diag.read_json_exact(
            path,
            "launch evidence trust root",
            expected_sha256=expected_sha256,
        )
    except native_diag.GateError as exc:
        raise FittedGateError(str(exc)) from exc
    root_path = Path(root_receipt["path"]).resolve()
    sealed = dict(root)
    observed_seal = sealed.pop("receipt_payload_sha256", None)
    expected_seal = native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(sealed)
    )
    exact_root_keys = {
        "schema_version",
        "artifact_type",
        "manifest_sha256",
        "commit_binding",
        "action_order",
        "authorization",
        "pre_registration",
        "entries",
        "receipt_payload_sha256",
    }
    if (
        set(root) != exact_root_keys
        or native_diag._integer(
            root.get("schema_version"), "launch trust root schema"
        )
        != 1
        or root.get("artifact_type")
        != "pre_registered_launch_evidence_trust_root_v1"
        or root.get("manifest_sha256") != manifest_sha256
        or root.get("action_order") != list(manifest.base.action_order)
        or observed_seal != expected_seal
    ):
        raise FittedGateError(
            "launch evidence trust root schema/manifest/seal mismatch"
        )
    commit_binding = native_diag._mapping(
        root.get("commit_binding"), "launch trust commit binding"
    )
    if (
        set(commit_binding)
        != {
            "schema_version",
            "authority",
            "embedded_commit",
            "validator_subset_blob_map_sha256",
        }
        or commit_binding.get("schema_version") != 1
        or commit_binding.get("authority")
        != "external_preexec_exact_commit_subset_blob_map_v1"
        or commit_binding.get("embedded_commit") is not False
    ):
        raise FittedGateError(
            "launch trust root must use external exact-commit subset-blob "
            "authority without an embedded self-referential commit"
        )
    expected_validator_subset_sha = native_diag._require_sha(
        commit_binding.get("validator_subset_blob_map_sha256"),
        "launch trust validator subset blob-map SHA",
    )
    authorization = native_diag._mapping(
        root.get("authorization"), "launch trust authorization"
    )
    preregistration = native_diag._mapping(
        root.get("pre_registration"), "launch trust pre-registration"
    )
    if (
        authorization.get("physical_gate_input_authorized") is not True
        or authorization.get("hardware_authorized") is not False
        or preregistration.get("registered_before_gate_run") is not True
        or not isinstance(preregistration.get("decision_id"), str)
        or not preregistration["decision_id"]
        or not isinstance(preregistration.get("human_dri"), str)
        or not preregistration["human_dri"]
    ):
        raise FittedGateError(
            "launch trust root lacks independent simulation-only preregistration"
        )
    entries = root.get("entries")
    if (
        not isinstance(entries, list)
        or [row.get("action_id") for row in entries if isinstance(row, dict)]
        != list(manifest.base.action_order)
    ):
        raise FittedGateError(
            "launch trust entries must follow the exact action order"
        )
    action_rows = {
        str(row["action_id"]): row for row in manifest.raw["actions"]
    }
    pinned_files: List[Dict[str, Any]] = [
        {
            "role": "launch_trust_root",
            "path": str(root_path),
            "sha256": expected_sha256,
            "size_bytes": int(root_receipt["size_bytes"]),
        }
    ]
    entry_receipts: List[Dict[str, Any]] = []
    validator_subset_rows: List[Dict[str, str]] = []
    exact_entry_keys = {
        "action_id",
        "action_uid",
        "motion_sha256",
        "source",
        "source_artifact_sha256",
        "upstream_evidence_path",
        "upstream_evidence_sha256",
        "raw_input_path",
        "raw_input_sha256",
        "validator_source_path",
        "validator_source_sha256",
    }
    for action, entry in zip(manifest.base.actions, entries):
        if not isinstance(entry, dict) or set(entry) != exact_entry_keys:
            raise FittedGateError(
                f"{action.action_id}: launch trust entry key set is not exact"
            )
        launch = manifest.launches[action.action_id]
        source_receipt = manifest.launch_source_receipts[action.action_id]
        upstream_receipt = native_diag._mapping(
            source_receipt.get("upstream_evidence"),
            f"{action.action_id}.trusted upstream receipt",
        )
        if (
            entry.get("action_id") != action.action_id
            or entry.get("action_uid") != action.action_uid
            or entry.get("motion_sha256") != action.motion_sha256
            or entry.get("source") != launch.source
            or entry.get("source_artifact_sha256")
            != launch.source_artifact_sha256
            or entry.get("upstream_evidence_sha256")
            != upstream_receipt["sha256"]
        ):
            raise FittedGateError(
                f"{action.action_id}: launch trust identity/hash mismatch"
            )
        upstream_path = native_diag._resolve_repo_file(
            entry.get("upstream_evidence_path"),
            f"{action.action_id}.trusted upstream path",
        )
        if str(upstream_path) != str(upstream_receipt["path"]):
            raise FittedGateError(
                f"{action.action_id}: trusted upstream path mismatch"
            )
        raw_input_path = native_diag._resolve_repo_file(
            entry.get("raw_input_path"),
            f"{action.action_id}.launch raw input path",
        )
        raw_input_sha = native_diag._require_sha(
            entry.get("raw_input_sha256"),
            f"{action.action_id}.launch raw input SHA",
        )
        validator_path = native_diag._resolve_repo_file(
            entry.get("validator_source_path"),
            f"{action.action_id}.launch validator path",
        )
        validator_sha = native_diag._require_sha(
            entry.get("validator_source_sha256"),
            f"{action.action_id}.launch validator SHA",
        )
        if len({root_path, upstream_path, raw_input_path, validator_path}) != 4:
            raise FittedGateError(
                f"{action.action_id}: launch trust roles must be distinct files"
            )
        _raw_path, raw_input_bytes = read_pinned_regular_file(
            raw_input_path,
            raw_input_sha,
            f"{action.action_id} launch raw input",
        )
        _validator_path, validator_bytes = read_pinned_regular_file(
            validator_path,
            validator_sha,
            f"{action.action_id} launch validator",
        )
        try:
            validator_repo_path = validator_path.relative_to(
                REPO_ROOT
            ).as_posix()
        except ValueError as exc:
            raise FittedGateError(
                f"{action.action_id}: launch validator must be tracked"
            ) from exc
        validator_blob = subprocess.run(
            ["git", "show", f"{expected_commit}:{validator_repo_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if (
            validator_blob != validator_bytes
            or hashlib.sha256(validator_blob).hexdigest() != validator_sha
        ):
            raise FittedGateError(
                f"{action.action_id}: launch validator is not the commit blob"
            )
        validator_subset_rows.append(
            {
                "action_id": action.action_id,
                "repo_path": validator_repo_path,
                "sha256": validator_sha,
            }
        )
        try:
            raw_input = json.loads(
                raw_input_bytes.decode("utf-8"),
                object_pairs_hook=native_diag._reject_duplicate_keys,
                parse_constant=native_diag._reject_nonfinite_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FittedGateError(
                f"{action.action_id}: launch raw input must be exact JSON"
            ) from exc
        if not isinstance(raw_input, dict):
            raise FittedGateError(
                f"{action.action_id}: launch raw input must be an object"
            )
        launch_row = native_diag._mapping(
            action_rows[action.action_id]["physical_ball_launch"],
            f"{action.action_id}.physical_ball_launch",
        )
        expected_state = {
            key: launch_row[key]
            for key in (
                "source",
                "activation_time_s",
                "position_w_m",
                "velocity_w_mps",
                "spin_w_radps",
                "required_incoming_table_bounces",
            )
        }
        common_valid = (
            raw_input.get("schema_version") == 1
            and raw_input.get("action_id") == action.action_id
            and raw_input.get("action_uid") == action.action_uid
            and raw_input.get("motion_sha256") == action.motion_sha256
            and raw_input.get("coordinate_frame") == "mujoco_world"
            and raw_input.get("units")
            == {
                "position": "m",
                "velocity": "m/s",
                "spin": "rad/s",
                "time": "s",
            }
        )
        if launch.source == "recorded_pre_hit_state_v1":
            sample_index = int(source_receipt["recording_sample_index"])
            samples = raw_input.get("samples")
            valid_source = (
                raw_input.get("artifact_type")
                == "recorded_ball_capture_v1"
                and isinstance(samples, list)
                and 0 <= sample_index < len(samples)
                and isinstance(samples[sample_index], dict)
                and native_diag.canonical_json_bytes(
                    samples[sample_index]
                )
                == native_diag.canonical_json_bytes(
                    {
                        "sample_time_s": expected_state[
                            "activation_time_s"
                        ],
                        "position_w_m": expected_state["position_w_m"],
                        "velocity_w_mps": expected_state[
                            "velocity_w_mps"
                        ],
                        "spin_w_radps": expected_state["spin_w_radps"],
                    }
                )
            )
        elif (
            launch.source
            == RECORDED_POSITION_VENUE_FIT_ZERO_SPIN_SOURCE
        ):
            validate_recorded_position_venue_fit_raw_input(
                raw_input,
                action=action,
                source_receipt=source_receipt,
                expected_launch_state=expected_state,
            )
            valid_source = True
        else:
            valid_source = (
                raw_input.get("artifact_type")
                == "native_shooting_solver_input_v1"
                and raw_input.get("pre_registered") is True
                and raw_input.get("launch_state") == expected_state
                and raw_input_sha
                == source_receipt["shooting_solver_input_sha256"]
            )
        if not common_valid or not valid_source:
            raise FittedGateError(
                f"{action.action_id}: independent raw launch input does not "
                "derive the exact manifest launch"
            )
        pinned_files.extend(
            (
                {
                    "role": f"launch_raw_input:{action.action_id}",
                    "path": str(raw_input_path),
                    "sha256": raw_input_sha,
                    "size_bytes": len(raw_input_bytes),
                },
                {
                    "role": f"launch_validator:{action.action_id}",
                    "path": str(validator_path),
                    "sha256": validator_sha,
                    "size_bytes": len(validator_bytes),
                },
            )
        )
        entry_receipts.append(
            {
                "action_id": action.action_id,
                "source": launch.source,
                "source_artifact_sha256": launch.source_artifact_sha256,
                "upstream_evidence_sha256": upstream_receipt["sha256"],
                "raw_input_sha256": raw_input_sha,
                "validator_source_sha256": validator_sha,
            }
        )
    observed_validator_subset_sha = native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(validator_subset_rows)
    )
    if observed_validator_subset_sha != expected_validator_subset_sha:
        raise FittedGateError(
            "launch trust validator subset blob-map seal mismatch"
        )
    return (
        {
            "root": root_receipt,
            "receipt_payload_sha256": observed_seal,
            "pre_registration": preregistration,
            "commit_binding": {
                **dict(commit_binding),
                "external_code_commit": expected_commit,
                "validated_subset_blob_map_sha256": (
                    observed_validator_subset_sha
                ),
            },
            "entries": entry_receipts,
        },
        pinned_files,
    )


def venue_fit_launch_runtime_pins(
    manifest: PhysicalManifest,
) -> List[Dict[str, str]]:
    """Return source files consumed by the fitted-contact/birth derivation.

    The launch source/upstream/raw-input receipts bind the embedded numeric
    inputs, but they do not by themselves keep the two solver source files
    immutable for the duration of the MuJoCo replay.  Expose those files as
    ordinary runtime pins so the post-runtime stability check covers them.
    """

    output: List[Dict[str, str]] = []
    for action in manifest.base.actions:
        action_id = action.action_id
        launch = manifest.launches[action_id]
        if (
            launch.source
            != RECORDED_POSITION_VENUE_FIT_ZERO_SPIN_SOURCE
        ):
            continue
        receipt = native_diag._mapping(
            manifest.launch_source_receipts[action_id],
            f"{action_id}.launch source receipt",
        )
        venue_fit = native_diag._mapping(
            receipt.get("venue_fit"),
            f"{action_id}.venue_fit runtime pin",
        )
        birth_solution = native_diag._mapping(
            receipt.get("birth_solution"),
            f"{action_id}.birth_solution runtime pin",
        )
        for role, row, path_key, sha_key in (
            (
                f"launch_fit_venue:{action_id}",
                venue_fit,
                "venue_yaml_path",
                "venue_yaml_sha256",
            ),
            (
                f"launch_fit_solver_source:{action_id}",
                venue_fit,
                "solver_source_path",
                "solver_source_sha256",
            ),
            (
                f"launch_birth_solver_source:{action_id}",
                birth_solution,
                "solver_source_path",
                "solver_source_sha256",
            ),
        ):
            path = native_diag._resolve_repo_file(
                row.get(path_key), f"{role}.path"
            )
            digest = native_diag._require_sha(
                row.get(sha_key), f"{role}.sha256"
            )
            output.append(
                {
                    "role": role,
                    "path": str(path),
                    "sha256": digest,
                }
            )
    return output


def capture_runtime_input_snapshot(
    *,
    args: argparse.Namespace,
    manifest: PhysicalManifest,
    profile: Mapping[str, Any],
    venue: VenueParams,
) -> List[Dict[str, Any]]:
    """Capture every runtime-consumed, externally pinned file before execution.

    The snapshot is checked again immediately before the receipt is sealed.
    Files used for simulation are additionally consumed from one hash-checked
    byte read where practical (MJCF, venue YAML, motion NPZ, and face STL).
    """

    entries: Dict[str, Dict[str, Any]] = {}

    def add(role: str, path: Path, expected_sha256: str) -> None:
        expected = str(expected_sha256)
        resolved, raw = read_pinned_regular_file(path, expected, role)
        key = str(resolved)
        prior = entries.get(key)
        if prior is not None:
            if prior["sha256"] != expected:
                raise FittedGateError(
                    f"conflicting expected SHAs for runtime input {resolved}"
                )
            prior["roles"].append(role)
            return
        entries[key] = {
            "path": key,
            "sha256": expected,
            "size_bytes": len(raw),
            "roles": [role],
        }

    add(
        "strict_training_manifest",
        args.training_manifest,
        args.training_manifest_sha256,
    )
    add(
        "physical_gate_manifest",
        args.physical_gate_manifest,
        args.physical_gate_manifest_sha256,
    )
    add(
        "physical_gate_materialization_receipt",
        args.physical_gate_materialization_receipt,
        args.physical_gate_materialization_receipt_sha256,
    )
    bundle_pin = getattr(args, "_physical_task_bundle_pin", None)
    if isinstance(bundle_pin, Mapping):
        add(
            "physical_task_bundle",
            Path(str(bundle_pin["path"])),
            str(bundle_pin["sha256"]),
        )
    add("profile_pins", args.profile_pins, args.profile_pins_sha256)
    for row in getattr(args, "_launch_trust_pinned_files", ()):
        add(str(row["role"]), Path(str(row["path"])), str(row["sha256"]))
    add(
        "mujoco_identity_manifest",
        CANONICAL_IDENTITY_MANIFEST,
        CANONICAL_IDENTITY_MANIFEST_SHA256,
    )
    identity, _ = native_diag.read_json_exact(
        CANONICAL_IDENTITY_MANIFEST,
        "MuJoCo identity snapshot",
        expected_sha256=CANONICAL_IDENTITY_MANIFEST_SHA256,
    )
    identity_expected = native_diag._mapping(
        identity.get("expected"), "identity.expected"
    )
    add(
        "vendor_root_mjcf",
        CANONICAL_MJCF,
        native_diag._require_sha(
            identity_expected.get("root_mjcf_sha256"),
            "identity root MJCF SHA",
        ),
    )
    add("venue_yaml", venue.path, venue.sha256)
    add("fitted_contact_model", CONTACT_MODEL_PATH, CONTACT_MODEL_SHA256)
    runtime_source_pins = native_diag._mapping(
        manifest.contract["runtime_source_sha256"],
        "physical contract runtime source pins",
    )
    for name, path in RUNTIME_SOURCE_PATHS.items():
        add(
            f"runtime_source:{name}",
            path,
            str(runtime_source_pins[name]),
        )
    execution_source_pins = native_diag._mapping(
        manifest.contract["runtime_execution_source_sha256"],
        "physical contract runtime execution source pins",
    )
    for name, path in RUNTIME_EXECUTION_SOURCE_PATHS.items():
        add(
            f"runtime_execution_source:{name}",
            path,
            str(execution_source_pins[name]),
        )
    execution_data_pins = native_diag._mapping(
        manifest.contract["runtime_execution_data_sha256"],
        "physical contract runtime execution data pins",
    )
    for name, path in RUNTIME_EXECUTION_DATA_PATHS.items():
        add(
            f"runtime_execution_data:{name}",
            path,
            str(execution_data_pins[name]),
        )
    geometry = manifest.base.racket_geometry_contract
    add(
        "racket_geometry_production_source",
        Path(str(geometry["source_path"])),
        str(geometry["source_sha256"]),
    )

    scene_paths = {
        "scripts/mujoco_table_scene.py": REPO_ROOT
        / "scripts/mujoco_table_scene.py",
        "scripts/audit_motion_schema2_table_net_clearance.py": (
            REPO_ROOT
            / "scripts/audit_motion_schema2_table_net_clearance.py"
        ),
        "table_tennis/geometry.py": (
            REPO_ROOT
            / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/table_tennis/geometry.py"
        ),
        "table_tennis/table_frame.py": (
            REPO_ROOT
            / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/table_tennis/table_frame.py"
        ),
    }
    scene_pins = native_diag._mapping(
        manifest.contract["scene_source_sha256"],
        "physical contract scene source pins",
    )
    for name, path in scene_paths.items():
        add(f"scene_source:{name}", path, str(scene_pins[name]))

    face_pins = native_diag._mapping(
        manifest.contract["selected_face_mesh_sha256"],
        "physical contract selected-face mesh pins",
    )
    for name, sign in FACE_MESH_PIN_KEYS.items():
        add(
            f"selected_face_mesh:{name}",
            FACE_MESH_PATHS[sign],
            str(face_pins[name]),
        )

    solver_source_dir = (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    for name, digest in profile[
        "solver_implementation_source_sha256"
    ].items():
        add(f"solver_source:{name}", solver_source_dir / name, str(digest))

    for action, row in zip(manifest.base.actions, manifest.raw["actions"]):
        action_id = action.action_id
        add(
            f"motion:{action_id}",
            action.motion_path,
            action.motion_sha256,
        )
        task_binding = manifest.task_bindings[action_id]
        add(
            f"solver_execution_receipt:{action_id}",
            task_binding.solver_execution_receipt_path,
            task_binding.solver_execution_receipt_sha256,
        )
        launch = manifest.launches[action_id]
        add(
            f"launch_source:{action_id}",
            launch.source_artifact_path,
            launch.source_artifact_sha256,
        )
        upstream = manifest.launch_source_receipts[action_id][
            "upstream_evidence"
        ]
        add(
            f"launch_upstream:{action_id}",
            Path(str(upstream["path"])),
            str(upstream["sha256"]),
        )
        admission = native_diag._mapping(
            row["admission"], f"{action_id}.admission"
        )
        for role in (
            "registry_entry",
            "compiler_manifest",
            "bank_gate_report",
        ):
            path = native_diag._resolve_repo_file(
                admission[f"{role}_path"],
                f"{action_id}.admission.{role}_path",
            )
            add(
                f"admission:{action_id}:{role}",
                path,
                str(admission[f"{role}_sha256"]),
            )
    for pin in venue_fit_launch_runtime_pins(manifest):
        add(pin["role"], Path(pin["path"]), pin["sha256"])

    output = sorted(entries.values(), key=lambda row: row["path"])
    for row in output:
        row["roles"].sort()
    return output


def assert_runtime_input_snapshot_stable(
    snapshot: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    for row in snapshot:
        path = Path(str(row["path"]))
        read_pinned_regular_file(
            path,
            str(row["sha256"]),
            f"post-runtime input {path}",
        )
    return {
        "stable": True,
        "checked_files": len(snapshot),
        "check": "pinned_sha256_before_and_after_runtime",
    }


def extend_snapshot_with_mujoco_source_closure(
    snapshot: Sequence[Mapping[str, Any]],
    *,
    model_root: Path,
    source_closure: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {
        str(row["path"]): {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
            "roles": list(row["roles"]),
        }
        for row in snapshot
    }
    members = source_closure.get("members")
    if not isinstance(members, (list, tuple)):
        raise FittedGateError("verified MuJoCo source closure lacks members")
    root = model_root.expanduser().resolve()
    for member in members:
        if not isinstance(member, Mapping):
            raise FittedGateError("MuJoCo source-closure member is malformed")
        relative = Path(str(member.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise FittedGateError(
                f"MuJoCo source-closure member path is unsafe: {relative}"
            )
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FittedGateError(
                f"MuJoCo source-closure member escapes model root: {relative}"
            ) from exc
        expected = str(member.get("sha256", ""))
        resolved, raw = read_pinned_regular_file(
            candidate,
            expected,
            f"MuJoCo source closure:{relative.as_posix()}",
        )
        key = str(resolved)
        role = f"mujoco_source_closure:{relative.as_posix()}"
        if key in entries:
            if entries[key]["sha256"] != expected:
                raise FittedGateError(
                    f"conflicting source-closure SHA for {resolved}"
                )
            entries[key]["roles"].append(role)
        else:
            entries[key] = {
                "path": key,
                "sha256": expected,
                "size_bytes": len(raw),
                "roles": [role],
            }
    output = sorted(entries.values(), key=lambda row: row["path"])
    for row in output:
        row["roles"] = sorted(set(row["roles"]))
    return output


def load_motion_from_pinned_bytes(
    action: native_diag.ActionSpec,
) -> motion_player.MotionClip:
    """Load a motion from the exact byte read that satisfied its manifest pin."""

    _path, raw = read_pinned_regular_file(
        action.motion_path,
        action.motion_sha256,
        f"{action.action_id} motion",
    )
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="fitted_gate_motion_", suffix=".npz", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return motion_player.load_motion(temporary_path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def verify_compiler_assets_against_source_closure(
    assets: Mapping[str, bytes],
    source_closure: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    members_raw = source_closure.get("members")
    if not isinstance(members_raw, (list, tuple)):
        raise FittedGateError("verified MuJoCo source closure lacks members")
    member_sha: Dict[str, str] = {}
    for row in members_raw:
        if not isinstance(row, Mapping):
            raise FittedGateError("MuJoCo source-closure member is malformed")
        path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        try:
            member_sha[path] = native_diag._require_sha(
                digest, f"source-closure member {path}"
            )
        except native_diag.GateError as exc:
            raise FittedGateError(str(exc)) from exc
    receipts: List[Dict[str, Any]] = []
    for key, raw in sorted(assets.items()):
        expected = member_sha.get(key)
        if expected is None:
            raise FittedGateError(
                f"compiler mesh asset is absent from verified source closure: {key}"
            )
        actual = native_diag.sha256_bytes(raw)
        if actual != expected:
            raise FittedGateError(
                f"compiler mesh asset changed after identity verification: {key}"
            )
        receipts.append(
            {
                "asset_key": key,
                "sha256": actual,
                "size_bytes": len(raw),
            }
        )
    if not receipts:
        raise FittedGateError("compiled model has no verified mesh assets")
    return receipts


def _launch_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source": raw["source"],
        "activation_time_s": raw["activation_time_s"],
        "position_w_m": raw["position_w_m"],
        "velocity_w_mps": raw["velocity_w_mps"],
        "spin_w_radps": raw["spin_w_radps"],
        "source_artifact_path": raw["source_artifact_path"],
        "source_artifact_sha256": raw["source_artifact_sha256"],
        "required_incoming_table_bounces": raw[
            "required_incoming_table_bounces"
        ],
    }


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    return native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(dict(value))
    )


def _require_exact_keys(
    row: Mapping[str, Any],
    expected: Iterable[str],
    label: str,
) -> None:
    expected_set = set(expected)
    if set(row) != expected_set:
        missing = sorted(expected_set - set(row))
        extra = sorted(set(row) - expected_set)
        raise FittedGateError(
            f"{label} key set is not exact; missing={missing}, extra={extra}"
        )


def _require_vector_equal(
    first: Sequence[float],
    second: Sequence[float],
    label: str,
    *,
    tolerance: float = FORMAL_TASK_VECTOR_IDENTITY_TOL,
) -> None:
    a = np.asarray(first, np.float64)
    b = np.asarray(second, np.float64)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or float(np.max(np.abs(a - b), initial=0.0)) > tolerance
    ):
        raise FittedGateError(f"{label} differs across the frozen binding")


def _case_launch_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "activation_time_s": raw["activation_time_s"],
        "position_w_m": raw["position_w_m"],
        "velocity_w_mps": raw["velocity_w_mps"],
        "spin_w_radps": raw["spin_w_radps"],
        "required_incoming_table_bounces": raw[
            "required_incoming_table_bounces"
        ],
    }


def _validate_case_launch(
    raw: Any,
    *,
    action_id: str,
    time_to_contact_s: float,
) -> CaseLaunchState:
    row = native_diag._mapping(raw, f"{action_id}.task_case.launch")
    _require_exact_keys(
        row,
        {
            "activation_time_s",
            "position_w_m",
            "velocity_w_mps",
            "spin_w_radps",
            "required_incoming_table_bounces",
            "state_sha256",
        },
        f"{action_id}.task_case.launch",
    )
    activation = _finite(
        row["activation_time_s"],
        f"{action_id}.task_case.launch.activation_time_s",
        nonnegative=True,
    )
    if (
        activation >= time_to_contact_s
        or time_to_contact_s - activation < FORMAL_MIN_LAUNCH_LEAD_S
    ):
        raise FittedGateError(
            f"{action_id}: task-case launch does not leave the formal "
            "pre-contact lead"
        )
    position = native_diag._vector(
        row["position_w_m"], 3, f"{action_id}.task_case.launch.position"
    )
    velocity = native_diag._vector(
        row["velocity_w_mps"], 3, f"{action_id}.task_case.launch.velocity"
    )
    spin = native_diag._vector(
        row["spin_w_radps"], 3, f"{action_id}.task_case.launch.spin"
    )
    if velocity[0] >= -1.0e-6:
        raise FittedGateError(
            f"{action_id}: task-case launch is not incoming from the opponent"
        )
    required_bounces = native_diag._integer(
        row["required_incoming_table_bounces"],
        f"{action_id}.task_case.launch.required_incoming_table_bounces",
    )
    if required_bounces != 1:
        raise FittedGateError(
            f"{action_id}: task-case launch must require one incoming bounce"
        )
    state_sha = native_diag._require_sha(
        row["state_sha256"],
        f"{action_id}.task_case.launch.state_sha256",
    )
    if _canonical_payload_sha256(_case_launch_payload(row)) != state_sha:
        raise FittedGateError(
            f"{action_id}: task-case launch state SHA mismatch"
        )
    return CaseLaunchState(
        activation_time_s=activation,
        position_w_m=position,
        velocity_w_mps=velocity,
        spin_w_radps=spin,
        required_incoming_table_bounces=required_bounces,
        state_sha256=state_sha,
    )


def _validate_fault_injection(
    raw: Any,
    *,
    action_id: str,
    case_role: str,
) -> Dict[str, Any]:
    row = native_diag._mapping(
        raw, f"{action_id}.{case_role}.fault_injection"
    )
    kind = row.get("kind")
    if case_role in PHYSICAL_TASK_POSITIVE_ROLES:
        _require_exact_keys(
            row,
            {"kind"},
            f"{action_id}.{case_role}.fault_injection",
        )
        if kind != "none":
            raise FittedGateError(
                f"{action_id}.{case_role}: positive control cannot inject a fault"
            )
    elif case_role == "negative_t_hit_offset":
        _require_exact_keys(
            row,
            {"kind", "offset_s"},
            f"{action_id}.{case_role}.fault_injection",
        )
        offset = _finite(
            row["offset_s"],
            f"{action_id}.{case_role}.fault_injection.offset_s",
        )
        if not 0.02 <= abs(offset) <= 0.25:
            raise FittedGateError(
                f"{action_id}.{case_role}: t_hit fault must be in "
                "[0.02,0.25] seconds by magnitude"
            )
    elif case_role == "negative_face_sign":
        _require_exact_keys(
            row,
            {"kind"},
            f"{action_id}.{case_role}.fault_injection",
        )
        if kind != "selected_face_sign_flip":
            raise FittedGateError(
                f"{action_id}.{case_role}: wrong face-sign fault"
            )
    elif case_role == "negative_ball_state_mismatch":
        _require_exact_keys(
            row,
            {"kind", "launch_velocity_delta_w_mps"},
            f"{action_id}.{case_role}.fault_injection",
        )
        delta = native_diag._vector(
            row["launch_velocity_delta_w_mps"],
            3,
            (
                f"{action_id}.{case_role}.fault_injection."
                "launch_velocity_delta_w_mps"
            ),
        )
        norm = float(np.linalg.norm(delta))
        if not 0.25 <= norm <= 1.0:
            raise FittedGateError(
                f"{action_id}.{case_role}: ball-state mismatch magnitude "
                "must be in [0.25,1.0] m/s"
            )
    else:
        raise FittedGateError(
            f"{action_id}: unsupported physical task case role {case_role!r}"
        )
    expected_kind = {
        "negative_t_hit_offset": "teacher_t_hit_offset",
        "negative_ball_state_mismatch": "launch_velocity_delta",
    }.get(case_role)
    if expected_kind is not None and kind != expected_kind:
        raise FittedGateError(
            f"{action_id}.{case_role}: fault kind must be {expected_kind!r}"
        )
    return dict(row)


def _validate_physical_task_case(
    raw: Any,
    *,
    action: native_diag.ActionSpec,
    case_role: str,
    solver_profile_sha256: str,
    physics_profile_sha256: str,
    solver_execution_identity_sha256: str,
) -> PhysicalTaskCase:
    row = native_diag._mapping(
        raw, f"{action.action_id}.{case_role}.task_case"
    )
    _require_exact_keys(
        row,
        {
            "case_id",
            "case_role",
            "sample_seed",
            "expected_physical_verdict",
            "expected_failure_reason",
            "ball_proposal",
            "ball_proposal_sha256",
            "task_payload",
            "task_payload_sha256",
            "fault_injection",
            "case_binding_sha256",
        },
        f"{action.action_id}.{case_role}.task_case",
    )
    case_id = native_diag._nonempty_string(
        row["case_id"], f"{action.action_id}.{case_role}.case_id"
    )
    if row["case_role"] != case_role:
        raise FittedGateError(
            f"{action.action_id}: physical task case order/role drifted"
        )
    sample_seed = native_diag._integer(
        row["sample_seed"],
        f"{action.action_id}.{case_role}.sample_seed",
    )
    if sample_seed < 0:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: sample_seed must be nonnegative"
        )
    expected_verdict = row["expected_physical_verdict"]
    expected_reason = row["expected_failure_reason"]
    if case_role in PHYSICAL_TASK_POSITIVE_ROLES:
        if expected_verdict != "PASS" or expected_reason is not None:
            raise FittedGateError(
                f"{action.action_id}.{case_role}: positive control expectation "
                "must be PASS with no failure reason"
            )
    else:
        if (
            expected_verdict != "FAIL"
            or expected_reason
            != PHYSICAL_TASK_NEGATIVE_EXPECTED_REASON[case_role]
        ):
            raise FittedGateError(
                f"{action.action_id}.{case_role}: negative control expectation "
                "does not match its frozen fault"
            )

    proposal = native_diag._mapping(
        row["ball_proposal"],
        f"{action.action_id}.{case_role}.ball_proposal",
    )
    _require_exact_keys(
        proposal,
        {
            "action_id",
            "action_uid",
            "motion_sha256",
            "sample_seed",
            "sample_index",
            "ball_contact_w_m",
            "time_to_contact_s",
            "incoming_velocity_w_mps",
            "incoming_spin_w_radps",
            "base_spawn_w_m",
            "base_goal_w_m",
            "landing_aim_w_xy_m",
            "launch",
        },
        f"{action.action_id}.{case_role}.ball_proposal",
    )
    if (
        proposal["action_id"] != action.action_id
        or proposal["action_uid"] != action.action_uid
        or proposal["motion_sha256"] != action.motion_sha256
        or proposal["sample_seed"] != sample_seed
    ):
        raise FittedGateError(
            f"{action.action_id}.{case_role}: ball proposal action/seed "
            "identity drifted"
        )
    sample_index = native_diag._integer(
        proposal["sample_index"],
        f"{action.action_id}.{case_role}.sample_index",
    )
    if sample_index < 0:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: sample_index must be nonnegative"
        )
    contact = native_diag._vector(
        proposal["ball_contact_w_m"],
        3,
        f"{action.action_id}.{case_role}.ball_contact_w_m",
    )
    ttc = _finite(
        proposal["time_to_contact_s"],
        f"{action.action_id}.{case_role}.time_to_contact_s",
        positive=True,
    )
    incoming = native_diag._vector(
        proposal["incoming_velocity_w_mps"],
        3,
        f"{action.action_id}.{case_role}.incoming_velocity_w_mps",
    )
    if incoming[0] >= -1.0e-6:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: proposal is not incoming"
        )
    incoming_spin = native_diag._vector(
        proposal["incoming_spin_w_radps"],
        3,
        f"{action.action_id}.{case_role}.incoming_spin_w_radps",
    )
    base_spawn = native_diag._vector(
        proposal["base_spawn_w_m"],
        3,
        f"{action.action_id}.{case_role}.base_spawn_w_m",
    )
    base_goal = native_diag._vector(
        proposal["base_goal_w_m"],
        3,
        f"{action.action_id}.{case_role}.base_goal_w_m",
    )
    landing_aim = native_diag._vector(
        proposal["landing_aim_w_xy_m"],
        2,
        f"{action.action_id}.{case_role}.landing_aim_w_xy_m",
    )
    launch = _validate_case_launch(
        proposal["launch"],
        action_id=action.action_id,
        time_to_contact_s=ttc,
    )
    proposal_sha = native_diag._require_sha(
        row["ball_proposal_sha256"],
        f"{action.action_id}.{case_role}.ball_proposal_sha256",
    )
    if _canonical_payload_sha256(proposal) != proposal_sha:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: ball proposal SHA mismatch"
        )

    task = native_diag._mapping(
        row["task_payload"],
        f"{action.action_id}.{case_role}.task_payload",
    )
    _require_exact_keys(
        task,
        {
            "action_id",
            "action_uid",
            "motion_sha256",
            "ball_proposal_sha256",
            "mount_normal_sign",
            "ball_contact_w_m",
            "racket_site_target_w_m",
            "racket_normal_w",
            "reference_racket_quat_wxyz",
            "reference_racket_angular_velocity_w_radps",
            "racket_command_quat_wxyz",
            "racket_face_center_velocity_w_mps",
            "racket_site_velocity_w_mps",
            "racket_command_angular_velocity_w_radps",
            "geometry_source_sha256",
            "reference_t_hit_s",
            "reference_t_cycle_s",
            "reference_racket_site_speed_mps",
            "required_racket_site_speed_mps",
            "reaction_margin_s",
            "teacher_rate_min",
            "teacher_rate_max",
            "teacher_rate",
            "scaled_t_hit_s",
            "scaled_t_cycle_s",
            "pre_swing_wait_s",
            "solver_residual_m",
            "landing_aim_w_xy_m",
            "solver_profile_sha256",
            "physics_profile_sha256",
        },
        f"{action.action_id}.{case_role}.task_payload",
    )
    if (
        task["action_id"] != action.action_id
        or task["action_uid"] != action.action_uid
        or task["motion_sha256"] != action.motion_sha256
        or task["ball_proposal_sha256"] != proposal_sha
        or task["solver_profile_sha256"] != solver_profile_sha256
        or task["physics_profile_sha256"] != physics_profile_sha256
    ):
        raise FittedGateError(
            f"{action.action_id}.{case_role}: task action/proposal/profile "
            "identity drifted"
        )
    mount_sign = native_diag._integer(
        task["mount_normal_sign"],
        f"{action.action_id}.{case_role}.mount_normal_sign",
    )
    if mount_sign != action.mount_normal_sign:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: solver task selected another face/action"
        )
    task_contact = native_diag._vector(
        task["ball_contact_w_m"],
        3,
        f"{action.action_id}.{case_role}.task.ball_contact_w_m",
    )
    _require_vector_equal(
        task_contact,
        contact,
        f"{action.action_id}.{case_role}.proposal/task contact",
    )
    site_target = native_diag._vector(
        task["racket_site_target_w_m"],
        3,
        f"{action.action_id}.{case_role}.racket_site_target_w_m",
    )
    normal = native_diag._unit_vector(
        task["racket_normal_w"],
        f"{action.action_id}.{case_role}.racket_normal_w",
    )
    reference_quat = native_diag._vector(
        task["reference_racket_quat_wxyz"],
        4,
        (
            f"{action.action_id}.{case_role}."
            "reference_racket_quat_wxyz"
        ),
    )
    reference_quat_norm = float(np.linalg.norm(reference_quat))
    if abs(reference_quat_norm - 1.0) > 2.0e-5:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: reference racket quaternion "
            "is not unit length"
        )
    reference_quat = reference_quat / reference_quat_norm
    reference_omega = native_diag._vector(
        task["reference_racket_angular_velocity_w_radps"],
        3,
        (
            f"{action.action_id}.{case_role}."
            "reference_racket_angular_velocity_w_radps"
        ),
    )
    command_quat = native_diag._vector(
        task["racket_command_quat_wxyz"],
        4,
        f"{action.action_id}.{case_role}.racket_command_quat_wxyz",
    )
    quat_norm = float(np.linalg.norm(command_quat))
    if abs(quat_norm - 1.0) > 2.0e-5:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: racket command quaternion "
            "is not unit length"
        )
    face_velocity = native_diag._vector(
        task["racket_face_center_velocity_w_mps"],
        3,
        (
            f"{action.action_id}.{case_role}."
            "racket_face_center_velocity_w_mps"
        ),
    )
    site_velocity = native_diag._vector(
        task["racket_site_velocity_w_mps"],
        3,
        f"{action.action_id}.{case_role}.racket_site_velocity_w_mps",
    )
    command_omega = native_diag._vector(
        task["racket_command_angular_velocity_w_radps"],
        3,
        (
            f"{action.action_id}.{case_role}."
            "racket_command_angular_velocity_w_radps"
        ),
    )
    geometry_sha = native_diag._require_sha(
        task["geometry_source_sha256"],
        f"{action.action_id}.{case_role}.geometry_source_sha256",
    )
    reference_t_hit = _finite(
        task["reference_t_hit_s"],
        f"{action.action_id}.{case_role}.reference_t_hit_s",
        positive=True,
    )
    reference_t_cycle = _finite(
        task["reference_t_cycle_s"],
        f"{action.action_id}.{case_role}.reference_t_cycle_s",
        positive=True,
    )
    reference_speed = _finite(
        task["reference_racket_site_speed_mps"],
        (
            f"{action.action_id}.{case_role}."
            "reference_racket_site_speed_mps"
        ),
        positive=True,
    )
    required_speed = _finite(
        task["required_racket_site_speed_mps"],
        (
            f"{action.action_id}.{case_role}."
            "required_racket_site_speed_mps"
        ),
        positive=True,
    )
    reaction = _finite(
        task["reaction_margin_s"],
        f"{action.action_id}.{case_role}.reaction_margin_s",
        nonnegative=True,
    )
    rate_min = _finite(
        task["teacher_rate_min"],
        f"{action.action_id}.{case_role}.teacher_rate_min",
        positive=True,
    )
    rate_max = _finite(
        task["teacher_rate_max"],
        f"{action.action_id}.{case_role}.teacher_rate_max",
        positive=True,
    )
    rate = _finite(
        task["teacher_rate"],
        f"{action.action_id}.{case_role}.teacher_rate",
        positive=True,
    )
    scaled_t_hit = _finite(
        task["scaled_t_hit_s"],
        f"{action.action_id}.{case_role}.scaled_t_hit_s",
        positive=True,
    )
    scaled_t_cycle = _finite(
        task["scaled_t_cycle_s"],
        f"{action.action_id}.{case_role}.scaled_t_cycle_s",
        positive=True,
    )
    wait = _finite(
        task["pre_swing_wait_s"],
        f"{action.action_id}.{case_role}.pre_swing_wait_s",
        nonnegative=True,
    )
    if (
        abs(reference_t_hit - action.t_hit_s)
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
        or abs(reference_t_cycle - action.t_cycle_s)
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
        or abs(reference_speed - action.racket_speed_mps)
        > FORMAL_TASK_VECTOR_IDENTITY_TOL
        or abs(reaction - action.reaction_margin_s)
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
        or abs(rate - required_speed / reference_speed)
        > FORMAL_TASK_VECTOR_IDENTITY_TOL
        or abs(scaled_t_hit - reference_t_hit / rate)
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
        or abs(scaled_t_cycle - reference_t_cycle / rate)
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
        or abs(wait - (ttc - scaled_t_hit))
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
        or rate_min > 1.0
        or rate_max < 1.0
        or not rate_min - 5.0e-7 <= rate <= rate_max + 5.0e-7
        or wait + EPS < reaction
        or wait > 1.0 + EPS
    ):
        raise FittedGateError(
            f"{action.action_id}.{case_role}: teacher rate/wait/timing "
            "does not match the frozen task formula"
        )
    residual = _finite(
        task["solver_residual_m"],
        f"{action.action_id}.{case_role}.solver_residual_m",
        nonnegative=True,
    )
    if residual >= FORMAL_SOLVER_RESIDUAL_MAX_M:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: solver residual is not "
            f"strictly below {FORMAL_SOLVER_RESIDUAL_MAX_M} m"
        )
    try:
        replayed_geometry = (
            racket_geometry._production.solve_exact_face_contact(
                ball_contact_w_m=contact.tolist(),
                racket_face_center_velocity_w_mps=(
                    face_velocity.tolist()
                ),
                solved_raw_a_normal_w=normal.tolist(),
                mount_normal_sign=mount_sign,
                reference_racket_quat_wxyz=reference_quat.tolist(),
                reference_racket_angular_velocity_w_radps=(
                    reference_omega.tolist()
                ),
                reference_racket_site_speed_mps=reference_speed,
                teacher_rate_min=rate_min,
                teacher_rate_max=rate_max,
            )
        )
    except Exception as exc:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: exact face/site geometry "
            f"cannot be replayed: {exc}"
        ) from exc
    geometry_vectors = (
        (
            replayed_geometry.racket_command_quat_wxyz,
            command_quat / quat_norm,
            "racket command quaternion",
        ),
        (
            replayed_geometry.racket_site_target_w_m,
            site_target,
            "racket site target",
        ),
        (
            replayed_geometry.racket_face_center_velocity_w_mps,
            face_velocity,
            "racket face-center velocity",
        ),
        (
            replayed_geometry.racket_site_velocity_w_mps,
            site_velocity,
            "racket site velocity",
        ),
        (
            replayed_geometry.racket_command_angular_velocity_w_radps,
            command_omega,
            "racket command angular velocity",
        ),
    )
    for replayed, observed, label in geometry_vectors:
        _require_vector_equal(
            replayed,
            observed,
            f"{action.action_id}.{case_role}.{label}",
            tolerance=2.0e-9,
        )
    if (
        replayed_geometry.geometry_source_sha256 != geometry_sha
        or replayed_geometry.mount_normal_sign != mount_sign
        or abs(replayed_geometry.teacher_rate - rate) > 2.0e-9
    ):
        raise FittedGateError(
            f"{action.action_id}.{case_role}: exact face/site geometry "
            "identity/rate differs from deterministic replay"
        )
    _require_vector_equal(
        task["landing_aim_w_xy_m"],
        landing_aim,
        f"{action.action_id}.{case_role}.proposal/task landing aim",
    )
    task_sha = native_diag._require_sha(
        row["task_payload_sha256"],
        f"{action.action_id}.{case_role}.task_payload_sha256",
    )
    if _canonical_payload_sha256(task) != task_sha:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: task payload SHA mismatch"
        )
    fault = _validate_fault_injection(
        row["fault_injection"],
        action_id=action.action_id,
        case_role=case_role,
    )
    binding_payload = {
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "case_id": case_id,
        "case_role": case_role,
        "sample_seed": sample_seed,
        "ball_proposal_sha256": proposal_sha,
        "task_payload_sha256": task_sha,
        "solver_execution_identity_sha256": (
            solver_execution_identity_sha256
        ),
        "fault_injection": fault,
        "expected_physical_verdict": expected_verdict,
        "expected_failure_reason": expected_reason,
    }
    case_binding_sha = native_diag._require_sha(
        row["case_binding_sha256"],
        f"{action.action_id}.{case_role}.case_binding_sha256",
    )
    if _canonical_payload_sha256(binding_payload) != case_binding_sha:
        raise FittedGateError(
            f"{action.action_id}.{case_role}: case binding SHA mismatch"
        )
    return PhysicalTaskCase(
        case_id=case_id,
        case_role=case_role,
        sample_seed=sample_seed,
        expected_physical_verdict=str(expected_verdict),
        expected_failure_reason=(
            None if expected_reason is None else str(expected_reason)
        ),
        launch=launch,
        ball_contact_w_m=contact,
        incoming_velocity_w_mps=incoming,
        incoming_spin_w_radps=incoming_spin,
        time_to_contact_s=ttc,
        base_spawn_w_m=base_spawn,
        base_goal_w_m=base_goal,
        landing_aim_w_xy_m=landing_aim,
        mount_normal_sign=mount_sign,
        racket_site_target_w_m=site_target,
        racket_normal_w=normal,
        reference_racket_quat_wxyz=reference_quat,
        reference_racket_angular_velocity_w_radps=reference_omega,
        racket_command_quat_wxyz=command_quat / quat_norm,
        racket_face_center_velocity_w_mps=face_velocity,
        racket_site_velocity_w_mps=site_velocity,
        racket_command_angular_velocity_w_radps=command_omega,
        geometry_source_sha256=geometry_sha,
        teacher_rate_min=rate_min,
        teacher_rate_max=rate_max,
        teacher_rate=rate,
        scaled_t_hit_s=scaled_t_hit,
        scaled_t_cycle_s=scaled_t_cycle,
        pre_swing_wait_s=wait,
        solver_residual_m=residual,
        ball_proposal_sha256=proposal_sha,
        task_payload_sha256=task_sha,
        case_binding_sha256=case_binding_sha,
        raw=dict(row),
    )


def validate_physical_task_binding(
    raw: Any,
    *,
    action: native_diag.ActionSpec,
    solver_profile_sha256: str,
    physics_profile_sha256: str,
    geometry_source_sha256: str,
) -> PhysicalTaskBinding:
    """Validate exact solver/task/control closure without running the solver.

    The formal Gate deliberately does not import Torch or recompute the task.
    It consumes this already-produced, manifest-sealed solver receipt, checks
    every identity and formula it can independently check, and then grades the
    bound task in real MuJoCo physical replay.
    """

    row = native_diag._mapping(
        raw, f"{action.action_id}.physical_task_binding"
    )
    _require_exact_keys(
        row,
        {
            "schema_version",
            "authority",
            "action_id",
            "action_uid",
            "motion_sha256",
            "ball_profile_sha256",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "solver_implementation_source_sha256",
            "solver_execution_receipt_path",
            "solver_execution_receipt_sha256",
            "solver_execution_identity",
            "solver_execution_identity_sha256",
            "selector_executed",
            "action_identity_frozen",
            "cases",
            "cases_sha256",
        },
        f"{action.action_id}.physical_task_binding",
    )
    if (
        row["schema_version"] != PHYSICAL_TASK_BINDING_SCHEMA_VERSION
        or row["authority"] != PHYSICAL_TASK_BINDING_AUTHORITY
        or row["action_id"] != action.action_id
        or row["action_uid"] != action.action_uid
        or row["motion_sha256"] != action.motion_sha256
        or row["solver_profile_sha256"] != solver_profile_sha256
        or row["physics_profile_sha256"] != physics_profile_sha256
        or row["selector_executed"] is not False
        or row["action_identity_frozen"] is not True
    ):
        raise FittedGateError(
            f"{action.action_id}: physical task binding identity/authority drifted"
        )
    expected_ball_profile_sha = _canonical_payload_sha256(
        action.ball_profile
    )
    ball_profile_sha = native_diag._require_sha(
        row["ball_profile_sha256"],
        f"{action.action_id}.ball_profile_sha256",
    )
    if ball_profile_sha != expected_ball_profile_sha:
        raise FittedGateError(
            f"{action.action_id}: physical task binding ball-profile SHA mismatch"
        )
    source_map = native_diag._mapping(
        row["solver_implementation_source_sha256"],
        f"{action.action_id}.solver_implementation_source_sha256",
    )
    expected_source_names = {
        "continuous_questions.py",
        "hope_commands.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
    }
    if set(source_map) != expected_source_names:
        raise FittedGateError(
            f"{action.action_id}: physical task binding must pin the exact "
            "five solver files"
        )
    normalized_source_map = {
        str(name): native_diag._require_sha(
            digest, f"{action.action_id}.solver_source.{name}"
        )
        for name, digest in source_map.items()
    }
    execution_identity = native_diag._mapping(
        row["solver_execution_identity"],
        f"{action.action_id}.solver_execution_identity",
    )
    _require_exact_keys(
        execution_identity,
        {
            "artifact_type",
            "execution_id",
            "executed_before_gate",
            "solver_replayed_exact",
            "selector_executed",
            "action_identity_frozen",
            "action_switching_allowed",
            "hardware_authorized",
        },
        f"{action.action_id}.solver_execution_identity",
    )
    if (
        execution_identity["artifact_type"]
        != "frozen_ball_to_task_solver_execution_v1"
        or not isinstance(execution_identity["execution_id"], str)
        or not execution_identity["execution_id"]
        or execution_identity["executed_before_gate"] is not True
        or execution_identity["solver_replayed_exact"] is not True
        or execution_identity["selector_executed"] is not False
        or execution_identity["action_identity_frozen"] is not True
        or execution_identity["action_switching_allowed"] is not False
        or execution_identity["hardware_authorized"] is not False
    ):
        raise FittedGateError(
            f"{action.action_id}: solver execution identity is not a frozen "
            "simulation-only execution"
        )
    execution_identity_sha = native_diag._require_sha(
        row["solver_execution_identity_sha256"],
        f"{action.action_id}.solver_execution_identity_sha256",
    )
    if (
        _canonical_payload_sha256(execution_identity)
        != execution_identity_sha
    ):
        raise FittedGateError(
            f"{action.action_id}: solver execution identity SHA mismatch"
        )
    cases_raw = row["cases"]
    if (
        not isinstance(cases_raw, list)
        or len(cases_raw) != len(PHYSICAL_TASK_CASE_ROLES)
        or [
            item.get("case_role")
            for item in cases_raw
            if isinstance(item, dict)
        ]
        != list(PHYSICAL_TASK_CASE_ROLES)
    ):
        raise FittedGateError(
            f"{action.action_id}: physical task cases must be exact ordered "
            f"{list(PHYSICAL_TASK_CASE_ROLES)}"
        )
    cases_sha = native_diag._require_sha(
        row["cases_sha256"], f"{action.action_id}.cases_sha256"
    )
    if native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(cases_raw)
    ) != cases_sha:
        raise FittedGateError(
            f"{action.action_id}: physical task case-list SHA mismatch"
        )
    cases = tuple(
        _validate_physical_task_case(
            case_raw,
            action=action,
            case_role=role,
            solver_profile_sha256=solver_profile_sha256,
            physics_profile_sha256=physics_profile_sha256,
            solver_execution_identity_sha256=execution_identity_sha,
        )
        for role, case_raw in zip(PHYSICAL_TASK_CASE_ROLES, cases_raw)
    )
    if len({case.case_id for case in cases}) != len(cases):
        raise FittedGateError(
            f"{action.action_id}: physical task case_id values must be unique"
        )
    center_first, center_second = cases[:2]
    if center_first.sample_seed == center_second.sample_seed:
        raise FittedGateError(
            f"{action.action_id}: center positive controls need distinct seeds"
        )
    for first, second, label in (
        (
            center_first.ball_contact_w_m,
            center_second.ball_contact_w_m,
            "center contact",
        ),
        (
            center_first.incoming_velocity_w_mps,
            center_second.incoming_velocity_w_mps,
            "center incoming velocity",
        ),
        (
            center_first.incoming_spin_w_radps,
            center_second.incoming_spin_w_radps,
            "center spin",
        ),
        (
            center_first.base_spawn_w_m,
            center_second.base_spawn_w_m,
            "center base spawn",
        ),
        (
            center_first.base_goal_w_m,
            center_second.base_goal_w_m,
            "center base goal",
        ),
    ):
        _require_vector_equal(
            first, second, f"{action.action_id}.{label}"
        )
    if (
        abs(
            center_first.time_to_contact_s
            - center_second.time_to_contact_s
        )
        > FORMAL_TASK_TIME_IDENTITY_TOL_S
    ):
        raise FittedGateError(
            f"{action.action_id}: center multi-seed controls are not the same "
            "frozen center task"
        )
    first_center_task = dict(center_first.raw["task_payload"])
    second_center_task = dict(center_second.raw["task_payload"])
    first_center_task.pop("ball_proposal_sha256")
    second_center_task.pop("ball_proposal_sha256")
    if (
        native_diag.canonical_json_bytes(first_center_task)
        != native_diag.canonical_json_bytes(second_center_task)
    ):
        raise FittedGateError(
            f"{action.action_id}: center multi-seed solver tasks differ"
        )
    support = cases[2]
    support_delta = max(
        float(
            np.linalg.norm(
                support.ball_contact_w_m
                - center_first.ball_contact_w_m
            )
        ),
        float(
            np.linalg.norm(
                support.incoming_velocity_w_mps
                - center_first.incoming_velocity_w_mps
            )
        ),
        float(
            np.linalg.norm(
                support.incoming_spin_w_radps
                - center_first.incoming_spin_w_radps
            )
        ),
        abs(
            support.time_to_contact_s
            - center_first.time_to_contact_s
        ),
    )
    if support_delta <= FORMAL_TASK_VECTOR_IDENTITY_TOL:
        raise FittedGateError(
            f"{action.action_id}: support positive control is only another "
            "center duplicate"
        )
    profile = action.ball_profile
    speed = float(np.linalg.norm(support.incoming_velocity_w_mps))
    spin_magnitude = float(
        np.linalg.norm(support.incoming_spin_w_radps)
    )
    if not (
        float(profile["time_to_contact_min_s"])
        - EPS
        <= support.time_to_contact_s
        <= float(profile["time_to_contact_max_s"]) + EPS
        and float(profile["incoming_speed_min_mps"]) - EPS
        <= speed
        <= float(profile["incoming_speed_max_mps"]) + EPS
        and float(profile["spin_magnitude_min_radps"]) - EPS
        <= spin_magnitude
        <= float(profile["spin_magnitude_max_radps"]) + EPS
    ):
        raise FittedGateError(
            f"{action.action_id}: support positive control lies outside the "
            "action profile support"
        )
    expected_geometry_sha = native_diag._require_sha(
        geometry_source_sha256,
        f"{action.action_id}.geometry_source_sha256",
    )
    for case in cases:
        if case.geometry_source_sha256 != expected_geometry_sha:
            raise FittedGateError(
                f"{action.action_id}.{case.case_role}: task geometry source "
                "does not match the manifest geometry contract"
            )
    solver_receipt_path = native_diag._resolve_repo_file(
        row["solver_execution_receipt_path"],
        f"{action.action_id}.solver_execution_receipt_path",
    )
    solver_receipt_sha = native_diag._require_sha(
        row["solver_execution_receipt_sha256"],
        f"{action.action_id}.solver_execution_receipt_sha256",
    )
    solver_receipt, solver_receipt_file = (
        native_diag.read_json_exact(
            solver_receipt_path,
            f"{action.action_id} solver execution receipt",
            expected_sha256=solver_receipt_sha,
        )
    )
    _require_exact_keys(
        solver_receipt,
        {
            "schema_version",
            "artifact_type",
            "producer",
            "action_identity",
            "profile_identity",
            "solver_execution_identity",
            "cases",
            "receipt_payload_sha256",
        },
        f"{action.action_id}.solver_execution_receipt",
    )
    sealed_solver_receipt = dict(solver_receipt)
    observed_solver_receipt_payload_sha = (
        sealed_solver_receipt.pop("receipt_payload_sha256", None)
    )
    expected_solver_receipt_payload_sha = (
        _canonical_payload_sha256(sealed_solver_receipt)
    )
    if (
        solver_receipt["schema_version"] != 1
        or solver_receipt["artifact_type"]
        != "frozen_action_ball_solver_execution_receipt_v1"
        or observed_solver_receipt_payload_sha
        != expected_solver_receipt_payload_sha
    ):
        raise FittedGateError(
            f"{action.action_id}: external solver execution receipt "
            "schema/seal mismatch"
        )
    producer = native_diag._mapping(
        solver_receipt["producer"],
        f"{action.action_id}.solver_execution_receipt.producer",
    )
    _require_exact_keys(
        producer,
        {
            "source_path",
            "source_sha256",
            "runtime_receipt_type",
            "exact_solver_replay_required",
            "selector_executed",
            "hardware_authorized",
        },
        f"{action.action_id}.solver_execution_receipt.producer",
    )
    expected_hope_commands_path = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    )
    if (
        producer["source_path"] != expected_hope_commands_path
        or producer["source_sha256"]
        != normalized_source_map["hope_commands.py"]
        or producer["runtime_receipt_type"] != "ActionBallTaskReceipt"
        or producer["exact_solver_replay_required"] is not True
        or producer["selector_executed"] is not False
        or producer["hardware_authorized"] is not False
    ):
        raise FittedGateError(
            f"{action.action_id}: external solver receipt producer is not "
            "the pinned exact runtime task-replay path"
        )
    producer_path = native_diag._resolve_repo_file(
        producer["source_path"],
        f"{action.action_id}.solver_execution_receipt.producer.source_path",
    )
    if (
        native_diag.sha256_file(producer_path)
        != producer["source_sha256"]
    ):
        raise FittedGateError(
            f"{action.action_id}: solver receipt producer bytes drifted"
        )
    action_identity = native_diag._mapping(
        solver_receipt["action_identity"],
        f"{action.action_id}.solver_execution_receipt.action_identity",
    )
    profile_identity = native_diag._mapping(
        solver_receipt["profile_identity"],
        f"{action.action_id}.solver_execution_receipt.profile_identity",
    )
    if (
        dict(action_identity)
        != {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "motion_sha256": action.motion_sha256,
        }
        or dict(profile_identity)
        != {
            "ball_profile_sha256": ball_profile_sha,
            "solver_profile_sha256": solver_profile_sha256,
            "physics_profile_sha256": physics_profile_sha256,
            "solver_implementation_source_sha256": (
                dict(source_map)
            ),
            "geometry_source_sha256": expected_geometry_sha,
        }
        or native_diag.canonical_json_bytes(
            solver_receipt["solver_execution_identity"]
        )
        != native_diag.canonical_json_bytes(execution_identity)
        or native_diag.canonical_json_bytes(
            solver_receipt["cases"]
        )
        != native_diag.canonical_json_bytes(cases_raw)
    ):
        raise FittedGateError(
            f"{action.action_id}: manifest binding differs from the "
            "external solver execution receipt"
        )
    return PhysicalTaskBinding(
        ball_profile_sha256=ball_profile_sha,
        solver_profile_sha256=solver_profile_sha256,
        physics_profile_sha256=physics_profile_sha256,
        solver_source_sha256=dict(sorted(normalized_source_map.items())),
        solver_execution_receipt_path=Path(
            solver_receipt_file["path"]
        ),
        solver_execution_receipt_sha256=solver_receipt_sha,
        solver_execution_receipt_payload_sha256=str(
            observed_solver_receipt_payload_sha
        ),
        cases_sha256=cases_sha,
        cases=cases,
        raw=dict(row),
    )


def validate_launch_source_artifact(
    *,
    path: Path,
    expected_sha256: str,
    source: str,
    action: native_diag.ActionSpec,
    launch: Mapping[str, Any],
    expected_venue_sha256: Optional[str] = None,
    expected_recorded_contact_position_w_m: Optional[
        Sequence[float]
    ] = None,
    expected_recording_sample_time_s: Optional[float] = None,
    expected_target_contact_time_s: Optional[float] = None,
    expected_contact_velocity_w_mps: Optional[
        Sequence[float]
    ] = None,
    expected_contact_spin_w_radps: Optional[
        Sequence[float]
    ] = None,
) -> Dict[str, Any]:
    """Parse the launch evidence and bind it to the exact action and state.

    Merely pinning an opaque file hash is insufficient: a file that does not
    attest the action/motion/frame/state cannot establish a physical birth.
    """

    try:
        artifact, artifact_receipt = native_diag.read_json_exact(
            path,
            f"{action.action_id} physical launch source artifact",
            expected_sha256=expected_sha256,
        )
    except native_diag.GateError as exc:
        raise FittedGateError(str(exc)) from exc
    if native_diag._integer(
        artifact.get("schema_version"), "launch source artifact schema"
    ) != 1:
        raise FittedGateError(
            f"{action.action_id}: launch source artifact schema must equal 1"
        )
    if artifact.get("artifact_type") != source:
        raise FittedGateError(
            f"{action.action_id}: launch artifact_type does not match source"
        )
    if (
        artifact.get("action_id") != action.action_id
        or artifact.get("action_uid") != action.action_uid
        or artifact.get("motion_sha256") != action.motion_sha256
    ):
        raise FittedGateError(
            f"{action.action_id}: launch source artifact identity mismatch"
        )
    if artifact.get("coordinate_frame") != "mujoco_world":
        raise FittedGateError(
            f"{action.action_id}: launch source artifact frame must be mujoco_world"
        )
    expected_units = {
        "position": "m",
        "velocity": "m/s",
        "spin": "rad/s",
        "time": "s",
    }
    if artifact.get("units") != expected_units:
        raise FittedGateError(
            f"{action.action_id}: launch source artifact SI units mismatch"
        )
    authorization = native_diag._mapping(
        artifact.get("authorization"),
        f"{action.action_id}.launch artifact authorization",
    )
    if (
        authorization.get("physical_gate_input_authorized") is not True
        or authorization.get("hardware_authorized") is not False
    ):
        raise FittedGateError(
            f"{action.action_id}: launch artifact is not an authorized "
            "simulation-only Gate input"
        )
    state = native_diag._mapping(
        artifact.get("launch_state"),
        f"{action.action_id}.launch artifact state",
    )
    expected_state_keys = {
        "source",
        "activation_time_s",
        "position_w_m",
        "velocity_w_mps",
        "spin_w_radps",
        "required_incoming_table_bounces",
    }
    if set(state) != expected_state_keys:
        raise FittedGateError(
            f"{action.action_id}: launch artifact state key set is not exact"
        )
    expected_state = {
        key: launch[key]
        for key in (
            "source",
            "activation_time_s",
            "position_w_m",
            "velocity_w_mps",
            "spin_w_radps",
            "required_incoming_table_bounces",
        )
    }
    if (
        native_diag.canonical_json_bytes(state)
        != native_diag.canonical_json_bytes(expected_state)
    ):
        raise FittedGateError(
            f"{action.action_id}: launch artifact state does not match manifest"
        )
    upstream_path = native_diag._resolve_repo_file(
        artifact.get("upstream_evidence_path"),
        f"{action.action_id}.launch artifact upstream evidence",
    )
    upstream_sha = native_diag._require_sha(
        artifact.get("upstream_evidence_sha256"),
        f"{action.action_id}.launch artifact upstream SHA",
    )
    if upstream_path == path.resolve():
        raise FittedGateError(
            f"{action.action_id}: launch artifact cannot cite itself as upstream"
        )
    _upstream_path, upstream_raw = read_pinned_regular_file(
        upstream_path,
        upstream_sha,
        f"{action.action_id} launch upstream evidence",
    )
    try:
        upstream = json.loads(
            upstream_raw.decode("utf-8"),
            object_pairs_hook=native_diag._reject_duplicate_keys,
            parse_constant=native_diag._reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FittedGateError(
            f"{action.action_id}: launch upstream evidence must be exact JSON"
        ) from exc
    if not isinstance(upstream, dict):
        raise FittedGateError(
            f"{action.action_id}: launch upstream evidence must be an object"
        )
    sealed_upstream = dict(upstream)
    observed_upstream_seal = sealed_upstream.pop(
        "receipt_payload_sha256", None
    )
    expected_upstream_seal = native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(sealed_upstream)
    )
    if observed_upstream_seal != expected_upstream_seal:
        raise FittedGateError(
            f"{action.action_id}: launch upstream payload seal mismatch"
        )
    if native_diag._integer(
        upstream.get("schema_version"), "launch upstream schema"
    ) != 1:
        raise FittedGateError(
            f"{action.action_id}: launch upstream schema must equal 1"
        )
    if (
        upstream.get("action_id") != action.action_id
        or upstream.get("action_uid") != action.action_uid
        or upstream.get("motion_sha256") != action.motion_sha256
        or upstream.get("coordinate_frame") != "mujoco_world"
        or upstream.get("units") != expected_units
    ):
        raise FittedGateError(
            f"{action.action_id}: launch upstream identity/frame/units mismatch"
        )
    source_specific: Dict[str, Any] = {}
    source_specific["upstream_receipt_payload_sha256"] = (
        observed_upstream_seal
    )
    if source == "recorded_pre_hit_state_v1":
        source_specific["recording_sample_index"] = native_diag._integer(
            artifact.get("recording_sample_index"),
            f"{action.action_id}.recording_sample_index",
        )
        if source_specific["recording_sample_index"] < 0:
            raise FittedGateError(
                f"{action.action_id}: recording sample index must be nonnegative"
            )
        source_specific["recording_sample_time_s"] = _finite(
            artifact.get("recording_sample_time_s"),
            f"{action.action_id}.recording_sample_time_s",
            nonnegative=True,
        )
        if upstream.get("artifact_type") != "recorded_ball_state_series_v1":
            raise FittedGateError(
                f"{action.action_id}: recorded upstream artifact type mismatch"
            )
        samples = upstream.get("samples")
        if not isinstance(samples, list):
            raise FittedGateError(
                f"{action.action_id}: recorded upstream samples must be a list"
            )
        sample_index = source_specific["recording_sample_index"]
        if sample_index >= len(samples) or not isinstance(
            samples[sample_index], dict
        ):
            raise FittedGateError(
                f"{action.action_id}: recorded sample index is out of range"
            )
        sample = samples[sample_index]
        expected_sample_keys = {
            "sample_time_s",
            "position_w_m",
            "velocity_w_mps",
            "spin_w_radps",
        }
        if set(sample) != expected_sample_keys:
            raise FittedGateError(
                f"{action.action_id}: recorded sample key set is not exact"
            )
        if (
            _finite(
                sample["sample_time_s"],
                f"{action.action_id}.upstream.sample_time_s",
                nonnegative=True,
            )
            != source_specific["recording_sample_time_s"]
            or source_specific["recording_sample_time_s"]
            != float(launch["activation_time_s"])
            or native_diag.canonical_json_bytes(
                {
                    "position_w_m": sample["position_w_m"],
                    "velocity_w_mps": sample["velocity_w_mps"],
                    "spin_w_radps": sample["spin_w_radps"],
                }
            )
            != native_diag.canonical_json_bytes(
                {
                    "position_w_m": launch["position_w_m"],
                    "velocity_w_mps": launch["velocity_w_mps"],
                    "spin_w_radps": launch["spin_w_radps"],
                }
            )
        ):
            raise FittedGateError(
                f"{action.action_id}: recorded upstream sample does not "
                "bind the launch state"
            )
    elif source == RECORDED_POSITION_VENUE_FIT_ZERO_SPIN_SOURCE:
        expected_artifact_keys = {
            "schema_version",
            "artifact_type",
            "action_id",
            "action_uid",
            "motion_sha256",
            "coordinate_frame",
            "units",
            "authorization",
            "launch_state",
            "upstream_evidence_path",
            "upstream_evidence_sha256",
            "recording_sample_index",
            "recording_sample_time_s",
        }
        if (
            expected_venue_sha256 is None
            or expected_recorded_contact_position_w_m is None
            or expected_recording_sample_time_s is None
            or expected_target_contact_time_s is None
            or expected_contact_velocity_w_mps is None
            or expected_contact_spin_w_radps is None
        ):
            raise FittedGateError(
                f"{action.action_id}: venue-fit launch lacks the frozen "
                "venue/contact target"
            )
        if set(artifact) != expected_artifact_keys:
            raise FittedGateError(
                f"{action.action_id}: venue-fit launch artifact key set "
                "is not exact"
            )
        expected_venue_sha = native_diag._require_sha(
            expected_venue_sha256,
            f"{action.action_id}.expected venue SHA",
        )
        source_specific["recording_sample_index"] = (
            native_diag._integer(
                artifact.get("recording_sample_index"),
                f"{action.action_id}.recording_sample_index",
            )
        )
        source_specific["recording_sample_time_s"] = _finite(
            artifact.get("recording_sample_time_s"),
            f"{action.action_id}.recording_sample_time_s",
            nonnegative=True,
        )
        if (
            source_specific["recording_sample_index"] < 0
            or source_specific["recording_sample_time_s"]
            != float(expected_recording_sample_time_s)
            or not np.array_equal(
                np.asarray(launch["spin_w_radps"], np.float64),
                np.zeros(3, np.float64),
            )
        ):
            raise FittedGateError(
                f"{action.action_id}: recorded-position venue-fit launch "
                "time/spin assumption is invalid"
            )
        expected_provenance = {
            "position": "recorded",
            "velocity": "venue_fit_not_measured",
            "spin": "assumed_zero_not_measured",
            "measured_velocity_used": False,
            "measured_spin_used": False,
        }
        expected_upstream_keys = {
            "schema_version",
            "artifact_type",
            "action_id",
            "action_uid",
            "motion_sha256",
            "coordinate_frame",
            "units",
            "recorded_sample",
            "venue_fit",
            "birth_solution",
            "spin_assumption",
            "provenance",
            "receipt_payload_sha256",
        }
        recorded_sample = native_diag._mapping(
            upstream.get("recorded_sample"),
            f"{action.action_id}.recorded_sample",
        )
        venue_fit = native_diag._mapping(
            upstream.get("venue_fit"),
            f"{action.action_id}.venue_fit",
        )
        birth_solution = native_diag._mapping(
            upstream.get("birth_solution"),
            f"{action.action_id}.birth_solution",
        )
        spin_assumption = native_diag._mapping(
            upstream.get("spin_assumption"),
            f"{action.action_id}.spin_assumption",
        )
        provenance = native_diag._mapping(
            upstream.get("provenance"),
            f"{action.action_id}.launch provenance",
        )
        if (
            set(upstream) != expected_upstream_keys
            or upstream.get("artifact_type")
            != "recorded_position_venue_fit_ball_state_v1"
            or set(recorded_sample)
            != {"sample_index", "sample_time_s", "position_w_m"}
            or recorded_sample.get("sample_index")
            != source_specific["recording_sample_index"]
            or recorded_sample.get("sample_time_s")
            != source_specific["recording_sample_time_s"]
            or native_diag.canonical_json_bytes(
                recorded_sample.get("position_w_m")
            )
            != native_diag.canonical_json_bytes(
                list(expected_recorded_contact_position_w_m)
            )
            or set(venue_fit)
            != {
                "status",
                "contact_velocity_w_mps",
                "target_contact_position_w_m",
                "target_contact_time_s",
                "venue_yaml_path",
                "venue_yaml_sha256",
                "solver_source_path",
                "solver_source_sha256",
                "fit_input",
                "fit_input_sha256",
            }
            or venue_fit.get("status") != "PASS"
            or native_diag.canonical_json_bytes(
                venue_fit.get("contact_velocity_w_mps")
            )
            != native_diag.canonical_json_bytes(
                list(expected_contact_velocity_w_mps)
            )
            or native_diag.canonical_json_bytes(
                venue_fit.get("target_contact_position_w_m")
            )
            != native_diag.canonical_json_bytes(
                list(expected_recorded_contact_position_w_m)
            )
            or venue_fit.get("target_contact_time_s")
            != float(expected_target_contact_time_s)
            or venue_fit.get("venue_yaml_sha256")
            != expected_venue_sha
            or set(birth_solution)
            != {
                "status",
                "activation_time_s",
                "position_w_m",
                "velocity_w_mps",
                "required_incoming_table_bounces",
                "solver_source_path",
                "solver_source_sha256",
                "solver_input",
                "solver_input_sha256",
            }
            or birth_solution.get("status") != "PASS"
            or birth_solution.get("activation_time_s")
            != float(launch["activation_time_s"])
            or native_diag.canonical_json_bytes(
                birth_solution.get("position_w_m")
            )
            != native_diag.canonical_json_bytes(
                launch["position_w_m"]
            )
            or native_diag.canonical_json_bytes(
                birth_solution.get("velocity_w_mps")
            )
            != native_diag.canonical_json_bytes(
                launch["velocity_w_mps"]
            )
            or birth_solution.get(
                "required_incoming_table_bounces"
            )
            != launch["required_incoming_table_bounces"]
            or set(spin_assumption)
            != {"source", "spin_w_radps"}
            or spin_assumption.get("source")
            != "assumed_zero_not_measured"
            or not np.array_equal(
                np.asarray(
                    spin_assumption.get("spin_w_radps"), np.float64
                ),
                np.zeros(3, np.float64),
            )
            or native_diag.canonical_json_bytes(
                spin_assumption.get("spin_w_radps")
            )
            != native_diag.canonical_json_bytes(
                list(expected_contact_spin_w_radps)
            )
            or dict(provenance) != expected_provenance
        ):
            raise FittedGateError(
                f"{action.action_id}: recorded-position/venue-fit/zero-spin "
                "upstream provenance or state is not exact"
            )
        venue_path = native_diag._resolve_repo_file(
            venue_fit["venue_yaml_path"],
            f"{action.action_id}.venue_fit.venue_yaml_path",
        )
        solver_path = native_diag._resolve_repo_file(
            venue_fit["solver_source_path"],
            f"{action.action_id}.venue_fit.solver_source_path",
        )
        birth_solver_path = native_diag._resolve_repo_file(
            birth_solution["solver_source_path"],
            f"{action.action_id}.birth_solution.solver_source_path",
        )
        solver_sha = native_diag._require_sha(
            venue_fit["solver_source_sha256"],
            f"{action.action_id}.venue_fit.solver_source_sha256",
        )
        fit_input_sha = native_diag._require_sha(
            venue_fit["fit_input_sha256"],
            f"{action.action_id}.venue_fit.fit_input_sha256",
        )
        birth_solver_sha = native_diag._require_sha(
            birth_solution["solver_source_sha256"],
            f"{action.action_id}.birth_solution.solver_source_sha256",
        )
        birth_input_sha = native_diag._require_sha(
            birth_solution["solver_input_sha256"],
            f"{action.action_id}.birth_solution.solver_input_sha256",
        )
        if (
            native_diag.sha256_file(venue_path) != expected_venue_sha
            or native_diag.sha256_file(solver_path) != solver_sha
            or native_diag.sha256_file(birth_solver_path)
            != birth_solver_sha
            or native_diag.sha256_bytes(
                native_diag.canonical_json_bytes(
                    venue_fit["fit_input"]
                )
            )
            != fit_input_sha
            or native_diag.sha256_bytes(
                native_diag.canonical_json_bytes(
                    birth_solution["solver_input"]
                )
            )
            != birth_input_sha
        ):
            raise FittedGateError(
                f"{action.action_id}: venue-fit input/source bytes drifted"
            )
        source_specific.update(
            {
                "provenance": expected_provenance,
                "recorded_sample": dict(recorded_sample),
                "venue_fit": dict(venue_fit),
                "birth_solution": dict(birth_solution),
                "spin_assumption": dict(spin_assumption),
                "venue_yaml_sha256": expected_venue_sha,
                "fit_solver_source_sha256": solver_sha,
                "fit_input_sha256": fit_input_sha,
                "birth_solver_source_sha256": birth_solver_sha,
                "birth_input_sha256": birth_input_sha,
            }
        )
    elif source == "pre_registered_native_shooting_receipt_v1":
        source_specific["shooting_solver_input_sha256"] = (
            native_diag._require_sha(
                artifact.get("shooting_solver_input_sha256"),
                f"{action.action_id}.shooting_solver_input_sha256",
            )
        )
        upstream_authorization = native_diag._mapping(
            upstream.get("authorization"),
            f"{action.action_id}.shooting upstream authorization",
        )
        if (
            upstream.get("artifact_type")
            != "native_shooting_solver_receipt_v1"
            or upstream.get("status") != "PASS"
            or upstream.get("pre_registered") is not True
            or upstream.get("solver_input_sha256")
            != source_specific["shooting_solver_input_sha256"]
            or upstream.get("launch_state") != expected_state
            or upstream_authorization.get(
                "physical_gate_input_authorized"
            )
            is not True
            or upstream_authorization.get("hardware_authorized") is not False
        ):
            raise FittedGateError(
                f"{action.action_id}: native shooting upstream is not a "
                "pre-registered PASS binding the exact launch"
            )
    else:  # pragma: no cover - caller validates source first
        raise FittedGateError(
            f"{action.action_id}: unsupported launch artifact source"
        )
    return {
        "artifact": artifact_receipt,
        "artifact_type": source,
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "coordinate_frame": "mujoco_world",
        "units": expected_units,
        "upstream_evidence": {
            "path": str(upstream_path),
            "sha256": upstream_sha,
            "size_bytes": len(upstream_raw),
        },
        **source_specific,
    }


def derive_action_uid(action_id: str, family: str, motion_sha256: str) -> int:
    identity = {
        "action_id": action_id,
        "content_sha256": motion_sha256,
        "family": family,
    }
    digest = hashlib.sha256(
        native_diag.canonical_json_bytes(identity)
    ).digest()
    return 1 + (int.from_bytes(digest, byteorder="big") % ((1 << 53) - 1))


def _candidate_artifact_contains_identity(
    value: Any, *, action_id: str, motion_sha256: str
) -> bool:
    if isinstance(value, Mapping):
        if (
            (
                value.get("action_id") == action_id
                and value.get("motion_sha256") == motion_sha256
            )
            or (
                value.get("motion_id") == action_id
                and (
                    value.get("npz_sha256") == motion_sha256
                    or value.get("sha256") == motion_sha256
                )
            )
        ):
            return True
        return any(
            _candidate_artifact_contains_identity(
                item,
                action_id=action_id,
                motion_sha256=motion_sha256,
            )
            for item in value.values()
        )
    if isinstance(value, list):
        return any(
            _candidate_artifact_contains_identity(
                item,
                action_id=action_id,
                motion_sha256=motion_sha256,
            )
            for item in value
        )
    return False


def _candidate_artifact_contains_pair(
    value: Any, key: str, expected: Any
) -> bool:
    if isinstance(value, Mapping):
        if value.get(key) == expected:
            return True
        return any(
            _candidate_artifact_contains_pair(item, key, expected)
            for item in value.values()
        )
    if isinstance(value, list):
        return any(
            _candidate_artifact_contains_pair(item, key, expected)
            for item in value
        )
    return False


def _assert_candidate_artifact_not_authorizing(
    value: Any, label: str
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in (
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            ) and item is not False:
                raise FittedGateError(
                    f"{label}.{key} must remain false candidate evidence"
                )
            if key == "authorization" and isinstance(item, Mapping):
                for purpose in (
                    "training_authorized",
                    "deployment_authorized",
                    "hardware_authorized",
                ):
                    if purpose in item and item[purpose] is not False:
                        raise FittedGateError(
                            f"{label}.authorization.{purpose} must be false"
                        )
            _assert_candidate_artifact_not_authorizing(
                item, f"{label}.{key}"
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_candidate_artifact_not_authorizing(
                item, f"{label}[{index}]"
            )


def _validate_grounded_bank_evidence(
    *,
    action: Any,
    scope: str,
    bank_gate_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require the exact clip to carry the modern grounded time-law PASS.

    A recursively discovered ``verdict=PASS`` is not grounded evidence.  The
    fitted-ball replay consumes the bank's non-authorizing result, so it must
    bind the selected action/scope row, the persisted schema-2 collocation
    trace, the left/midpoint/right grounded solve, and zero safety counts.
    """

    if (
        bank_gate_report.get("schema_version") != 1
        or bank_gate_report.get("verdict") != "PASS"
        or bank_gate_report.get("bank_gate_pass") is not True
        or bank_gate_report.get("candidate_integrity_pass") is not True
        or bank_gate_report.get("grounded_trace_status")
        != "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
        or bank_gate_report.get("publication_class")
        != "post_build_diagnostic_only"
        or bank_gate_report.get("training_authorized") is not False
        or bank_gate_report.get("hardware_authorized") is not False
    ):
        raise FittedGateError(
            f"{action.action_id}: bank gate is not the exact modern "
            "non-authorizing grounded PASS"
        )
    contracts = native_diag._mapping(
        bank_gate_report.get("contracts"),
        f"{action.action_id}.bank_gate.contracts",
    )
    grounded_claim = contracts.get("grounded_inverse_dynamics")
    if (
        contracts.get("shared_ready") is not True
        or contracts.get("six_endpoint_velocity_classes_exact_zero")
        is not True
        or contracts.get("grounded_trace_status")
        != bank_gate_report["grounded_trace_status"]
        or not isinstance(grounded_claim, str)
        or not grounded_claim
        or "incomplete" in grounded_claim.lower()
        or "missing" in grounded_claim.lower()
    ):
        raise FittedGateError(
            f"{action.action_id}: bank grounded/shared-ready contract "
            "is incomplete"
        )
    aggregate = native_diag._mapping(
        bank_gate_report.get("aggregate"),
        f"{action.action_id}.bank_gate.aggregate",
    )
    clip_count = native_diag._integer(
        aggregate.get("clip_count"),
        f"{action.action_id}.bank_gate.aggregate.clip_count",
        positive=True,
    )
    exact_complete_counts = (
        "joint_limit_pass_count",
        "geometry_pass_count",
        "complete_dynamics_pass_count",
        "grounded_lmr_pass_count",
        "time_law_artifact_count",
    )
    if any(
        native_diag._integer(
            aggregate.get(key),
            f"{action.action_id}.bank_gate.aggregate.{key}",
        )
        != clip_count
        for key in exact_complete_counts
    ):
        raise FittedGateError(
            f"{action.action_id}: bank grounded/safety aggregate is incomplete"
        )
    zero_counts = (
        "failed_count",
        "incomplete_fail_closed_count",
        "grounded_lmr_incomplete_count",
        "self_collision_violation_count",
        "foot_floor_penetration_violation_count",
        "nonfoot_floor_penetration_violation_count",
        "other_world_penetration_violation_count",
    )
    if any(
        native_diag._integer(
            aggregate.get(key),
            f"{action.action_id}.bank_gate.aggregate.{key}",
        )
        != 0
        for key in zero_counts
    ):
        raise FittedGateError(
            f"{action.action_id}: bank grounded/safety aggregate has "
            "failures or penetration"
        )
    clips = bank_gate_report.get("clips")
    if not isinstance(clips, list) or len(clips) != clip_count:
        raise FittedGateError(
            f"{action.action_id}: bank clip matrix is incomplete"
        )
    selected = [
        native_diag._mapping(row, f"{action.action_id}.bank_gate.clip")
        for row in clips
        if isinstance(row, Mapping)
        and row.get("motion_id") == action.action_id
        and row.get("scope") == scope
        and row.get("sha256") == action.motion_sha256
    ]
    if len(selected) != 1:
        raise FittedGateError(
            f"{action.action_id}: bank must contain exactly one bound "
            f"{scope} clip row"
        )
    selected_row = selected[0]
    time_law = native_diag._mapping(
        selected_row.get("canonical_time_law"),
        f"{action.action_id}.bank_gate.canonical_time_law",
    )
    if (
        time_law.get("schema_version") != 2
        or time_law.get("artifact_type")
        != "canonical_time_law_collocation_v2"
        or time_law.get(
            "schema2_joint_tick_q_exact_after_published_dtype_cast"
        )
        is not True
        or time_law.get(
            "schema2_joint_tick_qdot_exact_after_published_dtype_cast"
        )
        is not True
        or time_law.get(
            "solver_input_output_array_binding_recomputed"
        )
        is not True
        or time_law.get("finite_difference_reconstruction_used")
        is not False
        or time_law.get("soft_safety_envelope_pass") is not True
    ):
        raise FittedGateError(
            f"{action.action_id}: persisted canonical time-law v2 "
            "evidence is incomplete"
        )
    grounded = native_diag._mapping(
        selected_row.get("grounded_left_midpoint_right"),
        f"{action.action_id}.bank_gate.grounded_left_midpoint_right",
    )
    if (
        grounded.get("status")
        != "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
        or grounded.get("all_feasible") is not True
        or grounded.get("finite_difference_qacc_used") is not False
        or grounded.get("roles") != ["left", "midpoint", "right"]
        or grounded.get("qacc_contract")
        != "q_s*u+q_ss*x_from_persisted_compiler_trace"
        or native_diag._integer(
            grounded.get("cell_count"),
            f"{action.action_id}.grounded.cell_count",
            positive=True,
        )
        * 3
        != native_diag._integer(
            grounded.get("sample_count"),
            f"{action.action_id}.grounded.sample_count",
            positive=True,
        )
    ):
        raise FittedGateError(
            f"{action.action_id}: grounded left/midpoint/right evidence "
            "is incomplete"
        )
    return {
        "bank_gate_pass": True,
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "scope": scope,
        "grounded_trace_status": (
            "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
        ),
        "shared_ready": True,
        "six_endpoint_velocity_classes_exact_zero": True,
        "time_law": {
            "schema_version": 2,
            "artifact_type": "canonical_time_law_collocation_v2",
            "artifact_npz_sha256": native_diag._require_sha(
                time_law.get("artifact_npz_sha256"),
                f"{action.action_id}.time_law.artifact_npz_sha256",
            ),
            "artifact_manifest_sha256": native_diag._require_sha(
                time_law.get("artifact_manifest_sha256"),
                f"{action.action_id}.time_law.artifact_manifest_sha256",
            ),
            "artifact_bundle_sha256": native_diag._require_sha(
                time_law.get("artifact_bundle_sha256"),
                f"{action.action_id}.time_law.artifact_bundle_sha256",
            ),
        },
        "grounded_lmr": {
            "cell_count": int(grounded["cell_count"]),
            "sample_count": int(grounded["sample_count"]),
            "finite_difference_qacc_used": False,
        },
        "safety_counts": {
            key: int(aggregate[key])
            for key in zero_counts
        },
    }


def validate_candidate_pre_admission(
    action: Any,
    admission_raw: Any,
    *,
    repo_file_overrides: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    """Validate compiler-candidate evidence without minting final authority.

    This stage intentionally consumes no promotion certificate or trust set:
    the final promotion is downstream of this fitted-ball result.
    """

    admission = native_diag._mapping(
        admission_raw, f"{action.action_id}.admission"
    )
    expected_admission_keys = {
        "evidence_stage",
        "publication_class",
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
        "scope",
        "registry_entry_path",
        "registry_entry_sha256",
        "compiler_manifest_path",
        "compiler_manifest_sha256",
        "bank_gate_report_path",
        "bank_gate_report_sha256",
    }
    if (
        set(admission) != expected_admission_keys
        or admission.get("evidence_stage")
        != "compiler_candidate_pre_admission_v1"
        or admission.get("publication_class") != "compiler_candidate"
        or admission.get("training_authorized") is not False
        or admission.get("deployment_authorized") is not False
        or admission.get("hardware_authorized") is not False
    ):
        raise FittedGateError(
            f"{action.action_id}: fitted-ball evidence must be an exact "
            "non-authorizing compiler-candidate pre-admission identity"
        )
    staged_registry = validate_staged_registry_overrides(
        repo_file_overrides
    )
    registry_logical_path = _logical_repo_path(
        admission.get("registry_entry_path"),
        f"{action.action_id}.admission.registry_entry_path",
    )
    compiler_logical_path = _logical_repo_path(
        admission.get("compiler_manifest_path"),
        f"{action.action_id}.admission.compiler_manifest_path",
    )
    bank_logical_path = _logical_repo_path(
        admission.get("bank_gate_report_path"),
        f"{action.action_id}.admission.bank_gate_report_path",
    )
    if set(staged_registry) - {registry_logical_path}:
        raise FittedGateError(
            f"{action.action_id}: repo_file_overrides may contain only "
            "this action's registry entry"
        )
    if (
        compiler_logical_path in staged_registry
        or bank_logical_path in staged_registry
    ):
        raise FittedGateError(
            f"{action.action_id}: compiler/bank artifacts cannot be staged "
            "through repo_file_overrides"
        )
    admission_artifacts: Dict[str, Tuple[Path, str]] = {}
    for role in (
        "registry_entry",
        "compiler_manifest",
        "bank_gate_report",
    ):
        if role == "registry_entry":
            path = _resolve_registry_entry(
                registry_logical_path,
                f"{action.action_id}.admission.{role}_path",
                repo_file_overrides=staged_registry,
            )
        else:
            logical_path = (
                compiler_logical_path
                if role == "compiler_manifest"
                else bank_logical_path
            )
            path = native_diag._resolve_repo_file(
                logical_path,
                f"{action.action_id}.admission.{role}_path",
            )
        digest = native_diag._require_sha(
            admission.get(f"{role}_sha256"),
            f"{action.action_id}.admission.{role}_sha256",
        )
        if native_diag.sha256_file(path) != digest:
            raise FittedGateError(
                f"{action.action_id}: {role} artifact bytes drifted"
            )
        admission_artifacts[role] = (path, digest)
    registry, _ = native_diag.read_json_exact(
        admission_artifacts["registry_entry"][0],
        f"{action.action_id} registry entry",
        expected_sha256=admission_artifacts["registry_entry"][1],
    )
    compiler_manifest, _ = native_diag.read_json_exact(
        admission_artifacts["compiler_manifest"][0],
        f"{action.action_id} compiler manifest",
        expected_sha256=admission_artifacts["compiler_manifest"][1],
    )
    bank_gate_report, _ = native_diag.read_json_exact(
        admission_artifacts["bank_gate_report"][0],
        f"{action.action_id} bank gate report",
        expected_sha256=admission_artifacts["bank_gate_report"][1],
    )
    for label, artifact in (
        ("registry", registry),
        ("compiler_manifest", compiler_manifest),
        ("bank_gate_report", bank_gate_report),
    ):
        _assert_candidate_artifact_not_authorizing(
            artifact, f"{action.action_id}.{label}"
        )
        if not _candidate_artifact_contains_identity(
            artifact,
            action_id=action.action_id,
            motion_sha256=action.motion_sha256,
        ):
            raise FittedGateError(
                f"{action.action_id}: {label} identity does not bind "
                "action/motion"
            )
    if not _candidate_artifact_contains_pair(
        compiler_manifest, "publication_class", "compiler_candidate"
    ):
        raise FittedGateError(
            f"{action.action_id}: compiler manifest is not a "
            "compiler_candidate"
        )
    if not (
        _candidate_artifact_contains_pair(
            bank_gate_report, "verdict", "PASS"
        )
        and _candidate_artifact_contains_pair(
            bank_gate_report, "training_authorized", False
        )
    ):
        raise FittedGateError(
            f"{action.action_id}: independent bank gate is not a "
            "non-authorizing PASS"
        )
    if admission.get("scope") not in ("upper", "full"):
        raise FittedGateError(
            f"{action.action_id}: admission.scope must be upper or full"
        )
    grounded_evidence = _validate_grounded_bank_evidence(
        action=action,
        scope=str(admission["scope"]),
        bank_gate_report=bank_gate_report,
    )
    return {
        "evidence_stage": admission["evidence_stage"],
        "scope": admission["scope"],
        "training_authorized": False,
        "grounded_evidence": grounded_evidence,
        "artifacts": {
            role: {
                "path": str(path),
                "sha256": digest,
            }
            for role, (path, digest) in admission_artifacts.items()
        },
    }


def validate_physical_manifest(
    raw: Mapping[str, Any],
    *,
    trusted_action_set: Mapping[str, Any],
    repo_file_overrides: Optional[Mapping[str, Path]] = None,
) -> PhysicalManifest:
    try:
        trusted_action_set = action_set_contract.validate_contract(
            {
                key: trusted_action_set[key]
                for key in action_set_contract.CONTRACT_KEYS
            },
            profile_id=str(trusted_action_set["profile_id"]),
            profile_policies=action_set_contract.validate_profile_policies(
                action_set_contract.ACTION_SET_PROFILE_POLICIES
            ),
        )
    except (KeyError, action_set_contract.ActionSetContractError) as exc:
        raise FittedGateError(
            f"trusted action-set contract is invalid: {exc}"
        ) from exc
    try:
        base = native_diag.validate_manifest(
            raw, expected_actions=int(trusted_action_set["expected_n"])
        )
    except native_diag.GateError as exc:
        raise FittedGateError(str(exc)) from exc
    expected_order = tuple(trusted_action_set["ordered_action_ids"])
    expected_uids = tuple(trusted_action_set["ordered_action_uids"])
    if base.action_order != expected_order:
        raise FittedGateError(
            f"action order must be exact {list(expected_order)}, "
            f"got {list(base.action_order)}"
        )
    if base.mobility_mode != trusted_action_set["mobility_mode"]:
        raise FittedGateError(
            "manifest mobility_mode differs from the trusted action-set contract"
        )
    if len({action.motion_sha256 for action in base.actions}) != len(
        base.actions
    ):
        raise FittedGateError(
            "formal actions must bind distinct motion bytes"
        )
    if tuple(action.action_uid for action in base.actions) != expected_uids:
        raise FittedGateError(
            "manifest action UID order differs from the trusted action-set contract"
        )
    staged_registry = validate_staged_registry_overrides(
        repo_file_overrides
    )
    expected_override_paths = {
        _logical_repo_path(
            row.get("admission", {}).get("registry_entry_path")
            if isinstance(row.get("admission"), Mapping)
            else None,
            f"{action.action_id}.admission.registry_entry_path",
        )
        for action, row in zip(base.actions, raw["actions"])
    }
    if set(staged_registry) - expected_override_paths:
        raise FittedGateError(
            "repo_file_overrides contains unused or non-registry paths"
        )
    candidate_evidence: Dict[str, Mapping[str, Any]] = {}
    task_bindings: Dict[str, PhysicalTaskBinding] = {}
    for action, row in zip(base.actions, raw["actions"]):
        family = native_diag._nonempty_string(
            row.get("family"), f"{action.action_id}.family"
        )
        if (
            trusted_action_set["profile_id"]
            == "fresh_upper_nomove_n5_v3"
            and family != FRESH_N5_FAMILY[action.action_id]
        ):
            raise FittedGateError(
                f"{action.action_id}: N5 family must be "
                f"{FRESH_N5_FAMILY[action.action_id]!r}"
            )
        expected_uid = derive_action_uid(
            action.action_id, family, action.motion_sha256
        )
        if action.action_uid != expected_uid:
            raise FittedGateError(
                f"{action.action_id}: action_uid does not match canonical derivation"
            )
        registry_path = _logical_repo_path(
            row.get("admission", {}).get("registry_entry_path")
            if isinstance(row.get("admission"), Mapping)
            else None,
            f"{action.action_id}.admission.registry_entry_path",
        )
        candidate_override = (
            {registry_path: staged_registry[registry_path]}
            if registry_path in staged_registry
            else None
        )
        candidate = validate_candidate_pre_admission(
            action,
            row.get("admission"),
            repo_file_overrides=candidate_override,
        )
        if candidate["scope"] != trusted_action_set["scope"]:
            raise FittedGateError(
                f"{action.action_id}: admission scope differs from the "
                "trusted action-set contract"
            )
        candidate_evidence[action.action_id] = candidate
        task_bindings[action.action_id] = (
            validate_physical_task_binding(
                row.get("physical_task_binding"),
                action=action,
                solver_profile_sha256=base.solver_profile_sha256,
                physics_profile_sha256=base.physics_profile_sha256,
                geometry_source_sha256=str(
                    base.racket_geometry_contract[
                        "geometry_source_sha256"
                    ]
                ),
            )
        )
        if trusted_action_set["mobility_mode"] == "no_move":
            for task_case in task_bindings[action.action_id].cases:
                _require_vector_equal(
                    task_case.base_goal_w_m,
                    task_case.base_spawn_w_m,
                    (
                        f"{action.action_id}.{task_case.case_role} "
                        "no_move base goal/spawn"
                    ),
                )
    holdout = native_diag._mapping(raw.get("holdout"), "manifest.holdout")
    samples = native_diag._integer(
        holdout.get("samples_per_action"),
        "holdout.samples_per_action",
        positive=True,
    )
    if samples < FORMAL_HOLDOUT_PER_ACTION_MIN:
        raise FittedGateError(
            f"holdout.samples_per_action {samples} is below formal floor "
            f"{FORMAL_HOLDOUT_PER_ACTION_MIN}"
        )
    contract = native_diag._mapping(
        raw.get("physical_contact_contract"),
        "manifest.physical_contact_contract",
    )
    if native_diag._integer(
        contract.get("schema_version"), "physical contact schema"
    ) != CONTACT_CONTRACT_VERSION:
        raise FittedGateError("physical_contact_contract.schema_version must equal 2")
    if contract.get("authority") != CONTACT_AUTHORITY:
        raise FittedGateError(
            f"physical contact authority must be {CONTACT_AUTHORITY!r}"
        )
    if contract.get("native_ball_contact_disabled") is not True:
        raise FittedGateError("formal fitted Gate requires native ball contact disabled")
    if contract.get("contact_model_path") != str(
        CONTACT_MODEL_PATH.relative_to(REPO_ROOT)
    ):
        raise FittedGateError("physical contract contact-model path drifted")
    if contract.get("contact_model_sha256") != CONTACT_MODEL_SHA256:
        raise FittedGateError("physical contract contact-model SHA drifted")
    if native_diag.sha256_file(CONTACT_MODEL_PATH) != CONTACT_MODEL_SHA256:
        raise FittedGateError("live contact-model source drifted")
    runtime_source_pins = native_diag._mapping(
        contract.get("runtime_source_sha256"),
        "physical contract runtime_source_sha256",
    )
    if set(runtime_source_pins) != set(RUNTIME_SOURCE_PATHS):
        raise FittedGateError(
            "physical contract runtime source pin set is not exact"
        )
    for name, path in RUNTIME_SOURCE_PATHS.items():
        expected_sha = native_diag._require_sha(
            runtime_source_pins[name], f"runtime source {name}"
        )
        if native_diag.sha256_file(path) != expected_sha:
            raise FittedGateError(f"runtime source drift: {name}")
    execution_source_pins = native_diag._mapping(
        contract.get("runtime_execution_source_sha256"),
        "physical contract runtime_execution_source_sha256",
    )
    if set(execution_source_pins) != set(
        RUNTIME_EXECUTION_SOURCE_PATHS
    ):
        raise FittedGateError(
            "physical contract runtime execution source pin set is not exact"
        )
    for name, path in RUNTIME_EXECUTION_SOURCE_PATHS.items():
        expected_sha = native_diag._require_sha(
            execution_source_pins[name],
            f"runtime execution source {name}",
        )
        if native_diag.sha256_file(path) != expected_sha:
            raise FittedGateError(
                f"runtime execution source drift: {name}"
            )
    execution_data_pins = native_diag._mapping(
        contract.get("runtime_execution_data_sha256"),
        "physical contract runtime_execution_data_sha256",
    )
    if set(execution_data_pins) != set(RUNTIME_EXECUTION_DATA_PATHS):
        raise FittedGateError(
            "physical contract runtime execution data pin set is not exact"
        )
    for name, path in RUNTIME_EXECUTION_DATA_PATHS.items():
        expected_sha = native_diag._require_sha(
            execution_data_pins[name],
            f"runtime execution data {name}",
        )
        if native_diag.sha256_file(path) != expected_sha:
            raise FittedGateError(
                f"runtime execution data drift: {name}"
            )
    dt_values = contract.get("convergence_timestep_s")
    if dt_values != list(DEFAULT_DT_S):
        raise FittedGateError(
            f"physical contract must freeze convergence_timestep_s={list(DEFAULT_DT_S)}"
        )
    venue_sha = native_diag._require_sha(
        contract.get("venue_yaml_sha256"), "physical contract venue SHA"
    )
    scene_sources = native_diag._mapping(
        contract.get("scene_source_sha256"),
        "physical contract scene_source_sha256",
    )
    expected_scene_sources = {
        "scripts/mujoco_table_scene.py": REPO_ROOT
        / "scripts/mujoco_table_scene.py",
        "scripts/audit_motion_schema2_table_net_clearance.py": (
            REPO_ROOT
            / "scripts/audit_motion_schema2_table_net_clearance.py"
        ),
        "table_tennis/geometry.py": (
            REPO_ROOT
            / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/table_tennis/geometry.py"
        ),
        "table_tennis/table_frame.py": (
            REPO_ROOT
            / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/table_tennis/table_frame.py"
        ),
    }
    if set(scene_sources) != set(expected_scene_sources):
        raise FittedGateError("physical contract scene source set is not exact")
    for name, path in expected_scene_sources.items():
        expected_sha = native_diag._require_sha(
            scene_sources[name], f"scene source {name}"
        )
        if native_diag.sha256_file(path) != expected_sha:
            raise FittedGateError(f"scene source drift: {name}")
    face_mesh_pins = native_diag._mapping(
        contract.get("selected_face_mesh_sha256"),
        "physical contract selected_face_mesh_sha256",
    )
    if set(face_mesh_pins) != set(FACE_MESH_PIN_KEYS):
        raise FittedGateError(
            "physical contract selected-face mesh pin set must be exact "
            f"{sorted(FACE_MESH_PIN_KEYS)}"
        )
    for name, sign in FACE_MESH_PIN_KEYS.items():
        expected_sha = native_diag._require_sha(
            face_mesh_pins[name], f"selected {name} face mesh SHA"
        )
        if native_diag.sha256_file(FACE_MESH_PATHS[sign]) != expected_sha:
            raise FittedGateError(f"selected {name} face mesh bytes drifted")
    launches: Dict[str, LaunchState] = {}
    launch_source_receipts: Dict[str, Mapping[str, Any]] = {}
    rows = raw["actions"]
    for action, row in zip(base.actions, rows):
        launch = native_diag._mapping(
            row.get("physical_ball_launch"),
            f"{action.action_id}.physical_ball_launch",
        )
        source = native_diag._nonempty_string(
            launch.get("source"), f"{action.action_id}.launch.source"
        )
        if source not in (
            "recorded_pre_hit_state_v1",
            RECORDED_POSITION_VENUE_FIT_ZERO_SPIN_SOURCE,
            "pre_registered_native_shooting_receipt_v1",
        ):
            raise FittedGateError(
                f"{action.action_id}: unsupported/untrusted physical launch source {source!r}"
            )
        activation = _finite(
            launch.get("activation_time_s"),
            f"{action.action_id}.activation_time_s",
            nonnegative=True,
        )
        ttc = _finite(
            action.ball_profile["time_to_contact_center_s"],
            f"{action.action_id}.time_to_contact",
            positive=True,
        )
        if activation >= ttc:
            raise FittedGateError(
                f"{action.action_id}: launch activation must precede contact time"
            )
        if ttc - activation < FORMAL_MIN_LAUNCH_LEAD_S:
            raise FittedGateError(
                f"{action.action_id}: launch lead must be at least "
                f"{FORMAL_MIN_LAUNCH_LEAD_S} s"
            )
        position = native_diag._vector(
            launch.get("position_w_m"), 3, f"{action.action_id}.launch.position"
        )
        velocity = native_diag._vector(
            launch.get("velocity_w_mps"), 3, f"{action.action_id}.launch.velocity"
        )
        spin = native_diag._vector(
            launch.get("spin_w_radps"), 3, f"{action.action_id}.launch.spin"
        )
        source_artifact_path = native_diag._resolve_repo_file(
            launch.get("source_artifact_path"),
            f"{action.action_id}.launch.source_artifact_path",
        )
        source_artifact_sha = native_diag._require_sha(
            launch.get("source_artifact_sha256"),
            f"{action.action_id}.launch.source_artifact_sha256",
        )
        if (
            native_diag.sha256_file(source_artifact_path)
            != source_artifact_sha
        ):
            raise FittedGateError(
                f"{action.action_id}: launch source artifact bytes drifted"
            )
        launch_source_receipts[action.action_id] = (
            validate_launch_source_artifact(
                path=source_artifact_path,
                expected_sha256=source_artifact_sha,
                source=source,
                action=action,
                launch=launch,
                expected_venue_sha256=venue_sha,
                expected_recorded_contact_position_w_m=(
                    task_bindings[action.action_id]
                    .cases[0]
                    .ball_contact_w_m
                ),
                expected_recording_sample_time_s=action.t_hit_s,
                expected_target_contact_time_s=(
                    task_bindings[action.action_id]
                    .cases[0]
                    .time_to_contact_s
                ),
                expected_contact_velocity_w_mps=(
                    task_bindings[action.action_id]
                    .cases[0]
                    .incoming_velocity_w_mps
                ),
                expected_contact_spin_w_radps=(
                    task_bindings[action.action_id]
                    .cases[0]
                    .incoming_spin_w_radps
                ),
            )
        )
        if native_diag._integer(
            launch.get("required_incoming_table_bounces"),
            f"{action.action_id}.required_incoming_table_bounces",
        ) != 1:
            raise FittedGateError(
                f"{action.action_id}: formal table-tennis launch requires "
                "exactly one incoming table bounce"
            )
        state_sha = native_diag._require_sha(
            launch.get("state_sha256"), f"{action.action_id}.launch.state_sha256"
        )
        computed = native_diag.sha256_bytes(
            native_diag.canonical_json_bytes(_launch_payload(launch))
        )
        if computed != state_sha:
            raise FittedGateError(
                f"{action.action_id}: physical launch state SHA mismatch"
            )
        launches[action.action_id] = LaunchState(
            source=source,
            activation_time_s=activation,
            position_w_m=position,
            velocity_w_mps=velocity,
            spin_w_radps=spin,
            state_sha256=state_sha,
            source_artifact_path=source_artifact_path,
            source_artifact_sha256=source_artifact_sha,
        )
    return PhysicalManifest(
        base=base,
        raw=raw,
        contract={
            **dict(contract),
            "venue_yaml_sha256": venue_sha,
            "_candidate_pre_admission_evidence": candidate_evidence,
        },
        launches=launches,
        launch_source_receipts=launch_source_receipts,
        task_bindings=task_bindings,
        action_set_contract=dict(trusted_action_set),
    )


def load_binary_stl_face(sign: int) -> FaceMesh:
    if sign not in (-1, 1):
        raise FittedGateError("face sign must be -1 or +1")
    path = FACE_MESH_PATHS[sign]
    raw = path.read_bytes()
    if len(raw) < 84:
        raise FittedGateError(f"face STL is truncated: {path}")
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + count * 50:
        raise FittedGateError(f"face STL is not exact binary STL: {path}")
    triangles = np.empty((count, 3, 3), np.float64)
    for index in range(count):
        offset = 84 + index * 50 + 12
        triangles[index] = np.frombuffer(
            raw, dtype="<f4", count=9, offset=offset
        ).reshape(3, 3)
    outer_y = FACE_OUTER_Y_M[sign]
    mask = np.all(np.abs(triangles[:, :, 1] - outer_y) <= 1.0e-6, axis=1)
    face = triangles[mask][:, :, (0, 2)]
    if face.shape[0] < 3:
        raise FittedGateError(f"cannot extract selected outer face from {path}")
    area2 = np.abs(
        (face[:, 1, 0] - face[:, 0, 0])
        * (face[:, 2, 1] - face[:, 0, 1])
        - (face[:, 1, 1] - face[:, 0, 1])
        * (face[:, 2, 0] - face[:, 0, 0])
    )
    face = face[area2 > 1.0e-12]
    if face.shape[0] < 3:
        raise FittedGateError("selected face triangles are degenerate")
    edge_counts: Dict[
        Tuple[Tuple[float, float], Tuple[float, float]], int
    ] = {}
    edge_points: Dict[
        Tuple[Tuple[float, float], Tuple[float, float]],
        Tuple[np.ndarray, np.ndarray],
    ] = {}
    for triangle in face:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            a = tuple(np.round(first, 9).tolist())
            b = tuple(np.round(second, 9).tolist())
            key = tuple(sorted((a, b)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_points[key] = (first.copy(), second.copy())
    boundary = np.asarray(
        [
            np.stack(edge_points[key])
            for key, count_value in edge_counts.items()
            if count_value == 1
        ],
        np.float64,
    )
    if boundary.ndim != 3 or boundary.shape[0] < 3:
        raise FittedGateError("selected face STL has no closed outer boundary")
    return FaceMesh(
        sign=sign,
        path=path.resolve(),
        sha256=native_diag.sha256_bytes(raw),
        outer_y_m=outer_y,
        triangles_xz_m=face,
        boundary_edges_xz_m=boundary,
    )


def point_in_triangles(
    point_xz: Sequence[float], triangles_xz: np.ndarray, tolerance: float = 1.0e-9
) -> int:
    p = np.asarray(point_xz, np.float64)
    tri = np.asarray(triangles_xz, np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = c - a, b - a, p[None, :] - a
    dot00 = np.einsum("ij,ij->i", v0, v0)
    dot01 = np.einsum("ij,ij->i", v0, v1)
    dot02 = np.einsum("ij,ij->i", v0, v2)
    dot11 = np.einsum("ij,ij->i", v1, v1)
    dot12 = np.einsum("ij,ij->i", v1, v2)
    denominator = dot00 * dot11 - dot01 * dot01
    valid = np.abs(denominator) > EPS
    u = np.full(tri.shape[0], np.inf)
    v = np.full(tri.shape[0], np.inf)
    u[valid] = (dot11[valid] * dot02[valid] - dot01[valid] * dot12[valid]) / denominator[valid]
    v[valid] = (dot00[valid] * dot12[valid] - dot01[valid] * dot02[valid]) / denominator[valid]
    inside = valid & (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1.0 + tolerance)
    indices = np.flatnonzero(inside)
    return -1 if indices.size == 0 else int(indices[0])


def point_to_boundary_distance(
    point_xz: Sequence[float], boundary_edges_xz: np.ndarray
) -> float:
    point = np.asarray(point_xz, np.float64)
    edges = np.asarray(boundary_edges_xz, np.float64)
    start = edges[:, 0]
    delta = edges[:, 1] - start
    denominator = np.einsum("ij,ij->i", delta, delta)
    alpha = np.divide(
        np.einsum("ij,ij->i", point[None, :] - start, delta),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > EPS,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = start + alpha[:, None] * delta
    return float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))


def interpolate_face_state(a: FaceState, b: FaceState, alpha: float) -> FaceState:
    try:
        rotation = racket_geometry.polar_interpolate_rotation_matrix(
            a.rotation_w_from_local,
            b.rotation_w_from_local,
            alpha,
        )
    except ValueError as exc:
        raise FittedGateError(
            f"face rotation polar interpolation is invalid: {exc}"
        ) from exc
    site = (1.0 - alpha) * a.site_position_m + alpha * b.site_position_m
    linear = (
        (1.0 - alpha) * a.site_linear_velocity_mps
        + alpha * b.site_linear_velocity_mps
    )
    angular = (
        (1.0 - alpha) * a.angular_velocity_radps
        + alpha * b.angular_velocity_radps
    )
    sign = 1 if float(a.normal_w @ a.rotation_w_from_local[:, 1]) > 0 else -1
    center_offset = racket_geometry.face_center_from_site_local(sign)
    center = site + rotation @ center_offset
    normal = rotation @ racket_geometry.face_normal_local(sign)
    normal /= np.linalg.norm(normal)
    return FaceState(site, rotation, linear, angular, center, normal)


def swept_selected_face_intersection(
    *,
    ball_start_m: Sequence[float],
    ball_end_m: Sequence[float],
    ball_velocity_start_mps: Sequence[float],
    ball_velocity_end_mps: Sequence[float],
    face_start: FaceState,
    face_end: FaceState,
    mesh: FaceMesh,
    ball_radius_m: float,
) -> Optional[SweptFaceHit]:
    p0 = np.asarray(ball_start_m, np.float64)
    p1 = np.asarray(ball_end_m, np.float64)
    v0 = np.asarray(ball_velocity_start_mps, np.float64)
    v1 = np.asarray(ball_velocity_end_mps, np.float64)

    def clearance(alpha: float) -> float:
        face = interpolate_face_state(face_start, face_end, alpha)
        ball = (1.0 - alpha) * p0 + alpha * p1
        return float((ball - face.center_position_m) @ face.normal_w - ball_radius_m)

    d0, d1 = clearance(0.0), clearance(1.0)
    if not (math.isfinite(d0) and math.isfinite(d1)):
        raise FittedGateError("swept face clearance is non-finite")
    if d0 < -1.0e-7 or d0 * d1 > 0.0 or d1 > 1.0e-7:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if clearance(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    face = interpolate_face_state(face_start, face_end, alpha)
    ball = (1.0 - alpha) * p0 + alpha * p1
    velocity = (1.0 - alpha) * v0 + alpha * v1
    face_point = ball - ball_radius_m * face.normal_w
    local = face.rotation_w_from_local.T @ (face_point - face.site_position_m)
    triangle = point_in_triangles(local[[0, 2]], mesh.triangles_xz_m)
    if triangle < 0 or abs(float(local[1]) - mesh.outer_y_m) > 2.0e-6:
        return None
    edge_clearance = point_to_boundary_distance(
        local[[0, 2]], mesh.boundary_edges_xz_m
    )
    if edge_clearance <= ball_radius_m + FORMAL_FACE_EDGE_GUARD_M:
        return None
    point_velocity = racket_geometry.rigid_point_velocity(
        face.site_linear_velocity_mps,
        face.angular_velocity_radps,
        face_point - face.site_position_m,
    )
    relative_normal = float((velocity - point_velocity) @ face.normal_w)
    if relative_normal >= -1.0e-6:
        return None
    return SweptFaceHit(
        alpha=alpha,
        ball_center_m=ball,
        face_point_m=face_point,
        face_point_local_m=local,
        normal_w=face.normal_w,
        face_point_velocity_mps=point_velocity,
        relative_normal_speed_mps=relative_normal,
        triangle_index=triangle,
        edge_clearance_m=edge_clearance,
    )


def fitted_contact(
    velocity_mps: Sequence[float],
    surface_velocity_mps: Sequence[float],
    normal_w: Sequence[float],
    spin_radps: Sequence[float],
    *,
    e_eff: float,
    a_t: float,
    b_t: float,
    mu: float,
) -> Dict[str, Any]:
    result = contact_model.predict_contact(
        np.asarray(velocity_mps, np.float64)[None, :],
        np.asarray(surface_velocity_mps, np.float64)[None, :],
        np.asarray(normal_w, np.float64)[None, :],
        np.asarray(spin_radps, np.float64)[None, :],
        float(e_eff),
        float(a_t),
        float(b_t),
        float(mu),
    )
    return {
        "velocity_plus_mps": np.asarray(result["v_plus"][0], np.float64),
        "spin_plus_radps": np.asarray(result["omega_plus"][0], np.float64),
        "u_n_mps": float(result["u_n"][0]),
        "u_t_mps": float(result["u_t"][0]),
        "cap_binds": bool(result["cap_binds"][0]),
        "oriented_normal_w": np.asarray(result["n"][0], np.float64),
    }


def aero_acceleration(
    velocity_mps: Sequence[float],
    spin_radps: Sequence[float],
    venue: VenueParams,
) -> np.ndarray:
    velocity = np.asarray(velocity_mps, np.float64)
    spin = np.asarray(spin_radps, np.float64)
    return (
        np.asarray((0.0, 0.0, -venue.gravity), np.float64)
        - venue.k_d * float(np.linalg.norm(velocity)) * velocity
        + venue.k_m * np.cross(spin, velocity)
    )


def advance_fitted_flight(
    position_m: Sequence[float],
    velocity_mps: Sequence[float],
    spin_radps: Sequence[float],
    duration_s: float,
    venue: VenueParams,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(position_m, np.float64).copy()
    v = np.asarray(velocity_mps, np.float64).copy()
    spin = np.asarray(spin_radps, np.float64)
    h = float(duration_s)
    if h <= 0:
        return p, v

    def derivative(pp: np.ndarray, vv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        del pp
        return vv, aero_acceleration(vv, spin, venue)

    k1p, k1v = derivative(p, v)
    k2p, k2v = derivative(p + 0.5 * h * k1p, v + 0.5 * h * k1v)
    k3p, k3v = derivative(p + 0.5 * h * k2p, v + 0.5 * h * k2v)
    k4p, k4v = derivative(p + h * k3p, v + h * k3v)
    return (
        p + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p),
        v + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v),
    )


def ball_spin_world(
    data: Any, qpos_address: int, dof_address: int
) -> np.ndarray:
    rotation = motion_player.quaternion_wxyz_to_matrix(
        np.asarray(
            data.qpos[qpos_address + 3 : qpos_address + 7],
            np.float64,
        )
    )
    return rotation @ np.asarray(
        data.qvel[dof_address + 3 : dof_address + 6], np.float64
    )


def set_ball_spin_world(
    data: Any,
    qpos_address: int,
    dof_address: int,
    spin_world_radps: Sequence[float],
) -> None:
    rotation = motion_player.quaternion_wxyz_to_matrix(
        np.asarray(
            data.qpos[qpos_address + 3 : qpos_address + 7],
            np.float64,
        )
    )
    data.qvel[dof_address + 3 : dof_address + 6] = (
        rotation.T @ np.asarray(spin_world_radps, np.float64)
    )


def swept_table_crossing(
    p0: Sequence[float],
    p1: Sequence[float],
    v0: Sequence[float],
    *,
    center_surface_z_m: float,
    near_x_m: float,
    far_x_m: float,
    half_width_m: float,
) -> Optional[Tuple[float, np.ndarray]]:
    a, b = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    if not (
        a[2] > center_surface_z_m >= b[2]
        and np.asarray(v0, np.float64)[2] < 0.0
    ):
        return None
    alpha = float(
        (a[2] - center_surface_z_m)
        / max(a[2] - b[2], EPS)
    )
    point = a + alpha * (b - a)
    if not (
        near_x_m <= point[0] <= far_x_m
        and abs(point[1]) <= half_width_m
    ):
        return None
    return alpha, point


def segment_expanded_aabb_hit(
    p0: Sequence[float],
    p1: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    radius: float,
) -> Optional[float]:
    a = np.asarray(p0, np.float64)
    delta = np.asarray(p1, np.float64) - a
    lower = np.asarray(lo, np.float64) - radius
    upper = np.asarray(hi, np.float64) + radius
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(delta[axis]) < EPS:
            if not lower[axis] <= a[axis] <= upper[axis]:
                return None
            continue
        t1 = (lower[axis] - a[axis]) / delta[axis]
        t2 = (upper[axis] - a[axis]) / delta[axis]
        t1, t2 = min(t1, t2), max(t1, t2)
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min


def compare_convergence(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    *,
    contact_time_tol_s: float = 0.002,
    contact_position_tol_m: float = 0.005,
    outgoing_velocity_tol_mps: float = 0.10,
    net_height_tol_m: float = 0.01,
    landing_xy_tol_m: float = 0.02,
    landing_time_tol_s: float = 0.02,
) -> Dict[str, Any]:
    reasons: List[str] = []
    metrics: Dict[str, Optional[float]] = {}

    def distance(path: Tuple[str, ...]) -> Optional[float]:
        a: Any = coarse
        b: Any = fine
        try:
            for key in path:
                a, b = a[key], b[key]
            if a is None or b is None:
                return None
            if isinstance(a, list):
                return float(
                    np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64))
                )
            return abs(float(a) - float(b))
        except (KeyError, TypeError):
            return None

    checks = (
        ("contact_time_s", ("paddle_contact", "time_s"), contact_time_tol_s),
        (
            "contact_position_m",
            ("paddle_contact", "ball_center_m"),
            contact_position_tol_m,
        ),
        (
            "outgoing_velocity_mps",
            ("paddle_contact", "velocity_plus_mps"),
            outgoing_velocity_tol_mps,
        ),
        (
            "net_height_m",
            ("net_crossing", "ball_center_z_m"),
            net_height_tol_m,
        ),
        (
            "landing_xy_m",
            ("first_landing", "ball_center_xy_m"),
            landing_xy_tol_m,
        ),
        (
            "landing_time_s",
            ("first_landing", "time_s"),
            landing_time_tol_s,
        ),
    )
    for name, path, tolerance in checks:
        value = distance(path)
        metrics[name] = value
        if value is None or value > tolerance:
            reasons.append(f"nonconverged_{name}")
    return {
        "pass": not reasons,
        "metrics": metrics,
        "failure_reasons": reasons,
        "tolerances": {
            "contact_time_s": contact_time_tol_s,
            "contact_position_m": contact_position_tol_m,
            "outgoing_velocity_mps": outgoing_velocity_tol_mps,
            "net_height_m": net_height_tol_m,
            "landing_xy_m": landing_xy_tol_m,
            "landing_time_s": landing_time_tol_s,
        },
    }


def ready_recovery_metrics(
    clips: Mapping[str, motion_player.MotionClip],
    action_order: Sequence[str],
) -> Dict[str, Any]:
    """Measure the complete root+joint ready/recovery state.

    Comparing only the 31 joint angles misses exactly the coordinate failure
    that motivated this Gate: two clips can share joint values while their
    root translations/orientations place the robot at different stations.
    Endpoint velocity is part of the ready contract as well.
    """

    if not action_order or set(clips) != set(action_order):
        raise FittedGateError(
            "ready/recovery clip set does not match exact action order"
        )
    first = clips[action_order[0]]
    reference_joint = np.asarray(first.joint_pos[0], np.float64)
    reference_root_position = np.asarray(
        first.body_pos_w[0, 0], np.float64
    )
    reference_root_orientation = np.asarray(
        first.body_quat_w[0, 0], np.float64
    )
    per_action: Dict[str, Dict[str, float]] = {}
    shared_joint_linf = 0.0
    shared_root_position_l2 = 0.0
    shared_root_orientation = 0.0
    endpoint_joint_velocity_peak = 0.0
    endpoint_root_linear_velocity_peak = 0.0
    endpoint_root_angular_velocity_peak = 0.0
    for action_id in action_order:
        clip = clips[action_id]
        start_joint = np.asarray(clip.joint_pos[0], np.float64)
        end_joint = np.asarray(clip.joint_pos[-1], np.float64)
        start_root_position = np.asarray(
            clip.body_pos_w[0, 0], np.float64
        )
        end_root_position = np.asarray(
            clip.body_pos_w[-1, 0], np.float64
        )
        start_root_orientation = np.asarray(
            clip.body_quat_w[0, 0], np.float64
        )
        end_root_orientation = np.asarray(
            clip.body_quat_w[-1, 0], np.float64
        )
        shared_joint_linf = max(
            shared_joint_linf,
            float(np.max(np.abs(start_joint - reference_joint))),
        )
        shared_root_position_l2 = max(
            shared_root_position_l2,
            float(
                np.linalg.norm(
                    start_root_position - reference_root_position
                )
            ),
        )
        shared_root_orientation = max(
            shared_root_orientation,
            _shortest_quaternion_angle_rad(
                start_root_orientation,
                reference_root_orientation,
            ),
        )
        joint_velocity_peak = float(
            np.max(
                np.abs(
                    np.stack(
                        (clip.joint_vel[0], clip.joint_vel[-1])
                    )
                )
            )
        )
        root_linear_velocity_peak = float(
            np.max(
                np.abs(
                    np.stack(
                        (
                            clip.body_lin_vel_w[0, 0],
                            clip.body_lin_vel_w[-1, 0],
                        )
                    )
                )
            )
        )
        root_angular_velocity_peak = float(
            np.max(
                np.abs(
                    np.stack(
                        (
                            clip.body_ang_vel_w[0, 0],
                            clip.body_ang_vel_w[-1, 0],
                        )
                    )
                )
            )
        )
        endpoint_joint_velocity_peak = max(
            endpoint_joint_velocity_peak, joint_velocity_peak
        )
        endpoint_root_linear_velocity_peak = max(
            endpoint_root_linear_velocity_peak,
            root_linear_velocity_peak,
        )
        endpoint_root_angular_velocity_peak = max(
            endpoint_root_angular_velocity_peak,
            root_angular_velocity_peak,
        )
        per_action[action_id] = {
            "joint_linf_rad": float(
                np.max(np.abs(end_joint - start_joint))
            ),
            "root_position_l2_m": float(
                np.linalg.norm(
                    end_root_position - start_root_position
                )
            ),
            "root_orientation_angle_rad": (
                _shortest_quaternion_angle_rad(
                    end_root_orientation,
                    start_root_orientation,
                )
            ),
            "endpoint_joint_velocity_peak_radps": joint_velocity_peak,
            "endpoint_root_linear_velocity_peak_mps": (
                root_linear_velocity_peak
            ),
            "endpoint_root_angular_velocity_peak_radps": (
                root_angular_velocity_peak
            ),
        }
    return {
        "shared_ready": {
            "joint_linf_rad": shared_joint_linf,
            "root_position_l2_m": shared_root_position_l2,
            "root_orientation_angle_rad": shared_root_orientation,
            "endpoint_joint_velocity_peak_radps": (
                endpoint_joint_velocity_peak
            ),
            "endpoint_root_linear_velocity_peak_mps": (
                endpoint_root_linear_velocity_peak
            ),
            "endpoint_root_angular_velocity_peak_radps": (
                endpoint_root_angular_velocity_peak
            ),
            "thresholds": {
                "joint_linf_rad": FORMAL_SHARED_READY_JOINT_TOL_RAD,
                "root_position_l2_m": (
                    FORMAL_SHARED_READY_ROOT_POSITION_TOL_M
                ),
                "root_orientation_angle_rad": (
                    FORMAL_SHARED_READY_ROOT_ORIENTATION_TOL_RAD
                ),
                "endpoint_velocity_peak": (
                    FORMAL_ENDPOINT_VELOCITY_TOL
                ),
            },
        },
        "recovery_by_action": per_action,
        "recovery_thresholds": {
            "joint_linf_rad": FORMAL_RECOVERY_JOINT_TOL_RAD,
            "root_position_l2_m": (
                FORMAL_RECOVERY_ROOT_POSITION_TOL_M
            ),
            "root_orientation_angle_rad": (
                FORMAL_RECOVERY_ROOT_ORIENTATION_TOL_RAD
            ),
            "endpoint_velocity_peak": FORMAL_ENDPOINT_VELOCITY_TOL,
        },
    }


def retimed_teacher_state(
    clip: motion_player.MotionClip,
    *,
    world_time_s: float,
    pre_swing_wait_s: float,
    teacher_rate: float,
) -> Dict[str, Any]:
    """Evaluate a teacher time warp and scale every physical velocity.

    Pose phase is ``(world_time - wait) * rate``.  A time warp without the
    same rate applied to root/joint velocities gives MuJoCo a pose from one
    trajectory and a racket speed from another, which is precisely the silent
    retiming bug this Gate must reject.
    """

    rate = _finite(teacher_rate, "teacher_rate", positive=True)
    wait = _finite(
        pre_swing_wait_s, "pre_swing_wait_s", nonnegative=True
    )
    state = dict(
        native_diag.interpolate_teacher(
            clip, (float(world_time_s) - wait) * rate
        )
    )
    for key in ("root_lin_vel", "root_ang_vel", "joint_vel"):
        values = np.asarray(state[key], np.float64)
        if not np.isfinite(values).all():
            raise FittedGateError(
                f"retimed teacher {key} contains NaN/Inf"
            )
        state[key] = values * rate
    state["teacher_rate"] = rate
    state["source_motion_time_s"] = (
        float(world_time_s) - wait
    ) * rate
    return state


def assemble_fitted_scene_xml(
    canonical_xml: bytes,
    obstacle_rows: Mapping[str, Any],
    venue: VenueParams,
    timestep_s: float,
) -> Tuple[bytes, Dict[str, Any]]:
    table_xml = table_scene.augment_mjcf_xml(
        canonical_xml, obstacle_rows, collidable=True
    )
    root = ET.fromstring(table_xml)
    option = root.find("./option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", format(float(timestep_s), ".17g"))
    option.set("gravity", f"0 0 {-venue.gravity:.17g}")
    world = root.find("./worldbody")
    if world is None:
        raise FittedGateError("vendor MJCF has no worldbody")
    inertia = venue.inertia_coeff * venue.ball_mass * venue.ball_radius**2
    body = ET.SubElement(world, "body", {"name": BALL_BODY_NAME, "pos": "0 0 100"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": format(venue.ball_mass, ".17g"),
            "diaginertia": " ".join([format(inertia, ".17g")] * 3),
        },
    )
    ET.SubElement(body, "freejoint", {"name": BALL_JOINT_NAME})
    ET.SubElement(
        body,
        "geom",
        {
            "name": BALL_GEOM_NAME,
            "type": "sphere",
            "size": format(venue.ball_radius, ".17g"),
            "rgba": "1 0.5 0 1",
            "contype": "0",
            "conaffinity": "0",
            "condim": "1",
        },
    )
    final = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return final, {
        "canonical_xml_sha256": native_diag.sha256_bytes(canonical_xml),
        "table_xml_sha256": native_diag.sha256_bytes(table_xml),
        "fitted_scene_xml_sha256": native_diag.sha256_bytes(final),
        "timestep_s": float(timestep_s),
        "gravity_mps2": [0.0, 0.0, -venue.gravity],
        "ball_native_contact_disabled": True,
        "ball_mass_kg": venue.ball_mass,
        "ball_radius_m": venue.ball_radius,
        "ball_diagonal_inertia_kg_m2": inertia,
        "obstacle_geometry_sha256": native_diag.sha256_bytes(
            native_diag.canonical_json_bytes(obstacle_rows)
        ),
    }


def _face_state(
    mujoco: Any,
    model: Any,
    data: Any,
    binding: motion_player.ModelBinding,
    sign: int,
) -> FaceState:
    site = np.asarray(data.site_xpos[binding.racket_site_id], np.float64).copy()
    rotation = np.asarray(
        data.site_xmat[binding.racket_site_id], np.float64
    ).reshape(3, 3).copy()
    linear, angular = native_diag._site_twist(
        mujoco, model, data, binding.racket_site_id
    )
    center = site + rotation @ racket_geometry.face_center_from_site_local(sign)
    normal = rotation @ racket_geometry.face_normal_local(sign)
    normal /= np.linalg.norm(normal)
    return FaceState(site, rotation, linear, angular, center, normal)


def _obstacle_aabbs(rows: Mapping[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    all_rows = [rows["table_top"], rows["net"], *rows["net_posts"]]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for row in all_rows:
        center = np.asarray(row["center_mjcf_world_m"], np.float64)
        half = 0.5 * np.asarray(row["full_extents_m"], np.float64)
        out[str(row["name"])] = (center - half, center + half)
    return out


def validate_scene_against_profile(
    rows: Mapping[str, Any],
    profile: Mapping[str, Any],
    venue: VenueParams,
) -> Dict[str, Any]:
    aabbs = _obstacle_aabbs(rows)
    table_lo, table_hi = aabbs[TABLE_GEOM_NAME]
    net_lo, net_hi = aabbs["motion_net"]
    checks = {
        "table_near_x_m": (
            float(table_lo[0]),
            float(profile["opponent_near_x_m"]),
        ),
        "table_far_x_m": (
            float(table_hi[0]),
            float(profile["opponent_far_x_m"]),
        ),
        "table_surface_z_m": (
            float(table_hi[2]),
            float(profile["table_surface_z_m"]),
        ),
        "table_half_width_m": (
            float(max(abs(table_lo[1]), abs(table_hi[1]))),
            float(profile["table_half_width_m"]),
        ),
        "net_x_m": (
            float(0.5 * (net_lo[0] + net_hi[0])),
            float(profile["net_x_m"]),
        ),
        "ball_center_net_top_z_m": (
            float(net_hi[2] + venue.ball_radius),
            float(profile["ball_center_net_top_z_m"]),
        ),
    }
    mismatches = {
        key: {"scene": actual, "profile": expected}
        for key, (actual, expected) in checks.items()
        if abs(actual - expected) > 2.0e-9
    }
    if mismatches:
        raise FittedGateError(
            f"scene/profile table-net geometry mismatch: {mismatches}"
        )
    if abs(venue.ball_radius - float(racket_geometry.BALL_RADIUS_M)) > 1.0e-8:
        raise FittedGateError(
            "venue ball radius does not match the versioned racket geometry"
        )
    if abs(venue.ball_radius - float(contact_model.R_BALL)) > 1.0e-12:
        raise FittedGateError(
            "venue ball radius does not match the fitted contact model"
        )
    if abs(venue.inertia_coeff - float(contact_model.C_INERTIA)) > 1.0e-12:
        raise FittedGateError(
            "venue inertia coefficient does not match the fitted contact model"
        )
    return {
        key: {"scene": actual, "profile": expected}
        for key, (actual, expected) in checks.items()
    }


def validate_compiled_obstacles(
    mujoco: Any,
    model: Any,
    obstacle_rows: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    *,
    assembled_xml_sha256: str,
) -> Dict[str, Any]:
    """Bind the exact five-solid robot scene and robot-only keepout filter."""

    recomputed = table_scene.action_ball_policy_geometry_contract(
        obstacle_rows
    )
    if recomputed != dict(geometry_contract):
        raise FittedGateError("five-solid geometry contract drifted")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    expected = {
        row["name"]: row
        for row in table_scene.action_ball_policy_obstacle_rows(
            obstacle_rows
        )
    }
    if tuple(expected) != table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES:
        raise FittedGateError(
            "compiled five-solid obstacle expectation order is not exact"
        )
    receipts: List[Dict[str, Any]] = []
    obstacle_ids: List[int] = []
    for name in table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES:
        geom_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
        )
        if geom_id < 0:
            raise FittedGateError(f"compiled obstacle is missing: {name}")
        row = expected[name]
        expected_center = np.asarray(
            row["center_mjcf_world_m"], np.float64
        )
        expected_half = (
            0.5 * np.asarray(row["full_extents_m"], np.float64)
        )
        checks = (
            int(model.geom_type[geom_id])
            == int(mujoco.mjtGeom.mjGEOM_BOX),
            int(model.geom_bodyid[geom_id]) == 0,
            int(model.geom_contype[geom_id]) == 0,
            int(model.geom_conaffinity[geom_id]) == 7,
            bool(
                np.allclose(
                    data.geom_xpos[geom_id],
                    expected_center,
                    atol=1.0e-12,
                    rtol=0.0,
                )
            ),
            bool(
                np.allclose(
                    model.geom_size[geom_id, :3],
                    expected_half,
                    atol=1.0e-12,
                    rtol=0.0,
                )
            ),
            bool(
                np.allclose(
                    np.asarray(data.geom_xmat[geom_id]).reshape(3, 3),
                    np.eye(3),
                    atol=1.0e-12,
                    rtol=0.0,
                )
            ),
        )
        if not all(checks):
            raise FittedGateError(
                f"compiled obstacle type/body/pose/size/collision bits drifted: {name}"
            )
        receipts.append(
            {
                "name": name,
                "geom_id": geom_id,
                "type": "box",
                "body_id": 0,
                "center_m": expected_center.tolist(),
                "half_extents_m": expected_half.tolist(),
                "contype": 0,
                "conaffinity": 7,
            }
        )
        obstacle_ids.append(geom_id)

    ball_geom = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME
        )
    )
    keepout_geom = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME,
        )
    )
    if min(ball_geom, keepout_geom) < 0:
        raise FittedGateError("compiled scene lacks fitted ball or keepout")
    if (
        int(model.geom_contype[ball_geom]) != 0
        or int(model.geom_conaffinity[ball_geom]) != 0
        or _geom_pair_enabled(
            mujoco, model, ball_geom, keepout_geom
        )
    ):
        raise FittedGateError(
            "robot-only keepout can affect the fitted ball"
        )
    analytic_ball_obstacles = _obstacle_aabbs(obstacle_rows)
    if (
        table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
        in analytic_ball_obstacles
    ):
        raise FittedGateError(
            "robot-only keepout entered the analytic ball referee"
        )
    physics_robot_ids = _five_solid_robot_geom_ids(
        mujoco,
        model,
        ball_geom_id=ball_geom,
        obstacle_geom_ids=obstacle_ids,
    )
    if not physics_robot_ids:
        raise FittedGateError(
            "five-solid scene has no physics-enabled robot geoms"
        )
    if any(
        not _geom_pair_enabled(
            mujoco, model, geom_id, keepout_geom
        )
        for geom_id in physics_robot_ids
    ):
        raise FittedGateError(
            "a physics-enabled robot geom is filtered from the keepout"
        )
    try:
        xml_sha = native_diag._require_sha(
            assembled_xml_sha256,
            "assembled five-solid teacher XML SHA256",
        )
    except native_diag.GateError as exc:
        raise FittedGateError(str(exc)) from exc
    return {
        "five_solid_geometry_sha256": geometry_contract["sha256"],
        "assembled_xml_sha256": xml_sha,
        "compiled_obstacles": receipts,
        "physics_enabled_robot_geom_count": len(physics_robot_ids),
        "teacher_swept_subject": (
            "all_and_only_robot_body_geoms_collision_enabled_against_"
            "the_five_solid_scene_including_feet"
        ),
        "ball_keepout_native_pair_enabled": False,
        "ball_keepout_analytic_surface_enabled": False,
        "contact_force_threshold_n": TABLE_CONTACT_FORCE_THRESHOLD_N,
        "continuous_sweep_method": FIVE_SOLID_SWEEP_METHOD,
    }


def build_ground_contact_contract(
    mujoco: Any,
    model: Any,
    *,
    ball_geom_id: int,
) -> GroundContactContract:
    """Bind the one floor and the exact two legal support bodies.

    Legal support is a body identity contract, not a substring rule.  Only
    collision-enabled geoms directly attached to the two ankle-roll bodies
    may touch the floor without being an immediate non-foot violation.
    """

    floor_geom_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME
        )
    )
    if (
        floor_geom_id < 0
        or int(model.geom_bodyid[floor_geom_id]) != 0
        or int(model.geom_type[floor_geom_id])
        != int(mujoco.mjtGeom.mjGEOM_PLANE)
    ):
        raise FittedGateError(
            "teacher ground contract requires exact world plane 'floor'"
        )
    foot_body_ids = tuple(
        int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
        )
        for body_name in LEGAL_FOOT_BODY_NAMES
    )
    if (
        len(foot_body_ids) != 2
        or min(foot_body_ids) <= 0
        or len(set(foot_body_ids)) != 2
    ):
        raise FittedGateError(
            "teacher ground contract lacks the exact two foot bodies"
        )
    foot_body_set = set(foot_body_ids)
    foot_geom_ids: List[int] = []
    nonfoot_robot_geom_ids: List[int] = []
    for geom_id in range(int(model.ngeom)):
        if geom_id in (floor_geom_id, int(ball_geom_id)):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        collision_enabled = bool(
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
        if not collision_enabled:
            continue
        if not _geom_pair_enabled(
            mujoco, model, geom_id, floor_geom_id
        ):
            raise FittedGateError(
                "a collision-enabled robot geom is filtered from the floor: "
                f"{native_diag._geom_name(mujoco, model, geom_id)}"
            )
        if body_id in foot_body_set:
            foot_geom_ids.append(geom_id)
        else:
            nonfoot_robot_geom_ids.append(geom_id)
    if (
        not foot_geom_ids
        or {
            int(model.geom_bodyid[geom_id])
            for geom_id in foot_geom_ids
        }
        != foot_body_set
    ):
        raise FittedGateError(
            "teacher ground contract has incomplete collision-enabled "
            "foot geometry"
        )
    return GroundContactContract(
        floor_geom_id=floor_geom_id,
        foot_body_ids=(
            int(foot_body_ids[0]),
            int(foot_body_ids[1]),
        ),
        foot_geom_ids=tuple(foot_geom_ids),
        nonfoot_robot_geom_ids=tuple(nonfoot_robot_geom_ids),
    )


def ground_contact_contract_receipt(
    mujoco: Any,
    model: Any,
    contract: GroundContactContract,
) -> Dict[str, Any]:
    return {
        "floor_geom_name": FLOOR_GEOM_NAME,
        "floor_geom_id": int(contract.floor_geom_id),
        "floor_geom_type": "plane",
        "legal_foot_body_names": list(LEGAL_FOOT_BODY_NAMES),
        "legal_foot_body_ids": [
            int(value) for value in contract.foot_body_ids
        ],
        "legal_foot_geom_names": [
            native_diag._geom_name(mujoco, model, geom_id)
            for geom_id in contract.foot_geom_ids
        ],
        "legal_foot_geom_ids": [
            int(value) for value in contract.foot_geom_ids
        ],
        "nonfoot_floor_pair_enabled_robot_geom_count": len(
            contract.nonfoot_robot_geom_ids
        ),
        "all_collision_enabled_robot_geoms_floor_pair_enabled": True,
        "foot_floor_penetration_tolerance_m": (
            FOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_floor_penetration_tolerance_m": (
            NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_force_threshold_n": TABLE_CONTACT_FORCE_THRESHOLD_N,
        "continuous_nonfoot_clearance_guard_m": (
            FORMAL_NONFOOT_GROUND_CLEARANCE_GUARD_M
        ),
        "continuous_distance_query_cap_m": (
            FORMAL_GROUND_DISTANCE_QUERY_CAP_M
        ),
        "policy": (
            "exact ankle-roll foot support allowed within penetration "
            "tolerance; every collision-enabled robot geom must pair with "
            "the floor; any native non-foot contact or continuous non-foot "
            "clearance below the guard is rejected"
        ),
    }


def _scan_robot_contacts(
    mujoco: Any,
    model: Any,
    data: Any,
    ball_geom_id: int,
    obstacle_ids: Mapping[int, str],
    events: FittedEvents,
    time_s: float,
    ground_contract: Optional[GroundContactContract] = None,
) -> None:
    robot_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) != 0 and geom_id != ball_geom_id
    }
    five_solid_robot_ids = set(
        _five_solid_robot_geom_ids(
            mujoco,
            model,
            ball_geom_id=ball_geom_id,
            obstacle_geom_ids=tuple(obstacle_ids),
        )
    )
    force6 = np.zeros(6, np.float64)

    def contact_evidence(
        index: int, contact: Any
    ) -> Tuple[float, float]:
        force6.fill(0.0)
        mujoco.mj_contactForce(model, data, index, force6)
        force_n = float(np.linalg.norm(force6[:3]))
        penetration = max(0.0, -float(contact.dist))
        if not (
            math.isfinite(force_n) and math.isfinite(penetration)
        ):
            raise FittedGateError(
                "non-finite robot contact evidence"
            )
        return force_n, penetration

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if ball_geom_id in (g1, g2):
            events.native_ball_contact_count += 1
            continue
        if g1 in obstacle_ids or g2 in obstacle_ids:
            other = g2 if g1 in obstacle_ids else g1
            if other in five_solid_robot_ids:
                force_n, penetration = contact_evidence(
                    index, contact
                )
                if (
                    force_n <= TABLE_CONTACT_FORCE_THRESHOLD_N
                    and penetration <= 0.0
                ):
                    continue
                obstacle = obstacle_ids[
                    g1 if g1 in obstacle_ids else g2
                ]
                events.robot_obstacle_contact_count += 1
                events.robot_obstacle_contact_per_obstacle[obstacle] += 1
                if len(events.robot_obstacle_contacts) < 100:
                    events.robot_obstacle_contacts.append(
                        {
                            "time_s": time_s,
                            "robot_geom": native_diag._geom_name(
                                mujoco, model, other
                            ),
                            "obstacle": obstacle,
                            "force_n": force_n,
                            "penetration_m": penetration,
                            "force_threshold_n": (
                                TABLE_CONTACT_FORCE_THRESHOLD_N
                            ),
                        }
                    )
        elif (
            ground_contract is not None
            and ground_contract.floor_geom_id in (g1, g2)
        ):
            other = (
                g2
                if g1 == ground_contract.floor_geom_id
                else g1
            )
            known_ground_robot_geoms = set(
                ground_contract.foot_geom_ids
            ) | set(ground_contract.nonfoot_robot_geom_ids)
            if other not in known_ground_robot_geoms:
                raise FittedGateError(
                    "floor contacted an unbound robot geometry"
                )
            force_n, penetration = contact_evidence(index, contact)
            if (
                force_n <= TABLE_CONTACT_FORCE_THRESHOLD_N
                and penetration <= 0.0
            ):
                continue
            body_id = int(model.geom_bodyid[other])
            is_foot = body_id in set(ground_contract.foot_body_ids)
            events.ground_contact_count += 1
            if is_foot:
                events.legal_foot_support_contact_count += 1
                events.ground_max_foot_penetration_m = max(
                    events.ground_max_foot_penetration_m,
                    penetration,
                )
                violation = (
                    penetration
                    > FOOT_FLOOR_PENETRATION_TOLERANCE_M
                )
                if violation:
                    events.foot_floor_penetration_violation_count += 1
            else:
                events.ground_max_nonfoot_penetration_m = max(
                    events.ground_max_nonfoot_penetration_m,
                    penetration,
                )
                violation = bool(
                    force_n > TABLE_CONTACT_FORCE_THRESHOLD_N
                    or penetration
                    > NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
                )
                if violation:
                    events.nonfoot_ground_contact_violation_count += 1
            if violation and len(events.ground_contact_violations) < 100:
                events.ground_contact_violations.append(
                    {
                        "time_s": time_s,
                        "robot_geom": native_diag._geom_name(
                            mujoco, model, other
                        ),
                        "robot_body": (
                            mujoco.mj_id2name(
                                model,
                                mujoco.mjtObj.mjOBJ_BODY,
                                body_id,
                            )
                            or f"body_{body_id}"
                        ),
                        "robot_body_is_legal_foot": is_foot,
                        "force_n": force_n,
                        "penetration_m": penetration,
                        "force_threshold_n": (
                            TABLE_CONTACT_FORCE_THRESHOLD_N
                        ),
                        "penetration_tolerance_m": (
                            FOOT_FLOOR_PENETRATION_TOLERANCE_M
                            if is_foot
                            else NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
                        ),
                    }
                )
        elif g1 in robot_ids and g2 in robot_ids:
            if int(model.geom_bodyid[g1]) != int(model.geom_bodyid[g2]):
                events.self_contacts.append(
                    {
                        "time_s": time_s,
                        "geoms": [
                            native_diag._geom_name(mujoco, model, g1),
                            native_diag._geom_name(mujoco, model, g2),
                        ],
                    }
                )


def _record_net_events_on_segment(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    start_time_s: float,
    duration_s: float,
    returned: bool,
    events: FittedEvents,
    profile: Mapping[str, Any],
    aabbs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    ball_radius_m: float,
) -> None:
    if returned and events.net_crossing is None:
        net_x = float(profile["net_x_m"])
        if p0[0] <= net_x < p1[0]:
            alpha = float((net_x - p0[0]) / max(p1[0] - p0[0], EPS))
            crossing = p0 + alpha * (p1 - p0)
            events.net_crossing = {
                "time_s": start_time_s + alpha * duration_s,
                "ball_center_z_m": float(crossing[2]),
                "ball_center_y_m": float(crossing[1]),
                "required_center_z_m": float(
                    profile["ball_center_net_top_z_m"]
                ),
                "clearance_m": float(
                    crossing[2]
                    - float(profile["ball_center_net_top_z_m"])
                ),
                "cleared": bool(
                    crossing[2]
                    > float(profile["ball_center_net_top_z_m"])
                    and abs(crossing[1])
                    <= float(profile["table_half_width_m"])
                ),
            }
    if events.ball_net_collision is None:
        hits: List[Tuple[float, str]] = []
        for name in NET_GEOM_NAMES:
            lo, hi = aabbs[name]
            alpha = segment_expanded_aabb_hit(
                p0, p1, lo, hi, ball_radius_m
            )
            if alpha is not None:
                hits.append((float(alpha), name))
        if hits:
            alpha, name = min(hits)
            events.ball_net_collision = {
                "time_s": start_time_s + alpha * duration_s,
                "obstacle": name,
            }


def arbitrate_table_face_toi(
    *,
    paddle_hit: Optional[SweptFaceHit],
    table_hit: Optional[Tuple[float, np.ndarray]],
    segment_duration_s: float,
) -> Tuple[Optional[Tuple[float, str, Any]], bool]:
    """Choose the earliest table/face time-of-impact, failing closed on ties."""

    duration = float(segment_duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise FittedGateError("surface TOI segment duration must be positive")
    candidates: List[Tuple[float, str, Any]] = []
    if paddle_hit is not None:
        candidates.append((float(paddle_hit.alpha), "paddle", paddle_hit))
    if table_hit is not None:
        candidates.append((float(table_hit[0]), "table", table_hit))
    for alpha, kind, _event in candidates:
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise FittedGateError(f"{kind} surface TOI is outside [0,1]")
    if not candidates:
        return None, False
    rank = {"paddle": 0, "table": 1}
    candidates.sort(key=lambda row: (row[0], rank[row[1]]))
    if (
        len(candidates) > 1
        and abs(candidates[1][0] - candidates[0][0]) * duration
        <= FORMAL_EVENT_TIME_GUARD_S
    ):
        return None, True
    return candidates[0], False


def process_surface_events_chronologically(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    time_s: float,
    dt: float,
    face_before: FaceState,
    face_after: FaceState,
    face_mesh: FaceMesh,
    action: native_diag.ActionSpec,
    venue: VenueParams,
    profile: Mapping[str, Any],
    aabbs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    events: FittedEvents,
    returned: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    List[Tuple[float, float, np.ndarray, np.ndarray]],
]:
    """Apply the earliest fitted impulse, then integrate and inspect remainder.

    A single physics substep may contain an incoming bounce and a later paddle
    hit (or a paddle hit and a later landing).  The former implementation
    spliced the pre-impulse start to a post-impulse end and then applied a fixed
    paddle/table/net order.  This loop instead chooses the earliest state
    transition, records net events only on the chronological prefix, advances
    the post-impulse remainder, and repeats.
    """

    segment_start_time = float(time_s)
    segment_duration = float(dt)
    start_p = np.asarray(p0, np.float64).copy()
    start_v = np.asarray(v0, np.float64).copy()
    start_w = np.asarray(w0, np.float64).copy()
    end_p = np.asarray(p1, np.float64).copy()
    end_v = np.asarray(v1, np.float64).copy()
    end_w = np.asarray(w1, np.float64).copy()
    swept_segments: List[Tuple[float, float, np.ndarray, np.ndarray]] = []
    table_margin_m = venue.ball_radius + FORMAL_SHADOW_CLEARANCE_GUARD_M
    table_near_x_m = float(profile["opponent_near_x_m"]) + table_margin_m
    table_far_x_m = float(profile["opponent_far_x_m"]) - table_margin_m
    table_half_width_m = (
        float(profile["table_half_width_m"]) - table_margin_m
    )
    if (
        table_near_x_m >= table_far_x_m
        or table_half_width_m <= 0.0
    ):
        raise FittedGateError(
            "ball-radius-eroded table footprint is empty"
        )

    for _iteration in range(8):
        elapsed_fraction = float(
            np.clip((segment_start_time - time_s) / max(dt, EPS), 0.0, 1.0)
        )
        segment_face_start = interpolate_face_state(
            face_before, face_after, elapsed_fraction
        )
        paddle_hit = None
        if not returned:
            paddle_hit = swept_selected_face_intersection(
                ball_start_m=start_p,
                ball_end_m=end_p,
                ball_velocity_start_mps=start_v,
                ball_velocity_end_mps=end_v,
                face_start=segment_face_start,
                face_end=face_after,
                mesh=face_mesh,
                ball_radius_m=venue.ball_radius,
            )
        table_hit = swept_table_crossing(
            start_p,
            end_p,
            start_v,
            center_surface_z_m=(
                float(profile["table_surface_z_m"]) + venue.ball_radius
            ),
            near_x_m=table_near_x_m,
            far_x_m=table_far_x_m,
            half_width_m=table_half_width_m,
        )
        choice, ambiguous = arbitrate_table_face_toi(
            paddle_hit=paddle_hit,
            table_hit=table_hit,
            segment_duration_s=segment_duration,
        )
        if choice is None and not ambiguous:
            swept_segments.append(
                (
                    segment_start_time,
                    segment_duration,
                    start_p.copy(),
                    end_p.copy(),
                )
            )
            _record_net_events_on_segment(
                p0=start_p,
                p1=end_p,
                start_time_s=segment_start_time,
                duration_s=segment_duration,
                returned=returned,
                events=events,
                profile=profile,
                aabbs=aabbs,
                ball_radius_m=venue.ball_radius,
            )
            return end_p, end_v, end_w, returned, swept_segments

        if ambiguous:
            events.event_order_violations.append(
                "paddle_and_table_events_not_time_separable"
            )
            swept_segments.append(
                (
                    segment_start_time,
                    segment_duration,
                    start_p.copy(),
                    end_p.copy(),
                )
            )
            return end_p, end_v, end_w, returned, swept_segments
        assert choice is not None
        alpha, kind, event = choice
        event_time = segment_start_time + alpha * segment_duration
        event_position = start_p + alpha * (end_p - start_p)
        event_velocity = start_v + alpha * (end_v - start_v)
        event_spin = start_w + alpha * (end_w - start_w)
        if alpha > EPS:
            swept_segments.append(
                (
                    segment_start_time,
                    alpha * segment_duration,
                    start_p.copy(),
                    event_position.copy(),
                )
            )
        _record_net_events_on_segment(
            p0=start_p,
            p1=event_position,
            start_time_s=segment_start_time,
            duration_s=alpha * segment_duration,
            returned=returned,
            events=events,
            profile=profile,
            aabbs=aabbs,
            ball_radius_m=venue.ball_radius,
        )

        if kind == "paddle":
            hit: SweptFaceHit = event
            if (
                not np.isfinite(hit.normal_w).all()
                or abs(float(np.linalg.norm(hit.normal_w)) - 1.0) > 1.0e-8
                or float(hit.normal_w[0]) <= FORMAL_MIN_RETURN_NORMAL_X
            ):
                events.event_order_violations.append(
                    "selected_face_return_normal_not_positive_world_x"
                )
                return end_p, end_v, end_w, returned, swept_segments
            e_eff = venue.paddle_g1 * math.exp(
                venue.paddle_g2 * abs(hit.relative_normal_speed_mps)
            )
            fitted = fitted_contact(
                event_velocity,
                hit.face_point_velocity_mps,
                hit.normal_w,
                event_spin,
                e_eff=e_eff,
                a_t=venue.paddle_a_t,
                b_t=venue.paddle_b_t,
                mu=venue.paddle_mu,
            )
            next_p = hit.ball_center_m + 1.0e-7 * hit.normal_w
            events.paddle_impulse_count += 1
            events.paddle_contact = {
                "time_s": event_time,
                "ball_center_m": hit.ball_center_m.tolist(),
                "face_point_m": hit.face_point_m.tolist(),
                "face_point_local_m": hit.face_point_local_m.tolist(),
                "face_triangle_index": hit.triangle_index,
                "face_edge_clearance_m": hit.edge_clearance_m,
                "required_face_edge_clearance_m": (
                    venue.ball_radius + FORMAL_FACE_EDGE_GUARD_M
                ),
                "face_mesh_sha256": face_mesh.sha256,
                "selected_face_sign": action.mount_normal_sign,
                "selected_face_return_normal_w": hit.normal_w.tolist(),
                "selected_face_return_normal_x_margin": float(
                    hit.normal_w[0] - FORMAL_MIN_RETURN_NORMAL_X
                ),
                "relative_normal_speed_mps": hit.relative_normal_speed_mps,
                "face_point_velocity_mps": (
                    hit.face_point_velocity_mps.tolist()
                ),
                "velocity_minus_mps": event_velocity.tolist(),
                "velocity_plus_mps": fitted["velocity_plus_mps"].tolist(),
                "spin_minus_radps": event_spin.tolist(),
                "spin_plus_radps": fitted["spin_plus_radps"].tolist(),
                "e_eff": e_eff,
                "contact_model_sha256": CONTACT_MODEL_SHA256,
            }
            next_v = fitted["velocity_plus_mps"]
            next_w = fitted["spin_plus_radps"]
            returned = True
        else:
            _alpha, point = event
            returned_before_event = bool(returned)
            fitted = fitted_contact(
                event_velocity,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                event_spin,
                e_eff=venue.table_e,
                a_t=venue.table_a_t,
                b_t=venue.table_b_t,
                mu=venue.table_mu,
            )
            next_p = np.asarray(point, np.float64) + np.asarray(
                (0.0, 0.0, 1.0e-7)
            )
            next_v = fitted["velocity_plus_mps"]
            next_w = fitted["spin_plus_radps"]
            events.table_contacts.append(
                {
                    "time_s": event_time,
                    "ball_center_m": np.asarray(
                        point, np.float64
                    ).tolist(),
                    "normal_w": [0.0, 0.0, 1.0],
                    "returned_before_event": returned_before_event,
                    "eroded_footprint_margin_m": table_margin_m,
                }
            )
            if returned:
                if events.first_landing is None:
                    events.first_landing = {
                        "time_s": event_time,
                        "ball_center_xy_m": np.asarray(point)[:2].tolist(),
                        "ball_center_z_m": float(np.asarray(point)[2]),
                        "velocity_minus_mps": event_velocity.tolist(),
                        "authority": "venue_fitted_table_impulse",
                    }
                events.return_table_bounces += 1
                events.return_table_bounce_times_s.append(event_time)
            else:
                events.incoming_table_bounces += 1
                events.incoming_table_bounce_times_s.append(event_time)

        remainder = (1.0 - alpha) * segment_duration
        if remainder <= EPS:
            return (
                np.asarray(next_p, np.float64),
                np.asarray(next_v, np.float64),
                np.asarray(next_w, np.float64),
                returned,
                swept_segments,
            )
        segment_start_time = event_time
        segment_duration = remainder
        start_p = np.asarray(next_p, np.float64)
        start_v = np.asarray(next_v, np.float64)
        start_w = np.asarray(next_w, np.float64)
        end_p, end_v = advance_fitted_flight(
            start_p, start_v, start_w, segment_duration, venue
        )
        end_w = start_w.copy()

    events.event_order_violations.append(
        "too_many_fitted_surface_events_in_one_substep"
    )
    swept_segments.append(
        (
            segment_start_time,
            segment_duration,
            start_p.copy(),
            end_p.copy(),
        )
    )
    return end_p, end_v, end_w, returned, swept_segments


def _point_aabb_distance(
    point: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    outside = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    return float(np.linalg.norm(outside))


def _segment_intersects_inflated_aabb(
    start: Sequence[float],
    end: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    inflation_m: float,
) -> bool:
    """Exact line-segment slab test for the conservative geom sweep."""

    p0 = np.asarray(start, np.float64)
    p1 = np.asarray(end, np.float64)
    lower = np.asarray(lo, np.float64) - float(inflation_m)
    upper = np.asarray(hi, np.float64) + float(inflation_m)
    if (
        p0.shape != (3,)
        or p1.shape != (3,)
        or lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(
            np.concatenate((p0, p1, lower, upper))
        ).all()
        or not math.isfinite(float(inflation_m))
        or float(inflation_m) < 0.0
        or np.any(upper < lower)
    ):
        raise FittedGateError("invalid swept-sphere/AABB input")
    delta = p1 - p0
    enter, exit_ = 0.0, 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-15:
            if p0[axis] < lower[axis] or p0[axis] > upper[axis]:
                return False
            continue
        inverse = 1.0 / float(delta[axis])
        first = (float(lower[axis]) - float(p0[axis])) * inverse
        second = (float(upper[axis]) - float(p0[axis])) * inverse
        if first > second:
            first, second = second, first
        enter = max(enter, first)
        exit_ = min(exit_, second)
        if enter > exit_:
            return False
    return True


def scan_five_solid_robot_sweep(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    robot_geom_ids: Sequence[int],
    centers_before: np.ndarray,
    obstacle_aabbs: Mapping[
        str, Tuple[np.ndarray, np.ndarray]
    ],
) -> Dict[str, Any]:
    """Conservatively sweep every teacher robot geom over one substep."""

    ids = np.asarray(robot_geom_ids, np.int64)
    centers_before = np.asarray(centers_before, np.float64)
    centers_after = np.asarray(data.geom_xpos[ids], np.float64)
    if (
        centers_before.shape != (ids.size, 3)
        or centers_after.shape != centers_before.shape
        or not np.isfinite(centers_before).all()
        or not np.isfinite(centers_after).all()
    ):
        raise FittedGateError(
            "five-solid swept robot geom centers are invalid"
        )
    expected_names = table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    if tuple(obstacle_aabbs) != expected_names:
        raise FittedGateError(
            "five-solid swept obstacle order is not exact"
        )
    per_obstacle = {name: 0 for name in expected_names}
    hit_count = 0
    first_hit: Optional[Dict[str, Any]] = None
    for row_index, geom_id in enumerate(ids):
        radius = float(model.geom_rbound[int(geom_id)])
        if not math.isfinite(radius) or radius < 0.0:
            raise FittedGateError(
                "teacher robot geom has invalid MuJoCo rbound"
            )
        for obstacle_name in expected_names:
            lo, hi = obstacle_aabbs[obstacle_name]
            if not _segment_intersects_inflated_aabb(
                centers_before[row_index],
                centers_after[row_index],
                lo,
                hi,
                radius,
            ):
                continue
            hit_count += 1
            per_obstacle[obstacle_name] += 1
            if first_hit is None:
                first_hit = {
                    "robot_geom": native_diag._geom_name(
                        mujoco, model, int(geom_id)
                    ),
                    "obstacle": obstacle_name,
                    "geom_rbound_m": radius,
                    "center_before_m": (
                        centers_before[row_index].tolist()
                    ),
                    "center_after_m": centers_after[row_index].tolist(),
                }
    return {
        "hit_count": hit_count,
        "per_obstacle": per_obstacle,
        "first_hit": first_hit,
        "method": FIVE_SOLID_SWEEP_METHOD,
    }


def _geom_below_clearance(
    mujoco: Any,
    model: Any,
    data: Any,
    first_geom: int,
    second_geom: int,
    threshold_m: float,
) -> bool:
    observed = float(
        mujoco.mj_geomDistance(
            model,
            data,
            int(first_geom),
            int(second_geom),
            float(threshold_m),
            None,
        )
    )
    return not math.isfinite(observed) or observed < threshold_m


def _body_descends_from(
    model: Any, body_id: int, ancestor_body_id: int
) -> bool:
    current = int(body_id)
    ancestor = int(ancestor_body_id)
    while current != 0:
        if current == ancestor:
            return True
        current = int(model.body_parentid[current])
    return ancestor == 0


def _joint_to_geom_radius_bound_m(
    model: Any, joint_id: int, geom_id: int
) -> float:
    """Static upper bound from a joint anchor to any point on a descendant geom."""

    joint_body = int(model.jnt_bodyid[joint_id])
    geom_body = int(model.geom_bodyid[geom_id])
    if not _body_descends_from(model, geom_body, joint_body):
        raise FittedGateError(
            f"geom {geom_id} is not below ancestor joint {joint_id}"
        )
    bound = (
        float(np.linalg.norm(np.asarray(model.jnt_pos[joint_id], np.float64)))
        + float(np.linalg.norm(np.asarray(model.geom_pos[geom_id], np.float64)))
        + float(model.geom_rbound[geom_id])
    )
    body = geom_body
    while body != joint_body:
        bound += float(
            np.linalg.norm(np.asarray(model.body_pos[body], np.float64))
        )
        body = int(model.body_parentid[body])
    if not math.isfinite(bound) or bound <= 0.0:
        raise FittedGateError(
            f"nonpositive/nonfinite joint-to-geom radius bound for geom {geom_id}"
        )
    return bound


def build_shadow_kinematic_bounds(
    *,
    mujoco: Any,
    model: Any,
    binding: motion_player.ModelBinding,
    robot_geom_ids: Sequence[int],
) -> Tuple[ShadowGeomMotionBound, ...]:
    """Bind every robot geom to conservative ancestor-joint path lengths."""

    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
    root_matches = np.flatnonzero(
        np.asarray(model.jnt_qposadr, np.int64)
        == int(binding.root_qpos_adr)
    )
    if root_matches.size != 1:
        raise FittedGateError("cannot uniquely bind teacher free-root joint")
    root_joint = int(root_matches[0])
    if int(model.jnt_type[root_joint]) != free_type:
        raise FittedGateError("teacher root qpos does not address a free joint")
    joint_index_by_id: Dict[int, int] = {}
    for index, qpos_address in enumerate(
        np.asarray(binding.joint_qpos_adrs, np.int64)
    ):
        matches = np.flatnonzero(
            np.asarray(model.jnt_qposadr, np.int64)
            == int(qpos_address)
        )
        if matches.size != 1:
            raise FittedGateError(
                f"cannot uniquely bind teacher joint qpos address {qpos_address}"
            )
        joint_id = int(matches[0])
        if joint_id in joint_index_by_id:
            raise FittedGateError("teacher joint identity is duplicated")
        if int(model.jnt_type[joint_id]) != hinge_type:
            raise FittedGateError(
                f"teacher joint {joint_id} is not the required hinge type"
            )
        joint_index_by_id[joint_id] = index

    output: List[ShadowGeomMotionBound] = []
    for geom_id_raw in robot_geom_ids:
        geom_id = int(geom_id_raw)
        body = int(model.geom_bodyid[geom_id])
        ancestors: List[int] = []
        while body != 0:
            joint_start = int(model.body_jntadr[body])
            joint_count = int(model.body_jntnum[body])
            ancestors.extend(
                range(joint_start, joint_start + joint_count)
            )
            body = int(model.body_parentid[body])
        if root_joint not in ancestors:
            raise FittedGateError(
                f"robot geom {geom_id} is not below the teacher free root"
            )
        hinge_terms: List[Tuple[int, float]] = []
        slide_indices: List[int] = []
        for joint_id in ancestors:
            if joint_id == root_joint:
                continue
            index = joint_index_by_id.get(joint_id)
            if index is None:
                raise FittedGateError(
                    f"robot geom {geom_id} has an unbound ancestor joint "
                    f"{joint_id}"
                )
            joint_type = int(model.jnt_type[joint_id])
            if joint_type == hinge_type:
                hinge_terms.append(
                    (
                        index,
                        _joint_to_geom_radius_bound_m(
                            model, joint_id, geom_id
                        ),
                    )
                )
            elif joint_type == slide_type:
                raise FittedGateError(
                    f"teacher geom {geom_id} has unsupported slide ancestry"
                )
            else:
                raise FittedGateError(
                    f"unsupported ancestor joint type for geom {geom_id}"
                )
        output.append(
            ShadowGeomMotionBound(
                geom_id=geom_id,
                root_rotation_radius_m=_joint_to_geom_radius_bound_m(
                    model, root_joint, geom_id
                ),
                hinge_terms=tuple(hinge_terms),
                slide_indices=tuple(slide_indices),
            )
        )
    if len(output) != len(robot_geom_ids):
        raise FittedGateError("shadow kinematic bound coverage is incomplete")
    return tuple(output)


def _shortest_quaternion_angle_rad(
    first_wxyz: Sequence[float], second_wxyz: Sequence[float]
) -> float:
    first = np.asarray(first_wxyz, np.float64)
    second = np.asarray(second_wxyz, np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or first_norm <= EPS
        or second_norm <= EPS
    ):
        raise FittedGateError("teacher root quaternion is invalid")
    dot = abs(float((first / first_norm) @ (second / second_norm)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def robot_surface_path_bound_m(
    *,
    first_state: Mapping[str, Any],
    second_state: Mapping[str, Any],
    geom_bounds: Sequence[ShadowGeomMotionBound],
) -> float:
    """Upper-bound every robot-geom surface path over one interpolation span."""

    root_translation = float(
        np.linalg.norm(
            np.asarray(second_state["root_pos"], np.float64)
            - np.asarray(first_state["root_pos"], np.float64)
        )
    )
    root_rotation = _shortest_quaternion_angle_rad(
        first_state["root_quat"], second_state["root_quat"]
    )
    first_joint = np.asarray(first_state["joint_pos"], np.float64)
    second_joint = np.asarray(second_state["joint_pos"], np.float64)
    if first_joint.shape != second_joint.shape:
        raise FittedGateError("teacher joint-state shape changed")
    joint_variation = np.abs(second_joint - first_joint)
    if (
        not math.isfinite(root_translation)
        or not np.isfinite(joint_variation).all()
    ):
        raise FittedGateError("teacher surface-motion inputs are nonfinite")
    maximum = 0.0
    for geom in geom_bounds:
        bound = (
            root_translation
            + root_rotation * geom.root_rotation_radius_m
        )
        bound += sum(
            float(joint_variation[index]) * radius
            for index, radius in geom.hinge_terms
        )
        bound += sum(
            float(joint_variation[index])
            for index in geom.slide_indices
        )
        maximum = max(maximum, bound)
    if not math.isfinite(maximum):
        raise FittedGateError("robot surface path bound is nonfinite")
    return maximum


def continuous_floor_interval_lower_bound_m(
    *,
    distance_lower_m: float,
    distance_midpoint_m: float,
    distance_upper_m: float,
    surface_bound_lower_to_mid_m: float,
    surface_bound_mid_to_upper_m: float,
) -> float:
    """Certified signed-distance lower bound for one fixed-floor interval."""

    values = (
        distance_lower_m,
        distance_midpoint_m,
        distance_upper_m,
        surface_bound_lower_to_mid_m,
        surface_bound_mid_to_upper_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise FittedGateError(
            "continuous floor certificate contains NaN/Inf"
        )
    if (
        surface_bound_lower_to_mid_m < 0.0
        or surface_bound_mid_to_upper_m < 0.0
    ):
        raise FittedGateError(
            "continuous floor surface bound must be nonnegative"
        )
    lower_half = (
        max(distance_lower_m, distance_midpoint_m)
        - surface_bound_lower_to_mid_m
    )
    upper_half = (
        max(distance_midpoint_m, distance_upper_m)
        - surface_bound_mid_to_upper_m
    )
    return float(min(lower_half, upper_half))


def run_continuous_ground_probe(
    *,
    mujoco: Any,
    model: Any,
    probe_data: Any,
    binding: motion_player.ModelBinding,
    clip: motion_player.MotionClip,
    wait_s: float,
    teacher_rate: float,
    start_time_s: float,
    duration_s: float,
    ground_contract: GroundContactContract,
    ground_geom_motion_bounds: Sequence[ShadowGeomMotionBound],
    events: FittedEvents,
) -> None:
    """Prove floor clearance between physics samples for every robot geom."""

    bounds_by_geom = {
        int(bound.geom_id): bound for bound in ground_geom_motion_bounds
    }
    ground_geom_ids = (
        tuple(ground_contract.foot_geom_ids)
        + tuple(ground_contract.nonfoot_robot_geom_ids)
    )
    if (
        set(bounds_by_geom) != set(ground_geom_ids)
        or len(bounds_by_geom) != len(ground_geom_ids)
    ):
        raise FittedGateError(
            "continuous ground subject differs from its kinematic bounds"
        )
    try:
        sample_alphas, _max_ball_bound, _max_robot_bound = (
            adaptive_shadow_sample_alphas(
                clip=clip,
                wait_s=wait_s,
                teacher_rate=teacher_rate,
                start_time_s=start_time_s,
                duration_s=duration_s,
                start_ball_position_m=np.zeros(3, np.float64),
                end_ball_position_m=np.zeros(3, np.float64),
                geom_bounds=ground_geom_motion_bounds,
            )
        )
    except FittedGateError as exc:
        failure = f"ground_relative_motion_certificate_failed:{exc}"
        events.ground_shadow_certificate_failure = failure
        events.event_order_violations.append(
            failure
        )
        return

    state_cache: Dict[float, Mapping[str, Any]] = {}
    distance_cache: Dict[float, Dict[int, float]] = {}

    def sample(alpha: float) -> Tuple[Mapping[str, Any], Dict[int, float]]:
        key = float(alpha)
        cached_state = state_cache.get(key)
        cached_distances = distance_cache.get(key)
        if cached_state is not None and cached_distances is not None:
            return cached_state, cached_distances
        sample_time = start_time_s + key * duration_s
        state = retimed_teacher_state(
            clip,
            world_time_s=sample_time,
            pre_swing_wait_s=wait_s,
            teacher_rate=teacher_rate,
        )
        native_diag._set_teacher_state(
            mujoco, model, probe_data, binding, state
        )
        mujoco.mj_forward(model, probe_data)
        distances: Dict[int, float] = {}
        for geom_id in ground_geom_ids:
            sphere_lower = float(
                probe_data.geom_xpos[geom_id][2]
                - model.geom_rbound[geom_id]
            )
            if (
                math.isfinite(sphere_lower)
                and sphere_lower
                > FORMAL_GROUND_DISTANCE_QUERY_CAP_M
            ):
                distance = FORMAL_GROUND_DISTANCE_QUERY_CAP_M
            else:
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        probe_data,
                        int(ground_contract.floor_geom_id),
                        int(geom_id),
                        FORMAL_GROUND_DISTANCE_QUERY_CAP_M,
                        None,
                    )
                )
            if not math.isfinite(distance):
                raise FittedGateError(
                    "continuous ground distance is NaN/Inf"
                )
            distances[int(geom_id)] = distance
        state_cache[key] = state
        distance_cache[key] = distances
        events.ground_shadow_probe_samples += 1
        return state, distances

    foot_geom_set = set(ground_contract.foot_geom_ids)
    for lower_alpha, upper_alpha in zip(
        sample_alphas, sample_alphas[1:]
    ):
        midpoint_alpha = 0.5 * (lower_alpha + upper_alpha)
        lower_state, lower_distances = sample(lower_alpha)
        midpoint_state, midpoint_distances = sample(midpoint_alpha)
        upper_state, upper_distances = sample(upper_alpha)
        events.ground_shadow_certificate_intervals += 1
        for geom_id in ground_geom_ids:
            bound = bounds_by_geom[geom_id]
            lower_to_mid = robot_surface_path_bound_m(
                first_state=lower_state,
                second_state=midpoint_state,
                geom_bounds=(bound,),
            )
            mid_to_upper = robot_surface_path_bound_m(
                first_state=midpoint_state,
                second_state=upper_state,
                geom_bounds=(bound,),
            )
            lower_distance = lower_distances[geom_id]
            midpoint_distance = midpoint_distances[geom_id]
            upper_distance = upper_distances[geom_id]
            for first_distance, second_distance, path_bound in (
                (
                    lower_distance,
                    midpoint_distance,
                    lower_to_mid,
                ),
                (
                    midpoint_distance,
                    upper_distance,
                    mid_to_upper,
                ),
            ):
                if (
                    first_distance
                    < FORMAL_GROUND_DISTANCE_QUERY_CAP_M
                    and second_distance
                    < FORMAL_GROUND_DISTANCE_QUERY_CAP_M
                    and abs(first_distance - second_distance)
                    > path_bound + FORMAL_GROUND_DISTANCE_NUMERIC_TOL_M
                ):
                    raise FittedGateError(
                        "continuous ground distance/path bound "
                        "inconsistency"
                    )
            certified_lower = continuous_floor_interval_lower_bound_m(
                distance_lower_m=lower_distance,
                distance_midpoint_m=midpoint_distance,
                distance_upper_m=upper_distance,
                surface_bound_lower_to_mid_m=lower_to_mid,
                surface_bound_mid_to_upper_m=mid_to_upper,
            )
            is_foot = geom_id in foot_geom_set
            threshold = (
                -FOOT_FLOOR_PENETRATION_TOLERANCE_M
                if is_foot
                else FORMAL_NONFOOT_GROUND_CLEARANCE_GUARD_M
            )
            if is_foot:
                prior = events.ground_shadow_min_foot_lower_bound_m
                events.ground_shadow_min_foot_lower_bound_m = (
                    certified_lower
                    if prior is None
                    else min(prior, certified_lower)
                )
            else:
                prior = events.ground_shadow_min_nonfoot_lower_bound_m
                events.ground_shadow_min_nonfoot_lower_bound_m = (
                    certified_lower
                    if prior is None
                    else min(prior, certified_lower)
                )
            if certified_lower >= threshold:
                continue
            target = (
                events.shadow_foot_floor_penetration_violations
                if is_foot
                else events.shadow_nonfoot_ground_near_contacts
            )
            if len(target) < 100:
                target.append(
                    {
                        "start_time_s": (
                            start_time_s
                            + lower_alpha * duration_s
                        ),
                        "end_time_s": (
                            start_time_s
                            + upper_alpha * duration_s
                        ),
                        "robot_geom": native_diag._geom_name(
                            mujoco, model, geom_id
                        ),
                        "robot_body_is_legal_foot": is_foot,
                        "certified_lower_bound_m": certified_lower,
                        "required_lower_bound_m": threshold,
                        "sampled_distances_m": [
                            lower_distance,
                            midpoint_distance,
                            upper_distance,
                        ],
                        "surface_path_bounds_m": [
                            lower_to_mid,
                            mid_to_upper,
                        ],
                    }
                )
    events.ground_shadow_covered_duration_s += float(duration_s)


def adaptive_shadow_sample_alphas(
    *,
    clip: motion_player.MotionClip,
    wait_s: float,
    teacher_rate: float = 1.0,
    start_time_s: float,
    duration_s: float,
    start_ball_position_m: np.ndarray,
    end_ball_position_m: np.ndarray,
    geom_bounds: Sequence[ShadowGeomMotionBound],
) -> Tuple[List[float], float, float]:
    """Return samples whose between-sample relative motion is certified small.

    Every motion-frame knot is an interval boundary.  Within one such span the
    teacher uses linear root/joint interpolation and quaternion slerp, so the
    conservative kinematic path bound scales with an equal-time subdivision.
    """

    if duration_s <= 0.0:
        raise FittedGateError("shadow segment duration must be positive")
    rate = _finite(teacher_rate, "shadow teacher_rate", positive=True)
    ball_distance = float(
        np.linalg.norm(
            np.asarray(end_ball_position_m, np.float64)
            - np.asarray(start_ball_position_m, np.float64)
        )
    )
    base_count = max(
        1,
        int(math.ceil(duration_s / FORMAL_SHADOW_MAX_DT_S)),
        int(math.ceil(ball_distance / FORMAL_SHADOW_MAX_BALL_STEP_M)),
    )
    breakpoints = {index / base_count for index in range(base_count + 1)}
    segment_end = start_time_s + duration_s
    first_frame = max(
        0,
        int(
            math.ceil(
                (start_time_s - wait_s) * clip.fps * rate
            )
        ),
    )
    last_frame = min(
        clip.n_frames - 1,
        int(
            math.floor(
                (segment_end - wait_s) * clip.fps * rate
            )
        ),
    )
    for frame in range(first_frame, last_frame + 1):
        knot_time = wait_s + frame / (clip.fps * rate)
        if start_time_s < knot_time < segment_end:
            breakpoints.add(
                float((knot_time - start_time_s) / duration_s)
            )
    ordered = sorted(breakpoints)
    output = [ordered[0]]
    max_ball_bound = 0.0
    max_robot_bound = 0.0
    state_cache: Dict[float, Mapping[str, Any]] = {}

    def state(alpha: float) -> Mapping[str, Any]:
        if alpha not in state_cache:
            state_cache[alpha] = retimed_teacher_state(
                clip,
                world_time_s=start_time_s + alpha * duration_s,
                pre_swing_wait_s=wait_s,
                teacher_rate=rate,
            )
        return state_cache[alpha]

    for base_lower, base_upper in zip(ordered, ordered[1:]):
        pending = [(base_lower, base_upper)]
        accepted: List[Tuple[float, float, float]] = []
        while pending:
            lower, upper = pending.pop()
            robot_bound = robot_surface_path_bound_m(
                first_state=state(lower),
                second_state=state(upper),
                geom_bounds=geom_bounds,
            )
            ball_bound = ball_distance * (upper - lower)
            if (
                robot_bound
                <= FORMAL_SHADOW_MAX_ROBOT_SURFACE_STEP_M
            ):
                accepted.append((lower, upper, robot_bound))
                max_ball_bound = max(max_ball_bound, ball_bound)
                max_robot_bound = max(max_robot_bound, robot_bound)
                continue
            midpoint = 0.5 * (lower + upper)
            if (
                midpoint <= lower
                or midpoint >= upper
                or len(output) + len(accepted) + len(pending) + 2
                > 4097
            ):
                raise FittedGateError(
                    "shadow adaptive subdivision limit exceeded"
                )
            pending.append((midpoint, upper))
            pending.append((lower, midpoint))
        for lower, upper, _bound in sorted(accepted):
            if abs(output[-1] - lower) > 1.0e-12:
                raise FittedGateError(
                    "shadow adaptive intervals are not contiguous"
                )
            output.append(upper)
    if (
        max_ball_bound
        + max_robot_bound
        >= FORMAL_SHADOW_CLEARANCE_GUARD_M
        or 2.0 * max_robot_bound
        >= FORMAL_ROBOT_OBSTACLE_GUARD_M
    ):
        raise FittedGateError(
            "shadow relative-motion certificate has no positive clearance "
            "margin"
        )
    return output, max_ball_bound, max_robot_bound


def _geom_pair_enabled(
    mujoco: Any, model: Any, first_geom: int, second_geom: int
) -> bool:
    """Replicate MuJoCo's collision mask/weld/parent/exclude filtering."""

    first_type = int(model.geom_contype[first_geom])
    first_affinity = int(model.geom_conaffinity[first_geom])
    second_type = int(model.geom_contype[second_geom])
    second_affinity = int(model.geom_conaffinity[second_geom])
    if not (
        (first_type & second_affinity)
        or (second_type & first_affinity)
    ):
        return False
    first_weld = int(model.body_weldid[model.geom_bodyid[first_geom]])
    second_weld = int(model.body_weldid[model.geom_bodyid[second_geom]])
    if first_weld == second_weld:
        return False
    parent_filtering = not (
        int(model.opt.disableflags)
        & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    )
    if parent_filtering:
        first_parent_weld = int(
            model.body_weldid[model.body_parentid[first_weld]]
        )
        second_parent_weld = int(
            model.body_weldid[model.body_parentid[second_weld]]
        )
        if (
            first_weld != 0
            and first_weld == second_parent_weld
        ) or (
            second_weld != 0
            and second_weld == first_parent_weld
        ):
            return False
    first_body = int(model.geom_bodyid[first_geom])
    second_body = int(model.geom_bodyid[second_geom])
    lower, upper = sorted((first_body, second_body))
    signature = (lower << 16) + upper
    return not bool(
        (
            np.asarray(model.exclude_signature[: int(model.nexclude)])
            == signature
        ).any()
    )


def _five_solid_robot_geom_ids(
    mujoco: Any,
    model: Any,
    *,
    ball_geom_id: int,
    obstacle_geom_ids: Sequence[int],
) -> Tuple[int, ...]:
    """Select only robot geoms that MuJoCo can pair with a safety solid.

    Visual/collision-disabled geoms can have large ``geom_rbound`` values.
    Sweeping them would reject a physically legal teacher trajectory even
    though MuJoCo can never generate a contact for that geom.  The continuous
    guard therefore uses the same masks, welded-body, parent, and explicit
    exclude rules as the discrete-contact guard.
    """

    obstacles = tuple(int(geom_id) for geom_id in obstacle_geom_ids)
    if not obstacles:
        raise FittedGateError(
            "five-solid robot selection has no obstacle geoms"
        )
    return tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if geom_id != int(ball_geom_id)
        and int(model.geom_bodyid[geom_id]) != 0
        and any(
            _geom_pair_enabled(
                mujoco, model, geom_id, obstacle_geom_id
            )
            for obstacle_geom_id in obstacles
        )
    )


def _matches_authorized_paddle_contact(
    *,
    sample_time_s: float,
    ball_position_m: np.ndarray,
    events: FittedEvents,
) -> bool:
    contact = events.paddle_contact
    if contact is None:
        return False
    return bool(
        abs(sample_time_s - float(contact["time_s"]))
        <= FORMAL_SHADOW_EVENT_MATCH_S
        and float(
            np.linalg.norm(
                np.asarray(ball_position_m, np.float64)
                - np.asarray(contact["ball_center_m"], np.float64)
            )
        )
        <= FORMAL_SHADOW_EVENT_POSITION_MATCH_M
        and float(contact["selected_face_return_normal_w"][0])
        > FORMAL_MIN_RETURN_NORMAL_X
        and float(contact["face_edge_clearance_m"])
        > float(contact["required_face_edge_clearance_m"])
    )


def _matches_authorized_table_contact(
    *,
    sample_time_s: float,
    ball_position_m: np.ndarray,
    events: FittedEvents,
    table_aabb: Tuple[np.ndarray, np.ndarray],
    ball_radius_m: float,
) -> bool:
    position = np.asarray(ball_position_m, np.float64)
    table_lo = np.asarray(table_aabb[0], np.float64)
    table_hi = np.asarray(table_aabb[1], np.float64)
    expected_margin_m = (
        float(ball_radius_m) + FORMAL_SHADOW_CLEARANCE_GUARD_M
    )
    for contact in events.table_contacts:
        center = np.asarray(contact["ball_center_m"], np.float64)
        if (
            abs(sample_time_s - float(contact["time_s"]))
            > FORMAL_SHADOW_EVENT_MATCH_S
        ):
            continue
        if (
            float(
                np.linalg.norm(
                    position
                    - center
                )
            )
            > FORMAL_SHADOW_EVENT_POSITION_MATCH_M
        ):
            continue
        if not np.array_equal(
            np.asarray(contact["normal_w"], np.float64),
            np.asarray((0.0, 0.0, 1.0), np.float64),
        ):
            continue
        if (
            float(contact["eroded_footprint_margin_m"])
            != expected_margin_m
            or not (
                table_lo[0] + expected_margin_m
                <= center[0]
                <= table_hi[0] - expected_margin_m
                and table_lo[1] + expected_margin_m
                <= center[1]
                <= table_hi[1] - expected_margin_m
            )
            or abs(center[2] - (table_hi[2] + ball_radius_m))
            > FORMAL_SHADOW_CLEARANCE_GUARD_M
        ):
            continue
        return True
    return False


def run_shadow_forbidden_geometry_probe(
    *,
    mujoco: Any,
    model: Any,
    probe_data: Any,
    binding: motion_player.ModelBinding,
    clip: motion_player.MotionClip,
    wait_s: float,
    teacher_rate: float,
    ball_qpos_address: int,
    ball_dof_address: int,
    ball_geom_id: int,
    robot_collision_geom_ids: Sequence[int],
    five_solid_robot_geom_ids: Sequence[int],
    robot_geom_motion_bounds: Sequence[ShadowGeomMotionBound],
    self_collision_pairs: Sequence[Tuple[int, int]],
    ball_obstacle_ids: Mapping[int, str],
    ball_obstacle_aabbs: Mapping[
        str, Tuple[np.ndarray, np.ndarray]
    ],
    robot_safety_obstacle_ids: Mapping[int, str],
    robot_safety_obstacle_aabbs: Mapping[
        str, Tuple[np.ndarray, np.ndarray]
    ],
    swept_segments: Sequence[
        Tuple[float, float, np.ndarray, np.ndarray]
    ],
    events: FittedEvents,
    ball_radius_m: float,
) -> None:
    """No-response high-rate geometry probe for every authoritative segment.

    Native ball contact remains disabled in the authoritative data/model.  This
    separate MjData is only placed along the already-computed ball trajectory;
    `mj_geomDistance` never applies an impulse.  The probe covers all collision
    robot geoms (including hand, handle, rim mesh, torso and feet) and every
    real table/net/post ball obstacle.  The robot separately sees the exact
    five-solid ActionBall safety assembly through only geoms whose MuJoCo
    collision pair is enabled against those solids, including the under-table
    keepout; the fitted ball never sees that robot-only volume.  Sampling is
    bounded by time and ball travel, then tightened by a static-link-length
    surface-path certificate for every robot geom.  Between samples,
    ball+robot motion is strictly below the clearance guard and two robot
    surfaces move strictly less than the self-clearance guard, so a hidden
    zero-clearance crossing is impossible.
    """

    racket_geom_name = native_diag.RACKET_GEOM_NAME
    robot_obstacle_name_to_id = {
        name: geom_id
        for geom_id, name in robot_safety_obstacle_ids.items()
    }
    self_first_ids = np.asarray(
        [pair[0] for pair in self_collision_pairs], np.int64
    )
    self_second_ids = np.asarray(
        [pair[1] for pair in self_collision_pairs], np.int64
    )
    self_pair_radius_bounds = (
        np.asarray(model.geom_rbound, np.float64)[self_first_ids]
        + np.asarray(model.geom_rbound, np.float64)[self_second_ids]
        + FORMAL_ROBOT_OBSTACLE_GUARD_M
    )
    for start_time, duration, start_p, end_p in swept_segments:
        try:
            sample_alphas, max_ball_bound, max_robot_bound = (
                adaptive_shadow_sample_alphas(
                    clip=clip,
                    wait_s=wait_s,
                    teacher_rate=teacher_rate,
                    start_time_s=start_time,
                    duration_s=duration,
                    start_ball_position_m=start_p,
                    end_ball_position_m=end_p,
                    geom_bounds=robot_geom_motion_bounds,
                )
            )
        except FittedGateError as exc:
            events.event_order_violations.append(
                f"shadow_relative_motion_certificate_failed:{exc}"
            )
            return
        events.shadow_certificate_intervals += len(sample_alphas) - 1
        events.shadow_max_ball_path_bound_m = max(
            events.shadow_max_ball_path_bound_m, max_ball_bound
        )
        events.shadow_max_robot_surface_path_bound_m = max(
            events.shadow_max_robot_surface_path_bound_m,
            max_robot_bound,
        )
        for alpha in sample_alphas:
            sample_time = start_time + alpha * duration
            ball_position = start_p + alpha * (end_p - start_p)
            teacher = retimed_teacher_state(
                clip,
                world_time_s=sample_time,
                pre_swing_wait_s=wait_s,
                teacher_rate=teacher_rate,
            )
            native_diag._set_teacher_state(
                mujoco, model, probe_data, binding, teacher
            )
            probe_data.qpos[
                ball_qpos_address : ball_qpos_address + 3
            ] = ball_position
            probe_data.qpos[
                ball_qpos_address + 3 : ball_qpos_address + 7
            ] = np.asarray((1.0, 0.0, 0.0, 0.0))
            probe_data.qvel[
                ball_dof_address : ball_dof_address + 6
            ] = 0.0
            mujoco.mj_forward(model, probe_data)
            events.shadow_probe_samples += 1

            # The fitted ball is analytic and collision-disabled, so every
            # robot-body geom remains forbidden except the authorized face.
            for robot_geom in robot_collision_geom_ids:
                center = np.asarray(
                    probe_data.geom_xpos[robot_geom], np.float64
                )
                radius = float(model.geom_rbound[robot_geom])
                if (
                    float(np.linalg.norm(center - ball_position))
                    > radius
                    + ball_radius_m
                    + FORMAL_SHADOW_CLEARANCE_GUARD_M
                ):
                    continue
                if not _geom_below_clearance(
                    mujoco,
                    model,
                    probe_data,
                    ball_geom_id,
                    robot_geom,
                    FORMAL_SHADOW_CLEARANCE_GUARD_M,
                ):
                    continue
                name = native_diag._geom_name(
                    mujoco, model, robot_geom
                )
                selected_match = (
                    name == racket_geom_name
                    and _matches_authorized_paddle_contact(
                        sample_time_s=sample_time,
                        ball_position_m=ball_position,
                        events=events,
                    )
                )
                if not selected_match and len(events.ball_forbidden_contacts) < 100:
                    events.ball_forbidden_contacts.append(
                        {
                            "time_s": sample_time,
                            "geom": name,
                            "kind": "robot_collision_geom",
                            "clearance_guard_m": (
                                FORMAL_SHADOW_CLEARANCE_GUARD_M
                            ),
                        }
                    )

            for obstacle_id, obstacle_name in ball_obstacle_ids.items():
                lo, hi = ball_obstacle_aabbs[obstacle_name]
                if (
                    _point_aabb_distance(ball_position, lo, hi)
                    > ball_radius_m + FORMAL_SHADOW_CLEARANCE_GUARD_M
                ):
                    continue
                if not _geom_below_clearance(
                    mujoco,
                    model,
                    probe_data,
                    ball_geom_id,
                    obstacle_id,
                    FORMAL_SHADOW_CLEARANCE_GUARD_M,
                ):
                    continue
                table_match = (
                    obstacle_name == TABLE_GEOM_NAME
                    and _matches_authorized_table_contact(
                        sample_time_s=sample_time,
                        ball_position_m=ball_position,
                        events=events,
                        table_aabb=ball_obstacle_aabbs[TABLE_GEOM_NAME],
                        ball_radius_m=ball_radius_m,
                    )
                )
                if not table_match and len(events.ball_forbidden_contacts) < 100:
                    events.ball_forbidden_contacts.append(
                        {
                            "time_s": sample_time,
                            "geom": obstacle_name,
                            "kind": "table_net_or_post",
                            "clearance_guard_m": (
                                FORMAL_SHADOW_CLEARANCE_GUARD_M
                            ),
                        }
                            )

            if self_first_ids.size:
                self_center_distances = np.linalg.norm(
                    np.asarray(probe_data.geom_xpos, np.float64)[
                        self_first_ids
                    ]
                    - np.asarray(probe_data.geom_xpos, np.float64)[
                        self_second_ids
                    ],
                    axis=1,
                )
                self_candidates = np.flatnonzero(
                    self_center_distances <= self_pair_radius_bounds
                )
            else:
                self_candidates = np.empty(0, np.int64)
            for pair_index in self_candidates:
                first_geom = int(self_first_ids[pair_index])
                second_geom = int(self_second_ids[pair_index])
                if not _geom_below_clearance(
                    mujoco,
                    model,
                    probe_data,
                    first_geom,
                    second_geom,
                    FORMAL_ROBOT_OBSTACLE_GUARD_M,
                ):
                    continue
                if len(events.shadow_self_near_contacts) < 100:
                    events.shadow_self_near_contacts.append(
                        {
                            "time_s": sample_time,
                            "geoms": [
                                native_diag._geom_name(
                                    mujoco, model, first_geom
                                ),
                                native_diag._geom_name(
                                    mujoco, model, second_geom
                                ),
                            ],
                            "clearance_guard_m": (
                                FORMAL_ROBOT_OBSTACLE_GUARD_M
                            ),
                        }
                    )

            # Table safety follows MuJoCo's actual collision filter; inert
            # visual geoms must not create an analytic false positive.
            for robot_geom in five_solid_robot_geom_ids:
                center = np.asarray(
                    probe_data.geom_xpos[robot_geom], np.float64
                )
                radius = float(model.geom_rbound[robot_geom])
                for obstacle_name, obstacle_id in (
                    robot_obstacle_name_to_id.items()
                ):
                    lo, hi = robot_safety_obstacle_aabbs[
                        obstacle_name
                    ]
                    if (
                        _point_aabb_distance(center, lo, hi)
                        > radius + FORMAL_ROBOT_OBSTACLE_GUARD_M
                    ):
                        continue
                    if _geom_below_clearance(
                        mujoco,
                        model,
                        probe_data,
                        robot_geom,
                        obstacle_id,
                        FORMAL_ROBOT_OBSTACLE_GUARD_M,
                    ):
                        if (
                            len(events.shadow_robot_obstacle_near_contacts)
                            < 100
                        ):
                            events.shadow_robot_obstacle_near_contacts.append(
                                {
                                    "time_s": sample_time,
                                    "robot_geom": native_diag._geom_name(
                                        mujoco, model, robot_geom
                                    ),
                                    "obstacle": obstacle_name,
                                    "clearance_guard_m": (
                                        FORMAL_ROBOT_OBSTACLE_GUARD_M
                                    ),
                                }
                            )
        events.shadow_covered_duration_s += float(duration)


def _action_failure_reasons(
    events: FittedEvents,
    *,
    expected_contact_time_s: float,
    contact_time_tolerance_s: float,
    contact_position_w_m: Sequence[float],
    contact_position_tolerance_m: float,
    profile: Mapping[str, Any],
    teacher_physical_center_error_m: float,
    reference_site_speed_error_mps: float,
    teacher_site_target_error_m: float,
    teacher_face_normal_angle_error_rad: float,
    teacher_face_velocity_error_mps: float,
    teacher_site_velocity_error_mps: float,
    teacher_angular_velocity_error_radps: float,
    task_landing_aim_w_xy_m: Sequence[float],
) -> List[str]:
    reasons: List[str] = []
    if events.paddle_impulse_count != 1 or events.paddle_contact is None:
        reasons.append("fitted_paddle_impulse_count_not_exactly_one")
    else:
        if abs(events.paddle_contact["time_s"] - expected_contact_time_s) > contact_time_tolerance_s:
            reasons.append("physical_contact_time_mismatch")
        if float(
            np.linalg.norm(
                np.asarray(events.paddle_contact["ball_center_m"])
                - np.asarray(contact_position_w_m)
            )
        ) > contact_position_tolerance_m:
            reasons.append("physical_contact_position_mismatch")
    if teacher_physical_center_error_m > FORMAL_CONTACT_POSITION_TOL_M:
        reasons.append("teacher_physical_face_center_target_mismatch")
    if reference_site_speed_error_mps > FORMAL_REFERENCE_SITE_SPEED_TOL_MPS:
        reasons.append("reference_racket_site_speed_mismatch")
    if teacher_site_target_error_m > FORMAL_CONTACT_POSITION_TOL_M:
        reasons.append("teacher_task_site_target_mismatch")
    if (
        teacher_face_normal_angle_error_rad
        > FORMAL_TASK_FACE_NORMAL_ANGLE_TOL_RAD
    ):
        reasons.append("teacher_task_face_normal_mismatch")
    if teacher_face_velocity_error_mps > FORMAL_REFERENCE_SITE_SPEED_TOL_MPS:
        reasons.append("teacher_task_face_velocity_mismatch")
    if teacher_site_velocity_error_mps > FORMAL_REFERENCE_SITE_SPEED_TOL_MPS:
        reasons.append("teacher_task_site_velocity_mismatch")
    if (
        teacher_angular_velocity_error_radps
        > FORMAL_REFERENCE_SITE_SPEED_TOL_MPS
    ):
        reasons.append("teacher_task_angular_velocity_mismatch")
    if events.incoming_table_bounces != 1:
        reasons.append("incoming_table_bounce_count_not_exactly_one")
    if events.net_crossing is None:
        reasons.append("no_post_hit_net_crossing")
    elif not events.net_crossing["cleared"]:
        reasons.append("net_not_cleared")
    if events.ball_net_collision is not None:
        reasons.append("ball_intersects_net_or_post_geometry")
    if events.first_landing is None:
        reasons.append("no_fitted_first_table_landing")
    else:
        x, y = events.first_landing["ball_center_xy_m"]
        if not (
            float(profile["net_x_m"])
            + float(profile["minimum_landing_depth_m"])
            + FORMAL_LANDING_DEPTH_GUARD_M
            < x
            <= float(profile["opponent_far_x_m"])
            and abs(y) <= float(profile["table_half_width_m"])
        ):
            reasons.append("first_landing_outside_opponent_table")
        landing_error = float(
            np.linalg.norm(
                np.asarray((x, y), np.float64)
                - np.asarray(task_landing_aim_w_xy_m, np.float64)
            )
        )
        if landing_error > FORMAL_TASK_LANDING_TOL_M:
            reasons.append("first_landing_misses_frozen_task_aim")
    if events.native_ball_contact_count:
        reasons.append("native_ball_contact_invariant_broken")
    if events.robot_obstacle_contacts:
        reasons.append(
            "robot_hit_five_solid_table_net_post_or_under_table_keepout"
        )
    if events.robot_obstacle_swept_hit_count:
        reasons.append(
            "continuous_five_solid_robot_obstacle_sweep_hit"
        )
    if (
        events.foot_floor_penetration_violation_count
        or events.nonfoot_ground_contact_violation_count
    ):
        reasons.append("teacher_illegal_ground_contact")
    if events.self_contacts:
        reasons.append("robot_self_contact")
    if events.joint_limit_violation is not None:
        reasons.append("joint_limit_violation")
    if events.fall is not None:
        reasons.append("fall")
    if events.event_order_violations:
        reasons.append("ambiguous_or_nonchronological_surface_events")
    if events.ball_forbidden_contacts:
        reasons.append("ball_contacted_forbidden_robot_table_or_net_geometry")
    if events.shadow_robot_obstacle_near_contacts:
        reasons.append(
            "adaptive_shadow_robot_obstacle_clearance_below_guard"
        )
    if events.shadow_self_near_contacts:
        reasons.append("adaptive_shadow_self_clearance_below_guard")
    if events.shadow_nonfoot_ground_near_contacts:
        reasons.append(
            "adaptive_shadow_nonfoot_ground_clearance_below_guard"
        )
    if events.shadow_foot_floor_penetration_violations:
        reasons.append(
            "adaptive_shadow_foot_floor_penetration_exceeds_tolerance"
        )
    chronological_times: Optional[Tuple[float, float, float, float, float]] = None
    if (
        events.activation_time_s is not None
        and len(events.incoming_table_bounce_times_s) == 1
        and events.paddle_contact is not None
        and events.net_crossing is not None
        and events.first_landing is not None
    ):
        chronological_times = (
            events.activation_time_s,
            events.incoming_table_bounce_times_s[0],
            float(events.paddle_contact["time_s"]),
            float(events.net_crossing["time_s"]),
            float(events.first_landing["time_s"]),
        )
    if chronological_times is None or any(
        later - earlier <= FORMAL_EVENT_TIME_GUARD_S
        for earlier, later in zip(
            chronological_times or (), (chronological_times or ())[1:]
        )
    ):
        reasons.append(
            "event_order_not_activation_incoming_bounce_paddle_net_landing"
        )
    return reasons


def run_action_dt(
    *,
    mujoco: Any,
    model: Any,
    action: native_diag.ActionSpec,
    clip: motion_player.MotionClip,
    launch: CaseLaunchState,
    task_case: PhysicalTaskCase,
    venue: VenueParams,
    profile: Mapping[str, Any],
    face_mesh: FaceMesh,
    obstacle_rows: Mapping[str, Any],
    post_contact_s: float,
    contact_time_tolerance_s: float,
    contact_position_tolerance_m: float,
    capture_frames: bool,
    render_fps: int,
    teacher_wait_override_s: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    binding = motion_player.bind_model(mujoco, model)
    data = mujoco.MjData(model)
    probe_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    ball_joint = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT_NAME)
    )
    ball_geom = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME)
    )
    ball_body = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY_NAME)
    )
    if min(ball_joint, ball_geom, ball_body) < 0:
        raise FittedGateError("compiled scene lacks fitted dynamic ball")
    if int(model.geom_contype[ball_geom]) != 0 or int(model.geom_conaffinity[ball_geom]) != 0:
        raise FittedGateError("native ball contact is not disabled")
    qadr, dadr = int(model.jnt_qposadr[ball_joint]), int(model.jnt_dofadr[ball_joint])
    teacher_rate = task_case.teacher_rate
    task_wait = task_case.pre_swing_wait_s
    wait = (
        task_wait
        if teacher_wait_override_s is None
        else _finite(
            teacher_wait_override_s,
            "teacher_wait_override_s",
            nonnegative=True,
        )
    )
    teacher_hit = retimed_teacher_state(
        clip,
        world_time_s=task_case.time_to_contact_s,
        pre_swing_wait_s=wait,
        teacher_rate=teacher_rate,
    )
    native_diag._set_teacher_state(
        mujoco, model, data, binding, teacher_hit
    )
    face_at_reference_hit = _face_state(
        mujoco, model, data, binding, action.mount_normal_sign
    )
    physical_ball_center_at_reference_hit = (
        face_at_reference_hit.center_position_m
        + venue.ball_radius * face_at_reference_hit.normal_w
    )
    teacher_physical_center_error = float(
        np.linalg.norm(
            physical_ball_center_at_reference_hit
            - task_case.ball_contact_w_m
        )
    )
    teacher_site_target_error = float(
        np.linalg.norm(
            face_at_reference_hit.site_position_m
            - task_case.racket_site_target_w_m
        )
    )
    task_face_dot = float(
        np.clip(
            face_at_reference_hit.normal_w
            @ task_case.racket_normal_w,
            -1.0,
            1.0,
        )
    )
    teacher_face_normal_angle_error = math.acos(task_face_dot)
    actual_face_center_velocity = (
        face_at_reference_hit.site_linear_velocity_mps
        + np.cross(
            face_at_reference_hit.angular_velocity_radps,
            face_at_reference_hit.center_position_m
            - face_at_reference_hit.site_position_m,
        )
    )
    teacher_face_velocity_error = float(
        np.linalg.norm(
            actual_face_center_velocity
            - task_case.racket_face_center_velocity_w_mps
        )
    )
    teacher_site_velocity_error = float(
        np.linalg.norm(
            face_at_reference_hit.site_linear_velocity_mps
            - task_case.racket_site_velocity_w_mps
        )
    )
    teacher_angular_velocity_error = float(
        np.linalg.norm(
            face_at_reference_hit.angular_velocity_radps
            - task_case.racket_command_angular_velocity_w_radps
        )
    )
    reference_site_speed_actual = float(
        np.linalg.norm(face_at_reference_hit.site_linear_velocity_mps)
    )
    required_site_speed = float(
        np.linalg.norm(task_case.racket_site_velocity_w_mps)
    )
    reference_site_speed_error = abs(
        reference_site_speed_actual - required_site_speed
    )
    reference_return_normal_x_margin = float(
        face_at_reference_hit.normal_w[0] - FORMAL_MIN_RETURN_NORMAL_X
    )
    mujoco.mj_resetData(model, data)
    robot_safety_obstacle_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)): name
        for name in table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    if any(key < 0 for key in robot_safety_obstacle_ids):
        raise FittedGateError(
            "compiled scene lacks a five-solid robot safety obstacle"
        )
    ball_obstacle_ids = {
        geom_id: name
        for geom_id, name in robot_safety_obstacle_ids.items()
        if name in table_scene.OBSTACLE_NAMES
    }
    if tuple(ball_obstacle_ids.values()) != table_scene.OBSTACLE_NAMES:
        raise FittedGateError(
            "analytic ball obstacle order is not exact top/net/posts"
        )
    # The analytic fitted-ball/self-collision shadow must continue to cover
    # every robot-body geom: the fitted ball has native collision disabled, so
    # MuJoCo's table masks are not an authority for ball-vs-robot exclusion.
    robot_collision_geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if geom_id != ball_geom
        and int(model.geom_bodyid[geom_id]) != 0
    )
    five_solid_robot_geom_ids = _five_solid_robot_geom_ids(
        mujoco,
        model,
        ball_geom_id=ball_geom,
        obstacle_geom_ids=tuple(robot_safety_obstacle_ids),
    )
    ground_contract = build_ground_contact_contract(
        mujoco, model, ball_geom_id=ball_geom
    )
    if not robot_collision_geom_ids or not five_solid_robot_geom_ids:
        raise FittedGateError("compiled scene lacks robot collision geoms")
    robot_geom_motion_bounds = build_shadow_kinematic_bounds(
        mujoco=mujoco,
        model=model,
        binding=binding,
        robot_geom_ids=robot_collision_geom_ids,
    )
    robot_geom_motion_bound_by_id = {
        bound.geom_id: bound for bound in robot_geom_motion_bounds
    }
    ground_geom_motion_bounds = tuple(
        robot_geom_motion_bound_by_id[geom_id]
        for geom_id in (
            tuple(ground_contract.foot_geom_ids)
            + tuple(ground_contract.nonfoot_robot_geom_ids)
        )
    )
    self_collision_pairs = tuple(
        (first_geom, second_geom)
        for index, first_geom in enumerate(robot_collision_geom_ids)
        for second_geom in robot_collision_geom_ids[index + 1 :]
        if _geom_pair_enabled(
            mujoco, model, first_geom, second_geom
        )
    )
    aabbs = _obstacle_aabbs(obstacle_rows)
    safety_aabbs = table_scene.action_ball_policy_obstacle_aabbs(
        obstacle_rows
    )
    dt = float(model.opt.timestep)
    clip_duration = (clip.n_frames - 1) / clip.fps
    if abs(clip_duration - action.t_cycle_s) > 1.0 / clip.fps + 1.0e-9:
        raise FittedGateError(
            f"{action.action_id}: teacher duration does not match t_cycle"
        )
    scaled_clip_duration = clip_duration / teacher_rate
    total_time = max(
        wait + scaled_clip_duration,
        task_case.time_to_contact_s + post_contact_s,
    )
    events = FittedEvents()
    active = False
    returned = False
    trajectory: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    renderer = None
    expected_render_frames = 0
    executed_steps = 0
    if capture_frames:
        try:
            renderer = mujoco.Renderer(model, height=720, width=960)
        except Exception:
            renderer = None
    render_stride = max(1, int(round(1.0 / (render_fps * dt))))
    data.qpos[qadr : qadr + 3] = np.asarray((0.0, 0.0, 100.0))
    data.qpos[qadr + 3 : qadr + 7] = np.asarray((1.0, 0.0, 0.0, 0.0))
    data.qvel[dadr : dadr + 6] = 0.0

    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in motion_player.RUNTIME_JOINT_NAMES
        ],
        np.int64,
    )
    max_steps = int(math.ceil(total_time / dt)) + 2
    for step in range(max_steps):
        executed_steps = step + 1
        time_s = step * dt
        teacher = retimed_teacher_state(
            clip,
            world_time_s=time_s,
            pre_swing_wait_s=wait,
            teacher_rate=teacher_rate,
        )
        native_diag._set_teacher_state(
            mujoco, model, data, binding, teacher
        )
        # `_set_teacher_state` calls mj_forward at the exact teacher pose.
        # Scan that state before integration so forced kinematic placement
        # cannot hide a table/net contact that the following dynamics step
        # immediately resolves.
        _scan_robot_contacts(
            mujoco,
            model,
            data,
            ball_geom,
            robot_safety_obstacle_ids,
            events,
            time_s,
            ground_contract,
        )
        face_before = _face_state(
            mujoco, model, data, binding, action.mount_normal_sign
        )
        if not active and time_s + 0.5 * dt >= launch.activation_time_s:
            data.qpos[qadr : qadr + 3] = launch.position_w_m
            data.qpos[qadr + 3 : qadr + 7] = np.asarray((1.0, 0.0, 0.0, 0.0))
            data.qvel[dadr : dadr + 3] = launch.velocity_w_mps
            set_ball_spin_world(
                data, qadr, dadr, launch.spin_w_radps
            )
            active = True
            events.activation_time_s = time_s
            mujoco.mj_forward(model, data)
            face_before = _face_state(
                mujoco, model, data, binding, action.mount_normal_sign
            )
        root_z = float(data.qpos[binding.root_qpos_adr + 2])
        root_tilt = native_diag._root_tilt_rad(
            data.qpos[
                binding.root_qpos_adr
                + 3 : binding.root_qpos_adr
                + 7
            ]
        )
        if events.fall is None and (
            root_z < ROOT_Z_FALL_M or root_tilt > ROOT_TILT_FALL_RAD
        ):
            events.fall = {
                "time_s": time_s,
                "root_z_m": root_z,
                "root_tilt_rad": root_tilt,
            }
        q = np.asarray(data.qpos)[binding.joint_qpos_adrs]
        limited = np.asarray(model.jnt_limited)[joint_ids].astype(bool)
        ranges = np.asarray(model.jnt_range)[joint_ids]
        bad = np.flatnonzero(
            limited
            & (
                (q < ranges[:, 0] - 1.0e-7)
                | (q > ranges[:, 1] + 1.0e-7)
            )
        )
        if bad.size and events.joint_limit_violation is None:
            events.joint_limit_violation = {
                "time_s": time_s,
                "joints": [
                    motion_player.RUNTIME_JOINT_NAMES[int(index)]
                    for index in bad
                ],
            }

        if active:
            p0 = np.asarray(data.qpos[qadr : qadr + 3], np.float64).copy()
            v0 = np.asarray(data.qvel[dadr : dadr + 3], np.float64).copy()
            w0 = ball_spin_world(data, qadr, dadr)
            aero = (
                -venue.k_d * float(np.linalg.norm(v0)) * v0
                + venue.k_m * np.cross(w0, v0)
            )
            data.xfrc_applied[ball_body, :3] = venue.ball_mass * aero
        else:
            p0 = v0 = w0 = None
            data.xfrc_applied[ball_body, :] = 0.0
        robot_centers_before = np.asarray(
            data.geom_xpos[
                np.asarray(five_solid_robot_geom_ids, np.int64)
            ],
            np.float64,
        ).copy()
        mujoco.mj_step(model, data)
        after = float(data.time)
        _scan_robot_contacts(
            mujoco,
            model,
            data,
            ball_geom,
            robot_safety_obstacle_ids,
            events,
            after,
            ground_contract,
        )
        next_teacher = retimed_teacher_state(
            clip,
            world_time_s=after,
            pre_swing_wait_s=wait,
            teacher_rate=teacher_rate,
        )
        native_diag._set_teacher_state(
            mujoco, model, data, binding, next_teacher
        )
        # Also scan the exact end-of-substep teacher pose.  Together with the
        # post-mj_step scan above this covers ready, swing, hit, and recovery
        # at every 1.0/0.5 ms physics boundary.
        _scan_robot_contacts(
            mujoco,
            model,
            data,
            ball_geom,
            robot_safety_obstacle_ids,
            events,
            after,
            ground_contract,
        )
        run_continuous_ground_probe(
            mujoco=mujoco,
            model=model,
            probe_data=probe_data,
            binding=binding,
            clip=clip,
            wait_s=wait,
            teacher_rate=teacher_rate,
            start_time_s=time_s,
            duration_s=after - time_s,
            ground_contract=ground_contract,
            ground_geom_motion_bounds=ground_geom_motion_bounds,
            events=events,
        )
        swept_robot = scan_five_solid_robot_sweep(
            mujoco=mujoco,
            model=model,
            data=data,
            robot_geom_ids=five_solid_robot_geom_ids,
            centers_before=robot_centers_before,
            obstacle_aabbs=safety_aabbs,
        )
        if int(swept_robot["hit_count"]):
            events.robot_obstacle_swept_steps += 1
            events.robot_obstacle_swept_hit_count += int(
                swept_robot["hit_count"]
            )
            for name, count in swept_robot[
                "per_obstacle"
            ].items():
                events.robot_obstacle_swept_per_obstacle[
                    name
                ] += int(count)
            if events.robot_obstacle_swept_first_hit is None:
                events.robot_obstacle_swept_first_hit = dict(
                    swept_robot["first_hit"]
                )
        face_after = _face_state(
            mujoco, model, data, binding, action.mount_normal_sign
        )
        if active:
            p1 = np.asarray(data.qpos[qadr : qadr + 3], np.float64).copy()
            v1 = np.asarray(data.qvel[dadr : dadr + 3], np.float64).copy()
            w1 = ball_spin_world(data, qadr, dadr)
            p1, v1, w1, returned, swept_segments = (
                process_surface_events_chronologically(
                p0=p0,
                p1=p1,
                v0=v0,
                v1=v1,
                w0=w0,
                w1=w1,
                time_s=time_s,
                dt=dt,
                face_before=face_before,
                face_after=face_after,
                face_mesh=face_mesh,
                action=action,
                venue=venue,
                profile=profile,
                aabbs=aabbs,
                events=events,
                returned=returned,
                )
            )
            data.qpos[qadr : qadr + 3] = p1
            data.qvel[dadr : dadr + 3] = v1
            set_ball_spin_world(data, qadr, dadr, w1)
            run_shadow_forbidden_geometry_probe(
                mujoco=mujoco,
                model=model,
                probe_data=probe_data,
                binding=binding,
                clip=clip,
                wait_s=wait,
                teacher_rate=teacher_rate,
                ball_qpos_address=qadr,
                ball_dof_address=dadr,
                ball_geom_id=ball_geom,
                robot_collision_geom_ids=robot_collision_geom_ids,
                five_solid_robot_geom_ids=(
                    five_solid_robot_geom_ids
                ),
                robot_geom_motion_bounds=robot_geom_motion_bounds,
                self_collision_pairs=self_collision_pairs,
                ball_obstacle_ids=ball_obstacle_ids,
                ball_obstacle_aabbs=aabbs,
                robot_safety_obstacle_ids=(
                    robot_safety_obstacle_ids
                ),
                robot_safety_obstacle_aabbs=safety_aabbs,
                swept_segments=swept_segments,
                events=events,
                ball_radius_m=venue.ball_radius,
            )
            trajectory.append(
                {
                    "physics_step": step,
                    "time_s": after,
                    "ball_active": True,
                    "ball_position_m": p1.tolist(),
                    "ball_velocity_mps": v1.tolist(),
                    "ball_spin_radps": w1.tolist(),
                    "racket_face_center_m": face_after.center_position_m.tolist(),
                    "racket_face_normal_w": face_after.normal_w.tolist(),
                    "paddle_impulse_count": events.paddle_impulse_count,
                    "robot_obstacle_contact_records": len(
                        events.robot_obstacle_contacts
                    ),
                    "robot_obstacle_swept_steps": (
                        events.robot_obstacle_swept_steps
                    ),
                    "robot_obstacle_swept_hit_count": (
                        events.robot_obstacle_swept_hit_count
                    ),
                    "self_contact_records": len(events.self_contacts),
                    "ball_forbidden_contact_records": len(
                        events.ball_forbidden_contacts
                    ),
                    "shadow_robot_obstacle_near_contact_records": len(
                        events.shadow_robot_obstacle_near_contacts
                    ),
                    "shadow_self_near_contact_records": len(
                        events.shadow_self_near_contacts
                    ),
                    "root_z_m": root_z,
                    "root_tilt_rad": root_tilt,
                }
            )
        else:
            run_shadow_forbidden_geometry_probe(
                mujoco=mujoco,
                model=model,
                probe_data=probe_data,
                binding=binding,
                clip=clip,
                wait_s=wait,
                teacher_rate=teacher_rate,
                ball_qpos_address=qadr,
                ball_dof_address=dadr,
                ball_geom_id=ball_geom,
                robot_collision_geom_ids=robot_collision_geom_ids,
                five_solid_robot_geom_ids=(
                    five_solid_robot_geom_ids
                ),
                robot_geom_motion_bounds=robot_geom_motion_bounds,
                self_collision_pairs=self_collision_pairs,
                ball_obstacle_ids=ball_obstacle_ids,
                ball_obstacle_aabbs=aabbs,
                robot_safety_obstacle_ids=(
                    robot_safety_obstacle_ids
                ),
                robot_safety_obstacle_aabbs=safety_aabbs,
                swept_segments=(
                    (
                        time_s,
                        dt,
                        np.asarray((0.0, 0.0, 100.0), np.float64),
                        np.asarray((0.0, 0.0, 100.0), np.float64),
                    ),
                ),
                events=events,
                ball_radius_m=venue.ball_radius,
            )
            trajectory.append(
                {
                    "physics_step": step,
                    "time_s": after,
                    "ball_active": False,
                    "ball_position_m": None,
                    "ball_velocity_mps": None,
                    "ball_spin_radps": None,
                    "racket_face_center_m": face_after.center_position_m.tolist(),
                    "racket_face_normal_w": face_after.normal_w.tolist(),
                    "paddle_impulse_count": events.paddle_impulse_count,
                    "robot_obstacle_contact_records": len(
                        events.robot_obstacle_contacts
                    ),
                    "robot_obstacle_swept_steps": (
                        events.robot_obstacle_swept_steps
                    ),
                    "robot_obstacle_swept_hit_count": (
                        events.robot_obstacle_swept_hit_count
                    ),
                    "self_contact_records": len(events.self_contacts),
                    "ball_forbidden_contact_records": len(
                        events.ball_forbidden_contacts
                    ),
                    "shadow_robot_obstacle_near_contact_records": len(
                        events.shadow_robot_obstacle_near_contacts
                    ),
                    "shadow_self_near_contact_records": len(
                        events.shadow_self_near_contacts
                    ),
                    "root_z_m": root_z,
                    "root_tilt_rad": root_tilt,
                }
            )
        if step % render_stride == 0:
            expected_render_frames += 1
            if renderer is not None:
                try:
                    renderer.update_scene(
                        data, camera="torso_follow"
                    )
                    frames.append(renderer.render().copy())
                except Exception:
                    renderer = None
        if after + EPS >= total_time:
            break
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    reasons = _action_failure_reasons(
        events,
        expected_contact_time_s=task_case.time_to_contact_s,
        contact_time_tolerance_s=contact_time_tolerance_s,
        contact_position_w_m=task_case.ball_contact_w_m,
        contact_position_tolerance_m=contact_position_tolerance_m,
        profile=profile,
        teacher_physical_center_error_m=teacher_physical_center_error,
        reference_site_speed_error_mps=reference_site_speed_error,
        teacher_site_target_error_m=teacher_site_target_error,
        teacher_face_normal_angle_error_rad=(
            teacher_face_normal_angle_error
        ),
        teacher_face_velocity_error_mps=teacher_face_velocity_error,
        teacher_site_velocity_error_mps=teacher_site_velocity_error,
        teacher_angular_velocity_error_radps=(
            teacher_angular_velocity_error
        ),
        task_landing_aim_w_xy_m=task_case.landing_aim_w_xy_m,
    )
    if events.ground_shadow_certificate_failure is not None:
        reasons.append(
            "continuous_ground_relative_motion_certificate_failed"
        )
    if (
        events.ground_shadow_certificate_intervals <= 0
        or abs(events.ground_shadow_covered_duration_s - after)
        > 1.0e-9
    ):
        reasons.append(
            "whole_teacher_prep_hit_recovery_ground_coverage_incomplete"
        )
    incoming_velocity_error: Optional[float] = None
    incoming_spin_error: Optional[float] = None
    if events.paddle_contact is not None:
        incoming_velocity_error = float(
            np.linalg.norm(
                np.asarray(
                    events.paddle_contact["velocity_minus_mps"],
                    np.float64,
                )
                - task_case.incoming_velocity_w_mps
            )
        )
        incoming_spin_error = float(
            np.linalg.norm(
                np.asarray(
                    events.paddle_contact["spin_minus_radps"],
                    np.float64,
                )
                - task_case.incoming_spin_w_radps
            )
        )
        if (
            incoming_velocity_error
            > FORMAL_INCOMING_VELOCITY_TOL_MPS
        ):
            reasons.append("physical_incoming_velocity_mismatch")
        if incoming_spin_error > FORMAL_INCOMING_SPIN_TOL_RADPS:
            reasons.append("physical_incoming_spin_mismatch")
    if after + EPS < wait + scaled_clip_duration:
        reasons.append("ready_to_recovery_safety_window_incomplete")
    if abs(events.shadow_covered_duration_s - after) > max(EPS, 1.0e-9):
        reasons.append(
            "whole_teacher_prep_hit_recovery_shadow_coverage_incomplete"
        )
    if len(trajectory) != executed_steps:
        reasons.append("per_step_metrics_incomplete")
    if reference_return_normal_x_margin <= 0.0:
        reasons.append("teacher_selected_face_not_oriented_toward_opponent")
    landing_error_m = (
        None
        if events.first_landing is None
        else float(
            np.linalg.norm(
                np.asarray(
                    events.first_landing["ball_center_xy_m"],
                    np.float64,
                )
                - task_case.landing_aim_w_xy_m
            )
        )
    )
    physical_return_pass = (
        events.paddle_impulse_count == 1
        and events.paddle_contact is not None
        and events.incoming_table_bounces == 1
        and events.net_crossing is not None
        and bool(events.net_crossing.get("cleared"))
        and events.ball_net_collision is None
        and events.first_landing is not None
        and landing_error_m is not None
        and landing_error_m <= FORMAL_TASK_LANDING_TOL_M
        and incoming_velocity_error is not None
        and incoming_velocity_error <= FORMAL_INCOMING_VELOCITY_TOL_MPS
        and incoming_spin_error is not None
        and incoming_spin_error <= FORMAL_INCOMING_SPIN_TOL_RADPS
    )
    teacher_task_match_pass = (
        teacher_physical_center_error <= FORMAL_CONTACT_POSITION_TOL_M
        and teacher_site_target_error <= FORMAL_CONTACT_POSITION_TOL_M
        and reference_site_speed_error
        <= FORMAL_REFERENCE_SITE_SPEED_TOL_MPS
        and teacher_face_normal_angle_error
        <= FORMAL_TASK_FACE_NORMAL_ANGLE_TOL_RAD
        and teacher_face_velocity_error
        <= FORMAL_REFERENCE_SITE_SPEED_TOL_MPS
        and teacher_site_velocity_error
        <= FORMAL_REFERENCE_SITE_SPEED_TOL_MPS
        and teacher_angular_velocity_error
        <= FORMAL_REFERENCE_SITE_SPEED_TOL_MPS
    )
    teacher_clearance_pass = not any(
        (
            events.robot_obstacle_contacts,
            events.robot_obstacle_swept_hit_count,
            events.self_contacts,
            events.joint_limit_violation is not None,
            events.fall is not None,
            events.shadow_robot_obstacle_near_contacts,
            events.shadow_self_near_contacts,
            events.event_order_violations,
        )
    )
    teacher_ground_safety_pass = not any(
        (
            events.foot_floor_penetration_violation_count,
            events.nonfoot_ground_contact_violation_count,
            events.shadow_nonfoot_ground_near_contacts,
            events.shadow_foot_floor_penetration_violations,
            events.ground_shadow_certificate_failure is not None,
            events.ground_shadow_certificate_intervals <= 0,
            abs(events.ground_shadow_covered_duration_s - after)
            > 1.0e-9,
        )
    )
    return (
        {
            "timestep_s": dt,
            "verdict": "PASS" if not reasons else "FAIL",
            "failure_reasons": reasons,
            "paddle_impulse_count": events.paddle_impulse_count,
            "teacher_reference_hit": {
                "physical_ball_center_m": (
                    physical_ball_center_at_reference_hit.tolist()
                ),
                "manifest_contact_center_m": np.asarray(
                    task_case.ball_contact_w_m, np.float64
                ).tolist(),
                "center_error_m": teacher_physical_center_error,
                "site_target_task_m": (
                    task_case.racket_site_target_w_m.tolist()
                ),
                "site_target_error_m": teacher_site_target_error,
                "site_speed_actual_mps": reference_site_speed_actual,
                "site_speed_task_mps": required_site_speed,
                "reference_site_speed_manifest_mps": (
                    action.racket_speed_mps
                ),
                "site_speed_error_mps": reference_site_speed_error,
                "face_center_velocity_actual_mps": (
                    actual_face_center_velocity.tolist()
                ),
                "face_center_velocity_task_mps": (
                    task_case.racket_face_center_velocity_w_mps.tolist()
                ),
                "face_center_velocity_error_mps": (
                    teacher_face_velocity_error
                ),
                "site_velocity_task_mps": (
                    task_case.racket_site_velocity_w_mps.tolist()
                ),
                "site_velocity_error_mps": (
                    teacher_site_velocity_error
                ),
                "angular_velocity_task_radps": (
                    task_case.racket_command_angular_velocity_w_radps.tolist()
                ),
                "angular_velocity_error_radps": (
                    teacher_angular_velocity_error
                ),
                "selected_face_normal_w": (
                    face_at_reference_hit.normal_w.tolist()
                ),
                "task_face_normal_w": (
                    task_case.racket_normal_w.tolist()
                ),
                "face_normal_angle_error_rad": (
                    teacher_face_normal_angle_error
                ),
                "selected_face_return_normal_x_margin": (
                    reference_return_normal_x_margin
                ),
                "fixed_center_tolerance_m": (
                    FORMAL_CONTACT_POSITION_TOL_M
                ),
                "fixed_site_speed_tolerance_mps": (
                    FORMAL_REFERENCE_SITE_SPEED_TOL_MPS
                ),
            },
            "paddle_contact": events.paddle_contact,
            "net_crossing": events.net_crossing,
            "first_landing": events.first_landing,
            "first_landing_task_aim_w_xy_m": (
                task_case.landing_aim_w_xy_m.tolist()
            ),
            "first_landing_task_error_m": landing_error_m,
            "incoming_task_state_error": {
                "velocity_mps": incoming_velocity_error,
                "spin_radps": incoming_spin_error,
                "velocity_tolerance_mps": (
                    FORMAL_INCOMING_VELOCITY_TOL_MPS
                ),
                "spin_tolerance_radps": (
                    FORMAL_INCOMING_SPIN_TOL_RADPS
                ),
            },
            "ball_net_collision": events.ball_net_collision,
            "activation_time_s": events.activation_time_s,
            "incoming_table_bounces": events.incoming_table_bounces,
            "return_table_bounces": events.return_table_bounces,
            "incoming_table_bounce_times_s": (
                events.incoming_table_bounce_times_s
            ),
            "return_table_bounce_times_s": (
                events.return_table_bounce_times_s
            ),
            "table_contacts": events.table_contacts,
            "event_order_violations": events.event_order_violations,
            "ball_forbidden_contacts": events.ball_forbidden_contacts,
            "shadow_probe_samples": events.shadow_probe_samples,
            "shadow_robot_obstacle_near_contacts": (
                events.shadow_robot_obstacle_near_contacts
            ),
            "shadow_self_near_contacts": events.shadow_self_near_contacts,
            "shadow_relative_motion_certificate": {
                "intervals": events.shadow_certificate_intervals,
                "covered_duration_s": events.shadow_covered_duration_s,
                "required_duration_s": after,
                "max_ball_path_bound_m": (
                    events.shadow_max_ball_path_bound_m
                ),
                "max_robot_surface_path_bound_m": (
                    events.shadow_max_robot_surface_path_bound_m
                ),
                "ball_plus_robot_guard_margin_m": (
                    FORMAL_SHADOW_CLEARANCE_GUARD_M
                    - events.shadow_max_ball_path_bound_m
                    - events.shadow_max_robot_surface_path_bound_m
                ),
                "two_robot_surface_guard_margin_m": (
                    FORMAL_ROBOT_OBSTACLE_GUARD_M
                    - 2.0
                    * events.shadow_max_robot_surface_path_bound_m
                ),
                "self_collision_pair_count": len(self_collision_pairs),
                "robot_geom_count": len(robot_collision_geom_ids),
                "five_solid_robot_geom_count": len(
                    five_solid_robot_geom_ids
                ),
                "robot_safety_obstacle_names": list(
                    table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
                ),
                "ball_obstacle_names": list(
                    table_scene.OBSTACLE_NAMES
                ),
                "robot_only_keepout": (
                    table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
                ),
                "motion_frame_knots_are_interval_boundaries": True,
                "whole_prep_hit_recovery_required": True,
            },
            "native_ball_contact_count": events.native_ball_contact_count,
            "robot_obstacle_contacts": events.robot_obstacle_contacts[:50],
            "five_solid_robot_safety": {
                "contact_count": (
                    events.robot_obstacle_contact_count
                ),
                "contact_per_obstacle": dict(
                    events.robot_obstacle_contact_per_obstacle
                ),
                "contact_force_threshold_n": (
                    TABLE_CONTACT_FORCE_THRESHOLD_N
                ),
                "swept_steps": events.robot_obstacle_swept_steps,
                "swept_hit_count": (
                    events.robot_obstacle_swept_hit_count
                ),
                "swept_per_obstacle": dict(
                    events.robot_obstacle_swept_per_obstacle
                ),
                "swept_first_hit": (
                    events.robot_obstacle_swept_first_hit
                ),
                "continuous_sweep_method": (
                    FIVE_SOLID_SWEEP_METHOD
                ),
                "obstacle_order": list(
                    table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
                ),
                "robot_geom_count": len(five_solid_robot_geom_ids),
                "robot_geom_selection": (
                    "mujoco_pair_enabled_against_any_five_solid_obstacle"
                ),
                "ball_keepout_native_pair_enabled": False,
                "ball_keepout_analytic_surface_enabled": False,
            },
            "ground_contact_safety": {
                **ground_contact_contract_receipt(
                    mujoco, model, ground_contract
                ),
                "contact_count": events.ground_contact_count,
                "legal_foot_support_contact_count": (
                    events.legal_foot_support_contact_count
                ),
                "foot_floor_penetration_violation_count": (
                    events.foot_floor_penetration_violation_count
                ),
                "nonfoot_ground_contact_violation_count": (
                    events.nonfoot_ground_contact_violation_count
                ),
                "max_foot_penetration_m": (
                    events.ground_max_foot_penetration_m
                ),
                "max_nonfoot_penetration_m": (
                    events.ground_max_nonfoot_penetration_m
                ),
                "violations": events.ground_contact_violations[:50],
                "shadow_nonfoot_near_contacts": (
                    events.shadow_nonfoot_ground_near_contacts[:50]
                ),
                "shadow_foot_penetration_violations": (
                    events.shadow_foot_floor_penetration_violations[:50]
                ),
                "continuous_certificate": {
                    "intervals": (
                        events.ground_shadow_certificate_intervals
                    ),
                    "probe_samples": (
                        events.ground_shadow_probe_samples
                    ),
                    "covered_duration_s": (
                        events.ground_shadow_covered_duration_s
                    ),
                    "required_duration_s": after,
                    "min_nonfoot_lower_bound_m": (
                        events
                        .ground_shadow_min_nonfoot_lower_bound_m
                    ),
                    "required_nonfoot_lower_bound_m": (
                        FORMAL_NONFOOT_GROUND_CLEARANCE_GUARD_M
                    ),
                    "min_foot_lower_bound_m": (
                        events.ground_shadow_min_foot_lower_bound_m
                    ),
                    "required_foot_lower_bound_m": (
                        -FOOT_FLOOR_PENETRATION_TOLERANCE_M
                    ),
                    "method": (
                        "adaptive_motion_knot_subdivision_three_point_"
                        "signed_distance_plus_per_geom_surface_path_bound_v1"
                    ),
                    "whole_prep_hit_recovery_required": True,
                    "failure": (
                        events.ground_shadow_certificate_failure
                    ),
                },
            },
            "self_contacts": events.self_contacts[:50],
            "joint_limit_violation": events.joint_limit_violation,
            "fall": events.fall,
            "simulation_window": {
                "start_time_s": 0.0,
                "executed_end_time_s": after,
                "required_ready_to_recovery_end_time_s": (
                    wait + scaled_clip_duration
                ),
                "task_pre_swing_wait_s": task_wait,
                "executed_pre_swing_wait_s": wait,
                "teacher_rate": teacher_rate,
                "scaled_t_hit_s": task_case.scaled_t_hit_s,
                "scaled_t_cycle_s": task_case.scaled_t_cycle_s,
                "physics_steps": executed_steps,
                "exact_teacher_pose_safety_scans": 2 * executed_steps,
                "post_dynamics_safety_scans": executed_steps,
                "expected_render_frames": (
                    expected_render_frames if capture_frames else 0
                ),
            },
            "mandatory_gates": {
                "physical_ball_selected_face_return_and_first_landing": (
                    physical_return_pass
                ),
                "teacher_matches_frozen_solver_task": (
                    teacher_task_match_pass
                ),
                "teacher_robot_and_racket_table_net_post_clearance": (
                    teacher_clearance_pass
                ),
                "teacher_robot_and_racket_five_solid_clearance": (
                    teacher_clearance_pass
                ),
                "teacher_ground_contact_safety": (
                    teacher_ground_safety_pass
                ),
            },
            "frame_metrics": trajectory,
        },
        frames,
    )


def _preflight(
    args: argparse.Namespace,
) -> Tuple[
    List[str],
    Dict[str, Any],
    Optional[PhysicalManifest],
    Optional[Dict[str, Any]],
    Optional[VenueParams],
]:
    blockers: List[str] = []
    evidence: Dict[str, Any] = {}
    manifest: Optional[PhysicalManifest] = None
    profile: Optional[Dict[str, Any]] = None
    venue: Optional[VenueParams] = None
    raw_manifest: Dict[str, Any] = {}
    trusted_action_set: Optional[Dict[str, Any]] = None
    args._launch_trust_pinned_files = ()
    args._physical_task_bundle_pin = None
    if not args.preflight_only and args.render_dir is None:
        blockers.append("missing_required_render_dir_for_formal_gate")
    if args.render_fps <= 0:
        blockers.append("render_fps_must_be_positive")
    if not args.code_commit:
        blockers.append("missing_expected_clean_code_commit")
    else:
        try:
            evidence["checkout"] = validate_clean_checkout(args.code_commit)
        except Exception as exc:
            blockers.append(f"checkout:{exc}")
    try:
        trusted_action_set = action_set_contract.load_contract_from_source(
            ACTION_SET_CONTRACT_SOURCE_PATH.read_bytes(),
            getattr(
                args,
                "action_set_profile",
                "fresh_upper_nomove_n5_v3",
            ),
        )
        evidence["action_set_contract"] = {
            **dict(trusted_action_set),
            "source_path": str(
                ACTION_SET_CONTRACT_SOURCE_PATH.relative_to(REPO_ROOT)
            ),
            "source_sha256": native_diag.sha256_file(
                ACTION_SET_CONTRACT_SOURCE_PATH
            ),
        }
    except Exception as exc:
        blockers.append(f"action_set_contract:{exc}")
    training_manifest_path = getattr(args, "training_manifest", None)
    training_manifest_sha = getattr(
        args, "training_manifest_sha256", ""
    )
    physical_manifest_path = getattr(
        args, "physical_gate_manifest", None
    )
    physical_manifest_sha = getattr(
        args, "physical_gate_manifest_sha256", ""
    )
    materialization_receipt_path = getattr(
        args, "physical_gate_materialization_receipt", None
    )
    materialization_receipt_sha = getattr(
        args,
        "physical_gate_materialization_receipt_sha256",
        "",
    )
    if (
        training_manifest_path is None
        or not training_manifest_sha
        or physical_manifest_path is None
        or not physical_manifest_sha
        or materialization_receipt_path is None
        or not materialization_receipt_sha
    ):
        blockers.append(
            "missing_strict_or_physical_manifest_materialization_binding"
        )
    elif trusted_action_set is None:
        blockers.append(
            "manifests_cannot_bind_missing_trusted_action_set_contract"
        )
    else:
        try:
            if training_manifest_sha != trusted_action_set["manifest_sha256"]:
                raise FittedGateError(
                    "strict training manifest SHA differs from the trusted "
                    "action-set contract"
                )
            expected_manifest_path = (
                REPO_ROOT / trusted_action_set["manifest_path"]
            ).resolve()
            if (
                training_manifest_path.expanduser().resolve()
                != expected_manifest_path
            ):
                raise FittedGateError(
                    "strict training manifest path differs from the trusted "
                    "action-set contract"
                )
            _strict_path, strict_bytes = read_pinned_regular_file(
                training_manifest_path,
                training_manifest_sha,
                "strict action-set training manifest",
            )
            strict_manifest, strict_receipt = native_diag.read_json_exact(
                training_manifest_path,
                "strict ActionBall training manifest",
                expected_sha256=training_manifest_sha,
            )
            action_set_contract.verify_manifest_identity(
                trusted_action_set, strict_manifest, strict_bytes
            )
            _physical_path, _physical_bytes = read_pinned_regular_file(
                physical_manifest_path,
                physical_manifest_sha,
                "disposable physical-gate manifest",
            )
            raw_manifest, physical_receipt = native_diag.read_json_exact(
                physical_manifest_path,
                "disposable physical-contact-v2 manifest",
                expected_sha256=physical_manifest_sha,
            )
            physical_materialization = (
                validate_physical_materialization_closure(
                    strict_manifest=strict_manifest,
                    strict_manifest_path=training_manifest_path,
                    strict_manifest_sha256=training_manifest_sha,
                    physical_manifest=raw_manifest,
                    physical_manifest_path=physical_manifest_path,
                    physical_manifest_sha256=physical_manifest_sha,
                    receipt_path=materialization_receipt_path,
                    receipt_sha256=materialization_receipt_sha,
                    trusted_action_set=trusted_action_set,
                )
            )
            evidence["physical_materialization"] = physical_materialization
            args._physical_task_bundle_pin = dict(
                physical_materialization["physical_task_bundle"]
            )
            manifest = validate_physical_manifest(
                raw_manifest,
                trusted_action_set=trusted_action_set,
            )
            evidence["strict_training_manifest"] = strict_receipt
            evidence["physical_gate_manifest"] = physical_receipt
        except Exception as exc:
            blockers.append(f"manifest_closure:{exc}")
            if isinstance(raw_manifest, dict):
                if "physical_contact_contract" not in raw_manifest:
                    blockers.append(
                        "manifest:missing_physical_contact_contract_v2"
                    )
                order = raw_manifest.get("action_order")
                if isinstance(order, list) and "fh_loop" in order:
                    blockers.append(
                        "manifest:contains_retired_old_forehand_fh_loop"
                    )
    if args.launch_trust_root is None or not args.launch_trust_root_sha256:
        blockers.append("missing_independent_launch_evidence_trust_root")
    elif (
        manifest is None
        or not args.code_commit
        or not physical_manifest_sha
    ):
        blockers.append(
            "launch_evidence_trust_root_cannot_bind_missing_manifest_or_commit"
        )
    else:
        try:
            trust_receipt, trust_files = (
                validate_launch_evidence_trust_root(
                    path=args.launch_trust_root,
                    expected_sha256=args.launch_trust_root_sha256,
                    manifest_sha256=physical_manifest_sha,
                    expected_commit=args.code_commit,
                    manifest=manifest,
                )
            )
            evidence["launch_evidence_trust_root"] = trust_receipt
            args._launch_trust_pinned_files = tuple(trust_files)
        except Exception as exc:
            blockers.append(f"launch_evidence_trust_root:{exc}")
    if not args.profile_pins_sha256:
        blockers.append("missing_expected_profile_pins_sha256")
    else:
        try:
            pins, receipt = native_diag.read_json_exact(
                args.profile_pins,
                "profile pins",
                expected_sha256=args.profile_pins_sha256,
            )
            profile = native_diag.validate_profile_pins(
                pins, None if manifest is None else manifest.base
            )
            evidence["profile_pins"] = receipt
            if args.code_commit:
                evidence["solver_commit_blobs"] = (
                    validate_profile_sources_at_commit(
                        profile, args.code_commit
                    )
                )
        except Exception as exc:
            blockers.append(f"profile_pins:{exc}")
    try:
        identity, receipt = native_diag.read_json_exact(
            CANONICAL_IDENTITY_MANIFEST,
            "MuJoCo identity",
            expected_sha256=CANONICAL_IDENTITY_MANIFEST_SHA256,
        )
        expected = native_diag._mapping(
            identity.get("expected"), "identity.expected"
        )
        actual = native_diag.sha256_file(CANONICAL_MJCF)
        if actual != expected.get("root_mjcf_sha256"):
            raise FittedGateError("vendor MJCF root SHA mismatch")
        evidence["mujoco_identity"] = receipt
    except Exception as exc:
        blockers.append(f"mujoco_identity:{exc}")
    if manifest is not None and profile is not None:
        try:
            if (
                manifest.contract["venue_yaml_sha256"]
                != profile["venue_yaml"]["sha256"]
            ):
                raise FittedGateError(
                    "physical manifest and profile pins disagree on venue YAML"
                )
            venue = load_venue_yaml(
                Path(profile["venue_yaml"]["path"]),
                manifest.contract["venue_yaml_sha256"],
            )
            rows = (
                table_scene.action_ball_policy_obstacle_geometry()
            )
            safety_geometry_contract = (
                table_scene.action_ball_policy_geometry_contract(
                    rows
                )
            )
            evidence["scene_profile_geometry"] = (
                validate_scene_against_profile(rows, profile, venue)
            )
            evidence["five_solid_safety_geometry_contract"] = (
                safety_geometry_contract
            )
            for action in manifest.base.actions:
                launch = manifest.launches[action.action_id]
                task_binding = manifest.task_bindings[
                    action.action_id
                ]
                if (
                    dict(task_binding.solver_source_sha256)
                    != dict(
                        profile[
                            "solver_implementation_source_sha256"
                        ]
                    )
                ):
                    raise FittedGateError(
                        f"{action.action_id}: physical task binding solver "
                        "source map differs from profile pins"
                    )
                center_case_launch = task_binding.cases[0].launch
                if (
                    abs(
                        center_case_launch.activation_time_s
                        - launch.activation_time_s
                    )
                    > FORMAL_TASK_TIME_IDENTITY_TOL_S
                ):
                    raise FittedGateError(
                        f"{action.action_id}: center physical task launch is "
                        "not the independently trusted launch"
                    )
                for first, second, label in (
                    (
                        center_case_launch.position_w_m,
                        launch.position_w_m,
                        "position",
                    ),
                    (
                        center_case_launch.velocity_w_mps,
                        launch.velocity_w_mps,
                        "velocity",
                    ),
                    (
                        center_case_launch.spin_w_radps,
                        launch.spin_w_radps,
                        "spin",
                    ),
                ):
                    _require_vector_equal(
                        first,
                        second,
                        (
                            f"{action.action_id}.trusted center launch "
                            f"{label}"
                        ),
                    )
                if launch.position_w_m[0] <= float(profile["net_x_m"]) + 0.05:
                    raise FittedGateError(
                        f"{action.action_id}: physical birth is not beyond the "
                        "net plus 5 cm margin"
                    )
                if launch.velocity_w_mps[0] >= -1.0e-6:
                    raise FittedGateError(
                        f"{action.action_id}: physical birth velocity is not incoming"
                    )
                if launch.position_w_m[2] <= venue.ball_radius:
                    raise FittedGateError(
                        f"{action.action_id}: physical birth is at/below the floor"
                    )
            evidence["venue_yaml"] = {
                "path": str(venue.path),
                "sha256": venue.sha256,
            }
        except Exception as exc:
            blockers.append(f"venue:{exc}")
    return blockers, evidence, manifest, profile, venue


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", default="")
    parser.add_argument(
        "--action-set-profile",
        default="fresh_upper_nomove_n5_v3",
    )
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument(
        "--physical-gate-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--physical-gate-manifest-sha256", required=True
    )
    parser.add_argument(
        "--physical-gate-materialization-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--physical-gate-materialization-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--profile-pins",
        type=Path,
        default=native_diag.DEFAULT_PROFILE_PINS,
    )
    parser.add_argument("--profile-pins-sha256", default="")
    parser.add_argument("--launch-trust-root", type=Path)
    parser.add_argument("--launch-trust-root-sha256", default="")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--render-fps", type=int, default=30)
    return parser.parse_args(argv)


def _seal_receipt(receipt: Dict[str, Any]) -> None:
    receipt.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = native_diag.sha256_bytes(
        native_diag.canonical_json_bytes(receipt)
    )


def overlay_video_frames(
    frames: Sequence[np.ndarray],
    lines: Sequence[str],
) -> List[np.ndarray]:
    """Burn immutable action/physics facts into human-review MP4 frames."""

    if not frames:
        return []
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise FittedGateError(
            f"Pillow is required for formal video overlays: {exc}"
        ) from exc
    normalized_lines = tuple(str(line) for line in lines)
    if not normalized_lines or any(not line for line in normalized_lines):
        raise FittedGateError("formal video overlay lines must be nonempty")
    rendered: List[np.ndarray] = []
    for raw in frames:
        array = np.asarray(raw)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise FittedGateError("formal video frame shape is invalid")
        image = Image.fromarray(array.astype(np.uint8, copy=False)).convert(
            "RGB"
        )
        draw = ImageDraw.Draw(image)
        band_height = 6 + 14 * len(normalized_lines)
        draw.rectangle(
            (0, 0, image.width, min(image.height, band_height)),
            fill=(0, 0, 0),
        )
        for index, line in enumerate(normalized_lines):
            draw.text((6, 4 + 14 * index), line, fill=(255, 255, 255))
        rendered.append(np.asarray(image, dtype=np.uint8))
    return rendered


def formal_video_action_slots(action_count: int) -> Tuple[int, ...]:
    """Deterministic visual subset; numeric Gate always remains all-action."""

    if type(action_count) is not int or action_count < 1:
        raise FittedGateError("video action count must be positive")
    if action_count <= 5:
        return tuple(range(action_count))
    sample_count = min(8, action_count)
    return tuple(
        sorted(
            {
                int(round(index * (action_count - 1) / (sample_count - 1)))
                for index in range(sample_count)
            }
        )
    )


def build_teacher_return_safety_rows(
    actions_out: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten every action x case x timestep without any averaging."""

    rows: List[Dict[str, Any]] = []
    for action_row in actions_out:
        task_binding = native_diag._mapping(
            action_row.get("physical_task_binding"),
            "formal action physical_task_binding",
        )
        cases = task_binding.get("cases")
        if not isinstance(cases, list):
            raise FittedGateError(
                "formal action physical task cases must be a list"
            )
        for case_row in cases:
            case = native_diag._mapping(
                case_row, "formal physical task case"
            )
            dt_results = native_diag._mapping(
                case.get("dt_results"), "formal case dt_results"
            )
            if tuple(dt_results) != ("0.0010", "0.0005"):
                raise FittedGateError(
                    "formal return/safety rows require exact dt order"
                )
            for timestep, raw_result in dt_results.items():
                result = native_diag._mapping(
                    raw_result, "formal case dt result"
                )
                gates = native_diag._mapping(
                    result.get("mandatory_gates"),
                    "formal case mandatory gates",
                )
                safety = native_diag._mapping(
                    result.get("five_solid_robot_safety"),
                    "formal case five-solid safety",
                )
                ground_safety = native_diag._mapping(
                    result.get("ground_contact_safety"),
                    "formal case ground-contact safety",
                )
                rows.append(
                    {
                        "scope": action_row["scope"],
                        "action_id": action_row["action_id"],
                        "action_uid": action_row["action_uid"],
                        "family": action_row["family"],
                        "motion_sha256": action_row["motion_sha256"],
                        "profile_center_sha256": action_row[
                            "profile_center_sha256"
                        ],
                        "case_id": case["case_id"],
                        "case_role": case["case_role"],
                        "timestep_s": float(timestep),
                        "expected_physical_verdict": case[
                            "expected_physical_verdict"
                        ],
                        "observed_physical_verdict": result["verdict"],
                        "teacher_return_pass": bool(
                            gates[
                                "physical_ball_selected_face_return_"
                                "and_first_landing"
                            ]
                        ),
                        "teacher_five_solid_safety_pass": bool(
                            gates[
                                "teacher_robot_and_racket_five_solid_"
                                "clearance"
                            ]
                        ),
                        "five_solid_contact_count": int(
                            safety["contact_count"]
                        ),
                        "five_solid_swept_hit_count": int(
                            safety["swept_hit_count"]
                        ),
                        "teacher_ground_safety_pass": bool(
                            gates["teacher_ground_contact_safety"]
                        ),
                        "ground_contact_count": int(
                            ground_safety["contact_count"]
                        ),
                        "legal_foot_support_contact_count": int(
                            ground_safety[
                                "legal_foot_support_contact_count"
                            ]
                        ),
                        "foot_floor_penetration_violation_count": int(
                            ground_safety[
                                "foot_floor_penetration_violation_count"
                            ]
                        ),
                        "nonfoot_ground_contact_violation_count": int(
                            ground_safety[
                                "nonfoot_ground_contact_violation_count"
                            ]
                        ),
                        "fall": result["fall"],
                        "joint_limit_violation": result[
                            "joint_limit_violation"
                        ],
                        "failure_reasons": list(
                            result["failure_reasons"]
                        ),
                    }
                )
    return rows


def _reserve_receipt_path(path: Path) -> Tuple[Path, int]:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(resolved),
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except FileExistsError as exc:
        raise FittedGateError(
            f"refusing existing/case-colliding receipt path {resolved}"
        ) from exc
    return resolved, descriptor


def _write_reserved_receipt(
    path: Path, descriptor: int, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    data = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        raise FittedGateError(
            "reserved receipt pathname was replaced during runtime"
        )
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise FittedGateError("short write to reserved receipt")
        offset += written
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    readback_chunks: List[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        readback_chunks.append(chunk)
    readback = b"".join(readback_chunks)
    if readback != data:
        raise FittedGateError("reserved receipt descriptor readback mismatch")
    final_descriptor_stat = os.fstat(descriptor)
    final_path_stat = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(final_path_stat.st_mode)
        or final_descriptor_stat.st_dev != final_path_stat.st_dev
        or final_descriptor_stat.st_ino != final_path_stat.st_ino
        or final_descriptor_stat.st_size != len(data)
        or final_path_stat.st_size != len(data)
    ):
        raise FittedGateError(
            "reserved receipt pathname changed during/after durable write"
        )
    parent_descriptor = os.open(
        str(path.parent),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    receipt_sha256 = native_diag.sha256_bytes(readback)
    os.close(descriptor)
    return {
        "path": str(path),
        "sha256": receipt_sha256,
        "size_bytes": len(readback),
        "device": int(final_descriptor_stat.st_dev),
        "inode": int(final_descriptor_stat.st_ino),
        "descriptor_readback_verified": True,
        "pathname_identity_verified_after_write": True,
        "parent_directory_fsynced": True,
    }


def materialize_physical_task_control(
    action: native_diag.ActionSpec,
    task_case: PhysicalTaskCase,
) -> Tuple[
    native_diag.ActionSpec,
    CaseLaunchState,
    Optional[float],
    Dict[str, Any],
]:
    """Apply exactly one preregistered fault to an otherwise frozen case."""

    fault = native_diag._mapping(
        task_case.raw["fault_injection"],
        f"{action.action_id}.{task_case.case_role}.fault_injection",
    )
    kind = str(fault["kind"])
    executed_action = action
    executed_launch = task_case.launch
    wait_override: Optional[float] = None
    applied: Dict[str, Any] = {
        "kind": kind,
        "applied": kind == "none",
        "nominal_mount_normal_sign": action.mount_normal_sign,
        "executed_mount_normal_sign": action.mount_normal_sign,
        "nominal_pre_swing_wait_s": task_case.pre_swing_wait_s,
        "executed_pre_swing_wait_s": task_case.pre_swing_wait_s,
        "nominal_launch_velocity_w_mps": (
            task_case.launch.velocity_w_mps.tolist()
        ),
        "executed_launch_velocity_w_mps": (
            task_case.launch.velocity_w_mps.tolist()
        ),
    }
    if kind == "none":
        return executed_action, executed_launch, wait_override, applied
    if kind == "teacher_t_hit_offset":
        offset = float(fault["offset_s"])
        wait_override = task_case.pre_swing_wait_s + offset
        if wait_override < 0.0 or wait_override > 1.0:
            raise FittedGateError(
                f"{action.action_id}.{task_case.case_role}: injected wait "
                "left the physical replay horizon"
            )
        applied.update(
            {
                "applied": True,
                "offset_s": offset,
                "executed_pre_swing_wait_s": wait_override,
            }
        )
    elif kind == "selected_face_sign_flip":
        executed_action = replace(
            action, mount_normal_sign=-action.mount_normal_sign
        )
        applied.update(
            {
                "applied": True,
                "executed_mount_normal_sign": (
                    executed_action.mount_normal_sign
                ),
            }
        )
    elif kind == "launch_velocity_delta":
        delta = np.asarray(
            fault["launch_velocity_delta_w_mps"], np.float64
        )
        injected_velocity = task_case.launch.velocity_w_mps + delta
        if injected_velocity[0] >= -1.0e-6:
            raise FittedGateError(
                f"{action.action_id}.{task_case.case_role}: injected ball "
                "state is no longer an incoming ball"
            )
        executed_launch = replace(
            task_case.launch,
            velocity_w_mps=injected_velocity,
        )
        applied.update(
            {
                "applied": True,
                "launch_velocity_delta_w_mps": delta.tolist(),
                "executed_launch_velocity_w_mps": (
                    injected_velocity.tolist()
                ),
            }
        )
    else:
        raise FittedGateError(
            f"{action.action_id}.{task_case.case_role}: unsupported fault {kind!r}"
        )
    return executed_action, executed_launch, wait_override, applied


def evaluate_physical_task_control(
    task_case: PhysicalTaskCase,
    dt_results: Mapping[str, Mapping[str, Any]],
    convergence: Mapping[str, Any],
    fault_application: Mapping[str, Any],
) -> Dict[str, Any]:
    """Turn observed physics into a positive/negative control verdict."""

    observed_dt_verdicts = {
        key: str(value["verdict"])
        for key, value in dt_results.items()
    }
    positive = task_case.case_role in PHYSICAL_TASK_POSITIVE_ROLES
    if positive:
        observed = (
            "PASS"
            if all(value == "PASS" for value in observed_dt_verdicts.values())
            and convergence.get("pass") is True
            else "FAIL"
        )
        control_pass = observed == task_case.expected_physical_verdict
        observed_reason = None
        failure_reasons = (
            []
            if control_pass
            else ["positive_control_physical_replay_failed"]
        )
    else:
        both_failed = all(
            value == "FAIL" for value in observed_dt_verdicts.values()
        )
        no_legal_return = all(
            result.get("mandatory_gates", {}).get(
                "physical_ball_selected_face_return_and_first_landing"
            )
            is False
            for result in dt_results.values()
        )
        generic_signatures = {
            "negative_t_hit_offset": {
                "teacher_physical_face_center_target_mismatch",
                "teacher_task_site_target_mismatch",
                "teacher_task_face_velocity_mismatch",
                "teacher_task_site_velocity_mismatch",
                "physical_contact_time_mismatch",
                "fitted_paddle_impulse_count_not_exactly_one",
            },
            "negative_face_sign": {
                "teacher_physical_face_center_target_mismatch",
                "teacher_task_site_target_mismatch",
                "teacher_task_face_normal_mismatch",
                "teacher_selected_face_not_oriented_toward_opponent",
                "fitted_paddle_impulse_count_not_exactly_one",
            },
            "negative_ball_state_mismatch": {
                "physical_incoming_velocity_mismatch",
                "physical_contact_time_mismatch",
                "physical_contact_position_mismatch",
                "no_post_hit_net_crossing",
                "no_fitted_first_table_landing",
                "first_landing_misses_frozen_task_aim",
                "fitted_paddle_impulse_count_not_exactly_one",
            },
        }[task_case.case_role]
        observed_failure_rows = [
            reason
            for result in dt_results.values()
            for reason in result.get("failure_reasons", ())
        ]
        signature_seen = any(
            reason in generic_signatures
            for reason in observed_failure_rows
        )
        control_pass = (
            fault_application.get("applied") is True
            and both_failed
            and no_legal_return
            and signature_seen
        )
        observed = "FAIL" if both_failed else "PASS"
        observed_reason = (
            task_case.expected_failure_reason
            if control_pass
            else None
        )
        failure_reasons = (
            []
            if control_pass
            else ["negative_control_did_not_fail_as_preregistered"]
        )
    return {
        "expected_physical_verdict": (
            task_case.expected_physical_verdict
        ),
        "expected_failure_reason": task_case.expected_failure_reason,
        "observed_physical_verdict": observed,
        "observed_failure_reason": observed_reason,
        "observed_dt_verdicts": observed_dt_verdicts,
        "fault_application": dict(fault_application),
        "convergence_required": positive,
        "convergence_pass": (
            convergence.get("pass") if positive else None
        ),
        "control_verdict": "PASS" if control_pass else "FAIL",
        "failure_reasons": failure_reasons,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        args.out, receipt_descriptor = _reserve_receipt_path(args.out)
    except Exception as exc:
        print(
            f"[fitted-ball-gate][FATAL] {exc}",
            file=sys.stderr,
        )
        return 2
    blockers, evidence, manifest, profile, venue = _preflight(args)
    if not args.preflight_only and args.render_dir is None:
        blockers.append(
            "missing_required_per_action_physical_video_render_dir"
        )
    runtime_execution_attestation = globals().get(
        "RUNTIME_EXECUTION_ATTESTATION"
    )
    if runtime_execution_attestation is None:
        blockers.append("missing_pinned_runtime_execution_attestation")
    elif manifest is None or not args.code_commit:
        blockers.append(
            "runtime_execution_attestation_cannot_bind_missing_manifest_or_commit"
        )
    else:
        try:
            evidence["runtime_execution"] = (
                validate_runtime_execution_attestation(
                    runtime_execution_attestation,
                    manifest=manifest,
                    expected_commit=args.code_commit,
                    expected_input_bindings={
                        "strict_training_manifest": (
                            args.training_manifest,
                            args.training_manifest_sha256,
                        ),
                        "physical_gate_manifest": (
                            args.physical_gate_manifest,
                            args.physical_gate_manifest_sha256,
                        ),
                        "physical_gate_materialization_receipt": (
                            args.physical_gate_materialization_receipt,
                            args.physical_gate_materialization_receipt_sha256,
                        ),
                        "profile_pins": (
                            args.profile_pins,
                            args.profile_pins_sha256,
                        ),
                        "launch_evidence_trust_root": (
                            Path(args.launch_trust_root),
                            args.launch_trust_root_sha256,
                        ),
                    },
                    require_all_modules_loaded=False,
                )
            )
        except Exception as exc:
            blockers.append(f"runtime_execution:{exc}")
    safety_geometry_contract = evidence.get(
        "five_solid_safety_geometry_contract"
    )
    if (
        not blockers
        and not isinstance(safety_geometry_contract, dict)
    ):
        blockers.append(
            "missing_five_solid_safety_geometry_contract"
        )
    materialization_evidence = evidence.get("physical_materialization")
    materialization_schema_version = (
        materialization_evidence.get("schema_version")
        if isinstance(materialization_evidence, Mapping)
        else None
    )
    materialization_kind = (
        materialization_evidence.get("kind")
        if isinstance(materialization_evidence, Mapping)
        else None
    )
    schema2_materialization = bool(
        materialization_schema_version == 2
        and materialization_kind
        == GENERIC_PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND
    )
    if (
        materialization_schema_version == 2
        and not schema2_materialization
    ):
        blockers.append(
            "schema2_materialization_kind_is_not_exact_generic_kind"
        )
    formal_action_identity_matrix = (
        materialization_action_identity_matrix(
            manifest.raw, manifest.action_set_contract
        )
        if manifest is not None and schema2_materialization
        else None
    )
    receipt: Dict[str, Any] = {
        "schema_version": (
            2
            if schema2_materialization
            else SCHEMA_VERSION
        ),
        "gate": "mujoco_teacher_motion_fitted_ball_gate",
        "materialization_receipt_schema_version": (
            materialization_schema_version
        ),
        "materialization_receipt_kind": (
            materialization_evidence.get("kind")
            if isinstance(materialization_evidence, Mapping)
            else None
        ),
        "contact_authority": CONTACT_AUTHORITY,
        "native_ball_contact_enabled": False,
        "selector_executed": False,
        "ball_to_task_solver_executed": False,
        "ball_to_task_solver_executed_by_gate": False,
        "pre_registered_ball_to_task_solver_receipt_consumed": (
            manifest is not None
            and set(manifest.task_bindings)
            == set(manifest.base.action_order)
        ),
        "solver_execution_receipt_authority": (
            PHYSICAL_TASK_BINDING_AUTHORITY
        ),
        "analytic_return_scorer_executed": False,
        "teacher_return_safety_rows": [],
        "five_solid_safety_scene": (
            None
            if not isinstance(safety_geometry_contract, dict)
            else {
                "five_solid_geometry_sha256": (
                    safety_geometry_contract["sha256"]
                ),
                "geometry_payload": (
                    safety_geometry_contract["payload"]
                ),
                "obstacle_order": list(
                    table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
                ),
                "under_table_keepout_role": "robot_only",
                "ball_keepout_native_pair_enabled": False,
                "ball_keepout_analytic_surface_enabled": False,
                "contact_force_threshold_n": (
                    TABLE_CONTACT_FORCE_THRESHOLD_N
                ),
                "continuous_sweep_method": (
                    FIVE_SOLID_SWEEP_METHOD
                ),
                "ground_contact_policy": {
                    "floor_geom_name": FLOOR_GEOM_NAME,
                    "legal_foot_body_names": list(
                        LEGAL_FOOT_BODY_NAMES
                    ),
                    "all_collision_enabled_robot_geoms_floor_pair_enabled": (
                        True
                    ),
                    "foot_floor_penetration_tolerance_m": (
                        FOOT_FLOOR_PENETRATION_TOLERANCE_M
                    ),
                    "nonfoot_floor_penetration_tolerance_m": (
                        NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
                    ),
                    "nonfoot_force_threshold_n": (
                        TABLE_CONTACT_FORCE_THRESHOLD_N
                    ),
                    "continuous_nonfoot_clearance_guard_m": (
                        FORMAL_NONFOOT_GROUND_CLEARANCE_GUARD_M
                    ),
                    "continuous_distance_query_cap_m": (
                        FORMAL_GROUND_DISTANCE_QUERY_CAP_M
                    ),
                },
                "compiled_by_dt": {},
            }
        ),
        "expected_actions": (
            int(manifest.action_set_contract["expected_n"])
            if manifest is not None
            else None
        ),
        "expected_action_order": (
            list(manifest.action_set_contract["ordered_action_ids"])
            if manifest is not None
            else None
        ),
        "action_set_contract": (
            dict(manifest.action_set_contract)
            if manifest is not None
            else None
        ),
        "preflight": {
            "status": "BLOCKED" if blockers else "PASS",
            "blockers": blockers,
            "evidence": evidence,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "runtime_code_identity": (
            runtime_execution_attestation
            if isinstance(runtime_execution_attestation, dict)
            else None
        ),
    }
    if formal_action_identity_matrix is not None:
        receipt["action_identity_matrix"] = (
            formal_action_identity_matrix
        )
        receipt["action_identity_matrix_sha256"] = (
            native_diag.sha256_bytes(
                native_diag.canonical_json_bytes(
                    formal_action_identity_matrix
                )
            )
        )
    if args.preflight_only or blockers:
        receipt["status"] = "BLOCKED" if blockers else "PREFLIGHT_PASS"
        receipt["verdict"] = "BLOCKED" if blockers else "NOT_RUN"
        receipt["formal_gate_executed"] = False
        receipt["actions"] = []
        _seal_receipt(receipt)
        _write_reserved_receipt(
            args.out, receipt_descriptor, receipt
        )
        receipt_descriptor = -1
        print(
            f"[fitted-ball-gate] {receipt['status']} "
            f"blockers={len(blockers)} receipt={args.out}"
        )
        return 3 if blockers else 4
    assert manifest is not None and profile is not None and venue is not None
    receipt["formal_gate_executed"] = True
    assert args.render_dir is not None
    render_dir = args.render_dir.expanduser().resolve()
    try:
        render_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        receipt.update(
            {
                "status": "INFRASTRUCTURE_FAIL",
                "verdict": "FAIL",
                "error": (
                    "FittedGateError: refusing existing/case-colliding "
                    f"render dir {render_dir}"
                ),
            }
        )
        _seal_receipt(receipt)
        _write_reserved_receipt(
            args.out, receipt_descriptor, receipt
        )
        receipt_descriptor = -1
        return 2
    try:
        import mujoco
        from canonical_mujoco_identity import verify_exact_mujoco_identity

        receipt["runtime_environment"] = {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "python_cache_tag": sys.implementation.cache_tag,
            "mujoco_version": str(getattr(mujoco, "__version__", "")),
            "mujoco_module_path": str(
                Path(str(mujoco.__file__)).resolve()
            ),
        }

        runtime_input_snapshot = capture_runtime_input_snapshot(
            args=args,
            manifest=manifest,
            profile=profile,
            venue=venue,
        )
        receipt["runtime_input_snapshot"] = {
            "phase": "captured_before_runtime",
            "files": runtime_input_snapshot,
        }
        verified = verify_exact_mujoco_identity(
            mjcf_path=CANONICAL_MJCF,
            expected_manifest_path=CANONICAL_IDENTITY_MANIFEST,
            trusted_expected_manifest_sha256=(
                CANONICAL_IDENTITY_MANIFEST_SHA256
            ),
        )
        runtime_input_snapshot = extend_snapshot_with_mujoco_source_closure(
            runtime_input_snapshot,
            model_root=CANONICAL_MJCF.resolve().parent,
            source_closure=verified.receipt["source_closure"],
        )
        receipt["runtime_input_snapshot"]["files"] = runtime_input_snapshot
        expected_root_mjcf_sha = next(
            row["sha256"]
            for row in runtime_input_snapshot
            if "vendor_root_mjcf" in row["roles"]
        )
        _canonical_path, canonical_xml = read_pinned_regular_file(
            CANONICAL_MJCF,
            expected_root_mjcf_sha,
            "vendor root MJCF model-build input",
        )
        obstacle_rows = (
            table_scene.action_ball_policy_obstacle_geometry()
        )
        runtime_safety_geometry_contract = (
            table_scene.action_ball_policy_geometry_contract(
                obstacle_rows
            )
        )
        if (
            not isinstance(safety_geometry_contract, dict)
            or runtime_safety_geometry_contract
            != safety_geometry_contract
        ):
            raise FittedGateError(
                "five-solid geometry changed after preflight"
            )
        assets = table_scene._mesh_assets(
            canonical_xml, CANONICAL_MJCF.resolve().parent
        )
        compiler_asset_receipts = (
            verify_compiler_assets_against_source_closure(
                assets,
                verified.receipt["source_closure"],
            )
        )
        models: Dict[float, Any] = {}
        scenes: Dict[str, Any] = {}
        for dt in DEFAULT_DT_S:
            four_solid_xml, scene = assemble_fitted_scene_xml(
                canonical_xml, obstacle_rows, venue, dt
            )
            xml = table_scene.append_action_ball_policy_keepout_xml(
                four_solid_xml,
                obstacle_rows,
                collidable=True,
            )
            assembled_xml_sha = native_diag.sha256_bytes(xml)
            model = mujoco.MjModel.from_xml_string(
                xml.decode("utf-8"), assets=assets
            )
            ball_geom = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME
                )
            )
            ball_body = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY_NAME
                )
            )
            if min(ball_geom, ball_body) < 0:
                raise FittedGateError(
                    "compiled model lacks fitted ball body/geom"
                )
            if (
                int(model.geom_contype[ball_geom]) != 0
                or int(model.geom_conaffinity[ball_geom]) != 0
            ):
                raise FittedGateError(
                    "compiled model re-enabled native ball contact"
                )
            expected_inertia = (
                venue.inertia_coeff
                * venue.ball_mass
                * venue.ball_radius**2
            )
            if (
                abs(float(model.body_mass[ball_body]) - venue.ball_mass)
                > 1.0e-12
                or float(
                    np.max(
                        np.abs(
                            np.asarray(model.body_inertia[ball_body])
                            - expected_inertia
                        )
                    )
                )
                > 1.0e-12
                or abs(float(model.geom_size[ball_geom, 0]) - venue.ball_radius)
                > 1.0e-12
                or float(
                    np.max(
                        np.abs(
                            np.asarray(model.opt.gravity)
                            - np.asarray((0.0, 0.0, -venue.gravity))
                        )
                    )
                )
                > 1.0e-12
                or abs(float(model.opt.timestep) - dt) > 1.0e-15
            ):
                raise FittedGateError(
                    "compiled ball mass/radius/inertia/gravity/timestep "
                    "drifted from the physical contract"
                )
            scene = dict(scene)
            scene["four_solid_fitted_scene_xml_sha256"] = scene[
                "fitted_scene_xml_sha256"
            ]
            scene["fitted_scene_xml_sha256"] = assembled_xml_sha
            scene.update(
                validate_compiled_obstacles(
                    mujoco,
                    model,
                    obstacle_rows,
                    runtime_safety_geometry_contract,
                    assembled_xml_sha256=assembled_xml_sha,
                )
            )
            scene["ground_contact_safety_contract"] = (
                ground_contact_contract_receipt(
                    mujoco,
                    model,
                    build_ground_contact_contract(
                        mujoco,
                        model,
                        ball_geom_id=ball_geom,
                    ),
                )
            )
            models[dt] = model
            scenes[format(dt, ".4f")] = scene
        receipt["five_solid_safety_scene"][
            "compiled_by_dt"
        ] = {
            key: {
                field: row[field]
                for field in (
                    "five_solid_geometry_sha256",
                    "assembled_xml_sha256",
                    "compiled_obstacles",
                    "physics_enabled_robot_geom_count",
                    "teacher_swept_subject",
                    "ball_keepout_native_pair_enabled",
                    "ball_keepout_analytic_surface_enabled",
                    "contact_force_threshold_n",
                    "continuous_sweep_method",
                    "ground_contact_safety_contract",
                )
            }
            for key, row in scenes.items()
        }
        actions_out: List[Dict[str, Any]] = []
        clips = {
            action.action_id: load_motion_from_pinned_bytes(action)
            for action in manifest.base.actions
        }
        ready_recovery = ready_recovery_metrics(
            clips,
            manifest.base.action_order,
        )
        materialized_identity_by_action = {
            row["action_id"]: row
            for row in (
                formal_action_identity_matrix
                if formal_action_identity_matrix is not None
                else materialization_action_identity_matrix(
                    manifest.raw, manifest.action_set_contract
                )
            )
        }
        shared_ready = ready_recovery["shared_ready"]
        video_slots = frozenset(
            formal_video_action_slots(len(manifest.base.actions))
        )
        for action_slot, action in enumerate(manifest.base.actions):
            video_required = action_slot in video_slots
            clip = clips[action.action_id]
            action_recovery = ready_recovery[
                "recovery_by_action"
            ][action.action_id]
            center = native_diag.center_ball_state(action, clip)
            task_binding = manifest.task_bindings[action.action_id]
            nominal_face_mesh = load_binary_stl_face(
                action.mount_normal_sign
            )
            face_name = (
                "red" if action.mount_normal_sign == 1 else "black"
            )
            if (
                nominal_face_mesh.sha256
                != manifest.contract["selected_face_mesh_sha256"][face_name]
            ):
                raise FittedGateError(
                    f"{action.action_id}: selected face mesh changed before runtime"
                )
            teacher_ready_root = np.asarray(
                clip.body_pos_w[0, 0], np.float64
            )
            for task_case in task_binding.cases:
                _require_vector_equal(
                    task_case.base_spawn_w_m,
                    teacher_ready_root,
                    (
                        f"{action.action_id}.{task_case.case_role} "
                        "task base spawn/teacher ready root"
                    ),
                    tolerance=2.0e-4,
                )
            for center_case in task_binding.cases[:2]:
                _require_vector_equal(
                    center_case.ball_contact_w_m,
                    center["contact_position_w_m"],
                    (
                        f"{action.action_id}.{center_case.case_role} "
                        "profile-center contact"
                    ),
                    tolerance=2.0e-9,
                )
                _require_vector_equal(
                    center_case.incoming_velocity_w_mps,
                    center["incoming_velocity_w_mps"],
                    (
                        f"{action.action_id}.{center_case.case_role} "
                        "profile-center incoming velocity"
                    ),
                    tolerance=2.0e-9,
                )
                _require_vector_equal(
                    center_case.incoming_spin_w_radps,
                    center["spin_w_radps"],
                    (
                        f"{action.action_id}.{center_case.case_role} "
                        "profile-center spin"
                    ),
                    tolerance=2.0e-9,
                )
                if (
                    abs(
                        center_case.time_to_contact_s
                        - float(center["time_to_contact_s"])
                    )
                    > FORMAL_TASK_TIME_IDENTITY_TOL_S
                ):
                    raise FittedGateError(
                        f"{action.action_id}.{center_case.case_role}: "
                        "not the exact profile-center time-to-contact"
                    )
            case_rows: List[Dict[str, Any]] = []
            video_frames: List[np.ndarray] = []
            for task_case in task_binding.cases:
                (
                    executed_action,
                    executed_launch,
                    wait_override,
                    fault_application,
                ) = materialize_physical_task_control(
                    action, task_case
                )
                executed_face_mesh = load_binary_stl_face(
                    executed_action.mount_normal_sign
                )
                executed_face_name = (
                    "red"
                    if executed_action.mount_normal_sign == 1
                    else "black"
                )
                if (
                    executed_face_mesh.sha256
                    != manifest.contract[
                        "selected_face_mesh_sha256"
                    ][executed_face_name]
                ):
                    raise FittedGateError(
                        f"{action.action_id}.{task_case.case_role}: "
                        "executed face mesh bytes drifted"
                    )
                case_dt_results: Dict[str, Any] = {}
                for dt in DEFAULT_DT_S:
                    result, frames = run_action_dt(
                        mujoco=mujoco,
                        model=models[dt],
                        action=executed_action,
                        clip=clip,
                        launch=executed_launch,
                        task_case=task_case,
                        venue=venue,
                        profile=profile,
                        face_mesh=executed_face_mesh,
                        obstacle_rows=obstacle_rows,
                        post_contact_s=FORMAL_POST_CONTACT_S,
                        contact_time_tolerance_s=(
                            FORMAL_CONTACT_TIME_TOL_S
                        ),
                        contact_position_tolerance_m=(
                            FORMAL_CONTACT_POSITION_TOL_M
                        ),
                        capture_frames=(
                            video_required
                            and
                            task_case.case_role
                            == "center_positive_seed_0"
                            and dt == DEFAULT_DT_S[0]
                        ),
                        render_fps=args.render_fps,
                        teacher_wait_override_s=wait_override,
                    )
                    result["ready_recovery"] = {
                        "shared_ready": shared_ready,
                        "action_recovery": action_recovery,
                        "recovery_thresholds": ready_recovery[
                            "recovery_thresholds"
                        ],
                        "grounded_bank_evidence": manifest.contract[
                            "_candidate_pre_admission_evidence"
                        ][action.action_id]["grounded_evidence"],
                    }
                    case_dt_results[format(dt, ".4f")] = result
                    if frames:
                        video_frames = frames
                case_convergence = compare_convergence(
                    case_dt_results["0.0010"],
                    case_dt_results["0.0005"],
                )
                control = evaluate_physical_task_control(
                    task_case,
                    case_dt_results,
                    case_convergence,
                    fault_application,
                )
                task_payload = task_case.raw["task_payload"]
                solved_geometry_payload = {
                    key: task_payload[key]
                    for key in (
                        "mount_normal_sign",
                        "ball_contact_w_m",
                        "racket_site_target_w_m",
                        "racket_normal_w",
                        "reference_racket_quat_wxyz",
                        "reference_racket_angular_velocity_w_radps",
                        "racket_command_quat_wxyz",
                        "racket_face_center_velocity_w_mps",
                        "racket_site_velocity_w_mps",
                        "racket_command_angular_velocity_w_radps",
                        "geometry_source_sha256",
                        "landing_aim_w_xy_m",
                    )
                }
                case_rows.append(
                    {
                        "case_id": task_case.case_id,
                        "case_role": task_case.case_role,
                        "sample_seed": task_case.sample_seed,
                        "expected_physical_verdict": (
                            task_case.expected_physical_verdict
                        ),
                        "expected_failure_reason": (
                            task_case.expected_failure_reason
                        ),
                        "ball_proposal_sha256": (
                            task_case.ball_proposal_sha256
                        ),
                        "task_payload_sha256": (
                            task_case.task_payload_sha256
                        ),
                        "solved_task_geometry_sha256": (
                            _canonical_payload_sha256(
                                solved_geometry_payload
                            )
                        ),
                        "case_binding_sha256": (
                            task_case.case_binding_sha256
                        ),
                        "solver_execution_identity": dict(
                            task_binding.raw[
                                "solver_execution_identity"
                            ]
                        ),
                        "task_timing": {
                            "teacher_rate": task_case.teacher_rate,
                            "scaled_t_hit_s": (
                                task_case.scaled_t_hit_s
                            ),
                            "scaled_t_cycle_s": (
                                task_case.scaled_t_cycle_s
                            ),
                            "pre_swing_wait_s": (
                                task_case.pre_swing_wait_s
                            ),
                        },
                        "task_geometry": solved_geometry_payload,
                        "dt_results": case_dt_results,
                        "convergence": case_convergence,
                        "control": control,
                        "observed_physical_verdict": control[
                            "observed_physical_verdict"
                        ],
                        "control_verdict": control[
                            "control_verdict"
                        ],
                        "failure_reasons": control[
                            "failure_reasons"
                        ],
                    }
                )
            reasons: List[str] = []
            for case_row in case_rows:
                if case_row["control_verdict"] != "PASS":
                    reasons.append(
                        f"{case_row['case_role']}:control_failed"
                    )
            if (
                shared_ready["joint_linf_rad"]
                > FORMAL_SHARED_READY_JOINT_TOL_RAD
            ):
                reasons.append("shared_ready_bank_mismatch")
            if (
                shared_ready["root_position_l2_m"]
                > FORMAL_SHARED_READY_ROOT_POSITION_TOL_M
            ):
                reasons.append("shared_ready_root_position_mismatch")
            if (
                shared_ready["root_orientation_angle_rad"]
                > FORMAL_SHARED_READY_ROOT_ORIENTATION_TOL_RAD
            ):
                reasons.append("shared_ready_root_orientation_mismatch")
            if (
                max(
                    shared_ready[
                        "endpoint_joint_velocity_peak_radps"
                    ],
                    shared_ready[
                        "endpoint_root_linear_velocity_peak_mps"
                    ],
                    shared_ready[
                        "endpoint_root_angular_velocity_peak_radps"
                    ],
                )
                > FORMAL_ENDPOINT_VELOCITY_TOL
            ):
                reasons.append("shared_ready_endpoint_velocity_nonzero")
            if (
                action_recovery["joint_linf_rad"]
                > FORMAL_RECOVERY_JOINT_TOL_RAD
            ):
                reasons.append("teacher_does_not_recover_to_ready")
            if (
                action_recovery["root_position_l2_m"]
                > FORMAL_RECOVERY_ROOT_POSITION_TOL_M
            ):
                reasons.append(
                    "teacher_root_position_does_not_recover_to_ready"
                )
            if (
                action_recovery["root_orientation_angle_rad"]
                > FORMAL_RECOVERY_ROOT_ORIENTATION_TOL_RAD
            ):
                reasons.append(
                    "teacher_root_orientation_does_not_recover_to_ready"
                )
            if (
                max(
                    action_recovery[
                        "endpoint_joint_velocity_peak_radps"
                    ],
                    action_recovery[
                        "endpoint_root_linear_velocity_peak_mps"
                    ],
                    action_recovery[
                        "endpoint_root_angular_velocity_peak_radps"
                    ],
                )
                > FORMAL_ENDPOINT_VELOCITY_TOL
            ):
                reasons.append(
                    "teacher_ready_recovery_endpoint_velocity_nonzero"
                )
            center_result = case_rows[0]["dt_results"]["0.0010"]
            center_contact = center_result.get("paddle_contact") or {}
            center_landing = center_result.get("first_landing") or {}
            overlay_lines = (
                f"teacher action={action.action_id}",
                (
                    "t_hit={:.4f}s t_cycle={:.4f}s".format(
                        float(
                            center_result["simulation_window"][
                                "scaled_t_hit_s"
                            ]
                        ),
                        float(
                            center_result["simulation_window"][
                                "scaled_t_cycle_s"
                            ]
                        ),
                    )
                ),
                (
                    "contact_u_n={:.4f} m/s".format(
                        float(
                            center_contact.get(
                                "relative_normal_speed_mps",
                                float("nan"),
                            )
                        )
                    )
                ),
                (
                    "landing={} table_contacts={} net_collision={}".format(
                        center_landing.get("ball_center_xy_m"),
                        len(center_result.get("table_contacts") or ()),
                        bool(center_result.get("ball_net_collision")),
                    )
                ),
            )
            if video_required:
                video = native_diag._render_video(
                    overlay_video_frames(video_frames, overlay_lines),
                    render_dir
                    / f"{action.action_id}_fitted_teacher_ball.mp4",
                    args.render_fps,
                )
                video["overlay"] = {
                    "burned_in": True,
                    "lines": list(overlay_lines),
                }
            else:
                video = {
                    "status": "NOT_SAMPLED_NUMERIC_GATE_COMPLETE",
                    "path": None,
                    "overlay": {
                        "burned_in": False,
                        "lines": list(overlay_lines),
                    },
                }
            video["capsule_relative_path"] = (
                "artifacts/videos/"
                f"{action.action_id}_fitted_teacher_ball.mp4"
            )
            video["camera"] = "torso_follow"
            video["evidence_role"] = (
                "human_visualization_only_not_physical_or_analytic_grader"
            )
            if video_required and video.get("status") != "WRITTEN":
                reasons.append("required_video_not_written")
            elif video_required and int(video.get("frames", -1)) != int(
                case_rows[0]["dt_results"]["0.0010"][
                    "simulation_window"
                ]["expected_render_frames"]
            ):
                reasons.append("required_video_frame_coverage_incomplete")
            actions_out.append(
                {
                    "action_id": action.action_id,
                    "action_uid": action.action_uid,
                    "scope": materialized_identity_by_action[
                        action.action_id
                    ]["scope"],
                    "family": materialized_identity_by_action[
                        action.action_id
                    ]["family"],
                    "motion_path": str(action.motion_path),
                    "motion_sha256": action.motion_sha256,
                    "profile_center": materialized_identity_by_action[
                        action.action_id
                    ]["profile_center"],
                    "profile_center_sha256": (
                        materialized_identity_by_action[
                            action.action_id
                        ]["profile_center_sha256"]
                    ),
                    "launch": {
                        "source": manifest.launches[action.action_id].source,
                        "state_sha256": manifest.launches[
                            action.action_id
                        ].state_sha256,
                        "source_receipt": manifest.launch_source_receipts[
                            action.action_id
                        ],
                    },
                    "face_geometry": {
                        "sign": action.mount_normal_sign,
                        "mesh_path": str(nominal_face_mesh.path),
                        "mesh_sha256": nominal_face_mesh.sha256,
                        "outer_triangle_count": int(
                            nominal_face_mesh.triangles_xz_m.shape[0]
                        ),
                        "geometry_contract_sha256": (
                            manifest.base.racket_geometry_contract[
                                "source_sha256"
                            ]
                        ),
                    },
                    "t_hit_s": action.t_hit_s,
                    "t_cycle_s": action.t_cycle_s,
                    "reference_racket_site_speed_mps": action.racket_speed_mps,
                    "dt_results": case_rows[0]["dt_results"],
                    "convergence": case_rows[0]["convergence"],
                    "physical_task_binding": {
                        "ball_profile_sha256": (
                            task_binding.ball_profile_sha256
                        ),
                        "solver_profile_sha256": (
                            task_binding.solver_profile_sha256
                        ),
                        "physics_profile_sha256": (
                            task_binding.physics_profile_sha256
                        ),
                        "solver_source_sha256": dict(
                            task_binding.solver_source_sha256
                        ),
                        "solver_execution_receipt": {
                            "path": str(
                                task_binding.solver_execution_receipt_path
                            ),
                            "sha256": (
                                task_binding.solver_execution_receipt_sha256
                            ),
                            "receipt_payload_sha256": (
                                task_binding
                                .solver_execution_receipt_payload_sha256
                            ),
                        },
                        "cases_sha256": task_binding.cases_sha256,
                        "case_order": list(
                            PHYSICAL_TASK_CASE_ROLES
                        ),
                        "cases": case_rows,
                    },
                    "shared_ready_joint_linf_rad": (
                        shared_ready["joint_linf_rad"]
                    ),
                    "recovery_joint_linf_rad": (
                        action_recovery["joint_linf_rad"]
                    ),
                    "video": video,
                    "verdict": "PASS" if not reasons else "FAIL",
                    "failure_reasons": reasons,
                }
            )
        expected_video_names = {
            f"{action.action_id}_fitted_teacher_ball.mp4"
            for index, action in enumerate(manifest.base.actions)
            if index in video_slots
        }
        observed_video_paths = tuple(sorted(render_dir.iterdir()))
        if (
            {path.name for path in observed_video_paths}
            != expected_video_names
            or any(
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size <= 0
                for path in observed_video_paths
            )
        ):
            raise FittedGateError(
                "formal render directory must contain exactly the deterministic "
                "nonempty sampled MP4 set"
            )
        video_by_action = {
            row["action_id"]: row["video"] for row in actions_out
        }
        for index, action in enumerate(manifest.base.actions):
            if index not in video_slots:
                continue
            video_path = (
                render_dir
                / f"{action.action_id}_fitted_teacher_ball.mp4"
            )
            video = video_by_action[action.action_id]
            if (
                video.get("status") != "WRITTEN"
                or native_diag.sha256_file(video_path)
                != video.get("sha256")
            ):
                raise FittedGateError(
                    f"{action.action_id}: formal video bytes changed "
                    "before receipt seal"
                )
        scenes["formal_video_set"] = {
            "required": True,
            "camera": "torso_follow",
            "render_timestep_s": DEFAULT_DT_S[0],
            "fps": args.render_fps,
            "action_order": list(manifest.base.action_order),
            "sampled_action_slots": sorted(video_slots),
            "numeric_gate_action_count": len(manifest.base.actions),
            "files": [
                {
                    "action_id": action.action_id,
                    "capsule_relative_path": video_by_action[
                        action.action_id
                    ]["capsule_relative_path"],
                    "sha256": video_by_action[
                        action.action_id
                    ]["sha256"],
                    "frames": video_by_action[
                        action.action_id
                    ]["frames"],
                }
                for index, action in enumerate(manifest.base.actions)
                if index in video_slots
            ],
            "evidence_role": (
                "human_visualization_only_not_physical_or_analytic_grader"
            ),
        }
        overall = all(row["verdict"] == "PASS" for row in actions_out)
        teacher_return_safety_rows = build_teacher_return_safety_rows(
            actions_out
        )
        verified.assert_model_unchanged()
        runtime_finalizer = globals().get(
            "RUNTIME_EXECUTION_FINALIZER"
        )
        if not callable(runtime_finalizer):
            raise FittedGateError(
                "pinned runtime execution finalizer is missing"
            )
        receipt["runtime_code_identity_post_runtime"] = (
            runtime_finalizer()
        )
        receipt["runtime_code_identity_final"] = (
            validate_runtime_execution_attestation(
                runtime_execution_attestation,
                manifest=manifest,
                expected_commit=args.code_commit,
                expected_input_bindings={
                    "strict_training_manifest": (
                        args.training_manifest,
                        args.training_manifest_sha256,
                    ),
                    "physical_gate_manifest": (
                        args.physical_gate_manifest,
                        args.physical_gate_manifest_sha256,
                    ),
                    "physical_gate_materialization_receipt": (
                        args.physical_gate_materialization_receipt,
                        args.physical_gate_materialization_receipt_sha256,
                    ),
                    "profile_pins": (
                        args.profile_pins,
                        args.profile_pins_sha256,
                    ),
                    "launch_evidence_trust_root": (
                        Path(args.launch_trust_root),
                        args.launch_trust_root_sha256,
                    ),
                },
                require_all_modules_loaded=True,
            )
        )
        receipt["runtime_input_snapshot"]["post_runtime"] = (
            assert_runtime_input_snapshot_stable(runtime_input_snapshot)
        )
        receipt["runtime_input_snapshot"]["checkout_post_runtime"] = (
            validate_clean_checkout(args.code_commit)
        )
        receipt.update(
            {
                "status": "PASS" if overall else "FAIL",
                "verdict": "PASS" if overall else "FAIL",
                "manifest_id": manifest.base.manifest_id,
                "action_order": list(manifest.base.action_order),
                "base_mujoco_portable_identity_sha256": (
                    verified.portable_identity_sha256
                ),
                "base_mujoco_verification_receipt_sha256": (
                    verified.verification_receipt_sha256
                ),
                "compiler_mesh_assets": compiler_asset_receipts,
                "scene_contracts": scenes,
                "venue": {
                    "path": str(venue.path),
                    "sha256": venue.sha256,
                },
                "contact_model": {
                    "path": str(CONTACT_MODEL_PATH),
                    "sha256": CONTACT_MODEL_SHA256,
                },
                "teacher_return_safety_rows": (
                    teacher_return_safety_rows
                ),
                "actions": actions_out,
            }
        )
        _seal_receipt(receipt)
        output_identity = _write_reserved_receipt(
            args.out, receipt_descriptor, receipt
        )
        receipt_descriptor = -1
        print(
            f"[fitted-ball-gate] {receipt['status']} "
            f"actions={len(actions_out)} receipt={args.out} "
            f"receipt_file_sha256={output_identity['sha256']}"
        )
        return 0 if overall else 3
    except Exception as exc:
        receipt.update(
            {
                "status": "INFRASTRUCTURE_FAIL",
                "verdict": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _seal_receipt(receipt)
        if receipt_descriptor >= 0:
            try:
                _write_reserved_receipt(
                    args.out, receipt_descriptor, receipt
                )
            except Exception as receipt_exc:
                try:
                    os.close(receipt_descriptor)
                except OSError:
                    pass
                print(
                    "[fitted-ball-gate][FATAL] receipt write also failed: "
                    f"{type(receipt_exc).__name__}: {receipt_exc}",
                    file=sys.stderr,
                )
            receipt_descriptor = -1
        print(
            f"[fitted-ball-gate][FATAL] {exc}; receipt={args.out}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
