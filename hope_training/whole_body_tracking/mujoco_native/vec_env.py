"""Diagnostic sequential VecEnv adapter for the native MuJoCo N1 core.

The adapter deliberately stops before PPO.  It implements deterministic
batched reset, purpose-group observation flattening and finite physics rollout
for N independent ``MujocoN1BallCore`` instances.  Its rsl_rl-shaped ``step``
method raises before physics because the current core has no complete, bound
ActionBall reward/termination contract.  Returning zero or an improvised
distance reward would make an optimizer update look valid when it is not.

Use :meth:`diagnostic_step` for no-reward plumbing tests.  A future reward
port must close every item in :data:`REWARD_BLOCKERS` before enabling
``step`` or any PPO/checkpoint smoke.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import n1_ball_core
from . import n1_reward_event_kernel
from . import single_env
from . import table_termination


OBSERVATION_LAYOUT = (
    ("robot_joint_pos", 31),
    ("robot_joint_vel", 31),
    ("incoming_ball_position_w_m", 3),
    ("incoming_ball_linear_velocity_w_mps", 3),
    ("incoming_ball_spin_w_radps", 3),
    ("landing_aim_xy_w_m", 2),
    ("time_to_contact_s", 1),
    ("validity", 2),
)
OBSERVATION_WIDTH = sum(width for _name, width in OBSERVATION_LAYOUT)

REWARD_BLOCKERS = (
    "full_phase_nonwrist_teacher_and_measured_paddle_reference_not_exposed",
    "actual_official_racket_site_velocity_signed_face_long_axis_not_exposed",
    "desired_at_contact_target_and_window_eligibility_not_installed",
    "native_contact_material_aero_magnus_and_outcome_parity_not_authorized",
    "legal_net_landing_spin_event_ledger_not_complete",
    "three_layer_reward_weights_and_source_sha_not_bound",
    "termination_reset_and_reward_income_receipt_not_bound",
)

FORMAL_TERMINATION_BLOCKERS = (
    "native_core_phase_fidelity_reference_tape_not_installed",
)

# Exact subset copied from ``HOPEDeployParityTerminationsCfg``.  MuJoCo's
# pelvis world-up dot product is the same scalar as Isaac Lab's
# ``-projected_gravity_b[..., 2]`` used by ``bad_orientation``.
EXACT_BASE_TERMINATION_REASON_ORDER = (
    "base_fell_tilt",
    "base_too_low",
    "joint_qdes_forbidden",
    "joint_actual_forbidden",
)
EXACT_PHASE_FIDELITY_REASON_ORDER = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)
EXACT_HARD_TERMINATION_REASON_ORDER = (
    *EXACT_PHASE_FIDELITY_REASON_ORDER,
    "base_fell_tilt",
    "base_too_low",
    "robot_hit_table",
    "joint_qdes_forbidden",
    "joint_actual_forbidden",
)
EXACT_ACTIVE_TERMINATION_REASON_ORDER = (
    "time_out",
    *EXACT_HARD_TERMINATION_REASON_ORDER,
)
BASE_FELL_TILT_LIMIT_ANGLE_RAD = 0.7
BASE_FELL_TILT_MIN_UP_WORLD_Z = math.cos(BASE_FELL_TILT_LIMIT_ANGLE_RAD)
BASE_TOO_LOW_MINIMUM_HEIGHT_M = 0.5
TERMINATION_SOURCE_CONFIG = (
    table_termination.ISAAC_TERMINATION_CONFIG
)
TERMINATION_SOURCE_CALLABLES = (
    table_termination.ISAAC_TERMINATION_CALLABLES
)
TERMINATION_SOURCE_ACTION_LATCH = table_termination.ISAAC_ACTION_LATCH
TERMINATION_SOURCE_PHASE_WRAPPERS = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_rewards.py"
)
TERMINATION_SOURCE_PHASE_GATE = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)
TERMINATION_SOURCE_BASE_CONFIG = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
)
TERMINATION_SOURCE_A3_BODY_NAMES = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256 = (
    "0d70e9ac8e79bfd6e5f3ebba1321e7cd5463eb278b12290229fbd1d51baf37c2"
)
EXPECTED_PHASE_BASE_CONFIG_SEMANTIC_AST_SHA256 = (
    "aefdf83d0dbd39144da07cb4c7bcb2eee59c552174e00b8d28747cdde992e49c"
)
EXPECTED_PHASE_RAW_CALLABLES_SEMANTIC_AST_SHA256 = (
    "ca2bba2fb604d5624ccc7482228a11dd3a15f36d2e08a16a70cdfa453abdf8c4"
)
EXPECTED_PHASE_WRAPPERS_SEMANTIC_AST_SHA256 = (
    "cda50dc553aecc5657470f726e21c4d3ed572a829f2eec0ba951166286646420"
)
EXPECTED_PHASE_GATE_SEMANTIC_AST_SHA256 = (
    "9dba1fdf38fd2cd4bf4b105a2b0c288a25c5b0f1216dc6765d8f90e3d3237528"
)
EXPECTED_PHASE_BODY_NAMES_SEMANTIC_AST_SHA256 = (
    "3186d6d715304e7880e1cc576db1d2f27a6fa6d2dfb123741d84ca6e0e4afbc8"
)
JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD = single_env.JOINT_BOUNDS_TOLERANCE_RAD
JOINT_QDES_FORBIDDEN_MARGIN_RAD = 0.0
JOINT_QDES_FORBIDDEN_MARGIN_FRACTION = 0.02
JOINT_QDES_FINITE_PROJECTION_ENABLED = True
PHASE_ANCHOR_POS_Z_THRESHOLD_M = 0.25
PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD = 0.8
PHASE_EE_BODY_POS_Z_THRESHOLD_M = 0.25
PHASE_EE_BODY_NAMES = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
)
PHASE_CONTEXTS = (
    "non_hold_swing_or_follow_through",
    "recovery_hold",
)

CONTACT_EVENT_LABELS = ("racket", "table", "net", "floor")
PLANT_COUNTER_KEYS = (
    "qdes_clamp_joint_events",
    "effort_clip_joint_events",
    "velocity_limit_joint_events",
    "table_contact_pairs",
    "self_contact_pairs",
    "table_contact_substeps",
    "self_contact_substeps",
)
PLANT_MAX_KEYS = (
    "max_table_penetration_m",
    "max_self_penetration_m",
    "max_joint_velocity_ratio",
)


class VecEnvContractError(RuntimeError):
    """The diagnostic vector environment contract is invalid."""


class RewardContractMissing(VecEnvContractError):
    """PPO was requested before a real ActionBall reward was installed."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise VecEnvContractError("torch is required for the rsl_rl VecEnv adapter") from exc
    return torch


def _sha256_json(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _plain_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VecEnvContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _portable_ast_dump(node: ast.AST) -> str:
    """Serialize selected source semantics identically on Python 3.10+."""

    def normalize(value: Any) -> Any:
        if isinstance(value, ast.AST):
            fields = []
            for field, child in ast.iter_fields(value):
                if field == "type_params" and child == []:
                    continue
                fields.append([field, normalize(child)])
            return [type(value).__name__, fields]
        if isinstance(value, list):
            return [normalize(child) for child in value]
        if value is Ellipsis:
            return ["__constant__", "ellipsis"]
        if isinstance(value, bytes):
            return ["__constant_bytes_hex__", value.hex()]
        if isinstance(value, complex):
            return ["__constant_complex__", value.real, value.imag]
        return value

    return json.dumps(
        normalize(node),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _semantic_ast_sha256(
    path: Path, selectors: Sequence[tuple[str, str]]
) -> str:
    """Hash only selected source semantics, independent of unrelated file WIP."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise VecEnvContractError(
            f"cannot parse phase-fidelity authority source {path}"
        ) from exc

    def assignment_names(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            return ()
        return tuple(
            target.id for target in targets if isinstance(target, ast.Name)
        )

    def class_header(node: ast.ClassDef) -> dict[str, Any]:
        return {
            "decorators": [
                _portable_ast_dump(item)
                for item in node.decorator_list
            ],
            "bases": [
                _portable_ast_dump(item) for item in node.bases
            ],
            "keywords": [
                _portable_ast_dump(item) for item in node.keywords
            ],
        }

    selected = []
    nodes = tuple(ast.walk(tree))
    for kind, name in selectors:
        if kind == "class":
            matches = [
                node
                for node in nodes
                if isinstance(node, ast.ClassDef) and node.name == name
            ]
        elif kind == "function":
            matches = [
                node
                for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ]
        elif kind == "assignment":
            matches = [node for node in nodes if name in assignment_names(node)]
        elif kind == "class_header":
            classes = [
                node
                for node in nodes
                if isinstance(node, ast.ClassDef) and node.name == name
            ]
            matches = [] if len(classes) != 1 else [class_header(classes[0])]
        elif kind == "class_assignments":
            try:
                class_name, raw_names = name.split("|", 1)
            except ValueError as exc:
                raise VecEnvContractError(
                    "phase-fidelity class-assignment selector is malformed"
                ) from exc
            required_names = tuple(raw_names.split(","))
            if not required_names or any(not item for item in required_names):
                raise VecEnvContractError(
                    "phase-fidelity class-assignment selector is malformed"
                )
            classes = [
                node
                for node in nodes
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            matches = []
            if len(classes) == 1:
                assignments = [
                    node
                    for node in classes[0].body
                    if set(assignment_names(node)) & set(required_names)
                ]
                observed_names = tuple(
                    item
                    for node in assignments
                    for item in assignment_names(node)
                    if item in required_names
                )
                if (
                    len(observed_names) == len(required_names)
                    and set(observed_names) == set(required_names)
                ):
                    matches = [
                        {
                            "class_header": class_header(classes[0]),
                            "assignments_in_source_order": [
                                {
                                    "names": assignment_names(node),
                                    "ast": _portable_ast_dump(node),
                                }
                                for node in assignments
                            ],
                        }
                    ]
        elif kind == "function_if_assignment":
            try:
                function_name, attribute_name = name.split("|", 1)
            except ValueError as exc:
                raise VecEnvContractError(
                    "phase-fidelity function-if selector is malformed"
                ) from exc
            functions = [
                node
                for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ]
            if len(functions) != 1:
                matches = []
            else:
                matches = []
                for candidate in ast.walk(functions[0]):
                    if not isinstance(candidate, ast.If):
                        continue
                    target_attributes = []
                    for statement in ast.walk(candidate):
                        if isinstance(statement, ast.Assign):
                            targets = statement.targets
                        elif isinstance(statement, ast.AnnAssign):
                            targets = (statement.target,)
                        else:
                            continue
                        target_attributes.extend(
                            nested.attr
                            for target in targets
                            for nested in ast.walk(target)
                            if isinstance(nested, ast.Attribute)
                        )
                    if attribute_name in target_attributes:
                        matches.append(candidate)
        else:
            raise VecEnvContractError(
                f"unsupported phase-fidelity AST selector kind {kind}"
            )
        if len(matches) != 1:
            raise VecEnvContractError(
                f"phase-fidelity authority selector {kind}:{name} is not unique"
            )
        selected.append(
            {
                "kind": kind,
                "name": name,
                "ast": (
                    _portable_ast_dump(matches[0])
                    if isinstance(matches[0], ast.AST)
                    else matches[0]
                ),
            }
        )
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def _phase_fidelity_sample_contract_cached() -> dict[str, Any]:
    source_specs = (
        (
            "action_ball_config",
            TERMINATION_SOURCE_CONFIG,
            (
                ("class_header", "HOPEDeployParityTerminationsCfg"),
                (
                    "class_assignments",
                    "HOPEDeployParityTerminationsCfg|anchor_pos,anchor_ori,"
                    "ee_body_pos,base_fell_tilt,base_too_low,robot_hit_table",
                ),
                ("class_header", "HOPEActionBallTerminationsCfg"),
                (
                    "class_assignments",
                    "HOPEActionBallTerminationsCfg|joint_qdes_forbidden,"
                    "joint_actual_forbidden",
                ),
            ),
            EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256,
        ),
        (
            "raw_callables",
            TERMINATION_SOURCE_CALLABLES,
            (
                ("function", "bad_anchor_pos_z_only"),
                ("function", "bad_anchor_ori"),
                ("function", "bad_motion_body_pos_z_only"),
                ("function", "pre_clamp_qdes_forbidden_zone"),
                ("function", "actual_joint_position_forbidden_zone"),
            ),
            EXPECTED_PHASE_RAW_CALLABLES_SEMANTIC_AST_SHA256,
        ),
        (
            "hold_aware_wrappers",
            TERMINATION_SOURCE_PHASE_WRAPPERS,
            (
                ("function", "_ignore_hold"),
                ("function", "_action_ball_reference_terminations_mask"),
                ("function", "_gate_reference_termination"),
                ("function", "bad_anchor_pos_z_only_hold_aware"),
                ("function", "bad_anchor_ori_hold_aware"),
                ("function", "bad_motion_body_pos_z_only_hold_aware"),
            ),
            EXPECTED_PHASE_WRAPPERS_SEMANTIC_AST_SHA256,
        ),
        (
            "frozen_phase_gate",
            TERMINATION_SOURCE_PHASE_GATE,
            (
                ("assignment", "_REFERENCE_GUARD_PHASE_GATED"),
                ("assignment", "_REFERENCE_GUARD_METRICS_ONLY"),
                ("assignment", "_REFERENCE_GUARD_MODES"),
                ("function", "_reference_guard_mode"),
                ("function", "_action_ball_phase_center_mask_tensor"),
                ("function", "action_ball_reference_terminations_enabled"),
                (
                    "function_if_assignment",
                    "_sample_targets_action_ball|_action_ball_reference_term_center_latch",
                ),
            ),
            EXPECTED_PHASE_GATE_SEMANTIC_AST_SHA256,
        ),
        (
            "base_config",
            TERMINATION_SOURCE_BASE_CONFIG,
            (
                ("class_header", "TerminationsCfg"),
                ("class_assignments", "TerminationsCfg|time_out"),
            ),
            EXPECTED_PHASE_BASE_CONFIG_SEMANTIC_AST_SHA256,
        ),
        (
            "a3_body_names",
            TERMINATION_SOURCE_A3_BODY_NAMES,
            (
                ("assignment", "A3_FEET_BODIES"),
                ("assignment", "A3_HAND_BODIES"),
            ),
            EXPECTED_PHASE_BODY_NAMES_SEMANTIC_AST_SHA256,
        ),
    )
    sources: dict[str, dict[str, str]] = {}
    for label, path, selectors, expected_sha256 in source_specs:
        actual_sha256 = _semantic_ast_sha256(path, selectors)
        if actual_sha256 != expected_sha256:
            raise VecEnvContractError(
                f"exact phase-fidelity authority source {label} semantic AST "
                "SHA-256 drifted"
            )
        sources[label] = {
            "semantic_ast_sha256": actual_sha256,
        }
    payload = {
        "schema_version": 1,
        "kind": "a3_mujoco_phase_fidelity_sample_contract_v1",
        "sample_keys": [
            "schema_version",
            "kind",
            "motion_phase_context",
            "in_hold",
            "reference_terminations_enabled",
            "anchor_pos_z_error_m",
            "anchor_projected_gravity_z_error_abs",
            "ee_body_pos_z_error_m",
        ],
        "motion_phase_contexts": list(PHASE_CONTEXTS),
        "ee_body_order": list(PHASE_EE_BODY_NAMES),
        "thresholds": {
            "anchor_pos_z_error_m": PHASE_ANCHOR_POS_Z_THRESHOLD_M,
            "anchor_projected_gravity_z_error_abs": (
                PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD
            ),
            "ee_body_pos_z_error_m": PHASE_EE_BODY_POS_Z_THRESHOLD_M,
        },
        "comparison": "strict_greater_than",
        "gating": (
            "verdict AND NOT in_hold AND reference_terminations_enabled"
        ),
        "reason_order": list(EXACT_PHASE_FIDELITY_REASON_ORDER),
        "authority_sources": sources,
    }
    payload["content_sha256"] = _sha256_json(payload)
    return payload


def phase_fidelity_sample_contract() -> dict[str, Any]:
    """Caller-owned strict ABI for externally computed MotionCommand errors."""

    return copy.deepcopy(_phase_fidelity_sample_contract_cached())


def exact_phase_fidelity_reasons(sample: Mapping[str, Any]) -> tuple[str, ...]:
    """Port the three Isaac reference-envelope verdicts over one strict sample."""

    contract = _phase_fidelity_sample_contract_cached()
    expected_keys = set(contract["sample_keys"])
    if not isinstance(sample, Mapping) or set(sample) != expected_keys:
        raise VecEnvContractError("phase-fidelity sample keys differ from exact ABI")
    if (
        sample.get("schema_version") != 1
        or sample.get("kind") != "a3_mujoco_phase_fidelity_sample_v1"
    ):
        raise VecEnvContractError("phase-fidelity sample kind/schema differs")
    phase_context = sample.get("motion_phase_context")
    in_hold = sample.get("in_hold")
    enabled = sample.get("reference_terminations_enabled")
    if phase_context not in PHASE_CONTEXTS:
        raise VecEnvContractError("phase-fidelity motion phase context is unsupported")
    if type(in_hold) is not bool or type(enabled) is not bool:
        raise VecEnvContractError("phase-fidelity gates must be plain booleans")
    if in_hold != (phase_context == "recovery_hold"):
        raise VecEnvContractError("phase-fidelity hold gate disagrees with phase context")

    def finite_nonnegative(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise VecEnvContractError(f"phase-fidelity {name} must be finite and >=0")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise VecEnvContractError(
                f"phase-fidelity {name} must be finite and >=0"
            ) from exc
        if not math.isfinite(result) or result < 0.0:
            raise VecEnvContractError(
                f"phase-fidelity {name} must be finite and >=0"
            )
        return result

    anchor_pos_error = finite_nonnegative(
        sample.get("anchor_pos_z_error_m"), "anchor_pos_z_error_m"
    )
    anchor_ori_error = finite_nonnegative(
        sample.get("anchor_projected_gravity_z_error_abs"),
        "anchor_projected_gravity_z_error_abs",
    )
    try:
        ee_errors = np.asarray(sample.get("ee_body_pos_z_error_m"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VecEnvContractError(
            "phase-fidelity ee body errors must be four finite non-negative values"
        ) from exc
    if (
        ee_errors.shape != (len(PHASE_EE_BODY_NAMES),)
        or not np.isfinite(ee_errors).all()
        or np.any(ee_errors < 0.0)
    ):
        raise VecEnvContractError(
            "phase-fidelity ee body errors must be four finite non-negative values"
        )
    if in_hold or not enabled:
        return ()
    reasons: list[str] = []
    if anchor_pos_error > PHASE_ANCHOR_POS_Z_THRESHOLD_M:
        reasons.append("anchor_pos")
    if anchor_ori_error > PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD:
        reasons.append("anchor_ori")
    if bool(np.any(ee_errors > PHASE_EE_BODY_POS_Z_THRESHOLD_M)):
        reasons.append("ee_body_pos")
    return tuple(reasons)


def _canonical_phase_fidelity_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and convert the strict sample to JSON-stable primitive values."""

    exact_phase_fidelity_reasons(sample)
    return {
        "schema_version": 1,
        "kind": "a3_mujoco_phase_fidelity_sample_v1",
        "motion_phase_context": str(sample["motion_phase_context"]),
        "in_hold": bool(sample["in_hold"]),
        "reference_terminations_enabled": bool(
            sample["reference_terminations_enabled"]
        ),
        "anchor_pos_z_error_m": float(sample["anchor_pos_z_error_m"]),
        "anchor_projected_gravity_z_error_abs": float(
            sample["anchor_projected_gravity_z_error_abs"]
        ),
        "ee_body_pos_z_error_m": [
            float(value) for value in sample["ee_body_pos_z_error_m"]
        ],
    }


def _canonical_native_physical_event_facts(
    sample: Mapping[str, Any],
    *,
    expected_source: n1_reward_event_kernel.SourceBinding,
) -> dict[str, Any]:
    try:
        return n1_reward_event_kernel.validate_native_physical_event_facts(
            sample, expected_source=expected_source
        )
    except n1_reward_event_kernel.N1RewardEventKernelError as exc:
        raise VecEnvContractError(
            "native physical reward-event facts violate their exact ABI"
        ) from exc


def flatten_observation_groups(groups: Mapping[str, Any]) -> np.ndarray:
    """Flatten the provisional purpose groups without silently adding columns."""

    if set(groups) != {name for name, _width in OBSERVATION_LAYOUT}:
        raise VecEnvContractError("observation groups differ from diagnostic layout")
    rows = []
    for name, width in OBSERVATION_LAYOUT:
        value = np.asarray(groups[name], dtype=np.float64)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise VecEnvContractError(
                f"observation group {name!r} must be {width} finite scalars"
            )
        rows.append(value)
    flat = np.concatenate(rows)
    if flat.shape != (OBSERVATION_WIDTH,):
        raise VecEnvContractError("flattened observation width drifted")
    return flat


def reward_blocker_receipt() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "a3_mujoco_n1_vecenv_ppo_reward_blocker_v1",
        "status": "PPO_BLOCKED_MISSING_REAL_REWARD_CONTRACT",
        "reward_available": False,
        "zero_reward_allowed": False,
        "improvised_proxy_reward_allowed": False,
        "blockers": list(REWARD_BLOCKERS),
        "allowed_scope": [
            "deterministic_vecenv_reset",
            "purpose_group_observation_flattening",
            "finite_no_reward_physics_rollout",
            "rsl_rl_interface_shape_preflight",
            "validated_substep_contact_edge_transcript",
            "diagnostic_event_ledger",
            "exact_tape_time_out_latch",
            "exact_fall_height_joint_qdes_joint_actual_and_robot_table_termination_subset",
        ],
        "prohibited_scope": [
            "ppo_rollout",
            "optimizer_update",
            "training_checkpoint",
            "cold_load_resume",
            "learnability_claim",
        ],
        "enforcement_scope": {
            "vecenv_step_raises_before_physics": True,
            "assert_ppo_ready_always_raises": True,
            "upstream_runner_save_load_intercepted": False,
            "required_integration_rule": (
                "a future controlled runner/factory must call assert_ppo_ready before "
                "learn/save/load; do not invoke upstream checkpoint APIs directly"
            ),
        },
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
    }
    payload["content_sha256"] = _sha256_json(payload)
    return payload


@lru_cache(maxsize=1)
def _termination_blocker_receipt_cached() -> dict[str, Any]:
    """Validate the pinned Isaac config once and cache the immutable template."""
    phase_contract = _phase_fidelity_sample_contract_cached()
    phase_sources = phase_contract["authority_sources"]
    try:
        table_sources = table_termination.verify_isaac_source_authority()
    except table_termination.TableTerminationContractError as exc:
        raise VecEnvContractError(str(exc)) from exc
    source_config_semantic_sha256 = phase_sources["action_ball_config"][
        "semantic_ast_sha256"
    ]
    source_callables_semantic_sha256 = phase_sources["raw_callables"][
        "semantic_ast_sha256"
    ]
    source_action_latch_semantic_sha256 = table_sources[
        "action_latch_semantic_ast_sha256"
    ]
    payload = {
        "schema_version": 8,
        "kind": "a3_mujoco_n1_vecenv_termination_blocker_v8",
        "status": "FORMAL_TERMINATION_BLOCKED",
        "formal_termination_available": False,
        "terminated_tensor_available": False,
        "exact_base_subset_available": True,
        "exact_base_subset_terminated_tensor_available": True,
        "exact_robot_table_termination_available": True,
        "exact_hard_subset_terminated_tensor_available": True,
        "exact_time_out_latch_available": True,
        "exact_episode_done_tensor_available": True,
        "exact_phase_fidelity_predicate_available": True,
        "production_core_phase_reference_tape_contract_available": True,
        "exact_phase_fidelity_runtime_sample_available": False,
        "exact_active_reason_order": list(EXACT_ACTIVE_TERMINATION_REASON_ORDER),
        "exact_hard_reason_order": list(EXACT_HARD_TERMINATION_REASON_ORDER),
        "exact_phase_fidelity_subset": {
            "reason_order": list(EXACT_PHASE_FIDELITY_REASON_ORDER),
            "sample_contract": copy.deepcopy(phase_contract),
            "source_config_path": str(TERMINATION_SOURCE_CONFIG),
            "source_config_semantic_ast_sha256": (
                source_config_semantic_sha256
            ),
            "source_base_config_path": str(TERMINATION_SOURCE_BASE_CONFIG),
            "source_base_config_semantic_ast_sha256": phase_sources[
                "base_config"
            ]["semantic_ast_sha256"],
            "source_callables_path": str(TERMINATION_SOURCE_CALLABLES),
            "source_callables_semantic_ast_sha256": (
                source_callables_semantic_sha256
            ),
            "source_wrappers_path": str(TERMINATION_SOURCE_PHASE_WRAPPERS),
            "source_wrappers_semantic_ast_sha256": phase_sources[
                "hold_aware_wrappers"
            ]["semantic_ast_sha256"],
            "source_gate_path": str(TERMINATION_SOURCE_PHASE_GATE),
            "source_gate_semantic_ast_sha256": phase_sources[
                "frozen_phase_gate"
            ]["semantic_ast_sha256"],
            "source_body_names_path": str(TERMINATION_SOURCE_A3_BODY_NAMES),
            "source_body_names_semantic_ast_sha256": phase_sources[
                "a3_body_names"
            ]["semantic_ast_sha256"],
            "sample_timing": "post_control_step",
            "anchor_pos": {
                "predicate": "abs(reference_anchor_z-robot_anchor_z)>0.25",
                "threshold_m": PHASE_ANCHOR_POS_Z_THRESHOLD_M,
            },
            "anchor_ori": {
                "predicate": (
                    "abs(reference_projected_gravity_z-"
                    "robot_projected_gravity_z)>0.8"
                ),
                "threshold": PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD,
            },
            "ee_body_pos": {
                "predicate": "any(abs(reference_body_z-robot_body_z)>0.25)",
                "threshold_m": PHASE_EE_BODY_POS_Z_THRESHOLD_M,
                "body_order": list(PHASE_EE_BODY_NAMES),
            },
            "recovery_hold_semantics": "all three verdicts masked while in_hold=true",
            "action_ball_gate_semantics": (
                "verdict additionally masked by the episode-frozen reference-"
                "terminations-enabled bit"
            ),
            "runtime_enablement": (
                "each production core must install one external Isaac MotionCommand "
                "phase reference tape; all cores then advertise and return the exact "
                "sample contract"
            ),
        },
        "exact_base_subset": {
            "reason_order": list(EXACT_BASE_TERMINATION_REASON_ORDER),
            "source_config_path": str(TERMINATION_SOURCE_CONFIG),
            "source_config_semantic_ast_sha256": (
                source_config_semantic_sha256
            ),
            "source_callables_path": str(TERMINATION_SOURCE_CALLABLES),
            "source_callables_semantic_ast_sha256": (
                source_callables_semantic_sha256
            ),
            "reason_order_scope": (
                "priority inside the installed base subset; phase reasons precede it"
            ),
            "base_fell_tilt": {
                "source_callable": "isaaclab.envs.mdp.bad_orientation",
                "source_config": (
                    "HOPEDeployParityTerminationsCfg.base_fell_tilt"
                ),
                "limit_angle_rad": BASE_FELL_TILT_LIMIT_ANGLE_RAD,
                "mujoco_predicate": (
                    "pelvis_up_world_z < cos(limit_angle_rad)"
                ),
                "sample_timing": "post_control_step",
            },
            "base_too_low": {
                "source_callable": (
                    "isaaclab.envs.mdp.root_height_below_minimum"
                ),
                "source_config": (
                    "HOPEDeployParityTerminationsCfg.base_too_low"
                ),
                "minimum_height_m": BASE_TOO_LOW_MINIMUM_HEIGHT_M,
                "mujoco_predicate": (
                    "pelvis_link_origin_height_w_m < minimum_height_m"
                ),
                "sample_timing": "post_control_step",
            },
            "joint_qdes_forbidden": {
                "source_callable": "pre_clamp_qdes_forbidden_zone",
                "source_config": "HOPEActionBallTerminationsCfg.joint_qdes_forbidden",
                "limit_source": "joint_pos_limits",
                "margin_rad": JOINT_QDES_FORBIDDEN_MARGIN_RAD,
                "margin_fraction": JOINT_QDES_FORBIDDEN_MARGIN_FRACTION,
                "finite_preclamp_qdes_projection_enabled": (
                    JOINT_QDES_FINITE_PROJECTION_ENABLED
                ),
                "mujoco_predicate": "any(nonfinite(qdes_raw))",
                "finite_request_semantics": (
                    "project and retain transition; the projection penalty owns the event"
                ),
                "sample_timing": "post_control_step",
            },
            "joint_actual_forbidden": {
                "source_callable": "actual_joint_position_forbidden_zone",
                "source_config": "HOPEActionBallTerminationsCfg.joint_actual_forbidden",
                "limit_source": "MuJoCo model.jnt_range in runtime joint order",
                "bounds_tolerance_rad": JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD,
                "mujoco_predicate": (
                    "any(nonfinite(q/lower/upper) or upper<=lower or "
                    "q<=lower+tolerance or q>=upper-tolerance)"
                ),
                "sample_timing": "post_control_step",
            },
        },
        "exact_robot_table": {
            "reason": "robot_hit_table",
            "source_callable": "robot_hit_table",
            "source_config": "HOPEActionBallTerminationsCfg.robot_hit_table",
            "source_config_semantic_ast_sha256": table_sources[
                "config_semantic_ast_sha256"
            ],
            "source_callable_semantic_ast_sha256": table_sources[
                "callables_semantic_ast_sha256"
            ],
            "predicate": (
                "43 pinned component OBBs plus live racket OBB conservatively "
                "broadened to world AABBs against five inflated table AABBs"
            ),
            "sample_timing": "after_each_physics_substep",
            "sticky_within_control_step": True,
            "required_control_decimation": 4,
            "source_action_latch_path": str(TERMINATION_SOURCE_ACTION_LATCH),
            "source_action_latch_semantic_ast_sha256": (
                source_action_latch_semantic_sha256
            ),
            "substep_latch_scope": "current_control_step_plant_sample",
            "episode_sticky_owner": "DiagnosticEventLedger",
            "diagnostic_step_after_latch_requires_explicit_reset": False,
            "immediate_compact_reset_implemented": True,
            "collision_proxy_path": str(
                table_termination.COLLISION_PROXY_ARTIFACT
            ),
            "collision_proxy_sha256": (
                table_termination.EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256
            ),
            "table_geometry_sha256": (
                table_termination.EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256
            ),
            "table_aabb_margin_m": table_termination.TABLE_GUARD_MARGIN_M,
            "required_root_mjcf_path": str(table_termination.CANONICAL_MJCF),
            "required_root_mjcf_sha256": (
                table_termination.EXPECTED_CANONICAL_MJCF_SHA256
            ),
            "required_portable_mujoco_identity_sha256": (
                table_termination.EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256
            ),
            "required_owner_local_frame_binding": True,
            "resolved_contact_required": False,
        },
        "per_env_compact_reset": {
            "available": True,
            "reset_mask": "episode_dones=exact_hard_terminations OR time_outs",
            "nonterminated_rows_advance_without_reset": True,
            "returned_observations": "post_compact_reset_next_observations",
            "terminal_observations": (
                "pre_reset_observations_valid_only_where_terminal_observation_mask"
            ),
            "terminal_ledgers": "per_env_ledgers_before_compact_reset",
            "episode_local_length_hard_latch_and_ledger_cleared_on_reset": True,
            "reset_failure_invalidates_vecenv_until_full_reset": True,
        },
        "exact_diagnostic_facts": [
            "ball_racket_table_net_floor_contact_edges_at_physics_substep",
            "robot_obstacle_and_self_contact_substep_counts",
            "qdes_clamp_effort_clip_and_joint_velocity_limit_counts",
            "pre_clamp_qdes_nonfinite_hard_termination_latch",
            "pelvis_height_and_world_up_z_samples",
        ],
        "blockers": list(FORMAL_TERMINATION_BLOCKERS),
        "semantic_boundary": (
            "the production core computes exact samples only from an explicitly "
            "installed external MotionCommand phase reference tape plus live MuJoCo "
            "pelvis/feet/hands state; no ball-clock phase inference is permitted"
        ),
        "reward_paid": False,
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
    }
    payload["content_sha256"] = _sha256_json(payload)
    return payload


@lru_cache(maxsize=2)
def _termination_contract_receipt_cached(
    phase_fidelity_runtime_available: bool,
) -> dict[str, Any]:
    payload = copy.deepcopy(_termination_blocker_receipt_cached())
    if phase_fidelity_runtime_available:
        payload["status"] = "FORMAL_TERMINATION_AVAILABLE_DIAGNOSTIC_ONLY"
        payload["formal_termination_available"] = True
        payload["terminated_tensor_available"] = True
        payload["exact_phase_fidelity_runtime_sample_available"] = True
        payload["blockers"] = [
            blocker
            for blocker in payload["blockers"]
            if blocker
            != "native_core_phase_fidelity_reference_tape_not_installed"
        ]
        payload["content_sha256"] = _sha256_json(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
    return payload


def termination_blocker_receipt(
    *, phase_fidelity_runtime_available: bool = False
) -> dict[str, Any]:
    """Return an instance-specific caller-owned termination contract receipt."""

    if type(phase_fidelity_runtime_available) is not bool:
        raise VecEnvContractError(
            "phase_fidelity_runtime_available must be a plain boolean"
        )
    return copy.deepcopy(
        _termination_contract_receipt_cached(phase_fidelity_runtime_available)
    )


def _nonnegative_plain_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise VecEnvContractError(f"{name} must be a non-negative plain integer")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise VecEnvContractError(f"{name} must be a non-negative finite scalar")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise VecEnvContractError(f"{name} must be a non-negative finite scalar")
    return out


@dataclass
class DiagnosticEventLedger:
    """Cumulative facts plus the exact installed hard-termination subset."""

    control_decimation: int
    phase_fidelity_runtime_available: bool = False
    policy_ticks: int = 0
    physics_substeps: int = 0
    time_out_latched: bool = False
    exact_hard_termination_latched: bool = False
    exact_hard_reason_counts: dict[str, int] = field(
        default_factory=lambda: {
            reason: 0 for reason in EXACT_HARD_TERMINATION_REASON_ORDER
        }
    )
    first_exact_hard_termination: dict[str, Any] | None = None
    contact_edge_counts: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in CONTACT_EVENT_LABELS}
    )
    first_contact_edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    plant_counters: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in PLANT_COUNTER_KEYS}
    )
    plant_maxima: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in PLANT_MAX_KEYS}
    )
    first_robot_obstacle_contact: dict[str, Any] | None = None
    first_robot_self_contact: dict[str, Any] | None = None
    last_event_time_s: float | None = None
    latest_pelvis_height_m: float | None = None
    latest_pelvis_up_world_z: float | None = None
    phase_fidelity_samples: int = 0

    def __post_init__(self) -> None:
        if type(self.control_decimation) is not int or self.control_decimation != 4:
            raise VecEnvContractError(
                "exact Isaac robot/table termination requires control_decimation=4"
            )
        if type(self.phase_fidelity_runtime_available) is not bool:
            raise VecEnvContractError(
                "phase_fidelity_runtime_available must be a plain boolean"
            )

    def record_step(
        self,
        *,
        plant: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        time_out: bool,
        phase_fidelity_sample: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate one complete control tick, then commit its cumulative facts."""

        if not isinstance(plant, Mapping):
            raise VecEnvContractError("diagnostic plant row must be a mapping")
        if type(time_out) is not bool:
            raise VecEnvContractError("diagnostic time_out must be bool")
        phase_fidelity_reasons = (
            ()
            if phase_fidelity_sample is None
            else exact_phase_fidelity_reasons(phase_fidelity_sample)
        )
        if self.phase_fidelity_runtime_available and phase_fidelity_sample is None:
            raise VecEnvContractError(
                "phase-fidelity runtime ledger requires one sample per control tick"
            )
        counters = {
            name: _nonnegative_plain_int(plant.get(name), f"plant.{name}")
            for name in PLANT_COUNTER_KEYS
        }
        for name in ("table_contact_substeps", "self_contact_substeps"):
            if counters[name] > self.control_decimation:
                raise VecEnvContractError(
                    f"plant.{name} exceeds one control tick's substep count"
                )
        maxima = {
            name: _finite_nonnegative(plant.get(name), f"plant.{name}")
            for name in PLANT_MAX_KEYS
        }
        raw_pelvis_height = plant.get("pelvis_height_m", math.nan)
        raw_pelvis_up_z = plant.get("pelvis_up_world_z", math.nan)
        if isinstance(raw_pelvis_height, bool) or isinstance(raw_pelvis_up_z, bool):
            raise VecEnvContractError("plant pelvis diagnostic samples must be finite")
        pelvis_height = float(raw_pelvis_height)
        pelvis_up_z = float(raw_pelvis_up_z)
        if not math.isfinite(pelvis_height) or not math.isfinite(pelvis_up_z):
            raise VecEnvContractError("plant pelvis diagnostic samples must be finite")
        if not -1.0 <= pelvis_up_z <= 1.0:
            raise VecEnvContractError(
                "plant pelvis_up_world_z must be a normalized world-up dot product"
            )

        joint_pos = np.asarray(plant.get("q"), dtype=np.float64)
        qdes_raw = np.asarray(plant.get("qdes_raw"), dtype=np.float64)
        joint_limits = np.asarray(
            plant.get("joint_position_limits"), dtype=np.float64
        )
        if joint_pos.shape != (single_env.ACTION_DIM,):
            raise VecEnvContractError("plant.q must contain exactly 31 joint positions")
        if qdes_raw.shape != (single_env.ACTION_DIM,):
            raise VecEnvContractError(
                "plant.qdes_raw must contain exactly 31 pre-clamp joint targets"
            )
        if joint_limits.shape != (single_env.ACTION_DIM, 2):
            raise VecEnvContractError(
                "plant.joint_position_limits must have shape (31, 2)"
            )
        comparable = (
            np.isfinite(joint_pos)
            & np.isfinite(joint_limits[:, 0])
            & np.isfinite(joint_limits[:, 1])
            & (joint_limits[:, 1] > joint_limits[:, 0])
        )
        joint_actual_forbidden = bool(
            np.any(
                ~comparable
                | (joint_pos <= joint_limits[:, 0] + JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD)
                | (joint_pos >= joint_limits[:, 1] - JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD)
            )
        )
        substep_actual = plant.get("joint_actual_forbidden_substep")
        if type(substep_actual) is not bool:
            raise VecEnvContractError(
                "plant.joint_actual_forbidden_substep must be bool"
            )
        joint_actual_forbidden = joint_actual_forbidden or substep_actual
        robot_hit_table = plant.get("robot_hit_table_substep")
        if type(robot_hit_table) is not bool:
            raise VecEnvContractError(
                "plant.robot_hit_table_substep must be bool"
            )
        first_table_guard_substep = plant.get("robot_hit_table_first_substep")
        if robot_hit_table:
            if (
                type(first_table_guard_substep) is not int
                or not 0 <= first_table_guard_substep < self.control_decimation
            ):
                raise VecEnvContractError(
                    "positive robot/table guard must carry its first substep"
                )
        elif first_table_guard_substep is not None:
            raise VecEnvContractError(
                "negative robot/table guard cannot carry a first substep"
            )
        # The pinned ActionBall action uses explicit finite pre-clamp projection.
        # Isaac's q_des Done term therefore owns only a valid non-finite affine
        # request; finite out-of-envelope requests are projected and retained so
        # their projection-distance penalty can teach recovery.
        joint_qdes_forbidden = bool(np.any(~np.isfinite(qdes_raw)))

        exact_hard_reasons = list(phase_fidelity_reasons)
        if pelvis_up_z < BASE_FELL_TILT_MIN_UP_WORLD_Z:
            exact_hard_reasons.append("base_fell_tilt")
        if pelvis_height < BASE_TOO_LOW_MINIMUM_HEIGHT_M:
            exact_hard_reasons.append("base_too_low")
        if robot_hit_table:
            exact_hard_reasons.append("robot_hit_table")
        if joint_qdes_forbidden:
            exact_hard_reasons.append("joint_qdes_forbidden")
        if joint_actual_forbidden:
            exact_hard_reasons.append("joint_actual_forbidden")

        normalized_events = []
        previous_order: tuple[int, int, str] | None = None
        seen_edges: set[tuple[int, int, str]] = set()
        last_time = self.last_event_time_s
        for raw in events:
            if not isinstance(raw, Mapping) or set(raw) != {
                "policy_tick",
                "physics_substep",
                "time_s",
                "event",
            }:
                raise VecEnvContractError("substep contact event keys differ from schema")
            policy_tick = _nonnegative_plain_int(
                raw["policy_tick"], "event.policy_tick"
            )
            substep = _nonnegative_plain_int(
                raw["physics_substep"], "event.physics_substep"
            )
            label = raw["event"]
            event_time = _finite_nonnegative(raw["time_s"], "event.time_s")
            if policy_tick != self.policy_ticks:
                raise VecEnvContractError(
                    "substep contact event policy tick differs from ledger"
                )
            if substep >= self.control_decimation:
                raise VecEnvContractError("substep contact event index is out of range")
            if label not in CONTACT_EVENT_LABELS:
                raise VecEnvContractError("substep contact event label is unsupported")
            order = (policy_tick, substep, str(label))
            if previous_order is not None and order <= previous_order:
                raise VecEnvContractError("substep contact events are not strictly ordered")
            if order in seen_edges:
                raise VecEnvContractError("duplicate substep contact edge")
            if last_time is not None and event_time < last_time:
                raise VecEnvContractError("substep contact event time regressed")
            event = {
                "policy_tick": policy_tick,
                "physics_substep": substep,
                "time_s": event_time,
                "event": str(label),
            }
            normalized_events.append(event)
            seen_edges.add(order)
            previous_order = order
            last_time = event_time

        # Commit only after the complete row validates.
        for name, value in counters.items():
            self.plant_counters[name] += value
        for name, value in maxima.items():
            self.plant_maxima[name] = max(self.plant_maxima[name], value)
        for event in normalized_events:
            label = event["event"]
            self.contact_edge_counts[label] += 1
            self.first_contact_edges.setdefault(label, dict(event))
        if normalized_events:
            self.last_event_time_s = normalized_events[-1]["time_s"]
        if (
            self.first_robot_obstacle_contact is None
            and counters["table_contact_substeps"] > 0
        ):
            self.first_robot_obstacle_contact = {
                "policy_tick": self.policy_ticks,
                "pair": plant.get("first_table_contact_pair"),
            }
        if (
            self.first_robot_self_contact is None
            and counters["self_contact_substeps"] > 0
        ):
            self.first_robot_self_contact = {
                "policy_tick": self.policy_ticks,
                "pair": plant.get("first_self_contact_pair"),
            }
        self.policy_ticks += 1
        self.physics_substeps += self.control_decimation
        self.phase_fidelity_samples += int(phase_fidelity_sample is not None)
        self.time_out_latched = self.time_out_latched or time_out
        for reason in exact_hard_reasons:
            self.exact_hard_reason_counts[reason] += 1
        if exact_hard_reasons and self.first_exact_hard_termination is None:
            sample_timing = (
                "physics_substep"
                if exact_hard_reasons[0] == "robot_hit_table"
                else "post_control_step"
            )
            self.first_exact_hard_termination = {
                "policy_tick": self.policy_ticks - 1,
                "sample_timing": sample_timing,
                "physics_substep": (
                    first_table_guard_substep if sample_timing == "physics_substep" else None
                ),
                "robot_hit_table_first_substep": first_table_guard_substep,
                "reason": exact_hard_reasons[0],
                "all_reasons": list(exact_hard_reasons),
            }
        self.exact_hard_termination_latched = (
            self.exact_hard_termination_latched
            or bool(exact_hard_reasons)
        )
        self.latest_pelvis_height_m = pelvis_height
        self.latest_pelvis_up_world_z = pelvis_up_z
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": 4,
            "kind": "a3_mujoco_n1_diagnostic_event_ledger_v4",
            "policy_ticks": self.policy_ticks,
            "physics_substeps": self.physics_substeps,
            "phase_fidelity": {
                "exact_sample_count": self.phase_fidelity_samples,
                "exact_runtime_sample_seen": self.phase_fidelity_samples > 0,
                "sample_contract_sha256": (
                    _phase_fidelity_sample_contract_cached()["content_sha256"]
                ),
            },
            "contact_edge_counts": dict(self.contact_edge_counts),
            "first_contact_edges": {
                label: dict(value)
                for label, value in sorted(self.first_contact_edges.items())
            },
            "plant_counters": dict(self.plant_counters),
            "plant_maxima": dict(self.plant_maxima),
            "exact_hard_reason_counts": dict(self.exact_hard_reason_counts),
            "first_exact_hard_termination": (
                None
                if self.first_exact_hard_termination is None
                else copy.deepcopy(self.first_exact_hard_termination)
            ),
            "first_robot_obstacle_contact": copy.deepcopy(
                self.first_robot_obstacle_contact
            ),
            "first_robot_self_contact": copy.deepcopy(
                self.first_robot_self_contact
            ),
            "latest_pelvis_samples": {
                "height_m": self.latest_pelvis_height_m,
                "up_world_z": self.latest_pelvis_up_world_z,
            },
            "latches": {
                **{
                    f"ball_{label}_contact_seen": self.contact_edge_counts[label] > 0
                    for label in CONTACT_EVENT_LABELS
                },
                "robot_obstacle_contact_seen": (
                    self.plant_counters["table_contact_substeps"] > 0
                ),
                "robot_table_keepout_seen": (
                    self.exact_hard_reason_counts["robot_hit_table"] > 0
                ),
                "robot_self_contact_seen": (
                    self.plant_counters["self_contact_substeps"] > 0
                ),
                "qdes_clamp_seen": self.plant_counters[
                    "qdes_clamp_joint_events"
                ]
                > 0,
                "effort_clip_seen": self.plant_counters["effort_clip_joint_events"]
                > 0,
                "joint_velocity_limit_seen": self.plant_counters[
                    "velocity_limit_joint_events"
                ]
                > 0,
            },
            "termination": {
                "exact_time_out_latched": self.time_out_latched,
                "exact_base_subset_available": True,
                "exact_robot_table_termination_available": True,
                "exact_hard_subset_available": True,
                "exact_phase_fidelity_predicate_available": True,
                "exact_phase_fidelity_runtime_sample_seen": (
                    self.phase_fidelity_samples > 0
                ),
                "exact_hard_terminated": (
                    self.exact_hard_termination_latched
                ),
                "exact_hard_reason": (
                    None
                    if self.first_exact_hard_termination is None
                    else self.first_exact_hard_termination["reason"]
                ),
                "formal_hard_termination_available": (
                    self.phase_fidelity_runtime_available
                ),
                "formal_hard_terminated": (
                    self.exact_hard_termination_latched
                    if self.phase_fidelity_runtime_available
                    else None
                ),
                "blocker_sha256": _termination_contract_receipt_cached(
                    self.phase_fidelity_runtime_available
                )[
                    "content_sha256"
                ],
            },
            "reward_paid": False,
            "diagnostic_unauthorized": True,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload


@dataclass(frozen=True)
class DiagnosticBatchStep:
    """One transition with explicit pre-reset terminal and post-reset next state.

    ``terminal_observations`` contains the complete pre-reset batch; only rows
    selected by ``terminal_observation_mask`` are terminal states.  The public
    ``observations`` are the post-compact-reset next observations.  Ledgers are
    the pre-reset episode snapshots for this transition.
    """

    observations: Any
    terminal_observations: Any
    terminal_observation_mask: Any
    episode_dones: Any
    episode_done_reasons: tuple[str | None, ...]
    reset_env_ids: tuple[int, ...]
    exact_phase_fidelity_runtime_available: bool
    per_env_phase_fidelity_samples: tuple[Mapping[str, Any] | None, ...]
    native_physical_event_runtime_available: bool
    per_env_native_physical_event_facts: tuple[
        Mapping[str, Any] | None, ...
    ]
    per_env_events: tuple[tuple[Mapping[str, Any], ...], ...]
    per_env_ledgers: tuple[Mapping[str, Any], ...]
    time_outs: Any
    exact_hard_terminations: Any
    exact_hard_termination_reasons: tuple[str | None, ...]


class MujocoN1DiagnosticVecEnv:
    """Sequential CPU batch with an rsl_rl-compatible read-only surface."""

    def __init__(
        self,
        *,
        cores: Sequence[n1_ball_core.MujocoN1BallCore],
        robot_tape: single_env.FixedTape,
        questions: Sequence[n1_ball_core.N1Question],
        device: str = "cpu",
    ) -> None:
        if not cores or len(cores) != len(questions):
            raise VecEnvContractError("cores/questions must have one non-empty row per env")
        if device != "cpu":
            raise VecEnvContractError("diagnostic native MuJoCo VecEnv is CPU-only")
        if any(core.binding.binding_sha256 != robot_tape.plant_binding_sha256 for core in cores):
            raise VecEnvContractError("one or more cores differ from robot tape plant binding")
        for core, question in zip(cores, questions):
            if core.scene_binding_sha256 != question.scene_binding_sha256:
                raise VecEnvContractError("one or more questions differ from core scene binding")
        scene_bindings = {core.scene_binding_sha256 for core in cores}
        if len(scene_bindings) != 1:
            raise VecEnvContractError("all vector rows must share one physical scene binding")

        self.cores = tuple(cores)
        self.robot_tape = robot_tape
        self.questions = tuple(questions)
        self.question_source_sha256_by_env = tuple(
            _plain_sha256(
                getattr(question, "source_sha256", None),
                f"questions[{index}].source_sha256",
            )
            for index, question in enumerate(self.questions)
        )
        self.num_envs = len(self.cores)
        self.num_actions = single_env.ACTION_DIM
        self.max_episode_length = int(robot_tape.actions.shape[0])
        if self.max_episode_length < 1:
            raise VecEnvContractError("robot tape must contain at least one action row")
        self.step_dt = float(self.cores[0].binding.policy_step_dt_s)
        decimations = {int(core.binding.control_decimation) for core in self.cores}
        if decimations != {4}:
            raise VecEnvContractError(
                "exact Isaac robot/table termination requires every core to use "
                "control_decimation=4"
            )
        self.control_decimation = next(iter(decimations))
        expected_phase_contract_sha256 = _phase_fidelity_sample_contract_cached()[
            "content_sha256"
        ]
        advertised_phase_contracts = tuple(
            getattr(core, "phase_fidelity_sample_contract_sha256", None)
            for core in self.cores
        )
        if all(value is None for value in advertised_phase_contracts):
            self.exact_phase_fidelity_runtime_available = False
        else:
            if any(value is None for value in advertised_phase_contracts):
                raise VecEnvContractError(
                    "phase-fidelity runtime ABI must be advertised by every core or none"
                )
            for index, value in enumerate(advertised_phase_contracts):
                if (
                    _plain_sha256(
                        value,
                        f"cores[{index}].phase_fidelity_sample_contract_sha256",
                    )
                    != expected_phase_contract_sha256
                ):
                    raise VecEnvContractError(
                        "one or more cores advertise a different phase-fidelity "
                        "sample contract"
                    )
            self.exact_phase_fidelity_runtime_available = True
        self.phase_fidelity_reference_tape_sha256_by_env = tuple(
            (
                None
                if getattr(core, "phase_fidelity_reference_tape", None) is None
                else _plain_sha256(
                    getattr(core.phase_fidelity_reference_tape, "source_sha256", None),
                    f"cores[{index}].phase_fidelity_reference_tape.source_sha256",
                )
            )
            for index, core in enumerate(self.cores)
        )
        expected_native_event_contract_sha256 = (
            n1_reward_event_kernel.native_physical_event_facts_contract()[
                "content_sha256"
            ]
        )
        advertised_native_event_contracts = tuple(
            getattr(core, "native_physical_event_contract_sha256", None)
            for core in self.cores
        )
        if all(value is None for value in advertised_native_event_contracts):
            self.native_physical_event_runtime_available = False
            self.native_physical_event_source_bindings = (None,) * self.num_envs
        else:
            if any(value is None for value in advertised_native_event_contracts):
                raise VecEnvContractError(
                    "native physical event ABI must be advertised by every core or none"
                )
            sources = []
            for index, (core, value) in enumerate(
                zip(self.cores, advertised_native_event_contracts)
            ):
                if (
                    _plain_sha256(
                        value,
                        f"cores[{index}].native_physical_event_contract_sha256",
                    )
                    != expected_native_event_contract_sha256
                ):
                    raise VecEnvContractError(
                        "one or more cores advertise a different native physical "
                        "event contract"
                    )
                source = getattr(
                    core, "native_physical_event_source_binding", None
                )
                if (
                    type(source) is not n1_reward_event_kernel.SourceBinding
                    or source.event_contract_sha256
                    != expected_native_event_contract_sha256
                ):
                    raise VecEnvContractError(
                        "native physical event source binding is absent or differs"
                    )
                sources.append(source)
            self.native_physical_event_runtime_available = True
            self.native_physical_event_source_bindings = tuple(sources)
        torch = _require_torch()
        self.device = torch.device("cpu")
        self.cfg = {
            "kind": "a3_mujoco_n1_diagnostic_vecenv_v2",
            "num_envs": self.num_envs,
            "observation_width": OBSERVATION_WIDTH,
            "reward_available": False,
            "exact_phase_fidelity_runtime_available": (
                self.exact_phase_fidelity_runtime_available
            ),
            "native_physical_event_runtime_available": (
                self.native_physical_event_runtime_available
            ),
            "diagnostic_unauthorized": True,
        }
        self.unwrapped = self
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._exact_hard_terminated_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._observations = torch.empty(
            (self.num_envs, OBSERVATION_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        self._has_reset = False
        self._event_ledgers = tuple(
            DiagnosticEventLedger(
                self.control_decimation,
                phase_fidelity_runtime_available=(
                    self.exact_phase_fidelity_runtime_available
                ),
            )
            for _ in self.cores
        )
        # rsl_rl's OnPolicyRunner asks for observations during construction;
        # native VecEnv instances therefore have to own a valid reset state
        # before the runner is allowed to inspect them.
        self.reset()

    @classmethod
    def from_authorities(
        cls,
        *,
        contract_path: Path | str,
        robot_tape_path: Path | str,
        expected_robot_tape_sha256: str,
        question_path: Path | str,
        expected_question_sha256: str,
        num_envs: int,
        mjcf_path: Path | str = single_env.DEFAULT_MJCF,
        phase_fidelity_reference_tape_path: Path | str | None = None,
        expected_phase_fidelity_reference_tape_sha256: str | None = None,
    ) -> "MujocoN1DiagnosticVecEnv":
        if type(num_envs) is not int or num_envs < 1:
            raise VecEnvContractError("num_envs must be a positive plain integer")
        binding = single_env.load_plant_binding(contract_path)
        robot_source = Path(robot_tape_path).expanduser().resolve()
        if hashlib.sha256(robot_source.read_bytes()).hexdigest() != expected_robot_tape_sha256:
            raise VecEnvContractError("robot tape file SHA differs from external authority")
        robot_tape = single_env.load_fixed_tape(robot_source, binding)
        if (phase_fidelity_reference_tape_path is None) != (
            expected_phase_fidelity_reference_tape_sha256 is None
        ):
            raise VecEnvContractError(
                "phase reference tape path and expected SHA must be supplied together"
            )
        phase_reference_tape = (
            None
            if phase_fidelity_reference_tape_path is None
            else n1_ball_core.load_phase_fidelity_reference_tape(
                phase_fidelity_reference_tape_path,
                expected_file_sha256=(
                    expected_phase_fidelity_reference_tape_sha256
                ),
                sample_contract=phase_fidelity_sample_contract(),
            )
        )
        cores = tuple(
            n1_ball_core.MujocoN1BallCore(
                binding,
                mjcf_path=mjcf_path,
                phase_fidelity_reference_tape=phase_reference_tape,
            )
            for _ in range(num_envs)
        )
        scene_sha = cores[0].scene_binding_sha256
        if any(core.scene_binding_sha256 != scene_sha for core in cores):
            raise VecEnvContractError("fresh cores do not share one scene binding SHA")
        classifier_binding = cores[0].selected_rubber_classifier_binding
        classifier_sha = classifier_binding["content_sha256"]
        if any(
            core.selected_rubber_classifier_binding["content_sha256"]
            != classifier_sha
            for core in cores
        ):
            raise VecEnvContractError(
                "fresh cores do not share one selected-rubber classifier binding"
            )
        question = n1_ball_core.load_question(
            question_path,
            expected_file_sha256=expected_question_sha256,
            scene_binding_sha256=scene_sha,
            selected_rubber_classifier_binding=classifier_binding,
        )
        return cls(
            cores=cores,
            robot_tape=robot_tape,
            questions=(question,) * num_envs,
        )

    def _tensor_observations(
        self, groups: Sequence[Mapping[str, Any]]
    ) -> Any:
        torch = _require_torch()
        values = np.stack([flatten_observation_groups(row) for row in groups], axis=0)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        self._has_reset = False
        try:
            groups = [
                core.reset(robot_tape=self.robot_tape, question=question)
                for core, question in zip(self.cores, self.questions)
            ]
        except Exception as exc:  # noqa: BLE001 - core reset is an external boundary
            raise VecEnvContractError("full VecEnv reset failed") from exc
        self.episode_length_buf.zero_()
        self._exact_hard_terminated_buf.zero_()
        self._event_ledgers = tuple(
            DiagnosticEventLedger(
                self.control_decimation,
                phase_fidelity_runtime_available=(
                    self.exact_phase_fidelity_runtime_available
                ),
            )
            for _ in self.cores
        )
        self._observations = self._tensor_observations(groups)
        self._has_reset = True
        return self.get_observations()

    def _compact_reset(
        self, episode_dones: Any
    ) -> tuple[int, ...]:
        """Reset exactly the completed rows; invalidate the batch on failure."""

        torch = _require_torch()
        if (
            not isinstance(episode_dones, torch.Tensor)
            or episode_dones.dtype != torch.bool
            or episode_dones.device.type != "cpu"
            or tuple(episode_dones.shape) != (self.num_envs,)
        ):
            raise VecEnvContractError("compact-reset mask must be a CPU bool env vector")
        reset_env_ids = tuple(
            int(value)
            for value in torch.nonzero(
                episode_dones, as_tuple=False
            ).flatten().tolist()
        )
        if not reset_env_ids:
            return ()
        reset_groups: list[Mapping[str, Any]] = []
        try:
            for index in reset_env_ids:
                reset_groups.append(
                    self.cores[index].reset(
                        robot_tape=self.robot_tape,
                        question=self.questions[index],
                    )
                )
            reset_observations = self._tensor_observations(reset_groups)
        except Exception as exc:  # noqa: BLE001 - partial core reset is unrecoverable
            self._has_reset = False
            raise VecEnvContractError(
                "per-env compact reset failed; full VecEnv reset is required"
            ) from exc
        ledgers = list(self._event_ledgers)
        for row, index in enumerate(reset_env_ids):
            self._observations[index].copy_(reset_observations[row])
            self.episode_length_buf[index] = 0
            self._exact_hard_terminated_buf[index] = False
            ledgers[index] = DiagnosticEventLedger(
                self.control_decimation,
                phase_fidelity_runtime_available=(
                    self.exact_phase_fidelity_runtime_available
                ),
            )
        self._event_ledgers = tuple(ledgers)
        return reset_env_ids

    def get_observations(self) -> tuple[Any, dict[str, Any]]:
        if not self._has_reset:
            raise VecEnvContractError("VecEnv must be reset before observations")
        observations = self._observations.clone()
        return observations, {
            "observations": {"critic": observations.clone()},
            "reward_contract": reward_blocker_receipt(),
            "native_physical_event_contract": {
                **n1_reward_event_kernel.native_physical_event_facts_contract(),
                "runtime_available": self.native_physical_event_runtime_available,
            },
            "termination_contract": termination_blocker_receipt(
                phase_fidelity_runtime_available=(
                    self.exact_phase_fidelity_runtime_available
                )
            ),
        }

    def diagnostic_step(self, actions: Any) -> DiagnosticBatchStep:
        """Advance physics without manufacturing a reward tensor."""

        torch = _require_torch()
        if not self._has_reset:
            raise VecEnvContractError("VecEnv must be reset before diagnostic_step")
        if bool(torch.any(self.episode_length_buf >= self.max_episode_length).item()):
            raise VecEnvContractError(
                "compact-reset invariant violated by a latched time_out; full reset required"
            )
        if bool(torch.any(self._exact_hard_terminated_buf).item()):
            raise VecEnvContractError(
                "compact-reset invariant violated by a latched hard termination; "
                "full reset required"
            )
        if not isinstance(actions, torch.Tensor):
            raise VecEnvContractError("actions must be a torch.Tensor")
        if actions.shape != (self.num_envs, self.num_actions):
            raise VecEnvContractError(
                f"actions must have shape ({self.num_envs}, {self.num_actions})"
            )
        if actions.device.type != "cpu" or not torch.isfinite(actions).all():
            raise VecEnvContractError("actions must be finite CPU values")
        try:
            rows = []
            events = []
            plant_rows = []
            phase_fidelity_samples = []
            native_physical_event_facts = []
            for index, (core, action) in enumerate(
                zip(self.cores, actions.detach().cpu().numpy())
            ):
                result = core.step(action)
                rows.append(result["observation_groups"])
                events.append(tuple(dict(value) for value in result["new_events"]))
                plant_rows.append(result["plant"])
                phase_sample = result.get("phase_fidelity_sample")
                if self.exact_phase_fidelity_runtime_available:
                    if not isinstance(phase_sample, Mapping):
                        raise VecEnvContractError(
                            "phase-fidelity ABI-advertising core omitted its sample"
                        )
                    phase_sample = _canonical_phase_fidelity_sample(phase_sample)
                elif phase_sample is not None:
                    raise VecEnvContractError(
                        "core returned a phase-fidelity sample without advertising "
                        "the exact ABI"
                    )
                phase_fidelity_samples.append(phase_sample)
                native_facts = result.get("native_physical_event_facts")
                if self.native_physical_event_runtime_available:
                    if not isinstance(native_facts, Mapping):
                        raise VecEnvContractError(
                            "native-event ABI-advertising core omitted its facts"
                        )
                    native_facts = _canonical_native_physical_event_facts(
                        native_facts,
                        expected_source=(
                            self.native_physical_event_source_bindings[index]
                        ),
                    )
                elif native_facts is not None:
                    raise VecEnvContractError(
                        "core returned native physical event facts without "
                        "advertising the exact ABI"
                    )
                native_physical_event_facts.append(native_facts)
            self.episode_length_buf += 1
            pre_reset_observations = self._tensor_observations(rows)
            self._observations = pre_reset_observations.clone()
            time_outs = self.episode_length_buf >= self.max_episode_length

            # Validate and commit every ledger as one batch.  A bad row cannot
            # leave earlier ledgers advanced while later rows remain stale.
            candidate_ledgers = tuple(
                copy.deepcopy(ledger) for ledger in self._event_ledgers
            )
            ledgers = tuple(
                ledger.record_step(
                    plant=plant,
                    events=event_rows,
                    time_out=bool(time_outs[index].item()),
                    phase_fidelity_sample=phase_fidelity_samples[index],
                )
                for index, (ledger, plant, event_rows) in enumerate(
                    zip(candidate_ledgers, plant_rows, events)
                )
            )
            self._event_ledgers = candidate_ledgers
            exact_hard_terminations = torch.as_tensor(
                [
                    bool(row["termination"]["exact_hard_terminated"])
                    for row in ledgers
                ],
                dtype=torch.bool,
                device=self.device,
            )
            exact_hard_reasons = tuple(
                row["termination"]["exact_hard_reason"] for row in ledgers
            )
            self._exact_hard_terminated_buf.copy_(exact_hard_terminations)
            episode_dones = exact_hard_terminations | time_outs
            episode_done_reasons = tuple(
                "time_out"
                if bool(time_outs[index].item())
                else exact_hard_reasons[index]
                for index in range(self.num_envs)
            )
            reset_env_ids = self._compact_reset(episode_dones)
            return DiagnosticBatchStep(
                observations=self._observations.clone(),
                terminal_observations=pre_reset_observations.clone(),
                terminal_observation_mask=episode_dones.clone(),
                episode_dones=episode_dones.clone(),
                episode_done_reasons=episode_done_reasons,
                reset_env_ids=reset_env_ids,
                exact_phase_fidelity_runtime_available=(
                    self.exact_phase_fidelity_runtime_available
                ),
                per_env_phase_fidelity_samples=tuple(
                    copy.deepcopy(value) for value in phase_fidelity_samples
                ),
                native_physical_event_runtime_available=(
                    self.native_physical_event_runtime_available
                ),
                per_env_native_physical_event_facts=tuple(
                    copy.deepcopy(value)
                    for value in native_physical_event_facts
                ),
                per_env_events=tuple(events),
                per_env_ledgers=tuple(copy.deepcopy(row) for row in ledgers),
                time_outs=time_outs.clone(),
                exact_hard_terminations=exact_hard_terminations.clone(),
                exact_hard_termination_reasons=exact_hard_reasons,
            )
        except Exception:
            self._has_reset = False
            raise

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Refuse rsl_rl rollout until a real reward contract is ported."""

        del actions
        blockers = ",".join(REWARD_BLOCKERS)
        raise RewardContractMissing(
            "PPO step is blocked before physics: no real ActionBall reward contract; "
            f"missing={blockers}"
        )

    def assert_ppo_ready(self) -> None:
        raise RewardContractMissing(
            "PPO/save/cold-load/resume smoke is prohibited until reward_blocker_receipt "
            "reports reward_available=true"
        )

    def run_diagnostic_rollout(self, actions: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and run ``[steps, envs, 31]`` actions with no reward."""

        torch = _require_torch()
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise VecEnvContractError("rollout actions must be [steps, envs, actions]")
        if tuple(actions.shape[1:]) != (self.num_envs, self.num_actions):
            raise VecEnvContractError("rollout action batch shape differs from VecEnv")
        initial, _extras = self.reset()
        traces = [initial.detach().cpu().numpy().copy()]
        event_rows = []
        native_physical_event_rows = []
        ledger_rows = []
        termination_rows = []
        terminal_observation_rows = []
        for action in actions:
            step = self.diagnostic_step(action)
            traces.append(step.observations.detach().cpu().numpy().copy())
            terminal_observation_rows.append(
                step.terminal_observations.detach().cpu().numpy().copy()
            )
            event_rows.append(
                [[dict(value) for value in env_events] for env_events in step.per_env_events]
            )
            native_physical_event_rows.append(
                [
                    None if value is None else copy.deepcopy(dict(value))
                    for value in step.per_env_native_physical_event_facts
                ]
            )
            ledger_rows.append([dict(value) for value in step.per_env_ledgers])
            termination_rows.append(
                {
                    "episode_dones": step.episode_dones.tolist(),
                    "episode_done_reasons": list(step.episode_done_reasons),
                    "terminal_observation_mask": (
                        step.terminal_observation_mask.tolist()
                    ),
                    "time_outs": step.time_outs.tolist(),
                    "exact_hard_terminations": (
                        step.exact_hard_terminations.tolist()
                    ),
                    "exact_hard_termination_reasons": list(
                        step.exact_hard_termination_reasons
                    ),
                    "reset_env_ids": list(step.reset_env_ids),
                    "phase_fidelity_samples": [
                        None if value is None else dict(value)
                        for value in step.per_env_phase_fidelity_samples
                    ],
                }
            )
        trace = np.stack(traces, axis=0)
        terminal_observation_trace = (
            np.stack(terminal_observation_rows, axis=0)
            if terminal_observation_rows
            else np.empty(
                (0, self.num_envs, OBSERVATION_WIDTH), dtype=np.float32
            )
        )
        semantic = {
            "shape": list(trace.shape),
            "returned_trace_dtype": str(trace.dtype),
            "canonical_digest_dtype": "<f8",
            "observation_layout": [
                {"name": name, "width": width} for name, width in OBSERVATION_LAYOUT
            ],
            "plant_binding_sha256": self.robot_tape.plant_binding_sha256,
            "scene_binding_sha256": self.cores[0].scene_binding_sha256,
            "robot_tape_sha256": self.robot_tape.source_sha256,
            "question_source_sha256_by_env": list(
                self.question_source_sha256_by_env
            ),
            "exact_phase_fidelity_runtime_available": (
                self.exact_phase_fidelity_runtime_available
            ),
            "phase_fidelity_sample_contract_sha256": (
                _phase_fidelity_sample_contract_cached()["content_sha256"]
            ),
            "phase_fidelity_reference_tape_sha256_by_env": list(
                self.phase_fidelity_reference_tape_sha256_by_env
            ),
            "native_physical_event_runtime_available": (
                self.native_physical_event_runtime_available
            ),
            "native_physical_event_contract_sha256": (
                n1_reward_event_kernel.native_physical_event_facts_contract()[
                    "content_sha256"
                ]
            ),
            "observation_transition_semantics": (
                "trace_is_post_compact_reset_next_state; terminal_observation_trace_"
                "is_pre_reset_and_valid_only_under_termination_transcript_mask"
            ),
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(trace, dtype="<f8").tobytes())
        canonical_terminal_trace = np.ascontiguousarray(
            terminal_observation_trace, dtype="<f8"
        )
        terminal_observation_descriptor = {
            "schema_version": 1,
            "storage": "digest_only_not_returned",
            "shape": list(terminal_observation_trace.shape),
            "source_dtype": str(terminal_observation_trace.dtype),
            "canonical_digest_dtype": "<f8",
            "sha256": hashlib.sha256(
                canonical_terminal_trace.tobytes()
            ).hexdigest(),
            "validity_mask_source": (
                "termination_transcript[*].terminal_observation_mask"
            ),
        }
        digest.update(
            json.dumps(
                terminal_observation_descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(
            json.dumps(event_rows, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(
            json.dumps(
                native_physical_event_rows,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(
            json.dumps(ledger_rows, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(
            json.dumps(
                termination_rows, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        receipt = {
            "schema_version": 4,
            "kind": "a3_mujoco_n1_diagnostic_vecenv_rollout_v4",
            "status": "DIAGNOSTIC_NO_REWARD_ROLLOUT_COMPLETE",
            "num_envs": self.num_envs,
            "steps": int(actions.shape[0]),
            "observation_shape": list(trace.shape),
            "semantic": semantic,
            "question_source_sha256_by_env": list(
                self.question_source_sha256_by_env
            ),
            "event_transcript": event_rows,
            "native_physical_event_transcript": native_physical_event_rows,
            "event_ledger_transcript": ledger_rows,
            "termination_transcript": termination_rows,
            "terminal_observation_trace": terminal_observation_descriptor,
            "terminal_observation_semantics": (
                "pre_reset_full_batch_valid_only_where_terminal_observation_mask"
            ),
            "returned_observation_semantics": "post_compact_reset_next_observation",
            "trace_and_event_digest_contract": {
                "algorithm": "sha256",
                "ordered_inputs": [
                    "canonical_semantic_json_utf8",
                    "returned_trace_c_contiguous_little_endian_f8_bytes",
                    "canonical_terminal_observation_trace_descriptor_json_utf8",
                    "canonical_event_transcript_json_utf8",
                    "canonical_native_physical_event_transcript_json_utf8",
                    "canonical_event_ledger_transcript_json_utf8",
                    "canonical_termination_transcript_json_utf8",
                ],
                "json_sort_keys": True,
                "json_separators": [",", ":"],
            },
            "final_event_ledgers": [ledger.snapshot() for ledger in self._event_ledgers],
            "trace_and_event_sha256": digest.hexdigest(),
            "reward_blocker": reward_blocker_receipt(),
            "termination_blocker": termination_blocker_receipt(
                phase_fidelity_runtime_available=(
                    self.exact_phase_fidelity_runtime_available
                )
            ),
            "diagnostic_unauthorized": True,
            "authorization": {
                "training": False,
                "promotion": False,
                "deployment": False,
                "hardware": False,
            },
        }
        receipt["content_sha256"] = _sha256_json(receipt)
        return trace, receipt


__all__ = [
    "DiagnosticBatchStep",
    "DiagnosticEventLedger",
    "BASE_FELL_TILT_LIMIT_ANGLE_RAD",
    "BASE_FELL_TILT_MIN_UP_WORLD_Z",
    "BASE_TOO_LOW_MINIMUM_HEIGHT_M",
    "EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256",
    "EXPECTED_PHASE_BASE_CONFIG_SEMANTIC_AST_SHA256",
    "EXPECTED_PHASE_RAW_CALLABLES_SEMANTIC_AST_SHA256",
    "EXPECTED_PHASE_WRAPPERS_SEMANTIC_AST_SHA256",
    "EXPECTED_PHASE_GATE_SEMANTIC_AST_SHA256",
    "EXPECTED_PHASE_BODY_NAMES_SEMANTIC_AST_SHA256",
    "EXACT_ACTIVE_TERMINATION_REASON_ORDER",
    "EXACT_BASE_TERMINATION_REASON_ORDER",
    "EXACT_HARD_TERMINATION_REASON_ORDER",
    "EXACT_PHASE_FIDELITY_REASON_ORDER",
    "FORMAL_TERMINATION_BLOCKERS",
    "MujocoN1DiagnosticVecEnv",
    "OBSERVATION_LAYOUT",
    "OBSERVATION_WIDTH",
    "REWARD_BLOCKERS",
    "TERMINATION_SOURCE_CONFIG",
    "TERMINATION_SOURCE_CALLABLES",
    "TERMINATION_SOURCE_ACTION_LATCH",
    "TERMINATION_SOURCE_PHASE_WRAPPERS",
    "TERMINATION_SOURCE_PHASE_GATE",
    "TERMINATION_SOURCE_BASE_CONFIG",
    "TERMINATION_SOURCE_A3_BODY_NAMES",
    "PHASE_ANCHOR_POS_Z_THRESHOLD_M",
    "PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD",
    "PHASE_EE_BODY_POS_Z_THRESHOLD_M",
    "PHASE_EE_BODY_NAMES",
    "JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD",
    "RewardContractMissing",
    "VecEnvContractError",
    "flatten_observation_groups",
    "exact_phase_fidelity_reasons",
    "phase_fidelity_sample_contract",
    "reward_blocker_receipt",
    "termination_blocker_receipt",
]
