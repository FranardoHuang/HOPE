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
    "phase_fidelity_and_recovery_termination_contract_not_frozen",
    "terminated_batch_compact_reset_and_terminal_observation_not_implemented",
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
EXACT_HARD_TERMINATION_REASON_ORDER = (
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
EXPECTED_TERMINATION_SOURCE_CONFIG_SHA256 = (
    table_termination.EXPECTED_ISAAC_TERMINATION_CONFIG_SHA256
)
TERMINATION_SOURCE_CALLABLES = (
    table_termination.ISAAC_TERMINATION_CALLABLES
)
EXPECTED_TERMINATION_SOURCE_CALLABLES_SHA256 = (
    table_termination.EXPECTED_ISAAC_TERMINATION_CALLABLES_SHA256
)
TERMINATION_SOURCE_ACTION_LATCH = table_termination.ISAAC_ACTION_LATCH
EXPECTED_TERMINATION_SOURCE_ACTION_LATCH_SHA256 = (
    table_termination.EXPECTED_ISAAC_ACTION_LATCH_SHA256
)
JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD = single_env.JOINT_BOUNDS_TOLERANCE_RAD
JOINT_QDES_FORBIDDEN_MARGIN_RAD = 0.0
JOINT_QDES_FORBIDDEN_MARGIN_FRACTION = 0.02
JOINT_QDES_FINITE_PROJECTION_ENABLED = True

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
    try:
        source_config_sha256 = hashlib.sha256(
            TERMINATION_SOURCE_CONFIG.read_bytes()
        ).hexdigest()
        source_callables_sha256 = hashlib.sha256(
            TERMINATION_SOURCE_CALLABLES.read_bytes()
        ).hexdigest()
        source_action_latch_sha256 = hashlib.sha256(
            TERMINATION_SOURCE_ACTION_LATCH.read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise VecEnvContractError(
            "cannot bind the exact base termination source config"
        ) from exc
    if source_config_sha256 != EXPECTED_TERMINATION_SOURCE_CONFIG_SHA256:
        raise VecEnvContractError(
            "exact base termination source config SHA-256 drifted"
        )
    if source_callables_sha256 != EXPECTED_TERMINATION_SOURCE_CALLABLES_SHA256:
        raise VecEnvContractError(
            "exact termination source callables SHA-256 drifted"
        )
    if (
        source_action_latch_sha256
        != EXPECTED_TERMINATION_SOURCE_ACTION_LATCH_SHA256
    ):
        raise VecEnvContractError(
            "exact termination action-latch source SHA-256 drifted"
        )
    payload = {
        "schema_version": 6,
        "kind": "a3_mujoco_n1_vecenv_termination_blocker_v6",
        "status": "FORMAL_TERMINATION_BLOCKED",
        "formal_termination_available": False,
        "terminated_tensor_available": False,
        "exact_base_subset_available": True,
        "exact_base_subset_terminated_tensor_available": True,
        "exact_robot_table_termination_available": True,
        "exact_hard_subset_terminated_tensor_available": True,
        "exact_time_out_latch_available": True,
        "exact_active_reason_order": list(EXACT_ACTIVE_TERMINATION_REASON_ORDER),
        "exact_hard_reason_order": list(EXACT_HARD_TERMINATION_REASON_ORDER),
        "exact_base_subset": {
            "reason_order": list(EXACT_BASE_TERMINATION_REASON_ORDER),
            "source_config_path": str(TERMINATION_SOURCE_CONFIG),
            "source_config_sha256": source_config_sha256,
            "source_callables_path": str(TERMINATION_SOURCE_CALLABLES),
            "source_callables_sha256": source_callables_sha256,
            "reason_order_scope": (
                "priority inside the installed base subset; complete hard-reason "
                "ordering remains blocked"
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
            "predicate": (
                "43 pinned component OBBs plus live racket OBB conservatively "
                "broadened to world AABBs against five inflated table AABBs"
            ),
            "sample_timing": "after_each_physics_substep",
            "sticky_within_control_step": True,
            "required_control_decimation": 4,
            "source_action_latch_path": str(TERMINATION_SOURCE_ACTION_LATCH),
            "source_action_latch_sha256": source_action_latch_sha256,
            "substep_latch_scope": "current_control_step_plant_sample",
            "episode_sticky_owner": "DiagnosticEventLedger",
            "diagnostic_step_after_latch_requires_explicit_reset": True,
            "immediate_compact_reset_implemented": False,
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
        "exact_diagnostic_facts": [
            "ball_racket_table_net_floor_contact_edges_at_physics_substep",
            "robot_obstacle_and_self_contact_substep_counts",
            "qdes_clamp_effort_clip_and_joint_velocity_limit_counts",
            "pre_clamp_qdes_nonfinite_hard_termination_latch",
            "pelvis_height_and_world_up_z_samples",
        ],
        "blockers": list(FORMAL_TERMINATION_BLOCKERS),
        "semantic_boundary": (
            "the exact hard subset and tape time-out do not replace the remaining "
            "Isaac phase-fidelity termination union, "
            "terminal observation, or compact reset"
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


def termination_blocker_receipt() -> dict[str, Any]:
    """Return a caller-owned copy of the exact-subset/full-blocker receipt."""

    return copy.deepcopy(_termination_blocker_receipt_cached())


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

    def __post_init__(self) -> None:
        if type(self.control_decimation) is not int or self.control_decimation != 4:
            raise VecEnvContractError(
                "exact Isaac robot/table termination requires control_decimation=4"
            )

    def record_step(
        self,
        *,
        plant: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        time_out: bool,
    ) -> dict[str, Any]:
        """Validate one complete control tick, then commit its cumulative facts."""

        if not isinstance(plant, Mapping):
            raise VecEnvContractError("diagnostic plant row must be a mapping")
        if type(time_out) is not bool:
            raise VecEnvContractError("diagnostic time_out must be bool")
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

        exact_hard_reasons = []
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
                else dict(self.first_exact_hard_termination)
            ),
            "first_robot_obstacle_contact": self.first_robot_obstacle_contact,
            "first_robot_self_contact": self.first_robot_self_contact,
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
                "exact_hard_terminated": (
                    self.exact_hard_termination_latched
                ),
                "exact_hard_reason": (
                    None
                    if self.first_exact_hard_termination is None
                    else self.first_exact_hard_termination["reason"]
                ),
                "formal_hard_termination_available": False,
                "formal_hard_terminated": None,
                "blocker_sha256": _termination_blocker_receipt_cached()[
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
    observations: Any
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
        torch = _require_torch()
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
        self.num_envs = len(self.cores)
        self.num_actions = single_env.ACTION_DIM
        self.max_episode_length = int(robot_tape.actions.shape[0])
        self.device = torch.device("cpu")
        self.cfg = {
            "kind": "a3_mujoco_n1_diagnostic_vecenv_v2",
            "num_envs": self.num_envs,
            "observation_width": OBSERVATION_WIDTH,
            "reward_available": False,
            "diagnostic_unauthorized": True,
        }
        self.unwrapped = self
        self.step_dt = float(self.cores[0].binding.policy_step_dt_s)
        decimations = {int(core.binding.control_decimation) for core in self.cores}
        if decimations != {4}:
            raise VecEnvContractError(
                "exact Isaac robot/table termination requires every core to use "
                "control_decimation=4"
            )
        self.control_decimation = next(iter(decimations))
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
            DiagnosticEventLedger(self.control_decimation) for _ in self.cores
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
    ) -> "MujocoN1DiagnosticVecEnv":
        if type(num_envs) is not int or num_envs < 1:
            raise VecEnvContractError("num_envs must be a positive plain integer")
        binding = single_env.load_plant_binding(contract_path)
        robot_source = Path(robot_tape_path).expanduser().resolve()
        if hashlib.sha256(robot_source.read_bytes()).hexdigest() != expected_robot_tape_sha256:
            raise VecEnvContractError("robot tape file SHA differs from external authority")
        robot_tape = single_env.load_fixed_tape(robot_source, binding)
        cores = tuple(
            n1_ball_core.MujocoN1BallCore(binding, mjcf_path=mjcf_path)
            for _ in range(num_envs)
        )
        scene_sha = cores[0].scene_binding_sha256
        if any(core.scene_binding_sha256 != scene_sha for core in cores):
            raise VecEnvContractError("fresh cores do not share one scene binding SHA")
        question = n1_ball_core.load_question(
            question_path,
            expected_file_sha256=expected_question_sha256,
            scene_binding_sha256=scene_sha,
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
        groups = [
            core.reset(robot_tape=self.robot_tape, question=question)
            for core, question in zip(self.cores, self.questions)
        ]
        self.episode_length_buf.zero_()
        self._exact_hard_terminated_buf.zero_()
        self._event_ledgers = tuple(
            DiagnosticEventLedger(self.control_decimation) for _ in self.cores
        )
        self._observations = self._tensor_observations(groups)
        self._has_reset = True
        return self.get_observations()

    def get_observations(self) -> tuple[Any, dict[str, Any]]:
        if not self._has_reset:
            raise VecEnvContractError("VecEnv must be reset before observations")
        observations = self._observations.clone()
        return observations, {
            "observations": {"critic": observations.clone()},
            "reward_contract": reward_blocker_receipt(),
            "termination_contract": termination_blocker_receipt(),
        }

    def diagnostic_step(self, actions: Any) -> DiagnosticBatchStep:
        """Advance physics without manufacturing a reward tensor."""

        torch = _require_torch()
        if not self._has_reset:
            raise VecEnvContractError("VecEnv must be reset before diagnostic_step")
        if bool(torch.any(self.episode_length_buf >= self.max_episode_length).item()):
            raise VecEnvContractError(
                "diagnostic step after exact time_out requires an explicit reset"
            )
        if bool(torch.any(self._exact_hard_terminated_buf).item()):
            raise VecEnvContractError(
                "diagnostic step after exact hard termination requires an "
                "explicit reset"
            )
        if not isinstance(actions, torch.Tensor):
            raise VecEnvContractError("actions must be a torch.Tensor")
        if actions.shape != (self.num_envs, self.num_actions):
            raise VecEnvContractError(
                f"actions must have shape ({self.num_envs}, {self.num_actions})"
            )
        if actions.device.type != "cpu" or not torch.isfinite(actions).all():
            raise VecEnvContractError("actions must be finite CPU values")
        rows = []
        events = []
        plant_rows = []
        for core, action in zip(self.cores, actions.detach().cpu().numpy()):
            result = core.step(action)
            rows.append(result["observation_groups"])
            events.append(tuple(dict(value) for value in result["new_events"]))
            plant_rows.append(result["plant"])
        self.episode_length_buf += 1
        self._observations = self._tensor_observations(rows)
        time_outs = self.episode_length_buf >= self.max_episode_length
        ledgers = tuple(
            ledger.record_step(
                plant=plant,
                events=event_rows,
                time_out=bool(time_outs[index].item()),
            )
            for index, (ledger, plant, event_rows) in enumerate(
                zip(self._event_ledgers, plant_rows, events)
            )
        )
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
        self._exact_hard_terminated_buf.copy_(
            exact_hard_terminations
        )
        return DiagnosticBatchStep(
            observations=self._observations.clone(),
            per_env_events=tuple(events),
            per_env_ledgers=ledgers,
            time_outs=time_outs.clone(),
            exact_hard_terminations=(
                exact_hard_terminations.clone()
            ),
            exact_hard_termination_reasons=exact_hard_reasons,
        )

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
        ledger_rows = []
        for action in actions:
            step = self.diagnostic_step(action)
            traces.append(step.observations.detach().cpu().numpy().copy())
            event_rows.append(
                [[dict(value) for value in env_events] for env_events in step.per_env_events]
            )
            ledger_rows.append([dict(value) for value in step.per_env_ledgers])
        trace = np.stack(traces, axis=0)
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
            "question_sha256": self.questions[0].source_sha256,
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(trace, dtype="<f8").tobytes())
        digest.update(
            json.dumps(event_rows, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(
            json.dumps(ledger_rows, sort_keys=True, separators=(",", ":")).encode()
        )
        receipt = {
            "schema_version": 2,
            "kind": "a3_mujoco_n1_diagnostic_vecenv_rollout_v2",
            "status": "DIAGNOSTIC_NO_REWARD_ROLLOUT_COMPLETE",
            "num_envs": self.num_envs,
            "steps": int(actions.shape[0]),
            "observation_shape": list(trace.shape),
            "event_transcript": event_rows,
            "event_ledger_transcript": ledger_rows,
            "final_event_ledgers": [ledger.snapshot() for ledger in self._event_ledgers],
            "trace_and_event_sha256": digest.hexdigest(),
            "reward_blocker": reward_blocker_receipt(),
            "termination_blocker": termination_blocker_receipt(),
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
    "EXPECTED_TERMINATION_SOURCE_CONFIG_SHA256",
    "EXPECTED_TERMINATION_SOURCE_CALLABLES_SHA256",
    "EXPECTED_TERMINATION_SOURCE_ACTION_LATCH_SHA256",
    "EXACT_ACTIVE_TERMINATION_REASON_ORDER",
    "EXACT_BASE_TERMINATION_REASON_ORDER",
    "EXACT_HARD_TERMINATION_REASON_ORDER",
    "FORMAL_TERMINATION_BLOCKERS",
    "MujocoN1DiagnosticVecEnv",
    "OBSERVATION_LAYOUT",
    "OBSERVATION_WIDTH",
    "REWARD_BLOCKERS",
    "TERMINATION_SOURCE_CONFIG",
    "TERMINATION_SOURCE_CALLABLES",
    "TERMINATION_SOURCE_ACTION_LATCH",
    "JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD",
    "RewardContractMissing",
    "VecEnvContractError",
    "flatten_observation_groups",
    "reward_blocker_receipt",
    "termination_blocker_receipt",
]
