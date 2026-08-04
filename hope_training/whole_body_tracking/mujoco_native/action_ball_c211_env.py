"""Real-MjData producer for the fresh C211 actor/critic observation ABI.

The existing native lane intentionally keeps its historical 76-D observation
surface.  This module wraps that lane without mutating it and reconstructs the
fresh C211 211-D actor plus 319-D privileged critic from three independently
checked authorities:

* live MuJoCo plant/FK state from :class:`MujocoN1BallCore`;
* the measured full-body/racket NPZ used by the immutable N1 question; and
* the strict immutable question tape itself.

No missing column is padded.  RESET_WAIT is the sole authorized zeroing path:
the 9-D incoming-ball tuple, base goal, and two public clocks are masked while
the split-ready physical frame-zero birth and plant state remain present.  The
measured frame-zero teacher is an independently bound reference; it is not a
second physical reset target.

The wrapper owns the C211 task reward as well as the observation ABI.  At the
single nominal strike tick it grades the achieved official racket site against
the immutable incoming-ball centre.  It grades a landing only from the first
source-bound outgoing flight after an actual selected-rubber contact.  The
return also consumes the Isaac-synonymous subset whose authorities are present
in MuJoCo: always-on balance/action regularizers, non-right-wrist full-body
mimic, and the measured physical-paddle prior.  Those priors remain active in
RESET_WAIT while the task-dependent strike/contact/outcome terms are masked.

This is still a diagnostic-only lane.  It implements the exact seeded WAIT
schedule and reset-boundary continuation contract, but it does not claim full
Isaac reward parity: foot/contact-sensor, undesired-contact, applied-torque and
several safety-manager terms remain explicitly unavailable.  Cross-engine
contact-model parity and a 4096-environment GPU workload also remain open.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from . import action_ball_211_abi as abi
from . import n1_ball_core
from . import n1_reward_event_kernel
from . import physical_ball_scene
from . import selected_rubber_classifier
from . import single_env
from . import trainer


C211_ENV_KIND = "action_ball_c211_mujoco_native_vecenv_v1"
C211_PLANT_PROVIDER_KIND = abi.PLANT_OBSERVATION_AUTHORITY_KIND
C211_MIMIC_PROVIDER_KIND = abi.MEASURED_MIMIC_AUTHORITY_KIND
C211_TASK_PROVIDER_KIND = abi.TASK_QUESTION_AUTHORITY_KIND
C211_TARGET_RECIPE = "outcome_dense_only"
C211_REWARD_SCOPE = "action_ball_c211_partial_isaac_synonymous_reward_v3"
C211_REWARD_CONTRACT_IDENTITY = (
    "action_ball_c211_native_partial_isaac_synonymous_reward_v3"
)
C211_TASK_REWARD_CONTRACT_IDENTITY = "action_ball_c211_achieved_outcome_reward_v2"
C211_POLICY_DT_S = 0.02
C211_STRIKE_STD_M = 0.15
C211_STRIKE_POST_DT_WEIGHT = 4.8
C211_LANDING_SIGMA_M = 1.0
C211_LANDING_POST_DT_WEIGHT = 14.0
C211_LANDING_LEGAL_BASE_FRAC = 0.6
C211_LANDING_OFF_TABLE_FRAC = 0.5
C211_ROLLOUT_H_S = 0.01
C211_ROLLOUT_STEPS = 100
C211_UPRIGHT_STD = math.sqrt(0.2)
C211_ACTION_RATE_CLAMP = 9.0
C211_RACKET_LONG_AXIS_LOCAL = np.asarray(
    (math.sqrt(0.5), 0.0, math.sqrt(0.5)), dtype=np.float64
)

VIRTUAL_BALL_PY = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "tasks/tracking/mdp/virtual_ball.py"
)
VENUE_PHYSICS_YAML = single_env.REPO_ROOT / "configs/ball_physics_venue.yaml"
C211_TRAINABILITY_PY = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "tasks/tracking/action_ball_c211_trainability.py"
)
C225_REWARD_PY = C211_TRAINABILITY_PY.parent / "mdp/action_ball_c225_rewards.py"
HOPE_REWARDS_PY = C211_TRAINABILITY_PY.parent / "mdp/hope_rewards.py"
HOPE_ENV_CFG_PY = C211_TRAINABILITY_PY.parent / "config/agibot_a3/hope_env_cfg.py"
TRAIN_PY = (
    single_env.REPO_ROOT / "hope_training/whole_body_tracking/scripts/train.py"
)
VENDOR_V2_TASK_YAML = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBallA3VendorV2.yaml"
)
C211_TASK_YAML = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBallC211VendorV2N1Learnability.yaml"
)

TRACKED_BODY_NAMES = (
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)
ANCHOR_BODY_NAME = "torso_Link"
ROOT_BODY_NAME = "pelvis_link"
RIGHT_WRIST_BODY_NAME = "right_wrist_yaw_Link"
MIMIC_BODY_NAMES = tuple(
    name for name in TRACKED_BODY_NAMES if name != RIGHT_WRIST_BODY_NAME
)
C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES = (
    "upright_exp",
    "base_ang_vel_xy",
    "base_lin_vel_z",
    "joint_vel",
    "action_rate_clamped",
    "motion_global_anchor_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "motion_racket_position",
    "motion_racket_velocity",
    "motion_racket_normal",
    "motion_racket_long_axis",
)

C211_UNAVAILABLE_ISAAC_REWARD_TERMS = (
    {
        "term": "hit_unstable_support",
        "reason": "requires_Isaac_two_foot_contact_sensor_state_at_control_step",
    },
    {
        "term": "foot_slip_sq",
        "reason": "requires_Isaac_foot_contact_attribution_and_link_point_velocity_semantics",
    },
    {
        "term": "foot_soft_landing",
        "reason": "requires_Isaac_substep_force_history_and_first_contact_latch",
    },
    {
        "term": "undesired_contacts",
        "reason": "requires_Isaac_contact_sensor_body_filter_and_force_threshold_semantics",
    },
    {
        "term": "joint_torques",
        "reason": "Isaac_applied_torque_is_not_synonymous_with_current_MuJoCo_total_PD_or_actuator_force",
    },
    {
        "term": "foot_velocity",
        "reason": "Isaac_foot_link_velocity_point_semantics_not_bound_in_native_reward_receipt",
    },
    {
        "term": "qdes_limit_barrier",
        "reason": "native_transition_does_not_export_the_exact_Isaac_reward_manager_raw_term",
    },
    {
        "term": "qdes_projection_penalty",
        "reason": "native_transition_does_not_export_the_exact_Isaac_projection_raw_term",
    },
    {
        "term": "joint_limit",
        "reason": "native_transition_does_not_export_the_exact_Isaac_actual_joint_barrier_raw_term",
    },
    {
        "term": "death_penalty",
        "reason": "full_Isaac_hard_safety_union_and_once_only_reward_receipt_not_bound",
    },
)
C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS: tuple[dict[str, str], ...] = tuple()

FORMAL_BLOCKERS = (
    "c211_split_ready_physical_birth_cross_engine_parity_unmeasured",
    "c211_full_body_mimic_and_measured_paddle_priors_implemented_but_cross_engine_parity_unmeasured",
    "c211_isaac_reward_parity_incomplete_foot_contact_undesired_contact_applied_torque_and_safety_terms",
    "c211_seeded_wait_has_no_cross_engine_runtime_parity_receipt",
    "c211_mujoco_incoming_launch_is_explicit_native_gravity_not_cross_engine_parity",
    "c211_mujoco_phase_recovery_export_and_mid_episode_resume_not_closed",
    "c211_mujoco_cpu_sequential_vecenv_has_no_4096_matched_workload_receipt",
)

SAFE_READY_AUTHORITY_STATUS = (
    "split_ready_physical_birth_diagnostic_only_cross_engine_unmeasured"
)


class C211EnvError(RuntimeError):
    """The real C211 provider or wrapped VecEnv failed closed."""


def _load_virtual_ball_module() -> Any:
    """Load the shared Isaac scorer without importing the Isaac package tree."""

    name = "_action_ball_c211_native_virtual_ball"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, VIRTUAL_BALL_PY)
    if spec is None or spec.loader is None:
        raise C211EnvError("shared virtual-ball scorer cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C211EnvError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _finite_vector(value: Any, width: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise C211EnvError(f"{name} is not numeric") from exc
    if array.shape != (width,) or not np.isfinite(array).all():
        raise C211EnvError(f"{name} must contain {width} finite scalars")
    return array.copy()


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise C211EnvError(f"{name} is not numeric") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise C211EnvError(f"{name} must have finite shape {shape}")
    return array.copy()


def _rotation_from_wxyz(value: Any, name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise C211EnvError(f"{name} must be unit length")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotation_to_6d(rotation: Any, name: str) -> np.ndarray:
    value = _finite_array(rotation, (3, 3), name)
    if not np.allclose(
        value.T @ value, np.eye(3), rtol=0.0, atol=1.0e-6
    ) or not math.isclose(
        float(np.linalg.det(value)), 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise C211EnvError(f"{name} is not a proper rotation")
    return value[:, :2].reshape(6)


def _yaw_rotation(rotation: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _array_digest(rows: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(rows):
        value = rows[name]
        digest.update(name.encode("utf-8") + b"\0")
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(_canonical_bytes(list(array.shape)))
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(_canonical_bytes(value))
    return digest.hexdigest()


def _rotation_error_magnitude(
    first: Any, second: Any, name: str
) -> float:
    """Return the SO(3) geodesic angle used by Isaac's quaternion error."""

    left = _finite_array(first, (3, 3), f"{name} first rotation")
    right = _finite_array(second, (3, 3), f"{name} second rotation")
    for label, value in (("first", left), ("second", right)):
        if not np.allclose(
            value.T @ value, np.eye(3), rtol=0.0, atol=1.0e-6
        ) or not math.isclose(
            float(np.linalg.det(value)), 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise C211EnvError(f"{name} {label} value is not a proper rotation")
    cosine = 0.5 * (float(np.trace(left.T @ right)) - 1.0)
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def _unit_vector(value: Any, name: str) -> np.ndarray:
    row = _finite_vector(value, 3, name)
    norm = float(np.linalg.norm(row))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise C211EnvError(f"{name} must be unit length")
    return row / norm


def _cauchy_kernel(error: float, std: float) -> float:
    if not math.isfinite(error) or error < 0.0 or not math.isfinite(std) or std <= 0.0:
        raise C211EnvError("C211 Cauchy reward arguments are invalid")
    return 1.0 / (1.0 + (error / std) ** 2)


def _prior_term(
    *, raw_reward: float, manager_weight: float, details: Mapping[str, Any]
) -> dict[str, Any]:
    raw = float(raw_reward)
    weight = float(manager_weight)
    reward = raw * weight * C211_POLICY_DT_S
    if not all(math.isfinite(value) for value in (raw, weight, reward)):
        raise C211EnvError("C211 prior reward term is non-finite")
    return {
        "raw_reward": raw,
        "manager_weight": weight,
        "policy_dt_s": C211_POLICY_DT_S,
        "post_policy_dt_reward": reward,
        **dict(details),
    }


def _c211_isaac_synonymous_prior_terms(
    *,
    live: Mapping[str, Any],
    teacher: Mapping[str, Any],
    current_action: Any,
    previous_action: Any,
) -> dict[str, Any]:
    """Evaluate the exact implemented subset of the resolved Isaac reward pack.

    Every row exposes the callable's unweighted raw value and the manager-weight
    plus policy-dt contribution.  The function deliberately knows nothing about
    ``task_valid``: balance, action regularization, body mimic and measured
    paddle mimic are always on, including RESET_WAIT.
    """

    root_rotation = _finite_array(
        live.get("root_rotation"), (3, 3), "C211 prior root rotation"
    )
    root_ang_body = root_rotation.T @ _finite_vector(
        live.get("root_ang_vel_w"), 3, "C211 prior root angular velocity"
    )
    root_lin_body = root_rotation.T @ _finite_vector(
        live.get("root_lin_vel_w"), 3, "C211 prior root COM linear velocity"
    )
    qd = _finite_vector(live.get("qd"), 31, "C211 prior joint velocity")
    action = _finite_vector(current_action, 31, "C211 prior current action")
    previous = _finite_vector(previous_action, 31, "C211 prior previous action")

    body_shape = (len(TRACKED_BODY_NAMES), 3)
    body_rotation_shape = (len(TRACKED_BODY_NAMES), 3, 3)
    live_body_pos = _finite_array(
        live.get("body_pos"), body_shape, "C211 prior live body positions"
    )
    teacher_body_pos = _finite_array(
        teacher.get("body_pos"), body_shape, "C211 prior teacher body positions"
    )
    live_body_rotation = _finite_array(
        live.get("body_rotation"),
        body_rotation_shape,
        "C211 prior live body rotations",
    )
    teacher_body_rotation = _finite_array(
        teacher.get("body_rotation"),
        body_rotation_shape,
        "C211 prior teacher body rotations",
    )
    live_body_lin_vel = _finite_array(
        live.get("body_lin_vel_w"),
        body_shape,
        "C211 prior live body COM velocities",
    )
    teacher_body_lin_vel = _finite_array(
        teacher.get("body_lin_vel_w"),
        body_shape,
        "C211 prior teacher body COM velocities",
    )
    live_body_ang_vel = _finite_array(
        live.get("body_ang_vel_w"),
        body_shape,
        "C211 prior live body angular velocities",
    )
    teacher_body_ang_vel = _finite_array(
        teacher.get("body_ang_vel_w"),
        body_shape,
        "C211 prior teacher body angular velocities",
    )
    mimic_indices = np.asarray(
        [TRACKED_BODY_NAMES.index(name) for name in MIMIC_BODY_NAMES],
        dtype=np.int64,
    )

    projected_gravity_b = root_rotation.T @ np.asarray(
        (0.0, 0.0, -1.0), dtype=np.float64
    )
    upright_error_sq = float(np.sum(np.square(projected_gravity_b[:2])))
    base_ang_vel_xy_raw = float(np.sum(np.square(root_ang_body[:2])))
    base_lin_vel_z_raw = float(root_lin_body[2] ** 2)
    joint_vel_raw = float(np.sum(np.square(qd)))
    action_rate_unclamped = float(np.sum(np.square(action - previous)))
    action_rate_raw = min(action_rate_unclamped, C211_ACTION_RATE_CLAMP)

    anchor_rotation_error = _rotation_error_magnitude(
        teacher.get("global_anchor_rotation"),
        live.get("anchor_rotation"),
        "C211 motion anchor orientation",
    )
    body_position_mse = float(
        np.mean(
            np.sum(
                np.square(
                    teacher_body_pos[mimic_indices]
                    - live_body_pos[mimic_indices]
                ),
                axis=-1,
            )
        )
    )
    body_orientation_errors = np.asarray(
        [
            _rotation_error_magnitude(
                teacher_body_rotation[index],
                live_body_rotation[index],
                f"C211 motion body orientation {TRACKED_BODY_NAMES[index]}",
            )
            for index in mimic_indices.tolist()
        ],
        dtype=np.float64,
    )
    body_orientation_mse = float(np.mean(np.square(body_orientation_errors)))
    body_lin_vel_mse = float(
        np.mean(
            np.sum(
                np.square(
                    teacher_body_lin_vel[mimic_indices]
                    - live_body_lin_vel[mimic_indices]
                ),
                axis=-1,
            )
        )
    )
    body_ang_vel_mse = float(
        np.mean(
            np.sum(
                np.square(
                    teacher_body_ang_vel[mimic_indices]
                    - live_body_ang_vel[mimic_indices]
                ),
                axis=-1,
            )
        )
    )

    racket_position_error = float(
        np.linalg.norm(
            _finite_vector(live.get("racket_pos"), 3, "C211 live racket position")
            - _finite_vector(
                teacher.get("racket_pos"), 3, "C211 teacher racket position"
            )
        )
    )
    racket_velocity_error = float(
        np.linalg.norm(
            _finite_vector(
                live.get("racket_velocity"), 3, "C211 live racket velocity"
            )
            - _finite_vector(
                teacher.get("racket_velocity"), 3, "C211 teacher racket velocity"
            )
        )
    )
    racket_normal_error = math.acos(
        float(
            np.clip(
                np.dot(
                    _unit_vector(
                        live.get("racket_normal"), "C211 live signed racket face"
                    ),
                    _unit_vector(
                        teacher.get("racket_normal"),
                        "C211 teacher signed racket face",
                    ),
                ),
                -1.0,
                1.0,
            )
        )
    )
    racket_long_axis_error = math.acos(
        float(
            np.clip(
                np.dot(
                    _unit_vector(
                        live.get("racket_long_axis"), "C211 live racket long axis"
                    ),
                    _unit_vector(
                        teacher.get("racket_long_axis"),
                        "C211 teacher racket long axis",
                    ),
                ),
                -1.0,
                1.0,
            )
        )
    )

    terms = {
        "upright_exp": _prior_term(
            raw_reward=math.exp(-upright_error_sq / (C211_UPRIGHT_STD**2)),
            manager_weight=1.0,
            details={
                "projected_gravity_xy_squared": upright_error_sq,
                "std": C211_UPRIGHT_STD,
                "formula": "exp(-sum(projected_gravity_b_xy^2)/std^2)",
            },
        ),
        "base_ang_vel_xy": _prior_term(
            raw_reward=base_ang_vel_xy_raw,
            manager_weight=-0.05,
            details={"formula": "sum(base_ang_vel_body_xy^2)"},
        ),
        "base_lin_vel_z": _prior_term(
            raw_reward=base_lin_vel_z_raw,
            manager_weight=-0.5,
            details={"formula": "base_inertial_COM_lin_vel_body_z^2"},
        ),
        "joint_vel": _prior_term(
            raw_reward=joint_vel_raw,
            manager_weight=-1.0e-4,
            details={"formula": "sum(joint_velocity^2)"},
        ),
        "action_rate_clamped": _prior_term(
            raw_reward=action_rate_raw,
            manager_weight=-0.2,
            details={
                "unclamped_raw_reward": action_rate_unclamped,
                "value_clamp": C211_ACTION_RATE_CLAMP,
                "formula": "min(sum((action-prev_action)^2),value_clamp)",
            },
        ),
        "motion_global_anchor_ori": _prior_term(
            raw_reward=math.exp(-(anchor_rotation_error**2) / (0.4**2)),
            manager_weight=0.075,
            details={
                "error_rad": anchor_rotation_error,
                "std_rad": 0.4,
                "resolved_weight": "0.5*motion_scale_0.15",
            },
        ),
        "motion_body_pos": _prior_term(
            raw_reward=math.exp(-body_position_mse / (0.3**2)),
            manager_weight=0.15,
            details={
                "mean_squared_error_m2": body_position_mse,
                "std_m": 0.3,
                "body_names": list(MIMIC_BODY_NAMES),
            },
        ),
        "motion_body_ori": _prior_term(
            raw_reward=math.exp(-body_orientation_mse / (0.4**2)),
            manager_weight=0.15,
            details={
                "mean_squared_geodesic_error_rad2": body_orientation_mse,
                "std_rad": 0.4,
                "body_names": list(MIMIC_BODY_NAMES),
            },
        ),
        "motion_body_lin_vel": _prior_term(
            raw_reward=math.exp(-body_lin_vel_mse / (1.0**2)),
            manager_weight=0.15,
            details={
                "mean_squared_error_m2ps2": body_lin_vel_mse,
                "std_mps": 1.0,
                "point": "center_of_mass",
                "body_names": list(MIMIC_BODY_NAMES),
            },
        ),
        "motion_body_ang_vel": _prior_term(
            raw_reward=math.exp(-body_ang_vel_mse / (3.14**2)),
            manager_weight=0.15,
            details={
                "mean_squared_error_rad2ps2": body_ang_vel_mse,
                "std_radps": 3.14,
                "body_names": list(MIMIC_BODY_NAMES),
            },
        ),
        "motion_racket_position": _prior_term(
            raw_reward=_cauchy_kernel(racket_position_error, 0.70),
            manager_weight=0.20,
            details={"error_m": racket_position_error, "std_m": 0.70},
        ),
        "motion_racket_velocity": _prior_term(
            raw_reward=_cauchy_kernel(racket_velocity_error, 4.0),
            manager_weight=0.20,
            details={"error_mps": racket_velocity_error, "std_mps": 4.0},
        ),
        "motion_racket_normal": _prior_term(
            raw_reward=_cauchy_kernel(racket_normal_error, math.pi),
            manager_weight=0.20,
            details={
                "signed_face_error_rad": racket_normal_error,
                "std_rad": math.pi,
            },
        ),
        "motion_racket_long_axis": _prior_term(
            raw_reward=_cauchy_kernel(racket_long_axis_error, 1.0),
            manager_weight=0.10,
            details={"error_rad": racket_long_axis_error, "std_rad": 1.0},
        ),
    }
    if tuple(terms) != C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES:
        raise C211EnvError("C211 implemented prior term order drifted")
    total = float(
        sum(float(row["post_policy_dt_reward"]) for row in terms.values())
    )
    if not math.isfinite(total):
        raise C211EnvError("C211 Isaac-synonymous prior total is non-finite")
    return {
        "terms": terms,
        "total_post_policy_dt_reward": total,
        "always_on": True,
        "task_valid_mask_applied": False,
    }


def _c211_strike_reward_terms(
    *,
    official_racket_site_w_m: Any,
    immutable_ball_contact_w_m: Any,
) -> dict[str, Any]:
    """Evaluate the post-policy-dt C211 strike bridge for one exact tick."""

    site = _finite_vector(
        official_racket_site_w_m, 3, "C211 achieved official racket site"
    )
    ball = _finite_vector(
        immutable_ball_contact_w_m, 3, "C211 immutable ball contact"
    )
    distance_m = float(np.linalg.norm(site - ball))
    kernel = 1.0 / (1.0 + (distance_m / C211_STRIKE_STD_M) ** 2)
    reward = C211_STRIKE_POST_DT_WEIGHT * kernel
    if not all(math.isfinite(value) for value in (distance_m, kernel, reward)):
        raise C211EnvError("C211 strike reward is non-finite")
    return {
        "distance_m": distance_m,
        "kernel": kernel,
        "post_policy_dt_reward": reward,
    }


def _c211_landing_reward_terms(
    *,
    landing_xy_w_m: Any,
    landing_valid: bool,
    net_crossed: bool,
    net_clear: bool,
    landing_aim_w_xy_m: Any,
    net_x_w_m: float,
    far_x_w_m: float,
    half_width_m: float,
) -> dict[str, Any]:
    """Grade one achieved-flight landing with the C211/C225 legal-base rule."""

    for name, value in (
        ("landing_valid", landing_valid),
        ("net_crossed", net_crossed),
        ("net_clear", net_clear),
    ):
        if type(value) is not bool:
            raise C211EnvError(f"C211 {name} must be bool")
    landing = _finite_vector(landing_xy_w_m, 2, "C211 achieved landing")
    aim = _finite_vector(landing_aim_w_xy_m, 2, "C211 landing aim")
    geometry = (float(net_x_w_m), float(far_x_w_m), float(half_width_m))
    if (
        not all(math.isfinite(value) for value in geometry)
        or geometry[2] <= 0.0
        or geometry[1] <= geometry[0]
    ):
        raise C211EnvError("C211 landing geometry is invalid")
    dist2_m2 = float(np.sum(np.square(landing - aim)))
    kernel = math.exp(-dist2_m2 / (C211_LANDING_SIGMA_M**2))
    opponent_plane = bool(landing_valid and landing[0] > geometry[0])
    on_opponent_table = bool(
        opponent_plane
        and landing[0] <= geometry[1]
        and abs(float(landing[1])) <= geometry[2]
    )
    achieved_gate = bool(landing_valid and net_crossed and net_clear)
    legal = bool(achieved_gate and on_opponent_table)
    opponent_side_off_table = bool(
        achieved_gate and opponent_plane and not on_opponent_table
    )
    if legal:
        raw = C211_LANDING_LEGAL_BASE_FRAC + (
            1.0 - C211_LANDING_LEGAL_BASE_FRAC
        ) * kernel
        classification = "legal_opponent_table"
    elif opponent_side_off_table:
        raw = C211_LANDING_OFF_TABLE_FRAC * kernel
        classification = "opponent_side_off_table"
    else:
        raw = 0.0
        classification = "zero_ineligible_or_nonopponent"
    reward = C211_LANDING_POST_DT_WEIGHT * raw
    if not all(math.isfinite(value) for value in (dist2_m2, kernel, raw, reward)):
        raise C211EnvError("C211 landing reward is non-finite")
    if raw < 0.0 or raw > 1.0 or (
        opponent_side_off_table and raw > C211_LANDING_OFF_TABLE_FRAC
    ):
        raise C211EnvError("C211 landing reward exceeded its bounded contract")
    return {
        "landing_xy_w_m": landing.tolist(),
        "landing_aim_w_xy_m": aim.tolist(),
        "squared_error_m2": dist2_m2,
        "kernel": kernel,
        "landing_valid": landing_valid,
        "net_crossed": net_crossed,
        "net_clear": net_clear,
        "opponent_plane": opponent_plane,
        "on_opponent_table": on_opponent_table,
        "legal_opponent_table": legal,
        "opponent_side_off_table": opponent_side_off_table,
        "classification": classification,
        "raw_reward": raw,
        "post_policy_dt_reward": reward,
    }


@dataclass(frozen=True)
class C211TaskAuthority:
    """Strict immutable incoming-question and teacher-timing authority."""

    source_path: str
    file_sha256: str
    canonical_sha256: str
    question_sha256: str
    target_recipe: str
    target_producer_sha256: str
    target_column_sha256: str
    motion_sha256: str
    physics_sha256: str
    profile_sha256: str
    action_uid: int
    base_goal_w_m: tuple[float, float, float]
    ball_contact_w_m: tuple[float, float, float]
    incoming_velocity_w_mps: tuple[float, float, float]
    incoming_spin_w_radps: tuple[float, float, float]
    landing_aim_w_xy_m: tuple[float, float]
    time_to_contact_s: float
    teacher_rate: float
    pre_swing_wait_s: float
    scaled_t_hit_s: float
    scaled_t_cycle_s: float
    reference_t_hit_s: float
    reference_t_cycle_s: float

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        expected_file_sha256: str,
        target_recipe: str = C211_TARGET_RECIPE,
    ) -> "C211TaskAuthority":
        source = Path(path).expanduser().resolve(strict=True)
        expected = _plain_sha256(expected_file_sha256, "immutable tape SHA")
        try:
            module = n1_ball_core._load_fixed_question_tape_module()
            tape = module.load_immutable_n1_tape(source, expected_file_sha256=expected)
            question = tape.question_payload
            target = tape.targets[target_recipe]
            lineage = tape.target_lineage(target_recipe)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise C211EnvError(f"immutable C211 tape is invalid: {exc}") from exc
        if target_recipe != C211_TARGET_RECIPE or tuple(target.validity_mask) != (
            False,
            False,
            False,
        ):
            raise C211EnvError(
                "C211 requires the outcome_dense_only target with validity 000"
            )
        runtime = target.runtime_target
        required_runtime = (
            "teacher_rate",
            "pre_swing_wait_s",
            "scaled_t_hit_s",
            "scaled_t_cycle_s",
            "reference_t_hit_s",
            "reference_t_cycle_s",
        )
        values = {}
        for name in required_runtime:
            value = runtime.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise C211EnvError(f"immutable C211 target omits numeric {name}")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise C211EnvError(f"immutable C211 target {name} is invalid")
            values[name] = value
        ttc = float(question["time_to_contact_s"])
        if not math.isclose(
            values["pre_swing_wait_s"] + values["scaled_t_hit_s"],
            ttc,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            values["teacher_rate"] * values["scaled_t_hit_s"],
            values["reference_t_hit_s"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise C211EnvError("immutable C211 teacher/contact clocks disagree")
        return cls(
            source_path=str(source),
            file_sha256=expected,
            canonical_sha256=_plain_sha256(tape.canonical_sha256, "tape canonical SHA"),
            question_sha256=_plain_sha256(tape.question_sha256, "tape question SHA"),
            target_recipe=target_recipe,
            target_producer_sha256=_plain_sha256(
                lineage["target_producer_sha256"], "target producer SHA"
            ),
            target_column_sha256=_plain_sha256(
                lineage["target_column_sha256"], "target column SHA"
            ),
            motion_sha256=_plain_sha256(question["motion_sha256"], "motion SHA"),
            physics_sha256=_plain_sha256(question["physics_sha256"], "physics SHA"),
            profile_sha256=_plain_sha256(question["profile_sha256"], "profile SHA"),
            action_uid=int(question["action_uid"]),
            base_goal_w_m=tuple(
                _finite_vector(question["base_goal_w_m"], 3, "base goal")
            ),
            ball_contact_w_m=tuple(
                _finite_vector(question["ball_contact_w_m"], 3, "ball contact")
            ),
            incoming_velocity_w_mps=tuple(
                _finite_vector(
                    question["incoming_velocity_w_mps"], 3, "incoming velocity"
                )
            ),
            incoming_spin_w_radps=tuple(
                _finite_vector(question["incoming_spin_w_radps"], 3, "incoming spin")
            ),
            landing_aim_w_xy_m=tuple(
                _finite_vector(question["landing_aim_w_xy_m"], 2, "landing aim")
            ),
            time_to_contact_s=ttc,
            teacher_rate=values["teacher_rate"],
            pre_swing_wait_s=values["pre_swing_wait_s"],
            scaled_t_hit_s=values["scaled_t_hit_s"],
            scaled_t_cycle_s=values["scaled_t_cycle_s"],
            reference_t_hit_s=values["reference_t_hit_s"],
            reference_t_cycle_s=values["reference_t_cycle_s"],
        )

    @property
    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "kind": C211_TASK_PROVIDER_KIND,
            "source_path": self.source_path,
            "file_sha256": self.file_sha256,
            "canonical_sha256": self.canonical_sha256,
            "question_sha256": self.question_sha256,
            "target_recipe": self.target_recipe,
            "target_producer_sha256": self.target_producer_sha256,
            "target_column_sha256": self.target_column_sha256,
            "target_validity_mask": [False, False, False],
            "motion_sha256": self.motion_sha256,
            "physics_sha256": self.physics_sha256,
            "profile_sha256": self.profile_sha256,
            "action_uid": self.action_uid,
            "task_tuple": {
                "base_goal_w_m": list(self.base_goal_w_m),
                "ball_contact_w_m": list(self.ball_contact_w_m),
                "incoming_velocity_w_mps": list(self.incoming_velocity_w_mps),
                "incoming_spin_w_radps": list(self.incoming_spin_w_radps),
                "landing_aim_w_xy_m": list(self.landing_aim_w_xy_m),
                "time_to_contact_s": self.time_to_contact_s,
            },
            "teacher_timing": {
                "teacher_rate": self.teacher_rate,
                "pre_swing_wait_s": self.pre_swing_wait_s,
                "scaled_t_hit_s": self.scaled_t_hit_s,
                "scaled_t_cycle_s": self.scaled_t_cycle_s,
                "reference_t_hit_s": self.reference_t_hit_s,
                "reference_t_cycle_s": self.reference_t_cycle_s,
            },
            "selection": "constant_row_zero_no_rng_or_cursor",
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    @property
    def content_sha256(self) -> str:
        return self.receipt["content_sha256"]


@dataclass(frozen=True)
class MeasuredC211MimicAuthority:
    """Validated measured whole-body and physical-paddle source arrays."""

    source_path: str
    file_sha256: str
    uid: str
    fps: int
    frame_count: int
    body_names: tuple[str, ...]
    anchor_index: int
    tracked_indices: tuple[int, ...]
    reference_hit_frame: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_wxyz: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    measured_racket_site_pos_w: np.ndarray
    measured_racket_normal_w: np.ndarray
    measured_racket_long_axis_w: np.ndarray
    racket_long_axis_local: np.ndarray
    joint_order_contract_sha256: str
    measured_racket_receipt_sha256: str

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        expected_file_sha256: str,
        task: C211TaskAuthority,
    ) -> "MeasuredC211MimicAuthority":
        source = Path(path).expanduser().resolve(strict=True)
        expected = _plain_sha256(expected_file_sha256, "measured motion SHA")
        if _sha256_file(source) != expected or expected != task.motion_sha256:
            raise C211EnvError(
                "measured motion bytes differ from the immutable C211 question"
            )
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
            "body_pos_point",
            "body_lin_vel_point",
            "measured_racket_site_pos_w",
            "measured_racket_normal_w",
            "measured_racket_long_axis_w",
            "measured_racket_long_axis_semantics",
            "measured_racket_robot_butt_to_blade_axis_local",
            "measured_racket_schema_version",
            "measured_racket_position_semantics",
            "measured_racket_normal_semantics",
            "measured_racket_robot_mount_normal_sign",
            "measured_racket_retarget_admitted",
            "measured_racket_uid",
            "measured_racket_joint_order_contract_sha256",
            "measured_racket_retarget_receipt_sha256",
        }
        try:
            with np.load(source, allow_pickle=False) as motion:
                missing = required - set(motion.files)
                if missing:
                    raise C211EnvError(
                        f"measured C211 motion omits {sorted(missing)!r}"
                    )
                fps_raw = np.asarray(motion["fps"])
                fps = int(fps_raw.reshape(-1)[0])
                joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64).copy()
                joint_vel = np.asarray(motion["joint_vel"], dtype=np.float64).copy()
                body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64).copy()
                body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64).copy()
                body_lin_vel = np.asarray(
                    motion["body_lin_vel_w"], dtype=np.float64
                ).copy()
                body_ang_vel = np.asarray(
                    motion["body_ang_vel_w"], dtype=np.float64
                ).copy()
                site = np.asarray(
                    motion["measured_racket_site_pos_w"], dtype=np.float64
                ).copy()
                normal = np.asarray(
                    motion["measured_racket_normal_w"], dtype=np.float64
                ).copy()
                long_axis = np.asarray(
                    motion["measured_racket_long_axis_w"], dtype=np.float64
                ).copy()
                local_long_axis = np.asarray(
                    motion["measured_racket_robot_butt_to_blade_axis_local"],
                    dtype=np.float64,
                ).copy()
                body_names = tuple(
                    str(value) for value in motion["body_names"].tolist()
                )
                scalar = lambda name: np.asarray(motion[name]).reshape(-1)[0]
                uid = str(scalar("measured_racket_uid"))
                joint_order_sha = str(
                    scalar("measured_racket_joint_order_contract_sha256")
                )
                racket_receipt_sha = str(
                    scalar("measured_racket_retarget_receipt_sha256")
                )
                metadata = {
                    "body_pos_point": str(scalar("body_pos_point")),
                    "body_lin_vel_point": str(scalar("body_lin_vel_point")),
                    "measured_racket_schema_version": int(
                        scalar("measured_racket_schema_version")
                    ),
                    "measured_racket_position_semantics": str(
                        scalar("measured_racket_position_semantics")
                    ),
                    "measured_racket_normal_semantics": str(
                        scalar("measured_racket_normal_semantics")
                    ),
                    "measured_racket_long_axis_semantics": str(
                        scalar("measured_racket_long_axis_semantics")
                    ),
                    "measured_racket_robot_mount_normal_sign": int(
                        scalar("measured_racket_robot_mount_normal_sign")
                    ),
                    "measured_racket_retarget_admitted": int(
                        scalar("measured_racket_retarget_admitted")
                    ),
                }
        except (OSError, ValueError) as exc:
            if isinstance(exc, C211EnvError):
                raise
            raise C211EnvError(f"cannot load measured C211 motion: {exc}") from exc
        frame_count = int(joint_pos.shape[0])
        if (
            fps != 50
            or joint_pos.shape != (frame_count, 31)
            or joint_vel.shape != (frame_count, 31)
            or body_pos.shape != (frame_count, len(body_names), 3)
            or body_quat.shape != (frame_count, len(body_names), 4)
            or body_lin_vel.shape != (frame_count, len(body_names), 3)
            or body_ang_vel.shape != (frame_count, len(body_names), 3)
            or site.shape != (frame_count, 3)
            or normal.shape != (frame_count, 3)
            or long_axis.shape != (frame_count, 3)
            or local_long_axis.shape != (3,)
            or frame_count < 3
            or not all(
                np.isfinite(value).all()
                for value in (
                    joint_pos,
                    joint_vel,
                    body_pos,
                    body_quat,
                    body_lin_vel,
                    body_ang_vel,
                    site,
                    normal,
                    long_axis,
                    local_long_axis,
                )
            )
        ):
            raise C211EnvError("measured C211 motion array shapes/finiteness differ")
        if metadata != {
            "body_pos_point": "link_origin",
            "body_lin_vel_point": "center_of_mass",
            "measured_racket_schema_version": 4,
            "measured_racket_position_semantics": "physical_blade_center",
            "measured_racket_normal_semantics": "signed_physical_hitting_face",
            "measured_racket_long_axis_semantics": "measured_paddle_butt_to_blade",
            "measured_racket_robot_mount_normal_sign": 1,
            "measured_racket_retarget_admitted": 1,
        }:
            raise C211EnvError("measured C211 motion semantics differ")
        if len(body_names) != len(set(body_names)):
            raise C211EnvError("measured C211 body names are not unique")
        try:
            anchor_index = body_names.index(ANCHOR_BODY_NAME)
            tracked_indices = tuple(
                body_names.index(name) for name in TRACKED_BODY_NAMES
            )
        except ValueError as exc:
            raise C211EnvError("measured C211 motion omits one tracked body") from exc
        quaternion_norms = np.linalg.norm(body_quat, axis=-1)
        normal_norms = np.linalg.norm(normal, axis=-1)
        long_axis_norms = np.linalg.norm(long_axis, axis=-1)
        if not np.allclose(
            quaternion_norms, 1.0, rtol=0.0, atol=1.0e-5
        ) or not np.allclose(
            normal_norms, 1.0, rtol=0.0, atol=1.0e-5
        ) or not np.allclose(
            long_axis_norms, 1.0, rtol=0.0, atol=1.0e-5
        ) or not np.allclose(
            local_long_axis,
            C211_RACKET_LONG_AXIS_LOCAL,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise C211EnvError(
                "measured C211 quaternion/face/long-axis rows are not authoritative"
            )
        hit_float = task.reference_t_hit_s * fps
        hit_frame = int(round(hit_float))
        if (
            not math.isclose(hit_float, float(hit_frame), rel_tol=0.0, abs_tol=1.0e-12)
            or not 0 < hit_frame < frame_count - 1
            or not math.isclose(
                task.reference_t_cycle_s,
                (frame_count - 1) / fps,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise C211EnvError("measured C211 hit/cycle frame authority differs")
        return cls(
            source_path=str(source),
            file_sha256=expected,
            uid=uid,
            fps=fps,
            frame_count=frame_count,
            body_names=body_names,
            anchor_index=anchor_index,
            tracked_indices=tracked_indices,
            reference_hit_frame=hit_frame,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_wxyz=body_quat,
            body_lin_vel_w=body_lin_vel,
            body_ang_vel_w=body_ang_vel,
            measured_racket_site_pos_w=site,
            measured_racket_normal_w=normal,
            measured_racket_long_axis_w=long_axis,
            racket_long_axis_local=local_long_axis,
            joint_order_contract_sha256=_plain_sha256(
                joint_order_sha, "motion joint-order contract SHA"
            ),
            measured_racket_receipt_sha256=_plain_sha256(
                racket_receipt_sha, "measured-racket receipt SHA"
            ),
        )

    @property
    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "kind": C211_MIMIC_PROVIDER_KIND,
            "source_path": self.source_path,
            "file_sha256": self.file_sha256,
            "uid": self.uid,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "body_names": list(self.body_names),
            "tracked_body_names": list(TRACKED_BODY_NAMES),
            "anchor_body_name": ANCHOR_BODY_NAME,
            "reference_hit_frame": self.reference_hit_frame,
            "joint_order_contract_sha256": self.joint_order_contract_sha256,
            "measured_racket_receipt_sha256": self.measured_racket_receipt_sha256,
            "joint_semantics": "q_reference_minus_plant_default_q_and_retimed_dq",
            "body_semantics": "measured_link_origin_pose_yaw_aligned_to_live_torso",
            "body_velocity_semantics": (
                "measured_COM_linear_and_world_angular_velocity_retimed_without_"
                "pose_yaw_alignment_matching_Isaac_MotionCommand"
            ),
            "body_velocity_channels_available": True,
            "measured_racket_long_axis_available": True,
            "racket_long_axis_local": self.racket_long_axis_local.tolist(),
            "racket_semantics": (
                "measured_physical_blade_center_signed_face_butt_to_blade_long_"
                "axis_central_difference_retimed_and_yaw_aligned"
            ),
            "wait_semantics": (
                "consumer_bound_split_ready_physical_birth_hidden_wait_then_"
                "atomic_measured_frame0_teacher_reference"
            ),
            "safe_ready_authority_status": SAFE_READY_AUTHORITY_STATUS,
            "safe_ready_formal_pass_claimed": False,
            "single_stroke_timeout": True,
            "timeout_bootstrap_rule": trainer.TIMEOUT_BOOTSTRAP_RULE,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    @property
    def content_sha256(self) -> str:
        return self.receipt["content_sha256"]


@dataclass
class _EpisodeObservationState:
    safe_joint_pos: np.ndarray
    safe_body_pos_w: np.ndarray
    safe_body_rotation_w: np.ndarray
    safe_anchor_pos_w: np.ndarray
    safe_anchor_rotation_w: np.ndarray
    safe_racket_pos_w: np.ndarray
    safe_racket_normal_w: np.ndarray
    safe_racket_long_axis_w: np.ndarray
    previous_action: np.ndarray


@dataclass(frozen=True)
class _FirstContactKinematics:
    """Contact-substep kinematics sealed to the native first-contact stamp."""

    policy_tick_zero_based: int
    physics_substep: int
    selected_face_normal_w: tuple[float, float, float] | None
    ball_contact_point_velocity_w_mps: tuple[float, float, float] | None
    racket_site_velocity_w_mps: tuple[float, float, float] | None
    selected_face_closing_speed_mps: float | None
    finite: bool
    classifier_binding_sha256: str
    selected_rubber_lineage_sha256: str


@dataclass
class _EpisodeRewardState:
    strike_sampled: bool = False
    outcome_evaluated: bool = False
    actual_contact_observed: bool = False
    selected_contact_observed: bool = False
    valid_achieved_flight_observed: bool = False

    def reset(self) -> None:
        self.strike_sampled = False
        self.outcome_evaluated = False
        self.actual_contact_observed = False
        self.selected_contact_observed = False
        self.valid_achieved_flight_observed = False


@dataclass(frozen=True)
class _CoreBinding:
    body_ids: tuple[int, ...]
    root_body_id: int
    anchor_body_id: int
    racket_site_id: int
    hope_world_translation_m: tuple[float, float, float]
    table_geometry_sha256: str
    selected_rubber_classifier_sha256: str
    selected_rubber_lineage_sha256: str
    plant_receipt_sha256: str


class C211ObservationProducer:
    """Construct exact C211 NumPy groups from bound live cores."""

    def __init__(
        self,
        *,
        cores: Sequence[Any],
        questions: Sequence[Any],
        robot_tape: Any,
        task: C211TaskAuthority,
        mimic: MeasuredC211MimicAuthority,
        reset_wait_steps: int | None = None,
        reset_wait_steps_by_env: Sequence[int] | None = None,
        policy_dt_s: float,
    ) -> None:
        if not cores or len(cores) != len(questions):
            raise C211EnvError("C211 producer requires one question per core")
        if reset_wait_steps_by_env is None:
            reset_wait_steps_by_env = (reset_wait_steps,) * len(cores)
        waits = tuple(reset_wait_steps_by_env)
        if len(waits) != len(cores) or any(
            type(value) is not int or value < 1 for value in waits
        ):
            raise C211EnvError("C211 producer requires positive per-env RESET_WAIT steps")
        if not math.isclose(policy_dt_s, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise C211EnvError("C211 producer requires the 0.02 s policy clock")
        self.cores = tuple(cores)
        self.questions = tuple(questions)
        self.robot_tape = robot_tape
        self.task = task
        self.mimic = mimic
        self._wait_steps_by_env = waits
        self.policy_dt_s = float(policy_dt_s)
        self._validate_robot_tape_lineage()
        self._bindings = tuple(
            self._bind_core(index, core) for index, core in enumerate(self.cores)
        )
        plant_shas = {binding.plant_receipt_sha256 for binding in self._bindings}
        if len(plant_shas) != 1:
            raise C211EnvError("C211 cores do not share one plant provider identity")
        self.plant_observation_sha256 = next(iter(plant_shas))
        self.authorities = abi.ObservationAuthorities(
            plant_observation_sha256=self.plant_observation_sha256,
            measured_mimic_sha256=mimic.content_sha256,
            task_question_sha256=task.content_sha256,
        )
        self._validate_questions()
        self._states: list[_EpisodeObservationState] = []
        self._first_contact_kinematics: list[_FirstContactKinematics | None] = [
            None for _core in self.cores
        ]
        self.reset_rows(range(len(self.cores)))
        self._reward_capture_active = False
        self._reward_capture_valid: tuple[bool, ...] = tuple()
        self._reward_capture_actions: np.ndarray | None = None
        self._reward_capture_rows: list[dict[str, Any] | None] = [
            None for _core in self.cores
        ]
        self._reward_capture_final_substeps: tuple[int, ...] = tuple()
        self._original_substep_observers: tuple[Any, ...] = tuple()
        self._install_reward_capture_hooks()

    def _install_reward_capture_hooks(self) -> None:
        """Capture the final post-physics state before compact reset mutates MjData."""

        originals: list[Any] = []
        final_substeps: list[int] = []
        for index, core in enumerate(self.cores):
            original = getattr(core, "_observe_substep", None)
            owner = getattr(core, "_c211_prior_capture_owner", None)
            try:
                decimation = int(core.binding.control_decimation)
            except (AttributeError, TypeError, ValueError) as exc:
                raise C211EnvError(
                    f"core {index} omits control-decimation authority"
                ) from exc
            if not callable(original) or owner is not None or decimation < 1:
                raise C211EnvError(
                    f"core {index} cannot install one exclusive C211 prior capture"
                )
            final_substep = decimation - 1

            def observer(
                model: Any,
                data: Any,
                substep_index: int,
                *,
                _index: int = index,
                _original: Any = original,
                _final_substep: int = final_substep,
            ) -> None:
                _original(model, data, substep_index)
                self._capture_first_contact_kinematics(_index, substep_index)
                if self._reward_capture_active and substep_index == _final_substep:
                    self._capture_reward_row(_index, substep_index)

            core._observe_substep = observer
            core._c211_prior_capture_owner = self
            originals.append(original)
            final_substeps.append(final_substep)
        self._original_substep_observers = tuple(originals)
        self._reward_capture_final_substeps = tuple(final_substeps)

    def _capture_first_contact_kinematics(
        self, index: int, substep_index: int
    ) -> None:
        """Latch velocity/normal data at the exact substep of first contact.

        The native event transcript is cumulative, while endpoint velocity is
        already post-impact.  Capturing here is therefore required for a real
        positive-closing-speed eligibility check.
        """

        if self._first_contact_kinematics[index] is not None:
            return
        core = self.cores[index]
        stamp = getattr(core, "_first_racket_contact_stamp", None)
        if not isinstance(stamp, Mapping):
            return
        policy_tick = stamp.get("policy_tick")
        stamped_substep = stamp.get("physics_substep")
        if (
            type(policy_tick) is not int
            or type(stamped_substep) is not int
            or policy_tick != getattr(core, "policy_tick", None)
            or stamped_substep != substep_index
        ):
            return

        binding = self._bindings[index]
        finite = True
        selected_normal: np.ndarray | None = None
        ball_contact_velocity: np.ndarray | None = None
        racket_velocity: np.ndarray | None = None
        closing_speed: float | None = None
        try:
            lineage = getattr(core, "_selected_rubber_action_lineage", None)
            if not isinstance(lineage, Mapping) or lineage.get(
                "mount_normal_sign"
            ) not in (-1, 1):
                raise ValueError("selected-rubber mount sign is unavailable")
            rotation = np.asarray(
                core.data.site_xmat[binding.racket_site_id], dtype=np.float64
            ).reshape(3, 3)
            selected_normal = rotation[:, 1] * int(
                lineage["mount_normal_sign"]
            )
            ball_radius = float(
                core.model.geom_size[int(core.scene.ball_geom_id), 0]
            )
            ball_center = np.asarray(
                core.data.xpos[int(core.scene.ball_body_id)], dtype=np.float64
            )
            racket_site = np.asarray(
                core.data.site_xpos[binding.racket_site_id], dtype=np.float64
            )
            ball_angular, ball_linear = self._object_spatial_velocity(
                core,
                core.mujoco.mjtObj.mjOBJ_BODY,
                int(core.scene.ball_body_id),
            )
            racket_angular, racket_site_linear = self._object_spatial_velocity(
                core,
                core.mujoco.mjtObj.mjOBJ_SITE,
                binding.racket_site_id,
            )
            selected_arm = -ball_radius * selected_normal
            ball_contact_w = ball_center + selected_arm
            racket_contact_arm = ball_contact_w - racket_site
            racket_velocity = (
                racket_site_linear
                + np.cross(racket_angular, racket_contact_arm)
            )
            ball_contact_velocity = (
                ball_linear + np.cross(ball_angular, selected_arm)
            )
            closing_speed = -float(
                np.dot(ball_contact_velocity - racket_velocity, selected_normal)
            )
            finite = bool(
                rotation.shape == (3, 3)
                and np.isfinite(rotation).all()
                and np.allclose(
                    rotation.T @ rotation,
                    np.eye(3),
                    rtol=0.0,
                    atol=1.0e-9,
                )
                and math.isclose(
                    float(np.linalg.det(rotation)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                and math.isfinite(ball_radius)
                and ball_radius > 0.0
                and ball_center.shape == (3,)
                and np.isfinite(ball_center).all()
                and racket_site.shape == (3,)
                and np.isfinite(racket_site).all()
                and selected_normal.shape == (3,)
                and np.isfinite(selected_normal).all()
                and ball_contact_velocity.shape == (3,)
                and np.isfinite(ball_contact_velocity).all()
                and racket_velocity.shape == (3,)
                and np.isfinite(racket_velocity).all()
                and math.isfinite(closing_speed)
            )
        except (AttributeError, C211EnvError, IndexError, TypeError, ValueError):
            finite = False
            selected_normal = None
            ball_contact_velocity = None
            racket_velocity = None
            closing_speed = None
        if not finite:
            selected_normal = None
            ball_contact_velocity = None
            racket_velocity = None
            closing_speed = None

        self._first_contact_kinematics[index] = _FirstContactKinematics(
            policy_tick_zero_based=policy_tick,
            physics_substep=stamped_substep,
            selected_face_normal_w=(
                None
                if selected_normal is None
                else tuple(float(value) for value in selected_normal)
            ),
            ball_contact_point_velocity_w_mps=(
                None
                if ball_contact_velocity is None
                else tuple(float(value) for value in ball_contact_velocity)
            ),
            racket_site_velocity_w_mps=(
                None
                if racket_velocity is None
                else tuple(float(value) for value in racket_velocity)
            ),
            selected_face_closing_speed_mps=closing_speed,
            finite=finite,
            classifier_binding_sha256=binding.selected_rubber_classifier_sha256,
            selected_rubber_lineage_sha256=binding.selected_rubber_lineage_sha256,
        )

    @staticmethod
    def _object_spatial_velocity(
        core: Any, object_type: Any, object_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read MuJoCo's pre-integration ``cvel`` snapshot after ``mj_step``."""

        spatial = np.zeros(6, dtype=np.float64)
        try:
            core.mujoco.mj_objectVelocity(
                core.model,
                core.data,
                object_type,
                int(object_id),
                spatial,
                0,
            )
        except Exception as exc:  # noqa: BLE001 - external backend boundary
            raise C211EnvError("MuJoCo object velocity query failed") from exc
        spatial = _finite_vector(spatial, 6, "MuJoCo object spatial velocity")
        # MuJoCo spatial vectors are rotation first, translation second.
        return spatial[:3].copy(), spatial[3:].copy()

    def _capture_reward_row(self, index: int, substep_index: int) -> None:
        if (
            not self._reward_capture_active
            or self._reward_capture_actions is None
            or len(self._reward_capture_valid) != len(self.cores)
        ):
            raise C211EnvError("C211 prior capture fired outside one transition")
        live = self._live(index)
        teacher, _teacher_hit = self._teacher(
            index, live, self._reward_capture_valid[index]
        )
        prior = _c211_isaac_synonymous_prior_terms(
            live=live,
            teacher=teacher,
            current_action=self._reward_capture_actions[index],
            previous_action=self._states[index].previous_action,
        )
        self._reward_capture_rows[index] = {
            "task_valid": self._reward_capture_valid[index],
            "sample_time_s": float(self.cores[index].data.time),
            "physics_substep": int(substep_index),
            **prior,
        }

    def begin_reward_transition(
        self, task_valid: Sequence[bool], actions: Any
    ) -> None:
        valid = tuple(task_valid)
        action_rows = np.asarray(actions, dtype=np.float64)
        if (
            self._reward_capture_active
            or len(valid) != len(self.cores)
            or any(type(value) is not bool for value in valid)
            or action_rows.shape != (len(self.cores), 31)
            or not np.isfinite(action_rows).all()
        ):
            raise C211EnvError("C211 prior transition capture cannot be armed")
        self._reward_capture_active = True
        self._reward_capture_valid = valid
        self._reward_capture_actions = action_rows.copy()
        self._reward_capture_rows = [None for _core in self.cores]

    def abort_reward_transition(self) -> None:
        self._reward_capture_active = False
        self._reward_capture_valid = tuple()
        self._reward_capture_actions = None
        self._reward_capture_rows = [None for _core in self.cores]

    def finish_reward_transition(self) -> tuple[dict[str, Any], ...]:
        if not self._reward_capture_active:
            raise C211EnvError("C211 prior transition capture is not armed")
        rows = tuple(self._reward_capture_rows)
        self._reward_capture_active = False
        self._reward_capture_valid = tuple()
        self._reward_capture_actions = None
        self._reward_capture_rows = [None for _core in self.cores]
        if any(row is None for row in rows):
            raise C211EnvError("C211 final-substep prior capture is incomplete")
        typed_rows = tuple(row for row in rows if row is not None)
        if any(
            row.get("physics_substep") != expected
            for row, expected in zip(
                typed_rows, self._reward_capture_final_substeps
            )
        ):
            raise C211EnvError("C211 prior capture did not sample the final substep")
        return typed_rows

    def _validate_robot_tape_lineage(self) -> None:
        reset = getattr(self.robot_tape, "reset_state", None)
        mode = getattr(reset, "mode", None)
        if reset is None or mode != "action_specific_hold":
            raise C211EnvError(
                "C211 RESET_WAIT requires the split-ready action-specific "
                "stationary physical birth"
            )
        identity = (
            getattr(reset, "hold_candidate_kind", None),
            getattr(reset, "hold_candidate_schema_version", None),
        )
        if identity != (
            single_env.ACTION_SPECIFIC_HOLD_KIND,
            single_env.ACTION_SPECIFIC_HOLD_SCHEMA_VERSION,
        ):
            raise C211EnvError(
                "C211 RESET_WAIT rejects teacher-frame, legacy exact-frame0, "
                "and unknown hold identities; only the split-ready "
                "action-specific hold is active"
            )
        if (
            getattr(reset, "source_motion_sha256", None)
            != self.task.motion_sha256
            or getattr(reset, "source_motion_sha256", None) != self.mimic.file_sha256
            or getattr(reset, "source_motion_uid", None) != self.mimic.uid
            or getattr(reset, "source_frame_index", None) != 0
            or getattr(reset, "source_joint_order_contract_sha256", None)
            != self.mimic.joint_order_contract_sha256
        ):
            raise C211EnvError(
                "robot-tape split-ready birth, measured teacher, and task motion lineage differ"
            )
        try:
            reopened_candidate = single_env._revalidate_action_specific_reset_state(
                self.cores[0].binding,
                reset,
                self.robot_tape.history_fill_action,
            )
        except (single_env.ContractError, AttributeError, TypeError, ValueError) as exc:
            raise C211EnvError(
                "robot-tape split-ready artifact cannot be reopened and "
                "revalidated from its exact sources"
            ) from exc
        candidate_identity = (
            reopened_candidate.get("kind"),
            reopened_candidate.get("schema_version"),
        )
        semantics = reopened_candidate.get("semantics")
        static_evidence = reopened_candidate.get("static_evidence")
        static_gates = (
            static_evidence.get("gates")
            if isinstance(static_evidence, Mapping)
            else None
        )
        receipt_sha = (
            static_evidence.get("grounded_ready_receipt_sha256")
            if isinstance(static_evidence, Mapping)
            else None
        )
        content_sha = reopened_candidate.get("content_sha256")
        if (
            candidate_identity
            != (
                single_env.ACTION_SPECIFIC_HOLD_KIND,
                single_env.ACTION_SPECIFIC_HOLD_SCHEMA_VERSION,
            )
            or content_sha
            != getattr(reset, "hold_candidate_content_sha256", None)
            or not isinstance(semantics, Mapping)
            or semantics.get("physical_reset")
            != "shared_grounded_lower_root_plus_teacher_nonleg"
            or semantics.get("teacher_reference_unchanged") is not True
            or semantics.get("teacher_and_physical_reset_may_differ") is not True
            or not isinstance(static_gates, Mapping)
            or not static_gates
            or any(value != "PASS" for value in static_gates.values())
            or not isinstance(receipt_sha, str)
            or len(receipt_sha) != 64
            or any(character not in "0123456789abcdef" for character in receipt_sha)
        ):
            raise C211EnvError(
                "reopened hold cannot be relabeled as the active split-ready "
                "artifact/receipt identity"
            )
        try:
            reset_joint_pos = _finite_vector(
                reset.joint_pos, 31, "physical birth joint position"
            )
            reset_joint_vel = _finite_vector(
                reset.joint_vel, 31, "physical birth joint velocity"
            )
            reset_root_pos = _finite_vector(
                reset.root_pos, 3, "physical birth root position"
            )
            reset_root_quat = _finite_vector(
                reset.root_quat_wxyz,
                4,
                "physical birth root quaternion",
            )
            reset_root_lin = _finite_vector(
                reset.root_lin_vel_w,
                3,
                "physical birth root linear velocity",
            )
            reset_root_ang = _finite_vector(
                reset.root_ang_vel_w,
                3,
                "physical birth root angular velocity",
            )
        except (AttributeError, ValueError) as exc:
            raise C211EnvError(
                "stationary physical birth reset payload is incomplete"
            ) from exc
        if (
            not np.array_equal(reset_joint_vel, np.zeros(31))
            or not np.array_equal(reset_root_lin, np.zeros(3))
            or float(np.max(np.abs(reset_root_ang))) > 1.0e-14
        ):
            raise C211EnvError(
                "physical birth is not stationary within its angular roundoff allowance"
            )
        self.reset_birth_semantics = (
            "split_ready_physical_safe_birth_separate_from_measured_"
            "teacher_frame0_stationary_hidden_wait"
        )
        source_path = getattr(reset, "source_motion_path", None)
        if not isinstance(source_path, str):
            raise C211EnvError(
                "robot-tape birth omits its measured-motion path"
            )
        try:
            same_path = Path(source_path).expanduser().resolve(strict=True) == Path(
                self.mimic.source_path
            ).resolve(strict=True)
        except OSError as exc:
            raise C211EnvError(
                "robot-tape measured-motion source cannot be reopened"
            ) from exc
        if not same_path:
            raise C211EnvError(
                "robot-tape birth and measured teacher paths differ"
            )

    @staticmethod
    def _hope_translation_from_geometry(
        core: Any, index: int
    ) -> tuple[np.ndarray, str]:
        contract = getattr(core.plant, "geometry_contract", None)
        if not isinstance(contract, Mapping):
            raise C211EnvError(f"core {index} omits the table geometry contract")
        supplied_sha = _plain_sha256(
            contract.get("sha256"), "table geometry contract SHA"
        )
        payload = contract.get("payload")
        if (
            not isinstance(payload, Mapping)
            or _sha256_json(payload) != supplied_sha
            or payload.get("primitive") != "axis_aligned_box_full_extents_m"
        ):
            raise C211EnvError(f"core {index} table geometry seal differs")
        obstacles = payload.get("obstacles")
        if not isinstance(obstacles, list):
            raise C211EnvError(f"core {index} table geometry omits obstacles")
        top_rows = [
            row
            for row in obstacles
            if isinstance(row, Mapping)
            and row.get("role") == "top"
            and row.get("name") == "motion_table_top"
        ]
        if len(top_rows) != 1:
            raise C211EnvError(f"core {index} table geometry has no unique top")
        center = _finite_vector(
            top_rows[0].get("center_mjcf_world_m"), 3, "table-top center"
        )
        extents = _finite_vector(
            top_rows[0].get("full_extents_m"), 3, "table-top extents"
        )
        if np.any(extents <= 0.0) or not math.isclose(
            center[1], 0.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise C211EnvError(f"core {index} table geometry frame differs")
        translation = np.asarray(
            (
                center[0] - 0.5 * extents[0],
                0.5 * extents[1],
                center[2] + 0.5 * extents[2],
            ),
            dtype=np.float64,
        )
        scene_near = float(getattr(core.scene, "near_x", float("nan")))
        scene_surface = float(getattr(core.scene, "surface_z", float("nan")))
        if not np.allclose(
            translation[[0, 2]],
            np.asarray((scene_near, scene_surface), dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise C211EnvError(f"core {index} table pose and geometry differ")
        return translation, supplied_sha

    def _bind_core(self, index: int, core: Any) -> _CoreBinding:
        required = ("mujoco", "model", "data", "plant", "binding", "scene")
        missing = [name for name in required if not hasattr(core, name)]
        if missing:
            raise C211EnvError(f"core {index} omits live plant fields {missing!r}")
        mujoco = core.mujoco
        try:
            object_type = mujoco.mjtObj.mjOBJ_BODY
            body_ids = tuple(
                int(mujoco.mj_name2id(core.model, object_type, name))
                for name in TRACKED_BODY_NAMES
            )
            root_id = int(mujoco.mj_name2id(core.model, object_type, ROOT_BODY_NAME))
            anchor_id = int(
                mujoco.mj_name2id(core.model, object_type, ANCHOR_BODY_NAME)
            )
        except Exception as exc:  # noqa: BLE001 - external backend boundary
            raise C211EnvError(f"core {index} cannot resolve tracked bodies") from exc
        if any(value < 0 for value in (*body_ids, root_id, anchor_id)) or len(
            set(body_ids)
        ) != len(body_ids):
            raise C211EnvError(f"core {index} tracked body mapping differs")
        racket_id = int(getattr(core, "_racket_site_id", -1))
        if racket_id < 0:
            raise C211EnvError(f"core {index} has no official racket site")
        binding = core.binding
        plant = core.plant
        try:
            contract_path = Path(binding.source_path).expanduser().resolve(strict=True)
        except (AttributeError, OSError, TypeError) as exc:
            raise C211EnvError(
                f"core {index} plant contract source is unavailable"
            ) from exc
        contract_file_sha = _sha256_file(contract_path)
        expected_contract_sha = _plain_sha256(
            getattr(binding, "source_sha256", None), "plant contract source SHA"
        )
        if contract_file_sha != expected_contract_sha:
            raise C211EnvError(
                f"core {index} plant contract bytes differ from the bound authority"
            )
        default_q = np.asarray(binding.default_joint_pos, dtype=np.float64)
        joint_names = tuple(getattr(binding, "joint_names", ()))
        if (
            default_q.shape != (31,)
            or len(joint_names) != 31
            or len(set(joint_names)) != 31
            or tuple(getattr(self.robot_tape, "joint_names", ())) != joint_names
        ):
            raise C211EnvError(f"core {index} plant joint authority differs")
        hope_translation, geometry_sha = self._hope_translation_from_geometry(
            core, index
        )
        site_binding = selected_rubber_classifier.validate_classifier_binding(
            core.selected_rubber_classifier_binding
        )
        site_lineage = selected_rubber_classifier.validate_action_lineage(
            core._selected_rubber_action_lineage,
            classifier_binding=site_binding,
        )
        provider = {
            "schema_version": 1,
            "kind": C211_PLANT_PROVIDER_KIND,
            "plant_binding_sha256": _plain_sha256(
                binding.binding_sha256, "plant binding SHA"
            ),
            "plant_contract_file_sha256": contract_file_sha,
            "scene_binding_sha256": _plain_sha256(
                core.scene_binding_sha256, "scene binding SHA"
            ),
            "root_mjcf_sha256": _plain_sha256(
                core.scene.canonical_xml_sha256, "MJCF SHA"
            ),
            "selected_rubber_classifier_sha256": _plain_sha256(
                site_binding["content_sha256"], "classifier SHA"
            ),
            "joint_names": list(joint_names),
            "tracked_body_names": list(TRACKED_BODY_NAMES),
            "tracked_body_ids": list(body_ids),
            "root_body_name": ROOT_BODY_NAME,
            "anchor_body_name": ANCHOR_BODY_NAME,
            "official_racket_site_name": selected_rubber_classifier.RACKET_SITE_NAME,
            "official_racket_site_id": racket_id,
            "hope_world_translation_m": hope_translation.tolist(),
            "table_geometry_contract_sha256": geometry_sha,
            "base_velocity_semantics": "root_inertial_com_point_velocity_world",
            "tracked_body_velocity_semantics": (
                "mj_jacBodyCom_linear_and_angular_velocity_world"
            ),
            "racket_velocity_semantics": "official_site_point_velocity_world",
            "racket_long_axis_semantics": (
                "official_site_rotation_times_measured_authority_local_"
                "butt_to_blade_axis"
            ),
            "rotation6d_order": "R00_R01_R10_R11_R20_R21",
            "previous_action_reset_semantics": "robot_tape_history_fill_action",
            "physical_birth_reset_semantics": self.reset_birth_semantics,
            "teacher_reference_semantics": (
                "independent_measured_frame0_revealed_after_hidden_wait"
            ),
            "producer_source_sha256": _sha256_file(Path(__file__).resolve()),
            "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        provider["content_sha256"] = _sha256_json(provider)
        if np.asarray(plant.qpos_addr).shape != (31,) or np.asarray(
            plant.dof_addr
        ).shape != (31,):
            raise C211EnvError(f"core {index} plant q/dq address ABI differs")
        return _CoreBinding(
            body_ids=body_ids,
            root_body_id=root_id,
            anchor_body_id=anchor_id,
            racket_site_id=racket_id,
            hope_world_translation_m=tuple(float(value) for value in hope_translation),
            table_geometry_sha256=geometry_sha,
            selected_rubber_classifier_sha256=_plain_sha256(
                site_binding["content_sha256"], "classifier SHA"
            ),
            selected_rubber_lineage_sha256=_plain_sha256(
                site_lineage["content_sha256"], "selected-rubber lineage SHA"
            ),
            plant_receipt_sha256=provider["content_sha256"],
        )

    def _validate_questions(self) -> None:
        for index, (question, wait_steps) in enumerate(
            zip(self.questions, self._wait_steps_by_env)
        ):
            authority = getattr(question, "authority", None)
            if not isinstance(authority, Mapping):
                raise C211EnvError(f"question {index} has no immutable authority")
            if (
                authority.get("immutable_n1_tape_bound") is not True
                or authority.get("immutable_tape_file_sha256") != self.task.file_sha256
                or authority.get("immutable_tape_canonical_sha256")
                != self.task.canonical_sha256
                or authority.get("base_question_sha256") != self.task.question_sha256
                or authority.get("target_recipe") != self.task.target_recipe
                or authority.get("target_producer_sha256")
                != self.task.target_producer_sha256
                or authority.get("target_column_sha256")
                != self.task.target_column_sha256
                or authority.get("motion_sha256") != self.task.motion_sha256
                or authority.get("physics_sha256") != self.task.physics_sha256
                or authority.get("profile_sha256") != self.task.profile_sha256
                or int(authority.get("action_uid", -1)) != self.task.action_uid
            ):
                raise C211EnvError(f"question {index} differs from C211 task authority")
            if not np.array_equal(
                _finite_vector(
                    question.landing_aim_xy_w_m,
                    2,
                    f"question {index} landing aim",
                ),
                np.asarray(self.task.landing_aim_w_xy_m, dtype=np.float64),
            ):
                raise C211EnvError(
                    f"question {index} landing aim differs from C211 task authority"
                )
            expected_ttc = self.task.time_to_contact_s + wait_steps * self.policy_dt_s
            if not math.isclose(
                float(question.nominal_time_to_contact_s),
                expected_ttc,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise C211EnvError(f"question {index} RESET_WAIT clock differs")

    def set_episode_questions(
        self, questions: Sequence[Any], wait_steps_by_env: Sequence[int]
    ) -> None:
        rows = tuple(questions)
        waits = tuple(wait_steps_by_env)
        if len(rows) != len(self.cores) or len(waits) != len(self.cores) or any(
            type(value) is not int or value < 1 for value in waits
        ):
            raise C211EnvError("C211 episode question/wait cardinality differs")
        self.questions = rows
        self._wait_steps_by_env = waits
        self._validate_questions()

    def _site_point_velocity(self, core: Any, site_id: int) -> np.ndarray:
        jacobian_position = np.zeros((3, int(core.model.nv)), dtype=np.float64)
        jacobian_rotation = np.zeros((3, int(core.model.nv)), dtype=np.float64)
        try:
            core.mujoco.mj_jacSite(
                core.model,
                core.data,
                jacobian_position,
                jacobian_rotation,
                int(site_id),
            )
        except Exception as exc:  # noqa: BLE001 - external backend boundary
            raise C211EnvError("MuJoCo site Jacobian query failed") from exc
        velocity = jacobian_position @ np.asarray(core.data.qvel, dtype=np.float64)
        return _finite_vector(velocity, 3, "racket site point velocity")

    def _body_com_velocities(
        self, core: Any, body_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        qvel = np.asarray(core.data.qvel, dtype=np.float64)
        if qvel.shape != (int(core.model.nv),) or not np.isfinite(qvel).all():
            raise C211EnvError("MuJoCo generalized velocity ABI differs")
        linear_rows: list[np.ndarray] = []
        angular_rows: list[np.ndarray] = []
        for row, body_id in enumerate(body_ids):
            jacobian_position = np.zeros(
                (3, int(core.model.nv)), dtype=np.float64
            )
            jacobian_rotation = np.zeros(
                (3, int(core.model.nv)), dtype=np.float64
            )
            try:
                core.mujoco.mj_jacBodyCom(
                    core.model,
                    core.data,
                    jacobian_position,
                    jacobian_rotation,
                    int(body_id),
                )
            except Exception as exc:  # noqa: BLE001 - external backend boundary
                raise C211EnvError(
                    f"MuJoCo body-COM Jacobian query failed for row {row}"
                ) from exc
            linear_rows.append(jacobian_position @ qvel)
            angular_rows.append(jacobian_rotation @ qvel)
        return (
            _finite_array(
                np.stack(linear_rows, axis=0),
                (len(body_ids), 3),
                "tracked body COM linear velocities",
            ),
            _finite_array(
                np.stack(angular_rows, axis=0),
                (len(body_ids), 3),
                "tracked body angular velocities",
            ),
        )

    def _live(self, index: int) -> dict[str, np.ndarray]:
        core, binding = self.cores[index], self._bindings[index]
        data, plant = core.data, core.plant
        root_pos = _finite_vector(data.xpos[binding.root_body_id], 3, "root position")
        root_rotation = _finite_array(
            np.asarray(data.xmat[binding.root_body_id]).reshape(3, 3),
            (3, 3),
            "root rotation",
        )
        root_ang_body = _finite_vector(data.qvel[3:6], 3, "root body angular velocity")
        root_ang_w = root_rotation @ root_ang_body
        root_origin_lin_w = _finite_vector(
            data.qvel[0:3], 3, "root origin linear velocity"
        )
        root_to_com_w = root_rotation @ _finite_vector(
            core.model.body_ipos[binding.root_body_id],
            3,
            "root local inertial COM",
        )
        root_lin_w = root_origin_lin_w + np.cross(root_ang_w, root_to_com_w)
        body_pos = _finite_array(
            np.asarray(data.xpos)[list(binding.body_ids)],
            (len(TRACKED_BODY_NAMES), 3),
            "tracked body positions",
        )
        body_rotation = np.stack(
            [
                _finite_array(
                    np.asarray(data.xmat[body_id]).reshape(3, 3),
                    (3, 3),
                    f"tracked body rotation {row}",
                )
                for row, body_id in enumerate(binding.body_ids)
            ],
            axis=0,
        )
        body_lin_vel_w, body_ang_vel_w = self._body_com_velocities(
            core, binding.body_ids
        )
        anchor_row = TRACKED_BODY_NAMES.index(ANCHOR_BODY_NAME)
        racket_pos = _finite_vector(
            data.site_xpos[binding.racket_site_id], 3, "racket site position"
        )
        racket_rotation = _finite_array(
            np.asarray(data.site_xmat[binding.racket_site_id]).reshape(3, 3),
            (3, 3),
            "racket site rotation",
        )
        racket_velocity = self._site_point_velocity(core, binding.racket_site_id)
        lineage = getattr(core, "_selected_rubber_action_lineage", None)
        if not isinstance(lineage, Mapping) or lineage.get("mount_normal_sign") not in (
            -1,
            1,
        ):
            raise C211EnvError("live core has no selected-rubber face sign")
        racket_normal = racket_rotation[:, 1] * int(lineage["mount_normal_sign"])
        racket_normal /= np.linalg.norm(racket_normal)
        racket_raw_y_axis = racket_rotation[:, 1].copy()
        racket_raw_y_axis /= np.linalg.norm(racket_raw_y_axis)
        racket_long_axis = racket_rotation @ self.mimic.racket_long_axis_local
        racket_long_axis /= np.linalg.norm(racket_long_axis)
        q = _finite_vector(data.qpos[plant.qpos_addr], 31, "joint positions")
        qd = _finite_vector(data.qvel[plant.dof_addr], 31, "joint velocities")
        return {
            "root_pos": root_pos,
            "root_rotation": root_rotation,
            "root_lin_vel_w": root_lin_w,
            "root_ang_vel_w": root_ang_w,
            "q": q,
            "qd": qd,
            "body_pos": body_pos,
            "body_rotation": body_rotation,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "anchor_pos": body_pos[anchor_row].copy(),
            "anchor_rotation": body_rotation[anchor_row].copy(),
            "racket_pos": racket_pos,
            "racket_velocity": racket_velocity,
            "racket_normal": racket_normal,
            "racket_raw_y_axis": racket_raw_y_axis,
            "racket_long_axis": racket_long_axis,
            "hope_world_translation": np.asarray(
                binding.hope_world_translation_m, dtype=np.float64
            ),
        }

    def reset_rows(self, env_ids: Sequence[int]) -> None:
        ids = tuple(int(value) for value in env_ids)
        if any(value < 0 or value >= len(self.cores) for value in ids):
            raise C211EnvError("C211 reset row index is invalid")
        while len(self._states) < len(self.cores):
            self._states.append(None)  # type: ignore[arg-type]
        history = _finite_vector(
            self.robot_tape.history_fill_action, 31, "history fill action"
        )
        for index in ids:
            live = self._live(index)
            self._states[index] = _EpisodeObservationState(
                safe_joint_pos=live["q"].copy(),
                safe_body_pos_w=live["body_pos"].copy(),
                safe_body_rotation_w=live["body_rotation"].copy(),
                safe_anchor_pos_w=live["anchor_pos"].copy(),
                safe_anchor_rotation_w=live["anchor_rotation"].copy(),
                safe_racket_pos_w=live["racket_pos"].copy(),
                safe_racket_normal_w=live["racket_normal"].copy(),
                safe_racket_long_axis_w=live["racket_long_axis"].copy(),
                previous_action=history.copy(),
            )
            self._first_contact_kinematics[index] = None

    def contact_eligibility(
        self,
        index: int,
        contact: n1_reward_event_kernel.ContactEvidence,
        *,
        nominal_strike_tick_1based: int,
    ) -> dict[str, Any]:
        """Bind selected contact to the nominal transition and pre-impact speed."""

        if (
            type(index) is not int
            or not 0 <= index < len(self.cores)
            or type(contact) is not n1_reward_event_kernel.ContactEvidence
            or type(nominal_strike_tick_1based) is not int
            or nominal_strike_tick_1based < 1
        ):
            raise C211EnvError("C211 contact eligibility arguments differ")
        captured = self._first_contact_kinematics[index]
        if not contact.occurred:
            if contact.stamp is not None or captured is not None:
                raise C211EnvError(
                    f"env {index} no-contact evidence carries contact kinematics"
                )
            return {
                "eligible": False,
                "reason": "no_actual_contact",
                "exact_nominal_contact_transition": False,
                "selected_rubber": False,
                "finite_contact_kinematics": False,
                "positive_closing_speed": False,
                "selected_face_closing_speed_mps": None,
                "contact_policy_tick_zero_based": None,
                "contact_transition_tick_1based": None,
                "physics_substep": None,
            }
        if contact.stamp is None:
            raise C211EnvError(f"env {index} contact evidence omits its stamp")
        if captured is None:
            # A missing substep capture cannot be reconstructed from the
            # post-impact endpoint.  It is a fail-closed reward row, not a
            # license to infer a favorable closing speed.
            return {
                "eligible": False,
                "reason": "contact_substep_kinematics_unavailable",
                "exact_nominal_contact_transition": False,
                "selected_rubber": contact.selected_rubber,
                "finite_contact_kinematics": False,
                "positive_closing_speed": False,
                "selected_face_closing_speed_mps": None,
                "contact_policy_tick_zero_based": contact.stamp.policy_tick,
                "contact_transition_tick_1based": contact.stamp.policy_tick + 1,
                "physics_substep": contact.stamp.physics_substep,
            }
        binding = self._bindings[index]
        if (
            captured.policy_tick_zero_based != contact.stamp.policy_tick
            or captured.physics_substep != contact.stamp.physics_substep
            or captured.classifier_binding_sha256
            != binding.selected_rubber_classifier_sha256
            or captured.selected_rubber_lineage_sha256
            != binding.selected_rubber_lineage_sha256
        ):
            raise C211EnvError(
                f"env {index} contact kinematics differ from sealed event evidence"
            )
        transition_tick = captured.policy_tick_zero_based + 1
        exact = transition_tick == nominal_strike_tick_1based
        positive = bool(
            captured.finite
            and captured.selected_face_closing_speed_mps is not None
            and captured.selected_face_closing_speed_mps > 0.0
        )
        eligible = bool(contact.selected_rubber and captured.finite and positive and exact)
        if not contact.selected_rubber:
            reason = "first_contact_not_selected_rubber"
        elif not exact:
            reason = "contact_not_on_nominal_transition"
        elif not captured.finite:
            reason = "nonfinite_contact_kinematics"
        elif not positive:
            reason = "nonpositive_selected_face_closing_speed"
        else:
            reason = "eligible_exact_selected_positive_closing_contact"
        return {
            "eligible": eligible,
            "reason": reason,
            "exact_nominal_contact_transition": exact,
            "selected_rubber": contact.selected_rubber,
            "finite_contact_kinematics": captured.finite,
            "positive_closing_speed": positive,
            "selected_face_closing_speed_mps": (
                captured.selected_face_closing_speed_mps
            ),
            "selected_face_normal_w": captured.selected_face_normal_w,
            "ball_contact_point_velocity_w_mps": (
                captured.ball_contact_point_velocity_w_mps
            ),
            "racket_site_velocity_w_mps": captured.racket_site_velocity_w_mps,
            "contact_policy_tick_zero_based": captured.policy_tick_zero_based,
            "contact_transition_tick_1based": transition_tick,
            "physics_substep": captured.physics_substep,
            "classifier_binding_sha256": captured.classifier_binding_sha256,
            "selected_rubber_lineage_sha256": (
                captured.selected_rubber_lineage_sha256
            ),
        }

    def set_previous_actions(self, actions: Any, reset_env_ids: Sequence[int]) -> None:
        rows = np.asarray(actions, dtype=np.float64)
        if rows.shape != (len(self.cores), 31) or not np.isfinite(rows).all():
            raise C211EnvError("C211 previous actions must be finite [N,31]")
        reset = set(int(value) for value in reset_env_ids)
        history = _finite_vector(
            self.robot_tape.history_fill_action, 31, "history fill action"
        )
        for index, state in enumerate(self._states):
            state.previous_action = (
                history.copy() if index in reset else rows[index].copy()
            )

    def _aligned_teacher(
        self,
        *,
        live: Mapping[str, np.ndarray],
        frame: int,
        held: bool,
    ) -> dict[str, np.ndarray]:
        mimic = self.mimic
        raw_anchor_pos = mimic.body_pos_w[frame, mimic.anchor_index]
        raw_anchor_rotation = _rotation_from_wxyz(
            mimic.body_quat_wxyz[frame, mimic.anchor_index], "teacher anchor quaternion"
        )
        delta_rotation = _yaw_rotation(live["anchor_rotation"] @ raw_anchor_rotation.T)
        delta_pos = np.asarray(live["anchor_pos"], dtype=np.float64).copy()
        delta_pos[2] = raw_anchor_pos[2]
        raw_body_pos = mimic.body_pos_w[frame, list(mimic.tracked_indices)]
        raw_body_rotation = np.stack(
            [
                _rotation_from_wxyz(
                    mimic.body_quat_wxyz[frame, body_index],
                    f"teacher body quaternion {row}",
                )
                for row, body_index in enumerate(mimic.tracked_indices)
            ],
            axis=0,
        )
        body_pos = delta_pos + (delta_rotation @ (raw_body_pos - raw_anchor_pos).T).T
        body_rotation = delta_rotation[None, :, :] @ raw_body_rotation

        previous = max(frame - 1, 0)
        following = min(frame + 1, mimic.frame_count - 1)
        if following <= previous:
            raise C211EnvError("teacher central-difference frame span is empty")

        def aligned_site(step: int) -> np.ndarray:
            return delta_pos + delta_rotation @ (
                mimic.measured_racket_site_pos_w[step] - raw_anchor_pos
            )

        site_pos = aligned_site(frame)
        site_normal = delta_rotation @ mimic.measured_racket_normal_w[frame]
        site_normal /= np.linalg.norm(site_normal)
        site_long_axis = delta_rotation @ mimic.measured_racket_long_axis_w[frame]
        site_long_axis /= np.linalg.norm(site_long_axis)
        if held:
            site_velocity = np.zeros(3, dtype=np.float64)
        else:
            span_s = (following - previous) / (mimic.fps * self.task.teacher_rate)
            site_velocity = (aligned_site(following) - aligned_site(previous)) / span_s
        return {
            "joint_pos": mimic.joint_pos[frame].copy(),
            "joint_vel": (
                np.zeros(31, dtype=np.float64)
                if held
                else mimic.joint_vel[frame] * self.task.teacher_rate
            ),
            "body_pos": body_pos,
            "body_rotation": body_rotation,
            "body_lin_vel_w": (
                np.zeros((len(TRACKED_BODY_NAMES), 3), dtype=np.float64)
                if held
                else mimic.body_lin_vel_w[
                    frame, list(mimic.tracked_indices)
                ]
                * self.task.teacher_rate
            ),
            "body_ang_vel_w": (
                np.zeros((len(TRACKED_BODY_NAMES), 3), dtype=np.float64)
                if held
                else mimic.body_ang_vel_w[
                    frame, list(mimic.tracked_indices)
                ]
                * self.task.teacher_rate
            ),
            "anchor_pos": body_pos[TRACKED_BODY_NAMES.index(ANCHOR_BODY_NAME)].copy(),
            "anchor_rotation": body_rotation[
                TRACKED_BODY_NAMES.index(ANCHOR_BODY_NAME)
            ].copy(),
            "global_anchor_rotation": raw_anchor_rotation.copy(),
            "racket_pos": site_pos,
            "racket_velocity": site_velocity,
            "racket_normal": site_normal,
            "racket_long_axis": site_long_axis,
        }

    def _safe_teacher(self, index: int) -> dict[str, np.ndarray]:
        state = self._states[index]
        return {
            "joint_pos": state.safe_joint_pos.copy(),
            "joint_vel": np.zeros(31, dtype=np.float64),
            "body_pos": state.safe_body_pos_w.copy(),
            "body_rotation": state.safe_body_rotation_w.copy(),
            "body_lin_vel_w": np.zeros(
                (len(TRACKED_BODY_NAMES), 3), dtype=np.float64
            ),
            "body_ang_vel_w": np.zeros(
                (len(TRACKED_BODY_NAMES), 3), dtype=np.float64
            ),
            "anchor_pos": state.safe_anchor_pos_w.copy(),
            "anchor_rotation": state.safe_anchor_rotation_w.copy(),
            "global_anchor_rotation": state.safe_anchor_rotation_w.copy(),
            "racket_pos": state.safe_racket_pos_w.copy(),
            "racket_velocity": np.zeros(3, dtype=np.float64),
            "racket_normal": state.safe_racket_normal_w.copy(),
            "racket_long_axis": state.safe_racket_long_axis_w.copy(),
        }

    def _teacher(
        self, index: int, live: Mapping[str, np.ndarray], valid: bool
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not valid:
            safe = self._safe_teacher(index)
            return safe, safe
        elapsed = float(self.cores[index].data.time)
        active_motion_s = max(
            elapsed
            - self._wait_steps_by_env[index] * self.policy_dt_s
            - self.task.pre_swing_wait_s,
            0.0,
        )
        phase = active_motion_s * self.task.teacher_rate * self.mimic.fps
        frame = min(int(np.rint(phase)), self.mimic.frame_count - 1)
        held = active_motion_s <= 1.0e-12
        current = self._aligned_teacher(live=live, frame=frame, held=held)
        # The live Isaac producer uses the current Motion hold state when it
        # differentiates the explicit reference-hit row.  Match that detail:
        # pre-swing hit velocity is literal zero, then becomes retimed.
        hit = self._aligned_teacher(
            live=live, frame=self.mimic.reference_hit_frame, held=held
        )
        return current, hit

    @staticmethod
    def _heading_racket(
        live: Mapping[str, np.ndarray], racket: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        heading = _yaw_rotation(live["root_rotation"])
        return np.concatenate(
            (
                heading.T @ (racket["racket_pos"] - live["root_pos"]),
                heading.T @ racket["racket_velocity"],
                heading.T @ racket["racket_normal"],
            )
        )

    def groups(
        self, task_valid: Sequence[bool]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        valid = np.asarray(task_valid, dtype=np.bool_)
        if valid.shape != (len(self.cores),):
            raise C211EnvError("C211 task_valid batch shape differs")
        actor_rows: dict[str, list[np.ndarray]] = {
            name: [] for name, _width in abi.C211_PROFILE.actor.layout
        }
        critic_rows: dict[str, list[np.ndarray]] = {
            name: [] for name, _width in abi.C211_PROFILE.critic.layout
        }
        for index, is_valid in enumerate(valid.tolist()):
            live = self._live(index)
            teacher, teacher_hit = self._teacher(index, live, is_valid)
            heading = _yaw_rotation(live["root_rotation"])
            default_q = np.asarray(
                self.cores[index].binding.default_joint_pos, dtype=np.float64
            )
            task_position = heading.T @ (
                np.asarray(self.task.ball_contact_w_m) - live["root_pos"]
            )
            task_velocity = heading.T @ np.asarray(self.task.incoming_velocity_w_mps)
            task_spin = heading.T @ np.asarray(self.task.incoming_spin_w_radps)
            remaining_contact = float(
                self.questions[index].nominal_time_to_contact_s
            ) - float(self.cores[index].data.time)
            remaining_teacher = max(
                0.0,
                self._wait_steps_by_env[index] * self.policy_dt_s
                + self.task.pre_swing_wait_s
                - float(self.cores[index].data.time),
            )
            root_pose = np.concatenate(
                (
                    live["root_pos"] - live["hope_world_translation"],
                    _rotation_to_6d(live["root_rotation"], "root rotation"),
                    live["root_lin_vel_w"],
                )
            )
            actor_values = {
                "actual_base_pose_lin_vel_world": root_pose,
                "base_ang_vel_body": live["root_rotation"].T @ live["root_ang_vel_w"],
                "joint_pos": live["q"] - default_q,
                "joint_vel": live["qd"],
                "actions": self._states[index].previous_action,
                "racket_site_achieved_now_heading": self._heading_racket(
                    live,
                    {
                        "racket_pos": live["racket_pos"],
                        "racket_velocity": live["racket_velocity"],
                        "racket_normal": live["racket_normal"],
                    },
                ),
                "teacher_joint_pos": teacher["joint_pos"] - default_q,
                "teacher_joint_vel": teacher["joint_vel"],
                "racket_site_teacher_now_heading": self._heading_racket(live, teacher),
                "racket_site_teacher_at_reference_hit_heading": self._heading_racket(
                    live, teacher_hit
                ),
                "incoming_ball_contact_position_heading": task_position,
                "incoming_ball_contact_velocity_heading": task_velocity,
                "incoming_ball_contact_spin_heading": task_spin,
                "desired_base_xy_world": np.asarray(self.task.base_goal_w_m[:2])
                - live["hope_world_translation"][:2],
                "time_to_contact": np.asarray([remaining_contact]),
                "time_to_teacher_start": np.asarray([remaining_teacher]),
                "task_valid": np.asarray([float(is_valid)]),
            }
            anchor_rotation = live["anchor_rotation"]
            relative_body_pos = (
                anchor_rotation.T @ (live["body_pos"] - live["anchor_pos"]).T
            ).T
            relative_body_rotation = np.stack(
                [anchor_rotation.T @ rotation for rotation in live["body_rotation"]],
                axis=0,
            )
            critic_values = {
                "command": np.concatenate(
                    (
                        teacher["joint_pos"],
                        teacher["joint_vel"],
                    )
                ),
                "motion_anchor_pos_b": anchor_rotation.T
                @ (teacher["anchor_pos"] - live["anchor_pos"]),
                "motion_anchor_ori_b": _rotation_to_6d(
                    anchor_rotation.T @ teacher["anchor_rotation"],
                    "teacher anchor relative rotation",
                ),
                "body_pos": relative_body_pos.reshape(-1),
                "body_ori": np.concatenate(
                    [
                        _rotation_to_6d(rotation, "body relative rotation")
                        for rotation in relative_body_rotation
                    ]
                ),
                "base_lin_vel": live["root_rotation"].T @ live["root_lin_vel_w"],
                "base_ang_vel": live["root_rotation"].T @ live["root_ang_vel_w"],
                "joint_pos": live["q"] - default_q,
                "joint_vel": live["qd"],
                "actions": self._states[index].previous_action,
                "racket_site_teacher_at_reference_hit_heading": actor_values[
                    "racket_site_teacher_at_reference_hit_heading"
                ],
                "incoming_ball_contact_position_heading": task_position,
                "incoming_ball_contact_velocity_heading": task_velocity,
                "incoming_ball_contact_spin_heading": task_spin,
                "desired_base_xy_world": actor_values["desired_base_xy_world"],
                "time_to_contact": actor_values["time_to_contact"],
                "time_to_teacher_start": actor_values["time_to_teacher_start"],
                "task_valid": actor_values["task_valid"],
            }
            for name in actor_rows:
                actor_rows[name].append(
                    np.asarray(actor_values[name], dtype=np.float64)
                )
            for name in critic_rows:
                critic_rows[name].append(
                    np.asarray(critic_values[name], dtype=np.float64)
                )
        return (
            {name: np.stack(rows, axis=0) for name, rows in actor_rows.items()},
            {name: np.stack(rows, axis=0) for name, rows in critic_rows.items()},
        )

    def tensors(self, task_valid: Sequence[bool]) -> tuple[Any, Any]:
        try:
            import torch
        except ImportError as exc:
            raise C211EnvError("torch is required for the C211 VecEnv") from exc
        actor_groups, critic_groups = self.groups(task_valid)
        actor, critic = abi.flatten_profile_groups(
            abi.C211_PROFILE,
            actor_groups=actor_groups,
            critic_groups=critic_groups,
            task_valid=np.asarray(task_valid, dtype=np.bool_),
            authorities=self.authorities,
        )
        return (
            torch.as_tensor(actor, dtype=torch.float32, device="cpu"),
            torch.as_tensor(critic, dtype=torch.float32, device="cpu"),
        )

    def boundary_state_sha256(
        self, actor: Any, critic: Any, task_valid: Sequence[bool]
    ) -> str:
        rows: dict[str, Any] = {
            "actor": actor.detach().cpu().numpy(),
            "critic": critic.detach().cpu().numpy(),
            "task_valid": list(bool(value) for value in task_valid),
            "previous_action": np.stack(
                [state.previous_action for state in self._states], axis=0
            ),
            "authority_sha256": self.authorities.content_sha256,
        }
        for index, core in enumerate(self.cores):
            data = core.data
            rows[f"core{index}.qpos"] = np.asarray(data.qpos).copy()
            rows[f"core{index}.qvel"] = np.asarray(data.qvel).copy()
            rows[f"core{index}.ctrl"] = np.asarray(data.ctrl).copy()
            rows[f"core{index}.act"] = np.asarray(data.act).copy()
            rows[f"core{index}.qfrc_applied"] = np.asarray(data.qfrc_applied).copy()
            rows[f"core{index}.xfrc_applied"] = np.asarray(data.xfrc_applied).copy()
            rows[f"core{index}.qacc_warmstart"] = np.asarray(data.qacc_warmstart).copy()
            rows[f"core{index}.time"] = float(data.time)
            rows[f"core{index}.policy_tick"] = int(core.policy_tick)
            rows[f"core{index}.delay_steps"] = int(core.plant.delay.delay_steps)
            rows[f"core{index}.delay"] = core.plant.delay.state()
        return _array_digest(rows)


class MujocoC211DiagnosticVecEnv:
    """Asymmetric C211 adapter around the existing fixed-center real VecEnv."""

    def __init__(
        self,
        *,
        base_env: Any,
        task_authority: C211TaskAuthority,
        mimic_authority: MeasuredC211MimicAuthority,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise C211EnvError("torch is required for the C211 VecEnv") from exc
        required = (
            "base_env",
            "task_valid",
            "spec",
            "num_envs",
            "num_actions",
            "device",
            "reset",
            "step",
            "is_reset_boundary",
            "diagnostic_training_identity",
            "diagnostic_training_receipt",
        )
        missing = [name for name in required if not hasattr(base_env, name)]
        if missing:
            raise C211EnvError(f"fixed-center base omits {missing!r}")
        native = base_env.base_env
        if not hasattr(native, "cores") or not hasattr(native, "robot_tape"):
            raise C211EnvError("C211 adapter requires real MujocoN1BallCore rows")
        self.base = base_env
        self.num_envs = int(base_env.num_envs)
        self.num_actions = int(base_env.num_actions)
        self.num_observations = abi.ACTOR_WIDTH
        self.num_privileged_observations = abi.CRITIC_WIDTH
        self.device = torch.device("cpu")
        self.unwrapped = self
        self.cfg = {
            **copy.deepcopy(getattr(base_env, "cfg", {})),
            "kind": C211_ENV_KIND,
            "actor_observation_width": abi.ACTOR_WIDTH,
            "critic_observation_width": abi.CRITIC_WIDTH,
            "reward_scope": C211_REWARD_SCOPE,
            "c211_achieved_outcome_reward_available": True,
            "isaac_synonymous_prior_subset_available": True,
            "complete_isaac_reward_parity_claimed": False,
            "true_c211_training_lane_ready": False,
            "safe_ready_authority_status": SAFE_READY_AUTHORITY_STATUS,
            "safe_ready_formal_pass_claimed": False,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        allow_legacy_wait_test_double = (
            getattr(
                base_env,
                "allow_action_ball_legacy_fixed_wait_test_double",
                False,
            )
            is True
        )
        initial_wait_steps = getattr(base_env, "current_wait_steps", None)
        self._wait_schedule = getattr(base_env, "wait_schedule", None)
        if (
            initial_wait_steps is None or self._wait_schedule is None
        ) and not allow_legacy_wait_test_double:
            raise C211EnvError(
                "C211 requires the authoritative continuous WAIT schedule and "
                "current per-env assignments"
            )
        if initial_wait_steps is None:
            legacy_wait = getattr(base_env.spec, "reset_wait_steps", None)
            initial_wait_steps = (legacy_wait,) * self.num_envs
        if self._wait_schedule is None:
            fixed_wait = int(initial_wait_steps[0])
            self._wait_schedule = SimpleNamespace(
                min_wait_ticks=fixed_wait,
                max_wait_ticks=fixed_wait,
                canonical_sha256=_sha256_json(
                    {
                        "kind": "legacy_fixed_wait_test_double",
                        "wait_ticks": fixed_wait,
                    }
                ),
            )
        self.producer = C211ObservationProducer(
            cores=native.cores,
            questions=native.questions,
            robot_tape=native.robot_tape,
            task=task_authority,
            mimic=mimic_authority,
            reset_wait_steps_by_env=initial_wait_steps,
            policy_dt_s=float(native.step_dt),
        )
        self._native = native
        self._single_stroke_timeout_authority = (
            self._build_single_stroke_timeout_authority()
        )
        self._install_single_stroke_timeouts(
            range(self.num_envs), initial_wait_steps
        )
        self._reward_states = [
            _EpisodeRewardState() for _index in range(self.num_envs)
        ]
        self._reward_contract = self._build_reward_contract(task_authority)
        self._reward_audit: dict[str, Any] = {}
        self.reset_reward_audit()
        self._actor = None
        self._critic = None
        self._canonical_boundary_sha256: str | None = None
        self._current_boundary_state_sha256: str | None = None
        self._identity: dict[str, str] | None = None
        self._install_current_boundary(initial=True)

    def _build_single_stroke_timeout_authority(self) -> dict[str, Any]:
        install = getattr(
            self._native,
            "install_diagnostic_single_stroke_timeout_steps",
            None,
        )
        if not callable(install):
            raise C211EnvError(
                "native VecEnv lacks atomic single-stroke timeout installation"
            )
        dt = float(self._native.step_dt)
        if not math.isclose(dt, C211_POLICY_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise C211EnvError("native single-stroke timeout policy dt differs")
        payload = {
            "schema_version": 1,
            "kind": "action_ball_mujoco_single_stroke_timeout_authority_v1",
            "task_file_sha256": _plain_sha256(
                self.producer.task.file_sha256,
                "single-stroke task file SHA",
            ),
            "motion_file_sha256": _plain_sha256(
                self.producer.mimic.file_sha256,
                "single-stroke motion file SHA",
            ),
            "policy_dt_s": dt,
            "pre_swing_wait_s": float(self.producer.task.pre_swing_wait_s),
            "scaled_t_cycle_s": float(self.producer.task.scaled_t_cycle_s),
            "completion_rule": (
                "first_policy_tick_strictly_after_wait_plus_scaled_cycle_"
                "isaac_close_tick_timeout"
            ),
            "termination_reason": "action_ball_single_stroke_complete",
            "time_out": True,
            "bootstrap_rule": trainer.TIMEOUT_BOOTSTRAP_RULE,
            "time_to_contact_observation": "signed_unclamped_deadline",
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def _single_stroke_timeout_steps(
        self, wait_steps_by_env: Sequence[int]
    ) -> tuple[int, ...]:
        waits = tuple(wait_steps_by_env)
        if len(waits) != self.num_envs or any(
            type(value) is not int or value < 0 for value in waits
        ):
            raise C211EnvError("single-stroke WAIT rows are malformed")
        dt = C211_POLICY_DT_S
        fixed_motion_s = (
            float(self.producer.task.pre_swing_wait_s)
            + float(self.producer.task.scaled_t_cycle_s)
        )
        result = []
        for wait_steps in waits:
            cycle_end_ticks = int(
                math.ceil((wait_steps * dt + fixed_motion_s) / dt - 1.0e-12)
            )
            timeout_step = cycle_end_ticks + 1
            if (
                timeout_step < 1
                or timeout_step > int(self._native.max_episode_length)
            ):
                raise C211EnvError(
                    "single-stroke cycle plus close tick exceeds native horizon"
                )
            result.append(timeout_step)
        return tuple(result)

    def _install_single_stroke_timeouts(
        self, env_ids: Sequence[int], wait_steps_by_env: Sequence[int]
    ) -> None:
        ids = tuple(int(value) for value in env_ids)
        all_waits = tuple(wait_steps_by_env)
        if len(all_waits) != self.num_envs:
            raise C211EnvError("single-stroke WAIT batch cardinality differs")
        all_steps = self._single_stroke_timeout_steps(all_waits)
        try:
            self._native.install_diagnostic_single_stroke_timeout_steps(
                env_ids=ids,
                timeout_steps=tuple(all_steps[index] for index in ids),
                authority_sha256=self._single_stroke_timeout_authority[
                    "content_sha256"
                ],
            )
        except Exception as exc:  # noqa: BLE001 - wrapped runtime boundary
            raise C211EnvError(
                "native single-stroke timeout installation failed"
            ) from exc

    def fresh_actor_bootstrap_contract(self) -> dict[str, Any]:
        """Bind native fresh-policy initialization to the physical hold action.

        The split-ready reset is intentionally not the plant's default pose.
        Therefore normalized action zero is not a safe starting command.  This
        is the same fresh-only output-layer initialization used by the Isaac
        ActionBall lane; checkpoints restore their saved policy and never
        reapply it.
        """

        bias = _finite_vector(
            self.producer.robot_tape.history_fill_action,
            self.num_actions,
            "fresh actor physical-hold output bias",
        )
        bindings = tuple(core.binding for core in self._native.cores)
        if not bindings:
            raise C211EnvError("fresh actor bootstrap has no plant binding")
        first = bindings[0]
        for index, binding in enumerate(bindings[1:], start=1):
            if (
                binding.binding_sha256 != first.binding_sha256
                or binding.joint_names != first.joint_names
                or not np.array_equal(binding.default_joint_pos, first.default_joint_pos)
                or not np.array_equal(binding.action_scale, first.action_scale)
                or not np.array_equal(
                    binding.executed_qdes_limits, first.executed_qdes_limits
                )
            ):
                raise C211EnvError(
                    f"fresh actor bootstrap plant binding {index} differs"
                )
        mean_qdes_raw, mean_qdes, mean_projection_count = first.decode_action(bias)
        if (
            mean_projection_count != 0
            or not np.array_equal(mean_qdes_raw, mean_qdes)
        ):
            raise C211EnvError(
                "fresh actor sealed hold mean requires qdes projection"
            )
        excursion = 4.0 * 0.02 * np.abs(first.action_scale)
        sampled_lower = mean_qdes - excursion
        sampled_upper = mean_qdes + excursion
        executed_lower = first.executed_qdes_limits[:, 0]
        executed_upper = first.executed_qdes_limits[:, 1]
        lower_margin = sampled_lower - executed_lower
        upper_margin = executed_upper - sampled_upper
        unsafe = np.flatnonzero((lower_margin <= 0.0) | (upper_margin <= 0.0))
        unsafe_names = [first.joint_names[int(index)] for index in unsafe]
        payload = trainer.fresh_actor_bootstrap_contract(
            bias.tolist(), initial_action_std=0.02
        )
        payload["safety_gate"] = {
            "schema_version": 2,
            "kind": "a3_action_ball_fresh_actor_wait_bootstrap_gate_v2",
            "sealed_mean_gate": {
                "criterion": (
                    "actor_mean_uses_the_float32_canonical_form_of_the_sealed_"
                    "delay_history_fill_action_and_decodes_without_qdes_projection"
                ),
                "actor_mean_uses_sealed_history_fill_action": True,
                "float64_tape_to_float32_actor_rounding_audited_by_launcher": True,
                "first_action_command_discontinuity_audited_by_launcher": True,
                "mean_qdes_projection_joint_count": mean_projection_count,
                "passed": True,
            },
            "deterministic_runtime_gate": {
                "required_wait_ticks": 25,
                "criterion": (
                    "sealed_mean_survives_max_wait_without_hard_termination_"
                    "or_nonfinite_and_reset_remains_legal"
                ),
                "evaluated_by_launcher_canary": True,
            },
            "stochastic_runtime_gate": {
                "initial_action_std": 0.02,
                "criterion": (
                    "fresh_log_std_samples_during_max_wait_report_per_joint_"
                    "qdes_projection_and_have_zero_hard_or_nonfinite_events"
                ),
                "projection_is_visible_not_implicitly_forbidden": True,
                "evaluated_by_launcher_canary": True,
            },
            "four_sigma_projection_forecast": {
                "role": "analytic_projection_risk_forecast_not_launch_blocker",
                "sigma_envelope": 4.0,
                "criterion": (
                    "hold_qdes_plus_or_minus_4_times_0.02_times_abs_action_scale_"
                    "compared_with_executed_qdes_limits"
                ),
                "joints_not_strictly_inside": unsafe_names,
                "all_joints_strictly_inside": unsafe.size == 0,
            },
            "plant_binding_sha256": _plain_sha256(
                first.binding_sha256, "fresh actor plant binding SHA"
            ),
            "plant_contract_file_sha256": _plain_sha256(
                first.source_sha256, "fresh actor plant source SHA"
            ),
            "robot_tape_file_sha256": _plain_sha256(
                self.producer.robot_tape.source_sha256,
                "fresh actor robot tape SHA",
            ),
            "joint_names": list(first.joint_names),
            "mean_hold_qdes_rad": mean_qdes.tolist(),
            "four_sigma_qdes_excursion_rad": excursion.tolist(),
            "executed_qdes_lower_rad": executed_lower.tolist(),
            "executed_qdes_upper_rad": executed_upper.tolist(),
            "minimum_lower_margin_rad": float(np.min(lower_margin)),
            "minimum_upper_margin_rad": float(np.min(upper_margin)),
        }
        payload.pop("content_sha256")
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def reset_reward_audit(self) -> None:
        """Start one non-semantic finite-window raw reward ledger."""

        if getattr(self.producer, "_reward_capture_active", False):
            raise C211EnvError("cannot reset reward audit during a transition")
        self._reward_audit = {
            "transition_step_count": 0,
            "row_count": 0,
            "task_valid_row_count": 0,
            "wait_row_count": 0,
            "isaac_synonymous_prior_reward_sum": 0.0,
            "strike_sample_count": 0,
            "strike_reward_sum": 0.0,
            "closed_attempt_count": 0,
            "closed_attempt_without_selected_contact_count": 0,
            "actual_contact_count": 0,
            "selected_contact_count": 0,
            "valid_achieved_flight_count": 0,
            "outcome_evaluation_count": 0,
            "landing_evaluation_count": 0,
            "legal_opponent_table_count": 0,
            "opponent_side_off_table_count": 0,
            "landing_reward_sum": 0.0,
            "total_reward_sum": 0.0,
            "prior_terms": {
                name: {
                    "sample_count": 0,
                    "raw_reward_sum": 0.0,
                    "raw_reward_min": None,
                    "raw_reward_max": None,
                    "post_policy_dt_reward_sum": 0.0,
                    "post_policy_dt_reward_min": None,
                    "post_policy_dt_reward_max": None,
                }
                for name in C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES
            },
        }

    def _accumulate_reward_audit(
        self, reward_rows: Sequence[Mapping[str, Any]], dones: Any
    ) -> None:
        try:
            done_rows = tuple(bool(value) for value in dones.tolist())
        except (AttributeError, TypeError) as exc:
            raise C211EnvError("C211 reward audit done rows are malformed") from exc
        if len(reward_rows) != self.num_envs or len(done_rows) != self.num_envs:
            raise C211EnvError("C211 reward audit row count differs")
        audit = self._reward_audit
        audit["transition_step_count"] += 1
        for row, done in zip(reward_rows, done_rows):
            if not isinstance(row, Mapping) or type(row.get("task_valid")) is not bool:
                raise C211EnvError("C211 reward audit row is malformed")
            terms = row.get("isaac_synonymous_prior_terms")
            if (
                not isinstance(terms, Mapping)
                or tuple(terms) != C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES
            ):
                raise C211EnvError("C211 reward audit prior terms differ")
            audit["row_count"] += 1
            validity_key = (
                "task_valid_row_count" if row["task_valid"] else "wait_row_count"
            )
            audit[validity_key] += 1
            for key in (
                "isaac_synonymous_prior_reward",
                "strike_reward",
                "landing_reward",
                "total_reward",
            ):
                value = float(row.get(key))
                if not math.isfinite(value):
                    raise C211EnvError(f"C211 reward audit {key} is non-finite")
                audit[f"{key}_sum"] += value
            audit["strike_sample_count"] += int(
                row.get("nominal_strike_sampled_now") is True
            )
            active_done = bool(done and row["task_valid"])
            audit["closed_attempt_count"] += int(active_done)
            audit["closed_attempt_without_selected_contact_count"] += int(
                active_done and row.get("attempt_had_selected_contact") is False
            )
            audit["actual_contact_count"] += int(
                row.get("actual_contact_observed_now") is True
            )
            audit["selected_contact_count"] += int(
                row.get("selected_contact_observed_now") is True
            )
            audit["valid_achieved_flight_count"] += int(
                row.get("valid_achieved_flight_observed_now") is True
            )
            audit["outcome_evaluation_count"] += int(
                row.get("outcome_evaluated_now") is True
            )
            landing = row.get("landing_terms")
            classification = (
                landing.get("classification")
                if isinstance(landing, Mapping)
                else None
            )
            audit["landing_evaluation_count"] += int(
                isinstance(landing, Mapping)
            )
            audit["legal_opponent_table_count"] += int(
                classification == "legal_opponent_table"
            )
            audit["opponent_side_off_table_count"] += int(
                classification == "opponent_side_off_table"
            )
            for name, term in terms.items():
                if not isinstance(term, Mapping):
                    raise C211EnvError("C211 reward audit term is malformed")
                raw = float(term.get("raw_reward"))
                post_dt = float(term.get("post_policy_dt_reward"))
                if not math.isfinite(raw) or not math.isfinite(post_dt):
                    raise C211EnvError("C211 reward audit term is non-finite")
                target = audit["prior_terms"][name]
                target["sample_count"] += 1
                target["raw_reward_sum"] += raw
                target["post_policy_dt_reward_sum"] += post_dt
                for prefix, value in (("raw_reward", raw), ("post_policy_dt_reward", post_dt)):
                    minimum = f"{prefix}_min"
                    maximum = f"{prefix}_max"
                    target[minimum] = (
                        value if target[minimum] is None else min(target[minimum], value)
                    )
                    target[maximum] = (
                        value if target[maximum] is None else max(target[maximum], value)
                    )

    def reward_audit_receipt(self) -> dict[str, Any]:
        """Return actual raw/per-dt term aggregates for the current finite window."""

        payload = {
            "schema_version": 1,
            "kind": "action_ball_c211_mujoco_raw_reward_audit_v1",
            "reward_scope": C211_REWARD_SCOPE,
            "reward_contract_sha256": self._reward_contract["content_sha256"],
            "reward_parity_status": "partial_fail_closed",
            "closed_attempt_denominator_semantics": (
                "done_and_TASK_ACTIVE_transition_only_RESET_WAIT_done_excluded"
            ),
            "additive_reward_component_names": [
                "isaac_synonymous_prior_reward",
                "strike_reward",
                "landing_reward",
            ],
            "nonadditive_alias_map": {},
            **copy.deepcopy(self._reward_audit),
            "complete_isaac_reward_parity_claimed": False,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def _build_reward_contract(
        self, task: C211TaskAuthority
    ) -> dict[str, Any]:
        """Reopen and bind every source used by the C211 task scalar."""

        try:
            import torch
        except ImportError as exc:
            raise C211EnvError("torch is required for the C211 reward") from exc
        if (
            getattr(self._native, "native_physical_event_runtime_available", None)
            is not True
        ):
            raise C211EnvError("C211 reward requires the native physical-event ABI")
        sources = tuple(
            getattr(self._native, "native_physical_event_source_bindings", ())
        )
        if len(sources) != self.num_envs or any(
            type(value) is not n1_reward_event_kernel.SourceBinding
            for value in sources
        ):
            raise C211EnvError("C211 native physical-event source bindings differ")

        virtual_ball = _load_virtual_ball_module()
        try:
            params = virtual_ball.load_venue_params(str(VENUE_PHYSICS_YAML))
            table_scene = physical_ball_scene._load_table_scene_module()
            geometry, _table_frame = table_scene.load_geometry_and_frame()
        except Exception as exc:  # noqa: BLE001 - pinned source-loader boundary
            raise C211EnvError(f"C211 reward sources cannot be loaded: {exc}") from exc
        required_geometry = {
            "TABLE_LENGTH": 2.74,
            "TABLE_WIDTH": 1.525,
            "NET_X": 1.37,
            "NET_HEIGHT": 0.1525,
        }
        for name, expected in required_geometry.items():
            value = float(getattr(geometry, name, float("nan")))
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise C211EnvError(f"C211 table geometry {name} differs")
        if not math.isclose(
            float(params.ball_radius),
            float(getattr(geometry, "BALL_RADIUS", float("nan"))),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise C211EnvError("C211 venue and table ball radii differ")

        near_values: list[float] = []
        surface_values: list[float] = []
        ball_radii: list[float] = []
        event_rows: list[dict[str, str]] = []
        question_source_shas: list[str] = []
        outcome_rows: list[dict[str, str]] = []
        for index, (core, question, source, binding) in enumerate(
            zip(
                self._native.cores,
                self._native.questions,
                sources,
                self.producer._bindings,
            )
        ):
            try:
                near_x = float(core.scene.near_x)
                surface_z = float(core.scene.surface_z)
                ball_radius = float(
                    core.model.geom_size[int(core.scene.ball_geom_id), 0]
                )
                outcome_binding = core.observed_outcome_resolver_binding
                outcome_row = {
                    "resolver_binding_sha256": _plain_sha256(
                        outcome_binding["content_sha256"],
                        f"core {index} outcome resolver SHA",
                    ),
                    "question_binding_sha256": _plain_sha256(
                        core.observed_outcome_question_binding_sha256,
                        f"core {index} outcome question SHA",
                    ),
                    "scene_binding_sha256": _plain_sha256(
                        core.scene_binding_sha256,
                        f"core {index} scene SHA",
                    ),
                    "plant_binding_sha256": _plain_sha256(
                        core.binding.binding_sha256,
                        f"core {index} plant SHA",
                    ),
                }
                question_source_sha = _plain_sha256(
                    question.source_sha256, f"question {index} source SHA"
                )
            except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise C211EnvError(
                    f"core {index} omits C211 reward authority"
                ) from exc
            if (
                not all(
                    math.isfinite(value)
                    for value in (near_x, surface_z, ball_radius)
                )
                or ball_radius <= 0.0
                or not math.isclose(
                    ball_radius,
                    float(params.ball_radius),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not np.array_equal(
                    _finite_vector(
                        question.landing_aim_xy_w_m,
                        2,
                        f"question {index} reward landing aim",
                    ),
                    np.asarray(task.landing_aim_w_xy_m, dtype=np.float64),
                )
                or not np.allclose(
                    np.asarray(binding.hope_world_translation_m),
                    np.asarray(
                        (near_x, 0.5 * geometry.TABLE_WIDTH, surface_z)
                    ),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise C211EnvError(f"core {index} C211 landing frame differs")
            near_values.append(near_x)
            surface_values.append(surface_z)
            ball_radii.append(ball_radius)
            question_source_shas.append(question_source_sha)
            event_rows.append(
                {
                    "source_id": source.source_id,
                    "source_sha256": _plain_sha256(
                        source.source_sha256, f"core {index} native source SHA"
                    ),
                    "event_contract_sha256": _plain_sha256(
                        source.event_contract_sha256,
                        f"core {index} native event ABI SHA",
                    ),
                }
            )
            outcome_rows.append(outcome_row)
        if (
            len(set(near_values)) != 1
            or len(set(surface_values)) != 1
            or len(set(ball_radii)) != 1
        ):
            raise C211EnvError("C211 vector rows do not share one landing frame")

        self._virtual_ball = virtual_ball
        self._venue_params = params
        self._event_sources = sources
        self._table_near_x = near_values[0]
        self._table_surface_z = surface_values[0]
        self._ball_radius_m = ball_radii[0]
        self._table_net_x = self._table_near_x + float(geometry.NET_X)
        self._table_far_x = self._table_near_x + float(geometry.TABLE_LENGTH)
        self._table_half_width = 0.5 * float(geometry.TABLE_WIDTH)
        self._net_clear_center_z = (
            self._table_surface_z
            + float(geometry.NET_HEIGHT)
            + self._ball_radius_m
        )
        nominal_ticks = []
        for wait_steps in range(
            self._wait_schedule.min_wait_ticks,
            self._wait_schedule.max_wait_ticks + 1,
        ):
            tick_float = (
                task.time_to_contact_s + wait_steps * C211_POLICY_DT_S
            ) / C211_POLICY_DT_S
            tick = int(round(tick_float))
            if tick < 1 or not math.isclose(
                tick_float, float(tick), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise C211EnvError("C211 nominal strike is not on a policy tick")
            nominal_ticks.append(tick)
        self._nominal_strike_tick_range = tuple(nominal_ticks)

        params_payload = {
            name: float(getattr(params, name))
            for name in (
                "k_d",
                "k_m",
                "g",
                "ball_radius",
                "inertia_coeff",
                "paddle_a_t",
                "paddle_b_t",
                "paddle_mu",
                "paddle_e_g1",
                "paddle_e_g2",
            )
        }
        implemented_prior_terms = [
            {
                "term": "upright_exp",
                "manager_weight": 1.0,
                "params": {"std": C211_UPRIGHT_STD},
                "source": "live_root_rotation_projected_gravity_body",
            },
            {
                "term": "base_ang_vel_xy",
                "manager_weight": -0.05,
                "source": "live_root_body_frame_angular_velocity_xy",
            },
            {
                "term": "base_lin_vel_z",
                "manager_weight": -0.5,
                "source": "live_root_inertial_COM_velocity_body_z",
            },
            {
                "term": "joint_vel",
                "manager_weight": -1.0e-4,
                "source": "live_MuJoCo_joint_dof_velocity",
            },
            {
                "term": "action_rate_clamped",
                "manager_weight": -0.2,
                "params": {"value_clamp": C211_ACTION_RATE_CLAMP},
                "source": "raw_actor_action_minus_previous_raw_actor_action",
            },
            {
                "term": "motion_global_anchor_ori",
                "manager_weight": 0.075,
                "params": {"std_rad": 0.4},
                "source": "measured_teacher_and_live_torso_orientation",
            },
            {
                "term": "motion_body_pos",
                "manager_weight": 0.15,
                "params": {"std_m": 0.3},
                "body_names": list(MIMIC_BODY_NAMES),
            },
            {
                "term": "motion_body_ori",
                "manager_weight": 0.15,
                "params": {"std_rad": 0.4},
                "body_names": list(MIMIC_BODY_NAMES),
            },
            {
                "term": "motion_body_lin_vel",
                "manager_weight": 0.15,
                "params": {"std_mps": 1.0, "point": "center_of_mass"},
                "body_names": list(MIMIC_BODY_NAMES),
            },
            {
                "term": "motion_body_ang_vel",
                "manager_weight": 0.15,
                "params": {"std_radps": 3.14},
                "body_names": list(MIMIC_BODY_NAMES),
            },
            {
                "term": "motion_racket_position",
                "manager_weight": 0.20,
                "params": {"std_m": 0.70, "kernel": "cauchy"},
                "source": "measured_physical_blade_center",
            },
            {
                "term": "motion_racket_velocity",
                "manager_weight": 0.20,
                "params": {"std_mps": 4.0, "kernel": "cauchy"},
                "source": "measured_official_site_point_velocity",
            },
            {
                "term": "motion_racket_normal",
                "manager_weight": 0.20,
                "params": {"std_rad": math.pi, "kernel": "cauchy"},
                "source": "measured_signed_physical_hitting_face",
            },
            {
                "term": "motion_racket_long_axis",
                "manager_weight": 0.10,
                "params": {"std_rad": 1.0, "kernel": "cauchy"},
                "source": "measured_paddle_butt_to_blade",
            },
        ]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": C211_REWARD_CONTRACT_IDENTITY,
            "scope": C211_REWARD_SCOPE,
            "task_reward_contract_identity": C211_TASK_REWARD_CONTRACT_IDENTITY,
            "task_authority_sha256": task.content_sha256,
            "task_physics_sha256": task.physics_sha256,
            "task_profile_sha256": task.profile_sha256,
            "landing_aim_w_xy_m": list(task.landing_aim_w_xy_m),
            "desired_contact_position_velocity_face_consumed": False,
            "policy_dt_s": C211_POLICY_DT_S,
            "wait_reward_semantics": {
                "balance_action_body_and_measured_paddle_priors_active": True,
                "task_strike_contact_and_outcome_masked": True,
                "task_valid_applied_to_priors": False,
            },
            "isaac_synonymous_prior_subset": {
                "status": "implemented_partial_fail_closed",
                "complete_isaac_reward_parity_claimed": False,
                "post_physics_sampling": (
                    "final_MuJoCo_substep_callback_before_compact_reset"
                ),
                "implemented_terms": implemented_prior_terms,
                "unavailable_terms": [
                    dict(row) for row in C211_UNAVAILABLE_ISAAC_REWARD_TERMS
                ],
                "right_wrist_generic_body_mimic_excluded": True,
                "measured_body_velocity_channels_consumed": True,
                "measured_paddle_long_axis_consumed": True,
            },
            "cross_engine_reward_semantic_gaps": [
                dict(row) for row in C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
            ],
            "strike_bridge": {
                "source": "achieved_official_racket_site_vs_immutable_ball_contact",
                "single_nominal_tick": True,
                "nominal_policy_tick_range_1based": list(
                    self._nominal_strike_tick_range
                ),
                "task_valid_required": True,
                "std_m": C211_STRIKE_STD_M,
                "kernel": "1/(1+(distance/std)^2)",
                "post_policy_dt_weight": C211_STRIKE_POST_DT_WEIGHT,
            },
            "landing": {
                "source": (
                    "first_source_bound_outgoing_flight_after_actual_selected_rubber_contact"
                ),
                "contact_transition_tick_must_equal_nominal_tick": True,
                "contact_timing_semantics": (
                    "native_zero_based_contact_stamp_plus_one_equals_the_"
                    "one_based_nominal_transition_tick"
                ),
                "selected_face_finite_positive_closing_speed_required": True,
                "contact_substep_kinematics_source": (
                    "latched_preimpact_ball_contact_point_and_racket_site_"
                    "velocities_in_wrapped_native_substep_observer"
                ),
                "task_valid_required": True,
                "one_evaluation_per_attempt": True,
                "rollout_dtype": "torch.float32_cpu",
                "rollout_h_s": C211_ROLLOUT_H_S,
                "rollout_steps": C211_ROLLOUT_STEPS,
                "landing_sigma_m": C211_LANDING_SIGMA_M,
                "legal_raw": "0.6+0.4*exp(-squared_error/sigma^2)",
                "opponent_side_off_table_raw": (
                    "0.5*exp(-squared_error/sigma^2)"
                ),
                "post_policy_dt_weight": C211_LANDING_POST_DT_WEIGHT,
                "own_side_backwards_net_or_invalid_raw": 0.0,
                "observed_physical_landing_consumed": False,
            },
            "table": {
                "near_x_w_m": self._table_near_x,
                "surface_z_w_m": self._table_surface_z,
                "net_x_w_m": self._table_net_x,
                "far_x_w_m": self._table_far_x,
                "half_width_m": self._table_half_width,
                "net_clear_ball_center_z_w_m": self._net_clear_center_z,
                "landing_ball_center_z_w_m": (
                    self._table_surface_z + self._ball_radius_m
                ),
                "geometry_source_sha256": _sha256_file(
                    Path(geometry.__file__).resolve()
                ),
                "table_scene_source_sha256": _sha256_file(
                    Path(table_scene.__file__).resolve()
                ),
                "physical_scene_source_sha256": _sha256_file(
                    Path(physical_ball_scene.__file__).resolve()
                ),
                "geometry_contract_sha256s": sorted(
                    {value.table_geometry_sha256 for value in self.producer._bindings}
                ),
            },
            "predictor": {
                "virtual_ball_source_sha256": _sha256_file(VIRTUAL_BALL_PY),
                "venue_physics_source_sha256": _sha256_file(VENUE_PHYSICS_YAML),
                "venue_params": params_payload,
            },
            "semantic_authorities": {
                "c211_trainability_source_sha256": _sha256_file(
                    C211_TRAINABILITY_PY
                ),
                "c225_reward_source_sha256": _sha256_file(C225_REWARD_PY),
                "native_reward_event_kernel_source_sha256": _sha256_file(
                    Path(n1_reward_event_kernel.__file__).resolve()
                ),
                "native_event_facts_contract_sha256": (
                    n1_reward_event_kernel.native_physical_event_facts_contract()[
                        "content_sha256"
                    ]
                ),
                "hope_env_cfg_source_sha256": _sha256_file(HOPE_ENV_CFG_PY),
                "hope_rewards_source_sha256": _sha256_file(HOPE_REWARDS_PY),
                "reward_pack_resolver_source_sha256": _sha256_file(TRAIN_PY),
                "vendor_v2_task_yaml_sha256": _sha256_file(
                    VENDOR_V2_TASK_YAML
                ),
                "c211_task_yaml_sha256": _sha256_file(C211_TASK_YAML),
            },
            "runtime_bindings": {
                "event_sources": event_rows,
                "question_source_sha256s": question_source_shas,
                "outcome_validation": outcome_rows,
                "selected_rubber_classifier_sha256s": [
                    value.selected_rubber_classifier_sha256
                    for value in self.producer._bindings
                ],
                "selected_rubber_lineage_sha256s": [
                    value.selected_rubber_lineage_sha256
                    for value in self.producer._bindings
                ],
            },
            "wrapped_fixed_center_reward_consumed": False,
            "full_body_mimic_reward_consumed": True,
            "measured_paddle_prior_reward_consumed": True,
            "complete_isaac_reward_parity_claimed": False,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def _event_evidence(
        self, index: int, native_facts: Mapping[str, Any]
    ) -> tuple[
        n1_reward_event_kernel.ContactEvidence,
        n1_reward_event_kernel.OutgoingFlightEvidence,
    ]:
        core = self._native.cores[index]
        question = self._native.questions[index]
        try:
            expected = {
                "expected_source": self._event_sources[index],
                "expected_outcome_resolver_binding_sha256": (
                    core.observed_outcome_resolver_binding["content_sha256"]
                ),
                "expected_outcome_question_binding_sha256": (
                    core.observed_outcome_question_binding_sha256
                ),
                "expected_outcome_scene_binding_sha256": core.scene_binding_sha256,
                "expected_outcome_plant_binding_sha256": core.binding.binding_sha256,
                "expected_question_source_sha256": question.source_sha256,
                "expected_question_landing_aim_xy_w_m": tuple(
                    float(value) for value in question.landing_aim_xy_w_m
                ),
            }
            contact = n1_reward_event_kernel.contact_evidence_from_native_facts(
                native_facts, **expected
            )
            flight = n1_reward_event_kernel.outgoing_flight_evidence_from_native_facts(
                native_facts, **expected
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            n1_reward_event_kernel.N1RewardEventKernelError,
        ) as exc:
            raise C211EnvError(
                f"env {index} native reward evidence differs: {exc}"
            ) from exc
        return contact, flight

    def _predict_achieved_landing(
        self, flight: n1_reward_event_kernel.OutgoingFlightEvidence
    ) -> dict[str, Any]:
        """Run the shared fitted flight model from one achieved outgoing state."""

        try:
            import torch
        except ImportError as exc:
            raise C211EnvError("torch is required for the C211 reward") from exc
        if (
            type(flight) is not n1_reward_event_kernel.OutgoingFlightEvidence
            or not flight.valid
            or flight.position_w_m is None
            or flight.linear_velocity_w_mps is None
            or flight.spin_w_radps is None
        ):
            raise C211EnvError("C211 landing predictor requires one achieved flight")
        position = torch.as_tensor(
            [flight.position_w_m], dtype=torch.float32, device="cpu"
        )
        velocity = torch.as_tensor(
            [flight.linear_velocity_w_mps], dtype=torch.float32, device="cpu"
        )
        spin = torch.as_tensor(
            [flight.spin_w_radps], dtype=torch.float32, device="cpu"
        )
        if not (
            torch.isfinite(position).all()
            and torch.isfinite(velocity).all()
            and torch.isfinite(spin).all()
        ):
            raise C211EnvError("C211 achieved outgoing flight is non-finite")
        try:
            with torch.no_grad():
                result = self._virtual_ball.coarse_landing(
                    position,
                    velocity,
                    spin,
                    self._venue_params,
                    surface_z=self._table_surface_z + self._ball_radius_m,
                    net_x=self._table_net_x,
                    h=C211_ROLLOUT_H_S,
                    n_steps=C211_ROLLOUT_STEPS,
                )
            landing_xy = result["land_xy"]
            landing_valid_tensor = result["land_valid"]
            net_z_tensor = result["net_z"]
            net_valid_tensor = result["net_valid"]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise C211EnvError(f"C211 landing rollout failed: {exc}") from exc
        if (
            not isinstance(landing_xy, torch.Tensor)
            or tuple(landing_xy.shape) != (1, 2)
            or not isinstance(landing_valid_tensor, torch.Tensor)
            or tuple(landing_valid_tensor.shape) != (1,)
            or landing_valid_tensor.dtype != torch.bool
            or not isinstance(net_z_tensor, torch.Tensor)
            or tuple(net_z_tensor.shape) != (1,)
            or not isinstance(net_valid_tensor, torch.Tensor)
            or tuple(net_valid_tensor.shape) != (1,)
            or net_valid_tensor.dtype != torch.bool
            or not torch.isfinite(landing_xy).all()
            or not torch.isfinite(net_z_tensor).all()
        ):
            raise C211EnvError("C211 landing rollout returned a malformed row")
        landing = [float(value) for value in landing_xy[0].cpu().tolist()]
        landing_valid = bool(landing_valid_tensor[0].item())
        net_crossed = bool(net_valid_tensor[0].item())
        net_z = float(net_z_tensor[0].item())
        net_clear = bool(net_crossed and net_z > self._net_clear_center_z)
        return {
            "landing_xy_w_m": landing,
            "landing_valid": landing_valid,
            "net_crossed": net_crossed,
            "net_z_w_m": net_z,
            "net_clear": net_clear,
        }

    def _evaluate_reward_transition(
        self,
        *,
        base_extras: Mapping[str, Any],
        transition_valid: tuple[bool, ...],
        prior_rows: Sequence[Mapping[str, Any]],
        dones: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Consume preserved pre-reset facts and return the exact C211 scalar."""

        try:
            import torch
        except ImportError as exc:
            raise C211EnvError("torch is required for the C211 reward") from exc
        if not isinstance(base_extras, Mapping):
            raise C211EnvError("fixed-center extras must be a mapping")
        try:
            facts_rows = tuple(
                base_extras["diagnostic_native_physical_event_facts"]
            )
            physical_rows = tuple(base_extras["diagnostic_c_lite_physical_samples"])
            base_reward_rows = tuple(base_extras["reward_terms"])
            sideband_valid = tuple(base_extras["task_valid_transition"])
            done_rows = tuple(bool(value) for value in dones.tolist())
        except (KeyError, TypeError, AttributeError) as exc:
            raise C211EnvError(
                "fixed-center transition omits preserved C211 reward facts"
            ) from exc
        if (
            len(facts_rows) != self.num_envs
            or len(physical_rows) != self.num_envs
            or len(base_reward_rows) != self.num_envs
            or len(sideband_valid) != self.num_envs
            or len(prior_rows) != self.num_envs
            or len(done_rows) != self.num_envs
            or any(type(value) is not bool for value in sideband_valid)
            or sideband_valid != transition_valid
        ):
            raise C211EnvError("fixed-center C211 reward row counts or validity differ")

        rewards: list[float] = []
        rows: list[dict[str, Any]] = []
        next_flags: list[tuple[bool, bool, bool, bool, bool]] = []
        for index in range(self.num_envs):
            facts = facts_rows[index]
            physical = physical_rows[index]
            base_row = base_reward_rows[index]
            prior_row = prior_rows[index]
            if not all(
                isinstance(value, Mapping)
                for value in (facts, physical, base_row, prior_row)
            ):
                raise C211EnvError(f"env {index} C211 reward facts are malformed")
            try:
                policy_tick = facts["policy_tick"]
                sample_time_s = float(physical["sample_time_s"])
                classifier_sha = _plain_sha256(
                    physical["classifier_binding_sha256"],
                    f"env {index} physical classifier SHA",
                )
                lineage_sha = _plain_sha256(
                    physical["selected_rubber_lineage_sha256"],
                    f"env {index} physical lineage SHA",
                )
                prior_sample_time_s = float(prior_row["sample_time_s"])
                prior_substep = prior_row["physics_substep"]
                prior_reward = float(prior_row["total_post_policy_dt_reward"])
                prior_terms = prior_row["terms"]
            except (KeyError, TypeError, ValueError) as exc:
                raise C211EnvError(
                    f"env {index} physical C211 sample differs"
                ) from exc
            if (
                type(policy_tick) is not int
                or policy_tick < 1
                or not math.isfinite(sample_time_s)
                or not math.isclose(
                    sample_time_s,
                    policy_tick * C211_POLICY_DT_S,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or classifier_sha
                != self.producer._bindings[index].selected_rubber_classifier_sha256
                or lineage_sha
                != self.producer._bindings[index].selected_rubber_lineage_sha256
                or prior_row.get("task_valid") is not transition_valid[index]
                or type(prior_substep) is not int
                or prior_substep
                != self.producer._reward_capture_final_substeps[index]
                or not math.isfinite(prior_sample_time_s)
                or not math.isclose(
                    prior_sample_time_s,
                    sample_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isfinite(prior_reward)
                or not isinstance(prior_terms, Mapping)
            ):
                raise C211EnvError(f"env {index} physical C211 sample is unbound")
            nominal_strike_tick = base_row.get("nominal_strike_tick")
            if (
                type(nominal_strike_tick) is not int
                or nominal_strike_tick not in self._nominal_strike_tick_range
            ):
                raise C211EnvError(
                    f"env {index} nominal strike assignment is unbound"
                )
            strike_now = bool(
                transition_valid[index]
                and policy_tick == nominal_strike_tick
            )
            if (
                base_row.get("task_valid") is not transition_valid[index]
                or base_row.get("sample_policy_tick_1based") != policy_tick
                or base_row.get("nominal_strike_sampled_now") is not strike_now
            ):
                raise C211EnvError(
                    f"env {index} fixed-center strike sideband differs"
                )

            state = self._reward_states[index]
            next_strike = state.strike_sampled
            next_outcome = state.outcome_evaluated
            next_contact = state.actual_contact_observed
            next_selected_contact = state.selected_contact_observed
            next_valid_flight = state.valid_achieved_flight_observed
            contact_observed_now = False
            selected_contact_observed_now = False
            eligible_contact_observed_now = False
            valid_flight_observed_now = False
            strike_reward = 0.0
            strike_terms: dict[str, Any] | None = None
            if strike_now:
                if state.strike_sampled:
                    raise C211EnvError(
                        f"env {index} C211 nominal strike was sampled twice"
                    )
                strike_terms = _c211_strike_reward_terms(
                    official_racket_site_w_m=physical.get(
                        "official_racket_site_w_m"
                    ),
                    immutable_ball_contact_w_m=(
                        self.producer.task.ball_contact_w_m
                    ),
                )
                strike_reward = float(strike_terms["post_policy_dt_reward"])
                next_strike = True

            landing_reward = 0.0
            landing_terms: dict[str, Any] | None = None
            contact_row: dict[str, Any] | None = None
            contact_eligibility: dict[str, Any] | None = None
            flight_row: dict[str, Any] | None = None
            outcome_reason = "not_yet_observed"
            if transition_valid[index] and not state.outcome_evaluated:
                contact, flight = self._event_evidence(index, facts)
                contact_observed_now = bool(
                    contact.occurred and not state.actual_contact_observed
                )
                selected_contact_observed_now = bool(
                    contact.occurred
                    and contact.selected_rubber
                    and not state.selected_contact_observed
                )
                valid_flight_observed_now = bool(
                    flight.valid and not state.valid_achieved_flight_observed
                )
                next_contact = bool(next_contact or contact.occurred)
                next_selected_contact = bool(
                    next_selected_contact
                    or (contact.occurred and contact.selected_rubber)
                )
                next_valid_flight = bool(next_valid_flight or flight.valid)
                contact_eligibility = self.producer.contact_eligibility(
                    index,
                    contact,
                    nominal_strike_tick_1based=nominal_strike_tick,
                )
                eligible_contact_observed_now = bool(
                    contact_eligibility["eligible"]
                    and not state.outcome_evaluated
                )
                contact_row = {
                    "occurred": contact.occurred,
                    "selected_rubber": contact.selected_rubber,
                    "stamp": (
                        None
                        if contact.stamp is None
                        else {
                            "policy_tick": contact.stamp.policy_tick,
                            "physics_substep": contact.stamp.physics_substep,
                        }
                    ),
                    "eligibility": copy.deepcopy(contact_eligibility),
                }
                flight_row = {
                    "valid": flight.valid,
                    "stamp": (
                        None
                        if flight.stamp is None
                        else {
                            "policy_tick": flight.stamp.policy_tick,
                            "physics_substep": flight.stamp.physics_substep,
                        }
                    ),
                }
                if contact.occurred and not contact.selected_rubber:
                    next_outcome = True
                    outcome_reason = "first_contact_not_selected_rubber"
                elif contact.occurred and not contact_eligibility["eligible"]:
                    next_outcome = True
                    outcome_reason = str(contact_eligibility["reason"])
                elif contact.occurred and contact.selected_rubber and flight.valid:
                    if (
                        contact.stamp is None
                        or flight.stamp is None
                        or flight.stamp <= contact.stamp
                    ):
                        raise C211EnvError(
                            f"env {index} achieved flight does not follow contact"
                        )
                    next_outcome = True
                    prediction = self._predict_achieved_landing(flight)
                    landing_terms = _c211_landing_reward_terms(
                        landing_xy_w_m=prediction["landing_xy_w_m"],
                        landing_valid=prediction["landing_valid"],
                        net_crossed=prediction["net_crossed"],
                        net_clear=prediction["net_clear"],
                        landing_aim_w_xy_m=(
                            self.producer.task.landing_aim_w_xy_m
                        ),
                        net_x_w_m=self._table_net_x,
                        far_x_w_m=self._table_far_x,
                        half_width_m=self._table_half_width,
                    )
                    landing_terms["net_z_w_m"] = prediction["net_z_w_m"]
                    landing_reward = float(
                        landing_terms["post_policy_dt_reward"]
                    )
                    outcome_reason = "achieved_flight_evaluated"
                elif contact.occurred and contact.selected_rubber:
                    outcome_reason = "selected_contact_waiting_for_outgoing_flight"
                elif not contact.occurred:
                    outcome_reason = "no_contact_yet"

            total = prior_reward + strike_reward + landing_reward
            if not math.isfinite(total):
                raise C211EnvError(f"env {index} C211 reward is invalid")
            additive_components = {
                "isaac_synonymous_prior_reward": prior_reward,
                "strike_reward": strike_reward,
                "landing_reward": landing_reward,
            }
            if not math.isclose(
                sum(additive_components.values()),
                total,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise C211EnvError(
                    f"env {index} C211 additive reward accounting differs"
                )
            rewards.append(total)
            rows.append(
                {
                    "task_valid": transition_valid[index],
                    "sample_policy_tick_1based": policy_tick,
                    "sample_time_s": sample_time_s,
                    "nominal_strike_tick": nominal_strike_tick,
                    "nominal_strike_sampled_now": strike_now,
                    "isaac_synonymous_prior_terms": copy.deepcopy(prior_terms),
                    "isaac_synonymous_prior_reward": prior_reward,
                    "isaac_synonymous_prior_always_on": True,
                    "isaac_synonymous_prior_task_mask_applied": False,
                    "strike_terms": strike_terms,
                    "strike_reward": strike_reward,
                    "actual_contact": contact_row,
                    "actual_contact_observed_now": contact_observed_now,
                    "selected_contact_observed_now": selected_contact_observed_now,
                    "eligible_contact_observed_now": eligible_contact_observed_now,
                    "achieved_outgoing_flight": flight_row,
                    "valid_achieved_flight_observed_now": valid_flight_observed_now,
                    "landing_terms": landing_terms,
                    "landing_reward": landing_reward,
                    "outcome_reason": outcome_reason,
                    "outcome_evaluated_now": bool(
                        next_outcome and not state.outcome_evaluated
                    ),
                    "attempt_closed_now": bool(
                        done_rows[index] and transition_valid[index]
                    ),
                    "attempt_had_actual_contact": next_contact,
                    "attempt_had_selected_contact": next_selected_contact,
                    "attempt_had_valid_achieved_flight": next_valid_flight,
                    "additive_reward_components": additive_components,
                    "nonadditive_alias_map": {},
                    "total_reward": total,
                }
            )
            next_flags.append(
                (
                    next_strike,
                    next_outcome,
                    next_contact,
                    next_selected_contact,
                    next_valid_flight,
                )
            )

        for index, (
            next_strike,
            next_outcome,
            next_contact,
            next_selected_contact,
            next_valid_flight,
        ) in enumerate(next_flags):
            if done_rows[index]:
                self._reward_states[index].reset()
            else:
                self._reward_states[index].strike_sampled = next_strike
                self._reward_states[index].outcome_evaluated = next_outcome
                self._reward_states[index].actual_contact_observed = next_contact
                self._reward_states[index].selected_contact_observed = (
                    next_selected_contact
                )
                self._reward_states[index].valid_achieved_flight_observed = (
                    next_valid_flight
                )
        return (
            torch.as_tensor(rewards, dtype=torch.float32, device="cpu"),
            rows,
        )

    def _task_valid_rows(self) -> tuple[bool, ...]:
        try:
            rows = tuple(self.base.task_valid)
        except TypeError as exc:
            raise C211EnvError("fixed-center task_valid is not a row sequence") from exc
        if len(rows) != self.num_envs or any(type(value) is not bool for value in rows):
            raise C211EnvError("fixed-center task_valid must be exact bool[N]")
        return rows

    def _install_current_boundary(self, *, initial: bool) -> None:
        boundary = self.base.is_reset_boundary()
        task_valid = self._task_valid_rows()
        if type(boundary) is not bool or boundary is not True or any(task_valid):
            raise C211EnvError(
                "C211 canonical observation requires an explicit inactive reset boundary"
            )
        actor, critic = self.producer.tensors(task_valid)
        self._actor, self._critic = actor, critic
        digest = self.producer.boundary_state_sha256(actor, critic, task_valid)
        if initial:
            self._canonical_boundary_sha256 = _sha256_json(
                {
                    "schema_version": 1,
                    "kind": "action_ball_c211_seeded_wait_boundary_contract_v1",
                    "wait_schedule_sha256": (
                        self._wait_schedule.canonical_sha256
                    ),
                    "wait_preparation_sha256": (
                        getattr(
                            self.base,
                            "_continuous_wait_preparation_sha256",
                            self.base.diagnostic_training_identity()[
                                "reward_contract_sha256"
                            ],
                        )
                    ),
                    "observation_authority_sha256": (
                        self.producer.authorities.content_sha256
                    ),
                    "single_stroke_timeout_authority_sha256": (
                        self._single_stroke_timeout_authority[
                            "content_sha256"
                        ]
                    ),
                }
            )
            self._identity = self._build_identity(self._canonical_boundary_sha256)
        self._current_boundary_state_sha256 = digest

    def _build_identity(self, boundary_sha256: str) -> dict[str, str]:
        base = self.base.diagnostic_training_identity()
        observation = _sha256_json(
            {
                "schema_version": 1,
                "kind": "action_ball_c211_mujoco_observation_contract_v1",
                "profile_observation_contract_sha256": (
                    abi.C211_PROFILE.observation_contract_sha256
                ),
                "authority_sha256": self.producer.authorities.content_sha256,
                "canonical_reset_boundary_sha256": boundary_sha256,
                "actor_dtype": "torch.float32",
                "critic_dtype": "torch.float32",
                "device": "cpu",
                "diagnostic_unauthorized": True,
            }
        )
        contract = _sha256_json(
            {
                "schema_version": 1,
                "kind": "action_ball_c211_mujoco_partial_isaac_reward_contract_v3",
                "base_contract_sha256": base["contract_sha256"],
                "observation_contract_sha256": observation,
                "action_contract_sha256": base["action_contract_sha256"],
                "wrapped_reward_contract_sha256_not_consumed": base[
                    "reward_contract_sha256"
                ],
                "reward_contract_sha256": self._reward_contract[
                    "content_sha256"
                ],
                "reward_scope": C211_REWARD_SCOPE,
                "c211_achieved_outcome_reward_available": True,
                "isaac_synonymous_prior_subset_available": True,
                "complete_isaac_reward_parity_claimed": False,
                "actor_normalizer_identity": (
                    abi.C211_PROFILE.actor_normalizer_identity
                ),
                "critic_normalizer_identity": (
                    abi.C211_PROFILE.critic_normalizer_identity
                ),
                "single_stroke_timeout_authority_sha256": (
                    self._single_stroke_timeout_authority[
                        "content_sha256"
                    ]
                ),
                "timeout_bootstrap_rule": trainer.TIMEOUT_BOOTSTRAP_RULE,
                "diagnostic_unauthorized": True,
                "formal_authorized": False,
            }
        )
        return {
            "contract_sha256": contract,
            "observation_contract_sha256": observation,
            "action_contract_sha256": base["action_contract_sha256"],
            "reward_contract_sha256": self._reward_contract["content_sha256"],
        }

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.base.reset(seed=seed)
        current_wait_steps = getattr(
            self.base,
            "current_wait_steps",
            self.producer._wait_steps_by_env,
        )
        self.producer.set_episode_questions(
            self._native.questions, current_wait_steps
        )
        self._install_single_stroke_timeouts(
            range(self.num_envs), current_wait_steps
        )
        self.producer.reset_rows(range(self.num_envs))
        for state in self._reward_states:
            state.reset()
        self._install_current_boundary(initial=False)
        return self.get_observations()

    def get_observations(self) -> tuple[Any, dict[str, Any]]:
        if self._actor is None or self._critic is None:
            raise C211EnvError("C211 VecEnv has no valid current observation")
        task_valid = self._task_valid_rows()
        return self._actor.clone(), {
            "observations": {"critic": self._critic.clone()},
            "task_valid": list(task_valid),
            "observation_authorities": {
                "plant": self.producer.plant_observation_sha256,
                "mimic": self.producer.mimic.content_sha256,
                "task": self.producer.task.content_sha256,
                "composite": self.producer.authorities.content_sha256,
            },
            "canonical_reset_boundary_sha256": self._canonical_boundary_sha256,
            "current_reset_boundary_state_sha256": (
                self._current_boundary_state_sha256
            ),
            "physical_birth_reset_semantics": (
                self.producer.reset_birth_semantics
            ),
            "single_stroke_timeout_authority": copy.deepcopy(
                self._single_stroke_timeout_authority
            ),
            "safe_ready_authority_status": SAFE_READY_AUTHORITY_STATUS,
            "safe_ready_formal_pass_claimed": False,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise C211EnvError("torch is required for C211 stepping") from exc
        if (
            not isinstance(actions, torch.Tensor)
            or actions.device.type != "cpu"
            or tuple(actions.shape) != (self.num_envs, self.num_actions)
            or not torch.isfinite(actions).all()
        ):
            raise C211EnvError("C211 actions must be finite CPU [N,31]")
        actor_actions = actions.detach().cpu().numpy().copy()
        transition_valid = self._task_valid_rows()
        self.producer.begin_reward_transition(transition_valid, actor_actions)
        try:
            _base_obs, _base_rewards, dones, base_extras = self.base.step(actions)
            prior_rows = self.producer.finish_reward_transition()
        except Exception:
            self.producer.abort_reward_transition()
            self._actor = None
            self._critic = None
            raise
        try:
            rewards, reward_rows = self._evaluate_reward_transition(
                base_extras=base_extras,
                transition_valid=transition_valid,
                prior_rows=prior_rows,
                dones=dones,
            )
        except Exception:
            self._actor = None
            self._critic = None
            raise
        reset_ids = tuple(
            index for index, value in enumerate(dones.tolist()) if bool(value)
        )
        if reset_ids:
            current_wait_steps = getattr(
                self.base,
                "current_wait_steps",
                self.producer._wait_steps_by_env,
            )
            self.producer.set_episode_questions(
                self._native.questions, current_wait_steps
            )
            self._install_single_stroke_timeouts(
                reset_ids, current_wait_steps
            )
            self.producer.reset_rows(reset_ids)
        self.producer.set_previous_actions(actor_actions, reset_ids)
        task_valid = self._task_valid_rows()
        actor, critic = self.producer.tensors(task_valid)
        self._actor, self._critic = actor, critic
        if len(reset_ids) == self.num_envs:
            if self.base.is_reset_boundary() is not True or any(task_valid):
                self._actor = None
                self._critic = None
                raise C211EnvError(
                    "all-done compact reset lacks an explicit inactive boundary"
                )
            self._current_boundary_state_sha256 = (
                self.producer.boundary_state_sha256(actor, critic, task_valid)
            )
        extras = copy.deepcopy(base_extras)
        extras["fixed_center_reward_terms_not_consumed"] = extras.pop(
            "reward_terms"
        )
        extras["fixed_center_rewards_not_consumed"] = (
            _base_rewards.detach().cpu().clone()
        )
        extras["reward_terms"] = reward_rows
        extras["reward_contract"] = copy.deepcopy(self._reward_contract)
        extras["single_stroke_timeout_authority"] = copy.deepcopy(
            self._single_stroke_timeout_authority
        )
        extras["observations"] = {"critic": critic.clone()}
        # The underlying compact reset currently exposes only its legacy 76-D
        # pre-reset terminal row.  Do not relabel or pad it as C211.
        extras.pop("terminal_observations", None)
        extras["terminal_c211_observation_available"] = False
        extras["observation_authorities"] = {
            "plant": self.producer.plant_observation_sha256,
            "mimic": self.producer.mimic.content_sha256,
            "task": self.producer.task.content_sha256,
            "composite": self.producer.authorities.content_sha256,
        }
        extras["safe_ready_authority_status"] = SAFE_READY_AUTHORITY_STATUS
        extras["safe_ready_formal_pass_claimed"] = False
        extras["physical_birth_reset_semantics"] = (
            self.producer.reset_birth_semantics
        )
        extras["reward_scope"] = C211_REWARD_SCOPE
        extras["c211_achieved_outcome_reward_available"] = True
        extras["isaac_synonymous_prior_subset_available"] = True
        extras["complete_isaac_reward_parity_claimed"] = False
        extras["cross_engine_reward_semantic_gaps"] = [
            dict(row) for row in C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
        ]
        extras["true_c211_training_lane_ready"] = False
        extras["diagnostic_unauthorized"] = True
        extras["formal_authorized"] = False
        try:
            self._accumulate_reward_audit(reward_rows, dones)
        except Exception:
            self._actor = None
            self._critic = None
            raise
        return actor.clone(), rewards, dones, extras

    def diagnostic_training_identity(self) -> dict[str, str]:
        if self._identity is None:
            raise C211EnvError("C211 identity was not constructed")
        return dict(self._identity)

    def diagnostic_training_receipt(self) -> dict[str, Any]:
        base = self.base.diagnostic_training_receipt()
        if base.get("ppo_ready") is not True:
            raise C211EnvError("fixed-center base is not diagnostic PPO-ready")
        terminal_contract = trainer.terminal_row_telemetry_contract()
        if (
            base.get("terminal_row_telemetry_available") is not True
            or base.get("terminal_row_telemetry_contract") != terminal_contract
        ):
            raise C211EnvError(
                "fixed-center base omits exact terminal-row telemetry"
            )
        identity = self.diagnostic_training_identity()
        payload = {
            "schema_version": 1,
            "kind": trainer.DIAGNOSTIC_TRAINER_RECEIPT_KIND,
            "ppo_ready": True,
            "reward_available": True,
            "reward_scope": C211_REWARD_SCOPE,
            "c211_achieved_outcome_reward_available": True,
            "isaac_synonymous_prior_subset_available": True,
            "complete_isaac_reward_parity_claimed": False,
            "true_c211_training_lane_ready": False,
            "normal_step_available": True,
            "reset_boundary_checkpoint_available": True,
            "terminal_row_telemetry_available": True,
            "terminal_row_telemetry_contract": copy.deepcopy(
                terminal_contract
            ),
            "execution_resource_contract": {
                "mujoco_vecenv": "cpu_sequential",
                "execute_env_cap": 64,
                "torch_device": "cpu",
                "cuda_or_gpu_execution_used": False,
                "pod_gpu_assignment_consumed": False,
                "functional_canary_may_colocate_with_isaac_gpu_runs": True,
                "colocated_wall_time_is_speed_evidence": False,
            },
            **identity,
            "actor_width": abi.ACTOR_WIDTH,
            "critic_width": abi.CRITIC_WIDTH,
            "actor_normalizer_identity": abi.C211_PROFILE.actor_normalizer_identity,
            "critic_normalizer_identity": abi.C211_PROFILE.critic_normalizer_identity,
            "normalizer_binding": trainer.asymmetric_normalizer_binding(
                profile_observation_contract_sha256=(
                    abi.C211_PROFILE.observation_contract_sha256
                ),
                actor_width=abi.ACTOR_WIDTH,
                critic_width=abi.CRITIC_WIDTH,
                actor_normalizer_identity=(
                    abi.C211_PROFILE.actor_normalizer_identity
                ),
                critic_normalizer_identity=(
                    abi.C211_PROFILE.critic_normalizer_identity
                ),
                actor_task_mask_indices=(
                    abi.C211_PROFILE.actor.task_mask_indices
                ),
                critic_task_mask_indices=(
                    abi.C211_PROFILE.critic.task_mask_indices
                ),
                actor_task_valid_index=(
                    abi.C211_PROFILE.actor.task_valid_index
                ),
                critic_task_valid_index=(
                    abi.C211_PROFILE.critic.task_valid_index
                ),
                epsilon=1.0e-5,
            ),
            "fresh_actor_bootstrap": self.fresh_actor_bootstrap_contract(),
            "observation_authorities_sha256": self.producer.authorities.content_sha256,
            "reward_contract": copy.deepcopy(self._reward_contract),
            "canonical_reset_boundary_sha256": self._canonical_boundary_sha256,
            "single_stroke_timeout_authority": copy.deepcopy(
                self._single_stroke_timeout_authority
            ),
            "single_stroke_timeout_available": True,
            "single_stroke_timeout_bootstrap_rule": (
                trainer.TIMEOUT_BOOTSTRAP_RULE
            ),
            "time_to_contact_observation_semantics": (
                "signed_unclamped_deadline_matching_Isaac"
            ),
            "physical_birth_reset_semantics": (
                self.producer.reset_birth_semantics
            ),
            "teacher_reference_semantics": (
                "independent_measured_frame0_revealed_after_hidden_wait"
            ),
            "full_body_measured_mimic_observation_available": True,
            "full_body_measured_mimic_reward_available": True,
            "measured_paddle_prior_reward_available": True,
            "reward_parity_status": "partial_fail_closed",
            "cross_engine_reward_semantic_gaps": [
                dict(row) for row in C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
            ],
            "unavailable_isaac_reward_terms": [
                dict(row) for row in C211_UNAVAILABLE_ISAAC_REWARD_TERMS
            ],
            "terminal_c211_observation_available": False,
            "safe_ready_authority_status": SAFE_READY_AUTHORITY_STATUS,
            "safe_ready_formal_pass_claimed": False,
            "blockers": [],
            "formal_blockers": list(FORMAL_BLOCKERS),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
            "authorization": {
                "formal_training": False,
                "promotion": False,
                "deployment": False,
                "hardware": False,
            },
        }
        return payload

    def checkpoint_state(self) -> dict[str, Any]:
        if not self.is_reset_boundary():
            raise C211EnvError("C211 WAIT checkpoint requires a reset boundary")
        base_checkpoint = getattr(self.base, "checkpoint_state", None)
        payload = {
            "schema_version": 1,
            "kind": "action_ball_c211_mujoco_wait_boundary_state_v1",
            "identity": self.diagnostic_training_identity(),
            "base_wait_state": (
                base_checkpoint() if callable(base_checkpoint) else None
            ),
            "boundary_contract_sha256": self._canonical_boundary_sha256,
            "boundary_state_sha256": self._current_boundary_state_sha256,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def load_checkpoint_state(self, state: Any) -> None:
        # A failed sealed-state restore deliberately invalidates the derived
        # actor/critic arrays.  The checkpoint loader must still be able to
        # transactionally roll the environment back while the wrapped base is
        # at its intact reset boundary.
        if (
            not self.is_reset_boundary()
            and self.base.is_reset_boundary() is not True
        ):
            raise C211EnvError("C211 WAIT checkpoint load requires a reset boundary")
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "kind",
            "identity",
            "base_wait_state",
            "boundary_contract_sha256",
            "boundary_state_sha256",
            "content_sha256",
        }:
            raise C211EnvError("C211 WAIT checkpoint schema differs")
        payload = dict(state)
        declared = payload.pop("content_sha256")
        if (
            state["schema_version"] != 1
            or state["kind"] != "action_ball_c211_mujoco_wait_boundary_state_v1"
            or state["identity"] != self.diagnostic_training_identity()
            or state["boundary_contract_sha256"]
            != self._canonical_boundary_sha256
            or declared != _sha256_json(payload)
        ):
            raise C211EnvError("C211 WAIT checkpoint seal differs")
        base_loader = getattr(self.base, "load_checkpoint_state", None)
        if state["base_wait_state"] is not None:
            if not callable(base_loader):
                raise C211EnvError("C211 base cannot restore WAIT continuation")
            base_loader(state["base_wait_state"])
        current_wait_steps = getattr(
            self.base,
            "current_wait_steps",
            self.producer._wait_steps_by_env,
        )
        self.producer.set_episode_questions(
            self._native.questions, current_wait_steps
        )
        self._install_single_stroke_timeouts(
            range(self.num_envs), current_wait_steps
        )
        self.producer.reset_rows(range(self.num_envs))
        for reward_state in self._reward_states:
            reward_state.reset()
        self._install_current_boundary(initial=False)
        if self._current_boundary_state_sha256 != state["boundary_state_sha256"]:
            self._actor = None
            self._critic = None
            raise C211EnvError("restored C211 WAIT boundary state differs")

    def is_reset_boundary(self) -> bool:
        if (
            self._actor is None
            or self._critic is None
            or self.base.is_reset_boundary() is not True
        ):
            return False
        try:
            task_valid = self._task_valid_rows()
        except C211EnvError:
            return False
        if any(task_valid):
            return False
        if any(
            state.strike_sampled
            or state.outcome_evaluated
            or state.actual_contact_observed
            or state.selected_contact_observed
            or state.valid_achieved_flight_observed
            for state in self._reward_states
        ):
            return False
        return (
            self.producer.boundary_state_sha256(self._actor, self._critic, task_valid)
            == self._current_boundary_state_sha256
        )


__all__ = [
    "ANCHOR_BODY_NAME",
    "C211_ENV_KIND",
    "C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS",
    "C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES",
    "C211_REWARD_CONTRACT_IDENTITY",
    "C211_TARGET_RECIPE",
    "C211_REWARD_SCOPE",
    "C211_UNAVAILABLE_ISAAC_REWARD_TERMS",
    "FORMAL_BLOCKERS",
    "SAFE_READY_AUTHORITY_STATUS",
    "TRACKED_BODY_NAMES",
    "C211EnvError",
    "C211ObservationProducer",
    "C211TaskAuthority",
    "MeasuredC211MimicAuthority",
    "MujocoC211DiagnosticVecEnv",
]
