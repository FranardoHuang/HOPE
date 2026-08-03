"""Fixed-centre N1 diagnostic recipe layered over the native C-lite VecEnv.

This module intentionally does not claim the A211/C211 observation ABI or the
formal whole-body measured-motion reward.  It adds the narrow pieces that can
be derived from already sealed native authorities:

* an episode-local RESET_WAIT/TASK_ACTIVE state with atomic actor task masking;
* a frame-0 *joint-space* teacher reward loaded from the fixed tape lineage;
* a bounded pelvis upright/height balance reward around physical-ready; and
* the existing selected-rubber strike-distance and observed-outcome task terms.

The fixed question is installed once at reset and is never inverse-solved or
teleported at reveal.  A hidden racket contact/outcome before reveal is a hard
error, because the cumulative native event ABI cannot relabel it as active.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import n1_reward_event_kernel
from . import n1_ball_core
from . import n1_scalar_reward
from . import observed_outcome_resolver
from . import single_env
from . import trainer
from . import vec_env


RECIPE_KIND = "a3_mujoco_n1_fixed_center_diagnostic_recipe_v1"
RECIPE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
READINESS_KIND = trainer.DIAGNOSTIC_TRAINER_RECEIPT_KIND
TASK_SLICE = slice(62, 76)
FORMAL_BLOCKERS = (
    "fixed_center_76d_is_not_a211_c211_211d_319d_abi",
    "frame0_joint_teacher_is_not_full_body_measured_mimic",
    "physical_ready_200_tick_live_hold_receipt_not_bound",
    "reset_wait_is_diagnostic_outer_parent_not_cross_engine_parity",
    "continuous_native_gravity_wait_is_not_cross_engine_launch_parity",
    "fixed_question_freezes_curriculum_and_has_no_banded_question_bank",
    "formal_phase_recovery_export_and_mid_episode_resume_not_closed",
    "cpu_sequential_vecenv_has_no_4096_matched_workload_receipt",
)


class FixedCenterRecipeError(ValueError):
    """The diagnostic recipe or its source-bound runtime facts are invalid."""


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FixedCenterRecipeError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise FixedCenterRecipeError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class FixedCenterRecipeSpec:
    """Explicit engineering priors for the non-formal fixed-centre recipe."""

    reset_wait_steps: int = 1
    joint_position_scale_rad: float = 0.5
    joint_velocity_scale_radps: float = 2.0
    motion_reward_weight: float = 0.25
    pelvis_height_scale_m: float = 0.1
    pelvis_up_scale: float = 0.1
    balance_reward_weight: float = 0.1

    def __post_init__(self) -> None:
        if type(self.reset_wait_steps) is not int or self.reset_wait_steps < 1:
            raise FixedCenterRecipeError(
                "reset_wait_steps must be a positive plain integer"
            )
        for name in (
            "joint_position_scale_rad",
            "joint_velocity_scale_radps",
            "motion_reward_weight",
            "pelvis_height_scale_m",
            "pelvis_up_scale",
            "balance_reward_weight",
        ):
            _finite_positive(getattr(self, name), name)

    @property
    def content_sha256(self) -> str:
        return _sha256_json({"schema_version": 1, "kind": RECIPE_KIND, **asdict(self)})


@dataclass(frozen=True)
class Frame0JointTeacher:
    """Sealed frame-0 joint reference plus physical-ready balance anchor."""

    joint_pos: tuple[float, ...]
    pelvis_height_m: float
    source_motion_sha256: str
    source_motion_uid: str
    source_frame_index: int
    hold_candidate_content_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.joint_pos, dtype=np.float64)
        if values.shape != (single_env.ACTION_DIM,) or not np.isfinite(values).all():
            raise FixedCenterRecipeError("frame0 teacher joint_pos must be finite 31-D")
        _finite_positive(self.pelvis_height_m, "pelvis_height_m")
        for name in ("source_motion_sha256", "hold_candidate_content_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise FixedCenterRecipeError(f"{name} must be lowercase SHA-256")
        if not isinstance(self.source_motion_uid, str) or not self.source_motion_uid:
            raise FixedCenterRecipeError("source_motion_uid must be non-empty")
        if type(self.source_frame_index) is not int or self.source_frame_index < 0:
            raise FixedCenterRecipeError(
                "source_frame_index must be a non-negative integer"
            )

    @property
    def content_sha256(self) -> str:
        return _sha256_json(
            {
                "schema_version": 1,
                "kind": "a3_mujoco_frame0_joint_teacher_v1",
                "joint_pos": list(self.joint_pos),
                "joint_vel": [0.0] * single_env.ACTION_DIM,
                "pelvis_height_m": self.pelvis_height_m,
                "source_motion_sha256": self.source_motion_sha256,
                "source_motion_uid": self.source_motion_uid,
                "source_frame_index": self.source_frame_index,
                "hold_candidate_content_sha256": (self.hold_candidate_content_sha256),
                "velocity_semantics": "frame0_reference_forced_exact_zero",
            }
        )

    @classmethod
    def from_fixed_tape(
        cls, binding: single_env.PlantBinding, tape: single_env.FixedTape
    ) -> "Frame0JointTeacher":
        reset = tape.reset_state
        if reset.mode != "action_specific_hold":
            raise FixedCenterRecipeError(
                "fixed-center recipe requires action_specific_hold physical-ready"
            )
        if (
            reset.source_motion_path is None
            or reset.source_motion_sha256 is None
            or reset.source_motion_uid is None
            or reset.source_frame_index is None
            or reset.root_pos is None
            or reset.hold_candidate_content_sha256 is None
        ):
            raise FixedCenterRecipeError(
                "action-specific hold omits teacher/physical-ready lineage"
            )
        try:
            mapping, _center = single_env._teacher_frame_reset_payload(
                binding, reset.source_motion_path, reset.source_frame_index
            )
        except single_env.ContractError as exc:
            raise FixedCenterRecipeError(
                "frame0 teacher cannot be replayed from fixed-tape lineage"
            ) from exc
        if (
            mapping["source_motion_sha256"] != reset.source_motion_sha256
            or mapping["source_motion_uid"] != reset.source_motion_uid
            or mapping["source_frame_index"] != reset.source_frame_index
        ):
            raise FixedCenterRecipeError(
                "frame0 teacher replay differs from fixed-tape lineage"
            )
        return cls(
            joint_pos=tuple(float(value) for value in mapping["joint_pos"]),
            pelvis_height_m=float(reset.root_pos[2]),
            source_motion_sha256=reset.source_motion_sha256,
            source_motion_uid=reset.source_motion_uid,
            source_frame_index=reset.source_frame_index,
            hold_candidate_content_sha256=reset.hold_candidate_content_sha256,
        )


@dataclass(frozen=True)
class ContinuousWaitPreparation:
    """Content seal for a reset-to-reveal contact-free ballistic prefix."""

    spec_sha256: str
    wait_policy_steps: int
    wait_physics_substeps: int
    physics_step_dt_s: float
    per_env: tuple[Mapping[str, Any], ...]

    @property
    def content_sha256(self) -> str:
        return _sha256_json(
            {
                "schema_version": 1,
                "kind": "a3_mujoco_fixed_center_continuous_wait_preparation_v1",
                "spec_sha256": self.spec_sha256,
                "wait_policy_steps": self.wait_policy_steps,
                "wait_physics_substeps": self.wait_physics_substeps,
                "physics_step_dt_s": self.physics_step_dt_s,
                "per_env": [dict(row) for row in self.per_env],
                "integrator": "mujoco_semi_implicit_euler_contact_free_ballistic",
                "reveal_mutation": "none",
            }
        )


def _reverse_euler_ballistic_prefix(
    *,
    reveal_position: np.ndarray,
    reveal_velocity: np.ndarray,
    gravity: np.ndarray,
    physics_dt: float,
    substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    reset_velocity = reveal_velocity - gravity * (substeps * physics_dt)
    reset_position = (
        reveal_position
        - (substeps * physics_dt) * reset_velocity
        - gravity * (physics_dt**2) * (substeps * (substeps + 1) / 2.0)
    )
    return reset_position, reset_velocity


def prepare_continuous_wait_base(
    base_env: vec_env.MujocoN1DiagnosticVecEnv,
    spec: FixedCenterRecipeSpec,
) -> vec_env.MujocoN1DiagnosticVecEnv:
    """Rebuild a C-lite base with a physical trajectory that includes WAIT.

    Each original question birth state becomes the desired reveal state.  The
    derived reset position/velocity are the exact inverse of contact-free
    MuJoCo semi-implicit Euler gravity for the fixed number of WAIT substeps.
    No simulator state is changed at reveal.
    """

    if not base_env.c_lite_reward_enabled:
        raise FixedCenterRecipeError("continuous WAIT requires a C-lite base")
    if len(base_env.cores) != base_env.num_envs or len(base_env.questions) != (
        base_env.num_envs
    ):
        raise FixedCenterRecipeError("base core/question cardinality differs")
    control_decimation = int(base_env.control_decimation)
    physics_dt = float(base_env.step_dt) / control_decimation
    wait_substeps = spec.reset_wait_steps * control_decimation
    wait_s = wait_substeps * physics_dt
    questions = []
    rows = []
    for index, (core, question) in enumerate(zip(base_env.cores, base_env.questions)):
        integrator = int(core.model.opt.integrator)
        if integrator != 0:
            raise FixedCenterRecipeError(
                f"core {index} must use MuJoCo semi-implicit Euler integrator"
            )
        model_dt = float(core.model.opt.timestep)
        gravity = np.asarray(core.model.opt.gravity, dtype=np.float64)
        wind = np.asarray(core.model.opt.wind, dtype=np.float64)
        density = float(core.model.opt.density)
        viscosity = float(core.model.opt.viscosity)
        ball_dof = int(core.scene.ball_dof_adr)
        ball_body = int(core.scene.ball_body_id)
        ball_damping = np.asarray(
            core.model.dof_damping[ball_dof : ball_dof + 6], dtype=np.float64
        )
        ball_gravcomp = float(core.model.body_gravcomp[ball_body])
        ball_qfrc_applied = np.asarray(
            core.data.qfrc_applied[ball_dof : ball_dof + 6], dtype=np.float64
        )
        ball_xfrc_applied = np.asarray(
            core.data.xfrc_applied[ball_body], dtype=np.float64
        )
        if (
            not math.isclose(model_dt, physics_dt, rel_tol=0.0, abs_tol=1.0e-15)
            or gravity.shape != (3,)
            or not np.isfinite(gravity).all()
            or wind.shape != (3,)
            or not np.array_equal(wind, np.zeros(3))
            or density != 0.0
            or viscosity != 0.0
            or ball_damping.shape != (6,)
            or not np.array_equal(ball_damping, np.zeros(6))
            or ball_gravcomp != 0.0
            or ball_qfrc_applied.shape != (6,)
            or not np.array_equal(ball_qfrc_applied, np.zeros(6))
            or ball_xfrc_applied.shape != (6,)
            or not np.array_equal(ball_xfrc_applied, np.zeros(6))
        ):
            raise FixedCenterRecipeError(
                f"core {index} is not a pure-gravity undamped WAIT trajectory"
            )
        reveal_position = np.asarray(question.birth_position_w_m, dtype=np.float64)
        reveal_velocity = np.asarray(
            question.birth_linear_velocity_w_mps, dtype=np.float64
        )
        reset_position, reset_velocity = _reverse_euler_ballistic_prefix(
            reveal_position=reveal_position,
            reveal_velocity=reveal_velocity,
            gravity=gravity,
            physics_dt=physics_dt,
            substeps=wait_substeps,
        )
        row = {
            "env_index": index,
            "parent_question_source_sha256": question.source_sha256,
            "wait_s": wait_s,
            "gravity_w_mps2": gravity.tolist(),
            "wind_w_mps": wind.tolist(),
            "fluid_density": density,
            "fluid_viscosity": viscosity,
            "ball_dof_damping": ball_damping.tolist(),
            "ball_body_gravcomp": ball_gravcomp,
            "ball_qfrc_applied": ball_qfrc_applied.tolist(),
            "ball_xfrc_applied": ball_xfrc_applied.tolist(),
            "reset_ball_position_w_m": reset_position.tolist(),
            "reset_ball_linear_velocity_w_mps": reset_velocity.tolist(),
            "reveal_ball_position_w_m": reveal_position.tolist(),
            "reveal_ball_linear_velocity_w_mps": reveal_velocity.tolist(),
            "parent_nominal_time_to_contact_s": (question.nominal_time_to_contact_s),
            "derived_nominal_time_to_contact_s": (
                question.nominal_time_to_contact_s + wait_s
            ),
        }
        launch_content_sha = _sha256_json(
            {
                "schema_version": 1,
                "kind": "a3_mujoco_wait_extended_question_v1",
                "spec_sha256": spec.content_sha256,
                **row,
            }
        )
        row["launch_content_sha256"] = launch_content_sha
        rows.append(row)
        reset_position.setflags(write=False)
        reset_velocity.setflags(write=False)
        spin = np.asarray(question.birth_spin_w_radps, dtype=np.float64).copy()
        aim = np.asarray(question.landing_aim_xy_w_m, dtype=np.float64).copy()
        spin.setflags(write=False)
        aim.setflags(write=False)
        questions.append(
            n1_ball_core.N1Question(
                source_path=question.source_path,
                source_sha256=question.source_sha256,
                question_id=question.question_id,
                scene_binding_sha256=question.scene_binding_sha256,
                birth_position_w_m=reset_position,
                birth_linear_velocity_w_mps=reset_velocity,
                birth_spin_w_radps=spin,
                landing_aim_xy_w_m=aim,
                nominal_time_to_contact_s=(question.nominal_time_to_contact_s + wait_s),
                spin_valid=question.spin_valid,
                authority=copy.deepcopy(question.authority),
                selected_rubber_action_lineage=copy.deepcopy(
                    question.selected_rubber_action_lineage
                ),
            )
        )
    preparation = ContinuousWaitPreparation(
        spec_sha256=spec.content_sha256,
        wait_policy_steps=spec.reset_wait_steps,
        wait_physics_substeps=wait_substeps,
        physics_step_dt_s=physics_dt,
        per_env=tuple(rows),
    )
    rebuilt = vec_env.MujocoN1DiagnosticVecEnv(
        cores=base_env.cores,
        robot_tape=base_env.robot_tape,
        questions=tuple(questions),
        device="cpu",
        enable_c_lite_reward=True,
        diagnostic_episode_length=base_env.max_episode_length,
    )
    rebuilt._fixed_center_continuous_wait_preparation = preparation
    rebuilt._fixed_center_parent_identity = rebuilt.diagnostic_training_identity()
    rebuilt._fixed_center_parent_readiness = rebuilt.diagnostic_training_receipt()
    rebuilt._fixed_center_requires_outer_wrapper = True
    return rebuilt


@dataclass
class _EpisodeState:
    reset_wait_steps: int
    wait_remaining: int = 0
    strike_sampled: bool = False

    def reset(self) -> None:
        self.wait_remaining = self.reset_wait_steps
        self.strike_sampled = False

    @property
    def task_valid(self) -> bool:
        return self.wait_remaining == 0

    def advance(self) -> bool:
        """Advance one transition and report an atomic WAIT->ACTIVE reveal."""

        if self.wait_remaining == 0:
            return False
        self.wait_remaining -= 1
        return self.wait_remaining == 0


def _bounded_cauchy(error: np.ndarray, scale: float) -> float:
    ratio_sq = np.square(error / scale)
    return float(np.mean(1.0 / (1.0 + ratio_sq)))


def _joint_motion_and_balance(
    *,
    observation_row: np.ndarray,
    ledger: Mapping[str, Any],
    teacher_reference: Frame0JointTeacher,
    spec: FixedCenterRecipeSpec,
) -> tuple[float, float, dict[str, float]]:
    row = np.asarray(observation_row, dtype=np.float64)
    if row.shape != (vec_env.OBSERVATION_WIDTH,) or not np.isfinite(row).all():
        raise FixedCenterRecipeError("pre-reset observation row must be finite 76-D")
    q = row[: single_env.ACTION_DIM]
    qd = row[single_env.ACTION_DIM : 2 * single_env.ACTION_DIM]
    qref = np.asarray(teacher_reference.joint_pos, dtype=np.float64)
    position_kernel = _bounded_cauchy(q - qref, float(spec.joint_position_scale_rad))
    velocity_kernel = _bounded_cauchy(qd, float(spec.joint_velocity_scale_radps))
    motion = (
        float(spec.motion_reward_weight) * 0.5 * (position_kernel + velocity_kernel)
    )

    samples = ledger.get("latest_pelvis_samples")
    if not isinstance(samples, Mapping):
        raise FixedCenterRecipeError("ledger has no pelvis balance samples")
    height = float(samples.get("height_m", math.nan))
    up_z = float(samples.get("up_world_z", math.nan))
    if not math.isfinite(height) or not math.isfinite(up_z) or not -1.0 <= up_z <= 1.0:
        raise FixedCenterRecipeError("pelvis balance samples are invalid")
    height_kernel = 1.0 / (
        1.0
        + (
            (height - teacher_reference.pelvis_height_m)
            / float(spec.pelvis_height_scale_m)
        )
        ** 2
    )
    upright_kernel = 1.0 / (1.0 + ((1.0 - up_z) / float(spec.pelvis_up_scale)) ** 2)
    balance = float(spec.balance_reward_weight) * 0.5 * (height_kernel + upright_kernel)
    return (
        motion,
        balance,
        {
            "joint_position_kernel": position_kernel,
            "joint_velocity_zero_kernel": velocity_kernel,
            "pelvis_height_kernel": height_kernel,
            "pelvis_upright_kernel": upright_kernel,
        },
    )


def _hidden_task_facts_are_clean(
    native_facts: Mapping[str, Any], new_events: Sequence[Mapping[str, Any]]
) -> None:
    snapshot = native_facts.get("observed_outcome_snapshot")
    if not isinstance(snapshot, Mapping):
        raise FixedCenterRecipeError("WAIT reveal has no observed-outcome snapshot")
    if (
        len(new_events) != 0
        or native_facts.get("racket_contact_edge_count_total") != 0
        or native_facts.get("first_racket_contact_stamp") is not None
        or native_facts.get("outgoing_flight") is not None
        or snapshot.get("outcome_resolved") is not False
    ):
        raise FixedCenterRecipeError(
            "hidden task produced contact/outcome before atomic reveal"
        )


def _masked_task_observations(observations: Any, task_valid: Sequence[bool]) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise FixedCenterRecipeError(
            "torch is required for fixed-center VecEnv"
        ) from exc
    if (
        not isinstance(observations, torch.Tensor)
        or observations.ndim != 2
        or observations.shape[1] != vec_env.OBSERVATION_WIDTH
        or observations.shape[0] != len(task_valid)
    ):
        raise FixedCenterRecipeError("observation tensor shape differs from 76-D ABI")
    result = observations.clone()
    for index, valid in enumerate(task_valid):
        if type(valid) is not bool:
            raise FixedCenterRecipeError("task_valid must contain plain booleans")
        if not valid:
            result[index, TASK_SLICE] = 0.0
    return result


class FixedCenterDiagnosticVecEnv:
    """RESET_WAIT + frame0-joint/balance wrapper around a real C-lite VecEnv."""

    def __init__(
        self,
        *,
        base_env: vec_env.MujocoN1DiagnosticVecEnv,
        teacher_reference: Frame0JointTeacher,
        spec: FixedCenterRecipeSpec = FixedCenterRecipeSpec(),
    ) -> None:
        self.base_env = base_env
        self.teacher_reference = teacher_reference
        self.spec = spec
        self.num_envs = int(base_env.num_envs)
        self.num_observations = int(base_env.num_observations)
        self.num_actions = int(base_env.num_actions)
        self.device = base_env.device
        self.unwrapped = self
        preparation = getattr(
            base_env, "_fixed_center_continuous_wait_preparation", None
        )
        if not isinstance(preparation, ContinuousWaitPreparation):
            raise FixedCenterRecipeError(
                "base VecEnv must be rebuilt with continuous WAIT preparation"
            )
        if (
            preparation.spec_sha256 != spec.content_sha256
            or preparation.wait_policy_steps != spec.reset_wait_steps
            or len(preparation.per_env) != self.num_envs
        ):
            raise FixedCenterRecipeError(
                "continuous WAIT preparation differs from recipe spec"
            )
        self.continuous_wait_preparation = preparation
        self._continuous_wait_preparation_sha256 = preparation.content_sha256
        if not base_env.c_lite_reward_enabled:
            raise FixedCenterRecipeError(
                "base VecEnv must enable exact C-lite task facts"
            )
        if self.num_observations != vec_env.OBSERVATION_WIDTH:
            raise FixedCenterRecipeError(
                "fixed-center diagnostic requires 76-D base ABI"
            )
        if spec.reset_wait_steps >= base_env.max_episode_length:
            raise FixedCenterRecipeError(
                "RESET_WAIT must leave at least one TASK_ACTIVE transition"
            )
        self._states = [
            _EpisodeState(spec.reset_wait_steps) for _ in range(self.num_envs)
        ]
        nominal_ticks = []
        for index, question in enumerate(base_env.questions):
            tick_float = float(question.nominal_time_to_contact_s) / float(
                base_env.step_dt
            )
            tick = int(round(tick_float))
            if tick < 1 or not math.isclose(
                tick_float, float(tick), rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise FixedCenterRecipeError(
                    f"question {index} nominal strike is not one exact policy tick"
                )
            nominal_ticks.append(tick)
        self._nominal_strike_ticks = tuple(nominal_ticks)
        self._reward_latches = [
            n1_scalar_reward.N1ScalarRewardLatch() for _ in range(self.num_envs)
        ]
        self._observations = None
        self._last_transition_task_valid = tuple(False for _ in self._states)
        self.cfg = {
            **copy.deepcopy(base_env.cfg),
            "kind": "a3_mujoco_n1_fixed_center_diagnostic_vecenv_v1",
            "reward_scope": "fixed_center_joint_teacher_learnability_diagnostic",
            "reset_wait_steps": spec.reset_wait_steps,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        self._reward_receipt = self._build_reward_receipt()
        self._identity = self._build_identity()
        self.reset()

    def _validate_preparation_integrity(self) -> None:
        preparation = self.continuous_wait_preparation
        if preparation.content_sha256 != self._continuous_wait_preparation_sha256:
            raise FixedCenterRecipeError(
                "continuous WAIT preparation mutated after identity construction"
            )
        for index, (question, row) in enumerate(
            zip(self.base_env.questions, preparation.per_env)
        ):
            if (
                question.source_sha256 != row["parent_question_source_sha256"]
                or not np.array_equal(
                    np.asarray(question.birth_position_w_m, dtype=np.float64),
                    np.asarray(row["reset_ball_position_w_m"], dtype=np.float64),
                )
                or not np.array_equal(
                    np.asarray(question.birth_linear_velocity_w_mps, dtype=np.float64),
                    np.asarray(
                        row["reset_ball_linear_velocity_w_mps"], dtype=np.float64
                    ),
                )
                or not math.isclose(
                    float(question.nominal_time_to_contact_s),
                    float(row["derived_nominal_time_to_contact_s"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise FixedCenterRecipeError(
                    f"continuous WAIT question {index} mutated after preparation"
                )

    def _build_reward_receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "kind": RECIPE_KIND,
            "scope": "fixed_center_joint_teacher_learnability_diagnostic",
            "recipe_source_sha256": RECIPE_SOURCE_SHA256,
            "spec": asdict(self.spec),
            "spec_sha256": self.spec.content_sha256,
            "teacher_reference_sha256": self.teacher_reference.content_sha256,
            "wait": {
                "states": ["RESET_WAIT", "TASK_ACTIVE"],
                "task_valid_semantics": "WAIT=0,TASK_ACTIVE=1",
                "hidden_actor_slice": [TASK_SLICE.start, TASK_SLICE.stop],
                "hidden_groups": [
                    "incoming_ball_position_w_m",
                    "incoming_ball_linear_velocity_w_mps",
                    "incoming_ball_spin_w_radps",
                    "landing_aim_xy_w_m",
                    "time_to_contact_s",
                    "validity",
                ],
                "reward_mask": (
                    "WAIT masks strike distance, contact, closure and outcome; "
                    "joint teacher/balance remain active"
                ),
                "reveal_mutation": "actor_observation_only_no_simulator_write",
                "root_joint_ball_teleport_on_reveal": False,
                "physical_ball_parked_during_wait": False,
                "physical_ball_trajectory_includes_wait_from_reset": True,
                "preparation_sha256": (self.continuous_wait_preparation.content_sha256),
                "preparation": {
                    "integrator": "mujoco_semi_implicit_euler",
                    "wait_physics_substeps": (
                        self.continuous_wait_preparation.wait_physics_substeps
                    ),
                    "physics_step_dt_s": (
                        self.continuous_wait_preparation.physics_step_dt_s
                    ),
                    "per_env": [
                        dict(row) for row in self.continuous_wait_preparation.per_env
                    ],
                },
            },
            "motion": {
                "kind": "frame0_joint_position_plus_zero_velocity_cauchy",
                "full_body_measured_mimic": False,
                "max_per_step": self.spec.motion_reward_weight,
            },
            "balance": {
                "kind": "physical_ready_pelvis_height_plus_world_up_cauchy",
                "max_per_step": self.spec.balance_reward_weight,
            },
            "task": {
                "base_contract_sha256": self.base_env._c_lite_reward_receipt[
                    "content_sha256"
                ],
                "fixed_question": True,
                "online_inverse_solve_calls": 0,
                "contact_bonus": 0.0,
                "desired_contact_inverse": False,
                "strike_sampling_semantics": (
                    "exactly_one_post_control_policy_tick_at_derived_nominal_ttc"
                ),
                "strike_sample_pay_at_most_once_per_episode": True,
                "nominal_policy_tick_by_env": list(self._nominal_strike_ticks),
                "nominal_tick_reachable_by_env": [
                    tick <= self.base_env.max_episode_length
                    for tick in self._nominal_strike_ticks
                ],
                "c211_nominal_single_tick_subset": True,
            },
            "formal_blockers": list(FORMAL_BLOCKERS),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def _build_identity(self) -> dict[str, str]:
        base = getattr(self.base_env, "_fixed_center_parent_identity", None)
        if base is None:
            base = self.base_env.diagnostic_training_identity()
        observation_sha = _sha256_json(
            {
                "schema_version": 1,
                "kind": "a3_mujoco_fixed_center_masked_76d_observation_v1",
                "base_observation_contract_sha256": base["observation_contract_sha256"],
                "task_slice": [TASK_SLICE.start, TASK_SLICE.stop],
                "task_valid_semantics": "WAIT=0,TASK_ACTIVE=1",
                "not_a211_c211": True,
            }
        )
        contract_sha = _sha256_json(
            {
                "schema_version": 1,
                "kind": "a3_mujoco_fixed_center_diagnostic_training_contract_v1",
                "base_contract_sha256": base["contract_sha256"],
                "observation_contract_sha256": observation_sha,
                "action_contract_sha256": base["action_contract_sha256"],
                "reward_contract_sha256": self._reward_receipt["content_sha256"],
                "teacher_reference_sha256": self.teacher_reference.content_sha256,
                "spec_sha256": self.spec.content_sha256,
                "diagnostic_unauthorized": True,
                "formal_authorized": False,
            }
        )
        return {
            "contract_sha256": contract_sha,
            "observation_contract_sha256": observation_sha,
            "action_contract_sha256": base["action_contract_sha256"],
            "reward_contract_sha256": self._reward_receipt["content_sha256"],
        }

    @property
    def task_valid(self) -> tuple[bool, ...]:
        return tuple(state.task_valid for state in self._states)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self._validate_preparation_integrity()
        observations, extras = self.base_env.reset(seed=seed)
        for state in self._states:
            state.reset()
        self._reward_latches = [
            n1_scalar_reward.N1ScalarRewardLatch() for _ in range(self.num_envs)
        ]
        self._last_transition_task_valid = tuple(False for _ in self._states)
        self._observations = _masked_task_observations(observations, self.task_valid)
        extras = copy.deepcopy(extras)
        extras["observations"] = {"critic": self._observations.clone()}
        extras["reward_contract"] = copy.deepcopy(self._reward_receipt)
        extras["task_valid"] = list(self.task_valid)
        extras["diagnostic_unauthorized"] = True
        extras["formal_authorized"] = False
        return self._observations.clone(), extras

    def is_reset_boundary(self) -> bool:
        return self.base_env.is_reset_boundary()

    def diagnostic_training_identity(self) -> dict[str, str]:
        return dict(self._identity)

    def diagnostic_training_receipt(self) -> dict[str, Any]:
        base = getattr(self.base_env, "_fixed_center_parent_readiness", None)
        if base is None:
            base = self.base_env.diagnostic_training_receipt()
        if base.get("ppo_ready") is not True:
            raise FixedCenterRecipeError("base C-lite readiness is not available")
        return {
            "schema_version": 1,
            "kind": READINESS_KIND,
            "ppo_ready": True,
            "reward_available": True,
            "normal_step_available": True,
            "reset_boundary_checkpoint_available": True,
            **self.diagnostic_training_identity(),
            "reward_scope": "fixed_center_joint_teacher_learnability_diagnostic",
            "reset_wait_steps": self.spec.reset_wait_steps,
            "task_valid_available": True,
            "frame0_joint_teacher_available": True,
            "full_body_measured_mimic_available": False,
            "balance_reward_available": True,
            "fixed_question_online_inverse_solve_calls": 0,
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

    def get_observations(self) -> tuple[Any, dict[str, Any]]:
        if self._observations is None:
            raise FixedCenterRecipeError("fixed-center VecEnv is not reset")
        observations = self._observations.clone()
        return observations, {
            "observations": {"critic": observations.clone()},
            "reward_contract": copy.deepcopy(self._reward_receipt),
            "task_valid": list(self.task_valid),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }

    def _invalidate_after_failure(self) -> None:
        """Make an observed invalid transition impossible to continue from."""

        self._observations = None
        if hasattr(self.base_env, "_observations"):
            self.base_env._observations = None
        if hasattr(self.base_env, "_has_reset"):
            self.base_env._has_reset = False

    def _wait_eligibility(self) -> n1_reward_event_kernel.N1RewardEligibility:
        return n1_reward_event_kernel.N1RewardEligibility(
            motion_mimic_denominator=True,
            contact_target_denominator=False,
            closed_swing_denominator=False,
            actual_contact_numerator=False,
            achieved_outgoing_flight_denominator=False,
            predicted_outcome_denominator=False,
            predicted_net_clear_numerator=False,
            predicted_legal_landing_numerator=False,
            observed_outcome_denominator=False,
            observed_net_clear_numerator=False,
            observed_legal_landing_numerator=False,
            unresolved_achieved_flight=False,
            motion_mimic_pay_eligible=True,
            contact_target_pay_eligible=False,
            actual_contact_pay_eligible=False,
            predicted_outcome_pay_eligible=False,
            observed_outcome_pay_eligible=False,
        )

    def _validate_reveal_ball_state(self, index: int, observation_row: Any) -> None:
        expected = self.continuous_wait_preparation.per_env[index]
        expected_position = np.asarray(
            expected["reveal_ball_position_w_m"], dtype=np.float64
        )
        expected_velocity = np.asarray(
            expected["reveal_ball_linear_velocity_w_mps"], dtype=np.float64
        )
        if hasattr(self.base_env, "cores"):
            core = self.base_env.cores[index]
            qpos = core.scene.ball_qpos_adr
            dof = core.scene.ball_dof_adr
            actual_position = np.asarray(
                core.data.qpos[qpos : qpos + 3], dtype=np.float64
            )
            actual_velocity = np.asarray(
                core.data.qvel[dof : dof + 3], dtype=np.float64
            )
            tolerance = 1.0e-9
        else:
            row = np.asarray(observation_row, dtype=np.float64)
            if row.shape != (vec_env.OBSERVATION_WIDTH,):
                raise FixedCenterRecipeError(
                    f"env {index} reveal observation shape differs"
                )
            actual_position = row[62:65]
            actual_velocity = row[65:68]
            tolerance = 1.0e-6
        if not np.allclose(
            actual_position, expected_position, rtol=0.0, atol=tolerance
        ) or not np.allclose(
            actual_velocity, expected_velocity, rtol=0.0, atol=tolerance
        ):
            raise FixedCenterRecipeError(
                f"env {index} continuous WAIT did not reach sealed reveal ball state"
            )

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise FixedCenterRecipeError(
                "torch is required for fixed-center VecEnv"
            ) from exc
        if self._observations is None:
            raise FixedCenterRecipeError(
                "fixed-center VecEnv must be reset after a failed transition"
            )
        self._validate_preparation_integrity()
        transition_valid = self.task_valid
        self._last_transition_task_valid = transition_valid
        batch = self.base_env.diagnostic_step(actions)
        try:
            for index, valid in enumerate(transition_valid):
                if not valid:
                    _hidden_task_facts_are_clean(
                        batch.per_env_native_physical_event_facts[index],
                        batch.per_env_events[index],
                    )
        except Exception:
            self._invalidate_after_failure()
            raise
        outputs = []
        next_latches = list(self._reward_latches)
        reward_rows = []
        for index in range(self.num_envs):
            native_facts = batch.per_env_native_physical_event_facts[index]
            physical = batch.per_env_c_lite_physical_samples[index]
            if native_facts is None or physical is None:
                raise FixedCenterRecipeError(
                    "fixed-center transition omits native C task facts"
                )
            motion, balance, kernels = _joint_motion_and_balance(
                observation_row=(
                    batch.terminal_observations[index].detach().cpu().numpy()
                ),
                ledger=batch.per_env_ledgers[index],
                teacher_reference=self.teacher_reference,
                spec=self.spec,
            )
            policy_tick = int(native_facts["policy_tick"])
            sample_time_s = float(physical.get("sample_time_s", math.nan))
            expected_sample_time_s = policy_tick * float(self.base_env.step_dt)
            try:
                if not math.isclose(
                    sample_time_s,
                    expected_sample_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise FixedCenterRecipeError(
                        f"env {index} physical sample time differs from completed tick"
                    )
            except Exception:
                self._invalidate_after_failure()
                raise
            nominal_clock_error_s = sample_time_s - float(
                self.base_env.questions[index].nominal_time_to_contact_s
            )
            if transition_valid[index]:
                strike_sample = policy_tick == self._nominal_strike_ticks[index]
                if strike_sample and self._states[index].strike_sampled:
                    raise FixedCenterRecipeError(
                        f"env {index} nominal strike distance sampled twice"
                    )
                physical_for_reward = dict(physical)
                physical_for_reward["miss_sample_eligible"] = strike_sample
                eligibility = self.base_env._c_lite_event_eligibility(
                    index=index,
                    native_facts=native_facts,
                    physical_sample=physical_for_reward,
                    episode_done=bool(batch.episode_dones[index].item()),
                    time_out=bool(batch.time_outs[index].item()),
                )
                snapshot = native_facts["observed_outcome_snapshot"]
                opponent_out = bool(
                    snapshot["status"] == observed_outcome_resolver.STATUS_FLOOR_CONTACT
                    and snapshot["observed_net_clear"] is True
                )
            else:
                strike_sample = False
                eligibility = self._wait_eligibility()
                opponent_out = False
            output = n1_scalar_reward.evaluate_n1_c_lite_scalar_reward(
                n1_scalar_reward.N1ScalarRewardInput(
                    motion_reward=motion,
                    balance_reward=balance,
                    miss_sample_eligible=(transition_valid[index] and strike_sample),
                    selected_rubber_center_w_m=physical["selected_rubber_center_w_m"],
                    ball_center_w_m=physical["ball_center_w_m"],
                    event_eligibility=eligibility,
                    observed_opponent_side_out=opponent_out,
                    latch=self._reward_latches[index],
                )
            )
            outputs.append(output)
            next_latches[index] = output.next_latch
            reward_rows.append(
                {
                    "task_valid": transition_valid[index],
                    "nominal_strike_tick": self._nominal_strike_ticks[index],
                    "nominal_strike_sampled_now": strike_sample,
                    "sample_policy_tick_1based": policy_tick,
                    "sample_time_s": sample_time_s,
                    "nominal_clock_error_s": nominal_clock_error_s,
                    "exact_strike_timing_tick_count": int(strike_sample),
                    "strike_opportunity_count": int(strike_sample),
                    "motion_reward": output.motion_reward,
                    "balance_reward": output.balance_reward,
                    "miss_proximity_reward": output.miss_proximity_reward,
                    "observed_legal_landing_reward": (
                        output.observed_legal_landing_reward
                    ),
                    "observed_opponent_side_out_reward": (
                        output.observed_opponent_side_out_reward
                    ),
                    "total_reward": output.total_reward,
                    "observed_outcome_paid_now": output.observed_outcome_paid_now,
                    "contact_target_denominator": (
                        eligibility.contact_target_denominator
                    ),
                    "closed_swing_denominator": eligibility.closed_swing_denominator,
                    "actual_contact_numerator": eligibility.actual_contact_numerator,
                    "predicted_outcome_denominator": (
                        eligibility.predicted_outcome_denominator
                    ),
                    "observed_outcome_denominator": (
                        eligibility.observed_outcome_denominator
                    ),
                    "observed_legal_landing_numerator": (
                        eligibility.observed_legal_landing_numerator
                    ),
                    **kernels,
                }
            )

        reset_ids = set(batch.reset_env_ids)
        for index, state in enumerate(self._states):
            if index in reset_ids:
                state.reset()
                next_latches[index] = n1_scalar_reward.N1ScalarRewardLatch()
                continue
            if reward_rows[index]["nominal_strike_sampled_now"]:
                state.strike_sampled = True
            if state.wait_remaining == 1:
                try:
                    self._validate_reveal_ball_state(
                        index,
                        batch.terminal_observations[index].detach().cpu().numpy(),
                    )
                except Exception:
                    self._invalidate_after_failure()
                    raise
            revealed = state.advance()
            if revealed:
                # The hidden-fact validation above ran before reward/state mutation.
                pass
        self._reward_latches = next_latches
        next_valid = self.task_valid
        observations = _masked_task_observations(batch.observations, next_valid)
        terminal = _masked_task_observations(
            batch.terminal_observations, transition_valid
        )
        self._observations = observations.clone()
        rewards = torch.as_tensor(
            [output.total_reward for output in outputs],
            dtype=torch.float32,
            device="cpu",
        )
        extras = {
            "observations": {"critic": observations.clone()},
            "time_outs": batch.time_outs.clone(),
            "terminal_observations": terminal,
            "terminal_observation_mask": batch.terminal_observation_mask.clone(),
            "episode_done_reasons": list(batch.episode_done_reasons),
            "reward_terms": reward_rows,
            "task_valid_transition": list(transition_valid),
            "task_valid_next": list(next_valid),
            "reward_contract": copy.deepcopy(self._reward_receipt),
            "formal_blockers": list(FORMAL_BLOCKERS),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        return observations, rewards, batch.episode_dones.clone(), extras


__all__ = [
    "FORMAL_BLOCKERS",
    "ContinuousWaitPreparation",
    "FixedCenterDiagnosticVecEnv",
    "FixedCenterRecipeError",
    "FixedCenterRecipeSpec",
    "Frame0JointTeacher",
    "RECIPE_KIND",
    "TASK_SLICE",
    "prepare_continuous_wait_base",
]
