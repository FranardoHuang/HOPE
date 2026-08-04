"""Fixed-centre N1 diagnostic recipe layered over the native C-lite VecEnv.

This module intentionally does not claim the A211/C211 observation ABI or the
formal whole-body measured-motion reward.  It adds the narrow pieces that can
be derived from already sealed native authorities:

* an episode-local RESET_WAIT/TASK_ACTIVE state with atomic actor task masking;
* a frame-0 *joint-space* teacher reward loaded from the fixed tape lineage;
* a bounded pelvis upright/height balance reward around physical-ready; and
* the existing selected-rubber strike-distance and observed-outcome task terms.

The fixed question is installed once at reset and is never inverse-solved.
During RESET_WAIT the ball is parked at a sealed contact-free state because no
task exists yet.  WAIT->ACTIVE atomically installs the sealed native launch
state while revealing the task observation and public clock.  A hidden
contact/outcome before reveal remains a hard error because the cumulative
native event ABI cannot relabel it as active.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
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
    "parked_wait_atomic_native_ball_launch_is_not_cross_engine_parity",
    "fixed_question_freezes_curriculum_and_has_no_banded_question_bank",
    "formal_phase_recovery_export_and_mid_episode_resume_not_closed",
    "cpu_sequential_vecenv_has_no_4096_matched_workload_receipt",
)

TASK_WAIT_SOURCE = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/action_ball_task_wait.py"
)


def _load_task_wait_module() -> Any:
    """Load the backend-neutral counter sampler without importing Isaac."""

    name = "_action_ball_mujoco_native_task_wait"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module_spec = importlib.util.spec_from_file_location(name, TASK_WAIT_SOURCE)
    if module_spec is None or module_spec.loader is None:
        raise FixedCenterRecipeError("ActionBall WAIT schedule source is unavailable")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


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

    # ``reset_wait_steps`` is retained only for the historical fixed-WAIT
    # diagnostic.  C211 must set it to None and supply the exact seeded range.
    reset_wait_steps: int | None = 1
    reset_wait_min_steps: int | None = None
    reset_wait_max_steps: int | None = None
    reset_wait_seed: int = 20260804
    required_active_steps: int = 1
    joint_position_scale_rad: float = 0.5
    joint_velocity_scale_radps: float = 2.0
    motion_reward_weight: float = 0.25
    pelvis_height_scale_m: float = 0.1
    pelvis_up_scale: float = 0.1
    balance_reward_weight: float = 0.1

    def __post_init__(self) -> None:
        fixed = self.reset_wait_steps
        if fixed is not None:
            if type(fixed) is not int or fixed < 1:
                raise FixedCenterRecipeError(
                    "reset_wait_steps must be None or a positive plain integer"
                )
            if self.reset_wait_min_steps is not None or self.reset_wait_max_steps is not None:
                raise FixedCenterRecipeError(
                    "fixed RESET_WAIT cannot also declare a wait range"
                )
        else:
            if (
                type(self.reset_wait_min_steps) is not int
                or type(self.reset_wait_max_steps) is not int
                or self.reset_wait_min_steps < 1
                or self.reset_wait_max_steps < self.reset_wait_min_steps
            ):
                raise FixedCenterRecipeError(
                    "ranged RESET_WAIT requires positive inclusive min/max steps"
                )
        if (
            type(self.reset_wait_seed) is not int
            or not 0 <= self.reset_wait_seed <= (1 << 63) - 1
        ):
            raise FixedCenterRecipeError("reset_wait_seed must be a non-negative int64")
        if type(self.required_active_steps) is not int or self.required_active_steps < 1:
            raise FixedCenterRecipeError(
                "required_active_steps must be a positive plain integer"
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

    @property
    def min_wait_steps(self) -> int:
        return (
            int(self.reset_wait_steps)
            if self.reset_wait_steps is not None
            else int(self.reset_wait_min_steps)
        )

    @property
    def max_wait_steps(self) -> int:
        return (
            int(self.reset_wait_steps)
            if self.reset_wait_steps is not None
            else int(self.reset_wait_max_steps)
        )

    @property
    def is_seeded_range(self) -> bool:
        return self.reset_wait_steps is None


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
    """Content seal for hidden parking plus an atomic active-task launch.

    The historical class name is retained to avoid widening the checkpoint
    interface.  Its v2 content explicitly does *not* claim a continuous
    reverse-ballistic prefix.
    """

    spec_sha256: str
    wait_policy_min_steps: int
    wait_policy_max_steps: int
    control_decimation: int
    physics_step_dt_s: float
    per_env: tuple[Mapping[str, Any], ...]

    @property
    def content_sha256(self) -> str:
        return _sha256_json(
            {
                "schema_version": 2,
                "kind": "a3_mujoco_fixed_center_parked_wait_preparation_v2",
                "spec_sha256": self.spec_sha256,
                "wait_policy_min_steps": self.wait_policy_min_steps,
                "wait_policy_max_steps": self.wait_policy_max_steps,
                "control_decimation": self.control_decimation,
                "physics_step_dt_s": self.physics_step_dt_s,
                "per_env": [dict(row) for row in self.per_env],
                "wait_ball_state": "sealed_contact_free_parking",
                "reveal_mutation": "ball_only_atomic_sealed_native_launch",
            }
        )

    @property
    def wait_policy_steps(self) -> int | None:
        if self.wait_policy_min_steps == self.wait_policy_max_steps:
            return self.wait_policy_min_steps
        return None

    @property
    def wait_physics_substeps(self) -> int | None:
        if self.wait_policy_steps is None:
            return None
        return self.wait_policy_steps * self.control_decimation


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
    """Rebuild a C-lite base with hidden parking and a sealed task launch.

    RESET_WAIT has no ball task, so every variant starts the ball ten metres
    above its authoritative launch point with zero velocity.  The maximum
    frozen wait is only 0.5 s, leaving a large contact-free margin under pure
    gravity.  WAIT->ACTIVE writes only the ball free-joint to the original
    question birth state; robot state, time and controller history continue.
    """

    if not base_env.c_lite_reward_enabled:
        raise FixedCenterRecipeError("continuous WAIT requires a C-lite base")
    if len(base_env.cores) != base_env.num_envs or len(base_env.questions) != (
        base_env.num_envs
    ):
        raise FixedCenterRecipeError("base core/question cardinality differs")
    control_decimation = int(base_env.control_decimation)
    physics_dt = float(base_env.step_dt) / control_decimation
    initial_questions = []
    question_variants = []
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
        env_rows = []
        env_questions = []
        for wait_steps in range(spec.min_wait_steps, spec.max_wait_steps + 1):
            wait_substeps = wait_steps * control_decimation
            wait_s = wait_substeps * physics_dt
            reset_position = reveal_position + np.asarray(
                [0.0, 0.0, 10.0], dtype=np.float64
            )
            reset_velocity = np.zeros(3, dtype=np.float64)
            conservative_park_z_after_wait = float(
                reset_position[2]
                + gravity[2]
                * (physics_dt**2)
                * (wait_substeps * (wait_substeps + 1) / 2.0)
            )
            if conservative_park_z_after_wait <= reveal_position[2] + 5.0:
                raise FixedCenterRecipeError(
                    f"core {index} WAIT parking clearance is insufficient"
                )
            row = {
                "env_index": index,
                "wait_policy_steps": wait_steps,
                "wait_physics_substeps": wait_substeps,
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
                "reveal_ball_spin_w_radps": np.asarray(
                    question.birth_spin_w_radps, dtype=np.float64
                ).tolist(),
                "wait_ball_state": "sealed_contact_free_parking",
                "conservative_park_z_after_wait_m": (
                    conservative_park_z_after_wait
                ),
                "atomic_ball_launch_on_reveal": True,
                "parent_nominal_time_to_contact_s": (
                    question.nominal_time_to_contact_s
                ),
                "derived_nominal_time_to_contact_s": (
                    question.nominal_time_to_contact_s + wait_s
                ),
            }
            row["launch_content_sha256"] = _sha256_json(
                {
                    "schema_version": 1,
                    "kind": "a3_mujoco_parked_wait_question_v2",
                    "spec_sha256": spec.content_sha256,
                    **row,
                }
            )
            reset_position.setflags(write=False)
            reset_velocity.setflags(write=False)
            spin = np.asarray(question.birth_spin_w_radps, dtype=np.float64).copy()
            aim = np.asarray(question.landing_aim_xy_w_m, dtype=np.float64).copy()
            spin.setflags(write=False)
            aim.setflags(write=False)
            env_questions.append(
                n1_ball_core.N1Question(
                    source_path=question.source_path,
                    source_sha256=question.source_sha256,
                    question_id=question.question_id,
                    scene_binding_sha256=question.scene_binding_sha256,
                    birth_position_w_m=reset_position,
                    birth_linear_velocity_w_mps=reset_velocity,
                    birth_spin_w_radps=spin,
                    landing_aim_xy_w_m=aim,
                    nominal_time_to_contact_s=(
                        question.nominal_time_to_contact_s + wait_s
                    ),
                    spin_valid=question.spin_valid,
                    authority=copy.deepcopy(question.authority),
                    selected_rubber_action_lineage=copy.deepcopy(
                        question.selected_rubber_action_lineage
                    ),
                )
            )
            env_rows.append(row)
        rows.append(
            {
                "env_index": index,
                "parent_question_source_sha256": question.source_sha256,
                "variants": env_rows,
            }
        )
        question_variants.append(tuple(env_questions))
        initial_questions.append(env_questions[0])
    preparation = ContinuousWaitPreparation(
        spec_sha256=spec.content_sha256,
        wait_policy_min_steps=spec.min_wait_steps,
        wait_policy_max_steps=spec.max_wait_steps,
        control_decimation=control_decimation,
        physics_step_dt_s=physics_dt,
        per_env=tuple(rows),
    )
    rebuilt = vec_env.MujocoN1DiagnosticVecEnv(
        cores=base_env.cores,
        robot_tape=base_env.robot_tape,
        questions=tuple(initial_questions),
        device="cpu",
        enable_c_lite_reward=True,
        diagnostic_episode_length=base_env.max_episode_length,
    )
    rebuilt._fixed_center_continuous_wait_preparation = preparation
    rebuilt._fixed_center_wait_question_variants = tuple(question_variants)
    rebuilt._fixed_center_parent_identity = rebuilt.diagnostic_training_identity()
    rebuilt._fixed_center_parent_readiness = rebuilt.diagnostic_training_receipt()
    rebuilt._fixed_center_requires_outer_wrapper = True
    return rebuilt


@dataclass(frozen=True)
class _StagedQuestionReset:
    env_ids: tuple[int, ...]
    assignments: tuple[Any, ...]
    questions: tuple[n1_ball_core.N1Question, ...]


class _WaitQuestionResetProvider:
    """Two-phase bridge from counter assignments to sealed launch variants."""

    def __init__(
        self,
        *,
        schedule: Any,
        variants: Sequence[Sequence[n1_ball_core.N1Question]],
    ) -> None:
        wait = _load_task_wait_module()
        if not isinstance(schedule, wait.ActionBallTaskWaitSchedule):
            raise FixedCenterRecipeError("WAIT schedule type differs")
        self.schedule = schedule
        self.variants = tuple(tuple(row) for row in variants)
        expected_width = schedule.max_wait_ticks - schedule.min_wait_ticks + 1
        if not self.variants or any(len(row) != expected_width for row in self.variants):
            raise FixedCenterRecipeError("WAIT question variant cardinality differs")
        self._highwater = wait.ActionBallTaskWaitHighwater(schedule)
        self._current_assignments: list[Any | None] = [None] * len(self.variants)
        self._committed: dict[int, Any] = {}
        self._active: _StagedQuestionReset | None = None

    def stage(self, env_ids: Sequence[int]) -> _StagedQuestionReset:
        if self._active is not None:
            raise FixedCenterRecipeError("WAIT question reset transaction is nested")
        ids = tuple(int(value) for value in env_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(value < 0 or value >= len(self.variants) for value in ids)
        ):
            raise FixedCenterRecipeError("WAIT reset env ids are invalid")
        highwater_rows = dict(self._highwater._highwater_by_env)
        assignments = tuple(
            self.schedule.assignment(
                env_id=index,
                reset_generation=highwater_rows.get(index, 0) + 1,
            )
            for index in ids
        )
        questions = tuple(
            self.variants[index][
                assignment.wait_ticks - self.schedule.min_wait_ticks
            ]
            for index, assignment in zip(ids, assignments)
        )
        token = _StagedQuestionReset(ids, assignments, questions)
        self._active = token
        return token

    def commit(self, token: _StagedQuestionReset) -> None:
        if token is not self._active:
            raise FixedCenterRecipeError("WAIT reset commit token differs")
        recorded = []
        try:
            for assignment in token.assignments:
                value = self._highwater.record(
                    env_id=assignment.env_id,
                    reset_generation=assignment.reset_generation,
                )
                if value != assignment:
                    raise FixedCenterRecipeError("WAIT reset assignment drifted")
                recorded.append(value)
        except Exception:
            # ``record`` is deterministic and all generations were prechecked,
            # so this branch can only indicate internal corruption.  Do not
            # pretend a partially advanced provider remains usable.
            self._active = None
            raise
        for index, assignment in zip(token.env_ids, recorded):
            self._current_assignments[index] = assignment
            self._committed[index] = assignment
        self._active = None

    def abort(self, token: _StagedQuestionReset) -> None:
        if token is not self._active:
            raise FixedCenterRecipeError("WAIT reset abort token differs")
        self._active = None

    def consume_committed(self, env_ids: Sequence[int]) -> tuple[Any, ...]:
        ids = tuple(int(value) for value in env_ids)
        try:
            values = tuple(self._committed.pop(index) for index in ids)
        except KeyError as exc:
            raise FixedCenterRecipeError(
                "WAIT reset provider has no committed assignment"
            ) from exc
        return values

    @property
    def current_assignments(self) -> tuple[Any, ...]:
        if any(value is None for value in self._current_assignments):
            raise FixedCenterRecipeError("WAIT assignments are not initialized")
        return tuple(self._current_assignments)

    def state_dict(self) -> dict[str, Any]:
        if self._active is not None or self._committed:
            raise FixedCenterRecipeError(
                "WAIT provider state requested with an unconsumed reset transaction"
            )
        assignments = self.current_assignments
        payload = {
            "schema_version": 1,
            "kind": "a3_mujoco_c211_wait_question_provider_state_v1",
            "schedule": self.schedule.to_dict(),
            "highwater": self._highwater.state_dict(),
            "current_assignments": [value.to_dict() for value in assignments],
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def validate_state_dict(self, state: Any) -> tuple[Any, tuple[Any, ...]]:
        wait = _load_task_wait_module()
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "kind",
            "schedule",
            "highwater",
            "current_assignments",
            "content_sha256",
        }:
            raise FixedCenterRecipeError("WAIT provider checkpoint schema differs")
        payload = dict(state)
        declared = payload.pop("content_sha256")
        if (
            state["schema_version"] != 1
            or state["kind"] != "a3_mujoco_c211_wait_question_provider_state_v1"
            or declared != _sha256_json(payload)
        ):
            raise FixedCenterRecipeError("WAIT provider checkpoint seal differs")
        restored_schedule = wait.ActionBallTaskWaitSchedule.from_dict(state["schedule"])
        if restored_schedule != self.schedule:
            raise FixedCenterRecipeError("WAIT provider checkpoint schedule differs")
        restored_highwater = wait.ActionBallTaskWaitHighwater.from_state_dict(
            self.schedule, state["highwater"]
        )
        rows = state["current_assignments"]
        if not isinstance(rows, list) or len(rows) != len(self.variants):
            raise FixedCenterRecipeError("WAIT checkpoint assignment rows differ")
        assignments = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise FixedCenterRecipeError("WAIT checkpoint assignment is malformed")
            expected = self.schedule.assignment(
                env_id=index,
                reset_generation=row.get("reset_generation"),
            )
            if dict(row) != expected.to_dict():
                raise FixedCenterRecipeError("WAIT checkpoint assignment is noncanonical")
            restored_highwater.assert_recorded(expected)
            assignments.append(expected)
        return restored_highwater, tuple(assignments)

    def load_state_dict(self, state: Any) -> None:
        if self._active is not None or self._committed:
            raise FixedCenterRecipeError("WAIT provider is not at a load boundary")
        highwater, assignments = self.validate_state_dict(state)
        self._highwater = highwater
        self._current_assignments = list(assignments)

    def questions_for_assignments(
        self, assignments: Sequence[Any]
    ) -> tuple[n1_ball_core.N1Question, ...]:
        rows = tuple(assignments)
        if len(rows) != len(self.variants):
            raise FixedCenterRecipeError("WAIT assignment/question cardinality differs")
        return tuple(
            self.variants[index][
                assignment.wait_ticks - self.schedule.min_wait_ticks
            ]
            for index, assignment in enumerate(rows)
        )


@dataclass
class _EpisodeState:
    assignment: Any | None = None
    wait_remaining: int = 0
    strike_sampled: bool = False

    def reset(self, assignment: Any) -> None:
        self.assignment = assignment
        self.wait_remaining = int(assignment.wait_ticks)
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
            or preparation.wait_policy_min_steps != spec.min_wait_steps
            or preparation.wait_policy_max_steps != spec.max_wait_steps
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
        if spec.max_wait_steps + spec.required_active_steps > base_env.max_episode_length:
            raise FixedCenterRecipeError(
                "RESET_WAIT plus required active ticks exceeds episode horizon"
            )
        wait = _load_task_wait_module()
        self.wait_schedule = wait.ActionBallTaskWaitSchedule(
            seed=spec.reset_wait_seed,
            min_wait_ticks=spec.min_wait_steps,
            max_wait_ticks=spec.max_wait_steps,
            episode_horizon_ticks=int(base_env.max_episode_length),
            required_active_ticks=spec.required_active_steps,
        )
        variants = getattr(base_env, "_fixed_center_wait_question_variants", None)
        if not isinstance(variants, tuple) or len(variants) != self.num_envs:
            raise FixedCenterRecipeError("continuous WAIT question variants are absent")
        self._question_provider = _WaitQuestionResetProvider(
            schedule=self.wait_schedule, variants=variants
        )
        base_env.install_question_reset_provider(self._question_provider)
        self._states = [_EpisodeState() for _ in range(self.num_envs)]
        nominal_ticks_by_wait = []
        for index, env_variants in enumerate(variants):
            ticks = []
            for question in env_variants:
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
                ticks.append(tick)
            nominal_ticks_by_wait.append(tuple(ticks))
        self._nominal_strike_ticks_by_wait = tuple(nominal_ticks_by_wait)
        self._reward_latches = [
            n1_scalar_reward.N1ScalarRewardLatch() for _ in range(self.num_envs)
        ]
        self._observations = None
        self._last_transition_task_valid = tuple(False for _ in self._states)
        self.cfg = {
            **copy.deepcopy(base_env.cfg),
            "kind": "a3_mujoco_n1_fixed_center_diagnostic_vecenv_v1",
            "reward_scope": "fixed_center_joint_teacher_learnability_diagnostic",
            "reset_wait_schedule": self.wait_schedule.to_dict(),
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
        assignments = self._question_provider.current_assignments
        for index, (question, row, assignment) in enumerate(
            zip(self.base_env.questions, preparation.per_env, assignments)
        ):
            variant = row["variants"][
                assignment.wait_ticks - self.wait_schedule.min_wait_ticks
            ]
            if (
                question.source_sha256 != row["parent_question_source_sha256"]
                or not np.array_equal(
                    np.asarray(question.birth_position_w_m, dtype=np.float64),
                    np.asarray(variant["reset_ball_position_w_m"], dtype=np.float64),
                )
                or not np.array_equal(
                    np.asarray(question.birth_linear_velocity_w_mps, dtype=np.float64),
                    np.asarray(
                        variant["reset_ball_linear_velocity_w_mps"], dtype=np.float64
                    ),
                )
                or not math.isclose(
                    float(question.nominal_time_to_contact_s),
                    float(variant["derived_nominal_time_to_contact_s"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise FixedCenterRecipeError(
                    f"continuous WAIT question {index} mutated after preparation"
                )

    @property
    def current_wait_assignments(self) -> tuple[Any, ...]:
        return tuple(state.assignment for state in self._states)

    @property
    def current_wait_steps(self) -> tuple[int, ...]:
        return tuple(int(state.assignment.wait_ticks) for state in self._states)

    def _nominal_tick(self, index: int, assignment: Any) -> int:
        return self._nominal_strike_ticks_by_wait[index][
            assignment.wait_ticks - self.wait_schedule.min_wait_ticks
        ]

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
                "reveal_mutation": "ball_only_atomic_sealed_native_launch",
                "robot_state_teleport_on_reveal": False,
                "ball_free_joint_write_on_reveal": True,
                "physical_ball_parked_during_wait": True,
                "physical_ball_trajectory_includes_wait_from_reset": False,
                "counter_schedule": self.wait_schedule.to_dict(),
                "counter_schedule_source_sha256": hashlib.sha256(
                    TASK_WAIT_SOURCE.read_bytes()
                ).hexdigest(),
                "preparation_sha256": (self.continuous_wait_preparation.content_sha256),
                "preparation": {
                    "wait_ball_state": "sealed_contact_free_parking",
                    "reveal_ball_state": "sealed_native_question_launch",
                    "wait_policy_min_steps": self.spec.min_wait_steps,
                    "wait_policy_max_steps": self.spec.max_wait_steps,
                    "control_decimation": (
                        self.continuous_wait_preparation.control_decimation
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
                "nominal_policy_tick_by_env_and_wait": [
                    list(row) for row in self._nominal_strike_ticks_by_wait
                ],
                "nominal_tick_reachable": all(
                    tick <= self.base_env.max_episode_length
                    for row in self._nominal_strike_ticks_by_wait
                    for tick in row
                ),
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
        observations, extras = self.base_env.reset(seed=seed)
        assignments = self._question_provider.consume_committed(range(self.num_envs))
        for state, assignment in zip(self._states, assignments):
            state.reset(assignment)
        self._validate_preparation_integrity()
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
        if self.base_env.is_reset_boundary() is not True:
            return False
        return all(
            state.assignment is not None
            and state.wait_remaining == state.assignment.wait_ticks
            and not state.strike_sampled
            for state in self._states
        )

    def checkpoint_state(self) -> dict[str, Any]:
        if not self.is_reset_boundary():
            raise FixedCenterRecipeError(
                "WAIT checkpoint state requires an exact reset boundary"
            )
        payload = {
            "schema_version": 1,
            "kind": "a3_mujoco_fixed_center_wait_boundary_state_v1",
            "spec_sha256": self.spec.content_sha256,
            "preparation_sha256": self._continuous_wait_preparation_sha256,
            "provider": self._question_provider.state_dict(),
            "wait_remaining": [state.wait_remaining for state in self._states],
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload

    def load_checkpoint_state(self, state: Any) -> None:
        if not self.is_reset_boundary():
            raise FixedCenterRecipeError(
                "WAIT checkpoint load requires an exact reset boundary"
            )
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "kind",
            "spec_sha256",
            "preparation_sha256",
            "provider",
            "wait_remaining",
            "content_sha256",
        }:
            raise FixedCenterRecipeError("WAIT boundary checkpoint schema differs")
        payload = dict(state)
        declared = payload.pop("content_sha256")
        if (
            state["schema_version"] != 1
            or state["kind"] != "a3_mujoco_fixed_center_wait_boundary_state_v1"
            or state["spec_sha256"] != self.spec.content_sha256
            or state["preparation_sha256"]
            != self._continuous_wait_preparation_sha256
            or declared != _sha256_json(payload)
        ):
            raise FixedCenterRecipeError("WAIT boundary checkpoint seal differs")
        highwater, assignments = self._question_provider.validate_state_dict(
            state["provider"]
        )
        remaining = state["wait_remaining"]
        if (
            not isinstance(remaining, list)
            or remaining != [value.wait_ticks for value in assignments]
        ):
            raise FixedCenterRecipeError("WAIT boundary remaining state differs")
        questions = self._question_provider.questions_for_assignments(assignments)
        # All authority validation above is non-mutating.  The remaining
        # installation is deterministic and reconstructs a reset boundary.
        self._question_provider._highwater = highwater
        self._question_provider._current_assignments = list(assignments)
        observations = self.base_env.restore_reset_boundary_questions(questions)
        for episode, assignment in zip(self._states, assignments):
            episode.reset(assignment)
        self._reward_latches = [
            n1_scalar_reward.N1ScalarRewardLatch() for _ in range(self.num_envs)
        ]
        self._last_transition_task_valid = tuple(False for _ in self._states)
        self._observations = _masked_task_observations(observations, self.task_valid)
        self._validate_preparation_integrity()

    def diagnostic_training_identity(self) -> dict[str, str]:
        return dict(self._identity)

    def diagnostic_training_receipt(self) -> dict[str, Any]:
        base = getattr(self.base_env, "_fixed_center_parent_readiness", None)
        if base is None:
            base = self.base_env.diagnostic_training_receipt()
        if base.get("ppo_ready") is not True:
            raise FixedCenterRecipeError("base C-lite readiness is not available")
        terminal_contract = trainer.terminal_row_telemetry_contract()
        return {
            "schema_version": 1,
            "kind": READINESS_KIND,
            "ppo_ready": True,
            "reward_available": True,
            "normal_step_available": True,
            "reset_boundary_checkpoint_available": True,
            "terminal_row_telemetry_available": True,
            "terminal_row_telemetry_contract": copy.deepcopy(
                terminal_contract
            ),
            **self.diagnostic_training_identity(),
            "reward_scope": "fixed_center_joint_teacher_learnability_diagnostic",
            "reset_wait_schedule": self.wait_schedule.to_dict(),
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

    def _install_reveal_ball_state(
        self, index: int, assignment: Any, observation_row: Any
    ) -> Any | None:
        expected = self.continuous_wait_preparation.per_env[index]["variants"][
            assignment.wait_ticks - self.wait_schedule.min_wait_ticks
        ]
        expected_position = np.asarray(
            expected["reveal_ball_position_w_m"], dtype=np.float64
        )
        expected_velocity = np.asarray(
            expected["reveal_ball_linear_velocity_w_mps"], dtype=np.float64
        )
        expected_spin = np.asarray(
            expected["reveal_ball_spin_w_radps"], dtype=np.float64
        )
        refreshed = None
        if hasattr(self.base_env, "cores"):
            core = self.base_env.cores[index]
            if (
                core._events
                or core._active_contact_labels
                or core._racket_contact_edges != 0
                or core._outgoing_state is not None
                or core._contact_labels()
            ):
                raise FixedCenterRecipeError(
                    f"env {index} hidden WAIT is not clean at atomic launch"
                )
            qpos = core.scene.ball_qpos_adr
            dof = core.scene.ball_dof_adr
            core.data.qpos[qpos : qpos + 3] = expected_position
            core.data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
            core.data.qvel[dof : dof + 3] = expected_velocity
            core.data.qvel[dof + 3 : dof + 6] = expected_spin
            core.data.qacc_warmstart[dof : dof + 6] = 0.0
            core.mujoco.mj_forward(core.model, core.data)
            if core._contact_labels():
                raise FixedCenterRecipeError(
                    f"env {index} sealed reveal launch starts in contact"
                )
            actual_position = np.asarray(
                core.data.qpos[qpos : qpos + 3], dtype=np.float64
            )
            actual_velocity = np.asarray(
                core.data.qvel[dof : dof + 3], dtype=np.float64
            )
            actual_spin = np.asarray(
                core.data.qvel[dof + 3 : dof + 6], dtype=np.float64
            )
            tolerance = 1.0e-9
            refreshed = self.base_env._tensor_observations(
                [core.observation_groups()]
            )[0]
            self.base_env._observations[index].copy_(refreshed)
        else:
            row = np.asarray(observation_row, dtype=np.float64)
            if row.shape != (vec_env.OBSERVATION_WIDTH,):
                raise FixedCenterRecipeError(
                    f"env {index} reveal observation shape differs"
                )
            actual_position = row[62:65]
            actual_velocity = row[65:68]
            actual_spin = row[68:71]
            tolerance = 1.0e-6
        if not np.allclose(
            actual_position, expected_position, rtol=0.0, atol=tolerance
        ) or not np.allclose(
            actual_velocity, expected_velocity, rtol=0.0, atol=tolerance
        ) or not np.allclose(
            actual_spin, expected_spin, rtol=0.0, atol=tolerance
        ):
            raise FixedCenterRecipeError(
                f"env {index} atomic reveal differs from sealed ball launch"
            )
        return refreshed

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
        transition_assignments = self.current_wait_assignments
        transition_nominal_ticks = tuple(
            self._nominal_tick(index, assignment)
            for index, assignment in enumerate(transition_assignments)
        )
        transition_nominal_times = tuple(
            float(question.nominal_time_to_contact_s)
            for question in self.base_env.questions
        )
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
                transition_nominal_times[index]
            )
            if transition_valid[index]:
                strike_sample = policy_tick == transition_nominal_ticks[index]
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
                    "wait_assignment": transition_assignments[index].to_dict(),
                    "nominal_strike_tick": transition_nominal_ticks[index],
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
        reset_assignments = dict(
            zip(
                batch.reset_env_ids,
                self._question_provider.consume_committed(batch.reset_env_ids),
            )
        )
        for index, state in enumerate(self._states):
            if index in reset_ids:
                state.reset(reset_assignments[index])
                next_latches[index] = n1_scalar_reward.N1ScalarRewardLatch()
                continue
            if reward_rows[index]["nominal_strike_sampled_now"]:
                state.strike_sampled = True
            if state.wait_remaining == 1:
                try:
                    refreshed = self._install_reveal_ball_state(
                        index,
                        transition_assignments[index],
                        batch.terminal_observations[index].detach().cpu().numpy(),
                    )
                    if refreshed is not None:
                        batch.observations[index].copy_(refreshed)
                except Exception:
                    self._invalidate_after_failure()
                    raise
            revealed = state.advance()
            if revealed:
                # The hidden-fact validation above ran before reward/state mutation.
                pass
        self._reward_latches = next_latches
        self._validate_preparation_integrity()
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
            # Preserve the exact post-transition facts before the underlying
            # compact reset mutates each native core.  Outer diagnostic-only
            # ABI adapters may consume these facts for a stricter reward;
            # they are never inserted into the actor observation.
            "diagnostic_native_physical_event_facts": tuple(
                copy.deepcopy(value)
                for value in batch.per_env_native_physical_event_facts
            ),
            "diagnostic_c_lite_physical_samples": tuple(
                copy.deepcopy(value)
                for value in batch.per_env_c_lite_physical_samples
            ),
            "diagnostic_qdes_projection_masks": tuple(
                tuple(bool(flag) for flag in row)
                for row in batch.per_env_qdes_projection_masks
            ),
            "diagnostic_event_ledgers": tuple(
                copy.deepcopy(value) for value in batch.per_env_ledgers
            ),
            "diagnostic_exact_hard_terminations": (
                batch.exact_hard_terminations.clone()
            ),
            "diagnostic_exact_hard_termination_reasons": list(
                batch.exact_hard_termination_reasons
            ),
            "task_valid_transition": list(transition_valid),
            "task_valid_next": list(next_valid),
            "wait_assignment_transition": [
                value.to_dict() for value in transition_assignments
            ],
            "wait_assignment_next": [
                value.to_dict() for value in self.current_wait_assignments
            ],
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
