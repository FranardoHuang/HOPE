"""Real-MjData A211 adapter built beside, not on top of, the C211 task ABI.

The native C211 lane already owns the difficult engine-facing authorities:
live MuJoCo FK, the measured whole-body/paddle teacher, selected-rubber event
facts, achieved outgoing-flight reconstruction, and reset-boundary replay.
This module reuses those engine helpers while installing A211's different task
contract:

* ``current_lm`` supplies a complete desired contact position, face-centre
  velocity, and signed hitting-face normal;
* the actor and critic use the A211 ordered 211/319 layouts;
* the target reward is paid over the Isaac-synonymous tight position and wide
  velocity/face windows, using the fixed first-learnability widths; and
* landing is graded only from an actual selected-rubber contact followed by a
  source-bound achieved outgoing flight.

RESET_WAIT keeps the balance/body/paddle prior active and masks every task
term.  This remains a CPU, diagnostic-only partial-parity lane: the native
adapter still lacks the Isaac foot/contact, applied-torque, and complete safety
reward terms, and no 4096-environment or cross-engine parity claim follows.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from . import action_ball_211_abi as abi
from . import action_ball_c211_env as shared
from . import trainer


A211_ENV_KIND = "action_ball_a211_mujoco_native_vecenv_v1"
A211_TASK_PROVIDER_KIND = abi.TASK_QUESTION_AUTHORITY_KIND
A211_TARGET_RECIPE = "current_lm"
A211_REWARD_SCOPE = "action_ball_a211_partial_isaac_synonymous_reward_v1"
A211_REWARD_CONTRACT_IDENTITY = (
    "action_ball_a211_native_partial_isaac_synonymous_reward_v1"
)
A211_TASK_REWARD_CONTRACT_IDENTITY = (
    "action_ball_a211_desired_contact_and_achieved_landing_reward_v1"
)
A211_POLICY_DT_S = 0.02
A211_POSITION_HALF_WINDOW_S = 0.02
A211_WIDE_HALF_WINDOW_S = 0.10
A211_CAPTURE_POST_DT_WEIGHT = 0.5
A211_PASS_NET_POST_DT_WEIGHT = 0.4
A211_LANDING_DENSE_POST_DT_WEIGHT = 0.4
A211_LEGAL_LANDING_POST_DT_WEIGHT = 14.0
A211_NET_MARGIN_M = 0.12
A211_NET_SIGMA_M = 0.25
A211_LANDING_SIGMA_M = 1.0
A211_LANDING_LEGAL_BASE_FRAC = 0.6
A211_BASE_POSITION_WEIGHT = 0.0
A211_BASE_POSITION_STD_M = 0.20
A211_RACKET_PROGRESS_WEIGHT = 10.0
A211_RACKET_PROGRESS_POTENTIAL_CAP_M = 4.65
COUNTER_RALLY_PY = (
    shared.C211_TRAINABILITY_PY.parent / "mdp/counter_rally.py"
)
COUNTER_RALLY_TORCH_PY = (
    shared.C211_TRAINABILITY_PY.parent / "mdp/counter_rally_torch.py"
)

# The A211 N1 leaf disables adaptive sigma.  These are therefore the exact
# resolved rollout-zero widths/weights, including A211's narrower coarse p/v
# overrides.  Reward-manager weights are multiplied by policy_dt below.
A211_TARGET_CHANNELS = {
    "position": {
        "half_window_s": A211_POSITION_HALF_WINDOW_S,
        "coarse_weight": 11.5,
        "coarse_std": 0.20,
        "fine_weight": 4.6,
        "fine_std": 0.50,
        "precision_weight": 0.575,
        "precision_std": 0.075,
    },
    "velocity": {
        "half_window_s": A211_WIDE_HALF_WINDOW_S,
        "coarse_weight": 11.5,
        "coarse_std": 1.50,
        "fine_weight": 0.575,
        "fine_std": 3.0,
        "precision_weight": 0.2875,
        "precision_std": 0.50,
    },
    "face": {
        "half_window_s": A211_WIDE_HALF_WINDOW_S,
        "coarse_weight": 5.75,
        "coarse_std": 1.0,
        "fine_weight": 0.575,
        "fine_std": 2.10,
        "precision_weight": 0.575,
        "precision_std": 0.262,
    },
}

A211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS: tuple[dict[str, str], ...] = tuple()

A211_FORMAL_BLOCKERS = (
    "a211_split_ready_physical_birth_cross_engine_parity_unmeasured",
    "a211_full_body_mimic_and_measured_paddle_priors_cross_engine_parity_unmeasured",
    "a211_isaac_reward_parity_incomplete_foot_contact_undesired_contact_applied_torque_and_safety_terms",
    "a211_seeded_wait_has_no_cross_engine_runtime_parity_receipt",
    "a211_mujoco_phase_recovery_export_and_mid_episode_resume_not_closed",
    "a211_mujoco_cpu_sequential_vecenv_has_no_4096_matched_workload_receipt",
)


class A211EnvError(shared.C211EnvError):
    """The real A211 authority, observation, or reward failed closed."""


def _plain_sha256(value: Any, name: str) -> str:
    try:
        return shared._plain_sha256(value, name)
    except shared.C211EnvError as exc:
        raise A211EnvError(str(exc)) from exc


def _finite_vector(value: Any, width: int, name: str) -> np.ndarray:
    try:
        return shared._finite_vector(value, width, name)
    except shared.C211EnvError as exc:
        raise A211EnvError(str(exc)) from exc


def _unit_vector(value: Any, name: str) -> np.ndarray:
    row = _finite_vector(value, 3, name)
    norm = float(np.linalg.norm(row))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise A211EnvError(f"{name} must be unit length")
    return row / norm


def _load_source_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise A211EnvError(f"A211 source module cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class A211TaskAuthority:
    """Complete desired-contact question plus teacher timing for one fixed N1."""

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
    desired_contact_position_w_m: tuple[float, float, float]
    desired_contact_velocity_w_mps: tuple[float, float, float]
    desired_contact_signed_face_w: tuple[float, float, float]
    counter_rally_task_canonical_sha256: str
    counter_rally_objective_profile_sha256: str
    counter_rally_return_direction_env_xy: tuple[float, float]
    counter_rally_target_baseline_speed_mps: float
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
        target_recipe: str = A211_TARGET_RECIPE,
    ) -> "A211TaskAuthority":
        source = Path(path).expanduser().resolve(strict=True)
        expected = _plain_sha256(expected_file_sha256, "immutable tape SHA")
        try:
            module = shared.n1_ball_core._load_fixed_question_tape_module()
            tape = module.load_immutable_n1_tape(
                source, expected_file_sha256=expected
            )
            question = tape.question_payload
            target = tape.targets[target_recipe]
            lineage = tape.target_lineage(target_recipe)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise A211EnvError(f"fixed A211 question tape is invalid: {exc}") from exc
        if target_recipe != A211_TARGET_RECIPE or tuple(target.validity_mask) != (
            True,
            True,
            True,
        ):
            raise A211EnvError("A211 requires current_lm with target validity 111")
        runtime = target.runtime_target
        timing_names = (
            "teacher_rate",
            "pre_swing_wait_s",
            "scaled_t_hit_s",
            "scaled_t_cycle_s",
            "reference_t_hit_s",
            "reference_t_cycle_s",
        )
        timing: dict[str, float] = {}
        for name in timing_names:
            value = runtime.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise A211EnvError(f"fixed A211 target omits numeric {name}")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise A211EnvError(f"fixed A211 target {name} is invalid")
            timing[name] = value
        ttc = float(question["time_to_contact_s"])
        if not math.isclose(
            timing["pre_swing_wait_s"] + timing["scaled_t_hit_s"],
            ttc,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            timing["teacher_rate"] * timing["scaled_t_hit_s"],
            timing["reference_t_hit_s"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise A211EnvError("fixed A211 teacher/contact clocks disagree")
        desired_position = _finite_vector(
            target.desired_racket_site_w_m,
            3,
            "A211 desired contact position",
        )
        desired_velocity = _finite_vector(
            target.desired_racket_face_center_velocity_w_mps,
            3,
            "A211 desired contact face-centre velocity",
        )
        desired_face = _unit_vector(
            target.desired_racket_face_normal_w,
            "A211 desired signed contact face",
        )
        counter_rally = question.get("counter_rally_task")
        expected_counter_keys = {
            "schema_version",
            "canonical_sha256",
            "objective_profile_sha256",
            "return_direction_env_xy",
            "target_baseline_speed_mps",
        }
        if not isinstance(counter_rally, Mapping) or set(counter_rally) != expected_counter_keys:
            raise A211EnvError(
                "current A211 N1 requires one canonical counter-rally task"
            )
        counter_payload = {
            key: copy.deepcopy(value)
            for key, value in counter_rally.items()
            if key != "canonical_sha256"
        }
        counter_sha = _plain_sha256(
            counter_rally["canonical_sha256"], "counter-rally task SHA"
        )
        if (
            counter_rally["schema_version"] != 1
            or shared._sha256_json(counter_payload) != counter_sha
        ):
            raise A211EnvError("counter-rally task seal differs")
        counter_direction = _finite_vector(
            counter_rally["return_direction_env_xy"],
            2,
            "counter-rally return direction",
        )
        direction_norm = float(np.linalg.norm(counter_direction))
        if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise A211EnvError("counter-rally return direction must be unit length")
        counter_speed = float(counter_rally["target_baseline_speed_mps"])
        if not math.isfinite(counter_speed) or counter_speed <= 0.0:
            raise A211EnvError("counter-rally target baseline speed is invalid")
        return cls(
            source_path=str(source),
            file_sha256=expected,
            canonical_sha256=_plain_sha256(
                tape.canonical_sha256, "tape canonical SHA"
            ),
            question_sha256=_plain_sha256(
                tape.question_sha256, "tape question SHA"
            ),
            target_recipe=target_recipe,
            target_producer_sha256=_plain_sha256(
                lineage["target_producer_sha256"], "target producer SHA"
            ),
            target_column_sha256=_plain_sha256(
                lineage["target_column_sha256"], "target column SHA"
            ),
            motion_sha256=_plain_sha256(question["motion_sha256"], "motion SHA"),
            physics_sha256=_plain_sha256(
                question["physics_sha256"], "physics SHA"
            ),
            profile_sha256=_plain_sha256(
                question["profile_sha256"], "profile SHA"
            ),
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
            desired_contact_position_w_m=tuple(desired_position),
            desired_contact_velocity_w_mps=tuple(desired_velocity),
            desired_contact_signed_face_w=tuple(desired_face),
            counter_rally_task_canonical_sha256=counter_sha,
            counter_rally_objective_profile_sha256=_plain_sha256(
                counter_rally["objective_profile_sha256"],
                "counter-rally objective profile SHA",
            ),
            counter_rally_return_direction_env_xy=tuple(counter_direction),
            counter_rally_target_baseline_speed_mps=counter_speed,
            time_to_contact_s=ttc,
            **timing,
        )

    @property
    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "kind": A211_TASK_PROVIDER_KIND,
            "profile": "A211",
            "source_path": self.source_path,
            "file_sha256": self.file_sha256,
            "canonical_sha256": self.canonical_sha256,
            "question_sha256": self.question_sha256,
            "target_recipe": self.target_recipe,
            "target_producer_sha256": self.target_producer_sha256,
            "target_column_sha256": self.target_column_sha256,
            "target_validity_mask": [True, True, True],
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
                "desired_contact_position_w_m": list(
                    self.desired_contact_position_w_m
                ),
                "desired_contact_velocity_w_mps": list(
                    self.desired_contact_velocity_w_mps
                ),
                "desired_contact_signed_face_w": list(
                    self.desired_contact_signed_face_w
                ),
                "counter_rally_task": {
                    "canonical_sha256": self.counter_rally_task_canonical_sha256,
                    "objective_profile_sha256": (
                        self.counter_rally_objective_profile_sha256
                    ),
                    "return_direction_env_xy": list(
                        self.counter_rally_return_direction_env_xy
                    ),
                    "target_baseline_speed_mps": (
                        self.counter_rally_target_baseline_speed_mps
                    ),
                },
                "time_to_contact_s": self.time_to_contact_s,
            },
            "teacher_timing": {
                name: getattr(self, name)
                for name in (
                    "teacher_rate",
                    "pre_swing_wait_s",
                    "scaled_t_hit_s",
                    "scaled_t_cycle_s",
                    "reference_t_hit_s",
                    "reference_t_cycle_s",
                )
            },
            "selection": "constant_row_zero_no_rng_or_cursor",
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = shared._sha256_json(payload)
        return payload

    @property
    def content_sha256(self) -> str:
        return self.receipt["content_sha256"]


# The measured authority has no task-specific observation semantics; it only
# requires the shared timing/motion fields that A211TaskAuthority also binds.
MeasuredA211MimicAuthority = shared.MeasuredC211MimicAuthority


class A211ObservationProducer(shared.C211ObservationProducer):
    """Construct exact A211 groups from the shared live-engine authorities."""

    def _validate_questions(self) -> None:
        try:
            super()._validate_questions()
        except shared.C211EnvError as exc:
            raise A211EnvError(str(exc).replace("C211", "A211")) from exc

    def _capture_reward_row(self, index: int, substep_index: int) -> None:
        super()._capture_reward_row(index, substep_index)
        row = self._reward_capture_rows[index]
        if row is None:
            raise A211EnvError("A211 final-substep reward row is absent")
        live = self._live(index)
        row["achieved_contact_sample"] = {
            "position_w_m": live["racket_pos"].tolist(),
            "velocity_w_mps": live["racket_velocity"].tolist(),
            # Isaac's desired_racket_face_normal_w is the raw site +Y/A
            # command frame.  The selected/signed B face remains correct for
            # measured-paddle mimic and physical-contact eligibility only.
            "signed_face_w": live["racket_raw_y_axis"].tolist(),
        }
        row["a211_footwork_sample"] = {
            "base_position_w_m": live["root_pos"].tolist(),
            # Isaac's racket_progress potential uses the bare contact point,
            # not the moving swing-through position used by the window kernel.
            "racket_position_w_m": live["racket_pos"].tolist(),
        }

    def groups(
        self, task_valid: Sequence[bool]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        valid = np.asarray(task_valid, dtype=np.bool_)
        if valid.shape != (len(self.cores),):
            raise A211EnvError("A211 task_valid batch shape differs")
        actor_rows: dict[str, list[np.ndarray]] = {
            name: [] for name, _width in abi.A211_PROFILE.actor.layout
        }
        critic_rows: dict[str, list[np.ndarray]] = {
            name: [] for name, _width in abi.A211_PROFILE.critic.layout
        }
        for index, is_valid in enumerate(valid.tolist()):
            live = self._live(index)
            teacher, teacher_hit = self._teacher(index, live, is_valid)
            heading = shared._yaw_rotation(live["root_rotation"])
            default_q = np.asarray(
                self.cores[index].binding.default_joint_pos, dtype=np.float64
            )
            task_position = heading.T @ (
                np.asarray(self.task.desired_contact_position_w_m)
                - live["root_pos"]
            )
            task_velocity = heading.T @ np.asarray(
                self.task.desired_contact_velocity_w_mps
            )
            task_face = heading.T @ np.asarray(
                self.task.desired_contact_signed_face_w
            )
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
                    shared._rotation_to_6d(
                        live["root_rotation"], "root rotation"
                    ),
                    live["root_lin_vel_w"],
                )
            )
            actor_values = {
                "actual_base_pose_lin_vel_world": root_pose,
                "base_ang_vel_body": (
                    live["root_rotation"].T @ live["root_ang_vel_w"]
                ),
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
                "task_desired_contact_position_heading": task_position,
                "task_desired_contact_velocity_heading": task_velocity,
                "task_desired_contact_face_heading": task_face,
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
                [
                    anchor_rotation.T @ rotation
                    for rotation in live["body_rotation"]
                ],
                axis=0,
            )
            critic_values = {
                "command": np.concatenate(
                    (teacher["joint_pos"], teacher["joint_vel"])
                ),
                "motion_anchor_pos_b": anchor_rotation.T
                @ (teacher["anchor_pos"] - live["anchor_pos"]),
                "motion_anchor_ori_b": shared._rotation_to_6d(
                    anchor_rotation.T @ teacher["anchor_rotation"],
                    "teacher anchor relative rotation",
                ),
                "body_pos": relative_body_pos.reshape(-1),
                "body_ori": np.concatenate(
                    [
                        shared._rotation_to_6d(
                            rotation, "body relative rotation"
                        )
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
                "task_desired_contact_position_heading": task_position,
                "task_desired_contact_velocity_heading": task_velocity,
                "task_desired_contact_face_heading": task_face,
                "desired_base_xy_world": actor_values["desired_base_xy_world"],
                "time_to_contact": actor_values["time_to_contact"],
                "time_to_teacher_start": actor_values["time_to_teacher_start"],
                "task_valid": actor_values["task_valid"],
            }
            for name in actor_rows:
                actor_rows[name].append(np.asarray(actor_values[name], dtype=np.float64))
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
            raise A211EnvError("torch is required for the A211 VecEnv") from exc
        actor_groups, critic_groups = self.groups(task_valid)
        actor, critic = abi.flatten_profile_groups(
            abi.A211_PROFILE,
            actor_groups=actor_groups,
            critic_groups=critic_groups,
            task_valid=np.asarray(task_valid, dtype=np.bool_),
            authorities=self.authorities,
        )
        return (
            torch.as_tensor(actor, dtype=torch.float32, device="cpu"),
            torch.as_tensor(critic, dtype=torch.float32, device="cpu"),
        )


def _cauchy(error: float, std: float) -> float:
    return 1.0 / (1.0 + (float(error) / float(std)) ** 2)


def _gaussian(error: float, std: float) -> float:
    return math.exp(-((float(error) / float(std)) ** 2))


def a211_desired_contact_reward_terms(
    *,
    task_valid: bool,
    time_to_contact_s: float,
    achieved_position_w_m: Any,
    achieved_velocity_w_mps: Any,
    achieved_signed_face_w: Any,
    desired_position_w_m: Any,
    desired_velocity_w_mps: Any,
    desired_signed_face_w: Any,
) -> dict[str, Any]:
    """Evaluate A211's fixed-width desired-contact reward for one policy row."""

    if type(task_valid) is not bool:
        raise A211EnvError("A211 target task_valid must be bool")
    time_error = float(time_to_contact_s)
    if not math.isfinite(time_error):
        raise A211EnvError("A211 target clock is non-finite")
    achieved_position = _finite_vector(
        achieved_position_w_m, 3, "A211 achieved contact position"
    )
    achieved_velocity = _finite_vector(
        achieved_velocity_w_mps, 3, "A211 achieved contact velocity"
    )
    achieved_face = _unit_vector(
        achieved_signed_face_w, "A211 achieved signed contact face"
    )
    desired_position = _finite_vector(
        desired_position_w_m, 3, "A211 desired contact position"
    )
    desired_velocity = _finite_vector(
        desired_velocity_w_mps, 3, "A211 desired contact velocity"
    )
    desired_face = _unit_vector(
        desired_signed_face_w, "A211 desired signed contact face"
    )
    # Isaac's target-position channel follows the desired swing-through line.
    desired_position_now = desired_position - desired_velocity * time_error
    errors = {
        "position": float(np.linalg.norm(achieved_position - desired_position_now)),
        "velocity": float(np.linalg.norm(achieved_velocity - desired_velocity)),
        "face": math.acos(
            float(np.clip(np.dot(achieved_face, desired_face), -1.0, 1.0))
        ),
    }
    channels: dict[str, Any] = {}
    total = 0.0
    tolerance = 1.0e-12
    for name, contract in A211_TARGET_CHANNELS.items():
        eligible = bool(
            task_valid
            and abs(time_error) <= float(contract["half_window_s"]) + tolerance
        )
        error = errors[name]
        coarse = _cauchy(error, contract["coarse_std"]) if eligible else 0.0
        fine = _gaussian(error, contract["fine_std"]) if eligible else 0.0
        precision = (
            _gaussian(error, contract["precision_std"]) if eligible else 0.0
        )
        reward = A211_POLICY_DT_S * (
            float(contract["coarse_weight"]) * coarse
            + float(contract["fine_weight"]) * fine
            + float(contract["precision_weight"]) * precision
        )
        channels[name] = {
            "eligible": eligible,
            "error": error,
            "error_unit": "rad" if name == "face" else ("m" if name == "position" else "mps"),
            "half_window_s": float(contract["half_window_s"]),
            "coarse_kernel": coarse,
            "fine_kernel": fine,
            "precision_kernel": precision,
            "post_policy_dt_reward": reward,
        }
        total += reward
    if not math.isfinite(total):
        raise A211EnvError("A211 target reward is non-finite")
    return {
        "task_valid": task_valid,
        "time_to_contact_s": time_error,
        "desired_position_now_w_m": desired_position_now.tolist(),
        "channels": channels,
        "any_channel_eligible": any(row["eligible"] for row in channels.values()),
        "post_policy_dt_reward": total,
    }


def a211_prestrike_footwork_reward_terms(
    *,
    task_valid: bool,
    time_to_contact_s: float,
    achieved_base_position_w_m: Any,
    desired_base_position_w_m: Any,
    achieved_racket_position_w_m: Any,
    desired_contact_position_w_m: Any,
    previous_racket_distance_m: float,
    reset_progress_baseline: bool = False,
) -> dict[str, Any]:
    """Mirror the current Isaac A211 racket-progress term for one policy row.

    Both terms are eligible only while the public task is valid and the signed
    contact clock is positive.  Fixed initial-center N1 deliberately disables
    base-position pay because spawn already equals the base goal and the term
    would be a constant reward rather than guidance.  The progress potential
    is the previous-minus-current distance to the *bare* contact point, with
    the same [0, 4.65] cap and reset-to-zero baseline rule as
    ``RacketTargetCommand._update_footwork_signals``.
    """

    if type(task_valid) is not bool or type(reset_progress_baseline) is not bool:
        raise A211EnvError("A211 footwork eligibility must use plain booleans")
    time_error = float(time_to_contact_s)
    previous_distance = float(previous_racket_distance_m)
    if not math.isfinite(time_error) or not math.isfinite(previous_distance):
        raise A211EnvError("A211 footwork clock/distance is non-finite")
    if previous_distance < 0.0:
        raise A211EnvError("A211 previous racket distance must be nonnegative")
    base = _finite_vector(
        achieved_base_position_w_m, 3, "A211 achieved base position"
    )
    base_target = _finite_vector(
        desired_base_position_w_m, 3, "A211 desired base position"
    )
    racket = _finite_vector(
        achieved_racket_position_w_m, 3, "A211 achieved racket position"
    )
    contact = _finite_vector(
        desired_contact_position_w_m, 3, "A211 desired contact position"
    )
    current_distance = float(np.linalg.norm(racket - contact))
    previous_potential = min(
        max(previous_distance, 0.0), A211_RACKET_PROGRESS_POTENTIAL_CAP_M
    )
    current_potential = min(
        max(current_distance, 0.0), A211_RACKET_PROGRESS_POTENTIAL_CAP_M
    )
    raw_progress = (
        0.0
        if reset_progress_baseline
        else previous_potential - current_potential
    )
    pre_strike_eligible = bool(task_valid and time_error > 0.0)
    base_xy_squared_error = float(np.sum(np.square(base[:2] - base_target[:2])))
    base_kernel = math.exp(
        -base_xy_squared_error / (A211_BASE_POSITION_STD_M**2)
    )
    base_reward = (
        A211_POLICY_DT_S * A211_BASE_POSITION_WEIGHT * base_kernel
        if pre_strike_eligible
        else 0.0
    )
    progress_reward = (
        A211_POLICY_DT_S * A211_RACKET_PROGRESS_WEIGHT * raw_progress
        if pre_strike_eligible
        else 0.0
    )
    total = base_reward + progress_reward
    if not all(
        math.isfinite(value)
        for value in (
            current_distance,
            raw_progress,
            base_xy_squared_error,
            base_kernel,
            base_reward,
            progress_reward,
            total,
        )
    ):
        raise A211EnvError("A211 pre-strike footwork reward is non-finite")
    return {
        "task_valid": task_valid,
        "time_to_contact_s": time_error,
        "pre_strike_eligible": pre_strike_eligible,
        "reset_progress_baseline": reset_progress_baseline,
        "base_xy_squared_error_m2": base_xy_squared_error,
        "base_kernel": base_kernel,
        "base_position_reward": base_reward,
        "previous_racket_distance_m": previous_distance,
        "current_racket_distance_m": current_distance,
        "previous_racket_potential_m": previous_potential,
        "current_racket_potential_m": current_potential,
        "racket_progress_raw_m": raw_progress,
        "racket_progress_reward": progress_reward,
        "post_policy_dt_reward": total,
    }


def a211_achieved_outcome_reward_terms(
    *,
    task_valid: bool,
    selected_contact_observed_now: bool,
    landing_terms: Mapping[str, Any] | None,
    net_target_center_z_w_m: float,
    counter_rally_required: bool = False,
) -> dict[str, Any]:
    """Resolve A's capture/net/dense/legal terms without C's off-table pay."""

    if type(task_valid) is not bool or type(selected_contact_observed_now) is not bool:
        raise A211EnvError("A211 achieved-outcome eligibility must be bool")
    target_z = float(net_target_center_z_w_m)
    if not math.isfinite(target_z):
        raise A211EnvError("A211 net target height is non-finite")
    capture = (
        A211_CAPTURE_POST_DT_WEIGHT
        if task_valid and selected_contact_observed_now
        else 0.0
    )
    pass_net = dense = legal_landing = 0.0
    classification = "task_invalid" if not task_valid else "no_achieved_flight"
    detail: dict[str, Any] | None = None
    if task_valid and landing_terms is not None:
        if not isinstance(landing_terms, Mapping):
            raise A211EnvError("A211 achieved landing terms must be a mapping")
        counter_components = landing_terms.get(
            "counter_rally_reward_components"
        )
        if counter_rally_required and counter_components is None:
            raise A211EnvError(
                "current A211 outcome omits fitted counter-rally components"
            )
        if counter_components is not None:
            try:
                components = tuple(float(value) for value in counter_components)
                landing_valid = bool(landing_terms["landing_valid"])
                kernel = float(landing_terms["kernel"])
            except (KeyError, TypeError, ValueError) as exc:
                raise A211EnvError(
                    "A211 counter-rally outcome facts are malformed"
                ) from exc
            if (
                len(components) != 5
                or not all(math.isfinite(value) for value in components)
                or any(value < 0.0 or value > 1.0 for value in components)
                or not math.isfinite(kernel)
                or not 0.0 <= kernel <= 1.0
            ):
                raise A211EnvError(
                    "A211 counter-rally outcome facts are invalid"
                )
            # Current Take061 enables counter-rally: Isaac hard-disables the
            # legacy pass-net scalar, keeps the small achieved-landing kernel,
            # and routes virtual_landing(legal_base) to the fitted rally total.
            pass_net = 0.0
            dense = A211_LANDING_DENSE_POST_DT_WEIGHT * (
                kernel if landing_valid else 0.0
            )
            legal_landing = (
                A211_LEGAL_LANDING_POST_DT_WEIGHT * components[4]
            )
            classification = str(
                landing_terms.get("classification", "counter_rally_outcome")
            )
            detail = copy.deepcopy(dict(landing_terms))
            detail["classification"] = classification
            detail["opponent_side_off_table_reward"] = 0.0
            total = capture + dense + legal_landing
            if not math.isfinite(total):
                raise A211EnvError(
                    "A211 counter-rally achieved reward is non-finite"
                )
            return {
                "task_valid": task_valid,
                "selected_contact_observed_now": (
                    selected_contact_observed_now
                ),
                "capture_reward": capture,
                "pass_net_reward": pass_net,
                "landing_dense_reward": dense,
                "legal_landing_reward": legal_landing,
                "counter_rally_outcome_reward": legal_landing,
                "landing_terms": detail,
                "classification": classification,
                "post_policy_dt_reward": total,
            }
        try:
            landing_valid = bool(landing_terms["landing_valid"])
            net_crossed = bool(landing_terms["net_crossed"])
            net_clear = bool(landing_terms["net_clear"])
            legal = bool(landing_terms["legal_opponent_table"])
            kernel = float(landing_terms["kernel"])
            net_z = float(landing_terms["net_z_w_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise A211EnvError("A211 achieved landing facts are malformed") from exc
        if not all(math.isfinite(value) for value in (kernel, net_z)) or not (
            0.0 <= kernel <= 1.0
        ):
            raise A211EnvError("A211 achieved landing facts are invalid")
        height_kernel = math.exp(-((net_z - target_z) / A211_NET_SIGMA_M) ** 2)
        pass_raw = (height_kernel if net_crossed else 0.0) + (
            0.5 if legal and net_clear else 0.0
        )
        pass_net = A211_PASS_NET_POST_DT_WEIGHT * pass_raw
        dense = A211_LANDING_DENSE_POST_DT_WEIGHT * (
            kernel if landing_valid else 0.0
        )
        legal_raw = (
            A211_LANDING_LEGAL_BASE_FRAC
            + (1.0 - A211_LANDING_LEGAL_BASE_FRAC) * kernel
            if legal and net_clear
            else 0.0
        )
        legal_landing = A211_LEGAL_LANDING_POST_DT_WEIGHT * legal_raw
        classification = str(
            landing_terms.get(
                "classification",
                (
                    "legal_opponent_table"
                    if legal and net_clear
                    else "nonlegal_zero_legal_prize"
                ),
            )
        )
        detail = {
            **copy.deepcopy(dict(landing_terms)),
            "classification": classification,
            "height_kernel": height_kernel,
            "pass_net_raw": pass_raw,
            "legal_landing_raw": legal_raw,
            "opponent_side_off_table_reward": 0.0,
        }
    total = capture + pass_net + dense + legal_landing
    if not math.isfinite(total):
        raise A211EnvError("A211 achieved-outcome reward is non-finite")
    return {
        "task_valid": task_valid,
        "selected_contact_observed_now": selected_contact_observed_now,
        "capture_reward": capture,
        "pass_net_reward": pass_net,
        "landing_dense_reward": dense,
        "legal_landing_reward": legal_landing,
        "counter_rally_outcome_reward": 0.0,
        "landing_terms": detail,
        "classification": classification,
        "post_policy_dt_reward": total,
    }


class MujocoA211DiagnosticVecEnv(shared.MujocoC211DiagnosticVecEnv):
    """Asymmetric A211 adapter around the existing fixed-centre real VecEnv."""

    def __init__(
        self,
        *,
        base_env: Any,
        task_authority: A211TaskAuthority,
        mimic_authority: MeasuredA211MimicAuthority,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise A211EnvError("torch is required for the A211 VecEnv") from exc
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
            raise A211EnvError(f"fixed-center base omits {missing!r}")
        native = base_env.base_env
        if not hasattr(native, "cores") or not hasattr(native, "robot_tape"):
            raise A211EnvError("A211 adapter requires real MujocoN1BallCore rows")
        self.base = base_env
        self.num_envs = int(base_env.num_envs)
        self.num_actions = int(base_env.num_actions)
        self.num_observations = abi.ACTOR_WIDTH
        self.num_privileged_observations = abi.CRITIC_WIDTH
        self.device = torch.device("cpu")
        self.unwrapped = self
        self.cfg = {
            **copy.deepcopy(getattr(base_env, "cfg", {})),
            "kind": A211_ENV_KIND,
            "actor_observation_width": abi.ACTOR_WIDTH,
            "critic_observation_width": abi.CRITIC_WIDTH,
            "reward_scope": A211_REWARD_SCOPE,
            "a211_desired_contact_reward_available": True,
            "a211_achieved_outcome_reward_available": True,
            "a211_base_position_reward_enabled": False,
            "a211_prestrike_racket_progress_reward_available": True,
            "isaac_synonymous_prior_subset_available": True,
            "complete_isaac_reward_parity_claimed": False,
            "true_a211_training_lane_ready": False,
            "safe_ready_authority_status": shared.SAFE_READY_AUTHORITY_STATUS,
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
            raise A211EnvError(
                "A211 requires the authoritative continuous WAIT schedule and "
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
                canonical_sha256=shared._sha256_json(
                    {"kind": "legacy_fixed_wait_test_double", "wait_ticks": fixed_wait}
                ),
            )
        self.producer = A211ObservationProducer(
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
            shared._EpisodeRewardState() for _index in range(self.num_envs)
        ]
        self._previous_racket_distances = self._a211_current_racket_distances()
        self._suppress_next_racket_progress = np.ones(
            self.num_envs, dtype=np.bool_
        )
        self._reward_contract = self._build_reward_contract(task_authority)
        self._reward_audit: dict[str, Any] = {}
        self.reset_reward_audit()
        self._actor = None
        self._critic = None
        self._canonical_boundary_sha256: str | None = None
        self._current_boundary_state_sha256: str | None = None
        self._identity: dict[str, str] | None = None
        self._install_current_boundary(initial=True)

    def _a211_current_racket_distances(
        self, env_ids: Sequence[int] | None = None
    ) -> np.ndarray:
        ids = (
            tuple(range(self.num_envs))
            if env_ids is None
            else tuple(int(value) for value in env_ids)
        )
        if any(value < 0 or value >= self.num_envs for value in ids):
            raise A211EnvError("A211 progress reset row index is invalid")
        target = np.asarray(
            self.producer.task.desired_contact_position_w_m, dtype=np.float64
        )
        values = np.asarray(
            [
                np.linalg.norm(self.producer._live(index)["racket_pos"] - target)
                for index in ids
            ],
            dtype=np.float64,
        )
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise A211EnvError("A211 progress reset distance is invalid")
        return values

    def _reset_a211_progress_rows(self, env_ids: Sequence[int]) -> None:
        ids = tuple(int(value) for value in env_ids)
        values = self._a211_current_racket_distances(ids)
        for index, value in zip(ids, values.tolist()):
            self._previous_racket_distances[index] = value
            self._suppress_next_racket_progress[index] = True

    def _build_reward_contract(self, task: A211TaskAuthority) -> dict[str, Any]:
        # Reuse the live event/geometry/source checks, then replace every C task
        # claim before the contract is exposed or hashed.
        payload = super()._build_reward_contract(task)
        payload.pop("content_sha256", None)
        counter_module = _load_source_module(
            "_action_ball_a211_native_counter_rally", COUNTER_RALLY_PY
        )
        counter_torch = _load_source_module(
            "_action_ball_a211_native_counter_rally_torch",
            COUNTER_RALLY_TORCH_PY,
        )
        try:
            objective = counter_module.CounterRallyObjectiveProfile()
            physics = counter_module.VenueBallPhysics.from_venue_yaml(
                shared.VENUE_PHYSICS_YAML
            )
            binding = counter_torch.CounterRallyTorchBinding.from_mappings(
                objective_profile=objective.to_mapping(),
                venue_physics=physics.to_mapping(),
                expected_objective_profile_sha256=(
                    task.counter_rally_objective_profile_sha256
                ),
                expected_venue_physics_sha256=physics.sha256,
            )
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise A211EnvError(
                f"A211 counter-rally reward authority is invalid: {exc}"
            ) from exc
        geometry = {
            "table_near_x_env_m": self._table_near_x,
            "table_length_m": self._table_far_x - self._table_near_x,
            "table_half_width_m": self._table_half_width,
            "table_surface_z_env_m": self._table_surface_z,
            "net_height_m": self._net_clear_center_z
            - self._table_surface_z
            - self._ball_radius_m,
            "opponent_baseline_x_env_m": self._table_far_x,
        }
        drift = {
            name: (float(getattr(objective, name)), float(value))
            for name, value in geometry.items()
            if not math.isclose(
                float(getattr(objective, name)),
                float(value),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        }
        if drift or not math.isclose(
            float(physics.ball_radius_m),
            self._ball_radius_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise A211EnvError(
                f"A211 counter-rally geometry differs from native scene: {drift}"
            )
        self._counter_rally_module = counter_module
        self._counter_rally_torch = counter_torch
        self._counter_rally_objective = objective
        self._counter_rally_physics = physics
        self._counter_rally_binding = binding
        payload["kind"] = A211_REWARD_CONTRACT_IDENTITY
        payload["scope"] = A211_REWARD_SCOPE
        payload["task_reward_contract_identity"] = (
            A211_TASK_REWARD_CONTRACT_IDENTITY
        )
        payload["desired_contact_position_velocity_face_consumed"] = True
        payload["desired_contact"] = {
            "position_w_m": list(task.desired_contact_position_w_m),
            "velocity_w_mps": list(task.desired_contact_velocity_w_mps),
            "signed_face_w": list(task.desired_contact_signed_face_w),
            "target_validity_mask": [True, True, True],
            "target_recipe": task.target_recipe,
        }
        payload.pop("strike_bridge", None)
        payload["desired_contact_window_reward"] = {
            "position_half_window_s": A211_POSITION_HALF_WINDOW_S,
            "velocity_face_half_window_s": A211_WIDE_HALF_WINDOW_S,
            "target_position_semantics": "desired_position_minus_velocity_times_time_to_contact",
            "channels": copy.deepcopy(A211_TARGET_CHANNELS),
            "adaptive_sigma_enabled": False,
            "task_valid_required": True,
            "policy_dt_s": A211_POLICY_DT_S,
        }
        payload["prestrike_footwork_reward"] = {
            "task_valid_required": True,
            "pre_strike_rule": "signed_time_to_contact_strictly_greater_than_zero",
            "base_position": {
                "enabled": False,
                "reason": (
                    "fixed_initial_center_spawn_equals_base_goal_constant_pay_"
                    "is_not_guidance"
                ),
                "coordinates": "world_xy",
                "std_m": A211_BASE_POSITION_STD_M,
                "reward_manager_weight": A211_BASE_POSITION_WEIGHT,
                "policy_dt_s": A211_POLICY_DT_S,
                "raw": "exp(-squared_xy_error/std^2)",
            },
            "racket_progress": {
                "target": "bare_desired_contact_position_not_swing_through_line",
                "potential": "euclidean_racket_to_contact_distance",
                "potential_cap_m": A211_RACKET_PROGRESS_POTENTIAL_CAP_M,
                "raw": "previous_capped_potential_minus_current_capped_potential",
                "reset_or_question_change_raw": 0.0,
                "reward_manager_weight": A211_RACKET_PROGRESS_WEIGHT,
                "policy_dt_s": A211_POLICY_DT_S,
            },
        }
        payload["landing"] = {
            "source": (
                "first_source_bound_outgoing_flight_after_actual_selected_rubber_contact_then_fitted_1ms_counter_rally"
            ),
            "task_valid_required": True,
            "one_evaluation_per_attempt": True,
            "counter_rally_enabled": True,
            "counter_rally_task_canonical_sha256": (
                task.counter_rally_task_canonical_sha256
            ),
            "counter_rally_objective_profile_sha256": (
                task.counter_rally_objective_profile_sha256
            ),
            "counter_rally_venue_physics_sha256": physics.sha256,
            "counter_rally_raw": (
                "0.60*legal+0.05*landing_sigma_0.03+0.10*reverse+0.25*speed"
            ),
            "counter_rally_post_policy_dt_weight": (
                A211_LEGAL_LANDING_POST_DT_WEIGHT
            ),
            "opponent_side_off_table_raw": 0.0,
            "own_side_backwards_net_or_invalid_raw": 0.0,
            "landing_sigma_m": A211_LANDING_SIGMA_M,
            "capture_post_policy_dt_weight": A211_CAPTURE_POST_DT_WEIGHT,
            "pass_net_post_policy_dt_weight": 0.0,
            "pass_net_disabled_by_counter_rally": True,
            "landing_dense_post_policy_dt_weight": (
                A211_LANDING_DENSE_POST_DT_WEIGHT
            ),
        }
        payload["wait_reward_semantics"] = {
            "balance_action_body_and_measured_paddle_priors_active": True,
            "desired_contact_and_outcome_masked": True,
            "base_position_disabled_and_racket_progress_masked": True,
            "task_valid_applied_to_priors": False,
        }
        payload["cross_engine_reward_semantic_gaps"] = [
            dict(row) for row in A211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
        ]
        semantic = dict(payload.get("semantic_authorities", {}))
        semantic.pop("c211_trainability_source_sha256", None)
        semantic.pop("c211_task_yaml_sha256", None)
        semantic["a211_trainability_source_sha256"] = shared._sha256_file(
            shared.C211_TRAINABILITY_PY.parent / "action_ball_a211_trainability.py"
        )
        semantic["a211_task_yaml_sha256"] = shared._sha256_file(
            shared.C211_TASK_YAML.parent
            / "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml"
        )
        semantic["counter_rally_source_sha256"] = shared._sha256_file(
            COUNTER_RALLY_PY
        )
        semantic["counter_rally_torch_source_sha256"] = shared._sha256_file(
            COUNTER_RALLY_TORCH_PY
        )
        semantic["counter_rally_objective_profile_sha256"] = objective.sha256
        semantic["counter_rally_venue_physics_sha256"] = physics.sha256
        payload["semantic_authorities"] = semantic
        payload["content_sha256"] = shared._sha256_json(payload)
        return payload

    def _counter_rally_outcome_terms(self, flight: Any) -> dict[str, Any]:
        """Run the current Isaac fitted rally scorer on one achieved flight."""

        try:
            import torch

            position = torch.as_tensor(
                [flight.position_w_m], dtype=torch.float32, device="cpu"
            )
            velocity = torch.as_tensor(
                [flight.linear_velocity_w_mps],
                dtype=torch.float32,
                device="cpu",
            )
            spin = torch.as_tensor(
                [flight.spin_w_radps], dtype=torch.float32, device="cpu"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise A211EnvError(
                "A211 counter-rally requires one achieved outgoing flight"
            ) from exc
        if not (
            torch.isfinite(position).all()
            and torch.isfinite(velocity).all()
            and torch.isfinite(spin).all()
        ):
            raise A211EnvError("A211 counter-rally flight is non-finite")
        task = self.producer.task
        landing_aim = torch.as_tensor(
            [task.landing_aim_w_xy_m], dtype=torch.float32, device="cpu"
        )
        direction = torch.as_tensor(
            [task.counter_rally_return_direction_env_xy],
            dtype=torch.float32,
            device="cpu",
        )
        target_speed = torch.as_tensor(
            [task.counter_rally_target_baseline_speed_mps],
            dtype=torch.float32,
            device="cpu",
        )
        contact_valid = torch.ones((1,), dtype=torch.bool, device="cpu")
        objective_shas = (task.counter_rally_objective_profile_sha256,)
        try:
            with torch.no_grad():
                outcome = self._counter_rally_torch.rollout_counter_rally_torch(
                    position,
                    velocity,
                    spin,
                    binding=self._counter_rally_binding,
                    dt_s=0.001,
                    max_time_s=2.0,
                )
                rewards = self._counter_rally_torch.counter_rally_reward_raw_torch(
                    binding=self._counter_rally_binding,
                    landing_aim_env_xy_m=landing_aim,
                    return_direction_env_xy=direction,
                    target_baseline_speed_mps=target_speed,
                    paddle_contact_valid=contact_valid,
                    task_objective_profile_sha256=objective_shas,
                    outcome=outcome,
                )
                gates = self._counter_rally_torch.counter_rally_outcome_gates_torch(
                    outcome=outcome,
                    binding=self._counter_rally_binding,
                    landing_aim_env_xy_m=landing_aim,
                    return_direction_env_xy=direction,
                    target_baseline_speed_mps=target_speed,
                    paddle_contact_valid=contact_valid,
                    task_objective_profile_sha256=objective_shas,
                )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise A211EnvError(f"A211 counter-rally rollout failed: {exc}") from exc
        components = tuple(float(value) for value in rewards[0].tolist())
        landing_valid = bool(outcome.first_landing_valid[0].item())
        landing_xy_raw = tuple(
            float(value) for value in outcome.first_landing_env_xy_m[0].tolist()
        )
        landing_xy_finite = all(
            math.isfinite(value) for value in landing_xy_raw
        )
        if landing_xy_finite:
            landing_xy: list[float] | None = list(landing_xy_raw)
            aim = np.asarray(task.landing_aim_w_xy_m, dtype=np.float64)
            dist2 = float(
                np.sum(np.square(np.asarray(landing_xy_raw) - aim))
            )
            dense_kernel = math.exp(
                -dist2 / (A211_LANDING_SIGMA_M**2)
            )
        else:
            landing_xy = None
            dist2 = None
            dense_kernel = 0.0
        legal = bool(gates.legal_first_landing[0].item())
        objective = self._counter_rally_objective
        on_table = bool(
            landing_xy is not None
            and landing_xy[0]
            >= objective.table_near_x_env_m + objective.table_edge_margin_m
            and landing_xy[0]
            <= self._table_far_x - objective.table_edge_margin_m
            and abs(landing_xy[1])
            <= objective.table_half_width_m - objective.table_edge_margin_m
        )
        opponent_off_table = bool(
            landing_xy is not None
            and landing_xy[0] > self._table_net_x
            and not on_table
        )
        reason_code = int(gates.primary_reason_code[0].item())
        reason_names = tuple(self._counter_rally_torch.OUTCOME_PRIMARY_REASONS)
        if reason_code < 0 or reason_code >= len(reason_names):
            raise A211EnvError("A211 counter-rally reason code is invalid")
        classification = (
            "legal_opponent_table"
            if legal
            else (
                "opponent_side_off_table"
                if opponent_off_table
                else "zero_ineligible_or_nonopponent"
            )
        )
        return {
            "source": "fitted_counter_rally_torch_1ms",
            "landing_xy_w_m": landing_xy,
            "landing_aim_w_xy_m": list(task.landing_aim_w_xy_m),
            "squared_error_m2": dist2,
            "kernel": dense_kernel,
            "landing_valid": landing_valid,
            "net_crossed": bool(outcome.net_crossed[0].item()),
            "net_clear": bool(outcome.net_clear[0].item()),
            "legal_opponent_table": legal,
            "opponent_side_off_table": opponent_off_table,
            "classification": classification,
            "counter_rally_accepted": bool(gates.accepted[0].item()),
            "counter_rally_primary_reason": reason_names[reason_code],
            "counter_rally_reward_components": list(components),
            "counter_rally_component_names": [
                "legal",
                "landing",
                "reverse",
                "speed",
                "total",
            ],
        }

    def _evaluate_reward_transition(
        self,
        *,
        base_extras: Mapping[str, Any],
        transition_valid: tuple[bool, ...],
        prior_rows: Sequence[Mapping[str, Any]],
        dones: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        # The shared evaluator owns actual-contact/achieved-flight validation and
        # state latches.  Remove its C-only single-tick proximity scalar and
        # install the A window target scalar from the same final-substep sample.
        rewards, rows = super()._evaluate_reward_transition(
            base_extras=base_extras,
            transition_valid=transition_valid,
            prior_rows=prior_rows,
            dones=dones,
        )
        try:
            facts_rows = tuple(
                base_extras["diagnostic_native_physical_event_facts"]
            )
        except (KeyError, TypeError) as exc:
            raise A211EnvError("A211 transition omits event facts") from exc
        if len(facts_rows) != self.num_envs:
            raise A211EnvError("A211 event-fact row count differs")
        values: list[float] = []
        next_racket_distances: list[float] = []
        for index, (row, prior_row) in enumerate(zip(rows, prior_rows)):
            sample = prior_row.get("achieved_contact_sample")
            if not isinstance(sample, Mapping):
                raise A211EnvError(
                    f"env {index} A211 achieved contact sample is unavailable"
                )
            time_to_contact = (
                int(row["nominal_strike_tick"])
                - int(row["sample_policy_tick_1based"])
            ) * A211_POLICY_DT_S
            footwork_sample = prior_row.get("a211_footwork_sample")
            if not isinstance(footwork_sample, Mapping):
                raise A211EnvError(
                    f"env {index} A211 footwork sample is unavailable"
                )
            footwork_terms = a211_prestrike_footwork_reward_terms(
                task_valid=transition_valid[index],
                time_to_contact_s=time_to_contact,
                achieved_base_position_w_m=footwork_sample.get(
                    "base_position_w_m"
                ),
                desired_base_position_w_m=self.producer.task.base_goal_w_m,
                achieved_racket_position_w_m=footwork_sample.get(
                    "racket_position_w_m"
                ),
                desired_contact_position_w_m=(
                    self.producer.task.desired_contact_position_w_m
                ),
                previous_racket_distance_m=float(
                    self._previous_racket_distances[index]
                ),
                reset_progress_baseline=bool(
                    self._suppress_next_racket_progress[index]
                ),
            )
            next_racket_distances.append(
                float(footwork_terms["current_racket_distance_m"])
            )
            target_terms = a211_desired_contact_reward_terms(
                task_valid=transition_valid[index],
                time_to_contact_s=time_to_contact,
                achieved_position_w_m=sample.get("position_w_m"),
                achieved_velocity_w_mps=sample.get("velocity_w_mps"),
                achieved_signed_face_w=sample.get("signed_face_w"),
                desired_position_w_m=self.producer.task.desired_contact_position_w_m,
                desired_velocity_w_mps=self.producer.task.desired_contact_velocity_w_mps,
                desired_signed_face_w=self.producer.task.desired_contact_signed_face_w,
            )
            c_only_reward = float(row["strike_reward"])
            c_only_landing_reward = float(row["landing_reward"])
            target_reward = float(target_terms["post_policy_dt_reward"])
            achieved_landing = None
            if isinstance(row.get("landing_terms"), Mapping):
                _contact, flight = self._event_evidence(index, facts_rows[index])
                if not flight.valid:
                    raise A211EnvError(
                        f"env {index} A211 landing has no achieved flight"
                    )
                achieved_landing = self._counter_rally_outcome_terms(flight)
            outcome_terms = a211_achieved_outcome_reward_terms(
                task_valid=transition_valid[index],
                selected_contact_observed_now=(
                    row.get("eligible_contact_observed_now") is True
                ),
                landing_terms=achieved_landing,
                net_target_center_z_w_m=(
                    self._net_clear_center_z
                    - self._ball_radius_m
                    + A211_NET_MARGIN_M
                ),
                counter_rally_required=achieved_landing is not None,
            )
            outcome_reward = float(outcome_terms["post_policy_dt_reward"])
            outcome_quality_reward = float(outcome_terms["pass_net_reward"]) + float(
                outcome_terms["landing_dense_reward"]
            ) + float(outcome_terms["legal_landing_reward"])
            total = (
                float(row["total_reward"])
                - c_only_reward
                - c_only_landing_reward
                + target_reward
                + outcome_reward
                + float(footwork_terms["post_policy_dt_reward"])
            )
            if not math.isfinite(total):
                raise A211EnvError(f"env {index} A211 reward is non-finite")
            row["c211_single_tick_proximity_reward_removed"] = c_only_reward
            row["c211_landing_reward_removed"] = c_only_landing_reward
            row["desired_contact_terms"] = target_terms
            row["desired_contact_reward"] = target_reward
            row["prestrike_footwork_terms"] = footwork_terms
            row["base_position_reward"] = footwork_terms[
                "base_position_reward"
            ]
            row["racket_progress_reward"] = footwork_terms[
                "racket_progress_reward"
            ]
            row["achieved_outcome_terms"] = outcome_terms
            row["capture_reward"] = outcome_terms["capture_reward"]
            row["pass_net_reward"] = outcome_terms["pass_net_reward"]
            row["landing_dense_reward"] = outcome_terms[
                "landing_dense_reward"
            ]
            row["legal_landing_reward"] = outcome_terms[
                "legal_landing_reward"
            ]
            row["counter_rally_outcome_reward"] = outcome_terms[
                "counter_rally_outcome_reward"
            ]
            row["achieved_outcome_reward"] = outcome_reward
            # Keep the generic audit key, but its A contract is the complete
            # target-window scalar rather than C's nominal proximity bridge.
            row["strike_terms"] = (
                target_terms if target_terms["any_channel_eligible"] else None
            )
            row["strike_reward"] = target_reward
            row["landing_terms"] = outcome_terms["landing_terms"]
            # Keep generic landing accounting free of the capture-only bonus;
            # the A-specific audit records every outcome component separately.
            row["landing_reward"] = outcome_quality_reward
            additive_components = {
                "isaac_synonymous_prior_reward": float(
                    row["isaac_synonymous_prior_reward"]
                ),
                "desired_contact_reward": target_reward,
                "base_position_reward": float(
                    footwork_terms["base_position_reward"]
                ),
                "racket_progress_reward": float(
                    footwork_terms["racket_progress_reward"]
                ),
                "capture_reward": float(outcome_terms["capture_reward"]),
                "pass_net_reward": float(outcome_terms["pass_net_reward"]),
                "landing_dense_reward": float(
                    outcome_terms["landing_dense_reward"]
                ),
                "legal_landing_reward": float(
                    outcome_terms["legal_landing_reward"]
                ),
            }
            if not math.isclose(
                sum(additive_components.values()),
                total,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise A211EnvError(
                    f"env {index} A211 additive reward accounting differs"
                )
            row["additive_reward_components"] = additive_components
            row["nonadditive_alias_map"] = {
                "strike_reward": "desired_contact_reward",
                "landing_reward": (
                    "pass_net_reward+landing_dense_reward+legal_landing_reward"
                ),
                "achieved_outcome_reward": (
                    "capture_reward+pass_net_reward+landing_dense_reward+"
                    "legal_landing_reward"
                ),
                "counter_rally_outcome_reward": (
                    "legal_landing_reward_when_counter_rally_is_bound_else_zero"
                ),
            }
            row["total_reward"] = total
            values.append(total)
        self._previous_racket_distances[:] = np.asarray(
            next_racket_distances, dtype=np.float64
        )
        self._suppress_next_racket_progress[:] = False
        try:
            import torch
        except ImportError as exc:
            raise A211EnvError("torch is required for the A211 reward") from exc
        return torch.as_tensor(values, dtype=torch.float32, device="cpu"), rows

    def reset_reward_audit(self) -> None:
        super().reset_reward_audit()
        self._reward_audit.update(
            {
                "desired_contact_any_window_row_count": 0,
                "desired_contact_position_window_row_count": 0,
                "desired_contact_velocity_window_row_count": 0,
                "desired_contact_face_window_row_count": 0,
                "desired_contact_reward_sum": 0.0,
                "prestrike_footwork_eligible_row_count": 0,
                "base_position_reward_sum": 0.0,
                "racket_progress_reward_sum": 0.0,
                "racket_progress_raw_sum_m": 0.0,
                "capture_reward_row_count": 0,
                "counter_rally_outcome_evaluation_count": 0,
                "capture_reward_sum": 0.0,
                "pass_net_reward_sum": 0.0,
                "landing_dense_reward_sum": 0.0,
                "counter_rally_outcome_reward_sum": 0.0,
                "achieved_outcome_reward_sum": 0.0,
            }
        )

    def _accumulate_reward_audit(
        self, reward_rows: Sequence[Mapping[str, Any]], dones: Any
    ) -> None:
        super()._accumulate_reward_audit(reward_rows, dones)
        for row in reward_rows:
            target = row.get("desired_contact_terms")
            if not isinstance(target, Mapping):
                raise A211EnvError("A211 reward audit omits desired-contact terms")
            channels = target.get("channels")
            if not isinstance(channels, Mapping) or tuple(channels) != (
                "position",
                "velocity",
                "face",
            ):
                raise A211EnvError("A211 reward audit channel order differs")
            self._reward_audit["desired_contact_any_window_row_count"] += int(
                target.get("any_channel_eligible") is True
            )
            for name in ("position", "velocity", "face"):
                self._reward_audit[
                    f"desired_contact_{name}_window_row_count"
                ] += int(channels[name].get("eligible") is True)
            value = float(target.get("post_policy_dt_reward"))
            if not math.isfinite(value):
                raise A211EnvError("A211 audited target reward is non-finite")
            self._reward_audit["desired_contact_reward_sum"] += value
            footwork = row.get("prestrike_footwork_terms")
            if not isinstance(footwork, Mapping):
                raise A211EnvError("A211 reward audit omits pre-strike footwork terms")
            self._reward_audit[
                "prestrike_footwork_eligible_row_count"
            ] += int(footwork.get("pre_strike_eligible") is True)
            for name in ("base_position_reward", "racket_progress_reward"):
                component = float(footwork.get(name))
                if not math.isfinite(component):
                    raise A211EnvError(f"A211 audited {name} is non-finite")
                self._reward_audit[f"{name}_sum"] += component
            raw_progress = float(footwork.get("racket_progress_raw_m"))
            if not math.isfinite(raw_progress):
                raise A211EnvError("A211 audited racket progress is non-finite")
            self._reward_audit["racket_progress_raw_sum_m"] += raw_progress
            component_names = (
                "capture_reward",
                "pass_net_reward",
                "landing_dense_reward",
                "counter_rally_outcome_reward",
                "achieved_outcome_reward",
            )
            components: dict[str, float] = {}
            for name in component_names:
                value = float(row.get(name))
                if not math.isfinite(value):
                    raise A211EnvError(
                        f"A211 audited {name} is non-finite"
                    )
                components[name] = value
                self._reward_audit[f"{name}_sum"] += value
            self._reward_audit["capture_reward_row_count"] += int(
                components["capture_reward"] > 0.0
            )
            self._reward_audit[
                "counter_rally_outcome_evaluation_count"
            ] += int(isinstance(row.get("landing_terms"), Mapping))

    def reward_audit_receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "kind": "action_ball_a211_mujoco_raw_reward_audit_v1",
            "reward_scope": A211_REWARD_SCOPE,
            "reward_contract_sha256": self._reward_contract["content_sha256"],
            "reward_parity_status": "partial_fail_closed",
            "closed_attempt_denominator_semantics": (
                "done_and_TASK_ACTIVE_transition_only_RESET_WAIT_done_excluded"
            ),
            "additive_reward_component_names": [
                "isaac_synonymous_prior_reward",
                "desired_contact_reward",
                "base_position_reward",
                "racket_progress_reward",
                "capture_reward",
                "pass_net_reward",
                "landing_dense_reward",
                "legal_landing_reward",
            ],
            "nonadditive_alias_map": {
                "strike_reward": "desired_contact_reward",
                "landing_reward": (
                    "pass_net_reward+landing_dense_reward+legal_landing_reward"
                ),
                "achieved_outcome_reward": (
                    "capture_reward+pass_net_reward+landing_dense_reward+"
                    "legal_landing_reward"
                ),
                "counter_rally_outcome_reward": (
                    "legal_landing_reward_when_counter_rally_is_bound_else_zero"
                ),
            },
            **copy.deepcopy(self._reward_audit),
            "complete_isaac_reward_parity_claimed": False,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        payload["content_sha256"] = shared._sha256_json(payload)
        return payload

    def _install_current_boundary(self, *, initial: bool) -> None:
        boundary = self.base.is_reset_boundary()
        task_valid = self._task_valid_rows()
        if type(boundary) is not bool or boundary is not True or any(task_valid):
            raise A211EnvError(
                "A211 canonical observation requires an inactive reset boundary"
            )
        actor, critic = self.producer.tensors(task_valid)
        self._actor, self._critic = actor, critic
        digest = self.producer.boundary_state_sha256(actor, critic, task_valid)
        if initial:
            self._canonical_boundary_sha256 = shared._sha256_json(
                {
                    "schema_version": 1,
                    "kind": "action_ball_a211_seeded_wait_boundary_contract_v1",
                    "wait_schedule_sha256": self._wait_schedule.canonical_sha256,
                    "wait_preparation_sha256": getattr(
                        self.base,
                        "_continuous_wait_preparation_sha256",
                        self.base.diagnostic_training_identity()[
                            "reward_contract_sha256"
                        ],
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
        observation = shared._sha256_json(
            {
                "schema_version": 1,
                "kind": "action_ball_a211_mujoco_observation_contract_v1",
                "profile_observation_contract_sha256": (
                    abi.A211_PROFILE.observation_contract_sha256
                ),
                "authority_sha256": self.producer.authorities.content_sha256,
                "canonical_reset_boundary_sha256": boundary_sha256,
                "actor_dtype": "torch.float32",
                "critic_dtype": "torch.float32",
                "device": "cpu",
                "diagnostic_unauthorized": True,
            }
        )
        contract = shared._sha256_json(
            {
                "schema_version": 1,
                "kind": "action_ball_a211_mujoco_partial_isaac_reward_contract_v1",
                "base_contract_sha256": base["contract_sha256"],
                "observation_contract_sha256": observation,
                "action_contract_sha256": base["action_contract_sha256"],
                "wrapped_reward_contract_sha256_not_consumed": base[
                    "reward_contract_sha256"
                ],
                "reward_contract_sha256": self._reward_contract["content_sha256"],
                "reward_scope": A211_REWARD_SCOPE,
                "a211_desired_contact_reward_available": True,
                "a211_achieved_outcome_reward_available": True,
                "isaac_synonymous_prior_subset_available": True,
                "complete_isaac_reward_parity_claimed": False,
                "actor_normalizer_identity": (
                    abi.A211_PROFILE.actor_normalizer_identity
                ),
                "critic_normalizer_identity": (
                    abi.A211_PROFILE.critic_normalizer_identity
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
        result = super().reset(seed=seed)
        self._reset_a211_progress_rows(range(self.num_envs))
        return result

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        actor, rewards, dones, extras = super().step(actions)
        reset_ids = tuple(
            index for index, value in enumerate(dones.tolist()) if bool(value)
        )
        if reset_ids:
            # The final-substep row above intentionally used the terminal
            # potential.  Compact reset has now installed the next physical
            # birth, so seed the next episode from that live reset state.
            self._reset_a211_progress_rows(reset_ids)
        extras.pop("terminal_c211_observation_available", None)
        extras.pop("c211_achieved_outcome_reward_available", None)
        extras.pop("true_c211_training_lane_ready", None)
        extras["terminal_a211_observation_available"] = False
        extras["reward_scope"] = A211_REWARD_SCOPE
        extras["a211_desired_contact_reward_available"] = True
        extras["a211_achieved_outcome_reward_available"] = True
        extras["a211_base_position_reward_enabled"] = False
        extras["a211_prestrike_racket_progress_reward_available"] = True
        extras["cross_engine_reward_semantic_gaps"] = [
            dict(row) for row in A211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
        ]
        extras["true_a211_training_lane_ready"] = False
        return actor, rewards, dones, extras

    def diagnostic_training_receipt(self) -> dict[str, Any]:
        base = self.base.diagnostic_training_receipt()
        if base.get("ppo_ready") is not True:
            raise A211EnvError("fixed-center base is not diagnostic PPO-ready")
        terminal_contract = trainer.terminal_row_telemetry_contract()
        if (
            base.get("terminal_row_telemetry_available") is not True
            or base.get("terminal_row_telemetry_contract") != terminal_contract
        ):
            raise A211EnvError(
                "fixed-center base omits exact terminal-row telemetry"
            )
        identity = self.diagnostic_training_identity()
        return {
            "schema_version": 1,
            "kind": trainer.DIAGNOSTIC_TRAINER_RECEIPT_KIND,
            "ppo_ready": True,
            "reward_available": True,
            "reward_scope": A211_REWARD_SCOPE,
            "a211_desired_contact_reward_available": True,
            "a211_achieved_outcome_reward_available": True,
            "a211_base_position_reward_enabled": False,
            "a211_prestrike_racket_progress_reward_available": True,
            "isaac_synonymous_prior_subset_available": True,
            "complete_isaac_reward_parity_claimed": False,
            "true_a211_training_lane_ready": False,
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
            "actor_normalizer_identity": abi.A211_PROFILE.actor_normalizer_identity,
            "critic_normalizer_identity": abi.A211_PROFILE.critic_normalizer_identity,
            "normalizer_binding": trainer.asymmetric_normalizer_binding(
                profile_observation_contract_sha256=(
                    abi.A211_PROFILE.observation_contract_sha256
                ),
                actor_width=abi.ACTOR_WIDTH,
                critic_width=abi.CRITIC_WIDTH,
                actor_normalizer_identity=(
                    abi.A211_PROFILE.actor_normalizer_identity
                ),
                critic_normalizer_identity=(
                    abi.A211_PROFILE.critic_normalizer_identity
                ),
                actor_task_mask_indices=abi.A211_PROFILE.actor.task_mask_indices,
                critic_task_mask_indices=abi.A211_PROFILE.critic.task_mask_indices,
                actor_task_valid_index=abi.A211_PROFILE.actor.task_valid_index,
                critic_task_valid_index=abi.A211_PROFILE.critic.task_valid_index,
                epsilon=1.0e-5,
            ),
            "fresh_actor_bootstrap": self.fresh_actor_bootstrap_contract(),
            "observation_authorities_sha256": (
                self.producer.authorities.content_sha256
            ),
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
            "full_body_measured_mimic_observation_available": True,
            "full_body_measured_mimic_reward_available": True,
            "measured_paddle_prior_reward_available": True,
            "reward_parity_status": "partial_fail_closed",
            "cross_engine_reward_semantic_gaps": [
                dict(row) for row in A211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
            ],
            "unavailable_isaac_reward_terms": [
                dict(row) for row in shared.C211_UNAVAILABLE_ISAAC_REWARD_TERMS
            ],
            "terminal_a211_observation_available": False,
            "safe_ready_authority_status": shared.SAFE_READY_AUTHORITY_STATUS,
            "safe_ready_formal_pass_claimed": False,
            "blockers": [],
            "formal_blockers": list(A211_FORMAL_BLOCKERS),
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

    def checkpoint_state(self) -> dict[str, Any]:
        if not self.is_reset_boundary():
            raise A211EnvError("A211 WAIT checkpoint requires a reset boundary")
        base_checkpoint = getattr(self.base, "checkpoint_state", None)
        payload = {
            "schema_version": 1,
            "kind": "action_ball_a211_mujoco_wait_boundary_state_v1",
            "identity": self.diagnostic_training_identity(),
            "base_wait_state": (
                base_checkpoint() if callable(base_checkpoint) else None
            ),
            "boundary_contract_sha256": self._canonical_boundary_sha256,
            "boundary_state_sha256": self._current_boundary_state_sha256,
        }
        payload["content_sha256"] = shared._sha256_json(payload)
        return payload

    def load_checkpoint_state(self, state: Any) -> None:
        if not self.is_reset_boundary() and self.base.is_reset_boundary() is not True:
            raise A211EnvError("A211 WAIT checkpoint load requires a reset boundary")
        expected_keys = {
            "schema_version",
            "kind",
            "identity",
            "base_wait_state",
            "boundary_contract_sha256",
            "boundary_state_sha256",
            "content_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise A211EnvError("A211 WAIT checkpoint schema differs")
        payload = dict(state)
        declared = payload.pop("content_sha256")
        if (
            state["schema_version"] != 1
            or state["kind"] != "action_ball_a211_mujoco_wait_boundary_state_v1"
            or state["identity"] != self.diagnostic_training_identity()
            or state["boundary_contract_sha256"]
            != self._canonical_boundary_sha256
            or declared != shared._sha256_json(payload)
        ):
            raise A211EnvError("A211 WAIT checkpoint seal differs")
        base_loader = getattr(self.base, "load_checkpoint_state", None)
        if state["base_wait_state"] is not None:
            if not callable(base_loader):
                raise A211EnvError("A211 base cannot restore WAIT continuation")
            base_loader(state["base_wait_state"])
        current_wait_steps = getattr(
            self.base, "current_wait_steps", self.producer._wait_steps_by_env
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
        self._reset_a211_progress_rows(range(self.num_envs))
        self._install_current_boundary(initial=False)
        if self._current_boundary_state_sha256 != state["boundary_state_sha256"]:
            self._actor = None
            self._critic = None
            raise A211EnvError("restored A211 WAIT boundary state differs")


__all__ = [
    "A211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS",
    "A211_ENV_KIND",
    "A211_FORMAL_BLOCKERS",
    "A211_REWARD_CONTRACT_IDENTITY",
    "A211_REWARD_SCOPE",
    "A211_TARGET_CHANNELS",
    "A211_TARGET_RECIPE",
    "A211_TASK_REWARD_CONTRACT_IDENTITY",
    "A211EnvError",
    "A211ObservationProducer",
    "A211TaskAuthority",
    "MeasuredA211MimicAuthority",
    "MujocoA211DiagnosticVecEnv",
    "a211_achieved_outcome_reward_terms",
    "a211_desired_contact_reward_terms",
    "a211_prestrike_footwork_reward_terms",
]
