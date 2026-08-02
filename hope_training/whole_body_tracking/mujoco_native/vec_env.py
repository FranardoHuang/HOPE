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

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import n1_ball_core
from . import single_env


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


@dataclass(frozen=True)
class DiagnosticBatchStep:
    observations: Any
    per_env_events: tuple[tuple[Mapping[str, Any], ...], ...]
    time_outs: Any


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
            "kind": "a3_mujoco_n1_diagnostic_vecenv_v1",
            "num_envs": self.num_envs,
            "observation_width": OBSERVATION_WIDTH,
            "reward_available": False,
            "diagnostic_unauthorized": True,
        }
        self.unwrapped = self
        self.step_dt = float(self.cores[0].binding.policy_step_dt_s)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._observations = torch.empty(
            (self.num_envs, OBSERVATION_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        self._has_reset = False
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
        }

    def diagnostic_step(self, actions: Any) -> DiagnosticBatchStep:
        """Advance physics without manufacturing a reward tensor."""

        torch = _require_torch()
        if not self._has_reset:
            raise VecEnvContractError("VecEnv must be reset before diagnostic_step")
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
        for core, action in zip(self.cores, actions.detach().cpu().numpy()):
            result = core.step(action)
            rows.append(result["observation_groups"])
            events.append(tuple(dict(value) for value in result["new_events"]))
        self.episode_length_buf += 1
        self._observations = self._tensor_observations(rows)
        time_outs = self.episode_length_buf >= self.max_episode_length
        return DiagnosticBatchStep(
            observations=self._observations.clone(),
            per_env_events=tuple(events),
            time_outs=time_outs.clone(),
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
        for action in actions:
            step = self.diagnostic_step(action)
            traces.append(step.observations.detach().cpu().numpy().copy())
            event_rows.append(
                [[dict(value) for value in env_events] for env_events in step.per_env_events]
            )
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
        receipt = {
            "schema_version": 1,
            "kind": "a3_mujoco_n1_diagnostic_vecenv_rollout_v1",
            "status": "DIAGNOSTIC_NO_REWARD_ROLLOUT_COMPLETE",
            "num_envs": self.num_envs,
            "steps": int(actions.shape[0]),
            "observation_shape": list(trace.shape),
            "event_transcript": event_rows,
            "trace_and_event_sha256": digest.hexdigest(),
            "reward_blocker": reward_blocker_receipt(),
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
    "MujocoN1DiagnosticVecEnv",
    "OBSERVATION_LAYOUT",
    "OBSERVATION_WIDTH",
    "REWARD_BLOCKERS",
    "RewardContractMissing",
    "VecEnvContractError",
    "flatten_observation_groups",
    "reward_blocker_receipt",
]
