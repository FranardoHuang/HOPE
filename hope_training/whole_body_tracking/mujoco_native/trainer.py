"""Controlled, diagnostic-only PPO shell for the native MuJoCo trainer lane.

This module intentionally does not make ``MujocoN1DiagnosticVecEnv`` trainable.
It consumes a small VecEnv protocol so that reward/termination successors can
prove one finite PPO update and checkpoint continuity without importing
``rsl_rl``.  A runtime receipt must explicitly authorize that narrow operation
while retaining all formal/promotion/deployment prohibitions.

The shell supports reset-boundary continuation only.  It neither captures nor
restores an in-flight environment episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DIAGNOSTIC_TRAINER_RECEIPT_KIND = "a3_mujoco_controlled_diagnostic_ppo_readiness_v1"
DIAGNOSTIC_UPDATE_RECEIPT_KIND = "a3_mujoco_controlled_diagnostic_ppo_update_v3"
NORMALIZER_BINDING_KIND = "a3_mujoco_asymmetric_normalizer_binding_v1"
TERMINAL_ROW_TELEMETRY_CONTRACT_KIND = (
    "a3_mujoco_exact_terminal_row_telemetry_contract_v1"
)
TERMINAL_ROW_TELEMETRY_RECEIPT_KIND = (
    "a3_mujoco_exact_terminal_row_telemetry_receipt_v1"
)
NORMALIZER_UPDATE_RULE = "current_policy_rows_once_bootstrap_normalize_only_v1"
NORMALIZER_WAIT_OUTPUT_RULE = (
    "raw_mask_exact_zero_then_post_normalization_exact_zero_v1"
)
TIMEOUT_BOOTSTRAP_RULE = (
    "rsl_rl_equivalent_reward_plus_gamma_pre_step_value_on_time_out_v1"
)

_IDENTITY_FIELDS = (
    "contract_sha256",
    "observation_contract_sha256",
    "action_contract_sha256",
    "reward_contract_sha256",
)


class DiagnosticPPOError(RuntimeError):
    """Base class for controlled diagnostic trainer failures."""


class DiagnosticPPOBlocked(DiagnosticPPOError):
    """The runtime receipt does not authorize even a diagnostic PPO update."""


class DiagnosticPPOContractError(DiagnosticPPOError):
    """The trainer, VecEnv, or tensor ABI differs from the frozen contract."""


class ResetBoundaryRequired(DiagnosticPPOError):
    """Checkpointing was requested while an episode was in flight."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DiagnosticPPOError(
            "torch is required for the diagnostic PPO shell"
        ) from exc
    return torch


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DiagnosticPPOContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise DiagnosticPPOContractError(f"{name} must be a positive plain integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DiagnosticPPOContractError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DiagnosticPPOContractError(f"{name} must be finite and positive")
    return result


def terminal_row_telemetry_contract() -> dict[str, Any]:
    """Return the exact diagnostic terminal-row ABI consumed by the trainer."""

    payload = {
        "schema_version": 1,
        "kind": TERMINAL_ROW_TELEMETRY_CONTRACT_KIND,
        "source_extras": [
            "episode_done_reasons",
            "time_outs",
            "task_valid_transition",
            "wait_assignment_transition",
            "reward_terms",
            "diagnostic_event_ledgers",
            "diagnostic_exact_hard_terminations",
            "diagnostic_exact_hard_termination_reasons",
        ],
        "tick_semantics": {
            "episode_policy_tick_zero_based": (
                "DiagnosticEventLedger.first_exact_hard_termination.policy_tick_"
                "or_policy_ticks_minus_one"
            ),
            "episode_transition_tick_1based": (
                "DiagnosticEventLedger.policy_ticks_after_current_transition"
            ),
            "phase_transition_tick_1based": (
                "RESET_WAIT_uses_episode_tick_TASK_ACTIVE_subtracts_wait_ticks"
            ),
        },
        "phase_values": ["RESET_WAIT", "TASK_ACTIVE"],
        "timeout_reasons": {
            "action_ball_single_stroke_complete": "single_stroke_timeout",
            "time_out": "horizon_timeout",
        },
        "hard_timeout_axes_are_independent": True,
        "reward_or_termination_semantics_changed": False,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    payload["content_sha256"] = _canonical_json_sha256(payload)
    return payload


def _plain_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise DiagnosticPPOContractError(
            f"{name} must be a non-negative plain integer"
        )
    return value


def _plain_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DiagnosticPPOContractError(f"{name} must be a plain boolean")
    return value


def _exact_terminal_rows_from_step(
    *,
    extras: Mapping[str, Any],
    dones: Any,
    time_outs: Any,
    rollout_step_1based: int,
    num_envs: int,
) -> list[dict[str, Any]]:
    """Validate and canonicalize terminal facts before compact-reset data vanish."""

    torch = _require_torch()
    _positive_int(rollout_step_1based, "rollout_step_1based")
    _positive_int(num_envs, "num_envs")

    def rows(name: str) -> Sequence[Any]:
        value = extras.get(name)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != num_envs
        ):
            raise DiagnosticPPOContractError(
                f"terminal telemetry extras.{name} must have num_envs rows"
            )
        return value

    done_reasons = rows("episode_done_reasons")
    task_valid = rows("task_valid_transition")
    assignments = rows("wait_assignment_transition")
    reward_terms = rows("reward_terms")
    ledgers = rows("diagnostic_event_ledgers")
    hard_reasons = rows("diagnostic_exact_hard_termination_reasons")
    exact_hard = extras.get("diagnostic_exact_hard_terminations")
    if (
        not isinstance(exact_hard, torch.Tensor)
        or exact_hard.dtype != torch.bool
        or exact_hard.device.type != "cpu"
        or tuple(exact_hard.shape) != (num_envs,)
    ):
        raise DiagnosticPPOContractError(
            "terminal telemetry exact hard mask must be CPU bool [num_envs]"
        )

    result: list[dict[str, Any]] = []
    for env_index in range(num_envs):
        done = bool(dones[env_index].item())
        time_out = bool(time_outs[env_index].item())
        hard = bool(exact_hard[env_index].item())
        reason = done_reasons[env_index]
        hard_reason = hard_reasons[env_index]
        if not done:
            if time_out or hard or reason is not None or hard_reason is not None:
                raise DiagnosticPPOContractError(
                    "nonterminal telemetry row carries a terminal fact"
                )
            continue
        if not isinstance(reason, str) or not reason:
            raise DiagnosticPPOContractError(
                "terminal telemetry done row requires one exact reason"
            )
        if hard:
            if not isinstance(hard_reason, str) or not hard_reason:
                raise DiagnosticPPOContractError(
                    "exact hard terminal row requires one exact hard reason"
                )
        elif hard_reason is not None:
            raise DiagnosticPPOContractError(
                "non-hard terminal row cannot carry an exact hard reason"
            )
        single_stroke_timeout = bool(
            time_out and reason == "action_ball_single_stroke_complete"
        )
        horizon_timeout = bool(time_out and reason == "time_out")
        if time_out and not (single_stroke_timeout or horizon_timeout):
            raise DiagnosticPPOContractError(
                "timeout terminal row has an unsupported exact reason"
            )
        if not time_out and (not hard or reason != hard_reason):
            raise DiagnosticPPOContractError(
                "hard-only terminal row reason differs from exact hard reason"
            )

        assignment = assignments[env_index]
        if not isinstance(assignment, Mapping):
            raise DiagnosticPPOContractError(
                "terminal telemetry wait assignment must be a mapping"
            )
        wait_ticks = _positive_int(
            assignment.get("wait_ticks"), "terminal wait_ticks"
        )
        assignment_env = _plain_nonnegative_int(
            assignment.get("env_id"), "terminal assignment env_id"
        )
        reset_generation = _positive_int(
            assignment.get("reset_generation"),
            "terminal assignment reset_generation",
        )
        if assignment_env != env_index:
            raise DiagnosticPPOContractError(
                "terminal wait assignment env_id differs from row index"
            )

        ledger = ledgers[env_index]
        if not isinstance(ledger, Mapping):
            raise DiagnosticPPOContractError(
                "terminal telemetry diagnostic ledger must be a mapping"
            )
        episode_tick = _positive_int(
            ledger.get("policy_ticks"), "terminal ledger policy_ticks"
        )
        reward_row = reward_terms[env_index]
        if not isinstance(reward_row, Mapping):
            raise DiagnosticPPOContractError(
                "terminal telemetry reward row must be a mapping"
            )
        reward_tick = _positive_int(
            reward_row.get("sample_policy_tick_1based"),
            "terminal reward sample_policy_tick_1based",
        )
        reward_task_valid = _plain_bool(
            reward_row.get("task_valid"), "terminal reward task_valid"
        )
        if reward_tick != episode_tick:
            raise DiagnosticPPOContractError(
                "terminal reward tick differs from diagnostic ledger"
            )
        termination = ledger.get("termination")
        if not isinstance(termination, Mapping):
            raise DiagnosticPPOContractError(
                "terminal telemetry ledger termination is absent"
            )
        if (
            _plain_bool(
                termination.get("exact_hard_terminated"),
                "ledger exact_hard_terminated",
            )
            != hard
            or termination.get("exact_hard_reason") != hard_reason
        ):
            raise DiagnosticPPOContractError(
                "terminal telemetry ledger hard facts differ from batch masks"
            )
        first_hard = ledger.get("first_exact_hard_termination")
        if hard:
            if not isinstance(first_hard, Mapping):
                raise DiagnosticPPOContractError(
                    "hard terminal ledger omits its first exact event"
                )
            first_tick = _plain_nonnegative_int(
                first_hard.get("policy_tick"), "first exact hard policy_tick"
            )
            all_reasons = first_hard.get("all_reasons")
            if (
                first_tick != episode_tick - 1
                or first_hard.get("reason") != hard_reason
                or not isinstance(all_reasons, list)
                or not all_reasons
                or any(not isinstance(value, str) or not value for value in all_reasons)
                or hard_reason not in all_reasons
            ):
                raise DiagnosticPPOContractError(
                    "first exact hard event differs from terminal transition"
                )
            exact_hard_all_reasons = list(all_reasons)
            hard_sample_timing = first_hard.get("sample_timing")
            if hard_sample_timing not in ("post_control_step", "physics_substep"):
                raise DiagnosticPPOContractError(
                    "first exact hard event sample timing is unsupported"
                )
            hard_physics_substep = first_hard.get("physics_substep")
            if hard_sample_timing == "physics_substep":
                _plain_nonnegative_int(
                    hard_physics_substep, "first exact hard physics_substep"
                )
            elif hard_physics_substep is not None:
                raise DiagnosticPPOContractError(
                    "post-control hard event cannot carry a physics substep"
                )
        else:
            if first_hard is not None:
                raise DiagnosticPPOContractError(
                    "non-hard terminal ledger carries a first exact hard event"
                )
            exact_hard_all_reasons = []
            hard_sample_timing = None
            hard_physics_substep = None

        active = _plain_bool(
            task_valid[env_index], "terminal task_valid_transition"
        )
        if reward_task_valid != active:
            raise DiagnosticPPOContractError(
                "terminal reward phase differs from task_valid_transition"
            )
        if active:
            if episode_tick <= wait_ticks:
                raise DiagnosticPPOContractError(
                    "TASK_ACTIVE terminal tick does not follow its WAIT"
                )
            phase = "TASK_ACTIVE"
            phase_tick = episode_tick - wait_ticks
        else:
            if episode_tick > wait_ticks:
                raise DiagnosticPPOContractError(
                    "RESET_WAIT terminal tick exceeds its assigned WAIT"
                )
            phase = "RESET_WAIT"
            phase_tick = episode_tick

        termination_class = (
            "single_stroke_timeout"
            if single_stroke_timeout
            else "horizon_timeout"
            if horizon_timeout
            else "hard"
        )
        result.append(
            {
                "rollout_step_1based": rollout_step_1based,
                "env_index": env_index,
                "done_reason": reason,
                "termination_class": termination_class,
                "time_out": time_out,
                "single_stroke_timeout": single_stroke_timeout,
                "horizon_timeout": horizon_timeout,
                "exact_hard_termination": hard,
                "exact_hard_coincident_with_timeout": bool(hard and time_out),
                "exact_hard_reason": hard_reason,
                "exact_hard_all_reasons": exact_hard_all_reasons,
                "exact_hard_sample_timing": hard_sample_timing,
                "exact_hard_physics_substep": hard_physics_substep,
                "phase": phase,
                "task_valid_transition": active,
                "episode_policy_tick_zero_based": episode_tick - 1,
                "episode_transition_tick_1based": episode_tick,
                "phase_transition_tick_1based": phase_tick,
                "wait_ticks": wait_ticks,
                "reset_generation": reset_generation,
            }
        )
    if len(result) != int(torch.count_nonzero(dones).item()):
        raise DiagnosticPPOContractError(
            "terminal telemetry row count differs from done mask"
        )
    return result


def _terminal_row_telemetry_receipt(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_rows = [copy.deepcopy(dict(row)) for row in rows]

    def histogram(field: str, *, omit_none: bool = False) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in canonical_rows:
            value = row.get(field)
            if value is None and omit_none:
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return {key: counts[key] for key in sorted(counts)}

    composite_counts: dict[
        tuple[str, str, bool, bool, str, str, int, int], int
    ] = {}
    for row in canonical_rows:
        key = (
            str(row["done_reason"]),
            str(row["termination_class"]),
            bool(row["exact_hard_termination"]),
            bool(row["time_out"]),
            (
                ""
                if row["exact_hard_reason"] is None
                else str(row["exact_hard_reason"])
            ),
            str(row["phase"]),
            int(row["episode_transition_tick_1based"]),
            int(row["phase_transition_tick_1based"]),
        )
        composite_counts[key] = composite_counts.get(key, 0) + 1
    composite = [
        {
            "reason": key[0],
            "termination_class": key[1],
            "exact_hard_termination": key[2],
            "time_out": key[3],
            "exact_hard_reason": key[4] or None,
            "phase": key[5],
            "episode_transition_tick_1based": key[6],
            "phase_transition_tick_1based": key[7],
            "count": composite_counts[key],
        }
        for key in sorted(composite_counts)
    ]
    payload = {
        "schema_version": 1,
        "kind": TERMINAL_ROW_TELEMETRY_RECEIPT_KIND,
        "contract_sha256": terminal_row_telemetry_contract()["content_sha256"],
        "terminal_row_count": len(canonical_rows),
        "hard_only_row_count": sum(
            int(row["exact_hard_termination"] and not row["time_out"])
            for row in canonical_rows
        ),
        "exact_hard_termination_row_count": sum(
            int(row["exact_hard_termination"]) for row in canonical_rows
        ),
        "timeout_row_count": sum(int(row["time_out"]) for row in canonical_rows),
        "timeout_only_row_count": sum(
            int(row["time_out"] and not row["exact_hard_termination"])
            for row in canonical_rows
        ),
        "single_stroke_timeout_row_count": sum(
            int(row["single_stroke_timeout"]) for row in canonical_rows
        ),
        "horizon_timeout_row_count": sum(
            int(row["horizon_timeout"]) for row in canonical_rows
        ),
        "coincident_exact_hard_and_timeout_row_count": sum(
            int(row["exact_hard_coincident_with_timeout"])
            for row in canonical_rows
        ),
        "reason_histogram": histogram("done_reason"),
        "exact_hard_reason_histogram": histogram(
            "exact_hard_reason", omit_none=True
        ),
        "phase_histogram": histogram("phase"),
        "reason_phase_tick_histogram": composite,
        "rows": canonical_rows,
        "reward_or_termination_semantics_changed": False,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    payload["content_sha256"] = _canonical_json_sha256(payload)
    return payload


# 人话:和 Isaac 侧同名的两条初始化路线,字面量必须一模一样,否则跨引擎对不上。
#   zero_weight_ready_bias = 输出层清零 + bias 钉死物理 hold,初始策略是常数(老路)。
#   default                = 标准初始化,输出层保持框架默认,不做任何覆盖。
# Isaac source of truth: whole_body_tracking.utils.training_contract
# ACTION_BALL_ACTOR_INIT_MODE_*.  These literals are duplicated rather than imported because the
# native MuJoCo lane must stay importable without the Isaac package installed; the cross-engine
# test asserts the two spellings are identical.
ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS = "zero_weight_ready_bias"
ACTOR_INIT_MODE_DEFAULT = "default"
ACTOR_INIT_MODES = (
    ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
    ACTOR_INIT_MODE_DEFAULT,
)
FOUR_SIGMA_GATE_SKIPPED_REASON = (
    "default_initialized_actor_mean_is_not_the_constant_hold_qdes_so_the_"
    "four_sigma_envelope_geometry_does_not_apply"
)


def fresh_actor_bootstrap_contract(
    output_bias: Sequence[float],
    *,
    initial_action_std: float,
    actor_init_mode: str = ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
) -> dict[str, Any]:
    """Canonicalize the Isaac-equivalent fresh ActionBall actor bootstrap.

    ``actor_init_mode`` selects between the historical zero-weight hold bootstrap and the
    standard initialization.  The zero-weight payload is emitted verbatim -- same keys, same
    schema, same content SHA -- so every sealed native launch artifact stays reproducible.  The
    standard path emits its own schema/kind and self-declares that the 4-sigma envelope forecast
    does not apply to it, instead of quietly reusing the hold vocabulary.
    """

    if actor_init_mode not in ACTOR_INIT_MODES:
        raise DiagnosticPPOContractError(
            "fresh actor bootstrap actor_init_mode must be exactly one of "
            f"{list(ACTOR_INIT_MODES)}"
        )
    if isinstance(output_bias, (str, bytes)):
        raise DiagnosticPPOContractError("fresh actor output bias must be a sequence")
    try:
        values = tuple(float(value) for value in output_bias)
    except (TypeError, ValueError) as exc:
        raise DiagnosticPPOContractError(
            "fresh actor output bias must be a finite sequence"
        ) from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise DiagnosticPPOContractError(
            "fresh actor output bias must be a non-empty finite sequence"
        )
    if actor_init_mode == ACTOR_INIT_MODE_DEFAULT:
        std = float(initial_action_std)
        if not math.isfinite(std) or not 0.0 < std <= 1.0:
            raise DiagnosticPPOContractError(
                "standard actor initialization requires a finite "
                "initial_action_std in (0, 1]"
            )
        payload = {
            "schema_version": 1,
            "kind": "a3_action_ball_default_actor_init_bootstrap_v1",
            "actor_init_mode": ACTOR_INIT_MODE_DEFAULT,
            "fresh_only": True,
            "resume_overwrite_prohibited": True,
            "output_layer_weight": "default",
            "output_layer_bias": "default",
            "hold_reference_action": list(values),
            "initial_action_std": std,
            "noise_parameterization": "log_std",
            "four_sigma_hard_inner_gate": {
                "applied": False,
                "reason": FOUR_SIGMA_GATE_SKIPPED_REASON,
            },
        }
        payload["content_sha256"] = _canonical_json_sha256(payload)
        return payload
    if not math.isclose(
        float(initial_action_std), 0.02, rel_tol=0.0, abs_tol=0.0
    ):
        raise DiagnosticPPOContractError(
            "fresh actor bootstrap requires initial_action_std=0.02"
        )
    payload = {
        "schema_version": 1,
        "kind": "a3_action_ball_fresh_actor_hold_bootstrap_v1",
        "fresh_only": True,
        "resume_overwrite_prohibited": True,
        "output_layer_weight": "zeros",
        "output_layer_bias": list(values),
        "initial_action_std": float(initial_action_std),
        "noise_parameterization": "log_std",
    }
    payload["content_sha256"] = _canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True)
class TrainerIdentity:
    """Content identities that must match across VecEnv, update, and load."""

    contract_sha256: str
    observation_contract_sha256: str
    action_contract_sha256: str
    reward_contract_sha256: str

    def __post_init__(self) -> None:
        for field in _IDENTITY_FIELDS:
            _sha256(getattr(self, field), field)

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _IDENTITY_FIELDS}


@dataclass(frozen=True)
class DiagnosticPPOConfig:
    """Small fixed PPO recipe used only for a finite plumbing diagnostic."""

    observation_dim: int
    action_dim: int
    critic_observation_dim: int | None = None
    rollout_steps: int = 4
    rollout_reset_boundary_extension_steps: int = 0
    hidden_dims: tuple[int, ...] = (32, 32)
    seed: int = 0
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    initial_action_std: float = 0.2
    fresh_actor_output_bias: tuple[float, ...] = ()
    fresh_actor_bootstrap_authority_sha256: str | None = None
    actor_init_mode: str = ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
    normalizer_epsilon: float = 1.0e-5
    actor_normalizer_identity: str = "mujoco_diagnostic_actor_norm_v1"
    critic_normalizer_identity: str = "mujoco_diagnostic_critic_norm_v1"
    actor_task_mask_indices: tuple[int, ...] = ()
    critic_task_mask_indices: tuple[int, ...] = ()
    actor_task_valid_index: int | None = None
    critic_task_valid_index: int | None = None
    expected_profile_observation_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.observation_dim, "observation_dim")
        _positive_int(self.action_dim, "action_dim")
        if self.critic_observation_dim is not None:
            _positive_int(self.critic_observation_dim, "critic_observation_dim")
            if self.expected_profile_observation_contract_sha256 is None:
                raise DiagnosticPPOContractError(
                    "asymmetric critic requires an expected profile observation SHA"
                )
            _sha256(
                self.expected_profile_observation_contract_sha256,
                "expected_profile_observation_contract_sha256",
            )
        elif self.expected_profile_observation_contract_sha256 is not None:
            raise DiagnosticPPOContractError(
                "profile observation SHA requires an asymmetric critic"
            )
        _positive_int(self.rollout_steps, "rollout_steps")
        if (
            type(self.rollout_reset_boundary_extension_steps) is not int
            or self.rollout_reset_boundary_extension_steps < 0
        ):
            raise DiagnosticPPOContractError(
                "rollout_reset_boundary_extension_steps must be a non-negative "
                "plain integer"
            )
        if not self.hidden_dims or any(
            type(width) is not int or width < 1 for width in self.hidden_dims
        ):
            raise DiagnosticPPOContractError(
                "hidden_dims must be a non-empty tuple of positive integers"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise DiagnosticPPOContractError("seed must be a non-negative integer")
        _finite_positive(self.learning_rate, "learning_rate")
        _finite_positive(self.gamma, "gamma")
        _finite_positive(self.gae_lambda, "gae_lambda")
        _finite_positive(self.clip_param, "clip_param")
        _finite_positive(self.value_loss_coef, "value_loss_coef")
        if not math.isfinite(float(self.entropy_coef)) or self.entropy_coef < 0.0:
            raise DiagnosticPPOContractError(
                "entropy_coef must be finite and non-negative"
            )
        _finite_positive(self.max_grad_norm, "max_grad_norm")
        _finite_positive(self.initial_action_std, "initial_action_std")
        if self.actor_init_mode not in ACTOR_INIT_MODES:
            raise DiagnosticPPOContractError(
                f"actor_init_mode must be exactly one of {list(ACTOR_INIT_MODES)}"
            )
        if not isinstance(self.fresh_actor_output_bias, tuple) or any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in self.fresh_actor_output_bias
        ):
            raise DiagnosticPPOContractError(
                "fresh_actor_output_bias must be a finite tuple"
            )
        if (
            self.actor_init_mode == ACTOR_INIT_MODE_DEFAULT
            and not self.fresh_actor_output_bias
        ):
            # 人话:标准初始化也要求把物理 hold 写进合同 —— 它不再被装进网络,但收据必须留底,
            # 否则 VecEnv 那份权威和 trainer 这份就没有共同锚点了。
            raise DiagnosticPPOContractError(
                "standard actor initialization still requires the physical hold "
                "reference action so the VecEnv authority and the trainer share "
                "one anchor"
            )
        if (
            self.actor_init_mode == ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
            and self.fresh_actor_output_bias
            and (
                len(self.fresh_actor_output_bias) != self.action_dim
                or not math.isclose(
                    float(self.initial_action_std), 0.02, rel_tol=0.0, abs_tol=0.0
                )
            )
        ):
            raise DiagnosticPPOContractError(
                "fresh actor bootstrap requires one bias per action and "
                "initial_action_std=0.02"
            )
        if self.actor_init_mode == ACTOR_INIT_MODE_DEFAULT:
            if len(self.fresh_actor_output_bias) != self.action_dim:
                raise DiagnosticPPOContractError(
                    "standard actor initialization requires one hold reference "
                    "value per action"
                )
            if not 0.0 < float(self.initial_action_std) <= 1.0:
                raise DiagnosticPPOContractError(
                    "standard actor initialization requires a finite "
                    "initial_action_std in (0, 1]"
                )
        if self.fresh_actor_output_bias:
            _sha256(
                self.fresh_actor_bootstrap_authority_sha256,
                "fresh_actor_bootstrap_authority_sha256",
            )
        elif self.fresh_actor_bootstrap_authority_sha256 is not None:
            raise DiagnosticPPOContractError(
                "fresh actor bootstrap authority requires an output bias"
            )
        _finite_positive(self.normalizer_epsilon, "normalizer_epsilon")
        if self.gamma > 1.0 or self.gae_lambda > 1.0 or self.clip_param >= 1.0:
            raise DiagnosticPPOContractError(
                "gamma/gae_lambda must be <=1 and clip_param must be <1"
            )
        for name in ("actor_normalizer_identity", "critic_normalizer_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DiagnosticPPOContractError(f"{name} must be non-empty")
        self._validate_task_mask(
            lane="actor",
            width=self.observation_dim,
            indices=self.actor_task_mask_indices,
            valid_index=self.actor_task_valid_index,
        )
        self._validate_task_mask(
            lane="critic",
            width=self.effective_critic_observation_dim,
            indices=self.critic_task_mask_indices,
            valid_index=self.critic_task_valid_index,
        )
        actor_masked = bool(self.actor_task_mask_indices)
        critic_masked = bool(self.critic_task_mask_indices)
        if actor_masked != critic_masked:
            raise DiagnosticPPOContractError(
                "actor/critic task-valid masks must be configured together"
            )

    @staticmethod
    def _validate_task_mask(
        *,
        lane: str,
        width: int,
        indices: tuple[int, ...],
        valid_index: int | None,
    ) -> None:
        if not isinstance(indices, tuple) or any(
            type(value) is not int for value in indices
        ):
            raise DiagnosticPPOContractError(
                f"{lane}_task_mask_indices must be a tuple of integers"
            )
        if len(indices) != len(set(indices)) or tuple(sorted(indices)) != indices:
            raise DiagnosticPPOContractError(
                f"{lane}_task_mask_indices must be unique and ordered"
            )
        if any(value < 0 or value >= width for value in indices):
            raise DiagnosticPPOContractError(
                f"{lane}_task_mask_indices exceed lane width"
            )
        if indices:
            if type(valid_index) is not int or valid_index < 0 or valid_index >= width:
                raise DiagnosticPPOContractError(
                    f"{lane}_task_valid_index must identify one lane column"
                )
            if valid_index in indices:
                raise DiagnosticPPOContractError(
                    f"{lane}_task_valid_index cannot be part of the task value mask"
                )
        elif valid_index is not None:
            raise DiagnosticPPOContractError(
                f"{lane}_task_valid_index requires non-empty mask indices"
            )

    @property
    def effective_critic_observation_dim(self) -> int:
        return (
            self.observation_dim
            if self.critic_observation_dim is None
            else self.critic_observation_dim
        )

    @property
    def asymmetric_critic_required(self) -> bool:
        return self.critic_observation_dim is not None

    @property
    def content_sha256(self) -> str:
        # 人话:老配置(零权重那条路)的 SHA 一个字节都不能动,否则已有 checkpoint 全部作废。
        # 所以只有真正选了新路线时,这个自陈字段才进哈希 —— 它一进去 SHA 必然变,两条路的
        # checkpoint 也就再不可能互相冒充。
        payload = asdict(self)
        if payload.get("actor_init_mode") == ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS:
            payload.pop("actor_init_mode")
        return _canonical_json_sha256(payload)

    @property
    def normalizer_binding(self) -> dict[str, Any] | None:
        if not self.asymmetric_critic_required:
            return None
        return asymmetric_normalizer_binding(
            profile_observation_contract_sha256=(
                self.expected_profile_observation_contract_sha256
            ),
            actor_width=self.observation_dim,
            critic_width=self.effective_critic_observation_dim,
            actor_normalizer_identity=self.actor_normalizer_identity,
            critic_normalizer_identity=self.critic_normalizer_identity,
            actor_task_mask_indices=self.actor_task_mask_indices,
            critic_task_mask_indices=self.critic_task_mask_indices,
            actor_task_valid_index=self.actor_task_valid_index,
            critic_task_valid_index=self.critic_task_valid_index,
            epsilon=self.normalizer_epsilon,
        )

    @property
    def fresh_actor_bootstrap(self) -> dict[str, Any] | None:
        """Return the source-bound fresh-policy initialization contract.

        ActionBall's split-ready reset is held by an action-specific normalized
        command, not by normalized action zero.  Isaac therefore starts a fresh
        actor with a zero output weight matrix and that hold command as its
        output bias.  Native MuJoCo must do the same or the first stochastic
        action immediately abandons the physical birth before the task reveal.
        """

        if not self.fresh_actor_output_bias:
            return None
        payload = fresh_actor_bootstrap_contract(
            self.fresh_actor_output_bias,
            initial_action_std=self.initial_action_std,
            actor_init_mode=self.actor_init_mode,
        )
        payload["authority_content_sha256"] = (
            self.fresh_actor_bootstrap_authority_sha256
        )
        return payload


def asymmetric_normalizer_binding(
    *,
    profile_observation_contract_sha256: Any,
    actor_width: Any,
    critic_width: Any,
    actor_normalizer_identity: Any,
    critic_normalizer_identity: Any,
    actor_task_mask_indices: Any,
    critic_task_mask_indices: Any,
    actor_task_valid_index: Any,
    critic_task_valid_index: Any,
    epsilon: Any,
) -> dict[str, Any]:
    """Canonical profile-owned binding for asymmetric normalization and WAIT."""

    profile_sha = _sha256(
        profile_observation_contract_sha256,
        "profile_observation_contract_sha256",
    )
    actor_width = _positive_int(actor_width, "normalizer actor width")
    critic_width = _positive_int(critic_width, "normalizer critic width")
    epsilon = _finite_positive(epsilon, "normalizer epsilon")
    for name, value in (
        ("actor_normalizer_identity", actor_normalizer_identity),
        ("critic_normalizer_identity", critic_normalizer_identity),
    ):
        if not isinstance(value, str) or not value:
            raise DiagnosticPPOContractError(f"{name} must be non-empty")
    for lane, width, indices, valid_index in (
        (
            "actor",
            actor_width,
            actor_task_mask_indices,
            actor_task_valid_index,
        ),
        (
            "critic",
            critic_width,
            critic_task_mask_indices,
            critic_task_valid_index,
        ),
    ):
        if not isinstance(indices, tuple):
            raise DiagnosticPPOContractError(
                f"{lane} normalizer task-mask indices must be a tuple"
            )
        DiagnosticPPOConfig._validate_task_mask(
            lane=lane,
            width=width,
            indices=indices,
            valid_index=valid_index,
        )
    payload = {
        "schema_version": 1,
        "kind": NORMALIZER_BINDING_KIND,
        "profile_observation_contract_sha256": profile_sha,
        "actor": {
            "width": actor_width,
            "normalizer_identity": actor_normalizer_identity,
            "task_mask_indices": list(actor_task_mask_indices),
            "task_valid_index": actor_task_valid_index,
        },
        "critic": {
            "width": critic_width,
            "normalizer_identity": critic_normalizer_identity,
            "task_mask_indices": list(critic_task_mask_indices),
            "task_valid_index": critic_task_valid_index,
        },
        "epsilon": epsilon,
        "update_rule": NORMALIZER_UPDATE_RULE,
        "wait_output_rule": NORMALIZER_WAIT_OUTPUT_RULE,
    }
    payload["content_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_diagnostic_readiness_receipt(
    receipt: Mapping[str, Any], identity: TrainerIdentity
) -> dict[str, Any]:
    """Fail closed unless the receipt permits only controlled diagnostic PPO."""

    if not isinstance(receipt, Mapping):
        raise DiagnosticPPOBlocked("diagnostic PPO readiness receipt is absent")
    required = {
        "kind": DIAGNOSTIC_TRAINER_RECEIPT_KIND,
        "ppo_ready": True,
        "reward_available": True,
        "normal_step_available": True,
        "reset_boundary_checkpoint_available": True,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
        "mid_episode_resume": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise DiagnosticPPOBlocked(
                f"diagnostic PPO readiness receipt field {key!r} must equal {expected!r}"
            )
    expected_identity = identity.as_dict()
    for field, expected in expected_identity.items():
        try:
            actual = _sha256(receipt.get(field), f"receipt.{field}")
        except DiagnosticPPOContractError as exc:
            raise DiagnosticPPOBlocked(str(exc)) from exc
        if actual != expected:
            raise DiagnosticPPOBlocked(
                f"diagnostic PPO readiness receipt {field} differs from trainer identity"
            )
    blockers = receipt.get("blockers")
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        raise DiagnosticPPOBlocked(
            "diagnostic PPO readiness blockers must be a sequence"
        )
    if list(blockers):
        raise DiagnosticPPOBlocked(
            "diagnostic PPO is blocked: " + ",".join(str(item) for item in blockers)
        )
    return copy.deepcopy(dict(receipt))


class _RunningNormalizer:
    """Deterministic CPU running moments with an explicit state dictionary."""

    def __init__(self, width: int, epsilon: float) -> None:
        torch = _require_torch()
        self.width = width
        self.epsilon = float(epsilon)
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.m2 = torch.zeros(width, dtype=torch.float64)
        self.count = torch.zeros((), dtype=torch.float64)

    def update(self, observations: Any) -> None:
        torch = _require_torch()
        batch = observations.detach().to(dtype=torch.float64, device="cpu")
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        if float(self.count.item()) == 0.0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            self.count.fill_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean.add_(delta * (batch_count / total))
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.count.copy_(total)

    def normalize(self, observations: Any) -> Any:
        torch = _require_torch()
        denominator = torch.clamp(self.count, min=1.0)
        variance = self.m2 / denominator
        scale = torch.sqrt(torch.clamp(variance, min=self.epsilon))
        return (
            (observations.to(dtype=torch.float64, device="cpu") - self.mean)
            .div(scale)
            .to(dtype=torch.float32)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "epsilon": self.epsilon,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "count": self.count.clone(),
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        torch = _require_torch()
        if not isinstance(state, Mapping) or set(state) != {
            "width",
            "epsilon",
            "mean",
            "m2",
            "count",
        }:
            raise DiagnosticPPOContractError("normalizer checkpoint schema differs")
        if state["width"] != self.width or float(state["epsilon"]) != self.epsilon:
            raise DiagnosticPPOContractError("normalizer width/epsilon differs")
        if tuple(state["mean"].shape) != (self.width,) or tuple(state["m2"].shape) != (
            self.width,
        ):
            raise DiagnosticPPOContractError("normalizer moment shape differs")
        if tuple(state["count"].shape) != ():
            raise DiagnosticPPOContractError("normalizer count must be scalar")
        for name in ("mean", "m2", "count"):
            if (
                not isinstance(state[name], torch.Tensor)
                or not torch.isfinite(state[name]).all()
            ):
                raise DiagnosticPPOContractError(
                    f"normalizer checkpoint {name} is not a finite tensor"
                )
        if float(state["count"].item()) < 0.0 or bool(torch.any(state["m2"] < 0.0)):
            raise DiagnosticPPOContractError("normalizer moments are invalid")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        self.mean.copy_(state["mean"].to(dtype=self.mean.dtype, device="cpu"))
        self.m2.copy_(state["m2"].to(dtype=self.m2.dtype, device="cpu"))
        self.count.copy_(state["count"].to(dtype=self.count.dtype, device="cpu"))


def _mlp(widths: Sequence[int]) -> Any:
    torch = _require_torch()
    layers = []
    for index, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def _build_actor_critic(config: DiagnosticPPOConfig) -> Any:
    torch = _require_torch()
    module = torch.nn.Module()
    module.actor = _mlp(
        (config.observation_dim, *config.hidden_dims, config.action_dim)
    )
    module.critic = _mlp(
        (config.effective_critic_observation_dim, *config.hidden_dims, 1)
    )
    module.register_parameter(
        "log_std",
        torch.nn.Parameter(
            torch.full(
                (config.action_dim,),
                math.log(config.initial_action_std),
                dtype=torch.float32,
            )
        ),
    )
    if config.fresh_actor_output_bias:
        output = module.actor[-1]
        if (
            not isinstance(output, torch.nn.Linear)
            or output.bias is None
            or tuple(output.weight.shape)[0] != config.action_dim
        ):
            raise DiagnosticPPOContractError(
                "fresh actor bootstrap cannot identify the output Linear"
            )
        if config.actor_init_mode == ACTOR_INIT_MODE_DEFAULT:
            # 人话:标准初始化不动输出层;这里反过来断言它不是全零,免得别处偷偷清零后冒充。
            if int(torch.count_nonzero(output.weight).item()) == 0:
                raise DiagnosticPPOContractError(
                    "standard actor initialization found an exactly zero output "
                    "weight matrix, which is the zero-weight bootstrap this "
                    "config explicitly did not select"
                )
            return module
        expected_bias = torch.tensor(
            config.fresh_actor_output_bias,
            dtype=output.bias.dtype,
            device=output.bias.device,
        )
        with torch.no_grad():
            output.weight.zero_()
            output.bias.copy_(expected_bias)
        if int(torch.count_nonzero(output.weight).item()) != 0 or not torch.equal(
            output.bias, expected_bias
        ):
            raise DiagnosticPPOContractError(
                "fresh actor hold bootstrap did not install exactly"
            )
    return module


def _tensor_digest(state: Mapping[str, Any]) -> str:
    torch = _require_torch()
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class MujocoDiagnosticPPOTrainer:
    """One-update-at-a-time CPU PPO runner with a fail-closed VecEnv receipt."""

    def __init__(
        self,
        *,
        env: Any,
        identity: TrainerIdentity,
        config: DiagnosticPPOConfig,
    ) -> None:
        torch = _require_torch()
        self.env = env
        self.identity = identity
        self.config = config
        self.num_envs = _positive_int(getattr(env, "num_envs", None), "env.num_envs")
        for attribute, expected in (
            ("num_observations", config.observation_dim),
            ("num_actions", config.action_dim),
        ):
            actual = getattr(env, attribute, None)
            if actual != expected:
                raise DiagnosticPPOContractError(
                    f"env.{attribute}={actual!r} differs from configured {expected}"
                )
        if config.asymmetric_critic_required:
            actual = getattr(env, "num_privileged_observations", None)
            if actual is None:
                actual = getattr(env, "num_privileged_obs", None)
            if actual != config.effective_critic_observation_dim:
                raise DiagnosticPPOContractError(
                    "env privileged critic width differs from configured "
                    f"{config.effective_critic_observation_dim}"
                )
        if getattr(env, "device", "cpu") not in ("cpu", torch.device("cpu")):
            raise DiagnosticPPOContractError("diagnostic PPO shell is CPU-only")
        if not callable(getattr(env, "diagnostic_training_receipt", None)):
            raise DiagnosticPPOContractError(
                "VecEnv must expose diagnostic_training_receipt()"
            )
        if not callable(getattr(env, "is_reset_boundary", None)):
            raise DiagnosticPPOContractError("VecEnv must expose is_reset_boundary()")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = _build_actor_critic(config).to(device="cpu")
        self.fresh_actor_bootstrap = copy.deepcopy(config.fresh_actor_bootstrap)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.actor_normalizer = _RunningNormalizer(
            config.observation_dim, config.normalizer_epsilon
        )
        self.critic_normalizer = _RunningNormalizer(
            config.effective_critic_observation_dim, config.normalizer_epsilon
        )
        # Compatibility alias for read-only diagnostics that historically
        # inspected the sole actor/critic normalizer.  Checkpoint v2 never uses
        # this alias and always persists the two distinct objects.
        self.normalizer = self.actor_normalizer
        self.update_counter = 0
        self._actor_observations = None
        self._critic_observations = None
        self._last_update_receipt: dict[str, Any] | None = None

    def _validated_readiness(self) -> dict[str, Any]:
        try:
            receipt = self.env.diagnostic_training_receipt()
        except Exception as exc:  # noqa: BLE001 - external protocol boundary
            raise DiagnosticPPOBlocked(
                "VecEnv diagnostic readiness receipt could not be read"
            ) from exc
        validated = validate_diagnostic_readiness_receipt(receipt, self.identity)
        expected_binding = self.config.normalizer_binding
        if expected_binding is not None:
            actual_binding = validated.get("normalizer_binding")
            if not isinstance(actual_binding, Mapping):
                raise DiagnosticPPOBlocked(
                    "asymmetric VecEnv omits its profile-owned normalizer binding"
                )
            unsigned = dict(actual_binding)
            observed_sha = unsigned.pop("content_sha256", None)
            if (
                observed_sha != _canonical_json_sha256(unsigned)
                or dict(actual_binding) != expected_binding
            ):
                raise DiagnosticPPOBlocked(
                    "asymmetric VecEnv normalizer/profile/WAIT binding differs"
                )
        elif "normalizer_binding" in validated:
            raise DiagnosticPPOBlocked(
                "symmetric trainer cannot consume an asymmetric normalizer binding"
            )
        expected_bootstrap = self.config.fresh_actor_bootstrap
        actual_bootstrap = validated.get("fresh_actor_bootstrap")
        if expected_bootstrap is None:
            if actual_bootstrap is not None:
                raise DiagnosticPPOBlocked(
                    "VecEnv requires a fresh actor bootstrap absent from config"
                )
        else:
            if not isinstance(actual_bootstrap, Mapping):
                raise DiagnosticPPOBlocked(
                    "VecEnv fresh actor bootstrap authority is absent"
                )
            unsigned_bootstrap = dict(actual_bootstrap)
            actual_sha = unsigned_bootstrap.pop("content_sha256", None)
            if (
                actual_sha != _canonical_json_sha256(unsigned_bootstrap)
                or actual_sha != expected_bootstrap["authority_content_sha256"]
            ):
                raise DiagnosticPPOBlocked(
                    "fresh actor bootstrap authority SHA differs"
                )
            compared_keys = (
                "schema_version",
                "kind",
                "fresh_only",
                "resume_overwrite_prohibited",
                "output_layer_weight",
                "output_layer_bias",
                "initial_action_std",
                "noise_parameterization",
            )
            if self.config.actor_init_mode == ACTOR_INIT_MODE_DEFAULT:
                # 人话:新路线的合同键不一样,得按它自己的键比 —— 尤其是自陈的模式和"门跳过了"
                # 这两条,必须两边一致,否则一边零权重一边标准初始化就悄悄分叉了。
                compared_keys = (
                    "schema_version",
                    "kind",
                    "actor_init_mode",
                    "fresh_only",
                    "resume_overwrite_prohibited",
                    "output_layer_weight",
                    "output_layer_bias",
                    "hold_reference_action",
                    "initial_action_std",
                    "noise_parameterization",
                    "four_sigma_hard_inner_gate",
                )
            elif actual_bootstrap.get("actor_init_mode") is not None:
                raise DiagnosticPPOBlocked(
                    "VecEnv advertises a non-default actor initialization mode "
                    "the trainer config did not select"
                )
            for key in compared_keys:
                if actual_bootstrap.get(key) != expected_bootstrap.get(key):
                    raise DiagnosticPPOBlocked(
                        "fresh actor hold bootstrap differs from VecEnv authority"
                    )
        terminal_telemetry_available = validated.get(
            "terminal_row_telemetry_available", False
        )
        if type(terminal_telemetry_available) is not bool:
            raise DiagnosticPPOBlocked(
                "terminal_row_telemetry_available must be a plain boolean"
            )
        actual_terminal_contract = validated.get(
            "terminal_row_telemetry_contract"
        )
        if terminal_telemetry_available:
            if actual_terminal_contract != terminal_row_telemetry_contract():
                raise DiagnosticPPOBlocked(
                    "VecEnv exact terminal-row telemetry contract differs"
                )
        elif actual_terminal_contract is not None:
            raise DiagnosticPPOBlocked(
                "VecEnv cannot advertise a terminal telemetry contract while unavailable"
            )
        return validated

    def is_reset_boundary(self) -> bool:
        try:
            env_boundary = self.env.is_reset_boundary()
        except Exception as exc:  # noqa: BLE001 - external protocol boundary
            raise ResetBoundaryRequired(
                "VecEnv reset boundary could not be read"
            ) from exc
        if type(env_boundary) is not bool:
            raise ResetBoundaryRequired("VecEnv reset boundary must be a plain boolean")
        return (
            env_boundary
            and self._actor_observations is None
            and self._critic_observations is None
        )

    def assert_reset_boundary(self) -> None:
        if not self.is_reset_boundary():
            raise ResetBoundaryRequired(
                "only an explicit full reset boundary can be checkpointed; "
                "mid-episode resume is unsupported"
            )

    def _observations_tensor(self, value: Any, name: str, width: int) -> Any:
        torch = _require_torch()
        if not isinstance(value, torch.Tensor):
            raise DiagnosticPPOContractError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu" or tuple(value.shape) != (
            self.num_envs,
            width,
        ):
            raise DiagnosticPPOContractError(
                f"{name} must be finite CPU [{self.num_envs}, {width}]"
            )
        value = value.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise DiagnosticPPOContractError(f"{name} contains non-finite values")
        return value

    def _critic_from_extras(
        self, extras: Mapping[str, Any], actor: Any, *, name: str
    ) -> Any:
        observations = extras.get("observations")
        critic = (
            observations.get("critic") if isinstance(observations, Mapping) else None
        )
        if critic is None:
            if self.config.asymmetric_critic_required:
                raise DiagnosticPPOContractError(
                    f"{name} extras must expose observations.critic"
                )
            critic = actor
        return self._observations_tensor(
            critic,
            f"{name} critic observations",
            self.config.effective_critic_observation_dim,
        )

    def _observation_pair(
        self, actor_value: Any, extras: Mapping[str, Any], *, name: str
    ) -> tuple[Any, Any]:
        actor = self._observations_tensor(
            actor_value, f"{name} actor observations", self.config.observation_dim
        )
        critic = self._critic_from_extras(extras, actor, name=name)
        self._validate_task_validity(actor, critic, name=name)
        return actor, critic

    def _reset(self) -> tuple[Any, Any]:
        torch = _require_torch()
        reset_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        result = self.env.reset(seed=reset_seed)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[1], Mapping)
        ):
            raise DiagnosticPPOContractError(
                "VecEnv reset must return (observations, extras)"
            )
        return self._observation_pair(result[0], result[1], name="reset")

    @staticmethod
    def _mask_normalized_columns(
        normalized: Any, validity: Any | None, indices: tuple[int, ...]
    ) -> Any:
        if validity is None or not indices:
            return normalized
        torch = _require_torch()
        result = normalized.clone()
        invalid = ~validity
        if bool(invalid.any().item()):
            columns = torch.zeros(result.shape[1], dtype=torch.bool, device="cpu")
            columns[list(indices)] = True
            result[invalid.unsqueeze(1) & columns.unsqueeze(0)] = 0.0
        return result

    def _validate_task_validity(
        self, actor: Any, critic: Any, *, name: str
    ) -> Any | None:
        torch = _require_torch()
        if not self.config.actor_task_mask_indices:
            return None
        actor_values = actor[:, self.config.actor_task_valid_index]
        critic_values = critic[:, self.config.critic_task_valid_index]
        valid_values = torch.tensor((0.0, 1.0), dtype=torch.float32)
        if (
            not torch.isin(actor_values, valid_values).all()
            or not torch.isin(critic_values, valid_values).all()
        ):
            raise DiagnosticPPOContractError(
                f"{name} task_valid columns must contain only exact 0/1"
            )
        if not torch.equal(actor_values, critic_values):
            raise DiagnosticPPOContractError(
                f"{name} actor/critic task_valid columns differ"
            )
        validity = actor_values == 1.0
        invalid = ~validity
        if bool(invalid.any().item()):
            actor_task = actor[invalid][:, self.config.actor_task_mask_indices]
            critic_task = critic[invalid][:, self.config.critic_task_mask_indices]
            if bool(torch.any(actor_task != 0.0).item()) or bool(
                torch.any(critic_task != 0.0).item()
            ):
                raise DiagnosticPPOContractError(
                    f"{name} WAIT task/base/clocks columns must be exact zero"
                )
        return validity

    def _normalize_pair(
        self, actor: Any, critic: Any, *, update: bool
    ) -> tuple[Any, Any]:
        validity = self._validate_task_validity(actor, critic, name="normalizer input")
        if type(update) is not bool:
            raise DiagnosticPPOContractError("normalizer update flag must be bool")
        if update:
            self.actor_normalizer.update(actor)
            self.critic_normalizer.update(critic)
        normalized_actor = self.actor_normalizer.normalize(actor)
        normalized_critic = self.critic_normalizer.normalize(critic)
        normalized_actor = self._mask_normalized_columns(
            normalized_actor, validity, self.config.actor_task_mask_indices
        )
        normalized_critic = self._mask_normalized_columns(
            normalized_critic, validity, self.config.critic_task_mask_indices
        )
        return normalized_actor, normalized_critic

    def _normalized_pair(self, actor: Any, critic: Any) -> tuple[Any, Any]:
        """Update each current-policy row once, then normalize both lanes."""

        return self._normalize_pair(actor, critic, update=True)

    def _normalized_pair_without_update(
        self, actor: Any, critic: Any
    ) -> tuple[Any, Any]:
        """Normalize a bootstrap-only row without changing running moments."""

        return self._normalize_pair(actor, critic, update=False)

    def _distribution_terms(
        self,
        normalized_actor_observations: Any,
        normalized_critic_observations: Any,
        actions: Any | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        torch = _require_torch()
        means = self.model.actor(normalized_actor_observations)
        std = torch.exp(self.model.log_std).expand_as(means)
        if actions is None:
            actions = means + torch.randn_like(means) * std
        variance = std.square()
        log_prob = -0.5 * (
            ((actions - means).square() / variance)
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(std)).sum(dim=-1)
        values = self.model.critic(normalized_critic_observations).squeeze(-1)
        return actions, log_prob, entropy, values

    def run_update(self) -> dict[str, Any]:
        """Run one on-policy rollout and one full-batch PPO optimizer step.

        ``rollout_steps`` is the minimum batch length.  When the optional reset
        boundary extension is non-zero, collection continues under the same
        frozen policy until the first explicit all-environment reset boundary.
        Every extension transition remains in the update batch; no off-policy
        drain or discarded checkpoint-only tail is permitted.
        """

        torch = _require_torch()
        readiness = self._validated_readiness()
        if self._actor_observations is None:
            actor_observations, critic_observations = self._reset()
        else:
            if self._critic_observations is None:
                raise DiagnosticPPOContractError(
                    "cached actor/critic observation state is incomplete"
                )
            actor_observations = self._actor_observations
            critic_observations = self._critic_observations
        normalized_actor_rows = []
        normalized_critic_rows = []
        action_rows = []
        old_log_prob_rows = []
        raw_reward_rows = []
        reward_rows = []
        done_rows = []
        time_out_rows = []
        value_rows = []
        terminal_telemetry_available = bool(
            readiness.get("terminal_row_telemetry_available", False)
        )
        terminal_rows: list[dict[str, Any]] = []

        maximum_rollout_steps = (
            self.config.rollout_steps
            + self.config.rollout_reset_boundary_extension_steps
        )
        for _step in range(maximum_rollout_steps):
            normalized_actor, normalized_critic = self._normalized_pair(
                actor_observations, critic_observations
            )
            with torch.no_grad():
                actions, log_prob, _entropy, values = self._distribution_terms(
                    normalized_actor, normalized_critic
                )
            result = self.env.step(actions.detach().clone())
            if not isinstance(result, tuple) or len(result) != 4:
                raise DiagnosticPPOContractError(
                    "VecEnv step must return (observations, rewards, dones, extras)"
                )
            next_observations, rewards, dones, extras = result
            if not isinstance(extras, Mapping):
                raise DiagnosticPPOContractError("VecEnv step extras must be a mapping")
            next_actor, next_critic = self._observation_pair(
                next_observations, extras, name="step"
            )
            if (
                not isinstance(rewards, torch.Tensor)
                or rewards.device.type != "cpu"
                or tuple(rewards.shape) != (self.num_envs,)
                or not torch.isfinite(rewards).all()
            ):
                raise DiagnosticPPOContractError(
                    "VecEnv rewards must be finite CPU [num_envs]"
                )
            if (
                not isinstance(dones, torch.Tensor)
                or dones.dtype != torch.bool
                or dones.device.type != "cpu"
                or tuple(dones.shape) != (self.num_envs,)
            ):
                raise DiagnosticPPOContractError(
                    "VecEnv dones must be CPU bool [num_envs]"
                )
            time_outs = extras.get("time_outs")
            if (
                not isinstance(time_outs, torch.Tensor)
                or time_outs.dtype != torch.bool
                or time_outs.device.type != "cpu"
                or tuple(time_outs.shape) != (self.num_envs,)
            ):
                raise DiagnosticPPOContractError(
                    "VecEnv step extras.time_outs must be CPU bool [num_envs]"
                )
            if bool(torch.any(time_outs & ~dones).item()):
                raise DiagnosticPPOContractError(
                    "VecEnv time_outs must be an exact subset of dones"
                )
            if terminal_telemetry_available:
                terminal_rows.extend(
                    _exact_terminal_rows_from_step(
                        extras=extras,
                        dones=dones,
                        time_outs=time_outs,
                        rollout_step_1based=_step + 1,
                        num_envs=self.num_envs,
                    )
                )
            raw_rewards = rewards.to(dtype=torch.float32).detach()
            timeout_bootstrap = (
                self.config.gamma
                * values.detach()
                * time_outs.to(dtype=torch.float32)
            )
            ppo_rewards = raw_rewards + timeout_bootstrap
            if not torch.isfinite(ppo_rewards).all():
                raise DiagnosticPPOContractError(
                    "timeout-bootstrapped PPO rewards are non-finite"
                )
            normalized_actor_rows.append(normalized_actor)
            normalized_critic_rows.append(normalized_critic)
            action_rows.append(actions.detach())
            old_log_prob_rows.append(log_prob.detach())
            raw_reward_rows.append(raw_rewards)
            reward_rows.append(ppo_rewards)
            done_rows.append(dones.detach().clone())
            time_out_rows.append(time_outs.detach().clone())
            value_rows.append(values.detach())
            actor_observations = next_actor
            critic_observations = next_critic

            if (
                _step + 1 >= self.config.rollout_steps
                and self.config.rollout_reset_boundary_extension_steps > 0
                and bool(torch.all(dones).item())
            ):
                if self.env.is_reset_boundary() is not True:
                    raise DiagnosticPPOContractError(
                        "all done rows must expose an explicit VecEnv reset boundary"
                    )
                break

        actual_rollout_steps = len(reward_rows)
        if (
            self.config.rollout_reset_boundary_extension_steps > 0
            and not bool(torch.all(done_rows[-1]).item())
        ):
            raise ResetBoundaryRequired(
                "rollout boundary extension exhausted before an explicit "
                "all-environment reset boundary"
            )

        all_rows_done = bool(torch.all(done_rows[-1]).item())
        if all_rows_done:
            if self.env.is_reset_boundary() is not True:
                raise DiagnosticPPOContractError(
                    "all done rows must expose an explicit VecEnv reset boundary"
                )
            self._actor_observations = None
            self._critic_observations = None
            next_value = torch.zeros(self.num_envs, dtype=torch.float32)
        else:
            self._actor_observations = actor_observations.detach().clone()
            self._critic_observations = critic_observations.detach().clone()
            with torch.no_grad():
                (
                    _normalized_next_actor,
                    normalized_next_critic,
                ) = self._normalized_pair_without_update(
                    actor_observations, critic_observations
                )
                next_value = self.model.critic(normalized_next_critic).squeeze(-1)

        advantages = []
        gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for index in reversed(range(actual_rollout_steps)):
            nonterminal = (~done_rows[index]).to(dtype=torch.float32)
            delta = (
                reward_rows[index]
                + self.config.gamma * next_value * nonterminal
                - value_rows[index]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages.append(gae.clone())
            next_value = value_rows[index]
        advantages.reverse()

        flat_actor_observations = torch.cat(normalized_actor_rows, dim=0)
        flat_critic_observations = torch.cat(normalized_critic_rows, dim=0)
        flat_actions = torch.cat(action_rows, dim=0)
        flat_old_log_prob = torch.cat(old_log_prob_rows, dim=0)
        flat_advantages = torch.cat(advantages, dim=0)
        flat_returns = flat_advantages + torch.cat(value_rows, dim=0)
        advantage_mean = flat_advantages.mean()
        advantage_std = flat_advantages.std(unbiased=False)
        flat_advantages = (flat_advantages - advantage_mean) / torch.clamp(
            advantage_std, min=1.0e-8
        )

        _actions, new_log_prob, entropy, new_values = self._distribution_terms(
            flat_actor_observations, flat_critic_observations, flat_actions
        )
        ratio = torch.exp(new_log_prob - flat_old_log_prob)
        unclipped = ratio * flat_advantages
        clipped = (
            torch.clamp(
                ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
            )
            * flat_advantages
        )
        surrogate_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = (new_values - flat_returns).square().mean()
        entropy_mean = entropy.mean()
        loss = (
            surrogate_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy_mean
        )
        if not torch.isfinite(loss):
            raise DiagnosticPPOError("diagnostic PPO loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        if not torch.isfinite(pre_clip_grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise DiagnosticPPOError("diagnostic PPO gradient norm is non-finite")
        self.optimizer.step()
        if any(
            not torch.isfinite(parameter).all() for parameter in self.model.parameters()
        ):
            raise DiagnosticPPOError("diagnostic PPO produced non-finite parameters")

        self.update_counter += 1
        rollout_digest = hashlib.sha256()
        for tensor in (
            flat_actor_observations,
            flat_critic_observations,
            flat_actions,
            torch.cat(reward_rows, dim=0),
            torch.cat(done_rows, dim=0),
        ):
            rollout_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        terminal_telemetry = (
            _terminal_row_telemetry_receipt(terminal_rows)
            if terminal_telemetry_available
            else None
        )
        if terminal_telemetry is not None:
            rollout_digest.update(
                json.dumps(
                    terminal_telemetry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
        timeout_row_count = int(
            torch.count_nonzero(torch.cat(time_out_rows, dim=0)).item()
        )
        hard_terminal_row_count = int(
            torch.count_nonzero(
                torch.cat(done_rows, dim=0)
                & ~torch.cat(time_out_rows, dim=0)
            ).item()
        )
        if terminal_telemetry is not None and (
            terminal_telemetry["timeout_row_count"] != timeout_row_count
            or terminal_telemetry["hard_only_row_count"]
            != hard_terminal_row_count
        ):
            raise DiagnosticPPOContractError(
                "terminal telemetry aggregate differs from PPO done/timeout masks"
            )
        receipt = {
            "schema_version": 3,
            "kind": DIAGNOSTIC_UPDATE_RECEIPT_KIND,
            "status": "CONTROLLED_DIAGNOSTIC_PPO_UPDATE_COMPLETE",
            "update_counter": self.update_counter,
            "num_envs": self.num_envs,
            "rollout_steps": actual_rollout_steps,
            "minimum_rollout_steps": self.config.rollout_steps,
            "maximum_rollout_steps": maximum_rollout_steps,
            "reset_boundary_extension_steps_used": (
                actual_rollout_steps - self.config.rollout_steps
            ),
            "batch_size": self.num_envs * actual_rollout_steps,
            "timeout_bootstrap_rule": TIMEOUT_BOOTSTRAP_RULE,
            "timeout_row_count": timeout_row_count,
            "hard_terminal_row_count": hard_terminal_row_count,
            "hard_terminal_row_count_semantics": (
                "done_and_not_timeout_legacy_hard_only"
            ),
            "terminal_row_telemetry_available": terminal_telemetry_available,
            "terminal_row_telemetry": terminal_telemetry,
            "raw_reward_sum": float(
                torch.cat(raw_reward_rows, dim=0).sum().item()
            ),
            "timeout_bootstrap_reward_sum": float(
                (
                    torch.cat(reward_rows, dim=0)
                    - torch.cat(raw_reward_rows, dim=0)
                ).sum().item()
            ),
            "ppo_reward_sum": float(torch.cat(reward_rows, dim=0).sum().item()),
            "actor_observation_dim": self.config.observation_dim,
            "critic_observation_dim": self.config.effective_critic_observation_dim,
            **self.identity.as_dict(),
            "config_sha256": self.config.content_sha256,
            "fresh_actor_bootstrap": copy.deepcopy(
                readiness.get("fresh_actor_bootstrap")
            ),
            "readiness_receipt_sha256": _canonical_json_sha256(readiness),
            "loss": float(loss.detach().item()),
            "surrogate_loss": float(surrogate_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy": float(entropy_mean.detach().item()),
            "pre_clip_grad_norm": float(pre_clip_grad_norm.detach().item()),
            "rollout_sha256": rollout_digest.hexdigest(),
            "model_state_sha256": _tensor_digest(self.model.state_dict()),
            "actor_normalizer_identity": self.config.actor_normalizer_identity,
            "critic_normalizer_identity": self.config.critic_normalizer_identity,
            "actor_normalizer_state_sha256": _tensor_digest(
                self.actor_normalizer.state_dict()
            ),
            "critic_normalizer_state_sha256": _tensor_digest(
                self.critic_normalizer.state_dict()
            ),
            "at_reset_boundary": self.is_reset_boundary(),
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
        receipt["content_sha256"] = _canonical_json_sha256(receipt)
        self._last_update_receipt = copy.deepcopy(receipt)
        return receipt

    def checkpoint_state(self) -> dict[str, Any]:
        """Return complete trainer state, only at an explicit reset boundary."""

        self._validated_readiness()
        self.assert_reset_boundary()
        torch = _require_torch()
        environment_state = None
        checkpoint_state = getattr(self.env, "checkpoint_state", None)
        if callable(checkpoint_state):
            environment_state = copy.deepcopy(checkpoint_state())
        return {
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "actor_normalizer_state_dict": self.actor_normalizer.state_dict(),
            "critic_normalizer_state_dict": self.critic_normalizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().clone(),
            },
            "update_counter": self.update_counter,
            "last_update_receipt": copy.deepcopy(self._last_update_receipt),
            "environment_state": environment_state,
        }


__all__ = [
    "ACTOR_INIT_MODES",
    "ACTOR_INIT_MODE_DEFAULT",
    "ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS",
    "FOUR_SIGMA_GATE_SKIPPED_REASON",
    "DIAGNOSTIC_TRAINER_RECEIPT_KIND",
    "DIAGNOSTIC_UPDATE_RECEIPT_KIND",
    "NORMALIZER_BINDING_KIND",
    "NORMALIZER_UPDATE_RULE",
    "NORMALIZER_WAIT_OUTPUT_RULE",
    "TERMINAL_ROW_TELEMETRY_CONTRACT_KIND",
    "TERMINAL_ROW_TELEMETRY_RECEIPT_KIND",
    "TIMEOUT_BOOTSTRAP_RULE",
    "DiagnosticPPOBlocked",
    "DiagnosticPPOConfig",
    "DiagnosticPPOContractError",
    "DiagnosticPPOError",
    "MujocoDiagnosticPPOTrainer",
    "ResetBoundaryRequired",
    "TrainerIdentity",
    "asymmetric_normalizer_binding",
    "fresh_actor_bootstrap_contract",
    "terminal_row_telemetry_contract",
    "validate_diagnostic_readiness_receipt",
]
