from __future__ import annotations

import json
import math
import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    is_empirical_normalizer,
)
from whole_body_tracking.utils.training_contract import (
    CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY,
    CHECKPOINT_CONTRACT_SCHEMA_KEY,
    CHECKPOINT_CONTRACT_SHA_KEY,
    CHECKPOINT_LAUNCH_CLAIM_SHA_KEY,
    TRAINING_CONTRACT_SCHEMA_VERSION,
    validate_training_launch_claim_sha256,
)


_EXACT_BEHAVIOR_EVENT = "hope_exact_behavior_update"
_PLANNER_INITIAL_TTS_BUCKETS = (
    "lt_0p5",
    "eq_0p5",
    "gt_0p5_le_0p9",
    "gt_0p9",
)


def _ratio_or_none(counters: dict, numerator: str, denominator: str):
    """Return an honest derived value; an absent/zero denominator is unavailable, never zero."""

    denom = counters.get(denominator, 0)
    if denom is None or float(denom) <= 0.0:
        return None
    value = float(counters.get(numerator, 0)) / float(denom)
    return value if math.isfinite(value) else None


def exact_behavior_decision_values(counters: dict) -> dict[str, float | None]:
    """Derive dashboard values from one record; windows must sum counters before calling this."""

    values = {
        "swing_completion_rate": _ratio_or_none(
            counters, "swing_completion_count", "swing_outcome_count"
        ),
        "pre_strike_physical_fall_rate": _ratio_or_none(
            counters, "pre_strike_physical_fall_count", "swing_outcome_count"
        ),
        "post_strike_physical_fall_rate": _ratio_or_none(
            counters, "post_strike_physical_fall_count", "swing_outcome_count"
        ),
        "virtual_capture_per_strike": _ratio_or_none(
            counters, "virtual_capture_count", "strike_opportunity_count"
        ),
        "virtual_net_clear_per_capture": _ratio_or_none(
            counters, "virtual_net_clear_count", "virtual_capture_count"
        ),
        "virtual_landing_valid_per_capture": _ratio_or_none(
            counters, "virtual_landing_valid_count", "virtual_capture_count"
        ),
        "virtual_legal_return_per_strike": _ratio_or_none(
            counters, "virtual_legal_return_count", "strike_opportunity_count"
        ),
        "ready_tilt_rad_mean": _ratio_or_none(
            counters, "ready_tilt_rad_sum", "ready_tilt_eligible_sample_count"
        ),
        "ready_base_speed_xy_mps_mean": _ratio_or_none(
            counters,
            "ready_base_speed_xy_mps_sum",
            "ready_base_speed_eligible_sample_count",
        ),
        "ready_station_offset_m_mean": _ratio_or_none(
            counters,
            "ready_station_offset_m_sum",
            "ready_station_offset_eligible_sample_count",
        ),
        "ready_foot_contact_fraction_mean": _ratio_or_none(
            counters,
            "ready_foot_contact_fraction_sum",
            "ready_foot_contact_eligible_sample_count",
        ),
        "ready_foot_slip_speed_mps_mean": _ratio_or_none(
            counters,
            "ready_foot_slip_speed_mps_sum",
            "ready_foot_slip_eligible_sample_count",
        ),
    }
    for bucket_name in _PLANNER_INITIAL_TTS_BUCKETS:
        prefix = f"planner_initial_tts_{bucket_name}_"
        values[f"{prefix}swing_completion_rate"] = _ratio_or_none(
            counters,
            f"{prefix}swing_completion_count",
            f"{prefix}swing_outcome_count",
        )
        values[f"{prefix}virtual_capture_per_strike"] = _ratio_or_none(
            counters,
            f"{prefix}virtual_capture_count",
            f"{prefix}strike_opportunity_count",
        )
        values[f"{prefix}virtual_legal_return_per_strike"] = _ratio_or_none(
            counters,
            f"{prefix}virtual_legal_return_count",
            f"{prefix}strike_opportunity_count",
        )
    return values


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            trained_with_obs_norm = bool(self.empirical_normalization)
            normalizer = self.obs_normalizer if trained_with_obs_norm else None
            export_policy_as_onnx(
                self.alg.policy,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename,
                obs_norm_baked=is_empirical_normalizer(normalizer),
                trained_with_obs_norm=trained_with_obs_norm,
                source_checkpoint_path=path,
            )
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name=None,
        *,
        training_contract_schema_version: int | None = None,
        training_contract_sha256: str | None = None,
        training_contract_lineage_exact: bool = False,
        training_launch_claim_sha256: str | None = None,
    ):
        validated_launch_claim = None
        if training_launch_claim_sha256 is not None:
            validated_launch_claim = validate_training_launch_claim_sha256(
                training_launch_claim_sha256
            )
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.training_contract_schema_version = training_contract_schema_version
        self.training_contract_sha256 = training_contract_sha256
        self.training_contract_lineage_exact = bool(training_contract_lineage_exact)
        self.training_launch_claim_sha256 = validated_launch_claim
        if (training_contract_schema_version is None) != (training_contract_sha256 is None):
            raise ValueError("training contract schema and SHA256 must be supplied together")
        if training_contract_schema_version is not None:
            if int(training_contract_schema_version) != TRAINING_CONTRACT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported training contract schema {training_contract_schema_version}; "
                    f"expected {TRAINING_CONTRACT_SCHEMA_VERSION}"
                )
            digest = str(training_contract_sha256).strip().lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("training_contract_sha256 must be 64 lowercase hex characters")
            self.training_contract_sha256 = digest

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        if (
            self.training_contract_sha256 is not None
            or self.training_launch_claim_sha256 is not None
        ):
            if infos is None:
                infos = {}
            elif not isinstance(infos, dict):
                raise TypeError("runner checkpoint infos must be a dict for contract binding")
            else:
                infos = dict(infos)
        if self.training_contract_sha256 is not None:
            infos[CHECKPOINT_CONTRACT_SCHEMA_KEY] = int(
                self.training_contract_schema_version
            )
            infos[CHECKPOINT_CONTRACT_SHA_KEY] = self.training_contract_sha256
            infos[CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY] = (
                1 if self.training_contract_lineage_exact else 0
            )
        if self.training_launch_claim_sha256 is not None:
            infos[CHECKPOINT_LAUNCH_CLAIM_SHA_KEY] = self.training_launch_claim_sha256
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            trained_with_obs_norm = bool(self.empirical_normalization)
            obs_norm_baked = export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=self.obs_normalizer if trained_with_obs_norm else None,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename,
                obs_norm_baked=obs_norm_baked,
                trained_with_obs_norm=trained_with_obs_norm,
                source_checkpoint_path=path,
            )
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # Link the input motion artifact(s) to this run (lineage bookkeeping only — a W&B API
            # failure here must not kill the training run). W&B expects registry refs to include an
            # alias (for example, collection:latest).
            if self.registry_name is not None:
                registry_names = (
                    self.registry_name if isinstance(self.registry_name, (list, tuple)) else [self.registry_name]
                )
                for registry_name in registry_names:
                    try:
                        wandb.run.use_artifact(registry_name)
                    except Exception as e:
                        print(f"[MotionOnPolicyRunner] WARNING: use_artifact({registry_name!r}) failed: {e}")
                self.registry_name = None

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width=width, pad=pad)
        step = int(locs["it"])
        # Consume/print even when TensorBoard/W&B is disabled: this stdout JSON line is the exact
        # per-update receipt, while dashboard logging is optional presentation only.
        exact_behavior = self._consume_exact_behavior_updates(step)
        if self.disable_logs or self.writer is None:
            return
        self._log_live_metrics(step, exact_behavior=exact_behavior)

    def _consume_exact_behavior_updates(self, step: int) -> dict[str, dict]:
        """Consume the sole behavior ledger once and emit one canonical JSON line per PPO update."""

        if getattr(self, "_exact_behavior_consumed_step", None) == int(step):
            return getattr(self, "_exact_behavior_consumed_records", {})
        records: dict[str, dict] = {}
        env = self.env.unwrapped
        if hasattr(env, "command_manager"):
            providers = []
            for term_name in env.command_manager.active_terms:
                term = env.command_manager.get_term(term_name)
                consumer = getattr(term, "consume_exact_behavior_decision_counters", None)
                if callable(consumer):
                    providers.append((str(term_name), consumer))
            if len(providers) > 1:
                names = [name for name, _consumer in providers]
                raise RuntimeError(
                    "exact behavior receipt requires exactly one provider; found "
                    f"{names}"
                )
            for term_name, consumer in providers:
                counters = {
                    name: self._exact_counter_value(value)
                    for name, value in consumer().items()
                }
                record = {
                    "event": _EXACT_BEHAVIOR_EVENT,
                    "schema_version": 1,
                    "ppo_update": int(step),
                    "term": term_name,
                    "counters": dict(sorted(counters.items())),
                    "derived": exact_behavior_decision_values(counters),
                    "window_aggregation": "sum_counters_then_recompute_derived",
                }
                records[term_name] = record
                print(
                    "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
                    + json.dumps(record, sort_keys=True, separators=(",", ":")),
                    flush=True,
                )
        self._exact_behavior_consumed_step = int(step)
        self._exact_behavior_consumed_records = records
        return records

    def _log_live_metrics(
        self, step: int, *, exact_behavior: dict[str, dict] | None = None
    ) -> None:
        """Log current manager state means every PPO iteration for richer dashboards."""
        env = self.env.unwrapped
        if exact_behavior is None:
            exact_behavior = self._consume_exact_behavior_updates(step)

        if hasattr(env, "command_manager"):
            for term_name in env.command_manager.active_terms:
                term = env.command_manager.get_term(term_name)
                for metric_name, metric_value in term.metrics.items():
                    self._log_scalar(f"Live/{term_name}/{metric_name}", self._mean_tensor(metric_value), step)
                activation_consumer = getattr(
                    term, "consume_training_activation_counters", None
                )
                if not callable(activation_consumer):
                    # Backward-compatible fallback for command terms that expose only the first
                    # post-swing ledger API.  A term exposing both APIs is consumed exactly once
                    # through the aggregate transaction above.
                    activation_consumer = getattr(
                        term, "consume_post_swing_activation_counters", None
                    )
                if callable(activation_consumer):
                    # These are integer event/sample counts accumulated across every environment
                    # step in the just-finished PPO update.  They must be logged as totals, not
                    # averaged over num_envs like instantaneous CommandTerm metrics.  The
                    # aggregate consumer snapshots and resets all ledgers exactly once.
                    for counter_name, counter_value in activation_consumer().items():
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            self._scalar_tensor(counter_value),
                            step,
                        )
                exact_record = exact_behavior.get(str(term_name))
                if exact_record is not None:
                    for counter_name, counter_value in exact_record["counters"].items():
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            counter_value,
                            step,
                        )
                else:
                    sparse_reward_consumer = getattr(
                        term, "consume_sparse_reward_eligibility_counters", None
                    )
                if exact_record is None and callable(sparse_reward_consumer):
                    # Exact non-decayed counts, including per-action denominators.  These are
                    # intentionally a second transaction: MotionCommand owns imitation/replay
                    # activation, while RacketTargetCommand owns virtual strike outcomes.
                    for counter_name, counter_value in sparse_reward_consumer().items():
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            self._scalar_tensor(counter_value),
                            step,
                        )
                if hasattr(term, "command_counter"):
                    self._log_scalar(
                        f"Live/{term_name}/command_counter", self._mean_tensor(term.command_counter), step
                    )
        if hasattr(env, "reward_manager"):
            active_reward_terms = tuple(env.reward_manager.active_terms)
            self._log_scalar("Live/Reward/total", self._mean_tensor(getattr(env, "reward_buf", None)), step)
            for idx, term_name in enumerate(active_reward_terms):
                self._log_scalar(
                    f"Live/Reward/{term_name}", self._mean_tensor(env.reward_manager._step_reward[:, idx]), step
                )
            if "base_decel_activation_probe" in active_reward_terms:
                # The nonzero-weight, zero-valued probe is the common RewardManager-stage
                # observer for control and treatment.  Consume its weight-independent ledger
                # exactly once per PPO update; totals/raw sum must not be averaged over num_envs.
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_base_decel_activation_counters,
                )

                command_name = "racket_target"
                for counter_name, counter_value in (
                    consume_base_decel_activation_counters(env, command_name).items()
                ):
                    self._log_scalar(
                        f"Live/{command_name}/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "joint_velocity_limit_hinge_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_joint_velocity_limit_hinge_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_joint_velocity_limit_hinge_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/qdot/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "processed_qdes_slew_hinge_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_processed_qdes_slew_hinge_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_processed_qdes_slew_hinge_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/processed_qdes_slew/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if (
                "lower_body_pose_imitation_probe" in active_reward_terms
                or "lower_body_stability_bundle_probe" in active_reward_terms
            ):
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_lower_body_wave_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_lower_body_wave_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/lower_body_wave/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "post_swing_settle_debt_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_post_swing_settle_debt_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_post_swing_settle_debt_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/post_swing_settle_debt/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )

        if hasattr(env, "termination_manager"):
            tm = env.termination_manager
            self._log_scalar("Live/Termination/done_rate", self._mean_tensor(tm.dones), step)
            self._log_scalar("Live/Termination/terminated_rate", self._mean_tensor(tm.terminated), step)
            self._log_scalar("Live/Termination/timeout_rate", self._mean_tensor(tm.time_outs), step)
            for term_name in tm.active_terms:
                self._log_scalar(f"Live/Termination/{term_name}", self._mean_tensor(tm.get_term(term_name)), step)

        if hasattr(env, "action_manager"):
            action = getattr(env.action_manager, "action", None)
            prev_action = getattr(env.action_manager, "prev_action", None)
            if action is not None:
                action_abs = torch.abs(action)
                self._log_scalar("Live/Action/mean_abs", self._mean_tensor(action_abs), step)
                self._log_scalar("Live/Action/max_abs", self._mean_tensor(torch.max(action_abs, dim=-1).values), step)
            if action is not None and prev_action is not None:
                action_delta_abs = torch.abs(action - prev_action)
                self._log_scalar("Live/Action/delta_mean_abs", self._mean_tensor(action_delta_abs), step)
                self._log_scalar(
                    "Live/Action/delta_max_abs",
                    self._mean_tensor(torch.max(action_delta_abs, dim=-1).values),
                    step,
                )

        self._log_scalar("Live/Env/episode_length", self._mean_tensor(env.episode_length_buf), step)
        self._log_scalar("Live/Env/common_step_counter", float(getattr(env, "common_step_counter", 0)), step)

    @staticmethod
    def _mean_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.float().mean().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _scalar_tensor(value):
        """Convert one scalar event counter without accidentally averaging a vector."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("per-update activation counter must be a scalar tensor")
            return value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _exact_counter_value(value):
        """Preserve integer counters in JSON and reject vectors/non-finite sums."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("per-update exact behavior counter must be a scalar tensor")
            value = value.item()
        if isinstance(value, bool):
            raise TypeError("per-update exact behavior counter must not be boolean")
        if isinstance(value, int):
            return value
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("per-update exact behavior sum must be finite")
        return numeric

    def _log_scalar(self, tag: str, value, step: int) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.writer.add_scalar(tag, float(value), step)
