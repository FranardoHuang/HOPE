from __future__ import annotations

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
        if self.disable_logs or self.writer is None:
            return
        self._log_live_metrics(locs["it"])

    def _log_live_metrics(self, step: int) -> None:
        """Log current manager state means every PPO iteration for richer dashboards."""
        env = self.env.unwrapped

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
                if hasattr(term, "command_counter"):
                    self._log_scalar(
                        f"Live/{term_name}/command_counter", self._mean_tensor(term.command_counter), step
                    )
                if bool(
                    getattr(
                        getattr(term, "cfg", None),
                        "base_decel_activation_enabled",
                        False,
                    )
                ):
                    # The observer stores its weight-independent ledger on racket_target.  Consume
                    # it once at the same PPO-update boundary as the motion activation ledgers;
                    # totals/raw sum must not be averaged over num_envs.
                    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                        consume_base_decel_activation_counters,
                    )

                    for counter_name, counter_value in (
                        consume_base_decel_activation_counters(env, term_name).items()
                    ):
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            self._scalar_tensor(counter_value),
                            step,
                        )

        if hasattr(env, "reward_manager"):
            self._log_scalar("Live/Reward/total", self._mean_tensor(getattr(env, "reward_buf", None)), step)
            for idx, term_name in enumerate(env.reward_manager.active_terms):
                self._log_scalar(
                    f"Live/Reward/{term_name}", self._mean_tensor(env.reward_manager._step_reward[:, idx]), step
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

    def _log_scalar(self, tag: str, value, step: int) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.writer.add_scalar(tag, float(value), step)
