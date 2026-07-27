from __future__ import annotations

import io
import json
import math
import os
import pathlib
import random
import signal

import numpy as np
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
# Action families the command term books per-side outcome counters for (hope_commands _clip_names).
_STRIKE_FAMILIES = ("forehand", "backhand")
# Cumulative per-family strike opportunities after which zero legal returns stops being a warm-up
# artefact: a whole side that never once returns is a broken COMMAND (e.g. a target box under the
# table surface), not a policy that needs more samples.  Four HOPE runs sat at exactly 0.0000 for
# thousands of iterations behind a healthy aggregate before anyone looked.
_ZERO_RETURN_ALARM_OPPORTUNITIES = 500
_ZERO_RETURN_ABORT_OPPORTUNITIES = 5000
_EXACT_RESUME_SCHEMA_VERSION = 3
_ENVIRONMENT_RESUME_SCHEMA_VERSION = 3
_SUPPORTED_EXACT_RESUME_SCHEMAS = (1, 2, _EXACT_RESUME_SCHEMA_VERSION)


def _ratio_or_none(counters: dict, numerator: str, denominator: str):
    """Return an honest derived value; an absent/zero denominator is unavailable, never zero."""

    denom = counters.get(denominator, 0)
    if denom is None or float(denom) <= 0.0:
        return None
    value = float(counters.get(numerator, 0)) / float(denom)
    return value if math.isfinite(value) else None


def _scaled_ratio_or_none(counters: dict, numerator: str, denominator: str, scale: float):
    """``_ratio_or_none`` times a unit conversion (the ledger is integer-only: um -> mm)."""

    value = _ratio_or_none(counters, numerator, denominator)
    return None if value is None else value * float(scale)


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
    # CONTINUOUS question production. A bank arm never draws, so `continuous_question_draw_count`
    # is absent/zero there and _ratio_or_none makes every one of these read None — never a lying
    # 0.0 that would look like a healthy continuous run.
    values["continuous_question_exhausted_rate"] = _ratio_or_none(
        counters, "continuous_question_exhausted_count", "continuous_question_draw_count"
    )
    values["continuous_question_admit_rate"] = _ratio_or_none(
        counters, "continuous_question_admitted_count", "continuous_question_draw_count"
    )
    values["continuous_question_resid_mm_mean"] = _scaled_ratio_or_none(
        counters, "continuous_question_resid_um_sum", "continuous_question_admitted_count", 1e-3
    )
    values["continuous_question_closed_loop_mm_mean"] = _scaled_ratio_or_none(
        counters,
        "continuous_question_closed_loop_um_sum",
        "continuous_question_closed_loop_row_count",
        1e-3,
    )
    values["continuous_question_redraw_rounds_mean"] = _ratio_or_none(
        counters, "continuous_question_redraw_round_sum", "continuous_question_refill_count"
    )
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
    for family in _STRIKE_FAMILIES:
        values[f"virtual_capture_per_strike_{family}"] = _ratio_or_none(
            counters,
            f"virtual_capture_count_{family}",
            f"strike_opportunity_count_{family}",
        )
        values[f"virtual_legal_return_per_strike_{family}"] = _ratio_or_none(
            counters,
            f"virtual_legal_return_count_{family}",
            f"strike_opportunity_count_{family}",
        )
    return values


def zero_return_alarm_levels(cumulative: dict) -> dict[str, str]:
    """Families whose cumulative strike opportunities have produced exactly zero legal returns.

    Returns {family: "alarm" | "abort"}; families that returned at least once, or that have not yet
    accumulated enough opportunities, are absent.  This is the ONLY reading that separates "this side
    is never eligible / never satisfiable" from "this side sometimes fails" — the aggregate rate
    averages a dead side against a healthy one into a plausible-looking number.
    """

    levels: dict[str, str] = {}
    for family in _STRIKE_FAMILIES:
        opportunities = cumulative.get(f"strike_opportunity_count_{family}", 0) or 0
        returns = cumulative.get(f"virtual_legal_return_count_{family}", 0) or 0
        if float(returns) > 0.0:
            continue
        if float(opportunities) >= _ZERO_RETURN_ABORT_OPPORTUNITIES:
            levels[family] = "abort"
        elif float(opportunities) >= _ZERO_RETURN_ALARM_OPPORTUNITIES:
            levels[family] = "alarm"
    return levels


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
        require_exact_resume_state: bool = False,
    ):
        if type(require_exact_resume_state) is not bool:
            raise TypeError("require_exact_resume_state must be a bool")
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
        # This is a construction-time run contract, not a permissive load flag. Task-first
        # training passes True; evaluation and legacy/warm-start runs retain the historical False.
        self.require_exact_resume_state = require_exact_resume_state
        if require_exact_resume_state and self.log_dir is None:
            raise ValueError(
                "require_exact_resume_state is a training-resume contract and requires log_dir"
            )
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
        if require_exact_resume_state and (
            self.training_contract_sha256 is None
            or not self.training_contract_lineage_exact
        ):
            raise ValueError(
                "require_exact_resume_state requires an exact-bound training contract"
            )
        # Task-first checkpoints promise complete, strict command-state restoration.  Reject a
        # launch before its first rollout if even one active command term still relies on the
        # legacy heuristic attribute scanner.
        self._validate_task_first_exact_resume_terms()

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        if infos is None:
            infos = {}
        elif not isinstance(infos, dict):
            raise TypeError(
                "runner checkpoint infos must be a dict (contract binding and exact-resume "
                "state are embedded there)"
            )
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
        # --- 精确续训状态(jiayi hitterobs 9f684ae5 按 main 语义移植)---------------------------
        # 人话:环境的 common_step_counter 驱动所有"随步数渐进"的课程(扰动 ramp、自适应
        # sigma、成功门控扩幅…),但 base rsl_rl 的存档只有权重/优化器/迭代号 —— 不把它和各
        # 命令项的课程状态一起存进 PT,续训时全部课程静默回到第 0 步,对 2 万+ iter 的长训
        # 是真炸弹。main 的既有惯例是把 checkpoint 元数据放进 infos(合同绑定键就在里面),
        # 所以续训状态也走 infos,而不是像 hitterobs 那样整体复刻 base 的 save 再塞顶层键:
        # 复刻会随 rsl_rl 升级悄悄漂移,而且 base save 写完盘才排队上传 W&B,走 infos 让云端
        # 副本从第一份字节起就带状态。键名沿用 jiayi 的 hope_exact_resume_state 便于跨栈对账。
        infos["hope_exact_resume_state"] = self._build_exact_resume_state()
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

    # ------------------------------------------------------------------
    # 精确续训包(jiayi hitterobs 9f684ae5 按 main 语义移植)
    # ------------------------------------------------------------------

    # HER 已实现回放环(dict[clip_key -> tensor],RacketTargetCommand._ach_*):main 相对
    # hitterobs 的新增课程宿主,按名字点名整环入档。
    _RESUME_TENSOR_DICT_ATTRS = ("_ach_pos", "_ach_vel", "_ach_spd")
    # 回放环的填充度/写指针(dict[clip_key -> int]):同样点名,不落在下面的后缀规则里。
    _RESUME_SCALAR_DICT_ATTRS = ("_ach_fill", "_ach_ptr")

    def _build_exact_resume_state(self) -> dict:
        """打包"精确续训"所需的训练进度状态(jiayi hitterobs 布局的 main 版)。

        人话:除了权重,续训还需要 (1) 下一个该跑的迭代号 —— base rsl_rl 完成第 N 个迭代后
        存档且记 iter=N,直接续会静默重复一次 PPO 更新;(2) 环境课程主时钟 common_step_counter
        和各命令项的课程状态;(3) 下一批采样所需的 RNG 状态。只要 checkpoint 带这份精确
        续训包,load() 就会在第一次 env.reset() 前恢复 RNG;不带包的 legacy checkpoint
        仍保留旧的 warm-start 语义,绝不根据 seed 或迭代号猜 RNG。
        """
        wandb_run_id = None
        wandb_run_name = None
        if self.logger_type == "wandb" and not self.disable_logs:
            try:
                import wandb

                if wandb.run is not None:
                    wandb_run_id = wandb.run.id
                    wandb_run_name = wandb.run.name
            except Exception:
                pass
        return {
            "schema_version": _EXACT_RESUME_SCHEMA_VERSION,
            # Base rsl_rl saves after completing iteration N but stores iter=N. Exact resume must
            # begin at N+1, otherwise it silently performs one duplicate PPO update.
            "next_learning_iteration": int(self.current_learning_iteration) + 1,
            "target_learning_iterations": int(self.cfg.get("max_iterations", 0)),
            "tot_timesteps": int(self.tot_timesteps),
            "tot_time": float(self.tot_time),
            "algorithm_learning_rate": float(self.alg.learning_rate),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "torch_cuda_device_count": int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0,
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
            "wandb_run_id": wandb_run_id,
            "wandb_run_name": wandb_run_name,
            "environment_resume_state": self._capture_environment_resume_state(),
            **dict(getattr(self, "checkpoint_resume_context", {})),
        }

    @staticmethod
    def _is_scalar_tree(value) -> bool:
        """Return whether a value is cheap, device-independent command-manager state."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return True
        if isinstance(value, (list, tuple)):
            return all(MotionOnPolicyRunner._is_scalar_tree(item) for item in value)
        if isinstance(value, dict):
            return all(
                MotionOnPolicyRunner._is_scalar_tree(key)
                and MotionOnPolicyRunner._is_scalar_tree(item)
                for key, item in value.items()
            )
        return False

    def _capture_environment_resume_state(self) -> dict:
        """Capture schedule/curriculum state that affects the next rollout distribution."""
        env = getattr(self.env, "unwrapped", self.env)
        state = {
            # 内层 schema:1 = jiayi/hitterobs 布局(scalars/tensors 两段);2 = main 追加
            # tensor_dicts 段;3 = command term 可用成对显式 hook 接管完整状态。旧 schema
            # 仍走属性扫描兼容,新 schema 的显式状态则按 term 名、term 类型和 strict=True
            # fail-loud 恢复,不允许换动作目录后静默套用旧课程。
            "schema_version": _ENVIRONMENT_RESUME_SCHEMA_VERSION,
            "common_step_counter": int(getattr(env, "common_step_counter", 0)),
            "active_term_names": [],
            "command_terms": {},
        }
        # Re-check the formal task-first contract at every save.  This catches runtime drift (for
        # example, a manager or term disappearing after construction) instead of emitting a
        # deceptively complete schema-3 checkpoint.
        self._validate_task_first_exact_resume_terms()
        manager = getattr(env, "command_manager", None)
        if manager is None:
            return state

        raw_term_names = tuple(getattr(manager, "active_terms", ()))
        term_names = tuple(str(name) for name in raw_term_names)
        if len(term_names) != len(set(term_names)):
            raise RuntimeError("command manager active_terms contains duplicate names")
        state["active_term_names"] = list(term_names)
        task_first_exact = self._task_first_exact_resume_required()

        for raw_term_name, term_name in zip(raw_term_names, term_names):
            term = manager.get_term(raw_term_name)
            exact_state_getter = getattr(term, "exact_resume_state_dict", None)
            exact_state_loader = getattr(term, "load_exact_resume_state_dict", None)
            has_exact_state_getter = callable(exact_state_getter)
            has_exact_state_loader = callable(exact_state_loader)
            if has_exact_state_getter != has_exact_state_loader:
                raise RuntimeError(
                    f"command term {term_name!r} must implement exact_resume_state_dict() "
                    "and load_exact_resume_state_dict(state, strict=True) as a pair"
                )
            if task_first_exact and not has_exact_state_getter:
                raise RuntimeError(
                    "task-first exact resume requires every active command term to implement "
                    f"explicit hooks; missing on {term_name!r}"
                )
            if has_exact_state_getter:
                exact_state = exact_state_getter()
                if not isinstance(exact_state, dict):
                    raise TypeError(
                        f"command term {term_name!r} exact_resume_state_dict() must return a dict"
                    )
                state["command_terms"][term_name] = {
                    "capture_mode": "explicit",
                    "term_type": f"{type(term).__module__}.{type(term).__qualname__}",
                    "exact_state": exact_state,
                }
                # The explicit API owns the whole term state. Mixing it with the heuristic scanner
                # would create two competing truths and could overwrite a strict restore.
                continue

            term_state = {"scalars": {}, "tensors": {}, "tensor_dicts": {}}

            # 标量类课程驱动:EMA 成功率/摔倒统计(*_acc / *_acc_c)、成功门控扰动幅度
            # (_curr_perturb_scale)、自适应 sigma 及其误差 EMA(_adaptive_sigma_* /
            # *_err_sum / *_err_sum_c,main 侧新增的两个后缀)。这些决定下一个 rollout 的
            # 目标分布,但不属于策略权重。挑选规则宁多勿漏:多存一个标量无害(恢复端只
            # setattr 认得的),漏存一个课程就悄悄回零。
            for attr, value in vars(term).items():
                is_resume_scalar = (
                    attr == "_curr_perturb_scale"
                    or attr.startswith("_adaptive_sigma_")
                    or attr.endswith("_acc")
                    or attr.endswith("_acc_c")
                    or attr.endswith("_err_sum")
                    or attr.endswith("_err_sum_c")
                    or attr in self._RESUME_SCALAR_DICT_ATTRS
                )
                if is_resume_scalar and self._is_scalar_tree(value):
                    term_state["scalars"][attr] = value

            # RallyV14 samples 35% of true resets from a ring of completed follow-through states.
            # Keeping that ring avoids silently dropping the recovery population after a restart.
            # MotionCommand 的自适应失败分箱统计(bin_failed_count)同理。
            for attr in (
                "bin_failed_count",
                "_current_bin_failed",
                "_post_swing_root",
                "_post_swing_joint_pos",
                "_post_swing_joint_vel",
            ):
                value = getattr(term, attr, None)
                if torch.is_tensor(value):
                    term_state["tensors"][attr] = value.detach().cpu().clone()
            for attr in ("_post_swing_ptr", "_post_swing_count"):
                value = getattr(term, attr, None)
                if isinstance(value, int):
                    term_state["scalars"][attr] = value

            # main 侧新增:HER 已实现回放环本体(dict[clip_key -> tensor])。不存它们,
            # 续训后 achieved-target 回放要从零重新攒,等效于把 35%/HER 采样人口清空。
            for attr in self._RESUME_TENSOR_DICT_ATTRS:
                value = getattr(term, attr, None)
                if isinstance(value, dict) and value and all(
                    torch.is_tensor(item) for item in value.values()
                ):
                    term_state["tensor_dicts"][attr] = {
                        key: item.detach().cpu().clone() for key, item in value.items()
                    }

            # Schema 3 binds the complete ordered active-term tuple even when a legacy term has no
            # recognized persistent attributes.  Omitting an empty term would make a later
            # add/remove/reorder invisible to exact resume.
            state["command_terms"][term_name] = term_state
        return state

    def _task_first_exact_resume_required(self) -> bool:
        """Return whether this runtime is the formal task-first command path."""

        env = getattr(self.env, "unwrapped", self.env)
        commands = getattr(getattr(env, "cfg", None), "commands", None)
        racket = None if commands is None else getattr(commands, "racket_target", None)
        return str(getattr(racket, "target_mode", "")) == "task_first"

    def _validate_task_first_exact_resume_terms(self) -> None:
        """Fail before rollout unless every task-first command term has paired explicit hooks."""

        if not self._task_first_exact_resume_required():
            return
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "command_manager", None)
        if manager is None:
            raise RuntimeError("task-first exact resume requires a command manager")
        raw_names = tuple(getattr(manager, "active_terms", ()))
        names = tuple(str(name) for name in raw_names)
        if not names:
            raise RuntimeError("task-first exact resume requires active command terms")
        if len(names) != len(set(names)):
            raise RuntimeError("command manager active_terms contains duplicate names")
        missing = []
        for raw_name, name in zip(raw_names, names):
            term = manager.get_term(raw_name)
            getter = getattr(term, "exact_resume_state_dict", None)
            loader = getattr(term, "load_exact_resume_state_dict", None)
            if callable(getter) != callable(loader):
                raise RuntimeError(
                    f"command term {name!r} must implement exact resume hooks as a pair"
                )
            if not callable(getter):
                missing.append(name)
        if missing:
            raise RuntimeError(
                "task-first exact resume requires explicit hooks on every active command term; "
                f"missing={missing}"
            )

    def _restore_environment_resume_state(self, resume_state: dict) -> tuple[int, str]:
        """Restore saved environment progress, with an iteration-derived fallback for old PTs."""
        env = getattr(self.env, "unwrapped", self.env)
        saved = resume_state.get("environment_resume_state")
        if isinstance(saved, dict) and "common_step_counter" in saved:
            saved_common_step_counter = saved["common_step_counter"]
            common_step_counter = int(saved_common_step_counter)
            source = "checkpoint"
        else:
            saved_common_step_counter = None
            # 老 checkpoint(状态未入档)仍有精确的"下一个迭代号":每完成一个 PPO 迭代,
            # 每个 env 都走 num_steps_per_env 个控制步,所以课程主时钟可以由迭代号精确推算
            # 出来,而不是回零(jiayi 对既有 V14 长跑档的回退推算,原样移植)。
            common_step_counter = int(resume_state["next_learning_iteration"]) * int(
                self.num_steps_per_env
            )
            source = "derived-from-iteration"

        manager = getattr(env, "command_manager", None)
        # As at capture time, enforce the current formal task-first runtime before considering any
        # saved payload.  Schema 1/2 remain readable, but they cannot be loaded into a task-first
        # environment whose current active tuple lacks explicit hooks.
        self._validate_task_first_exact_resume_terms()
        restored_terms = []
        saved_term_states = {}
        active_terms = {}
        if isinstance(saved, dict):
            environment_schema = int(saved.get("schema_version", 1))
            if environment_schema not in (
                1,
                2,
                _ENVIRONMENT_RESUME_SCHEMA_VERSION,
            ):
                raise RuntimeError(
                    "unsupported environment exact-resume schema "
                    f"{environment_schema}; refusing to guess command state"
                )
            # Legacy schema 1/2 deliberately retain their historical best-effort behavior when no
            # command manager exists. Schema 3 is exact and must validate even in that case: a
            # missing manager is an empty current tuple, never a reason to skip identity checks.
            if manager is not None or environment_schema >= 3:
                saved_term_states = saved.get("command_terms", {})
                if not isinstance(saved_term_states, dict):
                    raise TypeError("environment exact-resume command_terms must be a dict")

                raw_active_term_names = (
                    tuple(getattr(manager, "active_terms", ()))
                    if manager is not None
                    else ()
                )
                active_term_names = tuple(str(name) for name in raw_active_term_names)
                if len(active_term_names) != len(set(active_term_names)):
                    raise RuntimeError("command manager active_terms contains duplicate names")
                active_terms = {
                    name: manager.get_term(raw_name)
                    for raw_name, name in zip(
                        raw_active_term_names, active_term_names
                    )
                }

            if environment_schema >= 3:
                if (
                    type(saved_common_step_counter) is not int
                    or saved_common_step_counter < 0
                ):
                    raise RuntimeError(
                        "schema-3 environment common_step_counter must be a "
                        "nonnegative plain integer"
                    )
                expected_keys = {
                    "schema_version",
                    "common_step_counter",
                    "active_term_names",
                    "command_terms",
                }
                if set(saved) != expected_keys:
                    raise RuntimeError(
                        "schema-3 environment exact-resume keys do not match the strict schema"
                    )
                saved_active_names = saved.get("active_term_names")
                if (
                    type(saved_active_names) is not list
                    or any(type(name) is not str for name in saved_active_names)
                    or len(saved_active_names) != len(set(saved_active_names))
                ):
                    raise RuntimeError(
                        "schema-3 environment active_term_names must be a unique string list"
                    )
                saved_active_names = tuple(saved_active_names)
                if saved_active_names != active_term_names:
                    raise RuntimeError(
                        "exact-resume ordered command term identity mismatch: "
                        f"checkpoint={saved_active_names}, current={active_term_names}"
                    )
                if tuple(saved_term_states) != active_term_names:
                    raise RuntimeError(
                        "schema-3 command_terms must contain every active term in exact order: "
                        f"checkpoint={tuple(saved_term_states)}, current={active_term_names}"
                    )
                saved_explicit = {
                    str(name)
                    for name, term_state in saved_term_states.items()
                    if isinstance(term_state, dict)
                    and term_state.get("capture_mode") == "explicit"
                }
                current_explicit = set()
                for term_name, term in active_terms.items():
                    getter = getattr(term, "exact_resume_state_dict", None)
                    loader = getattr(term, "load_exact_resume_state_dict", None)
                    has_getter = callable(getter)
                    has_loader = callable(loader)
                    if has_getter != has_loader:
                        raise RuntimeError(
                            f"command term {term_name!r} must implement exact resume hooks as a pair"
                        )
                    if has_getter:
                        current_explicit.add(term_name)
                if saved_explicit != current_explicit:
                    raise RuntimeError(
                        "exact-resume command term identity mismatch: "
                        f"checkpoint explicit terms={sorted(saved_explicit)}, "
                        f"current explicit terms={sorted(current_explicit)}"
                    )
                if self._task_first_exact_resume_required() and current_explicit != set(
                    active_term_names
                ):
                    raise RuntimeError(
                        "task-first schema-3 restore requires explicit state for every active "
                        f"command term; explicit={sorted(current_explicit)}, "
                        f"active={list(active_term_names)}"
                    )

        # Do not mutate even the curriculum clock until schema-3 structure and active-term identity
        # have passed. Explicit term loaders below remain responsible for their own atomicity.
        env.common_step_counter = common_step_counter

        if isinstance(saved, dict) and manager is not None:
            for term_name, term_state in saved_term_states.items():
                term_name = str(term_name)
                if not isinstance(term_state, dict):
                    raise TypeError(
                        f"command term {term_name!r} exact-resume state must be a dict"
                    )
                if term_state.get("capture_mode") == "explicit":
                    if term_name not in active_terms:
                        raise RuntimeError(
                            f"exact-resume command term {term_name!r} is absent from current config"
                        )
                    term = active_terms[term_name]
                    current_term_type = f"{type(term).__module__}.{type(term).__qualname__}"
                    saved_term_type = term_state.get("term_type")
                    if saved_term_type != current_term_type:
                        raise RuntimeError(
                            f"exact-resume command term {term_name!r} type mismatch: "
                            f"checkpoint={saved_term_type!r}, current={current_term_type!r}"
                        )
                    loader = getattr(term, "load_exact_resume_state_dict", None)
                    if not callable(loader):
                        raise RuntimeError(
                            f"command term {term_name!r} cannot load explicit exact-resume state"
                        )
                    exact_state = term_state.get("exact_state")
                    if not isinstance(exact_state, dict):
                        raise TypeError(
                            f"command term {term_name!r} explicit exact-resume state must be a dict"
                        )
                    # strict=True is an interface contract, not a best-effort hint: manifest SHA,
                    # action order, tensor shapes and any other term identity checks must fail loud.
                    loader(exact_state, strict=True)
                    restored_terms.append(term_name)
                    continue
                try:
                    term = manager.get_term(term_name)
                except (KeyError, ValueError):
                    # 档里多存的命令项(臂间配置差异):容忍跳过,其余项照常恢复。
                    continue
                restored = False
                for attr, value in term_state.get("scalars", {}).items():
                    if hasattr(term, attr):
                        # 整体 setattr(dict 也整个换):若 clip 集合在续训时变了,后续代码
                        # 按新 clip 取键会 KeyError —— fail-loud,好过静默混用两套课程统计。
                        setattr(term, attr, value)
                        restored = True
                for attr, value in term_state.get("tensors", {}).items():
                    if hasattr(term, attr) and torch.is_tensor(value):
                        current = getattr(term, attr)
                        device = current.device if torch.is_tensor(current) else term.device
                        setattr(term, attr, value.to(device=device))
                        restored = True
                for attr, saved_entries in term_state.get("tensor_dicts", {}).items():
                    current = getattr(term, attr, None)
                    if not isinstance(current, dict) or not isinstance(saved_entries, dict):
                        continue
                    for key, value in saved_entries.items():
                        # 只恢复当前配置也有的 clip 键:档里多出来的键容忍丢弃,新增 clip
                        # 保持新鲜初始化(HER 回放环对新 clip 从零攒起是正确语义)。
                        if key in current and torch.is_tensor(current[key]) and torch.is_tensor(value):
                            current[key] = value.to(device=current[key].device)
                            restored = True
                if restored:
                    restored_terms.append(str(term_name))
        print(
            "[MotionOnPolicyRunner] exact environment progress restored: "
            f"common_step_counter={common_step_counter} ({source}), "
            f"command_terms={restored_terms}",
            flush=True,
        )
        return common_step_counter, source

    @staticmethod
    def _validate_rng_tensor(saved, current, *, name: str) -> torch.Tensor:
        """Validate a serialized PyTorch RNG byte vector without mutating global RNG."""

        if not torch.is_tensor(saved):
            raise TypeError(f"{name} must be a torch.Tensor")
        if saved.dtype != current.dtype or tuple(saved.shape) != tuple(current.shape):
            raise RuntimeError(
                f"{name} shape/dtype mismatch: checkpoint={tuple(saved.shape)}/{saved.dtype}, "
                f"current={tuple(current.shape)}/{current.dtype}"
            )
        return saved.detach().cpu()

    def _validated_exact_rng_state(self, resume_state: dict):
        """Validate all sampling RNG payloads without mutating any global generator."""

        required = (
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "torch_cuda_random_states",
        )
        missing = [name for name in required if name not in resume_state]
        if missing:
            raise RuntimeError(
                "exact-resume checkpoint is missing RNG state fields: " + ", ".join(missing)
            )

        python_state = resume_state["python_random_state"]
        numpy_state = resume_state["numpy_random_state"]
        # Validate Python and NumPy payloads on private generators so a malformed CUDA payload
        # cannot leave the process half-restored.
        python_probe = random.Random()
        try:
            python_probe.setstate(python_state)
        except Exception as exc:
            raise RuntimeError("invalid python_random_state in exact-resume checkpoint") from exc
        numpy_probe = np.random.RandomState()
        try:
            numpy_probe.set_state(numpy_state)
        except Exception as exc:
            raise RuntimeError("invalid numpy_random_state in exact-resume checkpoint") from exc

        torch_state = self._validate_rng_tensor(
            resume_state["torch_random_state"],
            torch.get_rng_state(),
            name="torch_random_state",
        )

        saved_cuda_states = resume_state["torch_cuda_random_states"]
        if not isinstance(saved_cuda_states, (list, tuple)):
            raise TypeError("torch_cuda_random_states must be a list or tuple")
        cuda_available = bool(torch.cuda.is_available())
        current_cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and current_cuda_count <= 0:
            raise RuntimeError("torch.cuda reports available but has no visible devices")
        declared_cuda_count = int(
            resume_state.get("torch_cuda_device_count", len(saved_cuda_states))
        )
        if declared_cuda_count != len(saved_cuda_states):
            raise RuntimeError(
                "checkpoint CUDA RNG count is internally inconsistent: "
                f"declared={declared_cuda_count}, states={len(saved_cuda_states)}"
            )
        if declared_cuda_count != current_cuda_count:
            raise RuntimeError(
                "CUDA RNG device-count mismatch: "
                f"checkpoint={declared_cuda_count}, current={current_cuda_count}"
            )
        current_cuda_states = torch.cuda.get_rng_state_all() if current_cuda_count else []
        if len(current_cuda_states) != current_cuda_count:
            raise RuntimeError(
                "torch.cuda.get_rng_state_all() returned an unexpected number of states: "
                f"{len(current_cuda_states)} vs {current_cuda_count} devices"
            )
        cuda_states = [
            self._validate_rng_tensor(saved, current, name=f"torch_cuda_random_states[{idx}]")
            for idx, (saved, current) in enumerate(
                zip(saved_cuda_states, current_cuda_states)
            )
        ]
        return python_state, numpy_state, torch_state, cuda_states

    def _restore_exact_rng_state(self, resume_state: dict) -> None:
        """Restore every sampling RNG atomically after validation and before ``env.reset()``."""

        python_state, numpy_state, torch_state, cuda_states = (
            self._validated_exact_rng_state(resume_state)
        )

        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _checkpoint_exact_resume_state(loaded):
        """Return the canonical exact-resume payload from a loaded checkpoint, if present."""

        if not isinstance(loaded, dict):
            return None
        checkpoint_infos = loaded.get("infos")
        state = (
            checkpoint_infos.get("hope_exact_resume_state")
            if isinstance(checkpoint_infos, dict)
            else None
        )
        if state is None:
            # Cross-stack compatibility: jiayi/hitterobs stored the same payload at the PT top
            # level. Strict mode accepts it only if the payload itself satisfies schema 3.
            state = loaded.get("hope_exact_resume_state")
        return state

    @staticmethod
    def _checkpoint_byte_snapshot(path) -> io.BytesIO:
        """Read one immutable checkpoint snapshot for strict preflight and base loading."""

        try:
            filesystem_path = os.fspath(path)
        except TypeError as exc:
            raise TypeError(
                "required exact resume needs a filesystem checkpoint path"
            ) from exc
        with open(filesystem_path, "rb") as stream:
            return io.BytesIO(stream.read())

    def _validate_required_adam_state(
        self,
        optimizer_state,
        *,
        prefix: str,
        expected_learning_rate: float,
    ) -> None:
        """Validate a complete Adam continuation state against the live parameter layout."""

        optimizer = getattr(getattr(self, "alg", None), "optimizer", None)
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise RuntimeError(
                f"{prefix} cannot validate optimizer type "
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}; "
                "task-first exact resume currently requires Adam/AdamW"
            )
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
            "state",
            "param_groups",
        }:
            raise RuntimeError(
                f"{prefix} is missing a canonical optimizer_state_dict"
            )
        saved_state = optimizer_state["state"]
        saved_groups = optimizer_state["param_groups"]
        current_groups = optimizer.param_groups
        if (
            not isinstance(saved_state, dict)
            or not isinstance(saved_groups, list)
            or len(saved_groups) != len(current_groups)
            or not saved_groups
        ):
            raise RuntimeError(
                f"{prefix} optimizer state/group structure does not match the live Adam"
            )

        saved_ids = []
        live_parameters = []
        amsgrad_flags = []
        for group_index, (saved_group, current_group) in enumerate(
            zip(saved_groups, current_groups)
        ):
            if (
                type(saved_group) is not dict
                or set(saved_group) != set(current_group)
                or type(saved_group.get("params")) is not list
                or type(current_group.get("params")) is not list
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} is malformed"
                )
            group_ids = saved_group["params"]
            group_parameters = current_group["params"]
            if not group_ids or len(group_ids) != len(group_parameters):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has the wrong size"
                )
            if any(type(param_id) is not int for param_id in group_ids):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has non-integer IDs"
                )
            if any(not torch.is_tensor(parameter) for parameter in group_parameters):
                raise RuntimeError(
                    f"{prefix} live optimizer param group {group_index} is malformed"
                )
            saved_lr = saved_group.get("lr")
            if (
                type(saved_lr) not in (int, float)
                or not math.isfinite(float(saved_lr))
                or float(saved_lr) <= 0.0
                or float(saved_lr) != float(expected_learning_rate)
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has an invalid "
                    "or inconsistent learning rate"
                )
            betas = saved_group.get("betas")
            if (
                type(betas) is not tuple
                or len(betas) != 2
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    or float(value) >= 1.0
                    for value in betas
                )
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has invalid betas"
                )
            eps = saved_group.get("eps")
            weight_decay = saved_group.get("weight_decay")
            if (
                type(eps) not in (int, float)
                or not math.isfinite(float(eps))
                or float(eps) <= 0.0
                or type(weight_decay) not in (int, float)
                or not math.isfinite(float(weight_decay))
                or float(weight_decay) < 0.0
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has invalid "
                    "epsilon or weight decay"
                )
            for key in set(current_group) - {"params", "lr"}:
                saved_value = saved_group[key]
                current_value = current_group[key]
                if torch.is_tensor(saved_value) or torch.is_tensor(current_value):
                    equal = (
                        torch.is_tensor(saved_value)
                        and torch.is_tensor(current_value)
                        and torch.equal(saved_value, current_value)
                    )
                else:
                    equal = type(saved_value) is type(current_value) and saved_value == current_value
                if not equal:
                    raise RuntimeError(
                        f"{prefix} optimizer param group {group_index} changed {key!r}"
                    )
            saved_ids.extend(group_ids)
            live_parameters.extend(group_parameters)
            amsgrad_flags.extend(
                [bool(saved_group.get("amsgrad", False))] * len(group_ids)
            )

        if (
            len(saved_ids) != len(set(saved_ids))
            or any(type(param_id) is not int for param_id in saved_state)
            or set(saved_state) != set(saved_ids)
        ):
            raise RuntimeError(
                f"{prefix} optimizer state does not cover the exact parameter ID set"
            )

        for param_id, parameter, uses_amsgrad in zip(
            saved_ids, live_parameters, amsgrad_flags
        ):
            entry = saved_state[param_id]
            expected_entry_keys = {"step", "exp_avg", "exp_avg_sq"}
            if uses_amsgrad:
                expected_entry_keys.add("max_exp_avg_sq")
            if type(entry) is not dict or set(entry) != expected_entry_keys:
                raise RuntimeError(
                    f"{prefix} optimizer state for parameter {param_id} lacks Adam moments"
                )
            if any(
                not torch.is_tensor(entry[name])
                or tuple(entry[name].shape) != tuple(parameter.shape)
                or entry[name].dtype != parameter.dtype
                or (
                    torch.is_floating_point(entry[name])
                    and not bool(torch.isfinite(entry[name]).all())
                )
                for name in ("exp_avg", "exp_avg_sq")
            ):
                raise RuntimeError(
                    f"{prefix} optimizer moments for parameter {param_id} "
                    "do not match the live parameter"
                )
            step = entry["step"]
            if torch.is_tensor(step):
                step_value = (
                    float(step.detach().cpu().item())
                    if step.numel() == 1
                    and step.dtype != torch.bool
                    and not torch.is_complex(step)
                    else float("nan")
                )
                valid_step = (
                    step.numel() == 1
                    and step.dtype != torch.bool
                    and not torch.is_complex(step)
                    and bool(torch.isfinite(step).all())
                    and step_value > 0.0
                    and step_value.is_integer()
                )
            else:
                step_value = (
                    float(step) if type(step) in (int, float) else float("nan")
                )
                valid_step = (
                    type(step) in (int, float)
                    and math.isfinite(step_value)
                    and step_value > 0.0
                    and step_value.is_integer()
                )
            if not valid_step:
                raise RuntimeError(
                    f"{prefix} optimizer step for parameter {param_id} is invalid"
                )
            second_moment = entry["exp_avg_sq"]
            if not torch.is_floating_point(second_moment) or bool(
                (second_moment < 0.0).any()
            ):
                raise RuntimeError(
                    f"{prefix} optimizer second moment for parameter {param_id} is invalid"
                )
            if uses_amsgrad:
                maximum = entry.get("max_exp_avg_sq")
                if (
                    not torch.is_tensor(maximum)
                    or tuple(maximum.shape) != tuple(parameter.shape)
                    or maximum.dtype != parameter.dtype
                    or (
                        torch.is_floating_point(maximum)
                        and not bool(torch.isfinite(maximum).all())
                    )
                    or not torch.is_floating_point(maximum)
                    or bool((maximum < 0.0).any())
                    or bool((maximum < second_moment).any())
                ):
                    raise RuntimeError(
                        f"{prefix} optimizer AMSGrad state for parameter {param_id} "
                        "does not match the live parameter"
                    )

    def _preflight_required_exact_resume_checkpoint(
        self,
        loaded,
        *,
        path: str,
        load_optimizer: bool,
    ) -> dict:
        """Validate the non-negotiable task-first resume envelope before loading model bytes."""

        prefix = f"required exact resume checkpoint {path!r}"
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{prefix} must be a dictionary")
        if self.log_dir is None:
            raise RuntimeError(
                f"{prefix} cannot use the evaluation load path without environment restore"
            )
        if load_optimizer is not True:
            raise RuntimeError(
                f"{prefix} refuses actor-only loading: load_optimizer must be True"
            )
        if getattr(getattr(self, "alg", None), "rnd", None) is not None:
            # Upstream persists a second model/optimizer pair for RND. Until that pair has the
            # same complete-state validator as the policy optimizer, accepting it here would make
            # the "exact" construction contract false.
            raise RuntimeError(
                f"{prefix} does not yet support exact RND resume; disable RND or add "
                "strict rnd_state_dict/rnd_optimizer_state_dict validation"
            )
        checkpoint_infos = loaded.get("infos")
        checkpoint_contract_schema = (
            checkpoint_infos.get(CHECKPOINT_CONTRACT_SCHEMA_KEY)
            if isinstance(checkpoint_infos, dict)
            else None
        )
        checkpoint_lineage_exact = (
            checkpoint_infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY)
            if isinstance(checkpoint_infos, dict)
            else None
        )
        if (
            not isinstance(checkpoint_infos, dict)
            or type(checkpoint_contract_schema) is not int
            or checkpoint_contract_schema != self.training_contract_schema_version
            or checkpoint_infos.get(CHECKPOINT_CONTRACT_SHA_KEY)
            != self.training_contract_sha256
            or type(checkpoint_lineage_exact) is not int
            or checkpoint_lineage_exact != 1
        ):
            raise RuntimeError(
                f"{prefix} is not bound to this exact training contract"
            )

        checkpoint_iteration = loaded.get("iter")
        if type(checkpoint_iteration) is not int or checkpoint_iteration < 0:
            raise RuntimeError(f"{prefix} has an invalid base iteration")

        state = self._checkpoint_exact_resume_state(loaded)
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != _EXACT_RESUME_SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"{prefix} requires hope_exact_resume_state schema 3; "
                "legacy warm-start fallback is forbidden"
            )
        required_fields = {
            "next_learning_iteration",
            "tot_timesteps",
            "tot_time",
            "algorithm_learning_rate",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "torch_cuda_random_states",
            "torch_cuda_device_count",
            "environment_resume_state",
        }
        missing = sorted(required_fields - set(state))
        if missing:
            raise RuntimeError(
                f"{prefix} schema-3 state is incomplete; missing={missing}"
            )
        tot_timesteps = state["tot_timesteps"]
        tot_time = state["tot_time"]
        learning_rate = state["algorithm_learning_rate"]
        cuda_count = state["torch_cuda_device_count"]
        if type(tot_timesteps) is not int or tot_timesteps < 0:
            raise RuntimeError(f"{prefix} has invalid tot_timesteps")
        if (
            type(tot_time) not in (int, float)
            or not math.isfinite(float(tot_time))
            or float(tot_time) < 0.0
        ):
            raise RuntimeError(f"{prefix} has invalid tot_time")
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise RuntimeError(f"{prefix} has invalid algorithm_learning_rate")
        if type(cuda_count) is not int or cuda_count < 0:
            raise RuntimeError(f"{prefix} has invalid torch_cuda_device_count")
        next_iteration = state["next_learning_iteration"]
        if (
            type(next_iteration) is not int
            or next_iteration != checkpoint_iteration + 1
        ):
            raise RuntimeError(
                f"{prefix} has stale next_learning_iteration: "
                f"checkpoint={next_iteration!r}, expected={checkpoint_iteration + 1}"
            )
        nested = state["environment_resume_state"]
        if (
            not isinstance(nested, dict)
            or type(nested.get("schema_version")) is not int
            or nested["schema_version"] != _ENVIRONMENT_RESUME_SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"{prefix} requires a schema-3 environment_resume_state"
            )
        nested_common_step_counter = nested.get("common_step_counter")
        if (
            type(nested_common_step_counter) is not int
            or nested_common_step_counter < 0
        ):
            raise RuntimeError(
                f"{prefix} schema-3 environment common_step_counter must be a "
                "nonnegative plain integer"
            )
        # Dry validation occurs before policy/optimizer bytes are applied. The same payload is
        # validated once more and committed atomically immediately before the resumed env.reset().
        self._validated_exact_rng_state(state)
        self._validate_required_adam_state(
            loaded.get("optimizer_state_dict"),
            prefix=prefix,
            expected_learning_rate=float(learning_rate),
        )
        return state

    def load(self, path: str, load_optimizer: bool = True, **kwargs):
        """Load a checkpoint; on a training resume also restore curriculum progress.

        人话:base 的 load 只拿回权重/优化器/迭代号,环境课程全部回到第 0 步。这里在训练
        续跑(log_dir 不为 None)时把精确续训包里的课程主时钟和命令项状态一并恢复;老档没
        有状态就按迭代号精确推算主时钟。评测器(isaac_bank_exam / play)也走 runner.load,
        但它们 log_dir=None 且自带确定性调度 —— 那条路保持与移植前逐字节相同的行为。
        ``require_exact_resume_state=True`` 是 task-first 训练的构造期铁律:optimizer、外层
        schema 3、内层 schema 3、迭代连续性或命令项 strict state 任一缺失都拒绝,绝不把
        actor-only checkpoint 静默解释成 warm start。
        """
        strict_resume = bool(getattr(self, "require_exact_resume_state", False))
        snapshot = self._checkpoint_byte_snapshot(path) if strict_resume else None
        load_source = snapshot if snapshot is not None else path
        loaded = torch.load(load_source, map_location="cpu", weights_only=False)
        required_state = None
        if strict_resume:
            required_state = self._preflight_required_exact_resume_checkpoint(
                loaded,
                path=path,
                load_optimizer=load_optimizer,
            )
            # Upstream OnPolicyRunner.load performs its own torch.load. Rewind the exact same
            # in-memory bytes so policy/optimizer and curriculum/RNG cannot come from two path
            # reads of a concurrently replaced checkpoint.
            snapshot.seek(0)
        infos = super().load(load_source, load_optimizer=load_optimizer, **kwargs)
        if self.log_dir is None:
            return infos
        state = (
            required_state
            if required_state is not None
            else self._checkpoint_exact_resume_state(loaded)
        )
        if state is not None:
            if (
                not isinstance(state, dict)
                or int(state.get("schema_version", 0)) not in _SUPPORTED_EXACT_RESUME_SCHEMAS
            ):
                raise RuntimeError(
                    "unsupported hope_exact_resume_state schema "
                    f"{state.get('schema_version', None) if isinstance(state, dict) else type(state)}; "
                    f"refusing to guess how to resume: {path}"
                )
            if int(state["schema_version"]) >= 3:
                nested = state.get("environment_resume_state")
                if (
                    not isinstance(nested, dict)
                    or type(nested.get("schema_version")) is not int
                    or nested["schema_version"] != _ENVIRONMENT_RESUME_SCHEMA_VERSION
                ):
                    raise RuntimeError(
                        "schema-3 exact resume requires a schema-3 "
                        "environment_resume_state; refusing a partial command restore"
                    )
            # 一致性铁律:状态包写入时恒有 next_learning_iteration == iter+1。checkpoint 外科
            # 手术(make_hitter_warmstart / warm_start_realsensor 把 iter 归零但整份保留 infos)
            # 会打破它 —— 那时状态是"上一世"的,拿来续课程等于劫持一次刻意的全新热启动。
            # 响亮降级成"老档"语义:主时钟按本档 iter 推算,课程统计从头攒。
            expected_next = int(self.current_learning_iteration) + 1
            if int(state.get("next_learning_iteration", -1)) != expected_next:
                if getattr(self, "require_exact_resume_state", False):
                    raise RuntimeError(
                        "required exact resume checkpoint iteration changed during base load: "
                        f"state next={state.get('next_learning_iteration')!r}, "
                        f"base iter+1={expected_next}"
                    )
                print(
                    "[MotionOnPolicyRunner] WARNING: hope_exact_resume_state is stale "
                    f"(next_learning_iteration={state.get('next_learning_iteration')!r} vs "
                    f"checkpoint iter+1={expected_next}); treating as a legacy warm-start "
                    "checkpoint — curriculum statistics start fresh",
                    flush=True,
                )
                state = None
        if state is None:
            if getattr(self, "require_exact_resume_state", False):
                # Defensive invariant: strict preflight above must either return schema 3 or raise.
                raise RuntimeError(
                    "required exact resume state disappeared before environment restore"
                )
            # 老 checkpoint:课程细节无从恢复,但主时钟按迭代号精确推算(见 _restore_…)。
            # 迭代号本身维持 base 语义(iter=N,会重复一次第 N 个更新)——没有状态包时不敢
            # 替它做 N+1 的决定。
            self._restore_environment_resume_state(
                {"next_learning_iteration": int(self.current_learning_iteration) + 1}
            )
        else:
            self.current_learning_iteration = int(state["next_learning_iteration"])
            self.tot_timesteps = int(state.get("tot_timesteps", self.tot_timesteps))
            self.tot_time = float(state.get("tot_time", self.tot_time))
            if getattr(self.alg, "schedule", None) == "adaptive" and "algorithm_learning_rate" in state:
                # 自适应 KL 调度下 lr 是"续"出来的运行状态:不恢复的话第一次 update 会从
                # YAML 初值重新适应,平白抖一下。固定调度(fixed)不恢复 —— YAML 里改 lr
                # 再续训必须生效;这是 main 共享 load() 与 jiayi 独立 exact 路径的取舍差异。
                self.alg.learning_rate = float(state["algorithm_learning_rate"])
                for param_group in self.alg.optimizer.param_groups:
                    param_group["lr"] = self.alg.learning_rate
            self._restore_environment_resume_state(state)
            if int(state["schema_version"]) >= 3:
                # Restoring RNG any earlier lets command-state loading consume the resumed stream;
                # restoring it any later lets env.reset() sample from the constructor seed. This is
                # therefore deliberately the final operation before the first resumed reset.
                self._restore_exact_rng_state(state)
        # The simulator itself is not serialized. Reset once after restoring the curriculum counter
        # and command-manager globals so the first resumed rollout is sampled from the correct
        # distribution instead of the constructor-time step-zero curriculum.
        self.env.reset()
        return infos

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Defer SIGINT/SIGTERM to an iteration boundary and save that completed iteration.

        人话:Ctrl-C 不再当场掐死训练(半个迭代的档没有存在价值),而是登记一个"想停"标记,
        等当前 PPO 迭代完整跑完、log() 落账之后存一份带精确续训状态的档再退出。
        第二次 Ctrl-C/SIGTERM = "别等了,立刻死"(2026-07-25):迭代若卡住(env 步进不返回、
        logger 重试死循环),旧版会把后续信号全部静默吞掉,操作员只能另开 shell 找 PID 发
        SIGKILL。现在第二次信号恢复 OS 默认处置并重新对自己投递——不存档,直接终止。
        (老实说明:若主线程死死卡在 native Isaac/CUDA 调用里,Python 层任何 handler 都
        不会被执行,第一次信号同样没反应,那种情况仍然只有 SIGKILL 能救。)
        """
        original_update = getattr(self.alg, "update", None)
        if not callable(original_update):
            raise RuntimeError(
                "MotionOnPolicyRunner requires a callable alg.update() for exact rollout boundaries"
            )
        if getattr(self, "_rollout_update_wrapper_active", False):
            raise RuntimeError("MotionOnPolicyRunner.learn() cannot be entered recursively")
        pending_signal = None
        previous_handlers = {}

        def request_stop(signum, _frame):
            nonlocal pending_signal
            if pending_signal is None:
                pending_signal = signum
                print(
                    "\n[MotionOnPolicyRunner] interrupt received; finishing the current PPO "
                    "iteration before saving",
                    flush=True,
                )
            else:
                # 第二次信号:升级为立即终止。先恢复该信号的 OS 默认动作再 os.kill 自投——
                # 默认动作由内核执行,不依赖解释器回到字节码循环,比 raise KeyboardInterrupt
                # 硬(后者在 finally/except 链里还可能被吞)。代价:不存档(与提示语一致)。
                print(
                    "\n[MotionOnPolicyRunner] second interrupt — exiting immediately, "
                    "NO checkpoint",
                    flush=True,
                )
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
        except ValueError:
            # signal.signal is restricted to the main thread. Training normally runs there; retain
            # upstream behavior if an embedding invokes learn() from another thread.
            previous_handlers.clear()

        self._boundary_stop_requested = lambda: pending_signal is not None
        # Upstream RSL-RL calls ``log()`` only on rank 0.  Curriculum advancement is environment
        # state, not presentation, so attach it to the algorithm's one update call per PPO loop.
        # This executes on every rank even when ``disable_logs`` is true and preserves the upstream
        # loop rather than copying a version-sensitive implementation here.
        next_rollout_step = int(self.current_learning_iteration)

        def update_with_rollout_boundary(*args, **kwargs):
            nonlocal next_rollout_step
            result = original_update(*args, **kwargs)
            self._notify_command_terms_rollout_end(next_rollout_step)
            next_rollout_step += 1
            return result

        self._rollout_update_wrapper_active = True
        try:
            self.alg.update = update_with_rollout_boundary
            super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self.alg.update = original_update
            self._rollout_update_wrapper_active = False
            self._boundary_stop_requested = None
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        step = int(locs["it"])
        super().log(locs, width=width, pad=pad)
        # Consume/print even when TensorBoard/W&B is disabled: this stdout JSON line is the exact
        # per-update receipt, while dashboard logging is optional presentation only.
        exact_behavior = self._consume_exact_behavior_updates(step)
        if not self.disable_logs and self.writer is not None:
            self._log_live_metrics(step, exact_behavior=exact_behavior)
        # Ctrl-C 缓冲(见 learn()):恰好在一个 PPO 迭代完整结束、日志落账之后才真正存档退出。
        stop_requested = getattr(self, "_boundary_stop_requested", None)
        if callable(stop_requested) and stop_requested():
            checkpoint = pathlib.Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"
            self.save(str(checkpoint))
            print(
                "[MotionOnPolicyRunner] interrupt checkpoint saved at a completed PPO boundary: "
                f"{checkpoint}",
                flush=True,
            )
            raise KeyboardInterrupt

    def _notify_command_terms_rollout_end(self, step: int) -> None:
        """Notify each active command term once, before any per-update ledger is consumed."""

        step = int(step)
        previous_step = getattr(self, "_rollout_end_notified_step", None)
        if previous_step == step:
            return
        if previous_step is not None and step < int(previous_step):
            raise RuntimeError(
                f"PPO update moved backwards at rollout boundary: {step} < {previous_step}"
            )
        # Mark before invoking user code. An exception is fatal and propagates; this guard prevents
        # an outer logger retry from double-advancing an already-mutated curriculum.
        self._rollout_end_notified_step = step

        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "command_manager", None)
        if manager is None:
            return
        raw_term_names = tuple(getattr(manager, "active_terms", ()))
        term_names = tuple(str(name) for name in raw_term_names)
        if len(term_names) != len(set(term_names)):
            raise RuntimeError("command manager active_terms contains duplicate names")
        for raw_term_name, term_name in zip(raw_term_names, term_names):
            term = manager.get_term(raw_term_name)
            callback = getattr(term, "on_rollout_end", None)
            if callable(callback):
                callback(step)

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

    def _check_zero_return_alarm(self, term_name: str, counters: dict, step: int) -> None:
        """Accumulate the per-family strike ledger across updates and alarm on a pinned-at-zero side.

        The per-update record is a window, so the decision needs a run-lifetime total; the counters are
        non-decaying integers, so summing them is exact.
        """

        totals = getattr(self, "_zero_return_cumulative", None)
        if totals is None:
            totals = {}
            self._zero_return_cumulative = totals
        term_totals = totals.setdefault(term_name, {})
        for family in _STRIKE_FAMILIES:
            for key in (
                f"strike_opportunity_count_{family}",
                f"virtual_legal_return_count_{family}",
            ):
                if key in counters:
                    term_totals[key] = term_totals.get(key, 0) + float(counters[key])
        levels = zero_return_alarm_levels(term_totals)
        for family in _STRIKE_FAMILIES:
            opportunity_key = f"strike_opportunity_count_{family}"
            if opportunity_key not in term_totals:
                continue
            level = levels.get(family)
            self._log_scalar(
                f"Live/{term_name}/zero_return_alarm_{family}",
                1.0 if level else 0.0,
                step,
            )
            if level is None:
                continue
            message = (
                f"[HOPE ALARM] {term_name}: virtual_legal_return_count_{family} is still 0 after "
                f"{term_totals[opportunity_key]:.0f} cumulative strike opportunities "
                f"(ppo_update {int(step)}). A side that NEVER returns is a broken command, not a "
                f"slow learner — check that the {family} racket target box clears the table "
                f"(z_lo >= vb_table_surface_z + ball radius) before spending more GPU on this run."
            )
            print(message, flush=True)
            if level == "abort":
                raise RuntimeError(message)

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
                    self._check_zero_return_alarm(
                        str(term_name), exact_record["counters"], step
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
            if "qdes_limit_barrier_probe" in active_reward_terms:
                # Wave-Q qbar: weight-independent all-joint q_des limit-barrier ledger
                # (above-margin joint counts + max intrusion depth).  Consumed exactly once
                # per PPO update; counts/sums must not be averaged over num_envs.
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_qdes_limit_barrier_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_qdes_limit_barrier_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/qdes_limit_barrier/{counter_name}",
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
