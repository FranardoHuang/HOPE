from __future__ import annotations

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
        和各命令项的课程状态;(3) RNG/W&B 等审计信息。RNG 与 W&B run 字段目前只存不恢复:
        main 没有 jiayi 栈那种独立的 exact_resume 开关,共享的 load() 不能悄悄覆盖本次配置的
        seed 或 W&B run;存下来是为了跨栈对账、以及将来加显式开关时不用改档案格式。
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
            "schema_version": 2,
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
            # tensor_dicts 段。读端(_restore_environment_resume_state)对未知段/未知键一律
            # 容忍,只恢复认得上的 —— 多存无害,漏存有害。
            "schema_version": 2,
            "common_step_counter": int(getattr(env, "common_step_counter", 0)),
            "command_terms": {},
        }
        manager = getattr(env, "command_manager", None)
        if manager is None:
            return state

        for term_name in getattr(manager, "active_terms", ()):
            term = manager.get_term(term_name)
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

            if term_state["scalars"] or term_state["tensors"] or term_state["tensor_dicts"]:
                state["command_terms"][str(term_name)] = term_state
        return state

    def _restore_environment_resume_state(self, resume_state: dict) -> tuple[int, str]:
        """Restore saved environment progress, with an iteration-derived fallback for old PTs."""
        env = getattr(self.env, "unwrapped", self.env)
        saved = resume_state.get("environment_resume_state")
        if isinstance(saved, dict) and "common_step_counter" in saved:
            common_step_counter = int(saved["common_step_counter"])
            source = "checkpoint"
        else:
            # 老 checkpoint(状态未入档)仍有精确的"下一个迭代号":每完成一个 PPO 迭代,
            # 每个 env 都走 num_steps_per_env 个控制步,所以课程主时钟可以由迭代号精确推算
            # 出来,而不是回零(jiayi 对既有 V14 长跑档的回退推算,原样移植)。
            common_step_counter = int(resume_state["next_learning_iteration"]) * int(
                self.num_steps_per_env
            )
            source = "derived-from-iteration"
        env.common_step_counter = common_step_counter

        manager = getattr(env, "command_manager", None)
        restored_terms = []
        if isinstance(saved, dict) and manager is not None:
            for term_name, term_state in saved.get("command_terms", {}).items():
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

    def load(self, path: str, load_optimizer: bool = True, **kwargs):
        """Load a checkpoint; on a training resume also restore curriculum progress.

        人话:base 的 load 只拿回权重/优化器/迭代号,环境课程全部回到第 0 步。这里在训练
        续跑(log_dir 不为 None)时把精确续训包里的课程主时钟和命令项状态一并恢复;老档没
        有状态就按迭代号精确推算主时钟。评测器(isaac_bank_exam / play)也走 runner.load,
        但它们 log_dir=None 且自带确定性调度 —— 那条路保持与移植前逐字节相同的行为。
        """
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        infos = super().load(path, load_optimizer=load_optimizer, **kwargs)
        if self.log_dir is None:
            return infos
        state = None
        if isinstance(loaded, dict):
            checkpoint_infos = loaded.get("infos")
            if isinstance(checkpoint_infos, dict):
                state = checkpoint_infos.get("hope_exact_resume_state")
            if state is None:
                # 跨栈对账:jiayi/hitterobs 把同名状态放在 PT 顶层而不是 infos 里;两处都认,
                # hitterobs 导出的档在 main 上续训也能拿到课程状态。
                state = loaded.get("hope_exact_resume_state")
        if state is not None:
            if not isinstance(state, dict) or int(state.get("schema_version", 0)) not in (1, 2):
                raise RuntimeError(
                    "unsupported hope_exact_resume_state schema "
                    f"{state.get('schema_version', None) if isinstance(state, dict) else type(state)}; "
                    f"refusing to guess how to resume: {path}"
                )
            # 一致性铁律:状态包写入时恒有 next_learning_iteration == iter+1。checkpoint 外科
            # 手术(make_hitter_warmstart / warm_start_realsensor 把 iter 归零但整份保留 infos)
            # 会打破它 —— 那时状态是"上一世"的,拿来续课程等于劫持一次刻意的全新热启动。
            # 响亮降级成"老档"语义:主时钟按本档 iter 推算,课程统计从头攒。
            expected_next = int(self.current_learning_iteration) + 1
            if int(state.get("next_learning_iteration", -1)) != expected_next:
                print(
                    "[MotionOnPolicyRunner] WARNING: hope_exact_resume_state is stale "
                    f"(next_learning_iteration={state.get('next_learning_iteration')!r} vs "
                    f"checkpoint iter+1={expected_next}); treating as a legacy warm-start "
                    "checkpoint — curriculum statistics start fresh",
                    flush=True,
                )
                state = None
        if state is None:
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
        try:
            super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self._boundary_stop_requested = None
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width=width, pad=pad)
        step = int(locs["it"])
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
