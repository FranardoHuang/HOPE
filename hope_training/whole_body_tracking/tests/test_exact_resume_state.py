"""Host-only tests for the exact-resume state package (jiayi 9f684ae5 ported to main).

人话:环境的 common_step_counter 驱动所有随步数渐进的课程,但 base rsl_rl 的存档不含它,
续训时全部课程静默回到第 0 步。这里在不起 Isaac 的前提下,用 stub env/runner 验证:
  1. save→load round-trip:课程主时钟 + 各命令项课程状态(EMA 标量、post-swing 环、
     HER 已实现回放环)完整恢复,迭代号跳到 N+1,恢复后 reset 一次;
  2. 老 checkpoint(无状态包)按迭代号精确推算主时钟,而不是回零;
  3. 未知键/未知命令项/未知段一律容忍,认得的照常恢复;
  4. 评测器路径(log_dir=None)保持移植前行为,一个字节都不动环境;
  5. Ctrl-C 等当前 PPO 迭代完整结束才存档退出,存出来的档带状态包。
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "source/whole_body_tracking/whole_body_tracking/utils"


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "training_contract_exact_resume_under_test", UTILS / "training_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeOnPolicyRunner:
    """与真 rsl_rl OnPolicyRunner 的 save/load/learn 契约对齐的最小基类。

    save 真写 torch.save(model/optimizer/iter/infos),load 真读 —— round-trip 必须过真文件,
    否则测不出"状态是否真进了 PT"。learn 模拟 base 的 rollout→update→log 循环,让子类的
    Ctrl-C 边界逻辑有处可挂。
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = dict(train_cfg or {})
        self.log_dir = log_dir
        self.device = device
        self.logger_type = str(self.cfg.get("logger", "tensorboard"))
        self.writer = None
        self.disable_logs = True
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.num_steps_per_env = int(self.cfg.get("num_steps_per_env", 24))
        self.alg = SimpleNamespace(
            learning_rate=1.0e-3,
            schedule="adaptive",
            optimizer=SimpleNamespace(param_groups=[{"lr": 1.0e-3}]),
        )
        self.empirical_normalization = False
        self.obs_normalizer = None

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "iter": int(self.current_learning_iteration),
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True, **kwargs):
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        self.current_learning_iteration = int(loaded["iter"])
        return loaded["infos"]

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        start = int(self.current_learning_iteration)
        self.ran_iterations = []
        for it in range(start, start + int(num_learning_iterations)):
            self.current_learning_iteration = it
            hook = getattr(self, "_test_mid_iteration_hook", None)
            if hook is not None:
                hook(it)  # 模拟 rollout 进行中(Ctrl-C 会在这里到达)
            self.ran_iterations.append(it)
            self.log({"it": it})

    def log(self, locs, width=80, pad=35):
        return None


def _load_runner_module(monkeypatch, contract_module):
    fake_rsl_rl = _module("rsl_rl")
    fake_rsl_rl.__path__ = []
    fake_runners = _module("rsl_rl.runners")
    fake_runners.__path__ = []
    fake_isaaclab_rl = _module("isaaclab_rl")
    fake_isaaclab_rl.__path__ = []
    fake_wbt = _module("whole_body_tracking")
    fake_wbt.__path__ = []
    fake_utils = _module("whole_body_tracking.utils")
    fake_utils.__path__ = []
    modules = {
        "rsl_rl": fake_rsl_rl,
        "rsl_rl.env": _module("rsl_rl.env", VecEnv=type("VecEnv", (), {})),
        "rsl_rl.runners": fake_runners,
        "rsl_rl.runners.on_policy_runner": _module(
            "rsl_rl.runners.on_policy_runner", OnPolicyRunner=FakeOnPolicyRunner
        ),
        "isaaclab_rl": fake_isaaclab_rl,
        "isaaclab_rl.rsl_rl": _module(
            "isaaclab_rl.rsl_rl", export_policy_as_onnx=lambda *args, **kwargs: None
        ),
        "whole_body_tracking": fake_wbt,
        "whole_body_tracking.utils": fake_utils,
        "whole_body_tracking.utils.exporter": _module(
            "whole_body_tracking.utils.exporter",
            attach_onnx_metadata=lambda *args, **kwargs: None,
            export_motion_policy_as_onnx=lambda *args, **kwargs: False,
            is_empirical_normalizer=lambda value: False,
        ),
        "whole_body_tracking.utils.training_contract": contract_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "motion_runner_exact_resume_under_test", UTILS / "my_on_policy_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner_module(monkeypatch):
    return _load_runner_module(monkeypatch, _load_contract_module())


# ---------------------------------------------------------------------------
# Stub env:与真 ManagerBasedRLEnv/RslRlVecEnvWrapper 的最小接口对齐
# ---------------------------------------------------------------------------


class FakeCommandManager:
    def __init__(self, terms):
        self._terms = dict(terms)

    @property
    def active_terms(self):
        return list(self._terms)

    def get_term(self, name):
        return self._terms[name]  # 未知项抛 KeyError(与 Isaac 的 dict 语义一致)


class FakeVecEnv:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return None, {}


def _motion_term(filled: bool):
    """MotionCommand 的课程宿主子集:自适应失败分箱 + post-swing 收势环形缓冲。"""
    term = SimpleNamespace(device="cpu", metrics={})
    if filled:
        term.bin_failed_count = torch.tensor([3.0, 1.0, 0.5, 2.0])
        term._current_bin_failed = torch.tensor([1.0, 0.0, 0.0, 1.0])
        term._post_swing_root = torch.arange(26, dtype=torch.float32).reshape(2, 13)
        term._post_swing_joint_pos = torch.full((2, 5), 0.25)
        term._post_swing_joint_vel = torch.full((2, 5), -0.5)
        term._post_swing_ptr = 1
        term._post_swing_count = 2
    else:
        term.bin_failed_count = torch.zeros(4)
        term._current_bin_failed = torch.zeros(4)
        term._post_swing_root = None  # 真实类是懒初始化:恢复必须能从 None 起
        term._post_swing_joint_pos = None
        term._post_swing_joint_vel = None
        term._post_swing_ptr = 0
        term._post_swing_count = 0
    # 每 env 瞬态,绝不该入档
    term.time_steps = torch.zeros(4, dtype=torch.long)
    return term


def _racket_term(filled: bool):
    """RacketTargetCommand 的课程宿主子集:EMA 标量、成功门控扩幅、自适应 sigma、HER 回放环。"""
    term = SimpleNamespace(device="cpu", metrics={})
    if filled:
        term._curr_perturb_scale = 0.65
        term._adaptive_sigma_pos = 0.09
        term._adaptive_sigma_vel = 0.8
        term._exact_n_acc = 321.5
        term._exact_pass_comp_acc = 123.25
        term._vb_inb_acc_c = {0: 11.0, 1: 7.5}
        term._exact_pos_err_sum = 14.5
        term._exact_pos_err_sum_c = {0: 9.0, 1: 5.5}
        term._ach_fill = {0: 4, 1: 2}
        term._ach_ptr = {0: 0, 1: 2}
        term._ach_pos = {0: torch.ones(4, 3) * 0.1, 1: torch.ones(4, 3) * 0.2}
        term._ach_vel = {0: torch.ones(4, 3) * 1.1, 1: torch.ones(4, 3) * 2.2}
        term._ach_spd = {0: torch.ones(4) * 0.9, 1: torch.ones(4) * 1.3}
    else:
        term._curr_perturb_scale = 0.05
        term._adaptive_sigma_pos = 0.25
        term._adaptive_sigma_vel = 1.8
        term._exact_n_acc = 0.0
        term._exact_pass_comp_acc = 0.0
        term._vb_inb_acc_c = {0: 0.0, 1: 0.0}
        term._exact_pos_err_sum = 0.0
        term._exact_pos_err_sum_c = {0: 0.0, 1: 0.0}
        term._ach_fill = {0: 0, 1: 0}
        term._ach_ptr = {0: 0, 1: 0}
        term._ach_pos = {0: torch.zeros(4, 3), 1: torch.zeros(4, 3)}
        term._ach_vel = {0: torch.zeros(4, 3), 1: torch.zeros(4, 3)}
        term._ach_spd = {0: torch.zeros(4), 1: torch.zeros(4)}
    # 入档黑名单探针:cfg 常数、每 env 瞬态张量、名字撞后缀但值是张量的属性
    term._vb_net_x = 1.37
    term._prev_racket_dist = torch.zeros(4)
    term._weird_acc = torch.zeros(4)
    return term


def _make_runner(runner_module, *, log_dir, filled, iteration=0, counter=0):
    terms = {
        "motion": _motion_term(filled),
        "racket_target": _racket_term(filled),
    }
    inner = SimpleNamespace(
        common_step_counter=counter,
        command_manager=FakeCommandManager(terms),
        num_envs=4,
    )
    wrapper = FakeVecEnv(inner)
    runner = runner_module.MotionOnPolicyRunner(
        wrapper,
        {"num_steps_per_env": 24, "max_iterations": 25000, "logger": "tensorboard"},
        log_dir=log_dir,
        device="cpu",
    )
    runner.current_learning_iteration = iteration
    return runner, inner, wrapper, terms


# ---------------------------------------------------------------------------
# 1) round-trip:save 打包 → load 恢复
# ---------------------------------------------------------------------------


def test_save_embeds_state_and_load_round_trips(runner_module, tmp_path):
    save_dir = tmp_path / "run_a"
    save_dir.mkdir()
    runner, inner, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=137, counter=3288
    )
    runner.tot_timesteps = 137 * 24 * 4
    runner.tot_time = 456.75
    runner.alg.learning_rate = 3.3e-4
    checkpoint = save_dir / "model_137.pt"
    runner.save(str(checkpoint))

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded["infos"]["hope_exact_resume_state"]
    assert state["schema_version"] == 3
    assert state["next_learning_iteration"] == 138
    assert state["target_learning_iterations"] == 25000
    assert state["tot_timesteps"] == 137 * 24 * 4
    assert state["algorithm_learning_rate"] == pytest.approx(3.3e-4)
    env_state = state["environment_resume_state"]
    assert env_state["common_step_counter"] == 3288
    racket_saved = env_state["command_terms"]["racket_target"]
    assert racket_saved["scalars"]["_curr_perturb_scale"] == pytest.approx(0.65)
    assert racket_saved["scalars"]["_vb_inb_acc_c"] == {0: 11.0, 1: 7.5}
    assert racket_saved["scalars"]["_exact_pos_err_sum"] == pytest.approx(14.5)
    assert racket_saved["scalars"]["_ach_fill"] == {0: 4, 1: 2}
    assert torch.equal(racket_saved["tensor_dicts"]["_ach_pos"][1], torch.ones(4, 3) * 0.2)
    motion_saved = env_state["command_terms"]["motion"]
    assert torch.equal(motion_saved["tensors"]["bin_failed_count"], torch.tensor([3.0, 1.0, 0.5, 2.0]))
    assert motion_saved["scalars"]["_post_swing_ptr"] == 1
    # 入档黑名单:cfg 常数/每 env 瞬态/张量伪标量都不许进
    assert "_vb_net_x" not in racket_saved["scalars"]
    assert "_weird_acc" not in racket_saved["scalars"]
    assert "_prev_racket_dist" not in racket_saved["tensors"]
    assert "time_steps" not in motion_saved["tensors"]

    # 全新进程语义:另起 runner/env(全零课程),从档里恢复
    resume_dir = tmp_path / "run_b"
    resume_dir.mkdir()
    fresh, fresh_inner, fresh_wrapper, fresh_terms = _make_runner(
        runner_module, log_dir=str(resume_dir), filled=False
    )
    fresh.load(str(checkpoint))

    assert fresh.current_learning_iteration == 138  # N+1:不重复第 137 个更新
    assert fresh_inner.common_step_counter == 3288
    assert fresh_wrapper.reset_calls == 1  # 恢复后 reset 一次,rollout 按恢复分布采样
    assert fresh.tot_timesteps == 137 * 24 * 4
    assert fresh.tot_time == pytest.approx(456.75)
    assert fresh.alg.learning_rate == pytest.approx(3.3e-4)  # adaptive 调度:lr 续上
    assert fresh.alg.optimizer.param_groups[0]["lr"] == pytest.approx(3.3e-4)
    racket = fresh_terms["racket_target"]
    assert racket._curr_perturb_scale == pytest.approx(0.65)
    assert racket._adaptive_sigma_pos == pytest.approx(0.09)
    assert racket._exact_n_acc == pytest.approx(321.5)
    assert racket._vb_inb_acc_c == {0: 11.0, 1: 7.5}
    assert racket._exact_pos_err_sum_c == {0: 9.0, 1: 5.5}
    assert racket._ach_fill == {0: 4, 1: 2}
    assert torch.equal(racket._ach_pos[0], torch.ones(4, 3) * 0.1)
    assert torch.equal(racket._ach_spd[1], torch.ones(4) * 1.3)
    motion = fresh_terms["motion"]
    assert torch.equal(motion.bin_failed_count, torch.tensor([3.0, 1.0, 0.5, 2.0]))
    assert torch.equal(motion._post_swing_root, torch.arange(26, dtype=torch.float32).reshape(2, 13))
    assert motion._post_swing_count == 2
    # 瞬态未被档里的值污染
    assert racket._vb_net_x == pytest.approx(1.37)


def test_fixed_lr_schedule_keeps_config_lr(runner_module, tmp_path):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=10, counter=240
    )
    runner.alg.learning_rate = 9.9e-4
    checkpoint = save_dir / "model_10.pt"
    runner.save(str(checkpoint))

    fresh, _, _, _ = _make_runner(runner_module, log_dir=str(save_dir), filled=False)
    fresh.alg.schedule = "fixed"
    fresh.load(str(checkpoint))
    # 固定调度:YAML 里改 lr 再续训必须生效,不吃档里的旧 lr
    assert fresh.alg.learning_rate == pytest.approx(1.0e-3)
    assert fresh.alg.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)


# ---------------------------------------------------------------------------
# 2) 老 checkpoint:无状态包,按迭代号精确推算主时钟
# ---------------------------------------------------------------------------


def test_legacy_checkpoint_derives_counter_from_iteration(runner_module, tmp_path):
    checkpoint = tmp_path / "model_507.pt"
    torch.save(
        {"model_state_dict": {}, "optimizer_state_dict": {}, "iter": 507, "infos": None},
        checkpoint,
    )
    fresh, inner, wrapper, terms = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=False
    )
    fresh.load(str(checkpoint))
    # base 存档发生在完成第 507 个迭代之后:主时钟 = (507+1) * num_steps_per_env
    assert inner.common_step_counter == 508 * 24
    # 迭代号维持 base 语义(没有状态包时不敢替它做 N+1 的决定)
    assert fresh.current_learning_iteration == 507
    assert wrapper.reset_calls == 1
    # 课程细节无从恢复:保持新鲜初始化
    assert terms["racket_target"]._curr_perturb_scale == pytest.approx(0.05)


def test_hitterobs_top_level_state_key_is_also_accepted(runner_module, tmp_path):
    """跨栈对账:jiayi/hitterobs 把状态放 PT 顶层而不是 infos 里,也要认。"""
    checkpoint = tmp_path / "model_88.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 88,
            "infos": None,
            "hope_exact_resume_state": {
                "schema_version": 1,  # hitterobs v1:还没有 environment_resume_state
                "next_learning_iteration": 89,
                "tot_timesteps": 88 * 24 * 4,
                "tot_time": 12.0,
            },
        },
        checkpoint,
    )
    fresh, inner, _, _ = _make_runner(runner_module, log_dir=str(tmp_path), filled=False)
    fresh.load(str(checkpoint))
    assert fresh.current_learning_iteration == 89
    # v1 没有环境状态段 → 主时钟按 next_learning_iteration 推算
    assert inner.common_step_counter == 89 * 24


# ---------------------------------------------------------------------------
# 3) 未知键/未知项容忍;未知 schema fail-loud
# ---------------------------------------------------------------------------


def test_unknown_keys_terms_and_sections_are_tolerated(runner_module, tmp_path):
    checkpoint = tmp_path / "model_11.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 11,
            "infos": {
                "hope_exact_resume_state": {
                    "schema_version": 2,
                    "next_learning_iteration": 12,
                    "future_top_level_field": {"nested": True},
                    "environment_resume_state": {
                        "schema_version": 2,
                        "common_step_counter": 777,
                        "future_section": [1, 2, 3],
                        "command_terms": {
                            "ghost_term": {"scalars": {"_ghost_acc": 1.0}},
                            "racket_target": {
                                "scalars": {
                                    "_curr_perturb_scale": 0.44,
                                    "_attr_this_build_lacks_acc": 9.9,
                                },
                                "tensors": {"_tensor_this_build_lacks": torch.zeros(3)},
                                "tensor_dicts": {
                                    "_ach_pos": {
                                        0: torch.full((4, 3), 0.7),
                                        99: torch.full((4, 3), -1.0),  # 档里多出的 clip 键
                                    },
                                    "_dict_this_build_lacks": {0: torch.zeros(2)},
                                },
                                "future_subsection": {"z": 1},
                            },
                        },
                    },
                }
            },
        },
        checkpoint,
    )
    fresh, inner, _, terms = _make_runner(runner_module, log_dir=str(tmp_path), filled=False)
    fresh.load(str(checkpoint))
    assert fresh.current_learning_iteration == 12
    assert inner.common_step_counter == 777
    racket = terms["racket_target"]
    assert racket._curr_perturb_scale == pytest.approx(0.44)
    assert not hasattr(racket, "_attr_this_build_lacks_acc")
    assert not hasattr(racket, "_tensor_this_build_lacks")
    assert torch.equal(racket._ach_pos[0], torch.full((4, 3), 0.7))
    assert 99 not in racket._ach_pos  # 多出的 clip 键被容忍丢弃,不塞进当前配置
    assert not hasattr(racket, "_dict_this_build_lacks")


def test_iter_surgery_invalidates_stale_state_and_falls_back(runner_module, tmp_path, capsys):
    """make_hitter_warmstart / warm_start_realsensor 把 iter 归零但保留 infos:状态包与 iter
    失配时必须响亮降级成'老档'语义,不许劫持刻意的全新热启动。"""
    save_dir = tmp_path / "donor"
    save_dir.mkdir()
    donor, _, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=5000, counter=5001 * 24
    )
    checkpoint = save_dir / "model_5000.pt"
    donor.save(str(checkpoint))
    # 模拟外科手术:iter 归零、丢优化器,infos(含状态包)整份保留
    surgical = torch.load(checkpoint, map_location="cpu", weights_only=False)
    surgical.pop("optimizer_state_dict", None)
    surgical["iter"] = 0
    warmstart = tmp_path / "warmstart.pt"
    torch.save(surgical, warmstart)

    fresh, inner, wrapper, terms = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=False
    )
    fresh.load(str(warmstart))
    assert "hope_exact_resume_state is stale" in capsys.readouterr().out
    assert fresh.current_learning_iteration == 0  # 手术意图(全新热启动)得到尊重
    assert inner.common_step_counter == 1 * 24  # 主时钟按本档 iter 推算,而不是上一世的 5001*24
    assert terms["racket_target"]._curr_perturb_scale == pytest.approx(0.05)  # 课程从头攒
    assert wrapper.reset_calls == 1


def test_unknown_schema_version_fails_loud(runner_module, tmp_path):
    checkpoint = tmp_path / "model_5.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 5,
            "infos": {"hope_exact_resume_state": {"schema_version": 99}},
        },
        checkpoint,
    )
    fresh, _, _, _ = _make_runner(runner_module, log_dir=str(tmp_path), filled=False)
    with pytest.raises(RuntimeError, match="hope_exact_resume_state schema"):
        fresh.load(str(checkpoint))


# ---------------------------------------------------------------------------
# 4) 评测器路径(log_dir=None)行为逐字节不变
# ---------------------------------------------------------------------------


def test_eval_runner_without_log_dir_keeps_legacy_load_semantics(runner_module, tmp_path):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    trainer, _, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=42, counter=1032
    )
    checkpoint = save_dir / "model_42.pt"
    trainer.save(str(checkpoint))

    evaluator, inner, wrapper, terms = _make_runner(
        runner_module, log_dir=None, filled=False
    )
    evaluator.load(str(checkpoint))
    # isaac_bank_exam / play 语义:不动环境、不 reset、迭代号维持 base load 的值
    assert inner.common_step_counter == 0
    assert wrapper.reset_calls == 0
    assert evaluator.current_learning_iteration == 42
    assert terms["racket_target"]._curr_perturb_scale == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 5) Ctrl-C:等当前迭代完整结束才存档退出
# ---------------------------------------------------------------------------


def test_sigint_defers_to_iteration_boundary_and_saves_resumable_pt(runner_module, tmp_path):
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=True, iteration=0, counter=24
    )
    original_handler = signal.getsignal(signal.SIGINT)

    def _interrupt_mid_iteration(it):
        if it == 0:
            os.kill(os.getpid(), signal.SIGINT)  # rollout 中途按下 Ctrl-C

    runner._test_mid_iteration_hook = _interrupt_mid_iteration
    with pytest.raises(KeyboardInterrupt):
        runner.learn(num_learning_iterations=3)

    # 第 0 个迭代完整跑完并存档;第 1、2 个不再运行
    assert runner.ran_iterations == [0]
    boundary_checkpoint = tmp_path / "model_0.pt"
    assert boundary_checkpoint.is_file()
    loaded = torch.load(boundary_checkpoint, map_location="cpu", weights_only=False)
    state = loaded["infos"]["hope_exact_resume_state"]
    assert state["next_learning_iteration"] == 1
    assert state["environment_resume_state"]["common_step_counter"] == 24
    # 信号处理器恢复原状,进程退出行为不被劫持
    assert signal.getsignal(signal.SIGINT) is original_handler


def test_sigterm_is_also_deferred(runner_module, tmp_path):
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=True, iteration=7, counter=8 * 24
    )
    original_handler = signal.getsignal(signal.SIGTERM)

    def _terminate_mid_iteration(it):
        if it == 7:
            os.kill(os.getpid(), signal.SIGTERM)

    runner._test_mid_iteration_hook = _terminate_mid_iteration
    with pytest.raises(KeyboardInterrupt):
        runner.learn(num_learning_iterations=5)
    assert runner.ran_iterations == [7]
    assert (tmp_path / "model_7.pt").is_file()
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_second_sigint_escalates_to_immediate_default_action(
    runner_module, tmp_path, monkeypatch, capsys
):
    """第二次 Ctrl-C = 立即死(2026-07-25):恢复 SIG_DFL 并对自己重投该信号,不再静默吞掉。

    真升级会当场杀掉测试进程,所以把 os.kill 换成记录桩(monkeypatch 会还原);投递前两次
    信号用真 kill 的保存引用。断言:升级时点上 SIGINT 的处置已经是 SIG_DFL(即使桩没真杀,
    内核默认动作已就位),且提示了"不存档"。
    """
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=True, iteration=0, counter=24
    )
    real_kill = os.kill
    escalations = []

    def _recording_kill(pid, signum):
        escalations.append((pid, signum, signal.getsignal(signal.SIGINT)))

    def _double_interrupt(it):
        if it == 0:
            real_kill(os.getpid(), signal.SIGINT)  # 第一次:登记"想停"
            monkeypatch.setattr(os, "kill", _recording_kill)
            real_kill(os.getpid(), signal.SIGINT)  # 第二次:应升级为立即终止

    runner._test_mid_iteration_hook = _double_interrupt
    with pytest.raises(KeyboardInterrupt):  # 桩没真杀,流程走到边界照旧收尾
        runner.learn(num_learning_iterations=3)

    assert escalations == [(os.getpid(), signal.SIGINT, signal.SIG_DFL)]
    out = capsys.readouterr().out
    assert "second interrupt" in out and "NO checkpoint" in out
