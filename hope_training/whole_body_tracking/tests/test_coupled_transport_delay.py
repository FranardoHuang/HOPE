"""planner 修订 × 目标延迟 = 耦合传输(2026-07-25 A1 JOB2)— host-only 守卫测试。

人话背景:旧代码在 planner_revision_enabled 且 target_delay_steps>0 时直接 NO-LAUNCH
(旧延迟环只延迟 actor 观测,而相位调度器即时消费修订——两边看到两条流,矛盾)。
修复:延迟改发生在"提交"侧——生成于 t 的修订元组(pos/vel/normal/tts,原子)进在途环,
t+d 才提交给调度器。接受记账 / 调度器 desired_tts / actor 可见元组从此消费同一条延迟流
(mocap→relay 语义);actor 端不再叠观测延迟环(否则总延迟 2d)。BEGIN(任务安装)不是
mocap 流,保持即时,且安装新任务时作废旧球在途修订。

本文件钉死的合同:
1. d=0 逐字节不变:耦合模式不激活,生成即提交(现役行为);
2. d=2 恰好晚 2 步:t 步生成的元组内容(含跨步不变量)在 t+2 提交;
3. tts 随元组走:source_timestamp_compensated 提交时减 d*dt(actor 时钟连续),
   uncompensated 原样提交(显式陈旧阴性对照);
4. 调度器/actor 同拍:接受当拍 planner_visible(actor 数据源)= 提交给调度器的同一元组;
5. 守卫换防:修订+延迟不再 NO-LAUNCH,但 'live' tts 模式、event timing 组合仍 fail-loud;
6. 任务安装即时 + 作废在途:BEGIN 不过环,旧球元组不得提交进新任务。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_coupled_transport_delay.py -q
"""

from __future__ import annotations

import inspect
import math
import types

import numpy as np
import pytest
import torch

# py3.8 host 没有 math.ulp(这也是 test_reward_flags_mdp 里两个已知豁免失败的根因)。
# PhaseGovernorProfile 的构造校验用到它;这里补一个数值等价的垫片,只影响测试进程。
if not hasattr(math, "ulp"):
    def _ulp(x: float) -> float:
        x = abs(float(x))
        return float(np.nextafter(np.float64(x), np.inf) - np.float64(x))

    math.ulp = _ulp

from test_reward_flags_mdp import commands_mod, hope_commands_mod, planner_revision_mod


DT = 0.02


def _profile():
    return planner_revision_mod.PhaseGovernorProfile(
        policy_dt_s=DT,
        min_tts_s=0.10,
        max_tts_s=2.00,
        max_phase_rate_per_s=4.0,
        max_phase_acceleration_per_s2=20.0,
        max_deadline_revision_delta_s=0.25,
        max_position_revision_delta_m=0.10,
        max_velocity_revision_delta_mps=0.50,
        max_normal_revision_delta_rad=0.20,
    )


class _FakeMotion:
    """相位调度器仿体:记录每次 submit 的完整元组,接受掩码可编程。"""

    def __init__(self, n, tts0=0.8):
        self._planner_active = torch.zeros(n, dtype=torch.bool)
        self._planner_active[0] = True
        self._planner_truth_tts = torch.zeros(n)
        self._planner_truth_tts[0] = tts0
        self.event_timing_enabled = False
        self.submissions = []
        self.accept_next = True

    # 真代码里是 MotionCommand 的 staticmethod;仿体直接借用同一实现,数值口径一致。
    _planner_canonicalize_tts = staticmethod(
        commands_mod.MotionCommand._planner_canonicalize_tts
    )

    def submit_planner_revision(self, ids, **kwargs):
        self.submissions.append((ids.clone(), {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }))
        return torch.full((len(ids),), bool(self.accept_next), dtype=torch.bool)

    def advance(self):
        active = self._planner_active
        self._planner_truth_tts[active] = (self._planner_truth_tts[active] - DT).clamp(min=0.0)


def _coupled_cmd(n=2, delay=2, tts_mode="source_timestamp_compensated", tts0=0.8):
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.num_envs = n
    rt.device = "cpu"
    rt.planner_revision_enabled = True
    rt._planner_revision_profile = _profile()
    rt.cfg = types.SimpleNamespace(
        planner_revision_position_std_m=0.0,
        planner_revision_velocity_std_mps=0.0,
        planner_revision_normal_std_rad=0.0,
        planner_revision_tts_std_s=0.0,
    )
    rt._env = types.SimpleNamespace(step_dt=DT)
    motion = _FakeMotion(n, tts0=tts0)
    rt._motion = lambda: motion
    rt._coupled_transport = delay > 0
    rt._delay_steps = delay
    rt._delay_tts_mode = tts_mode
    if delay > 0:
        rt._pend_ptr = 0
        rt._pend_valid = torch.zeros(delay, n, dtype=torch.bool)
        rt._pend_pos = torch.zeros(delay, n, 3)
        rt._pend_vel = torch.zeros(delay, n, 3)
        rt._pend_normal = torch.zeros(delay, n, 3)
        rt._pend_tts = torch.zeros(delay, n)
    rt.racket_target_pos_w = torch.tensor([[0.4, -0.2, 0.9]] * n)
    rt.racket_target_vel_w = torch.tensor([[2.0, 0.0, 0.5]] * n)
    rt.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]] * n)
    rt._planner_control_epoch = torch.ones(n, dtype=torch.long)
    rt._planner_task_id = torch.ones(n, dtype=torch.long)
    rt._planner_task_revision = torch.ones(n, dtype=torch.long)
    rt._planner_visible_pos = rt.racket_target_pos_w.clone()
    rt._planner_visible_vel = rt.racket_target_vel_w.clone()
    rt._planner_visible_normal = rt.target_normal_cmd.clone()
    rt._planner_visible_tts = torch.full((n,), tts0)
    rt._planner_visible_last_precontact = torch.zeros(n, dtype=torch.bool)
    rt.metrics = {
        "planner_task_revision": torch.zeros(n),
        "planner_same_task_revision_active": torch.zeros(n),
    }
    return rt, motion


def _step(rt, motion):
    """一个控制步:先走调度器时钟(_advance_planner_phase 的仿真),再跑修订工序。"""
    motion.advance()
    rt._revise_same_ball_actor_tuple()


# --------------------------------------------------------------------------------------------- #
# 1. d=0 逐字节不变(legacy 生成即提交)
# --------------------------------------------------------------------------------------------- #
def test_d0_submits_generated_tuple_same_tick():
    rt, motion = _coupled_cmd(delay=0)
    assert rt._coupled_transport is False
    _step(rt, motion)
    # 同拍提交 + 同拍接受生效:revision 立即 2,visible tts = 本步 truth
    assert len(motion.submissions) == 1
    ids, kw = motion.submissions[0]
    assert ids.tolist() == [0]
    assert kw["desired_tts"][0].item() == pytest.approx(0.78)
    assert rt._planner_task_revision[0].item() == 2
    assert rt._planner_visible_tts[0].item() == pytest.approx(0.78)


def test_d0_mode_detection_is_structurally_off():
    # 耦合模式的唯一激活条件 = 修订 + d>0;d=0 走现役路径(结构性逐字节不变)
    assert hope_commands_mod._coupled_transport_mode(types.SimpleNamespace(
        planner_revision_enabled=True, target_delay_steps=0,
        target_delay_tts_mode="source_timestamp_compensated",
    )) is False
    assert hope_commands_mod._coupled_transport_mode(types.SimpleNamespace(
        planner_revision_enabled=False, target_delay_steps=3,
        target_delay_tts_mode="live",
    )) is False


# --------------------------------------------------------------------------------------------- #
# 2+3. d=2 恰好晚 2 步;tts 随元组 + 源时间戳补偿
# --------------------------------------------------------------------------------------------- #
def test_d2_shifts_revision_effect_by_exactly_two_steps():
    rt, motion = _coupled_cmd(delay=2)
    pos_t1 = rt.racket_target_pos_w[0].clone()

    _step(rt, motion)  # t1: 生成(truth 0.78)入环,无可提交
    assert motion.submissions == []
    assert rt._planner_task_revision[0].item() == 1  # 修订未生效

    # t1 之后改变生成源(证明提交的是 t1 时刻的元组内容,不是提交时刻的)
    rt.racket_target_pos_w = torch.tensor([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])
    _step(rt, motion)  # t2: 仍无可提交
    assert motion.submissions == []
    assert rt._planner_task_revision[0].item() == 1

    _step(rt, motion)  # t3: 提交 t1 生成的元组
    assert len(motion.submissions) == 1
    ids, kw = motion.submissions[0]
    assert ids.tolist() == [0]
    assert torch.allclose(kw["target_position"][0], pos_t1)  # t1 的内容,晚整整 2 步
    # 源时间戳补偿:t1 生成的 tts(0.78)减在途 2*dt = 0.74 = 提交时刻的真实剩余时间
    assert kw["desired_tts"][0].item() == pytest.approx(0.74)
    assert rt._planner_task_revision[0].item() == 2

    _step(rt, motion)  # t4: 提交 t2 生成的元组(新内容跟上)
    assert len(motion.submissions) == 2
    _, kw2 = motion.submissions[1]
    assert torch.allclose(kw2["target_position"][0], torch.tensor([9.0, 9.0, 9.0]))
    assert kw2["desired_tts"][0].item() == pytest.approx(0.72)


def test_uncompensated_submits_stale_tts_as_negative_control():
    rt, motion = _coupled_cmd(delay=2, tts_mode="uncompensated")
    _step(rt, motion)
    _step(rt, motion)
    _step(rt, motion)
    assert len(motion.submissions) == 1
    _, kw = motion.submissions[0]
    # 阴性对照:t1 生成的 0.78 原样提交(比提交时刻真实剩余时间陈旧 2*dt)
    assert kw["desired_tts"][0].item() == pytest.approx(0.78)


# --------------------------------------------------------------------------------------------- #
# 4. 调度器 / actor 同拍消费同一条延迟流
# --------------------------------------------------------------------------------------------- #
def test_governor_and_actor_see_the_same_tuple_on_the_same_tick():
    rt, motion = _coupled_cmd(delay=2)
    _step(rt, motion)
    _step(rt, motion)
    # 接受生效前,actor 数据源(planner_visible)保持 BEGIN 元组 + 本地时钟递减
    assert rt._planner_visible_tts[0].item() == pytest.approx(0.76)
    _step(rt, motion)
    _, kw = motion.submissions[0]
    # 接受当拍:actor 数据源 == 提交给调度器的同一元组(同拍、同值)
    assert rt._planner_visible_tts[0].item() == pytest.approx(kw["desired_tts"][0].item())
    assert torch.allclose(rt._planner_visible_pos[0], kw["target_position"][0])
    assert torch.allclose(rt._planner_visible_normal[0], kw["target_normal"][0])
    # actor 时钟连续:接受把 visible tts 从本地递减值 0.74 换成补偿后的 0.74(无跳变)
    assert rt._planner_visible_tts[0].item() == pytest.approx(0.74)


def test_rejected_submission_leaves_visible_stream_and_revision_untouched():
    rt, motion = _coupled_cmd(delay=2)
    motion.accept_next = False
    for _ in range(3):
        _step(rt, motion)
    assert len(motion.submissions) == 1  # 提交发生(记账走 fail-safe 拒绝)
    assert rt._planner_task_revision[0].item() == 1
    assert rt._planner_visible_tts[0].item() == pytest.approx(0.74)  # 只有本地递减
    assert torch.allclose(rt._planner_visible_pos[0], torch.tensor([0.4, -0.2, 0.9]))


def test_actor_side_observation_ring_is_disabled_in_coupled_mode():
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand.__init__)
    # 耦合模式 actor 环步数强制 0(延迟由提交侧承担,不许叠成 2d)
    assert "self._actor_ring_steps = 0 if self._coupled_transport else self._delay_steps" in src
    push = inspect.getsource(hope_commands_mod.RacketTargetCommand._push_actor_target)
    assert "if self._actor_ring_steps > 0:" in push
    reset = inspect.getsource(hope_commands_mod.RacketTargetCommand._reset_actor_target_state)
    assert "if self._actor_ring_steps > 0:" in reset


# --------------------------------------------------------------------------------------------- #
# 5. 守卫换防:接受修订+延迟,仍拒真正不自洽的组合
# --------------------------------------------------------------------------------------------- #
def test_guard_accepts_coupled_combo_and_rejects_live_tts():
    ok = hope_commands_mod._coupled_transport_mode(types.SimpleNamespace(
        planner_revision_enabled=True, target_delay_steps=2,
        target_delay_tts_mode="source_timestamp_compensated",
    ))
    assert ok is True
    ok2 = hope_commands_mod._coupled_transport_mode(types.SimpleNamespace(
        planner_revision_enabled=True, target_delay_steps=2,
        target_delay_tts_mode="uncompensated",
    ))
    assert ok2 is True
    # 'live':元组晚到、时钟即时 = 矛盾的传输语义,fail-loud
    with pytest.raises(ValueError, match="coupled transport tuple"):
        hope_commands_mod._coupled_transport_mode(types.SimpleNamespace(
            planner_revision_enabled=True, target_delay_steps=2,
            target_delay_tts_mode="live",
        ))
    # 旧 NO-LAUNCH 守卫必须已经拆除
    init_src = inspect.getsource(hope_commands_mod.RacketTargetCommand.__init__)
    assert "are not launchable" not in init_src
    assert "self._coupled_transport = _coupled_transport_mode(cfg)" in init_src


def test_event_timing_plus_coupled_delay_fails_loud():
    rt, motion = _coupled_cmd(delay=2)
    motion.event_timing_enabled = True
    with pytest.raises(RuntimeError, match="coupled transport delay \\+ event timing"):
        rt._revise_same_ball_actor_tuple()


# --------------------------------------------------------------------------------------------- #
# 6. 任务安装即时 + 作废旧球在途修订
# --------------------------------------------------------------------------------------------- #
def test_task_begin_flush_drops_stale_inflight_revisions():
    rt, motion = _coupled_cmd(delay=2)
    _step(rt, motion)  # t1 生成入环
    # 仿真 BEGIN(_begin_same_ball_planner_task 的作废工序):旧球在途修订全部无效
    rt._pend_valid[:, torch.tensor([0])] = False
    _step(rt, motion)
    _step(rt, motion)  # 原本 t3 会提交 t1 的元组
    assert all(len(ids) == 0 or 0 not in ids.tolist() for ids, _ in motion.submissions) \
        or len(motion.submissions) == 0
    # t1 的元组被作废;t2 生成的元组按期在 t4 提交(环本身不因作废而错位)
    _step(rt, motion)
    assert len(motion.submissions) == 1
    _, kw = motion.submissions[0]
    assert kw["desired_tts"][0].item() == pytest.approx(0.72)  # t2 生成 0.76 - 2*dt
    # 源码守卫:BEGIN 处必须作废在途(中继丢弃已结束任务的消息),且 BEGIN 元组不过环
    begin = inspect.getsource(
        hope_commands_mod.RacketTargetCommand._begin_same_ball_planner_task
    )
    assert "self._pend_valid[:, ids] = False" in begin
    assert "_exchange_pending_planner_revision" not in begin


def test_no_generation_step_still_pops_the_ring():
    # 调度器时钟走进 no-new-revision 截止区后(不再生成),已在途的修订仍必须按期提交
    rt, motion = _coupled_cmd(delay=2, tts0=0.15)
    _step(rt, motion)  # t1: truth 0.13 >= min_tts 0.10,生成入环
    assert rt._pend_valid[0, 0].item() is True or rt._pend_valid[0, 0].item() == 1
    motion._planner_truth_tts[0] = 0.05  # 跌破 min_tts:此后不再生成
    _step(rt, motion)  # t2: 无生成,但环照常推进
    _step(rt, motion)  # t3: t1 的元组按期出队提交(调度器自会按验收判据拒收过期件)
    assert len(motion.submissions) == 1


def test_exchange_rejects_malformed_tuple_shapes():
    rt, motion = _coupled_cmd(delay=2)
    with pytest.raises(ValueError, match="one atomic"):
        rt._exchange_pending_planner_revision(
            torch.tensor([0]), torch.zeros(2, 3), torch.zeros(1, 3),
            torch.zeros(1, 3), torch.zeros(1),
        )
