"""击球窗带符号孪生时钟(2026-07-25 C1 修复)— host-only 守卫测试。

背景:schema-4 的 truth tts 是任务期限语义,触球后 clamp 钉 0(obs/critic 合同如此)。
但 |tts|<=0.12 的击球窗掩码若也读它,窗从触球一直开到 clip 收尾——随挥全程 ~50-100 步
顶着 ±0.12 s 的设计语义:position/normal 触球后停拍可薅钱、站稳包/face 税全程计费、
模仿在恢复段被 0.25x 捂嘴(与"随挥要跟老师学完整动作"正相反)。

修复合同(本文件钉死):
* commands.py 维护 `_planner_truth_tts_signed`:任务安装时 = tts,逐步 -dt **不 clamp**
  (触球后转负),reset/planner 失活回 1e6 哨兵(新任务未装时窗保持关闭,fail-closed);
* hope_commands.py 的窗掩码(strike_window/_pos/_wide)在 planner 路径读孪生时钟,
  time_to_strike / pre_strike / exact_strike(一拍锁存)语义不变。

真实 _update 依赖整套 Isaac env,host 上用两层测试兜底:
1. 纯算术:孪生时钟推演——窗恰在 [-0.12, +0.12] 内为真,+0.12 后关闭;哨兵=窗关;
2. 源码守卫(repo 先例:test_face_sign_per_clip 的 getsource 钉法):窗掩码必须读
   tts_for_window / _planner_truth_tts_signed,不许退回钉 0 的 time_to_strike。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_strike_window_signed_clock.py -q
"""

from __future__ import annotations

import inspect
import re

import torch

from test_reward_flags_mdp import commands_mod, hope_commands_mod


def _simulate_signed_clock(tts0: float, dt: float, steps: int) -> list[float]:
    """按 commands.py 的孪生时钟算术逐步推演(减 dt 不 clamp)。"""
    clock = torch.tensor([tts0])
    out = []
    for _ in range(steps):
        clock = clock - dt  # 与 _planner_truth_tts_signed 更新式逐字一致:无 clamp
        out.append(float(clock))
    return out


def test_signed_clock_window_closes_at_plus_window():
    dt, window = 0.02, 0.12
    trace = _simulate_signed_clock(tts0=0.5, dt=dt, steps=40)
    in_window = [abs(t) <= window for t in trace]
    # 触球前 0.12 s 进窗,触球后 0.12 s 出窗:窗步数 = 2*window/dt ± 1,而不是"到收尾全开"
    assert sum(in_window) in (12, 13)
    # 触球(t=0.5/0.02=25 步)之后第 7 步(+0.14 s)必须已出窗
    assert not in_window[31]
    # 钉 0 的旧时钟(对照):同样 40 步,触球后永远 |0|<=0.12 → 窗恒开(这就是 C1 病灶)
    clamped = [max(0.0, t) for t in trace]
    assert all(abs(t) <= window for t in clamped[25:])


def test_sentinel_keeps_window_closed():
    assert not (abs(1.0e6) <= 0.12)


def test_commands_source_maintains_signed_twin():
    src = inspect.getsource(commands_mod)
    # 任务安装:孪生时钟与 truth tts 一起写入
    assert "_planner_truth_tts_signed[ids] = tts" in src, (
        "任务安装处不再初始化带符号孪生时钟 —— 击球窗将失去关闭依据"
    )
    # 更新:减 dt 且不 clamp(用负向断言防止有人顺手加 clamp)
    m = re.search(
        r"_planner_truth_tts_signed\[active\] = \(\s*"
        r"self\._planner_truth_tts_signed\[active\] - dt\s*\)(?!\s*\.clamp)",
        src,
    )
    assert m, "孪生时钟更新式被改动(必须减 dt 且不得 clamp,否则触球后窗关不上)"
    # reset:回哨兵(fail-closed)
    assert "_planner_truth_tts_signed[env_ids_t] = 1.0e6" in src, (
        "planner 失活处不再回哨兵 —— 新任务安装前窗会误开"
    )


def test_hope_commands_window_mask_reads_signed_clock():
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand)
    assert "tts_for_window = motion._planner_truth_tts_signed" in src, (
        "planner 路径的窗掩码不再读带符号孪生时钟 —— C1(窗覆盖整个随挥段)会复发"
    )
    assert "_tts_abs = tts_for_window.abs()" in src, (
        "窗掩码退回读 time_to_strike(触球后钉 0)—— C1 会复发"
    )
    # exact_strike 的一拍锁存必须仍在(它读钉 0 的 time_to_strike,靠 _exact_fired 保一次性)
    assert "exact_strike = exact_strike & ~self._exact_fired" in src
