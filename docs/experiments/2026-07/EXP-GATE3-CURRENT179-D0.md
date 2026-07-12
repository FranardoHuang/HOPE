# EXP-GATE3-CURRENT179-D0 — Gate3-D0 当前 exact 全栈能否完成一份考卷？

- 状态：blocked
- 当前环节：前置条件进行中，行为卷未启动
- 阶段/轴：部署验证线 / 厂商集成
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：E2

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

`Gate3-D0` 要求使用真实 planner、生产 C++ runner 和 fresh exact 179-D checkpoint，完成一次边界明确的
厂商 MuJoCo 固定考卷演示。它不是连续对打或真机证据。

已完成：严格 face179 exporter/loader、真实 formal 模型正/负例预检、plan-only 源码门，以及显式标记为
inexact 的 joined-source 诊断。默认关闭的 shadow-solver 接线和 formal shared-epoch/base-sequence
planner-policy tuple 源码已经完成集成；exact `c0a8e46` portable Release 为 `233 passed + 5 optional
skips + 0 failed`，latest-main host 回归也通过。详见
[exact build 卷宗](EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md)。

未完成：最终集成 ROS/Jazzy/AimRT Release、有明确所有权的 supervisor/publisher/fresh-ready ledger、
由 parser 解析并绑定的 config/MJCF/runtime 闭包、49 项发球绑定、真实厂商 backend 首 tick 和固定考卷行为 ledger。

决定：**模型门已通过；当前 179-D Gate3 行为尚未运行**。旧版诊断不能填充这些单元格。
07-11 确实跑过一份 110-D model-13200 legacy Gate3 行为诊断：`13 PASS / 7 FAIL`，
3 次发球只有 1 次合法回球，并出现 1 次摔倒和明显漂移。它证明旧链曾执行，不是 Gate3 pass，
也不能冒充 current exact-179 的结果。
权威资料：[G06](../../gates/G06_isaac_to_mujoco.md) 和 [first-tick 操作](../../operations/run_gate3_first_tick_harness.md)。
