# EXP-RECOVERY-TUPLE-ABC — 哪种击球后命令是连贯的？

- 状态：blocked
- 预注册状态：旧 A/B/C 结构已 preregistered；07-13 新 reward 次序尚未物化
- 阶段/轴：跨课程阶段的连续能力 / 等待与恢复
- 集成小目标：完成上一拍 + 可接受的等待动作/姿态 + 任意时刻启动下一拍
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：A/B/C 旧结构合同为 E1；2026-07-13 新 reward 次序为 E0

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

当前 deploy idle 行把新的 live-base 位置与上一拍的速度/法向组合在一起；没有任何已绑定训练转移会产生这种状态。
首轮考卷冻结奖励（reward），并比较 A：有明确所有权的安全桥接（bridge）；
B：原子 `canonical ready tuple`；C：在揭题（reveal）前保留完整的上一个 tuple。

2026-07-13 原文与现役代码审计进一步冻结了三个层次：T0 只在完整挥拍周期结束后换题；
T1 改为事件驱动揭题，但不改 reward；T2 才允许 learned shaping。硬安全始终是不可补偿约束，
随机到球首先是环境/题目/截止时间轴，由真实下一拍评分，不作为第三项 dense reward。

**机器合同尚未同步：**`configs/phase1_recovery_tuple_abc_prereg_20260712.json`、对应 validator
和 operation 仍把随机到球 readiness 当第三项 reward，并强制 full `2^3`。它们只证明旧 A/B/C
结构记录可被严格读取，不能证明 07-13 新次序已经物化；新内容寻址合同完成前禁止点火。

若 T1 仍失败，reward 诊断先做动态平衡债与 ready-set potential 的配对 seed `2^2`。
第三个 readiness critic 只有在独立训练/校准集上锁定、不泄露尚未揭示的下一题，并一次性通过
与 sealed Gate3B q50 隔离的 critic-gate q50 后，才允许扩成 `2^3`。存活组合再做固定总预算
mixture，并补第二个总 reward 量级；不能把三个单项正收益直接相加后宣布等待目标完成。

这里只隔离命令/handoff。只有当同一条 no-reset 序列也能安全吸收上一拍、保持已接受的等待动作/姿态，
并在任意有效 deadline 启动任一已启用的下一拍时，连续能力才算完成。一个良好 tuple 本身不能完成该目标。

尚未运行任何行为臂。权威资料：[恢复操作](../../operations/run_phase1_recovery_tuple_prereg.md)和
[连续时序原文审计](../../research/phase1_continuous_rally_timing_2026-07-11.md)。
