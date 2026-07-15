# EXP-RECOVERY-TUPLE-ABC — 哪种击球后命令是连贯的？

- 状态：blocked
- 预注册状态：旧 A/B/C 结构字节保持 immutable；07-15 frame-0 等待 v2 设计已物化但
  `NO-LAUNCH`；07-13 新 reward 次序尚未物化
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

2026-07-15 新增的
[`phase1_frame0_wait_recovery_contract_v2_20260715.json`](../../../configs/phase1_frame0_wait_recovery_contract_v2_20260715.json)
只物化等待/恢复参考语义，不改上面旧 prereg 的任何 byte。它把连续 episode 分成四段：上一拍结束后、
下一动作尚未揭示时，用**上一拍这个公开动作自己的第 0 帧姿态 + 全零速度**恢复；原子揭题后才切到
**新动作自己的第 0 帧姿态 + 全零速度**等待；随后按新动作原生 clip 挥拍。XY 只在每次参考阶段进入时
从当前站位捕获一次，不能每 tick 跟着已经漂移的机器人走。切换只改参考，不 teleport、不 reset，也不清
observation history、last executed action、action/target delay ring、noise/dropout 或 per-swing bias。

这里“第 0 帧”是 reference，不是 ready 判定。Ready 仍是站位/朝向、直立、低速、双脚支撑与防滑、
执行器余量、自碰/桌网/地面安全以及下一动作/截止时间可达性的**全部容差合取**；任何一项缺测、
non-finite 或失败都算 not-ready，不能靠另一个 Reward 抵消。v2 没有擅自填写这些数值门槛。

源码审计同时抓到现役冲突：`commands.py` 在 hold 时把关节参考换成
`robot.data.default_joint_pos`，root/anchor 速度没有随 hold 清零，body XY 还会逐 tick 重锚到 live robot。
只改关节会形成 frame0/default/root 混合参考，因此本次没有伪造一个局部 adapter。v2 的 source adapter、
phase-entry XY snapshot、揭题不泄漏断言、carry-state runtime receipt、ready 数值容差、Isaac full-scene
probe 和 vendor MuJoCo 连续门均保持 null；`launch-check` 必须失败。

若 T1 仍失败，reward 诊断先做动态平衡债与 ready-set potential 的配对 seed `2^2`。
第三个 readiness critic 只有在独立训练/校准集上锁定、不泄露尚未揭示的下一题，并一次性通过
与 sealed Gate3B q50 隔离的 critic-gate q50 后，才允许扩成 `2^3`。存活组合再做固定总预算
mixture，并补第二个总 reward 量级；不能把三个单项正收益直接相加后宣布等待目标完成。

这里只隔离命令/handoff。只有当同一条 no-reset 序列也能安全吸收上一拍、保持已接受的等待动作/姿态，
并在任意有效 deadline 启动任一已启用的下一拍时，连续能力才算完成。一个良好 tuple 本身不能完成该目标。

尚未运行任何 frame-0 等待行为臂。权威资料：[恢复操作](../../operations/run_phase1_recovery_tuple_prereg.md)、
[T1 接口](../../interfaces/t1_event_training_contract.md)和
[连续时序原文审计](../../research/phase1_continuous_rally_timing_2026-07-11.md)。
