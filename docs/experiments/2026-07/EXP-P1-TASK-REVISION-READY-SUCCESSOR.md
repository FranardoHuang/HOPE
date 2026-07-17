# EXP-P1-TASK-REVISION-READY-SUCCESSOR — 修好准备态量尺后，哪种组合值得继续训练

- 状态：`Partial / NO-LAUNCH`（source-only；full-scene probe 尚未运行）
- 阶段/轴：Phase-1 / 同球实时任务修订后的准备态与关节速度约束
- 集成小目标：先证明准备态整数账本在真实 Isaac 场景可测，再用最小四格比较 Reward 组合
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-17

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的
[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 是“关节实际速度超过运行时限位后才收费”的
惩罚项，**不是**随机推机器人；随机横向躯干力是另一条尚未获准点火的环境轴。

## 问题与假设

旧 task-revision 池的准备态（ready）分母结构性为零：planner 模式已用剩余击球时间作为唯一准备时钟，
但旧量尺只在已清零的 legacy hold 中采样。因此 19 份旧 `+1000` receipt 不能补算、排名或淘汰。

本 successor 回答两个依赖有先后顺序的问题：

1. 新量尺能否在完整 Isaac scene 中对每个新 `(control_epoch, task_id)` 恰采一次准备态，并把总样本、
   planner task-entry 样本、foot-contact/foot-slip 的可测/不可测样本守恒起来？
2. 量尺通过后，在同一 full-state parent 下，增强准备态 Reward 与启用关节速度限位惩罚分别是否改善两个
   独立 100-update 窗的完成率、平衡/准备债与安全，而没有牺牲 0.5 秒和宽准备时间覆盖？

若 probe 的任何整数守恒式、finite、source/contract binding 不成立，则四格保持 `NO-LAUNCH`。若四格已经
运行但稀疏合法回台的 eligible 机会不足，即使回台计数为零也保持 inconclusive，不能淘汰该格。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练 source | exact `d7c38fcf70e7e9420800437fd5b467168ae72580` |
| RunPod source root | Pod2 `/workspace/codexschema/nohope_main_d7c38fcf`，必须 clean detached exact |
| 机器可读实验纸 | `configs/phase1_task_revision_ready_successor_20260717.yaml` |
| 队列执行器 | `scripts/run_phase1_task_revision_ready_successor_queue.py`，SHA `2cf2f3dd…5c8f`，专项 `32 passed` |
| full-scene probe | Pod2，4096 environments × 2 PPO updates；只验证机制/账本，不判行为 |
| 四格共同 parent | Pod2 `taskrev_p2_equal_reward@model_5700`；完整 policy/value/optimizer/normalizer state 恢复 |
| 四格因子 | ready Reward：baseline=`foot_orientation -0.3, prestrike_upright -1.0`；strong=`-0.6, -2.0`；qdot-limit hinge 权重：`-5 / 0` |
| 训练长度 | parent 后最多 1001 updates；绝对 milestone 由 YAML 绑定为 `+200/+500/+1000` |
| 行为量尺 | 每个 milestone 必须是两个互不重叠、各 100-update 的整数窗口 |
| 资源 | 仅 Pod2；前三格先放 GPU0/1/2 各一，第四格选择当时最空闲 GPU，并避让 exact-0.5 K100 evaluator |
| 安全/重放 | plan-only 默认；fresh `O_EXCL` claim/binding；自动 retry=false；broad signal/真机=false |

实验纸是数值与路径真源；本文不另造一个可漂移的 Reward/claim/path 副本。Pod1 不在本实验范围内。

## Probe 硬门

四格 launch activation 只能在一次 fresh full-scene probe receipt 与现有 read-only parent inspection
输出都经过人工复核、并被逐字写回新 activation commit 后变为真；parent inspector 本身不在 Pod
写 receipt。probe 必须同时证明：

- `ready_phase_sample_count > 0`；
- `ready_planner_task_entry_sample_count == ready_phase_sample_count`；
- tilt、base 与 station 三类 ready eligible denominator 都等于 `ready_phase_sample_count`；
- foot-contact 与 foot-slip 两项分别满足 `eligible + unavailable == ready_phase_sample_count`；缺传感器记
  unavailable，不能伪造零；
- `ready_planner_legacy_hold_violation_count == 0`；
- 旧 ready sums/counts 与新增 witness 同时存在且整数守恒；
- task-revision attempt/accept/reject、last-precontact 与 actor-visible 计数完整；
- checkpoint filename iteration 等于 embedded iteration，所有浮点 tensor finite，schema-3 hard contract、
  claim、binding、source 与 lineage 对拍通过；
- 日志无 NaN/Inf/Traceback/OOM/malloc/Killed/提前退出，进程自然结束且无 worker/judge/Kit 残留。

任一项缺失都只能保留失败证据，不能改 activation、自动重试或先启动四格再补账。

## 四格与可证伪解释

| 人话名称 | ready Reward | qdot-limit hinge | 要回答的问题 | 当前状态 |
| --- | --- | ---: | --- | --- |
| 现役强度对照 | baseline | `-5` | 修量尺后，现役配方的准备/平衡是否可测 | `blocked on probe` |
| 只移除超速惩罚 | baseline | `0` | 取消超速惩罚是否缩短出手而不破坏平衡 | `blocked on probe` |
| 只增强准备态 | strong | `-5` | 更强 ready shaping 是否改善准备/平衡 | `blocked on probe` |
| 增强准备态并移除超速惩罚 | strong | `0` | stronger-ready 能否抵消取消超速惩罚后的平衡债 | `blocked on probe` |

这四格使用同一个 seed/parent，是机制漏斗的第一轮，不是 seed-stability 结论。失败 setting 不复制 seed；
只有通过预注册 milestone 门的胜者与匹配对照才有资格申请第二 seed。

## Milestone 与淘汰纪律

- `+200`：只判 full-state resume、finite、合同/身份、ready/qdot/task-revision 机制是否真实激活；不按稀疏回台零值淘汰。
- `+500`：只使用两个完整 100-update 整数窗判断明显崩坏、安全和准备/平衡债；缺任一窗或分母不完整则继续。
- `+1000`：同 parent 四格在共同 milestone 到齐后才做 tolerance-aware Pareto；不得用跨历史 EMA 或不同步 checkpoint 排名。
- 无 exact eligible 机会的稀疏回台为“未测”，不是 0% 能力。相同规则也适用于需要真实击球才出现的 Reward。
- 至少保留同 parent 两格；同时保留一个实际包含 exact-0.5 暴露的候选和一个宽准备时间候选。
- stop 必须另有 checkpoint-bound 单臂 receipt 与 same-parent portfolio receipt，signal 前重验 exact
  PID/PGID/starttime/argv/claim/binding；本 source-only 阶段不授权 stop。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| Parent 只读语义检查 | `passed` | equal-reward `model_5700` | checkpoint `521d41e9…984c`；hard contract `7d30b603…bf70`；lineage `0` | inspection content `e17cedb1…ade4`；evidence `85967393…1096` | 只证明 parent 身份、finite、optimizer 与合同；不是行为结果 |
| 准备态量尺 full-scene probe | `ready / NO-LAUNCH` | fresh 4096-env × 2-update probe | source exact `d7c38fc…580` | 尚无 receipt | 必须先过；不是行为结果 |
| 现役强度对照 `ready_baseline_qdot_minus5` | `blocked` | equal-reward `model_5700` full state | 尚无 | 尚无 | probe receipt 未写回 activation |
| 只移除超速惩罚 `ready_baseline_qdot_zero` | `blocked` | 同 parent/seed | 尚无 | 尚无 | qdot 不是随机力；probe 未过 |
| 只增强准备态 `ready_strong_qdot_minus5` | `blocked` | 同 parent/seed | 尚无 | 尚无 | probe receipt 未写回 activation |
| 增强准备态并移除超速惩罚 `ready_strong_qdot_zero` | `blocked` | 同 parent/seed | 尚无 | 尚无 | 第四格 GPU 需动态避让 K100 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

## 决定

- 决定：`inconclusive / NO-LAUNCH`
- 理由：source/queue 的 parent inspector 与最小 probe 已冻结并通过 32 项 source test；Pod2 只读 parent
  语义检查已经通过，但 full-scene probe 尚未执行，fill/behavior/portfolio/stop 入口也尚未实现，四格
  activation 不能成立。
- 是否已纳入当前 setting：`no`
- 下一门：按[操作文档](../../operations/run_phase1_task_revision_ready_successor.md)先 plan，再分别唯一消费
  read-only parent inspector 与 probe；两份证据通过后仍须实现 fill 与后续行为/portfolio consumer，再由新
  activation commit 解锁，不能直接启动四格。

## 证据边界

本阶段只在 Pod2 执行了 parent 的只读语义检查；没有启动 trainer、simulator、judge 或真机，也没有产生
新的行为分。32 项 source 单元测试、本地 `validate/plan` 和 parent inspection 只能证明 parent 身份与
probe 入口按设计 fail closed；不能冒充 full-scene、0.5 秒回球、vendor MuJoCo 或部署通过。G05 保持
`Partial`。
