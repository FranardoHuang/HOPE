# EXP-P1-V1V2-POST-SWING-INTERACTION — 精度组合下提高随挥后重放覆盖能否改善恢复

- 状态：`blocked`（activation 仪表 source `3ced5a2...` 已 exact 绑定；等待重跑 strict full-scene probe）
- 阶段/轴：Phase 1 fresh C；V1+V2 与随挥后状态重放起点概率的组合效应
- 集成小目标：保住 V1+V2 的击球精度，同时提高从上一拍余势中恢复并完成下一拍的能力
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（机器可读预注册与离线测试；没有本实验 runtime）
- 创建日期/最后复核日期：2026-07-14 / 2026-07-14

共享缩写见[术语与人话对照](../../DEFINITIONS.md)：[`V1`](../../DEFINITIONS.md#v1-free-wrist-velocity)
是在持拍手腕的线速度模仿中释放该手腕，[`V2`](../../DEFINITIONS.md#v2-strike-window-imitation)
是在击球窗把动作模仿缩放到四分之一；[`post-swing replay start`](../../DEFINITIONS.md#post-swing-replay-start)
是从策略自己完成上一拍后写入的状态缓冲中启动一次真实 episode。本实验是第三个独立因果问题，不是给已失败
setting 复制 seed，也不是把单机制 post-swing 结果改名重跑。

## 问题与可证伪假设

首轮 `V1+V2` 在 `+500` 有击球精度信号，但击球前摔倒率较高；单独把 post-swing replay 概率提高到
`0.50` 时，completion 与 pre-fall 方向改善，却因为没有 realized replay-start numerator/denominator，无法
证明机制实际覆盖，也没有保住 signed-normal/composite。两种机制可能互补：V1+V2 给击球控制留预算，较高
post-swing 覆盖则让策略更常练习吸收上一拍余势；也可能互相争夺恢复到击球的有限时间。

可证伪假设：在 V1、V2、动作、题库、plant、seed 和预算完全相同的条件下，只把
`post_swing_start_prob` 从 `0.25` 提到 `0.50`，并先证明两格真实 replay 覆盖后，treatment 在 replay-start
子样本上的 pre-fall 至少下降 3 个百分点、completion 至少提高 5 个百分点，post-fall 不恶化超过 2 个
百分点；位置、速度、带符号拍面法向和 strike composite 四项通过率各自不下降超过 5 个百分点。自打或
unsafe contact 任何一项失败都不可由 reward、恢复或精度收益补偿。

## 冻结 setting

机器真源是
[`phase1_fresh_c_v1v2_post_swing_interaction_queue_20260714.yaml`](../../../configs/phase1_fresh_c_v1v2_post_swing_interaction_queue_20260714.yaml)。

| 字段 | 冻结值 |
| --- | --- |
| Source | clean exact `3ced5a218eab322ebc4ebea6c73ecf64ee47cc5e` / `/workspace/codexschema/nohope_p1_post_swing_activation_3ced5a2`；含 activation counters/logger，但 strict probe 尚未运行 |
| 忽略 A3 资产 | donor `6d93bcb...`；46 files、15,378,264 bytes、tree SHA `0137f59b...26c6`；禁止 symlink |
| 初始化/seed | fresh / `3`；只买一个 seed |
| 预算 | `4096 environments × 1001 updates`，每 `100` 保存，配对 milestone `200/500/1000` |
| 动作 | v4rg 正反手 runtime-order 动作；shared-face signed action |
| Bank/exam | schema-3 rebound train bank；同 family immutable K100 只绑定身份，本实验不授权 judge |
| Plant | `zero_joint_friction=true` 的现行可重放训练协议；不声称是部署 plant |
| 共同机制 | V1=`true`；V2=`0.25`；base-decel=`0`；qdot hinge=`0`/margin=`0.85`；conditional face=`0` |
| 唯一差异 | control `post_swing_start_prob=0.25`；treatment `post_swing_start_prob=0.50` |
| 调度 | `dispatch_pods: [pod2]`；control 优先 Pod2 GPU1，treatment 优先 Pod2 GPU2；Pod1 不做 live snapshot/claim/launch |

顶层 `launch_authorized=false`，两行均为 `blocked`。上一个 `caeb9ad` strict probe 不能替新源码背书；YAML
现已绑定新增 activation 仪表的 clean exact `3ced5a2...`，必须在这个新 detached checkout 重跑 strict
full-scene terminal probe，审核通过后才能用另一个提交同时解除顶层闩和两行 blocker。当前 live caeb source
不修改。

## Activation：先证明真的从随挥后状态启动

两格做任何恢复、平衡或击球比较前，都必须在同一 milestone 的实际 runtime 证据中给出：

1. V1 eligible denominator 与持拍手腕 exclusion numerator 都大于零；V2 的 strike-window eligible
   denominator 与 `0.25` scaled numerator 都大于零。
2. `Live/motion/post_swing_replay_buffer_not_ready_reset_count` 单列缓冲尚未达到
   `post_swing_min_fill=256` 的 true resets；它们不能进入 eligible denominator。
3. `Live/motion/post_swing_replay_eligible_reset_count` 是缓冲 ready 后发生的 true episode reset 数；两格每个
   决策窗都必须大于零。`Live/motion/post_swing_replay_random_not_selected_reset_count` 单列概率抽样未选中的
   eligible reset。
4. `Live/motion/post_swing_replay_selected_reset_count` 是冻结概率区间选中的 numerator；
   `Live/motion/post_swing_replay_started_reset_count` 只有在 root 与 joint state 写入均返回后才增加。必须逐 update
   满足 `selected + random_not_selected == eligible` 和 `started == selected`，两种 numerator 都必须大于零。
5. realized `started/eligible` 必须与预注册概率相符：control 在 `[0.20, 0.30]`，treatment 在
   `[0.45, 0.55]`。只看到配置回显或缓冲有内容不算 activation。
6. 计数、所有用于比较的标量和 checkpoint 必须 finite；checkpoint 还需 filename=embedded iteration、fresh
   lineage、hard contract、source 与 claim 绑定。

缺任一 denominator/numerator 或条件子样本指标时，本 milestone 记为 `invalid/instrumentation-blocked`，不得把
aggregate completion/fall 的好坏解释成 post-swing 机制效果。这个 fail-closed 边界正是对首轮 post-swing
单格缺口的修复。

## 固定预算里程碑

比较都使用同 milestone 最后 21 个 updates 的配对均值。activation 必须由上面的 exact event counts 独立
证明，不能拿 aggregate reward 或 fall/completion 代替。行为 screen 使用现役同定义的
`swing_completion_rate`、`pre_strike_fall_rate`、`post_strike_fall_rate`；它回答“提高 replay 覆盖后的完整
setting 是否更稳”，不是同一初态分布下的 replay-conditioned policy 能力，因此即使通过也不能单独宣称已学会
通用 reset 或连续恢复。

| Milestone | 作用 | replay 恢复/平衡 | 四项击球精度 | 动作 |
| --- | --- | --- | --- | --- |
| `+200` | 方向与仪表检查 | treatment pre-fall−control `≤+0.05`；completion 差 `≥−0.10` | 每项差 `≥−0.10` | activation/finite 不通过则 invalid；不 stop/promote |
| `+500` | 固定预算 screen | pre-fall 差 `≤−0.03`；post-fall 差 `≤+0.02`；completion 差 `≥+0.05` | 每项差 `≥−0.05` | 记录 pass/fail/inconclusive，但两格预算不改，不 stop/promote |
| `+1000` | 训练内终档 | 与 `+500` 相同，检查改善是否持续 | 每项差 `≥−0.05` | 只形成 candidate/reject/inconclusive；不自动采用 |

四项精度分别是 racket position、racket velocity、**带符号** racket normal 和 strike composite pass rate，
逐项判；不能用 composite 或恢复收益掩盖单项退化。`self_hit_rate` 与 `unsafe_contact_rate` 是不可补偿门，
任何非安全变化都要单列并停止采用解释。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| V1+V2、随挥后重放概率 0.25 对照；`phase1_fresh_c_v1v2_post_swing_p025_control_seed3_20260714` | blocked | `200/500/1000`，seed 3 | exact `3ced5a2...` source 单测；新 strict probe 尚缺 | 无 | 不得发射 |
| V1+V2、随挥后重放概率 0.50 treatment；`phase1_fresh_c_v1v2_post_swing_p050_seed3_20260714` | blocked | `200/500/1000`，seed 3 | 与 control 完全配对、唯一概率 delta；新 strict probe 尚缺 | 无 | 不得发射 |

## 决定与边界

- 决定：`inconclusive`；activation 计数的源码/CPU 单测已补，但新 source 尚未过 strict full-scene runtime，
  两格仍不可发射。
- 是否纳入当前 setting：`no`。
- 第二 seed、任何 Isaac/MuJoCo judge、promotion、部署和真机均为 `false`；必须由后续独立预注册解锁。
- `post_swing_start_prob` 只改变真实 episode reset 的初态覆盖，不证明随时来下一球，也不等于
  carry-state、T1 event-driven timing 或 learned reset。

## 离线复现

本提交没有连接 Pod，也没有创建 claim、run、checkpoint 或结果。离线验收为：

```bash
pytest -q tests/test_phase1_fresh_c_v1v2_post_swing_prereg.py
pytest -q tests/test_run_lean_training_queue.py
```

专属测试证明唯一 delta、source/motion/bank/exam/seed/预算完全配对、Pod1 不 dispatch、当前 blocker 下无
assignment；只在显式解除 blocker 的离线反事实中，两条 job 才分别落到 Pod2 GPU1/GPU2。它也拒绝旧
namespace/source 或 placeholder。实际发射必须继续走统一 lean queue harness；本文不维护另一份算力优先级
队列。
