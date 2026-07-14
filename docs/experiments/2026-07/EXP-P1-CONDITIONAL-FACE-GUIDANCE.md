# EXP-P1-CONDITIONAL-FACE-GUIDANCE — 不逃离就绪区的固定预算 Reward

- 状态：`Partial`（E1 source gate 已合入 `main`；尚无 Pod runtime/checkpoint）
- 阶段/轴：阶段 1 固定点；Reward 争抢机制
- 集成小目标：在不牺牲触点、拍速、完成率或安全的前提下，降低有符号拍面误差
- 人类负责人：franco
- 执行者：Codex
- 复核/决策负责人：franco
- 最高证据等级：`E1`（`main@61007e9` source/tests；尚未运行 Isaac）
- 创建日期/最后复核日期：2026-07-14

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的
[`racket_face_conditional_guidance_weight`](../../DEFINITIONS.md#conditional-face-guidance)
是“在击球窗内把固定成本从就绪缺口连续换成拍面误差”的 Reward 开关，不是另一张拍面题库。

## 问题与假设

已有静态 signed-face 线性罚在整段 `pre_strike | strike_window`（击球前或击球窗）持续收费。
最近的 `-0.4/-0.2` 单 seed 方向筛显示：角度可以改善，但位置、拍速或完成率同时付税；详细 runtime
receipt 进入 `main` 前，这里不抄成正式成绩。可证伪假设是：负结果主要来自“拍子尚未到位时多个 Reward
争同一自由度”，而不只是线性罚的权重不对。

朴素的 `readiness * face_error` 不能使用：配非正权重后，策略可故意离开就绪门，把罚金降成零。
本实验改问：若击球窗内始终保留同一最大成本，但随位置/拍速就绪把成本连续换成拍面误差，能否相对
匹配对照降低有符号拍面误差，同时不造成位置/拍速/完成率/安全退化？若不能，该结构假设被证伪；
不再扫更多静态权重。

## 冻结公式与单一因果轴

令 `e_theta` 为 [`raw-A`](../../DEFINITIONS.md) 实测拍面与题目目标面的有符号夹角，`e_p` 为当前
swing-through 目标点的位置误差，`e_v` 为完整拍速向量误差。紧支撑门为：

```text
compact(e; full, zero) = clamp((zero - e) / (zero - full), 0, 1)
face_fraction = clamp((e_theta - 0.262) / (pi - 0.262), 0, 1)
readiness = compact(e_p; 0.075 m, 0.095 m)
            * compact(e_v; 0.5 m/s, 1.0 m/s)
penalty = wide_strike_window * (1 - readiness * (1 - face_fraction))
```

`penalty` 严格在 `[0,1]`；Reward 权重必须非正，因此 `|weight|` 就是每个仿真 step 的最大罚金预算。
位置 `7.5 cm` 是 exact pass、`9.5 cm` 是 virtual-ball capture；拍速 `0.5 m/s` 是 exact pass，
`1.0 m/s` 是外门。击球窗外成本为零；击球窗内任一就绪外门之外成本固定为 1、拍面梯度为零；完全
就绪后成本等于拍面误差分数。因为 `d penalty / d readiness = -(1-face_fraction) <= 0`，位置或拍速
越就绪，成本绝不会增加；策略不能通过故意离开门来躲罚。15° 以内的拍面分数为零。

这不是“纯 face、对 readiness 完全无标量影响”的虚假声明：它会在拍面已有正确方向时给就绪改善
减免，但方向只可能与位置/拍速目标一致。首轮的单一因果轴是这整个固定预算联合机制的开关；门宽、
公式和预算不另扫。

首轮只有一个逻辑差异：

| 单元 | 人话 | 唯一差异 |
| --- | --- | --- |
| control | 同 source 固定点 fresh 对照 | static guidance `0`；conditional guidance `0` |
| treatment | 不逃离就绪区的固定预算纠面 | static guidance `0`；conditional guidance `-0.4` |

两格必须同 source commit、seed 3、v4rg 动作/bank/exam、零摩擦 plant、4096 environments、
1001 updates、PPO/观测/action/动作模仿/其他 Reward 和 `200/500/1000` checkpoint。由于新增字段进入
training hard contract，旧 source 的 control checkpoint 不能冒充匹配对照。首轮不扫门宽、不扫 weight、
不买第二 seed。

## 不可补偿安全边界

该项只返回非负 magnitude，再乘非正权重；它不会提供正安全信用，也不改 termination、自碰/自打、
joint/torque/qdot limit、观测、动作或 plant。以下任一项都独立判失败，不能拿拍面或 composite 改善抵消：

- 新出现 self-hit、桌网碰撞、非有限状态、hard-contract 漂移或 guard 类别；
- treatment 的 physical/root fall 或 guard reset 相对 control 恶化；
- checkpoint 文件名/嵌入 iteration、finite、fresh lineage 或相邻 hard-contract SHA 不一致。

## 运行漏斗与 `+200` 早判

1. **E1 source gate：** focused math、Hydra fail-loud translation、raw-A 共享配对、hard-contract
   bounds、就绪单调性反例与默认关闭测试全部通过；feature 合入 `main` 后才允许 machine prereg。
2. **E2 发射门：** lean queue 的 `doctor --live` 与 first-iteration marker 通过；control/treatment 必须在
   同一新 source 下成对启动，不能用已有旧-source control。
3. **`+200` 硬早判：** 两 checkpoint 均可审计；treatment 的
   `face_conditional_guidance_gate`、`face_conditional_guidance_error_fraction`、
   `face_conditional_guidance_cost_fraction` 和该 Reward contribution 必须 finite。若最后 21 updates 的
   gate 全零，说明机制从未产生拍面纠偏信号，判机制无效；若 position/velocity/
   completion 任一比 control 低超过 5 个百分点，或出现上述安全退化，立即保全日志并拒绝该格。
4. **`+500` 方向门：** treatment 的较差侧 signed normal error 至少比 control 低 10°，同时 position、
   velocity、completion、composite 较差侧均不低超过 5 个百分点；否则不跑第二 seed。
5. **`+1000` 决策：** 同一 immutable 每侧 50 题卷；只有 signed-face 改善、两侧 composite 不退、
   安全不退三者同时成立，才给胜者和匹配对照购买第二 seed。任何 Isaac 结果仍须过 vendor MuJoCo/
   Gate3，不能用解析上台冒充最终演示。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 同 source fresh 对照（待 machine prereg） | 未启动 | seed3；200/500/1000 | E0 | 无 | 必须与 treatment 同 source |
| 不逃离就绪区的固定预算纠面（待 machine prereg） | 未启动 | seed3；200/500/1000 | E0 | 无 | 唯一逻辑差异为 conditional weight `-0.4` |

## 决定

- 决定：`inconclusive`
- 理由：公式、反向激励反例与 source gate 已进入 `main@61007e9`，但未做 runtime smoke 或训练。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：补 exact source/machine prereg 后成对进入 lean YAML；禁止直接手写 CLI 点火。

## 复现与证据

源码入口是 `hope_rewards.racket_face_conditional_guidance`；训练只暴露一个开关，固定门参数由 hard
contract 记录。运行方法与 first-iteration/claim 纪律见[训练操作](../../operations/run_training.md)和
[lean queue 操作](../../operations/run_lean_training_queue.md)。本实验不授权 Pods、judge 或真机。
