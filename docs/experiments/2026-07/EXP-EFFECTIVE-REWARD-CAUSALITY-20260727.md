# EXP-EFFECTIVE-REWARD-CAUSALITY-20260727 — 当前看起来学得好，是正确调参还是意外有效？

- 状态：`preregistered`
- 阶段/轴：Reward 配方真值 / task 自洽性 / paired causal A/B
- 集成小目标：先证明 trainer 真正吃到哪套 Reward，再区分 task 自洽与权重大小的贡献
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（source/config/receipt 审计；配对训练未跑）
- 创建日期/最后复核日期：2026-07-27 / 2026-07-27

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的
[`effective Reward recipe`](../../DEFINITIONS.md#effective-reward-recipe)指 Hydra/config compose
和所有 override 之后，Isaac 实际收到的 active Reward callable、weight 与 params，不是某个
`reward_pack` 标签或设计文档中的名义表。

## 结论先行

目前不能说“高质量权重调对了”。相反，现役 task YAML 显式值覆盖了 v2 pack 的名义冻结值：

| 通道 | v2 pack 名义值 | 实际 composed 值 | 实际/名义 |
| --- | ---: | ---: | ---: |
| 球拍位置 | 393.4 | 4.0 | 0.01017× |
| 球拍速度 | 295.1 | 0.5 | 0.00169× |
| 球拍拍面 | 229.5 | 0.5 | 0.00218× |

所以“以 393.4/295.1/229.5 训练得很好”这一叙述是错误的；显式 task key 后写后赢。按同一 v4rg
probe 的量纲估算，实际 quality potential scale 约 `0.0655`，名义 pack 约 `7.3838`，相差约
113 倍。这个估算只解释配方量级，不是训练成绩。

当前最合理、仍待 A/B 的解释是：

1. **主要改善来自 task 终于自洽。** 旧随机 racket velocity 与解析 landing 目标会互相打架；
   inverse-solved/单动作 task 让参考动作、目标拍速和结果方向首次相容。
2. **意外的低 quality 权重没有阻止学习，甚至可能避免了高方差冲突。** 但没有 matched run
   不能说低权重更优。
3. analytic Reward、metric 和题库共享 oracle/cache/physics，好的训练曲线可能有循环验证；
   必须用独立 heldout/physical receiver 才能晋级。
4. seed 方差仍大；不能挑一条好曲线归因。

## 问题与假设

问题：在 task、动作、题库、plant、PPO 和随机种子配对不变时，实际低质量权重
`4/0.5/0.5` 是否优于或不劣于名义高质量权重 `393.4/295.1/229.5`？

首要假设不是“低权重必胜”，而是：**task 自洽是当前改善的主要解释；把名义高质量权重真正打开
未必继续改善，可能因尖峰/方差伤害学习。**

如果 matched A/B 中名义高权重在独立 heldout 上稳定显著更好、且没有 unsafe/方差代价，则该假设
被证伪；若两者接近而都显著优于旧不自洽 task，则支持“task 自洽为主”。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练/eval/main commit | 待 task-first 整合 commit 冻结 |
| 动作/action 集 | 同一 training-authorized action set |
| 观测/action 合同 | 同一 `task_first_n<N>` |
| Reward | A=`4/0.5/0.5`；B=`393.4/295.1/229.5`；其余 active terms/callables/params 完全相同 |
| Plant/engine | 同一 Isaac scene/plant hard contract |
| 训练/考试 bank 或 schedule | 同一 predetermined curriculum schedule + 同一独立 heldout |
| Checkpoint/seed | fresh，2 个 paired seeds；不从好 parent 热启 |

两臂必须各自钉住 composed effective Reward receipt SHA。只声明 `reward_pack=v2` 不足以证明 B；
B 需 strict compose 后 receipt 确认三键正是名义值。

两臂还必须消费**同一冻结 racket-task 分布**。ball-first online 与 task-first replay 的 producer
顺序因果属于
[task-first 实验的独立 A/B](EXP-TASK-FIRST-N-ACTION-20260727.md#producer-顺序的独立因果对照)；
不得在本实验里让 A/B 同时换 producer，否则结果只能叫“Reward + producer package”，不能回答权重。

## 实验差异

- 对照 A：现役实际低质量权重 `4/0.5/0.5`。
- 处理 B：名义高质量权重 `393.4/295.1/229.5`。
- 改变的变量：仅三项球拍 quality 权重。
- 其余固定项：task、课程 schedule、动作、seed、PPO、landing/模仿/安全项、plant、预算、
  heldout。
- 决策规则：2 个 paired seeds；每 seed 同卡/同起点配对，宏平均 paired bootstrap。
- 停止/无效规则：effective receipt 不符、curriculum 分布不同、NaN、动作 starvation、
  unsafe 恶化、oracle/heldout 泄漏或任一 run 缺 all-attempt ledger。

纯 task-first executor 首跑不带 analytic landing Reward；本 A/B 是对“历史/ball-conditioned
自洽 task 下三质量权重”的因果复核，不应阻塞 ball-free task-first source/diagnostic smoke。

## 运行态快照（不是正式成绩）

2026-07-27 10:40Z 的只读 live 快照中，两 Pod 六卡仍被占用；按审计行顺序，六条进程的
analytic legal 百分比为
`11.03 / 43.08 / 31.91 / 0 / 36.95 / 0`。这些 run 还存在三类身份问题：

- `c_ep20s`（人话：声称 20 秒 episode 的候选臂）名称写 20 秒，但最终 composed environment
  仍是 10 秒；
- `c_speedwide`（人话：扩大来球速度范围的候选臂）重复写 override，最终值实际为
  `[0.5, 1.2]`；
- 一条 seed-1 候选重复写 seed，最后生效值为 `1`；六条 run 均没有
  [`run_binding.json`](../../DEFINITIONS.md#trainer-run-binding)。

因此这些百分比没有 exact 配方身份、正式分母、milestone 或配对关系，只能说明 seed/arm 方差和
运行账本风险都很大，**不进入成绩表、不选胜者、不证明任何 Reward 因果，也不是 task-first
成绩**。

历史单变量方向信号也很窄：同一波里把 landing 从 `1648.8` 降到 `791.9`，约 3.8k 时 analytic
legal 从 1.8% 到 6.8%；它只有一个 seed，且没有到预注册 8k 决策点，只能作为“高 landing
不一定更好”的方向线索。

### 2026-07-27 exact run 复核

Pod 保存的 composed `env.yaml` 进一步确认当前几条 banked run 真正使用：

```text
racket position / velocity / normal = 4.0 / 0.5 / 0.5
virtual landing = 1648.8
death penalty = -1800
```

名义 `393.4/295.1/229.5` 从未进入这些 run。`c_ep20s_seed0`（人话：声称把 episode 加到 20 秒）
把 override 错写成 `++env.episode_length_s=20.0`，最终 saved config 与
`c_base_bank_seed0` 都是 10 秒且 `env.yaml` SHA 同为
`edb32d4beab8dcacc0361251719406e70dddd8da108f9758bfb791517548ea54`。两份 agent config 去掉
`run_name` 后也相同，因此它是一次意外的同 seed 跨 GPU 复刻，不是 episode A/B。

最后 100 个 exact behavior update 先累加分子/分母：

| run（人话） | legal / strike | 聚合率 |
| --- | ---: | ---: |
| `c_ep20s_seed0`（实际仍 10 秒） | `16283 / 32743` | `49.73%` |
| `c_base_bank_seed0`（同 setting 复刻） | `16245 / 32650` | `49.75%` |
| `c_speedwide_seed0`（最终速度范围 `[0.5,1.2]`） | `10143 / 21655` | `46.84%` |
| `c_base_bank_seed1`（同 setting，仅 seed 不同） | `0 / 28123` | `0%` |
| 旧 unbanked seed0（有其他 source 混杂） | `0 / 25152` | `0%` |

这证明 seed0 的约 50% 可复刻，也同时证明同配方 seed1 到约 17k 仍完全分叉。现象最支持“inverse
solved bank 消除了球/task 冲突”，但高 `virtual_landing` Reward 与报表共用同一 analytic
contact/landing oracle，仍有循环验证风险；当前目录未找到这些 run 的独立 exam/judge 结果。

账本也不够精确：这些 run 都缺 `run_binding.json` 与 `params/effective_reward_recipe.json`；
现有 `training_contract.json` 不完整绑定 Reward/table，甚至 table on/off 候选共享
`16be6a4703dfd30914687218b40ed447db3278a6bfc69e6120f56c59e0989516`。所以这些结果只能作方向证据，
不能追认为 matched A/B 或采用结论。

在既有 solver A/B 之前，新增一个更直接的配对关系 canary：

- `P-paired`：保持同 action/ball/base/aim 经 fixed solver 得到的原始 ball↔task 配对；
- `P-shuffled`：保持完全相同的 ball 边际和 solved-task 边际，只在同速度/难度 stratum 内打乱
  ball↔task 配对。

它只改变物理配对关系，能直接检验“自洽是否是主因”。动态 curriculum 在该因果 A/B 期间关闭。
鉴于当前 seed0≈50%、seed1=0，正式 Reward/solver 结论不能只用两个 seed；先以两个 paired seeds
做 canary，采用至少补到四个 paired seeds或预注册 sequential CI。

## 判读

每动作独立 heldout 512 题：64 center，四个单轴各 96，64 joint edges。成功要求 exact strike、
位置 `<=7.5 cm`、速度误差 `<=0.5 m/s`、拍面 `<=15 deg`、base `<=10 cm`，并在 recovery
前无 table hit/physical fall。报告 Wilson LCB、逐轴、p50/p90 和 all-attempt unsafe。

里程碑只作如下用途：

- iteration 100：数值/receipt/activation health，不判优；
- iteration 500：只执行预注册的 dominated-candidate stop；
- iteration 2000：两 seed paired bootstrap，宏平均成功率改善需 `>5 pp`；每动作非劣界
  `>-5 pp`；table/fall 任一恶化 `>1 pp` 则不得胜出。

这里 `pp` 是 percentage points（百分点），不是相对百分比。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| composed Reward source audit（无科学 `run_name`） | complete | 无 | E1 | effective-recipe tests | 证明实际值与 nominal 不同 |
| 低质量权重 A，paired seeds | preregistered | fresh / 待冻结 | 未跑 | 未生成 | 不得用旧 live 格替代 |
| 名义高质量权重 B，paired seeds | preregistered | fresh / 待冻结 | 未跑 | 未生成 | receipt 必须确认名义值真生效 |

## 决定

- 决定：`inconclusive`
- 理由：已推翻“名义高权重正在生效”的前提；task 自洽是最强解释，但 matched A/B 尚未运行。
- 是否已纳入当前 setting：`no`（只采用 effective receipt 作为以后 launch 的身份门）
- 局限/下一个 gate：先让每个 run 在构造后写
  `params/effective_reward_recipe.json` 并嵌入 hard contract；再冻结 2×2 paired launch。

## 复现与证据

Source tests 与 receipt 入口见
[构建与测试](../../operations/build_and_test.md#task-first-and-capability-source-tests)；
训练前检查见[run_training](../../operations/run_training.md#effective-reward-truth)。
