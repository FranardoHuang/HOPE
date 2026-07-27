# EXP-ACTION-CAPABILITY-SELECTOR-20260727 — 任意击球目标能否按安全成功率与优先级选动作？

- 状态：`blocked`
- 阶段/轴：训练后 capability / planner 动作选择 / 任意 N 动作
- 集成小目标：每个合法击球目标选择一个有证据支持的动作，证据不足时可靠 abstain
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（pure Python core/source tests；生产未接线）
- 创建日期/最后复核日期：2026-07-27 / 2026-07-27

共享术语见[术语与人话对照](../../DEFINITIONS.md)，完整顺序与身份合同见
[动作能力 selector](../../interfaces/action_capability_selector_contract.md)。

## 问题与假设

问题：对任意一个击球目标，能否先排除不安全/无支持的动作，再按校准成功率保守下界选动作，并且
只在几乎并列时使用战术优先级？

假设：若能力工件精确绑定 policy/task/Reward/heldout/model/calibration，候选由 planner 的 trusted
producer 内容寻址，排序固定为“硬安全 → support/OOD → 成功率 LCB → delta tie 内标准化
quality/priority → abstain”，则动作库扩大到 N 条时不需要重写二动作 if/else，也不会让优先级盖过
安全性。训练 Reward 不允许直接跨动作充当 quality。

证伪条件：

- 硬失败动作可因高 LCB/priority 被选；
- 低支持、OOD、NaN 或低于最低 LCB 的动作可进入 tie；
- catalog reorder 后 stable UID 错配；
- capability artifact 任一 SHA 漂移仍继续选择；
- 无动作可选时偷偷退回默认正/反手，而不是 abstain；
- caller 可以同时自报 candidate/profile 内容与“期望 SHA”而生产链仍把它当独立授权。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练/eval/main commit | 功能分支最终 commit 待整合后填写；未合入 `origin/main` |
| 动作/action 集 | 任意 N catalog；首个行为卷计划用 action-ball 五动作 view |
| 观测/action 合同 | 训练见 `action_ball_n5`；生产 wire 尚未冻结 |
| Reward | capability artifact 绑定 composed effective Reward SHA |
| Plant/engine | planner candidate 物理 Gate 与 heldout evaluator 各自内容绑定 |
| 训练/考试 bank 或 schedule | 每动作 512 题建议卷；尚未生成 |
| Checkpoint/seed | task-first 胜任 checkpoint 尚不存在 |

## 实验差异

- 对照：只取最高成功率点估计、无 abstain 的朴素 selector（只作离线反例，不接生产）。
- 改变的变量：保守 LCB、support/OOD 门、delta-tie priority 与 abstain。
- 其余固定项：catalog、query、candidate producer、capability artifact、profile。
- 决策规则：每 query 输出所有 action assessment 和一个 selected UID/slot，或唯一 abstain identity。
- 停止/无效规则：任一 identity/SHA/schema/profile authority 不闭合，实验停在 source，不进 ROS。

## 组成与接口

- 正在隔离的组件：stable action catalog、capability artifact、selector profile、candidate/decision
  receipts、排序核心。
- 集成小目标所需的其他组件：ball-conditioned heldout、逐动作 planner adapter、可信 query/candidate
  producer、独立 profile activation、任意 N wire/C++ runner。
- 组件间的接口/交接：action-ball checkpoint → heldout artifact → planner candidates → selector
  decision → frozen action UID/task revision。
- 组件消融后的联合完成规则：source attacks 全拒后，先离线 replay；再 ROS/Jazzy producer/consumer
  first tick；最后同卷 Gate3，不跳级。

## 预注册能力卷

旧 512 题模板不足以覆盖 schema-v3 的 32-arm 能力，只保留为快速离线 canary。正式卷应从独立
768+ heldout 按 action/profile 的 time-to-contact、position、incoming speed/direction、spin
magnitude/direction、aim、base spawn/travel 及 joint edge 分层抽样。每题进入 all-attempt 分母。
报告逐动作/逐 arm support、OOD、校准成功率 LCB、误差 p50/p90、table/fall/recovery。

候选须同时满足 exact strike、位置 `<=7.5 cm`、速度误差 `<=0.5 m/s`、拍面 `<=15 deg`、
base `<=10 cm`，并在 recovery 前无 table hit/physical fall。阈值改变即新 task/heldout SHA。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| selector pure-core host tests（无科学 `run_name`） | source candidate | 无 | E1 | pytest 日志待整合 | 不含 trusted producer/ROS |
| 五动作 heldout 能力拟合 | blocked | 未产生 | 未测 | 未生成 | 依赖 action-ball checkpoint |
| ROS/Jazzy + C++ 任意 N first tick | blocked | 未产生 | 未测 | 未生成 | 当前仍为二动作 schema 4 |

## 决定

- 决定：`inconclusive`
- 理由：排序原则与 fail-closed core 已可审，但 capability data、trusted producer、独立 profile
  authority、wire/C++ 和 Gate3 全缺。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：先完成 action-ball heldout 与 query-conditioned normalized quality model；
  随后冻结 candidate producer 与 schema 5/UID
  transport，再做 ROS/vendor runtime。

## 复现与证据

Host source command 见
[构建与测试](../../operations/build_and_test.md#task-first-and-capability-source-tests)；
planner 运行边界见[run_planner](../../operations/run_planner.md#n-action-selector-boundary)。
