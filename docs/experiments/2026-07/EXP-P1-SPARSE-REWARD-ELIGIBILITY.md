# EXP-P1-SPARSE-REWARD-ELIGIBILITY — 稀疏 Reward 没触发时，早筛能否诚实地继续？

- 状态：`blocked`（E1 source/classifier 已实现；尚无新-source runtime receipt）
- 阶段/轴：阶段 1 milestone 早筛 / Reward 资格分母
- 集成小目标：早筛只淘汰结构性无效；hit-conditioned Reward 没得到机会时继续长曲线
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-15 / 2026-07-15

共享术语见[术语与人话对照](../../DEFINITIONS.md)。本实验不改变正在运行的 source、配方或 Pod 进程。

## 问题与假设

问题：`+200/+500/+1000` 看见某个 hit-dependent Reward 为零时，能否区分“策略真的失败”与“根本没有
击中，所以过网/落台 Reward 无资格出现”？

假设：只要记录 exact strike→virtual capture→net/landing→legal return 的同账整数链，以及 qdot
observed→hinge-active→excess 链，就能把缺机会/被 censor 的窗口强制归为 continue；只有连续两个
milestone 的完整分母才具备被外部决策规则读取的资格。

## 冻结合同

| 字段 | 值 |
| --- | --- |
| 计数 | 非衰减、每 PPO update consume-once；全局与每动作族 |
| 最少机会 | 总 exact strike `100`；每动作 `50` |
| hit-conditioned 激活 | 每动作 virtual capture 至少 `1`（只证明曾触发，不证明比例稳定） |
| 时间一致性 | 连续两个预注册 milestone |
| qdot | 对照 active=`0`；treatment active=`observed`；active treatment 至少一次 excess |
| classifier side effect | 只写 no-clobber receipt；所有状态 trainer 都 `CONTINUE_UNCHANGED` |
| 物理语义 | 仅 analytic virtual Phase A；PhysicalBall Phase B 明确未测 |

机器合同为
[`phase1_sparse_reward_eligibility_contract_20260715.yaml`](../../../configs/phase1_sparse_reward_eligibility_contract_20260715.yaml)，
接口为[稀疏 Reward 资格账本](../../interfaces/sparse_reward_eligibility_ledger.md)。

## 五态与证伪规则

- `NO_OPPORTUNITY_CONTINUE`：没有 exact strike，不能判 Reward。
- `CENSORED_CONTINUE`：机会不足、任一动作不足、捕获不足，或 qdot treatment 没有 excess。
- `MEASUREMENT_INVALID`：身份/schema/动作集合/计数闭包错误；只能判仪表无效。
- `DIRECTION_ONLY`：一个 milestone 资格完整，仍不能停训或晋级。
- `DECISION_ELIGIBLE`：连续两个 milestone 资格完整；只授权外部 paired/q50 决策读取。

若 classifier 能从一个缺 capture 的窗口给出 `DIRECTION_ONLY/DECISION_ELIGIBLE`，或任何状态会自动停
trainer，本假设与实现即被证伪。

## 实现与验证

源码 emitter 直接复用 `_vb_book_strike_step` 的五个真实 mask，不从 EMA rate 反推。qdot 的零返回 probe
和实际 hinge 共用 runtime 31-joint limit 数学与 `common_step_counter`，所以 treatment 同一步只记一次
observed，并另记 active。runner 将整数 total 写入 `Live/racket_target/*` 和 `Live/qdot/*`。

host 验证：classifier `14 passed`；qdot/virtual ledger focused `4 passed`；Hydra translation/hard-source focused
`18 passed`。完整两个 reward 文件为 `160 passed, 6 failed`；六个失败均在未改的 MotionLoader
`Path`/post-swing fixture 路径，focused 新功能未失败，不能写成全套绿。

## 决定与边界

- 决定：`inconclusive`（adopt source contract for review；runtime 尚未跑）
- 是否纳入当前 setting：`no`
- 当前 `2c2d70d` live runs 没有这些 exact counters，禁止用旧 EMA 回填；它们继续按原长曲线合同运行。
- classifier 的 `DECISION_ELIGIBLE` 不是 winner/stop/promote；仍需 matched control、immutable q50 和 vendor
  MuJoCo。
- PhysicalBall Phase A 禁用机器人触碰。physical hit/net/landing/legal-return 的 Phase-B 账本仍未实现与
运行；本 receipt 永远不得冒充物理回球证据。
复现命令见[操作文档](../../operations/run_sparse_reward_milestone_classifier.md)。
