# EXP-P1-HISTORICAL-SCHEMA3 — 同题考卷尺子能否区分候选？

- 状态：completed
- 阶段/轴：共用判分基础 + 课程阶段 1 / 评估尺子
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：E4（诊断）

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

由评估器所有的不可变 schedule、all-attempt 分母和同一 virtual-return scorer，成功区分了 Isaac 和 MuJoCo 中
已知良好/常见/已知失败的历史候选。该尺子可用于诊断排名。所有被评估模型仍为
`evaluation_contract_exact=false`，因此任何分数都不会建立正式 baseline。

同题考卷中，M3 old/S1 在 Isaac 中均约为 `.99`，而 MuJoCo 给出 `.42/1.00`。
该结果定位了跨引擎/仪器问题，但没有修复它。

权威结果：[`PHASE1_SCHEMA3_RESULTS_2026-07-11.md`](../../archive/PHASE1_SCHEMA3_RESULTS_2026-07-11.md)。
