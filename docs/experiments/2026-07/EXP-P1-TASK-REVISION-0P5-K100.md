# EXP-P1-TASK-REVISION-0P5-K100 — 0.5 秒能否实际接住

- 状态：`Partial / OPEN`（v2 在首题前暴露 native-clock 顺序错误并卡在清理；v3 被 exact-stop receipt 阻塞）
- 阶段/轴：Phase-1 / 准备时间与同球 revision
- 集成小目标：用固定 0.5 秒题表直接测回球能力，而不是从 Reward 或动作倍率推断
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-17

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的 [`K100`](../../DEFINITIONS.md#q50-and-k100)
是正反手合计 100 题；[`0.5 秒时序卷`](../../DEFINITIONS.md#timing-exam-0p5)要求每题从第 0 帧零速度
准备态开始，并在 25 个 50 Hz tick 后触球，没触球、来不及、摔倒与未上台都保留在分母。

## 问题与假设

问题：`taskrev_p2_equal_reward@model_5700` 在 exact 0.5 秒准备时间下，能否稳定触球并合法回台？

本卷不把它称为胜者；它只是 parent+1000 时的中性代表。若 100 题行为账没有证明实际击球/回台，或
运行合同、清理证明不完整，则“现役 policy 已能接 0.5 秒球”的假设被拒绝或保持 inconclusive。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 选定 checkpoint | `taskrev_p2_equal_reward@model_5700`，Pod2；由 milestone/behavior receipt 绑定 |
| 题表 | fixed exact-25-tick K100；100 题、每侧 50，缺失仍计入分母 |
| activation | v3 `configs/phase1_task_revision_0p5_exam_activation_v3_20260717.json`，SHA `0b8bc60f…6916`；v2 activation `2b91248b…0626` 只可用于 exact stop |
| 持久执行器 | `scripts/run_phase1_task_revision_0p5_exam.py`，SHA `cf3a765f…1e7f` |
| 输入资产 | bank `63,643 B / 60e1a7ad…d1ca`；重绑定报告 `18,795 B / dd4332ed…ad0` |
| v1 失败墓碑 | `configs/phase1_task_revision_0p5_exam_v1_failure_20260717.json`，SHA `f53a6813…1728` |
| 结果资格 | Isaac diagnostic only；`formal_evidence_eligible=false`、`evaluation_contract_exact=false` |
| 重试/信号 | 自动重试=false；trainer/robot/broad signal=false |

## v1/v2 真实失败、v3 source gate 与当前阻塞

- v1 是唯一一次 Pod2 launch：`rc=2`、墙钟约 `5.779 s`，在 `validate_inputs` 因 immutable exam bank
  缺失而 fail closed。它没有物化 supervisor、delegated cgroup、commit ACK 或 evaluator，也没有 signal
  trainer；失败 receipt 为 `f53a6813…1728`。这份 activation 已消费且 `retry_authorized=false`，永久禁止
  重发。
- 缺失的 bank 与 rebind report 后来按 exact SHA/size、`0444`、no-clobber 恢复到 Pod2：分别为
  `63,643 B / 60e1a7ad…d1ca` 与 `18,795 B / dd4332ed…ad0`。这是输入资产恢复，不是 v1 retry，也不是
  行为通过。
- v2 已在 Pod2 唯一启动，但 evaluator 在任何题目执行前以
  `IsaacBankExamError: timing rider requires a native-clock command before activation` 失败。日志 SHA 为
  `f8c3be8b…a9e28`，没有 scorecard；因此不是 `0/100`，而是“0 题执行 / 行为未测”。其 supervisor
  `502505` 与 evaluator `502542` 仍存活并低利用率卡在清理，禁止重复 launch。
- v3 把顺序改为：关闭保存的 planner clock owner，安装 native external command，逐项验证第 0 帧姿态
  不变且速度严格为零，最后才激活 retiming。K100 harness 与 timing adapter 合并专项回归为
  `61 passed, 1 skipped`；skip 仍是本地主机没有 delegated cgroup-v2 child，不能替代 Pod runtime。
- v2 exact-stop consumer 先两次核验 activation/queue、PID/PGID/SID/start ticks、ACK/guardian/cgroup、
  失败日志 SHA 与唯一错误原因，O_EXCL 写 stop intent 后只向 supervisor `502505` 发一次 `SIGTERM`；
  不直接 signal evaluator、不调用 `cgroup.kill`、不发 `SIGKILL`、不重试。v3 在任何 consumption 写入前
  必须读到 `failed_no_retry + D0/K0 + cgroup removed` 的 exact stop result。
- 2026-07-17 23:44 CST 的 Pod2 单次只读快照为 `0 trainer`；v2 evaluator 同时占三张 GPU 但利用率均约
  `1%`。source gate、错误日志与后续 stop 都不能写成 0.5 秒能力结果。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| v1 exact-0.5 中性代表卷 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1` | `failed_no_retry` | `model_5700` | Pod2 `rc=2`、约 `5.779 s`；receipt `f53a6813…1728` | 无 | bank 缺失；supervisor/cgroup/ACK/evaluator 均未创建；永久禁止重发 |
| 资产恢复版 v2 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_asset_restored_v2` | `failed before question 1 / cleanup pending` | `model_5700` | log `f8c3be8b…a9e28`；唯一 native-clock 顺序错误 | 无 scorecard | 0 题执行；supervisor/evaluator 尚待 exact stop；永久禁止重发 |
| native-clock 修正版 v3 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_native_clock_v3` | `blocked on v2 exact stop` | `model_5700` | source `61 passed, 1 skipped`；fresh activation/output/state | 尚无 | 只有 v2 stop result 完整后才可唯一启动 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

## 决定

- 决定：`inconclusive`
- 是否已纳入当前 setting：`no`
- 下一门：按[操作文档](../../operations/run_phase1_task_revision_0p5_exam.md)先且仅一次 exact-stop v2；确认
  `failed_no_retry + D0/K0 + cgroup removed` 后，才可在 Pod2 以 fresh v3 namespace 唯一启动并用只读
  `inspect` 验证。v1/v2 永久禁止重发；之后还要在 vendor MuJoCo 同题复核。
