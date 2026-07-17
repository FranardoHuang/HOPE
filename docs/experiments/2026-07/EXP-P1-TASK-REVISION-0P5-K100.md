# EXP-P1-TASK-REVISION-0P5-K100 — 0.5 秒能否实际接住

- 状态：`Partial / OPEN`（v1 输入门已真实失败且永久冻结；资产恢复后的 v2 尚未启动）
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
| activation | v2 `configs/phase1_task_revision_0p5_exam_activation_v2_20260717.json`，SHA `2b91248b…0626` |
| 持久执行器 | `scripts/run_phase1_task_revision_0p5_exam.py`，v2 SHA `be17289c…cc59` |
| 输入资产 | bank `63,643 B / 60e1a7ad…d1ca`；重绑定报告 `18,795 B / dd4332ed…ad0` |
| v1 失败墓碑 | `configs/phase1_task_revision_0p5_exam_v1_failure_20260717.json`，SHA `f53a6813…1728` |
| 结果资格 | Isaac diagnostic only；`formal_evidence_eligible=false`、`evaluation_contract_exact=false` |
| 重试/信号 | 自动重试=false；trainer/robot/broad signal=false |

## v1 真实失败、v2 source gate 与当前阻塞

- v1 是唯一一次 Pod2 launch：`rc=2`、墙钟约 `5.779 s`，在 `validate_inputs` 因 immutable exam bank
  缺失而 fail closed。它没有物化 supervisor、delegated cgroup、commit ACK 或 evaluator，也没有 signal
  trainer；失败 receipt 为 `f53a6813…1728`。这份 activation 已消费且 `retry_authorized=false`，永久禁止
  重发。
- 缺失的 bank 与 rebind report 后来按 exact SHA/size、`0444`、no-clobber 恢复到 Pod2：分别为
  `63,643 B / 60e1a7ad…d1ca` 与 `18,795 B / dd4332ed…ad0`。这是输入资产恢复，不是 v1 retry，也不是
  行为通过。
- v2 专项回归：`41 passed, 1 skipped`。skip 是本地主机没有可写的 delegated cgroup-v2 child；它不能替代
  Pod 上的实际探针。
- v2 在任何远端消费写入前依次验证资产 SHA/size、v1 failure tombstone 和 fresh v2 attempt marker，再
  执行其余 preflight；state/output 使用 `asset_restored_v2` 新 namespace，不会覆盖 v1。持久执行器把
  activation、queue、checkpoint、hard contract、claim/binding、运行时、题表与输出路径绑定；唯一
  `launch` 使用不可覆盖提交链，唯一 `inspect` 只读。监督器只拥有本次 evaluator，并要求
  guardian、真实 `cgroup.kill` 探针、精确 PID/PGID/starttime、心跳、总 deadline、终态清理确认。
- **当前 v2 仍为 `NO_LAUNCH`：** v2 尚未在 Pod2 执行，Pod delegated cgroup-v2 探针与行为卷也没有
  v2 结果。source gate 和资产恢复不能写成 0.5 秒能力通过。
- 2026-07-17 13:05Z 的单次只读资源快照为 Pod2 `0 trainer / 3 GPU 空闲`；Pod1 未核，记为
  `UNKNOWN`，不从旧状态推断。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| v1 exact-0.5 中性代表卷 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1` | `failed_no_retry` | `model_5700` | Pod2 `rc=2`、约 `5.779 s`；receipt `f53a6813…1728` | 无 | bank 缺失；supervisor/cgroup/ACK/evaluator 均未创建；永久禁止重发 |
| 资产恢复版 v2 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_asset_restored_v2` | `ready / NO_LAUNCH` | `model_5700` | source `41 passed, 1 skipped`；输入 SHA/size/0444 已恢复 | 尚无 | fresh namespace；v2 Pod launch 与 K100 score 均未跑 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

## 决定

- 决定：`inconclusive`
- 是否已纳入当前 setting：`no`
- 下一门：按[操作文档](../../operations/run_phase1_task_revision_0p5_exam.md)只使用 v2，先生成 plan，再在
  Pod2 且仅一次启动；终档必须由只读 `inspect` 验证。v1 永久禁止。之后还要在 vendor MuJoCo 同题复核。
