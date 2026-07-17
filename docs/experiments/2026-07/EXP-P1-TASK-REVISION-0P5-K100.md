# EXP-P1-TASK-REVISION-0P5-K100 — 0.5 秒能否实际接住

- 状态：`Partial / OPEN`（v2 暴露 native-clock 顺序错误；v3/v4 被冗余文本门假拒绝且零 namespace；v5 待唯一运行）
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
| activation | v5 `configs/phase1_task_revision_0p5_exam_activation_v5_20260718.json`，SHA `cb4b8e67…a51886`；v1/v2/v3/v4 永久禁止重发 |
| 持久执行器 | `scripts/run_phase1_task_revision_0p5_exam.py`，SHA `c0fe1555…7c55c` |
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
  `f8c3be8b…a9e28`，没有 scorecard；因此不是 `0/100`，而是“0 题执行 / 行为未测”。它随后自然结束为
  `failed_no_retry`：terminal file/content SHA=`2d3a9c7d…4894d/76cd8121…bb7d`，guardian=`D0`，owned
  cgroup 已在 `populated=0` 后删除；历史 supervisor `502505`、guardian `502506`、evaluator `502542`
  均 absent。一次 exact-stop 尝试在发 signal/写 intent 前因目标已不 live 而 fail closed；禁止重放该
  stop，也禁止对旧 PID 再发 signal。
- v3 把顺序改为：关闭保存的 planner clock owner，安装 native external command，逐项验证第 0 帧姿态
  不变且速度严格为零，最后才激活 retiming。K100 harness 专项为 `60 passed, 1 skipped`；与物化器、
  timing adapter 合并为 `88 passed, 1 skipped`。skip 仍是本地主机没有 delegated cgroup-v2 child，不能
  替代 Pod runtime。
- v3 不再等待不存在的人工 stop result。它在 attempt 写入前、attempt 发布前和资源检查后/创建 cgroup
  前三次稳定重读自然 terminal/错误日志，绑定两份 SHA，并要求 stop intent/result、三历史 PID 与旧
  cgroup 全部 absent；`/proc` permission/error 不能冒充 absent。heartbeat freshness 只取最后一个带换行
  且 strict UTC 的完整 JSONL record；完整 terminal 还必须逐字节绑定 final receipt。
- 2026-07-18 的 Pod2 单次只读取证已确认 v2 全链 absent；source gate、错误日志和自然终档都不能写成
  0.5 秒能力结果。随后 v3 唯一 launch 因旧校验要求 failure reason 在整份 traceback 中恰好出现一次而
  假拒绝；实际 raw SHA 完全一致、文案正常出现两次，且 v3 attempt/state/output/cgroup 全部 absent。
  v3 receipt `32525b0f…bbc82d3` 将它冻结为 `failed_no_retry`。v4 改为 raw SHA + 36,059 bytes + 最后
  异常行精确匹配，但真实类名带模块前缀，仍在零 namespace 阶段假拒绝。v4 receipt
  `b7ca44ba…7afed7` 将其冻结。v5 删除全部文本解析，仅保留 raw SHA+bytes；合并回归为
  `89 passed, 1 skipped`。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| v1 exact-0.5 中性代表卷 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1` | `failed_no_retry` | `model_5700` | Pod2 `rc=2`、约 `5.779 s`；receipt `f53a6813…1728` | 无 | bank 缺失；supervisor/cgroup/ACK/evaluator 均未创建；永久禁止重发 |
| 资产恢复版 v2 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_asset_restored_v2` | `failed_no_retry / naturally closed` | `model_5700` | log `f8c3be8b…a9e28`；terminal `2d3a9c7d…4894d`；D0/旧进程与 cgroup absent | 无 scorecard | 0 题执行；永久禁止重发或再 stop |
| native-clock 修正版 v3 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_native_clock_v3` | `failed_no_retry / zero remote namespace` | `model_5700` | log SHA 未漂移；reason 两次；receipt `32525b0f…bbc82d3` | 无 | source gate 假拒绝；禁止重发 |
| 最小门修正版 v4 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_native_clock_v4` | `source ready / not launched` | `model_5700` | runner/activation `1e12d791…d99d7c7/4f93c0c3…ccd864`；`88 passed, 1 skipped` | 尚无 | 只修日志语义门并使用 fresh namespace；待唯一 launch |
| SHA-only 修正版 v5 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_native_clock_v5` | `source ready / not launched` | `model_5700` | runner/activation `c0fe1555…7c55c/cb4b8e67…a51886`；`89 passed, 1 skipped` | 尚无 | 不再解析 traceback；待唯一 launch |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

## 决定

- 决定：`inconclusive`
- 是否已纳入当前 setting：`no`
- 下一门：按[操作文档](../../operations/run_phase1_task_revision_0p5_exam.md)在 Pod2 以 fresh v5 namespace
  唯一启动并用只读 `inspect` 验证；v1/v2/v3/v4 永久禁止重发。之后还要在 vendor MuJoCo 同题复核。
