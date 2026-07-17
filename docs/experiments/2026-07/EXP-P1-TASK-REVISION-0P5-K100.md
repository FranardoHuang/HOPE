# EXP-P1-TASK-REVISION-0P5-K100 — 0.5 秒能否实际接住

- 状态：`ready`（source gate 通过；Pod runtime 与行为卷未运行）
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
| activation | `configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json`，SHA `996775d6…7cfb6` |
| 持久执行器 | `scripts/run_phase1_task_revision_0p5_exam.py`，SHA `c2ce2784…1b63` |
| 结果资格 | Isaac diagnostic only；`formal_evidence_eligible=false`、`evaluation_contract_exact=false` |
| 重试/信号 | 自动重试=false；trainer/robot/broad signal=false |

## source gate 与当前阻塞

- 专项回归：`35 passed, 1 skipped`。skip 是本地主机没有可写的 delegated cgroup-v2 child；它不能替代
  Pod 上的实际探针。
- 持久执行器把 activation、queue、checkpoint、hard contract、claim/binding、运行时、题表与输出路径
  绑定；唯一 `launch` 使用不可覆盖提交链，唯一 `inspect` 只读。监督器只拥有本次 evaluator，并要求
  guardian、真实 `cgroup.kill` 探针、精确 PID/PGID/starttime、心跳、总 deadline、终态清理确认。
- **当前仍为 `NO_LAUNCH`：** Pod 上 delegated cgroup-v2 写入/迁移/`cgroup.kill` 探针尚未实测，正式
  launch 尚未执行，也没有 behavior score。source gate 通过不能写成 0.5 秒能力通过。
- 2026-07-17 13:05Z 的单次只读资源快照为 Pod2 `0 trainer / 3 GPU 空闲`；Pod1 未核，记为
  `UNKNOWN`，不从旧状态推断。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| exact-0.5 中性代表卷 `phase1_task_revision_0p5_k100_p2_equal_reward_model5700_v1` | `ready / NO_LAUNCH` | `model_5700` | source `35 passed, 1 skipped` | 尚无 | Pod probe、launch、K100 score 均未跑 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

## 决定

- 决定：`inconclusive`
- 是否已纳入当前 setting：`no`
- 下一门：按[操作文档](../../operations/run_phase1_task_revision_0p5_exam.md)在 Pod2 先通过实际 runtime
  probe，再且仅一次启动；终档必须由只读 `inspect` 验证。之后还要在 vendor MuJoCo 同题复核。

