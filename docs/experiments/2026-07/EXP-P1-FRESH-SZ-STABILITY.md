# EXP-P1-FRESH-SZ-STABILITY — 正式 setting 是否稳定？

- 状态：blocked
- Runtime 状态：`prepared_not_started`（model-4000 后续卷的 runtime contract 已生成，job 未启动）
- 阶段/轴：S1 / setting 与 seed 稳定性
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：model-2000 为 E4；model-4000 只有 readiness/prepare 证据，尚无行为等级
- 最后复核：2026-07-13

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

问题：fresh `SZ`——`v4rg_runtime_order_v3` schema-2 正手/反手、schema-3 bank、179-D `shared_plus_y`、零摩擦执行、
PPO（批量策略优化）从零训练——能否在同一份不可变 MuJoCo K100 上，跨独立 seed 与 milestone 保持单拍性能？

| Milestone | Seed 1 | Seed 2 | Seed 3 | Seed 4 | 决定 |
| --- | ---: | ---: | ---: | ---: | --- |
| model-2000 exact K100 | 83/100 | 100/100 | 100/100 | 20/100 | worst-seed、spread 和 worst-side 稳定性规则失败 |
| model-4000 matched K100 | 已知 50/100 | 未测 | 未测 | 未测 | 两 Pod 均已 prepare；`jobs_started=0`；family 稳定性在数学上已不可能 |

| 动作 | 一次挥拍物理不摔 | 一次挥拍解析击球 | 一次挥拍解析上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 200/200 | 137/200 | 133/200 | 未测 | 未测 | 未测 |
| 反手 | 200/200 | 170/200 | 170/200 | 未测 | 未测 | 未测 |

决定：最接近正式 setting 的方案**不具备 seed 稳定性**，不是已接受的 baseline。不得推广最佳 seed。
所有 attempt 都通过非物理 tracking guard/reset 路径结束，因此“不摔”不能证明连续恢复。
击球/上台列来自解析接触与落台推演，不是 physical ball 回放；拍面符号盲区见
[Face-sign forensic](EXP-P1-FACE-SIGN-FORENSIC.md)。

## Model-4000 后续卷就绪状态（2026-07-13）

- Pod1 seed1/3 与 Pod2 seed2/4 的 no-clobber readiness audit 已通过，file SHA 分别为
  `3fc325e1...247b8` 和 `4f25786b...565f7`。
- 两份 audit 的 exact union 生成 all-four activation：file SHA `9dea76c2...ce704`，
  content SHA `eaa92ca2...aa4fb`，四个 seed 全部覆盖，且明确保留
  `judges_started=0`。
- 两个 Pod 的 activation-consuming `contract-check` 均通过；紧接着的快照中没有
  child judge、MuJoCo evaluator、play/Kit 进程或 Kit-lock holder。
- 两个 Pod 随后各执行一次 activation-bound no-clobber `prepare`。Pod1 runtime
  file/content SHA 为 `2b76a5a...8201e` / `36e878f0...5ba73`，Pod2 为
  `dbecc102...d1c9b` / `91a0070a...30794`。
- 两份 runtime contract 都保持 `prepared_not_started`、`jobs_started=0`、
  `auto_start=false`；再次核对四 checkpoint、K100 与 Kit 锁后仍无 child judge。
- 一次性持久监督器的 source gate 已通过独立红队：P0/P1 均为 0，supervisor focused 为
  `24 passed`，queue+consumer+supervisor 合计 `64 passed`。Linux `/proc` fake-runner 冒烟仍是
  真实 q50 前置；详见[启动执行子记录](../phase1_fresh_sz_model4000_q50_20260713.md)。
- 这只完成了执行纸面准备，不是行为成绩。尚未运行 `run`、judge、aggregate，
  也没有 trainer signal 或真机动作。

证据：

- `configs/phase1_fresh_SZ_model2000_seed_stability_q50_pod1_result_20260711.json`
- `configs/phase1_fresh_SZ_model2000_seed_stability_q50_pod2_result_20260711.json`
- `configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`
- [Pod1 readiness audit](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_ready_audit_20260713.json)
- [Pod2 readiness audit](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_ready_audit_20260712.json)
- [All-four activation](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_activation_20260713.json)
- [Pod1 prepared runtime contract](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_runtime_contract_prepared_20260713.json)
- [Pod2 prepared runtime contract](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_runtime_contract_prepared_20260713.json)
- [Fresh lineage](../../PHASE1_FRESH_LINEAGE_2026-07-11.md)
- [Model-4000 操作](../../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md)
- [持久启动执行子记录](../phase1_fresh_sz_model4000_q50_20260713.md)

下一门：先在 Linux 上用 fake runner 验证真实 `/proc` 身份闭环，再按 seed1→3、seed2→4
运行同一 K100；结果只能判 seed4 晚熟或持续弱。
