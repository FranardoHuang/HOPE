# EXP-P1-FRESH-SZ-STABILITY — 正式 setting 是否稳定？

- 状态：completed（两个 milestone 均拒绝 seed-stable baseline）
- Runtime 状态：`terminal_result_validated`；Linux fake-runner 冒烟、两 Pod 一次性启动、四 seed
  K100 与正式 aggregate 均已完成，且没有遗留 judge/Kit 进程
- Trainer 状态：四个 formal seed 均已按 2026-07-13 负责人事后运营资源决定停止；四份
  model-4000 checkpoint 均已在停止前存在并通过 readiness，因此已完成后续卷输入未变
- 阶段/轴：S1 / setting 与 seed 稳定性
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：model-2000/model-4000 均为 E4 Python BankExam 解析诊断；不是 physical return
  或厂商 Gate3/Gate3B
- 最后复核：2026-07-13

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

问题：fresh `SZ`——`v4rg_runtime_order_v3` schema-2 正手/反手、schema-3 bank、179-D `shared_plus_y`、零摩擦执行、
PPO（批量策略优化）从零训练——能否在同一份不可变 MuJoCo K100 上，跨独立 seed 与 milestone 保持单拍性能？

| Milestone | Seed 1 | Seed 2 | Seed 3 | Seed 4 | 决定 |
| --- | ---: | ---: | ---: | ---: | --- |
| model-2000 exact K100 | 83/100 | 100/100 | 100/100 | 20/100 | worst-seed、spread 和 worst-side 稳定性规则失败 |
| model-4000 matched K100 | 50/100 | 88/100 | 98/100 | 0/100 | median `.69`、worst `.00`、spread `.98`、worst-side `.00`；四项全 FAIL |

| 动作 | 一次挥拍物理不摔 | 一次挥拍解析击球 | 一次挥拍解析上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 200/200 | 137/200 | 133/200 | 未测 | 未测 | 未测 |
| 反手 | 200/200 | 170/200 | 170/200 | 未测 | 未测 | 未测 |

决定：最接近正式 setting 的方案**不具备 seed 稳定性**，不是已接受的 baseline。不得推广最佳 seed。
model-4000 上 seed4 为正反手 `0/50,0/50`，并有 21 次 `fall_root_z` 物理倾倒，因此按
未改阈值判为 `persistent_weakness_through_model4000`，不是晚熟。这是 seed4 的特定问题，
不得改写成“四个 seed 都物理失衡”。
所有 attempt 都通过非物理 tracking guard/reset 路径结束，因此“不摔”不能证明连续恢复。
击球/上台列来自解析接触与落台推演，不是 physical ball 回放；拍面符号盲区见
[Face-sign forensic](EXP-P1-FACE-SIGN-FORENSIC.md)。

## 2026-07-13 trainer 资源裁剪边界

负责人在查看 q50 与连续 q10 曲线后，先另行决定不再为明显持续塌陷的 seed1/2/4 trainer 购买剩余
迭代。seed1/2/4 的最后保留 checkpoint 分别为 `model_12100.pt`、
`model_13000.pt`、`model_12900.pt`，均在信号前验证 finite、schema-3、fresh lineage 与相邻
hard-contract SHA。完整 checkpoint SHA、PGID、方向曲线和精确信号记录见
[拍面×plant 广度矩阵](EXP-P1-FACE-PLANT-SCALEOUT.md)。

这不是本实验预注册的停止规则。model-2000/model-4000 q50 合同中的
`whole_arm_stop_allowed=false` 仍按原义保留，q10 也仍不能晋级；本次只记录负责人后续做出的算力
运营决定。已准备的四 seed model-4000 K100 后续卷继续使用停止前已经内容绑定的四份 checkpoint，
可判 seed4 4k 是晚熟还是持续弱，但不能把这次停止反写成 q50 授权。

model-4000 结果后，seed3 最近方向卷也显示 4k→6k→8k parsed `1.00→.50→.00`，且其
正手 signed composite 始终为 0。因此它与其他剩余诊断臂一起进入第二波负责人运营停止，
最后保留 `model_13800.pt`（SHA `478efa8d...c9e6`）。同样，这不是改写 q10/q50 的 stop 权限；
完整第二波证据见 [广度矩阵](EXP-P1-FACE-PLANT-SCALEOUT.md)。

## Model-4000 后续卷结果（2026-07-13）

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
- 一次性持久监督器先通过 Linux `/proc` fake-runner 冒烟；该冒烟只启动假 runner，
  故其无 terminal result 的 `committed_child_failed` 是预期负控，没有启动 judge/Kit。
- 正式启动后，Pod1/Pod2 分别在 `2026-07-13T06:20:51Z` /
  `2026-07-13T06:20:33Z` 生成 `terminal_result_validated` 的 pod result；file SHA 为
  `02d0e58d...645d` / `d31323a6...4e6f`。每份均重验 exactness、fresh lineage、
  checkpoint↔hard-contract、K100/50-per-side、report/summary/attempt-ledger SHA。
- 只在一个 control host 上执行一次正式 aggregate：file SHA
  `1ba88e39e8395b8edce9365475404eafa660d4bf1b61d640a21d1d7cbb75d195`，content SHA
  `226e6050c3789ebbc3145d84ca40225ab0fe9e1b868143de8ea80ad5caab648d`。独立复算确认
  seed `1/2/3/4=.50/.88/.98/.00`，median `.69`，worst `.00`，spread `.98`，
  worst-side `.00`，四项阈值全失败。
- 更严格的 signed-face 切面把正手解析分推翻：seed1/2/3 的正手 raw-A 有符号
  法向误差为 `164.86°/172.33°/174.35°`，正手 strike composite 全为 `0/50`；
  但旧 parsed return 仍分别记为 `0/50,38/50,48/50`。因此 `.88/.98` 不能用于晋级，
  必须先修 signed-face scorer 并同卷复判。
- aggregate 中冻结的 `continue_all_arms_unmodified` 是 q50 原合同的不发信号语义，
  不是当前 runtime 事实。seed1/2/4 的 trainer 在另行负责人运营决定下早已停止；
  该决定不来自 q50 产物，也没有改写 `whole_arm_stop_allowed=false`。

证据：

- `configs/phase1_fresh_SZ_model2000_seed_stability_q50_pod1_result_20260711.json`
- `configs/phase1_fresh_SZ_model2000_seed_stability_q50_pod2_result_20260711.json`
- `configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`
- [Pod1 readiness audit](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_ready_audit_20260713.json)
- [Pod2 readiness audit](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_ready_audit_20260712.json)
- [All-four activation](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_activation_20260713.json)
- [Pod1 prepared runtime contract](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_runtime_contract_prepared_20260713.json)
- [Pod2 prepared runtime contract](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_runtime_contract_prepared_20260713.json)
- [Pod1 model-4000 result](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_result_20260713.json)
- [Pod2 model-4000 result](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_result_20260713.json)
- [Model-4000 aggregate](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_aggregate_result_20260713.json)
- [Fresh lineage](../../PHASE1_FRESH_LINEAGE_2026-07-11.md)
- [Model-4000 操作](../../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md)
- [持久启动执行子记录](../phase1_fresh_sz_model4000_q50_20260713.md)

下一门：不再为该 `SZ` family 扩 seed 或续训买晋级证据。先完成 `n/-n` 负控、
有符号 scorer 修正和同卷复判；然后才能判新 setting/checkpoint。厂商 Gate3/Gate3B 仍是
最终环境。
