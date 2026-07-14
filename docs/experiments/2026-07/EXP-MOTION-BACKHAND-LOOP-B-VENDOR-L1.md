# EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1 — 反手拉 B 整轨自碰/自打门

- 状态：`complete_cpu_vendor_l1_safety_pass_downstream_blocked`
- 阶段/轴：新动作库 / [`vendor L1 safety audit`](../../DEFINITIONS.md#motion-vendor-l1-safety)
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E2（exact Pod2 CPU dry-run + 唯一 no-clobber vendor L1 certificate）

## 问题与冻结输入

只问已通过 L0 的反手拉 B，在 exact vendor MuJoCo collision model 中整轨是否出现机器人自碰或
球拍/拍柄打到机器人。预注册
[`motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json`](../../../configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json)
SHA-256 为 `fd47a3989a9fa786d87b86382fd5923d3682860c88d365c0a7151d487f78a770`，绑定：

- L0 certificate SHA-256 `60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`；
- B schema-2 NPZ SHA-256 `e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc`；
- vendor MJCF `2ab1cd31...feb97`、75-file closure `e0381752...962de`、compiled collision
  contract `18e7f6ff...386e5`；
- Python/NumPy/MuJoCo `3.12.3 / 2.5.0 / 3.10.0` exact CPU runtime。

C 不在 manifest 中，不读取、不消费，也不是 fallback。

## 审计合同

复用现有 `root_xyz` linear、root quaternion shortest-arc slerp、joint linear 插值。B 的 151 个
50 Hz 原帧先按名字从 schema-2 的 runtime/Isaac 关节顺序显式置换到
`ground_gmr_pkl.A3_GMR_JOINT_NAMES`（vendor MJCF hinge 顺序），再每段取 8 个子步，共 1201 个
400 Hz 有限样本。两套名字必须各为 31 个唯一名字且集合完全相同；置换只移动 float32 列、不修改数值，
实际 `target_from_source_indices` 写入 runtime 结果。任一 dense sample 出现以下事件，整条动作
hard fail，不能由其他样本、reward 或分数补偿：

1. exact compiled model 的 enabled robot geom 之间穿透超过 `1e-6 m`；
2. `right_racket_collision` 或 `right_racket_handle_collision` 距 head/neck、trunk、对侧臂、右肩三轴/
   右肘或下肢小于 `5 mm`。

`5–20 mm` 只登记 warning。右腕/手/拍安装链从 `5 mm` proximity pair 排除，避免把机械安装本身判成
自打；但 enabled-robot contact 中实际穿透仍 hard fail。危险插值样本保守标记相邻两个 source frame，
同时直接否决整条动作。

hard gate 不使用 `geom_clearance()` 默认 `1e-4 m` 二分 midpoint 做边界决策，而直接使用同一 exact helper
的 saturation predicate：`distance < 5 mm` 当且仅当 `_far(..., 0.005)` 为 false。4.99 mm 必败，
5.00/5.01 mm 不触发 hard gate；报告用的 minimum 另以 `1e-6 m` 二分记录，不参与 pass/fail。

这是 finite dense sampling，不是数学连续时间 swept-volume certificate。桌/网、ground（由 exact L0
certificate 继承）、动力学、平衡、TOPP、击球效果、RL、Gate3 和真机全部不在本门。

## 当前结果与下一步

首次 Pod2 CPU `dry-run` 在载入 grounding helper 时以
`ModuleNotFoundError: No module named 'ground_gmr_pkl_for_vendor_l1'` 停止。根因不是 B 动作、MuJoCo
碰撞或模型失败，而是 harness 给 exact path helper 使用了只供本审计隔离的私有 module name；该名字
本来就不在 `sys.path`，却误调用了 `import_module(name)`。失败发生在轨迹审计前，没有 certificate，
不能写成 vendor L1 行为失败或通过。

第一轮修复后的 loader 按已冻结的
bytes/SHA 从 exact path 加载并注册私有 name；执行前后都复核内容与 `__file__`，module body 失败时恢复
调用方原有 `sys.modules` entry，没有原 entry 时清掉半初始化 module。focused test 直接加载真实
`scripts/ground_gmr_pkl.py` 到故障中的 private alias，并覆盖 SHA drift、stale module、body exception
的正负例。

该修复合入 `main@b75204d` 后，第二次 Pod2 CPU `dry-run` 进入 range gate，却报告 dense frame `704`
的 `left_ankle_pitch_joint` 超限 `0.656861334 rad`。只读复算证明这也是 harness 假拒绝：dense 704
精确等于 source frame `88.0`，没有插值新极值；原数组 column 23 实际是 runtime-order
`left_elbow_joint=1.1804603338 rad`，却被 GMR-order column 23 误标成 ankle，其上限 `0.523599`，差值
正好为 `0.656861334`。真正 ankle 在 runtime column 14，为 `-0.5744639635 rad`，合法区间
`[-0.907571, 0.523599]`；真正 elbow 区间 `[-0.959931, 1.74533]` 也包含 `1.1804603338`。L0 使用
runtime names 逐关节查 MJCF id，所以 exact certificate 正确报告 `max_excess_rad=0.0`。这不是 B 动作、
单位、插值或 MJCF range 失败，不能放宽阈值或修改 B/C。

当前 validator SHA-256 为
`6368bda72e57e646324c6ff7281a50c00c339ddedb954941e6455712954fac2e`。它在 densify/range/qpos
之前执行上述 fail-closed name bijection，并把 source/target names 与 31-index permutation 写入结果；
重复、缺失、额外名字或 plan adapter 语义漂移均 fail closed。source/static gate 与 L0 回归合跑
`33 passed`；其余反例继续证明
4.99/5.00/5.01 mm 边界、右肘旧组漏报、
任一 self-collision、continuous-time flag/阈值放宽、C 混入、缺 output parent、dangling symlink target
和 certificate overwrite 均 fail closed。

两个 harness 根因进入 `main@7dec698` 后，Pod2 clean detached exact source 先通过 full `dry-run`，再执行
唯一一次 `O_EXCL` audit。certificate SHA-256 为
`6840df34a6aa6e5636192c705a8ecaa563f751658fe538df428bc317c858db60`，完成时间
`2026-07-15 05:00 CST`。1201 个有限密扫样本中：

- enabled robot self-collision hard events=`0`；
- 球拍/拍柄—机器人 `<5 mm` hard events=`0`，`5–20 mm` warning=`0`；
- 最小余隙 `0.1382918358 m`，发生在 source frame `75.0` 的拍柄—右肘 pair；
- joint-order permutation 与两份完整名字表已写入 certificate，`mj_step_calls=0`；
- `continuous_time_certificate=false`，不能把有限 400 Hz 扫描改写为数学连续时间证明。

这令 `vendor_l1_complete=true`、`table_net_authorized=true`，但 dynamics/simulator/training/formal-motion/
hardware 仍全 false。B 下一门是独立整轨桌网余隙，不直接进入 RL；C 继续保持未消费后备。
