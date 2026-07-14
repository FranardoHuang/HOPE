# EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1 — 反手拉 B 整轨自碰/自打门

- 状态：source/static gate 通过；首次 exact runtime `dry-run` 在 helper import 处 fail closed，修复已通过源码回归，等待 clean runtime 重跑；certificate 不存在
- 阶段/轴：新动作库 / [`vendor L1 safety audit`](../../DEFINITIONS.md#motion-vendor-l1-safety)
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E1（源码、预注册和合成反例；无真实 B L1 行为结果）

## 问题与冻结输入

只问已通过 L0 的反手拉 B，在 exact vendor MuJoCo collision model 中整轨是否出现机器人自碰或
球拍/拍柄打到机器人。预注册
[`motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json`](../../../configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json)
SHA-256 为 `8b824e93eda96ca61103b4fb519c3896e009929db3ef4e643208ae4886810b9d`，绑定：

- L0 certificate SHA-256 `60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`；
- B schema-2 NPZ SHA-256 `e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc`；
- vendor MJCF `2ab1cd31...feb97`、75-file closure `e0381752...962de`、compiled collision
  contract `18e7f6ff...386e5`；
- Python/NumPy/MuJoCo `3.12.3 / 2.5.0 / 3.10.0` exact CPU runtime。

C 不在 manifest 中，不读取、不消费，也不是 fallback。

## 审计合同

复用现有 `root_xyz` linear、root quaternion shortest-arc slerp、joint linear 插值。B 的 151 个
50 Hz 原帧每段取 8 个子步，共 1201 个 400 Hz 有限样本。任一 dense sample 出现以下事件，整条动作
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

修复后的 validator SHA-256 为
`daa1f1bb700e9e0424101597900abc8cab013cc46d155f23f11c573d02bd66be`。loader 现在按已冻结的
bytes/SHA 从 exact path 加载并注册私有 name；执行前后都复核内容与 `__file__`，module body 失败时恢复
调用方原有 `sys.modules` entry，没有原 entry 时清掉半初始化 module。focused test 直接加载真实
`scripts/ground_gmr_pkl.py` 到故障中的 private alias，并覆盖 SHA drift、stale module、body exception
的正负例。source/static gate 与 L0 回归合跑 `28 passed`；其余反例继续证明
4.99/5.00/5.01 mm 边界、右肘旧组漏报、
任一 self-collision、continuous-time flag/阈值放宽、C 混入、缺 output parent、dangling symlink target
和 certificate overwrite 均 fail closed。

本修复任务没有连接 Pod、没有读私有 certificate/NPZ、没有运行 MuJoCo runtime，也没有创建输出目录或
certificate。因此不能声称 B 已过 vendor L1；G08 继续 Partial。代码合入并由人复核后，下一步按
[操作文档](../../operations/run_motion_backhand_loop_b_vendor_l1_safety.md)在 exact CPU runtime 先做一次
`dry-run`，保全逐帧结果；只有通过且另有显式发布授权，才允许唯一 no-clobber `audit`。通过证书只解锁
独立桌网整轨门，不直接解锁动力学或训练。
