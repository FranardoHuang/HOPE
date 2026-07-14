# EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1 — 反手拉 B 整轨自碰/自打门

- 状态：source/static gate 通过；exact runtime audit 未运行，certificate 不存在
- 阶段/轴：新动作库 / [`vendor L1 safety audit`](../../DEFINITIONS.md#motion-vendor-l1-safety)
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E1（源码、预注册和合成反例；无真实 B L1 行为结果）

## 问题与冻结输入

只问已通过 L0 的反手拉 B，在 exact vendor MuJoCo collision model 中整轨是否出现机器人自碰或
球拍/拍柄打到机器人。预注册
[`motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json`](../../../configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json)
SHA-256 为 `f8530d834392545105cc4dd89d6a177d4f34ce970cc1ba5d7bb3fdb4d04af699`，绑定：

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
2. `right_racket_collision` 或 `right_racket_handle_collision` 距 head/neck、trunk、对侧臂或下肢
   小于 `5 mm`。

`5–20 mm` 只登记 warning。右腕/手/拍安装链从 `5 mm` proximity pair 排除，避免把机械安装本身判成
自打；但 enabled-robot contact 中实际穿透仍 hard fail。危险插值样本保守标记相邻两个 source frame，
同时直接否决整条动作。

这是 finite dense sampling，不是数学连续时间 swept-volume certificate。桌/网、ground（由 exact L0
certificate 继承）、动力学、平衡、TOPP、击球效果、RL、Gate3 和真机全部不在本门。

## 当前结果与下一步

validator SHA-256 为 `300d581a8df0cb1d8d6bf6b306bd2b28bc511b15c8c2568cb7b66018386488fc`。
source/static gate 与 L0 回归合跑 `17 passed`；反例证明任一 self-collision 或 `<5 mm` clearance 都
fail closed，continuous-time flag/阈值放宽、C 混入和 certificate overwrite 均被拒绝。

本任务没有连接 Pod、没有读私有 certificate/NPZ、没有运行 MuJoCo runtime，也没有创建输出目录或
certificate。因此不能声称 B 已过 vendor L1；G08 继续 Partial。代码合入并由人复核后，下一步按
[操作文档](../../operations/run_motion_backhand_loop_b_vendor_l1_safety.md)在 exact CPU runtime 先做一次
`dry-run`，保全逐帧结果；只有通过且另有显式发布授权，才允许唯一 no-clobber `audit`。通过证书只解锁
独立桌网整轨门，不直接解锁动力学或训练。
