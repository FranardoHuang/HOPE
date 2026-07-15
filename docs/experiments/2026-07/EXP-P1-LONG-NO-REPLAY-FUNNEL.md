# EXP-P1-LONG-NO-REPLAY-FUNNEL — 先把早筛候选跑成可解释的 10000-update 长曲线

- 状态：`running`
- 阶段/轴：阶段 1，平衡约束与动作模仿强度
- 集成小目标：在不引入随挥状态回放的条件下，比较普通配方、关节速度边界惩罚与击球窗模仿放松
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-15

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本卷中的
[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 是“关节接近真实速度上限后才开始收费”的惩罚；
[`post-swing replay start`](../../DEFINITIONS.md#post-swing-replay-start) 是从上一拍随挥结束状态开始新 episode。

## 问题与假设

过去的 500/1000-update 曲线只能早筛，且关节速度惩罚在 500 到 1000 之间发生方向翻转。本实验问：
在相同 source、动作、题库、seed 与 10000-update 上限下，以下两种改动是否能比普通对照更早、更稳定地同时改善
击球精度与身体稳定性：

1. 只增加关节速度边界惩罚；
2. 放开手腕线速度模仿，并把击球窗内的动作模仿降为四分之一。

任一改动若只短暂领先、长曲线中回落，或以更高摔倒/关节饱和换取击球分，则不采用。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练 source | `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e`，Pod2 clean checkout `/workspace/codexschema/nohope_p1_activation_successor_2c2d70d`；该 source 同时含 qdot、V1/V2 激活计数与 immutable run binding |
| 动作 | 现役正反手 `v4rg` 两条动作，raw-A/physical-B 符号按 clip 固定为 `[+1,-1]` |
| 观测/action 合同 | `deploy_parity_face179` / 31 维关节目标 |
| Plant | Isaac，31 关节零摩擦；只作训练和方向诊断，最终行为仍由 vendor MuJoCo 判定 |
| 训练题库 | schema-3 rebound bank：`s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz` |
| Seed/预算 | 三格共同 seed `3`，4096 environments，`max_iterations=10001`，每 100 保存 |
| 检查点 | 200/500/1000/2000/3000/6000/10000 |
| 资源 | 只允许 Pod2 GPU2，最多同时三条；Pod1 与 Pod2 GPU0/GPU1 不在本实验权限内 |

该 exact source 已在 Pod2 通过 4096-environment full-scene terminal probe：result content
`4cbc9fc0...55fb083`、finite checkpoint `68d9809b...598713`、hard contract
`451cda47...c12291`。本实验只复用 source/scene 启动与 runtime-binding 证据，不复用旧行为结论。

## 实验差异

| 人话名称 | 唯一变化 |
| --- | --- |
| 普通对照 | 关节速度惩罚关闭；动作模仿不放松 |
| 关节速度边界惩罚 | `joint_velocity_limit_hinge_weight=-5.0`，其余与普通对照相同 |
| 击球窗模仿放松 | 放开手腕线速度模仿，击球窗内动作模仿缩到 `0.25`，关节速度惩罚关闭 |

三格共同把 `post_swing_start_prob` 固定为 `0`。这是为了去掉已经证实会被 policy 自身存活时间影响的
内生 curriculum，不是声称随挥恢复不重要。

## 决策规则

- 200：只检查 boot、finite、合同和机制是否非零；结构错误可停，行为差不能定输赢。
- 500/1000：只看方向与是否值得继续；不得采用、不得买第二 seed。
- 2000/3000：比较同窗击球位置/法向/虚拟回台、物理 fall、关节速度/力矩饱和和完成率的中段趋势。
- 6000/10000：形成完整长曲线；若某 treatment 已按预注册安全/机制门明确失败，可保全证据后单独停，
  共享对照为仍存活的 treatment 继续。
- 候选在长曲线中持续领先且没有安全债后，才进入 immutable MuJoCo/vendor 同卷；通过后才考虑第二 seed。
- 任一 checkpoint non-finite、source/recipe/plant/claim 漂移或机制未激活，立即判该格无效并保全证据。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 普通对照 attempt-1 `phase1_long_no_replay_control_seed3_20260715` | invalidated | 未到首迭代，seed3 | dynamic URDF import 后日志 180 秒无进展；exact PGID `410589` 已由 watchdog 收口 | claim/binding/log/TERM/KILL identity evidence | 纯基础设施失败，无 checkpoint，不是 Reward 结果；namespace 永不复用 |
| 关节速度边界惩罚 `phase1_long_no_replay_qdot_w5_seed3_20260715` | running | 同上 | PGID `411519`，04:15 UTC 到 iter24，fatal0，claim/binding present | Pod2 run directory | 只改变 qdot 惩罚权重 |
| 击球窗模仿放松 `phase1_long_no_replay_v1v2_seed3_20260715` | running | 同上 | PGID `412204`，04:15 UTC 到 iter9，fatal0，claim/binding present | Pod2 run directory | 只改变两项动作模仿开关 |
| 普通对照唯一重试 `phase1_long_no_replay_control_seed3_retry_v2_20260715` | running | 同上 | PGID `412899`，04:15 UTC 到 iter2，fatal0，claim/binding present | 全新 no-clobber namespace | 配方逐字不变；共享匹配对照 |

2026-07-15 04:06–04:09 UTC，attempt-1 在 source/assets/Hydra compose 通过后启动，停在动态 URDF import；
进程曾使用约 13 个 CPU core，但 `run.log` 固定在 32075 bytes，GPU2 仅约 2.3 GiB/1%。180 秒 stale-log
watchdog 先写同一 PGID 的 pre-TERM/pre-KILL identity evidence，再只收口 PGID `410589`，终态
`terminal_kind=stale_timeout`、`rc=125`、GPU2 回到 2 MiB。后两格当时尚未 claim；本卷只授权普通对照一次
逐字相同配方 retry，且先发两个仍未消费的 treatment，避免启动故障继续让 GPU 空等。

第二次 `fill --count 3` 严格按 qdot → V1+V2 → control-retry 顺序逐条等待真实
`Learning iteration`，三条都返回 `KIT_BOOT_READY`。04:15 UTC，GPU2 上恰有这三个 Codex PID，利用率
`97%`、显存 `17154/32607 MiB`；GPU0/1 的 Yikang PID 保持不变。当前只证明生产训练已启动，尚无
model-200、激活计数或行为结论。

## 复现与证据

机器队列：[`configs/phase1_long_no_replay_funnel_20260715.yaml`](../../../configs/phase1_long_no_replay_funnel_20260715.yaml)。
发射前先运行 `plan` 和 `doctor --live`；唯一写操作为：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_no_replay_funnel_20260715.yaml \
  fill --count 3 --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

该命令只能命中三条 `required_slot: pod2/gpu2` 行；不得改成未绑定 GPU 的临时命令。
