# 接口合同：球拍目标的物理有效性

**桌面以下没有球。** 现有的目标框检查全是"配置跟自己对账"——只有这一条拿物理常量来查。

## 常量真源（只有一份，别抄）

| 量 | 值 | 定义处 |
| --- | --- | --- |
| 桌面高度（离地） | `0.76 m` | `tasks/table_tennis/geometry.py:39` `TABLE_HEIGHT` |
| 球半径 | `0.020 m` | `tasks/table_tennis/physics/params.py:50` `ball_radius` |
| 近台边 x（env 系） | `0.5 m` | `racket.vb_table_near_x` |
| 半台宽 | `0.7625 m` | `geometry.py` `TABLE_WIDTH / 2` |

env 系 z=0 是地面，桌面在 `+0.76`；HOPE planner 系 z=0 是**桌面**。换算是纯平移。

## 规则

任何 x 伸到近台边之外（`x_hi > vb_table_near_x`）的球拍目标框，必须满足

```
z_lo  >=  vb_table_surface_z + ball_radius  =  0.76 + 0.02  =  0.78 m
```

更低的框在命令机器人往台面里/台面下打球，那里永远不会有球。
**这是构造期错误不是警告**——它的签名是"某一侧回球率恒等于 0.0000，其他什么都不报"。

选**触球帧**时门槛更高：拍高 ≥ `0.76 + 0.02 + 0.10 = 0.88 m`，
让整个 ±0.10 的目标高度框都离台。

另两条同族的物理判据：

- 目标速度前向下界必须 `> 0`（+x 朝对手）。`x_lo ≤ 0` 会命令一个永远落不到对面半台的球。
- 触球帧世界系拍面法向 `|n_x| ≥ 0.8`（参考 v4rg 实测 0.86–0.94）。
  拍面朝侧面 ⇒ 落点 `|y|` 冲到 0.85–1.86 m，而半台宽只有 0.7625 m ⇒ 出界。

## 谁在执行

| 检查 | 位置 | 形式 |
| --- | --- | --- |
| 击球点离台 | `tasks/tracking/mdp/hope_commands.py:1389-1415` `_assert_contact_clears_table` | 构造期 `ValueError` ✅ |
| 目标速度朝前 | 同文件 `_assert_target_velocity_points_forward` | 构造期 `ValueError` ✅ |
| 逐侧零回球报警/中止 | `utils/my_on_policy_runner.py:41-46` | 500 次机会报警 / 5000 次中止 ✅ |
| 逐片框静态检查 | `scripts/check_perclip_{pos,vel}_sampling.py` | ⚠ **只打印，没有退出码门** |
| 拍面法向朝向 | — | ⚠ **尚未实现** |

后两条的确切检查与落点见
[应当变成闸门的规则](../operations/rules_that_should_be_gates.md) P0 #1/#2/#4。

## Task-first 站位与整轨撞桌

目标点高于桌面只证明“要求的击球点存在”，不证明球拍从 ready 到击球再恢复的整条轨迹不撞桌。
[`station_center_shift_xy_m`](../DEFINITIONS.md#station-center-shift)必须把动作、球拍 task center 与
base center 一起平移；在本坐标约定里负 X 远离球台。不能只把 base Reward 往后移、却让 reference
球拍轨迹留在原位。

新正手只比较 `[0,0]`、`[-0.05,0]`、`[-0.10,0] m`，并取 upper/full 都通过时离原位最近的一档。
每档必须分别检查：

- exact motion 的连续/密集桌网扫掠；
- 球拍和手柄，而不只是 wrist link origin；
- 从 ready 到 post-strike recovery 的整个 `t_cycle`；
- Isaac 的 `robot_hit_table` 终止是否由 broad robot geometry 或专用
  `racket_table_contact` filtered wrist-vs-table sensor 触发；
- missing/malformed/non-finite contact evidence 是否失败封闭。

当前 source candidate 的 reference certifier 只产未授权检查，`diagnostic_smoke_authorized=false`；
Isaac filtered contact sensor 尚未在 Pod 做 negative/positive runtime smoke。因此新正手当前仍是
`training_authorized=false`，站远一点只是待证候选，不是已采用修复。完整任务合同见
[task-first](task_first_n_action_contract.md#6-新正手与-station-选择)。

## 代价（2026-07-26）

绑定的正手击球帧把拍子放在 `z = 0.694 m`，桌面在 `0.76 m`。
`z >= 0.60` 那条旧线放行了它。四条臂 `virtual_return_rate_forehand` 恒等于 `0.0000`
跑了几千迭代，藏在一个健康的聚合数字后面。
代码里的人话原文：**"桌面以下没有球。目标框/参考击球点掉到桌面以下就当场报错，
别再让某一侧的回球率无声地钉在 0.0000。"**

全链工序见[动作片→训练绑定第 14 步](../operations/run_motion_clip_to_training_binding.md)。
