# ActionBall 桌体安全 smoke

这页验收
[`ActionBall table safety assembly`](../DEFINITIONS.md#action-ball-table-safety)：解析球任务中的真实
桌板、floor→slab-underside 保守 robot keep-out、球网和两根网柱，以及覆盖四个 5 ms physics
substep 的桌碰 latch。它只做 source/runtime 验证，不训练、不创建 checkpoint、不授权真机。

## 1. Host E1

在仓库根目录运行：

```bash
cd hope_training/whole_body_tracking
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest \
  tests/test_table_obstacle_geometry.py \
  tests/test_table_obstacle_termination.py \
  tests/test_table_contact_substep_latch.py \
  tests/test_action_ball_task_config.py \
  tests/test_soft_joint_limit_barrier_v2.py -q
```

当前 source `2d9997a6` 在 Pod 的 table/producer/stage/canonical focused suite 为
`214 passed in 2.86s`；`f068555b` 的 table runtime focused suite为
`110 passed in 2.65s`。这只证明 pure geometry、32 个刚体逐体 pair-filter
termination kernel、四子步 sticky latch、partial reset、ActionBall source binding 与 joint
safety host 合同，不证明 Isaac collider。

## 2. Pod cfg 与 spawned scene

以下 `<canonical-motion.npz>` 是已恢复到 Pod 的 canonical 动作文件；脚本只用它构造 scene，不读取
policy 或训练 Reward。先做 cheap cfg：

```bash
python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 1 \
  --cfg-only
```

再构造真实 Isaac scene：

```bash
python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 1 \
  --motion-file <canonical-motion.npz>
```

必须看到 exact 5 prim、每件至少一个 enabled `UsdPhysics.CollisionAPI`、唯一且无 collider 的 visual
subtree、运行时 articulation 的 exact 32-body order，以及 **5 个**
`[env,1,32,3]` pair-filter force matrix：每个 table source（top/keep-out/net/左右 post）
按同一顺序显式过滤 32 个 A3 body（含双脚；拍面/拍柄归到
`right_wrist_yaw_Link`）。一个 Robot wildcard 不能代替这 32 个 expression；旧的
32-source × 5-filter 布局与 wildcard aggregate 布局均已拒绝。还必须看到 active
`robot_hit_table`；任一缺失退出 `1`。

## 3. 真实 actor-contact / 四子步

[`--contact-smoke`](../DEFINITIONS.md#action-ball-table-safety) 只允许在一次性 `num_envs=1` 进程使用：

```bash
python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 1 \
  --motion-file <canonical-motion.npz> \
  --contact-smoke
```

脚本不伪造 sensor/DoneTerm；5 个 table-source sensor 的 32-column exact matrix
shape/order 逐个 materialize，再用确有 collision geometry 的代表 body 对
top/keep-out/net/posts 做真实正控并推进 PhysX。必须同时满足：

- substep 1/2/3/4 各有一个单帧正 pulse，其他三帧为零；
- 五个 table source 的 32-body exact filter matrix 全部可构造；top/keep-out/net/左右 post 五个角色各有
  至少一个真实 representative-body 正控产生非零接触力；runtime 判定阈值只保留 `1e-6 N`
  数值零容差，不允许把 `<1 N` 轻蹭当合法动作；
- `robot_hit_table` raw reason 与 generic terminal ledger 各增加且只增加一次；
- automatic reset 后若 PhysX 暂留终止姿态的 final-substep contact report，只对原 table
  terminal row 的第一份 report 做一次 quarantine；同一步后续 substep 与其他 env 仍正常检测，
  零 pulse step 不再重复报告 `robot_hit_table`。
- cfg 固定 `seed=0`、官方 A3 stand 出生与 frame-0 hold，避免 pair-filter 正控被随机 RSI
  reset 姿态污染；场景构造完成后只做一次显式初始化 `env.reset()`。五件 collider 的 destructive positive
  controls 之间沿用上一脉冲 automatic reset 后已经通过的 clean state，不再调用会重采 generic
  motion command 的 `env.reset()`。否则测试会把无关随机 reset 姿态或不推进 PhysX 的旧 force
  report 误写成 table-sensor 失败。
- 初始 settle 只要求 `robot_hit_table=false`；reference/episode 等其他 generic termination
  继续进入各自账本，但不冒充桌碰传感器的负例或阻断本工具的五角色物理正控。
- 每个正 pulse 的 automatic reset 后，诊断器在 clean step 前原子恢复该 probe 保存的
  table-clear root/joint snapshot；这是对训练 dynamic-ready reset 的最小等价替身，避免 generic
  motion resample 自己生成桌碰姿态后被误判成跨 episode stale 泄漏。

输出最后一行 `HOPE_TABLE_OBSTACLE_CHECK_JSON=...` 原样保存。失败日志也保存，不得删失败尝试后重报
“全过”。

当前 explicit-32 首份通过证据是 Pod1 source `f068555b`、namespace
`/workspace/franco/n1speed_f068555b_table_contact_r1`：五 probe 全部出现
`contact_probe_done`，随后 `main_completed`、shell rc=`0`，且日志没有 unsupported、
did-not-match、FAIL 或 Traceback。source `2d9997a6` 的故意失败进程返回 shell rc=`1`，
证明 Kit teardown 不会把 Python 异常伪装为成功。旧
[`table_smoke_eb2799b1_gpu1_r26.receipt.json`](../../configs/n1_contact_dynamic_ready_20260730/table_smoke_eb2799b1_gpu1_r26.receipt.json)
只保留历史字节身份，不得冒充 explicit-32 v3 证据。

## 4. 4096-env 性能对照

同一 Pod/GPU、同一 commit、同一动作文件分别跑 on/off；两臂分进程串行，不能在一个 Kit 里建第二个
environment：

```bash
python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 4096 \
  --motion-file <canonical-motion.npz> \
  --table-obstacle on \
  --bench 10

python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 4096 \
  --motion-file <canonical-motion.npz> \
  --table-obstacle off \
  --bench 10
```

短探针固定 seed/官方 stand 并关闭 reference-envelope reset，只保留 physical hard/fall/table
安全项；它在已知早期 raw-hard onset 前结束，量的是 backend 固定税，不是完整 trainer wall。
source `2bddb440` 的 Pod1 两次结果为：

| replicate | table on | table off | 差值 |
| --- | ---: | ---: | ---: |
| r1 | `67.577 ms/step` | `60.233 ms/step` | `7.344 ms/step` |
| r2 | `72.000 ms/step` | `61.400 ms/step` | `10.600 ms/step` |

平均差约 `8.97 ms/policy-step`，折算 24-step rollout 为 `0.22 s/update`，不能解释现役
`17–25 s/update`。所以当前保留 exact filter 后端，不切换 box。只有后端再次失效或真实
trainer profile 显示非线性放大时，才启用已设计的“桌坐标系保守几何棱柱”后端；该降级必须使用 collision-geom center+bound/swept
包络并覆盖球拍偏移，禁止只检查 body origin。

## 已知限制

- tracked visual USD 含桌腿和网，但其 physics layer 是不准确的 whole-mesh convex hull；本合同只把
  USD 当显示层。floor→slab-underside keep-out 是保守机器人安全代理，不是腿几何证书。
- keep-out 会改变动力学球的桌下运动，所以 ActionBall 对 physical/shadow 球 fail closed。球物理实验
  继续使用自己的真实 top/net/post 路径，不得绕过该拒绝。
- runtime 的非零 pair-force 终止只负责抓 policy 执行期碰撞；teacher admission 另要求整个
  prep→hit→recovery 连续 swept-volume 对 table top/edge/underside/keep-out/net/posts 保持至少
  `5 mm`，二者不可相互替代。
- 2026-07-15 的旧 table/net clearance prereg 固定修改前 source bytes，只能作历史证据；当前实现需要
  新的内容寻址 prereg/receipt。
