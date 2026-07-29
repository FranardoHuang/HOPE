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

当前分支结果：`83 passed in 2.19s`。这只证明 pure geometry、32 个刚体逐体 pair-filter
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
subtree、运行时 articulation 的 exact 32-body order、32 个 `[env,1,5,3]` pair-filter force
matrix（含双脚；拍面/拍柄归到 `right_wrist_yaw_Link`）以及 active `robot_hit_table`。任一缺失退出
`1`。

## 3. 真实 actor-contact / 四子步

[`--contact-smoke`](../DEFINITIONS.md#action-ball-table-safety) 只允许在一次性 `num_envs=1` 进程使用：

```bash
python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 1 \
  --motion-file <canonical-motion.npz> \
  --contact-smoke
```

脚本不伪造 sensor/DoneTerm；它对 32 个 articulation body（含双脚）逐个做 top 实碰，再把
`right_elbow_Link` 或 PhysX 中已合并固定拍体的 `right_wrist_yaw_Link` 移入
edge/keep-out/net/posts，并真实推进 PhysX。必须同时满足：

- substep 1/2/3/4 各有一个单帧正 pulse，其他三帧为零；
- 32 个 body 的 top pair 与 keep-out/net/左右 post 对应 exact filter 列都产生非零接触力；runtime
  判定阈值只保留 `1e-6 N` 数值零容差，不允许把 `<1 N` 轻蹭当合法动作；
- `robot_hit_table` raw reason 与 generic terminal ledger 各增加且只增加一次；
- automatic reset 后的零 pulse step 不再报告 `robot_hit_table`。

输出最后一行 `HOPE_TABLE_OBSTACLE_CHECK_JSON=...` 原样保存。失败日志也保存，不得删失败尝试后重报
“全过”。

## 4. 4096-env 性能对照

同一 Pod/GPU、同一 commit、同一动作文件分别跑 on/off；两臂分进程串行，不能在一个 Kit 里建第二个
environment：

```bash
python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 4096 \
  --motion-file <canonical-motion.npz> \
  --table-obstacle on \
  --bench 200

python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 4096 \
  --motion-file <canonical-motion.npz> \
  --table-obstacle off \
  --bench 200
```

记录两份 JSON、GPU 型号、commit、Torch/Isaac 版本和 peak memory。32 个 exact sensors 的 4096-env
成本尚无实测；在负责人审阅实测差值前只能保持 `Partial`，不能据 host pass 启动 canary 或长训。

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
