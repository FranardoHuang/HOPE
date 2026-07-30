# ActionBall N5 学习策略 MuJoCo 物理回球门

这一步回答一个单独且不可由训练 reward 代替的问题：冻结策略在动作自己的
center/support 来球上，是否用对应击球帧真实碰球、过网并首落对面球台，同时不撞桌、
不自碰、不越硬限位、不超速、不倒下。

详细术语见 [`docs/DEFINITIONS.md`](../DEFINITIONS.md)。这个门不运行 selector，不重新
求解 task，也不调用 `mujoco_eval_onnx.py` 的 virtual landing scorer。

**当前状态：BLOCKED。** fresh N1 actor 是固定 194-D
`action_ball_table_pose_twist_heading_task_teacher_start_v2`，不含 action one-hot，且现有
MuJoCo/C++ producer 还不能逐元素构造 v2。v2 只允许 N1；formal N5/N73 必须先冻结固定宽、
由动作内容导出的 continuous future-motion intent/preview。因此当前没有可接受的 N5 actor
contract 名称或宽度，历史 199-D one-hot ONNX 不得冒充新合同，也不得部署或接真机。

## 必要输入

- exact clean commit；
- N=5 physical-contact-v2 manifest、profile pins 和 launch trust root；
- 已经正式通过的 `mujoco_teacher_motion_fitted_ball_gate` receipt；
- 同一个 checkpoint 的 checkpoint bytes、未来冻结的 fixed-width multi-action actor contract
  ONNX 和 `obs_norm.npz`；
- 全部输入的 SHA256。

ONNX 必须声明 schema-3 exact training contract、冻结后的 fixed-width continuous
future-motion intent contract、五个 ordered motion SHA/segment length，并将 source checkpoint
和 normalizer sidecar 绑定到输入字节。非零 PhysX load-dependent joint-friction coefficient 当前没有
exact MuJoCo 等价实现，会 fail-closed，而不是静默换成 MuJoCo `frictionloss`。

当前 MuJoCo/C++ producer 连 N1 fixed-194 v2 尚不能构造，因此本 Gate 在 formal N5 前必须先补
table-relative pose、heading twist、三条 frame-consistent task 向量、
`time_to_teacher_start_s` 和未来 continuous future-motion intent 的逐元素 parity；不得把旧
186-D/199-D ONNX 当作新合同输入。

## 运行

```bash
python -I hope_training/whole_body_tracking/scripts/mujoco_action_ball_policy_fitted_gate.py \
  --code-commit <clean_commit_sha> \
  --manifest <physical_manifest.json> \
  --manifest-sha256 <sha256> \
  --profile-pins <profile_pins.json> \
  --profile-pins-sha256 <sha256> \
  --launch-trust-root <launch_trust_root.json> \
  --launch-trust-root-sha256 <sha256> \
  --teacher-gate-receipt <teacher_fitted_gate.json> \
  --teacher-gate-receipt-sha256 <sha256> \
  --checkpoint <model_N.pt> \
  --checkpoint-sha256 <sha256> \
  --onnx <policy.onnx> \
  --onnx-sha256 <sha256> \
  --obs-normalizer <obs_norm.npz> \
  --obs-normalizer-sha256 <sha256> \
  --render-dir <new_empty_parent>/videos \
  --out <new_empty_parent>/policy_fitted_gate.json
```

先加 `--preflight-only` 可只检查输入合同。正式运行的 `render-dir` 和 `out` 都必须是
不存在的新路径；工具拒绝覆盖。

## 正式题目与通过条件

每个 action 由 Gate 控制面固定 stable UID/slot，actor 不接收 categorical identity；执行
teacher capsule 中三个正例：

1. `center_positive_seed_0`；
2. `center_positive_seed_1`；
3. `support_positive`。

每题都在 1.0 ms 和 0.5 ms 两个 MuJoCo physics step 下执行；policy step 固定 20 ms。
两个步长都必须满足：

- selected face 恰好一次 fitted physical contact，且在冻结 strike window 内；
- contact 后真实过网，首落点离 solver-bound aim 不超过 0.10 m；
- return table bounce 恰好一次且不撞网；
- table/self/hard joint limit/raw velocity/fall counter 全为零；
- imitation metrics 有限；
- receipt 明确记录 `selector_executed=false`、
  `solver_executed_by_gate=false`、`virtual_scorer_executed=false`。

每个 action 生成一段 center-case MP4；视频只供人工检查，不能替代上述解析物理判据。

## 返回码

- `0`：preflight 通过，或正式 gate 全部通过；
- `3`：输入完整但物理题有效失败，或 fail-closed blocker；
- `2`：基础设施/运行异常，不能形成正式物理结论。

receipt 会在运行结束再次 hash 所有外部输入并复查 clean checkout。任何输入漂移都使
本轮证据失效。
