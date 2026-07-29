# EXP-MUJOCO-CHECKPOINT-DIRECTION-SENTINEL-20260728 — 能否在 plant 未对齐时只保留可信方向/安全证据？

- 状态：`blocked`
- 阶段/轴：Isaac→MuJoCo / fresh N5 检查点诊断
- 集成小目标：在任何 MuJoCo 成功率出现前，先内容绑定 actor→q_des→raw qvel→球拍速度/拍面→桌碰/摔倒
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-28 / 2026-07-28

共享术语见[术语与人话对照](../../DEFINITIONS.md)；本实验的
[检查点方向哨兵](../../DEFINITIONS.md#mujoco-checkpoint-direction-sentinel)只产方向与安全 stop
evidence，不产正式上台率。

## 问题与假设

问题：在 MuJoCo 与训练 PhysX 的 plant、qvel 和桌碰传感器语义尚未等价时，能否仍从 fresh N5
检查点得到可审计的击球方向与安全证据，同时机械性阻止任何 pass/return/上台数字被误当成绩？

假设：若每个 milestone 都绑定 exact ONNX/MJCF/动作顺序/题库/ready state/live plant facts，
击球 tick 使用 post-`mj_step`、post-proxy 之前的 raw racket-site velocity，并把 score authority
作为完整 execution contract 的一部分，那么方向异常、q_des clamp 依赖、qvel 爆发、撞桌和摔倒可被
独立定位；任一等价性缺口都会让成功字段保持 `null`。

证伪条件：

- 更换 execution contract 后的 live policy/plant 对象仍可继续 rollout；
- post-step qvel proxy 后的速度可被误当 raw 物理证据；
- 桌碰/摔倒同 tick 仍能留下 numeric pass/return；
- 畸形 JSON/CSV、缩写参数、重复 flag 或旧输出目录能绕过 stop gate；
- 未获 MuJoCo↔PhysX 桌碰传感器证书时仍能发布 success score。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练/eval/main commit | 当前功能分支源码；尚未合入 main，发 fresh N5 前必须换成 exact clean commit |
| 动作/action 集 | fresh N5 ordered manifest，尚未绑定；旧 N4 永久 diagnostic-only |
| 观测/action 合同 | ONNX metadata 自证，actor output 必须为 finite 31-D |
| Reward | 不读取训练 Reward 作晋级尺；所有成功率在 authority false 时屏蔽 |
| Plant/engine | Python MuJoCo evaluator；in-memory Isaac-equivalent table-top slab；post-step qvel proxy 仅续诊断 |
| 训练/考试 bank 或 schedule | 同一 immutable schema-3 BankExam bank/schedule，fresh N5 尚未绑定 |
| Checkpoint/seed | fresh N5 milestone 尚未绑定 |

## 实验差异

- 对照：同一 fresh N5 paper 上的多个 checkpoint milestone。
- 改变的变量：只有 checkpoint ONNX bytes。
- 其余固定项：MJCF、ordered motion、题库、schedule、seed、ready state、控制周期、q_des clamp、
  table binding 和 evaluator source。
- 决策规则：只有方向 receipt、finite、raw qvel、q_des clamp、无桌碰、无物理摔倒和内容合同全部通过，
  才允许把 milestone 交给下一层；success authority 仍需独立 parity certificate。
- 停止/无效规则：NaN/Inf、31-D 不符、identity/hash 漂移、raw qvel 超界、桌碰、physical fall、
  summary/CSV 失配、score 泄漏或 evaluator failure 均 stop。已有 output 不覆盖。

## 已实现的证据链

1. evaluator 在模型 load 时绑定实际编译的 MJCF bytes；table 模式绑定 in-memory canonical XML SHA。
2. execution contract 包含 live policy/plant 的 joint order、default q、action scale、kp/kd、q_des
   limit、damping/friction/armature、ctrl range、integrator、actuator type、qvel limit/proxy、自碰策略、
   robot body/geom 和 ready-state receipt；rollout 前从 live 对象重算并比较。
3. 每个 physics substep 后先保存 raw racket-site world velocity，再执行显式 diagnostic qvel proxy；
   signed direction 使用前者，不能读 proxy 后速度。
4. exact strike receipt 同时绑定目标/实际世界速度、带符号目标/实际拍面、来球速度和
   `dot(v_racket - v_ball, n_target)`。
5. robot/table contact 在每个 physics substep force-qualified 扫描，脚底接触不算撞桌；table hit 与
   physical fall 分账。同 tick unsafe 会清除所有阈值式击球/回球 pass，但保留连续误差与 contact。
6. success-score authority 明确绑定 table sensor、plant exactness、qvel proxy、implicit effort 和
   fail-closed termination。当前 certificate 常量为空，因此 authority 必为 false。
7. CPU-only wrapper 独占自身 flags、拒绝 argparse 缩写/重复/空值、使用 fresh
   [`no-clobber`](../../DEFINITIONS.md#no-clobber) namespace，逐 artifact 重算 SHA，并审计 JSON `null`
   与 CSV 空格。

## 旧 N4 诊断为何无效

旧 upper N4 ONNX：

- path：
  `/workspace/codexschema/mujoco_n4_upper_baseline_model10000_20260728_v1/output/exported_model10000/policy.onnx`
- ONNX SHA-256：`23a97a4e22a5b962e431a6fe4ce5e5e358a02cee3e3f56aeb744370c3966704d`
- parent checkpoint SHA-256：`0ad4cd29ec9929f437d47ed90f8c8af75a4a239d54a92a710a297aefcdebeded`

旧 Python MuJoCo `0/52` **不是 policy 成绩**。诊断显示首帧 `|actor output|max` 约 `87–105`，
31 关节中 `26–27` 个 q_des 被 clamp，且 raw qvel 在首个约 `5 ms` 的 `mj_step` 后超过训练
PhysX bound。继续用 post-step clamp 播放只能显示一种非 exact proxy 轨迹。

以下视频只允许标记 `INVALID diagnostic`：

| 产物 | SHA-256 |
| --- | --- |
| `plant_parity_invalid_backhands_normal.mp4` | `5c3441a238bd58abe6e115defd677814bef9675e28f09769de161409b36ccb09` |
| `plant_parity_invalid_backhands_slow4x.mp4` | `9329776f5d4b1d44d79ef7e03e8ff090fd13ba98e19635939994ca81350f254e` |
| diagnostic ledger | `2f5fa419e07d3a7a19a634c60f23afcb4b46833e4e499634d0f9debbf3c606c0` |

Pod 镜像位于
`/workspace/codexschema/mujoco_n4_upper_baseline_model10000_20260728_v1/output/cpu_plant_parity_diag_v1/`；
本机副本位于 `/Users/Franco/.codex/tmp/mujoco_n4_eval_20260728/cpu_plant_parity_diag_v1/`。
不再生成 N4 正式分数或追加正式视频。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 旧 N4 MuJoCo proxy 诊断 `cpu_plant_parity_diag_v1` | invalidated | model10000 / seed 由旧 ledger 绑定 | E2 diagnostic | 上述 2 视频 + ledger | plant/qvel 合同失败；`0/52` 禁止当成绩 |
| fresh N5 方向哨兵 | blocked | 未绑定 | E1 source | 未创建 | 等 exact N5 ONNX、五动作 ordered manifest、bank/schedule 与 Pod CPU runtime |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh N5 action 0–4 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

旧 N4 `0/52` 不填入本表。当前 sentinel 也不会生成可填入“击球/上台”的成功率。

## 验证

```bash
python3 -m py_compile \
  hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
  hope_training/whole_body_tracking/scripts/mujoco_checkpoint_direction_sentinel.py

python3 -m pytest -q \
  tests/test_mujoco_table_scene.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_direction_sentinel.py
```

2026-07-28 结果：`109 passed, 6 skipped in 1.51s`；py_compile 与 `git diff --check` 通过。6 个 skip
来自当前 host 没有 MuJoCo Python runtime，因此 table/physics 集成仍须在 Pod CPU 环境跑，不能把
dependency-light 单测写成 E2。

## 决定

- 决定：`inconclusive`
- 理由：source-level fail-closed 方向链已闭合，但 fresh N5 exact 输入和 Pod CPU physics run 尚未执行。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：按
  [运行工序](../../operations/run_mujoco_checkpoint_direction_sentinel.md)绑定 fresh N5；先过
  direction/q_des/raw-qvel/table/fall gates，再补 MuJoCo↔PhysX table sensor parity certificate。
  Python BankExam 仍不替代 vendor Gate3/Gate3B，且本工作没有调用真机。
