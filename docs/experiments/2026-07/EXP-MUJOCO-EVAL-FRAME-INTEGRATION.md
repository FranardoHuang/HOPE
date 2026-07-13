# EXP-MUJOCO-EVAL-FRAME-INTEGRATION — 三路 evaluator 合同能否无损集成？

- 状态：`source_corrected_pending_main_merge`
- 阶段/轴：Isaac→MuJoCo / evaluator、导出与部署装载合同
- 人类负责人：yikang
- 执行者：Codex
- 工作分支：`Franco_codex/mujoco-frame-parity-integration-20260713`
- 基线：已 merge 对齐 `origin/main@c48fdc2`
- 最高证据等级：E1（源码/合同/单元回归；没有新的 simulator 行为结果）
- 决定：**adopt** 源码集成与 fail-closed 供证；**inconclusive** Isaac–MuJoCo 行为 gap 是否已缩小

## 问题与结论

当前 main 已采用 raw-A、逐 clip physical-B 的 signed-face scorer。现场训练 source `50c49e5` 还包含
三组未进 main 的能力：implicit total-effort/self-contact parity guard、`actor_leg_ref_mask` 供证，以及
pelvis COM→link-origin reset 与 link/IMU gyro frame 修复。本票只选择性移植这些源码提交到最新 main，
不吞入 50c 分支的旧 `NOW`、旧实验状态或其他并行功能；合并时保留 main 的 signed-face、exam rebind、
动作链和 B/C 选择器。

源码纠正了真实 frame/供证错误，但**没有**证明机器人在 MuJoCo 不再倒、回球成绩提升，或跨引擎 gap
已经关闭。formal `stand-keyframe` 的 reset qvel 为零，所以 COM reset 修复不改变该格；XBODY gyro
修复会改变 actor observation，但 vendor A3 上 link 与 inertia-principal axes 只差约 `0.3315 deg`。
必须用新的同一题表行为卷量化，不能用单元回归替代。

## 已采用的集成边界

- 保留 main 的 signed-face scorer API、逐 clip sign、raw-A achieved/target 与 physical-B facing；不恢复
  旧 unsigned normal 路径。
- bound implicit joint 不再用 `clip(P)-D` 或“P 电机 + D 被动阻尼”近似：每个 substep 先算
  `P-D`，再按 Isaac 的对称 effort limit 一次裁剪并把结果送进 motor。饱和本身是已复刻的正常行为，
  不再降级 exact；保留任何 MuJoCo passive damping 或缺 effort limit 的 implicit 路径只能显式
  diagnostic，formal 构造直接 fail closed。
- self-contact 只统计 `pelvis_link` articulation 子树内的两机器人 geom；球、桌网、mocap helper 和其他
  free body 不算机器人自碰。分类在每个 MuJoCo physics substep 的 `mj_step` 后执行；formal BankExam
  一旦出现机器人自碰立即失败，即使它在同一 control step 末已消失；diagnostic 累计全部 substep 且不改
  plant 物理。
- diagnostic `teacher-reference` reset 只对显式 `center_of_mass` 速度做
  `v_origin = v_com - omega_world x (R_world_body * body_ipos)`；显式 `link_origin` 直写但不能因此获得
  formal exact。actor gyro 改用 `mjOBJ_XBODY/local` 的 pelvis link/IMU axes。evaluator 要求
  `pelvis_link` 自身恰好一个、qpos/dof 地址均为零的 freejoint，同时允许球等其他 free body。
- native/standalone schema-3 export 为每个 clip 绑定 `motion_body_lin_vel_points`；旧 exact schema-2 只有
  窄的 all-COM 兼容规则，含糊或 inexact 的旧包在 teacher reset 前 fail loud。

## `actor_leg_ref_mask` 供证 epoch

命令观测为 62 维的 actor 必须携带 `actor_leg_ref_mask_provenance_epoch=1`。epoch 绑定实例化环境里实际
command-observation callable 的规范身份，并进入 canonical training contract、checkpoint contract digest
和 ONNX metadata：

- epoch 1 且没有 `actor_leg_ref_mask` 表示经过供证的 unmasked callable；
- epoch 1 且 `actor_leg_ref_mask=1` 表示 masked callable。当前 MuJoCo/C++ 179 builder 不能等价构造该
  语义，因此正式 evaluator/loader 拒绝，不会把“非击球臂不模仿”的候选误判为普通 policy；
- canonical callable 的**严格空且 exact built-in type** `functools.partial` 才会 unwrap 后按身份供证；
  exact-type 规则逐层执行。任何 subclass（可覆写 `__call__`）、bound args/kwargs、unknown wrapper、
  伪装 copy、noncanonical partial、缺 epoch 或 checkpoint↔contract digest 不一致都不能成为 exact；
- standalone export 必须从 checkpoint 合同覆盖或清除 donor 的 mask/epoch，不能相信旧 donor metadata。

禁止给旧 ONNX 后补 epoch/mask 来恢复 exactness：旧 checkpoint 没把该事实纳入 digest，事后 metadata
不能证明训练时用的 callable。冻结的旧 model-2000 2×2 paper 因此前缺 epoch，另用
`configs/phase1_cross_engine_instrument_parity_2x2_revocation_20260713.json` 撤销其 current-exact 身份；
历史诊断保留，但不能填进新正式纸。

## 独立红队后关闭的假绿

首次集成审计给出 `NO-MERGE`，随后在本分支逐项关闭：

1. 旧 effort guard 只在 `|clip(P)-D|>L` 时报警，漏掉 `P=8,D=2,L=6` 等抵消情形，也会把
   `P=5,D=-4,L=6` 执行为 9。现用纯函数与运行路径共同锁定 `clip(P-D,-L,L)`，覆盖抵消、同向、纯 D
   和正负边界；被动阻尼近似不得获得 formal exact。
2. 旧自碰分类把任意两个非 worldbody geom 都算机器人，未来合法球拍—动态球接触会误报。现由
   pelvis 子树显式构造 robot body/geom 集合，动态球负控不计入；formal 策略为 fail closed。
3. mask provenance 旧实现丢弃 partial args/kwargs 后只看底层函数身份；现只允许严格空 partial。
4. 旧 Phase-B rider 的撤销原先只由 2×2 validator 执行。direct adapter loader 现有内容 SHA denylist，
   并同时验证不可变 revocation receipt；receipt 缺失/篡改也只能拒绝。旧操作命令已改为禁止运行的
   历史说明。
5. scoreboard 遇旧 header 原先会把更宽新行直接追加，导致 CSV 错列。现于运行前和写入前双重校验，
   mismatch 不改一字节并要求新 output root/显式迁移。

第二轮独立红队仍给出 `NO-MERGE`，并在同一候选分支继续关闭两项：

6. `isinstance(x, functools.partial)` 会接受覆写 `__call__` 的 partial subclass；实测该对象实际执行
   `different-command-semantics`，却曾能 mint epoch 1。现只允许每层 `type(x) is functools.partial`，并以
   dependency-free subclass 负测锁死。
7. 第一版机器人自碰只在一个 control step 的全部 `mj_step` 后扫描最终 contact，可能漏掉前几个
   physics substep 中已经改变轨迹的短碰撞。现每个 `mj_step` 后立即扫描：formal 首碰即拒绝，diagnostic
   保留完整 substep 汇总；dependency-free fake-step 测试证明第一 substep 碰撞、末步干净也不会消失。

## 验证

本次选择性移植依次整合上游提交 `bebee04`、`3788fe7`、`50dabbd`、`57719ad`、`3f04890`，随后 merge
对齐 `origin/main@c48fdc2` 并保留 main 的动作/GMR/D-retry 文档与代码，未改训练配方或 checkpoint。

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_implicit_effort_guard_and_selfcontact.py \
  tests/test_gate3_first_tick_state_bridge.py \
  tests/test_phase1_cross_engine_instrument_parity_2x2.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_reference_reset_com_frame.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  hope_training/whole_body_tracking/tests/test_export_obs_norm_contract.py \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_adapter.py \
  hope_training/whole_body_tracking/tests/test_scoreboard_eval_contract.py

python3 -m pytest -q tests
```

当前集成树结果为 focused `147 passed, 2 skipped`、root suite `696 passed, 9 skipped`，`git diff
--check` 与四个修改源码的 `py_compile` 均通过。两个 focused skip 都是当前 host 缺 `mujoco`：tiny
effort/self-contact physics 模块和真实 A3 frame reset 各一项，因此这些 physics cases 不是本机行为通过。
本机也缺 `torch`，`test_isaac_bank_exam_phase_b.py` 无法收集；direct rider 撤销的 dependency-light adapter
负测已运行通过，但不能替代 Torch/Isaac 行为套件。上游 pelvis 修复另有真实 A3 MJCF CPU smoke 与
10 秒 plain-MuJoCo PD stand E2 证据；本次没有重跑 Pod、K100、PPO、vendor backend、Gate3 或真机。

## 后续门

1. 用 post-epoch、checkpoint/contract digest 完整的 fresh unmasked 179 actor 跑 current C++ real-model
   preflight；旧 ONNX 不回填供证。
2. 在同一 immutable paper 上重跑 kinematic replay → open-loop action → external-observation closed-loop →
   native closed-loop，记录 effort guard、self-contact、signed-face 与 ready-state 分层读数。
3. 源码进入 main 后再更新 [`TIMELINE`](../../TIMELINE.md)；行为卷未过前 G06 保持 `Partial`，也不把
   source fix 写成 Gate3/Gate3B 通过。
