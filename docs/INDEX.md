# 路由（找到你那一行就走，不要遍历 docs/）

先读 [`START_HERE.md`](START_HERE.md)。缩写看 [`DEFINITIONS.md`](DEFINITIONS.md)。
**动态状态不在本页**：当前阶段、队列、采用 setting 只看 [`origin/main` 的 NOW](NOW.md)；
实验状态只看 [`experiments/README.md`](experiments/README.md) 的索引表。

## 我要做的事

| 任务 | 只读这些 |
| --- | --- |
| **找一个工具 / 想写新脚本** | [工具目录](operations/tool_catalogue.md) |
| **把编译好的动作片变成能训的配方** | [动作片→训练绑定 16 步](operations/run_motion_clip_to_training_binding.md) |
| **发一波消融** | [消融波发射工序](operations/run_ablation_wave_launch.md) → [runbook 队列与算力](runbook.md#统一队列排序与算力纪律) |
| **判卷 / 报数** | [结果判读与报数](operations/read_and_report_results.md) → [runbook 判卷链](runbook.md#判卷链北极星数字怎么产2026-07-06-全链踩通) |
| 理解或修改训练 setting | [NOW 完整流程](NOW.md#1-当前一套训练是怎样完整跑起来的) → [G05](gates/G05_isaac_training_first_loop.md) → [`run_training.md`](operations/run_training.md) |
| 对照外部/智元 setting 并处理训练吞吐 | [DR、Reward 与智元外部尽调](research/dr_reward_external_diligence_20260731.md) → [N1 设计/加速审计](research/design_audit_and_speedup_20260729.md) → [MuJoCo 原生下一版准备账](experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) → [旧分阶段运行账](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md) |
| 当前 adopted ActionBall 发射 | [`origin/main` NOW](NOW.md) → [按动作条件化 Ball-first 合同](interfaces/action_conditioned_ball_first_contract.md) → [G05](gates/G05_isaac_training_first_loop.md)；本分支 successor 未合入前不改变 formal N5/N8/N12 当前路线 |
| 单动作 FullMDP → Isaac A/C 长跑 → MuJoCo GPU A/C | [当前简洁 TODO](operations/action_ball_dual_backend_longrun_todo_20260819.md) → [历史详细账](operations/action_ball_single_action_dual_backend_todo_20260817.md) → [Isaac 5.1环境身份](operations/action_ball_isaac51_environment_identity_20260818.md) → [G05](gates/G05_isaac_training_first_loop.md) → [MuJoCo 准备账](experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)；目标是同进程`4096×25000`，1000只是只读早期趋势节点 |
| 下一版 N1 可学门→MuJoCo→N73 候选 | [MuJoCo 原生下一版准备账](experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) → [按动作条件化 Ball-first 合同](interfaces/action_conditioned_ball_first_contract.md) → [旧分阶段运行账](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)；这是 branch-candidate 设计/迁移账，不是当前唯一 TODO |
| 训练任意 N 动作的来球/落点泛化 | [按动作条件化 Ball-first 合同](interfaces/action_conditioned_ball_first_contract.md) → [Ball-first 实验](experiments/2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md) → [MuJoCo 原生下一版候选](experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) → [桌体安全 smoke](operations/run_action_ball_table_safety_smoke.md) |
| 用成功率与优先级选择动作 | [capability selector 合同](interfaces/action_capability_selector_contract.md) → [selector 实验](experiments/2026-07/EXP-ACTION-CAPABILITY-SELECTOR-20260727.md) → [planner 边界](operations/run_planner.md#n-action-selector-boundary) |
| 核对 trainer 真正使用的 Reward | [effective Reward 因果审计](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md) → [发射前真值检查](operations/run_training.md#effective-reward-truth) |
| 认领工作、排队、分算力 | [NOW 唯一队列](NOW.md#统一工作队列唯一优先级账本) → [runbook](runbook.md#统一队列排序与算力纪律) |
| 新增实验 | [`experiments/README.md`](experiments/README.md) + [模板](experiments/TEMPLATE.md) |
| 动作库 / 新动作 | [canonical 终审](experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md) → [预处理合同](interfaces/motion_preprocessing_contract.md) → [编译操作](operations/run_motion_face_shift.md) |
| 原生 MuJoCo `Trainer-v0` | [MuJoCo 原生下一版准备账](experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) → [MuJoCo 旧实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md) → [G06](gates/G06_isaac_to_mujoco.md) |
| 拍面符号 / 判分复核 | [Face-sign forensic](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md) → [G05](gates/G05_isaac_training_first_loop.md) |
| Reward 是否按上台给分 | [physical-truth 审计](experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md) |
| 恢复 / 等待 / 连续对打 | [NOW 连续能力](NOW.md#22-连续能力每个课程阶段都要另考) → [Recovery A/B/C](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md) → [T1 接口](interfaces/t1_event_training_contract.md) |
| Gate3 演示 / planner / ROS | [G06](gates/G06_isaac_to_mujoco.md) → [`run_gate3_first_tick_harness.md`](operations/run_gate3_first_tick_harness.md) / [`run_planner.md`](operations/run_planner.md) |
| 真机 / 部署 | [G07](gates/G07_mujoco_to_real.md) → [`run_deploy_dryrun.md`](operations/run_deploy_dryrun.md)（安全 gate 未过不得下发真机命令） |
| 恢复 ignored/local 资产 | [`setup_local_sync.md`](operations/setup_local_sync.md) + [`ASSET_POLICY.md`](ASSET_POLICY.md) |

## 历史入口（不参与当前发射）

- [upper N3 三反手安全热启动实验](experiments/2026-07/EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md)
  与其[旧发射工序](operations/run_upper_n3_backhand_safe.md)只用于回看历史结果；当前训练入口是上表的
  fresh exact N5，不得再从 N3 路径领取活跃 GPU 或续成正式长跑。

## Gate

[G00 材料](gates/G00_materials_and_harness.md) ·
[G01 真实准备](gates/G01_real_preparation.md) ·
[G02 数据采集](gates/G02_data_acquisition.md) ·
[G03 处理与标定](gates/G03_data_processing_and_physics_calibration.md) ·
[G04 建模](gates/G04_sim_modeling_mujoco_isaac.md) ·
[G05 训练闭环](gates/G05_isaac_training_first_loop.md) ·
[G06 跨引擎](gates/G06_isaac_to_mujoco.md) ·
[G07 部署](gates/G07_mujoco_to_real.md) ·
[G08 盲点路线图](gates/G08_blind_spot_improvements.md)

## 接口合同（改约定前必读对应那份）

[Policy 观测/动作](interfaces/policy_observation_action.md) ·
[按动作条件化 Ball-first 任意 N 动作](interfaces/action_conditioned_ball_first_contract.md) ·
[Task-first 历史消融](interfaces/task_first_n_action_contract.md) ·
[动作能力 selector](interfaces/action_capability_selector_contract.md) ·
[关节顺序](interfaces/joint_order_and_robot_state.md) ·
[坐标系](interfaces/frames_and_coordinates.md) ·
[球拍接触几何](interfaces/racket_contact_geometry.md) ·
[**球拍目标物理有效性**](interfaces/racket_target_physical_validity.md) ·
[Plant 语义](interfaces/plant_semantics_contract.md) ·
[T1 事件时序](interfaces/t1_event_training_contract.md) ·
[稀疏 Reward 账本](interfaces/sparse_reward_eligibility_ledger.md) ·
[横向扰动 adapter](interfaces/lateral_perturbation_adapter_contract.md) ·
[q50 持久启动](interfaces/q50_persistent_supervisor_contract.md) ·
[轻量队列绑定](interfaces/lean_training_run_binding.md) ·
[随挥教师工件](interfaces/post_swing_teacher_artifact.md) ·
[动作预处理与注册](interfaces/motion_preprocessing_contract.md) ·
[脚步幅度](interfaces/footwork_scale_contract.md) ·
[场馆 profile](interfaces/venue_profile.md) ·
[ROS topic](interfaces/ros_topics.md)

## 操作页

四条主工序在上表。其余 `operations/*.md` 是**某一次实验波的专用运行器**，
一页对一个 `scripts/run_*.py`；从[实验索引](experiments/README.md)进，不在本页展开。
常驻的三页：[构建与测试](operations/build_and_test.md) ·
[共享 RunPod](operations/run_on_runpod.md) ·
[跑批作战手册 runbook](runbook.md)。

## 待工程化的规则

[应当变成代码闸门而不是文字的规则](operations/rules_that_should_be_gates.md)——
每条给出确切检查与落点。文档写十遍不如闸门拦一次。
