# 工作文档索引

这是人和 agent 共用的一站式文档路由。按要求先读
[`START_HERE.md`](START_HERE.md)，再读本文件，之后只打开与当前任务对应的那一行；不要遍历
整个 `docs/` 目录，也不要把旧归档当成默认阅读材料。

## 四个当前真源

| 要找什么 | 权威位置 |
| --- | --- |
| 当前完整训练流程、现行课程阶段、分动作成绩卡、横向连续能力和当前责任人 | [`NOW.md`](NOW.md) |
| 实验假设、run、证据与采用/拒绝决定 | [`experiments/README.md`](experiments/README.md) |
| 已进入 `main` 的重要能力和根因修复 | [`TIMELINE.md`](TIMELINE.md) |
| Gate 状态与可复现验收 | [`gates/`](gates/) |

遇到 `SZ`、`v4rg`、`q50`、`K100`、`Gate3-D0`、`E0–E5` 等缩写时，只看
[术语与人话对照](DEFINITIONS.md)。每个 `run_name/flag` 第一次出现也必须带人话，
不得要求读者去历史归档猜。

当前一句话状态：仍在阶段 1 固定点训练。最接近正式目标的模型在四个独立初始化间极不稳定，
拍面正负判分也未过诚实门；16 条 fresh 广度臂在 24/24 最近格的正手 signed composite 均为 0 后，
已分两波全部保留证据并停止，这不是改写 q10 预注册阈值；现有成绩只是每题重置的
Python BankExam 解析诊断，不是 `Gate3`。
第 4000 次迭代后续卷已完成，四 seed 为 `50/88/98/0`，预注册稳定门全失败；且
seed2/3 正手 parsed 高分与 signed composite `0/50` 直接矛盾；signed-face 源码/负控门已实现，
但 fresh 训练 canary 与修正后同卷尚未运行，所以新尺仍未通过行为验收；
原生 MuJoCo 训练仍只有不允许合入的预检候选；
当前 planner-policy exact tuple 源码已通过 portable Release 与 latest-main 本地回归，但
[半秒冲刺五臂](DEFINITIONS.md#half-second-sprint-arms)中的 W/Y 虽已通过真实零写入 plan、fresh `179→31`
ONNX 结构检查与 CPU 推理，两份 checkpoint lineage 和导出 contract 都是 inexact，因此制品只能诊断；
本分支对 [NOW 唯一队列](NOW.md#统一工作队列唯一优先级账本)的候选更新是先修 exact lineage，再实现同卷 vendor adapter。
Wave A v8 科学长训 attempt1 只发 W-N 后在 `sim.reset` 收口、没有首个 iteration 或 Reward 结果；fresh
v9/probe10 已预注册但尚未发射，六格仍 inconclusive / not adopted；
ROS/Jazzy/AimRT、backend first tick 和厂商行为仍未运行；`Gate3B`、标定后机器人物理和新真机测试也都
没有结果。

## 按任务划分的最小阅读集

| 任务 | 只读这些文件 |
| --- | --- |
| 理解或修改训练 setting | [NOW 完整流程与当前阶段](NOW.md#1-当前一套训练是怎样完整跑起来的) → 相关实验 → [G05](gates/G05_isaac_training_first_loop.md) → [`run_training.md`](operations/run_training.md) |
| 认领工作、排队或分配算力 | [NOW 唯一队列](NOW.md#统一工作队列唯一优先级账本) → [跑批作战手册](runbook.md#统一队列排序与算力纪律) → 对应实验 run table |
| 新增/运行消融 | [`experiments/README.md`](experiments/README.md) + [模板](experiments/TEMPLATE.md) → [G05](gates/G05_isaac_training_first_loop.md) → 训练操作文档 |
| 半秒击球冲刺（已结束） | [二十三个单 seed 问题与运行映射](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md) → [严格 0.5 秒负结果](experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md) → [G05](gates/G05_isaac_training_first_loop.md) |
| 当前平衡/动作平滑诊断 | [Wave A processed-qdes slew](experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md) → [连续恢复顺序](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md) → [G05](gates/G05_isaac_training_first_loop.md)；Wave B 的 M0 moving teacher 已 stance `0/4` 拒绝，只继续静态 v4rg/non-demo 合同审计 |
| 原生 MuJoCo `Trainer-v0`/fine-tune | [MuJoCo 实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md) → [v0 preflight](research/mujoco_training_v0_preflight_2026-07-12.md) → [G06](gates/G06_isaac_to_mujoco.md) |
| 评估/checkpoint 排名 | [Fresh 稳定性](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)或[历史尺](experiments/2026-07/EXP-P1-HISTORICAL-SCHEMA3.md) → [G06](gates/G06_isaac_to_mujoco.md) |
| 拍面符号/解析判分复核 | [Face-sign forensic](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md) → [术语：raw-A/physical-B](DEFINITIONS.md) → G05/G06 |
| 当前 Reward 是否真的按上台给分 | [Reward / physical-truth 审计](experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md) → [G05](gates/G05_isaac_training_first_loop.md) → [G06](gates/G06_isaac_to_mujoco.md) |
| 当前 179-D `Gate3-D0` 演示 | [`Gate3-D0` 实验](experiments/2026-07/EXP-GATE3-CURRENT179-D0.md) → [exact planner-policy build](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) → [G06](gates/G06_isaac_to_mujoco.md) → [`run_gate3_first_tick_harness.md`](operations/run_gate3_first_tick_harness.md) |
| 恢复/等待/连续对打 | [NOW 连续能力](NOW.md#22-连续能力每个课程阶段都要另考) → [Recovery A/B/C](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md) → [横向平衡扰动 source gate](experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md) / [adapter 事务接口](interfaces/lateral_perturbation_adapter_contract.md) → [T1 接口](interfaces/t1_event_training_contract.md) → G05/G06 |
| 借鉴 Jiayi V9 / Yikang 支线并核对泛化局限 | [选择性跨线审计](experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md) → [G05](gates/G05_isaac_training_first_loop.md) / [G06](gates/G06_isaac_to_mujoco.md)；旧 `7/7` 只测挥拍/恢复周期，未测物理触球或落台 |
| 新动作/动作库 | [空间重定向实验（含 B/C 主选 SE(2) 实体化）](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md) + [0.5秒短路径](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md) + [Franco/高点拍压/横移老师设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md) → [S0/M0 exact GVHMR](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md) → [post-GVHMR handoff](experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md) → [canonical-beta](experiments/motion_canonical_beta_s0_m0_20260713.md) → [exact GMR/横移脚距](experiments/motion_exact_gmr_s0_m0_20260713.md) → [G08](gates/G08_blind_spot_improvements.md) → 动作操作文档 |
| Planner 或 ROS runtime | [G04](gates/G04_sim_modeling_mujoco_isaac.md) / [G06](gates/G06_isaac_to_mujoco.md) → [`run_planner.md`](operations/run_planner.md) → 下方接口文档 |
| 真机/部署 | [G07](gates/G07_mujoco_to_real.md) → [`run_deploy_dryrun.md`](operations/run_deploy_dryrun.md)；安全 gate 通过前不得下发真机命令 |
| 恢复 ignored/local 资产 | [`setup_local_sync.md`](operations/setup_local_sync.md) + [`ASSET_POLICY.md`](ASSET_POLICY.md) |

## 当前实验

| ID | 简短状态 |
| --- | --- |
| [`EXP-P1-HALF-SECOND-SPRINT`](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md) | 已结束：U/V/W/X/Y 到 `+1000`，W/Y 为诊断候选；真实 plan/export 已通过结构与推理，但 exact-lineage=`0` 阻断 production/vendor |
| [`EXP-P1-BALANCE-ACTION-SLEW-20260720`](experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md) | v8 science attempt1=`infrastructure-only / non-science`：W-N 在 `sim.reset` 收口，余五格未发；fresh v9/probe10 已预注册、未发射，仍 inconclusive / not adopted。不得绕过 `T0→T1→T2`；Wave B M0 moving teacher 继续拒绝 |
| [`EXP-P1-TASK-REVISION-CUTOVER`](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md) | 旧 rolling 池已精确停止；同一物理球的原子 target/TTS revision、宽准备时间、相位 governor、0.5 秒卷与整数淘汰量尺处于 source 红队，full-scene/行为尚未通过 |
| [`EXP-P1-FACE-PLANT-SCALEOUT`](experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md) | 16 条 fresh 广度臂已分两波全部精确停止并保留证据；旧 face×plant 矩阵不能选 baseline |
| [`EXP-P1-FRESH-SZ-STABILITY`](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md) | 实验 completed/rejected；2k 与 4k 四 seed 稳定性都失败，seed4 持续弱；不晋级 baseline |
| [`EXP-P1-FACE-SIGN-FORENSIC`](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md) | `n/-n` 源码负控与 pre-orient physical-B 门已实现；fresh canary/修正后同卷未跑，旧分不晋级 |
| [`EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715`](experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md) | source/config 审计完成：现役同时使用目标匹配与 achieved-state 解析过网/落台 Reward；Phase-A engine-integrated ball 仍 metrics-only 且无拍球冲量 |
| [`EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN`](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md) | model500 身份通过，但 control/treatment 的 post-swing cold-start 在冻结窗分叉；+500 activation-invalid，已停止并转共享 natural-wrap teacher receipt 修复 |
| [`EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT`](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md) | 关闭已证实内生污染的 post-swing replay，只用单 seed 比较 V1+V2 下 base-decel `0/1`；Pod2 GPU1/2 新 namespace 已预注册 |
| [`EXP-P1-SIGNED-FACE-RESCUE-FUNNEL`](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) | C3/D3 显式零摩擦 L1 已配对终档；同卷 K100/L2 仍阻断 |
| [`EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1`](experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md) | E2 provenance：paired receipt `bb3cd749...bbde`；下一门是同一 immutable K100，不得重跑 |
| [`EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND`](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md) | E2 数据门通过：真实 371 题逐字节 replay 并发布 exact bank/report；新 schedule/paper activation/judge 继续阻断 |
| [`EXP-P1-SIGNED-FACE-EXAM-PAPER`](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md) | 新 bank 绑定的 K100 materializer/activation source gate 与攻击负测已通过；真实 private bank consume 未跑，所以 schedule/activation 尚不存在 |
| [`EXP-P1-HISTORICAL-SCHEMA3`](experiments/2026-07/EXP-P1-HISTORICAL-SCHEMA3.md) | 同题同卷尺已可用于诊断排名；所有历史模型仍为 inexact |
| [`EXP-MUJOCO-NATIVE-TRAINING`](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md) | 实验 blocked；off-main preflight 为 `NO-MERGE`，四个正确性缺口未修；尚无 trusted backend、`VecEnv` 或 PPO smoke |
| [`EXP-MUJOCO-EVAL-FRAME-INTEGRATION`](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md) | evaluator parity guard、pelvis COM/link-origin、XBODY gyro 与 leg-mask provenance 已在 feature 集成回归；行为卷未跑 |
| [`EXP-RECOVERY-TUPLE-ABC`](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md) | A/B/C 旧结构合同已验证；T0/T1/T2 与新 reward 次序仅完成文档设计，machine prereg 待同步 |
| [`EXP-V9-YIKANG-CROSS-LEARNING-20260715`](experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md) | 只读审计完成：保留定向恢复、vector settle、动作首帧准备态和随机长等待为候选；旧 `7/7` 无物理触球/落台，且固定正手区不能证明球路泛化 |
| [`EXP-P1-LATERAL-BALANCE-PERTURBATION`](experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md) | E1 source gate：scheduler/显式 COM adapter/default-off trainer 与 hard contract 已接，`173 passed`；full-scene/solver-response/throughput/held-out paper pending，禁止 launch |
| [`EXP-MOTION-SPATIAL-RETARGET`](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md) | B 主选的 schema-2/FK 一次性 consume 已通过并解锁 L0；C 保持未消费后备，安全/动力学/RL 仍阻断 |
| [`EXP-GATE3-CURRENT179-D0`](experiments/2026-07/EXP-GATE3-CURRENT179-D0.md) | 实验 blocked；`Gate3-D0` 严格模型 preflight 通过，当前行为尚未运行 |
| [`EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD`](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) | 实验 completed；exact 源码通过 portable Release 与 latest-main 回归，runtime gates 保持 open |
| [v12/高点拍压/横移视频登记](experiments/motion_video_intake_v12_static_motion_20260713.md) | 7 段私有视频逐字节登记完成；没有动作处理、安全或行为结论 |
| [Franco 优先、static/motion GVHMR 结果](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md) | Franco 六段复用旧 exact 结果；S0/M0 已 `1/1 + 4/4` finite structural pass；后续 exact GMR 状态见专门卷宗 |
| [S0/M0 post-GVHMR exact handoff](experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md) | runtime handoff 已完成：S0/M0 exact SHA `d57a93e0...a1054` / `60c55150...088ef`；后续 exact GMR 已回收，schema-2 仍未授权 |
| [S0/M0 exact donor canonical-beta](experiments/motion_canonical_beta_s0_m0_20260713.md) | 真实 `1+4` 条 PT 已在绑定 CPU runtime consume 且 non-beta bit-exact；后续 exact GMR completions 已回收，正式门仍关闭 |
| [S0/M0 exact GMR 与横移脚距](experiments/motion_exact_gmr_s0_m0_20260713.md) | S0/M0 v2 均 `complete_exact_gmr_diagnostic`：S0 结构通过但需独立高球题族；M0 四条结构通过但 stance `0/4`，input-gate rejected/no-RL；formal/schema2/training/hardware 全 false |
| [v12/高点拍压/横移组合设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md) | S0/M0 已完成 GVHMR、canonical-beta 与 exact GMR 诊断；S0 需高球题族，M0 需末态 stance 修复，schema-2 未授权 |
| [非击球臂模仿消融](experiments/non_striking_arm_imitation_ablation_20260713.md) | Partial：A0/A1 checkpoint、finite、lineage、contract 与 paired result 已闭合；signed K100 尚未判卷，不再写作运行中 |

## Gate 索引

| Gate | 范围 |
| --- | --- |
| [G00](gates/G00_materials_and_harness.md) | 材料与 harness |
| [G01](gates/G01_real_preparation.md) | 真实环境准备 |
| [G02](gates/G02_data_acquisition.md) | 数据采集 |
| [G03](gates/G03_data_processing_and_physics_calibration.md) | 数据处理与物理标定 |
| [G04](gates/G04_sim_modeling_mujoco_isaac.md) | MuJoCo/Isaac 建模 |
| [G05](gates/G05_isaac_training_first_loop.md) | 训练闭环与 checkpoint lineage |
| [G06](gates/G06_isaac_to_mujoco.md) | 跨引擎评估与 Gate3 前置条件 |
| [G07](gates/G07_mujoco_to_real.md) | 部署与真机安全 |
| [G08](gates/G08_blind_spot_improvements.md) | 动作/恢复/旋转/泛化的长期路线图 |

## 操作索引

| 操作 | 文件 |
| --- | --- |
| 构建与测试 | [`build_and_test.md`](operations/build_and_test.md) |
| Isaac 训练 | [`run_training.md`](operations/run_training.md) |
| 稀疏 Reward milestone 资格判读 | [`run_sparse_reward_milestone_classifier.md`](operations/run_sparse_reward_milestone_classifier.md) |
| A0/A1 非击球臂模仿配对 | [`run_phase1_non_striking_arm_imitation_a01.md`](operations/run_phase1_non_striking_arm_imitation_a01.md) |
| 随挥结束 natural-wrap 教师状态 capture/attestation | [`run_post_swing_teacher_capture.md`](operations/run_post_swing_teacher_capture.md) |
| 统一队列、排序与算力 | [`runbook.md`](runbook.md) |
| 共享 RunPod 作业 | [`run_on_runpod.md`](operations/run_on_runpod.md) |
| 动作专属 YAML 训练队列 | [`run_lean_training_queue.md`](operations/run_lean_training_queue.md) |
| Fresh model-4000 q50 考卷 | [`run_phase1_fresh_sz_model4000_seed_stability_q50.md`](operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md) |
| C3/D3 同卷 signed-face K100 | [`run_phase1_signed_face_c3d3_k100.md`](operations/run_phase1_signed_face_c3d3_k100.md) |
| Signed-face K100 checkpoint 取证（不启动判卷） | [`run_phase1_signed_face_k100_checkpoint_attestor.md`](operations/run_phase1_signed_face_k100_checkpoint_attestor.md) |
| 恢复 preregistration | [`run_phase1_recovery_tuple_prereg.md`](operations/run_phase1_recovery_tuple_prereg.md) |
| 有符号拍面单-seed L1 漏斗 | [`run_phase1_signed_face_rescue_funnel.md`](operations/run_phase1_signed_face_rescue_funnel.md) |
| 有符号拍面 C2/D2 provenance L1 | [`run_phase1_signed_face_cd_l1.md`](operations/run_phase1_signed_face_cd_l1.md) |
| 有符号拍面 C3/D3 显式零摩擦 L1 | [`run_phase1_signed_face_c3d3_l1.md`](operations/run_phase1_signed_face_c3d3_l1.md) |
| 有符号拍面 exam bank 严格重绑定 | [`run_phase1_signed_face_exam_bank_rebind.md`](operations/run_phase1_signed_face_exam_bank_rebind.md) |
| 有符号拍面 exact K100 物化（不启动判卷） | [`run_phase1_signed_face_exam_k100.md`](operations/run_phase1_signed_face_exam_k100.md) |
| Gate3 首个有效周期 | [`run_gate3_first_tick_harness.md`](operations/run_gate3_first_tick_harness.md) |
| Gate3 发球同步负向设计门 | [`run_gate3_serve_sync_prereg.md`](operations/run_gate3_serve_sync_prereg.md) |
| 端到端乒乓链路 | [`run_pingpong_end_to_end.md`](operations/run_pingpong_end_to_end.md) |
| 部署 dry-run | [`run_deploy_dryrun.md`](operations/run_deploy_dryrun.md) |
| 语义正确的 plant | [`prepare_semantics_correct_plant.md`](operations/prepare_semantics_correct_plant.md) |
| 动作空间重定向、主选与整轨实体化 | [`run_motion_spatial_retarget_screen.md`](operations/run_motion_spatial_retarget_screen.md) |
| 第0帧零速准备态到0.5秒触球候选 | [`run_ready_to_strike_motion.md`](operations/run_ready_to_strike_motion.md) |
| 新视频离线 GVHMR | [`run_motion_video_gvhmr_prereg.md`](operations/run_motion_video_gvhmr_prereg.md) |
| S0/M0 post-GVHMR exact 消费 | [`run_motion_post_gvhmr_exact.md`](operations/run_motion_post_gvhmr_exact.md) |
| S0/M0 exact donor canonical-beta | [`run_motion_handoff_canonical_betas.md`](operations/run_motion_handoff_canonical_betas.md) |
| S0/M0 exact GMR 与横移脚距 | [`run_motion_s0_m0_exact_gmr.md`](operations/run_motion_s0_m0_exact_gmr.md) |
| 本地/已忽略资产恢复 | [`setup_local_sync.md`](operations/setup_local_sync.md) |

## 接口索引

| 合同 | 文件 |
| --- | --- |
| Policy 观测/动作/导出 | [`policy_observation_action.md`](interfaces/policy_observation_action.md) |
| 关节顺序与机器人状态 | [`joint_order_and_robot_state.md`](interfaces/joint_order_and_robot_state.md) |
| 坐标系与坐标 | [`frames_and_coordinates.md`](interfaces/frames_and_coordinates.md) |
| 球拍接触几何 | [`racket_contact_geometry.md`](interfaces/racket_contact_geometry.md) |
| Plant 语义 | [`plant_semantics_contract.md`](interfaces/plant_semantics_contract.md) |
| T1 事件/恢复时序 | [`t1_event_training_contract.md`](interfaces/t1_event_training_contract.md) |
| 稀疏 Reward 资格账本 | [`sparse_reward_eligibility_ledger.md`](interfaces/sparse_reward_eligibility_ledger.md) |
| 横向扰动 scheduler→adapter 事务 | [`lateral_perturbation_adapter_contract.md`](interfaces/lateral_perturbation_adapter_contract.md) |
| q50 持久启动与只读复核 | [`q50_persistent_supervisor_contract.md`](interfaces/q50_persistent_supervisor_contract.md) |
| 轻量训练 queue claim→真实日志→checkpoint 绑定 | [`lean_training_run_binding.md`](interfaces/lean_training_run_binding.md) |
| 随挥结束教师状态 raw capture→attestation→首 reset | [`post_swing_teacher_artifact.md`](interfaces/post_swing_teacher_artifact.md) |
| ROS topic | [`ros_topics.md`](interfaces/ros_topics.md) |

## 责任归属与更新规则

“人类责任人”必须写人名；Claude/Codex 只写在“执行者”字段。运行前先更新对应实验记录；
重要逻辑变化进入 `main` 后，只在 TIMELINE 更新一次；采用新 setting 或出现新的最佳候选成绩表后，
更新 NOW。受影响的 gate 与 operation/interface 合同始终要同步更新。历史流水归档在
[`experiments/archive/`](experiments/archive/README.md)，不作为默认阅读材料。

`origin/main:docs/NOW.md` 是唯一运行态权威。feature 分支可以提交结构重构提案，
但分支副本不算认领或改了优先级；合入前必须基于最新 main 复核三本账。
