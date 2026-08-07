# 工具目录（扫一遍，找到那一行，去读它的 docstring）

**先查这一页再造新工具。** 每行只说"干什么"和"怎么算过"；完整说明在工具自己的 docstring 里，
本页不复制。路径默认相对 `hope_training/whole_body_tracking/scripts/`（下称 `wbt`）。

`退出码` 列：`0/1/2` = PASS/WARN/FAIL；`2` = fail-closed；`—` = 只打印**没有门**（人工读）。
`调用` 列：`工序` = 有 ops/gate 页引用；`内部` = 只被别的工具调；`测试` = 只有单测；
**`无`** = 全仓库无调用方（2026-07-26 扫描 822 个文件）。

## 动作片：转换与落地

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `csv_to_npz.py` | CSV→npz 的**规范生成器**，其他转换器对齐它 | — | 工序 |
| `csv_to_npz_mujoco.py` | 同上但用 MuJoCo FK，不需要 Isaac | — | 工序 |
| `reground_hope_frame.py` | 整体偏航归零，让第 0 帧骨盆朝 HOPE +X | — | 工序 |
| `migrate_motion_kinematics.py` | 老 npz → schema-2；**不带标记必须显式 `--source-point`** | — | 工序 |
| `motion_kinematics_contract.py` | schema-2 列语义单一真源（`body_lin_vel_w` 是**质心**速度） | 库 | 内部 |
| `make_static_motion.py` | 造一条"站着不动"的合法 npz，验管线用 | — | 工序 |
| `trim_motion_clip.py` | 裁成触球居中的对称窗 | — | 内部 |
| `replay_npz.py` | Isaac 放片子；`mj_step` 调用数 = 0，**不是动力学证据** | — | 工序 |
| `upload_hopex_0703.py` | 发 registry；上传前验第 0 帧 yaw==0，没落地的拒发 | — | 内部 |

## 动作片：三层可行性闸门（递进）

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `audit_motion_npz.py` | **L0 判炸器**：只看运动学数字对 URDF 限位。numpy-only，任何机器都能跑 | 0/1/2 | 工序 |
| `audit_self_collision.py` | **L1 自碰撞**：厂商 MJCF 逐帧 `mj_forward`，拍子↔躯干、前臂↔躯干 | 0/1/2 | 工序 |
| `motion_dynamic_replay.py` | **L2 真动力学**：唯一真调 `mj_step` 的工具。**先读它 docstring 的"已测事实"段**——纯厂商 PD 连静态站立都撑不满 2 s，要当**相对分级器**用 | 0/2 | **无** |
| `mujoco_motion_player.py` | schema-2 解码 + 32-body FK 冒烟 | 2 | 工序 |
| `scripts/feasibility_oracle.py` | A 层动力学裁判（`mj_inverse` 逐帧 CoP/摩擦/τ）。**剂量制不用峰值制** | 0/1/2 | 内部 |

## 动作片：路径与时间律（path / time / feasibility 刻意分三个文件，别合并）

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `topp_mintime.py` | **TOPP v3 min-time**：病动作被放慢、健康动作被加速，两侧推到同一预算边界 | — | 工序 |
| `synthesize_timing.py` | 时间律 v1：静止起步→匀加速→过触球→减速。**速度不可达=延长加速时间，绝不降速** | 1/2 | 内部 |
| `synthesize_timing_v2.py` | TOPP-lite：只在 oracle 判不可行的帧局部拉长 | 1 | 内部 |
| `retime_motion_clip.py` | 非均匀重定时：哪里猛哪里慢；触球窗内必须**逐位**拷贝 | — | 内部 |
| `extend_stroke.py` | 引拍加深（触球前）。`a_min=v*²/(2L)` 只有加行程能治 | 1 | 测试 |
| `bake_topp_strike_speed.py` | 把"拍速×r"烤成独立资产。**越界即拒绝出货，禁止静默降速**；**r<1 默认不再出货**（运行时慢放交付，`--force-slowdown-bake` 才强出） | 3 | 内部 |
| `canonical_playback_speed_gate.py` | 慢放门：验 `q(t)→q(st)` 在不在包络里。**重力项 c0 不随 s 缩放**，静态超限=任何慢放都不可行；接触力破坏二次律就 fail loud，拒绝给 full scope 发证 | 3 | 内部 |
| `rewrite_followthrough.py` | 随挥段路径重写，压 CoP 剂量 | 1 | **无** |
| `topp_budget_search.py` | 报 **timing-irreducible 剂量** = "时间律救不救得了这条片" | 1/2 | **无** |
| `scale_footwork.py` | 脚步幅度 β 组合规格**骨架**（出 JSON，不是可训练资产） | 2 | 测试 |
| `analyze_rally_intervals.py` | 保守的同人来回间隔（只收 A→B→A 连续检测） | — | 工序 |

## 相位、瞄向与拍面（**2026-07-26 事故现场**）

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| **`gen_stage1_questions.py --phase-scan`** | **可回球性相位扫描**：逐帧问"若在此帧触球，多少比例来球能被合法回台"。空挥片只有这个训练最优相位有意义。**从不回写 registry**，只打印 `train_phase_candidates` 给人抄 | 打印 | **无**（工具本体被大量调用，**这个模式**没有任何 ops/gate 页引用；canonical 动作库从未跑过 → 9/11 片一帧都回不了球） |
| `analyze_strike_phase.py` | 找触球相位。默认量**拍面**不是裸手腕（量手腕短 0.1–0.2 m、慢 0.5–0.9 m/s） | — | 工序 |
| `suggest_face_sign.py` | 拍面符号 `sign(n·v)`。**只能离线算成常量**，运行时动态定符号会把反面击球合法化 | 1 | 内部 |
| `solve_chingmu_canonical_racket_full_phase.py` | 把同钟实测 blade/face/long-axis/point-velocity 作为末端约束，重定向到官方 `right_racket`；逐动作全门通过才物化 repaired PKL，不授权训练 | 0/2 | [恢复/复现工序](setup_local_sync.md#chingmu-measured-racket-rebuild-contract) |
| `resign_chingmu_catalog_from_measured_hit.py` | 用逐动作 measured-hit 证据发现并冻结 robot mount face sign，生产 solver 只读固化 sign，不在优化中偷偷翻面 | 0/2 | [恢复/复现工序](setup_local_sync.md#chingmu-measured-racket-rebuild-contract) |
| `materialize_measured_racket_motion_npz.py` | 将 repaired q 与实测拍子对齐到同一 50 Hz 时钟，用当前 MJCF 重算全身 FK，并把 URDF-visual butt-to-blade axis/mesh SHA 写入 fail-closed measured-racket schema-v4 NPZ | 0/2 | [恢复/复现工序](setup_local_sync.md#chingmu-measured-racket-rebuild-contract) |
| `audit_materialized_measured_racket_npz.py` | 从物化后 joint/root 独立重算 site/face/long-axis/拍速的11门；不读 solver 成功标志代签 | 0/2 | [恢复/复现工序](setup_local_sync.md#chingmu-measured-racket-rebuild-contract) |
| `audit_measured_racket_mechanical_admission.py` | measured-racket bank 全动作/全帧机械准入：验 URDF position、stored/FD velocity，计算 FD acceleration；缺 authority acceleration、torque-speed curve 或逐帧 inverse-dynamics torque 即 fail-closed，永远保留 `diagnostic_unauthorized` | 0/1/2 | [G03](../gates/G03_data_processing_and_physics_calibration.md) |
| `optimize_reachable_face.py` | 物理最优拍面 vs 机器人可达最优的差距 | — | 测试 |
| `racket_geometry_contract.py` | 把 `site` / `face center` / `ball center` 三个点分清，别都叫"拍心" | 库 | 工序 |
| `racket_fk_ref.py` | C++ `pp_racket_fk.hpp` 的 ground truth，门 < 1e-4 m | — | **无** |

## canonical 五动作编译链（工序见 [`run_motion_face_shift.md`](run_motion_face_shift.md)）

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `canonical_motion_compile_cli.py` | 编译入口。**过了它不授权训练/部署/硬件** | 2 | 工序 |
| `materialize_a3_stable_upper_motion.py` | A3 upper 保留 head/arm 与腰动作增量，把腰 ready/lower/root 重建到 runtime stable stand；**拍位变化后必须重绑球题** | 2 | [工序](run_ablation_wave_launch.md#actionball-a3-upper-qqd-修复与-hot-path-快线) |
| `materialize_a3_dynamic_ready_contract.py` | 把动作 frame-0 physical ready 与低增益 A3 plant 所需 hold qdes 分开；用 exact MuJoCo ground LP 给候选，**仍须 Isaac nominal-hold 验证才可训练**。2026-08-07：hold LP 无解时的拒绝改成会自陈（哪个关节、差多少 `N·m`、要多大 `q_des`、卡的是电机还是 `kp × 行程`），门本身一格没松；`--physical-birth-composition-mode projected_teacher_frame0_grounded` 那条支路的 "failure baseline" 标注**理由已换**——不再是 `0/73` 悬空（已推翻），而是腰的保持增益不够，见 §5.6.7「十二」 | 1 | [工序](run_ablation_wave_launch.md#actionball-a3-upper-qqd-修复与-hot-path-快线) |
| `audit_position_hold_authority.py` | **未授权诊断**：位置指令撑不撑得住某个姿态。只对地面永远使不上力的那 `19` 个非腿关节说话（对它们 `tau` 唯一等于 `qfrc_bias`），把"增益不够"和"关节已站在可发指令包络之外"分开命名。支持单帧 / 整条 clip / 整库 / 现役 ready artifact | 0 | §5.6.7「十二」 |
| `canonical_motion_bank_gate.py` | **独立复核器**——编译器不许给自己发证，必须从 exact bytes 重算 | 2 | 工序 |
| `canonical_mujoco_dynamics_gate.py` | 逆动力学筛查。浮动基 fail-closed：λ 未定则 τ 未定，那 31 行只叫 `joint_effort_proxy` | 2 | 工序 |
| `canonical_face_manifold.py` | 七关节有符号拍面流形，保持 site 世界位置 | 2 | 工序 |
| `canonical_motion_registry.py` | registry → 四张否则会各自漂移的运行时表 | — | 工序 |
| `canonical_motion_admission.py` | **唯一**把证书变成运行时能力的地方；信任集代码拥有、**出厂为空** | 库 | 内部 |
| `canonical_protected_window.py` | 击球窗内容摘要，**窗口永不能跨件拼接** | 库 | 工序 |
| `canonical_motion_recipe.py` / `_markers.py` / `_body_scope.py` / `_geometry.py` / `_path_topp.py` / `_torque_path_topp.py` / `_schema2_builder.py` / `_root_pose_codec.py` / `_weighted_arc_path.py` / `_compiler.py` | 编译链各层（配方/marker/body scope/几何/运动学 TOPP/力矩 TOPP/schema-2/root 编码/弧长/主编译器）。每层只发一种证据，**不自动升级** | 库 | 工序 |
| `canonical_grounded_ready.py` | ready 位必须是真能双脚站住的静态姿势 | 库 | **无** |
| `canonical_frame_identity.py` | 时间帧身份收据（marker v2 的绑定前置） | — | **无** |
| `canonical_neutral_ready.py` + `_cli.py` | 面中立 ready 挑战者 | 2 | **无** |
| `canonical_mujoco_identity.py` / `_path_adapter.py` / `canonical_time_law_artifact.py` | 模型身份 / 切空间换基 / 时间律工件 | 库 | 测试 |

## 题库与考卷

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `gen_stage1_questions.py`（bank 模式） | 出题。**只留可答的题**；可答率低 = 击球点选坏了，换点不要硬训。`--check` 不许关 | — | 工序 |
| `build_action_ball_manifest.py` | 从 ChingMu batch 构造 ActionBall source manifest；`measured_channel` 必须绑定 schema-v4 bank receipt、NPZ/URDF-visual axis/mesh SHA 且禁止 FK fallback，但生成 JSON 不授权训练 | 0/2 | [measured-racket 工序](setup_local_sync.md#chingmu-measured-racket-rebuild-contract) |
| `venue_ball_sampler.py` | 场馆球采样：答"扛不扛得住真扔出来的球" | 库 | 工序 |
| `virtual_return_scorer.py` | 判分的 NumPy 规范实现。**别换成 planner 的 1 ms Euler**，那是另一个前向模型 | 库 | 工序 |
| `subset_stage1_question_bank.py` | 从 exact 多动作 schema-3 题库按源顺序投影子集；逐 clip 数组 bitwise 保持、重算 metadata SHA、拒绝覆盖并用 strict loader 复核 | 2 | 工序 |
| `bank_exam_schedule.py` / `materialize_bank_exam_schedule.py` | 不可变考卷排程；两条模拟器腿共用同一份 JSON | — | 工序 |
| `isaac_bank_exam.py` + `_adapter.py` + `isaac_timing_exam_adapter.py` | Isaac 侧考卷；每个 env 分到恰好一道不可变考题 | — | 工序 |
| `rebind_question_bank_motion_family.py` | 登记烤入件的 SHA。**题目张量一个字节都不改**；`contact.row_bitwise` 必须 true | 2 | 内部 |
| `stroke_guard_bank_audit.py` | 统计"开行程守卫会拦掉多少题"。**PASS ≠ 可行**（松弛的必要条件） | — | 测试 |
| `isaac_ball_inloop_check.py` | PhysX 球飞行 vs RK4。**只验飞行**，弹台/拍球接触是另一条 | — | 工序 |
| `diagnose_n1_teacher_rate_tape.py` | 只读复用 exact N=1 fixed-action solver，对确定性来球/落点 proposal tape 报 teacher-rate、目标拍速、admission 与逐 reason reject；速度/落点 override 只改内存副本，不授权训练 | 0/2 | [分阶段准备账本](../experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md) |
| `materialize_n1_fast_ball_bootstrap.py` | 从 sealed N1 bundle 只派生来球速度/落点 support，内容寻址生成 fast-ball diagnostic manifest/bundle；不改 action/motion/solver/physics/teacher-rate bounds，低于 formal admission 时必须显式降级 | 0/2 | [分阶段准备账本](../experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md) |

## 发射前静态检查（**2026-07-26 缺陷现场**）

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `check_motion_target_alignment.py` | 长训前快门：yaw≈0 / 拍速 +X 主导 / 参考起点对齐 | 1 | 工序 |
| `check_table_obstacle_scene.py` | 核对 legacy 单桌板或 [ActionBall 五件桌体安全总成](../DEFINITIONS.md#action-ball-table-safety)；可在 Pod 用真实机器人刚体逐子步撞 top/keep-out/net/posts，验证四子步 latch、raw reason 和 reset 不泄漏；`--nominal-hold` 还会分开抓 `raw_env_reset` 与 artifact-ready，再抓 step 1/10/final-or-preterminal 并闭环保持；每个构造/reset/probe 阶段会打印 `HOPE_TABLE_DIAGNOSTIC_STAGE`，防止把 Kit 静默零退出当成成功 | 1 | [桌体工序](run_action_ball_table_safety_smoke.md)；[dynamic-ready 工序](run_ablation_wave_launch.md#actionball-a3-upper-qqd-修复与-hot-path-快线) |
| `check_perclip_pos_sampling.py` | 不开 Isaac 查目标**位置**框，打印"低于桌面"占比 | **—** ⚠ 打印 83% 也 `exit 0`；`TABLE_H` 是本地硬编码副本 | **无** |
| `check_perclip_vel_sampling.py` | 同上，速度框 | **—** ⚠ 同样没有门 | **无** |
| `a3_joint_order_contract.py` | GMR→运行时关节顺序双射。**只是源码门**，不认证 schema-2 | — | 工序 |
| `prepare_a3_p1_0803_31d_asset.py` | 从 exact private 0803 raw intake 产生独立 future-primary 31-action successor：按项目明确采用的 raw-URDF `q=0` 固定9个 gripper coordinate，保留21-link/`0.76626209416 kg`夹爪子树，20个缺失 collision 显式 disabled，并拒绝覆盖现役 `agibot_a3/`。当前 `--check` 与 producer host 回归均 PASS（`6 passed`）；normalized closure=`73a47e85…8f08`。这只关闭可复现资产生成，不代签新USD、retarget/hold、MuJoCo identity或canonical long | 0 | [0803 归一化记录](../experiments/2026-08/EXP-A3-P1-0803-31ACTION-NORMALIZATION-20260803.md) |
| `termination_contract.py` | 终止项冻成评估合同。真源是 run 自己的 `env.yaml`，**不从 run 名推行为** | 库 | 工序 |
| `verify_realsensor.py` | deploy-parity 观测合同四种检查（layout/rollout/golden/onnx） | 2 | 工序 |
| `realsensor_obs_reference.py` | 证明球拍目标重构项**不需要世界系 base 位姿**（线性性把 base 约掉） | — | 工序 |
| `verify_hitter_task.py` / `verify_hitter_pure.py` | 177-D / 110-D 任务变体构建门 | — | 内部 / **无** |
| `inspect_a3_deploy_contract.py` | 训练侧 ONNX 元数据 vs 官方部署常量按关节名对表 | — | 内部 |
| `materialize_n1_fixed_domain_initial_receipt.py` | 从 registry 固定 action、contact bundle 与 production sampler/profile/curriculum 真源无覆盖铸造 [`n1_fixed_domain_initial_receipt_v1`](../DEFINITIONS.md#n1-fixed-domain-initial-receipt-v1)；硬验 32 arms、`no_move` mask、train 的 `SamplingMixture()` wiring 与 1:3:1 mixture，支持 registry SHA 回填后 `--verify` | 0/2 | 工序 |
| `materialize_action_ball_reward_ppo_economy_receipt.py` | 从 clean/tracked r6 registry 投影、两个 runtime contract、effective Reward、task/PPO 配置与 Pod `rsl_rl=2.3.1` 三个 exact source 无覆盖铸造静态 [`action_ball_reward_ppo_economy_receipt_v1`](../DEFINITIONS.md#action-ball-reward-ppo-economy-receipt-v1)；未物化 `noise_std_type=log + realized sigma=.02`、source dirty/index staged 或任一 SHA 漂移即拒绝 | 0/2 | 工序 |
| `audit_action_ball_reward_hierarchy.py` | 读取 A211/C211 严格 reward taxonomy，将旧73库 `T` 算术明确标成 partial；绑定 `--task-receipt` 后才把 hidden WAIT、pre-swing bridge、teacher-rate 和 `gamma=.99` 纳入选定 N1 的 eligible/discounted 会计。候选配方下 C211 的 task-valid `mimic < proximity < landing` 与 A211 的 `mimic < accepted window ≤ target+progress cap < landing` 均静态 PASS；运行时仍须分开 ready/swing mimic、绑定实际 compatible target 收入并监控 landing 与 death 的套利方向。不授权 N73/PPO/Promotion | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md#53-统一会计实际改动与静态层级-gate) |
| `play_table_tennis.py` | 起完整球场看物理和摆位（零动作） | — | 工序 |

## 发射、队列、进程

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| `train.py` | Hydra 训练入口 | 1 | 工序 |
| `launch_n1_measured_vendor_v2_diagnostic.py` + `vendor_v2_gpu_admission.py` | Take_061 fixed-question VendorV2 的 fresh/no-resume diagnostic launcher；前者集成参数/spec/claim 与发射，后者封装 lock、`/proc`、`nvidia-smi`、reservation/receipt validation 和 admission。code-owned [`oracle2`](../DEFINITIONS.md#vendor-v2-oracle2) 固定 `1 env/0 PPO/2 episodes` 作 live auto-reset/ledger smoke；[`oracle32`](../DEFINITIONS.md#vendor-v2-oracle32) 固定 `1 env/0 PPO/32 episodes`，同时验证 p/v/face、capture/reject/unknown、termination 与 weight-independent projection/soft-limit/reference exposure。两者都产 no-clobber oracle JSON，marker 后等 exact child clean exit、确认原 PGID 已空，再走不要求 live PID 的 `post_completion` admission；训练 stage 仍要求 post-boot live PID。默认空 GPU，只有 exact-claim 开启 [`--allow-vendor-v2-colocation`](../DEFINITIONS.md#vendor-v2-gpu-colocation) 才可在已证同 checkout/namespace receipt 的一个 VendorV2 PID 旁发第二个，硬上限两 PID 并保留 `8192 MiB` free headroom；post-boot 拒绝只清理本次 exact PID=PGID 并写 failure receipt，unknown live co-resident fail closed | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md#121-vendorv2-诊断单卡双进程-admission) |
| `materialize_action_ball_a211_frame0_exact_artifact.py` | **SUPERSEDED / COUNTEREXAMPLE 工具，但退役理由已被推翻，待重判**：从 measured motion 原样读 frame-0 root/q并构造零速度历史 artifact。退役写的理由是「direct exact-frame0 physical-birth 同门槛扫描已是 `0/73`」；2026-08-07 实测那个 `0` 是**全库共有的一个 `~1 cm` 悬空**被数了 `73` 遍，成因是重定向的脚部目标少了一项 `~31 mm` 的踝高补偿。补一次接地收尾解算（root 与 `19` 个非腿关节逐位不动、只解十二个腿关节）之后，同一批 clip 的 `sole_floor + double_support` 从 `0/73` 变成 `70/73`。**在重判之前，输出仍然只保留 provenance、`launch_authorized=false`，不得进入 current A/C lineage**；重判要连 artifact 重铸一起走，见 successor 账本 §5.6.7「十一」4 与「十」6(b) | 0 | [退役 exact-frame0 工序](run_action_ball_a211_frame0_nominal_hold.md) |
| `run_action_ball_a211_frame0_nominal_hold.py` / `consume_action_ball_a211_frame0_nominal_hold.py` | **SUPERSEDED / COUNTEREXAMPLE 工具**：历史 exact Pod `62 tick / 248 substep` probe 在 step9 因 `robot_hit_table` raw FAIL，未铸 PASS。当前 physical reset 使用 tracked split-ready `60/240/1.2 s` hold，WAIT 最多25 tick，reveal 后 teacher 切 measured frame0并由 dense mimic 学 bridge；不得用本工具补签 current receipt | 0/2 | [退役 exact-frame0 工序](run_action_ball_a211_frame0_nominal_hold.md) |
| `run_mujoco_c_lite_pod_diagnostic.py` | 默认只读 plan；显式 diagnostic 执行后用真 `MujocoN1DiagnosticVecEnv` 跑 `2 step x 2 PPO update`，在 reset boundary 保存，fresh Python process cold-load，并对比 next-update transition/reason/safety、model/Adam/normalizer/RNG 全部摘要；CPU-only、no-clobber、不授权 formal/deploy | 0/2 | [G06](../gates/G06_isaac_to_mujoco.md) |
| `materialize_action_ball_a211_lineage.py` | current lineage 必须严验 tracked split-ready artifact/hold、measured-frame0 teacher reveal、5--25 tick WAIT、motion/manifest/bundle、fresh 211/319 ABI 与 A complete-semantic cache；不得再消费 exact-frame0 artifact/hold 或 `immutable_tape` formal source | 0 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `launch_action_ball_a211_four_arm_diagnostic.py` | fresh four-grid 的 A211 半边（A0=fixed LR，A1=adaptive-KL initial LR）；formal source=`online_solver`，sampler/curriculum/RNG 每 reset 推进，cold Q/Q' 各解一次，只有完整语义相等才复用 exact answer。主链 `materialize -> recipe -> oracle32 -> scale4096 -> long4096`，512 只定位失败；cache/Pod integration 未闭合即 fail closed | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `launch_action_ball_c211_diagnostic.py` | fresh C211 211/319 ABI、incoming-ball p/v/spin、`task_valid`；formal source=`direct_ball`，不使用 `immutable_tape`/answer cache且总 inverse call=`0`。最小 task reward 只有 nominal-strike 拍心-球心距离与 actual-contact-gated 一次 landing，无 desired-contact、独立 hit bonus或额外 dense outcome | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `action_ball_211_four_grid_contract.py` | 冻结 A0/A1/C0/C1 四格的 task family、PPO schedule、211/319 ABI、teacher/question/plant/safety、4096×5 budget 和 GPU 布局；它是共享 manifest，不启动训练 | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `action_ball_211_four_grid_prelong_barrier.py` | 四格 scale 都自然结束后重开 claim/checkpoint/normalizer/telemetry/effective-reward 和 GPU layout；任何一格缺失或安全账不闭合就拒绝 long | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `action_ball_4096x5_prelong_gate.py` | 对单格 `4096 env × 5 update` 的 finite checkpoint、optimizer、WAIT/task eligibility、balance/mimic/strike/target/outcome 分母和逐原因 safety telemetry做 fail-closed 验收。implementation strict-zero 只含 qdes-hard/actual-hard/nonfinite；fall/too-low/table 按 hidden-WAIT/revealed-pre-strike/post-strike 守恒记行为账，不要求初始 policy 已命中、上台或零行为终止 | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `build_action_ball_banded_question_bank.py` | 可选未来 producer 优化：离线把已解且准入的题按 exact curriculum domain-level/action/profile/solver identity 发布为 no-clobber block bank。它不是 current A/C formal source，也不阻塞首个 expanding long；A 当前以 complete-semantic cache 保留 online curriculum | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `materialize_action_ball_c211_lineage.py` / `action_ball_c211_live_oracle.py` | 分别铸造 C-owned 211/319 lineage，以及收集 32 个 TASK_ACTIVE closed attempts 的 live actor/critic ball-state、contact/flight 和 `0/C` 账；WAIT 不稀释分母 | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `mujoco_native/scripts/launch_mujoco_action_ball_211_diagnostic.py` | MuJoCo-native A211/C211 plan/execute 入口；重开 211/319 source pins、split-ready physical reset、5--25 tick WAIT、measured-frame0 teacher reveal、A online-solver cache/C direct-ball、各自 task/reward 与 plant/MJCF lineage。Host code path 已有 finite PPO/save/cold-load；exact Pod 仍须 A/C **各**跑 `1 env x 2 PPO update + save/fresh-process cold-load`，当前均`未测`，不授权 canonical N1/4096/deploy | 0/2 | [G06](../gates/G06_isaac_to_mujoco.md) |
| `action_ball_211_transition_preflight.py` | 首批 four-grid 发射前在 GPU0/1/2 三把物理锁的同一 common-cut 内确认 legacy writer/reservation 已 drain、GPU 空闲且四个 namespace 尚不存在；放锁后不声称跨时原子性，每个 launcher 仍须即时重验 admission/no-clobber | 0/2 | [successor 账本](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| `launch_action_ball_curriculum.py` | Fresh upper/no-move exact N5 的 V3 双 GPU no-clobber 入口；`plan` 零 GPU 副作用，`launch` 只接受同一 claim 并固定走 smoke→canary→long | 0/2 | [工序](run_action_ball_curriculum_no_clobber.md) |
| `action_ball_stage_supervisor.py` | 继承 GPU0/GPU1 lifetime locks；执行 evaluator→trainer、ready→intent→ACK→accepted→commit-ACK，训练后停 evaluator，并持 boot flock 覆盖 GPU0 exact-resume 的完整 Kit lifetime；失败只 exact cancel/reap | 0/2 | 内部 |
| `action_ball_frozen_eval_sidecar.py` | GPU1 正式评估旁车：运行真实 frozen policy canary/heldout，发布 5 s heartbeat，并把一次性 V4 evidence 写入 inbox | 0/1 | 内部 |
| `action_ball_runtime_inventory.py` | 给 per-run overlay/IsaacLab 闭包 mint 或 verify no-clobber receipt；递归核对依赖、RECORD、pyvenv/.pth/editable Git bytes | 0/2 | [工序](run_action_ball_curriculum_no_clobber.md#pod-runtime-overlay-与-inventory) |
| `action_ball_exact_resume_verifier.py` | Trainer 自然结束且 evaluator 停止后，由仍持双锁的 supervisor 在 GPU0 做真实 Isaac 零 step restore→save→restore；任一状态不精确就拒绝 | 0/2 | 内部 |
| `action_ball_stage_evidence.py` | 签 prelaunch/stage receipt；重读 supervisor terminal、frozen evaluation、Reward、bootstrap、checkpoint 与 exact-resume 全链证据 | 0/2 | [工序](run_action_ball_curriculum_no_clobber.md#训练结束后的-exact-resume-与阶段签名) |
| `launch_upper_n3_backhand_warmstart.sh` | Pod1 GPU1 独占的三反手 175D non-exact warm-start；当前只放行 `1 env × 2 update` 安全 smoke，长跑 fail-closed | 2/3/4 | 工序 |
| `/workspace/bin/kit_boot_lock.sh`（pod 侧） | **pod 级启动锁**，同时只一个 Kit 在 boot | — | 工序 |
| `launch_kit_training_locked.sh` | 仓库版串行发射器。**⚠ 消融波不要用**（`180 s` stale 门是 Wave A v8/v9 死因） | — | 已弃用 |
| `exact_process_group.py` | 只给**一个**进程组发信号，不信任陈旧 PGID | 2 | 工序 |
| `lean_queue_runtime.py` | run binding + 里程碑取证，**从不扫日志猜目录** | 2 | 工序 |
| `full_scene_probe_runtime.py` | 冒烟 probe 监督者，**故意被动、从不发信号** | 2/3 | 工序 |
| `phase1_checkpoint_curve_worker.py` | 判卷流水，一次一个 judge 到 CPU 阶段 | 1 | 工序 |
| `audit_runpod_terminal_runs.py` | 只读盘点失联 run。**缺失/重复命中是硬错误**，不许猜 | 1/2 | 工序 |
| `v5_ablation_accelerator.py` | 大消融保守减半；小样本不确定的留着，机制对照受保护 | — | 工序 |

## 导出与判卷（判读规则见 [`read_and_report_results.md`](read_and_report_results.md)）

| 工具 | 干什么 | 退出码 | 调用 |
| --- | --- | --- | --- |
| **`judge.sh`** | **判卷链标准单命令入口**。解析不到就 fail-loud 要求手传，**绝不静默用默认值** | — | 工序 |
| `mujoco_eval_onnx.py` | 仓库内击球指标工具（官方部署 sim 是厂商 C++ harness，不是它） | 1 | 工序 |
| `make_std_sidecar.py` | 抽 `learned_std.npy` + `obs_norm.npz`。**原生导出都不产**——缺前者得 ~0 分，缺后者 FATAL | — | 工序 |
| `play.py` | 导出 ONNX。**导出后进死循环，必须 kill 进程组** | — | 工序 |
| `standalone_onnx_export.py` | Kit 起不来时的 CPU 导出。**⚠ donor 动作对锁定**，非 hopex 臂必死 | 1 | 工序 |
| `scoreboard_eval.py` | 三协议确定性成绩板；`explicit` = "在这里摔就是在机器人上摔" | — | 工序 |
| `eval_deterministic.py` | 多噪声档无头评估（基线 = 分布均值 = 部署路径） | — | 工序 |
| `probe_metric.py` | 核对记录的 metric 与现场重算是否一致 | — | 工序 |
| `pad_obs_cols.py` / `make_hitter_warmstart.py` / `warm_start_realsensor.py` | 补零列 / 按位插列 / 列选缩维，三种跨维热启手术 | — | 工序 / 内部 / **无** |
| `export_onnx_*.sh`（5 个） | 各观测合同的导出封装。**铁律：必须烤进和训练同一批片子** | — | 4 工序 / `_p4` **无** |
| `harvest_onnx_motion.py` | 从 ONNX 取回烤入的动作缓冲 | — | 内部 |

## 根 `scripts/`（约 100 个，绝大多数是一次性波运行器）

命名：`run_*` 跑某波 · `validate_*` 该波源码门 · `materialize_*` 物化工件 ·
`attest_*` 铸收据 · `audit_*` 只读审计 · `screen_*`/`select_*` 候选筛选。
一页对一个 `operations/run_*.md`，从[实验索引](../experiments/README.md)进。可复用的少数：

`feasibility_oracle.py`（见上）· `ground_gmr_pkl.py`（对 canonical 碰撞模型落地，工序）·
`validate_motion_role_catalog.py`（动作角色目录校验，工序）·
`validate_phase1_queue_governance.py`（队列治理 fail-closed，工序）·
`v2_weight_calibration.py` + `v2_probe_extract.py`（v2 定权，工序）·
`branch_dashboard.py`（分支追踪看板）· `classify_sparse_reward_milestones.py`（有 ops 页但实验 blocked）·
`planner_replay_eval.py`（**无**）。

## 无人调用清单

**A 类：有工序价值，应排期决定"进工序还是明确降级"**

`motion_dynamic_replay.py`（唯一真 `mj_step` 证据）·
`gen_stage1_questions.py --phase-scan`（唯一能查出"这条片一帧都回不了球"）·
`check_perclip_pos_sampling.py` / `check_perclip_vel_sampling.py`（唯一不开 Isaac 就能查出目标框在桌下）·
`optimize_reachable_face.py` · `rewrite_followthrough.py` · `topp_budget_search.py` ·
`canonical_grounded_ready.py` · `canonical_frame_identity.py` · `canonical_neutral_ready.py`+`_cli.py` ·
`racket_fk_ref.py` · `verify_hitter_pure.py` · `scripts/planner_replay_eval.py`

前四条已在[应当变成闸门的规则](rules_that_should_be_gates.md)里给出确切检查与落点。

**B 类：历史/一次性，无人调用属正常**

`arm_a_eval_sweep.sh` · `arm_a_mujoco_ab.sh` · `holdfix2_eval_sweep.sh` · `export_onnx_p4.sh` ·
`launch_footwork_ft_p1.sh` · `upload_npz.py` · `warm_start_realsensor.py` · `pp_recovery_eval.sh` ·
`play.py`(rsl_rl 旧版) · `cli_args.py`

## 维护

新增脚本 ⇒ 同一分支在本页加一行。工具进出工序 ⇒ 改「调用」列。
A 类清单每次大审计复扫，要么进工序要么写明降级理由后移 B 类。
