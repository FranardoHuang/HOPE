# 术语与人话对照

- **signed-face A2/B2**：同一个已审计父 checkpoint 的跨 Pod 热启动探索 L1；A2 是 guidance `0.0`
  对照，B2 只把 signed-face guidance 改为 `-0.4`，两条 child 都是 lineage-inexact，不能当 fresh 证据。
- **Wave-B B0/B1/B2**：下肢稳定实验内部的三格 matched-parent screen；B0 同时把两种新 reward
  设为 `0` 但保留双探针，B1 只开静态 `v4rg` 十二腿关节软模仿 `+0.5`，B2 只开不读取动作参考的
  脚距下界与实际腿速 bundle `-0.25`。它与上面的 signed-face `B2` 无关，使用时必须带
  `Wave-B` 前缀。

本文件是现行术语真源。新人和 agent 不需要去历史归档猜缩写。

## 先遵守这条：不用黑话

- `run_name`、`flag`、实验臂代号和缩写第一次出现时，必须同行写出：**它是什么、改了什么、用来回答什么**。
- 报告不许只写 `M3`、`R9c`、`SZ`这类裸代号；可以保留代号便于查路径，但后面必须跟一句人话。
- 新缩写先加到本表，再在 `NOW`、实验记录或 `Gate` 文档中使用。
- 代码里已存在的参数名不强行翻译，但文档必须解释开关开/关各会发生什么。

## 当前训练与判卷术语

| 术语 | 人话 |
| --- | --- |
| `setting` | 一整套可复现配方：动作、观测、reward、题库、plant、训练方法和裁决尺必须一起指定。只换一项就是新 setting。 |
| <a id="upper-n3-safe"></a>`upper N3 safe warm-start` / 三反手上肢安全热启动（历史） | 从四动作 175 维 VirtualBall 父 checkpoint 非 exact 地热启动同形 actor/optimizer，只保留有题库证据的 `bh_loop_c / bh_block / s0_highpress` 三个反手动作；静态题库、球/动作语义和 actor 维度不变。新叶子额外启用 physical-hard 关节终止、pre-physics hard-crossing latch、soft q_des clamp/qbar/joint-limit shaping，并让所有 termination 只收一次统一死亡罚。该实验当时曾独占 Pod1 physical GPU1；现已退出活跃发射入口，不再持卡、续跑或授权长训/真机。 |
| <a id="action-conditioned-ball-first"></a>`action-conditioned ball-first` / 按动作条件化的球优先训练 | 训练日程先冻结动作，再从该动作自己的到球时间/来球/base/落点域采样球，由固定动作方向的 solver 反解 task 和认证 teacher rate，最后执行 policy。训练期不运行 selector；它学习的是逐动作的 `P(安全合法回球 | 球,落点,base)`。完整合同见 [`action_conditioned_ball_first_contract.md`](interfaces/action_conditioned_ball_first_contract.md)。 |
| <a id="action-ball-manifest"></a>`action-ball manifest` / 动作来球启动清单 | 以 exact 文件 SHA-256 绑定任意 N 动作顺序、stable UID、motion/prototype/physics bytes、逐动作到球时间/来球/base/aim 域、参考 `t_hit/t_cycle/site speed`、teacher-rate 认证范围、课程和留出 split 的 metadata。manifest 不含自报授权；正式 launch 还必须另持 code-rooted opaque motion admission。 |
| <a id="action-ball-n-contract"></a>`action_ball_n<N>` / N 动作球优先 actor 合同（历史 181+N） | `N` 由 exact action-ball manifest 的动作数决定；actor 在共同 hitter/footwork 与拍面项之后追加 N 维 one-hot 动作身份。它只保留旧 checkpoint/receipt 解析，当前 trainer 不再实例化。fresh N1 使用下述 fixed-194 v2；formal N5/N73 等固定宽 continuous future-motion intent 后再切新合同。不同 N、不同合同名或不同列语义都不能只换 motion 列表续跑。 |
| <a id="action-ball-table-pose-n-contract"></a>`action_ball_table_pose_n<N>` / 带桌面相对底座位姿的 N 动作球优先 actor 合同（历史 191-D 候选） | `hitter_footwork(177) + base_position_table(3) + base_orientation_table_6d(6) + signed-face/rho(4) + action_one_hot(N)`，总宽 `190+N`，所以 N1 为 191。task 的 base/racket 位置仍是 robot-relative residual；新增 3+6 只表达 A3 相对桌面中心的完整 6-DoF 站位。历史注册器记录的是旧 `mocap_pose_history` producer 语义，尚未形成 OptiTrack+gyro 的部署合同。该合同已由下述 194-D successor 取代；arbitrary-N/191-D C++ consumer 也未闭合，不能接真机。 |
| <a id="action-ball-table-pose-twist-contract"></a>`action_ball_table_pose_twist_n<N>` / 带桌面位姿与底座线速度的 N 动作球优先 actor 合同（历史混合 frame 候选） | 总宽 `193+N`，N1 为 194；但历史前缀把 racket position residual 放在 yaw-heading frame，而 racket velocity 与 signed face normal 留在 world/table frame。它只保留为旧 checkpoint/诊断兼容，不能作为 fresh 训练或部署合同。 |
| <a id="action-ball-table-pose-twist-heading-task-contract"></a>`action_ball_table_pose_twist_heading_task_n<N>` / frame 一致的桌面位姿、底座速度与相对击球任务合同（兼容版） | 宽度 `193+N`，N1 为 194；target-minus-current-racket position、demanded racket velocity 与 raw-A signed face normal 全部统一到 base yaw-heading frame。2026-07-30 三条 N1 long 使用它并保留 exact 身份；fresh launch 改用下一行显式 teacher-start successor。 |
| <a id="action-ball-teacher-start-contract"></a>`action_ball_table_pose_twist_heading_task_teacher_start_v2` / 无动作槽位、显式老师启动倒计时的 ActionBall actor 合同 | 当前 fresh N1 合同，固定 **194-D**：前 177 不是“本体状态”，而是 `reference/teacher 68 + robot/runtime 99 + task/clock 10`；后 17 是 `table pose/twist 12 + demanded face/rho 4 + time_to_teacher_start_s 1`。`base_ang_vel(3)` 是 pelvis 三轴角速度，`base_lin_vel_heading(3)` 是独立的 root-COM 线速度；合起来才是 6 个 twist 标量，`base_orientation_table_6d(6)` 则是姿态编码。frame 一致合同的非身份列后增加 `time_to_teacher_start_s(1)`，并删除 `action_one_hot`；动作 UID/slot 仍只在 sampler/solver/curriculum/receipt 控制面冻结，绝不作为 policy 输入。该 v2 只授权 N1；formal N5/N73 必须先增加固定宽、由 reference 内容生成的连续 future-motion intent，再切新合同。绝对 9 值只回答 A3 相对桌体位姿，task 继续是 robot-relative residual；部署 pose 由 OptiTrack 锚定，角速度由 pelvis IMU gyro 提供。旧同宽 194-D/195-D checkpoint 都不能按宽度猜测或 exact resume；C++/MuJoCo producer、实测噪声延迟和外参 parity 未闭合前不得接真机。历史 `...teacher_start_n<N>`（`194+N`）仅保留旧 receipt/schema 解析，不再用于 fresh ActionBall 发射。 |
| <a id="continuous-rotation-6d"></a>`base_orientation_table_6d` / 连续 6D 桌面相对旋转 | 把 `R_table_base` 前两列按 `[R00,R01,R10,R11,R20,R21]` 喂给 actor；正交化后两轴唯一恢复第三轴，因此保留 roll/pitch/yaw 全部三自由度，并非 yaw-only 或“两角”。采用 Zhou et al. [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html) 的连续 6D 思路，避免 Euler wrap/gimbal 与 quaternion `q/-q` 符号不连续；本仓库列/行顺序以本合同为准。 |
| <a id="action-ball-asymmetric-support-uniform"></a>`asymmetric support-width uniform` / 不对称题域均匀采样 | 每个连续标量或 tangent 角只在 `[center-width_lower, center+width_upper]` 上作一次均匀采样，不先 50/50 选侧，也不是 half-normal。两侧 width 由各自课程 level 独立扩张，所以落在某侧的概率随该侧物理宽度变化；实现遗留的 `*_std` 字段名仍表示 support half-width，不是高斯标准差。 |
| <a id="action-ball-pressure-gate"></a>`N-dependent ActionBall pressure gate` / 随实际动作数缩放的运行压力门 | 在正式发射前，按本次 exact manifest 的动作数 `N`、环境数 `E` 和预注册轮数 `R`，实测分批增长、compaction、保存/加载、exact roundtrip、篡改负例、峰值内存与时延。N5 只需自己的 `5×E×R` 凭据；N93 压力门只约束真正的 N93，不得反向阻塞 N5。 |
| <a id="action-ball-target-mode"></a>`racket.target_mode=action_ball` / 动作条件球优先模式 | 只允许 action-ball sampler 产生来球，再由 fixed-action solver 解 task；训练 selector、旧 question/CQ bank、HER 和第二 task producer 都必须关闭。 |
| <a id="action-ball-manifest-flags"></a>`racket.action_ball_manifest_path` 与 `racket.action_ball_manifest_sha256` | 启动者预先给出的清单路径与 exact 文件字节 SHA-256。运行时必须重读并核对，不能在同一次 launch 中现算现填期望值。 |
| <a id="action-ball-policy-contract-flag"></a>`racket.action_ball_policy_contract_sha256` | 绑定观测、动作、runner、Reward 与 evaluator recipe 的不可变 policy 合同，不是每次 PPO 更新都会变化的 checkpoint SHA。 |
| <a id="action-ball-fixed-direction-flag"></a>`racket.action_ball_fixed_direction=true` | 每颗球只能在已经冻结的动作方向/拍面上求解；solver 不得换动作、重采样或用自由方向找到更容易的答案。 |
| <a id="action-ball-solver-knobs"></a>`cq_overdraw / cq_n_iters / cq_tol_m / cq_speed_budget / cq_max_redraw_rounds` | 沿用历史前缀的五个 fixed-action solver knob，分别控制外部独立 proposal 富余量、求解迭代数、落点容差、全局拍速上限和最多外部 proposal 轮数。它们不重新启用旧 CQ producer；所有 proposal 都进分母与 reject ledger，实际值进入 solver payload SHA。 |
| <a id="action-ball-native-first-episode"></a>`init_at_random_ep_len=false` / action-ball 完整首 episode | action-ball 首次 learn 不随机预填 episode length，确保首批 attempt 从完整 true reset/native cycle 开始。其他模式可保留历史默认；该差异进入 policy/preflight recipe。 |
| <a id="n1-milestone1000"></a>`milestone1000` / N1 千轮诊断里程碑 | N1 diagnostic launcher 的 exact `4096 env × 1001 iterations × save interval 100` 预算；RSL-RL 从 0 编号，所以自然产出 `model_1000.pt`。它用来跨过约千轮后再判断模仿、击球机会与安全趋势，仍是 diagnostic，不是 formal curriculum promotion、第二 seed 或真机授权。 |
| <a id="n1-diagnostic-long"></a>`long` / N1 有限诊断长跑 | N1 diagnostic launcher 的 exact `4096 env × 20001 iterations × save interval 100` reviewed 预算；自然产出 `model_20000.pt`。它替代不可达的超大 iteration 哨兵，让“一路跑”仍有可核对终点；diagnostic 身份不因预算变长而成为 formal、课程晋级或真机授权。 |
| <a id="stable-ready-plant"></a>`stable-ready plant` / 首个 N1 的稳定准备 plant | 由旧 `4096×5` full-DR probe 的 shared `waist_roll` raw-hard 反例触发：N1 diagnostic 保留历史 robot-material 随机化与 policy recipe 已钉住的 joint-default `±0.01 rad`，暂时关闭 torso CoM、link-mass 与 PD-gain DR，让 policy 先在可保持的 A3 plant 上学习。它不是最终 sim-to-real 配方；1000 update 后各 DR 轴须带 robust-hold/teacher-to-`t_hit` 证据逐级恢复。 |
| <a id="a3-vendor-v1-profile"></a>`HOPEPingPongActionBallA3VendorV1` / 智元 A3 新一代 ActionBall 任务 profile | fresh vendor N1 的不可变 task leaf：拥有 `[0,2]` 控制步延迟、`1–3 s` 六轴 velocity-only `axis_box_6d_v2` push 与粗细位置核。今晚 nominal 已冻结为非冲突 parkour 新表 + 三处 task/SKU fallback：waist-yaw Kp=`85`、waist-pitch effort=`118`、wrist-pitch/yaw=`Kp20 / effort6 / armature0.0008100893338`，wrist-roll=`Kp30 / effort24 / armature0.004968`；Pod 仍须对拍。profile 在原 `weight=4/std=0.075 m` 精位置核外独立开启 `weight=1/std=0.30 m` 粗位置核，两者使用同一 swing-through 目标和紧击球窗。旧 checkpoint/bundle/receipt 不得续成；选中 profile 也不等于 formal launch 授权。 |
| <a id="n1-vendor-baseline-diagnostic"></a>`N1 vendor baseline diagnostic` / 智元基线 N1 单卡诊断 | `launch_n1_vendor_baseline_diagnostic.py` 的窄发射身份，当前三 lane 固定为 A=`bh_loop_c` static、B=`bh_block` static、C=`bh_loop_c` monotonic adaptive-sigma，全部 [fresh-only/no-resume](#fresh-only-no-resume)。仅有三个 live stage：`smoke=1×2×save1`、综合 `probe=4096×5×save1` 与 `long=4096×20001×save100`。综合 probe 在同一次 run 内闭合 core safety/runtime、Reward/PPO economy 与智元 `1–3 s` 六轴 velocity-only push 证据；它的单份 v3 receipt 是 long 唯一前置收据。独立 `push_evidence=4096×32` 已退役，旧 spec/receipt 仅作 spent history。每条 run 独占一张空物理 GPU 并使用 fresh no-clobber namespace；它不接受任意 Hydra override，也不产生 formal evaluator、promotion、resume、export 或 judge authority。 |
| <a id="n1-vendor-probe-gate-receipt-v2"></a>`n1_vendor_probe_gate_receipt_v2` / 历史 N1 综合 probe 收据 | r5 scalar-std epoch 的 spent history。它没有 r6 `log_std`、fixed-domain 与 Reward/PPO economy 闭环，不得授权当前 long；保留此名字只为解释旧记录，不是可选发射格式。 |
| <a id="n1-vendor-probe-gate-receipt-v3"></a>`n1_vendor_probe_gate_receipt_v3` / 当前 N1 综合 probe 放行收据 | 只允许 `stages.probe`，不接受历史 `push_evidence` stage。在 Pod 真实 runner 的同一次 `4096×5` 中消费 [fixed-domain](#n1-fixed-domain-initial-receipt-v1) 与 [Reward/PPO economy](#action-ball-reward-ppo-economy-receipt-v1) 收据，硬门 source/plant、194-D actor + 318-D critic normalizer save→第二 runner load、checkpoint finite、`log_std`/LR、delay、Reward closure/advantage/PPO/gradient economy、joint actual-hard、qdes、nonfinite 与 natural completion，并要求 velocity push `event_call_count>0`、`env_application_count>0` 及六轴 extrema finite/in-range。table/fall 频率、strike-window 分布与 recovery 仍在 long 前 100 update 持续记 telemetry，不因 5-update 数值单独拒绝。 |
| <a id="vendor-r5-r6-artifact-epoch"></a>`vendor r5/r6 artifact epoch` / 智元 N1 工件纪元 | r5 是已闭合的历史 scalar-std 工件链；r6 因 policy ABI 改为 `noise_std_type=log` 而必须从 identity/runtime-contract/authority/candidate/hold/bundle 到 probe receipt 全链 fresh 重物化。这是 artifact identity epoch，不是算法版本号；任一 r5 SHA/checkpoint 不得回填 r6。 |
| <a id="noise-std-type-log"></a>`algo.policy.noise_std_type=log` / 原生 log 探索标准差 | 只在 vendor r6 exact launcher 显式开启；共享 `ppo.yaml` 仍以 `scalar` 作旧谱系默认。rsl_rl 存一个 31-D `log_std` 参数，运行标准差为 `exp(log_std)`，配置与实现初始 sigma 均精确为 `0.02`；bootstrap schema-3 和 checkpoint ABI 都必须同时证明 `parameter_name=log_std`与形状 `[31]`。 |
| <a id="fresh-only-no-resume"></a>`fresh-only/no-resume` / 仅全新训练、禁止任何续跑 | 不是“formal 还没证明但技术上可续”：当前 r6 log-std bootstrap 对任一 `checkpoint_path` 直接 fail-closed，包括 scalar 旧 checkpoint 和 r6 自身 checkpoint。中断只能换 fresh no-clobber namespace 从头训；旧 schema-1/2 scalar 只保留解析兼容，不会授权 r6 resume。 |
| <a id="n1-fixed-domain-initial-receipt-v1"></a>`n1_fixed_domain_initial_receipt_v1` / N1 固定初始域收据 | 每 action 一份，从 production sampler/profile/curriculum 物化 level-0、`no_move`、32 arms 的 width/cap/mask 和 `1:3:1` cell mixture；C 复用 loop 收据。它硬写 `domain_epoch=0`、`curriculum_promotion=false`、`diagnostic_unauthorized=true`，不接受 operator 手填域宽。 |
| <a id="action-ball-reward-ppo-economy-receipt-v1"></a>`action_ball_reward_ppo_economy_receipt_v1` / ActionBall Reward/PPO 经济收据 | 三 lane 共享的静态证据：绑两份 r6 runtime contract、Reward scalar=`1`、`dt=.02`、ordered nonzero terms、rsl-rl 2.3.1 三源、whole-rollout advantage normalization 与 `log_std` realized sigma=`.02`。它的 `authorization.training/resume=false`，不能单独代签 long；只有 probe v3 消费它并把 live economy telemetry 验成 PASS 后才可放行。 |
| <a id="control-step-action-delay"></a>`control-step action delay` / 控制步执行器命令延迟 | 在 actor 输出进入 `JointPositionAction` 仿射 `q_des=action*scale+offset` **之前**对整行 31-D action 做延迟；单位是 50 Hz policy/control step，不是 5 ms physics substep。每个 environment 在 true episode reset 时从闭区间 `[0,2]` 离散均匀抽一个 scalar lag，episode 内固定且五个 actuator group 共享；reset 队列先填 safe default，dynamic-ready 在同一事务改填动作专属 hold。它是训练 DR，不进 ONNX decoder。 |
| <a id="axis-box-6d-v2"></a>`axis_box_6d_v2` / 六轴速度增量推撞 | schema v2 的对称六轴 root-velocity delta：`x/y=±0.25 m/s`、`z=±0.10 m/s`、`roll/pitch=±0.26 rad/s`、`yaw=±0.39 rad/s`，ActionBall vendor profile 每 `1–3 s` 触发一次。当前合同是 velocity-only，显式 `force_push=false`、`combined_exclusive=false`，不安装旧 `68 N×0.3 s` force push，因而不存在同帧力推叠加。当前版本没有 strike/recovery phase gate，因此可在击球窗发生；不得冒充已实现门控。 |
| <a id="action-acc-jerk-probe"></a>`action_acc_jerk_probe` / 动作 jerk 零收入观测开关 | DeployParity/HitterPure 的显式诊断开关：启用后在 RewardManager 时序计算 raw action 二阶差分平方和，同时记 raw 值与 `36.0` 封顶值。manager weight 为 `1` 只是防止被零权重剪掉，函数返回严格零，不改 Reward。缺席或 `false` 时 cfg 槽位为 `None`，不构造 term、不增加 vendor N1 每步开销。 |
| <a id="implicit-pd-post-step-effort-proxy-probe"></a>`implicit_pd_post_step_effort_proxy_probe` / 隐式 PD 步末解析力矩需求代理 | 显式诊断开关：只对能证明全是 implicit actuator 的 action joints，用 live DR 后 `Kp/Kd`、最终下发 `q_des` 和 policy step 结束时 `q/qdot` 计算 `Kp*(q_des-q)-Kd*qdot`，报告超过 `0.9×limit` 和 `1.0×limit` 的比例账。它是 analytic/post-step/non-actual/substep-blind proxy：PhysX implicit drive 不提供实际力矩，本值也看不到四个 physics substep 内的峰值；不得称为实际力矩、实际饱和或子步峰值。缺席/关闭时不构造 term。 |
| <a id="strike-window-entry-distance"></a>`strike-window entry racket distance` / 入击球窗拍距 | 每拍只在第一个 strike-window tick 记一次 `||racket_FK-racket_target||`；八个互斥 finite bin 由 `0.075/0.15/0.20/0.30/0.50/0.70/1.00 m` 七个边界切分，另记 nonfinite、总数与 finite sum。它只是诊断 ledger，不改 Reward/observation/reset/curriculum；当前 vendor N1 在 long 前 100 update 持续看该分布，不用 5-update 的拍距比例单独拒绝 integrated-probe receipt。若长期仍多数 `>0.20 m`，它是停车定位粗+细核的信号，不是安全证书。 |
| <a id="vendor-runtime-json-markers"></a>`HOPE_ACTION_BALL_POLICY_BOOTSTRAP_JSON` / `HOPE_RSL_RL_RUNTIME_ABI_JSON` / `HOPE_POLICY_STD_UPDATE_JSON` / `HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON` | 分别钉住 fresh policy bootstrap 的 `log_std/[31]/realized .02`、RSL-RL/normalizer ABI、每次 optimizer update 的 realized policy std/LR，以及首个 rollout 前的 delay initialized-env 与 lag histogram。任一状态非 finite、std `<=0`、normalizer 缺失/形状错误或 delay 分母不守恒都 fail-loud；这些是运行收据，不是学习成功证据。 |
| <a id="a3-vendor-identity-smoke"></a>`A3 vendor identity smoke` / 智元身份物化冒烟 | 一次性的 `launch_a3_vendor_identity_smoke.py` 两段工序：固定 vendor task、`bh_loop_c` upper、seed0，先 recipe-only 物化 shared-ready policy recipe，再用 `1 env × 2 update` 产出首份 schema-3 runtime training contract。它不消费 bundle/dynamic-ready，只用于打破新 plant 首份合同的自举循环；不能授权 long、formal、promotion、export、judge 或硬件。 |
| <a id="a3-vendor-eval-profiles"></a>`vendor_play_v1` / `deterministic_ranking_v1` | 智元 task 的两种不可混报评测口径：vendor Play 关闭 startup plant DR 与 interval push，但保留 policy 观测噪声和每 episode actuator delay；deterministic ranking 在此基础上再关闭观测噪声、delay 和 reset-state 噪声，用于仓内 checkpoint 确定性排名。两者都保留明确 receipt，不能把 deterministic 分数冒充 vendor Play 鲁棒性。 |
| <a id="h-mech"></a>`Hmech` / `H_mech` / 机械硬位置边界 | 资产/URDF/MJCF 定义的关节真实最小与最大角度；actual-hard ledger 、终止与安全收据始终以它为真值，不得被控制保护边界改名或覆盖。 |
| <a id="h-ctrl"></a>`Hctrl` / `H_ctrl` / PhysX 控制位置保护边界 | 在 live PhysX solver 中比 `Hmech` 更靠内的最后一道位置 constraint。当前 vendor N1 候选只对两腰与左右 ankle-roll 每侧内缩硬 span 的 `2%`，其余 27 轴 `Hctrl=Hmech`；actor q-des、soft envelope、Reward 和 `Hmech` ledger 不变。 |
| <a id="obb"></a>`OBB` / oriented bounding box / 定向包围盒 | 中心和三条随物体旋转的半轴共同定义的碰撞代理盒。它比把旋转物体外包成 world-AABB 更紧，但仍是 proxy 几何，不等于 PhysX 已解析接触力。 |
| <a id="sat-collision-test"></a>`SAT` / separating-axis test / 分离轴定理碰撞检测 | 对 OBB-vs-AABB 的 15 条候选轴逐轴投影；只要有一条轴上区间分离就判无重叠。ActionBall 当前只把它作 diagnostic counterfactual，旧保守 terminal mask 在数据定谳前不变。 |
| <a id="action-ball-table-safety"></a>`ActionBall table safety assembly` / 动作球优先桌体安全总成 | 解析球 ActionBall 专用的五件碰撞总成：真实桌面、从地面到桌板底面的保守 robot keep-out、球网和左右网柱。keep-out 是防机器人穿进桌下的安全代理，不冒充真实桌腿；因此它与 physical/shadow 动力学球互斥。`--contact-smoke` 是一次性 Pod 检查开关：用真实机器人刚体逐子步撞五件 collider，核对四个 5 ms 子步 latch、原始终止原因、filter 列和 reset 后不泄漏；它不是训练。 |
| <a id="balanced-action-sampling-flags"></a>`motion.balanced_clip_sampling` 与 `motion.balanced_clip_sampling_seed` | 开启任意前缀动作数最多差一的均衡日程，并用内容绑定整数 seed 固定其顺序；这不是 planner selector。 |
| <a id="mobility-mode"></a>`mobility_mode=no_move/move` / 不移动与移动版本 | 两个独立训练身份。`no_move` 仍可在一个 spawn 分布中出生，但每个 swing 的 base goal 必须等于本 episode 的实际 spawn；`move` 才允许逐级扩大相对 spawn 的目标位移。二者不能在 runtime 临时互相覆盖或继承课程证据。 |
| <a id="safe-policy-failure"></a>`safe-policy failure` / 安全闭环后的策略失败 | 已经由 solver 接收、安装、开始并正常关闭，且未撞桌、未摔倒、未发生其他 unsafe 的 attempt 中，没有合法回球的比例。课程的 10%/20% 只指这个分母；solver 无解、基础设施 invalid 和 unsafe 分开记。 |
| <a id="action-ball-arm-catalog"></a>`action-ball arm catalog` / 来球课程方向表 | schema v3 把每个动作的到球时间、触球位置、来球速度/方向、旋转大小/方向、落点、base 出生与移动分别拆成 lower/upper 或 tangent 正负方向，共 32 个有序 arm；`no_move` 禁四个 base-travel arm 后有效 28 个。exact 顺序与 catalog SHA 必须被 sampler、课程、评估器、runtime 和 checkpoint 共同绑定。这里的 arm 是“课程扩张方向”，不是实验臂或机器人手臂。 |
| <a id="action-ball-compaction-segment"></a>`action-ball compaction segment` / 退休历史压缩段 | 只在 rollout/reset barrier 上，把已经关闭且不会再成为 active/pending 的连续 birth/sample/task 前缀折成逐动作 high-water、守恒计数和 append-only hash-chain head；活跃后缀仍保留完整 receipt。sampler、broker、pool、provider 和 Racket 必须在一次原子事务里对同一个 segment head 达成一致，不能各自删历史或只压 JSON。 |
| <a id="action-ball-resume-receipt"></a>`action-ball formal resume receipt` / 正式续训外部凭据 | 在本次 load 之前已经由独立来源钉住的内容寻址凭据，至少绑定 raw checkpoint SHA-256、shared action-ball state root、compaction segment head、run/training contract 与 iteration。内部 hash 只能防局部损坏；缺这份外部 pin 时只允许 fresh launch 或 diagnostic load，不能声称 formal exact resume。 |
| <a id="time-to-contact"></a>`time_to_contact` / 到球时间 | 从本拍 ready/reset 参考点到球应触拍的可用时间。每动作从中心向更早/更晚两侧独立扩张；Motion 先等待再按认证 teacher rate 播放，必须满足 `wait + scaled_t_hit = time_to_contact`，额外等待上限为 1 秒。 |
| <a id="teacher-rate"></a>`teacher_rate` / 老师动作时间倍率 | fixed-action solver 所需拍速除以该动作正式 physical racket-site 参考拍速；它同时缩放 `t_hit` 与 `t_cycle`。倍率必须位于动作已认证范围内，runtime 不得 clip；越界样本进入 solver reject。 |
| <a id="rolling100-arm-scheduler"></a>`rolling-100 arm scheduler` / 最近百次方向调度器 | 每动作、每候选方向只看最近 100 个同 cell 的安全闭环结果，优先安排成功率受影响较小的方向并强制防饥饿探索。它只决定下一步测谁，不授权扩域；正式晋级仍要 frozen canary 与独立 heldout。 |
| <a id="joint-domain-rho"></a>`joint-domain rho` / 联合域缩放量 | 各采样轴先各自找 marginal frontier，再用共同 `rho` 同比缩放这些 frontier，并在联合分布上把总 safe-policy failure 调到目标带。它避免把多个“单轴 10%”直接相乘成过难联合域。 |
| <a id="task-first"></a>`task-first` / 任务优先 executor 训练 | 已拒绝为当前主训练入口、仅保留为消融：先直接采球拍/base task，训练时不读来球。它不能天然保证球与 task 物理匹配；当前主线改用 `action-conditioned ball-first`。 |
| <a id="task-first-manifest"></a>`task-first manifest` / 任务优先启动清单 | 历史 task-first 候选的 metadata 合同；只用于复现实验/消融，不是当前运行 manifest，也不得自报训练授权。 |
| <a id="task-first-curriculum"></a>逐动作 task curriculum / 任务泛化课程 | 历史 task-first 的直接 task 扩域控制器；当前主线由逐动作 ball/base/aim marginal + joint-`rho` 课程取代。 |
| <a id="station-center-shift"></a>`station_center_shift_xy_m` / 整动作站位中心平移 | 同时平移 reference action、球拍 task center 和 base center 的 XY 常量，不是只改 base Reward。HOPE 中 `+X` 朝球台/对手，所以负 X 表示整个人和整段任务远离球台；后续 base curriculum 只学相对这个中心的 residual。 |
| <a id="wilson-confidence-bounds"></a>Wilson confidence bounds / Wilson 置信界 | 对有限 Bernoulli 样本给成功率区间的保守量尺。task curriculum 用成功率下置信界和 unsafe 率上置信界，加最少样本、dwell 与滞回决定晋级/回退；不能直接拿样本均值或跨动作平均。 |
| <a id="stable-action-uid"></a>stable action UID / 稳定动作唯一号 | 由规范化 `action_id + family + motion/content SHA-256` 派生的正整数身份，范围 `1..2^53-1`，可精确穿过 JSON/double/C++；换 bytes 或 family 必须换号。dense action slot 只是某个 catalog 的本地 `0..N-1` 数组索引，重排可变；UID `0` 保留给 selector abstain。 |
| <a id="capability-artifact"></a>`capability artifact` / 动作能力工件 | 用冻结 ball-conditioned 留出卷，对一个 exact catalog/policy/task/effective-Reward/model/calibration tuple 记录逐动作 support、OOD、校准成功率下界与 unsafe/误差的内容寻址工件。训练内 curriculum 成功率不能直接冒充它。 |
| <a id="ood"></a>`OOD` / out of distribution / 分布外 | 当前 query/candidate 超出能力工件有足够支持的数据域。selector 必须先用预注册 score/threshold 拒绝 OOD，不能靠高 priority 或点估计成功率放行。 |
| <a id="selector-lcb"></a>`LCB` / lower confidence bound / 成功率下置信界 | selector 对“该 exact 动作在该 task 邻域成功”的保守校准下界。先过最低 LCB floor，再按最高 LCB 排序；priority 只在与最高值相差不超过 `delta_tie` 的集合内生效。 |
| <a id="selector-abstain"></a>`abstain` / selector 明确不出手 | 没有动作同时通过硬安全、support/OOD 和最低 LCB 时返回唯一空身份 `(action_uid=0, action_id="", slot=-1)`；不得偷偷回退到默认正手/反手。 |
| <a id="effective-reward-recipe"></a>`effective Reward recipe` / 实际生效奖励配方 | 从已经 compose 完、所有 override 都落地的环境配置读取 active Reward callable、weight 和 params 后做 canonical JSON/SHA-256；它回答“Isaac 真吃到什么”，不是 `reward_pack` 标签或设计表。checkpoint、launch 与 A/B 必须绑定这个 receipt。 |
| <a id="reward-group-mjlab-balance"></a>`mjlab_balance_stability` / MJLab 平衡与稳定组 | ActionBall effective Reward 四分组之一：收纳直立、落脚/滑脚、动作平滑、底座速度、恢复稳定等平衡目标及其负向正则。名字表示设计来源/机制归类，不表示所有 term 都逐字来自 MJLab。它是可预注册科学调节组；必须逐动作报告 eligible、nonzero、raw/weighted sum 和正负贡献，不能只看总 Reward。 |
| <a id="reward-group-beyondmimic"></a>`beyondmimic_imitation` / BeyondMimic 模仿组 | 收纳 global anchor 与 body position/orientation/linear/angular velocity 等老师动作跟踪，以及明确登记的派生下肢模仿。它回答 policy 是否仍像 exact teacher；必须与击球任务组分账，避免高上台分掩盖动作已经学歪。 |
| <a id="reward-group-hope-task"></a>`hope_hit_landing_task` / HOPE 击球与落台任务组 | 收纳球拍位置、拍速、有符号拍面、击球捕获、过网、落点、旋转及相关 guidance。它回答动作有没有完成本项目自定义的击球/上台任务；teacher imitation 好不能替代本组的物理回球证据。 |
| <a id="reward-group-immutable-safety"></a>`immutable_safety` / 不可拿任务分抵消的安全组 | 收纳死亡/桌碰、实际与 q_des 软限位、速度限位和 tracking envelope 等安全项。它在四分组审计中必须显示实际激活，但不是可用上台收益交换的科学比例旋钮；硬终止、table/fall/joint raw ledger 与 motion admission 仍独立 fail closed。四组必须从 composed active recipe 精确分割全部 term，未知 term 不得落进模糊 `other` 桶，零权 probe 不得冒充 objective。 |
| <a id="action-ball-raw-terminal-signals"></a>`V4 raw terminal_signals` / 可重叠安全终端信号 | V4 evaluator 对每个 attempt 保留 `table/fall/collision/joint_qdes/joint_actual` 等 sticky 原始布尔量；它们可以在同一 closure 同时为真。unique unsafe=`C-L-F`，必须满足 `max(U_i) <= unique unsafe <= sum(U_i)`。curriculum state schema 10 把 scheduler 的 raw signals 原样保留并在 exact resume 重放，不能只存 precedence 后的单标签。 |
| <a id="mujoco-action-ball-policy-fitted-gate"></a>`MuJoCo ActionBall policy fitted gate` / 学成策略对应球物理回台门 | teacher fitted-ball PASS 只证明 exact 动作击球帧与对应来球/solver task 在物理上可行；本门另加载 exact frozen checkpoint→ONNX、normalizer、PD plant 和同一 sealed case capsule，让**学成 policy**在 MuJoCo 中执行，并用真实桌/网、两档 `dt`、swept 有符号拍面接触和首落点裁判逐动作检查 center/support 来球、桌碰/跌倒/限位与回台误差。它不是 virtual landing、BankExam 解析外推或 teacher replay。截至 2026-07-29 只有 source/测试占位，没有 fresh N5 formal PASS receipt，状态明确为**未通过**，不能授权部署或真机。 |
| `arm` / 实验臂 | 一条具体训练或对照配置，不是机器人的手臂。每条臂必须说清与对照相比只改了什么。 |
| `Wave-B` / 下肢稳定首轮 | W/V 两个旧 `model_6700` parent 内分别比较 B0/B1/B2 的六格 causal continuation；两种机制共用不看成功结果的挥拍前 `0.30 s` 与同 attempt 挥拍后 `0.40 s` inclusive gate。它不是课程 Stage，也不授权当前 0/4 stance 验收失败的 M0 横移老师。 |
| `run` | 某条实验臂的一次实际执行。同一实验可以有多个 run。 |
| <a id="launch-authorized"></a>`launch_authorized` / 科学训练发射闩 | 轻量训练 YAML 顶层的显式布尔门。`false` 时 `launch-next` 与 `fill` 在 SSH、claim 或 Kit 启动前拒绝；`plan/status/doctor` 和独立的 source-asset/probe/finalizer 前置门仍可运行。只有 terminal probe 证据审完并把目标行改为 `ready` 后才可设为 `true`；它不授权 judge、第二 seed、晋级或真机。 |
| <a id="dispatch-pods"></a>`dispatch_pods` / 活跃 Pod 集合 | YAML 队列里唯一允许 live snapshot、claim 检查和接收**新** trainer 的 Pod 名单。未列入的 Pod 不被普通 status/doctor/fill/launch/attest 访问；交接前必须先把其历史行改成终态，并给新任务使用从未发射的 namespace。它表达当前资源归属，不改变每卡容量上限。 |
| <a id="preferred-slot"></a>`preferred_slot` / 优先槽 | 某条 YAML job 希望优先落到的 `pod/gpu`，用于同卡配对或复用该 source 的 cold-boot receipt；只在该槽仍低于容量时生效，满载后仍回到全局 round-robin，不会越过 `dispatch_pods` 或容量上限。 |
| <a id="required-slot"></a>`required_slot` / 硬绑定槽 | 某条 YAML job 只能落到的唯一 `pod/gpu`。该槽满载时本 job 保持等待、不会 fallback，但其他槽上的独立 job 仍可调度；不能和 `preferred_slot` 同时声明。它用于保护他人保留卡或固定证据所在 GPU，不提供多 job 原子发射语义。 |
| <a id="trainer-run-binding"></a>`run_binding.json` / trainer 运行绑定 | trainer 在自己确定真实 RSL log directory 后原子写出的不可覆盖 sidecar；把 queue claim、PID/PGID、进程 starttime、物理 GPU、source/argv 与真实日志/checkpoint 根绑定起来。外部脚本不得用时间戳或 stdout 猜目录。 |
| <a id="milestone-attestor"></a>`milestone attestor` / 里程碑取证器 | 只沿已验证 `run_binding.json` 查找预注册 checkpoint，核对迭代号、finite、hard contract 和 launch lineage，再以 no-clobber 方式写证据；它不启动训练、不判卷，也不自动 stop/promote。 |
| <a id="runtime-binding-flag"></a>`runtime_binding: true` / exact 运行绑定开关 | 轻量训练 YAML 的显式 capability：只有 pinned source 已包含 trainer callback 与 `lean_queue_runtime.py` 才能设为 `true`，此时科学 run 才注入 claim/binding path 并允许里程碑取证。默认 `false` 保持旧 source 与非科学 boot warmup 兼容，绝不追认历史 run。当前 P1 只支持 fresh run，`checkpoint_path` 必须为空。 |
| <a id="boot-warmup"></a>`boot-warmup` / 1-env 缓存探针 | 为一个 exact source/Pod/GPU 单独创建的非科学小运行：沿用动作、题库和配方，但强制 1 environment、2 updates、独立 namespace/claim 与 180 秒 boot 上限；它只回答最小 importer/cache 路径能否越过 first iteration，不能代表正式环境数的 scene。其 checkpoint/指标永远不能当实验结果；失败只管理 claim 中的 exact PGID。 |
| <a id="full-scene-probe"></a>`full-scene-probe` / 完整场景启动探针 | 从一条 ready/blocked exact job 派生的非科学两次 update 运行：保留 source、Pod/GPU、完整 scene recipe 与原 `num_envs`，只改独立 `full_scene_probe_not_science_*` run name、`max_iterations=2`、save interval `1`，并写入 `_full_scene_probes/` namespace。只有看到首个 `Learning iteration` 才算 boot ready；它不可取证、判卷、晋级或当训练成绩。 |
| <a id="virtual-return-rate"></a>`virtual_return_rate` / 上台率 | **训练时的主曲线**，分正反手。按机会数算：到点击球算一次机会，球被打到 + 过网 + 落进对面台内才计 1；打不到就是 0。样本不足时是 `None` 不是 `0.0`。辅助指标是 `virtual_hit_rate`（击球率）；跟踪三合格 composite **只作诊断，不用它判死臂**。正式入账只认 MuJoCo 考卷版。判读规则见[结果判读](operations/read_and_report_results.md)。 |
| <a id="rally-denominator"></a>rally 全分母 vs 幸存者分母 | `virtual_return_rate_rally_*` 的分母是**挥拍起手数**（摔倒计败），`virtual_return_rate` 的分母只算活到击球帧的挥拍。两者可差近一倍（实测 0.8415 vs 0.5255）。**报人一律用 rally 全分母**，幸存者口径只作诊断辅尺。 |
| <a id="report-all-cells"></a>报数全表 / 十六格 | 任何臂的读数按 **正手/反手 × 击球率/上台率 × 训练内/考卷 × 单球/连续** 报，没数的写"未测"。**只报总数 = 漏病**：一次 45% 是"反手 0.85 + 正手 0.0000"平均出来的，一次全局击球率 0.998 盖住了反手全零。多 seed 同理，平均数掩盖最差初始化。 |
| <a id="phase-scan"></a>`--phase-scan` / 可回球性相位扫描 | `gen_stage1_questions.py` 的一个模式：逐帧问"若在此帧触球（拍面钉该帧法向、拍速在其自身速度锥内），采样来球里多少比例能被合法回台"。**空挥片只有这个训练最优相位有意义。** 它从不回写 registry，只打印 `train_phase_candidates` 给人抄。工序见[动作片→训练绑定第 6 步](operations/run_motion_clip_to_training_binding.md)。 |
| <a id="train-phase-candidates"></a>`train_phase_candidates` | 登记表里与 `phase` **并列的独立字段**。`phase` = 视频真值（人何时触球）；本字段 = 训练最优相位（哪一帧运动学最适合机器人回球）。两者可以合法不同，空挥片只有后者存在。**绝不回写 `phase`。** |
| <a id="min-contact-height"></a>最低合法击球高度 | `桌面 0.76 + 球半径 0.02 = 0.78 m`。目标框 `z_lo` 低于它 = 命令机器人往台子里打球，签名是某侧回球率恒 `0.0000` 而其他什么都不报。选触球帧时门槛是 `0.88 m`（让 ±0.10 目标框整个离台）。合同见[球拍目标物理有效性](interfaces/racket_target_physical_validity.md)。 |
| `PPO` | Proximal Policy Optimization，本项目使用的批量强化学习策略优化算法。测试/合同通过不等于 PPO 已实际训练。 |
| `VecEnv` | vectorized environment，并行推进多个仿真环境的训练接口。只有配置或 preflight 时，不能写成 `VecEnv` backend 已实现。 |
| `seed` | 随机种子。配方不变、只换 seed，用来看训练是否稳定，不许只挑最好的 seed。机制尚未成立时先用一个阻断 seed；第二 seed 只给胜者和匹配对照，`3–4` seed 只给正式候选。所有已运行 seed 仍须全量报告。 |
| signed-face 漏斗 `L1 / L2 / L3` | 该实验内部的三层证据购买：L1=`512 env × 25 update` 四格发射/合同冒烟；L2=`4096 env × 1001 update` 单-seed 机制 canary；L3=胜者与匹配对照通过预注册门后才购买第二 seed。它们不是下方 `E1/E2/E3` 证据等级，也不是课程 Stage。 |
| signed-face `C2 / D2` | v9 失败证据之后新建的两条 **fresh L1 provenance 对照**：C2 的位置/拍面 guidance 都为 `0`；D2 只把有符号拍面 guidance 权重改为 `-0.4`。两者使用同 host/seed/source/PYTHONPATH/`16/16` Kit thread cap、同动作/题库/plant/预算；按团队广度优先调度分别绑定 Pod1 GPU1/GPU2，Kit boot 由 host lock 串行，boot 后训练可并发。每个 checkpoint 同时绑定各自含权重的 hard contract 和带 GPU lane 的外层原子 launch claim。`2` 表示全新 namespace/provenance 修正版，不是第二 seed，也不是 motion 文档里的 C2。 |
| signed-face `C3 / D3` | C2 的零摩擦声明与真实非零 plant 不一致后新建的 fresh L1 配对：C3 关闭有符号拍面引导，D3 只把该引导设为 `-0.4`；两格都把 `task.plant.zero_joint_friction=true` 唯一地贯穿 argv、optimization recipe、outer claim、runtime marker、31/31 hard-contract 摩擦值和 checkpoint replay。`3` 是全新 namespace，不是第三个 seed；L1 通过也只证明合同/发射闭合，不证明动作效果。 |
| one-shot continuation | 旧 trainer 已按原配方完成，但外层验证器假拒绝后使用的**一次性续接控制器**。它只消费内容绑定的旧 claim/log/hard-contract/checkpoint，并且只能 claim 尚未存在的配对臂；不能覆盖旧证据、重跑已完成臂、改变训练配方或自动 retry。具体版本仍须单独说明，不能把 source gate 自动写成 runtime 已执行。 |
| signed-face C2/D2 `v1r1` | 第一版 D2-only one-shot continuation：修复了 v1 对 `[1.0,-1.0]` 的 float/int 假拒绝，但又错误要求 trainer 的五键 compact `question_bank` 记录直含第六个 `physics_contract_sha256`。最后一次成功只读快照证明它没有安装 control、写 continuation/claim 或启动 D2；因此 bytes 只作冻结负例，禁止运行。 |
| signed-face C2/D2 `v1r2` | 第二版 D2-only one-shot continuation：它正确接受 trainer 的五键 compact `question_bank` 并独立绑定 physics contract，但真实 runtime 在写任何 continuation/attestation/D2 claim 前证明 C2 的 31 个关节摩擦系数全为非零，与 manifest 的零摩擦声明矛盾。因此 C2 只保留 nonconforming 证据，D2/v1r2 永久 **NO-LAUNCH**；历史 absence 不构成授权。 |
| signed-face `v6r1` | `v6 retry validator 1`：为原 v6 的 D 单格准备的第一版补跑 validator。真实 Pod 只安装了 config/script，没有 claim、training run、signal 或训练；runtime `validate` 揭示它错误要求一个本应不存在的旧 training dir，因此已被 `v6r2` 取代，禁止启动。 |
| signed-face `v6r2` | `v6 retry validator 2`：修正 `v6r1` prelaunch validator 的纯源码版本，只支持 `static-validate`（静态合同检查）；它没有 runtime preflight、命令重建、进程检查、launch 或 finalizer consumer，明确 **NOT LAUNCHED**。original-v6 与 foreign-v8 都是前三格完成后，串行 launcher 的第 4 格 D 卡在 Kit/scene-start boot、均未到 hard contract/checkpoint；根因未知，不能写成学习、配方或拍面 reward 失败，也不得自动重试。下一步是 boot root-cause 与全新 v6r3-or-later preregistration，不是启动或复用 v6r2。 |
| `checkpoint` / `ckpt` | 训练到某个迭代时保存的模型存档，例如 `model_2000.pt`。 |
| `lineage` / 谱系 | 从初始模型、代码、资产到 checkpoint 的来源链。来源混了就不能声称严格单变量。 |
| `PID / PGID` | `PID` 是单个进程编号；`PGID` 是进程组编号。管理长任务时只能从经核对的 launch sidecar 读取 exact 数值并检查组成员，不能用相似命令行模式猜所有权。 |
| `fresh` / `causal continuation` | `fresh`=从零开始训，才可能成为新正式谱系；`causal continuation`=从旧 checkpoint 继续训，可看改动方向，但谱系不纯。 |
| `v4rg` | 项目内部的第四版参考挥拍动作对（正手+反手）：已重新对齐到机器人坐标，未发现明显起始毛刺或支撑脚滑移。它是资产族名，不是训练算法名。 |
| `v4rg_runtime_order_v3` | 当前 formal fresh setting 实际绑定的 v4rg 版本：schema-2、50 Hz，已迁移到 runtime body order。只写“v4rg”时只表示资产族，不足以复现 formal setting。 |
| `legacy v4rg` | 迁移到 `v4rg_runtime_order_v3` 之前的旧资产/顺序；可用于 causal 历史诊断，不得冒充当前 fresh exact 动作。 |
| `schema-2 motion` | 动作资产的第 2 版数据合同，包含 runtime 所需的关节/刚体顺序与元数据。 |
| <a id="canonical-ready"></a>`canonical ready` / 共同等待姿态 | 五类击球动作共用的零速等待状态。已登记的 `canonical_ready_v1` 只是 exact `bh_loop_c` source NPZ 第 0 帧的 donor baseline，不是最终或拍面中立 ready；“姿态相同”与“`joint_vel/body_lin_vel_w/body_ang_vel_w` 三组速度在 start/end 共六类 checks 为零”也是两个独立条件。该 baseline 尚未通过 face-neutrality、站立平衡、训练消费或部署认证，不能把普通 default stand 或某个 `adv2c3` 切片首帧悄悄当成它。 |
| <a id="canonical-ready-sidecar"></a>`canonical-ready sidecar` / 共同 ready 单状态旁车 | 不是 schema-2 motion 的单状态 NPZ：它保存 31 关节姿态/零速度、floating root 位置/四元数和 donor 注释；完整 runtime ready 还必须由独立的 32-body FK 真值旁车补齐。路径或文件 SHA 不能替代 donor source SHA、donor frame、runtime-float32 pose digest 和 exact-zero velocity digest。该旁车不能冒充可播放 source/output clip。 |
| <a id="face-neutral-ready-audit"></a>`face-neutral ready audit` / 拍面中立等待位审计 | 在同一 exact FK 模型和 scope 下，比较 ready 拍面法向到反手目标与翻面正手目标的球面角距离；“中立”要求两侧距离在预注册容差内相称，不等于把某个腕关节写成 `+90/-90`。当前 v1 的 upper 距离约 `33.4/146.6 deg`、full 约 `32.9/147.1 deg`，明确失败；下一候选须保留现有站姿/root/拍心设计目标并重解右臂，形成新路径/SHA，不能覆盖 v1。 |
| <a id="adv2c3"></a>`adv2c3` / 三分之二前奏切片 | 从 `floor(2 × 登记触球帧 / 3)` 附近开始、保留原触球窗和随挥的历史紧凑切片。它没有搜索入口/出口，没有把共同 ready 融入整条路径，也没有完整重定时；因此现行五动作编译只把它记作 comparator（比较基线），不把它预定为主件、搜索 seed 或 tie-break。 |
| <a id="motion-body-scope"></a>motion `upper/full body scope` / 上肢与全身动作范围 | 同一高层击球语义的两种数值参考：上肢版把人体髋部贡献按已登记规则折入机器人腰/躯干并维持站立下肢；全身版保留合法的 root、腰腿和足接触参考。两者共享动作 ID，不共享平衡、地面、动力学或行为证书。 |
| <a id="motion-timing-envelope"></a>`motion timing envelope` / 动作时序必要包络 | 对最终绝对关节轨迹检查位置、URDF 速度和具名加速度上限的必要筛选。当前 `diagonal_timing_envelope_v1=(effort-|bias(q,qdot=0)|)/Mjj` 只是一阶对角下界，不是完整逆动力学、恒扭矩、平衡或真机证明；不得简称 `C3`，以免与 signed-face C3 实验臂混淆。 |
| <a id="no-brake-time-law"></a>`no-brake time law` / 击球机会末前不提前刹车时间律 | 沿一条已选几何路径，把一维路径参数的加速度限制为在 exact `window_end` 前非负的运动学时间律；零加速度平台允许，窗口内也允许继续加速。它不是逐关节恒加速度、恒执行器扭矩、拍速单调或数学 `C3` 连续曲线；“bang-bang”只可描述某个受限求解器的切换结构，不能当这条合同或光滑度等级的名称。 |
| <a id="canonical-t-hit"></a>`t_hit` / 击球时刻 | compiler 可记录“共同零速 ready 到重定时后 source-anchor marker”的 diagnostic 时间，但 source anchor 只证明 lineage/排序，不能自封为正式触球真值。正式 post-retime `t_hit` 必须来自具名 behavior/contact authority，并使用动作特定可接受范围；通用 `t_hit<=0.5 s` 硬门已撤销。 |
| <a id="canonical-t-cycle"></a>`t_cycle` / 完整动作周期 | 从共同 ready 出发，经击球机会和随挥，再回到同一 ready/恢复边界的完整时间。它必须与 `t_hit`、site strike speed、无桌碰和恢复分别报告；更短 cycle 不能掩盖错误触球锚或撞桌。 |
| <a id="motion-artifact-class"></a>`motion artifact class` / 动作工件类别 | 先区分工件，再谈发布：`diagnostic_face_core` 只是一段 scope-specific 换面诊断核心，`canonical_compiler_output` 才是已经包含 direct canonical-ready→contact→ready bridge 的完整候选输出。诊断工件在 publication state machine 之外；改文件名、复制 bytes 或补一个 `publication_class` 字段都不能把它变成 `compiler_candidate`。 |
| <a id="motion-build-manifest"></a>`motion build manifest` / 动作构建清单 | 内容绑定 source、工具、模型、参数和输出 SHA 的构建收据。它说明“哪些字节怎样生成”，不等于任何安全能力证书；拒绝请求可以只有 rejection manifest，不得同时留下看似成功的 NPZ。 |
| <a id="canonical-ready-bridge-receipt"></a>`canonical-ready contact bridge receipt` / ready 到击球机会整桥收据 | 完整候选 manifest 的必需部分：绑定 ready/source、entry/exit、direct ready→core→ready 几何、时间律、source-marker→output-time 映射、首末共同 ready 和零速度摘要。face-core receipt 没有这座桥；缺桥或独立 verifier 不能从 exact bytes 重算时，工件只能保留 diagnostic 身份。 |
| <a id="protected-window-digest"></a>`protected-window digest` / 受保护击球窗摘要 | 分别对 exact source window 与 compiled output window 的六个 schema-2 时序通道、明确 frame-index 集合、dtype/endianness、shape 和 C-order bytes 做域分离 SHA-256。source digest 与 output digest 不是彼此相等的承诺；二者必须由同一 transformation receipt 连同 source/output 整文件 SHA、marker 映射和允许变换绑定，防止把别件或别 scope 的窗口拼接进来。 |
| <a id="mjb"></a>`MJB` / MuJoCo compiled model binary | MuJoCo 把顶层 MJCF、其 XML include、mesh 等资产解析后生成的二进制模型。动作构建同时绑定顶层 MJCF SHA 与 MJB SHA，避免“主 XML 没变、include/mesh 已漂移”的假 lineage；它仍不是碰撞、动力学或真机证书。 |
| <a id="motion-source-registry"></a>`motion source registry` / 动作源注册表 | 把 exact `<motion_id, body_scope, variant>` 与源 NPZ、完整合成 recipe、canonical-ready 和模型哈希绑定的版本化信任根。临时 JSON 或只有自报名的 registry 不是项目采纳；固定仓库路径和内容 SHA 必须在同一审查中钉住。缺 trust root 的可行结果只能叫 `pre-registry diagnostic`，不能升格为 registered candidate。 |
| <a id="motion-publication-state"></a>`motion publication state` / 动作发布状态机 | registry 行字段名统一为 `publication_class`，只允许逐级、单向、不可跳级地新建不可变记录：`compiler_candidate=(false,false,false)` → `training_adopted=(true,false,false)` → `deployment_adopted=(true,true,false)` → `hardware_adopted=(true,true,true)`；三个布尔值依次是 training/deployment/hardware authorization。每次晋级都要由独立 verifier 重读原件、完整证书链和必需工件后产生新 manifest/registry bytes，不能原地改名或手改布尔值。证据失效时另记 revoke/quarantine disposition，不把旧记录倒退改写成较低 class。 |
| <a id="registry-shared-ready-digest"></a>`registry shared-ready digest` / 注册表共同 ready 摘要 | bank 级、不是单行自报的 SHA-256：域分离绑定 canonical-ready 文件 SHA、donor source SHA/frame、runtime-float32 joint/root/32-body pose digest、共同 ready 的 `joint_vel/body_lin_vel_w/body_ang_vel_w` 三组 exact-zero 数组、ready-FK SHA，以及五个有序 clip 的首末 endpoint digest。每件的三组速度在 start/end 各查一次，合称六类 endpoint zero checks。五行必须导出同一个值；任一 endpoint、donor、scope 或 body order 漂移都会改变它。 |
| <a id="motion-alignment-digest"></a>`motion alignment digest` / 动作对齐摘要 | 对五个有序 registry row 的动作身份、路径/哈希、family、phase、击球机会、拍面符号、registry shared-ready digest，以及 source/build/model/applicability/evidence/adoption provenance 做确定性序列化后的 SHA-256。它防止运行时各列错位，不是 registry JSON SHA 的别名。 |
| <a id="canonical-runtime-four-pin"></a>`canonical runtime four-pin` / canonical 运行时四重钉住 | 非 audit 的动作消费必须由配置同时给出并精确核对：registry JSON SHA、motion alignment digest、canonical-ready 文件 SHA、canonical-ready FK 真值 SHA。缺任意一项都失败封闭；运行时自己刚读出的 digest 不能冒充调用方预先钉住的期望值。 |
| <a id="identity-only-registry-audit"></a>`identity-only registry audit` / 仅身份注册表审计 | `authorization_purpose=None` 的严格只读解析：可以核对 schema、内容哈希、共同 ready、provenance 和 alignment digest，但不授予 training/deployment/hardware 权限，也禁止导出可直接喂给动作 loader 的 runtime table。 |
| <a id="motion-capability-certificate"></a>`motion capability certificate` / 动作逐门能力证书 | 对一个 exact `<asset, body scope, variant, timing>` 只解锁一层能力的不可覆盖证据，例如 schema/L0、vendor L1、桌网、动力学或行为；上游通过不能跨资产、跨 scope 或跨变体继承下游能力。 |
| <a id="motion-evidence-certificate-chain"></a>`motion evidence certificate chain` / 动作证据证书链 | registry 声明 `E1–E5` 时，从 `E1` 到所声明等级每一级都必须恰有一份 passing certificate；manifest 和每份 certificate 都以路径与 SHA-256 内容寻址，并精确绑定同一 `<motion_id, scope, variant, NPZ SHA>`。只写 `evidence_level=E4`、复用别件证书或缺中间级都不能晋级。证书链只证明已登记证据的身份与连续性，不把未实际运行的 Gate 变成通过。 |
| <a id="strict-motion-artifact-parser"></a>`strict motion artifact parser` / 动作工件严格解析器 | 不只看路径、哈希或自报版本，还解析并核对工件内部合同：schema-3 训练题库绑定动作 SHA/帧数/触球锚，training config 绑定动作 lineage，ONNX metadata 绑定题库/config/model/ready/timing，ONNX model 必须由官方 parser 与 full checker 通过。所需 parser 缺失或 bytes 虽有正确哈希但内容无效时必须失败封闭。 |
| <a id="shared-zero-speed-boundary"></a>`shared zero-speed ready boundary` / 共同零速 ready 边界 | 五个动作的开始和结束都必须是同一个 runtime-float32 关节/root/32-body 姿态，且 `joint_vel/body_lin_vel_w/body_ang_vel_w` 三组 runtime 速度在 start/end 共六类 checks 均为字面零。episode 内只有处在该开始或结束边界时才允许换 `motion_id`；挥拍中途改槽必须拒绝。 |
| <a id="fail-closed-motion-preprocessing"></a>`fail-closed motion preprocessing` / 失败封闭动作预处理 | 遇到缺字段、未知顺序、非有限值、越限、模型漂移、坏 provenance 或窗口泄漏时非零退出，且不发布候选 NPZ或成功证书；禁止静默裁剪、自动降速、换 profile 或覆盖旧输出。 |
| <a id="motion-action-id"></a>`motion_id` / 五动作槽 | planner、题库、训练、ONNX 与 runtime 共同使用的版本化动作身份；至少区分正手拉、反手拉 C、正手挡合成、反手挡和高点拍压。它不同于正/反手 `swing_sign`，每条 clip 还需独立绑定物理拍面 sign。同一颗球选择后不可在挥拍中换槽；不可行时只能 abort/rearm。 |
| <a id="motion-l0-static"></a>motion `L0 static audit` / 动作 L0 静态审计 | 对 exact schema-2 动作做的纯 CPU 离散静态门：核对字段/顺序/finite/形状/时间、四元数、vendor MJCF 关节范围、逐帧 FK 与 root-foot 接地，但不调用 `mj_step` 或推进动力学。source/static gate 通过只说明计划、validator 和合成反例闭环；必须另有 exact 资产的 runtime certificate 才能声称 L0 runtime 通过。它不包含 vendor L1 自碰/自打、桌网扫掠、动力学、RL、Gate3 或真机，也不是 signed-face 的 L1/L2/L3 或证据等级 E1/E2/E3。 |
| <a id="motion-vendor-l1-safety"></a>motion `vendor L1 safety audit` / 动作厂商 L1 安全审计 | 在 exact vendor MuJoCo 碰撞模型中，把一条已通过动作 L0 的 schema-2 整轨做有限密集插值并逐样本检查机器人自碰、球拍/拍柄碰机器人及关键部位余隙。任一硬失败都会否决整条动作，不能由 reward 或其他好成绩补偿。当前 B 合同把 151 个 50 Hz 原帧按每段 8 个子步扫成 1201 个 400 Hz 样本；`<5 mm` 的 hard 判定直接使用 MuJoCo 饱和谓词，不用距离二分的近似 midpoint。关键组包含右肩三轴和右肘，仅右腕/手/球拍安装链从 proximity pair 排除。这仍是有限采样，不是数学连续时间扫掠证明，也不含桌网、动力学、训练、Gate3 或真机。它与训练阶段的 L1/L2/L3 不是同一层级。 |
| <a id="float32-ulp"></a>float32 `ULP` / 相邻格宽 | `unit in the last place`：某个 float32 数附近相邻两个可表示数之间的距离。它随数值量级变化，不是固定物理容差；动作 L0 V2 用预注册的格数、近零 floor 和独立物理上限共同约束纯舍入差，不能用它掩盖关节、接地、支撑或安全失败。 |
| `schema-3 bank` | 题库和判卷的第 3 版合同：训练题与考试题分开，题序、分母、动作和 SHA 可绑定。它不是 schema-2 motion 的升级同一件事。 |
| `q10` | 每个动作/侧各 10 题的快速方向卷；只看有没有苗头，不许据此停训或晋级。 |
| <a id="q50-and-k100"></a>`q50` | 每个动作/侧至少 50 题的同卷决策考试。当本项目只考正手和反手时，合计通常是 100 次。 |
| `K100` | 当前一张具体的、100 行 immutable paper：正手 50 + 反手 50，共用固定 schedule/order，且不删失败尝试。`q50` 是考试协议类型，`K100` 是这次的具体卷，两者不是普遍同义词。`K100` 也不自动表示 exact，还必须核对题库 bytes、语义和分母。 |
| `signed-face K100 checkpoint attestor` | 给一份**显式指定**的 checkpoint 做判卷前一次性取证：核对 filename/embed iteration、finite、fresh lineage、相邻 hard contract、producer claim、评测源码/runtime、MJCF/plant 和 actual K100 activation，再在 checkpoint-SHA 唯一路径写 evidence/claim。C3/D3 v2 还在 claim 前绑定训练时 ignored Isaac A3 asset inventory、hydrate/verify 角色及 `libGLU.so.1` 存在性。它不运行 judge、不产生成绩，也不授权停止或晋级。 |
| `C3/D3 K100 paired execution consumer` | 只消费已分别通过 checkpoint attestor 的 C3/D3 终档，在独立 eval worktree/runtime 中用同一 immutable K100 顺序判卷，并发布两侧 raw count 与 `D3-C3`；v2 还要求两侧共享同一已验证 ignored A3 asset closure。它不重跑训练，也不自动授权 L2、第二 seed 或 stop/promote。 |
| Python `BankExam` | 仓库内的独立 policy 考试：Python 在 MuJoCo 中物理推进机器人，每题单独重置，再从击球时的球拍状态用解析模型推算接触和落台。它没有真实球拍—球碰撞，也不包含 planner、生产 C++ runner 或完整厂商运行链，因此不是 `Gate3/Gate3B`。 |
| <a id="mujoco-checkpoint-direction-sentinel"></a>MuJoCo `checkpoint direction sentinel` / 检查点方向哨兵 | CPU-only、[`no-clobber`](#no-clobber) 的检查点诊断封装：在同一不可变 BankExam 上逐个运行 ONNX，记录 31 维 actor/q_des、proxy 前 raw qvel、世界系球拍速度、有符号拍面/闭合速度、撞桌和物理摔倒。它在 MuJoCo↔PhysX plant/桌碰传感器尚未获证时强制屏蔽所有 pass/return/上台成绩；因此是方向与安全 stop gate，不是正式 MuJoCo 分数或 Gate3。 |
| `readiness audit` | 开卷前的只读资格检查：核对 checkpoint、contract、题库/schedule 和本机路径，不启动 judge，不产生成绩。 |
| `all-four activation` | 四 seed 同卷的启动授权文件：只有 Pod1/Pod2 readiness audit 和四份 checkpoint 全对上才能生成。它只允许下一步 `prepare`，不是 judge 已启动，更不是新分数。`judges_started=0` 就是还没有子判卷进程启动。 |
| `prepared_not_started` | 两个 Pod 已按 activation 物化 no-clobber runtime contract 和 K100 路径，但 `jobs_started=0`、`auto_start=false`。这比 activation 多一步执行纸面，仍不是已开卷或有结果。 |
| <a id="persistent-supervisor"></a>`persistent supervisor` / 持久监督器 | 对一条已审过的长任务做内容绑定、脱离调用 shell、无覆盖启动并只读复核身份的窄封装。本项目 q50 版本只有一次 `launch` 和只读 `inspect`，没有重试、信号、远程登录、训练、部署或真机权限；详见[接口合同](interfaces/q50_persistent_supervisor_contract.md)。 |
| `no-clobber` | 只允许首次创建产物；目标路径已存在就拒绝，不会静默覆盖旧合同、日志或结果。 |
| `NO-MERGE` | 该候选当前禁止合入 `main`。通常表示即使部分测试通过，仍有会制造错误证据或越权的明确缺口；修复并复核前不得把它当现行能力。 |
| 解析击球/解析上台 | 没有在 simulator 里用真实球-拍-台-网接触重放，而是从触球时的拍位/拍速/拍面经解析接触模型推出击球和落台。它是诊断尺，不是 physical return。 |
| `raw-A / physical-B` | 179-D actor 消费球拍 mount 原始 `+Y` 面法向 A；planner/外部协议传对手向物理击球面 B。runner 按正/反手的 per-clip sign 把 B 还原成 A。只看无向平面或对法向做自动翻转，会隐去“用了反面”。 |
| `signed-face honesty gate` | 要求判分器保留拍面法向正负号，不得通过 `orient_normal`之类步骤把 `n` 和 `-n` 当成同一拍面。这条门未通过前，高解析上台率不得晋级 setting。 |
| `exact` | 训练与判卷的动作、题库、观测、动作输出和执行合同能逐项对上。它只说合同一致，不等于物理已对齐真机。 |
| `formal target` | 实验前预先指定、有资格进入正式决策卷的 setting。 |
| `accepted baseline` | 已通过预定稳定性、留出卷和必要部署门，团队可以正式往上比的基线。`formal target` 不会自动变成 accepted baseline。 |
| `plant` | 机器人与环境在仿真里的物理对象：质量、惯量、摩擦、驱动器和数值积分都在内。 |
| <a id="raw-action-rate-l2"></a>`action_rate_l2` / `action_rate_weight` / 相邻 raw action 二次差惩罚 | 现役 50 Hz 控制中逐 tick 计算 `sum((action[t] - action[t-1])^2)` 的全 31 维正则项；`action` 是 affine transform 与 q_des clamp **之前**的 policy 输出，RewardManager 再乘权重和 `0.02 s`。它每步都把当前输出连到上一输出，因而能平滑整条序列，但没有更长记忆、不是动作二阶差分，也不直接量到比例微分控制器收到的 q_des。现役权重为 `-0.10`；权重随控制频率改变不能直接横比。 |
| <a id="qdot-limit-hinge"></a>`qdot-limit hinge` / 关节速度限位铰链惩罚 | 只在实际关节速度超过各自运行时速度上限的一定比例后开始收费：`mean(relu(abs(qd)/limit-margin)^2)`。它读取 31 个 articulation 关节的真实速度和同顺序真实上限，不是 action-rate 平滑的别名，也**不施加任何随机力**；权重为非正惩罚，默认 `0` 表示关闭。 |
| <a id="processed-qdes-slew-hinge"></a>`processed_qdes_slew_hinge` / 腿腰执行目标突变铰链 | 默认关闭、只用于 50 Hz 的恢复期平滑项。它读取 affine transform 与 train=deploy clamp **之后**的 q_des，只看 3 个腰关节和 12 个腿关节；把相邻 q_des 变化除以各关节 `qdot_limit * 0.02 s`，仅对超过 `margin` 的尾部收费。现行预注册用 `margin=0.85`、同一拍触球后 `0.20–1.55 s`、weight `-0.25`；每次 reset 后首步无历史而强制 mask。它不是全程 dense action-rate 的另一个名字。 |
| <a id="balance-action-slew-matrix"></a>平衡 action-slew `W/V × C/N/H` 六格 | 2026-07-20 诊断矩阵：`W` 是“拍心优先 × 自由非击球臂”的稳定 `model_6700` parent，`V` 是“拍速优先 × 强准备姿态”的高摔倒 `model_6700` parent；`C` 保留 raw action-rate `-0.10`，`N` 把两种平滑都关掉，`H` 关闭 raw action-rate 并启用恢复期 processed-q_des 腿腰铰链 `-0.25`。这些字母只在该矩阵内成立。 |
| <a id="balance-stability-waves"></a>平衡稳定性 `Wave A / Wave B` | `Wave A` 是上面的 W/V×C/N/H action-slew 单变量矩阵；`Wave B` 是下一轮下半身 matched ablation。2026-07-20 输入门已拒绝 M0 四份横移动作作 moving teacher（[`stance_passed=0/4`](#motion-m0) 表示四条都没同时守住移动后的自身初始站姿与 no-narrowing 脚距门；formal/schema2/training/hardware 均为 false），所以 immediate Wave B 只能比较现役 upper-only、当前静态 v4rg 下半身参考或不依赖 demo 的稳定约束；实际接口、flag 和 run name 仍须另行冻结，不能从 Wave A 猜。两波不得混成一个多变量结论。 |
| <a id="half-second-sprint-arms"></a>半秒冲刺 `U/V/W/X/Y` 五臂 | 同一 task-revision sprint 的五个配方代号：`U`=拍心优先×强准备，`V`=拍速优先×强准备，`W`=拍心优先×自由非击球臂，`X`=拍速优先×自由非击球臂，`Y`=拍心优先×触球窗老师静音。字母只在该冲刺及其明确后代中成立；W/V 被 Wave A 作为两个 parent 复用。 |
| <a id="checkpoint-contract-mismatch"></a>`checkpoint_allow_contract_mismatch=true` / 显式允许旧合同热启动 | trainer 允许从训练合同不同的 parent checkpoint 恢复 policy/value/optimizer/normalizer，用于方向性 continuation。该 flag 必须把后代标为 lineage-inexact / formal-ineligible；它不能把 intentional mismatch 解释成 exact resume，也不能授权正式考卷或部署。 |
| <a id="zero-write-onnx-plan"></a>ONNX `--plan` / 零写入导出预检 | 加载并验证 checkpoint、动作、normalizer、题库与训练合同，但在创建目录、临时文件、计算图或 ONNX 制品前退出。通过只证明导出输入和合同检查可走通，不表示 graph 已导出、vendor 已运行或行为已通过。 |
| <a id="allow-inexact-contract"></a>`allow_inexact_contract=true` / 历史 inexact 诊断绕过 | 只允许明确标记的历史 Isaac 诊断加载训练合同谱系不精确的 checkpoint。它永久禁止用于 W/Y production/vendor、正式行为卷、晋级或部署；新实验不得把它当成修复 exact-lineage 的捷径。 |
| <a id="balance-launch-manifest"></a>action-slew `launch manifest` / 发射输入清单 | Wave A 命令生成前由人独立复核的 canonical JSON；同时绑定 clean detached source commit、queue config/runner SHA-256、A3 runtime asset tree、preconverted `model.usd` 文件及其完整 6-file sibling bundle tree，以及正反手动作、题库、W/V checkpoint 与 parent contract 的逐项 SHA-256。调用者必须同时给 exact 文件和其已复核的文件 SHA；缺项、占位 digest、内容 digest 不符、本地 queue 漂移、bundle symlink/文件数/总字节/tree digest 不符都会 fail closed。它只提供命令生成 authority，不执行 SSH 或 trainer。 |
| <a id="balance-command-render-latch"></a>Wave A `--authorize-launch` / 命令渲染闩 | action-slew queue 的显式 CLI flag。没有它时只打印 NO-LAUNCH plan；有它时仍须通过发射输入清单，并由 runner 的只读 Git 门验证 `HEAD == fetched origin/main`、相关 tracked bytes clean、manifest/config/runner 与 `origin/main` 相同、同一 NOW 条目绑定 human owner/executor/branch/queue id。脚本只把每格 exact SSH `launch_command` 渲染进 JSON，不自行执行。它不是对真实机器人、judge、stop、retry 或部署的授权。 |
| <a id="balance-probe-receipt-set"></a>Wave A `probe receipt set` / 六格探针收据集 | 六格 4096-env×24-step/env×2-update probe 各自自然退出后，由 queue 输出的 dedicated exact supervisor/verifier 生成的六份内容寻址收据。每份绑定 manifest/claim/run binding，证明 absolute milestone `[6701]` 与 `model_6701.pt`、finite policy/value/full optimizer/two normalizers、C/N/H exact weight-margin-window-applied markers、lineage=`0`，并在 6700/6701 两步检查 processed-q_des、completion/fall/legal-return、ready-tilt 与 qdot 的 tag 和守恒关系。processed-q_des 与 qdot observed 必须逐 update 精确为 `4096×24=98304`、两步总计 `196608`；首个 0.48 s rollout 可在触球后 0.20 s 恢复窗尚未产生 eligible sample，因此 processed-q_des 资格分母只要求逐步非负、两步合计非零；行为、ready 与 qdot 的其他预注册分母仍逐步非零。还要证明无 fatal 及 leader/PGID/GPU 已释放。它不能借用 fresh-probe 的相对 `[1]` 语义。本地必须按 `DIR/JOB_ID/probe_receipt.json` 收齐六格；train 命令生成会逐份重验并绑定整组 digest，不能用一个人工布尔值代替。 |
| <a id="balance-probe-generations"></a>Wave A `probe5 / probe6 / probe7 / probe8 / probe9 / probe10` / 第五至第十代探针 | 这些都只是非科学的六格完整场景探针，不是训练结果。`probe5`/fresh v4 得到五份 exact receipt，W-H 在 `fork→exec` 身份过渡窗口被拒绝；`probe6`/fresh v5 的 W-C 因 transaction wrapper 破坏 multiline Python 参数而在 trainer 前失败，其余五格未发。`probe7`/fresh v6 的 W-C、V-C、V-N 得到三份 exact receipt；W-N 在 scene boot 冻结，由 180 s locked watchdog 对 exact 组 TERM/KILL 后 rc=`125`，W-H/V-H 未发。成功的 W-C/V-N 排除了 W parent 与 `action_rate=0` 各自作为必要失败原因，所以只记 infrastructure transient，不改 Reward 结论或 timeout。`probe8`/fresh v7 只发 W-N：它在 Pod1 GPU1 的 `sim.reset` 阶段 SIGABRT（trainer exit=`-6`、外层 rc=`134`），日志末行为 `malloc(): invalid size (unsorted)`，没有首个 iteration、binding 或 receipt，其余五格未发；两轮 closure 证明六张 GPU、相关进程和 lock holder 全空。这是 pre-RewardManager infrastructure 失败，不是 C/N/H 机制结果。v4/v5/v6/v7 均 immutable，禁止重试、补格或跨代混收据。`probe9`/fresh v8 `/workspace/codexschema/phase1_balance_action_slew_v8_20260720` 已严格按 W-N Pod1 GPU0→receipt+closure→W-C Pod1 GPU1→receipt+closure→W-H→V-C→V-N→V-H 完成；六格均 natural exit=`0`、normal=`true`、first iteration=`true`、exact verifier passed，并各自在下一格前闭合。六份同代 receipt 的本地 set 重验 SHA-256 为 `cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`，只解锁当时 current-main 的科学 train 命令生成，不证明机制效果。`probe10`/fresh v9 保持同一 W-N GPU0/W-C GPU1 swap 和全局串行顺序，使用 stage-aware failure audit；当前未发射，且必须重新取得 fresh v9 同代 `6/6` receipt，禁止复用 probe9 收据。W-N GPU0 与 W-C GPU1 的历史通过只排除必然失败，不证明 GPU 等价，也不覆盖 probe7/8 历史。这里的 v4–v9 只是隔离运行目录的 namespace 代次，不是算法版本。 |
| <a id="balance-science-attempt-generations"></a>Wave A scientific train `attempt1 / retry2` / 科学长训尝试代次 | `attempt1` 是 v8 root 上 first current-main scientific launch：只发 W-N Pod1 GPU0，停在 `sim.reset`，无首个 iteration、Reward、checkpoint；locked launcher rc=`125`，外层 rc=`121` 来自 train stage 错要 probe-only child evidence。四次闭包后 exact group/GPU/locks 全空，其余五格未发，所以 v8 immutable、infrastructure-only / non-science，不能判断 C/N/H。`retry2` 是 fresh v9 候选的 run-name 代次，不表示已经重试成功；它只有在 probe10 fresh `6/6` receipt 后才可生成科学命令。两者都不改变 formal-ineligible、inconclusive / not adopted 状态。 |
| <a id="ppo-num-steps-per-env"></a>`algo.runner.num_steps_per_env` / PPO 每环境每次更新的 rollout 长度 | Hydra `algo=ppo` 配置中 `runner.num_steps_per_env` 的完整路径；值 `24` 表示 4096 个环境每次 PPO update 各产生 24 个 step，即 `98304` 个样本。`algo.num_steps_per_env` 不是现有 key；用 `+algo.num_steps_per_env` 强行新增只会得到训练不读取的死字段。 |
| <a id="vector-policy-step"></a>`vector policy step` / 并行策略步 | 一次 policy 推理后让全部并行环境各前进一个控制周期的墙钟步骤。4096 环境、`num_steps_per_env=24` 时，一个 PPO update 含 24 个 vector policy steps；`update_wall_s / 24` 是每个并行策略步耗时，不是单个机器人的 20 ms 仿真时长。 |
| <a id="environment-step-throughput"></a>`environment-steps/s` / 环境步吞吐 | 所有并行环境产生的控制步总数除以墙钟时间。4096 环境、每 update 24 步时分子为 `98304`；它与单个 vector policy step 的墙钟互为 `4096` 倍换算口径，不能写成百万级 transitions/s。 |
| <a id="collection-vector-step-wall-s"></a>`collection_vector_step_wall_s` / 纯采集并行步墙钟 | `collection_wall_s / num_steps_per_env`；只量 rollout collection，不把 PPO learning、保存或 update 间开销混进来。 |
| <a id="amortized-e2e-vector-step-wall-s"></a>`amortized_e2e_vector_step_wall_s` / 端到端摊销并行步墙钟 | `iteration_wall_s / num_steps_per_env`；包含 collection、learning 和该 iteration 的其他开销，用于回答真实长跑每个并行策略步要等多久。 |
| <a id="collection-environment-step-us"></a>`collection_environment_step_us` / 纯采集单环境步摊销微秒 | `collection_wall_s × 10^6 / (num_envs × num_steps_per_env)`；它是并行吞吐的摊销口径，不是单环境独立仿真的延迟。 |
| <a id="collection-environment-steps-per-s"></a>`collection_environment_steps_per_s` / 纯采集环境步吞吐 | `(num_envs × num_steps_per_env) / collection_wall_s`；必须和同一完整 iteration 的 reset reason 一起报告，避免把重置风暴造成的慢速误判为 PPO 慢。 |
| <a id="device-to-host-transfer"></a>`D2H` / `device-to-host transfer` / 设备到主机传输 | 把 GPU device tensor 的值读到 CPU/host。`float(cuda_tensor)`、`.item()`、`.cpu()` 或对 CUDA tensor 做 Python `bool(...)` 都可能强制等待当前 GPU stream；把一组已经完成、语义不变的标量 reduction 按固定顺序 stack 后一次读回，可减少同步次数，但不能借机改变 reduction、dtype、更新频率或状态机。 |
| <a id="finite-qdes-execution-projection"></a>`finite q_des execution projection` / 有限目标投影执行 | 对超出运行目标包络、但数值仍有限的 raw `q_des`，执行端取最近的包络内目标；包络内映射严格恒等，不改变 policy sample 或 PPO log-prob，也不因此 reset。非有限请求、实际或 physics-substep 关节 hard edge、桌碰和摔倒仍走 hard reset；仅 predicted ballistic crossing 选择有限 brake target 而不 reset，最终/子步真实越界由 actual joint term 终止。 |
| <a id="finite-projection-soft-inset"></a>`finite_projection_soft_envelope_inset_fraction` / 有限投影软限位内缩比例 | ActionBall 有限目标投影的每侧预留量，占该关节 soft-limit 总跨度的比例；`0.05` 表示上下侧各再留 `5%`，等效把标准化执行区间从 `[-1,1]` 缩到 `[-0.9,0.9]`。raw action/PPO log-prob 不变，非 ActionBall 路径不使用；该值必须进入 training contract，改变后不能 exact resume 旧 checkpoint。 |
| <a id="qdes-projection-penalty"></a>`qdes_projection_penalty` / 投影前目标超限惩罚 | 从 raw `q_des` 到执行投影值的**投影前**归一化超出量计算的非正 Reward；不能读取投影后的零超差，否则 policy 学不到收回均值。ActionBall 首发候选权重为 `-5`，`-20` 只作预注册消融；还须逐关节报告触发率、平均超出量和贴边饱和占比。 |
| <a id="reference-metrics-only"></a>`reference_guard_mode=metrics_only` / 参考包络只记指标 | reference anchor/body/末端包络谓词继续逐步计数，但不产生 reset 或额外 Reward；它防止老师跟踪偏差把 episode 免费清空，同时不削弱实际关节、桌碰、摔倒等物理安全终止。 |
| <a id="balance-receipt-trust-boundary"></a>Wave A receipt trusted-operator boundary / 收据信任边界 | 本地 probe receipt 目录是 trusted-operator capability。内容寻址与 manifest/claim/verifier SHA 绑定用于防漂移、误配和非故意替换，不声称能抵御同时掌握本地 repo、`main` 与 receipt 写权的恶意 root。如需要抵御该威胁，必须另接受信签名者或远程 attest service。 |
| <a id="lateral-balance-perturbation"></a>`lateral balance perturbation` / 恢复窗随机横向躯干推力 | 与 qdot 无关的仿真环境轴：只在击球后恢复或挥拍前等待、且非击球窗时，把有界 WORLD-Y 力施加在 `torso_link` 质心。L0 是同随机机会的零推力对照；L1 是冻结的 `0.04–0.08 m/s` 归一化冲量 treatment。当前只有 default-off trainer E1 接线，没有 full-scene/solver/throughput 证据，禁止点火。 |
| <a id="atomic-planner-tuple-timing"></a>`atomic planner tuple timing` / 同源时刻的规划目标元组 | policy 看到的击球位置、拍速、拍面、动作侧和剩余击球时间来自同一个 source sample。`source_timestamp_compensated` 会在延迟完整元组的同时扣除已知 transport delay；`uncompensated` 是故意不扣延迟的负控；`live` 保留旧行为。它闭合一次采样内的观测一致性；同一来球在挥拍中的连续更新另由 `planner task revision` 负责。 |
| <a id="planner-task-revision"></a>`planner task revision` / 同球实时任务修订 | 一颗物理来球只分配一个 `(control_epoch, task_id)`；planner 每次重估只递增 `task_revision`，并在触球前原子更新击球位置、速度、带符号拍面和剩余击球时间。动作侧和 clip 在 task 内不可变。invalid revision 只冻结上一份好目标；全局 revoke 才撤销。runner 按 task 高水位 exactly-once 消费，不能把每次发布当成下一拍。 |
| <a id="phase-governor"></a>`phase governor` / 有界动作相位调速器 | 根据实时剩余击球时间推进老师动作相位，但相位不倒退，速度和加速度有上限；目标/时钟 revision 必须留在 task 起始快照的不可变包络内。更晚 deadline 只能让下一步减速，更早且动力学不可达的 deadline 会 fail closed。它让训练与 C++ runner 共用同一套时钟语义，但不等于 0.5 秒动作已经动力学通过。 |
| <a id="initial-tts-mixture"></a>`initial TTS mixture` / 初始准备时间混合分布 | 每颗新球揭题时从显式加权的准备时间层抽样：包含低于 0.5 秒的压力层、精确 0.5 秒点质量、0.5–0.9 秒快速部署层和更长来球层。0.5 秒是必须单独测的基线，不是最小值。每层与总数按 PPO update 精确计数并写入 checkpoint 训练合同；部署 ONNX 只绑定总可接受范围，不在 runtime 随机抽样。 |
| <a id="timing-exam-0p5"></a>`0.5-second timing exam` / 0.5 秒时序卷 | 同一不可变 K100 题表的每题都从零速度第 0 帧开始，并要求 25 个 50 Hz tick 后触球；缺测、来不及、没触球和摔倒都留在分母。它回答 policy 在 0.5 秒题上的实际行为，而不是把固定 speed multiplier 或训练 Reward 当作通过。Isaac 结果仍是 inexact 诊断，最终裁判是 vendor MuJoCo。 |
| <a id="available-time-bucket"></a>`available-time bucket` / 可用准备时间档 | 旧 rolling 池用固定每动作倍率近似 1.0/0.7/0.5 秒；它只能作工程压力诊断，不能证明动力学可行，也不能代替真实准备时间分布或 0.5 秒卷。新协议以 `initial TTS mixture + phase governor` 为主。稀疏击球机会不足时，0 回球不能作为早停理由。 |
| <a id="sparse-reward-eligibility-ledger"></a>`sparse Reward eligibility ledger` / 稀疏 Reward 资格账本 | 不看 Reward 均值猜机制是否有效，而是逐级数 exact-strike 机会、virtual capture、解析过网/落点/合法回球，以及 qdot observed/hinge-active/excess。缺机会或 hit-conditioned 通道未触发时必须继续训练；两个连续 milestone 分母完整也只表示可交给外部预注册规则判读，不会自动 stop、晋级或买 seed。当前 virtual 结果仍是解析 Phase A，不是物理触球。 |
| <a id="task-revision-pruning-portfolio"></a>`task-revision pruning portfolio` / 同父本组合保护式淘汰 | 同一 parent checkpoint 派生的一组不同机制格必须在共同 absolute milestone 上一起判，不能让早启动者和晚启动者错位比赛。单臂整数行为 receipt 先给出结构/机制/行为证据；同父本 portfolio receipt 再按预注册容差决定淘汰，并始终至少保留两条、一条实际记录过 exact-0.5 样本的候选和一个 broad-arrival 候选。exact stop 必须同时消费两份 no-clobber receipt，并在 signal 前重验进程身份。 |
| <a id="conditional-face-guidance"></a>`racket_face_conditional_guidance_weight` / 不逃离就绪区的固定预算 Reward | 默认关闭的非正拍面/就绪联合塑形。它只在击球时间窗收费：未进入触点/完整拍速紧支撑门时保持固定最大成本，进入后按就绪度把成本连续换成 15° 以上的 signed-face 误差。位置或拍速越就绪，成本绝不会更高；在外门之外拍面梯度为零，策略不能靠故意离开门来免罚。输出 `[0,1]`，权重绝对值是每个时间窗 step 的最大罚金。它不替代 signed-face honesty、碰撞/跌倒或 Gate3。 |
| <a id="v1-free-wrist-velocity"></a>`V1` / 持拍手腕线速度模仿释放 | 在 body linear-velocity imitation 中排除持拍手腕，让球拍速度主要由击球目标 Reward 决定；位置、姿态、角速度模仿和所有安全约束仍保留。`V1=true` 只表示该排除已配置，必须另以 eligible denominator 与 exclusion numerator 证明运行时真的作用。 |
| <a id="v2-strike-window-imitation"></a>`V2` / 击球窗动作模仿四分之一 | 仅在预注册击球时间窗把动作模仿总尺度设为 `0.25`，给击球目标更多控制预算；窗外模仿不变。`V2=0.25` 必须另以击球窗 eligible denominator 与 scaled numerator 证明运行时真的作用。 |
| <a id="post-swing-replay-start"></a>`post-swing replay start` / 随挥后状态重放起点 | 策略完成挥拍后把自身状态写进环形缓冲；后续真实 episode reset 在缓冲已达到最小填充量时，按 `post_swing_start_prob` 从这些状态起步，以训练吸收上一拍余势和恢复平衡。它不是 carry-state 连续来球，也不是 learned reset；比较概率前必须记录 buffer-ready reset denominator 与实际 replay-start numerator。 |
| `post-swing retry authorization` / 随挥教师重签授权 | 一份内容寻址、一次性的 JSON 授权：它把唯一可接受的原始 capture producer tuple、修复后 attestor tuple、v3 plan、capture、teacher checkpoint 和输出 namespace 固定在一起。trainer 必须从配置给出的 exact 文件与 SHA 派生两条 source tuple，不能相信 receipt 自述；它只授权 attestor attempt-2，不授权重跑 capture、首 reset、科学训练、第二 seed 或 judge。 |
| `SZ` | fresh factorial 中的一格：`S`=正反手共用同一拍面语义，`Z`=31 个关节摩擦置零。它是当前执行合同的 formal target，不是标定后的真机 plant。 |
| `SP / LZ / LP` | 同一 factorial 的其他格：`L`=旧的正反手异号拍面语义；`P`=历史非零摩擦数字直填。`P` 存在单位/语义问题，因此只作诊断。 |
| `SC` | 计划中的“共用拍面语义+正确标定摩擦” plant。必须先有物理潜变量模型和 PhysX/MuJoCo 独立 adapter，不能把 `SP` 改名当成 `SC`。 |
| `carry-state` | 下一拍直接继承上一拍结束的真实机器人状态，不 teleport、不 reset。 |
| `T0 / T1 / T2` | 连续恢复的三层实验：T0 只在完整周期结束后换题；T1 在任意允许时刻事件驱动揭题，但冻结 reward；T2 才增加 learned recovery shaping。三者是实验层级，不是三个 reward。 |
| `PhysicalBall Phase A / B` | 物理球仪器的实现层级，不是课程阶段：A 只让来球在引擎中飞行，禁用机器人碰撞且不施加拍面冲量；B 才加入受合同约束的球拍接触和碰后飞行。B 有源码材料不等于已有被接受的运行或训练成绩。 |
| `readiness critic / critic-gate q50` | readiness critic 是估计“当前状态能否及时接住下一题”的模型。它必须用独立训练/校准数据且不能偷看未揭题信息；critic-gate q50 是在正式 Gate3B 之前单独封存的一次性 50 题/侧诚实考试。 |
| `guard reset` | 判卷器因跟踪包络或其他保护条件提前结束，但未发生真实倾倒。“物理不摔”不能隐去 guard reset，更不能据此证明连续恢复。 |
| <a id="wave-cgf"></a>`Wave CGF` / 抖动-地面-脚部消融波（`p1cgf_*`） | 2026-07-22 预注册的 8 臂单变量波：action_rate 剂量三档（ar02/ar05/ar10）、机器人材质摩擦抬高（grip）、随机凹凸地形（rough，fresh 铁律）、mjlab 落地冲击罚（footrw）、软惩罚减负（penlight）、被动阻尼折 kd（kdpassive，未接线锁死）。父本只用 W，对照＝矩阵 `w_c_s0`；谱系 diagnostic-only。 |
| <a id="ground-plant"></a>`ground_plant` 合同块 | schema-3 合同的地面/地形 plant 指纹：地面材质摩擦、机器人材质随机化范围、材质是否逐桶强制 `dynamic≤static`，以及平地或每环境零均值地垫。默认配方＝整块缺席（历史 checkpoint 逐字节兼容）；任何偏离＝落键，resume 对账把它当另一套 plant 拒绝静默续训（平地 checkpoint 上不了粗糙地）。`robot_side_zero_mean_patch` 是桌侧严格平、机器人侧围绕 z=0 起伏的候选地垫；目前只有 E1 host 证据。 |
| `foot_soft_landing` / 落地冲击罚 | mjlab soft_landing 思想：first-contact 步的脚底法向峰值力超阈（默认 300 N）部分按阈值归一后惩罚，教"轻放脚别砸"。量纲要点：输出是无量纲超阈倍数（单脚封顶 3），不是牛顿——mjlab -1e-5/N 的等效剂量 = -3e-3。默认 weight=0 字节等价。 |
| `foot_clearance` / 抬脚高度罚 | mjlab foot_clearance 思想：腾空脚 \|脚高-目标高\| × 水平速度，罚"贴地扫着走"。给允许跨步的臂用；站立击球默认不开（weight=0）。 |
| `foot_slip_sq_weight` / `foot_drag_weight` | 触地脚水平蹭滑（源码常开 -1.0）与拖脚（-0.5）的剂量键，2026-07-22 新接 CLI（此前够不着）。penlight 减负臂降到 -0.33/-0.17。 |
| `lower_body_imitation_scale_in_window` | 击球窗内下肢模仿衰减系数（上半身 motion_scale_in_window 的下肢版，同一个 WIDE 窗）：触球一瞬让下肢模仿小声点。默认 1.0＝字节等价；台账/探针记未衰减原值供对账。 |
| `penlight` / 惩罚减负臂 | Franco 第 8 条的消融：六个软惩罚统一降约 1/3（脚朝向/挥拍前直立/拍面条件引导/挥拍前脚滑/触地蹭滑/拖脚），硬保护不动，看击球是否恢复。不是加大击球权重（击球组本来就远重于模仿）。 |
| `branch_dashboard` / 三人分支看板 | `scripts/branch_dashboard.py`：只读打印 Franco/jiayi(dongc1)/yikang(Catrunaround) 各自远端分支领先/落后 main 的提交数。纪律：领先的每个提交要么搬进 main、要么在 branch_fix_audit 文档记"不搬+原因"，不允许失踪。 |
| <a id="action-specific-dynamic-ready"></a>`action-specific dynamic ready` / 动作专属动态准备合同 | A3 的 physical spawn 与 teacher/reference 仍使用该动作 motion frame 0；控制器出生目标、`last_action` 和 fresh actor 输出则共同使用该动作在现役 plant 上通过 Isaac nominal-hold 的 `hold_qdes`。两者允许不同，但 action、motion bytes、31 关节顺序、candidate、PASS receipt 和 runtime decoder 必须一次性绑定，true reset 必须原子写入或整体回滚。它不是 failure-buffer reset，也不改变 ball-first 的冻结动作语义。 |
| <a id="diagnostic-joint-safety-drain"></a>`diagnostic joint-safety drain` / 诊断训练关节安全账本排空 | `training_authorized=false` 的 ActionBall 诊断跑仍执行真实 qdes 投影、substep q/qdot freshness、brake、actual-hard/nonfinite/table/fall 保护，但不取得 formal Reward 或 curriculum 晋级权。新 compact 形式在 device 上累计每 update 的 readback 完整性、逐关节 count/min-gap，PPO 前冻结验证、optimizer 成功后才 ack/clear；不复制逐 substep dense transcript、逐 step identity，也不写 formal per-update 收据。该排空只改变诊断证据粒度和热路径成本，不改变 Reward、Done 或策略样本；formal 路径仍保留完整收据。 |

## 动作库术语

| 术语 | 人话 |
| --- | --- |
| `NPZ` / 动作归档 | NumPy 压缩归档文件：本项目所有正式动作参考（source 与 compiler output）的落盘格式，必须满足 exact schema-2 的 11/14 字段（fps、31 关节位置/速度、32 body 世界位姿/速度等）。文件存在或可读不等于动作可执行、可训练或已授权。 |
| `MJCF` / MuJoCo 模型描述 | MuJoCo 的机器人/场景 XML 描述格式；本项目的唯一厂商真源是 `a3_pingpong.xml` 及其 mesh 闭包，用整文件 SHA 内容绑定。它决定 FK/碰撞/动力学口径，任何换文件或改字节都必须换身份。 |
| `URDF` / 机器人描述 | Unified Robot Description Format：厂商 A3 的关节/连杆/限位描述文件，是关节速度与限位的硬件真源（拍面安装变换也从它读取）。与 MJCF 一样按 SHA 内容绑定，不许静默替换。 |
| `GVHMR` | Global Video-based Human Motion Recovery：把单目人物视频恢复成 SMPL-X 人体动作的离线前处理器。结构输出通过只说明人体重建文件形状和有限数合法，不等于机器人动作、安全或击球有效。 |
| `GMR` | General Motion Retargeting：把 GVHMR 的人体动作重定向到 Agibot A3 关节/刚体。GVHMR 结果不会自动授权 GMR；每一代输入、body shape、源代码和输出都要另做内容绑定。 |
| `FK` / forward kinematics / 正向运动学 | 给定 floating root 和关节位置，用机器人模型计算每个 link/刚体的世界位置与姿态。本项目的离线 MuJoCo FK 不推进动力学时间，也不等于 simulator、碰撞安全或动作有效性通过。 |
| `TOPP` | Time-Optimal Path Parameterization：在不改几何路径的前提下，按速度、加速度等约束重新分配动作时间。它可以压缩过长 clip 或对齐阶段，但不会自动修正碰撞、平衡、拍面或击球点。 |
| `SMPL-X` | 带身体、手和姿态参数的人体模型表示；本项目把它作为视频动作与机器人重定向之间的中间制品，不把它当作 A3 runtime-order 动作。 |
| `PT` / PyTorch tensor archive | GVHMR/GMR 离线链使用的 PyTorch 序列化动作文件；文件存在或可加载只证明对应结构制品可读，不自动证明动作、schema-2、仿真、训练或硬件资格。 |
| <a id="motion-s0"></a>`S0`（static high-press batch） | 2026-07-13 新视频的单条高点拍压离线批，只处理 `static/pai.mp4`，不是第 0 个随机种子或训练阶段。2026-07-20 回收确认其 exact-GMR 输出 finite/30 Hz/31-DoF 结构通过，但 ball contact/effectiveness 仍为 null，formal/schema2/training/hardware 全未授权；下一门是独立高球拍压题族，不是直接训练。 |
| <a id="motion-m0"></a>`M0`（motion lateral-teacher batch） | 同一代新视频的四条横移老师离线结构批，按 left-1/left-2/right-1/right-2 顺序处理。四条是动作候选，不是四个随机 seed。2026-07-20 回收确认 exact-GMR 文件均存在且 finite/31-DoF 结构通过，但机器人 `stance_passed=0/4`，formal/schema2/training/hardware 全未授权；因此当前是 moving-teacher input-gate reject，不是可训练资产。2026-07-20 语义修订把这四条改称共享横移脚步模块（见 `configs/motion_role_catalog.json`），改名不改安全事实：`0/4` reject 仍然有效。 |
| `High press` / 高点拍压 | 右手机器人用反手在较高击球点迎球，球拍向前且拍面朝下，把球压回台内的独立动作类型。它不是被动挡球或反手拉球，必须使用自己的高球来球考卷。 |
| `Shared lateral footwork module` / 共享横移脚步模块（旧称 lateral locomotion teacher） | `motion/` 下四条 dang 素材的现行语义：只描述准备迈步、击球支撑和恢复三段的下半身/根节点参考动作，跨所有击球动作复用，只能由有效击球意图触发；不存在独立 locomotion 动作或独立 stop teacher。它不是正手或反手挥拍本身，和上半身动作组合后仍须重新过全身安全与动力学门；语义真源为 `configs/motion_role_catalog.json`，合同见 [stroke_footwork_composition](interfaces/stroke_footwork_composition.md)。 |
| `Non-striking arm` / 非击球臂 | 当前右手 A3 动作库中的左臂。取消它的模仿 Reward 只表示允许左臂帮助平衡，不会关闭关节、力矩、自碰或安全停机约束。 |
| `A0/A1 non-striking-arm pair` / A0/A1 非击球臂配对 | A0 是当前上半身模仿对照；A1 只从位置、姿态、线速度、角速度四条 body-imitation Reward 中删除左 shoulder/elbow/wrist，躯干、右击球臂、权重、题库、seed、预算和所有安全项不变。它不是恢复实验里的 A/B/C，也不是传感延迟 A1。 |
| `SE(2)` / 平面刚体变换 | 在水平面内只做一次整体偏航旋转和 XY 平移；本项目的动作站位实体化把同一个 proper transform 原子地作用于整条 floating-root 轨迹，禁止镜像、Z、尺度、逐帧、关节或时间编辑。 |
| <a id="projection-root-pin-artifact"></a>投影钉根伪影 / projection reset-root artifact | 把动作投影到"根钉死为 reset 单位姿态"约定时，源动作的骨盆旋转（反手 ~40-60° 转体）被整个删掉，世界拍面随之系统性转歪——可行性扫描会把好动作误判成"拍面差 44-57°、全帧 0%"的假死刑。2026-07-22 三路对抗复核在 yikang stationary-v2 反手判决上坐实此机制（符号翻号与题库复用被排除）；任何"投影后 0%"判决必须先做源几何 A/B 对照再定死刑。 |
| 全库可行性复扫 / library-wide phase rescan | 对注册表全部 `_cal` clip 的"源几何 × 投影后 × 速度档"扫描矩阵（见分支修复审计 07-22 的复扫方案）：源几何高分而投影后 0% ⇒ 投影伪影，修投影约定；两边都低 ⇒ 动作真不可行，换参考动作。每 clip 一行落账，防"感觉不行"式死刑。 |

## 部署与全链路术语

| 术语 | 人话 |
| --- | --- |
| `Gate3` | 上真机前的全链路彩排：在厂商 MuJoCo 中把 planner、真机同款 C++ runner、消息和机器人执行串起来，先看能否稳定走完。独立 BankExam 只测 policy 与自己的 Python 评估器/机器人动力学配方，不能替代 Gate3。 |
| `Gate3B` | Gate3 的回球评分版：用当前阶段来球分布，并正式记击球率/上台率。它比“能稳定跑完”更近真机质量门。 |
| `first tick` | vendor simulator、通信、planner 和 runner 真正启动后，第一个有效控制周期。只过源码检查或 model preflight 不算 first tick。 |
| `portable Release` | 在 Linux 上用优化编译、但明确关闭 ROS 2 与 AimRT backend 的 C++ source/binary gate。它能证明 exact 源码可编译、链接并通过 native suite，不会启动 transport、simulator 或硬件。 |
| `native suite` | 编进 C++ `run_tests` 可执行文件的测试集合。缺可选资产导致的 skip 必须单列，不能算 pass。 |
| `AimRT` | 厂商部署 runner 使用的 middleware/backend 路径。AimRT 关闭的 portable Release 明确弱于 AimRT-enabled Release、backend first tick 和 Gate3 行为。 |
| `Gate3-D0` | 本项目的“第 0 版最短部署仿真闭环”：固定同卷、planner + policy + C++ runner + vendor runtime 的单拍演示；不冒充连续对打。它是项目内部标签，不是行业通用术语。 |
| `Trainer-v0` | native MuJoCo 训练的首卷。因现役 vendor main sim loop 没有球/球台/网，目前只能练单拍平衡与击球状态，不是 physical return 结果，也不阻塞几天内 `Gate3-D0`。它是并行候选训练轨；产物若晋级，仍须独立通过 Gate3/Gate3B。旧草案曾把它也叫 `D0`，从现行文档起停止这种重名。 |
| `Recovery-D0` | recovery A/B/C 预注册的第 0 步：只用现有 179-D checkpoint 做 A bridge 与 C previous-tuple 的 zero-shot 诊断，不选型、不晋级。原 config 字段仍叫 `D0`，文档必须写全 `Recovery-D0`，避免与 `Gate3-D0` 混淆。 |
| `STAND GAIN SOURCES` 横幅 / `planner_static` 列 / `tau_*` 列 | 2026-07-22 部署侧取证三件：启动日志打印两条站立路径（人工 PD_STAND 的 --stand-kp/--stand-kd 与 planner static 的官方高增益表）各自 Kp/Kd 来源与数值；obs/trace CSV 尾部新增 planner static 闩锁列与 31 列实测力矩（vendor SDK 只暴露 effort，无电流/温度）；build git 指纹每次运行第一行打印。默认行为逐字节不变；`--planner-static-gain-scale`（默认 1.0）是唯一新旗标。**2026-07-25 更正**：07-25 之前 runner 的分组增益块会在 PpPolicy 之后再改 STATIC 命令（--gain-scale 0.4 的既往实跑里腰/臂实发 0.4x 官方；held/--leg-stand-gains 配置又把腿/腰覆盖回 1.0x，审计旗标在这些关节上失效）——横幅"STATIC 恒为官方表、仅审计旗标可降"在当时是**不实**的，读旧日志须以 trace CSV 的最终 kp/kd 为准。07-25 起增益块以 `!planner_static_active()` 守卫，STATIC 命令原样直达，横幅自此为真；此变更待 Linux 编译回归后才可上真机。 |

## 证据和文档术语

动作 registry 使用下列 E-level 时，必须同时满足
[`motion evidence certificate chain`](#motion-evidence-certificate-chain)；裸 `evidence_level` 字段
没有晋级效力。

| 术语 | 人话 |
| --- | --- |
| `E0` | 只有设计或预注册，没有实现证据。 |
| `E1` | 源码、单测或静态检查证据。 |
| `E2` | 真实 runtime smoke、真实模型装载或 first-component 运行证据。 |
| `E3` | 受控训练已实际运行。 |
| `E4` | 留出仿真考试、Gate3 或 Gate3B 证据。 |
| `E5` | 真机证据。 |
| 课程阶段 / `Stage` | 只回答“机器人正在多学会哪种球技”。现行顺序是：阶段 1 固定点；阶段 2 虚拟球变到达状态，站位和脚步属于其中的解法；阶段 3 物理球进场，旋转在后段加入。连续恢复和部署验证不占课程阶段编号。 |
| 连续能力线 | 每个课程阶段都要另考的横向能力：上一拍安全收尾、等待动作/姿态合格、下一拍随时可启动。`T0/T1/T2` 描述它的实验层级。 |
| 部署验证线 | 独立于课程阶段的验收顺序：独立 BankExam → `Gate3` 全链稳定 → `Gate3B` 回球质量 → G07 真机安全。 |
| `Gate` | 可验收的项目里程碑。`Done` 必须有可复现命令、验证结果、输入/输出和已知限制；只有材料或代码时写 `Partial`。 |
| `P0 / P1` | `NOW` 唯一队列中的相对优先级：P0 是当前最高层，P1 是下一层。它们不表示证据等级，也不得复制到实验状态。 |
| `red-team P1` | 代码复核中的高优先级正确性缺口，不是 `NOW` 队列的 P1。文档首次使用时应写全“red-team P1 正确性缺口”。 |
| 吞吐继续门 | 在启动长训前，用 N=1/8/32/64 并行环境实测 sim-only 和完整 rollout+一次 PPO update 的速度、内存与扩展效率；只有预计两臂×两 seed 能在 48 小时内完成且留 30% 余量，才继续 CPU-Python 路线。 |
| `人类责任人` / `执行者` | 责任人只能是人；Claude/Codex 只是执行工具或 provenance。不知道人类责任人时写 `UNASSIGNED`。 |

## 基线目标

`HITTER-compatible baseline` 指保留 HITTER 风格的分层：基于模型的 planner
输出击球位置、拍速和时刻，RL 全身控制器消费 planner 目标与机器人状态，
输出关节位置目标。它不表示照搬 HITTER 的具体实现；硬件、动捕、simulator、
部署 runtime 和改进策略都以 A3 项目真实约束为准。

## 坐标系与资产类型

- `world`：HOPE 球台/世界坐标，遵循 ROS REP 103：X 朝对手，Y 向左，Z 向上。
- `base_link`：机器人机身参考坐标，精确物理位置以机器人模型和 SDK 为准。
- `racket`：由机器人 FK 和固定球拍安装关系推导的坐标，不是 mocap 直接跟踪点。
- `ball`：来自 mocap 或未来感知系统的球位置。
- `Source`：代码、launch、config、脚本、消息定义和文档。
- `Curated data`：用于测试/示例的小型 bag、CSV 或图表。
- `Runtime asset`：模型权重、ONNX/RKNN/TensorRT engine、sysroot、预编译二进制和 vendor 包。
- `External reference`：供研究或可选实现参考的上游仓库。

坐标系的完整合同见 [frames_and_coordinates](interfaces/frames_and_coordinates.md)。
