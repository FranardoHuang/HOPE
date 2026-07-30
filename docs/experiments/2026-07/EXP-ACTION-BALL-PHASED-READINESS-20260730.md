# EXP-ACTION-BALL-PHASED-READINESS-20260730 — ActionBall 分阶段训练准备账本

- 状态：`running`
- 阶段/轴：AgiBot A3，动作条件 Ball-first，N=1 诊断长训 → 正式 N=5 → N=73 → 部署
- 集成小目标：先产出一份可迭代的 N=1 policy，同时把不应阻塞首跑、但必须在后续阶段关闭的工程债写清楚
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E2`
- 创建日期/最后复核日期：2026-07-30 / 2026-07-30

共享缩写按[术语与人话对照](../../DEFINITIONS.md)解释。本文是依赖门和技术债账本，
**不是新的全项目优先级队列**；运行顺序、算力认领和当前采用 setting 仍只认
[`origin/main` 的 `NOW`](../../NOW.md)。本文只回答一件事：某项工作最迟应在首个 N=1
长训、1000 update 检查、正式 N=5、N=73 或部署中的哪一个边界前闭合。

## 0. 当前执行看板（本文唯一活跃 TODO）

### 0.1 维护规则

- 本节是本文唯一可领取、可更新状态的 TODO；[`origin/main` 的 `NOW`](../../NOW.md)
  仍是项目唯一优先级和算力队列，两者不得互相替代。
- 每次代码、合同、Pod 验证、发射状态或外部输入发生变化时，先更新本节，再把不可替代的数字、
  SHA、失败原因和运行证据追加到后文附录。不要把事件时间线继续写进看板。
- 状态只用 `IN_PROGRESS`、`READY`、`BLOCKED`、`LATER`。完成项从本节移到
  [已完成证据附录](#2-已完成证据附录-a不参与调度)，不在看板长期堆积。
- `BLOCKED` 必须写清缺的输入；`IN_PROGRESS` 必须写清唯一下一动作和验收条件。任何后文边界、
  决策或历史记录若未在本节出现，都不是当前可领取任务。

### 0.2 Now — fresh fixed-194 N=1 先形成可训练 lineage

| ID | 状态 | 当前交付 / 唯一下一动作 | 完成验收 | 阻塞输入 | 证据入口 |
| --- | --- | --- | --- | --- | --- |
| N1-RUN | `BLOCKED` | exact recipe 与 fresh r4 三段 spec 已发布，smoke canonical plan/claim `257c6ccc…d80c` 已 PASS；唯一下一动作是等待 launcher watchdog 对 r3 做 exact-PG 自然收口与全 Pod Kit boot lock 释放，再 launch r4 `1 env × 2 updates` | fresh smoke 有真实 PPO update 与 finite checkpoint；teacher-start 首帧等于完整 wait、每 tick 减一个 `policy_dt` 且不为负；随后 probe 跨 `t_hit` 且无 NaN/identity 漂移/持续 table、fall、raw-hard 爆炸，通过即发 long | r3 PID `1127383` 与 wrapper 自然退出（不得手工 signal/kill）；全 Pod Kit boot lock 和自然空闲 GPU 槽 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[结果判读](../../operations/read_and_report_results.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |

### 0.3 Next — long 已运行后的判读与 formal N=5 前置

| ID | 状态 | 当前交付 / 触发条件 | 完成验收 | 阻塞输入 | 证据入口 |
| --- | --- | --- | --- | --- | --- |
| N1-REVIEW | `BLOCKED` | N1-RUN 的 long 启动后按 update 100/300/1000 检查 teacher fidelity、strike denominator、return/landing、Reward income、unsafe 与课程 frontier；不按一次均值临时改超参 | finite/exact identity 持续；动作与 teacher 偏差可解释；strike opportunity 非零；table/fall/hard 与 curriculum failure 分账；1000 结论进入附录 | N1 long 数据；必要时 Franco 对动作语义/视频的人工裁定 | [结果判读](../../operations/read_and_report_results.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| N5-INTENT | `LATER` | formal N=5 前把 v2 的 N1-only 无身份合同替换为**固定宽、内容生成的连续动作意图**；首版使用归一化 `q_ref_at_hit-q_ready` 与 `teacher_rate*qd_ref_at_hit`，共享 ready 有混叠才加中间相位 preview | actor/critic 同源；宽度不随 N 变化；跨 N tensor/order、动作间距离、混叠检查和 Pod 构造 parity 通过；禁止恢复 N 维 one-hot、UID 数值或 per-slot embedding | exact N=5 ordered motion/reference；N1 不被此项阻塞 | [ActionBall 合同](../../interfaces/action_conditioned_ball_first_contract.md)、[Policy 观测合同](../../interfaces/policy_observation_action.md) |
| N5-RECIPE-ID | `LATER` | formal N=5 前把 policy bootstrap identity 从绝对 checkout path 改为 candidate/hold 的 content/file SHA；路径只保留在 runtime hard contract | 同一 bytes 在不同 clean checkout 得到相同 policy SHA；任一 artifact bytes/binding 改变仍 fail-closed；旧 recipe 可审计但不静默重标 | 不阻塞当前 exact-path r4；需 recipe schema 迁移与负例 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| N5-RECEIPT | `LATER` | formal N=5 前把正式审计从 per-env reset 仪式改为 device 紧凑 event tape + checkpoint/hourly 批量物化；同时清理残余 D2H、ledger clone、broker per-env Python | 旧式完整 receipt 可逐字段重建；checkpoint save/load/no-step、跨进程 exact resume、篡改负例与固定工作量吞吐验收通过；proposal/reject/action/domain/lifecycle/outcome 分母不丢 | 分段 profiler 和可用 Pod | [formal 发射工序](../../operations/run_action_ball_curriculum_no_clobber.md)、[设计与加速审计](../../research/design_audit_and_speedup_20260729.md) |
| N5-TEARDOWN | `LATER` | formal N=5 前让 `gym.make()` 之后的全部 pre-run/hard-contract 异常也在 `finally` 中关闭 env，避免未关闭 PhysX/Gym env 让 Kit teardown 自旋到 watchdog | 注入 pre-run 异常时 env close 被调用一次、进程及时退出、GPU/全 Pod boot lock 自然释放；正常 recipe-only 与训练行为不变 | 不阻塞当前 r4；需 Pod failure-path smoke | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| N5-PHYSICS | `LATER` | formal landing 或 N=5 前显式选择 2026-07-30 OptiTrack ball-physics profile，重新物化 physics/solver/question bundle；zero-weight term 做结构裁剪，不能以“权重为零”冒充未解析 | physics/solver/question SHA 重钉；Isaac/solver 配置 parity；active Reward/term ledger 可复算；未充分辨识切向参数明确保持 prior/canary | 现有科学源可先工程化；最终 table/tangential 参数仍需 OptiTrack 补测 | [G05](../../gates/G05_isaac_training_first_loop.md)、[Reward 因果审计](../../operations/run_action_ball_reward_causal_prelaunch.md) |
| N5-LAUNCH | `BLOCKED` | 上述 N5-INTENT、N5-RECEIPT、N5-PHYSICS 通过后，绑定 exact ordered N=5 manifest、逐动作 admission/ball support/new-forehand 安全证据并 fresh 发射 | formal receipt 绑定 continuous intent、motion bytes/order、Reward/PPO/plant/solver/physics/table/evaluator 和 exact resume；clean/no-clobber lineage | Franco 确认 exact 五动作顺序；新正手采用/站位裁定 | [formal 发射工序](../../operations/run_action_ball_curriculum_no_clobber.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |

### 0.4 Later — N=73 与部署边界

| ID | 状态 | 最迟边界 / 交付 | 完成验收 | 必须由人或硬件提供 | 证据入口 |
| --- | --- | --- | --- | --- | --- |
| N73 | `LATER` | N=73 发射前完成 exact ordered 73 manifest、逐件 compiler/safety/admission、动作专属 ready/ball center/support、full-body 与 fixed-width continuous intent 的任意 N 压力门 | 73 件 order/UID/motion bytes 全钉；逐动作 frontier 与 heldout 独立；broker/curriculum/checkpoint/exact resume 压力通过；不得由 N5 续成 | 若 repo/Pod 制品不完整，需提供缺失 bytes、证书、站位元数据和人类采用顺序 | [动作库终审](EXP-MOTION-CANONICAL-LIBRARY-20260723.md)、[ActionBall 合同](../../interfaces/action_conditioned_ball_first_contract.md) |
| DEP-OBS | `LATER` | 部署意图重训前落地 OptiTrack pose + gyro angular velocity + causal base linear velocity producer；用新合同加入 localization age/valid 和实测 noise/latency/dropout | capture/source/receive/consume timestamp 链、marker→`base_link`、venue→table、gyro frame、marker→COM 修正、hold-last/stale 语义及 Isaac/MuJoCo/C++ parity 全过；不得伪装成 194-D | OptiTrack v2 记录、Motive smoothing、遮挡/dropout、两组 SE(3) 外参、gyro bias/对齐 | [Policy 观测合同](../../interfaces/policy_observation_action.md)、[G07](../../gates/G07_mujoco_to_real.md) |
| DEP-PHYSICS | `LATER` | 任何正式 landing/真机前闭合 table effective restitution 与切向参数，并让训练/planner/MuJoCo/C++ 共用同一冻结接触模型 | 新测量 receipt、拟合 YAML、跨引擎 golden parity 和 binary metadata 拒载负例齐全 | OptiTrack 30 分钟落球/反弹补测及 Motive 配置 | [G03](../../gates/G03_data_processing_and_physics_calibration.md)、[G06](../../gates/G06_isaac_to_mujoco.md) |
| DEP-CONTROL | `LATER` | 只有 1000-update 量尺证明 policy 独有 bang-bang 后才评估 executed-qdes penalty/CAPS/EMA/governor；真机前无论采用与否都闭合 A3 硬边界 | 干预率、teacher phase delay、strike/return、拍速、imitation 与 Isaac/MuJoCo/C++ 同合同；G06/G07 dry-run、急停和 no-publish 门通过 | A3 逐电机 command velocity/acceleration/jerk（若采用）、PD/effort/torque-speed/delay 与 Franco 真机放行 | [G06](../../gates/G06_isaac_to_mujoco.md)、[G07](../../gates/G07_mujoco_to_real.md) |

## 1. 第一性原理裁定

### 1.1 先产 policy，但不能在错误合同上产

首个 N=1 训练只需要证明：

1. 动作身份在一拍内冻结；
2. 来球先采样，再由同一动作的 fixed-action solver 生成相对 task；
3. 出生、初始 `qdes`、上一动作、actor 观测和 teacher frame 0 描述同一 ready；
4. episode 能活过该动作 `t_hit`，checkpoint finite，table/fall/nonfinite/raw-hard 分账可信；
5. 训练设置、policy recipe、spec 和结果记录可复算。

它不需要等待正式 N=5 的完整审计基础设施、最大课程 support、N=73 动作 admission、
MuJoCo/C++ deploy consumer、full-body 对照或最终 command governor。反过来，观测列错位、
task 变绝对坐标、桌碰传感器恒零、出生姿态与控制目标冲突会直接改变 policy 学到的问题，
这些不能用“先跑再说”绕过。

### 1.2 观测合同：相对 task + 绝对桌体上下文

正在运行的 compatibility 合同是
[`action_ball_table_pose_twist_heading_task_n<N>`](../../DEFINITIONS.md#action-ball-table-pose-twist-heading-task-contract)；
fresh 首选合同是
[`action_ball_table_pose_twist_heading_task_teacher_start_v2`](../../DEFINITIONS.md#action-ball-teacher-start-contract)。
N=1 fresh actor 宽度为 **194**：

| 区间 | 项 | 维度 | 语义 |
| --- | --- | ---: | --- |
| `[0,177)` | frame-correct HITTER-derived prefix | 177 | reference、机器人本体状态、上一动作；racket position residual、velocity 均为 yaw-heading frame |
| `[177,180)` | `base_position_table` | 3 | base 在桌面中心坐标系中的 XYZ |
| `[180,186)` | `base_orientation_table_6d` | 6 | base 相对桌体的完整 roll/pitch/yaw，连续 6D 旋转表示 |
| `[186,189)` | `base_lin_vel_heading` | 3 | base/root COM 在 base yaw-heading frame 的三轴线速度 |
| `[189,193)` | demanded face + reserved scalar | 4 | raw-A 有符号拍面 normal 在 yaw-heading frame；`rho` 不旋转 |
| `[193,194)` | `time_to_teacher_start_s` | 1 | 同一 Motion phase governor 中，老师离开 ready frame 前的剩余秒数 |

动作 UID/slot 仍在 sampler/solver/curriculum/receipt 中冻结，但不进入 policy。历史 N1
one-hot 恒为 `[1]`、严格不含信息；formal N5/N73 前用固定宽 continuous action intent 补充未来
动作内容，而不是恢复槽位标签。旧 `193+N` / `194+N` one-hot 合同只保持历史读取。三个角度
没有被删掉：6D
旋转用旋转矩阵前两列表示完整
SO(3)，经正交化后第三轴唯一确定；它不是 yaw-only。采用 6D 而不是 Euler 三角或裸 quaternion
是表示合同修复，不是 Reward 假设，不做学习 A/B。依据是 Zhou 等人在 CVPR 2019 的
[连续旋转表示结果](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)。

task 位置继续保持机器人相对坐标：

- `base_target_pos_b` 是当前 base 到 base goal 的相对二维残差；
- `racket_target_pos_b` 是当前球拍 FK 到目标击球点的相对三维残差；
- 桌体 9 值回答“机器人相对球桌站在哪里、朝向如何”，不把 task 改回 world absolute。

同一个 actor 内的击球 task 不再混 frame：

```text
p_task_h = R_heading^T (p_target_table - p_racket_FK_table)
v_task_h = R_heading^T v_target_table
n_rawA_h = R_heading^T n_rawA_table
```

旧 `action_ball_table_pose_twist_n<N>` 同为 194-D，但 velocity/normal 仍在 world frame，
只保留兼容读取；同宽不代表同合同，旧 checkpoint 不得 resume 到新名称。planner wire、
fixed-action solver、Reward 和物理真值仍留在 canonical table/world frame，转换只发生在 actor
观测边界。这是确定性坐标合同修复，不做学习 A/B，只做 Pod tensor/构造 parity。

这种拆分同时满足泛化和可观测性：相同相对 task 可以在不同站位复用，而 policy 仍知道自己与桌边、
桌面的绝对几何关系。

### 1.3 传感器权威

部署侧按物理量拆分，不做“整套 OptiTrack 或整套 IMU”二选一：

| 量 | 维度 | 首选权威 | 原因 |
| --- | ---: | --- | --- |
| table-relative base position | 3 | 360 Hz OptiTrack rigid-body pose | 无积分漂移，直接给外部位置 |
| table-relative base orientation | 6 | OptiTrack orientation 经 marker→`base_link` 与 table SE(3) 标定 | 直接给全局姿态和 yaw，无 IMU yaw 漂移 |
| `projected_gravity` | 3 | 同一 OptiTrack orientation 派生 | 与绝对姿态使用同一权威，避免静默混源 |
| `base_ang_vel` | 3 | pelvis IMU 三轴陀螺仪 | 直接测角速度；mocap 姿态差分会放大噪声并引入滤波延迟 |
| `base_lin_vel_heading` | 3 | 因果状态估计器：OptiTrack 位置作无漂移锚，可融合 IMU 加速度 | 单帧 mocap 不直接给速度，纯差分噪声大；纯 IMU 积分会漂移；最终旋入与 task 相同的 yaw-heading frame |

所以 `base_ang_vel` 是 **3 值**，`base_lin_vel_heading` 也需要 **3 值**。线速度在当前无 history 的
feed-forward actor 中能区分“同一姿态、正在倒/正在平移”和“静止”，因此值得进入 actor；
但部署 producer 必须处理 marker 到 root/COM 的位移以及旋转刚体的
`omega × offset`，不能把 marker 速度冒充 COM 速度。

训练中这些量使用 simulator rigid-body truth 的同语义 counterpart。真实 producer 尚未闭合
marker rotational extrinsic、时间同步、丢帧/延迟和 causal velocity estimator，所以当前 v2
训练合同目前不授权真机。

2026-07-30 对近期分支的复核给出三条新的硬边界：

- 近期 OptiTrack producer 不在过期的 `origin/jiayi`，而在 `origin/hitter` /
  `origin/hitterobs`；当前 live P1 rigid-body 是 `360 Hz`、米制、完整 6-DoF；
- 当前消费的 `NamedPoseArray v1` 只带 ROS 到达时间，不能分解相机曝光、Motive 解算和网络延迟；
  驱动已有携带 camera mid-exposure 与 Motive latency 的 v2 消息，部署意图标定必须切到该
  timestamp 路径；
- `origin/hitterobs` 已有“只在收到新 pose 时按真实 `delta_t` 更新、EMA、age/stale、reset
  原子清状态”的二维线速度生产模式。ActionBall 应复用该**因果更新规则**并扩成三维
  root-COM heading velocity；重复的 held pose 绝不能被解释为零速度。

首个 194-D N1 运行保持其 exact 历史；fresh fixed-194 v2 用 teacher-start 时钟替换常量
one-hot，但仍不在没有
真实 dropout 分布时追加常量 freshness 列。最终
部署意图 actor 应在同一次自然断 warm-start 的版本迁移中增加至少
`base_localization_age_s(1) + base_localization_valid(1)`：短 dropout 时 policy 能区分
“新测量”和“hold-last”，超过 supervisor stale 门则直接安全停机。若只增加这两列，N=1
宽度将由 fresh 194 变为 196，必须使用新 contract 名并重钉 Python/C++/MuJoCo producer；
不能在 194-D 或 195-D 名称下静默追加。是否再加短 history 仍由实测延迟/别名 canary 决定。

旧青瞳（ChingMu）场地 profile 只可作数量级先验，不能直接冒充 OptiTrack 标定：旧位置模型是
`1.9 mm` white、`5.2 mm` AR(1) marginal、`rho=0.717`/50-Hz policy tick；旧场地记录还把
`40 ms` 作为保守整条 target 传输上界，而另一份实测说明传输 `<10 ms`、端到端 `<=20 ms`。
复核还确认这些数主要来自旧链的**球轨迹**，不是新 OptiTrack A3 base rigid-body；Jiayi V15
中的 `[3,3,5] mm`、`20 ms`、`1% dropout`、EMA `alpha=0.25` 也明确只是 guessed prior。
这些数字来自不同设备、对象和处理链。当前 N1 首波不把它们分别撒到 6-D rotation、
projected gravity 和 task residual 上；部署意图重训前必须重新记录 OptiTrack capture/source/
receive/consume timestamp、Motive smoothing、遮挡、gyro bias 和 marker→COM 线速度残差，并在
同一个因果 SE(3)+twist packet 上注入 episode extrinsic bias、tangent-space orientation noise、
colored position noise 与 hold-last latency。

### 1.4 近期球物理与 73 动作证据如何消费

`origin/main@ddfaaa02` 已加入 2026-07-30 OptiTrack 球物理：较可信的拟合量包括
`k_d=0.1253`、`k_m=0.00404`、球拍有效恢复系数 `0.646`，桌面恢复系数采用裁定值
`0.9215`；paddle speed decay、table/paddle tangential retention 仍欠充分辨识，只能作为
canary/prior。科学源 `ddfaaa02` 已以 `bed6661f` 进入当前分支，但正在运行的 exact
`f2c54fc3` N1 source 与已物化 bundle 仍绑定旧 physics profile，**不得声称该 policy 已经
使用新物理**。

首个 contact-only N1 的判据只是 task 击球位置/时间窗是否可达，新参数不阻塞 1000-update
diagnostic；任何正式 landing/return 结论、formal N=5 或部署意图训练前，必须显式合入新物理、
重新物化 physics/solver/question bundle 并重钉 SHA。参数改变不是 Reward A/B，但新旧 policy
表现也不能混成同一 lineage。

ChingMu-73 已不是“只有动作、来球要反推”的原料：exact N=73 manifest 已逐动作保存真实
`v_in/v_out`、击球点、station、yaw、contact phase 与 motion bytes。73 条 station 覆盖约
`0.260 m x 0.479 m`，yaw 约 `-4.76 deg .. 37.83 deg`。这进一步支持当前拆分：动作/task
保持 heading-relative 以复用，table-relative base pose 9 值保留绝对站位与三轴姿态；
每动作用 manifest 的真实来球作 ball center，而不是强行共享一个 spawn。

### 1.5 joint 和 task 信息是否够

fresh fixed-194 v2 合同中已经有：

- 31 个 joint position（相对 default）；
- 31 个 joint velocity；
- 31 个上一 policy action；
- 三轴 base angular velocity、三轴 base linear velocity、projected gravity；
- motion anchor orientation、62-D teacher command；
- 相对 base goal、相对 racket target、目标拍速、time-to-strike、teacher-start 倒计时；
- demanded signed face、完整 table-relative base pose。冻结 action identity 只在控制面，不在
  actor observation。

这里的 `HITTER` 是 177-D prefix 的来源；仓库没有一份独立、可替代上述布局的 “SMASH actor
contract”。SMASH 相关论文/实现可提供 reward、噪声或训练方法启发，但不能凭名字推断另一组列。
这也符合两篇论文的系统边界：
[HITTER](https://arxiv.org/abs/2508.21043)由上层 planner 生成击球位置、速度和时序，再让
whole-body controller 执行；[SMASH](https://arxiv.org/abs/2604.01158)把 onboard egocentric
perception 和 scalable whole-body skill 作为自己的系统组成。后者的感知输入不能在没有
frame、latency、deploy producer 和 checkpoint metadata 的情况下直接拼进 HITTER-derived prefix。
当前不再为首个 N=1 添加 raw ball state、角加速度、接触历史或 observation history：
fixed-action solver 已把每颗来球编译成当前相对 task；在真实 PPO 证明存在不可观测别名以前，
继续加列只会扩大接口和部署债。

单帧不是永远冻结的结论。若 OptiTrack/IMU 实测证明延迟或接触冲量导致同一当前帧对应不同最优
动作，先做关键通道 `0 vs 4/8` 帧的小预算 canary；ring buffer 必须进入 partial-reset 与
exact-resume state。它不是“纯拼列、无状态”的免费修改，GRU/蒸馏排在 feed-forward stack
之后，也不阻塞首个 N1 policy。

### 1.6 bang-bang：先量化，再加约束

现役合同已有 raw/processed action 平滑项、实际 joint acceleration/limit 账、
finite qdes projection、投影前超出量 penalty 和 actual soft/hard 分账。首个 N=1 long
**不再临时提高动作变化惩罚，也不加入 EMA、velocity/acceleration/jerk governor**，原因是：

1. 快速击球依赖短 contact window 内的高拍速；未按 A3 电机和 teacher 导数标定的硬限速会直接
   延迟 `t_hit` 或削掉拍速；
2. 更强软 penalty 可能压过模仿/击球收入，却不能给硬安全保证；
3. stateful governor 会改变 action→executed command 合同；若 actor 不看 executed state，
   同一个 actor action 会对应不同后继状态，制造新的部分可观测性；
4. 当前最大问题仍是先取得 healthy strike denominator，而不是凭视频印象调平滑权重。

1000 update 检查必须新增或确认以下量尺：逐关节 executed-command 换向率、归一化一阶/二阶差分、
torque saturation、qdes projection/saturation、实际关节 hard/soft 驻留、击球窗拍速和 table/fall。
只有这些量证明 bang-bang 持续且与参考无关，才按以下次序开小预算 canary：

1. 首选低剂量、按每个 A3 电机能力归一化的 executed-`qdes` 一阶/二阶差分 penalty；
2. 若仍有 policy 独有的持续高频抖动，再比较 CAPS temporal regularization；
3. EMA 只作分部位诊断，必须量相位延迟和拍速损失；
4. 最后才评估逐电机 velocity/acceleration governor；击球手臂与腿/腰使用不同限制，actor
   同时加入 executed `q_cmd`/`qd_cmd`，若限制 jerk 再加 `qdd_cmd`，并要求 Isaac、MuJoCo、
   C++ 逐字同合同。

每项都同时比较 intervention rate、teacher phase delay、strike/return、racket speed 和
imitation，不以总 Reward 单独晋级。当前两个 dynamic-ready recipe 的 raw actor bias 约落在
`[-5.87, 16.30]`，所以不能套用 `[-1,1]` clip；那会让部分动作专属 ready 本身不可达。

EMA 只可作为诊断对照，不是最终安全边界；Ruckig/jerk 级在线轨迹器是部署候选，不阻塞首个
policy。[Ruckig](https://arxiv.org/abs/2105.04830)确实能对多自由度轨迹施加 velocity、
acceleration 和 jerk 三阶约束，但这只证明该类 governor 有严格约束实现，不证明未经 A3
参数标定的 governor 对快速击球无损。

## 2. 已完成证据（附录 A；不参与调度）

本节只保存不可替代的运行数字、SHA、失败根因和完成边界。新工作不得从这里领取；状态变化先更新
[当前执行看板](#0-当前执行看板本文唯一活跃-todo)，完成后再把证据移入本节。

| 项 | 证据 | 当前边界 |
| --- | --- | --- |
| fixed-194 v2 policy recipe / r4 specs | Pod1 recipe-only 真实构造物化 policy contract `165645f5…bd9`，tracked recipe raw SHA `4b81c74b…7fb1`；旧/新 recipe 的 PPO 字段相同，变化来自 dynamic-ready 绝对 artifact/receipt path 与派生 binding SHA。fresh r4 smoke/probe/milestone1000 spec raw SHA 依次为 `6fc4e7ca…c369`、`6e7caeb1…b200`、`533b50d2…36ea`；smoke canonical plan PASS，claim `257c6ccc…d80c` | r4 namespace 全新；在 r3 exact-PG watchdog 收口与全 Pod Kit boot lock 释放前不启动 |
| fixed-194 v2 r3 构造失败 | Pod1 GPU2 exact source `8729104e` 已真实构造并验证 `action_ball_table_pose_twist_heading_task_teacher_start_v2 (194D)` 与 dynamic-ready bootstrap；随后在 PPO runner 创建前 fail-closed：spec 沿用 policy recipe `b7209710…077f`，post-compose 实际为 `165645f5…bd9`。spent namespace=`n1hr_smoke_fastball110_8729104e_block_gpu2_seed0_r3`，无 PPO update/checkpoint | 证明 observation/scene/ready 构造可达，不证明学习；r3 不复用，必须重新物化 policy recipe 并发 fresh r4 |
| fixed-194 v2 fresh r3 specs / claim | smoke / probe / milestone1000 三份 canonical spec 均绑定 exact source `8729104e6c9a…46c4`、fast-ball bundle `3c1076e3…c32b`、Pod1 GPU2 UUID、seed0 与 fixed `current_low` Reward；raw JSON SHA 依次为 `e1b63f00…5b8d`、`3b200542…dd34`、`b0396fbe…d442`，namespace 全部 fresh `_r3`；smoke canonical plan PASS，claim `7f9d12ca…4002` | 第一版缩进 JSON 在创建 namespace 前被 canonical-byte 门拒绝，已机械规范化且不覆盖旧 operator-control 目录；pin/spec/claim 已闭合，下一步是 launch smoke |
| fixed-194 v2 profile / question repin | Pod1 exact source `17c7258a` 物化 profile pins `08c8f9c7…c6b4`、base bundle `ed9fa0f7…afef` 与 1.1 倍 fast-ball bundle `3c1076e3…c32b`；solver profile 为 `52777b36…9754`，physics profile 仍为旧诊断值 `aa5c9085…f85b7`。derivative 的 4096-proposal tape 保持 `2763/4096=67.46%` admitted 与逐原因拒绝分账 | 工件内容已闭合，但在其 commit A 与 fresh spec/claim 生成前仍不可发；这是 diagnostic comparison，不是 formal 95% admission 或新 OptiTrack physics 证据 |
| fresh fixed-194 v2 source + focused suite | commits `291bc20e` / `0227cfe9` 已让当前 ActionBall trainer 只实例化固定 194-D v2，删除 `policy.action_one_hot`，N>1 fail-closed，并对 exact 17-term layout、旧同宽重标、teacher-start lazy bind 加回归；Pod1 exact `0227cfe9` focused suite 为 **391 passed, 12 skipped in 61.35 s** | dependency/contract 测试已闭合；真实 ObservationManager、PPO update 与 finite checkpoint 仍由 fresh `1 env × 2 updates` smoke 验证 |
| dynamic-ready 出生与 hold | loop/block 在 Pod 各闭环保持 `0.8 s / 40` policy steps，双脚接触率 `1.0`，table/fall/hard/nonfinite 零；可见 raw-reset 图证明原生 reset 直立 | 证明出生与 nominal hold，不证明 teacher 全轨、strike 或 long |
| dynamic-ready trainer 接线 | candidate/hold receipt 双 pin；physical state、初始 qdes、last action、actor bias 和 motion frame 0 原子一致；Pod focused `63 passed` | 当前只授权 exact N=1 diagnostic |
| 170-update overflow | diagnostic joint-safety summary 已改为每 update 按事务排空 | 旧 overflow checkpoint 不续；fresh run 重新开始 |
| qdes 安全语义 | finite request 投影到合法包络，penalty 读取投影前超出量；nonfinite/actual hard 仍终止 | Reward 剂量仍需 healthy baseline 后 canary |
| reset 诊断热路径 | diagnostic per-reset 逐 env 完整转录已移出；不可变 receipt SHA、strike timing、metrics D2H 等已做确定性优化 | formal 路径仍保留完整 per-reset 仪式；每步 ledger/broker 税仍开放 |
| 194-D actor source | `203b2d92` 实现 table pose + base twist；Pod dependency-light 合同回归 `290 passed, 9 skipped, 1 deselected` | C++/ONNX/MuJoCo producer 未实现，旧 182/191-D checkpoint 不可复用；policy recipe 与 observation 分层，现有 dynamic-ready recipe 可复用 |
| table contact source | Pod1 `eb2799b1` r26：五件 kinematic collider、32-body×5-role matrix、四子步与五 role 真实正控均通过；五 probe reset 后零泄漏，unsupported/Traceback/FAIL 均为 0，出现 `main_completed` | E2 receipt 已保存；训练仍逐 run 监测 table counter |
| 旧 194-D loop/block 构造 probe | exact `c9682591` 各 `4096×5` 均持续产 checkpoint 且 finite；mean episode 已到约 29–32 步，birth age≤1 hard 为零 | 第一次 PPO 前 update0 已有 `860/864` 个 env 撞 `waist_roll` raw hard；两动作 hard-env Jaccard `0.982`，不能据此发 1000 |
| shared waist 根因定位 | qdes forbidden 始终 0；teacher waist/hard 最小余量 `0.272–0.303 rad`；hard 首发约 step18/0.365s，且两动作 substep hard Jaccard `0.992` | 指向共享 plant DR，不指向 solver/Reward/teacher 贴限；下一轮使用 stable-ready plant |
| frame-consistent stable-ready smoke | exact `f2c54fc3` 的 loop/block `1 env×2` 均验证 194-D actor，四份 checkpoint finite，qdes/actual-hard/table/fall/nonfinite 全零；log SHA 为 `db75ef49…49b` / `0a26cbee…69f` | 证明真实构造和初始 plant，不判断学习 |
| frame-consistent stable-ready probe | loop/block `4096×5` 各有五份 finite checkpoint；update 0 全安全，mean episode 随后约 `48–72` steps 并跨各自 `t_hit`；loop 在 update2/3 有 `783/84` 个 strike opportunity，block 在 update2/4 有 `1838/430` 个 | 第一次 PPO 后共同出现 waist-roll/pitch actual-hard，qdes forbidden 始终 0；五轮只证明窗口可达，不判恢复上限 |
| 历史恢复反例 | 旧 exact `4ff48b21` 在 update 1–5 同样有大量 reset；loop/block 的 actual-hard terminal 到 update100 已降到 `14/11`，update169 均为 `3` | 支持按预注册运行到 100/300/1000 再判，不支持把旧 policy 当新合同结果 |
| 三条 milestone1000 | Pod1 GPU1 的反手挡 seed0 已越过 update 300 且 `model_300.pt` 的 80 个 tensor 全 finite；GPU0 fresh 反手拉 seed0（claim `3c523fde…0196`）与 GPU2 fresh 反手挡 seed1（claim `7ac32418…e3f`）均已进入真实 PPO update。三条 exact source 都是 `f2c54fc3`、`4096×1001`、194-D、stable-ready plant | GPU1 已跨 `t_hit` 但 virtual capture/return 仍为 `0/0`，table/fall 与劣质动作局部解仍开放；GPU0/GPU2 提供动作差异与 seed 复现，不把早期波动写成晋级。旧 GPU0/GPU2 `4ff48b21` overflow 进程已按 exact PGID 响应 TERM 正常退出，日志/checkpoint 均保留 |
| 2026-07-30 19:39 CST live 快照 | loop seed0 / block seed0 / block seed1 分别到 update `219 / 574 / 186`，mean episode `104.88 / 481.52 / 105.90`，三条均持续写真实 PPO update、无新 Traceback；当前 update 的 strike opportunity 为 `945 / 965 / 962` | 三条 virtual capture/return 仍全为 `0/0`。block seed0 当前 table/fall/actual-hard=`2/3/6`，但 965 个 contact proposal 全被 face gate 拒绝；loop seed0 post-strike fall=`887/946`；block seed1 table/actual-hard=`590/325`。说明窗口与 denominator 已通，但动作质量、signed-face/contact 对齐和 seed 稳定性仍未闭合，不得写成可部署 policy |
| 2026-07-30 19:50 CST block seed0 update 600 里程碑 | exact PID/PGID=`1055080`、cwd=`n1dr_10069d3c/.../whole_body_tracking`、GPU UUID=`GPU-a8f7…e6cb6`、source/claim/namespace 均与发射绑定一致；update608 持续训练，`model_600.pt` 为 7,197,343 bytes、SHA=`11bee491…8470f`，80 个 tensor 全 finite。mean episode=`440.77`、iteration=`20.94 s`、strike opportunity=`951`，table/fall/actual-hard/qdes-forbidden=`4/5/14/0` | 仍是零 capture/return；`937/951` 被 face gate 拒绝，exact strike position/velocity/normal error=`0.2426 m / 1.3928 m/s / 86.31°`，实际/目标拍速=`0.2832/1.2793 m/s`。安全与 episode 长度已恢复，当前主要问题是未学到 teacher 的击球位置、拍速和拍面，不是出生或 qdes reset storm |
| 显式 teacher-start source contract | 历史 `194+N`/N1 195-D source 已通过 exact `020dc8d9` Pod focused suite `390 passed, 9 skipped`，但未进入 PPO；当前 fresh N1 改为固定 194-D v2，用真实 teacher-start scalar 替换常量 one-hot | 这是接口正确性直接修，不做学习 A/B。旧 194-D/195-D run 保持 exact 历史；v2 仍须真实 ObservationManager `1×2` 构造与 finite checkpoint |
| 反手挡 1.1 倍中心来球公式带 | 固定 `bh_block` 动作身份、原落点中心 `2.555 m`、初始速度双侧宽度 `0.15 m/s` 和现有 solver/physics/prototype，在 Pod1 GPU2 对 4096 个确定性 proposal 只把中心来球从 `4.2376948` 提到 `4.6614643 m/s`。同一逐球 solver 得到 target site speed 均值 `1.14907 m/s`、teacher-rate 均值/中位 `0.72055/0.71595`；`2763/4096=67.46%` admitted，拒绝严格分账为 `resid_gt_tol=1327`、`teacher_rate_below_min=6`，proposal tape SHA 为 `0335220d…2581` | 这证明“更快来球但仍落同一台面中心”会让挡球 task 卸力并降低主动老师拍速，不需要另写反向 solver。它只是 GPU2 的机制诊断：solver rejection 会把实际训练题条件化，且 admission 未达 formal `95%` 门，所以不能称严格单变量 A/B，也不可作为 curriculum 晋级或 N5 证据；训练 sampler 必须保留全部 proposal 分母与拒绝原因 |

当前 source 证据随本次集成提交进入 `main` 后才生效；旧 `f2c54fc3` 三条运行始终保持其原
commit、194-D observation 和旧 physics 身份，不因文档或 main 前移而重标。

上表的 1.1 倍比较也澄清了 ball→task 变量边界。每个环境先冻结动作，再采来球、到球时间、
击球点、base 与 landing aim；solver 随后保持该动作的拍速方向，逐球求拍速大小、有符号拍面和
精确击球位置。**landing aim 是 solver 输入，不是 solver 自己选择的输出**。所以来球变快后，
若仍要求落在原台面中心，公式会主动卸力；target 本身仍由同一公式逐球计算，训练中没有
selector 或换动作。若未来希望自然借力挡得更深，才另把 landing aim 作为显式课程轴，而不在
本比较里混入。

第一次 plan 正确拒绝了旧 r9 bundle：最新 source 的 `hope_commands.py` blob SHA 已从
`0e650b…` 变为 `e24190…`，而 derivative 仍引用旧 profile pins。随后在 Pod 以 exact source
blob map 重物化 profile pins `9ccb9854…5788`、current-source base bundle
`0daa5bce…ace53` 和 1.1 倍 bundle `f2be2331…1a491`；physics SHA 保持
`aa5c9085…f85b7`，solver profile 因实现 source pin 更新为 `bf255a78…f26e`。
这次 repin 不改变来球、落点、动作或 Reward 数值。三份 canonical spec 现绑定 exact source
`77f01deb`、fast-ball bundle `f2be2331…1a491`、seed `0` 和 Pod1 GPU2；仍按
1 env×2 update → 4096 env×5 update → 4096 env×1001 update 串行。每一阶段自然完成、
checkpoint finite 且无 fatal 后才发下一阶段；三段都从新随机初始化开始，不把前段 checkpoint
当作续训，也不覆盖旧 GPU2 seed1 目录。

source `77f01deb` 的 canonical plan 已通过并产生 claim `13dc15a2…8e86f`。第一次 smoke
随后在 namespace `n1hr_smoke_fastball110_77f01deb_block_gpu2_seed0_r1` 创建 simulator 前
fail-loud：stable-ready N=1 guard 读取了不存在的 `racket_cfg.clip_names`，而训练翻译层真实安装的
字段一直是 `clip_names_per_clip`，因此把合法的 `("bh_block",)` 误读为空。该 guard 是
`1d4b8a11` 后新增，旧 `f2c54fc3` long 尚未经过它；这解释了旧波能跑而新 source 失败。修复只把
guard 改读 canonical 字段并在拒绝信息打印 mode/diagnostic/action tuple，不改训练数值。失败
namespace 永久保留，修复提交后用 fresh r2 spec/source/namespace 重发。
fresh r2 的 smoke/probe/milestone specs 现绑定 source `319ae8ff`，分别使用
`n1hr_smoke/probe/milestone1000_fastball110_319ae8ff_block_gpu2_seed0_r2` namespace；未发射的
旧 r1 probe/milestone spec 已删除，失败 smoke r1 spec 留作 exact 证据。

## 3. 分阶段最迟闭合项（附录 B；边界参考，不参与调度）

本节保存各阶段的完整门槛语义，避免看板为容纳细节而再次变成长流水账。所有当前动作均须在
[当前执行看板](#0-当前执行看板本文唯一活跃-todo)有且只有一行；本节未出现在看板中的条目
不得被解释为当前算力或实现队列。

### 3.1 首个 N=1 long 前（历史 194-D 波已完成；fresh fixed-194 v2 仍缺 Pod smoke）

直接修/验证，完成即开跑，不等待后续工程完美化：

1. table-contact Pod smoke 已在 `eb2799b1` 完成：五个 role 均有真实正控，32 个 matrix
   shape/order 正确，`robot_hit_table` 会触发，reset/settle 后零泄漏，日志无 unsupported
   filtered target；receipt 见
   [`table_smoke_eb2799b1_gpu1_r26.receipt.json`](../../../configs/n1_contact_dynamic_ready_20260730/table_smoke_eb2799b1_gpu1_r26.receipt.json)；
2. 本轮已先把 actor 迁移到 frame-consistent **194-D**
   `action_ball_table_pose_twist_heading_task_n1`；随后采用的 fresh successor 是同宽但新语义的
   fixed-194 `action_ball_table_pose_twist_heading_task_teacher_start_v2`，用老师开始倒计时
   替换 N1 常量 one-hot。
   recipe 只绑定 PPO/decoder/ready，observation name/width/term order 由训练 hard contract
   单独绑定，所以无需为该一维迁移重物化 recipe；旧 182/191-D、旧混合-frame 194-D 与当前
   compatibility 194-D checkpoint 一律不因同宽续成 v2；
3. 两动作各跑 `1 env × 2 updates`，确认真实 PPO update、finite checkpoint、q/qdes/last-action/
   ready 一致；
4. 旧 full-DR probe 已证明 shared waist-roll raw-hard 爆炸。fresh launcher 先选
   [`stable-ready plant`](../../DEFINITIONS.md#stable-ready-plant)：保留历史 robot-material DR
   与 policy recipe 已钉住的 joint-default `±0.01 rad`，关闭直接改变弱腰平衡的 torso CoM、
   link-mass 与 PD-gain DR；两动作各跑 `4096 env × 5 updates`，记录每 update 秒数、episode
   是否跨 `t_hit`、strike opportunity、raw-hard/table/fall/nonfinite 和 checkpoint；
5. 只要一条动作过构造门且没有硬错误，立即发该动作 fresh `1001 updates`（0 起数，确保生成
   `model_1000.pt`）；另一动作可串行排在同一空闲槽，不因它阻塞先过门者；
6. 把 exact spec、recipe、run record 和复现命令纳入 repo；所有 namespace fresh/no-clobber。
7. 下一条 fresh launch 使用 fixed-194 v2 teacher-start 合同；先在 Pod 做 focused tests 与
   `1 env × 2 updates`，验证 formal reset 首帧等于完整 `pre_swing_wait_s`、下一 tick 精确减一
   个 `policy_dt`、到零后不为负。该验证不倒改已经运行的 194-D 波。

这批 N1 的物理身份是当前 bundle 已钉住的旧 profile，只授权 contact/学习可行性诊断；不得在
结果中写成 2026-07-30 OptiTrack 新球物理或部署候选。

下列项**不阻塞**首个 diagnostic N=1 long：

- formal per-reset receipt checkpoint compaction；
- generic formal N5 launcher；
- full-body、Reward、reference guard、课程 failure target、entropy/sigma/RSI 消融；
- 8192 env；
- MuJoCo/C++/真机 fixed-194 v2 producer；
- EMA 或 command governor。

stable-ready plant 不是最终 sim-to-real 配方。它只把“先让 policy 学到动作”与“证明 ready
覆盖完整 DR support”拆开；robot-material DR 从未关闭，1000 后只按 base-CoM → PD → mass
的具名阶段重新加回，
每阶段先出 robust-hold/teacher-to-`t_hit` 证据再做短 canary，禁止一次性恢复整包后靠 Reward
解释 hard-limit。

### 3.2 1000 update 检查

千轮不是任意调参点。按以下顺序判读：

1. checkpoint 是否持续 finite、exact identity 是否未漂移、有无 Traceback/summary overflow；
2. 动作是否仍像 teacher：逐动作看 motion/pose error、击球窗相位、关节抽动、table/fall/hard；
3. 是否产生足够 strike opportunity；五轮没有 strike 不能判学习失败，1000 轮仍无 strike 才进入
   根因分支；
4. 长期 strike/return/landing 与四组 realized Reward income；
5. 课程 proposed/admitted/rejected、safe denominator 和每 arm/side frontier 是否开始推进；
6. bang-bang 量尺是否显示持续的 executed-command 高频换向或饱和。

如果 dynamic-ready/episode 已健康跨 `t_hit`，但 200–1000 update 仍无 strike，下一嫌疑是
official low-gain waist plant 下 teacher 动态不可跟踪；先比较部署一致的腰 gain 或 reference
retiming，不回头反复改出生，也不先放宽 hard limit。

只有 activation denominator 非零后，才允许小预算 canary：

- tracking 权重/Reward 剂量；
- reference guard 的 metrics-only、phase-gated 或连续约束方案；
- qdes projection penalty 剂量；
- curriculum safe-policy failure 目标（默认约 10%，20% 仅对照）；
- upper/full-body；
- entropy/sigma/RSI；
- bang-bang governor/EMA。

这些是经验选择，必须 canary；数学等价、合同修复和删除重复同步不做科学 A/B，但仍做 Pod
parity/吞吐验证。

### 3.3 正式 N=5 前

以下可以晚于首个 N=1 policy，但**不得晚于 formal N=5**：

1. **正式收据粒度改造。** 目前只有 diagnostic 快路径；formal 仍在每个 env-reset 做完整
   Python transcript/receipt 仪式。改为热路径紧凑 device event/assignment tape，
   checkpoint/hourly 批量物化完整 receipt；保留 proposal 分母、reject reason、action/domain/
   birth/sample/task/lifecycle/outcome，不接受仅 `seed+config`；
2. **重建与 exact resume。** fixed tape 下从紧凑日志重建旧式完整 receipt，逐字段 parity；
   checkpoint save/load/no-step roundtrip、篡改负例、segment head 和外部 resume pin 全过；
3. **分段 profiler 与 all-mode 热路径。** 分开 physics rollout、metrics/D2H、termination、
   safety archive、birth/retire、sampling/solver、state write、PPO；据实修每步 ledger clone/
   string 重建、残余 D2H、broker per-env Python。`~7 ms/env-reset` 只作为混合上界，未经 profiler
   不写成精确归因；
4. **固定宽连续动作意图。** fresh N1 已从 actor 删除常量 one-hot；formal N5/N73 不能直接复用
   无 future intent 的 v2，也不得恢复 N 维 `action_one_hot`。用动作内容生成固定宽 descriptor，
   首版至少包含归一化 `(q_ref_at_hit-q_ready)` 与
   `teacher_rate*qd_ref_at_hit`，同时提供给 actor/critic；对 shared-ready 动作做两两混叠检查，
   必要时再加一帧中间相位 preview。action UID/slot 继续只在 sampler/solver/curriculum 内冻结和
   审计。该接口修复不做“one-hot 是否好学”的科学 A/B，但必须做 tensor/order、跨 N 固定宽、
   动作间距离和 Pod 构造 parity；
5. **generic formal launcher 余项。** launcher 改认固定宽连续动作意图合同，不让 policy input
   随 N 改宽；仍须把 dynamic-ready、table-pose-twist producer identity、真实
   Reward/PPO/plant/solver SHA 和 exact resume 全部绑定进 formal receipt；
6. exact ordered N=5 manifest、五件 motion bytes/admission、动作专属 ball center/support、
   new-forehand `t_hit/t_cycle/site speed/table clearance/recovery` 和非空 trust set；
7. formal Reward causal receipt、frozen evaluator canary/heldout、table smoke、stage evidence 和
   checkpoint identity 全闭合；
8. 运行规模压力证据只按 N=5 的 N/E/R 合同给 N=5，不让 N=73 缺口反向阻塞。
9. 已把 `origin/main@ddfaaa02` 以 `bed6661f` 合入，并在第一次 byte pin 前修正两份 YAML
   的旧曲线示例注释；formal N5 仍必须显式选择该 OptiTrack profile，重新物化并重钉
   physics/solver/question bundle。未充分辨识的切向参数保留 canary 身份，不静默写成
   verified constant。

正式路径的 per-reset receipt 改造是**有意留到这里而不是漏忘**：它不改变 N=1 学习问题，
却决定 N=5/N=73 是否能在短 episode 下接近正常吞吐和可用 checkpoint 大小。该改造不做学习
A/B；验收是 numerical/state parity、旧收据重建、exact resume 和固定工作量 Pod 吞吐。

### 3.4 N=73 前

1. exact ordered 73 manifest 与每件 motion/compiler/safety/admission 证书逐项闭合；
2. 每动作自己的 ready、`t_hit/t_cycle/site speed`、ball center/support 和 solver profile，
   不把 N=5 的 shared-ready 假设扩散到 73；
3. 直接消费 exact manifest 已保存的逐动作真实 `v_in/v_out`、击球点、station 与 yaw 作为中心；
   不再反推或把 73 件归一到同一绝对 spawn；
4. full-body actor/reference 与固定宽 continuous action-intent 的任意 N 路径通过；不得重新用
   one-hot、UID 数值或每动作 learned slot embedding 绑定动作库大小；
5. N=73 对应的 sampler/pool/broker/curriculum/checkpoint 压力、compaction 和 exact resume；
6. 逐动作独立 frontier、强制覆盖/starvation、center/interior/frontier 混合和 frozen heldout；
7. 若启用 base move，先证明真实 spawn 与 no-move/move goal 语义、preparation window 和移动恢复；
8. N=73 仍在 Pod 独立 smoke/canary，不能由 N=5 run 续成或更换 action order。

### 3.5 部署前

1. C++/ONNX/MuJoCo 全部支持最终版本化 ordered terms、normalization、metadata 和 stale/dropout
   语义；若采用 `localization_age/valid`，必须使用新名称/宽度，不能伪装成 194-D 或 195-D；
2. OptiTrack marker cluster→`base_link`、venue→table 的完整 SE(3) 外参、时间同步、Motive
   smoothing/端到端 latency 和遮挡处理进入部署合同；时间链使用 v2 capture/Motive timestamp，
   不再只看 ROS 到达时间；
3. 三轴 gyro 与 OptiTrack pose 的 frame/extrinsic 对齐；base linear velocity estimator 完成
   mocap anchor、可选 accelerometer 融合、marker→COM 刚体速度修正；
4. 用真实噪声、延迟和 dropout 重新做 Isaac/MuJoCo observation parity；不能把 simulator truth
   直接当部署证据；
5. 新 OptiTrack 球物理在 Isaac/MuJoCo/solver 中同值，question bundle 已按新 identity
   重物化；未充分辨识的切向参数不冒充部署硬真值；
6. 若采用 governor，Isaac/MuJoCo/C++ 使用同一逐电机 velocity/acceleration/jerk 参数、executed
   state observation 和冲突处理；未采用时也必须保留 hard q/effort/velocity 安全边界；
7. 同一 checkpoint 在 MuJoCo 验证动作方向、teacher fidelity、table/ball/racket 物理和无桌碰；
8. G06/G07 dry-run、joint order、scale、stop 和 no-publish safety gate 通过前不得接真机。

### 3.6 Codex 无法自行生成、必须由人或硬件提供的输入

这些不是代码待办；在对应输入到位前只能保持 `OPEN/Partial`，不能用 simulator truth 或旧青瞳
数字代填：

| 最迟边界 | 必须提供的输入 | 为什么必须来自人/硬件 | Codex 收到后能做什么 |
| --- | --- | --- | --- |
| 1000-update 判读 | 对 loop/block teacher 动作是否“语义上像预期反手拉/挡”的人工视频裁定；如不接受，指出具体关节/相位 | 数值 imitation error 不能定义人的动作语义与真机可接受观感 | 绑定视频/checkpoint，定位 phase/joint/reward 冲突并设计最小 canary |
| formal N5 前 | 对 exact 五动作 ordered manifest 的最终人工确认，尤其新正手是否取代旧正手以及站位版本 | 动作集合是产品/运动学选择，不能由训练分数静默改写 | 重钉 manifest、action UID/order、motion/admission 与 fresh launch receipt |
| formal landing / N5 前 | OptiTrack 30 分钟落球/反弹补测及 Motive 配置，裁定 table effective restitution 和未充分辨识的切向参数 | 当前 tangential retention/paddle decay 仍只是 prior；软件不能从缺失实验恢复真值 | 重新拟合 physics YAML，生成 Isaac/MuJoCo golden parity 与新 bundle |
| 部署意图重训前 | OptiTrack v2 的 capture/source/receive/consume timestamps、Motive smoothing 档位、遮挡/dropout 记录、marker cluster→`base_link` 与 venue→table SE(3) 标定 | 延迟、外参和噪声是现场测量，不可由仓库推断 | 完成 196-D N1 freshness successor、因果 SE(3)+twist producer、噪声/延迟注入与 C++ parity |
| command governor canary / 部署前 | A3 逐电机可部署的 command velocity/acceleration（若要求再给 jerk）、实际 PD/effort/torque-speed/delay 约束及厂商确认 | 未标定硬 governor 可能削掉击球拍速或制造 sim-to-real gap | 生成同一 Isaac/MuJoCo/C++ governor、executed-command observation 与干预率验收 |
| N=73 / full-body 前 | exact ordered 73 件与 full-body motion 的缺失 bytes、逐件证书、站位元数据和人类采用顺序（仅当 repo/Pod 现有制品不完整） | Agent 不能臆造缺失动作资产，也不能替人决定动作优先级 | 做 inventory/admission、任意 N 压力门、schema-v2 ready 和 fixed-width intent smoke/probe |
| canary 采用时 | Reward/curriculum/full-body 等经验 canary 的最终采用裁定与 GPU/停跑预算 | 代码可出证据，但产品权衡与算力授权归 Franco | 冻结 chosen recipe，更新 NOW/Gate 并发下一阶段 |
| 真机前 | 人类对 G06/G07 dry-run、急停链、场地净空和发布权限的明确放行 | 涉及真实机器人和现场安全，Agent 无权自授权 | 只在门通过后执行被批准的 no-publish/有限发布步骤 |

当前 merge、Pod fixed-194 v2 构造验证、旧三条运行守护、正式收据热路径重构、zero-weight term 结构裁剪、
exact-resume parity 与 N5 launcher 工程本身**不需要新的人工信息**；只需要可用的 Pod 槽和现有
仓库/制品访问。

## 4. 决策账本（附录 C；采用/拒绝依据，不参与调度）

本表说明为什么采用、推迟或拒绝某项设计，不承担 TODO 状态。当前状态只维护在
[当前执行看板](#0-当前执行看板本文唯一活跃-todo)。

| 项 | 分类 | 最迟边界 | 是否需要学习 A/B | 当前决定 |
| --- | --- | --- | --- | --- |
| 6D table-relative orientation | 直接修 | N=1 long | 否；做 tensor/recipe parity | 已实现，保留完整三角信息 |
| base linear velocity 3-D | 直接修 | N=1 long | 否；做 observation parity | 已进入 194-D actor |
| racket task velocity/normal 统一 heading frame | 直接修 | N=1 long | 否；做 tensor/构造 parity | 新合同名；旧同宽 194-D 不续 |
| OptiTrack pose + gyro angular velocity | deploy contract | 部署 | 否；做传感器延迟/噪声实测 | 采用分量级最优源，不整套弃用 IMU |
| OptiTrack v2 timestamp + localization age/valid | deploy contract | 部署意图重训前 | 否；做 producer/tensor parity | 当前 194-D 不加猜测常量；实测后以新合同迁移，长 stale 由 supervisor 停机 |
| teacher 开始倒计时 | 直接修 / fresh observation contract | 下一条 fresh launch | 否；做 tensor/reset/Pod 构造 parity | fresh N1 固定 194-D v2 用 `time_to_teacher_start_s` 替换常量 one-hot；getter 在 ObservationManager shape probe 先走既有 lazy runtime bind，reset 后仍读取 receipt 真值。当前三条旧 194-D N1 不停机、不重标、不 exact resume |
| 动作身份 actor 表示 | 直接修接口 / N5 前 | formal N5 | 不做 one-hot 学习 A/B；做 descriptor 混叠与 Pod parity | fresh N1 已删除 policy one-hot；UID/slot 仍冻结 sampler/solver/curriculum。formal actor/critic 改吃由 reference 内容生成的固定宽 contact intent，必要时加中间相位 preview。禁止 N 维 one-hot、UID 数值或 per-slot learned embedding 作为 arbitrary-N 正式接口 |
| 2026-07-30 OptiTrack 球物理 | identity/physics 直接修 | formal N=5 或任何正式 landing 结论前 | 否；重新物化和 Pod parity | 科学源已合入；当前 N1 bundle 仍是旧 profile，formal N5 前必须切换并重 pin |
| ChingMu ball/base 噪声直接复用 | 暂不采用 | 永不作为 OptiTrack 硬合同 | 否 | 只作数量级先验；新系统按对象、时间戳与 Motive 设置重测 |
| dynamic-ready 原子合同 | 直接修 | N=1 long | 否 | 已实现；现有 policy recipe 已复用，194-D hard-contract/smoke/probe 已过 |
| stable-ready plant（关 CoM/mass/PD DR） | 直接修后 Pod probe | N=1 long | 否；旧 full-DR 已给失败反例 | 防 shared waist raw-hard；1000 后逐轴恢复 DR |
| table-contact filtered truth | 直接修 | N=1 long | 否 | Pod r26 E2 已过；训练继续逐 run 分账 |
| diagnostic receipt 快路径 | 直接修 | N=1 long | 否 | 已实现 |
| zero-weight RewardTerm 结构裁剪 | 数学等价直接修 | formal N=5 | 否；做 composed-config/Pod parity | N1 不删 table/contact sensors；formal ledger 仅遍历 active 非零列 |
| qdes weight-independent probe | 安全验收 | formal N=5；部署前冻结复测 | 否；冻结同 policy/task tape | 当前 penalty 权重归零会连遥测一起消失；需独立零值 probe 和 global residence `<0.005` 候选门 |
| 单一 physics/contact 真源 | 跨语言合同修复 | formal N=5 | 否；做 Python/MuJoCo/planner golden parity | 采纳 Jiayi 的模式，不搬其过期硬编码常量 |
| 部署 binary metadata 拒载 | deploy contract | 部署前 | 否；negative tests | 为 ActionBall 最终 obs/checkpoint/motion/physics 生成，不复用 V17 的 180-D 常量 |
| formal checkpoint 粒度 receipt | 直接修 | formal N=5 | 否；做 parity/吞吐 | 开放，不能拖到 N=73 |
| ledger/D2H/broker 热路径 | profiler 后直接修 | formal N=5 | 否；做固定工作量吞吐 | 开放 |
| Reward 权重与负项剂量 | canary | 1000 update 后 | 是 | 不阻塞首跑 |
| reference guard/CaT | canary | healthy baseline 后、formal N5 采用前 | 是 | 继续分账，不热改 |
| curriculum 10%/20% | canary | healthy baseline 后 | 是 | 10% 默认，20% 只作对照 |
| full-body | canary | N=73 前 | 是 | 不阻塞 upper N=1；现有 full bundle 仍是 schema-v1，须先完成 stable-full ready→core→ready、nominal hold、schema-v2 bundle/solver preflight 与 fresh fixed-width successor smoke/probe，不能冒充当前 upper 的可比对照 |
| EMA | canary | bang-bang 量尺异常后 | 是 | 仅诊断，不作最终安全边界 |
| executed-qdes 归一化 penalty / CAPS | canary | 1000 update 后且 bang-bang 量尺异常 | 是 | 优先于 EMA/governor，仍须验收击球相位与拍速 |
| velocity/acceleration governor | canary + deploy parity | 部署前；可在 1000 后试 | 是 | 不阻塞首个 N=1；需逐电机标定和 executed-command observation |
| jerk/Ruckig | 暂不修 | 部署后续候选 | 是 | 当前收益证据不足，接口成本高 |
| 8192 env | 性能 canary | formal N5 吞吐健康后 | 是 | 不用更多 env 掩盖热路径问题 |
| OptiTrack/IMU 噪声与延迟 | 实测后直接建模 | 部署意图重训前 | 不用猜值 A/B；估计器方案可 canary | 旧青瞳数字只作量级先验 |
| 关键通道 observation history | canary | 实测显示单帧别名后 | 是 | 首个 N1 不加；若做须绑定 reset/exact-resume state |

## 5. 收口判据（附录 D；验收参考）

### 可发下一条 fresh 194-D v2 N=1

- table smoke receipt 可信；
- 当前 dynamic-ready recipe 的 SHA 已验证，历史 194-D hard contract 已由 smoke 实例化；
  fresh fixed-194 v2 teacher-start 合同须另做 Pod 构造 smoke；
- `1 env×2` 与 `4096×5` 有真实 PPO update、finite checkpoint；
- stable-ready plant 下 episode 能跨动作 `t_hit`，没有 NaN/identity 漂移/持续
  table/fall/raw-hard 爆炸；
- exact spec、recipe、run record 已保存。

### 可从 1000 update 转 reviewed long

- checkpoint 和训练连续性正常；
- teacher imitation/strike denominator 可解释；
- unsafe 与 Reward income 分组未失效；
- 若 strike 仍为零，teacher dynamic feasibility 根因已判明，而不是靠临时改 Reward 掩盖。

### 可发 formal N=5

- formal receipt checkpoint compaction、重建、exact resume 与 fixed-workload throughput 通过；
- fixed-width continuous action-intent launcher、exact N=5 action/admission/ball/support、
  Reward/table/evaluator 全闭合；
- 使用新的 clean/no-clobber lineage，不把 N=1 diagnostic 结果冒充 formal 证据。

## 6. 证据入口（附录 E）

- 当前训练 Gate：[G05 Isaac training first loop](../../gates/G05_isaac_training_first_loop.md)
- ActionBall 语义：[按动作条件化 Ball-first 合同](../../interfaces/action_conditioned_ball_first_contract.md)
- actor/传感器接口：[Policy observation/action](../../interfaces/policy_observation_action.md)
- N=1 发射步骤：[消融与 dynamic-ready 发射工序](../../operations/run_ablation_wave_launch.md)
- formal N=5 工序：[no-clobber ActionBall 发射](../../operations/run_action_ball_curriculum_no_clobber.md)
- table truth：[ActionBall 桌体安全 smoke](../../operations/run_action_ball_table_safety_smoke.md)
- Reward truth：[ActionBall 发射前 Reward 因果审计](../../operations/run_action_ball_reward_causal_prelaunch.md)
- 设计和吞吐审计：[N1 设计背书审计与训练加速尽调](../../research/design_audit_and_speedup_20260729.md)

本文不自行声明 Gate `Done`，不更新 `NOW`，也不授权真机。
