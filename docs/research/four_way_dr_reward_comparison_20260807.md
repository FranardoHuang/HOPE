# 四方对照:智元 / build_1 / EXP 账本 / 现役字节(2026-08-07)

**这份表回答什么**:随机化(DR)、reward、终止、探索四组设置,在四个参照系里分别是什么数,
以及哪些地方三层(机制码 / 实验史裁定 / 现役 argv)对不上。数值来自已有尽调与账本,本文
**没有跑任何新实验、没有改任何一行运行时代码**。

## 四列分别是什么(以及能信到什么程度)

| 列 | 全称 | 性质 | 证据强度 |
|---|---|---|---|
| **智元** | `Instinct-Parkour-Target-Amp-A3-v0`(MuJoCo 栈,AMP 系,`a3_ultra` 29dof) | 同底盘、**不同任务**(parkour)的训练配方 | **二手摘要**(Franco 提供),无法克隆核验;不是 resolved config |
| **build_1** | jiayi 那条线的 `HitterPingPong` 臂 | **唯一已知能真打到球的同底盘配方**,场景里没有桌子 | 仓内同族 YAML(`cfg/task/HOPEPingPongHitter.yaml`)+ 账本对其运行曲线的取证 |
| **exp** | [`EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802`](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) §6 / §6.1 / §5.6 | 我们**自己写下的裁定**:首轮该开什么、什么时候恢复 | 文档裁定,不等于字节 |
| **现役** | A211/C211 四格 `DR-L0` / `DR-L0N` 真实发射字节 | 现在 pod 上真正在跑的 | argv 取自 pod1 落盘 `launch_claim.json`;配置取自 `cfg/task/*.yaml` + `train.py` |

**一句话总览**:智元那列是"同底盘、别的任务、别的 reward 经济";build_1 是"同任务、能打到球、
随机化其实开得比我们多";exp 是"先全关、逐轴 fresh 恢复";现役是"exp 说的全关**已经全关**,
而且比 exp 写的还关得更死(课程也几乎是恒等变换)"。

---

## 一、随机化(DR)逐轴

图例:`ON` = 真在跑;`OFF(机制在)` = 代码齐全但被配置关掉;`—` = 无机制。

| 轴 | 智元 | build_1 | exp 裁定 | **现役(A211/C211)** |
|---|---|---|---|---|
| **PD 增益 Kp/Kd** | `ON` startup 抽一次;Kp log-u `(0.8,1.2)`、Kd `(0.7,1.3)`(阻尼比刚度更宽);play 关 | `ON` 同款 `(0.8,1.2)`/`(0.7,1.3)`,startup | 同底盘**终态 baseline**;首波显式关 | `OFF(机制在)` —— `stable_ready_plant=true` 一个布尔同时关掉 PD + CoM + link mass 三条 |
| **link 质量** | `ON` **±20%**,只随机躯干/踝/腕 + pseudo-inertia | `ON` 全身 **±15%** | 测量值优先于随意 `±20%`;末端/拍子优先 | `OFF(机制在)` 全身 `(0.85,1.15)` + `recompute_inertia` |
| **CoM** | `ON` **全 body** xyz 各 `±0.02` | `ON` 仅 torso | `CANDIDATE`,须过惯量一致性/hit/safety 门 | `OFF(机制在)` 仅 `torso_link`,x `±0.025` / y,z `±0.05` |
| **摩擦** | `ON` static `(0.2,1.8)` / dynamic `(0.2,1.5)` | `ON` static `(0.3,1.6)` / dynamic `(0.3,1.2)` / restitution `(0,0.5)` | `DEFER TO MUJOCO CALIBRATION`(不把 PhysX 数字直搬 MuJoCo `frictionloss`) | `OFF(机制在)` `startup_physics_material=false` |
| **关节零点偏移** | `ON` `±0.01 rad`,双写 action offset | `ON` 同款 | DR-L1 恢复 | `OFF(机制在)` `startup_joint_default_pos=false` |
| **执行器延迟** | `ON` **`[0,2]` 控制步**,每 episode 抽一次、集内固定 | `OFF` `d=0` | 首轮 `d=0`;`DELAY-L1/L2` 各自 fresh lineage,`d=2` 前先补 history/alias | `OFF` `min=max=0`(父本 `A3VendorV1.yaml:27-28` 挂着厂商 `[0,2]`,被叶子清零) |
| **六轴推撞** | `ON` vx/vy `±0.25`、vz `±0.1`、r/p `±0.26`、yaw `±0.39`;interval `(1,3) s`〔见勘误 1〕 | `OFF`(该谱系历史默认无推) | 首轮关;未来幅值沿用同底盘,**cadence 放慢到 `10..30 s`** | `OFF` 叶子把八个字段逐个写 `null` + argv `task.push.enable=false`。**幅值原样躺在 `HOPEPingPongActionBallA3VendorV1.yaml:43-56`** |
| **本体感 obs 噪声** | `ON` ang_vel `±0.2`(**scale 0.25**)、gravity `±0.05`、joint_pos `±0.01`、joint_vel `±0.5`(**scale 0.05**),**全组 history=8** | `ON` 同区间(类默认 `enable_corruption=True`),**无 scale、无 history 堆叠** | DR-L0 首轮全关;nominal 学习门后单轴 fresh 恢复 | **四格第二轴**:A0/C0 = 关,A1/C1 = 开。三通道 `base_ang_vel ±0.2` / `joint_pos ±0.01` / `joint_vel ±0.5`,**无 scale、无 history** |
| **task/racket/time 噪声** | (不适用) | 机制全在但全零(`target_delay_steps=0`、白噪/AR1 = 0) | 会降低任务可观测性,恢复得**更晚** | `OFF` 继承来的十条全部清零 + `action_ball_target_observation_noise=false` |
| **reset 噪声(位姿/速度/关节)** | `ON` 位姿 x/y/yaw `±0.1`;速度六维 `±0.2`;关节 `±0.15`;关节速度 0 | `ON` 走 RSI 分支时带位姿/速度/关节噪声 | 首轮全零;终态三档到 `±0.1` / `±0.2` / `±0.15 rad` | `OFF` 六轴全零,`canonical_ready_mode` 验证器**强制**如此 |
| **地形** | `ON` Perlin heightfield + terrain_levels 课程 | `OFF` 平地 | **REJECT**(parkour 专属,任务不对齐) | `OFF` 缺键 = 平地。机制在 `terrain_patch.py`,没有任何 cfg 设置它 |
| **motion 速度缩放** | (不适用) | — | `ADOPT` `[1,1]`,禁止整条 clip 拉到最高速 | `[1.0,1.0]`;等价物是收据 `teacher_rate`,manifest 预算 `0.6..1.01`,**活值单点 `0.85135`** |
| **视觉/深度 DR** | `ON` 相机外参 `±0.01 m`/`±2°`/`±4°`、深度 artifact 三档、帧延迟 0-1 | — | 现阶段不适用 | — 无相机 obs |
| **Play/评测口径** | DR 全关但 **obs 噪声保留** | (未记) | 两种口径要在 judge 文档里明示 | `eval_deterministic` `noise_scales=0.0`(评测不带噪声) |

**这张表的一句话**:**现役是四列里随机化最少的一列**。DR-L0 是有意的归因对照(先问"标称机器能不能学会"),
但它同时意味着"我们比那条已经能打到球的配方(build_1)开得还少"。

---

## 二、起点 / 等待 / 课程(决定"题目有多难"的那半)

| 轴 | 智元 | build_1 | exp 裁定 | **现役** |
|---|---|---|---|---|
| **reset 起点分支** | (parkour 走 terrain_levels) | 三分支:站立 / 随挥回放 / RSI。仓内同族 YAML 是 `stand .25` + `post_swing .25`〔见勘误 2〕 | 无档可挂 | **100% 站立**。`stand_start_prob=1.0`,`commands.py:2079-2082` **硬性拒绝**任何别的值 |
| **两拍之间的 hold** | — | `hold_steps_range [50,200]` + `stand_start_min_hold 25` ⇒ `1.0..4.0 s` | 归收据所有 | 三个时钟:隐藏 `RESET_WAIT` `5..25` tick(`0.10..0.50 s`)+ 收据 `pre_swing_wait_s` 活值 `0.712 s`;遗留 `hold_steps_range=(0,0)`。**实际间隔 `0.81..1.21 s`,下限运行时断言非零** |
| **出生站位泛化(步法)** | — | 站位由 station 采样 + `base_position` reward 驱动 | 走 success-gated 课程,不走挂钟 ramp | **两套互不知情的机制**:`start_pose_ramp`(x `[-1,0]`、y `±1.2625`、yaw `±30°`,挂钟 `96000` 步 = 4000 次 PPO update)挂在**从未发射过的 DR-L1** 上;课程侧 `base_spawn` 上限是 `0`。四格实跑 `start_pose_ramp=None` |
| **来球题目分布(32 条曲线臂)** | — | uniform box 采样 | 冻结 ball-first 规则扩张,要**可逆回退** | **30/32 条臂的 `*_std_max` 就是 `0`**:即使课程拉满 `L=1.0`,只有 `time_to_contact` 能动 `±0.125 s`。manifest 顶层 `mobility_mode: "no_move"`。课程**只升不降**(`grep demote\|rollback\|shrink` 零命中) |
| **探索 σ(`init_noise_std`)** | (未知) | **`1.0`**(肩 pitch 1σ ≈ `21.5°`) | 曾裁 `0.1`(用掉 4σ 门 59% 余量) | **`1.0`** + 标准 rsl_rl 初始化 + scalar,4σ 硬内带门显式跳过(四格 matched,不是差异轴) |

---

## 三、Reward:主任务层

**先立换算警告**:智元是 **AMP 系**——平滑/自然度由 discriminator style 收入承担,显式正则只是补丁。
它的权重活在**另一套收入经济**里,**可迁移的是项的形状与相对结构,不是绝对权重**。智元的
discriminator/task 公式、是否乘 `dt`、normalization 全部未知,所以"我们的 reward scale 是否已与智元对齐"
这一格只能写 **未测**。

| 层 | 智元 | build_1 | exp 裁定 | **现役 A211** |
|---|---|---|---|---|
| 结构 | AMP style + task,**无显式模仿六项** | `reward_pack=v1`,上半身模仿 + 高权目标项 | `模仿 < 击球引导 < 上台`,按同一折扣、同一分母的**预算**比,不是比单步峰值 | `reward_pack=v2`,全身模仿 `motion_scale 0.15` + 三层拍子核 + 上台大奖 |
| 拍位 | — | `14.0` / std `0.15` | 三层(coarse/fine/precision)`ADOPT`,但"三层"≠"首跑必须自适应" | coarse `11.5`/std `0.20`;fine `4.6`/std `0.50`;precision `0.575`/std `0.075` |
| 拍速 | — | `14.0` / std `0.6` | 同上 | coarse `11.5`/std `1.50`;fine `0.575`/std `3.0`;precision `0.2875`/std `0.50` |
| 拍面 | — | `5.0` / std `0.30` | 同上 | coarse `5.75`/std `1.0`;fine `0.575`/std `2.10`;precision `0.575`/std `0.262` |
| σ 自适应 | — | 明确关(为可复现) | 机制具备,首波固定 | **关**(`adaptive_sigma=false`),`static_rollout0_widths` |
| 站位/步法 | — | `base_position 1.5` / std `0.20` | 需要 footwork 时必须恢复 | **`0.0`**(定题从 no-move 出生,付它等于白送钱) |
| 上台/结果 | — | 无(该谱系没有 landing 项) | 上台是最终 truth anchor,必须压过前两层 | `virtual_landing 700`(post-dt 上台 `8.4..14.0`) |
| 拍子模仿(全程低权) | — | — | 实测拍子 teacher 全相位低权 | `motion_racket_{pos,vel,normal} 0.20` / `long_axis 0.10` |
| 接近引导 | — | `racket_progress` | — | `racket_progress 10`(telescoping) |

---

## 四、Reward:正则与安全(这一格最能直接对齐)

| 项 | 智元(raw) | build_1 | exp 裁定 | **现役** |
|---|---|---|---|---|
| `action_rate` | `-1e-3`(AMP 经济,**勿抄**) | `-0.10` | 保留封顶版 | `action_rate_l2 = 0`,`action_rate_clamped = -0.2` |
| `action_acc` | — | — | — | `0.0`(`A3VendorV1` 显式压包) |
| 关节限位 | `dof_pos_limits -2.0` | `joint_limit -10` | 双通道 barrier,带宽必须 `0.05` | **2026-08-07 裁定二后**:`qdes_limit_barrier -10` / `joint_limit -10`,两条带宽都 `0.02`(改一边开机即拒);核已换成上游 BeyondMimic = IsaacLab `joint_pos_limits` 的 rad 口径线性尾巴(无上界、软带内连续、地板挪到机械硬限位)。**旧 `-5` 是"每关节归一 [0,1]"口径,与这里的 `-10` 不可比**;详见 [exp §5.6.24](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md) |
| 力矩 | `torque_limits -0.01`(>90% 限值才罚) | `joint_torques -3e-5`;`arm_torque_saturation 0` | **形状值得借**,但先解决隐式 PD 力矩可观测性 | `joint_torques -3e-5`;饱和项强制清零 |
| 自碰 | `self_collision -0.1`,**子步整数计数 [0,4]**,力阈 10 | `undesired_contacts -0.1`(力阈 1.0) | 计数代替力幅的鲁棒性技巧值得借 | `undesired_contacts -0.1`。**08-05 修 bug**:排除正则 `_link → _Link`(A3 是 `_Link`,原写法是 G1 命名 ⇒ 双脚双腕反被罚 `-2.0/episode`) |
| 直立 | `flat_orientation torso -0.6` + `pelvis -3.0`(**拆两 body**) | — | 辅助项收入不得倒置主层级 | `upright(税型) = 0`;`upright_exp(收入型) = 0.25`(原 `1.0`,折扣 `+1.99` > mimic 预算 `1.77` ⇒ "站着不动比学动作挣得多") |
| 击球窗稳定 | — | — | 同上 | `hit_unstable_support -1.0`(原 `-10`,最坏 `-2.2` > accepted window `+1.85`) |
| **死亡罚** | **无**(终止的代价就是失去未来收入) | **无** | 降到三库量级 post-dt ≈ `-0.2` | `death_penalty -10.0` ⇒ post-dt `-0.2` = 合法上台下界的 `6%`(原 `-300` ⇒ `-6.0` = 下界的 `180%`,"打成一次再摔"净亏) |
| 腰/髋偏离 | `joint_deviation_hip -0.01`、`torso -0.05` | `prestrike_waist_twist -1.0` | 起步用 `-0.01`(不是 `-0.1`) | `prestrike_waist_twist = 0`(v2 包清零,交给全身模仿) |
| 脚组 | air `+0.5`/slide `-0.1`/ori `-0.4`/plane `-0.1`/landing `-1e-5`/close_xy `+0.2` | `pre_strike_foot_slip -0.2`、`foot_drag -0.5` | 大体已覆盖 | `foot_slip_sq -0.1`、`foot_drag 0`、`foot_soft_landing -0.003` |
| 全身角动量 | `angular_momentum -1e-4` | — | **新形状**,零权重探针先行 | **无对应物** |
| parkour 专属 | `freeze_upper_body -0.004`、`volume_points_penetration -8.0` | — | **不可迁移**(我们手臂就是任务) | — |

---

## 五、终止(termination)

| 项 | build_1 | exp 原口径 | **现役(08-05 后)** |
|---|---|---|---|
| 关节实际越硬限位 | **不终止**(恒返 False 的 `DoneTerm`,自陈 "matching the Unitree training structure");`actual_q_hard_limit_audit` 从 iter 20 起恒 `0` | 硬终止,冻结 reason order | **改成 `terminate=False`**,只记账不 reset;telemetry 模式强制要求证据记录器,否则 fail closed。MuJoCo 侧同步。**2026-08-07 裁定三**再把 pre-long gate / four-grid barrier 里的 `actual_hard_edge_event_count` / `actual_hard_terminal_count` 从"拒收条件"降级为"照记不照拦 + WARN 进摘要"(计数缺失/畸形仍 fail closed) |
| 参考包络 `ee_body_pos` | V9 **删掉腕部**(fresh 策略 smoke 出 `1.67` 步 episode) | 脚 + 腕(父类) | **只留双脚**(腕是挥拍必须甩最远的一端) |
| 撞桌 | **场景里没有桌子** | — | `robot_hit_table` 在 `HARD_TERMINATION_UNION` 里 |
| 死亡罚 | 无 | `-300` | `-10` |

**取证**:确定性 replay `7/7` episode 在 tick `69..88` 被 `joint_actual_forbidden` 终止、全部早于 nominal strike;
实测老师**不贴限位**(31 关节 × 57 帧最小余量 `0.116 rad` = `16.6%` 行程、零越限),所以那不是参考造成的。

---

## 六、三层对不上的地方(点名)

1. **智元的推撞 cadence 到底是不是 `[1,3] s`,两处说法冲突。**
   尽调 §11.2 把 `interval (1,3) s` 当成厂商实跑值,`HOPEPingPongActionBallA3VendorV1.yaml:48` 已经照抄进仓;
   但 DR-L2 候选声明白纸黑字写着"**cadence 在摘要里根本没给,`1-3 s` 是 BeyondMimic / unitree_rl_lab 的数,
   不要当成智元的**"。**幅值(`±0.25/±0.1/±0.26/±0.39`)两边一致、可信;cadence 这一格应降级为"来源存疑"。**
   好消息是结论不变:三种"击球窗期"读法里的两种都判 `[10,30] s` 达标、`[1,3] s` 超标约 10 倍。

2. **build_1 的 reset 三分支比例,两处数不一样。**
   §5.6.2c 记的是 `25%` 站立 / `35%` 随挥回放 / `40%` RSI;仓内同族 `HOPEPingPongHitter.yaml:146/165` 是
   `stand .25` + `post_swing .25`。前者应是 build_1 自己 argv 的活值,后者是我们仓里的模板值。
   **引作"build_1 怎么做的"时用前者,并标明它没有仓内字节佐证。**

3. **`death_penalty` 的活值已经是 `-0.2` post-dt,而尽调 §22 把一整串轴排在"死亡尖峰降到三库量级"之后。**
   也就是说 **M0 前置在字节上已经满足**。但闸 1(支撑集)与闸 3(cadence)各自独立,M0 满足 ≠ 全部放行。

4. **`start_pose_ramp` 的终点比 manifest 自己声明的站位框大 3.2–3.3 倍**(x `1.0` vs `0.30`;y `1.2625` vs `0.40`),
   两条路径**没有任何交叉校验**。一道 `mobility_mode="no_move"` 的题目配一条把机器人扔到 `1.26 m` 外的 ramp,
   今天没有任何门会拒。且满幅位移 `1.61 m` 而 `time_to_contact` 只有 `1.825 s` ⇒ 需要 `0.88 m/s` 走位再加一整拍,
   **没有任何 extrema-feasibility 门在算这个**。

5. **课程只升不降**(`action_ball_curriculum.py` 全文无 demote/rollback/shrink),与 exp §6.1 要求的
   "有可逆回退"直接矛盾;而且 30/32 条臂上限为 `0`,**解冻课程在现役 manifest 上几乎什么都解冻不出来**——
   真正要动的是重建一份非退化 manifest(73 库那份已有非零预算)。

---

## 七、如果只看"我们和那条能打到球的配方差在哪"

把 build_1 那列和现役那列并排,差异只剩五条,按"多半是它让我们打不到球"排序:

| # | 差异 | build_1 | 现役 | 已处理? |
|---|---|---|---|---|
| 1 | 探索 σ | `1.0` | `1.0` | ✅ 已对齐(08-05 第二次改版) |
| 2 | 关节限位终止 | 不终止 | 不终止 | ✅ 已对齐 |
| 3 | 腕部参考包络终止 | V9 已删 | 只留双脚 | ✅ 已对齐 |
| 4 | death penalty | 无 | `-0.2` post-dt | ✅ 量级已降到三库档 |
| 5 | **reset 起点分支** | 三分支(站立/随挥/RSI) | **100% 站立,验证器硬性拒绝别的值** | ❌ **未处理**,且改它要动 `canonical_ready_mode` |
| 6 | **随机化开关** | 质量/摩擦/CoM/零点/PD/obs 噪声**全开** | **全关**(DR-L0 归因对照) | ❌ 有意为之,但要记得我们比它**更简单**的同时还**没学会** |
| 7 | **桌子** | 场景里**没有** | 有,且 `robot_hit_table` 是硬终止 | ❌ 结构差异,交接单见 exp §5.6.12 |

**收尾的诚实话**:build_1 第 0 迭代自己 `mean_episode_length` 也只有 `23.1`/`23.4` tick,我们是 `22.3`——
**短 episode 本身不是缺陷,不构成发车阻塞**。真正没对齐的是第 5、6、7 三条,而其中只有第 6 条是可以
"改个布尔就试"的;第 5 条要动 canonical-ready 合同,第 7 条是场景结构差。
