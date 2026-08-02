# EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802 — ActionBall 下一版系统与 MuJoCo 原生训练准备账

- 状态：`in_progress`
- 阶段/轴：ChingMu-73 动作库、Ball-first 自动扩域、Isaac 最小可学门、MuJoCo 原生训练
- 集成小目标：先用一个自然动作证明最终球任务配方可学，再把主训练迁到 MuJoCo，并直接扩到通过机械准入的完整 73 动作
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 本 successor 最高证据等级：`E1`；历史 negative-control 另有 `E3` 诊断，不传递为新系统 E3
- 创建日期/最后复核日期：2026-08-02 / 2026-08-03

共享缩写按[术语与人话对照](../../DEFINITIONS.md)解释。本文件是下一版系统的**依赖、证据充分性和
版本迁移账**，不是全项目优先级队列。当前采用 setting、认领和算力顺序仍只认
[`origin/main` 的 `NOW`](../../NOW.md)。本分支的 225/318-D dense-paddle 配方和本文均是候选更新，
合入 `main` 前不得写成当前 adopted setting。

本文首次出现的缩写都按[术语表](../../DEFINITIONS.md)使用：`N1/N2/N3/N73` 分别表示一/二/三动作与完整 73 动作；
`ABI` 是 policy 的固定有序输入输出合同；`PPO` 是本项目使用的批量强化学习算法；`DR` 是域随机化；
`AMP` 是用判别器学习动作风格奖励的 Adversarial Motion Priors；`FK` 是由关节状态计算球拍位姿的
正向运动学；`RNG` 是可恢复的随机数状态；`MJCF` 是 MuJoCo 场景/机器人 XML；`VecEnv` 是批量并行
环境接口；`EMA` 是指数移动平均；`CCD` 是连续碰撞检测；`C3D` 是同步动捕数据容器。本文的
`READY` 是迁移交付状态，不是优先级或采用授权。

旧的[分阶段准备账本](../2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)继续保存
历史 Stage1 V2 long、运行收据和全部 `READY` 事实；本文件不删除或改写那些证据。但
旧 Stage1 漏掉球任务、outcome 和若干 reward，只是不完整配方的历史 negative control，不再构成
对一步到位系统的 concern。本文接管下一版设计裁决；待 `main` 完成切换后，再把旧账标为
`superseded`。

## 1. 第一性原理总裁决

当前改动方向大体是正向的，但旧 TODO **尚未形成闭环体系**。缺的不是再堆一批 feature，而是把
下面这条唯一因果链写成可验收系统：

```text
外部来源/本地事实取证
  -> measured-racket data + URDF/MJCF official-site authority
  -> kinematic retarget admission + independent mechanical admission
  -> engine-independent 便携合同草案 + MuJoCo core scene/runner/PPO  [现在并行]
  -> 最终 ABI + 完整 reward + ball-first scheduler
  -> Isaac 3xN1 recipe canary + N2/N3 conditioning canary + 冻结 handoff
  -> MuJoCo canonical N1 authorization + fixed-tape parity
  -> MuJoCo N1 fresh 复现
  -> 73 件逐动作 admission/alias/吞吐门
  -> MuJoCo N73 + ball-first 自动扩域
  -> 独立 physical exam / vendor / hardware
```

核心选择如下。

| 问题 | 裁决 | 理由 |
| --- | --- | --- |
| Isaac 是否仍是主训练目标 | `REVISE` | Isaac 只负责证明最终 MDP/Reward 可学和合同可移交；长期训练和 N73 转到 MuJoCo，减少训完后再跨物理引擎搬策略的风险 |
| 动作规模 | `ADOPT formal N1 -> N73` | 三个独立 N1 先验证同配方跨 BH-quality/BH-diverse/FH 可学；N2/N3 只验证同一共享 policy 跨不同 teacher trajectory 的容量与逐动作结果，不新增 motion-intent/ID。N73 只能纳入逐件运动学+机械准入的动作，不恢复 learned N5/N8/N12 阶梯 |
| 训练 Stage | `REJECT 手工换 Stage` | 从 rollout 0 就使用相同网络、optimizer、观测字段和 reward weights；所谓阶段只描述后续事件 reward 尚未有分母/收入的时间区间 |
| 问题分布 | `REVISE “冻结分布”` | 冻结生成程序、字段、initial/max envelope、扩域/回退规则、RNG 和 checkpoint state；实际采样分布必须随 ball-first curriculum 自动扩张 |
| full-phase 与 window-only | `ADOPT BOTH WITH WEIGHT SEPARATION` | 非腕全身 mimic 全程保留；measured paddle 的低权 position/velocity/signed-face/long-axis 全程保留来学专业动作；window 内 ball-conditioned `desired_at_contact` 是更高权的 task master，不用硬 mask 制造指导空洞 |
| 三层 paddle reward | `ADOPT STRUCTURE / RECALIBRATE SCALE` | coarse、adaptive fine、precision 分别解决冷启动、随学习收紧和触球精度；外部证据支持结构，不支持当前具体权重 |
| 智元 A3 setting | `ADOPT AS PRIMARY BASELINE` | 同底盘、动态全身运动对 plant/DR/delay/push 是强先验；reward、reset 分布和乒乓接触数值仍须按本任务证据裁决 |
| mjlab / 宇树 / BeyondMimic | `ADOPT SELECTIVELY` | 可固定 imitation 经济、MuJoCo manager/VecEnv 结构、机器人 DR/正则先例；不能代签球拍、触球、落点、旋转或 N73 成功 |
| Sony ACE / PACE / SMASH | `ADOPT TASK STRUCTURE` | SMASH 支持 task/style 和 adaptive sigma，PACE 支持 predicted+true outcome 及数值锚，ACE 支持 miss<hit<return 与 landing/spin conditioning；三者的算法经济不同，不能逐字搬权重 |

## 2. 四个维度必须分开

旧账把训练阶段、动作数、验证 Gate 和课程扩域混在一起，容易产生错误依赖。下一版固定为：

1. **动作规模**：正式路线仍是 `N1 -> N73`。3个独立 N1 与1个便宜 N2/N3 只是
   recipe/conditioning validation；中间可以做 zero-PPO/scale smoke，但不训练正式 N5/N8/N12 policy。
2. **Reward eligibility phase**：所有 callable 和 weight 从第一步安装；尚无接触/落台事件时，相关
   denominator 为零，因此还没有收入。这只是同一次训练里的时间区间，不是 operator 开关。
3. **Ball-first curriculum**：从所选动作的可解中心来球开始，按 checkpointed 规则扩宽位置、速度、
   时间、旋转和目标分布；实际问题分布不固定。
4. **Validation Gate**：`1x2`、`4096x5`、短学习门、跨引擎 parity 和 heldout exam 都是验证，不是
   训练 Stage，也不改变网络或 reward recipe。

从 rollout 0 起，球、桌、网、物理接触场景、完整观测字段和完整 reward recipe 都必须存在。早期
未发生事件时用相同语义的 teacher-consistent 值与显式 validity/eligibility；禁止在后续阶段新增维度、
换列语义或热改权重。

## 3. 尽调证据是否足够做选择

### 3.1 判据

每条外部结论按五个问题裁决：是否一手源码/论文、是否 exact revision、是否同机器人、是否同任务、
是否有消融或本地复现。证据强度决定允许做什么：

- **硬件/接口真值**：同 SKU URDF、MJCF、deploy header 一致时可以直接采用。
- **同底盘动态运动 setting**：可作为首选 baseline，不必为每个低风险轴购买仪式性 A/B；仍需机械健康、
  reward income 和任务结果门。
- **同任务消融**：可以采用机制；若没有公开绝对权重，不能宣称数值复现。
- **框架默认或同谱系 port**：证明可实现/可运行，不是独立因果证据。
- **不同 RL 算法的 sparse reward**：支持层级和目标定义，绝对数值必须换算到本仓会计后再定。

### 3.2 来源矩阵

| 来源与 pin | 能支持的选择 | 不能支持的选择 | 裁决/缺口 |
| --- | --- | --- | --- |
| 智元 `Instinct-Parkour-Target-Amp-A3-v0` 摘要；exact-SKU URDF/MJCF/deploy 多原件 | A3 nominal、Kp/Kd 分键、startup plant uncertainty、`[0,2]` control-step delay、六轴 push 幅值、clean/noisy 双评测 | 完整 AMP/task reward scale；篮球/拳击与跑酷确实继承同一 resolved config；乒乓 reward/reset 最优值 | plant 直接 `ADOPT`；DR/push 为首选 baseline。须补三任务 resolved config、commit、dt、event/reward manager |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking/tree/cd65172032893724b445448818c34165846d847d)，本仓首导 `8a9d329c` | full-body/full-phase imitation 六核、`dt=.02`、raw peak `5`、post-dt peak `.10/step`、failed-bin sampling 先例 | paddle/hit/landing/spin、A3 plant、N73 联训成功 | `ADOPT` imitation 底座；upstream checkout 未随仓固定，formal provenance 仍要 source manifest |
| [mjlab `a0a83e8`](https://github.com/mujocolab/mjlab/tree/a0a83e8191d19d6e25eccac94a2749fe248550a6) | BeyondMimic tracking port、MuJoCo manager/VecEnv/PPO 架构、同 error 多尺度 capture/precision 先例 | A3+桌网球吞吐、当前三层 exact 参数、真球必然更快 | `ADOPT` 架构和固定版本；matched workload 后才能选 backend |
| [unitree_rl_lab `4960b84`](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3) | 同谱系 mimic 数值、robot DR/正则/action scale、1–3 s push 的运行先例 | native MuJoCo training、球拍/球任务、单 clip 外推 N73 | `ADOPT AS PRIOR`；不是独立 reward 消融，也不是 MuJoCo trainer 证据 |
| SMASH / HITTER | style/task 分账、触球窗口、拍位/拍速/拍面通道、adaptive sigma；SMASH 去掉 adaptive sigma 的强消融 | 三个 paddle 权重和 ActionBall EMA/floor；真实 return/outcome scale | `ADOPT` 机制；SMASH `86.38%` 是仿真 racket tracking，不是回球率 |
| PACE / TTRL | contact、predicted landing/net、true-bounce 锚；sparse-only 与 guidance removal 消融；完整 event 数值 | imitation 共存时的最优绝对 scale、spin | `ADOPT` predicted+true 双账；把数值只当同任务锚 |
| Sony ACE | `miss < hit < hit-and-return`、落点/旋转条件化、真实 terminal outcome；replay/HER/event-table 支撑 sparse training | 正文未公开的绝对 reward；把 SAC/HER 下纯 sparse 直接搬到 PPO | `ADOPT` outcome 层级，`DEFER` 绝对值与正式 spin promotion |
| [原始 AMP 论文](https://arxiv.org/abs/2104.02180) | discriminator 从动作数据学习 style reward，再与 task reward组合；原论文不是显式逐项 pose tracking | 智元实现公式、权重、dt 和 typical income | 只用于解释 AMP 概念；智元数值必须拿其源码/配置 |

### 3.3 智元动态运动 setting 的可迁移边界

跑酷、篮球、拳击与乒乓共享 A3 全身稳定、腰肩臂动力链、快速恢复和冲击扰动。因此，对任务无关的
plant、delay、gain/mass/CoM DR、传感器噪声/history 和 disturbance，智元 setting 应提升为首选
baseline，而不是每项先从零发明。若取得三任务共同 base-config 的证据，这个判断会进一步加强。

但相似运动不能代签：球拍 face/contact timing、球反弹、落点/旋转 reward、push 是否命中 strike
window、reset 的 ball-first 可解性、以及 history 对应的真实时间长度。跑酷专用的
`freeze_upper_body`、terrain、depth/ray、volume penetration 明确拒绝。

## 4. 动捕事实边界与 measured-racket authority

最终系统的 racket teacher 必须来自**实测 racket channel**，而不是把重定向关节经 FK 得到的球拍
再当作教师真值。本轮由 Franco 裁定 URDF/MJCF 为几何 ground truth；其
`official_racket_site` 就是 policy/model 要复现的控制点，不再以“实拍外形或惯量尚未复测”阻塞
motion retarget。实测 marker
刚体 `M` 与该控制点 `S` 一般不共点、也不共轴，必须先冻结一个内容/标定绑定、全帧恒定的刚体外参：

```text
T_W_S^mocap(t) = T_W_M(t) T_M_S
E(t) = (T_W_S^mocap(t))^-1 T_W_S^FK(q_retarget(t))
v_S = v_M + omega_M x R_W_M r_S/M
```

重定向必须把 `T_W_S^mocap(t)` 作为末端约束参与求解，而不只是求完人体骨架后做事后检查；否则身体
可以看似匹配，拍位/拍速/拍面仍系统性错误。验收须逐动作报告 `E(t)` 的位置、SO(3) 测地角、signed
face 和 point-consistent velocity 的 p50/p95，而不能只比 marker 原点。静态外参可先沿用采集设计的
`<5 mm / <2 deg` 噪声门；动态重定向阈值必须在恢复原件后根据实测残差预注册，不能在无数据时编数。
此外 `T_M_S` 仍必须与 unit/BVH 内实测 blade/face 的内容 SHA 绑定；这是“动捕拍子如何映射到
URDF ground-truth site”的数据合同。它不是允许改 URDF 去追动捕，也不是用 FK 自己教自己：
验收仍要证明重定向后的官方 site/face 与实测 paddle 对齐。球拍的真实质量/CoM/惯量若日后用于
sim2real 标定，作为独立 physics evidence 管理，不反向否定本轮 URDF motion authority。

之前“ChingMu-73 只有 ball sidecar，raw racket 未恢复”的结论是错的。当前实测事实为：

- 本机 `/Users/Franco/Downloads/ChingMu_Selected` 有41组人体 BVH、41组拍子 BVH、
  41组桌 BVH 和26组球 BVH，原始帧率120 Hz。Pod 的同源根目录为
  `/workspace/yikang/a3_vendor_194d_physical_83b5ba8e/ChingMu_Selected`。
- canonical source manifest 有74个 unit，Pod 上74/74具备 unit NPZ+JSON，且
  `/workspace/yikang/chingmu_retarget/chingmu_a3_units_v2` 有74/74 PKL。最终73库在
  `CLIP_ORDER.json`/动作 manifest 中明确排除 `Take_085_unit00_FH`，不是原件丢失。
- unit NPZ 已有同钟 `paddle_blade_hope_m`、`paddle_butt_hope_m`、
  `paddle_normal_hope`；unit JSON 的 `hits[].face_normal_hope` 是有符号的物理接触面。
  OptiTrack 管线仍是独立的球物理/校准方法证据，不必被拿来替代 ChingMu teacher。

这个发现已转成实际代码与两层准入结果：

1. `solve_chingmu_canonical_racket_full_phase.py` 用 pinned MJCF `right_racket` site，在全动作
   相位约束实测 blade/signed-face/long-axis/point-velocity。face sign 先从每条动作的 measured-hit
   发现固化到 signed catalog，生产 solver 只读该 sign，不在优化中偷偷翻面。
2. `materialize_measured_racket_motion_npz.py` 使用 repaired PKL 重建50 Hz joint/body/COM
   velocity，对 robot FK 和 measured paddle 应用同一 heading/pivot，并写入 source/receipt/
   sign/axis/mesh SHA。`MotionLoader` 要求 schema-v4 measured channel all-or-none，不得回退 FK。
3. URDF/MJCF 固定关节和同一 rigid mesh 确认 official site 没有隐藏旋转；butt→blade 本体轴是
   `(local +X + local +Z)/sqrt(2)`，不是旧实现的 site-local `+X`。mesh SHA-256=
   `442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd`。

2026-08-03 又按 URDF 外表面修正了 MuJoCo collision proxy 的拍厚：每面去掉
`.396240 mm` 的多余厚度，local-Y scale=`.943396221367`。`right_racket` site、
wrist→site 位姿、FK 和 geom 中心均未移动，所以这不是“用改 model site 追动捕”；但
collision/model identity 确实改变。当前 root MJCF SHA-256=`70c4fd65…36c0a`，v2 portable
identity=`472219ae…dfd7a`。历史 v1 manifest 保持原字节，正式 MuJoCo lane 必须新建
v2 L0→vendor-L1→table/net successor 链，不得原地 repin 旧证书。

旧 schema-v3 库因此被撤销：用正确轴复算时，long-axis p50/p95/max=
`45.042/45.719/47.770 deg`，full/hit long-axis 和 full SO(3) 都是 `0/73`。它只能保留为历史
diagnostic，不能继续作 canonical teacher。

新 v4 sibling 已在 repo 本地版本化物化：
`assets/motions/chingmu73_measured_v4_20260803/`，bank receipt SHA-256=
`e6f0283f87401d004249689fbef30729fa7744ff6076a62c89996a945b727a82`。catalog/report/repaired/
materialized/audit 的 UID 集均为 `73/73`，50 Hz 共 `5107` 帧，10个 measured-hit sign 被纠正。
独立 FK 复核最坏数为：

| 口径 | full-phase p95 最坏 | hit 最坏 |
| --- | --- | --- |
| position | `49.31 mm` | `0.879 mm` |
| signed face | `6.769 deg` | `0.174 deg` |
| butt-to-blade long axis | `7.920 deg` | `0.126 deg` |
| SO(3) | `9.521 deg` | `0.197 deg` |
| point velocity | 未设 full-phase 硬门 | direction `4.320 deg`; relative `12.33%` |

这个 solver 不是只在击球窗开拍子约束：它从 measured-hit anchor 开始，然后向前和向后解完
整条 clip，每帧都约束 position/signed-face/long-axis；准入用全相位 p95，hit anchor 另有更严
的单帧门。120 Hz 全库 `11715` 帧的相位审核也排除了“只是首帧对不上”：仅 `13`
帧 position error 超 `50 mm`，其中 first/other-pre-hit/hit/post-hit=`1/1/0/11`；73条动作的最大
position error 有 `52/73` 出现在 post-hit，step-cap saturation 也主要在 post-hit
(`219/23/1124` pre/hit/post)。首帧 position error 中位数只有 `.082 mm`，只有
`Take_061_unit05_BH` 首帧为 `52.5 mm`。因此，运动学压力主要在触球后随挥，不是把约束
错开成 window-only，也不是全库第一帧错位。

所以“重定向后拍子和动捕拍子对上”在**预注册运动学/FK 门**内按
`73/73` 成立，并且是 full-phase p95 + 严格 hit anchor 的结论；但这不等于
“这些 joint teacher 机械上正常”。新的 fail-closed mechanical auditor 已审计 `73/73`：

- **mechanically admitted=`0/73`**。只有 `16/73` 同时通过已知 URDF position 与 stored/finite-
  difference velocity 检查，分母为 BH `9/59`、FH `7/14`；另外 `57/73` 存在已观察的
  position 或 velocity 硬失败。
- 那 `16` 条不是 mechanically safe，只是在已有上限下没看到 position/velocity 反例。
  因为仍缺 authoritative acceleration limits、每关节 torque-speed curve 和逐帧 floating-base
  inverse-dynamics torque，它们必须 fail-closed 为 `UNKNOWN`，不得晋级。
- `Take_060_unit09_BH` 不是可以绕过机械门的“clean N1”：right-wrist-yaw 有38个样本
  越上限 `3.8e-6 rad`，right-shoulder-pitch finite-difference velocity 是 URDF limit 的
  `1.05185x`，而 finite-difference acceleration `669.46 rad/s^2` 仍无 authority limit。
- 较早的窄口径诊断也解释了为什么会失败：相对原始 GMR，10个优化关节的
  absolute delta p95/p99/max=`1.409/2.212/3.108 rad`；`55/73` 动作触及 `.12 rad`
  solver step cap，`58/73` 有近限位帧。该窄口径中 `37/73` 超 URDF 速度限，
  全库 finite-difference 最大速度/加速度为 `14.4 rad/s` / `1122 rad/s^2`。

因此 v4 只能发布为 `KINEMATIC_RACKET_ADMISSION=73/73` 且
`MECHANICAL_ADMISSION=FAILED_OR_NOT_ESTABLISHED`的 `diagnostic_unauthorized` sibling，不能覆盖历史库或提升成
training-ready N73 teacher。下一步是在 solver 中加 soft-limit/velocity/acceleration 与 torque-speed 三角域门后
重算，再做 reference-tracking rollout 逐动作验收。这一反例会阻塞 canonical N1/N73，但不阻塞只读几何/
ABI/MuJoCo core 工程。

fail-closed builder 已从 v4 receipt 实际生成
[`action_ball_chingmu73_measured_v4_f10_20260803.json`](../../../configs/action_ball_chingmu73_measured_v4_f10_20260803.json)：
73 actions，file/canonical SHA-256=
`925b964c2ce6f5c57f56ef27af90c66d1c2516135dbac676cd5a6abc3f40c1e3` /
`4e49656aa398174750f4b096fed569f4413dadb59f8b1f6d31c59bffe9c11548`。它仍不能 formal launch：机械准入已
有反例，schema-v1 prototype 缺 `velocity_contract`，最终 ball-conditioned ABI/真球 reward/launcher 也未闭合。
恢复合同见[本地忽略资产同步](../../operations/setup_local_sync.md)，几何真源见
[Racket Control Point And Contact Geometry](../../interfaces/racket_contact_geometry.md)。

exact Pod scratch 上已将 v4 资产、URDF/MJCF 和当前源文件合并验证，选定的
reward/geometry/builder/loader/mechanical-audit 等回归为 `588 passed`。Hydra 实际 resolve 也确认
VendorV2 的四个 free-wrist/full-body flag、全相位拍子权重和 landing `500`生效；同时也直接证实
`physical_ball=false` 且 `actor_obs_contract=null`，因而不能发 final N1。本轮没有启动 GPU/训练/namespace。

### 4.1 当前运行真值

`HOPEPingPongActionBallA3VendorV2` 目前是本地 branch-candidate YAML，不是 active runtime authority。正式
ActionBall launcher 仍绑定 VendorV1；三条 225/318-D launcher 是明确 ball-free 的历史/诊断 canary。
`ACTION_SET_CONTRACTS` 仍未注册 final N1/N73，V2 继承 `physical_ball=false`，且没有 final actor contract。
所以当前不存在一条诚实的“VendorV2 单动作打球 policy” launch 命令；把 task 名强换进旧
launcher 会同时搞错 observation、full-body scope、physical outcome 和 receipt identity。

### 4.2 2026-08-03 当晚 launch 边界

本轮**不启动 VendorV2 formal N1**，也不把 `Take_060_unit09_BH` 换名后强发。这不是 GPU
不足，而是当前没有一条可复现且语义真实的命令：formal launcher 仍绑 VendorV1，final
ball-conditioned actor/critic ABI 未冻结，V2 还没有基于 actual physical contact/outgoing
flight 的 outcome bridge，v4 teacher 机械门为 `0/73`；本批 source/asset/config 仍是 dirty/untracked
branch WIP。此时发 `4096` 只会得到无法解释和无法晋级的第三种 recipe。

不改变 [`origin/main` 的 `NOW`](../../NOW.md) 统一优先级的前提下，当晚可安全并行的
是下列**前置工件**，而不是训练发射：

- 用 soft-limit/velocity/acceleration 和 authoritative torque-speed/torque 门重解 v4，从
  `16` 条已知 position/velocity-pass 候选中寻找第一条真正 mechanically admitted N1；
- 冻结 final purpose-grouped ABI、two-action delay history 和 physical-contact outcome eligibility，产出
  VendorV2 的 resolved reward/policy receipt；
- 增加隔离 namespace 的 VendorV2 diagnostic launcher，先做 `d=0` 的 zero-PPO/
  `1x2`/`4096x5`，只有上述 teacher/ABI/outcome 门闭合后才允许 fresh 学习 canary；
- 继续不依赖 canonical N1 授权的 MuJoCo core scene/single-env/action-delay/fixed-tape/
  VecEnv-PPO 接口工作，但不把它误报为 formal trainer 已可训。

## 5. Reward 体系与数值裁决

### 5.1 完整层级

```text
R = R_body-style(non-wrist whole body, full clip)
  + R_measured-paddle-trajectory(teacher_now, low weight, full clip)
  + I_strike * R_contact-task(desired_at_contact, window)
  + I_valid_actual_contact * R_hit
  + I_valid_actual_contact * I_valid_achieved_outgoing_flight
      * R_predicted-outcome(net/landing/spin)
  + I_outcome * R_true-outcome(legal landing/net/spin)
  + R_regularization/safety
```

- **动作模仿组**=`R_body-style + R_measured-paddle-trajectory`：非腕全身模仿保持动力链；实测拍子
  teacher 在全相位低权跟 position/point-velocity/signed-face/long-axis，因此击球腕虽从 generic
  body-position/orientation/velocity mimic 释放，仍会通过刚体拍 teacher 学到引拍、加速、触球、
  随挥和手腕 twist。“释放”是移除另一个可能冲突的手腕 body owner，不是不学手腕。
- **目标击球组**=`R_contact-task + R_hit`：学习当前来球/目标所需的触球状态并建立 `miss < hit`。
- **上台/结果组**=`R_predicted-outcome + R_true-outcome`：建立
  `hit < predicted legal return < observed legal return < target quality`。尚未真实落台时，基于 achieved
  出球的 predicted net/landing 提供可辨认引导；一旦有真实结果，必须由观察到的合法上台事件锚定，
  不能让预测器自评替代物理结果。
- 硬安全首先由 guard/termination 保证，不能只靠 reward 价格。

上式修正了旧文的 `I_strike * R_predicted`：只到时间窗不等于已经打到球。当前 analytic
virtual-ball 路径不会让纯 miss 拿到 landing，因为 `vb_fired` 已经绑定 exact-strike/指定拍面门；
但这也意味着它不是最终需要的 actual-outcome semantics。一次真实接触即使没跟到指定 face target，
只要实际出球有效并合法上台，outcome 层仍应按真结果付款，target-face 正确性另做 diagnostic/
contact-quality。所以 canonical 路线必须新增
`actual selected-rubber contact ∧ valid achieved outgoing flight -> predicted outcome`，再由 observed net/landing 锚定；
不能只把现有 virtual face gate 改名后当成已完成。

主任务的硬顺序是：**动作模仿 < 目标击球 < 上台奖励**。这里的 `<` 不是理论单步峰值，而是同一
admitted swing 上、按一致折扣和条件分母统计的实际贡献预算：

```text
B_G^eligible = E[sum_t gamma^(t-t0) r_G(t) | group G has a valid opportunity]
B_motion^eligible < B_target_strike^eligible < B_table_outcome^eligible
```

同时另报 `B_G^rollout`（把未触发记零的真实 rollout 平均），监视 dense motion 是否在优化经济里长期
淹没后两层。某训练早期还没有 hit/landing，后两层 denominator 为零是“尚未进入该 reward phase”，
应写 `未测`，不算层级倒置；但一旦对应 opportunity 有效，就必须有非消失 shaping 和上述条件预算。

早期若球题与原 clip 一致，可令 `desired_at_contact == teacher_contact_nominal`；扩域后两者可能不同，
必须分别入账。所以“full-phase 与 window-only 都要”的固定规则是：measured teacher 拍子全程低权，
window 内 `desired_at_contact` 高权主导。`teacher_contact_nominal` 还可用于可行 answer set 内的
nearest-teacher 选解。两者不是字段冲突，但必须用数量级分离保证 task 大于 style。
balance、action-rate 等辅助项按训练健康调整，但其 typical/p95
收入不得倒置主层级；硬安全继续由约束/termination 守住。

### 5.2 三层 paddle 的结构为何成立

| 层 | 解决的问题 | 证据 | 数值成熟度 |
| --- | --- | --- | --- |
| fixed coarse | 大误差冷启动和换球目标时避免 kernel 死亡 | 本地死核数学诊断；IsaacLab/mjlab 多尺度先例 | 当前 `.70 m / 4 m/s / 1 rad` 是本地 candidate |
| adaptive fine | 随学习误差下降自动提高精度 | SMASH 同任务消融；PBHC/KungfuBot 同族机制 | 已接入 ActionBall；`4/.5/.5`、EMA/cadence/floor 仍是本地设计 |
| fixed strike precision | 自 rollout 0 保留最终触球精度目标 | HITTER/SMASH 触球窗 | `.50/.25/.50` 无外部绝对权重支持 |

因此三层结构足以 `ADOPT`；当前 sigma/weights 只能标成
`PREREGISTERED/POD_WIRED_BASELINE`，不能写成 paper-validated。

### 5.3 统一会计、实际改动与静态层级 Gate

本仓与三个公开 mimic 栈的可比口径均为：

```text
post_dt_step = raw_kernel * weight * policy_dt, policy_dt = 0.02 s
```

但本次不只更新文档或峰值表。已新增
`HOPEPingPongActionBallA3VendorV2.yaml`，实际改动为：

- 显式恢复 `full_body_mimic=true`（覆盖父 V1 的 upper-body-only 旧选择），body motion scale
  `.15`；释放右腕 body position/orientation/linear/angular-velocity mimic；
- measured-paddle position/velocity/normal 各 `.20`，再加 measured butt-to-blade long-axis `.10`；
  Cauchy sigma `.70 m / 4 m/s / pi rad / 1 rad`，四项全相位付款。window 内 task kernels 的峰值权重
  position/velocity/face=`14.5/10.75/6.0`，远高于 teacher `.2/.2/.2`；long-axis `.10` 还规定
  9-D contact target 没有规定的 wrist twist。这消除了原先击球窗内的 position guidance 空洞；
- window ball-contact broad Cauchy position/velocity/normal weights `10/10/5`，sigma `.70/4/1`；
  broad velocity/normal 不被 position proximity gate 关掉，precision kernels 保留；
- 显式分窗：position `+-0.02 s`=3 steps，velocity/normal `+-0.10 s`=11 steps。这对齐
  [SMASH 的窗口结构](https://arxiv.org/html/2604.01158v1#S4.SS2)；SMASH 用的是
  `exp(-e/sigma(t))` 与自适应 sigma，没有公开我们的绝对 weights，因此 `10/10/5` 是本地候选，
  不是“SMASH 数值”；
- `virtual_landing_weight=500`，合法事件 post-dt 收入 `+6..+10`。它仍严格高于完整 target
  kernel+progress 上界 `5.485`，同时把单次事件尾部从约两倍 death scale 降到同量级；`1000`
  只保留为未采用的高尾部候选。
- ActionBall adaptive fine 从 rollout 0 启用：position/velocity/normal 主核权重 `4/.5/.5`，
  sigma 从 `.50/3/2.10` 按 `ball_exact_strike` 误差单调收紧到 `.075/.50/.262`；另有
  固定 precision overlay `.50/.25/.50`。live sigma 和 exact-error EMA 已纳入 strict exact resume，
  恢复时同步重建 RewardManager 中三个实时宽度。

`audit_action_ball_reward_hierarchy.py` 直接解析该 profile 和实际
`chingmu73_20260728/CLIP_ORDER.json`，用每件的真实 `T`、分窗和同一 `dt` 给出。审计还沿
defaults 链检查 V2→VendorV1→ActionBall，硬断言
full-body mimic、measured-racket teacher、action-ball target、ball outcome、table obstacle 和
完整 reward pack 同时存在；不允许只恢复 reward 数值却继续跑旧 Stage1/upper-body 配方。

| 真实73库口径 | masked 动作 prior cap | fine 验收边界 target income | target kernel + progress upper | legal landing |
| --- | ---: | ---: | ---: | ---: |
| p50（69帧） | `1.8975` | final sigma `4.0296` / initial sigma `4.3104` | `5.4850` | `6..10` |
| p95（99.8帧） | `2.7445` | final sigma `4.0296` / initial sigma `4.3104` | `5.4850` | `6..10` |
| max：`Take_062_unit11_BH`（133帧） | `3.6575` | final sigma `4.0296` / initial sigma `4.3104` | `5.4850` | `6..10` |

因此当球拍刚好达到 precision 边界 `.075 m/.5 m/s/.262 rad` 时，当前配置的静态顺序确实是
`动作模仿 < 目标击球 < 上台结果`。这是相容 swing 上的收入，不是只比较理论峰值。作为辅助数，
broad 三通道在各自一 sigma 时收入只有 `1.95`；它不再被错当成要压过完整 motion cap 的硬门。
对历史真实
strike-window 观测误差 `position=.634 m, velocity=1.9595 m/s, normal=56.21 deg`，
用同一误差回放 reward 数学：

| 配置 | split-window 总收入 |
| --- | ---: |
| V1 precision/coarse | `0.000690` |
| V2 收紧后 | `2.664360`；broad=`0.329613+1.774226+0.560521`，fine/precision 近零但 broad 不消失 |
| V2 rollout-zero 宽 sigma | `2.872667`；在同一 broad 之上 adaptive fine 再给 `0.208307` |

之前的 `2364x` 主要是 V1 分母已经近零，不是 SMASH 的 scale，也不是好的定标依据。新值仍证明
实际 reward landscape 已改，而且远区不会直接消失；将 task-face coarse sigma 从
`pi` 收到 `1 rad` 后，`56.21 deg` 的 broad raw 约 `.509`，不再拿旧设置中约 `.911`的近满额。
Cauchy 仍只是固定 coarse backstop；adaptive sigma 的必要性由
[SMASH 消融](https://arxiv.org/html/2604.01158v1#S6.SS1)支持，但绝对数值要用 N1/N3 tape 实测定。

当前 branch-candidate V2（尚未成为 formal launcher/runtime authority）的实际四个组件是：full-phase measured-paddle Cauchy prior、window
broad Cauchy target、单调 adaptive-fine exponential target、固定 precision overlay。因此 adaptive fine
已从设计变成真实 runtime 和 exact-resume state；仍未关闭的是训练中实际收入、advantage 健康
和 learnability，不是“有没有接线”。

局限必须写清：这是**配置会计 + 冻结观测误差 counterfactual**，证明修改了进入 PPO
的 reward landscape；它不证明新 policy 已学会。当前还没有 exact 球任务训练的逐 term
`raw/post-dt/eligible p50/p95/per-swing income`，所以可以关闭“只改 doc/远区直接消失”问题，
不能关闭 learnability 和实际层级 Gate。

### 5.4 Canonical 数值与非消失引导 Gate

当前 `.15` body scale、全相位三个 `.2` measured-paddle + `.1` long-axis pin、`10/10/5` broad target 和
`virtual_landing=500` 已通过配置会计与冻结误差门，但仍是 branch candidate。它们与
BeyondMimic/mjlab/unitree 的 imitation 数量级及 PACE 事件经济对照后无明显数量级冲突，却还不能
写成“已与智元 AMP scale 对齐”，因为智元 discriminator/task 的 resolved income 仍未取得。最终
one-run 发射前必须用冻结 tape 和同一批 admitted swings 完成以下 Gate，再决定是否保留绝对权重：

1. 对每个动作/侧别/phase 报告关键误差的 p10/p50/p90/p95/p99；每个 kernel 同时输出
   `x=e/sigma`、raw kernel、post-`dt` 收入、eligible denominator 与 discounted per-swing income。
2. precision 指数核 `rho_exp(x)=exp(-x^2)` 在 `x=1/2/3` 时仅为约
   `.368/.0183/.000123`；这就是它不能单独承担远区引导的原因。V2 broad 改为
   `rho_c(x)=1/(1+x^2)`，在同三点为 `.5/.2/.1`，任意有限误差收入为正；其对实际误差的
   敏感度绝对值为 `2 * weight * dt * |e| / sigma^2 / (1+(e/sigma)^2)^2`。但“非零”
   不等于 PPO 中可辨认；仍要用高于传感器噪声、且小于一步可控改变量的 `delta`
   做有限差分，在实际误差分布内证明改善信号超过 advantage/noise floor。
3. PPO 不通过 simulator 对 reward 反向传播，上述导数只是 reward landscape 的 dead-zone 代理，不是
   policy gradient 本身。还须记录各组对 return/advantage 的 typical/p95 贡献及训练梯度健康；统计
   “状态明显错误但所有相关 kernel 都近乎无引导”的比例。dead 阈值由 float32、advantage 噪声和
   实测 `delta` 预注册，不写一个脱离实现的通用 epsilon。
4. coarse 层要覆盖 ball-first 初始中心域到当前 admitted 外沿；adaptive fine 收紧时，coarse 仍须
   保持覆盖。若题目落在所有核的支持外，generator 应拒绝/回退或暂时放宽支持，而不是让零梯度样本
   污染 PPO。`e≈0` 时导数为零是达到目标后的正常现象，只有 materially wrong 时无引导才是缺陷。
5. 在相容 eligibility 分母上冻结并验证 `B_motion < B_target_strike < B_table_outcome`，同时检查含零
   的 rollout 平均不被 dense motion 永久淹没。早期真实上台稀疏时，contact 时刻的 predicted
   net/landing shaping 必须提供连续引导，并由真实合法上台事件锚定；预测项不得在无有效接触/飞行时
   支付完整上台待遇。`R_hit` 与 quality 要明确是增量还是总包，避免双重计价。
6. balance/regularization 逐项报告 typical/p95、最坏界和终止影响；可以按健康调整，但不能在常见轨迹
   上淹没三层主任务。任何动作/侧别零分母继续写 `未测`，不得用全局平均掩盖。

### 5.5 “AMP reward 数值”到底是什么

AMP 是 Adversarial Motion Priors：判别器区分 motion 数据与 policy 状态转移，产生 style reward，
再与 task reward 一起训练 policy。它不是某个固定的 `action_rate` 数字，也不是 BeyondMimic 的六项
显式 imitation reward。

当前智元摘要能确认的只是 AMP trainer 中的显式 regularizer raw coefficients：

- `action_rate_l2=-1e-3`、`dof_pos_limits=-2`、`torque_limits=-.01`；
- `angular_momentum=-1e-4`、`self_collision=-.1`；
- hip/torso deviation `-.01/-.05`、torso/pelvis orientation `-.6/-3`；
- feet air/slide/orientation/plane/landing/close-xy=`+.5/-.1/-.4/-.1/-1e-5/+.2`；
- parkour-specific `freeze_upper_body=-.004`、penetration `-8`。

未知的是智元 discriminator/style 公式及系数、task reward 公式及系数、是否乘 `dt`、normalization 和
typical income。因此只能对齐正则的形状/符号/raw 数值，**不能确认 ActionBall absolute reward scale
已与智元 AMP 对齐**。取得 resolved config 和 reward manager 前，这一格必须写 `未测`。

## 6. 智元 setting 的采用表

| 轴 | 下一版选择 | 状态/健康门 |
| --- | --- | --- |
| exact-SKU effort/armature/nominal plant | 以 URDF/MJCF/deploy 多原件为真源；拒绝 parkour wrist regex 错表 | `READY / ADOPTED_BASELINE` |
| Kp/Kd startup DR | Kp `(0.8,1.2)`、Kd `(0.7,1.3)` 首选 baseline；31-DoF head 扩展标 HOPE-owned | canonical run 前逐关节 resolved receipt；hold/teacher-to-hit/safety |
| action delay | episode-fixed `[0,2]` control steps 是候选 support；在 50 Hz 下是 `{0,20,40} ms` | `CANDIDATE / ABI-BLOCKED`；当前 actor 缺第二个 pending raw action。先补 history/recurrence 并并行 d=0 与 uniform candidate，不得声称 BeyondMimic 已证明 linear ramp |
| 六轴 velocity push | `vx/vy ±.25, vz ±.1, roll/pitch ±.26, yaw ±.39`，cadence `1..3 s` | `READY / BASELINE`；按 pre-strike/strike/follow-through/recovery 分账，不做额外 force push |
| mass/CoM | torso/末端/拍子优先，测量值优先于随意 `±20%` | `CANDIDATE`；惯量一致性、hold、hit/safety 门 |
| friction | 不把 PhysX joint friction 数字直接搬成 MuJoCo `frictionloss` | `DEFER TO MUJOCO CALIBRATION` |
| obs noise/history | 采用分量化噪声与 history 结构；时间窗按本仓 50 Hz 重算 | `CANDIDATE`；会改变 ABI，必须在最终 contract 一次冻结 |
| reset noise | 由 ball-first 可解域扩张，不一次全开 parkour reset | `REVISE`；checkpointed scheduler/恢复门 |
| motion 速度 | 73 条自然动作优先原速，当前 `speed_scale_range=[1,1]` | `ADOPT`；禁止把整条 clip 统一拉到“最高速”；日后重定时须另过动力学门 |
| torque-speed | 使用 signed 转速-净力矩曲线/热包络，不把两个独立上限当矩形 | `BLOCKED ON VENDOR CURVE`；当前线性三角只作保守排序，不据此删 action |
| Motion-VAE | 等新一轮高质量/更完整动捕后再启 | `DEFER`；当前先用 teacher-trajectory-conditioned shared policy 建立基线，不加动作 ID |

“采用 baseline”表示根据同底盘/多动态运动证据选择首发 setting，不等于每项对乒乓表现的因果最优已
证明。低风险项不必逐轴做科学 A/B；任何导致 contact denominator、teacher-to-hit、hard safety 或
吞吐失真的项都 fail-closed/回滚。

### 6.1 “更真实”不等于“所有难度第一步全开”

尽调显示应区分两类变量：episode-fixed plant uncertainty（gain/mass/CoM/friction/encoder bias）常从
rollout 0 使用冻结 support；task difficulty（来球/落点/旋转、新 band、push 暴露比例）更适合由
ball-first/performance gate 扩张。没有一手 ablation 证明 `[0,2]` delay 必须线性 ramp。
BeyondMimic 论文没有讨论 action/observation-delay curriculum；官方 G1 使用普通
`ImplicitActuatorCfg`，通用 delayed actuator 默认 `min=max=0`，tracking `CurriculumCfg`
为空。因此“BeyondMimic 证明 40 ms 不能从头学，必须线性加”是无一手证据的
二手推断。SMASH 支持 task-distribution/adaptive-tolerance 课程；PACE/ACE 可从头带球与校准
delay/noise，但还依赖 history、predictor 或 SAC/replay/HER，不能单独外推到当前 PPO N1。

延迟的**代码路径/history 字段从 rollout 0 安装**，但学习风险高的幅度按 episode-level
分布扩张：`D0: P(d=0)=1`；稳定触球后 `D1: d∈{0,1}`；再到 `D2: d∈{0,1,2}`。
各带始终保留 `20%–30%` d0/低延迟 floor，扩张由分层 hit/return/safety + hysteresis 触发，
schedule/RNG 进 checkpoint；禁止在某 iteration 让所有环境突然切到 2-step。当前 actor 只
观察一个 previous action；`d=2` 前必须增加两步 raw-action history、recurrence 或显式已知 lag，
否则队列中第二个 pending action 是隐藏状态。当晚 contact-guidance 五臂为了单独回答
target semantics，统一用 action/observation delay=`0`、no push、no wide DR 和 fixed question tape；
延迟是优胜 recipe 之后的独立轴。

相同 mindset 下，下列“更真实”轴不得混成一次冷启动：

| 轴 | rollout 0 | 后续扩展门 |
| --- | --- | --- |
| nominal plant + 低风险 gain/mass/CoM/noise support | 采用，且整个 support 从开始可见 | hold/teacher-to-hit/task/safety 分层健康；禁止 startup 与 reset 双重抽同一轴 |
| 来球位置/速度/时间/落点 | 只用 ball-first 可解中心域 | checkpointed band curriculum，有可逆回退与独立 new-band 分母 |
| spin/off-centre contact | 列存在但 `spin_valid=false`，reward=0 | 飞行、摩擦/回弹、旋转传递和别名门全过后单独 promotion |
| push | cadence/幅值可从 rollout 0 存在，但不强求立即覆盖 strike window | 按 pre-strike/strike/follow-through/recovery 暴露分层，任一层 safety 恶化停车 |
| delay | learnability 臂用 `d=0`；完整 history 后并行 `{0,1,2}` candidate | 只在 paired d0/d2 证据需要时发 fresh curriculum 臂 |
| CCD/减半全场 dt/贵重 contact reporting | 不一次全开 | 通过单轴 matched throughput + tunneling/contact truth 门再采用 |

第一版同时明确 defer：`spin_valid=false`、未标定 off-centre spin/contact 不付款；push 需按
pre-strike/strike/follow-through/recovery exposure 统计，必要时扩张 exposure 而非改变 plant SHA；
ball/question distribution 始终由冻结 ball-first 规则扩张。这样保留真实场景目标，同时不把不可观测、
未标定或完全稀疏的困难混成一次冷启动。

## 7. 真球是否会让训练很慢

当前严格答案是：**未测，不能说“一个真球就会很慢”。**

- 现有 `4096x5 ~= 6.7 s/update` 基准是 `physical_ball=false`，最大段仍是
  `solver_solve_many`/reset；它不能给 ball tax。
- 仓内曾有 `physical_ball=true, 4096 env` 构造/finite checkpoint，但没有 matched wall-time。
- TTRL/PACE 公开配置用 4096 个动态碰撞球，说明“不可训练”不成立；其仓库没有公开本机可比 steps/s。
- 一个球只增加一个动态刚体；真正可能昂贵的是每 substep aero/root read-write、reverse RK4 发球、
  paddle/table scan、contact reporting、CCD 或把整个场景 dt 减半。
- 当前 ActionBall `PhysicalBall` 关闭 collider/CCD，用代码驱动拍球/桌弹，因此也不能代表原生接触成本。
- 当前 MuJoCo `a3_pingpong.xml` 尚无桌网球；已有 CPU 数据同样不能回答 MuJoCo 真球速度。

发移植前的 matched benchmark：

| Arm | 相对上一臂只增加 | 归因 |
| --- | --- | --- |
| A | 无球 | 基线 |
| B | 停放球、无 collider/callback | 刚体/状态缓存税 |
| C | flight/aero/serve，impulse off | RK4、aero、root read/write |
| D | code-driven paddle/table scan | substep FK/扫描 |
| E | 原生 collision，CCD off、无 reporting | contact solver |
| F | 单次净接触力读取 | 合理 reporting |
| G | CCD；另开一臂单独减半 dt | 分开 CCD 与全场 substep 税 |

每臂在 `1/512/4096 env`、同 GPU/commit/tape/solver 下交错运行；10 update warm-up、至少 50 update
profiler-off 计时，reset-free 与固定 reset-count 分层报告。记录 scene build、GPU memory、physics、
collection、PPO、serve/reset/callback、p50/p90 wall 和 env-steps/s，并做 RNG/reason/counter/safety/
reward/obs parity。选 CPU、mjlab Warp 或其他 backend 只能由 A3+桌网球的同工作量结果决定。

## 8. Canonical portable contract

当前 225/318-D V2 是 dense-paddle canary 合同，不是最终 ball-conditioned N73 合同。最终版本必须在
N1 开始前一次冻结：

- exact ordered actor/critic term、dim、unit、frame、source、validity/age、normalizer update rule；
- incoming ball、current achieved paddle、`teacher_now`、`teacher_contact_nominal`、可选
  `desired_at_contact`、desired landing/time-or-arrival-speed/spin；其中 achieved 来自 simulator/live
  FK，两个 teacher block 来自实测 racket 经冻结 `T_M_S` 映射，contact desired 若启用则来自
  ball/task planner，字段不得共享偷换 source；
- 由所选 teacher trajectory 本身表达动作；禁止 N-wide one-hot、UID/slot 或额外 motion-intent code；
- 两只独立的钟不得合并：`time_to_contact/t_hit` 描述球何时到触球点，
  `time_to_teacher_start/wait` 描述何时启动该动作 teacher；同一来球可因 clip 前摆时长不同而有不同 wait；
- `31-D action -> scale -> episode-fixed delay -> qdes` 和 A3 plant；
- ball/table/net/contact/reset/termination、完整 reward recipe 和 ball-first scheduler；
- checkpoint 恢复 optimizer、normalizer、delay、curriculum、eligibility 和 RNG state；
- portable semantics、Isaac binding、MuJoCo binding、normalizer、checkpoint、export/judge 使用
  分层 SHA lineage；不同引擎字节不要求一个错误的 literal plant SHA。

这里不再定义“N73 fixed-width content intent”。不同动作已经具有不同的全身
`q_ref/dq_ref`、body reference、`teacher_now` 和完整 measured-paddle trajectory；同一相对 phase
上，73 库的 `q_ref+dq_ref` 没有跨动作 exact collision。`teacher_contact_nominal` 是这条专业动作
在自然触球时的 nominal paddle state，用来让 policy 看见“老师本来会怎么打”与“当前球题要求
怎么打”的差，而不是让 policy 猜动作编号。若将来真的发现当前 teacher state 相同而必要未来不同，
才加入短时 future-teacher preview；不能为追求数值唯一性制造18-D伪身份。

最终有序分组固定为：

```text
actor  = robot/achieved -> teacher/reference -> incoming-ball/task target -> clocks/validity -> causal history
critic = privileged robot/teacher -> same exogenous task -> achieved outcome/eligibility
```

incoming ball 至少含预测到触球时的 `position3/velocity3/spin3 + time/valid/age`；achieved paddle
是当前实际 site 的 `position3/point-velocity3/signed-face3`；teacher 是当前 reference 与 nominal
contact baseline；desired contact 是 A/B 路线下 planner 给的接触要求；landing/spin 是目标出球
结果。第一版 N1 保留 spin 列但令 `spin_valid=false`，spin reward 不付款。normalizer 是按上述固定
列顺序保存的 mean/variance/count；checkpoint 还必须保存 actor/critic、optimizer、normalizer、
delay queue、curriculum/eligibility 和 RNG。

“宽度”是 actor/critic 输入的标量列数，不是隐藏层 `[512,256,128]`。当前 canary 是225/318；
最终宽度在 A/B/C contact 路线与两步 delay history 闭合前不宣告，且相同宽度但顺序不同也不是同一 ABI。

字段的人话含义固定为：

- `incoming ball`：policy 在触球时刻预期会面对的球位置/速度/旋转/时间；actor 只用因果可获的
  prediction，critic 可有 privileged truth 但不能泄漏给 actor。
- `achieved paddle`：sim/live FK 给出的当前实际拍位、point velocity 和 signed face，用来告诉 policy
  “我现在真的做到哪了”。
- `teacher_now` / `teacher_contact_nominal`：专业动作当前相位与自然触球时的拍状态，表达动作本身，
  不是 ID。
- `desired_at_contact`：若 A/B 路线启用，planner 对当前来球/落点所要求的触球拍状态；它与
  teacher 之差正是 task adaptation，不是用来识别动作。
- `landing/spin`：想要的出球结果。actual landing/spin 只进 critic/evaluator/reward truth，不得作为 actor 的
  未来泄漏。`valid/age` 用来区分“数值恰好为0”与“字段缺失/过期”。

SHA 不是要 Isaac、MuJoCo、export 三个引擎强行使用同一串字，而是可追溯的分层 DAG：

- portable semantics SHA 绑 term order/dim/unit/frame/source/validity/reward/event 语义；
- Isaac/MuJoCo backend-binding SHA 分别绑各引擎的 body/site/contact/plant bytes，两者共同指向 portable 父 SHA；
- actor/critic normalizer SHA 分别绑 ordered mean/variance/count，禁止把列顺序漂移藏在相同 width 里；
- checkpoint SHA 绑 model/optimizer/normalizers/curriculum/delay queue/eligibility/RNG 及所有父 SHA；
- export SHA 绑实际 ONNX/制品 bytes，声明 normalizer 是 baked-in 还是外置，并指回 checkpoint/ABI/backend 父链。

### 8.1 落点任务到击球控制的 A/B/C 路线

当前 ActionBall 只给 `aim_xy` 落点，运行路径使用 fixed-direction LM；它既不是完整
spin-aware `desired_at_contact`，也不是纯 outcome-only policy。三条路线必须在同一 immutable
feasible question tape 上并行比较，不能先假定最复杂者正确：

| 路线 | actor 获得什么 | 计算与证据 | 本轮裁决 |
| --- | --- | --- | --- |
| A：完整 contact oracle | incoming/task + `desired position/velocity/face`；teacher nominal 仅作可行集内近邻 pin | task→出球→接触可行集；SMASH/HITTER 只证明无旋、无摩擦的闭式 A-lite，不证明完整 spin/friction inverse | 主 oracle arm；必须事件级缓存、批量解析/LUT/固定轮数，env 热路禁 data-dependent LM |
| B：部分 contact guidance | incoming/task + position/face，速度由 policy 学 | 若 position/face 仍由 A 求出，则**不节省求解成本**；若直接用 teacher position/face 才真正便宜，但可能与新来球/落点不相容 | 拆成 `B_solver-face` 与 `B_teacher-face` 两臂，不能把前者宣传为加速方案 |
| C：无 contact target | incoming ball + landing/task；靠 actual contact 后 predicted landing/net shaping 与 observed legal return 学 | PACE 在4096 env证明 fixed-goal humanoid C-lite 可学；ACE证明 task-conditioned direct RL，但依赖 SAC/HER/replay。PACE sparse-only 消融失败 | 必须是 dense forward-outcome C，不允许退化成只给稀疏上台奖；作为最低 planner 成本基线 |

A 的目标合同若采用为：

```text
incoming ball + landing/time/speed + desired spin
  -> fixed-cost flight inverse/LUT + net inequality
  -> required outgoing ball state
  -> feasible contact cap/friction-cap
  -> nearest teacher-compatible solution
  -> exact close or a fixed number of batched refinement rounds
  -> desired_at_contact
```

无解题在构造期拒绝；`answer_sphere` 只在零入旋、固定恢复系数模型中是精确球冠，含旋/变摩擦
必须重新验算。A/B/C 共享 superset ABI 与 validity：A 的 contact p/v/n 有效；B 只令其实际提供的
子字段有效；C 全部 contact target 无效。所有臂仍看 incoming `p/v/spin`、desired landing/time/
speed/spin 和 teacher trajectory，保证差别只来自 contact guidance。

性能不能靠论文名背书。历史本机 `4096 env x 5 update` profiler-on 诊断总计 `33.499 s`
（约 `6.700 s/update`），其中 solver span `16.367 s`，占 collection `49.71%`、占总 wall
`48.86%`；两个分母不得混用。减去该 span 得到的
`3.427 s/update` 只是理想下界，不是新实测。Pod CPU fixed-tape microbench 中，每批4096 proposals
的 LM4/8/12 分别约 `6.71/15.15/23.18 s`；当前 analytic 实现约 `.157 s` lean、`.954 s`
default。另一批4000同模型 replay 为 analytic `.415 s` 对 LM12 `12.979 s`，analytic 的 admission
`100%`、landing error p50/p99/max `.128/.641/.946 mm`，而 LM12 为 `96.65%` 和
`1.662/7.831/19.592 mm`。这些是事件批次 microbench；solver 只在 reset/question construction
触发，并非每 physics step，所以不能直接把单批时间当 update 税。

固定 N1 的结论更强：五种 target 应离线各生成一次，训练 reset 只消费 immutable
tape 的同一行。纯 tape `4096` reset view 实测中位数 `.275 ms`，
`online_lm_calls=0`、`physical_rng_draws=0`。但当前 runtime adapter 仍会经 sampler 且逐环境
materialize compatibility receipt，该路径约 `1.736 s/4096`；所以在 device 上直接
expand/index 单行的 fast path 未闭合前，不得宣称完整 reset 已 O(1) 或
`online_sampler_calls=0`。LM vs analytic 训练对照只比 target 语义；producer cost 用另一份
benchmark 比，不再污染 PPO wall time。五个 target receipt 还必须共同绑定相同
teacher-rate/`t_hit`/cycle/pre-swing-wait，不能只比 base-question SHA 后就声称同题同钟。

正式选择门是 profiler-off、相同 reset strata 的 matched update：先以 analytic A
`<=.35 s/update` 且 total `<=3.6 s/update` 作为工程预算候选，同时要求 fixed-tape feasibility、
landing/net/contact residual 和安全 parity。超预算则先优化/缓存 A；若仍超预算，B/C 已是预注册
并行路线，不以热改 baseline 的方式临时删字段。

当前 fresh ActionBall 仍是 N1-only、ball-contact ABI 未冻结；formal launcher 也仍绑定 VendorV1。
N73 的实际 blocker 是 ball-conditioned producer、A/B/C选择、normalizer/checkpoint/backend consumers
与逐动作机械准入，不再是虚构一个 fixed-width motion intent。

### 8.2 PPO runtime receipt

静态 reward 会计不能代替一次真实 trainer 收据。canonical N1 发车前必须记录 exact Pod checkout、
`rsl_rl` source SHA、resolved actor/critic order+width、fresh/resume normalizer state、configured/realized
`init_noise_std=.02`、`noise_std_type=log`、`entropy_coef=.01`、optimizer/adaptive-KL 设置和 finite
iteration cap。前200 iteration 至少监视 `mean_noise_std`，前500 iteration记录 LR/KL、clip fraction、
explained variance、pre-clip grad norm、advantage/return tails 和逐 reward-group eligible income。
这些是 launch/health receipt，不是因为 reward 变了就一起调 entropy/std 的额外消融。

### 8.3 Reset、termination 与 exact resume

`canonical_ready`（动作数据能否提供准备位）与 reset policy 必须分开。reset 从逐环境 O(env) 工作改为
只处理 terminated batch；恢复 phase-gated fidelity termination、follow-through buffer 和
recovery-only RSI，明确禁止 mid-swing 把机器人/球“空投”到新状态。每个 reset reason、phase、动作、
side 和球题单独计数。旧吞吐目标保留为待复测门：Gate A `9–11 s/update`、Gate B `6–8 s/update`、
stretch `<5 s/update`；旧 profiler-on `6.7 s/update` 不能直接晋级。

checkpoint 除网络和 optimizer 外，还要保存 normalizer、每环境 delay 与完整 raw-action queue、ball
curriculum/arm assignment、eligibility/event latch、episode/reset counters 和全部 RNG。cold-load 后在首个
rollout 前对 fixed tape 检查 qdes、delay histogram、question、reason/counter 和 reward/obs parity；缺字段
fail-loud，不允许重新抽样假装 exact resume。

当前实现不满足这个完整定义：adaptive-sigma EMA 和部分 delay queue 可序列化，但 runner load 后
会立即 reset，重抽 lag 并重填 queue。所以现有证据只能叫 **reset-boundary resume**，不是
mid-episode rollout continuity。新 receipt 必须直接写这个语义，修好前不得使用 `exact resume` 的模糊简写。

## 9. Isaac 最小可学门与 MuJoCo 顺序

### 9.1 Isaac 最小可学门

Isaac 不再承担 N73、广域 long、最终 sim2real 或部署成功。它只回答“最终配方是否会学、能否冻结移交”：

1. 先选 BH-quality、BH-diverse、FH-control 三个通过机械 admission 的自然动作，各做一条独立
   N1；其同钟实测 racket teacher 映射到 `official_racket_site` 并通过逐动作残差门。三者使用同一
   最终球/台/网场景、portable ABI、完整 reward recipe 和 ball-first scheduler，全部从 rollout 0 存在。
2. `1x2`、`4096x5`、save->cold-load、finite export、normalizer、action-scale/delay/qdes exact。
3. 预注册短学习预算；在冻结中心 holdout 上，相对**实测 racket teacher**的 full-phase/exact-window
   paddle error 下降，并出现真实 physical hit 与 legal return 的学习，而非只看 motion mimic、
   FK-derived self-consistency 或总 reward。
4. 按[逐拍账本](../../interfaces/action_conditioned_ball_first_contract.md#5-attempt-账本)记录
   proposed/admitted/installed/started/closed/legal-return/safe-nonreturn/unsafe，连同动作/侧别、
   paddle/body income、hard/table safety；零分母写 `未测`，不跨动作平均。
5. 至少机械演练一次自动扩域、回退、checkpoint->resume，ABI/reward SHA 不变。
6. 产出冻结 handoff bundle：contract/plant/reward/physics bytes、checkpoint、fixed tapes、oracle 与性能预算。

三个 N1 通过只说明共同 recipe 能跨三种动作单独学，不能证明一个共享网络有足够容量同时跟踪不同
teacher 并解不同球题。因此保留短 N2/N3 canary，但验收是逐动作 teacher/task 结果、动作间串扰和
容量，不是 intent swap/shuffle/zero。该 canary 不新增 ID、不作 N73 checkpoint 起点，也不恢复动作数训练阶梯。

历史 Stage1 V2 `605 tests + 1x2 + 4096x5` 只证明当时那份不完整配方的构造、吞吐和九项
reward 活；旧无球 motion-prior long 也只是 historical negative control。它们不是对完整 one-run
设计的 concern，也不需要再跑一个手工 Stage 来“解除”；正确动作是从 rollout 0 把缺失的球任务/
outcome/scheduler/reward 全部加回。当前 successor 已有本地 v4 `73/73` measured-racket **运动学**
retarget/materialize/FK-audit 闭环和新 reward static/counterfactual Gate，但机械准入已发现超速/限位
反例，且尚未进入完整球任务的 exact Isaac boot/学习门。

### 9.2 MuJoCo 顺序

- **MuJoCo core 现在并行做**：pin mjlab/runtime，实现 MJCF/scene/plant、action/delay、deterministic
  reset、batched VecEnv、PPO、checkpoint/save-resume-export、ball-table-net contact harness、独立 reward/evaluator
  oracle 和 fixed tapes。当前只有 scene/contact/teacher-eval 积木是 `PARTIAL`，single-env plant/action 和
  VecEnv/PPO/checkpoint 均是 `NOT_IMPLEMENTED`；路线可以现在并行，不等于误报 trainer 已在开发中。
- **只有 canonical N1 authorization 被卡住**：最终 ABI/reward/measured authority/scheduler 冻结后，才允许把
  MuJoCo core 称为 canonical trainer 并跑 formal N1。开发期 robot-FK recipe 可用于 diagnostic engineering/
  learnability bring-up，但必须用不同 `teacher_source`、recipe SHA 和证据等级，不能代签 formal measured N1。
- **MuJoCo N1**：fresh-from-scratch 是主结果；Isaac actor-only warm-start 只作对照，critic/optimizer fresh。
- **MuJoCo N1 复现后**：开启完整 N73，而不是回到 Isaac 购买 formal N5。

MuJoCo 验收拆两层确定性：Tier-1 对 question/curriculum/receipt/ABI/action identity 要求 exact；
Tier-2 对 Warp/GPU 物理轨迹默认只要求统计等价，除非 CPU golden 已证明 bit-exact。native contact harness
必须具名包括 ball-racket/table/net、`solref/solimp`、摩擦/恢复、drag/Magnus/spin、CCD/tunneling 和
contact/event latch；只能程序驱动球或“有一个 scene”不能关闭该门。

迁到 MuJoCo 会减少 PhysX-policy -> MuJoCo-policy 的二次迁移，但不会自动消灭 simulator overfit；最终仍需
独立 heldout evaluator、vendor Gate 和硬件证据。

## 10. N1 直接到完整 73 的门

“一个动作能学就全上”精确定义为：N1 通过后，允许**完整 73 catalog**进入 MuJoCo 训练实验；它不
证明 73 件已分别学会，也不允许不合格动作静默消失。发 N73 前必须：

- 冻结 exact ordered 73 manifest；逐动作 compiler/FK/face-sign/`t_hit/t_cycle`/table-clearance/
  dynamics/measured-racket teacher/fitted-ball MuJoCo admission；
- 审计 teacher observation 的 Markov 性；只有发现相同当前 teacher 状态却要求不同未来时，才增加
  short future-teacher preview，禁止用动作 ID 解决；
- 直接跑 N73 zero-PPO、`1x2`、`4096x5` scale smoke，不需要中间动作数 learned stage；
- 记录逐动作/逐侧 usable closed attempts、reward income、hit/return/safety、min denominator 与最大 starvation age；
- 选择并冻结采样意图：现库正手 `FH=14`、反手 `BH=59`，per-action uniform 会形成约 19%/81%
  家族收入；若这不是目标，
  用 family-balanced -> within-family uniform，同时保留每动作 floor；
- 热路径保持 O(envs) vectorized，比较 N1/N73 sim fps、reset/solver p95、GPU memory、PPO wall；
- 关闭 checkpoint/ledger compaction 压力。`4096/73 ~= 56` env/action 只能证明能跑，不能用单 update
  晋级；formal holdout 继续逐动作满足自己的最小分母。

Ball-first 是从可解中心球逐步扩宽，不是冻结问题分布。但扩宽算法本身必须冻结并补齐以下
R1–R9 保护：单臂决定可逆且到期重测；new-band 有独立 eval 配额；样本不足不作决策而是作废重测；
global safety hold 与当前 probed-arm sleep 分层；普通失败有 hysteresis/dwell，zero-tolerance 立即旁路；
training-side 失败加权仍保留 `>=10%` uniform 与 center floor，而认证窗保持冻结混合；并行探 2–3 臂前必须
先完成可逆性、新带配额和 safety attribution，每 env 恰属一个 `probed_arm`。当前实现尚未全部闭合，
`BALL-FIRST-SCHEDULER` 不能因“已有扩域代码”就标 completed。

## 11. READY 迁移账（切换期保留）

这里的 `READY` 只表示“这项交付物已准备好、可被下一版复用”，不表示进入 `main`、可领取、已证明
任务有效或已经 promotion。`main_adoption=BRANCH_CANDIDATE` 是统一默认。

| 旧/当前交付 | delivery_state | decision/evidence | 下一版处理 |
| --- | --- | --- | --- |
| `LATEST-DILIGENCE-SNAPSHOT` | `READY` | `SOURCE_SNAPSHOT_ONLY` | 迁入 source manifest；补 external exact commits/UNKNOWN，不再把 scratch JSON 当证据 |
| `PLANT-AUTHORITY-FREEZE` | `READY` | `ADOPTED_BASELINE` | portable 到 MuJoCo；exact-SKU literal 继续优先于 parkour regex |
| `VENDOR-PUSH-EVIDENCE` | `READY` | `BASELINE_WIRING_ONLY` | 复用幅值/cadence；新增按 strike/follow-through/recovery exposure 分账，不冒充收益因果 |
| `REWARD-SCALE-ECONOMY` | `READY` | `COMMON_BASELINE_ONLY` | style/death/landing/action-rate 账保留；完整 contact/hit/outcome recipe 仍未关闭 |
| `MOTION-PRIOR-PADDLE-TASK` 的 smoke/probe | `READY` | `CANARY_ONLY` | 历史225/318构造、normalizer、三层 wiring 可复用；v3 teacher 已 revoked，v4 三条 ball-free diagnostic lane 已换本地 SHA，但 v4 机械准入失败，且不代签最终 ball-conditioned ABI 或 learnability |
| `OBS-CONTRACT-L7` | `READY` | `LEGACY_194_ONLY` | 保留 layout/SHA/consumer 方法；旧 194 width 不作为 canonical producer |
| `RUNTIME-ASSET-LOADER-V2` | `READY` | `INFRASTRUCTURE` | 直接复用 threat model/loader 收据 |
| `DYNAMIC-READY-PATH-IDENTITY` | `READY` | `LEGACY_IDENTITY_ONLY` | 复用 no-clobber/identity 协议，不复用旧 r4 action pins |
| `LIVE-CONTRACT-MATERIALIZER` | `READY` | `LEGACY_IDENTITY_ONLY` | 复用 materialization/反向核验方法，不冒充新 recipe 已物化 |

## 12. 下一版交付账

本表只记录依赖与完成条件，不给全项目排优先级。

| ID | 状态 | 完成条件 |
| --- | --- | --- |
| `SOURCE-CLAIM-MANIFEST` | `IN_PROGRESS` | 智元/mjlab/unitree/BeyondMimic/SMASH/PACE/ACE 的 revision、文件、证据等级、允许结论、UNKNOWN 可复算；外部源码按资产策略固定 |
| `MOCAP-RACKET-AUTHORITY` | `PARTIAL` | v3 因错长轴 revoked；v4 本地 sibling 已完成 exact `73/73` full-phase kinematic solver/materializer/FK audit、receipt 与 73-action manifest，但尚未 tracked/adopted。Mechanical audit 为 `0/73` admitted：`57/73` 已知硬失败，另 `16/73` 只通过 position/velocity，仍因缺 acceleration/torque-speed/inverse-dynamics authority 而 `UNKNOWN`。关闭仍需 mechanical-safe re-solve、schema-v2 prototype（当前缺 `velocity_contract`）、schema-v4 source-capsule/compiler 无损传递和 content-bound marker→official-site 原始生成收据 |
| `RACKET-PHYSICS-CALIBRATION` | `BLOCKED` | 真实拍子 mass/CoM/inertia 与接触参数仍需测量；只阻塞 calibrated sim2real/真机声明，不回溯否定 URDF-grounded motion retarget |
| `PORTABLE-SYSTEM-CONTRACT` | `IN_PROGRESS` | 便携草案和 MuJoCo core 不被 mocap 阻塞；canonical freeze 才依赖 measured authority。最终 actor/critic purpose-group order/width、两只钟、ball/paddle/outcome/validity、两步 delay history 与分层 SHA lineage 单值化；225/318 是 canary，不预宣告最终宽度 |
| `MOTION-REFERENCE-OBSERVABILITY` | `IN_PROGRESS` | 不新增 motion-intent/ID；teacher trajectory 已表达动作。N2/N3 只查共享容量/串扰。仅当出现相同当前 teacher state、不同必要未来的反例时，才加 short future-teacher preview |
| `CONTACT-GUIDANCE-ABC` | `IN_PROGRESS` | 同一 fixed question 的 `current_lm/analytic_full/analytic_no_velocity/teacher_pos_face_no_velocity/outcome_dense_only` 五种 recipe、显式 validity 与 immutable-tape solver 已实装；纯4096 tape view `.275 ms`、online LM/RNG=0。尚缺真实 Take_061 五 target receipt/tape/report、device 单行 fast path、全 accessor 无泄漏审计、exact dynamic-ready+hold 和 Pod profiler-off 学习比较；纯 sparse C 仍禁止 |
| `CANONICAL-REWARD-RECIPE` | `IN_PROGRESS` | V2 已实改为非腕全身 mimic + 全相位低权 measured paddle + window 内高权 task master，SMASH split window，broad `10/10/5`，adaptive `4/.5/.5`，landing `+6..10`，且 live sigma/EMA state 已接线。静态会计：max motion `3.6575` < target final/initial `4.0296/4.3104` < landing `6`；历史坏误差 final/initial `2.6644/2.8727`，不用近零分母宣传。关闭仍需 physical-contact outcome bridge、全部 event reward rollout-0 安装和实测 tape 条件收入/advantage 健康 |
| `PPO-RUNTIME-RECEIPT` | `BLOCKED` | exact Pod `rsl_rl` source SHA、resolved actor/critic order+width、fresh/resume normalizer、configured/realized std、LR/KL/clip fraction/explained variance/pre-clip grad norm、finite cap 和逐 reward-group income 闭合；旧 194/318 receipt 不代签 final ABI |
| `RESET-TERMINATION-RESUME` | `IN_PROGRESS` | atomic reserve/commit 可复用；关闭 terminated-batch compact reset、phase fidelity termination、follow-through/recovery RSI、完整 mid-episode resume。当前只允许声称 reset-boundary resume |
| `BALL-FIRST-SCHEDULER` | `IN_PROGRESS` | 冻结 generator、initial/max envelope、扩域/回退、RNG、heldout、checkpoint state；补齐可逆重测、new-band配额、样本不足作废、global/arm attribution、hysteresis、uniform/center floor 与并行探臂前置；实际分布自动扩张且 resume 连续 |
| `ISAAC-N1-LEARNABILITY-HANDOFF` | `BLOCKED` | 3x measured N1 + N2/N3 conditioning canary；依赖 canonical measured authority/portable contract/reward/scheduler，满足 §9.1 的真实 hit/legal return、逐分母、安全、resume/export/handoff，不要求 Isaac N73 |
| `MUJOCO-SCENE-CONTACT-HARNESS` | `PARTIAL` | A3 MJCF、桌网 scene assembly、teacher-motion fitted-ball/contact/eval/oracle 积木已有；尚非 policy environment |
| `MUJOCO-SINGLE-ENV-PLANT-ACTION` | `IN_PROGRESS / SAFETY_FAIL` | schema-3 31-D action、implicit total-PD、delay/reset 和100-tick fixed-tape runner 已实装。Exact Take_061 v5 的 d0/d1/d2 均跑满100 ticks，但 tick9 发生 hand↔hip 自碰和 wrist↔table 碰撞；常值q0/root-z反例指向 free-root/PD hold 不稳定，不得绕过 safety |
| `MUJOCO-VECENV-PPO-CHECKPOINT` | `NOT_IMPLEMENTED` | 仓内尚无 native VecEnv/PPO trainer/exact checkpoint/export；可立即并行，但不得标为 runner in progress |
| `MUJOCO-RUN-CONFIG-DETERMINISM` | `NOT_IMPLEMENTED` | single-source RunProfile/覆盖层、Tier-1 exact 和 Tier-2 statistical 收据；native ball-racket/table/net、solref/solimp、aero/spin、CCD/tunneling/event latch 逐项闭合 |
| `MUJOCO-CANONICAL-N1-AUTHORIZATION` | `BLOCKED` | 只有在 final ABI/reward/scheduler/measured authority 与 fixed-tape parity 冻结后，MuJoCo core 才可宣称 canonical 并发 formal N1；FK diagnostic 证据不混报 |
| `MUJOCO-N1-REPRODUCE` | `LATER` | fresh N1 重现 Isaac learnability；warm-start 只作同预算对照 |
| `N73-CATALOG-ADMISSION` | `BLOCKED` | v4 的 73-action manifest 已产生且 receipt-bound，但完整 mechanical audit 是 `0/73` admitted：`57/73` position/stored-or-FD-velocity 硬失败，`16/73` 仅通过这些已知门且仍为 `UNKNOWN`。较早窄口径反例为 `37/73` URDF 超速和 `58/73` 近限位。必须重算并逐件闭合 velocity/acceleration/limit-margin、signed torque-speed/thermal、floating-base inverse dynamics、足底接触/摩擦、自碰/桌净空、fitted-ball，再补 prototype/strict load/alias/family sampling |
| `SPIN-CONTACT-CALIBRATION` | `BLOCKED` | ABI 保留 spin 列但首版 `spin_valid=false`。只有 incoming producer、off-centre friction/restitution/spin transfer、drag/Magnus flight、marker alias/effective-domain 全过后才能 promotion 且付 spin reward |
| `N73-SCALE-COMPACTION` | `LATER` | N73 zero-PPO/1x2/4096x5、O(envs) hotpath、memory/ledger compaction、逐动作 starvation 门 |
| `MUJOCO-N73-BALL-FIRST` | `LATER` | 完整 73 从 fresh recipe 训练，自动扩域，逐动作/侧别/题格 denominator 和 heldout，不从 N5 checkpoint 续 |
| `DR-RESTORE-HEALTH` | `LATER` | 同底盘 DR 作为 baseline 接入；mass/CoM/PD/noise/history 每轴过 hold/teacher-to-hit/task/safety/receipt 门 |
| `DUAL-EVAL-PROFILES` | `LATER` | deterministic ranking 与 noisy vendor-play 分开，不能混报 |
| `INDEPENDENT-PHYSICAL-EXAM` | `LATER` | independent MuJoCo/vendor/hardware；physics/contact/spin 未测格写 `未测`，不能靠 analytic return promotion |

## 13. 关闭条件

本文只有在以下事实全部成立后才可标 `completed`：

1. `origin/main` 已采用实测 racket authority、单值 portable ABI、完整 reward、ball-first scheduler 和
   N1->N73 顺序；
2. Isaac N1 最小可学门通过并产冻结 handoff；
3. MuJoCo native trainer、fixed-tape parity、N1 fresh 复现通过；
4. 73 件 admission/alias/scale/compaction 门通过，N73 训练有逐动作/逐侧/逐题格证据；
5. independent physical exam 完成；缺数据的 formal 格仍明确 `未测`，没有被平均数掩盖。

在此之前，当前最诚实的总体状态是：**设计方向已收敛；ChingMu 实测 racket 已完成
本地 schema-v4 `73/73` current-site full-phase 运动学重定向、50 Hz 物化和独立 FK 闭环，三条
ball-free diagnostic lane 已换 v4 SHA；reward 也已实装 long-axis、split-window、coarse、adaptive fine、
precision 和 adaptive-state resume，静态收入顺序在 adaptive 初始/收紧两端都是 motion < target < landing。
但 mechanical audit 已确认 `0/73` admitted：`57/73` 有已知 position/velocity 硬失败，另
`16/73` 仍因缺 acceleration/torque-speed/inverse-dynamics authority 而 `UNKNOWN`；较早窄口径的
`37/73` 超速与 `58/73` 近限位是其中的机理证据。因此 v4 只是未跟踪的 diagnostic
kinematic sibling，不是 training-ready teacher。机械安全重算、prototype/source-capsule 正式消费链、
physical-contact outcome bridge、最终 ABI 与 two-step delay history、PPO/reset receipt、Isaac 球任务 learnability、
MuJoCo trainer 与 N73 逐动作闭环尚未完成。保留的是 9-D `desired_at_contact`；被否决的是额外
18-D synthetic motion intent。旧 Stage1 的不完整结果不再是新系统 concern。**
