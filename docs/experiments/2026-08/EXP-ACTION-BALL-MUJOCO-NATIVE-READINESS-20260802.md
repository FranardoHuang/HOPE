# EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802 — ActionBall 下一版系统与 MuJoCo 原生训练准备账

- 状态：`in_progress`
- 阶段/轴：ChingMu-73 动作库、Ball-first 自动扩域、Isaac 最小可学门、MuJoCo 原生训练
- 集成小目标：用一个自然动作在 Isaac 验证可学性的同时并行完成 MuJoCo trainer；共享 bundle 冻结后两引擎 N1 并行，主训练在 MuJoCo 直接扩到通过机械准入的完整 73 动作
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 本 successor 最高证据等级：`E1`；历史 negative-control 另有 `E3` 诊断，不传递为新系统 E3
- 创建日期/最后复核日期：2026-08-02 / 2026-08-03

共享缩写按[术语与人话对照](../../DEFINITIONS.md)解释。本文件是下一版系统的**依赖、证据充分性和
版本迁移账**，不是全项目优先级队列。当前采用 setting、认领和算力顺序仍只认
[`origin/main` 的 `NOW`](../../NOW.md)。本分支同时保留 `L194` legacy fixed-question
194/318-D、`H225` historical ball-free 225/318-D、已 supersede 的 `A225-proto/C225-proto`
225/318-D prototype，以及当前 fresh `A211/C211` 211/319-D successor。A211/C211 从
actor 删除 raw `teacher_base_now_world(15)`，在 actor/critic 末尾新增原子
`task_valid(1)`；WAIT 内 task/base-goal/两只钟归零，任务 reward 与相应 denominator
也不记账，平衡、非任务全身 mimic 和 safety 仍工作。它们是新 ABI，不接受任何
225/318 normalizer 或 checkpoint。2026-08-03 的同宽 v2 又把 actor `[0:15]` 冻结为
world localizer pose+linear velocity `[0:12]` 与 pelvis/body-frame IMU gyro `[12:15]`，无 projected
gravity、无 world angular velocity 重复列；A211/C211 actor normalizer/trainability 均换 v2，critic
内容/normalizer 保持 v1。final N73 宽度仍未冻结。这些配方和本文均是候选更新，
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

2026-08-03 termination 历史增量：当时 native diagnostic ledger 已绑定
`joint_actual_forbidden` 的 actual-q predicate。每个 control step 后使用 MuJoCo
`model.jnt_range` 与 Isaac-consistent exact-zero bounds tolerance，并 sticky 保留 tick 内 substep 触边；非有限/无效区间或 raw hard-edge 状态触发；
tilt→height→joint actual 的 reason order 与 sticky latch 已冻结，Isaac config/callable 双源码
SHA 漂移均拒绝。Host 聚焦回归 `45 passed, 8 skipped`。这是中间收据；robot/table、qdes 与
compact reset 的后续状态以 §9.2/§12 当前账为准，phase/recovery 与 Reward/PPO 仍未闭合。

2026-08-03 qdes termination 历史增量：该时点 native ledger 继续绑定 Isaac
`pre_clamp_qdes_forbidden_zone` 的 ActionBall projection-mode 语义。源配置明确使用
`joint_pos_limits`、`margin_rad=0`、`margin_fraction=0.02` 和 finite projection；所以有限越界
proposal 被投影并保留 transition，不能误写成 reset，只有有效 pre-clamp affine qdes 含 NaN/Inf
才触发。冻结 reason order 为 tilt→height→joint qdes→joint actual，双源码 SHA 漂移仍拒绝。
Host 三组聚焦回归 `45 passed, 10 skipped`；正常 PPO `step()` 仍在 physics 前 fail closed。
这是中间收据；robot/table/compact reset 后续状态看 §9.2/§12，phase/recovery、Reward/PPO/
save/resume/export 仍未闭合。

2026-08-03 fixed-question long 最终裁决：exact `e9a27247` 的旧 `L194` A/B 已分别在
`498/810` updates 停止，累计 `14,509/18,026` strike opportunities 仍均为 `0 capture / 0 legal
return`，exact-strike position error 反而由约 `.45–.47 m` 恶化到 `.89–.90 m`。它们不能证明
目标拍速或 cheap B 可学；A-fast/C long 未发。B successor 因没有可执行 partial-field ABI 已 defer，
不能继续占正式路线。

当前改动方向大体是正向的，但旧 TODO **尚未形成闭环体系**。缺的不是再堆一批 feature，而是把
下面这条唯一因果链写成可验收系统：

```text
外部来源/本地事实取证
  -> measured-racket data + URDF/MJCF official-site authority
  -> kinematic retarget admission + independent mechanical admission
  -> engine-independent 便携合同草案 + MuJoCo core scene/runner/PPO  [现在并行]
  -> 最终 ABI + 完整 reward + ball-first scheduler
  -> shared portable bundle freeze
       |-> Isaac 真人对拉录制单拍 N1 recipe canary + 冻结 handoff
       |-> MuJoCo canonical authorization + fixed-tape parity + fresh N1
  -> 73 件逐动作 admission/alias/吞吐准备 [与 N1 并行；正式发 N73 才等待 N1]
  -> MuJoCo N73 + ball-first 自动扩域
  -> online incoming producer + event scheduler + no-reset recovery/next-shot curriculum
  -> continuous heldout / stateful export Gate3B
  -> 独立 physical exam / vendor / hardware
```

核心选择如下。

| 问题 | 裁决 | 理由 |
| --- | --- | --- |
| Isaac 是否仍是主训练目标 | `REVISE` | Isaac 只负责证明最终 MDP/Reward 可学和合同可移交；长期训练和 N73 转到 MuJoCo，减少训完后再跨物理引擎搬策略的风险 |
| 动作规模 | `CANDIDATE_ROUTE N1 -> N73` | 先用一条来自真人对拉录制、逐件准入的单拍 measured N1 证明完整配方可学，随后把当时逐件通过 admission 的动作一次全上；该 clip 不是 no-reset 连续对拉证据。不恢复 learned N2/N3/N5/N8/N12 阶梯。额外独立 N1 或 N2/N3 只在 N73 失败时诊断跨侧泛化、共享容量或动作串扰，不是 promotion 前置门，也不新增 motion-intent/ID。在 §9.1 的数值门仍为 `UNSET` 时，本路线不得称 formal |
| 训练 Stage | `REJECT 手工换 Stage` | 从 rollout 0 就使用相同网络、optimizer、观测字段和 reward weights；所谓阶段只描述后续事件 reward 尚未有分母/收入的时间区间 |
| 问题分布 | `REVISE “冻结分布”` | 冻结生成程序、字段、initial/max envelope、扩域/回退规则、RNG 和 checkpoint state；实际采样分布必须随 ball-first curriculum 自动扩张 |
| full-phase 与 window-only | `ADOPT BOTH WITH WEIGHT SEPARATION` | 非腕全身 mimic 全程保留；measured paddle 的低权 position/velocity/signed-face/long-axis 全程保留来学专业动作；window 内 ball-conditioned `desired_at_contact` 是更高权的 task master，不用硬 mask 制造指导空洞 |
| 三层 paddle reward | `ADOPT STRUCTURE / RECALIBRATE SCALE` | coarse、adaptive fine、precision 分别解决冷启动、随学习收紧和触球精度；外部证据支持结构，不支持当前具体权重 |
| 智元 A3 setting | `ADOPT AS PRIMARY BASELINE` | 同底盘、动态全身运动对 plant/DR/delay/push 是强先验；reward、reset 分布和乒乓接触数值仍须按本任务证据裁决 |
| mjlab / 宇树 / BeyondMimic | `ADOPT SELECTIVELY` | 可固定 imitation 经济、MuJoCo manager/VecEnv 结构、机器人 DR/正则先例；不能代签球拍、触球、落点、旋转或 N73 成功 |
| Sony ACE / PACE / SMASH | `ADOPT TASK STRUCTURE` | SMASH 支持 task/style 和 adaptive sigma，PACE 支持 predicted+true outcome 及数值锚，ACE 支持 miss<hit<return 与 landing/spin conditioning；三者的算法经济不同，不能逐字搬权重 |

## 2. 四个维度必须分开

旧账把训练阶段、动作数、验证 Gate 和课程扩域混在一起，容易产生错误依赖。下一版固定为：

1. **动作规模**：候选路线是 `N1 -> N73`。一个来自真人对拉录制的单拍 measured N1 学会后，直接启动当时
   逐件通过 admission 的完整动作集；中间不训练正式 N2/N3/N5/N8/N12 policy。额外小动作集
   只在全库失败后作为定位共享容量/串扰的诊断，不构成 promotion Gate。
2. **Reward eligibility phase**：所有 callable 和 weight 从第一步安装。早期即使零接触，
   hit denominator 仍是已结束的 eligible swing，必须报 `0/C`；只有尚未形成 valid achieved
   outgoing flight 时，outcome denominator 才可为零。contact-target 的分母是有效击球窗 sample。
   这些只是同一次训练里不同事件还未 eligible 的时间区间，不是 operator 开关。
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

0803 新 A3-P1 交付已经内容寻址保存，但不能被“主要只加左夹爪”这句话直接晋级为现役模型。
它保留了旧31轴和右拍局部挂载，却同时引入9个未耦合夹爪轴、body/mesh 大小写漂移、缺失
collision mesh、夹爪 mount 冲突以及右肘/右髋/躯干/双臂 plant 变化。故本轮把它定为**未来
successor 的 raw source authority**，而不是今天 A211/C211 的 runtime authority；现役模型和历史
receipt 不原地改写。normalized 31-D Isaac asset 与 MuJoCo identity v3 必须在授权/耦合/mount 闭合后
另立 lineage，并重做拍心全局 FK 和动力学/碰撞 parity。

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

当晚实际选中的 `Take_061_unit04_BH` 还有更强的逐帧结论：v4 共57帧，全相位最大
site-center/point-velocity/signed-face/long-axis 残差分别为 `.21378 mm / .00607 m/s /
.02670 deg / .02148 deg`；第0帧约 `.038 mm / .014 deg / .012 deg`。因此这一条上
既不是只在 strike window 才对齐，也不是第一帧拍子就错位。后来出现的
teacher frame0 与 physical reset 差 `.120455 m / 89.596 deg` 是“动态老师帧不能直接
作为静态出生姿态”，不是重定向失败。

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
`physical_ball=false` 且 `actor_obs_contract=null`，因而不能发 final N1。该时点没有启动 GPU/训练/namespace；
下方记录随后建立的新 diagnostic-only 链。

### 4.1 当前运行真值

`HOPEPingPongActionBallA3VendorV2` 目前是本地 branch-candidate YAML，不是 active runtime authority。正式
ActionBall launcher 仍绑定 VendorV1；三条 225/318-D launcher 是明确 ball-free 的历史/诊断 canary。
`ACTION_SET_CONTRACTS` 仍未注册 final N1/N73，V2 继承 `physical_ball=false`，且没有 final actor contract。
所以当前不存在一条诚实的“VendorV2 单动作打球 policy” launch 命令；把 task 名强换进旧
launcher 会同时搞错 observation、full-body scope、physical outcome 和 receipt identity。

### 4.2 2026-08-03 当晚 launch 边界

本轮**不启动 VendorV2 formal N1**，也不把 `Take_060_unit09_BH` 换名后强发。但为了不让
formal blocker 阻止 learnability 诊断，已显式使用
`allow-mechanical-unknown-diagnostic`为 `Take_061_unit04_BH` 新建一条 no-clobber 诊断链。这个例外
只允许 simulator learnability，不允许 canonical/N73/hardware promotion。

先将 yaw-aligned full seed 的每个足底支撑点最小法向力设为 `20 N`，避免旧 LP 把 CoP 选在
支撑三角形边界。当前 exact PhysX nominal hold 实测 `1.2 s / 60 policy / 240 physics`
通过：双脚 contact ratio=`1.0`，无 terminal，root 最低z=`1.0672 m`，最大倾角=`.01808 rad`，
最终最小 hard gap 为 waist-roll `.028525 rad`。dynamic artifact/hold receipt file SHA 分别为
`ab6b7e41…8069` / `c8b92a28…bb19`。这关闭了本 diagnostic 的 physical-birth/hold blocker，
不关闭 motion acceleration/torque-speed 的 `UNKNOWN`。

下列是 **SUPERSEDED pre-split-ready predecessor**，只保留历史工件追溯：

- prepared core SHA=`353a56c0…12ba8a8`；
- one-row tape/report SHA=`6f0ad062…beb69c` / `27930d5c…4a553`，base-question
  SHA=`adb93bee…dbc19`，reset online LM=`0`；
- final `current_lm/analytic_full/outcome_dense_only` bundle SHA=
  `93ad5f21…f786a8 / d1c62f55…5c6b288 / 06e68047…180d4b`。

它们当时仍是 actor/critic=`194/318`、`analytic_virtual_ball_authoritative_physx_disabled`的
`PASS_DIAGNOSTIC_ONLY`；它们可以跑 zero-PPO/`1x2` 来验证发射器、reward 和有限学习步，
但不能答案真实拍球接触、合法上台或 final ABI。当前 split-ready 真值是
§4.3 的 `current_lm/analytic_no_velocity/outcome_dense_only`、新 tape/bundle SHA 和
`PENDING / 未测`收据槽，不再从这组 predecessor 启动。

不改变 [`origin/main` 的 `NOW`](../../NOW.md) 统一优先级的前提下，当晚可安全并行的
是下列前置工件与显式无授权的有限 smoke：

- 用 soft-limit/velocity/acceleration 和 authoritative torque-speed/torque 门重解 v4，从
  `16` 条已知 position/velocity-pass 候选中寻找第一条真正 mechanically admitted N1；
- 冻结 final purpose-grouped ABI、two-action delay history 和 physical-contact outcome eligibility，产出
  VendorV2 的 resolved reward/policy receipt；
- 增加隔离 namespace 的 VendorV2 diagnostic launcher，先做 `d=0` 的 zero-PPO/
  `1x2`/`4096x5`，只有上述 teacher/ABI/outcome 门闭合后才允许 fresh 学习 canary；
- 继续不依赖 canonical N1 授权的 MuJoCo core scene/single-env/action-delay/fixed-tape/
  VecEnv-PPO 接口工作，但不把它误报为 formal trainer 已可训。

### 4.3 Isaac diagnostic launch receipt 占位（结果待实际运行）

下表只是预注册的收据槽，不是已运行或已通过。每条必须由 exact Pod 进程在自然
exit 后回填；没有收据文件和 SHA 时一律保持 `PENDING / 未测`，禁止根据命名空间、
launcher 输出计划或存在的 checkpoint 路径推定 PASS。

| diagnostic arm | p/v/face mask | launch receipt | runtime result | 证据边界 |
| --- | --- | --- | --- | --- |
| `current_lm` | `111` | `PENDING` | `未测` | fixed-question 当前 LM target 语义基线；仍非真球 outcome |
| `analytic_no_velocity` | `101` | `PENDING` | `未测` | 当前仍先求完整 analytic target 再 mask 拍速；只回答速度 target 是否必要，不证明省 solver |
| `outcome_dense_only` | `000` | `PENDING` | `未测` | 当前 bundle 仍是 analytic virtual-ball/PhysX-disabled 诊断；未来 C 语义必须是 valid-actual-contact-conditioned dense forward outcome，不是 sparse-only |

2026-08-03 的 pre-launch 证据不改变上表的 `PENDING`：source `90baeba5` 已固定
prepared core/tape=`c5212ce9…0370 / 22052606…9e66` 和三条 final bundle
`a223d4c9…71734 / d3c2632c…a516b / 589db839…0418a`；exact Pod 聚焦测试
`35 passed`，shared reward 零 PPO 物化也已成功。但 policy recipe r1 在构建时因
`body_ang_vel_w=2.77555756e-15 rad/s` 的静止四元数派生舍入残差而
fail closed，并未进入 PPO。该 namespace 永不复用，旧进程不人工发 signal。

为继续今晚的 diagnostic smoke，runtime 桥的可证范围被锁死为：仅
`action_ball_diagnostic_split_ready_teacher=true`，仅 teacher-start `body_ang_vel_w`，
仅 `max_abs<=1e-14 rad/s`，且首三帧的 `joint_pos/body_pos_w/body_quat_w`
原始数组必须是 native float32 且 C-order row bytes 完全一致；float64 sub-ULP 运动在
转换后消失、或 `+0/-0` 仅数值相等，都不取得豁免。它不覆写原始 motion；hold getter 仍返回 literal zero，播放时
原字节保留。任何 joint/body-linear 非零、超阈 body-angular、非静止前缀、
短 clip 或 formal mode 仍 fail closed。长期不用 threshold 修老资产，而是从 producer
生成新的不覆盖 motion 版本：SO(3) 差分 stencil 两端逐位相同时直接写
literal zero，并重签 motion/bank/core/tape/bundle 链。

每份收据的最小必填字段为：exact source commit/checkout、Pod/GPU/namespace、完整 argv 与
natural exit code、final bundle/tape/reward/policy/backend SHA、resolved actor/critic term order + width、
`teacher_source`、ball/contact authority、action/observation delay、`rsl_rl` source SHA、逐 update wall-time、
finite checkpoint/normalizer count、reward-group eligibility/income、hit/return 与 hard/table/nonfinite 分母。
这三条是 `diagnostic_unauthorized` 目标语义对照；即使 smoke 成功，也不关闭 final ABI、
physical-ball outcome 或 mechanical admission。

## 5. Reward 体系与数值裁决

### 5.1 完整层级

```text
R = R_body-style(non-wrist whole body, full clip)
  + R_measured-paddle-trajectory(teacher_now, low weight, full clip)
  + I_target_valid * I_strike * R_contact-task(desired_at_contact, window)
  + I_valid_actual_contact * R_hit
  + I_valid_actual_contact * I_valid_achieved_outgoing_flight
      * R_predicted-outcome(net/landing/spin)
  + I_eligible_achieved_flight * R_true-outcome(net/landing/out/timeout/spin)
  + R_regularization/safety
```

- **动作模仿组**=`R_body-style + R_measured-paddle-trajectory`：非腕全身模仿保持动力链；实测拍子
  teacher 在全相位低权跟 position/point-velocity/signed-face/long-axis，因此击球腕虽从 generic
  body-position/orientation/velocity mimic 释放，仍会通过刚体拍 teacher 学到引拍、加速、触球、
  随挥和手腕 twist。“释放”是移除另一个可能冲突的手腕 body owner，不是不学手腕。
- **接触目标组**=`R_contact-task`：仅 A/B 在 `target_valid ∧ strike_window` 样本上学习所需触球状态；
  C 没有此组，不能把 dense mimic 或 outcome 伪装成 oracle target。
- **真实击球组**=`R_hit`：分母是 installed/started/closed swing opportunity，成功事件只能是实际
  selected-rubber contact；不得与 target-window 收入合账后声称已经学会 hit。
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
A: B_motion^eligible < B_contact_target^eligible; all routes: B_motion^eligible < B_hit^eligible < B_table_outcome^eligible
```

同时另报 `B_G^rollout`（把未触发记零的真实 rollout 平均），监视 dense motion 是否在优化经济里长期
淹没后两层。四套分母必须独立报告：contact-target=`target_valid ∧ strike_window` sample；
hit denominator=`eligible closed swings`，actual selected-rubber contact 是 numerator/event；predicted
outcome denominator=`actual contact ∧ valid achieved outgoing flight`；true-outcome denominator=所有
应闭合物理结果的 eligible achieved flights，legal landing 是 numerator。miss、timeout、net-fail 和 out
必须留在失败分母，不能通过条件化删掉。早期没有 eligible flight 时 outcome 分母可能为零，但 closed
swing 后即使零 hit，hit denominator 也不为零；contact-target 在有效窗内已经可以有收入。不得用 target 收入隐藏零 hit。对应分母为零写
`未测`，一旦有效就必须有非消失 shaping 和上述条件预算。

早期若球题与原 clip 一致，可令 `desired_at_contact == teacher_contact_nominal`；扩域后两者可能不同，
必须分别入账。所以“full-phase 与 window-only 都要”的固定规则是：measured teacher 拍子全程低权，
window 内 `desired_at_contact` 高权主导。`teacher_contact_nominal` 还可用于可行 answer set 内的
nearest-teacher 选解。两者不是字段冲突，但必须用数量级分离保证 task 大于 style。
C 没有 `desired_at_contact`，其窗内仍保留低权 measured-paddle/body mimic，只有 actual contact 与
valid achieved outgoing flight 后才由更高价值的 dense forward-outcome/上台结果主导；不能给 C
偷偷接回 A 的 target reward。
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
[SMASH 消融](https://arxiv.org/html/2604.01158v1#S6.SS1)支持，但绝对数值要用 N1 和全库逐动作
tape 实测定。

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
5. 分开验证 A 的 `B_motion < B_contact_target`、所有路线的 `B_motion < B_hit < B_table_outcome`，
   不把 contact-target 与 hit 合账；同时检查含零
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
| action delay | 首轮固定 `d=0`；未来 fresh `DELAY-L1/L2` 分别测 `{0,1}` / `{0,1,2}` | `DEFER / FRESH-LAUNCH BOUNDARY`；不写 in-loop scheduler，d=2 前先闭合 history/alias |
| 六轴 velocity push | 首轮关闭；未来幅值可沿用同底盘 baseline，cadence 放慢到 `10..30 s` | `DEFER / FRESH-LAUNCH BOUNDARY`；目标是每 episode 命中击球窗期不超过约 `.1` 次，不照搬 `1..3 s` |
| mass/CoM | torso/末端/拍子优先，测量值优先于随意 `±20%` | `CANDIDATE`；惯量一致性、hold、hit/safety 门 |
| friction | 不把 PhysX joint friction 数字直接搬成 MuJoCo `frictionloss` | `DEFER TO MUJOCO CALIBRATION` |
| obs noise/history | joint-pos `±.01` / joint-vel `±.5` 本体感噪声 day-1 保留；task/racket/time 噪声关闭 | 前者不改题目支撑集；后者会降低任务可观测性，须 fresh 单轴验证 |
| reset noise | 首轮全零；终态候选按三个 fresh 档位到位姿 `±.1`、速度 `±.2`、关节 `±.15 rad` | `DEFER / FRESH-LAUNCH BOUNDARY`；不与 reward 首轮混变量 |
| motion 速度 | 73 条自然动作优先原速，当前 `speed_scale_range=[1,1]` | `ADOPT`；禁止把整条 clip 统一拉到“最高速”；日后重定时须另过动力学门 |
| torque-speed | 使用 signed 转速-净力矩曲线/热包络，不把两个独立上限当矩形 | `BLOCKED ON VENDOR CURVE`；当前线性三角只作保守排序，不据此删 action |
| Motion-VAE | 等新一轮高质量/更完整动捕后再启 | `DEFER`；当前先用 teacher-trajectory-conditioned shared policy 建立基线，不加动作 ID |

“采用 baseline”表示根据同底盘/多动态运动证据选择首发 setting，不等于每项对乒乓表现的因果最优已
证明。低风险项不必逐轴做科学 A/B；任何导致 contact denominator、teacher-to-hit、hard safety 或
吞吐失真的项都 fail-closed/回滚。

### 6.1 “更真实”不等于“所有难度第一步全开”

尽调收口为三道闸：支撑集（是否改变“哪些动作能击球”）、终止率放大器和扰动
cadence。startup 级 plant 异质性最安全，reset 级次之；per-step 只允许零均值观测噪声，不允许
每步改动力学。但这个框架不能把旧 `-72`/`sigma=.075` 经济直接套到 A211/C211：
当前 base death 是 `-6` post-dt，corrected learnability 臂为 `-.6`；A 有 broad Cauchy 和宽
adaptive start，C 用 `sigma=.15` 拍心-球心 Cauchy。因此旧 69% break-even/500x 结论不是当前事实，
必须按 exact reward arm 重算。

当前首轮不改既有 startup plant DR：robot material、joint-default `±.01`、torso CoM、link mass
`±15%`、Kp/Kd 和本体感噪声保留；delay/push/reset noise/task sensor noise 全关，`physical_ball=false`。
这个裁决使 reward/ABI 修复保持单变量，不代表终态 sim2real 永不恢复这些轴。
没有一手 ablation 证明 `[0,2]` delay 必须线性 ramp。
BeyondMimic 论文没有讨论 action/observation-delay curriculum；官方 G1 使用普通
`ImplicitActuatorCfg`，通用 delayed actuator 默认 `min=max=0`，tracking `CurriculumCfg`
为空。因此“BeyondMimic 证明 40 ms 不能从头学，必须线性加”是无一手证据的
二手推断。SMASH 支持 task-distribution/adaptive-tolerance 课程；PACE/ACE 可从头带球与校准
delay/noise，但还依赖 history、predictor 或 SAC/replay/HER，不能单独外推到当前 PPO N1。

延迟不写 in-loop scheduler。`DELAY-L0/L1/L2` 是三个 fresh launch/resume-boundary recipe，
分别为 `d=0`、`d∈{0,1}`、`d∈{0,1,2}`；每次升档都换 argv/recipe SHA/namespace，不在同一
optimizer 运行中热改 support。当前 actor 只观察一个 previous action；`d=2` 前必须增加两步
raw-action history、recurrence 或显式已知 lag，否则队列中第二个 pending action 是隐藏状态。
当晚 contact-guidance 五臂为了单独回答
target semantics，统一用 action/observation delay=`0`、no push、no wide DR 和 fixed question tape；
延迟是优胜 recipe 之后的独立轴。

相同 mindset 下，下列“更真实”轴不得混成一次冷启动：

| 轴 | rollout 0 | 后续扩展门 |
| --- | --- | --- |
| nominal plant + 低风险 gain/mass/CoM/noise support | 采用，且整个 support 从开始可见 | hold/teacher-to-hit/task/safety 分层健康；禁止 startup 与 reset 双重抽同一轴 |
| 来球位置/速度/时间/落点 | 只用 ball-first 可解中心域 | checkpointed band curriculum，有可逆回退与独立 new-band 分母 |
| spin/off-centre contact | 列存在但 `spin_valid=false`，reward=0 | 飞行、摩擦/回弹、旋转传递和别名门全过后单独 promotion |
| push | cadence/幅值可从 rollout 0 存在，但不强求立即覆盖 strike window | 按 pre-strike/strike/follow-through/recovery 暴露分层，任一层 safety 恶化停车 |
| delay | learnability 臂用 `d=0` | `DELAY-L1/L2` 各自 fresh lineage；不在 loop 内扩幅，d=2 前先闭合 history/alias |
| CCD/减半全场 dt/贵重 contact reporting | 不一次全开 | 通过单轴 matched throughput + tunneling/contact truth 门再采用 |

第一版同时明确 defer：`spin_valid=false`、未标定 off-centre spin/contact 不付款；push 需按
pre-strike/strike/follow-through/recovery exposure 统计，必要时扩张 exposure 而非改变 plant SHA；
ball/question distribution 始终由冻结 ball-first 规则扩张。这样保留真实场景目标，同时不把不可观测、
未标定或完全稀疏的困难混成一次冷启动。

将上表压缩成可执行的尽调裁决：

- **ADOPT**：rollout 0 使用实测校准且 episode-fixed 的 compact plant/sensor support；保留
  dense near-miss/contact/outcome 学习支架、failed-region 采样与 uniform/center floor；reward tolerance
  可按已冻结误差规则从 coarse 收紧到 fine。
- **DEFER**：`{0,20,40} ms` action FIFO、strike-window push 暴露、spin/off-centre contact、
  完整比赛来球 tail、动态 reset-plan replay、CCD/全场减半 dt；分别等 ABI、实测标定、
  physical truth 和单轴吞吐门。
- **REJECT**：把 `40 ms` 写成 BeyondMimic/PACE/ACE 的论文推荐；在 rollout 0 同时开启
  所有 realism 且用最大强度；把 PACE 的固定分布或 ACE 的 event-table replay 写成
  performance-adaptive curriculum；用 `2x/4x` 噪声/延迟冒充“更保守”的 realism。

这个先后顺序是结合 primary evidence 与 ActionBall POMDP/安全边界的工程推断，不冒充
论文作者给出的通用配方。

## 7. 真球是否会让训练很慢

当前严格答案是：**已有小批 CPU physics-only 结果显示不会因“多一个球”就爆炸，
但4096-env、GPU/VecEnv/PPO 同负载税仍未测。**

- 现有 `4096x5 ~= 6.7 s/update` 基准是 `physical_ball=false`，最大段仍是
  `solver_solve_many`/reset；它不能给 ball tax。
- 仓内曾有 `physical_ball=true, 4096 env` 构造/finite checkpoint，但没有 matched wall-time。
- TTRL/PACE 公开配置用 4096 个动态碰撞球，说明“不可训练”不成立；其仓库没有公开本机可比 steps/s。
- 一个球只增加一个动态刚体；真正可能昂贵的是每 substep aero/root read-write、reverse RK4 发球、
  paddle/table scan、contact reporting、CCD 或把整个场景 dt 减半。
- 当前 ActionBall `PhysicalBall` 关闭 collider/CCD，用代码驱动拍球/桌弹，因此也不能代表原生接触成本。
- MuJoCo 已有一条 native physical-ball scene 小批 benchmark。在相同 single-env runner 上，
  N1/N8/N32/N64 相对无球增加约 `6.593%/5.743%/5.594%/5.703%`。它只量了 CPU
  physics-only 的边际 ball tax，没量4096、PPO、aero/spin/CCD 或全套 reporting，不得线性外推。

发移植前的 matched benchmark：

| Arm | 相对上一臂只增加 | 归因 |
| --- | --- | --- |
| `PHYS-A` | 无球 | 基线 |
| `PHYS-B` | 停放球、无 collider/callback | 刚体/状态缓存税 |
| `PHYS-C` | flight/aero/serve，impulse off | RK4、aero、root read/write |
| `PHYS-D` | code-driven paddle/table scan | substep FK/扫描 |
| `PHYS-E` | 原生 collision，CCD off、无 reporting | contact solver |
| `PHYS-F` | 单次净接触力读取 | 合理 reporting |
| `PHYS-G` | CCD；另开一臂单独减半 dt | 分开 CCD 与全场 substep 税 |

每臂在 `1/512/4096 env`、同 GPU/commit/tape/solver 下交错运行；10 update warm-up、至少 50 update
profiler-off 计时，reset-free 与固定 reset-count 分层报告。记录 scene build、GPU memory、physics、
collection、PPO、serve/reset/callback、p50/p90 wall 和 env-steps/s，并做 RNG/reason/counter/safety/
reward/obs parity。选 CPU、mjlab Warp 或其他 backend 只能由 A3+桌网球的同工作量结果决定。

## 8. Canonical portable contract

当前旧 canary 包括 `H225` historical ball-free dense-paddle 合同和当晚 fresh diagnostic 使用的
`L194` 合同，它们都不是最终 ball-conditioned N73 合同。最终版本必须在
N1 开始前一次冻结：

| 唯一身份 | actor / critic | ball/task authority | 可复用 normalizer/checkpoint | 当前授权 |
| --- | --- | --- | --- | --- |
| `L194` | `194 / 318` | legacy solved-target + component mask；`000` 没有 incoming-ball actor state | 仅本身份内部 | historical diagnostic only |
| `H225` | `225 / 318` | ball-free；desired-contact 是 teacher copy | 仅本身份内部 | historical canary only |
| `A225-proto` | `225 / 318` | `[212:221]=desired contact p/v/face`，仍含 raw teacher-base 15 | 只在本身份 fresh lineage 内 | superseded diagnostic history；不再发射 |
| `C225-proto` | `225 / 318` | `[212:221]=incoming ball-at-contact p/v/spin`，仍含 raw teacher-base 15 | 只在本身份 fresh lineage 内 | superseded diagnostic history；不再发射 |
| `A211` | `211 / 319` | 删 teacher-base 15；保留 desired-contact 9；末尾 `task_valid=1` | A211-owned fresh lineage | current fixed-question successor；frame-0 exact + oracle32 未过 |
| `C211` | `211 / 319` | 删 teacher-base 15；保留 incoming-ball p/v/spin 9；末尾 `task_valid=1` | C211-owned fresh lineage；不可复用 A | current fixed-midpoint successor；C oracle/PPO 未过 |
| `FINAL-N1/N73` | width unfrozen | varying-ball/task、两步 delay history 与完整 outcome | 必须新建 lineage | proposed only |

下文禁止用裸 `225` 代表一种语义；相同宽度绝不意味着合同、normalizer 或 checkpoint 相容。

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
当晚 194-D 兼容合同里还保留一列 legacy `swing_type=-1`；在 N1 中它对所有样本是常数，
不携带任何动作信息。它只为了不破坏当晚旧 consumer 而保留；canonical N73 删除该列，
不用一个换名后的 type/intent 欺骗合同。

最终有序分组固定为：

```text
actor  = robot/achieved -> teacher/reference -> incoming-ball/task target -> clocks/validity -> causal history
critic = privileged robot/teacher -> same exogenous task -> achieved outcome/eligibility
```

`teacher_base_now_world(15)` 在 A211/C211 中整块删除，不再换成 residual15。这是有意的
信息选择，不是为了省算力：policy 已看到 actual base、`q_ref/dq_ref`、teacher paddle-now/
at-hit 和自己 achieved paddle；对当前专业单拍 clip，这些量才直接决定专业动作与击球差。

旧反例依赖 split-ready 允许 physical birth 与 teacher frame 0 的 root-Z/tilt/joints 不同；
fresh A211/C211 reset 强制从 measured teacher frame 0 的 root/q 且零速度开始，该反例不再存在。
当前裁决不使用 robot-centric teacher-base residual 作为默认列。base 空间适配由
“老师 nominal contact 与当前球题 contact/ball 的差”表达，不需要 policy 另外知道老师
pelvis 离自己多远。若未来出现具体 alias 反例，应先用 teacher paddle/body future preview 定位，
不默认恢复 raw world root。
SMASH  motion/robot anchor 与 task-space `p_hit/v_hit/time`，没有把 teacher root twist 当乞乓落点适应通道。

incoming ball 至少含预测到触球时的 `position3/velocity3/spin3 + time/valid/age`；achieved paddle
是当前实际 site 的 `position3/point-velocity3/signed-face3`；teacher 是当前 reference 与 nominal
contact baseline；desired contact 是 A 路线或未来另立合同的 B 路线下 planner 给的接触要求；landing/spin 是目标出球
结果。A211/C211 已用 `task_valid` 区分 WAIT 与 TASK_ACTIVE，但尚无 estimate age 列，不能冒充
final varying-ball ABI；final
N1 是否保留 `spin_valid=false` 的列必须随完整 ABI 单独冻结，spin 无 authority 时 reward 不付款。normalizer 是按上述固定
列顺序保存的 mean/variance/count；checkpoint 还必须保存 actor/critic、optimizer、normalizer、
delay queue、curriculum/eligibility 和 RNG。

“宽度”是 actor/critic 输入的标量列数，不是隐藏层 `[512,256,128]`。当前 fresh
A/C 是211/319：`225 - teacher_base15 + task_valid1 = 211`；critic 原本就没有这个 actor-only teacher-base
block，所以是 `318 + task_valid1 = 319`。历史 `L194` 是194/318，`H225` 是225/318；
actor v2 的前15列不再复用旧聚合 base-state producer：`[0:12]` 是 localizer world
position/orientation6D/linear velocity，`[12:15]` 是 body-frame IMU gyro；projected gravity 不在 actor。
lineage v2 将完整 ordered layout、这两段 exact slice/producer 与 content SHA 一起冻结，因此 pre-IMU
同名211也不能消费 v2 normalizer。最终宽度在 A/C contact 路线与两步 delay history 闭合前不宣告，且
相同宽度但顺序/来源不同也不是同一 ABI。

字段的人话含义固定为：

- `incoming ball`：policy 在触球时刻预期会面对的球位置/速度/旋转/时间；actor 只用因果可获的
  prediction，critic 可有 privileged truth 但不能泄漏给 actor。
- `achieved paddle`：sim/live FK 给出的当前实际拍位、point velocity 和 signed face，用来告诉 policy
  “我现在真的做到哪了”。
- `teacher_now` / `teacher_contact_nominal`：专业动作当前相位与自然触球时的拍状态，表达动作本身，
  不是 ID。
- `desired_at_contact`：若 A 路线或未来另立合同的 B 路线启用，planner 对当前来球/落点所要求的触球拍状态；它与
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
spin-aware `desired_at_contact`，也不是纯 outcome-only policy。当前只实现 A/C matched
comparison；B 保留概念行但已 defer，不能被算作第三条已可运行路线：

| 路线 | actor 获得什么 | 计算与证据 | 本轮裁决 |
| --- | --- | --- | --- |
| A：完整 contact oracle | 固定中点 N1 只给 `desired position/velocity/face`；teacher nominal 显示老师自然触球状态，二者之差就是 task adaptation | task→出球→接触可行集；SMASH/HITTER 只证明无旋、无摩擦的闭式 A-lite，不证明完整 spin/friction inverse | 主 oracle arm；必须事件级缓存、批量解析/LUT/固定轮数，env 热路禁 data-dependent LM |
| B：部分 contact guidance | 概念上给 position/face，速度由 policy 学 | 若 position/face 仍由 A 求出则**不节省 producer 成本**；若直接用 teacher position/face 才真正便宜但可能与新来球不相容 | `DEFERRED / NO EXECUTABLE ABI`；A/C matched 结果不足时才另建带明确 validity 的 B 合同，不得把 A225 的 velocity 置零冒充 B |
| C：无 contact target | 固定中点 N1 只给 incoming ball-at-contact `p/v/spin`；台中点是环境常量，不重复作为 task 输入 | 不做 desired-contact 反解；在 nominal strike tick 用实际拍心-球心 Cauchy 距离保留 miss 梯度，actual selected-rubber contact 只用于 outcome eligibility | 当前 C211 只有距离项与落点项；无独立 hit bonus、无 desired-contact 奖励，对方侧出界不超过同质量合法落台的一半 |

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
必须重新验算。固定中点 N1 不再假装 A/C 是同一 observation content：`A225-proto/C225-proto` 分别注册独立
actor width=`225` 的 ABI，前212维 robot/teacher/achieved paddle 相同，`[212:221]` 在 A 中是 task-derived
desired contact p/v/face，在 C 中是 incoming ball-at-contact p/v/spin，末尾 base station 与两只钟
相同。相同宽度只控制 MLP 规模，不允许共享 normalizer/checkpoint 或偷换 term source。未来若落点/
出球旋转变为 policy 输入，再新建 versioned task-conditioned ABI；不能为了未来泛化给今晚固定台中点
C 塞常量 task。

性能不能靠论文名背书。目前可比的 exact `L194` fixed-tape 长训为每 update
`512 env x 24 = 12,288 env-step`：

| 轨迹 | 平均 wall/update | update/min | env-step/s | 对 A 的速率变化 | 可裁决性 |
| --- | ---: | ---: | ---: | ---: | --- |
| A，完整 target，`n=498` | `3.125775 s` | `19.195` | `3,931.19` | baseline | exact 学习轨迹，但 `0/14,509` capture |
| B，cheap target，全轨 `n=810` | `3.023811 s` | `19.843` | `4,063.75` | `+3.37%` | 效应小；不足以支付第三条 ABI |
| B，与 A 同时段 `n=519` | `2.983061 s` | `20.114` | `4,119.26` | `+4.78%` | CI 跨零；B 仍 defer |
| 旧“C proxy” `n=5` | `2.447672 s` | `24.513` | `5,020.28` | 表面 `+27.70%` | **不可用**：不同 checkout/长度/reset，且 actor 没有 ball p/v/spin，不是 C225 |

因此本轮可以直接删去 B 作为主路线，但 **A/C 速率差仍是 `未测`**。C 的主要目的
是验证“给 contact target 学”与“给球状态由 outcome 学”两套架构，不是预先承诺
固定球 consumer 会快多少。相同固定题下 A/C 都必须 `reset_inverse_solve=false`、
`online_solver_calls=0`；两者的 PPO wall 比较不得偷混“一个算新题、一个读 tape”。
动态 novel-question producer 的 inverse/LUT 税用另一个 `seconds/4096 novel questions` Gate 表达。

用 A 的 profiler-off 分解，collection=`3.034175 s` (`97.07%`)，PPO=`.091600 s`
(`2.93%`)。所以即使完全删掉 PPO，理论也只有 `1.030x`；删 15-D teacher-base 的
全 PPO MLP 算术上限约 `1.01%`，端到端实际更小，因此 teacher-base 不按速度删。下一轮
真正有价值的切分是对 current checkout 的 physics/table/termination/reset/receipt/observation/
Reward/contact scorer 分段 profiler；没有分段证据前不填“预计可砍多少”。

历史 `4096 env x 5 update` profiler-on 诊断总计 `33.499 s`（约 `6.700 s/update`），
它每 update 处理 `98,304 env-step`，是上面 512 轨迹的 **8 倍工作量**。所以原始秒数
不可直接对比：旧轨迹约 `14,672.24 env-step/s`，吞吐反而是当前 A 的 `3.73x`；
这不证明 4096 新配方能 boot/能学。当时 solver span `16.367 s`，占 collection
`49.71%`、占总 wall `48.86%`；两个分母不得混用。减去该 span 得到的
`3.427 s/update` 只是理想下界，不是新实测。Pod CPU fixed-tape microbench 中，每批4096 proposals
的 LM4/8/12 分别约 `6.71/15.15/23.18 s`；当前 analytic 实现约 `.157 s` lean、`.954 s`
default。另一批4000同模型 replay 为 analytic `.415 s` 对 LM12 `12.979 s`，analytic 的 admission
`100%`、landing error p50/p99/max `.128/.641/.946 mm`，而 LM12 为 `96.65%` 和
`1.662/7.831/19.592 mm`。这些是事件批次 microbench；solver 只在 reset/question construction
触发，并非每 physics step，所以不能直接把单批时间当 update 税。

历史 `L194` 的五种 target 曾各自离线生成；新 A/C fixed-center N1 则各生成自己的 versioned
[`immutable_tape`](../../DEFINITIONS.md#action-ball-immutable-tape) 单行缓存。这里不是“参数配错”，
而是适用范围必须写准：该 source 在 reset 只取同一中心题，online LM/inverse solve 为零，同时也把
curriculum 固定在中心档。它正好适合今晚早期尚无成功信号的 fixed-center finite run，避免每次 reset
重复解同一道题；它不能冒充可扩域题源，也不能进入 full-curriculum long。

扩域前必须切换到 [`banded_question_bank`](../../DEFINITIONS.md#action-ball-banded-question-bank)：
按 exact domain-level/action/profile/solver identity 预生成缓存块，reset 只索引当前档的已解行，升档即
换块，缺块 fail closed 而不是回退 online solver 或沿用中心题。这个 bank 是 expanding/full-curriculum
long 的硬前置，**不是** fixed-center `oracle32` 或 `4096x5` finite run 的前置。因此 rollout 热路
A/C 都可以是零次反解，但 A 的 novel-question producer 仍须按新档离线运行，C 不需要。
纯 tape `4096` reset view 实测中位数 `.275 ms`，
`online_lm_calls=0`、`physical_rng_draws=0`。但当前 runtime adapter 仍会经 sampler 且逐环境
materialize compatibility receipt，该路径约 `1.736 s/4096`；所以在 device 上直接
expand/index 单行的 fast path 未闭合前，不得宣称完整 reset 已 O(1) 或
`online_sampler_calls=0`。LM vs analytic 训练对照只比 target 语义；producer cost 用另一份
benchmark 比，不再污染 PPO wall time。历史五个 `L194` target receipt 还必须共同绑定相同
teacher-rate/`t_hit`/cycle/pre-swing-wait，不能只比 base-question SHA 后就声称同题同钟。

新 hot-path 审计还找到三个比删网络列更值得做的点：

1. 当前 `_emitted` 历史会无界增长，并在 receipt 路径重复 copy/serialize 全历史；应改为
   compact high-water/counter/digest，保留 replay 证据而不保留无界 Python 对象。
2. actor/critic 当前每 step 对同一 9-D task snapshot 做多次 host receipt scan；`4096x24`
   可达 `786,432` 次 row check/update。应在 command 边界生成一次 device snapshot，actor/critic/
   Reward 共享。
3. C211 的 desired-contact target 分支恒为无效，应在坐标变换、acos 和细分 metric 之前
   直接返回预分配零/不注册该 term，但必须保留 selected-rubber contact、拍心-球心距离、
   flight/net/landing 路径。

三项都要在 fixed tape/RNG/reason/counter/safety/reward/observation/checkpoint parity 后才能报加速。

正式性能选择拆成两个不同单位的 Gate，禁止混账：

- **离线/事件 producer**：同机同批 `seconds / 4096 novel questions`，并同时过 feasibility、
  landing/net/contact residual 与安全 parity；未来动态来球只按实际新题/refill 事件频率摊销。
- **fixed-tape consumer**：强制 `online_solver_calls=0`，同 checkout/tape/seed/GPU、相同 reset
  strata 的 profiler-off total wall 与 envstep/s 才是主结果；profiler-on 只归因 physics、table/
  termination、reset/receipt/H2D、observation、Reward/contact scorer 和 PPO。

旧 `analytic A <=.35 s/update` 没有 fixed-tape 热路对象，删除为选择门。A/C 用 10 update warm-up
加至少 50 measured updates 的同卡串行 `A→C→C→A` 交错；`4096x5` 只验证 scene/finite scale，不能替代性能门。
B 未注册，不再写成可直接并行发射的路线。

C211 的 strike 距离 manager weight 现冻结为 `220`；在 `dt=.02 s`、`sigma=.15 m`
时单拍峰值为 `4.4`，距离 `.075/.15/.30/.45/.90 m` 时收入分别为
`3.52/2.20/.88/.44/.119`。因此相容命中区间里保持 motion 静态上界 `3.6575 < 4.4 < 6`
legal-landing 下界，远区又不会直接变成零。这是配置/reward-landscape 会计，真实
eligible income 与 policy gradient 仍须 Pod 训练验证。

当前 fresh ActionBall 仍是 N1-only、ball-contact ABI 未冻结；formal launcher 也仍绑定 VendorV1。
N73 的实际 blocker 是 ball-conditioned producer、A/C选择、normalizer/checkpoint/backend consumers
与逐动作机械准入，不再是虚构一个 fixed-width motion intent。

### 8.2 PPO runtime receipt

静态 reward 会计不能代替一次真实 trainer 收据。canonical N1 发车前必须记录 exact Pod checkout、
`rsl_rl` source SHA、resolved actor/critic order+width、fresh/resume normalizer state、configured/realized
`init_noise_std=.02`、`noise_std_type=log`、`entropy_coef=.01`、optimizer/adaptive-KL 设置和 finite
iteration cap。前200 iteration 至少监视 `mean_noise_std`，前500 iteration记录 LR/KL、clip fraction、
explained variance、pre-clip grad norm、advantage/return tails 和逐 reward-group eligible income。
这些是 launch/health receipt，不是因为 reward 变了就一起调 entropy/std 的额外消融。

当前 learnability 四臂所需的 `qdes_projection_penalty_weight` 已采用为唯一窄 override：默认
`-5.0`，允许区间仅 `[-5,0]`，`-5` 是最强剂量、`0` 是显式零权对照；历史 `-20` 在这条
measured-VendorV2 路线明确拒绝。每个值必须改变 effective-Reward/hard-contract lineage，且不允许
通过通用 Hydra reward override 顺带改函数或其它 term。该实现只是发车能力，Pod 学习结果仍为`未测`。

`FOUR-ARM-LEARNABILITY = CODE_IMPLEMENTED / ORACLE TABLE-HIT BLOCKED`。A225 的 producer、318-D
critic、fresh normalizer identities、Gym leaf 和 dedicated launcher 已实现；四臂 exact Pod
materialize 都已将实际 composed Reward 反向读回并绑定，不再只信 planned SHA。第一轮
固定题局部因果矩阵全部绑定 A225 同一 contact target；A/C 算法比较是另一矩阵，
不能把 ABI 与 learnability 轴混在四臂里：

| arm | ABI | soft weights: death / qdes-limit / projection / joint-limit | guard | PPO/LR | 目的 |
| --- | --- | --- | --- | --- | --- |
| `L0-corrected-metrics-fixedlr` | `A225-proto` | `-30 / -.5 / -.5 / -.5` | `metrics_only` | fixed `lr=1e-4` | 最低阻断 reference |
| `L1-legacy-penalty-fixedlr` | `A225-proto` | `-300 / -5 / -5 / -5` | `metrics_only` | fixed `lr=1e-4` | 只测旧负收入剂量 |
| `L2-corrected-phase-fixedlr` | `A225-proto` | `-30 / -.5 / -.5 / -.5` | `phase_gated` | fixed `lr=1e-4` | 只测 reference guard |
| `L3-corrected-phase-adaptive` | `A225-proto` | `-30 / -.5 / -.5 / -.5` | `phase_gated` | adaptive-KL，initial `lr=.001`、desired KL `.01`、clip `.2`、epochs `5`、minibatches `4` | 只测 LR collapse |

四臂共同冻结同一 admitted teacher/tape/seed、A225 actor/critic ABI、hard table/fall/qdes/actual
termination、其它 Reward、network、budget 与 stop gate。这里没有 `qdes=0` 臂；`0` 只保留为未来
显式对照能力，且历史 `-20` 继续拒绝。

2026-08-03 fresh closure 审计对四个 outer roots、递归25个 JSON、28个 distinct repo-relative
references 得到 missing=`0`、SHA mismatch=`0`。实际 materialize 随即暴露运行时合同矛盾：A225
leaf 按本节设计从 rollout 0 启用 position/velocity/normal 三路 adaptive sigma，旧 fixed-question
finalizer 却一刀切要求三旗标全 false。当前修复只允许 dedicated `action_ball_a225` 使用三旗标
全 true 且 `adaptive_sigma_source='ball_exact_strike'`；C225、L194 与其它 immutable-tape 诊断继续
强制全 false。host training/task/A-launcher 聚焦回归=`66 passed,15 skipped`。这只恢复配置与
运行时合同一致。exact Pod `2e743932` 随后通过完整 closure/Git pin，却在第二层 materialization
profile fail closed：旧 `measured_vendor_v2_n1_static_v1` 仍期待 flags=`000` 和固定 success widths
`.075/.5/.262`，实际 A225 的 command 按既有 lockstep 语义从 rollout-zero `.5/3/2.1` 开始并随
adaptive sigma 单调收紧。当前新增独立 `measured_vendor_v2_a225_monotonic_v1` 身份，只接受 flags=`111`、
start/success widths=`.5/3/2.1` 与同一 min/max schedule；旧 L194 static profile 原样保留。host
reward/train/A-launcher 回归=`323 passed`。fresh exact Pod materialize→recipe→oracle32 仍是4096发车前置。

2026-08-03 runtime materialization 收据：L0/L2/L3 实际 Reward SHA 均为
`d263513d…e41fcb`，L1 为 `dbb0de09…f2794`；均解析出 42 个实际计价 term，并反读
L0/L2/L3 的 soft weights=`-30/-.5/-.5/-.5`、L1=`-300/-5/-5/-5`。然而首次 L0
oracle32 在 `0/32 episode` 时 fail closed：launcher 用简化 policy envelope 得到
`f344e2db…df55`，trainer 要求的 exact dynamic-ready PPO recipe 却是 `3a3a8f4a…c6f9b`。
这是 policy recipe materialization 缺失，不是学习失败；失败 namespace 已保留、exact
PGID 已受控 TERM，其余 oracle 暂停。launcher 现已把因果链修成
`materialize Reward (0 PPO) -> recipe exact dynamic-ready PPO (0 PPO) -> oracle32 -> scale4096 -> long4096`：
`recipe` 绑定 trainer 产生的 artifact path/file SHA、semantic policy SHA、dynamic-ready binding、arm/
lineage/content seal，后续阶段反向重验；不硬编码已观测的 `3a3…`。旧 reward-only materialize
receipt 只允许作严格 legacy 输入，其 planned policy SHA 被明确忽略。host launcher 回归
`42 passed`。`smoke/probe512/long512` 只保留为失败定位支线，不再阻塞或代签
4096；`long4096` 只接受同臂 `scale4096` 恰好 5 update、finite save、natural clean exit
和全 lineage seal 的 terminal result，不接受仅 launch-accepted receipt。

exact Pod `454416b9` 曾将 L0/L1/L2/L3 四臂 `recipe` 阶段全部 materialize 为
clean/0 PPO，并暴露初次 `env.reset()` 缺失。修复后 exact Pod `299145e9` 因 train
source pin 改变而没有复用旧 receipt，又对四臂全部 fresh 重做 clean/0-PPO
materialize+recipe。L0/L1 oracle 都完整跑32回合，teacher-qdes preclamp max error
`5.96e-8 rad`、零 projection/nonfinite/soft-limit intrusion；但两臂均在每回合第15
control step 触发 `robot_hit_table=32/32`，因此 `single_stroke=0/32`、exact-strike 和
capture denominator=`0`。这是真正的 oracle 安全/可达性失败；下段只给出离线归因假设。
`scale4096/long4096` 没有启动。后置 validator 也已改为正确解析 projection 的
RewardManager `weight=-1` 与 callable `params.objective_weight`，并将 raw reward SHA 绑回已
重验 materialization；该 receipt 修复不改变碰桌门的失败。

随后在 exact `299145e9` tape/guard 上做了两条只读几何重放。两者都指出
`pre_swing_wait_s=0.712376` 会让 motion clock 在首个 `0.30 s` 内保持 frame 0；但 reset 写入
physical-ready 后，oracle 立即提交 teacher frame-0 qdes。physical-ready 相比 frame 0 的
root-Z 高约 `0.177 m`、姿态少约 `29.6 deg` tilt，最大 joint discontinuity 为
`2.243 rad`。一条重放在其解释的实际 tape
`base_spawn=(-0.192232, 0.285279, 1.068400)` 下得到 `left_ankle_roll_Link` 对 keepout 的 exact
OBB-vs-AABB SAT overlap；另一条使用不同 world-root 解释时不能复现，所以目前不能把左踝写成
live offender。teacher frames 39--41 的 right-hand proxy conservative-AABB-only、SAT-negative
命中同样保持为独立待验证假设。合法修复必须先给 oracle32 开启并导出已有 first-hit attribution
ledger，然后在 actual tape base 上把 physical-ready、hold、ready-to-teacher swept transition 和全部
teacher frame 加入 admission；重解 base/question/tape 或重定向 lower-body transition，不能手改 base、
关闭 feet/racket 或放掉 table termination。

下一次 oracle32 已补齐 live first-hit sidecar：只在 finite teacher-qdes oracle 路径启用既有
`table_contact_attribution_diagnostic`，每回合导出 episode/control step、motion frame、
component/body、obstacle、blade-or-proxy、exact-vs-conservative 和 selected owner-body 的 world
pose；现有接口没有 physics-substep ordinal，故显式写 `null + unavailable_reason`，不推测。
它在原 dense ledger 记账后只读复制，不改变 terminal/Reward/observation/RNG，host 联合回归=
`203 passed`。exact Pod clean `254f115b` launcher=`46 passed`；用该 validator 离线重验旧 L0/L1
raw oracle 时，projection 的 manager `-1` 和 objective `-.5/-5` 均正确解析，最终仍到达
`robot_hit_table=32/32` 驱动的 oracle acceptance failure，故旧 parser 假失败已排除。

exact Pod `513a1592` 的新一轮已经让 A225 materialize 和 exact policy recipe 都在
`0 PPO` 清洁退出，随后 oracle32 又暴露一个更早的初始化顺序问题：
first-hit exporter 在 action term 首次 `process_actions` 之前启用，command 上尚无
table-attribution schema，因此在 `0/32` 前 fail closed。修复后的唯一新动作是：
initial `env.reset()` 后、任何 `env.step()` 前显式让 action term 解析/prepare 同一份
full-table pose guard，并先验证 `full_table_assembly=true` 和
`attribution_diagnostic=true`，然后才开 exporter。这不执行 policy step、不计算或
改写 termination truth。host 联合回归=`218 passed`；仍需新 exact-source Pod fresh
materialize→recipe→oracle32，不能跨 source SHA 复用旧 receipt。

对“现在 setting 是否能学”的裁决是：旧 L194 已实测不可学；新 A225 reward/ABI 本身是
**有理由可学、但当前 plant 初始轨迹不可发车**的设置。它从 rollout 0 同时安装 upright/
body+paddle mimic/contact-target/hit/landing，通过 event eligibility 自然形成
`balance -> mimic -> hit -> landing`，不中途换 Stage。固定 N1 令 `reset_inverse_solve=false`、
delay/push/noise/wide DR 全关、动态 ready 从 frame 0 开始、原速 Take-061，且用 analytic virtual
ball 先验证梯度；这正是早期最小难度。它也因 `physical_ball=false` 而只是 learnability
canary，不证明 selected-rubber 真接触。当前 oracle32 已证明 ready-to-teacher 过渡会先碰桌，必须修好并重验；只有 oracle32 证明 teacher-qdes 在 live plant 可达，
然后 smoke/probe/long 出现实际的 mimic/contact/outcome 收入和安全趋势，才能回答“能学会”。

发车实现必须逐臂 exact 写出全部 soft weight、PPO 参数、ABI/source SHA、termination union、
`max_iterations` 与 continuation/stop gate；不能让未列出的轴暗变。所有臂都需
weight-independent projection probe；未来若启用 `qdes=0` 对照，即使 reward callable 因零权被裁掉，也记录 observed/
projected sample、逐关节 count/distance 和 hypothetical unweighted penalty。暴露分母为零时该轴只能写
`未测/INELIGIBLE`，不能判胜负。

`oracle2` 只验证 live auto-reset、ledger、lineage 和无残留进程，不判定 teacher 可追踪。RL 四臂前还需
code-owned `oracle32`：预注册 single-stroke denominator、exact-strike p/v/face 阈值、capture/reject、
reference-only 与 hard termination 上限、projection/soft-limit exposure 和 unknown attribution 上限。
固定题四臂即使通过，也只授权 `LOCAL FIXED-QUESTION DIAGNOSTIC`；canonical same-run 仍需 A/C
producer、final ABI、actual-contact outcome bridge 与 checkpointed ball-first scheduler。

### 8.3 Reset、termination 与 exact resume

`canonical_ready`（动作数据能否提供准备位）与 reset policy 必须分开。reset 从逐环境 O(env) 工作改为
只处理 terminated batch；恢复 phase-gated fidelity termination、follow-through buffer 和
recovery-only RSI，明确禁止 mid-swing 把机器人/球“空投”到新状态。每个 reset reason、phase、动作、
side 和球题单独计数。旧 Gate A/B `9–11/6–8 s/update` 单位与 workload 未绑定，正式退役；
旧 profiler-on `6.7 s/update` 也不能直接晋级。新的 `4096x5` scale pass 只要求同 claim 自然退出、
恰好5个 finite PPO updates、零 hard/table/nonfinite、全程 PID/UUID receipt 和 `>=8192 MiB` min-free；
速度结论另用10 warm-up+至少50 measured 的 exclusive profiler-off workload。
exact `ad4ba3f4` 的历史 4096 B 在 scene/USD bootstrap 后 1808 s 无 PPO，同 commit A
又在首次 reset 因 birth-stratum contract 退出，两个失败不能合并成单一根因。
2026-08-03 最新裁决是不再因此把 512 放在正式训练前置：A225 主链直接
`oracle32 -> scale4096(4096 env, 5 update, completion-wait) -> long4096(4096 env, 1000 update)`。
`smoke/probe512/long512` 仅在 4096 失败时做定位，不能作为 long4096 predecessor。
只有 scale4096 自身 finite/natural clean exit 才能发 long4096；若失败，再用
`512 -> 1024 -> 2048 -> 4096` 梯子定位，而不是先默认降规模。

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

1. 先选一条通过当前准入门、来自真人对拉录制的单拍自然动作做 N1；其同钟实测 racket teacher 映射到
   `official_racket_site` 并通过逐动作残差门。它从 rollout 0 使用最终球/台/网场景、portable
   ABI、完整 reward recipe 和 ball-first scheduler。该 N1 学会后直接把当时逐件通过 admission
   的动作一次全上；不插入按动作数递增的训练阶梯。
2. `1x2`、`4096x5`、save->cold-load、finite export、normalizer、action-scale/delay/qdes exact。
3. 预注册短学习预算；在冻结中心 holdout 上，相对**实测 racket teacher**的 full-phase/exact-window
   paddle error 下降，并出现真实 physical hit 与 legal return 的学习，而非只看 motion mimic、
   FK-derived self-consistency 或总 reward。
4. 按[逐拍账本](../../interfaces/action_conditioned_ball_first_contract.md#5-attempt-账本)记录
   proposed/admitted/installed/started/closed/legal-return/safe-nonreturn/unsafe，连同动作/侧别、
   paddle/body income、hard/table safety；零分母写 `未测`，不跨动作平均。
5. 至少机械演练一次自动扩域、回退、checkpoint->resume，ABI/reward SHA 不变。
6. 产出冻结 handoff bundle：contract/plant/reward/physics bytes、checkpoint、fixed tapes、oracle 与性能预算。

当前 `N1-PASS-THRESHOLDS = NOT PRE-REGISTERED`，所以“出现一次 hit/return”不能授权 N73。正式发
N1 前必须把下表 `UNSET` 数值写进 code-owned judge 与 launch claim；先看到结果再填无效：

| Gate | 冻结统计 | 当前硬边界 |
| --- | --- | --- |
| 独立性 | fresh seeds、training/heldout tape 隔离 | `fresh seeds >= 3`；checkpoint 不跨 seed |
| 分母 | `P/A/I/S/C/L/F/U/X` 每 seed 最小数 | `UNSET`；不足统一写 `未测`，不得跨 seed/action 平均补齐 |
| mimic/contact 进步 | heldout full-phase/window p/v/face error 相对 init 的 effect size 与 bootstrap CI | `UNSET`；三类误差分别过门，不能只过总 reward |
| hit/return | actual contact rate 与 legal-return rate 的 heldout 95% lower confidence bound | `UNSET`；必须高于 fresh-init 同题上界与预注册绝对 floor |
| reward economy | 同 opportunity 下 motion/contact/hit/outcome/aux typical+p95 income | 主层级不倒置，target income 不代签 hit |
| safety | table/hard/nonfinite/unknown-attribution | formal holdout `0` hard/table/nonfinite；unknown 上限=`UNSET` |
| resume/export | cold-load 与 finite export 的逐 tick parity | exact ABI/normalizer/action/qdes/clock/reason SHA；任何 mismatch fail |

`4096x5` 仍只作 scene/finite scale smoke；上表学习门使用预注册长于 5 update 的 budget，二者不能互代。

如果直接 N73 失败，再用额外独立 N1 或短 N2/N3 canary 区分“某动作本身不可学”与“共享网络容量/
动作串扰”。这类诊断验收逐动作 teacher/task 结果，不做 intent swap/shuffle/zero，不新增 ID，
也不作 N73 checkpoint 起点；它不是 N1→N73 的前置门。

历史 Stage1 V2 `605 tests + 1x2 + 4096x5` 只证明当时那份不完整配方的构造、吞吐和九项
reward 活；旧无球 motion-prior long 也只是 historical negative control。它们不是对完整 one-run
设计的 concern，也不需要再跑一个手工 Stage 来“解除”；正确动作是从 rollout 0 把缺失的球任务/
outcome/scheduler/reward 全部加回。当前 successor 已有本地 v4 `73/73` measured-racket **运动学**
retarget/materialize/FK-audit 闭环和新 reward static/counterfactual Gate，但机械准入已发现超速/限位
反例，且尚未进入完整球任务的 exact Isaac boot/学习门。

### 9.2 MuJoCo 顺序

- **MuJoCo core 现在并行做**：pin mjlab/runtime，实现 MJCF/scene/plant、action/delay、deterministic
  reset、batched VecEnv、PPO、checkpoint/save-resume-export、ball-table-net contact harness、独立 reward/evaluator
  oracle 和 fixed tapes。scene/contact/teacher-eval 和 single-env plant/action 均已是 `PARTIAL`。
  当前 successor 已把 observed selected-rubber resolver、C-lite scalar reward、normal VecEnv step、finite
  PPO shell 与 reset-boundary cold-load parity 接通；但 motion/balance 仍显式为0，且拒绝尚未移植的
  RESET_WAIT/task-valid，所以只能称 C-lite plumbing/learnability smoke，不能称完整 ActionBall trainer。
- **formal MuJoCo N1 另受 canonical authorization AND 门**：final portable ABI、admitted teacher、
  pinned sim contact/physics profile、full termination/reset、reward/evaluator parity、trainer/save/resume、
  run determinism 与 fixed-tape cross-engine parity 缺一不可。开发期 robot-FK recipe 可用于 diagnostic
  engineering，但用独立 `teacher_source`/recipe SHA，不能代签 formal measured N1。
- **Isaac 与 MuJoCo N1 在 shared bundle freeze 后并行**：fresh MuJoCo N1 是主结果；Isaac actor-only
  warm-start 只作同预算对照，critic/optimizer fresh。Isaac 学会不是 fresh MuJoCo N1 的硬前置。
- **N73 准备现在并行**：逐动作 mechanical admission、manifest/alias、zero-PPO scale/compaction
  不等待 N1；formal N73 learning 要等 Isaac canary 与 fresh MuJoCo N1 两个定量门都过。

`b8355f23` 的 exact Pod 验证路径为
`/workspace/franco/mujoco_vecenv_b8355f23_integration`，MuJoCo/PyTorch focused suite 为
`42 passed in 15.19 s`。N8 实例构造用时 `11.926 s`；同一份 `3`步 action tape 两次
diagnostic rollout 为 `27.72/25.16 ms`，trace shape=`[4,8,76]`，全部 finite、逐元素重复且两次
trace SHA 相同。这仅证明8个 CPU MuJoCo core 能按明示76列布局 deterministic reset/rollout；
它不是 `4096`、没有 PPO update，也没有 throughput 外推权。

该 adapter 故意使正常 `step()` 在触碰 physics **之前**抛出
`PPO_BLOCKED_MISSING_REAL_REWARD_CONTRACT`，且明确禁止 optimizer update、checkpoint 和
cold-load resume。其 successor `deec4a52c758b1f173436d4522e3e13e7ccb7bfd` 已增加 strict
physics-substep contact-event ledger 和 tape-timeout exact latch；exact Pod clean worktree
`/workspace/franco/actionball_mujoco_deec4a52_20260803` 的三组联合测试为
`42 passed in 15.24 s`。这只关闭这两个具名合同；其他 formal termination predicates 仍
fail-closed，reward 和 PPO 仍被禁止。

其 successor `41411c3b6a6ef3ad03c2cba41370e84709066d8d` 再从 Isaac
`HOPEDeployParityTerminationsCfg` 绑定 `base_fell_tilt` 和 `base_too_low` 两个
strict/sticky/order-aware exact subset；源语义或源 SHA 漂移就 fail closed。clean Pod
`/workspace/franco/actionball_mujoco_41411c3b_20260803` 三组聚焦回归为
`48 passed in 15.71 s`，4096 次 cached blocker-receipt 调用合计 `.446 ms`，
receipt SHA=`353382b4…3789`。这仍不包括桌/机器人碰撞、joint actual/qdes hard edge、
phase fidelity、terminated-batch compact reset，因而不是完整 termination union。

exact Pod `7135d5ce` 已继续验证 `joint_actual_forbidden`、`joint_qdes_forbidden`、robot/table
43-component guard、canonical owner-frame、decimation4 与 hard reason order，四组
`72 passed in 17.44 s`。current worktree 又实现 per-env done latch、terminated-row compact reset、
pre-reset terminal observation/post-reset next observation、caller-owned ledger、异构 question lineage
与 independently-recomputable v3 receipt。Host 当时结果为 `62 passed, 13 skipped`；这些
component paths 之后已在 exact clean Pod `ebe963f5` 的当前组合 suite 中执行，结果为
`108 passed, 0 skipped, 0 failed`，不再写作 Pod 组件未测。仍未测的是用户可执行 C-lite runner 的
真实两次 PPO update、save/cold-load receipt。剩余 blocker 是 phase/recovery、完整 canonical Reward、
formal save/resume/export 与4096规模；component PASS 不会关闭这些格。

关闭整个 reward blocker 不是把零 reward 接给 `rsl_rl`，而是继续补齐 remaining formal
termination、teacher + `official_racket_site`、tape 的 position/velocity/face validity、legal
actual contact→achieved outgoing flight→net→landing event/reward parity。只有这些语义闭合后，
才能实现 PPO/save/resume 并量 `1/512/4096` matched workload。

MuJoCo trainer 关闭清单必须逐项有 code-owned receipt：final actor/critic ABI、normalizer 与两步
history consumer；controlled runner/factory；real Reward/done `step()`；optimizer/checkpoint 全状态、
cold-load 与 export；`1/512/4096` matched workload；ball contact/aero/Magnus 与 independent outcome
evaluator。当前 C-lite 已有自己的 diagnostic trainer/checkpoint，不再是“normal step 一律 fail closed”；
但 upstream full-recipe runner/export、WAIT/mimic、完整 termination/reward parity 和4096 workload 仍未闭合。
故“完整移植完成”的答案仍是**没有**；可以上 Pod 做 `1 env x 2 step x 2 PPO update + cold-load`
真实 plumbing smoke，不能把它写成 canonical N1 或4096主训练。

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

### 10.1 单拍 N73 不是连续对拉

本文件的 `legal_return` 只表示**当前这一拍**合法过网并落在对方台面，不等于 no-reset rally 已成立。
单拍 N73 之后仍有一条独立的连续时序链：

```text
online incoming-ball estimator/producer
  -> atomic reveal + action/reference selection event scheduler
  -> current shot without teleport/history reset
  -> follow-through/recovery/ready carry-state
  -> next-shot variable lead-time and sequence curriculum
  -> continuous heldout + stateful export/runtime parity
```

T0/T1/T2 若属于同一 checkpoint lineage，recovery callable/weight 必须从 rollout 0 安装，
T0/T1 只是 `I_recovery_eligible=0`；随机下一球先作为环境和 deadline 时序进入。只有 T1
证明单拍能力在恢复/下一拍上失败后，才能让已安装的 recovery 项取得 eligibility。
若届时要新增 callable 或改 weight，T2 必须是 fresh recipe/new SHA/new lineage，不得称同一
N1->N73 run。两种情况都不能靠 shaping 掩盖 selector/reveal 或 carry-state bug。
连续账除逐动作外还要分 `prev_action -> next_action`、reveal lead-time、sequence position、streak length
和实际 selector 支持的 transition floor；不要求穷举 `73^2`，但未覆盖的转移不能被每动作总数掩盖。
export 必须做 no-reset sequence 逐 tick parity，只有真正 sequence boundary 才能清 actor history、delay
queue 或 episode-local recurrent state。冻结 policy normalizer 是全局 model state，sequence boundary
也不得清零或重估。

## 11. READY 迁移账（切换期保留）

这里的 `READY` 只表示“这项交付物已准备好、可被下一版复用”，不表示进入 `main`、可领取、已证明
任务有效或已经 promotion。`main_adoption=BRANCH_CANDIDATE` 是统一默认。

| 旧/当前交付 | delivery_state | decision/evidence | 下一版处理 |
| --- | --- | --- | --- |
| `LATEST-DILIGENCE-SNAPSHOT` | `READY` | `SOURCE_SNAPSHOT_ONLY` | 迁入 source manifest；补 external exact commits/UNKNOWN，不再把 scratch JSON 当证据 |
| `PLANT-AUTHORITY-FREEZE` | `READY` | `ADOPTED_BASELINE` | portable 到 MuJoCo；exact-SKU literal 继续优先于 parkour regex |
| `VENDOR-PUSH-EVIDENCE` | `READY` | `BASELINE_WIRING_ONLY` | 复用幅值/cadence；新增按 strike/follow-through/recovery exposure 分账，不冒充收益因果 |
| `REWARD-SCALE-ECONOMY` | `READY` | `COMMON_BASELINE_ONLY` | style/death/landing/action-rate 账保留；完整 contact/hit/outcome recipe 仍未关闭 |
| `MOTION-PRIOR-PADDLE-TASK` 的 smoke/probe | `READY` | `CANARY_ONLY` | 历史 `H225` 构造、normalizer、三层 wiring 可复用；v3 teacher 已 revoked，v4 三条 ball-free diagnostic lane 已换本地 SHA，但 v4 机械准入失败，且不代签最终 ball-conditioned ABI 或 learnability |
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
| `PORTABLE-SYSTEM-CONTRACT` | `IN_PROGRESS` | 便携草案和 MuJoCo core 不被 mocap 阻塞；canonical freeze 才依赖 measured authority。最终 actor/critic purpose-group order/width、两只钟、ball/paddle/outcome/validity、两步 delay history 与分层 SHA lineage 单值化；`H225` 只是 canary，不预宣告最终宽度 |
| `MOTION-REFERENCE-OBSERVABILITY` | `IN_PROGRESS` | 不新增 motion-intent/ID；teacher trajectory 已表达动作。N1 学会后不等待 N2/N3 即进入逐件准入后的全库；只有全库失败时才用小动作集诊断共享容量/串扰。仅当出现相同当前 teacher state、不同必要未来的反例时，才加 short future-teacher preview |
| `CONTACT-GUIDANCE-ABC` | `IN_PROGRESS / B DEFERRED / A-C UNMEASURED` | 旧 `L194` A/B long 已停：每 update 是 `512 env x 24=12,288 env-step`；A/B 同时时片约 `3.126/2.983 s/update`、约 `3931/4119 env-step/s`，B 只快 `4.78%` 且 CI 跨零，不值得保留第三条 ABI。legacy profiler-on `4096x24 / 6.700 s` 是8倍 env-step/update，原始秒数不可混比。最终 `14,509/18,026` opportunities 都是0 capture；旧 `outcome_dense_only/000` 又没有 ball-state actor，不能冒充 C。fresh A211/C211 均已有独立 211/319 consumer；C211 observed-evidence producer 已能从 runtime rows 重算 selected-rubber/preflight/raw sidecar，但 `train.py` 尚缺 `000` 分流、runner-before-oracle hook 和 live contact ledger，所以 C oracle/PPO 保持 fail closed，真 A/C 学习与速率均=`未测`。固定题 A/C 都强制 `reset_inverse_solve=false/online_solver_calls=0`；producer 另报 `s/4096 novel questions`。C 的当前最小 reward 冻结为 nominal strike tick 拍心-球心距离与 actual-contact-gated 落点，不再私自添加其它 desired-contact 或 dense outcome 项。 |
| `CANONICAL-REWARD-RECIPE` | `IN_PROGRESS` | V2 已实改为非腕全身 mimic + 全相位低权 measured paddle + window 内高权 task master，SMASH split window，broad `10/10/5`，adaptive `4/.5/.5`，landing `+6..10`，且 live sigma/EMA state 已接线；A225 leaf 已移除错误的 `adaptive_sigma=false` 覆写，恢复 rollout-zero 宽、exact-strike 后单调收紧。静态会计：max motion `3.6575` < target final/initial `4.0296/4.3104` < landing `6`；历史坏误差 final/initial `2.6644/2.8727`，不用近零分母宣传。关闭仍需 physical-contact outcome bridge、全部 event reward rollout-0 安装和实测 tape 条件收入/advantage 健康 |
| `PPO-RUNTIME-RECEIPT` | `BLOCKED` | exact Pod `rsl_rl` source SHA、resolved actor/critic order+width、fresh/resume normalizer、configured/realized std、LR/KL/clip fraction/explained variance/pre-clip grad norm、finite cap 和逐 reward-group income 闭合；旧 194/318 receipt 不代签 final ABI |
| `RESET-TERMINATION-RESUME` | `IN_PROGRESS` | Isaac atomic reserve/commit 可复用；MuJoCo diagnostic lane 已实现 per-env done latch、terminated-row compact reset、pre-reset terminal observation 与 post-reset next observation、caller-owned ledger、per-env question lineage和可独立复算 receipt。关闭仍需 phase fidelity termination、follow-through/recovery RSI 与完整 mid-episode resume；当前只允许声称 reset-boundary resume |
| `BALL-FIRST-SCHEDULER` | `IN_PROGRESS / FIXED-CENTER SOURCE SEPARATED` | `immutable_tape` 是当前单行 fixed-center cache：零 online LM，但同时冻结 curriculum；可用于今晚 finite canary，不能用于扩域/full-curriculum long。扩域前必须用 `banded_question_bank` 按 exact domain level 换预解块并在缺块时 fail closed；此外仍须冻结 generator、initial/max envelope、扩域/回退、RNG、heldout/checkpoint state，并补齐可逆重测、new-band配额、样本不足作废、global/arm attribution、hysteresis、uniform/center floor 与并行探臂前置。 |
| `ISAAC-FOUR-ARM-FIXED-QUESTION` | `A211 V2 CODE IMPLEMENTED / EXACT FRAME0 RESET REJECTED BY LIVE SAFETY` | fresh A211 是当前四臂 consumer：211/319、actor localizer12+body-gyro3 ordered-layout v2、task-valid WAIT、full-body/measured-paddle/contact/outcome eligibility 和 fresh actor normalizer v2 已实现；critic内容/norm仍v1。tracked frame0 candidate file SHA=`ad17d984…bc54`。exact Pod commit `ea8c7e1d` 用该 root/q、零速与同 q hold 真跑时，在 policy step9（`.18 s / 36` substep）触发 `robot_hit_table`，实际关节 hard-edge 计数为0；因此没有 PASS receipt，也不准用关 termination 的方式继续。主链需恢复经验证 physical-ready 作 reset、WAIT 内 teacher 冻结 measured frame0 并证明过渡时间足够，再重造 lineage -> oracle32 -> scale4096。receipt/lineage/oracle32/4096 均仍阻断。C211 属于独立 A/C comparison，不是四个 A learnability 臂的前置。 |
| `ISAAC-N1-LEARNABILITY-HANDOFF` | `BLOCKED` | 一条来自真人对拉录制的单拍 measured N1；依赖 canonical measured authority/portable contract/reward/scheduler，满足 §9.1 的定量真实 hit/legal return、逐分母、安全、resume/export/handoff，不要求 Isaac N73。额外 N1/N2/N3 仅为失败定位，不阻塞 handoff |
| `MUJOCO-SCENE-CONTACT-HARNESS` | `PARTIAL / SELECTED-RUBBER CONTACT RECEIPT CLOSED` | native ball/table/racket scene、strict contact pairs、portable/backend SHA closure、substep contact/recontact/outgoing latch 已实装。exact Pod `592835dc` 同题真实 rollout 得 generic edge=1/table=0/valid outgoing，sidecar 分类正号红面，tick/substep=1/3，切向距 `0.007168732 < 0.044263876 m`，invalid=[]；receipt-v2 已在 exact detached `95382a53` replay=`18 passed`，classification 与 backend seals 独立重算一致。Reward/PPO/incoming-question parity 仍未授权 |
| `MUJOCO-SINGLE-ENV-PLANT-ACTION` | `IN_PROGRESS / PORTABLE HOLD V2 PASS` | schema-3 31-D action、implicit total-PD、delay/reset/fixed-tape 和 native ball observation/contact receipt 已实装。action-specific hold v2 用 repo-relative logical path+SHA，consumer 拒绝旧 v1、absolute/traversal/repo-escape；host=`18 passed,6 skipped`、exact Pod 真 MuJoCo d0/d1/d2=`24 passed,0 skipped`。immutable authority probe 仍只有 table edge，没有 racket hit/reward/learnability授权 |
| `MUJOCO-VECENV-PPO-CHECKPOINT` | `PARTIAL / EXECUTABLE C-LITE POD PASS` | current successor 已接 observed selected-rubber resolver、C-lite scalar reward、真 VecEnv normal step、finite PPO shell 和 reset-boundary checkpoint exact parity。无 contact bonus；距离+observed legal/out=.5。为避免伪造 full recipe，当前 motion/balance=0，只接受 immediate TASK_ACTIVE，拒绝未移植 RESET_WAIT/task-valid。exact clean Pod commit `ebe963f5` component suite=`108 passed, 0 skipped`。exact Pod commit `42500ade` 又真正运行 CPU-only/no-clobber entrypoint：`1 env x 2 step x 2 update`，checkpoint save/load SHA=`e623d214…0026`，fresh child 自然退出，下一 update receipt/model/optimizer/normalizer/RNG/reason+safety transcript 全 exact，result file SHA=`ad62b45d…377a`。这关闭 C-lite executable plumbing，不授权 formal phase/mimic、mid-episode resume、export 或4096吞吐。 |
| `MUJOCO-RUN-CONFIG-DETERMINISM` | `NOT_IMPLEMENTED` | single-source RunProfile/覆盖层、Tier-1 exact 和 Tier-2 statistical 收据；native ball-racket/table/net、solref/solimp、aero/spin、CCD/tunneling/event latch 逐项闭合 |
| `ISAAC-MUJOCO-CROSS-ENGINE-PARITY` | `NOT_IMPLEMENTED` | paired tape；question/curriculum/ABI/action/reason/reward Tier-1 exact；contact/flight/landing Tier-2 指标、容差、样本数、差异归因与 fail/waiver receipt |
| `MUJOCO-CANONICAL-N1-AUTHORIZATION` | `BLOCKED` | 显式合取门：portable ABI ∧ admitted teacher ∧ pinned sim contact/physics profile ∧ full termination/reset ∧ reward/evaluator parity ∧ trainer/save/resume ∧ run determinism ∧ fixed-tape cross-engine parity。真实拍子质量/惯量可只阻塞 sim2real，但 formal sim 仍需具名接触 profile |
| `MUJOCO-N1-REPRODUCE` | `BLOCKED / PARALLEL AFTER BUNDLE FREEZE` | shared bundle 冻结后 fresh MuJoCo N1 与 Isaac canary 并行；Isaac actor-only warm-start 仅同预算对照，不是 fresh MuJoCo N1 的前置 |
| `N73-CATALOG-ADMISSION` | `BLOCKED` | v4 的 73-action manifest 已产生且 receipt-bound，但完整 mechanical audit 是 `0/73` admitted：`57/73` position/stored-or-FD-velocity 硬失败，`16/73` 仅通过这些已知门且仍为 `UNKNOWN`。较早窄口径反例为 `37/73` URDF 超速和 `58/73` 近限位。必须重算并逐件闭合 velocity/acceleration/limit-margin、signed torque-speed/thermal、floating-base inverse dynamics、足底接触/摩擦、自碰/桌净空、fitted-ball，再补 prototype/strict load/alias/family sampling |
| `SPIN-CONTACT-CALIBRATION` | `BLOCKED` | ABI 保留 spin 列但首版 `spin_valid=false`。只有 incoming producer、off-centre friction/restitution/spin transfer、drag/Magnus flight、marker alias/effective-domain 全过后才能 promotion 且付 spin reward |
| `N73-SCALE-COMPACTION` | `IN_PROGRESS / PREP PARALLEL` | admission/manifest/alias/zero-PPO scale 可与 N1 并行准备；formal N73 才等待 N1。N73 zero-PPO/1x2/4096x5、O(envs) hotpath、memory/ledger compaction、逐动作及实际 selector transition starvation 门 |
| `ISAAC-VENDORV2-4096-SCALE` | `AUTHORIZED AFTER A211 ORACLE / NOT YET RUN` | 历史 exact `ad4ba3f4` B 臂在 `gym.make()` scene construction 中最后停于 table-USD load/其后静默 clone 区间，约 `1808 s`无 PPO；同 commit A 臂 scene=`2.968 s`、simulation start=`11.221 s`，随后首次 reset 因 birth-stratum contract 退出。fresh A211 主链在 frame0 hold+oracle32 后直接跑 `4096x5`，且只有恰好5 update、finite save、资源收据、natural clean exit 的 content-sealed terminal result 才可以发 `long4096`。一卡最多两进程可显式 colocation，但该结果不进 speed evidence；512 支线只作失败定位。 |
| `MUJOCO-N73-BALL-FIRST` | `LATER` | 完整 73 从 fresh recipe 训练，自动扩域，逐动作/侧别/题格 denominator 和 heldout，不从 N5 checkpoint 续 |
| `ONLINE-INCOMING-PRODUCER` | `NOT_IMPLEMENTED` | estimator→portable ABI→Isaac/MuJoCo/export 的 frame/time/age/validity/noise/delay 与 fixed-tape parity |
| `RALLY-EVENT-SCHEDULER` | `NOT_IMPLEMENTED` | 对手/发球机来球揭题、selector、teacher start 和 task revision 原子提交；无 mid-swing teleport/clear-history |
| `RECOVERY-READY-CARRY-STATE` | `NOT_IMPLEMENTED` | 随挥→恢复→ready 跨拍保留 robot/ball/history/delay/RNG，T0/T1 失败后才评估 T2 shaping |
| `RALLY-SEQUENCE-CURRICULUM` | `LATER` | variable-length sequence、supported transition floor/starvation、lead-time/streak strata 与 checkpoint compaction |
| `CONTINUOUS-HELDOUT-EXAM` | `LATER` | no-reset rally length、逐转移/逐侧/逐题格分母、安全和独立物理 exam；单拍 legal return 不代签 |
| `STATEFUL-EXPORT-GATE3B` | `LATER` | Python→ONNX/C++/vendor no-reset sequence 逐 tick observation/normalizer/action/qdes/history/delay parity |
| `DR-RESTORE-HEALTH` | `LATER` | 同底盘 DR 作为 baseline 接入；mass/CoM/PD/noise/history 每轴过 hold/teacher-to-hit/task/safety/receipt 门 |
| `DUAL-EVAL-PROFILES` | `LATER` | deterministic ranking 与 noisy vendor-play 分开，不能混报 |
| `INDEPENDENT-PHYSICAL-EXAM` | `LATER` | independent MuJoCo/vendor/hardware；physics/contact/spin 未测格写 `未测`，不能靠 analytic return promotion |

### 12.1 VendorV2 诊断单卡双进程 admission

2026-08-03 只收口 launcher 算力放置，不改 MDP/配方。默认 spec 仍是
`require_empty=true`；只有由
[`--allow-vendor-v2-colocation`](../../DEFINITIONS.md#vendor-v2-gpu-colocation) 生成的 exact
claim 才把它改为 false。该路径与旧 launcher 共用物理 GPU flock：旧独占锁与
VendorV2 共享生命期锁互斥，两个 VendorV2 launch 则用短 admission byte-lock 串行
count-and-reserve，并用 live namespace reservation 覆盖 CUDA PID 尚未出现的 boot 窗口。

只有下列条件全部成立才允许已有一个 compute PID 时放行第二个：

1. 新旧双方 exact claim 都 opt-in，同一物理 UUID 上硬上限为两个唯一 PID；
2. `nvidia-smi` 的 PID/UUID/total/free/used-memory MiB 可解析，且 admission 后至少保留
   `8192 MiB` free headroom，PID 可在 `/proc` 重新打开；
3. PID starttime、cwd、environment、exe 和 cmdline 绑定同一 checkout+commit、exact Isaac
   Python/`train.py` 完整 argv 与 dedicated VendorV2 namespace；environment 再绑 namespace
   receipt 的 path+SHA 和 launch claim SHA；
4. namespace 内 canonical no-clobber receipt 与 `launch_claim.json` 反向绑回同 PID/GPU/
   checkout/commit/namespace。

pre-launch、pre-exec 和 post-boot 分别写 snapshot，其中 post-boot 必须看到当前
namespace 的 verified compute PID；收据一起记录 PID、UUID、显存 MiB 与 namespace
receipt pin。dead 历史 reservation 先用 PID+starttime/live runtime handoff 判 stale 后忽略；
同一 experiment root 中其它 GPU 的 live reservation 先按 index+UUID 过滤，不参与当前卡的
checkout/claim admission。若 post-boot admission（包括 `8192 MiB` headroom）拒绝，launcher
只接受本次 `run.log.launch` 中 PID=PGID/starttime 与 canonical leader evidence 完整一致的
process group，按 TERM→五轮等待→必要时 KILL→五轮等待收口，并写 no-clobber
`post_boot_admission_failure.json`；既有 co-resident 不在该 group snapshot 中，不能成为信号目标。
post-boot 验证/receipt 写入的受控 `LaunchRefused`、`FileNotFoundError`、`ValueError` 和
`OSError` 都必须先走该闭包；`SystemExit` 等意外 `BaseException` 不被吞掉。
第三 PID、同 namespace 多 PID、未知 live 进程、无法读 `/proc`、异 checkout/commit、
receipt/完整 claim/launcher/argv 漂移都拒绝。Host CPU-only launcher suite=`47 passed`；未在 Pod
真实共驻发射，因此 runtime result 仍为 `未测`，不改 `diagnostic_unauthorized`。
实现上已把 lock、`/proc`、`nvidia-smi`、reservation/receipt validation 与 admission 机械提取到
`vendor_v2_gpu_admission.py`；launcher 只保留参数/spec/claim 集成和调用，两份源码都进入 exact
runtime-source pin，行为与上述门保持不变。

这只叫 **launch-mechanics admission**，不是持续共驻性能授权。§8.1 的 contact-guidance
`A225-proto/C225-proto` 和所有 `4096` 主 benchmark 必须
exclusive 单进程。单卡双进程另做同卡 `solo -> colocated -> solo` 交错测试，全时段记录两个 PID/
PGID、GPU UUID、used/total/free memory、peak 与 min-free、存活/OOM、p50/p90 update wall、envstep/s
和 reset strata；共驻数据不得混入主 A/C 因果或 scale 结论。当前真实 Pod 共驻=`未测`。

同一 launcher 现增加 code-owned
[`oracle2`](../../DEFINITIONS.md#vendor-v2-oracle2) 诊断 stage，作为长训前的最小 live-plant
因果门：只接受 `current_lm/111`、已 materialize 的 reward/policy、fresh namespace、
`num_envs=1/max_iterations=0`，并自动写 `<namespace>/teacher_qdes_oracle_2ep.json`。
claim 固定 output contract、两条新增 Hydra 参数和完整 training argv；trainer 在 PPO runner 前
  完成两个 terminal episode，launcher 先调用完整 schema-3 hard-contract 结构验证，
  再把 canonical JSON 与实际 hard contract、runtime source、task/reward/PPO/policy/
  dynamic-ready/manifest/motion/tape SHA 逐字段交叉验证。该实现不再把 oracle
  正常退出当成 post-boot PID 竞态：marker 后等 exact child exit；leader 非零退出
  或留有 descendant 时，必须先对已绑定原 PGID 做 descendant snapshot→TERM→必要时
  KILL，且只能 signal exact snapshot 子集；identity 漂移则写 quarantine 状态并拒绝。
  证明原 PGID 空后才走不要求 live PID 的 `post_completion` admission。训练 stage
  的 post-boot live-PID 门不变。Host 五个相关集成 suite 为 `102 passed`，
未启动 Pod、未产生 runtime result，因此本实验状态不晋级。两回合只验证 live auto-reset、ledger、
lineage 与 process cleanup；它允许零 exact-strike/capture，不能叫 teacher tracking PASS。其后还要实现并
运行带预注册 p/v/face、termination、projection exposure 和 unknown 上限的 code-owned `oracle32`。

### 12.2 PRE-LONG 基础闭包（2026-08-03）

这一节是 A211/C211 与 MuJoCo C-lite **任何 long 之前**的单一基础 checklist。它不是新的训练 Stage，
也不取代 `origin/main:docs/NOW.md` 的项目队列。今晚可以运行下面用于关闭 checklist 的 fixed-center
finite probe；但七项没有全部给出 exact receipt 前，不发 `long4096`，也不把 component test 写成 trainer
ready：

1. **ABI/IMU：**A/C actor 必须解析为211列的 ordered-layout v2：localizer world
   `position3+orientation6D+linear_velocity3` 12-D、pelvis/body-frame IMU gyro3、无
   `teacher_base_now_world15`、无 `projected_gravity`、无 world angular-velocity 重复列；actor
   trainability/normalizer 都是 v2，pre-IMU 同宽211 fail closed，critic 保持319/v1。
2. **WAIT masks：**`task_valid=0` 时 A task/C ball 9-D、base goal 和两只钟全零；task/contact/outcome
   reward 以及 opportunity/closed-swing/outcome denominator 都不记账，balance/safety/非任务 whole-body
   mimic 继续工作。task reveal 必须整 tuple 原子提交；TASK_ACTIVE miss 必须报 `0/C`，不能靠 WAIT
   稀释分母。
3. **safe-reset live hold（BLOCKED）：**exact measured frame0 zero-velocity/same-q hold 已在 Pod
   step9 触发 `robot_hit_table`，因此被拒绝为 physical ready。恢复已验证 physical-ready 后，
   它必须跑满 `200 policy tick / 800 physics substep`，zero terminal、actual-hard/qdes/table/foot
   安全账与截图/源 SHA 收据全部通过；WAIT 内 teacher 仍冻结 measured frame0。
4. **fixed-center zero inverse：**今晚 finite N1 使用单行 `immutable_tape`，逐 reset/step
   `online LM/inverse solve=0`，且 receipt 明写它冻结 curriculum。`oracle32`/`4096x5` 不等待 band bank；
   expanding/full-curriculum long 必须先换成按 exact domain level 索引的 `banded_question_bank`。
5. **MuJoCo executable runner（PASS）：**exact Pod `42500ade` 已通过真实 C-lite entrypoint 的
   `1 env x 2 step x 2 PPO update + save/cold-load`；fresh process 的 next update/state 全 exact，
   tracked result file SHA=`ad62b45d…377a`。它只关闭 C-lite plumbing，不代表 formal trainer。
6. **Isaac finite live gates：**A211 先过 code-owned `oracle32` 的 teacher-qdes、p/v/face、termination、
   selected-face/unknown、projection 与分母收据；随后在4096 env 恰好跑5 update，checkpoint/normalizer
   recursive finite、自然退出且 source/recipe/tape/reward/safety lineage 完整。launcher 必须
   实际定位并用 CPU `weights_only` 安全加载 checkout-bound `model_5.pt`，绑定文件/内嵌
   iteration 和 launch claim，对 model/optimizer/actor+critic normalizer 所有 tensor 做 finite audit；
   还要从5个连续 runtime telemetry update 重算 actual-hard/table/nonfinite 计数全零。
   long 前再重算并匹配该 terminal acceptance。512 只作失败定位。
7. **launcher colocation：**同一 GPU 最多两个进程的 exact claim、独立 no-clobber namespace、PID/UUID/
   checkout/commit/显存余量与 cleanup 收据必须在 Pod 实测；共驻只用于并行发四臂和 MuJoCo 工作，
   共驻 wall 不进入 A/C 主速率证据。

以上检查不能用历史225/318、旧194/318、host aggregate、source review 或 unexecuted plan 代签。0803
新 URDF 仍只是 content-addressed successor raw intake：右拍局部挂载虽然未变，但夹爪耦合/mesh、link-name
ABI、mount 与 plant 差异未闭合；normalized 31-D Isaac asset 与 MuJoCo identity v3 产生并重验前，不在
本 checklist 中偷偷替换现役 runtime model。

## 13. 关闭条件

本文只有在以下事实全部成立后才可标 `completed`：

1. `origin/main` 已采用实测 racket authority、单值 portable ABI、完整 reward、ball-first scheduler 和
   N1->N73 顺序；
2. shared bundle 后 Isaac canary 与 MuJoCo fresh N1 的各自定量门、fixed-tape parity 和冻结 handoff 关闭；
3. MuJoCo native trainer、full authorization AND gate、save/resume/export 与 `1/512/4096` 门通过；
4. 73 件 admission/alias/scale/compaction 门通过，N73 训练有逐动作/逐侧/逐题格证据；
5. online incoming producer、rally scheduler、carry-state recovery、sequence curriculum、continuous heldout
   与 stateful export Gate3B 通过，且按实际 transition/lead-time/streak 报分母；
6. independent physical exam 完成；缺数据的 formal 格仍明确 `未测`，没有被平均数掩盖。

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
