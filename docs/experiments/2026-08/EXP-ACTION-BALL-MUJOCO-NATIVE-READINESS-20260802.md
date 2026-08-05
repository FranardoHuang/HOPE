# EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802 — ActionBall 下一版系统与 MuJoCo 原生训练准备账

- 状态：`in_progress`
- 阶段/轴：ChingMu-73 动作库、Ball-first 自动扩域、Isaac 最小可学门、MuJoCo 原生训练
- 集成小目标：用一个自然动作在 Isaac 验证可学性的同时并行完成 MuJoCo trainer；共享 bundle 冻结后两引擎 N1 并行，主训练在 MuJoCo 直接扩到通过机械准入的完整 73 动作
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 本 successor 最高证据等级：`E1`；历史 negative-control 另有 `E3` 诊断，不传递为新系统 E3
- 创建日期/最后复核日期：2026-08-02 / 2026-08-04

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

### 2026-08-05 CST 实时收口状态

这张小表覆盖本文件后文中任何尚未明确标为 historical 的旧运行口径；每次关键状态变化必须先改
这里和§12交付账，再发下一步命令。

| 关键路径 | 当前事实 | 下一道可执行门 |
| --- | --- | --- |
| **termination/reward 对齐（2026-08-05 新增，明细见 §5.6）** | 反向审计发现 A211 运行时 `42` 个非零 term 而 §5.3 只覆盖 `22` 个；三项零命中项压过主层级。已落字节：`joint_actual_forbidden` 改 `terminate=False`（只记账不 reset，telemetry 模式强制证据记录器）、`ee_body_pos` 去腕只留脚、`upright_exp 1.0→0.25`、`hit_unstable_support -10→-1`、`death_penalty -300→-10`、`undesired_contacts` 正则 `_link→_Link`（**bug**：A3 是 `_Link`，原为 G1 命名，双脚双腕反被罚 `-2.0/episode`）、`qdes_limit_barrier_margin_frac 0.08→0.05`（消除护栏自造的 `-0.0844/关节/步` 底噪）、`init_noise_std` 四处硬钉解开且 4σ 门改为按真实 σ 计算（原为字面量 `0.02`，**假绿**）。MuJoCo 侧 `joint_actual_forbidden` 已同步 | 重跑 A/C focused suite；让 `audit_action_ball_reward_hierarchy.py` 接受 DRL0 leaf 并重算全部静态数值（**当前它拒收实际发射的 profile**）；`counter_rally_v1` 与 `virtual_landing` 的口径差待裁决 |
| observation/reward | A/C=`211/319`；无 teacher-base；唯一 actor 角速度是 body-frame IMU gyro；C 只有 nominal-strike 拍心距离与`vb_fired` selected-rubber swept analytic contact-gated单次落点；`physical_ball=false`时不是PhysX observed landing。`.99/.95`保持A3/BeyondMimic/mjlab基线。runtime/training-contract已安装fixed-N1 A `base_position 1.5→0`、九个window项`×1.15`、C proximity`240`、A/C landing`700`；progress10保留。按Take061 task-valid折扣账，A `1.773<1.852≤3.009<3.332`，C `1.773<1.904<3.332`。C launcher与oracle的旧`v2/220/500`当前已改成`v3/240/700`，等待focused test确证 | ready/swing mimic ledger和schema-3 runtime交叉检查已过；补C fixture/live ledger全链、landing∧post-contact-fall监控；真球另走promotion |
| reset/teacher | direct frame0 physical birth=`0/73`；split-ready artifact+`60/240` hold 已有；WAIT 5--25 tick期间机器人/teacher保持split-ready、球停在无接触park位；reveal原子安装来球并切measured frame0，机器人不reset。A/C leaf显式钉`backhand`。旧`.7123759904781779 s`来自 tracked interior receipt(`r4_splitready`, tick92)；A literal-center已收紧到TTC=`1.82`/tick91/wait=**`.6923799138976297 s`**，A/C materializer分别=`12/12`、`11/11`；C走独立family-C receipt。**2026-08-05 更正**：本行此前写作`.69237599 s`，那是从旧 tracked receipt 减一 tick 推出的；仓库存在两份相差`3.9e-6`的 interior 权威——tracked receipt 的`0.7123759904781779`与 code-owned `action_ball_211_four_grid_contract.CANONICAL_TEACHER_PROJECTION`(`:201`)的`0.7123799138976297`。launcher 导入期自检比对的是后者，producer 也从 prepared core 重算出与后者一致的值，故以 code-owned 常量为准，旧数只作历史。误删helper已恢复。bridge schema-v3专项=`56 passed`，但共享4096 gate仍写schema-v2，A launcher在第84项正确fail-closed，未被绕过 | 把shared gate升级为严格消费v3 `reveal_to_playback_bridge`与唯一counter表；随后A/C launcher余下测试、旧interior负例→exact-source suite→S0/S1 |
| question source | A cache schema-v2保存所有active-birth semantic rows+每动作跨reset hot row，mixed Q/Q' pure/cold replay correctness已过；C=`direct_ball`且formal A/C不用immutable tape。但当前level-0 TTC grid仍强制center±1 tick并携带stratum provenance，所以“fixed-N1”是小有限题带，不是用户要求的严格单Q | 增加curriculum-owned initial-center single-Q模式；升档后才扩题，Pod断言A cold=1/warm=0、C inverse=0 |
| checkpoint/resume | action FIFO/containment schema-v4与outer optimizer/RNG/normalizer组件各自存在，但当前non-fixed-view diagnostic command payload明确`exact_resume_supported=false`，普通A/C不保存WAIT/reveal、curriculum/domain、sampler、A hot-cache/active task；所以fresh `4096x5`可跑，当前long只能是不可恢复的fresh进程，不能再写成exact-resume已闭合 | long前补所有A/C command state、211/319 normalizer与outer schema3/inner schema4的mutation-before-load preflight，并做mid-WAIT/cache/curriculum/RNG/optimizer冷恢复逐tick镜像 |
| DR | shared DR-L0 finalizer/leaf专项 host=`31 passed`；A/C launcher profile已切严格all-off DR-L0：material、joint-default offset、CoM/mass/PD、push、reset/target/proprio/body-gyro corruption与delay均关，PPO探索噪声不属于DR且保持算法配方 | exact Pod复核resolved config；nominal learnability后才以fresh lineage单轴恢复 |
| Isaac | 4096环境与admission层每GPU最多2进程的合同已有；四格尚未运行。pod-wide `.kit_boot.lock` 原来会把scale全Pod串行；补丁已把锁收窄到Kit/extension boot，真实fcntl host suite=`22 passed`。Pod headless Isaac App 同GPU0双进程overlap已PASS：B在A尚未退出且双方各占约641 MiB时进入READY，二者自然退出；这关闭boot串行，不代签两个4096场景的显存/吞吐，正式scale仍记peak/min-free，跨GPU对照在补。当前analytic `physical_ball=false` learnability不冒充PhysX outcome。pre-long barrier/reward链专项=`71 passed`；A helper已恢复，C v3/各自timing receipt仍在focused test收口 | cross-GPU lock对照→initial-center single-Q四处一致→整组回归→oracle32→A0/A1/C0/C1各4096x5；正式四格共驻只为缩短总等待，rate证据另跑exclusive ABBA |
| MuJoCo | parked-ball/reveal、A/C task/reward、runtime seals、fresh hold-bias、single-stroke timeout与RSL式timeout bootstrap已实现；native+legacy组合回归=`219 passed,2 skipped,0 failed`。exact Pod WIP r6 的A/C各`1 env×2 update`均`COMPLETE`，211/319有限，fresh WAIT canary、reset-boundary save与cold-load exact均通过；decoded mean→tape qdes最大误差`6.62e-8 rad`，mean-action projection=0，随机WAIT transition projection=`31/775=4.0%`。确定性checkpoint replay已把每个update的7个hard-terminal全部定位成`joint_actual_forbidden`：A在episode tick `70..84`，C在`69..88`，均早于nominal strike，timeout/base/table/contact/strike/landing均0；所以它证明移植主链可执行/可冷载，同时明确反证当前plant/action bootstrap可直接做4096学习。现役hold的WAIT25另有`1000/1000`、`0` hard；4σ inward mean仍只是未sealed候选 | 先做sealed current mean-only/std.02与4σ-inset同条件100+ tick诊断，区分静态漂移、探索累积、PD/plant或projection根因；正式receipt补reason/phase/tick后才能发MuJoCo scale。inset若胜出须新lineage，不能偷换r6授权。完整reward/safety、mid-episode resume、4096与cross-engine parity继续阻塞formal promotion |
| 0803 plant | normalized successor可复现，host producer=`6 passed`；但 world racket FK因右肘原点变化约9 mm，旧retarget/hold/MuJoCo identity不可代签 | 当前旧plant只跑`OLD-PLANT-FINITE`；canonical long另走promotion DAG |
| 文档合同 | G04/G05/G06、policy ABI、工具目录与旧frame0操作页已同步；Gate保持Partial | 代码收口后再写exact test/Pod receipt与PROGRESS |

因此截至这个时间点，MuJoCo WIP A/C 执行 smoke 已发射并完成，Isaac 四格学习尚未发射；Isaac
阻塞原因是 source/lineage/runtime 合同未闭合，而不是Pod没有GPU。

**2026-08-05 补充两条会改变上表判读的事实。** 其一，r6 的 MuJoCo Pod 收据只代签**旧字节**：
该 runner 用 16 个模块的 byte-SHA 封装自身身份，与当前工作区逐条比对有 `5` 个不一致
（`a211_env`/`c211_env`/`trainer`/`fixed_center_recipe`/`vec_env`），即本轮 reward 重标定与
termination 改动**一次都没有在 Pod 上执行过**；pod checkout 的 `vec_env.py` 还停在旧版，而新版
恰恰是 r6 唯一失败根因的修复，现在直接在 Pod 上跑仍是那个 `7/7` 早死的旧 MDP。其二，阻挡 MuJoCo
上 `4096` 的不是任何授权门——所有 `*_FORMAL_BLOCKERS` 只写进收据、一条也不挡执行——真正 raise 的
只有 `MAX_EXECUTE_ENVS=64`，其下是纯 Python 顺序循环。因此 MuJoCo GPU-native 的路线判定不变：
按 §9.2 走 mjlab，而 CPU 顺序实现不再作为通往 `4096` 的路径投入。
由于 lineage 必须反向绑定 clean source commit，本次发车采用两提交协议：`S0` 先提交代码、测试、
authority 与文档；从独立 clean `S0` checkout 物化 A/C lineage 后再提交为 `S1`。Pod1 只 checkout
`S1`，runtime receipt/recipe/oracle 写入各自 ignored、no-clobber namespace；不能在同一个 dirty
worktree 一边改源码一边给自己签 lineage。

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
| 三层 paddle reward | `ADOPT STRUCTURE / FIRST N1 STATIC FINE` | coarse、fine、precision 分别解决冷启动、中距离引导和触球精度；SMASH 支持日后收紧 sigma 的候选机制，但首波 A211 为保持固定配方已将 fine width 固定在 `.50/3.0/2.10`，adaptive controller 关闭 |
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

从 rollout 0 起，球任务、桌网几何、contact/outcome eligibility、完整观测字段和完整 reward recipe
都必须存在。当前 analytic lane 从 rollout 0 使用真实 achieved paddle trajectory × virtual ball 的
selected-rubber swept contact，并不冒充 PhysX observed contact；physical-ball promotion 只能在保持
ABI、reward group与eligibility语义不变的前提下更换/校验 truth provider。早期未发生事件时用相同语义的
teacher-consistent 值与显式 validity/eligibility；禁止在后续阶段新增维度、换列语义或热改权重。

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
receipt 不原地改写。project-owned 31-D normalized asset 可以独立生成，但切成 canonical runtime 前
仍须另立 exact Isaac USD/collision、new-plant retarget/hold 与 MuJoCo identity v3 lineage，并重做
拍心全局 FK 和动力学/碰撞 parity。
后续 project-owned 决策已经覆盖早期“等待九维neutral authority”的阻塞：对这份明确要用31-D
body action 的 successor，把9个夹爪 coordinate 固定在 raw URDF 的合法 `q=0`，但保留原包
21-link子树及 `0.76626209416 kg` 全部质量/惯量；20个原包缺失的 gripper collision element
显式 disabled，不伪造 mesh。producer 已生成独立 normalized output（101 files、56,443,416 bytes，
closure=`73a47e85…8f08`，URDF=`2f15df8a…2535`），host `--check`/回归=`6 passed`。它仍只是
future-primary successor：右肘原点变化会让共同q=0拍心world位置移动约`9.013878 mm`，所以旧
retarget/hold/USD/MuJoCo identity都不能代签，现役A/C finite也不原地切plant。

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

仓库另有 tracked 的 `chingmu_n1_take061u04_mechanical_candidate_v5_20260803`，但当前裁决是
`DEFER / ISOLATED DIAGNOSTIC`，不覆盖首轮 A/C 的 v4 teacher。两版57帧 measured 拍心/face/
long-axis逐字节相同；v4 对 official site 的 p95/max/hit 拍心误差为
`.142/.214/.030 mm`，v5 为`.146/.337/.030 mm`，所以 v5 没有拍心或球目标收益。v5 的唯一明确
好处是 finite-difference acceleration peak `216.14→153.80 rad/s²`，但 velocity peak、accel/jerk
RMS反而略差；它在击球前60 ms还会把肩/肘 teacher 改到最大`.115 rad`、肘位差`12.4 mm`和
姿态差`4.77 deg`。其 receipt 明确`mechanical_admission=false`，torque-speed/权威加速度/逆动力学
仍 UNKNOWN，现文件名还会触发 canonical UID mismatch。因此首个 Isaac/MuJoCo analytic N1继续
共同消费v4，避免把teacher差异混进sim2sim；v5日后只在同题/seed/reward的成对消融中凭
projection/clamp/termination与achieved-paddle证据晋级。

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

`HOPEPingPongActionBallA3VendorV2` 及 A211/C211 leaf 目前仍是本地 branch candidate，不是
`origin/main` active runtime authority。但“没有可执行 successor”已经过时：A211/C211 已有分离的
211/319 Gym consumer、normalizer/checkpoint lineage、materialize->recipe->oracle32->scale4096->
long4096 launcher 和共享四格 manifest。它们仍是 `diagnostic_unauthorized`，且
`physical_ball=false`；C 的落点是 actual selected-rubber analytic achieved-flight，不是观测到的
PhysX 真球落台。因此可以执行 finite learnability gate，不能称 formal N1/N73 或真机授权。
当前还没有一条过了 split-ready launch-lineage + A/C oracle32 + 四格
`4096x5` aggregate pre-long 的长跑命令。Pod1 已对 73 条 measured clip 的原始第0帧做
direct physical-birth screen，结果是 `0/73` 通过当前双足/地面/支撑门；因此 exact frame0 只保留为
teacher 字节真值，不再作为 physical reset 权威。物理出生改为已有 `60 policy / 240 physics /
1.2 s` PhysX hold PASS 的 split-ready；隐藏 WAIT 最多25 tick，因而该 receipt 已覆盖 WAIT。
WAIT 尚无任务时球必须停在不会触桌/地/机器人碰撞的 parked state，不能用反向弹道把未来来球
倒推到 reset 后再让它自由飞；否则长 WAIT 会先发生隐藏碰撞并改坏 reveal 球态。任务 reveal 在
同 tick 原子把球写入 sealed launch state、把 teacher 切换为 measured frame0，机器人状态不 reset；actor 同时看到
由该族 current-center provider receipt 派生的 `time_to_teacher_start`（literal center 当前预计
A约`.692376 s`、C约`.86 s`），由 dense mimic 学习 safe-ready→frame0 过渡。Pod 上另一条
4.0 s 被动 hold 在 step81 因 `robot_hit_table` 结束，这是行为反例，不是把
`200/800` 重新升格为开训前置的理由。

### 4.2 2026-08-03 当晚 launch 边界（历史，已被 A211/C211 successor 取代）

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

### 4.3 Isaac diagnostic launch receipt 占位（A225 历史，不得作为当前 TODO）

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
  + I_valid_actual_contact * R_hit(optional sparse event bonus)
  + I_valid_actual_contact * I_valid_achieved_outgoing_flight
      * R_predicted-outcome(net/landing/spin)
  + I_eligible_achieved_flight * R_true-outcome(net/landing/out/timeout/spin)
  + R_regularization/safety
```

- **动作模仿组**=`R_body-style + R_measured-paddle-trajectory`：非腕全身模仿保持动力链；实测拍子
  teacher 在全相位低权跟 position/point-velocity/signed-face/long-axis，因此击球腕虽从 generic
  body-position/orientation/velocity mimic 释放，仍会通过刚体拍 teacher 学到引拍、加速、触球、
  随挥和手腕 twist。“释放”是移除另一个可能冲突的手腕 body owner，不是不学手腕。
- **击球引导组**：A 用 `R_contact-task` 在 `target_valid ∧ strike_window` 样本上学习所需触球状态；
  C 用 nominal-strike 拍心-球心距离。`R_hit` 只是可选稀疏事件 bonus，不是必须独立存在的第三个
  reward 层；当前 C 按用户裁决明确没有 hit bonus。无论是否付款，closed-swing 分母和
  selected-rubber contact numerator 都必须独立报告，不能用 target/distance 收入冒充已经 hit。
- **上台/结果组**=`R_predicted-outcome + R_true-outcome`：建立
  `motion < hit/contact guidance < legal return`，其中 observed legal return 是最终 truth anchor。尚未真实落台时，基于 achieved
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
all routes: B_motion^eligible < B_strike_guidance^eligible < B_table_outcome^eligible
A strike_guidance = desired-contact p/v/face terms (+ optional hit event);
C strike_guidance = nominal-strike racket/ball distance (no hit bonus)
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
| fixed coarse | 大误差冷启动和换球目标时避免 kernel 死亡 | 本地死核数学诊断；IsaacLab/mjlab 多尺度先例 | base V2 是 `.70 m / 4 m/s / 1 rad`；当前 A211 实际 override 为 `.20 m / 1.50 m/s / 1 rad` |
| fine | 在 coarse 与 precision 之间提供中距离引导；日后可随误差收紧 | SMASH 同任务消融；PBHC/KungfuBot 同族机制 | runtime 具备 adaptive 机制，但首波 A211 固定 `4.6/.575/.575` 与 sigma `.50/3.0/2.10`，不在运行中改 recipe |
| fixed strike precision | 自 rollout 0 保留最终触球精度目标 | HITTER/SMASH 触球窗 | A211=`.575/.2875/.575`；是本地层级会计候选，不宣称外部绝对权重支持 |

因此三层结构足以 `ADOPT`；但“三层”不等于“首跑必须自适应”。当前 A211
sigma/weights 只能标成 `PREREGISTERED/POD_WIRED_STATIC_BASELINE`，不能写成
paper-validated。

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
- 父 VendorV2 的历史 `virtual_landing_weight=500` 仍只是 base profile。当前 A211/C211 leaf
  都 override 为 `700`，合法事件 post-dt 收入 `+8.4..+14`；对方半场出台上界为 `+7`。
  这使 `gamma=.99`的 A `target-window-max + racket_progress` 保守上界仍低于 landing floor，
  且 C distance 也低于 landing；不修改 A3/BeyondMimic/mjlab 的 `.99/.95` 时域。
- base VendorV2 的 adaptive fine 从 rollout 0 启用：position/velocity/normal 主核权重 `4/.5/.5`，
  sigma 从 `.50/3/2.10` 按 `ball_exact_strike` 误差单调收紧到 `.075/.50/.262`；另有
  固定 precision overlay `.50/.25/.50`。live sigma 和 exact-error EMA 已纳入 strict exact resume，
  恢复时同步重建 RewardManager 中三个实时宽度。当前首波 A211 leaf 显式关闭该 controller，保持
  `.50/3/2.10` 静态；C211 不消费 desired-contact 三通道，因此该 controller 不适用。

`audit_action_ball_reward_hierarchy.py` 直接解析该 profile 和实际
`chingmu73_20260728/CLIP_ORDER.json`，用每件的真实 `T`、分窗和同一 `dt` 给出。审计还沿
defaults 链检查 V2→VendorV1→ActionBall，硬断言
full-body mimic、measured-racket teacher、action-ball target、ball outcome、table obstacle 和
完整 reward pack 同时存在；不允许只恢复 reward 数值却继续跑旧 Stage1/upper-body 配方。

上面的 `4.0296/4.3104` 表是 **base VendorV2 adaptive 候选**，不是当前 A211
实际发射值。2026-08-04 同一审计器已增加 A211 leaf 的本地继承解析，实际计算为：

| A211 实际配方口径 | 数值 |
| --- | ---: |
| 73库最长动作 motion prior cap | `3.6575` |
| fine acceptance 边界 target income（static，因此 start=final） | `4.6656116` |
| target kernel + bounded progress 上界 | `6.16825` |
| 合法上台最小/最大 | `8.4 / 14.0` |
| 历史坏误差 `.634 m / 1.9595 m/s / 56.21 deg` 下 target income | `1.8813874` |

当前 A leaf 还将 fixed-center 时白拿的 `base_position_weight` 从 `1.5`清零；
reveal 后到击球前的正向 task bridge 只保留 `racket_progress=10`。九个 window 项统一
乘 `1.15`：coarse=`11.5/11.5/5.75`，fine=`4.6/.575/.575`，precision=
`.575/.2875/.575`。按 Take061 从 task reveal 起算、`gamma=.99`的 eligible 账：

```text
A: task-valid mimic 1.77331 < accepted window 1.85151
   < window max 2.07876 + progress theoretical cap .93 = 3.00876
   < legal landing floor 3.33209
C: task-valid mimic 1.77331 < nominal-strike proximity 1.90405
   < legal landing floor 3.33209
```

所以当前字节的静态顺序为 `motion < target/strike < landing`，且坏误差下引导不是
零；但它比 base V2 宽 coarse/adaptive 反事实的 `2.6644/2.8727` 弱，必须在
pre-long 中用真实 eligible income/advantage 证明仍可辨认。

这个定价会带来一个必须单独监控的 safety-economy 风险：death 一次 post-dt=`-6`，
而 legal landing floor=`+8.4`，所以“合法上台后同回合摔倒”的最低事件净值仍可为 `+2.4`。
这不把 landing 降回 target 以下，但 pre-long/long 必须独立报告
`legal_landing ∧ post_contact_fall_or_termination`，不得把它平均进全部 TASK_ACTIVE；
若该层显著上升，则修正 outcome eligibility/恢复稳定性或 termination 经济，不得把摔倒回球当成成功。

同一审计器现在也接受 C211 leaf，而不是继续拿 A 的 target 公式手算 C。它同时绑定 C211
dependency-light reward contract 与 runtime env-config source SHA，并复算：73库最长动作
motion cap=`3.6575`、一次 nominal strike 拍心-球心 Cauchy peak=`240*.02=4.8`、合法上台=
`8.4..14`、对方侧出台最多=`700*.02*.5=7`。因此73件都满足
`motion < strike guidance < legal landing`，且出台不能压过合法落台下界。在旧 `.634 m`
距离上收入仍为`.25444`、对距离导数=`-.76011`，证明远区梯度非零；这仍只是配置后果，
不代签 PPO 可学。

| 父 VendorV2 adaptive 历史口径（非当前 A211 leaf） | masked 动作 prior cap | fine 验收边界 target income | target kernel + progress upper | legal landing |
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

当前 A211 branch candidate（尚未成为 formal launcher/runtime authority）的实际四个组件是：
full-phase measured-paddle Cauchy prior、window broad Cauchy target、固定宽度 fine exponential target、
固定 precision overlay。adaptive controller 在首波 A211 明确关闭；它只是父 VendorV2/后续扩域
候选。仍未关闭的是训练中实际收入、advantage 健康和 learnability，不是“有没有接线”。

局限必须写清：这是**配置会计 + 冻结观测误差 counterfactual**，证明修改了进入 PPO
的 reward landscape；它不证明新 policy 已学会。当前还没有 exact 球任务训练的逐 term
`raw/post-dt/eligible p50/p95/per-swing income`，所以可以关闭“只改 doc/远区直接消失”问题，
不能关闭 learnability 和实际层级 Gate。

运行时 pre-long ledger 已在不改 reward/mask 的前提下，用每个 env step 之前冻结的
`task_valid` 将 mimic 拆成 `task-invalid ready` 和 `task-valid swing`两账，并强制
denominator/income 互斥且完整加回 aggregate mimic；专项与 reward audit 合计
`58 passed`。这关闭了“把 WAIT 内 ready mimic 算进击球机会”的静态账本漏洞；
launcher fixture、exact Pod marker 和 compatible-swing 实测仍须串行 repin/验证。

### 5.4 Canonical 数值与非消失引导 Gate

当前 `.15` body scale、全相位三个 `.2` measured-paddle + `.1` long-axis pin、A211
`11.5/11.5/5.75` broad target、C211 `240` proximity 和 A/C `virtual_landing=700`
已通过配置会计与冻结误差门，但仍是 branch candidate。它们与
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
5. 分开验证所有路线的 `B_motion < B_strike_guidance < B_table_outcome`；A 与 C 的
   strike-guidance 内容分别按上面的固定定义入账，contact 成功率另用 closed-swing 分母报告；同时检查含零
   的 rollout 平均不被 dense motion 永久淹没。早期真实上台稀疏时，contact 时刻的 predicted
   net/landing shaping 必须提供连续引导，并由真实合法上台事件锚定；预测项不得在无有效接触/飞行时
   支付完整上台待遇。若某路线启用 `R_hit`，必须明确是增量还是总包，避免双重计价。
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

### 5.6 与本文设计的偏离记录（2026-08-04/05）

本节记录**已经落进字节、但与本文前面章节所写不同**的改动。规则：偏离必须写在这里，不得只改代码。
对齐口径按 Franco 2026-08-05 裁定：**先对齐 scale 与比例，再对齐里面的值**；参照系只有两个——
本文 §5.3 的折扣 per-swing 账，或[`build_1` 的 `HitterPingPong` 臂](#5-6-1-build-1)。

#### 5.6.1 触发本轮偏离的两项取证 {#5-6-1-build-1}

**(a) 反向审计：本文的层级账只覆盖了实际配方的一半。** A211 运行时共 `42` 个非零 reward term，
§5.3 的静态层级账只覆盖 `22` 个；零命中的 17 项里有 3 项量级压过主层级预算。其中
`upright_exp=1.0` 是**每步无条件发钱**、无 `task_valid` mask、无窗口、`RESET_WAIT` 内照付，
500 步 `gamma=.99` 折扣 `+1.9869`，为 §5.3 所写 task-valid mimic `1.77331` 的 `112%`、
accepted window `1.85151` 的 `107%`。即在本文自己的口径下，**“站着不动”的收入高于“学会动作”**。

**(b) `build_1` 的 `HitterPingPong` 臂是唯一已知能打到球的同底盘配方**，与本分支的三处结构差：
`init_noise_std=1.0`（本分支 `0.02`，差 50 倍，折算肩 pitch 1σ 为 `21.5°` vs `0.43°`）；
关节硬限位越界**不终止**（其 `actual_q_hard_limit_telemetry` 是恒返回 False 的 `DoneTerm`，
代码注释自陈 "intentionally not a PPO episode termination, matching the Unitree training structure"），
且 V9 已删除腕部参考包络终止（理由记在代码里：fresh-policy smoke 出 `1.67` 步 episode，
几乎每次 reset 都是腕 guard）；无 death penalty。三条本分支全部相反。

#### 5.6.2 偏离清单

| # | 项 | 本文原口径 | 改后 | 对齐依据 |
| --- | --- | --- | --- | --- |
| 1 | `joint_actual_forbidden` | 硬终止（§1 行 62-67 冻结 reason order） | `terminate=False`，只记账不 reset；telemetry 模式强制要求证据记录器，否则 fail closed | 确定性 replay `7/7` episode 在 tick `69--88` 被它终止、全部早于 nominal strike；CaT（arXiv:2403.18765）消融“二值终止回报恒零”。实测老师**不贴限位**（31 关节 × 57 帧最小余量 `0.116 rad` = `16.6%` 行程、零越限），故非参考所致。对齐 `build_1` |
| 2 | `ee_body_pos` | 脚 + 腕（父类 `HOPEDeployParityTerminationsCfg`） | ActionBall 子类覆盖为**只留双脚** | 腕是挥拍必须甩最远的一端，`0.25 m` z 包络正打在要学的行为上。对齐 `build_1` V9 |
| 3 | `init_noise_std` | 四处硬钉 `!= 0.02 -> raise` | 放开为 `(0, 1]` + 要求配置值与实际值一致；发射值取 `0.1` | 焊死可调参数使消融必须改源码，而改源码即破坏谱系。安全性改由下条真门承担。取值依据见下 |
| 4 | 4σ 硬内带门 | `radius = 4.0 * 0.02 * gain`（字面量） | `radius = sigma * noise_std * gain` | 原式在 σ 调大后仍按 `0.02` 计算、照常放行，是**假绿**而非仅仅冗余 |

**`init_noise_std` 取值的可复算依据。** 4σ 门的真实约束不在 clip 上，而在 **bootstrap hold 姿态**——
零权重 actor 的 bias 被钉在 split-ready，探索包络是从那一点向外张的。用 tracked split-ready
`physical_ready.joint_pos_rad`、MJCF `jnt_range` 与 receipt 内 `action_scale_rad`（`0.25*effort/Kp`）
逐关节算 `σ_max = min(q - inner_lo, inner_hi - q) / (4 * action_scale)`，`inner` 为 `2%` 行程内缩：

| 绑定关节（最紧 3 个） | 余量 (rad) | action_scale | `σ_max` |
| --- | ---: | ---: | ---: |
| `waist_pitch_joint` | `0.4007` | `0.5900` | **`0.1698`** |
| `left_shoulder_roll_joint` | `0.2572` | `0.3750` | `0.1715` |
| `right_shoulder_roll_joint` | `0.2606` | `0.3750` | `0.1737` |

故全局上界 `0.1698`，取 `0.1`（用掉 `59%` 余量）。折算肩 pitch 1σ 由 `0.43°` 提到 `2.15°`；
`build_1` 的对应值是 `1.0`（`21.5°`），故本值仍远低于已知可击球配方，是一步而非一跳。
注意零权重 actor + bias 钉死 ready 姿态意味着**初始策略是常数**，mimic 项的梯度只能经由
“探索产生了不同回报”传导，因此 σ 是本配方的一阶量而非二阶量。
| 5 | `upright_exp` | 本文全文零命中 | `1.0 -> 0.25`（折扣 `+0.4967` = mimic 预算 `28%`） | 对齐 §5.1“辅助项收入不得倒置主层级” |
| 6 | `hit_unstable_support` | 本文全文零命中 | `-10.0 -> -1.0`（窗内最坏 `-2.2 -> -0.22` = accepted window `12%`） | 同上。原值使“进窗但重心转移”劣于“不挥拍”，而重心转移是击球必然 |
| 7 | `death_penalty` | `-300`（post-dt `-6.0`） | `-10`（post-dt `-0.2` = 合法上台折扣下界 `3.33209` 的 `6%`） | 原值为上台下界的 `180%`，“打成一次再摔”净亏。外部三库与 `build_1` 均无此项；尽调目标 `-0.2` |
| 8 | `undesired_contacts` 排除正则 | —（未记载） | `_link -> _Link` | **Bug**：A3 body 名为 `_Link`，原写法是 Unitree G1 命名，`re.fullmatch` 大小写敏感 → 四条负向前瞻全部落空 → 双脚双腕反在惩罚名单，站立每步恒扣 `-0.004`、每 episode `-2.0` |
| 9 | `qdes_limit_barrier_margin_frac` | `0.08` | `0.05` | **构造性重叠**：投影包络内沿在 `0.05*span`，barrier 带宽 `0.08` → 任何被钳关节恒扣 `-0.0844/关节/步`（理论上限 `84%`），3 关节即吃掉窗内 dense 收入 `44%`，且由护栏自身造成、策略无法规避 |
| 10 | MuJoCo `joint_actual_forbidden` | 与 Isaac 同为硬终止 | 同步改为不进 `exact_hard_reasons`；事件走独立 `joint_actual_forbidden_observed_ticks` / `first_..._observed`，并在收据里自陈 `terminates_episode=false` / `mode=telemetry_only`，另出 `promotion_blocking_evidence.promotion_blocked` 结论位 | 两引擎对同一物理事件必须给出相同 Done，否则 cross-engine parity 比较的是两个不同 MDP |

**第 10 条的一处返工，值得单独记。** 首版实现只把事件挂成一个模块属性，既没有计数也没有进收据；
独立复核用变异测试发现该字段**全仓无人读取**（`checkpoint.py` 的 `observed`/`NO-GO`/`blocker` 关键字
零命中），即"不终止但卡晋级"只落实了前半句。这恰好就是本节反复强调、也是删除硬终止时唯一必须
一并搬过来的对冲：`build_1` 的结构是"不缩短 episode，但任何非零 fault 阻断 checkpoint"，
少了后半句就会训出一个**通过了全部门、却不能上机**的策略。现已补为三层：原始计数、`first_*`
四元组归因、以及供下游直接消费的 `promotion_blocked` 结论位（缺字段与 `True` 同义，fail closed）。
教训是通用的：**把一个硬门改软时，"记录"与"阻断"必须同一次改完**；只出计数器等于把护栏换成了一个
需要人记得去看的数字。该测试文件的断言数由 `318` 增至 `358`，复核以 10 个变异体验证（含"偷偷把终止
塞回去"与"收据里把计数写死 0"，均被杀），判决 `SOUND`。

#### 5.6.2b split-ready artifact 的坐标系合同（2026-08-05，应 yikang 之问）

世界系定义与 yikang 的口径一致且代码属实：HOPE `world` 原点是 **P1 近端左桌角**，桌面 `z=0`，
地板 `z=-0.76`；台面 `2.74`（+X）× `1.525`（−Y），网在 `x=1.37`。机器人地面原点在 world
`[-0.5, -0.7625, -0.76]`，即离近台边 `0.5 m`、对齐桌宽中心。

**摆位本身合理**，但 artifact 有一个雷：tracked split-ready 的字段名是 `root_pos_w_m`（`_w_` 读作 world），
值却在**机器人局部地面系** `a3_robot_origin_ground_z0`，而且该 JSON **自身没有任何 frame 声明**
（`a3_robot_origin_ground_z0` / `hope_world` / `world_frame` 全仓零命中）。换算：

| | 机器人局部系（artifact 原值） | 换算到 HOPE world | 人话 |
| --- | --- | --- | --- |
| pelvis（split-ready） | `[0.1526, -0.1778, 1.0684]` | `[-0.347, -0.940, +0.308]` | 台边后 `0.35 m`，桌面上方 `0.31 m` |
| pelvis（measured frame0） | `[-0.0018, -0.0010, 0.8918]` | `[-0.500, -0.763, +0.132]` | 正好 `0.5 m` 站位、对齐中心、比 split-ready 低 `0.18 m` |

照字面把它当 HOPE world 用，机器人会**站到台面上去约 `0.15 m`**。artifact 的 SHA 已被谱系钉死不能改，
因此 frame 合同写在**消费端**：`mdp/commands.py` 的 dynamic-ready 装载处新增逐轴范围断言并 fail closed，
注释写明两系的纯平移桥 `p_robot = p_world + [0.5, 0.7625, 0.76]`。边界取得很松，只需分开两个系、
不承担站位认证——HOPE-world 向量的 `x <= 0`、`y ≈ -0.76` 必然落在范围外。

仍待确认（非代码）：`0.347 m` 的站位离台边偏近（人类选手一般 `0.5--1 m`），需对着实际击球点位置复核。

#### 5.6.2c 探索包：零权重 bootstrap 与 `init_noise_std` 是同一件事（2026-08-05）

**事实。** 本分支的 actor 用零权重输出层 + bias 钉死在 ready 姿态，因此**初始策略是一个常数**；
`training_contract.py` 的 4σ 硬内带门正是为它而存在，按 hold 姿态逐关节复算给出 σ 上界 `0.1698`
（绑定关节 `waist_pitch`）。`build_1` 的 `HitterPingPong`——目前唯一已知能打到球的同底盘臂——
**三样都没有**：无零权重初始化、无钉死 bias、无 4σ 门，`init_noise_std: 1.0`。该 bootstrap 在
`git log -S` 中查不到引入点，即它只存在于尚未提交的 v5 改写里。

**机制修正（此前 §5.6 表述有误，以本节为准）。** “σ 小 -> 梯度小 -> 学不动”是错的：PPO 对均值的梯度
约为 `A * (a - mu) / sigma^2`，`1/sigma^2` 反而放大单位 advantage 的梯度。真正的机制是
**σ 小则采样动作几乎相同，advantage 本身趋于零**——不是梯度小，是信号没有了。

**本任务特有的叠加，也是本节的关键。** 本配方存在一份**不受策略控制**的回报方差源：
每 episode `5--25` tick 的随机隐藏 WAIT。等待长度差 `20` tick，仅 `upright_exp`（对齐前 `+0.02/`步）
一项就造成约 `±0.4` 的回报差；而 `sigma=0.02` 时肩 pitch 的 1σ 动作扰动只有 `0.43 度`，其在 mimic 核上
引起的回报差要小若干数量级。**动作造成的回报差被等待造成的回报差淹没**，优势估计实际上在拟合
“这一局等了几 tick”。这是结构性的信噪比问题，不是超参数调优问题。

**bootstrap 的前提已经消失（决定性证据）。** 它由 commit `b1d299e1`（2026-07-29
"Fix ActionBall launch safety and bootstrap"）一次落地，理由写在同批文档 hunk 里：当时
`joint_qdes_forbidden` 是**硬终止**，标准初始化 + `sigma=1.0` 的 fresh policy 出现
`24/24 policy steps` 全为 `joint_qdes_forbidden`、mean episode length 约 `1.0`
（`docs/gates/G05_isaac_training_first_loop.md:4135`）——出生即被 reset。把初始策略钉成常数、
`sigma` 压到 `0.02`，正是为了让 `q_des` 物理上出不了那条带。

**但同一个 commit 还引入了 `finite q_des execution projection` 与 `qdes_projection_penalty`**，
即**把该 reset 本身取消掉的机制**。两个修复同批落地，此后无人回头复核 bootstrap 是否仍有必要。
按 §1 已冻结的当前语义，有限越界 proposal **被投影并保留 transition**，只有有效 pre-clamp affine
`q_des` 含 `NaN/Inf` 才终止。**因此 bootstrap 所防的威胁已不存在**，而它的代价——初始策略是常数、
`sigma` 被 4σ 门压到 `0.1698` 以下——恰好就是当前的可学性病灶。4σ 门同理：它保护的是一件
clamp/投影/罚三层已经处理的事。

**保留意见（因此不直接照搬）。** `build_1` 的 reset 分布与本分支不同（三分支：`25%` 站立 /
`35%` 随挥回放 / `40%` RSI 带位姿速度关节噪声，每 episode `3--4` 拍），所以“`build_1` 用 `1.0` 能打到球”
不能直接推出“本分支改 σ 就能”。σ 与 bootstrap 是一包，reset 分布是另一包。

**裁决：测这一包，而非照搬。** 一卡两进程 × 3 卡 = 6 槽，把第二轴由 PPO schedule（二阶）换成
探索包（一阶）：

| 格 | 初始化 | `init_noise_std` | 回答的问题 |
| --- | --- | ---: | --- |
| `A0 / C0` | 零权重 + 钉 bias（现状） | `0.1` | 当前结构在 4σ 门下的上限 |
| `A1 / C1` | 标准初始化 | `1.0` | `build_1` 对齐 |
| `A2 / C2` | 标准初始化 | `0.3` | 中间点 |

判读：`A1/C1` 出现接触而 `A0/C0` 没有，则 bootstrap 是病灶；三档都没有接触，则排除探索包、
下一嫌疑是 reset 起点分布（§5.6 第 4 条与尽调 §9 的“起点塌缩”）。原四格的 `fixed-lr1e-4` 与
`adaptive-KL-lr1e-3` 对照降级为 later，理由是在**从未观测到一次接触**的前提下，LR schedule 的
差异无法被任何指标分辨。

**已落地的是四格，不是六格（2026-08-05）。** 中间档 `A2/C2`（标准初始化 + `sigma=.3`）暂缓：
两端点先分出胜负再决定要不要插值，六格会把 `gpu2` 也占满而 MuJoCo lane 需要它。
落地形态见上文 §5.5 的四格表；code-owned 权威在
`hope_training/whole_body_tracking/scripts/action_ball_211_four_grid_contract.py`
（`schema_version=3`，content seal `1bc1df34…b1ca`）。

#### 5.6.3 尚未对齐、需单独裁决的

- **`virtual_landing` 的实际 raw 不是本文 §5.3 所写的 `legal_base` 底薪 + 中心核。** 当前 launcher 绑定的
  `take_061_unit04_bh` manifest **8/8** 均带 `counter_rally_objective`，运行时替换为
  `0.60*legal + 0.05*落点(σ=.03 m) + 0.10*方向(8°) + 0.25*速度`，并附加 `table_bounce_count==1` 条件；
  同一 flag 还把 `virtual_pass_net` 与 `virtual_spin` 静默清零。后三档对早期策略基本不可达，
  故**合法上台的实际收入是平的 `+8.4` 台阶，不是 `8.4 -> 14` 的连续梯度**。本文全文 `counter_rally` 零命中。
- **治理断链**：两个 launcher 实际发射的 profile 是 `...DRL0Learnability`，而生成 §5.3/§5.4 静态收据的
  `audit_action_ball_reward_hierarchy.py` **明确拒收该 profile**（只接受 VendorV2 与非 DRL0 leaf）。
  因此 §5.3 的那份账**不是对发射配方计算的**。发车前必须让审计器接受 DRL0 leaf 并重算全部数值。
- **`qdes_limit_barrier_probe` / `actual_joint_limit_barrier_probe`**：与 live barrier 逐字节同一 kernel、
  返回恒零、记账幂等恒 no-op，每步在 `4096x31` 上白算两遍；但其非零权重是躲开 RewardManager
  零权重剪枝、从而让 barrier ledger 落盘的唯一手段。**省算力与保遥测冲突，待裁决**，暂不改。

## 6. 智元 setting 的采用表

| 轴 | 下一版选择 | 状态/健康门 |
| --- | --- | --- |
| exact-SKU effort/armature/nominal plant | 以 URDF/MJCF/deploy 多原件为真源；拒绝 parkour wrist regex 错表 | `READY / ADOPTED_BASELINE` |
| Kp/Kd startup DR | Kp `(0.8,1.2)`、Kd `(0.7,1.3)` 是同底盘终态 baseline；首波 learnability 的 `stable_ready_plant=true` 显式关闭 | 先过 nominal ready/teacher-to-hit/safety；后续 fresh 恢复时逐关节 resolved receipt |
| action delay | 首轮固定 `d=0`；未来 fresh `DELAY-L1/L2` 分别测 `{0,1}` / `{0,1,2}` | `DEFER / FRESH-LAUNCH BOUNDARY`；不写 in-loop scheduler，d=2 前先闭合 history/alias |
| 六轴 velocity push | 首轮关闭；未来幅值可沿用同底盘 baseline，cadence 放慢到 `10..30 s` | `DEFER / FRESH-LAUNCH BOUNDARY`；目标是每 episode 命中击球窗期不超过约 `.1` 次，不照搬 `1..3 s` |
| mass/CoM | torso/末端/拍子优先，测量值优先于随意 `±20%` | `CANDIDATE`；惯量一致性、hold、hit/safety 门 |
| friction | 不把 PhysX joint friction 数字直接搬成 MuJoCo `frictionloss` | `DEFER TO MUJOCO CALIBRATION` |
| obs noise/history | 首个因果长跑的 DR-L0 连 joint-pos/joint-vel/body-gyro corruption 都关闭；task/racket/time 噪声同样关闭 | 本体感噪声虽不直接改题目支撑集，但仍会改变估计误差与终止率；nominal learnability后以 fresh 单轴恢复。task/racket/time 噪声会降低任务可观测性，恢复得更晚 |
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
当前四格 base death 都是 `-6` post-dt；较早 corrected learnability 臂的 `-.6` 已不是本轮发射值。
A 有 broad Cauchy 与固定宽 fine kernel，C 用 `sigma=.15` 拍心-球心 Cauchy。因此旧 69%
break-even/500x 结论不是当前事实，
必须按 exact reward arm 重算。

特别地，外部尽调文档 2026-08-04 新增的 §22--§23 仍以 A225/225-D、旧 termination 经济和
228/231-D 扩列建议为对象；它是研究输入，不是当前 A211/C211 runtime authority。首轮不得据此
恢复 material/mass/CoM/joint-offset/PD/本体噪声、增加 projected-gravity/第二份角速度，或改回
immutable-tape。当前字节、strict DR-L0 manifest 与本 successor 的 211/319 ABI 优先；旧段落中可
复用的只有 support-set/termination/cadence 三闸思想，而且还必须补 extrema-feasibility、
observability/Markov、phase/eligibility 和 fresh-lineage causal matching 四闸。

当前首轮的实际 DR-L0 leaf 使用 `stable_ready_plant=true`，并进一步把 historical
robot-material、recipe-bound joint-default offset、本体感/body-gyro corruption、torso CoM、
link-mass、Kp/Kd startup DR、delay、push、reset noise 与 task sensor noise 全部关闭，
`physical_ball=false`；只有不属于DR的PPO action-distribution exploration保持算法配方。
原因不是认定这些轴“不真实”，而是它们会改变当前弱腰平衡与
限时触球的动力学可达集；在 timing/plant-derived birth hold 尚未过门时加入会混淆
ready/reward 根因。
这是 learnability 发射器的实际变更，不是文档措辞；终态 sim2real 仍须在 fresh 边界逐轴恢复。
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
历史 contact-guidance 五臂为了单独回答 target semantics，曾统一用 action/observation delay=`0`、
no push、no wide DR 和 fixed question tape；它只保留为 historical negative-control 说明。当前正式
A/C 四格不用 tape，且共同绑定严格 all-off DR-L0；延迟仍是优胜 recipe 之后的独立轴。

相同 mindset 下，下列“更真实”轴不得混成一次冷启动：

| 轴 | rollout 0 | 后续扩展门 |
| --- | --- | --- |
| nominal plant（DR-L0 exact all-off） | 首个因果长跑采用；material/joint-default/proprio/body-gyro corruption均不可见 | nominal hold/teacher-to-hit/task/safety 分层健康后，以fresh lineage逐轴恢复；禁止 startup 与 reset 双重抽同一轴 |
| Kp/Kd、link mass、torso CoM | 首波关闭 | nominal N1 学习门通过后，在 fresh launch 边界分轴恢复；不在 loop 里热改 |
| 来球位置/速度/时间/落点 | 只用 ball-first 可解中心域 | checkpointed band curriculum，有可逆回退与独立 new-band 分母 |
| spin/off-centre contact | 列存在但 `spin_valid=false`，reward=0 | 飞行、摩擦/回弹、旋转传递和别名门全过后单独 promotion |
| push | 当前 fixed-N1 leaf 关闭 | nominal N1 学习门后用 fresh lineage 单独测；cadence/幅值按 pre-strike/strike/follow-through/recovery 暴露分层，任一层 safety 恶化停车 |
| delay | learnability 臂用 `d=0` | `DELAY-L1/L2` 各自 fresh lineage；不在 loop 内扩幅，d=2 前先闭合 history/alias |
| CCD/减半全场 dt/贵重 contact reporting | 不一次全开 | 通过单轴 matched throughput + tunneling/contact truth 门再采用 |

第一版同时明确 defer：`spin_valid=false`、未标定 off-centre spin/contact 不付款；push 需按
pre-strike/strike/follow-through/recovery exposure 统计，必要时扩张 exposure 而非改变 plant SHA；
ball/question distribution 始终由冻结 ball-first 规则扩张。这样保留真实场景目标，同时不把不可观测、
未标定或完全稀疏的困难混成一次冷启动。

将上表压缩成可执行的尽调裁决：

- **ADOPT**：首个因果 long 的 rollout 0 使用实测 nominal plant 与严格 all-off DR-L0；保留
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
| `A211` | `211 / 319` | 删 teacher-base 15；保留 desired-contact 9；末尾 `task_valid=1` | A211-owned fresh lineage | current split-ready + online-solver/cache successor；lineage + oracle32 未过 |
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

Pod 对 73 条 clip 的 direct physical frame-0 birth 已完成同门槛扫描，结果为 `0/73`；所以
exact measured frame 0 只保留为 **teacher authority**，不再作为 physical reset authority。
fresh A211/C211 的 physical reset 使用 tracked split-ready artifact
`ab6b7e41…d38069`，其关节速度逐字节为零；`60 policy tick / 240 physics substep / 1.2 s`
nominal hold receipt `c8b92a28…b19b` 已覆盖 hidden WAIT 的最大25个 control tick。每个 episode
随机等待5--25 tick，WAIT 中 teacher 也停在 split-ready；reveal 同 tick 原子切到 measured frame0
teacher，并公开A/C各自current-center receipt派生的 `time_to_teacher_start`，让同一 policy 用 dense mimic 学
safe-ready→frame0 bridge。4.0 s 被动 hold 在 step81 因 `robot_hit_table` 终止，只是行为反例，
不能把 `200/800` 或4 s 被动稳定重新升格为开训前置。这个裁决同样不要求恢复 raw
teacher-base：base 空间适配由“老师 nominal contact 与当前球题 contact/ball 的差”表达；若未来
出现具体 alias 反例，先用 teacher paddle/body future preview 定位。
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
| C：无 contact target | 固定中点 N1 只给 incoming ball-at-contact `p/v/spin`；台中点是环境常量，不重复作为 task 输入 | 不做 desired-contact 反解；在 nominal strike tick 用实际拍心-球心 Cauchy 距离保留 miss 梯度；当前 analytic lane 仅在实际拍轨迹与虚拟球形成 selected-rubber swept contact (`vb_fired`) 后开放 outcome eligibility | 当前 C211 只有距离项与落点项；无独立 hit bonus、无 desired-contact 奖励，对方侧出界不超过同质量合法落台的一半；不冒充 PhysX observed contact |

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
固定球 consumer 会快多少。A 保留 `online_solver` 为 curriculum 题源，只对完整语义题 SHA
做 exact-answer reuse：第一个新 Q 真正解一次，同批 4096 个相同 Q 复用，后续 reset
只在语义不变时命中；连续球量、domain level/stratum、base/plant/motion/solver pin 任一改变
就得到 Q' 并重解。C 直接消费 incoming-ball 题，不调 inverse，也不使用 answer cache。
cache schema-v2 不再错误地“每动作只留最后一个Q”：它保存所有 active-birth semantic rows，
另保留每动作一个跨reset hot row；birth退休时精确释放非hot行，checkpoint/cold restore保存rows与
birth refs。这样同批mixed Q/Q'的immediate pure replay也不重解。
因此要分开固定 Q 的稳态 A/C consumer wall 和 A 在 Q' 上的
`seconds/4096 novel questions`；四格共驻 wall 不是主速率证据。
host 微基准中，32-arm完整语义 payload 的4096次 key JSON/SHA 由去掉无语义 deepcopy 前的约
`.629 s` 降到约 `.202 s`（约 `49 us/env`）；真实 update 通常只处理 reset 子集。它只说明 cache-key
本身不像旧33 s级 solve/reset那样是第一瓶颈，不是Pod update-rate证据。

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

formal A211/C211 不再使用
[`immutable_tape`](../../DEFINITIONS.md#action-ball-immutable-tape)。它的真实岗位是单行目标信息消融夹具：
它会冻结题和 curriculum authority，不是“当档没变就复用数值答案”的 cache。新路径
保留正常 sampler/curriculum/RNG 账，只缓存完整语义相等的 solver answer；所以初始零宽题带
可以免反复解算，domain 升档后也无需换成一个会停权 curriculum 的 source。
`banded_question_bank` 保留为可选的未来离线 producer 优化，不再是 expanding long 的必选前置。
当前长跑前硬门是：cache key 覆盖完整语义，首次/Q'/checkpoint 计数精确，纯性复核不得偷偷
重解，以及 sampler/reason/counter/reward/observation 与无 cache 路径的 parity。历史
`L194`/tape receipt 只保留追溯，不作为新 A/C 发车输入。

新 hot-path 审计还找到三个候选，但它们不再阻塞首个 finite/long baseline：

1. 历史 fixed-tape `_emitted` 会无界增长；formal A/C 已不走 tape，当前 diagnostic pool retirement
   会删除 live rows/provider history。formal expanding curriculum 的 retired/lifecycle compaction 仍是未来债，
   不是今晚发车加速项。
2. actor/critic 当前每 step 对同一 9-D task snapshot 做多次 host receipt scan；上界估算
   `4096x24x8=786,432` row checks/update。command-boundary device snapshot 可能减少这些检查，但会改
   observation transaction/checkpoint 语义，先拿 baseline profiler，再单独实现和验 parity。
3. C211 的 desired-contact target 分支恒为无效。当前 host 实现已在 validity=`000`
   时保留 position norm 供内部进度/距离语义，但在 velocity norm、face frame/dot/acos 之前
   直接跳过并将对外 target metrics 归零；selected-rubber contact、拍心-球心距离、flight/net/landing
   路径均保留。这是 semantics-preserving 候选，真实节省仍等 matched Pod profiler。

任何改动都要在 fixed-Q/RNG/reason/counter/safety/reward/observation/checkpoint parity 后才能报加速。
第1项不适用于当前 formal A/C；第2项不在首个 baseline 前冒险；第3项已写代码但未量。
此外，单卡双进程只缩短四格的总等待时间，不会提高单 run update 速率；共驻 wall
不进入 A/C 主速率证据。

正式性能选择拆成两个不同单位的 Gate，禁止混账：

- **离线/事件 producer**：同机同批 `seconds / 4096 novel questions`，并同时过 feasibility、
  landing/net/contact residual 与安全 parity；未来动态来球只按实际新题/refill 事件频率摊销。
- **steady fixed-Q consumer**：A 先对首个新 Q 解一次并预热完整语义 cache，随后测量窗口要求
  `incremental_online_solver_calls=0`；C 从启动起就是 `direct_ball`、总 inverse call=0。同 checkout/
  question/seed/GPU、相同 reset strata 的 profiler-off total wall 与 envstep/s 才是主结果；另把
  A 的 cold-Q/Q' producer 成本独立记为 `seconds/4096 novel questions`。profiler-on 只用于归因
  physics、table/termination、reset/receipt/H2D、observation、Reward/contact scorer 和 PPO。

旧 `analytic A <=.35 s/update` 没有 fixed-tape 热路对象，删除为选择门。A/C 用 10 update warm-up
加至少 50 measured updates 的同卡串行 `A→C→C→A` 交错；`4096x5` 只验证 scene/finite scale，不能替代性能门。
B 未注册，不再写成可直接并行发射的路线。

C211 的 current reward-v3 strike 距离 manager weight 现冻结为 `240`；在
`dt=.02 s`、`sigma=.15 m` 时单拍峰值为 `4.8`，距离
`.075/.15/.30/.45/.90 m` 时收入分别为
`3.84/2.40/.96/.48/.12973`。但层级不再用73库最长动作的全回合静态峰值代替
Take061 真实 task-valid 支撑集；从 reveal 起用 `gamma=.99`折扣后是
`mimic 1.77331 < strike 1.90405 < legal landing floor 3.33209`。远区仍有正收入和非零
导数。这是配置/reward-landscape 会计，真实 eligible income 与 policy gradient
仍须 Pod 训练验证；旧 reward-v2 `220/4.4/500`只是 2026-08-03 的历史快照。

当前 fresh ActionBall 仍是 N1-only；A211/C211 diagnostic launcher 已存在，但 final
ball/contact ABI 仍未冻结，也未进入 `origin/main` runtime authority。N73 的实际 blocker 是
ball-conditioned producer、A/C选择、normalizer/checkpoint/backend consumers 与逐动作机械准入，
不再是虚构一个 fixed-width motion intent。

### 8.2 PPO runtime receipt

静态 reward 会计不能代替一次真实 trainer 收据。canonical N1 发车前必须记录 exact Pod checkout、
`rsl_rl` source SHA、resolved actor/critic order+width、fresh/resume normalizer state、configured/realized
`init_noise_std=.02`、`noise_std_type=log`、`entropy_coef=.01`、optimizer/adaptive-KL 设置和 finite
iteration cap。前200 iteration 至少监视 `mean_noise_std`，前500 iteration记录 LR/KL、clip fraction、
explained variance、pre-clip grad norm、advantage/return tails 和逐 reward-group eligible income。
这些是 launch/health receipt，不是因为 reward 变了就一起调 entropy/std 的额外消融。

当前唯一可发的四格不再是 A225 的 penalty/guard 四臂，而是
`A211/C211 x 探索包` 的最小矩阵。共享 code-owned manifest 冻结同一 teacher、
base question、seed、211/319 各自 ABI、ActionBall base-safety、death manager weight `-300`、
actual-q/qdes barrier manager weight 各 `-5`，以及 qdes projection manager weight `-1`
配 `objective_weight=-5` 的同剂量目标、`metrics_only`、
`[512,256,128]` network、`entropy=.01`、delay=0 和 static contact sigma：

**2026-08-05 第二轴改版（本表已按 §5.6.2c 裁决重写；旧的 `x PPO schedule` 四格
（`A0/C0 fixed-lr1e4` 对 `A1/C1 adaptive-KL-lr1e3`）为 SUPERSEDED，其 cell_id
`*-base-safety-fixed-lr1e4` / `*-base-safety-adaptive-kl-initial-lr1e3` 已从代码中移除。）**
理由：在**从未观测到一次接触**的前提下，LR schedule 的差异无法被任何指标分辨；探索包才是
一阶量。四格现在共用 `fixed lr=1e-4`（沿用原 A0/C0 的保守值，使对照格相对上一版一字未动），
第二轴换成初始化方式 + `init_noise_std` + `noise_std_type`：

| cell | ABI / task semantics | 初始化 / 探索包 | 唯一要回答的问题 |
| --- | --- | --- | --- |
| `A0-base-safety-zero-weight-bootstrap-sigma0p1` | `A211`，desired-contact p/v/face | 零权重输出层 + 钉死 ready bias，`init_noise_std=.1`，`noise_std_type=log`，4σ 硬内带门开 | 当前结构在 4σ 门下的上限 |
| `A1-base-safety-standard-init-sigma1p0` | `A211`，desired-contact p/v/face | 标准 rsl_rl 初始化，`init_noise_std=1.0`，`noise_std_type=scalar`，4σ 门显式跳过 | 对齐 `build_1`：bootstrap 是不是病灶 |
| `C0-base-safety-zero-weight-bootstrap-sigma0p1` | `C211`，incoming ball p/v/spin | 同 `A0` | 无 contact oracle 时的直接球状态方案 |
| `C1-base-safety-standard-init-sigma1p0` | `C211`，incoming ball p/v/spin | 同 `A1` | 同 `A1`，在 outcome-only 奖励下 |

判读：`A1/C1` 出现接触而 `A0/C0` 没有，则 bootstrap 是病灶；两档都没有接触，则排除探索包，
下一嫌疑是 reset 起点分布。GPU 布局不变：A 对同卡 `gpu0`、C 对同卡 `gpu1`、`gpu2` 留给 MuJoCo。
`A1/C1` 的运行时收据是 schema 2（带 `actor_init_mode` / `four_sigma_hard_inner_gate_applied`），
只认 schema 1 + `sigma=0.02` 的 n1_vendor probe-gate 会拒收它们——这是刻意的 fail-closed，
那条冻结 gate 本来就不适用于这条新路线。

四格不再沿用历史 A225 `corrected-metrics` 的十分之一 safety 价格。hard
termination 仍只收一次 `-300*.02=-6`；actual-q/qdes barrier 的 manager weight 为 `-5`，
qdes projection 的 manager weight 为 `-1`、callable 内 `objective_weight=-5`，三路有效目标剂量
相同且四格完全一致。四格的 PPO KL learning-rate schedule 现已统一关闭（`fixed`），
manifest 里 `adaptive` 一词只用于说明"它指 KL learning-rate schedule，不是 contact sigma"。
Build_1 中 `std=.6/entropy=.0005/AdamW` 建议来自 generic `ppo.yaml`，而正式 Hitter success
lineage 又是 `std=1/entropy=.01/Adam/adaptive-lr1e-3`；没有因果证据说明哪个单项导致 hit。
因此本波**只**引入 Build_1 的一个变量——`std=1`（连同它所必需的标准初始化），
entropy/优化器/LR schedule 一律不动，避免把三件事混成一次冷启动。

以下 2026-08-03 A225 L0--L3 为 **SUPERSEDED HISTORICAL DIAGNOSTIC**，仅保留根因和
namespace 追溯，不再是当前 TODO 或发射矩阵。

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

对“现在 setting 是否能学”的裁决是：旧 L194 已实测不可学；当前 A211/C211
是 **有理由可学、但 pre-long 与 live plant 尚未授权**的设置。两者从 rollout 0 同时安装
upright/body+paddle mimic/触球引导/落台，通过 event eligibility 自然形成
`balance -> mimic -> hit -> landing`，不中途换 Stage。初始零宽 N1 中 A 在首个语义 Q 上
解一次并复用 exact answer，C 从始至终不解 inverse；
delay/push/reset/task noise 全关、CoM/link-mass/PD DR 关闭、原速 Take-061；A 用 contact target，
C 用 causal incoming-ball 加一次拍心距离/落点。这是早期最小难度，但 `physical_ball=false`
仍只允许 learnability canary，不证明 PhysX 原生球接触。只有 split-ready artifact/hold 与
WAIT/reveal bridge 合同、A/C oracle32、真 4096x5 pre-long marker 与实际
mimic/contact/outcome income 全部过门，
才能回答“能学会”。

发车实现必须逐臂 exact 写出全部 soft weight、PPO 参数、ABI/source SHA、termination union、
`max_iterations` 与 continuation/stop gate；不能让未列出的轴暗变。所有臂都需
weight-independent projection probe；未来若启用 `qdes=0` 对照，即使 reward callable 因零权被裁掉，也记录 observed/
projected sample、逐关节 count/distance 和 hypothetical unweighted penalty。暴露分母为零时该轴只能写
`未测/INELIGIBLE`，不能判胜负。

`oracle2` 只验证 live auto-reset、ledger、lineage 和无残留进程，不判定 teacher 可追踪。A211/C211 四格前还需
code-owned `oracle32`：预注册 single-stroke denominator、exact-strike p/v/face 阈值、capture/reject、
reference-only 与 hard termination 上限、projection/soft-limit exposure 和 unknown attribution 上限。
固定题四格即使通过，也只授权 `LOCAL FIXED-QUESTION DIAGNOSTIC`；canonical same-run 仍需
varying-ball causal producer、final ABI、physical actual-contact outcome bridge 与 checkpointed ball-first scheduler。

### 8.3 Reset、termination 与 exact resume

`canonical_ready`（动作数据能否提供准备位）与 reset policy 必须分开。reset 从逐环境 O(env) 工作改为
只处理 terminated batch；恢复 phase-gated fidelity termination、follow-through buffer 和
recovery-only RSI，明确禁止 mid-swing 把机器人/球“空投”到新状态。每个 reset reason、phase、动作、
side 和球题单独计数。旧 Gate A/B `9–11/6–8 s/update` 单位与 workload 未绑定，正式退役；
旧 profiler-on `6.7 s/update` 也不能直接晋级。新的 `4096x5` scale pass 只要求同 claim 自然退出、
恰好5个 finite PPO updates，且 qdes/actual-hard/nonfinite 这些 implementation strict-zero 账为零。
fall/too-low/robot-hit-table 仍是真实 termination，但对初始 policy 是 behavioral evidence：必须按
hidden-wait/revealed-pre-strike/post-strike 分项，用 wait-start/reveal/nominal-strike 作分母并守恒，
不以“必须零次”循环要求未开训 policy 已经学会平衡。全程还需 PID/UUID receipt 和
`>=8192 MiB` min-free；
速度结论另用10 warm-up+至少50 measured 的 exclusive profiler-off workload。
exact `ad4ba3f4` 的历史 4096 B 在 scene/USD bootstrap 后 1808 s 无 PPO，同 commit A
又在首次 reset 因 birth-stratum contract 退出，两个失败不能合并成单一根因。
2026-08-04 当前裁决是不再把 512 放在 fixed-N1 前置：A211/C211 四格每格都走
`oracle32 -> scale4096(4096 env, 5 update, completion-wait)`；只有 A0/A1/C0/C1 四个
scale terminal result 都被 aggregate barrier 重开并复核为 PASS，才允许任一格进入
`long4096(4096 env, 1000 update)`。
`smoke/probe512/long512` 仅在 4096 失败时做定位，不能作为 long4096 predecessor。
只有 scale4096 自身 finite/natural clean exit 且四格 aggregate barrier 通过才能发 long4096；若失败，再用
`512 -> 1024 -> 2048 -> 4096` 梯子定位，而不是先默认降规模。

checkpoint 除网络和 optimizer 外，还要保存 normalizer、每环境 delay 与完整 raw-action queue、ball
curriculum/arm assignment、eligibility/event latch、episode/reset counters 和全部 RNG。cold-load 后在首个
rollout 前对 exact question/cache state 检查 qdes、delay histogram、question、reason/counter 和 reward/obs parity；缺字段
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
| reward economy | 同 opportunity 下 motion/strike-guidance/outcome/aux typical+p95 income，contact rate另报 | `motion < strike-guidance < landing`不倒置，target/distance income不代签hit |
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
  历史 successor 已把 76-D C-lite 的 observed selected-rubber resolver、scalar reward、normal VecEnv
  step、finite PPO shell 与 reset-boundary cold-load parity 接通。当前分支另有独立 A211/C211
  211/319-D consumer；A 消费 desired-contact，C 消费 incoming-ball、nominal strike distance 与
  achieved analytic selected-rubber contact-gated flight outcome。两族都保存 checkpoint state，并使用 split-ready physical
  reset、seeded per-env 5--25 tick WAIT 与 reveal 后 measured-frame0 teacher。它们已消费一个
  与 Isaac 数值/集合同义的 partial prior subset：upright、base angular/vertical velocity、
  joint velocity、action rate、非击球腕 body position/orientation/linear/angular velocity mimic，
  以及 measured-paddle position/velocity/signed-face/long-axis。WAIT 中这些 prior 继续工作，
  task reward 仍严格 mask。但脚接触/滑动/落地、undesired-contact、Isaac applied-torque、
  完整 safety/projection、termination/export/4096 workload 与 cross-engine parity 仍未闭合，
  WAIT 也尚无 exact Pod/cross-engine receipt。
  因此当前只能称 A211/C211 code-path partial，不能称完整 ActionBall trainer 或已完成移植。
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
`108 passed, 0 skipped, 0 failed`，不再写作 Pod 组件未测。该快照当时尚未测用户可执行 C-lite runner；
后续 `42500ade/934b7c03` 已关闭历史76-D runner 的两次 PPO update 与 reset-boundary save/cold-load，
但不覆盖当前 C211。剩余 blocker 是 phase/recovery、完整 canonical Reward、
formal save/resume/export 与4096规模；component PASS 不会关闭这些格。

关闭整个 reward blocker 不是把零 reward 接给 `rsl_rl`，而是继续补齐 remaining formal
termination、teacher + `official_racket_site`、tape 的 position/velocity/face validity、legal
actual contact→achieved outgoing flight→net→landing event/reward parity。只有这些语义闭合后，
才能实现 PPO/save/resume 并量 `1/512/4096` matched workload。

MuJoCo trainer 关闭清单必须逐项有 code-owned receipt：final actor/critic ABI、normalizer 与两步
history consumer；controlled runner/factory；real Reward/done `step()`；optimizer/checkpoint 全状态、
cold-load 与 export；`1/512/4096` matched workload；ball contact/aero/Magnus 与 independent outcome
evaluator。历史 C-lite 已有自己的 diagnostic trainer/checkpoint；当前 A211/C211 已接通独立
211/319-D ABI、任务有效位、各自 task/reward、split-ready reset、seeded 5--25 tick WAIT、measured
frame0 reveal 和 checkpoint continuation，并已把上述 partial balance/body+paddle mimic subset 纳入
真实 scalar reward。但 upstream
full-recipe runner/export、脚/接触/applied-torque 等剩余 prior、完整 termination/reward parity
和4096 workload 仍未闭合，
WAIT 也还没有 exact Pod/cross-engine receipt。故“完整移植完成”的答案仍是
**没有**；合入并在 exact Pod 重放前，只能计划分别做 A211/C211 的
`1 env x 2 PPO update + save/cold-load` plumbing smoke，不能把它写成 canonical N1、GPU-native
4096 或完成迁移。

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
| `CONTACT-GUIDANCE-ABC` | `IN_PROGRESS / B DEFERRED / A-C UNMEASURED` | 旧 `L194` A/B long 已停：每 update 是 `512 env x 24=12,288 env-step`；A/B 同时时片约 `3.126/2.983 s/update`、约 `3931/4119 env-step/s`，B 只快 `4.78%` 且 CI 跨零，不值得保留第三条 ABI。legacy profiler-on `4096x24 / 6.700 s` 是8倍 env-step/update，原始秒数不可混比。最终 `14,509/18,026` opportunities 都是0 capture；旧 `outcome_dense_only/000` 又没有 ball-state actor，不能冒充 C。fresh A211/C211 均已有独立 211/319 consumer；C211 的 runner-before-oracle live hook、32个 TASK_ACTIVE closed-attempt collector、selected-rubber H/C ledger、achieved-flight sidecar 与 actor/critic incoming-ball逐值校验已经实现并通过 host 回归，但 exact Pod oracle32 仍未执行。因此真 A/C 学习与速率均=`未测`。A 只对 distinct semantic Q 调一次 online solver并缓存；C 是 direct-ball、总 inverse call=0。C 的当前最小 reward 冻结为 nominal strike tick 拍心-球心距离与`vb_fired` analytic selected-rubber contact-gated一次落点，不再私自添加其它 desired-contact 或 dense outcome 项，也不冒充PhysX observed landing。 |
| `CANONICAL-REWARD-RECIPE` | `IN_PROGRESS / STATIC N1 ORDER PASS` | V2 已实改为非腕全身 mimic + 全相位低权 measured paddle + window 内高权 task master。当前 A211 fixed-center 将 `base_position 1.5→0`，保留 `racket_progress=10`，coarse/fine/precision 九项均为父配方`×1.15`；C211 proximity=`240`；A/C landing=`700` (`+8.4..14`)。Take061 task-valid、`gamma=.99`的静态账为 A `1.773<1.852≤3.009<3.332`，C `1.773<1.904<3.332`；ready/swing mimic 分账专项已过。C reward identity 为 v3，schema-3 training-contract 已与 runtime facts exact cross-check。关闭仍需 launcher/oracle/fixture 和 MuJoCo consumer 串行 repin、pre-long 实测 eligible income/advantage、`landing∧post-contact-fall` 专项和 physical outcome truth |
| `PPO-RUNTIME-RECEIPT` | `BLOCKED` | exact Pod `rsl_rl` source SHA、resolved actor/critic order+width、fresh/resume normalizer、configured/realized std、LR/KL/clip fraction/explained variance/pre-clip grad norm、finite cap 和逐 reward-group income 闭合；旧 194/318 receipt 不代签 final ABI |
| `RESET-TERMINATION-RESUME` | `IN_PROGRESS` | Isaac atomic reserve/commit 可复用；MuJoCo diagnostic lane 已实现 per-env done latch、terminated-row compact reset、pre-reset terminal observation 与 post-reset next observation、caller-owned ledger、per-env question lineage和可独立复算 receipt。关闭仍需 phase fidelity termination、follow-through/recovery RSI 与完整 mid-episode resume；当前只允许声称 reset-boundary resume |
| `BALL-FIRST-SCHEDULER` | `IN_PROGRESS / FIXED-CENTER READY, EXPANSION UNMEASURED` | formal A 使用 `online_solver + complete-semantic exact-answer cache`：sampler/curriculum/RNG 每次 reset 正常推进，只有 Q 字节语义全同才复用；cold Q/Q' 各真实解一次。formal C 使用 `direct_ball` 且从不反解。`immutable_tape` 只保留历史目标信息消融，不进入 A/C formal lineage；`banded_question_bank` 只是可选未来 producer 优化，不阻塞首个 expanding long。仍须冻结 generator、initial/max envelope、扩域/回退、heldout state，并补齐可逆重测、new-band配额、样本不足作废、global/arm attribution、hysteresis、uniform/center floor 与并行探臂前置。 |
| `ISAAC-FOUR-CELL-FIXED-QUESTION` | `A211/C211 CODE IMPLEMENTED / INTEGRATION + PRE-LONG BLOCKED` | 当前四格是 `A/C x {fixed-lr1e-4, adaptive-KL-initial-lr1e-3}`。两者分别用独立211/319 ABI/normalizer/checkpoint，共享 measured teacher/seed/old plant/safety/network/budget。physical reset 使用 tracked split-ready，WAIT 5--25 tick；reveal 同 tick teacher 切到 measured frame0并公开本族current-center receipt派生的启动钟（literal center当前预计A约`.692376 s`、C约`.86 s`），由 dense mimic 学 bridge；禁止共用历史`.712376 s`。direct frame0 birth 已实测 `0/73`，不再授权。当前还须把 A cache/C direct-ball、DR-L0 leaf 与 split-ready lineage 在同一 clean exact SHA 闭合，随后跑两族 oracle32 和四格真 4096x5；全局 barrier 重开并逐份复核 source/claim/model5/telemetry 前 long 全阻断。 |
| `ISAAC-N1-LEARNABILITY-HANDOFF` | `BLOCKED` | 一条来自真人对拉录制的单拍 measured N1；依赖 canonical measured authority/portable contract/reward/scheduler，满足 §9.1 的定量真实 hit/legal return、逐分母、安全、resume/export/handoff，不要求 Isaac N73。额外 N1/N2/N3 仅为失败定位，不阻塞 handoff |
| `MUJOCO-SCENE-CONTACT-HARNESS` | `PARTIAL / SELECTED-RUBBER CONTACT RECEIPT CLOSED` | native ball/table/racket scene、strict contact pairs、portable/backend SHA closure、substep contact/recontact/outgoing latch 已实装。exact Pod `592835dc` 同题真实 rollout 得 generic edge=1/table=0/valid outgoing，sidecar 分类正号红面，tick/substep=1/3，切向距 `0.007168732 < 0.044263876 m`，invalid=[]；receipt-v2 已在 exact detached `95382a53` replay=`18 passed`，classification 与 backend seals 独立重算一致。Reward/PPO/incoming-question parity 仍未授权 |
| `MUJOCO-SINGLE-ENV-PLANT-ACTION` | `IN_PROGRESS / PORTABLE HOLD V2 PASS` | schema-3 31-D action、implicit total-PD、delay/reset/fixed-tape 和 native ball observation/contact receipt 已实装。action-specific hold v2 用 repo-relative logical path+SHA，consumer 拒绝旧 v1、absolute/traversal/repo-escape；host=`18 passed,6 skipped`、exact Pod 真 MuJoCo d0/d1/d2=`24 passed,0 skipped`。immutable authority probe 仍只有 table edge，没有 racket hit/reward/learnability授权 |
| `MUJOCO-VECENV-PPO-CHECKPOINT` | `PARTIAL / POD WIP A+C 1x2 COMPLETE` | tracked `42500ade/934b7c03` 只证明历史 76-D C-lite。当前分支的独立 A211/C211 211/319-D WIP 已在 exact Pod 各完成 `1 env x 2 PPO update + reset-boundary save/cold-load`；A/C result SHA 分别为 `d58cb750a5f1e4d97e72c0b5adde018e65ae06ef4b58baec1847b5c40a083bb2` / `440a1f2e3b9f37dfa1491fd3916cc8d9676abd47621fe4ef3dc6d9cc5d323733`，cold-load exact与update2 exact均为true，fresh WAIT deterministic hard/nonfinite/projection均0。随机WAIT joint-event projection为`31/775=4.0%`；每个update有7个无selected-contact TASK_ACTIVE hard-terminal row，但receipt未导出具体reason，属于长跑前telemetry blocker。该顺序CPU/cap64 diagnostic path不等于4096 native training或formal checkpoint。脚/接触、undesired-contact、applied-torque、完整safety/termination、mid-episode resume/export、4096与cross-engine parity仍缺失。 |
| `MUJOCO-RUN-CONFIG-DETERMINISM` | `NOT_IMPLEMENTED` | single-source RunProfile/覆盖层、Tier-1 exact 和 Tier-2 statistical 收据；native ball-racket/table/net、solref/solimp、aero/spin、CCD/tunneling/event latch 逐项闭合 |
| `ISAAC-MUJOCO-CROSS-ENGINE-PARITY` | `NOT_IMPLEMENTED` | paired tape；question/curriculum/ABI/action/reason/reward Tier-1 exact；contact/flight/landing Tier-2 指标、容差、样本数、差异归因与 fail/waiver receipt |
| `MUJOCO-CANONICAL-N1-AUTHORIZATION` | `BLOCKED` | 显式合取门：portable ABI ∧ admitted teacher ∧ pinned sim contact/physics profile ∧ full termination/reset ∧ reward/evaluator parity ∧ trainer/save/resume ∧ run determinism ∧ fixed-tape cross-engine parity。真实拍子质量/惯量可只阻塞 sim2real，但 formal sim 仍需具名接触 profile |
| `MUJOCO-N1-REPRODUCE` | `BLOCKED / PARALLEL AFTER BUNDLE FREEZE` | shared bundle 冻结后 fresh MuJoCo N1 与 Isaac canary 并行；Isaac actor-only warm-start 仅同预算对照，不是 fresh MuJoCo N1 的前置 |
| `N73-CATALOG-ADMISSION` | `BLOCKED` | v4 的 73-action manifest 已产生且 receipt-bound，但完整 mechanical audit 是 `0/73` admitted：`57/73` position/stored-or-FD-velocity 硬失败，`16/73` 仅通过这些已知门且仍为 `UNKNOWN`。较早窄口径反例为 `37/73` URDF 超速和 `58/73` 近限位。必须重算并逐件闭合 velocity/acceleration/limit-margin、signed torque-speed/thermal、floating-base inverse dynamics、足底接触/摩擦、自碰/桌净空、fitted-ball，再补 prototype/strict load/alias/family sampling |
| `SPIN-CONTACT-CALIBRATION` | `BLOCKED` | ABI 保留 spin 列但首版 `spin_valid=false`。只有 incoming producer、off-centre friction/restitution/spin transfer、drag/Magnus flight、marker alias/effective-domain 全过后才能 promotion 且付 spin reward |
| `N73-SCALE-COMPACTION` | `IN_PROGRESS / PREP PARALLEL` | admission/manifest/alias/zero-PPO scale 可与 N1 并行准备；formal N73 才等待 N1。N73 zero-PPO/1x2/4096x5、O(envs) hotpath、memory/ledger compaction、逐动作及实际 selector transition starvation 门 |
| `ISAAC-VENDORV2-4096-SCALE` | `BLOCKED ON CLEAN INTEGRATION + A/C ORACLES / NOT YET RUN` | 历史 exact `ad4ba3f4` 仅作 scene/reset 失败定位。fresh A211/C211 必须先在同一 exact SHA 绑定 split-ready/WAIT bridge、A cache/C direct-ball 与 DR-L0，再各跑 oracle32，随后执行 `A0/A1/C0/C1` 四个独立 `4096x5`。每格须恰好5 update、finite `model_5`/normalizer、完整 source/recipe/question-cache/reward/safety lineage、连续 telemetry 和 natural clean exit；四格可独立完成 scale，但任何 long 前全局 aggregate barrier 必须同时重验。GPU0=`A0+A1`、GPU1=`C0+C1`，每卡最多两个同族进程；共驻 wall 不进 A/C 主速率证据，512 只作失败定位。 |
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

切换期边界不被夸大：旧 N1/A225/C225 launcher 在其它 checkout 先写本地 pending、
但 CUDA PID 还没出现的瞬间，新全局 registry 不可能反向发现它。本轮 Pod 因此把下列
事实写入 barrier receipt：发车前 drain 所有 legacy pending/live trainer，随后只允许一个
fresh exact checkout 的 A211/C211 writer，旧 launcher 禁用。这是本 rollout 的 transition
invariant，不是“已对任意历史 checkout 完成双向原子互斥”的泛化声明。

这只叫 **launch-mechanics admission**，不是持续共驻性能授权。§8.1 当前的
`A211/C211` 固定题 `scale4096` 允许在显式 same-family/max-two claim 下两两共驻，
用来快速关闭四格 finite/scene/checkpoint gate；此类 result 必须写
`rate_evidence_eligible=false`。A/C update 速率的主 benchmark 仍必须 exclusive 单进程。
单卡双进程另做同卡 `solo -> colocated -> solo` 交错测试，全时段记录两个 PID/
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

这一节是 A211/C211 与 MuJoCo C211 **任何 long 之前**的单一基础 checklist。它不是新的训练 Stage，
也不取代 `origin/main:docs/NOW.md` 的项目队列。今晚可以运行下面用于关闭 checklist 的 fixed-center
finite probe；但七项没有全部给出 exact receipt 前，不发 `long4096`，也不把 component test 写成 trainer
ready：

1. **ABI/IMU：**A/C actor 必须解析为211列的 ordered-layout v2：localizer world
   `position3+orientation6D+linear_velocity3` 12-D、pelvis/body-frame IMU gyro3、无
   `teacher_base_now_world15`、无 `projected_gravity`、无 world angular-velocity 重复列；actor
   trainability/normalizer 都是 v2，pre-IMU 同宽211 fail closed，critic 保持319/v1。
2. **WAIT masks：**`task_valid=0` 时 A task/C ball 9-D、base goal 和两只钟全零；task/contact/outcome
   reward 以及 opportunity/closed-swing/outcome denominator 都不记账，balance/safety/非任务 whole-body
   mimic 继续工作。ledger 必须用 env-step 前冻结的 `task_valid` 分开
   task-invalid ready-mimic 和 task-valid swing-mimic，两边 denominator/income 互斥且完整加回 aggregate mimic。
   task reveal 必须整 tuple 原子提交；TASK_ACTIVE miss 必须报 `0/C`，不能靠 WAIT
   稀释分母。runner 还必须用 raw validity 在 empirical normalizer **之后**再次清零 actor
   `[197:210]` 与 critic `[305:318]`；fresh initial、rollout next 与 bootstrap/terminal value 三条
   forward 路径都走同一 hook，防止 normalizer mean 把 WAIT 零变成隐式任务信号。
3. **split-ready birth + learned reveal bridge（E1 CLOSED / exact Pod integration pending）：**direct
   measured-frame0 physical birth 同门槛扫描为 `0/73`，因此只保留它作为 teacher frame0。
   physical reset 必须消费 tracked split-ready artifact `ab6b7e41…d38069`，其 joint velocity=0；
   `60/240/1.2 s` hold receipt `c8b92a28…b19b` 覆盖最大25 tick hidden WAIT。WAIT 中 teacher/physical
   都在 split-ready；reveal 同 tick teacher 切到 measured frame0，公开各族 task receipt 派生的
   teacher-start clock，由 dense mimic 学 bridge。literal-center 当前预计 A约`.692376 s`、C约`.86 s`，
   两者不得共用旧`.712376 s`常量或 receipt。4 s step81 table collision 只记行为反例，不恢复 `200/800` 门。
4. **A semantic cache / C no inverse（integration pending）：**A formal source 是 `online_solver`；
   每次 reset 仍运行 sampler/curriculum/RNG，cold Q/Q' 各解一次，同批4096和后续相同语义 Q 复用，
   replay/assert/checkpoint 路径不得重解。C formal source 是 `direct_ball`，总 inverse=0。
   `immutable_tape` 不进入两族 formal lineage，banded bank 只是未来可选 producer 优化。
5. **MuJoCo A211/C211 executable runner（POD r3 CAUSAL FAILURE / r4 FIX IN PROGRESS）：**历史 exact Pod
   `42500ade/934b7c03` 只关闭76-D C-lite。当前分支 A/C 两族已有独立211/319-D ABI、task-valid、
   split-ready reset、seeded5--25 tick WAIT、measured-frame0 reveal、各自 task/reward 与
   checkpoint-v3 reset-boundary continuation。Pod WIP r3 已进入真实physics/update并验证park→reveal，
   但A因未遍历完整`3/11/11` raw-reward窗失败，C因update没有结束在reset boundary失败。复核定位到
   native fresh actor均值近0，而split-ready hold的归一化action范围约`[-13.3,7.9]`：首个policy step
   就放弃安全准备姿态。修复必须复用Isaac的fresh-only初始化合同——末层weight清零、bias写入
   normalized hold、初始std=`.02`——并纳入config/receipt/checkpoint lineage；warm-start不得重置。
   Pod bootstrap探针已能穿过16-tick WAIT，但仍在tick74因`waist_roll_joint` actual hard-limit失败，
   早于nominal strike tick108；此时qdes恒为`-0.0816`，说明bootstrap必要但不充分，旧Isaac
   split-ready/PD不是MuJoCo plant的被动静态平衡点，但sealed mean已覆盖最大25-tick WAIT；这正是
   balance policy应从rollout0学习的状态，不能把“常量qdes开环500 tick”偷换成发射前提。发射门改为
   reset合法、fresh actor首动作等于sealed hold、deterministic mean覆盖25 WAIT，以及std=`.02`的
   stochastic WAIT canary按joint报告projection并保持hard/nonfinite门；finite越界proposal按既有
   projection语义训练，不因4σ理论包络触边就拒绝。随后分别要求 A/C
   `1 env x 2 PPO update + save/cold-load`；learned policy再过≥500-tick稳定promotion门。
   PPO rollout允许继续收集到reset boundary，但不能丢掉或覆盖中间transition。未实现项保持 fail-closed；
   它不代签4096/native completion。
6. **Isaac finite live gates：**A211 与 C211 分别过 code-owned `oracle32` 的 teacher-qdes、p/v/face或
   incoming-ball、termination、selected-face/unknown、projection 与分母收据；随后各自的 fixed/adaptive
   格在4096 env 恰好跑5 update，checkpoint/normalizer
   recursive finite、自然退出且 source/recipe/tape/reward/safety lineage 完整。launcher 必须
   实际定位并用 CPU `weights_only` 安全加载 checkout-bound `model_5.pt`，绑定文件/内嵌
   iteration 和 launch claim，对 model/optimizer/actor+critic normalizer 所有 tensor 做 finite audit；
   还要从5个连续 runtime telemetry update 重算 qdes-hard/actual-hard/nonfinite strict-zero；
   fall/too-low/table 仍终止，但按 hidden-wait/revealed-pre-strike/post-strike 计数、守恒和分母报告。
   long 前再重算并匹配每一格 terminal acceptance。512 只作失败定位。
7. **launcher colocation + four-cell barrier：**同一 GPU 最多两个进程的 exact claim、独立 no-clobber namespace、PID/UUID/
   checkout/commit/显存余量与 cleanup 收据必须在 Pod 实测；共驻只用于并行发四臂和 MuJoCo 工作，
   共驻 wall 不进入 A/C 主速率证据。计划布局为 GPU0=`A0+A1`、GPU1=`C0+C1`、GPU2=MuJoCo；跨族/第三
   进程都必须 fail closed。四格 scale 可以独立完成，但任何 long 前必须由一个全局 aggregate barrier
   同时重验四格 source、launch claim、`model_5`、normalizer、telemetry、question/motion/ready lineage。

以上检查不能用历史225/318、旧194/318、host aggregate、source review 或 unexecuted plan 代签。0803
新 URDF 仍只是 content-addressed successor raw intake：右拍局部挂载虽然未变，但夹爪耦合/mesh、link-name
ABI、mount 与 plant 差异未闭合；normalized 31-D Isaac asset 与 MuJoCo identity v3 产生并重验前，不在
本 checklist 中偷偷替换现役 runtime model。

### 12.3 PRE-LONG 独立复核后的实际裁决（2026-08-03）

这次复核把会改变 PPO 实际输入、reward 或 reset 的项目与纯文档措辞分开，裁决如下。

1. **A/C 不是同一 observation 做开关消融。**两者都是 actor/critic=`211/319`，共同删除
   `teacher_base_now_world15`，因为 policy 需要的是老师拍心与本题击球点之间的差，不需要老师底座与
   当前机器人底座之间的差。A 的9维 task是 desired contact p/v/signed-face；C 的9维 task是 incoming
   ball contact-time p/v/spin。C 的固定台中点是环境常量，不重复进 actor。`225-15+task_valid1=211`；
   historical critic 没有这15维，因此是 `318+1=319`。
2. **角速度只保留一份 body-frame gyro。**actor 前12维仍是 localizer world
   position3+orientation6D+linear velocity3，第12:15维是 pelvis/body-frame angular velocity。
   不再保留 world angular velocity，也不增加 projected gravity。基础非 L0 profile 可挂 simulator
   body-gyro `+-0.2` robustness noise，但当前首个 A0/A1/C0/C1 的 strict DR-L0 明确把它清零；部署
   映射仍是 bias-corrected pelvis IMU。`+-0.2` 的幅度/时间相关性尚未用真 IMU 标定，只能在 nominal
   learnability 后作为 fresh 单轴候选，不能叫 sensor-calibrated model，更不能冒充首跑现状。
3. **RESET_WAIT 与有效任务分开。**physical reset/hidden WAIT 使用 tracked split-ready，关节速度为零；
   exact measured frame0 是 reveal 后的 teacher authority，不是 physical birth。5--25 tick 隐藏等待期间
   `task_valid=0`，A/C task9、base-goal2、两只钟均为零，teacher/physical 都在 split-ready。task reveal
   前球停在无接触 parked state；task reveal 同 tick 原子显示完整 task、安装 sealed incoming-ball
   launch state，并把 teacher 切到 measured frame0，公开由当前 A/C 各自 task receipt
   派生的 `time_to_teacher_start`；policy 用 dense mimic 学 bridge，不隐式 teleport physical state。WAIT 中
   balance/safety/non-task mimic 继续工作；task/contact/outcome reward 与分母不工作。
   TASK_ACTIVE swing 一旦闭合，即使没击球也必须记 `0/C`，不得把 WAIT 当零分母稀释失败。
4. **C reward 不是 A reward 去掉 solver。**C 从 rollout0 只有 nominal strike tick 的 URDF official
   paddle-centre/ball-centre Cauchy distance（`sigma=.15 m`，post-dt peak `4.8`）和实际拍轨迹×虚拟球
   形成 selected-rubber swept contact (`vb_fired`) 后的一次 achieved analytic flight outcome。合法对方台面收入 `8.4..14`；落在对方半场但出台
   最多是对应 landing kernel 的一半；own-side/backward/net-fail/miss为零。没有 desired-contact
   p/v/face reward，没有连续 dense outcome，也没有无接触的假想落点。exact face-centre offset只用于
   contact/flight，不再重复移动 C distance 的拍心。
5. **formal A/C 不用 `immutable_tape`。**用户要的是“curriculum 题语义未变时不重复反解”，不是
   冻结课程权威。A 每次 reset 仍采题、记 RNG/curriculum，只对完整语义相等的 Q exact-cache answer；
   cold Q/Q' 各解一次，同批4096和后续相同 Q 命中。C 直接观测 incoming ball，不存在 inverse。
   `immutable_tape` 仅保留目标信息消融；`banded_question_bank` 是未来可选 producer 优化，不是
   fixed 或 expanding long 的硬前置。
6. **旧随机性报告的关键数值不再支配 A211/C211。**`-72/69%/sigma=.075` 来自旧 A225 配方。
   当前四格是 `A/C x {fixed-lr1e-4, adaptive-KL-initial-lr1e-3}`，四格恢复
   ActionBall base-safety：death post-dt=`-6`，actual-q/qdes barrier manager weight=`-5`，
   qdes projection manager weight=`-1` 且 `objective_weight=-5`；A 用
   `.20 m / 1.50 m/s / 1 rad` coarse、固定 `.50/3.0/2.10` fine 与 precision
   overlay，C 使用 `.15 m` 球拍距离核。因此 support-set/cadence/termination 三闸只保留为
   必要条件，还必须增加 calibration、observability/Markov、contact-income 和 lineage 四闸。
   `stable_ready_plant=true` 本身并不等于 nominal：未经过DR-L0 finalizer时仍会保留全机material、
   joint-default `+-0.01 rad` 与 body-gyro/joint proprio noise，这些都可改变闭环限时可达集。
   当前A0/A1/C0/C1四格launcher已经全部改绑fresh `DR-L0`，不是先跑retained-DR scale、再只给long
   切DR-L0；scale与long必须共享同一strict all-off resolved contract、fresh normalizer/checkpoint、
   recipe/lineage/namespace，不在同optimizer内热改。
   当前 A/C DR-L0 leaf、shared finalizer 与 manifest 已把缺失 joint-offset event 显式编码为 ordered
   31-D zero delta，并对 material/joint-offset/CoM/mass/PD、push、proprio、reset、task transport/noise和
   action delay `[0,0]` fail closed；专项 host 回归 `31 passed`。尚未关闭的是 launcher/lineage 对
   DR-L0 leaf+manifest+hard-contract 的 clean S0/S1 lineage 与 exact Pod resolved-config验证。
   retained-DR 若日后重跑，只能另立工程 comparator namespace，不能进入这次四格aggregate barrier。
7. **延迟不能只看终止率。**当前 actor 只有 last action，没有 applied-action/lag 或足够 action queue；
   hidden two-step lag 会破坏 Markov 性。顺序固定为 d0先学会；若要解冻，先增最小充分的延迟可观测
   合同，再做 fresh `DELAY-L1/L2` 或实现完整 checkpoint/optimizer/normalizer/RNG continuation。
   当前 launcher fresh-only，所以“在 checkpoint 边界升档”尚不是已实现能力。

实际 learnability 的自然链为：稳定站立/等待收入先可得，随后 full-body+measured-paddle mimic 学完整
专业动作，在 nominal strike/击球窗获得接触引导，当前analytic lane只有`vb_fired` selected-rubber swept contact后才出现出球和上台
收入。它们从 rollout0 安装但按事件自然 eligible，不是人工切换 Stage。难度顺序也按同一因果链：
先用 clean DR-L0 证明 balance/mimic/hit/landing 可学，再用 fresh recipe 逐轴恢复 plant/proprio、
delay、push、reset 和 task 分布，不开 in-loop DR scheduler。能否学会仍取决于 split-ready lineage/
WAIT-reveal bridge exact integration、A/C oracle32、4096x5 finite/telemetry 与真实 per-group income；源代码静态闭合不能
代签这些 gate。

这里必须区分两个门，否则会形成循环依赖：`4096x5` pre-long 只证明可构造、
TASK_ACTIVE/closed-swing 分母可见、balance/mimic 收入非零、task kernel 的反事实梯度与
safety/telemetry finite；它不要求初始策略已有 contact/landing income，否则就要求“学会后
才允许开始学”。但 fixed-N1 diagnostic long 在预注册学习预算内必须将 contact/
achieved-flight/landing 分母和收入从零推上去；不然只能裁决为不可学，不得 promotion。

本轮为 frame0 gate 新生成了覆盖 exact measured bank 的机械审计，而不是沿用只有一条动作、没有 bank
receipt 的旧选动作审计。新审计分母为 `73/73`，其中 `16` 条仅通过 URDF 位置/速度运动学检查；全库
`0/73` 获得机械准入，因为加速度权威、torque-speed 曲线和逐帧逆动力学力矩仍缺失。所选
`Take_061_unit04_BH` 是这16条之一，结论仍是 `UNKNOWN`，只能在显式
`allow-mechanical-unknown-diagnostic` 下进入仿真 hold 诊断，不能授权正式训练、真机或 promotion。

### 12.4 Threshold-first direct-frame0 尝试：已被 Pod 反证（2026-08-04）

本节保留一次被推翻的设计尝试，避免它再次混入当前 TODO。host 曾预注册13项 slack evaluator、
independent physical-blade centre/face/long-axis authority、保守 collision pair 与原子 no-clobber
artifact I/O；这些检查方法仍可复用。但 exact Pod 对73条 measured clip 的 direct physical-frame0
同门槛扫描结果是 `0/73`，因此“通过 direct 后用 transition=0、再跑 `62/248` hold”的运行路线已
**REJECTED**，不再是 `未测` 或 blocker。

当前 adopted route 是 tracked split-ready physical state + zero joint velocity，消费
`60/240/1.2 s` nominal hold receipt覆盖 hidden WAIT；reveal 同 tick teacher 切到 measured frame0，
policy 通过公开 teacher-start clock和 dense mimic学习非零 bridge。直接 frame0、历史 same-q hold、
leg-only projection、4 s被动稳定和 `200/800` durability 都不得成为隐式 fallback。安全 termination
保持不降级；bridge 的桌/跌倒/too-low事件按 phase 作行为证据，qdes-hard/actual-hard/nonfinite
才是实现 strict-zero。

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
schema-v4 `73/73` current-site full-phase 运动学重定向、50 Hz 物化和独立 FK 闭环，Take-061
拍心/face/long-axis 已进入 A211/C211 的 source-sealed fixed-N1 链。A/C 已有独立 211/319-D
observation/normalizer/checkpoint、split-ready physical reset、5--25 tick hidden WAIT、reveal 后 measured
frame0 teacher 与分开的 reward：A 为 desired-contact 三通道并使用 online-solver完整语义 cache；C 为
direct incoming-ball 九维、一次拍心距离和analytic selected-rubber contact-gated一次落点，不调用 inverse。静态审计均满足
motion < strike/target < legal landing；这只是 E1。旧 C tape/frame0 materializer合同已清除，
两族launcher已绑定DR-L0；仍待initial-center严格单Q、clean exact SHA、两族oracle32、四格4096x5 aggregate barrier和
真实 eligible income仍未测。
mechanical audit 仍是 `0/73` admitted：`57/73` 有已知 position/velocity 硬失败，另`16/73` 因缺
acceleration/torque-speed/inverse-dynamics authority 为 `UNKNOWN`，所以当前只能
`allow-mechanical-unknown-diagnostic`，不能 formal promotion/真机。MuJoCo 当前已接 A211/C211 ABI/reward、
split-ready reset、5--25 tick seeded WAIT、measured-frame0 reveal 与 reset-boundary continuation checkpoint，
但 exact Pod A/C cold-load smoke、cross-engine WAIT receipt、完整 termination/resume/export 和真实4096
GPU-native trainer未闭合。
保留的是 A 的9-D `desired_at_contact` 与 C 的 causal
ball state；被否决的是额外 synthetic motion intent。旧 Stage1 与历史76-D C-lite 都不能代签当前系统。**
