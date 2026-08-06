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
| **termination/reward 对齐（2026-08-05 新增，明细见 §5.6）** | 反向审计发现 A211 运行时 `42` 个非零 term 而 §5.3 只覆盖 `22` 个；三项零命中项压过主层级。已落字节：`joint_actual_forbidden` 改 `terminate=False`（只记账不 reset，telemetry 模式强制证据记录器）、`ee_body_pos` 去腕只留脚、`upright_exp 1.0→0.25`、`hit_unstable_support -10→-1`、`death_penalty -300→-10`、`undesired_contacts` 正则 `_link→_Link`（**bug**：A3 是 `_Link`，原为 G1 命名，双脚双腕反被罚 `-2.0/episode`）、soft-limit v2 两条通道带宽 `0.08→0.05`（`qdes_limit_barrier_margin_frac` 与新增的 `joint_limit_margin_frac`，消除护栏自造的 `-0.0844/关节/步` 底噪）、`init_noise_std` 四处硬钉解开且 4σ 门改为按真实 σ 计算（原为字面量 `0.02`，**假绿**）。MuJoCo 侧 `joint_actual_forbidden` 已同步 | 重跑 A/C focused suite；~~让 `audit_action_ball_reward_hierarchy.py` 接受 DRL0 leaf~~（**2026-08-06 更正：已于 `635252f6` 接受**，见 §5.6.13 (D)2）**并重算全部静态数值**（这半句仍欠，至今无对 DRL0 leaf 重算的收据）；`counter_rally_v1` 与 `virtual_landing` 的口径差待裁决 |
| observation/reward | A/C=`211/319`；无 teacher-base；唯一 actor 角速度是 body-frame IMU gyro；C 只有 nominal-strike 拍心距离与`vb_fired` selected-rubber swept analytic contact-gated单次落点；`physical_ball=false`时不是PhysX observed landing。`.99/.95`保持A3/BeyondMimic/mjlab基线。runtime/training-contract已安装fixed-N1 A `base_position 1.5→0`、九个window项`×1.15`、C proximity`240`、A/C landing`700`；progress10保留。按Take061 task-valid折扣账，A `1.773<1.852≤3.009<3.332`，C `1.773<1.904<3.332`。C launcher与oracle的旧`v2/220/500`当前已改成`v3/240/700`，等待focused test确证 | ready/swing mimic ledger和schema-3 runtime交叉检查已过；补C fixture/live ledger全链、landing∧post-contact-fall监控；真球另走promotion |
| reset/teacher | direct frame0 physical birth=`0/73`；split-ready artifact+`60/240` hold 已有；WAIT 5--25 tick期间机器人/teacher保持split-ready、球停在无接触park位；reveal原子安装来球并切measured frame0，机器人不reset。A/C leaf显式钉`backhand`。旧`.7123759904781779 s`来自 tracked interior receipt(`r4_splitready`, tick92)；A literal-center已收紧到TTC=`1.82`/tick91/wait=**`.6923799138976297 s`**，A/C materializer分别=`12/12`、`11/11`；C走独立family-C receipt。**2026-08-05 更正**：本行此前写作`.69237599 s`，那是从旧 tracked receipt 减一 tick 推出的；仓库存在两份相差`3.9e-6`的 interior 权威——旧 tape(`r3`/`r4_splitready`)receipt 的`0.7123759904781779`与现役 tape(`fresh_592835dc_take061/rematerialized_1d5d9d44`)receipt 的`0.7123799138976297`。producer 从 prepared core 重算出的正是后者，故以现役 tape receipt 为准，旧数只作历史。**2026-08-06 更正**：此前本句写作"以 code-owned 常量 `CANONICAL_TEACHER_PROJECTION`(`:201`)为准、launcher 导入期自检比对的是后者"——那条描述有误：该常量只是把同一份 tape receipt 的字节在代码里抄了一遍(22 个 `runtime_target` 字段逐一相等，3 个顶层键只是改名)，所谓"导入期自检"仅是常量与自身 sha 的循环比对，从不读磁带，因此对漂移零保护。该常量与 `CANONICAL_BASE_QUESTION` / `CANONICAL_SOURCE_TAPE` / 两个 validate 函数已于 2026-08-06 整体删除，唯一权威改为 tracked 磁带及其 `current_lm.target.task_receipt.v5.5e09858672ac.json`。误删helper已恢复。bridge schema-v3专项=`56 passed`，但共享4096 gate仍写schema-v2，A launcher在第84项正确fail-closed，未被绕过 | 把shared gate升级为严格消费v3 `reveal_to_playback_bridge`与唯一counter表；随后A/C launcher余下测试、旧interior负例→exact-source suite→S0/S1 |
| question source | A cache schema-v2保存所有active-birth semantic rows+每动作跨reset hot row，mixed Q/Q' pure/cold replay correctness已过；C=`direct_ball`且formal A/C不用immutable tape。但当前level-0 TTC grid仍强制center±1 tick并携带stratum provenance，所以“fixed-N1”是小有限题带，不是用户要求的严格单Q | 增加curriculum-owned initial-center single-Q模式；升档后才扩题，Pod断言A cold=1/warm=0、C inverse=0 |
| checkpoint/resume | action FIFO/containment schema-v4与outer optimizer/RNG/normalizer组件各自存在，但当前non-fixed-view diagnostic command payload明确`exact_resume_supported=false`，普通A/C不保存WAIT/reveal、curriculum/domain、sampler、A hot-cache/active task；所以fresh `4096x5`可跑，当前long只能是不可恢复的fresh进程，不能再写成exact-resume已闭合 | long前补所有A/C command state、211/319 normalizer与outer schema3/inner schema4的mutation-before-load preflight，并做mid-WAIT/cache/curriculum/RNG/optimizer冷恢复逐tick镜像 |
| DR | shared DR-L0 finalizer/leaf专项 host=`31 passed`；A/C launcher profile已切严格all-off DR-L0：material、joint-default offset、CoM/mass/PD、push、reset/target/proprio/body-gyro corruption与delay均关，PPO探索噪声不属于DR且保持算法配方 | exact Pod复核resolved config；nominal learnability后才以fresh lineage单轴恢复 |
| Isaac | 4096环境与admission层每GPU最多2进程的合同已有；四格尚未运行。pod-wide `.kit_boot.lock` 原来会把scale全Pod串行；补丁已把锁收窄到Kit/extension boot，真实fcntl host suite=`22 passed`。Pod headless Isaac App 同GPU0双进程overlap已PASS：B在A尚未退出且双方各占约641 MiB时进入READY，二者自然退出；这关闭boot串行，不代签两个4096场景的显存/吞吐，正式scale仍记peak/min-free，跨GPU对照在补。当前analytic `physical_ball=false` learnability不冒充PhysX outcome。pre-long barrier/reward链专项=`71 passed`；A helper已恢复，C v3/各自timing receipt仍在focused test收口 | cross-GPU lock对照→initial-center single-Q四处一致→整组回归→oracle32→A0/A1/C0/C1各4096x5；正式四格共驻只为缩短总等待，rate证据另跑exclusive ABBA |
| MuJoCo | parked-ball/reveal、A/C task/reward、runtime seals、fresh hold-bias、single-stroke timeout与RSL式timeout bootstrap已实现；native+legacy组合回归=`219 passed,2 skipped,0 failed`。exact Pod WIP r6 的A/C各`1 env×2 update`均`COMPLETE`，211/319有限，fresh WAIT canary、reset-boundary save与cold-load exact均通过；decoded mean→tape qdes最大误差`6.62e-8 rad`，mean-action projection=0，随机WAIT transition projection=`31/775=4.0%`。确定性checkpoint replay已把每个update的7个hard-terminal全部定位成`joint_actual_forbidden`：A在episode tick `70..84`，C在`69..88`，均早于nominal strike，timeout/base/table/contact/strike/landing均0；所以它证明移植主链可执行/可冷载，同时明确反证当前plant/action bootstrap可直接做4096学习。现役hold的WAIT25另有`1000/1000`、`0` hard；4σ inward mean仍只是未sealed候选 | 先做sealed current mean-only/std.02与4σ-inset同条件100+ tick诊断，区分静态漂移、探索累积、PD/plant或projection根因；正式receipt补reason/phase/tick后才能发MuJoCo scale。inset若胜出须新lineage，不能偷换r6授权。完整reward/safety、mid-episode resume、4096与cross-engine parity继续阻塞formal promotion |
| **MuJoCo GPU / mjlab lane（2026-08-06 发车 + 同日四方独立复核，明细见 §9.2.2/§9.2.3/§9.2.4）** | **已在 pod1 GPU2 发车**：`4096 env x 5 update` 跑完 `10.9 s`，PID `2862997` 实测在 index `2`（`11,290 MiB`），GPU0/GPU1 全程 `2 MiB, 0 %` 未被碰（依据是 Warp 横幅只枚举 `1` 张 CUDA 卡，不是采样密度）；`nonfinite_state=0`、吞吐 `45,706` env-step/s。**数据干净但门不可信**：08-05 双 seed 历史 run 已被事后判定无溢出（两份 `5292` 行日志 `grep -ci overflow = 0`），复跑余量 `6.29--6.65x`；但当时新加的容量看门狗只守 `nefc` 一条轴，对 broadphase 溢出**静默放行**（`--nconmax 10` 实测 `1134` 行引擎警告仍判 `PASS`），且 `--iterations 0` 零测量也会签发 `PASS`。**门已于同日重做完（§9.2.6）**：改读引擎自己的 `d.overflow`（`9` 类全覆盖），判决延迟 `480 → 20` substep，零测量判 `NO_SAMPLES`；同三条变异**修前**（退出 `0` + `PASS`、CUDA 非法访问且无收据、零样本满余量 `PASS`）**修后**全部变成非零退出 + 点名 `BROADPHASE`/`NARROWPHASE`、崩前拦住并落收据、`NO_SAMPLES`。配对实测吞吐代价 ≤ `1%`（§9.2.2 那个"探针吃掉 `9%`"是拿不同长度的两条跑比出来的，已撤回）。**门可信 ≠ 可放行**：容量数值那几条（普查收敛、策略驱动构型分布）仍在 §9.2.7。容量普查的"最坏情况 `95` 行"被证伪（合法力矩即到 `117--120`，随机构型到 `188`；**这三个数本身
也是定长窗口的下界，T9 收敛普查后是 `135--137` 与 `265`，见 §9.2.7**），"接触余量 `9.45x`" 算在了错的计数器上（真值 `~3--8x`）。**汇报口径也是坏的，而且比容量门更要紧（T11，已修完，见 §9.2.8）**：被引用的"`touch 4e-5 → 0.21`"是**加权奖励项**（上限 `4.0`）不是接触率，真正的二值接触率当时只在 eval 有——`0.12% → 49.2%/97.8%`，即比零策略强 `400--800` 倍，**一个报法像没学会、另一个是学得不错**。现在：两项改名并自带上限与核均值、二值接触率进训练曲线（配对实测代价 `13%` 吞吐，如实记）、`--report` 把"零策略对照 + 二值接触率 + run 间散布"写成会拒绝的门（`11` 条拒绝规则各有代号），`--analyze` 只给一份文件从退出 `0` 改成退出 `2`。今天两条全新 `4096 x 300` run 复现了这件事：同样这两条，旧报法是 `touch 0.003 → 0.25`，新报法是**零策略 `0.14%` → `80.7%` / `56.0%`**（`570` / `395` 倍）**2026-08-06 再补（§9.2.9）**：与 Isaac A211/C211 的逐项活值对齐台账已落地，`17` 轴 = `5` 对齐 / `10` 要紧差异 / `2` 有理由差异；每条收据从此自带该台账与一句"这是本车道内部陈述"。同轮量到一件事实：**这条 lane 的机器人在第 3 次 PPO 更新后几乎每局都在碰桌子（`0% → 100%`，两 seed 复现，主犯是球拍本身），而 Isaac 对同一事件是硬终止**；现已逐集测量并由 `--report` 拒绝（`ROBOT_LEANED_ON_THE_TABLE`），但**没有**装成硬终止（那会改训练分布，属发车决定）。| 这条 lane 是 court/ready/reach-touch 任务，**不代签** canonical N1：缺 measured teacher、完整 reward 层级、§9.2 的 termination union 与 cross-engine parity。**且在 §9.2.4 的 T1--T4 待办落地前，不得再引用容量门作为放行依据**——它只证明过 `nefc` 没超 。**现在还要加一条**：引用这条 lane 的任何数字前先看它收据里的 `isaac_alignment.blocking_axes`；非空就不是 Isaac 的结果 |
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
| 9 | soft-limit v2 两条通道带宽：`qdes_limit_barrier_margin_frac` + `joint_limit_margin_frac`（新键） | 均 `0.08` | 均 `0.05` | **构造性重叠**：投影包络内沿在 `0.05*span`，barrier 带宽 `0.08` → 任何被钳关节恒扣 `-0.0844/关节/步`（理论上限 `84%`），3 关节即吃掉窗内 dense 收入 `44%`，且由护栏自身造成、策略无法规避。**两条通道必须同时改**：见下方返工记 |
| 10 | MuJoCo `joint_actual_forbidden` | 与 Isaac 同为硬终止 | 同步改为不进 `exact_hard_reasons`；事件走独立 `joint_actual_forbidden_observed_ticks` / `first_..._observed`，并在收据里自陈 `terminates_episode=false` / `mode=telemetry_only`，另出 `promotion_blocking_evidence.promotion_blocked` 结论位 | 两引擎对同一物理事件必须给出相同 Done，否则 cross-engine parity 比较的是两个不同 MDP |

**第 9 条的返工：只改一半的带宽，是开机即死的配置（2026-08-06 发现）。**
首版只把 `q_des` 通道改到 `0.05`，`actual-q` 通道（`joint_limit`）留在 `0.08`，
当时写下的理由是"actual-q 不经过投影"。这条理由有两处不成立：

1. **它根本跑不起来。** `train.py:_actual_joint_limit_barrier_reward_contract` 逐字段要求
   两条通道的 `weight` / `margin_frac` / `penalty_floor` 完全相同——它们是同一条限位带的
   两个记账观察者，不是两条可独立调剂量的带。任何 ActionBall 发射在构建硬合同时立刻
   `RuntimeError: qdes/actual soft-limit barrier v2 margin_frac must match exactly`，
   A211 的 `oracle32` 就是这样被拒的（`/tmp/a211_oracle32.log`，`train.py:5559`）。
2. **底噪的算术在两条通道上一模一样。** 护栏把命令投影到 `d = 0.05*span` 后，PD 会把实际
   关节角拉到同一位置；两条通道都读 `articulation.data.soft_joint_pos_limits`，于是
   `0.08` 带宽下被钳关节的**实际角**同样恒扣 `-0.0844/关节/步`。"不经过投影"说的是
   命令通路，不是稳态位置。

已落字节：新增显式覆盖键 `joint_limit_margin_frac`（沿用 qbar 的 fail-loud 信封——不给
`joint_limit_weight` 就拒收，越界值不留半改），`HOPEPingPongActionBall.yaml` 两条通道同为
`0.05`，`audit_reward_run.py` 的 `ADOPTED_SOFT_LIMIT_MARGIN_FRAC_BY_TERM` 同批改（此前它
把 `0.08` 写成 actual 通道的"已采纳值"，等于用审计脚本给一个开不了机的配置背书）。
真正的越界（实际 q 冲过投影内沿继续贴限位）在 `0.05` 带内照罚，硬终止仍是安全底线。
教训与第 10 条同源：**一处数值有两个通道时，改一个就必须同一次改完另一个和它的审计常量**；
这次是硬合同自己拦下来的，但它拦在发射时刻，代价是一次 GPU 排队。

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

#### 5.6.2d reveal bridge 的可学性从未被验证（2026-08-05；**下表的"四格第二轴"已被取代，2026-08-06 更正**）

> **读之前先看这条（2026-08-06 就地更正）。** 本节下面那张
> 「`A0/C0` 阶跃 / `A1/C1` 插值」的表**不是现役四格**——§8.2「第二轴改版（第二次）」把第二轴
> 换成了**本体感观测噪声开关**，现役身份是
> `action_ball_211_four_grid_contract.py:115` 的 `..._proprio_obs_noise_on_v1`。
> 本节的**问题**（reveal bridge 到底可不可学）仍然有效，但它后来是被 §5.6.6/§5.6.7 用
> **另一条路**回答的：测量出来的运动学参考根本不是能产生力矩的指令，卡点在腿的几何
> （`57/57` 帧脚底离地、`35/57` 帧质心出支撑多边形、站宽差 `0.350 m`），不在桥；
> `34f8cf25` 把 reveal 的 `2.24 rad` 阶跃改成 ramp，实测只买到 `1` tick。
> **所以：不要照本节下表去设计对照格。** 详见 §5.6.13 (F)。

本文多处写“reveal 同 tick **原子切**到 measured frame0 ... 由 **dense mimic 学 bridge**”
（:1027、:1597、:1713）。这是一句**设计声明，不是已验证事实**——全文没有任何一处给出该 bridge
可学的证据，而 §5.4 的非消失引导 Gate 六条**只覆盖 task 核**，从未对唯一负责桥接的 mimic 核
做过支撑度复算。

从 tracked split-ready artifact 逐字节复算 reveal 那一 tick 的阶跃量：

| 量 | 值 |
| --- | ---: |
| pelvis 高度下降 | `0.1766 m` |
| 去 yaw 后残余 tilt | `0 -> 0.5171 rad`（`29.63 deg`） |
| 单关节最大 `abs(dq)` | `2.2434 rad`（`right_wrist_yaw`） |
| 关节偏差 L2 / L1 | `3.6719` / `13.9110` |
| 预算 | `0.6923799138976297 s` |

在该误差下唯一能精确复算的 mimic 项 `motion_global_anchor_ori` 的 raw kernel 为
`exp(-0.5171^2 / 0.4^2) = 0.1882`，即峰值的 `18.8%`；同族 body 方向核 `exp(-mse / 0.4^2)` 的
`1%` 峰值半径是 `2.146 * 0.4 = 0.858 rad`，而仅左右 `hip_pitch` 的偏差就已是 `1.2902 / 1.3270 rad`、
`knee` 为 `0.9003 / 0.9904 rad`。**若 body 方向核在桥接段进入亚 `1%` 区，则“用 dense mimic 学 bridge”
在数学上是空的**：桥接段唯一还有梯度的是位置核与 Cauchy 球拍核，而同段内 teacher 的关节/身体/球拍
速度又被硬置零（`mujoco_native/action_ball_c211_env.py:2355-2385`，`held` 条件恒真），
位置目标与被置零的速度目标互相打架。

这条在因果链的**最前面**（平衡 -> **桥接** -> 模仿 -> 击球 -> 上台）。若它不可学，四格会一起失败
且失败形态相同，A/C 这条主对照将得不到任何信息。因此四格第二轴取它：

| 格 | family | reveal 时 teacher 过渡 |
| --- | --- | --- |
| `A0` / `C0` | A211 / C211 | **阶跃**（本文现状：同 tick 原子切） |
| `A1` / `C1` | A211 / C211 | **插值**：在 `time_to_teacher_start` 窗内由 split-ready 平滑过渡到 frame0 |

插值使 mimic 核全程留在高梯度区，把桥接从“学一个看不见的目标”变成“跟一条看得见的轨迹”。
判读：插值格出现接触而阶跃格没有 -> 桥接是病灶，且本文上述三处表述必须改；两者皆无接触 ->
桥接不是瓶颈，可排除该嫌疑，下一嫌疑转向起点分布塌缩（尽调 §9）。

**被否决的第二轴候选及理由。** `init_noise_std`：参考实现无歧义（rsl_rl 上游、BeyondMimic、
`build_1` 全为 `1.0`，无反例），故 `0.02` 是本分支自身缺陷而非待测假设，四格统一取 `1.0`，
不占对照格。`seed`：独立性属 §9.1 的验收要求，可在通过后补齐，不是当前最不确定的量。
`counter_rally_v1` 与 landing 口径：只在**已经发生接触之后**才影响判读，排在桥接之后。

#### 5.6.3 尚未对齐、需单独裁决的

- **`virtual_landing` 的实际 raw 不是本文 §5.3 所写的 `legal_base` 底薪 + 中心核。** 当前 launcher 绑定的
  `take_061_unit04_bh` manifest **8/8** 均带 `counter_rally_objective`，运行时替换为
  `0.60*legal + 0.05*落点(σ=.03 m) + 0.10*方向(8°) + 0.25*速度`，并附加 `table_bounce_count==1` 条件；
  同一 flag 还把 `virtual_pass_net` 与 `virtual_spin` 静默清零。后三档对早期策略基本不可达，
  故**合法上台的实际收入是平的 `+8.4` 台阶，不是 `8.4 -> 14` 的连续梯度**。本文全文 `counter_rally` 零命中。
- **治理断链**：两个 launcher 实际发射的 profile 是 `...DRL0Learnability`，而生成 §5.3/§5.4 静态收据的
  `audit_action_ball_reward_hierarchy.py` 曾**明确拒收该 profile**（只接受 VendorV2 与非 DRL0 leaf）。
  因此 §5.3 的那份账**不是对发射配方计算的**。发车前必须让审计器接受 DRL0 leaf 并重算全部数值。
  **2026-08-06 就地更正（前半句已过期）**：审计器自 `635252f6`（2026-08-05 05:56）起已显式接受
  两片 DRL0 leaf 并按 `<leaf> -> <非 DRL0 leaf> -> VendorV2 -> VendorV1 -> ActionBall` 解析继承链
  （`audit_action_ball_reward_hierarchy.py:371-391`）。**仍然欠的是后半句**：至今没有一份
  对 DRL0 leaf 重算的静态数值收据，§5.3/§5.4 的账仍是对上一层 profile 算的。逐条复查见 §5.6.13 (D)2。
- **hold 窗口只做了一半：下限非零已满足，"随熟练收窄"的课程不存在。** Franco 2026-08-05 的要求是
  「hold 的窗口也是一点点扩大的，0 肯定是不对的；每次击球之间肯定有一些时间间隔，但确实是变化的，
  而且可以随着学习熟练一点点减小下限」。现状分两处，**不要混为一谈**：
  (1) ActionBall 训练侧的隐藏 WAIT 是 `5..25` control step，下限非零，符合要求的前半句；但它是**静态区间**，
  没有任何随 competence 收窄下限的机制。(2) `isaac_bank_exam.py` 与 `mujoco_eval_onnx.py` 里的
  `hold_steps_range` 默认 `(0, 100)` 那个 `0`，是**评测侧 resample hold**，与训练 WAIT 不是同一概念，
  不能拿来当"下限是 0"的证据。
  **暂不实现**：hold 课程会随训练进度改变题目分布，若在四格 DR-L0/DR-L0N 归因跑期间引入，
  正好破坏这四格"只差一个轴"的设计。应挂到 DR-L1 或其后的课程臂，与 `start_pose_ramp` 同批裁决，
  并且要先定清楚 competence 的度量（用什么信号驱动收窄、收窄是否可逆、回退条件）。
- **`qdes_limit_barrier_probe` / `actual_joint_limit_barrier_probe`**：与 live barrier 逐字节同一 kernel、
  返回恒零、记账幂等恒 no-op，每步在 `4096x31` 上白算两遍；但其非零权重是躲开 RewardManager
  零权重剪枝、从而让 barrier ledger 落盘的唯一手段。**省算力与保遥测冲突，待裁决**，暂不改。

#### 5.6.4 build_1 的真实早期曲线：短 episode 与力矩饱和都不是缺陷（2026-08-06）

本轮曾把两项观测读成硬件/plant 缺陷并据此提出改 ready pose 或改硬件。**两项都读错了，此处更正并给出反证数据**，
以免下一轮再拿同样的现象当阻塞。

**数据源**：`BerkeleyPingPong/hope_wbc`，jiayi（wandb 账号 `dongc_1`）。注意 yikang（`yyk956614`）名下也有
`..._build1_...` 命名的 run（`lyhm86vl`），那**不是**原版，不要拿它当基线。原版取
`830xw9hy`（`hitter_pingpong_build_fresh_r1`，3437 iter）与
`i4dxpbwy`（`hitter_pingpong_v14_batchaligned_fresh_r4`，21896 iter）。

| iter | `830xw9hy` mean_ep_len / 主终止项 | `i4dxpbwy` mean_ep_len / 主终止项 |
| --- | --- | --- |
| 0--4 | `23.1` / `base_fell_tilt=.0064 -> .303` | `23.4` / `base_fell_tilt=.964` |
| 35--60 | **`2.1` / `base_fell_tilt=1.00`** | **`2.2` / `base_fell_tilt=1.00`** |
| 120--136 | `5.3 -> 10.6` | `3.5 -> 14.3` |
| 300--377 | `36 -> 52` | **`228`** |
| 1500--1618 | `221` / `time_out=.461` | `235` / `time_out=.582` |
| 终点 | `229` / `time_out=.517`、`fell_tilt=.480` | `249` / `time_out=.985`、`fell_tilt=.0030` |

由此定两条**验收口径**，写死以防复犯：

- **早期 episode 长度约 `20..25` tick 不是异常，是基线的第 0 迭代值。** 我们 A211/C211 oracle32 实测
  `714/32 = 22.3` tick，与 build_1 的 `23.1`/`23.4` 同量级。任何"活不过隐藏 WAIT 所以 plant 不可用"的推论
  都缺乏依据。§12.3 已写过 `4096x5` 不要求初始策略有 contact/landing income；本节补上它的正面证据。
- **基线是先变差再变好：`23 -> 2` tick、`fell_tilt` 冲到 `1.00`，到 iter `300..377` 才反弹。**
  因此**四格 `scale4096` 只有 5 个 update，其区间完全落在"尚未开始下降"的最前段，看不到任何上升、
  甚至看到退化都属预期**，不得据此判失败。`scale4096` 的验收只应是：能跑完、收据落盘、
  逐 reward 组 eligible 分母可见、无 fail-closed 触发。趋势判断最早要到 iter `400+` 量级才有意义。

**同时撤回一条错误归因**：本轮曾以 `wait25_current_hold_std002_n1000.json` 中四个腕关节
（`left/right_wrist_pitch`、`left/right_wrist_yaw`）`70..83%` 的力矩饱和，推断 `±6 N·m` 腕执行器"握不住拍子"。
该推断错误：拍子约 `0.17 kg`、力臂约 `0.15 m`，静力矩量级仅 `0.25 N·m`，余量约 20 倍。饱和来自 PD 与
噪声/速度较劲，不是重力。旁证：build_1 在 3437 iter 收敛态下 `Episode_Reward/rally_joint_qdes_saturation`
仍为 `-.0994`，饱和惩罚在成熟策略上同样长期存在。**`±6 N·m` 不构成 ready pose 或硬件的否决理由。**

#### 5.6.5 `robot_hit_table=32/32` 结案：不是 keep-out 误判，也不存在跨引擎不一致（2026-08-06）

本节先撤回本文档自己提出过的一个疑点。§5.6.4 初稿曾写「MuJoCo 侧同一 split-ready hold 为 `1000` 集 x `25` tick、
`failure_count=0`，而 Isaac 是 `robot_hit_table=32/32`，故存在跨引擎不一致」。**这个对比是错的**：
两边跑的根本不是同一件事。

| | 下发的指令 | 离出生姿态 | 结果 |
| --- | --- | --- | --- |
| MuJoCo 那个 `1000/1000` | `hold_qdes`（LP 解出的保持指令） | max `.375` / rms `.137` rad | `0` 失败 |
| Isaac oracle32（等待期） | **机器人自己的出生关节角** | **`0`（就是它自己）** | `22.3` tick 后终止 |
| MuJoCo 对照 `arm=teacher0` | teacher frame 0 | max `2.243` rad（右腕偏航 `128°`），骨盆另差 `(.154, -.177, .177)` m | **tick 1** 就撞 |

在 MuJoCo 里用**同一出生状态、同一 plant、仓库自己的 guard** 补跑对照：`arm=hold` → `0/1` 失败、`30` tick 全过
（复现了那个 `1000/1000`）；`arm=teacher0` → `1/1` 失败、**tick 1 就撞**，first-hit `left_hand_Link` vs `top`，
精确 SAT `-7.2 mm`。**两个引擎在同一指令下给出同样结论，不一致不存在。**

> **就地更正（同日，本节初稿的机制描述是错的）**：初稿把 Isaac 等待期下发的指令写成 teacher frame 0，
> 因此把两边差异归结为"指令幅度不同"。**实测推翻**：等待期 `MotionCommand.joint_pos` 在 split-ready 模式下
> 返回的是**机器人自己的出生关节角**，被 `_run_teacher_qdes_oracle` 原样当位置指令发下去，于是
> `tau = kp*(q_des - q) - kd*qd = kp*0 - kd*0 = 0`，**31 个关节全零力矩**，从第 1 tick 起自由下坠。
> 旧收据本身就写着这件事：`raw_action_max_abs = 16.575954` = 出生 `right_wrist_yaw` `1.2432 / action_scale .075`；
> 若发的真是 teacher frame 0，上界只会是 `13.3354`。而且"发 teacher frame 0"对应的是 MuJoCo 那条
> **tick 1 就撞**，与 Isaac 观测到的 `22.3` tick 对不上；**零力矩慢塌**才对得上。
> 修复见 `9e4ffb5e`：等待期改发契约里 LP 解出的 `hold_qdes_joint_pos_rad`
> （`kp*(hold_qdes - q_birth)` 与存档保持力矩逐项吻合到 `3e-15`，它就是重力补偿本身）。
> 修复后平均集长 `22.31 -> 30.41` tick，集长对等待长度的斜率 `.32 -> 1.000`（每集恰好 `wait + 15`），
> **死亡时钟从"出生就开始走"变成"揭示才开始走"，等待窗口从此是白拿的**。
> 后续 `34f8cf25` 与 §5.6.6 记录了揭示阶跃假设同样被证伪、以及真正的根因。

**keep-out 无罪，且不是过期结构。** oracle32 first-hit 台账 `32/32` 全部 `obstacle="top"`；把 pod 上所有存过
`table_first_hit` 的证据文件扫全（A211 五个 run + A225 一个），**`192/192` 全是 `("top", "right_wrist_yaw_Link")`，
keepout 记录数 `0`**。针对 Franco 2026-08-06「我不确定 `table_robot_keepout` 是有用的」的三条查证：
(1) 历史触发 `0` 次；(2) **不冗余** —— 真实桌子的物理体只有 `5 cm` 台面板，可视 USD 的物理层是整网格凸包
（代码明确写了不用，会把自由空间填实），**桌底那块体积除 keepout 外没有任何碰撞体覆盖**；它在两个引擎里都是
真实碰撞体（Isaac kinematic cuboid / MuJoCo `motion_table_robot_keepout` `conaffinity=7`），不是纯判据；
(3) **不挡合法站位** —— 只覆盖桌子自身投影 `x∈[.5, 3.24]`，机器人站在 `x=0`，出生姿态双脚离它 `137 mm`。
来历是 2026-07-29 `a93ccf8f` 作为明确安全代理引入，不是调试残留。**结论：保留，删它没有证据支持。**

**撞的"top"是代理余量打出来的，不是真接触。** 终止瞬间只有 `right_hand_pingpang_Link`（手+拍整体网格的粗包围盒）
重叠：对**加了 `20 mm` 余量**的台面盒是 `-4.1 / -2.5 mm`（重叠），对**真实台面板**却是 `+24.2 / +20.7 mm`（净空）；
真实拍叶 OBB 对真实台面板有 `+32.7 / +39.7 mm`。人话：**机器人离真桌子还有 2~4 厘米，拍子离桌面 3~4 厘米，
一点没碰到**。这正是 guard docstring 自陈的行为（"can terminate before resolved physical contact"）。
`20 mm` 余量是 fail-closed 门，未改动。

**顺带证伪一条旧诊断**：曾记为「ready pose 不可达，机器人从未离开出生姿态」。用 r12 台账重算，终止瞬间拍体
离**出生位姿** `.596 / .571 m`、离**教师 frame0** 也有 `.598 / .483 m` —— 它不但动了，还跑到两个端点都不在的
位置，是欠阻尼过冲，不是没动。

**真正值得记的脆弱点（不是 bug，暂不处理）**：出生姿态把**左手**停在离真实台面板只有 `32 mm`
（对加余量盒 `12 mm`）的地方，而这是**非持拍手**；教师 frame0 自身很干净（最近间隙 `122 mm`）。
所以贴边的是出生姿态不是教师动作，任何让左臂前伸的瞬态都会立刻撞线 —— MuJoCo tick 1 撞的就是它。
**但按 §5.6.4，`22.3` tick 与 build_1 第 0 迭代的 `23.1` 同量级，终止率本身不异常，因此不构成发车阻塞**；
要压低撞桌率时，该动的是出生姿态而不是 guard，且必须走 fresh 臂、不与四格归因混变量。

#### 5.6.6 真因：测量出来的运动学参考，不是能产生力矩的指令（2026-08-06）

§5.6.5 修掉等待期的零力矩之后，`oracle32` 仍然 `robot_hit_table=32/32`，只是死因换了。本节记录后续两层，
以及最终查到的根因。**两层假设都是被测量证伪的，不是被论证推翻的。**

**第二层假设（已证伪）：揭示时的阶跃太大。** 揭示那一 tick `q_des` 从出生姿态直接跳到 teacher frame 0，
右腕偏航差 `2.24` rad，PD 需求 `-44.9 N·m` 对限幅 `6.0`（`7.48x`），另有 3 个关节同时饱和，两膝
`+247.6 / +225.1 N·m`。据此实现了 reveal bridge（`34f8cf25`）：把剩余差额按 `1/(frozen+1)` 逐步收敛，
等分落在 clip 开始推进的那一刻。

| | 加 ramp 前 | 加 ramp 后 |
| --- | ---: | ---: |
| `bridge_ramp_command_steps` | `0` | **`544`** |
| `wait_hold_command_steps` | `461` | `461` |
| `teacher_reference_command_steps` | `512` | **`0`** |
| `reveal_reference_step_max_abs_rad` | — | `2.2226` |
| 集长 min/mean/max | `20 / 30.41 / 40` | `21 / `**`31.41`**` / 41` |
| 终止 | `robot_hit_table` `32/32` | `robot_hit_table` `32/32` |

**把 `2.22` rad 从 1 步摊到约 35 步，只买到 1 个 tick。** 而 `teacher_reference_command_steps = 0`
说出了旧收据说不出的一件事：**这一跑从未走到桥的另一端** —— 每一集都死在桥中间，约 35 tick 走到 17，
指令才走完 `46%`。所以阶跃速度不是死因。**ramp 保留**：它本身是正确的仪器行为，而且正是它把
"我们以为阶跃致命"变成了一个可测量的否定。

**根因（与 §5.6.5 同一类，深一层）**：**一个测量出来的运动学参考，对受重力的双足机器人不是能产生力矩的指令。**
等待期之所以能撑住，唯一原因是契约里带了 **LP 解出来的** hold `q_des`
（`kp*(hold_qdes - q_birth)` 复现所需保持力矩：`right_hip_roll` `36.5`、`waist_pitch` `18.7`、
`left_ankle_pitch` `15.7 N·m`）。**frame 0 没有对应物，clip 的任何后续帧也没有。**
指令一旦离开 LP 解，保持力矩就衰减、机器人下沉，**而衰减速率几乎与指令移动快慢无关** ——
这正是"只多活 1 tick"的机制解释。frame 0 的腿是非对称半蹲（膝 `.62/.52` vs 站立 `.25`），
与 hold 差 `1.29/1.33` rad（`hip_pitch`）与 `.90/.99` rad（膝）。

**两条早已在仓库里、一直被读成别的东西的旁证**：
(1) §5.6.5 的 MuJoCo 单变量对照 —— 同出生状态、同 plant、同 guard，`arm=hold` `0/1` 失败跑满 30 tick，
`arm=teacher0` **第 1 tick 就终止**。两个引擎一致，而且都与阶跃大小无关。
(2) §12.3 的机械审计 `0/73` 准入，写明原因是「加速度权威、torque-speed 曲线和**逐帧逆动力学力矩**仍缺失」。
**缺的那一项，正是开环位置回放这条 clip 所需要的东西。**

**推论**：在存在逐帧可执行 `q_des`（或前馈重力/逆动力学力矩）之前，`oracle32` 的 `32/32` exact-strike 门
**对任何 clip 都不可达**。这不是门太严，是**仪器提不出它被要求回答的那个问题**。
正在按 MuJoCo `mj_inverse` 逐帧逆动力学补前馈的路线处理；若测得"即使有前馈这条 clip 也不可开环执行"，
则应重新定义该证书的含义（闭环策略可达性证书 / 运动学可解性证书），**不得为让门通过而降低门**。

**与 A 族无关的一条并列结论（C 族）**：`C211 oracle32` 跑的是**全新未训练策略**的 rollout
（`run_live_policy_episodes` 调 `runner.get_inference_policy()`），而 launcher 要求
`single_stroke == 32`、`robot_table_contact_count == 0`、并用错了 hard-termination union。
**这一条是真的定错范围**，且本仓两处早有定论：§12.4（严格零只涵盖 `qdes-hard`/`actual-hard`/`nonfinite`
三项，table/fall/too-low 属按阶段归因的行为证据）与 §8.3（「不以『必须零次』循环要求未开训 policy
已经学会平衡」）；§5.6.4 的基线更直接 —— **build_1 第 0 迭代自己也过不了这个门**。
正确词表（`STRICT_HARD_TERMINATION_UNION`、`PHYSICAL_FALL_REASONS`、`PHYSICAL_FALL_PHASES`、
`TASK_WAIT_STARTED_COUNTER`/`TASK_REVEAL_REACHED_COUNTER`）就在同一文件里、且 `scale4096` 验证器已在消费，
差的只是接线。**修法是让 oracle32 去消费 `scale4096` 已经在消费的那份守恒普查，对两类实现故障保持严格零 ——
不是删检查。**

#### 5.6.7 逐帧逆动力学力矩已经补上了；它不是这条 clip 的卡点（2026-08-06 实测）

§12.3 的机械审计把 `0/73` 准入的原因之一写成「**逐帧逆动力学力矩**仍缺失」，§5.6.6 据此推断
「补上前馈力矩，`oracle32` 才可能可达」。**力矩这一项现在补上了，结论是：它不是卡点。**
工具是 `hope_training/whole_body_tracking/scripts/audit_measured_teacher_executability.py`
（人话：把「这条 clip 能不能被当成开环位置指令发下去」拆成四个互不替代的问题逐帧给数），
在 pod1 CPU、`hope_isaac_venv`（`mujoco 3.10.0` + `scipy 1.15.3`）上跑
`Take_061_unit04_BH`（`57` 帧 / `50` fps）。**fail-closed 锚点**：先要求
`kp*(hold_qdes - q_birth)` 复现存档保持力矩，实测残差 `3.11e-15 N·m`（容差 `1e-9`），
复现不出就拒绝出报告 —— 不出一份没有校准的数。逆动力学在 **Isaac 等效 plant** 上做
（armature 用运行时表、`dof_damping = 0`、`dof_frictionloss = 0`），因为这份 `tau_ff` 是要给 Isaac 用的。

**一、力矩不是卡点，而且腕关节彻底出局。** 逐关节峰值需求（`sg7_2` 微分档，`57×31 = 1767` 格）：

| 关节 | 峰值 \|τ\| | 限幅 | 占用 | 超限帧 |
| --- | ---: | ---: | ---: | ---: |
| `waist_roll` | `45.15` | `46.0` | **`98.1%`** | `0` |
| `waist_pitch` | `76.39` | `118.0` | `64.7%` | `0` |
| `right_shoulder_yaw` | `5.71` | `24.0` | `23.8%` | `0` |
| `right_wrist_pitch` | `1.19` | `6.0` | `19.8%` | `0` |
| **`right_wrist_yaw`（持拍腕）** | **`0.98`** | `6.0` | **`16.4%`** | `0` |
| `right_wrist_roll` | `0.72` | `24.0` | `3.0%` | `0` |

**持拍腕全程只用掉 `6 N·m` 里的 `0.98`。** 此前记过的 `-44.9 N·m` 是**指令阶跃**那一刻 PD 的瞬态需求
（`kp × 2.24 rad`），不是这个动作本身要的力矩，两者差 `38` 倍。**「`6 N·m` 握不住拍子」这条到此为止**，
与 Franco `2026-08-06` 的判断一致，也与 §5.6.4 已经撤回的那次误读一致。

**唯一贴限的是 `waist_roll`**，而且它对数值微分档位敏感（加速度要微分两次，噪声被放大两次）：

| 微分档 | 峰值 \|τ\| | `waist_roll` 占用 | 全表超限格 |
| --- | ---: | ---: | ---: |
| `raw`（中心差分） | `75.78` | `126.5%` | `3 / 1767` |
| `sg5_2` | `73.83` | `107.4%` | `1 / 1767` |
| **`sg7_2`（默认）** | `76.39` | **`98.1%`** | **`0 / 1767`** |
| `sg9_3` | `75.47` | `111.0%` | `1 / 1767` |
| `sg13_3` | `98.13` | `104.6%` | `2 / 1767` |

判定：**边缘，不是硬拒**。峰值落在限幅两侧、超限格 `0..3`、完全由滤波档位决定，
不能据此说这条 clip 力矩不可行，也不能说它有余量。要定这一条，需要的是厂商 torque-speed 曲线
（§6 采用表里 `BLOCKED ON VENDOR CURVE` 的那一项），不是再换一个滤波器。

**二、真正的卡点是三件几何/静力学事实，跟力矩无关。**

| 问题 | measured clip（`57` 帧） | split-ready 出生姿态 |
| --- | --- | --- |
| 脚踩没踩到地（鞋底最低顶点离地板） | **`+10.5 .. +12.9 mm`，`57/57` 帧全部悬空** | `-1.5 mm`（压着地，`sole_floor` 门 PASS） |
| 站不站得稳（质心到双脚支撑多边形的有符号裕度） | `-25.7 .. +15.6 mm`，均值 `-7.0 mm`，**`35/57` 帧质心在支撑面外**；frame 0 是 `-11.9 mm` | **`+123.3 mm`**（`support_margin` 门 PASS） |
| 站姿宽度（两踝水平距离） | `0.6113 m`（`57` 帧内变化 `< 0.1 mm`——**这条 clip 全程不迈步**） | `0.2613 m` |

**站姿差 `0.350 m`。** 折算成脚在骨盆系里必须走的距离：左脚 `0.314 m`、右脚 `0.213 m`。
clip 的两只脚还各自外翻着站（`ankle_roll` 左 `-0.185`、右 `+0.186 rad`，鞋底离水平 `3.6°/6.0°`），
出生姿态是 `0.000°` 双脚平放。

**为什么前馈救不了**：关节力矩是**内力**。它既不能给悬空 `1 cm` 的机器人变出地面反力，
也不能把被摩擦钉住的两只脚挪开 `0.35 m`。所以 §5.6.6 结尾那句「补上逐帧逆动力学力矩，`oracle32` 才可能可达」
**不成立** —— 力矩补上了，门依然不可达，而且原因不在被测系统里。
顺带一条：仓库自己那套产出 WAIT `hold_qdes` 的双支撑地面 LP（`MujocoGroundContactLPSolver`），
对这条 clip 的 **`57` 帧全部 fail-closed 拒绝**，理由原文是
`both named feet must have active MuJoCo floor contact`。**那道门是对的，被拒的是 clip。**

**三、MuJoCo 单变量 A/B（同出生状态、同 plant、同 guard，只改下发的 `q_des`）。**
出生骨盆高度 `1.0684 m`；`hold` 是存档 LP 保持指令，`前馈` 指
`q_des = q_ref + (τ_ff + kd·q̇_ref)/kp`（`kp`/`kd` 逐关节）。

| 臂 | 步进指令，`60` tick | `20`-tick ramp，`60` tick | guard 关、`250` tick（`5 s`）后骨盆高度 |
| --- | --- | --- | --- |
| `hold`（对照） | 撞桌 `t=54` | 撞桌 `t=54` | `0.105` 倒 |
| `teacher0_raw`（现役 oracle） | 撞桌 `t=1` | 撞桌 `t=5` | `0.119` 倒 |
| `teacher0` + 全身前馈 | 撞桌 `t=1` | 撞桌 `t=4` | `0.147` 倒 |
| 腿保持 `hold` + 上身 `teacher`，**无**前馈 | 撞桌 `t=1` | 撞桌 `t=5` | `0.974` **站着** |
| 腿保持 `hold` + 上身 `teacher`，**有**前馈 | 撞桌 `t=1` | 撞桌 `t=5` | `0.946` **站着** |

人话：**只要不发 teacher 的腿，`5 s` 后机器人还站着；只要发 teacher 的腿，有没有前馈都躺在地上。**
这就是把「腿（站姿）」和「上身（姿态）」分开的单变量结果 —— 卡点在腿。
撞桌那两列五条全撞、且与站不站得住无关，那是 §5.6.5 已经定性的另一件事
（出生姿态把**非持拍的左手**停在离真实台面板 `32 mm` 处，上身一动就碰线）。

**四、三条顺带测到、必须一并记的事。**

1. **存档 `hold` 只在 `25~30` tick 尺度上被验证过。** 跑满 `5 s`，骨盆从 `1.068` 掉到 `0.105`
   （倒在 `2~4 s` 之间）。这不推翻任何已有结论（`oracle32` 集长本来就是 `30` tick 级，
   §5.6.5 那个 `1000/1000` 也是 `25` tick），但**「hold 站得住」不得外推到长跑**；
   `fixed-N1 diagnostic long` 之前必须另有长时保持证据。
2. **跨引擎执行器模型（本轮 item c）**：Isaac 侧 `joint_actuator_types` 全是 `implicit`，
   而 MuJoCo parity harness 把同一个 `kd` **显式**施加。显式阻尼的数值稳定指数 `kd·dt/M_ii`
   （`>2` 即发散）只有 `1/31` 个关节越界：`left_wrist_yaw` `2.44`（次高 `left_wrist_roll` `1.72`）。
   把 `kd` 改成隐式（`dof_damping = kd`）整套重跑，**没有一条判定翻转**
   （`hold` 撞桌 `t=54 -> 37`，`teacher0_raw` 仍 `t=1`，`5 s` 后都倒）。
   所以 §5.6.5 的跨引擎一致结论不受影响；**但这条偏离登记在案**，
   连同厂商 MJCF 自带、两个引擎都清零的 `dof_damping` `0.5..2.0` 与 `dof_frictionloss` `0.1..2.43`。
3. **前馈指令自己也未必送得出去。** `q_des = q_ref + (τ_ff + kd·q̇_ref)/kp` 有
   `78 / 1767` 格落在运行时 `executed_qdes` 包络之外，最大越界 `1.088 rad`，主要来自 `kd·q̇_ref/kp` 项
   （高速帧的阻尼补偿）。**力矩合法 ≠ 能表达出该力矩的位置指令合法**，这是位置控制接口自带的第三道限制。

**五、建议（不代签，供 Franco 裁决）。**

- **问题在重定向，不在控制。** 这份 measured bank 的 retarget 没有约束地面接触、没有约束静态平衡、
  也没有把站姿锚到 split-ready 出生姿态。两条出路：(i) 重做重定向，把「双脚踩地」「质心在支撑面内」
  「站姿等于出生姿态」写成硬约束；(ii) 把出生姿态改成 clip 自己的站姿 —— 但 clip 自己 `35/57` 帧站不住，
  所以这条路得先修 clip。**在此之前 §12.3 的 `0/73` 机械准入结论不变**，只是现在有了具体机制。
- **`oracle32` 该重定范围。** 它现在问的是「开环下发 measured `q_des` 能不能打到 `32/32`」；
  对这条 clip 这个问题**恒为否**。建议拆成两道**都能失败**的门，替换掉一道恒假的门：
  - **运动学可解性证书**：参考自身的地面接触 / 静平衡 / 站姿一致性，就用本节这四问逐 clip 判定。
    它**现在就会拒绝** `Take_061_unit04_BH`（`57/57` 悬空、`35/57` 失衡、站姿差 `0.350 m`），
    也就是说它是一道真会开火的门，不是空判。
  - **闭环策略可达性证书**：交给训练后的 policy 判，而不是零 PPO 的开环回放。
- **不放宽任何现有 fail-closed 判据。** 上面没有一条建议是降门槛：地面 LP 的拒绝保留、
  `sole_floor`/`support_margin` 静态门保留、`20 mm` 台面余量保留、`0/73` 机械准入保留。
  变的是「用哪道门去问哪个问题」，而且新门自带能开火的证据。

**收据。** 审计工具 `hope_training/whole_body_tracking/scripts/audit_measured_teacher_executability.py`
（锚点变异测试 `hope_training/whole_body_tracking/tests/test_audit_measured_teacher_executability.py`，
pod1 `10 passed`）；本节 JSON 报告 pod1 `/workspace/franco/s10_executability.json`；
A/B 与执行器移植对照 pod1 `/workspace/franco/s10_probe11.py` `s10_probe12.py`；
worktree `/workspace/franco/s10_ff_20260806`（`33c9bdc3`）。**未跑 Isaac `oracle32`**：
本节已经证明这条 clip 的开环回放必然失败且原因不在被测系统里，再跑一遍只会复现已知结果、
并占住一张 GPU；要跑的是重定向修好之后的那一次。

#### 5.6.8 `C211 oracle32` 验收门定错了范围：它要求一个没训过的策略已经会打球（2026-08-06）

**这一条跟 §5.6.6/§5.6.7 是两件事。** 那两节说的是"机器人为什么摔"（参考轨迹不可执行，卡在腿）；
本节说的是"摔了以后**发射器该不该拒绝这一跑**"。前者是被测系统的问题，后者是**量具**的问题。

**症状。** `C211 oracle32` 跑的是 `runner.get_inference_policy()` —— 一个刚初始化、**一次 PPO 更新都没做过**
的策略的 32 集 rollout（`action_ball_c211_live_oracle.run_live_policy_episodes`）。
而 `launch_action_ball_c211_diagnostic.py` 的验收要求它表现得像已经训练好的：

| 旧判据 | 人话 |
| --- | --- |
| `completion["single_stroke"] != 32` → 拒绝 | 32 集必须每集都打完一整拍 |
| 每集 `termination_reasons != ["…single_stroke_complete"]` → 拒绝 | 任何一集只要不是"打完一拍"就拒绝 |
| `by_reason != {"…single_stroke_complete": 32}` → 拒绝 | 同上，聚合口径再来一遍 |
| `any(hard[name] != 0 for name in HARD_TERMINATION_UNION)` → 拒绝 | **摔倒/太低/撞桌也算"必须零次"** |
| `safety["robot_table_contact_count"] != 0` → 拒绝 | 撞桌一次就拒绝 |

**仓库自己在三处写着不该这么要求**：

- **§12.4**：「bridge 的桌/跌倒/too-low 事件按 phase 作**行为证据**，
  qdes-hard/actual-hard/nonfinite 才是实现 strict-zero」——严格零的集合只有三项。
- **§8.3**：「fall/too-low/robot-hit-table 仍是真实 termination，但对初始 policy 是 behavioral evidence……
  **不以『必须零次』循环要求未开训 policy 已经学会平衡**」。
- **§5.6.4**：参考实现 build_1 自己第 0 迭代 `mean_episode_length ≈ 23` tick、iter `35..60` 时
  `base_fell_tilt = 1.00`。**参考实现自己也过不了这个门。**

而正确的词表**早就在同一个文件里**，并且同一个发射器的 `scale4096` 验收器已经在消费它：
`STRICT_HARD_TERMINATION_UNION`（严格零那两项）、`PHYSICAL_FALL_REASONS`、`PHYSICAL_FALL_PHASES`。
所以这不是"少了一个功能"，是**门指错了对象**。

**改法：把门指到正确的对象上，不是删掉检查。**

1. **严格零仍然是严格零，范围收到两类实现故障**：`joint_qdes_forbidden`、`joint_actual_forbidden`、
   `projection_nonfinite_count`。任一非零 → `LaunchRefused`。等强，一个字没松。
2. **摔倒 / 太低 / 撞桌改为按阶段计数上报**，不再拒绝。
3. **新增一份守恒普查**：验收器不信收据自己写的总数，拿 32 集**逐集重数**一遍
   （`_oracle_termination_census`），然后要求三条独立通道全部对上 ——
   `termination.by_reason` / `termination.phase_by_reason` / `termination.unexpected_by_reason`（重数结果）、
   `safety.hard_termination_by_reason` 与 `safety.robot_table_contact_count`（运行时另一条通道累加的结果）、
   `completion.single_stroke`。任何一处对不上就拒绝。
4. **终止原因词表收紧**：一集的原因集合必须非空、无重复、且**全部落在**
   `{single_stroke_complete} ∪ HARD_TERMINATION_UNION` 里。重定范围只放行**已知的**行为证据；
   一个没人认识的死法（例如 hold 期禁用的 `anchor_pos`）照旧拒绝。这一条是**净增**的护栏。
5. **分母补上（净增）**：WAIT 期就死掉的复位既不算一次尝试、也不进这份证据 —— 这个排除是对的，
   但它**以前完全不可见**。现在生产端（`collect_live_oracle_bundle`）发
   `rollout_census = {source_episodes_consumed, wait_only_reset_excluded, closed_attempts}`，
   一路带到收据，并要求 `closed + excluded == consumed`、`closed == 32`。
   没有这一条，"32 集里只有 3 集摔倒"可能是从 300 次 WAIT 猝死里挑出来的。
6. **收据自陈 telemetry**：`oracle32` 收据新增 `termination_census`，一眼能看出这一跑
   各阶段各原因分别多少集、strict-zero 那三项各是多少、WAIT 排除了多少次 —— 而不是只有一个 `PASS`。

**这不是在说 bridge 没问题。** 重定范围之后，一次 `32/32` 全摔的 oracle32 会 `PASS`，
但它的收据上会白纸黑字写着 `base_fell_tilt: 32`、`single_stroke_complete_count: 0`。
"这条 clip 能不能学"的裁决在 §5.6.6/§5.6.7 和训练里，不在发射器的准入门里；
把它塞进准入门的结果只是**没有任何一跑能留下证据**。

**受影响的 schema（同批全部升版，旧 artifact 一律 fail-closed 而不是被静默接受）**：
`action_ball_c211_observed_oracle_bundle_v2 -> v3`、`action_ball_c211_oracle_raw_evidence_v2 -> v3`、
`action_ball_c211_oracle_evidence_publication_v2 -> v3`、`action_ball_c211_oracle32_receipt_v2 -> v3`。

**变异测试（10 个变异体，`0` 存活）。** 只加断言不算数：逐个把新门里的**某一条**守卫改成恒真，
看有没有测试变红；一个删掉不变红的守卫就是死代码。harness 见本节收据。

| 变异体 | 改动 | 变红的测试 |
| --- | --- | --- |
| M1 | `by_reason` 不再和逐集重数比 | 守恒用例 1 |
| M2 | 阶段x原因表不再和逐集重数比 | 守恒用例 1 |
| M3 | 去掉 qdes-hard/actual-hard/nonfinite 的 strict-zero | **等强用例 3**（两项硬终止 + nonfinite） |
| M4 | 终止原因词表放开 | 词表用例 1 |
| M5 | 不再要求 WAIT 排除数守恒 | 分母用例 2 |
| M6 | safety 通道不再与终止普查交叉核对 | 守恒用例 2 |
| M7 | **把旧的错范围门装回去** | **重定范围用例 1**（未训练策略摔倒/碰桌） |
| M8 | 把旧的 `single_stroke != 32` 装回去 | 12 个 |
| M9 | `completion.single_stroke` 不再与重数绑定 | 守恒用例 1 |
| M10 | `unexpected_by_reason` 不再重算 | 守恒用例 1 |

M7/M8 就是这次改动要消除的那个回归：**旧门一装回去，"未训练策略摔倒/碰桌"立刻被误拒。**
M3 则证明重定范围没有降门槛：两类实现故障非零时，新门照旧拒绝。

**收据。** pod1、`hope_isaac_venv`。**基线对拍是真的 A/B**：另开一棵**未改动**的 `de0641be`
worktree（`/workspace/franco/c211_rescope_BASELINE_20260806`），与改动树
（`/workspace/franco/c211_oracle32_rescope_20260806`）跑**同一份 64 模块清单**
（清单 = 全仓所有提到本次四个被改脚本或 `train.py` 的测试文件，即真实爆炸半径），`pytest -n 32`：

| | 结果 |
| --- | --- |
| 基线（未改动 `de0641be`） | `17 failed, 2497 passed, 86 skipped in 111.09s` |
| 改动后 | `17 failed, 2514 passed, 86 skipped in 112.32s` |
| 失败集合 | **逐条相同**（`diff` 为空）——那 17 条在未改动树上就红，与本改动无关 |
| 差值 | `+17 passed` = 本节新增的 17 个用例 |

那 17 条既有失败分布在 `test_action_ball_table_pose_observation` / `test_audit_reward_run` /
`test_event_timing_scheduler` / `test_foot_contact_shaping` / `test_launch_a3_vendor_identity_smoke` /
`test_launch_n1_measured_vendor_v2_diagnostic` / `test_reward_flags_overrides` /
`test_training_launch_claim`，**没有一条是 C211 oracle/launcher 测试**；本节不认领也不掩盖它们。

推送后在**第三棵干净 worktree** 上按提交号复核（`/workspace/franco/c211_rescope_verify_20260806`，
`e8079c33`）：`17 failed, 2514 passed, 86 skipped in 104.59s`，与改动树逐条一致 ——
证明提交是自洽的，没有把哪个改动漏在工作区里。

变异 harness `/tmp/mutate_c211_gate.py`：`10/10` 变异体被杀，`SURVIVING MUTANTS: none`。

#### 5.6.9 手抄的 Isaac 常量改成读活值：指纹只证明"字节没动"，不证明"抄对了"（2026-08-06）

**这是同一个形状的第三次。** 前两次分别是 `5ed998f1`（把桌面终局从广相 AABB 改成精确 SAT，
**同一个提交**把复刻侧的 AST 指纹扩到覆盖新函数并重新盖章，复刻语义没跟上，两天没人发现 ——
因为旧测试的盒子全是轴对齐的，轴对齐时广相恒等于精确，根本区分不了）和 `5c4ced66`
（改了 trainability 叶子却没重钉镜像 SHA；语义其实没漂，**真正的缺陷是测试没能力说这句话**）。

**这一次的对象**：`mujoco_native` 里那批**从 Isaac 手抄过来的常量** ——
`table_termination.py` 的桌面外扩 `2 cm`、拍面盒子的中心与半轴、五段桌台的名字、碰撞代理的
路径与 SHA。**值现在全都还对得上**，所以这不是一次事故复盘，是把"离下一次粗心重钉只差一步"
这件事关掉。它们此前的全部保护只有语义 AST 指纹，而指纹只说"源文件那几个节点的字节没动过"：
源文件一动，把指纹重钉成新值是一行的事，副本跟没跟上**没有任何机制在看**。

**做法**：新增 `mujoco_native/isaac_live_constants.py`，**把 Isaac 源码里那个数直接读出来**。
`hope_env_cfg.py` 拉的是整棵 Isaac Lab，host 上装不了，所以走 AST 取值而不是 import ——
但取的是**值**，不是哈希：把 `0.02` 改成 `0.03` 再重钉一遍指纹，这里照样红。求值器是白名单式的
（常量 / 元组列表 / 模块级名字 / `list()`、`tuple()` / 序列相加），读不出来的一律 fail closed
报 blocker，不猜。范式与 `action_ball_211_abi.live_source_parity_blockers` 相同，
差别只是那边的叶子 dependency-free、可以直接 host-load 比活值。

比对的**锚点特意选在 `table_hit_done_term()` 真正塞进 `DoneTerm` 的 `params`**，而不是同名模块
常量。这样"把常量改了"和"把这个 term 改成用另一个常量"两种漂移都拦得住 —— 后者是只比模块常量
的检查会整个漏掉的一类。`verify_isaac_source_authority()` 现在对活值门 fail closed，收据自陈
`live_constant_parity` 与 `live_constant_parity_constants_compared=7`。

**第二处是另一个形状**：`n1_reward_event_kernel.py` 手抄的四个兄弟模块字节指纹
（`observed_outcome_resolver` / `n1_ball_core` / `physical_ball_scene` / `mujoco_table_scene`）。
这里常量本身就是摘要，"重钉"即"移植"，没有"钉了但没抄"的中间态；真正的缺陷是**过去只有在 pod 上
真开起一个 MuJoCo core 才会去核对** —— host 侧改了 `n1_ball_core.py` 而忘了重钉，本地全绿，
要烧一次 pod 时间才红。现在 `native_physical_event_facts_contract()` 第一件事就核对这四个摘要。
落地当天就抓到一个活的：另一位 agent 改了 `n1_ball_core.py` 尚未重钉，门当场开火。

**变异测试**（21 条新用例，全部构造成"粗一个档次的检查就抓不到"，且**每条都先替那个粗心的作者
把指纹重钉好**、再要求活值门拦下来）：

| 变异 | 为什么粗一档就抓不到 |
| --- | --- |
| 拍面半轴 `(0.082, 0.008, 0.082)` → `(0.082, 0.010, 0.082)` | 只动厚度那一维：长度、个数、对角结构全不变 |
| 五段桌台 `post_left`/`post_right` 换序 | 集合与长度一字不差，集合式检查放行 |
| 参考包络把脚和腕的顺序对调 | 同上，四个身体一个不少 |
| `"margin"` 改指向 `TABLE_HIT_FORCE_THRESHOLD_N` | `TABLE_HIT_MARGIN_M` 自己没动，只比模块常量的检查放行 |
| `margin_fraction` `0.02` → `0.05` | `0.02` 在同一文件里还有别的出处，"这个数还在"式检查放行 |
| 碰撞代理只改路径不改 SHA | 只比 SHA 的检查放行 |
| `TABLE_HIT_MARGIN_M` 改成 `os.environ` 表达式 | 必须报 `live_value_unreadable`，不得静默当"相等" |

其中 `test_repinned_margin_change_is_still_refused` 把 `5ed998f1` 完整重放一遍：**关掉活值门之后，
重钉过的指纹独自放行了漂移的源文件** —— 这就是当年发生的事，写成了一条常驻断言。

**两处确认不适合改成活值比对，理由写在码里，不硬做**：

- `COMPONENT_WORLD_AABB_GUARD_M = 1e-6`：Isaac 侧是 `.add_(1.0e-6)` 行内字面量，上游没有具名符号
  可读；它只作用于 broad-phase 预筛（判决归 15 轴 SAT），且方向是保守放大，
  已被 `_geometric_table_contact_hit_mask_unchecked` 的 AST 指纹覆盖。
- `TABLE_CONTACT_BODY_NAMES`（32 个 body）：已经在跟碰撞代理 artifact 的 `body_order` 逐位比对 ——
  那本来就是活值比对，不重复第二遍。

**收据。** pod1 worktree `/workspace/franco/livevalue_final_20260806`（`391f41c9` + 本改动），
`hope_isaac_venv`、`pytest -n 32`。基线对拍：`391f41c9` 同 14 个模块 `469 passed`，
本改动后 `490 passed`（`+21`，零回归）。提交 `61cd804c`。

> **同轮相邻发现（不在本提交内）**：`vec_env.PHASE_EE_BODY_NAMES` 抄的是父类
> `HOPEDeployParityTerminationsCfg` 的"两脚 + 两腕"，而这条 native 车道复刻的
> `HOPEActionBallTerminationsCfg` 早在 `635252f6` 就把包络**收窄成只有两脚**（腕是挥拍时要甩最远
> 的那一端，`0.25 m` 的 z 包络套上去等于在惩罚要教的动作）。该车道目前没有生产调用点
> （没有任何 launcher 装 phase reference tape），所以是潜伏漂移而不是在跑的错。
> 这一条由同轮另一位 agent 以 `mujoco_native/isaac_reference_envelope.py` 修复（直接读活值，
> 不再留第四份手抄），并复用了本提交的 `isaac_live_constants` 求值器。

#### 5.6.10 参考包络复刻错了类：抄的是父类的"两脚 + 两腕"，现役是子类的"只有两脚"（2026-08-06）

**人话一句：** MuJoCo 复刻会在 Isaac 明确放行的**腕部位移**上把这一局掐掉，而且不会响。

**这是 §5.6.9 那个形状的第四次，也是第一次抓到语义真的漂了**（前三次里 `5c4ced66` 的语义其实没漂，
`5ed998f1` 漂了并已修，§5.6.9 那批常量值全都还对得上）。

**事实链**（逐条核实过，行号以 `837e6af6` 为准）：

1. `vec_env.py` 的 `PHASE_EE_BODY_NAMES` 是四个身体：`left/right_ankle_roll_Link` +
   `left/right_wrist_yaw_Link`。那是父类 `HOPEDeployParityTerminationsCfg.ee_body_pos`
   （`A3_FEET_BODIES + A3_HAND_BODIES`）。
2. 这条 native 车道复刻的其实是**子类** `HOPEActionBallTerminationsCfg` —— 同一个文件里
   `joint_qdes_forbidden` / `joint_actual_forbidden` 的 `source_config` 自己写着这个类名。
   子类在 `635252f6` 把 `ee_body_pos` 覆写成 `list(A3_FEET_BODIES)`，**只剩双脚**。
   覆写的理由记在码里：腕是挥拍要甩最远的那一端，`0.25 m` 的 z 包络套上去等于在惩罚要教的动作，
   `build_1` V9 实测新策略几乎每次 reset 都在 `1.67` 步内被腕部 guard 掐掉。**这个覆写是对的，没动它。**
3. 所以 `exact_phase_fidelity_reasons` 会对四个身体里任一个 `|dz| > 0.25` 触发 ——
   在现役 kernel 放行的腕部位移上终止。
4. **它不会响**：相位保真的 AST 指纹选择器只点名了
   `HOPEActionBallTerminationsCfg|joint_qdes_forbidden,joint_actual_forbidden`，
   `class_header` 只哈希装饰器/基类/关键字、不含类体，所以覆写从 `635252f6` 那天起
   `EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256` **一个 bit 都没动过**。
   （原来那行注释写着"包络已收窄到脚"，是错的：那次重钉真正的原因只有 `terminate=False`。已改正。）
5. **现有测试也看不见**：它们给四个身体喂的是同一个数（`[0.0]*4` / `[x]*4`）——
   和轴对齐盒那次同一个错误，同值向量对"包络看哪几个身体"完全是瞎的。
6. 第四份手抄在磁带侧：`n1_ball_core._phase_sample_contract_fields` 写死 `len(body_order) != 4`。

**没有在跑的错，是潜伏漂移。** 全仓没有任何 launcher 装 phase reference tape
（`launch_mujoco_fixed_center_diagnostic.py` 有 `--phase-fidelity-reference-tape` 这个开关，
但没有任何 config / 脚本传它），所以这条谓词今天只在测试里跑。

**做法**（新增 `mujoco_native/isaac_reference_envelope.py`，复用 §5.6.9 的 `isaac_live_constants` 求值器）：

| 改动 | 人话 |
| --- | --- |
| `PHASE_EE_BODY_NAMES` / `PHASE_EE_BODY_POS_Z_THRESHOLD_M` 改成**从活的 cfg 读值** | 不再留第五份手抄；子类覆写了就沿子类，没覆写就沿继承往上找，和 Python 自己解析一致 |
| 指纹选择器加上 `ee_body_pos` | 以后有人再动这条覆写，`action_ball_config` 指纹会当场开火逼人重看 |
| 新增"这个类声明了哪几条 term"的集合门 | 指纹按名字点名，**新加**一条终止项它天生看不见；集合门兜住新增/删除 |
| body 名单必须全部出自活的 `A3_FEET_BODIES`/`A3_HAND_BODIES` | 大小写写错（`_link` vs `_Link`）这类 Isaac-only 拼法会被拦，个数检查看不出来 |
| 磁带侧 `len != 4` 改成"跟活的 ActionBall 名单逐位比" | 个数对、顺序错（把腕当成脚）正是要拦的那种漂移 |
| 收据自陈 `ee_body_order_mirrors_isaac_class` / `..._declared_by_isaac_class` / `..._source` / `live_declared_terms_compared` | 收据自己说清楚"我镜像的是哪个类、名单是读来的不是抄的" |

**读活值 + 指纹门是一对，缺一不可**：读活值保证"人重钉指纹之后复刻是跟着动的"，
指纹门保证"上游一动就必须有人来重钉"。单靠读活值会让上游放宽包络悄悄传导到复刻。

**变异测试**（12 条新用例，全部构造成"粗一个档次的检查就抓不到"）：

| 变异 | 为什么粗一档就抓不到 |
| --- | --- |
| **只改腕、不改脚**的位移（脚 `0.0`，腕超阈） | 现有测试四个格子喂同一个数，对这条完全是瞎的；修好后这个四长向量直接被拒 |
| 只改现役覆写（`list(A3_FEET_BODIES)` → `+ A3_HAND_BODIES`）不改复刻 | 门必须拒绝：活值比对与指纹**双双**开火 |
| 把覆写整条删掉（退回父类四个身体） | 指纹开火；且复刻读的是活值，会跟着变回四个身体，不会像 `5ed998f1` 那样停在原地 |
| 只改覆写的 `threshold` `0.25` → `0.35` | 名单一字没动，只比名单的检查放行 |
| 往 ActionBall 类里**新加**一条 `base_fell_tilt` 覆写 | 断言过指纹**确实一个 bit 没动**，只有声明项集合门抓得到 |
| 从父类**删掉** `base_too_low` | 同上，反方向 |
| 磁带 `ee_body_order` 顺序颠倒（个数不变） | 个数检查完全看不见 |
| body 名单改成 `left_ankle_roll_link`（小写 `_link`） | 个数、集合大小、拼写几乎一样；IsaacLab 大小写敏感，MuJoCo 根本查不到这个 body |
| `"body_names": _pick_bodies_at_runtime()` | 必须报 unreadable、fail closed，不得静默当"相等" |
| 负对照：在 cfg 别处追加一个无关函数 | 语义选择器不该动；防止这批门变成"文件一改就红" |

**收据。** pod1 worktree `/workspace/franco/eebody_20260806`（`837e6af6` + 本改动），
`hope_isaac_venv`、`pytest -n 64`。基线对拍（同一棵 worktree，`git stash` 前后各跑一次，
14 个模块 = 全部 `mujoco_native/tests` + 每个 import 过 `vec_env`/`n1_ball_core` 的模块）：
`837e6af6` `415 passed`，本改动后 `428 passed`（`+13`，零回归）。同批重钉两枚：
`EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256`（选择器新增 `ee_body_pos`）与
`EXPECTED_N1_BALL_CORE_SOURCE_SHA256`（`n1_ball_core.py` 改了磁带校验）。

#### 5.6.11 收口：把"指纹盖了章、语义没跟上"这个形状变成枚举题（2026-08-06）

**人话一句：** 前四次都是事后一处一处捞，因为从来没有一张"这条车道里到底有多少份手抄件"的
清单；这一轮补的是那张清单本身，外加清单在扫的过程中抓到的第四类窟窿——**终止原因的先后顺序**。

##### (a) 先验收上两轮：声称的变异测试是不是真的会开火

不看代码里的注释，直接把修复改回旧实现，看测试红不红。全部在 pod1 独立 worktree
`/workspace/franco/closeout_mut_20260806`（`dbf40773` 检出，未拷 `logs/`）上做。

| 回退的东西 | 声称 | 实测 |
| --- | --- | --- |
| `PHASE_EE_BODY_NAMES` / `..._Z_THRESHOLD_M` 从"读活值"改回旧的四身体手抄 | §5.6.10 说 9 条转红 | **至少红了 30 条**（输出截到 30 行；含点名的 5 条：`..._is_the_live_action_ball_override_not_the_parent`、`wrist_only_displacement...`、`..._carry_the_live_two_body_order`、`changing_only_the_live_override...`、`unrelated_edits...`）。比声称的多，是因为测试侧的向量宽度也改成跟着活包络走了，回退后宽度对不上——方向一致，声称偏保守 |
| `table_termination.live_isaac_constant_blockers()` 直接 `return ()`（指纹全留着不动） | §5.6.9 说重钉之后值门照样红 | **红了 6 条**（5 条 `test_repinned_*` + 1 条`test_a_runtime_value_is_reported_as_unreadable_not_as_a_match` 的 fail-closed），含那条直接复刻 `5ed998f1` 的 `test_repinned_margin_change_is_still_refused` |
| `n1_reward_event_kernel.live_source_digest_blockers()` 直接 `return ()` | §5.6.9 说 host 侧就能看见陈旧兄弟钉 | **红了 `test_the_event_facts_contract_refuses_a_stale_native_pin`** |

**结论：上两轮的变异证据核实无误**，只有一处声称偏保守（实际影响面更大）。

##### (b)(c) 全仓扫同形状：`mujoco_native` 20 个文件、309 个模块级常量，逐个过

扫的判据就是任务给的四条：同名/同义的常量与名单、只有指纹没有活值比对的跨模块一致性、
**AST 选择器覆盖面小于它保护的语义面**、以及拿"第三份手抄字面量"当期望值的测试。

扫出来最要紧的一条，是第三条判据的又一个实例，而且比 `ee_body_pos` 高一层：

**终止原因的先后是一份跨两个文件、三个类的手抄件，而它的指纹只点了一个名字。**

事实链（行号以 `dbf40773` 为准）：

1. Isaac 评估终止项的顺序不是 `hope_env_cfg.py` 一个文件能决定的。两个 HOPE 终止类最终都派生自
   **另一个文件**里的 `tracking_env_cfg.TerminationsCfg`，而 `configclass` 是 dataclass 底子：
   字段顺序 = 先父类按声明序、再接子类新加的字段，**子类覆写一条不会把它挪到队尾**。
   实际顺序 = `time_out, anchor_pos, anchor_ori, ee_body_pos`（根类）→
   `base_fell_tilt, base_too_low, robot_hit_table`（父类）→
   `joint_qdes_forbidden, joint_actual_forbidden`（子类）。
2. `vec_env.py` 把这份顺序抄成了四个元组（`EXACT_ACTIVE_/HARD_/PHASE_FIDELITY_/BASE_..._REASON_ORDER`）。
   **同一步里两条终止都成立时，排在前面的那条才是被记进收据的原因**——顺序就是"实验把锅算在谁头上"。
   §5.6.5 那次腕部 guard 的误判之所以难查，正是这一类。
3. 唯一罩着它的是 `base_config` 这枚指纹，而它的选择器写的是 `TerminationsCfg|time_out`——
   **只点了一个名字**。往根类里新加一条终止项、或者把 `anchor_pos` 和 `anchor_ori` 换个位置，
   这枚指纹一个 bit 都不会动。和 `ee_body_pos` 那个窟窿完全同形，只是高了一层。
4. 更糟的是 `live_class_chain` 原来的行为：**出了文件的基类直接结束遍历**。所以上一轮新加的
   "这个类声明了哪几条 term"集合门，压根没看过根类——顺序的整个头部无人看管。

今天**没有漂**（活值推出来的顺序与手抄的四个元组逐位相等），所以这是潜伏漂移，不是在跑的错。

**做法**（`isaac_reference_envelope.py` 扩，`vec_env.py` 接线）：

| 改动 | 人话 |
| --- | --- |
| 登记 `EXTERNAL_TERMINATION_BASES`，链遍历跨文件走到根类 | 以前"看不见的基类"= 悄悄停下；现在看不见就**报错**，断链不许当成到头了 |
| `live_termination_reason_order()` 按 dataclass 字段序推出现役顺序 | 覆写留在父类给的格子里，这一步最容易想当然写反 |
| `live_timeout_term_names()` 从 `DoneTerm(..., time_out=True)` 读"哪几条是截断" | "哪几条算硬终止"不再是隐含常识 |
| 四份原因名单必须**恰好划分**现役硬终止项（相位 / 基座与关节 / 撞桌三个桶） | 新增一条 Isaac 终止项会落进"谁都没认领"，重钉任何指纹都救不了 |
| 根类进 `DECLARED_TERMS`；`base_config` 选择器点全四条并重钉 | 新增/删除项由集合门兜底，顺序由逐位比兜底，字节改动由指纹逼人重看 |
| 收据自陈 `live_reason_order_compared` / `live_reason_order_class_chain` | 收据自己说清楚"我比过顺序，比的是这三个类" |

##### (d) 通用护栏：新增一处只有指纹保护的手抄常量，这件事本身会被测试发现

新增 `mujoco_native/mirrored_constant_registry.py`。测试把 `mujoco_native` 下每个文件的**模块级常量
全数枚举出来**，要求每一个都被显式分类；**没有兜底通配，没有"其余的都算本地常量"**。
新加一个常量而不分类——当场红；新加一个文件而不登记——当场红。

理由档位是一张封闭词表（自由文本的理由没人能机器检）：

| 档位 | 机器检的是什么 | 计数 |
| --- | --- | --- |
| `live_value_compared` | 把常量的值跟**活值比对入口实际拿去比的那个值**对上 | 25 |
| `live_value_derived` | 赋值不是字面量（活读不许被"简化"回它当时返回的那个数） | 1 |
| `derived_in_module` | 同上，必须还是算出来的 | 55 |
| `live_source_path` | 文件真的在（上游改名在 host 测试就红，不用烧 pod 时间） | 36 |
| `pinned_file_digest` | **在 host 上重算一遍**该文件的 SHA | 4 |
| `pinned_external_digest` | 主体不是本模块指名的文件（派生载荷/只有 launcher 能解析的路径），必须写明谁在重算 | 15 |
| `flows_into_live_comparison` | 常量本身不被比，但它按序包含在某个被活值比对的对象里 | 3 |
| `not_mirrored` | 本车道自己的词汇（收据 kind、schema 版本、blocker 名单、状态枚举） | 165 |
| `mirrored_isaac_value_not_yet_live_compared` | **强制**在 `OPEN_MIRROR_DEBT` 里写清"真源在哪 / 怎么修 / 为什么这轮没修" | 5 |

`mirrored_constant_registry.py` 把**自己**也登记进去了，否则"给护栏加个常量"就成了唯一的免检通道。

**这道门诚实的边界**（写进了代码注释和一条专门的测试）：它拦不住有人把一份真手抄的 Isaac 常量
硬标成 `not_mirrored`——它没办法知道上游有没有同义的数。它保证的是**这件事必须有人动手写一行、
署上一个理由档位**，而不是像前四次那样悄无声息地混进来。真正的语义防线仍然是
`live_value_compared` 那一档；这张表的作用是让"哪些还没进那一档"变成一个可以数出来的数字。

##### 这一轮明确没做的（`OPEN_MIRROR_DEBT`，五条，全在 `action_ball_c211_env.py`）

| 常量 | 真源 | 为什么这轮没做 |
| --- | --- | --- |
| `C211_UPRIGHT_STD` | `hope_env_cfg.py` `upright_exp` 的 `params={"std": math.sqrt(0.2)}` | 求值器折不出 `math.sqrt(0.2)` 这种 Call，要先给白名单加一小撮纯函数；**放宽求值器会扩大"猜"的面**，这轮定调是收紧不放宽，不同批做 |
| `C211_ACTION_RATE_CLAMP` | `action_rate_clamped` 的 `params={"value_clamp": 9.0}` | 值是纯字面量、现在就读得出；缺的是给 `c211_env` 建一张和 `table_termination` 同款的镜像表并接进收据。要做就一次做完整张表 |
| `C211_RACKET_LONG_AXIS_LOCAL` | 球拍长轴局部方向 | 与 `C211_UPRIGHT_STD` 卡在同一个求值器限制上 |
| `TRACKED_BODY_NAMES` | Isaac 的 motion 跟踪 body 名单 | **真源还没定位到唯一符号**；没定位清楚就注册等于给门喂一个猜的答案 |
| `C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES` | `HOPERewardsCfg` 那 14 条 `RewTerm` 的名字与顺序 | 机制现成（就是本节的类链推导），但奖励项有 `weight=0.0` 的"默认跳过"语义，要先想清楚"声明了但权重为零算不算实现"，不带着未定义的语义上门 |

还有一条扫到、**判定必须做但不在这一批**的（不进 `OPEN_MIRROR_DEBT`，因为它不是常量而是选择器）：
`table_termination.verify_isaac_source_authority()`（`table_termination.py:461-478`）的 config 选择器
里，`HOPEActionBallTerminationsCfg` 只有一个 `class_header`、**没有 `class_assignments`**，而
`class_header` 只哈希装饰器/基类/关键字、不含类体。所以万一子类哪天**覆写** `robot_hit_table`
（父类那条是 `table_hit_done_term()`），这枚指纹一个 bit 都不会动 —— 和 `ee_body_pos` 一模一样的洞，
换了一条 term。**今天已经被兜住了**，但兜它的是 `vec_env` 那条链上的
`live_declared_term_blockers()`（子类多一个名字 → 集合不等 → 开火），而不是 table 车道**自己**的门；
`verify_isaac_source_authority()` 单独跑是看不见的。修法：在
`table_termination.live_isaac_constant_blockers()` 里把 `isaac_reference_envelope.
live_declared_term_blockers()` 也折进去（无环：`isaac_reference_envelope` 只依赖 `isaac_live_constants`，
而 `table_termination` 已经在 import 它）。这轮没做的原因只有一个：全量套件基线对拍已经在跑，
不想为一处已被别处兜住的洞把 A/B 作废；它需要自己的变异测试（子类覆写 `robot_hit_table` → 断言
`verify_isaac_source_authority()` 单独调用时也必须红）。

另外两条扫到但**判定不必做**的，理由记在码里：`table_termination.COMPONENT_WORLD_AABB_GUARD_M`
（Isaac 侧写成内联 `.add_(1.0e-6)`，没有可指名的符号；它只放宽广相**预筛**、方向保守，且那个函数体
已经在 callables 指纹里）、`TABLE_CONTACT_BODY_NAMES`（已经与碰撞代理产物的 `body_order` 逐位比过，
而那份产物的 SHA 本身是活值比对项，不重复造门）。

##### 变异测试（新增 25 条，每条都构造成"粗一个档次的检查就抓不到"）

顺序这条链（10 条）：

| 变异 | 为什么粗一档抓不到 |
| --- | --- |
| 把根类里 `anchor_pos` 和 `anchor_ori` **换位** | 集合一样、数量一样、每条项的字节一样，只有位置变了；测试里**先断言**集合与计数确实不变，再要求门开火 |
| 把 `base_too_low` 从父类**搬进**子类 | 整条链声明的项名、数量、集合统统不变，只有位置变了——只看"一共有哪些项"的检查完全是瞎的 |
| 往**根类**里新加一条 `base_out_of_bounds` | 测试里**直接断言旧的窄选择器指纹仍等于旧的钉子值**（`aefdf83d…`），即旧门确实一个 bit 没动；新门必须报"这条项落进谁都没认领" |
| `time_out=True` → `False` | 名字、顺序、数量全不变，只是不再算截断；硬终止名单必须跟着变 |
| 把根类改名（断链） | 必须报 `unreadable` 并 fail closed，不许像修复前那样"看不见就当到头了" |
| 负对照：在根类所在文件别处追加一个无关函数 | 门不该响；防止这批门退化成"文件一改就红" |

护栏这一层（15 条）：

| 变异 | 为什么粗一档抓不到 |
| --- | --- |
| 真往 `vec_env.py` 文件尾**追加**一行 `SOME_NEW_ISAAC_THRESHOLD_M = 0.42` | 就是任务点名的那一条：新增一处只有指纹保护的手抄常量，测试必须红 |
| 真往车道里**新增一个文件** | 模块级也不许默认放行 |
| 把活读 `_ACTION_BALL_REFERENCE_ENVELOPE = live_reference_envelope(...)` 换成它**当时返回的那个字面量** | `5ed998f1` 的形状在登记表这一层重放：值在那一刻**完全正确**（测试里先断言这一点），指纹一个 bit 都不用动，但复刻从此不再跟着上游走 |
| 把 `mirrored_isaac_termination_entries()` 里 `base_fell_tilt` 那条的镜像值换成另一个数 | `5c4ced66` 的形状：常量本身一个字节没动，只有"到底拿谁去比"变了 |
| 编辑一个被钉的文件（只加一行注释）却不重钉 | 以前只有 pod 上开起来才红 |
| 上游源文件改名 | 路径必须在 host 测试就报不在 |
| `MIRRORED_TODO` 没有配套的债务说明 / 债还了却留在清单里 / 说明是空串 | 债务清单不许烂成沉默，也不许开始骗人 |
| 某个 provider 没有任何常量引用它 | 接线断了要说出来 |
| 负对照：追加一个**小写**模块级赋值 | 普通 helper 不是常量，不该被这道门缠上 |
| 边界声明：一份真手抄常量被硬标成 `not_mirrored` | **这道门确实拦不住**——写成一条测试，免得下一个人以为它比实际更强 |

##### 收据

- 变异验收 worktree：pod1 `/workspace/franco/closeout_mut_20260806`；基线/对拍 worktree：
  `/workspace/franco/closeout_20260806`。两棵都是独立 fork，未拷 `logs/`，全程只用 CPU，
  没有碰 GPU0/GPU1/GPU2。
- 同批重钉一枚：`EXPECTED_PHASE_BASE_CONFIG_SEMANTIC_AST_SHA256`
  （`aefdf83d…` → `65e9c395…`，因为选择器从 1 个名字扩到 4 个）。
- **全量套件基线对拍**（`pytest tests hope_training/whole_body_tracking/tests
  hope_training/whole_body_tracking/mujoco_native/tests -n 64`，四次都是同一条命令）：

  | 跑法 | failed | passed | skipped | errors | 失败集合条数 |
  | --- | --- | --- | --- | --- | --- |
  | 基线 `dbf40773`（第 1 次） | 266 | 9929 | 173 | 19 | 285 |
  | 基线 `dbf40773`（第 2 次，**同一棵树同一个 commit**） | 265 | 9930 | 173 | 19 | 284 |
  | 本改动（**工作区未提交**） | 266 | 9946 | 173 | 27 | 293 |
  | 推上去的 `9ea4d0c0`（**干净检出**） | 266 | 9966 | 173 | 19 | 285 |

  **先看第 1 行和第 2 行**：同一个 commit、同一棵 worktree 连跑两次，失败集合就差了 5 条
  （`test_canonical_motion_compile_cli`、`test_joint_limit_safety`、
  `test_run_phase1_q50_persistent_supervisor`×2、`test_motion_backhand_loop_b_table_net_clearance`）。
  **这条套件本来就有一条约 2--3 条/次的抖动尾巴**，`-n 64` 下和别的 agent 抢资源时尤其明显。
  没有这条对照，任何"改动前后失败集合不完全相同"的结论都是没意义的。

  **第 3 行那 8 个多出来的 `errors` 已经查清**：全在
  `tests/test_launch_a3_vendor_identity_smoke.py`，起因是**我的工作区当时还没提交**——
  那个模块的 fixture 要求源码树干净（`test_dirty_source_refuses_before_any_runtime_or_gpu_work`
  就在那 8 条里）。提交推送后按干净检出重跑（第 4 行），`errors` 回到 `19`，和基线一模一样。
  这正是 Franco 交代的"别在共写树上跑"那条纪律的另一面：**带着未提交改动跑全量套件，
  收据本身会被污染**。

  第 4 行对第 1 行，失败集合差 4 条（进 2 出 2）；对第 2 行差 5 条（进 3 出 2）——
  **都不比基线自己跟自己的差异更大**，而且这 6 个模块 `grep -c mujoco_native` **全是 `0`**，
  本改动全部落在 `mujoco_native/` 内，没有 import 路径能传过去。

- 干净检出 `9ea4d0c0` 上把本改动直接相关的 8 个模块单独跑：`355 passed`（含新增 25 条）。
- 已知既有基线本来就红着 `266 failed / 19 errors`，不在本轮爆炸半径内；本轮没有让任何一条
  稳定绿的测试变红。

#### 5.6.12 交接：`build_1` 加桌子那天会撞上的六个坑（2026-08-06，Franco 交代）

**人话一句：** `build_1` 现在**没有桌子**（Franco 2026-08-06 确认），所以它的终止项里没有
`robot_hit_table`；**但桌子是要加的**。我们这条线为了这张桌子已经趟过六个坑，全部写在这里，
免得那天从头再趟一遍。**这不是待办，是交接单**——每条都配「症状 / 证据在哪 / 加桌子那天先做什么」。

| # | 坑 | 加桌子那天先做什么 |
| --- | --- | --- |
| 1 | **`20 mm` 代理余量会在"没真接触"时终止** | 先接受它是 fail-closed 门，别当接触真值；把 first-hit 台账打开 |
| 2 | **复刻侧曾拿广相 AABB 当终局判据** | 复刻任何终止判据前先问"这是预筛还是判决" |
| 3 | **出生姿态把非持拍左手停在离真台面板 `32 mm`** | 加桌子前先量一次出生姿态到桌面的逐 body 间隙 |
| 4 | **子类覆写一条终止项，桌子车道自己的指纹一个 bit 都不动** | 给 `HOPEActionBallTerminationsCfg` 补 `class_assignments` 选择器 |
| 5 | **多一条终止项 = 终止原因的先后全变，而先后决定"锅算在谁头上"** | 让 `live_termination_reason_order()` 重推，四份原因名单必须重新恰好划分 |
| 6 | **真桌子的物理体只有 `5 cm` 台面板，桌底那块体积没有任何碰撞体** | 决定要不要一起加 `table_robot_keepout`，否则机器人能走进桌子肚子里 |

逐条展开：

**坑 1：`20 mm` 余量是门，不是接触。** Isaac 侧 `TABLE_HIT_MARGIN_M = 0.02`
（`hope_env_cfg.py:700`），把台面盒各向外扩 `2 cm` 再判重叠。`hope_env_cfg.py:697-699` 的注释写着
"`2 cm` 是一个拍叶厚度的余量，远小于 `5 cm` 台板，所以它够不到任何没在碰桌子的东西"——
**这句话被实测推翻了**：§5.6.5 的 `32/32` 终止瞬间，触发体 `right_hand_pingpang_Link`
（手+拍整体网格的**粗包围盒**）对加余量盒是 `-4.1 / -2.5 mm`（重叠），对**真实台面板**却是
`+24.2 / +20.7 mm`（净空），真实拍叶 OBB 对真台板还有 `+32.7 / +39.7 mm`。
即门的实际伸手距离比标称 `20 mm` 还远，因为触发体是粗包围盒不是拍叶。
guard 的 docstring 自己也承认这一点（"can terminate before resolved physical contact"）。
**加桌子那天**：`robot_hit_table` 的计数**不能**当"真的撞到桌子了"读，必须配 first-hit 归因
（哪个 body、对哪块板、精确 SAT 间隙多少）才有意义；`20 mm` 本身**不许放宽**，它是 fail-closed 门。

**坑 2：广相不是判决。** `5ed998f1` 把 Isaac 的终局从 `any(component_overlap)` 换成
`_obb_aabb_sat_overlap`，**同一个 diff 里**把复刻的 AST 指纹扩到覆盖新 helper 并重新盖章——
指纹跟上了，语义没跟上，复刻的 `geometric_robot_table_hit` 继续拿保守世界 AABB 当终局，
于是复刻会在活 kernel 放行的姿态上终止，**两天没人发现**。修复见 `142874da`。
当时的测试为什么看不见：**它们造的盒子全是轴对齐的**，而轴对齐盒的广相 AABB 就是精确凸包，
两种判据**按构造恒等**。新测试专门造了一个转 `45°`、空角伸进桌子体积的盒子（旧实现读 `True`、
新实现读 `False`），并断言随机样本里**确实存在**两种判据不一致的姿态，否则 parity 测试自己判红。
**加桌子那天**：任何"复刻一条桌子终止"的动作，第一句话必须是"这是宽相预筛还是终局判决"。

**坑 3：贴边的是出生姿态，不是教师动作。** 出生姿态把**左手**（**非持拍手**）停在离真实台面板
只有 `32 mm`（对加余量盒 `12 mm`）的地方；教师 frame0 自身很干净，最近间隙 `122 mm`。
所以任何让左臂前伸的瞬态都会立刻撞线——MuJoCo `arm=teacher0` 对照 **tick 1** 撞的就是它
（first-hit `left_hand_Link` vs `top`，精确 SAT `-7.2 mm`）。
**加桌子那天**：先逐 body 量一次出生姿态到桌面的间隙；要压低撞桌率，**该动的是出生姿态而不是
guard**，且必须走 fresh 臂、不与任何归因实验混变量（§5.6.5 已定此口径）。

**坑 4：子类覆写对桌子车道自己的指纹是隐形的。** `table_termination.verify_isaac_source_authority()`
（`table_termination.py:461-478`）的 config 选择器里，`HOPEDeployParityTerminationsCfg` 有
`class_assignments|robot_hit_table`，而 `HOPEActionBallTerminationsCfg` **只有 `class_header`**；
`class_header` 只哈希装饰器/基类/关键字、**不含类体**。而 ActionBall 子类**确实在覆写终止项**
（`hope_env_cfg.py:2471` 的 `ee_body_pos` 就是覆写来的）。所以哪天子类覆写 `robot_hit_table`，
这枚指纹一个 bit 都不会动。今天被别处兜住了（`vec_env` 链上的 `live_declared_term_blockers()`
会因为声明项集合不等而开火），但 `verify_isaac_source_authority()` **单独跑是看不见的**。
修法与无环性见 §5.6.11 末尾。**加桌子那天**：先补这条选择器，再动桌子。

**坑 5：多一条终止项 = 归因顺序全变。** 两个 HOPE 终止类最终派生自 `tracking_env_cfg.TerminationsCfg`，
`configclass` 是 dataclass 底子：字段顺序 = 先父类按声明序、再接子类新加字段，**子类覆写不会挪位**。
现役顺序 = `time_out, anchor_pos, anchor_ori, ee_body_pos`（根类）→
`base_fell_tilt, base_too_low, robot_hit_table`（父类）→ `joint_qdes_forbidden, joint_actual_forbidden`（子类）。
**同一步里两条终止都成立时，排在前面的那条才被记进收据**——这个顺序就是"实验把锅算在谁头上"。
§5.6.5 那次腕部 guard 的误判之所以难查正是这一类。`build_1` 现在没有 `robot_hit_table`，
**加进来就会改动整条顺序**。好消息是 `9ea4d0c0` 之后这件事会 fail closed 而不是静默：
`live_termination_reason_order()` 按 dataclass 字段序重推，四份复刻原因名单必须**恰好划分**
现役硬终止项（相位 / 基座与关节 / 撞桌三个桶），新增一条落进"谁都没认领"就开火。
**加桌子那天**：预期会红，红了就按新顺序重推并重钉，不要绕过。

**坑 6：桌底那块体积除 keepout 外没有碰撞体。** 真实桌子的物理体只有 `5 cm` 台面板；
可视 USD 的物理层是整网格凸包（代码明确写了不用，会把自由空间填实）。
`table_robot_keepout` 在两个引擎里都是**真实碰撞体**（Isaac kinematic cuboid /
MuJoCo `motion_table_robot_keepout` `conaffinity=7`），不是纯判据；它只覆盖桌子自身投影
`x∈[.5, 3.24]`，机器人站在 `x=0`、出生姿态双脚离它 `137 mm`，**不挡合法站位**；历史触发 `0` 次。
2026-07-29 `a93ccf8f` 作为明确安全代理引入，不是调试残留。
**加桌子那天**：只加台板不加 keepout，机器人可以直接走进桌子肚子里而没有任何碰撞体阻止。

**这一节不代签什么。** 它不代签 `build_1` 的桌子该长什么样、放哪里、用哪种碰撞体；
也不代签这六条在 `build_1` 的代码结构下是同样的行号。它只代签**这六个坑我们已经踩过并留下了
可复算的数字**，`build_1` 那边不必重新发现一遍。

#### 5.6.13 全面复查"漏做的"：声称做了但没接线 / 没人读的（2026-08-06）

**人话一句：** 这一轮不查新 bug，只把本文声称"已做/待裁决/待补"的每一条与仓库现状逐条对齐。
下面按「后果 x 静默失败可能性」排序，**只列真影响判读的**。

**(A) `4096x5` 共享 gate 至今不读 `reveal_to_playback_bridge`——而生产方每个 update 都在出它。**
`action_ball_4096x5_prelong_gate.py` 的严格 v3 消费方 `_validate_reveal_bridge`（`:685`）
**全仓零调用点**：整个文件里 `reveal_to_playback_bridge` 只出现在它自己的 docstring（`:698`）。
真正跑的 `validate_semantic_updates`（`:1216`）逐字段 `row.get(...)`，**不要求键集合精确**，
所以带 bridge 的行被原样收下、bridge 一个字段都没被看过。而 A launcher 的
`scale4096` 阶段确实在调 `validate_prelong_gate`（`launch_action_ball_a211_four_arm_diagnostic.py:2445`），
生产方 `action_ball_prelong_semantics.py:957` 每个 update 都写这块记录，且
`require_bridge_telemetry` 默认 `True`（`:1673`）。
**没人读的是这些**：`question_sha256` / `sampler_contract_sha256` /
`effective_reward_recipe_sha256` / `wait_schedule_sha256` / `timing_contract_sha256` /
`wait_cohort_ticks` / `policy_dt_s` 这七项权威、逐 WAIT 档的 reveal→playback 寿命守恒、
以及 `status` 位。**尤其是 `status`**：桥没配起来时生产方返回
`{"status": "not_configured", ...}`（`:3062-3065`），而严格消费方要求
`status == "active_fail_closed"`——**接了线就是拒收，现在是静默放行**。
**该做什么**：把 `_validate_reveal_bridge` 接进 `validate_semantic_updates`，同批加变异测试
（把 `status` 改成 `not_configured` / 把任一权威 SHA 改一位 / 把某档 WAIT 的
`reveal_count != start + terminal + censored`，三条都必须转红）。
**为什么现在没做**：它会改变 gate 的拒绝面，而四格 `scale4096` 正在发；本轮不动正在跑的门。
**这一条不能只当"清理遗留"读**——`_validate_reveal_bridge` 存在这件事本身，容易让人误以为
"gate 已经严格消费 v3 了"。

**(B) `promotion_blocked` 这个结论位全仓没有任何消费者，也没有任何测试。**
§5.6.2 第 10 条记着："把一个硬门改软时，记录与阻断必须同一次改完；只出计数器等于把护栏换成了
一个需要人记得去看的数字。" 现状是**只做了改名**：`mujoco_native/vec_env.py:1846` 发出
`promotion_blocking_evidence.promotion_blocked`，代码注释就写着"只有结论会被下游读"——
但全仓 `grep promotion_blocked` 的命中**只有这个发出点**，`checkpoint.py` 零命中，
`launch_mujoco_action_ball_211_diagnostic.py` 只读 `plant_counters`（`:672-685`），
`tests/test_mujoco_native_vec_env.py` 里 `promotion` 出现 `0` 次（断言的全是原始计数
`joint_actual_forbidden_observed_ticks`）。**后果**：`joint_actual_forbidden` 从硬终止改软之后，
"不终止但卡晋级"仍然只实现了前半句；把这个结论位整段删掉，**没有一条测试会红**。
**该做什么**：给它一个真消费者（晋级/落盘路径读到 `True` 或字段缺失即拒），并配变异测试
（"收据里把 `promotion_blocked` 写死 `False`" 必须被杀）。
**为什么现在没做**：这条改的是 MuJoCo 侧晋级路径，而 §5.6.2 第 10 条明确要求"记录+阻断同批"，
不能只补一半；且它需要先定清楚"哪一步算晋级"。

**(C) 另外几个零调用点的门，逐个给判决。** 全仓扫了 `scripts/` 与 `mujoco_native/` 的
`4736` 个模块级 `def`（token 频次索引法：把全仓 `.py` 的标识符出现次数建索引，
出现次数 `<= 1` 即"除自己的 `def` 外无人提及"）。零调用点的函数共 `14` 个，
其中**门形状的 `5` 个**：(A) 那条，加下面两条（本条），再加两条在 canonical 动作库车道的
——`canonical_motion_bank_gate.py:3905 _validate_artifact_path_hash`（校验绑定路径与 SHA 同时对上）
与 `canonical_neutral_ready.py:3907 _reverify_receipt_contact_source_files`
（收据落盘前重算源动作 SHA）。后两条不在现役四格/N1 车道上，本轮只登记不判决。
其余 `9` 个是纯数值 helper（`central_diff` / `_unit_quaternion` / `roll_pitch_from_quat_wxyz` 之类），
删或留都不影响任何判据。本条要判的两个：

- `launch_action_ball_curriculum.py:844` `_require_fresh_order_sentinel`：
  **判定接线，不删**。它校验的是 MuJoCo 侧 `mujoco_teacher_motion_fitted_ball_gate.py:223`
  的 `FRESH_N5_ORDER` 与本 launcher 的 `ACTION_ORDER`（`:42`）逐位相等。
  **今天两边确实相等**（`bh_loop_c, v12_forehand_block, bh_block, s0_highpress, fh_loop_high`），
  所以这是**潜伏漂移**不是在跑的错——和 §5.6.11 那条终止顺序完全同形。
  它的成本是一次 AST 解析，收益是"两个车道的动作顺序不许各走各的"。
  测试侧 `test_launch_action_ball_curriculum.py:504-505` 已经在给它写 fixture，
  说明当初是打算接的。**注意**：这是 N5 curriculum 车道，不是现役四格，所以优先级低于 (A)。
- `launch_n1_vendor_baseline_diagnostic.py:1711` `_valid_table_guard_attribution_summary`：
  **判定接线，不删**，但要连它的生产方一起看。它要求 forensic 摘要三路自洽
  （`first_hit_total_count == terminal_count == table_count`、`category_counts` 与
  `phase_counts` 各自**恰好划分**同一个 total、`nonfinite == 0`），
  正是 §5.6.5 那次"撞桌到底算在谁头上"最需要的那种守恒账。生产方在
  `hope_commands.py:23455 _consume_table_guard_attribution_counts` /
  `:23514 _validate_table_guard_attribution_conservation`，**已经存在且有测试**
  （`test_reward_flags_mdp.py:365-409`）。差的就是 launcher 侧这一步接线。
  **为什么现在没做**：`hope_commands.py` 此刻有另一条 workflow 在改，本轮不碰。

**(D) §5.6.3「尚未对齐、需单独裁决的」四条逐条现状。**

1. **`virtual_landing` 的实际 raw ≠ §5.3 写的 `legal_base`：仍然成立，且是活的。**
   现役 launcher 绑的 `take_061_unit04_bh` manifest 确实带 `counter_rally_objective`
   （`configs/action_ball_n1_measured_20260803/fresh_core_seed0_20260803_take061_robust20n_r8_splitready/
   take_061_unit04_bh.full.manifest.v3.7d2139028427.json`），
   `hope_commands.py:5823` 据此置 `_counter_rally_enabled=True`，于是
   `hope_rewards.py:4836` 让 `virtual_pass_net` **恒返回零**、`:5019` 让 `virtual_spin` 恒返回零、
   `:4932` 与 `:4993` 把 `virtual_landing` 换成 counter-rally 的五项复合。
   **本轮新查到的一层**：A211 的 admitted 非零权重表里
   `virtual_pass_net: 20.0` 还在（`action_ball_prelong_semantics.py:141`），
   而这张表是**活值比对**的（`classify_prelong_reward_profile` 拿它跟运行时
   RewardManager 权重逐项对，漂了就拒）。也就是说：**一个权重非零、被门确认"在编"、
   但 kernel 结构上恒为零的奖励项，现在没有任何机制会说出来**。
   `4096x5` gate 只按 `motion/strike/target/outcome` 四组报分母与收入，不逐项报，
   所以 outcome 组里躺着一个死项是看不出来的。**待裁决内容不变**（§5.3/§5.4 的
   `8.4 -> 14` 连续梯度对现役配方不成立，实际是平的 `+8.4` 台阶）；
   **新增一条建议**：给"admitted 非零权重但整窗恒零收入"的项加一条会说话的检查，
   否则每次改 counter-rally 开关都要靠人记得。
2. **治理断链（审计器拒收 DRL0 leaf）：已修，但本文两处仍写着"当前它拒收"。**
   `audit_action_ball_reward_hierarchy.py:371-391` 自 `635252f6`（2026-08-05 05:56）起
   显式接受 `HOPEPingPongActionBall{A211,C211}VendorV2N1DRL0Learnability.yaml`
   并按 `<leaf> -> <非 DRL0 leaf> -> VendorV2 -> ...` 解析继承链。
   **仍然欠的是后半句**："并重算全部静态数值"——§5.3/§5.4 的那份账目前**没有**一份
   对 DRL0 leaf 重算的收据。本文顶部状态表与 §5.6.3 第 2 条的"（**当前它拒收实际发射的
   profile**）"这句话已经过期，此处就地更正。
3. **hold 窗口课程：只做了一半，已明确 defer（`838ead25`）。** 状态不变、无遗漏。
4. **两个 barrier probe 白算两遍：省算力与保遥测冲突，仍待裁决、暂不改。** 状态不变、无遗漏。

**(E) DR-L1 四格到哪一步了：合同层齐了，发射层一步都没走。**
已有：`training_contract.py:4987-5140` 的 payload/finalizer/digest、
`train.py` 的运行时合同与逐项漂移检查（`_ACTION_BALL_DR_L1_RUNTIME_ATTR` 等）、
两片 hydra 叶子（`HOPEPingPongActionBall{A211,C211}VendorV2N1DRL1Learnability.yaml`）、
tracked 候选 manifest（`configs/action_ball_n1_measured_20260805/
action_ball_211_dr_l1_restored_plant_candidate.v1.json`）、专项测试
`tests/test_action_ball_start_pose_ramp_dr_l1.py`。
**没有的**：任何能发它的路径。两个 launcher 都把 `TASK_PROFILE_ID` 与
`DR_L0_MANIFEST_SOURCE` 硬钉在 DR-L0
（`launch_action_ball_a211_four_arm_diagnostic.py:188/396`、
`launch_action_ball_c211_diagnostic.py:184`），
`action_ball_211_four_grid_contract.py` 的 DR 档身份只封了 L0 与 L0N 两个值。
**这是设计使然不是遗漏**（四格刻意只差 obs-noise 一根轴），但它的后果要写明：
**`start_pose_ramp` 与 hold 窗口课程这两件"挂到 DR-L1 同批裁决"的事，现在挂在一个
没有发射路径的档上**——不给 DR-L1 一个 lineage kind 与一次 materialize，这两件事就一直悬着。

**(F) §5.6.2d 的第二轴表与 §8.2 的第二轴表是两张不同的表，且 §5.6.2d 没标 SUPERSEDED。**
§5.6.2d 写「`A1/C1` = 在 `time_to_teacher_start` 窗内由 split-ready **插值**到 frame0」，
§8.2「第二轴改版（第二次）」写「`A1/C1` = 本体感观测噪声**开**」。现役是后者
（`action_ball_211_four_grid_contract.py:115` 的 `..._proprio_obs_noise_on_v1`）。
§5.6.2d 通篇没有 superseded 标记，单独读它会得出错误的四格定义。
另外它提的那个问题（"reveal bridge 到底可不可学"）后来是被 §5.6.6/§5.6.7 用**别的方式**
回答的（测量出来的运动学参考根本不是能产生力矩的指令；卡点在腿，不在桥），
而 `34f8cf25` 的 ramp 实测只买到 `1` tick。**这两件事没有在 §5.6.2d 就地写清楚，
下一个人会照着 §5.6.2d 去设计一组已经被证伪的对照格。**

**(G) 本 session `28` 个提交留下的尾巴，逐条查过。**
带机器可检归宿的（**没问题**）：`9ea4d0c0` 的 `OPEN_MIRROR_DEBT` 五条常量债
——注册表强制每条写"真源在哪 / 怎么修 / 为什么这轮没修"，债还了却留在清单里也判红。
**没有归宿的只有一条**：同一节末尾那个
`table_termination.verify_isaac_source_authority()` 缺 `class_assignments` 的洞
——它不是常量所以进不了 `OPEN_MIRROR_DEBT`，只活在本文的散文里。
本节 §5.6.12 坑 4 已把它写成加桌子那天的前置动作，但它**仍然没有一条会红的测试**。
其余带"这轮不做"的尾巴（`391f41c9` 建议 oracle32 重定范围、§9.2.8 的 `13%` 探针成本、
§9.2.4 E5 策略驱动构型分布未普查）都已在各自小节里写明并留了理由，不重复列。

**这一节没做的：** 本轮**零代码改动**。(A)(B)(C) 三条都要改门或改收据消费面，
按「改软硬门要连证据一起改」必须记录+阻断+变异测试同批落地；而四格 `scale4096`
正在发、`hope_commands.py` / `train.py` 另有 workflow 在改，本轮只出判决与依据。

#### 5.6.14 随机性/DR 完整性审计：逐轴三层核对、缺什么、range 合不合理（2026-08-06）

本节回答 Franco 2026-08-05 的五问（起点分支 / 站立扰动 / 起始位置 ramp / hold 窗口 /
「缺很多随机性，甚至 range 都不合理」）。**本轮只审计与记账，一行运行时代码都没改**：
四格 DR-L0/DR-L0N 归因跑正在进行，任何轴的改动都会破坏「只差一个轴」的设计。

**与 `action_ball_211_dr_l2_vendor_push_and_footwork_candidate.v1.json` 的关系**：
同日另有一条 workflow 落了一份 DR-L2 候选声明（`configs/action_ball_n1_measured_20260805/`），
覆盖推撞幅值/cadence 与步法两轴。本节是**独立复核**：两边在「episode ≈ `2.3 s`」、
「cadence 取 `[10,30] s` 而不是厂商 `[1,3] s`」、「步法应走课程侧而不是挂钟 ramp」、
「`base_travel_std_*` 在 builder 里硬编码 `[0,0]` 且无 CLI 旗标」、「32 臂里没有 `base_yaw`」
五点上**各自独立得到同一结论**。本节额外补的是那份候选没有的四件事：
(1) 全轴三层核对表（含 obs/reset/speed/terrain/起点分支）；
(2) 现役 manifest **30/32 条曲线臂上限为 `0`** 的实测；
(3) hold 窗口的三时钟账与「下限不可能是 0」的运行时证据；
(4) 三处三层不一致（`death_penalty` 活值 vs 文档四处、push 的机制位置被说错、
`action_ball_task_wait` 的 docstring 过时）。

判据沿用 §6.1 的三闸（支撑集 / 终止率放大器 / 扰动 cadence）与尽调
[`docs/research/dr_reward_external_diligence_20260731.md`](../../research/dr_reward_external_diligence_20260731.md)
§22 的分档表，不另起炉灶。每条按「机制码 / 实验史裁定 / 现役 argv」三层查；三层不一致就写出来。

**现役 argv 的取证方式**：不是读 launcher 源码推断，而是从 pod1 上一份真实落盘的
`launch_claim.json`（`/workspace/franco/l0n_tmp/bt3/…/A1-…-proprio-obs-noise-on-scale4096/launch_claim.json`
的 `canonical_payload.training_argv`）里逐条读出来的。

##### 一、逐轴现状表

「跑没跑过」一列只回答**这条轴在 ActionBall 谱系上有没有以非零幅度真正发射过**。

| 轴 | 机制码在哪 | 现役值 | 谁决定 | 哪档 DR 启用 | 跑没跑过 |
| --- | --- | --- | --- | --- | --- |
| 出生位姿 x/y/yaw ramp | `training_contract.py:5232`（端点常量）、`:5349`（校验器）、`commands.py:6207-6274`（施加）、`commands.py:1511-1526`（四条遥测） | **关**（`start_pose_ramp=None`） | `action_ball_211_four_grid_contract.py:420-423` 把它钉成 `None` | DR-L1（`cfg/task/HOPEPingPongActionBall{A,C}211VendorV2N1DRL1Learnability.yaml:47`） | **从未**。全仓没有任何 launcher 引用 `DRL1Learnability`；只有一枚 commit `4420345a` |
| 站立推扰动（六轴速度踢） | `hope_env_cfg.py:1144-1250`（`HOPEPushRobotCfg` + `apply_push_robot_event`，legacy 与 `axis_box_6d_v2` 两种拼写）、`:1258-1300`（力推）、`training_contract.py:3400/3565`（装配函数） | **关**（`task.push.enable=false`） | 叶子 `…A211VendorV2N1Learnability.yaml:55-65` 把八个字段显式写 null；argv 再关一次 | 未挂任何 DR 档（DR-L1 也把它列在 `ABSENT_EVENTS` 里） | ActionBall 谱系**从未**。旧 P1 谱系发过 14 臂（`EXP-P1-PUSH-ROBUSTNESS-20260721`），裁定 `closed_incomplete / superseded`、`no dose winner` |
| 推扰动的相位门控 | `mdp/lateral_perturbation.py`（`recovery_hold` 资格窗、strike 中断计数器、冻结 L0/L1 冲量 `0.04–0.08 m/s`、机会 `0.5 s`、`p=0.5`、脉冲 `0.1 s`、硬上限 `0.15 m/s` / `2.0 m/s²` / `200 N`） | **关**（缺 `task.lateral_perturbation` 键 = 历史无 hook 路径） | 模块自述 launch-ineligible：全场 solver-response 与吞吐门没跑过 | 不属任何 DR 档，是冻结的两格 CRN 实验单元 | **从未** |
| hold / 两拍之间的等待 | 三个独立时钟，见下面第三节 | 训练侧隐藏 `RESET_WAIT` = `5..25` policy tick + 收据派生 `pre_swing_wait_s` | `launch_action_ball_a211_…:204-205` / `launch_action_ball_c211_…:238-239` / 四格合同 `:372-373`（三处手抄同一组 `5/25`，`canonical_sha256=58aa7bb6…`） | 属 DR-L0 身份的一部分（进内容哈希） | **在跑**，但是静态区间，无任何随熟练收窄机制 |
| action delay | `hope_actions.py:1722-1723`（消费 cfg）、`:6299-6300`（字段）、`:5363-5433`（运行时收据） | `min=max=0` | 叶子 `:49-53` + argv 两条 | DR-L1 也显式保持 `0/0`（另立 `DELAY-L0/L1/L2`） | ActionBall **从未**非零。父本 `A3VendorV1.yaml:27-28` 挂着厂商 `[0,2]`，被叶子清零 |
| Kp/Kd startup | `hope_env_cfg.py:1126-1136`（`randomize_pd_gains`，Kp log_u `(0.8,1.2)` / Kd `(0.7,1.3)`） | **关** | `task.domain_rand.stable_ready_plant=true` 一个布尔同时关掉 CoM + link mass + PD 三条 | DR-L1（`stable_ready_plant=false`） | ActionBall N1 四格**从未**开过 |
| link mass ±15% | `hope_env_cfg.py:1113-1123`（scale `(0.85,1.15)`，`recompute_inertia=True`） | **关**（同上一条捆绑） | 同上 | DR-L1 | 同上 |
| torso CoM | `tracking_env_cfg.py:185-192`（x±0.025 / y,z±0.05） | **关**（同上捆绑） | 同上 | DR-L1 | 同上 |
| friction（机器人体） | `tracking_env_cfg.py:163-173`（static `(0.3,1.6)` / dynamic `(0.3,1.2)` / restitution `(0,0.5)`，64 桶） | **关** | `task.domain_rand.startup_physics_material=false` | DR-L1 | 同上 |
| 关节零点偏移 ±0.01 rad | `tracking_env_cfg.py:175-183` + `mdp/events.py:16-52`（双写 sim default 与 action offset） | **关** | `task.domain_rand.startup_joint_default_pos=false` | DR-L1（并换成**采样**解码器） | 同上 |
| 本体感 obs 噪声 | `hope_env_cfg.py:2908-2922`（A211）/`:3042-3056`（C211）：`base_ang_vel ±0.2` / `joint_pos ±0.01` / `joint_vel ±0.5`；档位定义 `training_contract.py:4910-4986` | **A1/C1 = 开，A0/C0 = 关** | argv `task.domain_rand.policy_observation_corruption` | 开 = DR-L0N，关 = DR-L0 | **正在跑**（这就是四格第二轴） |
| task/racket/time 噪声（A1 八旋钮） | `HOPEPingPongHitter.yaml:357-366` 定义全部通道 | 全零 | `…ActionBall.yaml:256-267` 把继承来的十条全部清零 + argv `action_ball_target_observation_noise=false` | 无（§6.1 判它改支撑集，晚恢复） | **从未** |
| reset noise（位姿/速度/关节） | `…ActionBall.yaml:94-108`（`joint_position_range` / `pose_range` 六轴 / `velocity_range` 六轴） | 全零 | `canonical_ready_mode` 验证器强制 | DR-L1 也保持全零（出生扰动全归 ramp） | **从未** |
| 起点分支（stand / post-swing / RSI） | `…ActionBall.yaml:77-83`；失败加权 bin-EMA 采样器在 `commands.py:4633-4684` | `stand_start_prob=1.0`、`post_swing_start_prob=0.0` | `commands.py:2079-2082` **硬性拒绝**任何别的值 | 无档可挂 | **从未**。build_1 谱系（`HOPEPingPongHitter.yaml:146/158/165`）活跑 stand .25 + post-swing .25 |
| motion 速度 `speed_scale_range` | `…ActionBall.yaml:90` | `[1.0, 1.0]` | 通用采样器被 `bind_action_ball_task_authority` 拒收 | — | **从未**。ActionBall 的等价物是收据的 `teacher_rate`：manifest 预算 `0.6..1.01`，**活值是单点 `0.85135`** |
| 球的发球分布（位置/速度/旋转/落点/出生位） | 32 条曲线臂：`mdp/action_ball_sampling.py:740-774`、`mdp/action_ball_curriculum.py:24-56`；升级判据 canary→heldout 双窗口 | 冻结在 manifest initial；`question_bank` 空、`action_ball_initial_center_single_question=true` | argv + `action_ball_diagnostic_unauthorized=true` 短路 frozen_evaluation_boundary | — | **从未解冻**。见下面第四节的量化 |
| 地形凹凸 | `tasks/tracking/terrain_patch.py`（per-env 零均值凹凸垫）、门在 `train.py:14046-14080` | 缺键 = 平地 | 没有任何 cfg/launcher 设置 `task.plant.terrain_rough_height_range` | — | **从未**，且 §3.3 已**明确拒绝**（parkour 专属，任务不对齐）。这条不算缺口 |

##### 二、三层不一致的地方（三条，都写出来）

1. **`death_penalty` 的活值已经改了，本文档四处还写着旧数。**
   四格合同 `action_ball_211_four_grid_contract.py:347` 是 `-10.0`（post-dt `-0.2`），
   pod1 收据里的 argv 也是 `task.rewards.death_penalty_weight=-10.0`。
   但本文 §5.3（行 `586`）、§6.1（行 `1623`）、§8.2（行 `2019`）、§12（行 `3834`）仍写
   post-dt `-6` / weight `-300`。**这不是措辞问题**：尽调 §22 的闸 2 把推撞、reset 噪声、
   摩擦外扩、地形、执行器延迟**全部**排在「死亡尖峰降到三库量级（post-dt ≈ `-0.2`）」之后，
   而现役发射值**已经就是 `-0.2`**。也就是说 §22 的 M0 前置在字节上已被满足，
   却没有任何一处文档跟着更新。§6.1 那句本节就地更正；**本节不据此恢复任何轴**——
   闸 1（支撑集）与闸 3（cadence）各自独立，M0 满足不等于全部放行，恢复顺序仍由 Franco 裁。
2. **`push_robot` 不是「只在 `launch_n1_vendor_baseline_diagnostic.py` 里出现」。**
   厂商那组完整六轴数值就写在 ActionBall 自己的继承链上：
   `cfg/task/HOPEPingPongActionBallA3VendorV1.yaml:43-56`，`enable: true`、
   `interval_range_s: [1.0, 3.0]`、`x/y ±0.25`、`z ±0.1`、`roll/pitch ±0.26`、`yaw ±0.39`。
   A211/C211 叶子在下游把八个字段逐个写 null 关掉。所以这条轴是**接线完整、值被关掉**，
   不是「没有机制」。
3. **`action_ball_task_wait.py` 的模块 docstring 说自己「deliberately not wired into an
   environment」，这句已经过时。** `mdp/hope_commands.py:4638-4640` 直接 import 并构造
   `ActionBallTaskWaitSchedule`，`:4995-5013` 建调度器与 highwater，`:11783-11800` 每次真
   reset 消费。按「说『没有』先查三层」，光读那句 docstring 会得出「训练侧没有 WAIT」的错误结论。

##### 三、hold 窗口：训练侧其实有三个时钟，而且下限**不可能**是 0

Franco 说「0 肯定是不对的」。现状比 §5.6.3 记的更强一层：

| 时钟 | 值 | 出处 | 变不变 |
| --- | --- | --- | --- |
| 隐藏 `RESET_WAIT` | `5..25` policy tick = `0.10..0.50 s` | 四格合同 `:372-373`；**两层独立拒零**：schedule 自己把 `min_wait_ticks` 的下界钉在 `1`（`action_ball_task_wait.py:84-89`），运行时再断言一次 `wait_ticks<=0` 即 `RuntimeError`（`hope_commands.py:11788-11791`） | 每 env 每次 reset 变 |
| 收据派生 `pre_swing_wait_s` | 活值 `0.71238 s`，硬界 `reaction_margin_s(0.1) ≤ · ≤ 1.0` | 派生式 `hope_commands.py:9981-9989`：`time_to_contact_s − reference_t_hit_s/teacher_rate`；叠加在 `:11793-11794` | 随题目变；四格只有一道题，所以是常数 |
| 遗留 `motion.hold_steps_range` | `(0,0)` | `…ActionBall.yaml:79`；`hope_commands.py:4817-4824` 硬性要求（两个时钟不能同时数同一段等待） | 不变，且不该变 |

所以**两拍之间的实际间隔 = `0.81 .. 1.21 s`，下限非零，而且运行时会在 `wait_ticks<=0` 上直接抛异常**。
`isaac_bank_exam.py:687` / `mujoco_eval_onnx.py:6549` 的 `(0,100)` 是**评测侧 resample hold**，
与训练 WAIT 不是同一概念（§5.6.3 已记）。

**真正缺的只有「随熟练收窄下限」**，而且缺得很具体：`5..25` 是内容寻址的
（seed/min/max/horizon/required_active 一起进身份哈希，A/C 共享同一个 SHA），
就地改它会**连带改掉 DR-L0 的身份**，所以必须新开 schedule kind。

**同批发现一处自相矛盾**：`training_contract.py:5228-5229` 的注释写着
「hold 窗口起点 45-60 步（build_1 的 hold），终点下限收到 20 步」，而 40 行之后的
`ACTION_BALL_START_POSE_RAMP_FOUR_CELL`（`:5270-5272`）钉的是
`hold_steps_range_start=[0,0]` / `hold_steps_range_end=[0,0]`（因为 hold 归收据所有）。
注释描述的是一个代码里并不存在的方案。而且那个「45-60」在仓内没有出处：
`HOPEPingPongHitter.yaml:158` 是 `[50,200]`（+`stand_start_min_hold: 25`），
`hope_env_cfg.py:3734`（HitterPureRally）是 `(25,125)`，两个都不是 45-60。
**建议改注释，不改常量**（常量是对的：ActionBall 的 hold 本来就归收据）。

##### 四、球的题目分布：不是「range 不合理」，是**30/32 条臂的上限就是 0**

现役 manifest
`configs/action_ball_n1_measured_20260803/fresh_592835dc_take061/rematerialized_1d5d9d44/core/take_061_unit04_bh.full.manifest.v3.fa9d108d9be1.json`
的实测值：

| 曲线臂族 | `*_std_*_max`（level=1.0 时的最大半宽） | 对照：73 库 `configs/action_ball_chingmu73_measured_v4_f10_20260803.json` |
| --- | --- | --- |
| `time_to_contact` 上/下 | **`0.12525 s`**（唯一非零） | `0.1 s` |
| `contact_offset` x/y/z 上/下 | `0.0 / 0.0 / 0.0` | `0.08 / 0.20 / 0.15 m` |
| `incoming_speed` 上/下 | `0.0` | 下 `1.849` / 上 `1.0 m/s` |
| `spin_magnitude` 上/下 | `0.0` | 上 `40 rad/s` |
| `base_spawn` x/y 上/下 | `0.0` | `0.15 / 0.25 m` |
| `base_travel` x/y 上/下 | `0.0` | `0.0`（两边都是零） |
| `landing_aim` x/y 上/下 | `0.0` | `0.25 / 0.45 m` |
| `incoming_direction` u/v 正负 | `0.0` | `0.0` |
| `spin_direction` u/v 正负 | `0.0` | `0.0` |

**即使把课程等级拉满到 `L=1.0`，现役这道题也只有接触时刻能动 ±0.125 s，其余 30 条臂纹丝不动。**
manifest 顶层还写着 `mobility_mode: "no_move"`。这是 N1 定题诊断的**有意**设计
（`action_ball_initial_center_single_question=true`），不是 bug；但它意味着
「解冻 32-arm 课程」在现役 manifest 上**几乎什么都解冻不出来**——真正要动的是重建一份
非退化 manifest（73 库那份已经有非零预算）。

另外，课程只有升不许降：`action_ball_curriculum.py` 全文没有 demote/rollback/收缩路径
（`grep -n "demote\|rollback\|shrink"` 零命中），与尽调 §13 记的「单臂不可逆」一致，
也就是 §6.1 要求的「有可逆回退」这一条**目前不成立**。

##### 五、起始位置：仓里有**两套互不知情**的机制，量级差 5–8 倍

| | `start_pose_ramp`（DR-L1） | `base_spawn_{x,y}` 曲线臂（课程） |
| --- | --- | --- |
| 动的是什么 | **物理出生点**：`commands.py:6254-6261` 在收据写完之后，直接给 `root_pos[:,0]`、`root_pos[:,1]` 加均匀偏移、给 root 乘一个 yaw 增量 | **题目里的站位**：`base_spawn_w_m` 进收据，`base_goal` / 接触点 / B_yaw 框跟着一起走 |
| 端点 | x `[-1.0, 0]`、y `±1.2625`、yaw `±30°`（正是 Franco 要的桌后 1 m / 左右各出界 0.5 m / 歪 30°） | 现役 manifest `0.0`；73 库 `±0.15 / ±0.25 m` |
| manifest 自己声明的站位硬框 | 不受约束（**没有任何校验把 ramp 和它对齐**） | `base_spawn_min/max_w_xy_m` ⇒ 中心 `[-0.192, 0.285]`、x `±0.30`、y `±0.40 m`；越界在 `build_action_ball_manifest.py:1454-1457` 会被拒 |
| 驱动信号 | **挂钟**：`common_step_counter / 96000`（`training_contract.py:5475-5494`） | competence：冻结策略 canary → 不相交 heldout 双窗口 |
| 可逆 | 否（单调 `min(1, step/N)`） | 否（同上，`action_ball_curriculum.py` 无降级路径） |

三件必须写下来的事实：

1. **ramp 的终点比 manifest 自己声明的站位框大 3.2–3.3 倍**（x `1.0` vs `0.30`；
   y `1.2625` vs `0.40`），比 73 库的课程预算大 5.1–6.7 倍。两条路径**没有任何交叉校验**：
   `grep mobility_mode` 在 `commands.py` 零命中，DR-L1 的测试文件里也没有 `base_spawn` /
   `no_move`。也就是说，一道 `mobility_mode="no_move"` 的题目配上一条把机器人扔到
   1.26 m 外的 ramp，今天没有任何门会拒。
2. **ramp 的早期斜率其实很温和，问题只在终点。** `ramp_steps=96000` 控制步 ÷
   `num_steps_per_env=24`（`cfg/algo/ppo.yaml:19`）= **4000 个 PPO update** 才到满幅。
   在 build_1 曲线刚回弹的 iter `300..377`（§5.6.4），进度只有 `7.5%`：后退 `≤0.075 m`、
   横移 `≤0.095 m`、歪 `≤2.25°`。对 `racket_position_coarse_std=0.20 m` 的核，
   `exp(−(0.095/0.20)²)=0.80`，几乎不掉收入。
   到 iter `2000`（50%）横移 `±0.63 m`，同一核只剩 `4.9e-5`；到 `4000`（100%）是 `4.9e-18`。
   长程梯度只剩 `racket_progress`（weight `10`，telescoping 到「拍到目标的距离减少量」，
   `hope_env_cfg.py:1538`），而 `base_position_weight` 被叶子钉在 `0.0`（`…A211…Learnability.yaml:99`）。
3. **相对模仿会跟着走，全局锚不会**：`motion_body_pos` 用的是
   `body_pos_relative_w`（`commands.py:9439-9449` 每步以机器人**当前**锚点重建参考），
   所以出生点被挪开不会被模仿项罚；而 `motion_global_anchor_pos` 会被罚，
   但它在 ActionBall 栈里本来就是 `None`（`hope_env_cfg.py:1528`）。这条是好消息：
   ramp 不会和模仿打架。**真正的缺口是教师本身**——`take_061_unit04_bh` 是一条站着不动的
   反手，clip 里没有任何步法，`mobility_mode` 也写着 `no_move`。

##### 六、站立推扰动：cadence 该取多少，以及 §6 那句判据本身不自洽

先把现役 episode 长度算出来（这是所有 cadence 算术的分母）：
`RESET_WAIT 0.10..0.50 s` + `pre_swing_wait 0.712 s` + `scaled_t_cycle = 1.12/0.85135 = 1.3155 s`
≈ **`2.13..2.53 s`**（单挥拍即终止，`action_ball_single_stroke_complete`），取 `2.3 s`。

| cadence | (a) 每 episode 期望推数 | (b) 落进 `0.10 s` 窄窗（尽调 §22 口径） | (c) 落进 reveal 之后全部敏感期（`87%`，DR-L2 候选口径） |
| --- | --- | --- | --- |
| `[1,3] s`（`A3VendorV1.yaml:48` 现值，厂商原值） | `1.15` | `≈0.050` | `≈1.0` |
| `[5,15] s`（尽调 §22 的终态建议） | `0.23` | `≈0.010` | `≈0.20` |
| `[10,30] s`（本文 §6 的建议） | `0.115` | `≈0.005` | `≈0.10` |

**§6 那句「目标是每 episode 命中击球窗期不超过约 `.1` 次」没有定义「击球窗期」，三种读法差 20 倍**：
(a) 与 (c) 都判 `[10,30] s` 达标、`[1,3] s` 超标约 10 倍；只有 (b) 那个 `0.10 s` 窄窗读法会
**连 `[1,3] s` 都放行**。尽调 §22 正文自己引用的是 (a)（`1.24` 次/episode）；同日的 DR-L2
候选用的是 (c)。**所以 `[10,30] s` 这个结论在三种读法里的两种下成立且互相独立地被得到，
是稳的；不稳的是判据的措辞。建议把 §6 那句钉成 (c)**：「reveal 之后到本拍结束的全部时间」
才是「击球窗期」，`0.10 s` 那个窄窗只是接触瞬间。本节不擅自改 §6 的判据，只把歧义与三个数记下来。

**更重要的一条**：`2.3 s` 的单挥拍 episode 上做 interval 推撞，`[10,30] s` 意味着
**约九成 episode 一次都不会被推到**——它是一个高方差、低暴露的处置，样本效率很差。
仓里已经有更好的答案：`mdp/lateral_perturbation.py` 的 `recovery_hold` 资格窗
（每 `0.5 s` 一次机会、`p=0.5`、脉冲 `0.1 s`、带 strike 中断计数器），
暴露量可控且天然把冲量赶出击球窗。尽调 §22 已经把这条列为「仓库里已有的答案」；
本节按现役 episode 长度的算术**支持把相位门控排在「把 interval 拉长」之前**，
而不是两者并列。它当前 launch-ineligible 的原因是全场 solver-response 与吞吐门没跑过，
这是一道**可以现在就跑的 CPU/单卡门**，不需要动四格。

##### 七、三分类结论

**(i) 机制都没有，真缺（3 条）**

| 缺什么 | 为什么算缺 |
| --- | --- |
| **hold 窗口的 competence 收窄** | `5..25` 是静态区间且进 DR-L0 身份哈希，没有任何随熟练收窄的接口；要做必须新开 schedule kind |
| **推撞的相位分层暴露统计**（pre-strike / strike / follow-through / recovery） | §6.1 明写「push 需按四相位 exposure 统计」，但 `apply_push_robot_event` 装的是裸 interval 事件，**没有任何相位计数器**；`lateral_perturbation` 那套计数器不在 push 路径上 |
| **课程的可逆回退** | `action_ball_curriculum.py` 无 demote/rollback；§6.1 要求的「可逆回退 + 独立 new-band 分母」目前只有后半句 |

**(ii) 机制有、但没接线（6 条）**

| 轴 | 断在哪一环 |
| --- | --- |
| `start_pose_ramp` | 契约 + 校验器 + 运行时 + 遥测 + 两份 DR-L1 profile + 候选 config 全齐，**但没有任何 launcher 能选中 `DRL1Learnability`** |
| DR-L1 的五条 plant 轴（friction / joint offset / CoM / link mass / Kp-Kd） | 同上：全靠那两份没人发射的 profile |
| 六轴推撞 | 值和装配函数都在（`A3VendorV1.yaml:43-56`），叶子和 argv 双重关掉 |
| 相位门控扰动 | `lateral_perturbation` 实现完整，卡在自述的 launch-ineligible 门 |
| 起点分支（post-swing / 失败加权 RSI） | `commands.py:2079-2082` 把 `stand_start_prob` 硬钉 `1.0`；失败加权 bin-EMA 采样器（`commands.py:4633-4684`）所有已注册任务都绕过它。**所以「起点分支可以直接 adapt」这句在今天不成立**：要先做尽调 §9.5 R2（把 `canonical_ready_mode` 的「契约绑定」和「reset 分布裁定」两个职能拆开），才谈得上 adapt build_1 的 25/25 分流 |
| 地形凹凸 | 机制在、门在、无人设键——但 §3.3 已明确拒绝，**这条不该补** |

**(iii) 接了线、但 range 不合理（4 条，给建议值）**

| 轴 | 现值 | 建议 | 依据 |
| --- | --- | --- | --- |
| 推撞 cadence | `A3VendorV1.yaml:48` = `[1,3] s` | 首档 `[10,30] s` + 半幅（`x/y ±0.125`、`z ±0.05`、`r/p ±0.13`、`yaw ±0.195`）；**或者直接走相位门控，跳过 cadence 这个旋钮** | 本节第六节的暴露算术；厂商 `[1,3] s` 是连续行走场景，我们是 `2.3 s` 单挥拍 |
| `start_pose_ramp` 端点 | x `-1.0 m`、y `±1.2625 m` | **首档砍到 manifest 自己的站位框以内**：x `[-0.30, 0]`、y `±0.40`、yaw `±10°`；Franco 那组 1 m / ±0.5 m / ±30° 留作**终态**，并且必须与一份 `mobility_mode="move"`、`base_travel` 预算非零的 manifest 一起上 | ramp 终点是 manifest `base_spawn_min/max` 的 3.2–3.3 倍，而 `mobility_mode="no_move"`；两者之间**没有任何校验** |
| `ramp_steps` 的驱动 | 挂钟 `96000` 步（= `4000` update） | 换成 competence 驱动，复用课程已有的 canary→heldout 双窗口 + checkpoint 化发布 | §6.1 对「来球位置/速度/时间/落点」要求「checkpointed band curriculum，有可逆回退」；起点位移改的同样是支撑集，不该用挂钟 |
| 现役 manifest 的课程预算 | 30/32 臂 `max=0` | 这不是要马上改，而是要**知道**：解冻 32-arm 课程在这份 manifest 上是 no-op，真正的动作是切到非退化 manifest（73 库那份已有预算） | 第四节的实测表 |

**没有发现问题的地方**（一并写出来，免得下轮重查）：
本体感 obs 噪声的区间（`±0.2 / ±0.01 / ±0.5`）与厂商、BeyondMimic、build_1 三家逐字一致，
不需要动；`speed_scale_range=[1,1]` 是对的（ActionBall 的速度轴是收据 `teacher_rate`，
不是通用采样器）；task/racket 噪声全零、reset 噪声全零、地形关闭这三条都有明确裁定支撑，
不是遗漏；`death_penalty` 的**活值**已经在三库量级，只是文档没跟上。

##### 八、Franco 四条要求的可执行落地方案（写作时的口径是「本轮不实现」，**已被 §九 (1) 更正**）

> **先读 §九 (1)**：本节标题里的「本轮不实现」是**写作本节的那条 workflow 收到的约束**
> （四格归因跑期间不得动任何轴），不是 Franco 的目标。Franco 2026-08-06 的口径是
> 随机性要交付，做法是把它放到自己那条臂上（即下表 P0），而不是整件事往后推。
> 下表的**内容**（挂哪档 / 什么信号 / 可不可逆 / 什么条件停车）不受这条更正影响，只有排期受影响。

四格 DR-L0/DR-L0N 归因跑期间在**同一条 leaf 上**引入任何一条都会破坏「只差一个轴」的设计；
放到 fresh lineage 的新 leaf 上则不冲突。以下是排好的下一批。

| # | 做什么 | 挂哪档 | competence 信号 | 可逆 | 回退条件 |
| --- | --- | --- | --- | --- | --- |
| P0 | 给 `DRL1Learnability` 两份 profile 接一个 launcher（它今天没有入口） | DR-L1 | — | — | 无新科学，纯接线 |
| P1 | 起始位置：**两套机制先二选一**。DR-L2 候选与第五节各自独立地判「课程侧（`base_spawn`）为主、挂钟 ramp 降为不用」——因为 Franco 那句「一点点泛化」和「随着学习熟练」是同一件事，只有课程侧由熟练度驱动、可逆、有逐臂分母。**若走课程侧**：解开 `_freeze_ball_profile` 只冻 `_initial_` 不冻 `_max_`、给 `base_travel_std_*` 补 CLI 旗标、新增 `base_yaw` 两臂（32 臂里没有偏航轴，`±30°` 今天无处安放）。**若临时先用 ramp**：首档必须收到 manifest 站位框以内（x `[-0.30,0]`、y `±0.40`、yaw `±10°`），并新增一道 fail-closed 门——ramp 端点必须落在该 action 的 `base_spawn_min/max_w_xy_m` 内、`mobility_mode=="no_move"` 时禁止非零 ramp、`x` 上界 `0` 写成门而不是巧合 | DR-L1 / DR-L2 | 课程侧 = canary→heldout；ramp 侧 = 挂钟（这正是它该被降级的原因） | 课程侧可逆（前提是先补 (i) 那条降级路径）；ramp 侧不可逆 | `base_fell_tilt` 或 `robot_hit_table` 相对同档零位移对照上升 > 0.5 pp 即停；满幅前必须先过 extrema-feasibility 门（满幅位移 `1.61 m` / `time_to_contact 1.825 s` ⇒ 需 `0.88 m/s` 走位再加一整拍） |
| P2 | hold 下限收窄：新开 `action_ball_pre_task_wait_schedule_v2`，把 `(min,max)` 做成**分档事实**（如 `5..25 → 3..25 → 2..25`），每档一个新 SHA、一次新发射边界 | DR-L1 之后的课程臂 | 复用课程的冻结策略 canary → 不相交 heldout 双窗口（`action_ball_curriculum.py`），判据用 `strike_opportunity` / `legal_landing` 的 Wilson 下界 | 是（换 argv = 换档；**并且必须允许降档**，这是 (i) 里那条「可逆回退」的第一个用户） | 任一档 `legal_landing` 率的 Wilson 下界跌破上一档，立即回上一档并冻结 |
| P3 | 推撞：先跑 `lateral_perturbation` 的全场 solver-response + 吞吐门（CPU/单卡即可），过了就用它的 `recovery_hold` 相位门控起步；**不要**先去调 interval | DR-L1 之后 | 无（固定幅度，不做课程） | 是 | 四相位 exposure 表里 strike 相位命中 > 0 即视为门控失效 |
| P4 | 起点分支：先做尽调 §9.5 R2（拆 `canonical_ready_mode` 双职能），再谈 post-swing / 失败加权 | 独立轴 | — | — | R2 本身零行为变化 |

**不做的事**：不动四格任何一格；不放宽 `wait_ticks<=0`、`pre_swing_wait ≤ 1.0`、
`base_spawn` 越界这三道现有 fail-closed 门；不在同一个 optimizer 运行内热改任何支撑集。

##### 九、补：与智元逐项的**差额**表，以及 Franco 2026-08-06 对本节定位的更正（第二次审计）

上面一到八节是同日另一条 workflow 写的。本节是**独立第二次审计**的补充，只写前八节没覆盖的三件事，
不重复已经写对的部分。

**(1) Franco 2026-08-06 的更正，直接改变本节的定位。** 原话：「随机不就是这两天我让你加上的吗？
build_1 测试下来的问题就是随机加的不够多，所以要先和智元那里对齐，然后再把起始位置和乒乓的
环境对齐」。所以：
- **§八 的「本轮不实现」不是 Franco 的目标。** 随机性是要交付的东西，不是等归因跑完再议的。
  归因洁净度的顾虑仍然成立，但正确的处理是**把随机性放到它自己那条臂上并且真把那条臂建起来**
  （P0 那一步），而不是整件事往后推。
- **判「做没做」的口径要改**：`start_pose_ramp` 挂在一个**从未被 materialize 过、也没有任何
  launcher 入口**的 DR-L1 leaf 上 = **没做**，不是「故意的」。§5.6.13 (E) 与 §八 P0 说的是同一件事。
- **优先级是 Franco 给的**：**第一步和智元对齐，第二步起始位置与乒乓环境对齐。**

**(2) 前八节没做的那张表：我们和智元逐项差多少。** 智元那两段话的原始出处是
`docs/research/dr_reward_external_diligence_20260731.md:1073`（Franco 提供的二手摘要，
不是 resolved config —— 按 §3.1 只能当首选 baseline）。前八节核对了 push 的六个幅值，
但没核对其余轴。补齐：

| 轴 | 智元 | 我们（DR-L1 恢复值） | 差在哪 |
| --- | --- | --- | --- |
| Kp / Kd | `(0.8,1.2)` / `(0.7,1.3)`，startup-only | `hope_env_cfg.py:1126-1136` 逐字相同 | **无差** |
| friction | static `(0.2,1.8)` / dynamic `(0.2,1.5)` | `tracking_env_cfg.py:163-173` static `(0.3,1.6)` / dynamic `(0.3,1.2)` | 我们更窄，但**这是既定设计，不是缺口**（Franco 2026-08-06：「摩擦应该不用改」）。**不要再提。** |
| link mass | 末端（躯干/踝/腕）`±20%` **+ pseudo-inertia** | `hope_env_cfg.py:1113-1123` 全身 `±15%`，`recompute_inertia=True` | 幅值窄 `5 pp`；作用域更宽（全身是末端的超集，不会漏）。**pseudo-inertia 独立扰动仓内无机制** |
| CoM | **全身** `±0.02 m` | `tracking_env_cfg.py:185-192` **只有 `torso_link`**，x `±0.025` / y,z `±0.05` | 幅值更宽但**只覆盖一根 link** |
| 六轴 push 幅值 | `vx/vy ±0.25`、`vz ±0.1`、`r/p ±0.26`、`yaw ±0.39` | `A3VendorV1.yaml:43-56` 逐字相同（叶子关掉） | 值无差，接线有差（§二.2 已记） |
| action delay | 每 episode `[0,2]` 控制步 | argv 钉 `0/0` | 未接线（§6.1 已裁需先补 history） |
| obs noise 通道值 | 逐通道手调 | `±0.2 / ±0.01 / ±0.5` 三通道 | **无差**（§七已核） |
| obs history | `history=8` | actor 只有一步 previous action | **这是既定设计，不是缺口**（Franco 2026-08-06：「obs history 是设计，不是缺」）。**不要再提。** |

> **2026-08-06 Franco 裁定（就地更正本表初稿）**：初稿把 friction 与 obs history 两行判成"真缺"并给了
> 对齐建议。**两条都被驳回**：摩擦不用改，obs history 是设计选择。本表保留这两行**只是为了记录已裁定**，
> 免得下一轮 review 又把它们当缺口提一遍——本文档已经出现过多次"同一件事被反复重新发现"的浪费。
> 注意 friction 那一行还有独立理由：本仓摩擦是对着 MuJoCo 标定过的（见 §9.2.1 与地形/摩擦修复记录），
> 不是从智元数值漂过来的，所以"与智元不同"本身不构成缺陷证据。
> **`action delay` 那行原写"需先补 history"——该前提随本裁定失效**，delay 要不要做需按自身理由重新评估。

**建议（依据都在左右两列，不新编数；已按上面的裁定删去被驳回的两条）**：
link mass 幅值提到 `±20%`，作用域保持全身 —— 这是本表**唯一**方向明确、代价可控的幅值差。
pseudo-inertia 独立扰动仓内无机制，属"要新写"，与幅值调整不是同一件事，单列。
CoM 是否从 `torso_link` 扩到全身**不在本轮建议**：扩全身会动到拍子所在链，
与 measured-racket authority 交叉，要单独评。
六轴 push 值与智元逐字相同、接线完整，**只是被叶子写成 `null` 再被 argv 关掉**，
所以打开它是改配置不是写实现 —— 这是本轮**最低代价、最高优先**的一条。

**(3) 加了桌子之后，起始位置那三个数还合法吗（Franco 特别问的）。**
先纠正一个容易混的前提：**我们的 Isaac ActionBall 场景已经有桌子**
（`robot_hit_table` 在 A launcher `:233-239` 的 `HARD_TERMINATION_UNION` 里）；
没有桌子的是 build_1。所以这个问题对我们是「现在就已经成立的约束」，不是未来的。

世界系：机器人地面原点 `[-0.5, -0.7625, -0.76]`，台面 `x ∈ [0, 2.74]`、`y ∈ [-1.525, 0]`。

- **合法。** ramp 的 `x` 偏移只允许 `[-1.0, 0]`，所以 root `x` 恒 `≤ -0.5` ——
  **永远在桌子近沿之外**；既然 x 方向不重叠，`y` 再怎么走（`±1.2625`，即左右各出界 `0.5 m`）
  都不可能压到台面足迹。`yaw ±30°` 只转不移，同理。
- **§5.6.12 坑 3 那个 `32 mm` 不会更糟**：ramp 只把机器人推得**离桌子更远**，
  非持拍左手到台板的余量只会变大。
- **真正会被桌子卡的是反方向。** 一旦有人把 `x` 上界从 `0` 放宽成正值（往桌子靠），
  那 `32 mm` 立刻是硬约束，而 §5.6.12 坑 1 说触发体是手+拍的粗包围盒、`20 mm` 代理余量
  实际在离真台板还有 `24 mm` 时就终止。**所以 `x` 上界 `0` 必须写成一道 fail-closed 的门，
  不是一个碰巧写成 0 的数**——这一条应当并进 §八 P1 那道新门里一起做。
- 顺带确认 §八 P1 的首档收缩仍然安全：`x [-0.30, 0]` → root `x ∈ [-0.80, -0.50]`，同样在近沿之外。

**(4) 本次落的东西**：
`configs/action_ball_n1_measured_20260805/action_ball_211_dr_l2_vendor_push_and_footwork_candidate.v1.json`
——和已有的 `..._dr_l1_restored_plant_candidate.v1.json` 同一种工件（声明式候选，无运行时代码路径），
逐轴写明智元原值 / 我们现值 / 差在哪 / 建议值 / 依据 / 三闸判定 / 停车条件，
外加 push cadence 的那笔算术和桌子合法性的推导。**P0 接线时照着对即可，不必重新推导。**

**本节没做的**：没有改 launcher、没有改 `_freeze_ball_profile`、没有重建 manifest。
三条都卡在同一条纪律上：`train.py` / `hope_commands.py` 另有 workflow 在改，
同一条 workflow 又正要用这两个 launcher 发 `C0/C1`；重建 manifest 会换 SHA，
而那串 SHA 已经钉进他们的 lineage。**按「改软硬门要连证据一起改」，
(c) 类修法一上线就会拒掉今天这份活 manifest——那正是不能背着正在发射的人做的动作。**

#### 5.6.15 诊断跑拿一本自己故意不写的账去核对一个正常增长的计数器（2026-08-06）

**人话一句：** 四格 `scale4096` 每次都跑完 update 0、然后在**存 checkpoint 那一刻**死掉，
报 `action-ball emitted task count cache drifted`。错的不是那个计数器，是**对账的范围**：
它被拿去和一本这个模式**故意一行都不写**的账做相等比较。

**现场（`c0_scale4096_s10r5/run.log:883`）。** 调用链是
`runner.save` → `_capture_environment_resume_state` → `_action_ball_exact_resume_state_dict`
→ `broker.state_dict()` → `_callback_states()` → 出生 provider 的 `state_dict()` →
`_action_ball_solver_mutable_state_dict`，在那里 `transcript_counts` 全零、
`_action_ball_emitted_task_count_by_uid` 是 `4096 x N`，于是 `raise`。
`save_interval=1` 是四格预算写死的，所以**每一个诊断格必死，不是 flake**。

**为什么说是范围错，不是计数器错。** 三条独立证据：

1. **生产方自陈。** `_action_ball_retire_previous_births` 里原文写着
   "Batched diagnostic births never enter either formal proof catalog"；
   `_action_ball_provide_births`（批量出生，只在 `diagnostic_unauthorized` 下绑）
   只有在 `fixed_view` 时才写 `provider_history` 和逐出生 transcript。
   4096 个环境每次 reset 多两次哈希表写入加一次 `sha256`，而这两本存档在诊断跑里没有消费者
   ——空是设计，不是漏写。
2. **计数器有真消费者。** `LazyActionTaskPool.state_dict()` 会拿
   `_solver_emitted_task_counts()` 去和它自己的 `_pool_emitted_task_counts()` 对账。
   把计数器停掉会立刻在别处红。
3. **命名在骗人。** `_action_ball_emitted_task_count_for` 的 docstring 叫它
   "transcript count"，报错叫它 "cache"——它既不是 transcript 的视图，也不是缓存，
   是一个自己有生产者的活计数。**已改成实话。**

**只修那一行会把崩溃往后挪两帧。** 同一个范围错还埋着两颗雷，顺序在报错点之后：
`broker.state_dict()` 会对 `_diagnostic_consumed_receipt_by_env` 里每一条收据调
`assert_issued_birth`，而它要求收据出现在**空的** `provider_history` 里；
再往后 `pool.state_dict()` 会调 `_assert_all_task_transcripts_pure()`，
逐出生去问 solver 要 root，而 `_action_ball_task_transcript_for_birth`
对**空的**目录只会抛 "unknown birth"。**这三处是同一个范围错的三个出口，必须一起改。**

**改了什么（`eccb30cd`）。**

- 状态包新增一块**跟着签名一起落盘**的自陈牌子 `task_transcript_scope`：
  `exact_per_birth` / `diagnostic_live_births_only`。读的人不必再从
  "`provider_history` 恰好是空的" 去猜。
- `diagnostic_live_births_only` 这一档的对账换成**这个模式真的在记的那本账**：
  接纳提案账 `A`。每接纳一条提案，`A` 和逐动作任务计数在同一笔生产者事务里各加一
  （`_action_ball_note(slot, "A", len(indices))` 与 `staged_uid_counts`），
  所以**多一条少一条照样红**；同时要求那两本存档**确实**是空的。
  精确那一档的严格对账**一个字没动**。
- 另外两个出口按同一个范围收口：`assert_issued_birth` 在这一档不再问空目录，
  改由 `ActionBallSampler.assert_issued_birth`（它对 `_issued_births_by_action`
  做逐字段+身份哈希的精确匹配，且诊断跑一直在维护它）承担签发证明；
  池子被明确告知逐出生 root 归它自己所有——复用 banded bank 已有的
  `pool_owns_birth_task_transcripts` 概念，不是新发明。
- 标量出生入口 `_action_ball_provide_birth` 也接上同一个判据：
  **记不记这两本账是"这次跑"的属性，不是"走了哪个入口"的属性。**

**resume 语义一起做掉，全部 fail-closed。**

- 两种 scope 的 checkpoint **互不相认**（decoder 逐字比较牌子，不匹配直接拒，
  错误里同时打印双方的值）。
- `diagnostic_live_births_only` 的 checkpoint **干脆拒绝做精确续跑**：
  它按设计就不含精确续跑所需的那半份材料；而 A211/C211 两个 launcher 的发射合同里
  本来就写着 `resume_prohibited: True` / `fresh_only: True`
  （`launch_action_ball_c211_diagnostic.py:3556/4011`、
  `launch_action_ball_a211_four_arm_diagnostic.py:4281/4290`）。
  **运行时现在说的是和发射合同同一句话**，而不是走到一半才发现缺料。
- 没有这块牌子的老状态包也拒，理由写在错误里：它无法自证空账是故意的。

**顺带修掉的一笔热路径开销。** 这块状态包**不是只在存 checkpoint 时生成**：
`LazyActionTaskPool.request_many` 每一批 reset 都会在纯净性信封里重建它。
新的 `A` 见证若各读各的，就是每次 reset 多一次 device→host 同步；已改成读一次、
对账与 payload 共用同一份主机行（`_action_ball_host_proposal_rows`），净增 `0`，
并配了一条"生产方里不许再出现 `_action_ball_live_ledger()`"的会红测试。

**收据（变异测试，`tests/test_action_ball_task_transcript_scope.py`，15 例）。**
每一条都构造成"粗一个档次就通不过"，五种改法各自杀一条指定用例：

| 把守卫改粗成 | 必须变红的用例 |
| --- | --- |
| 删掉生产方的范围分支（退回拿空 transcript 对账） | `..._can_serialize_its_clean_solver_state` |
| 留分支但不换对账（"诊断模式就别查了"） | `..._still_rejects_a_real_admitted_task_drift` |
| 牌子只查"是不是已知值"，不查是否相等 | `..._cannot_be_decoded_by_an_exact_run` |
| 去掉 live-births-only 的续跑拒绝 | `..._refuses_exact_resume_outright` |
| 停掉**精确档**的逐出生对账 | `..._reconciliation_is_unchanged_and_still_catches_drift` |

五条实测全部 `RED (good)`，恢复源码后 15 例全绿。

**全量对拍（同一个 pod worktree，`-n 64`）。** 基线 `423f5409`：
`119 failed / 7158 passed / 109 skipped / 21 errors`；本轮改动后同口径复跑，
逐 node-id 比对 **`GONE=0`**，新增两条 —— 且两条都被证明是**跑 Isaac 链把
`logs/` 留在了同一个 worktree** 造成的污染（那两个用例会整树 `shutil.copytree`，
撞上 launcher 留下的 named pipe `run.log.launch.start_gate`）：删掉 `logs/` 后
两条各自单独跑均通过，基线 worktree 上也通过。**判定：零回归。**
（教训：端到端链和全量 pytest 不要共用一个 worktree。）

**没做的、以及交接给下一个人的一件事。**

1. **诊断跑的"真续跑"没有实现，实现的是拒绝。** 要真支持，还得补：出生断言的历史锚、
   池子恢复后的逐出生 root、以及 broker `domain_claim_counts` 之外的第二个 cursor 见证。
   在 `resume_prohibited` 还立着的前提下，拒绝比半实现更诚实。
2. **`hope_commands.py` 的文件字节是 solver profile SHA 的输入**（`:5250-5287`
   把它列进 `solver_source_names`），所以这次修改让所有钉死旧 pin 的 A211/C211 manifest
   **一律拒收**——这正是 §5.6.14 末尾那条纪律说的同一件事。用仓库自己的
   `pin_action_ball_profile_contracts.py --source-rev <commit>` 重算，
   **只有一个字段动**：

   | 字段 | 旧（`take_061_unit04_bh.full.manifest.v3.653670aed246.json`） | `eccb30cd` | `308db7f0`（本轮末态） |
   | --- | --- | --- | --- |
   | `solver_profile_sha256` | `9d9a6d09…d72a0eb` | `4bee68b2…f6358360` | `3e0926c1…db8921b6` |
   | `physics_profile_sha256` | `aa5c9085…f4af85b7` | **不变** | **不变** |

   （`solver_profile_sha256` 直接哈希 `hope_commands.py` 的字节，所以**每一次**改这个文件
   都会换值；接线时以当时的 HEAD 重跑那支脚本为准，上表只是"动的是哪一个字段"的样本。）

   因此本轮**端到端只跑到 `materialize`（`MATERIALIZE_EXIT=0`）**，
   `recipe` 阶段被这道 pin 拦下（`ValueError: action-ball solver profile SHA mismatch`），
   `scale4096` 未跑。**重建 manifest / lineage 会换掉正在发射的四格身份，
   属于 Franco 的判断题，本轮不背着发射的人做。** 上面那张表就是接线时要照抄的全部内容。

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
当前四格 base death 是 **`-0.2` post-dt**（weight `-10`，见 §5.6.2 第 7 条与
`action_ball_211_four_grid_contract.py:347`；pod1 收据里的 argv 同样是
`task.rewards.death_penalty_weight=-10.0`）。旧的 `-6` post-dt（weight `-300`）与更早的
`-.6` 都**不是**本轮发射值——2026-08-06 就地更正，取证与后果见 §5.6.12 第二节：
尽调 §22 闸 2 那句「一切排在死亡尖峰降到三库量级（post-dt ≈ `-0.2`）之后」的前置**在字节上
已经满足**，但闸 1（支撑集）与闸 3（cadence）各自独立，M0 满足不等于全部放行。
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
| `A225-proto` | `225 / 318` | `[212:221]=desired contact p/v/face`，仍含 raw teacher-base 15 | 只在本身份 fresh lineage 内 | **2026-08-06 整族退役**（§8.1）；历史 artifact 保留 |
| `C225-proto` | `225 / 318` | `[212:221]=incoming ball-at-contact p/v/spin`，仍含 raw teacher-base 15 | 只在本身份 fresh lineage 内 | **2026-08-06 整族退役**（§8.1）；历史 artifact 保留 |
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
`A211/C211 x 本体感观测噪声开关` 的最小矩阵。共享 code-owned manifest 冻结同一 teacher、
base question、seed、211/319 各自 ABI、ActionBall base-safety、death manager weight `-300`、
actual-q/qdes barrier manager weight 各 `-5`，以及 qdes projection manager weight `-1`
配 `objective_weight=-5` 的同剂量目标、`metrics_only`、
`[512,256,128]` network、`entropy=.01`、delay=0 和 static contact sigma：

**2026-08-05 第二轴改版（第二次，本表已按 §5.6.2d 裁决重写）。** 上一版（§5.6.2c）把第二轴
从 PPO schedule 换成了探索包（`A0/C0` 零权重 bootstrap + sigma 0.1 对 `A1/C1` 标准初始化 +
sigma 1.0）；那一版同样为 SUPERSEDED，其 cell_id `*-base-safety-zero-weight-bootstrap-sigma0p1`
/ `*-base-safety-standard-init-sigma1p0` 已从代码中移除。

理由：探索包这一轴上，`build_1` / BeyondMimic / 正式 Hitter success lineage 的证据都指向
`std=1` + 标准初始化，**再花两格去验证"零权重 bootstrap 是不是更好"是拿实验位去证一个没人
主张的方案**。所以本轮把探索包**定死**在四格共用的标准 rsl_rl 初始化 + `init_noise_std=1.0` +
`noise_std_type=scalar`（4σ 硬内带门显式跳过），把腾出来的两格换给**本体感观测噪声开关**：

* 尽调 §22 判本体感噪声"D1 开满"，证据是外部 **9/9 库 day-1 全开** + 智元连 `play` 都保留 +
  `build_1` 全开，**零反例**；
* DR-L0 的裁定正好相反——它判这条"会改估计误差与终止率"，所以为归因先关；
* **两边都是推理，谁都没实测过**，而成本只是一个布尔。上一轮恢复的那批随机性（摩擦 /
  连杆质量 / PD / CoM / 关节零点 / 出生位姿斜坡）里，这是唯一有真冲突的一条 —— 花两格测它
  是全表性价比最高的 A/B。

噪声幅度用通道里已经定义好的值（与智元、`build_1` 同区间），本轮不新增通道也不改数：
`joint_pos ±0.01 rad` / `joint_vel ±0.5 rad·s⁻¹` / `base_ang_vel ±0.2 rad·s⁻¹`。
**任务通道不加噪**（§22 闸 1）：给 desired-contact / incoming-ball / 时间通道加噪会改支撑集，
等于换题而不是换传感器；finalizer 逐项复核，多一路带噪当场拒。

实现上新开一档 `DR-L0N`（"L0 + Noise"），它的 payload **由 DR-L0 的 payload 派生**，只允许差
`identity` / `policy_observation_corruption` / `proprioceptive_observation_noise` 三个键，
module 导入期断言把差异面钉死。它不是 `L2`：plant 与 L0 逐字节相同，跟 DR-L1 不在同一维度上，
排进 `L0<L1<L2` 会误导。DR-L0 的身份与 digest `fd22321e…` 一个字节没动。

| cell | ABI / task semantics | 本体感观测噪声 | 唯一要回答的问题 |
| --- | --- | --- | --- |
| `A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off` | `A211`，desired-contact p/v/face | **关**（DR-L0，现状） | 归因基线：干净传感器下学不学得会 |
| `A1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on` | `A211`，desired-contact p/v/face | **开**（DR-L0N，三路本体感通道） | §22 的"D1 开满"对不对：噪声是帮手还是病灶 |
| `C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off` | `C211`，incoming ball p/v/spin | 同 `A0` | 无 contact oracle 时的直接球状态方案 |
| `C1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on` | `C211`，incoming ball p/v/spin | 同 `A1` | 同 `A1`，在 outcome-only 奖励下 |

判读：`A1/C1` 出现接触而 `A0/C0` 没有，则 DR-L0 关噪声的裁定是错的，§22 的"day-1 开满"成立；
`A0/C0` 有接触而 `A1/C1` 没有，则噪声确实在这个阶段压制学习，DR-L0 的保守做法有据；
两档都没有接触，则这根轴被排除，下一嫌疑回到 reset 起点分布（上一轮已落地的 `start_pose_ramp`
挂在 DR-L1 上，正好是下一轮的候选）。GPU 布局不变：A 对同卡 `gpu0`、C 对同卡 `gpu1`、
`gpu2` 留给 MuJoCo。四格的运行时收据是 schema 3（arm/recipe 合同带 `policy_observation_corruption`
/ `proprioceptive_observation_noise_channels` / `dr_level_identity`），只认 schema 1 +
`sigma=0.02` 的 n1_vendor probe-gate 会拒收它们——这是刻意的 fail-closed，那条冻结 gate 本来
就不适用于这条新路线。

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
不以“必须零次”循环要求未开训 policy 已经学会平衡。
**这条口径 2026-08-06 才被真正执行到 `oracle32` 上**：在此之前 `C211 oracle32` 的验收器把
fall/too-low/robot-hit-table 也当成必须零次，`scale4096` 那边却已经在按本段口径分项守恒 ——
同一个发射器里两套口径。重定范围、守恒普查与收据自陈见 §5.6.8。
`oracle32` 的阶段轴是它自己的两值口径（`post_strike` / `pre_strike_or_same_step_unknown`），
因为 WAIT-only 复位根本不进那份证据；它们改成单独的 `wait_only_reset_excluded` 分母来记。
全程还需 PID/UUID receipt 和
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

### 9.2.0 mjlab/mujoco-warp 实测（2026-08-05，pod1 GPU2）

按 Franco 2026-08-05 裁定，MuJoCo 走 mjlab、CPU 顺序实现不再作为通往 `4096` 的路径。本节是**实测**，
不是估算。环境：driver `590.48.01`，`3x RTX 5090`（sm_120 Blackwell），独立 venv
`/workspace/mjlab_venv`（py3.12，未污染 `hope_isaac_venv`）。装的是
`mjlab 1.5.3` + `mujoco 3.10.0` + `mujoco-warp 3.10.0.3` + `warp-lang 1.16.0` + `torch 2.13.0+cu130`。
**sm_120 不是阻塞**：torch 的 `arch_list` 已含 `sm_120`。

| 项 | 实测 |
| --- | ---: |
| 本仓 `a3_pingpong.xml`，`nworld=4096` | **`3,954,523` steps/s**（realtime `3,955x`） |
| mjlab 完整 PPO 端到端（`4096` env，含推理与学习） | **`196,329` env-step/s**，`0.50 s`/iteration |
| 显存（`4096` humanoid） | `758 MiB` / `32,607 MiB` |
| 对照：Isaac A 轨迹（§8.1，`512 env x 24`） | `3,931` env-step/s |

即 **mjlab 端到端吞吐约为当前 Isaac 轨迹的 `50` 倍**。另外确认：阻挡 `4096` 的
`MAX_EXECUTE_ENVS=64` 纯粹是旧纯 Python 顺序循环的产物，与授权门、与 MuJoCo 本身都无关。

**确定性：实测不成立，且比 §14 预估更糟。** 同一进程、同一初始态、同一串固定 `ctrl`、全程零 RNG、
背靠背两次 rollout：

| 模型 | 逐位相同 | 发散世界 | `max abs(dqpos)` |
| --- | --- | ---: | ---: |
| `humanoid` | 否 | `1024/1024` | `1.6e-05` |
| 本仓 `a3_pingpong` | 否 | `959/1024` | `1.4e-08` |
| **`pendula`（无接触）** | 否 | `1007/1024` | `1.4e-05` |

跨进程 `sha256(qpos|qvel)` 亦不同。**`pendula` 那一行是关键**：完全没有接触也发散，说明它不在
接触/约束求解路径上，而是 smooth-dynamics kernel 中浮点累加顺序不结合导致的**结构性**不确定；
调模型、关接触、降 solver 迭代都绕不过，且 mujoco-warp **无 CPU 回退**可供对拍。

**因此 exact-resume 与精确课程续跑在 mujoco-warp 下不成立**：seed 给出的是分布而非轨迹。
本文 §9.2 原本就把 MuJoCo 验收分成 Tier-1（question/curriculum/receipt/ABI/action identity 要求 exact）
与 Tier-2（Warp/GPU 物理轨迹只要求统计等价）——现在这条分层由实测坐实，并须补上：
Tier-2 的复现口径改为 **N-seed 统计带**，checkpoint 续跑必须**容忍轨迹漂移**，
不得再要求逐 tick bitwise parity。§9.1 的独立性门（`fresh seeds >= 3`）与该口径同源，可复用。

判断：以 `50x` 吞吐换逐位复现，在当前阶段**值得**——我们尚未观测到任何一次接触，需要的是迭代速度；
而 exact-resume 本就未闭合（§12 的 `RESET-TERMINATION-RESUME` 仍为 `IN_PROGRESS`，只允许声称
reset-boundary resume）。真正要改的是**验收口径**，不是放弃该路线。

### 9.2.1 plant 必须继承智元 MJCF，不是 mjlab 默认（2026-08-05 实测）

按 Franco 裁定「mjlab 只是框架，设置要继承智元的 MuJoCo」，本节把三方逐字段 compile 后对齐。
方法是**真 compile 读 `MjModel` 字段**，不读源 XML——MJCF 有 `<default>` 继承与 class 覆盖。
另用一份无 `<option>` 的 URDF compile 出 MuJoCo `3.10` 出厂默认作基准线，用于区分
「智元刻意选的」与「MuJoCo 默认」。

**智元 `<option>` 只显式写了四项**：`timestep=0.001`、`gravity`、`noslip_iterations=3`、
`noslip_tolerance=1e-6`。`solver=Newton` / `iterations=100` / `ls_iterations=50` **全是 MuJoCo 默认**，
不是 A3 调参。这个区分决定了哪些不能动、哪些可以为 GPU 调。

| 字段 | 智元 MJCF（权威） | mjlab 默认 | 我们 Isaac / MJN |
| --- | --- | --- | --- |
| `opt.timestep` | **`0.001`**（显式） | `0.005` | `0.005`（慢 `5x`） |
| `opt.integrator` | `EULER`（默认） | **`IMPLICITFAST`** | MJN `EULER` |
| `opt.noslip_iterations` | **`3`**（显式；MuJoCo 默认 `0`） | `0` | MJN `3` |
| `opt.ccd_iterations` | `35`（默认） | **`50`**（mjlab 显式） | MJN `35` |
| geom `solref` | **`(0.005, 1)`**——收硬到默认的 `1/4` | `(0.02, 1)` | MJN `(0.005,1)` |
| geom `friction` | **`(1.5, 0.005, 0.0001)` 含地面** | 躯干 `1.0` / 足 `0.6` | **Isaac 地面 `1.0`（低 `33%`）** |
| geom `condim` | `3`（全部） | 躯干 `1` / 足 `3` | MJN `3` |
| **`dof_damping`** | 腰 `1.0/0.5/0.8`、头 `1.0`、肩 `1.5`、膝 `2.0`、踝 `2.0` | `0.0` | **Isaac 无此项；MJN 显式清零** |
| **`dof_frictionloss`** | `1.1971 / 0.69223 / 1.7 / ...` | — | **MJN 显式清零** |
| `ctrlrange`（力矩上限） | 腰 `220/46/118`、膝 `320` 等 | — | **Isaac 逐位完全相同** |
| `dof_armature` | 全精度值 | G1 自己的 | Isaac 圆整表，`15/18` 组相对差 `~1e-4` |

**`dof_damping` 与 `dof_frictionloss` 是真实物理项，智元设了值而我们两个引擎都清零**——
这不是随机性缺失，是 plant 不同。

**执行器结构三方都不同。** 智元是 **31 个纯力矩 `motor`**（`biastype=NONE`、`gainprm=[1,0,0]`），
PD 律在 aimrt 的 C++ 插件里逐消息计算：`ctrl = effort_ff + kp*(q_des - q) + kd*(qd_des - qd)`，
`kp/kd` 在仿真侧**不固定**（随消息传入，真值在 deploy 端）。我们的 MJN 只实现
`tau = clip(kp*(qdes - q) - kd*qd, ±effort)`，**缺 `effort_ff` 前馈与 `qd_des` 两项**；
mjlab 则用 MuJoCo 内置 position 执行器。

**落地方案（两段式，缺一不可）。** 实测 `MjSpec.attach()` **会丢掉 entity 的 `<option>`**
（`timestep 0.001 -> 0.002`、`noslip 3 -> 0`），且 `MujocoCfg.apply()` 之后还会无条件再写 12 个字段。
故：MJCF 负责 body/geom/joint/actuator/exclude（attach 后逐字段验证原样保留），
`SimulationCfg` **逐字段显式**写智元 opt 值；`decimation=20`（`1000 Hz` 物理 / `50 Hz` 策略），
不是 mjlab 的 `4`；mjlab 的便利改写器（`CollisionCfg`、`BuiltinPositionActuator`、terrain、
`joint_pos` 默认 keyframe）**一律不用**。

**具名偏离（无法继承）**：`noslip_iterations=3` 是 mujoco-warp 的硬缺口，带不过去，登记为偏离。

**球/台/网没有智元权威可继承**（`a3_pingpong.xml` 是机器人模型，无球无台无网）。我们的权威是两份
实测拟合 `ball_physics_venue.yaml` 与 `ball_physics_optitrack_20260730.yaml`，但它们是**解析接触/飞行
模型的参数**，不是 MuJoCo 的 `solref/solimp/friction`。**当前原生接触参数是错的**：台面恢复系数
实现值 `e=0.131` 而实测为 `0.9215`（venue，`58` 次门控回弹；OptiTrack 独立机位 `0.9102`
CI95 `[0.8825, 0.9311]` 包含之）。现役 C211 走解析路径（`physical_ball=false`）故未暴露，
一旦上原生接触必踩。原生接触路线上除几何与 `condim=3` 外**尚无一个物理参数被裁定**。

**有效性包络（写进扩域闸门）**：实测覆盖球速 `1--7 m/s`、旋转 `0--15 rev/s`、台面 `v_n 1.0--4.5 m/s`、
拍面 `u_n 1.4--7.2 m/s`；`SR>1.6` 完全空白。而 `build_1` 给 `400 Hz + CCD` 的理由是
`15--25 m/s` 回击速度——**该速度段我们一个参数都没测过**，扩域不得越过该边界。

### 9.2.2 mjlab lane 在 pod1 GPU2 上发车 + 接触容量看门狗（2026-08-06 实测，已按同日复核就地更正）

> **2026-08-06 复核后就地更正。** 本节初稿有两处说错，已在下文改掉，不另起段落自相矛盾：
> (1) mujoco-warp **不是**"悄悄丢行"——它会 printf 一行并置 `d.overflow` 位；真正的缺陷是我们的
> 训练循环两样都不读。(2) GPU 归属的证据是"掩码生效"，不是"采样了 268 行"。
> 看门狗本身的覆盖面缺口另见 §9.2.4；**那道门已在同日重做（§9.2.6），所以本节描述的是
> 一个已经被换掉的实现**——引用时请标日期，别当现役行为。

**人话**：MuJoCo GPU 这条线现在能自己跑训练了，不再只是"能跑物理"。`4096` 个环境一起跑，`5` 次
PPO 更新 `10.9` 秒跑完，全程没有任何一个环境算出 NaN。这次补上的缺口是一个**接触容量看门狗**：
如果某一步需要的约束行数超过预分配的 `njmax`，mujoco-warp 会把多出来的行丢掉、训练照常往下跑、
曲线照常上升，但那些世界的物理已经是错的。现在这种情况会当场停机并写明差多少。

**关于"静默"的准确说法**（初稿写错了，这里更正）。mujoco-warp `3.10` 在丢行时**会说话**：
`_src/forward.py:249` 直接 `wp.printf("nefc overflow - please increase njmax to %u")`，并在
`d.overflow` 这个**逐世界粘性位掩码**上置 `OverflowType.NEFC`；`opt.warn_overflow` 在
`_src/io.py:436` 是硬写 `True`，没有开关。
所以缺陷的正确表述是：**引擎在喊，只是没人读**——那行字混在 `5292` 行训练日志里，既不进摘要也不影响
退出码。这正好是 MEMORY 里"只出计数器＝无人读"那条老毛病，不是引擎的锅。它决定了正确修法是
**读 `d.overflow`**（见 §9.2.4 T1、已在 §9.2.6 落地），而不是自己重算 max。

> **更正六（2026-08-06 第二轮，逐行读源码后就地改）：上面这段原本还写了"`5` 种会 printf、
> `HFIELD`/`NVMAX`/`CONTACT_MATCH`/`EPA_HORIZON` 这 `4` 种只置位不打印"，这句话是错的，撤回。**
> 在 mujoco-warp `3.10.0.3` 上**九种全都会打印**：`HFIELD` 在 `collision_convex.py:427`
> （`"height field collision overflow, number of collisions >= %u"`）、`NVMAX` 在
> `island.py:995`（`"nvmax overflow: world %d needs %d active DOFs..."`）、`CONTACT_MATCH` 在
> `sensor.py:2438`（`"contact match overflow: please increase..."`）、`EPA_HORIZON` 在
> `collision_gjk.py:1392/1411`。原判据把"`atomic_or` 那一行"当成了整段，漏看了紧挨着上面的
> `if warn_overflow: wp.printf(...)`。
> **但真正的坑比"四种静默"更阴**，而且新发现的这两条才是要记住的：
> (1) **`EPA_HORIZON` 打的字里根本没有 "overflow" 这个词**——原文是
> `"Warning: EPA horizon = %d isn't large enough."`。所有历史上"`grep -ci overflow` = `0`"的
> 结论**都不覆盖 `EPA_HORIZON`**（含 §9.2.4 C1 对 08-05 双 seed 的事后判定：那条结论对会打
> "overflow" 字样的八种成立，对 `EPA_HORIZON` 不成立，只能靠 `d.overflow` 补测）。
> (2) **`BROADPHASE`/`NARROWPHASE` 只由 `worldid == 0` 那个世界打印**（`forward.py:263/270`），
> 所以**行数不等于受影响世界数**，拿 `1134` 行去推"多少世界坏了"是错的。
> 这两条正好说明为什么 stdout 只能当**旁证通道**，`d.overflow` 才能当**门**。

**改了什么**（`hope_training/whole_body_tracking/mjlab_lane/a3_train_ppo.py`）。默认开启，逐 physics
substep 在 GPU 上累积 `nefc`（每世界约束行）与 `nacon`（全世界接触）的滚动最大值，每次迭代随其它
统计量一次性读回（**不额外增加 GPU 同步**；零拷贝与流序已实测：`t.data_ptr() == a.ptr`，warp 默认流
是 blocking 流，与 torch 的 legacy default stream 有隐式同步）；任一项触到分配上限即
`raise RuntimeError` `CAPACITY_OVERFLOW`，训练进程非零退出。每次迭代的峰值写进 `.jsonl`，run 级
峰值与余量写进 `.json` 的 `capacity` 段。

**这个门守到什么程度（复核后的准确边界，不要读过头）**：

- **只守 `nefc`/`njmax` 一条轴。** `nacon` 那一路在整轮 `--nconmax` 扫描里**一次都没真正开过火**；
  broadphase（`ncollision`）根本不在监视范围内，而那恰恰是 `nconmax` 真正管住的东西。详见 §9.2.4。
- **判决延迟一整个 PPO 迭代**（`24` env-step x `20` decimation = `480` 个 physics substep）。
  累积不丢峰值，但"触顶即退出"要改成"触顶后最多 `480` 个 substep 才退出，前提是进程还活着"。
- **`verdict` 的三值只在"探针被显式关掉"那条路上守住**（实测 `--no-capacity-probe` 确为
  `NOT_MEASURED`）。初稿那句"没测过的 run 不允许声称容量门成立"**是假的**：`--iterations 0` 实测
  拿到 `verdict: PASS_NO_OVERFLOW` + `njmax_headroom_x: 572.0`，零样本签发满余量 PASS。守卫加错了变量。

**变异测试**（按"验收用变异测试"准绳）：同一份 smoke 把 `--njmax` 故意压到 `70`，进程抛
`CAPACITY_OVERFLOW` 退出码 `1`。两次独立复现分别落在 iteration `1`（`nefc peak 71`）与
iteration `2`（`nefc peak 77`）——mujoco-warp 非确定，机制一致但**具体迭代号和峰值不可复现**，
引用时不要写死。同一跑里 `38` 行 warp printf 全部落在抛异常的那一迭代，看门狗没有漏迭代。
**这一组变异只覆盖了 `nefc`**，不构成对整个容量门的验收（§9.2.4 T4 列出缺的三条）。

**发车实测**（`SMOKE5CAP`，pod1，`CUDA_VISIBLE_DEVICES=2`，`4096 env x 5 update`，规模对齐 Isaac 侧）：

| 项 | 实测 |
| --- | ---: |
| PID / GPU | `2862997` on `GPU-473a79f3…`（`nvidia-smi` index **`2`**） |
| 显存峰值 | `11,290 MiB` / `32,607 MiB` |
| update 计数 | `0,1,2,3,4` 全部落盘，逐条有 loss/reward/termination |
| 吞吐 | `45,706` env-step/s（`2.13 s`/iteration，collect 占 `95.3%`） |
| `nonfinite_state` 终止 | **`0`**（`5/5` 迭代全为零） |
| GPU0 / GPU1 | 全程 `2 MiB, 0 %`，无任何 compute proc |

**GPU 归属：结论成立，但初稿把证据强度说过头了，这里更正。** 初稿写的"`268` 行采样"是
`GPU2_SAMPLER.log` 的**文件总行数**，绝大多数是 `set -x` 的 trace 行。自己重数过的真实数字：
**采样 `25` 次**（跨度 `52` 秒，间隔 `2--3` 秒）、**compute-proc 观测行 `18` 条**、出现过的 PID
**`3` 个**（`2862997` smoke / `2863381` census / `2863723` nanprobe），非 GPU2 的 uuid **`0` 行**。
把证据量夸大了约 `10--15` 倍，任何下游拿"`268` 行"论证"密集采样无死角"的说法都要撤回。

**真正封死这个洞的不是轮询密度，是掩码结构性生效**：`SMOKE5CAP.log` / `CENSUS_TRAINALLOC.log` /
`NANPROBE_4096.log` 三份日志的 Warp 启动横幅在一台三卡机器上**只枚举出 `1` 个 CUDA 设备**
（`"cuda:0" : "NVIDIA GeForce RTX 5090"`），说明 `CUDA_VISIBLE_DEVICES=2` 生效，进程在**整个
生命周期内**物理上无法在 GPU0/GPU1 上建 context——这比"轮询那一刻没建"强得多。`2` 秒轮询本身有
约 `75--90%` 的时间盲区，而三张卡的 `Accounting Mode` 都是 `Disabled`，事后没有任何 driver 侧历史
可以补证。三层证据（掩码 + `25` 次轮询 + `2` 份进程内 `nvidia_smi_start/end` 收据）一致：
Isaac 那两张卡没被碰过。

**下次会咬人的隐患（这次没咬到）**：`run_gpu2_smoke.sh` 只 `export CUDA_VISIBLE_DEVICES=2`，
**没锁 `CUDA_DEVICE_ORDER`**（默认 `FASTEST_FIRST`），而 CVD 的序号走的是 `CUDA_DEVICE_ORDER` 的
顺序，与 `nvidia-smi` 的 PCI 顺序**没有契约保证一致**。本机三张同型号 `5090`，实测两种 order 给出
同一张卡（`uuid 473a79f3`、`pci_bus_id 190`），所以这次退化成一致——这是**事后确认，不是事前保证**。
换混合型号机器或改插卡顺序，`=2` 完全可能落到 Isaac 的卡上，而 `SMOKE5CAP.json` 只记了
`cuda_visible_devices: "2"` 这个**意图字符串**、不记实际拿到的 uuid，事后无法自证。修法见 §9.2.4 T6。

~~**吞吐代价要如实记**：同配置不开看门狗的历史 run（`TRAIN_s0`，`300` iter）是 `50,221` env-step/s，
开了是 `45,706`，即探针吃掉约 **`9%`**。~~

> **更正七（2026-08-06 晚，见 §9.2.6 第三节）：这个 `9%` 撤回，它不是探针的成本。**
> 那是拿 `5` 迭代的 `SMOKE5CAP` 去比 `300` 迭代的 `TRAIN_s0`——两条跑长度不同、当天机器状态
> 也不同，**不可比**。改成配对实测（`4096 env x 12 iter`，同一张 GPU2 背靠背交替，
> 收据逐条记了跑前/跑后卡上有没有别人的进程）：**旧看门狗 ON `44,923` vs OFF `45,202`，
> 约 `0.6%`；新的 `d.overflow` 门 ON `45,041` vs OFF `44,833`，差值在噪声里（三对里两对 ON 反而略快）。**
> 每条曲线都是同一个形状——iteration `1` 冲到 `50.2k--50.6k`，之后稳在 `44.8k--45.3k`，
> 所以**长度不同的两条跑不能直接比均值**。准确说法：这道门（新旧都是）代价 ≤ `1%`，
> 新门用同样的代价把覆盖面从 `1` 类换到 `9` 类。

### 9.2.3 njmax/nconmax 加桌加球后的重新标定（2026-08-06 实测，已按同日复核就地更正）

> **2026-08-06 复核后就地更正。** 本节初稿的**操作性结论（现役 `572`/`128` 够用）仍然成立**，
> 但支撑它的三处推理与两个数字被证伪，已在下文改掉：(1) "桌子接住机器人所以行数降低"是
> **摩擦锥换算**造成的假因果；(2) "最坏情况是 `ctrl=0` 摊平"低估约 `2` 倍；(3) 接触余量 `9.45x`
> 算在了错的计数器上。原来的"6.02x/9.45x"表述不要再引用。

> **2026-08-06 第二次就地更正（T9 落地后）。** 本节下面这张表的每一个峰值都是 `3000`/`12000` 步
> 定长窗口的结果，**已被 §9.2.7 的收敛普查全面取代**。取代不是推翻方向，是把每个数往上抬：
> 同样的 `4096` 世界 court elliptic，`flail 117→135`、`bang 120→137`、**`randpose 188→265`**。
> 因此本节这句"余量至少 `3x`"**不成立**，实测最坏合法构型只有 `2.16x`（court pyramidal）。
> 引用容量数字请一律走 §9.2.7，本节表格只作为"定长窗口会低估多少"的历史对照保留。

§9.2 的老数 `njmax=508` / `nconmax=128` 是**光机器人**时期的，必须在桌子和球进场后重测。测了，
结论是**旧数在行数上够用**——但当初给出的理由是错的，"余量至少 `3x`"这个口径也已被 §9.2.7 证伪
（真值 `2.07--2.16x`），余量数字要按下面 + §9.2.7 改口径。

`508` 是 plant-only 的 `suggest_njmax` 结果；court（robot + 桌 + 网 + 球，`ngeom=82`、`npair=5`）
自动定出的是 **`572`**，也就是 trainer 的现役值，配 `nconmax=128`/世界 →
`naconmax = 128 x 4096 = 524,288`。

实测峰值（`nefc` 单位是行/世界，`ncollision` 是 broadphase 候选对、全世界）：

| 场景 | 世界数 x 步数 | 摩擦锥 | `nefc` 峰值 | `ncollision` 峰值 | 行余量 vs `572` |
| --- | ---: | :---: | ---: | ---: | ---: |
| `ctrl=0` 摊平（初稿当成"最坏情况"） | `4096 x 3000` | elliptic | `95` | 未记 | `6.02x` |
| `ctrl=0` 摊平，**跑够 4 倍长** | `2048 x 12000` | elliptic | **`106`** | `82,906` | `5.40x` |
| **flail**：力矩在模型自己声明的 `ctrlrange` 内随机 | `4096 x 3000` | elliptic | **`117`** | `81,257` | `4.89x` |
| **bang**：满力矩正反跳变 | `4096 x 3000` | elliptic | **`120`** | `71,742` | `4.77x` |
| **randpose**：`jnt_range` 内随机构型 + 随机根姿态释放 | `4096 x 3000` | elliptic | **`188`** | `61,843` | **`3.04x`** |
| 实际 PPO 训练（`5` update） | `4096` | elliptic | `83` | 未记 | `6.89x` |
| 实际 PPO 训练（`300` iter x 2 seed，门全开复跑） | `4096` | elliptic | `86` / `91` | 未记 | `6.65x` / `6.29x` |
| 对照：plant-only 无桌球 | `4096 x 2000` | **pyramidal** | `115` | — | — |
| 对照：plant-only 无桌球，另一份普查 | `4096 x 3000` | **pyramidal** | `136` | `242,714` | — |
| 对照：court 有桌球 | `4096 x 3000` | **pyramidal** | `115` | — | — |

**更正一：那个"桌子接住机器人"的因果解释是假的。** 初稿拿 plant-only 的 `115` 和 court 的 `95` 相减，
但这两个数**摩擦锥不同**：plant 默认是 pyramidal（`a3_plant_env.py:68`），court 跑的是 elliptic。
MuJoCo 每个接触占的行数 pyramidal 是 `2*(condim-1)=4`、elliptic 是 `condim=3`（`suggest_njmax`
自己就这么写的），所以同样的物理接触数，换个锥就差 `4/3`。同锥对比才有意义：
**plant-pyramidal `136` → court-pyramidal `115`**，桌子确实让机器人少摊了一点，但幅度是 `15%`，
不是初稿暗示的那样由桌子造成 `115→95`。**这条要记住的实际后果**：哪天为了摩擦保真度把锥切回
pyramidal，行价从 `3` 涨到 `4`，`randpose` 的 `188` 会变成约 `(188-31)*4/3+31 ≈ 240`，余量掉到 `2.4x`。

> **更正一之续（T9 落地后，§9.2.7）。** 上面那句"桌子让机器人少摊 `15%`"和那个 `≈240` 的换锥
> 外推，现在都换成了**同锥直测**，两条都要改口径：
> (1) **"桌子降低行数"是分场景的，不是一个常数。** 同为 pyramidal、同为 `4096` 世界、都跑到收敛，
> plant→court 的 `nefc` 变化是：`ctrl=0` `159→131`（**`-18%`**）、`slam` `244→211`（`-14%`）、
> `randpose` `276→265`（`-4%`）、`flail` `170→170`（`0%`）、`bang` `167→174`（**`+4%`**）。
> 桌子只在"机器人自己摊下去"这类场景里帮忙；一旦有力矩在驱动，帮助归零甚至反号。
> (2) **换锥外推低估了。** 实测 court-pyramidal `randpose` 是 **`265`**，不是外推的 `240`，
> 余量 `2.16x` 而不是 `2.4x`。外推公式把"哪些行是接触行"猜错了——不要再用它，直接测。

**更正二："最坏情况是瘫倒摊平"被证伪，差约 `2` 倍。** `flail` 和 `bang` 是**完全合法的输入**
（初始就是 ready pose，力矩不越 `ctrlrange`），它们单独就把 `95` 顶穿 `23--26%`。`randpose` 的
`52` 接触/世界已经吃掉 `a3_court_env.py:344` 那个 `ncon_per_world=56` 假设的 `93%`。真实训练里
早期随机策略 + reset 随机化产生的构型分布，比 `flail/bang` 更靠近 `randpose` 那一端。
**按 `ctrl=0` 摊平定容量不是保守口径，是乐观口径。**

**更正三：报出来的每个 peak 都是下界，不是上界。** `3000` 步远没收敛。`ctrl=0` 跑到 `12000` 步：
running max 依次 `92`(3k) → `95`(5k) → `101`(8k) → `106`(12k)，**最后一次刷新纪录在第 `11,640` 步
（全程 `97%` 处）**，`nacon`/`ncollision` 峰值分别落在第 `11,999`/`11,998` 步——三条曲线在窗口末尾
全都还在涨。收据自己也这么说：plant 普查 `peak_at_step = 2534/3000`（`84%` 处）。
所以这些数只能支撑"这个步数内没超"，**不能支撑任何"余量 Nx"的断言**。

> **更正三之续（T9 落地后，§9.2.7）：这条不但成立，而且比想的更狠。** 把 court `4096` 世界的
> `ctrl=0` 拉到 **`30,000` 步**（原来只在 `2048` 世界跑过 `12,000`），elliptic 的最后一次刷新
> 纪录落在第 **`25,590`** 步（`85%` 处），`ncollision` 峰值落在第 **`29,985`** 步——
> **倒数第 15 步**。也就是说 `ctrl=0` 摊平这一条**在 `30,000` 步内根本不收敛**，
> §9.2.7 的收敛判据直接把它判成 `NOT_CONVERGED`，它的余量数字带 `_lower_bound` 后缀落盘。
> 反过来，`flail`/`bang`/`randpose`/`slam` 四条**都收敛了**（`7,000--28,000` 步不等），
> 所以"跑不收敛"不是普查方法的通病，是 `ctrl=0` 这个场景本身的性质：瘫倒之后世界还在缓慢
> 重新排布接触，几万步都停不下来。**结论不变但要说反过来**：`ctrl=0` 既不是最坏场景，也是
> 唯一一个测不完的场景——它两头都不占，不该再当容量基准。

**更正四：`9.45x` 接触余量算在了错的计数器上。** `naconmax` 同时是**三个**数组的上限：窄相接触
`nacon`、宽相候选对 `ncollision`、以及 broadphase 的 `collision_pair` 三件套。必须塞进去的是
`ncollision`，它恒 `≥` 接触数（实测比值 `2.2--2.3` 倍）。而收据里的
`naconmax_headroom_x = naconmax / nacon_peak` 只除了 `nacon`。按正确分母重算：
`4096` 世界 court 三个对抗场景是 **`6.45x` / `7.31x` / `8.48x`**；`ctrl=0` 跑 `12000` 步的
`2048` 世界结果线性折算到 `4096` 约 `166k` → **`~3.2x`**；plant-only pyramidal 摊平那份是
`242,714` → **`2.16x`**（不同场景/不同锥，仅作方向参考）。**真实区间 `~3--8x`，随场景和步数变，
且未收敛——不是 `9.45x`。** `a3_plant_env.py:108` 的注释（"backs the broadphase array"）其实早就
知道这件事，但这个数字既没进收据也没进训练门（§9.2.4 T2）。

> **更正四之续（T9 落地后，§9.2.7）：分母对了，区间要收窄，而且最窄的那一格在 plant。**
> 现在是 `4096` 世界直测、不再线性外推：**court** 的 `ncollision` 峰值区间是
> `71,311--172,521`（elliptic）/ `71,711--168,991`（pyramidal），对 `naconmax=524,288` 是
> **`3.04x--7.35x`**，其中 `3.04x` 那一格（`ctrl=0`）**未收敛，是下界**。
> 上面那句"`~3--8x`"因此基本站得住，只是上限收到 `7.35x`。
> **但 plant-only 那一格更窄**：pyramidal `ctrl=0` 跑 `30,000` 步是 `268,846` 候选对
> （旧数 `242,714` 又是个下界），**只剩 `1.95x`，而且仍未收敛**。plant 不是训练场景，
> 但谁要在 `4096` 世界长跑光机器人诊断，broadphase 是第一个会先撑爆的东西。

**没被推翻的部分**：我没能把它跑溢出。造出的最大值是 `188` 行/世界和约 `81k` 候选对，
`world_substeps_at_or_over_njmax572 = 0`、`world_samples_at_or_over_nconmax128 = 0`。
所以**现役 `njmax=572` / `nconmax=128` 在测到的所有场景里都安全，`constraint_headroom_ok` 的
操作性结论仍然成立**——只是它现在是"碰巧对"，不是"被证明对"，因为支撑它的三个理由都得换。

> **这一段在 T9 之后仍然成立，但余量的量级要改（见 §9.2.7）。** 收敛普查把最大值从
> `188` 行推到 **`276`** 行/世界、候选对从 `81k` 推到 **`268,846`**，`4096` 世界
> `20` 组场景 x 锥的组合里 `world_steps_over_reference_njmax` 与
> `world_samples_over_reference_nconmax` **仍然全是 `0`**，引擎的 `d.overflow` 九位掩码
> **也全是 `0`**。所以"现役配置安全"这句话现在是**被证明的**，不再是碰巧；
> 但"余量至少 `3x`"要改成 **`2.07x`（行）/ `2.06x`（接触/世界）/ `1.95x`（plant 宽相，下界）**。

**NaN / 发散**三条独立证据全绿：court 训练 `5` 迭代 `nonfinite_state=0`；court `ctrl=0` 摊平
`3500` 步 `worlds_with_nan=0`、`worlds_with_inf=0`、`qvel_absmax=7.77`；`nan_probe.py` 在 plant
`4096 x 2000` 步 `first_nonfinite_step=None`。

顺带**复核了球的恢复系数**（§9.2.1 的 `0.92150`）：本次 `4096` 世界重测 `e_mean=0.9214117`，
实测权威是 `0.9215`，`4096/4096` 世界全部落在接受带 `[0.88, 0.93]` 内。

> **更正五：上面这句"`4096/4096` 在带内"是空判，目前不构成任何证据。** 三个理由：
> (1) **带宽荒谬**。`E_ACCEPT = (0.88, 0.93)` 宽 `0.05`，而实测 `e_std = 3.47e-6`、
> 全幅 `e_max - e_min = 1.48e-4`——**带宽约等于实测全幅的 `340` 倍、`14,400σ`**。
> 这个带过不了变异测试：接触刚度 `k` 改 `10` 倍（`1000` vs `10000`）两个都"通过"。
> (2) **带没有出处**。`calibrate_restitution.py:95` 的注释只写 "acceptance band handed down for
> this task"，谁定的、依据哪次测量，都没有。而 `e_mean` 与权威值差 `8.8e-5`，按实测 σ 算是 `25σ`
> 的显著偏低，被宽带整个盖住。
> (3) **`4096` 不是 `4096` 个样本**。这是同一个确定性落球重复 `4096` 次，差异只来自 mujoco-warp
> 的非确定性，不是独立采样。
> **还有一个收据层面的误导**：`e_vs_v_n_slope_per_m_s: 0.0` **不是测出来的**。`CENSUS_TRAINALLOC`
> 的 `drop_height_m` 是 `[0.33, 0.33]`——单一落高、单一 `v_n = 2.5445 m/s`，`a3_court_env.py` 在
> 退化情形下直接写死 `0.0`。而权威 `E_TABLE_MEASURED` 覆盖 `v_n 1.0--4.5 m/s`。
> 把"没测"写成"斜率为 0"，会被读成"实测无速度依赖"。修法见 §9.2.4 T7。
> **结论：任何以"球的弹性已核实"为前提的下游推断（击球质量奖励、sim2real 弹跳预算）目前都没有支撑。**
>
> **更正五之续（2026-08-06 晚，T10 落地后）。** 上面这段的**主张全部核实成立并已修**，
> 落地明细与实测数字见 **§9.2.5**；这里只就地改**一处读数**：
> "`e_mean` 差 `8.8e-5`，按实测 σ 算是 `25σ` 的**显著**偏低"——偏差数字对（今天单一落高
> `512` 世界重现 `-8.82e-5`），但**"显著"这个读法撤回**。那个 `25σ` 的分母是仿真 σ `3.47e-6`；
> 同一场景我今天量到 `1.27e-7`，差 `27` 倍。**仿真 σ 是调度非确定性，不是测量不确定度**，
> 随跑随变，拿它数 σ 得不出任何物理结论（正是本节 B5 提醒的那类错误）。按**场地** σ `0.005`
> 算，`8.8e-5` 只有 `0.018σ`，且落在新标定容差 `1.667e-3` 之内 `19` 倍——这是**标定层面的小偏置，
> 不是物理分歧**。上面"宽带把它整个盖住、`e_in_accept_band_all_worlds: true` 不构成证据"这个
> 判定不变。

收据在 pod1 `/workspace/mjlab_lane/`：`SMOKE5CAP.json`（发车）、`CENSUS_TRAINALLOC.json`（容量
`ctrl=0` 口径）、`NANPROBE_4096.json`（NaN + plant-only 对照 `115`/`66,676`）、
`GPU2_SAMPLER.log`（GPU 归属）、`CAPMUT`（`nefc` 变异测试）、
`contact_census_4096.json`（plant-pyramidal `136`/`242,714`）、
`CENSUS2_pyramidal.json` 与 `CENSUS2_elliptic.json`（同场景换锥的 `115` vs `96` 对照）。
复核新增：`/workspace/advcheck/ADV_4096.json`、`ADV_ZERO_LONG.json`、`ADV_SCEN.json`（对抗场景与长跑），
`/workspace/mjlab_lane/AUDIT_*`（门的变异与盲区扫描，逐条见 §9.2.4）。

**这一节不代签什么**：这是 mjlab lane 自己的 court/ready/reach-touch 任务，**不是** canonical
ActionBall N1。它没有 measured teacher、没有完整 reward 层级、没有 §9.2 要求的 termination union、
没有 cross-engine parity，也没有 exact-resume（§9.2.0 已裁定 mujoco-warp 下逐位复现不成立）。
它证明的是：GPU-native `4096` 训练回路在这条 lane 上真的转起来了。
**"容量与 NaN 两个 fail-closed 门现在有代码在守"这句初稿的话要收回**——容量门只守住 `nefc` 一条轴，
NaN 那条链也被 `nan_to_num` 掐断了（都见 §9.2.4）。准确说法是：**数据干净，门不可信。**

### 9.2.4 四方独立证伪后的汇总裁定（2026-08-06）

**人话总结一句**：这条 lane 现在**跑出来的数据是干净的**（历史和复跑都查不到任何一次溢出，
余量至少 `3` 倍），但**看门的那道门本身不可信**——它只盯着一个计数器、零测量也会盖 PASS 章。
（**"余量至少 `3` 倍"这句话的适用范围，2026-08-06 T9 之后要限定**：它对**实际跑过的训练 run**
成立——那些 run 的 `nefc` 峰值是 `83--91`，余量 `6.29--6.89x`。它对**对抗普查造出来的最坏合法
场景不成立**：收敛后最坏是 `276` 行/世界，余量 `2.07x`。见 §9.2.7。）
所以现在能说的是"这些 run 没坏"，**不能说**"门会拦住下一次坏"。

四条独立证伪（看门狗 / GPU 隔离 / 容量普查 / 历史 run）的裁定如下。**冲突处已逐条查证，不和稀泥。**

#### A. 成立、必须改的真问题

| 编号 | 一句人话 | 硬证据 |
| --- | --- | --- |
| **P1** | **门对 broadphase 溢出完全失明，而那正是历史上真出过事的那条轴。** 更坏的是失明方向与被监视的信号**反相关**：宽相溢出时候选对在进窄相**之前**就被丢掉，于是 `nacon` 永远到不了上限——**溢出越深，被监视的数字看起来越健康**。 | `--nconmax 10` 实测：引擎打了 `1134` 行 `broadphase overflow - please increase nconmax to 11 or naconmax to 2561`，收据却写 `verdict: PASS_NO_OVERFLOW` + `naconmax_headroom_x=1.42`，**退出码 `0`**。`--nconmax 8` 同样（`1227` 行警告，`PASS`）。这正是 `a3_plant_env.py:92-101` 记录的历史事故（越界写 → CUDA illegal access），而 `CAPACITY_OVERFLOW` 的报错文案还在叫人 "Re-size with --njmax/--nconmax"。 |
| **P2** | **零测量也能签发 PASS。** §9.2.2 初稿那句"没测过的 run 不允许声称容量门成立"是假的。 | `--iterations 0` 实测收据：`nefc_peak=0`、`nacon_peak=0`、`njmax_headroom_x: 572.0`、`naconmax_headroom_x: 8192.0`、`verdict: PASS_NO_OVERFLOW`。根因是 `_capacity_summary()` 只检查"探针接上了没"（`env._cap_ok`），从不检查"是否真的记录过样本"；而 `cap_peak` 初值是 `0`，`0` 和"测了且真是 0"无法区分。 |
| **P3** | **`nacon` 那一路一次都没真正开过火。** | 扫 `--nconmax ∈ {10, 8, 6, 4, 2}`：`10`/`8` 静默 `PASS` 退出 `0`；`6`/`4`/`2` 直接 **CUDA illegal memory access** 崩掉（不是看门狗拦的）。**没有任何取值让 shipped 脚本打印出 `nacon` 路径的 `CAPACITY_OVERFLOW`。** 只读探针在 `nconmax=8` 观测到 `nacon` 峰值 `== naconmax`（原理上可达），但同配置的真实训练跑出 `1806 < 2048` 就放行——**同一个 `nconmax=8`，一跑拦一跑放**。 |
| **P4** | **判决延迟一整个 PPO 迭代（`480` 个 physics substep），而 broadphase 那条轴"溢出→越界写→CUDA fault"的时间窗比这短。** | `--nconmax 6` 那跑里 warp 已经打了 `363` 行 `narrowphase overflow`，进程在看门狗到点读数之前就被 CUDA 非法访问打死。门永远抢不到那一拍。 |
| **P5** | **eval 路径开了探针但完全没有门。** | `evaluate()` 传了 `capacity_probe=`，把含 `njmax_saturated` 的 `stats["capacity"]` 写进 JSON，但从不检查、从不调 `_capacity_summary`、永远 `return 0`。所有 `EVAL_*`/`EVALC_*`/`AUDIT_EVAL_*` 收据一律无门、无 `verdict`。 |
| **P6** | **`sim.forward()` 是采样盲区。** | 探针只在 `sim.step()` 的 decimation 循环里调（`:600`）。而 `step()` 末尾在 env reset/补发球时还会调 `self.sim.forward()`（`:635`，`4096` env 下几乎每个控制步都会走），`reset()` 也调（`:528`）。`mjwarp.forward` 会重建碰撞与约束、同样会溢出，其 `nefc` 被下一次 step 覆盖，永不采样。 |
| **P7** | **另外 `7` 种溢出类型无人看。**（主结论成立；括号里那句"其中 `4` 种连引擎都不打印"**已被 §9.2.6 证伪并就地更正**，见下） | 引擎的 `d.overflow` 是 `9` 位粘性掩码，训练循环一位都不读——这部分成立，已在 §9.2.6 修掉。~~会 printf 的只有 `NEFC`/`NJMAX_NNZ`/`BROADPHASE`/`NARROWPHASE`/`CCD`；`HFIELD`/`NVMAX`/`CONTACT_MATCH`/`EPA_HORIZON` 只置位、不打印~~ —— **撤回**：`3.10.0.3` 上九种全都打印（`collision_convex.py:427`、`island.py:995`、`sensor.py:2438`、`collision_gjk.py:1392/1411`，每处 `atomic_or` 上面紧贴一行 `if warn_overflow: wp.printf`，原判据只看了 `atomic_or` 那一行）。**换成两条更阴的真事**：(1) `EPA_HORIZON` 打的是 `"Warning: EPA horizon = %d isn't large enough."`，**整句没有 "overflow" 这个词**，所有 `grep -i overflow` 的历史结论都不覆盖它；(2) `BROADPHASE`/`NARROWPHASE` **只由 `worldid == 0` 打印**（`forward.py:263/270`），行数 ≠ 受影响世界数。另注意 `forward.py` 里 `NJMAX_NNZ` 用的是 `elif`，只在 `nefc` **没**溢出时才检查。**再补一条 P6 的加强版**：`mjwarp.forward()` 根本不跑 `_next_time`（那个 kernel 只在 `_advance` 里，`forward.py:276/324`，而 `_advance` 只被 `step()` 的积分器调用），所以 `sim.forward()` 里溢出时 `NEFC`/`NJMAX_NNZ`/`BROADPHASE`/`NARROWPHASE` **四位一位都不会被置**——这不只是"我们没采样"，是引擎压根没检查。 |
| **P8** | **`nan_to_num` 把 NaN 报警链掐断了，`nonfinite_state=0` 的证明力比看上去弱。** | `a3_train_ppo.py:572` 对 obs、`:671` 对 reward 都做了 `torch.nan_to_num`，rsl_rl 自带的 `check_nan(obs, rewards, dones)` 因此永远看不到。"溢出 → NaN → 崩"这条自然报警链不存在。（终止判据里的 `torch.isfinite(qpos/qvel)` 仍在，所以不是全无防线，但 obs/reward 这两路是哑的。） |
| **P9** | **门禁一旦真的开火，落卡收据就同时消失。** | `CAPMUT`/`AUDIT_nc6`/`nc4`/`nc2` 这些非零退出的跑**只有 `.jsonl` 没有 `.json`**，没存 stderr、没存退出码、没存实际拿到的 GPU uuid。最需要证据的那一跑反而没有证据——和 MEMORY 里"改软硬门要连证据一起改"是同一类问题：失败路径上没有 telemetry。 |
| **P10** | **`>=` vs `>` 差一行（良性，顺带记）。** | 引擎判据是 `nefc > njmax`，丢行判据是 `if efcid >= njmax_in`，所以 `nefc == njmax` 是**正好装下**；看门狗用 `peak >= njmax` 会在这一点误报。方向是 fail-closed 无害，但会在一个其实没坏的 run 上打出报错文案。 |

数字层面的更正（`115→95` 的假因果、"最坏情况 `95`"低估 `2` 倍、`3000` 步未收敛、`9.45x` 用错分母、
恢复系数带宽 `14,400σ`）已经就地写进 §9.2.3，不在这里重复。

#### B. 证伪方自己出的错（一并记下，免得下游照抄）

1. **`115` 的出处被认错了。** 有一方判定"`115` 出自 `CENSUS2_pyramidal.json`，那是 court 场景不是
   plant-only"，据此说初稿引错了收据。**这条不成立**：初稿那行写的是"plant-only，`2000` 步，
   `115` / `66,676`"，我按 `66676` 反查，源头是 `NANPROBE_4096.json`
   （plant、`4096` 世界、`2000` 步、`njmax=508`），末条 trace 正是 `nefc_max: 115, nacon_max: 66676`。
   **它确实是 plant-only。** `CENSUS2_pyramidal.json` 的 `115` 只是行数碰巧相同（那份是 `3000` 步、
   `nacon 58,850`、court）。**但该方的实质结论仍然成立**——plant 默认 pyramidal、court 跑 elliptic，
   `115` vs `95` 的锥混淆是真的，只是理由要换成"plant 那份普查本身就是 pyramidal"。
2. **broadphase 真实余量被说窄了。** 有一方写"真实余量 `~2--3x`"，另一方写"约 `4--8x`"。两个都不是
   在训练尺度上直接测的：`~2--3x` 来自 `2048`→`4096` 的线性外推和 plant-only pyramidal 那份
   （不同场景、不同锥、不同分配），`4--8x` 来自 `256` 世界的比值外推。**直接在 `4096` 世界 court
   场景测到的是 `6.45x` / `7.31x` / `8.48x`。** 正确写法是"`~3--8x`，随场景与步数变、且未收敛"，
   见 §9.2.3 更正四。
3. **"`9` 种 overflow 字符串一条都没有"这个说法不严谨。** 只有 `5` 种会打印，见 P7。这不影响该方的
   主结论（下面 C1），因为与 `njmax`/`nconmax` 有关的那几种都在会打印的那一组里。
4. **两处行号笔误**：`nan_to_num(obs)` 在 `:572` 不是 `:672`（`:671` 是 reward 那一处）；两处都真实
   存在，不影响 P8 的实质。
5. **变异测试的迭代号和峰值被当成固定值引用**（"iteration 1 / peak 71"）。两次独立复现是
   iteration `1`/`71` 和 iteration `2`/`77`。mujoco-warp 非确定，这类数字不能写死。

#### C. 被证伪方推翻、必须撤回的判定

1. **"08-05 那次 `4096 env x 300 iter` 双 seed 训练的物理从未被验证、不可判定" —— 撤回。**
   这条判定的地基是"引擎静默"，而地基是错的（见 §9.2.2 更正）。**可以事后判定，而且不需要重跑**：
   - 两份历史日志各 `5292` 行、`grep -ci overflow` = **`0`**（我自己重跑过这条 grep，两份都是 `0`；
     `2>&1` 两路都收了）。会打印的 `5` 类溢出——包括 broadphase——一条都没有。
   - 同一脚本、同一环境的**对照组证明这条通道当时是活的**：`--njmax 60` 立刻打出 `13,008` 行
     `nefc overflow`，`--nconmax 10` 打出 `1134` 行 `broadphase overflow`。
   - 发车前 `31` 分钟就有一份**同分配值**的 census（`RECEIPT_COURT_4096_elliptic.json`，同
     `572`/`524288`，余量 `8.06x`/`11.86x`）。所以"从不和实际需求比"只对训练循环内部成立，
     对整条 lane 不成立。
   - 今天补的两组主动实测一致：历史 ckpt 回放 `nefc 69`/`72`（`8.29x`/`7.94x`），
     门全开同 seed 全量重跑 `86`/`91`（`6.65x`/`6.29x`），都是 `PASS_NO_OVERFLOW`、`0` 条 overflow、
     `nonfinite` 终止 `0`。
   - 顺带答一个反向担心：**晚期策略比早期更省行数**（回放 `69--72` < 训练期 `83--91`），
     "学会以后接触变多可能撑爆"这个方向是反的。
   **所以那两条 run 不需要打"物理存疑"标签，也不需要重训。**
2. **但那两条学习曲线仍然不能按原来的方式引用 —— 理由与溢出无关，是口径和复现性。**
   （**这一条已于同日按 T11 修完，见 §9.2.8**：改名 + 二值接触率进训练曲线 + `--report`
   把口径写成会拒绝的代码。下面三小条的**判定全部成立、原文保留**，只在末尾各加一句"现在怎样"。）
   - `reach`/`touch` 是**带权奖励项**，不是概率：`w_reach=2.0`（上限 `2.0`）、`w_touch=4.0`
     （上限 `4.0`）。"`0.53→0.98`"和"`4e-5→0.21`"并排写必然被读成两个百分比。真实含义是：
     球拍平均离球从约 `1.0 m` 缩到约 `0.57 m`；`touch` 高斯核均值 `0.21/4 = 5.25%`
     （原文写 `5.4%`，是笔误，就地更正；不影响结论）。
     **`0.21` 完全不是接触率。**
     —— **现在**：收据里这两项叫 `reach_term_weighted` / `touch_term_weighted`，同时落
     `reward_terms_max_possible`（`2.0`/`4.0`）与 `reward_kernel_mean`（除掉权重之后的核均值），
     并自带一句"这不是概率、要接触率请看二值那项"。
   - **真正的二值接触率只在 eval 路径有**（`count_contacts` 只在 eval 开）：
     零策略对照 `0.12%`、s0 `49.2%`、s1 `97.8%`（今天复现 `49.1%`/`97.6%`）。策略确实学到了东西
     （`400--800` 倍于零策略对照），**但支撑它的是这组 eval 二值接触率，不是训练曲线上的 `touch` 奖励项**。
     —— **现在**：`count_contacts` 训练路径默认打开，
     `fraction_of_episodes_with_a_racket_touch` 逐迭代上曲线（配对实测吞吐代价见 §9.2.8 第四节）。
   - **单 seed 单点不可复现**：同配置同 seed 四次跑出 `touch` = `0.21` / `0.46` / `0.59` / `0.61`，
     近 `3` 倍散布，而 `0.21` 是四次里最差的一次。要报就报带，别报点。
     —— **现在**：`--analyze` 给一份文件直接退出 `2`（以前会给出零宽度的"带"），
     `--report` 少于两条 run 退出 `2` 并点名 `SINGLE_SEED_NOT_EVIDENCE`。
   - 顺带排除了"溢出造成穿透、反而更容易够到"这条假阳性路径：前提就不成立（无溢出）；
     零策略 `0.0012` vs 训练后 `0.49`/`0.98` 是 `400--800` 倍不是噪声；穿透会把距离推向 `0`、
     `touch` 冲向上限 `4.0`，而实测最小距离 `0.086 m`/`0.047 m`、`touch` 只有 `0.21--0.62`，
     没有穿透签名；掉接触会让机器人陷进地面而 `height` 项 `300` iter 最低 `0.329/0.5`，骨盆一直在位。
     （**这一处用 `touch` 是对的、保留**：这里问的是"它有没有顶到自己的上限 `4.0`"，
     即把它当**带上限的加权项**用；错的是把同一个数当成百分比。新收据的
     `reward_terms_max_possible` 正是为这种用法准备的。）

#### D. 现在到底可信到什么程度（一句话）

**MuJoCo GPU lane 目前"数据可信、门不可信"：已经跑完的每一条 run（含 08-05 双 seed）都有当场或
事后的证据表明没有发生任何一次约束/接触溢出，行余量至少 `3x`（**限定：指实际训练 run 的
`6.29--6.89x`；对抗普查的最坏合法场景收敛后只有 `2.07x`，见 §9.2.7**）；但守门的代码只覆盖 `9` 类溢出里的
`1` 类，能在零测量时签发 PASS，也拦不住历史上唯一真出过事的那条轴——所以它现在只够用来**记录**，
不够用来**放行**。**

> **进度补记（同日晚些时候）**：上面这句描述的是**复核当时**的代码状态，作为裁定它没有变。
> 门本身已按 T1--T8/T12 重做完并逐条变异验收，落在 **§9.2.6**：判据换成引擎的 `d.overflow`
> （`9` 类全覆盖）、判决延迟从 `480` substep 缩到 `20`、零测量判 `NO_SAMPLES` 不判 PASS、
> broadphase 那条轴实测能开火并点名。**但"门可信"不等于"可以放行"**：§9.2.4 E 里
> E4/E5（普查未收敛、策略驱动构型分布没测过）和 §9.2.7 的收敛后余量都没被这轮碰过，
> 放行还得看那几条。

#### E. 还剩哪些洞没被任何证据覆盖

> 下面 `1`--`9` 是复核当时的清单，逐条标了后来谁关掉了它。**§9.2.6 关掉的是 `1`/`2`/`7`/`8`
> 这四条（都属于"门与收据"），并顺带更正了 `3` 的措辞。容量数值那几条（`4`/`5`/`9`）
> 归 §9.2.7，恢复系数（`6`）归 §9.2.5。**

1. **`ncollision`（宽相候选对）在训练期从未被采样过**，只在事后的只读探针里量过；训练收据里至今没有
   这个字段。—— **§9.2.6 已关**：每 substep 采样，收据出 `ncollision_peak_all_worlds_running`。
2. **深度压 `nconmax` 时是 CUDA illegal access 先到、门后到**，中间没有任何一段由门接管；
   在缩短判决延迟之前，这段区间无法被守住。—— **§9.2.6 已关**：`--nconmax 4` 实测在 `reset`
   那一次 `forward()` 就被拦下，`0` 行 CUDA 报错。
3. ~~**`EPA_HORIZON` 溢出真静默**（不 printf、不进日志）~~ **两半各改一半（2026-08-06，T9，
   见 §9.2.7）**："真静默"不成立——它**会** printf（`collision_gjk.py:1392/1411`，
   `atomic_or` 上面紧贴着一行 `if warn_overflow: wp.printf`，原判据只看了 `atomic_or` 那一行），
   只是那句话是 `Warning: EPA horizon = %d isn't large enough.`、**整句没有 "overflow" 这个词**，
   所以 `grep -i overflow` 结构上看不见它；"历史 run 对这一类没有任何证据"**已补上**：用真字符串重扫全部
   历史日志命中 `0` 条。**但这一类本身现在有实证了**——普查在 `randpose`+pyramidal+`seed 13`
   上实际观测到 `4096` 世界里 `1` 个置位，分配远没用满，`njmax`/`nconmax` 修不了它。
4. ~~**容量普查从未跑到收敛**：`ctrl=0` 到 `12000` 步仍在刷新纪录，所有"余量 Nx"都是下界。~~
   **已修（2026-08-06，T9，见 §9.2.7）**：普查改成收敛判据，`4` 个非准静态场景全部跑到
   "最近 K 步无新纪录"为止（`7,000--28,000` 步）。**但 `ctrl=0` 这一条仍然没收敛**——
   court `4096` 拉到 `30,000` 步，最后一次刷新在第 `25,590` 步，`ncollision` 峰值在倒数第 `15` 步；
   现在它会被判成 `NOT_CONVERGED`、余量带 `_lower_bound` 后缀落盘、退出码非零，不再冒充峰值。
5. **策略驱动的构型分布从未被普查**。现有普查都是 `ctrl=0` / 随机力矩 / 随机构型；
   "早期随机策略 + reset 随机化 + 课程"下的真实分布没测过。
6. ~~**恢复系数带宽 `14,400σ`，且 `v_n` 依赖从未测过**（单一落高，斜率被写死 `0.0`）。
   `e_in_accept_band_all_worlds: true` 目前不构成证据。~~
   **已修（2026-08-06 晚，T10，见 §9.2.5）**：带子从 `(0.88, 0.93)` 收到 `(0.9065, 0.93)`
   并换上真出处；`v_n` 斜率在 court 实景 `4096` 世界上真扫出来了（`+1.60e-4 /(m/s)`，
   场地 CI `[-0.007, +0.018]` 内）；退化情形改输出 `null` / `NOT_MEASURED`。
   **但"带宽仍然荒谬"这半句照旧成立**：新带 `0.0235` 宽，对仿真 σ 仍是几千倍，
   所以**真正能开火的不是带子**，是新加的"标定完整性"两道门——`k` 改 `10` 倍当场判 `FAIL`
   （退出码 `4`），旧带对它是放行的。
7. ~~**落卡 uuid 从不自证**：收据只记 `cuda_visible_devices` 这个意图字符串，不记实际拿到的 uuid；
   `CUDA_DEVICE_ORDER` 未锁，本机是三张同型号卡才退化成一致。~~
   **已修（2026-08-06，T7(a)/(c)，见 §9.2.6）**：收据出 `device_uuid` / `pci_bus_id` /
   `torch_cuda_device_count`，并按 PID 与 `nvidia-smi` 的 `compute_procs` 对账
   （六条验收跑全部 `device_uuid_matches_nvidia_smi: true`、`device_count: 1`）；
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` 在 pod 脚本和 `a3_train_ppo.py` 进程内**双重**锁上。
8. ~~**失败路径无 telemetry**（P9）：门开火那一刻的证据是缺的。~~
   **已修（2026-08-06，T7(b)，见 §9.2.6）**：`train()`/`evaluate()` 全包在 `try/finally` 里，
   任何退出路径都落 `.json`（`status` / `exit_code` / 异常 / traceback / 落卡 uuid / `argv`），
   连"场景还没建成就开火"那一格也落——`--nconmax 4` 的收据实测有
   `status: gate_fired`、`overflow_flags: ["BROADPHASE","NARROWPHASE"]`。
9. ~~**`4096` 世界 court 场景的 `ctrl=0` 长跑（≥`12000` 步）没做过**~~
   **已做（2026-08-06，T9，见 §9.2.7）**：`4096` 世界 court `ctrl=0` 跑了 `30,000` 步，
   两个锥各一遍。`ncollision` 实测 `172,521`（elliptic）/ `168,991`（pyramidal），
   不再靠 `2048→4096` 的线性外推（那个外推给的是 `~166k`，方向对、仍是下界）。

#### F. 给实现方的待办（**本次复核方未改任何实现代码**；Isaac lane 可能正在动同一批文件，不要就地改）

全部指向 `hope_training/whole_body_tracking/mjlab_lane/a3_train_ppo.py`，行号以 commit `8afcae8a` 为准。

- **T1（最高优先）换判据：读引擎自己的 `d.overflow`，不要自制近似。**
  `mujoco_warp/_src/types.py:2350` 已经暴露 `d.overflow: array("nworld", int)`，逐世界粘性 OR 累积，
  一次覆盖全部 `9` 类。在现有 decimation 循环里再读一个同形状小数组、GPU 侧 OR 归约，成本与现有
  `nefc` 读回同量级。任何非零位即 fail，并把 flag 名字打进报错。
  **必须注意**：`mjwarp.reset_data` 会把被 reset 世界的 `d.overflow` 清零（`io.py:2483`），而
  `_reset_idx` 调 `sim.reset(ids)`（`:482`）——所以 OR 归约**必须留在 decimation 循环里**
  （`:596-600` 现在的位置就对），不能挪到 step 末尾的 reset 之后。
  现有 `nefc`/`nacon` 峰值保留，但降级为**报告用**（算余量），不再当门。顺带这能退掉那 `9%` 吞吐税。
- **T2 把 `ncollision` 纳入监控并改 headroom 分母。** `naconmax_headroom_x` 应当是
  `naconmax / max(nacon_peak, ncollision_peak)`；`.json` 里单列 `ncollision_peak_all_worlds`。
  改的位置：`:741-751`（per-iteration stats）与 `:1042-1071`（run 级 summary）。
- **T3 缩短判决延迟。** 现在每 `480` 个 substep 才判一次（门在 `log_hook` 里，`:881-896`）。
  至少把 overflow 的 OR 结果累进一个 device 标量，在**每个 env-step 边界**读一次，而不是每个
  PPO iteration 读一次。
- **T4 PASS 必须以证据为前提。** `_capacity_summary()`（`:1042-1071`）增加"是否记录过样本"的判定，
  不要只看 `env._cap_ok`。建议 `cap_peak` 初值用 `None`/`-1` 而非 `0`（`:863`），并在 `log_hook`
  里累计 `cap_peak["iters"] += 1`；`iters == 0` 时 verdict 写 `NOT_MEASURED`（或新增 `NO_SAMPLES`），
  **绝不写 PASS**。
- **T5 eval 路径要么同样设门，要么明写 `NOT_GATED`。** `evaluate()`（`:981-1030`）现在把
  `njmax_saturated` 写进 JSON 却不检查、恒 `return 0`。
- **T6 补 `sim.forward()` 的采样点**（`:528` 与 `:635` 两处）。
- **T7 收据要自陈落卡与失败。**
  (a) `:919-921`/`:931-932` 现在只记 `device` 和 `cuda_visible_devices`（都是"我打算用哪张卡"）；
  加上进程内真值 `torch.cuda.get_device_properties(0).uuid` / `.pci_bus_id` / `device_count`，
  验收标准是 receipt 里的 uuid 能和 `nvidia_smi_start.compute_procs` 对上且 `device_count == 1`。
  (b) receipt 落盘放进 `finally`，门开火时至少写
  `{status: "gate_fired", device_uuid: ..., exit_code: ...}`。
  (c) pod 上 `run_gpu2_smoke.sh` / `audit.sh` 只 `export CUDA_VISIBLE_DEVICES=2`，
  加 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`，或更硬地直接用 uuid 钉卡（实测本机可用）。
- **T8 变异测试补齐并入收据**（按"改软硬门要连证据一起改"准绳，至少三条，且必须自陈 telemetry）：
  `--nconmax` 落在盲区带（`nworld=256` 时是 `9~11`，`4096` 时需重新定位）——修复前退出 `0` + `PASS`，
  修复后必须非零退出并点名 `BROADPHASE`；`--nconmax` 深压（`≤4`）——目前是 CUDA 非法访问，修复后应
  在崩之前被门拦住；`--iterations 0`——修复后不得出现 `PASS`。
- ~~**T9 普查改成收敛判据，别用固定 `3000` 步。**~~ **DONE（2026-08-06，见 §9.2.7）。**
  原文：`contact_census.py` 已经记了 `peak_at_step`，直接拿它当门：
  `peak_at_step > 0.7 x steps` ⇒ 判 `NOT_CONVERGED` 而不是 `PASS`；收据输出 running-max
  时间序列而不是单个标量；跑到"最近 K 步无新纪录"为止。同时把非准静态场景加进普查
  （`ctrlrange` 内随机力矩、`jnt_range` 内随机构型释放），参考实现在 pod
  `/workspace/advcheck/adv_capacity.py`，一张卡不到 `20` 分钟。
  **落地时多做的三件**：摩擦锥进收据 + 跨锥比较硬拒绝（不然会再造一次"加桌降低行数"）、
  读 `d.overflow` 九位掩码（普查这一侧提前吃掉 T1 的覆盖面）、失败路径也落收据。
  **落地后的操作性结论**：现役 `572`/`128` 够用，余量 `2.07x`（行）/ `2.06x`（接触/世界），
  "至少 `3x`"那句撤回。
- ~~**T10 恢复系数验收带要么收紧要么别再当证据。**~~ **DONE（2026-08-06 晚，见 §9.2.5）。**
  原文：`calibrate_restitution.py:95` 的 `E_ACCEPT` 改成能失败的宽度（例如权威 ±`3--5` 倍**场地**
  测量 σ），把 "handed down" 换成真实出处；走 `--height-sweep` 让 `e_vs_v_n_slope` 真的在
  `1.0--4.5 m/s` 上测出来；`a3_court_env.py` 退化情形应输出 `null`/`"NOT_MEASURED"` 而
  **不是 `0.0`**。验收用变异测试：`k` 改 `10` 倍，新带必须判 `FAIL`。
  **落地时的一处偏离**：`±3--5` 倍场地 σ 这条**单独用达不到变异测试要求**——实测 `k` 改 `10` 倍
  只让 `e` 偏离权威 `0.0043`，任何按场地 σ（`0.005`）定的带都放行。所以带子照收（且上沿因
  "不许放宽"被夹在 `0.93`），另**加了两道按推导精度定的门**，开火的是后者。全部数字见 §9.2.5。
- ~~**T11 口径修正（比容量门更要紧）。**~~ **DONE（2026-08-06 晚，见 §9.2.8）。**
  原文：
  (a) `reward_terms_mean` 里的 `reach`/`touch` 是加权后的核均值（上限 `2.0`/`4.0`），
  receipt 要么改名（`reach_term_weighted`/`touch_term_weighted`），要么同时输出
  `touch_kernel_mean = touch/4.0`，并在 json 里写明 `max_possible`。
  (b) 把 `count_contacts` 也接进训练路径（现在只有 eval 开），让二值的
  `fraction_of_episodes_with_a_racket_touch` 出现在训练曲线上——这才是唯一有物理意义的接触指标。
  (c) 汇报一律用"零策略对照 + 二值接触率"（`0.12% → 49.2%/97.8%`），不用 `touch 4e-5→0.21`。
  (d) 曲线一律带 run 间散布（`BAND_2seed.json` 已有 N-seed band 机制）。
  **落地时多做的三件**：(a) 两样都做了（改名**并且**同时输出核均值与上限），因为只改名挡不住
  下一个人照旧把 `0.21` 当百分比；(c)/(d) 从"约定"改成**会拒绝的代码**——新增 `--report`，
  少于两条 run / 没有零策略对照 / 拿 ckpt eval 冒充对照 / run 没有二值接触率 / run 没过容量门，
  五种情形一律退出 `2` 并点名；`--analyze` 只给一份文件也从"退出 `0` 打出零宽度的带"改成退出 `2`。
  另外顺手修了一个会造假趋势的 bug：`_spearman` 用 `argsort(argsort())` 处理并列，
  **全平的曲线会被算成 `+1.0`（"单调上升"）**——二值接触率早期正好长期全 `0`。
- **T12 把 warp 的 overflow printf 接进"WARN 必进摘要"的通道。** 按 MEMORY 里的发射工序教训，
  那行字当初就在 stdout 上，只是淹在 `5292` 行里没人读。

### 9.2.5 恢复系数验收带：从空判改成能失败的门（T10 落地，2026-08-06 实测，pod1 GPU2）

**人话总结一句**：原来那道"球弹得对不对"的检查宽到**接触刚度改 `10` 倍照样盖 PASS 章**；
现在换成能真失败的门，`10` 倍刚度当场被拦下（退出码 `4`），而且球在 `1.0--4.5 m/s` 整个速度段
的斜率**第一次真测出来了**（`+1.60e-4 /(m/s)`，场地权威的 CI 是 `[-0.007, +0.018]`）。
**注意：变宽的地方一处都没有。**

改的是 `hope_training/whole_body_tracking/mjlab_lane/calibrate_restitution.py` 与
`.../a3_court_env.py` 两个文件。

#### 一、先核实 §9.2.4 说的"空判"是不是真的：是真的，我当场重现了

不是照抄 `CAL_k1000`/`CAL_k10000` 两个旧文件，是用**现役配方**（`--calibrated-b`，即 court 真正
在用的 `analytic_seed_b + 0.39`）今天在 GPU2 上重跑的：

| 跑（人话） | `e` 全幅 实测 / 闭式 | 旧带 `(0.88, 0.93)` | 新门 | 退出码 |
| --- | ---: | :---: | :---: | ---: |
| 现役刚度 `k = 1000` | `8.98e-4` / `9.03e-4` | 通过 | **PASS** | `0` |
| 刚度改 `10` 倍 `k = 10000` | `3.88e-3` / `9.03e-3` | **也通过** | **FAIL** | `4` |

旧带对两者都说"通过"。**§9.2.4 那条判定成立。** 顺带把五个旧 `CAL_k*` 收据按新门重评了一遍：
`k = 300 / 1000` PASS，`k = 3000 / 10000 / 30000` FAIL（`3000` 是 `1.15` 倍越线，勉强越）。

#### 二、新带的宽度与出处

| 量 | 值 | 出处（**每个都能翻到行**） |
| --- | ---: | --- |
| 权威 `E_TABLE_MEASURED` | `0.9215` | `configs/ball_physics_venue.yaml` → `contact.table.e_eff`。`58` 次门控弹跳，`v_n 1.0--4.5 m/s`；同一段还写了 forensics 全 `218` 次的 `0.925`、CI95 `[0.920, 0.937]`、**`± 0.005 systematic`** |
| 独立台架 `E_TABLE_OPTITRACK` | `0.9102` | `configs/ball_physics_optitrack_20260730.yaml` → 同键注释。`n = 20`，CI95 `[0.8825, 0.9311]`，**这个 CI 包含场地值** |
| **场地 σ** `E_FIELD_SIGMA` | `0.005` | 就是上面那个场地 systematic。**是场地 σ，不是仿真 σ** |
| ITTF 参考带 | `(0.876, 0.931)` | `configs/ball_physics_optitrack_20260730.yaml:125`、`docs/ball_physics_optitrack_20260730.md:209` |
| 场地平坦性 CI | `[-0.007, +0.018] /(m/s)` | 场地 F3（接触时刻修正后）："flat, slope +0.005/m/s CI [-0.007, +0.018]" |

新带 = 权威 `± 3 ×` 场地 σ = `(0.9065, 0.9365)`，**上沿被夹回 `0.93`**——因为不许放宽任何
fail-closed 门；`0.93` 同时约等于 ITTF 上沿 `0.931`。最终：

**`E_ACCEPT = (0.9065, 0.93)`，宽 `0.0235`（原 `0.05`），下沿抬高 `0.0265`，上沿一动没动。**

**旧带 `(0.88, 0.93)` 的出处：查无实据，如实标成 `UNCONFIRMED -- OPEN`，没有编。**
查过：引入 commit `3d2fce66` 及其 message、全部 `configs/*.yaml`、`docs/ball_physics*`、本卷宗。
数值上它就是 ITTF 带 `(0.876, 0.931)` 往内取整两位小数——但这是**重建，不是出处**，
代码注释里就是这么写的，不许下游把它升格成引用。

#### 三、光收紧带子不够——带子仍然不是那道门

新带 `0.0235` 宽，对仿真侧 σ（单一落高 `512` 世界今天实测 `1.27e-7`）是 `~185,000σ`；
就算按 §9.2.4 引用的 `3.47e-6` 算也还有 `~6,800σ`。
**直接证据**：`k` 改 `10` 倍，`e` 最远也只偏离权威 `0.0043`，任何按场地 σ 定的带都拦不住。
所以 T10 里"±`3--5` 倍场地 σ"这条**单独用过不了变异测试**——这一点是实测出来的，
没有靠放宽任何东西绕过去，而是**另加了两道按推导精度定的门**。

一条规则定这两个数：**标定自身的误差预算必须落在场地 systematic 的 `1/3` 以内**
（`E_CAL_TOL = 0.005 / 3 = 1.667e-3`），这样任何标定假象都不可能被当成、也不可能藏进一个
真实测到的效应里。

| 门（人话） | 判据 | 现役实测 | 余量 |
| --- | --- | ---: | ---: |
| **弹得准**：均值还落在权威上 | `\|e_mean − 0.9215\| ≤ 1.667e-3` | `1.74e-5`（扫落高）/ `8.82e-5`（单一落高） | `19--96` 倍 |
| **弹得稳**：`e` 在 `1.0--4.5 m/s` 上的全幅 | `≤ 1.667e-3` | `1.28--1.31e-3` | **只有 `1.27--1.30` 倍** |
| **不造速度依赖**：斜率落在场地 CI 内 | `[-0.007, +0.018]` | `+1.60e-4` | 距下沿 `~44` 倍 |
| **覆盖不足就不许盖章** | `<8` 个不同落高、或跨度 `<90%` 包络 ⇒ 后两项 `NOT_MEASURED` | — | — |

"弹得稳"那条有**闭式**，这是它能当门的原因：接触第一次被看见时的嵌入量是 `d ~ U(0, |v|dt]`，
它通过刚度项进 `e`，全幅正好 **`dt² · ieff · imp · k`**。**这不是物理，是积分器**，随 `k` 线性。
`k = 1e3` 预测 `9.025e-4`，实测 `8.98e-4`——对得上。
判据取**实测全幅与闭式预测的较大者**：落高取样稀会低估实测全幅（`8` 个落高比 `4096` 个低估），
闭式又比 court 实景低估约 `1.46` 倍（court 多了 elliptic 锥 + 显式 `<pair>` + `solreffriction`），
两边互补，谁也不能单独放行。

**`NOT_MEASURED` 不是 PASS。** 整跑的 verdict 只有三档：`PASS` / `FAIL` / `NOT_MEASURED`。

#### 四、(b) 扫出来的真实 `v_n` 斜率（court 实景，`4096` 世界，GPU2）

人话：让 `4096` 个世界各拿一个落高，一次跑完 `1.0--4.5 m/s` 整个速度段。

```
a3_court_env.py --nworld 4096 --ctrl pd --height-sweep 1.0 4.5 \
    --bounce-steps 1400 --steps 200 --njmax 572 --nconmax 128
```

**三次独立复跑**（mujoco-warp 非确定，**给带不给点**）：

| 量 | 第一次 | 第二次 | 第三次 |
| --- | ---: | ---: | ---: |
| 斜率 `de/dv_n` | `+1.6054e-4` | `+1.6038e-4` | `+1.6050e-4` |
| `e_mean` | `0.9214827` | `0.9214826` | `0.9214827` |
| 全幅 `e_max − e_min` | `1.3145e-3` | `1.2774e-3` | `1.2772e-3` |
| `e_min` / `e_max` | `0.9206478` / `0.9219623` | `0.9206478` / `0.9219252` | `0.9206478` / `0.9219250` |

（落盘的 `T10_SWEEP_court_4096.json` 是第三次。前两次的判定与第三次一致，重跑只是为了让收据
与最终代码逐字对上。）

- **覆盖**：`v_n` 实测跨度正好 `1.000 -- 4.500 m/s`，`4096` 个**互不相同**的冲击速度；
  `4096/4096` 世界都弹起来了，`worlds_apex_not_bracketed = 0`。
- 斜率 `+1.6e-4` 稳稳落在场地 CI 内——**仿真在这段速度上确实是平的，而且现在是量出来的，
  不是写死的**。
- **余量只有 `1.27--1.30` 倍**（全幅 `1.28--1.31e-3` 对上限 `1.667e-3`）。这道门是活的，不是摆设；
  哪天有人动 `k`、动 `dt` 或换锥，它会先叫。
- 最大嵌入 `4.31 mm`（在 `v_n = 4.5` 处）；对照单一落高 `v_n = 2.54` 时是 `0.30 mm`。
- **别引用这份收据里的 `steps_per_s`**：这一跑与另一条 workflow 的 `contact_census.py` 同时占着
  GPU2（`2974347`），吞吐数不干净。弹跳数字是确定性物理，不受影响。

#### 五、(c) 退化情形不再输出 `0.0`

对照跑（人话：故意只用一个落高，看它会不会假装测过）：
`a3_court_env.py --nworld 512 --ctrl pd --bounce-steps 900`。

- `e_vs_v_n_slope_per_m_s: null`、`e_vs_v_n_slope_status: "NOT_MEASURED"`，
  外加一行"为什么"和"重跑请加 `--height-sweep 1.0 4.5`"。
- 整跑 `restitution_acceptance.verdict = NOT_MEASURED`，**不是 PASS**。
- 退出码 `0`：容量普查跑本来就没声称测过弹跳，不该被它阻断；但收据自己说清楚它**不能当证据**。
  真的 `FAIL` 才阻断（`a3_court_env.py` 退出码 `3`；`calibrate_restitution.py --confirm`
  退出码 `4`，`NOT_MEASURED` 是 `5`）。

#### 六、(d) 独立样本如实记账

收据里新增 `independent_samples` 块：

- `n_worlds: 4096`；`n_distinct_impact_speeds: 4096`（扫落高）/ `1`（单一落高）
- `worlds_are_not_independent_samples: true`
- **`independent_dof_for_e_mean: 1`**——均值永远只有一个独立自由度，跑几个世界都一样。
- 单一落高时世界之间差的是什么：**mujoco-warp 调度非确定性**，不是测量噪声、不是参数采样，
  这条探针里根本没有随机种子。实测 `e_std = 1.27e-7`（`512` 世界）。
- 一句人话直接写进 json：**"不要拿按世界数算出来的标准误当证据。"**

#### 七、顺带补上的一个静默坑（原来没人提）

court 的弹跳分析原来直接取 `max(z[i1:i2])` 当反弹顶点，**不检查顶点有没有被时间窗夹住**。
窗口太短时顶点还没到，`e` 会**静默偏低**——一个看起来像"测过"的数其实不是测量。
现在逐世界检查、不合格的世界从统计里剔除并计数（`worlds_apex_not_bracketed`），
跑之前还会按最大落高算出需要多少步并打 WARNING。
本次扫描 `v_n = 4.5`（落高 `1.032 m`）光升到顶点就要约 `930` 步，所以用 `--bounce-steps 1400`；
历史那份 `0.33 m` / `900` 步是安全的（约 `738` 步就落回桌面），**旧收据不受影响**。

#### 八、零回归

| 检查（人话） | 结果 |
| --- | --- |
| 训练脚本还能 import court（`a3_train_ppo`） | OK |
| court 的模型逐名核对路径（`--verify --no-bench`） | 退出 `0`，`n_unregistered_mismatch = 0` |
| `contact_census.py --scene court` 还能建场景 | 退出 `0`，`status: complete` |
| `calibrate_restitution.py --rest`（走了被重写的 `_ieff`） | 退出 `0`，静止嵌入 `0.429 mm` = 球半径 `2.14%`，与文档一致 |
| `calibrate_restitution.py --sweep`（非 confirm 路径必须不被设门） | 退出 `0`，`b_solved = 2022.58` ≈ 现役 `2022.55` |
| `calibrate_restitution.py --validate-model` | 退出 `0`，闭式 vs 引擎 `max_abs_error = 7.08e-3` |

仓库里没有任何测试 import 这两个文件（`grep` 全仓确认），所以"相关测试"就是上面这六项真实消费者。

#### 九、这一节没解决的（不许当成已闭合）

1. **新带的上沿仍然是继承来的 `0.93`，不是推导出来的**——按场地 σ 推出来是 `0.9365`，
   那会**放宽**上沿，这轮不许。等哪天拿到 `(0.88, 0.93)` 的真出处，或者补一次 ITTF 30 cm
   落球测试，再重定上沿。
2. **场地 σ 只有一个可引的数**（`± 0.005` systematic）。两台设备（场地 `0.9215` /
   OptiTrack `0.9102`）相差 `0.0113`，是这个 systematic 的 `2.3` 倍——**哪台对没有裁定**，
   `configs/ball_physics_optitrack_20260730.yaml` 自己写着要"到比赛桌上做一次 30 cm 落球试验"
   才能判。T10 没动这条。
3. **球拍那一路完全没碰**：paddle 的 `e` 是速度相关的（`0.759·exp(−0.0441·u_n)`），
   静态 solref 表达不了，仍是 §9.2.1 记着的named gap。
4. **网**（`NET_E_ASSUMED = 0.10`）依旧是假设值，没有任何测量，也不在这套门的管辖内。

#### 十、收据（pod1 `/workspace/mjlab_lane/`）

| 文件 | 一句人话 |
| --- | --- |
| `T10_SWEEP_court_4096.json` | court 实景 `4096` 世界扫 `1.0--4.5 m/s`，真斜率就在里面 |
| `T10_MUT_k1000.json` | 变异测试对照组：现役刚度，退出 `0`，`status: restitution_pass` |
| `T10_MUT_k10000.json` | 变异测试实验组：刚度 `10` 倍，退出 `4`，`status: restitution_fail`，stderr 点名开火的那条门 |
| `T10_SINGLEH_court_512.json` | 退化情形对照：单一落高，斜率 `null`、整跑 `NOT_MEASURED` |
| `T10_device.json` / `T10_smi_start.txt` / `T10_smi_end.txt` | 落卡自证：`device_count = 1`、uuid `473a79f3-8736-6c7f-c3db-290c6be385b8`，与 `nvidia-smi` 的 `compute_procs` 对得上；GPU0/GPU1 全程 `2--5 MiB, 0 %` |
| `T10.status` | 四条跑的退出码 |
| `T10_REG_*.json`、`T10_VERIFY.json`、`T10_CENSUS_SMOKE.json` | 上面第八节那六项零回归检查 |
| `t10_run.sh` | 复跑脚本；`export CUDA_VISIBLE_DEVICES=2` **加了** `CUDA_DEVICE_ORDER=PCI_BUS_ID`（T7(c)） |

**失败路径也留收据**：门开火那一跑照样落 `.json`，里面自带 `status: restitution_fail` 与
`restitution_verdict`；`main()` 外面还包了一层，进程崩了也会先把 `status: crashed` 和异常写盘。
这是 §9.2.4 P9（"门一开火证据就消失"）在这条 lane 上的对症修法。

### 9.2.7 容量普查改成收敛判据 + 非准静态场景（T9 落地，2026-08-06 实测，pod1 GPU2）

**人话总结一句**：以前的普查是"跑 `3000` 步，把见过的最大值抄下来当峰值"；现在是"跑到不再刷新
纪录为止，刷不停就当场判 `NOT_CONVERGED`、余量数字改名加 `_lower_bound` 后缀、退出码非零"。
同时把机器人**真的开起来**（合法力矩、合法构型），并把**摩擦锥写进收据、跨锥比较直接拒绝**。
结论：**现役 `njmax=572` / `nconmax=128` 够用，但余量是 `2.07x`，不是原来说的"至少 `3x`"。**
**这一轮没有任何一处放宽，全部是收紧。**

顺带撞到一条以前没人见过的东西：**`EPA_HORIZON` 溢出在这个场景里是真会发生的**——
`4096` 个世界里 `1` 个，分配远没用满，纯靠读 `d.overflow` 才看得见。详见下面"意外收获"。

改的是一个文件 `hope_training/whole_body_tracking/mjlab_lane/contact_census.py`（重写），
外加纯逻辑单测 `hope_training/whole_body_tracking/tests/test_contact_census_convergence.py`
（`31` 条，不需要 GPU/torch/mujoco，`0.07 s` 跑完），以及把
`hope_training/whole_body_tracking/mjlab_lane/a3_plant_env.py` 里那段**已经被证伪的容量注释**
就地换成实测值。**`a3_train_ppo.py` 一行没动**——T1--T8 是另一批活。

#### 改了什么（每条配一句人话）

| 改动 | 人话 | 不改会怎样 |
| --- | --- | --- |
| **收敛判据**：`peak_at_step > 0.7 x steps` ⇒ `NOT_CONVERGED` | 峰值必须落在跑程的前 `70%`；落在后面说明曲线还在爬，你只是先不看了 | 就是 §9.2.3 更正三那件事：`3000` 步的"峰值"其实是下界 |
| **跑到"最近 K 步无新纪录"为止**（`--stall-steps`，默认 `3000`）。停机条件实际用的是 `max(K, 0.3 x 已跑步数)`，否则会停进一个自己造出来的 `NOT_CONVERGED` | 不再拍脑袋定步数，由数据自己说什么时候够 | 定长窗口对 `ctrl=0` 永远不够，对 `bang` 又浪费 `4` 倍时间 |
| **收据输出 running-max 时间序列**，不是单个标量 | 一眼看出"早就平了"还是"到最后一刻还在涨" | 单个数字读不出趋势，只能事后再跑一遍 |
| **非准静态场景**：`flail`（`ctrlrange` 内随机力矩）、`bang`（满力矩正反跳变）、`randpose`（`jnt_range` 内随机构型 + 随机根姿态释放）、`slam`（randpose 再加向下根速度砸桌子） | 机器人被驱动起来才是真的最坏情况，瘫倒摊平不是 | §9.2.3 更正二：`ctrl=0` 低估约 `2` 倍 |
| **摩擦锥进收据**，记的是**建出来的模型**报的锥（`m.opt.cone`），不是命令行那个字符串；请求与实际不符直接退出。`--cone` 改成**必填** | 收据自己说得清"这是 pyramidal 还是 elliptic、一个接触几行" | §9.2.3 更正一那个假因果就是跨锥相减造出来的 |
| **跨锥比较硬拒绝**（`--compare A.json B.json`，退出码 `2`） | 一个接触 pyramidal 收 `4` 行、elliptic 收 `3` 行；跨锥的差是记账差，不是物理 | 会再生产一次"加桌子降低行数"这种结论 |
| **未收敛的信号按格拒绝**，不是整份收据拒绝 | `ctrl=0` 的宽相没收敛，不该连累同一份收据里 `bang` 那格能不能比 | 整份拒绝太钝，会逼人绕过工具手算 |
| **读引擎自己的 `d.overflow`**（九位粘性掩码，逐世界 OR） | 九类溢出一次全覆盖 | 自制近似只能看见 `nefc` 一条轴（§9.2.4 P1/P7） |
| **`naconmax` 余量换分母**：`max(nacon_peak, ncollision_peak)` | 宽相候选对和窄相接触共用同一块 `naconmax`，而候选对恒 `≥` 接触数 | §9.2.3 更正四：`9.45x` 是除错了分母 |
| **零测量绝不签 PASS**：空序列返回 `None` 而不是 `0`，`verdict` 走 `NO_SAMPLES` | "没测"和"测了且真是 0"必须分得开 | §9.2.4 P2 在训练门上的同款毛病 |
| **`>` 而不是 `>=` 判溢出** | 引擎判据是 `nefc > njmax`，`nefc == njmax` 是正好装下 | §9.2.4 P10：会在没坏的 run 上打报错文案 |
| **失败路径也落收据**：写盘放在 `finally`，崩了先写 `status: crashed` | 门开火那一跑恰恰是最需要证据的一跑 | §9.2.4 P9 |
| **收据自陈落卡**：进程内 `device_uuid` / `pci_bus_id` / `device_count` | 事后能和 `nvidia-smi` 对上 | §9.2.4 T7(a) |

#### 跑出来的数（`4096` 世界，逐场景逐锥，跑到收敛判据说停为止）

测量分配故意开大（`njmax=1024`、`nconmax=192`/世界 → `naconmax=786,432`），这样引擎不会先把
要量的东西裁掉；**打分是拿实测需求去比现役的 `572` / `128`**。所有跑都在 GPU2
（收据自陈 `uuid 473a79f3-8736-6c7f-c3db-290c6be385b8`、`pci_bus_id 190`、`device_count=1`），
GPU0/GPU1 全程 `2--5 MiB`。

**court（robot + 桌 + 网 + 球，真正在训练的那个场景）**

| 场景 | 锥 | 跑了多少步 | 为什么停 | `nefc` 行/世界 | 峰值在第几步 | 收敛 | 接触/世界 | `ncollision` |
| --- | :---: | ---: | --- | ---: | ---: | :---: | ---: | ---: |
| `zero`（ctrl=0 摊平） | elliptic | `30,000` | 撞上限 | `110` | `25,590` | **否** | `25` | `172,521`（第 `29,985` 步） |
| `flail` | elliptic | `21,000` | 不再刷新 | `135` | `13,114` | 是 | `29` | `82,188` |
| `bang` | elliptic | `15,000` | 不再刷新 | `137` | `8,713` | 是 | `34` | `71,311` |
| **`randpose`** | elliptic | `18,000` | 不再刷新 | **`265`** | `11,437` | 是 | **`57`** | `99,619` |
| `slam` | elliptic | `14,000` | 不再刷新 | `161` | `6,080` | 是 | `40` | `97,753` |
| `zero` | pyramidal | `30,000` | 撞上限 | `131` | `13,713` | 行收敛 / 宽相**否** | `24` | `168,991` |
| `flail` | pyramidal | `11,000` | 不再刷新 | `170` | `554` | 是 | `27` | `82,418` |
| `bang` | pyramidal | `7,000` | 不再刷新 | `174` | `559` | 是 | `34` | `71,711` |
| **`randpose`** | pyramidal | `14,000` | 不再刷新 | **`265`** | `3,007` | 是 | **`57`** | `103,978` |
| `slam` | pyramidal | `19,000` | 不再刷新 | `211` | `9,758` | 是 | `44` | `100,987` |

**plant（光机器人，诊断场景，不训练）**

| 场景 | 锥 | 步数 | 为什么停 | `nefc` 行/世界 | 收敛 | 接触/世界 | `ncollision` |
| --- | :---: | ---: | --- | ---: | :---: | ---: | ---: |
| `zero` | pyramidal | `30,000` | 撞上限 | `159` | 行收敛 / 宽相**否** | `31` | **`268,846`** |
| `flail` | pyramidal | `28,000` | 不再刷新 | `170` | 是 | `32` | `77,615` |
| `bang` | pyramidal | `15,000` | 不再刷新 | `167` | 是 | `31` | `71,838` |
| **`randpose`** | pyramidal | `26,000` | 不再刷新 | **`276`** | 是 | `57` | `108,666` |
| `slam` | pyramidal | `18,000` | 不再刷新 | `244` | 是 | `44` | `120,825` |
| `zero` | elliptic | `14,000` | 不再刷新 | `108` | 是 | `24` | `239,378` |
| `flail` | elliptic | `16,000` | 不再刷新 | `128` | 是 | `28` | `77,220` |
| `bang` | elliptic | `20,000` | 不再刷新 | `140` | 是 | `33` | `70,630` |
| `randpose` | elliptic | `24,000` | 不再刷新 | `217` | 是 | **`62`** | `105,401` |
| `slam` | elliptic | `12,000` | 不再刷新 | `161` | 是 | `36` | `118,300` |

这 `20` 组的 `world_steps_over_reference_njmax` 与 `world_samples_over_reference_nconmax`
**全是 `0`**，`d.overflow` 九位掩码**也全是 `0`**。这次"没溢出"是**被九位掩码证明的**，
不是"我们盯的那一条计数器没响"。

#### 三件这轮才看清楚的事

1. **`ctrl=0` 是唯一一条测不完的场景。** court `4096` 拉到 `30,000` 步（原来只在 `2048` 世界
   跑过 `12,000`），elliptic 最后一次刷新纪录在第 `25,590` 步（`85%` 处），`ncollision` 峰值在第
   `29,985` 步——**倒数第 15 步**。其余四条 `7,000--28,000` 步都停了。所以"普查跑不到收敛"不是
   方法的通病，是这个场景的性质：瘫倒之后世界还在缓慢重排接触，几万步都停不下来。
   **`ctrl=0` 既不是最坏场景、又是唯一测不完的场景，两头都不占，不该再当容量基准。**
   （这条同时补上 §9.2.4 E9："`4096` 世界 court 的 `ctrl=0` 长跑没做过"——做了。）
2. **`randpose` 的峰值本身是个分布，不是一个点。** 同配置换随机种子，court `nefc` 峰值：
   elliptic `seed 0/7/13/29 = 265 / 203 / 208 / 200`；pyramidal `= 265 / 263 / 268 / 255`。
   **elliptic 的带是 `200--265`（`265` 是四次里唯一的极端值），pyramidal 的带是 `255--268`。**
   按 MEMORY 里"要报就报带，别报点"，下面余量一律按**带的上沿**算。
   顺带说明一件事：elliptic `seed 0` 和 pyramidal `seed 0` 都是 `265`，这是**巧合**——
   四个种子一比就散开了，不是什么结构性天花板。
3. **"桌子降低行数"是分场景的，不是一个常数。** 同锥（pyramidal）、同 `4096` 世界、同收敛判据，
   `--compare` 直接给出 plant→court：`ctrl=0` `159→131`（`-18%`）、`slam` `244→211`（`-14%`）、
   `randpose` `276→265`（`-4%`）、`flail` `170→170`（`0%`）、`bang` `167→174`（**`+4%`**）。
   桌子只在"机器人自己摊下去"这类场景里帮忙；有力矩驱动时帮助归零甚至反号。
   同一次 `--compare` 里 `zero` 的 `nacon` / `ncollision` 两格被**按格拒绝**并写明理由
   （两边都没收敛，差是无符号的），其余五格照常给出——这就是"按格拒绝"的用处。

#### 意外收获：`EPA_HORIZON` 第一次被真的看见了

`randpose` + pyramidal + `seed 13` 那一跑，`d.overflow` 上出现了 **`EPA_HORIZON`**（`4096` 个世界
里 `1` 个）。这是六次 seed 跑里唯一一次，**也是这条 lane 有史以来第一次实际观测到任何一位溢出
在非人为压小分配的情况下被置起来**。测量分配是 `njmax=1024` / `nconmax=192`，两个都远没用满
（实测需求 `268` 行 / `57` 接触/世界）——所以这一位跟容量无关。

**它并不是"只置位不打印"**（§9.2.2 与 §9.2.4 P7 原来那句"`HFIELD`/`NVMAX`/`CONTACT_MATCH`/
`EPA_HORIZON` 四种只置位、不打印"就此撤回）：`collision_gjk.py:1392/1411` 的 `atomic_or` 上面
紧贴着一行 `if warn_overflow: wp.printf(...)`，原判据只看了 `atomic_or` 那一行。
**但它打的那句话里没有 "overflow" 这个词**——原文是
`Warning: EPA horizon = 24 isn't large enough.`，这才是真正的坑：

- **补做了正确口径的历史复查。** 拿 `EPA horizon = %d isn't large enough` 这个真字符串重扫
  `/workspace/mjlab_lane/` 下所有历史 `.log` / `.out`，**命中 `0` 条**（同一次扫描里
  `nefc overflow` / `broadphase overflow` / `narrowphase overflow` 分别命中
  `4,653,815` / `2,742` / `244` 行，全部来自那几次故意压小分配的变异跑，说明模式是有效的）。
  所以 §9.2.4 C1 对 08-05 双 seed 的清白判定**仍然成立**，只是它原来那条
  `grep -ci overflow` 结构上覆盖不到 `EPA_HORIZON`，得换成这一组模式——**而正解还是读
  `d.overflow`**。
- **它不是 `njmax`/`nconmax` 能修的。** EPA horizon 是凸体碰撞 GJK/EPA 里一个**编译期定长**
  的缓冲（这里是 `24`），跟约束行数和接触数组都无关。`CAPACITY_OVERFLOW` 那句
  "Re-size with --njmax/--nconmax" 对这一类是**错误建议**。
- **登记为具名缺口**：已知可达，观测频率是"六次 `4096` 世界长跑出现一次、影响 `1` 个世界"，
  后果是那一步那一个世界的接触法向/穿深可能算错，不崩、不 NaN（同跑
  `worlds_with_nan` 路径无异常）。目前无修法，只有检出。

#### 变异测试：证明门真的会开火

按"改软硬门要连证据一起改"的准绳，每一条都是**先让它开火**，不是只证明它现在不报错。
收据在 pod1 `/workspace/advcheck/mut/`，每份自带 `verdict`、`flags` 和退出码。

| 变异 | 想证明什么 | 实测结果 |
| --- | --- | --- |
| `--max-steps 0` | 零测量绝不签 PASS | **`NO_SAMPLES`，退出 `1`**；四个场景的 `peak` 全是 `null`（不是 `0`） |
| `4096 x 3000` 定长窗口跑 `ctrl=0` | 老普查签过字的那个形状，现在会被判未收敛 | **`NOT_CONVERGED`，退出 `1`**；峰值 `93` 落在第 `2,628` 步 = `88%` 处 |
| `--ref-njmax 100` 跑 `randpose` | 需求超过参考分配会被拦 | **`OVER_REFERENCE_ALLOCATION`，退出 `1`**（实测需求 `167` > `100`） |
| `--njmax 70` | 引擎真 `nefc` 溢出时 `d.overflow` 读得到 | **`ENGINE_OVERFLOW ['NEFC']`，退出 `1`** |
| **`--nconmax 30`，`4096` 世界** | **§9.2.4 P1 那个盲区**：宽相溢出、窄相没溢出 | **`ENGINE_OVERFLOW ['BROADPHASE']`，`4096/4096` 世界置位，退出 `1`**；引擎同时打了 `192` 行 broadphase printf |
| 同上，但 `--nconmax 40` | 阈值另一侧的对照，证明不是恒报 | 无 flag、`0` 行 printf（`naconmax=163,840` > 需求 `138,021`） |
| `--compare` 一份 pyramidal 和一份 elliptic 收据 | 跨锥比较会被拒绝 | **`REFUSED`，退出 `2`**，报错点名两边的锥和 `4` vs `3` 行/接触 |
| `--compare` 两份 pyramidal 收据 | 同锥能比，且只丢没收敛的格 | 退出 `0`，`5` 个 `nefc` 格给出差值，`zero` 的两个宽相格被按格拒绝并写明理由 |

**`--nconmax 30` 这一格值得单独看**：它复现了 §9.2.4 P1 说的反相关。同样 `2,000` 步，
`nconmax=40`（不溢出）时 `nacon` 峰值 `46,999`；`nconmax=30`（宽相溢出）时 `nacon` 峰值反而
**降到 `41,897`**——候选对在进窄相**之前**就被丢掉了，所以**溢出越深，被监视的 `nacon` 看起来
越健康**。shipped 训练门盯的正是 `nacon`，它在这一格会放行；这一版普查读 `d.overflow`，当场判死。

**一条没做成的**：`--nconmax 10` 在 `4096` 世界直接 **segfault（退出 `139`）**，进程在门读数之前
就死了，连 `.json` 都没落。这正是 §9.2.4 P4 描述的那段区间——**深压 `nconmax` 时 CUDA 非法访问
先到、门后到**，普查这一侧同样抢不到那一拍。想守住这段区间只能缩短判决延迟，本轮没做。

#### 回答 T9 的那个问题：现役 `572` / `128` 到底够不够

**够。但"至少 `3x`"这句要撤回。** 按最坏那一格算（`randpose` 取四个种子的上沿）：

| 管什么 | 现役 | 最坏实测需求（收敛后） | 余量 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 一个世界的约束行 `njmax` | `572` | `276`（plant pyramidal `randpose`） | **`2.07x`** | 训练场景 court 是 `268` → **`2.13x`** |
| 一个世界的接触数 `nconmax` | `128` | `62`（plant elliptic `randpose`） | **`2.06x`** | court 是 `57` → `2.25x` |
| 宽相候选对 `naconmax` | `524,288` | `172,521`（court elliptic `ctrl=0`，**未收敛，是下界**） | **`≥3.04x`** | plant-only 那格是 `268,846` → **`≥1.95x`** |

只看**合法力矩**（`flail`/`bang`：从 ready pose 出发、力矩不越 `ctrlrange`，这是策略真能输出的
东西）：最坏是 court-pyramidal `bang` 的 `174` 行 → **`3.29x`**。
`randpose` 那 `2.07x` 对应的是**"reset 随机化如果放开到在 `jnt_range` 里自由采样"**这个假设，
现役 court 的 reset 并不这么做——所以 `2.07x` 是**上界式的悲观口径**，不是当前工况。

**要不要改分配：现在不用改。** 如果哪天同时满足 (a) 为摩擦保真度切回 pyramidal、
(b) reset 随机化放开到 `jnt_range` 自由采样，想把余量拉回 `3x`，需要
`njmax >= 3 x 276 = 828`、`nconmax >= 3 x 62 = 186`。本轮测量用的 `njmax=1024` / `nconmax=192`
就已经越过这条线，且 `20` 组场景一次溢出都没有，可以直接当推荐值。
**另有一格现在就低于 `2x`**：plant-only（光机器人诊断场景，不训练）在 `4096` 世界跑
`ctrl=0` 长跑时宽相只剩 `1.95x` 且未收敛——谁要做这件事，先把 `nconmax` 抬到 `192`。

#### 这一节不代签什么

- **它不修训练门。** §9.2.4 的 T1--T8 全部指向 `a3_train_ppo.py`，这一轮一行都没动那个文件。
  **"数据可信、门不可信"这句裁定继续成立**：普查这一侧现在覆盖九类溢出，
  但**训练循环里那道门仍然只看 `nefc`**。
- **它不代表策略驱动的真实构型分布。** §9.2.4 E5 那个洞还在：这里的场景是
  `ctrl=0` / 随机力矩 / 随机构型，"早期随机策略 + reset 随机化 + 课程"下的真实分布没测过。
  `flail`/`bang` 是它的下界、`randpose` 是它的上界，真值在中间，位置不知道。
- **`ctrl=0` 那两格的 `ncollision` 余量仍然是下界。** 到 `30,000` 步它还在涨。
- **深压 `nconmax` 那段区间仍然无人接管**（上面那条 segfault）。

#### 收据在哪

pod1 `/workspace/advcheck/`：`T9_COURT_elliptic_4096.json` / `T9_COURT_pyramidal_4096.json` /
`T9_PLANT_pyramidal_4096.json` / `T9_PLANT_elliptic_4096.json`（四份主矩阵，各含
running-max 时间序列）、`seed/randpose_{elliptic,pyramidal}_s{7,13,29}.json`（种子带 +
`EPA_HORIZON` 那一份）、`mut/M*.json` 与 `mut/M6*.log`（变异测试）、
`t9_matrix2.sh` / `t9_seedcheck.sh` / `t9_mutations.sh` / `t9_mut2.sh`（复跑脚本，
全部 `CUDA_VISIBLE_DEVICES=2` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`）。

**踩到的坑，记一笔**：容器的根 overlay（`30G`）在这轮中途被撑满（`/tmp/IsaacLab` `15G` +
`/tmp/pytest-of-root` `5.5G`，都不是本 lane 的），warp 写内核缓存直接 `ENOSPC`，三份跑挂掉。
修法是照 `run_gpu2_smoke.sh` 的老规矩把 `WARP_CACHE_PATH` / `TMPDIR` / `CUDA_CACHE_PATH`
全指到 `/workspace`（`153G` 空闲）——**没有删任何别人的东西**。新写的脚本都带这三个 export。

### 9.2.6 容量门重做：不再自己数 nefc，改读引擎的 d.overflow（T1--T8/T12 落地，2026-08-06 实测，pod1 GPU2）

**人话一句**：旧看门狗自己数"这一步用了多少约束行、多少接触"，再跟预分配上限比——它数错了地方。
`nconmax` 真正管住的是**宽相候选对**，而宽相一溢出，多出来的候选对在进窄相**之前**就被扔掉，
于是被监视的那个数永远碰不到上限：**溢出越深，仪表读数越健康**。现在不自己数了，直接读引擎自己的
记录：mujoco-warp 给每个世界留了一个整数 `d.overflow`，九种溢出各占一位，引擎自己置位、不清零。
任何一位亮起就当场停机，并把亮的是哪一位（`BROADPHASE` / `NEFC` / …）写进报错和收据。

改的是 `hope_training/whole_body_tracking/mjlab_lane/a3_train_ppo.py`，加
`tests/test_mjlab_lane_capacity_gate.py`（`17` 条，pod1 mjlab venv 全绿）。

#### 一、逐条改了什么（每条配一行人话）

| 编号 | 改动 |
| --- | --- |
| **T1** | 判据换成引擎的 `d.overflow`（`mujoco_warp/_src/types.py:2350`，`array("nworld", int)`，逐世界粘性 OR，一次覆盖全部 `9` 类）。每个 physics substep 在 GPU 上做一次逐世界按位或累进，不同步；`nefc`/`nacon` 峰值**降级为只用来算余量的报告值**，不再是门。 |
| **T1 的陷阱** | 采样点**必须留在 decimation 循环里**。`step()` 末尾的 `sim.reset(ids)` 会把被 reset 世界的 `d.overflow` 清零（`io.py:2483`），挪到 reset 之后就正好丢掉要抓的那份证据。 |
| **T2** | `ncollision`（宽相候选对）纳入监控，收据单列 `ncollision_peak_all_worlds_running`；接触余量的分母从 `nacon_peak` 改成 `max(nacon_peak, ncollision_peak)`。 |
| **T3** | 判决从"每个 PPO 迭代（`480` substep）一次"缩到"**每个 env step（`20` substep）一次**"：把逐世界掩码 OR 归约成一个 `9` 位数读回，每 env step 一次同步（那里本来就有 `dones.nonzero()` 的同步）。 |
| **T4** | PASS 必须有证据。分开数 `capacity_samples_stepped`（真跑过 physics step）与 `capacity_samples_forward`；stepped 为 `0` 时判 `NO_SAMPLES`，并且**所有 headroom 字段写 `null`**，不再写一个大数。 |
| **T5** | eval 路径同样设门——门就在 `env.step()` 里，eval 走的是同一条。收据加 `capacity_gate: ENFORCED` / `NOT_GATED` 与 `verdict`，`OVERFLOW` 时退出码非零；`--no-capacity-probe` 时明写 `NOT_GATED` 并打 WARN。 |
| **T6** | `reset()` 与每控制步补发球后的 `sim.forward()` 两处补采样，**并且补一个手算判据**。原因是新查到的：`mjwarp.forward()` 根本不跑 `_next_time`（那个 kernel 只在 `_advance` 里，只被 `step()` 的积分器调用），所以 forward 里溢出时 `NEFC`/`NJMAX_NNZ`/`BROADPHASE`/`NARROWPHASE` **一位都不会被置**。这两处按引擎自己的谓词（`nefc > njmax`、`ncollision > naconmax`、`nacon > naconmax`，全是严格大于）现算一遍再或进去。 |
| **T7(a)** | 收据加进程内落卡自证：`device_uuid` / `pci_bus_id` / `torch_cuda_device_count` / `torch_current_device_index`，并按 PID 与 `nvidia_smi` 的 `compute_procs` 对账（`device_uuid_matches_nvidia_smi`）。 |
| **T7(b)** | `train()` / `evaluate()` 整个包进 `try/finally`：**任何退出路径都落 `.json`**，含 `status`（`completed`/`gate_fired`/`crashed`）、`exit_code`、异常与 traceback、落卡 uuid、`argv`。**场景构建期开火也照落**——第一次 `forward()` 就撞上时连 env 都没建成，这时用异常自带的掩码补进收据（`_merge_gate_error`）。 |
| **T7(c)** | `CUDA_DEVICE_ORDER=PCI_BUS_ID`：pod 的 `run_gpu2_smoke.sh` / `audit.sh` 加了 export，**而且 `a3_train_ppo.py` 自己在 import 任何会初始化 CUDA 的东西之前做 `os.environ.setdefault`**——shell 里忘写也不会退回 `FASTEST_FIRST`。 |
| **T12** | `--warn-scan-log PATH`：跑完扫自己的 stdout 日志，按引擎的原话（不是裸 `overflow` 一个词）数告警，进收据 `warp_stdout_overflow_scan` 并打 `[WARN][...]` 块。**GPU 侧读干净但 stdout 有告警 → 判 `OVERFLOW_PRINTF_ONLY` 且非零退出**：两个通道不一致本身就是不合格。pod 两个脚本也加了同样的 WARN 扫描。 |
| **P10** | `nefc == njmax` 是**正好装下**（引擎丢行判据是 `nefc > njmax`，`forward.py:248`）。`>=` 改 `>`，另出 `nefc_exactly_fills_njmax` 字段留痕。 |

**收据字段改名，下游要跟着改**：per-iteration 峰值改成**全跑累计**峰值，所以
`nefc_peak_per_world` → `nefc_peak_per_world_running`、`nacon_peak_all_worlds` →
`nacon_peak_all_worlds_running`；新增 `ncollision_peak_all_worlds_running` /
`naconmax_binding_peak_all_worlds` / `overflow_mask` / `overflow_flags` /
`worlds_with_any_overflow_flag` / `capacity_samples_stepped` / `capacity_samples_forward`；
删掉 `njmax_saturated` / `naconmax_saturated`，换成 `nefc_over_njmax` /
`nefc_exactly_fills_njmax` / `naconmax_binding_over`。`verdict` 从三值变五值：
`NOT_MEASURED`（门被显式关掉）/ `NO_SAMPLES`（门开着但一个 physics step 都没跑）/
`OVERFLOW` / `OVERFLOW_PRINTF_ONLY`（GPU 侧干净但 stdout 有告警）/ `PASS_NO_OVERFLOW`。

#### 二、变异测试：修复前 / 修复后（同一张 GPU2、同一天、同一份 `a3_court_env.py`）

前一栏跑的是当天保下来的 `a3_train_ppo_BEFORE.py`（与 `git HEAD` 的 shipped 版逐字节相同），
后一栏跑的是本次改完的版本。规模都是 `--nworld 256`，除 T8(3) 外都是 `3` 迭代。

| 变异 | 修复前 | 修复后 |
| --- | --- | --- |
| **T8(1) 盲区带 `--nconmax 10`**（`naconmax = 2560`，真实 `ncollision` 峰值 `2625`） | 退出码 **`0`**；`verdict: PASS_NO_OVERFLOW`；`naconmax_headroom_x = 1.4175`（分母用了 `nacon = 1806`）；同一跑里引擎打了 `1134` 行 `broadphase overflow` | 退出码 **`1`**；`CAPACITY_OVERFLOW at env step 16: d.overflow = 4 = BROADPHASE`；`verdict: OVERFLOW`、`overflow_flags: ["BROADPHASE"]`；**换对分母后 `naconmax_headroom_x = 0.975`**（`2625 / 2560`，本来就 `< 1`）；收据落盘且 `device_uuid_matches_nvidia_smi: true` |
| **T8(2) 深压 `--nconmax 4`** | 退出码 `1`，但**不是门拦的**：`387` 行 `CUDA error 700: an illegal memory access`；**没有 `.json` 收据**（P9 那条"最需要证据的那跑反而没证据"） | 退出码 **`1`**，**在崩之前拦住**：`CAPACITY_OVERFLOW at reset: d.overflow = 12 = BROADPHASE\|NARROWPHASE`，`0` 行 CUDA 报错；`.json` 收据落盘，`status: gate_fired`、`verdict: OVERFLOW`、`overflow_flags: ["BROADPHASE","NARROWPHASE"]`、`overflow_reported_by: "reset"`、GPU uuid 对得上 |
| **T8(3) `--iterations 0`** | 退出码 `0`；`verdict: PASS_NO_OVERFLOW`；`njmax_headroom_x: 572.0`、`naconmax_headroom_x: 32768.0`（零样本签发满余量 PASS） | 退出码 `0`；`verdict: **NO_SAMPLES**`；`capacity_samples_stepped: 0`；**所有 headroom 字段是 `null`**；`[WARN][CAPACITY NOT MEASURED]` 进摘要。**不再出现 PASS** |

**T8(2) 那一格顺带证了两件事**：(a) 门比 CUDA fault 早到，`reset` 那一次 `forward()` 就拦住了；
(b) 那一跑引擎 printf **`0` 行**——因为 `forward()` 不跑 `_next_time`，stdout 通道在这里是瞎的，
**只有读 `d.overflow`（外加 T6 的手算判据）才看得见**。这也是为什么 T12 只能当旁证、不能当门。

**另外两条对照（不在 T8 要求里，但没有它们不算验收）**：

| 对照 | 结果 |
| --- | --- |
| `--njmax 70`（`nefc` 轴，`--nworld 64`） | 退出 `1`，`CAPACITY_OVERFLOW at env step 46/49: d.overflow = 1 = **NEFC**`，`nefc` 峰 `71 > 70`。两次复现落在 env step `46` 与 `49`——**mujoco-warp 非确定，这个数不要写死** |
| 健康配置 `--nconmax 128`（假阳性检查） | 退出 `0`，`verdict: PASS_NO_OVERFLOW`，`nefc 77--80`、`nacon 1870`、`ncollision 4692`，`naconmax_headroom_x = 6.98`（若照旧除 `nacon` 会写成 `17.5`，虚高 `2.5` 倍） |
| `--no-capacity-probe` | 退出 `0`，`verdict: NOT_MEASURED`，`[WARN][CAPACITY GATE OFF]` 进摘要 |
| **eval 路径同一条变异**（`--eval zero --nconmax 10`，`256` 世界） | 退出 **`1`**，`status: gate_fired`、`capacity_gate: ENFORCED`、`verdict: OVERFLOW`、`overflow_flags: ["BROADPHASE"]`、`overflow_reported_by: "env step 18"`、`naconmax_headroom_x = 0.973`，收据落盘。**修复前 `evaluate()` 恒 `return 0`、从不判决**（§9.2.4 P5） |
| eval 健康 + eval 关门 | 健康：退出 `0`、`capacity_gate: ENFORCED`、`PASS_NO_OVERFLOW`、`capacity_samples_stepped = 800`，接触探针照常（`ball_table_contact_substeps = 143`）。`--no-capacity-probe`：退出 `0`、`capacity_gate: **NOT_GATED**`、`[WARN][EVAL NOT GATED]` 进摘要 |

#### 三、吞吐：§9.2.2 那个"探针吃掉 9%"不成立，撤回

配对实测：`--nworld 4096 --iterations 12 --seed 0`，同一张 GPU2 背靠背交替跑，
取**去掉 iteration 0 之后的中位数**（收据里逐条记了跑前/跑后 GPU2 上有没有别的进程，
下表六条全是 `others_before = 0, others_after = 0`）。

| 配置 | 各次中位 env-step/s | 中位 |
| --- | --- | ---: |
| 新门 ON | `45,091` / `44,767` / `45,041` | `45,041` |
| 新门 OFF（`--no-capacity-probe`） | `44,994` / `44,833` / `44,800` | `44,833` |
| 旧看门狗 ON | `44,991` / `44,855` | `44,923` |
| 旧看门狗 OFF | `45,296` / `45,107` | `45,202` |

**新门的代价在噪声里**（三对里两对 ON 反而略快，差值 `±0.6%`）；**旧看门狗也只有约 `0.6%`**。
所以 §9.2.2 写的"探针吃掉约 `9%`"**不是探针的成本**，那是拿 `5` 迭代的 `SMOKE5CAP`（`45,706`）
去比 `300` 迭代的 `TRAIN_s0`（`50,221`）得到的，两条跑长度不同、当天机器状态也不同。
本次每一条曲线都长一个样：iteration 1 冲到 `50.2k--50.6k`，随后稳在 `44.8k--45.3k`，
**所以不同长度的两条跑不能直接比均值**。正确说法是：
**这道门（旧的和新的）在这条 lane 上的吞吐代价都 ≤ `1%`；新门用同样的代价换到了 `9` 类覆盖而不是 `1` 类。**

#### 四、零回归

- `tests/test_mjlab_lane_capacity_gate.py`：`17` 条全绿（pod1 `/workspace/mjlab_venv`）。
  覆盖：位序与 `mujoco_warp.OverflowType` 逐位对齐、`BROADPHASE` 能按名字解回来、
  余量分母、`nefc == njmax` 不算溢出、零样本不给 headroom、五种 verdict、
  日志扫描不把我们自己的 `"overflow_mask"` 键当成引擎告警、
  **`EPA horizon` 那行没有 "overflow" 字样也要能扫到**、落卡 uuid 对账。
- 健康配置端到端仍 `PASS_NO_OVERFLOW`（上表），`4096 x 12` 训练跑完曲线正常、退出 `0`。
- 该模块在本机（py3.8、无 mujoco）自动 skip，不影响 host 测试集。

#### 五、收据（pod1 `/workspace/mjlab_lane/T1T8/`）

| 文件 | 是什么 |
| --- | --- |
| `a3_train_ppo_BEFORE.py` | 改动前的 shipped 版，变异测试的"修复前"一栏就是它跑的 |
| `BEFORE.status` / `BEFORE_m*.out` / `BEFORE_m*.json` | 修复前三条变异的退出码、引擎告警行数、收据 |
| `AFTER.status` / `AFTER_m*.out` / `AFTER_m*.json` | 修复后六条（三条 T8 + `njmax70` + 健康对照 + 门关掉对照） |
| `FPSCLEAN.status` / `FPST*.json`、`FPSTIE.status` / `FPSC*.json` | 配对吞吐，含每次跑前/跑后 GPU2 上的他人进程数 |
| `WIRE.json` | 接线冒烟：`capacity_samples_stepped = 960`、`overflow_mask = 0`、uuid 与 `nvidia-smi` 对上 |
| `EVALGATE.json` / `EVALMUT.json` / `EVALOFF.json` | eval 路径三态：设门通过 / 设门开火（点名 `BROADPHASE`，退出 `1`） / 明写 `NOT_GATED` |
| `test_mjlab_lane_capacity_gate.py` | 与 repo 同一份的单测 |
| `run_gpu2_smoke.sh.pre-T7` / `audit.sh.pre-T7` | 两个 pod 脚本改前的备份 |

#### 六、这一节不代签什么

只修了**门**，没有改任何容量数值：`njmax=572` / `nconmax=128` 原封不动，§9.2.3 的
"余量 `~3--8x`、随场景与步数变、且未收敛"照旧成立。§9.2.4 E 里的
E4（普查没跑到收敛）、E5（策略驱动的构型分布没普查）、E6（恢复系数——已由 §9.2.5 单独处理）
这三条本节没碰。**能改口的只有 E1（`ncollision` 现在训练期就采样并进收据）、
E2（深压 `nconmax` 时门现在比 CUDA fault 先到）、E8（失败路径现在有 telemetry）；
E3（`EPA_HORIZON` 历史 run 无证据）从"真静默"改成"会打印但不含 overflow 字样，
历史 `grep` 一样覆盖不到"——结论不变，仍然只能靠新跑补测。**

### 9.2.8 汇报口径：把"加权奖励项"当接触率的那条链，改成会拒绝的代码（T11 落地，2026-08-06 实测，pod1 GPU2）

**人话一句**：这条 lane 之前把自己**报坏了**。被反复引用的那句"`touch 4e-5 → 0.21`"里，
`touch` 根本不是"碰到球的比例"，而是**加权奖励项** `4.0 * exp(-(d/0.15)^2)`，上限就是 `4.0`；
`0.21` 折成核均值是 `5.25%`（`0.21/4.0`；§9.2.4 那里写的 `5.4%` 是笔误，已就地更正）。真正问"球拍到底有没有碰到球"的那个二值指标当时只在 eval 打开，
它的答案是**零策略 `0.12%` → 训练后 `49.2%` / `97.8%`**。同一批策略，一个口径像"几乎没学会"，
另一个口径是"比什么都不做强 `400--800` 倍"。这一轮把口径本身变成代码。

> **为什么裁定说这条比容量门更要紧**：容量门错了，是让**坏数据**看起来像好数据；口径错了，是让
> **好结果**看起来像坏结果，然后有人据此砍掉一条其实在学的配方。前者骗审计，后者骗决策。

**今天用现役代码把这件事复现了一遍**（`4096 x 300` 迭代，seed `0`/`1`，两条全新 run）：

| 同样这两条 run，两种报法 | seed 0 | seed 1 |
| --- | ---: | ---: |
| **旧报法**：`touch` 加权项（上限 `4.0`） | `0.003 → 0.252` | `0.004 → 0.189` |
| 同一个数折成核均值 | `0.1% → 6.3%` | `0.1% → 4.7%` |
| **新报法**：二值"这一局摸到球了吗"（训练曲线） | `0.44% → 74.1%` | `0.49% → 64.6%` |
| **新报法**：确定性 eval，对零策略 `0.14%` | **`80.7%`（`570` 倍）** | **`56.0%`（`395` 倍）** |

**同一批 run，"0.25/4.0" 和 "80.7%" 说的是同一件事。** 这就是 T11 要修的东西。

改的是 `hope_training/whole_body_tracking/mjlab_lane/a3_train_ppo.py` 一个文件，
新单测 `tests/test_mjlab_lane_reporting_gate.py`。

#### 一、改了什么（逐条对上 T11 的 (a)--(d)）

| T11 | 改法 | 人话 |
| --- | --- | --- |
| **(a)** | `reward_terms_mean` 里 `reach`/`touch` 改名 `reach_term_weighted`/`touch_term_weighted`；**并且**同时新增 `reward_terms_max_possible`（`2.0`/`4.0`）与 `reward_kernel_mean`（除掉权重后的核均值），外加一句 `reward_terms_note` 自陈"这不是概率、要接触率看二值那项" | 名字里写着"带权"、旁边写着"上限多少"、再写一句"你要的不是我"——三层都绕过去才可能再错读。只改名不够：`0.25` 一样能被当成 `25%` |
| **(b)** | `count_contacts` **训练路径默认开**（原来只有 eval 开）；`fraction_of_episodes_with_a_racket_touch` 逐迭代进 `.jsonl`、进控制台行（`touchEp=`）、进 `learning.binary_contact_rate`，run 结束再打一行 `[HEADLINE]` | 训练曲线上第一次有了"这一局到底摸没摸到球"这个有物理意义的数 |
| **(c)** | 新增 `--report`：**只**输出"零策略对照 + 二值接触率 + run 间散布"这一种句式；证据不够就**退出 `2` 并点名**，不降级成弱一点的说法 | 口径从"约定"变成"会拒绝的门" |
| **(d)** | `--analyze` 只给一份文件，从"退出 `0` 打出零宽度的带"改成**退出 `2`**；带里新增二值接触率一项（没测到的迭代写 `null`，不按 `0` 平均） | 单 seed 单点从此报不出来 |

顺手修的两个**会造假象**的坑（都属于"记录与阻断必须同批"）：

1. **`_spearman` 把全平的曲线算成 `+1.0`。** 原实现 `argsort(argsort(y))` 处理并列时按下标排成
   `0,1,2,...`，于是一条**完全不动**的曲线被算成"单调上升"。二值接触率早期正好长期全 `0`——
   这个 bug 会给一个一次球都没碰到的策略打出"在上升"。改成并列取平均秩：常数序列秩方差为 `0`，
   判 `nan`（"测不出趋势"），不判 `+1.0`。
2. **"没有分母"被写成 `0.0`。** 二值接触率的分母是"这一窗口内**结束**的 episode 数"，原来除的是
   `max(episodes, 1)`，所以一个没有任何 episode 结束的窗口打印 `0.0`——与"真的一次没碰到"
   无法区分。现在这种情况写 `null` + `reason: NO_EPISODES_FINISHED`；探针被关掉写 `null` +
   `reason: CONTACT_PROBE_OFF`；**真的测出来的 `0` 仍然是 `0` 且 `measured: true`**。
   这与容量门"零样本不许判 PASS"是同一类修法。

`--report` 的拒绝规则（每条有代号，变异测试断言的是**哪一条**开火，不是"有东西开火了"）：

| 代号 | 什么时候开火 |
| --- | --- |
| `SINGLE_SEED_NOT_EVIDENCE` | 给的 run 少于 `2` 条 |
| `NO_ZERO_POLICY_BASELINE` | 没给 `--report-zero-policy` |
| `BASELINE_IS_NOT_A_ZERO_POLICY_RUN` | 拿 ckpt eval 冒充"什么都不做"的对照 |
| `BASELINE_HAS_NO_BINARY_CONTACT_RATE` / `BASELINE_DID_NOT_COMPLETE_OR_PASS` | 对照收据没有二值接触率，或它自己没跑完 / 没过容量门 |
| `NO_BINARY_CONTACT_RATE` | 某条 run 的训练曲线上没有二值接触率（旧收据，或 `--no-contact-probe`） |
| `RUN_DID_NOT_COMPLETE` / `RUN_HAS_NO_CAPACITY_PASS` | 某条 run 没跑完，或容量门不是 `PASS_NO_OVERFLOW` |
| `EVAL_IS_NOT_A_CHECKPOINT_RUN` / `EVAL_HAS_NO_BINARY_CONTACT_RATE` / `EVAL_DID_NOT_COMPLETE_OR_PASS` / `EVAL_COUNT_DOES_NOT_MATCH_RUNS` | 可选的 `--report-eval`（确定性评估）那一格填错 |

#### 二、变异测试：修前 / 修后（pod1 GPU2，同一天，同一份 `a3_court_env.py`）

"修前"跑的是 `T11/a3_train_ppo_T11BEFORE.py`，与本次改动前的 shipped 版**逐字节相同**
（`md5 417ce53b496140423f8cbf64335f10d1`）。

| 变异 | 修前 | 修后 |
| --- | --- | --- |
| **一条训练收据怎么说自己的单位**（`--smoke`，`64` 世界 `x 3` 迭代） | 退出 `0`；键叫 `reach`/`touch`；**没有** `reward_terms_max_possible`、**没有** `reward_kernel_mean`、训练收据里**完全没有** `contact` 段 | 退出 `0`；键叫 `reach_term_weighted`/`touch_term_weighted`；上限、核均值、"不是概率"那句都在；`contact` 段在且 `measured: true`（`11` 个 episode 结束、`0` 次触球——**测出来的 `0`**） |
| **给 `--analyze` 一份文件** | 退出 **`0`**，写出 `n_seeds = 1`、`rel_spread_max_pct = 0.0` 的"带"——读起来像完美复现 | 退出 **`2`**，`[WARN][BAND REFUSED]`，不写文件 |
| **`--report` 只给一条 run** | 子命令**当时不存在**（`unrecognized arguments`，退出 `2`） | 退出 **`2`**，`SINGLE_SEED_NOT_EVIDENCE` |
| **不给零策略对照** | 同上 | 退出 **`2`**，`NO_ZERO_POLICY_BASELINE` |
| **拿 ckpt eval 冒充零策略对照** | 同上 | 退出 **`2`**，`BASELINE_IS_NOT_A_ZERO_POLICY_RUN` |
| **训练时 `--no-contact-probe`，再拿它汇报** | 同上；且那种 run 的收据里连 `contact` 段都没有，读的人只剩加权项 | 训练退出 `0` 但收据写 `fraction: null` + `reason: CONTACT_PROBE_OFF`，`[WARN][CONTACT RATE NOT MEASURED]` 进摘要；`--report` 退出 **`2`**，`NO_BINARY_CONTACT_RATE` |
| **eval 窗口短到没有 episode 结束**（`--eval zero --nworld 512 --eval-steps 5`） | 退出 `0`，`fraction_of_episodes_with_a_racket_touch = 0.0`，无任何 WARN——与"真的一次没碰到"无法区分 | 退出 `0`，`fraction: null` + `reason: NO_EPISODES_FINISHED` + `[WARN][CONTACT RATE NOT MEASURED (eval)]` |
| **拿 08-05 那两份历史收据汇报**（当初被引用的正是它们） | 无从拒绝 | 退出 **`2`**，逐条点名：两条 run 各 `RUN_DID_NOT_COMPLETE` + `RUN_HAS_NO_CAPACITY_PASS` + `NO_BINARY_CONTACT_RATE`，对照 `EVALC_zero.json` 再加 `BASELINE_DID_NOT_COMPLETE_OR_PASS`（旧收据没有 `status`，训练路径也没有二值接触率） |
| **拿一条真开过容量门的 run 汇报**（真收据 `T1T8/AFTER_m1_blind_nc10.json`） | 无从拒绝 | 退出 **`2`**，`RUN_DID_NOT_COMPLETE` + `RUN_HAS_NO_CAPACITY_PASS` + `NO_BINARY_CONTACT_RATE` |
| **两条 run 只配一份 ckpt eval** | 同上 | 退出 **`2`**，`EVAL_COUNT_DOES_NOT_MATCH_RUNS` |
| **把零策略 eval 塞进"训练后"那一格** | 同上 | 退出 **`2`**，`EVAL_IS_NOT_A_CHECKPOINT_RUN` |
| **健康路径**（两条 run + 零策略对照 + 两份 ckpt eval） | —— | 退出 **`0`**，写出下面第三节那一句 |

#### 三、这一轮的实测：现在唯一允许的那句话长什么样

两条全新训练（`4096` 世界 `x 300` 迭代，seed `0`/`1`，接触探针 ON，容量门 ON），
三条评估（`4096` 世界 `x 750` 步，各 `20,480` 个结束的 episode）：

```
[REPORT] binary per-episode racket-ball contact rate (deterministic eval):
         zero policy 0.14%  ->  trained 80.7% / 56.0%  (band 56.0--80.7% over 2 runs)
[REPORT] 人话: 零策略基本碰不到球, 训练后每局摸到球的比例见上;
         括号里是 run 之间的散布, 单次跑不作数.
[REPORT] the weighted `touch_term_weighted` reward term (ceiling 4.0) is NOT a
         contact rate and is printed here only for context: s0=0.252, s1=0.189
```

| 量 | seed 0 | seed 1 | 备注 |
| --- | ---: | ---: | --- |
| 二值接触率，训练曲线首/末十分位 | `0.44% → 74.08%`（峰 `80.75%`） | `0.49% → 64.64%`（峰 `73.46%`） | 带探索噪声，是保守数 |
| 二值接触率，确定性 eval | **`80.72%`** | **`56.00%`** | 对零策略 `0.1416%` 是 `570` / `395` 倍 |
| run 间散布（eval / 训练曲线） | `1.44x` / `1.15x` | | 单点仍然不作数 |
| `touch_term_weighted`（上限 `4.0`） | `0.0030 → 0.2520` | `0.0039 → 0.1893` | **这就是当初被当成百分比的那个数** |
| `reach_term_weighted`（上限 `2.0`） | `0.6921 → 1.0079` | `0.7009 → 0.9721` | |
| 每局最小拍球距离 | `0.42 → 0.076 m` | `0.41 → 0.080 m` | |
| 二值接触率 spearman vs 迭代 | `0.758` | `0.715` | 并列已按平均秩处理 |
| 容量 | `nefc` 峰 `86`，`PASS_NO_OVERFLOW` | `nefc` 峰 `88`，`PASS_NO_OVERFLOW` | 引擎 overflow printf `0` 行；落卡 uuid `473a79f3…` 与 `nvidia-smi` 对得上 |

`--analyze` 的两条 run 带：`mean_episode_return` 的 `rel_spread_max_pct = 5.66%`、
`learning_gain_vs_seed_spread = 18.2`（涨幅是 seed 间散布的 `18` 倍，所以"确实在学"这句成立），
二值接触率末十分位 `64.6%--74.1%`。

**注意两个数不要混**：训练期二值接触率（带探索噪声、按"这一迭代内结束的 episode"算）和
eval 二值接触率（确定性策略、`750` 步窗口）是两个测量。收据里分开放，`--report` 的句子里
写明了用的是哪一个（`headline_measurement` 字段），倍数与散布也各自带后缀，不共用一个裸键名。

**与 08-05 那两条历史 run 的关系**：那次 eval 是 `49.2%` / `97.8%`，今天是 `80.7%` / `56.0%`。
**方向一致、量级一致、seed 间散布依旧大**（历史 `2.0x`，今天 `1.44x`）。这正是"要报就报带"
的理由，也是为什么 `--report` 把"至少两条 run"写成硬门。

#### 四、吞吐：这道探针**真的要钱**，如实记

配对实测（`--nworld 4096 --iterations 12 --seed 0`，同一张 GPU2 背靠背交替，去掉 iteration 0
后取中位数；六条收据里跑前/跑后 GPU2 上他人进程数全是 `0`）：

| 配置 | 三次中位 env-step/s | 中位 | 相对 |
| --- | --- | ---: | ---: |
| 接触探针 **ON**（现在的默认） | `38,904` / `38,627` / `38,947` | `38,904` | **`-13.0%`** |
| 接触探针 OFF（`--no-contact-probe`） | `44,735` / `45,044` / `44,603` | `44,735` | 基准 |

**这和容量门那 `≤1%` 不是一个量级，不要混为一谈。** 原因是结构性的：容量探针每 substep 只扫
`nworld = 4096` 个 int；接触探针必须扫**整个预分配接触数组**（`4096 x nconmax 128 = 524,288` 行），
而且**必须每个 physics substep 扫一次**——球拍碰球只持续 `1--2` 个 substep，
按 env-step（`20` 个 substep）采样会漏掉大部分接触。

优化过一版（把"逐个球拍 geom 比一遍再 `any()`"换成一张 geom 分类查找表，kernel 数减少），
**实测没有区别**：优化前 `38,816`，优化后 `38,904`；同期两次 OFF 是 `44,987` / `44,735`，
即这台机器同配置的噪声约 `0.6%`。两组都留在 `T11/probe_v1/` 与 `T11/THR_*`。
**如实写：`13%` 是这条探针的固有成本，不是实现没写好。**

值不值：**值**。没有二值接触率的训练曲线，是一条只能靠加权奖励项去猜的曲线，而那正是这次要修的病。
`--no-contact-probe` 保留给纯计时跑，且那种 run 的收据会自己说"我没测过接触"，`--report` 会拒绝它。

#### 五、零回归

- `tests/test_mjlab_lane_reporting_gate.py`（新，`45` 条）+ `tests/test_mjlab_lane_capacity_gate.py`
  （原有 `17` 条）在 pod1 `/workspace/mjlab_venv` 上 **`62 passed`**。新单测覆盖：加权项上限与核均值
  换算（含 `0.21/4.0 = 5.25%` 这条换算本身）、"没有分母"写 `null`、探针关掉写 `null`、
  **真的测出来的 `0` 仍算测过**、带里跳过未测的 run 而不是按 `0` 平均、全平曲线 spearman 判 `nan`、
  十一条拒绝规则各一条、端到端 `--report` 的句子/倍数/"headline 的倍数必须与 headline 同源"。
- 该模块在本机（py3.8、无 mujoco）自动 skip，host 测试集不受影响。
- 全仓 `grep` 确认：除 `a3_train_ppo.py` 与新单测外，**没有任何脚本或文档读
  `reward_terms_mean.reach/touch` 这两个旧键名**，改名没有下游破坏。
- 老功能不变：两条 `300` 迭代 run 均 `status: completed`、`PASS_NO_OVERFLOW`、
  `warp_overflow_printf_lines = 0`、落卡 uuid 与 `nvidia-smi` 对账通过、`nonfinite_state` 终止 `0`。

#### 六、收据（pod1 `/workspace/mjlab_lane/T11/`）

| 文件 | 是什么 |
| --- | --- |
| `a3_train_ppo_T11BEFORE.py` | 改动前的 shipped 版（`md5 417ce53b…`），变异表"修前"一栏是它跑的 |
| `THR_on_*.json` / `THR_off_*.json` | 配对吞吐六条，收据自带跑前/跑后 GPU2 上他人进程数 |
| `probe_v1/THR_*.json` | 探针第一版的同一组配对（"优化没带来区别"这句是量出来的） |
| `TRAIN_s0.json/.jsonl`、`TRAIN_s1.json/.jsonl` | 两条 `4096 x 300` 训练，逐迭代带二值接触率 |
| `EVAL_zero.json`、`EVAL_ckpt_s0.json`、`EVAL_ckpt_s1.json` | 零策略对照与两条 ckpt 的确定性评估 |
| `REPORT.json` / `REPORT_curve_only.json` | `--report` 输出（带 eval 一份、不带 eval 一份） |
| `BAND_2seed_t11.json` | `--analyze` 的两条 run 带，含二值接触率带 |
| `mut/MUT.status`、`mut/MUT_FINAL.status`、`mut/M*.json/.log` | 变异 battery 的退出码与逐条收据 |
| `test_mjlab_lane_reporting_gate.py` | 与 repo 同一份的新单测 |
| `t11_run2.sh` / `t11_mut.sh` / `t11_collect.py` | 这一轮的跑法、变异脚本、取数脚本 |

一处如实交代：两条 `300` 迭代 run 之间，文件差了一行**已死变量的删除**（`_probe_contacts`
换成查找表之后，那个按 geom 列表建的 tensor 不再被用到，在 `TRAIN_s0` 跑完后删掉）。
物理与统计路径逐字相同；`TRAIN_s1` 与全部评估、全部 `--report`/`--analyze` 变异
都是用**与本次提交完全一致**的文件跑的。

#### 七、这一节不代签什么

- **只改了口径，没改物理，也没改容量数值**：`njmax=572` / `nconmax=128`、恢复系数带、普查结论
  全部原样。§9.2.7 的 `2.07x` 行余量与 §9.2.5 的标定门不受影响。
- **不代签"这条 lane 可以放行"**：它仍然是 court/ready/reach-touch 任务，没有 measured teacher、
  没有完整 reward 层级、没有 cross-engine parity（§9.2.2 末尾那段照旧）。
- **二值接触率不等于"回球"**：它只回答"球拍和球有没有碰上"，不回答上不上台、过不过网、旋转对不对。
  真正的回球指标要等 §9.2 的 reward 层级落地。
- **两条 run 不是"复现"的终点**：`--report` 的"至少两条"是**下限**不是**标准**。这条 lane 已知
  同配置能有 `1.4--3` 倍散布，认真的结论应当要更多 seed。
- **`13%` 的探针成本没有被优化掉**，只是被量准了并写进了默认值的理由里。要更便宜的写法，
  得改成引擎侧 kernel（warp 里做一次归约），那是另一件事。

### 9.2.9 MuJoCo GPU lane 和 Isaac A211/C211 到底差什么：一张逐项活值对齐台账（2026-08-06 实测，pod1 GPU2）

**人话（先看这四句）**

1. **两条车道现在问的不是同一道题**，`17` 条对齐轴里 `10` 条是**要紧的差异**、`5` 条真的对上了、
   `2` 条是有理由的差异。差异不在"参数没调"，在**观测长什么样、动作是什么意思、什么算这一局结束、
   奖励在付什么钱**这四件事上。
2. 所以 **mjlab lane 的曲线是"这条车道内部"的陈述**。它可以说"球拍碰到球的比例涨了"，
   不能说"ActionBall 学会了"，更不能跟 Isaac 的曲线并排读。
3. **本轮量到一件必须马上说的事**：这条 lane 的机器人**在第 3 次 PPO 更新之后，几乎每一局都在碰桌子**
   （`0% -> 100%`，两个 seed 独立复现；按 geom 名点出来,主犯是**球拍本身** `robot/right_racket_collision`，
   `4262/4550` 行）。同一件事在 Isaac 的 ActionBall 里是**硬终止**（`robot_hit_table`，跟摔倒同级，
   还挂 `-6` 安全罚）。**在此之前没有任何东西在看这个通道。**
4. Franco 那条「MuJoCo 设置要继承智元的 MuJoCo，不是 mjlab 默认」的规矩**今天仍然成立**，
   当场复验：`92` 组字段匹配、`1` 条不匹配且是已登记的具名偏离、`0` 条未登记。

**这不是一张手写对照表。** 每一行两侧都是**活值**：Isaac 侧 AST 从源码取值 / host-load 无依赖的
trainability 叶子 / 直接解析智元 MJCF / 读 `cfg/algo/ppo.yaml`；mjlab 侧直接 import 本车道模块读常量。
每一行的**裁定**（表里写死的那个词）会跟**当场量出来的裁定**对账，**两个方向都对**——
说"对齐了"其实没对齐会炸，说"差着"其实已经对上了也会炸（后者才是会烂掉的那种：它让一个已经补好的
洞看起来还开着，然后没人再去读）。写这一节时第一次跑就被它抓到一条我自己写错的裁定
（`control_rate` 我记成"差着"，实测两边 policy dt 都是 `0.02 s`，是真对齐）。

新增：`hope_training/whole_body_tracking/mjlab_lane/isaac_alignment.py`、
`hope_training/whole_body_tracking/mjlab_lane/tests/test_isaac_alignment.py`。

#### 一、逐项差异表（`17` 轴，活值，`ledger_sha256=c977c47e…9a8a`）

| 轴 | Isaac A211/C211 | mjlab GPU lane | 裁定 | 要不要紧 |
| --- | --- | --- | --- | --- |
| actor 观测 ABI | `211` 维 `17` 行（含 measured teacher `31+31+9+9`、任务包 `9`、两个时钟、`task_valid`） | `114` 维 `10` 行（本体感 + 球的相对位置） | **要紧** | 输入不同 = 学到的映射不可互相解释；这边连 mimic 层都不存在 |
| critic 观测 ABI | `319` 维特权（`command 62`、`body_pos 42`、`body_ori 84`、两个 anchor） | 与 actor 同一份 `114`（对称） | **要紧** | 非对称 critic 改价值估计方差，同预算曲线不可比 |
| 动作解码 | 逐关节 `0.25*力矩上限/Kp`，`0.0375`（头 / 腕俯仰）到 `0.6875`（髋偏航 / 髋俯仰） | **默认 flat `0.25` rad 全关节**（vendor 模式已实现但不是默认） | **要紧** | 不是缩放是**重新加权哪个关节动得动**；实验史三层核对：机制默认 `flat`、pod 上 `103` 条收据 `flat` / `2` 条 `vendor` |
| PD 增益 Kp/Kd | 活的 `stiffness`/`damping` | `VENDOR_KP`/`VENDOR_KD` 手抄件 | **对齐**（逐关节 `31/31`） | — |
| 力矩上限 | `effort_limit_sim` | 智元 MJCF `<motor ctrlrange>` | **对齐**（逐关节 `31/31`） | — |
| 终止 union | `9` 条：`time_out` / `anchor_pos` / `anchor_ori` / `ee_body_pos`(只脚) / `base_fell_tilt` / `base_too_low` / **`robot_hit_table`** / `joint_qdes_forbidden` / `joint_actual_forbidden`(`terminate=False`) | `3` 条终止（`fall_height` / `fall_tilt` / `nonfinite_state`）+ 超时截断 | **要紧** | 终止 union 决定回报支撑集（CaT）；撞桌那条见下文第三节 |
| 摔倒阈值（两边都有的那两条） | `limit_angle=0.7 rad`（`40.1°`）、`minimum_height=0.5 m` | `max_tilt_proj_g=-0.5`（`60.0°`）、`min_pelvis_z=0.70 m` | **要紧** | 这边对倾角更宽容、对下蹲更严格，早期终止率不可比 |
| reward 组 | 完整 ActionBall 层级（balance/mimic/strike/target/outcome），锚点项 `base_position`/`death_penalty`/`qdes_limit_barrier`/`joint_limit`/`c225_strike_ball_paddle_center_proximity`/`virtual_landing` 全在 | `10` 项；按 Isaac 词汇分组后 **mimic / strike / target / outcome 四组覆盖 = `0` 项**，两边项名交集 `0` | **要紧** | 这边的 `reach`/`touch` 是**拍球距离整形**，不是击球质量、更不是上台 |
| episode 结构 | `500` tick（`10 s`）；开局 `5--25` tick WAIT 遮住任务行，揭示后至少 `200` tick 有效 | `150` tick（`3 s`）；**没有 WAIT / 揭示** | **要紧** | reveal bridge 是四格第二根轴（§5.6.2d），这边测不到；集长差 `3.3` 倍 |
| 控制频率 | `sim.dt=0.005 x decimation 4 = 0.02 s` | `timestep=0.001 x decimation 20 = 0.02 s` | **对齐** | 物理步长不同（`5 ms` vs `1 ms`），后者是智元显式值，记在行里不按要紧记 |
| 观测噪声与 DR | 四格 = `{corruption off, on}`，plant 全冻结；噪声三通道 `base_ang_vel_body ±0.2` / `joint_pos ±0.01` / `joint_vel ±0.5` | 观测**无噪声**，但复位时加了 `joint ±0.05 rad`、`root xy ±0.02 m`、`yaw ±0.05 rad` | **要紧** | 它既不是 A0/C0 也不是 A1/C1：多了一份四格里**没有**的复位随机化 |
| 题目分布 | **一道固定题**（`initial_center_single_question`、`all_32_domain_levels_exact_zero`、profile 中心点） | 每次发球从 `ServeConfig.reachable_returner()` 的均匀盒子重采（pos/vel 各 3 维） | **要紧** | 固定题 vs 分布题是两种可学性，混着读会得出相反结论 |
| plant 继承智元 MJCF | 智元 `<option>` 只显式写 `timestep/gravity/noslip_*` | 逐字段显式写同一份；`92` 组匹配 / `1` 条已登记偏离 / `0` 未登记 | **对齐** | `noslip_iterations=3` 带不过去，mujoco-warp 无 noslip pass（具名偏离） |
| 球的接触模型 | 现役 C211 走**解析**路径（`physical_ball=false`） | **真接触**：`ball_solref=(-902.5,-1921.42)`、`solimp` 常阻抗、`solreffriction`、球拍 `e=0.654` 常数、网 `e=0.10` 假设 | **要紧** | 一边引擎解、一边解析给；命中率/上台率不可比，且球拍与网都还没标定（具名缺口） |
| PPO 超参 | 网络 `[512,256,128]` elu、`init_noise_std=1.0`、lr `1e-3` adaptive、KL `0.01`、epochs `5`、mb `4`、gamma `.99`、lam `.95`、`max_grad_norm 1.0`；**熵系数 `0.01`** | 以上全同；**熵系数 `0.002`** | 有理由的差异 | `31` 维动作下 rsl-rl 的逐维熵奖励把 std 从 `1.00` 推到 `1.16`，实测后减半（理由写在 `build_agent_cfg`） |
| geometry 来源 | 仓库 `tasks/table_tennis/geometry.py` | 解析顺序：环境变量 → **自己旁边的同名拷贝** → 仓库 | **对齐**（在仓库 checkout 里） | 见下文第四节：pod 部署形态下这一行会变红 |
| 确定性层级 | Tier-1 exact（question/curriculum/receipt/ABI/action identity） | mujoco-warp 无 CPU 回退、实测非确定 | 有理由的差异 | 跨引擎**只能**统计对拍；见第五节 |

**顺带纠正一条文档里的手抄错误**：`TaskCfg.action_scale_mode` 的 docstring 原写 vendor 解码上界是
`0.647 rad`（腰偏航 `220/85`）。逐关节活值算出来是 **`0.6875`**（髋偏航 / 髋俯仰 `220/80`）。
已改，并在 docstring 里注明是活值读出来纠正的。**这就是"手抄件默认已漂"的一个当场标本。**

#### 二、(b) 「继承智元 MuJoCo」这条规矩今天还成立——当场复验

`pod1 GPU2`，`a3_plant_env.py --verify --nworld 64`：

```
matched groups : 92
mismatches     : 1 (0 not covered by a named deviation)
  opt.noslip_iterations  mjlab=0  mjcf=3   registered_deviation=true
dof_damping      : mjcf 31 个非零，mjlab 31 个非零，sum 38.3 == 38.3
dof_frictionloss : mjcf 31 个非零，mjlab 31 个非零，sum 25.57215 == 25.57215
```

比 §9.2.1 那张表更新的一点：**`dof_damping` 与 `dof_frictionloss` 现在是带过去了的**
（§9.2.1 记的是"mjlab 默认 `0.0`"）。所以在这两项真实物理项上，**MuJoCo 车道比 Isaac 更接近智元**——
Isaac 侧根本没有 `dof_damping`，而 `frictionloss` 被搬成了一个**未标定**的 PhysX 无量纲 `friction` 系数
（`agibot_a3.py` 自己写着这句话）。这是一条跨引擎不可比因素，方向是"MuJoCo 更对"，不是缺陷。

#### 三、(c) 本轮真补上的：撞桌通道，从"没人看"变成"测得到 + 会拒绝"

**先说量到的事实。** `512` 世界、`12` 次 PPO 更新、`seed 0`：每局至少碰一次桌子的比例

```
iter  0    1      2      3 ... 11
      0.0  0.037  0.633  1.0 ... 0.978        peak = 1.000
最后一格窗口: 45 局里 44 局碰到，接触子步 106,255
```

`seed 1` 独立复现 `peak = 1.000`。**按 geom 名独立点名**（把接触数组拉回 host 数名字，不信探针自己的分类表）：

```
robot/right_racket_collision      4262   <-- 主犯:球拍本身
robot/left_elbow_collision         125
robot/left_wrist_roll_collision_1  123
robot/right_hip_roll_collision      15
...
```

**人话**：这条 lane 的 `reach`(权重 `2.0`) / `touch`(权重 `4.0`) 在付钱让球拍靠近球，而球在桌子上方，
于是策略学会了**把球拍搁在桌面上**。同一件事在 Isaac 的 ActionBall 里第一时间终止（`robot_hit_table`，
跟摔倒同级，racket blade OBB 明确在护栏几何里）。这正是 Franco 2026-08-06 预判的那批坑
（"build_1 之后都会遇到"）——**它已经在 MuJoCo 侧发生了，只是没人在看。**

**改法（记录 + 阻断同一批，不改训练分布）：**

- 接触探针**同一趟**里多数一个通道：`table geom` 对 `robot/` 前缀 geom（排除 `robot/floor`）的接触。
  四个逐元素算子，不加同步。
- 收据逐迭代写 `robot_table` 块、run 级写 `learning.robot_table_contact`（报 **peak** 不报 last：
  问题是"这条曲线里有没有 Isaac 判死的行为"，一格就够污染）。**没测到报 `null` 不报 `0`。**
- `--report` 新增两条拒绝：`ROBOT_LEANED_ON_THE_TABLE`（非零就拒）与
  `ROBOT_TABLE_CONTACT_NOT_MEASURED`（没这块也拒——"没测"不许长得像"是零"）。
- **没有**装成硬终止。装了就改训练分布，那是发车决定不是 review 改动；台账里如实写着
  "这是证据+阻断，不是护栏"，而且这句话**是被机器检的**（见下）。

同批把 `tests/test_mjlab_lane_reporting_gate.py` 的 `_good_run()` 收据形状一起改了——
门和它判的那个形状必须同批动，否则门在判一个没人写的形状。

**其余本轮做到的**（都是"让两边能被同一句话描述"，不是假装对齐）：

- `_compute_obs` 改成**由 `OBS_LAYOUT` 逐行拼**，名字变成承重件；GPU 上逐元素证明是 no-op
  （同进程同一份 `_state()` 快照，新旧两式 `torch.equal` 为真，`max abs diff = 0`）。
- `VENDOR_KP/KD` 的匹配规则抽成纯函数 `vendor_pd_for_joint_names()`，**运行时与台账走同一份实现**，
  不再各写一遍。
- vendor 动作解码逐关节对上活的 Isaac 表（`31/31`），所以"把默认从 flat 改成 vendor"从此是一个
  **已验证**的 flag，不是一次赌博。

#### 四、变异测试：每一条都做成"粗一个档次的检查会照样通过"

全部在子进程里跑，树是临时拷贝，本仓不动。每条**先断言粗检查确实过得去**，再断言台账当场红。

| 变异 | 为什么粗检查抓不到 | 结果 |
| --- | --- | --- |
| 交换两个关节的 `Kp`（`shoulder_yaw 30` ↔ `wrist_pitch 20`，各匹配 2 个关节） | `31` 个数的**和不变、排序后多重集不变、个数不变**（测试里逐条断言过） | `pd_gains` 变红 ✅ |
| 交换两条 `<motor>` 的 `ctrlrange`（`shoulder_yaw 24` ↔ `wrist_pitch 6`） | 同上，和与多重集都不变 | `effort_limits` 变红 ✅ |
| 改 Isaac `self.sim.dt` `0.005 -> 0.004` | 这个值在 `__post_init__` 里，**只扫模块级赋值的读法看不见它**（测试里断言 `"decimation" not in _module_consts`） | `control_rate` 变红 ✅ |
| 改智元 MJCF `<option timestep>` `0.001 -> 0.002` | — | `vendor_plant_inheritance` 变红 ✅ |
| 车道旁的 `geometry.py` 拷贝**只追加一行注释**（语义完全不变） | 只比"车道今天读到的那几个值"的检查会全过 | `geometry_provenance` 变红 ✅ |
| Isaac 新增一条终止项 | — | 枚举门开火：`invented_guard` 未分类，点名 `ISAAC_TO_MJLAB_TERMINATION` ✅ |
| 改掉 Isaac 唯一一处 strike 锚点项名 | 选的是**父类没有影子**的那一项；换成有影子的 `virtual_landing` 时台账**正确地不报警**（第一版测试就是这么写错的，被自己的断言抓住） | `reward_surface` 锚点门开火 ✅ |
| 把 `--report` 里 `ROBOT_LEANED_ON_THE_TABLE` 这条拒绝改名 | 计数器、收据字段、docstring **全都还在**——典型的"计数器没人读"形状 | 台账查的是**阻断**不是测量，变红 ✅ |
| 给 `TaskCfg` 加一个没分类的旋钮 / 删掉一个已分类的 | — | 枚举门两个方向都开火 ✅ |

另外两条不需要变异的硬规则：`assert_cross_engine_claim` **无条件**拒绝 `bitwise_parity`
（哪怕台账一条 blocking 都没有），以及台账有 blocking 时拒绝 `cross_engine_comparable`。

#### 五、(e) 跨引擎只能统计对拍——这条写进了代码

§9.2.0 实测：mujoco-warp 无 CPU 回退，**连没有接触的 `pendula` 都发散**（`1007/1024` 世界，
`max abs dqpos 1.4e-05`）。所以**任何"逐位一致"的跨引擎验收都是错的标准，不是更严的标准**。
`assert_cross_engine_claim(ledger, CLAIM_BITWISE_PARITY)` 永远抛，理由字符串里写着为什么。

**注意区分**：本节里那条 `_compute_obs` 重构的 no-op 证明**是允许逐位的**——它在**同一个进程、
同一份快照**里比两个表达式，中间不走物理。跨**引擎**才是只能统计。

#### 六、(d) 补不动的，以及为什么

- **actor `211` / critic `319` 观测 ABI**：需要 measured teacher artifact（`teacher_joint_pos/vel`、
  三组 racket-site heading）与在线 question solver；critic 还需要 Isaac 的 motion command manager
  才有 `command 62` / `body_pos 42` / `body_ori 84`。这不是本车道能单独造的，造一个形状对、
  内容是零填充的 `211` 就是**假对齐层**，明确不做。
- **mimic / target / outcome 三组 reward**：同上，分别卡在 measured teacher、question packet、
  analytic outcome evaluator。
- **参考包络三条终止**（`anchor_pos` / `anchor_ori` / `ee_body_pos`）：需要 motion reference，没有。
- **WAIT / 揭示结构**：计数器和掩码本身好搬，但**被掩掉的那些行这边根本不存在**，搬过来是个空 WAIT，
  测不到 reveal bridge 的可学性。属于"形式能对、语义不能对"，登记不做。
- **球的接触模型**：要么两边同上原生接触、要么两边同走解析，当前谁都不是。而且本车道球拍恢复系数
  （`e=0.654` 常数，实测是速度相关 `0.759*exp(-0.0441 u_n)`）与网（`e=0.10` 纯假设）本就是具名缺口。
- **摔倒阈值 / 动作解码默认 / 撞桌硬终止 / 复位随机化**：这四条**技术上一行就能改**，
  但每一条都会改训练分布、让新 run 与既有 `103` 条 flat 收据不可比。**属发车决定，不在 review 里替 Franco 定。**
  台账里每条都写了 `closable_by`。

#### 七、这一节不代签什么

- **不代签"补上这些差异之后两边就会学出同一个策略"**。台账只回答"问的是不是同一道题"。
- **不代签 mjlab lane 的任何一条曲线**。相反：这一节的结论是那些曲线现在**只能**当本车道内部陈述读，
  而且从今天起每条收据自己带着这句话（`isaac_alignment.scope_sentence`）。
- **不代签 Isaac 侧 reward 权重的完整性**：台账读的是**类体里声明的**那一份，发车时 reward-pack YAML
  仍可覆盖，这一点写在该行的 `caveat` 里。
- **不代签撞桌率的绝对数**：`512` 世界、`12` 迭代、两个 seed 是**烟测规模**，`100%` 这个数在
  `4096` 世界、长预算下会是多少没测。能代签的是**方向和机制**：它从 `0` 学到 `~1`，主犯是球拍，
  而 Isaac 判它死。
- **不代签吞吐结论**：同配置背靠背一对（`seed 1`，`512` 世界）探针关 `3168.5` / 开 `2739.0` env-step/s
  （`-13.5%`），与 §9.2.8 在 `4096` 世界量的 `~13%` 同量级，但**一对不是吞吐结果**，
  且 `512` 世界的绝对值与 `4096` 的不可比（同一天同配置另一条 seed 0 的 median 是 `7908.9`，
  散布本身就说明这个规模下的计时不可引用）。

#### 八、收据

- pod1 独立 worktree `/workspace/franco/mjalign_20260806`（`e309b5b5` + 本轮改动），
  收据在 `/workspace/franco/mjalign_20260806/RECEIPTS/`：
  `ALIGN_LEDGER.json`（`ledger_sha256=c977c47e3a2d1a5c23462b1308a0f6114434c7ec8297373bf3149435569a9a8a`，
  `17` 轴 = `5` 对齐 / `10` 要紧差异 / `2` 有理由差异 / `0` 读不到）、
  `SMOKE_TABLE.json`（seed 0 撞桌曲线）、`PROBE_OFF.json` / `PROBE_ON2.json`（seed 1 探针关/开配对）、
  `SMOKE_ALIGN.json`（收据里第一次带 `isaac_alignment` 块）。
- 测试（`/workspace/mjlab_venv`，host-only，不占 GPU）：
  `mjlab_lane/tests/test_isaac_alignment.py` **`21 passed`**；
  既有 `tests/test_mjlab_lane_reporting_gate.py` + `tests/test_mjlab_lane_capacity_gate.py`
  **`64 passed`**（改动前 `62`，本轮新增 2 条撞桌拒绝测试）。零回归。
- plant 复验：`/tmp/PLANTVERIFY_20260806.json`（`92 match / 1 registered mismatch / 0 unregistered`）。
- 部署形态验证：把本车道拷到一个**没有仓库**的目录（复现 pod 上 `/workspace/mjlab_lane` 的形状），
  台账给出 `17` 条 `unverifiable`、`cross_engine_comparable=false`，并拒绝 comparability 主张。
  **"我读不到"从此不会长得像"我读了且对上了"。**

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

> **2026-08-06 执行更正**：上面这句在 `C211 oracle32` 的验收器里**没有被执行** —— 它把摔倒/太低/撞桌
> 也当成了"必须零次"，于是一个未开训的策略永远拿不到 oracle32。已按本节口径重定范围，
> 并同批交付守恒普查、词表收紧、WAIT 排除分母和收据自陈 telemetry，见 §5.6.8。

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
