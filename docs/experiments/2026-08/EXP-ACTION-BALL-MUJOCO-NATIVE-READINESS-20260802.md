# EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802 — ActionBall 下一版系统与 MuJoCo 原生训练准备账

- 状态：`r36-v5-dual-learning-live / model300-visual-closed / contact-prior-candidate-validated / diagnostic_unauthorized`
- 阶段/轴：ChingMu-73 动作库、Ball-first 自动扩域、Isaac 最小可学门、MuJoCo 原生训练
- 集成小目标：用一个自然动作在 Isaac 验证可学性的同时并行完成 MuJoCo trainer；共享 bundle 冻结后两引擎 N1 并行，主训练在 MuJoCo 直接扩到通过机械准入的完整 73 动作
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 本 successor 当前最高证据等级：v5 bundle、Isaac nominal hold、teacher 与真实 `model_300`
  固定视角、双端 CUDA fixed-action、Isaac fresh ACK 与 Mu 自然 contact/crossing 已有真 Pod 证据；
  Isaac 旧配方卡在 mimic→hit，Mu 的 hit 入口正在形成但 landing/recovery 仍未形成
- 创建日期/最后复核日期：2026-08-02 / 2026-08-31

共享缩写按[术语与人话对照](../../DEFINITIONS.md)解释。本文件是下一版系统的**依赖、证据充分性和
版本迁移账**，不是全项目优先级队列。当前采用 setting、认领和算力顺序仍只认
`origin/main:docs/NOW.md`；功能分支内的 `docs/NOW.md` 只能是待合入提案。

> **阅读规则：**当前执行合同只认“2026-08-31 current correction”与
> [双后端TODO当前节](../../operations/action_ball_dual_backend_longrun_todo_20260819.md#fullmdp-v6-todo-current)。从
> “2026-08-21 portable successor事实纠正”起的229/399、211/319、H24、`history=8`与旧schema/gate结论
> 都是可追溯历史；除非本节明确引用，不得反向覆盖H48、三段reference、`10/5/6` wire、PPO V5或
> Observation V3 `215/231`。V2 `203/219`只作旧checkpoint ABI和paired control。

## 2026-08-31 current correction

### 当前结论

R36 v5 已把“起始动作与老师动作互相代替”的错误拆掉：physical-ready 是 A3 闭环可站稳的
出生姿态，teacher motion 保留真实 frame0 与速度，preparation window 由 Motion clock 冻结老师，不再
伪造零速 motion。v5 bundle 绑定 teacher、dynamic-ready、nominal-hold、OptiTrack physics 与 current Mu plant。

真 Isaac nominal hold 为 `1.2 s / 240 physics / 60 policy`，无 hard-limit/table/fall；固定视角已直接看到
raw env reset 不是训练出生，atomic ready write 后的姿态才是实际 ready，teacher contact frame 中球、拍心与
挥拍目标对齐。这类截图从现在起是 reset/reference/task/contact 改动的必做工作流，但不作为新
发射 Gate。

`5dd39786` 的双端真 CUDA fixed-action 都完成 `512×48×31`。Isaac fresh root
`/workspace/franco/runs/fullmdp-r36-v5-5dd39786-isaac-h48-20260831T0939Z`
已持续到 update 149，recent-30 p50/p90 约 `10.74/10.96 s`。update 149 的 D05 due/accepted=
`174/174`、playback-started=`179`，physical-launch/R03/contact=`0/0/0`；`200` 个完成 episode 平均
132.48 tick`，table/tilt=`120/80`。到 update 305，近 30 轮已有 physical launch/observed=`79/79`，
说明 hit 入口自然打开；但 R03/contact=`0/0`，平均 episode=`148.67 tick`，仍多数早于
约 tick 300/313 的 launch/R03 时钟终止。四项 playback 误差为
`0.313 m / 0.579 m/s / 0.216 rad / 0.224 rad`，nonfinite/conservation=`0/0`。这是
balance→mimic 仍在学、hit 刚出现分母的证据，不是 target 错，也不是 mimic 已成功。

到 update 513，GPU0 Isaac 近 100 轮 episode 平均已到 `353.37 tick`，launch=
`4,041`、timeout/tilt/table=`3,862/2,093/878`，但 R03/contact 仍=`0/0`。因为晚期
playback 样本比例已显著改变，四项 raw 均值只用来定位 matched phase 评测，不直接评判
政策退化。GPU2 Mu 到 update 690 的近 100 轮为 episode `154.78 tick`，
launch/R03/contact=`2/2/0`，tilt/table=`8,192/7,709`；它仍主要败在 balance/mimic。

Mu 精确诊断已给出 `uid_rows=0/24576, identity_rows=0/24576`。根因是 runner 和离线 consumer
仍各自手抄 0807 旧 UID `2552478955674699`，而v5 portable catalog 真源为 `4098890508575574`。
`e5c02ea6` 删掉这两个生产常量，统一从 exact portable catalog 取身份，但保留 ledger 对每行真漂移的
检测；Pod runner/ledger/consumer focused=`275 passed, 1 skipped`。fresh Mu diagnostic 自然完成
61 update，RC=0，p50/p90=`6.667/7.153 s`，每轮 UID/identity=`24,576/24,576`，storage finite且
reward conservation/fact-integrity 无故障。同 source 已发 fresh long：
`/workspace/franco/runs/fullmdp-r36-v5-e5c02ea6-mujoco-h48-20260831T1034Z`。

### 结构裁决

| 处理 | 内容 | 第一性原理理由 |
|---|---|---|
| 直接删/合并 | 同 writer postcondition、每步 full-manager snapshot、死 registry、可离线重建转录 | 没有增加独立信息，却放大结构和热路税 |
| 保留 | nonfinite、跨 owner identity/generation、真 contact/outcome、有限计数与 durable checkpoint | 这些是不可从同 writer 重建的真值 |
| 后续独立落地 | owner 窄投影、metric/Reward 同周期 pack、external closure 单命令物化、exact resume | 提升可维护性，但不热补正在学的 source |
| 拒绝 | 为了弥补同 writer/时钟/坐标错误再增 success/safety Gate | 会掩盖根因并使项目更难审计 |

当日已执行第一个“死 registry”减法：整套 Racket observation token/registry 没有任何生产
consumer，production binder 永久 HOLD，只由 focused test 自行 mint/consume。因此删除该 capability、record、
retained fields、binder、测试和 stale semantic exclusions；不动 ActionEpoch 实时 selected-rubber、cold table 或
D05/R05/generation owner。这是删除无消费者的模拟权威，不是放宽真值校验。

第二个确定性减法将 Reward28 每个 control step 的 28 次独立 row 转录收敛为一次
batched close。Pod1 同 seed 两条 61-update probe 的整份 ACK JSON 逐字节相等；
p50/p90=`11.060/11.291→10.755/10.943 s`。它只证明约 `3%` 的确定收益，
没有把 D05/reset/command 主墙写成已解决。`67109ec2` 另将存档频率改为
`300 update`，并从 fresh GPU1 root
`/workspace/franco/runs/fullmdp-r36-67109ec2-isaac-h48-gpu1-20260831T1153Z`
开始后继；首个 `model_300` 用于固定机位开发检查，不是新 Gate。

checkpoint 回放链在 `d8fc5d12..5b32f42e` 完整收口：它复用训练的 FullMDP owner factory、typed PPO
identity、dynamic-ready binding 与代码拥有的 motion catalog；capture 不再误触发 ONNX export，只接受当前
RSL3 grouped observation，不为旧 actor-tensor/checkpoint 增兼容支路。Pod focused=`7 passed`；真实
`model_300` 400-step 回放 RC=0，影片 SHA256=
`3e5c3a58fe831b3fd362f4ece02fab3bcbb4d1c95540569ef343f134bc6da13e`，证据目录为
`/workspace/franco/evidence/r36-policy-u300-fixedcam-gDJxLV`。画面显示策略不是出生即倒，能产生周期性
动作并存活/重置，但没有清晰形成老师反手挡的接触窗动作或可信击球；这与同窗 R03/contact=`0/0`
一致。确定性 teacher 视频仍证明中心题球、拍心和 contact target 对齐，所以不再改 task/球公式。

到 update `690/268/1182`（Isaac GPU0 / Isaac GPU1 / Mu GPU2），三条旧配方给出分层而不是
一刀切的结论：GPU1 仍在较早 balance/mimic 窗；GPU0 最近 100 轮 episode 已均值 `496.013 tick`、
physical observed=`4,915`，却仍 `0 R03 / 0 contact`，所以旧配方的 mimic→hit 不是“再等就能解释”;
Mu 最近 100 轮已有 `33 launch / 28 R03 / 1 selected contact / 1 crossing`，证明 hit 入口结构上
可达，但概率极低且 legal landing/recovery 仍未出现。GPU0 同窗 paddle 位置/速度/拍面/长轴误差为
`0.6270 m / 0.3192 m/s / 0.3677 rad / 0.2980 rad`，与大量非接触帧 paddle income 同时存在，
形成了明确的时间信用稀释证据。

后续只读窗进一步把“耐心等待”和“需要 treatment”分开：GPU0 Isaac update 760--859 的 episode
均值=`491.204 tick`、observed=`4,853`，但 R03/contact/crossing/landing 仍全零，已是稳定的
mimic→hit 负例；GPU1 Isaac update 326--425 的 episode 均值=`190.906 tick`、observed=`1,596`，仍是
较早 balance/mimic 对照，不能据此判长期失败；GPU2 Mu update 1559--1658 已有
`3,416 launch / 3,364 R03 / 1,337 raw / 229 selected / 230 crossing`，selected/launch 约
`6.70%`，但 legal landing/recovery=`0/0`。因此保留三条自然对照，不在已有学习链正在推进时截断；
contact-prior treatment 等独立 lane 后按 matched window 比较，而 landing 只有在 hit 基本形成后才调整。

最新只读窗口继续支持同一裁决而不是翻转它。GPU0 update 849--948 为 episode/observed=
`464.011/4,735`、R03/contact=`0/0`；GPU1 update 412--511 为 `347.648/4,000/0/0`，且与 GPU0
同 seed 的 update 0--499 逐窗学习账相同，因此只能充当实现等价/耐心对照。Mu update 1879--1978
已到 launch/R03/raw/selected/crossing=`4,938/4,890/3,283/465/464`，selected/launch=`9.42%`，
但 legal landing/recovery=`0/0`。三条 nonfinite/conservation 均为0。

“fresh 早期少动”可接受的因果条件也据此固定：episode length 和 teacher error 正在改善时，它是先学
balance 的合理策略；接近 horizon 且 due/observed 充分后仍零 R03/contact，则是成熟 mimic→hit 负例。
Build4 不能裁决该问题：`origin/build_4@324e60d1` 的 manifest/YAML 明确要求从 Build1
`model_21800.pt` 位级载入8个 actor tensor，再重置 sigma/critic/optimizer。当前仓库与 Pod 没有原始
Build1 model0/早期 checkpoint 序列，所以不存在公平的“Build4 fresh 早期会不会动”证据。

`dad61048..00814042` 因此直接修已有 reward 的时间分布，而不新增 gate：ready/preparation/recovery
以及远离接触的 playback 保留 `1x`，真实 playback 内按 raised-cosine 在 `|TTC|<=0.12 s` 连续增强，
`0.06 s` 为 `2.5x`、接触为 `4x`。四项 kernel、teacher/achieved producer、最大权重和
balance→mimic→hit→landing 自然链均不变；Isaac 与 Mu 共用同一纯 tensor 函数，各自从同义动作钟
产生 TTC。Pod exact `00814042` 回归为共享/Isaac `368 passed, 26 skipped`、Mu
`214 passed, 6 skipped`。这关闭数学/接线正确性，学习效果只由下一条 fresh canary 裁决，不能由
旧 run 或 total return 代签。

## 2026-08-30 current correction（已被上节取代）

### 当前 adopted contract

R36 保持单一自然重叠课程 `balance → mimic → hit → landing/recovery`，不增加硬 Stage、success Gate、
Observation 或 oracle。D05 到期即发布 frozen ActionEpoch；未请求 physical launch 的行用 typed
`UNPLAYED`无债退休，ready 且已请求却没有 launch 才是合同故障。physical launch 和 TTC 改为
reveal-relative，Mu 只有在 teacher 真正离开 frame 0 时才开始 playback；R03 的位置、速度和有符号拍面
唯一读取同一 ActionEpoch。

Phase1/2 负责几何与连续解，Phase3 只为最终 plant search 排 seed；cross-intent face equality 和
seed face distance 都不是 admission。Phase4 才以最终 A3 plant、qdes/torque/support、racket-site、
桌网球物理和 recovery 独立准入。其击球 deadline 向上对齐到 20 ms policy grid，contact 后保持
safe follow-through，canonical bundle 同时绑定 exact physics YAML 与 compiled Mu plant。

### R35 最终冻结负例

以下是同一只读刷新得到的最终 50-update 窗；这是详细数值唯一真源。两端均无
fact/nonfinite/conservation 故障，故负例可信，但学习差异不能冒充 physics parity。

| 项 | Isaac update 4007--4056 | Mu update 11133--11182 |
|---|---:|---:|
| completed episodes | 976 | 12,982 |
| mean episode length | 1,349.511 tick | 94.637 tick |
| timeout / tilt / table | 802 / 155 / 20 | 0 / 12,981 / 0 |
| playback position error | 0.225952 m / 363,949 | 0.804959 m / 23,596 |
| playback velocity error | 0.800322 m/s / 363,949 | 2.020818 m/s / 23,596 |
| signed-face / long-axis error | 0.293570 / 0.195999 rad | 0.307874 / 0.628629 rad |
| due / admitted | 5,048 / 5,043 | 12,979 scheduled/revealed |
| physical launch | 4,951 | 0 |
| R03 valid / selected contact | 0 / 0 | 0 / 0 |
| legal landing / recovery-ready | 0 / 0 | 0 / 0 |
| qdes projection | 0 joint samples | 192,700 rows |
| actual hard edge | 5,643,734 joint samples | 2,207 rows |

Isaac survival/balance 已基本形成，且 launch 分母充分，但 mimic→hit 交接为 `0/4,951`；不能再解释成
“太早”。Mu 几乎每个 completed episode 都带 tilt，launch 分母为 0，故先败在 balance/mimic，
hit/landing 写`未测`而不是 0。R35 两条 root 继续只读，禁止 resume、hot-patch 或复用 namespace。

### R36 已落地的直接修复

| 因果缺口 | 当前修复 | 状态 |
|---|---|---|
| physical launch 和 teacher clock 混用绝对时钟 | `cc29cbb8` 改为 reveal-relative | source 已落地 |
| R03 caller 重算 target | `c35a348d` 只读 frozen ActionEpoch | source 已落地 |
| Reward bundle 构造后同写者自证 | `87f4156f` 删除 metadata self-proof，保留跨 owner 门 | source 已落地；final Pod`未测` |
| Mu filtered-net 派生 plant 漂移 | `d2144334` 重编译并 repin MJB | source 已落地 |
| Phase2/3 seed 与 admission 混同 | `64cb026b`、`66eebb33`、`fcd8c8ab` 分离 | source 已落地 |
| final plant authority 丢失 | `f5eb6bfe` 保留 Phase4 准入 | source 已落地 |
| contact 后轨迹不安全 | `e4067411` 保持 safe follow-through | source 已落地 |
| bundle 未绑定 exact ball physics | `f59d485b` 内容绑定 physics YAML | source 已落地 |
| TTC 不在 policy grid | `4fe23765` ceil 对齐 20 ms grid | source 已落地 |
| command metric 重复 D2H | `bf941777` 候选窄化 transfer | source 已落地；parity/rate`未测` |

当前 exact MJB SHA-256 为
`d9c88297d4a687815c347a064792c66d99b61965ecea3576d235c8b13c286685`，大小
`113,765,788` bytes。它只证明内容身份已更新；final Pod loader、fixed-action 和 long 仍须逐端验证。

### Phase4 证据边界

pre-final predecessor evidence root
`/workspace/franco/evidence/r36-optitrack-final-6678d6a2-kYZRqrIe`曾给出一个独立最终 plant 可行例：
qdes margin `+0.022761 rad`、face `6.878°`、solver velocity error `0.01095`、racket-site
position/velocity error `0.002402 m / 0.010924 m/s`、table clearance `0.020 m`、torque slack
`+4.6086 Nm`、support `1`，并得到有效 net/landing（net z `1.2203 m`、landing error
`0.000603 m`）、`t_hit=5.088 s`、`t_cycle=5.936 s`。它证明 Phase4 这种最终 plant 复核路线可行，
但发生在 `4fe23765` 的 physics binding、follow-through 与 policy-grid deadline 之前，不能移签给 final
HEAD。final bundle materialize/reopen、trainer consumer 和 Pod 重验仍全部写`未测`。

### adoption table

| 裁决 | 内容 | 理由 |
|---|---|---|
| 采用 | reveal-relative launch、ActionEpoch 单一 target、Phase4 final plant、内容绑定 physics/plant | 确定性合同修复，不做学习 A/B |
| 采用但待 Pod | Reward owner 减法、metric transfer | 数学语义不变仍需 fixed-tape/Pod parity 与墙钟 |
| 需要 canary | reward 剂量、curriculum failure target、full-body、entropy/sigma | 会改变学习分布或经济 |
| 暂缓 | exact-resume consumer、teacher plant 动态重定时、sim2sim torque/contact 对签、monolith 拆分 | 非本轮 fresh long 的必要前置 |
| 拒绝 | 以 total return 代替 denominator、为结构问题加 success Gate、用旧 Phase4 evidence 代签 final HEAD | 会掩盖真实因果或伪造证据等级 |

### final Pod 与 fresh run：当前未测

- [ ] final clean exact checkout 恢复并核 external assets、EPA48/RSL3、current MJB 和 physics YAML。
- [ ] final Phase4 bundle materialize/reopen，且 trainer consumer 真读取 bundle、deadline 与 ActionEpoch。
- [ ] focused/combined Pod tests。
- [ ] Isaac 与 Mu 各自真实 CUDA `512×48×31` fixed-action。
- [ ] `bf941777` fixed-tape/reason/counter/safety/payment parity 与 profiler-off matched rate。
- [ ] fresh O_EXCL run root/namespace、首个 finite durable ACK 和后续真实 denominator 窗。

以上任一未完成都不能写 PASS。Mu 与 Isaac 可在各自 GPU 自然可用时独立启动，不要求串行等待；两端都保持
`diagnostic_unauthorized=true`。

## 2026-08-28 current correction

**环境与实现必须分层：**当前pinned `/opt/IsaacLab-8320e0be`、Isaac Sim Kit Python、sealed RSL wheel和
A3 USD已经在Pod1完成真实environment construction、训练与durable ACK；没有证据支持“Pod安装损坏”。
`/workspace/IsaacLab`是另一个可变checkout，不能混入当前run身份，也不能用它解释Jiayi本机差异。当前已证
错误在训练配方/接线：V8混用了Take058 teacher、Take061 physical-ready与被legacy YAML再次覆盖的bootstrap；
catalog attachment又漏装action order，导致motion `N=1`而action identity `N=0`。所以“起始动作错”确实
属于当前sim训练实现问题，但不能偷换成整个Pod或physics engine安装坏了。

raw teacher frame0与可执行初态也不是同一事实。现有Take061 nominal收据只证明physical-ready→teacher
frame0 bridge可在60 policy tick内维持；末端`waist_roll=-0.3205 rad`且仍以约`-1.19 rad/s`向
`-0.3491 rad`限位运动。旧first reveal tick295把1.2秒收据外推到5.9秒，而fresh policy在tick77--82已
倾倒，故旧配方看不到mimic不是设计上必须多等step，而是课程时钟写错。

V9最小修复把Take061 ready、teacher与action identity原子绑定，首次due改为tick48，六次due为
`48/233/418/603/788/973`；185-tick完整task/recovery cadence、Reward28、Observation V3和PPO V6不变。
真实Isaac在update4得到`due/selected/accepted=512/512/511`、Take061身份`511/511`、playback=`248`，
完整61-update窗累计`18,432/18,432/18,419/13` due/selected/accepted/rejected与`9,120` playback；
terminal `18,431/18,431`均为base tilt，recent10 episode mean=`81.916 tick`，physical launch=0。
active-task p50/p90=`10.470/13.863 s/H48`，而tick295空业务为`6.635/7.153 s/H48`，说明真实task热路仍
需大砍。同源码MuJoCo有限窗p50/p90=`6.644/6.854 s/H48`，scheduled/reveal=`10,861/10,860`、
launch=`6,658`、R03 valid=`5,107/5,107`、selected contact=`0/6,658`；paddle位置/速度误差也未改善。
两端有限窗因此只证明balance→mimic入口已修，不证明mimic成功、hit/landing成功或跨引擎parity。

环境诊断保持分层：首个Mu root因clean checkout未恢复ignored EPA48/RSL3 bundle在首ACK前失败；按现有
`setup_local_sync`固定SHA恢复后r2自然完成。这是资产同步问题，不是运行后机器人动力学分叉的解释。当前
Isaac约82 tick全base tilt且无launch，Mu约140 tick、table/tilt混合且有大量launch；必须用同一初态与固定
31-D action tape逐tick对齐后，才能判断Jiayi本机、Pod、Isaac与Mu哪个callpoint先偏。所有证据继续
`diagnostic_unauthorized=true`。

该对齐已先找到并修复一个确定实现错误，而不是把它升级成“Pod整体坏”：clean `981327de`中Isaac出生写
asset default、Mu写Take061 dynamic-ready，首action前joint/root/racket-position最大差达
`1.5199 rad/.1778 m/.5376 m`。RSL wrapper本来就是唯一genesis reset owner；正确修复是让唯一Isaac reset
Event消费Motion physical-ready窄projection，不是再加第二次reset或新Gate。clean `179148e3`修后initial
q/dq逐位相同，root/racket-position最大差降为`4.1e-7 m/.467 mm`，done/time-out逐格一致。

同时v1 fixed tape被审出问错对象：它围绕raw action `0`，而fresh actor真正以Take061 normalized hold action
为mean（最大绝对值`16.3001`）。所以v1 post-step是在测试“physical-ready后猛拉回asset default”，并造成
每8 tick全量终止/reset；它只授权birth结论。当前v2 tape改为同一tracked live actor mean中心的`±.02`
扰动，并要求两个backend的live mean逐位匹配。clean `3343fe90` centered compare中两端48 tick均无Done，
但Isaac相对初态的joint/root/racket最大漂移为`.349 rad/.092 m/.170 m`，Mu仅`.012 rad/.008 m/.010 m`；
跨端joint max差从tick0 `.0059 rad`增至tick47 `.3480 rad`。v3 record显示tick0--34实际executable
`joint_qdes`最大只差一个float32 ULP（`5.96e-8 rad`），而q在首个20 ms控制步已经分叉；tick35后
Isaac waist-roll逼近hard-inner才让guard改写q_des，这是plant分叉的后果。shared decoder错误已排除。
旧Isaac nominal-hold PASS本身也记录60 tick末`waist_roll=-.3205 rad, dq=-1.19 rad/s`；因此它只能称
60-tick nonterminal prefix，不能称静态hold/readiness。剩余差异属于backend plant/controller response；
它不阻止两端分别学习balance，但不授权physics parity，也不支持“Pod安装坏了”。

同一clean exact source=`eb57233b4522d527455a0cbd7c547eb2ec49a68c`随后在Pod1发射双fresh长期
replacement。Mu namespace=`fullmdp-a-h48-v9-mujoco-genesisfix-eb57233b-20260828T093350Z`，Isaac
namespace=`fullmdp-a-h48-v9-isaac-genesisfix-eb57233b-20260828T094243Z`；两端各自产生连续durable ACK后，
V8才按exact PID/PGID/cwd/source/namespace停止，旧root/checkpoint保持只读、无伪造completion。
`observed_at=2026-08-28T09:48:33Z`时Mu ACK0..83=`2,064,384` transitions，首10→近10 episode mean=
`137.31→141.96 tick`，近10 launch/R03/contact=`1,210/855/0`，wall mean=`6.57 s/H48`；Isaac
ACK0..18=`466,944` transitions，episode mean=`67.34→73.55 tick`，近10 D05 due=`3,414`但physical
launch/R03/contact仍为0，wall mean=`16.76 s/H48`。两端reward nonfinite、conservation与fact/attribution
fault全0。Mu hit已有早期`0/launch`分母但尚不足以裁决新学习配方；Isaac仍在balance且hit/landing为`未测`。
当前前缀既不支持mimic成功，也不支持Pod损坏、physics parity或formal promotion。

结构裁决不新增安全Gate：把catalog变成一个typed原子安装对象、把bootstrap归给唯一owner、把backend留作
薄adapter；保留plant/full-key/optimizer/durable事实边界。warm-start、replay、双LR、sigma、Build4数值、
新observation与DR均延后，直到当前同源课程产生可归因的未来窗。

## 2026-08-26 current correction

**当前fresh与旧run分层：**V6 exact source=`caddecb76727ea55b0ce089453eea91cb5a9f8ea`两端已经在
精确PID/startticks/PGID/source/namespace复核后停止，其run root与科学结论冻结为negative；继续到25k
不再是当前学习结论的前置条件。Mu固定
`673,087,488` transitions后有`0/1,439,028 launch` selected contact，Isaac固定`246,644,736`
transitions后有`0/409,414 launch`；landing因没有eligible contact仍为`未测`，不能写成零成功率。
两端episode mean length已接近1500-tick horizon，说明balance/survival形成；mimic收入反降且wire没有
teacher-achieved真实误差，因此不能宣称mimic基本成功。冻结分母和因果裁决只认
[课程实验§10.7](EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#107-v6结论翻转生存形成但mimichit桥与joint经济失败)。

**V6/V7已执行学习合同与V8 replacement：**V6/V7 learner采用[`PPO V5`](../../DEFINITIONS.md#fullmdp-ppo-v5)的
`2048 env × H48 × U25000`、save500、E5/MB4，继承V4的GAE`lambda=.98`、entropy0、learned`log_std`、
fresh sigma`.05`且无强制decay。总transition、minibatch大小和optimizer-step数虽保持，但刷新/GAE/KL/
WAL/checkpoint边界改变，因此已另做exact rate与fresh学习prefix；结果只认
[热路实验§16](EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md#fullmdp-v6-rate-current)和
[课程实验§10.7](EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#107-v6结论翻转生存形成但mimichit桥与joint经济失败)。
源码确认[`Reward24`](../../DEFINITIONS.md#fullmdp-reward24)整块替换时删除了`action_rate_l2`、qdes
projection/qdes limit和actual joint-limit四项连续学习代价；actual hard edge却有意只记账不Done。Mu
actual-hard-edge从早期低位升到recent10=`51.57%`、末段500窗=`58.42%`，Isaac从first10=`1.843%`
升到recent10=`3.366%`。V7现已实现[`Reward28`](../../DEFINITIONS.md#fullmdp-reward28)：恢复共享
`action_rate_l2=-.1`、qdes soft=`-10`、projection=`-1`和actual joint soft=`-10`四项非正连续成本；
paddle改为固定50/50 precision-exp + coarse-Cauchy并保留四项分账。这不是新增安全Gate。Observation继续采用
[`V3`](../../interfaces/policy_observation_action.md#current-portable-fullmdp-semantic-observation-v3-actor-215--critic-231)
`215/231`：四个全phase、同clock measured-teacher-minus-achieved paddle heading residual关闭Reward source
未直接actor-visible的representation gap；不加入raw ball/aim/rate/history/action ID，也不宣称strict
Markov alias、完整Markov或部署已验证。motion
normal是signed physical face；ball-task normal仍是raw-A，两者不可复用。

**Cadence与wire纠正：**真实due只有tick`295/588/881/1174`；`1467`是第四球settlement boundary，不是
第五个机会。V3 actor `[208] time_to_next_opportunity_s`在第四次due消费后用raw `-1`表示exhausted，两个
backend同义。Isaac的due+terminal overlap来自独立ResetTelemetry与D05 scheduled-due在既有CPU
pre-optimizer drain求交集，不新增每步Gate/D2H/owner。V7 Isaac milestone schema8按具名字段解码，新增
四项teacher-achieved真实误差finite/sum/sumsq；Mu evidence/completion/summary为`10/5/6`。显式升版避免
新旧wire伪兼容；旧schema7、`9/5/6`和`6/5/5`只作各自source历史。

**MuJoCo native表示与Pod结果：**真实Full-A已证明pose事实源是底层
`data.struct.xpos/xquat` native Warp `wp.array[vec3/quat]`，不是外层TorchArray代理；最终实现直接消费
native arrays，不做Torch中转或host sync。72b失败分类、最终`1,036 passed, 11 skipped` CPU矩阵与
PPO V5性能边界只认[热路实验§16](EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md#fullmdp-v6-rate-current)。
V7随后在两端共约218万launch后仍零contact，且两个checkpoint optimizer LR均卡`1e-5`；当前V8候选只换成
[`PPO V6`](../../DEFINITIONS.md#fullmdp-ppo-v6)的fixed LR与512-env刷新，Reward28/Observation/plant不变。
source `0ad85ae1…`已完成exact Pod目标测试、双rate与fresh双端启动；Mu/Isaac p50/p90分别为
`3.796/3.999`与`6.835/7.612 s/H48`。启动期仍未到首个due，mimic/hit/landing为`未测`，不能把运行写成
学习成功。
最终两端61-update空业务rate曾为Mu p50/p90=`5.468/5.526 s/H48`、Isaac=`7.81/8.38 s/H48`；active
业务近期已恶化到Mu约`9.10 s/H48`、Isaac约`32.06 s/H48`，后者collection约`29.73 s`。空任务率不能
代表真实训练率。V7已把milestone逐term eager reduction合为每control step少量batched reduction，并将
Motion同writer完整record往返缩为四字段projection与一个bool结果；跨writer事实、optimizer→WAL/fsync→ACK
仍保留。逐CommitEntry/row完整业务重放继续是后续性能项，不能在未验证compact分层前直接删掉。

V7 exact source=`1d33130ba07288918aa73d1323e1106303b7cad1`的14个受影响测试文件已在Pod1逐进程隔离为
`706 passed, 32 skipped`；Mu/Isaac launcher dry-run、fresh root与首批durable ACK均通过。现役namespace为
`fullmdp-a-h48-v7-mujoco-reward28-obsv3-1d33130b-20260825T180216Z-r2`与
`fullmdp-a-h48-v7-isaac-reward28-obsv3-1d33130b-20260825T180216Z`。Mu首个可复算启动窗ACK0..16的
recent10 mean/p50=`5.360/5.366 s/H48`，Isaac update0..61为`8.423/8.380 s/H48`；两端均是28项、
finite、conservation fault 0，Mu schema10与Isaac milestone schema8成立。当前尚是balance-only，paddle
误差、contact和landing没有eligible分母，全部写`未测`；active/mimic matched wall与matched physics也仍
`未测`，不能整体抬成跨后端`E2`。

**Build4辩证对照：**mandatory Build1 `model21800` actor warm-start是相对fresh run最直接的已证配置
差异；因此Build4不能被当作fresh-from-zero证据。缺actual model0 receipt且配方混杂，早期行为也不能独立
归因给继承。只采用direct-paddle
objective/correction-information原则；Reward28是本轮直接Reward实现，Observation V3 residual是本轮独立
实现且必要性仍待paired control。曝光频率、warm-start、
replay、双LR、sigma与具体权重延后。当前V6仍是single slot0，所以2 clips与73动作的差距只限制最终外推，
不解释当前early-learning差异。Build4 selected candidate仍`NOT_PROVEN`；commit、混杂因素与
adopt/defer/reject详见课程实验§10.3。

**结构与安全边界：**目标仍是`PlantFacts → ActionBallState transition → StepTelemetry`。same-writer echo、
无人消费Gate或seal后source recheck只能作为逐callpoint删除候选；未点名writer/seal并同批提供mutation反例前
不得删除。跨writer conservation、native plant finite/joint/table、full-key/generation、optimizer边界、
run-owned no-clobber、WAL/fsync/ACK和GPU lock继续保留。
本轮已删除dead keyed work与post-ACK identity echo；Motion路径又以四字段窄projection替代同writer完整
record往返，milestone也改为按term批处理。静态payload只能说明删除了无用数据流，不能换算为秒；active
profiling仍须覆盖PPO-boundary完整业务重放。只保留学习、分层聚合、独立事实与durability
真正需要的最小状态；不得把臃肿owner graph原样翻译到C++，也不得用新Gate补偿目标函数缺项。

### 2026-08-28 current adoption table

| 项 | 当前裁决 | 原因 |
| --- | --- | --- |
| learner | 采用PPO V6 `512/H48/MB1/E5`、fixed LR`1e-4`；总transition/minibatch/optimizer step与transition-save cadence保持 | V7 adaptive-KL在task曝光前耗尽到`1e-5`；512-env让policy刷新快4倍是显式算法取舍，不冒充等价加速 |
| Reward | 采用Reward28：shared action-rate/qdes-projection/qdes-limit/actual-limit经济、真实paddle误差账和固定composite核 | V6在巨大launch分母下contact为0且硬边恶化；这是目标漏洞，不是课程尚未开放或需新增Gate |
| actor/critic | 采用Observation V3 `215/231`；V2只作旧ABI/paired control | 同producer最小残差关闭representation gap；拒绝raw ball/aim/history/ID扩张 |
| cadence | 采用tick`48/233/418/603/788/973`六次真实due与耗尽`-1` sentinel | 一轮H48先学balance，随后在60-tick实测ready窗内打开mimic；完整185-tick task/recovery cadence不变 |
| Build4 | 采用早期连续曝光、direct-paddle objective/correction-information原则；延后warm-start/replay/双LR/sigma与具体权重 | mandatory actor warm-start排除fresh-from-zero比较；tick48来自本系统H48/60-tick证据，不复制混杂数值 |
| performance | active-task Isaac p50/p90=`10.470/13.863 s/H48`；due count与wall相关系数约`.77`，先profile task construction/ACK热路 | tick295空业务`6.635/7.153`不能代表现役速度；保留跨writer事实与WAL/ACK，不保留无用完整record往返 |
| authority | V8已停止并只读冻结为negative；V9双端fresh长期仍`diagnostic_unauthorized`，G04/G05/G06 `Partial` | mimic趋势、hit/landing、physics parity与transfer均未闭合；启动ACK不是阶段成功 |

## HISTORICAL / FROZEN — 2026-08-23 current correction

**旧V4最终冻结，不迁移为V5验收：**MuJoCo最终连续durable ACK `0..4798`、`943,521,792`
transitions；recent20 wall mean/median=`14.146/13.994 s/H48`。累计public due/reveal/defer=
`1,637,789/1,637,789/0`、launch=`527,957`、R03 physically-valid=`145,814`，racket contact、
selected contact与landing均为0；完成episode=`3,376,589`，tilt/table/base-low terminal bit=
`1,521,819/1,860,583/383`，bit可重叠。Reward/storage finite且fault/nonfinite/conservation为0。
它在精确PID/startticks/PGID/cwd/source/namespace/GPU复核后停止，未跑满12500，故无`completion.json`。
旧schema-4的`racket_contact_eligible`与launch逐行exact相等，只是已删除的冗余别名，从来不是contact
opportunity或命中分母。

V4的episode已稳定越过first due 295，只证明survival-to-due可达且task/mimic/hit输入会自然
打开，不证明balance已基本形成；但reference、
typed identity与transition chronology实现错误污染了该lineage。零contact只能裁决**旧实现**
mimic→hit handoff失败，不能裁决自然课程设计。corrected V5现在已fresh训练；启动验收时两端还没有due/
action分母，随后MuJoCo已出现首批public due。当前可判断balance学习趋势与mimic输入开始自然开放；独立
playback成功、landing与recovery继续`未测`；后续V5已有Mu contact=`0/6 launch`的diagnostic negative，
Isaac因launch=0才是hit/contact`未测`。
课程保持balance→mimic→hit→landing：上一阶段开始成功时，下一阶段分母应已自然出现，不增硬Stage、
balance Reward或R07 admission。fixed slot0可完成due仍是`295/588/881/1174`；`1467`是第四球的
retirement boundary，不是第五次机会。

**Reference与Observation：**首次ACCEPT前joint/body共同使用reset-ready；post-transition ACCEPT把
selected measured action frame0原子安装到返回的`obs_{t+1}`，第一笔task-conditioned action/reward从
下一transition开始；recovery/ready使用completed-action frame0。V5修的是与该合同相反的旧selector回归，
不是新增offset。actor/critic保持semantic `203/219`，`history_length=0`；无
same-observation/different-required-action alias反例，不新增237-D、history、账本或不可部署oracle。
H48、GAE `lambda=.98`、E5/MB8、entropy 0、fresh sigma `.02`保持；约“6秒”只是继续大砍迭代时间的
方向目标，不是formal、launch或safety Gate。
actor已有Motion phase（`prepare_visible/swing/follow_through/recover_hidden/ready_hold`）；当前缺的
是独立durable playback denominator，故playback成功继续`未测`，不得由reveal代写。

**V5 exact Pod与fresh启动事实：**两条live run的clean/pushed source为
`39f9481950a660e198dedac7fd402806d648906b`。Pod broad CPU/ABI矩阵=`792 passed, 57 skipped, 0 failed`；
另以clean no-`PYTHONPATH`进程复跑plant/runtime/runner/consumer=`77 passed, 0 skipped`，两组不相加
伪装unique总数。Mu真实GPU direct/RSL H48=`5+1 passed`；Isaac CUDA projection/selected-rubber/
runner-drain=`32 passed`，其中runner-drain只签CUDA/RSL callpoint，真实Kit fresh run才是Isaac集成路径。
双launcher dry-run与absent-root检查均通过。

- Isaac namespace=`fullmdp-a-h48-v5-isaac-chronology-39f94819-20260823T144237Z`，GPU1。启动验收
  durable ACK=`0..63`（`12,582,912` transitions），最近五个完整console wall=
  `7.78/7.47/8.17/6.79/8.19 s/H48`；该窗口139,264个episode全因tilt终止，所有D05/Motion/shot事件为0，
  Reward finite、attributed fault与conservation为0。后续到ACK97，10-update mean episode length/return
  从首窗`87.606/7.052`升到末窗`97.776/8.705`，说明balance开始学习但没有基本成功，仍无due。
- MuJoCo namespace=`fullmdp-a-h48-v5-mujoco-chronology-39f94819-20260823T144237Z`，GPU0。启动验收
  schema-6 ACK=`0..8`（`1,769,472` transitions），post-warm recent5 wall mean/median=
  `8.727/8.777 s/H48`、`22,528.7 transitions/s`；16,378个episode mean length=`105.21 tick`且均为
  `robot_hit_table`。所有scheduled/public due、launch、R03、contact、outcome、landing、recovery与retire为0，
  Reward/storage/domain、Mu fact-integrity与conservation全绿，std=`.020000→.019945`。

当前学习轨迹只冻结到Isaac ACK450 / MuJoCo ACK385，不逐ACK追写。Isaac从早期PPO回退恢复并创新高；
最新20窗`23,081`个episode mean length/return=`170.249/15.625`且全部tilt，累计due/public=`20/20`、
ACCEPT/reject=`17/3`、playback=2，launch及其下游仍0。MuJoCo最新20窗`21,033`个episode mean
length/return=`186.693/17.771`，累计scheduled/public/overlap=`307/297/10`、defer=0、natural
launch/missed=`6/0`、R03 present/physically-valid=`1/1`；contact/outcome/landing/recovery仍0。两边
finite/fault/conservation均clean。这证明mimic输入、真实playback和自然launch/R03链已经开放，不需要硬Stage。
Mu racket/selected contact=`0/6 launch`，是小样本diagnostic negative而非`未测`；Isaac因launch=0，contact
仍`未测`，两端landing因selected contact=0也仍`未测`。详细冻结分母只认
[双后端TODO §0.4](../../operations/action_ball_dual_backend_longrun_todo_20260819.md#04-2026-08-23-v4最终冻结与v5第一性原理自查)。

两条都从fresh root、无resume/hot-patch/namespace复用运行，并保持`diagnostic_unauthorized=true`。最新冻结
窗口为Isaac stdout辅助total mean/median=`9.488/9.465 s`、Mu durable wall mean/median=`9.235/9.223 s`；不是
matched-strata稳态对拍，不能正式归因，约6秒方向尚未达到。正式50-update rate window、独立playback、自然
hit/landing、12500 completion与physics/transfer parity仍未闭合。

**唯一transition顺序：**freeze scheduled due → 结算既有launch/park → physics/terminal/facts/reward
继续归因给产生`action_t`的`obs_t` teacher → 结算outcome/recovery → 只对survivor分类public due →
安装frame0到`obs_{t+1}`。只有结算后仍busy才`DEFER`；起始busy但同边界自然`RETIRED`可立即
`ACCEPT`；due与terminal重叠不public。launch与outcome同tick合法，outcome与natural recovery互斥。
Epoch业务phase与Motion teacher/reference phase是两个独立clock，不要求伪同步。

**Fault namespace不混计：**

- Isaac共享ActionEpoch owner有36项single-bit row fault：bit0--26为27项既有原因，R03
  identity/stale/nonfinite为bit27--29，Physical/R06六项为bit41--46。R07 local ledger另属本地编号。
- MuJoCo每control只latch本transition的四项packed fact-integrity cause：R03 nonfinite、R06
  source-invalid、R07 sequence、R07 nonfinite。R03 stale因per-tick invocation与same-step consume按构造
  不可达，删除而不保留恒0 Gate。四项只进入既有唯一pre-optimizer drain，不加第二owner或逐step D2H。

**Evidence wire与诚实分母：**最终版本固定为`evidence/update schema=6`、`completion schema=5`、
`summary schema=5`；上一个已提交版本是`5/5/4`，本轮未发布中间版不再二次升号。
`scheduled_due_rows`表示schedule在本transition到达，`due_terminal_overlap_rows`表示due+terminal且
actor永不可见，`reveal_due_rows`只表示surviving public due。consumer只验fresh prefix上的可证明边界：
`launch<=reveal`、`racket contact<=launch`、`selected contact<=racket contact`、
`R03 present<=launch`、`flight outcome<=launch`、`landing crossing<=selected contact`、
`shot retired<=launch`。`invalid_contact + done`是真reset而非retire，只见marginal的consumer不臆造
same-writer overlap。当前`business_chain_complete`是producer逐row attestation加consumer边际聚合一致性，
不是独立same-env/same-epoch重放；晋级前仍需keyed carry-state重算或可重放trace。R03 exact-strike与contact
使用不同clock，`selected_contact/R03_valid`只作描述比；
selected-contact的正式分母是launch。`r06_common_per_eligible`才是closed task-landing成功率；
`opponent_landing_per_crossing`只是在crossing条件下的比例，不得代称总成功率，也不新增event或Gate。

**Terminal与authority：**`pure_timeout = raw_timeout & ~plant_terminal`；只有pure timeout获得RSL
bootstrap与canonical timeout reason，horizon与tilt/table/qdes重叠不bootstrap。Mu
`robot_hit_table` bit覆盖keepout或resolved-table，resolved只是子fact。真实authority顺序是optimizer →
WAL/fsync → owner ACK → EPOCH_ACK/fsync。live `39f94819`的stdout为regular file且未失败，但post-launch审计
发现snapshot/completion后的两处裸print在闭管时会把已durable状态伪装成失败；clean/pushed下一source
`a3c528f1b4c9b0a60f5cd3aeec28a11e990044b3`已改为best-effort structured warning并增加closed-pipe反例，
exact Pod fresh checkout全文件=`52 passed, 1 skipped`（显式real-GPU skip）。
该修复不改学习语义，不hot-patch或重启live。

按`HANDOFF_TO_CODEX_20260808.md` §3，只保留可由独立反例击穿的plant/finite/full-key/optimizer/
durable WAL+ACK边界；task success、R07 ready、stdout与same-writer echo不是安全Gate。结构逐步收敛为
`PlantFacts → ActionBallState transition → StepTelemetry`，backend physics与WAL/ACK留在边界，每次抽取
做fixed-tape parity；不以一次巨型重写制造新的多真源。

live clean SHA、host/Pod CPU/CUDA与EPA48/RSL3、双launcher dry-run及fresh Isaac/MuJoCo 1/5 ACK已经按上面
闭合；旧a103与V4的Pod/ACK仍不代签V5。正式matched-strata/fixed-tape physics parity、自然contact/landing、
独立playback denominator、12500 completion与transfer仍**pending**。本节只确认诊断发车与可信短前缀，
不授权resume、promotion、export、部署或真机。

### HISTORICAL / FROZEN — 2026-08-23 current adoption table

| 项 | 当前裁决 | 原因 |
| --- | --- | --- |
| learner | 采用H48、GAE `lambda=.98`、E5/MB8、entropy 0、fresh sigma `.02` | H48是学习取舍；约6秒只是速度方向，须另用profiler-off matched wall验收 |
| actor/critic | 保持semantic `203/219`与`history_length=0` | 已含必要root/task/teacher/clock；无alias反例不扩大部署与normalizer合同 |
| curriculum | 保持balance→mimic→hit→landing自然交接 | 上一阶段开始成功即应开放下一阶段分母；不增加硬Stage、balance Reward或R07 admission |
| reference | 采用reset-ready → post-transition selected frame0 → completed-action frame0三段selector | 使`action_t`、reward与返回Observation的因果一致；不新增offset |
| lifecycle | 采用post-settlement due分类、双phase解耦 | 只让仍busy row DEFER；terminal overlap不public；保留合法same-tick launch/outcome |
| fault/evidence | Isaac36与Mu4分开，统一进入各自唯一pre-optimizer drain；wire为6/5/5 | 具名根因可审计；不增恒0或same-writer Gate |
| safety | 保留finite、plant、full-key、optimizer与durable WAL/ACK；拒绝task成功、R07、stdout和same-writer echo作为Gate | 前者保护可独立击穿的可信事实；后者是学习结果、镜像或同源自证 |
| structure | 增量收敛到`PlantFacts → ActionBallState transition → StepTelemetry` | 减少可变真源；不以巨型重写替代逐步parity |

> **2026-08-21 portable successor事实纠正（supersede下文旧229/399、H24与active-run叙述）：**当前
> MuJoCo portable Full-A的执行真源是
> [EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819](EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md)。
> 本页后文把R06/R07/Reward11--13写成尚不存在、或把zero-action下的table/fall/contact零值及额外
> keepout witness写成长跑前置，均只保留为历史诊断，不得覆盖当前代码事实。当前host已闭合raw-action
> proposal/runtime order/scale/default-offset、R06/R07/Reward0--20、thin ledger和独立consumer；MuJoCo
> hard-range与Isaac soft-inset/brake仍显式`DIVERGENT_DECLARED`，所以没有transfer/matched authority。
> 原有exact table keepout保留，错误的额外witness已拒绝。
> true Gym reset现为runtime default plant/zero velocity/zero action history；30 s / 1500-tick cadence
> 的due是`2 + 293k`，每次按live state作`ACCEPT/DEFER`。HOLD joint teacher为default/zero velocity，
> body/R07 target为measured frame0；phase只允许`0/2/5/6/8`。natural `shot_retired`保留
> robot/action/episode/generation，只有Gym done发布`selected_reset`并增加generation。ledger为26 events，
> `completed_action_epoch`必须来自同一env行的完整业务闭合，不能由跨env边际拼接；
> consumer分开engineering与slot0 business完成；本代因73动作、双侧和科学窗口报告未闭合，
> `full_a_complete`固定为`false`。
> r3曾真实运行到durable ACK `10249`，终段last-100 wall mean/median=`4.890/4.886 s/update`，随后由
> MuJoCo-Warp `EPA_HORIZON` overflow fail-stop；旧229-D IDLE clock也会随global step漂移。因此r3已停止、
> 无resume authority，不能继续25k。`ACCEPT=0`只记录当时balance/mimic/readiness课程阶段，不是失败门。
>
> successor采用H48/U12500/save500/E5/MB8/lambda=.98和semantic A203/219。Observation V2恢复历史
> fixed-194/A211已有、229迁移遗漏的table-relative root XYZ/heading/COM velocity，同时删除raw task与
> owner/fault/reward账本；R06 broad projection缩为owner内canonical key8+publication join后输出live slot和
> 三枚latch。它们已是host实现面，但exact GPU observation/snapshot parity仍`未测`。EPA48的exact
> source/build chain已有一次真实`PASS_BUILD_CHAIN_ONLY` host receipt；无CUDA/隔离runtime时形成的通用
> search/capture WIP因体量大且只能同源自证而被拒绝。r3 exact pair未保存，本轮也没有live GPU或
> CUDA-qualified synthetic fixture。即便以后捕获，它也只关闭stock反复exact
> `EPA_HORIZON`-only overflow / fork反复zero-overflow finite active contact差分，不能代签ActionBall
> fixed-tape parity或instrumented/ASan独立oracle；三者未闭合前仍不授权训练。
>
> 本次live SSH认证失败，未取得当前Pod/GPU状态，所以本文不声明GPU空闲或存在active process。下一步只按
> [当前长跑TODO](../../operations/action_ball_dual_backend_longrun_todo_20260819.md)执行：Observation原子闭合
> → EPA fixture/GPU/oracle → 单一ActionBallState/zero-live skip → matched H48性能 → fresh namespace。
> Gate继续`Partial`，authority继续`diagnostic_unauthorized`。

## 2026-08-19 4096 scene首证据与successor裁决

commit `53156327…` 首次越过资产与runtime identity，真实构造4096 env、229/399 observation和Reward20；
但约25分钟后在runner构造发现compact joint-safety producer identity不满足新adapter，PPO/WAL为0。
因此本次只回答“4096 scene可构造”，没有return、contact或学习趋势。失败namespace永久封存。

耗时主因不是物理步而是homogeneous replicated scene仍对4096份继承内容重复做Mesh/collision/BBox审计。
successor采用IsaacLab真实`replicate_physics=True`/env0 source contract：完整组成内容只审env0一次，所有env的
具体prim、K-ball rigid actor和contact binding仍逐行验证。它删除重复工作，不放宽per-env runtime身份。
另将Motion catalog从旧plant bank切到同一0807 A3P plant重解的73条bank；逐文件根`pelvis_link` frame-0
yaw均在`1e-6 rad`内，旧日志是把`torso_Link`合法挥拍yaw当grounding的错对象gate。新bank机械admission仍
为`0/73`，只改善诊断lineage，不授权formal。compact的唯一writer本来就在`train.py` runtime binding；
commit `75b18ba2…`错误地又从task cfg写了一次，因ownership门在scene前自然RC1，证实不应复制状态。
回读exact IsaacLab表明live action class继承`ManagerTermBase(ABC)`，其metaclass必为`ABCMeta`；旧adapter却
要求`type(class) is type`，所以531 run拒绝的是错误的metaclass gate，不是compact ledger缺失。successor
保留runtime唯一writer，只把class检查改为“是type的实例+exact module/name+live object identity”，随后直接
fresh `4096×25000`，不再插小N。

## 2026-08-18 scale与native A1000最终裁决

小环境数边界纠正：`N=2`只回答构造、reset、ABI、finite和optimizer调用点，不回答学习效果。旧Isaac
`N=2` A1000在ACK470前虽有22560个finite Reward sample，却是296/296 episode全base tilt、260个
admitted opportunity全not-ready、0 ACCEPT；随后LM device assert终止。它是结构诊断，不是失败或成功的
4096学习实验。下一条科学run直接使用同一冻结进程`4096×25000`，update0--4只读scale门，健康后不重启，
里程碑为20/50/100/200/500/1000/2500/5000/10000/25000。

native MuJoCo A的`1024×1000`已自然完成，但只能回答简化MDP的趋势：

| update窗口 | binary racket-ball contact | mean minimum distance | 解释 |
| --- | ---: | ---: | --- |
| 50--100 | `6/8452 = 0.071%` | `0.461 m` | 稀少早期接触 |
| 100--200 | `32/17025 = 0.188%` | `0.462 m` | 仍很弱 |
| 200--500 | `361/51243 = 0.704%` | `0.431 m` | 开始靠近球 |
| 500--1000 | `4078/87546 = 4.658%` | `0.325 m` | 有方向性，但未形成合格任务 |

最后窗口吞吐约`15.2k environment steps/s`、约`1.62 s/update`。这不能与Isaac `N=2`的约
`5.13 steps/s`作引擎等任务对比：native lane一次并行1024环境，只有114/114-D observation、10-term
Reward，缺WAIT/reveal/ActionEpoch/outcome/recovery；Isaac则让固定Kit/PhysX/FullMDP transaction和WAL
开销压在2个环境上。更重要的是native最后窗口robot-table episode fraction仍约`97%`，因此只采用其
吞吐和“接触率能上升”作为下一代设计证据，不采用为portable A成功。

下一代工作不等长跑结束：Isaac 4096运行期间并行闭合portable MuJoCo A的question→reveal→flight→
outcome→recovery lifecycle、准备rough独立课程，并从zero-callpoint/旧RSL2/重复receipt开始做2.0删除清单。
任何并行改动只进入新commit和fresh namespace，不热补正在训练的process、scene或manifest。

third 4096 one-shot回到normal direct-script入口后不再segfault，AppLauncher完成约10秒正常初始化；首错变成
post-App identity门错误地拒绝AppLauncher合法加载的Torch。该run仍未进入scene/PPO/WAL，因而只证明启动形状
修复有效、门的对象过宽。下一件不增加gate：pre-App继续禁止Torch/RSL/TensorDict，post-App只禁止policy
runtime的RSL/TensorDict，并用既有版本/来源检查验证App-owned Torch。

fourth one-shot越过该范围修正后，Kit Python因未导出`F_SEAL_*`符号别名在同一门拒绝；memfd与preexec
seal证据未漂移。改法是直接消费Linux uapi的`F_GET_SEALS=1034`和固定seal bitmask，而不是删除检查。
长跑预算也从“1000个PPO update的早期趋势”恢复为历史同级的25000 update终局：单进程
`4096×25000=2,457,600,000` transitions，1000只读里程碑不停机。portable MuJoCo的R03/R06/R07与
Reward0--13仍缺，继续明确`full_a_complete=false`，不因Isaac发车而假称两边齐备。

fifth one-shot（`6b3078bc…`）在host serializer contract、preexec与AppLauncher约10秒正常启动后，因
post-App把sealed fd18必须保持`sys.path[0]`当成identity而RC1；实际Kit/AppLauncher会前插不含`rsl_rl`的
extension path。该run仍未进入scene/PPO/WAL。下一版对实际消费的root/package/leaf逐一核live resolver
loader/archive/prefix/origin指向fd18，再逐module/class核source与wiring；不替换或锁住Kit全局import机制。

长跑同时需要保留可评估策略，但当前owner graph还不能安全resume。采用最窄边界：upstream RSL仍序列化policy、
optimizer、iteration/model state，文件强制命名为`model_N.diagnostic_nonresumable.pt`，payload明确
`checkpoint_authority=false`、`resume_authority=false`，而lean runner的`load`继续拒绝。每1000 update和
自然终点留一份；save返回后生成绑定同一文件size/SHA、iteration、payload kind与两项authority=false的
no-clobber sidecar；签收前以`weights_only=True`从同一fd重新核真实payload四键、iteration、非空
model/optimizer state和全tensor finite，prelaunch另跑冻结RSL3真实serializer contract。终局reader反绑exact
experiment/run_name并要求26组逐份一致。它们可用于离线权重检查/后续显式评估，不是完整checkpoint或恢复授权。

## 2026-08-18 科学裁决：先真实运行，不再扩骨架

Jiayi/build_2的可复现说明与Pod2实机证明，旧Isaac4.5/RSL2和新Isaac5.1/RSL3不是等价学习环境。
因此采用精确5.1/8320/RSL3栈，并将FullMDP训练面缩成upstream RSL3 loop加一枚optimizer-boundary
adapter。真实Kit N=2 reset/readback已过；Pod1 GPU1 fresh A canary进一步证明真实环境能安装
229/399-D observation与20-term Reward，但在PPO/WAL前因IsaacLab8320 manager cfg的`dict` API与旧
history consumer不匹配而RC1（日志SHA=`40c6631…5051b9f`）。这是迁移首错，不是Reward或学习反例；
其exact-dict修复后的第二个fresh namespace又在PPO/WAL前暴露schema-3把FullMDP finite-`q_des`
误判成legacy ActionBall-only（日志SHA=`dba73962…e5e99a1`）。第二窄修后第三个fresh namespace继续
在同一structural boundary fail closed：live manager已安装229/399-D，但runtime fact writer漏掉
FullMDP critic block，故exact validator拒绝（日志SHA=`9e6b066c…35b8a2`）。这三项都是按真实调用顺序
剥出的环境迁移/合同接线错误，没有一次进入optimizer或产生WAL，不是Reward或学习反例。第三修只让
code-owned ActionEpoch actor从live critic manager写唯一399-D事实。第四个fresh namespace随后自然RC0：
`N=2 × 2 update`完成exact两个optimizer update和四行严格PENDING/ACK WAL；update0/1各48个Reward
sample全部有限，实际Reward和=`3.1311374828/3.0717186332`，无poison/nonfinite，平均约
`8.46 s/update`。这是第一份真实training-loop闭合证据。

这两次更新尚无学习结论：总共只有2个forehand opportunity，均selected后defer/not-ready；contact、
flight、R06 outcome与R07 recovery的eligible denominator仍为零，14个稀疏/任务项收入也为零，只有6个
dense motion项为有限正收入。当时据此启动的同代码`N=2` A1000现已由本文件§scale裁决降为历史结构诊断；
它后续到ACK470的节点仍保留如下，但不再决定下一实验规模。2/5/14/60 update只回答工程链，不回答可学性；
下一条科学实验改为同一进程`4096×25000`；1000只是早期趋势节点，不是终点。

### A1000 update20

- 运行可信度：20 updates=`960` environment steps，WAL `40`行严格PENDING/ACK，Reward sample
  `960/960` finite，nonfinite/poison/conservation violation均为0；wall=`00:03:08`。
- 任务入口：D05 transactions=`481`，due/selected=`12/12`，construction/key admitted=`11/11`；
  其中11个forehand opportunity均defer/not-ready，另1个unknown opportunity reject。side strata当前
  producer明确`not_produced`，所以side结果为`未测`。
- 事件梯：r03 first physically valid和physical observed/contact均`0/11 key-admitted`；没有launch、
  playback、R06 settlement或R07 outcome，因此R06/R07成功率没有分母，写`未测`而不是0%。
- episode：completed=`10`，length sum=`876`，10个均base tilt，rolling mean length=`87.6`；这仍是
  随机早期policy的可学习终止，不触发停机。
- Reward20：14个task/sparse项eligible=0；6个dense motion项收入依次为anchor-pos `5.6696`、
  anchor-ori `4.9781`、body-pos `10.4131`、body-ori `0.0489`、lin-vel `16.5698`、ang-vel
  `18.3530`，合计configured income=`56.0325`。policy noise std=`0.0200157`，LR=`1e-5`；KL未产出。

裁决：链路可信但尚未进入swing/physical event，不改Reward、不停机，继续看50/100节点能否把
admitted opportunity推进到playback与contact。

### A1000 update50

- 运行仍可信：2400 steps、100行exact WAL、Reward `2400/2400` finite，0 nonfinite/poison/fault；
  wall=`00:07:51`。
- selected=`28`，其中25个forehand construction/key admitted后全为defer/not-ready，另3个unknown
  reject；r03/physical=`0/25 admitted`，仍无launch/playback/R06/R07分母。
- 26个episode全部base tilt，rolling mean length=`86.38`、mean reward=`4.986`；相比update20的
  `87.6/5.056`没有改善。六个dense motion收入合计=`138.896`，每sample约`0.05787`，也与update20
  的约`0.05837`基本持平略降。noise std=`0.0200191`、LR=`1e-5`、KL仍未产出。

裁决：目前是“readiness尚未被policy达到”，不是已证Reward权重根因；不改配置，继续update100。

### A1000 update100

- 科学边界：只消费WAL中update0--99的100组完整`PENDING-v2 -> EPOCH_ACK-v2`；前缀为
  `1,297,132 B`，SHA256=`fdce022b…4606ed9`。累计4800 steps/200行WAL，Reward
  `4800/4800` finite，nonfinite/poison/conservation violation均为0，actual sum=`276.6504531`。
- 任务入口：D05 transactions=`2401`，due/selected=`58/58`，construction/key admitted=`54/54`；
  54个forehand全部defer/not-ready，4个unknown reject，ACCEPT=`0`。因此playback、launch、r03、
  physical contact、R06 outcome和R07 recovery仍没有合法分母，保持`未测`。
- episode：56个episode全部base tilt，length sum=`4780`，mean length=`85.36`。单看50→100窗口为
  30个episode、mean length=`84.47`，没有相对A50改善。
- Reward：14个task/sparse项仍eligible=0；六个dense motion累计income依次为anchor-pos `27.7644`、
  anchor-ori `24.5404`、body-pos `51.5558`、body-ori `0.2171`、lin-vel `81.2119`、ang-vel
  `91.3609`，合计=`276.6505`，约`0.05764/sample`。A20/A50约为`0.05837/0.05787`，当前是平坦略降，
  不是已证权重根因。
- TensorBoard step99只作learner辅源：value loss=`5.34370`、surrogate=`-0.07934`、entropy=
  `-77.2432`、noise std=`0.0200275`、LR=`1e-5`；KL与全optimizer/normalizer state finite receipt未产出。

裁决：100个update仍没有swing，方向性尚差，但按预注册1000预算这不是停机条件。继续到200，届时优先
判断admitted-but-not-ready是否开始松动；若仍为0 ACCEPT，才把“readiness/模仿入口可能结构性太难”提升为
需要因果诊断的候选，而不是直接改Reward权重。

### A1000 update200

- WAL update0--199为200组完整PENDING/ACK；前缀`2,597,276 B`，SHA256=`6132cf21…1c82045`。
  9600个Reward sample全部finite，0 poison/nonfinite/conservation violation，actual sum=`547.6872326`。
- D05 transactions=`4801`，due/selected=`118/118`，105个forehand admitted后全为defer/not-ready，
  13个unknown reject；ACCEPT、playback、launch、contact、R06和R07仍全部没有合法分母。
- 116个episode全部base tilt，累计mean length=`81.56`；100→200窗口60个episode、mean=`78.02`，
  相比A50/A100继续下降。dense configured income累计=`547.6872`，约`0.05705/sample`；窗口=
  `271.0368/4800=0.05647/sample`，同样没有上升。
- TensorBoard step199仅作learner辅证：value loss=`0.01449`、surrogate=`-0.03678`、entropy=
  `-77.2043`、noise std=`0.0200528`、LR=`1e-5`；collection/learning=`7.72/0.40 s`，环境采样占主导。

裁决：这不是数值故障，长跑继续500/1000；但“200 update、105个独立admitted opportunity仍0 ACCEPT”
已经把readiness producer/阈值或可达性提升为明确的因果审计对象。审计只能读取冻结source与live WAL，
不得热补active run；只有证明readiness输入恒不可达或奖励在该入口前没有有效梯度，才为下一条fresh run
改合同或经济。

后续只读因果审计把该候选拆成两层。截至ACK437，242个admitted仍全部not-ready且R07 first-ready=0，
所以现役run还没有触发协议fault；这是当前policy/plant尚未满足13项两拍readiness的观测事实。独立源码
反例同时证明，旧bytes即使第一次满足bootstrap readiness，也会用neutral shot key发布只允许full-key的
R07 event并sticky overflow。该第二层是结构错误，不是学习结果。采用的下一条fresh修复只mask bootstrap
的shot-keyed telemetry，Motion readiness和completed-shot telemetry均保持。CPU生产链已从两拍真实
plant facts走到owner projection、Motion reveal和D05 settle，得到两行ACCEPT且Epoch无overflow；
现役A1000不热补，也不再能到500/1000：最后完整边界为ACK470（零基update469、22560 steps），随后
fixed-try LM在`solve_ex`后因`info!=0`或非有限`dq`触发device assert；具体row、类别和更上游数值原因未
落盘。同一坏context随后令PhysX view与D05 poison attribution失败，子进程保留为停滞态。WAL没有悬空
PENDING，故0--469仍是可信负证据；500/1000和Reward趋势均为`未测`。下一实验必须用fresh namespace
验证新的逐行数值拒绝：`solve_ex info!=0`和非有限`dq`不再参与候选选择，分别写入既有proposal ledger的
`lm_solve_info_nonzero`/`lm_solve_nonfinite`，不再使用会摧毁CUDA context的`_assert_async`。该修改只把
可预期的数值不可解改成可学习流程中的具名construction rejection，不把异常称为安全证明，也不改变Reward。
不能从后续PhysX/Python观察面猜因或重试现有namespace。
隔离进程回归为integration=`1 passed`、recovery-device=`80 passed`、live-facts=`25 passed, 6 skipped`、
Epoch rowwise=`51 passed, 7 skipped`；旧completed-action true-writer fixture也继续验证 keyed telemetry 正常发布。

本轮不预调Reward。A1000先记录20个term的signed income、opportunity/contact/flight/outcome/recovery
分母、termination reason和episode length；到1000后才区分权重失衡、触发率不足、分母错误或环境不可学。
历史C曲线中action penalty负正比约`10.39 -> 3.45`及episode-length谷底恢复只作为待观察模式，
不能直接复制权重。EXP只保留这类假设、指标、结果和裁决；命令、锁和namespace状态只在
[唯一执行 TODO](../../operations/action_ball_single_action_dual_backend_todo_20260817.md)记录。

MuJoCo M05骨架因没有真实`step/reset`成功路径已撤回。后续只读审计又确认当前WIP有73个未跟踪
Python文件、约12.4万行，并面向历史A211/C211 `211/319`；继续增量只会扩大无法Git交付的平行系统。
当前portable真源已由live Isaac固定为FullMDP ActionEpoch `229/399`。因此MuJoCo 2.0从tracked
`a3_train_ppo.py`的真实plant/reset callpoint重建：第一纵切片只做all-world reset后的initial-WAIT
229/399 TensorDict，硬上限500 production LOC；不迁移M04/synthetic VecEnv/receipt/schema。真实step、
Reward、termination和masked-reset lineage未闭合前，不声称接近GPU训练或执行`learn(1)`。

该切片的host实现为production净增253 LOC：共享合同只定义列序；Isaac与MuJoCo分别提供live tensor。
Pod1 GPU2随后从fresh Git `495a0870`执行唯一真实N=1 reset，Python3.12/Torch2.13/MuJoCo3.10/
MuJoCo-Warp3.10.0.3上自然RC0。policy229/critic399 finite，live robot rows与readback一致，ActionEpoch
phase为IDLE one-hot，+10m park球qvel为零且raw contact array中无ball row。result SHA256=
`6fe2e70c43fc239c3f4a041e3e349c991ff1fb4ff312c17266d66e0580ba6030`，log SHA256=
`2cc20c745293f39b5e46a1d10f19a066d5738648e85d8c02261bd6785901ba1c`；checkout保持clean。

科学裁决为`PASS-live-reset / HOLD-step`。attach warning本身不能代签最终model option；因此新commit
`d28a7eac`在新namespace直接断言live compiled `physics_dt=0.001`、decimation=`20`、control dt=
`0.02`和`noslip_iterations=0`，再次自然RC0，result SHA256=
`9b40214fe9f615f55cf0182b39eac7bcc1d91e22b4db083c567a20cd1e11eddc`。这证明SimulationCfg正确
恢复vendor timestep；`noslip=0`则是MuJoCo-Warp没有noslip pass的已登记backend deviation，不伪称
vendor exact。`/workspace/mjlab_venv`的RSL-RL仍为5.4.0而Isaac为3.1.2；任何`learn(1)`前须隔离并
锁定共同RSL3.1.2训练ABI，不能用兼容分支宣称等价。

下一纵切片保持同一root并净增396 production LOC，已在host接入真实plant step、六个dense Reward20、
四项shared termination和masked reset。三个能改变Reward的语义错误已先由反例关闭：raw MuJoCo `cvel`不是body
inertial-COM线速度，必须按`xipos - subtree_com[body_rootid]`做刚体点速度平移；且MuJoCo-Warp每次
`step`在integration后不会自动重算derived tensors，故policy边界要先`forward`再读Reward/termination。
body position/orientation imitation则保留Isaac的上一拍Motion cache，Reward之后才按最终live anchor刷新
下一拍。post-forward resolved contact只导出具名backend bool；它不是Isaac component-[OBB](../../DEFINITIONS.md#obb)
[SAT](../../DEFINITIONS.md#sat-collision-test) keepout，
后者仍阻塞trainer。mixed-nonfinite qdes逐关节回退、其余finite joints继续执行，同时raw qdes仍进入终止证据。
host组合=`8 passed, 4 skipped`，native旋转偏置body Jacobian oracle通过；
fresh Git commit `e71ee1a350d…` 在Pod1物理GPU2自然RC0：两份门合计`12 passed`，覆盖N=1
reset、非零step后derived-tensor forward、timeout与同tickReward20、park，以及同一N=2 world内selected-reset
对peer plant/buffer rows不改。result/log SHA256=`f4d41aa9…/5fa33b26…`；运行后GPU2和queue lock均释放。
因此WAIT纵切片改判`PASS-live-step`。这不是A训练证据：Isaac的component-OBB/table-AABB SAT keepout尚未
成为MuJoCo device producer，resolved contact只能作backend telemetry；实现并GPU反例验证前禁止`learn(1)`。

SAT producer现已在host落成，但尚未由GPU授权。它没有新增root/receipt/schema，而是在现有plant的
20-substep loop后加device-only observer，复用已独立验证的62-component/five-AABB construction authority；
runtime判据是fixed-shape 15-axis SAT。MJWarp的derived pose在integration后一拍才刷新，因此hook消费
state1--19，policy边界最终forward消费state20。测试不只比较同一套正例：额外固定一个45°旋转小盒，
其world-AABB空角与table相交而真实OBB分离，明确要求`broad=true/exact=false`，从而阻止退回旧
broadphase仍假绿。CPU float32/64、nonfinite、authority别名和时序反例通过；本纵切片production净增
249 LOC。科学状态为`PASS-host-SAT / HOLD-live-SAT`；fresh GPU门通过前不接trainer、不执行`learn(1)`。

fresh GPU整合先排除了“snapshot绝对路径必须等于开发机canonical路径”这一伪identity，随后把下一首错
定位为MJLab attach的确定性命名空间：canonical 32个A3 body在live scene统一加`robot/`。采用的修复不是
模糊查找或重新注册plant，而是唯一adapter显式声明该prefix；authority仍用canonical名称序列化并逐项比较
父子关系、local frame、root bytes与portable closure。这样同一几何在bare native model与attached MJLab
model中有同一语义，而不会把路径/名称表象误当plant identity。host namespace反例与device SAT分别为
`7 passed`和`4 passed,1 skipped`；live 19-test与RSL update仍为`未测`。

fresh commit `61887b43…` 随后在Pod1 GPU0共卡完成19个direct GPU测试、0 skip，并由同一clean checkout
直接执行upstream RSL-RL3.1.2：`N=2 × 24`得到一次PPO update、48 transitions、229/399 observation，
全部合同字段匹配，`final_rc=0`。result/log SHA为`322592ce…f07a`/`7d4fbee7…2a3b`。该结果关闭
SAT与trainer调用点，不回答可学性：receipt明确`idle_wait_only`，没有A action lifecycle、击球或回合分母。
下一纵切片只接真实A reveal→flight→outcome到现有VecEnv，不复制runner或新增第二套MDP；短live通过后才
启动MuJoCo A长跑。

trainer不再另起一套架构。新薄launcher只把现有WAIT TensorDict交给upstream RSL-RL 3.1.2
`OnPolicyRunner.learn(1)`，并在runner构造前后分别绑定module与实际PPO、ActorCritic、RolloutStorage、
Adam来源；有副作用的同名预载runner/algorithm不能先执行再被拒。host focused=`6 passed, 1 skipped`，
WAIT/SAT组合=`18 passed, 6 skipped`。
这证明真实production调用点与反例，不证明GPU update：唯一live test仍skip。下一实验只有两个串行问题：
fresh空卡先跑SAT/WAIT；通过后在隔离RSL3.1.2 overlay跑`N=2 × 24`一次update，检查exact一次PPO
`alg.update()`调用、48 transitions、229/399 finite。一次PPO update内部有20次Adam step，不能写成
一次optimizer step。该run仍是`IDLE WAIT`工程门，没有question、contact、outcome或
recovery分母，不能被解读成MuJoCo A学习结果。

2026-08-18真实整合首次在Pod1 GPU0以单卡双进程运行，进程树固定CPU `32-47`且启动前free显存超过
20GiB。它通过环境、输入快照、RSL3安装和16个direct测试；剩余3个真实env测试均被同一绝对路径门拒绝，
所以没有进入RSL/PPO。A3 snapshot的77个文件、总字节及逐文件SHA与clean Git canonical树完全相同；
绝对路径不同不构成物理差异。采用的fresh修复以root SHA、portable closure与live owner-frame取代path
equality，保留不同内容fail-closed。该失败不说明Reward或MuJoCo-Warp不可运行，状态仍`HOLD-live-RSL3`。

本分支同时保留 `L194` legacy fixed-question
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

> **2026-08-07 起，"下一步做什么"以 §5.6.17 为准。** 那一节把 08-06 三条独立 review
> （随机性 / MuJoCo 对齐 / 复查漏做的）合并成一份去重、消解矛盾后的清单，
> 分成**发车前必须做 / 发车后补 / 记着但暂不做**三档，并单列 `build_1` 加桌子的交接单。
> 本表各行的"下一道可执行门"仍然有效，但**排序**看 §5.6.17。
> 同批就地更正：§5.3 / §9.2.9 / §12 三处把死亡罚写成 post-dt `-6` 的旧数
> （活值是 `-0.2`，weight `-10`），以及 §5.6.13 (A) 那句"四格 `scale4096` 正在发"（**没有在发**）。
>
> **§5.6.18 是 §5.6.17 的验收**：§5.6.17 读的是别人的收据，§5.6.18 把每一道声称
> "已接线"的门改回接线前的样子实测（`15` 个变异全红），并核了每一条"删了因为别的机制管"
> 的那个机制今天真的在跑。两处新发现：**平时跑套件的那个 venv 缺 `cryptography`，
> 课程 launcher 的 `53` 条护栏测试在历次全量对拍里一次都没执行过**；
> **零调用点的门有第六个**（§5.6.13 (C-2) 原写"没有第六个"，已就地更正）。

| 关键路径 | 当前事实 | 下一道可执行门 |
| --- | --- | --- |
| **termination/reward 对齐（2026-08-05 新增，明细见 §5.6）** | 反向审计发现 A211 运行时 `42` 个非零 term 而 §5.3 只覆盖 `22` 个；三项零命中项压过主层级。已落字节：`joint_actual_forbidden` 改 `terminate=False`（只记账不 reset，telemetry 模式强制证据记录器）、`ee_body_pos` 去腕只留脚、`upright_exp 1.0→0.25`、`hit_unstable_support -10→-1`、`death_penalty -300→-10`、`undesired_contacts` 正则 `_link→_Link`（**bug**：A3 是 `_Link`，原为 G1 命名，双脚双腕反被罚 `-2.0/episode`）、soft-limit v2 两条通道带宽 `0.08→0.05`（`qdes_limit_barrier_margin_frac` 与新增的 `joint_limit_margin_frac`，消除护栏自造的 `-0.0844/关节/步` 底噪）、`init_noise_std` 四处硬钉解开且 4σ 门改为按真实 σ 计算（原为字面量 `0.02`，**假绿**）。MuJoCo 侧 `joint_actual_forbidden` 已同步 | 重跑 A/C focused suite；~~让 `audit_action_ball_reward_hierarchy.py` 接受 DRL0 leaf~~（**2026-08-06 更正：已于 `635252f6` 接受**，见 §5.6.13 (D)2）**并重算全部静态数值**（这半句仍欠，至今无对 DRL0 leaf 重算的收据）；`counter_rally_v1` 与 `virtual_landing` 的口径差待裁决 |
| observation/reward | A/C=`211/319`；无 teacher-base；唯一 actor 角速度是 body-frame IMU gyro；C 只有 nominal-strike 拍心距离与`vb_fired` selected-rubber swept analytic contact-gated单次落点；`physical_ball=false`时不是PhysX observed landing。`.99/.95`保持A3/BeyondMimic/mjlab基线。runtime/training-contract已安装fixed-N1 A `base_position 1.5→0`、九个window项`×1.15`、C proximity`240`、A/C landing`700`；progress10保留。按Take061 task-valid折扣账，A `1.773<1.852≤3.009<3.332`，C `1.773<1.904<3.332`。C launcher与oracle的旧`v2/220/500`当前已改成`v3/240/700`，等待focused test确证 | ready/swing mimic ledger和schema-3 runtime交叉检查已过；补C fixture/live ledger全链、landing∧post-contact-fall监控；真球另走promotion |
| reset/teacher | direct frame0 physical birth=`0/73`；split-ready artifact+`60/240` hold 已有；WAIT 5--25 tick期间机器人/teacher保持split-ready、球停在无接触park位；reveal原子安装来球并切measured frame0，机器人不reset。A/C leaf显式钉`backhand`。旧`.7123759904781779 s`来自 tracked interior receipt(`r4_splitready`, tick92)；A literal-center已收紧到TTC=`1.82`/tick91/wait=**`.6923799138976297 s`**，A/C materializer分别=`12/12`、`11/11`；C走独立family-C receipt。**2026-08-05 更正**：本行此前写作`.69237599 s`，那是从旧 tracked receipt 减一 tick 推出的；仓库存在两份相差`3.9e-6`的 interior 权威——旧 tape(`r3`/`r4_splitready`)receipt 的`0.7123759904781779`与现役 tape(`fresh_592835dc_take061/rematerialized_1d5d9d44`)receipt 的`0.7123799138976297`。producer 从 prepared core 重算出的正是后者，故以现役 tape receipt 为准，旧数只作历史。**2026-08-06 更正**：此前本句写作"以 code-owned 常量 `CANONICAL_TEACHER_PROJECTION`(`:201`)为准、launcher 导入期自检比对的是后者"——那条描述有误：该常量只是把同一份 tape receipt 的字节在代码里抄了一遍(22 个 `runtime_target` 字段逐一相等，3 个顶层键只是改名)，所谓"导入期自检"仅是常量与自身 sha 的循环比对，从不读磁带，因此对漂移零保护。该常量与 `CANONICAL_BASE_QUESTION` / `CANONICAL_SOURCE_TAPE` / 两个 validate 函数已于 2026-08-06 整体删除，唯一权威改为 tracked 磁带及其 `current_lm.target.task_receipt.v5.5e09858672ac.json`。误删helper已恢复。bridge schema-v3专项=`56 passed`，但共享4096 gate仍写schema-v2，A launcher在第84项正确fail-closed，未被绕过 | ~~把shared gate升级为严格消费v3 `reveal_to_playback_bridge`~~**（2026-08-07 已接线，§5.6.22）**与唯一counter表；随后A/C launcher余下测试、旧interior负例→exact-source suite→S0/S1 |
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

> **2026-08-07 就地更正：上面这张层级表只有收入，一行罚都没有，而罚才是这个经济体的大头。**
> 本节从头到尾按"折扣 per-swing 收入"排序，这个单位本身没错；错在 dense 每步罚项**从来没有
> 被换算进这个单位、也没有进过这张表**。用同一把尺子补上（`gamma=.99`；按 C0/C1 实测每步终止率
> `1.82%` 反推平均 episode ≈`55` 步、折扣和 `42.2`，整回合 `500` 步则折扣和 `99.3`）：
>
> | 项 | 实测 post-dt 每步 | 折扣到 §5.3 的单位（55 步 / 500 步） |
> | --- | ---: | ---: |
> | `qdes_projection_penalty` | `-0.19401` | `-8.19` / `-19.27` |
> | `qdes_limit_barrier` | `-0.03926` | `-1.66` / `-3.90` |
> | `action_rate_clamped` | `-0.03600` | `-1.52` / `-3.58` |
> | `joint_limit` | `-0.01716` | `-0.72` / `-1.70` |
> | `joint_torques` | `-0.01417` | `-0.60` / `-1.41` |
> | **负向合计** | **`-0.30847`** | **`-13.02` / `-30.63`** |
> | **正向合计（同口径实测）** | **`+0.01872`** | **`+0.79` / `+1.86`** |
>
> 也就是说：**这张表的天花板（合法上台下界 `3.33209`）只有 dense 罚的 `1/3.9`~`1/9.2`**；
> 而实测的正收入连表里最低那一档 task-valid mimic `1.77331` 都只兑现了 `45%`。
> 单位是对的，**分母漏了**。取证与三方对账见 §5.6.20。

这个定价会带来一个必须单独监控的 safety-economy 风险：death 一次 post-dt=**`-0.2`**
（weight `-10`；**2026-08-07 就地更正**，本句原写 `-6`，那是 weight `-300` 时代的旧值，
取证见 §5.6.17 矛盾 1 与 `action_ball_211_four_grid_contract.py:347`），
而 legal landing floor=`+8.4`，所以“合法上台后同回合摔倒”的最低事件净值是 **`+8.2`**（原写 `+2.4`）。
**这条更正把风险放大了 `3.4` 倍，不是缩小**：按活值算，"打成一次再摔"几乎不亏钱。
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

   > **2026-08-07 就地更正：这条 Gate 第一次有实测答案了，判定是"没通过"，而且它在发射前
   > 一次都没被执行过。** C0/C1 `scale4096_s15r1` 实测：regularization 层每步 `-0.30847`，
   > 主任务三层每步 `+0.01872` —— **罚是主任务的 `16.5` 倍**，其中 qdes 两项独占罚金的 `75.6%`。
   > "不能在常见轨迹上淹没三层主任务"这句话现在有了可证伪的读数，而**当时没有任何门在读它**：
   > 仓内三处叫 "reward economy" 的门全是字节漂移门，只验"配置权重没被改过"。
   > 同时更正本条隐含的一个口径漏洞：本条只说"逐项报告"，没说**跟谁比**。正确的分母是
   > **全相位每步正收入合计**，不是击球窗内峰值收入 —— 后者正是 §5.6.2 第 9 条踩过的坑。
   > 完整对账（含 `build_1` 同 iter 对照与尽调处方）见 §5.6.20。

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

> **就地更正（2026-08-07，§5.6.24 取证）：上面这段把"被钳关节常驻软带内、恒扣 `-0.0844`"
> 当成了 `0.08` 独有的病，因此以为改到 `0.05` 就治好了 —— 这是错的。**
> `0.05` 只是把"必然发生的底噪"换成了"由浮点舍入决定的底噪"：barrier 带外沿
> （`m_eff = 0.05`）与被钳关节的落点（投影内沿 `0.05`）**恰好是同一个点**，31 个关节里
> 有 29 个的 `m_eff` 正好命中它，于是 `intrusion` 是正是零由 1 ulp 的浮点抖动决定，
> 而旧核在那一点是 `0 -> 0.25` 的跳变。带宽已按 Franco 2026-08-07 附加裁定改为 `0.02`
> （严格窄于内沿，被钳关节确定性零罚），同批把地板从软带挪到机械硬限位。见 §5.6.24。

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
> **另一条路**回答的：卡点在腿的几何（出生姿态与 clip 是两个站位、差 `0.350 m`，
> 而开环位置指令不会迈步），不在桥；
> ——**注意（2026-08-07 四方复核后就地更正，见 §5.6.7「十」）**：这里原先还列了
> 「`57/57` 帧脚底离地、`35/57` 帧质心出支撑多边形」，现在的账是：
> 脚底离地成立（全库 `73/73`，是导出时没有地面约束），但**一次 `2.44°` 的接地收尾解算就修好**；
> `35/57` 失衡**证伪**，是量具假象（诚实口径 `0/57`）；
> 站宽差那一项**异常的是出生姿态不是 clip**（clip 与动捕真人按腿长归一化只差 `0.9%`）。
> **所以别再把这段读成「那条动捕 clip 不可用」。**
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

> **先读这一段再读下面（2026-08-07 四方独立复核后的最终裁定，全文在本节末「十」）。**
> 本节初稿的结论是「那条 measured clip 重定向坏了，所以 `oracle32` 被拒」。**这个结论是错的。**
> 现在的账：**(1)** 重定向只有一个缺陷 —— 导出时没做接地收尾，全库 `73/73` 条的 frame 0
> 悬空 `6.46 .. 21.26 mm`；**一次 `2.44°` 的接地解算就修好**，站宽只动 `0.77 mm`，
> 修完之后仓库自己那台最严的静态门对这条 clip 判 `50/57 FEASIBLE`、frame 0 `FEASIBLE`。
> **(2)** 「质心出支撑面 `35/57`」**是量具假象，不存在**（诚实口径 `0/57`）。
> **(3)** 站宽异常的是**出生姿态**（`0.2613 m`，只有真人归一化站宽的 `43%`），
> 不是 clip（与真人差 `0.9%`）；**曾经报出去的「这条 clip 站得太开」方向反了**。
> **(4)** 那道双支撑地面 LP 是**静止**证书，历史上只放行过腿被冻住的 clip
> （冻腿 `8` 条 `100%`，自带腿 `5` 条 `8.0%`），拿它判一条挥拍中的动捕参考是范畴错误。
> 下面「一」到「九」按时间顺序保留了推导过程与各自的更正标记；**有冲突时以「十」为准。**

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

**二、卡点是几何/静力学，跟力矩无关；但下面这三行里有两行本节初稿判错了对象。**

> **就地更正（2026-08-07，Franco 质疑「A 选的动作在重定向前后差那么多吗？这是动捕出来的动作啊」后复核）。**
> 初稿默认「出生姿态是对的、clip 是错的」。逐条查下来，**这个默认是反的**：站姿那一行箭头指错了方向，
> 质心那一行是量具自己打出来的假象，只有脚离地那一行成立、而且它是**全库共有**的一个标量偏移。
> 复核证据见本节末「六」。下表保留初稿量到的三个数、另加一列「裁定」；被证伪的那一行把
> 初稿数字标成「原记法」并列出诚实口径，方便对账。
>
> **再一次更正（同日稍晚，见「七」）：本节所有毫米数——包括「六」1~5 项里的——都还差 `1.12 mm`。**
> 审计工具当时把 ankle_roll 上的**纯视觉网格**也算成了鞋底，而那块网格比真正参与碰撞的网格低
> `1.12 mm`。工具已改成只取碰撞网格；换算后的活值一律以「七」为准。方向没变，两个结论更强了：
> 悬空比记的更多，质心「出界」比记的更少。

| 问题 | measured clip（`57` 帧） | split-ready 出生姿态 | 裁定 |
| --- | --- | --- | --- |
| 脚踩没踩到地（**碰撞**鞋底最低顶点离地板） | `+12.92 .. +19.60 mm`（左 `+18.37..+19.60`、右 `+12.92..+15.16`），`57/57` 帧全部悬空、`ncon = 0`（原记法 `+10.5 .. +12.9` 量的是视觉网格，见「七」5） | `-0.33 mm`（压着地，`sole_floor` 门 PASS） | **成立，但不是这条 clip 的事** —— 全库 `73` 条 frame 0 **`73/73`** 都不落地，是重定向导出时**没有地面约束**。**「只整体下沉」不够，「只把脚放下来碰到地」也不够**：必须**把脚放平**，因为这台 LP 要求每只脚至少 `3` 个不共线的真实接触点，倾斜的鞋底只碰到一条棱，见「十」3 |
| 站不站得稳（质心到双脚支撑多边形的有符号裕度） | **诚实的整鞋底 footprint 上：`+86.2 .. +128.7 mm`，均值 `+109.6`，frame 0 `+105.6`，`0/57` 帧在支撑面外** | `+133.6 mm`（`support_margin` 门 PASS） | **不成立** —— `-11.9` 是审计的 `SUPPORT_BAND_M = 6 mm` 撞上本 clip 左右脚 `4.8 mm` 高差产生的量具假象；换成碰撞网格后连这条最苛刻的规则都判 frame 0 **在支撑面内**（`+0.15 mm`）。三条独立复核给出同一组数，见「十」1 |
| 站姿宽度（两踝水平距离） | `0.6113 m`（`57` 帧内变化 `0.09 mm`——**这条 clip 全程不迈步**） | `0.2613 m` | **数字对，异常的是右边那个** —— 动捕真人两踝是 `0.5653..0.5693 m`；按腿长（两侧都用**链长**口径）归一化，真人 `0.7313`、clip `0.7378`，**差 `0.9%`**；全库 `73` 条是 `0.506..0.676 m`；`0.2613` = A3 十二个腿关节全零的 `0.2490 m` 加 `12 mm`，即**厂商 deploy 默认站姿** |

> **就地更正一句报出去的话（2026-08-07）。** 本节初稿、以及据它向 Franco 转述的
> 「站宽 `.611` vs `.261` 说明**这条 clip 站得太开**」，**是错的，方向正好反了**。
> 重定向把真人的站宽按腿长几乎逐比例搬了过来（差 `0.9%`），它是整条链路里做得**最准**的一项。
> 站得不对的是 `0.2613 m` 那一侧 —— 两脚几乎并拢的**厂商 deploy 默认站姿**，
> 它只有真人归一化站宽的 `43%`（`0.3153` vs `0.7313`），对一个要左右移动接球的机器人不合理。

**站姿差 `0.350 m` 仍然是真的，但它是出生姿态欠的账。** 折算成脚在骨盆系里必须走的距离：
左脚 `0.314 m`、右脚 `0.213 m`。clip 的两只脚各自外翻着站（`ankle_roll` 左 `-0.185`、右 `+0.186 rad`，
鞋底离水平 `3.6°/6.0°`），出生姿态是 `0.000°` 双脚平放 —— 这正是「预备站位」和「立正」的区别，
不是缺陷和正常的区别。

**为什么前馈救不了**：关节力矩是**内力**，它挪不开被摩擦钉住的、与参考差 `0.35 m` 的两只脚。
所以 §5.6.6 结尾那句「补上逐帧逆动力学力矩，`oracle32` 才可能可达」**仍然不成立** ——
力矩补上了，开环回放依然不可达。**但根因要改口**：不是「clip 悬空且失衡所以物理上不成立」，
而是「**出生姿态与 clip 是两个不同的站位，而开环位置指令不会迈步**」。这一条直接对上「三」的
单变量结果 —— 腿一发 teacher 就倒、腿保持 hold 就站着，因为发 teacher 的腿等于要求它在
不迈步的前提下瞬间张开 `0.35 m`。
顺带一条：仓库那套产出 WAIT `hold_qdes` 的双支撑地面 LP（`MujocoGroundContactLPSolver`）
对这条 clip 的 `57` 帧全部 fail-closed 拒绝（`both named feet must have active MuJoCo floor contact`）。
**那道门没错，但它拒的是上面第一行那个「没接过地」，不是「这条 clip 站不住」。**
把脚接地并**放平**（根高 `-24.0 mm`、十二个腿关节最大改动 `2.44°`、站宽只动 `0.77 mm`），
**同一台 LP、同一套默认容差，frame 0 当场判 `FEASIBLE`**；整条 clip `50/57 FEASIBLE`。实测见「十」3。

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

- **重定向要修的只有一样：导出时没做接地收尾解算。**（初稿写的是「retarget 没约束地面接触、
  没约束静态平衡、也没把站姿锚到出生姿态」，后两项经复核不成立，见「六」。）站姿和静态平衡 retarget
  都做对了；错的是导出这一步从来没把机器人放到地面上，全库 `73` 条 frame 0 **`73/73`** 悬空
  `6.46 .. 21.26 mm`。**修法就地更正两次**：初稿写「一条 clip 一个 z 偏移，不动关节角」——
  「七」3 实跑证伪（纯平移 `0/57`），「十」3 给出成立的那一种：**十二个腿关节 + 根高，脚放平是硬约束**，
  代价是腿最多动 `2.45°`、站宽最多动 `3.7 mm`。**注意不要写「站姿等于出生姿态」这条硬约束** ——
  它会把动捕真人的预备站位压成立正，是把唯一做对的那一项也毁掉。
  **另有一条独立缺陷**：`9/73` 条 clip 的 frame 0 关节角本身就出 MJCF 限位（名单见「九」1），
  接地解算修不了，要回重定向侧单独处理。**§12.3 的 `0/73` 机械准入结论不变**（那是另一套判据），
  只是它的原因里「悬空/失衡」两条要按本节改口。
- **该换的是出生姿态，不是 clip。** 但这条路有一个真实的前置约束：出生姿态的 `12` 个腿关节
  **逐位等于** `AGIBOT_A3_CFG.init_state.joint_pos`，而该表同时是 `use_default_offset=True` 的
  **动作零点**，与 C++ `pp_policy` lockstep；2026-07-26 `EXP-V2-REWARD-FREEZE` §0.13 已明令
  「禁止改资产 `default_joint_pos`」。改出生姿态**不等于**改动作零点，两者必须分开走。
  **换成什么，见「九」** —— 本条初稿写的是「给 ActionBall 谱系一个独立的 ready 姿态」，
  「九」查下来更省事也更符合原始设计意图的是**接地后的 teacher frame 0 本身**，不需要第三个姿态。
  无论走哪条，都必须重新过 `sole_floor`/`support_margin`/地面 LP/`nominal_hold` 全套静态门。
- **`oracle32` 该重定范围。** 它现在问的是「开环下发 measured `q_des` 能不能打到 `32/32`」；
  对这条 clip 这个问题**恒为否**（原因见上：站位差 `0.35 m` 而开环指令不会迈步）。
  建议拆成两道**都能失败**的门，替换掉一道恒假的门：
  - **运动学可解性证书**：逐 clip 判参考自身的地面接触 / 静平衡 / 站姿一致性。
    **判据要按本节复核后的口径写，不能照抄初稿那三条**：
    「悬空」按修完 z 偏移之后判（照初稿口径会一次拒掉 `71/73`，等于不判）；
    「静平衡」用两只脚**完整鞋底投影**的多边形判，不能用 `6 mm` 接触带
    （审计工具已在并行修正，新增 `com_footprint_margin_m` 与 `sole_lowest_vertex_z_by_foot_m`）；
    「站姿一致性」比的对象是**该谱系的 ready 姿态**，不是厂商 deploy 默认站姿。
    照新口径，`Take_061_unit04_BH` 现在只会因为**地面偏移**被拒，不会因为失衡或站宽被拒。
    （**「十」6(c) 就地细化**：这道证书要拆成「接地**可解性**」与「接地后的**静态站立**」两段，
    每段自带能两头开火的变异测试；判据表见「十」6(c)。接地之后 `Take_061_unit04_BH`
    连地面偏移这一条也不会被拒 —— 实测 frame 0 `FEASIBLE`、全片 `50/57`。）
  - **闭环策略可达性证书**：交给训练后的 policy 判，而不是零 PPO 的开环回放。
- **不放宽任何现有 fail-closed 判据。** 上面没有一条建议是降门槛：地面 LP 的拒绝保留、
  `sole_floor`/`support_margin` 静态门保留、`20 mm` 台面余量保留、`0/73` 机械准入保留。
  变的是「用哪道门去问哪个问题」，而且新门自带能开火的证据。

**收据。** 审计工具 `hope_training/whole_body_tracking/scripts/audit_measured_teacher_executability.py`
（变异测试 `hope_training/whole_body_tracking/tests/test_audit_measured_teacher_executability.py`，
pod1 `hope_isaac_venv` `14 passed`：初稿 `10` 条 + 「七」5 新增的 `4` 条鞋底选网格变异测试，
`0 skipped`）；初稿 JSON 报告 pod1 `/workspace/franco/s10_executability.json`，
**「七」更正后的报告 pod1 `/workspace/franco/fp_20260807/s10_executability_collision_only.json`**；
「七」的四条独立路径 / 跨库对照 / 地面 LP 实跑探针 pod1 `/workspace/franco/fp_20260807/p{2,3,4,5,6,7,8}_*.py`；
A/B 与执行器移植对照 pod1 `/workspace/franco/s10_probe11.py` `s10_probe12.py`；
worktree `/workspace/franco/s10_ff_20260806`（`33c9bdc3`）。**未跑 Isaac `oracle32`**：
本节已经证明这条 clip 的开环回放必然失败且原因不在被测系统里，再跑一遍只会复现已知结果、
并占住一张 GPU；要跑的是出生姿态换掉之后的那一次。

**六、复核：异常的是出生姿态，不是 clip（2026-08-07）。**

Franco `2026-08-07` 的质疑「A 选的动作在重定向前后差那么多吗？这是动捕出来的动作啊」成立。
下面每一条都是在 pod1 `hope_isaac_venv`（`mujoco 3.10.0`）上、用「二」那张表同一个 plant
量出来的活值，探针 `/workspace/franco/franco_stance_probe{,2,3,4}_20260807.py`。

**1. 站宽 `0.611` 是对的，`0.261` 才是异常的那个。** 三个独立口径同时指向这个方向：

| 口径 | 数 |
| --- | ---: |
| **动捕真人**（重定向前，`/workspace/yikang/chingmu_gmr/units/Take_061_unit04_BH.npz`） | 两踝 `0.5653 .. 0.5693 m`，两趾 `0.684 .. 0.688 m` |
| 真人髋宽 / 腿长 | `0.2097 m` / `0.7759 m` → 站宽 = `2.71x` 髋宽 = `0.731x` 腿长 |
| **重定向后的 clip** | `0.6113 m` = `2.49x` A3 髋宽（`0.246 m`）= **`0.7377x`** A3 腿长（`0.8287 m`） |
| A3 十二个腿关节**全零**（力学中位） | `0.2490 m` |
| split-ready 出生姿态 | `0.2613 m` = 力学中位 `+12 mm` |
| 全库 `73` 条 measured clip | `0.5502 .. 0.6882 m`，均值 `0.6143 m` |
| ~~现役训练里的活门~~ `lower_body_stability.min_stance_width_m` | `0.22 m`（`hope_env_cfg.py:2112`）。**更正（「十」1）：这不是活门** —— 同一个 `RewTerm` 的 `weight = 0.0`（`:2108`），而 ActionBall 的 schema-3 四格合同**根本没有** `lower_body_stability_bundle_reward` 这一块。`0.22` 只是一个设计参考值，对本谱系不构成约束 |

> **口径钉死（「十」1 复核）**：上表「腿长」两侧都用**链长**（髋→膝→踝逐段相加，与姿态无关），
> 真人 `0.77586 m`、A3 `0.82872 m`。**不要换成直线距离** —— 屈膝时直线距离会缩短
> （真人 `0.7425`、clip frame 0 的 A3 `0.6967`），两边口径一混就会得出完全不同的比值。
> 三个数是独立量的：真人 `0.5674 / 0.77586 = 0.7313`，clip `0.61126 / 0.82849 = 0.7378`，
> 出生姿态 `0.26127 / 0.82872 = 0.3153`。

**按腿长归一化，重定向把真人的 `0.7313` 变成 `0.7378`，差 `0.9%`；出生姿态是 `0.3153`，只有真人的 `43%`。** 也就是说站宽这一项 retarget
不但没有引入缺陷，还是它做得最准的一项。代价也不大：`right_hip_roll` 用掉 `-0.491` rad
（限位 `-1.606`，`31%`），`left_hip_roll` `+0.125`（限位 `+1.606`，`8%`），
`ankle_roll` `±0.14..0.23`（限位 `±0.349`）。**一个要左右移动接球的机器人，`0.261 m` 两脚并拢
才是那个不合理的数。**

**2. 出生姿态的来历（查三层，三层一致）。** ①**机制码**：`materialize_a3_dynamic_ready_contract.py`
走的是 `full_seed` 分支（`teacher_yaw_aligned_full_seed_plus_exact_teacher_reference`），
它消费一份**内容钉死的历史数值 seed**、`seed_all_joints_exactly_preserved = True`，
只绕世界 z 转 `-1.4771 rad` 对齐 teacher 的 root yaw ——
**没有一个腿关节来自这条 clip**。②**数值**：那 `12` 个腿关节与
`hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py`
的 `AGIBOT_A3_CFG.init_state.joint_pos` **逐位相同**（`hip_pitch -0.1311`、`knee 0.2468`、
`ankle_pitch -0.1204`、`hip_roll ±0.0056`、`hip_yaw ∓0.0348`、`ankle_roll ∓0.0078`），
root z 也正是 `1.0684`；该文件自陈这是「a3.py `default_angles`（厂商 deploy 源
`a3_policy_parameters.hpp`），同时用作 reset 姿态**和动作零点**（`use_default_offset=True`），
必须与 deploy 动作解码器逐位一致」。③**实验史**：`docs/runbook.md` 记着「XML stand 骨盆 `1.068`」，
`EXP-V2-REWARD-FREEZE-20260726` §0.13 裁过一次**形状完全相同**的错配
（出生摆成旧站姿、拍在 `0.882 m`，canonical clip frame 0 的拍在 `1.229 m`），
当时的裁定原文就是**「clip 无罪」**，并且明令禁止改资产 `default_joint_pos`。

**所以它不是为了让 hold LP 有解才挑出来的**，它是**厂商 deploy 默认站姿 / 动作零点**被继承下来的。
它能过 hold LP 是**结果**不是**原因**（接近力学中位的站立当然好保持）。
**clip 与出生姿态那 `0.350 m` 的差，是 deploy 合同带来的设计差，不是 clip 的缺陷。**

**3. 支撑多边形算的是 clip 自己的脚，但 `-11.9 mm` 是量具打出来的。** 先排除最容易犯的错：
审计确实是用**当帧自己的**鞋底顶点建凸包（`ground_geometry`），没有误用出生姿态的站宽。
真正的问题在带宽规则 —— `lowest` 取的是**两只脚合起来**的最低顶点，再收 `SUPPORT_BAND_M = 6 mm`。
而这条 clip 的**左脚鞋底比右脚高 `3.38 .. 5.68 mm`（均值 `4.81`）**，于是 `6 mm` 的带只能捞到
左脚最底下约 `1 mm` 的一条边，左半个支撑多边形塌掉：凸包面积 `501..828 cm²`，
而两脚各自踩实时应是 `966..1063 cm²`。带宽扫描（同一份代码、同一批帧）：

| 规则 | frame 0 | 最小 | 均值 | 在支撑面外 |
| --- | ---: | ---: | ---: | ---: |
| 全局 `2 mm` | `-310.9` | `-310.9` | `-301.4` | `57/57` |
| 全局 `4 mm` | `-323.6` | `-323.6` | `-253.5` | `57/57` |
| **全局 `6 mm`（现役，本节初稿的来源）** | **`-11.95`** | **`-25.72`** | **`-7.04`** | **`35/57`** |
| 全局 `8 mm` | `+14.49` | `+3.52` | `+21.00` | **`0/57`** |
| 全局 `10 mm` | `+38.27` | `+29.88` | `+45.35` | `0/57` |
| 全局 `20 mm` | `+109.75` | `+90.64` | `+113.77` | `0/57` |
| **每只脚各自取带 `6 mm`** | **`+26.37`** | **`+11.08`** | **`+27.91`** | **`0/57`** |

**答案坐在悬崖上：`6 mm` 往两边各挪 `2 mm`，判定就在「`57/57` 全在外」和「`0/57` 全在内」之间翻。**
出生姿态之所以不受影响，是因为**它的两只脚只差 `0.14 mm`** —— 全局 `2/6/20 mm` 分别给
`+111.6 / +123.3 / +133.6 mm`，稳在平地上。**「二」那张表原本是拿悬崖上的数去比平地上的数。**

**变异测试（改的是量具，不是机器人）**：取 clip frame 0，把**左脚鞋底顶点**整体下移实测的
`4.86 mm`（机器人姿态、质心、关节角一律不动），再跑**同一条**现役规则 ——
裕度从 `-11.95` 变成 `+26.37 mm`，凸包面积 `586 -> 966 cm²`。
**那 `4.9 mm` 的鞋底高差，而不是机器人的平衡，值 `38.3 mm` 的「裕度」。**
诚实的双支撑多边形上，`57/57` 帧质心都在支撑面内，binding 边全程是**后跟那条边**，
裕度 `11.1 .. 48.9 mm` —— 一个前倾发力的人本来就该长这样。
（审计工具已由并行 workflow 补上 `com_footprint_margin_m` / `sole_lowest_vertex_z_by_foot_m`，
本节不重复改它。）

**4. 左右脚 `4.8 mm` 高差是真人自己的重心分配，不是重定向捏出来的。** 动捕源里
`LeftToeBase` 比 `RightToeBase` 高 `+0.5 .. +4.2 mm`（本条 take `+3.2 mm`，同符号、同量级），
两个踝关节则齐平到 `< 1 mm`。全库 `73` 条重定向后的左右鞋底差是 `-13.2 .. +9.3 mm`，绝大多数在
`+2 .. +8 mm`。**真人本来就是左脚略虚、右脚略实。**

**5. 唯一真属于重定向的缺陷是地面高度，而且它是全库的。** 本条 clip 要整体下沉
`12.05 .. 14.43 mm` 才能让较低那只脚坐到出生姿态那个 `-1.5 mm`，而 `57` 帧内这个下沉量只散
`2.38 mm` —— **近乎一个常数偏移，正是坐标系/地面高度约定错一档的形状**。
扫全库 `73` 条：**`71` 条的最低鞋底从不触地**，`54` 条落在 `5..25 mm`，整体范围 `-5.33 .. +13.85 mm`。
所以这不是「`Take_061_unit04_BH` 坏了」，是**导出这一步的地板 z 约定**，一条 clip 一个标量。

**6. `nonleg_exact_teacher_q0` 那份 MuJoCo 权威出生姿态救不了场（本轮 item e 被证伪）。**
`take061.measured_v4.split_physical_hold.v2.json` 的 `semantics.physical_reset` 写着
`shared_grounded_lower_root_plus_teacher_nonleg`，`sources.shared_lower_root_seed` 指回**同一份**
`dynamic_ready.v2`，root 与 `12` 个腿关节照抄，只换 `19` 个非腿关节。
实测它的站宽**也是 `0.2613 m`**，两踝坐标、鞋底高度（`-1.31 / -1.45 mm`）、凸包全部与出生姿态 A 逐位相同，
`support_margin` `+127.2 mm`（与 `+123.3` 的差只来自手臂改变了质心）。
**它那份 `1000 x 25` tick 零失败的收据对站宽没有发言权。**

**裁定。** 三条支柱里，**一条成立但归错了对象、一条不成立、一条箭头指反了**：
`57/57` 悬空成立，但那是全库共有的一个标量地面偏移；`35/57` 失衡不成立，是 `6 mm` 带宽撞上
`4.8 mm` 左右脚高差的量具假象；站宽差 `0.350 m` 数字对，但异常的是 `0.261 m` 那一侧。
**结论改口为：clip 合法（除去一个全库共有的 `~1 cm` 地面偏移），异常的是出生姿态；
两者不匹配是 deploy 动作零点带来的设计差。** 力矩不是卡点这一条不受影响，「二」的其余结论按上表执行。

**七、独立复核那把尺子本身：尺子是对的，但它量错了一块网格（2026-08-07，另一条 workflow）。**

「六」回答的是「这三条症状说明了什么」。本条回答的是**上一层**的问题 ——
Franco 那句「这是动捕出来的动作啊」还有第二种读法：**会不会根本没悬空，是我们把坐标系读错一档？**
`10.5 mm` 正是「地板高度 / 世界系 vs 机器人局部地面系」错一档的典型量级，本轮已经在同类问题上栽过两次。
所以这一条是**专门去证伪那次测量**的，默认假设「它是约定错误」。**证伪失败：测量是对的。**
但顺手抓到量具的另一处真错，把「六」的所有毫米数往同一个方向挪了一档。

> **就地更正（「十」1）：那一档不是一个常数 `1.12 mm`。** `1.12 mm` 是**脚放平**时视觉网格比
> 碰撞网格低的量（出生姿态就是这个数）；脚一倾斜，那圈挑檐的竖直投影就变大 ——
> 本 clip 的脚倾斜 `3.62°/5.96°`，实测差值是**左 `2.41 mm`、右 `2.37 mm`**。
> 本条第 `5` 项那张换算表里的数是对的（它是逐帧重算的，不是加一个常数）；
> 错的只是这里这句「挪了 `1.12 mm`」的概括。凡是引用「差 `1.12 mm`」去手工换算别处数字的，都要重量。

**1. 四条互相独立的路径给同一个答案，所以不是坐标系问题。**

| 路径 | 做法 | 结果 |
| --- | --- | --- |
| ① 审计原路 | 鞋底网格顶点逐个做 FK，取最低点的 `z` | frame 0 左 `+18.37` 右 `+12.98 mm` |
| ② MuJoCo 自己的几何距离 | `mj_geomDistance(鞋底, floor)`，与顶点枚举**完全不同**的一段代码 | 与①逐位相同（`1e-9` 级） |
| ③ MuJoCo 自己的碰撞检测 | `mj_forward` 后数 `data.ncon` | clip **`57/57` 帧 `ncon = 0`**；出生姿态 `ncon = 6`，接触深度 `-0.19 / -0.33 mm` |
| ④ clip 自带的运动学 | npz 里 `body_pos_w`（重定向管线在 **Isaac** 侧写下的、`body_pos_point = "link_origin"`） | 与 MuJoCo FK 对 `32` 个 body × `57` 帧全比，最大差 **`9.5e-8 m`** |

路径④是关键：**clip 自己记下来的世界坐标，和我们喂进 MuJoCo 得到的世界坐标，是同一个数。**
所以不存在「差一个 `0.76`」、不存在 world/局部地面系混淆、也不存在 link 原点与质心搞错。
地板就是厂商 MJCF 里那块 `pos = 0` 的 `plane`（`z = 0`），clip 的 root 和出生姿态的 root 在同一个系里。
出生姿态过、clip 不过，是**同一把尺子**量出来的。

**2. 跨库对照：同一把尺子在另一个动作库上读出「踩着地」。** 同一段代码、同一个 MJCF、同一个系：

| 动作库 | clip 数 | 每条 clip 最低鞋底的最好成绩 | 有多少条**从不**进到离地 `+2 mm` 以内 |
| --- | ---: | --- | ---: |
| `fivebind_20260727`（老 canonical 库） | `16` | `-6.30 .. -0.33 mm`，中位 `-0.50` | **`0 / 16`** |
| `chingmu73_20260728` | `74` | 中位 `+9.59 mm` | `70 / 74` |
| `chingmu73_measured_v4_20260803` | `73` | 中位 `+9.57 mm` | `68 / 73` |

**量具没坏 —— 它在老库上一条不落地读出「踩着地」。** 悬空是 chingmu 这条导出链路独有的。

**3. 「整体压下去 1 cm」这条修法被实跑证伪，不是被推理否掉的。** 直接调用仓库自己那台
`MujocoGroundContactLPSolver`（产出 WAIT `hold_qdes` 的同一个求解器），把 clip 整体下移 `d` 再送进去：

- 单一常数 `d`：可行区间**是空的** —— 要让每帧较高那只脚不悬空需要 `d >= 17.60 mm`，
  要让较低那只脚不过度穿透需要 `d <= 14.92 mm`。
- 放宽到**每帧各用各的最优 `d`**（取两脚高度的中点）：**`0/57` 帧被接受**。
  拒绝理由从「两只脚都必须有真实地面接触」（`8` 帧）翻成「穿透过深」（`49` 帧）——
  **压不下去的那一头总有一只脚出界。**

原因是两只脚**不共面**：左脚鞋底比右脚高 `3.51 .. 5.66 mm`（均值 `4.83`），而 LP 的接触窗口
（`contact_gap 2 mm` + `penetration 2 mm`）只有 `4 mm` 宽。**一个刚体平移修不好一个共面性问题。**
这与「六」5 的判断一致，并把它从估算升级成实测。

**4. 偏移到底长在哪一节：不是整机 z，是踝高。** A3 自己的 `left_foot`/`right_foot` **site** 写在
ankle_roll 系的 `[0.04, 0, -0.067]`，碰撞网格在脚放平时也正好在踝原点下 `0.06746 m` —— 模型自洽。
这条 clip 的重定向把两个 ankle_roll 原点钉在 `0.0920 m`（`57` 帧内只散 `0.3 mm`）：
**脚要是放平，鞋底就该在 `+24.7 / +24.5 mm`（两只脚只差 `0.2 mm`）。**
剩下的 `12.9 .. 19.6 mm` 是脚面倾斜（左 `3.62°`、右 `5.96°`）把外缘压下去之后的余量。
全库的踝高是 `0.0749 .. 0.1581 m`，**所以它不是一个全库常数，是「导出时没有地面约束」**；
「一条 clip 一个标量」只在**单条 clip 内部**成立。修法仍是每条 clip 一个 z 偏移，
但**必须外加双脚共面收尾**，否则地面 LP 照拒（见 3）。

**5. 量具的真错：审计把纯视觉网格也当成了鞋底。** `ankle_roll` 上挂两块 mesh，一块
`contype=1 conaffinity=7`（真碰撞），一块 `contype=0 conaffinity=0`（纯视觉），
**视觉那块比碰撞那块低 `1.12 mm`**。审计原来按「body 上所有 mesh」选，于是把视觉网格算了进去；
而 MuJoCo 的接触、以及地面 LP，用的都是碰撞网格（`contype != 0 and conaffinity != 0`）。
已改成只取碰撞网格并 fail-closed（选不到就拒绝出数），另加四条变异测试。换算结果：

| 数 | 原记法（含视觉网格） | **更正（只用碰撞网格）** |
| --- | ---: | ---: |
| clip 鞋底离地，左 | `+15.96 .. +17.49 mm` | **`+18.37 .. +19.60 mm`** |
| clip 鞋底离地，右 | `+10.55 .. +12.93 mm` | **`+12.92 .. +15.16 mm`** |
| 出生姿态鞋底 | `-1.45 mm`（「压进地里」） | **`-0.33 mm`**（与 MuJoCo 实际接触深度 `-0.33 mm` 逐位吻合） |
| 现役 `6 mm` 全局带的质心裕度，frame 0 | `-11.95 mm`（**在支撑面外**） | **`+0.15 mm`（在支撑面内）** |
| 同上，全片 | `-25.72 .. +15.61`，均值 `-7.04`，`35/57` 在外 | **`-13.48 .. +21.54`，均值 `+3.88`，`22/57` 在外** |
| 「两脚放平踩实」的鞋底 footprint 裕度 | （原无） | **`+86.2 .. +128.7 mm`，均值 `+109.6`，`0/57` 在外**（出生姿态 `+133.6`） |

**这解答了「出生姿态为什么会『压进地面 1.5 mm』」** —— 一个合法姿态压进地里本来就可疑：
`1.12 mm` 是视觉网格的挑檐，剩下 `0.33 mm` 才是求解器正常的接触余量。**不是约定，是量错了网格。**
同时，被「六」3 判为量具假象的那个 `-11.9 mm`，**错了两层**：量错网格 + `6 mm` 带宽踩在悬崖上。
换成碰撞网格之后，连现役那条最苛刻的规则都判 frame 0 **在支撑面内**。

**本条裁定。** 对「这是不是坐标系约定错一档」这个假设：**证伪失败，测量成立** ——
悬空是真的，而且比记的还多 `~2.4 mm`。对「六」的三条改口：**方向全部维持**，
只是（i）悬空那条更严重、且**平移修不好**（实测 `0/57`）；（ii）失衡那条比「六」说的还要更不成立。
`5.6.7` 的力矩结论、`5.6.8` 的门重定范围，均不受影响。

**八、把那道地面 LP 自己送上变异测试台；以及「平移修不好」不等于「修不好」（2026-08-07，第三条 workflow）。**

「七」验的是量鞋底高度的那把尺子。本条验的是**做裁决的那道门** —— 仓库自己那台
`MujocoGroundContactLPSolver`（产出 WAIT `hold_qdes` 的同一个求解器）。初稿写「那道门是对的，被拒的是 clip」，
本条把这句话拆成两半：**门没错，但它拒的是那 1 cm，不是这条 clip。**

**1. 它说的「真实地面接触」是什么、容差多少：实测 `1.7 mm` 宽，而且是穿透侧单边。**
`GroundContactConfig` 里写着 `contact_gap_tolerance_m = 2.0e-3`，读起来像「离地 2 mm 以内都算接触」。
**这个参数在正方向上是死的** —— 厂商 MJCF 里 floor 与两只脚碰撞几何的 `geom_margin` **全是 `0`**，
MuJoCo 只在真穿透时才生成接触记录，所以鞋底高出地面 `0.06 mm` 时 `ncon` 已经是 `0`，
那 `2 mm` 的分支永远走不到。把已知通过的出生姿态整体抬/压，`0.1 mm` 一档扫：

| 整体位移 `dz` | LP 判定 |
| --- | --- |
| `<= -1.70 mm` | `REJECT: MuJoCo reports excessive foot-floor penetration` |
| **`-1.65 .. +0.15 mm`** | **`FEASIBLE`**（窗口宽 `1.80 mm`；本表按 `0.05 mm` 步长复扫，初稿的 `1.7 mm` 是 `0.1 mm` 步长的取整） |
| `>= +0.20 mm` | `REJECT: both named feet must have active MuJoCo floor contact` |

**变异测试的意义在这里：把一个已经通过的姿态抬高 `0.2 mm`，报的是一字不差的同一句话。**
所以 `both named feet must have active MuJoCo floor contact` 只说明一件事 ——
**至少有一只脚没压进地板。** 它不含平衡、站宽、力矩、也不含「这条 clip 好不好」的任何信息。

**2. 只动根高 + 四个踝关节，`57/57` 帧的接触就都成立了 —— 这条 clip 差的是一次收尾解算，不是重做重定向。**
「七」3 实跑证伪的是「整体平移」这一种修法，成立；但把踝关节放开就修好了。
做法：不动髋/膝（站宽 `0.6113 m` 一个数没变），不动上身/手臂/拍子（一个数没动），
只解 `根高 + left/right_ankle_pitch/roll` 五个自由度，让两只脚的碰撞鞋底同时**放平并**贴到 `-0.5 mm`：

| | 实测 | 独立复核（「十」3，另一份代码） |
| --- | --- | --- |
| 需要的根高下移 | `24.87 .. 25.10 mm` | `25.03 .. 25.19 mm` |
| 需要的最大踝关节改动 | `4.20 .. 5.14°`（`57/57` 帧全部落在 MJCF 关节限位内） | `4.205 .. 5.143°`，同样 `57/57` 在限位内 |
| 修正后两脚碰撞鞋底落地深度 | 左 `-0.493 .. -0.468 mm`，右 `-0.532 .. -0.507 mm`（LP 收的鞋底深度区间是 `-2.0 .. 0 mm`） | 左 `-0.40 .. -0.31`，右 `-0.69 .. -0.60`（同一族解，最小二乘落点略不同） |
| `both named feet must have active MuJoCo floor contact` | **`57/57` -> `0/57`** | **`57/57` -> `0/57`** |
| 同一台 LP、同一套默认容差下的逐帧判定 | `29/57 FEASIBLE`、`21/57 INFEASIBLE_LP`、`5/57` 接触点跑出鞋底面片、`2/57` 数值不收敛 | **`28/57 FEASIBLE`**、`23/57 INFEASIBLE_LP`、`6/57` LP 不收敛 |

> **就地补一个本条初稿漏掉的必要条件（「十」3 实测）：「把脚放下来碰到地」不够，必须「把脚放平」。**
> 上表的 `4.20 .. 5.14°` 里，绝大部分不是用来「下降」的，是用来**把脚面转平**的
> （这条 clip 的鞋底原本离水平 `3.62°/5.96°`）。如果只解「两只鞋底最低顶点到 `-0.5 mm`」而
> 不约束脚放平，最小二乘会给出一个更省力的解（根高只降 `16.0 .. 17.8 mm`、踝只动 `0.82 .. 1.27°`），
> 两只脚确实都碰到地了，**但同一台 LP `57/57` 全拒**，报的是另一句话：
> `foot collision mesh does not expose a two-dimensional sole polygon`。
> 原因是这台 LP 用**MuJoCo 真实生成的接触点**建支撑多边形，每只脚至少要 `3` 个不共线的点；
> 倾斜的鞋底压下去只碰到一条棱。**这是接地收尾解算的硬需求，不是可选项。**

**这正是出生姿态那一侧每天在做、而这条链路从来没做过的事** ——
`canonical_grounded_ready` 的 G1 就是「解腿，让两只脚贴平到 `target_contact_preload_m = 0.5 mm`」。
出生姿态过门、clip 不过门，差的是**这一步跑没跑过**，不是动捕好不好。

**3. 剩下那 `21/57` 的 `INFEASIBLE_LP` 不能记在 clip 头上：这台 LP 是「静止」证书。**
它的接触静止容差是 `contact_velocity_tolerance_m_s = 2.0e-5`（`0.02 mm/s`）与
`contact_acceleration_tolerance_m_s2 = 2.0e-4`。而这条 clip 的脚是**真踩着**的：
取 frame 0 的 `8` 个最低碰撞顶点全程跟踪，`1.14 s` 里相对 frame 0 的最大位移左 `2.0 mm`、右 `2.7 mm`，
逐帧点速度中位 `4.0 / 5.5 mm/s`、最大 `23.8 / 10.6 mm/s`。
**按人的标准这就是「脚没动」，但它比 LP 的容差大 `200~1200` 倍。**
所以每一帧只能当**静态姿势**送进去（`qvel = qacc = 0`），得到的 `29/57` 回答的是
「这一帧站得住吗」，不是「这一帧执行得了吗」。一个挥拍中途、重心正在转移的帧站不住是正常的，
不是缺陷。

**4. 实验史（Franco 问的「有没有别的 clip 通过过」）：通过过，但全是腿被冻住的。**
同一台 LP、同一个 MJCF、逐帧跑老 canonical 库：

| clip | 帧 | root z | LP 逐帧结果 |
| --- | ---: | --- | --- |
| `bh_block_upper_stable_v2` | `54` | 恒 `1.0684` | **`54/54 FEASIBLE`** |
| `bh_loop_c_upper_stable_v2` | `71` | 恒 `1.0684` | **`71/71 FEASIBLE`** |
| `bh_block_upper_qvel_fix_v1` | `54` | 恒 `0.9207` | **`54/54 FEASIBLE`** |
| `bh_block_upper_fivebind` | `54` | 恒 `0.9207` | **`54/54 FEASIBLE`** |
| `bh_block_full_full_fivebind` | `55` | `0.9207..0.9308` | `3` FEASIBLE / **`39` 无地面接触** / `11` 穿透过深 / `2` 鞋底面片 |
| `s0_highpress_full_full_fivebind` | `44` | `0.9207..0.9375` | `4` FEASIBLE / **`28` 无地面接触** / `9` 穿透过深 / `3` 其他 |

通过的四条的腿**全是冻住的** —— 收据自陈 `leg_joint_velocity_exact_zero = true`、
`root_xy_all_frames_bitwise_equal = true`，root z 逐帧恒定；它们的腿就是 G1 解出来的那个静态站姿，
被 `materialize_grounded_upper_motion.py` 原样贴到每一帧上。**保留自己腿的那两条 full-body clip，
被一字不差的同一句 `both named feet must have active MuJoCo floor contact` 拒了 `39/55` 和 `28/44`。**

**所以答案是：从来没有任何「腿是自己的」运动 clip 通过过这道 LP。**
通过的只有静态姿势，以及腿被换成静态姿势的 clip。「七」2 说老库「一条不落地」成立，
但落地只是必要条件 —— 老库自己的 full-body clip 一样过不了这道门。

**5. 链路里那一步本来该谁做（三层查证：机制码 / 实验史 / 现役产物）。**

| 步 | 谁 / 在哪 | 产物 | 动了根和腿吗 |
| --- | --- | --- | --- |
| 源动捕 | ChingMu，`Take_061_unit04_BH.bvh`（`human_height = 1.687 m`） | — | — |
| 单元切分 | `/workspace/yikang/a3_vendor_194d_physical_.../ChingMu_Selected/units/` | `.json` + `.npz` | — |
| **GMR 全身重定向** | `/workspace/yikang/chingmu_gmr/chingmu_to_a3_gmr.py` | `out/*.pkl`（`world_z0 = "floor"`） | **是，且只有这一步动过** |
| 腕/击球窗精解 | `refine_wrist_face.py` / `solve_hit_window.py` | `out_refined/`、`chingmu_a3_units_v1/` | 否 |
| **脚锁** | `solve_foot_lock.py` | `chingmu_a3_units_v2/`（`foot_locked = true`） | 只改腿的**水平位姿**，不管高度 |
| 拍面全程精解 | `/workspace/codexschema/.../solve_chingmu_canonical_racket_full_phase.py` | `*.v70a150.pkl` + report | 否（`optimized_joints` 只有腰 `3` + 右臂 `7`） |
| materialize | `materialize_measured_racket_motion_npz.py` | `hope_*.measured_v4.npz` | 否 |

**「根高从 GMR 之后再没被动过」是活值比出来的，不是读代码猜的**：
`out/`、`out_refined/`、`chingmu_a3_units_v1/`、`chingmu_a3_units_v2/` 四个 pkl 的 root z 全是
`[0.8519, 0.8934]`，最终 npz 是 `[0.8520, 0.8933]` —— 同一个数。

链路里**确实有**一个脚接触步骤，但它管的是另一件事：`solve_foot_lock.py` 自己的注释写着
「L0 audit fails `74/101` on foot_skate（支撑脚滑 `0.5-1.0 m/s`）而源人类脚是 `0.03 m/s` —— 纯重定向假象」，
它的目标是**把脚钉在自己的锚点位姿上止滑**，锚点本身离地多高，它不问。
而验收那一步看不见这个偏移：`check_retarget_contact.py` 的接地项是 `data.xpos[1:, 2].min()`，
量的是 **body 原点**的最低 `z`，期望值写在注释里「foot at ~`0.03-0.06` normal」。
**用 body 原点去量 `1 cm` 的鞋底偏移，量不出来。**

**所以「哪一步坏的」的答案是：没有哪一步把 clip 放到地面上过，
而唯一一道会发现它的检查，用的是看不见它的那把尺。**

**6. 顺带核实 §12.3 的 `0/73`：它对这条 clip 不含负面信息。**
`0/73` 不是「73 条都被判失败」。§6 采用表自己写着 `57/73` 是已知硬失败，另 `16/73`
**通过了 position/velocity、只因为缺加速度权威 / torque-speed 曲线 / 逐帧逆动力学力矩而停在 `UNKNOWN`**。
`Take_061_unit04_BH` 是那 `16` 条之一。而且 `mechanical_admission = false` 在这条链路里是**常量不是判定** ——
v5 候选的 `REPRODUCE.sh` 直接 `assert report["mechanical_admission"] is False`。
三项缺失里的逐帧逆动力学力矩已由本节「一」补上，结论是不是卡点；另两项仍 `BLOCKED ON VENDOR CURVE`。

**本条裁定。** 「二」那句「顺带一条：地面 LP 对这条 clip 的 `57` 帧全部 fail-closed 拒绝，那道门是对的，
被拒的是 clip」——**前半句成立，后半句就地更正为：被拒的是那 `1 cm` 加左右脚 `~4.8 mm` 的不共面。**
把这两样修掉（根高 `-25 mm` + 踝 `<=5.14°`），同一台门当场收 `29/57`。
**不放宽任何判据**：LP 的默认容差一个字没动，本条的所有实跑都用默认值。
要动的是**链路里加一步接地收尾解算**（G1 已经有现成实现），以及**给 `oracle32` 换一道能失败的门**
（「五」已经写了，本条只是把它的可行性从推理变成实测）。

**收据。** pod1 `/workspace/hope_isaac_venv`（`python 3.10.18` / `mujoco 3.10.0` / `scipy 1.15.3` / `numpy 1.26.4`），
worktree `/workspace/franco/s10_ff_20260806`（`33c9bdc3`）；探针
`/workspace/franco/s11_p{2,3,4,5,6,7,9,10}*.py`（链路 pkl 对照 / 接地几何 / LP 容差扫描 /
碰撞-视觉网格分离 / 历史 clip 逐帧 LP / 最小接地修正 / 鞋底点速度 / 窗口与落地深度复核）。
**本条没有跑 Isaac，也没有改任何仓库代码或门限。**

**九、那出生姿态到底该是什么：Franco 的设计意图本来就是「出生 = frame 0，等待 = hold 住 frame 0」（2026-08-07，第四条 workflow）。**

「六」「七」「八」回答的是「clip 有没有毛病」（答：只有一个全库共有的 `~1 cm` 地面偏移）。
本条回答**另一半**：既然 clip 没毛病，那个跟它差 `0.350 m` 的出生姿态**本来就该存在吗**。

**Franco `2026-08-05`**：「task 里面的前序等待是输入的就是第 0 帧的值，只是 time to start 就是给还需要
多久动作开始这件事，然后等待动作到 frame 0 的跳变应该不是问题。」
**Franco `2026-08-07`**：「出生不应该是第 0 帧吗？等待也是啊？」
**设计意图从一开始就是「出生 = frame 0，等待 = hold 住 frame 0」，那样揭示阶跃按构造等于零。**
现役实现不是这样：出生是一个独立的 split-ready 姿态，等待 hold 在**那个**姿态上，揭示才切 frame 0，
于是凭空多出 `2.243 rad`（右腕偏航 `128°`）的阶跃，并为此做了一整轮 reveal bridge
（§5.6.6 实测：摊平到约 `35` 步只多活 `1` tick）。

**1. 这条路被试过，退役理由是 `0/73`；而那个 `0/73` 就是「六」「七」「八」的同一个 `1 cm`。** 查三层：
①**机制码**存在：`materialize_action_ball_a211_frame0_exact_artifact.py`，产出
`configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json`
（root/q 原样读 frame 0、零速度）。②**工具台账** `docs/operations/tool_catalogue.md:128` 把它标成
**「SUPERSEDED / COUNTEREXAMPLE 工具」**，理由原文「direct exact-frame0 physical-birth 同门槛扫描已是
`0/73`」，输出只留 provenance、`launch_authorized=false`。③**实验史**：本文档多处
（`§6` 采用表、`E1` 条目）写「direct frame0 physical birth `0/73`，因此 exact frame0 只保留为 teacher frame0」。

**但 `0/73` 不是 73 条动作各自不合格，是同一个标量偏移被数了 73 遍。**
实测（探针 `franco_bank_float_census_20260807.py`，v4 全库 `73` 条的 **frame 0**，同一 plant、同一把尺子）：

| 量 | 全库 frame 0（**已按「十」1 换成碰撞网格重量**） |
| --- | --- |
| 两只脚都不落地的 clip | **`73 / 73`**；较低那只鞋底离地 **`6.46 .. 21.26 mm`，均值 `13.04`、中位 `13.16`**（初稿 `3.96 .. 19.59 / 10.85 / 10.91` 量的是**视觉网格**，整体偏低约 `2.2 mm`；`73/73` 这个计数不变） |
| 站宽（两踝） | `0.5060 .. 0.6757 m`，均值 `0.6056`（动捕真人 `0.567`；出生姿态 `0.2613`）—— 与网格选择无关，**复核一致** |
| 质心对**完整鞋底投影**双支撑多边形的裕度 | `-16.6 .. +70.8 mm`，均值 `+41.9`，**只有 `1/73` 为负** |
| 同一批帧换成现役全局 `6 mm` 带宽规则 | **`38/73` 为负**，最低 `-355 mm` |
| **（「十」4 新增）frame 0 关节角就已经出 MJCF 限位的 clip** | **`9 / 73`**（`060_unit08`、`060_unit13`、`061_unit05`、`061_unit10`、`061_unit20`、`061_unit22`、`062_unit05`、`063_unit02`、`063_unit05`）—— 这是**另一条**重定向缺陷，跟地面高度无关，此前没人记过 |
| **（「十」4 新增）frame 0 有非足部几何在碰东西的 clip** | `4 / 73`（`063_unit00/01/03/04`，各 `1` 个接触） |

（口径说明：本表是**每条 clip 的 frame 0**，因为出生姿态问的就是 frame 0；「七」那张跨库表统计的是
**每条 clip 全程的最好成绩**，所以两边的计数不同、不矛盾。）
人话：**这个库在 frame 0 的系统性不合格项是「整体悬空约 1.3 cm」，平衡和站宽都是合格的**
（外加 `9/73` 条另有关节超限，需单独修）。
换句话说，`0/73` 这道退役判决**判的是那 `1 cm`，不是「出生取 frame 0」这个想法** ——
「十」4 把这句话从推断变成了实测：同一台 LP，接地前 `0/73`，接地后 `27/73 FEASIBLE`。

> **⚠ 2026-08-08 就地更正**：上面这句只推翻了"那个 `0` 的成因"，**没有**证明「出生取 frame 0」可行。
> 那台 LP 用的是电机限幅边界；换成生产链路真正用的 `kp × 行程` 边界，接地后的 frame 0 **仍然被拒**
> （`waist_pitch` 要 `-49.155 N·m`，位置指令只能出 `-21.704`）。全库同口径实测是 **`1/73`**。见「十二」「十三」。

**2. MuJoCo 权威侧那份「只做一半」，卡的是同一个 `1 cm`。** 机制码只有一行 ——
`hope_training/whole_body_tracking/mujoco_native/action_specific_hold.py`：

```python
ready_q = teacher_q.copy()
ready_q[leg_indices] = seed_q[leg_indices]     # 31 个先全取 teacher frame 0，再把 12 个腿关节盖回 seed
```

root 也用 seed 的，语义字段自陈 `shared_grounded_lower_root_plus_teacher_nonleg`。
腿之所以留 seed 值，是因为下一步 `grounded._audit_and_build_result` 要过 `sole_floor`/`double_support`，
而 frame 0 的腿过不了 —— 过不了的原因就是上面那 `1 cm`。
**所以「只做一半」和「clip 悬空」是同一个根因，不是两件事。**

**3. 实测：出生取 frame 0，卡在哪、不卡在哪。** 探针 `franco_frame0_hold_20260807.py`，
走**仓库自己的**静态门 + `MujocoGroundContactLPSolver`，不写任何 artifact。
**fail-closed 锚点**：先用同一套代码重算现役 split-ready，得到 `support_margin +24.80 mm`、
`max|tau| 39.07 N·m`、`max|归一化动作| 13.31`、`exactly_executable=True` ——
与存档 `take061.measured_v4.split_physical_hold.v2.json` 逐项吻合，工具是对表的。

| 变体 | 结果 |
| --- | --- |
| `V0` 现役 split-ready（复算，锚点） | 九门全 PASS，LP 可行 `rho = 0.9960` |
| `V1` **31 个关节 + root 全取 frame 0** | **只挂 `sole_floor` / `double_support`**（以及依赖它们的 `support_margin` / `static_ground_dynamics`）。两脚离地 `18.37 / 12.98 mm`、`foot_contact_count = (0,0)`；而 `joint_limits`、`collision`、`foot_pose`、`leg_to_foot_jacobian`、`exact_model_identity` **全 PASS** |
| `V2` frame 0 + 整体下沉（扫 `5..20 mm`） | **纯平移永远到不了双支撑**：`-13 mm` 时右脚落地、左脚仍高 `5.4 mm`；`-20 mm` 时右脚压进 `7 mm`，`collision` 也挂。与「八」3 的结论一致 |
| `V3` frame 0 的关节 + seed 的 root | 两脚离地 `314 / 275 mm` —— root 与腿必须配套，不能混搭 |

**关键读数：`V1` 没有一条挂在「站不住」「关节超限」「自碰撞」上。** 挡住「出生 = frame 0」的
**只有脚没踩到地**这一件事，而那正是全库共有的那个偏移。

> **⚠ 2026-08-08 就地更正 —— 不要照「只差接地」这句话往下推。**
> 这句话只对 `V1` **当时跑到的那几道门**成立。脚的账「十一」已经还清（接地实跑通过、全库 `70/73`），
> 但接地之后还有一道 `V1` **根本没跑到**的门 —— hold LP。它现在实跑拒了：撑住接地后的 frame 0，
> `waist_pitch` 需要 `-49.155 N·m`，位置指令最多出 `-21.704`（**要 `226.5%`**）。
> **「出生 = frame 0」现在卡的是腰的保持增益，不是脚。** 见「十二」（证据）与「十三」（出路与代价）。

**4. 差多少：一个膝关节 `1.5°` 的量级。** 只动左腿俯仰链一个标量（膝 `-0.026 rad = -1.49°`，
髋/踝俯仰各 `+0.013` 保持脚掌水平）再整体下沉 `13.28 mm`，两只鞋底同时落到 `-0.50 / -0.30 mm`，
**`sole_floor` 与 `double_support` 双双 PASS**，其余 `28` 个关节保持 frame 0 精确值。
仍挂 `support_margin` —— 因为这么「贴」上去只有一条边接触，门用的是**活接触点**建的多边形，
不是整块鞋底投影（与「六」3 同一类量具效应）。**要把这一步做干净，用仓库已有的
`whole_body_safe_ready`**（自陈「exact 测量 frame 0 若已过全部门则原样返回，只有不安全时才退到
lexicographic 搜索」），或「八」那条 `根高 -25 mm + 踝 <= 5.14°` 的接地收尾，**不是手改两个关节**。

**5. 一条必须分清的事：hold 住 frame 0 仍然要解一次 LP。** 那是**重力补偿**，是物理，不是设计缺陷。
「frame 0 不能直接当 `q_des` 原样发下去」（§5.6.6）与「frame 0 不能当出生姿态」是两句话，
前者成立不蕴含后者。正确做法是**给（接地后的）frame 0 解一次 hold LP**，
而不是「因为 frame 0 不可 hold 所以另造一个姿态」。

**6. 代价与待办（本轮不动，只出方案）。**

- **要重铸的 artifact 与 SHA**：`...dynamic_ready.v2.json`、`take061.measured_v4.split_physical_hold.v2.json`、
  `take_061_unit04_bh.frame0_exact.v1.json`；`a211_split_ready_lineage.v5.json`；
  A211/C211 launcher 与 materializer 里钉死这些 SHA 的 lineage 断言；
  以及所有以 split-ready 为 `physical reset` 的四格配方与 `nominal_hold` 收据。
- **前置**：先把接地收尾做进重定向链路（「五」第一条 + 「八」结论），否则出生仍然取不到能落地的 frame 0。
  > **⚠ 2026-08-08 就地更正**：这一条**已经做完了**（「十一」），而且它**不是唯一前置**。
  > 接地之后还有一道 hold LP，它现在拒了，卡的是腰的保持增益（「十二」）。**别再把它当成"最后一步"。**
- **顺带消掉的东西**：出生若等于（接地后的）frame 0，揭示那一刻的 `2.243 rad` 阶跃**按构造为零**，
  reveal bridge 与它那 `544` 步 ramp 失去存在理由。§5.6.6「阶跃不是死因」的结论不受影响 ——
  那一节测的是「摊平阶跃能不能救命」（不能），本条说的是「这个阶跃本来就不该存在」。
  > **⚠ 2026-08-08 就地更正**：这个「按构造为零」的前提**没有兑现，也暂时兑现不了**——
  > 现役增益下出生**不能**等于 frame 0。实测阶跃仍然是 `2.2227 rad`（`right_wrist_yaw`）、
  > `L2 3.8208`、`31` 个关节里 `30` 个非零，另加骨盆下沉 `176.6 mm`。**reveal bridge 因此保留**（「十二」6）。
- **不能顺手做的**：出生姿态的 `12` 个腿关节现在逐位等于动作零点
  `AGIBOT_A3_CFG.init_state.joint_pos`（`use_default_offset=True`，与 C++ `pp_policy` lockstep）。
  换出生姿态**必须**同时明确动作零点保持不变，否则会撞 `EXP-V2-REWARD-FREEZE` §0.13 的禁令。
- **不放宽任何判据**：本条没有改任何门限、没有改任何仓库代码、没有写任何 artifact。

**收据。** pod1 `/workspace/hope_isaac_venv/bin/python`（`mujoco 3.10.0` / `scipy 1.15.3`）；探针
`/workspace/franco/franco_stance_probe{,2,3}_20260807.py`（站宽/带宽扫描/变异测试）、
`franco_bank_float_census_20260807.py`（全库 `73` 条 frame 0 普查）、
`franco_frame0_hold_20260807.py`（`V0..V3` 门 + LP，含 split-ready 锚点复算）、
`franco_frame0_project_20260807.py`（最小接地修正扫描）；
动捕源（重定向前）`/workspace/yikang/chingmu_gmr/units/Take_061_unit04_BH.{npz,bvh,json}`；
worktree **只读**引用 `/workspace/franco/s10_ff_20260806`（`33c9bdc3`）。**未跑 Isaac。**

**十、四方对账与最终裁定：`oracle32` 被拒是四件事的组合，动捕源本身没有一处不合格（2026-08-07 独立复核）。**

（说清楚一点：**动捕源**三个数全合法、**重定向**只欠一步接地收尾、
**出生姿态**才是站宽异常的那一侧、**量具**造出了一个不存在的失衡症状、
**那道门**问的根本是另一个问题。逐条见下。）

「六」「七」「八」「九」是四条并行 workflow 各写各的，彼此有冲突（「七」说「平移修不好」，
「八」说「修得好」；三份的毫米数各差一档）。本条**用第五份独立代码把每一个承重数字重量一遍**，
把冲突判掉，并给出可执行的下一步。全部实测在 pod1 `/workspace/hope_isaac_venv/bin/python`
（`python 3.10.18` / `mujoco 3.10.0` / `scipy 1.15.3` / `numpy 1.26.4`），
自建只读 checkout `/workspace/franco/adj_20260807`（`ffdca6af`），探针 `/workspace/franco/adj_p{1,2,3,4,5}.py`，
输出 `/workspace/franco/adj_p{1,2,3,4,5}.json`。**未跑 Isaac、未占 GPU、未改任何仓库代码或门限。**

**1. 复核结果：十四项活值逐位对上，四项要更正。**

| 承重数字 | 谁说的 | 我量到的 | 裁定 |
| --- | --- | --- | --- |
| clip frame 0 碰撞鞋底 左 / 右 | 「七」`+18.37 / +12.98 mm` | `+18.3659 / +12.9785 mm`（顶点枚举与 `mj_geomDistance` 两条路**逐位相同**） | **成立** |
| `57/57` 帧 `ncon = 0` | 「七」 | `57/57` | **成立** |
| 左右鞋底高差 | 「七」`3.51 .. 5.66`，均值 `4.83` | `3.514 .. 5.659`，均值 `4.827` | **成立** |
| 出生姿态鞋底 / 接触 | 「七」`-0.33 mm`、`ncon = 6` | `-0.190 / -0.332 mm`、`ncon = 6`、接触深度 `-0.178 .. -0.332` | **成立** |
| 站宽：clip / 出生 / 腿关节全零 | 「六」`0.6113 / 0.2613 / 0.2490` | `0.61126 / 0.26127 / 0.24897`，clip `57` 帧内只散 `0.09 mm` | **成立** |
| 动捕真人两踝 / 两趾 / 髋宽 | 「六」`0.5653..0.5693 / 0.684..0.688 / 0.2097` | 逐位相同 | **成立** |
| 腿长归一化站宽 真人 / clip | 「六」`0.731 / 0.7377`，差 `0.9%` | `0.7313 / 0.7378`，差 `0.9%`（口径已钉死在「六」1，两侧都用链长） | **成立** |
| 质心裕度（整鞋底 footprint） | 「七」`+86.2..+128.7`，均值 `+109.6`，`0/57` 在外 | `+86.23..+128.74`，均值 `+109.62`，frame 0 `+105.57`，`0/57` | **成立** |
| 质心裕度（现役 `6 mm` 带） | 「七」`-13.48..+21.54`，均值 `+3.88`，`22/57` | 逐位相同，frame 0 `+0.148` | **成立** |
| LP 接受窗口与两句拒绝词 | 「八」1 | `dz ∈ [-1.65, +0.15] mm` 宽 `1.80 mm`；`>= +0.20` 报 `both named feet must have active MuJoCo floor contact`，`<= -1.70` 报 penetration | **成立**（宽度按 `0.05 mm` 步长订正为 `1.80`） |
| 老库 `bh_block_full_full` / `s0_highpress_full_full` 逐帧 LP | 「八」4 `3/39/11/2` 与 `4/28/9/3` | 逐个计数完全相同 | **成立** |
| 四条冻腿 clip `100%` FEASIBLE | 「八」4 | 相同，并已扩到全部 `16` 条，见本条 2 | **成立** |
| 审计工具变异测试 | 「七」`14 passed / 0 skipped` | `14 passed in 0.17s`，`0 skipped` | **成立** |
| `ready_q[leg_indices] = seed_q[leg_indices]` / A3 `default_angles` 十二个值 / `check_retarget_contact` 用 `xpos[1:,2].min()` / `solve_foot_lock` 只管止滑 / `world_z0 = "floor"` | 「八」5、「九」2 | 逐条对上原文 | **成立** |
| 「视觉网格比碰撞网格低 `1.12 mm`」当常数用 | 「七」 | 脚放平时 `1.12 mm`，本 clip 脚倾斜时**左 `2.41` 右 `2.37 mm`** | **更正**，见「七」表头 |
| 全库 frame 0 较低鞋底 `3.96..19.59 / 10.85 / 10.91` | 「九」1 | **`6.46..21.26 / 13.04 / 13.16`**（初稿量的是视觉网格） | **更正**，见「九」1 |
| 「只解根高 + 四踝」`29/57 FEASIBLE` | 「八」2 | 只有**把脚放平**才复现得出（`28/57`）；只贴不放平 `57/57` 全拒 | **更正**，见「八」2 |
| `min_stance_width_m = 0.22` 是「现役活门」 | 「六」1 | 同一个 `RewTerm` `weight = 0.0`，ActionBall 四格合同没有这一块 | **更正**，见「六」1 |

**2. 决定性的一条：这道 LP 从来没有放过任何一条「腿是自己的」运动 clip。**
同一台 `MujocoGroundContactLPSolver`、同一套默认容差，逐帧跑**整个** `fivebind_20260727` 老 canonical 库
（`16` 条，不是「八」4 抽的 `6` 条）：

| 类别 | 条数 | 腿关节相对 frame 0 的最大变化 | 逐帧 FEASIBLE |
| --- | ---: | ---: | --- |
| `*_upper_*`（腿被 G1 静态站姿冻住） | `8` | **`0.00e+00` rad（逐位不动）** | `54/54`、`54/54`、`54/54`、`71/71`、`71/71`、`104/104`、`88/88`、`41/41` = **`100%`** |
| `*_full_full_*`（腿是自己的） | `5` | `0.31 .. 0.50` rad | `3/55`、`7/81`、`8/104`、`9/104`、`4/44` = **`31/388 = 8.0%`** |

全部 `5` 条自带腿的老 clip，主拒绝理由都是同一句
`both named feet must have active MuJoCo floor contact`（`295/388 = 76%`）。
**所以「这条动捕 clip 被 LP 拒了 `57/57`」这件事，对「这条 clip 好不好」不含任何信息** ——
把老库自己最正统的五条 full-body clip 拿来，同一道门也只放行 `8%`。

**3. 接地收尾解算：把腿关节全放开，同一台门当场收 `50/57`。**
「八」2 只放开四个踝关节（还剩一个共面性自由度不够用，`5` 个自由度对 `6` 个约束）。
放开全部 `12` 个腿关节、脚硬约束放平、两只鞋底同时压到 `-0.5 mm`（`13` 个自由度对 `6` 个约束，够用）：

| | 只放开四踝（「八」2） | **放开十二个腿关节（本条）** |
| --- | --- | --- |
| 根高下移 | `25.03 .. 25.19 mm` | `21.48 .. 25.85 mm` |
| 腿关节最大改动 | `4.21 .. 5.14°` | **`1.75 .. 2.45°`** |
| 落地深度（左 / 右） | `-0.40..-0.31` / `-0.69..-0.60 mm`（凑不齐） | **精确 `-0.500 / -0.500 mm`** |
| 脚面倾角 | `< 3e-6°` | `< 3e-6°` |
| 每只脚的真实接触点数 | `3 / 3`，`57/57` 帧 | `3 / 3`，`57/57` 帧 |
| 站宽变化 | `0`（髋膝没动） | `0.6109 .. 0.6149 m`，最多动 `3.7 mm` |
| **同一台 LP 逐帧判定** | `28` FEASIBLE / `23` INFEASIBLE_LP / `6` 不收敛 | **`50` FEASIBLE / `7` INFEASIBLE_LP / `0` 拒绝** |
| 关节限位 | `57/57` 在限位内 | `57/57` 在限位内 |

**上身、手臂、拍子一个数没动；腿最多动 `2.45°`；站宽最多动 `3.7 mm`。**
换来的是**接触拒绝从 `57/57` 归零、静态可行从 `0/57` 到 `50/57`**。
剩下 `7` 帧的 `INFEASIBLE_LP` 按「八」3 处理（那台 LP 的接触静止容差是 `0.02 mm/s`，
而这条 clip 的脚点速度中位 `4.0 / 5.5 mm/s` —— 挥拍中途重心正在转移的帧站不住是正常的，不是缺陷）。

**4. 把 `0/73` 那道退役判决重新判一遍：接地前 `0/73`，接地后 `27/73`。**
对 v4 全库 `73` 条的 **frame 0** 各做一次上面同样的接地收尾解算，再送同一台 LP：

| | 接地前 | 接地后 |
| --- | --- | --- |
| `FEASIBLE` | **`0 / 73`** | **`27 / 73`** |
| `both named feet must have active MuJoCo floor contact` | `60 / 73` | **`0 / 73`** |
| `INFEASIBLE_LP`（静止证书判「这一帧站不住」） | `0` | `32 / 73` |
| 关节角出 MJCF 限位 | `9 / 73` | `9 / 73`（同一批 clip；探针没加限位约束，另有 `2` 条被推出去，属探针缺陷不是结论） |
| 非足部几何在碰东西 | `4 / 73` | `3 / 73` |

**`Take_061_unit04_BH` 的 frame 0 在接地后是 `FEASIBLE`** ——
根高 `-24.01 mm`、十二个腿关节最大改动 `2.44°`、站宽只动 `0.77 mm`。

> **⚠ 2026-08-08 就地更正：这个 `FEASIBLE` 不等于"可以拿它当出生姿态"。**
> 本条这台 LP 用的是**电机限幅**那套力矩边界（`waist_pitch ±118.0 N·m`，`-49.155` 当然放得进去）；
> 而生产链路的 hold LP 还要**再交上一层**"位置指令能出多大力矩"的边界
> `kp × (q_des 还能走多远)`（`waist_pitch` 只有 `±21.7 N·m`）。**同一台求解器，两套边界。**
> 用生产那套边界实跑：接地后的 frame 0 **`feasible = false`**（对照组现役出生姿态 `true`）。
> 所以本表的 `27/73` 读作"接地之后**地面这一关**过了"，**不能**读作"出生可以取 frame 0"——
> 后者的实测是 `1/73`（唯一一条是 `hope_Take_062_unit00_BH`，且它自己 `70` 帧里只有 `12` 帧撑得住）。见「十二」「十三」。
**也就是说：Franco 一直想要的「出生 = teacher frame 0」，在仓库自己那台最严的静态门上，
差的只是一次 `2.44°` 的接地收尾解算。** 「九」的方向成立，且现在是实测不是推断。

**顺带抓到一条没人记过的独立缺陷**：`9 / 73` 条 clip 的 **frame 0 关节角本身就出 MJCF 限位**
（`060_unit08`、`060_unit13`、`061_unit05`、`061_unit10`、`061_unit20`、`061_unit22`、
`062_unit05`、`063_unit02`、`063_unit05`）。这跟地面高度无关，接地解算也修不了，
**必须在重定向侧单独处理**，否则这 `9` 条永远进不了任何门。

> **就地更正（2026-08-07 稍晚，实测见「十一」5(ii)）：这一条不成立，它是 float32 存储精度。**
> 这 `9` 条超出的是 `right_shoulder_pitch` 的 `-165.000°` 限位，超出量 **`6.767e-08 rad`
> （`3.9e-06` 度）= `0.57` 个 float32 最低位**；同一口径下还有 `16` 条超 `right_wrist_yaw`
> 的 `+93.000°`（`4.616e-09 rad`），合计 `25/73` 而不是 `9/73`。bank 的 `joint_pos` 存的是
> float32，`25/25` 条那个值都**正是限位在 float32 上的最近可表示数**，而审计容差
> `1e-10 rad` 比 float32 在该角度的分辨率严 `1000` 倍。
> **不要回重定向侧修**：要么把这道门的容差抬到 float32 量级，要么把 bank 存成 float64。

**5. 一句话裁定（对 Franco 的问题）。**

> **答案是「组合」，而且四项不等权：主因是 (iv) 判据用错地方 + (iii) 出生姿态异常 + (ii) 量具错，
> (i) 重定向确实有缺陷但只欠一步收尾解算 —— 没有一项支持「这条动捕不能用」。**
> 按贡献排序：
> **(iv) 判据用错了地方** —— 那道双支撑 LP 是**静止**证书，历史上只放行过腿被冻住的 clip
> （冻腿 `8` 条 `100%`，自带腿 `5` 条 `8.0%`）；用它判一条挥拍中的动捕参考是范畴错误。
> **(iii) 出生姿态才是异常的那个** —— 站宽 `0.2613 m` 只有真人归一化站宽的 `43%`
> （`0.3153` vs `0.7313`），而 clip 的 `0.6113 m` 与真人差 `0.9%`。
> **(ii) 我们的量具错了两处** —— 量到了视觉网格（脚倾斜时差 `2.4 mm`）、
> 以及 `6 mm` 接触带撞上 `4.8 mm` 左右脚高差，两者合起来凭空造出了「`35/57` 帧质心出支撑面」
> 这个根本不存在的症状（诚实口径 `0/57`）。
> **(i) 重定向确实有缺陷，但只有一条，而且是全库的** —— 导出时没有地面约束，
> frame 0 全库 `73/73` 悬空 `6.46 .. 21.26 mm`；外加**另一条**独立缺陷 `9/73` 关节超限。
> 修法是一次 `2.44°` 的接地收尾解算，代价是站宽动 `0.77 mm`，**不需要重做重定向**。

**6. 下一步，分三档。**

**(a) 现在就能做（不动 artifact、不动门、不占 GPU）。**

1. **把接地收尾解算写成仓库里的一个函数**，签名与 `canonical_grounded_ready` 的 G1 对齐
   （`target_contact_preload_m = 0.5 mm`），但必须满足本条 3 证出来的**三个硬需求**：
   （i）自由变量是**十二个腿关节 + 根高**，不是四个踝；（ii）**脚放平是硬约束**，
   不是「最低顶点到位」就行（否则 LP 报 `does not expose a two-dimensional sole polygon`）；
   （iii）解完必须**逐关节校 MJCF 限位并 fail-closed**（本条 4 的探针没校，把 `2` 条推出了限位）。
   > **已交付并就地更正（i）（2026-08-07 稍晚，见「十一」2 与 5(i)）。**
   > 工具是 `hope_training/whole_body_tracking/scripts/ground_measured_clip_to_floor.py`，
   > 直接调仓库自己的 `solve_g1_donor_root`（必要时 `solve_g1_support_edge_projection`），
   > （ii）（iii）都由 G1 自带。**（i）里的「+ 根高」不是必需项**：只放开十二个腿关节、
   > root 逐位冻结，`Take_061_unit04_BH` `57/57` 帧、全库 `70/73` 条都能把两只碰撞鞋底
   > 放平压到 `-0.49 mm`，**而且拍子与整个上身的世界坐标位移是 `0.000e+00 mm`**；
   > 动根高会让拍子跟着沉 `24 mm`。改成「优先只解腿十二；解不出来（实测 `3/73`）才允许根高参与，
   > 并且必须把上身与拍子的位移一起报出来」。
2. **修 `SUPPORT_BAND_M` 那个退化多边形缺陷**（审计侧）：现役全局 `6 mm` 带在左右脚差 `4.8 mm` 时
   会塌掉半个多边形。**改门要连证据一起改**：同批交付
   （i）逐脚各自取带 + 整鞋底 footprint 两个口径都进报告；
   （ii）**变异测试要两头都能开火** —— 把左脚鞋底单独下移 `4.86 mm`（机器人不动）必须让裕度
   从 `-11.95` 跳到 `+26.37 mm`（证明旧口径在测量具不是测机器人），
   同时构造一个真正失衡的姿态必须仍然被判为负（证明新口径没变成空判）。
   > **已交付（2026-08-07 稍晚，见「十一」6）。** 逐脚取带 + 悬空脚不参与建多边形 +
   > 退化多边形出具名状态而不是负数；`14 passed → 22 passed / 0 skipped`，
   > 四种打回原形的变异逐一变红。
3. **把「腿是不是自己的」写进 LP 报告的自陈字段**：任何一次 `MujocoGroundContactLPSolver` 的收据
   都应该记下「本次输入的腿关节相对 frame 0 变化多少」。本条 2 那张表说明，
   这个字段是读懂 `FEASIBLE` 比例的唯一前提；没有它，`54/54 FEASIBLE` 和 `3/55` 看起来是同一种收据。

**(b) 需要重做 artifact（要排期，不要顺手做）。**

4. **全库 `73` 条重新 materialize 一版「接地后」的 measured bank**，然后重跑 §5.6.7「三」那张
   MuJoCo A/B —— 那张表里「腿一发 teacher 就倒」用的是**没接地**的 teacher 腿，
   接地之后还倒不倒，**是未测的**，而它是「卡点在腿」这个结论的唯一支柱。
5. **把出生姿态换成（接地后的）teacher frame 0**，按「九」6 的清单重铸
   （`dynamic_ready.v2` / `split_physical_hold.v2` / `frame0_exact.v1` / `a211_split_ready_lineage.v5`
   与钉死它们 SHA 的 launcher/materializer 断言）。**动作零点 `AGIBOT_A3_CFG.init_state.joint_pos`
   必须显式保持不变**，否则撞 `EXP-V2-REWARD-FREEZE` §0.13 的禁令。
   顺带按构造消掉 reveal 那个 `2.243 rad` 阶跃。
6. **`9/73` 关节超限**交回重定向侧单独修，接地解算修不了。

**(c) 需要 Franco 拍板。**

7. **`oracle32` 到底换成什么门。** 「五」已经提了拆成「运动学可解性证书 + 闭环策略可达性证书」，
   本条把可行性变成了实测，但**没有在本轮动 A 族的门**（它牵连 artifact 重铸）。
   建议的新判据、以及它必须自带的变异测试：

   | 新门 | 判什么 | 该拦的仍拦（变异测试） | 误拦的不再拦（变异测试） |
   | --- | --- | --- | --- |
   | **接地可解性证书**（离线，逐 clip） | 接地收尾解算能否在 MJCF 限位内让两只脚放平贴地，且站宽/上身改动小于阈值 | 把某帧的髋外展人为加 `30°`（解不出来）必须 `FAIL`；关节角故意超限必须 `FAIL`（现成样本：那 `9/73`） | `Take_061_unit04_BH` 必须 `PASS`（实测 `2.44°` / 站宽 `0.77 mm`） |
   | **静态站立证书**（离线，逐帧，就是现在这台 LP） | 接地后逐帧过 LP，**报比例不报全通过** | 把根整体抬 `0.2 mm` 必须回到 `no_contact`（「八」1 已证）；把质心人为移出鞋底必须 `INFEASIBLE` | 接地后的 `50/57` 必须被记成 `50/57` 而不是「失败」；**收据必须自陈腿关节变化量**，否则冻腿 clip 的 `100%` 会被误当成基准 |
   | **闭环可达性**（在线） | 交给训练后的 policy，不是零 PPO 的开环回放 | 用一个随机初始化 policy 必须 `FAIL` | 用一个训到位的 policy 必须 `PASS` |

   **一条都不放宽现有 fail-closed 判据**：LP 默认容差、`sole_floor` / `support_margin` 静态门、
   `20 mm` 台面余量、§12.3 的 `0/73` 机械准入，全部保留。变的是**用哪道门去问哪个问题**。
8. **要不要保「站宽 = 出生姿态」这条隐含约束。** 本节复核的结论是**不要保** ——
   它会把动捕真人的预备站位压成立正，毁掉重定向做得最准的那一项。
   但换出生姿态牵连 deploy 动作零点，这一步不能由 subagent 代签。

**收据。** pod1 `/workspace/hope_isaac_venv/bin/python`（`python 3.10.18` / `mujoco 3.10.0` /
`scipy 1.15.3` / `numpy 1.26.4`）；只读 checkout `/workspace/franco/adj_20260807`（`ffdca6af`，
自建，不与其它 workflow 的树共用）；探针与输出
`/workspace/franco/adj_p1.{py,json}`（几何 / 出生姿态 / 全库 frame 0 普查）、
`adj_p2.{py,json}`（动捕源 / LP 变异测试 / 老库 `16` 条逐帧 LP）、
`adj_p3.{py,json}`（接地变体 / 腿长口径 / 质心裕度）、
`adj_p4.{py,json}`（脚放平硬约束的四个接地变体 + 逐帧接触点数）、
`adj_p5.{py,json}`（全库 `73` 条 frame 0 接地后重判）。
审计变异测试 `14 passed / 0 skipped`（`hope_isaac_venv`）。
**未跑 Isaac、未占 GPU、未改任何仓库代码或门限、未写任何 artifact。**

**十一、那 `1 cm` 的单一成因找到了；而且接地不用动根，拍子逐位不动（2026-08-07，第六条 workflow，本条**改了代码**）。**

「十」6(a) 列的三件"现在就能做"，本条交付了前两件（接地函数、`SUPPORT_BAND_M` 退化多边形），
并对「十」的两处结论**就地更正**（本条 5）。前十条都在问"症状说明什么"；
本条先回答**上一层**：那 `1 cm` 到底是哪一行代码欠的。

**1. 单一成因：重定向的脚部目标少了一项 `~31 mm` 的"人的踝比机器人的踝高"补偿。查三层。**

**① 机制码。** GMR 的 IK 配置
`GMR/general_motion_retargeting/ik_configs/smplx_to_a3.json`（`chingmu_to_a3_gmr.py` 原样复用，
只把 `human_scale_table` 全改成 `1.0`）里，两个脚的任务是：

```
"left_ankle_roll_Link":  ["left_foot",  pos_cost 100, rot_cost 10, offset [0.0, +0.02, 0.0], ...]
"right_ankle_roll_Link": ["right_foot", pos_cost 100, rot_cost 10, offset [0.0, -0.02, 0.0], ...]
"ground_height": 0.0
```

也就是**把 `ankle_roll_Link` 的 body 原点，直接拉到人体 `left_foot`/`right_foot` 上**
（ChingMu BVH 里就是 `LeftFoot`/`RightFoot`，**踝关节中心**），位置偏移只有横向 `±0.02 m`，
**竖直方向是 `0`**；`ground_height` 也是 `0.0`（它在 `motion_retarget.py:145` 只被用作
`pos_offsets = pos_offset - ground`，等于没有）。驱动脚本再把 `actual_human_height` 设成 `None`
（缩放比 `1.0`）、喂**绝对**世界坐标、只减掉 xy 站位。
**整条链路没有任何一项去问"这个人的踝离地多高、A3 的踝离鞋底多高"。**

**② 活值（同一把尺子，两边都量）。** 用与驱动脚本**逐行相同**的 BVH FK
（`M_YUP_TO_ZUP @ pos_cm / 100`）重算 GMR 的输入，对 v4 全库 `73` 条的 frame 0：

| 量（`73` 条 frame 0） | 左 | 右 |
| --- | ---: | ---: |
| **人体踝关节中心离地** | `+91.3 .. +116.6 mm`，均值 **`+98.8`** | `+85.9 .. +111.3 mm`，均值 **`+97.1`** |
| **重定向后机器人踝原点离地** | `+87.7 .. +111.2 mm`，均值 `+93.7` | `+83.0 .. +136.8 mm`，均值 `+92.0` |
| 两者之差（IK 残差，负 = 机器人踝更低） | `-15.4 .. +4.0 mm`，均值 **`-5.1`** | `-19.4 .. +39.4 mm`，均值 **`-5.1`** |
| **A3 踝原点在鞋底放平贴地时该在** | **`+67.46 mm`** | **`+67.46 mm`**（与「七」4 的 `0.06746 m` 逐位一致） |

**GMR 确实把机器人的踝原点放到了人的踝关节上**（差 `-5.1 mm`，是 IK 里骨盆任务同权重 `100` 抢出来的残差）。
于是 `98 mm − 67.5 mm ≈ 30.6 mm` 这一项**没有任何人补**：脚要是放平，鞋底就该在 `+26 mm` 左右
（「七」4 量到 `+24.7 / +24.5`，一致）；脚一倾斜（本 clip 左 `3.62°`、右 `5.96°`）外缘压下去一截，
剩下的就是「十」1 更正后的实测 `+6.46 .. +21.26 mm`。

**③ 动捕的地面标定没错。** 同一批帧，人体**最低关节**（`RightToeBase` `62` 条 / `LeftToeBase` `11` 条）
的高度是 `-9.5 .. -1.2 mm`，均值 `-5.5 mm`。**地板就在 `z = 0`，人是站在地上的。**

**一句话：不是坐标系错一档，也不是"少跑了一步"，是重定向的脚部目标少了一个 `~31 mm` 的常数项 ——
人的脚从踝到地比 A3 的脚从踝到鞋底"高" `~31 mm`。** 这解释了为什么它是**全库**的、
为什么每条 clip 的量不一样（脚的倾角逐帧不同）、也为什么「七」1 那四条独立路径全都说"测量没错"。

**能不能从源头修？能，但不该只修源头。** 在 IK 配置里给两个踝任务加一个 `-0.031 m` 的竖直偏移，
可以把全库整体搬到大致正确的高度；但它**修不好**左右脚不共面（`±5 mm`，源自真人自己的重心分配，见「六」4）
和逐帧 `±5 mm` 的 IK 残差，而且要重跑 GMR 全库、作废下游全部 SHA。
**正确做法是两件都做**：源头补上那一项（在 `yikang` 的重定向侧，不在本仓库），
下游保留一次接地收尾解算做 fail-closed 收口。**本轮只做后者。**

**2. 接地：根一个 bit 不动，只解十二个腿关节；拍子的世界坐标逐位不变。**

工具 `hope_training/whole_body_tracking/scripts/ground_measured_clip_to_floor.py`（未授权诊断，
只出报告、不写 artifact）。它不自造解算器，直接调仓库自己的
`canonical_grounded_ready.solve_g1_donor_root`（G1：root 与 `19` 个非腿关节逐位冻结，
每只脚的地面投影位置与 yaw 保持原值，两块**碰撞**鞋底放平贴到 `target_contact_preload_m = 0.5 mm`），
只有当 G1 留下红门时才升级到同模块的 `solve_g1_support_edge_projection`（G1S：多一次由
"最紧那条支撑边"推出来的、两只脚共同的地面内平移）。

**为什么根不动**（这是对「十」6(a)(i) 的更正，见本条 5）：缺陷长在**脚**上 ——
踝被放高了 `31 mm`；骨盆是按人的骨盆匹配的，那一项 retarget 做对了。
把根整体压下去 `24 mm`，等于把**拍子也压下去 `24 mm`**，是拿动作去补量具。
只解腿的代价是腿多动几度，换来的是**上身完全不动**。

`Take_061_unit04_BH` frame 0 逐关节改动（G1；G1S 的值在括号里，差 `< 0.2°`）：

| 关节 | 改动 | 关节 | 改动 |
| --- | ---: | --- | ---: |
| `left_knee` | `-6.360°`（`-6.460`） | `right_knee` | `-5.715°`（`-5.786`） |
| `right_hip_pitch` | `+4.514°`（`+4.384`） | `right_hip_yaw` | `-4.469°`（`-4.418`） |
| `left_hip_pitch` | `+3.641°`（`+3.578`） | `right_ankle_roll` | `+2.484°`（`+2.431`） |
| `right_ankle_pitch` | `-2.410°`（`-2.244`） | `left_hip_yaw` | `+1.393°`（`+1.424`） |
| `left_hip_roll` | `+1.258°`（`+1.199`） | `left_ankle_pitch` | `-0.991°`（`-0.816`） |
| `left_ankle_roll` | `+0.814°`（`+0.871`） | `right_hip_roll` | `-0.525°`（`-0.488`） |
| **其余 `19` 个非腿关节** | **`0.000000` rad（逐位）** | **root 位置 / 姿态** | **逐位不动** |

**为什么必须是这十二个而不是别的**：把两只脚从"悬空且倾斜"放到"贴地且放平"，
每只脚要满足 `3` 个位置 + `3` 个姿态共 `6` 个约束，两只脚 `12` 个约束；
在 root 冻结的前提下，能动的自由度恰好只有这条腿链上的 `6` 个关节 × `2` = `12` 个。
**`12` 对 `12`，一个不多一个不少**；这也是 `leg_to_foot_jacobian` 门要求 `rank = 12` 的原因，
实测 `57/57` 帧、`70/73` 条全 `PASS`。

**3. 逐项复验（`Take_061_unit04_BH`，全 `57` 帧，pod1 `hope_isaac_venv`）。**

| 门 | 接地前 | 接地后 |
| --- | ---: | ---: |
| `sole_floor` | `0/57` | **`57/57`** |
| `double_support` | `0/57` | **`57/57`** |
| `joint_limits` | `57/57` | **`57/57`** |
| `collision` | `57/57` | **`57/57`** |
| `foot_pose` | `57/57` | **`57/57`** |
| `leg_to_foot_jacobian` | `57/57` | **`57/57`** |
| `support_margin` | `0/57` | **`57/57`**（G1 收 `31`，其余 `26` 帧由 G1S 收，共同平移 `1.74 .. 25.42 mm`） |
| 整份收据 `verdict` | `FAIL_STATIC_GROUNDED_READY` | **`PASS_STATIC_GROUNDED_READY_CANDIDATE` `57/57`** |

几何数字：

| 量 | 接地前 | 接地后 |
| --- | --- | --- |
| 碰撞鞋底最低顶点 | `+12.920 .. +19.603 mm` | **`-0.4942 .. -0.4888 mm`**（目标 `-0.5`） |
| 踝原点高度 | `+91.1 .. +96.3 mm` | `+66.97 mm`（= `67.46 − 0.49`） |
| 脚面倾角 | 左 `3.62°` / 右 `5.96°` | `< 0.002°` |
| 站宽（两踝水平距离） | `0.61120 .. 0.61129 m` | `0.61120 .. 0.61130 m`，**逐帧最大变化 `0.0119 mm`** |
| 质心裕度（门用的接触点多边形） | 建不出来（`ncon = 0`） | `+0.5 .. +19.2 mm`，`0/57` 为负 |
| 质心裕度（两只**完整**鞋底投影） | `+86.2 .. +128.7 mm` | `+114.3 .. +134.7 mm`，`0/57` 为负 |

**动作没被破坏，这是实测不是断言。** root 与 `19` 个非腿关节逐位不动，所以从骨盆往上
（腰、双臂、拍子）的正运动学按构造不可能变 —— 本工具把它当判据实测出来：

> **`right_racket` site 的世界坐标位移：`0.000e+00 mm`，`57/57` 帧。
> 骨盆及以上全部非腿 body 的世界坐标位移：`0.000e+00 mm`，`57/57` 帧。
> `19` 个非腿关节的最大改动：`0.000e+00 rad`。**

「十」3 那条"根高 `-24.01 mm` + 腿 `2.44°`"的做法会让**拍子和整个上身一起下沉 `24 mm`**；
本条的做法是 `0 mm`。两者都能把脚放到地上，代价不在同一个量级。

**4. 全库 `73` 条 frame 0：`0/73` 变成多少。**

| | 接地前 | 接地后（G1，必要时 G1S） |
| --- | ---: | ---: |
| 解算成功 | — | `70 / 73`（`3` 条解不出来，见下） |
| `sole_floor` | `0 / 73` | **`70 / 73`**（= 解出来的全部） |
| `double_support` | `0 / 73` | **`70 / 73`** |
| `foot_pose` | `73 / 73` | `70 / 73` |
| `leg_to_foot_jacobian` | `73 / 73` | `70 / 73` |
| `joint_limits` | `48 / 73` | `47 / 73`（接地**没有**改变这一项，见本条 5 的更正） |
| `collision` | `69 / 73` | `67 / 73`（接地**没有**改变这一项） |
| `support_margin` | `0 / 73` | `34 / 73` |
| **两门口径（`sole_floor + double_support`，就是 `0/73` 判的那一件事）** | **`0 / 73`** | **`70 / 73`** |
| **六门口径（Franco 点名的那六道）** | `0 / 73` | **`44 / 73`** |
| **七门口径（再加 `support_margin`）** | `0 / 73` | **`26 / 73`** |
| 鞋底最低顶点 | `+6.463 .. +37.700 mm` | `-0.5000 .. -0.4812 mm` |
| 站宽 | `0.5060 .. 0.6690 m` | 同上，**逐条最大变化 `0.0178 mm`** |
| 腿关节最大改动 | — | `4.64 .. 13.26°`（最吃力的是 `right_knee`） |
| 拍子 / 上身位移 | — | **`0.000e+00 mm`，`70/70`** |
| 质心裕度（两只完整鞋底投影） | — | `+34.2 .. +139.8 mm`，**`0/70` 为负** |

**所以 §12.3 那条 `0/73` 退役判决要重新裁定。** 它判的是"直接拿 exact frame 0 当物理出生，
同门槛扫描 `0/73`"，而「九」1 已经证明那 `0` 是同一个 `1 cm` 被数了 `73` 遍。
现在把那 `1 cm` 修掉之后：**它判的那件事（脚踩没踩到地 + 双支撑）从 `0/73` 变成 `70/73`。**
判决本身不是错的（当时确实 `0/73`），**但它的原因已经消失**，因此
「exact frame 0 只能保留为 teacher frame0、不能当物理出生」这条推论**不再成立**，
`materialize_action_ball_a211_frame0_exact_artifact.py` 在
`docs/operations/tool_catalogue.md:128` 的
`SUPERSEDED / COUNTEREXAMPLE` 标签**应当重判**（重判要连 artifact 重铸一起走，见「十」6(b)，本轮不代签）。
剩下的 `3/73` 不通过两门口径，是**解不出来**（下段），不是"脚放不到地上"。

**解不出来的三条**：`Take_062_unit10_BH` / `Take_063_unit04_BH` / `Take_064_unit04_BH`，
报 `G1 leg continuation did not reach the requested foot poses`。
**不是求解预算问题** —— 把 `continuation_steps` 从 `12` 提到 `96`、每步迭代 `80 → 400`、
步长上限 `0.12 → 0.04 rad`（这三个都是**预算**不是容差，验收容差一个没动），三条依旧解不出来。
它们的共同点是站宽在全库最宽的一档（`0.627 / 0.672 / 0.676 m`，全库上限 `0.669..0.676`）
且左脚要下沉 `18 .. 21 mm`：**在骨盆钉死的前提下，腿已经伸不到那么远。**
这三条要么允许根高动一点（「十」3 的做法），要么允许脚在地面内平移（G1S，但它第一步就是 G1，所以也一并被拒）。
**这是三条真正的、只靠"冻住根"修不了的样本，不是量具问题。**

**5. 就地更正「十」的两处（本条有直接实测，故写在这里，「十」处已加指针）。**

**(i) 「十」6(a) 第 1 条写"自由变量是十二个腿关节 + 根高，不是四个踝"——**
"不是四个踝"成立，"**要根高**"不成立。实测：只放开十二个腿关节、root 逐位冻结，
`Take_061_unit04_BH` `57/57` 帧、全库 `70/73` 条都能把两只碰撞鞋底放平压到 `-0.49 mm`，
七门全绿。**根高是可选项，不是必需项；而且用了它就要付 `24 mm` 的拍子位移。**
建议改成"**自由变量优先只取十二个腿关节；只有解不出来（实测 `3/73`）时才允许根高参与，
并且必须把上身与拍子的位移一起报出来**"。

**(ii) 「十」4 那条"顺带抓到一条没人记过的独立缺陷：`9/73` 条 clip 的 frame 0 关节角本身就出
MJCF 限位，接地解算修不了，必须在重定向侧单独处理"—— 这是 float32 存储精度，不是重定向缺陷。**
把每条 clip frame 0 的 `joint_pos` 逐关节对 `model.jnt_range` 量出**超出量的绝对值**：

| | 实测 |
| --- | --- |
| 超出审计容差（`joint_limit_tolerance_rad = 1e-10`）的 clip | `25 / 73` |
| 最大超出量 | **`6.767e-08 rad` = `3.9e-06` 度** |
| 涉及的关节 | `right_shoulder_pitch`（限位 `-165.000°`，超 `6.767e-08 rad`，`9` 条 —— **正是「十」记的那 `9` 条**）；`right_wrist_yaw`（限位 `+93.000°`，超 `4.616e-09 rad`，`16` 条） |
| 换算成 float32 的一个最低位 | `6.767e-08 / 1.192e-07` = **`0.57` 个 ulp** |
| clip 里那个值是不是"限位在 float32 上的最近可表示数" | **`25/25` 全是** |

bank 的 `joint_pos` 存的是 **float32**；重定向把关节顶到限位、存成 float32、再读成 float64，
就会落在限位外**不到一个 float32 最低位**的地方。而审计容差 `1e-10 rad` 比 float32 在该角度上的
分辨率（`1.2e-07 rad`）严 **`1000` 倍**。
**所以这不是"回重定向侧单独处理"的活**，两条正当修法二选一：
把这道门的容差抬到 float32 量级（`~2e-07 rad`，并写清理由），或者把 bank 的 `joint_pos` 存成 float64。
在没定之前，`joint_limits` 那 `25/73` 的红灯**不含任何运动学信息**。

**6. 顺带交付：`SUPPORT_BAND_M` 那个退化多边形是真 bug，已修，带变异测试。**

原来的写法把 `6 mm` 接触带锚在**两只脚合起来**的最低顶点上，于是较高那只脚能进多少支撑点，
取决于左右脚差多少毫米，而**不取决于它到底踩没踩到地**。改法（`audit_measured_teacher_executability.py`）：

1. **每只脚的带锚在它自己的最低顶点上**；
2. **一只脚只有它自己的最低顶点落进地面 LP 窗口（`[-2, +2] mm`）时才参与建多边形** ——
   悬空的脚不再贡献支撑点。这一条以前是漏的，所以"整条 clip 悬空 `1 cm`"也照样能算出一个支撑多边形；
3. 多边形少于三个顶点、或最小宽度 `< DEGENERATE_SUPPORT_WIDTH_M = 0.1 mm`，
   **不出裕度数字，出一个具名状态**（`DEGENERATE_SUPPORT_POLYGON` / `NO_FOOT_ON_FLOOR` /
   `SINGLE_FOOT_SUPPORT` / `DOUBLE_SUPPORT`）；报告里 `frames_com_outside_support` 只统计
   **裕度有定义且为负**的帧，建不出多边形的帧进另一个计数器 `frames_support_polygon_undefined`。
   **"量具失效"从此不会被读成"质心出界"。**

**变异测试两头都能开火**（`hope_isaac_venv`，`14 passed → 22 passed / 0 skipped`）。
把修好的代码按四种方式打回原形，逐一确认对应的测试变红、其余仍绿：

| 变异 | 结果 |
| --- | --- |
| `M1` 把带宽锚回"两只脚合起来的最低点"（原 bug） | `1 failed`（`test_the_band_is_anchored_per_foot_not_to_the_lower_foot`） |
| `M2` 让不在地上的脚重新参与建多边形 | `3 failed` |
| `M3` 把 `DEGENERATE_SUPPORT_WIDTH_M` 改成 `0`（不再认退化） | `1 failed` |
| `M4` 把 `SUPPORT_BAND_M` 放宽 `10` 倍 | `1 failed`（**粗一档就过不了**） |
| 还原 | `22 passed` |

"该拦的仍拦"由 `test_mutation_lifting_one_foot_off_the_floor_removes_it_from_the_polygon` 钉住：
一只脚真的抬离地面、质心确实落在剩下那只脚外面时，新口径仍然报**负**裕度。

**7. 另外两处量具问题 + 一处解算器脆性：本条只记录，不改门。**

**(a) `support_margin` 那道门用 MuJoCo 的接触点当支撑多边形，而 MuJoCo 每只脚只给 `3` 个。**
在接地后的 `Take_061_unit04_BH` frame 0 上实测：真正贴到地（`≤ 0.5 mm`）的碰撞鞋底顶点有
**`123 / 704`** 个，张成 `0.183 m × 0.109 m`；而 `data.contact` 只报 `3` 个点，
三点凸包 `55.1 cm²` 对这只鞋底完整投影的 `284.8 cm²`；两只脚合起来门看到的是 `848 cm²`，
真实是 `2081 cm²`。于是同一个姿态，门报 `-0.99 mm`（判"质心出支撑面"），
两只完整鞋底口径报 `+112.8 mm`。全库接地后 `36/70` 条被这条判负，
而完整鞋底口径 **`0/70` 为负**（`+34.2 .. +139.8 mm`）。
**这是与本条 6 同一族的量具问题，只是它长在准入门上（`canonical_grounded_ready._support_margin`），
改它会牵连仓库里每一份 ready 收据的 SHA，所以本轮只出证据不动门。**

**(b) `joint_limits` 的 float32 问题**，见本条 5(ii)。

**(c) `canonical_grounded_ready.ReadyState.__post_init__` 每次构造都做 `q / ‖q‖`，而这个操作
在浮点上不幂等。** 实测 `Take_061_unit04_BH` 的 `57` 帧里有 **`15` 帧**第二次归一化就改最低位，
其中 frame `26` 连第三次都还在变。G1 解完之后用 `np.array_equal` 逐位比对 root 有没有被动过，
于是这些帧会被自己的护栏以 `G1 changed the donor root` 拒掉 —— **拒的是浮点尾数，不是解算器**。
本工具先把四元数迭代到"归一化的不动点"（`q/‖q‖` 逐位等于 `q`）再送进去，迭代不收敛就 fail-closed；
全库扫描的解算拒绝因此从 `5` 条降到 `3` 条（剩下的 `3` 条是本条 4 那三条真解不出来的）。
**这是 `canonical_grounded_ready` 的一处脆性，本轮不改它。**

**收据。** pod1 `/workspace/hope_isaac_venv/bin/python`（`python 3.10.18` / `mujoco 3.10.0` /
`scipy 1.15.3` / `numpy 1.26.4`）；自建 worktree `/workspace/franco/gnd_20260807`（`2f7e9394`），
基线对照 worktree `/workspace/franco/gnd_base_20260807`（同一 commit，未打补丁）；
探针 `/workspace/franco/gnd_probes/p{1,2,3,4,5,6,7}_*.py`
（`p1` 全库 frame 0 机器人几何、`p2` 与驱动脚本逐行相同的 BVH FK 与人体踝高、`p3` 门的接触点凸包
对整鞋底投影、`p4` 面积量化 + G1S 定价、`p5` 鞋底平不平 + 关节限位普查、`p6` 限位 / 自碰撞明细、
`p7` 三条解不出来的 clip 加预算复跑）、变异脚本 `mutate.sh`；
接地报告 `/workspace/franco/gnd_probes/g2_take061_all.json`（`57` 帧）与
`g2_lib_f0.json`（全库 `73` 条 frame 0）。
测试：`test_audit_measured_teacher_executability.py` 基线 `14 passed / 0 skipped` →
本条 `22 passed / 0 skipped`；新增 `test_ground_measured_clip_to_floor.py` `19 passed / 0 skipped`
（都在 `hope_isaac_venv`，`CUDA_VISIBLE_DEVICES=` 纯 CPU）。
**整包对拍**（`hope_training/whole_body_tracking/tests` 全跑，同一解释器、`-p no:randomly`
`--continue-on-collection-errors`，两棵树各跑一遍）：

| | 基线 `gnd_base_20260807`（未打补丁） | 本条 `gnd_20260807` |
| --- | --- | --- |
| 结果 | `376 failed, 6913 passed, 62 skipped, 28 errors`（`32:27`） | `376 failed, **6940** passed, 62 skipped, 28 errors`（`34:06`） |

**`failed` / `skipped` / `errors` 三项逐位相同，`passed` 恰好多 `27`** ——
`8`（审计变异测试新增）`+ 19`（接地工具新测试），没有一条别的动过。
那 `376 failed / 28 errors` 是**基线本来就有的**：这两棵树都是 detached worktree，
一批 artifact/vendor-identity 测试要求干净 checkout 与特定 commit，本条不碰它们。
四元数不动点另跑一次穷举:bank 里全部 `5107` 个 root 四元数都收敛到不动点，
与朴素归一化的最大差 `2.31e-16`（一个最低位）。
**未跑 Isaac、未占 GPU、未写任何 artifact、未放宽任何门限**；
改动是两个已有文件加两个新文件：审计工具的支撑多边形口径、它的变异测试、
新增的接地工具与接地工具的测试。

---

### 十二、「出生就该是第 0 帧、等待就是 hold 住第 0 帧」这件事：脚的账已经平了，卡点换成了腰（2026-08-07 实测）

> **本条已由「十四」用另写的一份脚本从零重算：主结论逐位成立
> （`waist_pitch` 需求 `-49.15464109801158` 一个 bit 都不差），只有电机限幅那个数
> `118.2` 要改成 `118.0`（`118.2` 是踝俯仰的）。**

**Franco 的原话（08-07）**：「出生不应该是第 0 帧吗？等待也是啊？」
08-05 也说过同一句：「task 里面的前序等待是输入的就是第 0 帧的值，只是 time to start
就是给还需要多久动作开始这件事。」——**这是设计意图，本条去把它接进去，结果撞到一堵新墙。**
先说结论：**脚那条路已经通了，腰这条路没通，而且不是我们能自己拍板的那一类。**

#### 1. 先按 Franco 说的做了：给第 0 帧解那一次 hold LP，没有另造姿态

本轮**没有**走"因为 frame 0 撑不住所以再造一个出生姿态"的老路。做的就是他说的那件事：
把接地后的 measured frame 0（`root` 与 `19` 个非腿关节逐位不动、只重解 `12` 个腿关节，
即「十一」那一步的产物）当成物理出生，**给它解一次 hold LP** 求可执行的 `q_des`。
仓库里本来就有这条支路：`materialize_a3_dynamic_ready_contract.py` 的
`--physical-birth-composition-mode projected_teacher_frame0_grounded`。它从来没被跑通过，
文档里写着"remains a failure baseline"——**那句话的理由是「十一」推翻的那个 `0/73`，理由已经不成立了，所以本轮重跑它。**

跑的结果：**接地那一段全过了**（G1S 静态几何 + 地面 LP + 支撑裕度都 PASS，
root 与非腿关节逐位未动、拍子 site 的 FK 逐位未动），**卡在最后一步 hold LP**，
原话是 `no static double-support hold exists inside the executed qdes envelope`。

#### 2. 这次不猜：那个"无解"有一个唯一、可自证的原因

关键是一条运动学事实：**两只脚都不在腰以上那截身体里**，
所以脚底的支撑力在腰、双臂、脖子这 `19` 个关节上产生的力矩**恒等于零**（接触雅可比的这些列全是零）。
于是对这 `19` 个关节，"撑住这个姿态要出多大力矩"根本**没有解算自由度**，
它就等于 MuJoCo 的 `qfrc_bias`（零速度零加速度下的重力项）。
而策略这一侧只发位置指令，静止时能产生的力矩上限就是 `kp × (q_des 还能走多远)`，再被电机限幅一刀。
两者一比，不用解任何 LP 就能判——而且这是**必要条件**，不满足就一定 hold 不住。

现役 A3、接地后的 `Take_061_unit04_BH` frame 0，两个关节不满足：

| 关节 | 撑住这个姿态需要 | 位置指令最多能到 | 差 | 卡在哪 |
| --- | --- | --- | --- | --- |
| `waist_pitch_joint` | **`-49.155 N·m`** | `[-21.704, +15.053] N·m` | **`27.451 N·m`** | `kp × 行程`（不是电机） |
| `waist_roll_joint` | `-14.243 N·m` | `[-10.020, +18.255] N·m` | `4.223 N·m` | 同上 |

`waist_pitch` 那一条换成人话：要撑住它，`q_des` 得发到 **`-0.9515 rad`**，
而这个关节的指令量程下限是 `-0.4433 rad`、**机械限位**下限是 `-0.4887 rad`。
**把 `5%` 投影内沿和 `2%` 硬内沿全部丢掉、指令顶死到机械限位，也只能出 `-26.014 N·m`，
是需求的 `52.9%`——差的是将近 2 倍，不是一个容差**（现役包络内是 `-21.704`，`44.2%`）。
电机反而是宽裕的：限幅 `118.0 N·m`，只用到 `41.66%`。
**卡的是增益乘以可用行程，不是电机。**（"`6 N·m` 握不住拍子"那条已被推翻两次，这里第三次：本条卡的不是腕，是腰。）

**接地这一步跟这个卡点无关，是实测不是断言**：接地只动 `12` 个腿关节，
腰以上的正运动学按构造不变，所以腰的重力力矩在接地前后**逐位相同**
（原始 frame 0 与接地后 frame 0 都是 `-49.15464109801158 N·m`）。
**换句话说：就算「十一」那 `1 cm` 一点都没修，这堵墙也在原地。两件事互不相欠。**

#### 3. 查三层，三层一致

**① 机制码**：`configs/a3_vendor_runtime_authority_20260802_r9/bh_loop_c.shared_ready.training_contract.json`
给 `waist_roll` / `waist_pitch` 的 `kp` 都是 `50`（`waist_yaw` 是 `85`），
`q_des` 被投影进关节量程再收 `5%` 内沿；动作项是纯位置目标，没有任何前馈力矩通道。

**② 活值**：上表。全 `57` 帧、全库 `73` 条的普查见本条 4。

**③ 厂商 deploy**：`agi/a3_deploy_example/.../include/a3_policy_parameters.hpp`
里**并排放着两套增益**，注释写明来源是
`aimrt_motion_control_a3/.../robot/a3_t2d5/pd_stand/default.yaml`：

| | `waist_yaw` | `waist_roll` | `waist_pitch` |
| --- | --- | --- | --- |
| `a3_kps`（策略跑的那套，`:96`，与我们的 `contract` 一致） | `85` | `50` | `50` |
| `a3_pd_stand_kps`（**真机站立/保持用的那套**，`:194`） | `400` | **`500`** | **`500`** |

**厂商自己就是分两套增益的：挥拍用软的，站住用硬的，腰上差 `10` 倍。**
`kp = 500` 时 `waist_pitch` 在同一条执行包络内能出 `500 × 0.434 = 217 N·m`，
这时轮到电机限幅 `118.0 N·m` 当边界——对需求 `49.2` 仍然**够用还有一倍多**。

**而且这不是"厂商代码里躺着一组没人用的常数"——它是一个真实的运行模式。**
`a3_pingpong_main.cpp:7` 写着模式阶梯 `PASSIVE（软趴）→ PD_STAND（保持 nominal）→ SHADOW（算不发）→ MOTION（发）`，
`pp_policy.hpp:743` 的 `official_stand_kp()` 直接返回 `a3_pd_stand_kps`（`:606`）。
**"保持一个姿态"在厂商那里是一个单独的模式、配单独的增益；策略跑的 `MOTION` 用的是软的那套。**
再往下一层还有一件对得上的事：`PD_STAND` 保持的那个姿态是 `a3_default_angles`，
也就是「十」查过的、我们现役出生姿态所用的**同一个 URDF 默认姿态**。
**厂商用站立增益去保持它自己的默认姿态；我们在用挥拍增益去保持一个动捕运动员的预备架势。**

**所以这句话是可以直说的：不是"第 0 帧这个姿态不对"，是"我们在用挥拍的增益去要求一个站立的保持"。**
Franco 说的"hold 住 frame 0 仍然需要重力补偿，那是物理不是缺陷"完全成立；
这一轮的发现是：**那份重力补偿，现役这套增益开不出来。**

#### 4. 普查：这不是 frame 0 一帧的事，也不是一条 clip 的事

新工具 `audit_position_hold_authority.py`（未授权诊断，只出报告）：

| 口径 | 结果 |
| --- | --- |
| `Take_061_unit04_BH` **全 `57` 帧** | **`0/57`** 位置指令撑得住；`waist_pitch` `57/57` 不够、`waist_roll` `19/57` 不够；最大缺口 `27.451 N·m` |
| 全库 `73` 条的 frame 0 | **`72/73`** 撑不住（唯一一条是 `hope_Take_062_unit00_BH`）；其中 `69` 条是**增益不够**（`waist_pitch 66`、`waist_roll 18`、`right_shoulder_pitch 4`、`right_wrist_yaw 1`），最大缺口 `32.233 N·m` |
| 同上，另一种病 | `42/73` 有关节**站在可发指令的 `q_des` 包络之外**（`right_wrist_yaw 41`、`right_shoulder_pitch 17`、`right_elbow 6`、`waist_pitch 5`、`right_shoulder_yaw 2`）——这时连"零力矩"都发不出来，毛病在限位/姿态，不在增益，**工具把这两种病分开命名，不混报** |
| **现役出生姿态**（`teacher_yaw_aligned_full_seed`，那份 `dynamic_ready` artifact） | **撑得住，缺口 `0.0`**（`waist_pitch` 只要 `-18.746 N·m`，在 `±21.7` 里） |

最后一行是这条判据的**"误拦的不再拦"**正面证据：同一把尺子拒接地后的 frame 0、放行现役出生姿态，
它不是一把"什么都拒"的尺子。而 `-18.746` 这个数与 §5.6.2 早就记下的
"WAIT hold 丢掉 `18.7 N·m` 于 `waist_pitch`"**逐位吻合**，说明尺子量的是同一件东西。

顺带解释了一件旧事：`34f8cf25` 那次 ramp 实验的终止全是 `robot_hit_table 32/32`，
平均活 `31.41 / 35` tick。**腰撑不住上半身、人往前栽到桌子上**，与这里的数字是同一个故事。

#### 5. 交付：把那个"无解"改成会自己说话的拒绝（记录 + 阻断同批）

原来 hold LP 一旦无解，只抛一句 `no static double-support hold exists inside the executed qdes envelope`——
**没人能从这句话里知道该去修什么**，于是这条支路被记成"failure baseline"就再没人回头看。本轮改成：

- `materialize_a3_dynamic_ready_contract.py` 新增
  `contact_free_actuated_rows()`（**从当前 MJCF 实测**哪些行地面永远使不上力，不是写死名单）、
  `static_hold_required_generalized_force()`、`contact_free_hold_torque_shortfall()`；
- **拒绝的那一行现在直接报出**：哪个关节、需要多少 `N·m`、位置指令能到多少、差多少、
  需要多大的 `q_des`、执行包络是多少、卡的是**电机限幅**还是 `kp × 行程`。实测输出：
  > `... : waist_roll_joint needs -14.243 N*m but a position command can only reach [-10.020, +18.255] N*m (short 4.223 N*m, limited by kp_times_available_qdes_travel, kp=50, motor limit 46 N*m; it would need q_des=-0.3672 rad and the executed envelope is [-0.2827, +0.2827] rad); waist_pitch_joint needs -49.155 N*m ...`
- **这道门一格都没有放松**：条件仍是 `not solution.feasible`，拒绝仍是拒绝，只是拒绝会自陈了。
  attribution 自己算不出来时退回原来那句话，不吞异常也不放行。

**变异测试 `16 passed / 0 skipped`**（新增 `test_audit_position_hold_authority.py`；`materialize` 那一支自己的 `35 passed` 一条没动），两头都开火：

| 变异 | 必须发生 |
| --- | --- |
| 把腿上的关节也当成 contact-free（忘掉 `contact_free`） | 膝盖被误点名 → `test_mutation_a_ground_loaded_joint_is_never_named` 抓住（**误拦方向**） |
| 只查上边界不查下边界 | take061 全部 `57` 帧静默通过 → `test_mutation_only_checking_the_upper_side_would_miss_this_entire_finding` 抓住（**该拦仍拦**） |
| 把可达力矩区间放宽 `2` 倍 | **仍然必须拒**（真需求 `-49.155` 对 `-43.4`，还差 `5.7 N·m`）——粗一档就过不了；放到 `3` 倍才够 |
| 把"关节站在包络外"当成"增益不够"报 | `test_a_joint_parked_outside_the_qdes_envelope_gets_its_own_name` 抓住（会把人引去调错的旋钮） |
| 现役出生姿态 | 一条都不许报 |

#### 6. 被这堵墙挡住、因此**本轮没有做**的四件事（不是忘了，是不该由 subagent 代签）

任务书要求"退役被取代的结构"。**取代它们的东西还没建成，所以一件都没退**：

- **split-ready 那套独立出生姿态与 `dynamic_ready` artifact**：**保留**。
  现役这份是目前**唯一**能被位置指令撑住的出生姿态（本条 4 最后一行）。
  在腰增益这件事定谳之前退掉它，等于让系统没有出生姿态。
- **reveal bridge（`34f8cf25` 的 ramp）**：**保留**。它的退役前提是"出生 = frame 0 之后阶跃按构造归零"。
  任务书要求"先证明确实归零了再删"——**量了，没归零**：拿现役 artifact 自己的
  `hold_qdes_joint_pos_rad`（等待期真正发下去的指令）对 `teacher_reference.joint_pos_rad`（揭示那一刻的指令）逐关节比，
  **最大阶跃 `2.2227 rad`（`right_wrist_yaw`）、`L2 = 3.8208 rad`、`31` 个关节里 `30` 个非零**，
  另加骨盆下沉 `176.6 mm`。**前提不成立，所以不删。** 遥测字段一个没动。
- **A/C 两族同改**：本轮没有改任何出生姿态，所以两族**天然仍然只差 obs 和 reward**。
  `53040fb0` 那道门**实跑过**，不是嘴上说的：在本轮打了补丁的树上
  `test_action_ball_211_ac_family_config_parity.py` `23 passed / 0 skipped`。
- **"出生姿态与 frame 0 不一致就拒绝"这道新门：本轮不加。**
  它会立刻拒掉现役唯一能用的出生姿态，等于用一道门把系统关停。
  这道门的正确落地时机是腰增益定谳、`projected_teacher_frame0_grounded` 真能出产物之后，
  与那次 artifact 重铸**同批**落地——那时它才是"防止漂回去"，现在加只是"把灯砸了"。

#### 7. 需要 Franco 拍板的一条（本条的真正出口）

**要让"出生 = 接地后的 frame 0、等待 = hold 住它"成立，只有一个已知的旋钮：腰的保持增益。**
厂商自己的做法摆在那里（`pd_stand` 的 `500` 对策略的 `50`），三条路各有代价：

1. **等待期切到厂商 `pd_stand` 增益，任务开始再切回策略增益**——最贴近厂商真机的做法，
   但引入"两套增益 + 切换时刻"，动作零点 `action_scale = 0.25 × effort / stiffness` 也跟着变，
   整条 runtime contract 与下游全部 artifact 要重签。
2. **只把腰的 `kp` 提上去**（`waist_roll` / `waist_pitch` `50 → ?`）——改动面小，但那是 plant 改动，
   会改变整条 clip 的可跟踪性，且现役 `contract` 的 `kp` 是与厂商 `a3_kps` 对齐的，一改就不再对齐。
3. **接受出生姿态与 frame 0 不同**（现状），代价是揭示那一刻仍然有阶跃，reveal bridge 继续留着。

**这一条超出 subagent 的授权范围**（改 `kp` = 改 plant = 作废下游全部 SHA），本条只把账摆平：
脚的债已经还清（「十一」），剩下的债只有腰这一笔，而且它的名字、数字、和厂商的对照都在上面。

**收据。** pod1 `/workspace/hope_isaac_venv/bin/python`（`python 3.10.18` / `mujoco 3.10.0` /
`scipy 1.15.3` / `numpy 1.26.4`），`CUDA_VISIBLE_DEVICES=` 纯 CPU；
自建 worktree `/workspace/franco/birth_20260807`（`559b95f4`），
基线对照 worktree `/workspace/franco/birth_base_20260807`（同一 commit，未打补丁）；
报告 `/workspace/franco/birth_out_20260807/hold_authority_take061.json`（`57` 帧 + 现役 artifact）与
`hold_authority_lib_f0.json`（全库 `73` 条 frame 0）。

**整包对拍**（`hope_training/whole_body_tracking/tests` 全跑，同一解释器、`-p no:randomly`
`--continue-on-collection-errors`，两棵树各跑一遍；两棵树的三个改动文件在跑之前逐字节核过 SHA-256）：

| | 基线 `birth_base_20260807`（未打补丁） | 本条 `birth_20260807` |
| --- | --- | --- |
| 结果 | `123 failed, 7501 passed, 62 skipped, 19 errors`（`35:44`） | `123 failed, **7517** passed, 62 skipped, 19 errors`（`33:23`） |

**`failed` / `skipped` / `errors` 三项逐位相同，`passed` 恰好多 `16`** —— 正好是新增的那一个测试文件，
没有一条别的动过。那 `123 failed / 19 errors` 是**基线本来就有的**：两棵树都是 detached worktree，
一批 artifact/vendor-identity 测试要求干净 checkout 与特定 commit，本条不碰它们。
另外两个受影响模块单独再跑一遍：`test_materialize_a3_dynamic_ready_contract.py` `35 passed`（与基线同数）、
`test_audit_position_hold_authority.py` `16 passed`，`0 skipped`；
A/C 同族门 `test_action_ball_211_ac_family_config_parity.py` `23 passed / 0 skipped`。
拒绝路径在**最终字节**上又实跑一次确认：消息与上面引的那段逐字相同，且**没有产出任何 artifact**（fail-closed）。

**未跑 Isaac、未占 GPU、未写任何 artifact、未放宽任何门限、未退役任何结构。**

### 十三、那两套增益各管什么、四条出路各要付什么价（2026-08-08 裁定材料，等 Franco 拍板）

> **本条与「十二」已由「十四」从零独立重算过一遍：主结论成立，三处数字已就地更正
> （电机限幅 `118.0` 不是 `118.2`；切增益后腰塌下去是 `114..144 ms` 不是 `39 ms`；
> 本条 3 (ii) 的 `1.055 / 1.517` 是对的）。分档方案见「十四」5。**

「十二」把账停在"只有一个旋钮：腰的保持增益"。本条把这个旋钮**查清、把出路摆全、每条标上价钱**，
并**就地更正**了三处还写着"出生改 frame 0 只差接地"的旧话（见「九」3、「九」6 与工具目录）。
**本条没有改任何增益、没有改任何出生姿态、没有改任何门限、没有写任何 artifact。**

#### 1. 先给结论

**推荐顺序：先跑第 (v) 条（零 plant 改动、一次 materialize 就有答案），拿到数再决定要不要动增益；
真要动增益就动第 (iv) 条（腰 `kp 50 → 150`），不要动第 (i) 条（等待期切站立增益）。**
理由一句话：**第 (i) 条不是把阶跃消掉，是把它从"位置差"换成"力矩悬崖"，而且落在动作开始的那一刻**——
实测切换瞬间腰上少 `27.451 N·m`，腰往前塌的起始角加速度 `-9.7 .. -15.4 rad/s²`，
**倒 `0.1 rad` 要 `114 .. 144 ms`（约 6--7 个控制步）**。

> **⚠ 2026-08-08 就地更正（「十四」独立复算）**：这两个数原来写的是 `-128.7 rad/s²` / `39 ms`。
> 那是把机器人当**悬空自由漂浮**算出来的（用整个质量矩阵逆的对角元 `0.214 kg·m²`，
> 等于允许骨盆和两条腿在空中反向甩）。**脚踩在地上时这个边界条件不对。**
> 脚着地的正确区间是 `1.788 .. 2.828 kg·m²`（下界＝根固定、其余关节自由；上界＝其余关节也被各自
> 的 PD 拉住），对应 `-15.35 .. -9.71 rad/s²`。**方向和结论都不变，只是塌得比原来写的慢约 3 倍。**

#### 2. 查三层：`a3_kps` 与 `a3_pd_stand_kps` 各在什么时候用

**① 机制码**（厂商 deploy 里 `a3_pd_stand_kps/kds` 一共四个真实调用点）：

| 调用点 | 什么时候 | 保持的是哪个姿态 | 谁触发 |
| --- | --- | --- | --- |
| `main.cpp:2989` `DeployMode::kPdStand` | 上电/人工起身（`s` 键、`--start pd_stand`） | 从**实测姿态**线性 blend 到 `a3_default_angles` | 人按键 |
| `main.cpp:2738` `--auto-start` 的 warmup 窗口 | 开跑前 `warmup_ticks` 个 tick | **策略算出来的 `q_des`**，但用站立增益发 | 自动，到点自己切回 `a3_kps` |
| `main.cpp:3323` teleop 兜底 | 遥操数据源断了 | 站立兜底姿态（`fallback_use_pd_stand_gains` **默认 `true`**） | 数据源丢失 |
| `pp_policy.hpp:1362` planner **STATIC-stand 闩锁** | **策略正在跑**、两拍之间、`level==0` 且已回站位 | `q_des` 在 `hold_blend_s = 0.8 s` 内 ramp 到 `nominal_q_sdk_`（= ONNX 的 `default_q`） | 策略自己，`planner_static_gain_scale` 默认 `1.0` = 官方增益逐字节原样 |

**② 实验史裁定**：`docs/DEFINITIONS.md` 的 `STAND GAIN SOURCES` 一条记着 2026-07-22 的部署侧取证——
真机上**同时存在两条站立增益路径**（人工 `PD_STAND` 的 `--stand-kp/--stand-kd`，与 planner static 的官方高增益表），
`--stand-kp` **够不到**后者；07-25 之前 runner 的分组增益块还会在 `PpPolicy` 之后再改 STATIC 命令。
`run_pingpong_end_to_end.md:1588` 有一条现场读数：**不带 `--official-stand` 时按 `s` 会屈膝站不住**——
说明真机上"站住"确实靠的是那套高增益，不是策略增益。

**③ 现役 argv**：我们的 `training_contract` 腰是 `85/50/50`，与 `a3_kps` 一致；仓库里**没有任何一处**把
`a3_pd_stand_kps` 接进仿真侧。厂商侧 `planner_static_gain_scale` 默认 `1.0`。

**所以 (a) 的答案是**：**会**。真机上策略跑着的时候确实会切到站立增益——planner 的 STATIC-stand 闩锁就是干这个的，
`--auto-start` 的 warmup 窗口更是"先用站立增益保持策略的 `q_des`、到点切策略增益"这个形状本身。
**但厂商用它保持的**始终是**自己的默认姿态**（`a3_default_angles` / ONNX `default_q`），
而且代码注释写得很直白：静态站立**不会主动平衡**（"a static stand cannot"），
所以闩锁挂了三道前置（回到站位、朝向摆正、站稳），一旦冻住一个歪斜的姿势就会几秒后倒。
**"用站立增益去保持一个动捕运动员的预备架势"，厂商从来没这么干过。**

#### 3. 五条出路，逐条标价（全部实测，解释器与收据见本条 7）

| | 做法 | 实测代价 |
| --- | --- | --- |
| **(i)** | 等待期用站立增益，揭示时切回策略增益 | **形状可复现**（真机有这条路），但**切换瞬间腰上凭空少 `27.451 N·m`**（需要 `-49.155`，策略增益最多 `-21.704`）：腰角加速度 `-9.7 .. -15.4 rad/s²`，倒 `0.1 rad` 要 `114 .. 144 ms`（「十四」更正，原写 `-128.7` / `39 ms` 是按悬空算的）。指令侧同时跳 `-0.336 rad`。**还有一条更硬的**：`action_scale = 0.25 × effort / kp`，腰 `kp 50 → 500` 会把 `waist_pitch` 的动作尺度从 `0.59` 压到 `0.059`——**切增益就是切动作语义**，除非明确把 `action_scale` 钉死不动（那就脱离厂商公式了） |
| **(ii)** | 腿取接地后的 frame 0，腰/臂取"能撑住的最近姿态" | **数学上有解，工程上没解**。最近解只要动上身：最大 `1.055 rad`（右肩俯仰）、`L2 1.517`，比现役的 `2.243 / 3.672` 小一半——**但它不可用**：拿它去跑仓库自己的支撑边投影直接失败（`support-edge projection reached its support margin but the static ground LP did not pass`），拿它直接进 hold LP 会**激活一个不是脚-地的接触**（自碰撞）。而沿"上身朝现役 ready 姿态插值"这条自然方向**永远到不了**：缺口在 `α=0.4` 反而涨到 `28.6 N·m`，`α=1.0` 也还差 `25.4`。**原因是负担来自髋，不是臂**：frame 0 的髋俯仰是 `-1.42 / -1.46 rad`（`81°/84°` 前折），整个躯干挂在腰前面，手臂只能调动其中 `±4 N·m` |
| **(iii)** | 换 `hope_Take_062_unit00_BH` 当首发（全库唯一 frame 0 撑得住的那条） | **不适合**。它 frame 0 撑得住，但**已经用掉 `89.6%` 的腰权限**（要 `-26.052`，只有 `-29.074` 可用），而且**整条 clip `70` 帧里只有 `12` 帧撑得住，第 `12` 帧就撑不住了**，最大缺口 `28.432 N·m`。它不是"更安全的 clip"，只是"起手那一帧刚好没超" |
| **(iv)** | 只把腰的 `kp` 提上去 | **`50 → 150`（3 倍）就够本条 clip**：`Take_061_unit04_BH` 从 `0/57` 变 **`57/57` 全帧撑得住**（`kp 100` 只到 `44/57`）。全库 frame 0 从 `1/73` 到 `32/73`——**再往上加没用**（`kp 200/300/500` 都是 `32/73`），因为剩下的 `41` 条卡的是**右腕站在可发指令包络之外**，那是另一种病，跟增益无关。代价：按厂商公式 `action_scale = 0.25 × effort / kp`，腰的动作尺度同步**除以 3**（`waist_pitch 0.590 → 0.197`）、`contract` 的 `kp` 不再逐位等于厂商 `a3_kps`、下游 SHA 全部重签。**参考量级**：`150` 是厂商站立那套 `500` 的 `30%` |
| **(v)** | **什么增益都不动，先跑仓库里已有、但这条谱系从没跑过的那个模式** | `materialize_a3_dynamic_ready_contract.py` 早就有 `--physical-birth-composition-mode whole_body_safe_teacher_frame0_grounded`（自陈"frame 0 若过全部门就原样返回，不安全才退到 lexicographic 搜索"），它的安全评估器**里面就含 hold LP**。现役这份出生姿态走的是 `full_seed`（`teacher_yaw_aligned_full_seed_plus_exact_teacher_reference`）——**那是一个"从站姿种子出发"的姿态，不是"离 frame 0 最近的可保持姿态"**。跑一次就知道 `2.243 rad` 这个阶跃还能压到多小。**零 plant 改动、零增益改动、一次 materialize** |

#### 4. (e) 「出生姿态 = 动作零点 = 增益标定点」这条三位一体：**成立，但要说准**

三个实测数据点，同一把尺子（策略增益、同一执行包络）：

| 姿态 | `waist_pitch` 要多少 | 用掉多少权限 |
| --- | --- | --- |
| **厂商编译进 MJCF 的站立 keyframe**（就是 `a3_default_angles`：髋 `-0.1315`、膝 `0.2515`、肩俯仰 `0.295`、肘 `0.807`、腰 `≈0`） | `-5.594 N·m` | **`26.9%`**（还有 `3.7` 倍余量） |
| **现役出生姿态**（腿 + 腰逐位 = 动作零点，手臂是动捕来的 ready） | `-18.746 N·m` | **`93.2%`** |
| **接地后的 frame 0** | `-49.155 N·m` | **`226.5%`**（撑不住） |

**所以准确的说法不是"增益刚好按默认姿态标定、卡得死死的"**——在厂商自己的默认姿态上，
策略增益的腰有 `3.7` 倍余量。**真正的规律是：离动作零点越远、质量越往腰前面挪，这份余量吃得越快。**
现役出生姿态之所以撑得住，是因为它**腿和腰逐位停在动作零点**（`AGIBOT_A3_CFG.init_state.joint_pos`，
「九」6 已记：`use_default_offset=True`、与 C++ `pp_policy` lockstep），只有手臂离开了零点；
即便如此也已经吃掉 `93.2%`。frame 0 把髋折了 `81°`，一次就超了 `2.3` 倍。

**结论写死**：这三件事**是绑在一起的**——动作零点（`default_joint_pos`）既是 `action = 0` 的落点，
也是 `action_scale = 0.25 × effort / kp` 里 `kp` 的另一半，还是唯一一个腰上留着 `3.7` 倍余量的姿态。
**动其中任何一个，另外两个必须同批重估**：改 `kp` 就改了动作尺度；换出生姿态就换了余量；
换动作零点就同时换了前两样。这正是 `EXP-V2-REWARD-FREEZE-20260726` §0.13「不许动资产默认值」护的东西。

#### 5. 顺手撞见的一件事（不在任务书内，只报不修）

`left/right_shoulder_roll` 的**动作零点是 `±0.12 rad`，而执行 `q_des` 包络的内沿是 `±0.1697`**——
**动作零点站在自己发得出去的指令范围之外，差 `0.0497 rad`。**
也就是说这两个关节上 `action = 0` 发下去会被投影一刀，落地变成 `±0.1697`，
恒定偏 `0.0497 × kp 40 ≈ 1.99 N·m`。今天不挡任何事（现役出生姿态这两个关节在 `±0.224/0.227`，在包络内），
但它跟本条是同一族问题（"出生姿态必须发得出去"），**留给 Franco 判要不要收**。

#### 6. 「十二」6 那四件事的状态：**一件都没退，理由不变**

新增一条实测支撑：**接地后的 frame 0 现在是走完整条生产路径被拒的**，
不是靠一条必要条件推断的——`solve_g1_support_edge_projection` 返回完整 `G1S`（几何 PASS、
地面动力学 PASS、腿最大改 `6.46°`、root 一个 bit 没动），紧接着**同一台 `MujocoGroundContactLPSolver`
在同一套 `hold_tau` 边界下判 `feasible = false`**；同一段代码对现役出生姿态判 `feasible = true`。
**误拦的不再拦、该拦的仍拦，在生产求解器上又验了一遍。**

#### 7. 收据

pod1 `/workspace/hope_isaac_venv/bin/python`（`python 3.10.18` / `mujoco 3.10.0` / `scipy 1.15.3` /
`numpy 1.26.4`），`CUDA_VISIBLE_DEVICES=` 纯 CPU，**未占 GPU、未跑 Isaac**；
worktree `/workspace/franco/birth_20260807`；四个诊断脚本与报告在
`/workspace/franco/birth_out_20260807/`：`adjudicate_waist_20260808.py`（增益扫描 / 切换冲击 /
take062 普查）、`interp_waist_20260808.py`（frame 0 → 现役出生姿态整条插值）、
`lp_stage2_20260808.py`、`lp_stage3_20260808.py`（生产 LP 复跑 + 厂商站立 keyframe + 上身插值）；
对应 JSON `waist_adjudication_20260808.json` / `waist_interp_20260808.json` / `lp_stage3_20260808.json`。
**全部是未授权诊断：不写 artifact、不改门限、不授权上机。**

### 十四、从零重算一遍「十二」「十三」：结论成立，三处数字要更正（2026-08-08 独立复核）

#### 1. 一句话

**「出生 = 接地后的 frame 0」确实卡住，卡的是腰能不能顶住上半身的重量，
而且卡的既不是电机、也不是接地、也不是内沿松紧 —— 是腰的刚度乘以指令还能走的行程。**
`waist_pitch` 撑住这一帧要 `-49.155 N·m`，位置指令最多发得出 `-21.704`（**差 `27.451`，只有需求的 `44.2%`**）；
把所有内沿全拆掉、指令顶死到机械限位，也只有 `-26.015`（`52.9%`，仍差 `23.140`）。
同一时刻电机限幅是 `118.0 N·m`，**只用掉 `41.66%`**。

#### 2. 复核怎么做的：没有沿用上一轮任何一行代码

上一轮（`81379ea2`）的数是用仓库自己的 `materialize` 支路 + 新工具 `audit_position_hold_authority.py` 算的。
这一轮**另写了一份脚本**，只读三样原件 —— 这条谱系自己绑定的 MJCF（`a3_pingpong.xml`）、
动捕库 npz、以及现役 `dynamic_ready` artifact 里那份**从真实 Isaac 启动读回来记下的活值台账** ——
不 import 仓库任何模块。**九项逐位重现**：

| 量 | 「十二」写的 | 本轮独立算的 |
| --- | --- | --- |
| `waist_pitch` 需求 | `-49.15464109801158` | `-49.15464109801158`（**逐位相同**） |
| `waist_pitch` 可发 / 缺口 | `-21.704` / `27.451` | `-21.70368` / `27.45096` |
| 顶到机械限位 | `-26.014` / `52.9%` | `-26.01464` / `52.93%` |
| `waist_roll` 需求 / 可发 / 缺口 | `-14.243` / `-10.020` / `4.223` | `-14.24287` / `-10.01957` / `4.22329` |
| `take061` 全 57 帧 | `0/57` | `0/57`（缺口 `14.879 .. 27.451`） |
| 全库 73 条 frame 0 | `72/73` 撑不住，最大缺口 `32.233` | `72/73`，最大缺口 `32.2325`，唯一撑得住的仍是 `hope_Take_062_unit00_BH` |
| `42/73` 有关节站在包络外 | `42/73` | `42/73` |
| 现役出生姿态 | 缺口 `0.0` | 缺口 `0.0`（`waist_pitch` 要 `-18.7463`，可用 `-20.1236`，**用掉 `93.16%`**） |
| 厂商站立 keyframe | `-5.594 N·m` / `26.9%` | `-5.5936` / `26.88%` |

**四条"这不是别的东西"的证伪，也都重跑了：**

1. **不是接地。** 把 12 个腿关节在各自限位内随机重掷 64 次，腰/臂/头这 19 行的需求力矩最大只飘
   `1.4e-14 N·m`。理由是运动学的：直接打脚（`ankle_roll` 两个 body）的雅可比，
   **这 19 列的最大元素恰好是 `0.000000e+00`**，而另外 12 列最小也有 `0.624`。
   地面反力在这 19 个关节上使不上劲，所以"要多大力矩"没有解算自由度。
2. **不是别的力，就是重力。** 重力乘 `0 / 0.5 / 1 / 2`，需求力矩得到 `0` / 恰好一半 / 全量 / 恰好两倍。
3. **不是电机。** 把力矩边界从"`kp ×` 行程"换成"电机限幅"，再解一次"离 frame 0 最近的可保持姿态"：
   **答案是 frame 0 自己，一个关节都不用动（`dq = 0.0`）。** 也就是说这台机器人的**电机**完全撑得住这个姿态，
   撑不住的是**位置指令这条通道**。
4. **不是关节顺序搞错了。** npz 的 31 列不是仓库那份 contract 顺序，是 Isaac 的关节顺序。
   按 artifact 自陈的顺序装进 MuJoCo，全部 32 个 body 的正运动学位置误差 `9.5e-8 m`；
   按 MJCF 声明顺序装，误差 `0.712 m`。**这一步错了后面全是废数**，所以先钉死再算。

#### 3. 三处要更正的数字（已就地改，不追加矛盾段落）

**① `waist_pitch` 的电机限幅是 `118.0 N·m`，不是 `118.2`。**
`118.2` 是**踝俯仰**的。三处原件一致：URDF `effort="118"`、MJCF `actuatorfrcrange="-118 118"`、
Isaac `effort_limit_sim=118.0`（`agibot_a3.py:301`；同文件 `:288` 的 `118.2` 属于 `.*_ankle_pitch_joint`）。
利用率因此是 `41.66%` 不是 `41.6%`。**结论不受影响**，已改「八」「十二」三处。

**② 「切增益瞬间腰塌下去」那个速度写快了约 3 倍。**
原写 `-128.7 rad/s²` / `倒 0.1 rad 只要 39 ms`。本轮把三种边界条件全算了一遍：

| 边界条件 | 有效惯量 | 起始角加速度 | 倒 `0.1 rad` 用时 |
| --- | --- | --- | --- |
| **悬空自由漂浮**（整个质量矩阵求逆的对角元） | `0.214 kg·m²` | `-128.15 rad/s²` | `39.5 ms` |
| **根固定、其余关节自由**（脚焊在地上） | `1.788 kg·m²` | `-15.35 rad/s²` | `114.1 ms` |
| **根固定、其余关节也被各自 PD 拉住** | `2.828 kg·m²` | `-9.71 rad/s²` | `143.5 ms` |

原来那个数取的是第一行 —— **那等于允许骨盆和两条腿在空中反向甩**，而这台机器人是双脚踩在地上的。
脚着地的正确区间是后两行：`-9.7 .. -15.4 rad/s²`、`114 .. 144 ms`（约 `6--7` 个控制步）。
**方向和裁定都不变**（"切增益只是把这一刻从 `t=0` 推到揭示那一刻，一分钱没省"仍然成立），
但**别再拿"2 个控制步就倒"去说服人**，那个数是错的。

**③ 「十三」3 (ii) 那个 `1.055 rad / L2 1.517` 是对的，不要改成 `1.786 / 2.330`。**
本轮用 SLSQP（带梯度、显式不等式约束）重解"腿钉在接地后 frame 0、19 个非腿关节取离 frame 0 最近的
可保持姿态"，得到 **`max|dq| = 1.05545 rad`（右肩俯仰）、`L2 = 1.51702`** —— 与已提交的数逐位吻合。
另一条并行复核给出的 `1.7859 / 2.3296` 是**无导数方法（Powell）落进的一个更差的局部解**，不是另一个口径。
差别不小：真解把揭示阶跃从现役的 `2.2434 / 3.6719` 砍掉 `53% / 59%`，那个坏解只砍 `20% / 37%`。
**「十三」3 (ii) 的定价（"数学上有解、工程上没解"，卡在支撑边投影与自碰撞）不变。**

#### 4. Franco 那句"策略增益是智元调出来的、应该问题不大"：数据说一半对，另一半**不够，而且不是差一点**

同一把尺子（策略增益 `85/50/50`、同一条执行包络），三个姿态：

| 姿态 | `waist_pitch` 要多少 | 用掉多少权限 | 判 |
| --- | --- | --- | --- |
| 厂商自己编译进 MJCF 的站立 keyframe | `-5.594 N·m` | `26.88%` | 过，还有 `3.7` 倍余量 |
| 我们现役的出生姿态 | `-18.746 N·m` | `93.16%` | 过，但只剩 `1.377 N·m` |
| **接地后的 frame 0** | `-49.155 N·m` | **`226.5%`** | **不过** |

**支持他的那一半**：这套增益在它被调出来的那件事上确实没问题 —— 厂商默认站姿腰角约等于零、
重力力矩几乎为零，`3.7` 倍余量；我们现役出生姿态也过得去。**增益本身没调错，也不必去改它。**

**不支持他的那一半**：到了动捕运动员那个预备架势上，**不够，而且不是差一点，是差 `2.26` 倍**。
这不是内沿留窄了 —— 把 `5%` 投影内沿和 `2%` 硬内沿**全部拆掉**、指令顶死机械限位，仍然差 `23.140 N·m`。
要在现行包络内刚好撑住，`waist_pitch` 的 `kp` 得从 `50` 提到 `113.24`（顶到机械限位也要 `94.47`）。
而且没有 DR 抖动可指望：现役四格跑的是 `stable_ready_plant=true`，
`train.py` 的 DR-L0 收尾直接把 `events.randomize_pd_gains` 置 `None`，**跑的是精确标称值**。

**准确的说法是：不是"厂商调错了"，是我们在拿这套增益问一个它没被调来回答的问题。**
腰关节本身几乎是直的（`+1.81°`），那 `49 N·m` 不是腰弯出来的，是**整个躯干在骨盆处前倾约 `30°`**、
把 `29` kg 的上半身质心顶到腰轴前面约 `0.18 m` 造成的 —— 而那个前倾正是运动员预备架势本身。
所以"把腰摆直一点"没用，腰已经是直的。

#### 5. 分档方案（依据都在上面，代价按从小到大）

**A 档 —— 现在就能做，不动 plant、不重铸任何产物：**

- **A1（推荐）**：**把"frame 0 在标称增益下静态撑得住"变成首发 clip 的一条离线普查判据**，
  写进题库准入。今天的答案是 `1/73`；若按厂商同底盘 PD 随机化的 `0.8×` 下限算则是 `0/73`
  （唯一那条 `hope_Take_062_unit00_BH` 只有 `3.022 N·m` 余量，`kp` 掉到 `44.80` 就不成立 = 标称的 `0.896×`）。
  代价：一个扫描脚本 + 一条门，零 plant 改动。收益：以后不用再为一个力学上无解的目标反复重铸。
- **A2**：**盯住现役出生姿态那 `1.377 N·m`**（`93.16%` 已用）。它今天是这套 plant 上**唯一**被证明
  能被位置指令撑住的姿态，但余量很薄，任何一次动作零点或手臂 ready 角的改动都要重算这一格。
- **A3**：`left/right_shoulder_roll` 的动作零点 `±0.12 rad` 站在自己执行包络内沿 `±0.1697` 之外
  （「十三」5 已记）—— 今天不挡事，但属于同一族"出生姿态必须发得出去"，一并纳入 A1 那条普查。

**B 档 —— 需要重铸 artifact / 重签下游：**

- **B1**：出生姿态改取"frame 0 在可保持子空间上的投影"。**有解**（`max 1.055 rad` / `L2 1.517`，
  比现役的 `2.243 / 3.672` 小一半多），但「十三」3 (ii) 实测它**过不了仓库自己的支撑边投影**、
  且会激活一个自碰撞接触。代价：一次 materialize + 整条 `211/C211` 谱系重铸重签，
  换来的姿态在语义上已经不太像那条 clip 的引拍。
- **B2**：换 `hope_Take_062_unit00_BH` 当首发。**不推荐**：它 frame 0 撑得住但只剩 `3.022 N·m`
  （`89.6%` 已用），整条 `70` 帧里只有 `12` 帧撑得住，而且它相对现役出生姿态的差是
  `max 2.4884 rad`，**比现在还大**。换 clip 并不消阶跃。

**C 档 —— 只能 Franco 拍板：**

- **C1**：**把腰的 `kp` 提上去**（包络内 `≥113.24`，机械限位处 `≥94.47`；「十三」3 (iv) 实测 `150` 能让
  take061 全 `57` 帧撑住、全库 frame 0 从 `1/73` 到 `32/73`，再往上加没用 —— 本轮复核 `×2→28`、
  `×3→32`、`×4→32`、`×10→32`，逐位吻合）。**代价是硬的**：
  `kp × action_scale = 0.25 × effort` 是这套厂商解码器的恒等式（本轮 31 个关节逐个核过，相对误差 `6.1e-8`），
  所以**抬 `kp` 会等比缩小 `action_scale`，动作 ABI 变了**，整条 reward 经济和下游 SHA 全部要重估，
  而且 `contract` 的 `kp` 不再逐位等于厂商 `a3_kps`。
- **C2**：**等待期切厂商站立增益（腰 `400/500/500`）、揭示时切回**。**能力是真的**
  （「十三」2 查到真机上 planner STATIC 闩锁、`--auto-start` warmup、`--auto-leg-hold` 都在逐 tick 这么切，
  且全部是无斜坡的硬阶跃），**但它买不到我们要的东西**：厂商每一次切回时被保持的都是它**自己的默认站姿**，
  那个姿态策略增益本来就撑得住；我们会在切回的那一 tick 交出一个策略增益撑不住的姿态，
  腰上瞬间少 `27.451 N·m`。**能力是真的，用法是自欺。**
- **C3**：**加重力前馈力矩**。这是唯一"不切增益、不改姿态、不换 clip"还能成立的路 ——
  本轮第 2 条第 3 点已证：在电机限幅这条边界下，frame 0 **一个关节都不用动**就撑得住，
  `49.155` 对限幅 `118.0` 只用 `41.66%`。deploy 侧 `RobotCommand` 的 `tau_ff` 通道在硬件上是通的
  （`include/a3_io/publish_helpers.hpp:79` 把它发给 SDK），现有代码三处一律填零。
  代价最大：训练侧要产出 `tau_ff`、obs/action 契约变更、部署合同加一路，
  而且仿真的执行器模型会比真机多一项（除非同批把真机那一路也接上）。
- **C4**：**承认出生本来就不该要求"静态可保持"**，让策略动态接住。代价：那道 hold 门给的保证没了，
  得换一条别的验收；好处是零改动。

**本轮不替 Franco 选任何一条 C 档，也没有动任何增益、姿态或产物。**

#### 6. 收据

pod1 `/workspace/hope_isaac_venv/bin/python`（`python 3.10.18` / `mujoco 3.10.0` / `numpy 1.26.4` /
`scipy`），`CUDA_VISIBLE_DEVICES=` **纯 CPU，未占任何 GPU，没有 pytest 所以没有 skip 数**。
盘 `a3_pingpong.xml`（`A3T2.5_pingpong_0519`，总质量 `58.2573 kg`，`nq=38` / `nv=37`，正是现役
`dynamic_ready` artifact 的 `sources.mujoco_model`），库 `chingmu73_measured_v4_20260803`（73 条）。
脚本与 JSON 在 `/workspace/franco_adjudicate_20260808/`：`adj.py` / `adj.json`（需求-可达全表、
脚部雅可比零列、接地不变性、重力证伪、有效惯量三种边界、现役出生姿态）、
`adj2.py` / `adj2.json`（57 帧扫描、全库 73 条扫描、腰 `kp` 扫描、三种边界下的最近可保持姿态投影、
厂商恒等式核对）、`adj3.py`（厂商站立 keyframe 与动作零点）。**未授权诊断：不写 artifact、不改门限、不改增益。**

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
| ~~`C211_ACTION_RATE_CLAMP`~~ → `C211_ACTION_RATE_POST_DT_WEIGHT` | `train.py` `_REWARD_PACK_V2_DIRECT` 的 `("action_rate_l2", -0.1)` | **2026-08-08 就地更正**：这条债原本写的真源是 `action_rate_clamped` 的 `value_clamp: 9.0`，但封顶版当天已退役（§5.6.25），**上游 `action_rate_l2` 根本没有 `value_clamp` 这个键** —— 再挂一条 clamp 的镜像债等于在镜像一个不存在的东西。常量随之更名，真源改指包里那个权重。缺的仍是那张镜像表本身 |
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
> **2026-08-07 结案：已接线，见 §5.6.22。** 下面这段保留为接线前的现状记录。
> 同轮还纠正了一条会让它被误杀的前提：这块记录跟 `bridge_ramp_command_steps` 是两样东西，
> 「出生改成 frame 0 之后阶跃归零」退役的是那条 ramp，不是这块账（对照表在 §5.6.22 一）。
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
~~**为什么现在没做**：它会改变 gate 的拒绝面，而四格 `scale4096` 正在发；本轮不动正在跑的门。~~
> **2026-08-07 就地更正：这条理由不成立，本条已提到"发车前必须做"（§5.6.17 二.B1）。**
> 四格 `scale4096` **没有在发**：`eccb30cd` 改了 `hope_commands.py`，`recipe` 阶段被
> solver profile pin 拦下（§5.6.15 末尾自己写了"`scale4096` 未跑"）；2026-08-07 复核 pod1，
> 唯一的 `scripts/train.py` 是别人的 `task=HitterPingPongPhase114`，GPU1/GPU2 各 `8 MiB` 空闲。
> 本条原文自己那节的 pod 复核也写着"无 `train.py` / `launch_action_ball` 进程"，与这句理由自相矛盾。
> 另补一条支持它现在就该做的事实：`solver_profile_sha256` 只哈希 `hope_commands.py` /
> `continuous_questions.py` / `racket_contact_geometry.py` / `stroke_adapt_torch.py` /
> `virtual_ball.py`（counter-rally 开时再加两个），
> **`action_ball_4096x5_prelong_gate.py` 不在里面**——接这条线是零 lineage 代价的改动。

**这一条不能只当"清理遗留"读**——`_validate_reveal_bridge` 存在这件事本身，容易让人误以为
"gate 已经严格消费 v3 了"。`4c9cf280` 已在该函数 docstring 里写明"**尚未接线**"，
读代码的人不会再被误导；但接线本身仍然欠着。

**接线方案（2026-08-06 核实，可照着做；~~等 `scale4096` 落地后执行~~ **→ 2026-08-07 更正：
现在就执行，`scale4096` 没有在跑，理由见上面那个更正块与 §5.6.17 矛盾 2**）。**
先纠正一处措辞：**不需要"升版"**。`SEMANTIC_SCHEMA_VERSION`（`:63`）就是
`_SEMANTICS.PRELONG_SEMANTICS_SCHEMA_VERSION`，而生产方那个常量**已经是 `3`**
（`action_ball_prelong_semantics.py:27`），`_ordered_updates`（`:178-195`）逐行要求
`schema_version == 3`。所以现役 gate 收的**本来就是 v3 行**，缺的只是"把 v3 多出来的那块字段
读一读"。这是**纯增量的字段消费**，没有版本号要动，也没有第二个 v2 消费方要同批升。

1. **改哪几行。** 在 `validate_semantic_updates`（`:1216`）的 `for index, row in enumerate(ordered)`
   循环里，紧挨现有 `row.get("reward_groups")` 那一组取值之后，加一次调用：
   把 `_validate_reveal_bridge(row.get("reveal_to_playback_bridge"), profile=row_profile,
   update=index, previous_lifetime=<上一轮的 lifetime>, expected_authority=<第一轮返回的 authority>)`
   的三个返回值分别接住。函数签名已经为跨 update 状态设计好了：`previous_lifetime`
   第一轮传 `None`、之后传上一轮返回的第二个值（它内部按 WAIT 档做单调性比较），
   `expected_authority` 第一轮传 `None`、之后传第一轮返回的 authority
   （它内部做 `authority != expected_authority` 的整块相等比较，即"五个 update 的权威身份不许漂"）。
   循环外把逐档 lifetime 汇总，塞进 `:1465` 那个 `return {...}` 的 `aggregate` 里
   （建议键名 `reveal_to_playback_bridge`），这样收据能**自陈**这一步跑过了——
   否则又是一个"只出结论、没人读"的位。
2. **拒绝面会怎么变（这是本条唯一有风险的地方）。** 新增的硬拒收有四类：
   (a) `status != "active_fail_closed"` —— 桥没配起来时生产方返回
   `{"status": "not_configured"}`（`:3065`）。
   > **2026-08-07 就地更正：原文这里要求"接线前先拿一格已落盘的 `scale4096` 收据
   > 实测一遍 `status`"。这一步既做不到也不必做——没有任何一格跑完过，没有收据可读。
   > 静态读三行就能给出更强的答案，而且今天就能读：**
   > `_prelong_bridge_authority`（`action_ball_prelong_semantics.py:1369`）**只有在
   > WAIT schedule 缺席且 `required=False` 时才返回 `None`**；`required` 就是
   > `require_bridge_telemetry`，默认 `True`（`:1673`），而**全仓只有三处传 `False`，
   > 全在 `test_action_ball_prelong_semantics.py` 里**，零个生产调用方。
   > 所以现役路径上 `_bridge_enabled` 恒真，桥要么是 `:3285` 的 `active_fail_closed`，
   > 要么在 ledger 构造那一刻就 `PrelongSemanticLedgerError` 直接死掉——
   > **`not_configured` 在生产上根本产不出来**。
   > 附带一条：真产出了 `not_configured`，它只有 `status` / `reason` 两个键，
   > 而 `_exact_keys` 要六个，所以它会先在**键集合**上被拒、拒绝信息不会提 `status`。
   > 接线前该做的那一步因此变成：**确认没有人给 `require_bridge_telemetry` 传 `False`**
   > （一次 grep），不是等一份不存在的收据。
   (b) 七项权威 SHA / `wait_cohort_ticks` / `policy_dt_s == 0.02` 的逐项形状与相等；
   (c) 每个 WAIT 档 `reveal = start + terminal + censored`，且 `reveal/start/terminal`
   跨 update 单调不减，且整窗 `reveal_delta > 0`；
   (d) `timing_at_reveal.reveal_count` 必须等于各档 reveal 之和。
   `BRIDGE_WAIT_COHORTS = tuple(range(5, 26))`（`:106`）= 21 档，`_exact_keys` 是**精确键集**
   不是子集，所以生产方**多写或少写一个字段都会拒**——这正是要的，但也意味着
   producer/consumer 必须同一版本，接线的 commit 里要把两边的 SHA 一起钉。
3. **需要哪些变异测试**（每条都必须能杀掉，且"粗一个档次的检查"要放过）：
   - `status` 改 `not_configured`（其余字段全对）→ 必须红。粗版：只检查 `status` 存在。
   - 任一权威 SHA 改一位十六进制 → 必须红。粗版：只检查长度 64。
   - 某一档 WAIT 把 `censored_count` 减一、同时把 `reveal_count` 也减一
     —— **该档自己仍然守恒**，但 `timing_at_reveal.reveal_count` 与各档之和对不上 → 必须红。
     这条专门打"只逐档看守恒、不看跨块总和"的粗版。
   - 第 3 个 update 把 `authority` 里某项换掉（前两个 update 一致）→ 必须红。
     粗版：只校验第一个 update 的 authority。
   - 某档 `start_count` 比上一 update 变小（其余守恒）→ 必须红。粗版：只看单 update 快照。
4. **验收**：与本节 (C) 两条同样的做法——先跑 `test_action_ball_4096x5_prelong_gate.py`
   取基线失败集，接线后逐条对拍；再把接线本身注释掉，确认上面五条变异**全部回绿**
   （证明测试确实在测这道门，而不是在测别的东西）。

**(B) `promotion_blocked` 这个结论位全仓没有任何消费者，也没有任何测试。**
（**2026-08-06 已接线，见 §5.6.16**；下面这段保留为接线前的现状记录。
唯一被后来推翻的是最后那句"需要先定清楚哪一步算晋级"——不需要，见 §5.6.16 (6)。）
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

**(C) 另外几个零调用点的门，逐个给判决。**
> **2026-08-07 就地更正：下面两条判"该接线"的门，已于 `8a6554c7` 接线并各配变异测试。**
> `launch_action_ball_curriculum.py:4378` 调 `_require_fresh_order_sentinel`、
> `launch_n1_vendor_baseline_diagnostic.py:2261` 调 `_valid_table_guard_attribution_summary`。
> 同轮把零调用点普查重跑了一遍，**代码引用与文档提及分开计数**（上一遍混在一起，
> 而本文档正好按名字讨论这些函数，反而抬高了计数、盖住了要找的东西），结论仍是五个，没有第六个。
> `_validate_reveal_bridge`（(A) 那条）**仍然是零调用点**，是本文档目前唯一真开着的那个口。
> （**2026-08-07 就地更正：这句已过期，该函数当天接线，见 §5.6.22。**）

全仓扫了 `scripts/` 与 `mujoco_native/` 的
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
  **2026-08-06 已接线并执行。** 三层复核把"有没有别人管着"问死了：
  (i) **机制码**——`FRESH_ORDER_SOURCE` 不在 `RUNTIME_CODE_SOURCES`（`:158-172`）里，
  所以那份 blob 连 SHA 都没人钉；两条车道确实有耦合（launcher 收
  `fitted_ball_gate_receipt` 等四个 fitted-ball 输入），但走的是
  `_verify_external_pin`，**只核 path+sha256、从不解析收据内容**，顺序永远读不到。
  (ii) **实验史/文档**——`docs/operations/run_action_ball_curriculum_no_clobber.md:71`
  白纸黑字写着 launcher "会交叉核对 …… committed `FRESH_N5_ORDER` ……"，
  **这句话在接线之前是假的**；这正是"假护栏"最贵的形态：文档替一个没跑的检查背书。
  (iii) **现役 argv**——N5 正式发射本身 fail-closed，所以这是潜伏漂移，不是在跑的错。
  接法：在 `runtime_code_sha256[HOPE_COMMANDS_SOURCE]` 之后写
  `runtime_code_sha256[FRESH_ORDER_SOURCE] = _require_fresh_order_sentinel(...)`——
  一行同时做两件事：**跑检查**，并把该 blob 的 SHA 落进 launch claim，
  让收据**自陈**这一步跑过了。变异测试见下方"变异证据"。
- `launch_n1_vendor_baseline_diagnostic.py:1711` `_valid_table_guard_attribution_summary`：
  **判定接线，不删**，但要连它的生产方一起看。它要求 forensic 摘要三路自洽
  （`first_hit_total_count == terminal_count == table_count`、`category_counts` 与
  `phase_counts` 各自**恰好划分**同一个 total、`nonfinite == 0`），
  正是 §5.6.5 那次"撞桌到底算在谁头上"最需要的那种守恒账。生产方在
  `hope_commands.py:23455 _consume_table_guard_attribution_counts` /
  `:23514 _validate_table_guard_attribution_conservation`，**已经存在且有测试**
  （`test_reward_flags_mdp.py:365-409`）。差的就是 launcher 侧这一步接线。
  **2026-08-06 已接线并执行，且不需要碰 `hope_commands.py`。** 上一轮把这条挂在
  "生产方在 `hope_commands.py`、本轮不碰"上，其实**挂错了地方**：launcher 读的不是
  runtime 那份扁平计数器，而是 `materialize_n1_vendor_probe_gate_receipt.py:1150-1160`
  归并出来的那块摘要，**字段名与本函数期待的完全一致**（`enabled` /
  `first_hit_total_count` / `terminal_count` / `category_counts` / `phase_counts` /
  `sparse_cell_total_count` / `conserves`）。所以接线只动 launcher 一个文件。
  三层复核：
  (i) **机制码**——runtime 那道
  `_validate_table_guard_attribution_conservation` 确实 fail-closed，但它
  `if not attribution: return` 早退，且比的是**活张量**；materializer 算出
  `conserves` 之后**主动降级成 `telemetry_only`**（只对 `nonfinite != 0` 抛
  `ReceiptRefused`）；launcher 原有那段（`:2185` 附近）只查 `telemetry_only is True`
  与 `category_counts.nonfinite == 0`。**"收据里的账对不上"这件事，三处没有一处会拒。**
  (ii) **实验史**——`test_n1_vendor_probe_gate_consumer.py` 的 pass-receipt fixture
  就是活证据：每个 update `termination_reason_robot_hit_table_count = 1000`，
  而 first-hit 账本全 `0`，`conserves` 是 `False`，
  **`_validate_vendor_probe_gate_receipt` 照样返回 `vendor_n1_long_launch: True`**。
  这个洞不是理论上的，它已经被测试固化下来了。
  (iii) **现役 argv**——launcher 自己在 probe 阶段**强制追加**
  `+task.table_contact_attribution_diagnostic=true` 且要求 exact-once、非 probe 阶段
  必须缺席（`:1224-1231`）。所以 probe 收据里"有撞桌终止、账本却记 0"**只能是缺陷**，
  不可能是"仪器没开"——`enabled/{"enabled": False}` 那条分支与这条 argv 规则同构。
  接法：放在 `aggregate == recomputed_behavior` **之后**，这样传进去的 table 终止数是
  **从逐 update 原始行重算出来的**，不是从待检摘要里读的；`categories`/`phases`
  取 `gate_module._TABLE_ATTRIBUTION_*` 的**活值**，不手抄。
  fixture 同批改成真守恒（1000 首击 = 1000 撞桌终止，落在
  `strike` / `proxy_exact_overlap` 一格），"质量很差但账要平"这层语义保留。

**(C-1) 变异证据：证明这两道门真会开火，而且"粗一个档次"的检查会漏。**
只证明"加完不报错"等于没证明。四个变异**改的都是被测源码本身**，
每个都必须让新测试转红（pod1，`/usr/bin/python3` + `pytest 9.1.1`）：

| 变异 | 改法 | 结果 |
| --- | --- | --- |
| M1 门没接 | 删掉 `runtime_code_sha256[FRESH_ORDER_SOURCE] = _require_fresh_order_sentinel(...)` 这一句 | 转红 ✓ |
| M2 门太粗 | 接着，但把 `tuple(raw) != ACTION_ORDER` 换成 `set(raw) != set(ACTION_ORDER)` | 转红 ✓ |
| M3 门没接 | 把 `_valid_table_guard_attribution_summary(...)` 那次调用短路掉 | 转红 ✓ |
| M4 门太粗 | 接着，但 `table_count` 改成读摘要自己的 `first_hit_total_count`（自证循环） | 转红 ✓ |

M2/M4 是关键：两条测试的变异**故意造成"低一档的检查看不出来"**——
M2 的变异只把五个动作里的两个**换位**（集合相同、长度相同，只有位置不同），
M4 让摘要**内部完全自洽**、`conserves` 依旧写着 `True`，只是整体比真实撞桌终止少 5 笔。
控制组（未变异）与四次变异后恢复的对照都跑过，均为绿。

**(C-2) 独立重扫：本轮没有找到第六个。** 用 AST + 全仓 token 频次重扫
（`1009` 个 `.py`、`19507` 个模块级 `def`、连嵌套共 `24698` 个 `def`），
把**代码引用**（只数 `.py`）与**文档提及**（`.md`/`.json`/`.yaml`…）分开数——
上一版把两者混在一起数，会因为本文反复讨论这些函数名而把它们的计数抬到 `>1`、
**恰好把要找的东西藏起来**。判据：某个名字在全仓 `.py` 里只出现 `1` 次 = 除自己的 `def`
外无人提及。这个数法对动态引用是安全的：`getattr(mod, "_validate_x")`、
`importlib` 共享库的 `_FRAME0._validate_x` / `_L._validate_x`、
`monkeypatch.setattr(M, "_validate_x", ...)` 里都含有那个字面标识符，计数都会 `>1`
（`launch_action_ball_a211_four_arm_diagnostic.py` 同时被 C211 launcher 与 materializer
当共享库用，单文件 AST 会假阳性，token 法不会）。
把名字形状放宽到 `_validate_*` / `_require_*` / `_valid_*` / `_check_*` / `_assert_*` /
`_verify_*` / `_reverify_*` / `_ensure_*` / `_reject_*` / `_forbid_*` / `_guard_*` /
`_refuse_*` / `_enforce_*` / `_must_*` 之后，结果**仍然是这 5 个**，与本节原先登记的一致：
(A) 一条、本条两条、canonical 车道两条。canonical 车道那两条维持"只登记不判决"。

> **2026-08-07 就地更正：原文这里写着"没有第六个"，这句是错的，有第六个。**
> 上面那张名字形状清单**每一条都带前导下划线**，于是漏掉了公开方法。
> 2026-08-07 用同样的方法但把 `assert_*` / `validate_*` / `verify_*` / `require_*` /
> `check_*` / `ensure_*` / `enforce_*` 这些**不带下划线**的形状也算进去、
> 并且把 `def` 的收集范围从模块级放宽到含嵌套（即类里的方法也算），
> 扫出第六个：`ActionBirthBroker.assert_known_generation`
> （`mdp/action_ball_runtime.py:6390`，全仓 `.py` 命中 `1` 次、文档命中 `0` 次）。
> 判决与证据见 §5.6.18 二.2 —— **该判决已于 2026-08-07 改判：删除，不是接线。见 §9.2.13。**
> 那次扫描本身**没有落盘**；方法学现在固定在
> `scripts/audit_zero_call_site_gates.py`（自带 `--self-test`，证明老做法会漏公开方法）
> 和 `tests/test_audit_zero_call_site_gates.py`。

**(C-3) 上面那四条变异证据，只在 `/usr/bin/python3` 下有效——别拿正式 venv 复现。**
`test_launch_action_ball_curriculum.py` 全模块 `59` 条，在 `/workspace/hope_isaac_venv`
下 **`53` 条被 skip**（`could not import 'cryptography'`），其中就包括 M1/M2 要杀的那条。
也就是说：拿 venv 跑 M1（把门整句删掉），pytest 会报 `1 skipped`、退出码 `0`，
**看上去是绿的**。2026-08-07 独立复跑时先踩了这个坑，详见 §5.6.18 二.1。

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
回答的（卡点在腿，不在桥），而 `34f8cf25` 的 ramp 实测只买到 `1` tick。**这两件事没有在 §5.6.2d
就地写清楚，下一个人会照着 §5.6.2d 去设计一组已经被证伪的对照格。**
（**2026-08-07 补**：§5.6.7「十」把"卡在腿"这句话的账重算了一遍 ——
卡的不是这条动捕 clip，是**导出时没做接地收尾解算**；一次 `2.44°` 的接地解算之后，
仓库自己那台最严的静态门对同一条 clip 判 `50/57 FEASIBLE`、frame 0 `FEASIBLE`。
所以 §5.6.2d 这段既不能读成"桥的问题"，也不能读成"clip 的问题"。）

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
   （**2026-08-07 收口：四处已全部就地更正**——§6.1 由本节当轮改，§5.3 / §8.2 / §12
   由 §5.6.17 那轮改；同轮还发现 §9.2.9 在 08-06 又新写了一次 `-6`，也已改。
   §5.3 那处的更正**改变了结论**：按活值重算，"合法上台后同回合摔倒"的最低净值是 `+8.2` 而不是 `+2.4`，
   套利口子比原文大 `3.4` 倍。下面保留原始取证。）
   四格合同 `action_ball_211_four_grid_contract.py:347` 是 `-10.0`（post-dt `-0.2`），
   pod1 收据里的 argv 也是 `task.rewards.death_penalty_weight=-10.0`。
   但本文 §5.3、§6.1、§8.2、§12 四处当时仍写
   post-dt `-6` / weight `-300`（行号以当时的版本为准，此后已多次移位）。**这不是措辞问题**：尽调 §22 的闸 2 把推撞、reset 噪声、
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
| `start_pose_ramp` | 写的时候：契约 + 校验器 + 运行时 + 遥测 + 两份 DR-L1 profile + 候选 config 全齐，**但没有任何 launcher 能选中 `DRL1Learnability`**。**2026-08-08 起入口已接**（`--dr-level dr_l1`，§12 `DR-LEVEL-LAUNCH-ENTRANCE`）；断点前移到"DR-L1 还没有自己的 lineage 工件" |
| DR-L1 的五条 plant 轴（friction / joint offset / CoM / link mass / Kp-Kd） | 同上：入口已接，断点是 lineage/recipe 还没 materialize |
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
| P0 | ~~给 `DRL1Learnability` 两份 profile 接一个 launcher（它今天没有入口）~~ **2026-08-08 DONE**:两个 launcher 都有 `--dr-level {dr_l0,dr_l1}`,DR-L0 解析结果逐字节不变。**剩下的一步**是给 DR-L1 跑一次 materialize(见 §12 `DR-LEVEL-LAUNCH-ENTRANCE` 的清单) | DR-L1 | — | — | 无新科学,纯接线 |
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

#### 5.6.16 §5.6.13 (B) 落地：`promotion_blocked` 接上消费方（2026-08-06）

**人话一句：** §5.6.13 (B) 点名的那个"没人读的结论位"这轮接线了。
它现在有三个真消费方，加上一条只在出事时才说话的摘要行；
并且**收据自己说谎会被拒收**——这是本轮唯一新增的硬拒绝面。

**(1) 那一位到底是什么、谁产生、本该谁读。**

| | 内容 |
| --- | --- |
| 生产方 | `mujoco_native/vec_env.py:1846` `DiagnosticEventLedger.snapshot()` 里的 `promotion_blocking_evidence.promotion_blocked`（值 = `joint_actual_forbidden_observed_ticks > 0`） |
| 流到哪 | 每步进 `extras["diagnostic_event_ledgers"][env]`；落盘进 rollout 收据的 `final_event_ledgers` / `event_ledger_transcript` |
| 接线前谁读 | **没有人。** 全仓 `grep promotion_blocked` 只命中生产点本身与它上面那行注释；`checkpoint.py` 零命中；`launch_mujoco_action_ball_211_diagnostic.py` 只从 ledger 里取 `plant_counters`；测试里 `promotion` 出现 0 次 |
| 本该谁读 | `joint_actual_forbidden` 改软时（§5.6.2 第 10 条）承诺的后半句"不终止但卡晋级"的**执行者**：跑 update 的 trainer、被晋级的 checkpoint、决定发不发的 launcher |

**裁决：接线，不删。** 删不掉的理由是 §5.6.2 第 10 条那次返工的原始动机没有别的机制在管——
`joint_actual_forbidden` 的谓词一个字没改，改的只是它的后果；
把"记录"留下、"阻断"扔掉，正好是那条教训点名的失败形态。

**(2) 接上的三个消费方（记录+阻断同批）。**

1. **`mujoco_native/trainer.py`：真拒绝面。** 新增
   `promotion_blocking_samples_from_step()` / `promotion_blocking_evidence_receipt()`，
   在 `run_update` 的 rollout 循环里**逐步逐 env**读（不是只读 done 行——这个 fault
   按定义不终止 episode，只看 done 行正好漏掉它存在的理由）。
   它同时读**原始计数和结论位并对账**：`promotion_blocked != (observed_ticks > 0)`
   直接 `DiagnosticPPOContractError`。汇总结果进 update 收据的
   `promotion_blocking_evidence`（`promotion_blocked` / `reasons` / `blocked_env_indices` /
   `blocked_sample_count` / `checked_sample_count` / `first_blocked_sample`），
   并折进 `rollout_sha256`。`checked_sample_count` 必须为正——
   "一个样本都没看过"算出来的"没问题"不是结论。
2. **`mujoco_native/checkpoint.py`：结论跟着被晋级的东西走。**
   save/load 收据新增 `promotion_blocked`，从 trainer 最近一次 update 收据取，
   **缺字段与 `True` 同义**：没有 update 收据、收据里没有证据块、结论不是 `bool`，
   三种情况都记"卡住"。被晋级的是这份权重，不是那次 update，所以结论必须写在权重的收据上。
3. **两个 MuJoCo launcher：发射时刻的门 + 摘要。**
   - `_fresh_wait_bootstrap_canary` 的 `passed` 现在**要求 `promotion_blocked is False`**。
     这是本轮真正补上的那个洞：改软之前，起手 hold 贴到关节硬边会进
     `hard_termination_count`，canary 直接不过；改软之后它不再进 hard，
     **没人读结论位的话这条 canary 就会静默放行一个"训得出来但不能上机"的起手姿态**。
     成本是 25 tick，拦在 GPU 排队之前。
   - 新增 `_promotion_blocking_summary()`：把 canary、每一份 update 收据、
     checkpoint save 收据的结论**并起来挂到 `result` 最外层**（`status` 旁边），
     并且**只在被卡住时**往 stderr 打一行
     `[MUJOCO-<profile>] WARN promotion_blocked=True blocked_sources=...`。
     照 Franco 的准绳走"WARN 必进摘要 / 摘要抓异常不抓预期"：
     没事不出声，出事同时出现在 result 顶层和 stderr，不留在几千行收据里等人翻。
     四个来源任一说不出结论就算被卡（`absent_verdict_counts_as_blocked: true`）。

**(3) 变异测试（证明门会开火，不是证明它现在不报错）。**
`test_action_ball_211_trainer.py` 新增 7 个变异体，全部必须转红：
`evidence_removed` / `conclusion_hardcoded_false` / `conclusion_hardcoded_true` /
`conclusion_is_a_truthy_int` / `reasons_emptied_while_blocked` / `counter_removed` /
`evidence_is_not_a_mapping`。其中 `conclusion_hardcoded_false` 就是 §5.6.13 (B)
点名"必须被杀"的那一个。**这些都构造成"粗一个档次的检查过不了"**：
只断言"字段还在"或"类型对"的检查会放过 `hardcoded_false`，只有拿计数对账才杀得掉。
另外 `test_mujoco_native_vec_env.py` 新增一条**活值**producer→consumer 交叉测试：
拿真的 `DiagnosticEventLedger.record_step()` 快照直接喂给 trainer 的消费方，
断言两边对同一个 reason 词表和同一个结论的判断一致——
不是第三份手抄的期望值，vec_env 那侧的拼写一漂就红。
launcher 侧新增 canary 的正/负对照与摘要 fail-closed 三格。

**(4) 治理表同批。** `mirrored_constant_registry.py` 的 `trainer.py` 段登记了三个新常量
（`PROMOTION_BLOCKING_EVIDENCE_KIND` / `PROMOTION_BLOCKING_REASON` /
`PROMOTION_BLOCKING_EVIDENCE_SOURCE`），档位 `not_mirrored`：
它们是本车道自己的收据词表，Isaac 那侧这一项仍是硬终止，压根没有"卡晋级"这个概念。
`PROMOTION_BLOCKING_REASON` 与 vec_env 里那份字面量的一致性**不靠这张表**，
靠上面那条活值交叉测试。（这道门本轮自己拦了一次：三个常量没登记时它就红了。）

**(5) 顺手扫的同类"没人读的结论位"。** 判据是"名字自称结论 + 全仓文本命中 ≤ 2 次"：
把 `scripts/` 与 `hope_training/` 下所有 `.py` 的 dict 字面量键取出来，
挑出名字以 `_blocked`/`_verdict`/`_violation`/`_passed`/`_blocker`/`_warning` 结尾
或含 `promotion_block`/`no_go`/`warn` 的，再和全仓标识符频次索引对。
命中 38 个候选，**逐个查了发出点的上下文，只有本条是真病**：

| 形态 | 例子 | 判决 |
| --- | --- | --- |
| 发出点自己就 raise / 决定退出码 | `oracle32_verdict`（`launch_n1_measured_vendor_v2_diagnostic.py:2094`，同一段里 `if failed: raise LaunchRefused`）、`runtime_blocker`（`mujoco_action_ball_policy_fitted_gate.py:2142`，同批 `status/verdict=BLOCKED` + `return 2`）、`torch_blocker`/`fitted_receipt_blocker`（`audit_action_ball_cross_engine_physics.py`，`verdict` 决定退出码 `0/3`） | **不是病**：结论位是已执行拒绝的存根 |
| 收据里的证据明细，不是结论 | `shadow_foot_penetration_violations`（`mujoco_teacher_motion_fitted_ball_gate.py:9184`，切片列表）、`mechanical_verdict_counts` | **不是病**：它本来就是给人看的账 |
| 结论位无人读、无测试、无分支 | `promotion_blocking_evidence.promotion_blocked` | **本条，已接线** |

结论：**这一类在本仓不是普遍现象**，`promotion_blocked` 是唯一一处
"文档明写供下游消费、实际零消费方零测试"的。§5.6.13 (A)(C) 那三处仍是
**零调用点的门**（另一种病），不在本轮范围内，判决与依据不变。

**(6) 没做什么、为什么。** 没有让 `checkpoint.save()` 在被卡住时**拒绝写盘**。
理由是证据：`joint_actual_forbidden` 的谓词没变，改软前 exact r6 的 A/C 每个 update
7 个硬终止全部就是这一条（§顶部状态表），所以改软后它在现役诊断跑里大概率非零；
拒绝写盘会把整条 cold-load parity 证据链一起打掉，那不是"卡晋级"，是"删证据"。
本车道的 `authorization.promotion` 本来就恒 `False`（`vec_env.py` 与 update 收据都是字面量），
**今天没有一个"晋级动作"可拦**——所以本轮把结论位钉在
"收据不许自相矛盾"+"canary 不许放行"+"摘要不许沉默"这三处能真开火的地方，
而不是造一个新的晋级仪式来给它拦。**哪一步算晋级仍然待定**，
但它不再是"接线的前置条件"：无论将来哪一步被定为晋级，
它要读的字段现在已经在 update 收据和 checkpoint 收据上了。

**(7) 收尾对拍。** 见提交信息里的 before/after 失败集合。

#### 5.6.17 三条 review 合并成一份带排序的清单（2026-08-07）

**人话一句：** 2026-08-06 三条独立 review（随机性 / MuJoCo 对齐 / 复查漏做的）各出了一份清单。
这一节把它们**合成一份**：去掉重复、把互相矛盾的当场查清后下判断（不各打五十大板）、
按「不做会不会毁掉四格的可信结论」×「会不会**静默**失败」排成三档，
最后单列一份给 `build_1` 加桌子的交接单。**本轮零代码改动，只有文档。**

##### 零、本轮实际复跑/复测了什么（不是转述别人的收据）

| 复核项 | 谁声称的 | 怎么验的 | 结果 |
| --- | --- | --- | --- |
| mjlab 对齐台账的 `21` 条测试 | mujoco-align | **自建**独立 pod worktree `/workspace/franco/mergerev_20260806`（`git clone --local` + `fetch origin` + `checkout fd5471da`），`mjlab_venv` 复跑 | `21 passed / 936.59 s`，与声称一致 |
| **那些变异测试真的会开火吗**（本轮验收重点） | — | 我**反过来把台账自己的检查改粗一档**再跑（见下一张表） | 两条都**如期变红**；恢复后 md5 回到 `fe1113a6…` |
| pod 上四个文件 = 本地提交版 | mujoco-align | `md5sum` 逐一比对 | 四份全等（`a9ada5c3…` / `4e95a22a…` / `fe1113a6…` / `30a88762…`） |
| 撞桌 `0% → 100%` | mujoco-align | 直接读 `RECEIPTS/SMOKE_TABLE.json` | `curve=[0.0, 0.037, 0.633, 1.0, ×8, 0.978]`、`peak=1.0`、`iterations_measured=12` |
| `_validate_reveal_bridge` 零调用点 | missed | 全仓 grep | **成立**（全仓只有 `:685` 那个 `def` 自己） |
| `promotion_blocked` 零消费方零测试 | missed | grep + `git show HEAD:` 对比 | 写的时候成立、**现在不成立**（矛盾 3） |
| 两道零调用点门 | missed | 同上 | 写的时候成立、**现在已闭**（`8a6554c7`，矛盾 4） |
| DR-L1 没有任何 launcher 入口 | randomness + missed | grep `DRL1Learnability` / `TASK_PROFILE_ID` | 写的时候**成立**（两个 launcher 都硬钉 `…DRL0Learnability`）；**2026-08-08 已修**：入口接好了，见 §12 `DR-LEVEL-LAUNCH-ENTRANCE`。剩下的不是"没入口"，是"这一档还没 materialize 过自己的 lineage" |
| 课程曲线臂的扩张上限就是 `0` | randomness | 从活 manifest 直接取值 | **成立，且本轮取到更精确的数**（见下） |
| `death_penalty` 活值 | randomness | `action_ball_211_four_grid_contract.py:347` + pod argv | **成立**：weight `-10.0`，post-dt `-0.2` |
| 六轴 push 值在厂商继承链上、只是被关掉 | randomness | `HOPEPingPongActionBallA3VendorV1.yaml:43-56` | **成立**：`enable: true`、`interval_range_s: [1.0, 3.0]`、六轴速度盒逐字 |
| 课程没有降级/回退路径 | randomness | grep `demote|rollback|shrink|regress` | **成立**：`action_ball_curriculum.py` 零命中 |
| `canonical_ready_mode` 硬拒起点分流 | randomness | `commands.py:2075-2082` | **成立** |
| `mobility_mode` 与 ramp 之间零交叉校验 | randomness | grep | **成立**：`mobility_mode` 在 `commands.py` 零命中 |
| `action_ball_task_wait.py` 的 docstring 说"没接线" | randomness | grep | **成立**：`hope_commands.py:4638` 真 import、`:4969+` 每次真 reset 消费 |
| 「四格 `scale4096` 正在发」 | missed（拿它当不接线的理由） | pod1 `ps` + `nvidia-smi` | **不成立**（矛盾 2） |

**反向变异（本轮验收的核心）。** 判据是「**把台账自己的检查改粗一个档次，指定的那条测试必须变红**」——
只跑一遍绿的不算数，那只证明它今天不报错。

| 我把台账改粗成什么 | 必须变红的测试 | 实测 |
| --- | --- | --- |
| `pd_gains` 由**逐关节**比对改成比对**总和**（`sum(kp)` 相等就判对齐） | `test_mutation_swapped_kp_flips_the_pd_gains_axis` | **FAILED**（`:179`，"the swapped Kp table was accepted"）—— 正是期望的红 |
| 撞桌那条的 `refusal_wired` 由"拒绝码在不在拒绝表里"改成"计数器字段存不存在" | `test_mutation_unwiring_the_table_refusal_is_caught_by_the_ledger` | **FAILED**（`:344`）—— 正是期望的红 |
| （对照）不改任何东西 | 上面两条 | `2 passed / 190.22 s` |

**结论：mujoco-align 那组变异测试不是摆设，它钉在守卫上而不是钉在"字段还在"上。**
选这两条不是随便挑的：`pd_gains` 那条对应"手抄常量"的形状，撞桌那条是本轮新增、
最可能被人当成"有计数器就够了"的那一条。

**本轮取到的两组精确数（前面几条 review 只给了比值，容易被下一轮照抄错）。**

其一，现役 launcher 绑的活 manifest
（`configs/action_ball_n1_measured_20260803/fresh_core_seed0_20260803_take061_robust20n_r8_splitready/take_061_unit04_bh.full.manifest.v3.7d2139028427.json`）
的课程 σ 上限逐项：

| 曲线轴 | `_std_lower_max` | `_std_upper_max` |
| --- | --- | --- |
| `base_spawn`（x,y） | `[0.0, 0.0]` | `[0.0, 0.0]` |
| `base_travel`（x,y） | `[0.0, 0.0]` | `[0.0, 0.0]` |
| `contact_offset`（3 轴） | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` |
| `incoming_speed` | `0.0` | `0.0` |
| `spin_magnitude` | `0.0` | `0.0` |
| `incoming/spin direction` 四个 tangent | `0.0` deg（共 `8` 项） | — |
| **`time_to_contact`** | **`0.12524752…` s** | **`0.12524752…` s** |

即 **`28` 条 σ 上限里 `26` 条是 `0.0`**，唯一非零的是 `time_to_contact` 上下各 `±0.1252 s`。
**把课程等级拉满到 `L=1.0`，题目也几乎不动**——在这份 manifest 上"解冻 32 臂课程"是 no-op。

其二，起始位置两套机制的实测量纲（此前只写了倍数）：
活 manifest 的站位硬框是 `base_spawn_min_w_xy_m = [-0.49223234, -0.11472119]`、
`base_spawn_max_w_xy_m = [0.10776766, 0.68527881]`（**半宽 x `0.30` / y `0.40`，中心不在原点**），
顶层 `mobility_mode: "no_move"`，`base_travel_max_b_yaw_xy_m = [0.0, 0.0]`；
而 `start_pose_ramp` 的端点是 x `[-1.0, 0]`、y `±1.2625`、yaw `±30°`
（`training_contract.py:5231-5246`）。y 半宽是站位框的 **`3.16` 倍**，x 最远处是 **`3.33` 倍**。
`mobility_mode` 在 `commands.py` 里 **零命中**——两条路径之间没有任何校验。

##### 一、三条 review 之间的矛盾，逐条裁决

**矛盾 1：死亡一次到底罚多少？`-6` 还是 `-0.2`（post-dt）。**
randomness 查到活值是 `-0.2`（weight `-10`），而 mujoco-align 在**同一天新写的 §9.2.9 里又写了一次 `-6`**。
**裁决：`-0.2`，依据是活值不是文档**——`action_ball_211_four_grid_contract.py:347` 写着
`"death_penalty": -10.0`，同一处注释自陈「`-300 -> -10`，post-dt 由 `-6.0` 降到 `-0.2`」；
pod1 收据 argv 同样是 `task.rewards.death_penalty_weight=-10.0`。
**后果不是措辞**：§5.3 那段拿 `-6` 算出"合法上台后同回合摔倒的最低事件净值 `+2.4`"；
按活值重算是 **`+8.2`**，**这个套利口子比文档写的大 `3.4` 倍**，那段"必须单独监控"的理由因此更强。
本轮把 §5.3、§9.2.9、§12 三处 `-6` **就地更正**。

**矛盾 2：四格 `scale4096` 是不是"正在发"。**
missed 拿「正在发」当作不接 reveal bridge 的理由，但**它自己那一节的 pod 复核**又写着
"无 `train.py` / `launch_action_ball` 进程"。
**裁决：没有在发。** 2026-08-07 复核 pod1：唯一的 `scripts/train.py` 是别人的
`task=HitterPingPongPhase114`，GPU1/GPU2 各 `8 MiB`、`0 %`；且 §5.6.15 已写明
`recipe` 阶段被 solver profile pin 拦下、`scale4096` 未跑。
**这条理由作废，(A) 因此从"发车后"提到"发车前"**（补强依据见 §5.6.13 (A) 的就地更正块）。

**矛盾 3：`promotion_blocked` 有没有消费方。**
missed 说全仓零消费方零测试——**写的时候成立，现在不成立**：工作区里
`mujoco_native/trainer.py`（`promotion_blocking_samples_from_step` /
`promotion_blocking_evidence_receipt` / `promotion_blocked_from_evidence`）、
`checkpoint.py`（save/load 收据带这一位、缺字段与 `True` 同义）、两个 MuJoCo launcher、
以及 `test_mujoco_native_vec_env.py`（`+68` 行）都已接线，另有 §5.6.16 的写作。
**但全部未提交**（`git show HEAD:` 对 `trainer.py` / `checkpoint.py` grep `promotion_blocked` 均 `0` 命中）。
**裁决：这一条不是"要做"，是"要落地"。未提交 = 不存在。**

> **2026-08-07 就地更正：已经落地了，本条作废。** `52e6199b`（11 个文件、`+746/-7`）
> 把三个消费方连同测试一起提交并 push。2026-08-07 独立复跑四个变异（C1 拆掉
> `run_update` 里的消费、C2 保留消费但去掉"计数 vs 结论"对账、C3 把结论位从 canary
> 的 `passed` 里摘掉、C4 让 checkpoint 收据把这一位写死 `False`）**四条全部转红**，
> 见 §5.6.18 一。下面二.B2 那一行同步作废。

**矛盾 4：三道零调用点的门。**
missed 说三道全是零调用点——写的时候成立。**本节写作过程中前两道落地了**：
`8a6554c7`（2026-08-07 00:17）把 `_require_fresh_order_sentinel`
（`launch_action_ball_curriculum.py:4378`）与 `_valid_table_guard_attribution_summary`
（`launch_n1_vendor_baseline_diagnostic.py:2261`）都接上了，并各配两条变异测试
（M2 "改成按集合比而不是按位比"、M4 "table_count 取自摘要自己"——都是**粗一档就能过**的形状）。
它同轮独立重跑了零调用点普查（`1009` 个 `.py`、`19507` 个模块级 `def`，
**把代码引用与文档提及分开计数**——上一遍混在一起，而本文档正好按名字讨论这些函数，
把它们的计数抬高、恰好盖住了要找的东西），结论仍是同样五个，没有第六个。
**`_validate_reveal_bridge` 仍是零调用点**，且同轮已把逐步接线方案写进 §5.6.13 (A)。
（**2026-08-07 就地更正：当天已按该方案接线，见 §5.6.22；本行只作接线前的记录。**）
**裁决：前两条已闭；第三条是本文档目前唯一真开着的那个口，见 B1。**

**矛盾 5：推撞该先做什么。**
Franco `48174f23` 已裁：六轴 push 值与厂商逐字相同、接线完整，**打开它是改配置不是写实现，
是最低代价最高优先**。randomness 的 P3 说"不要先去调 interval，先跑 `lateral_perturbation` 的门"。
**裁决：两句话讲的不是同一件事，合起来才是可执行结论。** Franco 裁的是**做哪一件**（push，因为便宜）；
randomness 给的是**用什么剂量**。剂量的算术如下（本轮重算，用活值）：

- 四格 episode 预算 `10 s`（`HOPEPingPongActionBall.yaml:52`），但单挥拍任务真正"活着"的窗口是
  WAIT `0.10..0.50 s` + `time_to_contact` `1.70..1.95 s` ≈ **`1.8..2.45 s`**。
- 厂商 cadence `[1,3] s`（均值 `2 s`）：活窗口内约 **`1.1` 次/episode**，整 `10 s` 预算内约 `5` 次。
- §6 写的 `10..30 s`（均值 `20 s`）：活窗口内约 **`0.11` 次/episode**，整 `10 s` 预算内约 `0.5` 次。
- §6 的目标是「每 episode 命中击球窗期不超过约 `.1` 次」。
  **只有把"击球窗期"读成"活窗口"，`10..30 s` 才恰好达标**；读成整 episode 预算就超标 `5` 倍。

**所以：打开 push，cadence 用 §6 的 `10..30 s`，不要照抄厂商的 `[1,3] s`（那是目标的约 `10` 倍）；
并且把 §6 那句判据钉成「reveal 之后到本拍结束」——正好对上"活窗口"那一读。**
`lateral_perturbation` 的相位门控留在后面，**不作为打开 push 的前置**：它自陈 launch-ineligible
（`lateral_perturbation.py:16/24-25`，全场 solver-response 与吞吐门未跑）。

**矛盾 6：friction 与 obs history。**
randomness 初稿判成真缺并给了对齐建议，Franco `48174f23` **两条都驳回**。
**裁决：已裁定，本清单不收这两条，下一轮 review 也不要再提。**
连带作废的还有 action delay 那行"需先补 history"的理由——delay 要做得按自身理由重新论证。

##### 二、发车前必须做的（判据：不做，四格就会给出一个"看起来通过了"的结论）

| # | 一句人话 | 这是什么 | 为什么非发车前不可 | 代价 |
| --- | --- | --- | --- | --- |
| **B1** | 让 `4096x5` 那道共享门**真的去读**每个 update 都在产的 reveal→playback 桥 | 桥 = "揭示任务那一刻的题目 / 采样器 / 奖励配方 / WAIT 表 / 计时合同这 `7` 个权威 SHA，加上逐 WAIT 档的 reveal→playback 寿命守恒，加上一个 `status` 位" | 桥没配起来时生产方返回 `status="not_configured"`，严格消费方要求 `active_fail_closed`——**接了线就是拒收，现在是静默放行**。四格的全部结论都建立在这道门上 | **具体怎么做 §5.6.13 (A) 已经写了逐步方案（`8a6554c7` 落的），照着做即可，本节不重复**。这里只补三条：(1) 纯 CPU，且**不动 `solver_profile_sha256`**（该门不在那 `5`(+`2`) 个源文件里），零 lineage 代价；(2) 那份方案的排期写着"等 `scale4096` 落地后执行"——**该前提不成立**（矛盾 2），已就地更正；(3) 方案 2(a) 那句"接线前先拿一格已落盘的 `scale4096` 收据实测一遍 `status`"是**真的要先做的一步**，因为现在没有一格跑完过，`status` 到底是 `active_fail_closed` 还是 `not_configured` 谁都没实测过 |
| ~~**B2**~~（**2026-08-07 已完成**，`52e6199b`，变异复跑见 §5.6.18 一） | 把 `promotion_blocked` 那三个消费方落地（写完了，**没提交**） | 见矛盾 3；两道零调用点门已于 `8a6554c7` 落地，不在本条范围内了 | 未提交的护栏在发车那一刻不存在。而 `promotion_blocked` 守的正是"`joint_actual_forbidden` 改软之后**不终止但卡晋级**"的后半句——四格恰恰是它会非零的场景（exact r6 每个 update 的 `7` 个硬终止全是这一条） | 已经写完了（`trainer.py` / `checkpoint.py` / 两个 launcher / `+68` 行测试 / §5.6.16），只差提交 |
| **B3** | 把 §5.3/§5.4 的静态奖励层级账**对真正要发的 DRL0 leaf 重算一次**并落收据 | 那份账是"模仿 < 击中 < 上台"这条准绳的数值证明 | 审计器自 `635252f6` 起已经能读 DRL0 leaf，但**至今没有一份对它重算的收据**。四格的头号读数就是这三层的相对高低——账没重算 = 头号读数没有基准。而且 §5.3 那段的算术本轮刚被证明是错的（矛盾 1） | 纯 CPU host，不占 GPU |
| **B4** | 给"权重非零、被门确认在编、但 kernel 结构上恒返回零"的奖励项加一条会说话的检查 | 现役 manifest 开着 `counter_rally_objective`，于是 `virtual_pass_net`（admitted 权重 `20.0`）与 `virtual_spin` **整窗恒零**（`hope_rewards.py:4836` / `:5019`） | `4096x5` gate 只按 `motion/strike/target/outcome` 四**组**报分母与收入、不逐**项**报，所以 outcome 组里躺着一个死项**看不出来**——四格读出来的 outcome 收入结构是错的，且没有任何机制会说 | 在 prelong 语义收据里逐项报 eligible 分母与收入，并对"权重非零 × 连续五个 update 收入恒为 `0`"出一条 blocker |

##### 三、发车后补的

| # | 一句人话 | 这是什么 | 为什么可以等 |
| --- | --- | --- | --- |
| **A1** | ~~给 `DRL1Learnability` 两片 profile 接一个 launcher 入口~~ **2026-08-08 DONE**，见 §12 `DR-LEVEL-LAUNCH-ENTRANCE` | DR-L1 = 恢复五条 plant 随机化（friction / 关节零点 / CoM / link mass / Kp-Kd）+ `start_pose_ramp` 的那一档 | DR-L1 **按定义不能进四格**（四格刻意只差 obs-noise 一根轴）。入口已接：`--dr-level dr_l1` 会解析到 DR-L1 的 profile/manifest/finalizer 合同，并把六条轴的实际取值写进收据；今天它在 lineage 那一步 fail closed，因为这一档还没 materialize 过自己的 lineage |
| **A2** | mjlab lane 的撞桌要不要装成硬终止（**要 Franco 拍板**） | 该 lane 的策略在第 `3` 次 PPO 更新后几乎每局都把球拍搁在桌上（`0%→100%`，主犯 `robot/right_racket_collision`），Isaac 对同一事件是硬终止 | 是**发车决定**不是护栏问题：装了会改训练分布，新 run 与既有 `103` 条收据不可比。但不装 = 一直付钱让它学一个 Isaac 判死的动作 |
| **A3** | 删掉 pod 上 `/workspace/mjlab_lane/geometry.py` 那份**优先于仓库**的字节拷贝，或在启动时比摘要并 fail closed | 部署目录自带一份 `geometry.py`，`a3_court_env` 的解析顺序优先用它 | 今天两份内容相同（`md5 0f334186…`）。这是"**连指纹都没有**"，比"指纹不等于语义"还低一档 |
| **A4** | 那个部署目录发车前先同步到 `fd5471da` | 它还是旧代码：没有撞桌探针、没有对齐台账 | 不同步 = 新收据里没有 `isaac_alignment` 块，`--report` 会以 `ROBOT_TABLE_CONTACT_NOT_MEASURED` 拒 |
| **A5** | 打开六轴 push，cadence 用 `10..30 s`（**不是**厂商的 `[1,3] s`） | 见矛盾 5 | 必须走 fresh lineage 的新臂，不能进四格 |
| **A6** | 三处零风险文字修复 | (a) `training_contract.py:5226-5228` 的"hold 起点 `45-60` → 终点 `20`"注释：**"45-60" 在仓内只出现在一份 DR-L1 测试 fixture 里，没有任何活配置出处**（Hitter yaml 是 `[50,200]`、HitterPureRally 是 `(25,125)`）。注意这段并不是天真的自相矛盾——同一个 block 往下 `40` 行有一段 2026-08-05 的事实核查，已经把 hold 的归属交给 `task_wait`、并解释了常量为什么钉 `[0,0]`；**要改的是那段过期的头注释，不是常量**。(b) `action_ball_task_wait.py` 的模块 docstring「deliberately not wired into an environment」——`hope_commands.py:4638` 真 import、每次真 reset 消费；只读这句会得出"训练侧没有 WAIT"的错误结论。(c) §6 那句「每 episode 命中击球窗期不超过约 `.1` 次」钉成「reveal 之后到本拍结束」 | 都不改行为 |

##### 四、记着但暂不做的（每条写清**为什么现在不做**）

| # | 一句人话 | 为什么现在不做 |
| --- | --- | --- |
| **C1** | hold 窗口随熟练收窄（新开 `action_ball_pre_task_wait_schedule_v2`，`5..25 → 3..25 → 2..25` 分档） | 两个理由，任一都足够：(1) 它**改的是题目分布**，四格归因跑期间动它就毁掉"只差一根轴"的设计；(2) `seed/min/max/horizon/required_active` 一起进 DR-L0 的**内容哈希**，就地改会连带换掉 DR-L0 的身份。**必须新开 schedule kind，不能在 loop 内改** |
| **C2** | 起始位置泛化（`base_spawn` 课程臂 **或** `start_pose_ramp`，两套先二选一） | 它**改支撑集**。而且今天两套机制之间没有任何交叉校验（零节的量纲表）。要做的第一步是补一道 fail-closed 门——ramp 端点必须落在 `base_spawn_min/max_w_xy_m` 内、`mobility_mode=="no_move"` 时禁止非零 ramp、`x` 上界 `0` 写成门而不是巧合——**不是先放大范围**。两条独立分析都判"课程侧为主"：只有它由熟练度驱动、可逆、有逐臂分母 |
| **C3** | 课程的降级 / 回退路径 | `action_ball_curriculum.py` 全文没有 demote/rollback/收缩。它是 C1 的前置（"必须允许降档"），C1 不做它就不急。但要记着：§6.1 要求的"可逆回退"目前只有"独立 new-band 分母"那半句成立 |
| **C4** | 推撞的四相位（pre-strike/strike/follow-through/recovery）暴露统计 | §6.1 明写要按相位统计，而 `apply_push_robot_event` 装的是**裸 interval 事件、零相位计数器**。等 A5 打开 push 那一批一起做——现在 push 是关的，先做也没数可统 |
| **C5** | `lateral_perturbation` 的相位门控扰动（`recovery_hold` 资格窗、strike 中断计数、冻结 L0/L1 冲量 `0.04–0.08 m/s`） | 实现完整，但自陈 launch-ineligible：全场 solver-response + 吞吐门没跑过。那道门 CPU/单卡就能跑，但不在四格关键路径上 |
| **C6** | 起点分支（post-swing / 失败加权 RSI），即尽调 §9.5 R2 | `commands.py:2075-2082` 在 `canonical_ready_mode` 下**硬拒** `stand_start_prob != 1.0` / `post_swing_start_prob != 0.0`。要 adapt `build_1` 的 `25/25` 分流，必须先拆 `canonical_ready_mode` 的两个职能（契约绑定 vs reset 分布裁定）；那本身是零行为变化的重构，但排在四格之后 |
| **C7** | action delay `[0,2]` 控制步 | Franco 驳回 obs history 之后，delay 那行"需先补 history"的理由**作废**；它现在**没有理由**，要做得先给一个 |
| **C8** | 解冻 `32` 臂课程 | 在现役 manifest 上是 **no-op**（零节实测：`28` 条 σ 上限里 `26` 条是 `0.0`）。真正的动作是切到非退化 manifest（`73` 库那份有预算），那是**换题**不是调参 |
| **C9** | link mass `±15% → ±20%`；pseudo-inertia 独立扰动 | Franco `48174f23` 已排序：mass 是幅值差里唯一方向明确、代价可控的一条；pseudo-inertia 仓内**无机制**，属"要新写"，与幅值调整不是一件事。两者都排在 A1（DR-L1 能发车）之后 |
| — | ~~friction 对齐厂商 `(0.2,1.8)/(0.2,1.5)`~~ | **Franco 2026-08-06 驳回：不是缺口。不要再提。** 独立理由：本仓摩擦是对着 MuJoCo 标定过的，"与厂商不同"本身不构成缺陷证据 |
| — | ~~obs history=8~~ | **Franco 2026-08-06 驳回：一步 previous action 是设计选择，不是缺。不要再提。** |
| — | ~~CoM 从 `torso_link` 扩到全身~~ | Franco：扩全身会动到**拍子所在链**，与 measured-racket authority 交叉，不在范围内 |
| — | ~~地形凹凸垫~~ | §3.3 已明确拒绝（parkour 专属、任务不对齐）。机制存在但无人设键 = 平地，**这不算缺口，不该补** |

##### 五、给 `build_1` 加桌子的交接清单

§5.6.12 已经写了**六个坑**（代理余量 `20 mm` 是门不是接触 / 广相 AABB 不是终局判据 /
出生姿态非持拍左手离台板 `32 mm` / 子类覆写对桌子车道指纹隐形 / 多一条终止项会改整条归因顺序 /
桌底除 keepout 外无碰撞体）。**本轮跨 review 合出第七个**，它不在那六个里，
而且是唯一一个"**不改代码也会持续发生**"的：

**坑 7：奖励在付钱让球拍靠近球，而球在桌子上方——没有终止项的话，策略会学会把球拍搁在桌面上。**

- **症状（实测，不是推测）**：mjlab lane，`512` 世界 `12` 次 PPO 更新，两个 seed 独立复现：
  每局至少碰一次桌子的比例 `0.0 → 0.037 → 0.633 → 1.000` 并稳在 `1.0`。
  按 geom 名独立点名，主犯是**球拍本身** `robot/right_racket_collision`（`4262` 行），
  其次 `left_elbow`（`125`）、`left_wrist_roll_1`（`123`）。
- **机制**：该 lane 的 `w_reach=2.0` / `w_touch=4.0` 在付钱让球拍靠近球；球在桌面上方；
  把拍子搁在桌上**同时**降低"够不着"的代价并买到平衡。这条 lane 没有任何终止项或罚款拦它。
- **和 `build_1` 的关系**：`build_1` 现在**没有桌子**，所以也没有 `robot_hit_table`。
  加了桌子而**不同时**加终止或撞桌罚，它会不会长出同一个套利，取决于它自己的 reach/touch 类整形项——
  **加桌子那天先看这个，再看那六个坑。**
- **该做什么**：加桌子的**同一批**里，要么装 `robot_hit_table` 硬终止（Isaac ActionBall 的做法，
  `hope_env_cfg.py:704-723`），要么装 `table_hit_penalty` 这一族窄罚
  （`hope_env_cfg.py:762`，默认 `weight=0.0`，`reward_pack=v2` 给真值，
  只认 `robot_hit_table` 一个终止原因，所以能和摔倒**分开定价、分开消融**）。
  **"只加桌子不加价"那条路已经被实测走过了，结果是 `100%`。**
- **顺带纠正一个会被照抄的数**：撞桌/摔倒的 post-dt 罚在现役四格是 **`-0.2`（weight `-10`）**，
  不是本文旧处写的 `-6`（矛盾 1）。

**排序提醒（这条本轮从"散文"升级为"必须有测试"）**：§5.6.12 坑 4 —— 给
`table_termination.verify_isaac_source_authority()` 的 `HOPEActionBallTerminationsCfg`
补上 `class_assignments|robot_hit_table` 选择器 —— **必须排在加桌子之前**。
今天它只有 `class_header`（只哈希装饰器/基类/关键字、**不含类体**），
而该子类确实在覆写终止项（`hope_env_cfg.py:2471` 的 `ee_body_pos`）。
它不是常量，所以进不了 `OPEN_MIRROR_DEBT`（那张表对常量债是强制且会红的），
**目前只活在散文里，没有一条会红的测试逼人做**。补选择器的同批要加它自己的变异测试：
让子类覆写 `robot_hit_table`，**单独调** `verify_isaac_source_authority()` 也必须红
（今天它被 `vec_env` 链上的 `live_declared_term_blockers()` 兜住，但那不是这枚指纹的功劳）。

##### 六、这一节不代签什么

1. **不代签"发车前必须做"里任何一条已经做了。** 本轮零代码改动，只有文档。
2. **不代签 mjlab lane 的撞桌率在 `4096` 世界、长预算下还是 `100%`。**
   那个数是 `512` 世界 / `12` 迭代 / 两 seed 的烟测规模。
   能代签的是**方向与机制**（从 `0` 学到约 `1`，主犯是球拍），不是绝对数。
3. **不代签工作区里 `promotion_blocked` 那三个未提交的消费方会被提交。**
   矛盾 3 的状态是"待落地"，不是"已完成"（矛盾 4 的两条已于 `8a6554c7` 落地）。
   本节写作过程中并发方就落了一次地——**读这一节时先 `git log` 看一眼，别把"待落地"当成"没做"**。
4. **不代签 A 族那条 measured clip 的 `oracle32` 拒绝该怎么处理。** 按交代本轮未碰 A 族的门。
5. **不代签本节的三档排序在 Franco 的目标函数下是最优的。**
   排序判据只有两条（对四格结论可信度的影响 × 静默失败可能性），
   凡是"要 Franco 拍板"的（A2、C2 的二选一、push 剂量）都已单独标出，没有替他定。

#### 5.6.18 三条 review 的独立验收：把门改回去，看测试红不红（2026-08-07）

**人话一句：** §5.6.17 是把三份自述合并成一张清单，**它读的是别人的收据**。
这一节相反：**默认三份自述都不准，逐条自己动手核**——把每一道声称"已接线"的门
改回接线前的样子，看指定的测试会不会真的变红；把每一条声称"已有别的机制管"的删除，
去核那个机制今天真的在跑。结论是**三条自述基本都站得住，但有两处新发现，
其中一处直接推翻了一句写进本文的结论**。

##### 一、把门改回去：16 个变异，16 个转红

判据不是"跑一遍绿"，是"**把源码改回没有这道门的样子，指定的测试必须红**"。
16 个变异**全部改的是被测源码本身**，不是测试。改完跑、跑完 `git checkout --` 还原，
还原后的对照组两次都是绿的。

| # | 我把源码改成什么 | 必须红的测试 | 实测 |
| --- | --- | --- | --- |
| A1 | 把 `_require_fresh_order_sentinel(...)` 那一句整个删掉（回到零调用点） | 课程 launcher 的 N5 顺序哨兵 | **红** |
| A2 | 门还在，但 `tuple(raw) != ACTION_ORDER` 改成**按集合比** | 同上 | **红** |
| A3 | 门还在，但**只比长度** | 同上 | **红** |
| A4 | 检查照跑，但**不把它的摘要记进 launch claim**（收据不自陈） | 同上 | **红** |
| B1 | 把 `_valid_table_guard_attribution_summary(...)` 那次调用短路掉 | N1 probe-gate 消费方 | **红** |
| B2 | 门还在，但去掉"与独立重算出来的撞桌终止数对账"那一项 | 同上 | **红** |
| B3 | 门还在，但只查 category/phase，不查 sparse-cell 展开 | 同上 | **红** |
| B4 | 门还在，但不再要求 `enabled is True`（即允许 probe 收据自称"仪器没开"） | 同上 | **红** |
| C1 | 把 `run_update` 里读 `promotion_blocked` 那段拆掉（回到没人读） | trainer / 活值 / launcher 三处 | **红** |
| C2 | 还读，但去掉"结论位 vs `joint_actual_forbidden` 计数"的对账 | trainer + 活值 | **红** |
| C3 | 把结论位从 canary 的 `passed` 条件里摘掉 | MuJoCo launcher | **红** |
| C4 | checkpoint 存/取收据把 `promotion_blocked` 写死 `False` | trainer | **红** |
| D1 | 把 `train.py` 那道 legacy set **只删掉一个名字**（字符串仍留在文件里） | 225 退役收据 | **红** |
| D2 | 把 runner 那道 `if` 整个删掉 | 同上 | **红** |
| D3 | 删掉 runner 那道 `if`，但留一句写着 `action_ball_a225` 的 TODO 注释 | 同上 | **红** |
| E1 | 从 mirrored-constant 台账里删掉三个新常量中的**一个** | `test_mirrored_constant_registry.py` | **红**（`2 failed / 13 passed`） |

**A4 / B4 / C4 是本轮新加的三条**，原三份自述里没有，专门补三个"半边"：
A4 打的是"检查跑了但收据不自陈"（Franco 的准绳里这算没做完），
B4 打的是"忘了 argv 规则"（probe 阶段强制 `+task.table_contact_attribution_diagnostic=true`，
所以收据自称没开仪器只能是缺陷），
C4 打的是"结论只写在 update 收据上、不写在被晋级的权重上"。

**D1 / D3 是"grep 会放过、AST 才杀得掉"的那一对**：D1 之后 `action_ball_c225`
这个字符串仍然在 `train.py` 里（只是不在判据 set 里了），D3 之后 `action_ball_a225`
仍然在 runner 里（在注释里）。`grep "action_ball_a225"` 对这两个变异体都是通过的。
收据读的是**判据节点本身**（`Compare(Name(x), In, Set(...))`），所以两个都杀得掉。

**E1 验的是治理这一层**：`52e6199b` 自述"mirrored 台账那道门在三个新常量没登记时
自己拦了一次"。这条是**非人为的开火证据**，比人写的变异更可信，所以专门复现了一遍——
只删一个登记项就 `2 failed`，还原后 `15 passed`。

对照组：正式 venv 下 `94 passed / 1 skipped`，`/usr/bin/python3` 下 `31 passed`，
变异前后各跑一次，两次都一样。

##### 二、两个新发现

###### 2.1 那道 N5 顺序哨兵的测试，在大家平时跑套件的 venv 里根本不执行

**这条差点让我自己写出一份假收据。** A1（把门整句删掉）在
`/workspace/hope_isaac_venv` 下的输出是 `1 skipped`、**退出码 `0`**——看上去是绿的，
差一点就要写"做不到，说明没接线"。换 `/usr/bin/python3` 再跑，四条 A 变异全红。

| 解释器 | `test_launch_action_ball_curriculum.py`（`59` 条） | 缺什么 |
| --- | --- | --- |
| `/workspace/hope_isaac_venv/bin/python` | `6 passed / 53 skipped` | `cryptography` |
| `/usr/bin/python3` | `59 passed` | —— |

反过来也一样：`/usr/bin/python3` 跑 225 退役收据那三个模块时 `43 passed / 17 skipped`，
skip 理由全是 `No module named 'hydra'`；同样三个模块在正式 venv 下 `60 passed`。

**所以：pod1 上没有任何一个解释器能跑全这套护栏。** `cryptography` 在仓内
任何 `requirements*.txt` / `*.toml` / `*.cfg` 里都**没有声明**，
所以这不是"某人忘了装"，是没人写下来过。
后果不是抽象的：本文 §5.6.13 (C-1) 那张 M1–M4 表**只在 `/usr/bin/python3` 下成立**
（原文确实标了解释器，但没说另一个解释器会静默跳过），
而§5.6.2 之后所有"全量套件对拍"都是用正式 venv 跑的，
**课程 launcher 的 `53` 条护栏测试在那些对拍里一次都没执行过**。
**该做的最小动作**：往 `hope_isaac_venv` 里装 `cryptography`，
并把它写进依赖清单——一行 `pip install`，但要连"写下来"一起做。

> **2026-08-07 已执行并复验**（本节写作时未做，理由是 venv 多 workflow 共用；实测该顾虑不成立，
> `cryptography` 是纯新增依赖、不动任何已装包的版本）：
> `hope_isaac_venv` 已装 `cryptography 50.0.0`，三件套现为 `cryptography / hydra / torch` 全 OK。
> 立即在正式 venv 下重跑 `tests/test_launch_action_ball_curriculum.py`：**`58 passed`**
> （此前是 `6 passed / 53 skipped`）。
> **好消息是那 53 条本来就是绿的，所以这段盲区没有藏着回归**；但"我们对它们是瞎的"这件事本身是真的，
> §5.6.2 之后所有用正式 venv 做的"全量对拍"都不覆盖这 53 条，**那些对拍的结论要按此打折**。
> `/usr/bin/python3` 仍缺 `hydra`，所以"没有单一解释器能跑全"这句**依然成立**——
> 真正收口要么给 `/usr/bin/python3` 装 hydra，要么明确规定护栏套件只在正式 venv 下跑并让 CI 强制。
> **依赖清单那一半仍未做**（`cryptography` 至今没被任何 `requirements*.txt` / `*.toml` / `*.cfg` 声明），
> 所以换一台机器/重建 venv 会原样复发。这半件事保留为待办，不要因为"pod 上现在能跑了"就当已完成。

###### 2.2 零调用点的门有第六个：`ActionBirthBroker.assert_known_generation`

§5.6.13 (C-2) 写着"重扫仍是 5 个，没有第六个"。**这句是错的。**
错因很具体：那次重扫的名字形状清单 `_validate_*` / `_require_*` / … / `_must_*`
**每一条都带前导下划线**，于是所有**公开方法**都被漏掉；并且 `def` 只收模块级。
2026-08-07 用同样的 token 频次法，但 (i) 加上 `assert_*` / `validate_*` / `verify_*` /
`require_*` / `check_*` / `ensure_*` / `enforce_*` 这些不带下划线的形状，
(ii) `def` 收集含嵌套（类里的方法也算），(iii) 排除 `.claude/worktrees/`
（那底下有别的 agent 会话留下的 `2674` 个 `.py`，不排掉会把别人树里的调用点当成本仓的）。
规模：`1005` 个 `.py`、`24593` 个 `def`（含嵌套）、门形状 `1911` 个。
结果是**四个**零调用点（代码命中 `<= 1`）：

| 函数 | 位置 | 代码命中 | 文档命中 | 状态 |
| --- | --- | --- | --- | --- |
| `_validate_reveal_bridge` | `action_ball_4096x5_prelong_gate.py:685` | 1 | 6 | **2026-08-07 已接线并配 7 条变异测试（§5.6.22）**；**2026-08-08 第一次碰真数据就拒了跑完的 C0/C1，两条门自己有问题，已修（§5.6.27）** |
| `_validate_artifact_path_hash` | `canonical_motion_bank_gate.py:3905` | 1 | 1 | 已登记，只登记不判决 |
| `_reverify_receipt_contact_source_files` | `canonical_neutral_ready.py:3907` | 1 | 1 | 已登记，只登记不判决 |
| ~~**`assert_known_generation`**~~ | ~~`mdp/action_ball_runtime.py:6390`~~ | 1 | 0 | **2026-08-07 已删除，见 §9.2.13** |

（原五个里的另外两个已于 `8a6554c7` 接线，所以现在剩四个；本条删除后剩三个。）

> **2026-08-07 就地更正：下面这段"自证循环"的判断被实跑推翻了一半，判决也随之改了。**
> 下文说"没有任何一处拿 broker 的 transcript 当第二个证人"——**这句是错的**。
> `LazyActionTaskPool.load_state_dict` 在那句自证的下面十几行，就把**每一份**
> retired birth 拿去过 broker 的 `assert_consumed_birth`（`action_ball_runtime.py:11125`）。
> 拿活对象实跑：伪造一条 broker 从没发过的代次的退役记录，正是被这一句拒的
> （`birth is not the env's exact consumed generation`）。
> 而且这道死门**比它粗一档**——只问"代次发过吗"，会放过"代次真、内容假"的伪造。
> **所以判决从"登记不接线"改成"删除"**，并配了 6 个变异体（5 红 1 诚实地没红）。
> 全部证据见 §9.2.13。下文保留原样，作为"当时是怎么判错的"的记录。

**它是什么。** 它自己的 docstring 写着："a checkpoint must never invent retirement
provenance for an env/generation the broker has not issued"——
**存盘不许凭空编出一个 broker 从来没发过的 env/代次的退役来历**。三层复核：

- **机制码**：现役退役路径 `LazyActionTaskPool.retire_many` / `_retire_many_diagnostic`
  （`:9930+`）比的是**池子自己**那张活 birth 表和它自己的 `_retired_generation`；
  `load_state_dict`（`:11087-11096`）从**同一份存档**里的退役 birth 记录重算
  `expected_retired_generations` 再和存档里的台账比。两边都来自同一个文件，
  **是自证循环**——正是本轮变异 B2/M4 专门用来打的那个形状。
  没有任何一处拿 broker 的 transcript 当第二个证人，而那正是这个函数存在的理由。
  同一个类上另一个 `assert_consumed_birth` 是有调用方的，所以这不是"整类方法都没人用"。
- **实验史**：`3e64bea9`（08-07 凌晨）刚刚查过隔壁——三处自称"独立见证在
  `ActionBirthBroker.load_state_dict`"的注释在 live-only 档下**根本走不到**，
  因为 broker 的 `load_state_dict` 只在 `_action_ball_load_exact_resume_state_dict`
  里被调用，而那个函数第一句就把 live-only 顶回去。也就是说 broker 侧的见证
  在这条车道上本来就薄，这个零调用点是同一片区域的第二处。
- **现役 argv**：四格与 N1 都是 live-only 诊断跑，跨进程续跑（exact-resume）不在
  发车关键路径上。**所以它今天不会造成错误结论，但它和 (A)(B)(C) 是同一个病。**

**裁决：登记，本轮不接线。** 理由是它属于 exact-resume / broker 那条车道，
而那条车道 `3e64bea9` 刚动过；接线要动 `hope_commands.py` 附近，
且要先定清楚"退役来历该跟谁对账"。**不接线不等于可以忘掉**——见六。

##### 三、三份自述里被推翻或需要就地改的

| 自述 | 谁说的 | 核出来 |
| --- | --- | --- |
| "重扫仍是 5 个，没有第六个" | dead-gates（写进 §5.6.13 C-2） | **推翻**，见二.2。已就地更正 |
| "`promotion_blocked` 写完了但未提交 = 不存在" | §5.6.17 矛盾 3 / B2 | **过期**：`52e6199b` 已提交并 push。已就地更正两处 |
| "四格 `scale4096` 正在跑，所以不碰那道共享门" | dead-gates / unread-bit | **第三次证伪**。2026-08-07 复核 pod1：GPU0 上唯一的 `train.py` 是别人的 `task=HitterPingPongPhase114`（`6390 MiB`），GPU1/GPU2 各 `2 MiB`；全机无 `launch_action_ball*` 进程。§5.6.17 矛盾 2 已经判过一次，这里只是再确认一次 |
| "接线前先拿一格已落盘的 `scale4096` 收据实测 `status`" | dead-gates 的方案 2(a) | **既做不到也不必做**，静态三行就能给出更强的答案。已就地更正，见六 |
| "M1–M4 四条变异全红" | dead-gates | **成立**，但只在 `/usr/bin/python3` 下成立。已补 §5.6.13 (C-3) |
| "225 四道门任一都足以拦死" | family-225 | **成立**，逐道核过，见四 |
| "三个被删的测试只测死文件" | family-225 | **成立**：三个文件各自只用 `importlib.util.spec_from_file_location` 加载**一个**被删脚本，再无其它本仓 import |
| "零指纹需要重钉" | family-225 | **成立**：`test_launch_action_ball_a211_four_arm_diagnostic.py:1962` 明确断言 `action_ball_225_trainability.py` **不在** `RUNTIME_SOURCE_PATHS` 里 |
| "38 个候选里只有 `promotion_blocked` 是真病" | unread-bit | 抽查两条**成立**：`oracle32_verdict` 的 `raise LaunchRefused` 就在同一函数上方 8 行；`runtime_blocker` 与 `status/verdict=BLOCKED` + `return 2` + stderr `[FATAL]` 同批写出 |

##### 四、"删了因为别的机制管着" —— 那个机制真的在跑吗

225 整族退役靠四道门。逐道核：

| 门 | 谁在守 | 今天真的在跑吗 |
| --- | --- | --- |
| 1. gym 注册表里没有 225 | `test_action_ball_task_config.py:313-314` 对**活的** `registrations` 映射断言两条 225 id 不存在 | **在跑**（该模块正式 venv 下 `60 passed`），且注册文件里确实只有 `A211Learnability` / `C211Learnability` 两条 |
| 2. 没有 EnvCfg 声明 225 的 `obs_mode` | 本轮新增收据里的 AST 检查 | **在跑**，且**不空转**：它先断言"至少解析到一个 `obs_mode` 默认值"，读法坏掉会自己红 |
| 3. `train.py` 拒 actor 合同 | 新增收据比对**活值** `_LEGACY_MODES` | **在跑**：变异 D1 转红 |
| 4. runner 拒 `obs_mode` | 同上 | **在跑**：变异 D2/D3 转红 |

留下来的 225 字样也逐条核过，全部是**拒绝表 / 历史件**，不是活路径：
`action_ball_211_transition_preflight.py` 的 `LEGACY_EXPERIMENT_NAMES` / `WRITER_SOURCE_NAMES`
（拿去和 `/proc/*/cmdline` 与旧日志命名空间比对，不做路径解析）、
A211 launcher 那 11 条"复活退役血统必须被拒"的参数化、
安全词表的 8 个 holder（`== 8` 与"路径去重后也是 `8`"两条都钉着，8 个全是活的 211 车道文件）、
以及 `configs/action_ball_n1_measured_20260803/` 下两份 0803 的历史血统 JSON。

##### 五、全量套件对拍（pod1，`-n 64`）

两棵独立 clone：`before` = `fd5471da`（三条改动之前那一版），`after` = `52e6199b`（HEAD）。
同一个解释器（`/workspace/hope_isaac_venv`）、同一组目录
（`tests` + `wbt/tests` + `wbt/mujoco_native/tests`）、`-p no:randomly`。

| | before（`fd5471da`） | after（`52e6199b`） |
| --- | --- | --- |
| collect-only | `10440` | `10366` |
| failed | `265` | `267` |
| passed | `9982` | `9909` |
| skipped | `175` | `172` |
| errors | `19` | `19` |
| 失败节点数（`FAILED` + `ERROR` 去重） | `284` | `286` |
| 用时 | `23:55` | `25:38` |

**收集数差 `-74`，四笔账加起来正好是 `-74`**：225 退役 `-124 / +12 = -112`、
`8a6554c7` `+2`、`52e6199b` `+16`、区间里别人那笔 `16b842d8` 新增
`test_action_ball_diagnostic_pool_checkpoint.py` `+20`。**没有一条测试是"消失了"的。**

**失败节点集合的对称差是 `6` 条，逐条单跑核过，`6` 条全部与本轮三条改动无关**：

- 只在 before 红的 `2` 条（`test_action_ball_stage_supervisor.py` /
  `test_launch_kit_training_locked.py` 各一条），在 after 树上单跑 `2 passed`；
- 只在 after 红的 `4` 条里，`test_run_phase1_q50_persistent_supervisor.py`、
  `tests/test_action_ball_runtime.py`、`test_canonical_motion_compile_cli.py`
  三条单跑全绿（第三条连跑 3 次全绿）；
- 剩下 `tests/test_motion_backhand_loop_b_table_net_clearance.py::test_l1_certificate_path_swap_cannot_change_consumed_bytes`
  **是一条真·不确定性测试，两棵树上都会随机红**（after 树连跑 6 次 `4` 红 `2` 绿，
  before 树连跑 3 次 `1` 红 `2` 绿）。**它不是这次改动带来的**，
  但"证书路径换了不许改变被消费的字节"这种测试自己不确定，是个独立的待办。

这五条全是 supervisor / 活进程 / 原子发布 / 信号清理这一类**对并发和落盘时序敏感**的测试；
pod1 当时同时还有别的 workflow 在跑套件，`load average` 一度到 `78`。

**另外补一次"黑掉的那一面"的对拍。** 因为二.1 那个 venv 缺包问题，
课程 launcher 那个模块在上面的对拍里几乎全被 skip 掉了，所以又用 `/usr/bin/python3`
单独对了一次：`before` `58 passed`、`after` `59 passed`，
**`+1` 恰好是新增的那条 N5 顺序哨兵测试**，没有一条旧测试因为这次改动变红。

##### 六、`_validate_reveal_bridge` 的接线方案，逐步查了一遍能不能照做

按交代本轮**不动** `action_ball_4096x5_prelong_gate.py`，只审方案。结论：**方案可执行**，
签名、插入点、返回值顺序三处都对得上，另有一处该改、一处该补。

- **插入点对得上**：生产方那一行就在 `reward_groups` 的**下一行**
  （`action_ball_prelong_semantics.py:956-957`），方案说"紧挨 `row.get("reward_groups")` 之后"，
  和生产方的字段顺序一致。
- **返回值顺序对得上**：`_validate_reveal_bridge` 的 `return` 在 `:1190`，
  三元组是 `(摘要, lifetime, authority)`；方案说"第二个值当下一轮的 `previous_lifetime`、
  第一轮的 authority 当之后的 `expected_authority`"，与实际顺序一致。
- **"不需要升版"这句成立**：`SEMANTIC_SCHEMA_VERSION` 解析到 `3`，
  `_ordered_updates`（`:178-195`）逐行要求 `schema_version == 3`，现役 gate 收的本来就是 v3 行。
- **该改的一处**：方案 2(a) 要求"接线前先拿一格已落盘的 `scale4096` 收据实测 `status`"——
  **没有一格跑完过，没有收据可读**。静态答案更强且今天就能拿到：
  `require_bridge_telemetry` 默认 `True`，**全仓只有三处传 `False`，全在测试里**；
  而 `_prelong_bridge_authority` 在 `required=True` 时**只会抛异常，不会返回 `None`**。
  所以生产上 `not_configured` 产不出来。已在 §5.6.13 (A) 就地更正。
- **该补的一处**：方案说这条改动"零 lineage 代价"，对 `solver_profile_sha256` 成立
  （那道门不在被哈希的 5(+2) 个源文件里），但
  **`action_ball_4096x5_prelong_gate.py` 本身是被两个 211 launcher 钉住的运行时源**
  （`launch_action_ball_a211_four_arm_diagnostic.py:418` `PRELONG_GATE_SOURCE`、
  C211 launcher `:470` 同名），所以改它**会挪动 launch claim 里的那个 SHA**。
  这不是问题，是必须在同一个 commit 里连生产方 SHA 一起钉的理由——方案末尾其实
  已经说了"producer/consumer 必须同版本"，这里只是把"为什么"补上。

##### 七、这一节没做什么

1. **没有接 `_validate_reveal_bridge`**：按本轮交代该文件不动。方案已审完可照做，见六。
   （**2026-08-07 就地更正：已照该方案接线，见 §5.6.22。**）
2. ~~**没有接 `assert_known_generation`**：新发现，属于 exact-resume/broker 车道，见二.2。~~
   **2026-08-07 就地更正：已判决——不是"接线"而是"删除"，理由是它比同一条链上
   已经在跑的 `assert_consumed_birth` 粗一个档次，接上去是降级。见 §9.2.13。**
3. **没有往共用 venv 里装 `cryptography`**：会影响别的 workflow 正在跑的东西，见二.1。
4. **没有替 canonical 车道那两条零调用点的门下判决**：仍是"只登记不判决"，
   需要有人带着 canonical 动作库那条车道的上下文来判。
5. **没有修那条会随机红的测试**：
   `tests/test_motion_backhand_loop_b_table_net_clearance.py::test_l1_certificate_path_swap_cannot_change_consumed_bytes`
   在两棵树上都随机红（见五）。它和本轮三条改动无关，但它自己不确定这件事要单独查——
   一条"字节不许变"的证书测试自己不稳定，等于这条判据平时是靠运气过的。

#### 5.6.19 `scale4096` 第一次存盘必死的第四个出口，以及它后面那道从没跑到过的验收门（2026-08-07）

**人话一句：** 四格 `scale4096` 还是跑完 update 0 就在**第一次存 checkpoint** 那一刻死掉，
但报的已经不是 §5.6.15 那句话了。这次是 **sampler 自己**拿一本它这个模式**故意一行都不写**
的账（逐样本 `sample -> birth` 的 assignment）去核对一个正常增长的 `sample_count`。
**同一个病的第四个出口，错的还是对账范围。** 修完之后崩溃点往后挪了一格，露出第二个
真 bug（一个比生产方更粗的收据解码器）；再修完，训练**五个 update 全部跑完、五个
checkpoint 全部落盘**，剩下的 `SCALE4096_EXIT=2` 已经是**发射器终局验收门自己的三处口径
不一致**，不再是崩溃。

**现场（`c0_scale4096_s12r1/run.log:881`，C1 逐字相同）。**

```
runner.save -> _checkpoint_infos -> _build_exact_resume_state
  -> _capture_environment_resume_state -> _action_ball_exact_resume_state_dict
  -> broker.state_dict() -> _callback_states() -> provider.state_dict()
  -> _action_ball_solver_mutable_state_dict   ("sampler": ...state_dict())
  -> action_ball_sampling.py:7089
RuntimeError: sample authority ledger is inconsistent with retired/sample counts
```

`save_interval=1` 是四格预算（`SCALE_BUDGET`）写死的，所以**每个诊断格必死，不是 flake**。

**为什么说还是范围错。** `ActionBallSampler.sample()` 末尾原文就写着
`if not self._diagnostic_fast_path:` 才 `sample_birth_indices.append(...)` 并推
`assignment_head` 哈希链；`forget_diagnostic_births` 又把出生表收成"只留还活着的那些"。
而 `_sample_count_by_action` 每个样本无条件 `+1`。同一个类里**发放时**那条同名检查
（`:5460`）本来就已经带着 `not self._diagnostic_fast_path` 前缀 —— **存盘时那条忘了带**。

**换成什么见证（等强，不是删检查）。** 这个模式自己维护着另一本能证伪 `sample_count` 的账：
每次出生恰好吃 `DRAWS_PER_BIRTH=3` 个随机数、每个样本恰好吃 `DRAWS_PER_SAMPLE=18` 个
（两处在发放时就断言），而诊断退休**按定义不动 RNG、不动 retired 前缀**。所以 per-action 的
`draw_count` 仍然把两个计数器钉死：多一个样本差 18，少一个也差 18。`load_state_dict`
审精确档状态包用的**就是这同一条格子**（`per_action[uid] draw_count is inconsistent with
birth/sample counts`），不是新发明的判据。

同时把这一档自己声称的四件事写成会拒绝的检查：一行 assignment 都没有、一段 compaction
都没折、retired 前缀没动、留着的出生下标都在 `[0, birth_count)` 里。
（最后一条用上下界比较而不是 `set(range(birth_count))`：`birth_count` 会随整个诊断跑一路涨到
百万级，活的出生表却永远只有 `num_envs` 行。）**精确那两条严格对账一个字没动。**

**resume 语义一起做掉，两个方向 fail-closed。** 状态包新增一块**跟着签名一起落盘**的自陈牌子
`transcript_scope`（`exact_per_birth` / `diagnostic_live_births_only`）：

- 两种 scope 的状态包**互不相认**（错误里同时打印双方的值）；
- `diagnostic_live_births_only` **一旦发过样本就根本不能做精确续跑**——它按设计不含逐样本
  assignment；A211/C211 四格的发射合同本来就写着 `fresh_only`；
- 但**"只有出生、零样本"的不可变 fixed-view 状态仍然完整复原**（那条车道靠
  `_action_ball_restore_fixed_sampler_highwaters` 续跑，不能被误伤）；
- 没有这块牌子的老状态包按名字拒绝，理由写在错误里。

**修完之后崩溃往后挪一格，露出第二个真 bug。** 同一条链的下一站
`assert_issued_birth -> BaseBirthReceipt.from_identity_receipt` 报
`ValueError: birth sampling_stratum disagrees with mixture schedule`。

这是"**期望值是第三份手抄**"的老形状。`initial_center_single_question` 的合同写着
「32 条 curriculum arm 全是**精确 0** 时，物理支撑集就是 profile 中心那一个点」，所以生产方
`_sampling_plan_for_request` 的 initial-center 分支对**每一个** proposal_index 都发
`("center", 全零 sampling levels, 无 frontier arm)`。两个独立收据解码器却直接拿
`mixture.stratum_for(index)` 对答案——比生产方粗一档。四格（DR-L0 全关 +
`action_ball_initial_center_single_question=true`）每 5 个 birth 里有 4 个必然对不上，
所以**这条路径在这个配置下从来没有通过过**，只是以前被更早的崩溃挡住了。

实测同一组零 level 上的两种生产方：

| `initial_center_single_question` | birth 0..4 的 stratum |
| --- | --- |
| `False` | `interior / center / interior / frontier(base_spawn_x_lower) / interior` |
| `True` | `center / center / center / center / center` |
| 排班表 `stratum_for(i)` | `interior / center / interior / frontier / interior` |

新增 `_point_support_stratum_collapse`：只承认这张收据**自己的数字**能证明的那一档——
`stratum=="center"` **且**无 frontier arm **且** sampling levels 全零 **且** 32 条 domain level
精确为 0，四件事缺一不可。三个检查点（birth 解码、sample 里的 birth 档、sample 档）统一走它。
**为什么不是放宽：** 上表第一行那些收据没有一条能同时满足四件事，判决完全不变；`stratum`
本身已经进了 `birth_id`/`sample_id` 的规范身份哈希，几行之后就重算比对；签发证明还要再拿活的
transcript 逐字段对一遍。

**端到端收据（pod1 GPU1，四阶段全新跑）。**

| 轮次 | commit | materialize | recipe | oracle32 | scale4096 |
| --- | --- | --- | --- | --- | --- |
| `s12r1`（修之前） | `64036cb1` | `0` | `0` | `0` | `2`——sampler 对账，**0 个 checkpoint** |
| `s13r1`（只修第一处） | `abcdaf12` | `0` | `0` | `0` | `2`——挪到 `from_identity_receipt`，**0 个 checkpoint** |
| `s14r1`（两处都修，C0 与 C1） | `aae61869` | `0` | `0` | `0` | `2`——**5 个 update 全跑完、`model_0..4.pt` 全部落盘、`run.log` 零 traceback、`terminal_kind=clean_completion` / `completion_exit_code=0`**，死在发射器的终局验收门 |

**`s13r1` 这一格正好复刻了上一轮那句教训：只修一处会把崩溃往后推。** 所以这次两处一起修，
并把第三处也写清楚——但**没有替发射合同做判断题**。

**剩下的 `SCALE4096_EXIT=2` 是什么（交接，不在本轮修）。**
`REFUSED: C211 scale4096 exact checkpoint is missing`，来自
`launch_action_ball_c211_diagnostic.py:_audit_scale4096_terminal`（A211 launcher `:2475` 同形）。
它对终局 checkpoint 有**三条**期望，而**同一个发射器自己命令 `train.py` 产出的东西一条都不满足**：

| 验收门要的 | 实跑真的产出 | 出处 |
| --- | --- | --- |
| 文件名 `model_{BUDGETS["scale4096"][1]}.pt` = `model_5.pt` | 只有 `model_0..4.pt` | rsl_rl `on_policy_runner.py:270` 循环内按 `it` 存（`it∈[0,5)`），`:285` 收尾再用 `current_learning_iteration`＝`4` 存一次，**`model_5.pt` 不存在** |
| `checkpoint["iter"] == 5` | 实读 `model_4.pt`，`iter == 4` | 同上 |
| `infos["training_launch_claim_sha256"] == launch_claim_sha256` | `infos` 里**没有这个键**（只有 `hope_exact_resume_state` / `training_contract_*`） | `training_argv` 里没有 `++training_launch_claim_sha256=`；全仓唯一会加这个 override 的是 `action_ball_exact_resume_verifier.py:897` |

三条都不是"跑得不对"，是**验收门与生产方的口径从来没对过**（这道门此前从没被跑到过）。
收口有三种可能方式（文件名 `-1`／让 runner 补发一份 `model_N.pt`／让发射器把 claim override
加进 `training_argv`），**每一种都改 `launch_action_ball_*_diagnostic.py` 的字节，而这个文件是
A211/C211 两个 launcher 都钉住的运行时源（`LAUNCHER_SOURCE`），改它会挪动两族的 launch claim
SHA。属于 Franco 的判断题，本轮不背着发射的人做。**

**变异测试（十种"粗一个档次"的改法，各自杀掉指定用例）。**

`tests/test_action_ball_sampler_transcript_scope.py`（25 例）：

| 把守卫改粗成 | 必须变红的用例 | 实测 |
| --- | --- | --- |
| 删掉范围守卫（`_diagnostic_fast_path` 恒 `False`） | `..._live_births_only_run_can_serialize_and_says_so` 等 | `14 failed` |
| 停掉替换后的随机带对账 | `..._sample_counter_drift_is_still_refused[±1]` 等 | `3 failed` |
| 牌子只查"是不是已知值" | `..._a_known_but_wrong_scope_value_is_not_enough` 等 | `3 failed` |
| 去掉 live-only 的续跑拒绝 | `..._state_with_samples_cannot_be_resumed` | `1 failed` |
| 去掉"没牌子就拒绝" | `..._without_the_brand_is_refused_by_name` | `1 failed` |
| 去掉出生下标上下界 | `..._refuses_a_birth_row_it_never_issued` 等 | `2 failed` |
| 去掉"这一档不写 assignment" | `..._refuses_an_assignment_row_it_claims_not_to_write` | `1 failed` |

`tests/test_action_ball_point_support_stratum.py`（5 例）：

| 把守卫改粗成 | 必须变红的用例 | 实测 |
| --- | --- | --- |
| 让路永不生效 | `..._level_zero_initial_center_births_and_samples_decode` 等 | `2 failed` |
| 让路只看 stratum 不看物理事实 | `..._a_widened_run_may_not_claim_the_collapse` 等 | `2 failed` |
| 把"精确 0"放宽成"约等于 0" | `..._needs_every_one_of_its_four_facts` | `1 failed` |

源码复原后 30 例全绿。

**基线对拍（pod1 专用 worktree `samplerscope_20260807`，解释器
`/workspace/hope_isaac_venv/bin/python`，两批都是 `0 skipped`）。**

| 批次 | 基线 `20303a3d` | 本轮 `aae61869` | 判定 |
| --- | --- | --- | --- |
| sampler / runtime / curriculum / bank / adapter / transcript-scope 六模块 | `6 failed / 276 passed` | `6 failed / 306 passed`（+30＝本轮新增用例） | 失败集合逐 node-id 相同 |
| launcher / materializer 六模块 | `15 failed / 142 passed` | 同样 15 条，`diff` 为空 | 零回归 |

两批的既有失败都是**本轮之前就红的**（curriculum 那 6 条是
`profile_order must contain unique ActionProfileKey values`；另 15 条是这棵 bare worktree
缺 assets 造成的收集期失败），与本轮改动无关。

**没做的、交接给下一个人的两件事。**

1. **上面那道终局验收门的三条口径**（见表），需要发射合同的主人来定。
2. **`_sampling_plan_for_request` 还有第二个塌缩分支**：level 不为零、但该 scope 的活跃 arm
   物理宽度都是 `0`，生产方同样发 `center`。那一档收据**自己证不出来**（需要 profile 宽度），
   所以解码器仍会拒——**是拒绝不是放行**，本轮留档不改。

**跑 Isaac 链的 worktree 和跑 pytest 的 worktree 全程分开**（`pinv3_isaac_20260807` /
`samplerscope_20260807`），沿用 §5.6.15 末尾那条教训。

#### 5.6.20 三方对账：单位先对齐，然后"差 25~46 倍"这个数就不存在了（2026-08-07）

**人话一句：** 触发本轮的那句"我们的罚/收入比 `14.3`，比 `build_1` 差 `25~46` 倍"，是**拿我们
第 5 个 update 去比 `build_1` 训练完的收敛值**。把单位对齐、再拿 `build_1` **同一个 iteration**
来比，差距是 **`1.5~2.4` 倍**；而且**每步罚金我们比它当时还轻**（我们 `-0.308/步`，
`830xw9hy` 在 iter 2~4 是 `-0.399/-0.391`）。**所以不要因为这个比值动任何权重。**
但对账过程中掉出来三件真的不一样的事，和四处尽调与 `build_1` 互相打架、需要 Franco 拍板的地方。

##### 一、单位：三方到底怎么换算（单位没对齐就下的结论一律作废）

共同口径 = **post-dt 每 env-step 收入** = `weight × raw × dt`，`dt = 0.02 s`。

| 来源 | 原始单位 | 换算到每步 | 怎么验的 |
| --- | --- | --- | --- |
| 我们 | `per_term_weighted_dt_sum`（98304 样本加权和） | `÷ 98304` | 逐项闭合误差 ≤`1.8e-07`；与 `run.log` 里原生 `Episode_Reward` 块算出的比值 `16.31` 对上 `16.48` |
| `build_1` | `Episode_Reward/x` | **`× episode_length_s ÷ Train/mean_episode_length`** | IsaacLab `managers/reward_manager.py:119-120`：`Episode_Reward = mean(episode_sums) / max_episode_length_s`，而 `:154` 的 `episode_sums += weight*raw*dt`。`build_1` 两跑的 `env_cfg.episode_length_s = 10`（wandb config 实读） |
| 尽调 | §16.3 S3 本来写的就是 post-dt 每步 | 不用换 | — |

**这一轮真有人踩了这个坑，必须写下来。** 中途一份对账把 `build_1` 每步算成
`Episode_Reward ÷ mean_episode_length`，**漏了 `× 10`**，于是 `build_1` 的所有每步剂量小了 10 倍，
得出"我们的正收入是 `build_1` 收敛态的 `1.9~2.0` 倍"。**真相是低 `5.2` 倍。**
凡引用过那一版每步数字的地方一律 `× 10`。

**比值 `|负|/正` 是口径不变量**（换算因子是所有项共享的同一个标量），所以 `14.3~16.5` 这个数
本身不用修正 —— 要修正的是**拿它跟谁比**。两边各自自证过：`build_1` 用 `Episode_Reward` 与
`Live/Reward` 两套单位得 `0.5543` / `0.5494`；我们用 `weighted_dt_sum` 与原生 `Episode_Reward`
得 `16.483` / `16.307`。

**第二处更隐蔽的单位坑：投影罚的 `raw` 已经乘过 5 了。** 现役是
`manager_weight=-1.0` + `params.objective_weight=-5.0`（weight-independent exposure 硬合同，
`train.py:5202-5257`；发射器 `launch_action_ball_c211_diagnostic.py:4130` 逐字校验；
`run.log:177` 自陈）。callable 返回的是 `Σ_j(1-exp(-4·d_j)) × 5`，所以日志里
`raw/样本 = 9.7005` 的**值域是 `[0,155]`，不是 `[0,31]`**：

- 真实的 `Σ_j(1-exp(-4·d_j))` = `9.7005 / 5` = **`1.94`**（满值 31）；
- 每步平均**至少 2 个关节**被投影（不是 11 个）；Markov 下界"至少 `6.3%` 的样本有投影"（不是 `32.9%`）；
- 该项处在自身渐近上限 `-3.1/步` 的 **`6.3%`**（不是 `33%`）。

##### 二、先答那条判据：`build_1` 早期也是负向压倒，而且比我们还狠

这条不查清就不许动权重。下表是我自己从 wandb history 按上面的换算重算的
（`BerkeleyPingPong/hope_wbc`，`dongc_1` 名下）：

| iter | `830xw9hy` 比值 | 正/步 | 负/步 | `i4dxpbwy` 比值 | 正/步 | 负/步 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `5.35` | `+0.0356` | `-0.1906` | `9.27` | `+0.0113` | `-0.1051` |
| 2 | **`7.12`**（峰） | `+0.0560` | `-0.3989` | `11.05` | `+0.0204` | `-0.2251` |
| 4 | `6.93` | `+0.0564` | `-0.3908` | `10.87` | `+0.0193` | `-0.2102` |
| 20 | `4.91` | `+0.0767` | `-0.3768` | `7.49` | `+0.0244` | `-0.1824` |
| 60 | `3.16` | `+0.0776` | `-0.2448` | `2.86` | `+0.0278` | `-0.0796` |
| 100 | `1.59` | `+0.0811` | `-0.1287` | **`1.01`**（首次 <1） | `+0.0322` | `-0.0325` |
| 133 | **`0.99`**（首次 <1） | `+0.0807` | `-0.0800` | `0.83` | `+0.0303` | `-0.0251` |
| 末 | `0.554`（3437） | `+0.0982` | `-0.0544` | `0.307`（21896） | `+0.0926` | `-0.0285` |

我们 C0 `scale4096_s15r1`：`u0 = 14.37 / +0.0206 / -0.2960`，`u4 = 16.48 / +0.0187 / -0.3085`
（C1 逐位可比：`16.484`）。

1. **`build_1` 头 100 个 iter 都是负向压倒 `5~12` 倍**；`i4dxpbwy` 到 iter 100、`830xw9hy` 到
   iter 133 才第一次跌破 `1.0`。
2. **我们 5 个 update 完整落在 `build_1` 自己那段"看起来很糟"的窗口里。** 同 iter 比，
   `16.48` vs `6.93`/`10.87` = **`1.5~2.4` 倍**，不是 `25~46` 倍。
3. **每步罚金我们不是最重的那个**：我们 `-0.3085`，`830xw9hy` 在 iter 2~4 是 `-0.399`/`-0.391`。
4. **正收入我们和 `i4dxpbwy` 早期几乎一样**（`+0.0187` vs `+0.0193`）。
5. 顺带：`build_1` 两跑在 iter 40~60 平均 episode 掉到 **`5.0` 步**，然后照样恢复
   （与 §5.6.4 一致）。我们现在按 `1.82%` 每步终止率反推是 ~`55` 步，**不在崩溃态**。

**判据答案：比值本身不构成动权重的理由。**

##### 三、同口径三方对照（全部 post-dt 每步）

| 轴 | 尽调 §16.3 S3/S6 | `830xw9hy`@4 | `i4dxpbwy`@4 | `830` 收敛 | `i4dx` 收敛 | 我们 @u4 | 判定 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 正收入合计 | `+0.05~+0.10` | `+0.0564` | `+0.0193` | `+0.0982` | `+0.0926` | **`+0.0187`** | 收敛态**三方一致**；我们与 `i4dx` 早期一致 |
| 负收入合计 | 常态 ≤`-0.03`／峰值 ≤`-0.30` | `-0.3908` | `-0.2102` | `-0.0544` | `-0.0285` | **`-0.3085`** | **早期 `build_1` 同样破 S3** → S3② 是收敛态判据 |
| `\|负\|/正` | 隐含 `0.30~0.60` | `6.93` | `10.87` | `0.554` | `0.307` | **`16.48`** | 收敛目标**三方一致**；早期我们 `1.5~2.4×` |
| qdes 轴合计 | S6 权重 `-10` | `-0.0635` | `-0.0616` | `-0.0041` | `-0.0022` | **`-0.2333`** | **我们 `3.5~3.8×` `build_1` 历史峰值** |
| qdes 占罚金 | 无判据 | `16.2%` | `29.3%` | `7.6%` | `7.7%` | **`75.6%`** | **份额倒挂** |
| `action_rate` | `-0.1` 是四家共识 | `-0.1262` | `-0.1264` | `-0.0241` | `-0.0216` | **`-0.0360`** | **我们比 `build_1` 早期轻 `3.5×`** |
| `joint_limit` | S6 `-10` | `-0.0032` | `-0.0029` | `-0.0006` | `-0.0002` | **`-0.0172`** | 权重只有一半，剂量却 `5.4~5.9×` |

##### 四、三类分档

**（甲）三方一致，没问题：**

1. **收敛态 `|负|/正` 目标 `0.30~0.60`。** 尽调从 dense 锚 `+0.05~+0.10` 与"单步罚常态总和
   ≤`0.3×` 锚"推出 `0.30~0.60`；`build_1` 实测 `0.307`/`0.554`。
   **研究结论与唯一已知能击球的实现独立吻合 —— 这是本轮最强的一条证据。**
2. **dense 正收入锚 `+0.05~+0.10/步`。** `build_1` 收敛 `+0.0926`/`+0.0982`，落在处方上沿。
3. **`action_rate` 量级。** 尽调说 `-0.1` 是四家模仿系共识；`build_1` 就是 `-0.1`；
   我们 `-0.2` 但带 clamp，实际付得比 `build_1` 早期还少 `3.5` 倍。
4. **参考轨迹不贴限位。** 尽调 `design_audit:179-180` 的工单第 3 步要求先查参考直方图；
   已查并排除（31 关节 × 57 帧最小余量 `0.116 rad`，零越限，§5.6.2 表与 `EXP:733`）。

**（乙）我们与两方都不同，是我们错了：**

1. **qdes 轴的份额倒挂。** 我们 `75.6%`，`build_1` 早期 `16~29%`、收敛 `7.6%`。
   绝对值上我们 `-0.2333/步` 是 `build_1` 历史最高点 `-0.0667` 的 **`3.5` 倍**。
2. **权重上我们在这条轴上比 `build_1` 重 `7.7` 倍，而且多一条。**
   `build_1`：`rally_joint_qdes_saturation = -0.65` + `rally_ankle_qdes_saturation = -0.3`；
   我们：投影 `-5.0` + barrier `-5.0`。
   > **就地更正（2026-08-07，§5.6.24 取证）：原文这里写的"只在 rally 窗内付款"是错的。**
   > 那是从项名推出来的强推断，不是读过的源码。`build_1` 的 `rally_joint_qdes_saturation`
   > 实际绑定的 callable 是 `rally_all_joint_qdes_barrier`，函数体里 `time_to_strike`
   > 一次都没出现、`return debt` 没乘任何 gate —— **它和我们一样是全相位付款**。
   > 真正有窗的只有 2 个踝关节那条（占它 qdes 轴 ~5%）。逐字取证见 §5.6.24。
3. **`joint_limit` 同名不同形状。** 我们权重是 `build_1` 的一半（`-5` vs `-10`），
   实付剂量却是它的 `5.4~5.9` 倍 —— 因为我们有 `penalty_floor=0.25` 非零地板，
   `build_1` 与三个外部库都是**纯越界尾巴**（限位没越就恒为 0）。尽调 §16.2 骂的就是这个，
   现在有实测了。
4. **`action_rate_clamped` 的 `raw` 逐位恒等于 clamp 上限 `9.000000`**（两格 × 5 更共 10 格全同），
   该项对动作的导数恒为 0。**而这条轴正是 `build_1` 恢复的引擎**：它占 `build_1` 早期罚金
   `32~60%`，每步从 `-0.126` 一路降到 `-0.022`，这个衰减本身就是它比值穿过 `1.0` 的原因。
   **我们把那台引擎焊死了。**（机制与是否 bug 归另一条线查，本节只记账：它占我们罚金 `11.7%`，
   在它被处理之前，我们负收入的成分诊断要重跑。）
5. **收入分层塌成一层**：模仿 `99.45%` / 击中 `0.55%` / 质量 `0.00%`。
   **但要注意**：`build_1` 在 iter 4 的击球类收入也是 `0`，**这一条是收敛态差异、不是开局差异**，
   现在不能算我们错。

**（丙）尽调与 `build_1` 互相打架，需要 Franco 拍板：**

1. **投影距离该不该收费。** 尽调 `design_audit:172` 说"投影 + 罚 pre-clamp 超出量"是有文献背书
   的那条（Chou ICML'17；Fujita & Maeda CAPG ICML'18），并明确无主流栈对指令级越界 reset；
   `build_1` **测的就是同一个量**（`Instrumentation/qdes_safety/qdes_projection_distance_raw_rms
   = 0.0253`、`qdes_clamp_fraction = 0.0823`），**收费为零**，
   而且 `Episode_Termination/actual_q_hard_limit_audit = 0`。我们两条罚合起来 `75.6%`。
   > **就地更正（2026-08-07）：原文"只罚 rally 窗内的 saturation"是错的**，同上条。
   > `build_1` 的 all-joint qdes 罚**没有任何相位门控**。
2. **qdes 轴的剂量差 15 倍。** 尽调 S6 要 `-10`；`build_1` 实跑 `-0.65`。
   我们取 `-5` 恰好在两者中间。
   > **就地更正（2026-08-07）：原文那个"先决问题"不存在。** 读过源码后确认
   > `build_1` 主项全相位付款，所以**没有 rally 窗门控可抄**；差的是核函数形状、
   > 聚合方式与量纲，不是占空比。Franco 2026-08-07 裁定二已按开源形状定案，见 §5.6.24。
3. **"违规驻留 <`0.5%`"这条线量的到底是哪个量。** 尽调 `design_audit:174` 引 CaT
   （arXiv:2403.18765）。按**实际 q 越硬限**读，`build_1` = `2.68e-06`（**远远达标**）；
   按**指令级钳位占比**读，`build_1` = `7.6%~8.2%`（**差 15 倍**）。
   两个量差三万倍，这条线不写清口径就没法当门。**我们这两个数一个都没有** ——
   机制层算好了但没往外吐（见下）。
4. **death penalty 存废。** 尽调 S4 要 `-300`（post-dt `-6.0`）；三个外部库与 `build_1`
   **全都没有这一项**（`build_1` 的 54 项里确认无此项）；现役 `-10`（`-0.2`/事件，
   实测 `-0.00365/步`，占罚金 `1.2%`）。已经是主动偏离，但没有正式裁定记录。

##### 五、当初的 scale 对齐到底怎么错的（三层查证）

**结论：权重移植没错，现役 argv 没错，错在分母。**

| 层 | 查到什么 |
| --- | --- |
| **权重常量** | `hope_env_cfg.py:2380-2386` 的剂量注释算到"`-0.033`/受影响关节/步、渐近 `-0.10`/关节/步"，然后直接下结论"cannot easily swamp early imitation/hit income" —— **这句话后面没有分母**。`HOPEPingPongActionBall.yaml:160-176` 的 barrier 注释**有**分母，但用的是**击球窗内峰值** `0.575/步`。`train.py:3066-3068`、`action_ball_211_four_grid_contract.py:343-351` 只是把同一组数钉住，没有再算一次账。 |
| **实验史裁定** | §5.6 抬头写着 Franco `2026-08-05` 裁定：**参照系只有两个 —— §5.3 的折扣 per-swing 账，或 `build_1`**。提交 `635252f6` 就是按第一个参照系做的：`upright_exp 1.0→0.25`、`hit_unstable_support -10→-1.0`、`death_penalty -300→-10`，三项**都换算成折扣 per-swing 收入再比**，这一步是对的。**但同一批里没有任何一条 dense 每步罚项进过那张表** —— §5.3 的层级表从头到尾只有收入没有罚（已就地更正）。 |
| **现役 argv** | `launch_action_ball_c211_diagnostic.py:3724-3729` 发的是 `death_penalty=-10` / `qdes_limit_barrier=-5` / `+qdes_projection_penalty=-5` / `joint_limit=-5`，与配置逐字一致，运行时落成 `manager -1.0 × objective -5.0`。**argv 没有偷偷改剂量。** |

三个候选的判定：

- **(i) 量纲搞错 —— 成立过一次，已修。** `qdes_limit_barrier` 带宽 `0.08` 比护栏自己的投影内沿
  `0.05` 宽，任何被钳关节构造性恒扣 `-0.0844`/关节/步（§5.6.2 第 9 条，`2026-08-04` 与
  `2026-08-06` 两次才修完两条通道）。
- **(ii) 对齐了权重、没对齐触发率 —— 成立，但形状更具体：是分母拿错了。** 两个因子：
  - **分母口径**：barrier 那条注释拿"击球窗内、误差为零时"的峰值收入 `0.575/步` 当分母。
    实测全相位平均正收入是 **`+0.0187/步`**，**乐观 `31` 倍**。
  - **占空比**：罚项每步都付（一回合 `500` 步），窗内收入只付 `3~11` 步。
    同一句话按回合算：`3` 个关节被钳 = `-0.253/步 × 500 步 = -126.6`，而窗内收入
    `0.575/步 × 11 步 = +6.3` —— 注释写的是"吃掉 `44%`"，真值是 **`2000%`**，
    **低估 `45` 倍**，正好是 `500/11` 这个占空比。
  - 换算回 §5.3 自己的折扣单位：当前 dense 罚合计 **`-13.0`~`-30.6`/回合**，
    而 §5.3 那张表的天花板（合法上台下界）只有 **`+3.332`**。
- **(iii) 两边根本不是同一个东西 —— 部分成立，三处**：我们的 `qdes_projection_penalty`
  在 `build_1` 里**根本不是 reward，是仪表**；我们的 `joint_limit` 有非零地板、
  `build_1` 的是纯尾巴；我们的 `upright_exp` 是正收入、`build_1` 的 `upright` 是罚（`-1.0`）。

一句话：**权重对上了，触发率没人量，分母还拿错了。** 这正是 MEMORY 里那条
「对齐经济不是对齐权重」的形状 —— 只不过这次连"经济"的分母都换错了单位。

##### 六、分档建议（**本节没有改任何 reward 权重**）

**（A）现在就该做，零激励改动、零风险：**

1. **把机制层已经每步在算、但没往外吐的 qdes 计数器接进 economy JSON。**
   `hope_rewards.py` 里 `projection_sample_count` / `projection_joint_count` /
   `..._distance_sum` / `..._max_distance` / `intrusion_sample_count` /
   `intrusion_joint_count` / `max_intrusion` 全都已定义并每步累加，**`run.log` 里零命中**。
   零新增计算。**理由：`qdes_limit_barrier` 现在是"`38%` 样本各罚 1 个关节"还是
   "`1.2%` 样本各罚 31 个"完全判不出来，而这两种情况的处置完全相反。**
   这也是（丙）3 那条门能不能重定的前提。
2. **给 29 项统一加一个 `per_term_nonzero_sample_count`**（每项一次 `(values != 0).sum()`）。
   现在 `per_term_eligible_denominator` 对 29 项恒等于 `98304`，自陈语义
   `all_rollout_environment_samples_including_gated_zero` 就写着它**故意不区分**
   "被门控成 0"和"激活但为 0"，对触发率零信息量。
3. **给三处 "reward economy" 门补一条读实际经济的检查。** 现在
   `action_ball_211_four_grid_prelong_barrier.py:305/727/919`、
   `materialize_action_ball_reward_ppo_economy_receipt.py`、
   `launch_n1_vendor_baseline_diagnostic.py:486` **全是字节漂移门**，只验"配置权重没被改过"。
   要读的是：`|负|/正`、per-term 每步 post-dt、qdes 轴占罚金比例。
   按 Franco 的准绳，**同批配变异测试**（"粗一档就过不了"）。
4. **把尽调 S3 的四条判据抄成一条仓内准绳条目。** S3 自己指定的落点就是"新增 docs 量级准绳条目"，
   `grep "dense 锚|折扣视野收入|量级准绳"` 全仓除尽调本身零命中；S1 要求写进 receipt 的六个数
   （per-term 占比 / raw return mean·std / explained_variance / clip fraction / grad norm /
   rsl_rl 版本）同样查不到落点。
5. §5.3 与 §5.4 Gate 6 已在本轮就地更正。

**（B）等长跑数据再定：**

1. **任何 qdes / barrier / `action_rate` 的权重改动。** 判据已经查清：`build_1` 早期同样负向压倒，
   我们的每步罚金甚至比 `830xw9hy` 当时还轻。**5 个 update 读不出趋势**
   （我们 `14.37→16.48` 是 `+15%`，`build_1` 同期 `12.10→10.87` 是 `-10%`，方向不同但样本太小）。
2. **决定性实验：一格跑到 ~`150` update，看 `|负|/正` 有没有穿过 `1.0`。**
   `build_1` 在 iter `100`/`133` 穿过。这是唯一能把"配置病"和"正常早期"分开的实验。
   同时报告 qdes 轴每步剂量的轨迹 —— `build_1` 那条在 iter 20→100 之间自己降了 `15~25` 倍
   （`-0.0654 → -0.0040`），我们 5 更只动了 `-5%`。
3. 正收入 `+0.0187/步` 目前**不构成异常**（`i4dxpbwy` 同期 `+0.0193`），
   但收敛后必须落进 `+0.05~+0.10`。这是长跑的验收线，不是现在的问题。

**（C）需要 Franco 拍板的四条**：即上面（丙）的四项 —— 投影距离收不收费、qdes 轴剂量取谁
（先决：要不要 rally 窗门控）、"违规驻留 `0.5%`"量哪个量、death penalty 存废。
按 Franco「不合理的门就改」的准绳，第 3 条那道线目前既没被任何门读、口径也没写清；
但**改线要连证据一起改**，证据得先有 (A)1。

##### 七、这一节不代签什么

1. ~~**`build_1` 的 reward 源码我们从来没读过。**~~ **已作废（2026-08-07）：源码已逐字读过，
   那条强推断被证伪。** `rally_joint_qdes_saturation` 绑的是 `rally_all_joint_qdes_barrier`，
   **无门控、全相位**；"门控差异是我们 3.5 倍的头号解释"这个假设**不成立**。
   真正的解释是核函数形状（饱和 vs 线性尾）+ 聚合（SUM vs top4 混合）+ 量纲。取证见 §5.6.24。
2. **5 个 update 不能读趋势**（见 B1）。
3. **`i4dxpbwy` 声明 54 项却只记了 20 项**，它的正收入合计可能被低估，比值是上界；
   `830xw9hy` 记全 54 项，是更安全的参照。两跑配置只差一个权重
   （`post_strike_leg_quiet` `0` vs `-0.04`），却一个 `48%` 摔倒结束、一个 `98.5%` 超时结束，
   **不要当成同一配方的两个样本。**
4. **`action_rate_clamped` 吐常数** 本节只做了记账（占罚金 `11.7%`、该项零梯度），
   没有查机制，归另一条线。它被处理之前，我们负收入的成分诊断要重跑。
5. **跨仓项名配对有一半是按语义猜的**，不是官方对应表。可信的只有 qdes 系、`action_rate`、
   `joint_torques`、`joint_limit`、`joint_vel`、`base_ang_vel_xy`、`base_lin_vel_z`、
   `motion_body_*`；`racket_*` 三项我们用 Cauchy 核 + 权重 `4.0`，`build_1` 用指数核 + `14.0`，
   **不可直接对**；`upright_exp`（我们，正）对 `upright`（`build_1`，`-1.0` 罚）**符号约定就相反**。
6. **本节没有改任何 reward 权重、任何门、任何配置。**

##### 八、收据

- 我们的实测（两格 × 5 update，逐项闭合验过）：pod1
  `/workspace/franco/s11_cells_c_20260807/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_action_ball_c211_diagnostic/c{0,1}_scale4096_s15r1/run.log`；
  提取器 `/workspace/franco/recon3_20260807.py`（解释器 `/workspace/hope_isaac_venv/bin/python`）。
- `build_1` 两跑的 wandb 全量 history 与 config：
  `/workspace/franco/b1_econ_20260807/{830xw9hy,i4dxpbwy}.{hist,meta}.json`；
  本节的每步换算与逐 iter 表由 `/workspace/franco/b1_recon{2,3,4,5}_20260807.py` 重算，
  **不是转述**（`830xw9hy` 3438 行 / 54 项，`i4dxpbwy` 21897 行 / 20 项）。
- 换算公式的出处：IsaacLab `source/isaaclab/isaaclab/managers/reward_manager.py:119-120` 与 `:154`。
- 尽调判据：`docs/research/dr_reward_external_diligence_20260731.md` §16.3 S3（`:1566`）、
  S4（`:1571`）、S6（`:1581`）、S7（`:1586`）；触发率验收线
  `docs/research/design_audit_and_speedup_20260729.md:172-180`。

#### 5.6.21 §5.6.19 交接的那道门：checkpoint 与 claim 两条口径修完,露出第四条（2026-08-07 落地并实跑验证）

**人话一句：** §5.6.19 把 `SCALE4096_EXIT=2` 的三条"验收门与生产方口径从来没对过"列成判断题
交了出去。本节把其中两条做完：**修法是"把门瞄准",不是删门也不是调松**——严格程度一个字没动,
只是原来指着一个**在任何预算下都不存在的文件**。实跑证明这两条真的过了：拒收理由从
`exact checkpoint is missing` 变成了**它后面那道检查**的理由。同时露出第四条口径不一致
（发射器要一族这个诊断跑从不发射的遥测),本节**只定位、不修**,理由见末尾。

**D1（结构性差一格,A/C 同病）。** RSL-RL `OnPolicyRunner.learn` 的
`for it in range(start_iter, tot_iter)` 在**循环体内**做
`self.current_learning_iteration = it`（venv 里那份 `rsl_rl_lib 2.3.1` 的 `:264`,
仓内 vendored 那份 `:295` 同形）,循环结束后的收尾存盘（`:286` / `:321`）用的就是那个末值。
所以跑满 N 个 update 落盘的是 `model_0..model_{N-1}.pt`,**`model_N.pt` 在任何预算下都不存在**。
这不是"预算写错了",是**门指错了对象**：`expected_updates` 是预算,`expected_updates - 1` 才是
末位编号。实测 `torch.load(model_4.pt)["iter"] == 4`。

修法（三处消费方,一处出处）：

| 位置 | 修前 | 修后 |
| --- | --- | --- |
| `action_ball_4096x5_prelong_gate.py` | 无 | 新增 `TERMINAL_CHECKPOINT_ITERATION = EXPECTED_UPDATES - 1` / `TERMINAL_CHECKPOINT_FILENAME`,**两族唯一出处** |
| `launch_action_ball_{a211,c211}_*.py` | 各自手抄 `model_%d.pt % expected_updates`、`iter != expected_updates` | 走 `_terminal_checkpoint_iteration()`：从共享出处取,并当场核对"这个编号确实是自己预算的末位、文件名与编号没脱钩",对不上 `LaunchRefused` |
| `action_ball_211_four_grid_prelong_barrier.py` | `_audit_cell:815` 手抄 `!= 5`,聚合字段名 `model_5`,收据里两个 `5` | 全部走 `TERMINAL_MODEL_ITERATION`;字段改名 `terminal_model`,kind/schema 升 `v4 → v5` |

`_audit_cell:815` 那处是**第三份手抄**,`s15r1` 之前没有任何测试碰过它（现存测试全部从
`_audits()` 直接造行,绕过 `_audit_cell`）。升 kind 的代价为零：这道门从没通过过,
**世上不存在任何一份 v4 聚合收据**。

**A/C 一起修的理由。** Franco 2026-08-07 的口径是"A 和 C 除了 obs 和 reward 之外应当处处相同"。
这个差一格两族逐字同形,只修 C 就是亲手造出他正在担心的那种不一致。

**D2（checkpoint 里根本没有 launch claim）——缺的接线不在发射器,在 trainer。**
§5.6.19 的表格第三行给了三种可能收口方式,其中"让发射器把 claim override 加进 `training_argv`"
**走不通**,两个独立理由：

1. `train.py` 自己明文禁止诊断跑消费正式 launch claim
   （`diagnostic ActionBall training cannot consume a formal launch-claim action-set identity`）,
   而 `training_launch_claim_path` 与 `training_launch_claim_sha256` 必须成对出现;
2. claim 的 canonical payload 里**就有 `training_argv`**,把 claim 塞进 argv = 让 argv 的哈希
   包含 argv 自己的哈希,**自指循环**。

真实情况是：两个发射器**早就**在 exec 那一刻把 claim 放进环境变量
`HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256`（C211 `:4454` / A211 `:4513`,正是为了绕开上面第 2 条）,
而 `train.py` 只在**同时**配了 `n1_vendor_diagnostic_stage` 和
`vendor_runtime_training_contract_sha256` 时才去读它——诊断跑这两个 key 一个都发不了。
于是 runner 的 `training_launch_claim_sha256` 恒为 `None`,`my_on_policy_runner.py:2435` 的
`is not None` 从不成立,infos 里永远没有那个键。

所以补的是 trainer 的**准入**：`target_mode=action_ball` 且
`action_ball_diagnostic_unauthorized=true` 时也读 exec 边界那个值。**仍然不是"看见环境变量就信"**：
不满足准入时那个环境变量连读都不读（别处残留的同名变量不能炸掉无关的正式跑）;两边都在
且不一致当场炸;准入位必须是**真 bool**。这条只多做一件事——让 checkpoint 自陈它是哪一次
发射产出的;它不解锁任何正式路径（frozen-eval identity / runtime bootstrap receipt 由
`diagnostic_unauthorized` 单独把门,与这个值无关,`_validated_runtime_bootstrap_binding`
对诊断跑照旧返回 `{}`）。

**测试：这道门此前零测试,所以是从零写。**
新增 `tests/test_action_ball_4096x5_terminal_index.py`（19 例)。它**不手抄那个 4**：
读**能找到的每一份** RSL-RL `on_policy_runner.py` 活源码（pod 上是 venv 里装的
`rsl_rl_lib 2.3.1`,host 上是仓内 vendored 那份;一份都看不到时 fail closed）,用 AST 核对
四件事——起点从 0、赋值在循环体内、循环外没有第二次赋值、收尾存盘用的就是这个属性——
然后**真的跑一遍**同形状的 `for it in range(0, N)` 取末值。另核对 A/C 两族同一个答案、
决定终局编号的那份源码在两边的 `RUNTIME_SOURCE_PATHS` 钉子表里、以及 claim 没有进 argv。

**变异证据（pod1,`/workspace/hope_isaac_venv/bin/python`,worktree
`/workspace/franco/gate_offby1_20260807`,均为真文件改动后重跑）。**

该拦的仍拦（7/7 被杀）：

| 变异 | 实测 |
| --- | --- |
| M1 共享常量退回 `EXPECTED_UPDATES` | import 期 fail closed（barrier 的自检);把那条自检也去掉后 → `test_shared_gate_constants_match_the_live_rsl_rl_convention` 等 **4 failed** |
| M2 C211 重新手抄 `model_%d.pt % expected_updates` | C211 launcher 模块 **1 failed** |
| M3 A211 重新比 `iter == expected_updates` | A211 launcher 模块 **1 failed** |
| M4 barrier 终局编号退回手抄 `5` | barrier 模块 **红** |
| M5 `train.py` 撤掉诊断 claim 接线（回到恒 `None`) | completion 模块 **1 failed** |
| M6 `train.py` 改成"看见环境变量就信" | completion 模块 **1 failed**（准入没被放宽） |
| M7 `train.py` 把 claim 布尔挪回消费之后 | completion 模块 **1 failed** |

pin 读的是活源码,不是手抄（4/4 被杀）：把 venv 里那份 `rsl_rl` 复制出来改成
"赋值挪出循环 / `it + 1` / 收尾存盘改用循环变量 / 循环后再赋一次",四种都被
`_terminal_iteration_from_live_source` 当场拒;未变异的活源码算出 **4**。

误拦的不再拦：`iter=4` 的正常产物通过;同一批里 `iter=5`（旧门自己要的数字）、
`iter=3`（少跑一格）、claim 缺失、claim 错配四种仍然 `LaunchRefused`。

**端到端收据（pod1 GPU1,四阶段全新跑,worktree `gate_offby1_20260807`）。**

| 轮次 | commit | materialize | recipe | oracle32 | scale4096 |
| --- | --- | --- | --- | --- | --- |
| `s15r1`（修之前,§5.6.19 的下一轮） | `92fa48a7` | `0` | `0` | `0` | `2`——`REFUSED: C211 scale4096 exact checkpoint is missing`,而 `model_0..4.pt` 其实全在 |
| `s16r1`（本节） | `a970a58a` | `0` | `0` | `0` | `2`——**理由换了**：`REFUSED: C211 scale4096 reward-safety counters lacks exactly 5 contiguous terminal updates` |

**"理由换了"就是本节两条修好的live 证据**：那两条检查排在 reward-safety 计数之前
（checkpoint 定位 → `weights_only` 加载 → `iter`/claim 绑定 → 各族计数),门走过去了才可能报后面的。
直接读产物再确认一遍（`/workspace/franco/verify_live_s16r1.py`,解释器同上):

```
claim               = 12b2f1519fdb4ac10d1da6c44ffe41317000b7c4b0d99ebf8315a994886f50a9
argv elements       = 71        argv 里出现 claim = 没有(无自哈希循环)
argv launch_claim 键 = 没有
checkpoints         = model_0.pt .. model_4.pt        (没有 model_5.pt)
model_4.pt  iter=4  claim_in_infos=True  matches_launch_claim=True
```

`iter=4` 正是修后这道门要的末位;`claim_in_infos=True` 是 `s15r1` 里**根本不存在的键**。
五份 checkpoint 每一份都带上了发射身份。

**回归账。** 直接相关的 9 个模块（prelong-gate / terminal-index / four-grid-barrier /
两个 launcher / n1-completion / ac-parity / shared-constants / isaac-four-grid）:
**`714 passed / 0 failed / 0 skipped`**。同一批在 `ffdca6af` 上是 `133 failed / 493 passed` ——
那是并行 workflow 的 frozen-term 门（`ba52baea`）落地时没跟着改 A211/C211 两个 launcher 测试的
economy 夹具（`motion` 写死 `1.0`,正好撞在新门上）。本轮顺手把那两处夹具改成逐 update 变化,
否则本节的门连正例都跑不起来。

**露出来的第四条口径不一致（本节只定位,不修 —— 交接）。**

`REFUSED: ... reward-safety counters lacks exactly 5 contiguous terminal updates` 来自
`launch_action_ball_c211_diagnostic.py:3022`（A211 `:2597` 同形）：它要 5 行
`HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=`（事件 `hope_reward_safety_transition_update`）。
实测 `s16r1` 的 `run.log` 里这个前缀 **0 行** —— 而且 `s15r1` 的两格逐字相同,**这与本节的改动无关**：

| 前缀 | s15r1 | s16r1 |
| --- | --- | --- |
| `HOPE_JOINT_SAFETY_UPDATE_JSON=` / `HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON=` / `HOPE_EXACT_BEHAVIOR_UPDATE_JSON=` / `HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_UPDATE_JSON=` / `HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS_UPDATE_JSON=` / `HOPE_POLICY_STD_UPDATE_JSON=` / `HOPE_PUSH_VELOCITY_DIAGNOSTIC_UPDATE_JSON=` | 各 5 | 各 5 |
| `HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=` 及同族的 `HOPE_EFFECTIVE_REWARD_*_UPDATE_JSON=` | **0** | **0** |

出处是同一个形状：`my_on_policy_runner.py:9007` 那三条只在
`prepared_reward_evidence is not None` 时才打印,而它由 `:6235` 的
`if reward_activation_ledger is not None:` 决定 —— 诊断跑没有这本账,所以整族遥测**从不发射**。
和 D2 一样是"门要的东西这条路本来就不产",但**判断题不一样**：D2 的答案明确（claim 本来就在
exec 边界躺着,只是没人读),这一条要先回答"**诊断跑到底该不该有 reward activation ledger**"——
该有就接线,不该有就把这道门的适用范围改成只管正式跑。**两种都要连记录与阻断一起改**,
所以本节不替发射的人做判断,只把证据摆在这里。

**还没关的洞（本轮,别当成已解决）。**

- **`scale4096` 仍然 `EXIT=2`**,只是理由前进了一格(见上)。`long4096` 本轮**没有发**。
- **A 族一格都没实跑 `scale4096`。** A211 的同名门与 C211 逐字同形、同批修、同批变异,
  但"A 族也能跑到终局验收"本轮**没有实跑证据**。
- `_audit_cell`（四格聚合的 per-cell 审计）**仍然零测试覆盖**,本轮只把它里面的手抄 `5` 换成
  共享常量;它的其他分支照旧只有 import 期的常量自检兜底。
- 本节只证明了这道门的**前两段**（终局 checkpoint 的身份、发射身份绑定）能被真实产物满足;
  "整道 `scale4096 → long4096` 门能通过"仍然**没有证据**。

#### 5.6.22 `_validate_reveal_bridge` 接线落地；顺带纠正一条会让它被误杀的前提（2026-08-07）

**人话一句：** §5.6.13 (A) 那个"写好了但全仓没人调"的严格消费方，现在真的在跑了；
接线之前先查了一件可能让整件事作废的事（"出生改成 frame 0，这块记录是不是就该退役"），
**答案是不该退役，而且那条推论本身建立在一次名字撞车上。**

**一、先查那个可能作废它的前提：`reveal bridge` 与 `bridge ramp` 是两样东西。**

退役理由原话是："`34f8cf25` 加 reveal bridge 是为了摊平揭示时的 `2.243 rad` 阶跃；
出生若等于 frame 0，阶跃按构造归零，这东西就没有存在理由。" **前半句对，后半句挂错了对象。**
仓库里叫"bridge"的是两个不相干的东西：

| | 那条 **ramp** | 这块 **`reveal_to_playback_bridge`** 记录 |
| --- | --- | --- |
| 在哪 | `scripts/train.py:7269 / :7547` 的 teacher-q_des oracle，收据字段 `bridge_ramp_command_steps`；消费方是 `launch_n1_measured_vendor_v2_diagnostic.py:236/1776` | 生产方 `action_ball_prelong_semantics.py:3283-3311`；消费方就是 `_validate_reveal_bridge` |
| 干什么 | 把揭示那一 tick 的 `q_des` 阶跃按 `1/(frozen+1)` 摊到约 `35` 步 | 记揭示→回放开始那段窗口的账 |
| 内容 | 一个位置指令的插值 | `5..25` 共 `21` 个 WAIT 档的 `揭示 = 开始回放 + 开始前终止 + 截断`、七项权威 SHA 跨 `5` 个 update 不许漂、隐藏等待期 task 收入必须恰为 `0`、逐 mimic 项核函数/分母/收入、窗口内边界安全量 |
| 出生改成 frame 0 之后 | **失去存在理由**（§「九」6 说的就是它） | **一条检查都不失效**：六个块没有一个引用出生姿态或阶跃幅度，改的只是 `pre_swing_wait_s` / `timing_contract_sha256` 这些**取值** |

而且它**反而更承重**：阶跃这个借口消失之后，回放开始前的每一次死亡都是纯平衡/plant 故障，
而这是唯一按 WAIT 档把它们数出来的账。

**二、顺带把"出生改 frame 0"这条改动的现状查了三层**（因为它是本条的前置）：
①**机制码**：`materialize_action_ball_a211_frame0_exact_artifact.py` 在，
但 `docs/operations/tool_catalogue.md:128` 仍标 `SUPERSEDED / COUNTEREXAMPLE`、`launch_authorized=false`；
②**实验史裁定**：§12.4 的 `0/73` 退役判决已被「十」4 实测重开（接地后两门口径 `70/73`），
但那一节自己写着"重判要连 artifact 重铸一起走，本轮不代签"，§「九」6 写"本轮不动，只出方案"，
§「九」8 写"换出生姿态牵连 deploy 动作零点，这一步不能由 subagent 代签"；
③**现役代码**：`git log` 里只有 `c1b5ca10`（重定向接地收尾），
`commands.py` / split-ready artifact / lineage **一处未动**。
**裁定：它是一份待 Franco 签字 + 待重铸 4 份 artifact 的方案，不是即将落地的改动**；
即便落地，按"一"也不动这块记录。**所以走接线，不走退役。**

**三、接线前的两条硬前置，都查了。**
- **没有在跑的 `scale4096`**：pod1 无 `train.py` / `launch_action_ball` 进程，
  GPU 上唯一的 python 是别人的 `usd_check.py`（`1392 MiB`）。
- **`not_configured` 在生产上产不出来**：`require_bridge_telemetry` 默认 `True`，
  全仓三处传 `False` 全在 `test_action_ball_prelong_semantics.py`，零个生产调用方
  （这是 §5.6.13 (A) 那条就地更正的复核，结论不变）。

**四、改了什么。**
1. `validate_semantic_updates`（`action_ball_4096x5_prelong_gate.py`）逐 update 调
   `_validate_reveal_bridge`，跨 update 串 `previous_lifetime` / `expected_authority`；
   汇总写进 `aggregate.reveal_to_playback_bridge`（带七项权威 SHA、WAIT 档表、
   末轮逐档寿命、逐 update 摘要）——**记录与阻断同一批**，收据自陈这一步跑过了。
2. 导入期新增一条**活值**对表：gate 的 `BRIDGE_WAIT_COHORTS` / `BRIDGE_MIMIC_TERMS` /
   `BRIDGE_CAUCHY_MIMIC_TERMS` 必须等于生产方的 `_PRELONG_BRIDGE_WAIT_COHORTS` /
   `_PRELONG_MIMIC_EXP_TERMS ∪ _PRELONG_MIMIC_CAUCHY_TERMS` / `_PRELONG_MIMIC_CAUCHY_TERMS`。
   **不钉文件 SHA**——指纹只证明字节没动。`BRIDGE_TIMING_FIELDS` 对不上（生产方那份是实例属性），
   靠 `_exact_keys` 的精确键集在运行时兜底，这一点写进注释而不是假装已覆盖。
3. 夹具只留一份 `tests/prelong_bridge_fixture.py`，共享 gate 与 A/C 两个 launcher 的测试共用；
   字段名/档位/核函数归类从**生产方**取活值，不手抄。

**五、拒绝面变了什么（这是唯一有风险的地方）。** 新增硬拒四类，与 §5.6.13 (A) 方案一致：
`status != active_fail_closed`；七项权威 SHA / `wait_cohort_ticks` / `policy_dt_s == 0.02`
的形状与跨 update 相等；每档守恒 + `reveal/start/terminal` 跨 update 单调不减 + 整窗有新揭示；
`timing_at_reveal.reveal_count` 等于各档 reveal 之和。
另外 `_exact_keys` 是精确键集，生产方多写或少写一个字段都会拒 —— 所以两边必须同版本，
这也是本条把上面第 2 点那道活值对表一起加进来的原因。
**代价**：`action_ball_4096x5_prelong_gate.py` 是两个 211 launcher 的 `PRELONG_GATE_SOURCE`，
改它会挪动 launch claim 里那个源 SHA（运行时现算，没有存量期望值被打破）；
`solver_profile_sha256` 不含这个文件，所以对 solver pin 零代价。

**六、验收（`/workspace/hope_isaac_venv/bin/python`，pod1 CPU，`-p no:randomly`）。**
- 四个直接相关模块（共享 gate + A/C launcher + 生产方）：基线 **`436 passed / 0 skipped`**，
  接线后 **`447 passed / 0 skipped`**，`+11` **恰好**是新增的 `7` 个测试函数
  （其中 SHA 那条参数化 `5` 档），无一条旧测试变红。
- **变异测试真的在测这道门**：把门逐条改粗一档再跑，红的**只**是对应那一条：

| 粗一档改法 | 该红的 | 实测 |
| --- | --- | --- |
| 消费方整段变成 no-op（= 接线前的语义） | 全部 | `11 failed` |
| SHA 只量长度 `64`，不管十六进制/大小写 | 那 `5` 档 | `5 failed, 6 passed` |
| 只逐档看守恒，不看跨块总和 | 跨块那条 | `1 failed, 10 passed` |
| 只校验第一个 update 的 authority | 漂移那条 | `1 failed, 10 passed` |
| 只看单 update 快照，不做跨 update 单调性 | 回退那条 | `1 failed, 10 passed` |
| `status` 只要求"有值" | `not_configured` 那条 | `1 failed, 10 passed` |

（夹具里的假 SHA 特意带字母：全数字的假 SHA 会让"改一位成大写"变成空操作，
那样这条变异测试就成了自证。）
- 收据：pod1 `/workspace/franco/rvbridge_20260807/{base,wt}`（`base` = 本地 HEAD，
  `wt` = `base` + 本条改动），日志与探针在同目录 `out/` 与 `coarsen_probe.py`。
  **未跑 Isaac，未占 GPU。**

**七、没做什么。** 没有改任何阈值、没有改生产方、没有替"出生改 frame 0"下判决
（那条要 Franco 签字，见二）、没有发任何 `scale4096`。

#### 5.6.23 第四道门重定范围：诊断跑没有 reward activation ledger，那族证据在诊断跑里结构上不存在（2026-08-07 落地）

**人话一句：** `scale4096` 终局验收要 5 行 `HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=`，
诊断跑实测 0 行。查清楚之后的答案是**不该接线，该重定范围** —— 那本账是正式跑专用的，
诊断跑按设计根本不建它，而且运行时还有一道硬门禁止两本账并存。

**先回答那个先决问题：诊断跑到底该不该有 `reward_activation_ledger`？答案是不该。**
三层查证，逐层列证据：

| 层 | 问题 | 证据 |
| --- | --- | --- |
| 一 | 它是什么 | `utils/effective_reward_recipe.py` 的 `ActionBoundRewardEvidenceLedger` / `EffectiveRewardActivationLedger`。它是一笔**两段式提交**：optimizer 前 prepare、落 durable artifact、optimizer 成功后 commit、再 acknowledge。它给 PPO 上证据栅栏，并铸**可晋级**的正式收据。 |
| 二 | 正式跑拿它做什么 | 把 PPO 钉在 RewardManager 缓存合同上，并产出 `audit_reward_run.py` 那条正式审计链要读的四种事件（`EVENT_PREFIXES` 逐条列着）。 |
| 三 | 诊断跑不给它是有意还是遗漏 | **有意。** `my_on_policy_runner._effective_reward_activation_task_kind()` 对 `action_ball_diagnostic_unauthorized` 的跑直接 `return None`，注释原话是 "Diagnostic reward screens deliberately cannot mint formal evidence or promotion authority"；引入提交是 **`790714b3` "train(n1): keep formal audits off diagnostic PPO"（2026-07-29）**。而且诊断跑另有一本**替代**账 `reward_ppo_economy_ledger`（→ `HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_UPDATE_JSON=`，实测 5 行），运行时还有一道硬门：`"reward/PPO economy diagnostic cannot share a formal Reward ledger"`。 |

所以「给诊断跑接上正式账」不是补一处遗漏，而是推翻一条有名有姓的设计裁定，
并且会当场撞上那道互斥门。**改的是门的适用范围，不是跑。**

**顺带查出来的更硬的事实：这道门此前对任何一种跑都不可满足。**
同一个 `_audit_scale4096_terminal` 既要求 joint-safety 收据是
`hope_joint_safety_diagnostic_compact_update` / `diagnostic_compact_optimizer_committed_and_ledger_acknowledged`
（**正式跑不产**），又要求 5 行 `hope_reward_safety_transition_update`（**诊断跑不产**）。
两条合取起来，没有任何一种跑能同时满足。

**第五条同形的确实存在，而且就在同一条链上。**
`action_ball_4096x5_prelong_gate.validate_prelong_gate` 里的
`validate_group_income_updates(log_text)` 读 `HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON=`，
同样只有正式账才铸，实测同样 0 行。它排在 reward-safety 之后，所以以前一直没轮到它报错。
整条验收链逐项过完之后，**结构上不可满足的只有这一族**（两条阻断 + 两条同族只记录），
其余每一项诊断跑都真的在产：

| 验收项 | 诊断跑实测（`c0_scale4096_s16r1/run.log`） | 判定 |
| --- | --- | --- |
| 终局 checkpoint 定位 / `weights_only` 加载 / `iter`+claim 绑定 / 四组张量有限 | 全部满足（§5.6.21 已实跑证明） | 诊断原生 |
| `HOPE_JOINT_SAFETY_UPDATE_JSON=`（诊断紧凑事件） | 5 | 诊断原生 |
| `HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON=` | 5 | 诊断原生 |
| `HOPE_EXACT_BEHAVIOR_UPDATE_JSON=` | 5 | 诊断原生 |
| `HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_UPDATE_JSON=` | 5 | 诊断原生（就是那本替代账） |
| `HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS_UPDATE_JSON=` | 5 | 诊断原生 |
| `HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=` | **0** | **结构上不存在** |
| `HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON=` | **0** | **结构上不存在**（第五条） |
| `HOPE_EFFECTIVE_REWARD_ACTIVATION_UPDATE_JSON=` / `HOPE_REWARD_EPISODE_SEGMENTED_CLOSURE_UPDATE_JSON=` | **0 / 0** | 同族，本轮只记录不阻断 |

**改法。** 唯一出处放在两族共用的 `action_ball_4096x5_prelong_gate.py`：

* `classify_reward_evidence_regime(log_text)` —— **体制从跑自己的收据里读**，不接受调用方声明。
  每个 PPO update 的 joint-safety 收据都写着自己是正式账还是诊断紧凑账；一跑里两种混着出现就拒收，
  一行都没有也拒收（体制无法确定 ≠ 默认放行）。
* `reward_activation_evidence_scope(...)` —— 正式跑**照旧要满 5 行**（等强，缺一条就拒）；
  诊断跑标注「不适用」，并且**必须是 0 行**：诊断跑一旦发出这一族，说明它越权铸了正式证据，同样拒收。
  **记录与阻断同批，不是静默跳过。**
* 收据自陈：`validate_prelong_gate` 的结果和两个发射器的终局收据里都多一块
  `reward_activation_evidence_scope`，写清体制名、要不要、每个前缀实际读到几行、
  以及那三项严格零计数的**取源**。终局收据 kind 同批 `v2 -> v3`。

**关键的一步：不许出现「没观测到所以记 0」。**
`joint_qdes_forbidden_terminal_count` / `joint_actual_forbidden_terminal_count` /
`strict_hard_termination_count` 这三项原来**只**从 reward-safety 那一族推。摘掉那一族之后，
它们如果就地留 0，收据就会报三个从未被观测过的零 —— 那正是 oracle32 那次重定范围明令禁止的假收据。
现在改由诊断跑真的在产的 `HOPE_EXACT_BEHAVIOR_UPDATE_JSON=` 的
`termination_reason_joint_qdes_forbidden_count` / `termination_reason_joint_actual_forbidden_count` 供给
（这两格是「每个活跃终止项一格」的固定 producer ABI），并加一道**守恒普查**：
重新取源之后三个数必须跟它们自己的逐项出处对得上。
实测 `s16r1` 这两格都是 **0** —— 严格零这条主张在诊断跑里**真的可验证**，不是空转。

**A/C 两族同批同形。** 两个发射器逐字同形改；`test_action_ball_211_ac_family_config_parity.py`
与 `test_action_ball_211_launcher_shared_constants.py`（`53040fb0` 那两道「只差 obs 和 reward」门）全绿，
没有新增会逃逸的常量 —— 新增的名字全在共享的 `_P` 模块里，两族读同一份。

**顺带修掉一处第三份手抄。** 两个发射器各自手写了一份 `strict_zero_keys` 六元组，
是 pre-long gate / 四格 barrier 那条策略的**第三份副本**。代价是真实的：并行 workflow 的
「实际-q 硬边照记不照拦」裁定改了前两处、漏了发射器，发射器就会在 barrier 还没看到这一跑之前
先把它拒掉，另外两处的修改**等于没生效**。现在两个发射器直接读 `_P.STRICT_ZERO_SAFETY_COUNTERS`，
并有一条新门盯着不许再抄回去（此改动在本轮 HEAD 上是**逐值等价**的空操作：两份元组当时逐字相同）。

**变异证据（pod1，`/workspace/hope_isaac_venv/bin/python`，Python 3.10.18，每条都是真改文件后重跑）。**

| # | 变异 | 结果 |
| --- | --- | --- |
| M1 | 重定范围整体撤销（诊断跑也当正式跑要证据） | **KILLED** |
| M2 | 正式跑不再数那 5 行（把等强门拆掉） | **KILLED** |
| M3 | 诊断跑允许铸正式证据（阻断的另一半拆掉） | **KILLED** |
| M4 | 体制不再从收据读，写死成诊断 | **KILLED** |
| M5 | group income 不再随体制分派（诊断跑照旧硬要） | **KILLED** |
| M6a/M6b | C/A 发射器：严格零计数不再改由 exact-behavior 供给（留 0） | **KILLED**（守恒普查抓到） |
| M7a/M7b | C/A 发射器：去掉「只审诊断跑」的体制门 | **KILLED** |
| M8 | 发射器 `strict_zero_keys` 退回手抄 | **KILLED** |
| M9 | 收据不再自陈体制（去掉 scope 块） | **KILLED** |
| M10 | 去掉重新取源后的守恒普查 | **SURVIVED** |

**M10 这条要说实话**：单独去掉守恒普查没有任何测试会红，因为取源正确时两侧不可能不一致。
它的价值就是抓 M6a 那一类「换了出处却漏接一项」—— 实测 **M6a+M10 一起改则两条都活**，
也就是说 M6a 之所以被杀，靠的正是这条守恒普查。所以它不是装饰，但也不该被算进「11/11 全杀」里。

**两类误拒/漏拒都点名验过：**

* 等强侧：正式跑日志缺 `HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=` 或
  `HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON=`（整族缺、或只缺第 5 行）→ 仍然拒收；
  正式跑日志喂给诊断发射器 → 拒收，理由是「只审诊断跑」。
* 重定范围侧：诊断跑 0 行 → **不再拒收**，`applicable: false` 写进 PASS 收据；
  诊断跑一旦混入任一条该族 marker → 拒收。

**实跑对照（用真实的 `s16r1` 产物，不是夹具）。**
用打好补丁的 `_audit_scale4096_terminal` 直接审 pod1 上那份真日志：

```
reward_activation_evidence_scope.regime  = diagnostic_no_reward_activation_ledger
reward_activation_evidence_scope.applicable = false
observed_rows_per_prefix                 = 四条全 0
termination_reason_joint_qdes_forbidden  = 0     termination_reason_joint_actual_forbidden = 0
拒收理由前进到                            = actual_hard_edge_event_count=17 / actual_hard_terminal_count=13
```

也就是说**第四条墙已经过去了**。剩下那条正是并行 workflow 的「裁定三」正在降级的那一条
（`_P` 与 barrier 两处已改，发射器这一处本轮接上共享出处）。
把那两项按裁定三移出严格零之后再审一次，下一条墙是
**`reward term(s) action_rate_clamped are bitwise identical and non-zero across all 5 updates`**
—— 那是一条**关于这一跑内容**的发现（未申报的常数奖励项），不是「诊断跑给不出」的结构问题。
整条链到此不再有第六条同形。

**回归账（解释器 `/workspace/hope_isaac_venv/bin/python` = Python 3.10.18，12 个直接相关模块）：**

* 开工时的基线 `559b95f4`：**3 failed / 752 passed / 0 skipped**；同一批文件改完后 **3 failed / 775 passed / 0 skipped**
* 落地提交的**真实父提交** `34674c0e`（并行 workflow 中途插进来的一条，给四格 barrier 加了 1152 行测试）：
  **3 failed / 848 passed / 0 skipped**；本节提交 `bc08543a`（**直接 checkout 已推送的那份，不是本地工作区**）：
  **3 failed / 871 passed / 0 skipped**

两组对拍**同样的 3 条失败**，全在 `test_audit_reward_run.py`，与本节无关
（`soft_limit_recipe_params` 与 `negative_reward_semantics`，属于并行 workflow 的「裁定二」限位调价那条线）。
两组都是**净增 23 条通过、skip 数两边都是 0**。

**没关的洞（别当成已解决）：**

* 四格 barrier 的聚合收据**没有**把 `reward_activation_evidence_scope` 再陈述一遍。
  它拿得到（pre-long gate 的结果里有同一块，而 barrier 对那份结果做了内容寻址
  `prelong_gate.content_sha256`），但没有单独一行明账。没直接改是因为
  `_validate_terminal_safety` 用 `_exact` 把 `safety_counters` 钉成 16 键 producer schema，
  而那个文件本轮正被并行 workflow 改。体制声明因此放在终局收据的**顶层**，不在 `safety_counters` 里。
* `scale4096` 仍未通过全链，只是**理由又前进了一格**。`long4096` 本轮没有发。
* A 族仍然一格都没实跑过 `scale4096`；A 的改动与 C 逐字同形、同批变异，但缺实跑证据。

#### 5.6.24 三条裁定落地：形状照开源、验收看深度、硬超限只记不拦（2026-08-07 落地）

**人话一句：** 上一节（§5.6.20）把"我们的限位罚比 `build_1` 重 `3.5` 倍"归给了"门控差异"，
那是**从项名推出来的**，读完源码发现推错了 —— `build_1` 的主项**根本没有门控**。真正差的是
**核函数形状、聚合方式和量纲**。本节按 Franco 2026-08-07 的三条裁定把这三样改掉，并把
"什么算合格"从**频率**改成**深度**。**本轮没有发任何训练。**

##### 一、先把上一节推错的那条钉死：`build_1` 的 qdes 罚是全相位的

三层证据全部对上，缺一层我就会写"挖不到"：

| 层 | 证据 |
| --- | --- |
| **机制码** | `origin/build_1` 提交 `d7dcbdf484a5202619566839be0b031987342d88`（`dongc1`，`2026-08-02 03:40:47 +0800`），文件 `.../tasks/tracking/mdp/hope_rewards.py`，函数 `rally_all_joint_qdes_barrier`。在 `origin/build_2` 的 `b7ac2680` 上**逐字节相同**（diff 为空）。 |
| **实验史** | wandb `BerkeleyPingPong/hope_wbc/830xw9hy`（创建于该提交后 **7 分钟**）与 `i4dxpbwy`。两跑落盘 config 的 `rewards.rally_joint_qdes_saturation.func` 都是 `...hope_rewards:rally_all_joint_qdes_barrier`，weight `-0.65`，params `{safe_margin_fraction: 0.08, std_fraction: 0.03, topk: 4, topk_blend: 0.9}`，**两跑逐字一致**。 |
| **现役 argv** | 同一提交的 `cfg/task/HitterPingPong.yaml:206-210` 逐字相同；同文件 `:150-152` 把**旧的**同名 `rally_joint_qdes_saturation_weight` 显式设为 `null`。 |

**证伪式核实（不是读项名，是切出函数体做程序化检查）：**
`'time_to_strike' in body -> False`；`'gate' in body -> False`；`cmd` 的全部用法 =
`['cmd.metrics'] × 4`；唯一 `return` = `return debt`，没乘任何东西。
对照组 `rally_ankle_qdes_saturation_penalty` 同法检查 → 命中
`gate = (cmd.time_to_strike < t_pre) & (cmd.time_to_strike > -t_post)`，`return raw * gate_f`。
**该拦的拦住了（踝那条确实有门），误判的放行了（all-joint 那条确实没门）。**

它罚什么：`safe_excess = relu(safe_lo - 未钳 q_des) + relu(未钳 q_des - safe_hi)`，
带边是**硬限位内缩 `0.08×量程`**；`scaled = safe_excess/(0.03×量程)`；
per-joint 是 **Huber**（`0.5s²` if `s≤1` else `s-0.5`，**无上界、尾部线性**）；
聚合是 `0.1·mean(31) + 0.9·mean(top4)`。
因为 Isaac 的 `soft_joint_pos_limit_factor = 0.9`，软限位在中心 `±0.45×量程`、带边在 `±0.42`，
**差恰好一个 `std_fraction`** —— 也就是说"`q_des` 正好顶在部署钳位边界"这一点，
**恰好是 Huber 的拐点 `s == 1`**。这是刻意标定的。

**闭合校验（用它自己的公式重算它自己的日志，不是转述）：**
iter 3437 实测 `topk_debt = 0.29154` → 公式给出 reward/step ∈ `[-0.00341, -0.00379]`，日志 `-0.00412`；
iter 20 实测 `topk_debt = 5.31036` → `[-0.06213, -0.06904]`，日志 `-0.06540`。
两处都贴在区间边上（`Episode_Reward` 是回合内均值、`Metrics` 是瞬时 env 均值，口径差 ~10%）。
weight / dt / topk_blend 任一项搞错一档，这个区间会差 `1.5~15` 倍，立刻露馅。

> **顺带一个反例，已入账：** `hope_env_cfg.py:2602-2613` 里的**静态默认**是
> `-0.45 / 0.05 / 0.75`，与实跑不同。只看 env_cfg 会把三个数全读错。

##### 二、裁定二：形状照开源对齐（读代码，不按项名推）

**四家 + 我们自己的上游，实现是同一条。** 逐字核过的结论：

| 来源 | 核函数 | 输入 | 对照的限位 | 归一 | 聚合 | 死区 | 地板 | 上限 | 门控 | 全身权重 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IsaacLab `joint_pos_limits` | 线性 hinge | 实际 q | soft | **无（raw rad）** | SUM | 有 | 无 | **无** | 无 | — |
| **BeyondMimic**（= 我们自己的上游 `tracking_env_cfg.py`） | 同上（直接复用） | 实际 q | soft | raw rad | SUM | 有 | 无 | 无 | 无 | **`-10`，全关节** |
| mjlab tracking | 同上 | 实际 q | soft | raw rad | SUM | 有 | 无 | 无 | 无 | `-10`，全关节 |
| mjlab 步行 | 同上 | 实际 q | soft | raw rad | SUM | 有 | 无 | 无 | 无 | `-1` |
| unitree_rl_lab mimic / 步行 | 同上 | 实际 q | soft | raw rad | SUM | 有 | 无 | 无 | 无 | `-10` / `-5` |
| 厂商 AMP parkour `dof_pos_limits` | 同上（legged_gym 式） | 实际 q | soft | raw rad | SUM | 有 | 无 | 无 | 无 | `-2.0` |
| `build_1` qdes | **Huber**（近端二次、远端线性） | **pre-clamp q_des** | 硬限位内缩带 | `/(0.03×量程)` | `0.1·mean+0.9·top4` | 有 | 无 | **无** | **无** | `-0.65` |
| **我们（改之前）** | `1-exp(-4d)` | 两条通道 | soft 内缩带 | 每关节归一 `[0,1]` | SUM | 有 | **0.25** | **1.0** | 无 | `-5` |

**选定的形状基准 = 我们自己的上游 BeyondMimic（= IsaacLab `joint_pos_limits`）。**
理由三条：(i) 它就在我们仓里、是本分支的父类，四家彼此逐字同核，是唯一无争议的共识；
(ii) 它用 **raw rad**，所以**权重可以直接照抄**而不是靠猜换算；
(iii) `build_1` 那条虽然是同一族（线性尾、无上界），但它是**指令级**、还带 top-k 混合聚合，
而 Franco `2026-07-21` 已裁定 SUM 不用 top-k —— 取开源那条正好两不冲突。

**采纳的两处偏离，都写下理由：**

1. **软限位处的折角用一个宽 `b` 的 Huber 过渡磨圆**（`b = margin_frac × soft_span`）。
   `margin_frac -> 0` 时**逐点退回**开源原式，所以这是"同一条曲线的光滑版"，不是第二笔罚。
2. **反利用地板不删，挪到机械硬限位。** 旧地板 `0.25` 挂在"踩进软带"上，四家都没有；
   挪到硬边之后软带内完全连续，而"不存在一串正违规、罚金却趋于零的路径"这个性质仍然成立。

**同批还修掉一条不是形状的病：`margin_frac` `0.05 -> 0.02`。**
`0.05` 恰好等于护栏的投影内沿 —— barrier 的带外沿与被钳关节的落点**是同一个点**，
31 个关节里 **29 个**的 `m_eff` 正好命中它（另 2 个是肩 roll，站姿豁免把它压到 `0.0246`）。
于是 `intrusion` 是正是零由 **1 ulp 的浮点抖动**决定，而旧核在那一点是 `0 -> 0.25` 的跳变。
`0.02` 让带边真正离开 clamp 边：被钳关节**确定性零罚**。

**落地后的核（两条通道共用，单位 rad）：**

```
m_eff  = min(margin_frac, d(default_q) - 0.005)      # 站姿豁免,无量纲
b      = m_eff * (hi - lo)                            # 带宽,rad
x      = b - min(q-lo, hi-q)                          # x<=0 带外;x=b 压在软限位;x>b 已越限
ramp   = x^2/(2b)  if x <= b  else  x - b/2           # 尾部斜率恒 1 rad/rad,无上界
hard   = relu(hard_lo-q) + relu(q-hard_hi)            # 只有撞机械边才 >0
value  = 0  if x<=0  else  ramp + penalty_floor*b*(hard>0)
return   sum_j value_j                                 # SUM,全相位
```

投影罚同族（`d = |pre_clamp_qdes - 投影点|`，rad；`c = knee_frac × 包络跨度`）：
`cost = d²/(2c)` if `d ≤ c` else `d - c/2`。**深处的梯度不再衰减** —— 旧核在 `d = 0.92`
处的梯度只剩边界处的 `1/40`，railed 的关节几乎感觉不到往回拉的力，这正是
"浅处罚太重、深处罚太轻"的机制。

**权重（重算过，不是沿用）：**

| 项 | 旧 | 新 | 依据 |
| --- | ---: | ---: | --- |
| `joint_limit`（实际 q） | `-5`（归一 `[0,1]`） | **`-10`**（rad） | 上游 BeyondMimic / mjlab-tracking / unitree-mimic 全身版同值。**交叉验证**：`build_1` 收敛态全身越软限位总量 `0.003 rad`，`-10 × 0.003 × 0.02 = -0.0006/步`，与它日志逐位吻合；早期峰值 `-0.0040/步` 对应 `0.02 rad`。 |
| `qdes_limit_barrier`（命令 q_des） | `-5` | **`-10`** | v2 硬合同要求两条通道逐字段同权同带宽（改一边开机即拒）。 |
| `qdes_projection_penalty` | `-5` | **`-1`** | Franco 裁定的起点 `-1`，换核后按"同等策略水平交叉验证"复核成立，见下表。 |

> **注意别照抄权重号码。** 旧 `-5` 作用在**每关节归一 `[0,1]`** 上，新 `-10`/`-1` 作用在
> **rad** 上，两者**不可比**。这正是 §5.6.20「对齐经济不是对齐权重」那条的具体形状。

##### 三、裁定一：验收判据看**深度**，不看**频率**

`build_1`（`830xw9hy` 3438 iter 全 54 项 + `i4dxpbwy` 21897 iter）的实测曲线：

| iter | 指令级带外关节占比 | top4 债 | 硬钳关节数/步 | **越硬限最大深度 (rad)** | 钳位累计率 | 投影距离 RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.1307` | `5.306` | `2.539` | **`0.3622`** | `0.1768` | `0.2350` |
| 20 | `0.1543` | `5.310` | `2.869` | **`0.3769`** | `0.1764` | `0.2167` |
| 60 | `0.0827` | `1.475` | `0.929` | `0.0975` | `0.1734` | `0.0789` |
| 100 | `0.0331` | `0.335` | `0.201` | `0.0144` | `0.1435` | `0.0172` |
| 250 | `0.0231` | `0.250` | `0.167` | `0.0145` | `0.1040` | `0.0117` |
| 700 | `0.0193` | `0.180` | `0.106` | `0.0069` | `0.0870` | `0.0131` |
| 2000 | `0.0266` | `0.258` | `0.150` | `0.0133` | `0.0821` | `0.0224` |
| 3437 | `0.0320` | `0.292` | `0.175` | **`0.0148`** | — | `0.0219` |

**读法（这就是 Franco 那句"深度降了其实就安全了"的依据）：**

1. **它到最后也没有"不超"。** 收敛态仍有 `3.2%` 的关节-请求落在带外、每步 `0.175` 个关节被硬钳、
   钳位累计率 `7.7%~8.2%`（开局 `17.7%`）。**频率只降了约 5 倍，而且会反弹。**
2. **它降的是幅度。** 越硬限最大深度 `0.377 -> 0.0148 rad`（**24 倍**）、
   top4 债 `5.31 -> 0.29`（**18 倍**）。**这才是收敛信号。**
3. **`iter 500 -> 3437` 出现反弹：最大深度 `+117%`、投影距离 RMS `+81%`，带外占比 `0.0193 -> 0.0320`。**
   这不是退化 —— 它说明 **`build_1` 的罚轻到允许策略"学会打球之后重新把行程用回去"**。
   **这正是"所有 penalty 都是 trade off"这句话的实测证据，也是不能把罚调重的理由。**
   任何"频率必须单调趋零"的门都会在这一段把一条正在变好的跑判死。

**因此本分支采纳的验收判据（写成可以失败的门，全部只读深度）：**

| # | 指标 | 阈值 | 依据 |
| --- | --- | --- | --- |
| **D1** | 实际 q 越**硬**限最大深度 | `≤ 0.02 rad` | `build_1` 收敛态 `0.0069~0.0148`（含反弹段），开局 `0.377` |
| **D2** | 指令级投影距离 RMS（归一化） | `≤ 0.03` | `build_1` 收敛 `0.0117~0.0224` |
| **D3** | 本跑自身的深度收敛倍数 | 最大深度相对本跑开局 **降 ≥ 10×** | `build_1` 实测 24×；留一半余量 |
| **D4** | `max_intrusion_depth_frac`（带宽倍数，`1.0` = 正好压在软限位） | 报告 + 趋势，**不设硬阈值** | 新口径无历史对照，先积累 |
| **不设门** | 钳位频率 / 带外关节占比 / 硬钳关节数 | **只报告** | `build_1` 收敛仍 `3.2%`/`7.7%`，只降 5× 且反弹；把它当门等于拒收唯一已知能打到球的配方 |

**前置已落地：把机制层每步都在算、但从来没往外吐过的深度计数器接进 economy JSON。**
新增只读快照 `hope_rewards.peek_qdes_depth_telemetry(env)`（不清零，与既有 `Live/` 消费方并存），
由 `my_on_policy_runner` 写进 `reward.joint_limit_depth`：
`projection_{observed_sample_count, sample_count, joint_count, normalized_distance_sum,
max_normalized_distance}` + 两条通道各自的
`{observed_sample_count, intrusion_sample_count, intrusion_joint_count, max_intrusion_depth_frac}`。
**理由：没有深度就没法按深度验收**；而且此前 `qdes_limit_barrier` 到底是
"`38%` 的样本各罚 1 个关节"还是"`1.2%` 的样本各罚 31 个"完全判不出来，两种情况的处置完全相反。

##### 四、裁定三：取消**实际-q 硬超限**那条惩罚/门（取消 ≠ 静默删除）

依据是本轮自己测的：`build_1` 的 `Episode_Termination/actual_q_hard_limit_audit`
**从 iter 20 起恒 0** —— 策略学会之后本来就不硬越限，这条轴买不到东西，
却会在 5 个 update 的新策略上直接拒收。

**逐条判过范围，不是一刀切：**

| 对象 | 属于"硬超限"? | 处置 | 理由 |
| --- | --- | --- | --- |
| `actual_joint_position_forbidden_zone`（DoneTerm） | 是 | **已是 `terminate=False`**（`2026-08-05`），本轮不再改 | 惩罚/reset 早已取消，遥测与归因记录器仍强制存在 |
| `actual_hard_edge_event_count` / `actual_hard_terminal_count`（pre-long gate + four-grid barrier 的 `STRICT_ZERO`） | **是** | **从阻断降级为"照记不照拦"** | 本裁定的正主。计数仍必须**存在且是合法非负整数**（缺失/畸形照样拒收），非零时进摘要 `WARN`，但不再是拒收理由 |
| `joint_actual_forbidden_terminal_count` | **否** | 保留 `STRICT_ZERO` | 它必须是 0 是因为该 DoneTerm 已配成 `terminate=False`；非零 = **我们自己的接线退化回了 reset**。验的是接线，不是机器人 |
| `joint_qdes_forbidden_terminal_count` | 否 | 保留 | ActionBall 投影模式下只对 **NaN/Inf** 的 q_des 触发，那是数值 bug，不是学得会的行为 |
| `strict_hard_termination_count` / `nonfinite_count` | 否 | 保留 | 同上，实现层数值健康 |
| `actual_joint_limit_barrier_v2`（`joint_limit` 罚） | **否**（是软限位罚） | 由**裁定二**处理（换开源形状） | 它罚的是软限位邻域，不是机械硬边 |
| `qdes_projection_penalty` | 否（指令级） | **保留按超出量收费** | Franco 明确要保留 |
| A211 launcher / `sweep_..._physical_ready_qdes.py` 的 nominal-hold 零硬边检查 | 否 | **保留，是真安全兜底** | 它们跑的是**脚本化保持姿势**，不是学习中的策略；那里撞硬限位是台架/资产缺陷，学习学不掉 |
| `launch_n1_vendor_baseline_diagnostic.py` 的 `zero_actual_hard_edge` | — | **本轮不动，另一条 lane** | vendor-N1 基线，不在 A/C 两族范围内；若要一并放宽须单独裁定 |

**「WARN 必进摘要」已落实：** pre-long gate 结果顶层新增 `warnings`（不埋在 `safety` 子树里），
`safety` 里新增 `actual_hard_edge_counters` / `actual_hard_edge_blocking: false` /
`actual_hard_edge_warnings`；four-grid 聚合收据同结构，并**逐位交叉核对**两侧的硬边观测
与"它不阻断"这个自陈（不一致即拒收）。`TERMINAL_ACCEPTANCE_POLICY` 新增
`actual_q_hard_edge` 块，把"测量并汇总、永不作为拒收理由"写成收据里能读到的政策。

##### 五、预期剂量与 `|负|/正`（发车前给 Franco 对三方表用）

基线 = `c1_scale4096_s15r1` 第 5 个 update 实测（逐项闭合误差 `≤1.8e-07`）：
正 `+0.0187/步`、负 `-0.3085/步`、`|负|/正 = 16.48`，其中 qdes 三项合计 `-0.2504/步`（占罚金 `75.6%`）。

换核换价后，**按同一份实测行为**（同样的 `Σ(1-exp(-4d))=1.9492`）重算：

| 项 | 改前（实测） | 改后（预测） | 说明 |
| --- | ---: | ---: | --- |
| `qdes_projection_penalty` | `-0.19492` | **`-0.019 ~ -0.031`**（全带 `-0.004 ~ -0.038`） | 带宽取决于同一份超出量摊在几个关节上（`k=4~8` 为中心估计；`k=3` 给上界、`k=31` 给下界） |
| `qdes_limit_barrier` | `-0.03914` | **`-0.000 ~ -0.009`** | 带宽 `0.02 < 0.05` 内沿 ⇒ 被钳关节确定性零罚；ActionBall 上很可能**恒为 0** |
| `joint_limit` | `-0.01639` | **`-0.000 ~ -0.004`** | 落进 `build_1` 自己的历史区间（收敛 `-0.0006`、早期峰值 `-0.0040`） |
| **qdes 轴合计** | **`-0.2504`** | **`-0.019 ~ -0.037`**（中心） | **降 `7~13` 倍** |
| **负收入合计** | `-0.3085` | **`-0.077 ~ -0.095`**（中心） | 非 qdes 那部分 `-0.0581` 不动 |
| **`\|负\|/正`** | **`16.48`** | **`4.1 ~ 5.1`**（全带 `3.3 ~ 5.8`） | 正收入按不变的 `+0.0187` 计 |
| **qdes 轴占罚金** | `75.6%` | **`25% ~ 39%`** | |

对三方表：`build_1` `830xw9hy` @iter2 峰值 `7.12`、@iter4 `6.93`；`i4dxpbwy` @4 `10.87`；
收敛 `0.554`/`0.307`。**改后我们 `4.1~5.1` 落在 `build_1` 早期之下、收敛之上。**
qdes 轴占比 `25~39%` 与 `build_1` 早期 `16.2%`/`29.3%` 同档，**份额倒挂消失**。

> **诚实话两条。** (1) 这是**预测**，不是实测：它假设行为不变而只换价，而换价本来就会改行为。
> 真值要等下一跑的 economy JSON（现在它已经会吐深度了）。
> (2) 改后我们可能**比 `build_1` 当时还轻**。如果长跑显示学不会"别顶限位"，
> 该往回加的是**深度侧的斜率**，不是把频率重新当门。

##### 六、顺手记一条与权重、形状都无关的病（本轮**没有**修）

`left/right_shoulder_roll_joint` 的**设计站姿本身就落在投影包络之外 `0.0497 rad`**
（软限位 `[0.0480, 2.4827]`、站姿 `0.1200`、包络 `[0.1697, 2.3610]`）。
一个输出恒零动作的策略每步白付投影罚，而且梯度方向是把两肩往外推。

- 旧核旧价下这条的成本是 **`-0.01736/步`** = 我们全部正收入 `+0.0187` 的 **93%**；
- 新核新价下降到 **`-0.00045/步`**（近端二次 + 权重 `-1`），**小了 38 倍** —— 换形状顺手把它压住了，
  **但没有治好**：梯度方向仍然是错的。
- `qdes_limit_barrier` 的站姿豁免 `m_eff = min(margin, d_default-0.005)` 早在 `2026-07-25`
  就为这个病打过补丁，**投影包络没打**。正解是把同一条豁免搬到包络内缩
  （`inset_eff = min(0.05, d_default - eps)`），但那是改**安全包络**，
  **不在本轮三条裁定的授权范围内，留给 Franco 拍板。**

##### 七、这一节不代签什么

1. **不代签"改完就能打到球"。** 本轮只改价与形状，一次训练都没发。
2. **不代签第五节那张预测表是实测。** 它是拿旧行为算新价，`k`（超出量摊在几个关节上）没有直接测量，
   只有 Markov 下界 —— 这正是第三节那批深度计数器要解决的事。
3. **不代签 `qdes_limit_barrier` 还有用。** 带宽 `0.02` 严格窄于护栏内沿之后，
   它在 ActionBall 上很可能**恒为 0**，实质是一条零价遥测通道。
   `build_1` 与四个开源库**都没有** post-clamp 的邻近 barrier。要不要正式退役，需要 Franco 一句话。
4. **不代签 Stage-1 那条 lane 的调价是独立决定。** `HOPEStage1NaturalClipRewardsCfg` 继承
   `HOPEActionBallRewardsCfg`，共用同一个 kernel；它的权重跟着从 `-5` 换到 `-10`
   **是为了不让它继承一个换了单位的旧号码**，不是给它单独定价。
5. **不代签 vendor-N1 lane 的硬边门也该放宽**（见第四节表末行）。

#### 5.6.25 一阶平滑罚照开源改形状：封顶版退役，换回上游无封顶的 `action_rate_l2` −0.1（2026-08-08 落地）

**一句话**：机器人每一步动作和上一步差多少，我们要罚。原来这条罚有个"封顶"——
再抖也最多罚到某个数为止。今天把封顶去掉了，改成四家开源一模一样的那条，权重 `−0.1`。

##### 1. 先把我自己传播过的一个错说法收回来

我之前说过"封顶 `9.0` 是按 `σ=0.02` 标定的，我们把探索 `σ` 提到 `1.0`，所以标定错配了"。
**这句话是错的，别再往下传。**

翻原始 docstring 就知道：它自己算过 **"fresh 随机策略一步能把 31 维动作甩出 `‖Δa‖²≈60+`"**——
`60+` 正好是 `2 × 31 × 1.0²  = 62`。**写这个封顶的人当时就知道 `σ≈1.0`。**
所以 `9.0` 不是照着 `0.02` 标的，是**在明知 `σ≈1.0` 的前提下有意选的一个数**：
他想解决的问题（早期净流为负 → 摔死比活着划算）也确实存在，不是臆想出来的。

**它今天退役，不是因为标错了，是因为"代价全付了、想买的东西没买到"。**

##### 2. 代价：这一项在整条谱系上是焊死的，不是暂时饱和

| | 数 | 出处 |
| --- | --- | --- |
| 封顶档位 | `9.0` | `hope_env_cfg.py :: action_rate_clamped` `params` |
| 我们实测 `‖Δa‖²` 期望 | `2 × 31 × 1.001² = 62.1` | s15r1 五个 update 的 `policy_std_mean` |
| `build_1` 开局实测 | `63.1` | iter 4 反推 |
| `build_1` **收敛时**实测 | `10.8 ~ 12.05` | 21896 iter，两跑 |

**收敛了都还在 `10.8`，从来没掉到 `9.0` 以下。** 所以"现在饱和、以后自然解冻"这件事
在这条谱系上**不会发生**。实测佐证：s15r1 的 C0/C1 两格 × 5 个 update，这一项的 raw
**逐位**等于 `9.000000`（`raw_sum = 98304 × 9`），加权后恒为 `−3538.945068` ——
导数处处为零，只剩每步 `−0.036` 的死税。

（这也正是那道通用护栏当初抓住的东西：任何奖励项跨 update 跨 cell 逐位相同都要被拒。
**这轮没有放宽它** —— `action_ball_4096x5_prelong_gate.py` 的 `DECLARED_CONSTANT_REWARD_TERMS`
**仍然是空的 allowlist，那道门一行没改**。换成无封顶形状之后 raw 随策略变动，它本来
就不再是常数项，不需要任何申报。**是这道门把这件事逼出来的，不是被它绕开的。**）

##### 3. 它想买的东西没买到：带着封顶，今天**已经**是"摔死最优"

这是**证伪封顶的那张表**。同一把尺子，都是每步、都乘过 `dt = 0.02`：

| | 正收入/步 | 负罚金/步 | 净流/步 | `V_继续`（55 步、γ=0.99） | `V_摔死` | 谁更划算 |
| --- | --- | --- | --- | --- | --- | --- |
| **我们**（C0 u4，post-`24254020`，**带着封顶**） | `+0.0187` | `−0.0776` | `−0.0589` | **`−2.50`** | `−10 × dt = ` **`−0.20`** | 摔死，**12.5×** |
| **`build_1`**（iter 4，同等策略水平，**唯一已知能打到球的**） | — | — | `−0.191 ~ −0.334` | **`−19 ~ −33`** | **`0`**（它没有 death penalty） | 摔死，**∞** |

两件事同时成立：

1. **封顶没有改变符号。** 带着它，`V_继续 −2.50` 仍然远差于 `V_死 −0.20`。
   封顶只把倍数从 `32×` 压到 `12.5×`，**从来没把它压到 1 以下**。买了个"更浅的盆地"，
   盆地还是那个盆地。
2. **坐在这个盆地里照样能学会。** `build_1` 坐得**比我们深 3~5 倍**（净流 `−0.191~−0.334` vs
   我们 `−0.0589`），而且它**根本没有 death penalty`（`V_死 = 0`，摔死是干净的零）——
   条件比我们恶劣得多，它学会了打球。
   补一句原理：PPO 的 advantage 是对着**价值基线**算的，不是对着绝对零；
   而我们 u1--u4 的死亡 **88% 是 `robot_hit_table` + `base_fell_tilt`**，
   **是被判定的、非自愿的**，不是策略够得到、能主动去选的目标。
   "净流为负 ⇒ 策略会学着自杀"这个推理链，两头都不成立。

3. **更要命的是：它爬出去用的引擎，正好就是这一项。**
   `build_1` 早期 `action_rate` 占罚金 `32~60%`，每步从 `−0.126` 一路衰减到 `−0.022`（`5.2~5.8×`）。
   **这条衰减就是它的 `|负|/正` 穿过 `1.0` 的原因**——策略把动作抖动学平了，罚金自己就下去了。
   封顶把这台引擎的梯度置零：**我们卖掉的，是唯一那条能自己爬出去的路。**

##### 4. 四家开源到底怎么写的（逐字读的，不是按项名推的）

| 来源 | 实现 | 权重 | 有封顶吗 |
| --- | --- | --- | --- |
| **IsaacLab 2.1.0**（我们实际跑的那份） | `envs/mdp/rewards.py:245-247`：`torch.sum(torch.square(action - prev_action), dim=1)` | — | **无** |
| **BeyondMimic**（我们自己的上游） | `tasks/tracking/tracking_env_cfg.py:237` | `−1e-1` | **无** |
| **mjlab-tracking** | `src/mjlab/tasks/tracking/tracking_env_cfg.py` | `−1e-1` | **无** |
| **unitree_rl_lab-mimic**（g1_29dof） | `tasks/mimic/robots/g1_29dof/*/tracking_env_cfg.py` | `−1e-1` | **无** |

**四家一个封顶都没有，四家都是 `−1e-1`。**（unitree 的**步行**臂是 `−0.05`，不是模仿臂；
智元 AMP parkour 的 `−1e-3` 活在 discriminator 的收入经济里，量纲不同，**勿抄**。）

**顺带纠正一处旧记录**：`docs/research/four_way_dr_reward_comparison_20260807.md:85` 那张表
把我们这一列写成"保留封顶版"，并说我们是"四家唯一带值钳的"——**前半句今天起作废，
后半句当时就说明了问题**：唯一一家和其他三家不一样，通常不是我们发现了什么，是我们漂了。

还有一件事值得单独说：**我们仓库的 `tracking_env_cfg.py:237` 本来就继承着上游那个 `−0.1`。**
换句话说，这次不是"引进一个新数字"，是**把 v2 包里那个把它归零、改用封顶版的动作撤销掉**，
回到继承链本来的样子。而且我们连实现都不写第二份 —— `mdp/__init__.py` 是
`from isaaclab.envs.mdp import *`，用的就是上游那个函数本体。

##### 5. 权重为什么取 `−0.1`：四个锚，互相独立

1. **四家共识**：`−1e-1`（上表）。
2. **第一性推导**：31 维、相邻两步独立采样 ⇒ `E‖Δa‖² = 2 × 31 × σ²`；
   我们实测 `σ ≈ 1.001` ⇒ `62.1`。每步剂量 `= 0.1 × 62.1 × dt 0.02 = ` **`−0.1242`**。
3. **同等策略水平实测**：`build_1` iter 4（同底盘、同 31 维、`σ` 同为 `1.0`、两边都还不会击球）
   实测 `action_rate = ` **`−0.1262 / −0.1264`** 每步。
   **和上面第 2 条预测差 `1.6%`** —— 差值就是均值漂移项 `|μ_t − μ_{t−1}|² ≈ 1.0`。
   两条完全独立的路径落到同一个数上，这才是"对上了"。
4. **聚合后的经济比值**：改完后 `|负|/正` 预测 `8.85 ~ 8.95`，
   落在 `build_1` iter-4 的实测带 `6.93 ~ 10.87` **之内**。

**否决掉的两个候选**（同一把尺子量的）：

| 候选 | 每步剂量 | 罚金/收入 比值 | 判 |
| --- | --- | --- | --- |
| `−0.2`（无封顶） | `−0.248` | `15.5` | **超出 `build_1` 带，否决**——旧的 `−0.2` 是配封顶用的号码，去掉封顶就翻倍了 |
| **`−0.1`** | **`−0.1242`** | **`8.85~8.95`** | **采用** |
| `−0.05` | `−0.0621` | `5.5` | 低于带，定价不足 |

这里有一件容易看反的事：**旧的 `−0.2` 看起来比 `−0.1` 狠一倍，实际付得比它轻 3.5 倍**
（`−0.036` vs `−0.1242`），因为封顶把 `62` 削到了 `9`。**权重对上不等于触发率对上。**

##### 6. 改完之后的账

- **每步剂量**：发射时 `−0.1242`（σ≈1.0），学平之后掉到 `−0.0216 ~ −0.0241`（按 `build_1`
  收敛时的 `‖Δa‖² = 10.8~12.05`）——**`5.2~5.8×` 的衰减，这就是我们买的东西。**
- **`|负|/正`**：`8.85 ~ 8.95`，在 `build_1` 同期带内。
- **收据里要自陈的一件事**：`action_rate_l2` 的 raw **没有上界**（raw 动作是未截断的高斯样本，
  `ClampedJointPositionAction` 只钳 `q_des`，`ActionManager._action` 存的是原始动作）。
  所以收据里写的 `63.1` **不是"可达上界"，是"发射时探索 σ 下的工作区包络"**，
  `evidence_class` 相应改成 `measured_operating_envelope_not_a_reachable_cap` ——
  否则读的人会拿一个期望值当最坏值去做预算。

##### 7. 落地清单与两处"指纹对上不等于语义对上"

Isaac 侧：`train.py`（v2 包 DIRECT 表 + stage-1 期望权重 + drop-message）、`hope_env_cfg.py`、
`hope_rewards.py`、`action_ball_prelong_semantics.py`、`action_ball_reward_causal_prelaunch.py`。
C 侧同批同形：`action_ball_c211_env.py`（`C211_ACTION_RATE_CLAMP` → `C211_ACTION_RATE_POST_DT_WEIGHT`）。
镜像与注释：`mirrored_constant_registry.py`、`effective_reward_recipe.py`、`4096x5_prelong_gate.py`。

**A/C 同批同形**是 Franco 的准绳「A 和 C 应该大多数设置都一样，除了 obs 和 reward」的直接后果。
改完确认**没有**触发 `53040fb0` 那两道"只差 obs 和 reward"的门
（`test_action_ball_211_ac_family_config_parity` / `test_action_ball_211_launcher_shared_constants`）——
它们盖的是 task profile 叶子与发射器模块级常量，`c211_env` 的这个常量不在其中，
所以改名不该惊动它们，实跑也确实没惊动。

**两处踩过同一个形状的坑**，一起记下来：

1. **`effective_reward_recipe.py` 里 `action_rate_clamped` 的出处标签写的是 `"MJLab-aligned"`。**
   **是错的**——mjlab 自己就没有封顶。项名对得上 MJLab，**形状从来没对上过**，
   而这张表把没对上的那部分写成了对上。已就地改成 `HOPE-only (retired 2026-08-08; no upstream has a value cap)`。
2. **收据的 `EXPECTED_EFFECTIVE_REWARD_SHA256` 需要重钉，而重钉不能只换号码。**
   做法：先用同一条流水线（r5 契约 + `adopted` 表 + `_reward_receipt_payload` 的规范化）
   **复算旧值，逐位得到当时钉的 `b096b79c…a59d`**，证明我复现的是活值那条链；
   然后在同一条链上生成新值 `41631955…53d7`（两个 action_id 各自算出来相同）。
   **先能复现旧的，再换新的** —— 否则换的只是一个号码，不是指纹。
   同批修的还有测试夹具：退役件必须**从收据里消失**（`_reward_receipt_payload` 明令
   拒收 `weight == 0` 的项），不能留一条零权重的尸体；换项时名字/callable/params/weight
   四样一起变，项数仍是 30，字典序仍成立。

##### 8. 收据

pod1 `/workspace/hope_isaac_venv/bin/python`（**Python 3.10.18**，唯一能跑全护栏的那个；
`/usr/bin/python3` 缺 hydra 会**静默跳 17 条**）。`CUDA_VISIBLE_DEVICES=` 纯 CPU，
**未占 GPU、未跑 Isaac、未写任何 artifact、未放宽任何门限**。
两棵独立 worktree A/B，同一份测试清单（13 个文件，含两个 `mujoco_native/tests/`）：

| 树 | 结果 | skip |
| --- | --- | --- |
| `ar_base_20260808` @ `7b3bc9ba`（基线） | `12 failed, 750 passed` | **0** |
| `ar_head_20260808` @ 本轮工作区 | `12 failed, 759 passed` | **0** |

**新增失败 0 条，消失失败 0 条，两边失败集合逐条相同。** 多出来的 9 个 pass 是本轮新增的
变异测试（该动的要动 / 同序列逐位相同 / 无天花板 / 权重复现 build_1 实测剂量 / 不许影子实现）。

**基线本就红的那 12 条**（detached worktree 上的既有失败，与本轮无关，点名如下）：
**11** 条挂在 `hope_rewards.py:63` 的 `RuntimeError: reward eligibility requires pre_strike or strike_window`
（`test_v2_reward_terms.py` 的 deferred-prize / strike-capture / legal-base / virtual-landing / climb-mode 族），
1 条是 `test_reward_flags_overrides.py:3033` 的 `assert 9 == 7`（YAML 声明计数）。

> **就地更正（2026-08-08，同日复核）：上面这段原写"10 条 / 9 条 + 1 条"，与它自己那张
> 表里的 `12 failed` 对不上。** 逐条数过：`test_v2_reward_terms.py` 上是 **11** 条同一个
> `hope_rewards.py:63` 的 `RuntimeError`（`pytest --tb=line | sort | uniq -c` 给出 `11`），
> 加 `test_reward_flags_overrides.py` 那 1 条，正好 `12`。表没错，是这段分项写漏了 2 条。

**第二份独立对拍（同日，另一棵 worktree、另一个基线点、13 文件清单逐字如下）。**
上面那份的基线取在 `7b3bc9ba`；这份把基线换成**落地提交的真实父提交** `d7e44c75`，
用的是同一个解释器、同样 `CUDA_VISIBLE_DEVICES=` 纯 CPU：

| 树 | 版本 | 结果 | skip |
| --- | --- | --- | --- |
| `arate_20260808_wt` | `d7e44c75`（落地提交的父） | `12 failed, 1016 passed` | **0** |
| `arate_20260808_wt` | `4683840e`（落地提交 + 表格修复） | `12 failed, 1025 passed` | **0** |

清单：`test_v2_reward_terms` / `test_reward_flags_overrides` / `test_reward_flags_mdp` /
`test_action_ball_prelong_semantics` / `test_action_ball_reward_causal_prelaunch` /
`test_materialize_action_ball_reward_ppo_economy_receipt` / `test_action_ball_4096x5_prelong_gate` /
`test_action_ball_task_config` / `test_action_ball_211_four_grid_prelong_barrier` /
`test_action_ball_211_isaac_four_grid` / `test_action_ball_211_launcher_shared_constants` /
`mujoco_native/tests/test_action_ball_c211_env` / `mujoco_native/tests/test_mirrored_constant_registry`。

**失败集合两边逐条相同、`+9` 个 pass 全部是本轮新增的变异测试。** 这份清单里显式包含了
`test_action_ball_211_launcher_shared_constants` 与 `test_action_ball_211_isaac_four_grid`
——就是 §7 说的那两道"只差 obs 和 reward"的门，实跑证明本轮改名没有惊动它们。

##### 9. 逐 update 的预测账（把 §6 那个聚合数拆开）

§6 给的是聚合区间 `8.85 ~ 8.95`。下面把它按 update 拆开，输入只有两组：
post-`24254020` 的 C0 实测（`u0` 该项占罚金 `48.9%`、是全部正收入的 `1.75×`；
`u4` 占 `46.4%`、是正收入的 `1.92×`，`|负|/正 = 4.139`，去掉该项 `2.219`），
以及新剂量 `0.1 × 63.1 × dt 0.02 × 98304 = 12406.0`（旧的是 `3538.945`，`3.51×`）。

| | 正收入/步 | 其余罚金/步 | 旧 `action_rate`/步 | **新** `action_rate`/步 | 旧 `\|负\|/正` | **新** `\|负\|/正` | `build_1` 同 iter 实测带 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 `u0` | `+0.02057` | `−0.03762` | `−0.036` | **`−0.1262`** | `3.58` | **`7.96`** | iter 0：`5.35` / `9.27` |
| C0 `u4` | `+0.01874` | `−0.04159` | `−0.036` | **`−0.1262`** | `4.139` | **`8.95`** | iter 4：`6.93` / `10.87` |

**两端都落在 `build_1` 同一迭代的实测带里**（`u0` 的 `7.96 ∈ [5.35, 9.27]`，
`u4` 的 `8.95 ∈ [6.93, 10.87]`）。而且**负罚金总额我们仍然比它轻**：
新的 `−0.1678/步` vs `build_1` iter 4 的 `−0.2102 / −0.3908`。

三件必须一起说清楚、不许只报好消息的：

1. **`|负|/正` 变差了一倍多**（`4.139 → 8.95`）。这不是副作用，是"把 3.5 倍的欠付补上"
   的直接结果。判据不是"比值要小"，是"比值要落在同等策略水平的参照带里" —— §5.6.20
   已经裁过：**比值本身不构成动权重的理由**，拿它跟收敛态比是单位没对齐。
2. **该项占我们罚金的份额会到 `75.2%`**，高于 `build_1` 早期的 `32~60%`。原因不在它涨了，
   在 `24254020` 之后**其余罚金塌得更多**（qdes 轴从 `−0.2333/步` 掉到 `−0.019~−0.031`）。
   份额高于带这件事**留在台账上**，等第一份长跑的实测账回来再判要不要动。
3. **真正的缺口在正收入这一侧，本轮不碰。** 我们 `+0.0187/步`，`build_1` 收敛 `+0.0926/0.0982`
   （`5×`）。它的比值穿过 `1.0` 靠的是**两条腿**：`action_rate` 衰减 `5.2~5.8×`（本轮买回来的）
   **加上**正收入长 `5×`（收入分层塌成一层，另账）。只买回一条腿不会让比值自己下去。

##### 10. 交接：这次改价让哪些已铸产物过期

`EXPECTED_EFFECTIVE_REWARD_SHA256` 变了（`b096b79c…a59d` → `41631955…53d7`），
所以**所有按旧配方铸出来的内容寻址产物都对不上新活值**，下一次发车前必须重铸：

- `configs/` 下 **18 份**含 `action_rate_clamped` 的已铸产物（`a3_vendor_runtime_authority_*`
  的 `*.shared_ready.training_contract.json` 与 `n1_reward_economy_*/reward_economy.v1.json`）。
  **它们是历史收据，不要就地改**——按日期新开一轮重铸，旧的原样留档。
- 四格 `A0/A1/C0/C1` 的 `materialize → recipe` 两级要重跑；`oracle32 / scale4096` 随之重跑。
  materializer 会**主动拒收**旧 SHA（`reward term ... weight drifted` / SHA 不符），
  所以这一步不会被静默跳过。
- 已经跑完的 `s15r1` 那批读数（`|负|/正 = 14.37~17.90`）是 `24254020` **之前**的，
  §9 那张表用的是 `24254020` **之后**的 C0 实测；两批不要混用。

#### 5.6.26 接 §5.6.25 §10 去重铸，先卡在一件事上：那枚"新指纹"本身不是活值（2026-08-08 清点）

**人话先说**：上一节交接说"18 份产物过期了，下一次发车前必须重铸"。去铸之前先按规矩
**复算了一遍旧值**，结果发现要拿来当靶子的那个新号码 `41631955…53d7` **不是活代码算出来的**
——它是**一份手改的测试夹具**算出来的，而那份夹具漏抄了 `24254020`（08-07 限位那次）改的
**三项**。所以现在往下铸，铸出来的东西会跟这枚靶子对不上；而把靶子改对，
**改的是这条谱系的身份**，那要先报告再动。**本轮因此只清点、只出证据，一份产物都没铸。**

##### 1. 清点：现在到底有几处还挂着退役配方，分别在什么名字底下

`git grep` 全树（排除 `logs/`），按"能不能影响下一次启动"分四类：

| 类 | 处数 | 位置 | 说明 |
| --- | ---: | --- | --- |
| **活代码里的价目钉** | **3**（+1 支线） | `materialize_action_ball_reward_ppo_economy_receipt.py:73` `EXPECTED_EFFECTIVE_REWARD_SHA256` = `41631955…`；`launch_n1_vendor_baseline_diagnostic.py:165` `STATIC_EFFECTIVE_REWARD_RECIPE_SHA256` = `845d75b4…`；`launch_a3_vendor_identity_smoke.py:88` `EXPECTED_REWARD_RECIPE_SHA256` = `845d75b4…`；支线 `MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256` = `ce910ac2…` | **同一条谱系的同一份配方，三个不同字段名，装着两个不同的号码**，而活值是第三个。后两个是**硬启动门**（`if reward_sha != …: raise LaunchRefused`），所以 vendor-V1 那两支发射器**今天已经启动不了** |
| **在役产物里封着的配方** | **3 份** | `configs/a3_vendor_runtime_authority_20260802_r9/{bh_loop_c,bh_block}.shared_ready.training_contract.json`、`configs/n1_reward_economy_20260802_r9/reward_economy.v1.json` | 被 `a3_vendor_action_registry.py` 按 path+sha 钉住，是在役的那三份 |
| **留档产物** | **15 份** | `a3_vendor_runtime_authority_{20260731,_r2,_r4,20260801_r5,_r6,_r7,20260802_r8}` 的契约 + `n1_reward_economy_{20260801_r7,20260802_r8}` | 历史收据，按 §10 的规矩**原样留档，不就地改** |
| **夹具（会把漂移遮住的那层）** | **3 个模块** | `tests/test_materialize_action_ball_reward_ppo_economy_receipt.py` 的 `_r6_contract_from_r5`（那张手抄的 `adopted` / `adopted_params` 表）、`tests/test_launch_n1_vendor_baseline_diagnostic.py`、`tests/test_launch_a3_vendor_identity_smoke.py` | 见 §4：正是它让"指纹对上了"而语义没对上 |

18 = 3 在役 + 15 留档，和 §10 那个数对得上。**另有一处同谱系的过期钉不是 reward 引起的**，
顺手记在这里：registry 的 `identity_repin_producer` 钉 `b90bac5f…`，
而 `materialize_a3_vendor_identity_manifest.py` 在 `d4e1e70c`（上一轮 solver-pin v3）动过，
现在是 `ab7f8fdb…`。**上一轮重铸漏了这一处** —— 又是一次"改一份就暴露一份没跟上的"。

##### 2. 复算收据：两条链都当场复现出来了，不是拿别人的数

pod1 `/workspace/hope_isaac_venv/bin/python`（Python 3.10.18），`CUDA_VISIBLE_DEVICES=` 纯 CPU，
干净 worktree `/workspace/franco/remint_20260808` @ `17f4bae7`，**未占 GPU、未写任何 artifact**：

| 链 | 输入 | 算出来 | 和当时钉的比 |
| --- | --- | --- | --- |
| **真产物链** | 在役 r9 两份契约里的 `effective_reward_recipe`，按 `_reward_receipt_payload` 的规范化取 sha256 | `845d75b4f409725e…`（两个 action_id 相同，30 项） | **逐位相同**（自陈 `sha256`、顶层 `effective_reward_recipe_sha256` 三处一致） |
| **夹具链** | r5 契约 + 手抄 `adopted` 表 + 08-08 换项，同一套规范化 | `41631955ece024ce…`（两个 action_id 相同，30 项） | **逐位相同**，就是现在钉在物化器里的那个 |

**两条链都复现得出来，所以下面这句不是推测**：这枚在役的新号码，出处是**夹具那条链**，
不是**真产物那条链**。往前追：`845d75b4`（08-02）是最后一次由**真的物化契约**产生的值；
`b096b79c`（08-07 限位那次）和 `41631955`（08-08 一阶平滑这次）都是在夹具上算的。

##### 3. 活值对拍：新号码和活代码差在三项、五个叶字段

活值取自**真的一次 Isaac 启动**（并行 worker 的 s19 队列，commit `2dcde6b8`，
A0 `materialize` 阶段吐出的 `a211_effective_reward_recipe.json`，41 项，`sha 0e633996…`；
C0 同阶段 `5d876e1b…`）——不是读代码推的：

| 项 | 夹具（= 在役钉 `41631955`） | **活值（实跑）** | 何时漂的 |
| --- | --- | --- | --- |
| `action_rate_l2` | `-0.1`，`{}` | `-0.1`，`{}` | 对上 |
| `qdes_limit_barrier` | `-10.0`，`margin_frac 0.02` | 同 | 对上 |
| `joint_limit` | `-10.0`，`margin_frac 0.02` | 同 | 对上 |
| **`qdes_projection_penalty`** | **`-5.0`**，`{shape_rate: 4.0}` | **`-1.0`**，`{knee_frac: 0.05, objective_weight: -1.0}` | `24254020` |
| **`qdes_limit_barrier_probe`** | `margin_frac` **`0.08`** | `margin_frac` **`0.02`** | `24254020` |
| **`actual_joint_limit_barrier_probe`** | `margin_frac` **`0.08`** | `margin_frac` **`0.02`** | `24254020` |

**漏的是同一件事的另一半**：`24254020` 那次同时改了主项和投影罚、也把两条 probe 的带宽跟着改了，
而 08-07 重钉时手抄的 `adopted` / `adopted_params` 只覆盖了 `qdes_limit_barrier` 和 `joint_limit`
两条主项。08-08 这次在那份已经漏了三项的夹具上继续往下算，于是把漏抄一并继承了。

**这个对拍用的是 A211 剖面**，vendor-V1 的剖面项数不同（30 vs 41）。但这三项来自
`HOPEActionBallRewardsCfg`，`HOPEPingPongActionBallA3VendorV1.yaml` 的 `rewards:` 段
**只设了 `action_acc_weight` / `racket_position_coarse_weight` / `racket_position_coarse_std`**，
一个字都没碰这三项，所以漂移同样落在 vendor-V1 上。**唯一能定这个数的仍然是 vendor-V1 自己启动一次**
——这一条是本轮**没做到**的（见 §6）。

##### 4. 门是好的：物化器当场拒收，测试却是绿的

在干净 worktree 里对**真的在役 r9 产物**跑一次 `--verify`：

```
REFUSED: effective reward recipe differs from the adopted common coarse+fine recipe
```

同一棵树上，四个直接相关的测试模块 `154 passed`，**没有一条是因为 reward 红的**
（红的 5 条全在 identity-smoke，原因是 §1 末尾那处 `identity_repin_producer` 过期钉）。
**这就是"指纹不等于语义一致"的现场**：夹具自己造一份带新配方的契约再去校验，
于是**真产物已经对不上了，测试还是绿的**。文件级指纹也一样看不见——
registry 的 30 枚 path+sha 钉逐个复核，只有 1 枚漂（还不是 reward 那件事），
因为**漂的是文件内容里的语义，不是文件本身有没有被换掉**。

##### 5. 题目身份：没动，而且这次本来就动不了

`base_question_sha256` 在最近两轮 tape build report（`v3pin_tape_seed0_20260807_r1`、
`escapefix_tape_seed0_20260807_r1`）里都是 **`81eed5139b98…`**，与上一轮一致。
更强的一条：**reward 配方根本不是抽题的输入** —— 全树扫 `action_rate_clamped`，
命中只落在上面那 18 份契约/收据里，**tape / bundle / manifest 谱系一处都没有**。
所以这次改价在构造上就碰不到题目身份。

**反话也写清楚**：本轮**没有**铸任何产物，所以"`canonical_sha256` 会不会动"这句话
现在无从谈起；一旦真的开铸，**封着 recipe 的那些 `canonical_sha256` 必然会动**，
那是构造决定的，**不能拿它当"题变了"的证据，也不能拿它当"题没变"的证据**。

##### 6. 为什么停在这里，以及下一步的顺序

停下来的理由只有一条：**靶子不对，铸出来的东西也不会对**。真按流程铸，产物会带着活值配方
（第四个号码），而物化器手里的靶子是夹具那个，照样拒收。要往下走必须先把靶子改对，
而"把 `EXPECTED_EFFECTIVE_REWARD_SHA256` 改成别的值"= 改这条谱系声称自己在跑哪份配方，
**那是身份问题，先报告再动**。

**正确顺序（下一轮照这个走）**：

1. **实跑一次 `HOPEPingPongActionBallA3VendorV1`**，把 vendor-V1 的活值配方 sha 拿到手。
   **不许再用夹具推** —— 夹具就是这次出事的地方。
2. 三个字段名下的钉**一起改**（`EXPECTED_EFFECTIVE_REWARD_SHA256` /
   `STATIC_EFFECTIVE_REWARD_RECIPE_SHA256` / `EXPECTED_REWARD_RECIPE_SHA256`，
   外加 canary 支线那枚），并且**把夹具从"手抄一张 adopted 表"改成"从真契约取"**，
   否则下一次改价还会漏。
3. 顺手补上 `identity_repin_producer` 那枚过期钉。
4. 然后才按 §10 的规矩**新开一轮日期目录**重铸 r9 那三份在役产物，旧的原样留档。

##### 7. 顺带确认的两件事

- **四格 A0/C0 的 `recipe` 阶段在新配方下确实能过 boot 门 —— 本轮自己在 HEAD 上跑过一遍，
  不是引用别人的**。干净 worktree `/workspace/franco/remint_20260808` @ `17f4bae7`，GPU1
  （等它空出来才起，`gpu1 clear after 630s`；GPU0=yikang、GPU2=mjlab 全程未碰），
  只跑 `materialize → recipe` 两级（`oracle32 / scale4096 / long` 一律没跑）：

  | 格 | materialize | recipe | 吐出来的活值配方 sha |
  | --- | --- | --- | --- |
  | `C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off` | `EXIT 0`（48 s） | `EXIT 0`（53 s） | `5d876e1bac865277…` |
  | `A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off` | `EXIT 0`（47 s） | `EXIT 0`（54 s） | `0e633996af4790bb…` |

  这两个 sha 与并行 worker 在 `2dcde6b8` 上跑出来的**逐位相同**，
  等于把"`2dcde6b8 → 17f4bae7` 那段差集（动捕 npz、`configs/a3p_p1_0807_*`、
  两支动作物化脚本、legacy 目录）不是 reward 配方的输入"这句话**量出来了**，而不是看 diff 推的。
  第一次起还吃了一记 `REFUSED: namespace parent must be an existing real directory`
  ——新 worktree 里 `logs/rsl_rl/<experiment>/` 不存在，建完目录重跑即过；记在这里省下一次重踩。
- **本轮一行代码、一份产物都没改**，所以没有"改动 vs 基线"的对拍要做；
  上面所有数字都是在 `17f4bae7` 的干净 worktree 上读出来的。

#### 5.6.27 揭示->回放桥接线后第一次开火：拦下来的两件事，一件是浮点、一件是过期的门（2026-08-08）

**一句话**：`scale4096` 四格终局验收被拒的那条 `teacher_rate min/mean/max are unordered`，
**不是数据坏，是门自己不可能被满足**；顺着它往下清，同一个函数里还有第二道门是
**四个提交之前就已经被裁定取消、但没扫到**的。两条都改了，A/C 两族同一份改动。

##### 1. 现场：门第一次真跑，就把跑完的两格全拒了

C0/C1 都跑满 5/5 个 update（`terminal_kind=clean_completion`），然后被拒。
`_validate_reveal_bridge` 是 2026-08-07（`39953498`）才接进 `validate_semantic_updates` 的，
在那之前**零调用点** —— 每个 update 都在写这本账，从来没人读。所以这是它第一次开火。

##### 2. 定性：min > mean 还是 mean > max，差多少

把真实 `run.log`（`/workspace/franco/fourcell_20260808/…/c0_scale4096_s19r1/run.log`）里
update 0 那条记录的三个数打出来（离线读日志，没占 GPU）：

| 字段 | min | mean | max | 结论 |
| --- | --- | --- | --- | --- |
| `teacher_rate` | `0.9990914883334084` | `0.9990914883334086` | `0.9990914883334084` | `min == max` 逐位相等，`mean` 高 **1 ULP** |
| `scaled_t_hit_s` | `0.9608729642981773` | `0.9608729642981775` | `0.9608729642981773` | 同上，高 **2 ULP** |

**不是个别档**：C0 和 C1 各 5 个 update，**10 条记录全部**在这三个浮点字段上有 1~5 ULP 的
越界（最坏 5 ULP ≈ `1.1e-15`）。两个整数刻度字段（`time_to_contact_tick`、
`expected_bridge_ticks`）一次都没越界 —— 它们是精确的。

> 顺带回答一个问的问题：这五个计时字段**不是按 21 个 WAIT 档各存一组**的。
> 21 个 WAIT 档在 `lifetime_conservation.wait_cohorts` 那个块里（逐档守恒 + 跨 update 单调，
> 真数据全过）；计时字段是**整窗一组**，分母就是 `reveal_count`（update 0 是 3690）。

##### 3. 三层查证生产方：三个数同一批样本，问题出在"和不等于逐项相加"

`utils/action_ball_prelong_semantics.py`：

- `_bridge_timing_sum` / `_bridge_timing_min` / `_bridge_timing_max` 三个累加器**用同一个
  `reveal_column` 掩码、在同一处更新**（约 2370 行）。所以**不存在**"min/max 跨 update 累计、
  mean 只算本 update"那种口径错配 —— 那个猜想排除了，三个数确实是一批样本。
- `min`/`max` 走 `torch.minimum` / `torch.maximum`：**次序统计量**，只有比较没有算术，精确。
- `mean` 是 `total / timing_count`，而 `total` 是**逐 policy tick `add_` 上去的 float64 累加器**。

于是：当一个 update 里所有揭示样本取值相同（单 clip 的 `teacher_rate` 对 4096 个环境就是同一个
常数，此时 `min == max` 逐位相等），`sum/count` 几乎不可能正好落回那个值。
**这是算术事实，换任何写法都消不掉**，也就是说原来那条零容差的 `min <= mean <= max`
在退化分布上**本来就不可能被满足**。

- "空样本填默认值"那个形状**这里没有**：`timing_count == 0` 时生产方写的是 `None`，
  门的 `_finite_number` 照样 fail closed。不是"没观测到所以记 0"。

##### 4. 处理一：缝只给 mean，min/max 之间照旧零容差

改的不是"把阈值调松"，是**把一条检查拆成两条**，因为那三个数根本不是一类东西：

- `min > max` —— **零容差，一个 ULP 都不放**。两个都是原样本值，没有算术，倒挂只可能是
  取样集不同。这条比原来更严（原来它被裹在一个复合条件里）。
- `mean` 落在 `[min - 缝, max + 缝]` —— 缝 = `256 ULP × max(|min|,|max|)`，按量级缩放。

缝宽的来历（不是拍脑袋）：累加链长 ≈ 每 update 24 tick × 5 update，加上每次 batch 内
4096 元素树形归约约 12 层，量级 ~`1.3e2` 次舍入，理论最坏 ~`1.3e2` ULP；实测 10 条真收据最坏
5 ULP。取 256 比理论最坏留 2 倍、比实测留 50 倍，同时仍比任何**语义上的**乱序
（`1e-3` 起）紧 4 个数量级以上。**实跑核对：真数据只用掉缝的 1.0%**（收据里自陈，见下）。

**记录与阻断同批**：收据每档新增 `mean_slack_fraction_used`，写明这一档实际用掉多少缝。
接近 1 就该回头看生产方，而不是继续放宽。

##### 5. 处理二：桥里那条硬边是**四个提交之前就已经取消**的门，漏网了

修完浮点，同一批真数据在 update 2 上撞出第二道门：`bridge safety values are inconsistent`。
拆开看，六个子条件里响的是 `minimum_physical_hard_gap_rad < 0`：

| 格 | update | 实测值 |
| --- | --- | --- |
| C0 | 2 | `-0.00026083` rad |
| C1 | 2 | `-0.00022596` rad |

这个**不是浮点噪声**（`2.6e-4` 比 ULP 大 12 个数量级），是真事：实际关节越过机械硬限位 0.26 mrad。
生产方那里 `hard_gap = min(q - 下限, 上限 - q)`，取的是**实际 q**。

而 Franco 2026-08-07 的**裁定三**（`24254020`）已经就这件事拍过板：
**实际-q 硬超限照记不照拦**，当时把 `actual_hard_edge_event_count` /
`actual_hard_terminal_count` 从 pre-long gate 与 four-grid barrier 的 STRICT_ZERO 名单挪进
`REPORTED_HARD_EDGE_COUNTERS`，并且"验收改看深度不看频率"。

**桥这块漏网了，因为它当时还是零调用点的死代码** —— 裁定扫过了活检查，没扫到它。
它讲的是同一个物理事实，只是从"次数"换成了"深度"。所以按同一条裁定处理：出 WARN、
一路抬到 gate 结果顶层的 `warnings`、值本身留在收据里，**不拒收**。

**取消 != 静默删除**：`hard_gap` 照旧过 `_finite_number`，缺失/非数/非有限**照样 fail closed**，
只是它的**取值**不再阻断。安全自洽那五条（root min>max、余弦越界、滑移 max<mean 等）**原样保留**。

##### 6. 变异测试：两个方向都钉住，每条都构造成"粗一个档次会放行"

pod `/workspace/hope_isaac_venv/bin/python`（**Python 3.10.18**）实跑，每次都是真改文件再重跑：

| 变异（把门改粗/改错） | 结果 | 被哪条抓住 |
| --- | --- | --- |
| M1 退回原版零容差 `min<=mean<=max` | **35 条红** | 全线 —— 真收据的形状根本过不去 |
| M2 把 mean 的缝放粗到 `1e-9` 相对 | 2 条红 | `…mean_outside_min_max_by_more_than_rounding…` |
| M3 偷懒：把缝**同时**套在 min/max 上 | 1 条红 | `…min_above_max_is_refused_even_by_a_single_ulp` |
| M4 用固定绝对 epsilon 代替按量级缩放 | 1 条红 | `…mean_slack_scales_with_magnitude…` |
| M5 静默删掉硬边 WARN | 2 条红 | 两条硬边收据门 |
| M6 不做处理二，硬边继续阻断 | 2 条红 | 两条硬边收据门 |

M1 的 35 条红是最强的一条证据：**因为夹具也一并改成了真值**。
`prelong_bridge_fixture.py` 原来把 `teacher_rate` 写成 `1.0/1.0/1.0` —— 数学上完美有序，
于是这条检查在测试里**永远是绿的**，一碰真数据就把跑完的两格全拒。夹具扮演生产方，
就得带上生产方真有的浮点行为；现在它带的是 C0 update 0 的实测三元组。
**这正是"指纹不等于语义一致"的又一个实例：夹具长得像收据，但保护的是一个不存在的世界。**

##### 7. 把这个函数的每一条硬拒对着真数据过了一遍（免得下一轮再一条条撞）

用 `sys.settrace` 对 10 条真桥记录做行覆盖，再 AST 枚举函数里每一条
`raise PreLongGateRefused` 及其守卫条件：

**34 条硬拒：32 条被真数据走到并通过，0 条误报，剩 2 条从没被触及**：

| 从没被触及的 | 为什么 | 什么时候会碰上 |
| --- | --- | --- |
| `A211 bridge income is not progress-only` | 这批日志全是 C211 | **A 族跑到 `scale4096` 时**（今天还没跑到，见下） |
| `bridge mimic … empty errors differ` | 真数据每个 mimic 项都有有限误差，`finite_errors == 0` 那支没进过 | 某一项全零核时 |

也就是说：**除了这两条，桥里所有硬拒都已经被真数据验证过一遍**，
下一轮不会再一条一条撞。（不含 `_exact_keys` / `_finite_number` / `_counter` / `_sha256`
这些共享形状助手 —— 它们被所有字段走了无数遍。）

##### 8. A/C 同批同形，两道 parity 门未触发

改动全在**共享**的 `action_ball_4096x5_prelong_gate.py` 与**共享**夹具里，
A/C 两族拿的是同一份，没有一个 launcher argv、一片 Hydra 叶子被动过。
`53040fb0` 的两道门实跑确认：`test_action_ball_211_ac_family_config_parity.py` +
`test_action_ball_211_launcher_shared_constants.py` **169 passed, 0 skipped**。

另外记一笔现状：**A 族今天还没跑到 `scale4096`** ——
`a211_four_arm_diagnostic` 下只有 `materialize` / `recipe`（`oracle32` 两次都是
`DIAGNOSTIC_UNAUTHORIZED`）。所以上面 §2 的 ULP 证据全部来自 C 族；
但这两条修的都是**与 profile 无关**的共享代码路径，A 族跑到时会走同一条。

##### 9. 实跑数字

- 离线复现与验证：`/workspace/franco/bridge_ulp_20260808`（本轮自己 fork 的 worktree，
  `git worktree add … 2dcde6b8 --detach`，没有拷 `logs/`）。**全程没占 GPU** ——
  只读日志文本。GPU0=yikang、GPU1=别的 session 的 probe、GPU2=mjlab，全程未碰。
- 修前：10 条真记录里 update 0 就被拒（两格都是）。修后：**10 条全过**，
  最坏只用掉缝的 **1.0%**（C0 update 3 `scaled_t_hit_s`）、**0.9%**（C1 update 2 `pre_swing_wait_s`）。
- 测试（`/workspace/hope_isaac_venv/bin/python`，**Python 3.10.18**，`-rs`）：
  桥/门/生产方/两个 launcher 六个模块 **638 passed, 0 skipped**；
  加上两道 parity 门共 **807 passed, 0 skipped**。
- 新增 8 条测试（4 条计时顺序 + 4 条硬边收据），全部配了"粗一档就过不了"的变异证据。

##### 10. 一条留给下一轮的诚实提醒

本轮这两条**都是"门错了"**，但性质不同，别混成一件事：

- 处理一是**门在数值上不可满足** —— 它验的东西对，写法错，任何正确的生产方都过不去。
- 处理二是**门的适用范围过期** —— 它验的东西已经被裁定不该阻断，只是裁定没扫到死代码。

两条都**不是**"因为它挡路所以判它错"：处理一有 10 条真收据 + 算术必然性，
处理二有四个提交之前的白纸黑字裁定。**接线一个长期零调用点的检查，等于第一次做验收；
写它的时候脑子里那份"应该长这样"的假设，要拿真数据一条条对过。**

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

### 8.1 225 家族整族退役（2026-08-06 裁决 + 落地）

**人话一句**：`A225/C225` 不是"旧一点的 211"，是**另一套 ABI**（actor `225` vs `211`，
critic `318` vs `319`），2026-08-03 之后就再也发不出车了。这一轮核实到**四道各自独立的门**
任何一道都足以拦死它，于是把它整族删了（约 `9.5k` 行），并把"它为什么回不来"写成了会失败的测试。

**上一轮清理留的线索只对了一半。** 那份报告说"`MotionOnPolicyRunner.__init__` 硬拒 225 的
obs_mode，所以整族不可达"。前提**成立，而且被低估了**——runner 那道门其实是**第四道**，
前面还有三道；但同一份报告顺带的"两个 225 trainability validator 是零调用点"也**只对了一半**，
因为同名的**测试文件**并不是 225 的（见下"文件名在撒谎"）。

#### 三层核实

| 层 | 结论 | 活证据 |
| --- | --- | --- |
| 机制码 | **不可达，四道门** | 见下表 |
| 实验史 | **最后一次真跑 = 2026-08-03 17:26，且从未越过 `oracle32`** | pod1 上全部 `logs/rsl_rl/agibot_a3_action_ball_a225_four_arm_diagnostic/*` 目录名都带 `-DIAGNOSTIC_UNAUTHORIZED`，`model_*.pt` 一个都没有；`c225` 连一个 run 目录都没有 |
| 现役 argv | **无人拉起** | 仓库无 CI；无队列/runbook 引用这两个发射器；唯一活引用是 `action_ball_211_transition_preflight.py` 的 `LEGACY_EXPERIMENT_NAMES` / `WRITER_SOURCE_NAMES`——那是**排空清单**（拿名字去扫 `/proc/*/cmdline` 和旧 log namespace），不解析文件路径，**删文件不影响它，也不该跟着删** |

四道门（任何一道单独就够）：

1. **gym 注册表里根本没有它**。`config/agibot_a3/__init__.py` 只注册 A211/C211；
   两个 225 yaml 指向的 `HOPE-PingPong-ActionBall-A225Learnability-AgibotA3-v0` / `C225...`
   不存在，`gym.make` 在任何 obs_mode 逻辑之前就 `NameNotFound`。
   （`test_action_ball_task_config.py:313-314` 已经把"这两个 id 不在注册表里"钉成断言。）
2. **没有对应的 EnvCfg 类**。`hope_env_cfg.py` 里 `obs_mode` 的默认值集合中没有 `action_ball_a225/c225`。
3. **`train.py:3325` 拒 actor 合同**：`configured_actor_contract in {a225,c225,a210,c210}` → `_OverrideError`。
4. **`my_on_policy_runner.py:358` 拒 obs_mode**：同一组四个名字 → `RuntimeError`，在 `super().__init__` **之前**。

两个 225 发射器自己的 argv 写死了 `task.actor_obs_contract=action_ball_a225`（`launch_action_ball_a225_four_arm_diagnostic.py:1366`）
和 `...=action_ball_c225`（`launch_action_ball_c225_diagnostic.py:2018`），所以它们连第 3 道门都过不去——
是**结构性死代码**，不是"暂时不用"。

#### 文件名在撒谎：七个"225 测试"里有四个其实是 211 的

这是这一轮最容易踩的坑，也是"指纹不等于语义一致"的教科书例子。按名字删会**误删活的 211 覆盖**：

| 文件名 | docstring 自陈的真实对象 | 处置 |
| --- | --- | --- |
| `test_action_ball_a225_trainability.py` | "the fresh trainable **A211** leaf"，加载的是 `action_ball_a211_trainability.py` | **留**（活的 A211 覆盖） |
| `test_action_ball_c225_trainability.py` | "the fresh trainable **C211** leaf" | **留**（活的 C211 覆盖） |
| `test_action_ball_a225_c225_contract.py` | "the fresh fixed-midpoint **A211/C211** split" | **留** |
| `test_action_ball_225_observation_producers.py` | "real-task **A211** and causal-ball **C211**" | **留** |
| `test_launch_action_ball_a225_four_arm_diagnostic.py` | 真测 A225 发射器 | **删** |
| `test_launch_action_ball_c225_diagnostic.py` | 真测 C225 发射器 | **删** |
| `test_materialize_action_ball_a225_lineage.py` | 真测 A225 materializer | **删** |

同样的坑在 production 侧更贵：**`mdp/action_ball_c225_rewards.py` 是活的 C211 奖励模块**，
被 `mdp/__init__.py:17` 星号导入，两个 term（`c225_strike_ball_paddle_center_proximity`、
`c225_landing_outcome_actual_contact`）直接接在现役 `hope_env_cfg.py:2428-2438` 上，
并被 `training_contract.py`、`action_ball_c211_trainability.py`、`mujoco_native/action_ball_c211_env.py`、
`mjlab_lane/isaac_alignment.py`、`action_ball_prelong_semantics.py`、`effective_reward_recipe.py` 六处引用。
**名字带 225，身份是 211 的承重墙，一个字节都不能动。**

#### 删了什么 / 留了什么

**删（10 个文件，`9531` 行）**：两个发射器 + `materialize_action_ball_a225_lineage.py` +
两个零调用点 validator（`action_ball_225_trainability.py`、`action_ball_c225_trainability.py`）+
两份 task yaml + 三个只测这些死文件的测试。

**留**：

- `mdp/action_ball_c225_rewards.py` 与 `test_action_ball_c225_rewards.py`——活的 C211 奖励。
- `actor_observation_contract.py` 的 `ACTION_BALL_A225/C225`——它们是**ABI 台账**，
  记着"`225` 宽度的 checkpoint 为什么绝不能塞进 `211` 网络"，并且被活的
  `test_action_ball_a225_c225_contract.py` 核对着。删掉只会把明确的"legacy 被拒"退化成含糊的"未知合同"。
- 四处 legacy 拒绝名单（`train.py`、`my_on_policy_runner.py`、`a211_trainability._LEGACY_MODES`、
  以及两个 211 发射器/测试里的 a225 负例）——**这些是让 225 保持不可达的门本身，不许动**。
  `test_launch_action_ball_a211_four_arm_diagnostic.py:1819-1862` 与
  `test_launch_action_ball_c211_diagnostic.py:2738-2774` 是拿 a225 字段做的**变异测试**，
  证明 211 发射器会拒绝带 225 合同的 lineage——它们必须继续拿着 225 的名字。
- `action_ball_211_transition_preflight.py` 的两张排空清单。
- 历史 artifact：`configs/action_ball_n1_measured_20260803/a225_take061u04_fixed_question_lineage.v1.json`
  及其 rematerialized 副本——它们是 0803 那一轮的证据，不是可执行路径。

#### 同批交付的证据（不是只删代码）

新增 `hope_training/whole_body_tracking/tests/test_action_ball_225_family_retired.py`。
**它不是仪式，是补一个真实的洞**：核对过——`train.py` 和 `my_on_policy_runner.py` 那两道门
在 2026-08-06 之前**一个测试都没有**，而删掉 225 发射器等于删掉了唯一会踩到它们的代码路径。
现在谁把那个 set 清空、或者把整个 `if` 删掉，全仓都不会红。

这份收据做三件事：

- 七个已删的**非测试**文件保持删除（两个发射器、materializer、两个 validator、两份 yaml；三个测试文件不钉，删测试本来就该看得见）；
- `hope_env_cfg.py` 的 `obs_mode` 默认值集合里不出现 `a225/c225`（第 2 道门）；
- **拿活值比两道门**：唯一权威是 `action_ball_a211_trainability._LEGACY_MODES`（纯 stdlib、直接 import、
  且被 `test_action_ball_a225_trainability.py` 用真调用逐个跑过），断言 `train.py` 与
  `my_on_policy_runner.py` 两处**内联 set 判据本身**跟它逐字相同。测试期望值**不是第三份手抄**。

读法特意做成"粗一个档次就过不了"：不是 grep 源码里有没有 `"action_ball_a225"`
（那种检查在 set 被清空、甚至整个 `if` 被删掉之后照样通过，因为名字还留在 docstring 和错误信息里），
而是用 AST 定位**判据节点**
`Compare(left=Name("configured_actor_contract"/"runtime_obs_mode"), ops=[In], comparators=[Set(...)])`，
门没了就取不到，名单少一个就不等。

**变异测试(2026-08-06 pod1 实跑,五发全中;不是跑一个自制脚本,是真改源码再跑 `pytest`)**：

| 变异 | 本收据 | 换成 `grep "action_ball_a225"` 这种粗检查 |
| --- | --- | --- |
| M0 不改（对照） | `11 passed` | PASS |
| M1 从 `train.py` 的 set 里删掉 `"action_ball_c225"` 一个名字 | **FAIL** `[train.py:configured_actor_contract]` | **PASS（漏）** |
| M2 整段删掉 runner 的 `if runtime_obs_mode in {...}` | **FAIL** | FAIL |
| M3 删掉同一段，但留一行 `# TODO: re-add the action_ball_a225 ... refusal` | **FAIL** `[my_on_policy_runner.py:runtime_obs_mode]` | **PASS（漏）** |
| M4 某个 EnvCfg 把 `obs_mode` 改回 `"action_ball_a225"` | **FAIL** `test_no_env_cfg_declares_a_225_obs_mode` | 不适用 |
| M5 某份 225 task yaml 复活 | **FAIL** ×2（本收据 + `test_action_ball_task_config.py`） | 不适用 |

M1 和 M3 就是"粗一个档次就过不了"要防的两种真实写法：**名单被削窄**和**门被删但名字还留在注释里**。
两种情况下源码里都还有 `action_ball_a225` 这个字符串，grep 一样绿。

> **2026-08-07 独立复跑：M1 / M2 / M3 三条由另一轮在独立 pod clone 上重做过一遍，
> 三条全红，还原后对照绿；四道门也逐道核了"今天真的在跑"。** 见 §5.6.18 一与四。

同批还替换了一条本来就在读被删 yaml 的测试：
`test_action_ball_task_config.py::test_historical_a225_c225_task_receipts_remain_named_and_distinct`
（旧意图是"225 的 yaml 不许被悄悄改写成 211 车道"）换成
`test_no_shipped_task_yaml_claims_a_retired_225_actor_contract` —— 遍历 `cfg/task/*.yaml`，
**任何**幸存 yaml 声称 `actor_obs_contract: action_ball_a225/c225` 都红。yaml 删了以后，
同一个担心的正确形态比原来更强（原来只盯那两份，现在盯全部）。

顺带同批改的门：`test_action_ball_safety_vocabulary_single_source.py` 的硬安全终止并集持有者
从 `10` 份降到 `8` 份——**少的是载体不是覆盖率**，两个 225 发射器随文件一起没了，现役持有者一个没漏；
那条 `assert len(...) == 10` 的数量钉也同批改成 `8`（记录 + 阻断同批，不留半改状态）。

#### 边界

- **不影响 211**：删除集合与 211 无 import 交集；两族真正的共用模块只有 `action_ball_c225_rewards.py`，已明确保留。
- **零指纹需要重钉**：被删的文件从来不在任何 launch claim 的 tracked-source 钉子表里 ——
  `test_launch_action_ball_a211_four_arm_diagnostic.py:1962` 本来就断言
  `action_ball_225_trainability.py` **不在** A211 的 `RUNTIME_SOURCE_PATHS` 里。211 两个发射器
  提到 225 的地方全是 `FORBIDDEN_VALUE_TOKENS` 与 `forbidden_namespace_experiment_names`
  两张**禁用表**，删文件不改它们的值，也不许删。
- **收集面机器核对**：删前删后各跑一次 `pytest --collect-only`，两边 `exit=0`（没有任何模块因为
  文件消失而 import 失败），节点差恰好 `-124 / +12` 且逐条对得上：`117` 条来自三个被删测试模块，
  `4` 条是 `test_commanded_contact_geometry` 对两份被删 yaml 的参数化，`2` 条是安全词表少的两个持有者，
  `1` 条是被替换的 task-config 测试；新增 `11 + 1` 条。这条是"没有别的模块依赖被删文件"的决定性证据 ——
  运行期漏引用会红在执行，import 期漏引用会红在收集，两边都干净。
- **执行面 before/after 对拍（受影响面全量，两棵树各跑一次，pod1 2026-08-06）**：
  删前 `591 passed / 34 skipped / 0 failed`，删后 `483 passed / 30 skipped / 0 failed`。
  **失败集合两边都是空集**，差的 `112` 条与上面的收集差逐条同源。
  受影响面 = 所有引用被删路径的模块 + 三个枚举 `cfg/task/*.yaml` 的模块 + 两个 211 发射器/materializer
  测试 + transition preflight 测试。
  （全仓一万条的整跑当时 pod1 上有五个并发全量套件在抢，两边都没跑完；上面这两条比整跑更能定位，
  因为整跑在那种争抢下会混进超时抖动。）
- **不放宽任何 fail-closed 门**：四道门一道没动，另外补了两道门的测试覆盖。
- **不代签"225 的结论"**：0803 那一轮 225 从未越过 `oracle32`（0 PPO），所以它本来也没有任何 learnability 结论可继承；
  §5.6 里引用 A225 数值的段落照旧只是历史，不是当前基线。

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
base question、seed、211/319 各自 ABI、ActionBall base-safety、death manager weight **`-10`**
（post-dt `-0.2`；**2026-08-07 就地更正**，本句原写 `-300`，那是 2026-08-05 层级对齐之前的值，
见 §5.6.17 矛盾 1）、
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
   并按 `death_penalty` 计一次 post-dt **`-0.2`** 的安全罚——**2026-08-07 就地更正**，
   本句原写 `-6`，那是 weight `-300` 时代的旧值，现役 weight 是 `-10`，见 §5.6.17 矛盾 1）。
   **在此之前没有任何东西在看这个通道。**
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

### 9.2.10 solver profile pin 从"整文件字节"改成"逐符号语义面"（2026-08-07 落地，pod1 host-only）

**人话先说**：那枚让训练在 boot 处硬崩的 `solver profile SHA mismatch`，根源不是有人改坏了题，
而是**这枚锁根本分不清"改了求解器的数学"和"改了注释 / 改了 checkpoint 怎么存盘"**。
它对五份源文件做整文件 SHA。于是三笔跟题目毫无关系的提交各自铸出一枚新锁，
而一条正在跑的课程拿着自己的 manifest 就进不去门了。

同一枚锁在另一个方向上还**不够严**：`strike_spec_torch.py` 里的定向逆解种子函数
`_seed` / `_face_from_angles`，**不在任何一枚 pin 里**（不在 solver 的五份、不在 runtime 的十二份、
也不在离线钉针脚本的七份）。而 `stroke_adapt_torch` 从它 import，定向逆解每一行都要过 `_seed`。
上一轮实测过：改 `_seed` 的镜面律初值就能改掉答案，而 pin 纹丝不动。

本轮把这枚锁**指准**，不是放松。

#### 新锁是什么

`.../mdp/action_ball_solver_semantic_surface.py`（新增，只依赖 `ast`/`hashlib`/`json`，
所以离线钉针脚本能像 runtime 一样把它 host-load 起来）。

- **覆盖清单**：显式列出进指纹的 **204 个符号**（初版 198，2026-08-07 补了 6 个，见
  §9.2.10.2），分布在 **6 份**源码
  （`hope_commands.py` / `continuous_questions.py` / `racket_contact_geometry.py` /
  `stroke_adapt_torch.py` / **`strike_spec_torch.py`（第一次被钉进来）** / `virtual_ball.py`）。
  每个符号的摘要取的是"**剥掉 docstring、跨 Python 版本归一化后的 AST**"，
  所以注释、空行、换行位置、docstring 都不动它；任何表达式、常量、字段顺序的改动都动它。
  刻意不用 `ast.dump`：它的字段集合在 3.8→3.12 之间变过（`Index` 包装、`type_params`），
  同一份源码在不同解释器上会给出不同摘要。
- **排除清单**：显式列出 **95 个**有意排除的符号，每一个带一个理由码，共 **14 种**理由
  （`checkpoint_state_serialization` / `birth_audit_ledger` / `runtime_wiring` /
  `telemetry_and_counters` / `grading_and_observation` / `question_production_sampling` /
  `other_product_line` / `venue_parameter_loading` / `stroke_selector` / `swept_contact_grading` /
  `convenience_accessor` / `self_check_only` / `module_export_list` /
  `overapproximated_name_collision`）。**排除是列举式的，不是默认放行。**
- **三道 fail-closed 的门**（`surface_blockers`，boot 时在比 SHA 之前先跑）：
  1. 五份纯求解器源码里**每一个**符号都必须出现在覆盖或排除清单里 ——
     新加一个函数不分类，直接拒绝启动；
  2. 任何被覆盖符号引用、且能解析到被钉文件里的名字，也必须已分类 ——
     入口开始调一个新助手函数而不分类，直接拒绝启动；
  3. 排除理由里凡是**声称"这条路根本走不到它"**的那几种（`other_product_line` /
     `stroke_selector` / `convenience_accessor`，见 `UNREACHABLE_CLAIM_REASONS`），
     必须真的走不到 —— 某个覆盖符号一旦引用了它，这条理由就是假话，直接拒绝启动。
     这道门是 2026-08-07 补的（§9.2.10.2）：排除理由分"它不在这条路上"（可验证）和
     "它在路上但改不了答案"（要人判）两种，前者以前只是**写在文件里的一句话**。
- **排除清单不进指纹**。这正是收窄的机制：新增一个"存盘/记账/遥测"符号必须被**显式分类**
  （门会开火），但分类完之后 pin 不动，训练不再被无关提交打断。
  反过来，把一个**已覆盖**符号挪出覆盖清单一定会动 pin，因为它的摘要从 payload 里消失了。
- **收据自陈**：离线 pins 文档新增 `solver_semantic_surface`（密封的 payload + SHA）和
  `solver_semantic_surface_declaration`（覆盖了哪些符号、排除了哪些、每条理由的人话）。
  "这枚 SHA 到底保护了什么"不用读源码就能回答。

参考的是本仓已有的正确范式 `action_ball_211_abi.live_source_parity_blockers`：
host-load 真源、逐符号比、fail-closed、收据自陈比了哪些符号 —— 不是又造一个整文件哈希的变体。

`counter_rally.py` / `counter_rally_torch.py` **没有**做过符号级裁定，继续走整文件 SHA，
并在 payload 里以 `unadjudicated_whole_file_sha256` 明说是"未裁定"。这是记账，不是放松。

#### 变异测试：三类都做了

- **等强（必须仍然拒绝）**：21 条变异，覆盖尽调列出的每一类"必须仍能抓到"的改动。
  每条**先断言"这段文本确实在源码里出现过"再断言指纹变了** —— 上一轮 A7 就是 sed 没匹配上
  被误记成"存活"，这次不许再发生。包括上一轮 pod 实测过的四条：
  `strike_spec_torch._seed` 镜面律 `e0 0.5→0.93`（A3）、定点迭代 `3→1`（A4）、
  `virtual_ball.flight_accel` 重力 `×1.05`（A5）、`TEACHER_RATE_BOUNDARY_ABS_TOL 5e-7→0.5`（A6），
  以及 `CONTACT_NORMAL_SPEED_MIN_MPS 1.4→-1000`。还包括三条**在 payload 里根本没有被声明、
  只靠代码存在**的裸字面量：`solver_field_contract` 的字段顺序、`pre_swing_wait <= 1.0`、
  cycle-vs-horizon 的 `1.0e-12`。全部杀死。
- **收窄有效（必须放行）**：纯注释、纯 docstring、换行重排都不动指纹。
  更硬的一条：把**今天**的语义面模块拿去评估四个历史 revision ——
  `423f5409`（现役 manifest 那枚 pin 铸造时的 revision）、`eccb30cd`、`308db7f0`（两笔
  checkpoint 序列化作用域重构）、`3e64bea9`（纯注释）—— **四个指纹完全相同**。
  旧口径下这四个 revision 是四枚不同的 pin。
- **不许自我豁免**：往纯求解器源码里塞一个没分类的函数 → `symbol_unclassified` 拒绝；
  让入口开始调一个没分类的助手 → `referenced_symbol_unclassified` 拒绝；
  悄悄把一个符号移出覆盖清单 → pin 变；同一个符号既覆盖又排除 → 拒绝。

#### 迁移：可审计，不静默

现役 manifest 里那枚是**旧口径**算出来的，一定要迁移一次。迁移由
`scripts/migrate_action_ball_solver_pin_to_semantic_surface.py` 做，它**自己会拒绝**：
只有当"决定题目身份"和"决定答案"的每一个**覆盖**符号在新旧两个 revision 上逐字节相同、
且只有 pin 自己的声明半边动了，它才肯出收据。
（**作用域**：只比覆盖面里的符号。2026-08-07 修正了收据里那句越界的话，见 §9.2.10.1。）

实跑收据：`configs/action_ball_n1_measured_20260806/fresh_core_seed0_20260806_r2/`
`solver_pin_semantic_surface_migration.v1.7e85e97e6c1c.json`

| 项 | 迁移前 | 迁移后 |
| --- | --- | --- |
| `solver_payload.schema_version` | `2` | `3` |
| pin 口径 | 五份源文件的整文件 SHA-256 | 六份源码里 198 个符号的语义面摘要 |
| `solver_profile_sha256` | `9d9a6d09f326d511…` | `c196cf79001df76d…` |
| `semantic_surface.sha256` | 无 | `59f03840b85fc47f…` |
| `physics_profile_sha256` | `aa5c9085f9b48ca6…` | **不变** |
| `contact_geometry.sha256` | `2451e2fa1c29036d…` | **不变** |

**为什么这次迁移不改变题目身份**（收据里逐条列了，不是嘴上说）：
- 比对了 **198 个**覆盖符号，只有 **2 个**动了，都是这枚 pin 自己的声明半边：
  `action_ball_solver_profile_contract` 和 `_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION`。
- 三个"给题目取名字"的函数摘要**逐字节相同**：`_action_ball_exact_question_payload`
  = `42a9ec9de40f…`、`_action_ball_semantic_levels` = `5033a5ea95b4…`、
  `_action_ball_canonical_sha256` = `64c3cf96bd4a…`。
- 球的物理和精确面接触几何两枚 SHA 不变。
- `423f5409..d4e1e70c` 之间碰过 mdp 目录的提交只有四笔：`eccb30cd`、`308db7f0`、`16b842d8`、
  `3e64bea9`，加上本轮这笔。这就是"只有 2 个符号动"的原因。

**所以这是重签名，不是重画题。** 内容寻址的那 13 份产物（task receipt / immutable tape /
prototype / manifest / bundle / lineage）文件名里带着自己的摘要，必须由离线流水线按新 pin
重新物化 —— 但**物理题目身份不动**，这正是上面那份收据认证的东西。
迁移脚本**不会**去偷偷改写它们；它只负责把"可以重签"这件事连同证据一起立字据。

新的 v3 pins 文档已随收据落在同一目录：
`action_ball_profile_pins.live.v2.5564d5b3c09d.json`
（`source_authority = external_exact_commit_subset_blob_map_v1`，绑定 `d4e1e70c`）。

迁移脚本的门也做了变异验收，三次都真的开火：
`--from-rev origin/main` 与 `--from-rev 739ba275` → 覆盖符号在其中一个 revision 上不存在，
拒绝；把 worktree 的 `flight_accel` 重力改成 `×1.05` 再迁移 →
`this is not a re-signing: … {"virtual_ball.py": ["flight_accel"]}`，拒绝并**点名**。

#### 9.2.10.1 那份迁移收据说了两句它没资格说的话（2026-08-07 就地改准）

收据里原话是"**每一个**命名题目或计算答案的符号在两个 revision 上都相同"。两处越界：

1. 它只比了**覆盖面里**的符号，95 条排除一个没比 —— 而排除清单正是本轮收窄的产物，
   拿"没比"当"没变"的证据是循环论证。
2. 它对**造产物的那套工具**结构性失明。**实测**：`fresh_tape_seed0_20260806_r1` 和
   `v3pin_tape_seed0_20260807_r1` 两条 N1 tape 是在两个不同版本的 `training_contract.py`
   上建的 —— `8c9eec4c33c54a94…` vs `c6608440c30e1b86…`，两份 build report 里白纸黑字
   （`producer_contracts/*/payload/implementation_source_sha256/training_contract`），
   而迁移脚本一个字都看不到，因为 `training_contract.py` 不在 `PINNED_SOURCES` 里。
   emit 出来的题目数字确实一个没变（548 个 leaf 逐条对拍，64 个差异全是指针和封章，
   `base_question_sha256 = 81eed5139b98` 两边一致），**所以结论不受影响 —— 但话要说准**。

改法：`claim` 加作用域前缀（"Scoped to the symbols this surface COVERS"），新增
`not_claimed` 三条、`coverage_this_receipt_does_not_check`（95 条排除逐个列名 + 为什么它们
不是证据）、`producer_lineage_outside_this_pin`（四份生产工具在两个 revision 上的文件 SHA
**测出来**写进去，动了就在 stderr 里喊一声）。

**`training_contract.py` 该不该进某个 pin？——它已经在该在的 pin 里，且不该进 solver 语义面。**
它不命名题目也不计算答案；五份纯求解器源码一个字都没提它；`hope_commands.py` 提到它
（那是 task-first 训练合同那一摊，26k 行里的另一个 command term），但**没有任何覆盖符号**
引用得到它 —— 这条现在是测试断言，不是判断。它真正的 pin 是 fixed-tape producer contract 的
`implementation_source_sha256`，而那枚 sha **确实**跟着它动了
（`7a91405c…` → `c6a08020…` 等五条 recipe 全动）。所以正确的说法是：
**这枚 pin 在，而迁移收据以前假装自己也覆盖了它。**

#### 回归账（pod1，两棵干净 worktree，都从 `61e71a34` 起）

| 集合 | 基线 | 改后 |
| --- | --- | --- |
| 窄集 12 模块（pin/manifest/stage-evidence/launcher/materializer/adapter/bootstrap + 新增变异模块） | `2 failed / 330 passed / 19 errors` | `2 failed / 365 passed / 19 errors` |
| 宽集 11 模块（curriculum/admission/table-obstacle/lineage/fitted-ball/train-wiring 等） | `17 failed / 349 passed / 9 skipped` | `17 failed / 349 passed / 9 skipped` |

两组的失败与 error **名字逐条相同**，全部是改动前既有（`test_materialize_a3_vendor_identity_manifest`
的 19 个 error、`test_audit_action_ball_cross_engine_physics` 与
`test_mujoco_teacher_motion_native_ball_diagnostic` 各 1 个 formal-authority 断言等）。零回归。

#### 还没关的洞（明写，别当成已解决）

- **R2**：`action_ball_runtime.derive_action_teacher_timing` 家族算 teacher rate 与 pre-swing wait，
  solver payload 的四条拒绝理由直接来自它，但**该文件仍不在语义面里**。
  它现有的 `RUNTIME_CONTRACT_SHA256` 是对一段手写声明取 sha，数学改了它照样不动。
- **R4**：语义面看得见 cfg 旋钮的**默认值**（本轮把 `cq_*` / `vb_rollout_*` / `mount_normal_sign`
  的声明纳入覆盖），但看不见"发射时用 YAML 覆盖了旋钮却没给钉针脚本 `--override`"。
  这是现存洞，本轮不改善也不恶化。
- **R5**（2026-08-07 已关，见 §9.2.10.2）：`action_ball_solver_profile_contract` 里
  `minimum_mps_inclusive: 1.4` / `maximum_mps_inclusive: 7.2` / `net_margin_m: 0.05`
  仍然是**字面量**（离线钉针脚本按 git revision 的源码文本铸这份 payload，
  builder 去 import 活体常量反而会铸出一枚"被钉的那个 revision 并不描述"的 pin），
  但"两者是否一致"**不再靠人看**：boot 时
  `action_ball_assert_solver_runtime_matches_declaration` 拿它们逐个和活体常量比，
  不等就拒绝出题。
- **R9**：打分侧（`_action_ball_exact_achieved_contact_state`、`torch_swept_selected_face_contact`、
  `classify_action_ball_contact`）被显式排除，理由码写进了收据。
  "题没变但分变了"仍然可能，那归 reward/grading 合同管，别指望 solver profile 拦。

#### 9.2.10.2 上面那份收窄**自己漏了一个洞**：声明与实际之间那条传递线（2026-08-07 修，pod1 host-only）

**人话先说**：pin 封的是"**payload 里声明的数字**"。真正喂给求解器的数字要经过一条
**传递线**，而那条线住在 `RacketTargetCommand._initialize_action_ball_runtime` ——
1731 行接线，以理由码 `runtime_wiring` 被排除，理由原文是"只是选绑哪种题源并构造 adapter"。
**这句话不成立**：该函数里第 5343 行是**把被钉住的旋钮交给求解器的唯一一处映射**：

```python
solver_cfg = ContinuousQuestionCfg(
    tol_m=float(self.cfg.cq_tol_m), n_iters=int(self.cfg.cq_n_iters),
    speed_budget=float(self.cfg.cq_speed_budget),
    max_redraw_rounds=int(self.cfg.cq_max_redraw_rounds),
    fixed_direction=True)
```

旋钮的**默认值**是 covered、payload 里也声明了数字，**但这条传递线是自由的**。
独立验收实测三条变异**全部逃逸**（blockers `[]`、pin 不变 `c196cf79`、boot 门放行），
而**旧的整文件 SHA 三条全拦**：

| 变异 | 旧口径（整文件 SHA） | 收窄后（本节修复前） | 真实后果 |
| --- | --- | --- | --- |
| `tol_m × 0.5` 且 `n_iters + 5` | 拦 | **逃逸** | 静默改答案 |
| `speed_budget × 2.0` | 拦 | **逃逸** | 静默改答案 |
| `fixed_direction=True → False` | 拦 | **逃逸** | 不静默：`solve_proposals` 的前置 `_validate_external_proposals` 当场拒("solve_proposals is fixed-direction only")，第一次补池硬崩；但 payload 里 `"fixed_direction": True` 这句**声明**变成假话而 pin 一动不动 |

第三条要说准：`generate` 那条会分叉到 free-direction 的路**不是** action-ball 走的路
（`solve_proposals` 无条件走 `_solve_fixed_direction_batch`）。它的伤害是"声明变假话"，
不是"悄悄换了求解线路"。

**这是本轮收窄自己引入的 fail-closed 净放宽，不是遗留问题。**

**修法（三件，都窄到可以整个进覆盖面）**

1. `action_ball_declared_solver_knobs(cfg)` —— 那五个旋钮的**唯一出处**。
   `action_ball_solver_profile_contract` 拿它写 payload 的 `solve` 块，boot 自检拿它对活值。
   期望值不允许有第三份手抄。
2. `action_ball_solver_cfg_from_declaration(cfg, ContinuousQuestionCfg)` ——
   "旋钮 → 求解器 cfg"的**唯一一处映射**。搬出 1731 行接线，进覆盖面，改一个字 pin 就动。
3. `action_ball_assert_solver_runtime_matches_declaration(...)` —— 逐字段 fail-closed 自检：
   把封好的 solver / physics payload 和**活着的** `solver_cfg`、`prm`、三个平面、
   rollout `h`/`n_steps`、有效 overdraw / 重抽轮数比对，**29 个字段**，不等就拒绝。
   （29 是补池那个调用点的数；另外两个入口点一次解完、不重抽，不比 overdraw 与
   重抽轮数，是 **27 个**。原文写的 31 是错的，2026-08-07 就地改准，并给
   `compared_field_count` 补了一条**把数字钉死**的测试 —— 原来那条只断言它等于
   字段名列表的长度，自己和自己比，数错了永远不会红。这个数在 §9.2.10.3 之后又变了，
   见那一节。）
   诊断跑（`diagnostic_unauthorized`）只允许两处偏离，且两处都是**具名常量**
   （overdraw `1.0`、`_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS`），
   不是自由表达式。返回一份自陈收据：哪个调用点跑的、比了几个字段、字段名逐个列出。

**载重的是调用点，不是 boot**：boot 里也调一次（坏接线死在启动而不是死在第一次补池），
但真正堵死洞的是**三个入口点里各调一次** ——
`_action_ball_refill_pool_many` / `_action_ball_replay_emitted_tasks` /
`_action_ball_frozen_eval_solve` 都在覆盖面里，所以**把调用删掉这件事本身也会动 pin**。
只放在 boot 里等于把门装在排除区，删掉不留痕。

**同类洞系统性扫过一遍**（判据：payload 声明了一个值，而代码里存在一条把别的值喂进去的路径）。
方法是机器扫，不是靠读：把三个入口点读的 `self.*` 属性全列出来，看哪些只由被排除的接线函数写。
结果 —— `_action_ball_solver_cfg`、`_action_ball_effective_cq_overdraw`、
`_action_ball_effective_cq_max_redraw_rounds`、`_action_ball_planes`、`_action_ball_prm`
五个都是同一个病，现在全部进了自检的比对表；
`integrator.h_s`/`n_steps` 三个入口点是直接读 `self.cfg` 的，本来就在覆盖面里（一并比，成本为零）。
95 条排除逐条查了可达性。**门 3 到底管哪几条理由码，以 `UNREACHABLE_CLAIM_REASONS`
的真实内容为准：`other_product_line` / `stroke_selector` / `convenience_accessor`
这三条。**（原文这句写成 `other_product_line` / `convenience_accessor` /
`self_check_only`，两处都错：`stroke_selector` 那 7 个符号**在**门 3 的管辖里
——它们只被 `select_stroke_batch` 引用，动作 ball 的动作逐 episode 冻结，选择器走不到；
而 `self_check_only` 那两个符号**不在**门 3 管辖里，它们的理由是"只报不判"、
不是"走不到"，所以门 3 不该、也没有对它们开火。2026-08-07 就地改准。）
这三条结论现在由**门 3** 执行，不再是一句话。反过来，**确实**被覆盖闭包碰到的排除符号
（`_action_ball_note`、几个 `state_dict`/`load_state_dict`、几个同名碰撞）由
`semantic_surface_declaration` 的新字段 `excluded_but_reached_from_covered` **逐条列出来**：
它们靠的是更强的那句"够得着，但改不了答案"，读的人有权知道是哪几个。

**变异测试（两边都做了）**

- **必须拦（新增 9 条，原有 21 条里有 1 条随 payload 字面量改写替换掉，净 29 条）**：映射里 `tol_m × 0.5`、`speed_budget × 2.0`、
  `declared_knobs` 里 `n_iters + 5`、`_ACTION_BALL_SOLVER_FIXED_DIRECTION` 翻 `False`、
  把自检的比较式改成 `if False`、把 refill 的 `overdraw`/`maximum_rounds` 改成不比、
  从 frozen evaluator 里整段删掉自检调用、从 `_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES`
  里删掉 `paddle_mu`、诊断轮数常量 `64 → 6400`。**全部杀死。**
- **运行时自检本身的变异（`test_action_ball_runtime_wiring.py` 净增 14 条）**：把三条逃逸变异
  施加在**活对象**上（也就是"接线把别的数喂进来"这个真实形状），自检逐条报出字段名；
  平面 `net_top_z` 漂 1 cm、`prm.paddle_mu × 1.5`、overdraw 翻倍、重抽轮数 +1、
  三个 payload 字面量各自和活体常量分家 —— 全部拒绝并点名。
- **必须放行（保住本轮收窄的目的）**：纯注释、纯 docstring、换行重排仍不动指纹；
  `423f5409 → eccb30cd → 308db7f0 → 3e64bea9` 四个历史 revision 仍是**同一枚**指纹
  （比较时按名字剔除本轮**新写进源码**的 5 个符号；`_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS`
  一直都在，只是这次才进覆盖清单，所以照比不误。剔除名单写死在测试里，不是"取交集"，
  否则将来删一个符号就能悄悄缩小这条历史断言）。

**pin 因此第二次移动**（这是加覆盖必然的代价，不是回归）：

| 项 | §9.2.10 落地时 | 本节落地后 |
| --- | --- | --- |
| `semantic_surface.sha256` | `59f03840b85fc47f…` | `5fb9e472bc5fe76c…` |
| 覆盖符号数 | 198 | 204 |
| `solver_profile_sha256` | `c196cf79001df76d…` | `dad9c1c853e4e77a…` |
| `physics_profile_sha256` | `aa5c9085f9b48ca6…` | **不变**（`_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES` 这次重构逐字节等价） |

#### 回归账（pod1，两棵独立 worktree，都从 `20303a3d` 起；解释器 `/workspace/hope_isaac_venv/bin/python`）

| 集合 | 基线 | 改后 |
| --- | --- | --- |
| 受影响 82/83 个模块（逐模块单独起一个 pytest 进程；整目录一次跑会撞上几个测试自己装的合成包，那是既有现象） | `1683 passed / 19 failed / 31 skipped / 52 errors` | `1718 passed / 17 failed / 31 skipped / 52 errors` |
| 直接改到的 5 个模块（surface / runtime-wiring / bundle-materializer / migrate-receipt / pinner） | `116 collected, 2 failed`（`0 skipped`） | **`145 passed`**（`0 failed / 0 skipped`） |

`failed` 少的那 2 条是 `test_action_ball_runtime_wiring` 里**从 `d4e1e70c` 起就死了**的两条
（`action_ball_solver_profile_contract` 加了 `semantic_surface` 必填参数，测试没跟着改，
`TypeError` 直接崩）—— 一条死掉的护栏等于没有护栏，本轮顺手修好并给它们补了
schema-v3 的冻结摘要与 `unadjudicated_whole_file_sha256` 断言。
其余 17 个 failed 与 52 个 errors **逐模块逐条同名**，全部是改动前既有
（`test_materialize_a3_vendor_identity_manifest` 的 19 个 error、
若干模块整目录/合成包 collection error、两条 formal-authority 断言等）。**零回归。**
`0 skipped` 这一格要连着解释器一起看：`1 skipped + 退出码 0` 看着全绿其实什么都没跑，
所以这五个模块是**全跑全绿**，不是"跳过全绿"。

另外一条顺带的自证：把本节的**注释和 docstring** 改完之后重新铸 pin，
`solver_profile_sha256` 仍然是 `dad9c1c853e4e77a…` —— 收窄本身在这次改动里是活的。

**因此：`configs/action_ball_n1_measured_20260807/` 那 18 份内容寻址产物现在是过期的**，
A/C 四个 cell 在重签之前起不来 —— 流程与 `ac8430b0` / `64036cb1` 完全一样
（离线 re-pinner `prepare` + `variants`，再重物化两条 lineage），**本轮没有做**。
离线 re-pinner 的门是**对的**：它现在会点名拒绝
"the template's sealed solver semantic surface … is not the live one"。
`test_materialize_measured_action_ball_n1_bundle` 里那条不动点断言也因此改了口径 ——
模板改成**当场用钉针脚本铸**，而不是读一份发出去的谱系文档：
把"发出去的谱系此刻能不能启动"和"离线重签是不是不动点"捆在一条断言上，
每次 pin 合法移动都会让后者莫名其妙地变红。

#### 9.2.10.3 独立证伪找到四条新逃逸：三条是"指纹看不见入参"，一条是"门不验自己"（2026-08-07 修，pod1 host-only）

**人话先说**：§9.2.10.2 补的那道自检，堵的是"**声明的数字**和**喂进去的数字**不一样"。
独立证伪把口径换了一个：不改数字，改**入参**、改**绑定**、改**payload 本身**。
四条全部实测逃逸 —— blockers `[]`、`semantic_surface` 仍是 `5fb9e472…`、四个调用点全放行。

| # | 逃逸怎么做 | 为什么指纹看不见 | 真实后果 |
| --- | --- | --- | --- |
| 1 | 在**被排除**的接线里把 `_ActionBallPoolSolverAdapter` 的 `solve=`/`solve_many=` 改绑到一个未覆盖、不跑自检的新方法 | 新方法**只被排除区引用**，门 2 的闭包够不着它；三个覆盖入口点的函数体一个字没变 | 整条出题路被接管，覆盖入口点**根本不再运行** |
| 2 | `prototypes.speed_max.mul_(1.5)` / `speed_min.mul_(0.5)` / `face_sign.neg_()` / 保范数的 `v_hat_b` 旋转 | `protos` 是 `solve_proposals` 的**入参**、不是符号；两份 payload 都没声明它，29 项自检也没比过它 | 静默改答案（`face_sign` 那条是把每个答案的**物理拍面翻过来**） |
| 3 | 旋转 `ref_normal`（缩放无效，求解器会重新归一化） | 同上，是入参；而且产它的 `RacketTargetCommand._ensure_reference_strike_state` **既不在 COVERED 也不在 EXCLUDED** —— 覆盖符号只碰属性名 `_ref_racket_normal_raw_w_per_clip`、不碰产它的方法名，闭包永远够不着 | 静默改答案 |
| 4 | 铸造之后**同时**改 sealed payload 和活 cfg（两行） | 自检只校验 `kind` 和 `schema_version`，**从不重算** `_action_ball_canonical_sha256(payload)` 去和 `manifest.solver_profile_sha256` 对 | 29 个字段照过，四个调用点全放行 |

第 1 条要说准，它是**内容指纹的固有边界**，不是这次实现的疏忽：
**pin 能证明"这三个函数体没变"，证明不了"跑的就是这三个函数体"。**
把覆盖面越扩越大解决不了它 —— 扩到哪儿，绑定都在被扩的那一层之外。
所以补的是**别的机制**（运行时自证 + 绑定表达式列举），不是更大的面。

**修法**

1. **门自己先验身份**（堵 4）。`action_ball_assert_solver_runtime_matches_declaration`
   现在多比两项：`_action_ball_canonical_sha256(solver_declaration)` 对
   `manifest.solver_profile_sha256`（内容寻址，由 launcher 的
   `action_ball_manifest_sha256` 钉住），`_action_ball_canonical_sha256(physics_declaration)`
   对 solver payload 里自陈的 `physics_profile_sha256`。改 payload 就必须连 manifest 一起伪造。
2. **把两个入参写进声明并进比对表，且摘要从活张量重算**（堵 2、3）。新增
   `action_ball_live_answer_input_digest`（原型表 24 列 + 逐 clip 参考法向表，
   一次主机传输，float64 精确）和 `action_ball_answer_input_contract`（封 + 锚）。
   `StrokePrototypeTensors.derived_sha256` 帮不上忙 —— 它是**加载时**对 JSON 记录取的，
   之后没人从活张量重算过，这正是逃逸 2 能成立的原因。
   **"锚"这一半单独说**：原型表里有三件事是 manifest 说了算的（家族顺序、逐动作
   `face_sign`、"这个动作还开着"），加上 `speed_min ≤ cq_speed_budget`。
   这四条检查以前**长在那条被排除的 1731 行接线里**，删一行没人知道；现在搬进覆盖面。
   于是 `face_sign.neg_()` 是**两头都堵**的：封之前改被锚拦，封之后改被活值摘要拦。
   其余几列只有"封之后"那一半，**这句要说准**。
3. **覆盖入口点当场自证身份**（堵 1，运行时半边）。
   `action_ball_assert_solver_adapter_binds_these_entry_points` 让入口点问适配器一句
   "你手里握的七个槽位是不是我"，先比对象同一性、再比 `__func__`/`__self__`；
   少一个槽位也拒。补池那个入口点 `required=True`（它只可能在在线求解路上跑），
   另外两个 `required=False`（不可变题带 / 题库题源根本不构造这个适配器，
   它们的身份由自己的 tape/bank SHA 钉住）。
4. **语义面新增两道门**（堵 1 的静态半边 + 堵"未分类符号"本身）。
   - **门 4（产出者闭包）**：凡是**写了某个覆盖符号读的实例属性**的符号，必须已分类。
     门 1/2 都按"提到的名字"做闭包，而覆盖符号提到的是属性名、不是产它的方法名，
     所以闭包**结构上**够不着产出者 —— `_ensure_reference_strike_state` 就是这么漏的。
     实现上刻意不用 `ast.walk` 走赋值目标：那样 `self._by_uid[key.action_uid] = row`
     会被读成"这个符号写了 `action_uid`"，门就被噪声淹了；下标的 **value 跟进、slice 不跟**，
     所以 `self.x[i] = v` 这种**原地写**照样算（那正是取过一次的摘要看不见的形状）。
   - **门 5（绑定列举）**：`_ActionBallPoolSolverAdapter` 的五个槽位只能绑
     `POOL_SOLVER_BINDINGS` 列举过的表达式，且一个槽都不许缺。清单**不进指纹**，
     所以加一条合法绑法不作废题库，但加这一条必须是有意识的一次编辑。
     `ast.unparse` 是 3.9+ 而 host 测试环境是 3.8，所以表达式渲染是手写的受限形式
     （`self.method` / `name`），其余形状退回到那份跨版本归一化的 AST 摘要。
5. **顺带修一条已核实的错绑**：`_action_ball_decode_solver_mutable_state` 里造暂存
   采样器时传的是 `self._action_ball_fixed_view_enabled`，同文件另外五处都用
   `self._action_ball_diagnostic_unauthorized`。
   **实测方向和上一轮口头描述的相反，这里按实测写**：
   `immutable_tape` 是**诊断专用**（`action_ball_target_source='immutable_tape'` 在
   非诊断跑当场拒），所以"正式 + fixed-view"根本不存在；真正会踩的是**诊断跑但不是
   fixed-view**（在线求解 + 4096 环境那一档）：那时 `fixed_view_enabled=False`
   → 暂存采样器被建成 `exact` scope，而存档里签的是 `diagnostic` scope，
   `load_state_dict` 当场拒（"sampler state transcript scope mismatch"）。
   这条路是池子回滚/续跑要走的，所以是活的缺陷，不是理论缺陷。

**`_ensure_reference_strike_state` 分类成 EXCLUDED，理由码是新的，而且它说的是实话。**
`reference_strike_state_production` 的原文明说：**它能改答案**（旋转那张表就行），
之所以不进覆盖面，是因为覆盖它会把整个参考 FK 家族（击球帧选取、body 状态读取、
mount 偏移，连同它那段诊断打印文本）一起拖进指纹，一次无关编辑就作废整个题库。
拦它的是活值门（封之后改，三个入口点每次出题前重算都会发现）。
**封之前改 —— 也就是改 `_ensure_reference_strike_state` 自己的数学 —— 仍然不动 pin。
这是一个具名的开口（下面 R10），不是已解决的问题**，理由码原文里也这么写着，
测试 `test_the_exclusion_that_can_move_an_answer_says_so_in_its_own_words` 读的就是那段文字。

**变异测试（两边都做了，一条都不许少）**

- **必须拦（语义面，净增 7 条，共 36 条）**：从
  `_ACTION_BALL_ANSWER_INPUT_PROTOTYPE_COLUMNS` 删一列、活值摘要不再覆盖参考法向表、
  取消 `face_sign` 的 manifest 锚、把 payload 摘要那条比对改成自己对自己、
  整条删掉活值摘要比对、把适配器自证的 `if drift:` 改成 `if False:`、
  从补池入口点整段删掉自证调用。**全部杀死。**
- **必须拦（新门本身）**：把 `solve_many=` 改绑到 `self._action_ball_smuggled_many`
  → `pool_solver_binding_undeclared` 拒；整个删掉 `solve=` 槽位 →
  `pool_solver_binding_absent` 拒；把 `_ensure_reference_strike_state` 从排除清单里拿掉
  → `attribute_producer_unclassified` 拒；新写一个也写
  `_ref_racket_normal_raw_w_per_clip` 的方法而不分类 → 同样拒。
- **必须拦（运行时，`test_action_ball_runtime_wiring.py` 净增 11 个函数 / 13 条用例）**：
  在**活对象**上改 `speed_max ×1.5`、`speed_min ×0.5`、`v_hat_b` 取反、
  参考法向表旋转 → 全部报 `answer_inputs.live_digest_sha256`；
  doctored payload + 配套改活 cfg（每一个**数字字段都对得上**）→ 只有
  `solver.payload.canonical_sha256` 报出来，这正是逃逸 4 的形状；
  改物理 payload → `physics.payload.canonical_sha256`；
  `face_sign` 翻面、原型文件 SHA 不是 manifest 钉的那份 → 封的时候就拒；
  适配器换了 `solve_many` / 少一个槽位 / 根本说不出自己握着谁（`required=True`）→ 拒。
- **必须放行（保住收窄的目的，一条都没牺牲）**：纯注释、纯 docstring、换行重排仍不动指纹；
  `423f5409 → eccb30cd → 308db7f0 → 3e64bea9` 四个历史 revision **仍然是同一枚指纹**
  `680aa87d84323740…`，且四个 revision 上**新加的门 4、门 5 都是 0 blocker**
  （比较时按名字剔除本轮新写进源码的 6 个符号，剔除名单写死在测试里、不是取交集）。
- **数字钉死**：`compared_field_count` 现在有一条**手钉数字**的断言
  （补池 36、另外两个入口点 34）。原来那条只写 `count == len(fields)`，
  自己和自己比，所以 §9.2.10.2 文里那个错的"31"一直没人发现。改门必须连这个数字一起改。

**pin 因此第三次移动**（加覆盖的必然代价，不是回归）：

| 项 | §9.2.10.2 落地后 | 本节落地后 |
| --- | --- | --- |
| `semantic_surface.sha256` | `5fb9e472bc5fe76c…` | `d293eed4474846bc…` |
| 覆盖符号数 | 204 | **210** |
| 排除符号数 / 理由码种数 | 95 / 14 | **98 / 15** |
| fail-closed 的门 | 3 道 | **5 道** |
| 自检比对字段数（补池 / 另外两个入口点） | 29 / 27 | **36 / 34** |

`solver_profile_sha256` 跟着 `semantic_surface.sha256` 一起动（它把后者写进 payload），
所以**内容寻址产物又一次过期**，重签流程与 `ac8430b0` / `64036cb1` 一样，**本轮没有做** ——
和 §9.2.10.2 一样，重铸必须排在逃逸修完之后，否则要铸两遍。

#### 回归账（pod1，两棵独立 worktree，都从 `f880b5df` 起；解释器 `/workspace/hope_isaac_venv/bin/python`）

| 集合 | 基线 | 改后 |
| --- | --- | --- |
| 直接改到 / 直接相关的 7 个模块（surface / runtime-wiring / pinner / migrate-receipt / measured-bundle / stage-evidence / contact-bundle） | `214 passed`（`0 failed / 0 skipped`） | **`240 passed`**（`0 failed / 0 skipped`） |
| 另外 4 个引用语义面的模块（launch-n1-screen / table-obstacle / canonical-admission / train-wiring） | `13 failed / 214 passed` | `13 failed / 214 passed`，**逐条同名** |

多出来的 26 条全是本节新增的变异与门测试（语义面 `46 → 59`，runtime-wiring `76 → 89`）。
`tests/test_action_ball_train_wiring.py` 那 13 条失败**两棵树逐条同名**，是改动前既有
（`test_action_ball_manifest_order_and_physical_truth_fail_closed`、
`test_action_ball_allows_only_pinned_solver_cq_knobs` 等），本节没碰它们。**零回归。**
`0 skipped` 这格要连着解释器一起看：`/usr/bin/python3` 缺 hydra 会静默跳过一批，
`1 skipped + 退出码 0` 看着全绿其实什么都没跑；上面两组都是**全跑全绿**。

#### 还没关的洞（本节新增，明写）

- **R10**：`_ensure_reference_strike_state` 自己的数学（击球帧选取、参考 FK、
  `mount_normal_axis` 取列）**不在指纹里**。改它能改掉每一个答案而 pin 不动。
  现在它至少是**被分类、被列名、理由码里明说"能改答案"**的，
  并且 `semantic_surface_declaration` 的新字段 `attribute_producers` 会把它连同
  它写的属性一起列出来。要真关上它，得把整个参考 FK 家族纳入覆盖面（代价：
  一次无关编辑作废题库），或者把 quat→法向那一步抽成一个窄函数单独覆盖。本轮两者都没做。
- **R11**：`stroke_prototypes_torch.load_stroke_prototype_tensors` 自己怎么把
  那份被 SHA 钉住的 JSON 变成张量，**不在 `PINNED_SOURCES` 里**。
  JSON 的字节和记录摘要都钉住了，读法没钉。本节的活值门管的是"加载之后再改"，
  不是"加载时就读错"。

### 9.2.11 迁移做完了：离线重签这一步自己还是 v2，所以"重新物化"生不出能启动的 manifest（2026-08-07 落地并实跑验证）

**人话先说**：§9.2.10 已经把新锁做好、收据也开好了，结论是"那 13 份内容寻址产物
要由离线流水线按新 pin 重新物化"。**照做了，还是崩**——重新物化出来的 manifest 拿着
`d26aaace…`，而 runtime 现算的是 `c196cf79…`。原因不在锁，也不在迁移脚本，
**在离线流水线自己**：它重签的那一步还在按 v2 的规矩盖章。

#### 断点在哪一行

`scripts/materialize_measured_action_ball_n1_bundle.py` 的 `_materialize_live_profile_pins`。
它的活是"把 pins 模板按活体源码重新封章"。v2 时代 payload 的身份就是七份源文件的
**整文件 SHA 映射**，所以它做的唯一合理的事就是：把活体映射写进
`solver_payload.implementation_source_sha256`，然后重算封章 SHA。

v3 的 payload **根本没有这个键**。它封的是逐符号语义面，加上 counter-rally 两份
**未裁定**源码的整文件 SHA。继续按 v2 塞，等于在密封件里多按一枚指纹——
离线铸出的 pin 与 runtime 现算的那枚**永远差一个键**。

**实测，不是推断**：同一份 v3 pins 文档 `action_ball_profile_pins.live.v2.5564d5b3c09d.json`，

| 谁来铸 | 铸出的 `solver_profile_sha256` |
| --- | --- |
| `pin_action_ball_profile_contracts.py`（钉针脚本） | `c196cf79001df76d…` |
| runtime boot 门现算 | `c196cf79001df76d…` |
| `_materialize_live_profile_pins`（旧口径） | `d26aaace c335d143…` |

两份 solver payload **逐键只差 `implementation_source_sha256` 一项**，去掉它逐字节相同。
上一轮在 pod 上留下的 `fresh_core_seed0_20260807_r3` 就是这么来的：manifest 换了个数字，
一样起不来。

#### 改法：按 payload 自报的 schema 分叉，三条路都 fail-closed

- `schema_version == 2`：维持整文件字节映射，**一行没动**（老谱系、老测试继续走）。
- `schema_version == 3`：
  1. 从**活体**源码重建语义面（`semantic_surface_contract` 自带那两道覆盖门，
     有未分类符号就直接拒绝铸 pin）；
  2. 只把 counter-rally 那两份未裁定源码的整文件 SHA 刷成活体值——runtime 也是这么算的，
     **不刷才会对不上**；
  3. 模板封着的语义面**只要不是活体那份就拒绝**——那是有覆盖符号动了，
     属于"改题"不是"换章"，必须走 `migrate_action_ball_solver_pin_to_semantic_surface.py`
     并留收据；
  4. 模板声称 v3 却仍在密封 payload 里带着 v2 字节映射 → 拒绝，明说"这不是一枚 v3 pin"。
- 其他 schema：**点名拒绝**，不再悄悄按 v2 盖章。

注意第 3 条是**收紧**：v2 时代这一步会默默吸收源码漂移，正是它让 manifest 反复过期。

#### 这支函数此前一条测试都没有——这就是错法能活下来的原因

现在六条（2026-08-07 加了一条，见下）。载重的一条是**不动点**断言：把活体 v3 文档喂进去，
吐回来的 `solver_profile_sha256` 必须**还是那一枚**（也就是 runtime 现算的那枚）。
2026-08-07 把这条的模板从"读一份发出去的谱系文档"改成"**当场用钉针脚本铸**"，
并把镜像源码放进临时根：原写法把"发出去的谱系此刻能不能启动"和"离线重签是不是不动点"
捆在同一条断言上，于是每次 pin 合法移动（比如 §9.2.10.2）都会让后者莫名其妙地变红，
红的原因跟它要测的东西无关。新增的第六条测的是**模板封着的语义面过期时必须点名拒绝**
（`is not the live one`）—— 那正是每条谱系在重签完成之前所处的状态。
拒绝那条的变异只动**一个覆盖符号的函数体**（`virtual_ball.flight_accel` 重力 `×1.05`），
被钉源码清单、覆盖符号名、覆盖符号数**全都不变**——**粗一个档次的检查**
（比个数、比名字、比文件清单）会直接放行。把旧行为塞回去，五条里**死两条**
（不动点那条 + "拒绝混血文档"那条），pod 实测过。

#### 重签账：动了什么，逐字段量出来的

三次物化都在干净 checkout 里跑（`materialize_action_ball_*_lineage.py` 会**拒绝脏树**，
所以产物和谱系分成两笔提交）。

| 产物 | 叶字段数 | 变了几个 | 变的是什么 |
| --- | --- | --- | --- |
| action manifest | `182` | `3` | `solver_profile_sha256`，以及由它派生的 `prototype.path` / `prototype.sha256` |
| task receipt（5 份同形） | `227` | `7` | `solver_sha256`、`manifest_sha256`，以及由这两个派生的 5 枚封章 |
| c211 lineage | `62` | `6` | 三个指针（manifest / task receipt / bundle）的 path+sha |
| c211 bundle | `58` | `5` | 两个指针 + 自己的 `content_sha256` |

球的 profile、curriculum、counter-rally objective、holdout、landing aim、motion 绑定、
task profile、actor/critic 宽度与 normalizer 身份、question-RNG 政策、DR-L0 绑定、
trainability 合同——**全部逐字节相同**。

**题目身份没动**：`offline_n1_tape_build_report` 里 `base_question_sha256` 前后都是
`81eed5139b98…`，在它自陈的 **18 个**位置上逐个比过。

**同时要明写一句反话，别让人引错字段**：`canonical_sha256` **是变了的**
（`94b79245…` → `697227c8…`），因为它按构造就封着 solver pin。
"题没变"的证据是 `base_question_sha256` **加上** 那 220 个没动的物理字段，
**不是** `canonical_sha256`。拿 `canonical_sha256` 当同一性证据就是引错了字段。

另有两处诚实记账：prototype 里 `provenance.producer_source_sha256` 变了（本轮改了 producer），
`full_solver_admission_preflight` 下 `action_ball_curriculum` / `action_ball_sampling`
两枚运行时模块摘要也变了——它们在 `92c4ce94`、`4c9cf280` 动过，而 r2 核心是在那之前封的。
这是"这份核心是什么时候铸的"的出处记录，不是题目身份。

#### 收据自陈，且"漏没漏"是数出来的不是嘴说的

`configs/action_ball_n1_measured_20260807/solver_pin_v3_re_signing_ledger.v1.bfe7f6bc3a8f.json`
一一列出 **18** 份产物的 before path+sha → after path+sha、各自变了哪些叶字段，
**每个数字都是出账时现量的**，不是抄来的。里面还有一项
`residual_references_to_retired_digests`：对三枚退役摘要（旧 pin / 旧 manifest / 旧 task receipt）
做全树 `git grep` 并分类，结果 **`unresolved_live_reference_count: 0`** ——
剩下的每一条要么是留作历史的 `2026-08-06` 旧谱系，要么是按设计记录 before 值的迁移收据。
**漏一个引用就是下一轮的 boot 硬崩，所以它是数出来的。**

`A211` 谱系也指着同一枚旧 manifest，**同批重签**了，不留半条断链。
`2026-08-06` 旧谱系**原地保留**——崩掉的那些 run 就是拿它发的，删了等于抹掉崩因证据。

#### 迁移脚本本身：三道门真的跑过并通过

`migrate_action_ball_solver_pin_to_semantic_surface.py` 在本轮 checkout 上重跑
（`--to-rev` 指向本轮提交，而不是上一轮的 `d4e1e70c`），**没有拒绝**：
`198` 个覆盖符号在两个 revision 上逐个对拍，只有 `2` 个动了且都是 pin 自己的声明半边，
三个"给题目取名字"的函数摘要逐字节相同。收据落在
`…/v3pin_core_seed0_20260807_r1/solver_pin_semantic_surface_migration.v1.cce63a1bac8c.json`。

#### 实跑验证：C0 recipe 真的过了那道门

pod1 GPU1，干净 worktree `pinv3_isaac_20260807`，`materialize` → `recipe` 两段都 `EXIT=0`，
`terminal_kind = clean_completion`。run.log 里 `solver profile SHA mismatch` 与 `Traceback`
命中数**都是 0**，运行时自陈那行是：

```
[RacketTargetCommand] action-ball runtime bound: actions=1, mobility=no_move,
manifest=bb65ca3f3a8e…, solver=c196cf79001df76d…, physics=aa5c9085f9b48ca6…,
target_source=direct_ball, target_recipe=outcome_dense_only
```

#### 回归账（pod1，`/workspace/hope_isaac_venv/bin/python`，**不是** `/usr/bin/python3`）

| 集合 | 基线（`085df5ec`） | 改后 |
| --- | --- | --- |
| 6 模块（bundle / contact-training-bundle / tape-variants / c211-lineage / semantic-surface / pinner） | `81 passed`，**`0 skipped`**，退出 `0` | `86 passed`，**`0 skipped`**，退出 `0` |
| 2 个发射器模块（c211 / a211 four-arm） | —— | `310 passed`，**`0 skipped`**，退出 `0` |

`+5` 就是本轮新增的五条。**skip 数写清是因为 `1 skipped` + 退出 `0` 看着像绿的、其实什么都没跑**；
这两组都是 `0 skipped`。

#### 还没关的洞（本轮新增，别当成已解决）

- 旧的 `fresh_core_seed0_20260807_r2` / `_r3`（上一轮在 pod 上按旧口径铸的两份）**没有提交**，
  也不该被拿来用；它们的 manifest 拿着 `d26aaace`，一样起不来。
- 本轮只验到 `recipe` 段起得来。`oracle32` / `scale4096` 两段**没跑**，
  它们各自的门是否还有别的过期指针，本轮没有证据。

### 9.2.12 第三次重签：语义面涨到 210 个符号，18 份产物一次铸完，C0 实跑过了 boot 门（2026-08-07 落地并实跑验证）

**人话先说**：§9.2.10.3 修完四条逃逸之后，求解器语义面**又动了一次**——这次是最后一次。
封在产物里的还是 `59f03840…`／`198` 个符号，活体已经是 `d293eed4…`／`210` 个，
于是 `solver_profile_sha256` 从 `c196cf79…` 变成 `3af6c505…`，18 份内容寻址产物全部过期，
四个格子一个都开不了机。本节把它们**一次性**重铸完（刻意不铸两遍：要重铸的那两个函数
本身就在覆盖面里），并且实跑 C0 到 `recipe` 段确认真的过了那道门。
**注意标题说的是 C0**：A 族只重签了谱系，本轮一格都没实跑。

#### 先把数字对齐：中途那个 `204` 不是终值

| 口径 | 语义面 SHA | 覆盖符号数 | `solver_profile_sha256` | 这数从哪来 |
| --- | --- | --- | --- | --- |
| 上一轮**封在产物里**的 | `59f03840b85f…` | `198` | `c196cf79001d…` | 读 `…/v3pin_core_seed0_20260807_r1/action_ball_profile_pins.live.v1.553ef7d14f3c.json` |
| `3a72ea5b` / `f880b5df` 现算 | `5fb9e472bc5f…` | `204` | `dad9c1c853e4…` | 钉针脚本 `--source-rev` 现铸，两个 revision 同值 |
| `42232b84` 现算（**本轮铸的就是它**） | `d293eed44748…` | `210` | `3af6c505f9ed…` | 钉针脚本 `--source-rev 42232b84` 现铸 |

中间那一行说明产物**在 §9.2.10.3 之前就已经过期了**：`3a72ea5b` 的声明桥加了 5 个覆盖符号、
另有 1 个重新分类，`198 → 204`；`42232b84` 的四条逃逸修复再加 6 个，`204 → 210`。
所以"逃逸修完再铸"这个顺序是对的——先铸就要铸两遍，而中间那枚 `dad9c1c853e4…`
**从来没有被任何产物封过**。

第一行只能读、不能重铸：现在的钉针脚本要读
`_ACTION_BALL_SOLVER_FIXED_DIRECTION` / `_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES` 两个常量，
它们是 `3a72ea5b` 才加的，所以在 `28175559` 上跑会直接报
`missing module constants` 而不是铸出一枚假 pin。**这是它拒绝，不是我们抄数。**

#### 迁移脚本被跑了，它**拒绝**了，我们把拒绝原样登出来而不是把门放宽

`migrate_action_ball_solver_pin_to_semantic_surface.py` 在本轮 checkout 上跑了
（`--from-rev 423f5409`，`--to-rev 42232b84`），退出码 `1`，原话是：

```
a covered symbol is missing at one of the two revisions and is not on the
allowed-introduction list, so the invariance claim cannot be checked:
{'hope_commands.py': ['action_ball_live_answer_input_digest',
 'action_ball_answer_input_contract', '_ACTION_BALL_ANSWER_INPUT_SCHEMA_VERSION',
 '_ACTION_BALL_ANSWER_INPUT_PROTOTYPE_COLUMNS',
 'action_ball_assert_solver_adapter_binds_these_entry_points',
 '_ActionBallPoolSolverAdapter.action_ball_bound_entry_points']}
```

它后面还有第二道会拒的：**三个覆盖的求解入口动了**——
`_action_ball_refill_pool_many`、`_action_ball_frozen_eval_solve`、
`_action_ball_replay_emitted_tasks`，因为它们现在在求解前要先调用两份新合同。

**这不是脚本坏了，是它在正确工作。** 它认证的是"纯换章"：schema v2→v3、除了 pin 自己的
声明半边之外没有覆盖符号动过。本轮**根本不是那件事**——我们故意改了求解入口（加了拒绝）
并加了 6 个覆盖符号，所以那条静态不变性断言按构造就是假的，它照实说了。

**允许清单一个字没加。** 把 `_action_ball_refill_pool_many` 加进 `ALLOWED_MOVED_SYMBOLS`
等于**预先批准**那个唯一负责抽题的符号，正好是这道门存在的反面。
台账里 `migration_gate` 一节把脚本的拒绝原文、以及**用脚本自己的常量重算出来的**
introduced／moved／refused 三张表一起登出来——不是第三份手抄。

#### 那三个动了的符号到底有没有改答案：拿活值判，不拿论证判

静态摘要分不清"加了一句拒绝"和"改了一步计算"，所以静态层面这题无解。能判的是活值：
**tape 就是跑 `_action_ball_refill_pool_many` 产出来的**，而
`offline_n1_tape_build_report` 里 `base_question_sha256` 前后都是 `81eed5139b98…`，
在它自陈的 **18 个**位置上逐个比过。抽题没动，这是量出来的。

另外两条同向证据，都是逐叶字段比出来的：

- 两份 pin 模板（上一轮的 `5564d5b3c09d` 与本轮的 `9a909b56d9b6`）之间，
  共有键里**只有 20 个叶字段不同，全部在 solver 一侧**；
  `cfg`（每一个旋钮）、`physics_profile_sha256`（`aa5c9085…`）、
  exact-face contact geometry 摘要**逐字节相同**。
- pins 文档只增不减：`leaf_fields_only_before_count == 0`，
  多出来的 553 个键就是新增覆盖符号的逐符号摘要。

**反话也要明写**：每份 task receipt 的 `canonical_sha256` **都变了**，
而且永远会变——它按构造就封着 solver pin。拿它当"题目没变"的证据就是引错了字段；
拿它当"题目变了"的证据同样是引错了字段。台账的
`what_this_does_and_does_not_claim` 把这句写进了文件里。

#### 漏没漏是数出来的：14 处 pin 副本、56 条指针、三次全树扫描

- **pin 副本 14 处**，分布在**两个字段名**下：`solver_profile_sha256`（6 处：两份 pins、bundle、
  contact_alignment、manifest、prototype）与 `solver_sha256`（8 处：5 份 target receipt、
  base question receipt、immutable tape、tape build report）。
  仍带着退役 pin 的位置：**0 处**。
- **指针链 56 条**（新四个目录里每一个 `{path, sha256}` 对）全部解析成功，`broken: 0`。
  `A211` 谱系同批重签——上一轮差点漏掉它，漏了就是下一次开 A 格的硬崩。
- **三次扫描**，因为任何一次单独都有洞：按**退役摘要**扫（漏掉只写路径不写 SHA 的引用）、
  按**退役完整路径**扫（漏掉只写 SHA 的，也漏掉把路径拆成两行字符串的）、
  按**退役目录名**扫（专门补上一条——`tests/test_migrate_action_ball_solver_pin_receipt.py`
  正好把路径拆在目录边界上，`git grep` 整条路径看不见它）。
  三次合计 **`unresolved_live_reference_count: 0`**。

#### 拆成两笔提交不是洁癖，是被门逼的

`materialize_action_ball_c211_lineage.py` **拒绝脏树**，
`materialize_action_ball_a211_lineage.py` 还额外**拒绝未跟踪的 action manifest**。
所以顺序只能是：core+tape 先落一笔提交（`c35747e9`），谱系才铸得出来、落第二笔（`f39e7869`）；
台账另落第三笔（`92fa48a7`，这一笔是为了好读，不是被门逼的）。
两条谱系的 `--source-commit` 都是 `c35747e9`，
而 `c35747e9` 的求解器源码与 `42232b84` 逐字节相同。

#### 实跑验证：C0 到 `recipe` 段真的过了那道门

pod1 GPU1，worktree `s11_cells_c_20260807`（干净，`92fa48a7`），
`materialize` → `recipe` 两段都 `EXIT=0`、`terminal_kind = clean_completion`，
两段 run.log 里 `solver profile SHA mismatch` 与 `Traceback` 命中数**都是 0**。
运行时自陈那行：

```
[RacketTargetCommand] action-ball runtime bound: actions=1, mobility=no_move,
manifest=12031a74be708b53…, sampler=03d8a77968134b61…,
solver=3af6c505f9ed2333…, physics=aa5c9085f9b48ca6…,
target_source=direct_ball, target_recipe=outcome_dense_only
```

`manifest` 和 `solver` 都是本轮新铸的那两枚，`physics` 没动。

#### 反向对照：拿退役谱系发车，同一个 HEAD 上必须崩

光看"新的能跑"证明不了是重签起的作用——也可能那道门根本没在看。所以同一个 worktree、
同一个 HEAD、同一份 spec，只把 `lineage` 换回**退役**的那份
（`a327854762…`，封着 `c196cf79`），`recipe` 段的 run.log 里是：

```
File ".../mdp/hope_commands.py", line 5894, in _initialize_action_ball_runtime
    raise ValueError(
ValueError: action-ball solver profile SHA mismatch:
manifest=c196cf79001df76d…, runtime=3af6c505f9ed2333…
(… Its live surface is d293eed4… over 210 symbols in 6 sources. …)
```

门是活的，报的两个数正是本节表里的那两个。顺带更正一个行号：这道门在本轮 HEAD 上是
`hope_commands.py:5894`（`_initialize_action_ball_runtime` 里），
早前记的 `5568` 已经漂掉了；判据本身没变。

**这条对照没有退出码可报**：拒绝之后 Kit 进程挂在 shutdown 上不退（run.log 停在 `04:06`，
到 `04:28` 一个字没长），最后是按纪律 `readlink /proc/<pid>/cwd` 确认属于自己之后
`kill` 掉自己的两个 pgid（`319832` / `321315`）收的场，GPU1 已清空。
所以这条的证据是 **run.log 里的那个 `ValueError`**，不是退出码——写清楚免得有人去引一个不存在的数。
崩溃后 Kit 不退这件事是既有行为，不是本轮引入的。

#### 顺带查实的一件事：`materialize` 段**不碰**这道门，所以它不能当开机验证

上面那条反向对照第一次跑的时候只跑了 `materialize` 段，结果是
**`EXIT=0`、`CELL_OK`**——拿着**退役**谱系也一样绿。原因是 `materialize` 段压根没构造
ActionBall env，自然也没算 solver profile。
**只跑 `materialize` 绿了就宣布"开机没问题"是假收据。**
真正开机的是 `recipe` 段——"runtime bound"那行和上面那个 `ValueError` 都只在
`recipe` 的 log 里出现。本节的正向验证因此报的是 `materialize` **和** `recipe` 两段。

#### 回归账（pod1，`/workspace/hope_isaac_venv/bin/python`，**不是** `/usr/bin/python3`）

本轮**一行代码都没改**，只产出 `configs/` 下的产物和这一节文档，所以回归的意义是
"新产物有没有踩到某条会扫 `configs/` 的测试"。

| 集合 | 结果 |
| --- | --- |
| 重铸前（`42232b84`），7 个模块 | `200 passed`，**`0 skipped`**，退出 `0` |
| 重铸后（`f39e7869`），同 7 个模块 **+** `test_migrate_action_ball_solver_pin_receipt.py` | `206 passed`，**`0 skipped`**，退出 `0` |

`+6` 全部来自新加进来的那个模块，不是行为变化。**skip 数写清是因为
`1 skipped` + 退出 `0` 看着像绿的、其实什么都没跑**；两次都是 `0 skipped`。
最后那个模块特意加进来跑，是因为它按名字引用了一份**退役**的 tape build report
（`…/v3pin_tape_seed0_20260807_r1/offline_n1_tape_build_report.v1.58eb977b1566.json`）——
旧谱系原地保留，所以它照常通过；哪天有人清理旧目录，它会第一个红。

#### 还没关的洞（本轮，别当成已解决）

- 本轮只验到 `recipe` 段。`oracle32` / `scale4096` 两段**没跑**，
  它们各自的门是否还有别的过期指针，本轮没有证据。
  `scale4096` 另有已知的独立阻断（§5.6.19），与本轮重签无关。
- A 族只重签了谱系，**一格都没实跑**。"A 族也能开机"本轮没有证据，只有"指针不再断链"。
- 迁移脚本目前只会讲 v2→v3 这一种故事。**真正的 v3→v3 重签它认证不了**，
  本轮是靠"活值 + 逐叶字段"补的证据。要让它以后能认证这一类，得给它一个
  "覆盖符号确实动了、但只加了拒绝"的判据——而那个判据必须自己能被变异测试打红，
  否则就是把门写松。本轮**没做**，只把缺口写在这里。
- §9.2.10.3 记的 R10 / R11 两个洞照旧没关。

### 9.2.13 第六个零调用点的门判了：`assert_known_generation` 删掉，不是接线（2026-08-07）

**人话一句：** §5.6.18 二.2 把它挂起来时给的理由是"存档的退役来历没人管，是自证循环"。
**这句话错了一半，而错的那一半正好是判决所依赖的那一半。** 拿活对象实跑之后：
那个"自证循环"只是**第一步**，紧接着的第二步就把每份退役记录拿去问了 broker——
也就是二.2 说"没有任何一处"的那个第二证人。而且**现役那道门比这道死门严一档**：
这道死门只问"这个 env 的这一代次你发过吗"，会放过"代次是真的、内容被换过"的伪造。
所以接上去是**降级**，不是加固。**删掉，并把"为什么不许加回来"写成变异测试。**

#### 三层查证，逐层给活证据

跑法：pod1 `/workspace/franco/wt_gate40_20260807`（HEAD `c1b5ca10` 的干净 worktree），
解释器 `/workspace/hope_isaac_venv/bin/python`（`cryptography 50.0.0` + `hydra` 都在，
所以本节所有数字**`0 skipped`**），只用 CPU（`CUDA_VISIBLE_DEVICES=""`）。

**(1) 机制码：存档的退役来历一共有几个证人。** 用真的 `ActionBirthBroker` +
真的 `LazyActionTaskPool` 跑完整旅程（预约→提交→消费→发任务→真 reset 退休→存盘），
再逐层伪造：

| 伪造 | 结果 | 谁拦的 |
| --- | --- | --- |
| 只把 `retired_generations` 台账从 `[[0,1]]` 改成 `[[0,7]]` | 拒 | `retired generation ledger differs from compact retired birth records`（同源自证那一步） |
| 整条退役记录改写成 broker 从没发过的 gen2（收据重算 canonical SHA、transcript 重算、integrity 重签） | 拒 | **`BirthProtocolError: birth is not the env's exact consumed generation`** —— 即 `assert_consumed_birth`，`action_ball_runtime.py:11122/11125` |
| 同上，但先把 `assert_consumed_birth` 变异成空函数 | 仍拒 | `ValueError: proposal sample was not assigned to exact birth/refill`（solver 那本提案账，第三个证人） |
| 同上变异 + 老实存档 | 载入成功 | 控制组：变异体本身没有把老实的路也弄红 |

**所以二.2 那句"没有任何一处拿 broker 的 transcript 当第二个证人"是被推翻的**，
第二个证人就在同一个 `load_state_dict` 里、那句自证的下面十几行。
二.2 说对的那一半是：`retired_generations != expected_retired_generations` 那一句
**确实**只证明存档自己前后一致。已在源码那一行补了人话注释说明它不是来历校验。

**(2) 它比现役那道门粗一档 —— 这是删而不是接的真正理由。** 同一个 broker 上：

- 给它一份"贴着 env0 gen1 标签、内容其实是另一格出生"的收据：
  被删的那道门**放行**（代次是真的），`assert_consumed_birth` **拒绝**。
- 反过来，凡是 `assert_consumed_birth` 放行的 `(env, 代次)`，被删那道门必然也放行：
  broker 自己的 `load_state_dict` 有一句 `consumed generation exceeds last generation`
  （`action_ball_runtime.py:6990`），三种方向的伪造（consumed 超前、last 退后、
  consumed 有而 last 无该行）**全部被拒**。所以 `consumed <= last` 在跨进程续跑下也成立，
  删它**没有丢掉任何拦截面**。

**(3) 实验史 / 现役 argv：** 它是 `6b13e0ff`（07-27 那个 WIP 大快照）带进来的，
从落地起就没有调用方。全仓搜索排除 `.claude/worktrees/` 后，字面量、字符串、
`getattr`/`hasattr`、`commands.py:3371-3377` 那张动态方法名清单
（只列 `binding_for_slot` / `reserve_many_true_reset` / `pending_receipt` /
`commit_many_true_reset` / `state_dict` / `load_state_dict`，**没有它**）、
`test_action_ball_motion_batch_handoff.py:43` 那个 `__getattr__` 转发壳、以及 docs，
命中的只有 `def` 那一处和本文件里这场判决自己的 4 次提及。

#### 变异测试：六个变异体，五个转红，一个诚实地没转红

新模块 `hope_training/whole_body_tracking/tests/test_action_ball_retirement_provenance.py`，
`10 passed / 0 skipped`。变异是**改生产文件**再跑，不是改测试：

| 变异 | 改了什么 | 转红的测试 |
| --- | --- | --- |
| M1 | 删掉 pool 载入时对每份 retired birth 的 `assert_consumed_birth` | `test_checkpoint_cannot_invent_a_generation_the_broker_never_issued` |
| M2 | 把 `assert_consumed_birth` **削成被删那道门的强度**（保留"代次到了"，去掉"就是这一份收据"） | `test_the_deleted_gate_would_have_waved_a_content_swap_through` |
| M3 | 把 `assert_known_generation` 原样加回来 | `test_the_coarse_gate_is_gone_and_the_owner_is_still_here` |
| M4 | 去掉 broker 的 `consumed generation exceeds last generation` | 参数化那条的 **3 格全红** |
| M5 | 去掉 `retired generation ledger differs...` 那一句 | `test_bumping_only_the_retired_ledger_is_caught_by_the_same_load` |
| M6 | 把收据比对削成"只比 SHA 的前 2 个十六进制字符" | **没转红** |

**M2 就是 Franco 那条"粗一个档次的检查就过不了"的正面证据**：把现役那道门削到
死门的强度，测试立刻红。M1–M5 每个变异**一一对应**杀掉一条测试，没有一条是
"反正全红了"。

**M6 要如实记：本轮的变异套件盖不住"截断哈希"这一档粗化。** 原因是要打红它得
构造一个前 2 字符碰撞的 SHA，按需构造不出来。这不是本次判决的形状（本次判决的形状
是 M2），但**它是这套变异测试的已知盲区**，别把"六个变异全过"当成"任意粗化都拦得住"。

#### 回归账（pod1，`/workspace/hope_isaac_venv/bin/python`，`-n 16`，`-p no:randomly`）

受影响模块 = 全仓 `tests` 里提到 `action_ball_runtime` 或 `ActionBirthBroker` 的 **26 个**。

| | before（HEAD `c1b5ca10` 原样） | after（本轮改动） |
| --- | --- | --- |
| passed | `811` | `821` |
| failed | `15` | `15` |
| errors | `2` | `2` |
| **skipped** | **`0`** | **`0`** |

**失败/错误集合逐条相同**（`test_frozen_canonical_sha_*` 两条、`train.py` actor 合同
那一族 13 条、`test_action_ball_fixed_view_motion.py` 与 `test_action_ball_motion_birth.py`
两个收集期 ERROR），**全部是先于本轮就红的**。`+10 passed` 恰好是新模块那 10 条。

#### 谱系代价：零

`action_ball_runtime.py` 的文件 SHA 目前是 `8dcae9be…`，**全仓没有任何产物记着它**。
唯一按文件名钉它的是 `configs/a3_vendor_runtime_authority_202607*/…training_contract.json`
那 8 个目录，它们记的是 `8f2fe8c7…`——**在本轮之前就已经和 HEAD 对不上了**，
本轮不新增这个问题。活着的 N1/A211/C211 谱系（`configs/action_ball_n1_measured_20260803/`）
**根本不钉这个文件**。另外 `RUNTIME_CONTRACT_SHA256` 是
`_sha256_json(_CONTRACT_DESCRIPTION)`（一份声明字典的哈希），**不是文件字节的哈希**，
所以删一个方法不会动它，已有 checkpoint 的兼容性不受影响。

#### 顺带：把"上一轮为什么会漏"这件事修在方法学上，不是修在结果上

上一轮那份扫描**根本没有落盘**（全仓找不到脚本或测试），所以"它现在能不能扫到公开方法"
这个问题的答案是"它不存在"。补了两件：

- `scripts/audit_zero_call_site_gates.py`：把方法学固定下来——名字形状**公开与带下划线
  都收**、`def` 用 `ast.walk` **收嵌套**、排除 `.claude/worktrees/`、判据用 token 频次
  （这样 `getattr` / `monkeypatch.setattr` / `importlib` 共享库那些形式不会被当成没人用）。
  它自带 `--self-test`：同一份 fixture 喂给**上一轮那个做法**（只认带下划线、只收模块级），
  老做法只看见 1 个，新做法看见 3 个——**如果哪天老做法也看见 3 个，自检自己会红**，
  因为那说明这条自检失去了判别力。
- `tests/test_audit_zero_call_site_gates.py`（`22 passed`）：只钉方法学，**不钉结果**。
  钉结果要维护一张允许名单，每加一道门改一次，那是仪式。这里测的是那两条真会重犯的：
  公开形状认不认、嵌套 `def` 收不收；外加"别人 worktree 里的调用点不许算数"和
  反向判别力（加一个本仓调用点就不该再报）。全仓扫描本身**不进套件**（要读一千多个文件，
  而且它的输出是给人判决用的），手动跑。

在本轮改动之后跑一次全仓：`1020` 个 `.py`、`1959` 个门形状 `def`、
零调用点**剩 3 个**——`_validate_reveal_bridge`、`_validate_artifact_path_hash`、
`_reverify_receipt_contact_source_files`，与 §5.6.18 二.2 那张表去掉本条后完全一致。
（**2026-08-07 同日就地更正：`_validate_reveal_bridge` 当天已接线（§5.6.22），所以现在剩 2 个。**）

#### 这一节没做什么

1. **没碰** `hope_rewards.py` / `commands.py` / `train.py` / 两个 211 launcher —— 本轮有三条
   别的 workflow 在改它们。本轮只读它们（`commands.py:3371` 那张动态方法名清单、
   `hope_commands.py:17331/17340` 的 exact-resume 校验、`train.py:9042` 的源码 SHA 映射）。
2. **没接** `_validate_reveal_bridge`（剩下三个里唯一判过的那个），按分工那是另一条待办。
   （**2026-08-07 同日就地更正：那条待办当天已做完，见 §5.6.22。**）
3. **没补** M6 那一档（截断哈希）的变异证据，理由见上。

### 9.2.14 L1 证书"换路径不能换字节"那条测试：报的病因是错的，真病因是内核文件时间戳只有 1 ms 刻度（2026-08-07 落地，pod1 host-only，无 GPU）

**人话**：有条测试是"验证器已经打开证书之后，有人把那个路径下的文件掉包，验证器读到的
字节必须还是原来那一份"。它一直时红时绿。交上来的病因是"它把整个进程的 `os.read` 换掉了，
会被同进程里别的测试的 fd 抢跑"。**这个说法不成立**。真病因是它断言错了东西：它断言的不是
"字节没变"，而是"改名顺带把 `st_ctime_ns` 改了、于是元数据漂移告警响了"——而这条告警响不响，
是在跟内核的文件时间戳刻度赌博。

#### 9.2.14.1 先证伪交上来的病因

按票面复现（pod1，`/workspace/hope_isaac_venv/bin/python` = Python 3.10.18 / pytest 9.1.1，
干净 worktree `/workspace/codexschema/nohope_l1flake_20260807` @ `c1b5ca10`）：

| 跑法 | 通过次数 |
| --- | --- |
| 只跑 `test_l1_certificate_path_swap_cannot_change_consumed_bytes` | **4 / 10** |
| 与 `test_plan_claim_is_a211_fresh_and_denies_retired_lineage` 配对跑 | **2 / 10** |

票面写的是"单独跑必过、配对跑必失败"。实测**两种跑法都会红**，只是配对时更容易红（配对跑得更快）。
所以它跟执行顺序、跟别的测试的 fd 都没关系——**它自己单独跑就是个抛硬币**。

插桩看活值也证实：钩子每一次都正确命中了证书那个 inode（`swapped` 每次都是 `True`），
从来没有被别的 fd "抢跑"过。变的不是钩子，是**被测代码那条告警响不响**。

（顺带：票面把配对对象写成 `a225`，实际那条测试叫
`test_plan_claim_is_a211_fresh_and_denies_retired_lineage`，仓库里没有 a225 版本。）

> **2026-08-07 独立验收就地更正：括号里"仓库里没有 a225 版本"这半句不成立。**
> 仓库里有多个 a225 测试模块（例如
> `hope_training/whole_body_tracking/tests/test_action_ball_a225_trainability.py`），
> 拿它跟修之前那条配对，同样把它打红（`3/12`）。结论不变——那条测试跟谁配对都会红——
> 但"不存在"这个理由是错的。复跑收据见 §9.2.15.5。

> **2026-08-07 独立验收就地更正：括号里"仓库里没有 a225 版本"这半句不成立。**
> 仓库里有多个 a225 测试模块（例如
> `hope_training/whole_body_tracking/tests/test_action_ball_a225_trainability.py`），
> 拿它跟修之前那条配对，同样把它打红（`3/12`）。结论不变——那条测试跟谁配对都会红——
> 但"不存在"这个理由是错的。复跑收据见 §9.2.15.5。

#### 9.2.14.2 真病因：窗口 43 µs，时钟刻度 1 ms

被测代码 `_read_open_fd_snapshot`（`scripts/audit_motion_schema2_table_net_clearance.py:155`）
在读前后各 `fstat` 一次，比较
`(st_dev, st_ino, st_mode, st_size, st_mtime_ns, st_ctime_ns)`。
旧测试用 `path.rename(...)` 做掉包——**改名只动得了 `st_ctime_ns` 这一项**。

在 pod1 的 `/tmp`（overlay）上实测 200 次"写文件 → open → rename → fstat"：

- **196 / 200 次 `st_ctime_ns` 前后完全相同**；
- 变了的那 4 次，差值只有 `999999 ns` 或 `1000000 ns`；
- 另测 300 次连续写只落出 **18 个不同的 `st_mtime_ns`**，步长同样是 1 ms。

也就是说**内核给文件时间戳用的是粗时钟，刻度 1 ms**；而从"写下证书"到"rename 掉包"整个窗口
中位数只有 **43 µs**。绝大多数情况下 rename 落在同一个毫秒刻度里，`st_ctime_ns` 一个数都不变，
告警当然不响，验证器正常返回，`pytest.raises` 就报 `DID NOT RAISE`。

**这不是被测代码的漏洞。** 读走的是 fd 不是路径，掉包本来就改不了消费到的字节——这才是这条护栏
真正的保证。旧测试断言的是这条保证的一个**偶然副作用**，而那个副作用受时钟精度支配。

#### 9.2.14.3 改法：断言那句护栏自己的名字，而不是它的副作用

`tests/test_motion_backhand_loop_b_table_net_clearance.py` 改一处、加两处：

1. `test_l1_certificate_path_swap_cannot_change_consumed_bytes`（同名保留）
   —— 注入点从"全局 `os.read`"换成"被测模块自己的 `_read_open_fd_snapshot`"，
   并且**只对证书那一个路径生效**。掉包发生在 `fstat` 基线之前，所以元数据告警**确定不会响**，
   剩下唯一可判定的就是"消费到的字节"本身。断言改成：
   `snapshot.data` / `snapshot.sha256` / `snapshot.size` 全等于原始那一份，
   解析出的 `cert["runtime"]` 里没有伪造标记，同时确认 `path` 上确实已经是伪造字节（否则测试是空的）。
   **零时间戳依赖。**
2. 新增 `test_l1_certificate_inode_mutated_mid_read_fails_closed`
   —— 补上另一半：攻击者不换路径，直接往同一个 inode 上追加字节。
   判定用 `st_size`（整数，不是时间戳），所以确定会触发
   `inode/metadata changed during immutable read`。
   拦截走新加的 `_ModuleScopedOs`：**只替换被测模块命名空间里的 `os` 这个名字**，
   全局 `os.read` 一个字节都不碰（测试里直接断言 `os.read is real_read`）。
3. `_ModuleScopedOs` 是个小壳：覆盖的名字走覆盖，其余全部 `__getattr__` 转发给真 `os`。

**没有用 `-p no:randomly`、没有固定顺序、没有跳过配对、没有 sleep。**

#### 9.2.14.4 变异测试：等强，而且比旧版更强

在 `scripts/audit_motion_schema2_table_net_clearance.py` 上打三个真回归（跑完即还原）：

| 变异 | 内容 | 结果 |
| --- | --- | --- |
| M1 | SHA 仍按 fd 字节校验（所以不会报错），但交给下游的 `FileSnapshot.data` 改成 `path.read_bytes()` | **只有换路径那条红** ✅ |
| M2 | 整个读改成走路径 `path.read_bytes()`，不走 fd | **两条都红** ✅ |
| M3 | 把中途漂移的身份元组从 6 项砍成 `(st_dev, st_ino)`（粗一个档次） | **只有 inode 篡改那条红** ✅ |

未变异基线：`2 passed`；三个变异各自还原后复跑：`2 passed`。

**M1 是关键的那个**：它是一次真实的"换路径导致消费字节变了"，但它**不抛异常**——
任何"只问有没有报错"的粗测试都看不见它。把改之前那条测试原封不动搬过来对着 M1 跑：

| 旧测试 | 通过次数 |
| --- | --- |
| 对着**未变异**的源码 | 16 / 20（本该 20/20；这 4 次红就是抛硬币） |
| 对着 **M1 变异**的源码 | **13 / 20 —— 真回归有 65% 的概率溜过去** |

所以这次不是"把测试改弱好让它绿"，是**旧测试根本拦不住它自己名字上写的那个回归**，
新测试拦得住。

#### 9.2.14.5 稳定性收据

配对跑（a211 那条 + q50 那条 + 新旧两条 L1），`/workspace/hope_isaac_venv/bin/python`：

- **12 / 12 全绿**（每次 `4 passed`，约 1.0 s）。
- 整模块交叉跑、正反顺序各若干次，L1 两条从未再红。

#### 9.2.14.6 与基线对拍

解释器 `/workspace/hope_isaac_venv/bin/python`（3.10.18，pytest 9.1.1），
四个受影响模块：`test_motion_backhand_loop_b_table_net_clearance` /
`test_run_gate3_first_tick_harness` / `test_run_phase1_q50_persistent_supervisor` /
`test_launch_action_ball_a211_four_arm_diagnostic`。

| | 结果 | skipped |
| --- | --- | --- |
| 改之前（同一 worktree，只把这两个测试文件还原） | `11 failed, 233 passed` | **0** |
| 改之后 | `10 failed, 235 passed` | **0** |

失败集合逐条 diff，**唯一的差别就是少了 `test_l1_certificate_path_swap_cannot_change_consumed_bytes` 这一行**；
其余 10 条前后完全一致。用例总数 244 → 245（+1 = 新增那条）。

那 10 条**是既有的红、与本轮无关**：它们全部来自
`scripts/audit_motion_schema2_table_net_clearance.py` 里对 `geometry.py` 的冻结几何源钉子
与当前源码漂了（样例报错：`build_net_post_cfg post height formula changed`）。
这是内容漂移，需要单独裁决，**本轮没碰**，记在这里。

#### 9.2.14.7 顺带扫的：还有谁在无差别改全局 I/O 原语

扫 `tests/` 和 `hope_training/whole_body_tracking/tests/` 里所有
`monkeypatch.setattr(<模块或类>, <I/O 原语>, ...)`。**结论：真会随顺序变结果的只有这一条**，
其余按危险度排：

**已一并修（1 处）**

- `tests/test_run_gate3_first_tick_harness.py:579` `test_contract_change_during_read_fails_closed`
  —— 它把 `Path.read_bytes` 换成一个**无条件**在读完之后往 `self` 追加一个空格的钩子。
  钩子挂上期间，**任何人**读任何文件都会被就地改写，包括仓库里的文件。
  已加 `if self == path` 限定 + `assert mutated == [path]`（拦截没生效就说测试是空的）。
  变异测试证明没改弱：把 `load_bound_json` 的身份元组砍成 `(st_dev, st_ino)`（粗一档），
  这条测试**仍然红**；未变异 / 还原后都是 `1 passed`。

**留着的（有作用域限定，或只罩住一次调用，不会随顺序变，但记账）**

- `tests/test_run_phase1_q50_persistent_supervisor.py:255` 把 `os.environ` 整个换成普通 `dict`
  —— 只罩住一次 `_require_invoking_environment` 调用；但期间任何 `os.environ[...] = x` 不会
  同步到真环境（不走 `putenv`）。窗口极短，暂不动。
- `hope_training/whole_body_tracking/tests/test_post_swing_teacher.py:431`
  把 `Path.read_text` 换成**无条件抛异常**——罩住一次 `_load` 调用。
  期间任何无关的文本读都会炸，且报错信息会误导（说成"收据被重新打开了"）。
- `hope_training/whole_body_tracking/tests/test_canonical_frame_identity.py:606`
  `Path.read_bytes` 计数器统计的是**全进程**的读，只是恰好罩住一次调用。
- 三处 `builtins.__import__` 钩子
  （`test_motion_backhand_loop_b_table_net_clearance.py:249`、
  `test_action_ball_update_profiler.py:453`、`test_full_scene_probe_runtime.py:768`）
  —— 都正确转发给 `original_import`，都只罩住一次调用。
- `test_exact_resume_state.py:1804` 的 `os.kill`、
  `test_launch_action_ball_a211_four_arm_diagnostic.py:3435` 的 `os.execve`
  —— 全局但窗口极短，且这两个原语在窗口内没有别的调用者。
- 已经限定好的（无需动）：`test_canonical_motion_markers.py:129`、
  `test_run_ready_to_strike_join_ladder_stage2.py:1770`、
  `test_launch_action_ball_curriculum.py:1949`、
  `test_launch_n1_measured_vendor_v2_diagnostic.py:939`、
  `test_run_phase1_q50_persistent_supervisor.py:504/549`。

**明确待办（本轮没做）**：上面"留着的"那一档目前是靠"窗口短"活着，不是靠机制。
真要根治，得给测试层一个统一的"作用域内替换某模块的 I/O 名字"工具
（`_ModuleScopedOs` 是第一个雏形），并把这几处迁过去。本轮只落了雏形，**没有推广**。

#### 9.2.14.8 票面另外两条的裁定

票面还点了 `test_launch_observes_exact_exec_and_duplicate_is_no_clobber` 和
`test_plan_claim_is_a211_fresh_and_denies_retired_lineage`，说"单独跑在两棵树上都通过"。
实测这两条**根本没有 monkeypatch 全局 I/O 原语**（前者压根没用 monkeypatch，走真 fork/exec），
在上面 12 次配对跑里**每次都绿**。它们只是当初和那条 flake 同批被观察到，**没有病**。

#### 这一节没做什么

1. **没碰** `hope_rewards.py` / `commands.py` / `train.py` / 两个 211 launcher。
2. **没修**那 10 条既有的几何源钉子红（§9.2.14.6），那是另一条待办。
3. **没有**把 `_ModuleScopedOs` 推广到 §9.2.14.7 列的其余几处。

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

### 9.2.15 四格聚合的入口函数 `_audit_cell` 一直是零覆盖：测试从它的出口造数据（2026-08-07 落地，pod1 host-only，无 GPU）

**人话**：四格 scale4096 聚合收据的每一格，都要先过
`action_ball_211_four_grid_prelong_barrier.py` 里的 `_audit_cell` —— 它负责打开这一格的
launch claim、重算摘要、核对 selector/来源/GPU/产出契约、复审运行期安全罚价、验安全台账、
验 pre-long 门、验终局 checkpoint，最后拼出一行审计记录。

它到今天为止**一行都没被跑过**。现存的 barrier 测试全部调用文件里那个 `_audits()` 夹具，
直接手写"已经审计完的行"喂给下游的 `document_from_audits`。也就是说测试是从
`_audit_cell` 的**出口**那一侧开始造数据，被测函数本身被整个跳过。

这不是"覆盖率低一点"，是一类**系统性的形状**：上一轮那句手抄的
`filename_iteration != 5`（跑满 5 个 update 之后落盘的末位其实是 `model_4.pt` / `iter=4`，
差一格）正是这样活下来的 —— 它写在 `_audit_cell` 里，而没有任何测试经过 `_audit_cell`。
`a970a58a` 把那个 5 换成了共享常量，但**测试仍然是零**。本轮补的就是这一块。

#### 补了什么

只动一个文件：`hope_training/whole_body_tracking/tests/test_action_ball_211_four_grid_prelong_barrier.py`
（`54 -> 150` 条）。新增的一组一律走 `barrier._audit_cell` / `build_receipt_document` /
`validate_receipt` 的真实路径：真的落一份 scale4096 结果文件、真的落一份 `launch_claim.json`、
真的让 `_audit_cell` 自己去读、去比、去重算摘要。

两族发射器用替身 —— `modules=` 本来就是现役注入点，`build_receipt_document` 和
`validate_receipt` 都把它透传给 `_audit_cell`。替身只做**发射器该做的解析**（pin↔文件绑定、
canonical JSON、把 payload 拆成 spec/lineage/selector），**不做任何 `_audit_cell` 自己的判断**；
所以下面每一条拒收都必须是 `_audit_cell` 出的。替身刻意只装本族那三个名字
（A 装 `_validate_predecessor_result` 一族、C 装 `_validate_scale_predecessor` 一族），
分派错了就是 AttributeError，不会被"两边都有"糊过去。

覆盖到的拒收面 42 条（每条 A/C 各跑一遍 = 84 个 case），按面分组：
launch claim 文件本身 5 条、selector/来源/GPU/产出契约 14 条、发射器与 pre-long 的安全词表 3 条、
终局安全台账 4 条、pre-long 门与安全台账的绑定 5 条、终局 checkpoint 7 条、四格共享绑定 4 条。
每条都是"同一份夹具先当正例过一遍，再变异一处必须被拒"，且变异一律构造成
**粗一个档次的检查会放行**（每条 param 上都写了那一档是什么）。

另外补了四件以前没人验过的事：

1. **`_audit_cell` 的出口形状与 `_validate_audit_row` 的入口形状对得上** —— 以前全靠人眼。
2. **`_audits()` 这份手抄副本被钉回真值** —— 副本一旦漂了，那一堆"从 `_audits()` 造行"的
   老用例就会开始给一个不存在的形状发合格证。
3. **`_family_modules()` 不再是零覆盖** —— 这是同一个病的最深一层：`modules=` 在测试里
   永远被注入，于是现役唯一加载真发射器的地方全仓没人跑过，没人验过真模块到底给不给得出
   `_audit_cell` 要的那一面。现在 AST 抓出 `_audit_cell` / `_claim_for_result` /
   `_reaudit_runtime_safety_reward_economy` 里每一个 `module.<名字>`，逐个断言真发射器与替身
   都有，且对方族的名字**不在**这一边。
4. **barrier 手抄的 `STRICT_ZERO_SAFETY_KEYS` 与活 pre-long gate 的
   `STRICT_ZERO_SAFETY_COUNTERS` 直接比活值** —— 这两份一旦漂开，四格聚合会**每一格都拒收**。

#### 变异测试：老那 54 条对 19 个变异里的 18 个是瞎的

在 pod1 clean worktree `/workspace/franco/auditcell_20260807`（`559b95f4`，无 GPU）上，
把 `_audit_cell` / `_claim_for_result` / `_validate_prelong_behavioral_binding` 逐个改粗一档，
分别用**新套件**和**老那 54 条**跑：

| 变异（把严格度降一档） | 新套件 | 老 54 条 |
| --- | --- | --- |
| M1 两个迭代号只比"互相自洽"，不比共享常量（**上一轮真实存在过的那一档**） | 杀 2 | 全绿 |
| M2 `require_empty` 改真值判断 | 杀 2 | 全绿 |
| M3 共址 opt-in 改真值判断 | 杀 2 | 全绿 |
| M4 `all_tensors_finite` 改真值判断 | 杀 2 | 全绿 |
| M5 `tensor_groups` 只判类型不判非空 | 杀 2 | 全绿 |
| M6 安全词表比集合不比顺序 | 杀 4 | 全绿 |
| M6b strict-zero 词表"是子集就行" | 杀 2 | 全绿 |
| M7 checkout 比 normpath 之后的值 | 杀 2 | 全绿 |
| M8 claim 摘要不再重算 payload | 杀 3 | 全绿 |
| M9 pre-long 分母只比总数 | 杀 2 | 全绿 |
| M10 acceptance cutoff 改真值判断 | 杀 3 | 杀 1 |
| M11 predecessor"形状合法就行" | 杀 2 | 全绿 |
| M12 selector 只比 `A0`/`C1` 那个标签 | 杀 2 | 全绿 |
| M13 GPU index 只判"在 (0,1) 里" | 杀 2 | 全绿 |
| M14 rate/speed 两个排除位改真值判断 | 杀 4 | 全绿 |
| M15 claim 与结果的 namespace 不再要求一致 | 杀 2 | 全绿 |
| M16 claim 与结果的 stage 不再要求一致 | 杀 2 | 全绿 |
| M17 审计行少一个键 | 杀 7 | 全绿 |
| M18 pre-long 绑定直接不接线 | 杀 10 | 全绿 |
| M19（对照，语义等价的改写：审计行回显 checkpoint 里的迭代号） | 存活（应当） | 全绿 |

19 个真变异 **19/19 被新套件杀**；老那 54 条只对 M10 有反应，那是因为
`_validate_prelong_behavioral_binding` 本来就有一条直接调用的用例。**其余 18 个对老套件完全隐形**。
M19 是故意放的对照 —— 它语义等价（上面的检查已经保证两个迭代号等于共享常量），存活是对的，
说明这套用例没有过拟合到实现细节。

#### 回归账（解释器与 skip 数写清）

解释器 `/workspace/hope_isaac_venv/bin/python`（Python 3.10.18），pod1 clean worktree
`/workspace/franco/auditcell_20260807` @ `559b95f4`，host-only、不占 GPU。

六个相关模块（barrier / terminal-index / prelong gate / 两个发射器共享常量 / 四格 authority /
A-C family parity）：

* 基线（改动前）：**336 passed, 0 skipped**
* 本轮之后：**432 passed, 0 skipped**（+96）
* barrier 单模块：`54 -> 150`

#### 顺带扫出来的:同一批里还有没有别的"测试绕过被测函数"

判据是"某个函数有拒收逻辑，但所有测试都从它的下游或上游造数据"。做法是对 barrier /
pre-long gate / 四格 authority / 两个发射器建模块内调用图，以"被任何测试文件点过名的函数"
为入口取闭包，再看哪些带 `*Refused` 的函数落在闭包外。

* `action_ball_211_four_grid_prelong_barrier.py`、`action_ball_4096x5_prelong_gate.py`、
  `action_ball_211_four_grid_contract.py`：本轮之后**没有**带拒收的函数落在闭包外。
  pre-long gate 那批 `validate_safety_audit` / `validate_survival_denominators` /
  `_validate_reveal_bridge` 虽然测试里不点名，但都从 `validate_prelong_gate`（上游入口）
  真的跑到 —— 那是正常的端到端覆盖，不是本节说的绕过。
* `launch_action_ball_a211_four_arm_diagnostic.py` 的
  `_validate_frame0_live_safety_evidence`（3 处 `LaunchRefused`）：**全仓零调用点、零测试**。
  C211 那边 2026-08-05 的注释写着"shim 已退役，底层权威仍在 A211"，但 A211 这边也没人叫它。
  这既是又一个零调用点的门，也是一处 A/C 不对称（C 删了、A 留了个死的）。
  本轮**不动它**，另开一条单独裁定 —— 改门要连证据一起改，不许顺手删。
* `_admission_training_argv`（A/C 各一个）是扫描的假阳性：它在模块级作为
  `training_argv=` 绑定进 admission，不是死代码。

**这一节没有改任何生产代码**，只增测试文档。`a970a58a` 的三处终局编号出处、
`solver_profile_sha256` 那族 pin、§9.2.12 刚重签的 18 份内容寻址产物，一个字都没碰，
不需要重铸任何东西。

### 9.2.16 三条"做完了"的独立验收：主结论都成立，但每条都漏了同一件事的另一半（2026-08-07，pod1 host-only，全程未占 GPU）

**人话一句：** 同一天有三条 workflow 各自报"做完了"。这一节不是转述他们的收据，是**在自己的两棵树上从头重跑一遍**：
接了线的就改回没接线的样子看测试红不红；说"已经有别的机制管"的就去查那个机制是不是真在跑；
说"没人调用"的就自己再搜一遍；说"不再抖了"的就自己配对跑十几遍。

结论：**三条的主结论都成立**。但每一条都漏了同一类东西——**一件事的另一半没人管**。
本轮把其中两个半边补上了（各配变异测试，共 `6` 条新测试），第三个是既有问题、只登记。
另外顺手记一个会让**任何**变异测试自证的坑：我自己踩了一次，收据在 9.2.16.6。

#### 9.2.16.0 怎么验的

两棵**互相独立**的 pod worktree，都从 `/workspace/codexschema/nohope` 上
`git worktree add --detach 559b95f4`：`/workspace/franco/verify_20260807/wt`（跑变异）与
`.../wt2`（跑 flake 与扫描器）。解释器**只用** `/workspace/hope_isaac_venv/bin/python`
（Python `3.10.18` / pytest `9.1.1`）—— `/usr/bin/python3` 缺 hydra，会静默跳 17 条，
"`1 skipped` + 退出码 `0`"看着全绿其实什么都没跑。`CUDA_VISIBLE_DEVICES=` 纯 CPU；
`PYTHONDONTWRITEBYTECODE=1`（理由见 9.2.16.6）。下面每个数字都写了 skip 数。

**验收基线是 `559b95f4`**（下面每个数字都在这个提交上取的）。写这一节时分支已经被别的 workflow
推到 `3a045948`（那批改动给同一道门加了 `+202` 行），所以本轮**新增的两个测试模块又在 `3a045948`
上重跑并重做了变异**：`16 passed / 0 skipped`；把 import 期那段检查换成 `pass` → 2 条红，
把"活着那批出生"的证人循环删掉 → 1 条红，还原后 `16 passed`。**结论在两个提交上都成立。**

**全程只读已提交的 HEAD。** 工作区里同时有别的 workflow 在改
`action_ball_4096x5_prelong_gate.py` / `train.py` / `hope_rewards.py` / 两个 211 launcher
（那道门当时的未提交改动是"取消实际-q 硬超限拒收、改成只记账"），本轮一律不含这些在途改动。

#### 9.2.16.1 (a) 揭示->回放那道门：接线是真的接上了

| 树的状态 | 四个相关模块（gate + A/C launcher + 生产方） | 结果 |
| --- | --- | --- |
| HEAD `559b95f4` 原样 | 同上 | **`447 passed / 0 skipped`** |
| 只把 `action_ball_4096x5_prelong_gate.py` 退回 `39953498^`（接线前那一版） | 同上 | **`11 failed / 436 passed`** |
| 再退回 HEAD | 同上 | **`447 passed / 0 skipped`** |

那 11 条红的**正是**新加的 7 个测试函数（SHA 那条参数化 5 档，`7 + 4 = 11`），
第一条报的就是 `KeyError: 'reveal_to_playback_bridge'` —— 接线前收据里根本没有这一块。
**"改回接线前必须红、改回来必须绿"这条验收通过。**

粗化判别力也自己重做了一遍（改的是**生产文件**，跑完按 SHA 校验还原）：

| 粗化 | 改了什么 | 红的是 |
| --- | --- | --- |
| V1 | SHA 只量长度 `64`，不管是不是小写十六进制 | 那 5 档参数化，其余 71 绿 |
| V2 | 逐档"寿命不许倒退"改成"不许为负" | `..._playback_start_count_may_not_regress...` 一条 |
| V3 | `status` 只要求有值 | `..._status_must_be_active_fail_closed` 一条 |
| V4 | 去掉"timing 的 reveal 总数 == 逐档 reveal 之和" | `..._cohort_may_conserve_and_still_miss_the_reveal_total` 一条 |
| **V5** | **权威身份跨 update 只比 `profile` 一个字段**（不是整块） | `..._authority_may_not_drift_after_the_first_update` 一条 |

**V5 不在原报告的清单里**，是本轮新造的：原报告那条是"整段删掉"，删掉当然会红；
V5 是"留着但只比一个字段"，才是字面意义上的"粗一个档次"。它照样只杀那一条。
**判别力成立，不是"反正全红了"。**

#### 9.2.16.2 (a) 漏掉的另一半：新加的那条 import 期活值对表，自己零证据

接线那一批还加了一条 **import 期**检查：门里的 21 个 WAIT 档位、mimic 项集合、
哪些 mimic 项走 Cauchy 核，必须逐一等于生产方 `action_ball_prelong_semantics.py` 的**活值**，
对不上就 `RuntimeError`，模块导都导不进来。它是承重的（档位错位 → 逐档守恒式照样成立、
但比的是错档号；核函数归类漂了 → 会拒掉正确收据或放过错的）。

它确实会响：把 `BRIDGE_WAIT_COHORTS` 改成 `range(5, 25)`，导入立刻
`RuntimeError: pre-long gate reveal-bridge contract differs from the semantic producer`。

**但它当时零测试。** 把那整段 `if ... raise RuntimeError` 换成 `pass`，
四个相关模块 **`447 passed`，一条都不红**。这条新护栏哪天被人顺手删掉，没有人会知道。

**已补**：新增 `hope_training/whole_body_tracking/tests/test_action_ball_prelong_gate_producer_contract.py`（`3` 条）。
做法是把源码里的常量改窄一档（一条改 WAIT 档位、一条把某项 mimic 挪出 Cauchy 名单），
在新命名空间里重新 `exec`（`__file__` 仍指真文件，所以照样按路径找得到生产方），必须抛那句原话；
再用没改过的同一份源码 `exec` 一次作**控制组**。
变异复核：把那段检查换成 `pass` → **2 条红（两条"必须拒"）、控制组那条仍绿**，
不是"整个模块一起红"。锚点找不到时测试会用明确的原话喊出来，不会退化成空跑。

刻意**没有**写进 `test_action_ball_4096x5_prelong_gate.py`：本轮期间那个文件的未提交改动
从 `+418` 行变到 `+372` 行（别的 workflow 正在大改），单开一个模块不跟他们抢同一片字节。

#### 9.2.16.3 (b) 删掉的那道门：那个"别的机制"确实在跑，而且比报告说的还靠前

报告说"存档载入时会逐份过 `assert_consumed_birth`"。三层查下来，这句**成立，而且说少了**：

1. **机制码**：`assert_consumed_birth` 有**三个**调用点。`action_ball_runtime.py:8727` 在
   `_validate_birth` 里；`:11122` 管存档里**还活着**的出生；`:11125` 管存档里**已退役**的出生。
2. **现役调用链**：`_validate_birth` 被 `request_many`(`:9494`) 与 `_request_many_diagnostic`(`:9183`) 调用，
   而 `request_many` 的生产调用方是 `hope_commands.py:12427` —— **每一次要题都会过这道门**，
   不是只在续跑载入时走一次。存档那两处的生产调用方是 `hope_commands.py:16943 / :17414`。
3. **实跑**：七个变异体逐个改生产文件、跑完还原（SHA 校验），基线与还原后都是 `13 passed / 0 skipped`。

| 变异 | 改了什么 | 红的是 |
| --- | --- | --- |
| N1 | 删掉存档载入时对**已退役**出生的证人 | `..._cannot_invent_a_generation_the_broker_never_issued` 一条 |
| N2 | 把 `assert_consumed_birth` **削到被删那道门的强度**（保留"代次到了"，去掉"收据就是那一份"） | `..._the_deleted_gate_would_have_waved_a_content_swap_through` 一条 |
| N3 | 把 `assert_known_generation` 原样加回来 | `..._the_coarse_gate_is_gone_and_the_owner_is_still_here` 一条 |
| N4 | 去掉 broker 的 `consumed <= last` | 参数化那条 3 格全红 |
| N5 | 去掉 `retired generation ledger differs...` 那句同源自证 | `..._bumping_only_the_retired_ledger...` 一条 |
| **N6** | **只删掉存档载入时对"还活着"那批出生的证人**（退役那半留着） | **补测试之前：一条都没红** |
| **N7** | 精确收据比对只留 `consumed_receipt != birth`，去掉代次那半 | 一条都没红（**不是洞，见下**） |

N1--N5 一一对应，**其中 N2 就是"粗一个档次就过不了"的正面证据**：把现役那道门削成死门的强度，
测试立刻红。**"删掉 `assert_known_generation` 而不是接上它"这个判决成立。**

#### 9.2.16.4 (b) 漏掉的另一半：`load_state_dict` 里挨着的两个循环，只有一个有人管

`LazyActionTaskPool.load_state_dict` 里是**两个**紧挨着的循环：一个把存档里**还活着**的出生
（`births`）、一个把**已退役**的出生（`retired_births`）逐份交给 broker 那个证人。
上一轮的测试只钉了退役那一半。**把活着那一半的整个循环删掉（N6），全部 26 个相关测试模块
`851 passed / 15 failed / 0 skipped / 0 errors` 一条都不动**（那 15 条红是本轮之前就红的，见 9.2.16.7）。

**已补**：`test_action_ball_retirement_provenance.py` 加 `3` 条（`10 passed` → `13 passed`）：

- 一条老实的活出生存档能载入（控制组，防"红是因为哪都跑不通"）；
- 一条**自洽的**伪造必须被 `assert_consumed_birth` 按原话拒 ——
  同一份存档里指回这份出生的东西全部跟着改签（收据里复制的 `reset_generation` / `birth_sha256`、
  收据自己的 canonical SHA、`pending_order`、`seen_sha256`、整包 integrity），
  所以解码期那一串"存档自己前后一致"的检查全部通过，只剩"问 broker 发没发过"能拆穿它；
- 一条证明红的确实来自这个证人：把证人换成空函数后，伪造改由 solver 那本提案账拦
  （原话变成 `task was not emitted by exact solver`），而老实存档在证人被拔掉时仍能载入。

补完之后 **N6 只杀那一条新测试**，其余 12 条仍绿。

**N7 不是洞，是冗余；这里把理由写下来，而不是拿一条测试假装。** 非诊断路径下
`consumed_receipt` 按 `(env_id, reset_generation)` 取，取得到就说明那一代次被 `_consume` 过，
而 `_consume` 同一步就把 `_consumed_generation[env]` 推到该代次；诊断路径下按 env 取最新一份，
相等即说明那份就是它。所以"收据相等"蕴含"代次到了"，代次那半永远不会单独触发。
它是带保险丝的写法，不是可被利用的缺口 —— **没有**测试，也不该硬造一条。

**谱系代价复核**：`configs/a3_vendor_runtime_authority_*`（`8` 个目录、`15` 个文件、`30` 处）
按文件名把 `action_ball_runtime.py` 钉在 `8f2fe8c7…`；而**在本轮之前**（`c1b5ca10`）该文件已是
`8dcae9be…`，本轮之后是 `eb36095f…`。**这个钉子早就对不上，不是这次弄坏的**，与原报告一致。

#### 9.2.16.5 (c) 零调用点复查 与 (d) flake 复跑

**(c) 零调用点**（自己搜一遍，排除 `.claude/worktrees/`）：
`assert_known_generation` 全仓已无 `def`（只剩测试里那个当变异体用的复刻、和文档里的提及）；
`_validate_reveal_bridge` 是 `def` + **1 个真调用点**；
`_validate_artifact_path_hash`（`canonical_motion_bank_gate.py:3905`）与
`_reverify_receipt_contact_source_files`（`canonical_neutral_ready.py:3907`）**仍然只有 `def` 自己**。
另外独立跑了那份扫描器（`scripts/audit_zero_call_site_gates.py`，在干净 HEAD 的 wt2 上）：
`--self-test` 通过；全仓扫 `1022` 个 `.py` / `1960` 个门形状 `def`，**零调用点剩 `2` 个**，
正是上面那两个。**"剩 2 个"成立。**

**(d) flake**：同一棵干净树，**不用** `-p no:randomly`、不固定顺序、不 sleep，每档跑 12 次。

| 档 | 跑什么 | 通过 |
| --- | --- | --- |
| A | 修好的那条 + `test_plan_claim_is_a211_fresh_and_denies_retired_lineage` | **12/12** |
| A2 | 修好的两条 L1 + 上面那条 a211 | **12/12** |
| B | 修好的那条 + `test_action_ball_a225_trainability.py` 整模块 | **12/12** |
| C | **退回修之前**那条，单独跑 | 5/12 |
| D | 退回修之前那条 + a211 那条 | **2/12** |
| E | 退回修之前那条 + a225 那个模块 | 3/12 |

失败原话每次都是 `Failed: DID NOT RAISE TableNetError`，与报告认定的病因（断言压在一条
受内核时间戳刻度支配的告警上）一致。**"修好了"成立；"票面写的『只有配对才失败』不成立"也成立**
（单独跑本来就 5/12）。

另外独立做了一次"新测试是不是真的更强"的变异：把审计脚本改成
**SHA 仍按 fd 字节校验、但交给下游的 `FileSnapshot.data` 改成 `path.read_bytes()`**
（这是一次真实的"换路径就换了消费字节"的回归，而且**不抛任何异常**）——
新测试 **10/10 全部抓到**，还原后 5/5 全绿。

> **就地更正 §9.2.14 末尾那句**："票面把配对对象写成 `a225`，仓库里没有 a225 版本"——
> **这半句不成立**。仓库里有好几个 a225 测试模块，
> `hope_training/whole_body_tracking/tests/test_action_ball_a225_trainability.py` 就是一个，
> 拿它配对同样能把旧版那条打红（上表 E 档 3/12）。结论不变（那条测试跟谁配对都会红），
> 但"不存在"这个理由是错的。

#### 9.2.16.6 一个会让变异测试自证的坑：同长度改动 + 同一秒还原 = `__pycache__` 里留着变种

本轮我自己踩了一次，如实记下来，因为它会**静默地**把变异测试变成自证：

我做过一次 `sed -i` 把 `range(5, 26)` 改成 `range(5, 25)`（**字节长度完全相同**），
跑完用 `cp` 把备份盖回去。之后 `test_launch_action_ball_c211_diagnostic.py` **收集期直接炸**，
报的正是那条 import 期检查。查下来：`.pyc` 头里记的源文件 `(mtime, size)` 是
`(1786106737, 81424)`，和还原后的源文件**一模一样** —— `sed` 与 `cp` 落在同一秒内、长度又没变，
于是 Python 认为缓存有效，继续用**变种**的字节码。直接 import 那道门的路径当时恰好各自重编过
所以没露馅，只有经 `importlib` 从源文件加载的 c211 链踩到了。
`find . -name __pycache__ -prune -exec rm -rf {} +` 之后同一条命令 `174 passed`。

**本轮所有变异数字都是清完缓存并 `PYTHONDONTWRITEBYTECODE=1` 之后重跑的**，
9.2.16.1 / 9.2.16.3 两张表都是重跑后的值（与首轮逐条一致，说明那两张表没受影响）。

**给后来人的规矩**：跑"改生产文件 → 跑 → 还原"这类变异，要么每次在一次性副本上改，
要么 `PYTHONDONTWRITEBYTECODE=1` 并先清 `__pycache__`。
**只对比 SHA 证明"文件还原了"是不够的** —— 还原的是源文件，跑的是字节码。

#### 9.2.16.7 全量对拍

`hope_training/whole_body_tracking/tests` + `tests` 两个目录整包，同一棵树、同一解释器、
`-p no:randomly -n 12 --continue-on-collection-errors`：

| | before（HEAD `559b95f4` 原样） | after（+ 本轮 6 条新测试） |
| --- | --- | --- |
| passed | `10138` | `10143` |
| failed | `267` | `268` |
| **skipped** | **`132`** | **`132`** |
| errors | `19` | `19` |
| 用时 | `877.52 s` | `886.86 s` |

`+5 passed / +1 failed` 而不是 `+6 passed`：失败集合逐条 diff，**唯一多出来的那条**是
`test_canonical_motion_compile_cli.py::test_signal_after_atomic_publication_waits_for_identity_then_cleans`，
与本轮无关。在**干净 HEAD 的另一棵树**上单独跑它 10 次：`9/10` 通过 —— **它本身就是一条 flake**，
已登记进待办。`10138 + 6 - 1 = 10143` 恰好对上。`19` 个 error 前后逐条相同
（全是 `test_materialize_a3_vendor_identity_manifest.py` 的 vendor-identity 类，先于本轮）。

受影响面窄一档的那份（全仓 grep 出的 `26` 个提到 `action_ball_runtime` / `ActionBirthBroker` 的模块）：
`15 failed / 851 passed / 0 skipped / 0 errors`，失败集合是 2 条 `frozen_canonical_sha` +
13 条 `train.py` actor 合同族，**全部先于本轮就红**，与上一轮报告列的集合逐条相同
（上一轮报的 `2 errors` 本轮已不存在，那两个模块现在能收集了）。

`tests/test_motion_backhand_loop_b_table_net_clearance.py` 在干净 HEAD 上 `10 failed / 38 passed`
—— 与 §9.2.14 的说法一致，是冻结几何钉子与现役 `geometry.py` 漂了，不是本轮引入。

#### 9.2.16.8 这一节没做什么

1. **没碰** `hope_rewards.py` / `commands.py` / `train.py` / 两个 211 launcher /
   `action_ball_4096x5_prelong_gate.py` / `test_action_ball_4096x5_prelong_gate.py` ——
   本轮期间有别的 workflow 正在改它们。所有验收都跑在**已提交的 HEAD** 上。
2. **没判** canonical 车道剩下那两个零调用点的门，仍是"只登记不判决"。
3. **没修** `test_motion_backhand_loop_b_table_net_clearance.py` 那 10 条既有的红，
   也**没修**上面新发现的 `test_signal_after_atomic_publication_waits_for_identity_then_cleans` 那条 flake。
4. **没有**把 9.2.16.6 那条 `__pycache__` 规矩写成任何脚本或护栏，只写进了文档。

### 9.2.17 切 0807 新盘 + 0808 新动作库：噪声开那一档两格都实跑过了 boot 门；重铸**没做**，卡在两份 GPU 出生证据（2026-08-08，pod1 GPU1）

**人话先说四句。**

1. **机器人（plant）其实早就换完了**——`82ee3ae8` 已经把 `agibot_a3.py` 的默认资产指到 0807，
   本轮不需要"切"，只需要核实。真正没换的是**动作库**：现役谱系仍然指着
   `assets/motions/chingmu73_measured_v4_20260803/`，而新库
   `assets/motions/chingmu73_measured_a3p0807_20260808/` 已经在树里了。
2. **噪声开的那一档（A1/C1）今天就能开机**——本轮在 HEAD 上实跑过，两格 `materialize → recipe`
   全部 `EXIT 0`。**它不需要另一套 profile/lineage/artifact**，与 A0/C0 共用同一份谱系；
   开关是发车时选格子决定的。所以"噪声必须开"这件事**不构成任何新的重铸工作量**。
3. **重铸本轮没有做，而且是故意的。** 换库会让两份**Isaac 出生证据**过期
   （split-ready 动态就绪 artifact + nominal-hold 收据），它们各自绑着旧库的 npz 摘要，
   而核心物化器要求它们 `dynamic_ready_status == "PASS"` 才肯往下走。
   这两份只能由 GPU 跑出来，本轮没有跑。**铸一半 = 下次还得重铸一遍，违反"只铸一次"。**
4. **题目身份这次一定会变**，见下面「题目身份」小节，别把它当成没变。

#### 9.2.17.1 落点核实（三层查，不按提交标题猜）

| 东西 | 落点 | 怎么核的 |
| --- | --- | --- |
| 新 plant（MuJoCo） | `agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3p_pingpong_0807/a3p_pingpong_0807.xml`，SHA `7bbda723f339bdf252a20622afa7a7d53a6fca97464252c66c6e1a45199bcae1` | 与 `configs/a3p_p1_0807_pod_verification_v1.json` 的 `mujoco_root_sha256`、`configs/chingmu73_measured_a3p0807_bank_v1.json` 的 `plant.mjcf_sha256` 三处同值 |
| 新 plant（Isaac） | `assets/agibot_a3p_p1_0807_v1/urdf/model.urdf`，由 `agibot_a3.py:48` 的 `AGIBOT_A3_ASSET_ROOT` 指向 | 读活代码，不读提交标题；**已经是默认，本轮无需改** |
| 新动作库 | `assets/motions/chingmu73_measured_a3p0807_20260808/`，73 条 npz，闭包 SHA `1a6a2f579e765af1…` | `configs/chingmu73_measured_a3p0807_bank_v1.json` 的 `closure` |
| 本轮那一条 | `hope_Take_061_unit04_BH.npz`：旧 `aab1953b9a857d0a…` → 新 `c7eb9af036b5f47b…` | 本机 `shasum -a 256` 直接量 |
| 力矩-转速权威 | `configs/a3_motor_tn/*` + `configs/a3_motor_tn_envelope_v1.json`，审计工具 `scripts/audit_bank_motor_tn_envelope.py` | 它是**审计侧**权威，**不进** A/C 发车链；四格谱系一处都不引它 |

#### 9.2.17.2 影响面：三次扫描，最危险的那一处三次都扫不到

按老规矩扫三遍（按摘要、按完整路径、按目录名），排除 `logs/`。合计 **214 处路径命中 / 约 180 个文件**。
三遍各自补到的洞都真实存在：只带摘要不带路径的有 3 个文件（其中就有四格合同本身），
把路径拆在目录边界上的有 1 个（`mujoco_native/tests/test_action_ball_c211_env.py`，
`git grep` 整条路径看不见它——和上一轮 `test_migrate_action_ball_solver_pin_receipt.py` 同一个形状）。

**活代码里的钉（漏一个就是硬崩）：**

| 位置 | 常量 | 漏了会怎样 |
| --- | --- | --- |
| `scripts/action_ball_211_four_grid_contract.py:184` | `CANONICAL_MOTION_SHA256` | `action_ball_211_four_grid_prelong_barrier.py:686` 会拒掉**每一个** C211/A211 发车 |
| `scripts/materialize_measured_action_ball_n1_bundle.py:70-79` | `ACTION_FACTS`（`motion_path` 拆成两行 + `motion_sha256`） | 核心物化器根本铸不出新 bundle |
| `scripts/materialize_action_ball_n1_fixed_tape_variants.py:44-48` | `MOTION_PATH` / `MOTION_SHA256` | 磁带在读文件算哈希那一步就死 |
| `scripts/launch_n1_measured_vendor_v2_diagnostic.py:146-150` | `MOTION_PATH` / `MOTION_SHA256` | VendorV2 诊断发车被拒 |
| `tasks/tracking/stage1_natural_clip_contract.py:29-75` | 三条 Stage-1 车道，**另外三条 clip** 的路径+摘要 | import 期权威，训练/发车/进度台账都读它；而且它还带 `frame_count`/`strike_frame`/`cycle_seconds`，**不是只换摘要就完事** |

**最危险的一处，三次扫描全都看不见**：
`scripts/launch_action_ball_a211_four_arm_diagnostic.py:124-132` 手抄了三份**下游 artifact 的文件摘要**——
`SPLIT_READY_DYNAMIC_ARTIFACT_SHA256 = ab6b7e41…`、`SPLIT_READY_NOMINAL_HOLD_SHA256 = c8b92a28…`、
`SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256 = ad17d984…`，在 `:1945-1950` 判等，不等就
`raise LaunchRefused("A211 split-ready authority bytes differ")`。
这三个常量里**没有动捕摘要**，所以按动捕扫怎么扫都扫不到它；
但这三份 artifact 的**内容里**都封着旧库的 `aab1953b…`，一重铸它们文件摘要就变，A211 直接起不来。
`launch_action_ball_c211_diagnostic.py:75` 把 A211 launcher 当模块 import，`:1199-1206` 用的就是这三个常量，
所以 **C 族也一起崩，而且 C 族没有自己的副本可改——改 A 那一处两族一起好**。
上一轮的教训是"A211 的指针链差点被漏掉"；这一轮它换了个形状又出现了一次：
**这次不是指针漏了，是指针指向的东西的摘要被手抄进了活代码。**

**产物侧**：现役四个目录里，一个事实（这条动捕）用了**五个不同字段名**存：
`motion.path`/`motion.sha256`（嵌套）、`motion_path`/`motion_sha256`（平铺）、`npz_sha256`，
以及 `offline_n1_tape_build_report` 里一个**光叫 `motion`** 的裸字段
（`materialize_action_ball_n1_fixed_tape_variants.py:1243` 写的）——
任何按 `*_sha256` 结尾做的机械改写都会静默漏掉它。
指针链：C211 谱系 8 条指针**过期 7 条**，A211 谱系 7 条**过期 6 条**；
唯一干净的是 `dr_l0_manifest`，而它偏偏用 `file_sha256`/`contract_sha256` 而不是 `sha256`——
按 `"sha256"` 键写的机械改写器会跳过它（这次跳对了，下次不一定）。

**顺带查实的一条好消息**：全树没有一处路径与摘要**互相矛盾**的钉，
也没有任何一处带着"既不是 v4 也不是 0808"的孤儿摘要。35 对相邻的 v4 路径+摘要全是 `aab1953b…`，零例外。

#### 9.2.17.3 为什么本轮**不铸**：卡在两份只能由 GPU 产出的出生证据

`materialize_measured_action_ball_n1_bundle.py prepare` 的合同要求
"dynamic-ready artifact 与 nominal-hold receipt 必须点名**这个** action 和**这条** motion SHA"，
并在 `:1305` 硬判 `prepared["claims"]["dynamic_ready_status"] != "PASS"` 就拒。现役这两份是：

- `configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json`（`kind = agibot_a3_action_dynamic_ready_candidate_v2`，内容里 3 处旧摘要、2 处旧路径）
- 同目录 `take061.robust20n.nominal_hold.v1.json`（`kind = isaac_action_ball_nominal_hold_v1`，顶层 `motion_sha256 = aab1953b…`，正文带 5 张 Isaac 截图的 SHA）

第二份按构造就是一次**真实 Isaac 启动**的产物（截图、逐 tick 安全 telemetry、终止原因），
不是离线能算出来的东西。第三份 `take_061_unit04_bh.frame0_exact.v1.json` 同理绑着旧摘要。
`--offline-core-only-without-dynamic-ready` 这条旁路会把状态写成 `BLOCKED_EXTERNAL_EVIDENCE`，
于是 `finalize` 与两个谱系物化器都过不去——**它是给"证据还没有"的场景留的，不是给"证据过期了"用的。**

所以本轮的判断是：**先不铸**。理由和 `17f5e30d` 那次一样——
靶子不齐就往下铸，铸出来的东西下一轮还得再铸一遍，而纪律是"只铸一次"。

**要铸需要按这个顺序，缺一不可（本轮已核实每一步的工具与入口，未执行）：**

1. 新库缺 `BANK_IMPORT_RECEIPT.json`。三个活的消费者要它：
   `materialize_measured_action_ball_n1_bundle.py:879`、`build_action_ball_manifest.py:530`
   （两处都判 `kind == "chingmu73_measured_racket_schema_v4_repo_import"`）、
   `audit_measured_racket_mechanical_admission.py:431,527-528`（读它并钉它的摘要）。
   现役 prepared-core 还按 `{path, sha256}` 钉着 v4 那一份的 `e6f0283f…`。
   **注意那个 `kind` 里的 "v4" 是 schema 版本，不是库版本**——新库的收据要**保留同一个 kind 串**，
   改成 `..._a3p0807_...` 会让三处判等一起拒。全树扫过：`_v4_` / `schema_v4` 这类命中里，
   有 5 处是 schema 版本（`materialize_a3_dynamic_ready_contract.py:66`、
   `build_action_ball_manifest.py:530`、`materialize_action_ball_a211_lineage.py:496`、
   `check_table_obstacle_scene.py:750,897`），**一处都不该跟着库改**。
   还有两处是彻底的假阳性：两支 mujoco_native 发射器里的
   `"phase_aware_measured_v4_teacher": True` 是**开关名**，不是库引用。
2. `build_action_ball_manifest.py build` → 新的 73 条源清单 + buildreport。
3. `audit_measured_racket_mechanical_admission.py --bank <新库>` → 机械准入。
4. `audit_materialized_measured_racket_npz.py` → 这一条的拍面 FK 审计（11 道门）。
5. **GPU**：重出 dynamic-ready v2 + nominal-hold 收据 + frame0_exact artifact。
6. `pin_action_ball_profile_contracts.py --source-rev <新提交>` → 新 pin 模板。
7. `prepare` → 磁带 → C211/A211 两条谱系（两个谱系物化器都**拒脏树**，
   A211 还额外拒未跟踪的 action manifest，所以顺序上必须先落一笔提交）。
8. 同批改掉 9.2.17.2 表里的活代码钉，**包括** `launch_action_ball_a211_four_arm_diagnostic.py:124-132`
   那三个手抄摘要。

#### 9.2.17.4 噪声开的那一档：两格都实跑过了 boot 门

干净 detached worktree `/workspace/franco/bankswap_20260808` @ `766ccf91`，GPU1
（GPU0 = yikang 的 `phase114_v2_prep15`、GPU2 = mjlab，全程未碰；GPU1 上另有一个并行 session 的
`remint_20260808` 在跑，等它空出来才起，**没有 kill 任何不属于自己的进程**）。
只跑 `materialize → recipe` 两级，`oracle32 / scale4096 / long` 一律没跑。

| 格 | materialize | recipe | 活值 effective reward sha | 项数 |
| --- | --- | --- | --- | --- |
| `C1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on` | `EXIT 0`（45 s） | `EXIT 0`（57 s） | `5d876e1bac865277…` | 29 |
| `A1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on` | `EXIT 0`（47 s） | `EXIT 0`（57 s） | `0e633996af4790bb…` | 41 |

两格 `terminal_kind = clean_completion`。运行时自陈那行（C1）：

```
[RacketTargetCommand] action-ball runtime bound: actions=1, mobility=no_move,
manifest=12031a74be708b53…, sampler=03d8a77968134b61…,
solver=3af6c505f9ed2333…, physics=aa5c9085f9b48ca6…,
target_source=direct_ball, target_recipe=outcome_dense_only
```

**发车前可核的那个字段在哪**（这是要点，别再靠"应该开了"）：
`<namespace>/launch_claim.json` 的 `canonical_payload.bundle.recipe`（C 族）/
`canonical_payload.bundle.arm`（A 族）里逐字段写着：

```
policy_observation_corruption = True
dr_level_identity              = action_ball_dr_l0n_plant_all_off_proprio_obs_noise_on_v1
observation_noise_axis         = policy_observation_corruption_on_proprioceptive_channels_only
task_channel_observation_noise = False
proprioceptive_observation_noise_channels =
    {base_ang_vel_body: [-0.2, 0.2], joint_pos: [-0.01, 0.01], joint_vel: [-0.5, 0.5]}
```

同一份 claim 里的 `isaac_four_grid_manifest.cells[*]` 仍然四格都在（`cells[0]`/`cells[2]` 是 off、
`cells[1]`/`cells[3]` 是 on）——**那是共用的合同表，不是本格的取值**。
只有 `bundle.recipe` / `bundle.arm` 那一份才是本格真正跑的。引错了字段会得到相反的结论。

**噪声开这一档不需要任何新产物**：`launch_action_ball_c211_diagnostic.py:1637-1660` 的判据是
"谱系绑的是**共用的 DR-L0 leaf**，真正跑哪一档由本格 cell 决定"，
噪声开时它去要 `action_ball_dr_l0n` 那份 payload（由 `training_contract.py:4939`
`action_ball_dr_l0n_contract_payload()` **现算**，不是磁盘上的 artifact），
并要求另一档的键**不存在**。所以 A1/C1 与 A0/C0 **共用同一条谱系、同一份 dr_l0_manifest**。
**结论：噪声开那一档的 profile / lineage / artifact 全齐，一样都不缺。**

**顺带量出来的一件事**：`effective reward sha` 在 C1 与 C0、A1 与 A0 之间**逐位相同**
（C0/A0 的值取自并行 session 在 `0f5fb0dd` 记录的那一跑）。变的是 `recipe_contract_sha256`
（C0 `935246ebcd75e433…` → C1 `dc760b21f2a11ec9…`），它封的是格身份与 DR 档。
**噪声是观测侧的事，不进 reward 配方**——这句话现在是量出来的，不是推的。

#### 9.2.17.5 reward 配方那枚钉：本轮取到的是 A/C 族的活值，**不是** VendorV1 那一枚

`17f5e30d` 判过：`EXPECTED_EFFECTIVE_REWARD_SHA256 = 41631955…53d7` 出自**测试夹具**那条链，
不是活代码。本轮补两条独立佐证，都来自真实 Isaac 启动（不是夹具、不是手抄）：

- A1/C1 两跑的 `runtime_soft_weights` 里 `qdes_projection_penalty = -1.0`——
  与夹具链的 `-5.0` 不符，与活值一致。
- 两跑的 `runtime_effective_reward_sha256` 是 `5d876e1b…`（C，29 项）/ `0e633996…`（A，41 项）。

**但必须把话说死，免得下一轮又用错链**：这两个数**不能**去填
`EXPECTED_EFFECTIVE_REWARD_SHA256` / `STATIC_EFFECTIVE_REWARD_RECIPE_SHA256` /
`EXPECTED_REWARD_RECIPE_SHA256` 那三处。那三处钉的是 **VendorV1 / n1-baseline** 族的配方
（`launch_n1_vendor_baseline_diagnostic.py:165`、`launch_a3_vendor_identity_smoke.py:88`、
`materialize_action_ball_reward_ppo_economy_receipt.py:73`），**30 项**，与 A/C 的 29 / 41 项
不是同一个对象。要拿它的活值，必须实跑一次 `HOPEPingPongActionBallA3VendorV1`——**本轮没跑**。
"复算证明复现的是活值那条链"这条纪律的正确用法就是这个：**先确认自己复现的是哪条链**。

**这枚钉不挡 A/C**：三处常量里没有一处被两个 211 launcher 引用，
而 A1/C1 刚刚在 HEAD 上开机成功，这是活证据而不是论证。它挡的是那两支 VendorV1 发射器。

#### 9.2.17.6 腰的保持增益：新盘上结论不变，而且**落点就是本轮核实的那一份**

另一轮独立重算给的结论（本轮**没有重跑**，只核落点）：
现役盘 `waist_pitch` 需求 `-49.1546 N·m`、包络内只到 `-21.7037`、顶死机械限位 `-26.0146`；
新盘同一帧需求 `-49.8872`、缺口 `28.16`（机械限位下仍缺 `23.85`）。全库 73 条 frame 0 两盘同为 **72/73 撑不住**。

**落点核实（这是唯一要我确认的事，结论：是同一份）**：
`/workspace/franco_waisthold_20260808/out_0807_0808_f0.json` 自陈
`mjcf_sha256 = 7bbda723f339bdf252a20622afa7a7d53a6fca97464252c66c6e1a45199bcae1`、
`npz = assets/motions/chingmu73_measured_a3p0807_20260808/hope_Take_061_unit04_BH.npz`、
`npz_sha256 = c7eb9af036b5f47b62f3e646d68e4fc8523c433a79d0fc037981407b4d7918a7`——
与 9.2.17.1 表里的两个落点**逐位相同**。

**顺带排掉一个本来会成立的质疑**：那一跑的 `grounding.requested = False`（脚离地约 1.3–1.8 cm），
而旧盘那个 `-49.1546` 是"接地后"的数，看起来口径不同。**实际不影响**：
同目录的 `out_0409_v4_f0.json`（接地）与 `out_0409_v4_f0_ungrounded.json`（不接地）
在 `waist_pitch` 上给出**逐位相同**的 `required_torque_qfrc_bias_Nm = -49.15464109801158`，
原因是接地只动 12 个腿关节，而脚的雅可比在这 19 行"contact-free"需求上恰好是 0
（该跑自陈 `leg_perturbation_max_abs_change_on_contact_free_rows_Nm = 1.42e-14`）。
**所以两盘的对比是同口径的**，这一条不用重算。

#### 9.2.17.7 题目身份：这次**一定会变**，别粉饰

上几轮重签时反复引用"`base_question_sha256 = 81eed5139b98…` 没动"来证明题目身份没变。
**这一轮不能这么引。** 上几轮变的是求解器语义面/reward 配方，那些**不是抽题的输入**；
这一轮换的是**动捕本身**——题面里 `time_to_contact`、教师投影、接触对齐全部由这条 clip 算出来，
`base_question_sha256` **按构造必然会动**。

具体会变的三层（重铸时必须逐层量，不能只报一个数）：

1. **`base_question_sha256`**——抽题输入变了，会动。上几轮"它没动"的那句话本轮**作废**。
2. **每份 task receipt 的 `canonical_sha256`**——按构造封着上游，必动。
   （它两个方向都不能当证据：既不能说"变了所以题变了"，也不能说"没变所以题没变"。）
3. **`action_uid`：会变，而且它有两份、会互相打架。**
   `materialize_measured_action_ball_n1_bundle.py:71` 的
   `ACTION_FACTS["action_uid"] = 5527597793770800` 是**硬编码常量**，换库不会自己变；
   但清单侧是**派生**的——`build_action_ball_manifest.py:1362` 调
   `derive_action_ball_action_uid(action_id, family, motion_sha256)`，换库后实测变成
   **`2552478955674699`**。两者在 `materialize_measured_action_ball_n1_bundle.py:644` 判等，
   不等就拒。**详见 9.2.17.10 末尾那一节**（本条已按那里的实测就地更正过一次：
   初稿只写了"硬编码常量所以不会变"，那只说对了一半）。

跨轮可比性的代价，直说：**0808 之后的任何一格，与 0808 之前跑过的所有格子，题目不是同一道。**
所以旧格的曲线只能当"同一个任务族的历史参考"，**不能与新格同图比较、不能算 delta**。
这正是"可以直接用新的模型和动作"这句授权本身的代价——它是对的，但它有价格，价格就是这一条。
残差层面的安慰是有的（`configs/a3p_p1_0807_retarget_attempt_v1.json`：73 条 clip × 18 道门，
两盘**零门差异**，p95 残差最大动 0.11°/0.82 mm，约 1% 门预算），但**"题目相近"不等于"题目相同"**。

#### 9.2.17.8 回归账，以及 registry 那枚漂了的钉：它**一条测试都没红**，这比"红 5 条"更糟

**回归账**（pod1，worktree `/workspace/franco/bankswap_20260808` @ `766ccf91`，干净；
解释器 `/workspace/hope_isaac_venv/bin/python`（Python 3.10.18），**不是** `/usr/bin/python3`；
`CUDA_VISIBLE_DEVICES=` 空，**全程未占 GPU**；`-q -rs -p no:randomly -n 16`）：

| 集合 | 结果 |
| --- | --- |
| 8 个模块：四格合同 / DR-L0N 噪声 / A-C 配置对等 / 两个 211 launcher / 核心物化器 / A211 谱系物化器 / vendor action registry | **`427 passed`，`0 skipped`**，退出 `0`（36.8 s） |

本轮一行代码都没改，所以这份回归的意义是**基线**，不是"改动 vs 基线"。

**然后是那枚钉。** 交接说 `a3_vendor_action_registry.py` 的 `identity_repin_producer`
钉着 `materialize_a3_vendor_identity_manifest.py @ b90bac5f30d8…`，而该文件已经变成
`ab7f8fdb0d53…`，并说这会**红 5 条测试**。

**漂移是真的，逐位复算过**：`scripts/materialize_a3_vendor_identity_manifest.py` 活值
`ab7f8fdb0d532b3f0c1f51d9c50e366b235e3195eaae5363bab0df80d8910bb6`；
钉在 `scripts/a3_vendor_action_registry.py:85`（`bh_loop_c`）和 `:161`（`bh_block`）的是
`b90bac5f30d801b02e4c074a95ae207493214d91938d91890590a7c1aeeb801a`。

**但"红 5 条"这个数不成立，真实是红 0 条 —— 而这正是坏消息。**
把 registry 的 15 个 `ArtifactPin` 字段 × 2 个 action = **30 枚钉逐个对磁盘解析**，
结果是 **30 枚解析、2 枚漂**，两枚都是 `identity_repin_producer`。
而 `tests/test_a3_vendor_action_registry.py:98-111` 那个"对磁盘验摘要"的循环
**只遍历 6 个字段名**：

```
runtime_authority_receipt, dynamic_ready_candidate, nominal_hold_receipt,
contact_bundle, fixed_domain_initial_receipt, reward_economy_receipt
```

`identity_repin_producer` **不在里面**。15 个字段里有 **9 个**（`stable_motion`、
`stable_source_manifest`、`stable_source_prototype`、`identity_repin_producer`、
`identity_prototype`、`identity_repin_receipt`、`identity_manifest`、
`required_identity_manifest`、`runtime_contract`）**从来没有被对磁盘验过** ——
30 枚钉里 18 枚在这道门的视野之外。所以 `427 passed` 不是"钉没漂"，是**没人看**。

**同一件事的另一半**：`tests/test_launch_a3_vendor_identity_smoke.py:112-122` 造夹具时
把 `b90bac5f30d8…` **手抄了一遍**。所以就算把 `identity_repin_producer` 加进那 6 个字段，
这条测试也会跟退役值达成一致 —— 又是"夹具自己造契约"那个形状，和 reward 配方那枚钉
（`17f5e30d`）一模一样。**修的时候两处必须同批**：把 9 个字段补进解析循环，
并把夹具的期望值改成从 registry 现取而不是手抄。

**结论口径**：这枚钉与动作库无关，本轮不修（不在授权范围内，且有并行 workflow 在动这批文件），
但把数字更正在案：**不是"会红 5 条"，是"一条都不会红，因为这类钉有 60% 没人验"。**

#### 9.2.17.10 新库那四份前置产物全部做出来了（CPU，未占 GPU），并且撞出一个会让 `prepare` 直接拒收的硬拦路

pod1，worktree `/workspace/franco/bankswap_20260808` @ `766ccf91`（**全程保持干净**，未提交未推送），
产物全部落在 checkout 之外的 `/workspace/franco/bankswap_out_20260808/`；
解释器**全部**是 `/workspace/hope_isaac_venv/bin/python`（3.10.18，mujoco 3.10.0、numpy 1.26.4、scipy 1.15.3），
`CUDA_VISIBLE_DEVICES=` 空，**一块 GPU 都没占**。逐条 argv / 退出码 / 输出 SHA 记在
`bankswap_out_20260808/RUN_LOG.json`。

| # | 产物 | 状态 | 输出 SHA-256 |
| --- | --- | --- | --- |
| 1 | 新库 `BANK_IMPORT_RECEIPT.json` | DONE | `01995ac4a6150898…` |
| 2 | `Take_061_unit04_BH` 拍面 FK 审计（新盘） | DONE，**11 道门全 true** | `9a7ca2854217e8d8…` |
| 3 | 新库全库机械准入审计 | DONE（退出 `2` = 观测到硬失败，是它的正常语义） | `b1ab074dc5cb506e…` |
| 4 | 新库 ActionBall 源清单 + buildreport | DONE，**73 条，零门开火** | 文件 `4177928a098d0a0c…` / canonical `c726a6c50d7dcc75…` |

**1. `BANK_IMPORT_RECEIPT.json` 没有生产者——这是查出来的，不是猜的。**
全树 `git grep`（含 `scripts/legacy/`）只找到**消费者**，没有任何脚本写它。
再往前一层查清了 v4 那份**是怎么来的**：它的 `actions[]` / `denominators` / `created_utc` /
`authorization` / `source_manifest` 是 v4 `COMPLETION_MANIFEST.json` 的**逐字节切片**——手工切出来的。
这条路对新库**走不通**，因为新库的 `COMPLETION_MANIFEST.json` 结构完全不同，**根本没有 `actions[]`**。
所以本轮另写了 `derive_bank_import_receipt.py` 机械派生，任一交叉校验对不上就 fail-closed：
`frames` 取自 **npz `joint_pos.shape[0]`**（与 catalog `T`、manifest `T` 三方对拍），
`robot_mount_normal_sign` 取自 **npz 自身字段**（与 catalog 对拍），
`sha256` **重新哈希 npz 字节**（与 `chingmu73_measured_a3p0807_bank_v1.json` 的 `closure.files[]` 对拍）。
**一个数都没从 v4 那份抄。**

派生完再与 v4 逐字段比对，"源没变"这句话因此是**量出来的**：
`sha256` **73/73 全不同**；`clip_id` / `file` / `frames` / `hit_frame_50` / `uid` /
`robot_mount_normal_sign` **73/73 逐位相同**；11 个 denominator 全部与 v4 逐位相同（而且是重算的，不是抄的）；
两个库的 `SOURCE_MANIFEST.json` 与 `SOURCE_CLIP_ORDER.json` **逐字节相同**。

**2. 机械准入：`COMPLETION_MANIFEST` 那六个数，六个全部复现。**

| reason | 收据声称 | 本轮复现 |
| --- | --- | --- |
| `joint_position_limit_violation` | 55 | **55** |
| `stored_joint_velocity_limit_violation` | 20 | **20** |
| `finite_difference_joint_velocity_limit_violation` | 39 | **39** |
| 三条 `*_unavailable` | 73 / 73 / 73 | **73 / 73 / 73** |

**本轮这条 clip 自己的账要单独说**：`take_061_unit04_bh` 是
`mechanical_verdict = UNKNOWN` / `kinematic_limit_verdict = PASS`。
关节位置 0 违规（最小归一化余量 `0.1404`，最紧的是 `right_ankle_roll_joint` @ frame 14）、
关节速度 0 违规（存储峰值比 `0.606`、有限差分 `0.611`）。
**它没有任何一条真实包络违规**，挡它的完全是那三条"判它所需的数据不存在"的厂商缺口。
这与 §9.2.17.6 的腰那条是**两回事**，别混：腰那条是"撑不住"，这条是"没法判"。

**3. 求解器 pin：runbook 里那个数是退役值，不能抄。**
`docs/operations/setup_local_sync.md` 的 v4 命令写着 `--solver-profile-sha256 6b2c7c669bfa…`。
本轮按钉针脚本自己的规矩，在 HEAD 上跑 `pin_action_ball_profile_contracts.py --source-rev HEAD` **现铸**，
它**逐字节复现**了在役的 `action_ball_profile_pins.live.v2.9a909b56d9b6.json`——
这就是"复算证明复现的是活值那条链"该有的样子。于是：
runbook 的 `6b2c7c66…` **STALE**（`42232b84` 那次重签把它换掉了），
本轮用的是活值 `3af6c505f9ed2333…`；`physics` 侧 `aa5c9085f9b48ca6…` 重铸后同值，**确认稳定**。

**4. 新清单 vs v4 清单的逐字段差**：`action_id` / `family` / `mount_normal_sign` / `strike_phase` /
`reference_t_hit_s` / `reference_t_cycle_s` / `reaction_margin_s` / `teacher_rate_*` **73/73 相同**；
`motion_path` / `motion_sha256` / `action_uid` **73/73 全变**；
`reference_racket_site_speed_mps` 11 条、`ball_profile` 4 条在浮点噪声量级上变
（中位数 `2.1469600 → 2.1469598`）。两条 WARN 与 v4 build **是同一批 UID，逐个相同**。

##### 撞出来的硬拦路：`action_uid` 会变，而它在两个地方各有一份，**会互相打架**

这一条**更正 §9.2.17.7 里我自己写错的那半句**（已就地改掉，不留矛盾段落）：

- `materialize_measured_action_ball_n1_bundle.py:71` 的 `ACTION_FACTS["action_uid"] = 5527597793770800`
  **是硬编码常量**，换库不会自己变——这半句是对的。
- 但清单侧**不是**常量：`build_action_ball_manifest.py:1362` 调
  `derive_action_ball_action_uid(action_id, family, motion_sha256)`，
  而该函数的 docstring 写得明明白白：*"a manifest cannot … replace its motion bytes while keeping an old
  wire identity"*。所以换库之后这条 action 的 uid 是
  **`5527597793770800` → `2552478955674699`**（本轮新清单里实测的值）。
- 两者**会打架**：`materialize_measured_action_ball_n1_bundle.py:644` 硬判
  `source_identity["action_uid"] != action_uid` 就拒。
  **换库而不同批改 `ACTION_FACTS["action_uid"]`，`prepare` 第一步就 fail-closed。**

所以 §9.2.17.3 那张八步单子要补一条：第 7 步之前，`ACTION_FACTS` 要改的**不止 `motion_path`/`motion_sha256`，
还有 `action_uid`**。而且这一条同时把题目身份的话说死了：
**wire identity 按设计就会变**，这不是副作用，是 `derive_action_ball_action_uid` 的设计意图。

##### 顺带查实的三件事，下一轮会撞上

1. **仓库自己编译不了 0807 的 MJCF。** 逐字报错：
   `ValueError: Error: Error opening file 'meshes/pelvis_link.STL'`。
   `agi/A3_MuJoCo_Sim/.../a3p_pingpong_0807/` 里只有 `a3p_pingpong_0807.xml` 和一个把 `meshes/`
   全部忽略掉的 `.gitignore`。本轮是从 pod 上另一个工作根**补齐 94 个文件**并逐个对
   `configs/a3p_p1_0807_model_set_v1.json` 的 `mujoco.closure` 验过（SHA + 字节数 + 总计 25389870，零多余）才跑起来的。
   **任何人在干净 checkout 上重跑新盘 FK 审计都会先撞这一记。**
2. **v4 那条链在 HEAD 上已经不可复现了。** v4 `COMPLETION_MANIFEST` 钉的 `canonical_mjcf` 是 `2ab1cd31bff…`，
   而 `agi/.../a3_pingpong/a3_pingpong.xml` 在 `766ccf91` 上是 `70c4fd6534f…`。
   **能在 HEAD 上复现的只有 0807 那条。** 顺带说明：`configs/a3p_p1_0807_model_set_v1.json` 里
   `compiled_with_mujoco: false` / `required_next: 在带 MuJoCo 的 pod 上编译` 这一条，本轮**顺手完成了**
   （MJCF 载入并在 73 条 × 5107 帧上 FK 干净）。
3. **0807 的 73 份求解器报告在仓库外**：`/workspace/franco/a3p0807_retarget_20260807/v4out/0807/*.report.json`
   （73 份全 `admitted:true`、全门 true、MJCF 钉 `7bbda723…`；该工作根 `bank/0807` 下的 73 条 npz
   与仓库里的**逐字节相同**）。那个临时目录一旦被清，`solver_admitted` / `solver_all_gates_true`
   就**再也算不回来**——新库自己的 `COMPLETION_MANIFEST` 只写了 `solver_admitted: 73`，
   **压根没写 `solver_all_gates_true`**。
   同一个工作根下还有另一套**更严格**的求解器输出（`out/0807`，73 条只出了 17 份、`admitted:false`），
   **别把那套误当成本库的求解器报告**——本库是用 legacy `v4d` 求解器建的，
   `configs/chingmu73_measured_a3p0807_bank_v1.json` 的 `tools.solver` 写着。

**这四份产物本轮一份都没有提交进仓库。** 它们是重铸的**输入**，而重铸本身仍然卡在
§9.2.17.3 那两份 GPU 出生证据上；先落输入再落不了后半段，等于把"只铸一次"拆成两次。

#### 9.2.17.9 这一节没做什么

1. **没铸任何产物**，一份都没有。理由见 9.2.17.3。
2. **没改任何活代码**——9.2.17.2 那张表是清点，不是改动。
3. **没跑** VendorV1，所以那三处 reward 钉的活值本轮**没有**取到。
4. **没跑** `oracle32 / scale4096 / long4096`，也没跑 MuJoCo-GPU（mjlab）那两格——
   本轮的 GPU 证据只覆盖 Isaac 侧 A1/C1 的 `materialize → recipe`。
5. **没碰** GPU0（yikang）与 GPU2（mjlab），也**没 kill** GPU1 上并行 session 的进程。
6. **没修** 9.2.17.8 那枚 registry 漂钉，也**没补**那 9 个从没被对磁盘验过的字段 ——
   本轮期间有并行 workflow 在改这批文件，且它与动作库无关。只更正了数字并写清修法。
7. **没把** 9.2.17.10 那四份新库前置产物提交进仓库 —— 它们是重铸的输入，
   而重铸卡在两份 GPU 出生证据上；先落输入再落不了后半段，等于把"只铸一次"拆成两次。
8. **提交元数据勘误**：`454acc4f` 那一笔的**标题写错了**，它显示的是 `0f5fb0dd` 的标题
   （一个并行 session 留在 `.git` 里的待用消息被 `git commit` 捡了去）。
   它**实际装的是** 9.2.17.8 那一节（回归基线 + registry 漂钉）。
   该提交已经推送，按纪律不 amend，在此登记更正。

### 9.2.18 把 A211/C211 接上 MuJoCo GPU 回路：能接的只有一条，而那一条底下压着一个漂了四天的 4 倍价（2026-08-08，pod1 host-only，全程未占 GPU）

**人话（先看这四句）**

1. **差异表不用重造**，§9.2.9 那 `17` 轴今天在 HEAD 上当场重跑过，仍然是
   `5` 对齐 / `10` 要紧差异 / `2` 有理由差异,一条没变。
2. 那 `10` 条要紧差异里,**没有一条是"接线就能接上"的**:要么卡在 measured teacher /
   question solver / motion command manager 这种**本车道造不出的产物**,要么是
   "一行就能改、但会改训练分布"的**发车决定**。这是 §9.2.9 六已经写过的结论,今天复核成立。
3. **但有第十一条,§9.2.9 没数进去,而它恰恰是能接的那条**:两条车道对"Isaac 的奖励权重
   到底是多少"这个问题,**读的层级都是错的** —— 读的是 cfg 类体里声明的那一份,而真正
   生效的是**发车解析后**的那一份。
4. 顺着这条查下去,查到一个真的错值:MuJoCo 侧 C 族镜像的 `upright_exp` 权重是 `1.0`,
   Isaac 现役是 `0.25`,**差 4 倍,从 2026-08-04 起漂了四天,期间所有的门都是绿的**。

#### 9.2.18.1 (a) 差异表:§9.2.9 那张今天仍然作数,原样复用

在 HEAD `766ccf91` 的干净 worktree 上重跑 `mjlab_lane/isaac_alignment.py`
(`/workspace/mjlab_venv/bin/python`,Python 3.12):

```
axes : 17    aligned : 5    divergent_blocking : 10    divergent_declared : 2
unverifiable : 0            cross-engine comparable : False
```

与 §9.2.9 一致。**所以本轮没有重造差异表**,按票面要求直接复用。
配套的 `mjlab_lane/tests/test_isaac_alignment.py` **`21 passed / 0 skipped`**。

顺带就地改准一处**手抄错的第二份拷贝**:台账里 vendor 动作解码上界那句写的是
`0.647 rad / 腰偏航`,活值是 `0.6875 rad / 髋偏航·髋俯仰`。同一句话在
`a3_train_ppo.py` 的 docstring 里 08-06 已经按活值改过(该文件第 `118--126` 行还留着
"原写 0.647,活值读出来纠正"的记录),**台账这一行当时漏了**。
一处改准、另一处没跟,正是本 session 反复栽的第二个形状。

#### 9.2.18.2 (b) 按"影响跨引擎结论可比性"排序之后,能接的到底有几条

票面点名的四条(观测 ABI、动作空间、终止 union、reward 组),逐条查完的裁定:

| 轴 | 能不能接 | 卡在哪 / 为什么不接 |
| --- | --- | --- |
| actor `211` / critic `319` 观测 ABI | **不能** | 需要 measured teacher artifact(`teacher_joint_pos/vel`、三组 racket-site heading)+ 在线 question solver;critic 还需要 Isaac 的 motion command manager 才有 `command 62` / `body_pos 42` / `body_ori 84`。造一个形状对、内容零填充的 `211` 就是**假对齐层**,明确不做 |
| 动作空间 / `action_scale` | **技术上一个 flag** | vendor 模式已实现且逐关节对过活值(`31/31`);把默认从 flat 改成 vendor 会让新 run 与既有 `103` 条 flat 收据不可比 —— **发车决定,不替 Franco 定** |
| 终止 union | **部分已在,其余不能** | `robot_hit_table` 已经"测得到 + 会拒绝"(§9.2.9 三),装成硬终止会改训练分布,是发车决定;参考包络三条(`anchor_pos`/`anchor_ori`/`ee_body_pos`)需要 motion reference,**没有** |
| reward 组(mimic/strike/target/outcome) | **不能** | 分别卡在 measured teacher、question packet、analytic outcome evaluator |
| **reward 权重的"权威层级"** | **能,而且本轮接了** | 见下。这一条 §9.2.9 没单独立轴,但它不需要任何产物、也不是发车决定 —— 纯粹是**两边都在问错一层** |

**所以本轮 (b) 的实际产出只有最后一行。** 把前四条再推一遍只会重新推导出 §9.2.9 六
已经写过的同一份结论;真正没人做过、且做得动的,是权威层级这一条。

#### 9.2.18.3 权威层级:类体声明的权重 ≠ 真正生效的权重

Isaac 侧一条 reward 项的权重要经过**三层**才落地:

```
cfg 类继承链的 weight=   ->   YAML rewards.* 键(含 motion_scale 这种乘法)   ->   reward_pack=v2 直接改写
```

`train.py :: _expand_reward_pack`(`13510` 起)里那句是**赋值**,不是默认值:
`getattr(R, name).weight = float(weight)`(`13588`)。所以在包里的项,包是最终权威。

C211 的 YAML 链(`C211Learnability -> A3VendorV2 -> A3VendorV1 -> ActionBall`)
在 `HOPEPingPongActionBall.yaml` 上设了 `reward_pack: v2`,**三层全部生效**。

这一层区分不是学术问题,它直接决定读出来的数对不对:

| 项 | cfg 类体声明 | 发车真正生效 | 差 |
| --- | --- | --- | --- |
| `upright_exp` | `0.0` | **`0.25`**(pack) | 只读类体会读出一个**根本不会生效**的数 |
| `motion_body_pos` | `1.0` | **`0.15`**(× YAML `motion_scale=0.15`) | `6.7` 倍 |
| `motion_racket_position` | `0.0` | **`0.20`**(YAML 键) | 只读类体会以为这项不存在 |

**两条车道都读错了这一层。** mjlab 台账的 `_probe_reward_surface` 只比项名与锚点,
所以它的裁定没被污染(而且该行的 `caveat` 里本来就写着这句),但 MuJoCo 侧的镜像是**比数值**的
—— 它错得很具体。

#### 9.2.18.4 查到的真错值:`upright_exp` 差 4 倍,漂了四天

`mujoco_native/action_ball_c211_env.py` 那份自称
`action_ball_c211_partial_isaac_synonymous_reward_v3` 的镜像里:

```
mirror  upright_exp = 1.0          Isaac 现役 = 0.25        (差 4 倍)
```

`0.25` 是 **2026-08-04 层级对齐**那次定的价,理由写在 `train.py:13363-13376`:
`upright_exp=1.0` 每步无条件发钱、无 `task_valid` 掩码、RESET_WAIT 内照付,
`500` 步 `gamma=.99` 折扣 `+1.9869` **压过** task-valid mimic 预算 `1.77331`(`112%`)
和 accepted window `1.85151`(`107%`)—— 即"站着不动"比"学动作"挣得多。
改成 `0.25` 后折扣 `+0.4967`,回到辅助项该在的位置。

**Isaac 侧改了,这份 MuJoCo 镜像没跟。** A 族通过
`from . import action_ball_c211_env as shared` 吃的是同一份,所以**两族都受影响**。

其余 `13` 条逐条核对下来**都对**(含 08-08 刚落地的 `action_rate_l2 = -0.1`,票面
点名要核实的那一条 —— 它确实同步了)。所以本轮**只**同步了 `upright_exp` 这一个数,
而且它是同步 Franco 08-04 已经做过的定价决定,**不是新的定价决定**。

#### 9.2.18.5 为什么四天没人看见:唯一在"看"的是一个天天变的整文件指纹

`14` 条镜像权重里,**`13` 条是写死在 `_c211_isaac_synonymous_prior_terms()` 函数体里的
裸字面量**,而且同一个数在本文件里**抄了两遍**(奖励表一遍、收据表一遍,相距 `2800` 行)。
`mirrored_constant_registry` 只看得见**模块级**常量 —— 函数体里的字面量对它是隐形的。

唯一"看着"这件事的东西是收据里这一行:

```
"reward_pack_resolver_source_sha256": _sha256_file(TRAIN_PY)
```

`train.py` 有 `19270` 行,几乎每天都因为无关原因变一次,这个数就天天变。
全仓 grep,**没有任何消费者读它**,也没有任何门会因为它拒收。
**它是"只出计数器没人读"的一个标本,而且比一般的更坏:它出的还是一个天天报警的计数器。**

这与 §9.2.17.8 同一天独立发现的形状是同一个:那边是 `30` 枚 `ArtifactPin` 里 `18` 枚
从没被对磁盘验过(`60%` 没人看),这边是 `14` 条权重里 `13` 条不在任何门的视野内。
**两处都不是"门判错了",是"门根本没看这一格"。**

#### 9.2.18.6 改法:照 `table_termination` 那道已经交过学费的门的同一形状

`table_termination.py:508-512` 那段注释把这件事说清楚过了:
"上面三道门只说源文件的字节跟我钉的一样。源文件一动,把这三行重钉成新值是一行的事,
而这个文件顶部那几个手抄常量跟没跟上,过去没有任何机制在看。"
`OPEN_MIRROR_DEBT` 里 `C211_ACTION_RATE_POST_DT_WEIGHT` 那条债的"怎么修"栏,
写的也正是"给 c211_env 建一张和 `table_termination` 同款的表"。本轮就是把它做了。

- **抬成模块常量并逐个比活值**:`upright_exp` / `base_ang_vel_xy` / `base_lin_vel_z` /
  `joint_vel` / `action_rate_l2` 五条,走 `mirrored_isaac_reward_weight_entries()` +
  `isaac_live_constants.parity_blockers()`。A 族的 `racket_progress` 一并接上。
- **比的是发车解析后那一层**:在包里的项(`upright_exp` / `action_rate_l2`)指向
  `_REWARD_PACK_V2_DIRECT`,不在包里也没有 YAML 键的项才指向 cfg 类体。
- **求值器加两种选择器**,都 fail closed:
  `class_term_weight`(RewTerm 的 `weight=` 关键字,`class_term_param` 够不到它)与
  `pair_table_value`(读 `(name, value)` 表;**遇到读不懂的行是拒绝,不是跳过** ——
  跳过是最坏的失败模式,被跳过的那行完全可能就是要找的那行)。
- **影子检查**:选择器只指一个类。若 C211 继承链上有下游类开始重声明同名项,
  单类选择器会继续拿被遮住的旧值报"对齐"。`declaring_classes()` 扫全链,
  只认唯一那个类,否则开火。这不是假想:`base_ang_vel_xy` / `base_lin_vel_z` 在
  `HOPEHitterPureRewardsCfg` 里各有一份**逐字相同**的拷贝(那个类不在 C211 链上),
  连写变异测试的锚点都得带上行尾注释才唯一;`racket_progress` 在那个类里也有一份,
  只是权重是 `0.0` 而不是 `10.0` —— 即"同名项在别处取着另一个值"这件事本来就在发生。
- **覆盖面自检**:`14` 条实现项必须要么在"比过了"那批、要么在"明写没比"那批,
  **不许有第三种状态**(没人比、也没人记)。
- **记录与阻断同一批**:门放在 `_build_reward_contract` **最前面**(纯静态检查,
  不需要 torch、不需要 native ABI),不一致直接拒收;收据同时写明谁被比过、谁没有、卡在哪。

#### 9.2.18.7 (c) 接不动的,以及为什么

- **剩下 `9` 条 `motion_*` 权重**:权威都要过一道 Hydra YAML
  (`rewards.motion_scale` 或 `rewards.motion_racket_*_weight`),而
  `isaac_live_constants` 的求值器只读 Python AST。要闭掉它得让求值器会展开 Hydra 的
  defaults 链并按后写后赢合成 —— **那要连 defaults 顺序、`@_here_` 语义、包展开一起做对,
  做半截比不做更危险:会给一个"比过了"的假象。** 登记成明写的债,不假装比过。
  同一个能力一次能顺带闭掉 A 族的 `A211_BASE_POSITION_WEIGHT`。
- **那 9 条的"没比"名单里只写项名和权威路径,不写数值** —— 再抄一份数字进去,
  就是给这个文件添**第四份**手抄件。
- **§9.2.9 六列的那批**(`211`/`319` ABI、mimic/target/outcome 三组、参考包络三条终止、
  WAIT/揭示结构、球的接触模型、摔倒阈值 / 动作解码默认 / 撞桌硬终止 / 复位随机化)
  今天复核**结论不变**,不重复列。

#### 9.2.18.8 变异测试:`13` 条,每条都先证明"粗一个档次的检查照样通过"

全部在 tmp 拷贝上跑,本仓源文件不动。每条**先断言粗检查确实过得去**,再断言门变红。

| 变异 | 粗检查为什么抓不到(测试里逐条断言过) | 结果 |
| --- | --- | --- |
| 重放 08-04 那次重定价(`0.25 -> 1.0`) | 包的行数不变、项名不变、`0.25` 这个字面量在 `train.py` 里还有别的出处 | `upright_exp` 变红 ✅ |
| 两条权重对调(`base_ang_vel_xy` ↔ `base_lin_vel_z`) | **和不变**(`-0.55`)、**排序后多重集不变**、**项数不变** | 两条都变红 ✅ |
| 下游类重声明 `joint_vel`,**值写成一模一样** | 逐项比值这一层**完全过得去**(测试里断言 `parity_blockers() == ()`) | 只有影子检查抓得住 ✅ |
| 同上,`racket_progress`(A 族) | 同上 | 影子检查开火 ✅ |
| 删掉包里 `("action_rate_l2", -0.1)` 那一行 | **陷阱题**:它的 cfg 类体值**也是 `-0.1`**,任何"包里没有就退回类体"的读法都会宣布"对齐" | 拒绝作答,不猜 ✅ |
| 包里混进一行非二元组 | "名单格式对不对"式的检查会放行 | 拒绝,不跳过 ✅ |
| "没比"名单少一条 | 名单仍是合法 dict、其余 8 条一字不动 | 覆盖面自检开火 ✅ |
| 合成 blocker 注入 `_build_reward_contract` | —— | 真的拒收,不是只写收据 ✅ |
| 选择器指错层(读类体的 `0.0` 而不是包的 `0.25`) | —— | 该测试把"为什么必须读包那一层"钉死 ✅ |

另加一条**闭环**测试:`_MIRRORED_PRIOR_WEIGHTS` 里的每个值,必须等于奖励内核跑出来
真正收的 `manager_weight`。没有它,这张对账表就可能在给**另一个数**发合格证 ——
也就是又多了一份手抄。

#### 9.2.18.9 回归账

pod1 独立 worktree `/workspace/franco/mjwire_20260808` @ `766ccf91`,**全程 host-only,
未占任何 GPU**(GPU0 = yikang、GPU1 = 我们自己的四格、GPU2 = mjlab,都没碰)。
`-q -rs -p no:randomly`,`PYTHONDONTWRITEBYTECODE=1`。

| 集合 | 解释器 | 基线(干净 `766ccf91`) | 本轮 |
| --- | --- | --- | --- |
| `mujoco_native/tests/` + `test_mujoco_native_isaac_live_constants.py` | `hope_isaac_venv` (3.10.18) | **`233 passed / 0 failed / 0 skipped`** | **`246 passed / 0 failed / 0 skipped`**(`+13` 新增) |
| 8 个相关 Isaac 模块(C211 oracle/launcher/物化器、A/C 配置对等、A211 谱系与四臂发射器、reward flag 覆盖) | `hope_isaac_venv` (3.10.18) | **`648 passed / 1 failed`** | **`648 passed / 1 failed`**(逐位相同) |
| `mjlab_lane/tests/` | `mjlab_venv` (3.12) | — | **`21 passed / 0 skipped`** |

那 `1` 条红是 `tests/test_reward_flags_overrides.py:3036`,**基线上就红**
(`assert 9 == 7`,一个 YAML 声明面的计数),与本轮无关,本轮也没修。

**门的运行代价**(实测):冷启 `179.7 ms`(解析 `train.py` `19270` 行 + `hope_env_cfg.py`),
**每个进程只付一次**(`_build_reward_contract` 在构造时调一次)。
热路径原本 `223 ms`/次(每次问一个类都 `ast.walk` 整棵树,C+A 两族共 `14` 趟),
加了 `_class_index` 缓存后是 **`0.46 ms`**。

#### 9.2.18.10 这一节没做什么

1. **没有把 A211/C211 的任务搬到 mjlab GPU 车道上**。搬不动的理由在 9.2.18.2 那张表里
   逐条写着,和 §9.2.9 六一致。**"MuJoCo GPU 的 A/C"今天仍然不存在。**
2. **没有造任何对齐层**。零填充一个形状对的 `211` 会让台账变绿而语义全假,明确不做。
3. **没跑训练,也没占 GPU2**。本轮改的是 Isaac / MuJoCo-CPU 侧的活值对账,
   在 mjlab GPU 上跑一趟不会经过这段代码,所以那种 smoke 不构成证据,没跑。
4. **没改任何 reward 权重的定价**。`upright_exp` 那一个数是把镜像同步到 Franco
   08-04 已经做过的决定上,不是新决定;其余 `13` 条一个字没动。
5. **没读 Hydra YAML**,所以那 `9` 条 `motion_*` 与 A 族 `base_position` 仍然只是
   "明写没比",不是"比过了"。
6. **没碰**别的 session 正在改的那批文件(`action_ball_curriculum.py`、两个 launcher、
   `build_action_ball_manifest.py` 等),提交只 stage 了本轮这 8 个文件。
7. **没修** §9.2.17.8 那枚 registry 漂钉,也没补那 9 个从没被对磁盘验过的字段 ——
   那是另一条待办,但它和本节是**同一个病**:钉了指纹,没人比值。

### 9.2.19 两条"接完了"的独立验收：14 条变异我自己从零重跑，四格一个字节没动；两处**理由**要更正（2026-08-08，pod1，全程 host-only，未占任何 GPU）

**人话（先看这五句）**

1. 两条接线（`04434abd` DR 档位入口 / `9e69239c` MuJoCo 镜像权重活值对账）的**主结论都成立**。
   我没有沿用它们任何一行验收脚本，自己在两棵新 worktree 上重做了一遍。
2. **14 条变异，14 条全部致命**：拆掉机制 → 指定的测试必红；装回去 → 必绿。
   没有一条是"检查恒真、变异自证"的那种假变异（怎么证的见本节 3）。
   （构成：DR 档位那 7 条**全部**重跑；MuJoCo 那 13 条里挑了最吃劲、最可能空转的 **6 条**重跑；
   外加**我自己加的 1 条** —— 而那一条推翻了一段写在代码注释里的理由。）
3. **四格归因洁净没被破坏，而且是逐字节的**：两族发车 argv 的整串摘要、四格权威的
   `content_sha256`、`TASK_PROFILE_ID` / `TASK_PROFILE_SOURCE` / `LINEAGE_KIND` /
   `DR_L0_MANIFEST_SOURCE` —— 改动前后**逐位相同**。
4. **两处"理由"要更正**（两条都不影响已落地的代码，只影响下一个人会相信什么）：
   一处把"其实已经能做的事"说成了"另一批活"；一处把一道门的失败模式**说反了**。
5. **全量对拍零回归**（数字见本节 6）。

---

#### 1. 怎么验的：不复用它们的脚本，也不复用它们的 worktree

| | |
| --- | --- |
| 机器 / 解释器 | pod1，`/workspace/hope_isaac_venv/bin/python`（Python 3.10.18），mjlab 那一条用 `/workspace/mjlab_venv/bin/python` |
| GPU | **一张都没占**。GPU0 = yikang 的 `phase114_v2_prep15`、GPU1 = 我们自己的四格 `scale4096`（s20r1）、GPU2 空着也没用 |
| worktree | 三棵全新的：`audit_head_20260808`（HEAD `04434abd`）、`audit_base_20260808`（同一 HEAD 上把 `04434abd` 与 `9e69239c` 两枚 **revert 掉**）、`audit_mut_20260808`（变异用） |
| 基线口径 | **不是**"上一个干净提交"，是"HEAD 减掉这两枚提交"。中间还夹着别的 session 的 `993a52f6`，用旧提交当基线会把别人的活算到这两枚头上 |
| pytest | `-q -rs -p no:randomly -p no:cacheprovider --tb=no -rf`，`PYTHONDONTWRITEBYTECODE=1`，串行 |

两枚 revert 都干净落地（`git revert --no-commit`，无冲突），所以 head 与 base 的差就是这两枚提交本身。

---

#### 2. 变异复跑：14 条，14 条全红

（MuJoCo 那 13 条我没有全跑 —— 挑了 6 条：一条历史真事故重放、一条"和与多重集都不变"的对调、
一条"值逐位相同"的影子、一条陷阱题、一条覆盖面、一条"记录与阻断同批"。
没跑的 7 条是它们自带夹具的形状检查，风险面比上面这 6 条低。）

**DR 档位入口（`04434abd`）—— 7 条**

| | 拆掉什么 | 实测 | 红的是谁 |
| --- | --- | --- | --- |
| M1 | 候选 manifest 与代码侧 payload 的**逐字段对拍**（`if declared_axis[k] != resolved[k]` → `if False`） | `39 passed` → **`5 failed / 34 passed`** | 五条轴各一条（`physics_material` / `add_joint_default_pos` / `base_com` / `randomize_link_mass` / `randomize_pd_gains`） |
| M2 | 收据只查封印、不逐字段回比（`if declared != resolved` → `if False`） | → **`2 failed / 37 passed`** | `test_a_resealed_lie_is_still_refused`（改完值把封印也算对）、`test_a_receipt_with_blanked_axes_is_refused` |
| M3 | 两档只比档名（逐轴比对换成空集） | → **`1 failed / 38 passed`** | 反退化那条 |
| M4 | `_validate_lineage` 忽略传进来的档、永远解析默认档 | `184 passed` → **`1 failed`** | A211 的 DR-L1 停车用例 |
| M5 | 去掉斜坡的 `ramp_steps` / `hold_clock_owner` 对拍 | → **`2 failed / 37 passed`** | 斜坡漂移那两条（第三条由另一道独立摘要门挡着，仍绿 —— 也就是斜坡有两道互不依赖的门） |
| M6 | preflight 永远放行 | `362 passed` → **`3 failed`** | 清单那条 + 两族各一条停车用例 |
| M7 | DR-L0 的 profile 中缀飘一个字母（`DRL0` → `DRL0b`） | `518 passed` → **`4 errors`** | 发射器**导入期**就炸，测试根本收集不起来 |

**MuJoCo 镜像权重（`9e69239c`）—— 6 条**（每条都先断言"粗一档次的检查照样过"，见本节 3）

| | 拆掉什么 | 实测 |
| --- | --- | --- |
| N1 | 在 `train.py` 里**重放 08-04 那次真事故**（`upright_exp` 包里的值改回 `1.0`） | `246 passed` → **`16 failed`**（含 `test_c211_reward_weight_mirror_is_green_on_the_real_sources` 与 10 条环境构造用例 —— 门是真的拦，不是只记一笔） |
| N2 | 把 `base_ang_vel_xy` 与 `base_lin_vel_z` 两个权重**对调** | → **`14 failed`**（含专门那条 `test_swapping_two_weights_between_terms_turns_the_mirror_red`） |
| N3 | 让链上更靠后的类**用一模一样的值**重声明 `joint_vel` | → **`12 failed`**（影子检查开火：逐项比值这层完全过得去） |
| N5 | 删掉包里 `("action_rate_l2", -0.1)` 那一行（**陷阱题**：类体值也是 `-0.1`） | → **`15 failed`**（含 `test_a_deleted_pack_row_refuses_instead_of_falling_back` —— 拒绝作答，不许猜） |
| N7 | "没比"名单少一行 | → **`12 failed`**（覆盖面自检开火） |
| N8 | 门只记不拦（`weight_blockers = ()`） | → **`1 failed`** |

**外加一条我自己加的（不在它们的清单里）**：把 `action_ball_211_transition_preflight.py` 那段扫描
改回"只认 v3"。见本节 5(b) —— 这条的结果**推翻了它们写在代码注释与测试 docstring 里的理由**。

**另外我直接问了一次入口本身**（不走它们的测试）：拿一个**根本不存在的** lineage 路径去调
两个发射器的 `_validate_lineage(..., level="dr_l1")`：

* 两族都当场拒，报的是**三条清单原文**（缺 lineage 工件 / 缺 reward 与 policy recipe /
  四格权威装不下 DR-L1），**一个字节的磁盘都没读**（不存在的文件从没被打开）；
* 同样的调用不带 `level`（= 默认 DR-L0）则**继续往下走**，最后死在 `No such file`。

这证明了两件事：DR-L1 的入口是真通的（不是摆设），而且"入口没接"和"入口通了但那一档还没
materialize"在收据里是两种能分辨的失败 —— 这正是它们声称做到的那件事。

---

#### 3. 怎么排除"检查恒真、变异自证"（本 session 栽过的第 1 类坑）

三件事分别查过：

1. **两个 DR 档没有被喂成同一份值。** 夹具读的是磁盘上**两份不同的 tracked manifest**
   （L0 那份在 `configs/action_ball_n1_measured_20260803/`，L1 那份在 `.../20260805/`），
   加上 `training_contract` 里两个不同的 payload builder。M3 证明了这一点是**被机器守着**的：
   把逐轴比对拆成空集，反退化那条立刻红。
2. **两边引擎没有被喂成同一个数。** N1/N2/N3 我都**先断言粗一档的审计照样通过**再要求门变红：
   N1 里包的行数没变、项名没变、`0.25` 这个字面量在文件里还有别的出处；
   N2 里两个权重的**和不变、排序后的多重集不变、项名不变**；
   N3 里下游那份声明的值与上游**逐位相同**。三组粗检查我在测试里逐条 `assert` 过是 `True`，门仍然红。
3. **DR-L0 那张冻结表不是从今天的代码抄回来的**，是改动之前发射器里的原文。
   我另外做了一次**跨 worktree**的核对（本节 4），不依赖那张表。

---

#### 4. 四格归因洁净：逐字节，跨 worktree 对拍

不看它们的测试，直接在 base（revert 掉两枚提交）与 head 两棵树里各跑一遍，比活值：

| 量 | base | head |
| --- | --- | --- |
| A 族发车 argv 里的 `task=` | `HOPEPingPongActionBallA211VendorV2N1DRL0Learnability` | **同** |
| C 族发车 argv 里的 `task=` | `HOPEPingPongActionBallC211VendorV2N1DRL0Learnability` | **同** |
| A 族**整串 argv** 的 sha256（前 16 位） | `e9651d03c3494e1b` | **同** |
| C 族**整串 argv** 的 sha256（前 16 位） | `8daab6ee6049521e` | **同** |
| 四格权威 `CONTENT_SHA256` | `b31d894ea45010985f79abfacec97e723decca18d23784c6159cf017f4e5f44e` | **同** |

再把两个发射器**全部**模块级大写常量各导出一份逐行 diff，差的只有这几类，一条不多：

* `SCHEMA_VERSION 2→3`、`SPEC_KIND`/`CLAIM_KIND` `v2→v3`；
* 新名字 `TASK_FAMILY` / `DR_LAUNCH_LEVELS_SOURCE` / `DR_L1_TASK_PROFILE_SOURCE` / `DR_L1_MANIFEST_SOURCE`；
* `RUNTIME_SOURCE_PATHS` 多了三行钉子（档位权威、DR-L1 profile、DR-L1 manifest）；
* 各种 `*_FILE` 绝对路径（两棵 worktree 目录名不同，属预期）。

**`TASK_PROFILE_ID` / `TASK_PROFILE_SOURCE` / `LINEAGE_KIND` / `DR_L0_MANIFEST_SOURCE` 根本没出现在 diff 里 —— 也就是逐字节没动。**
`53040fb0` 那两道"只差 obs 和 reward"的门在 HEAD 上实跑：连同档位入口与共享常量清单一起 **`368 passed`**。

---

#### 5. 两处理由要更正

**(a)〔把"已经能做的事"说成了"另一批活"〕那 9 条 `motion_*` 权重的活值比对，不需要新造能力。**

`9e69239c` 的自陈是：剩下 9 条 `motion_*` 与 A 族 `base_position` 之所以只能"明写没比"，
是因为"权威要过一道 Hydra YAML，而 `isaac_live_constants` 的求值器只读 Python AST，
要闭掉得让它展开 defaults 链……是另一批活"。

**前半句成立，后半句不成立。** 仓库里**已经有**一份这样的解析器，而且是被 hydra 亲自核对过的：
`hope_training/whole_body_tracking/tests/test_action_ball_211_ac_family_config_parity.py`
里的 `resolve_task_profile()`（`53040fb0` 落地，只依赖 PyYAML，配套
`test_local_resolver_and_overlay_reproduce_hydra` 在有 hydra 的环境里逐字段核对自己和 hydra 的结果）。
我在 HEAD 上直接调它，两族 DR-L0 叶子都当场解析出来了：

```
chain: /base/env_base → /base/sim_base → /base/randomization_base → HOPEPingPongHitter
     → HOPEPingPongActionBall → …A3VendorV1 → …A3VendorV2 → …N1Learnability → …N1DRL0Learnability
rewards.motion_scale = 0.15
rewards.motion_racket_position_weight = 0.2   motion_racket_velocity_weight = 0.2
rewards.motion_racket_normal_weight   = 0.2   motion_racket_long_axis_weight = 0.1
```

—— 正好就是那 9 条缺的两样东西。合成规则也已经查清、不含未知量：
`train.py` 的 `_apply_reward_overrides` 先吃 per-term 的 `motion_<t>_weight` 覆盖，
再把 `motion_scale` 乘到六个 `motion_*` 上（term 为 `None` 的跳过）；
这 9 条**一条都不在** `_REWARD_PACK_V2_DIRECT` 里，所以没有第三层改写。

**所以正确的归档是"还差一步"，不是"接不动"。** 工作量是"把那 85 行解析器
（`_profile_path` / `_split_ref` / `_deep_merge` / `_read_profile` / `resolve_task_profile` / `profile_chain`）
从测试模块提成共享件、再写一条 `class weight × motion_scale` 的合成器"，不是"新建一套 Hydra 求值能力"。
（提出来是必须的：那个测试模块在导入期会 `_load` 两个发射器，生产代码不能直接 import 它。）
它们把这条写进"接不动"，是**把没做的事包装成了做不到**——虽然是善意的（怕做半截给一个假的"比过了"）。

**(b)〔失败模式说反了〕`transition_preflight` 那段扫描，"只认 v3"并不会让旧 claim 变成看不见。**

`04434abd` 在 `scripts/action_ball_211_transition_preflight.py` 里新增了 `RETIRED_CLAIM_KINDS`，
理由写的是（代码注释与测试 docstring 两处都这么写）：
"只认 v3 会让磁盘上的旧 v2 claim 变成看不见 —— 于是可以在已经花掉的 namespace 上重发，
那是放宽一道 fail-closed 门"。

**实测不是这样。** 我把那段扫描改回只认 v3，然后在磁盘上放一份 v2 的 `launch_claim.json`：
它**照样拒**，只是拒的措辞从 `"scale4096 was already claimed before this preflight"`
变成 `"experiment root contains an invalid launch claim"`。看代码也印证：
`outer["kind"] not in claim_kinds` 走的是 `raise TransitionPreflightRefused`，**不是 `continue`**。

**结论：这次改动本身是对的**（把一次"整个 preflight 报错中止"改成"这一格记为已占用"，
措辞准确、行为更可用），**但它不是在补一个洞** —— 严格说它是**略微放宽**（原来会硬中止）。
配套的那条测试也是真的（`match="already claimed"`，只认 v3 时会因为措辞对不上而红，我实测过），
只是它验证的东西和 docstring 说的不是一回事。
**为什么值得写下来**：下一个人如果照这句话去设计新门，会以为"不认识的 kind 会被跳过"，
而这个前提是假的。

**(c)〔顺手记一条，不阻塞〕DR-L1 候选 manifest 的 `runtime_integration_blockers` 现在过期了。**
`configs/action_ball_n1_measured_20260805/action_ball_211_dr_l1_restored_plant_candidate.v1.json`
里那条唯一的 blocker 原文仍然是"两个 launcher 还硬绑 `dr_l0_lineage_v5` 与 `DR_L0_MANIFEST_SOURCE`"——
`04434abd` 之后**这半句已经不成立了**（发射器改成解析、DR-L1 的 profile 与 manifest 都进了钉子表），
真正还缺的只有"materialize 出这一档自己的 lineage"。
发射器读这个字段时**只检查它空不空**（未 materialize 的档必须非空、已 materialize 的档必须为空），
不看内容，所以这条过期文本今天不挡任何事 —— 但它就是"记录与阻断不同批"里"记录那一半没跟上"的样子。
下次给 materializer 加 `--dr-level` 时顺手改准即可。

---

#### 6. 全量对拍：零回归

同机、同解释器、同参数、同一时刻并行跑；测试集 = `hope_training/whole_body_tracking/tests`
\+ `hope_training/whole_body_tracking/mujoco_native/tests`。

| | base（HEAD 减两枚提交） | head（`04434abd`） |
| --- | --- | --- |
| 结果 | `122 failed / 7893 passed / **53 skipped** / 19 errors`（`37:00`） | `122 failed / **7955** passed / **53 skipped** / 19 errors`（`37:02`） |
| 失败集合逐条 `comm` | 两侧各 `122` 条，**新增 0、消失 0** | |
| 出错数 / 跳过数 | `19` / `53` | `19` / `53`（两侧相同） |

**通过数 `+62`，正好等于新增用例数**：新模块 `test_action_ball_211_dr_launch_levels.py` `39` 条
\+ 共享常量清单扩表 `10` 条（`146 → 156`，5 个新名字 × 2 条参数化门）
\+ MuJoCo 侧 `13` 条（`233 → 246`：12 条变异 + 1 条闭环）。**零回归。**

口径如实标注：这一轮我用的是 `-rf`（只列失败），它盖掉了同一条命令里的 `-rs`，
所以**跳过的 53 条我只对了个数、没有逐条列原因**；`19` 个 collection error 同样是按数对的（两侧都是 19）。
失败那 122 条是逐条名字比对过的。

针对性复核（最终字节，HEAD 上）：

* 档位入口 + 共享常量 + A/C 配置对等 + 四格 barrier 四个模块：**`368 passed`**
* `mujoco_native/tests` + `test_mujoco_native_isaac_live_constants.py`：**`246 passed / 0 skipped`**
* `tests/test_action_ball_211_transition_preflight.py`：**`34 passed`**
* mjlab 侧差异表（`/workspace/mjlab_venv/bin/python` 现跑 `isaac_alignment.build_ledger()`）：
  `17 轴 = 5 aligned / 10 divergent_blocking / 2 divergent_declared / 0 unverifiable`，
  `cross_engine_comparable = False`，`blocking_axes` 十条逐字与 §9.2.9 / §9.2.18 相同。

---

#### 7. "接不动"逐条核：一条是"其实能做"，其余都成立

| 谁说的 | 说的什么 | 我的裁定 | 依据 |
| --- | --- | --- | --- |
| drl1 | 两个 materializer 今天只产 DR-L0 lineage，没有 `--dr-level` | **成立** | 两个文件里 `dr_level` 零命中，只有 `dr_l0_manifest` |
| drl1 | DR-L1 **按定义**进不了四格 | **成立** | `action_ball_211_four_grid_contract.py` 是内容封印的（`validate_manifest` 要求与唯一权威**逐字相等**），`matched_contract.start_pose_ramp` 钉死 `None`，DR 身份只封了 L0/L0N，`_require_one_registered_difference_axis` 硬性要求同族两格只在 obs-noise 五键上不同 |
| drl1 | train.py 侧 DR-L1 运行时合同早就落过、本轮一行未动 | **成立** | `_ACTION_BALL_DR_L1_RUNTIME_ATTR` 全历史只被 `4420345a` 碰过 |
| mujoco-ac | actor 211 / critic 319 观测 ABI 接不动 | **成立** | 差异表 `actor_observation_abi` / `critic_observation_abi` 都是 `divergent_blocking`，`closable_by` 点名 measured teacher artifact + 在线 question solver + Isaac motion command manager |
| mujoco-ac | mimic/strike/target/outcome 四组 reward、参考包络三条终止、WAIT/揭示结构、球的接触模型 | **成立** | 同上，`reward_surface` / `termination_union` / `ball_contact_model` 三轴都是 `divergent_blocking` |
| mujoco-ac | 四条一行就能改但会改训练分布的（动作解码默认、摔倒阈值、撞桌硬终止、复位随机化） | **成立，而且它们自己也没把这叫"接不动"** | 差异表里 `fall_thresholds` 的 `closable_by` 原文就是"把两个数改成活值即可……属发车决定" |
| mujoco-ac | 剩下 9 条 `motion_*` + A 族 `base_position` 卡在 Hydra，是另一批活 | **不成立 → 改判"还差一步"** | 见本节 5(a) |
| mujoco-ac | 本提交之前铸的 MuJoCo 收据带旧经济，但仓里没有 committed 产物钉死这个 sha | **成立** | 全仓 `git grep` `reward_contract_sha256`，committed 的只有 `configs/mujoco_c_lite_20260803/…receipt.v1.json` 一份，而它是 `c_lite` 那条另一套 reward（motion/balance 都固定为 0），且那个 sha 全仓**只在它自己那个文件里出现一次**，没有任何消费者 |

---

#### 8. 发车决策单（给 Franco）

##### A. 现在就能发的

| 发什么 | 占哪张卡 | 多久 | 备注 |
| --- | --- | --- | --- |
| **什么都不用重发。** 正在 GPU1 上跑的四格 `scale4096`（s20r1）**不受这两枚提交影响** | — | — | 本节 4 已证：argv 逐字节相同 |
| **把 frame-0 出生那条"零改动"支路真跑一次**：`materialize_a3_dynamic_ready_contract.py --physical-birth-composition-mode whole_body_safe_teacher_frame0_grounded` | **不占卡**（纯 CPU，numpy + mujoco，不 import Isaac） | 一次 materialize | 这是「十三」标价表里的 (v)，**至今没人跑过**。零 plant 改动、零增益改动，跑完就知道现役 `2.243 rad` 那个揭示阶跃还能压到多小 —— 这个数会直接改变下面 C.1 该怎么选。**唯一需要你点头的点：它会铸一份新的 `dynamic_ready` artifact** |

##### B. 还差一步的

| 差什么 | 谁做 | 占卡 | 多久 |
| --- | --- | --- | --- |
| **发一格 DR-L1**：① 给两个 materializer 加 `--dr-level`，产 DR-L1 lineage（kind 名字发射器已预留、会逐字核对） | subagent | **不占** | 30–60 min |
| ② DR-L1 的 reward recipe 与 dynamic-ready policy recipe 重新 materialize（按档内容寻址，DR-L0 的产物不能代签） | subagent | **要一张卡**（GPU2 空着，别碰 GPU0/GPU1） | 两族 20–40 min |
| **闭掉那 9 条 `motion_*` + A 族 `base_position` 的活值比对**：把 `resolve_task_profile()` 从测试模块提成共享件，写一条 `class weight × motion_scale` 的合成器 | subagent | **不占** | 2–4 h（含变异测试） |
| **重出 spec**：`04434abd` 把 spec/claim 从 v2 提到 v3、`dr_launch_level` 变必填。磁盘上已生成的 v2 spec 不能再重放 | subagent | **不占** | 分钟级，但**下次同步 pod 到新 HEAD 之后必须先做** |

##### C. 需要你拍板的

**C.1〔最大的一条〕出生改成 frame 0 = 要不要动腰的保持增益**

事实（`81379ea2` + 「十四」独立复算，两轮逐位吻合）：
接地后的 frame 0，`waist_pitch` 撑住要 `-49.155 N·m`，现役这套增益的**位置指令**最多发得出 `-21.704`；
把 5% 投影内沿和 2% 硬内沿全丢掉、指令顶死到机械限位也只有 `-26.014`。
**电机不是瓶颈**（限幅 118.2，只用 41.6%），卡的是 `kp × 指令还能走的行程`。

**你问的那个问题（"真机在策略接管期间会不会切增益"）的答案是：会，而且是一个真实的运行模式** ——
但这个"会"帮不上我们想要的那个忙：

| 厂商在哪切 | 什么时候 | 保持的是什么姿态 |
| --- | --- | --- |
| `pp_policy.hpp:1362` planner STATIC-stand 闩锁 | **策略正在跑**、两拍之间、`level==0` 且已回站位 | `q_des` 在 `0.8 s` 内 ramp 到 **ONNX 的 `default_q`**（`planner_static_gain_scale` 默认 `1.0` = 官方 `400/500/500` 逐字节） |
| `main.cpp:2738` `--auto-start` warmup | 开跑前 N 个 tick | 策略算出来的 `q_des`，用站立增益发，到点自己切回 `a3_kps` |
| `main.cpp:2989` `kPdStand` / `:3323` teleop 兜底 | 上电起身 / 数据源丢失 | `a3_default_angles` |

这三行我**自己打开厂商源码核过**（不是转抄「十三」）：
`agi/a3_deploy_example/.../a3_policy_parameters.hpp` 里 `a3_kps` 腰是 `85 / 50 / 50`、
`a3_pd_stand_kps` 腰是 `400 / 500 / 500`；
`.../a3_pingpong/pp_policy.hpp` 的 planner-static 分支原文就是
`if (cfg_.planner_static_gain_scale == 1.0) { cmd.kp = official_kp_sdk_; ... } // default: official gains VERBATIM`，
而它 ramp 过去的目标是 `nominal_q_sdk_`（注释写明 `== a3_default_angles`）。

**所以"切不了"是假的，"切了就能保持 frame 0"也是假的。** 厂商每一次切增益，保持的都是**它自己的零点姿态**，
而且代码注释写明静态站立**不会主动平衡**，所以闩锁挂了三道前置。用站立增益去保持一个动捕运动员的预备架势，
厂商从来没这么干过。

**真正的代价在第二层**：`action_scale = 0.25 × 力矩上限 / kp` 是厂商公式。
`kp 50 → 500`，`waist_pitch` 的动作尺度从 `0.59` 掉到 `0.059`。**切增益就是切动作语义**，
除非把 `action_scale` 钉死不动 —— 那就脱离厂商公式了。

五条路的价（「十三」实测，我核过口径，未重跑）：

| | 做法 | 代价 |
| --- | --- | --- |
| (i) | 等待期站立增益、揭示时切回 | 切换瞬间腰上凭空少 `27.451 N·m`，倒 `0.1 rad` 要 `114–144 ms`；指令侧同时跳 `-0.336 rad`；且动作语义随之变 |
| (ii) | 腿取 frame 0、腰/臂取"能撑住的最近姿态" | 数学有解、工程无解（支撑边投影直接失败 / 激活自碰撞）；负担来自髋（前折 `81°`），手臂只能调 `±4 N·m` |
| (iii) | 换全库唯一撑得住的那条 clip 当首发 | 它 frame 0 已用掉 `89.6%` 腰权限，整条 `70` 帧里只有 `12` 帧撑得住 |
| (iv) | **只把腰的 `kp` 提上去，`50 → 150`** | 本条 clip 从 `0/57` 变 **`57/57`**；全库 frame 0 从 `1/73` 到 `32/73`（再往上加没用，剩下 41 条是右腕站在包络外，另一种病）。代价：腰的动作尺度除以 3、`contract` 的 `kp` 不再逐位等于厂商 `a3_kps`、下游 SHA 全部重签。参考量级：`150` 是厂商站立那套 `500` 的 `30%` |
| (v) | **先跑上面 A 那条零改动支路** | 零 plant 改动、零增益改动、不占卡。先拿到数再决定要不要动 (i)/(iv) |

**建议的问法**：先做 A 的 (v)，拿到"揭示阶跃还能压多小"这个数，再在 (i) 与 (iv) 之间选。
(iv) 是唯一一条能让"出生 = frame 0"这句话在整条 clip 上都成立的路，但它同时改了动作尺度，
按 `EXP-V2-REWARD-FREEZE §0.13` 属于"动资产默认值"那一类，必须由你拍。

**C.2 `upright_exp` 从 `1.0` 同步到 `0.25`（已落地在 `9e69239c`）**
这是把 MuJoCo 镜像同步到你 08-04 在 Isaac 侧已经做过的定价决定上，**不是新定价**。
但它确实改了 MuJoCo-native 那条车道的奖励经济。要么确认，要么回滚。其余 13 条一个字没动。

**C.3 mjlab 那四条一行就能改、但会让新旧收据不可比的**
动作解码默认 `flat → vendor`（vendor 模式已实现且逐关节对过活值 31/31）、
摔倒阈值（Isaac `40° / 0.5 m` vs 本车道 `60° / 0.70 m`）、
`robot_hit_table` 装成硬终止（目前"测得到 + 会拒收"但不是护栏）、
复位随机化（本车道多了一份四格里没有的）。每条在差异表里都写了 `closable_by`。

**C.4 DR-L1 首档 `start_pose_ramp` 的终点取值**
现值 x `-1.0 m` / y `±1.2625 m` / yaw `±30°`，是该 action manifest `base_spawn_min/max` 的 `3.2–3.3` 倍，
而 manifest 顶层写的是 `mobility_mode: "no_move"` —— **这两者之间今天没有任何校验**。
另外 `ramp_steps 96000` 是挂钟驱动，而 §6.1 要求支撑集扩张走 competence 驱动 + 可逆回退。
两条都属于"改随机性幅度 / 改课程口径"，不由 subagent 拍。

**C.5 顺手记一条（不阻塞）**
`left/right_shoulder_roll` 的动作零点是 `±0.12 rad`，而执行 `q_des` 包络内沿是 `±0.1697` ——
**动作零点站在自己发得出去的范围之外**，`action = 0` 发下去会被投影一刀。今天不挡任何事，
但它和 C.1 是同一族问题（"出生姿态必须发得出去"）。

---

#### 9. 收据 / 本轮没做什么

* worktree：`/workspace/franco/audit_{head,base,mut}_20260808`，全部 `git worktree add --detach 04434abd`；
  变异跑完逐条还原，收工时 `git status --porcelain` 为空。
* 变异原始输出：`/workspace/franco/audit_out_20260808/mutations.json`、`mutations_mj.json`、
  `mut_batch2.jsonl`、`mut_mj.jsonl`；常量对拍：`consts_base.json` / `consts_head.json`。
* **未占任何 GPU**（三张卡全程没碰）、**未跑 Isaac 训练**、**未铸任何 artifact**、**未放宽任何门限**、
  **未改任何 reward 权重**。
* **未重跑**「十三」那五条路的实测数（我核的是口径与出处，数值沿用 `81379ea2` +「十四」两轮已逐位吻合的结果）。
* 别的 session 正在改的文件（`action_ball_curriculum.py`、`build_action_ball_manifest.py`、
  `launch_n1_reward_screen_diagnostic.py` 等）一行没碰；本节写进 exp 时也只入库了自己这一段。

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
| `PORTABLE-SYSTEM-CONTRACT` | `IN_PROGRESS / V8 SINGLE-SLOT RUNTIME CLOSED (DIAGNOSTIC)` | V8单值绑定actor/critic `215/231`、Reward28、PPO V6、四due/exhausted clock、Isaac schema8与Mu `10/5/6`；不加raw ball/aim/history/ID。exact Pod与fresh双端ACK已闭合；最终N73、incoming producer、两步delay history和export producer仍未闭合 |
| `MOTION-REFERENCE-OBSERVABILITY` | `IN_PROGRESS` | 不新增 motion-intent/ID；teacher trajectory 已表达动作。N1 学会后不等待 N2/N3 即进入逐件准入后的全库；只有全库失败时才用小动作集诊断共享容量/串扰。仅当出现相同当前 teacher state、不同必要未来的反例时，才加 short future-teacher preview |
| `CONTACT-GUIDANCE-ABC` | `IN_PROGRESS / B DEFERRED / A-C UNMEASURED` | 旧 `L194` A/B long 已停：每 update 是 `512 env x 24=12,288 env-step`；A/B 同时时片约 `3.126/2.983 s/update`、约 `3931/4119 env-step/s`，B 只快 `4.78%` 且 CI 跨零，不值得保留第三条 ABI。legacy profiler-on `4096x24 / 6.700 s` 是8倍 env-step/update，原始秒数不可混比。最终 `14,509/18,026` opportunities 都是0 capture；旧 `outcome_dense_only/000` 又没有 ball-state actor，不能冒充 C。fresh A211/C211 均已有独立 211/319 consumer；C211 的 runner-before-oracle live hook、32个 TASK_ACTIVE closed-attempt collector、selected-rubber H/C ledger、achieved-flight sidecar 与 actor/critic incoming-ball逐值校验已经实现并通过 host 回归，但 exact Pod oracle32 仍未执行。因此真 A/C 学习与速率均=`未测`。A 只对 distinct semantic Q 调一次 online solver并缓存；C 是 direct-ball、总 inverse call=0。C 的当前最小 reward 冻结为 nominal strike tick 拍心-球心距离与`vb_fired` analytic selected-rubber contact-gated一次落点，不再私自添加其它 desired-contact 或 dense outcome 项，也不冒充PhysX observed landing。 |
| `CANONICAL-REWARD-RECIPE` | `IN_PROGRESS / STATIC N1 ORDER PASS` | V2 已实改为非腕全身 mimic + 全相位低权 measured paddle + window 内高权 task master。当前 A211 fixed-center 将 `base_position 1.5→0`，保留 `racket_progress=10`，coarse/fine/precision 九项均为父配方`×1.15`；C211 proximity=`240`；A/C landing=`700` (`+8.4..14`)。Take061 task-valid、`gamma=.99`的静态账为 A `1.773<1.852≤3.009<3.332`，C `1.773<1.904<3.332`；ready/swing mimic 分账专项已过。C reward identity 为 v3，schema-3 training-contract 已与 runtime facts exact cross-check。关闭仍需 launcher/oracle/fixture 和 MuJoCo consumer 串行 repin、pre-long 实测 eligible income/advantage、`landing∧post-contact-fall` 专项和 physical outcome truth |
| `PPO-RUNTIME-RECEIPT` | `PARTIAL / V8 FIXED-LR RUNTIME CLOSED` | V7两端optimizer LR均卡`1e-5`且大分母零contact；V8绑定RSL3.1.2、PPO V6 fixed`1e-4`、`512/H48/U100000/MB1/save2000`、Reward28、215/231、fresh optimizer/WAL。exact Pod、双rate、fresh ACK与`model_0` LR=`1e-4`已闭合；snapshot仍`diagnostic_nonresumable`，formal resume/normalizer、完整clip/grad、100k completion与独立逐reward-group consumer未闭合。 |
| `RESET-TERMINATION-RESUME` | `IN_PROGRESS` | Isaac atomic reserve/commit 可复用；MuJoCo diagnostic lane 已实现 per-env done latch、terminated-row compact reset、pre-reset terminal observation 与 post-reset next observation、caller-owned ledger、per-env question lineage和可独立复算 receipt。关闭仍需 phase fidelity termination、follow-through/recovery RSI 与完整 mid-episode resume；当前只允许声称 reset-boundary resume |
| `BALL-FIRST-SCHEDULER` | `IN_PROGRESS / FIXED-CENTER READY, EXPANSION UNMEASURED` | formal A 使用 `online_solver + complete-semantic exact-answer cache`：sampler/curriculum/RNG 每次 reset 正常推进，只有 Q 字节语义全同才复用；cold Q/Q' 各真实解一次。formal C 使用 `direct_ball` 且从不反解。`immutable_tape` 只保留历史目标信息消融，不进入 A/C formal lineage；`banded_question_bank` 只是可选未来 producer 优化，不阻塞首个 expanding long。仍须冻结 generator、initial/max envelope、扩域/回退、heldout state，并补齐可逆重测、new-band配额、样本不足作废、global/arm attribution、hysteresis、uniform/center floor 与并行探臂前置。 |
| `ISAAC-FOUR-CELL-FIXED-QUESTION` | `A211/C211 CODE IMPLEMENTED / INTEGRATION + PRE-LONG BLOCKED` | 当前四格是 `A/C x {fixed-lr1e-4, adaptive-KL-initial-lr1e-3}`。两者分别用独立211/319 ABI/normalizer/checkpoint，共享 measured teacher/seed/old plant/safety/network/budget。physical reset 使用 tracked split-ready，WAIT 5--25 tick；reveal 同 tick teacher 切到 measured frame0并公开本族current-center receipt派生的启动钟（literal center当前预计A约`.692376 s`、C约`.86 s`），由 dense mimic 学 bridge；禁止共用历史`.712376 s`。direct frame0 birth 已实测 `0/73`，不再授权。当前还须把 A cache/C direct-ball、DR-L0 leaf 与 split-ready lineage 在同一 clean exact SHA 闭合，随后跑两族 oracle32 和四格真 4096x5；全局 barrier 重开并逐份复核 source/claim/model5/telemetry 前 long 全阻断。 |
| `DR-LEVEL-LAUNCH-ENTRANCE` | `ENTRANCE LANDED / DR-L1 LINEAGE PENDING` | **人话**：以前"这一跑用哪一档随机性"是两个 launcher 里各写死的一行常量，都指向 DR-L0，所以 DR-L1 那两片 profile（五条 plant 轴 + `start_pose_ramp`，六条随机性轴共用一扇门）**选不中**。2026-08-08 把它换成发射时的显式选择：`template --dr-level {dr_l0,dr_l1}`，默认 `dr_l0`。唯一权威是 `scripts/action_ball_211_dr_launch_levels.py`（两族共用、已进两边的 `RUNTIME_SOURCE_PATHS` 钉子表）。**四格不受影响**：DR-L0 的 profile / profile 路径 / lineage kind / 候选 manifest / finalizer 身份五项解析结果与硬钉常量时代**逐字节相同**，有专门的冻结字面量测试守着。**收据自陈的是取值不是档名**：`dr_launch_level_contract` 里逐轴写出 friction `(0.3,1.6)/(0.3,1.2)`、关节零点 `±0.01 rad`、`torso_link` CoM `x±0.025 / y,z±0.05`、link mass `(0.85,1.15)`、Kp `(0.8,1.2)` / Kd `(0.7,1.3)`、`start_pose_ramp`（`ramp_steps 96000`，x `[-1.0,0]`、y `±1.2625`、yaw `±30°`），这些值从 `training_contract` 的 payload 与 tracked 候选 config `action_ball_211_dr_l1_restored_plant_candidate.v1.json` **两个独立出处逐字段对拍**得到，权威模块里一个数字都没手抄。spec/claim 因此 v2→v3。**发一格 DR-L1 还需要**：(1) 给 DR-L1 各跑一次 `materialize_action_ball_{a211,c211}_lineage.py`，产物 kind 必须是 `action_ball_a211_split_ready_online_question_dr_l1_lineage_v1` / `action_ball_c211_direct_ball_split_ready_dr_l1_lineage_v1`；(2) reward recipe 与 dynamic-ready policy recipe 按档内容寻址，`materialize`/`recipe` 两个 zero-PPO 阶段各重跑一次；(3) DR-L1 **按定义进不了四格**（四格刻意只差 obs-noise 一根轴），要成组跑就要另开一份属于它自己的格局权威。在 (1) 之前 `--dr-level dr_l1` 会在 lineage 那一步 fail closed，并把这三条原样报出来 —— 这与"入口没接"是两种可分辨的失败 |
| `ISAAC-N1-LEARNABILITY-HANDOFF` | `BLOCKED` | 一条来自真人对拉录制的单拍 measured N1；依赖 canonical measured authority/portable contract/reward/scheduler，满足 §9.1 的定量真实 hit/legal return、逐分母、安全、resume/export/handoff，不要求 Isaac N73。额外 N1/N2/N3 仅为失败定位，不阻塞 handoff |
| `MUJOCO-SCENE-CONTACT-HARNESS` | `PARTIAL / SELECTED-RUBBER CONTACT RECEIPT CLOSED` | native ball/table/racket scene、strict contact pairs、portable/backend SHA closure、substep contact/recontact/outgoing latch 已实装。exact Pod `592835dc` 同题真实 rollout 得 generic edge=1/table=0/valid outgoing，sidecar 分类正号红面，tick/substep=1/3，切向距 `0.007168732 < 0.044263876 m`，invalid=[]；receipt-v2 已在 exact detached `95382a53` replay=`18 passed`，classification 与 backend seals 独立重算一致。Reward/PPO/incoming-question parity 仍未授权 |
| `MUJOCO-SINGLE-ENV-PLANT-ACTION` | `IN_PROGRESS / PORTABLE HOLD V2 PASS` | schema-3 31-D action、implicit total-PD、delay/reset/fixed-tape 和 native ball observation/contact receipt 已实装。action-specific hold v2 用 repo-relative logical path+SHA，consumer 拒绝旧 v1、absolute/traversal/repo-escape；host=`18 passed,6 skipped`、exact Pod 真 MuJoCo d0/d1/d2=`24 passed,0 skipped`。immutable authority probe 仍只有 table edge，没有 racket hit/reward/learnability授权 |
| `MUJOCO-VECENV-PPO-CHECKPOINT` | `PARTIAL / V8 512-H48 FRESH ACTIVE` | V8保持schema10 ACK、Reward28/V3，learner改为PPO V6 fixed LR；exact Full-A rate与fresh连续ACK已闭合。禁止resume；尚无100000 completion、selected contact/landing、mid-episode restore或formal checkpoint authority。 |
| `MUJOCO-RUN-CONFIG-DETERMINISM` | `PARTIAL / V8 EXACT SOURCE RUNNING` | one-shot launcher单值绑定clean source `0ad85ae1…`、PPO V6、GPU UUID/flock、run-owned cache、EPA48/RSL3/MJLab runtime stack及source-plant/runtime-attach；exact Pod rate与fresh ACK已闭合。paired-tape Tier-1、接触/飞行Tier-2统计与跨run复算仍未完成。 |
| `ISAAC-MUJOCO-CROSS-ENGINE-PARITY` | `PARTIAL / V8 SHARED LEARNER RUNTIME CLOSED` | 两端继续共用215/231、Reward28、H48、三段reference和同一课程；V8再共用PPO V6 fixed LR。V7约218万launch仍零contact且LR贴底，只作negative历史；V8 exact双runtime已闭合，但contact/landing、数值、physics与transfer parity均未测。 |
| `MUJOCO-CANONICAL-N1-AUTHORIZATION` | `BLOCKED` | 显式合取门：portable ABI ∧ admitted teacher ∧ pinned sim contact/physics profile ∧ full termination/reset ∧ reward/evaluator parity ∧ trainer/save/resume ∧ run determinism ∧ fixed-tape cross-engine parity。真实拍子质量/惯量可只阻塞 sim2real，但 formal sim 仍需具名接触 profile |
| `MUJOCO-N1-REPRODUCE` | `IN_PROGRESS / V8 NEGATIVE, V9 FINITE VALIDATING` | V8两端合计`1,089,548`次launch仍零selected contact，且已证混源/legacy override，不能再解释为启动期；V9 finite已修同源bootstrap和tick48曝光，但两端paddle误差尚未改善。关闭仍需fixed-tape first-divergence后，balance→mimic→hit→landing自然重叠的逐分母fresh未来证据。 |
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
   ActionBall base-safety：death post-dt=**`-0.2`**（weight `-10`；**2026-08-07 就地更正**，
   本句原写 `-6`，见 §5.6.17 矛盾 1），actual-q/qdes barrier manager weight=`-5`，
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

在此之前，当前总体状态只认页首`2026-08-26 current correction`与
[双后端TODO §0.5](../../operations/action_ball_dual_backend_longrun_todo_20260819.md#fullmdp-v6-todo-current)：
PPO V5、Reward28与Observation V3 `215/231`合同已冻结；V7 exact source、两端Pod rate与fresh启动
canary仍待闭合，mimic→contact、landing/recovery、formal checkpoint、physics/transfer parity与25000
completion也未测或未闭合。下文A211/C211、211/319-D、oracle32、四格与`0/73` mechanical audit是predecessor历史，不得覆盖
V3、H48、三段reference、`10/5/6` wire或`diagnostic_unauthorized=true`边界。
### 2026-08-18：WAIT RSL3 输入身份修正

薄launcher原先没有把one-shot运行器冻结的ready-pose传进production env；direct GPU测试与真实
`learn(1)`因此可能读取不同输入。现在production callpoint用`O_NOFOLLOW`单次读取absolute、regular、
单链接的`ACTIONBALL_READY_POSE`，校验fixed SHA与inode后让WAIT env直接解析同一份bytes；显式输入不再
fallback。此项只闭合输入身份，live RSL3仍未测。

### 2026-08-19：portable Full-A 首纵切片裁决

**问题。** WAIT `learn(1)`只证明229/399-D训练调用点；要判断是否可以把portable MuJoCo称作A，至少要
区分“真实plant事件已经发生”与“runner成功返回”。本轮只采用最窄因果切片：row-wise reveal、真实ball
launch state、20-substep plant、live contact、bounded terminal和selected reset；不为缺失语义造receipt。

**结果。** host生产调用点与反例为`27 passed,9 skipped`。runner只累计`env.step`返回的事件，固定发布
`task_lifecycle=full_a_slice_attempted`与`full_a_complete=false`。R03 strike fact、R06 landing outcome、
R07 recovery和Reward项0--13仍是`not_produced`；因此这一结果不能解释为portable Full-A，也不能与
native 114/114-D A1000的contact曲线直接比较。新增的GPU节点从真实racket geom和MuJoCo contact rows
独立证明ball-racket pair，再穿production step验证latch。第一次Pod节点把mesh geom frame原点误当成
mesh内部点，真实contact array按规格没有pair；改用production measured-racket site作为独立live blade点后，
clean Git `2c8ef444…`在GPU1得到`1 passed`。这证明selected-contact调用点，不生产landing/recovery。

**裁决。** 保留这一纵切片作为下一批producer的真实消费端，拒绝把PPO成功或手造extras当Full-A完成。
R03/R06/R07与Reward0--13接通前，不启动portable MuJoCo A长跑，也不以它满足“MuJoCo完整”。Isaac的
LM CUDA异常路径已在Jiayi Python3.11/Torch2.7-cu128上三参数`3 passed`；下一阻塞收敛为v11 compact-safety
真实前5次receipt和一次性4096 wrapper。旧失败run的PID已自然消失，没有stop动作。

### 2026-08-18：native A长跑预注册

目的不是用旧lane代签portable FullMDP，而是尽快取得一条真实MuJoCo学习曲线。采用既有court/ball/
contact/native Reward/PPO，新增的production变更仅是显式`--xml-path`和`--ready-pose`输入。首个长跑固定
seed0、vendor action scale、initial std `.02`、1000 update；与Isaac并行但用独立GPU/CPU核组和namespace。

中途在20/50/100/200/500/1000读取同一run：episode length、binary racket-ball contact、各Reward term、
`action_rate_l2`、nonfinite/capacity和wall time。旧长跑中`action_rate_l2`由约`-0.126`改善到`-0.022`，
并推动负/正Reward比跨过1，是本次重点复核的可迁移因果模式；不提前改权重，也不因早期跌倒/撞桌停止。
该run若成功，只回答native scene的Reward经济和可学习性，不回答229/399-D portable A/C parity。

canary已自然完成：`1024 × 2`、49,152 transitions、capacity `PASS_NO_OVERFLOW`、31动作、114-D
actor/critic observation，两个update的Reward/time均finite。min racket-ball distance约
`1.095 -> 0.511 m`，binary contact仍`0%`；不把两点曲线当学习结论。result SHA256=
`8847a1b5…e09c`。同配方A1000已从fresh namespace启动，CPU48--63，GPU2最多一个既有peer；
里程碑仍为20/50/100/200/500/1000。

update20/50完整JSONL显示：累计episode length约`145.45/145.56`；节点min racket-ball distance约
`0.454/0.448 m`；binary contact仍0；action-rate income约`-3.65e-5/-3.78e-5`；累计负/正Reward比
约`0.00270/0.00286`；capacity overflow与nonfinite均为0。与历史“负惩罚压垮正收入”不同，当前
更像contact geometry/可达性尚未学到。按预注册继续100/200，不在50步改权重或停止。

50--100窗口首次产生`6/8452=0.071%`binary racket-ball contact；100--200窗口为
`32/17025=0.188%`，约2.65倍。mean min distance仍约`0.461/0.462 m`，episode length约
`145.22/144.47`，所以只称有稀少接触率上升，不能称整体更接近球。action-rate income均值约
`-3.96e-5/-4.34e-5`、负/正Reward比约`0.00307/0.00326`，负惩罚不是当前主瓶颈。
capacity overflow/nonfinite均为0，继续500。

### 2026-08-19：Isaac长跑发射与portable MuJoCo并行裁决

本轮采用“一个长跑、里程碑只读”的实验结构：Isaac A直接`4096×25000`，update0--4是同一进程的
容量/finite/事务观察，不再用重复小N smoke累加退出码。发射件要求actual Kit trainer进程内绑定
Python3.11/Torch2.7/TensorDict0.10/RSL3.1.2 class source，随后才进入`train.py`；末端证据必须消费
25000组PENDING/EPOCH_ACK、25000份v11/post-safety ACK、Reward20 finite/conservation和非零D05分母。
普通tilt/table/fall只记telemetry，不作为提前停机理由。

MuJoCo侧不等待Isaac结束，但当前只承认`full_a_slice_attempted`：真实reveal/launch/generic racket
contact/terminal/reset已有live消费点，R03与Reward0--9已有host真实FK消费点；selected-rubber、R06/R07
与Reward10--13仍`not_produced`。因此下一批工作是在Isaac长跑期间先接共享73-action identity/mount sign，
再接selected-rubber和outcome/recovery；portable Full-A未闭合前不发其长跑，也不拿native A1000曲线做等价比较。
本节是采用/拒绝边界，不记录逐条shell命令。

first 4096 one-shot没有进入学习：commit `5ee1ffa6…`、wrapper `022c13f5…`的preexec与sealed runtime
identity通过，Hydra解析出4096，但身份代码在`AppLauncher`前先导入Torch/RSL；Kit记录startup后约
0.34秒segfault，`Learning iteration`、scene、PPO与WAL全缺席。这个反例把失败归因到启动顺序，不支持
“4096装不下”或“Reward有问题”。采用的窄修是pre-App验不可变archive/interpreter并拒绝Hydra预载
Torch/RSL/TensorDict；证明与五项attestation值只能交接一次，AppLauncher成功后的入口先消费、再拒绝值漂移和其期间预载runtime，然后
在同一Kit进程再导入和核class source、写v2 receipt，然后才进训练；successor仍直跑同一4096长预算。

second 4096 one-shot（commit `d341931c…`、wrapper `60cfdeed…`）也没有进入App后hook：preexec receipt
已落盘，v2 receipt仍空，Kit约2秒内segfault，scene/PPO/WAL为零。由于pre-App状态机已直接证明
Torch/RSL/TensorDict未预载，该因素不是唯一根因；保留下来的共同异常是两次都绕过常规direct-script入口，
采用`python.sh -P -S -B -c runpy(...)`。下一successor回到成功N2同类的direct script形状，删除`-S/-c/runpy`，
只保留`-P/-B`与production内同一pre/post-App attestation；仍直接使用fresh 4096长预算，不另跑小N。

第六个`4096×25000`尝试没有产生学习样本。commit=`00cc5425…`、wrapper=`edb7fec4…`的GPU preexec和
AppLauncher均通过，post-App在scene/PPO/WAL之前拒绝`post-AppLauncher dependency escaped the frozen
venv`。只读根因对照显示：AppLauncher选择Isaac Sim 5.1
`exts/omni.isaac.ml_archive/pip_prebundle/torch`，版本`2.7.0+cu128`且入口SHA与冻结venv Torch相同；
旧门错误地把“必须来自venv”当成Torch规格。采用的修复是Kit Torch exact entrypoint或venv二选一，且
live `torch.*`必须全在同一selected package root，且实际消费的parent attributes必须与`sys.modules`同一对象；TensorDict仍只能来自venv，RSL仍来自sealed archive。
由于runtime receipt、PPO和WAL均为0，本次不能解释4096容量或Reward。

第七个尝试（commit `73d607f0…`、wrapper `324d36ad…`）继续把首错定位到证据门本身：exact Torch
entrypoint已经通过，随后closure把`sys.modules['torch.ops']`里的动态`_Ops`对象当作file-backed module；
其`__file__`访问是operator dispatch，得到的run-root `_ops.py`不是provider。采用修复只检查真实
`types.ModuleType`，仍硬验`torch.optim/_C`路径与parent/sys.modules identity。此run同样零PPO/零WAL。

第八个尝试（commit `5db486cd…`、wrapper `a1d97aae…`）反证上述按类型修法：`torch.ops`是
`ModuleType`子类，仍触发同一动态`__file__`伪路径。最终采用不是继续枚举Torch内部对象，而是删除没人消费的
blanket closure，只验top-level与RSL真实消费的`torch.optim/_C`、parent和PPO wiring。该改变缩小错层gate，
不改变Torch/TensorDict/RSL版本或训练MDP；本次仍无PPO/WAL样本。

第九个尝试（commit `f919ff1d…`、wrapper `d307dc29…`）首次通过post-App真实runtime attestation，随后在
env cfg构造、scene之前自然RC1。首错不是资产内容：fresh run私有快照与共享源目录13个文件逐字相同，二者
`model.usd` SHA256均为`a3cd3829…8140`；拒绝来自consumer要求pathname恰等于共享目录。这个反例再次支持
HANDOFF §3的判据：实际物理引擎消费的是snapshot bytes，路径字符串并非模型规格。采用的修复仍由tracked
producer对实际selected root使用代码内固定pins重建URDF、STL、IsaacLab asset hash和derived USD语义；不采用
wrapper自产receipt或expected hash。live collider几何也改从同一已验root读取，避免“验共享源、用私有快照”
的错层。host回归`77 passed,37 skipped`，尚无4096 scene/PPO/WAL证据，因此不改变学习裁决。

successor `b64cb944…`给出了第一段可消费的4096学习前缀。它不是另一个5-update smoke，而是目标
25,000 update的同一进程；update0--8已有9组完整durable pair、884,736个finite Reward sample且守恒
violation为0，证明scene、runtime、compact safety、optimizer和WAL在真实规模共同工作。与此同时，
D05的12,288个due/selected全部落入10,836 not-ready defer或1,452 reject，ACCEPT、launch、contact、
outcome、recovery仍为0；8,192个episode暂全由base tilt结束。因此当前科学裁决只有两条：工程边界通过；
任务入口尚未ready。前9点不足以判断长期可学性，run继续，首次趋势判断仍等20/50/100/...，1000不停机。

### 2026-08-19：A100科学窗与下一代因果选择

本节只消费100个完整`PENDING/EPOCH_ACK` pair，不用console rolling均值代签。0--99共
`9,830,400` transitions，actual Reward全部finite、nonfinite=`0`、conservation violation=`0`；
max residual=`2.492405e-7`，低于窗口max tolerance=`1.233621e-5`。0--49→50--99窗口的每transition
Reward为`0.057989→0.058106`，episode return为`5.076→5.251`，episode length为`88.02→90.29`。
这是一点dense imitation改善，但幅度很小，不能称击球学习。

真正业务分母给出相反边界：累计`106,645`个episode全部由`base_fell_tilt`终止；D05
`110,664`次due/selected分成`97,328`次not-ready defer与`13,336`次reject，ACCEPT/CENSOR均0。
R03 first-valid、physical observed/contact、launch/playback、R06 settlement、R07 recovery、payment/retire
全部0。因而当前不是Reward0--13权重太小，而是这些项尚无eligible样本；下一代先修ready producer和
action identity，不能在分母为0时调稀疏Reward比例。现役run继续200/500/1000/.../25000，以后窗口若
产生非零业务分母再讨论Reward经济。

### 2026-08-19：portable R03与rough地形采用边界

portable MuJoCo当前把同一postphysics racket site的scene-local position、COM point velocity和+Y normal
写入R03 achieved row，并由engine-neutral Reward20 kernel计算项0--9；host combined为
`28 passed,9 GPU skipped`。这证明计算图能消费真实FK，尚不证明73条动作的face方向：当前deterministic
question没有Isaac action UID、family、mount sign或selected-rubber authority。generic ball-racket contact
因此保持generic，Reward10--13保持0；下一实现必须复用共享catalog identity与既有保守selected-rubber
classifier，而不是从generic contact或固定+Y猜拍面。R03 fresh GPU、R06/R07完成以前不发portable长跑。

rough地形不是逐格随机。producer继续使用固定seed的空间相关场、桌侧exact-flat和smooth transition；
确定性FK发现旧`0.20 m`出生平地小于双脚collision envelope。固定MJCF `70c4fd65…`与ready pose
`ab6b7e41…`的四个踝部mesh最大XY半径为`0.470508504 m`，所以采用`0.60 m` exact-flat core和
`0.80 m` blend，包含一个10 cm cell guard；host性质=`14 passed`。这是确定性几何纠错，不做A/B。
现役nominal run仍为plane；rough只在fresh namespace中按`±5/±10/±20 mm`独立阶段验证2-env foot/table
几何与4096吞吐，不把shared clone pattern说成per-env curriculum。

### 2026-08-19：A200窗口与portable Reward10纵切片

**A200方法。** 只消费active Isaac WAL最早200个完整、严格交替的`PENDING(v2)/EPOCH_ACK(v2)`
pair；忽略未完成尾片，不用console rolling平均代签。每个scope对counter/sum/sum_sq求和，对residual和
tolerance取max，gauge取窗口首尾。结果累计`19,660,800` transitions，actual Reward sample全部finite、
nonfinite=`0`、conservation violation=`0`，actual sum=`1,090,246.0043500704`，每transition=
`0.0554527793553706`。completed episode=`233,267`，mean length=`84.03598`、mean return=`4.65837`；
tilt=`232,058`、robot-table=`1,209`，其余termination为0。

D05 due/selected=`237,221`，construction/key admitted=`208,796`，这些全部not-ready defer；另
reject=`28,425`，ACCEPT/CENSOR=`0`。没有completed shot、R03、physical launch/contact、R06、R07、
payment或retire。Reward0--13的eligible/nonzero都为0，只有dense motion Reward14--19出值。
100--199窗口Reward/transition=`0.0528580`，episode mean length/return=`79.7261/4.23285`，均低于
0--99约`0.0580476/89.15/5.163`。因此采用“保持active长跑，平行修ready producer”，拒绝“停止到1000”、
“继续增加短smoke”与“零分母时调稀疏Reward”。

**动作身份纠正。** 冷catalog有73行，但current fresh cadence固定slot0，genesis与每次Device-R05也写
slot0，selected reset只回同一clip frame0；legacy balanced sampler不在fresh调用链。WAL进一步只出现
slot0、UID `6907688916670928`、forehand机会，unknown仅来自reject。因而本代的backhand denominator=
`0/未测`，不能拿bank的59/14 family分布冒充训练覆盖。

**portable实现。** 新portable catalog复用同一manifest与73个motion byte pin，冷算与Isaac同式的击球帧
site position/quaternion/angular velocity、中心差分site velocity、raw normal、reach和base quaternion，
并把slot0 UID/mount sign安装到MuJoCo per-env state。observed selected-rubber kernel只接受backend已观察的generic contact edge，
同substep用`R^T(ball-site)`分类selected/opposite/edge/between/invalid；strict tangential radius、red/black
outer plane和invalid rotation/sign都由shared racket geometry决定。MuJoCo只将selected分类作为一次
Physical event写Reward10，不把persistent contact latch反复付款。host组合=`38 passed,9 skipped`；
Isaac catalog/cadence/timing=`47 passed,24 skipped`。另用真实`MotionLoader + RacketTargetCommand`
生产冷builder对全73条motion逐列比对portable FK，所在文件=`12 passed,3 skipped`；这避免固定数值自证，
但仍不是GPU运行证据。

**裁决。** 这关闭了action0 identity和Reward10 host计算图，不关闭portable Full-A。当前临时midpoint
question、selected motion teacher、R06、R07和Reward11--13仍缺，fresh GPU contact/reset节点仍skip。
下一依赖顺序固定为`catalog -> action-conditioned question/teacher -> observed contact GPU -> R06/R07 ->
4096×25000`。直到这些真实调用点闭合，runner必须继续写`full_a_slice_attempted`与
`full_a_complete=false`。

**fresh GPU first attempt。** commit `e6cefaa0…`在Pod1空闲物理GPU1的一次性fresh namespace中运行
三个Full-A direct节点，首个N1节点`rc=1`后按`-x`停止，未重试。真实ball已经launch并移动；失败是测试把
任何姿态/桌面terminal都要求带Full-A flight outcome，而production对此类terminal正确保留
`outcome_code=NONE`。tests-only修复将该shot deadline钉在launch transition，避免无关terminal抢先，
同时保留真实physics、movement、outcome和selected reset断言；host=`11 passed,6 skipped`，fresh GPU复核待办。

紧接的`cdfc5ad1` fresh件因一次性脚本手抄错误full SHA而在checkout阶段`rc=128`；GPU/pytest均零调用，
namespace `_p88w9df`保留且不复用。按fail-stop本轮不再尝试；下次必须机器读取完整HEAD SHA生成执行件。

**2026-08-19 fresh GPU infrastructure stop。** 后续件已机器读取完整HEAD `0790504c…`，并在Pod1自然
空闲GPU1上消费fresh namespace `mujoco-fullmdp-gpu-gate.0790504c.Pod1GPU1.E6oHC4xX`；checkout与
transition-test SHA均匹配。首个真实错误是Pod不存在`/usr/bin/numactl`，wrapper line85自然`rc=127`。
因此Python、pytest、MuJoCo-Warp和GPU callpoint均为0，不能把它计为contact/reset RED或GREEN。result/log
保留且namespace不复用；本轮fail-stop。下一件在prewrite阶段必须确认`/usr/bin/taskset`存在并把GPU1绑到
独立CPU集合，避免多GPU训练默认挤在同一CPU/NUMA集合。
