# ActionBall 双后端长跑：当前执行 TODO

> 状态：`ACTIVE-successor-construction / no-live-run / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-21
>
> `origin/main:docs/NOW.md` 仍是全项目唯一优先级权威。本页只记录当前分支这条
> FullMDP 单动作双后端路线的依赖、证据、阻塞和完成条件，不建立影子队列。

## 0. 2026-08-21 当前执行面（supersede 下文旧 active-run 叙述）

三条 `4096` 长跑都已经停止；当前三张 Pod1 GPU 没有本项目 compute process。两条 Isaac run 分别止于
ACK `4603/3467`，steady wall 约 `21.95/22.01 s/update`，退出原因未判定。MuJoCo r3 止于 ACK
`10249`，终段 last-100 wall mean/median=`4.890/4.886 s/update`；它因真实 MuJoCo-Warp EPA horizon
overflow fail-stop，同时旧 229-D actor observation 存在 IDLE clock 随全局 step 无界漂移，所以不得
resume，也不得把累计 `ACCEPT=0`解释成课程失败。下文把这些进程写成 active 的段落只保留历史，不能再作
发车依据。

Franco 2026-08-21 明确：此前“约 6 秒”是要求大幅缩短迭代时间的量级信号，不是把 rollout 固定为
24 steps 的绝对约束。下一代允许为更长 credit assignment 采用 48-step rollout，但仍必须大砍
environment collection 与 owner/transaction 固定税，并同时报告 update latency 与 transitions/s；不能用
batch 翻倍掩盖吞吐没有改善。

### 0.1 本代采用、延后、拒绝

| 项目 | 裁决 | 原因与边界 |
| --- | --- | --- |
| invalid-task epoch clock mask | **采用** | `task_valid=false`时五个clock严格为0；旧run的全局时间泄漏使MDP非平稳。Isaac/MuJoCo必须同源反例闭合，旧snapshot不resume。 |
| R07付款阶段 | **采用** | 只在`OUTCOME_SETTLED && deadline-relative age in [10,77]`付款；readiness仍每tick计算，REVEAL/LAUNCH/RETIRED不付。 |
| semantic observation V2 | **采用 A=`203/219`；C另立`202/218`合同** | 既有194/A211已经包含table-relative root pose与COM velocity；229/399是direct-lean diagnostic迁移时的遗漏，不是此前尽调否决。A actor只保留183-D robot/teacher prefix和20-D signed task residual；critic只加16-D未来因果物理状态。删除raw task45、owner fact blob、fault/age、reward due/paid等控制面，不做机械`237/407`扩维，也不宣称单帧完全Markov。 |
| FullMDP PPO recipe V2 | **采用** | `H=48, lambda=.98, epochs=5, minibatches=8, max_iterations=12500, save_interval=500`；总transitions、minibatch size、总optimizer steps与snapshot的env-step cadence保持旧`24x25000/MB4/save1000`量级。它是学习算法取舍，不是性能修复。 |
| one-pose reset/D05/R07 | **拒绝** | stable birth、action-specific stroke entry、post-shot recovery服务不同意图；强行相等会绕过balance->mimic->entry课程。 |
| Reward0--13调权 | **拒绝当前改动** | eligible denominator为0时调权没有因果作用；先修observation/clock/phase与学习配方。 |
| balance safe-set debt | **延后** | 仅在fixed-clock/less-aliased successor仍显示robot-centric奖励缺口时，才考虑一个shared pure-tensor barrier；不新增owner/receipt/gate。 |
| potential progress reward | **拒绝当前改动** | terminal/reset/timeout/phase边界未闭合，可能产生fall/reset正奖励。 |
| MuJoCo EPA horizon `24->48` | **采用为依赖候选** | 保留overflow fail-stop；先用同一稀有pair证明24确定失败、48 finite且CPU oracle/固定tape一致，再钉fork/version/SHA。不是调`njmax/nconmax`。 |

### 0.2 唯一依赖顺序

只有前一项闭合后才能开始后一项；每项独立提交、可单独回退。禁止把全部变化揉成一条无法归因的长跑。

1. **基线与TODO冻结**：从 stable `ee6571ba…`建立clean successor；整合已验证的clock与R07修复。
2. **Observation V2**：尽调已确认旧194/A211有root/table pose与COM velocity，direct-lean 229迁移遗漏了
   floating-base state，却机械拼入raw task45和170-D owner/reward账本。A冻结为actor203/critic219：common183
   包含table-relative root XYZ、continuous heading2、heading-frame COM velocity、proprioception、teacher与anchor
   residual；A tail20只给racket position/velocity/normal error、base-goal error、三个per-env倒计时、phase5与
   task-valid。critic suffix16只给episode余时、live ball9、selected contact/net history、foot support2和cadence
   dwell。实现为一个纯tensor semantic pack供Isaac/MuJoCo复用；用独立native-state正负扰动验证frame/单位/
   COM速度/alias，不能用shared layout自己生成expected再自证。C将来使用独立202/218合同，不补零凑宽度。
3. **PPO recipe V2**：只改FullMDP typed recipe，不改共享`ppo.yaml`；launcher receipt记录effective
   `H/lambda/epochs/minibatches/budget/save`，但recipe hash只证明provenance，不证明学习更好。
4. **MuJoCo capacity**：在project-owned pinned wheel中修EPA horizon，跑24-fail/48-pass deterministic fixture与
   exact GPU focused；不删除容量gate，不从r3恢复。
5. **Phase-B结构删除**：物理删除zero-callpoint formal owner、reveal adapter、旧Reward/Physical/R06 exact-pin
   family及专属测试。不得为了旧壳添加compatibility adapter。
6. **single-owner hot path**：一个device-resident mutable `ActionBallState`拥有phase/generation/shot/contact/
   outcome/fault；K-row候选只构造一次、一次sparse commit，只有真实transition写compact event delta。
   zero-live-flight成对跳过Physical/scene/R06/Epoch空事务；PPO boundary再统一汇总。
7. **性能与语义验收**：clean source做fixed-tape RNG/highwater/reason/done/reset/Reward20/237/407 parity；
   同卡、profiler-off、matched zero/mixed/active strata A/B。总update与transitions/s都必须显著改善；若最大span
   转移则重新profile，只处理新的首墙，不继续堆小clone补丁。
8. **fresh运行**：先运行clock-fixed、EPA-fixed的portable MuJoCo V2，随后运行通过matched性能验收的Isaac V2；
   都使用fresh commit/namespace，不插zero-policy表现门，不要求早期`ACCEPT>0`。里程碑观察balance->mimic->
   entry->strike->landing的分母和readiness margins，只有环境不可学或证据不可信才停止。

### 0.3 HANDOFF约束：什么是真安全，什么必须删

保留的边界只有跨真实权威的事实：nonfinite/overflow、真实contact与joint/table limit、reset generation、
source/asset provenance、optimizer后durable ACK及失败后的sticky poison。以下不得继续称安全：同一writer的
digest/receipt互证、无事件也写journal、每substep重复验证已在construction固定的class/bound method、没人消费的
counter、用随机rollout证明确定性几何，以及`ACCEPT>0`/zero-policy不跌倒这类“学会后才允许开始学”的门。
删除门时同批保留真正需要的人类telemetry和正/负反例；不能静默绕过真实物理或证据边界。

### 0.4 当前代码状态

- stable已有：discarded Reward record clone删除、invalid Isaac clock mask、owner deep scan移到cold boundary、
  direct-lean Phase-A、partial-construction simulator cleanup；这些主要降低固定税和维护面，尚无matched Pod wall。
- MuJoCo clock修复与R07 phase-window修复已作为独立提交整合到本clean successor；前者窄反例
  `2 passed`，后者当前独立focused=`138 passed,6 skipped`，完整既有190-test口径待Pod/依赖齐备后复跑。
- observation三份独立审计已闭合：旧229能构造root translation/yaw/COM velocity三类同obs异动作alias；
  这些量在旧194/A211已存在，HITTER/SMASH/BeyondMimic也分别用相对base/anchor、history或base velocity解决
  同类负担。A V2采用203/219并删除raw task/ledger冗余；OptiTrack marker->root、table extrinsic与causal
  marker->COM velocity可生产，但部署builder尚未接通，故当前仍只授权simulation diagnostic。
- D05 compact WIP虽然production净删2167行且`517 passed,40 skipped`，仍保留full-N publication/journal，
  并使dormant formal/reveal壳construction-broken；因此明确`HOLD / uncommitted`，不能直接作为successor。
- `origin/main:docs/NOW.md`仍是项目优先级唯一权威；本节只是本功能分支的执行依赖与验收清单。

## 1. 训练目标与口径

- 长跑目标是**同一进程 `4096 env × 25000 PPO update`**；`1000 update`只是早期趋势里程碑，
  不是终点，也不为它重启或停止。
- 当前代严格使用 `action_slot=0`。manifest与motion bank仍冷加载并校验73行，但fresh Isaac cadence、
  genesis、Device-R05和selected reset都保持slot 0；这不是73动作训练。
- A完成工程和科学判读后再跑C。portable MuJoCo只有消费同一MDP语义、真实事件和同一action identity后，
  才能称Full-A/Full-C。
- 全部运行都是[`diagnostic_unauthorized`](../DEFINITIONS.md)：不授权promotion、export、deployment、
  真机或物理安全结论。

## 2. 不做什么

- 不热补active run、不signal/kill、不复用namespace、不用新磁盘代码解释旧进程。
- 不再累计小N或短update smoke；最小测试只用于能区分实现的确定性反例。
- 不用fixture/self-SHA/同writer sidecar/无人消费的counter代签真实调用点。
- 不把generic racket contact叫selected-rubber，不把距离或sweep推断叫observed contact。
- 不在eligible denominator为0时调Reward0--13，也不把dense imitation均值称击球学习。
- 不把native 114/114-D MuJoCo吞吐或WAIT `learn(1)`冒充229/399-D portable FullMDP。

## 3. 当前运行真相

### Isaac A：两条immutable run只读推进

- 目标：`4096 × 25000`，fresh immutable namespace，1000不停机。
- GPU1旧run commit=`e8eef4fb…`，2026-08-20只读前缀为完整ACK `0..1186`：
  `116,686,848` transitions；Reward sample
  全部finite，nonfinite=`0`，conservation violation=`0`。
- completed episode=`1,011,944`，mean length=`78.460`、mean return=`3.952`；termination bit累计
  tilt=`1,010,092`、base-too-low=`25,895`、robot-table=`1,209`，同一episode可有多个reason bit。
- D05 due/selected=`1,015,878`；not-ready defer=`895,105`，reject=`120,773`，ACCEPT/CENSOR=`0`。
- R03、launch、selected contact、R06、R07、payment和retire全部`0`。Reward0--13没有eligible样本，
  只有dense motion Reward14--19在工作。
- 分窗趋势并未改善：0--199的Reward/transition=`0.055453`、episode length/return=`84.036/4.658`；
  200--499为`0.052140/90.449/4.719`；500--772降到`0.045175/67.829/3.066`。当前科学结论是
  “工程长跑成立，business producer仍未ready，dense imitation后段变差”，不是“已经学会”或
  “Reward0--13权重错了”，因为这些项的eligible denominator仍为0。
- WAL action identity只出现slot 0、UID `6907688916670928`、forehand；unknown仅来自reject，
  不能拿73行冷bank宣称训练覆盖73动作或backhand。
- update `0--843`的console分解为collection均值约`19.88/19.69/20.40/21.05/21.73/21.55 s`
  （窗口`0--4/5--49/50--199/200--499/500--799/800--843`），learning始终约`1.5--1.6 s`；
  当前约93%的iteration墙钟在collection。Pod采样显示trainer约`110% CPU`，其中主线程约`97.5%`，
  GPU0约`19%`，所以不是PPO或GPU饱和。active affinity=`32--47`，Yikang进程仍允许`0--127`，配置上
  确有重叠；但两秒样本中Yikang在`32--47`只消耗约`8%` CPU、全机node2忙约`4.3%`，不足解释相对
  `6.7 s/update`旧diagnostic校准的三倍差。旧数来自legacy diagnostic hot path；当前高reset FullMDP
  collection与正式transaction/owner路径不是同一工作量。CPU不要求互斥，只要求没有持续争抢或浪费；
  性能主线已在当前fresh `4096×25000`同一进程前5 update完成bounded profile并自动关闭。结果把首墙钉在
  96次/update的`post_physics_publish`（约7--8秒inclusive），不是CPU互斥、sim step或owner反patch扫描。候选已把D05的
  N平方唯一性改为O(N)，并只对construction rows执行solver/exact/Physical；它的唯一动态compact同步
  是否值得保留由同一successor实测决定。随后再批量化empty-flight/reset/owner gate，不热改active进程。详细量级和验收见
  [热路径实验记录](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md)。
- GPU0 successor commit=`ddb1e7c4…`使用同样`4096×25000`，只删除一次重复scene state read与无所有权clone；
  2026-08-20已自然运行到至少iteration 27，GPU约`6.3 GiB`。前5个profile row的collection均值
  `14.987 s`、postphysics均值`7.522 s`，故该cut没有消除首墙，尚不能称严格优于GPU1旧run。两条run均不
  热改、不复用namespace；在matched profile-off与业务证据形成前不停止任一条。下一工程candidate只针对
  zero-live-flight时成对跳过arm/capture/R06/retire/park，不能恢复已被反例否定的单边postphysics早退。

### MuJoCo portable Full-A：`4096×24×25000` active，已过update 5，尚未完成

当前host已闭合的代码合同：

- actor/critic observation为229/399维，action为31维；trainer必须隔离使用upstream RSL-RL 3.1.2；
- production engineering surface已覆盖R03、R06、R07、Reward0--20与exact phase
  `0/2/5/6/8`，fresh环境仍绑定slot 0及其UID/side；
- true Gym reset使用runtime default joints、configured default root加env origin、零速度和零action history；
  `take061/q_ready`不是birth authority。raw action不再做错误的`[-4,4]`裁剪，runtime/schema-2顺序与
  default-offset affine已闭合；MuJoCo executable q-des相对Isaac soft-inset/brake仍明确
  `DIVERGENT_DECLARED`，所以本run无transfer或matched authority；
- 30 s / 1500-tick cadence的due opportunity固定为`2 + 293k`（tick `2..1467`）；每个due按live
  readiness产生`ACCEPT/DEFER`，DEFER是zero-write且不在下一tick补试；
- HOLD joint teacher保持default/zero velocity，body/R07 target使用measured frame0。natural shot close
  发布`shot_retired`并停在phase8，保留robot/action/episode/generation；只有Gym done发布
  `selected_reset`并使generation恰增1；
- R07确定性host lifecycle反例必须走完age `10..77`、expected/eligible=`68/68`且没有sticky fault；
  不要求未训练zero-action policy在真实rollout中活过该窗口，其table/fall/contact只是telemetry；
- exact table keepout termination仍保留；曾加入、会误杀合法recovery的额外keepout witness已拒绝并回退；
- 26-event thin ledger在optimizer前只prepare；第26个`completed_action_epoch`由同一env行完整闭合
  launch/selected contact/fault-free physically-valid R03/fault-free eligible+source-valid R06/
  exact 68-cell R07/natural RETIRE后原子发布，跨env边际不得代签。optimizer成功后才写snapshot与ACK。独立consumer将
  engineering与business完成分开。host runner/ledger/consumer=`96 passed, 1 skipped`；
  env/action/outcome=`59 passed, 7 skipped`；alignment=`14 passed, 21 deselected`。

当前live执行边界：

1. r0/r1两条Pod1 GPU2 fresh one-shot均已封存且不重试：它们分别暴露direct system Python缺
   `tensordict`和旧GPU tests误把合法DEFER写成必然ACCEPT；详细身份与首错见
   [portable Full-A实验](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md)。
2. fresh r2 exact commit `9e7c1c614b1e22eeec4de243f55d58293da155ce`、namespace
   `mujoco-fullmdp-a4096-u25000-9e7c1c61-20260820t073755cst-r2`、wrapper SHA-256
   `36cc7a6166e1249061a45f7a3f7f1145a014a2b5a1f6f8417b84e0f58fefce5b`通过真实GPU focused
   `8 passed in 23.00s`。同一执行件进入实际4096-env trainer且首个optimizer update返回；随后stock
   save写出7,882,391-byte `model_0.pt`，却因disable-logs路径缺`runner.logger_type`而
   `AttributeError/rc=99`。evidence 0 bytes、ACK=0；未ACK文件不算snapshot，namespace封存不可重用。
3. fresh r3 exact source `dc62684c41e70e40dedaf191a32921b6cd98b344`、namespace
   `mujoco-fullmdp-a4096-u25000-dc62684c-20260820t074950cst-r3`、wrapper SHA-256
   `0f5adc6024f01ffee7e761ab7b620d70855e541dbea298216ab9093e30695fd6`；worker/trainer
   PID=`864055/865285`。真实GPU focused=`8/8`，单一trainer进程的`4096×24×25000`已经active；不得
   signal、热补、复用namespace或另发短跑。
4. 当前durable ACK=`0..7`，`update_1`/`update_5`只读consumer均`passed`。已ACK `model_0.pt`
   为7,882,391 bytes、SHA-256
   `06883851e67ccaaa921cfeeb8bf5c983ee6b3443d67465d8cde1d08ed63f528f`；它仍是
   `diagnostic_unauthorized/checkpoint_authority=false/resume_authority=false`。前8个pre-ACK core
   iteration为`4.889..5.640 s`、median约`5.025 s`；进程alive且result仍0 bytes，25k尚未完成。
5. Reward20/actual每update各98,304行finite，conservation fault=0且policy std finite。update0
   due/defer/ACCEPT=`4096/4096/0`；到update5累计`8192/8192/0`，update4 exact-table/Gym reset=4,096。
   这些是未训练policy的行为telemetry，不阻断engineering run；当前
   `engineering_run_complete/business_chain_complete/full_a_complete=false`，final仍`未测`。
6. 本代只授权A/slot 0；C/backhand未授权，backhand denominator必须为0。

#### 25k runner与consumer合同

- runner固定使用[`--full-a`、`--evidence-jsonl`、`--snapshot-dir`、`--completion-json`、
  `--source-commit`、`--run-namespace`及`--save-interval=1000`](../DEFINITIONS.md#mujoco-fullmdp-longrun-flags)：
  分别表示启用portable 26-event engineering surface，指定逐update证据流、不可覆盖snapshot目录、最终completion seal、
  exact source/run身份，以及每1000 update保存一次诊断快照；
- run形状固定为`4096 env`、每update `24 steps`、`25000 updates`，不得把1000当终点；
- JSONL是唯一逐update证据源。每个update先验证finite/counter/conservation，再调用optimizer；只有optimizer
  成功返回后才能ACK。ledger每update至多做一次必要的host transfer，不复制整套env tensor；
- snapshots仅允许update `0`、`1000..24000`每1000一次及final `24999`，共26个；它们都是
  `diagnostic_nonresumable snapshot`，不授权resume/promotion/deployment；
- 独立consumer必须在进程结束后读到exact 25000行、连续update `0..24999`、无重复/缺口，并离线计算
  per-action/per-family/per-side分母与里程碑。零分母rate写`null/未测`，不得写0%；
- `engineering_run_complete`要求exact 25k、25000 ACK与26份snapshot；
  `business_chain_complete`独立表示slot0必需业务producer非零闭合；因73动作、双侧与
  科学窗口报告未闭合，本代`full_a_complete`固定为`false`；
- update `0..4`只验容量、finite与合同，随后同一进程继续到25000；科学里程碑只读，不触发重启；
- table、fall、contact与zero-action观测都是telemetry；不得重新变成R07 scope或发车阻塞；
- 全部产物保持[`diagnostic_unauthorized`](../DEFINITIONS.md)，不构成Full-A科学成功或物理安全授权。

## 4. 下一代按依赖推进

| 顺序 | 唯一改动面 | 当前证据 | 下一验收 |
| --- | --- | --- | --- |
| 1 | raw-action ABI（原始动作接口约定） | **CLOSED-host**：无`[-4,4]` raw clip，nonfinite fallback与joint envelope反例通过 | fresh Pod真实调用点保持同一动作语义 |
| 2 | thin ledger与独立consumer | **CLOSED-live-prefix**：runner/ledger/consumer host `96 passed, 1 skipped`；r3 durable ACK=`0..7`，update 1/5 consumer均通过，`model_0.pt`已ACK | 守护同一run的连续ACK、后续25份snapshot和terminal consumer |
| 3 | exact Pod focused gate | **CLOSED-live**：r3真实GPU `8/8`，自然verdict与tests-only downstream admission已分离 | active run保持同一语义，不另发gate/smoke |
| 4 | portable A长跑 | **ACTIVE / update 5 passed / incomplete**：单一`4096×24×25000` trainer alive，前8 ACK durable；business/full-A仍false | 同一进程到ACK 24999、26份已ACK snapshot及exact 25000-row终端消费 |
| 5 | 终跑删除与2.0瘦身 | **待长跑后** | production callpoint census证明无人消费后再删legacy runner、重复validator/receipt与无restore consumer的carry graph |

catalog、question/teacher、observed contact、R06、R07和Reward0--20已经属于当前portable surface，
不再作为重复的依赖阻塞；历史失败只保存在
[G06](../gates/G06_isaac_to_mujoco.md)，本页不维护另一份影子事故队列。
live 25k被独立consumer完整验收后只允许写`engineering_run_complete=true`；只有必需业务producer
也非零闭合时，才允许写slot0 `business_chain_complete=true`。本代`full_a_complete`必须保持`false`。

## 5. 地形

- 现役nominal Isaac和参考tracking任务都是plane；A200失败不能归因rough terrain。
- rough producer已改为空间相关场，并有spawn exact-flat core、smooth apron和table-side exact-flat；
  不是每10 cm独立白噪声，也不逐step重采样。
- 地形只在fresh独立阶段启用：`plane -> ±5 mm -> ±10 mm -> ±20 mm`。先过2-env foot/table几何和
  4096吞吐，再观察站立/移动/恢复；不热改active scene，不把shared static mesh说成per-env curriculum。

## 6. 停旧run与切换successor

只有以下条件之一成立才停止旧run：

1. 进程已失败且证据frontier不再推进；
2. 发现使任务结构上不可学或使证据不可信的确定性错误；
3. fresh、immutable、同MDP且已通过真实调用点的successor已经就绪，旧run被它严格支配。

普通tilt/table/fall、早期return下降或ACCEPT仍为0都只是telemetry，不自动触发stop。MuJoCo r3虽已
通过真实调用点并active，但q-des仍`DIVERGENT_DECLARED`且没有matched Pod authority，尚未严格支配
active Isaac，**现在不停active Isaac**。同一MuJoCo r3继续自然推进25k；不因zero-policy表现另发gate、
1000-update中转run或restart。

## 7. 里程碑与完成条件

同一run只读：`20/50/100/200/500/1000/2500/5000/10000/25000`。每次至少报告：

- exact WAL frontier、transition数、Reward finite/conservation；
- D05 due/selected/admitted/ACCEPT/CENSOR/DEFER/REJECT；
- action UID/family/side denominator，unknown单列；
- R03、generic/selected contact、R06、R07、payment/retire；
- episode reason、length、return；零分母写`未测`，不写0%成功率。

本页只有在Isaac A/C与portable MuJoCo A/C都完成各自fresh `4096 × 25000`、真实业务分母和
terminal consumer闭合，并完成删除清单后才可标`completed`。checkpoint只叫
`diagnostic_nonresumable snapshot`；没有完整plant/manager/trainer/RNG restore前，
`checkpoint_authority=false`、`resume_authority=false`。
