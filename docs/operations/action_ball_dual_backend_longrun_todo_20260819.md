# ActionBall 双后端长跑：当前执行 TODO

> 状态：`ACTIVE / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-20
>
> `origin/main:docs/NOW.md` 仍是全项目唯一优先级权威。本页只记录当前分支这条
> FullMDP 单动作双后端路线的依赖、证据、阻塞和完成条件，不建立影子队列。

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

### MuJoCo portable Full-A：host长跑件已闭合，live focused到正确DEFER，25k尚未启动

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

当前唯一发车边界：

1. exact commit `4aadd698…`的两次Pod1 GPU2 fresh one-shot均已封存且不重试：首轮namespace
   `...071549cst`在`rsl3_source_gate`因direct system Python缺`tensordict`停止；r1 namespace
   `...072045cst-r1`修正依赖入口后到真实GPU focused `5 passed, 3 failed in 23.11s`，但三个旧
   downstream测试错误要求zero action在tick 2必然ACCEPT。两轮均`failed_no_retry/rc=99`、ACK=0，
   trainer未启动；详细身份与首错见[portable Full-A实验](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md)。
2. 语义裁决是不改production：tick 2只有due，readiness=false时正确结果是DEFER、zero-write、tick 3
   不补试。下一fresh commit把自然`ACCEPT XOR DEFER`留给独立GPU反例，并只在contact/outcome的
   downstream tests显式tests-only admission；该admission不得计入业务证据。
3. 下一one-shot必须使用新commit、新namespace和exact RSL-RL 3.1.2 overlay；focused gate通过后由
   同一执行件直接进入`4096 × 25000`，不再添加短跑或zero-policy表现门；任一失败仍封存且不重试。
4. 真正的`4096 env × 25000 PPO update`尚未启动，Pod wall time、吞吐与学习趋势均为**未测**。
5. 本代只授权A/slot 0；C/backhand未授权，backhand denominator必须为0。

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
| 2 | thin ledger与独立consumer | **CLOSED-host**：runner/ledger/consumer `96 passed, 1 skipped`，含production writer→consumer prefix | Pod写出一行真实ACK、snapshot并由独立consumer读取 |
| 3 | exact Pod focused gate | **live attempted / HOLD**：两条fresh namespace均在trainer前封存；r1到真实GPU `5/8`，失败来自stale unconditional-ACCEPT tests，不是production cadence | fresh commit/new namespace；自然verdict反例与tests-only downstream admission分离；同一one-shot gate通过后直接进入25k |
| 4 | portable A长跑 | **未启动** | 同一进程`4096 × 25000`及exact 25000-row终端消费 |
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

普通tilt/table/fall、早期return下降或ACCEPT仍为0都只是telemetry，不自动触发stop。当前MuJoCo仍有
Pod真实调用点与live 25k证据缺口，Isaac zero-flight candidate也只有host语义证据、没有matched Pod wall；
所以它们都尚未严格支配active Isaac，**现在不停active Isaac**。这不等于继续暂停MuJoCo：exact Pod focused
gate一旦自然通过，就直接在fresh namespace发同一进程25k，不再增加zero-policy表现门或1000-update中转run。

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
