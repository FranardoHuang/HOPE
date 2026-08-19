# ActionBall 双后端长跑：当前执行 TODO

> 状态：`ACTIVE / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-19
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

### MuJoCo portable Full-A：尚未允许长跑

已闭合的真实路径：

- 229/399 observation、31 action、upstream RSL-RL 3.1.2薄调用点；
- WAIT reset/step、row-wise reveal、真实ball qpos/qvel launch、20 physics substep、bounded terminal、
  selected row reset；
- 真实postphysics racket FK进入R03 achieved fact；
- engine-neutral 73行catalog冷校验，并将fresh env绑定同一slot 0/UID/mount sign；
- 只在MuJoCo实际generic racket-contact edge出现的同一physics substep读取ball/site/rotation，区分
  selected、opposite、edge/rim、between-planes和invalid；
- selected-rubber只发布一个control-step事件脉冲并驱动Reward10，不作shot终身latch。
- slot0 manifest center、live base yaw与shared Physical reverse-integration生成integer-tick question/launch；
  旧midpoint/linear/gravity shortcut已删除；
- sealed take058 measured teacher进入同一actor/critic/Reward20 seam：reveal原子切frame0，prepare velocity为0，
  measured rounded clock随后推进；take061 physical-ready reset不被改写。

仍是发车阻塞：

1. take061 physical-ready到take058 frame0仍缺certified production hold-qdes/bridge或逆动力学消费链；纯
   diagnostic recurrence helper无production consumer，不能代签3.2918 rad physical command gap；
2. R06 legal landing/outcome、R07 recovery和Reward11--13已完成host纵切片，但真实MuJoCo-Warp GPU尚未验收；
3. per-action/per-family/per-side分母和生命周期证据未闭合；本代backhand denominator应明确为0；
4. 新catalog、question/teacher、selected-rubber与R06/R07/Reward10--13新增部分仍需同一fresh GPU纵切片。host当前已锁定：
   no-contact与selected-contact不同settlement clock、R07以deadline而非早落点为年龄原点、10N双脚support、0.9 soft joint limit、
   invalid evidence fail-closed、R06/Physical critic retention与event payment分离、shot完成是非终止selected reset而非Gym full reset；
   R07另以row-wise账本硬验age `10..77`连续、expected/eligible=`68/68`且全程无sticky fault；跳格或NaN后恢复
   都fail closed且不再支付Reward13。四份direct suite当前=`49 passed, 8 skipped`。2026-08-19空闲GPU1 fresh件使用完整机器读取
   HEAD与新namespace，但在Python/GPU零调用前因Pod没有`/usr/bin/numactl`自然`rc=127`；真实接触与reset门
   仍未跑。该namespace封存、不重试。successor `5d0044b8…`改用Pod已有`taskset`并在GPU1/CPU`0--15`
   真实执行：N2 timeout-reset peer exact与N1 reveal/launch/settlement两节点通过，selected-rubber节点在
   production callpoint前的测试构造失败，因为球心被放在恰好一个球半径的零穿入相切位置，MuJoCo合法地
   没有生成resolved pair；结果为`1 failed, 2 passed`、`rc=1`，第四节点未执行。namespace
   `mujoco-fullmdp-gpu-gate.5d0044b8.Pod1GPU1Taskset.TXXdHBJ7`封存且GPU/锁自然释放。tests-only successor
   改为selected侧1 mm明确穿入。fresh `a6f83b24…`自然复核已证明live pair、eligible、generic contact、
   selected classification与Reward10=`0.02`均由production真实产生；随后tests-only断言把CUDA tensor直接交给
   `pytest.approx`，因其试图`.numpy()`而TypeError，结果仍为`1 failed,2 passed`，第四节点未执行。
   该namespace同样封存且GPU/锁自然释放。断言已改为device-native `torch.testing.assert_close`；fresh完整
   四节点未过前仍不关闭contact/reset runtime缺口。2026-08-20 commit `ed11390e…`又在GPU0 slot-2/CPU
   `16--31`真实运行：WAIT N1/N2两节点通过，首Full-A节点在构造期因base reset先于Full-A buffer安装而
   `AttributeError`，结果=`2 passed,1 failed`、RC=1。namespace
   `mujoco-fullmdp-gpu-gate.ed11390e.Pod1GPU0Slot2Taskset.50yWOSvf`封存且不重试。窄修只允许
   `_fullmdp_initialized=false`的构造observation走WAIT surface，host反例后direct suite=`50 passed,8 skipped`；
   commit `3534eeb0…`的fresh successor已越过构造并完成首个真实Full-A step，但GPU-only extras exact-key
   fixture漏列生产字段`full_a_landing_opponent_bound`，在物理断言前结果=`1 failed`、RC=1；namespace
   `mujoco-fullmdp-gpu-gate.3534eeb0.Pod1GPU0Slot2Taskset.GqVOyqcQ`封存不重试。tests-only补齐exact集合后仍须
   fresh commit/namespace重新跑三个Full-A节点；
5. portable Full-A `4096 × 25000`的唯一fresh wrapper、terminal consumer与学习趋势尚不存在；在上述真实GPU
   纵切片通过前不生成长跑success receipt。

因此当前runner必须继续写`full_a_slice_attempted`、`full_a_complete=false`，不得改名为Full-A成功。

## 4. 下一代按依赖推进

| 顺序 | 唯一改动面 | 真实验收 | 完成后删除/合并 |
| --- | --- | --- | --- |
| 1 | engine-neutral portable action table | 73行identity/timing/FK与Isaac冷builder对齐；4096 env genesis/reveal/reset仍全slot0 | fresh lane不再构造未消费的balanced sampler；删除重复manifest pin和临时question常量 |
| 2 | action-conditioned question/teacher | slot0 center经shared reverse-integration产生可揭示ball state；R03 target与reference timing同源 | 删除MuJoCo midpoint serve和`normal=-incoming`临时代码 |
| 3 | observed selected-rubber | fresh GPU actual contact edge同substep分类；held contact不重复、recontact单独记账、masked reset不改peer | generic/selected双重临时命名与重复face推断 |
| 4 | R06/R07与Reward11--13 | live landing/outcome、recovery窗口、eligible denominator和Reward20守恒 | 后端自建outcome/recovery副本 |
| 5 | portable 4096长跑件 | 同一进程25k；早期5只看容量/finite，科学里程碑只读；终点独立消费 | WAIT-only launcher和一次性过渡receipt |
| 6 | 2.0瘦身 | production callpoint census证明无人消费后再删 | legacy RSL2 runner、重复validator/owner/receipt、无restore consumer的carry graph |

依赖关系是 `catalog -> question/teacher -> observed contact -> R06/R07 -> longrun`。这些纵切片可在
Isaac active长跑期间开发和做host反例，但不能用未完成的下游测试反向授权上游。

当前第1项的cold数据部分已由全73条真实motion的双实现逐列对拍闭合；固定slot0的4096 live genesis/
reveal/reset仍需fresh运行证据。第2项的host question/teacher纵切片已闭合；下一真实阻塞是把split-ready
控制消费与当前R06/R07/Reward11--13纵切片送入同一fresh GPU gate，不能把无consumer的diagnostic qdes helper
或host的`8 skipped`代签执行。

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
上述六项阻塞，下一代尚未严格支配active Isaac，所以**现在不停**。

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
