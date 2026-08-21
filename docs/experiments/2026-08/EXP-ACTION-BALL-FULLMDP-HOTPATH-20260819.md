# EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819

> 问题：为什么fresh FullMDP `4096×24` rollout从旧诊断校准的约6--9秒变成约22秒，怎样在不删业务语义和证据的前提下恢复吞吐？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`baseline-runs-stopped-incomplete / profiled / observation-v2-host-implemented / zero-flight-host-validated / direct-lean-phase-a+b-host-validated / phase-c0+c1+c2+c2b-host-validated / Pod-matched-wall-pending`
> 证据等级：E2 fresh Pod steady wall/profiler + E1源码/host fixed-tape；Phase-C0--C2b只有静态payload proxy与host语义证据，Pod matched wall仍未测

## 1. 采用、延后、拒绝

并行调研的建议没有整包采用；以下是按当前代码、运行墙时和课程语义做的独立裁决。这里的H24/H48指
rollout horizon（每次策略更新、每个环境收集的control step数；见[术语定义](../../DEFINITIONS.md)）为
24/48，`lambda`指GAE（广义优势估计）的跨步权重参数。

| 项目 | 裁决 | 当前理由与边界 |
| --- | --- | --- |
| H24 | 历史baseline | 旧墙钟与profile继续按H24保存，不再作为V2执行配方；比较时换算H24-equivalent。 |
| H48 + 12,500 update + 8 minibatch | 采用为PPO V2 | 用户接受更长rollout作为学习算法取舍。总transition/minibatch/optimizer-step量级保持，但按旧吞吐仍约`44 s/update`，所以它不是性能修复；后续主报transitions/s与H24-equivalent。 |
| `lambda=0.98` | 采用为PPO V2 | 它提高未来TD residual的GAE trace权重，不是“reward衰减修复”；bias/variance已改变，必须fresh并由真实学习证据判断。 |
| `0 ACCEPT` | 保留为课程telemetry | 表示due opportunity仍DEFER在站稳/模仿准备阶段；balance→mimic→strike→landing需要很多step是合理课程。不得把`ACCEPT>0`做发射、安全或学习成败门。 |
| semantic Observation V2 A=`203/219` | 采用 | 历史fixed-194/A211已有root state，229迁移遗漏了它却带入raw task/owner/reward账；恢复table-relative root XYZ、heading与heading-frame COM velocity，同时删控制账本。两后端独立producer与真实scale/符号反例仍是GPU验收项，不宣称Markov闭合。 |
| 一套pose统一reset/D05/R07 | 拒绝 | stable birth、stroke entry、completed-shot recovery是三个不同意图；强行相等是错误等价和同源自证，应共享安全envelope而不是共享唯一姿态。 |
| 当前task Reward0--13 | 不改 | 业务eligible/income分母仍为零，改权重没有因果作用；先报告分母和课程阶段。 |

Gate纪律继续按HANDOFF执行：删除同一writer自证、zero-callpoint与无consumer的“安全”门；保留能被真实
production反例触发的nonfinite/overflow、identity join、物理contact/termination和durable ACK边界。
删减和host回归通过只叫结构GO；exact Pod profiler-off matched wall通过后才叫性能GO。

旧“约6秒”来自H24迭代尺度；对H48的线性等价约为`12 s/update`，只表达必须大砍迭代时间，不是固定
rollout或形式Gate。旧Isaac H24约22秒意味着按原吞吐H48约44秒；下一代至少要显著提升transitions/s，
不能用每update transition翻倍掩盖数据流仍慢。所有H48结果同时报告原始wall、transitions/s和
`wall_s × 24 / 48`的H24-equivalent。

采用：

- 在下一条fresh `4096×48×12500`同一进程的前若干PPO update开启
  [`HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES`](../../DEFINITIONS.md#hope_action_ball_full_mdp_profile_updates)，
  记录真实FullMDP env/manager/runtime的inclusive host-wall分段；达到预算后自动撤掉全部wrapper，继续同一长跑。
- 先删除D05 row uniqueness的两张`[N,N]`比较矩阵，改成有界环境索引直方图；有效行fault语义不变，
  out-of-range仍由原range gate处理。
- 已把D05 `construction_mask`前移到solver/exact/Physical之前；只对active rows执行数值主体，再按原索引
  scatter回full-N bank/chronology。该候选新增唯一一处动态`nonzero`同步，是否采用须由下一条Pod
  profiler-off配对确认；随后再处理empty-flight、selected-reset和重复owner verification。
- `4aadd698…`已实现成对zero-live-flight idle路径；host反例与consumer语义卷已闭合，
  但exact Pod matched wall仍是`未测`。
- `43d95275…`删除唯一production caller完全丢弃的Reward整record clone；`2ec32858…`把源码/
  API/lease/dependency-DAG深验收回构造边界；`e15e279d…`使`task_valid=false`的epoch clocks不再
  泄漏global training step。三者均不改active task的物理或Reward经济。
- Phase-A已把exact A/C FullMDP现役construction固定为direct lean，删除formal mode/pins/DAG/SHA
  validator/reset receipt/standalone installers/post-only escape等不可达分支；保留六个真实callpoint、
  lease/seal、component identity、reset generation/overflow、sticky poison和PhysX close。production
  净删`1,455`行，exact七文件回归=`310 passed, 15 skipped`。
- Phase-B branch candidate在旧class/import production caller静态census为0后，物理删除4,466行formal
  owner源和Reward/Physical/R06中只服务该死路径的top-cycle/exact-pin适配面；五个production文件合计
  `+58/-7,621`、净删`7,563`行。现役direct-lean reveal、lease/seal、reset generation/overflow、sticky
  poison、真实物理边界和optimizer后durable ACK不变；没有compatibility adapter。
- `7b7c9510`独立修复partial-construction teardown：缺manager、manager析构异常、live PhysX shutdown
  异常或callback重入均不能阻断其余terminal清理；in-memory stage严格按detach→stop→clear→callback
  clear→instance clear执行。terminal失败不伪装closed，也不跨调用重试旧sim，而是sticky要求cold
  process exit。该安全修复不计入性能收益，合并回归=`317 passed, 15 skipped`。

延后：

- CUDA graph、`torch.compile`和整条substep capture。动态transaction/clone/sync没有先瘦身时，capture只会把臃肿结构固化。
- rough terrain与站稳配方改动。它们改变学习分布，不能和纯工程吞吐补丁混成一个因果结论。

拒绝：

- 为每个job强制独占CPU集合。CPU可共享；验收只要求没有明显浪费或持续争抢。active样本中主线程约1核、
  GPU低利用，Yikang落在同一CPU集合的实际用量不足以解释三倍差。
- 通过降低LM迭代、关闭contact/termination/receipt或接受stale substep状态换速度。这些都会改变题目或证据语义。
- 用旧legacy `6.700 s/update`、MuJoCo native 114-D或WAIT `learn(1)`冒充当前FullMDP等价A/B。
- 为第二阶段D05 compact再加一层adapter。formal runtime owner的旧冻结inventory曾要求
  `prepare_many/preview/stage/arm/commit/journal`旧ABI，但没有production construction/hot callpoint；
  Phase-A从现役`train.py`/env wiring退役该lane，Phase-B现已物理删除它。后续只做一套compact
  transaction，不并存两套接口，也不为已删除类恢复compatibility adapter。

## 2. 当前运行证据

2026-08-20终局只读快照：

- `e8eef4fb…`目标为`4096 env × 25000 update`，止于durable ACK `4603`、累计
  `452,591,616` transitions；`ddb1e7c4…`止于ACK `3467`、累计`340,918,272` transitions。
  两条run均在2026-08-20约`14:41Z`不完整结束，不能用当前磁盘WIP解释它们。
- 两份result均写`status=failed`、`final_rc=0`、`phase=training`。日志未见traceback、OOM、
  signal/kill证据，cgroup也无OOM记录；所以退出因果是`未判定`，既不臆测OOM/外部停止，也不把
  `final_rc=0`当成完成。2026-08-20终局快照当时显示三张GPU无compute process；本次live SSH认证失败，
  不能把该快照当成当前资源状态。
- profile只在同一进程update `0..4`开启并自动关闭。五个collection wall为
  `16.150/12.890/14.560/15.000/14.358 s`；这五个带profiler且reset strata不同，不能冒充稳定吞吐成绩。
- 五个row中`post_physics_publish`固定96次/update，inclusive host wall为
  `6.998--7.809 s`；`sim_step`仅`0.904--0.960 s`，`owner_binding_assert`的552次合计
  `0.143--0.170 s`。因此第一优先级是Physical/R06/Epoch数据流，不是CPU互斥或先磨反patch断言。
- `after_command_to_observation_gap`为`2.290--3.538 s`，reward为`0.525--0.924 s`；它们是第二层。
- 运行中资源快照的trainer约`6.4 GiB`、GPU利用约十几到二十个百分点；CPU主线是单进程Python/
  小kernel/同步固定税，没有整机CPU饱和证据。终局GPU为空只表示现在有测量窗口，不改变历史归因。
- profiler-off steady窗口已收敛到约`22 s/update`：GPU0 `ddb1e7c4…`约`21.94 s`（collection
  约`20.50 s`、learn约`1.45 s`），GPU1 `e8eef4fb…`约`22.45 s`（collection约`20.74 s`、
  learn约`1.71 s`）。两者GPU利用约`27%`、trainer各只占约`1.44--1.46`个CPU core，
  所以CPU不是这条回归的因果瓶颈，也不需要CPU互斥；`6 s/update`未达。

## 3. 四个参考栈给出的共同答案

| 参考 | 保留的共同负载 | 没有的当前固定税 | 结论 |
| --- | --- | --- | --- |
| Unitree RL Lab `4960b847…` | 4096 env、24 rollout、decimation4、PPO 5×4、相近网络 | FullMDP owner graph、球生命周期、R03/R06/R07、逐substep证据事务 | CPU affinity不是其速度来源；steady step是薄GPU向量化路径 |
| BeyondMimic `cd651720…` | IsaacLab manager env、motion tracking | empty ball transaction、全场景retire write、复杂selected reset | reset仅批量采样与两次indexed sim write，普通step不写额外刚体 |
| build_3 `7b88a021…` | 相同dt/decimation/PPO | 当前23次/control的深owner验证及大量packet/clone | 6秒不能直接外推，但结构膨胀是明确差异 |
| MJLab `0fb8a681…` / Playground `e74217bb…` | batched GPU physics与manager/JIT env | Python包围20个substep、重复contact census、每步同步 | 先合并数据流与同步，再考虑更大graph capture |

这个结构税也可直接从Phase-B前源码尺度看到：现役env/lean runtime/Physical/Epoch/R06/Scene六个核心
文件合计`42,742 LOC`、`644`个`.clone()`调用点、`1,127`个class/function definition；另有
`6,539 LOC` formal owner/factory历史源，Phase-A后已不在production construction graph，Phase-B又物理删除
其中4,466行owner及其专属适配面。这不直接乘等于wall time，但它解释了为什么一个“empty flight
应为no-op”的改动会跨Physical/Scene/R06/Epoch/Env重签协议、测试与gate。优化目标不只是减
kernel，而是把mutable business state收回一个owner，让其他层只做IO或边界证据。

后续结构固定为四层，不再添加第五层“safety token”：

1. Env只拥有control/substep调度和Gym reset边界；
2. 一个device-resident mutable `ActionBallState`唯一拥有phase/generation/shot/pending/live/
   contact/outcome/fault；
3. Scene只做simulator I/O、active-flight persistent mapping和原始contact latch，不拥有business key/
   lifecycle；
4. Epoch/R03/R06/R07是transition-scoped fact/event ledger，只在真实event时写，PPO边界统一汇总。

真实安全边界继续保留：joint/nonfinite/overflow，slot/generation/key的transition join，真实PhysX
contact/net/table/landing，selected-reset行集，Reward/observation finite，optimizer成功后才持久ACK，以及
source/asset/run namespace provenance。要删的是同一writer的重复projection/receipt/digest、empty journal/park、
每substep代码身份重扫和无consumer的formal/R10对象图；这些不是独立事实源。

Observation V2也按这个结构原则落地，而不是继续给旧229/399尾部加字段：actor common183只保留机器人、
teacher与anchor状态，tail20只保留actor可观测的raw A/+Y task-normal residual、per-env倒计时、Motion phase/mask；
critic再加16-D live ball/contact/net/support/dwell。raw task45、owner/fault/age与Reward due/paid账本被删除。
base goal不等于root pose，torso anchor不等于table-frame base，q/dq也不能唯一推出COM velocity；本地
root translation/yaw/COM-velocity alias反例因此支持恢复这8个量。外部参考只作为相对base/anchor和velocity
设计方向，不代替本地因果反例。

同一轮把R06 broad observation projection从81个tensor缩为一个owner内join：用canonical shot key八字段
和publication ordinal唯一选择INBOUND/OPEN live row，只返回`flight_slot`及contact/net-crossed/net-clear三枚
latch。slot仅是读取Physical `[N,K,13]`的locator，不成为policy business key；SETTLED_RETAINED、EMPTY或
无匹配均返回无live slot。这里复用已有R06 owner和opaque projection，没有新增owner、receipt或Gate。

R07 observation也不再冒充Motion consumer或为两个support bit第二次重扫root/joint/all-body/contact：
support与dwell直接复用同一真实post-physics plant read的窄状态。cold genesis明确为zero；selected reset只把
generation精确`+1`的行在当帧置零，peer保持，下一真实post-physics恢复。host红队未见P0/P1；但Motion
observation仍是34个full-N tensor的broad clone-only view，4096环境静态约`1,224,704 bytes/control tick`，
V2只消费其中8个。这是后续single-state/窄projection的明确P2性能债，不能由host测试换算成Pod秒数。

## 4. 已定位的热路径

### 4.1 D05在mask前做全N重计算

当前每个control step固定把`4096×3×3=36,864`个candidate送进12轮LM、exact-face和Physical horizon，
最后才应用`construction_mask`。active ACK `0..808`平均只有约`52.32` selected row/control，意味着
约`78.3×`行工作没有业务机会；这是复杂度倍率，不直接冒充wall倍率。

粗略每update为约`42.5M`个3×3 solve、`7.43B`个RK4 row-step。下一profiler先单列command compute结束到
mandatory observation开始之间的after-command-to-observation gap；它包含D05 settlement及可选interval event，
不依赖本task不存在的interval event callpoint。随后再在composer内拆materialize/solver/exact/Physical。

### 4.2 二次方唯一性检查

旧实现每control建立两张`[4096,4096]`bool矩阵，约`33.55M` comparisons/control、`805M/update`。
本候选改成`torch.bincount`的`O(K+N)`有效索引计数；它本身不新增`.item/.cpu/nonzero/synchronize`。
另一个独立的mask-first数值边界包含唯一一次`construction_mask.nonzero`，不能把两项混为一谈。

### 4.3 empty-flight仍跑完整Physical/R06/Epoch

每update有`24×4=96`个physics substep。即使ACCEPT/launch/contact/retire全0，仍重复：

- arm/clone `[N,K]` identity与`[N,K,32]` hash；
- 两次scene state读取和多层packet clone；
- R06 publish/retire与ActionEpoch refresh；
- 两ball、pose+velocity共`384`次full-grid root-state write/update。

这条优化必须由不可伪造active-slot token驱动，empty为真实no-op；不能用每substep`.any().item()`换另一种同步。

### 4.4 reset与安全门重复自证

- episode length约70--80使每update约1250--1400 reset row，几乎每control都触发一次`reset_buf.nonzero`和
  selected-reset packed D2H；当前还复制full-N generation/reason packet并串行走多owner commit。
- owner executable surface约23次/control重扫，即约552次/update；protected manager capture/compare约
  192 cycle/update。完整source/DAG验证应留在construction/update boundary，热路径只保留lease/generation/
  poison/chronology token。

`2ec32858…`已删除热路径中的模块/类/方法/lease/DAG全表重扫及16个调用点，但保留构造时
admission、已绑定method dispatch、poison/reentrancy、protected manager不变式和返回ABI。它不再拦截
构造完成后受信同进程主动monkeypatch的情形；这不是运行边界上可发生的外部故障，也没有独立
事实writer可供它核验。这符合HANDOFF §3.1：不把同一writer的自洽扫描当安全边界。

### 4.5 Reward/observation clone与MuJoCo重复contact

- Isaac Reward14项支付中，调用者丢弃`pay_reward()`返回的完整ActionEpochRecord，约336次无用全record
  clone/update；R03十consumer还重复计算相同误差。
- portable MuJoCo Full-A每policy step的20个substep各扫两次full contact buffer，末尾又扫一次table；
  4096×128时每次524,288 rows。既有matched数据已证明仅base probe就造成约13%吞吐损失。

`43d95275…`已把`pay_reward()`改为只完成append/journal/ordinal/fault/close，不再给唯一且
丢弃返回值的production caller制造最后一份整record clone。这不删除Reward事实或守恒证据。

## 5. 本候选实现与验收

- 新profiler默认完全不import、不wrap；opt-in只接受canonical `1..50`。
- 使用`perf_counter_ns`，不新增CUDA同步；nested inclusive spans只作归因，不当GPU kernel计时或速度成绩。
- 覆盖env step、action/sim/scene、owner deep gate、protected state、before-policy/post-physics/after-reward外层，
  after-command-to-observation gap，以及Reward/termination/reset/command/event/observation/recorder。profiler不替换
  env已验签的top-runtime bound method，避免归因工具本身破坏身份门。
- 每update要求观测到exact 24次env step；到预算后恢复所有原instance方法并关闭自己。
- mask-first mixed/empty/full fixed-tape active rows与dense path逐位一致；invalid slot/index/fault保持同批拒绝，
  empty mask不调用五个numeric owner，mixed mask的numeric batch严格等于`active×3×3`。
- 第一条Physical瘦身cut只删除一次重复的全`[4096,2,13]`scene read和若干无所有权用途的clone：scene
  producer返回同一次fresh stack，Physical立即校验并保留自己的after-image；foreign/direct请求仍必须显式提供
  state并逐值join。它没有引入`.item/.cpu/nonzero/synchronize`，也没有做危险的empty fast return。
- 该cut focused回归=`81 passed, 5 skipped`。它不会单独把22秒变成6秒；真正的empty-flight no-op仍需一个
  能同时drain concrete fact-owner的可靠active-slot authority，不能只看Python端`pending is None`就跳过capture。
- D05 compact与profiler先前跨层回归=`110 passed, 1 skipped`；当前首个Physical cut的Pod wall改善仍`未测`。
- Reward clone cut的epoch/lean-reward回归=`76 passed, 7 skipped`；owner validation cold-boundary cut回归=
  `29 passed, 1 skipped`。idle clock新增反例证明：同一invalid task在`common_step=10`与`1,000,000`时
  229/399输出完全一致，valid row仍保留原tick差。
- 第二阶段D05 compact ABI尝试完整回退，production/test diff均为0。可达性反例进一步证明：
  formal owner只是静态stale API consumer，无construction/hot callpoint；唯一“formal成功”测试是自造
  `launch_authorized=true`的fixture。Phase-A据此把exact A/C FullMDP现役入口固定为direct lean：
  `train.py +176/-478`、`full_mdp_env.py +108/-1261`，production合计`+284/-1739`、净删`1,455`行。
  exact七文件回归=`310 passed, 15 skipped`；formal owner源当时仍dormant，现已由Phase-B物理删除。
- Phase-A是结构GO而非性能GO。它删的是无consumer的formal dispatch与同一writer自证，保留construction
  executable binding、六个真实callpoint、lease/seal、component identity、reset generation/overflow、
  sticky poison与PhysX close；没有Pod matched wall，不能把这`1,455`行删除换算成秒数。

Phase-A当时的exact host回归命令为：

```bash
PYTHONPATH=hope_training/whole_body_tracking/source/whole_body_tracking \
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_env_runtime_callpoints.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_lean_env_install.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_policy_bootstrap.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_post_physics_env.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_runtime_factory_callpoint.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_train_wiring.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_runtime_owner.py
```

Phase-A提交结果=`310 passed, 15 skipped`；加上`7b7c9510`的partial-close故障注入后，同一命令结果=
`317 passed, 15 skipped`。15个skip均按host dependency条件触发，不是失败。该回归验证现役construction/
callpoint/reset/owner wiring，不替代exact Pod fixed-tape、真实GPU语义或profiler-off matched wall。

Phase-B继续做物理删除，而不是给不可达壳再加兼容协议：删除4,466行
`action_ball_full_mdp_runtime_owner.py`，并从Reward、Physical与R06移除它专属的top-cycle token、
exact-class/source pin和close graph；同时删除formal owner、continuous racket transaction bridge与旧R10
runner checkpoint的专属测试。五个production文件合计`+58/-7,621`、净删`7,563`行。静态census只证明
旧class/import没有production caller，不代签动态GPU语义；仍可见的通用runner命名与个别leaf formal shell不在
本slice内，后续只能按零caller逐项删除，不能把本次结果写成single-owner state已经完成。

只含Phase-B staged slice的独立临时worktree使用以下scoped union复现：

```bash
PYTHONPATH=hope_training/whole_body_tracking/source/whole_body_tracking:hope_training/whole_body_tracking/scripts:hope_training/whole_body_tracking/tests \
PYTHONDONTWRITEBYTECODE=1 \
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_rsl3_adapter.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_runner_drain.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_lean_runtime.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_env_runtime_callpoints.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_train_wiring.py \
  hope_training/whole_body_tracking/tests/test_action_ball_landing_outcome_device.py \
  hope_training/whole_body_tracking/tests/test_action_ball_landing_outcome_epoch_direct.py \
  hope_training/whole_body_tracking/tests/test_action_ball_landing_outcome_selected_reset.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_epoch_rowwise.py::test_open_reward_debt_blocks_reset_drain_and_checkpoint \
  hope_training/whole_body_tracking/tests/test_action_ball_physical_flight_device.py \
  hope_training/whole_body_tracking/tests/test_action_ball_physical_epoch_hot_lane.py \
  hope_training/whole_body_tracking/tests/test_action_ball_physical_flight_selected_reset.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_ball_scene_postphysics.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_cfg_registration.py \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_lean_rewards.py
```

结果=`489 passed, 37 skipped`；独立终审=`P0=0/P1=0`。这些skip是host dependency条件，不是失败。
输入是只含本Phase-B staged slice的源码，输出只证明direct-lean host contract与故障反例未被旧壳删除破坏；
exact Pod fixed-tape、真实GPU与profiler-off matched wall仍`未测`，所以Phase-B同样只有结构GO。

successor验收是同一`4096×48×12500`进程：前若干update只读profile，之后profile自动关闭继续；性能结论使用
profile-off后相同reset/live-flight/D05 strata，同时报告原始wall、transitions/s与H24-equivalent。旧H24约6秒
线性对应H48约12秒，只是加速尺度，不是硬阈值。任何数值优化还须逐step对齐done/reason/reset IDs/
generation/RNG、D05/R03/R06/R07/payment/retire、203/219 obs和Reward20。

## 6. 下一步

1. 先原子闭合Observation V2 A203/219及training/snapshot contract；本次live SSH认证失败，未取得当前
   GPU状态，不能沿用旧“GPU均空”快照。候选source/runner须先冻结为clean、可复现执行件，不得从dirty WIP发射。
   本轮一份未入Git的565行ABBA实现已因契约自证删除：测试自造ACK schema v10，
   exact baseline/candidate runner均发schema v11，真跑必然被其自己拒绝。successor测量器必须从
   exact runner的live wire contract建立fixture，不另造一份schema/gate。
2. 在MuJoCo以同一deterministic pair闭合EPA24-fail/48-finite、exact GPU和独立oracle；容量gate保持fail-stop，
   不从r3恢复。
3. 把D05 producer→R05→Physical/R06/Epoch收敛为一个device-resident `ActionBallState`：K-row只构造一次、
   sparse commit、真实event才写delta，并让`K=0`/zero-live-flight成为自然的无业务路径。
4. 只有fixed-tape parity和exact Pod profiler-off matched H48 wall都通过，才把successor称为严格更强；两条旧run已经结束，
   successor必须fresh且不得resume旧checkpoint。没有净收益就撤回，不靠host测试或主观保留。
5. 通过后使用fresh namespace先短验真实构造/finite/callpoint，再在同一进程继续训练；不插`ACCEPT>0`表现门，
   按balance→mimic→entry→strike→landing报告分母。

## 7. `ddb1e7c4` successor实测：single-read cut没有打中首墙

2026-08-20在自然空闲的Pod1 GPU0启动fresh immutable successor：commit=`ddb1e7c4…`、
`4096 env × 25000 update`，CPU=`32--47`。第一次wrapper在namespace创建前被GPU XML的全机graphics-process
列表误拒；目标GPU0的compute-process查询实际为空。该尝试没有创建run root，也没有Python/Kit/GPU调用。
successor只把resource gate改为按目标UUID过滤`nvidia-smi --query-compute-apps`，使用fresh approval与namespace，
未复用失败尝试。

有效successor前5个profile row为：

| update | collection (s) | postphysics (s) | sim (s) | owner gate (s) | command->obs gap (s) | reward (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15.424 | 7.624 | 0.934 | 0.160 | 3.070 | 0.772 |
| 1 | 13.776 | 7.101 | 0.916 | 0.159 | 3.058 | 0.605 |
| 2 | 13.981 | 7.208 | 0.904 | 0.172 | 2.989 | 0.582 |
| 3 | 15.555 | 7.082 | 0.926 | 0.174 | 3.087 | 0.676 |
| 4 | 16.198 | 8.593 | 0.929 | 0.190 | 3.306 | 0.655 |

均值collection=`14.987 s`、median=`15.424 s`；均值postphysics=`7.522 s`、sim=`0.922 s`、
owner gate=`0.171 s`、command-to-observation gap=`3.102 s`、reward=`0.658 s`。这组带profile的前缀不作为
最终profiler-off吞吐成绩，但它足以否定“删除一次scene read/clone即可消除首墙”：postphysics仍与旧run同量级。

该successor此后曾继续同一25k进程收集profile-off与学习证据，最终止于ACK `3467`并不完整结束；旧GPU1
run最终止于ACK `4603`。当前cut从未证明严格支配旧版。下一刀必须让**可证明的zero-live-flight窗口**
同时跳过prephysics arm与postphysics capture/R06/
retire/park，并保持arm/capture成对。仅在postphysics以Python `pending is None`早退会让scene fact-owner未drain，
下一substep重arm直接失败，已明确拒绝。验收仍须matched fixed-tape与profile-off wall，不以代码更少或前5次
总时间下降作结论。

profile自动关闭后的matched `20..97`窗口进一步否定该cut的wall收益。旧GPU1与新GPU0在iteration 97的
science telemetry逐字同为mean reward=`5.44`、mean episode length=`92.99`、total timesteps=`9,633,792`，
说明固定seed语义轨迹没有因single-read cut改变；但iteration wall为：

| run | n | mean (s) | median (s) | p95 (s) | min--max (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| old `e8eef4fb…` GPU1 | 78 | 17.570 | 16.415 | 22.81 | 13.50--23.46 |
| new `ddb1e7c4…` GPU0 | 78 | 18.004 | 17.610 | 22.98 | 15.00--25.15 |

不同物理卡仍可能贡献噪声，因此这不证明cut本身“变慢”；但它明确没有达到采用所需的可测净收益，更没有
接近6秒目标。两条run现均已不完整结束，single-read cut不作为性能胜利；下一candidate只准针对完整
empty-flight事务，并从fresh namespace验证。

## 8. zero-live-flight成对idle路径：host语义已闭合，Pod wall未测

zero-live-flight是“下一个control step没有任何Physical、R06或scene callback业务”，不是
`pending is None`的Python快捷判断。实现在D05 settle之后、每个control boundary只做一次合并
device-to-host verdict：同时检查Physical pending/published/lifecycle/active slot/selected-contact、
R06 live state与scene callback activity。任一输入的shape/dtype/device/chronology不合同或读取失败
都先清旧cache并fail dense，不会沿用上一control的idle结论。

只有上述writers全部为空时，本control的四个physics substep才走成对轻量路径：

- prephysics打开一个与当前callback epoch绑定的idle fact-source；
- postphysics清理candidate/binding fault、关闭binding，并严格推进scene heartbeat fence、scene exact
  stamp和Physical exact stamp；
- 不读/写scene state tensor，不构造Physical facts packet，不做空R06 publish/retire、全false
  Epoch journal或false-mask full-grid ball write。

mixed/full、false→D05 ACCEPT、active flight、retire所在control或任何sticky fault仍走原完整dense
path；selected reset、true reset、restore会使cache失效，checkpoint不能穿过未配对的idle lease。
空R06 mutation与全false Epoch journal没有科学consumer，所以本路径不为旧结构伪造它们；
203/219 semantic observation、Reward20、done/reason/reset/RNG和decoded evidence必须保持同义。

host zero-flight三个focused文件各用独立Python进程，避免既有flat/package双namespace测试收集冲突；
回归为`112 passed, 4 skipped` / 约4秒。更宽的consumer semantic回归为
`166 passed, 8 skipped` / 约8秒。反例覆盖idle→ACCEPT→active→retire→idle、R06-live/Physical-empty、
pre/post缺失、重复/倒序stamp、callback heartbeat/candidate、selected reset、restore/checkpoint与mixed/full
fixed tape。这些只是host语义证据：exact Pod fixed-tape与profiler-off `4096×24` matched wall均
仍`未测`，因此尚不声称从22秒恢复到6--9秒，也不声称严格支配已结束的baseline run。

此前只有算术预算：zero-flight加两个结构cut曾估算H24 iteration约`8.4--9.1 s`；这不是Pod实测，
也没有证明任何H48速度。formal legacy consumer现已由Phase-B删除，下一刀不是继续适配旧full-N接口，
而是把mutable business state收成单一`ActionBallState`后用matched A/B判定真实wall；不能用预算代签。
这也符合HANDOFF §3.5/3.6：只把不可学或证据不可信当阻塞，确定性结构事实用
直接consumer/dataflow追溯，不用随机rollout为它代言。

后续steady hot path每新增一个`.clone/.item/.cpu/.nonzero/full-N write`，都必须同时记录它的
频率×shape和profile证据；一个事实只保留一个mutable owner，证据汇总尽量放在control/PPO
boundary，不再把无业务事件的结构自证写回每个physics substep。

## 9. Phase-C0：K=0只推进chronology，不制造D05事务

Phase-C0把after-command的业务行定义为
`due | closed | (close_reason != NONE) | previous_paid.valid`。仅当该mask全false时，持久状态只推进
`_next_epoch`与`_last_motion_common_step`两个scalar chronology；不创建`_active_d05`，也不改变公开
ActionEpoch record、commit head、milestone或sticky fault。这样K=0不再进入Question/RNG数值构造、第二次
Motion projection、Motion/Racket/R05三writer、post-D05 Motion reseal，也不再制造10条全空journal entry。
按`N=4096`逐tensor payload静态相加，后者约为`3,141,632 B/control`；这是源码shape/dtype proxy，
不包含allocator开销，也不是实测CUDA流量、显存峰值或wall收益。

独立的settlement减法把唯一caller不消费的`settle_d05_transaction()`整份record返回clone改为`None`；
按当前record的44个tensor静态相加，少构造约`6,242,304 B/settlement`。真实consumer需要状态时改读owner
当前record，不新建shadow owner或receipt。K>0、任何`closed`/non-NONE close reason、previous-paid row仍走
原dense D05与payment/close chronology；既有fault会继续sticky并在下一业务行走原CENSOR/fail-stop反例。
Phase-C0也没有改§8的Physical pre/post成对idle callback，不能用D05 K=0早退拆散arm/capture。

可复现host focused union为：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=hope_training/whole_body_tracking/source/whole_body_tracking:hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp:hope_training/whole_body_tracking/tests \
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_epoch_rowwise.py \
  hope_training/whole_body_tracking/tests/test_action_ball_d05_rowwise_transaction.py \
  hope_training/whole_body_tracking/tests/test_action_ball_motion_rowwise_accept_writer.py \
  hope_training/whole_body_tracking/tests/test_action_ball_bootstrap_ready_d05_accept.py \
  hope_training/whole_body_tracking/tests/test_action_ball_racket_rowwise_accept.py \
  hope_training/whole_body_tracking/tests/test_action_ball_continuous_runtime_transaction_device.py
```

结果=`160 passed, 12 skipped`；独立终审=`P0=0/P1=0`。反例覆盖K0→K>0的epoch/RNG连续性、invalid-close
fault跨K0保持、previous-paid强制进入业务路径及writer/settlement合同。尚存P2是CUDA上的
`torch.equal(business_rows, zeros_like(...))`会产生device-to-host同步；必须在exact Pod profile中量化后再决定
device-resident替代，不能把它藏进“无业务”命名。当前没有Pod fixed-tape、profiler-off matched wall或H48
成绩，因此不声称达到6秒或任何性能GO。该cut按HANDOFF的原则直接退役zero-business工作，而不是为它增加
gate、owner或空journal；G05保持`Partial`。

## 10. Phase-C1：Motion只发布真实consumer并集

生产callgraph中，Motion的34-tensor observation view只有两个字段consumer：semantic Observation V2读取
`control_tick/phase/reset_generation/action_uid/task_valid`与三个float64倒计时，Physical launch读取
`control_tick/reset_generation/action_uid/task_identity/swing_generation`。二者并集精确为6个`int64[N]`、
1个`bool[N]`和3个`float64[N]`；其余24个字段没有production consumer。Epoch/D05使用的15-field cadence
projection是另一份真实事务合同，本刀没有混删。

Phase-C1把owner-private broad dict替换为现有Motion owner内的一份frozen typed snapshot。publication只对7个
live owner buffer执行`detach().clone()`；三个倒计时是该publication刚产生的新tensor，只需`detach()`。
opaque token、publication identity、common step、D05写前撤销/写后同tick重发及selected-reset row-wise
generation/tick语义全部保留。两个内部consumer当前虽均只读，红队仍拒绝了“validator直接返回同一snapshot”
的试作：`frozen dataclass`不能冻结Tensor内容。最终validator为每个consumer返回一份
10-field隔离clone，误写一个view不会污染owner-private record或同tick另一consumer。下一publication创建
新record，旧view字节保持；没有新增owner、adapter、receipt或Gate。

`N=4096`时，旧单份34-tensor payload静态为`1,224,704 B`；新10-field snapshot为`299,008 B`。
按publish加两个consumer各一次读取，静态proxy从`3,674,112 B/tick`降为`897,024 B/tick`，少
`2,777,088 B/tick`（约75.6%）。纯K0只走publish+Observation getter时，H48尺度约少
`88,866,816 B/update`复制payload与约2,448个clone call；这些只由
shape/dtype/frequency相加，不代表allocator、CUDA流量、kernel wall或显存峰值。

主线将四个consumer面分进程复跑：Motion chronology=`21 passed`、row-wise due/closure=`6 passed`、semantic
Observation=`14 passed`、Physical hot lane=`32 passed`；pycompile与diff-check通过。`task_valid`虽可由Motion
phase严格推出，但删除它会把policy ABI从203/219改成202/218，因此明确延后为独立learning-interface裁决，
不与本次复制减法混在同一因果刀。Pod fixed-tape、profiler-off matched H48 wall及6秒级目标仍`未测`，G05
保持`Partial`。

## 11. Phase-C2：mutator不返回无人消费的整record

`ActionEpochRecord.clone()`在`N=4096,S=1`包含44个tensor，静态大小精确为`6,242,304 B`
（`5.953125 MiB`）。production caller census显示下列五个mutator的返回值全部被丢弃：
`merge_runtime_owner_fault`、`publish_owner_facts`、`refresh_physical_launch_rows`、
`refresh_physical_postphysics_rows`和`refresh_r06_outcome_rows`。其中postphysics是经已验证bound method动态调用，
同样只执行而不绑定返回。Phase-C2把这五个方法改为`None`返回并删除末尾整record clone；真正被Motion phase
和Reward cycle消费的`publish_motion_playback_started`、`open_reward_cycle`返回合同没有动。

R03每次publish还曾为读取`fact_f32.shape[-1] == 32`而额外调用一次`epoch.current()`；现改用Epoch公开的
`OWNER_FACT_F32_WIDTH`。旧AST/源码形状测试同步替换为真实R03发布反例，逐行验证24个业务值、8个保留零、
valid/source-step/fault与ineligible行。没有删除任何mutation、journal append、milestone、fault latch、sticky
poison、selected-reset或Physical/R06 paired callback；这些都在旧return clone前已完成。

静态下界按真实频率计算：固定每control的三次fault merge、两次owner fact publish和一次R03 width read，
H48共少288份record，即`1,714.5 MiB`（约`1.674 GiB`）与12,672个tensor clone/copy call。active dense在
decimation4的每个substep再少Physical launch/postphysics/R06三份返回，H48共少576份，即`3,429 MiB`
（约`3.349 GiB`）与25,344个call；合计active下界约`5.022 GiB/update`与38,016个call。这里只是
shape/dtype/frequency proxy，不能折算wall或显存峰值。

主线分进程复跑Epoch rowwise、R03/R07 alignment、R03 direct Reward、Physical hot lane、lean Rewards、lean
runtime与postphysics env，共`168 passed, 7 skipped`；pycompile/diff-check通过。独立caller/paired/reset审查
为`P0=0/P1=0`。下一步仍需exact Pod fixed-tape逐项对齐Reward20、actor203/critic219、Epoch final state、
Physical/R06 substep lifecycle及fault/reset/overflow，再做profiler-off matched H48 wall；G05保持`Partial`。

## 12. Phase-C2b：Observation先查cache再clone

semantic Observation的policy与critic由同一source、同一control step依次读取。旧实现虽然缓存已打包的
actor/critic，却先调用`ActionEpochOwner.current()`复制44个tensor，再用复制出的epoch/version发现第二组
命中cache。该副本没有独立consumer，也不参与任何安全判定。

Phase-C2b把cache判定前移到clone之前，复用已有的`common_step`与owner host scalar `commit_head`。
Genesis满足`version=0, commit_head=1`；所有真实Epoch mutation唯一经`_append()`同时令二者各加一，carry
restore也强制`current.version == commit_head - 1`。因此同step、同commit才返回cache；same-step真实commit或
新control step都会miss并读取新record。owner poison仍在cache之前检查，不会用旧finite observation逃逸。
这不是新鲜度Gate或第二份状态，只是把现有cache放到昂贵copy之前。

critic由`torch.cat((actor, critic_extension))`构造，完整包含actor prefix。旧实现随后分别扫描actor和critic
finite；前一次扫描不可能发现后一次扫描看不到的actor nonfinite。现只保留critic reduction，hard-coded
actor行同时注入NaN/Inf的真实pack反例仍fail，没有zero fallback。

`N=4096,S=1`时一次record clone静态为`6,242,304 B`、44个tensor；按每control一对policy/critic调用，H48
少48份，即`299,630,592 B = 285.75 MiB/update`与2,112个clone/copy call，另少一次actor finite
scan/control。focused Observation回归=`14 passed`，覆盖policy→critic只读一次current、same-step真实commit
重建、新step重建、热cache后poison fail及actor NaN/Inf。该数字仍只是shape/dtype/frequency proxy；真实
manager调用频率、CUDA traffic和wall须由exact Pod profile确认。R07同一事务三份Epoch snapshot的收敛是
下一独立提交，不与本刀混因果；G05保持`Partial`。

## 13. Phase-C2b：R07一次事务只读一份Epoch快照

R07的单次post-physics publication此前在bundle、Motion frame-0 reference和reward view中各调用一次
`ActionEpochOwner.current()`。三次之间没有合法Epoch mutation，后两份44-tensor record只有一致性检查消费，
却每control重复复制。现在bundle在栈内读取唯一一份snapshot，并把它传给真实Motion public producer和
non-mutating reward view；snapshot不写入owner字段，不成为token、receipt或新的authority。

第一版曾错误地连stale caller边界一起删掉，独立红队据此构造出内部一致但已过期的
`(epoch, reference)`。最终实现保留两处仅比较Python整数的真实freshness：view要求
`epoch.version == reference.epoch_version == owner.commit_head - 1`，bundle在首次ActionEpoch mutation紧前
再次要求snapshot仍对应当前head。两项检查都不调用`current()`、不clone tensor；行为反例分别覆盖stale pair
和计算完成后、首写前owner推进。parent/owner/full-key、selected reset、fault/nonfinite、dwell与首次写入后的
原事务顺序不变。

`N=4096,S=1`时，一份record静态为`6,242,304 B`与44个tensor clone/copy call。本刀把每control的3份降为
1份，H48静态少`599,261,184 B = 571.5 MiB/update`与4,224个call；这是shape/dtype/frequency proxy，
不是wall或显存峰值。五个相关文件分进程回归合计`99 passed, 6 skipped`，pycompile/diff-check通过；独立
终审`P0=0/P1=0`。exact Pod fixed-tape、profiler-off matched H48 wall仍`未测`，G05保持`Partial`。
