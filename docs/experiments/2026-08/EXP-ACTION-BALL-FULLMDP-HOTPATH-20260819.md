# EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819

> 问题：为什么fresh FullMDP `4096×24` rollout从旧诊断校准的约6--9秒变成约22秒，怎样在不删业务语义和证据的前提下恢复吞吐？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`profiled / first-semantics-preserving-cut-host-validated`
> 证据等级：E2 fresh Pod profiler + E1源码/host反例；首个瘦身cut仍待Pod配对

## 1. 采用、延后、拒绝

采用：

- 在下一条fresh `4096×25000`同一进程的前若干PPO update开启
  [`HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES`](../../DEFINITIONS.md#hope_action_ball_full_mdp_profile_updates)，
  记录真实FullMDP env/manager/runtime的inclusive host-wall分段；达到预算后自动撤掉全部wrapper，继续同一长跑。
- 先删除D05 row uniqueness的两张`[N,N]`比较矩阵，改成有界环境索引直方图；有效行fault语义不变，
  out-of-range仍由原range gate处理。
- 已把D05 `construction_mask`前移到solver/exact/Physical之前；只对active rows执行数值主体，再按原索引
  scatter回full-N bank/chronology。该候选新增唯一一处动态`nonzero`同步，是否采用须由下一条Pod
  profiler-off配对确认；随后再处理empty-flight、selected-reset和重复owner verification。

延后：

- CUDA graph、`torch.compile`和整条substep capture。动态transaction/clone/sync没有先瘦身时，capture只会把臃肿结构固化。
- rough terrain与站稳配方改动。它们改变学习分布，不能和纯工程吞吐补丁混成一个因果结论。

拒绝：

- 为每个job强制独占CPU集合。CPU可共享；验收只要求没有明显浪费或持续争抢。active样本中主线程约1核、
  GPU低利用，Yikang落在同一CPU集合的实际用量不足以解释三倍差。
- 通过降低LM迭代、关闭contact/termination/receipt或接受stale substep状态换速度。这些都会改变题目或证据语义。
- 用旧legacy `6.700 s/update`、MuJoCo native 114-D或WAIT `learn(1)`冒充当前FullMDP等价A/B。

## 2. 当前运行证据

2026-08-20只读快照：

- active commit=`e8eef4fb…`，目标=`4096 env × 25000 update`；完整WAL已到ACK `1186`，累计
  `116,686,848` transitions。该run只读继续，不能用当前磁盘WIP解释。
- profile只在同一进程update `0..4`开启并自动关闭。五个collection wall为
  `16.150/12.890/14.560/15.000/14.358 s`；这五个带profiler且reset strata不同，不能冒充稳定吞吐成绩。
- 五个row中`post_physics_publish`固定96次/update，inclusive host wall为
  `6.998--7.809 s`；`sim_step`仅`0.904--0.960 s`，`owner_binding_assert`的552次合计
  `0.143--0.170 s`。因此第一优先级是Physical/R06/Epoch数据流，不是CPU互斥或先磨反patch断言。
- `after_command_to_observation_gap`为`2.290--3.538 s`，reward为`0.525--0.924 s`；它们是第二层。
- 2026-08-20资源快照中trainer约`6.4 GiB`、GPU利用约十几到二十个百分点，GPU0自然空闲；CPU主线仍是
  单进程Python/小kernel/同步固定税，没有整机CPU饱和证据。

## 3. 四个参考栈给出的共同答案

| 参考 | 保留的共同负载 | 没有的当前固定税 | 结论 |
| --- | --- | --- | --- |
| Unitree RL Lab `4960b847…` | 4096 env、24 rollout、decimation4、PPO 5×4、相近网络 | FullMDP owner graph、球生命周期、R03/R06/R07、逐substep证据事务 | CPU affinity不是其速度来源；steady step是薄GPU向量化路径 |
| BeyondMimic `cd651720…` | IsaacLab manager env、motion tracking | empty ball transaction、全场景retire write、复杂selected reset | reset仅批量采样与两次indexed sim write，普通step不写额外刚体 |
| build_3 `7b88a021…` | 相同dt/decimation/PPO | 当前23次/control的深owner验证及大量packet/clone | 6秒不能直接外推，但结构膨胀是明确差异 |
| MJLab `0fb8a681…` / Playground `e74217bb…` | batched GPU physics与manager/JIT env | Python包围20个substep、重复contact census、每步同步 | 先合并数据流与同步，再考虑更大graph capture |

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

### 4.5 Reward/observation clone与MuJoCo重复contact

- Isaac Reward14项支付中，调用者丢弃`pay_reward()`返回的完整ActionEpochRecord，约336次无用全record
  clone/update；R03十consumer还重复计算相同误差。
- portable MuJoCo Full-A每policy step的20个substep各扫两次full contact buffer，末尾又扫一次table；
  4096×128时每次524,288 rows。既有matched数据已证明仅base probe就造成约13%吞吐损失。

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

successor验收是同一`4096×25000`进程：前5个update只读profile，之后profile自动关闭继续；性能结论使用
profile-off后相同reset/live-flight/D05 strata，目标median collection `<=6.5 s`、p95 `<=8 s`。任何数值优化还须
逐step对齐done/reason/reset IDs/generation/RNG、D05/R03/R06/R07/payment/retire、229/399 obs和Reward20。

## 6. 下一步

1. commit/push当前single-read/clone瘦身cut，使用自然空闲GPU0和fresh namespace启动同样`4096×25000` successor；
   旧GPU1 run在successor真实通过前继续只读，不因普通坏return而停。
2. 消费successor前5个profile row，并比较profile自动关闭后的matched collection；没有净收益就撤回，不靠主观保留。
3. 后续依次做可证明且能配对drain的empty-flight no-op、selected-reset compact transaction、cold/update-boundary owner validation、
   Reward/obs批处理。
4. MuJoCo并行先完成action0 question/teacher，再合并contact census；R06/R07/Reward11--13闭合前不叫Full-A长跑。

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

该successor继续同一25k进程，用于收集profile-off与学习证据；旧GPU1 run也继续只读，因为当前cut尚未证明
严格支配旧版。下一刀必须让**可证明的zero-live-flight窗口**同时跳过prephysics arm与postphysics capture/R06/
retire/park，并保持arm/capture成对。仅在postphysics以Python `pending is None`早退会让scene fact-owner未drain，
下一substep重arm直接失败，已明确拒绝。验收仍须matched fixed-tape与profile-off wall，不以代码更少或前5次
总时间下降作结论。
