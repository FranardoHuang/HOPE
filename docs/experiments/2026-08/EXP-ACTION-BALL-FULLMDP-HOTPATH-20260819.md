# EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819

> 问题：为什么fresh FullMDP `4096×24` rollout从旧诊断校准的约6--9秒变成约22秒，怎样在不删业务语义和证据的前提下恢复吞吐？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`implementation-candidate / Pod-profiler-not-yet-run`
> 证据等级：E1源码与host反例；active run只读E2运行快照

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

2026-08-19 16:40 CST只读快照：

- active commit=`b64cb944…`，目标=`4096 env × 25000 update`；完整WAL为`2260`行，即ACK `0..1129`；
  累计`111,083,520` transitions。
- update约1062到1129的26分钟窗口约`23.3 s/update`；此前console长期是collection约`21--24 s`、
  learning约`1.3--1.8 s`。
- trainer约`144% CPU`、135 threads；GPU0约`6.4 GiB`、瞬时利用约`22%`。GPU1当时无compute进程且仅`2 MiB`。
- active不可热补、不可用磁盘新代码解释。它继续作为旧代长跑；successor使用另一fresh namespace和自然空闲卡。

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
interval event开始之间的after-command settlement gap；随后再在composer内拆materialize/solver/exact/Physical。

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
  after-command settlement gap，以及Reward/termination/reset/command/event/observation/recorder。profiler不替换
  env已验签的top-runtime bound method，避免归因工具本身破坏身份门。
- 每update要求观测到exact 24次env step；到预算后恢复所有原instance方法并关闭自己。
- mask-first mixed/empty/full fixed-tape active rows与dense path逐位一致；invalid slot/index/fault保持同批拒绝，
  empty mask不调用五个numeric owner，mixed mask的numeric batch严格等于`active×3×3`。
- 跨D05/FullMDP/profiler/RSL focused回归=`110 passed, 1 skipped`；GPU/真实Isaac binding、dynamic compact
  同步代价与wall改善仍`未测`。

successor验收是同一`4096×25000`进程：前5个update只读profile，之后profile自动关闭继续；性能结论使用
profile-off后相同reset/live-flight/D05 strata，目标median collection `<=6.5 s`、p95 `<=8 s`。任何数值优化还须
逐step对齐done/reason/reset IDs/generation/RNG、D05/R03/R06/R07/payment/retire、229/399 obs和Reward20。

## 6. 下一步

1. commit/push profiler、线性uniqueness和mask-first候选，生成fresh GPU1 successor wrapper；不停止active GPU0。
2. 消费前5个profile row，并比较profile自动关闭后的collection；dynamic compact若没有净收益就撤回，不靠主观保留。
3. 后续依次做empty-flight no-op、selected-reset compact transaction、cold/update-boundary owner validation、
   Reward/obs批处理。
4. MuJoCo并行先完成action0 question/teacher，再合并contact census；R06/R07/Reward11--13闭合前不叫Full-A长跑。
