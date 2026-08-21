# ActionBall 双后端长跑：当前执行 TODO

> 状态：`ACTIVE-dual-backend-H48-longruns / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-22
>
> `origin/main:docs/NOW.md` 是全项目唯一优先级权威。本页只维护
> [FullMDP](../DEFINITIONS.md)（完整球路、击球、落点与恢复状态机）单动作双后端
> successor 的依赖顺序、运行事实和完成条件，不维护竞争性的优先级队列。旧的单动作执行页已转为
> [只读历史账](action_ball_single_action_dual_backend_todo_20260817.md)。

## 0. 2026-08-22 学习阻塞修复与 fresh 重启 TODO

本节是当前 branch-scoped successor 的顺序清单，不改变 `origin/main:docs/NOW.md` 的项目优先级。
旧 MuJoCo/Isaac H48 长跑已经在successor验证后精确停止并只读封存；不得hot-patch、resume或复用namespace。
两条fresh successor各自从独立clean checkout运行，不共享可变source。

- [x] 保存两条现役 run 的连续 ACK、finite、速度、episode、阶段分母和 Reward14--19 趋势快照。
- [x] 查清两个 backend 的 `motion_body_ori` 都长期只有约理论最大值 `0.3%` 的根因；区分 reference/frame
  错配、body 集合污染和指数 Reward 饱和，不能凭 aggregate return 猜修法。
- [x] 按 `HANDOFF_TO_CODEX_20260808.md` §3 四问审计 D05/R07：真实 safety invariant 与
  balance→mimic→entry→strike→landing 课程推进必须分离；退役为别种诊断 run 设计或按构造恒真的门。
- [x] 冻结 adopt/defer/reject：课程采用自然事件 eligibility；上一阶段开始可用时下一阶段立即已有分母，不新增
  冗余或部署不可观测的 actor observation，不用 task Reward 调权掩盖零 eligible denominator。
- [x] 实现最小可学修复，并保留现有 13 项 readiness 分解的更新摘要；诊断 telemetry
  不得伪装成 policy observation、owner、receipt 或新的 safety Gate。
- [x] 复核 PPO action-noise/entropy/adaptive-KL 实际曲线，明确采用或拒绝显式有界 schedule；H48/
  `lambda=.98` 保持已采用，不能把 rollout 变化冒充性能修复。
- [x] 完成 host 聚焦回归、exact Pod 测试、有限短验和必要的固定条件数值/行为检查；学习语义变更必须
  使用 fresh checkpoint lineage。
- [x] successor 就绪后保存旧 run 最终证据，按精确 PID/PGID/namespace 收口旧 MuJoCo 与 Isaac；确认
  GPU/lock/process absent 后，用两个 fresh namespace 同时重启。
- [ ] 验收新 run 的 durable ACK、finite、wall/throughput、六项 mimic 梯度和逐阶段 denominator；
  正式判断仍按 balance→mimic→entry→strike→landing，不以早期 `ACCEPT=0` 单独停车。

### 本轮冻结的 adopt / defer / reject

- **Adopt — 课程推进：** episode 前 `295` 个 policy tick（约 `5.9 s`）只学习可持续站立；活到首个
  due tick 且仍是有限、未终止的 row 就进入 task reveal。随着能活到该 tick 的 row 增多，mimic 样本自然
  与 balance 样本重叠；mimic 稍有成形后，同一个 task 自带的 ball flight/contact/landing 奖励立即可学，
  不再另设一扇会把下游分母清零的课程门。
- **Adopt — reference：** hidden balance 阶段的 joint/body reference 必须来自同一份 reset 后静态 ready
  tuple；reveal 后才原子切到 action frame 0 和 bridge。不得再让 joint teacher 要求 runtime default、而
  14-body teacher 同时要求 measured action frame 0。
- **Adopt — safety 边界：** finite、joint envelope、跌倒/碰台等真实 plant invariant 继续硬失败；R07 的
  13 项 recovery 误差只做已发生 shot 后的恢复奖励和可读诊断，不再授权 task 是否可以出现。
- **Adopt — learner：** 保持已经采用的 `rollout=48`、`lambda=.98` 和 fresh lineage；先修零分母与错误
  reference，再判断 exploration schedule，不能用噪声或调 Reward 权重替代可学性修复。
- **Reject：** 不新增 actor observation；当前 semantic actor 已有 base pose/velocity/heading、teacher、
  task geometry、阶段和倒计时。13 项误差来自 critic 可观测的 plant/reference，只需要摘要 telemetry，
  再喂给 actor 会冗余并扩大部署合同。
- **Adopt — orientation梯度：** 先修正reference，再对同一14-body角误差使用`.4 rad` fine与`1.0 rad`
  coarse等权核；coarse只恢复大误差区梯度，不改变target或制造额外状态。
- **Reject：** 不通过单独放宽 `motion_body_ori` 的 `std` 掩盖 reference 错配；不把 R07 all-of 阈值改名成
  “安全”；不添加 receipt/owner/counter/gate 来证明按构造恒真的事实。
- **Defer：** `solver_solve_many` 的更低 `cq_n_iters` 和其他 MuJoCo 数值性能取舍独立做 fixed-tape parity
  canary；它们不与本轮学习语义修复捆成一个不可归因的改动。

实现、证据与fresh验收的详细真源见
[2026-08-22课程解阻实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md)。

## 1. 当前运行事实

### 2026-08-22课程解阻后的现役successor

- Isaac现役：commit `333f9490…`，namespace
  `fullmdp-a-h48-v2-isaac-unblock-333f9490-20260822`，GPU1，launcher/child=`2213515/2213532`；
  22:55 UTC durable ACK已到至少69且每轮Reward `196,608/196,608` finite。profiler-off最近20轮中位约
  `15.775 s/H48`（H24-equivalent `7.89 s`），仍未达目标；前5轮profile把主要墙定位到
  `post_physics_publish`。
- portable MuJoCo现役：commit `23c0f6c…`，namespace
  `fullmdp-a-h48-v2-mujoco-unblock-23c0f6c8-20260822`，GPU0，launcher/child=`2219700/2219718`；
  22:55 UTC durable ACK已到至少47且Reward/storage全finite。最近20轮wall中位约`9.634 s/H48`
  （H24-equivalent `4.82 s`），child cwd与全部运行输出均钉到run root，source clean。
- Isaac仍未有due；MuJoCo update45已首次出现`due=1/reveal=1/deferred=0`，随后27个phase-2 task row中
  R03 physically-valid为26且Reward0--9非零，证明mimic入口与balance自然重叠。contact/landing仍为0，
  击球/上台继续写`未测`。不能用早期`ACCEPT=0`停车；也不能在真实due出现后仍无reveal时继续盲等。
- 被替换的Isaac/MuJoCo分别精确停止于ACK631/3023，旧root、snapshot和admin-stop pre/post证据保留，
  completion均缺席且未伪造。下述`96f0ca69…`与`99405266…`段落是本次修复前的历史快照，已被以上两条
  fresh successor取代。

三条旧长跑均已停止，不可继续推进：

- Isaac `e8eef4fb…`止于 durable ACK（optimizer成功后已持久化的更新签收）`4603`，累计
  `452,591,616` transitions；
- Isaac `ddb1e7c4…`止于 durable ACK `3467`，累计 `340,918,272` transitions；
- portable MuJoCo r3 止于 durable ACK `10249`，终段 last-100 wall mean/median=
  `4.890/4.886 s/update`。

两条 Isaac 的退出因果仍是`未知`：现有日志没有足够证据把它归因于 OOM、外部停止或代码异常，
也不能因 `final_rc=0`把不完整 run 写成完成。MuJoCo r3 由真实 `EPA_HORIZON`（扩展多面体碰撞算法的
迭代深度上限）overflow fail-stop；
它同时消费了会在 IDLE 时随全局 step 漂移的旧 229-D observation，所以不允许 resume。

fresh successor已经从clean detached `96f0ca69887aba44c71983529d05e759e1a4cd2f`在Pod1真实发射：

- namespace=`fullmdp-a-h48-v2-96f0ca69-20260821`，run root=
  `/workspace/franco/runs/fullmdp-a-h48-v2-96f0ca69-20260821`；
- GPU2 UUID=`GPU-473a79f3-8736-6c7f-c3db-290c6be385b8`；发射前empty-app与nonblocking lock门通过，
  launcher PID=`2030437`持lock等待唯一child PID=`2030453`自然退出；
- exact argv为Full-A `4096×48×12500/save500`，fresh runtime site，无resume/retry/signal/`ACCEPT`门；
- 首个durable ACK为update `0`、`196,608` transitions，collection/learning/pre-ACK=
  `9.354775/0.284285/9.639704 s`；Reward20/storage finite，conservation/nonfinite fault均为0；
  `model_0.pt` SHA-256=`50ebc7c9…7b26`；
- 最近一次只读检查已见update `0..4`共5个连续durable ACK，child仍为`R`。这是运行态快照，不预测最终
  completion；source、namespace与PID只作为本次run身份，不为后续run复用。

Isaac successor也已从clean detached `9940526684a4ea068b08bf7a2627a6e07c1452f1`在Pod1真实发射：

- namespace=`fullmdp-a-h48-v2-isaac-99405266-20260822`，GPU0 UUID=
  `GPU-889b1712-8d89-0536-5c9e-e79aae30523d`，PID=PGID=`2095711`、Kit child=`2095727`；
- exact argv同为Full-A `4096×48×12500/save500`，fd16 runtime receipt精确为
  `trainer_runtime_attested_v2`，fd18 sealed RSL archive与GPU0 lock由live child继承；
- durable ACK已连续到`0..10`；已完整打印的10次wall范围=`16.02--22.52 s`、median=`18.445 s`，
  H24-equivalent median=`9.2225 s`。每个ACK均有`196,608/196,608` actual Reward finite，nonfinite与conservation
  fault均为0；早期`ACCEPT=0`不作停车门。

第一次Isaac real曾因launcher只复制单个USD、缺同目录sealed source bundle而在PPO前自然RC1；该root封存、
GPU/lock自然释放，未resume或复用。successor改为拒绝non-regular entry并复制完整61 MiB asset package，
训练侧enclosed-source reconstruction不放宽。

旧 run 的 `ACCEPT=0`只是课程 telemetry：当时 opportunity 仍停在 balance/mimic/readiness 阶段。
balance→mimic→entry→strike→landing 本来就可能需要很多 step；`ACCEPT>0`不是启动门、安全门或早期
学习成败门。业务 eligible denominator 为零时也不调整 Reward0--13。

## 2. 已采用的 successor 合同

### 2.1 PPO（近端策略优化）V2

FullMDP A/C 与 portable MuJoCo 统一使用 code-owned typed recipe：

- rollout horizon（每次策略更新、每个环境收集的control step数）`H=48`；
- `max_iterations=12500`、`save_interval=500`；
- `num_learning_epochs=5`、`num_mini_batches=8`；
- `gamma=.99`、GAE（广义优势估计）`lambda=.98`。

它保持旧 `H24/U25000/save1000/E5/MB4`的大致总 transition、minibatch size、optimizer step 和按
environment step 计的保存节奏，但改变 policy refresh、GAE 和每次 update 的优化分组，因此只允许
fresh launch。旧 H24 snapshot 不能 resume 到 V2。

H48 是学习算法取舍，不是性能修复或速度豁免。旧“约 6 秒”是 H24 尺度下要求大砍迭代时间的量级
信号；线性换算到 H48 约为 `12 s/update`，不是硬 Gate。性能必须同时报告：

- 原始 `wall_s/update`；
- `transitions/s`；
- `H24-equivalent = wall_s × 24 / H`。

旧 Isaac H24 稳态约 `22 s/update`；若吞吐不变，H48约为 `44 s/update`。这意味着仍需数量级明显的
算法/数据流升级，不能靠把 batch 翻倍掩盖吞吐不变。

### 2.2 semantic Observation V2

family A 冻结为 actor `203-D`、critic `219-D`。这是删冗余后的语义替换，不是把旧 229/399机械
扩成237/407：

- actor common `183-D`加入 table-relative root XYZ、continuous heading XY 与 heading-frame COM
  velocity，并保留重力/角速度、proprioception、last action、teacher、anchor 与 Motion phase；
- actor task tail `20-D`只给可部署观测的 delayed racket position/velocity/raw A/+Y normal residual、
  base-goal residual、三个 per-env countdown、learning phase 与 Motion-visible task mask；
- critic只追加 `16-D`未来因果训练事实：episode剩余时间、live ball position/velocity/spin、selected
  contact/net latch、双脚 support 与 cadence dwell；
- 删除 raw task45、owner fact blob、fault/age、Reward due/paid 等控制账本。

此前尽调没有否决 root state：历史 fixed-194 和 A211 已包含 table-relative root pose/velocity；遗漏发生
在 direct-lean 229-D 迁移。当前代码反例可构造 root translation、yaw 和 COM velocity不同但旧 actor
observation 相同的状态；这些量不能由关节 q/dq、base goal 或 torso anchor唯一恢复。因此本次增加的是
非冗余且在目标传感链可观测的状态，而不是“多给 policy 一切真值”。HITTER、SMaSH、BeyondMimic只提供
相对 base/anchor、history/base velocity 的设计方向，不能代替本地 alias 反例。

actor不读取 contact/support/spin/fault/reward ledger。部署侧仍需真实接通 IMU、table/root定位、
marker→COM 的因果速度估计、encoder/FK和planner tuple；这条 producer 尚未完成，所以 V2 当前仍是
simulation diagnostic，不是部署就绪或“单帧 Markov”声明。family C 将使用独立 `202/218`合同，不补零
凑成 A 的宽度。

R06（落点结果发布者）observation projection也同步瘦身：旧 broad projection复制81个 tensor，并可能从 legacy key plane
读取到全零 current flight；successor复用现有R06 owner，在owner内部按 canonical shot key八字段加
publication ordinal唯一选择 live INBOUND/OPEN row，只导出 `flight_slot + contact/net-crossed/net-clear`
四个 tensor。无匹配、EMPTY或SETTLED_RETAINED均返回无 live slot；slot只用于读取Physical行，不成为
新的业务identity。没有新增owner、receipt或Gate。

R07的support/dwell复用唯一真实post-physics plant read，不在Observation再次扫描全机器人。cold genesis
明确为zero；selected reset只把generation精确`+1`的行在当帧归零，same-generation peer仍严格对齐tick，
下一真实post-physics恢复。Phase-C1又把Motion broad observation view从34 tensor收成两个真实consumer的
10-field并集；publication只复制真实并集，validator保留窄consumer隔离clone。该结构债已在host闭合，但
不把静态payload减法写成Pod wall收益。

### 2.3 真安全边界与结构减法

继续保留能跨独立事实源失败的边界：nonfinite/overflow、真实contact和joint/table limit、selected-reset
generation、shot key/publication join、source/asset provenance、optimizer成功后的durable ACK，以及失败后的
sticky poison/fail-stop。

继续删除的不是这些边界，而是同一writer的digest/receipt互证、zero-callpoint gate、无事件仍写的journal、
每substep重验construction已固定的class/bound method、无人消费的counter和用zero policy表现作启动许可。
Phase-B已经物理删除zero-caller formal owner及专属适配层；下一步不为死接口补compatibility adapter。

### 2.4 portable Full-A runtime package binding

branch WIP已在fresh CLI import前把EPA48/RSL3两wheel绑定同一site；Pod1无CUDA actual import=
`19 passed in 4.33s`。durable identity只增加EPA mapping，RSL仍走既有process-local gate；wire为`3/4/3`，
legacy WAIT不绑定，checkpoint/resume authority仍false。完整合同与未授权边界只在
[portable Full-A实验§0](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md#epa48-fresh-runtime-binding-20260821)
维护；恢复与调用见[`setup_local_sync`](setup_local_sync.md#bind-the-exact-epa48--rsl-rl-312-site-for-portable-full-a)。

## 3. 唯一依赖顺序

每项必须独立提交、可单独回退；前一项闭合后才进入后一项。

1. **Observation V2原子闭合**：Isaac/MuJoCo、training hard contract、snapshot receipt与文档同时切到
   A203/219；两后端用独立native state正负扰动验证frame、符号、scale、mask和R06 live selection。
   V1只为明确的历史 WAIT consumer保留，不允许Full-A静默fallback。
2. **MuJoCo EPA48**：build chain与fresh dual-wheel actual import已在branch candidate闭合；tracked replay
   也已在Pod同卡完成stock24 mask256/contact0与fork48 zero-overflow/contact1+finite raw contact各10次，
   branch只保留fixed fixture+replay-only工具。H48 fixed-tape已在exact Pod同卡生成两份record：离散/
   reason/events与初态exact，连续repeat envelope已保存；五个自然business strata因due全DEFER仍`未测`。
   instrumented/ASan独立oracle继续阻塞physics promotion/transfer claim，但不再作为
   `diagnostic_unauthorized`长跑前的表现式准入门：fork只改project-pinned容量常量，runtime overflow/nonfinite
   fail-stop仍在，且真实Full-A 61-update已自然rc0。stock CPU同样硬编码24，不能充当golden；r3仍不resume。
3. **单一 ActionBallState**：一个device-resident mutable state唯一拥有phase/generation/shot/contact/
   outcome/fault；K-row候选只构造一次并sparse commit，只有真实transition写compact event delta；
   `K=0`/zero-live-flight成对跳过Physical/scene/R06/Epoch空事务，PPO boundary统一汇总。
   Phase-C0已先退役zero-business D05事务，Phase-C1已收窄Motion publication，Phase-C2已删除五个无人消费的
   Epoch整record返回clone，Phase-C2b把Observation cache移到整record clone之前，并把R07同一事务三份
   Epoch snapshot收成一份栈内快照；stale caller与首写前推进仍由零clone version/head检查拒绝。剩余dense
   Epoch/R06/Physical state与host sync仍须按真实profile继续收敛，不能把这些局部刀称为single-state完成。
4. **matched H48性能验收**：clean exact source、同卡、profiler-off，按zero/mixed/active strata对比；
   同时验证fixed tape RNG/highwater/reason/done/reset/Reward20/203/219 parity。报告原始wall、transitions/s和
   H24-equivalent；首墙转移后重新profile，不继续堆零碎clone patch。MuJoCo的真实生产热路rate入口已固定
   61 update；Isaac只增加同样`10+50+1`的code-owned diagnostic budget，不开放任意短跑覆盖12500配方。
   MuJoCo actual p50/p90=`9.448/9.661 s/update`、throughput=`20,779.64/s`、H24-equivalent p50=`4.724 s`，
   因而本后端不继续堆微优化；Isaac同卡pre/post Phase-C wall仍`未测`。
5. **fresh namespace启动与训练**：clock-fixed、EPA-fixed的portable MuJoCo V2与Isaac V2均使用fresh
   namespace；matched性能是并行诊断，不是Isaac启动前置。两条都不复用旧runtime site或snapshot，不插
   `ACCEPT>0`门。MuJoCo caller必须传
   [`--mujoco-warp-runtime-site`](../DEFINITIONS.md#mujoco-fullmdp-longrun-flags)指向一个尚不存在的绝对路径；
   future launcher先从clean Git truth取得并传`source_commit`，binder不得自报。短验只回答构造、有限性和
   真实调用点，随后同一进程继续训练并按balance→mimic→entry→strike→landing分阶段报告分母。
   branch候选one-shot launcher只负责clean Git、fresh root、GPU UUID/空卡/lock与固定H48 argv，并等待child
   自然退出；它不监控、不重试、不发signal，也不以`ACCEPT>0`作为启动或停止条件。Pod1 clean detached
   `2e4279ba` dry-run已PASS且未建root、未查GPU、未改lock；随后clean detached `96f0ca69`已按上述fresh
   identity真实发射并取得update `0..4`连续durable ACK。61-update实际rate与fixed-tape关闭发射前的有限
   构造/吞吐证据；运行现在只允许自然推进，`ACCEPT=0`及五strata未出现不作为表现门。Isaac V2的同卡
   pre/post Phase-C测量仍在后续，不能由MuJoCo代签。Isaac one-shot候选已复用现有Kit boot owner并通过
   host双launcher回归`19 passed`；exact Pod dry-run及GPU0 fresh real均已闭合，source=`99405266…`、
   PID=PGID=`2095711`，durable ACK已有`0..10`连续且Reward finite/fault0。当前Isaac前10个完整H48 wall
   median=`18.445 s`、H24-equivalent median=`9.2225 s`；这是自然长跑中的早期观测，不是matched稳态
   测量或6秒GO。

## 4. 当前完成条件

- Observation：A203/219在两个backend由独立producer生成，static scale、heading退化、task mask、R06
  live selection、cold genesis、row-wise selected reset和critic-only边界有可区分反例；snapshot receipt
  绑定同一training contract SHA。
- Physics：EPA24-fail/48-finite deterministic fixture、GPU复测、独立oracle和overflow fail-stop全部闭合。
- Performance：exact Pod的matched H48数据证明transitions/s显著提升；host测试或删行数不能代签。
- Training：fresh run有连续durable ACK、finite Reward/observation和可解释的per-stage/per-action/per-side
  denominator；缺失格写`未测`，不以总均值覆盖零格。
- Authority：所有运行保持`diagnostic_unauthorized`；未完成restore/physics/deployment Gate前，不授权
  resume、promotion、export、部署或真机安全结论。

热路径证据和结构设计详见
[FullMDP hot-path实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md)；
MuJoCo fork状态见[G06](../gates/G06_isaac_to_mujoco.md)，runtime资产与调用见
[`setup_local_sync`](setup_local_sync.md#bind-the-exact-epa48--rsl-rl-312-site-for-portable-full-a)，
其余训练操作边界见[训练工序](run_training.md)。
