# ActionBall 双后端长跑：当前执行 TODO

> 状态：`ACTIVE-dual-backend-H48-longruns / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-23
>
> `origin/main:docs/NOW.md` 是全项目唯一优先级权威。本页只维护
> [FullMDP](../DEFINITIONS.md)（完整球路、击球、落点与恢复状态机）单动作双后端
> successor 的依赖顺序、运行事实和完成条件，不维护竞争性的优先级队列。旧的单动作执行页已转为
> [只读历史账](action_ball_single_action_dual_backend_todo_20260817.md)。

## HISTORICAL / SUPERSEDED — 0. 2026-08-22 学习阻塞修复与 fresh 重启 TODO

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

## HISTORICAL / SUPERSEDED — 0.1 2026-08-22 active-path简化与第二次fresh重启

本节承接上节，不改变项目优先级。现役Isaac在durable ACK 257之后由匿名CUDA
`_assert_async`毒化context，最终在下一次PhysX scene write显错并卡住；它已按精确PID/start-ticks保全现场并
停止。MuJoCo继续只读运行。第二次successor必须先闭合下列顺序，不能hot-patch旧source或复用namespace。

- [x] 保全Isaac最后完整ACK、run log/WAL/checkpoint和进程身份；确认context已不可继续训练后，只对精确
  child执行TERM，TERM两次无效才KILL，并验证child/leader/GPU compute app均absent。
- [x] 把ActionBall active-path的匿名异步assert改为单一、具名、device-local fault状态：无效行不得写PhysX，
  在已有control/PPO边界统一同步并报告具体reason；保留真正的finite/identity/joint/contact安全，不保留
  同一writer按构造恒真的echo检查。
- [x] 让Isaac与MuJoCo共用同一纯tensor `q_des` guard语义；finite fallback、soft/hard inset、state-dependent
  brake和终止位必须有反例测试，不能继续把MuJoCo的简单hard clamp写成已对齐。
- [x] 用构造期确定性几何证明slot0的teacher strike pose与question ball/contact tuple闭合；该证明回答
  “mimic成功后是否已经能碰球”，不新增actor observation，也不用随机rollout冒充几何证明。
- [x] 收敛Isaac post-physics同一调用栈内的重复full-grid clone/同写者校验；只删除同步consumer完成前不会
  被修改的副本，并用alias/mutation反例与fixed-tape检查保护真实跨owner边界。
- [x] launcher加入有界profiler窗口和GPU-local NUMA放置的显式参数；profiler只作归因，profiler-off同卡
  `10 warmup + 50 measured`才记wall/throughput，且必须同时报告raw H48和H24-equivalent。
- [ ] host聚焦回归、exact Pod隔离测试、Reward20/actor203/critic219/done/reset/reason/RNG/fault parity全部通过；
  学习语义改变一律fresh lineage。
- [ ] successor就绪后保全仍在运行的MuJoCo最终证据并精确停止；两个backend用新clean detached source、
  fresh run root和fresh namespace同时重启，至少取得连续ACK `0..4`、finite/fault0后才宣布切换完成。
- [ ] 按balance→mimic→entry→strike→landing继续报告逐阶段分母：上一阶段基本成形时下一阶段应已有非零
  exposure；contact/landing为零就明确写零或`未测`，不靠aggregate return掩盖。

本轮冻结的取舍：采用active-path状态收敛、共享q_des guard、确定性几何闭合与有界诊断；延后resume、
CUDA Graph、solver iteration和物理步长变化；拒绝先写C++、增加actor observation、增加same-writer receipt，
以及用Reward调权掩盖zero denominator。100-step RK4 fused CUDA/Triton路径已经存在，本轮只验证真实绑定，
不重复实现。

## HISTORICAL / SUPERSEDED — 0.2 2026-08-22 `aa42418b`性能闭合与监测

这是上节的实测收口，不新增优先级。219-D critic的no-key support/dwell语义已经变化，因此旧`661ff84b`
checkpoint只保留历史，不是本轮exact-resume父本。

- [x] 用细分profile确认旧主墙是no-key阶段读取Isaac ContactSensor两脚net force，约`7.41--8.03 s/H48`；
  chronology snapshot/store各只有毫秒量级。
- [x] no-key路径删除contact读取，只保留独立source/reset-generation/Motion cadence chronology；
  critic `[216:219]`保持宽度但按N/A填零。keyed R07仍读取真实contact并计算support/slip/recovery。
- [x] 删除同writer自证：R07 bundle不再重验已绑定ActionEpoch `snapshot_idle_*` getter identity；LeanRuntime
  production component callpoint binding与callee owner核验仍保留。D05测试不再手造Motion true输入再断言
  ACCEPT；独立consumer chronology、catalog tick295/cadence293和真实keyed反例仍保留。
- [x] exact Pod分进程回归=`304 passed, 6 skipped`；no-key contact `.data` sentinel、keyed recovery、
  Epoch、profiler、Motion bridge、Physical hot lane、env install与D05 transaction均覆盖。
- [x] 同卡H48 profile后`r07_idle_support_read=0 calls`；profiler-off匹配5轮中位`6.346 s/H48`，
  相对旧`14.194 s/H48`约降55%，且fault/CENSOR/nonfinite/conservation全0。
- [x] 新Isaac连续ACK后精确停止旧Isaac；补齐ignored runtime三文件manifest后启动新MuJoCo，连续ACK后再
  精确停止旧MuJoCo。最终两条active run都绑定clean source `aa42418b…`且是fresh lineage。
- [ ] 继续只读监测balance→action mimic→launch/contact→landing。两端均已验证row活过tick295时task exposure
  立即打开；当前零分母只剩launch/contact/landing并记`未测`。只有action mimic已基本形成而后续仍无launch/
  contact exposure，或hit已基本形成而landing仍无exposure，才判对应课程交接故障。
- [ ] MuJoCo早期qdes forbidden已在update26基本归零，随后robot-table与fall占比继续波动；Isaac仍以fall为主。
  两者都属于policy可学习行为，先按原因/分母观察，不绕过真实joint/table termination，也不增加“安全”门。

## HISTORICAL / SUPERSEDED — 0.3 2026-08-23 V3审计与fresh V4

本节只记录当前branch-scoped successor的依赖顺序，不改变`origin/main:docs/NOW.md`的统一队列。
[`fullmdp-a-h48-v4-*`](../DEFINITIONS.md#fullmdp-correction-lineage-v4)表示第四批fresh实现纠错lineage，
仍消费PPO V3配方；名称中的V4不是新PPO算法、阶段或promotion等级。现役V3 source/run保持只读，所有修复只能
进入clean detached successor、新namespace和新run root。

- [x] 保存现役V3双后端的进程身份、continuous ACK、finite/fault、H48 wall、episode与
  `survive→reveal/ACCEPT→playback→launch→R03→contact→landing`逐阶段分母；零格不得用aggregate return
  或几何证明替代。
- [x] 重新对账本页既有TODO与
  [`EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802`](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)：
  hidden阶段reset-ready joint/body reference、reveal后原子切measured frame0与bridge、以及teacher/contact
  几何闭合都已经是已采用设计；当前零launch/contact本身不是“又有一个reference偏移”的证据。
- [x] 复核203-D actor / 219-D critic合同：actor已有deploy-observable robot、teacher、task geometry、phase与
  clocks；本轮没有构造出“同一现有actor状态却需要不同动作”的alias反例，故不增加冗余、不可部署观测。
- [x] 按`HANDOFF_TO_CODEX_20260808.md`§3裁决实现问题：R03/physical/R07只付fresh source step，R06只在
  settlement后完整付一次；missed launch不得catch-up；第五个episode内不可完成的shot从共享cadence删除；
  invalid contact退休shot而不伪造Gym reset；q-des真实终止语义双后端一致。
- [x] 把portable MuJoCo missed launch变成具名pre-optimizer ledger fault；完整rollout storage在同一次已有
  host reduction中检查policy/critic observation、action/value/logprob/mu/sigma/reward/return/advantage有限，
  并检查done严格二值，不新增逐step同步。
- [x] 分文件完成host聚焦回归、consumer/launcher、`py_compile`与`git diff --check`；混合import-alias造成的
  exact-type假失败须分进程验证，不能为让测试绿而放宽production type identity。
- [x] commit后在exact Pod clean detached checkout恢复固定SHA的EPA48/RSL3 ignored资产，运行隔离回归与两个
  one-shot dry-run；Isaac必须使用目标GPU的NUMA-local CPU set，并由ready child的`/proc/.../fdinfo`证明
  继承同一GPU lifetime flock。
- [x] 只在live lane owner/queue再次对账后，以fresh V4分别替换现役V3；先取得连续durable ACK、完整finite/
  fault0、run-owned cache与source clean，再宣布切换完成。空闲GPU不等于已获授权，不借用未协调lane。
- [ ] 继续按自然课程报告：balance基本成功前允许下游样本稀少；但一旦mimic基本形成，launch/contact必须已有
  非零分母；一旦hit基本形成，policy-settled R06/landing必须已有非零分母，否则才判对应交接故障。

结构裁决也冻结在本轮：真正独立的finite、full-key/generation、plant terminal、scene writer、optimizer与GPU
lock边界继续fail-stop；task成功、每update必须出事件、R07 readiness、同一writer回声/hash、late catch-up都不是
安全Gate。当前Physical/Epoch/R06等owner文件已经过大且存在多份sticky fact/payment副本与import/type耦合，
这确实提高了开发和审计成本；但V4前只做已经有因果证据的减法，不先造巨型state、C++重写或新owner。
V4取得运行证据后，下一轮结构目标是“一条lifecycle spine + 两个backend StepFeatures adapter + 纯fact/payment
kernels + 一个telemetry funnel”，每次删除self-proof都必须同时保留独立mutation反例和fixed-tape parity。

## 0.4 2026-08-23 V4最终冻结与V5第一性原理自查

本节是本页唯一现役执行合同，不新增`origin/main:docs/NOW.md`之外的优先级。V4两条run均已停止并最终冻结；
V5已从clean `39f9481950a660e198dedac7fd402806d648906b`在Pod1以fresh namespace启动Isaac与MuJoCo，
仍只是`diagnostic_unauthorized` implementation lineage，不是新PPO算法、课程Stage、checkpoint兼容或
晋级等级。学习配方继续是H48、GAE
`lambda=.98`、E5/MB8、entropy 0、fresh learned `log_std`/init sigma `.02`。H48是学习取舍；旧“6秒”只表示
需要大幅算法/数据流提速的量级方向，不是形式Gate。

**V4最终冻结（2026-08-23；本段替换临时前缀，不再追加运行流水）：**

- 两条均绑定clean source `9e26afd3342e1da8643b225c987d4a3c91a3ff2f`；exact Pod隔离回归为
  `589 passed, 12 skipped`（skip均为需真实GPU的用例），Isaac与MuJoCo one-shot dry-run都通过。Isaac
  namespace为`fullmdp-a-h48-v4-isaac-correctness-9e26afd3-20260823`，portable MuJoCo namespace为
  `fullmdp-a-h48-v4-mujoco-correctness-9e26afd3-20260823`；它们只是fresh运行身份，均保持
  `diagnostic_unauthorized=true`，不是算法版本或promotion标签。
- Isaac最终连续durable ACK=`0..748`（749轮、`147,259,392` transitions）；update749在optimizer前报
  `epoch drain decoded a terminal overflow`，无ACK749且进程已退出。V4只有generic bool，不能唯一恢复
  具体row/cause或把它归成EPA容量故障。recent50 episode均长/return=`179.802/12.157`，wall mean/median=
  `9.128/9.100 s/H48`；D05 accepted/due=`270/300`、playback=`61`、launch=`1`、contact/R03=`0/0`。
  R06一次settlement没有contact/net/cross/table/common，R07/retire仍为0。
- portable MuJoCo最终连续durable ACK=`0..4798`（4799轮、`943,521,792` transitions），evidence在精确
  PID/startticks/PGID/cwd/source/namespace/GPU核验后冻结，旧进程随后TERM并确认退出；未自然跑满12500，
  因而没有`completion.json`。recent20 wall mean/median=`14.146/13.994 s/H48`；Reward/storage finite，
  命名fault、nonfinite和conservation为0。累计public due/reveal/defer=`1,637,789/1,637,789/0`、
  launch=`527,957`、R03 physically-valid=`145,814`，racket/selected contact与landing均为0。完成episode
  `3,376,589`，tilt/table/base-low terminal bit=`1,521,819/1,860,583/383`；bit可重叠，不能相加伪装
  独占分类。旧schema-4字段
  `racket_contact_eligible`只是launch的同写者别名，不再作为接触机会或分母；当前缺的是独立durable
  playback denominator。actor已有Motion phase，但playback成功分母继续`未测`，不得由reveal代写。
- V4只证明survival-to-due 295可达且mimic/hit输入会自然打开，不证明balance已基本成形；它同时显示旧实现的
  mimic→hit交接不符合预期。由于下述reference、typed identity与transition chronology实现错误已污染这条
  lineage，该结果不能裁决corrected V5的课程设计。V5已经fresh训练；启动验收时两个后端都还没有due/
  action分母，随后MuJoCo已自然出现public due与launch，见下述阶段里程碑。mimic成功的独立分母仍`未测`；
  Mu hit/contact已有`0/6 launch` diagnostic negative，Isaac contact与两端landing/recovery仍为`未测`。

**V5 live-source Pod、fresh启动与学习轨迹审计（原位替换，不作live流水账）：**

- 两条live run的source为clean/pushed `39f9481950a660e198dedac7fd402806d648906b`。Pod broad CPU/ABI矩阵=
  `792 passed, 57 skipped, 0 failed`；另以无`PYTHONPATH`的fresh进程复跑plant/runtime/runner/consumer=
  `77 passed, 0 skipped`。两组有重复验证目的，不相加伪装unique总数。Mu真实GPU direct五节点与真实
  RSL H48一轮=`5+1 passed`；Isaac CUDA projection/selected-rubber/runner-drain=`32 passed`，其中历史命名的
  runner-drain只签CUDA/RSL callpoint，不冒充Isaac集成；当前真实Kit fresh训练提供更强的端到端路径。
  双launcher dry-run均在建root前通过，且两个real root此前不存在。
- Isaac fresh namespace=`fullmdp-a-h48-v5-isaac-chronology-39f94819-20260823T144237Z`，物理GPU1。
  启动验收前缀连续durable ACK=`0..63`（`12,582,912` transitions），每轮`196,608=4096×48`；最近五个
  完整console wall=`7.78/7.47/8.17/6.79/8.19 s`，但不是matched-strata稳态速度。该前缀139,264个episode
  全因tilt终止，全部D05/Motion/shot事件为0，Reward finite、attributed fault与conservation均为0。
  后续只读到ACK97时，10-update mean episode length/return从首窗`87.606/7.052`升到末窗
  `97.776/8.705`；这说明balance开始学习，不等于balance基本成功，且仍没有due。
- MuJoCo fresh namespace=`fullmdp-a-h48-v5-mujoco-chronology-39f94819-20260823T144237Z`，物理GPU0。
  启动验收前缀连续schema-6 ACK=`0..8`（`1,769,472` transitions）；post-warm recent5 wall
  mean/median=`8.727/8.777 s/H48`，加权`22,528.7 transitions/s`。该前缀16,378个episode均为
  `robot_hit_table`，mean length=`105.21 tick`；所有scheduled/public due、launch、R03、contact、outcome、
  landing、recovery与retire均为0，Reward/storage/domain、Mu四项fact-integrity与conservation全绿，
  std=`.020000→.019945`。这同样只是balance早期，不以零shot分母裁决击球实现。
- **固定学习轨迹审计：**不逐ACK追写，当前只认Isaac `0..450`（`88,670,208` transitions）与MuJoCo
  `0..385`（`75,890,688` transitions）两个冻结前缀。Isaac
  已从`201..220`的局部低谷`120.451/10.792`恢复；最新`431..450`共`23,081`个episode，mean
  length/return=`170.249/15.625`，超过先前`170..189`局部峰值`160.279/14.398`。该窗全部因tilt结束，
  qdes/table/low/timeout、fault、nonfinite与conservation均为0；这证明此前回退是可恢复的早期PPO非单调性，
  仍不等于balance基本成功。累计due/public=`20/20`、ACCEPT/reject=`17/3`，已有`2`次真实playback，
  launch及其下游仍为0。
  MuJoCo最新`366..385`共`21,033`个episode，mean length/return=`186.693/17.771`；terminal为
  tilt/table=`11,314/9,719`，base-low/qdes/timeout均0。累计scheduled/public/terminal-overlap=
  `307/297/10`、defer=0、natural launch/missed=`6/0`、R03 present/physically-valid=`1/1`，contact、
  outcome、landing与recovery仍为0，fault/nonfinite/conservation全0。slot0的tick45 playback、tick76 launch
  已由自然run走通launch边界；Mu racket/selected contact均为`0/6 launch`，是小样本diagnostic negative，
  不再写`未测`。Isaac因launch=0，contact仍`未测`；两端selected contact=0，故landing仍`未测`。现在的
  未闭合点是独立mimic/playback成功与Mu launch→contact，不是Stage或launch断链。
- 两条run均为fresh、无resume/hot-patch/namespace复用，且仍`diagnostic_unauthorized=true`。Isaac早期wall
  已出现`6.79 s`单点，Mu早期也比旧V4末段快，但状态未匹配，不能正式归因算法提速；约6秒方向尚未达到。
  与上述学习前缀对齐的recent20为Isaac stdout辅助total mean/median=`9.488/9.465 s`、Mu durable wall
  mean/median=`9.235/9.223 s`，进一步证明单个6.79秒不能代称稳态。正式rate window、独立playback分母、自然
  mimic→hit→landing、12500 completion与physics/transfer parity继续未闭合。

**对齐既有TODO与MuJoCo native readiness后的裁决：** reference只有三个权威切点：
`fresh_bound && policy_opportunities_created==0`的**首次ACCEPT前** balance行使用同一reset-ready
joint/body tuple；D05在post-transition ACCEPT边界原子安装selected measured action frame0 joint/body并把它
放进返回的`obs_{t+1}`，第一笔task-conditioned action/reward发生在下一transition；后续recovery/ready参考是
上一个completed action frame0，不是回到reset-ready。这正是V5修复的旧Isaac偏移：旧fresh路径用
generic `in_hold`把joint留在default，却让body读frame0。它是与既有
合同相反的实现回归，不是需要新offset、新课程或新actor Observation的证据。current
`mount_normal_sign=+1` slot0的selected-face球心位置几何仍闭合，稀少launch或`0/2`真实失败
也不构成新位置offset证据。

actor 203-D / critic 219-D已包含fixed-center family A所需的deployment-intended robot、teacher、task
geometry与clock；真实table/root/velocity producer仍由G07闭合，也没有构造出同观测却需不同动作的
alias反例。现役`history_length=0`；对照系统中的`history=8`只作deferred候选，在无alias反例前不接入。
同样不把contact/support/fault/reward账本、未来oracle或其他冗余/不可部署量加入actor。N73
negative-face normal另有现有合同实现错，V5以raw +Y mimic/R03与signed physical face分离修复，
不扩Observation。

phase也不再混称：**Epoch phase**是shot业务carry-state（`IDLE→REVEAL→LAUNCH→OUTCOME→RETIRED`），
回答“这一球的事实账走到哪”；**Motion phase**是teacher/reference的播放与可见性状态，回答
“policy此刻应跟哪段动作”。合法launch可在同tick进入outcome；旧shot的outcome/R06/R07/retire也可在同一
surviving边界后被新reveal原子替换，因此final phase不能单独解释该transition的全部事件。outcome与natural
recovery必须互斥，DEFER只允许结算后仍busy的row。actor的两个5-way one-hot分别表达Epoch与Motion，
`task_valid`由Motion当前可见性决定，不用Epoch中保留到RETIRED的payload-valid代替。

**当前采用/延后/拒绝（其余历史表不再覆盖本表）：**

| 裁决 | 当前合同 | 为什么 |
| --- | --- | --- |
| 采用 | PPO V3：H48、GAE `lambda=.98`、E5/MB8、entropy 0、fresh sigma `.02` | H48是学习取舍；性能另按真实wall/throughput验收 |
| 保持 | semantic actor/critic `203/219` | 203-D已含建议的8维root状态：`base_position_table(3)+base_heading_table_xy(2)+base_com_lin_vel_heading(3)`；没有证据再造237-D |
| 延后 | `history_length=0`保持，`history=8`仅作候选 | 尚无same-observation/different-action alias反例；先不扩大部署与normalizer合同 |
| 采用 | 修正首次ACCEPT前/ACCEPT/recovery三段reference selector | ACCEPT在post-transition边界安装frame0到`obs_{t+1}`；旧Isaac selector与此相反 |
| 拒绝 | 新offset、新stage、新balance Reward或用R07 ready授权task | V4只证明survival-to-due与下游曝光可达；尚无因果证据要新增这些机制，也不等于balance或安全存活毕业 |
| 采用 | scheduled/public due分账，DEFER只表示结算后仍busy | 同tickterminal不能假报actor看见过reveal；自然RETIRED可立即ACCEPT |
| 采用 | Epoch/Motion解耦、typed launch identity与具名row fault | launch同tickoutcome与retire后同边界reveal合法；只拦真实跨owner错配 |
| 保持 | finite、pure timeout、真实plant terminal、full-key、optimizer与durable WAL/ACK | 这些保护可独立击穿的可信事实；task成功、R07 ready、stdout显示与same-writer echo不是Gate |

**V5 adopt / defer / reject（只写现态，不写实现流水）：**

- **Adopt — 两个可归因的性能方向：**MuJoCo合并base metric与Full-A的contact-buffer遍历；Isaac热consumer
  读取ActionEpoch窄投影而不复制dense `current()`。保持H48，最终源码须用
  fixed-tape/RNG/reason/counter/safety parity与profiler-off matched-strata wall验收。理论调用数、LOC、
  host microbenchmark与旧Pod结果都不代签V5速度。冻结到Mu ACK385的recent20把
  collection/learning/wall mean分成`9.056/0.177/9.235 s`，collection约占`98%`；同期Isaac ACK450
  对齐的stdout辅助recent20 total/collection/learning mean为`9.488/8.625/0.862 s`，collection占`90.9%`。
  因此下一轮不能靠改日志、减少Gate或微调PPO声称
  达到6秒，必须先在matched state量出physics/step/solver热墙，再对该墙做算法或kernel级减法。
- **Adopt — 单一事实消费链：**Isaac R03/Physical/R06 fault先进入共享ActionEpoch owner的36项
  `row_fault_bits`，MuJoCo只保留R03 nonfinite、R06 source-invalid、R07 sequence、R07 nonfinite四项
  per-transition cause；两套编号不混计，且只进各自既有唯一pre-optimizer drain。direct Physical/R06只认
  typed 8-field key/publication，由Physical唯一推进ordinal与previous/current ball centre；legacy plane隔离，
  不造假digest。Motion exact-resume保留reset-ready/pending状态与三段reference；negative-face把raw +Y
  teacher/R03 normal与signed physical face分开。这些修复不增加Observation、owner或业务Gate。
- **Adopt — wire只记录可界定的事实：**MuJoCo update/evidence固定schema6、completion保持schema5、consumer
  summary为schema5。`scheduled_due_rows`记录schedule命中，`due_terminal_overlap_rows`记录actor不可见的
  due+terminal，`reveal_due_rows`只记录surviving public due；删除与launch按构造相等的
  `racket_contact_eligible_rows`。fresh-prefix检查`launch≤reveal`、racket/selected contact链，以及
  R03/outcome/landing/retire各自的因果上界；launch同tick outcome合法。R03 exact-strike与contact属于不同
  clock，`selected_contact/R03-valid`只报描述性ratio，正式contact rate以launch为分母。
  `r06_common_per_eligible`才是closed task-landing成功率；`opponent_landing_per_crossing`只是crossing
  条件比例，不能代称总成功率，也不新增event或Gate。`invalid_contact + Gym done`只reset、不retire；
  aggregate consumer对未知overlap只验可证明上下界。当前`business_chain_complete`仍是producer逐row检查后的
  attestation加consumer边际聚合一致性，不是独立same-env/same-epoch重放；晋级前须增加keyed carry-state重算或
  可重放trace，当前只能称`producer-attested + aggregate-consistent`。
- **Adopt — runtime/authority边界：**单一path-free `runtime_stack`绑定EPA48、RSL3.1.2与MJLab1.5.3；
  正交`source_plant/runtime_attach`绑定base receipt、actual geometry/clock/capacity/owner-frame与run-owned
  augmented `runtime.mjb`，consumer在ACK前独立复验。launcher把已核GPU UUID与同一flock
  open-file-description交给唯一child。optimizer→WAL/fsync→owner ACK→EPOCH_ACK/fsync才是authority。live
  `39f94819`的stdout为regular file且没有失败，但审计发现snapshot/completion后的两处裸print在闭管时会把已
  durable状态伪装成进程失败；clean/pushed post-launch successor
  `a3c528f1b4c9b0a60f5cd3aeec28a11e990044b3`已改成best-effort structured warning并由closed-pipe反例覆盖，
  exact Pod fresh checkout全文件=`52 passed, 1 skipped`（skip为显式real-GPU节点）。该修复不改学习语义，也不
  hot-patch或重启live。
- **Preserve：**backend physics separation、独立measurement/plant verification、finite与真实plant terminal、
  full-key/generation、optimizer成功边界、durable WAL/ACK与失败后sticky fail-stop。这些保护可被独立反例
  击穿的可信事实，不因追求简洁或速度删除。
- **Defer：**per-ordinal Reward snapshots、重复D2H、canonical/flat双import与多层stamp/receipt echo先等Pod
  matched-state profile，再逐项抽取并保持parity。CUDA Graph、solver iteration、物理步长、resume与actor
  Observation/history扩张也分别延后，不与当前P0捆绑。
- **Reject：**giant rewrite、新mega-state/C++先行、强行统一backend physics、live hot-patch/resume/namespace
  复用、same-writer receipt/hash/echo自证、无人消费的Gate，以及用task成功、R07 ready、stdout或旧
  microbenchmark冒充安全/速度证据。长期只增量收敛到
  `PlantFacts → ActionBallState transition → StepTelemetry`，每次抽取都做fixed-tape parity。

### V5执行清单（本页唯一现役局部步骤；不改变`origin/main:docs/NOW.md`）

- [x] 代码合同已采用：三段reference、post-transition due结算、launch/outcome/recovery相位关系、Isaac 36项
  ActionEpoch fault、Mu四项fact-integrity、pure timeout/table reason与schema `6/5/5`。stdout在live 39f不是
  authority；下一source `a3c528f1…`另关闭了post-durable裸print的假失败路径。
- [x] 结构/性能边界已采用：single runtime stack、run-owned augmented `runtime.mjb`、contact census减法、Epoch
  窄投影、selected-rubber row-neutral fault、Motion exact-resume与legacy Reward lazy import；不新增业务Gate。
- [x] 最终代码逐fresh process完成受影响host矩阵、`py_compile`与`git diff --check`；重叠suite按各自矩阵
  报告而不相加，skip只标明确CUDA/Isaac/superseded边界，不能算green。
- [x] live训练使用已push clean commit `39f9481950a660e198dedac7fd402806d648906b`；post-launch只新增
  stdout nonauthority修复`a3c528f1…`，不把它伪装成当前live source或据此迁移checkpoint。
- [x] exact Pod使用独立`--no-hardlinks` clean checkout恢复固定SHA资产。Pod root overlay/`/tmp`容量有限，所有
  HOME/TMP/XDG/CUDA/Warp/pycache/pytest basetemp显式落到`/workspace`；逐文件fresh-process回归后再跑双launcher
  dry-run、Mu真实GPU direct/RSL H48一轮、Isaac projection/selected-rubber、fixed-tape和1/5 ACK consumer；
  broad与clean-runtime/GPU矩阵如上；此前任何旧commit结果都不代签最终源码。跨后端formal fixed-tape/physics
  parity仍是下项，不由这些单后端测试代签。
- [x] successor exact ready且live queue/lock重新对账后，V4 MuJoCo按精确进程身份收口；fresh Isaac/MuJoCo
  均已连续ACK `0..4`且finite/fault0，双后端切换完成。
- [ ] 继续按balance→mimic→hit→landing报告学习：上一阶段基本成形时下一阶段必须已有非零自然分母；zero
  denominator写`未测`。Isaac已从显著回退恢复并创新高，累计出现2次playback；Mu已自然出现6次launch与
  1次R03-valid。两者的balance仍未基本成功，继续到预注册窗口而不因早期非单调性改配方。下一fresh source
  只补playback/open-cohort与optimizer/KL等纯telemetry，不新增安全Gate；Mu contact当前为diagnostic `0/6`，
  Isaac contact与两端landing为`未测`；独立playback成功、recovery与12500 completion仍未闭合。
- [ ] 用next-source测试替换Isaac的fixture/self-echo证据：真实执行wrapped `alg.update()`并核
  fault→optimizer→WAL/ACK顺序；同batch覆盖pure-timeout/timeout+plant/plant-only；用两个同时eligible row核
  坏row隔离与健康peer；逐owner做foreign-bit拒绝矩阵。退役无production callpoint的direct-Reward与永久skip
  测试不得再计现役green。这些是审计债，不是当前live的新增安全Gate。
- [ ] 在不打断两条fresh长跑的前提下，后续以独立rate diagnostic或自然matched strata核正式50-update wall、
  p50/p90、throughput和H24-equivalent；再按profile决定下一项算法/数据流减法。单点低于6秒不算达到方向目标。

## 1. 历史运行事实（current只认§0.4）

### HISTORICAL / SUPERSEDED — 2026-08-22 `aa42418b`两条fresh lineage

下列namespace都是[仅全新训练、禁止续跑](../DEFINITIONS.md#fresh-only-no-resume)的一次性运行身份；名称中的
backend/commit用于回答“哪套源码在哪个后端训练”，不是算法版本或可复用checkpoint标签。

- Isaac：namespace `fullmdp-a-h48-v2-isaac-idle-zero-aa42418b-20260822`，物理GPU2，launcher/child=
  `2423802/2423818`。ACK176只读刷新中recent10 collection中位`8.238 s/H48`，Reward finite且fault/CENSOR/
  nonfinite/conservation均0；episode均长first10→recent10约`87.73→148.93 tick`。reset-ready imitation和
  survival均有上升，但balance/action mimic尚未成功。
- portable MuJoCo：namespace `fullmdp-a-h48-v2-mujoco-idle-zero-aa42418b-r3-20260822`，物理GPU1，
  launcher/child=`2426696/2426711`。update118只读刷新中recent10 collection中位`9.585 s/H48`，storage/
  Reward finite且全部lifecycle/conservation fault为0；episode均长从first10 `67.32`升到recent10
  `151.17 tick`。早期qdes forbidden从近全episode降为recent10仅1次；fall/table比例仍波动，balance未成功。
- 两端都严格是fresh、`diagnostic_unauthorized=true`；219 width只保留ABI，不给旧idle-foot-bit snapshot
  exact-resume语义。性能数字不授权physics promotion/export/deploy。
- 最新阶段穿越：Isaac累计`due/selected/construction/key=2/2/2/2`且CENSOR/fault=0；MuJoCo累计
  `due/reveal=15/15`、deferred=0，R03 present/physically-valid=`174/174`。说明上一阶段一有row活过tick295，
  task/action-mimic入口立即开放。Isaac仍未playback；两端launch/contact/landing仍为0分母，击球与上台
  继续`未测`。

### HISTORICAL / SUPERSEDED — 2026-08-22课程解阻后的上一批successor

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

### 2.1 PPO（近端策略优化）V3

FullMDP A/C 与 portable MuJoCo 统一使用 code-owned typed recipe：

- rollout horizon（每次策略更新、每个环境收集的control step数）`H=48`；
- `max_iterations=12500`、`save_interval=500`；
- `num_learning_epochs=5`、`num_mini_batches=8`；
- `gamma=.99`、GAE（广义优势估计）`lambda=.98`。
- `entropy_coef=0`，保留learned `log_std`、fresh init sigma `.02`与adaptive KL；旧V2永久
  `entropy=.01`只作std无界上升的历史反例，不得resume。

它保持旧 `H24/U25000/save1000/E5/MB4`的大致总 transition、minibatch size、optimizer step 和按
environment step 计的保存节奏，但改变 policy refresh、GAE 和每次 update 的优化分组，因此只允许
fresh launch。旧 H24或PPO V2 snapshot不能resume到V3。

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

V5 branch candidate用唯一path-free `runtime_stack` v1原子替换旧EPA-only
`mujoco_warp_runtime`，不保留双wire。它同时绑定EPA48 exact wheel/runtime、`rsl-rl-lib==3.1.2` wheel和
`mjlab==1.5.3`选定193文件/`1,399,177` bytes/tree digest。runner在Torch/MJLab import前cold verify，env
构造后核所有loaded `mjlab.*` module；consumer在读run-owned `runtime.mjb`和ACK前独立执行同一验证。
V5 evidence/completion/summary wire为exact `6/5/5`；上一个已提交版本`5/5/4`以及V4
`4/4/3`、更早`3/4/3`只作历史并被新consumer拒绝，本轮未发布中间版不再二次升号。
legacy WAIT不绑定，checkpoint/resume authority仍false。完整合同与未授权边界只在
[portable Full-A实验§0](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md#epa48-fresh-runtime-binding-20260821)
维护；恢复与调用见[`setup_local_sync`](setup_local_sync.md#bind-the-exact-epa48--rsl-rl-312-site-for-portable-full-a)。

## HISTORICAL / SUPERSEDED — 3. 旧依赖顺序（现役只认§0.4清单）

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
3. **SUPERSEDED：不再造单一mega-state。** 现役中心链是single
   `PlantFacts → ActionBallState transition → StepTelemetry`；backend-local adapter与existing WAL/ACK留在
   两端。backend physics、独立plant/measurement、finite、full-key/generation、optimizer与fsync边界保持
   分离；只删除zero-live事务、same-writer echo与重复派生。
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

## HISTORICAL / SUPERSEDED — 2026-08-22 dd82之后的执行TODO

本节保留`661ff84b`到首次细分profile的执行轨迹；未完成项已由本页§0.2的`aa42418b`裁决取代，不是第二份
active队列。

- [x] 保留`dd82bb7b` Isaac fresh只读运行，确认连续finite ACK、fault/CENSOR为0；早期episode尚未到tick295时，
  mimic/strike/landing按零分母记`未测`。
- [x] 封存MuJoCo首次fresh自然RC1 root；根因为launcher遗漏显式ready-pose输入，不复用namespace。
- [x] 把ready pose变成required CLI + fixed-SHA child binding；host launcher `15 passed`。
- [x] 实现同一D2H内transport/keyed双activity事实；空业务跳R03与R07 keyed写，但保留R07 readiness/Motion。
- [x] exact Pod分进程运行fast-path、retired反例、R07 no-key与launcher focused tests：候选
  `8aa3…`合计`546 passed,32 skipped`；随后修复active-path noncontiguous fixed flight-slot后，最终
  `661ff84b…`重跑直接覆盖的Physical/Epoch两文件=`92 passed,7 skipped`。另一个未改区域的旧N2 landing
  fixture因缺`scene.cfg.replicate_physics=true`有`2 failed`，不混入本候选通过数，也不伪写成已修。
- [x] `661ff84b…` fresh Isaac已取得连续ACK0--10、fault/CENSOR/nonfinite为0；idle Epoch commit约
  `816--824/update`，低于`dd82bb7b`约`1104--1152/update`。但profiler-off ACK5--10 wall约
  `14.94--15.91 s/H48`，仍高于约`12 s/H48`量级目标，故只完成发车/正确性，不完成性能项。
- [x] 同一commit的新MuJoCo用required ready-pose fresh启动并连续取得ACK0--4以上；ACK1--28
  collection大致`9.11--9.61 s/H48`、Reward/storage finite且conservation fault为0。两个backend现同时运行。
- [x] 在空闲GPU用只增加host-clock、5 update后自动卸载的嵌套profile拆开R07。修复后的
  `a7ae7c6f…`显示全批无shot-key时`r07_readiness_no_key=7.00--8.32 s/update`，几乎等于整个
  post-physics wall；首次`8cbccad8…`因替换frozen实例方法自然RC1的root保留，不冒充训练失败。
- [x] 证伪`3e53f991…/19877d8d…`的845行bootstrap readiness旁路：同H48五轮collection仍为
  `13.84--14.59 s/update`，没有可用提速。它只缩窄Epoch clone，却保留Motion frame0、完整plant读取、
  13项R07误差和Motion install，因此不继续维护第二套bootstrap ABI。
- [x] **SUPERSEDED：**没有采用same-tick双脚support，因为它仍会触发昂贵ContactSensor读取；`aa42418b`
  采用更窄且可证的N/A-zero no-key语义，keyed post-shot recovery保持现状。
- [x] exact Pod按新语义完成`304 passed, 6 skipped`，包括contact `.data`禁读sentinel、selected reset/stale
  chronology、keyed recovery和203/219 Observation。
- [x] 同H48/GPU先跑5-update bounded profile再跑profiler-off matched 5轮；旧support-read归零，后者中位
  `6.346 s/H48`，因此启动fresh Isaac successor。
- [x] successor连续ACK/fault0后精确退役旧Isaac；随后同source启动fresh MuJoCo并在连续ACK后退役旧MuJoCo。
