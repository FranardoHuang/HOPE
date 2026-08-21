# ActionBall 双后端长跑：当前执行 TODO

> 状态：`ACTIVE-successor-construction / no-authorized-live-run / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-21
>
> `origin/main:docs/NOW.md` 是全项目唯一优先级权威。本页只维护
> [FullMDP](../DEFINITIONS.md)（完整球路、击球、落点与恢复状态机）单动作双后端
> successor 的依赖顺序、运行事实和完成条件，不维护竞争性的优先级队列。旧的单动作执行页已转为
> [只读历史账](action_ball_single_action_dual_backend_todo_20260817.md)。

## 1. 当前运行事实

三条旧长跑均已停止，不存在可继续推进的 active run：

- Isaac `e8eef4fb…`止于 durable ACK（optimizer成功后已持久化的更新签收）`4603`，累计
  `452,591,616` transitions；
- Isaac `ddb1e7c4…`止于 durable ACK `3467`，累计 `340,918,272` transitions；
- portable MuJoCo r3 止于 durable ACK `10249`，终段 last-100 wall mean/median=
  `4.890/4.886 s/update`。

两条 Isaac 的退出因果仍是`未知`：现有日志没有足够证据把它归因于 OOM、外部停止或代码异常，
也不能因 `final_rc=0`把不完整 run 写成完成。MuJoCo r3 由真实 `EPA_HORIZON`（扩展多面体碰撞算法的
迭代深度上限）overflow fail-stop；
它同时消费了会在 IDLE 时随全局 step 漂移的旧 229-D observation，所以不允许 resume。

本次尝试只读刷新 Pod 状态时 SSH 认证失败；因此本文不声称当前 GPU 是否空闲，也不把旧快照写成
“当前没有 compute process”。任何新运行都必须重新取得 live 资源证据。

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

## 3. 唯一依赖顺序

每项必须独立提交、可单独回退；前一项闭合后才进入后一项。

1. **Observation V2原子闭合**：Isaac/MuJoCo、training hard contract、snapshot receipt与文档同时切到
   A203/219；两后端用独立native state正负扰动验证frame、符号、scale、mask和R06 live selection。
   V1只为明确的历史 WAIT consumer保留，不允许Full-A静默fallback。
2. **MuJoCo EPA48**：同一 deterministic geom pair/pose必须在stock-24确定overflow、fork-48 finite；
   再跑exact GPU focused、固定tape reason/counter/safety parity和独立instrumented/ASan oracle。
   stock CPU同样硬编码24，不能充当golden。未闭合前`training_authorized=false`，r3不resume。
3. **单一 ActionBallState**：一个device-resident mutable state唯一拥有phase/generation/shot/contact/
   outcome/fault；K-row候选只构造一次并sparse commit，只有真实transition写compact event delta；
   `K=0`/zero-live-flight成对跳过Physical/scene/R06/Epoch空事务，PPO boundary统一汇总。
   Phase-C0已先退役zero-business D05事务，Phase-C1已收窄Motion publication；剩余dense Epoch/R06/Physical
   state与host sync仍须按真实profile继续收敛，不能把两刀称为single-state完成。
4. **matched H48性能验收**：clean exact source、同卡、profiler-off，按zero/mixed/active strata对比；
   同时验证fixed tape RNG/highwater/reason/done/reset/Reward20/203/219 parity。报告原始wall、transitions/s和
   H24-equivalent；首墙转移后重新profile，不继续堆零碎clone patch。
5. **fresh namespace短验与训练**：先做clock-fixed、EPA-fixed的portable MuJoCo V2，再做通过matched性能
   验收的Isaac V2；不复用旧namespace或snapshot，不插`ACCEPT>0`门。短验只回答构造、有限性和真实调用点，
   随后同一进程继续训练并按balance→mimic→entry→strike→landing分阶段报告分母。

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
MuJoCo fork状态见[G06](../gates/G06_isaac_to_mujoco.md)，操作边界见[训练工序](run_training.md)。
