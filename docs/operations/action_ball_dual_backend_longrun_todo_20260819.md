# ActionBall 双后端长跑：当前执行 TODO

> 状态：`r36-source-integrated / final-pod-validation-open / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-30
>
> `origin/main:docs/NOW.md` 是全项目唯一优先级权威。本页只维护
> [FullMDP](../DEFINITIONS.md)（完整球路、击球、落点与恢复状态机）单动作双后端
> successor 的依赖顺序、运行事实和完成条件，不维护竞争性的优先级队列。旧的单动作执行页已转为
> [只读历史账](action_ball_single_action_dual_backend_todo_20260817.md)；本页过去的 superseded 章节已移到
> [双后端 TODO 历史归档](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md)。

<a id="fullmdp-v6-todo-current"></a>

## 0.6 2026-08-30 current：R36 单一因果链与最终 Pod 闭合

本节是唯一现役局部执行合同；`origin/main:docs/NOW.md`仍是项目优先级权威。R35 两条长期 run
已冻结为反例，R36 功能合同锚点为 `4fe23765`；后续热路候选 `bf941777` 不改变训练语义，且在 exact Pod
完成等价性和墙钟复核前仍为`未测`。所有结论继续限定为 `diagnostic_unauthorized`，不授权 promotion、
export、部署或真机。

### 已采用合同

- 课程是一条自然重叠链：`balance → mimic → hit → landing/recovery`。没有硬 Stage，也不以成功 Gate
  阻断下一层；上一层基本形成时，下一层应已经有真实 eligible 分母。
- D05 在题目到期时发布 frozen ActionEpoch。物理启动尚未请求时，以 typed `UNPLAYED` 无债退休；
  已 ready 且已请求启动却没有 launch 才是合同故障。
- physical launch 的 TTC 和时钟一律相对 public reveal；Mu 只有在 teacher 真正离开 frame 0 时才开始
  playback。R03 的击球位置、速度和有符号拍面唯一读取同一 frozen ActionEpoch，caller 不再复述自证。
- Phase1/2 只负责几何和连续解；cross-intent face equality 不再作为 Phase2 admission。Phase3 的
  face distance 只参与 seed 排序，不能冒充可行性证明；Phase4 才以最终 A3 plant、qdes/torque/support、
  racket-site、桌网球物理和 recovery 做独立准入。
- Phase4 击球 deadline 向上对齐到 20 ms policy grid，必须完整落在 episode/cadence 内；contact 后保持
  safe follow-through。canonical bundle 同时绑定 exact ball-physics YAML 和 compiled Mu plant bytes。
- Observation V3 保持 actor/critic `215/231`；本轮没有新增 Stage、Gate、oracle 或 actor 字段。

### 当前反例结论

R35 已不是“还太早”。Isaac 在最近冻结窗内已基本形成 survival/balance，且有 `4,951` 次 physical launch，
但 R03-valid 和 contact 仍为 0，故 mimic→hit 交接失败；Mu 同窗几乎所有完成 episode 都带 tilt，尚未到
launch，故先败在 balance/mimic。两端 fault/nonfinite/conservation 仍为 0，说明负例可信但不证明
physics parity。详细唯一数值真源见
[MuJoCo native readiness 实验 current correction](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)。

### 已直接修复

- [x] reveal-relative physical launch 与 teacher playback 起点。
- [x] R03 从 frozen ActionEpoch 取唯一 target，并删除 caller/same-writer 自证。
- [x] Reward bundle 构造后的同写者 metadata 自证删除；跨 owner identity、env 安装和 durable 边界保留。
- [x] filtered-net Mu runtime MJB 从 source MJCF 重编译并 repin；旧派生 plant 不再可加载。
- [x] Phase2/3/4 的 seed、admission 与最终 plant authority 分层。
- [x] Phase4 safe follow-through、exact physics YAML 绑定和 policy-grid deadline 闭合。
- [x] Isaac RSL3 只在成功 `env.step()` 后累加 observed rollout；PPO 前与 storage cursor和
  typed `H`三方核对，公式只作反例而不再自证。
- [x] 现役课程和 observation 合同未被上述修复扩成新的状态机。

### 发 long 前仅剩的阻断项

以下各项必须来自同一个 fresh、clean exact checkout；任何未勾选项都写`未测`，不能由 host 单测、
旧 evidence 或另一 backend 代签。

- [ ] 在 Pod 恢复并核验 private/ignored 输入、EPA48/RSL3、当前 exact MJB 与 physics YAML。
- [ ] 物化 final Phase4 canonical bundle，reopen 后逐字核身份；证明 trainer 真消费该 bundle、deadline
  和 frozen ActionEpoch，而不是旧默认值。
- [ ] 跑 final focused/combined Pod tests；Reward bundle、phase producer/consumer、observed-rollout chronology、
  journal/WAL 与 interface hostile tests 全部通过。
- [ ] Isaac 与 Mu 分别完成真实 CUDA `512×48×31` fixed-action：finite、0 contract fault，并对齐
  action identity、RNG/tape、reason/counter/safety 和逐项 payment。
- [ ] Mu 完成 profiler-off matched-strata 墙钟复核；`bf941777` 的 metric transfer 候选只在 fixed-tape
  数学等价后计为速度结果，当前`未测`。
- [ ] 每端使用 fresh O_EXCL run root/namespace 发射并写出首个 finite durable ACK。两端独立发射；
  Mu 不必等待 Isaac，任一端的学习结果也不代签另一端。

### 不阻塞本轮 long 的诊断与结构债

- command metric D2H/Reward 同周期 pack 的进一步窄化、monolith 拆分、teacher plant 动态可追踪性、
  sim2sim torque/contact first-divergence、blank-machine private asset distribution 和 formal
  exact-resume consumer 都继续记账，但不以“更优雅”为由阻塞已闭合的 fresh 训练入口。
- 性能改动只能删除重复同步或重复计算；不得删 reason/counter/safety/durable truth，也不得用更多 env
  掩盖热路。旧 command-metric 大改已经撤回，当前候选必须重新给 Pod parity 和 profiler-off 证据。
- reward 剂量、curriculum 目标失败率、full-body、entropy/sigma 等学习选择仍需 canary；确定性的时钟、
  坐标、单一 owner 和内容绑定修复不再做学习 A/B。

### 继续与停止规则

- 继续：持续 finite ACK；fault/nonfinite/conservation 为 0；`balance→mimic→hit→landing` 的真实分母按
  自然重叠逐层出现。零分母写`未测`，已有分母的零结果写 `0/denominator`。
- 停止并保留 root：NaN/nonfinite、identity 或 receipt 漂移、counter/journal/WAL 不变量破坏、contract
  fault、持续 table/fall/hard-limit 爆炸。
- 学习层未改善时先按真实分母区分 balance、mimic、launch、contact 和 landing；不以 total return 粉饰，
  也不增加成功 Gate 掩盖时钟、target 或 plant 根因。

### 当前执行顺序

1. [x] 冻结 R35 负例并把详细数字收敛到实验真源。
2. [x] 合入 reveal/R03/Reward owner/plant/Phase4 的确定性修复。
3. [ ] 在 final exact Pod checkout 完成 bundle consumer、focused tests、双端 fixed-action 与 matched rate。
4. [ ] 满足本节全部 pre-long 阻断项后，按自然可用 GPU 分别 fresh 发射 Mu 和 Isaac。
5. [ ] 运行中只按真实 denominator 判读课程；结构债在独立提交闭合，不 hot-patch active run。

## 历史链接兼容入口

旧章节不再留在现役 TODO，避免历史流水遮住唯一可执行清单。以下 anchor 只用于保持旧文档链接可解析；
内容和证据都在[历史归档](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md)，不构成当前
运行 authority。

<a id="fullmdp-v9-superseded"></a>

- [V8/V9 及后续 superseded 章节](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md#fullmdp-v9-superseded)

<a id="04-2026-08-23-v4最终冻结与v5第一性原理自查"></a>

- [V4/V5 第一性原理自查历史](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md#04-2026-08-23-v4最终冻结与v5第一性原理自查)
