# Phase-1 连续挥拍与任意来球时序审计（2026-07-11）

Status: Audit complete; the current 24-arm pairing×plant pool remains unchanged. The fail-closed
T1 core scheduler is implemented at `be5d7cf`, but no event schedule has been materialized and no
T0/T1 arm is launch-authorized or trained.

## 结论

现役 Phase-1 已经训练了“动作播完后不传送、带着真实身体状态进入下一拍”，但没有训练实战意义的
“下一题在合理时间窗内随时出现”。它能回答慢节奏 carry-state 恢复，不能证明机器人可以在场馆节奏下
连续接战。把 episode 单独从 10 秒改长不会解决这个错位，因为新题现在只在完整动作 clip 结束时安装。

因此：

- 正在跑的 24 臂继续保持冻结合同，用于回答 face pairing×plant 问题；
- 不把它们的单球 checkpoint 曲线或自然 clip-wrap 指标写成“连续实战已通过”；
- 另开只改变时序的 `T0/T1` 配对臂，先证明 event-driven next-task，再做更长 episode；
- T1 仍先过单球保真门，避免用“更会恢复”掩盖第一拍精度下降。

## 已实现的核心与仍未实现的卷

`be5d7cf` 新增独立的 post-strike event path，不复用会清 actor history/noise 的单拍
BankExam 安装接口：只有接受的 exact strike 才能启动绝对 tick；揭题时同一 command-manager
step 原子安装显式 train-bank row、native clip、精确 hold 和固定 deadline。miss、unavailable、
infeasible 都在原 deadline 消耗机会；不传送、不 reset robot/last-action/history/noise，也不把
来球静默推迟。所有会改变 timing 的字段和 schedule byte SHA 都进入 hard contract。

这只关闭训练端核心状态机缺口。原 prereg `2e7c4a34...2289c` 是冻结审计件，不回填 null；
必须新建 launch prereg 并补齐 materialized screen/decision schedules、连续 Isaac/MuJoCo judge、
self-hit instrumentation、fresh exact baseline 和 semantics-correct plant。上述绑定未齐前，
`launch-check` 继续按设计失败，任何 Pod 都不得点火。

## 现役到底已经做了什么

现役 launcher 明确设置：

- `episode_length_s=10.0`；
- `wrap_teleport=false`：clip wrap 不重置机器人或 last action；
- 每拍 `hold_steps_range=[0,100]`，即 50 Hz 下等待 `0–2 s`；
- 真 episode reset 中 `post_swing_start_prob=0.25`，可从策略自己的收拍状态环形缓冲起步；
- actor 已看到当前目标和 `time_to_strike`；
- `clip_switch_prob=0`，`midswing_resample_prob=0`。

所以它不是“一局只打一拍”。动作结束后，`MotionCommand` 保留模拟器里的真实机器人状态，在同一
episode 内换 clip、换题，再经过 hold 开始下一拍。问题在于**换题时刻被绑在 clip 末尾**，不是由下一球
何时被对手击出、何时到达决定。`post_swing_start_prob` 只增加 reset 初态覆盖，也不是连续时间调度。

现有两个看似相关的开关不能直接代替 T1：

- `clip_switch_prob` 是整段动作每一步都可能 Bernoulli 中断，连本拍击球前也会打断；它不是“击球后按
  下一球时间线切题”；
- `midswing_resample_prob` 只改变 target 的 WHERE，不改变 strike deadline 的 WHEN；schema-v3 原子题也
  不能被拆开后随意重抽。

physical-ball 分支目前还是 metrics-only truth instrument，没有真实连续拍球冲量，因此也不构成连续
对拉训练环境。

## 现役时序的理论范围

fresh exact 动作合同为：

| clip | 长度 | 击球帧 | 频率 |
| --- | ---: | ---: | ---: |
| forehand | 141 | 66 | 50 Hz |
| backhand | 134 | 45 | 50 Hz |

若当前拍为 `c`、下一拍为 `n`，两次击球间隔是：

`delta = (L_c - strike_c + hold_steps + strike_n) / 50`。

| 转手 | hold=0 时 | hold=100 时 |
| --- | ---: | ---: |
| FH→FH | 2.82 s | 4.82 s |
| FH→BH | 2.40 s | 4.40 s |
| BH→FH | 3.10 s | 5.10 s |
| BH→BH | 2.68 s | 4.68 s |

正反手等概率、hold 均匀时，混合分布约为 `q10/median/q90 = 2.90/3.75/4.60 s`，绝对最短
`2.40 s`。每个 10 秒 episode 通常只提供约 2–4 次机会，不足以稳定量出第 5 拍后的漂移和恢复债。

## 场馆数据的保守核对

数据源是 Pod 上的：

`/workspace/yikang/latest_data/analysis/segments/strikes.json`

SHA-256：`6ad3c45959c94b6fdd4033130403c32e0f1b612a138738c12afa43a58f752841`，154 个
检测接触。用仓库工具复现：

```bash
python3 hope_training/whole_body_tracking/scripts/analyze_rally_intervals.py \
  "$BALLFIT_DATA_ROOT/analysis/segments/strikes.json" \
  --max-leg-s 2.5 --summary-only
```

过滤规则故意保守：按 `take,t_c` 排序；排除 `dianqiu*`；只接收三个**连续检测到的**接触
`A paddle -> B paddle -> A paddle`；两段各自都不得超过 2.5 秒。这样排除明显局间空档和长段漏检，
得到 `n=21`（`gaoqiu=16, tantiao=4, zhengchang=1`）：

| 量 | q10 | median | q90 |
| --- | ---: | ---: | ---: |
| 同一球员相邻两次击球 | 1.757 s | 1.903 s | 3.356 s |
| 我方击球→对方击球 | 0.833 s | 0.951 s | 1.493 s |
| 对方击球→我方下一击 | 0.823 s | 0.948 s | 1.683 s |

样本偏向高球、没有 rally id，而且检测器可能漏接触；它不能直接冒充最终 match-play 分布。但它已经
足以否证“现役 2.90/3.75/4.60 秒时序覆盖了场馆核心”：实测下一题最自然的出现点（对方击球）中位约
在我方击球后 0.95 秒，而现役要等本拍 clip 在击球后约 1.50 秒（FH）或 1.78 秒（BH）播完才换题。

## 最小可归因的新训练设计

先做一对，不把 timing、guard、奖励和 motion 同时改掉：

| 臂 | 唯一结构差异 | 其余合同 |
| --- | --- | --- |
| `T0` | 现役完整 clip-wrap 后换题 | 选定的 SZ checkpoint/seed、motion、bank、plant、pairing、reward 全冻结 |
| `T1` | 击球后 event-driven 安装下一题，并保留 carry state | 与 T0 相同 |

T1 的事件语义：

1. 本拍到 exact strike 后，从同一个冻结的 A-B-A 时间表读取“我方→对方”和“对方→我方”两段；
2. 在对方击球/新预测可用事件原子安装下一题，不重置 robot、last action、历史观测或传感噪声状态；
3. 下一题的 `time_to_strike` 直接等于“对方→我方”剩余时间，不等上一 clip 自然播完；
4. 只允许击球后触发，不使用全相位 Bernoulli switch；miss 后也继续发下一题并留失败票；
5. 若原动作相位来不及到 deadline，使用有界、合同内的 phase controller/机器人本位时间律，或者把题
   明确记成 infeasible；禁止静默推迟来球；
6. 第一阶段先覆盖约 `1.75–2.20 s` 的核心同侧间隔，再扩到保守样本约 `1.28–3.68 s` 的压力范围；
7. episode 至少 30 秒，或者按 12 个 scheduled opportunities 结束，以便量到第 5 拍以后。

现 hard contract 已绑定 episode、hold、stand-start、wrap 和 speed，但还没有绑定
`post_swing_start_prob`、`clip_switch_prob`、`stagger_initial_clock`、`midswing_resample_prob`。T1 上 GPU
前必须把所有能改变 next-task 时序的字段纳入新合同版本；当前 fresh checkpoint 不因事后新增字段被
重新贴标签。

## checkpoint 怎么早判但不误杀

T0/T1 仍采用 checkpoint 漏斗，不等 terminal：

1. 每个里程碑先跑同一 clean 单球 q10，确认 T1 没先破坏第一拍；
2. 再跑同一冻结的短连续小卷（建议每个序列 5 次 scheduled opportunities），比较 T0/T1 的 carry
   衰减方向；
3. 小卷只能决定是否值得补证，不能 stop/promote；决定点仍需同一 immutable schedule 的
   `50/侧` 单球和连续卷；
4. 只在两个独立 seed 都显示第一拍不退、后续拍改善后，才补完整间隔尾部、噪声、Isaac/MuJoCo
   同题和 Gate 3B。

连续卷至少报告：

- 第 1/2/3/4/5+ 拍的 exact、击球、上台、不摔、tracking-loss；
- FF/FB/BF/BB 四种转手和真实 interval bin；
- `next_task_started / scheduled_opportunities`；
- `return_and_recover_and_start_next` 全分母率；
- deadline miss、ready time、ready margin 和 infeasible 题数；
- 第一失效前连续拍数，以及连续合法回球长度的 p10/p50；
- base drift、倾角/高度、脚滑、力矩饱和、q-des clamp；
- reset/teleport 次数（应为 0，否则逐次入账）；
- 第 2/3/4/5+ 拍相对第 1 拍的质量衰减；
- 同一 immutable question+interval schedule 的 Isaac/MuJoCo 差异。

已有 `return_and_recover_rate` 只证明自然播完完整 clip 后能进入下一题，不能证明 1.8–2.2 秒快速来球
仍能接战。

## 2026-07-12：T1 解决“何时出题”，不自动解决“两题之间给 actor 什么”

对现有实现的逐行审计把这两个问题分开了。现在的训练语义始终保持题目 tuple 同代：

- 普通 wrap 前，当前拍的 `position/velocity/normal/side` 整套不变；wrap 时整套换成下一题；
- T1 在 reveal 前把旧 clip 夹在末帧，依然给整套旧题；reveal 的同一 command-manager step
  原子写入新 clip、hold、位置、要求速度、拍面法向、来球、base anchor 和 side；
- actor 的延迟/dropout ring 也把 pos/vel/normal/side 作为一个原子世代；T1 不 reset 机器人、
  last action、history 或 noise state。

当前 vendor Gate3 C++ 的 179-D idle 路径不同：每 tick 用 live base 重算一个新位置，却继续搭配上一拍
的击球速度和法向/rho。这是训练中不存在的混代移动目标，故只标 `OOD diagnostic`，不是可以靠调
anchor 变成正式结论的第四臂。静态交接还会把 last-action 清零，也不等价于 T1 carry-state。

### 先比较三种结构，后决定是否需要 reward

| 臂 | 击球后到 reveal 前的语义 | 现有 179 checkpoint |
| --- | --- | --- |
| A explicit bridge | 可打断、不挪 deadline 的安全 PD/轨迹桥进 ready set；actor 交接历史必须内容绑定 | 只可作为桥接证书后的冻结 swing diagnostic |
| B canonical tuple | 同一 actor 原子接收 `ready-set position + zero velocity + neutral normal + rho0 + ready phase` | 不可用；当前训练没有这种 tuple，必须 fresh |
| C previous tuple | 整套上一题原样保持到 reveal，然后整套切换 | 只可 zero-shot 同代 tuple diagnostic；学会任意来球的声明仍需 fresh |

A 若在 actor 不控制时执行 bridge，actor 的 last-action 通道必须写入实际 executed bridge action
到 actor-action 坐标的内容绑定精确投影；shadow actor action 只作诊断，不能冒充执行值。无法精确投影
就 fail closed。在完整 handoff、逐 tick ownership 和 PPO rollout contract 出来前 A 不得启动正式卷。
只有 `actor_control_mask=1` 且 actor sample 真正执行的 tick 才有对应实际 action 的 logprob；bridge tick
的 policy/entropy/value loss mask 全为 0。真实 bridge reward 折叠进前一个 actor option transition，非真
终止在下一 actor state bootstrap，miss/infeasible 不伪装 terminal。三臂预先固定相同 env-step、题目
机会、update、actor-owned 样本、minibatch 和 epoch；B/C 多余样本按不看结果的固定索引下采样，A 样本
不足就整对 fail，不补样、不复用、不多跑 A。B 的 ready 位置是集合内确定性投影，不是死跟 frame 0；
它要同时满足站位/直立/低速/支撑脚滑/执行器余量/
自碰桌网余隙，且能在每个启用动作与随机 deadline 前安全启动。

三臂首卷使用同一 immutable random-arrival 题顺、reveal tick、deadline、FF/FB/BF/BB 转手、plant、
face、motion、reward bytes/比例/总预算、seed 和 checkpoint cadence；miss/infeasible/fall 都留在全机会分母。
q10 只看方向，q50 才决策；同卷先过 Isaac，再由智元 vendor MuJoCo Gate3 对同一
C++/MJCF/plant/model 做 first-tick 与连续稳定硬前置，最后 Gate3B 复用同 runtime 跑随机来球 q50，
主判 first-strike non-regression 与 return quality。

只有结构臂证明 B/C 无法获得 ready set 且第一拍未退化，才打开 reward 消融。随机来球先保持为
environment/schedule axis，并由真实 next-strike 成绩判定，不能默认变成第三项 dense reward。先用冻结
rollout 归一化“收拍平衡债 D”和“ready-set potential R”，做配对 `2^2`；只有 readiness critic 在
独立 train/calibration split 锁定，并一次性通过与 formal Gate3B 题纸隔离的 prereg critic-gate q50，
证明无未来信息泄漏且能预测真实下一拍，才允许把它作为第三因子进入 `2^3`。
幸存项先做固定总预算 `W` 的 simplex/centroid，再至少补一个第二总量级，区分比例效应与 reward 总力。
安全/自碰始终是不可补偿硬门。完整修订与论文边界见下方 2026-07-13 审计。

这套结构预注册在 `configs/phase1_recovery_tuple_abc_prereg_20260712.json`（SHA
`ca7806df...d810616`），50 个红队测试已通过。它仍是 launch-blocked 设计，没有 materialize schedule、没有训练也没有
Gate3/Gate3B 成绩。复现命令见
`docs/operations/run_phase1_recovery_tuple_prereg.md`。

## 2026-07-13：自主恢复、通用 ready 与随机来球的 primary-source 审计

### 决策摘要

论文证据不支持把“上一拍卸载、回到 ready、随时可接下一题”立即写成三项同窗 dense
reward。更可归因也更省算力的拆法是：

1. **hard safety** 始终在 reward 之外：跌倒、自碰、桌网/地面碰撞、滑移、非法关节/
   力矩/q-des 都不可由击球分补偿；
2. **balance debt** 是收拍后需要消散的动态债务，而不是 frame-0 位置误差；至少包含 base
   倾角/角速度、支撑与滑移、CoM/capture margin、执行器余量；
3. **ready-set potential** 衡量到一个可生存且可启动下一动作的集合，而不是到单一姿态；
4. **random arrival** 首先是环境/题表轴：在 actor 不知道下一题时随机 reveal，随后用真实
   deadline 下的下一拍击球/回台结果回传信用。它不是默认的第三个 dense reward。

因此近期 D0 以 ACE-style 可打断安全 bridge 为最快可证结构；长期自主恢复先做 T1 的
event-driven state coverage。只有真实下一拍回报仍稀疏到学不动，而且一个独立 readiness
critic 只在 `critic_train` 拟合、在不重叠的 `critic_calibration` 锁阈值，再一次性通过预注册的
`critic_gate_q50`，才允许把其**对不可变题库的期望或保守分位**做成第三个 potential。
`critic_gate_q50` 与最终 Gate3B formal q50 内容隔离；formal q50 只解封一次，绝不能做开发集。
pre-reveal reward、actor observation 和 critic privileged state 都不得泄露尚未 reveal 的真实下一题。

### 三项职责的接口边界

| 职责 | 首选实现 | 必须避免的假绿 |
| --- | --- | --- |
| hard safety | supervisor/termination/trajectory certificate；all-step、all-opportunity 分母 | 把碰撞写成有限负 reward，允许回球分抵消 |
| balance debt | 有界、负向或 potential-difference 的动态债；自然随挥 grace 后激活 | 关节回到第 0 帧但仍带角动量、脚滑或执行器饱和 |
| return-to-ready | 到 ready **集合**的 progress；到达后不持续发正 income | `hold_ready` 按停留时长刷分，或僵住换稳定 |
| random-arrival readiness | 不可变 reveal/deadline 分布 + 真实下一拍目标；必要时用校准 critic 的题库期望/低分位 | 偷看真实 future tuple、用 pose proxy 冒充可接战性 |

通用 ready 是 pre-reveal 的 robust centre/set；新题 reveal 后，目标才可以变成
next-task-conditioned prepare。若 Franco、v6、v7 的起始集合没有安全交集，就保留 family-specific
ready sets 和显式 transition graph，不能为了一个“通用第 0 帧”删掉难题。potential 若采用
`F(s,s') = gamma * Phi(s') - Phi(s)`，reveal mode、deadline 和 command generation 必须进入
Markov state，且 true terminal/option boundary 要正确处理；否则不能借用 policy-invariance 结论。
正的 per-tick hold/brake income 仍禁止。理论边界见
[Ng, Harada and Russell, 1999](http://aima.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。

### 论文机制与不可误类比边界

| Primary source | 可以直接映射的机制 | 不能据此声称什么 |
| --- | --- | --- |
| [ACE, Nature 2026](https://www.nature.com/articles/s41586-026-10338-5) | 每个 32 ms 轨迹段都有近时间最优 MPC reset；终止后执行；目标可为固定 neutral 或由 incoming-ball/落点意图条件化的 prepare network 给出；训练初态可来自历史 reset plans | ACE 是专用固定安装的多轴桌球机构，没有自由站立 humanoid 的 support/CoM/foot-slip 债；它证明 bridge/prepare 架构可行，不证明 A3 自主 balance recovery |
| [HITTER v2](https://arxiv.org/html/2508.21043v2) | 10 s episode 内连续 swing；每个 swing 完成后重采正/反手与 base/racket 目标；base reward 在 strike 前激活；真实系统有长 rally | 公开训练合同仍在 **swing 完成后**换题，两条参考各 1.88 s；没有 mid-followthrough random-reveal 对照，不能代替 T1 |
| [SMASH](https://arxiv.org/html/2604.01158v1) | 1.08 s strike-centred clip 前后各 0.54 s，显式含 preparation/recovery；cyclic phase、生成边界平滑、tracker 动力学过滤 | Motion-VAE 用于离线扩库，运行时是 motion matching；没有证明 online 生成两拍 transition，也没有任意 recovery 时刻 reveal 实验 |
| [PACE](https://arxiv.org/html/2509.21690v4) | action residual 锚在举拍 nominal stand；episode 最多五个连续 serve，fall early termination | 发球机序列和 episode-start 随机位姿不等于任意 mid-swing arrival；论文没有独立 recover-to-ready reward 结论 |
| [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf) | phase-conditioned tracking、skill selector、composite policy；可在 cycle 间学 transition | phase 线性推进且 selector 在 cycle 边界换 skill；论文自己指出这限制 timing adjustment 和灵活 recovery |
| [BeyondMimic](https://arxiv.org/html/2508.08241) | diffusion inpainting 展示 walking->agile skill->walking 的在线组合；motion phase 和 IMU twist 支持稳定/恢复 | 预测 horizon 仅 0.64 s，作者报告 mode switching 起止会 stumble、fine-grained objective 较弱；不是近期精确 racket-contact runtime 替代品 |
| [MaskedMimic](https://research.nvidia.com/labs/par/maskedmimic/) | 同一 physics-based character controller 可接随机变化目标和不同 goal modalities，支持“goal-conditioned ready”方向 | 这是 simulated character 结果，不含 A3 plant、vendor MuJoCo、桌球接触或 hardware safety 证书 |
| [Options framework](https://doi.org/10.1016/S0004-3702%2899%2900052-1) | recovery/bridge 应显式绑定 initiation set、termination、timeout，并可在新题到达时 interrupt | interruption theorem 假设正确 value/model，不是物理安全证书，也不解决 PPO bridge-tick accounting |
| [Leave No Trace](https://arxiv.org/abs/1711.06782) / [Recovery RL](https://arxiv.org/abs/2010.15920) | 分离 task 与 reset/safety policy；recovery value 判断不可逆/危险区 | “回安全区”不等于“回到下一拍高 dexterity ready set”；非高速 humanoid table-tennis 结果 |
| [HER](https://arxiv.org/abs/1707.01495) / [Residual RL](https://arxiv.org/abs/1812.03201) | off-policy sparse-goal relabel、nominal controller + learned residual 是以后可用工具 | 当前 PPO 不能无算法变更直接套 HER；achieved goal 可不安全；residual 也不会自动保留 bridge 证书 |

这些来源共同支持“结构/状态覆盖先于 reward 比例”，但没有一篇证明 HOPE 的三项权重、任意
mid-followthrough reveal 或 vendor MuJoCo Gate3B 已通过。

### 最小 T0/T1/T2 因果阶梯

| 阶段 | 唯一新增能力 | 允许的结论 |
| --- | --- | --- |
| `T0` cycle-bound control | 保留当前 full-clip install；robot/action/history carry；仍按原 immutable reveal/deadline 评分，不顺延 | 量出完整 clip 延迟造成的 deadline/infeasible 基线；不声称任意来球 |
| `T1` event-driven structure | reveal tick 原子装整套 tuple；无 teleport/reset/history clear；使用 A/B/C 结构筛出的 bridge/tuple 方案；**不加 recovery shaping** | 回答正确时序、状态覆盖和 handoff 本身是否足够；近期 demo 首选 A safe bridge |
| `T2` learned autonomous recovery | T1 全部字节/题表/时序不变，只增加经验证的 balance/ready shaping；arrival 仍为环境轴 | 回答 learned shaping 的增量；只有校准 readiness potential 存在时才把 arrival 加为第三 reward factor |

A/C 现有 checkpoint 只能做冻结 diagnostic；正式 T1 因 bridge ownership/训练分布改变仍需 fresh
exact paired。B 必须 fresh。T2 不得同时换 motion、TOPP、动作数、plant、planner 或 episode
schedule。每个 checkpoint 先过 frozen one-shot non-regression，再过同一 no-reset schedule。

### Reward DOE：先 `2^2`，有校准才 `2^3`

若 T1 已过安全、ready latency 和下一拍回球门，T2 没有必要，为 demo 直接保留结构方案。若 T1
失败且证据指向 learned shaping：

1. 在冻结 post-strike rollout 上，把 balance-debt `D` 和 ready-set potential `R` 按每个
   scheduled opportunity 的折扣绝对贡献、激活率和 policy-gradient 量级归一化；重复信号先删；
2. 用 paired seed/block 的完整 `2^2` 跑 `none/D/R/D+R`，而不是 OFAT 单项赢家相加；
3. 只有独立 arrival critic 在 `critic_train` 拟合、在不重叠的 `critic_calibration` 锁定，再一次性
   通过与 formal 题纸隔离的 `critic_gate_q50`，且加入后仍不泄露 hidden future tuple，才升级完整
   `2^3`；最终 Gate3B q50 保持 sealed；
4. `2^k` 每格必须有重复。一个 seed 的八角只能抓 catastrophic interaction，不能给交互误差或正式
   因果结论。完整 factorial 可估主效应、两两和三阶交互，见
   [NIST 2^3 design](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3332.htm)；
5. 存活三项再做固定总辅助 reward `W` 的 7 点 simplex-centroid：三个 vertex、三个 50/50 edge、
   一个 1/3--1/3--1/3 centre
   ([Scheffé, 1963](https://doi.org/10.1111/j.2517-6161.1963.tb00506.x))。fixed `W` 只识别**比例**；
   PPO 还可能依赖总量，因此至少在 centre
   增加第二个 `W` 水平，必要时把 `W` 当 process factor。混料实验的“只依赖比例”假设见
   [Scheffé, 1958](https://doi.org/10.1111/j.2517-6161.1958.tb00299.x) 和
   [NIST mixture boundary](https://www.itl.nist.gov/div898/handbook/pri/section5/pri54.htm)。

seed 是 block；每个 seed 消费相同 post-strike state reservoir、question/timing order、初始化、
optimizer budget 和 checkpoint milestones。q10 只看方向且不能不对称停 cell；q50 才作决定。

### q50 失败判据

以下任一项成立即失败，不用 surrogate 平均掩盖：

- 任一 self-hit、桌网/地面非法碰撞、guard loss、不可接受 fall 或执行器合同违规；
- 任一 mid-sequence teleport/reset、last-action/history/noise clear、deadline shift、miss/infeasible
  替换或删失、hidden next-task 泄露；
- ready/debt 指标改善，但 paired all-opportunity 下一拍 exact/return 没有改善；
- frozen one-shot 每侧 first-strike/return 超过预注册 non-regression margin，或靠降低拍速/缩动作换恢复；
- FF/FB/BF/BB 或早/中/晚 reveal cell 出现被总平均遮住的塌陷；
- 第 `5+` opportunity 相对前四拍继续衰减，说明债务仍在累计；
- 交互方向只在单 seed/单 GPU/单 engine 出现，或 Isaac 增益在 vendor MuJoCo 反转；
- readiness critic calibration 失效，却仍用其 dense surrogate 选模型；
- critic/factor/model 开发重复查看 `critic_gate_q50`，或提前解封/重复消费最终 Gate3B formal q50。

non-regression 的数值 margin 应从冻结 baseline 的 q50 二项波动和 paired schedule 事前确定；上述论文
不支持事后凭空写一个 `5%`。Isaac 是开发/归因腿，最终必须先过同 runtime 的 Gate3 first-tick+
continuous stability，再由 Gate3B random-arrival q50 判 first-strike 与 return quality；两引擎不平均。
本节是文献/设计审计，没有启动训练、sim、Pod 或真机，也没有把 G05/G06 改成 Done。
