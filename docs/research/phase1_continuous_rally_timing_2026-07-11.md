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

A 若在 actor 不控制时执行 bridge，下次交接究竟把“实际执行的 bridge action”还是“shadow actor
action”写入历史，不能凭感觉二选一；在完整 handoff contract 出来前 A 也不得启动正式卷。B 的 ready
位置是集合内确定性投影，不是死跟 frame 0；它要同时满足站位/直立/低速/支撑脚滑/执行器余量/
自碰桌网余隙，且能在每个启用动作与随机 deadline 前安全启动。

三臂首卷使用同一 immutable random-arrival 题顺、reveal tick、deadline、FF/FB/BF/BB 转手、plant、
face、motion、reward bytes/比例/总预算、seed 和 checkpoint cadence；miss/infeasible/fall 都留在全机会分母。
q10 只看方向，q50 才决策；同卷先过 Isaac，最后以智元 vendor MuJoCo Gate3 no-reset 主判。

只有结构臂证明 B/C 无法获得 ready set 且第一拍未退化，才开三个同阶段 reward：收拍平衡债、
ready-set potential、随机来球可接战性。先用冻结 rollout 归一化量纲，再做完整 `2^3` 开/关交互，只在
确有交互时补 fixed-total-budget mixture。安全/自碰是不可补偿硬门，不允许用其他 reward 抵消。

这套结构预注册在 `configs/phase1_recovery_tuple_abc_prereg_20260712.json`（SHA
`39b97915...b71e1a`），20 个红队测试已通过。它仍是 launch-blocked 设计，没有 materialize schedule、没有训练也没有
Gate3 连续成绩。复现命令见
`docs/operations/run_phase1_recovery_tuple_prereg.md`。
