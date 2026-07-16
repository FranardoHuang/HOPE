# EXP-P1-ROLLING-TIMING-SUPERCOMBO — 快速准备、时间戳补偿与组合训练漏斗

- 状态：`running / demo-only inexact`（24 个唯一 claim 已消费；最近可信状态为 22 live、2 个 importer 基础设施拒绝）
- 阶段/轴：阶段 1，动作重定时、actor 可见目标/TTS 延迟、预测收敛抖动、击球 Reward 配比与脚朝向
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E2`（4096-env full-scene source probe 与真实 continuation 已运行；尚无 vendor MuJoCo 行为结果）
- 创建日期/最后复核日期：2026-07-16 / 2026-07-16

共享术语按[术语与人话对照](../../DEFINITIONS.md)解释。本实验的机器草案是
[`phase1_rolling_timing_supercombo_20260716.yaml`](../../../configs/phase1_rolling_timing_supercombo_20260716.yaml)。
顶层 [`launch_authorized=true`](../../DEFINITIONS.md#launch-authorized)，24 条 job 已在一次性 parent inspect
通过后消费各自唯一 claim；只授权本轮仿真续训，不授权自动重试、第二 seed、判卷、晋级或真机。

## 先回答：qdot 是不是随机施加力

不是。本轮的 `qdot=-5` 指
[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge)：读取 31 个真实 articulation 关节的速度与对应速度上限，
只有 `abs(qdot)/limit` 超过 margin `0.85` 后才按平方超限量收费。它是 policy 优化目标里的**确定性关节速度惩罚**，
不会向机器人 base、关节或刚体随机施加外力。随机 push/wrench 属于另一套扰动机制；本队列没有打开它。

## 本轮问题与共享组合

现实测试暴露出的首要问题是“球已经很近，动作却仍按老师的慢时钟走”。本轮先回答：在不丢失击球位置、拍速、拍面
与站姿的前提下，能否把正反手从动作开始到触球的可用时间压到约 `1.0/0.7/0.5 s`，并让 actor 看到与同一条
planner 消息一致、按 source timestamp 补偿过的剩余击球时间。

24 条臂都共享以下已知方向，不把它们各自再当 seed 复制：

- episode 固定 `16 s`，允许同一物理状态中连续多拍；
- [`V1`](../../DEFINITIONS.md#v1-free-wrist-velocity) 放开持拍手腕线速度逐帧模仿；
- [`V2`](../../DEFINITIONS.md#v2-strike-window-imitation) 把击球窗模仿强度降到 `0.25`；
- qdot-limit hinge 权重 `-5`、margin `0.85`；
- [`conditional face`](../../DEFINITIONS.md#conditional-face-guidance) 权重 `-0.4`；
- 非击球臂不再逐帧模仿老师；
- 同一 `v4rg_runtime_order_v3` 正反手动作、schema-3 signed-face train bank、16 秒 carry setting、零摩擦
  Isaac plant、物理球 metrics-only 与完整 optimizer 热启动。

变化只集中在五组可解释轴：动作 timing、TTS timestamp compensation、预测收敛 jitter、`14/10/5` 对三项近似均分
`9.67/9.67/9.66`、脚朝向 `-0.3/-0.6`。每格仍是组合机制问题，不声称完整正交 factorial。

## 为什么约 0.5 秒必须训练 motion retiming

当前正反手原生动作从起始帧到触球分别约 `66/45` 个 50 Hz control tick，即 `1.32/0.90 s`。planner 即使把
“还有 0.5 秒”算对，policy 若只在这种慢 reference 上训练，也没有学过在 0.5 秒内沿参考路径完成准备和挥拍；仅把
planner 的预警值改小，会把 deadline 与老师动作时钟互相矛盾。

因此使用训练已有的 `task.motion.speed_scale_per_clip=[FH,BH]` float-clock 路径，同时缩放参考速度：

| 人话 timing | 每 clip playback scale `[FH,BH]` | 推导 start-to-contact `[FH,BH]` |
| --- | --- | --- |
| 约 1 秒 | `[1.32,1.0]` | `[1.00,0.90] s` |
| 约 0.7 秒 | `[1.886,1.286]` | `[0.700,0.700] s` |
| 约 0.5 秒 | `[2.64,1.8]` | `[0.500,0.500] s` |

另有 `speed_scale_range=[1.286,1.886]` 的随机速度格。它对两个 clip 产生的 envelope 不相同：正手约
`0.700–1.026 s`，反手约 `0.477–0.700 s`；所以本文只称“clip 相关的 deadline diversity”，不把它误写成
统一的 `[0.5,1.0] s` 均匀分布。source gate 必须在实际动作 bytes 上复算 66/45 tick，并证明 retiming 后 bank 的
逆解答案仍是 absolute physical answer，而不是被错误再缩放一遍。

## actor 的原子 TTS timestamp compensation

`TTS` 是 time to strike，即离计划触球还剩多少秒。`main@704bf3a2` 已集成的
`task.racket.target_delay_tts_mode` 有三种人话语义：

- `live`：旧行为。pos/vel/normal/sign 已经延迟，但 actor 的 TTS 永远读当前 live clock，tuple 的“目标来自何时”
  与“还剩多久”并不一致；本轮不购买单独 live 臂。
- `source_timestamp_compensated`：把 source TTS 与 pos/vel/normal/sign 同步写进一个 delay ring；读出旧消息时，
  再减 `target_delay_steps × policy_step_dt`，恢复到消费时刻真正剩余的时间。这是主线处理。
- `uncompensated`：同样原子延迟整条 tuple，但故意不减已经过去的时间，因而给 actor 一个陈旧、偏大的 TTS；只保留
  一条与 0.7 秒主线逐项匹配的 negative control。

delay `2/4` 个 50 Hz step 分别只是 `40/80 ms` 工程 stress，不是从真实 capture→host 链路量出的标定值，不能据此
宣称 VRPN 延迟就是 40 或 80 ms。新 hard contract 必须同时绑定 `racket_target_delay_steps` 与
`racket_target_delay_tts_mode`；日志需同时给 live `time_to_strike_s` 与 actor-visible `actor_time_to_strike_s`，这样才能
看出补偿是否真的激活。

## prediction convergence jitter 的边界

现役 `target_noise_white=0.0019 m` 与 `target_noise_ar1_sigma=0.0052 m` 是不随 TTS 缩小的 measurement noise。
本轮另设的 `target_jitter_pos_per_s/target_jitter_vel_per_s` 模拟“预测越接近触球越收敛”：每步标准差乘
`clamp(TTS,0,1)`。中等 stress 为 `0.03 m / 0.15 m/s`，强 stress 为 `0.05 m / 0.25 m/s`。这两档尚未由真实
trajectory residual 标定，只能回答工程鲁棒性方向，不能冒充现实误差分布。为不污染时间戳因果问题，唯一
compensated/uncompensated matched pair 都保持 prediction jitter 为零。

## 24 格机制漏斗

`run_name` 是每条实际运行的唯一机器名；下表先用短 ID 表示同一 YAML job，并在“人话目的”里说明改了什么、回答什么。
24 格都沿用父 run 的 seed `3` 与完整 optimizer；它们是不同机制问题，不是同配方换 seed。

| # | Pod/槽 | timing | TTS delay | jitter pos/vel | Reward | foot | 人话目的 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | P1/G0 | 1.0 s | comp-2 | 0/0 | 14/10/5 | -0.3 | 宽松时限是否保住速度与平衡 |
| 2 | P1/G1 | 0.7 s | comp-2 | 0/0 | 14/10/5 | -0.3 | 快动作主参考，也是 #5 的匹配处理组 |
| 3 | P1/G2 | 0.5 s | comp-2 | 0/0 | 14/10/5 | -0.3 | 正反手 0.5 秒能否直接学会 |
| 4 | P1/G0 | range | comp-2 | 0/0 | 14/10/5 | -0.3 | 随机 deadline 是否优于固定时钟专精 |
| 5 | P1/G1 | 0.7 s | **uncomp-2** | 0/0 | 14/10/5 | -0.3 | 唯一陈旧 TTS 负控，只去掉 elapsed-time subtraction |
| 6 | P1/G2 | 0.7 s | comp-4 | 0/0 | 14/10/5 | -0.3 | 80 ms 未标定延迟 stress 下补偿是否仍稳 |
| 7 | P1/G0 | 1.0 s | comp-2 | .03/.15 | 14/10/5 | -0.3 | 长准备窗能否吸收早期预测误差 |
| 8 | P1/G1 | 0.7 s | comp-2 | .03/.15 | 14/10/5 | -0.3 | 主速度档对收敛目标的鲁棒性 |
| 9 | P1/G2 | 0.5 s | comp-2 | .03/.15 | 14/10/5 | -0.3 | 极速准备还容得下多少预测误差 |
| 10 | P1/G0 | 0.7 s | comp-2 | .05/.25 | 14/10/5 | -0.3 | 找到 prediction stress 开始压坏策略的边界 |
| 11 | P1/G1 | 1.0 s | comp-2 | 0/0 | equal | -0.3 | 非 timing 瓶颈时均分是否仍有益 |
| 12 | P1/G2 | 0.7 s | comp-2 | 0/0 | equal | -0.3 | 主速度档的 Reward 配比比较 |
| 13 | P2/G0 | 0.5 s | comp-2 | 0/0 | equal | -0.3 | 均分 Reward 是否帮助极短准备窗 |
| 14 | P2/G1 | range | comp-2 | 0/0 | equal | -0.3 | 均分 Reward 是否支持 deadline diversity |
| 15 | P2/G2 | 1.0 s | comp-2 | 0/0 | 14/10/5 | -0.6 | 宽松时限下强脚约束能否先稳脚再挥拍 |
| 16 | P2/G0 | 0.7 s | comp-2 | 0/0 | 14/10/5 | -0.6 | 快动作与站姿纪律的直接折中 |
| 17 | P2/G1 | 0.5 s | comp-2 | 0/0 | 14/10/5 | -0.6 | 强脚约束帮助还是压死极速动作 |
| 18 | P2/G2 | range | comp-2 | 0/0 | 14/10/5 | -0.6 | 站姿纪律能否跨 deadline 泛化 |
| 19 | P2/G0 | 0.7 s | comp-2 | 0/0 | equal | -0.6 | 均分是否抵消强脚约束对挥拍的挤压 |
| 20 | P2/G1 | 0.7 s | comp-2 | .03/.15 | equal | -0.3 | 均分是否提高追踪收敛目标的韧性 |
| 21 | P2/G2 | 0.7 s | comp-2 | .03/.15 | 14/10/5 | -0.6 | prediction 与 foot 两种稳健化是互补还是冲突 |
| 22 | P2/G0 | 0.7 s | comp-4 | .03/.15 | 14/10/5 | -0.3 | 80 ms delay 与 prediction jitter 的组合压力 |
| 23 | P2/G1 | 0.5 s | comp-2 | .03/.15 | equal | -0.6 | 完整极速稳健栈是否仍可学习 |
| 24 | P2/G2 | range | comp-4 | 0/0 | 14/10/5 | -0.3 | timestamp compensation 能否跨随机速度一致工作 |

YAML 按四轮排列；每轮先给六张 GPU 各一条，再进入下一轮。最终 Pod1/Pod2 各 12 条、每张 GPU 恰好 4 条。
`required_slot` 是 harness 需强制兑现的冻结分配，不允许操作者手工先塞满一张卡，也不允许因某卡未释放就把 Pod1 parent
的后代发到 Pod2。

Pod2 另外把 parent basin 当作**工程 portfolio 轴**：#13–18 从 qdot-face0.2 的精度 basin 继续，#19–24 从
16s-carry 的连续/低跌倒 basin 继续，各六条；Pod1 的 12 条全部从 free16 继续。parent 不同造成的分数差绝不能解释成
某个 timing/Reward/foot 主效应；这样分配只为明早同时保留两种较强初始能力，增加找到可用候选的机会。

## 旧 24 条停止与 parent 选择

在新 source 或新 job 动作前必须先停止旧自动训练。两台 Pod 已对 24 个绑定 PID/PGID 做 exact identity
检查后停止，随后逐项确认 process absent、NVML compute context 为空且 fatal=`0`。Pod1 的 12 条最后 checkpoint 在
`model_1500–1700`，Pod2 的 5 条 mature + 7 条 demo 在 `model_1900–5100`，两边报告 fatal count 都是 `0`。
本轮没有额外制造一份 stop-receipt 文件；发射时仍须在同一原子容量检查中再次确认 GPU occupancy，并对被选
parent 重验 checkpoint/hard/claim/binding。停止事实本身不再作为独立 blocker。

Pod1 parent 是旧 grid 中当前 matched 方向最强的“16 秒、自由非击球臂”：

- job `p1_pod1_arm_free_ep16_seed3`；
- run directory `/workspace/codexschema/phase1_long_pod1_20260715/runs/phase1_pod1_arm_free_ep16_seed3`；
- read-only `run_binding.json` 已解析出 exact RSL directory
  `/workspace/codexschema/nohope_p1_activation_successor_2c2d70d/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-15_16-18-03_phase1_pod1_arm_free_ep16_seed3_20260715`，
  checkpoint 是其中 `model_1600.pt`，SHA-256
  `fd69dd6c9eac8ba4910de2a75de2709b5927a8d5302a8763ce1205c91c209b57`；这是真实 binding 解析，不是根据 timestamp 猜路径。

Pod2 使用两个各六条的本地 parent：

- 精度 basin：job `demo_qdot_v1v2_face_w0p2`，run directory
  `/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_qdot_v1v2_face_w0p2`；binding 解析到
  `2026-07-15_18-12-12_phase1_demo_qdot_v1v2_face_w0p2_seed3_20260716/model_4700.pt`，SHA-256
  `32adc2a816e8742df93695a866f1621af22c6454cdcdea6e0ee5fc0e6e866b17`；
- 连续 basin：job `demo_qdot_long_carry_free_arm_16s`，run directory
  `/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_qdot_long_carry_free_arm_16s`；binding 解析到
  `2026-07-15_18-37-52_phase1_demo_qdot_long_carry_free_arm_16s_seed3_20260716/model_4500.pt`，SHA-256
  `ea818f2ac5433e2736d5224c0f1a1cc686bc18608bee495a06c57f75ca2af198`；
- 两个 exact checkpoint 的完整 RSL root、hard contract、claim 与 binding SHA 都已写进机器 YAML；旧
  parent-snapshot receipt 不会冒充本轮 parent runtime attestation。

选择时的只读方向快照（还不是 stop receipt）是：Pod1 free16 `model_1600` 末窗
composite/return/completion=`.358/.393/.825`、pre/post fall=`.0645/.0281`；Pod2 qdot-face0.2 `model_4700`
pos/vel/normal/composite=`.990/.636/.512/.423`、10 cm `=.995`、return `=.415`；Pod2 16s-carry `model_4500`
composite/return/completion
`=.430/.425/.870`、pre/post fall=`.0163/.0114`。full SHA 只绑定 bytes；窗口指标仍不能替代同卷或正式排名。

三者已经按 post-stop frozen ranking 选定。continuation runner 仍必须在同一 Pod 检查 checkpoint finite、embedded
iteration、hard contract、queue claim、run binding、完整 optimizer `state/param_groups` 与当前 GPU 容量。
不得从另一 Pod 复制看似更成熟的 checkpoint 充当默认 parent，也不得用目录 mtime 猜“最新”。

## milestone、稀疏机会与判读

所有观察点是相对各自 parent 的 `+200/+500/+1000/+2000`，不是绝对 model 编号。热启动 harness 必须从 attested
embedded iteration 物化绝对路径，并在 claim/hard contract 中同时记录 parent 与 offset：

- `+200/+500`：只看 source/asset/optimizer 恢复、retime/TTS mode/jitter/Reward/foot 是否真的激活，及 non-finite、
  关节限位、跌倒等结构/安全问题；
- `+1000`：只作同一 parent 内的机制方向筛选；
- `+2000`：只形成单 seed portfolio 排名，仍不是正式因果或 vendor MuJoCo 结论。

尤其约 0.5 秒格在早期可能因为完整动作机会少而没有 hit-conditioned sparse outcome。只要没有独立结构或安全失败，
eligible exact-strike / virtual-capture / qdot-active 分母未达到冻结下限时就必须继续，不能用“稀疏 Reward=0”早停。

### 自动淘汰与把吞吐还给胜者

自动审计每 30 分钟检查一次，但不把“经常看”变成“随便停”。`+200` 只允许停止 non-finite、fatal、
parent/optimizer/claim 恢复错误或已配置 dense 机制完全没接上的结构失败；不按行为分数淘汰。`+500` 的稀疏
回球仍不得因零值停止，只有连续两个完整窗口同时出现 completion `<0.40`、pre+post fall `>0.10`，且没有
任何 dense 平衡/追踪项相对同一 parent 改善，才算明显崩坏。

`+1000` 才在**同一 parent** 内做多目标 Pareto 淘汰：最大化 completion、signed composite 与解析合法
回球，最小化 pre/post fall；只有在所有已观察目标均被另一格支配、至少两项严格更差时才停。每个 parent
至少保留两格，并全局保留至少一个约 0.5 秒格和一个随机 timing 格。停臂只对绑定的 numeric PGID 做 exact
处置；不自动 retry。这样少掉的并发会直接提高同卡幸存者到 `+2000` 的吞吐。随机横向躯干力只有完成
trainer/hard-contract/full-scene source gate 后，才可作为空槽的 matched no-force/force replacement；当前
不能用 qdot 代替。

2026-07-16 10:20 CST 的每 Pod 单连接只读审计确认：Pod1/Pod2 分别 `11 live + 1 rejected`，三卡并发为
`4/3/4` 与 `4/4/3`，22 条 live 均 PID=PGID/starttime/binding 一致、source=`704bf3a`、fatal=`0`；六卡
利用率为 `94–97%`，swap=`0`。两条 rejected 仍为首迭代前 importer malloc `rc134`，进程和 NVML context
均 absent。旧 budget-v1 诊断臂当前 `model_2200`，尚未到 `model_3600` exact-stop 门；Pod2 最快两条为
`model_5000`，尚未到该母本 `+500/model_5200`。因此本轮没有达到行为淘汰条件，22 条继续。

2026-07-16 11:29 CST 的下一轮每 Pod 单连接审计仍为 Pod1/Pod2 各 `11 live + 1 rejected`，并发
`4/3/4` 与 `4/4/3`、fatal=`0`、GPU 利用率 `89–98%`。Pod1 的旧 budget-v1 臂只到
`model_2400`，`model_3600` 不存在；Pod2 已有两份 `model_5200`，但没有对应 no-clobber milestone receipt。
这两份文件的出现不是行为门通过，也不授权 stop。

2026-07-16 12:45 CST，Pod1 的本轮唯一连接确认 `11 live + 1 rejected`、fatal=`0`，11 份 latest
`model_2100–2600` 的 embedded iteration、74 个浮点 tensor/1,762,715 elements finite、schema-3 hard、
claim/binding 与 lineage=`0` 全部一致；budget-v1 的 `model_3600` 仍不存在。Pod2 的唯一连接用于
`rolling_p2_t05_comp2_j0_equal_f03@5200` 的已注册 no-clobber attestation。它在 claim 稳定读取阶段即
fail closed：actual immutable claim 的 canonical digest 不等于按当前 YAML 与冻结 `c49dc314` runner SHA
重建的 `aee7132...`；没有 materialize attestor runtime、没有 load checkpoint、没有发布 receipt，也没有
retry/signal。训练和 checkpoint 不能因此判失败。

本地逐提交复算得到三种可能的合法历史 claim：旧 runner `27202d...` 为 `b639160...`，修正 budget runner
`428cbf...` 为 `7878d92...`，cross-Pod runner `90d7f26...` 为 `aee7132...`。runner SHA 本身写入
`continuation.continuation_runner_script_sha256`，更早版本还叠加 budget 语义差异；但失败信息没有返回
actual digest，所以“远端是旧 launcher claim”目前只是假设。禁止通过改写 claim、放宽摘要或重复
attestation 来验证。下一轮先使用 YAML 派生、严格只读、单 SSH 的 `inspect-milestone-binding`，稳定
O_NOFOLLOW 自校验 actual claim/binding、绑定进程与 checkpoint/receipt presence，并报告 actual/expected
digest、runner SHA、budget 与字段差异；只有 actual 谱系无歧义后，才能预注册新的兼容 attestor。

2026-07-16 13:46 CST，下一轮 Pod2 唯一连接执行上述 inspector 并得到无歧义结果：actual claim canonical/
declared digest 均为 `7878d92e98a10d6326b83a2bcbeb191b1da3c457e296aa9e1fbf6c5d4a27d2a9`，
`continuation_runner_script_sha256=428cbf590a9f93ce7c3c0badcda93a511661152c7f3680344457abc5f574dabb`；
binding self-valid 且 exact 指向 actual claim，PID=PGID `437711` / starttime `560049168` 仍 `live_exact`，
`model_5200.pt` 是 non-empty regular file，receipt absent。actual 与现行 `aee7132.../90d7f26...` 的
`content_diff_keys=[continuation]`；两边 budget 都是 4096 env、追加 `2001`、absolute exclusive `6701`、
milestones `4900/5200/5700/6700`、save interval `100`，逐字段相同。因此根因是 corrected-budget runner
在 cross-Pod 并发升级前已经消费该 claim，不是 queue claim 损坏、budget 漂移或训练失败。

兼容 consumer 不采用“允许一个字段 diff”的开放规则。独立 attestation contract 只登记 reviewed
`428cbf...` 与 `90d7f26...`；runner 用同一完整构造器、只替换 runner SHA，为每个 YAML job/slot/parent
分别重建两个 schema-2 claim；全 24 job 的生产合同测试断言两候选只有该叶及其派生 claim-SHA argv 不同，
remote actual 仍须**逐字段等于其中恰好一个完整候选**。preflight
输出 actual digest，reviewed runtime 用该 digest 再验 binding，防 TOCTOU；旧 budget-v1 `b639160.../
27202d...`、任意第三 runner、budget/recipe/parent/source/run/slot/bank/argv 任一漂移全部拒绝。旧
budget-v1 job 还在 contract 中显式 deny，继续等待独立 exact-stop 路径。

约 15:10 CST，production dry-run 只列出上述两个 exact variant 后，本轮 Pod2 唯一 SSH 对
`rolling_p2_t05_comp2_j0_equal_f03@5200` 执行一次 reviewed attestation 并 rc=`0`。content-addressed runtime
`ec90e18...` 以 `created_no_replace` 物化；receipt 以 O_EXCL/no-replace 发布，content SHA=`521910d...`。
checkpoint SHA=`72dbcb9...`，filename/embedded iteration 均为 `5200`，74 个浮点 tensor、`1,762,715`
elements、nonfinite=`0`；hard contract schema=`3` / SHA=`4e84c51...` / lineage=`0`，actual claim
SHA=`7878d92...`；binding content SHA=`4b9c5b2...`，且 binding 内 claim 引用与 actual 一致；取证时
process=`live`。没有 judge、stop、retry 或第二 job；该 receipt 只关闭 checkpoint 身份、finite 与 lineage，
不改变“量尺不完整，继续训练”。

同轮 Pod1 唯一只读 SSH 已改用 `/proc` 专用双读，不再把伪文件套 regular-file size/mtime 门：source clean
exact `704bf3a`，11 条 PID=PGID/starttime/argv 与 immutable binding 均 `live_exact`，一个既有 importer
malloc job 保持 rejected；GPU 并发 `4/3/4`、util=`92/97/97%`，accepted fatal=`0`。11 份 latest
`model_2600–3100` 均 filename=embedded、74 个浮点 tensor（总 tensor 76）/ `1,762,715` elements、
nonfinite=`0`、schema-3 hard/
claim/lineage=`0`、optimizer state/groups 完整。budget-v1 PGID `2199057` latest=`3100`，`model_3600`
不存在，未 signal；worker/judge/Kit process 均为 `0`。

13:46 CST 的上一轮 Pod1 审计曾证明 source clean、12/12 claim 的 job/Pod/GPU/run/source 一致、GPU `4/3/4`、
11 个 trainer PID 同时在 `/proc` 与 NVML、accepted fatal regex=`0`；但审计脚本错误地把 `/proc/<pid>/stat`
当 regular file 做 size/mtime stable-read，11 条 live 在 identity 阶段统一 fail closed。当时 exact argv/
starttime/checkpoint 因而是 UNKNOWN，不得据此推断 `model_3600` absent 或停止 PGID `2199057`；这不是 trainer
失败，下一轮需用 proc 专用双读。

同轮 source/event-schema 审计进一步推翻了“现役 `704bf3a` 可以直接执行上面的 `+500/+1000` 行为淘汰”这一
假设：completion 和 pre/post-fall 是跨全历史、逐 simulator step 衰减的 EMA，比值无法重建两个不重叠的
100-update 窗；pre/post numerator 又包含所有 `terminated`，不是 `base_fell_tilt ∪ base_too_low` 的绝对
物理跌倒 union。ready/balance tags 是瞬时全环境均值或 phase-diluted Reward，没有 recovery/hold eligible
分母与累计和；parent 末窗也没有 content-bound 同语义 receipt。因此现役 22 条的行为状态统一为
“量尺不完整，继续训练”：只允许 fatal/non-finite/合同/恢复错误等结构淘汰，不允许从历史 EMA
自动停臂或写 Pareto 胜者。

后续可比较 source 必须 consume-once 地逐 update 写整数 `swing_start`、exact completion、pre/post absolute
physical-fall union、各 termination 原因，并为 ready/hold 的倾角、接触、滑动、base speed/station offset 写
`eligible_count + sum`。materializer 再绑定 event/checkpoint/hard/claim/binding/parent SHA，严格消费两个各
100 个唯一 update 的窗口；缺步、重复、非有限或缺字段只能发布“量尺不完整，继续训练”。这些
整数不能倒灌进已经运行的 source；现役 checkpoint 若要提前排序，必须另立同题、不可变、checkpoint-bound
行为卷，不能把描述性 TensorBoard 曲线升级成淘汰依据。

## 为什么 formal-ineligible

每个 child 都从不同历史合同的 optimizer 热启动，并同时改动共享组合与 timing；`checkpoint_allow_contract_mismatch=true`
是显式机制探索，不是 fresh lineage。队列因此在顶层和每臂都写死 `formal_evidence_eligible=false`，不授权 judge、第二 seed、
晋级或真机。24 格只用于漏斗：先找“能在真实准备时间内完成动作”的方向，再为胜者与匹配对照另立 exact/fresh 预注册。

## 激活前验证清单

1. **已完成：** atomic TTS change 已进入 `main@704bf3a2`，YAML 已绑定 clean detached checkout；
   `DEFINITIONS.md`、G05、训练 operation、policy 接口与 `PROGRESS.md` 已同步。
2. **已完成：** 跑 host/Pod2 单元合同门：三种 TTS mode、delay ring wrap/reset、同一步多次 observation 不重抽、compensated TTS 精确减
   `delay_steps × 0.02 s`、uncompensated 负控不减、hard contract/metric 字段、默认 `live` 路径回归。
3. **已完成：** 对 retiming + schema-3 bank 跑 source gate；bank answer 保持
   absolute；错误 clip count、非正 scale、同时误启 event timing 必须 fail closed。
4. **已完成 exact stop：** 旧 24 条均按绑定 identity 停止且 GPU compute 为空；不额外制造不参与
   后代 provenance 的 stop receipt。
5. **已完成：** 对三份选中 parent（Pod1 一份、Pod2 两份）做一次只读 runtime attest；每 Pod 仅一条 SSH。
   三者分别为 embedded `1600/4700/4500`，均有 actor/critic 各 `8` keys、`74` 个浮点 tensor /
   `1,762,715` elements、nonfinite `0`、optimizer state `17` entries / `1` param group，且 checkpoint/hard/
   claim/binding/source lineage 均通过。parent 选择变化时先更新本实验，不在聊天里暗换。
6. **训练路径已完成：** 直接做 4096-env×2 update full-scene probe；不再串一个不能代表正式 scene 的
   512-env smoke。Pod2 0.7 秒补偿格自然 rc0，`model_1` 的 1,762,715 个浮点元素全 finite、schema-3/
   fresh lineage 通过且 fatal0。严格 finalizer 因 Popen 后第一次 `/proc` identity 读取竞态得到 null
   starttime 而 fail closed；训练本体、checkpoint 与自然退出证据均已独立复核。该基础设施 bug 另修，
   不把一次成功 physics run 重跑成“更好结果”。
7. **已完成：** 对队列做静态审计：24 unique ids/run names/run dirs；12/12 Pod；六槽各 4；四轮各 6；仅一条
   `uncompensated`，且它与 #2 除 TTS mode 外逐项相等。激活前全部 blocked、`launch_authorized=false`；上述门
   通过后只把全局状态与两个继承 anchor 切为 ready，formal-ineligible 始终不变。
8. 只有以上 receipt/测试审完并由人明确授权，才另做 activation patch；激活本身不得自动执行 `fill` 或 retry。

## 决定

- 决定：`activated / demo-only inexact engineering portfolio`
- 是否纳入当前 setting：`yes, for the 24-run overnight engineering funnel only`
- 下一个 gate：activated queue dry-run → 24 个 child 首迭代/identity/optimizer lineage

## 点火首条抓到的 resume budget 语义反例

第一条 Pod1 约 1 秒格按 runner v1 点火后，parent=`1600`、CLI `max_iterations=3601` 的真实 RSL 日志写成
`Learning iteration 1601/5201`，而不是预期 `1601/3601`。这证明 RSL 把该 CLI 字段解释为恢复后的**追加
update 数**。trainer 本身健康、binding/identity 正确且持续产生日志；本地 fill/SSH 已在远端 boot watchdog
动作前中止，避免错误 marker 把健康 PGID 当失败清理。没有重发或覆盖该 namespace。

runner v2 改为传 `2001`，plan/claim 另记 absolute exclusive bound=`parent+2001`，最后 checkpoint 是
`parent+2000`；所以剩余 23 格的首 marker 与终档分别仍是 parent+1 与 parent+2000。首条 v1 只保留为
demo-only extra-budget schedule 诊断；它在目标
`model_3600` 之前的 checkpoint 仍可看候选方向，但不能与 v2 格作 matched learning-schedule 因果比较，且到
`model_3600` 后按 exact identity 收口，不消费到 `5201`。这不是自动 retry 授权。

点火实测又表明 4096-env child 从创建到首 iteration 约需数分钟；本地逐条等待会让两台彼此独立的 Pod
无意义串行。runner 因此只在**跨 Pod**并发：每批 Pod1/Pod2 各至多一条，同 Pod 仍由 host boot lock 串行。
一个 future 失败时必须先等 sibling settle、保留成功 claim，再停止后续批次；不自动重试。首轮目前保全
四条健康 child 和一条 Pod1 importer rc134 失败；该失败没有 iteration，不能作配方负结果，也不会按同名重发。
