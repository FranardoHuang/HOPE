# EXP-P1-ROLLING-TIMING-SUPERCOMBO — 快速准备、时间戳补偿与组合训练漏斗

- 状态：`blocked`（source/full-scene training path 已过；等待 parent runtime attestation 与 continuation runner）
- 阶段/轴：阶段 1，动作重定时、actor 可见目标/TTS 延迟、预测收敛抖动、击球 Reward 配比与脚朝向
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E1`（训练源码与增量回归；尚无本轮 4096-env/行为结果）
- 创建日期/最后复核日期：2026-07-16 / 2026-07-16

共享术语按[术语与人话对照](../../DEFINITIONS.md)解释。本实验的机器草案是
[`phase1_rolling_timing_supercombo_20260716.yaml`](../../../configs/phase1_rolling_timing_supercombo_20260716.yaml)。
顶层 [`launch_authorized=false`](../../DEFINITIONS.md#launch-authorized)，24 条 job 也全部为 `blocked`；本文没有
授权启动、自动重试、第二 seed、判卷、晋级或真机。

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
5. 对三份选中 parent（Pod1 一份、Pod2 两份）做只读 runtime attest；验证完整 optimizer、finite、contract/claim/binding 和
   source lineage。parent 选择变化时先更新本实验，不在聊天里暗换。
6. **训练路径已完成：** 直接做 4096-env×2 update full-scene probe；不再串一个不能代表正式 scene 的
   512-env smoke。Pod2 0.7 秒补偿格自然 rc0，`model_1` 的 1,762,715 个浮点元素全 finite、schema-3/
   fresh lineage 通过且 fatal0。严格 finalizer 因 Popen 后第一次 `/proc` identity 读取竞态得到 null
   starttime 而 fail closed；训练本体、checkpoint 与自然退出证据均已独立复核。该基础设施 bug 另修，
   不把一次成功 physics run 重跑成“更好结果”。
7. 对队列做静态审计：24 unique ids/run names/run dirs；12/12 Pod；六槽各 4；四轮各 6；仅一条
   `uncompensated`，且它与 #2 除 TTS mode 外逐项相等；全部 blocked、`launch_authorized=false`、formal-ineligible。
8. 只有以上 receipt/测试审完并由人明确授权，才另做 activation patch；激活本身不得自动执行 `fill` 或 retry。

## 决定

- 决定：`blocked / preregistration draft only`
- 是否纳入当前 setting：`no`
- 下一个 gate：per-Pod parent runtime attestation + continuation runner validate/dry-run
