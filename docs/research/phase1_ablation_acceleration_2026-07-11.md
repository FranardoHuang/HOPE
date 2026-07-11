# Phase-1 消融加速制度（2026-07-11）

Status: Active. This is the scheduling and evidence policy for Phase-1 breadth runs. It does not
relax any schema, lineage, exam, simulator, or robot-safety gate.

## 目标不是“最快跑完一条”

优化目标是单位墙钟内得到最多个**可复现的因果结论**，不是让一条训练曲线尽快到终点。
因此广度消融允许同卡单臂变慢，换取更多成对实验；真正进入终审的少数赢家再转为独占卡。

以下旧规则继续有效：

- 训练固定 `4096` environments；
- 广度消融每张 RTX 5090 同时跑 `3–4` 条，常态目标是 `4` 条；历史实测四臂约占
  `22/32.6 GB`、GPU 利用率约 `87–94%`，单臂墙钟变慢 `25–45%` 可以接受；
- 同一 Pod 上所有 Kit/Isaac 启动走单一启动锁，并错峰至少 `75 s`；
- 每条训练使用独立 process group。只允许按已记录 PGID 管理本臂，禁止 broad `pkill`；
- 训练 checkout 在任一本地 arm 存活时冻结，不 pull、不切 commit、不修改；评测从独立 detached
  worktree 运行；
- 导出一次只启动一个 Kit 进程；MuJoCo BankExam 使用 CPU、`OMP_NUM_THREADS=1`，可与训练并行；
- judge/export 与 training launcher 共用 `/workspace/.kit_boot.lock`；CPU 判卷可并行，不代表两个
  Kit scene-create 可以重叠；
- 不运行真机命令。Gate 3B 仍是候选终审，不是广度消融的日常筛子。

## 五级漏斗

| 级别 | 花费 | 必须回答的问题 | 不通过时 |
| --- | --- | --- | --- |
| L0 离线合同 | 秒到分钟 | 动作/题库同 family、train/exam 内容不重叠、body/joint order、哈希、exact/inexact 声明是否正确 | 不上 GPU |
| L1 机制冒烟 | `512 env × 25 iter`，约 3 分钟 | 开关是否真的 applied、actor 维度/植体/题库/恢复合同是否生效，是否 finite | 保留日志，修完重新冒烟 |
| L2 广度信号 | `4096 env`、每卡四臂 | 单变量配对在训练内主指标和固定 checkpoint 考卷上是否产生方向一致的增长 | 明确支配的臂止损，让位给队首 |
| L3 复现确认 | 至少两个独立 seed | 增益是否大于 seed 噪声；正反手较差侧是否也改善 | 不宣称赢家；必要时补 seed |
| L4 正式门禁 | terminal + 双引擎 + 连续卷 + Gate 3B | 候选是否可迁移、可恢复并满足完整 24 切面 | 不晋级部署 |

L0/L1 是 fail-closed 机制检查，不因“曲线看起来好”而豁免。L2/L3 可以省 GPU；L4 不能省证据。

## checkpoint 抽查，而不是等终点猜答案

训练曲线和考卷曲线承担不同职责：

1. **高频 Isaac 训练内曲线**：持续看击球率、上台率、真实 rally 分母、摔倒/跟踪终止、
   reward income 和 policy std。没有开启防同步旗标的存量臂一律看至少 `21` 个 iteration 周期的
   均值，不用单点做决定。
2. **低频 immutable-exam 曲线**：训练虽每 `100` iteration 存一次 checkpoint，但长跑只在
   固定的 `1000–2000` iteration 里程碑后台导出并做 BankExam；检测到峰值、斜率反转或成对差
   突然放大时，才用相邻的 `100`-iteration 存档加密。
3. **判卷点**：准备晋级或停止一组实验前，用同一冻结 schedule 做至少 `50/侧`；terminal
   赢家再跑完整 exam、Isaac 同题、MuJoCo 单球/连续卷和 Gate 3B。

当前批次预注册抽查点：

- `16999` 起步、执行 4000 次 causal continuation：`17000` 基线、`18000`、`19000`、
  `20000`、terminal `20998`；RSL runner 的最后循环索引为 20998，不会写 `model_20999.pt`；
  所有 old/S1 配对必须在同一 checkpoint 点一起判；
- fresh from-scratch `0 -> 16999`：`0`、`1000`、`2000`，之后每 `2000` 一次直到
  `16000`，再判 terminal `16999`；`<8000` 的绝对低分只叫 immature，不能单凭低分杀掉 fresh；
- 需要加密时用固定 schedule 的相邻 checkpoint，不重新抽一套更有利的题。

每一条考卷曲线记录必须绑定：checkpoint path/SHA、相邻 hard-contract SHA、lineage exact flag、
exam-bank SHA、schedule SHA、evaluator commit、seed、实际 attempt 数及原始 ledger。causal 后代永远
标 diagnostic/inexact；只有 fresh schema-2 motion + zero-friction lineage 可以作为 exact candidate。
显式 `--allow-inexact-contract` 只是一张“允许做诊断”的票，两套 evaluator 都必须因此强制
`evaluation_contract_exact=false`，不能因底层 checkpoint lineage exact 而洗白 legacy pairing。

## 成对判读与止损

- 一个消融问题共享起点、动作 family、train bank、预算和题表；只改变预注册的一项。配对差比
  不同 run 的绝对分更重要。
- 同一 `(family, seed)` 的 old/S1 是不可拆的一对。除 hard failure 外，不准只停差的一边留下
  好的一边；否则会把墙钟和 checkpoint 选择偏差写进结论。
- 每道 immutable exam 题保留 paired outcome，用 paired bootstrap/Wilson interval 报差值和不确定性；
  只报平均回球率不够。
- **硬止损**：合同/哈希错、NaN/Inf、不可恢复 crash、开关未生效、train/exam 泄漏、结果删失或
  分母错误。先保留日志，不自动换配方重试。
- **证据止损**：至少两个相邻里程碑和至少一个 `50/侧` 判卷点都显示候选在较差侧被对照支配，
  且没有其他预注册主指标补偿，才停整对/整组。fresh 在 `8000` 前只做 hard-stop，不做低分 stop。
- **晋级**：至少两个独立 seed 的差值方向一致，较差侧不倒退，且 MuJoCo 没有把 Isaac 增益全部
  抹掉。峰值 checkpoint 与 terminal 都保存；terminal 不是自动最佳模型。
- 最少两个 seed 是保护样本，不参与第一轮淘汰；额外 seed 用来估计噪声，可在结论已经稳定后止损。

## 当前六卡的正确用法

两台 Pod 各有三张 5090，广度阶段目标布局是 `4/4/4 + 4/4/4 = 24` 个并发槽；本轮矩阵也恰好
预注册 24 个实验。现有六条只占每卡一个槽，不能称为“跑满”。但“24 个预注册实验”不自动等于
“任一时刻 24 个活进程”：先发臂若自然终档，实时并发要以 `.launch`、进程和 NOW 为准并从合法队首
回填。额外 18 条不能全部浪费成同配方重复；当前最小而高信息密度的矩阵是：

- legacy continuation：`2 motion families × 2 face pairings × 2 continuation seeds = 8` 条；
  seed 2 把 old/S1 所在 GPU 对调，以免 pairing 和 GPU 编号绑定。它们共享 historical parent，
  只能叫 continuation RNG 复现，不能叫独立 from-scratch seed；
- fresh：`2 face pairings × 2 plant settings × 4 from-scratch seeds = 16` 条，形成平衡 2×2 因子
  设计，可同时估计 face 配置效应、legacy zero-toggle 配置效应和交互项；
- 合计 `8 + 16 = 24`。不提前混入 guidance、N1、R8 或下一阶段变量。

fresh 四格缩写如下：

| 格 | face pairing | plant | 用途 |
| --- | --- | --- | --- |
| `SZ` | `shared_plus_y` | 31/31 zero friction | 当前 schema-v3 跨引擎执行合同的 formal target；不是部署 plant 候选 |
| `SP` | `shared_plus_y` | 历史 non-zero PhysX 系数 | legacy zero-toggle 配置诊断；**不是**语义修正的标定摩擦对照 |
| `LZ` | `legacy_signed_vs_A` | zero friction | face 主效应诊断；judge 必须标 inexact evaluation |
| `LP` | `legacy_signed_vs_A` | declared non-zero plant | 双旧设置诊断；judge 必须标 inexact evaluation |

四格不能只排四个均值。每个 seed 先做阻断内对比，再跨四个 seed 报 paired bootstrap 区间：

- face 主效应：`0.5 × [(SZ - LZ) + (SP - LP)]`；
- legacy zero-toggle 配置效应：`0.5 × [(SZ - SP) + (LZ - LP)]`；
- face×plant 交互：`(SZ - SP) - (LZ - LP)`。

Pod、GPU 编号和启动 layer 一起写进 ledger，作为阻断/审计字段而不是训练变量；如果某个效应只在
单一 GPU 或单一启动层出现，先按硬件/墙钟混杂处理，不宣布机制收益。四个 seed 的完整格点正是为了
把这三种效应与 seed 噪声分开，而不是把 16 条 fresh 当成 16 次互不相关的排行榜尝试。

### Plant 语义张力与必补的标定摩擦格

`SZ` 成为当前 formal 格，只因为“全零”能在 PhysX 和 MuJoCo 中以同一语义精确重放。
这不会抹掉 2026-07-07 的历史观察：零摩擦策略换到有静摩擦厂房时，virtual hit
`0.9997 -> 0.63`，跌倒 `0.27 -> 0.87`。但该次 raw artifact 未做 SHA 绑定，因而只能作为
部署 blocker/待复现假设，不能冒充摩擦标定数据或用来拟合 adapter。因此：

- `SZ` 可用来回答 face 合同/学习曲线在当前精确可重放 plant 上是否成立；
  q10 只看方向，仍必须用同 immutable schedule 做 q50 才能在这个 execution-contract
  范围内做模型选择；
- `SZ` 在标定摩擦格完成前不能晋级 sim-to-real、Gate3B 或真机候选；
- `SP/LP` 仍是“把 MuJoCo constant-Nm `frictionloss` 数字填成 PhysX 载荷相关系数”的
  历史配方，只能做旧 plant 诊断，不是 calibrated-friction arm。

因此 `SZ-SP`/`LZ-LP` 不能解释成物理摩擦主效应；MuJoCo 对 `SP/LP` 只允许 inexact
direct-number proxy。它们只回答“这组历史 zero-toggle 配置是否改变结果”，不回答哪个 engine
更接近真机，也不证明旧数值有物理标定意义。

必补的新格暂称 `SC` (shared face + calibrated friction)，不从现在 24 臂中伪造。具体合同已在
`docs/research/phase1_plant_semantics_repair_2026-07-11.md` 与
`configs/phase1_plant_semantics_repair_prereg_20260711.json` 预注册，当前状态为
`blocked_on_calibration_evidence`。它的前置是：

1. 对单关节做低速/零速、正反向、不同载荷的 breakaway/dynamic-friction 探针，分开
   PhysX coefficient 与 MuJoCo constant-Nm `frictionloss` 语义；
2. 用 A3/AGI 可验证数据拟合一个版本化的摩擦模型，为两引擎分别实现 adapter；MuJoCo 不得
   再直接数值代入 `dof_frictionloss`；
3. 把模型方程、参数、实例化逐关节值、源数据 SHA 和 engine adapter SHA 全部进 hard contract；
4. 最小训练轴固定 shared face，`Z/C x 2 fresh paired seeds = 4` 臂；每个 checkpoint 做
   `4 policy x Z/C eval plant x Isaac/MuJoCo = 16` 个同题格，q10 只 screen、q50 决策，
   复现或解除 07-07 的迁移阻断观察；
5. `SC` 才有资格进入部署 plant 选择、连续实战卷和 Gate3B；`SZ` 的当前
   execution-contract q50 不被 SC 阻断。若两引擎仍不能语义对齐，如实标 inexact，
   不因“更像真机”就放宽 formal 标签。

`training-contract lineage exact` 与“是否是本轮 formal target”不是同一个概念：fresh schema-2 motion
可使 provenance 精确绑定，但 legacy pairing 仍只能走 diagnostic judge；本轮只有 `SZ` 被预注册为
formal target 格，其他三格是同 family 的因果尺。

每次先补一层（每卡从 1 条变 2 条），验收六卡显存、GPU、host RAM、日志和 contract，再补到
3 条、4 条；每次 Kit boot 间隔 `75 s`。完整 GPU/seed 交叉布局记录在
`configs/phase1_scaleout_matrix_20260711.json`，不靠口头记忆。
一层中途失败时保留该臂目录；重跑同层只会跳过“命令逐字匹配、ready marker 存在且原 PID
仍活”的已成功臂。若较早臂已经自然结束，用 `PHASE1_ONLY_ARM=<run_name>` 精确补发剩余臂，
不需要也不允许删除成功臂状态。

本轮实测已复现旧容量规则：六张 5090 都达到四条 4096-env 并发，显存约
`22.9–23.2/32.6 GiB`、util `87–97%`，host RAM 仍有大量余量。这个数字证明“4/卡可运行”，
不等于以后任何配方都无条件塞四条；obs/asset/scene 变大时仍按 2→3→4 层逐层验收。

扩容后 checkpoint worker 只消费预注册里程碑，不对每个 `model_*.pt` 重复判卷。这样训练保存频率
仍保持恢复能力，而 CPU/GPU 评测预算集中在能改变决策的节点上。
`phase1_checkpoint_curve_worker.py --wait-for-checkpoints` 可以在里程碑出现前常驻，但只等待 manifest
里的精确路径；文件大小/mtime 稳定后才 hash，并在每次导出前重验 judge、eval checkout 和 frozen
training checkout。它不是“看到任何新文件就盲判”的目录 watcher。

### 第一池终档后的因果三角回填

原 24 臂的配方和标签保持冻结；自然终档释放的槽由第二张、独立预注册纸回填，不把新变量
伪装成原矩阵。`configs/phase1_causal_followups_20260711.json` 把之前混在结果里的两个因素拆成
`old-helper / S1-only / S1+guidance` 三角：

- M3/swing 现有 old-helper 与 S1+guidance 都是 guidance `-0.95`，故补 S1-only
  guidance `0` 的 seed 1/2；
- M2/v4rg 现有 old-helper 与 S1-only 都是 guidance `0`，故补 S1+guidance
  `-0.95` 的 seed 1/2。

四臂仍从各自原始 `model_16999.pt` 续训 4000 update，只改变上述 guidance 缺边与 seed。
parent 只有原路径/SHA 引用，**绝不**复制到新 run 的 hard-contract sidecar 旁判卷，避免把旧
checkpoint 洗成新合同 lineage。新臂只判 `17000/18000/19000/20000/20998`；每臂 launcher
验证 emitted hard-contract 后自动启动独立 q10 worker。q10 仍是 10/侧方向纸，不能停臂或晋级；
q50 是空 jobs 的 inactive template，只有预注册差值/峰终反转触发且同 family/seed 对照齐全后
才另开。M3 seed2 还绑定旧 M3-old PGID `1310472` 的只读终档释放门，launcher 永远不 signal
该前驱。

外置 launcher 位于训练 checkout 之外，按配置/脚本显式 SHA 授权；启动前重验 frozen train/eval
commit 与 clean 状态、全部父模型/动作/题库/Kit launcher/judge/worker SHA、目标 GPU 少于四个
compute/trainer 且至少 `5500 MiB` 可用、run 名未存在。它以原子 mkdir 抢占单臂目录，并只对自己
刚创建且由 sidecar+`/proc` 双重绑定的 trainer/worker PGID 做失败清理。这个回填提高因果信息密度，
不改变 causal/inexact 标签，也不授权真机。

## 把判卷当调度器，而不是事后报告

评测队列按“这次读数改变下一步决定的概率 × 能释放/避免的后续 GPU 小时 ÷ 判卷成本”排序，
而不是按 checkpoint 文件出现的先后顺序排序。落地优先级是：

1. 同一 `(family, seed, milestone)` 缺一边的 paired 对照；不齐对就没有因果差值；
2. `SZ` formal target 的跨 seed 复现点；它决定是否值得进入完整双引擎门禁；
3. 正好能确认斜率反转或止损条件的下一相邻里程碑；
4. 已明显被支配的诊断格和 terminal 归档点。

不同训练年龄的绝对分不直接互比，不能拿候选的峰值点对对照的谷值点。中间点可先用同一 immutable
paper 的固定小配额作趋势筛选；任何停臂、晋级或正式结论仍需要预注册的 `50/侧` 判卷点。若小配额
结论接近、置信区间跨零或与训练内 rally 曲线矛盾，就增加题量而不是擅自换卷。这样把 sequential
testing 的成本花在不确定边界，而不是给显然相同或显然失败的臂反复跑满卷。

卡池也按证据动态回填：四槽是并发目标，不是要求每条实验都活到 terminal。hard failure 释放的槽先
补同一配对缺口；证据止损释放的槽补预注册下一 seed/格；没有合法队首时宁可短暂空槽，也不临时发一条
无法回答问题的配方。独占卡只给已经由 paired checkpoint 曲线晋级的终审候选。CPU BankExam、训练
内曲线归并和合同审计应尽量与 GPU 训练重叠，但所有 Kit scene-create 仍经过同一启动锁。

## 连续时序是一条独立因果轴

“不传送地播完动作再换题”与“场馆节奏下随时收到下一题”不是同一个能力。2026-07-11 的保守
A-B-A 场馆审计得到同一球员击球间隔 `q10/median/q90=1.757/1.903/3.356 s`；现役 fresh
完整 clip-wrap + `hold U[0,2s]` 的理论混合分布约为 `2.90/3.75/4.60 s`。因此当前 24 臂继续只回答
pairing×plant，不把 episode length、next-task timing 或 guard 改动混进去。

消融加速的正确拆法是另开 `T0/T1` paired timing lane：T0 保持现役 clip-wrap，T1 只增加击球后
event-driven next-task/carry-state；两臂共享同一 SZ 起点、题表、plant、seed 和预算。每个 checkpoint
先过单球 q10 防精度遗忘，再过固定五拍连续小卷看第 2–5 拍斜率；任何 stop/promote 仍需要 q50。
episode 变长、tracking guard 软化、奖励修订或动作重定时若需要，分别作为后续因素，不在 T1 首轮
一起打开。完整数据口径、T1 事件合同和指标见
[连续挥拍与任意来球时序审计](phase1_continuous_rally_timing_2026-07-11.md)。

## Recovery 加速：先消融 tuple 结构，再消融同阶段 reward 组合

2026-07-12 的源码审计发现，当前 179-D deploy idle 把“新的 live-base 位置”与“上一拍速度和法向”
混在一起，而训练只出现过整套旧 tuple 或原子安装的整套新 tuple。所以当前最大的信息瓶颈是
**recovery 命令结构**，不是三个 reward 权重还不够精细。

预注册 `configs/phase1_recovery_tuple_abc_prereg_20260712.json` 因此分三阶段：

1. `D0`：只用内容绑定的现有 179 checkpoint 做 A explicit bridge 与 C complete-previous-tuple
   的 zero-shot diagnostic；不选型、不晋级；
2. `S1`：A/B/C 全部 fresh exact paired，共用相同 random-arrival 卷、题库、motion、face、plant、
   179 schema、现有 reward bytes/权重/总预算、seed、optimizer、update 和 checkpoint cadence；
3. `R2`：只有 S1 表明 B/C 缺 learned shaping 且第一拍不退化时才解锁 reward 试验。

这个次序避免了一个常见的伪加速：同时换 tuple、加 balance reward、加 ready reward 和改等待
逻辑，最后虽然分数变了却不知道哪个机制有效。A 的外部 bridge 要单独绑定逐 tick ownership：
last-action 必须是 executed bridge action 到 actor 坐标的精确投影，shadow action 只诊断；只有真执行的
actor sample 才有 logprob。bridge tick 的 policy/entropy/value loss mask 全 0，真实 bridge reward
折叠回前一个 actor option transition，并在下一 actor state 或真终止处 bootstrap。未绑定前不点火。

为防止 A 因 actor 少控制一段就暗中获得不同训练预算，S1 同时锁定每臂每 update 的 env-step、
scheduled opportunities、optimizer update、actor-owned sample、minibatch 和 epoch。B/C 多出的
actor sample 用预绑定、不看 outcome 的索引下采样；A 若不足固定样本数，整个 paired update fail，
不 padding/复用/额外多跑。评测分母仍是全部预定机会，不跟 loss mask 缩水。

### 同一恢复阶段的三项 reward 为什么要看交互

用户指出的三个职责——吸收上一拍平衡债、进入通用 ready set、在下一题随时 reveal 时可接战——
确实发生在同一段状态分布上。它们不能默认相加：快速压低速度可能帮助平衡却损害临时转向；追单点
ready 可能缩小下一动作可达集；task-readiness 又可能让机器人一直移动而无法 settle。因此 R2 不用单因素
OFAT 继续猜比例，而是：

1. 在同一批 frozen S1 rollout 上估计三项的自然尺度、方差和激活占比，先归一化量纲；
2. 跑配对 seed 的全 `2^3` presence/absence 因子设计，直接估计主效应与二/三阶交互；
3. 只当交互非零且组合值得优化，才在固定总 recovery-reward budget 上做 simplex 或 D-optimal
   mixture，不让“加了更多总分”偷渡成方法改善；
4. 每个 checkpoint 同时过冻结单拍 non-regression 与同一 no-reset random-arrival 连续卷。

跌倒、自碰、桌网碰撞、执行器越界是约束，不是可以用击球分补偿的第四项 reward。正的 hold/brake
survival income 仍禁止，避免通过 GAE 把“什么都不做”的收益传回击球段。q10 只筛方向，q50 才决定结构/
混合；最终先由智元 vendor MuJoCo Gate3 在同 C++/MJCF/plant/model 上做 first-tick+连续稳定硬前置，
再由共用同 runtime 的 Gate3B random-arrival q50 主判第一拍不退化与回球质量，不在 Isaac/MuJoCo 之间
平均。当前只完成设计合同与 `50 passed`，没有 GPU 或训练结果。

## 什么叫“加速成功”

不是“24 条都跑到终点”。满足以下四点才算：

1. 同样墙钟得到更多合法的单变量/多 seed 结论；
2. 每个结论都有随训练进度变化的考卷曲线，不靠 terminal 单点；
3. 明确失败的臂及时释放槽，但 hard failure、早停和重试都有可审计记录；
4. 终审资源只给经 paired evidence 存活的少数候选，且完整 gate 不降级。

## 同一恢复阶段的 reward 不能按单因素胜者直接相加

击球后到下一题到达前至少有三个职责：卸掉上一拍的动量并保持可生存、回到跨动作共用的
ready 集、在未知到达时刻保持对下一题的低延迟可达性。它们共享同一状态区间，确实存在交互：
过强的 pose reset 会在机器人仍有角动量时把它拉倒；只奖励不摔会学成僵住；只追 ready 帧会
牺牲对突然来球的响应；把三个各自最优的标量权重相加也会改变总 reward 尺度和 PPO 梯度。

先检验是否真的需要混权重，而不是先扫比例：

1. 把跌倒、非法接触、关节/力矩越界保留为 constraint/termination，不允许靠其他 reward 抵消；
2. 将“回 ready”写成到 ready **集合**的 potential progress（姿态、速度、base heading/位置），
   不按停留时间持续发钱，避免站着刷分；
3. 用随机 next-task probe 的成功率/响应延迟衡量“随时可打”，它首先是 evaluation target，
   只有机制证明稀疏到学不动时才加入 dense surrogate；
4. 在冻结的 post-strike 轨迹上检查三个候选项的归一化尺度、到账区间、梯度 cosine 与重复信息。
   若用 phase gate 后职责基本不重叠，就优先采用顺序状态机，省掉无意义的比例扫。

还要先把“结构选择”放到“比例选择”前面。现有论文给出两种不同且都成功的边界方案：

- Ace 的击球 RL 按单球训练；每个 32 ms segment 同时生成可行的后续 reset plan，触球或碰撞风险
  出现后由近时间最优 MPC 执行 recovery。reset 目标既可固定，也可由 prepare policy 根据下一球选择
  高 dexterity 姿态；训练初态还从历史 reset plan 的动态终态采样。它证明“strike policy + 显式
  recovery/prepare controller”是认真候选，而不是失败后的临时补丁；碰撞门必须检查 strike segment
  **连同** reset trajectory ([Dürr et al., Nature 2026](https://www.nature.com/articles/s41586-026-10338-5))。
- HITTER 走相反路线：10 s episode 内连续训练多拍，每拍完成后重采样正/反手、球拍和 base 目标；
  dense imitation/regularization 贯穿 episode，拍位/拍速/拍面只在触球短窗，base-position reward
  只在触球前，给击球后向下一目标转换留空间。它证明统一 actor 可以学出多拍 carry-state，但公开
  合同仍是 swing 完成后换题，不等于任意 mid-recovery 到达
  ([Su et al., HITTER 2025](https://arxiv.org/abs/2508.21043))。

因此首轮结构因子应是 `explicit safe bridge/prepare` 对 `learned recovery option`，而不是直接扫三项
reward 比例。若显式 bridge 已在同一随机到达卷上满足安全、ready latency 和下一拍回球率，就不必让
strike actor 同时背三份信用；若它因状态/动作族覆盖不足失败，再进入统一 option 的 factorial/mixture。
Ace 还把 reward 权重作为 skill state 并偏向采样稀疏/边界组合；这支持“条件化多目标策略”作为后续
方向，但不能被引用成某个固定权重比已经最优。

若仍有真实重叠，实验分两层。第一层用等梯度尺度的 `2^3` factorial（none、三个单项、三个
两两组合、三项全开）和 paired seeds，回答主效应与交互项；单项胜出只表示该项值得保留，
不表示组合最优。第二层只在存活组合上固定辅助 reward 总预算，做小型 simplex/D-optimal mixture，
比较比例而不同时改变总 reward；并把“顺序 phase gate”与“同窗混合”作为显式结构因子。
每个 checkpoint 报告：physical fall/guard reset、strike 后回 ready 的时间、base/heading/角动量
残差、按 next-task delay 分桶的第二拍回球率与响应延迟，以及第 2--5 拍斜率。q10 仍只筛方向；
最终必须在智元 MuJoCo Gate 3 的无 reset 连续卷上满足 `falls=0`、`rescues=0`、每次 engaged
swing 完成并恢复，Isaac 训练分不能替代这一门。
