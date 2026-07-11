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

- `16999 -> 20999` causal continuations：`17000` 基线、`18000`、`19000`、`20000`、
  terminal `20999`；所有 old/S1 配对必须在同一 checkpoint 点一起判；
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

两台 Pod 各有三张 5090，广度阶段目标布局是 `4/4/4 + 4/4/4 = 24` 条训练。现有六条只占
每卡一个槽，不能称为“跑满”。额外 18 条不能全部浪费成同配方重复；当前最小而高信息密度的矩阵是：

- legacy continuation：`2 motion families × 2 face pairings × 2 continuation seeds = 8` 条；
  seed 2 把 old/S1 所在 GPU 对调，以免 pairing 和 GPU 编号绑定。它们共享 historical parent，
  只能叫 continuation RNG 复现，不能叫独立 from-scratch seed；
- fresh：`2 face pairings × 2 plant settings × 4 from-scratch seeds = 16` 条，形成平衡 2×2 因子
  设计，可同时估计 face 主效应、plant 主效应和交互项；
- 合计 `8 + 16 = 24`。不提前混入 guidance、N1、R8 或下一阶段变量。

fresh 四格缩写如下：

| 格 | face pairing | plant | 用途 |
| --- | --- | --- | --- |
| `SZ` | `shared_plus_y` | 31/31 zero friction | formal target candidate；四个从零 seeds 都保留 |
| `SP` | `shared_plus_y` | declared non-zero plant | plant 主效应诊断；即使 provenance 可精确重放，也不替代 `SZ` 目标格 |
| `LZ` | `legacy_signed_vs_A` | zero friction | face 主效应诊断；judge 必须标 inexact evaluation |
| `LP` | `legacy_signed_vs_A` | declared non-zero plant | 双旧设置诊断；judge 必须标 inexact evaluation |

`training-contract lineage exact` 与“是否是本轮 formal target”不是同一个概念：fresh schema-2 motion
可使 provenance 精确绑定，但 legacy pairing 仍只能走 diagnostic judge；本轮只有 `SZ` 被预注册为
formal target 格，其他三格是同 family 的因果尺。

每次先补一层（每卡从 1 条变 2 条），验收六卡显存、GPU、host RAM、日志和 contract，再补到
3 条、4 条；每次 Kit boot 间隔 `75 s`。完整 GPU/seed 交叉布局记录在
`configs/phase1_scaleout_matrix_20260711.json`，不靠口头记忆。
一层中途失败时保留该臂目录；重跑同层只会跳过“命令逐字匹配、ready marker 存在且原 PID
仍活”的已成功臂。若较早臂已经自然结束，用 `PHASE1_ONLY_ARM=<run_name>` 精确补发剩余臂，
不需要也不允许删除成功臂状态。

扩容后 checkpoint worker 只消费预注册里程碑，不对每个 `model_*.pt` 重复判卷。这样训练保存频率
仍保持恢复能力，而 CPU/GPU 评测预算集中在能改变决策的节点上。

## 什么叫“加速成功”

不是“24 条都跑到终点”。满足以下四点才算：

1. 同样墙钟得到更多合法的单变量/多 seed 结论；
2. 每个结论都有随训练进度变化的考卷曲线，不靠 terminal 单点；
3. 明确失败的臂及时释放槽，但 hard failure、早停和重试都有可审计记录；
4. 终审资源只给经 paired evidence 存活的少数候选，且完整 gate 不降级。
