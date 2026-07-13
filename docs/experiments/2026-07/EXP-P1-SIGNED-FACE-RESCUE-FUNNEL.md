# EXP-P1-SIGNED-FACE-RESCUE-FUNNEL：有符号拍面修复后的单-seed 机制漏斗

状态：`proposed`  
证据等级：E0（只有设计，尚无新训练）  
人类负责人：franco  
执行者：Codex  
全局优先级：只继承 [`NOW` 队列第 1 项](../../NOW.md#统一工作队列唯一优先级账本)，本页不另建队列。

术语统一见[术语与人话对照](../../DEFINITIONS.md)。这里的实验臂是一个训练配置，不是机器人的手臂；
`seed` 是随机初始化/采样种子；`checkpoint` 是训练中间保存的模型。

## 问题与第一性原理

旧零摩擦候选在正手 signed-face（保留拍面正反号的判分）上长期为零。训练中的指数拍面 Reward 在
约 `170°` 误差处几乎没有梯度，因此只修 evaluator 不会自动把 policy 从反面局部最优拉回来。
需要区分两个问题：

1. 旧 checkpoint 在诚实 telemetry 下继续训练，是否会自行恢复；
2. 默认关闭的线性拍面角度引导是否同时能救热启动，并避免从零训练再次进入反面局部最优。

首轮不需要四个 seed。机制主效应尚未成立前，seed 复制只会更精确地估计一个失败配方。最小高信息
设计是同一 seed 的“初始化方式 × 线性引导”`2×2`。

## 假设

- H1：只修有符号量尺、不开线性引导，热启动臂不会在 1000 次更新内明显脱离正手约 `170°` 的死区。
- H2：线性引导权重 `-0.4`、角度截断上限 `pi` 能给反面状态非零梯度；相对匹配对照，它会降低
  正手 signed-face 误差并提高正手有符号拍面通过率。
- H3：若 H2 只在热启动成立而 fresh 不成立，最快演示路径可继续热启动，但它不能成为 fresh baseline；
  若 fresh 也成立，才值得购买第二个 seed。

## 首轮四个机制单元

所有单元固定：seed 3、`v4rg_runtime_order_v3` 动作对、同一 schema-3 train bank、零关节摩擦 plant、
179 维观测、31 维动作、PPO 配置、4096 environments、checkpoint cadence，并只跑到相对 checkpoint
`+1000`。

| 单元 | 人话名称 | 初始化 | 唯一机制变化 |
| --- | --- | --- | --- |
| A | 热启动诚实对照 | 历史 `SZ` seed3 的已审计强 checkpoint | 线性拍面引导关闭 |
| B | 热启动拍面救援 | 与 A 完全相同的父 checkpoint | 线性拍面引导 `-0.4`，角度上限 `pi` |
| C | 从零诚实对照 | seed 3 从零初始化 | 线性拍面引导关闭 |
| D | 从零拍面预防 | 与 C 相同 seed 从零初始化 | 线性拍面引导 `-0.4`，角度上限 `pi` |

A/B 的父 checkpoint 路径、SHA、嵌入迭代、相邻 training-contract SHA 必须在 machine prereg 中逐项
冻结。C/D 的随机源和完整 launch argv 必须逐字节相同，除预注册的引导字段外不得漂移。

## 执行漏斗与 GPU 预算

1. **源码门：** signed-face 的 `n/-n` 负控、NumPy/MuJoCo/Torch 一致性和全回归进入 `main`；source
   commit 未知或工作树不 clean 时禁止启动。
2. **L1 机制冒烟：** 四单元各跑 `512 env × 25 iter`。必须在日志中看到引导 on/off 的实际 applied
   值、`theta_max=pi`、有符号正反手指标、finite checkpoint 与相邻合同。任何一项缺失都不上 L2。
3. **L2 单-seed canary：** 一张 RTX 5090 四槽并发，各跑到相对 checkpoint `+1000`；按 Pod 级
   Kit lock 错峰启动。fresh 用 `max_iterations=1001`，保证 0 起数的 runner 真正写出
   `model_1000.pt`；热启动也用 1001 次调用预算，并把偏移加到父 checkpoint 迭代号。保存相对
   `+200/+500/+1000`，不跑旧 17000-iteration terminal，不增发 seed。
4. **判读：** `+200/+500` 只看训练内 paired 曲线和 hard failure；`+1000` 才在同一 immutable 小卷上
   做每侧固定题量的方向筛。所有单元到 1000 自然结束，不把小卷当正式晋级分。
5. **L3 解锁：** 只有 B 相对 A 或 D 相对 C 在正手 signed-face、较差侧综合命中和安全三者都不退化，
   才给该胜者**连同匹配对照**补第二个 seed。没有第二 seed 复现，不买第三、第四 seed。

其余 GPU 优先给已过各自离线安全门的 v12 视频动作预处理、同卷导出/评估或 `NOW` 队首的其他独立
机制；不得为了显示利用率而复制 A–D。没有合法输入时允许空闲。

## 指标与决策规则

每个 checkpoint 全量记录：

- 正手/反手 signed normal error、normal-pass、position-pass、velocity-pass 与 composite；
- 修正后 virtual-return telemetry；它只作诊断，不能替代 signed composite；
- root fall、guard reset、异常接触、KL、value loss、policy std、NaN/Inf/OOM/Traceback；
- checkpoint 文件名迭代 = 嵌入迭代、所有浮点 tensor finite、checkpoint ↔ 相邻 hard-contract SHA、
  source commit、launch argv、父 checkpoint lineage。

硬失败（合同/SHA 漂移、非有限值、开关未 applied、错误 exactness、crash）立即保留日志并停止本臂；
只允许在配方和合同不变、根因明确且安全时精确 PGID 重试。

L2 的“值得买第二 seed”必须同时满足：

1. 引导臂相对匹配对照在相对 `+500` 和 `+1000` 两个点的正手 signed normal error 都更低，且 `+1000` 的
   改善不少于 `20°`；
2. `+1000` 小卷正手 signed composite 至少从零变为非零，反手和 position/velocity 较差侧不出现
   超过 `10` 个百分点的绝对退化；
3. root fall 不增加，且没有新的 guard-reset/异常接触集中失败。

这些阈值只决定是否购买复现，不是 accepted-baseline 阈值。L3 后仍须 q50、fresh stability、厂商
MuJoCo Gate3/Gate3B 和连续卷；A/B 的热启动结果永远不能洗成 fresh 证据。

## 发射前缺口

- signed-face 修复尚未以最终 source SHA 进入 `main`；
- machine prereg、validator、no-clobber launcher、四条完整 argv 和 checkpoint worker manifest 尚未物化；
- 热启动父 checkpoint 的 Pod 路径需要重新只读核对；
- 尚未决定相对 `+1000` 点小卷的 exact schedule 文件与 SHA。

以上任一缺口存在时保持 E0/`proposed`，不授权训练、judge、部署或真机。
