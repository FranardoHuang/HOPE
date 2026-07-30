# NOW — 当前训练流程、课程阶段与下一步

`origin/main` 最近复核：2026-07-30 CST。本页说明现在到底在训什么、整套训练怎样连起来、每个课程阶段在解决
什么问题，以及下一项工作为什么值得做。实验过程放在[实验登记册](experiments/README.md)，
复现命令和 Gate 结果放在对应 [Gate](gates/) 与操作文档。

`origin/main` 上的本页才是运行态权威。功能分支里的修改只是一份提案；合入前必须重新对账最新的
`NOW/TIMELINE/PROGRESS`。缩写和机器代号的完整释义见[术语与人话对照](DEFINITIONS.md)。

## 功能分支候选提案（2026-07-27，尚未进入 `origin/main`）

> 本节不改变下文已采用 setting、统一工作队列、优先级或人的责任归属；合入 `main` 前必须逐项
> 与最新主板对账。它只记录本分支准备交审的训练边界。

- executor 候选改为
  [按动作条件化的 `ball-first`](interfaces/action_conditioned_ball_first_contract.md)：训练日程先
  均衡选定并冻结动作，再从该动作自己的到球时间、来球、base 与落点域采样，最后由
  fixed-action solver 解出物理自洽 task 和认证 teacher rate。训练期不运行 selector；旧
  `task-first` 只保留为历史消融。schema v3 把 lower/upper 与方向 tangent 正负侧拆成 32 个
  curriculum arms（`no_move` 有效 28 个），每动作先做 center，再分别找 marginal frontier，
  最后用 joint `rho` 把 safe closed policy failure 调到 10% 目标带。最近 100 次只安排下一个
  候选方向，不能替代 frozen canary 与独立 heldout 晋级。
- 候选五动作视图为 `bh_loop_c / fh_block_syn / bh_block / s0_highpress / fh_loop_high`；
  历史旧正手 bytes 不删除，但不再进入新训练 view。新正手尚未获得正式 `t_hit`、`t_cycle`、
  site strike speed、无桌碰和 Pod Isaac smoke 证据，故 manifest 必须保持
  `training_authorized=false`；站远只能通过整动作、task 与 base 一起沿负 X 平移后重验。
- planner 候选改为读取逐动作 held-out capability artifact，对任意击球目标依次执行硬安全、
  support/OOD、校准成功率下界、同 `delta_tie` 内球质/priority 与明确 abstain。训练 Reward 不能
  直接跨动作比较；球质必须是同一部署 utility/calibration。当前生产仍是 schema-4 两侧路径，
  任意 N selector 尚未接线；详见
  [动作能力 selector 合同](interfaces/action_capability_selector_contract.md)。
- 当前训练效果好的原因仍未归因。运行时实际 Reward 主项是 `4.0/0.5/0.5`，不是旧设计表的
  `393.4/295.1/229.5`；现象更符合“球/task 自洽、问题熵较低”的解释，也可能混有 free solver
  总挑容易方向的偏差。两条配置逐字相同的 seed0 banked run 最近 100 窗都约 49.7%，但同配置
  seed1 为 0%；这证明好现象可复刻，却远未稳定。现有 landing Reward 与报表还共用 analytic
  oracle，且 run 缺 effective-recipe/run-binding、table on/off 甚至共用旧 hard-contract SHA。
  必须先做同边际的 paired-vs-shuffled ball/task 配对 canary，再做
  [同 action/ball/base/aim proposal tape 的 free-vs-fixed solver A/B](experiments/2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md)，
  再在 fixed solved tape 上做
  [Reward A/B](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)，不能把两种因果问题
  混在一臂。

## 2026-07-30 ActionBall 下一条 fresh observation setting

- 当前 Pod1 三条 N1 diagnostic 继续绑定 exact `f2c54fc3`、frame-consistent 194-D
  `action_ball_table_pose_twist_heading_task_n1` 与旧 physics profile；它们不因 `main`
  更新而重标、续成新合同或成为部署证据。
- 下一条 fresh ActionBall 采用
  [`action_ball_table_pose_twist_heading_task_teacher_start_n<N>`](DEFINITIONS.md#action-ball-teacher-start-contract)：
  在 frame-consistent task 与 final action one-hot 之间显式加入
  `time_to_teacher_start_s=max(pre_swing_wait_s-task_age_s,0)`，总宽 `194+N`
  （N1=195、N5=199）。这是 phase-governor 的直接可观测量，不要求 policy 从剩余击球时间、
  目标拍速与动作身份反推；不做学习 A/B，但 fresh launch 前必须做 Pod tensor/reset/构造
  parity，并使用新 hard contract、action-set SHA、claim 和 no-clobber namespace。
- 当前阶段、后续 formal N5/N73/deploy 最迟闭合项，以及必须由人/硬件提供的信息，统一见
  [ActionBall 分阶段准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)；
  本节只记录已采用的下一条 fresh setting，不另建竞争工作队列。

## 先看结论

- **当前仍在课程阶段 1：题目使用固定目标站位、固定击球位置和无旋正反手。** 来球速度会变化，
  policy 的下半身没有锁死；训练中动作可以不传送地重复，但这不等于连续不同来球。现有可信
  成绩只有每题重置的单拍，连续格仍未测。
- 最接近正式目标的零关节摩擦配方已在第 2000 与 4000 次迭代各跑完四个独立
  初始化同卷。2k 为 `83、100、100、20`；4k 为 `50、88、98、0`，稳定性在两个
  milestone 都失败。4k seed4 有 21 次物理 root fall，不是晚熟。
- 当前成绩卡是 **Python BankExam 单拍解析诊断，不是 Gate3**。4k seed2/3 的旧分虽是
  `88/98`，但正手 raw-A 法向误差为 `172.33°/174.35°`，signed composite 都是
  `0/50`；这证明旧 scorer 的正手符号盲区，所以没有新 baseline 晋级。fresh 广度池
  最近 24 个 K20 格子又全部正手 signed composite=0，因此 16 臂已分两波全部保留证据并
  停止；这不改写 q10/q50 合同。`n/-n` 负控和 signed-face source gate 已进入 main；下一项会改变
  阶段 1 判断的证据是一个 seed 的四格机制 canary，以及修正 scorer 后的同卷结果。
- 部署侧的 planner-policy exact tuple 源码已通过 portable Release 和 latest-main 本地回归；
  这解决了同 tick base/target 因果配对与源码可构建问题，但 ROS/Jazzy/AimRT、backend first tick、
  厂商 MuJoCo 行为和真机仍未运行。
- 动作库的功能分支正在重建五动作 × upper/full 十件 direct shared-ready 候选：
  `adv2c3` 只作比较，正手挡改用七关节拍面流形。它**没有改变当前
  `v4rg_runtime_order_v3` 训练 setting**，目前仍是 `0/10` 获训练授权；细节只看
  [正式化实验](experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)，优先级只看下方
  [统一工作队列](#统一工作队列唯一优先级账本)。P1IU/CGF 发射冻结未解除。
  07-24 晚已续完中断的 marker-authority v2 迁移（窗口 seed=legacy ge80、名义空挥事件不再
  混叠成窗口）并实现 §5.1 受保护窗口摘要模块；`t_hit<=0.5s` 已按 Franco 07-25 裁定从硬筛门
  降为参考值。face-neutral ready v2 组合候选（中立右臂挑战者 + G1 平足落地）已发布为
  diagnostic。07-25/26 深夜：十件 probe 成品交付、冒烟训练 800 iter 跑完（管线 PASS、学习
  信号待 5000 iter 延长冒烟裁决）、源平滑修活后正手窗开 1.33→0.69s（各件时间见实验 §7.3
  逐件最优表）、probe 消融首波队列见实验 §10.1.1。
- **合并警告 → 已解除并合 main(Franco 2026-07-28 裁定)**:legacy 空信任集问题由
  admission v2(`7805b2c8`)按"门限定到 canonical 消费路径"一案落地——默认 `motion_file`
  通道直读 NPZ 字节不过信任集,canonical/action_ball 正式路径的证书门保持 fail-closed
  (两个 action_ball 信任集仍为空 = 课程正式发射的现行阻断,见下方 07-28 运行态更新)。
  14 条现役臂全部在该代码上正常构造与训练,行为证据成立。

## 2026-07-28 运行态更新(Fable 值守,Franco 全程裁定)

- **Ball-first banked 舰队 14 臂在训**,代码谱系统一(pod1 `fivebank2` @`8ba15f38`,pod2
  `chingmu101` 分支含同两笔多 clip 修复;逐进程 cwd 已核)。矩阵:两组(五动作 4-clip upper /
  ChingMu 73 件全库)×{move 旗舰、nomove、qhigh(冻结质量表 393/295/229 首次真正生效,≈98×)、
  combo(landing 791.9+face 引导 −0.08)} + 五动作 full(单 clip)/加速 [0.5,1.2]/qmid(40/15/10)/
  penlight + 101 land04(landing 剂量 1648.8/791.9/400 三档)。**淘汰制**:每臂 +5000 iter 按
  legal/strike 与摔率裁尾腾卡。`table_hit_penalty=-1800` 与 `death_penalty=-1800` 是当前默认
  (撞桌=死亡级重罚+reset,Franco 07-28 确认)。动作集入库:`assets/motions/`(fivebind 六件 +
  ChingMu 73+1 件含实测 v_in 的 manifest)。
- **早期读数(单窗,≈6k iter)**:f5_upper legal/strike 0.075(反手挡 18%、反手拉 9% 扛旗,
  正手拉 0——疑 07-26 拍面死区,combo 臂将确诊;s0 解多在老师挥速锥外=通用速度带与动作
  不匹配,匹配带 v2 题库已排队)。昨日终档对照:单 clip banked 基线 25k 收官 legal/strike
  ~0.5,uniform 对照 0.0,bank+seed1 崩塌("不打球"收入更优)——题库必要不充分。
- **动态课程(默认候选)**:代码全绿 462 测试——单段 uniform 采样(Franco 两次裁定)、
  球出生具名拒绝门、新增带 rolling-30 环、base_spawn z 合同修复、receipt 校验对齐、可执行
  pin 工具 + f10/f20 manifest 逐字节可复现。**发射阻断在信任根**:A 路线也无法正门签发——
  晋级证书链缺五类构建器且 73 件 npz 不合 canonical-ready 端点合同(要重过编译链),评测
  receipt 三字段全仓无定义;两信任集保持空集,未手编(停点全录于解锁线报告)。
  另:环 n=30 + z=1.96 下 f10(10%±2.5pp)的 expand 统计上不可达(0 失败的 Wilson UCB
  ≈0.114>0.075),环长或判据待 Franco 裁定。reference 一致性终止(anchor/ee)按 Franco 裁定
  将在课程扩张阶段起默认关闭(安全终止保留)——随信任根解锁波一起落地并重钉 pin。


## 1. 当前一套训练是怎样完整跑起来的

动作、题目、观测、Reward、PPO 和判卷不是几根互不相关的配置轴。它们按下面的顺序组成一套
完整训练配方（setting）；其中任何一层改变，都会产生一套新配方。

1. **先出一道题。** 阶段 1 的题目规定用正手还是反手、来球速度，以及击球时球拍应达到的
   位置、速度、拍面方向和时刻。训练题与正式考试题分开。
2. **再给一段全身参考动作。** 当前正手和反手参考来自 `v4rg_runtime_order_v3` 动作对。它提供
   全身姿态和时序参照，不直接等于这道球题的答案；现役 Reward 只模仿其中的上半身。
3. **把“机器人现在怎样”和“本题要什么”一起给 policy。** 当前 179 维输入包括机器人状态、
   上一步动作、参考动作、目标球拍位置/速度/拍面、距击球时刻还有多久，以及正反手类型。
4. **policy 每 20 ms 输出一次动作。** 输出是 31 个目标关节位置，不是力矩或球拍轨迹；目标先过
   关节范围保护，再由 Isaac 的关节控制器执行。
5. **Isaac 同时推进 4096 个机器人。** 每个环境使用同一规则但不同采样；当前配方把 31 个关节
   摩擦设为零，以便和现有独立考卷合同对账。这不是标定后的真机物理。
6. **Reward 告诉 policy 哪种行为更好。** 一部分模仿上半身参考动作，另一部分分别检查击球时
   球拍的位置、速度和拍面；下半身可以按题调整，脚步正则与硬保护限制明显坏姿态。
7. **PPO 根据这些得分更新 policy。** 当前正式训练从随机初始化开始，步数目标 2 万–2.5 万次
   迭代（Jiayi 最新版口径）；本页成绩卡考的是第 2000 次迭代保存的 checkpoint。
8. **训练外再判卷。** checkpoint 先进入独立 Python BankExam。质量晋级必须先有可信判分和
   跨初始化稳定性，再过厂商 `Gate3/Gate3B`；只检查运行链能否接通的 `Gate3-D0` 可并行推进，
   但不产生质量晋级。

一句话概括：**题目给出本拍答案，参考动作给出姿态和时序参照，179 维输入把两者交给 policy，
policy 输出 31 个关节目标，PPO 根据 Reward 改进它，独立考卷再检查实际行为。**

## 2. 课程、连续能力和部署验证是三条不同的线

### 2.1 课程阶段：机器人正在多学会哪一种球技

1. **阶段 1：固定点养成。** 固定站位、击球位置、目标落点和无旋条件，只改变来球速度。
2. **阶段 2：虚拟球的不同到达状态。** 球到达击球区域时的位置、速度、方向和高度开始变化；
   近球可以只用手臂，远球允许移动站位和脚步，也可调整整套动作的朝向、时序和拍面。
3. **阶段 3：物理球进入训练。** 球在仿真中真正从对面发出、飞行、弹台并与球拍接触；先做
   无旋或低旋，再加入旋转。旋转是否以后单列阶段 4，尚未决定。

所以，**阶段 2 大体是“适应不同到达状态”，“动起来”是阶段 2 的一种解法；阶段 3 的关键变化
是物理球真正进入训练。**

### 2.2 连续能力：每个课程阶段都要另考

阶段 1、2、3 都应分别报告单拍和连续成绩。连续能力必须在同一条不重置的序列中同时做到：

1. 上一拍结束后吸收身体的角动量和失衡，不摔、不滑、不撞桌；
2. 进入允许范围内的等待动作或准备姿态；
3. 下一球在任意有效时刻出现时，能在截止时间前启动合适动作。

组件消融只解释贡献。只收好上一拍、只模仿等待姿态，或只在固定时刻能启动下一拍，都不算完成。

### 2.3 部署验证：不是课程阶段

- Python BankExam 测的是 policy + Python 评估器 + BankExam 的 MuJoCo 机器人动力学配方，
  不包含 planner、消息、生产 C++ runner 或完整厂商运行时。
- `Gate3` 把 planner、消息、生产 C++ runner 和厂商 MuJoCo 串起来，先看整链能否稳定完成。
- `Gate3B` 复用同一完整运行链，再正式记录击球、回球和连续质量。
- 真机还要另过 [G07](gates/G07_mujoco_to_real.md) 的安全门。

## 3. 共用前提：先保证尺子可信

这不是课程阶段。所有阶段都依赖三个共同条件：

- **文件和接口对得上。** 当前动作、训练/考试题、179 维排列、checkpoint 和导出合同已经做
  内容绑定，不一致就拒绝继续；但合同一致只说明文件对上，不说明物理对上。
- **判分不骗人。** 2k/4k 的正手原始拍面误差多在 165–174°，旧解析判分器又会
  自动翻转法向；4k 已出现 parsed `48/50` 但 signed composite `0/50` 的同 checkpoint 反例。源码层
  `n/-n` 负控和有符号误差已通过；行为通过条件仍要求 fresh canary 和同一考卷重跑。详见
  [拍面判分复核](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- **机器人物理能外推。** 当前零关节摩擦只便于复现；历史非零摩擦数字存在单位/语义问题。
  通过条件是从真实数据拟合共享动力学参数，再分别用 Isaac、MuJoCo 适配器和厂商运行时验证。

当前只有第一项较完整；后两项未过。因此任何解析高分都不能直接称为物理回球或真机基线。

## 4. 阶段 1：固定点单拍基线（当前阶段）

### 4.1 目标和离理想还有多远

机器人站在固定位置，用指定正手或反手，在固定击球位置把不同速度的无旋来球打向固定落点。
它要在保持平衡的同时，让球拍在正确时刻达到正确位置、速度和朝向。

- **研究推进最低线：** 有效同卷中每个动作的解析回球率至少 50%，只用于判断训练管线是否值得
  继续扩题，不是部署质量线。
- **正式候选质量目标：** 跨独立初始化稳定达到每动作约 80%，同时拍面误差 p90 小于 15°、
  落点误差小于 0.3 m、零摔，并且判分尺可信。

现有结果只在平均解析回球率上越过最低线；初始化稳定性、拍面、落点毕业指标和连续能力都未过。

### 4.2 动作：身体应该怎样挥

**问题：** 只给球拍终点，机器人可能用不安全或不可连续的身体动作碰到目标；死跟参考动作，
又可能无法满足题目要求的拍面和拍速。

**当前解法：** 使用经过关节/刚体顺序整理的正反手动作对
（`v4rg_runtime_order_v3`）。触球时由题目给出的球拍目标主导，其余时段保留上半身动作模仿；
下半身不再被参考动作锁死。

**效果与差距：** 这套动作已进入当前训练和考卷合同，能产生可比较结果；但没有可信证据证明它
优于其他动作。Franco/v6/v7 新动作只完成离线安全和重定位检查；V5 专业动作有历史训练与诊断，
但没有在当前严格同卷下获胜。两类都未进入现行配方。

### 4.3 观测与输出：policy 知道什么、控制什么

**问题：** policy 看不到本题要求的拍面就无法按题调整；训练和部署的输入排列不同也无法迁移。

**当前解法：** 179 维输入在旧 175 维部署输入后增加目标拍面法向 3 维，并预留旋转参数 1 维；
输出固定为 31 个目标关节位置。严格导出器和加载器检查维度和来源。

**效果与差距：** 当前模型文件及负例已通过严格装载检查，但厂商运行链还没有当前 179 维模型的
首个有效控制周期；目标拍面通道的真实延迟/噪声也未与其他目标通道完全同处理。

### 4.4 Reward：每一组分数在教什么

**问题：** 只模仿动作可能“挥得像”却没达到题目要求；直接用旧解析回球分训练又可能鼓励 policy
钻拍面判分漏洞。

**当前解法：** 现役 Reward 分为四组：

- **回答这道球题：** 球拍位置匹配 `14`（宽度 `0.20 m`）、速度匹配 `10`
  （`1.0 m/s`）、拍面匹配 `5`（`0.3 rad`）；三个连续核的乘积另有击球奖金 `5`，不是三项
  过阈值后的二值分；接近目标的进度奖励为 `10`。
- **保持可用动作形态：** 只模仿上半身的参考位置、朝向和速度，各项权重 `1`；下半身模仿已解除，
  让腿可以按题调整，球拍手腕的朝向模仿也按现行开关释放。
- **等待和收拍：** 等待时接近准备状态的正奖励为 `2`，只在目标仍可原地够到时生效。
- **压制坏动作：** 包括脚滑/拖脚/过快、过度伸臂、腰部扭转、击球窗倾斜/角速度/脚速度/上下
  晃动、力矩饱和，以及直立、关节速度、动作变化、关节限位和异常接触等正则项；脚朝向惩罚由
  当前发射配方设为 `-0.3`。

此外，现役 `HOPEPingPongVirtualBall` 仍启用由**实际执行球拍状态**推演的解析 outcome Reward：
解析过网 `20`、解析落点 `30`、解析旋转 `5`。`vb_metrics_only=true` 不会关闭这三个 task 自带项；
它与本 task 已开启的 `virtual_ball=true` 是冗余 OR 条件。解析过网和落点都有完整合法回球前的稠密部分分，所以它们是训练
shaping，不是独立物理裁判。`physical_ball=true` 的 Phase-A engine-integrated 来球诊断目前才是
**只记指标、不进 Reward**：位置由 PhysX 积分，场馆 aero/table bounce 由代码驱动；它没有开启拍球冲量，
球会穿过机器人。减速入位、额外拍面引导和额外球拍引导当前都关闭。
目标关节范围裁剪和摔倒/跟踪终止另属硬保护，不靠其他 Reward 抵消。详见
[Reward 真值审计](experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md)。

**效果与差距：** 有的独立初始化达到 100/100，另一个只有 20/100。结果属于整套配方，不能归因
到某一个 Reward 项。拍面尺和稳定性解决前冻结 Reward；理想状态还需用单变量配对证明各项贡献，
并验证融合后不伤平衡和连续恢复。解除左侧非击球臂模仿、让它参与平衡的单-seed A0/A1 配对已经
完成 checkpoint/合同/lineage 配对闭环；但 signed K100 尚未判卷，因此不属于已采用 Reward，也不再
写作运行中；见
[非击球臂实验](experiments/non_striking_arm_imitation_ablation_20260713.md)。

### 4.5 时序与连续：一拍结束后怎么办

**问题：** 总从同一静止姿态开始，只会学到“摆好再挥”；动作结束就传送回起点，也学不会收拍。

**当前解法：** 每局 10 秒；25% 从站立等待开始，25% 从历史收拍状态开始；挥拍前随机等待
0–2 秒；动作循环时不传送机器人；目标有 2 个控制周期延迟，并加入白噪声和连续相关噪声。

连续实验按三层推进：`T0` 只在完整挥拍周期结束后换题；`T1` 在事件时刻揭示下一题，不传送、
不清历史且冻结 Reward；只有 `T1` 仍学不会时，`T2` 才增加平衡债和准备状态引导。

**效果与差距：** 当前训练见过更多起始状态，也能在完整动作后保留机器人状态；但 BankExam 每题
仍单独重置，没有连续成绩。旧机器预注册与上述次序不一致，在配置、校验器和操作文档同步前禁止
启动。详见[等待/恢复实验](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md)。

### 4.6 训练方法和机器人动力学

当前使用 `rsl_rl` 的 PPO、4096 个 Isaac 环境和 50 Hz 控制，从随机初始化训练。零关节摩擦只为
当前执行合同对账；训练配置虽打开来球轨迹真值仪，现有 Reward 和成绩卡都没有采用真实
球拍—球物理接触结果。Reward 中的过网/落台是 achieved 球拍状态驱动的解析推演，不能改名为物理结果。

原生 MuJoCo 微调是减少 Isaac→MuJoCo 迁移损失的候选训练引擎，不是 Gate3。现有候选仍有动作
保护、源码闭包、严格 JSON 和 MJCF 路径四个正确性缺口，没有可信训练环境、并行接口、PPO 冒烟
或微调结果，当前禁止合入。详见
[原生 MuJoCo 训练实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)。

### 4.7 当前成绩：Python BankExam 单拍解析诊断（不是 Gate3）

这张卡回答：**在每题重置的 Python MuJoCo 评估器中，机器人是否触发倾倒，以及击球时产生的
实际球拍状态被当前解析模型判为接触/合法落台的比例。** 它不等于球拍位置、速度、拍面三项全部
命中，也不回答完整部署链是否真的把球打回去。

实际链路是：四个独立初始化的第 2000 次迭代 checkpoint → 同一张 100 题考卷（正手 50、
反手 50，每题重置）→ Python 在 MuJoCo 中推进机器人 → 击球时读取球拍状态 → 解析模型推算球
结果。它虽然加载同一份厂商 MJCF 源文件，但使用不同的 Python runner 和 BankExam 动力学配方；
同一 XML 不等于 Gate3 运行时。

- **解析接触判定：** 拍中心距题目接触点小于 9.5 cm，且自动定向后的接近速度大于 0.3 m/s。
  自动定向正是当前拍面正负号风险所在。
- **解析合法落台推算：** 解析接触成立后，用数学接触冲量和 RK4 飞行推算球过网，并首次落在
  对面台内；全过程没有 MuJoCo 球碰撞。

| 动作 | 单拍：MuJoCo 机器人未触发倾倒 | 单拍：解析接触判定 | 单拍：解析合法落台推算 | 连续：未倾倒 | 连续：解析接触 | 连续：解析合法落台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | `200/200 = 100%` | `137/200 = 68.5%` | `133/200 = 66.5%` | **未测** | **未测** | **未测** |
| 反手 | `200/200 = 100%` | `170/200 = 85.0%` | `170/200 = 85.0%` | **未测** | **未测** | **未测** |

判读边界：

- “未触发倾倒”只检查 MuJoCo 中机器人根高度和倾角，没有球物理，更不是真机不摔。
- 四个初始化的解析合法落台数为 `83、100、100、20`；平均数掩盖最差初始化，稳定性结论为失败。
- 所有尝试最后都经过非倾倒的跟踪保护结束，不能据此证明收拍或连续稳定。
- 正手 66.5% 低于约 80% 的正式候选目标；反手平均 85% 也因最差初始化只有 40% 且拍面尺有问题
  不能晋级。当前也没有可信的拍面 p90、毕业用落点误差或连续成绩。

第 4000 次迭代的后续卷已完成：

| Seed | 解析合法落台 | 正手 / 反手 | 正手 signed composite | 物理 root fall |
| --- | ---: | ---: | ---: | ---: |
| 1 | `50/100` | `0/50` / `50/50` | `0/50` | 0 |
| 2 | `88/100` | `38/50` / `50/50` | `0/50` | 0 |
| 3 | `98/100` | `48/50` / `50/50` | `0/50` | 0 |
| 4 | `0/100` | `0/50` / `0/50` | 无正手 strike | 21 |

稳定门的 median/worst/spread/worst-side 全失败；详见
[稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。这张表同时说明为什么不能只看
旧解析落台数选 seed。

2026-07-13 的运行态是：16 条 fresh 广度臂已分两波全部按负责人运营决定停止。第二波前
重验最新 checkpoint 的迭代、76 tensor/`1,762,715` 浮点值 finite、schema-3、fresh lineage 和
相邻合同 SHA，且确认 24/24 最近 K20 格的正手 signed composite 都为 0。只向各自登记
PGID 发信号，无 broad kill/judge/真机命令；两 Pod GPU 已空。该动作不构成 q10 阈值正式判死或
setting 晋级；完整曲线、
checkpoint SHA 与信号边界见
[拍面×plant 广度矩阵](experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。

## 5. 阶段 2：虚拟球的不同到达状态

### 5.1 目标和实验结构

阶段 2 用虚拟来球分布生成题目，使球到达击球区域时的位置、速度、方向和高度变化，但只出当前
动作接口确实有可能完成的题。出题器和按题适配器是共同前置，不是一个实验臂。

正式比较应在同一批中做四臂：不额外适配的对照、站位钉死只用手臂补偿、站位随球平移并使用
脚步、整套动作绕击球锚点旋转。移动时序是正交消融，同批比较先移动后挥、边移动边挥和不规定
结构三种方案，而不是把这些方案预先排成固定串行课程。

### 5.2 动作与出题

**问题：** 直接放大目标框会生成动作根本做不到的题，也会把手臂、脚步、转身和重新定时混在一起。

**计划解法：** 用正向模拟生成不同来球的到达状态；每题先过动作接口可行性检查，再按题调整
整身朝向、时序和拍面。

**当前状态：** 整身旋转、时间律和动作静态安全筛已有部分工具。7 段私有新视频已精确登记；其中
高点拍压第五动作和左右横移四候选已通过 GVHMR 结构门，v12 未执行。既有 Franco 反手拉 B/C 的
signed 重定位产生 `19/3` 个 proposal，但还没有一个拿到 schema-2/L0/L1/桌网/动力学证书。
正式出题器、按题拍面/路径适配、站位与挥拍联合训练和留出卷均未完成，没有阶段 2 行为成绩。
详见[动作空间重定位实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)
和[新动作组合设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md)。

### 5.3 观测

当前 179 维输入已包含目标球拍位置、速度、拍面和击球倒计时，可以承接一部分变到达题。只有在
实验确认必须显式给出目标站位时，才考虑 181 维扩展；它不是当前配方。

### 5.4 Reward

正式对照使用同一份变到达题表，并冻结阶段 1 Reward、机器人动力学、训练预算和其他输入；只改变
预注册的手臂/站位/整身转向等动作补偿机制。减速入位只加入“站位随球移动”的对应臂；固定站位
臂不应得到同一项 Reward。

### 5.5 时序、连续和切真球条件

每个单拍臂都要另做不重置连续卷，检查上一拍收尾、等待和下一拍就绪是否一起成立。目前没有结果。

切到阶段 3 前，至少需要：训练内按全部挥拍机会计分的正反手解析回球率各 ≥50%、变到达状态
留出卷 ≥50%、虚拟预测与物理球落点差 <0.1 m，并且 `Gate3B` 评分链已建好。这里的训练内
回球率不是 `T1` 连续能力证据，两者必须分开报告。当前所有切换条件均未满足。

## 6. 阶段 3：物理球进入训练

### 6.1 目标

阶段 3 才让球从对面真实发出，在仿真中飞行、弹台并碰球拍；击球位置、预警时间和是否需要移动
由球的轨迹自然产生。阶段 2 先测清动作和站位边界，才能把阶段 3 的失败归因到球物理或接触。

### 6.2 动作与观测

动作适配和站位能力沿用阶段 2。179 维输入预留的旋转参数当前填零；先通过无旋物理球，再按
低旋 → 场馆旋转逐级加入旋转信息和随旋调整拍面/切向拍速的能力。

### 6.3 Reward 与物理成绩

先建立真实碰拍、过网和落台的物理结果，再决定落点、旋转等 Reward；不能继续用解析回球列冒充
物理训练信号。每个动作仍需分别报告单拍与连续的未倾倒、物理击球和物理上台。

### 6.4 当前状态

Isaac 的来球飞行真值仪和拍面接触源码已有材料，但拍面接触尚无被接受的 Pod 运行、训练结果、
完整物理成绩卡或跨引擎四格对账；球旋转传递也未完成实测标定。阶段 3 的行为训练尚未开始。

## 7. 部署验证线：当前卡在哪里

当前 179 维模型已经通过严格装载和负例拒绝。planner 与 C++ policy 之间的 formal tuple 现已把
shared epoch、command/base sequence、side、target 和最新 tick-start base 绑定到同一 actor gate；
exact 源码也已通过 portable Release。但这次构建明确关闭 ROS/AimRT，runner 没有执行，因此还没有
最终 ROS/Jazzy/AimRT 集成、厂商 MuJoCo 首个有效控制周期或固定考卷行为记录。当前 `Gate3-D0`
只要求先完成一份单拍全链演示，不冒充连续对打，详见
[Gate3-D0 实验](experiments/2026-07/EXP-GATE3-CURRENT179-D0.md)和
[exact build 卷宗](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md)。

2026-07-11 的 `13 PASS / 7 FAIL` 来自旧 110 维模型：3 次发球只有 1 次合法回球，出现 1 次摔倒
和明显漂移。它证明旧链曾执行，不是当前 179 维模型的 Gate3 结果。

## 统一工作队列（唯一优先级账本）

队列按主题组织，方括号内数字是全局执行顺序。人类责任人只能写人，Codex/Claude 只写执行者；
详细现状和命令留在对应实验记录。

### 动作主线

- **[1｜P0] Franco canonical 五动作库正式化。** 责任人 Franco；执行者 Codex。
  反手拉槽位保留 C；五动作 × upper/full 十件正按
  `shared zero-speed ready → selected core/window → shared ready` 重建，`adv2c3` 只作比较，
  正手挡由两套 scope-specific 七关节拍面流形生成。当前仍是 `0/10` 获训练授权；下一证据是十件
  candidate-only 构建与 Agibot MuJoCo 播放报告，随后闭合 grounded torque/contact、碰撞/地面/桌网、
  新窗口行为、strict registry/consumer。S0 full 的旧 `7.37 cm` 地穿必须由新件独立重验；M0 stance
  `0/4` 仍不占 RL GPU。详细状态与停止条件只在
  [正式化实验](experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)维护。

### 尺子与阶段 1

- **[2｜P0] 拍面正反与解析判分。** 责任人 franco；执行者 Codex；C2 的真实非零 plant 已证伪旧配对，
  D2 永久不发；C3/D3 显式零摩擦 L1 与 fresh paired receipt `bb3cd749...bbde` 已闭合，不得重跑。
  真 blocker：这对 checkpoint 的 immutable K100 目前卡在一份 parity 合同上——Isaac（PhysX）会对
  第 8 号关节做 velocity-limit 刹车，MuJoCo 没有等价语义，判卷器因此在第 0 题开始前就对 C3/D3
  两侧 fail closed（`scheduled=50/side`、`asked=0`，没有任何 K100 成绩）；该 parity 合同修复前
  paired exact judge 保持 NO-LAUNCH（见
  [C3/D3 K100 v2 操作页](operations/run_phase1_signed_face_c3d3_k100_v2.md)）。
  下一证据是用同一 immutable K100 判这对 checkpoint，再决定是否启动一个 seed 的“热启动/从零 ×
  线性引导关/开”四个机制单元到相对 checkpoint
  `+200/+500/+1000`。只有胜者连同匹配对照才解锁第二 seed，不再给已失败配方复制四 seed。
  [量尺实验](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)；
  [机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)

### 部署验证

- **[3｜P0] W/Y 179 维模型的 0.5 秒同卷 Gate3-D0 单拍全链。** 责任人 franco；执行者 Codex；
  两份 `model_6700` 的唯一定位、finite、`179→31`、零写入 `--plan` 和 fresh ONNX 结构/推理检查均
  已闭合，但 checkpoint 的 `training_contract_lineage_exact=0`，ONNX 的
  `training_contract_exact=0`。当前最短 P0 是 exact-lineage remediation 与重新导出；过门后才实现
  “同一 100 题 → 六元发球 →
  同球 `task_revision` → owned 生产 planner → C++ runner → 同一厂商 MJCF/plant”的适配器，并逐题
  输出 attempt/completion/hit/return/fall/deadline。exact lineage 与适配器通过前不启动行为、不使用
  隔离中的旧连续演练脚本；W/Y 仍只是演示优先候选，U 是稳定备选。
  [半秒卷宗](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)；
  [Gate3-D0 实验](experiments/2026-07/EXP-GATE3-CURRENT179-D0.md)
- **[7｜P1] Gate3 历史谱系复核。** 责任人 yikang；执行者 Codex/direct；V9 force 分支未向
  `set_external_force_and_torque` 传 `position_data`，所谓 pelvis-COM force 实际施于 link origin；旧
  A/B_FORCE/AB_FORCE/AB_FORCE_FRESH 曲线只能保留为 mechanics/throughput，不得作平衡因果结论。
  RallyV10 W&B `qpl08mug` 已终档 iteration `9999`，但没有 MuJoCo/Gate3；V11 fast/prestrike 分别停在
  `2816/18274`，已有 stand/trajectory 代理也不是正式行为 Gate。下一证据是在当前 `main` 重做位置语义
  正确的独立 physics/reference oracle，再走同运行链 vendor Gate3；旧分支不得整体合并。
- **[OPS｜P1] RallyV10 真机单命令部署与板载 VRPN。** 责任人 dongc1；执行者 Codex；分支
  `hitter`。把 CMTracker/VRPN、relay 与 planner 收敛到 HDU，Laptop 只保留完整 SSH/TTY
  入口，并为 MDU HAL、SHADOW、真机 runner 提供互不自动串联的单命令脚本。下一证据：脚本
  静态/打包检查、HDU preflight，以及 `run_pingpong_end_to_end.md` 9.9 的可复现终端矩阵；不由
  Codex 执行真机 MOTION。
- **[16｜P1] 全动作库可行性复扫 + 投影钉根约定修复。** 责任人 yikang（投影工具在其分支）；
  执行者 Codex 可代跑扫描。人话：yikang 反手"结构性死亡"判决经 07-22 三路对抗复核改判
  **应复扫**——伪影机制＝投影把骨盆根钉成 reset 单位姿态、删掉反手 ~40-60° 转体 +
  stationary 剖面取消 -40° 归一化（仓库内有同 clip 100%→0% 的 A/B 反证）。Franco 拍板：
  不止反手，注册表全部现役 `_cal` 族 + 全部投影变体都按"源几何 × 投影后 × 两速度档"矩阵
  重扫，先修投影约定（钉根保留源骨盆 yaw + 被删根旋转角进合同 fail-loud）再全量重扫；
  修复合入前 stationary-v2 家族所有"结构性排除"判决降级为存疑。下一证据：投影约定修复
  合入 + 逐 clip 落账表（pod2 空卡跑，排位在判卷欠账之后）。
  [扫描与修复方案](research/branch_fix_audit_20260722.md#全动作库扫描与修复方案franco-07-22-拍板不止反手全库都扫都要修好)
- **[15｜P2] 部署侧站立异响四欠账——源码已修，待 Linux 编译回归。** 责任人 franco；执行者
  Claude。人话：站立异响 PDF 排名 2/3/6 的三项取证欠账 07-22 已改（只动 agi/，默认行为逐字节
  不变）：两条站立路径增益来源启动摘要 + static handoff 时间戳事件进 CSV/日志 + 可选审计降档
  旗标；build 指纹（git SHA+脏标）每次运行第一行打印；obs/trace CSV 尾部纯增 31 列实测力矩
  `tau_*`（SDK 不暴露电流/温度，已记档）。下一证据：Linux 构建机跑 portable Release +
  run_tests 回归（macOS 无 vendor 栈，本机只做了改动块逐字 -Werror 编译+运行断言）。
  [裁决与细节](research/branch_fix_audit_20260722.md#pdf-五缺陷对账部署侧四欠账-2026-07-22-已修)

### 训练引擎与机器人物理

- **[10｜P1] RallyV11 TOPP 加速动作证据复核。** 责任人 yikang；执行者 Codex；fast run 已停在
  iteration `2816`，prestrike run 已停在 `18274`，当前无 live trainer。`model_6600` 的 stand-center
  代理为 `0.8889` 且 `0/24` fall；后续 `model_11100` 代理反而下降到约 `0.714/0.730`。这些只说明
  代理曲线未单调改善，没有 MuJoCo/Gate3 晋级结论。下一证据是把可保留的触球锁窗重定时思想在当前
  `main` 选择性重做，并先过 exact 动作、MuJoCo oracle 与冻结行为卷；不得整体 merge 旧分支。
- **[4｜P0] 原生 MuJoCo 训练候选。** 责任人 franco；执行者 Codex；下一证据：修正四个源码缺口
  后复核，再测单环境核心、并行吞吐和一次限预算 PPO 更新。[实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)
- **[11｜P0] 平衡×时序 24 格广度矩阵（吸收 Wave A/B）——已发射，收口进行中。** 责任人 franco；
  执行者 Claude；执行分支 `Franco_codex/balance-temporal-matrix-20260720`（已并入 main）。人话：把
  "动作平滑怎么给"（N 无平滑 / C 现役全身 raw action-rate / H 恢复期腿腰执行目标铰链）和"稳定机制
  给哪种"（S0 无 / S1 挥拍后安顿债务包 / S2 无参考支撑包 / S3 下肢软模仿）在 W、V 两个 `model_6700`
  诊断父本上交叉成 24 格，单 seed 3。**2026-07-20 全部 24 格发射；同日实测本任务代际同卡 SM 分时
  （4 条/卡 ≈15 s/iter）后把筛选终点前移到 +4000（终档约 model_10700），并按 Franco 授权动态清退：
  N 交互 6 格永久停（容量让位非科学负例）、8 格暂停待续，台账在两 pod 的
  `scheduling_ledger_20260720.jsonl`。已到线收口：`w_h_s0`/`v_c_s0`/`v_p035`/`v_h_s0`（终档
  model_10700–10900）。中期信号（非终判）：T 轴 C≫H——V 父本回球/拍 0.90 vs 0.54、fall 无差，
  H 行淘汰待全格 +4000 终判；S 轴四档暂无分离。** Wave A probe10 科学位与 Wave B 六格队列由本矩阵
  取代（superseded），probe9 六份 receipt 只作 runtime mechanics 证据；父本谱系 mismatch 故后代
  formal-exact ineligible，胜者机制须另在 exact-lineage（qdot `model_1000` 族）重跑正名。机器真源：
  [矩阵实验](experiments/2026-07/EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md)、
  [队列配置](../configs/phase1_balance_temporal_matrix_20260720.yaml)、
  [操作页](operations/run_phase1_balance_temporal_matrix.md)。
- **[12｜P0] Wave P push 鲁棒性波（Franco：push 是平衡的希望）——进行中。** 责任人 franco；执行者
  Claude。人话：现役配方从不推机器人（`push_robot=None` 对齐 HITTER），三篇对标论文全带 push；本波
  在 C+S0 配方（对照=矩阵 `w_c_s0`/`v_c_s0`，不重复买）上测 18 臂——速度冲击大小 ±0.2/0.35/0.5/0.8
  m/s、方向（+yaw / +全角速度）、频率（1–3 s 高频）、以及**同冲量力推**（68 N/155.4 N × 0.30 s @
  pelvis link 原点，诚实标注非 COM）比较"瞬时速度注入练回正 vs 持续力练抵抗"。速度/力臂已上卡 12 条，
  其余按空槽队列补发。RunPod 两次重启 Pod1 容器（19:43Z/04:35Z），值班 routine 自动整机重建。
  [push 实验](experiments/2026-07/EXP-P1-PUSH-ROBUSTNESS-20260721.md)。
- **[14｜P1] 抖动-地面-脚部消融波（Wave CGF）——已预注册，闸门锁定。** 责任人 franco；执行者
  Codex。人话：站立异响 E1 复核 PDF + Franco 07-22 八条里训练侧成立的欠账各买一臂——action_rate
  剂量曲线 `ar02/ar05/ar10`（-0.2/-0.5/-1.0）、机器人材质摩擦抬高 `grip`、随机凹凸地形 `rough`
  （fresh-from-random 20001 updates，平地 checkpoint 禁止静默上粗糙地）、mjlab 落地冲击罚 `footrw`
  （-3e-3，无量纲超阈倍数量纲）、软惩罚减负 `penlight`（六个软惩罚降 ~1/3，含 07-22 新接线的蹭滑/拖脚剂量键，Franco 第 8
  条）、被动阻尼折 kd `kdpassive`（源码未接线，闸门锁死）。父本只用 W，对照＝矩阵 `w_c_s0`
  不重复买；续训臂 13301 updates（到 ~20001，对齐 2万-2.5万 目标下沿）。三件套+单测已落盘
  （47/40/38 项全绿）。下一证据：07-22 wiring 合并钉 40-hex + groundfoot 闸门翻真 → pod 恢复后
  一格 probe smoke 通过即按 launch_order 发全矩阵。
  [预注册](experiments/2026-07/EXP-P1-CHATTER-GROUND-FOOT-WAVE.md)
- **[17｜P1] 集成升级波（Wave IU）——冻结；只作兼容性压力测试候选。** 责任人 franco；执行者
  Codex。2026-07-23 独立审计推翻“组件已分别证明有效”的表述：`action_acc` 已接线但没有行为证据，
  qbar×action-rate、速度推×力推和摩擦/地形组合均未做可归因验证；CGF 也尚未运行。现有
  `combo_franco` 仍绑定已淘汰的反手 B，runner 只替换第二条 motion、没有同步重绑击球相位，禁止按旧
  命令发射。先完成五动作库和 matched full-body 消融；之后若 Franco 明示解除发射冻结，再把 IU 当作
  “能否共存”的压力测试，而不是最终升级证明。
  [预注册](experiments/2026-07/EXP-P1-INTEGRATED-UPGRADE-WAVE.md)
- **[13｜P0] Wave Q 情报波——已有诊断，机制仍未定。** 责任人 franco；执行者 Codex。账面只有
  `v_qbar` 到过终点，其余臂在 pod 重启时中断；没有同代、同卷、完整配对结果。所谓
  `lower_body_pose_imitation=2.0` 同时把模仿窗从 `0.3/0.4` 放到 `10/10`，而 actor 无论开关都看见
  31 关节参考，所以旧“全身有效”读数无法归因。动作库收口后首轮只做 matched 三格：
  `U0`（upper 资产、腿模仿关）、`F0`（full 资产、腿模仿关）和 `F1`（与 F0 同一 full bytes、
  只开 12 腿关节 position 模仿）；`U1` 静态 ready 腿教师延到第二阶段。首轮关闭 IU 升级包、
  push、rough 和额外 locomotion，逐动作报告平衡与回球；旧队列不续发，发射冻结不变。
  [动作库终审实验](experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)

<details>
<summary>旧 Wave A/B/Q 发射与基础设施取证（历史，不是当前优先级账本）</summary>

  Wave A 只比较同父本
  `phase1_balance_action_slew_20260720` 队列的
  W/V 的原 dense action-rate、无 action-rate 与腿/腰 processed-qdes slew hinge 六格，先过短合同 probe，
  再按 `+200/+500/+1000` 看 fall、completion、return 与姿态/目标突变尾部。所有子项因显式新 Reward
  合同而 formal-ineligible；结果只能筛机制，不能晋级 policy。Wave B 的 M0 moving teacher 已因 stance
  `0/4` 拒绝；当前只设计 upper-only control 对静态 v4rg 下半身模仿或 non-demo constraint，其 exact
  flags/矩阵必须先由源码/合同审计冻结，当前不猜。Wave A 的 probe2/3 verifier 缺口与 runtime 已保留；
  probe4 W-C 在创建任何运行产物前因 Hydra key 错写 fail closed。probe5 已从 v4 no-clobber 根自然完成
  W-C、V-C、W-N、V-N、V-H 五格并通过五份
  [六格探针收据](DEFINITIONS.md#balance-probe-receipt-set)；W-H 在 trainer 启动前因为 supervisor 对
  fork→exec 过渡期的 `/proc/<pid>/cmdline` 只采样一次而 fail closed。事后 exact PGID、child PID、
  assigned GPU 与 Kit/cache locks 均为空，所以这是发射身份竞态，不是 processed-qdes 机制负例；五份
  probe5 收据也不能与一次重试混成六格集合。probe6 改用 v5 no-clobber 根并加入同一 child 的有界
  exact-identity wait，但 W-C 在 `2026-07-20T03:01:10Z` 于 trainer/probe supervisor 内、进入 trainer 前
  fail closed：`_failure_audited_transaction_shell` 给 transaction 每行加两个空格，破坏了 shlex-quoted
  multiline Python 参数，81 B `run.log` 只有 `IndentationError: unexpected indent`。locked launcher
  无法绑定已快速退出的 child，且未发 signal；leader evidence、child evidence、binding、terminal、
  checkpoint、receipt 与 RSL 均 absent。PID=PGID `2712318` 已双扫稳定 absent，Pod1 GPU0=`0 MiB / 0%`，
  Kit/cache locks free；整批停止，其他五格未发。故 v5 是永久基础设施失败历史，不是 C/H/N 机制负例。
  probe7 使用 v6 fresh root；W-C、V-C、V-N 均 natural exit=`0`、normal exit=`true`，两步各 `98304`
  samples，并发布 exact receipt。W-N 于 `03:22:50Z` 到达 RSL/scene config 后冻结在
  `Starting the simulation`，没有 Learning iteration、binding、terminal 或 receipt；180 s locked watchdog
  对 exact 组依次 TERM/KILL，rc=`125`，事后进程组、GPU1 与 locks 全空。V-N 在该失败被确认前 6 s 已发，
  后续自然完成且验证通过；W-H/V-H 未发。W-C/V-N 的成功和 exact argv 对比排除了 W parent、
  `action_rate=0` 各自作为必要失败原因，因此只记 infrastructure transient，不改 Reward 结论或 timeout。
  v6 immutable，三份收据不可与重试或新代混用。
  [probe8 替换批](DEFINITIONS.md#balance-probe-generations) 使用
  `/workspace/codexschema/phase1_balance_action_slew_v7_20260720` no-clobber 根与
  `phase1_balance_slew_probe8_{w_c,w_n,w_h,v_c,v_n,v_h}_seed3_20260720`（第八版六格零训练合同探针，
  不是行为成绩）。它绑定 [launch manifest](DEFINITIONS.md#balance-launch-manifest) 文件/内容 SHA-256
  `887c0b9e097e50300d83eef27e587d112f70132958e6e8d9b68af74437fa7231` /
  `13f92d5eda71e90abd6a14a1498c2afd98d3cb825cb26e2f5b74958b5a795f84`，config/runner SHA-256 为
  `0c84613f05439237f6e36d37e0c9210984465d928b9c0cba50999bd8995145f9` /
  `2bc9d59e21413a812a742529c6f3291f5710c384e0f4af7ee7098f33b25ba17d`。该批只发 W-N：它于
  `2026-07-20T03:44:19.857Z–03:44:32.112Z` 在 Pod1 GPU1 的 `sim.reset` 阶段 SIGABRT，trainer
  exit=`-6`、外层 rc=`134`，`run.log` 末行为 `malloc(): invalid size (unsorted)`；没有首个 Learning
  iteration、binding、receipt 或 RSL run directory，其余五格未发。关键文件 SHA-256 为 claim
  `8472ecf98edd269b50b03925be1ab778d5d195b816e2e8bcc8e0be29499ffa8b`、launch spec
  `334ed262de0fbb4549087fcd4f5c216d3fe3c12f46954f45e545eafd4c71c23c`、run log
  `b3437c878aa1b0d5d8e0646e01c6d3a3d1d822a042289411cdd0e79613c3f49c`、launch log
  `edc4782f57d0c9e498351161fbe224f38a4db7a93f1fde588a681a5346e13ce8`、leader evidence
  `b7f981c8d59cb92ef96b6d7851350386cf913212090df070105abd0fa27e8815`、terminal
  `ad5c46a7755f9b684fa2acad94387e2ebfc08db4f9adc186336fe8e295356268`、child evidence
  `c2d6c31bb231a30716bc5fc1c160f8f5855d9529102a2241ca1187c75a96226f`。事后 Pod1 v7 只有
  `probes/w_n`，Pod2 v7 root 不存在；`03:49:31–03:49:33Z` 两轮 closure 均确认 exact PID/PGID、所有
  probe8/v7 argv、六张 GPU compute context 与 Kit/cache lock holder 全空。故 probe8 是
  **pre-RewardManager infrastructure** 失败，不是 C/N/H 机制结果；v7 immutable，禁止重试、补格或混收据。
  [probe9 替换批](DEFINITIONS.md#balance-probe-generations) 使用 fresh no-clobber root
  `/workspace/codexschema/phase1_balance_action_slew_v8_20260720`，六个
  `run_name` 为 `phase1_balance_slew_probe9_{w_n,w_c,w_h,v_c,v_n,v_h}_seed3_20260720`（第九版六格零训练
  合同探针）。config/runner SHA-256 为
  `c7ec75a9917b8bdcf7976186633b021c69fb82e591898dad6b8d5c93cfdb37d5` /
  `24b5f7831ad49c2b88266fed65c37e6e4bcdddaacab28ffa917fce66ef918db1`，manifest 内容/文件 SHA-256 为
  `97c36e471fb8fc6b93fe212f20846de6697db518192e7a45c6618e5924947e28` /
  `688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353`。该批已于
  `2026-07-20T04:14:32Z–04:22:42Z` 严格按 W-N（Pod1 GPU0）→receipt+closure→W-C（Pod1
  GPU1）→receipt+closure→W-H→V-C→V-N→V-H 逐格完成；每格都在下一格启动前 natural exit=`0`、
  normal=`true`、first iteration=`true`、exact verifier passed，并完成进程/GPU/lock closure。六份 receipt
  的 file/content SHA-256 分别为 W-N `ee8c5378…8c5ff` / `afce94d7…80f9`、W-C
  `b948a4d8…18d5` / `e5daf19c…d3f0`、W-H `a80502c9…8111` / `32abf562…7e68`、V-C
  `c3db6c38…edc1` / `ae30d1b5…3446`、V-N `06919a60…4bc7` / `bc99acd0…979c`、V-H
  `b7a24015…1ec2` / `0905ad8f…32a7`；共同 verifier SHA-256 为 `d736a205…0ebc`。本地六收据集重验
  succeeded，set SHA-256 为 `cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`。
  当前两 Pod 六张 GPU、相关进程与 Kit/cache lock holder 全空。W-N GPU0 与 W-C GPU1 的成功只排除各自
  在该卡必然失败，不证明 GPU 等价，也不覆盖 probe7/8 的 immutable 失败历史。
  六格科学长训的 same-swapped mapping 已解锁命令生成。`origin/main=16263be5` 上曾渲染
  `/tmp/phase1_balance_slew_train_commands.json`，SHA-256 为
  `fc6f1ea38a5a823016d83675d56fc41b50b70dbde1bba60602b26d6c743802df`，但它从未执行 SSH；本 NOW
  更新进入 `main` 后旧 commit/NOW authority 过期，禁止执行。发射前必须在最新 `origin/main`
  重新 render 并审计。
  current-main 命令制品 `be346f94cf6bf738da36804bf59f6a60bc5249f3c6bf5474abf617358db4b42a`
  随后只发了 [v8 科学长训 attempt1](DEFINITIONS.md#balance-science-attempt-generations) 的 W-N（Pod1
  GPU0）。它于 `2026-07-20T04:45:02Z–04:48:25.971Z` 停在 `sim.reset`，没有首个 Learning iteration、
  Reward 指标或 checkpoint；locked launcher terminal rc=`125`，exact PID=PGID `2728928`、leader
  starttime=`335835722`。外层 caller rc=`121` 不是另一种 trainer 错误，而是旧 post-failure audit 在
  train stage 错误要求只有 probe supervisor 才会产生的 `trainer_child_evidence.json`。`04:49:40Z`、
  `04:49:47Z`、`04:52:49Z`、`04:52:51Z` 四次闭包均确认 leader/PGID、GPU compute process 与
  Kit/cache lock holder 全空；其余五格未发。结果 JSON 为
  [`phase1_balance_action_slew_train_v8_attempt1_result_20260720.json`](../configs/phase1_balance_action_slew_train_v8_attempt1_result_20260720.json)，
  SHA-256=`ac09b70a1df89a501165504f4c07158858687127172a8d9d5a6bdf1473e61a75`。v8 root immutable，
  该次只算 infrastructure-only / non-science，不能给 C/N/H 机制结论。
  fresh v9/probe10 候选继续保持 W-N Pod1 GPU0、W-C Pod1 GPU1 的 swap 与六格全局串行顺序；fresh
  root 是 `/workspace/codexschema/phase1_balance_action_slew_v9_20260720`，config/runner SHA-256 为
  `3bf5085ea8396513d162b9cce249dfb761b39b2827ec722959343c953683e59e` /
  `0fff4515cbe7e62798e8c39f701851c46e68287c7321e7618161fa9dde4789ce`，manifest 内容/文件 SHA-256 为
  `36ceb3c77dc056f4565378a92b03da58865378d86c5849085ba066631cea456c` /
  `664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a`。新 runner 的 failure audit
  显式区分 probe/train：probe 仍严格要求 child evidence；train 要求不存在 probe-only child/identity
  文件，并验证 launcher leader 组闭包。probe10 当前 **未发射**；必须先全局串行取得 fresh v9 同代
  `6/6` receipt，才允许渲染任何科学 train 命令，禁止复用 probe9/v8 收据。probe receipt 不是科学
  结果；六格机制状态仍 inconclusive / not adopted。该条目只有进入 `origin/main` 后才构成运行
  authority。两波都不能替代
  `T0/T1/T2` 连续恢复卷。[Wave A 实验](experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md)

  2026-07-22 前瞻风险挂账(yikang 只读核对,不改配方——reward 塑形属 franco lane):qbar 两臂的
  `qbar_contract.qbar_wiring_confirmed` 已在 `configs/phase1_intel_wave_20260721.yaml` 翻 `true`,
  渲染闸门已开,两臂进入可发射状态(07-21 晚 Pod1 boot 锁队列中实见 `w_qbar_res1` 排队等发)。
  但该合同 `wiring_note` 的人话仍写 margin "0.08 rad",而实现与 CLI 键是行程比例
  `qdes_limit_barrier_margin_frac`(hope_rewards: `d=min(q−lo,hi−q)/(hi−lo)`;0.08 行程比在肩偏航
  ≈26° 罚带、在腰侧倾≈3.2°,同一数字两种语义差一个量级)。main `5c23a3f6` 只修了键名漂移
  (margin→margin_frac),rad vs 行程比例的语义裁定仍悬空;按 wiring_note 自己的条款"与本段人话
  不符时必须先修订预注册再渲染命令",qbar 臂在读数/采信前应先由 franco 裁定 margin 语义并修订
  预注册,否则两臂读数按"量尺语义未定"处理。
  2026-07-22 裁定（Claude 查 Jiayi 源码，证据确凿）：v14 的键名即
  `all_joint_qdes_barrier_safe_margin_fraction: 0.08`（HitterV11 分支 V14 yaml 第 28 行）——**行程
  比例**，与我们 `qdes_limit_barrier_margin_frac` 实现完全同语义；wiring_note 的"0.08 rad"是人话
  笔误、已修订。qbar 两臂读数可采信；感谢 yikang 的只读挂账。

</details>

- **[9｜P0] Hitter 实机 planner 时序与训练反应时间基线。** 责任人 yikang；执行者 Codex；分支
  `codex/hitter-lowerbody-mujoco-alignment`，基于 `hitter`。2026-07-16 用户将本轮顺序收敛为：
  先审计最新 `main`、`hitter` 与 frame0-wait-v2 已有实现，再补齐端到端球样本时间戳/短 TTS、
  planner 求解限频与回调阻塞隔离、逐球结构化可观测日志，以及训练来球时间分布与最大反应时间
  监控。原下半身 Reward/接口对齐材料保留但暂缓扩展，其余连续挥拍、可行性、mid-swing/TOPP
  等项进入后续 TODO。下一证据是可复现的 remote-reuse 台账、schema/runner/planner 单测、离线
  timing replay 与训练指标测试；这些证据完成前不启动长跑 PPO、不执行真机 MOTION。

### 连续能力与后续接口
- **[5｜P1] 等待/恢复结构卷。** 责任人 franco；执行者 Codex；下一证据：同步机器合同后，用冻结
  Reward 先跑 `T0`，再跑 `T1`；只有前两层仍失败才允许 `T2` 平衡 shaping。并行单拍 slew 诊断不改变
  此顺序。[实验](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md)
- **[6｜P1] Hitter V3 规划器—policy 输入对齐。** 责任人 jiayi；执行者 direct；下一证据：旧观测
  排列、训练第 24100 次迭代 checkpoint 归属和第 7 版击球平面三项来源对齐。
- **[8｜P1] 110 维 RallyV10 左腕/恢复修复。** 责任人 dongc1；执行者 Codex；分支 `hitter`；
  V10 fresh run 的训练曲线已显示击球精度成熟，但 yaw-rate 约 `0.46-0.52 rad/s`、左腕超差率约
  `0.42-0.54`，尚未证明修复通过。当前工作：在不改 V10 基线的前提下增加独立
  constrained-resume 任务，从当前 V10 checkpoint 严格续训，固定最终 planner/yaw 分布，对左腕
  position+velocity 做全阶段独立模仿并收紧 yaw-rate。下一证据：host 合同门、Isaac mechanics
  preflight 与用户手动发射的 resume run；Codex 不停止、不启动 PPO。

队列排序与算力规则见[跑批作战手册](runbook.md#统一队列排序与算力纪律)。完整实验索引只在
[实验登记册](experiments/README.md)维护；本页不再复制实验索引或最近测试流水。

---

<details>
<summary>展开 2026-07-19 至 2026-07-22 的历史 Pod、probe 与实验流水（非当前队列）</summary>

## 历史运行流水（非当前运行态）

**2026-07-22 pod 状态：上午失联、下午原端口恢复。** 两 pod 上午全部 SSH 端点超时（本机网络
已排除），下午实测以**原端口**恢复：pod1（开机 ~2 h，重启过）三卡全被 yikang 四条训练占用
（v14_legfreeze ft6 + stationary-v2 formal/slow2fast x3，全 wandb，其中两条共挤一张卡）；
pod2（开机 ~1 h）三卡全空。checkpoint 卷完好——W 父本 model_6700 与 codexschema 各命名空间
均在，续训谱系断裂危机解除。07-22 下午验收后已处置：11 条被杀续训(25k 八格+intel 三臂)按原 argv 重算剩余步数复活,8 条在训
3 条等停滞裁决;判卷修正队列(--exam-bank 手传)两 pod 串行清偿首批 11 个终档;v_qbar 已到终档
10700。`hope-pods-watch` 已在新账号重建(每小时,含停滞裁决与回填队列)。三人分支追踪看板（防修复失踪）：`python scripts/branch_dashboard.py --min-ahead 1`，
裁决账本见[分支修复审计 07-22](research/branch_fix_audit_20260722.md)。

**2026-07-20 合入 `main` 后生效的候选覆盖：** 双 Pod 各三张 GPU 的 NVML compute process 与显存占用均为零；当前没有
训练作业。这个快照不代表永久 GPU 归属，也不证明无 GPU 的旧 shell/ROS 进程不存在；每次发射仍须
重新核验具体 PID/PGID、Kit lock 与卡位。半秒冲刺、Pod1 十二格 long-grid 及相邻长曲线都已经结束或
在 task-revision cutover 时收口，后文带日期的 `live/running` 只保留为历史快照，不能用于当前调度。

当前最短 P0 仍是 **vendor MuJoCo 同卷准备**，但新增了一道更早的 exact-lineage 门。`W`（拍心优先 ×
自由非击球臂）/`Y`（拍心优先 × 触球窗老师静音）真实零写入 `--plan` 均已通过，fresh `179→31`
ONNX 也通过独立结构检查与 CPU 有限值推理；然而两个 checkpoint 都记录
`training_contract_lineage_exact=0`，两个 ONNX 都记录 `training_contract_exact=0`。因此制品只能诊断，
不能发布到 production/vendor。先修复或重训出 exact-lineage checkpoint，再实现“同一 K100 → 发球 →
同球 `task_revision` → 生产 planner → C++ runner → 厂商 MJCF/plant”适配器；`U`（拍心优先 × 强准备）
仍是稳定备选。G05/G06 继续 `Partial`，`Gate3-D0` 继续 `Open`。

近期支线审计不产生直接合并或行为晋级：Jiayi V13 只有未验证的 post-swing balance/recovery 组合，
没有 checkpoint/行为结果，且其分支历史含大量无关/破坏性改动，禁止整体 merge；有用思想只能在最新
`main` 选择性重做。Yikang V9 的所谓 pelvis COM force 未传 `position_data`，实际在 link origin 施力，
所以现有 force runs 只算 mechanics/throughput，不能给平衡结论；V10 已终档 iteration `9999` 但没有
MuJoCo/Gate3，V11 fast/prestrike 分别停在 `2816/18274`，代理 checkpoint 也没有闭合正式行为 Gate。

新的 processed-qdes action-slew 六格矩阵是 Wave A，只用于判断腿/腰执行目标突变是否能降低单拍后的
fall 与姿态尾部风险；它是默认关闭、同父本、单 seed 的诊断，不是完整稳定方案或已采用 setting。
probe2 W-C 已自然到 `model_6701.pt`，但旧 verifier 错把首个 `0.48 s` rollout 合法的
recovery-eligible=`0` 当成失败。probe3 的 W-C/W-N/V-C 也均自然退出；独立复核发现 verifier 未绑定
`24 steps/env`，所以旧 W-C receipt 与三格 runtime 都不能解锁长训，其余三格未启动。probe4 W-C
在建 run-dir/Kit 之前被真实 Hydra compose 门拒绝：错写了 `algo.num_steps_per_env=24`；v3 root、
log、checkpoint 与 GPU compute 均不存在，不是科学负例。probe5 随后以
[`algo.runner.num_steps_per_env=24`](DEFINITIONS.md#ppo-num-steps-per-env) 在 v4 no-clobber 根完成五格 exact
receipt；W-H 只因 fork→exec 身份采样竞态在 trainer 前 fail closed，事后进程/GPU/locks 全空，不能当
H 机制负例，也不能拿五份旧收据补新批。probe6 的 v5 W-C 又在 `2026-07-20T03:01:10Z` 于
trainer/probe supervisor 内、进入 trainer 前 fail closed：transaction wrapper 给每一行加两个空格，破坏了
shlex-quoted multiline Python 参数，81 B `run.log` 只有 `IndentationError: unexpected indent`。locked
launcher 来不及绑定已经快速退出的 child，也没有发 signal；PID=PGID `2712318` 双扫稳定 absent，GPU0
为 `0 MiB / 0%` 且 locks free，其他五格未发。这是 launcher 包装缺陷，不是机制负例，v5 永久只作历史。
probe7/v6 随后让 W-C、V-C、V-N 自然 exit `0` 并各发布 exact receipt，三格每步均为 `98304` 样本且
进程/GPU 闭合。W-N 于 `03:22:50Z` 到达 RSL/scene config 后冻结在 `Starting the simulation`，没有
Learning iteration、binding、terminal 或 receipt；180 s locked watchdog 对 exact 组依次 TERM/KILL，
rc=`125`，事后组/GPU1/locks 全空。V-N 在 W-N 失败被确认前 6 s 已发，随后自然完成并验证；W-H/V-H
从未发。W-C 与 V-N 的成功分别排除了 W parent 和 `action_rate=0` 作为必要失败原因，故当前只记一次
基础设施 transient，不修改 Reward 结论或 timeout。v6 永久不可重试、补格或混收据。
[probe8 替换批](DEFINITIONS.md#balance-probe-generations) 随后只发了 v7 的 W-N：它于
`2026-07-20T03:44:19.857Z–03:44:32.112Z` 在 Pod1 GPU1 的 `sim.reset` 阶段 SIGABRT
（trainer exit=`-6`，外层 rc=`134`），`run.log` 末行为 `malloc(): invalid size (unsorted)`；没有首个
Learning iteration、binding 或 receipt，其余五格未发。事后 Pod1 v7 只有 `probes/w_n`，Pod2 v7 root
不存在；`03:49:31–03:49:33Z` 两轮 closure 均确认六张 GPU、相关进程与 Kit/cache lock holder 全空。
这是进入 RewardManager/机制比较前的基础设施失败，不是 C/N/H Reward 结果；v7 永久 immutable、禁止重试或
混收据。[probe9 替换批](DEFINITIONS.md#balance-probe-generations) 已于
`2026-07-20T04:14:32Z–04:22:42Z` 在 fresh v8 root 严格按 W-N（Pod1 GPU0）→closure→W-C（Pod1
GPU1）→closure→W-H→V-C→V-N→V-H 完成六格，每格都在下一格前 natural exit=`0`、normal=`true`、
看到首个 iteration、通过 exact verifier 并闭合进程/GPU/locks。六份同代 receipt 的本地重验已通过，
receipt-set SHA-256 为 `cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`；
科学长训只有未执行的合入前历史 render，本 NOW 进入 `main` 后必须保持 W-N/W-C GPU swap 重渲染并审计。W-N 在 GPU0 与 W-C 在 GPU1 都通过只排除
“该格在该卡必然失败”，不证明 GPU 等价，也不抹掉 probe7/8 的历史。六格科学结果仍未知、未采用；
所有旧 manifest/目录禁止重用。随后 current-main v8 的
[科学长训 attempt1](DEFINITIONS.md#balance-science-attempt-generations) 只发 W-N（Pod1 GPU0）：
`2026-07-20T04:45:02Z–04:48:25Z` 停在 `sim.reset`，未见首个 iteration、Reward、checkpoint；exact
组已由 locked launcher 收口，其余五格未发，因此仍是 infrastructure-only、inconclusive / not adopted。
fresh v9/probe10 只是已冻结、尚未发射的替换候选；它修正 train/probe 失败审计的 stage 语义，并要求
全局串行完成同代六份 probe receipt 后才允许生成 train 命令。
Wave B 的 M0 左右 moving teacher 已因 stance `0/4` 被 input gate 拒绝；当前只允许审计
upper-only matched control 对静态 v4rg 下半身模仿或 non-demo stability constraint，exact flags/矩阵仍待
源码审计，不得猜写。
两波都不得绕过连续能力的固定顺序：先 `T0`，再 `T1`，只有前两层仍失败才进入 `T2` 平衡 shaping。
Franco 五种用途动作、横移老师和生产 planner 继续并行，但都不能取代厂商裁判。

2026-07-19 本地只读源码定位曾确认：现有 [0.5 秒时序卷](DEFINITIONS.md#timing-exam-0p5)使用
[K100（100 道固定同卷题）](DEFINITIONS.md#q50-and-k100)，但属于直接 policy 诊断并绕过生产 planner；
Python MuJoCo 评估器虽支持 179 维与固定题库，却不消费逐题 25 周期
时序卷；Gate3 假球入口只接每题“初始位置 + 初始速度”的六元发球。当前缺少把同一 100 题经发球、
[`task_revision`](DEFINITIONS.md#planner-task-revision)（同一来球实时任务修订）、生产 planner、C++
runner 送入同一 MuJoCo XML 场景模型（MJCF）与 plant 的适配器。W/Y 两份 `model_6700`
已完成只读制品前置检查；2026-07-20 的 plan/export 与 exact-lineage 结论以上述权威覆盖为准。
旧连续演练脚本保持隔离禁用。G05/G06 继续
`Partial`，`Gate3-D0` 继续 `Open`，不能宣称演示成功。详细合同见
[半秒冲刺卷宗](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)与
[G06](gates/G06_isaac_to_mujoco.md)。

2026-07-19 一次 Pod1 只读全域精确查找已为 W/Y 各唯一定位 `model_6700.pt`。两份 checkpoint
内部 iteration 均为 `6700`，均有 `74` 个浮点 tensor、`1,762,715` 个浮点元素且
non-finite 为 `0`；actor 结构均为 `179→31`，两份 `params` 内的训练合同、`env.pkl`、
`agent.pkl` 与 `env.yaml` 也都存在。这只是静态训练制品检查，不是 vendor 成绩。
standalone ONNX exporter 同时新增真正零写入的 `--plan`：它以 `weights_only`
加载 checkpoint，走完 finite、donor、motion、harvest、train-bank 与 formal face-179 材料验证，
然后在第一次输出写入之前返回；JSON 显式给出 `checkpoint_iteration`、
`artifact_written=false` 和 `graph_export_not_executed=true`。本地聚焦回归为 `97 passed in 0.38s`，普通导出的
fake smoke 仍通过。2026-07-20 两份真实 plan 都已通过；W/Y fresh ONNX SHA-256 分别为
`ee0e2e83...d970` / `72da43d9...f995`，但 checkpoint lineage 与导出 contract 都为 inexact，故停止在
production/vendor 前。G05/G06 仍为 `Partial`，`Gate3-D0` 仍为 `Open`。

- **本轮 task-revision 训练池（已结束）：** formal 179-D 与训练现已统一为“一颗球一个 `task_id`、同球估计用递增
  `task_revision`”，挥拍中位置、速度、signed 拍面与剩余击球时间每个 policy tick 继续原子更新；phase
  governor 只接受可达的动作加速。准备时间不是以 `0.5 s` 为下限，而是同时采样 `<0.5 s` 压力、exact
  `0.5 s` 基线、`0.5–0.9 s` 快球和 `0.9–1.7 s` 宽分布。4096-env `A6` 已证明这些机制真实激活；随后
  22 个 delay-zero 格全部各发射一次。最终 `+1000` 取证当时，Pod1 八条有效臂已到档、两条基础设施
  拒绝，只有“强脚底准备”仍 live；Pod2 十一条有效臂已到档且全部自然结束、一条 importer 拒绝。
  该 Pod1 尾项后来也已收口，2026-07-20 没有仍在运行的半秒冲刺训练。
  因此旧说法“19 条仍 live”已经过期，也没有需要用 signal 清理的 Pod2 训练。两个 importer malloc
  `rc134` 与一个 boot stale-timeout 不算科学失败且不自动重跑；两个 positive-delay 格因 governor/actor
  transport 尚非同 tick 原子继续 NO-LAUNCH。0.5 秒 K100 paper
  已在两 Pod 物化；当时 W/Y 尚无该卷行为分数，现役 v4rg 正反手 TOPP 已在 Pod2 CPU-only 一次完成，
  两侧安全证书均通过。Stage2 v8 随后在 Pod2 唯一执行，四个中点均被 TOPP 合法求解且 MJCF 闭包
  通过，但时间为 `0.80/0.86/1.10/0.94 s`，`0/4` 达到 `0.5 s`；当前 join-ladder family 因此按预注册
  停止，不再靠更多同族中点碰运气。这不证明 0.5 秒绝对不可能，却证明现役动作形状仍是瓶颈，旧固定
  倍率和当前拼接族都不能冒充可行动作加速。
  `+200` 机制失活、`+500` 极端崩坏和 `+1000` 同父本容差 Pareto 的组合保护式淘汰 consumer
  已闭合。2026-07-17 双 Pod 各一次 no-clobber `+1000` cycle 已为全部 19 条到档臂补齐行为 receipt；
  3 条基础设施失败仍排除，`ssh_signal_count=0`。这次仍没有合法淘汰项：不是因为没执行比较，而是旧
  ledger 的 ready tilt/base speed/foot contact/foot slip 四项分母全部为零，预注册组合器禁止填补缺失值，
  因而没有发布 portfolio-stop receipt。旧臂多数已经自然终档；这一批不能再宣称排出胜者，也不能靠
  事后删指标制造胜者。后续新池必须先在 full-scene probe 证明 ready 分母非零，才允许占用长训槽。
  自动 rolling 任务在本次直接训练冲刺期间保持暂停。0.5 秒 K100 已废弃版本化
  launch/receipt 和人工 SHA 对拍，改为直接 evaluator。修掉 planner 两端半配置和
  MotionLoader body-velocity 副本写入后，Pod2 对 `model_5700` 完成 100/100 题：正反手
  触球、回台都为 `0/50`，总计物理摔倒 `0/100`，全部因半秒 deadline guard 结束。
  这个 checkpoint 已被严格半秒要求否定；正手最新完整卷的 hit/return 仍为 `0/50`。

  2026-07-19 约 10:26 CST，Pod1 单次只读 SSH 已确认 L2 的 PGID `2457829` 成员数为 `0`，
  NVML compute app 为空，GPU0/1/2 的利用率和显存占用均为零。L2 先前的最终状态 `UNKNOWN`
  因而闭合为进程组与计算进程完全 absent。至此 Pod1 的 V/L2/Z3 与 Pod2 的 D2/F 均已收口，
  双 Pod Isaac 训练池结束；V/L2/D2/F 仍只记作终档 teardown，不改写成自然终档。当前工作转为
  上述 W/Y 厂商 MuJoCo 同卷的准备阶段，U 保留为稳定备选。详细证据见
  [半秒冲刺卷宗](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

  2026-07-18 20:52 CST，四条停在终档后的训练已完成精确收尾。审计改为读取 NUL 分隔的完整
  `run_name` token，并按完整日志行判断真正 fatal；V、L2、D2、F 均通过 trainer 身份、唯一日志、
  最后 iteration=`6700`、10 秒不再增长、fatal=`0` 和配方指纹核对。Pod1 的 V 已精确
  TERM→KILL 且进程组最终 absent；L2 同样只处置自己的数值进程组，NVML 已 absent、三卡显存与
  利用率均为零，但短等待后 `/proc` 仍见一个组成员，所以其最终状态仍是 `UNKNOWN`，以后只读确认
  absent/zombie，绝不再次 signal。Pod2 的 D2/F 均精确 TERM→KILL，最终组成员为零且 NVML absent；
  三卡显存为零，GPU2 瞬时 `51%` 利用率没有对应 NVML 进程，不能倒推训练仍在运行。这四条都是
  **终档 teardown 收尾**，不是自然终档；model 和结果沿用此前已取得的证据。详细运行映射与处置见
  [半秒冲刺卷宗](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

  旧 18 条 source 不能事后补出“初始准备时间（TTS）× outcome”分桶；
  新增 5 条已实际输出 `<0.5`、`=0.5`、`(0.5,0.9]`、`>0.9 s` 四桶的整数
  机会、完成、触球和合法回台计数，且 `5701–5800` 的完整 `+100` 窗已经到档、各桶分母足够。
  拍速优先×强准备姿态（V）的四桶合法回台率为 `21.96/36.03/38.21/27.01%`，完成率为
  `62.29/61.14/59.40/55.10%`；拍心优先×自由非击球臂（W）的完成率最高，为
  `87.43/87.16/85.92/82.41%`，合法回台率为 `19.41/29.90/30.75/18.97%`。拍心优先×强准备姿态
  （U）与拍心优先×触球窗老师静音（Y）呈折中，拍速优先×自由非击球臂（X）居中；没有一条在
  所有维度被同父本明确支配，因此 `+100` 不淘汰。`<0.5 s` 已不是零能力，但这些都只是 Isaac
  训练内 virtual outcome，不是 vendor MuJoCo 成绩。旧 18 臂不与新账本假配对；Z3 后来仍
  未越过首个 `Learning iteration`，已按启动挂起精确收口且不重放。二十三个单 seed 问题、判读边界和
  实际 run 映射见
  [半秒冲刺卷宗](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)；严格半秒负结果见
  [0.5 秒卷宗](experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md)。Isaac 结果仍需 vendor MuJoCo 同题。

  `5801–5900` 第二个独立 100-update 窗也已完整到档。五臂的总体“完成率/合法回台率/总 fall
  （pre/post）”依次为：U `93.80/31.29/0.956% (0.143/0.814%)`，V
  `44.19/62.59/21.80% (20.17/1.62%)`，W `94.37/30.81/0.901% (0.150/0.751%)`，X
  `46.38/59.55/21.39% (19.61/1.77%)`，Y `93.58/31.52/0.855% (0.126/0.729%)`。
  位置优先组相较上一窗完成/回台改善且 fall 下降；速度优先组回台约增 `27.56` 个百分点，但 V/X
  完成率分别下降 `16.07/24.01` 个百分点且 pre-fall 约 `20%`。当前 Y 安全最好、W 完成最好、V
  回台最高，仍无一臂在 return、completion 与 safety 三维全部支配另一臂，因此继续训练、不淘汰。
  这些合法回台仍是训练内 virtual outcome，不是 vendor MuJoCo 行为。

  `5701–6200` 的 `+500` 累计窗现已完整。U/V/W/X/Y 的“完成率/合法回台率/fall 率”分别为
  `91.68/31.42/1.432%`、`48.66/61.38/20.71%`、`93.02/30.78/1.241%`、
  `50.91/59.08/20.35%`、`92.00/31.52/1.322%`；最近 `5901–6200` 则为
  `94.47/31.98/0.612%`、`46.93/70.82/22.84%`、`94.92/31.47/0.552%`、
  `47.49/69.41/23.14%`、`94.56/32.15/0.572%`。这把候选清楚分成稳定位置组 U/W/Y 与激进
  拍速组 V/X：Y/W 位于当前稳定 demo 前沿；V/X 尚非 demo-ready，却是唯一高回台前沿。由于没有
  全维支配，按既定规则 stop=`0`。这些仍是训练内 virtual outcome，vendor MuJoCo 尚未验证。
  约 19:48 CST，U/V/W/X/Y 的 `+1000` 已完整：`5701–6700` 1000-update 窗与
  `6201–6700` 最近 500-update 窗都为 missing=`0`、duplicate=`0`。累计“完成率/合法回台率/fall 率”
  依次为 U `93.31/31.94/0.87%`、V `48.40/70.14/22.05%`、W `94.14/31.52/0.74%`、
  X `49.68/68.08/22.00%`、Y `93.57/32.31/0.82%`；最近 500 update 为 U
  `94.99/32.46/0.30%`、V `48.14/79.06/23.40%`、W `95.28/32.27/0.24%`、
  X `48.49/77.33/23.61%`、Y `95.18/33.10/0.31%`。最近窗的 `<0.5 s`“完成/回台”分别为
  U `96.42/26.15%`、V `48.88/63.26%`、W `96.61/26.39%`、X `49.22/60.31%`、
  Y `96.53/26.34%`。在完成、回台、fall、短于半秒完成和短于半秒回台五维上五臂均非支配，
  因此不按单指标杀臂。W/Y 是 demo 优先双候选，U 是稳定备选；V/X 虽保持唯一高回台前沿，
  但约 `22%–24%` fall 使其不具备 demo-ready 安全性。下一步是 W/Y 跑同卷 vendor MuJoCo，
  不再盲加 Isaac step。训练内 virtual outcome 不能代替该部署裁判。

  Z3 已精确收口并保留证据，不重放；V/L2/D2/F 的终档 teardown 也已按各自数值进程组收尾，
  但不能写成自然终档。7 月 19 日的只读复核已确认 L2 进程组和 NVML 计算进程完全 absent，
  因此双 Pod Isaac 训练池已经结束。

- **300 Hz 动捕不会再触发 300 Hz planner solve：** 正式 arena、schema-4 与 Gate3 profile 显式绑定
  latest-only worker。每个合格样本都进入 estimator 并替换唯一 pending snapshot；Stage 2/3 最多 50 Hz，
  无 FIFO、无补跑 burst，发布和 task lifecycle 留在 executor。source/no-ball/close-rearm/epoch/base revoke
  会淘汰旧 completion，普通同球 revision 与仍有效的新 base sample 不会误杀；strike-spec 诊断另用独立
  worker。全 planner suite 为 `225 passed, 2 skipped`。真实 ROS/Jazzy 300 Hz 慢求解压力、Release build
  和 planner→DDS→C++/vendor 行为仍是部署 Gate，不能把源码回归写成场馆已通过。

- **动作加速当前边界：** 完整旧 v4rg 只重定时得到 `0.98/0.78 s`，因此新增的 host-only
  ready-to-strike 生成器从动作第0帧零速准备态直接接入保真的触球前0.1秒。四元数符号跳变与
  joint-velocity 过度声明已经红队修复。Pod2 首次真实生成又在发布前暴露 source gate 假拒绝：正式
  v4rg 的完整 migration 三元组被当成额外字段，正反手均无候选、无 TOPP、无 GPU 行为；失败 namespace
  已冻结。修复版精确接受并逐位保留 canonical 三元组，仍拒绝残缺、坏值和未知字段，专项 `21 passed`。
  第二次新 namespace 已成功生成两条真实候选；production-FK TOPP hard gate 都通过，但正/反手可行
  run-up 上界为 `0.64/0.94 s`，仍没有 0.5 秒证书。下一轮已冻结 `6/12/17` 主梯、`9/14` 按需细化，
  并用两侧 minimax 区分 forehand/backhand frame0 哪个可作共同 ready。端点实测已显示 `d=17` 全面
  慢于 `d=6`；反手 own-ready 把 raw `0.94` 改善到 `0.78 s`，但正手自己的 ready 仍更好，故冻结规则
  选择两个 ready 的四个 `d=12` 中点。红队发现一次性 runner 未持久绑定 consumer 且 certificate parser
  不够严格后，独立 historical attestor 已全量重验并发布唯一 receipt
  `7cf1c7c9…c377f`：六个端点升级为可信的 screening evidence，分别为正手 `1.28/0.70/1.54 s`、反手
  `1.94/0.78/1.42 s`，仍全部高于 `0.5 s`。四个 `d=12` 中点的 v1 execute 中 generator 全部成功，
  但量尺把 candidate float32 producer-gradient 混成 TOPP float64 workspace-gradient，故四格在 TOPP 前
  fail closed；旧 summary `f92e6b8b…63c0e` 已冻结且不重放。v2 只修该量尺，绑定旧失败并使用唯一新
  CPU-only namespace，动作/join/预算/acceptance 不变。v2 唯一 execute 已证明四份 candidate 与 v1
  逐字节一致；四个 TOPP 均 rc1、没有 timing，summary `6910db28…f1476` 冻结且不得重放。这是 v2 的
  全部正式结论，诊断 `run.log` 不作根因或授权输入。v3 直接复验并复用这四份 candidate，零 generator
  调用，并从 frozen runtime Git objects 提供 `1 XML + 74 mesh` 的 exact closure（75 文件、
  14,127,373 字节、manifest `e0381752…b962de`）。但 v3 唯一 Pod2 dry-run 在结果 root 前抓到 v1/v2
  contract SHA 账本混用，未启动 execute/TOPP。v4 用新 activation/namespace 绑定 v2 四份实际 contract，
  随后 dry-run 又发现 exact log SHA 后的英文文本猜测会假拒绝真实日志，同样未启动 execute/TOPP。v5
  删除重复文本解释却仍因一份 log SHA 手抄一字符错误 pre-root fail closed。main `8b371eb7` 上的 v6
  随后移除旧 V1 summary、generator 副本和全部日志前置，只复验 v2 四份 candidate/contract，保持零
  generator 调用，并在 Git-object 完整闭包中执行唯一一次远端 dry-run/execute。dry-run 全绿；execute
  natural terminal，summary=`b5209bc7…`：四格都是 generator=`0`、TOPP rc=`1`、无 timing，`75` 文件/
  `74` mesh closure 与残留进程检查均正确。后续只读 forensics 发现四份日志逐字节相同
  （SHA=`f1d5088e…`），共同首错为 `/usr/bin/python3` 缺少 `mujoco`，因此这是 **runtime dependency
  closure** 失败，不是四个动作共同失败。项目实际 TOPP import closure 只有 `numpy+mujoco`；此前把
  `scipy` 当硬门属于过度检查，现已废除。去掉 `PYTHONPATH` 的 targeted probe 又证明
  `/workspace/hope_mjeval_venv/bin/python` 可加载 `numpy 2.5`、`mujoco 3.10` 和 exact MJCF
  (`nq=38,nv=37,nbody=33,ngeom=79,nmesh=74`)。v7 增加该解释器/包 closure 与 preflight，动作、join、
  budget 和 acceptance 不变。后续已把两份 Python package 的完整 RECORD、实际加载 ELF、每条
  `DT_NEEDED` 解析边、canonical `ldd/readelf`、MJCF pre/post snapshot 和四个 child 的 terminal 状态全部
  绑定；TOPP rc 非零时不再能假报 `mjcf_closure_exact=true`。本地 Stage-2 专项 `112 passed`，独立红队
  P0/P1 均为 0；但唯一远端 v7 dry-run 把 venv symlink 的 `readlink()` 字面 target 误当身份，在进一步
  核验 canonical binary/包闭包前就因文本不同 fail closed，结果 root/child 均未创建、execute=`0`；
  该次尝试无法判断实际 binary 是否漂移。v8
  改以 canonical realpath、binary SHA、Python version、venv prefix 与 RECORD/ELF closure 定义身份，
  科学四格不变；runner/activation=`40e89c6a…ae09/e878de11…0447`，专项 `91 passed`，远端尚未执行。
  因而仍没有 TOPP timing、
  TOPP≤0.5、L0/L1、桌网、动力学或行为通过，不能写成0.5秒动作已完成。见
  [短路径实验](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- **已停止的前代 rolling 池（只作背景）：** 旧 `24/24` 长曲线已按绑定身份停止，三份较强母本的 optimizer 被完整恢复到
  [24 格快速准备组合漏斗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。2026-07-16
  11:29 CST 的每 Pod 单连接只读审计确认 `24` 条都已消费唯一 claim，其中 `22` 条 live、fatal=`0`；两条在首迭代前由动态 URDF importer
  以 malloc `rc134` 退出，精确 PID/PGID 与 GPU context 均 absent，记为**基础设施拒绝**而不是 Reward
  失败，且不自动重跑。Pod1/Pod2 分别为三卡 `4/3/4` 与 `4/4/3`；GPU 利用率约
  `90–98% / 89–97%`，显存约 `17–23 GiB/卡`，主机可用内存约 `907/916 GiB`、swap=`0`。本轮比较约
  `1.0/0.7/0.5 s` 与随机准备时间、时间戳补偿、预测抖动、Reward 配比和脚朝向；全部是单 seed、热启动、
  demo-only 的工程候选，不能冒充正式因果结论。每 30 分钟继续检查结构/finite/恢复和 checkpoint；但
  11:29 CST 的量尺审计推翻了“现役 source 可自动执行 `+500/+1000` 行为淘汰”的假设：completion 与
  pre/post-fall 是跨历史 EMA，termination 混有非物理 guard，dense ready/balance 又缺 phase 分母与母本
  receipt，无法重建两个互不重叠的 100-update 窗。因此现役 22 条只能结构淘汰，行为状态统一为
  “量尺不完整，继续训练”；稀疏击球零值仍绝不算失败。12:45 CST 的 Pod1 单连接刷新仍是
  `11 live + 1 importer rejected`、fatal=`0`，11 份 latest `model_2100–2600` 的 embedded iteration、finite、
  hard/claim/binding 与 lineage 全过；旧 budget-v1 诊断臂现到 `model_2600`，`model_3600` 尚不存在。
  13:46 CST 的 Pod2 唯一只读 inspector 已闭合根因：第一份 `model_5200` 的 actual claim=`7878d92...`，
  自洽绑定修正 budget 的旧 runner=`428cbf...`；与现行 `aee7132.../90d7f26...` 的完整 content 唯一差异是
  `continuation_runner_script_sha256`，budget/题目/source/run/slot 全相同，绑定进程仍 `live_exact`、checkpoint
  为 regular、receipt absent。这不是 trainer 或 checkpoint 失败，也不授权直接重试。milestone consumer
  只精确白名单两份逐 job 完整重建的 corrected-budget claim，明确排除旧 budget-v1；任何第三 runner 或其他
  字段差异仍 fail closed。约 15:10 CST，reviewed consumer 对该 job 只连 Pod2 一次并成功 O_EXCL 发布首份
  `model_5200` receipt：filename/embedded=`5200`，74 个浮点 tensor、`1,762,715` 元素全部 finite，schema-3
  hard/claim/binding 对齐，lineage=`0`，取证时进程仍 live；receipt content SHA=`521910d...`。这只关闭
  checkpoint 完整性，不是行为领先。15:36 CST，第二个 job 的独立 dry-run 正确生成自身 per-job claim
  `691a52c.../0968d24...`（仍绑定 reviewed runner `428cbf.../90d7f...`），唯一 Pod2 SSH 随后也成功发布
  `model_5200` receipt：content SHA=`37d6bd2...`、checkpoint=`ff1b210...`、hard=`aa80162...`、binding=
  `7593d66...`，同为 embedded=`5200`、全部 finite、lineage=`0`、process live；第一份 receipt 未触碰。
  两条都继续、不排名、不停止。15:50 CST 的 Pod1 唯一只读连接仍为 `11 live_exact + 1 importer
  rejected`，GPU=`4/3/4`、util=`84/84/96%`、accepted fatal=`0`；11 份 latest checkpoint 均
  embedded/finite/schema-3 hard/claim/binding/lineage=`0` 且 optimizer 完整。budget-v1 PGID `2199057`
  仍 live、latest=`3300`、`model_3600` 不存在，未 signal；Pod2 同轮为 `11 live_exact + 1 importer
  rejected`、GPU=`4/4/3`、fatal=`0`，accepted checkpoints 全部通过同一完整性门。后续 source 必须增加 per-update 整数机会/
  完成/物理跌倒 union 与 ready-phase `sum+count`，现役模型若要提前排序只能另走绑定 checkpoint 的不可变
  同卷评估。`qdot` 是确定性的关节速度超限惩罚，不是随机外力；随机横向躯干推力正在
  补 trainer/物理响应门，过门后才占释放槽做同母本 no-force/force 配对。

- **V10 现场问题复核后的采用状态：** 旧 formal 179-D runner 的 active-swing target/TTS freeze、同球重复开
  task、VRPN capture age 未补偿和 planner/runner 盲盒 trace 已被 task-revision 协议取代：一颗物理球固定
  一个 `task_id`，触球前每个 policy tick 用递增 `task_revision` 原子更新目标、signed 拍面与剩余时间，
  capture age 进入外推，低频 trace 关联 ball-plane distance、source age、intercept、revision 与接受/拒绝
  原因。训练 actor 同样逐 tick 看同球修订，题库/Reward truth/critic 保持 immutable，避免把追踪噪声写成
  答案。尚未闭合的是行为层：Stage-1 来球位置仍不等于场馆 residual，positive-delay transport 仍
  NO-LAUNCH，0.5 秒 K100 和 vendor MuJoCo 尚未给出能否回球的成绩。详细边界见
  [G07 的八项现场审计](gates/G07_mujoco_to_real.md#audit-update-2026-07-16-rallyv10-field-test-timing-and-task-lifecycle-gaps)。

- **最新可分享结果：** 反手拉 B/C 的 rank-0 主选已在 Pod1 CPU-only runtime 完成 exact 整轨
  SE(2) 实体化；后续 schema-2/FK 源码门也已进入 `main` 并通过 `17` 项专项回归。逐资产 no-write
  runtime inspection 又在 exact donor/私有 PKL 下通过：B/C 分别 `91/98` 帧，未写输出。一次性 consume
  runner/activation 已进入 `main` 并通过 latest-main 全回归。2026-07-14 08:22 CST，B 在 fresh
  absence preflight 后花掉唯一一次 consume：`91` 帧 schema-2/FK NPZ 的 SHA-256 为
  `e2eb99e6...d28cc`，独立 `validate-result` 确认 `runner_lineage=true`、`npz_bound=true`，completion-last
  ledger SHA-256 为 `c0a25f2c...f4f8b`。这只解锁 B 的下一张 L0 静态证书；vendor L1、整轨桌网余隙、
  动力学与 RL 仍未跑。C 保持未消费，只在 B 后续外部安全门失败或证明有独有覆盖时作为后备，不复制
  一个 RL 槽。B 的历史绝对路径 portability 根因已修复并合入 `main`；Pod2 的 exact
  CPU-only full L0 `dry-run` 随后真正进入 151 帧 `mj_forward`，并在旧合同要求的
  float32 逐 bit 相等处 fail closed。实测最大差为 position `1.1920929e-7 m`、quaternion
  `5.9604645e-8`、COM velocity `2.9802322e-6 m/s`、angular velocity `5.9679151e-6 rad/s`；
  无证书、不占 GPU。这不是碰撞/自打/动作安全失败，而是 Pod1 producer → Pod2 重算的
  float32/50 Hz 差分可复现性合同不自洽。旧 v1 失败保留；v2 从 ULP 与差分误差解析推导，
  不放宽关节、地面、支撑脚或安全门。该 v2 已在 exact `main@cc1a2b1` 上通过 Pod2 full
  `dry-run` 与唯一 no-clobber formal audit，certificate SHA-256 为
  `60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`。vendor L1 随后在修复
  runtime→GMR 关节顺序 adapter 后完成 Pod2 full dry-run 与唯一 no-clobber audit：1201 个 400 Hz
  有限样本中自碰=`0`、拍/柄 `<5 mm` 自打=`0`、warning=`0`，最小余隙 `0.13829 m`；certificate
  SHA-256 `6840df34...db60`。下一张整轨桌网门也已在 clean Pod2 `main@c047ea7` 完成唯一 v4 audit：
  `1201 @ 400 Hz` 逐帧 `37×4` 对，hard/warning/unsafe=`0/0/0`；certificate `93fd5435...9b0e7`
  只声明诚实 saturated lower `0.099999999999 m`，pair/midpoint/time=null，并经独立复核接受。B 现只解锁
  vendor 动力学/平衡门；RL、回台、Gate3 和真机仍未授权，C 保持后备。
  高点拍压 S0 与四条横移 M0 不仅通过 exact GVHMR 帧数/finite 审计，还已完成真实五条 PT 的
  canonical-beta `inspect/consume`，non-beta 内容逐 bit 不变。2026-07-20 回收的 Pod1 evidence 又确认两批
  exact-GMR v2 实际都已完成；较后的 Pod2 rc127 仍保留为另一失败 location，不能再据它写“两批 root
  absent / 未 consume”。S0 completion manifest SHA=`a762d6df...d1a23`，唯一 `88`-frame 输出
  finite/`30 Hz`/`31 DoF` structural pass，但 ball contact/effectiveness=`null`，下一门是独立高球拍压题族。
  M0 manifest SHA=`fdd60fcf...396e`，四份 moving 输出同样结构通过，却全部 stance fail (`0/4`)；因此
  M0 input gate 为 reject/no-launch，不得进 schema-2 或占 RL GPU。下一步先修复横移动作，在保留左右
  位移的同时回到各自初始 stance；S0/M0 随后才分别进入 schema-2、L0/L1、桌网和动力学。
- **当前运行态：** 2026-07-15，V1+V2×base-decel fresh v4 两臂已收口。两份 `model_500` 都通过
  filename=embedded、finite、fresh lineage、claim 与同 hard-contract attestation，checkpoint SHA-256 为
  `22f78f88...a6a` / `a1735fbb...c14`。但冻结的 `480–500` 窗内 control post-swing
  eligible/selected/started=`0/0/0`，treatment 已为 `15087/3750/3750`；control 到 step519 才激活，不能
  倒灌 model-500。源码确认 buffer 只收 policy 自己活到自然 clip wrap 的状态，因此 base-decel 会反过来改变
  curriculum 何时 ready，这一对按预注册判 `activation-invalid`，不比较行为、不买第二 seed、不判卷。
  exact PGID=`380610/381237` 已停止。2026-07-15 04:15 UTC 的只读快照中，Pod2 GPU0/1 分别仍是
  Yikang V9/V10（PID `379550/396374`），Codex 没有触碰；GPU2 已运行三条同源 10000-update 长曲线：
  关节速度边界惩罚 PGID `411519`、击球窗模仿放松 PGID `412204`、普通对照唯一重试 PGID `412899`。
  三条分别到 iteration `24/9/2`，fatal=`0/0/0`，claim/binding 都存在；GPU2 为 `97%`、17154 MiB。
  200/500/1000 只早筛，2000/3000 看中段，6000/10000 才看完整曲线，不买第二 seed。任何 Yikang
  进程仍在的 GPU 都保持保留。下一次同轴训练必须先用
  两臂共享、自然-wrap provenance 绑定的 immutable teacher-state receipt 做外生 cold-start，并在首 update
  前 fail closed。

  2026-07-15 15:49 UTC 的更新取代上面的 04:15 资源快照：Yikang 在 Pod2 GPU0/GPU1 的 compute PID
  已自然为空，因此 Codex 按 GPU0→GPU1 逐圈各一条补齐六格 10000-update 单 seed 长曲线。V1-only、
  `qdot=-1`、V2-only 唯一 retry、`qdot=-2.5`、脚部朝向惩罚 `0/-0.6` 的 exact PGID 分别为
  `419643/420298/423502/421479/422126/422783`；连同 GPU2 原三条，Pod2 三卡各恰三条 trainer，
  利用率 `97%/97%/91%`，九条 live 日志 fatal0。V2-only attempt-1 PGID `420947` 在首迭代前的动态
  URDF import 以 malloc rc134 退出，无 checkpoint；证据保全且 namespace 永不复用，不是 Reward 失败。
  这六格补齐“单独/组合放松动作模仿”、关节速度和脚部朝向三个因果问题，不复制 seed。200/500/1000
  只判结构；只有真实击球后才有意义的 Reward 若 eligible hit 样本不足必须继续，6000/10000 才看完整
  单 seed 曲线。[六格实验](experiments/2026-07/EXP-P1-LONG-SCALEOUT-SIX-ARM.md)

  2026-07-15 16:40 UTC 的历史快照中，Pod1 也曾按 GPU0→GPU1→GPU2 四圈铺满 12 条不同问题的
  10000-update 单 seed 长曲线：非击球臂是否继续模仿 × 10/16/24 秒连续 episode，以及六种击球
  位置/速度/拍面 Reward 配比。两个首发 namespace 在动态 URDF import 的 180 秒 stale 门失败，均为
  0 iteration/0 checkpoint；各自唯一同配方 retry 已过首迭代。现为三卡各四条、利用率
  `97%/93%/97%`，12 条接受臂 PID=PGID、fatal0。Pod2 同时保持九条三卡满池；两 Pod 合计 21 条。
  当时只证明生产池已铺满，不把早期稀疏回球零值判成失败；该 long-grid 后来已在 task-revision
  cutover 收口，2026-07-20 不再运行。
  [Pod1 十二格实验](experiments/2026-07/EXP-P1-POD1-LONG-BALANCE-REWARD-GRID.md)

  clean main-effect 也已自然终档：两臂关闭 post-swing replay、固定 V1+V2，只比较 base-decel `0/1`。
  `model_1000` 两份 filename=embedded、finite、fresh lineage、claim 与共同 hard contract exact，原
  PGID `385320/385948` 均已退出。980–1000 的 treatment/control raw base speed=`1.00882×`，按冻结
  `≤0.90×` 门正式 reject，不买第二 seed/judge。源码复核同时发现这个 Reward 实际追踪随目标距离变化的
  `v_des`，并非始终让 raw speed 更低；尾窗 raw-kernel-per-eligible 反而提升 `1.6003×`。因此不改写
  verdict，但后续必须先用 `|v_base-v_des|` 的近/中/远分桶重做量尺，不能复制当前 weight=`1`。

  连续恢复当前最短关键路径已完成共享外生 teacher 制品：v3 在 Pod2 GPU2 自然收满 4096 条
  `natural_clip_wrap` state；唯一授权 attestor attempt-2 从固定 `a38b7e9` rc0，发布 4103-byte receipt
  `e20a6989...d2aba4`。merged-main controller status 已确认 `teacher_receipt_binding_exact=true`。当前只差
  4096-env、首个 PPO rollout 前的 first-reset 采用率与 simulator root/joint readback probe；它未通过前，
  科学 pair、第二 seed 与 judge 继续 fail closed。其后的“仅 Pod2 GPU2 可用/三槽已满”只是当日资源
  快照，已由 2026-07-20 六卡 NVML 空闲状态取代。

  Fresh C 的五条单 seed 机制格均已越过 `+500` 且 checkpoint
  finite/contract/lineage 正确。V1+V2 出现当前最强击球精度信号（composite `0.0893`、normal pass
  `0.268`），但平衡债仍高；V2 单独格明显落后，已列为可替换且不复制 seed。base-decel、V1 和
  post-swing 保到 `+1000` 看权衡是否收敛。qdot-limit 第一次发射在第 0 update 的 A3 URDF import 阶段
  超时，launcher 只收口其 exact PGID；无 hard contract/model，因此是基础设施失败而非 reward 失败，
  全新 retry-v2 已通过 no-Kit compose 与真实 boot marker。qdot treatment/control 均已自然到
  `model_1000` 并退出；control 终档 SHA-256 `b6672869...12cb9`，filename/embedded iter=`1000`、76 tensors/
  1,762,717 elements finite、fresh lineage=`1`、schema-3 hard contract 与 claim 均 exact。两份 `model_500`
  的 checkpoint/claim/hard-contract/finite 也全过。`480–500` 的末 21 点里 treatment 的 qdot max
  `-16.4%`、near-limit `-20.1%`、torque saturation `-35.5%` 且 fall 改善，但 position pass
  `0.418→0.107`，当时是 mixed signal。`980–1000` 的同口径曲线随后翻转：treatment/control 的 position
  pass=`0.878/0.593`、position error=`4.74/9.62 cm`、signed composite=`0.310/0.146`、virtual
  return=`0.454/0.265`，而 pre/post fall 与 completion 基本持平。这把 `-5` 从“准备调低权重”改判为
  **晚熟候选**；仍不直接采用、不买第二 seed，也不启动交互，先对两份 finite `model_1000` 跑 immutable
  MuJoCo/vendor judge。
  qdot control 的首次冷启动随后在 scene creation 前卡住，iter0/无 contract；exact PGID 已收口并保全。
  仅允许相同配方的 retry-v2 再试一次；若同 phase 重复，停止 retry并转 importer 根因线。
  新 source 的正式 run 前现先走独立 `boot-warmup`（1 env×2 update、180 秒、非科学 namespace），避免再拿
  control/treatment 的证据目录承担冷缓存失败。
  通用 launcher 也已加入 180 秒日志无进展 watchdog；marker 优先，失败只收口自身 exact PGID并写 sidecar。
  conditional source 的 Pod2 GPU1 1-env warmup 虽自然退出并通过 2/2 updates，但随后 4096-env control
  在 dynamic URDF import 后停住，iter0、无 scene/contract/checkpoint；精确 PGID 收口并保全，treatment
  从未创建。这个反例推翻了“1-env warmup 可授权正式 run”。后续 `077e70c` 非科学 probe 又因 clean
  checkout 缺 Git-ignored A3 runtime tree 在 iter0 自然失败；它不是 4096-env 容量或 Reward 负结果，原
  attempt 永不重放。46-file A3 tree 的 YAML source closure/no-clobber hydrate 随后闭合，`main@c7e1a90`
  canary 已自然越过首迭代并终档：result/model/hard-contract SHA-256 分别为
  `02780b52...c4186` / `a813ea9b...38e68` / `c39cf1ae...df838`，76 个 tensor 的 1,762,715 个浮点元素
  全 finite，fatal=0，原 PGID 已自然为空。但它的 `unlock_authorized=true` 只符合 c7 旧终档语义，不能
  解锁科学训练。当前 conditional `0/-0.4` 与 `V1+V2 × base-decel` 两组单 seed pair 已改绑 strict
  `main@caeb9ad` checkout `/workspace/codexschema/nohope_p1_caeb9ad`。新 attempt
  `caeb_strict_terminal_pod2_gpu1_a1` 已真正通过：result/claim/model/hard-contract SHA-256 分别为
  `0d03bd03...9b1d1` / `7437db48...ad5` / `e1b79d14...3106` / `c39cf1ae...df838`，并绑定实际
  4096 environments、物理球/三实体、76 个 tensor 的 1,762,715 个浮点元素全 finite、fatal0 与自然空
  PGID。该非科学 receipt 已被显式队列变更消费：两对均为 ready、
  [`launch_authorized=true`](DEFINITIONS.md#launch-authorized)。conditional control/treatment 随后已分别在
  Pod2 GPU1/GPU2 越过 first iteration（PID=PGID `357023/357679`）；两份 `model_200.pt` 已写入 exact
  milestone receipt，checkpoint SHA-256 分别为 `b55b7d3b...b4b41` / `c07b1f12...bd51`，76 个 tensor、
  1,762,715 个浮点元素全 finite，fresh lineage/claim/schema-3 hard contract 均匹配。冻结
  step `180..200` 后，treatment 的 conditional gate/cost/reward mean/min/max 全为零，由源码公式可
  严格推出 eligibility 为零。这一 setting 已按预注册判 activation-invalid：不买第二 seed、
  不晋级，也不把方向差异写成 Reward 效果。
  interaction control PID=PGID `358331` 则在 first iteration 前的 dynamic URDF import 以
  `malloc(): invalid size (unsorted)`、`rc=134` 自然退出；treatment 未发射，claim/namespace 已保全。
  这不是 interaction Reward/行为失败，也不能写成 pair 已运行。旧 control 行已 rejected、禁止重发；
  逐字相同配方的 `control_retry_v2` 与从未 claim 的 treatment 均 ready，只允许同一 `fill --count 2` 事务
  先观察 retry first iteration 再发 treatment。该事务随后按序成功：retry-v2 PID=PGID `359240` 在 Pod2
  GPU1、treatment PID=PGID `359872` 在 GPU2，二者均已越过 first iteration。两份
  `model_200.pt` 也已通过 finite/iteration/lineage/claim/hard-contract receipt，checkpoint SHA-256
  为 `44a709ac...035a` / `b04e2338...e56b`。但 V1/V2/base-decel 的计数级 activation 仪表不全；
  V2 treatment 窗口 proxy 为零，base-decel Reward 虽非零也不能替代 denominator。该 pair 按
  activation-invalid/instrumentation-blocked 记账，不解释为 base-decel 负结果；旧 `358331`
  永久 rejected。
  “每卡只能发一条”的 launch-lock 根因已修：旧 `flock FILE command` fd 被 trainer 继承；新 controller
  用 fd8 持短锁并对子 launcher `8>&-`。conditional control/treatment 优先同落已 warm 的 Pod2 GPU1，
  正好验证容量不再退化为1；现役 qdot 的旧锁不做任何强制处理。

  2026-07-14 19:00 CST 起 Pod1 全部留给 Yikang 冲刺：Codex 的 V1/V2/V1+V2 只对精确 PGID 发出
  `TERM`，分别停在 iter `792/782/743`，三条 `model_700.pt` 与日志完整保留，未进入 `KILL`；复核 Pod1
  三卡无 Codex compute process。Codex 后续只使用 Pod2。qdot treatment/control 均已自然终档；c7/caeb
  canary 也已自然退出。最近可信 Pod2 快照中 GPU0 只有 Yikang 的外部 trainer，Codex 不管理；GPU1/GPU2
  分别运行 conditional control/treatment 与 interaction retry-v2/treatment，每卡两条 Codex trainer。机器可读
  [`dispatch_pods: [pod2]`](DEFINITIONS.md#dispatch-pods) 已使新 assignment 不可能
  落到 Pod1，普通 live snapshot 也不会读取 Pod1。

  2026-07-15 03:17 CST 的 Yikang 训练快照：纯 A 继续在 Pod1 GPU0（W&B `5nso93g0`）；旧的
  root-velocity B/AB/AB_FRESH 已在保全 finite checkpoint 后按 exact PGID 停止。物理
  `pelvis_link` force 替代组使用 branch `yikang-standhit-0714` 的 exact source
  `0db9a9a8bf27b3389080edef456551257c08a170`：B_FORCE 在 Pod1 GPU1（`e0jg8n35`）、AB_FORCE
  在 Pod1 GPU2（`qfn5xlnr`），两者从原始 `ayzxv1ma/model_10600` 严格恢复；AB_FORCE_FRESH
  在保留给 Yikang 的 Pod2 GPU0（`147kd92u`）从 iteration zero 启动。三条均为 8000 envs/seed 0，
  10 秒 timeout 与 fall termination 未改；更长 recovery horizon 仍是独立 C 实验，不混入本矩阵。

  发射 harness 已在 `main` 收紧：YAML recipe 先拒绝重复/owned key 和 Hydra 控制语法，真实最终 argv 在
  claim 前做 no-Kit compose，run directory 原子创建，canonical claim digest 自动绑定 source/argv/
  预算/motion/bank/exam 并进入 checkpoint provenance。trainer-owned
  [`run_binding.json`](DEFINITIONS.md#trainer-run-binding) 与
  [milestone attestor](DEFINITIONS.md#milestone-attestor) 已进入 main；代表性 full-scene probe 也已实现。
  full-scene finalizer 现在不再信任外层 wrapper 的旧 doctor 结论：它在 terminal 末端自行重算
  target/donor inventory、URDF mesh closure、donor clean commit 与 receipt，并把 `current_closure`
  写入 immutable result；direct-finalize 绕过、资产漂移与 bool iteration/lineage 均 fail closed。
  Pod2-only probe 的外层快照现在只读 selected dispatch slot，不再因普通 fill 的全 Pod 快照访问 reserved
  Pod1；`fill --execute` 每臂也已从“独立 doctor SSH + 内嵌 doctor launch SSH”收为一次原子远端调用。
  ignored asset prepare 与 strict caeb 自然终档已经闭合；两组 pair 的 `+200` 身份和冻结曲线也已复核。
  两组都因 activation 为零或仪表不完而不能产生机制结论；当前 exact PGID 在替换 source
  过门前仍 live，不会复制 seed。下一对 post-swing 概率消融已有 5 项整数计数源码，但红队发现
  共同 V1/V2 计数仍缺，故不跑 strict probe、不发射；先补成真正可早判的 source。

  非击球臂 A0/A1 已完成 checkpoint 层闭环。A1 自然退出；
  A0 的 `model_1000.pt` 写完后在 Kit/Python teardown 挂起近三小时，正式 failure regex 无命中，终档
  内嵌 iteration `1000`、`1,762,715` 个浮点元素全部 finite、fresh lineage `1`，hard-contract SHA
  与相邻文件一致。它对精确 PGID `1811464` 的 `TERM` 20 秒无响应后，只对同一个单成员 PGID 发出
  `KILL`；没有重启、重发或影响别的进程。既有 v1r1 finalizer 随后通过两臂
  `200/500/1000` 的 filename↔embedded iteration、finite、lineage、hard-contract 与唯一 body-mask
  差异，发布 paired result SHA-256 `30ba716b...d7d9`。结果仍写明 signed K100 尚未判卷、不得晋级或买
  第二 seed。两边归档 checkout 仍保持 clean exact `6d93bcb...480b`。

  C2 虽自然终档，但 2026-07-14 的 v1r2 runtime gate 发现它**不是所声明的零摩擦配方**：manifest 写
  `zero_joint_friction=true`，实际 launch argv 没有 `task.plant.zero_joint_friction=true`，相邻 hard
  contract 的 31 个系数全为非零 PhysX 默认值。v1r2 在任何 attestation/claim/D2 run 前 fail closed；
  D2 永久不续跑，C2 只保留为 nonconforming 根因证据。全新 C3/D3 已在 Pod1 GPU1/GPU2 各只发一次并
  自然到 `model_24.pt`；31/31 零摩擦、finite/iter24/lineage/claim/hard binding 与 paired receipt
  `bb3cd749...bbde` 均通过。它只证明 L1 provenance，下一步是同一 immutable K100，不得重跑或直接
  晋级 L2。2026-07-14 的 Pod1/Pod2 每卡 `4/3` trainer 是当轮临时容量，不是永久分区；当前每次发射都
  重新读取六卡占用并按无固定归属的队列纪律分配。空槽只给已过前置门且有预注册
  早判的不同机制，不复制失败
  seed，也不拿未过动作安全门的任务凑数。
  signed-face exam bank 已过 E2：371 题 old/new replay 逐字节一致并发布新 bank/report；K100 paper
  已物化为 100 个唯一题、正反手各 50。C3/D3 actual L1 与 paired receipt 已闭合；下一门是消费同一
  immutable K100，不得重跑 L1。L2/judge/第二 seed/晋级仍阻断。
- **Franco focus：** 五种动作的用途、动作专属来球题族、空挥视觉锚点和横移终态站距语义；反手拉
  B/C 先补证，高点拍压作为第五动作，v12 只作后续 Jiayi 对照。
- **Jiayi focus：** V13 目前无运行/行为证据且分支不可整体合并；只允许把经审计的候选思想在最新
  `main` 选择性重做，再用相同挡球专卷和厂商 MuJoCo 与 Franco 主线对照，不能用版本号直接晋级。
- **Yikang focus：** V9 COM-force 语义失败，V10/V11 只有终档/代理材料而无正式 Gate3；下一步是独立
  physics/reference oracle 与当前 179-D planner-policy tuple 的 vendor runtime 行为，不续写旧分支结论。
- **Codex 执行：** 六卡任务发射/里程碑早判 harness、单 seed 机制漏斗与失败槽替换；并行推进 exact
  动作 lineage、B 主选/C 后备的逐层证书、S0/M0 exact GMR 前置、C3/D3 零摩擦与 signed-face
  checkpoint execution contract 和 main 账本；
  每通过一层就合 main，不把多个未验层绑成一个大任务。

</details>
