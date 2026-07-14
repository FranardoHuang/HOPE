# EXP-V9-YIKANG-CROSS-LEARNING-20260715 — Jiayi V9 与 Yikang 部署支线有哪些机制值得选择性学习？

- 状态：`completed`（只读跨分支审计；没有选择性移植或行为复跑）
- 阶段/轴：连续能力线 / 等待、恢复、再接战；部署验证线 / planner-policy 适配
- 集成小目标：上一拍收尾后稳定回到动作可共用的准备状态，并在下一题到来时不牺牲击球质量
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（精确提交的源码/配置/历史记录审计；没有本实验 runtime）
- 创建日期/最后复核日期：2026-07-15 / 2026-07-15

共享术语见[术语与人话对照](../../DEFINITIONS.md)。本文的 `V9` 指 Jiayi `hitter` 路线中第九版
连续挥拍配方，不是课程阶段、证据等级或当前 adopted setting；[`Gate3`](../../DEFINITIONS.md#部署与全链路术语)
指厂商 MuJoCo 中 planner、生产 C++ runner、消息和 policy 的全链彩排。

本文**不维护优先级或算力影子队列**。是否实施、何时排队以及人类责任变化，仍只由
[`origin/main` 的 NOW 统一队列](../../NOW.md#统一工作队列唯一优先级账本)决定。本文也没有合入任何代码、
没有修改当前 setting、没有连接 Pod、没有运行训练/仿真/真机，更没有产生新的行为结果。

## 问题与审计边界

问题不是“能否把另一条分支整体合进来”，而是：哪些做法表达了可跨动作、跨 checkpoint 复用的控制原理；
哪些只能作为当前动作/模型的配对消融；哪些是为了某个 checkpoint 或旧 harness 勉强接通的 quick fix。

本次以 `origin/main@c994489e6f4a5ae5e59fd26d2bec92a9f924ed10` 为基线，逐一核对下列提交；
`git merge-base --is-ancestor <commit> origin/main` 对七项均返回 false，所以它们都是 off-main 证据，
不能写成 main 已具备或 Gate 已通过。

| 精确提交 | 只读核对到的内容 | 本文判定 |
| --- | --- | --- |
| `8c8cd530a482d588f28aec96b02d61805a1e3c3e` | 从 `model_10600` 续训的 reach-box、root 速度推扰及二者组合矩阵；含 episode 级调度/计数 | 调度与计数思路可借；直接改 root velocity 的扰动语义拒绝 |
| `fbf245ea9c0d245e6d47c5438c4515581a2b6e30` | V9 的定向 x 恢复、二维 station 目标速度、航向/角速度 debt，以及 reach 支撑 | 三个 Reward 核心可作独立配对候选，不整体移植权重 |
| `05ff64b9b16109a931f2085316e60d097ddbf09e` | hold 时模仿当前 clip 第 0 帧的上肢准备姿态；腿明确排除；站距项权重保持 0 | “准备态来自动作入口”可复用；具体关节集合与权重需按动作族消融 |
| `bfa2f2186548f94509aa2251585b57f4e37c7923` | v12fix-era Gate3 handoff、重复 checkpoint 比较、per-side/tempo 诊断和旧本地 harness | 重复判 checkpoint 的纪律可复用；含 broad `pkill -f` 的 harness 明确拒绝 |
| `609ddc8443c2f29b24b0ca4affaf8d3aa032ad98` | 为 v12fix 模型加入正反手不同击球 x 平面与 side split | model/planner 几何绑定候选；不能越过 main 的同 tick tuple/epoch 合同整包合入 |
| `8615eb57802cbf63dc57837c2f7bc7b4c2b74ff8` | runner 增加 `q_des` 软限幅；`q_des` 是 policy 发布的期望关节位置 | checkpoint 绑定开关；已有相反 A/B 结果，禁止设成全局默认 |
| `12f78ac1cb7659ebcd12d1c12ae8f42c8a00be5c` | 部署数值安全和 planner 输入校验加固 | 值得对 current main 做逐块 diff/red-team；不能从旧分支整包 cherry-pick |

## 第一性原理分类

### A. 可复用的通用机制

1. **只惩罚“继续变坏”的速度，给纠正动作留出通道。** `fbf245e` 的 `windup_x_recovery_debt`
   在机器人离 station 较远时只收费离 station 更远的 x 速度；朝 station 回去的速度免费。进入死区后才约束
   双向速度，避免穿过目标。`rally_heading_debt` 对 yaw/yaw-rate 使用同一原则。这个结构比无条件
   `|v| -> 0` 更通用，因为后者会把机器人冻结在错误位置或错误朝向。

2. **把“移动”和“停稳”写成同一个连续目标。** `pre_strike_station_settle` 不是单纯压低底座速度；
   它先令期望二维速度朝 station，且随距离缩小连续收敛到零。它直接对应“先回准备位，再稳住”的控制目的，
   比 scalar base-speed Reward 更能区分有效纠正和无效晃动。它与当前
   [base-decel measurement rerun](EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)属于同一作用阶段，
   因此不能在没有交互消融时直接叠加。

3. **等待时长必须是外生且有随机长尾。** V9 继承的 V8 配方含随机 hold 和 3–6 秒长 hold 尾部；
   policy 不能通过拖延自己延长拿分时间。这正好补足“多挥几拍才暴露站窄/站歪”的稀疏学习机会，但长 hold
   只能先作为环境轴测试，不能把它误写成 learned reset 已解决。事件和 carry-state 边界仍以
   [T1 合同](../../interfaces/t1_event_training_contract.md)与
   [Recovery A/B/C](EXP-RECOVERY-TUPLE-ABC.md)为准。

4. **准备姿态应与动作入口连续。** `05ff64b` 用所选 clip 的第 0 帧作为 hold 上肢目标，解决 hold 期姿态
   无约束而垂臂/扭臂的问题。这一接口与 Franco 动作都尽量采用共同开始/结束姿态的设计一致。可复用的是
   “动作入口定义准备态”，不是“全身照抄某条 clip”：原实现刻意排除腿；对 Franco 五动作还必须分别核对
   第 0 帧、末帧、非击球臂自由度和横移老师的起始脚距。

5. **checkpoint 必须由部署裁判反复选，而不是只看训练曲线。** `bfa2f21` 记录同 lineage 的较晚
   checkpoint 在 Isaac 曲线近似不变时，branch-local Gate3 诊断反而更抖。可复用结论是同一 immutable
   题表至少重复三次、保存峰值 checkpoint，并同时报告摔倒、骨盆高度、漂移、恢复和回球分母。
   但这些旧 harness 含 broad process matching，本项目当前仍以
   [current exact-179 Gate3-D0](EXP-GATE3-CURRENT179-D0.md)为权威；分支自述不能替代它。

### B. 需要单变量或交互消融的候选

| 候选 | 最小可证伪比较 | 为什么不能直接组合 |
| --- | --- | --- |
| 定向位置/速度恢复 | 无该 term vs 仅启用该 term；同 checkpoint、动作、题库、hold 分布 | 可能与 station settle 重复收费 |
| 二维 station settle | scalar base-decel vs vector desired-velocity；固定总 Reward 预算 | 两者同在击球前恢复阶段，权重会改变纠正速度与停稳的折中 |
| 航向/yaw-rate debt | 关 vs 开；分别看向错误方向继续转、纠正转和进入死区后的过冲 | 可能与 ready-pose imitation 同时控制腰/躯干 |
| clip 第 0 帧准备态 | 无准备态模仿 vs 仅躯干+击球臂 vs 双上肢；腿保持独立 | 双上肢模仿可能抵消非击球臂用于平衡的自由度 |
| 随机长 hold | 原时长分布 vs 加长尾；Reward 不变 | 先验证暴露稀疏平衡债是否增加，不能同时换 Reward 后失去归因 |

前三个候选在同一恢复/等待阶段相互作用。先做单变量 activation，再对存活项做固定总 Reward 预算的组合；
不能把三个单项正方向直接相加宣布恢复完成。真正的联合完成标准仍是同一 no-reset 序列中同时通过：上一拍
收尾、等待姿态、任意合法时刻下一拍就绪，以及击球质量非退化。

### C. model-bound quick fixes 与明确拒绝项

- **正反手不同击球平面：model-bound。** `609ddc8` 暴露“不同动作 reach depth 不应被一个 x 平面强行统一”
  的真问题，但数值来自 v12fix clip/model。current main 已有 side split 与 hysteresis，且 planner-policy
  采用更严格的同 tick snapshot/epoch 合同；后续只能把 per-side plane 做成 model metadata 绑定的选择性
  变化，不能整体移植旧 `node.py`。
- **`q_des` soft clamp：checkpoint-bound。** `8615eb5` 的开关在一条 lineage 上缓解 hold fall，
  但 `bfa2f21` 的 `TRAINING_ISSUES.md` 同时记录另一 checkpoint 打开 0.9 clamp 后更差。它只适合每个
  checkpoint 的明确 A/B，不能成为 runner 全局默认，更不能代替训练侧稳定性。
- **放宽 station gate 或 velocity gate：quick fix / reject by default。** branch-local 记录显示放宽
  station margin 会把训练分布外命令送给 x-locked policy 并导致 walk-off；大 velocity margin 也只是在
  teacher 与 planner 物理需求不一致时补洞。正确长期解是动作题库、planner 和 policy metadata 对齐。
- **旧 Gate3 launch scripts：拒绝。** `bfa2f21` 的 local scripts 使用多条 `pkill -9 -f`，违反 exact
  PID/PGID ownership；只保留其题目设计和指标，不运行、不合入。
- **直接写 root velocity 的随机推扰：拒绝。** `8c8cd53` 调用 Isaac Lab
  `push_by_setting_velocity`，把 root x/y 速度直接改写。它不经过 `F=ma`、不表示力作用点/持续时间，
  也会绕开足接触与躯干惯量的真实响应，难以与 vendor MuJoCo 做物理同义比较。

## 随机躯干横向力：替代接口与首轮消融

用户提出的平衡学习稀疏性判断成立为一个可证伪假设：如果多拍后才偶尔出现窄脚、歪站或漂移，当前训练
对恢复策略提供的有效样本太少；在恢复/hold 期施加受控横向扰动，应该提高恢复机会密度。但首轮应测环境
机制，不先改 Reward。

建议的预注册接口如下；这里只冻结语义，不冻结数值或授权发射：

1. 在 parser 解析并内容绑定的**躯干刚体**施加外力，方向沿机器人当前 heading 的本体左右轴 `±Y`，
   左右等概率；禁止直接写 root pose/velocity。
2. 先随机一个有界的目标 impulse 强度，再按当时 total mass 做归一化；同时随机一个有界持续时间，使用
   `F = J / duration` 逐 physics step 施力。这样 mass domain randomization 下仍能比较相近的速度扰动，
   又保留真实足接触和全身惯量响应。
3. 第一轮只在击球后 recovery 或下一拍 hold 窗施加一次；“episode 任意时刻受扰”是后续独立轴，不能与
   首轮混在一起。episode reset、提前终止和作用窗结束都必须显式清零外力。
4. 记录 eligible episode、scheduled、applied、window-ended-before-apply、实际方向、force、duration、
   impulse 和 cleared 计数；缺 denominator 或 `applied != cleared` 时整臂 activation-invalid。
5. 最小 pair 是 Reward/动作/题库/checkpoint 全相同的 no-force control 与 torso-force treatment。
   两边都跑 clean exam；treatment 再跑训练带内扰动和不重叠的更强 held-out 扰动。

首轮主要指标是：物理摔倒与 guard reset 分列、恢复到 station/heading/低速死区的时间、COM 或 capture-point
相对双足支撑域的最小裕量、双脚水平分离向量及“额外变窄”、足朝向/接触、下一题揭示后的 ready 成功率，
以及分动作击球/上台不退化。只提高抗推不算成功；clean 击球和无推连续表现必须守住。

`8c8cd53` 可借用的仅是“一次 episode 最多一次、reset 时抽样、计数 scheduled/applied”的 harness 结构；
它的 velocity-write 实现和 `[-0.20,0.20] m/s` 数值都不继承。实现前还要给 Isaac 与 vendor MuJoCo 分别
定义同一力/impulse 合同，并用静态反例证明 force clear 与 body binding fail closed。

## “两只脚都要上前”核查

在上述七个精确提交及其涉及的 V8/V9 Reward/config 中，没有找到一个独立、可验证、名为或语义等价于
“两只脚都要上前”的 Reward/事件。能找到的是近击球窗的 lower-body plant imitation、双足接触/滑移、
station movement 和 stance-width 诊断；这些都不等于要求两只脚向前，而且源码注释明确警告过早/过强的
plant imitation 会阻止必要移动。

因此本文不把该尝试归功于任何提交、不提议 merge，也不从视频现象反推动机。它可能存在于未推送工作树、
未列入本审计的历史 commit 或口头实验中；只有拿到 exact commit/run/config 后才重新分类。

## 选择性集成判断

| 类别 | 当前决定 | 进入 main 前的最小门 |
| --- | --- | --- |
| 定向 recovery、vector settle、yaw debt | `ablation candidate` | 在 current main 重新实现为默认关闭；activation 计数、同 phase 配对和击球非退化门 |
| 第 0 帧 upper-body ready | `ablation candidate` | 按动作族绑定 frame0；与非击球臂自由度、横移起始脚距做交互测试 |
| 外生随机长 hold | `reusable environment axis` | 不改 Reward 的单变量 pair；同一 no-reset exam |
| 躯干横向力 | `proposed replacement` | force/impulse 双引擎合同、clear 负测、control/treatment activation 后才可训练 |
| repeated Gate3 checkpoint selection | `adopt evaluation principle only` | 使用 reviewed exact-owned supervisor 与 immutable paper；禁止旧 broad-kill harness |
| per-side hit plane | `model-bound integration candidate` | metadata/clip/题库/planner-policy tuple 同时绑定；current-main fresh red-team |
| `q_des` soft clamp | `checkpoint-bound A/B only` | 每模型 clean + Gate3 重复卷；默认保持原语义 |
| `12f78ac` numeric/input safety | `selective source-audit candidate` | 与 current main 逐函数去重、攻击测试、portable Release；不整体 cherry-pick |

这与较早的 [Yikang selective-integration audit](../../research/yikang_selective_integration_20260712.md)
采用同一原则：移植机制，不移植旧基底、旧权重或未经当前合同验证的行为结论。

## 决定

- 决定：`inconclusive for code adoption; selective ablations identified`
- 是否已纳入当前 setting：`no`
- 是否有代码 merge：`no`
- 是否有新的训练、Gate3 或真机行为结果：`no`
- 下一个 gate：若 NOW 统一队列认领该工作，先实现默认关闭的 torso-force source/activation 负测；Reward
  候选各自单变量通过后，再按固定总预算测试同阶段组合。部署侧只把 repeated-checkpoint 纪律移到当前
  exact-owned Gate3 harness，不复用旧脚本。

## 复现与证据

本审计只使用 Git 对象与当前 main 文件；不需要 ignored asset：

```bash
git fetch origin main hitter yikang-standhit-0714
git show --stat 8c8cd53 fbf245e 05ff64b bfa2f21 609ddc8 8615eb5 12f78ac
for c in 8c8cd53 fbf245e 05ff64b bfa2f21 609ddc8 8615eb5 12f78ac; do
  git merge-base --is-ancestor "$c" origin/main || true
done
git grep -n -E \
  'windup_x_recovery|pre_strike_station_settle|rally_heading_debt|hold_upper_pose|balance_push' \
  fbf245e 05ff64b 8c8cd53 -- hope_training
git grep -n -E 'x_hit_fh|x_hit_bh|side_split_y|qdes-soft-clamp|qdes_soft_clamp' \
  609ddc8 8615eb5 -- hope_ws agi
git grep -n -E 'pkill|killall' bfa2f21 -- agi/a3_deploy_example/scripts/gate3_v12fix
git diff --check
```

以上命令只证明本审计引用与分类可复核；它们不运行 simulator、policy、planner、Gate3 或硬件。
