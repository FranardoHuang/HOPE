# EXP-EFFECTIVE-REWARD-CAUSALITY-20260727 — 当前看起来学得好，是正确调参还是意外有效？

- 状态：`preregistered`
- 阶段/轴：Reward 配方真值 / task 自洽性 / paired causal A/B
- 集成小目标：先证明 trainer 真正吃到哪套 Reward，再区分 task 自洽与权重大小的贡献
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（source/config/receipt 审计；配对训练未跑）
- 创建日期/最后复核日期：2026-07-27 / 2026-07-27

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的
[`effective Reward recipe`](../../DEFINITIONS.md#effective-reward-recipe)指 Hydra/config compose
和所有 override 之后，Isaac 实际收到的 active Reward callable、weight 与 params，不是某个
`reward_pack` 标签或设计文档中的名义表。

## 结论先行

目前不能说“高质量权重调对了”。相反，现役 task YAML 显式值覆盖了 v2 pack 的名义冻结值：

| 通道 | v2 pack 名义值 | 实际 composed 值 | 实际/名义 |
| --- | ---: | ---: | ---: |
| 球拍位置 | 393.4 | 4.0 | 0.01017× |
| 球拍速度 | 295.1 | 0.5 | 0.00169× |
| 球拍拍面 | 229.5 | 0.5 | 0.00218× |

所以“以 393.4/295.1/229.5 训练得很好”这一叙述是错误的；显式 task key 后写后赢。按同一 v4rg
probe 的量纲估算，实际 quality potential scale 约 `0.0655`，名义 pack 约 `7.3838`，相差约
113 倍。这个估算只解释配方量级，不是训练成绩。

当前最合理、仍待 A/B 的解释是：

1. **主要改善来自 task 终于自洽。** 旧随机 racket velocity 与解析 landing 目标会互相打架；
   inverse-solved/单动作 task 让参考动作、目标拍速和结果方向首次相容。
2. **意外的低 quality 权重没有阻止学习，甚至可能避免了高方差冲突。** 但没有 matched run
   不能说低权重更优。
3. analytic Reward、metric 和题库共享 oracle/cache/physics，好的训练曲线可能有循环验证；
   必须用独立 heldout/physical receiver 才能晋级。
4. seed 方差仍大；不能挑一条好曲线归因。

## 问题与假设

问题：在 task、动作、题库、plant、PPO 和随机种子配对不变时，实际低质量权重
`4/0.5/0.5` 是否优于或不劣于名义高质量权重 `393.4/295.1/229.5`？

首要假设不是“低权重必胜”，而是：**task 自洽是当前改善的主要解释；把名义高质量权重真正打开
未必继续改善，可能因尖峰/方差伤害学习。**

如果 matched A/B 中名义高权重在独立 heldout 上稳定显著更好、且没有 unsafe/方差代价，则该假设
被证伪；若两者接近而都显著优于旧不自洽 task，则支持“task 自洽为主”。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练/eval/main commit | 待 task-first 整合 commit 冻结 |
| 动作/action 集 | 同一 training-authorized action set |
| 观测/action 合同 | 同一 `task_first_n<N>` |
| Reward | A=`4/0.5/0.5`；B=`393.4/295.1/229.5`；其余 active terms/callables/params 完全相同 |
| Plant/engine | 同一 Isaac scene/plant hard contract |
| 训练/考试 bank 或 schedule | 同一 predetermined curriculum schedule + 同一独立 heldout |
| Checkpoint/seed | fresh，2 个 paired seeds；不从好 parent 热启 |

两臂必须各自钉住 composed effective Reward receipt SHA。只声明 `reward_pack=v2` 不足以证明 B；
B 需 strict compose 后 receipt 确认三键正是名义值。

两臂还必须消费**同一冻结 racket-task 分布**。ball-first online 与 task-first replay 的 producer
顺序因果属于
[task-first 实验的独立 A/B](EXP-TASK-FIRST-N-ACTION-20260727.md#producer-顺序的独立因果对照)；
不得在本实验里让 A/B 同时换 producer，否则结果只能叫“Reward + producer package”，不能回答权重。

## 实验差异

- 对照 A：现役实际低质量权重 `4/0.5/0.5`。
- 处理 B：名义高质量权重 `393.4/295.1/229.5`。
- 改变的变量：仅三项球拍 quality 权重。
- 其余固定项：task、课程 schedule、动作、seed、PPO、landing/模仿/安全项、plant、预算、
  heldout。
- 决策规则：2 个 paired seeds；每 seed 同卡/同起点配对，宏平均 paired bootstrap。
- 停止/无效规则：effective receipt 不符、curriculum 分布不同、NaN、动作 starvation、
  unsafe 恶化、oracle/heldout 泄漏或任一 run 缺 all-attempt ledger。

纯 task-first executor 首跑不带 analytic landing Reward；本 A/B 是对“历史/ball-conditioned
自洽 task 下三质量权重”的因果复核，不应阻塞 ball-free task-first source/diagnostic smoke。

## 运行态快照（不是正式成绩）

2026-07-27 10:40Z 的只读 live 快照中，两 Pod 六卡仍被占用；按审计行顺序，六条进程的
analytic legal 百分比为
`11.03 / 43.08 / 31.91 / 0 / 36.95 / 0`。这些 run 还存在三类身份问题：

- `c_ep20s`（人话：声称 20 秒 episode 的候选臂）名称写 20 秒，但最终 composed environment
  仍是 10 秒；
- `c_speedwide`（人话：扩大来球速度范围的候选臂）重复写 override，最终值实际为
  `[0.5, 1.2]`；
- 一条 seed-1 候选重复写 seed，最后生效值为 `1`；六条 run 均没有
  [`run_binding.json`](../../DEFINITIONS.md#trainer-run-binding)。

因此这些百分比没有 exact 配方身份、正式分母、milestone 或配对关系，只能说明 seed/arm 方差和
运行账本风险都很大，**不进入成绩表、不选胜者、不证明任何 Reward 因果，也不是 task-first
成绩**。

历史单变量方向信号也很窄：同一波里把 landing 从 `1648.8` 降到 `791.9`，约 3.8k 时 analytic
legal 从 1.8% 到 6.8%；它只有一个 seed，且没有到预注册 8k 决策点，只能作为“高 landing
不一定更好”的方向线索。

### 2026-07-27 exact run 复核

Pod 保存的 composed `env.yaml` 进一步确认当前几条 banked run 真正使用：

```text
racket position / velocity / normal = 4.0 / 0.5 / 0.5
virtual landing = 1648.8
death penalty = -1800
```

名义 `393.4/295.1/229.5` 从未进入这些 run。`c_ep20s_seed0`（人话：声称把 episode 加到 20 秒）
把 override 错写成 `++env.episode_length_s=20.0`，最终 saved config 与
`c_base_bank_seed0` 都是 10 秒且 `env.yaml` SHA 同为
`edb32d4beab8dcacc0361251719406e70dddd8da108f9758bfb791517548ea54`。两份 agent config 去掉
`run_name` 后也相同，因此它是一次意外的同 seed 跨 GPU 复刻，不是 episode A/B。

最后 100 个 exact behavior update 先累加分子/分母：

| run（人话） | legal / strike | 聚合率 |
| --- | ---: | ---: |
| `c_ep20s_seed0`（实际仍 10 秒） | `16283 / 32743` | `49.73%` |
| `c_base_bank_seed0`（同 setting 复刻） | `16245 / 32650` | `49.75%` |
| `c_speedwide_seed0`（最终速度范围 `[0.5,1.2]`） | `10143 / 21655` | `46.84%` |
| `c_base_bank_seed1`（同 setting，仅 seed 不同） | `0 / 28123` | `0%` |
| 旧 unbanked seed0（有其他 source 混杂） | `0 / 25152` | `0%` |

这证明 seed0 的约 50% 可复刻，也同时证明同配方 seed1 到约 17k 仍完全分叉。现象最支持“inverse
solved bank 消除了球/task 冲突”，但高 `virtual_landing` Reward 与报表共用同一 analytic
contact/landing oracle，仍有循环验证风险；当前目录未找到这些 run 的独立 exam/judge 结果。

账本也不够精确：这些 run 都缺 `run_binding.json` 与 `params/effective_reward_recipe.json`；
现有 `training_contract.json` 不完整绑定 Reward/table，甚至 table on/off 候选共享
`16be6a4703dfd30914687218b40ed447db3278a6bfc69e6120f56c59e0989516`。所以这些结果只能作方向证据，
不能追认为 matched A/B 或采用结论。

在既有 solver A/B 之前，新增一个更直接的配对关系 canary：

- `P-paired`：保持同 action/ball/base/aim 经 fixed solver 得到的原始 ball↔task 配对；
- `P-shuffled`：保持完全相同的 ball 边际和 solved-task 边际，只在同速度/难度 stratum 内打乱
  ball↔task 配对。

它只改变物理配对关系，能直接检验“自洽是否是主因”。动态 curriculum 在该因果 A/B 期间关闭。
鉴于当前 seed0≈50%、seed1=0，正式 Reward/solver 结论不能只用两个 seed；先以两个 paired seeds
做 canary，采用至少补到四个 paired seeds或预注册 sequential CI。

## 判读

每动作独立 heldout 512 题：64 center，四个单轴各 96，64 joint edges。成功要求 exact strike、
位置 `<=7.5 cm`、速度误差 `<=0.5 m/s`、拍面 `<=15 deg`、base `<=10 cm`，并在 recovery
前无 table hit/physical fall。报告 Wilson LCB、逐轴、p50/p90 和 all-attempt unsafe。

里程碑只作如下用途：

- iteration 100：数值/receipt/activation health，不判优；
- iteration 500：只执行预注册的 dominated-candidate stop；
- iteration 2000：两 seed paired bootstrap，宏平均成功率改善需 `>5 pp`；每动作非劣界
  `>-5 pp`；table/fall 任一恶化 `>1 pp` 则不得胜出。

这里 `pp` 是 percentage points（百分点），不是相对百分比。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| composed Reward source audit（无科学 `run_name`） | complete | 无 | E1 | effective-recipe tests | 证明实际值与 nominal 不同 |
| 低质量权重 A，paired seeds | preregistered | fresh / 待冻结 | 未跑 | 未生成 | 不得用旧 live 格替代 |
| 名义高质量权重 B，paired seeds | preregistered | fresh / 待冻结 | 未跑 | 未生成 | receipt 必须确认名义值真生效 |

## 决定

- 决定：`inconclusive`
- 理由：已推翻“名义高权重正在生效”的前提；task 自洽是最强解释，但 matched A/B 尚未运行。
- 是否已纳入当前 setting：`no`（只采用 effective receipt 作为以后 launch 的身份门）
- 局限/下一个 gate：先让每个 run 在构造后写
  `params/effective_reward_recipe.json` 并嵌入 hard contract；再冻结 2×2 paired launch。

## 复现与证据

Source tests 与 receipt 入口见
[构建与测试](../../operations/build_and_test.md#task-first-and-capability-source-tests)；
训练前检查见[run_training](../../operations/run_training.md#effective-reward-truth)。

### 2026-07-29：负向 Reward 激活门（源码级完成，Pod 行为证据未生成）

`audit_reward_run.py` 现在只接受 adopted ActionBall 安全配方：

- `policy_dt=0.02 s`，唯一 generic `death_penalty=-3600`，每个 hard-limit、table-hit 或 fall
  terminal transition 必须逐事件闭合为恰好一次 `-72`；
- `table_hit_penalty` / `terminated_by_term` 不得出现在 active effective recipe，因而 table
  不能再叠加第二份 terminal charge；
- `qdes_limit_barrier` 与 actual-q 的 `joint_limit` 必须分别是 v2 callable、`weight=-40`、
  `margin_frac=0.08`、`penalty_floor=0.25`。每条必须同时看到至少一个内区零输出、至少一个
  非终止非零输出；正 intrusion 的 raw sum 不能低于 `0.25 × active_sample_count`，所以不能以
  趋近零的费用蹭软限位；
- hard-limit、table-hit、fall 都必须至少有一条独立单原因 transition。零触发或只与其他原因
  同时发生，只能是 `FAIL_CLOSED`，不能写作 activated。

Host Python 3.8 focused 回归为 `59 passed`。这仍是 E1 source/schema 证据；测试 fixture 和离线
JSON 不能冒充 Isaac 行为。正式 E2 必须来自 clean Pod 真实 rollout 的
`effective_reward_recipe.json`、RewardManager activation、action-bound safety transition、
ActionBall outcome ledger 和 joint-safety sidecar 全闭合；当前没有生成这样的 PASS receipt。

### 2026-07-29：四组 Reward 的 live 单轴因果发射门

新增 [1-env ActionBall Reward 因果审计工序](../../operations/run_action_ball_reward_causal_prelaunch.md)。
它在与训练相同的 post-Hydra compose、motion identity 和 physical-validity source guard 后创建真实
Isaac 环境，再从 live `RewardManager` 重建 effective recipe；term 集合、callable、weight 或 params
与发射 receipt 任一不等即拒绝。每个 active objective 必须用权威输入 tensor 构造 controlled
baseline/worsening，调用生产 Reward callable，并证明 `weight × raw × policy_dt` 严格下降。
unsupported/probe error 会保留逐 term coverage 并令总结果 fail closed，不能用自然 nonzero 代替。

receipt 分开报告 MJLab 平衡稳定、BeyondMimic 模仿、HOPE 击球/上台和 immutable safety 四组的
dense 每控制步与 one-shot 每拍/每终止剂量，同时只列出 A（实际 compose）和 B（仅三项 racket
tracking ×4）候选，绝不自动改训练权重。正式 receipt 还要求 checkout 包括 untracked 在内完全
clean、producer 必须由 HEAD 跟踪且 blob 字节一致，并绑定 motion/manifest/policy contract/config/
effective-recipe SHA。

当前 host focused 为 `15 passed`，Reward 相关联合回归为 `433 passed`；覆盖
no-clobber/content binding、untracked/隐藏 producer
漂移、root authoritative-state→production-getter 读回、advanced-index tensor 写入、live recipe
不等和未知 objective fail-closed。这仍是 E1；尚未从 clean commit 在 Pod 生成真实 Isaac PASS，
因此实验最高证据等级与采用结论不变。

### 2026-07-29：N=1 Reward 首次真实构造 smoke 与生命周期修复

按[标准消融发射工序](../../operations/run_ablation_wave_launch.md)改用 Pod 侧
`/workspace/bin/kit_boot_lock.sh`；旧 N=1 diagnostic launcher 只负责生成 canonical plan/argv，
不再用它的“整卡必须空闲 + 训练全生命周期持锁”发射路径。Pod1/Pod2 各自的 GPU0 在发射前均为
零 compute PID；以下两个运行名（`run_name` 的通用含义见
[术语表](../../DEFINITIONS.md)）都由 exact clean `bbefa277` 构造：

- `bh_loop_c_upper_current_low_smoke_s0_r1`：反手拉、upper、实际低 tracking Reward、seed0；
- `bh_block_upper_current_low_smoke_s0_r1`：反手挡、upper、实际低 tracking Reward、seed0。

两条都完成 Isaac scene、182D observation contract、真实 RewardManager 回读和
`q_des CLAMP ACTIVE`，随后在首个 PPO update 前同因失败：
diagnostic canonical motion 把 `_motion_payloads` 留为 `None`，birth broker 无法证明自己绑定的是
MotionLoader 已采用的同一份 bytes。失败 namespace 保留；只对两条已核对的 exact PID/PGID 发
TERM，没有复用目录或影响其他进程。这是构造生命周期失败，不是 Reward 或动作效果结论。

源码随后闭合三个相邻问题：

1. diagnostic canonical motion 在 MotionLoader 前冻结 bytes，并与初始 SHA 再对账；
2. diagnostic motion binding 生成明确 `training_authorized=false` 的 content-addressed receipt，
   hard contract 不再先读取不存在的 formal evaluator receipt 字段；
3. Motion↔Racket shared-state digest probe 延迟到两端 runtime 已发布之后，避免初始化中的
   hard-contract 重入。

host 可运行联合回归为 `236 passed`；Torch 行为套件需在 Pod exact checkout 复跑。因
`hope_commands.py` 属于 solver source，重新生成的 pins SHA-256 为
`dd609422...f6925`，solver profile 为 `8fbb8bbd...bd0b7`；两动作的新 contact bundle 分别为
`90b88645...bc6a` 与 `fa03627f...4bc2`。下一次只能用 fresh `_r2` namespace，先自然完成
1 env / 2 update，再铺 upper Reward 矩阵；当前仍没有科学结果或采用结论。

`_r2` 已在 Pod2 通过 Torch birth/exact-resume `10 passed`，两动作也都成功写出 hard training
contract 与 effective Reward receipt；随后在首个 true reset 发现 runtime receipt API 漏接新增的
`registry_sha256` 必填参数。birth 与 task 两处调用已统一绑定 broker registry SHA，host 相关回归
`60 passed`。因此 solver pins 再更新为 `26eb1ff2...6804d`、solver profile
`cfba7f28...9349f`；反手拉/挡 bundle 更新为 `c2399571...05d0` /
`c53d1669...41a2`。`_r2` 仍是构造失败而非科学结果，下一次使用 fresh `_r3`。

`_r3` 的 Pod1 反手拉再次通过 scene、runtime bind、182D observation、hard contract、effective
Reward 和 q_des clamp，随后仍在首个 PPO update 前由
`ActionBallTaskReceipt.from_dict(receipt.to_dict())` 拒绝。根因是 receipt canonicalizer 对已经是
binary64 单位四元数的 tuple 再除一次范数；约 `1e-16` 的表示漂移不改变物理姿态，却改变 canonical
JSON SHA。Pod2 因 Pod1 已给出同一构造路径的确定性失败而没有执行 `_r3` trainer；两边都没有
checkpoint 或科学结果。

修复不放宽动作、球题、Reward 或物理门：仅对距单位范数不超过 `2e-15` 的输入保留原 binary64
tuple，非单位输入仍规范化，符号 canonicalization 仍执行。四元数 bitwise 幂等与生产形态
birth→task roundtrip 已进入回归；核心联合测试为 `119 passed, 14 skipped`，独立 20 万随机单位
四元数复核无二次漂移。新 pins 文件 SHA-256 为
`52000401142ca955bef175ce8faafc4a2422363d7e000b20c314c7aa8501f465`，solver profile 为
`714ed22b89208f370978be5c48e6c4b71cc379845e70394fa0ea225f78a49485`；反手拉/挡 bundle 为
`baad5b95012ef0786d9a63c833a29de2782364f5e20c327851a6e07c5bf0acbf` /
`0d3c80f437bb842515fa74e9adf4aea823c90e728ac4e9d87cf4ce1a3d8692ab`。下一步只用 fresh `_r4`
完成两动作各 `1 env / 2 update`；通过后立即发 upper Reward canary，不以 full-body 或最大课程
support 的后续形式化工作阻塞首批 policy。

`_r4` 两动作都穿过上述 receipt seam，但在首个 reset 的第 4 个 per-action birth、首个 PPO
update 前同因停止：固定 `1/3/1` mixture 排到 frontier，而 level-0 的 current physical width 与
center initial width 相同，旧 eligible 条件错误地把“尚未 promotion”当作“没有合法 support”。
两条均为 `0 iteration / 0 checkpoint`；Pod1 的 Isaac traceback 后 cleanup 挂住，核对 exact
PID/PGID/start ticks 后只处理该 run，最终无残余。

修复保留 mixture、stratum、frontier arm、固定覆盖、proposal 计数和 exact replay：优先选择已经
扩张到 center/interior 之外的 arm；若尚无 promoted arm，则在当前非零物理 support 的 outer band
采 frontier；只有该 scope 所有物理宽度真为零才在 draw 前原子拒绝。birth 与 swing 共用同一规则，
不改 manifest、action identity、schema、Reward 或 PPO。sampling/runtime/receipt/launcher 联合回归
`171 passed, 14 skipped`；profile pins 与两动作 bundle 内容 SHA 不变。下一次使用 fresh `_r5`。

### 2026-07-29：把“先产 policy”与 formal 证明层拆开

fresh `_r5` 的两个 Pod 都完成了真实 rollout，却在 optimizer 前被 tensor fingerprint 读取
`torch.inference_mode()` tensor 的 `_version` 拒绝。该属性在 inference tensor 上本来就不可读；
修复改为对可读 version 与不可读 inference tensor 分型取证，不改变 tensor、Reward、PPO 或物理。

fresh `_r6` 的 Pod1 继续完成 rollout，随后由 formal Reward activation 的 terminal-edge
conservation 审计停止；fresh `_r7` 已完成一次真实 optimizer step，又在 rollout-end curriculum
证明回调读取旧 `ExpectedDomain.levels` 字段时停止，尚未来得及打印下一 iteration 或保存
checkpoint。Pod2 同代的一次失败是远端动作资产恢复失败，不是 Reward/动作科学结果。

这些失败说明原路径把两种目的错误串成一条：

- diagnostic/canary 的目的，是尽快证明同一动作、球题、Reward、PPO 能产生可学习 policy；
- formal evidence/curriculum 的目的，是为 Gate 晋级提供严格 receipt 与 held-out 证明。

因此 exact `e469d85b5c9f493e5c1fbb6861eefe84b0926a32` 明确把 N=1
`training_authorized=false` diagnostic 运行冻结在 level-0 domain，并跳过 formal Reward/joint
evidence fence 与 rollout-end curriculum advancement。**实际环境 Reward 没有关闭**；
`q_des` clamp、soft/hard-limit penalty、hard-limit/table/fall termination 也没有关闭。这样的
checkpoint 只能参加 Reward screen，不能冒充正式 curriculum promotion 或 Gate 证据。

该 exact commit 的 fresh `_r8` smoke 已在两台 Pod 各自然完成 `1 env × 2 update`，零
Traceback，且 `model_0.pt/model_1.pt` 全部 tensor finite：

- Pod1 `bh_loop_c_upper_current_low_smoke_s0_r8_e469d85b`；
- Pod2 `bh_block_upper_current_low_smoke_s0_r8_e469d85b`。

Pod1 两次 mean episode length 为 `1.04/1.02`；第二个 rollout 的 24/24 policy steps 以
`joint_qdes_forbidden` 结束，table/fall 为零。这不是发射基础设施失败，但说明随机初始 policy
正在请求物理 hard-limit 内缩边界之外的目标，必须在首批 20–50 updates 观察它是否学会避开；若
长期不恢复，应先查 default pose、动作 affine mapping 与逐关节请求，而不是降低 hard-limit
penalty。

这批六条 upper Reward canary 是后续 `4ff48b21` 波的历史前身；当前可复算身份与运行状态统一看
[`n1_live_wave_4ff48b21.v1.json`](../../../configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json)。
同配方续跑按[未变配方诊断快线](../../operations/run_ablation_wave_launch.md#未变配方的诊断续跑快线)
执行，不再重复 repin、物化与无关 host 大回归。

### 2026-07-29：`4ff48b21` 4096-env 长跑与每 step 实测

这一代不再用启动瞬间的 NVML 利用率判断训练是否工作，而是绑定 RSL-RL 完整 iteration block。
每个 update 是 `4096 × 24 = 98,304` 个环境控制步；一个
[`vector policy step`](../../DEFINITIONS.md#vector-policy-step) 是 4096 个环境一起前进一步，
墙钟为 `iteration_time / 24`。逐 run 的 exact namespace、PID、合同和快照真源是
[`n1_live_wave_4ff48b21.v1.json`](../../../configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json)。

11:20 UTC 的三条活跃 upper 诊断如下；`current_low` 是现行低任务权重，
`mimic_x2` 只把动作模仿尺度乘二：

| action / Reward | update | update 墙钟 | vector step 墙钟 | 环境步吞吐 | terminal / qdes / actual / ee | strike |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bh_loop_c / current_low` | 35 | 61.69 s | 2.57 s | 1,594 step/s | 3,058 / 909 / 539 / 2,246 | 0 |
| `bh_block / current_low` | 7 | 271.99 s | 11.33 s | 361 step/s | 31,505 / 30,730 / 75 / 778 | 0 |
| `bh_loop_c / mimic_x2` | 10 | 149.60 s | 6.23 s | 657 step/s | 15,393 / 13,850 / 311 / 1,577 | 0 |

reason mask 会重叠，不能把后三项相加当 terminal。PPO learning 每轮只有约 `0.07–0.20 s`；
主要墙钟都在 collection 和提前 reset，所以“GPU 瞬时利用率低”不是 optimizer 没跑。小时巡检应
连续记录完整 iteration 的 `collection/learning/update/vector-step/environment-steps/s`，并与
qdes/reference reset 同报。

#### Reward 因果结论

同动作 `bh_loop_c` 的前五轮里：

- `current_low`：总墙钟 `863.83 s`，terminal `150,846`，qdes reason `148,267`；
- `mimic_x2`：总墙钟 `844.45 s`，terminal `151,201`，qdes reason `148,651`。

两边 strike 都是零，行为几乎没有分离；但 `mimic_x2` 的 raw motion
anchor-orientation、linear-velocity、angular-velocity 项从约
`0.0098/0.0137/0.0060` 变成 `0.0195/0.0275/0.0120`，约精确两倍。因此动作模仿 Reward
**实际生效**，只是其早期 `O(1e-2)` 信号相对终止/死亡尺度太小，不能挽救随机 policy 的
hard-limit reset 风暴。下一轮首先修初始化，而不是继续把 mimic 权重盲目放大。

#### 下一 fresh wave

根因是默认 action offset 把左右 shoulder-roll 放在 hard-inner 边界附近，`0.15` 初始策略噪声
使大量随机 q_des 越界。候选修复保持 deploy decoder
`q_des = default + scale × action` 不变，只对 fresh N=1/N=5 显式启用：

1. actor 最后一层 weight 清零；
2. bias 设为 `(shared_ready - default) / scale`；
3. 初始 policy std 固定 `0.02`；
4. 把真实启动 calibration 扰动和四倍标准差一起证明仍在现役 2% hard-inner 边界内；
5. resume 永不覆盖；N=73/N=93 不启用常量 bias，继续原路径，直到有 action-conditioned
   bootstrap。

当前 `bh_loop_c` 已从 update 2 的 48,681 个 qdes reason 降到 update 35 的 909 个，说明旧 run
确实在学“先活下来”，所以不打断它；新初始化的价值是省掉这段昂贵、无击球机会的自救期。

full-body 反手拉的第二次启动已完成 Isaac scene 和 Reward receipt，但首 reset 某 birth 的 solver
零接纳。离线精确复算显示普通 solver 的 512 个 proposal 有 506 个可解，经过 face-center 到
official-site teacher-rate 映射后仅 93 个接纳：中心样本约 `0.5976 < 0.6`。这不是 full-body
launcher 或 PPO 失败，而是物化器把 `face_speed_min = nominal × teacher_rate_min` 错当成严格等价
的 site-rate 下界；必须重物化并加 exact post-solver admission 预飞，不能降低 runtime 的
`teacher_rate_min=0.6`。

### 2026-07-29：reset 历史对照与有限 q_des 候选切换

本节是 `curr-launch-fix` 功能分支候选，不改变 `origin/main` 上 `docs/NOW.md` 的运行态权威；
下述“预计”也不是修复后实测结果。

#### 健康基线不能取自 reset 风暴

旧 v2 并非从来没有 reset storm：早期 update 1–9 的 `ee_body_pos` 为
`1,690–2,301 reset/update`、mean episode length `44–50`，collection 为
`4.61–10.34 s`（均值 `6.78 s`）；同一 run 到 update 4181–4187 已学到 mean episode length
`770–795`、`ee_body_pos=1–7/update`，collection 稳在 `4.58–4.71 s`。因此它支持“策略可以学掉
一部分 reference 偏差”，不支持继续用 reference reset 消耗采集时间。

另一个被反复引用的 `6.4 s collection/update` 来自失败 probe：mean episode length 恒为 `1`，
每轮 `4096×24=98,304` 个环境步全部 reset。相同代码只修正 stand start hold 后为
`4.41–4.65 s`，代表值约 `4.49 s`。所以 `6.4 s` 不是健康 ActionBall 的性能目标，修复后的首要
比较尺应是 `4.49 s + ActionBall birth/solver/receipt 的真实增量`。

当前旧语义 ActionBall 稳态快照约为：

| action | collection/update | 主 reset reason | 判读 |
| --- | ---: | --- | --- |
| `bh_loop_c` | 约 `27 s` | `ee_body_pos` | 有限 q_des 已基本学掉，reference reset 仍主导 |
| `bh_block` | 约 `48 s` | `joint_qdes_forbidden` | 有限目标请求 reset 仍主导 |

按同一 update 的 reason/time 对账，移除 **finite q_des request** reset 对 block 预计节省
`14–17 s/update`；这是发射预算估计，必须由 fresh 4096-env run 的
[`collection_vector_step_wall_s`](../../DEFINITIONS.md#collection-vector-step-wall-s) 实测复核。

#### “老师贴限”反例被排除

对 exact upper/full 反手拉与反手挡老师做逐帧、逐关节限位余量审计：

| clip | 全片 normalized hard margin min | 全片 normalized soft margin min | 击球窗 hard margin min |
| --- | ---: | ---: | ---: |
| loop upper | `0.111954` | `0.068838` | `0.149400` |
| block upper | `0.115081` | `0.072312` | `0.136338` |
| loop full | `0.113493` | `0.070548` | `0.152600` |
| block full | `0.115081` | `0.072312` | `0.136573` |

四件动作的 hard、soft 与 2% hard-inner crossing 均为 `0`；block 全片余量还略大于 loop。
因此 block 的 q_des reason 比 loop 高约 180 倍，不能归因于“模仿老师必然越限”。下一层根因应查
policy 输出分布、block 对应观测/目标与 episode reset 后状态，而不是删老师关节或降低安全代价。

#### 候选语义与 Reward

有限 q_des 的新候选采用
[`finite q_des execution projection`](../../DEFINITIONS.md#finite-qdes-execution-projection)：
包络内严格恒等，包络外执行最近合法目标；raw policy sample 和 PPO log-prob 不改，也不因有限请求
reset。为了避免 clipped-action aliasing，新增
[`qdes_projection_penalty`](../../DEFINITIONS.md#qdes-projection-penalty)，只吃投影前归一化超出量。
首发 weight=`-5`，`-20` 只作预注册消融，不能在观察收入账前直接成为主导负项。

安全边界没有一起放宽：

- raw q_des 非有限、实际/physics-substep 关节 hard edge、table hit 和 fall 仍 hard reset；
- 仅由当前 q/qdot 预测出来的 ballistic crossing 继续选择有限 brake target，但不再在实际越界前
  reset；最终/子步真实越界仍由独立 actual term 的 sticky 证据终止；
- reference anchor/body/ee 谓词改为
  [`reference_guard_mode=metrics_only`](../../DEFINITIONS.md#reference-metrics-only)，继续计数但不
  reset、不额外给 Reward；
- 每关节必须报告投影触发率、投影前平均/最大超出量、正负侧与执行值恰贴投影边界的饱和占比；
  只有冻结策略、零该罚分复测仍低，才可说 policy 学会限位，而不能只说执行器兜住了。

小时巡检固定输出四个互不混淆的 step timing：

1. [`collection_vector_step_wall_s`](../../DEFINITIONS.md#collection-vector-step-wall-s)；
2. [`amortized_e2e_vector_step_wall_s`](../../DEFINITIONS.md#amortized-e2e-vector-step-wall-s)；
3. [`collection_environment_step_us`](../../DEFINITIONS.md#collection-environment-step-us)；
4. [`collection_environment_steps_per_s`](../../DEFINITIONS.md#collection-environment-steps-per-s)。

CaT（连续违规量调制 Reward/termination、仿真不中断）和 PPO policy-mean bound loss 都是合理后续；
它们会改变训练目标或 runner，今晚先不叠加。先用 execution projection + raw excess Reward 验证
block reset/吞吐与任务学习，再据饱和遥测决定是否购买下一层机制。

### 2026-07-29：`b1d299e1` 发射工件

候选实现已固定为 exact
`b1d299e1e57bd0909aa402ca2701b3901975337b` 并推送到 `curr-launch-fix`。除上一节语义外，
immutable schema-3 hard contract 现在只允许 ActionBall runtime fact
`finite_preclamp_qdes_projection_enabled=true`；配置/runtime 漂移、缺字段、非布尔值或 resume
切回旧语义都会 fail closed。nonfinite raw q_des 仍终止，但其名义投影会先锚到有限 brake target，
因此 RewardManager 在 reset 应用前读取投影距离也不会传播 NaN。

host 整合回归为 `589 passed, 1 failed`；唯一失败是新增测试漏 `import math`，修正后包含该测试与
完整 schema-3 合同的 `107 passed`，另有 `py_compile` 与 `git diff --check` PASS。物化使用：

- profile pins：
  [`action_ball_profile_pins.v1.b1d299e1.json`](../../../configs/n1_contact_20260729/action_ball_profile_pins.v1.b1d299e1.json)，
  文件 SHA-256 `47a00a6a35ea4709603634deeb062febc3a6e7bb2b9f57aab5c573781d330488`；
- upper loop/block bundle：
  [`29adc3cf...c85c4`](../../../configs/n1_contact_20260729/bh_loop_c.bundle.v1.29adc3cf69f9.json) /
  [`fb1ed6ee...b6c5a`](../../../configs/n1_contact_20260729/bh_block.bundle.v1.fb1ed6ee4371.json)；
- full loop/block bundle：
  [`d94c7f0a...223a2`](../../../configs/n1_contact_20260729/bh_loop_c.full.bundle.v1.d94c7f0a79f0.json) /
  [`ca13d958...2f0f`](../../../configs/n1_contact_20260729/bh_block.full.bundle.v1.ca13d9583ef5.json)。

full post-solver 固定 512 proposal 预飞分别为 loop `511/512=99.80%`（1 个
`resid_gt_tol`）与 block `443/512=86.52%`（69 个 `resid_gt_tol`）。两者 diagnostic status
均 PASS；block 的 formal rate threshold 为 FAIL，所以本轮最多作为固定 level-0 diagnostic
候选，不能冒充 formal curriculum/admission 证据。下一证据必须来自 clean Pod exact checkout 的
真实 effective Reward/PPO receipt、`1 env × 2 update` finite checkpoint 和 4096-env
collection/reset 账。

第一次 Pod 真实 Hydra compose 在 Kit 前发现 N1 launcher 将未在 task YAML 声明的
`reference_guard_mode` 写成普通 override；正确 canonical token 是
`+task.racket.reference_guard_mode=metrics_only`。follow-up 同时把
`init_noise_std=0.02`、shared-ready bootstrap 和 metrics-only 接入 formal N5 launcher，并把这三项
列入 launcher-owned keys；full N1 launcher 新增 prototype 内 solver preflight PASS 门，旧的无
provenance bundle 不再可能进入 birth。两套 launcher 联合回归为 `80 passed`。

### 2026-07-29：reference ET 研究裁定与 `5e94f21b` 反例

Reference tracking error 不能类比 q_des 做 clamp：它是实际状态与老师的结果误差，没有一个可把
机器人无损投影回去的动作可行集。clamp error 会隐藏偏离，clamp state 等于瞬移，clamp reference
等于改老师。原始证据边界如下：

- [BeyondMimic](https://arxiv.org/html/2508.08241) 对 anchor/end-effector 高度和 anchor
  orientation 使用 hard early termination，并把 terminated phase 喂给困难段采样；
- [DeepMimic Table 5](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf)
  中 RSI+ET 对比 RSI-only，backflip/sideflip/spinkick 为
  `0.791/0.823/0.848` 对 `0.379/0.355/0.358`；walk 基本不变；
- [PHC](https://arxiv.org/pdf/2305.06456) 只选择性放宽 ankle/toe 的 ET，并另训 recovery
  primitive；[Stubborn](https://arxiv.org/html/2606.12814) 用概率终止和 tracking-error-driven
  sampling。两者都不是 blanket metrics-only 或 reference clamp。

本仓库 ActionBall 又有一个关键差异：birth broker 已绕过 BeyondMimic 单 clip failed-bin sampler；
当前 `metrics_only` raw counter 只记日志，不推进 live curriculum。因此论文不能直接证明
ActionBall hard ET 必胜，也不能证明全关无害。最终需 fixed seed 比较：

1. `phase_gated` hard reference ET；
2. `metrics_only`，只作诊断；
3. 击球窗放宽、窗外持久/概率终止的 hybrid。

三臂同时看 strike/return、teacher error、reference breach dwell、fall/table/actual-limit、episode
长度和吞吐；actual-joint storm 清除前不做该因果判断。

Pod1 的 receding-horizon 候选 `5e94f21b` 已完成 updates 1–16：

| candidate | s/update | mean episode | actual-joint reset/update | reference-only/update | strike total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5e94f21b` | `36.48` | `20.19` | `4,791.6` | `43.3` | `2` |
| `5dbb` 同窗 | `38.35` | `20.68` | `4,728.8` | `41.9` | `1` |

最近十窗 `5e94f21b` 反而约慢 `4%`，actual reset 没降，故 20 ms receding horizon 不晋级。
reference-only occupancy 仅 `0.044%` transitions/update，也不能解释 mass reset。

下一候选只将 ActionBall finite executed q_des 在 soft limits 内上下侧各额外保留 `5%`。四条
loop/block × upper/full 老师轨迹均无 crossing，最小剩余余量约 `0.046 rad`。新增
[`finite_projection_soft_envelope_inset_fraction`](../../DEFINITIONS.md#finite-projection-soft-inset)
同时进入 config、runtime property 与 schema-3 training contract；raw proposal、PPO log-prob、
actual hard band、Reward 和非 ActionBall 路径不变。该结论目前只到 source-level，Pod smoke 和
4096 同 seed 才是行为验收。

### 2026-07-29：`478f485b` 反例、逐关节定位与吞吐拆解

Pod1 exact `478f485b119f48807892870621a5350842ecd733` 的额外 `5%` finite-q_des reserve
没有解掉实际关节 reset：

| evidence | result |
| --- | --- |
| 1-env × 2-update smoke | 自然完成；mean episode `19/18`；fresh bootstrap=`APPLIED_FRESH` |
| 4096-env updates 0--6 | `27.11--41.66 s/update`；mean episode `18.67--23.74` |
| actual-joint reset | update 1--6 为 `4,457/5,087/4,664/4,815/4,813/4,791` |
| q_des / projection | q_des termination=`0`；projection penalty/contribution=`0` |
| task progress | strike opportunity=`0`；table=`0`；fall 仅零星 |
| stop artifact | 完整 PPO boundary 的 finite `model_6.pt`；随后停止 |

这否定了“只要让有限 q_des target 再远离包络就会恢复训练”的单变量假设。用
`configs/a3_runtime_articulation_joint_order.txt` 解码 teacher 后，31 个关节全片都不进入
实际 hard-limit 内缩 `2%`，`q+0.02*qdot` 也为零 crossing；frame-0 最近的是
`right_ankle_roll_joint`，但仍有约 `0.137 rad` 到 inner edge，不能仅凭静态余量点名根因。
physical reset 会写老师 frame-0 + zero qdot，policy 日志又证明 shared-ready fresh bootstrap
已应用，所以剩余高概率分支是 ground/contact/implicit-PD transient 或某个实际关节在 rollout
中漂入 inner band。

当前 `joint_actual_forbidden` 把当前 q 的 2%-inner 下侧/上侧、非有限状态和 substep raw-hard
latch 合成一个 env bit；diagnostic run 又故意不走 formal joint-safety receipt，所以日志没有
逐关节证据。下一 source 候选只加以下非晋级遥测：

- 每个 terminal env 记录 exact articulation order 下的 lower/upper/nonfinite/current-inner、
  substep actual-hard、pre-apply nonfinite q_des 与 predicted crossing overlap；
- 以 episode age `<=1` 和 `>1` 分桶，保留 terminal 分母与 mean/max age；
- 所有计数留在 GPU，只有 PPO update 边界一次小批量同步和 canonical JSON；不改 action、
  Reward、Done、physics、solver 或 curriculum。

墙钟数据也说明 reset 不是“不可避免的 Isaac 固有速度”：同一 run 中 reset 增加约
`629--1,270/update` 时，update 额外增加约 `3.9--10.6 s`；按该局部关系外推，无 mass reset
时约为 `4--6 s/update`，与仓库旧健康跑一致。这只是定位依据，不作为未来吞吐承诺。性能修复顺序
冻结为：

公开实现只支持方向判断，不能直接拿现代 GPU 数字要求当前 V100/contact-heavy 场景：
[Isaac Lab 官方 benchmark](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
明确把 env-only、inference 与 training FPS 分开；[IsaacGymEnvs Humanoid](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/humanoid.py)
和 [legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
的 reset 核心都是按 `env_ids` 的 tensor 索引批量写回，不含逐 env JSON/SHA 收据。
[PyTorch 官方 tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)
又明确把 `.item()`、`.cpu()` 与依赖 CUDA tensor 的 Python control flow 列为会同步 CPU/GPU
的模式。因此外部证据支持“批量化并延后同步”，但本项目最终目标仍以同 Pod 旧跑和 fresh A/B
为准。

1. 先消除错误语义造成的 mass reset，不用更多 env 掩盖 CPU 瓶颈；
2. 将 command EMA/计数留在 device、每 update 同步一次，并去掉逐步
   `float(reduce)` / `bool(any)`；
3. 每 policy step 的 `_compute_strike_timing()` 只计算一次；
4. immutable receipt SHA 缓存，reset 批次一次
   [device-to-host transfer（D2H，设备到主机传输）](../../DEFINITIONS.md#device-to-host-transfer)，
   禁止 per-env `.item()` / JSON / SHA；
5. formal joint ledger 改为预分配聚合，只有 terminal/unsafe env 保留完整 transcript；
6. 最后才做 2048/4096/8192 env A/B。

首阶段验收是相同 physics/Reward/seed 下达到
[`collection_environment_steps_per_s`](../../DEFINITIONS.md#collection-environment-steps-per-s)
`>=15k`、collection GPU utilization `>60%`，同时 reset reason、task/solver counters 与
checkpoint exact-resume 不变。

#### `8d2a1bcd` Pod 逐关节结果与软/硬限位裁定

diagnostic source `8d2a1bcd5194e1a70490b7a73bb5228e3fe610d9` 没有改 action、Reward、Done 或
physics，只在 device 端按 joint×side×episode-age 累计旧 actual reason，并在 PPO update
边界一次 D2H。Pod1 结果：

| update | old actual reason / event denominator | left ankle pitch lower | waist pitch upper | right ankle roll upper | waist roll current side(s) | mean event age |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `3,187` | `2,304` | `628` | `276` | `13` | `17.91` |
| 1 | `4,457` | `2,416` | `843` | `930` | `345` | `20.84` |
| 2 | `5,087` | `3,112` | `875` | `821` | `340` | `20.27` |

关节计数可重叠，不能相加冒充分母。三轮 age<=1 都为零，nonfinite q_des 为零。substep raw-hard
overlap 远小于 current-inner 计数；例如 update 0 的四个主要关节 raw-hard overlap 仅
`8/139/1/32`。1-env 两轮也都由左脚踝 pitch 下侧在 age `19/17` 触发，分别有/无 predicted
crossing overlap。该 run 在 update 2 完整边界保存 finite checkpoint 后停止。

因此采用以下第一性原理拆分：

- **软约束**：实际 q 进入 hard limit 内侧 `2%` 是危险但可恢复的状态。现役 actual-q barrier
  已从 soft envelope 内侧 `8%` 开始、weight=`-40`、floor=`0.25` 收费；2%-inner 继续记
  joint/side/dwell，但不 reset，让 policy 得到恢复 transition。
- **硬约束**：nonfinite/invalid state、当前 raw mechanical edge、任一 physics substep raw
  hard edge 仍令 `joint_actual_forbidden` Done；table/fall 完全不动。
- **新对账**：diagnostic schema v2 同时输出 `total_safety_event_count` 和
  `total_hard_terminal_count`；只有后者应与 termination reason 相等。

这不是 CaT，也不是放宽机械边界；只是把已有强连续 penalty 和不可恢复 Done 放回各自正确层级。
下一 fresh Pod A/B 的 primary 是 episode 是否越过 `t_hit≈31`、strike 是否出现和 steps/s；
hard/table/fall/nonfinite 不得上升。source tests 不算行为结果。

### 2026-07-29：`7a14b0b9` 真实 smoke、4096 反例与 crossing 所有权修正

Pod1 clean checkout 已按
[本地资产恢复合同](../../operations/setup_local_sync.md) no-clobber 恢复相同 A3 生成树；
`model.urdf` SHA-256 为
`79655f05d204c24f028778425aa971410773d1f8bbbd214de6fdb8f8ae75d1cc`，checkout 的 tracked
状态仍 clean。upper loop/block 真实 Isaac policy contract SHA 分别为
`8e07609d...0f4d`、`a35f0c72...9a07`，共同的 composed Reward SHA 为
`c2f13419...6c11`。

反手拉 upper 的 exact
[`smoke spec`](../../../configs/n1_contact_20260729/smoke_loop_upper_gpu1_7a14b0b9.json)
自然完成 `1 env × 2 update`；两轮 iteration 为 `2.85/2.02 s`，两个 checkpoint 各
`7,122,931 B`，共 `80` 个 tensor / `1,775,488` 个元素且 nonfinite=`0`。同 source 的
[`4096-env spec`](../../../configs/n1_contact_20260729/long_loop_upper_gpu1_7a14b0b9_r1.json)
随后进入真实 PPO；最初两轮为 `28.36/39.82 s`，而不是预期的健康基线。到 update 6：

- projection sample/nonfinite/penalty 均为 `0`，证明 finite raw q_des reset wall 已移除；
- live `joint_qdes_forbidden=0.05249/env-step`，但该旧名包含 q/qdot 预测 crossing 与 actual/substep
  hard latch；
- live `joint_actual_forbidden=0.02490/env-step`，说明未训练 policy/implicit plant 的实际动量
  仍会进 hard band；老师轨迹离线重算本身为零 crossing。

因此不能把 `7a14b0b9` 长跑当健康训练。最小 successor 保留 action term 的 finite brake 与
actual sticky hard-edge 终止，只把 projection 模式的 q_des DoneTerm 收窄为 nonfinite raw request；
predicted-only crossing 不再重复 reset。legacy 模式不变。Pod host joint-safety suite 为
`80 passed`；必须由 fresh namespace 的 4096 replacement 验证吞吐和 actual-hard 比率下降后，
才能继续比较 Reward。

### 2026-07-29：`5dbb4e58` replacement 暴露实际关节动态余量不足

反手拉 upper 的
[`1-env smoke`](../../../configs/n1_contact_20260729/smoke_loop_upper_gpu1_5dbb4e58.json)
自然完成两轮，iteration 为 `4.72/3.00 s`，q_des termination 为零。随后 exact
[`4096-env replacement`](../../../configs/n1_contact_20260729/long_loop_upper_gpu1_5dbb4e58_r1.json)
证明移除 predicted-only reset 并未单独恢复训练吞吐：

- update 3--8 与旧 `7a14b0b9` 同序号相比，iteration mean 由 `37.73 s` 变为
  `36.78 s`，只改善约 `2.5%`；terminal reset 只少约 `7.9%`；
- update 0--17 的 mean episode length 始终约 `19.30--23.81` steps，低于反手拉约
  `31` steps 的击球窗；update 12 仅有一次 strike opportunity，其余为零；
- finite q_des projection、nonfinite 与 projection penalty 始终为零；每轮约
  `4,658--5,092` 个 `joint_actual_forbidden`，而 fall 仅 `0--23`，所以慢速主因不是倒地；
- 该 actual term 不等于“已经撞机械硬挡”：它在当前真实 q 进入 hard limit 内缩 `2%`
  的安全带时终止，并 OR physics-substep 真实 hard-edge sticky latch。q_des clamp 约束的是
  drive target，不能据此推出带惯性、重力和足底接触的真实 q 不会动态超调。

因此保留原 run 到早期学习观察窗，不能因前几个慢 update 单独判死；晋级判据仍是 episode length
越过 `t_hit` 且 strike opportunity 持续非零。并行的单变量候选只把每个 fresh `5 ms`
physics-substep 的 q/qdot 预测与 brake 保持为滚动 `20 ms` policy/control horizon；Reward、
Done、2% safety band 和 nominal safe target 均不变，安全行要求 bitwise 不变。host
joint-safety focused suite `81 passed`，与 ActionBall runtime wiring 的联合回归为
`125 passed`。该候选尚无 Pod A/B，不得写成已提速或已解决。

### 2026-07-29：`eaf55fba` 排除 soft-band Done；upper 根因收敛为 q/qd 自相矛盾

`eaf55fba5e201d76153162ab2f7f482bb66b3f22` 已将 recoverable 2%-inner occupancy 留给
actual-q barrier/遥测，只有 nonfinite、当前 raw mechanical edge 或 physics-substep raw-hard
latch 继续 Done。Pod1 fresh 4096-env upper `bh_loop_c` updates 0--4 的 hard terminal 仍为
`2,549/3,986/4,225/4,188/4,162`，episode age 约 `22--24<t_hit≈31`，strike opportunity
恒为零；q_des projection、projection penalty 和 nonfinite 恒为零。这是对上一节“先等 early
policy 学会避开”的反例：事件在有效 PPO 学习前已经大规模发生，继续调 Reward、q_des margin
或 PPO 不能产生击球数据。

老师全片、20-ms ballistic prediction 和 fresh exact-ready actor bias 均不贴 hard edge。第一轮
只检查 ready donor 的旧证据，曾误判为现役 upper qpos 未接地；随后在 Pod 对**实际训练 NPZ**
逐列和 exact A3 MuJoCo 复核，得到更强的相反证据：

- `canonical_ready_v1` 本身确实只是 donor，旧 exact MuJoCo 证据为
  `active_contact_count=0`；但这不能外推到后续 fivebind upper 的实际下肢；
- 两条现役 upper 的 12 个腿关节 `joint_pos` 已全片恒定，并逐位等于 content-bound A3
  grounded-ready candidate；其 SHA-256 为
  `585bbd7d643857abd08108eac7b4dd997b228d0df1a9921334ca845cd931d71e`，receipt file SHA-256
  为 `ee7dea1aec81169e1d002bbe0b2cfa75c793a97a3f89e1e740d0064dc8be7c46`；
- 这里的 `candidate_id=G1` 是 A3 grounded-ready 构造候选的代号，exact model 为
  `A3T2.5_pingpong_0519` 和 A3 31-joint order，绝不是 Unitree G1 机器人；
- 两条 actual upper 在 exact A3 MuJoCo frame 0 都为双脚 `3+3` 接触，sole 约
  `-0.000498 m`，joint/collision/support/static-ground LP PASS；
- 真正的 schema 不一致是腿 `joint_pos` 每帧恒定，但相同 12 列 `joint_vel` 在中间帧非零。
  Pod qvel-only 原型把腿速度归零、用 `canonical_schema2_builder` 重建 body FK/velocity 后，
  两条动作每帧 `right_racket` site position/orientation/linear/angular velocity 的最大差均为
  exact zero，frame count/strike frame 不变。

因此首个 successor 改为 A3 upper q/qd 一致性资产修复，保留
raw-hard/table/fall/nonfinite Done。这是数学/运动学合同修复，不做 Reward 学习 A/B：

1. upper 快线不得再改 qpos/root/retime，只把 12 个恒定腿位置对应的 stale `joint_vel` 归零，
   并重算 body FK/velocity；
2. 每帧 right-racket site position/orientation/linear/angular velocity、frame count、strike frame
   和既有球题必须保持不变，才能复用原 N1 contact binding；
3. full 不能用该 qvel-only 变换，必须完整重编 `grounded ready→core→grounded ready`，随后重绑
   aim/strike/contact；
4. fresh `1 env × 2 update` 后跑 4096-env 五轮；primary 是 episode 越过 `t_hit`、strike
   opportunity 非零和 environment-steps/s，raw-hard/table/fall/nonfinite 不得上升。

与此同时只合入可证明等价的 hot-path 优化：immutable receipt SHA 外部缓存、正常同一步
strike-timing 一次计算、global+per-action 原 error reduction 与可精确表示的 float32 count
的一次 batched
[device-to-host transfer（D2H，设备到主机传输）](../../DEFINITIONS.md#device-to-host-transfer)、以及空集保持旧
metric 的 `fired_valid` device mask。它们只需 Pod parity/exact-resume/profiler，不开启学习
A/B。Reward 权重、reference termination/CaT、death/entropy/sigma/RSI、8192 env 与 actual
hard-edge 放宽继续作为健康 baseline 之后的单变量 canary。

### 2026-07-30：qvel-fixed upper 已完成最短 smoke

Pod1 串行完成反手拉 `n1_qvelfix_smoke_5ecf0e06_loop_gpu1_r3` 和反手挡
`n1_qvelfix_smoke_5ecf0e06_block_gpu1_r4` 的 `1 env × 2 update`。实际 iteration 时间分别为
`4.65/3.18 s` 与 `4.67/2.92 s`，四个 `model_0/1.pt` 均已在 Pod 载入并逐 tensor 验证 finite。
两条均无 q_des、table 或 fall termination；N=1 actual raw-hard 主要来自踝关节，loop 第二轮
`2` 次、block 每轮 `1` 次。由于分母极小，禁止据此调负 Reward 或终止语义。

下一步不是 Reward A/B，而是同 setting 的 exact `4096 env × 5 update × save1` probe。其
primary evidence 为 episode 是否越过各动作 `t_hit`、strike opportunity 是否非零、
environment-steps/s，以及 actual hard/table/fall/nonfinite 的独立分账。只有该 probe 健康后，
才启动本记录中 tracking/mimic/negative Reward 的单变量 screen。

### 2026-07-30：qvel-fixed 4096 双动作反证与 stable-upper 决策

Pod1 的 exact `4096 env × 5 update × save1` 已完成：

| 动作 | iteration time | mean episode | actual raw-hard | strike |
| --- | --- | --- | --- | --- |
| `bh_loop_c` | `30.51/40.39/40.78/35.47/35.82 s` | 约 `23` steps | 约 `2.5k--4.2k/update` | `0` |
| `bh_block` | `38.35/50.39/43.55/43.14/54.73 s` | 约 `12` steps | 末轮约 `7.7k` | `0` |

两条都有逐轮 finite checkpoint；q_des projection/nonfinite 为零、table 为零、fall 很低。
因此失败不是 Reward 还没学够，也不是 qdes clamp 失效。当前约 `2--3k
environment-steps/s` 来自 episode 在击球帧前被 actual raw-hard reset；继续跑 long 不产生
有意义 strike 样本。

新的 A3 事实链是：两条 upper frame 0 共享深蹲腿位、root `z=0.920683 m` 与 pitch
`-11.19°`；actor bootstrap 已正确给出该 ready，age<=1 hard 为零。故根因不是坐标跳变，而是
“几何双脚接触的 canonical pose”被错误当成“现役 implicit-PD 闭环 stable ready”。下一唯一
replacement 将 lower/root 改到 `AGIBOT_A3_CFG.init_state` 的 runtime stand，保留腰以上 q/qd
并重建 exact A3 schema-2 FK；racket world pose 变化后重新物化 contact/task bundle。

这是正确性 replacement，不做学习 A/B。验收顺序为 Pod source regression → deterministic
closed-loop hold → `1 env×2` → `4096×5`；通过后立即发 finite long。Reward、reference、
curriculum 10%、full-body 和 8192 env 的科学 canary 继续等 healthy strike baseline。

producer focused regression 已在 Pod `12 passed`。stable-upper 反手拉/挡 exact motion
SHA 为 `4343a85e…` / `08aeafaf…`，两者 exact A3 static audit 均为双脚 `3+3` 接触、
LP `feasible=true`；击球帧拍速保持到数值精度。root 修复让 selected face world position
最多变化 `0.138/0.064 m`，因此旧题明确作废；N1 producer 保留原 profile 各轴宽度，只把
完整 contact min/center/max 共同平移到新 face center。

发射器随后补齐 stable-upper receipt 的等价读取路径：仅当 upper alignment 的 exact keyset
等于 retargeted schema 时才进入该分支，并要求 authority 为 pinned strike-frame selected
rubber-face center、`upper_contact_center_preserved=false`、world-Z 与
`ready_root_z + task_z` 闭合。legacy upper corrected-Z 与 full retargeted 两条路径保持不变。
这是 artifact schema 兼容修复，不构成 Reward/学习 A/B；下一步是 Pod focused regression 后
串行跑两动作 `1 env×2` smoke。

两条 smoke 都自然完成并产出 finite `model_0/1.pt`，但每轮都在 episode age `16--17` 由
`waist_pitch_joint` 上侧 raw mechanical edge reset；腿/踝、q_des projection、table、fall、
nonfinite 全为零。共同事件把 stable-upper v1 的剩余缺口缩到 frame-0 腰 ready：旧 motion 为
`+0.103 rad`，A3 runtime default 为零。下一资产将三腰 q 轨迹整体平移到 runtime ready，逐帧
保持相对 frame-0 增量与 qd，重建 FK/contact 后重跑同 smoke；此项不改变 Reward 或学习问题。
