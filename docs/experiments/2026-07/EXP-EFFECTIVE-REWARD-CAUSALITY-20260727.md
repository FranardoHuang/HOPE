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

- raw q_des 非有限、实际关节 hard-limit、有限目标在一个 physics step 内产生的 ballistic/substep
  crossing、table hit 和 fall 仍 hard reset；
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
