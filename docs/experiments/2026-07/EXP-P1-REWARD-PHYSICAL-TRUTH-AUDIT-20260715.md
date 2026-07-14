# EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715 — 现役 Reward 到底有没有检查球能否上台？

- 状态：`completed`（source/config 语义审计；没有改动在跑配方）
- 阶段/轴：阶段 1 Reward 诚实性 / 解析结果与物理结果分层
- 集成小目标：Reward 提供可学习梯度，但选档与晋级必须由独立物理结果判定
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（exact source/config 审计；没有新增 simulator 行为）
- 创建日期/最后复核日期：2026-07-15 / 2026-07-15

共享术语见[术语与人话对照](../../DEFINITIONS.md)。本文只解释当前 Pod2 fresh v4
control/treatment 的 Reward 与真值仪语义，不改变它们的训练合同，不授权重启、第二 seed、judge、
晋级或真机。

## 结论

答案是：**会检查，但目前检查的是由实际球拍状态推演出来的解析回球，不是真实物理球回球。**

planner/题库先给出目标触球位置、拍速和有符号拍面。训练时并不只看 planner 的目标值：源码读取 policy
实际执行后的球拍 FK 位置、速度和拍面，先过有符号拍面、`9.5 cm` 捕获半径与 `0.3 m/s` 接近速度门，
再用场馆拟合的拍球冲量和飞行方程推演是否过网、落在哪里以及出球旋转。这个解析结果确实进入当前
`HOPEPingPongVirtualBall` Reward。

但当前 `physical_ball=true` 只是另一条 Phase-A engine-integrated 诊断：位置由 PhysX 积分，空气力与
台面反弹由代码驱动的场馆模型负责。现役配方没有启用
`physical_ball_impulse=True`，所以物理球不被球拍击出、会穿过机器人；该模块也明确禁止 Reward/obs
读取它。因此当前训练**没有**“真实球拍碰到物理球并真实落台才给分”的项。

## exact 运行绑定

当前配对绑定 source `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e` 和
[`phase1_fresh_c_v1v2_base_decel_measurement_rerun_queue_20260715.yaml`](../../../configs/phase1_fresh_c_v1v2_base_decel_measurement_rerun_queue_20260715.yaml)
（审计时 SHA-256 `4c438b88...1dfa`）。相关 exact source bytes：

| 文件 | SHA-256 | 本审计读取的语义 |
| --- | --- | --- |
| `cfg/task/HOPEPingPongVirtualBall.yaml` | `1cb74cfb...8ad` | 任务选择 virtual-ball outcome Reward |
| `config/agibot_a3/hope_env_cfg.py` | `b1347124...6b5` | outcome 权重与 `virtual_ball=True` |
| `mdp/hope_rewards.py` | `f9abd1c4...a9e` | 目标匹配、解析过网/落点/旋转的实际公式 |
| `mdp/hope_commands.py` | `b43e85da...576b` | achieved FK → 捕获门 → 解析接触/飞行 |
| `mdp/physical_ball.py` | `c15a9cf5...314` | PhysX 真值仪 metrics-only、默认无拍球冲量 |

## 当前真正生效的 Reward

配方选择 `task=HOPEPingPongVirtualBall`。这个 task 的 Reward class 明确包含：

| 通道 | 当前权重 | 读到的量 | 能否单独证明真实物理上台 |
| --- | ---: | --- | --- |
| 球拍位置匹配 | `14` | achieved FK 拍心 vs 题目目标/挥拍轨迹 | 否 |
| 球拍速度匹配 | `10` | achieved FK 拍速 vs 题目目标 | 否 |
| 有符号拍面匹配 | `5` | achieved raw-A/physical-B vs 题目目标 | 否 |
| 三核乘积击球奖金 | `5` | 连续的位置核×速度核×拍面核；不是三项过阈值后二值发奖 | 否 |
| 朝目标靠近 | `10` | 击球前拍心到目标距离的减少 | 否 |
| 解析过网 | `20` | achieved 球拍状态解析推演出的过网高度 | 否 |
| 解析落点 | `30` | achieved 球拍状态解析推演出的落点 | 否 |
| 解析旋转 | `5` | achieved 球拍状态解析推演出的出球旋转 | 否 |

队列把 VirtualBall YAML 的目标匹配默认 `4/0.5/0.5` 显式覆盖成 `14/10/5`，但没有删除
`virtual_pass_net/virtual_landing/virtual_spin` 三个 outcome term。`vb_metrics_only=true` 只是保证即使某个
任务没有 outcome Reward 也能记录 `virtual_*` 曲线；它**不是**“关闭解析 Reward”的开关。当前 task
同时令 `virtual_ball=True`，所以上表两组通道都生效。仓库现有 source-bound 回归也把
`vb_metrics_only=true`、`virtual_ball=true` 与 `20/30/5` 同时锁为合法组合；在当前 VirtualBall task 中，
这个 metrics-only 开关对是否计算解析曲线几乎是冗余的 OR 条件。

## 哪些分数可以在完整合法回球前拿到

解析 outcome 不是全稀疏的“落台才发钱”：

- 所有 virtual outcome 都先要求一次 achieved-state 捕获：exact strike、有符号拍面正确、拍心误差
  `<9.5 cm`，且球拍沿接触法向的接近速度 `>0.3 m/s`。
- `virtual_pass_net` 的高度 kernel 只要解析轨迹到达网平面就能拿部分分；额外 `0.5` bonus 才要求
  解析轨迹同时合法落台。
- `virtual_landing` 的距离 kernel 只要解析 rollout 给出落点就能拿部分分，不要求先过网；额外 `1.0`
  bonus 才要求过网、落在对面且足够深。
- `virtual_spin` 才由合法解析落台完整门控。
- 位置、速度、拍面、接近目标和三项乘积本来就是 planner-target shaping，不依赖落台。

所以这套 Reward 的设计意图是“先让 policy 找到球，再把解析回球推深并过网”，而不是只在最终成功时
给一个稀疏分。代价是解析接触模型、拍面约定或跨引擎执行一旦有误，Reward 可以系统性虚高；历史
signed-face 和 Isaac→MuJoCo 反例已经证明这个风险是真实的。

## physical-ball 真值仪为什么还没有变成 Reward

`physical_ball.py` 在现役 Phase A 只记录来球抵达误差、穿透机器人后的 engine-integrated 轨迹/落点等
serve 诊断；它不会产生物理触拍或 return net/landing。物理触拍、回球过网/落点与解析—engine gap 是
Phase-B 拍球冲量打开后的能力，当前没有产生。两阶段都刻意不改 Reward 和 observation，以免一边训练
一边把裁判并入目标造成无法归因。只有未来内容寻址 rider 显式打开 Phase B，才会产生物理回球轨迹；
即使打开，当前源码也仍然只记指标，不自动发 Reward。

## 下一步：先闭合裁判，再做 outcome-source 消融

本轮在跑的 base-decel pair 保持原合同；不能中途换 Reward，也不能把解析曲线改名为物理成绩。
后续按以下依赖顺序推进：

1. **Phase-B 真值门：** 在同题、同动作、同 checkpoint 上启用 code-authoritative 拍球冲量，先验证
   physical hit→net crossing→opponent-table landing 事件链、all-serves 分母和解析/物理落点差；该步只
   记指标，不训练。
   必须同时注明：Phase-B 球拍冲量仍复用 `virtual_ball.predict_paddle_contact`，所以它能验证接触检测和
   engine-integrated post-contact flight，却不是完全独立的接触物理裁判。
2. **裁判跨引擎门：** 同一 actor action / racket trajectory 在 Isaac physical instrument 与厂商 MuJoCo
   上逐层对拍；vendor MuJoCo 仍是最终票。
3. **单 seed、固定 outcome 总预算配对：** 两臂都打开同一 Phase-B metrics-only 路径，并冻结题库、
   动作、planner、direct shaping、PPO 预算与安全项。将 outcome 写成
   `W * ((1-λ) * A_norm + λ * P)`：`A_norm` 是归一化解析 outcome，`P` 是一次性的
   physical-contact ∧ net-clear ∧ opponent-table-landing 终局事件；首轮只比 `λ=0` 与 `λ=0.25`，避免把
   “是否使用物理真值”和“完全拿掉稠密探索梯度”混成一个问题。两边都由独立 clean physical receiver
   判分，解析训练分不能作晋级分。`A_norm` 在 prereg 中按当前 raw 理论上界固定为
   `(20*pass_net + 30*landing + 5*spin)/95` 并截到 `[0,1]`：pass-net raw 上界 `1.5`、landing `2`、
   spin `1`，所以加权上界是 `95`，不是权重表面和 `55`。
4. **统一奖励结算时序：** 解析 `A_norm` 虽在 exact strike 可算，也必须先锁存；两臂都只在同一个
   outcome settlement event 发放 outcome Reward。pending receipt 绑定不可变
   `(episode_epoch, swing_id, question_id)`，exactly-once/no-clobber；物理落台或冻结 deadline 到达时结算，
   miss/net/out/timeout 的 `P=0`，不得把上一拍落台分记到下一题。若连续任务允许下一题更早揭示，ledger
   仍按旧 swing 归属并在训练合同中明确延迟；否则首轮保持 option transition 到 settlement 后再发生。
   这样 `λ=0` 与 `λ=0.25` 的支付时刻相同，才主要隔离 outcome source；若选择在 strike 立刻支付解析分、
   落台才支付物理分，就必须诚实改名为“source+timing package”消融，不能声称纯来源比较。
5. **比例/交互：** 只有 `λ=0.25` 的 activation、clean 非退化与物理留出方向都过门，才继续扫描比例或
   做 analytic shaping × physical terminal 的 `2×2`；不能先跑一排权重碰运气。

任何胜者都必须同时守住平衡、分动作 clean 回球、无推连续表现与 held-out 扰动表现；不能用更高的解析
训练分抵消真实物理回球退化。

## 复现

```bash
git show 2c2d70d:hope_training/whole_body_tracking/cfg/task/HOPEPingPongVirtualBall.yaml
git show 2c2d70d:hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py
git show 2c2d70d:hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py
git show 2c2d70d:hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py
git show 2c2d70d:hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/physical_ball.py
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_virtual_return_scorer.py::test_training_face_reward_forensic_summary_is_canonical_and_source_bound
```

最后一项在本机为 `1 passed`。依赖 Hydra 的 config-composition 测试在本机系统 Python 因
`ModuleNotFoundError: hydra` 未收集，不能把缺依赖写成源码失败；source-pinned YAML/queue/Reward class
与既有 source-bound fixture 已足以回答本审计的静态语义问题。没有 simulator、Pod 写操作或真机命令。
