# EXP-P1-LATERAL-BALANCE-PERTURBATION — 用可恢复的横向推力增加平衡学习机会

- 状态：`proposed`
- 预注册状态：机器可读草案；`launch_authorized=false`
- 阶段/轴：连续能力线 / 击球后恢复与等待姿态的状态覆盖
- 集成小目标：上一拍后吸收余势、保持可接战平衡、按时启动下一拍
- 人类负责人：franco
- 执行者：Codex
- 复核/决策负责人：franco
- 最高证据等级：[E1（源码与单测）](../../DEFINITIONS.md#证据和文档术语)；没有 runtime、训练或行为证据
- 创建日期/最后复核日期：2026-07-15 / 2026-07-15

## 问题与第一性原理假设

现役短片段里，机器人前几拍通常不会立刻摔；脚距逐渐变窄、站姿逐渐歪等平衡债，往往要很多拍后才显现。
因此真正能教策略“如何从将要失衡的状态恢复”的样本很稀疏。单纯延长训练不一定高效，因为绝大多数
rollout 仍停留在容易状态。

首个可证伪假设是：**只在击球后的恢复或等待窗口，给躯干质心施加左右对称、强度有界的短时横向力，
能增加可恢复的平衡误差样本；相对零推力对照，它应在留出的强扰动考试中更快回到可接战状态，同时不降低
无扰动考试的下一拍质量。**

这不是额外 Reward，也不直接修改 root velocity。直接改速度会跳过接触、驱动和惯性响应，无法回答策略
是否学会通过脚、腿和躯干吸收真实外力。这里冻结的是力-时间脉冲：用随机脉冲对应的整机速度增量预算
`Δv` 表示强度，再按实际随机化后的整机总质量换成躯干力：

```text
F_world_y = total_articulation_mass * sampled_Δv_y / pulse_duration
F_world_x = F_world_z = 0
torque_world = 0
```

力作用于 `torso_link` 的质心，因此显式 torque 为零，也不会产生相对**该 link 自身质心**的额外
lever-arm moment。但 torso 质心不是整机质心：WORLD-Y 外力相对整机质心仍可产生物理的 `r×F` 角冲量，
地面接触也会改变实际 `Δv`。这里没有“纯平移”或“整机角动量不变”的声明。`Δv` 只是冲量除以整机质量
的归一化单位，代码不会把它写进机器人速度。

## 首轮冻结的因果轴

权威草案是
[`phase1_lateral_balance_perturbation_prereg_20260715.json`](../../../configs/phase1_lateral_balance_perturbation_prereg_20260715.json)。
首轮只有一对单 seed 配对：

| 格 | 人话 | 归一化脉冲强度 | 其他 |
| --- | --- | --- | --- |
| `L0` | 零推力对照 | `[0, 0] m/s` | 保留与 treatment 相同的机会、选择、方向和虚拟 pulse 占用时间 |
| `L1` | 随机横向躯干脉冲 | `Uniform(0.04, 0.08) m/s`，方向左右各 `0.5` | 0.10 s 脉冲；每 0.50 s 一个随机相位机会，eligible 后以 `0.5` 概率选择 |

两个格使用同一个无状态 schedule seed。这里的“共同随机数”是指：用
`Philox4x32-10` counter generator 按 `seed/environment/episode/opportunity` 生成题目，再用四个固定的
32-bit domain tag 分开机会相位、是否选择、左右方向和单位幅度；即使两个策略之后走到不同状态，也不会
因为调用 RNG 的次数不同而悄悄换题。共同随机题身份 SHA-256 是
`d157bd6e7c063df80d41ca03b9eb4acae2a4b45c9ee0967b5dcbce5b76d14593`，并且每步显式暴露 potential
selection/direction/unit-magnitude draw，允许直接对账 `L0/L1`，不再用线性 stream 偏移冒充独立随机流。
Philox 的单位随机数取值严格在 `(0,1)`；非退化幅度区间因此不包含配置端点，只有上下界相等时才是一个
点质量，不能写成离散意义上的 closed interval。

### Eligibility 与硬安全边界

第一次只允许以下交集：

```text
(post-strike recovery OR pre-swing hold)
AND NOT strike_window
AND safe_window_remaining_steps >= pulse_duration_steps
```

- pulse 启动后若击球窗意外开始，下一步必须写零并记 `interrupted_for_strike`，不能把残余外力带进挥拍。
- episode reset 若截断 pulse，reset 当步禁止立刻重启；账本同时保存原采样冲量、reset 前已 command/已
  applied 冲量、尚未 command 和已 command 未 applied 的放弃量，并逐环境满足
  `sampled = commanded + abandoned_uncommanded` 与
  `commanded = applied + abandoned_unapplied`。
- 每个 simulator step 都必须覆盖完整 torso wrench buffer；无脉冲的环境也显式写零。漏一步就 fail closed，
  因为旧外力可能继续存活。
- 允许任何时刻扰动的 `anytime` 版本是**后续独立因果轴**。它不能混进首格；只有 recovery/hold 格在留出
  考试存活后，才允许新建预注册。
- self-hit、桌网碰撞、关节/力矩上限和物理摔倒仍是不可补偿硬门，推力训练不能用别的 Reward 抵消它们。

源码另冻结了一个不允许调用者放大的 hard safety envelope（身份 SHA-256
`7de6f9a7ab418a63973e1680a56d7ca82d9b8c19cd1ac52d32d332cb6819dc45`）：

| 量 | 硬上限 |
| --- | --- |
| 归一化冲量 `|Δv|` | `0.15 m/s` |
| 归一化加速度 `|Δv|/duration` | `2.0 m/s²`（约 `0.204 g`） |
| pulse 时长 | `[0.02, 0.20] s`（50 Hz 下 1–10 tick） |
| 最终 WORLD-Y force | `200 N` |
| WORLD X/Z force；显式 torque | 恒为 0 |

这些值覆盖冻结的 L1（最大 `0.08/0.10=0.8 m/s²`）和 held-out strong（最大
`0.14/0.10=1.4 m/s²`），同时给任意配置、float64→runtime dtype 转换和 mass multiplication 一个独立
fail-closed 后挡板。配置的 impulse/duration/derived acceleration、cast 后 acceleration、最终 wrench 与
WORLD-Y force 每层都必须 finite 且在界内；极大但 finite 的输入也不能穿透。`200 N` 是仿真命令上限，
不是任何真机安全证书。

## 源码边界与激活账本

[`lateral_perturbation.py`](../../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/lateral_perturbation.py)
现在只提供两个部分：

1. 纯 torch 的确定性 scheduler/kernel：Random123 已知向量一致的 `Philox4x32-10`、有界随机幅度、
   左右对称、完整安全窗、同一步幂等、漏步 fail closed、reset 中断冲量对账。
2. fail-closed adapter seam：把归一化脉冲按**整机总质量**变成 WORLD-Y 躯干质心力，并要求未来 adapter
   对完整 batch 写入且返回 typed receipt。receipt 与 application ledger 必须逐环境绑定随机化后实际总
   质量、runtime dtype 的 normalized acceleration、命令 WORLD-Y force/impulse、applied mask，以及
   WORLD→backend transform 的 content identity；同一步若换质量、力或 transform identity 会拒绝。

本提交**没有**把 seam 接进 Isaac。Isaac Lab 2.1 的实际接口接线仍需确认 body-frame wrench 语义、
WORLD→BODY 变换、`write_data_to_sim` 时序、随机化后总质量读取和 zero overwrite 生命周期；在这些通过
真实 runtime smoke 前不得写成 launch-ready。

首轮机器账至少同时记录：

- opportunity 总数、eligible denominator、selected numerator；
- 左/右选择数与采样强度总量；
- 非零 pulse command、strike-window skip、窗口不足、意外中断；
- 每步 full-buffer write receipt、真正 applied pulse 数与 applied impulse 总量；
- 每次 pulse 的 environment/episode/step、采样 `Δv_y`、命令 `F_y`、剩余 pulse step 和 adapter receipt。
- potential 随机 draw、共同随机题 SHA，以及 reset 截断时 sampled/commanded/applied/abandoned 五项冲量账。
- 实际总质量、命令 WORLD force 和 transform identity 三者必须由 typed receipt 原样回显，再与 dispatch
  输入逐 tensor 对账；这使 `F_y / total_mass = normalized_accel_y` 可独立复算，但仍不是 simulator 已执行
  的行为证据。
- 每步 sample/application ledger 同时绑定 hard safety envelope identity；adapter 调用前保留质量与 wrench
  snapshot，adapter 若原地修改输入再重写回执也会 fail closed。

`L0` 必须有 eligible/selected，但 `applied_pulse_count=0`；`L1` 必须有非零 applied pulse。采样、命令和
application 三本冲量账对不上时，结果无效而不是“近似通过”。

## 留出考试与决策规则

训练 seed/时序不得用于考试：

| 考试 | 脉冲 | 用途 |
| --- | --- | --- |
| clean | `[0,0] m/s` | 判断无扰动下的摔倒率、guard reset 和下一拍击球是否退化 |
| strong | `Uniform(0.10,0.14) m/s`，held-out seed | 超出训练强度的恢复压力测试；不参与 PPO 或 checkpoint 选择 |

两张卷必须共用内容寻址、不可变的 `ball-arrival-bin × action-family` 题表、episode 和 deadline schedule，
每个非空 bin 至少 50 题，以 all-attempt 为分母，并同时报告所有 bin 与 worst bin。当前动作族、来球分桶
和题表 SHA 都仍是 `pending`，因此既不能 launch 也不能晋级；最终同卷还必须在 vendor MuJoCo 复核。
草案的首轮晋级线是：

1. 激活与 application 账完全闭合；
2. clean 物理摔倒率不比 `L0` 高超过 1 个百分点，下一拍 composite 不低超过 3 个百分点；
3. strong 的 balance-debt AUC 与回到 ready set 的中位时间都至少降低 10%，ready-by-deadline 至少提高
   5 个百分点，且物理摔倒率不升；
4. strong 同时记录 capture point（按当前质心位置/速度估计的落脚稳定点）与 COM（质心）到支撑域的最小
   margin、相对**每个 episode 起始脚距**的最大变窄量、左右脚 yaw 误差。相对 `L0`，`L1` 不得让
   capture-point/COM 最小 margin 低超过 1 cm、p95 脚距变窄多超过 2 cm，或 p95 脚 yaw 误差多超过 5°；
5. runtime 前先在同 GPU、同 scene/seed/schedule 下测 512 env 和 production env count：相对 no-hook
   control，environment-steps/s 不低于 `0.95×`、p95 sim-step time 不高于 `1.05×`，且 hot path 不得有
   host-device sync；未过不允许 launch；
6. 上述门和 held-out `ball × action-family` 每 bin/worst-bin 报告同时通过，才买第二 seed。
   `+200/+500/+1000` checkpoint 只作早判，不给失败格复制 seed。

这些阈值仍是预注册草案的一部分；在内容寻址题表和 runtime adapter 未闭合前不允许点火或事后改成
“已经采用”。最终部署裁判仍是厂商 MuJoCo；Isaac 训练曲线只能筛方向。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 零推力对照 `pending` | 未启动 | seed 1 | 无 | 无 | runtime adapter 与 hard contract 未绑定 |
| 横向脉冲 treatment `pending` | 未启动 | seed 1 | 无 | 无 | runtime adapter 与 hard contract 未绑定 |

## 决定

- 决定：`inconclusive`
- 理由：第一性原理设计和纯源码门成立，但没有 Isaac adapter、full-step application ledger、训练或留出考试。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：实现独立 Isaac adapter 与 runner/hard-contract 接线；验证随机化后整机质量、真实
  WORLD-Y 脉冲积分和 pulse 后连续零写；完成 GPU throughput 门并冻结内容寻址的 ball×action-family
  held-out paper；再做小环境 runtime smoke，最后才生成可点火配对 queue。

## 复现源码证据

不需要 Isaac、Pod 或真机：

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_lateral_perturbation.py
```

当前 `24 passed`。测试覆盖：Random123 Philox 零 counter/key 已知向量、四个 domain 与相邻 seed 的分桶
均匀性和交叉相关性、`L0/L1` potential draw/SHA 完全相同、左右对称与幅度界、recovery/hold eligibility、
strike skip、完整 pulse 冲量、reset 中断五项账、同一步幂等、漏步/漏 application receipt fail closed、
按总质量缩放、质量/力/transform typed ledger，对极大 finite impulse、duration overflow、cast overflow、
mass×acceleration overflow 和 force 上限的负测、X/Z force 与 torque 恒零，以及 pulse 后 full-batch 零写。
它不证明 simulator 真正执行了这些命令，也没有 GPU throughput 证据。

相关 Gate：[`G05 Isaac training first loop`](../../gates/G05_isaac_training_first_loop.md)；连续恢复的结构顺序见
[`EXP-RECOVERY-TUPLE-ABC`](EXP-RECOVERY-TUPLE-ABC.md)。
