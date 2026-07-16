# EXP-P1-LATERAL-BALANCE-PERTURBATION — 用可恢复的横向推力增加平衡学习机会

- 状态：`proposed`
- 预注册状态：机器可读草案；`launch_authorized=false`
- 阶段/轴：连续能力线 / 击球后恢复与等待姿态的状态覆盖
- 集成小目标：上一拍后吸收余势、保持可接战平衡、按时启动下一拍
- 人类负责人：franco
- 执行者：Codex
- 复核/决策负责人：franco
- 最高证据等级：[E1（源码与单测）](../../DEFINITIONS.md#证据和文档术语)；已有 Isaac adapter/hook、
  trainer 入口与 checkpoint hard-contract 候选源码和 mock 生命周期证据，但没有 full-scene runtime、
  训练或行为证据
- 创建日期/最后复核日期：2026-07-15 / 2026-07-16

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

- pulse 启动后若击球窗意外开始或 recovery/hold 安全窗缩短，下一步必须写零，不能把残余外力带进挥拍。
- episode reset、strike 和 safe-window closure 截断 pulse 时都必须先保存逐环境账再清状态；reset 当步禁止
  立刻重启。三类账都同时保存原采样冲量、已 command/已 applied 冲量、尚未 command 和已 command 未
  applied 的放弃量，并逐环境满足
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
提供 simulator-independent scheduler 与事务 seam：

1. 纯 torch 的确定性 scheduler/kernel：Random123 已知向量一致的 `Philox4x32-10`、有界随机幅度、
   左右对称、完整安全窗、同一步幂等、漏步 fail closed、reset 中断冲量对账。
2. fail-closed adapter seam：把归一化脉冲按**整机总质量**变成 WORLD-Y 躯干质心力，并要求 adapter
   先做不改 live backend 的 typed preflight，再做完整私有 WORLD command overwrite + 同步 exact readback；成功只返回
   `None`。Python/CUDA copy 不能自证 atomic/noexcept；异常/非 `None` 必须在 application ledger 前把 backend
   标成 terminal `DIRTY/UNKNOWN`，禁止 retry、advance 或下一 simulator step。
   preflight receipt 与 application ledger 必须逐环境绑定随机化后实际总质量、runtime dtype 的 normalized
   acceleration、命令 WORLD-Y force/impulse、applied mask、WORLD→backend transform SHA、adapter/backend
   SHA 与 live backend object token；同一步换质量、力、transform 或 live backend 都会拒绝。详细接口见
   [横向扰动 adapter 事务合同](../../interfaces/lateral_perturbation_adapter_contract.md)。

[`isaac_lateral_perturbation.py`](../../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/isaac_lateral_perturbation.py)
新增了一个**默认关闭**的 Isaac Lab `v2.1.0` 候选 adapter/hook；它固定到
Isaac Lab commit `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`。2026-07-16 的 E1 后续已把它接进
`train.py` 的显式 opt-in wrapper，但 full-scene/solver-response/throughput 未过，仍禁止点火。二次源码审计
纠正了更关键的语义：pinned
`Articulation.write_data_to_sim()` 最终以 `position_data=None` 调用 PhysX，这表示 link origin，**不是 COM**。
原 BODY-buffer 候选因此被红队否决；新候选每个 substep 都显式计算/传入当前 WORLD torso COM：

- 从 `root_physx_view.get_masses()` 读取当前随机化后全身质量；Isaac 内建完整 robot force/torque buffer 必须
  始终全零、identity 不变且 `has_external_wrench=false`，只作为 same-tick/non-torso 竞争 writer 哨兵；
  EventManager interval term 在 probe 中一律拒绝；
- 每个 policy step 完整覆盖 adapter 私有 WORLD force/torque command。每个 physics substep 读取
  `body_pos_w`、`body_quat_w` 与已由 pinned IsaacLab 搬到 articulation device 的 `data.com_pos_b`，计算当前
  torso COM，然后 direct 调用
  `apply_forces_and_torques_at_position(position_data=<explicit COM>, is_global=true)`；`position_data=None` 禁止；
- scene write 异常、wrong return、竞争 writer 或任何 post-dispatch 验证失败都会进入 terminal guard；没有竞争
  writer 时先 direct 提交全零，随后禁止 retry/advance/下一 simulator step；
- strike/window closure 当步提交全零；发生 subset reset 时，Isaac 的额外全-scene write 前先把**全部环境**
  清零，避免非 reset 环境在 decimation 之外多吃一次力，并在下一 policy step 按账重新施加仍有效的 pulse；
- 逐步回执绑定 episode index/step、strike/eligible/safe-window、application ledger、每个 substep 的 WORLD
  command、显式 COM/local offset、同步态和 reset 清零态；`enabled=false` 直接原样委托 `env.step(action)`。

诚实边界不变：Isaac Lab 2.1 没有可读取“PhysX solver 实际消费的 wrench”的 getter。当前回执只证明 direct
setter 收到 exact WORLD command/显式 COM，**不证明 solver execution**；而且正确性优先路径仍有 host
sync，尚未过同 GPU throughput/no-host-sync 门。严格 full-scene probe 已提供在
[`probe_lateral_perturbation_runtime.py`](../../../hope_training/whole_body_tracking/scripts/probe_lateral_perturbation_runtime.py)，
但本记录没有运行它，`launch_authorized` 与 `runtime_adapter.implemented` 继续为 `false`。操作边界和命令见
[`run_lateral_perturbation_runtime_probe.md`](../../operations/run_lateral_perturbation_runtime_probe.md)。
probe 的 reset/strike 结论必须非空覆盖：若零动作场景没有自然 reset 或 active-pulse strike interruption，
结果只能写 explicit-COM/lifecycle-uncovered，reset evidence 为 `null`，不能用 vacuous all-pass 升级。

### 2026-07-16 trainer-facing E1 接线（仍然 `NO-LAUNCH`）

训练入口现在只接受三个 Hydra 字段：`enabled`（是否启用）、`cell`（`L0` 零推力同调度对照或 `L1`
随机推力 treatment）和 `seed`（counter-based 随机题种子）。不存在可由命令行修改的 body/frame/torque/
XYZ force/持续时间/强度字段：`L1` 继续冻结为每 `0.50 s` 一个机会、eligible 后 `p=0.5`、持续 `0.10 s`、
归一化冲量 `Uniform(0.04, 0.08) m/s`；`L0` 使用相同题但冲量为零。`enabled=false` 或配置完全缺失时，
不向 env cfg 附加字段、不包装 `env.step`、也不在历史 hard contract 中增加 key；`cell/seed` 与 disabled
混用会 fail closed。

启用时，checkpoint hard contract 逐项绑定 resolved policy dt/整数 tick、cell/seed、共同随机题 SHA、hard
safety envelope、Isaac backend/显式 COM transform identity 和 metric schema。trainer 不保留逐 step 的完整
4096-env probe transcript，避免 receipt 随 PPO step 无界增长；它只向 `extras['log']` 的副本发布标量：
opportunity/eligible/selected/commanded/applied/zero-overwrite 整数、reset/strike/window 的 abandoned impulse、
sampled/commanded/applied impulse，以及随机化后整机质量 min/mean/max。原 env-owned `extras` 不被改写；
metric key 竞争、非五元 Gym output、非 dict log 或关闭时 terminal zero 失败都让 run 无效。

这个接线没有改变 `launch_authorized=false`：当前 adapter 每个 substep 仍有 correctness-first host sync，
也没有 full-scene、solver dynamics response 或同 GPU throughput 证据。下面的 probe/throughput/held-out
门全部照旧；今晚的 rolling 组合训练不能依赖本分支。

首轮机器账至少同时记录：

- opportunity 总数、eligible denominator、selected numerator；
- 左/右选择数与采样强度总量；
- 非零 pulse command、strike-window skip、窗口不足、意外中断；
- 每步 side-effect-free preflight receipt、成功 full-buffer commit/readback、真正 applied pulse 数与 applied impulse 总量；
- 每次 pulse 的 environment/episode/step、采样 `Δv_y`、命令 `F_y`、剩余 pulse step 和 adapter ledger。
- potential 随机 draw、共同随机题 SHA，以及 reset/strike/window 截断时各自的
  sampled/commanded/applied/abandoned 五项冲量账。
- 实际总质量、命令 WORLD force、transform identity 和 backend identity 必须由 typed preflight receipt
  原样回显，再与 dispatch
  输入逐 tensor 对账；这使 `F_y / total_mass = normalized_accel_y` 可独立复算，但仍不是 simulator 已执行
  的行为证据。
- 每步 sample/application ledger 同时绑定 hard safety envelope identity；adapter 调用前保留质量与 wrench
  snapshot，adapter 若原地修改输入再重写回执也会 fail closed。
- adapter 只能收到 mass/force/torque 的隔离深拷贝；preflight 原地写坏三者后拒绝或抛异常，调用者持有的
  mass 与 scheduler step tensors 仍逐 bit 不变。只有 side-effect-free preflight 拒绝可 discard 后用同
  step token、全新 preflight nonce 重试；commit 一旦进入后若抛异常或返回非 `None`，backend 永久标成
  `DIRTY/UNKNOWN`，普通 retry/advance 必须拒绝。
- scheduler 的 application ledger cache 永不直接公开：首次返回、显式 cache 查询和同一步 duplicate
  dispatch 每次都生成新的 tensor 深拷贝。调用者改坏任一返回 ledger 后，后续副本和计数仍须保持原值。
- public step 中的 tensor 虽属于 frozen dataclass，内容仍可被外部改写；因此 dispatch 必须在读取任何
  adapter 字段或计算 wrench **之前**，先把 public step 与 scheduler 私有 canonical step 全字段比较，
  再只从私有 validated clone 派生 force。若比较失败，adapter call 必须严格为 0、application cache 仍空，
  reviewed caller 可用同一 step token 取得未污染的 canonical clone 并安全重试。
- 不存在公开的 application acknowledgement；scheduler bookkeeping 只认模块内 dispatch identity
  capability，所有 expected 都从私有 canonical step 推导。source 生成的一次性 token 必须由 preflight
  object-identity 原样回显；它是 Python API 防误用，不是抵抗同进程恶意 introspection 的安全边界。
- CUDA async assert 不能保证 Python 在 device failure 可见前不调用 writer，因此当前 source seam 在 public
  step、mass/cast/final wrench、cache duplicate 和 receipt tensor/mask 各层使用**多次** host-visible
  completion。这是正确性优先的 E1 实现，同时意味着它尚未满足 hot-path no-host-sync 门；runtime 接线必须
  消除所有这些同步或重设计 handoff，再通过同 GPU throughput 门，才能把 `launch_authorized` 改成 true。

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
| 零推力对照 `pending` | 未启动 | seed 1 | source/mock only | 无 | trainer/hard-contract 源码已接；adapter 未过 full-scene/solver-response/throughput，禁止点火 |
| 横向脉冲 treatment `pending` | 未启动 | seed 1 | source/mock only | 无 | trainer/hard-contract 源码已接；adapter 未过 full-scene/solver-response/throughput，禁止点火 |

## 决定

- 决定：`inconclusive`
- 理由：第一性原理设计、scheduler、Isaac adapter/hook 与 trainer/hard-contract 候选的源码/mock 生命周期
  成立，但没有 full-scene runtime、solver-response、throughput、训练或留出考试。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：在 exact clean Isaac Lab `v2.1.0` 环境先运行一次独立、无 trainer 的 strict full-scene
  probe，核验实际 mass/body/frame/write/reset 生命周期；再以独立 dynamics-response probe 证明非零力确实进入
  solver，并完成 GPU throughput/no-host-sync 重设计。trainer 与 hard contract 已有 E1 接线，但只有上述门
  通过后才允许激活 launch、冻结内容寻址的 ball×action-family held-out paper 并生成配对 queue。

## 复现源码证据

不需要 Isaac、Pod 或真机：

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_lateral_perturbation.py \
  hope_training/whole_body_tracking/tests/test_isaac_lateral_perturbation.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py
```

当前 trainer/scheduler/adapter/translation 聚焦测试为 `170 passed`：scheduler/transaction 文件 `40`
项、dependency-light adapter/hook 文件 `34` 项、训练 override/hard-contract 文件 `96` 项。覆盖包括：
Random123 Philox 零 counter/key 已知向量、四个 domain 与相邻 seed 的分桶
均匀性和交叉相关性、`L0/L1` potential draw/SHA 完全相同、左右对称与幅度界、recovery/hold eligibility、
strike skip、完整 pulse 冲量、reset 中断五项账、同一步幂等、漏步/漏 application receipt fail closed、
按总质量缩放、质量/力/transform typed ledger，对极大 finite impulse、duration overflow、cast overflow、
mass×acceleration overflow 和 force 上限的负测、X/Z force 与 torque 恒零，以及 pulse 后 full-batch 零写。
另含 adapter preflight 原地篡改/异常后的 caller bit-exact、不公开 acknowledge、CUDA async-assert
neutered、坏 receipt/stale token、不同 live backend cache replay、commit 抛异常/非 `None` 返回等攻击回归；
所有 precommit 失败都满足 backend write=0/cache 空，side-effect-free staging 会 discard，同 tick 可安全重试。
strike/window/reset 的逐环境 sampled/commanded/applied/abandoned 恒等式及中断 tick backend 全零也已覆盖。
adapter/hook/artifact 测试覆盖：随机化后真实总质量读取、全量 active EventManager term manifest 绑定、
任一 interval term 在首次 force submit 前 fail closed、显式非零 local-COM
offset 旋转到 WORLD、派生 WORLD COM overflow 在 setter 前 terminal 拒绝、每 substep 传非 `None position_data`、
只写 `torso_link` WORLD force、既有 wrench owner、
same-tick/non-torso 与 reset writer 拒绝、direct setter 内同 tick 竞争 writer、scene/direct-setter exception、
scene/env wrong return 与 scene-hook restore 失败后的 terminal zero、clean rollout 在任何验收/落盘前 terminal
zero、terminal-zero 失败禁止发布、默认关闭直接委托、strike/reset 当步全零及 episode/impulse 对账、T1
event-driven 时序拒绝、motion inode swap、output parent symlink swap、stable-dirfd no-clobber，以及所有回执显式
`solver_execution_readback_available=false`。trainer 接线测试另覆盖：默认关闭不附加 runtime spec、
L0/L1-only Hydra translation、unknown/body/anytime/非 uint32 seed 拒绝、50 Hz frozen schedule、L0/L1
共同随机 SHA、conditional checkpoint hard contract、无界 receipt 禁止、连续两 step 的非侵入 extras 标量账、
metric 竞争后的 terminal zero。它不证明真实 simulator 满足相同 lifecycle，也没有 solver-response 或 GPU
throughput 证据。

在最新 `origin/main@107102f` 重放后，whole-body tracking 的 57 文件整合套件为
`847 passed, 22 skipped, 3 failed`。三项失败是未改动路径中的既有主线基线：MotionLoader 对两个
`PosixPath` case 抛 `TypeError`，virtual scorer 的 `0.9999999999979997` 超出 `1e-12` 容差；同环境在
`origin/main` 原样重跑三项均失败，且四个相关源码/测试文件相对 `origin/main` 无 diff。因此本 source
gate 没有新增整合回归，但也没有顺手修改这三个不在本实验范围内的问题。

相关 Gate：[`G05 Isaac training first loop`](../../gates/G05_isaac_training_first_loop.md)；连续恢复的结构顺序见
[`EXP-RECOVERY-TUPLE-ABC`](EXP-RECOVERY-TUPLE-ABC.md)。
