# EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720 — 时序平滑与挥拍后稳定机制是否正交（24 格矩阵）

- 状态：`preregistered`
- 运行态：`E1 source/prereg only；NO-LAUNCH`。24 格均未发射；命令渲染被 queue config 的占位
  `source.commit=PENDING_EXACT_COMMIT` 锁死（fail-closed），发射前还须过 `--checklist`
  依赖核对单与 `origin/main` NOW 认领。
- 阶段/轴：Phase 1 / 单拍后的平衡恢复：时序平滑（T 轴）× 挥拍后稳定机制（S 轴）
- 集成小目标：在高摔倒 parent 上降摔、在稳定 parent 上不伤击球与回台，并回答两轴是否正交
- 人类负责人：Franco
- 执行者：Claude（Codex 亦参与同轮实现，见下文分工）
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（源码 + CPU 单测）。发射后升级路径：24 份 full-scene probe receipt →
  `E3` runtime mechanics；长训自然终档 + 固定窗判读 → `E3` 单 seed 机制诊断；K100 judge 同卷 →
  仍为 diagnostic（谱系 inexact，见判卷合同）；正名须 exact-lineage 重跑。
- 创建日期/最后复核日期：2026-07-20 / 2026-07-20

人话导言：本轮把两个此前分开预注册的问题合成一个 `{W,V} × {N,C,H} × {S0,S1,S2,S3}` 的 24 格
矩阵。[`W/V`](../../DEFINITIONS.md#balance-action-slew-matrix) 是半秒冲刺留下的两个 `model_6700`
parent：`W`＝拍心优先 × 自由非击球臂的稳定父本，`V`＝拍速优先 × 强准备的高摔倒父本。T 轴
`N/C/H` 沿用 [Wave A](../../DEFINITIONS.md#balance-stability-waves) 的三档时序平滑：`N`＝完全无
平滑（[`action_rate_l2`](../../DEFINITIONS.md#raw-action-rate-l2)＝`0` 且
[`processed_qdes_slew_hinge`](../../DEFINITIONS.md#processed-qdes-slew-hinge)＝`0`）；`C`＝现役全身
raw 平滑（`action_rate_l2=-0.10`）；`H`＝关闭 raw 平滑、只在恢复窗对腿腰执行目标突变收费
（`processed_qdes_slew_hinge_weight=-0.25`、margin `0.85`、恢复窗触球后 `0.20–1.55 s`）。S 轴是四档
互斥的挥拍后稳定机制，人话见下文“S 轴语义”。每格 `4096 environments`、`seed=3`、从各自 parent
`model_6700` 续训 `10001` updates。

## 问题与可证伪假设

问题：时序平滑（对“怎么动”收费）与挥拍后稳定机制（对“动完后是否安顿”收费）是两类不同的
约束。它们是否近似正交——即 S 机制的收益在 N/C/H 三档下方向一致——还是强耦合（例如 H 已经把
S1 的腿速债务提前收掉）？哪个 T×S 组合能在高摔倒 `V` parent 上显著降摔，同时在稳定 `W` parent
上不伤击球完成与合法回台？

可证伪假设（每条都能被固定窗数据拒绝）：

1. **正交性假设：** 对每个 parent，S1/S2/S3 相对同 T 档 S0 的 fall 改善方向在 N/C/H 三档一致。
   若某机制只在特定 T 档有效（例如 S1 仅在 N 下降摔、在 H 下无效或反向），则拒绝正交性，
   后续必须把该机制与其依赖的 T 档绑定预注册，不能单独宣称机制有效。
2. **降摔假设：** 至少存在一个 `S≠S0` 机制，使 V parent 在同 T 档下 fall 相对下降 `≥25%` 且
   绝对下降 `≥5` 个百分点，同时 completion 下降 `≤2` 个百分点、legal return 下降 `≤3` 个百分点。
   若所有机制都过不了这道门，则拒绝“本轮任一挥拍后稳定机制在当前剂量下有效”。
3. **无害性假设：** 同一机制在 W parent 同 T 档下 fall 不高于 `max(0.5%, S0 + 0.2 个百分点)`，
   completion/return 分别不劣于 S0 超过 `2/3` 个百分点。只在 V 上有效但把 W 打坏的机制记
   `reject at this dose`，不晋级。
4. **T 轴假设（继承 Wave A）：** 同 parent 同 S 档内，`H` 相对 `C` 满足 Wave A 冻结的
   fall/completion/return 与 q_des-tail/tilt 门（见
   [Wave A 预注册](EXP-P1-BALANCE-ACTION-SLEW-20260720.md#预算量尺与停止规则)）；`N` 是负控，
   不是预先允许晋级的候选。

physical-fall 硬失败不能由更高 reward、composite 或更平滑的 action 抵消。

## 与 Wave A / Wave B 的治理关系（关键声明）

1. **本矩阵的 `{N,C,H}×S0` 六格取代 Wave A probe10/v9 的科学长训位。**
   [Wave A 预注册](EXP-P1-BALANCE-ACTION-SLEW-20260720.md)的六格 `science_retry2` 科学位
   **永久不再单独发射**：其 v8 科学首格与 v9/probe10 尝试均死于 locked launcher `180 s`
   stale_timeout（见下文基础设施修正记录），而本矩阵的 S0 行在同 parent、同 T 档、更长预算下
   完整覆盖同一 C/N/H 单变量问题。Wave A probe9 六格 receipt
   （receipt-set SHA-256 `cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`）继续作为
   `W/V × N/C/H` 全部六种配方能在 4096-env full scene 自然完成两个 update 的 **runtime mechanics
   证据（E3 mechanics）**被本矩阵引用；它不是科学结果，也不解锁本轮任何命令。
2. **S2/S3 行吸收 Wave B 的 B2/B1 机制，Wave B 独立六格队列由本矩阵取代。**
   [Wave B 预注册](EXP-P1-LOWER-BODY-STABILITY-20260720.md)的 `phase1_lowerbody-wave` 六格队列
   （config `phase1_lower_body_stability_20260720.yaml`）**不再单独发射**；其 B2 机制成为本矩阵
   S2、B1 机制成为本矩阵 S3，公式、默认参数与 probe ledger 逐字继承（见下文 S 轴语义），且在
   本矩阵内额外获得 T 轴交叉（Wave B 原设计把 `action_rate_l2=-0.1` 固定为唯一 T 档）。
3. **两个旧预注册应标 `superseded`，理由如下并链接本文档：** 二者的科学位均被本矩阵在
   同 parent、同机制、更完整的 T×S 交叉下覆盖；继续单独发射只会花掉 GPU 却回答本矩阵的
   真子集问题。Wave A 的 probe 代次历史（probe2–probe10）、probe9 receipt 与全部
   infrastructure 失败记录，以及 Wave B 的 E1 源码/单测与 M0 输入门 0/4 拒绝，全部保持
   immutable 且被本文引用；`superseded` 只指“科学发射位”，不改写任何历史证据。对旧文档的
   状态行编辑由主控在合入时执行，本文件不改动它们。

## S 轴语义（四档互斥）

四格共用规则：每格必须**显式写出全部三个机制 weight**（`post_swing_settle_debt`、
`lower_body_stability_bundle`、`lower_body_pose_imitation`），S1/S2/S3 互斥开启；只给一个 weight、
两个机制同时非零、bool/NaN/Inf/错符号都必须在修改配置前 fail closed。所有格（含 S0）都开启
weight-independent 测量 probe ledger，保证 S0 与 treatment 记同一套账；机器实现见 queue config
`probe_contract` 段（S2/S3 由显式 weight 键强制 probe=1，S1 用显式 CLI probe 键且发射前须
grep 复核该键已进 trainer 白名单）。

- **S0＝纯对照。** 三个新机制 weight 全为 `0`，probe 全开。它同时是 S 轴对照和 Wave A 科学位
  的替代载体。
- **S1＝`post_swing_settle_debt`（挥拍后安顿债务包，weight `-0.25`）——本轮唯一新实现。**
  人话：挥拍结束后机器人还欠多少“没安顿下来”的账——把挥拍后恢复窗内仍未收敛的量拆成多笔
  债务分别计量、有界求和后按 `-0.25` 收费。机制思想来自 Jiayi HitterV11 V13 分支的 post-swing
  五债务组合（来源与审计结论见下节）；分量定义与全部数值**不抄分支**，按本仓惯例重定：有界
  `[0,1)` 尾部核、逐关节按实际速度上限归一化、同一拍恢复窗相位门、reset 后首步强制清零、
  finite/顺序校验 fail closed。exact 公式与冻结数值的机器真源是本轮 queue config 与
  `hope_rewards.py` 内的实现及其单测；若实现落盘后的语义与本段人话不符，必须先修订本记录再
  允许渲染任何命令。
- **S2＝`lower_body_stability_bundle`（无参考支撑/腿速包，weight `-0.25`）——逐字继承 Wave B B2。**
  人话：不看任何示教动作，只罚两件事——双脚横向支撑宽度塌到 `0.22 m` 下界以下（尺度
  `0.05 m`），以及 12 个腿关节实际速度超过 `1.0 rad/s` 免罚区后的尾部（尺度 `0.5 rad/s`）；
  相位门为挥拍前 `0.30 s` 至挥拍后 `0.40 s` 的支撑窗。五个机制参数全取 Wave B 冻结默认。
- **S3＝`lower_body_pose_imitation`（静态腿姿软模仿，weight `+0.5`）——逐字继承 Wave B B1。**
  人话：只在同一支撑窗（pre `0.30 s`/post `0.40 s`）用很软的高斯核（`std=0.35 rad`）奖励 12 个
  腿关节贴近当前静态 v4rg 参考姿态；正奖励软先验，不是硬姿态锁。只对静态 v4rg teacher 有
  发射资格；混入未过门的移动老师则整格失格。

### S1 机制来源与审计澄清

2026-07-20 对 Jiayi HitterV11 V13 分支的 exact-commit 只读审计结论是：post-swing 五债务的**机制
思想 Adopt / Rerun on main，分支禁止整体 merge**——该分支没有 checkpoint/行为证据，且历史含大量
无关/破坏性改动（另见 [NOW](../../NOW.md) 支线审计段）。必须写清的澄清：分支材料所称
“训练了 2 万 steps”实为 2 万 **iterations**（PPO updates），且其自报曲线到 `19999` iterations 时
相关量仍越限；因此 V13 既没有可比行为成绩，也没有证明其原始数值配比有效。本轮只继承
“挥拍后按多笔安顿债务分别计量”的思想，数值按本仓有界尾部/归一化惯例重定，并以本矩阵的
S1 对 S0 单变量对照重新买证据。

审计中同时评估过 PACE 式 torso/root 姿态子集惩罚，**本轮不为它设独立臂**：baseline 配方已含
`upright`、`base_ang_vel_xy`、`base_lin_vel_z` 同类项，再开一臂只会与现役项混杂、无法单变量归因；
若未来要动这组项，应对现役权重做 matched 消融，而不是引入换名的重复机制。

## 冻结的 setting

| 字段 | 冻结值 |
| --- | --- |
| 队列/渲染器 | [`configs/phase1_balance_temporal_matrix_20260720.yaml`](../../../configs/phase1_balance_temporal_matrix_20260720.yaml)、[`scripts/run_phase1_balance_temporal_matrix_queue.py`](../../../scripts/run_phase1_balance_temporal_matrix_queue.py)。本轮是 lean runner：不设 manifest-SHA 多层审批链，改用统一计划表（`--plan`）+ 依赖核对单（`--checklist`）+ 逐字渲染人工核对（`--render-stage`）；renderer 自身绝不 SSH、绝不发信号、绝不写远端 |
| parent SHA 钉死 | W checkpoint `2caab3dd...fcce`、V checkpoint `ad901910...2716` 直接写死在 renderer 内；发射前须远端 `sha256sum` 逐一比对 |
| source | 远端必须是 clean、detached 的 exact commit。config 当前为占位 `PENDING_EXACT_COMMIT`，渲染在占位符下被拒绝；主控合并全部并行实现后填入 40-hex，不得以启动时任意 HEAD 代替 |
| 动作 | `v4rg_runtime_order_v3` 正/反手（S3 只绑定静态 v4rg teacher） |
| 观测/action | `deploy_parity_face179`，actor `179→31`；`qdes_clamp=true`（日志必须出现 `q_des CLAMP ACTIVE`） |
| plant/engine | Isaac `HOPEPingPongVirtualBall`，`dt=0.005`、decimation `4`（50 Hz）；零关节摩擦 |
| parent/seed | `W@model_6700` 与 `V@model_6700`（路径与 hard contract 沿用 [Wave A queue config](../../../configs/phase1_balance_action_slew_20260720.yaml)）；全部 24 格 `seed=3`，完整恢复 policy/value/optimizer/normalizer |
| PPO rollout | [`algo.runner.num_steps_per_env=24`](../../DEFINITIONS.md#ppo-num-steps-per-env)；probe 每 update observed 必须精确为 `4096×24=98304` |
| 课程/题库 | short-focus `0.10/0.45/0.40/0.05`；同一 schema-3 train bank |
| 其他 Reward | 每个 parent 内保持原击球/准备配方；[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge)＝`0`；post-swing replay＝`0` |
| 谱系 | [`checkpoint_allow_contract_mismatch=true`](../../DEFINITIONS.md#checkpoint-contract-mismatch)；W/V contract 有意 mismatch，所有后代只作诊断、永久 formal-exact-ineligible |
| 预算 | 每格续训 `10001` updates（absolute 至 `model_16700`），save 间隔 `100` |

## 24 格表与唯一 run names

run name 模式为 `p1btm_{w|v}_{n|c|h}_{s0..3}_seed3_20260720`。每行三个机制 weight 都显式写；
T 轴两列（raw＝`action_rate_l2`，slew＝`processed_qdes_slew_hinge_weight`）同样显式写，避免
“默认值”冒充单变量。槽位按 queue config 写死：按 `(S,T,parent)` 排序轮流发到 Pod1
GPU0/1/2、Pod2 GPU0/1/2，S0 六格最先、S1/S2/S3 依次成圈；同一 `(parent,T)` 的四个 S 档共卡，
每卡恰好 4 条（Pod1 十二格 long-grid 已有同密度先例）。发射服从单一队列纪律：空槽拉最前
就绪项；实际落位以每格 claim/binding 实录为准。

| # | 人话 — `run_name` | raw | slew | S1 debt | S2 bundle | S3 pose | 名义槽 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | W 无平滑·纯对照 — `p1btm_w_n_s0_seed3_20260720` | `0` | `0` | `0` | `0` | `0` | Pod1 GPU0 |
| 2 | W 现役全身平滑·纯对照 — `p1btm_w_c_s0_seed3_20260720` | `-0.10` | `0` | `0` | `0` | `0` | Pod1 GPU2 |
| 3 | W 恢复期腿腰铰链·纯对照 — `p1btm_w_h_s0_seed3_20260720` | `0` | `-0.25` | `0` | `0` | `0` | Pod2 GPU1 |
| 4 | V 无平滑·纯对照 — `p1btm_v_n_s0_seed3_20260720` | `0` | `0` | `0` | `0` | `0` | Pod1 GPU1 |
| 5 | V 现役全身平滑·纯对照 — `p1btm_v_c_s0_seed3_20260720` | `-0.10` | `0` | `0` | `0` | `0` | Pod2 GPU0 |
| 6 | V 恢复期腿腰铰链·纯对照 — `p1btm_v_h_s0_seed3_20260720` | `0` | `-0.25` | `0` | `0` | `0` | Pod2 GPU2 |
| 7 | W 无平滑·安顿债务 — `p1btm_w_n_s1_seed3_20260720` | `0` | `0` | `-0.25` | `0` | `0` | Pod1 GPU0 |
| 8 | W 现役全身平滑·安顿债务 — `p1btm_w_c_s1_seed3_20260720` | `-0.10` | `0` | `-0.25` | `0` | `0` | Pod1 GPU2 |
| 9 | W 恢复期腿腰铰链·安顿债务 — `p1btm_w_h_s1_seed3_20260720` | `0` | `-0.25` | `-0.25` | `0` | `0` | Pod2 GPU1 |
| 10 | V 无平滑·安顿债务 — `p1btm_v_n_s1_seed3_20260720` | `0` | `0` | `-0.25` | `0` | `0` | Pod1 GPU1 |
| 11 | V 现役全身平滑·安顿债务 — `p1btm_v_c_s1_seed3_20260720` | `-0.10` | `0` | `-0.25` | `0` | `0` | Pod2 GPU0 |
| 12 | V 恢复期腿腰铰链·安顿债务 — `p1btm_v_h_s1_seed3_20260720` | `0` | `-0.25` | `-0.25` | `0` | `0` | Pod2 GPU2 |
| 13 | W 无平滑·支撑腿速包 — `p1btm_w_n_s2_seed3_20260720` | `0` | `0` | `0` | `-0.25` | `0` | Pod1 GPU0 |
| 14 | W 现役全身平滑·支撑腿速包 — `p1btm_w_c_s2_seed3_20260720` | `-0.10` | `0` | `0` | `-0.25` | `0` | Pod1 GPU2 |
| 15 | W 恢复期腿腰铰链·支撑腿速包 — `p1btm_w_h_s2_seed3_20260720` | `0` | `-0.25` | `0` | `-0.25` | `0` | Pod2 GPU1 |
| 16 | V 无平滑·支撑腿速包 — `p1btm_v_n_s2_seed3_20260720` | `0` | `0` | `0` | `-0.25` | `0` | Pod1 GPU1 |
| 17 | V 现役全身平滑·支撑腿速包 — `p1btm_v_c_s2_seed3_20260720` | `-0.10` | `0` | `0` | `-0.25` | `0` | Pod2 GPU0 |
| 18 | V 恢复期腿腰铰链·支撑腿速包 — `p1btm_v_h_s2_seed3_20260720` | `0` | `-0.25` | `0` | `-0.25` | `0` | Pod2 GPU2 |
| 19 | W 无平滑·腿姿软模仿 — `p1btm_w_n_s3_seed3_20260720` | `0` | `0` | `0` | `0` | `+0.5` | Pod1 GPU0 |
| 20 | W 现役全身平滑·腿姿软模仿 — `p1btm_w_c_s3_seed3_20260720` | `-0.10` | `0` | `0` | `0` | `+0.5` | Pod1 GPU2 |
| 21 | W 恢复期腿腰铰链·腿姿软模仿 — `p1btm_w_h_s3_seed3_20260720` | `0` | `-0.25` | `0` | `0` | `+0.5` | Pod2 GPU1 |
| 22 | V 无平滑·腿姿软模仿 — `p1btm_v_n_s3_seed3_20260720` | `0` | `0` | `0` | `0` | `+0.5` | Pod1 GPU1 |
| 23 | V 现役全身平滑·腿姿软模仿 — `p1btm_v_c_s3_seed3_20260720` | `-0.10` | `0` | `0` | `0` | `+0.5` | Pod2 GPU0 |
| 24 | V 恢复期腿腰铰链·腿姿软模仿 — `p1btm_v_h_s3_seed3_20260720` | `0` | `-0.25` | `0` | `0` | `+0.5` | Pod2 GPU2 |

`H` 格的 slew 参数固定 margin `0.85`、恢复窗触球后 `0.20–1.55 s`。`W` 与 `V` 的不同 parent 配方不
构成 W-vs-V 单变量比较；一切机制比较都在各自 parent 内进行。

## 预算与观察点

1. **每格先跑 `4096 env × 24 step/env × 2 update` 的 [full-scene probe](../../DEFINITIONS.md#full-scene-probe)**
   （独立 `probes/` namespace、run name 前缀 `p1btm_probe_`），自然退出到 `model_6701.pt`。
   本轮为 lean 逐格解锁：某格 probe 自然退出并通过人工核对（`Learning iteration` 出现、无
   Traceback/fatal、`model_6701.pt` 存在、`q_des CLAMP ACTIVE` 在日志中、三机制 probe ledger
   tag 实际落盘）后，才允许渲染**该格**的科学长训命令；完整 update 的 ledger observed 期望为
   `4096×24=98304`，恢复窗资格分母允许单 update 为零但两步合计应非零，S 档 enabled 计数须与
   该格 weight 一一对应，异常即停发该格并排查。
2. **科学长训每格从 `model_6700` 续 `10001` updates、save/100**。观察点为相对 parent 的 offsets
   `+200/+500/+1000`（absolute `6900/7200/7700`）——只判崩溃、non-finite、lineage/contract 与机制
   接线，不判胜负；`+2000/+4000`（absolute `8700/10700`）——看中段方向；`+6000/+10000`
   （absolute `12700/16700`）——形成完整单 seed 结论。
3. **不因稀疏 reward 早期为零而停。** 需要真实击球机会才有的 sparse 通道样本不足时必须继续训练
   （见 [sparse 资格账本](../../DEFINITIONS.md#sparse-reward-eligibility-ledger)）。
4. **单 seed 3 只做 screening。** 固定窗上领先且机制不同的 `2–4` 格另立预注册补 seed；本 queue 的
   descendants 永远不能因追加 seed 变成 formal exact。

每个观察点固定使用截至该 absolute milestone 的**最后 100 个完整 PPO update**（如 `16700` 用
`16601..16700`），先对 exact ledger counter 求和再做除法，不得插值、改累计窗或事后挑点；
completion/fall/legal-return、q_des tail、ready tilt、qdot 的分子分母口径逐字沿用
[Wave A 量尺](EXP-P1-BALANCE-ACTION-SLEW-20260720.md#预算量尺与停止规则)，S2/S3 的
narrow-rate/stance-tail/qdot-tail/pose-error 口径逐字沿用
[Wave B 量尺](EXP-P1-LOWER-BODY-STABILITY-20260720.md#行为指标与裁决)；S1 的 ledger 口径以其
实现单测冻结的 counter 名为准，判读同样是“先求和再除”。

## 停止规则

只有下列三类情况允许停格；停格必须记录原因与最后 checkpoint，其余一律跑满预算：

1. **NaN/OOM 不可恢复**：训练标量或 checkpoint 出现 non-finite，或重复 OOM 无法在同配方下继续；
2. **lineage 错误**：claim/binding/hard contract 与预注册不符（如 parent 错、weight 漂移、
   `training_contract_lineage_exact` 意外为 `1`）；
3. **被同 parent 同 T 档明显支配**：同 parent 同 T 档内存在另一格，其 fall 与击球（completion 与
   legal return）**全部**严格更优，且该支配已持续 `≥4000` iterations。

不允许因“早期难看”“稀疏 reward 为零”或跨 parent/跨 T 档比较而停格。停止执行只能读取该格
launcher sidecar、重验 numeric PID/PGID/starttime/argv 后按
[RunPod 纪律](../../operations/run_on_runpod.md#已登记-phase-1-实验臂的算力释放)对 exact PGID 处置；
禁止 `pkill -f`/`killall`。队列无自动 stop/retry/promote 权限。

## 判卷合同

1. **Isaac 内先行指标**：fall（`physical_fall_count/swing_outcome_count`）、ready tilt、
   processed-q_des tail、S 轴各自 probe ledger——全部按固定 100-update 窗先求和再除。
2. **终点同卷**：`+10000` 终档用 `judge.sh` + signed-face composite 的
   [K100 同卷](../../DEFINITIONS.md#q50-and-k100)（100 道固定题、正反手各 50）对领先格与其
   matched S0/C 对照判卷；同卷、同 scorer、同 checkpoint 家族，不得混卷。
3. **谱系限制**：全部 24 格因 W/V contract 有意 mismatch 而永久 formal-exact-ineligible，Isaac 与
   K100 结果都只是 **diagnostic**。胜者机制要正名，必须在 exact-lineage 链上重跑——当前唯一
   exact P0 候选是 qdot treatment `model_1000` 一族（fresh lineage=`1`、schema-3 hard contract
   exact，见 [NOW](../../NOW.md)）；即把胜出的 T×S 配方作为单变量 continuation 或 fresh 预注册
   接到该 exact 家族上，而不是给本轮后代补 seed。
4. 最终行为裁判仍是 vendor MuJoCo 同卷；本轮 Isaac 结果不得直接宣称部署或真机资格。

## 基础设施修正记录

Wave A 的 v8 科学首格与 v9/probe10 尝试都死于 locked launcher 的 `180 s` stale_timeout：4096-env
scene 在 `sim.reset` 前后长时间无日志推进，watchdog 按 exact process-group 合同 TERM/KILL（v8
attempt1 terminal kind=`stale_timeout`、exit=`125`）。`180 s` 对该 boot 阶段过紧，把冷启动杀成
infrastructure 失败。本轮修正并冻结（machine truth 是 queue config `watchdog` 段）：

- **不再使用 locked launcher**：远端改用 `/workspace/bin/kit_boot_lock.sh` + `setsid nohup`
  起进程，boot 由 lock 串行化，收口靠事后 artifact 验证（run.log/checkpoint/launcher.out）；
- **boot 容忍 `1800 s`**：首个 `Learning iteration` 出现之前允许最长 `1800 s` 无日志推进；
- **iteration 后 `900 s`**：首个 iteration 之后 stale 容忍收紧为 `900 s`；
- **每格一次逐字 retry（`_r2`）**：任一格因基础设施失败（stale/SIGABRT/身份竞态等，非 NaN/OOM
  科学失败）收口后，允许在 fresh no-clobber namespace 下用逐字同配方、run name 加 `_r2` 后缀
  重发一次；同 phase 第二次失败不再重试，转根因线。原 namespace 永久只读。

watchdog 仍只对已绑定的 exact PGID 发信号；超时放宽不改变 fail-closed 语义。

## 风险与已知边界

- **S1∩S2 共享腿速分量**：S1 债务包与 S2 bundle 都对腿关节速度尾部收费（免罚区/尺度/窗口
  不同）。两者互斥开启，所以矩阵内无叠加格；但若两者同时胜出，不能把二者收益相加解读，
  须另开正交化消融。此重叠在此显式披露。
- **W/V 不跨 parent 比较**：两 parent 配方不同，绝对值不可横比；矩阵一切结论按 parent 分层。
- **本轮不含移动老师**：[M0](../../DEFINITIONS.md#motion-m0) 四条横移动作 stance 门 `0/4` 未过，
  moving-teacher 输入维持 reject/no-launch；S3 只绑定静态 v4rg。
- **T 档与卡位重合**：槽位把同一 `(parent,T)` 的四个 S 档钉在同一张卡上，因此 parent 内的
  T 轴比较必然跨卡/跨 Pod。Wave A probe9 crossover 只排除“某格在某卡必然失败”，不证明 GPU
  等价；此混杂在此披露，claim/binding 记录的实际落位必须留档。
- **诊断谱系**：全部后代 formal-exact-ineligible；任何“采用”都要走 exact-lineage 重跑，见判卷合同。
- **S1 实现落盘晚于本预注册**：S1 exact 公式以实现+单测为机器真源，语义与本文不符时必须先
  修订本记录再渲染命令（见 S 轴语义）。

## 运行表

| 运行 | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 24 格（上表 #1–#24） | `preregistered / not launched` | W/V `model_6700` / seed3 | E1 source | 无 | 占位 commit 解锁、`--checklist` 全过、`origin/main` 认领齐全前不得执行任何 SSH 命令；probe 未过的格不得发 science |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

“物理不摔”将使用 `physical_fall_count`（all-attempt 分母）。本矩阵是 single-swing continuation
诊断，不声称 T0/T1/T2 连续恢复；即使 Isaac fall 改善，仍须独立 vendor MuJoCo/Gate3 行为卷。

## 决定

- 决定：`pending`（preregistered，未发射）
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：queue config/renderer 与 S1 实现合入最新 `origin/main`，把占位
  `source.commit` 换成冻结 40-hex；`origin/main:docs/NOW.md` 完成 owner/executor/branch/queue id
  认领；`--checklist` 全过、逐格 probe 核对通过后才发对应格科学长训；固定窗判读后领先且机制
  不同的 2–4 格另立预注册补 seed，胜者机制到 exact-lineage 链正名。

## 复现与证据

从零到发射的操作步骤、错峰纪律、监控与收口清单见
[run_phase1_balance_temporal_matrix](../../operations/run_phase1_balance_temporal_matrix.md)。
本记录当前没有任何 SSH、claim、checkpoint 或行为结果；judge、第二 seed、部署与真机均未授权。

## 2026-07-20 g1 探针失败与修复（发射前）

第一格 probe `w_n_s0`（Pod1 GPU0，原 root `/workspace/codexschema/phase1_balance_temporal_matrix_20260720`）在
build hard contract 阶段 fail-closed：`lower-body reward contracts require the exact 31-joint A3 runtime order`。
根因：Wave B 合同/reward 把 31 关节的**部署顺序整条硬编码**为相等条件，而真实 Isaac articulation 是
广度优先顺序（`left_hip_pitch, right_hip_pitch, waist_yaw, …`，以 W 父本 hard contract 为证）。修复：改为
与已过 probe9 真机验证的 `processed_qdes_slew_hinge` 同款纪律——只要求 31 唯一名字集合与 articulation
一致，目标关节**按名字**选取；同步修正把"换序必须失败"当作合同的旧单测，并新增真实 BFS 顺序回归测试。
原 root 连同该失败 probe 按不可覆盖惯例封存，只作历史；全部 24 格转到新 root
`/workspace/codexschema/phase1_balance_temporal_matrix_20260720d`，source.commit 同步更新。该失败不是任何机制的负例。

### g2 第二层拒绝（同日）

修复 reward/contract 层后，g2 probe `w_n_s0` 在 schema-3 结构校验被第二层拒绝：合同里 12 腿关节
列表按运行时枚举顺序写入，而校验器要求规范部署顺序。修复：contract builder 一律输出规范部署顺序
的腿关节列表（`joint_order` 字面量改为 `canonical_deploy_order_selected_by_name`，pose/bundle 两块
builder 与 validator 同步；probe9 已验证的 slew hinge 合同保持原字面量不动）。root 再次换代到
`/workspace/codexschema/phase1_balance_temporal_matrix_20260720d`；两次失败都发生在任何训练更新之前，
不构成机制证据。
