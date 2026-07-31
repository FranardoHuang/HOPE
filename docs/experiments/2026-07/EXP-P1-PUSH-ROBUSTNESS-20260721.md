# EXP-P1-PUSH-ROBUSTNESS-20260721 — 训练时随机推撞是否是平衡的希望（18 臂计划，14 臂实发）

- 状态：`superseded`
- 收口：`incomplete`；决定：`superseded; no dose winner`。新一代 ActionBall 只继承“应有
  push”的方向，不继承本波未完成的剂量排名。
- 终档审计（2026-07-31）：计划 18 臂；实际发射并产生 checkpoint 的是
  `{W,V} × {p02,p035,p05,yaw,p08,f035,f08}` 共 14 臂，`{W,V} × {ang,fast}` 四臂从未发射。
  所有实发臂均有 `model_8700`，但最高仅到 9200–13900，无一到达预注册
  `model_16700`。只找到 `w_p02` 一份逐臂 full-scene probe；其余 13 臂跳过 admission，
  续训还跨 `f67c844f → 08ca8e83 → ca078850/4624c824` 多个训练 source。
- 阶段/轴：Phase 1 / 平衡鲁棒性：训练时外部推撞（push）单轴消融（幅度 × 角速度 × 频率）
- 集成小目标：给"从未被推过"的现役配方补上文献标配的训练时随机推撞，回答多大/什么成分/
  多频繁的 push 能降摔而不伤击球
- 人类负责人：Franco（拍板：push 是平衡的希望，本波最高优先）
- 执行者：Claude
- 复核/决策负责人：Franco
- 最高证据等级：`E2`（`w_p02` 为 runtime mechanics 证据；14 臂 checkpoint/日志和
  5 份 non-exact judge 只作历史 directional evidence，不能选出 dose winner）。
- 创建日期/最后复核日期：2026-07-20 / 2026-07-31

人话导言：这波已证明 push 事件能挂载，但执行纪律和终点不足以裁决剂量。
原计划以两个 `model_6700` 父本
[`W/V`](../../DEFINITIONS.md#balance-action-slew-matrix)（`W`＝拍心优先×自由非击球臂的稳定
父本，`V`＝拍速优先×强准备的高摔倒父本）交叉幅值、方向、频率和力推变体，
其余配方与当时的 24 格矩阵
[`w_c_s0`/`v_c_s0`](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md)（现役全身平滑
[`action_rate_l2`](../../DEFINITIONS.md#raw-action-rate-l2)＝`-0.10`、
[`processed_qdes_slew_hinge`](../../DEFINITIONS.md#processed-qdes-slew-hinge)＝`0`、三个稳定
机制 weight 全 `0`、测量 probe 全开）**逐字相同**——所以那两格就是本波的无 push 对照，
不再买新对照。每臂 `4096 environments`、`seed=3`、从各自 parent `model_6700` 续训 `10001`
updates。

## 动机（为什么现在补 push）

1. **Franco 拍板：push 是平衡的希望，最高优先。** 平衡问题此前两波（Wave A 时序平滑、
   Wave B 下肢机制，均已并入 24 格矩阵）都在改"怎么罚"，本波改"环境里有没有扰动"——
   这是文献里更标准、也更被反复验证的降摔手段。
2. **现役 `push_robot=None` 是对齐 HITTER 的历史决定，不是实验结论。**
   代码真源：`hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py:452-455`
   ——注释原文明说"HITTER alignment: no external push"，据此把基类的 interval push 关掉。
   该决定从未被单变量对照检验过，本波推翻它、买证据。
3. **对标的三篇论文全部带训练时 push（本地已审计）：**
   - **PACE**：`t1_tt_config` push 事件——每 `5–15 s` 一次、`vx,vy ∈ ±0.2 m/s`
     （本仓 PACE 代码审计笔记见
     [pace.md](../../research/ball_physics_2026-07-03/pace.md)，该页此前只审 reward，push
     数值为本轮补充审计的同源 config）；
   - **BeyondMimic**：上游 `tracking_env_cfg.py:159-200` push 每 `1–3 s` 一次、
     `vx,vy ∈ ±0.5 m/s`、`roll/pitch ∈ ±0.52`、`yaw ∈ ±0.78 rad/s`。本仓 vendored 基类同款
     可当场验证：`tasks/tracking/tracking_env_cfg.py:30-36`（`VELOCITY_RANGE`）与
     `:195-200`（`push_robot` EventTerm，`interval_range_s=(1.0, 3.0)`）——正是被
     `hope_env_cfg.py:455` 关掉的那一项；
   - **SMASH**：p.6 明说训练含外部扰动，但未给数值。
4. **不稳定病像与"从未被推过"一致。** 四个独立初始化（seed 1/2/3/4）同卷回球
   `83/100、100/100、100/100、20/100`（零物理摔倒卷，脆性表现在回球质量全凭初始化运气；
   见 [fresh lineage 四 seed 记录](../../archive/PHASE1_FRESH_LINEAGE_2026-07-11.md#remaining-gates)）；
   矩阵运行态快照里 V 父本全部格子窗口摔倒计数 ~`380–440` 而 W 父本 `0–1`。策略只会一条
   窄走廊、姿态吸引盆没有被扰动拓宽过，与文献里"无 push 训练→脆"的病像一致。

## 可证伪假设（每条都能被同卷数据拒绝）

1. **存在最优幅度（非单调）：** 存在 `a ∈ {0.2, 0.35, 0.5} m/s` 使该幅度臂同时过两道门：
   V 父本终点 K100 摔倒数低于 `v_c_s0` 且回球不低于 `v_c_s0 − 3` 题；W 父本回球不低于
   `w_c_s0 − 3` 题且摔倒不增。若三个幅度全过不了门→拒绝"本剂量族的 push 有效"；若收益
   随幅度单调升到 `0.5` 仍未回落→拒绝"最优在带内"的非单调子假设（下波向更大幅度探）。
2. **角速度扰动有额外收益：** `yaw`/`ang` 相对同幅度同间隔的纯线速度臂 `p035`（同 parent
   同卷），摔倒或回球至少一项严格更优、且无一项劣化超过 `3` 题。否则拒绝。
3. **高频小间隔是否优于低频：** `fast`（`1–3 s`、`±0.35`）相对 `p035`（`5–15 s`、`±0.35`）
   同 parent 同卷比较，判据同上。更优→采高频；更差→拒绝并保留低频。

physical-fall 硬失败不能由更高 reward、composite 或更平滑的 action 抵消。

## 冻结的 setting

| 字段 | 冻结值 |
| --- | --- |
| 队列/渲染器 | [`configs/phase1_push_robustness_20260721.yaml`](../../../configs/phase1_push_robustness_20260721.yaml)、[`scripts/run_phase1_push_robustness_queue.py`](../../../scripts/run_phase1_push_robustness_queue.py)（并行实现已落盘、待合入 main；lean runner 纪律与[矩阵](../../../configs/phase1_balance_temporal_matrix_20260720.yaml)同款：`--plan` 统一计划表＋`--checklist` 依赖核对单＋`--render-stage probe\|science --render-job <id> --pod <pod> --gpu <n>` 逐字渲染人工核对——本波不写死 pod/gpu，认领的空槽在渲染时注入；renderer 绝不 SSH/发信号/写远端；机器真源以合入 main 的 config 为准） |
| push 机制 | 基类 `push_robot` EventTerm 同款机制（Isaac Lab `push_by_setting_velocity`，interval 模式）：每个 env 独立按 `interval_range_s` 内均匀采样的间隔触发一次，把 `velocity_range` 内均匀采样的速度增量叠加到 root 速度上；**全程可触发，不做恢复窗/击球窗门控**（与 PACE/BeyondMimic 同款，混杂披露见下）。exact 语义以实现+单测为机器真源，与本段人话不符时必须先修订本记录再渲染命令 |
| push 覆盖键 | 统一键面（2026-07-20 已对齐，train.py `_PUSH_KEYS`、queue yaml、renderer 单测三方逐字交叉断言）：五个 `task.push.*` 键，每臂**全部显式写**：`++task.push.enable=true`、`interval_range_s=[lo,hi]`、`vel_xy_mps=<a>`（x/y 对称 `±a` 由 contract 单源展开，z 永不推）、`ang_vel_radps=<w>`、`ang_axes=none\|yaw\|rpy`（未开角速度必须 `none`+`0.0`，错配 boot 即 fail-loud）。缺键/多键/非法值全部 fail closed |
| 无 push 对照 | 矩阵在跑的 `p1btm_w_c_s0_seed3_20260720`、`p1btm_v_c_s0_seed3_20260720`（同配方、不写任何 push 键、现役默认 `push_robot=None`）；**不再买新对照** |
| 其余配方 | 与矩阵 `C+S0` 格逐字相同：base_overrides、planner revision、各 parent recipe_overrides、`action_rate_weight=-0.1`、slew hinge `0`（含全部 margin/窗参数显式写）、三个稳定机制 weight `0`＋全参数显式写、probe 全开（weight 键显式即强制 probe=1，见矩阵 `probe_contract`） |
| parent SHA 钉死 | W checkpoint `2caab3dd...fcce`、V checkpoint `ad901910...2716`（与矩阵同两份 `model_6700`）；发射前远端 `sha256sum` 逐一比对 |
| source | 远端 clean、detached exact commit；config 出厂占位 `PENDING_EXACT_COMMIT`，渲染在占位符下被拒绝；push 键实现合入 main 后才填 40-hex |
| 动作/观测/plant | `v4rg_runtime_order_v3` 正反手；`deploy_parity_face179`、`qdes_clamp=true`；Isaac `HOPEPingPongVirtualBall`，`dt=0.005`、decimation `4`（50 Hz）、零关节摩擦——全部与矩阵同 |
| parent/seed | `W@model_6700` 与 `V@model_6700`，全部 12 臂 `seed=3`，完整恢复 policy/value/optimizer/normalizer |
| PPO rollout | [`algo.runner.num_steps_per_env=24`](../../DEFINITIONS.md#ppo-num-steps-per-env)；probe 每 update observed 精确 `4096×24=98304` |
| 谱系 | [`checkpoint_allow_contract_mismatch=true`](../../DEFINITIONS.md#checkpoint-contract-mismatch)；W/V contract 有意 mismatch，所有后代只作诊断、永久 formal-exact-ineligible |
| 预算 | 每臂续训 `10001` updates（absolute 至 `model_16700`）、save/100；观察点 offsets `+200/+500/+1000/+2000/+4000/+6000/+10000`（absolute `6900/7200/7700/8700/10700/12700/16700`）——与矩阵逐字相同 |

## 12 臂表与唯一 run names

run name 模式为 `p1push_{w|v}_{p02|p035|p05|yaw|ang|fast}_seed3_20260721`。间隔为均匀采样
`uniform[a,b] s`；线速度/角速度列为对称 `±` 幅度；`vz` 全臂显式冻结 `[0.0,0.0]`（config 里
不省略）。六种变体的人话：`p02/p035/p05`＝低频（5–15 s）纯水平线推的三档幅度阶梯；`yaw`＝
中档线推加偏航角速度扭；`ang`＝中档线推加三轴角速度扭；`fast`＝中档线推提频到 1–3 s。

| # | 人话 — `run_name` | interval s | vxy m/s | roll rad/s | pitch rad/s | yaw rad/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | W 小幅低频线推 — `p1push_w_p02_seed3_20260721` | `5–15` | `0.2` | `0` | `0` | `0` |
| 2 | V 中幅低频线推 — `p1push_v_p035_seed3_20260721` | `5–15` | `0.35` | `0` | `0` | `0` |
| 3 | W 大幅低频线推 — `p1push_w_p05_seed3_20260721` | `5–15` | `0.5` | `0` | `0` | `0` |
| 4 | V 中幅低频线推＋偏航扭 — `p1push_v_yaw_seed3_20260721` | `5–15` | `0.35` | `0` | `0` | `0.5` |
| 5 | W 中幅低频线推＋三轴扭 — `p1push_w_ang_seed3_20260721` | `5–15` | `0.35` | `0.5` | `0.5` | `0.5` |
| 6 | V 中幅高频线推 — `p1push_v_fast_seed3_20260721` | `1–3` | `0.35` | `0` | `0` | `0` |
| 7 | W 中幅低频线推 — `p1push_w_p035_seed3_20260721` | `5–15` | `0.35` | `0` | `0` | `0` |
| 8 | V 小幅低频线推 — `p1push_v_p02_seed3_20260721` | `5–15` | `0.2` | `0` | `0` | `0` |
| 9 | W 中幅低频线推＋偏航扭 — `p1push_w_yaw_seed3_20260721` | `5–15` | `0.35` | `0` | `0` | `0.5` |
| 10 | V 大幅低频线推 — `p1push_v_p05_seed3_20260721` | `5–15` | `0.5` | `0` | `0` | `0` |
| 11 | W 中幅高频线推 — `p1push_w_fast_seed3_20260721` | `1–3` | `0.35` | `0` | `0` | `0` |
| 12 | V 中幅低频线推＋三轴扭 — `p1push_v_ang_seed3_20260721` | `5–15` | `0.35` | `0.5` | `0.5` | `0.5` |
| — | V 无 push 对照（矩阵在跑，不发射） — `p1btm_v_c_s0_seed3_20260720` | 无 push | `0` | `0` | `0` | `0` |
| — | W 无 push 对照（矩阵在跑，不发射） — `p1btm_w_c_s0_seed3_20260720` | 无 push | `0` | `0` | `0` | `0` |

表序即 config `launch_order` 冻结的单一队列顺序：父本与档位交错，前 6 臂就把全部 6 档各买
一次（W/V 各 3 档），尽早锚定假设 1–3 的比较轴。每臂的完整 override＝矩阵 `C+S0` 全套
（逐字照抄，config 的 `recipe` 两段有单测交叉断言防漂移）＋该臂八个 `task.push.*` 键显式值；
机器真源是 queue config，其 12 条 job 的 override 与本表不符时必须先修订本记录再渲染命令。
`W` 与 `V` 配方不同，一切比较都在各自 parent 内进行。

## F 轴：力推 vs 速度推（同冲量）（2026-07-21 追加，Franco 拍板）

本节把队列从 14 臂扩到 **18 臂**：新增 2 个力推档 `f035/f08` × 2 父本＝4 条 F 臂
（`{w,v}_{f035,f08}`），`launch_order` 排在全部 14 条速度臂之后
（`w_f035 → v_f08 → w_f08 → v_f035`）。机器真源仍是 queue config。

### 动机（Franco：力推与速度推都要，且同冲量可比）

稳站姿**能抗力但抗不了速度注入**：速度推（基类 `push_by_setting_velocity`）把 Δv
一步写进 root 速度，支撑腿刚度再大也拦不住，练的是"被打飞后回正"；力推是持续
`0.30 s` 的推挤，可以靠站姿刚度与踝/髋力矩当场顶住，练的是"抵抗"。两种扰动的最优
响应不同，文献标配只买了速度推——F 轴是"能不能只靠抵抗活下来"的独立探针。**同冲量
配对才可比**：不配平冲量，"力推更容易/更难"只是剂量差，不是机制差。

### 机制（冻结）

- **间隔触发的持续水平力**：每 `interval_range_s`（默认 `5–15 s`）触发一次，对
  pelvis/base 施加水平随机方向、幅度 `force_n` 的**恒力**，持续 `duration_s`
  （`0.30 s` = 15 个 50 Hz 控制步）后清零。
- **键面**：`task.force_push.{enable,interval_range_s,force_n,duration_s}`，默认全
  关，缺席＝逐字节 no-op。**F 臂不写任何 `task.push.*` 键**（单变量，队列单测断言
  双向：速度臂也不含 `task.force_push.*`）。
- **同冲量换算**：`Δv_equiv = force_n × duration_s / m_robot`。质量真源：2026-07-21
  pod1 只读核对 A3 URDF（`assets/agibot_a3/urdf/model.urdf`，sha256
  `79655f05d204c24f028778425aa971410773d1f8bbbd214de6fdb8f8ae75d1cc`）43 个 link 质量
  逐项求和＝**58.27723163 kg**；W 父本 hard contract `training_contract.json` 同日核
  对**无质量字段**，故以 URDF 为准。`force_n` 由主控按真实质量算好写死进 config
  （配置面只有 `force_n` 与 `duration_s`，不搞自动换算魔法）：
  - `f035`：`68.0 N × 0.30 s` → `Δv_equiv 0.35005 m/s`（对表 `p035` 的 `0.35`，偏差 +0.015%）
  - `f08`：`155.4 N × 0.30 s` → `Δv_equiv 0.79997 m/s`（对表 `p08` 的 `0.8`，偏差 −0.004%）
- **运行时合同**：`training_contract` 的 force_push 块必须记录**运行时真实读到的**
  机器人总质量、换算出的 `Δv_equiv` 与 `application_point`（必填），F 臂 probe 收口
  与本节数值对表，对不上停发。

### 施力点诚实标注（Yikang V9 反例）

V9 的教训：文档标"COM 推力"、实现施在 link 原点，判读时把杠杆效应误当机制差。本波
施在 **pelvis link 原点**就诚实写 `pelvis_link_origin`（逐字 =
`training_contract.FORCE_PUSH_APPLICATION_POINT`），**不许标 COM**；合同字段
`application_point` 必填、逐字冻结（config/renderer/单测三方断言，wiring 侧
schema-3 validator 同 token 再验一遍），probe 收口再对表一次。

### 可证伪假设 4（力 vs 速度、同冲量）

同 parent 同卷下，`f035/f08` 臂的 stance/倾角响应（固定窗 fall、ready tilt）优于同
冲量速度臂 `p035/p08`，且回球不低于对应速度臂 `− 3` 题。成立→"抵抗"训练比"回正"
训练便宜，力推可作为更温和的鲁棒性剂量；F 臂两项全面更差→拒绝，说明速度注入练出
的"回正"才是缺失技能，F 轴收档。判据同波：终点 `model_16700` K100 同卷（无 push
考卷）为主判据，固定 100-update 窗 Isaac 先行指标只作方向参考，环境混杂披露、停止
规则、谱系限制逐字沿用本记录前文。

### F 臂渲染闸门（fail-closed）

`task.force_push.*` wiring 由并行 agent 落盘。queue config 的
`force_push_contract.wiring_confirmed_in_source_commit=false` 在主控逐字核对
train.py `_FORCE_PUSH_KEYS == (enable, interval_range_s, force_n, duration_s)`、把
wiring 合入 main 并冻结新 40-hex commit 之前，锁死全部 F 臂渲染（速度臂不受影响）；
本地单测同时断言"wiring 未落盘时闸门必须还关着"。

## 发射策略：空槽即填，不建专属池

矩阵 24 格（现 23/24 在跑）占满两 Pod 六卡、每卡 4 条。本波**不抢占任何在跑矩阵格**，服从
[单一队列纪律](../../operations/run_phase1_push_robustness_wave.md)：矩阵格因三类停止规则
停格、或跑满 `10001` updates 自然完结时释放槽位，空槽即拉本表最前的就绪臂填入；矩阵 `7700`
观察点（预计发射日当晚）是最早可能出现裁剪的时点。当前已知的第一个空槽是 `v_h_s2` 两次
boot 冻结封格后的 Pod2 GPU2 槽位。槽位不预分配 pod/gpu 角色，实际落位以每臂 claim/binding
实录为准；每卡并发上限沿用矩阵修正后的"同卡 compute PID < 4"预检。watchdog 与重试规则
逐字继承矩阵：boot 停滞 `1800 s`、首个 iteration 后停滞 `900 s` 才判死；基础设施失败允许
fresh no-clobber namespace 下逐字重发一次（run name 加 `_r2`），同 phase 二次失败转根因线。

每臂发射前须过本臂 full-scene probe（`4096 env × 24 step × 2 update`，独立 `probes/`
namespace）并逐臂人工核对解锁，纪律与矩阵相同；push 特有核对项见操作页——注意 probe 只有
约 `0.96 s` 仿真时间，`5–15 s` 间隔臂在 probe 内**预期零次触发**，只核对 push 配置回显与
ledger tag 落盘，不得把"probe 内没推过"当失败，也不得当机制证据。

## 预算量尺与判卷合同

1. **Isaac 内先行指标**：fall（`physical_fall_count/swing_outcome_count`，重点看
   post-strike 物理摔倒率）、ready tilt、processed-q_des tail、completion/legal-return——
   全部按固定 100-update 窗（如 `16700` 用 `16601..16700`）先对 exact ledger counter 求和再
   除，不得插值、改窗或事后挑点；口径逐字沿用
   [矩阵量尺](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md#预算与观察点)。**环境混杂披露**：
   push 臂的训练窗指标是在被推环境里测的，对照是在无 push 环境里测的——训练窗跨臂比较只
   作方向参考（"被推着还能不摔/能击球"本身有信息量），不作主判据。
2. **主判据＝终点同卷（无 push 考卷）**：`model_16700` 终档用 `judge.sh` + signed-face
   composite 的 [K100 同卷](../../DEFINITIONS.md#q50-and-k100)（100 题、正反手各 50、考卷内
   无 push），对每臂与其 matched 矩阵对照（`w_c_s0`/`v_c_s0` 同 absolute milestone 终档）
   判卷；同卷、同 scorer，不得混卷。假设 1–3 的门全部在这张卷上裁决。
3. **不因稀疏 reward 早期为零而停**（见
   [sparse 资格账本](../../DEFINITIONS.md#sparse-reward-eligibility-ledger)）；观察点
   `6900/7200/7700` 只判崩溃/合同/接线（含 push 触发计数量级核对：`5–15 s` 间隔、16 s 局长
   下每 env 每局期望 1–3 次），`8700/10700` 看中段，`12700/16700` 才有完整单 seed 结论。
4. **谱系限制**：全部 12 臂因 W/V contract 有意 mismatch 永久 formal-exact-ineligible，Isaac
   与 K100 结果都只是 **diagnostic**。胜者剂量要正名，必须到 exact-lineage 链重跑——当前唯
   一 exact P0 候选是 qdot treatment `model_1000` 一族（见 [NOW](../../NOW.md)）；即把胜出
   push 剂量作为单变量 continuation 或 fresh 预注册接到 exact 家族上，不给本轮后代补 seed。
5. 最终行为裁判仍是 vendor MuJoCo 同卷；本轮 Isaac 结果不得直接宣称部署或真机资格。
6. **单 seed 3 只做 screening**；固定窗与 K100 上领先且剂量不同的 2–3 臂另立预注册补 seed。

## 停止规则（与矩阵相同）

只有下列三类情况允许停臂；停臂必须记录原因与最后 checkpoint，其余一律跑满预算：

1. **NaN/OOM 不可恢复**；
2. **lineage 错误**（parent 错、push 键漂移、`training_contract_lineage_exact` 意外为 `1`）；
3. **被同 parent 支配**：同 parent 内存在另一臂（含该 parent 的矩阵 `C+S0` 对照），fall 与
   击球（completion 与 legal return）**全部**严格更优且持续 `≥4000` iterations——环境混杂
   披露同上，引用对照裁臂时须同时给 K100 中期证据或明确标注混杂。

不允许因"早期难看""被推得摔多"“稀疏 reward 为零”或跨 parent 比较而停臂。停止执行只能读
launcher sidecar、重验 numeric PID/PGID/starttime/argv 后按
[RunPod 纪律](../../operations/run_on_runpod.md#已登记-phase-1-实验臂的算力释放)对 exact
PGID 处置；禁止 `pkill -f`/`killall`。队列无自动 stop/retry/promote 权限。

## 风险与已知边界

- **push 全程可触发，不避击球窗**：与 PACE/BeyondMimic 同款；击球瞬间被推会直接压低
  completion，这是设计的一部分（要买的是"被扰动还完成任务"），但意味着 completion 对照差
  里混着"环境更难"与"策略更脆"两种成分——终点无 push 考卷是唯一干净分离。
- **`vz=0` 冻结**：BeyondMimic 带 `z ±0.2`，本波不带（保持幅度阶梯单变量）；列 follow-up。
- **实现落盘晚于本预注册**：push 覆盖键与 EventTerm 接线是并行新实现，exact 语义以实现+
  单测为机器真源；与本文人话不符时必须先修订本记录再渲染命令（与矩阵 S1 条款同款）。
- **两侧键面尚未统一（落盘时已知，显式披露）**：queue config/renderer/队列单测冻结的是
  （已解决）落盘竞态曾造成文档写八键区间面、trainer 写五键幅度面；主控冻结前已统一为五键幅度面并由 renderer 单测逐字交叉断言 train.py `_PUSH_KEYS`，本条只作历史记录。
  `vel_xy_mps/ang_vel_radps/ang_axes` 五键**幅度面**。对本波 6 档二者语义等价（对称区间⇔
  幅度＋轴选择），但键名不同，主控合并时必须统一成一个键面并同步 config/renderer/trainer/
  单测。`--checklist` 的远端 grep 与 trainer boot fail-loud 是双保险——键面不一致时任何臂
  过不了核对也 boot 不起来；此漂移必须在冻结 40-hex commit 前清零。
- **对照跨 queue**：对照 `w_c_s0`/`v_c_s0` 属矩阵 queue（20260720），与本波（20260721）
  source commit 可能不同——发射前须 diff 两 commit 间训练路径无行为性改动并记录结论，
  否则对照失效、必须自买同 commit 对照。
- **槽位时点不齐**：空槽即填意味着 12 臂启动时间不同；一切比较都按 absolute milestone
  对齐（[组合保护式淘汰](../../DEFINITIONS.md#task-revision-pruning-portfolio)同款纪律），
  不许错位比赛。
- **诊断谱系**：全部后代 formal-exact-ineligible；任何"采用"都要走 exact-lineage 重跑。
- 与 [`lateral balance perturbation`](../../DEFINITIONS.md#lateral-balance-perturbation)
  （恢复窗内有界 WORLD-Y 推力，E1 接线、从未点火）不是同一机制：那是窗口门控的定向力，
  本波是全程随机速度扰动；两者不得混称，胜者若要合并须另立消融。

## 后续（列入 follow-up，不进本波）

1. 胜者幅度 × 矩阵机制胜者（T×S）组合臂；
2. 窄带 vs 全带幅度分布消融（固定 `|v|` vs `uniform ±`）、`vz` 是否补上；
3. push 方向条件化（打拍侧 vs 非打拍侧、恢复窗 vs 全程）——与 lateral perturbation 轴合流。

## 运行表

| 运行 | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| W 已发 7 臂 | `closed_incomplete` | p02 `10300`; p035 `11600`; p05 `11800`; yaw `12000`; p08 `12900`; f035 `11000`; f08 `9200` | checkpoint/日志；仅 p02 有逐臂 probe | 原资产全保留 | 均未到 16700，且 source/admission 不 exact；只作 directional evidence |
| V 已发 7 臂 | `closed_incomplete` | p02 `13100`; p035 `11400`; p05 `13000`; yaw `13900`; p08 `11500`; f035 `12700`; f08 `9800` | checkpoint/日志 | 原资产全保留 | 均未到 16700，且 source/admission 不 exact；只作 directional evidence |
| W/V 未发 4 臂 | `canceled / never launched` | `{w,v}_{ang,fast}` | 远端无 namespace/checkpoint | 无 | 旧波已被新 vendor baseline 取代，不再补发 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

“物理不摔”将使用 `physical_fall_count`（all-attempt 分母）。本波是 single-swing continuation
诊断，不声称连续恢复；即使 Isaac/K100 改善，仍须独立 vendor MuJoCo/Gate3 行为卷。

## 决定

- 决定：`closed_incomplete / superseded`；`no dose winner`
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：不续训、不补旧 judge、不删除原资产。push 方向依据改由
  v2.3 既有裁定 + 外部三库 + 智元同底盘幅值共同背书；新 ActionBall vendor profile
  用 fresh exact lineage 重建，不把本波写成已验证胜者。

## 复现与证据

从零到发射的操作步骤、空槽认领、监控与收口清单见
[run_phase1_push_robustness_wave](../../operations/run_phase1_push_robustness_wave.md)。
本记录保留 14 臂 checkpoint/日志与 5 份 diagnostic judge 的历史定位；它们均不授权
剂量胜者、部署或真机结论。
