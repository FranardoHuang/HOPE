# EXP-P1-BALANCE-ACTION-SLEW-20260720 — 腿腰恢复期执行目标突变是否比全身 raw-action 平滑更适合乒乓

- 状态：`ready`
- 运行态：`probe2 W-C natural exit / outer-verifier rejected；probe3 manifest-bound / not launched`
- 阶段/轴：Phase 1 / 单拍后的平衡恢复与动作平滑
- 集成小目标：降低高回台候选的摔倒与腿腰突变，同时不压低稳定候选的击球完成和回台
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：Wave A runtime mechanics=`E3`（仅 W-C 两步探针）；机制结论=`E1`（无受理收据）；外部 M0 moving input gate=`E2`
- 创建日期/最后复核日期：2026-07-20 / 2026-07-20

本记录使用的 [`raw action-rate`](../../DEFINITIONS.md#raw-action-rate-l2) 是每个 50 Hz tick
连接当前与上一 policy 输出的全 31 维二次差；[`processed-q_des slew hinge`](../../DEFINITIONS.md#processed-qdes-slew-hinge)
是只看执行前已变换、已 clamp 的 15 个腿腰目标的恢复期阈值惩罚。
[`W/V × C/N/H`](../../DEFINITIONS.md#balance-action-slew-matrix) 是本轮六格的完整人话定义。
本实验是[稳定性 Wave A](../../DEFINITIONS.md#balance-stability-waves)，不是唯一平衡方案；下文另预留
Wave B 的下半身老师消融，但不猜尚未审计的 flag。

## 问题与假设

问题：对需要快速挥拍、但击球后容易姿态混乱的乒乓策略，现役只看相邻 raw action 的全身 dense
惩罚是否已经有效；如果需要更强的下肢约束，能否只在同一拍恢复窗、只对腿腰执行目标的极端变化收费，
而不抑制击球臂的快速动作？

可证伪假设：把现役全身 `action_rate_l2=-0.10` 换成恢复期 processed-q_des 腿腰铰链，会在高摔倒
`V` parent 上显著降摔，并在稳定 `W` parent 上保持完成、回台和摔倒非劣；如果只让动作变平滑却过不了
下文的击球与硬摔倒门，或根本没有激活预注册尾部，则拒绝这一 replacement。

### 50 Hz 下“只连上一 action”是否太弱

不是“只产生一次联系”。每一个 tick 都收费，因此相邻边沿串成覆盖整段轨迹的离散平滑先验；对频率为
`f` 的单维正弦 action，其每步均方差分按 `sin^2(pi*f/50)` 增长，所以 50 Hz 下它优先压高频抖动，
而不是慢漂。现有历史对照支持“它并非小到完全没有作用”，但不是 formal 因果证明。它有三个明确边界：

1. 它只记一个相邻样本，没有更长时域状态，也不是二阶差分或恢复策略。
2. 它对 31 维 raw policy action 全程一视同仁，可能同时压住需要快速变化的击球臂。
3. 它量的是 affine transform 和 q_des clamp 之前的 action；真正送给比例微分控制器的执行目标可能不同。

因此 Wave A 不直接把全身 dense 权重做得更大。“腿部可以大一些”须拆成两种不同意思：腿腰的异常突变
可以使用更强、更有针对性的惩罚；合法迈步的每帧免罚变化量也应随各关节速度许可变大。这里同时采用
相位化、阈值化、按实际关节速度许可归一化的 processed-q_des 项；击球前、触球窗、手臂和下一拍揭题
都不收费。

## 已有因果线索（不是本轮结果）

[半秒冲刺记录](EXP-P1-HALF-SECOND-SPRINT.md#2026-07-20-action-rate-证据回收)保存了 runtime 配置差分只剩
action-rate weight、同 `qdot=0` 的 `-0.05` 与 `0` 两格完整 5701–6700 指标、119 MB 证据副本和
checkpoint SHA；历史 hard contract 未绑定该轴，所以只作方向性诊断。
关闭惩罚后 raw action delta、最大关节速度和 base pitch 都明显增大，显示这两条历史轨迹中的相邻项
与高频平滑有关；但
completion、return 与 fall 有交叉取舍，不能据此把全局 `-0.05` 或更强全身权重宣布为平衡答案。

## BeyondMimic 给出的边界

[BeyondMimic 项目](https://beyondmimic.github.io/)、[论文 v4](https://arxiv.org/abs/2508.08241v4)和
[官方代码的冻结 reward 配置](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L198-L278)
采用的是组合平衡学习：全身 tracking、`action_rate_l2=-0.1`、关节限位和接触约束、低阻抗比例微分
控制与 action scale/armature、失败终止、随机初态、1–3 秒外扰，以及按失败区间自适应采样。其直接
消融最支持的是自适应采样，不是“action-rate 单项足以解决平衡”。

所以本实验只隔离 Wave A 的 action-slew 机制，不声称复现 BeyondMimic。下半身老师与失败采样等机制
必须另开 matched ablation，不能把整套上游组合的效果归给本轮一个 Reward。

## Wave A：冻结的 setting

机器真源是
[`phase1_balance_action_slew_20260720.yaml`](../../../configs/phase1_balance_action_slew_20260720.yaml)；
[`run_phase1_balance_action_slew_queue.py`](../../../scripts/run_phase1_balance_action_slew_queue.py)默认只验证并
打印计划，不 SSH、不写远端、不发 signal。远端训练 source 必须是 clean、detached 的 exact commit
`54c9a62656f0e60e5bb41cbcfa0e5a972b793906`；不能以启动时的任意 HEAD 代替。
即使 source 已冻结，命令生成仍被独立复核的 [`launch manifest`](../../DEFINITIONS.md#balance-launch-manifest)
硬门阻断；它必须绑定 source、queue 与全部远端输入的 SHA-256，其中 preconverted `model.usd` 还要连同
依赖的完整 6-file sibling bundle 做 tree hash。当前冻结清单是
[`phase1_balance_action_slew_launch_manifest_20260720.json`](../../../configs/phase1_balance_action_slew_launch_manifest_20260720.json)，
文件 SHA-256=`2d3e7955a1f7e6af3624826f1e7ceaaebd9b2bb0a2c1c72d4b43d0a7ce3bae17`、content
SHA-256=`e86873a2aeedc1c408d9d1c19f95c8818455213811852cc92858f92ea3399215`。它只把 probe command
从“缺清单”解锁为“可渲染”，不表示已经 SSH 或启动。train command 仍必须消费六份
[`probe receipt`](../../DEFINITIONS.md#balance-probe-receipt-set)。

| 字段 | 冻结值/SHA |
| --- | --- |
| source checkout | `/workspace/codexschema/nohope_balance_action_slew_20260720`；clean detached `54c9a62656f0e60e5bb41cbcfa0e5a972b793906` |
| 动作 | `v4rg_runtime_order_v3` 正/反手，路径固定在 YAML |
| 观测/action | `deploy_parity_face179`，actor `179→31`；`qdes_clamp=true` |
| plant/engine | Isaac `HOPEPingPongVirtualBall`，`dt=0.005`、decimation `4`，即 50 Hz；零关节摩擦 |
| parent/seed | `W@model_6700` 与 `V@model_6700`；六格均 `seed=3`，完整恢复 policy/value/optimizer/normalizer |
| 课程/题库 | short-focus `0.10/0.45/0.40/0.05`；同一 schema-3 train bank |
| 其他 Reward | 每个 parent 内保持原击球/准备配方；qdot-limit hinge=`0`，post-swing replay=`0` |
| 谱系 | [`checkpoint_allow_contract_mismatch=true`](../../DEFINITIONS.md#checkpoint-contract-mismatch)；所有后代只作诊断、永久 formal-exact-ineligible |

### processed-q_des 腿腰铰链

对 exact 3 个腰关节和 12 个腿关节逐关节计算：

```text
u_j    = abs(q_des[t,j] - q_des[t-1,j]) / (qdot_limit[j] * 0.02)
tail_j = 1 - exp(-(relu(u_j - 0.85) / 0.15)^2)
value  = mean(tail_j over the exact 15 joints)
```

`q_des` 已经过 action affine transform 与 train=deploy joint clamp。只有同一拍触球后
`0.20 <= age <= 1.55 s`、且 previous q_des 有效时才返回 `value`；reset 后第一步强制为零，不能把 episode
边界伪造成大突变。`tail_j` 位于 `[0,1)`，weight=`-0.25`；RewardManager 再乘 `0.02 s`，因此满激活时
每 tick 的惩罚幅值小于 `0.005`、每连续 eligible 秒小于 `0.25`。50 Hz 漂移、关节重排、腿腰集合不完整
或非有限/非正速度上限都 fail closed。这里既把惩罚集中到腿腰，又没有给所有腿关节同一个绝对阈值：
速度上限更大的关节会自然得到更大的每帧免罚变化量，合法快速迈步不会仅因绝对角度变化较大就被收费。
按当前 pinned A3 nominal limits，`0.85` margin 对执行目标的免罚变化量约为 hip `0.204`、knee `0.248`、
ankle-pitch `0.184`、ankle-roll `0.328 rad/tick`；runtime 仍以实际 articulation limit 为准。这已经是较宽的
腿部 allowance，先看 tail activation、fall 与 tilt，再决定是否另开 `1.25×/1.5×` allowance 消融，不能
仅凭“腿应该更快”修改本轮冻结值。

### 六格矩阵与唯一 run names

`C`、`N`、`H` 三格都显式写入 raw/processed 权重与 processed margin/window，避免“默认值”冒充单变量。
下面每行先给人话，再给唯一 `run_name`：

| 运行（人话 + `run_name`） | parent | raw action-rate | processed-q_des | 固定槽 |
| --- | --- | ---: | ---: | --- |
| W 当前 dense 对照 — `phase1_balance_slew_w_c_dense_m0p10_seed3_20260720` | W 稳定 parent | `-0.10` | `0` | Pod1 GPU0 |
| W 无平滑负控 — `phase1_balance_slew_w_n_none_seed3_20260720` | W 稳定 parent | `0` | `0` | Pod1 GPU1 |
| W 恢复期腿腰 replacement — `phase1_balance_slew_w_h_processed_qdes_seed3_20260720` | W 稳定 parent | `0` | `-0.25` | Pod1 GPU2 |
| V 当前 dense 对照 — `phase1_balance_slew_v_c_dense_m0p10_seed3_20260720` | V 高摔倒 parent | `-0.10` | `0` | Pod2 GPU0 |
| V 无平滑负控 — `phase1_balance_slew_v_n_none_seed3_20260720` | V 高摔倒 parent | `0` | `0` | Pod2 GPU1 |
| V 恢复期腿腰 replacement — `phase1_balance_slew_v_h_processed_qdes_seed3_20260720` | V 高摔倒 parent | `0` | `-0.25` | Pod2 GPU2 |

`W` 与 `V` 的不同 parent 配方不构成 W-vs-V 单变量比较；机制只在各自 parent 内比较 `C/N/H`。

### 2026-07-20 W-C probe2 假拒绝与 probe3 修订

Pod1 GPU0 的 `phase1_balance_slew_probe2_w_c_seed3_20260720` 是唯一实际启动的旧探针。它从 W
`model_6700.pt` 自然跑完两个 update，`2026-07-20T01:08:47.435447Z` 开始、
`2026-07-20T01:09:24.372852Z` 退出，terminal status 为 `exit_code=0`、`normal_exit=true`，并生成
`model_6700.pt`、`model_6701.pt`、hard contract 和完整 TensorBoard ledger。进程组与 GPU 已自然释放；
没有人工 signal。

旧 outer verifier 没有发布 receipt，因为它错误要求**每个** update 都有非零恢复期分母。第 6700 步
包含 `24 × 4096 = 98304` 个样本，其中 previous-q-des valid/invalid=`89899/8405`，但
recovery-eligible=`0`；初始准备时间最短 `0.36 s`，再加触球后 `0.20 s` 的窗口起点，最早资格时刻
`0.56 s` 已超出首个 `24 × 0.02 = 0.48 s` rollout。第 6701 步已有 recovery-eligible=`31459`、
tail-active=`31456`、above-margin joints=`189191`、gated-tail sum=`11670.224609375`，证明通道实际激活。
故这次拒绝是量尺错误，不是 trainer 崩溃，也不能写成 W-C 机制失败。

probe3 的修订仍逐 update 强制 observed=`valid+invalid`、tail/joint/value 守恒；它只允许某一 update 的
recovery-eligible 为零，并新增**两步合计必须大于零**的硬门。为保持 no-clobber，全部六格转到新根
`/workspace/codexschema/phase1_balance_action_slew_v2_20260720`，旧 probe2 目录原样保留且不得补写 receipt。
probe3 的 config、runner、manifest 进入 `origin/main` 前，不允许重发任何一格；旧 manifest 也不再提供
发射权限。

## 预算、量尺与停止规则

1. 六格先各跑 `4096 env × 2 update` 的完整场景合同探针。每格必须自然退出到独立
   `model_6701.pt`（absolute milestone=`[6701]`，exclusive iteration upper bound=`6702`）；dedicated exact
   verifier 须证明 6700/6701 两步 processed-q_des、completion/fall/legal-return、ready-tilt、qdot tag 的
   分母与守恒账；processed-q_des 恢复资格允许单个 update 为零，但两步合计必须非零，其余预注册行为
   分母仍须逐 update 非零。另须证明 finite policy/value/full optimizer/two normalizers、C/N/H exact
   weight-margin-window-applied markers、lineage=`0`、fatal scan，以及 leader/PGID/GPU 已释放并发布不可覆盖
   收据。不能借用 lean fresh-probe 的相对 `[1]` 语义。只有本地按 `DIR/JOB_ID/probe_receipt.json` 收齐并
   重验全部六份，才允许生成科学训练命令；人工写一个“probe 已通过”布尔值无效。
2. 科学 continuation 为每格最多 `+1001 update`，预注册相对 parent 的 `+200/+500/+1000`
   （absolute `6900/7200/7700`）里程碑。`+200` 只看激活/明显崩坏，`+500` 看方向，`+1000` 决策。
3. 单 seed 只筛机制；只有 surviving replacement 与其 matched control 才可另开至少 3 seed 的 formal-fresh
   预注册。本 queue 的 descendants 永远不能因追加 seed 变成 formal exact。

每个里程碑固定使用**截至该 absolute milestone 的最后 100 个完整 PPO update**：`6900` 用
`6801..6900`，`7200` 用 `7101..7200`，`7700` 用 `7601..7700`。窗口内 100 个 step 必须逐个存在，
不得插值、改成累计窗或事后挑最近若干点。行为账统一读 `Live/racket_target/`，q_des 账统一读
`Live/processed_qdes_slew/`，qdot 账统一读 `Live/qdot/`；先逐 tag 求和 exact ledger counters，再做下列
除法，不能先对每 update 的比例作平均：

- completion=`sum(swing_completion_count)/sum(swing_outcome_count)`；physical fall=
  `sum(physical_fall_count)/sum(swing_outcome_count)`；legal return=
  `sum(virtual_legal_return_count)/sum(strike_opportunity_count)`；
- q_des tail mean=`sum(gated_tail_value_sum)/sum(recovery_eligible_sample_count)`，tail-active rate=
  `sum(tail_active_sample_count)/sum(recovery_eligible_sample_count)`，above-margin joint fraction=
  `sum(above_margin_joint_count)/(15*sum(recovery_eligible_sample_count))`；
- ready tilt mean=`sum(ready_tilt_rad_sum)/sum(ready_tilt_eligible_sample_count)`；qdot normalized excess mean=
  `sum(Live/qdot/normalized_excess_square_sum)/sum(Live/qdot/observed_sample_count)`，qdot excess rate=
  `sum(Live/qdot/excess_sample_count)/sum(Live/qdot/observed_sample_count)`。

processed-q_des 的 observed/previous-valid/reset-invalid/recovery-eligible/tail-active/above-margin 整数账仍须逐
update 守恒。base angular velocity、strike composite 和训练标量可作为同一固定窗的辅助诊断报告，但当前
instrumentation 没有 q_des-tail p95/max，因此它们不进入硬接受式，也不得由外部脚本事后伪造。任一硬指标
分母为零、100-step 不完整、probe 未激活或计数不守恒时结果无效，不把零写成成功。

`H` 相对同 parent `C` 的最终接受门：

- V：fall 相对下降至少 `25%` 且绝对下降至少 `5` 个百分点；completion 下降不超过 `2` 个百分点，
  legal return 下降不超过 `3` 个百分点。
- W：fall 不高于 `max(0.5%, C + 0.2 个百分点)`；completion/return 分别不劣于 C 超过 `2/3` 个百分点。
- 两个 parent：q_des tail mean 或 above-margin joint fraction 至少下降 `20%`，并且 ready tilt mean 或
  qdot normalized excess mean 至少下降 `10%`；这些比例只能由上面的固定 100-step exact ledger 计算。
- physical-fall 硬失败不能由更高 return、较小 action delta 或其他平均分抵消。

`N` 是机制负控：它用于确认完全无平滑时尾部是否放大，不是预先允许晋级的候选。任何自动 stop、自动 retry、
第二 seed、judge、部署或真机命令都不在 queue 权限内；停止只能在证据落盘后重验 exact PID/PGID/starttime/argv，
再按 [RunPod 操作页](../../operations/run_on_runpod.md#已登记-phase-1-实验臂的算力释放)处置数值进程组。

## Wave B：下半身 matched ablation（moving-teacher 输入已拒绝）

Wave A 只回答“怎么对执行目标突变收费”。它不回答机器人是否应模仿静态下半身准备姿态、左右移动老师，
或改用不依赖 demo 的下身稳定约束。下一波的 matched control 是现役 upper-only imitation；treatment 只能从
当前静态 v4rg 下半身参考或 non-demo stability constraint 中选择，上半身动作、击球 Reward、题库、parent、
seed、预算和判读尺保持匹配。

2026-07-20 对 live [`M0`](../../DEFINITIONS.md#motion-m0) 输入的只读回收已经闭合“左右移动老师能否
立即使用”：exact manifest 的 `completed_utc=2026-07-14T05:06:21.749762Z`，文件位于
`/workspace/codexschema/motion_video_intake_20260713_m0/exact_gmr_v2/completion_manifest.json`，SHA-256
`fdd60fcfdc7290677aa51ec7804278568a267e239de548cdb623d0565dac396e`。四份 exact-GMR moving 输出都通过
30 Hz、31-DoF、finite 结构审计，但 `stance_passed=false` 为 `4/4`：left1/left2/right1 的末态脚间横向
分量相对各自初始值偏离超过 `3 cm`；left1/right1/right2 的站距收窄超过 `5 mm`。right2 只通过横向分量
band，仍因 no-narrowing 失败。manifest 顶层和每个结果
均为 `formal_eligible=false`、`schema2_authorized=false`、`training_authorized=false`、
`hardware_authorized=false`。因此本轮把 M0 moving-teacher input gate 判为 **reject / no-launch**；“文件存在”
不能越过后续独立 schema2、L0/L1、桌网、动力学预注册。

当前不为 Wave B 发明 flag 或 run name。先审计 repo 中实际 body-mask/phase 接口、静态 v4rg 的 exact
下半身语义与 non-demo constraint，再单独冻结 machine-readable contract、run table 和安全门。Wave B 不得
借用 Wave A 的 launch authority，也不能与六格 action-slew 矩阵混为一个多变量结论。

## 运行表

| 运行 | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| Wave A probe2 W-C | `natural exit / verifier rejected` | W `model_6700` / seed3 | E3 mechanics only | `model_6701.pt` 与两步 ledger；无 receipt | 首步 recovery 分母合法为零；不作机制结论，不重用旧目录 |
| Wave A probe3 六格 | `manifest-bound / not launched` | W/V `model_6700` / seed3 | E1 source/queue/manifest | launch manifest `2d3e7955…3bae17`；新 v2 namespace 尚无 runtime | intentional parent-contract mismatch；diagnostic only |
| Wave B 下半身 matched ablation | `design pending / M0 moving rejected` | 未冻结 | E2 input gate | M0 manifest `fdd60fcf…396e` | moving teacher no-launch；只继续静态 v4rg 或 non-demo constraint 设计 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

Wave A 是 single-swing continuation 诊断，不能声称完成 T0/T1/T2 连续恢复；即使 Isaac fall 改善，仍须
独立 vendor MuJoCo/Gate3 行为卷。

## 决定

- 决定：`inconclusive`（W-C probe2 只闭合 trainer mechanics，outer-verifier 量尺已修订；六格仍无受理收据）
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：先过六格 2-update full-scene probe，再按里程碑购买 Wave A；Wave B 另行审计与预注册。

## 复现与证据

只读 plan、命令生成和 Pod 启动纪律见
[Run Training](../../operations/run_training.md#恢复期腿腰-processed-q_des-slew-wave-a)与
[RunPod](../../operations/run_on_runpod.md#2026-07-20-action-slew-wave-a-启动前状态与发射纪律)。
当前记录只启动过上述 Pod1 W-C 两步 Isaac 探针；没有启动科学长训、judge、部署或真机。
