# EXP-P1-LOWER-BODY-STABILITY-20260720 — 静态下肢软老师与无参考稳定约束消融

- 状态：`superseded`（2026-07-20：本六格队列被 [24 格平衡×时序矩阵](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md) 取代；B1/B2 机制以 S3/S2 档原样并入该矩阵，本记录只作机制与源码出处，不再单独发射）
- 运行态：`E1 source/queue only；NO-LAUNCH`。六格完整场景启动探针尚未产生受理 receipt；六条科学
  长训在六份 exact probe receipt 全部通过前保持锁定。
- 阶段/轴：Phase 1 / 下肢稳定学习
- 人类负责人：Franco
- 执行者：Codex
- 创建日期/最后复核日期：2026-07-20 / 2026-07-20
- 最高证据等级：`E1` 源码、队列、CPU/模拟 Torch 单测；没有 Isaac full-scene、训练或行为结果

本记录中的 B0、B1、B2 是三条[实验臂](../../DEFINITIONS.md)：B0 是两种新机制都关闭但两套
测量探针都开启的对照；B1 只开启静态 v4rg 十二腿关节软模仿；B2 只开启无动作参考的支撑宽度与腿速
稳定惩罚。这里的 [`v4rg`](../../DEFINITIONS.md) 是当前正反手参考动作族；正式发射必须进一步绑定
`v4rg_runtime_order_v3` 的 exact schema-2 bytes，不能只凭名字认定。

## 要回答的问题

相邻 action 的变化率惩罚能压制 50 Hz 控制中的高频跳变，但对慢姿态漂移、脚距塌缩、支撑切换和
挥拍后的腿速收敛没有直接目标。本轮因此不把“平滑”和“稳定”视为同一件事，而比较两个互斥解释：

1. B1：机器人是否需要一个很软、只作用于当前静态挥拍支撑窗的下肢动作先验；
2. B2：不增加任何 motion-reference tracking，仅用物理支撑宽度下界和实际腿关节速度尾部，能否得到
   相同或更好的稳定收益。

B1 与 B2 不可同时开启。现有 `action_rate_l2`、`foot_orientation`、`upright`、`foot_slip` 和普通
`joint_vel_l2` 在每个 parent 内的 B0/B1/B2 三格保持完全相同；否则不能把差异归因到本轮机制。

## 一手文献给出的边界

- [BeyondMimic](https://arxiv.org/abs/2508.08241) 使用统一的身体位置、朝向、线速度和角速度软跟踪，
  再配少量 action smoothness、关节限位和自碰正则。它支持“模仿与平滑是不同通道”，但不证明只给
  乒乓球腿关节加 reward 就一定稳定。
- [OmniH2O](https://omni.human2humanoid.com/resources/OmniH2O_paper.pdf) 把站立/深蹲的固定 root 和
  下肢数据加入训练分布，并报告这对上身任务中的稳定站立重要；同时强调 regularization curriculum。
  这支持先测静态软老师，而不是立即把未知横移动作当真源。
- [ALMI](https://papers.neurips.cc/paper_files/paper/2025/hash/6b081a311e0b9c75590ba97b104a2ce3-Abstract-Conference.html)
  让下肢学习速度条件化 locomotion、上肢学习动作模仿，并交替训练两者协调。它说明未来横移老师应当
  按静止/左移/右移或速度命令条件化，不能把一条固定站姿老师强加到所有横移状态。
- [ExBody](https://arxiv.org/abs/2402.16796) 明确放松双腿的逐姿态模仿，只要求腿稳健跟随速度命令。
  这是反例边界：全程硬追人腿轨迹可能伤害机器人的任务适应和稳定。
- [PHC](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Perpetual_Humanoid_Control_for_Real-time_Simulated_Avatars_ICCV_2023_paper.html)
  把跌倒恢复作为可扩展的独立能力，并展示无外部稳定力的 fault-tolerant imitation。它支持后续单独训练
  recovery/push，不支持把本轮任一 reward 单测写成恢复能力已经成立。

因此本轮只购买最小可归因的 B0/B1/B2。AMP、PMP、双策略 ALMI、moving-teacher 组合和 push-training
都留到本轮出现胜者之后，不与首轮 reward 主效应混在一起。

## B0/B1/B2 的精确语义

### 公共相位门

两种机制共用同一个不看命中结果的 inclusive gate：

- 挥拍前：`motion.in_hold=false`、`pre_strike=true` 且 `0 <= time_to_strike <= 0.30 s`；
- 挥拍后：由 phase-aligned exact-strike opportunity 启动的 same-attempt control-time clock，
  `0 <= age <= 0.40 s`；reset、动作 wrap 或新题 reveal 会关闭旧 attempt；
- 不读取 contact、hit、return 或 `_exact_attempt_completed`，所以失败挥拍不会从分母中消失。

### B1：静态 v4rg 十二腿关节软模仿

精确关节是左右各六个：hip pitch、hip roll、hip yaw、knee、ankle pitch、ankle roll，共 12 个；腰、头和
双臂不进入误差。运行时 `joint_names`、`articulation_joint_names` 和 motion reference 必须与当前 A3
31 关节顺序完全一致，否则 fail closed。

令公共门为 `g`，默认 `std=0.35 rad`：

```text
r_B1 = g * exp(-mean_j((q_j - q_ref_j)^2) / 0.35^2)
weight_B1 = +0.5
```

这是 positive soft kernel，不是 joint target、termination 或硬姿态锁。B1 只对当前静态 v4rg teacher
有发射资格；如果启动配置混入尚未过门的移动老师，整格失去资格。

### B2：无参考支撑稳定 bundle

B2 不读取 `motion.joint_pos` 或 loaded joint reference。它只有两个等权、有界分量：

1. 把左右 ankle-roll link 的世界位置差投影到 base-yaw 横向轴，得到有符号 left-minus-right stance
   width；只惩罚低于固定物理下界 `0.22 m` 的塌缩/交叉，scale 为 `0.05 m`；
2. 在 12 个腿关节的实际 `|qdot|` 超过 `1.0 rad/s` 后才收费，scale 为 `0.5 rad/s`。

```text
stance_tail = 1 - exp(-(relu(0.22 - signed_width) / 0.05)^2)
qdot_tail   = mean_j(1 - exp(-(relu(|qdot_j| - 1.0) / 0.5)^2))
r_B2        = g * (stance_tail + qdot_tail) / 2
weight_B2   = -0.25
```

这里故意只有 stance-width **下界**，没有上界；它防止脚距塌缩但不向横移步幅收费。B2 不重复
`foot_orientation`、slip 或 upright。唯一剩余的方向性重叠是 qdot tail 与很小的全关节
`joint_vel_l2` 都偏好低速度；区别是 B2 只看 12 个腿关节、保留 `1.0 rad/s` 免费区、使用有界尾部且只在
支撑窗激活。该相关性必须披露，不能把 B2 写成“完全没有任何速度正则重叠”。

## 冻结矩阵与建议起跑量

每个 cell 都必须显式给出 B1/B2 两个 weight；只给一个或任何 bool/NaN/Inf/错符号都会在修改配置前
失败。W 和 V 是两个不同 parent stratum，不跨 parent 直接比较绝对值：

| Cell | B1 weight | B2 weight | 人话解释 |
| --- | ---: | ---: | --- |
| B0 | `0.0` | `0.0` | 新 reward 全关，但两套 probe/contract 都在，提供同信号对照 |
| B1 | `+0.5` | `0.0` | 只测试静态 v4rg 十二腿关节软老师 |
| B2 | `0.0` | `-0.25` | 只测试 reference-free stance floor + realized leg-qdot settle |

拟议的六个 `run_name`（即每条 run 的唯一名字，见[术语表](../../DEFINITIONS.md)）为：

| Parent | Cell | 拟议 run_name | 含义 |
| --- | --- | --- | --- |
| W：拍心优先 × 自由非击球臂 `model_6700` | B0 | `phase1_lbwave_w_b0_seed3_20260720` | W parent 的双机制关闭对照 |
| W | B1 | `phase1_lbwave_w_b1_static_pose_seed3_20260720` | W parent 的静态腿 pose 软模仿 |
| W | B2 | `phase1_lbwave_w_b2_support_settle_seed3_20260720` | W parent 的无参考支撑/腿速约束 |
| V：拍速优先 × 强准备 `model_6700` | B0 | `phase1_lbwave_v_b0_seed3_20260720` | V parent 的双机制关闭对照 |
| V | B1 | `phase1_lbwave_v_b1_static_pose_seed3_20260720` | V parent 的静态腿 pose 软模仿 |
| V | B2 | `phase1_lbwave_v_b2_support_settle_seed3_20260720` | V parent 的无参考支撑/腿速约束 |

冻结建议是 6 个独立 GPU slot、`seed=3`、`4096 environments`。先做每格 2-update
[`full-scene-probe`](../../DEFINITIONS.md#full-scene-probe)，全部通过后才允许同 parent 内并行买
`+200/+500/+1000` continuation；不因某格早期好看而改变其他格预算。它们从旧 `model_6700` 改 reward
继续训练，必须显式使用 `checkpoint_allow_contract_mismatch=true`，属于 causal continuation，不是
fresh/formal exact lineage；必须保留 parent，不得声称连续谱系完全匹配。

## 队列与不可变输入预注册

本轮唯一机器队列是
[`configs/phase1_lower_body_stability_20260720.yaml`](../../../configs/phase1_lower_body_stability_20260720.yaml)，
命令渲染器是
[`scripts/run_phase1_lower_body_stability_queue.py`](../../../scripts/run_phase1_lower_body_stability_queue.py)，
操作步骤见
[`run_phase1_lower_body_stability_wave.md`](../../operations/run_phase1_lower_body_stability_wave.md)。二者只定义
局部依赖 `validate -> reviewed manifest -> 六份 probe receipt -> long`；全局顺序、算力认领和暂停/恢复仅以
最新 `origin/main:docs/NOW.md` 为准。

- exact source：`5db7366aaa1562d592093dc0d512ec212f14e39e`，远端必须是 clean detached checkout；
- `cfg/algo/ppo.yaml` SHA `ffa64ffe...7a42` 把 `num_steps_per_env=24` 绑入 manifest/claim，每个
  probe update 因此必须恰有 `4096 * 24 = 98304` 个 observed sample；
- W parent：checkpoint `2caab3dd...fcce`，相邻 hard contract `e208b682...8551`；
- V parent：checkpoint `ad901910...2716`，相邻 hard contract `274cb3bd...aee0`；
- 静态 v4rg 正/反手：`f2cb2d9f...1687` / `17225533...7534`；schema-3 train bank：
  `3a9d8851...5b71`；
- A3 ignored runtime tree：46 files / 15378264 bytes / tree `071640ea...b070`；预转换 USD bundle：
  6 files / 21897893 bytes / tree `716487df...beb4d`，其中 `model.usd` 为 `1b3fecd7...693c`；
- 三格共同显式固定 `action_rate_l2=-0.1`，并要求 processed-qdes slew block 缺席；因此本轮唯一
  within-parent 科学差异是 11 个成对 Wave-B override；
- config SHA `7193d789...3146`；runner SHA `abc8d34d...8556`；checked-in reviewed manifest
  [`phase1_lower_body_stability_launch_manifest_20260720.json`](../../../configs/phase1_lower_body_stability_launch_manifest_20260720.json)
  的 file SHA 为 `bae461b0...5565`、canonical content SHA 为 `feed3ed1...d576`。它绑定十份 required
  source file 和上述输入，但自身不是 ambient launch authority；每次授权渲染仍必须显式传入 manifest
  路径与 exact file SHA。

长训固定到 `model_6900`、`model_7200`、`model_7700` 三个 absolute milestones；renderer 只有在本地目录
同时存在六格 verifier 生成的 canonical receipt 且每份都与同一 manifest/claim/probe program 绑定时，才会
生成任何 long SSH argv。continuation checkpoint 的 `training_contract_lineage_exact` 必须继续为 `0`。

## Probe 与 hard-contract 发射门

未带任何 Wave-B override 时，四个 reward/probe term 都是 weight `0`，两份新 hard-contract block
完全缺席，默认训练合同保持旧 bytes。显式 B0/B1/B2 时，两套零值 probe 都设为 manager weight `1`，
reward 本身仍按 cell weight 工作，并且 hard contract 必须同时出现：

- `lower_body_pose_imitation_reward`；
- `lower_body_stability_bundle_reward`。

两块都必须写 `probe_enabled=true`、`activation_ledger=weight_independent_control_step_counters`，并绑定公式、
相位门、权重、尺度、12 腿关节顺序和 ankle-roll body 名。B2 还必须写
`uses_motion_reference=false`；B1/B2 同时 `enabled=true`、单块缺失、关节/刚体顺序漂移都必须拒绝。

2-update receipt 至少满足：

1. pose 与 bundle 的 `observed_sample_count` 都非零且相等；共同
   `support_eligible_sample_count` 非零且相等，所有 counter finite；
2. B0 的两种 `reward_enabled_eligible_sample_count=0`；B1 只有 pose 等于 eligible；B2 只有 bundle
   等于 eligible；
3. `0 <= gated_kernel_sum <= eligible`，且 `gated_reference_motion_l1_mean_sum>0`，证明 B1 probe
   实际看到非平凡腿参考；
4. `0 <= narrow_or_crossed_sample_count <= eligible`，stance/qdot/bundle 三个 gated sum 都在
   `[0, eligible]`，signed width sum finite；
5. terminal checkpoint finite，内嵌/相邻 schema-3 hard-contract SHA、两个 block、source、parent、seed、
   motion 和题库全部与 claim 一致；日志无 traceback、OOM、NaN/Inf 或 fatal；
6. probe 只证明 manager 顺序、公式、ledger 和 full scene 能运行，不进入行为成绩，也不授权第二 seed、
   vendor judge 或真机。

如果 `disable_logs` 或 writer 缺失，现有 runner 不消费/输出 dashboard ledger；这种运行不能产生合格
probe receipt，必须 fail closed，而不是把“没有日志”解释成零激活。

## 行为指标与裁决

所有格按 parent 分层。每个 absolute milestone 只读截至该点的最后 100 个完整 PPO update：
`6900 -> 6801..6900`、`7200 -> 7101..7200`、`7700 -> 7601..7700`。100 个 step 必须逐个存在，
不得插值、换成累计窗、从邻近点补数或事后挑最好 checkpoint。先对 exact ledger counter 求和，
再作除法，不先平均每 update 的 ratio：

- physical fall rate = `sum(physical_fall_count) / sum(swing_outcome_count)`；
- completion rate = `sum(swing_completion_count) / sum(swing_outcome_count)`；
- legal return rate = `sum(virtual_legal_return_count) / sum(strike_opportunity_count)`；
- narrow/crossover rate = `sum(bundle.narrow_or_crossed_sample_count) /
  sum(bundle.support_eligible_sample_count)`；
- stance-tail mean 与 leg-qdot-tail mean 分别是 `sum(bundle.gated_stance_tail_sum) / sum(eligible)` 和
  `sum(bundle.gated_leg_velocity_tail_sum) / sum(eligible)`；
- B1 pose absolute-error readout = `sum(pose.gated_joint_abs_error_mean_sum) /
  sum(pose.support_eligible_sample_count)`；
  它只证明模仿通道改变，不是稳定晋级主门。

前三项读 `Live/racket_target/`，后四项读 `Live/lower_body_wave/{pose,bundle}/`。分母为零、
100-step 不完整、任一 update 的 pose/bundle observed 不是 `4096 * 24 = 98304`、计数不守恒或
fatal 非零，整个 milestone 记为 `invalid`，不把缺失数当成零。base roll/pitch、base angular velocity、
signed stance width、foot slip/contact、腿部 `Delta action`、`Delta^2 action`、hit/error/composite/deadline 都在
同一窗报告，但只是辅助诊断，不得代替下面冻结的门。

`model_6900` 只看 activation/明显崩坏，`model_7200` 只报方向；单 seed 晋级决定只使用
`model_7700` 固定窗。B1 和 B2 各自相对同 parent B0 判读，禁止跨 W/V 比绝对值。某个 treatment
只有同时通过下列全部条件，才能与 matched B0 申请新的多 seed formal-fresh 预注册：

1. V stratum 的 fall 相对下降至少 `25%` 且绝对下降至少 `5` 个百分点；
2. V 的 narrow/crossover rate 或 leg-qdot-tail mean 至少相对下降 `20%` 且绝对下降
   `0.01`（两者均是 `[0,1]` 有界量）；另一项不得高于 `B0 + max(0.02, 0.10*B0)`；
3. W 的 fall 不高于 `max(0.5%, B0 + 0.2 个百分点)`，且 W 的 narrow rate 与
   leg-qdot-tail mean 都不高于各自 `B0 + max(0.02, 0.10*B0)`；
4. W 和 V 的 completion 分别下降不超过 `2` 个百分点，legal return 分别下降不超过
   `3` 个百分点；physical-fall 硬失败不能由更高 reward/composite 抵消。

如果 B1 只降低 pose error，或 B2 只降低自己的 bundle reward 而没通过独立 stability/task 门，
都不晋级。若 B1/B2 同时通过，两者都进多 seed 与 push evaluation，不用单 seed 强选一个；
若有效数据下都失败，则拒绝当前剂量。push-training 仍是另一个实验轴。

## M0 横移老师的明确阻塞

[`M0`](../../DEFINITIONS.md) 是 left-1、left-2、right-1、right-2 四条横移动作候选，不是四个 seed。
2026-07-20 对 Pod1 exact-GMR completion manifests 的只读复核确认文件存在，但 manifest 仍声明
`formal_eligible=false`、`schema2_authorized=false`、`training_authorized=false` 和
`hardware_authorized=false`。预注册 stance 门为横向分量变化不超过 `0.03 m` 且 terminal 相对 initial
不得缩窄超过 `0.005 m`；结果是 0/4 通过：

| M0 candidate | initial -> terminal signed width | 失败原因摘要 |
| --- | ---: | --- |
| left-1 | `0.508335 -> 0.412910 m` | 变化/缩窄 `0.095425 m` |
| left-2 | `0.283615 -> 0.484172 m` | 分量变化 `0.200557 m` |
| right-1 | `0.427067 -> 0.350534 m` | 变化/缩窄 `0.076532 m` |
| right-2 | `0.305483 -> 0.281183 m` | 缩窄 `0.024300 m`，超过 `0.005 m` 门 |

因此本轮**不得**创建“B1 + M0 左右移动老师”或“B2 + M0”训练格，也不得把 diagnostic GMR 输出写成
schema-2 teacher。移动条件化实验保持 deferred；下一步必须另建 schema-2 L0、vendor L1、桌网和动力学
预注册，得到至少一条被正式接受的左移和右移老师后，再做静止/左移/右移分层，而不是借用当前 0/4
失败资产。

## 当前证据与下一步

本分支已覆盖：50 Hz inclusive 边界、reset/same-attempt、12 腿关节且无手臂、B2 不读损坏的 motion
reference、base-yaw stance 投影、配置原子拒绝、双 probe ledger、schema-3 pair/顺序/body fail-closed。
独立队列还覆盖默认不发射、六卡唯一映射、显式双 weight、M0 零枚举、manifest/source/input SHA 绑定、
claim no-clobber、自然退出、完整 policy/value/optimizer/双 normalizer checkpoint、GPU/PGID 释放和六 receipt
解锁。用本地 `fast` CPU Python 共同运行 Wave-B source、reward override、schema-3 contract 与 queue suites
得到 `269 passed`（分别为 `35 + 124 + 68 + 42`）；这仍是 E1，不是 Isaac runtime。
本地 Torch 是旧 CPU 测试运行时，不能替代目标 Isaac/RunPod 环境；正式 runtime-valid 状态必须等 clean
detached source 在 RunPod 复跑 focused Torch tests 和六格 full-scene probe 后另行记录。

M0 是独立输入资产门的 negative evidence，不是 Wave-B 第七格；当前 0/4 只拒绝这四份输出进入本轮，
不否定未来重新通过 schema-2/L0/L1/桌网/动力学门的左右移动老师。本记录不更新 `NOW` 队列，不宣称
launch、checkpoint 或行为结果；真正点火前必须先在最新 `origin/main` 的唯一队列里完成 owner/executor/
branch 认领。
