# EXP-P1-LOWER-BODY-STABILITY-20260720 — 静态下肢软老师与无参考稳定约束消融

- 状态：`Partial / NO-LAUNCH`（源码与本地单测候选；RunPod full-scene probe 尚未运行）
- 阶段/轴：Phase 1 / 下肢稳定学习
- 人类负责人：Franco
- 执行者：Codex
- 创建日期：2026-07-20
- 最高证据等级：源码与 CPU/模拟 Torch 单测；没有 Isaac full-scene、训练或行为结果

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

所有格按 parent 分层、同 checkpoint 窗和同题分母报告：

- 稳定：pre/post fall、survival/completion、base roll/pitch、base angular velocity、signed stance width、
  narrow/crossover rate、foot slip/contact；
- 控制平滑：腿部 `Delta action`、`Delta^2 action`、12-leg qdot tail；action-rate 与本轮机制分开报告；
- 模仿：B1 的 12-leg absolute error 与 reference-motion magnitude；
- 乒乓球任务：exact hit、return、position/velocity/normal error、composite 和 deadline；
- 不得只报平均 reward，也不得因跌倒、未触球或 reset 删除 attempt。

首轮只作单 seed 机制 screen。任何候选必须同时改善稳定主指标且不显著破坏 hit/return；若 B1 只降低
腿误差却压住合法重心转移，或 B2 只降低 qdot 却降低击球完成率，都不能采用。胜者才与 matched B0
购买第二 seed 和统一 push evaluation；push-training 仍是另一个实验轴。

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
本地 Torch 是旧 CPU 测试运行时，不能替代目标 Isaac/RunPod 环境；正式 runtime-valid 状态必须等 clean
detached source 在 RunPod 复跑 focused Torch tests 和六格 full-scene probe 后另行记录。

本记录不更新 `NOW` 队列，不宣称 launch、checkpoint 或行为结果。合入主分支后，由统一队列另行绑定
exact source commit、六格 config/claim、Pod/GPU slot 和 terminal receipts。
