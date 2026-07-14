# EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN — 补齐 activation 后重跑底座减速配对

- 状态：`blocked`（same-phase activation successor `0f3900a...` 已 exact 绑定；等待该 source 自己的 strict
  full-scene terminal probe）
- 阶段/轴：Phase 1 fresh C；组合击球精度下，底座减速是否有净收益
- 集成小目标：保住击球精度信号，同时降低击球前底座速度与击球前摔倒率
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（机器可读 blocked 预注册与离线测试；没有本 replacement pair runtime）
- 创建日期/最后复核日期：2026-07-15 / 2026-07-15

共享术语见[术语与人话对照](../../DEFINITIONS.md)。本文的 V1 是“从线速度动作模仿中释放持拍手腕”，
V2 是“在击球时间窗把动作模仿缩放到四分之一”，`base-decel` 是“在击球前非冻结阶段奖励底座追踪随距离
递减的目标速度”，post-swing replay 是“从策略自己上一拍的随挥状态缓冲启动一次 episode”。这些名称继承自
[原始配对记录](EXP-P1-V1V2-BASE-DECEL-INTERACTION.md)，本文不改变它们的 Reward、动作或题库语义。

## 为什么需要 fresh replacement，而不是继续解释旧 run

原始 `base_decel_weight=0/1` 配对已经实际训练，但它的日志没有给出三条共同/处理机制各自的 runtime
eligible denominator。只看配置回显、`Live/Reward/base_decel` 的全环境均值或
`Live/racket_target/base_speed_xy_prestrike` 的窗外置零均值，无法区分：

1. 机制真的在足够样本上作用；
2. 机制只在极少样本上触发；
3. 指标因窗外补零而看似变小。

因此原配对保留为不可覆盖的基础设施与训练证据，但 activation 之后的因果解释记为
`invalid/instrumentation-blocked`。本记录新建 run name、run directory 和 claim namespace；不会删除、覆盖、
重发或改写原始 control retry/treatment。

## 源码审计结论

机器真源是
[`phase1_fresh_c_v1v2_base_decel_measurement_rerun_queue_20260715.yaml`](../../../configs/phase1_fresh_c_v1v2_base_decel_measurement_rerun_queue_20260715.yaml)。

历史 source `312669c7bd61f8fc8f5ea99c8e94cfc3ffae9b94` 冻结了以下五个随挥重放计数：

| 语义 | TensorBoard tag |
| --- | --- |
| 缓冲未 ready 的 true reset | `Live/motion/post_swing_replay_buffer_not_ready_reset_count` |
| 缓冲 ready 后的 eligible reset denominator | `Live/motion/post_swing_replay_eligible_reset_count` |
| eligible 但随机未选中 | `Live/motion/post_swing_replay_random_not_selected_reset_count` |
| 被概率抽样选中的 numerator | `Live/motion/post_swing_replay_selected_reset_count` |
| root/joint 写入成功后的 started numerator | `Live/motion/post_swing_replay_started_reset_count` |

它们必须逐 update 满足 `selected + random_not_selected == eligible`、`started == selected`，且共同的
`post_swing_start_prob=0.25` realized `started/eligible` 在 `[0.20, 0.30]`。这只闭合随挥 replay；它不自动
证明 V1、V2 或 base-decel 被充分激活。

`312669c...` 的父提交是 `2171302...`，不是 main hardening `f00c497...` 的后代，因此仍只作历史 telemetry
reference。当前 YAML 已原子改绑 clean exact
`0f3900a612863faf326dca6ad3e8d38bfe8df3c9`（checkout
`/workspace/codexschema/nohope_p1_activation_successor_0f3900a`）。该 successor 同时包含 main hardening、五个
post-swing counters、V1/V2 execution counters、base-decel raw observer 和 runner logger；源码缺口已闭合，
但它尚未跑自己的 strict full-scene terminal probe，所以仍不是 launch-ready source，禁止原地修改 checkout。

## Successor 的最小 telemetry

| 机制 | eligible denominator | numerator | 当前结论 |
| --- | --- | --- | --- |
| V1 持拍手腕线速度释放 | `Live/motion/v1_velocity_mimic_eligible_sample_count` | `Live/motion/v1_held_wrist_excluded_sample_count` | 两者必须正且相等 |
| V2 击球窗模仿缩放 | `Live/motion/v2_strike_window_eligible_imitation_sample_count` | `Live/motion/v2_quarter_scaled_strike_window_imitation_sample_count` | 两者必须正且相等 |
| base-decel raw kernel | `Live/racket_target/base_decel_eligible_sample_count` | `Live/racket_target/base_decel_raw_kernel_nonzero_sample_count`；另记 `Live/racket_target/base_decel_raw_kernel_sum` | control/treatment 三项都必须正；nonzero 不得大于 eligible |

语义必须固定为每个 PPO update 内的**样本总数**，不能对 environment 求均值后再猜计数。V1 每个 eligible
environment-step 只计一次；只有 runtime body list 确实排除右手腕才计 numerator，因此 numerator 必须等于
denominator。V2 的单位是“一个 imitation RewardTerm × 一个 environment sample”：每条 non-None motion
imitation term 在 wide strike window 真正走到缩放 `torch.where` 就计 denominator，且 command setting 与函数
实际 scale 都严格为 `0.25` 才计 numerator，因此两者必须相等。base-decel 的 eligible 是
`pre_strike & ~in_hold`。Isaac 2.1 的 step 顺序是 reward → reset → command，因此 control 与 treatment 都由
RewardManager 内同一个 instrumentation term 先观察旧状态：该 probe 的 manager weight 是 `1.0`，但每环境
返回严格 `0`，不改变总 reward；treatment 随后的真实 `base_decel` RewardTerm 用同一
`common_step_counter` 去重。command stage 不再记账。故两臂的 raw nonzero 与 raw sum 都必须大于零；**不能**
要求 control 的 numerator 为零，因为这里量的是未乘 Reward weight 的真实 kernel opportunity。所有计数与 raw
sum 都要 finite、非负。

现有 `Live/Reward/base_decel` 是加权后的 step Reward 均值，control 权重为零时甚至可能不是 active term；
`Live/racket_target/base_speed_xy_prestrike` 在窗外写零后再取均值。二者都可以继续作为行为/Reward 量，但不能
替代 `pre_strike & ~in_hold` eligible denominator。

## 冻结的 replacement pair

| 字段 | 冻结值 |
| --- | --- |
| Source | clean exact `0f3900a612863faf326dca6ad3e8d38bfe8df3c9`；strict full-scene probe pending |
| 初始化/seed | fresh / `3`；只买一个 seed |
| 预算 | `4096 environments × 1001 updates`；每 `100` 保存；milestone `200/500/1000` |
| 动作/题库/plant | 与原配对逐字相同的 v4rg runtime-order 正反手、schema-3 rebound bank、zero-joint-friction 训练协议 |
| 共同机制 | V1=`true`；V2=`0.25`；post-swing replay=`0.25`；qdot hinge=`0`；conditional face=`0` |
| 唯一差异 | control `base_decel_weight=0.0`；treatment `base_decel_weight=1.0` |
| 调度 | 只允许 Pod2；control 优先 GPU1，treatment 优先 GPU2；Pod1 不 snapshot/claim/probe/launch |
| 权限 | `launch_authorized=false`；两格 `status=blocked`；第二 seed/judge/promotion 均未授权 |

fresh namespaces：

- control：`phase1_fresh_c_v1v2_base_decel_measurement_control_seed3_v3_20260715`
- treatment：`phase1_fresh_c_v1v2_base_decel_measurement_w1_seed3_v3_20260715`

`v3` 表示新的 measurement-complete namespace，不是第三个 seed，也不继承旧 checkpoint。

## Activation 先于行为早判

每个 milestone 必须先核对：五个 post-swing count 的算术闭合、V1/V2/base-decel denominator/numerator、
所有标量 finite、checkpoint filename=embedded iteration、fresh lineage、hard-contract/source/claim binding。
缺任一项，整个 milestone 直接记为 `invalid/instrumentation-blocked`，不得再解释速度、摔倒或击球精度。

只有 activation 全过后，才使用同 milestone 最后 21 updates 的配对均值：

| Milestone | 作用 | 底座速度 | pre-fall | 四项精度非劣 | 权限 |
| --- | --- | --- | --- | --- | --- |
| `+200` | 仪表与方向 | treatment/control `≤1.00` | 差 `≤+0.05` | 每项差 `≥−0.10` | 不 stop/promote |
| `+500` | 单 seed 因果 screen | treatment/control `≤0.90` | 差 `≤+0.02` | 每项差 `≥−0.05` | 不 stop/promote |
| `+1000` | 训练内终档诊断 | treatment/control `≤0.90` | 差 `≤+0.02` | 每项差 `≥−0.05` | 只记 candidate/reject/inconclusive |

四项精度分别是 racket position、racket velocity、**带符号** racket normal 和 strike composite pass rate，
逐项判，不能用 composite 或速度改善掩盖单项退化。control 速度接近零导致比值不稳定时记 inconclusive，
不得加任意 epsilon 制造通过。

## 运行表与当前决定

| 运行 | 状态 | 证据 | 有效性 |
| --- | --- | --- | --- |
| measurement control，base-decel 关 | blocked | exact successor + offline tests | strict probe 尚缺，不得发射 |
| measurement treatment，base-decel 权重 1 | blocked | 唯一 Reward 权重 delta | strict probe 尚缺，不得发射 |

- 决定：`inconclusive`；尚无 replacement runtime。
- 当前阻塞不是 Reward 负结果；source measurement contract 已闭合，runtime strict probe 仍未闭合。
- 本记录不建立算力优先级；是否排队仍只由 main 的 `docs/NOW.md` 统一队列决定。
- 不授权 Isaac/MuJoCo judge、第二 seed、正式 setting、部署或真机。

## 离线复现

本提交不连接 Pod、不写 claim、不运行 probe/trainer/judge：

```bash
python -m pytest -q hope_training/whole_body_tracking/tests/test_base_decel_activation.py
python -m pytest -q hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py -k base_decel
python -m pytest -q hope_training/whole_body_tracking/tests/test_training_launch_claim.py
pytest -q tests/test_phase1_fresh_c_v1v2_base_decel_measurement_rerun.py
pytest -q tests/test_run_lean_training_queue.py
git diff --check
```

专项测试证明：Pod2-only 且当前无 assignment、source/五 tag 精确绑定、三组缺失 telemetry fail-closed、
replacement 与原配方逐字一致且只改 base-decel 权重、fresh namespace 不复用旧证据、activation 明确先于
behavior screen。
