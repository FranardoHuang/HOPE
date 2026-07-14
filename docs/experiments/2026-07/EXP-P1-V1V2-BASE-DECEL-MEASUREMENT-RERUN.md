# EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN — 补齐 activation 后重跑底座减速配对

- 状态：`running`（`0f3900a...` 的 probe 抓到 inference-counter logger 真 bug；修复源码 exact
  `2c2d70d...` 的 fresh strict probe 已通过，单 seed v4 pair 已越过首迭代）
- 阶段/轴：Phase 1 fresh C；组合击球精度下，底座减速是否有净收益
- 集成小目标：保住击球精度信号，同时降低击球前底座速度与击球前摔倒率
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E2`（4096-env probe 越过首个 update 后按真实异常失败；没有本 replacement pair runtime）
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
`0f3900a612863faf326dca6ad3e8d38bfe8df3c9`。该 successor 同时包含 main hardening、五个 post-swing
counters、V1/V2 execution counters、base-decel raw observer 和 runner logger；measurement 源码表面完整，
但仍含跨 InferenceMode consumer bug。它自己的 strict full-scene attempt 已在首个 update 后失败，所以永久
不是 launch-ready source，禁止原地修改 checkout 或复用旧 attempt。

当前 replacement 已改绑 clean exact `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e`（checkout
`/workspace/codexschema/nohope_p1_activation_successor_2c2d70d`）。它只在消费后 reset 三个私有 scalar 时回到
inference mode，未改 Reward、scene、题库或 optimizer。资源隔离由 main
`8b0a08414aef390d3b45664c2cd3746e87453fff` 的
[`required_slot`](../../DEFINITIONS.md#required-slot) 合同提供；source/queue 已精确绑定，但 fresh strict probe
现已通过，终档证据见下节；科学 pair 只因此解锁单 seed 训练，不解锁 judge、第二 seed 或晋级。

## `2c2d70d` fresh strict probe 通过

唯一 attempt `inferencefix_2c2d70d_pod2_gpu1_a1` 使用 Pod2 GPU1、4096 environments 和与 control
逐字相同的完整 scene recipe，完成两个 update 后自然退出。exact PGID `378694` 已为空；finalizer 重新核验
physical ball 与 `pb_ball/pb_table/pb_table_visual`、face179、31/31 零关节摩擦、schema-3 hard contract、
source/ignored asset closure、checkpoint filename=embedded iteration=`1`，以及 76 个 tensor 的
1,762,715 个浮点元素全部 finite。fatal 计数全零。

- `probe_result.json` file SHA-256：`4b12854c5deca075ddf886fea3c5806aa0838b1d2bc9d3739e2fa13cd1840b27`
- result content SHA-256：`4cbc9fc0bf7a5e5bdc5dfaa06386463e325dab141945344d0b0064b1b55fb083`
- claim content SHA-256：`52298bf11cb16e11cd67a198ca713c542423d576d1052e4178e42868b9bcfb9f`
- model/hard-contract SHA-256：`68d9809b...8713` / `451cda47...2291`

queue 以这些 exact 字段把 `launch_authorized` 改为 true、两臂改为 ready；receipt 只授权这对 fresh v4
science namespace，不是行为成绩。

## Science pair 已顺序点火

同一次 `fill --count 2` 先让 control 的 embedded preflight 与首迭代通过，再发 treatment；两臂都没有
fallback 到 GPU0：

| Arm | Pod/GPU | PID=PGID | claim content SHA-256 | 最近只读迭代 |
| --- | --- | --- | --- | --- |
| control，base-decel 关 | Pod2 GPU1 | `380610` | `576724deeeaf2386005e804fc5f2dc5600a7c63d104cb8f7c8f1f2f5131ba49d` | `25/1001` |
| treatment，base-decel 权重 1 | Pod2 GPU2 | `381237` | `1a529430eda6f5d69636b8992fec7499867a7933760c231e6cd599b33dccd4c5` | `11/1001` |

两份 trainer-owned binding 都指向 clean exact `2c2d70d...`、4096 environments、fresh checkpoint、相同
schema-3 hard contract；最近日志 fatal scan 为零。treatment 的普通 weighted `base_decel` 已非零、control 为
零，但这还不能替代 count-level denominator/numerator；必须等 `model_200` 后先 attest checkpoint，再从
TensorBoard event 做 activation 算术闭合，最后才看行为差异。

## `0f3900a` strict probe 的两次不可覆盖负结果

Pod2-only control probe 的 `a1` 在 trainer/Kit 启动前等待旧 `/workspace/.kit_boot.lock`。根因是一康仍在
GPU0 运行的旧 launcher 把锁 fd 继承给 trainer；这不是一康训练失败，也不能触碰其进程。operator 只对我们
自己的 exact PGID 写入 no-clobber identity/TERM/result 收据并清空残留，随后以“旧 inode 硬链接保全 + 新文件
原子 replace canonical path”换代锁。旧 inode 仍只由一康原 PID/启动时刻持有，新 canonical inode 的
nonblocking exclusive lock 通过；`a1` 永久记为 `trainer_started=false / scientific_result=false`，没有同 attempt
重试。

全新 `a2` 使用同一 canonical lock、exact source、GPU1、4096 environments 和完整 scene recipe，成功写 binding、
越过第一个 `Learning iteration`，随后 runner 第一次消费 base-decel activation ledger 时抛出：

```text
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed
```

RewardManager 在 `torch.inference_mode()` 内首次创建三个 device scalar，runner logger 在 normal mode clone 后对原
scalar `zero_()`，因此真实 Isaac 路径会失败，而原 synthetic test 都在 normal mode，造成假绿。修复只把私有
counter 的 reset 放回 `torch.inference_mode()`；snapshot、device/dtype、last-step dedup token、Reward 与 simulator
状态均不变。新增回归明确覆盖“inference mode 创建/累计 → normal logger consume/reset → 下一 simulator step 再累计
→ 再 consume”。本提交的离线专项为 activation `10 passed`、base-decel MDP `2 passed`、launch claim
`11 passed`；这仍只是 source fix，不追认 `a2`，也不授权当前队列。

`a2` 抛出上述不可逆 fatal 后，Kit teardown 连续 15 分钟无日志变化，原 PGID 仍占 GPU1。operator 先以
O_EXCL 收据冻结 leader/child 启动时刻、claim/binding/log SHA、fatal 与 exit/result absence；双扫描确认 PGID
只含本 probe 的 supervisor/trainer 后，仅向该 exact PGID 发 TERM 并证明 residual=0。一康 PID/启动时刻未变。
本路径没有伪造 `full_scene_probe_exit.json` 或 `probe_result.json`，因此 `a2` 是“真实 fatal + operator cleanup”，
不是 terminal probe result，也永远不可 finalize、解锁或同 attempt 重试。

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
| 失败 source | clean exact `0f3900a612863faf326dca6ad3e8d38bfe8df3c9`；fatal evidence 冻结，永久 NO-LAUNCH |
| replacement source | clean exact `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e`；strict terminal probe passed |
| 初始化/seed | fresh / `3`；只买一个 seed |
| 预算 | `4096 environments × 1001 updates`；每 `100` 保存；milestone `200/500/1000` |
| 动作/题库/plant | 与原配对逐字相同的 v4rg runtime-order 正反手、schema-3 rebound bank、zero-joint-friction 训练协议 |
| 共同机制 | V1=`true`；V2=`0.25`；post-swing replay=`0.25`；qdot hinge=`0`；conditional face=`0` |
| 唯一差异 | control `base_decel_weight=0.0`；treatment `base_decel_weight=1.0` |
| 调度 | 只允许 Pod2；control 硬绑定 GPU1，treatment 硬绑定 GPU2；GPU0 与 Pod1 均不得接收 Codex 作业 |
| 权限 | `launch_authorized=true`；两格 `status=ready`；第二 seed/judge/promotion 均未授权 |

fresh namespaces：

- control：`phase1_fresh_c_v1v2_base_decel_measurement_control_seed3_v4_20260715`
- treatment：`phase1_fresh_c_v1v2_base_decel_measurement_w1_seed3_v4_20260715`

`v4` 表示 inference-reset 修复后的 fresh namespace，不是第四个 seed，也不继承旧 checkpoint。

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
| measurement control，base-decel 关 | running | PGID `380610`；claim `576724de...a49d` | 等 `model_200` attestation + activation |
| measurement treatment，base-decel 权重 1 | running | PGID `381237`；claim `1a529430...4c5` | 等 `model_200` attestation + activation |

- 决定：`inconclusive`；source/scene 门已过且 pair 正在运行，尚无有效 milestone 结论。
- `0f3900a` 的 runtime logger 已证伪；`2c2d70d` 只解锁同配方单 seed pair，不追认任何 Reward 效果。
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

专项测试证明：Pod2-only 且当前无 assignment、source/五 tag 精确绑定、两臂分别 hard-bound GPU1/GPU2、
三组缺失 telemetry fail-closed、replacement 与原配方逐字一致且只改 base-decel 权重、fresh namespace 不复用
旧证据、activation 明确先于 behavior screen。
