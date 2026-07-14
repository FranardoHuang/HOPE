# EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN — 补齐 activation 后重跑底座减速配对

- 状态：`completed / +500 activation-invalid`（两份 exact `model_500` 均通过身份门，但 control 在冻结
  `480–500` 窗仍未激活 post-swing；本 pair 已停止且不解释 base-decel 行为）
- 阶段/轴：Phase 1 fresh C；组合击球精度下，底座减速是否有净收益
- 集成小目标：保住击球精度信号，同时降低击球前底座速度与击球前摔倒率
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E2`（replacement pair 4096-env runtime 与 exact `model_200/500` receipt；无有效行为比较）
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

## `+200`：checkpoint 有效，但 post-swing activation 无效

两臂均在 trainer 继续运行时发布 no-clobber `model_200` receipt：

| Arm | checkpoint SHA-256 | receipt content SHA-256 | attestation 时状态 |
| --- | --- | --- | --- |
| control | `d065441bb96eb6bc9b5466eea5647dd03a7e881600034a0537e7c50559f4c77b` | `8561016ffb436dca463b6486ff5e2431d32b145b3e606211b91058362c20851d` | live |
| treatment W1 | `e1d2b43f0b608fa2015912641cdbceb909c8408da8032403a5799f5e480f4fb7` | `bf10ccf03d545dd1c07905b57bd89ea6bb8e6ed831c328b7e1f4748dbc53f4fb` | live |

两份文件均为 filename/embedded iteration=`200`、76 tensors、1,762,715 个浮点元素、nonfinite=`0`，
fresh lineage=`1`，并绑定同一 schema-3 hard contract `451cda47...2291` 与各自 launch claim。完整日志 fatal
扫描为零；只读快照时 control/treatment 已继续到 iteration `231/238`。

activation 算术如下：

- V1 两臂都满足 `eligible == excluded`，prefix 各 `19,759,104`，末 21 updates 各 `2,064,384`。
- base-decel 两臂的 eligible、raw nonzero 和 raw sum 都为正；末 21 updates control 为
  `463,231 / 463,231 / 189,693.86`，treatment 为 `505,785 / 505,785 / 314,143.20`。
- V2 在有样本处满足 `eligible == quarter_scaled`；control prefix=`4,555`、末 21=`245`，treatment
  prefix=`4,335`、末 21=`0`。末窗无样本不违反逐项等式，但不提供末窗行为归因。
- post-swing 两臂的 `buffer_not_ready` prefix 分别为 `1,810,024 / 1,814,343`；从 step 0 到 200，
  `eligible=random_not_selected=selected=started=0`。这违反预注册的 eligible/selected/started 必须为正，
  所以共同机制没有在 +200 得到可解释的启动样本。

因此 `+200` 的正式判定是 `invalid/instrumentation-blocked`，不是 base-decel control/treatment 的行为输赢。
trainer 不停、不重跑，也不解锁第二 seed；继续到 `+500` 只为判断 buffer 是晚激活还是 execution 路径仍有
缺口。若 +500 仍为零，先修 post-swing activation/缓冲根因，不能用 V1/base-decel 的正计数绕过共同机制门。

## `+500`：checkpoint 有效，但 matched activation 分叉

两份 no-clobber receipt 均在 trainer live 时发布：

| Arm | checkpoint SHA-256 | receipt content SHA-256 | `480–500` post-swing |
| --- | --- | --- | --- |
| control | `22f78f882397c48d1c8763186748935517d669cf4f36205baf69d07dc9e08a6a` | `67d76a2b0b60c6817fdc76ad32b6d1d641af361a7e2093dd6042436a0168e6d0` | not-ready=`24,646`；eligible/selected/started=`0/0/0` |
| treatment W1 | `a1735fbbaf82685a0ed9184dd726947aa757a0c142fdcbd447d40dbf3aa5bc14` | `e8cdfc87c1ac8c7da8818d58ededb01b9ec3b8f3d544cfc7c08c4ea32f8a1cf7` | eligible=`15,087`；not-selected=`11,337`；selected=started=`3,750`（`0.248558`） |

两份 checkpoint 都是 filename/embedded iteration=`500`、76 tensors、1,762,715 个浮点元素全 finite、
fresh lineage=`1`，并绑定同一 schema-3 hard contract `451cda47...2291` 和各自 claim；完整日志 fatal scan
为零。V1、V2 与 raw base-decel 在两臂各自有样本的地方均闭合。treatment 的 post-swing 算术也逐 update
满足 `selected + not_selected == eligible` 与 `started == selected`。

但 control 在**预注册冻结窗**没有共同 post-swing 分母，故 `+500` 仍是
`invalid/instrumentation-blocked`。control 到 step `519` 才首次出现 `eligible=1,077`、
`selected=started=282`（`0.26184`）；这个晚到证据不能倒灌覆盖 model-500 窗。

源码状态机解释了为什么这不是单纯“多等十九步”即可解决：buffer 只在 policy 活到自然 clip wrap 时写入；
跌倒/跟踪终止的 true reset 不写。base-decel treatment 本身改变了活到自然 wrap 的概率，于是本来声明为
两臂共同的 post-swing curriculum 反而成了 treatment 的下游变量。两臂首次 ready 时刻因此不是外生一致的，
从第一次一侧 ready 起就不能再把差异纯归因于 base-decel。下一版必须让两臂在训练前消费同一份、绑定自然
wrap provenance 的 immutable teacher-state receipt，并在首个科学 update 前 fail closed；不能用任意 timeout
状态伪造随挥结束。

两臂发布 +500 receipt 后，operator 对 exact PID=PGID/starttime 发出 `TERM`；Kit/Python 忽略该信号并继续
推进到 `564/573`。再次核验同一进程组身份后只对这两个 PGID 发出 `KILL`，最终 GPU1/GPU2 均为空，GPU0
的 Yikang 进程未变化。没有 `model_600`，所以有效保存点仍止于 model-500；不重发、不买第二 seed、不判卷。

## 外生 teacher cold-start 源码修复（尚未解锁训练）

红队否决首版 receipt loader：receipt 与 NPZ 都是“先 hash、再重新按路径读取”，存在 rename/rewrite
TOCTOU；`np.load(path)` 还会把重复 ZIP key 折叠。更关键的是首版 `teacher` 字段可自行声称 checkpoint、
schema-3/fresh lineage 和 natural-wrap provenance，没有一个真实 producer/consumer 去核对这些字节，因此不能
给 replacement pair 用。

修正版改为两段式 source gate：capture 由 `MotionCommand` natural-wrap 路径直接读取 live articulation tensors，
没有 module-global capability、公开 writer 或接收任意 arrays 的签发接口；配置时先占有 `O_EXCL` claim fd，随后以
CUDA→CPU 同步和 atomic no-clobber 发布 state NPZ 与 raw result。artifact 只绑定 reviewed source/claim/contract，
不把自报 callback label 当密码学证明。独立 one-shot attestor 以 `weights_only=True` 核对 actual checkpoint bytes、内嵌
schema-3/fresh-lineage/launch claim、相邻 hard contract、checkpoint 与 capture 两份 clean source、实际 motion
bytes、runtime joint order 和速度 limits；只有全部相等才发布 trainer receipt。trainer 会同时重读 receipt、
raw result 和 NPZ，三者不能靠自报字段互相替代。详细合同与命令见
[teacher artifact interface](../../interfaces/post_swing_teacher_artifact.md) 和
[producer operation](../../operations/run_post_swing_teacher_capture.md)。

消费端对 receipt/NPZ/raw result 均使用一次 `O_NOFOLLOW` open + 同 fd 前后 `fstat` + 单 immutable bytes；SHA、
JSON、ZIP inventory 和 `np.load(BytesIO)` 不再重开路径。ZIP 重名、JSON bool 冒充 int、float/int coercion、
source/hash/lineage 漂移、旧 public-writer/callback-label forgery、恶意 pickle、重复发布都有负测。root linear/angular norm 与逐关节 velocity limit
在 attestor 和 trainer 两次检查；首 reset 由“selected>0”收紧为预注册 adopted count/fraction、概率偏差以及可选
强制 simulator state readback。receipt-free default 通过 canonical contract byte-equivalence 回归，不给旧 hard
contract 添加 null/default 字段。

当前只完成 source gate：dependency-light 专项 `13 passed`、`py_compile` 与 `git diff --check` 通过；本机没有
Torch/Isaac，因此含真实 `MotionCommand`、4096 environments、GPU writer/readback 的测试尚未执行。状态保持
`Partial`、`launch_authorized=false`；必须等修复合入 `main` 后在 clean Pod 做 inference capture、one-shot
attestation 与 4096-env 首 reset probe，才可另建 replacement pair。不能用现役 clean base-deceleration pair 的
checkpoint 或曲线倒推 teacher 功能已通过。

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
| measurement control，base-decel 关 | stopped after +500 | PGID `380610` 已空；model500 `22f78f88...a6a` | 冻结窗 post-swing 分母为零；invalid |
| measurement treatment，base-decel 权重 1 | stopped after +500 | PGID `381237` 已空；model500 `a1735fbb...c14` | 自身 activation 闭合，但不能单臂解释 |

- 决定：`+500 invalid/instrumentation-blocked`；source/checkpoint 有效，但共同 curriculum 的 ready 时刻被
  treatment 影响，不看行为差异；pair 已停止，等待 immutable teacher-state cold-start 修复。
- `0f3900a` 的 runtime logger 已证伪；`2c2d70d` 只解锁同配方单 seed pair，不追认任何 Reward 效果。
- 本记录不建立算力优先级；是否排队仍只由 main 的 `docs/NOW.md` 统一队列决定。
- 不授权 Isaac/MuJoCo judge、第二 seed、正式 setting、部署或真机。

## 外生 teacher capture 的冻结实例

source gate 合入后，首个真实采集实例在任何输出前冻结于
[`phase1_post_swing_teacher_capture_prereg_20260715.json`](../../../configs/phase1_post_swing_teacher_capture_prereg_20260715.json)。
teacher 选择本卷 measurement control 的 fresh exact `model_500`（SHA-256 `22f78f88...a6a`），因为它的
训练合同已绑定 `post_swing_start_prob=0.25`，且 model/claim/binding/schema-3 hard contract 已通过身份门。
采集只允许 Pod2 GPU1、4096 environments、4096 条 natural-wrap live state、最多 20000 inference steps；
root linear/angular velocity 上限分别冻结为 `2.0 m/s` / `4.0 rad/s`。两条动作、题库、ignored A3 tree 与
四个关键 source 文件都按 exact bytes/SHA 绑定，输出 namespace 必须初始不存在且只可占有一次。

这个预注册只把 `capture_authorized` 改为 true；attestation 必须等完整 capture 后另跑一次，首 reset probe、
科学 pair、第二 seed、judge、promotion 和硬件仍全部未授权。timeout/partial/失败 reset 不能补齐样本，也不能
复用同一 namespace 重发。

实际 v1 在创建 capture directory/claim/process 前的 Hydra compose fail closed：训练 argv 派生时虽然删除了
`checkpoint_path`，却遗漏 `checkpoint_tolerant`、`checkpoint_allow_missing_contract` 与
`checkpoint_allow_contract_mismatch` 三个只存在于 train 配置的键，play structured config 以 rc1 拒绝。
进一步 source 审计又发现 play 没有 `seed` 字段，也没有把冻结 training seed 赋给 env/runner。完整证据 SHA
见 [`phase1_post_swing_teacher_capture_attempt_v1_result_20260715.json`](../../../configs/phase1_post_swing_teacher_capture_attempt_v1_result_20260715.json)。
v1 不重发；successor 必须先合 seed parity，删全 train-only 键，并使用全新 source/output/launch namespace。

seed parity 已于 2026-07-15 单独闭环：`play.yaml` 现在显式提供 `seed`，`play.py` 只接受
`[0, 2^32-1]` 的 plain integer，并在 `gym.make` 前把同一个 seed 写入 environment 与 PPO runner。
真实 Hydra compose 逐项证明三个 train-only checkpoint 键只要残留就 fail closed，净化后的 seed-3
capture 配方可 compose。这个 source fix 不追认 v1，也不授权 v2；一次性 controller 的独立红队、全新
schema-2 prereg、4096-environment capture、attestation 与首 reset 仍按顺序阻断。

为避免再次靠聊天手拼 argv，successor source 新增
[`run_preregistered_post_swing_capture.py`](../../../scripts/run_preregistered_post_swing_capture.py)：它从 exact
run binding 机器派生 play argv，强制删除十二个 train-only/ownership 键、保留并核对 seed，先 compose、
后二次复算 source/input，再创建 output 和独立 PGID。任何失败都会花掉 launch plan，工具不提供 retry、
stop、SSH 或 trainer 功能。当前仍只是 dependency-light source gate；独立红队、全新 schema-2 prereg、
4096-environment capture、attestation 与首 reset 未闭前不能用它解锁科学训练。

controller 的独立红队随后把九类 blocker 收口到 schema-v2：固定 direct namespace/dirfd no-follow；历史
teacher train entry 与当前 play entry 分离；claim→binding→model-500 milestone receipt→checkpoint/hard-contract
逐层交叉绑定；Pod2 hostname/machine-id/boot-id 与 physical GPU2 UUID；共享 GPU2 lease；absolute byte-bound
Git/`nvidia-smi`；safe environment allowlist；`--resolve` compose timeout 与全阶段 failure evidence；same-PID
exec handoff；status 的 symlink/zombie/PID reuse/receipt-rebinding 拒绝。离线 builder 已可生成并回读同一 validator
接受的 plan，focused 为 `40 passed, 4 skipped`。本分支未连接 Pod；V10 已占 GPU1，因此任何 successor 只可
另冻 GPU2 新 namespace，旧 GPU1 v1 仍不重发。完整 source/Isaac import roots 本机时序为 119 MB/2432 files
约 `0.709 s`，9.8 MB/852 files 约 `0.284 s`；这不是 Pod 冷缓存数字，也不声称覆盖 venv/stdlib/native/rootfs。

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
