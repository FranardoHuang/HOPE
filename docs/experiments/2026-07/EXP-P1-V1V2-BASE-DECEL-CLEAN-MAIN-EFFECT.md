# EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT — 关闭随挥重放后只测底座减速

- 状态：`completed / treatment rejected；no second seed`
- 阶段/轴：Phase 1 fresh C；V1+V2 下的 base-decel 单变量主效应
- 集成小目标：判断底座减速本身能否降低击球前底座速度和摔倒，同时不伤四项击球精度
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E2`（两份 exact model-1000 receipt 与独立终档审计；没有物理回球）
- 创建日期/最后复核日期：2026-07-15 / 2026-07-15

共享术语见[术语与人话对照](../../DEFINITIONS.md)。V1 是释放持拍手腕线速度模仿，V2 是在击球窗把
动作模仿缩放到四分之一，base-decel 是“击球前按距离递减底座目标速度”的 Reward。机器真源是
[`phase1_fresh_c_v1v2_base_decel_clean_main_effect_queue_20260715.yaml`](../../../configs/phase1_fresh_c_v1v2_base_decel_clean_main_effect_queue_20260715.yaml)。

## 为什么这不是失败 pair 的原地重跑

[上一对](EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)在 model-500 证明 post-swing buffer 的 ready 时刻
并非共同外生条件：它只收 policy 自己活到自然 clip wrap 的状态，而 base-decel treatment 会改变活到 wrap
的概率。control 在冻结窗分母为零，treatment 已激活，所以把后续差异归给 base-decel 会同时混入“谁更早
拿到随挥重放课程”。

本实验不修补或复用那个 buffer，而是把两臂 `post_swing_start_prob` 都冻结为 `0.0`，先回答更小也更干净的
问题：在同一个 V1+V2、seed、动作、题库、plant、PPO 和预算下，只把 base-decel 从 `0` 改为 `1`，有没有
净收益。连续恢复与 post-swing interaction 仍是后续独立轴；它必须消费两臂相同的 immutable natural-wrap
teacher receipt 后另跑，不能用本卷冒充。

## 冻结配方

| 项目 | 两臂共同值 |
| --- | --- |
| source | clean exact `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e` |
| 初始化 | from scratch，seed `3` |
| 训练 | 4096 environments、1001 updates、每 100 保存，milestone `200/500/1000` |
| 动作/题库 | signed-face v4rg 正反手动作与 schema-3 train bank |
| V1/V2 | `free_wrist_vel_mimic=true`；`motion_scale_in_window=0.25` |
| post-swing | `post_swing_start_prob=0.0`；五个 activation counter 每 update 必须全零 |
| 资源 | 只用 Pod2；control 硬绑 GPU1，treatment 硬绑 GPU2；GPU0 只归 Yikang |

唯一科学差异：control 的 `base_decel_weight=0.0`，treatment 为 `1.0`。两个新 job id、run name、run
directory 和 claim namespace 与所有旧 run 不相交；旧 checkpoint、曲线和行为结论都不继承。

## source/scene 门为何可复用

exact `2c2d70d...` 已在同一 Pod2 GPU1 通过 4096-env full-scene 自然终档，并在上一对实际产出 finite
model-200/model-500。队列只复用该 receipt 对 source、A3 ignored asset、physical-ball/table scene、face179、
零摩擦 plant、checkpoint wiring 和 Kit boot 的证明；明确写
`scientific_recipe_identity_reused=false` 与 `post_swing_setting_covered_by_probe=false`。本卷不会把 probe 或
旧 pair 当作 Reward 证据。每个新 job 仍须在 claim 前重新跑最终 argv 的 no-Kit compose、检查 source clean
exact 和新 run-dir absence，并越过自己的首迭代。

## activation 与早判

每个 milestone 先做 checkpoint filename=embedded iteration、finite、fresh lineage、claim、相邻 schema-3
hard contract 和 fatal scan。随后必须满足：

- post-swing 的 buffer-not-ready/eligible/not-selected/selected/started 五个计数**每个 update 全零**；任一非零
  说明 disabled 合同失效，整个 milestone invalid；
- V1 `eligible == excluded > 0`；
- V2 `eligible == quarter_scaled > 0`；
- 两臂 base-decel raw eligible、nonzero 与 sum 都为正；weighted Reward 只允许 treatment 非零，不能用
  weighted mean 代替 raw activation。

只有 activation 全过，才比较相同 milestone 最后 21 updates：

| milestone | base speed treatment/control | pre-fall 差 | 四项精度逐项非劣 | 动作 |
| --- | ---: | ---: | ---: | --- |
| +200 | `≤1.00` | `≤+0.05` | `≥−0.10` | 方向检查，不 stop/promote |
| +500 | `≤0.90` | `≤+0.02` | `≥−0.05` | 单 seed screen |
| +1000 | `≤0.90` | `≤+0.02` | `≥−0.05` | 只记 candidate/reject/inconclusive |

四项精度是位置、速度、**带符号**拍面和 composite pass rate；不能互相补偿。训练内解析 return 只作诊断，
不替代物理球或 vendor MuJoCo。首个 seed 过门前不买第二 seed，不启动 judge、交互、晋级、部署或真机。

## 预运行验收

```bash
python3 -m pytest -q tests/test_phase1_fresh_c_v1v2_base_decel_clean_main_effect.py
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_v1v2_base_decel_clean_main_effect_queue_20260715.yaml plan
git diff --check
```

实际 launch 只能由 queue 的一次 `fill --count 2` 顺序事务完成；先让 GPU1 control 越过首迭代，再发 GPU2
treatment。timeout 标 unknown，不得因 SSH 抖动重复 claim 或重发。

## 运行证据（首个 checkpoint 前）

2026-07-15 04:27 CST，队列按上述顺序在 Pod2 成功发射两条 fresh run；Pod1 与 Pod2 GPU0 均未进入
Codex 发射路径：

| arm | GPU | exact PID=PGID | queue-claim SHA-256 | 只读复核时最新 update |
| --- | ---: | ---: | --- | ---: |
| control，底座减速关闭 | 1 | `385320` | `a039226a...1746e` | 106（日志已越过 91；TensorBoard 已写到 106） |
| treatment，底座减速权重 1 | 2 | `385948` | `673bf6c6...9392` | 89（日志已越过 73；TensorBoard 已写到 89） |

两进程的 `/proc` 身份与 launch sidecar 的 start-time 一致；归档训练 checkout 仍为 clean exact
`6d93bcb...80b`，实际训练源码为 clean exact `2c2d70d...607e`。GPU0 只有 Yikang 的 PID `379550`；
GPU1/GPU2 分别只有本表一条 Codex trainer。两份完整日志的 fatal 扫描均为零，host 可用内存约
`965 GiB`、无 swap。

TensorBoard 的第一轮在线合同核验通过：两臂
`buffer_not_ready/eligible/not_selected/selected/started` 五个 post-swing 计数在**每个已写 update**均为零；
两臂 base-decel raw eligible/nonzero/sum 均为正，control weighted Reward 为零，treatment weighted Reward
非零。此处只证明禁用合同和 Reward 路径按预注册执行，不比较行为；首个可比较点仍是 exact
`model_200` receipt。

## `model_200` 与 `+200` 早判

两份 no-clobber receipt 已分别发布并由独立只读审计复算：

| arm | checkpoint SHA-256 | receipt content SHA-256 | common hard contract |
| --- | --- | --- | --- |
| control | `6cb55718...94f1` | `5847f050...ce52` | `ca57a94f...cc2e` |
| treatment | `d61998ac...6892` | `9a42dc25...f49b` | `ca57a94f...cc2e` |

两边 filename iteration 与 embedded iteration 都是 `200`，76 tensors / 1,762,715 floating elements
全 finite，fresh lineage=`1`，claim、binding 与 schema-3 hard contract 都 exact；trainer 仍保持原
PID=PGID 存活，日志 fatal=0。

step 0–200 的每个 activation tag 都恰有 201 点。两臂五个 post-swing counter 每点严格为零；V1
eligible/excluded 每点严格相等且为 `98,304`。V2 eligible/quarter-scaled 在各臂内每点相等、累计为
control `4,555`、treatment `4,335`；但 treatment 的 180–200 尾窗为零，因此这里只满足预注册的
milestone 累计语义，不能声称尾窗持续激活。base-decel raw eligible/nonzero/sum 两边每点为正；control
weighted Reward 每点为零、treatment 每点非零。

冻结的 step 180–200（每项 21 点）结果：

| 指标 | control | treatment | 差异 | +200 判定 |
| --- | ---: | ---: | ---: | --- |
| 击球前底座速度 | `0.71340` | `0.75008` | `1.05142×` | **FAIL**，要求 `≤1.00×` |
| pre-strike fall rate | `0.999981` | `1.000000` | `+0.000019` | 仅数值过 `≤+0.05` |
| 位置/速度/signed-face/composite pass | 全 `0` | 全 `0` | `0` | 空洞非劣，不是成功 |
| 解析合法回球 | `0` | `0` | `0` | 无可读行为信号 |

因此 `+200` 方向门失败：treatment 没有降低底座速度，反而高 `5.14%`；两边精度零对零，pre-fall 又
接近 100%，不能把非劣阈值的数值通过解释成行为收益。按冻结规则，`+200` 不 stop/promote；trainer
继续到 `+500` 只为判断是否晚熟翻转，不买第二 seed、不 judge、不晋级。

## `model_500` 与单 seed 因果 screen

control/treatment checkpoint SHA-256 分别为 `3b67962f...9020` / `7cf137fe...0065`，receipt content
SHA-256 为 `ba4b92c6...7dd6` / `a717e246...55c0`。两边 filename=embedded `500`、1,762,715 floating
elements 全 finite、fresh lineage/claim/common hard contract `ca57a94f...cc2e` exact，trainer identity 仍
匹配且 fatal=0。

step 0–500 的 501 点 activation 全过：五个 post-swing counter 两臂每点全零；V1/V2 的 numerator 与
denominator 逐点相等；base raw 三项每点为正；control weighted Reward 每点为零、treatment 每点非零。
480–500 尾窗两臂 V2 也都有充分样本，不再有 +200 treatment 尾窗空洞。

| 480–500 指标 | control | treatment | 差异 | +500 门 |
| --- | ---: | ---: | ---: | --- |
| 击球前底座速度 | `0.479838` | `0.545428` | `1.13669×` | **FAIL**，要求 `≤0.90×` |
| pre-strike fall rate | `0.616979` | `0.584106` | `−0.032873` | PASS |
| position pass | `0.621397` | `0.580054` | `−0.041343` | PASS，要求 `≥−0.05` |
| velocity pass | `0.296915` | `0.403085` | `+0.106170` | PASS |
| signed-face pass | `0.267027` | `0.100937` | `−0.166089` | **FAIL** |
| composite pass | `0.088143` | `0.018723` | `−0.069420` | **FAIL** |
| 解析合法回球 | `0.247710` | `0.122821` | `0.49583×` | 诊断明显退化 |

这次不是零分母假象：尾窗 decayed exact-strike count 为 `1915.25/1466.06`。当前 weight=`1.0` 的
base-decel 虽降低 fall、提高速度 pass，却让底座更快、signed face 与 composite 过门失败，并把解析回球
压到约一半，故 +500 single-seed screen **reject treatment**。队列冻结 `stop_or_promote_allowed=false`，
所以两臂继续到 +1000 只收 terminal diagnostic；拒绝结论不买第二 seed、不 judge、不做交互或晋级。

## `model_1000` 终档与量尺语义纠偏

两臂均自然退出，原 PID=PGID `385320/385948` 已不存在；完整日志各有 1001 个 update、fatal=`0`。
control/treatment checkpoint SHA-256 分别为 `1237a984...d4d5` / `ddf9947a...8393`，receipt content
SHA-256 为 `15c7f970...5c1d` / `9fad83b5...bb47`。两边 filename=embedded iteration=`1000`、
1,762,715 个 floating elements 全 finite、fresh lineage=`1`，claim/binding 与共同 schema-3 hard contract
`ca57a94f...cc2e` 独立复算 exact。GPU1/GPU2 自然释放；GPU0 的 Yikang 进程未触碰。

step 0--1000 的 activation 合同完整闭合：五个 post-swing counter 每点全零；V1 numerator=denominator；
V2 numerator=denominator；base-decel raw eligible/nonzero 每点为正。control 的 weighted base-decel Reward
1001 点全零，treatment 1001 点全非零。

| 980--1000 指标 | control | treatment | 差异 | +1000 门 |
| --- | ---: | ---: | ---: | --- |
| 击球前底座速度 | `0.208314` | `0.210152` | `1.00882x` | **FAIL**，要求 `<=0.90x` |
| pre-strike fall rate | `0.087505` | `0.096135` | `+0.008630` | PASS |
| position pass | `0.856432` | `0.842119` | `-0.014314` | PASS |
| velocity pass | `0.465879` | `0.454219` | `-0.011660` | PASS |
| signed-face pass | `0.494282` | `0.476421` | `-0.017860` | PASS |
| composite pass | `0.280008` | `0.265502` | `-0.014506` | PASS |
| 解析合法回球 | `0.451768` | `0.436228` | `0.96560x` | 诊断退化 |

因此按结果前冻结的 raw-speed 判据，终档正式结论仍是 **REJECT treatment**；不事后改门、不买第二
seed、不 judge、不晋级。

但源码复核同时纠正了本卷对 Reward 的人话表述。实际实现是
`v_des=clamp(2*planar_racket_target_error, 0, 1.6)`，并奖励
`exp(-(v_base-v_des)^2/0.4^2)`；它要求远离目标时移动、临近目标时减速，并非任何时刻都让 raw speed
更低。当前 primary metric 却把所有 pre-strike（还包含 hold）的 raw speed 零掩码后取均值，既不测
`|v_base-v_des|`，也没有按距离分桶。980--1000 的 raw-kernel-per-eligible 实际从 `0.475268` 提高到
`0.760572`（`1.6003x`）。这不能翻转冻结 verdict，却说明“raw speed 未下降”不能单独证明实现的
pseudo-speed tracking 失败。若继续此机制，必须先另行预注册 `|v_base-v_des|` 的近/中/远距离分桶与
近目标速度，再用新的单 seed 配对；当前 weight=`1` 不直接复制。
