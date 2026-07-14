# EXP-P1-V1V2-BASE-DECEL-INTERACTION — 组合击球精度下的底座减速是否仍有净收益

- 状态：`Partial / +200 activation-invalid`（strict probe 与 paired `model_200` 身份门已过；
  V1/V2/base-decel 的计数级 denominator/numerator 仪表不完整，不得解释 Reward 因果效果）
- 阶段/轴：Phase 1 fresh C；V1+V2 与击球前底座减速的组合效应
- 集成小目标：保住 V1+V2 的击球精度信号，同时降低击球前底座速度和摔倒率
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E2`（strict Pod2 启动/终档、paired checkpoint 身份与冻结方向曲线；
  activation 合同未闭合）
- 创建日期/最后复核日期：2026-07-14 / 2026-07-14

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的 V1 指“在线速度模仿中释放持拍手腕”，
V2 指“在击球窗口内把动作模仿缩放到四分之一”；`base-decel` 指“接近击球点时奖励底座平滑减速”。
本实验不是重新扫描三个 reward，而是在已经出现击球精度信号的 **V1+V2 组合**上，只问是否应再加入
底座减速。

## 问题与假设

Fresh C 首轮的 V1+V2 在 `+500` 出现 composite/法向和三项误差的强精度信号，但 pre-fall（击球前摔倒）
仍高；单独 base-decel 则让 `base_speed_xy_prestrike`（击球前底座平面速度）比对照低约 13.6%，同时有
精度退化。两项可能互补，也可能互相争夺同一时段的动作控制预算。

可证伪假设：在 V1+V2 完全相同的条件下，把 `base_decel_weight` 从 `0.0` 改为 `1.0`，能让
`base_speed_xy_prestrike` 至少下降 10%，pre-fall 不恶化超过 2 个百分点，同时位置、速度、带符号拍面法向和
composite 四项击球精度通过率各自不下降超过 5 个百分点。任何 activation（机制实际作用覆盖）不成立时，
这不是 reward 负结果，而是无效运行。

## 冻结的 setting

机器真源是
[`phase1_fresh_c_v1v2_decel_interaction_queue_20260714.yaml`](../../../configs/phase1_fresh_c_v1v2_decel_interaction_queue_20260714.yaml)。

| 字段 | 冻结值 |
| --- | --- |
| Source | `main@caeb9ad` / `/workspace/codexschema/nohope_p1_caeb9ad`；含 runtime binding、milestone attestor 与 strict full-scene probe |
| 初始化/seed | fresh / `3`；只买一个 seed |
| 预算 | `4096 environments × 1001 updates`，每 `100` 保存，配对检查 `200/500/1000` |
| 动作 | v4rg 正反手 runtime-order 动作；shared-face signed action |
| Bank/exam | schema-3 rebound train bank；同 family 的 immutable K100 exam 只绑定身份，本实验不授权 judge |
| Plant | `zero_joint_friction=true` 的当前可重放训练协议；不声称是部署 plant |
| 共同机制 | V1=`true`，V2=`0.25`，post-swing start=`0.25`，qdot hinge=`0.0`/margin=`0.85`，conditional face=`0.0` |
| 唯一差异 | control `base_decel_weight=0.0`；treatment `base_decel_weight=1.0` |
| 调度 | 只允许 Pod2；Pod1 仅保留在 harness 固定 schema 中，`dispatch_pods: [pod2]` 不会给它发新任务 |

strict receipt 已被显式队列变更消费，顶层
[`launch_authorized=true`](../../DEFINITIONS.md#launch-authorized)。首次 control 旧行现为 `rejected`，不再
发射；配方逐字不变的 `control_retry_v2` 与从未 claim 的 treatment 随后由同一
`fill --count 2` 顺序发射。两份 `model_200.pt` 均已由 milestone attestor 验证
filename/embedded iteration=`200`、76 tensors / 1,762,715 浮点元素 finite、fresh lineage=1 与
schema-3 hard contract `451cda47...2291`。control/treatment checkpoint SHA-256 分别为
`44a709ac...035a` / `b04e2338...e56b`，receipt content SHA-256 为 `ad47c826...4d1f` /
`49234348...7748`。probe 的 model/reward 永不进入实验成绩。

## Activation 与配对早判

所有比较都使用同 milestone 最后 21 个 updates 的配对均值，不能拿不同 checkpoint 或单点互比。开始算
reward 效果前必须先闭合以下 activation：

1. 两格都必须证明 V1 的 eligible denominator（可参与线速度模仿的样本数）大于零，且持拍手腕被排除的
   numerator 大于零；只看到配置回显不够。
2. 两格都必须证明 V2 的击球窗 eligible denominator 大于零，实际按 `0.25` 缩放的 window numerator 大于零。
3. 两格 base-decel eligible denominator 都大于零；control 的非零 reward numerator 必须为零，treatment 的
   非零 reward numerator 必须大于零。所有计数和 reward 都必须 finite。
4. checkpoint 必须通过 filename=embedded iteration、tensor finite、fresh lineage、hard-contract/source/claim
   绑定；缺任一项即 invalid，不得解释为机制失败。

早判规则：

| Milestone | 作用 | 底座速度 | pre-fall | 四项精度非劣 | 动作 |
| --- | --- | --- | --- | --- | --- |
| `+200` | 只看方向与仪表是否工作 | treatment/control `≤1.00` | treatment−control `≤0.05` | 每项 treatment−control `≥−0.10` | 不晋级；activation 失败则 invalid，其他失败只预警 |
| `+500` | 因果 screen | treatment/control `≤0.90` | treatment−control `≤0.02` | 每项 treatment−control `≥−0.05` | 三门全过才跑到 `+1000`；任一明确失败可在保存配对证据后一起停两格 |
| `+1000` | 训练内终档诊断 | treatment/control `≤0.90` | treatment−control `≤0.02` | 每项 treatment−control `≥−0.05` | 只产出 candidate/reject/inconclusive；不自动采用 |

四项精度是 racket position、racket velocity、**带符号** racket normal 和 strike composite 的 pass rate，
逐项判，不能用 composite 掩盖单项倒退。若 control 的底座速度接近零导致比值不稳定，则本 milestone 对
速度门记为 inconclusive，不得用任意 epsilon 制造通过；同时保留两格原始均值。

这里的早判只管理这一对训练槽。它不授权第二 seed、Isaac/MuJoCo judge、stop/promote 到正式成绩、部署或
真机。即使 `+1000` 三门都过，也只得到“值得另行预注册 exact exam”的单 seed 候选。

### `+200` 实测激活与方向

只读屏审冻结 TensorBoard step `180..200`，每个选定 tag 恰好 21 点；两次重读 digest 对
control/treatment 分别为 `95d19d16...49bb` / `a4036b15...5788`。结果：

- V1 没有 eligible denominator 与“持拍手腕已排除”numerator tag，因此第 1 条 activation 直接 blocked。
- V2 也没有计数级 denominator/numerator。仅有的 `strike_window_hit_rate` proxy 在 treatment 的
  mean/min/max 全为 `0/0/0`；当前 `strike_window_wide_s=None`，故该窗口在这 21 点没有实际作用。
- base-decel treatment 的 Reward contribution mean `0.1527856744`（min `0.141066`，max `0.162585`），
  证明至少有 active samples；但源码只记加权 Reward，没有 eligible denominator/raw-kernel numerator。
  control weight=0 的 Reward=0 也不能证明其 denominator>0，第 3 条仍 blocked。
- 描述性方向不构成因果 screen：底座速度 treatment/control=`0.4068215/0.3938566`，比值
  `1.032918`（方向不通过预注册 `<=1.00`）；pre-fall 差 `+0.00238` 个百分点；position error
  好 `1.43 cm`，velocity error 差 `0.02157 m/s`，signed normal error 差 `0.500°`。四项 exact pass 均为零，
  exposure 太低，不能用于 precision 比较。

因此这一对是 **activation-invalid / instrumentation-blocked**，不是 base-decel Reward 负结果；
不买第二 seed，不授权 judge 或晋级。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| V1+V2，底座减速关的首次 namespace；`phase1_fresh_c_v1v2_base_decel_control_seed3_20260714` | rejected，禁止重发 | first iteration 前，seed 3 | PID=PGID `358331` + claim/namespace | dynamic URDF import `malloc(): invalid size (unsorted)`，自然 `rc=134` | 基础设施失败，不是 Reward/行为失败 |
| 同配方 control 唯一重试；`phase1_fresh_c_v1v2_base_decel_control_seed3_retry_v2_20260714` | `model_200` 身份过；记录时 live | `200/500/1000`，seed 3 | PID=PGID `359240`；model `44a709ac...035a` | receipt `ad47c826...4d1f` | V1/V2/base-decel denominator 不完整 |
| V1+V2，底座减速权重 1；`phase1_fresh_c_v1v2_base_decel_w1_seed3_20260714` | `model_200` 身份过；记录时 live | `200/500/1000`，seed 3 | PID=PGID `359872`；model `b04e2338...e56b` | receipt `49234348...7748` | Reward 非零，但不能替代 denominator/numerator 合同 |

## 决定

- 决定：`inconclusive because activation instrumentation is incomplete`
- 理由：paired checkpoint 身份与 `180..200` 冻结曲线均有效，但三组预注册计数门没有闭合；
  且 V2 window 的实测 proxy 为零。这不允许进入行为比较。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：先在 source 中为 V1、V2 与 base-decel 分别加 per-update integer
  eligible/numerator 仪表，用真实多 env 反例证明 snapshot+reset，再决定是否值得 fresh namespace 重跑。
  当前 live pair 不继续买 seed；替换时只管理 exact PGID 并保全日志。旧 `358331` 永久 rejected。

## 复现与证据

当前只允许离线验证 YAML 和 fail-closed 行为：

```bash
pytest -q tests/test_phase1_fresh_c_v1v2_base_decel_prereg.py
```

本实验现有科学 trainer 只产生 checkpoint 身份与 activation-invalid 方向证据，没有可采用的 Reward
因果结果。后续运行必须走项目统一 lean queue harness；本文不会另建竞争性的算力优先级队列。

2026-07-14 同 source family 的首个 full-scene probe 在 iter0 暴露 clean detached `077e70c` 缺少 Git
忽略 A3 URDF/mesh tree；它在 Reward、hard contract 和 checkpoint 之前，不能评价本交互轴。两行现显式绑定
donor `6d93bcb`、46 files / 15,378,264 bytes / tree SHA `0137f59b...26c6`，并要求先由
`prepare-source-assets` 产生 source 外 receipt、science doctor 再重算并消费。本分支只闭合 source gate，
当时不远程水合、不解锁状态，故两格当时仍 blocked。

后续 c7 非科学 canary 的 result/model/hard-contract SHA-256 为 `02780b52...c4186` /
`a813ea9b...38e68` / `c39cf1ae...df838`，76 个 tensor / 1,762,715 个浮点元素全 finite、fatal0、原 PGID
自然为空；但其 `unlock_authorized=true` 只符合旧终档语义，不能解锁本实验。strict `main@caeb9ad`
attempt `caeb_strict_terminal_pod2_gpu1_a1` 随后通过，result/claim/model/hard-contract SHA-256 分别为
`0d03bd0305a56e56440b14e1f41278a26c0cad3a84cc1245325faed1ef29b1d1`、
`7437db488d8aa062aba8de91fb517362cc609a81900f0e953f80e15174c36ad5`、
`e1b79d142c13bc2df513b2a7311fbeb7b610fc64047e095c1a54c76571fe3106`、
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`；result 绑定 actual 4096
environments、physical ball/三实体、76 tensors / 1,762,715 浮点元素全 finite、fatal0、自然空 PGID 与
clean caeb source。显式 unlock 后的首次 control PID=PGID `358331` 在 first iteration 前 dynamic URDF import
报 `malloc(): invalid size (unsorted)`、自然 `rc=134`；treatment 未发射。旧 control 行已 rejected，唯一
unchanged-recipe retry-v2 与 treatment 保持 ready，并由同一 fill 的 first-iteration 顺序约束保护。probe
非科学、不可晋级。该 fill 随后让 retry-v2 PID=PGID `359240`（Pod2 GPU1）和 treatment PID=PGID
`359872`（GPU2）依次越过 first iteration；matched pair 现 live，尚无 checkpoint/早判，交互轴结论仍为
`inconclusive`。
