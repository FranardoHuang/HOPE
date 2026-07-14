# EXP-P1-V1V2-BASE-DECEL-INTERACTION — 组合击球精度下的底座减速是否仍有净收益

- 状态：`blocked`（已绑定 P1 exact source；等待同 scene family 的 4096-env 非科学启动探针）
- 阶段/轴：Phase 1 fresh C；V1+V2 与击球前底座减速的组合效应
- 集成小目标：保住 V1+V2 的击球精度信号，同时降低击球前底座速度和摔倒率
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E0`（只有机器可读预注册，没有训练结果）
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
| Source | `main@077e70c` / `/workspace/codexschema/nohope_p1_077e70c`；含 runtime binding、milestone attestor 与 full-scene probe |
| 初始化/seed | fresh / `3`；只买一个 seed |
| 预算 | `4096 environments × 1001 updates`，每 `100` 保存，配对检查 `200/500/1000` |
| 动作 | v4rg 正反手 runtime-order 动作；shared-face signed action |
| Bank/exam | schema-3 rebound train bank；同 family 的 immutable K100 exam 只绑定身份，本实验不授权 judge |
| Plant | `zero_joint_friction=true` 的当前可重放训练协议；不声称是部署 plant |
| 共同机制 | V1=`true`，V2=`0.25`，post-swing start=`0.25`，qdot hinge=`0.0`/margin=`0.85`，conditional face=`0.0` |
| 唯一差异 | control `base_decel_weight=0.0`；treatment `base_decel_weight=1.0` |
| 调度 | 只允许 Pod2；Pod1 仅保留在 harness 固定 schema 中，`dispatch_pods: [pod2]` 不会给它发新任务 |

两格仍为 `status: blocked`，所以 normal plan 没有 assignment。source 绑定已关闭，但这不等于 scene boot
已经通过：必须先用独立 `_full_scene_probes/` namespace、同 source/Pod2/GPU、同动作/bank/plant 与
`4096 environments` 越过首个 `Learning iteration`，再核对 probe 自然退出和 fatal0，才可另提交解除
blocker。probe 的 model/reward 永不进入实验成绩。

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

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| V1+V2，底座减速关；`phase1_fresh_c_v1v2_base_decel_control_seed3_20260714` | blocked | `200/500/1000`，seed 3 | exact P1 source + machine prereg | 无 | 等代表性 full-scene probe |
| V1+V2，底座减速权重 1；`phase1_fresh_c_v1v2_base_decel_w1_seed3_20260714` | blocked | `200/500/1000`，seed 3 | exact P1 source + machine prereg | 无 | 等代表性 full-scene probe |

## 决定

- 决定：`inconclusive`
- 理由：两格 recipe、差异、早判与 P1 source 已冻结，但代表性 4096-env probe 尚未执行，未产生科学 claim/run。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：先让同 source/scene family 的 full-scene probe 在 Pod2 空闲卡越过 first iteration并
  自然退出；另一个结果后提交才可把两格改为 ready。不得顺手改 recipe、seed、阈值或授权 judge。

## 复现与证据

当前只允许离线验证 YAML 和 fail-closed 行为：

```bash
pytest -q tests/test_phase1_fresh_c_v1v2_base_decel_prereg.py
```

本实验没有执行命令、Pod 结果或忽略资产的新要求。后续若解除 blocker，运行必须走项目统一 lean queue
harness；本文不会另建竞争性的算力优先级队列。
