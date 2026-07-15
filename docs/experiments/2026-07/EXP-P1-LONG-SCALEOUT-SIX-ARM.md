# EXP-P1-LONG-SCALEOUT-SIX-ARM — 两张空卡的六条单 seed 长曲线

- 状态：`running`
- 阶段/轴：阶段 1，击球窗模仿、关节速度约束、脚部朝向约束
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E2`（真实首迭代/运行时）
- 创建日期/最后复核日期：2026-07-15

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本卷不以内部代号作结论：

- “放开手腕线速度模仿”指击球手腕不再被老师动作的线速度逐帧拉住；
- “击球窗模仿降到四分之一”指靠近触球时仍保留老师姿态，但把动作模仿强度从 `1.0` 降为 `0.25`；
- [`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 指关节速度接近真实上限后才开始收费；
- “脚部朝向惩罚”约束髋/踝朝向，不直接奖励合脚或固定站位。

## 为什么现在跑

Pod2 在 2026-07-15 15:27 UTC 的只读快照中，GPU0/GPU1 都没有 compute PID；GPU2 的普通对照、
`qdot=-5` 和两项模仿同时放松三条 10000-update 长训已经约到 3.2k。早先单独放开手腕、单独降低
击球窗模仿的训练因 Pod1 让卡而停在 800 update 以内，不能据此作学习结论。现在用同一个已经通过
4096-environment full-scene 门的 source，在两个空 GPU 上补齐两个完整因果曲面：

1. 普通对照、只放开手腕、只降低击球窗模仿、两者同时开启的 `2×2` 组合；
2. 关节速度惩罚 `0/-1/-2.5/-5` 的单 seed 剂量曲线；
3. 脚部朝向惩罚 `0/-0.3/-0.6` 的单 seed剂量曲线。

三组共用 GPU2 上仍在训练的普通对照；不会复制第二 seed，也不会把六个新格互相冒充对照。

## 冻结合同

| 字段 | 值 |
| --- | --- |
| source | `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e`；Pod2 clean checkout `/workspace/codexschema/nohope_p1_activation_successor_2c2d70d` |
| 动作/题库 | 现役 `v4rg` 正反手；schema-3 signed-face rebound train bank |
| Plant | Isaac、31 关节零摩擦；只作训练与方向诊断，最终由 vendor MuJoCo 判 |
| Seed/预算 | 共同 seed `3`；4096 environments；10001 updates；每 100 保存 |
| Checkpoint | `200/500/1000/2000/3000/6000/10000` |
| 资源 | 只用 Pod2；按 GPU0→GPU1 逐圈发射，每卡最多三条；Pod1 永不访问 |
| 随挥回放 | 全部关闭，避免 policy 自身存活时间改变 curriculum |

机器队列是
[`phase1_long_scaleout_funnel_20260715.yaml`](../../../configs/phase1_long_scaleout_funnel_20260715.yaml)。
它绑定与现役 GPU2 三格相同的 full-scene probe、动作、题库、source、seed、plant 与运行时 provenance。

## 六条新格

| 发射顺序 | GPU | 人话问题 | 唯一科学变化 |
| --- | --- | --- | --- |
| 1 | 0 | 单独放开手腕线速度，长训后是否保留击球收益且不增加平衡债？ | `free_wrist_vel_mimic=true` |
| 2 | 1 | 小强度关节速度约束能否比 `-5` 更少伤害早期击球？ | `qdot=-1` |
| 3 | 0 | 单独降低击球窗模仿是否有效，还是必须和放开手腕同时出现？ | `motion_scale_in_window=0.25` |
| 4 | 1 | 中等关节速度约束能否取得安全/击球折中？ | `qdot=-2.5` |
| 5 | 0 | 去掉脚部朝向收费会不会让脚更窄/更歪，或反而释放有效步法？ | `foot_orientation=0` |
| 6 | 1 | 加强脚部朝向收费能否稳住站姿而不压坏挥拍？ | `foot_orientation=-0.6` |

## 早筛与继续规则

- `200/500/1000` 只允许发现结构问题：崩溃、non-finite、合同漂移、应激活机制始终为零。不得以此宣称
  某种学习方式输赢。
- 只有真实击球后才有信息的回台/落点 Reward，必须先累计到冻结的最少 eligible hit 事件；样本不足时
  结论只能是“继续跑”，不能早停。精确 denominator 阈值在本轮启动后补入 observer 合同，不倒灌旧窗口。
- `2000/3000` 看中段趋势；`6000/10000` 才形成完整单 seed 曲线。安全失败可以单独收口；表现暂时差但
  没有结构/安全结论的格继续到下一正式 milestone。
- 长曲线胜者仍须同一 immutable vendor MuJoCo 卷通过后，才允许买第二 seed；本卷不授权 judge、晋级或
  真机。

## 运行表

| 运行 | 状态 | 证据 |
| --- | --- | --- |
| 只放开手腕 `phase1_long_no_replay_v1_only_seed3_20260715` | running | PGID `419643`；15:49 UTC 到 iter127，fatal0 |
| `qdot=-1` `phase1_long_no_replay_qdot_w1_seed3_20260715` | running | PGID `420298`；同窗到 iter82，fatal0 |
| 只降低击球窗模仿 attempt-1 `phase1_long_no_replay_v2_only_seed3_20260715` | invalidated | PGID `420947` 在动态 URDF import 阶段 `malloc(): invalid size`、rc134；首迭代前退出、无 checkpoint；namespace 永不复用，不是 Reward 结果 |
| 只降低击球窗模仿唯一重试 `phase1_long_no_replay_v2_only_seed3_retry_v2_20260715` | running | PGID `423502`；同窗到 iter2、fatal0、`KIT_BOOT_READY` |
| `qdot=-2.5` `phase1_long_no_replay_qdot_w2p5_seed3_20260715` | running | PGID `421479`；同窗到 iter59，fatal0 |
| 关闭脚部朝向惩罚 `phase1_long_no_replay_foot_orientation_w0_seed3_20260715` | running | PGID `422126`；同窗到 iter62，fatal0 |
| 加强脚部朝向惩罚 `phase1_long_no_replay_foot_orientation_w0p6_seed3_20260715` | running | PGID `422783`；同窗到 iter35，fatal0 |

发射命令：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_scaleout_funnel_20260715.yaml \
  fill --count 6 --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

controller 必须先验证 GPU0/GPU1 没有任何外部 compute PID；若 Yikang 在发射前重新占卡，相关行 fail
closed，不迁移、不抢占。attempt-1 的 importer rc134 已完整保全；唯一 retry 仍失败时停止重试并转 importer
根因线。

2026-07-15 15:49 UTC，Pod2 三卡分别恰有三条 trainer，利用率 `97%/97%/91%`，显存
`17294/17160/17270 MiB`。GPU0/GPU1 六条新格与 GPU2 三条长训合计九条均在继续；本快照只证明生产池
已铺满，不产生行为胜负。新格到 model-200 后再做 finite/lineage/contract/eligibility receipt。
