# EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728

## 问题

把现役四动作 upper baseline 去掉旧正手，只保留三个已有有效题库信号的反手，并独占一张卡，
能否以更高吞吐继续改善每个反手，同时不再允许 policy 借 physical-hard 关节请求或实际越界获利？

这条实验称为
[`upper N3 safe warm-start`](../../DEFINITIONS.md#upper-n3-safe)。人类负责人是 Franco，
执行者是 Codex。状态为 `invalidated / superseded`：它从未进入 Kit、训练 update 或 checkpoint，
且随后发现父模型与三件 teacher 的 reset/clip frame-0 坐标不一致，policy 回放也没有保持
teacher fidelity。Franco 已决定不再恢复这条 upper/static-bank 热启动，改为从 fresh exact N=5
的 action-conditioned ball-first 任务重新训练。本页只保留诊断数据，不再提供可执行入口。

## 1. 假设与非目标

可证伪假设：

1. N4→N3 保持 actor/critic/optimizer shape 不变，移除当前合法回球为零且已决定淘汰的旧正手后，
   每个 update 不再给无用动作分配环境，三个反手的样本吞吐应提高。
2. N4 父本已经学到三个反手的共同平衡/挥拍表示，完整 PPO 状态热启动应比 fresh N3 更快进入有效
   区；但任务合同改变，所以它只能是 non-exact warm-start。
3. physical-hard 双终止、pre-physics crossing guard、soft clamp/qbar/joint-limit 与统一 terminal
   罚可以让 unsafe 严格压过成功，不再出现超范围 q_des 被 clamp 后从账上消失。

非目标：不回答 action-ball 动态课程、73/93 动作、selector、MuJoCo 迁移或真机效果；也不把独占
GPU 当算法改进。

## 2. 父本终档

父 run 是 `f5_upper_seed0`（四动作上肢 static-bank、seed 0 的训练基线；本实验只把它作为
warm-start 父系，术语见
[`upper N3 safe warm-start`](../../DEFINITIONS.md#upper-n3-safe)）。Pod1 在完整 PPO boundary
收口到 `model_10809.pt`：

- checkpoint SHA-256：
  `74d481771c8f0a1ea3e7c1627db188c11e699b03eac1558144f50e118c17ff23`；
- 训练合同 SHA-256：
  `75f46c92495d8fc96b5dfb67294ee2236e3094f4e2c46544221db4141a9589bc`；
- `92` tensors、`1,876,333` values，全部 finite，actor `175 → 31`；
- 最后 100 窗为 iteration `10710..10809`。

| 动作 | rally return mean / latest | legal / strike | capture / strike |
| --- | ---: | ---: | ---: |
| `bh_loop_c` | `44.394% / 48.22%` | `9648/18009 = 53.57%` | `12831/18009 = 71.25%` |
| `bh_block` | `46.907% / 48.82%` | `10081/18333 = 54.99%` | `13683/18333 = 74.64%` |
| `s0_highpress` | `37.369% / 40.19%` | `8009/18859 = 42.47%` | `12299/18859 = 65.22%` |

旧父本的安全读数不能满足新门：near-limit fraction mean `25.916%`、joint velocity absolute max
mean `11.700 rad/s`、arm torque saturation mean `20.480%`；`440/85646 = 0.514%` physical falls，
table hit 为零。它没有 raw pre-clamp、substep hard-gap、qbar 或 joint-reason 账，因此成绩不能
反推安全。

## 3. N3 输入与唯一变量

动作顺序冻结为：

1. `bh_loop_c`，phase `0.442857`；
2. `bh_block`，phase `0.452830`；
3. `s0_highpress`，phase `0.250000`。

三件 family 均为 backhand，signed face 均为 `-1`。motion SHA 依次为：

- `c950a73e473cad84d0fafcd51c552ec4fef085580bbeaec0f4e96be2acd7e2fc`；
- `0cd94aa47bf8feb59bbe7cc7a0306abb57ee7ec8ebcec6443a80bbdc58894309`；
- `2bc5f6483644aa69fe3dfc42775461a6fbdf8cd1c407d0b0c8e593a86dfd8f50`。

N3 bank 由 N4 bank 按上述源顺序确定性投影，SHA-256 为
`6d61fda0011321e77095904118a348512f8784a023079b8f66e5218b5589df22`；clip 题数依次为
`6617 / 6460 / 6510`，37 个 schema keys 与源数组逐 bit 保持。旧正手不进入 N3 view，源资产不删除。

除 N4→N3 动作/题库和下节安全叶子外，首跑保持父本 static-bank、175 维 actor、动作 phase/sign、
target noise、stand start、teacher speed `[0.6,1.0]` 与 PPO recipe。Reward 明确使用实际成功配方：
tracking `4.0/0.5/0.5` + `virtual_landing=1648.8`；`393.4/295.1/229.5` 只作后续消融。

## 4. 安全叶子

`HOPEPingPongUpperSafe` 继承 175 维 VirtualBall，不继承 177 维 Hitter。它新增：

- pre-clamp q_des 与 actual q 两个独立 physical `joint_pos_limits` hard-margin-zero termination；
- `0.02 s` runtime-matched pre-apply hard crossing latch，送 PhysX 的 processed target 仍 clamp 到
  soft envelope；
- qbar `-0.65 / 0.08`、continuous joint-limit `-10`；
- generic `death=-3600`，在 50 Hz 下任何 termination 一次 `-72`；table-specific 与
  joint-specific 均为零，避免双罚。

每个 unsafe reason 的 counter 独立保留，unsafe 不得混入 legal-return failure。

2026-07-28 source 收口把消费改成 fail-closed 两阶段：action term 先返回 zero-copy frozen view，
runner 在 simulator device 上验证每个 environment 的 exact `4 apply + 1 post`、连续 sequence、
action UID、birth/swing generation、birth receipt、archive transcript 与 accumulator 守恒，再只把
sparse COO counter/reduction 转到 CPU，durable 发布 prepared sidecar；optimizer 成功并写
commit marker 后才 ack/清账。公开 one-shot destructive consume 已禁用。actual q 到达/越过
physical hard edge 或 non-finite 会先写 fatal sidecar、禁止 optimizer 并保留账本。4096 env ×
24 step × 31 joint 的 host safe-case 实测 prepare `0.0443 s`、Python peak `15,183 B`，完整
validate→compact→双 fsync→commit→ack `0.3649 s`，sidecar `956,704 B`（core payload
`892,002 B`）；focused `70 passed`。这些仍不是 Isaac/Pod 行为证据。

## 5. 发射与资源预注册

Pod1 physical GPU1 独占本 N3；GPU0 的 combo 和 GPU2 的 MuJoCo/视频不触碰。本实验只管理自己创建、
身份完全匹配的 no-clobber namespace。

阶段固定为：

1. host focused tests；
2. exact clean commit/checkout；
3. `1 env × 2 update` 构造与安全 smoke；
4. 小预算 canary；
5. 只有安全、finite、身份和三动作最低行为门全过才允许独占长跑。

操作与命令只看
[`run_upper_n3_backhand_safe.md`](../../operations/run_upper_n3_backhand_safe.md)。

## 6. 判定门

smoke 必须提供：

- N3 order/phase/sign/bank、actor `175 → 31`、父 checkpoint/contract exact pin；
- finite pre-clamp proposal 的 soft/hard分层反例，以及 NaN/Inf 不进入 PhysX 的反例；
- 每个 physics substep 实际 q 的 finite/hard-gap readback；
- 两个 joint reason、fall、table 与 generic death 一次收费的 counter 不变量；
- 每个 update 恰好一份 joint-safety stdout 收据、prepared sidecar 与 optimizer commit marker；
  sidecar 能跨 reset 还原旧 birth identity，且完整 policy-step 数等于 `num_steps_per_env`；
- actual hard edge/non-finite q 的反例必须先 durable 写 fatal sidecar，optimizer/ack/curriculum
  调用数均为零；
- finite checkpoint、两次 update、无未解释 WARN/Traceback。

canary 才开始看行为。每个动作分别报 capture/strike/legal return、fall/table/joint unsafe、
near-limit/qdot/torque saturation；禁止只报 N3 平均。若任一 sampled actual q 触到或越过 physical
hard limit、counter 守恒破坏、checkpoint 非 finite 或身份漂移，按预注册规则停止。

吞吐收益必须用同一 Pod/GPU、相同 `num_envs` 与 PPO recipe 比较 N4/N3 的 wall-clock
updates/hour；不能用独占卡与旧混卡的差异冒充“删动作”因果。若后续要正式归因，另开 fresh matched
N4/N3 A/B。

## 7. 已发生的无效 probe

旧 namespace
`/workspace/codexschema/upper_n3_backhand_20260728` 曾尝试
`upper_n3_backhand_warm10000_safety_smoke1`（基于旧 `8ba15f38`/`model_10000` 的两次 update
构造探针；它不是本实验的新安全配方）。第一次在 launcher 的 awk 变量名处失败；修正后在 Hydra
compose 因旧 source 没有 qbar YAML key 被拒绝。它没有进入 Kit、没有 RSL run directory 或
checkpoint；launch receipt SHA-256
`47b3bf7a0ae8cf9967c1b00cadddbcc725158892bff82ed09bfa06ae2109f478` 保留为
infrastructure-invalid 证据，禁止重用或写成训练负例。

## 8. 当前结论

本实验已作废，不得发射、resume 或作为 N=5 的 warm-start。旧父本的三反手行为统计仍可作为
历史参考，但其坐标错配、动作抽搐、关节安全账缺失和 teacher fidelity 失败使 checkpoint
不具备迁移资格。通用的两阶段 joint-safety ledger、软/硬限位和桌碰终止能力继续由 fresh
ActionBall 任务复用；UpperSafe 专用 task、registry 和 launcher 从候选提交撤回。
