# 非击球臂模仿消融

- 状态：`running / Partial`（A0/A1 均已 exact 启动；尚无 paired checkpoint 终档或同卷结果）
- 证据等级：[E2](../DEFINITIONS.md)（两臂 runtime/合同已绑定；尚无 A0/A1 行为结论）
- 人类负责人：Franco
- 执行者：Codex
- 工作分支：`Franco_codex/non-striking-arm-a01-runtime-receipt-20260714`
- 训练源码 commit：`353a11419ae8589ed4a374ed97169cd7a50d50a3`
- 预注册/runner commit：`40db3fe5a61d3643cc6a50188a0615666b2d8d91`

## 问题

当前右手 A3 是否需要模仿左侧[非击球臂](../DEFINITIONS.md)，还是让左臂脱离老师动作后能更好地调节
平衡，同时不伤害击球、回球与安全？第一轮只回答“直接 mask 有没有效”，不把 reward 预算重新分配混进来。

## A0/A1 直接 mask

[A0/A1 非击球臂配对](../DEFINITIONS.md)使用同一 fresh 初始化、seed `17`、题库、正反手动作、
训练预算和 checkpoint 节奏。两条实验臂唯一命令差异为：

- A0（当前上半身模仿对照）：`++task.rewards.free_non_striking_arm_mimic=false`；
- A1（左臂不模仿）：`++task.rewards.free_non_striking_arm_mimic=true`。

现役配方同时保留 `free_wrist_ori_mimic=true`，所以右腕在 A0/A1 的 orientation/angular-velocity
模仿中都已按既有配方排除；A1 不会进一步改变右臂或躯干。四个 Reward 的精确 `body_names` 是：

| Reward | A0：当前上半身模仿 | A1：只删左侧非击球臂 |
| --- | --- | --- |
| `motion_body_pos` | torso；左 shoulder/elbow/wrist；右 shoulder/elbow/wrist | torso；右 shoulder/elbow/wrist |
| `motion_body_ori` | torso；左 shoulder/elbow/wrist；右 shoulder/elbow | torso；右 shoulder/elbow |
| `motion_body_lin_vel` | torso；左 shoulder/elbow/wrist；右 shoulder/elbow/wrist | torso；右 shoulder/elbow/wrist |
| `motion_body_ang_vel` | torso；左 shoulder/elbow/wrist；右 shoulder/elbow | torso；右 shoulder/elbow |

源码只接受上述七-body 基线或既有 free-racket-wrist 六-body 变体；丢 body、改顺序或拼错布尔值都
fail closed。单测逐项证明 A1 只改这四份列表，reward weight/std、关节/动作限位、力矩/接触/自碰、
终止与安全停机均不变。

## 冻结输入与预算

机器真源是
[`phase1_non_striking_arm_imitation_a01_prereg_20260714.json`](../../configs/phase1_non_striking_arm_imitation_a01_prereg_20260714.json)，
文件 SHA-256 `b2462527b6573ce6accaf8e626fe264c3da10e8994dba133d8f0aeaeed870506`。它冻结：

- Pod1 GPU0；Pod1 GPU2 不在本实验范围；
- `4096 env × 1001 update`、fresh seed `17`、`save_interval=100`；
- `v4rg_runtime_order_v3` 正/反手动作 SHA
  `f2cb2d9f...f141687` / `17225533...367534`；
- signed-face rebound schema-3 train bank SHA `3a9d8851...85b71`；
- zero-friction plant、179-D actor、31-D action、相同正反手 phase 和既有 signed-face 配方；
- 只在 `+200/+500/+1000` 做 paired checkpoint 判读；不先买第二 seed。

没有另跑两条 25-update throwaway。依赖无关单测已验证 mask；运行态 smoke 是 reviewed locked-Kit
launcher 到第一条 `Learning iteration`，随后同一长臂直接继续到有效 milestone，避免多花两次 Kit 启动。

## 发射与 no-clobber 合同

四项 post-override `body_names` 同时进入 checkpoint 内嵌 hard contract 的
`motion_imitation_body_names` 字段，因此即使 checkpoint 脱离外层 run dir，A0/A1 也有不同且可复算的
hard-contract SHA。runner 默认只输出 plan。真正发射必须在 Pod root 下显式提供
`ROOT_APPROVES_SIM_ONLY_A0_A1_V1`，并通过 clean source/tree、关键源码、A3 ignored asset、motion/bank、
Python module origin、host memory、GPU0 初始空闲与显存门。每条臂使用新 run dir、exclusive
launch/runtime/result ledger；已存在或失败的 claim 禁止自动重试。wrapper 只管理刚创建的该臂 exact
PGID，没有 broad signal，也没有 judge 或真机命令。完整命令见
[操作文档](../operations/run_phase1_non_striking_arm_imitation_a01.md)。

## 2026-07-14 v1 outer-verifier 假拒绝与 v1r1 单臂续接

Pod1 GPU0 上的 v1 首次点火只成功启动 A0：`2026-07-13T19:48:35Z` 创建 exact
PID=PGID `1811464`，`19:49:15Z` 越过 `KIT_BOOT_READY`。A0 的稳定外层证据为
`launch_contract.json` SHA `4c059aa6...53153`、`run.log.launch` SHA `045518bc...a342`；相邻
schema-3 hard contract SHA 为 `14ef410b...29f1`、fresh lineage `1`。截至本次入账，A0 已产出
embedded iteration=`200`、finite 且绑定该 hard SHA 的 `model_200.pt`，训练日志未见硬失败。

v1 launcher 随即以精确错误
`[non-striking-arm-a01] FATAL: hard contract train-bank binding changed` 退出，A1 的 claim、训练目录和
进程均从未创建。根因不是训练合同漂移：真实 schema-3 hard contract 的 compact `question_bank` 只包含
bank SHA、schema、split、source-family SHA 和 exactness；bank 文件 SHA 已绑定完整 `meta_json`。旧外层
verifier 错把仅存在于 bank metadata/source-family contract 的 `physics_contract_sha256` 当成 compact
record 的 direct leaf，故在 A0 已启动后假拒绝。

一次性机器合同
[`phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json`](../../configs/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json)
（SHA `addcffa1...5fc3`）与
[`run_phase1_non_striking_arm_imitation_a01_v1r1.py`](../../scripts/run_phase1_non_striking_arm_imitation_a01_v1r1.py)
（SHA `9f98e360...ecbb`）只允许补 A1。它必须先逐字节验证旧 control、A0 三份稳定 SHA、exact PID/PGID/argv、
A1 全面 absent、bank file SHA，并从 NPZ `meta_json` 独立重算 source-family/physics 绑定；同时还要让冻结
v1 verifier 复现同一错误。随后先 no-clobber 写 recovery attestation，race recheck 后才可创建唯一的新
A1 claim。它没有 A0 launch 路径、自动 retry、broad signal、judge 或真机命令；A0 死亡、任一证据变化、
A1 预存在或 bank 漂移都永久 fail closed。

### v1r1 runtime receipt

2026-07-14，冻结 v1r1 external control 的 `validate-runtime` 全部通过，随后唯一允许的
`launch-a1` 成功：

- A1 exact PID=PGID `1816234`，已越过 Kit ready；
- A1 emitted hard-contract SHA 为
  `c85b52a28ad64a667a7b522562842466270b3741591f6daf09afc1d0f7c6b146`；
- A1 `runtime_verified.json` 的现场 SHA 以 `1277cf` 开头、`77f4` 结尾；recovery attestation SHA 以
  `604288` 开头、`e9cb` 结尾。当前仓库只收到这两个摘要，故不把它们伪装成可独立复算的完整 machine
  receipt；完整字节仍由 frozen external control/no-clobber ledger 保管；
- A0 仍为原 PID=PGID `1811464`，continuation 未重启或改动 A0；
- judge 未启动，A2、第二 seed、晋级和真机继续阻断。

external control 下单独运行 `--mode plan` 暴露了一个只读路径问题：`build_plan` 从外部 launcher 路径取
`parents[1]`，会把旧相对 manifest 解析为 `control/configs/...`。该调用在读文件前失败，没有写 attestation、
claim 或进程；exact repo-source plan 仍由新旧 runner `30 passed` 覆盖。runtime 的
`validate-runtime/launch-a1` 使用冻结的绝对 v1 control 路径，不经过该 plan 路径，实际已全绿并成功启动
A1。因为 v1r1 manifest/runner SHA 已进入 recovery、launch 与 runtime 账，**本次不得修改冻结字节**；
路径问题只在后续新版本中用独立 source/runtime-root 参数修复。

## 判读与晋级规则

训练 checkpoint 先由 finalizer 验证 filename iteration、checkpoint 内 embedded iteration、finite tensor、
fresh lineage 与各自相邻 hard-contract SHA；A0/A1 的 hard SHA 必须不同，删除唯一预注册差异
`motion_imitation_body_names` 后的完整合同必须逐项相同。A0/A1 必须使用同一份另外激活的 immutable signed paper；本 launcher
不自动开卷。只有 A1 与 A0 都形成完整绑定且 A1 在同卷下回球/有符号几何不劣、平衡/恢复端点改善、没有
新的左臂振荡/自碰/间隙/关节/力矩违规，才允许给 A0/A1 匹配对照购买第二 seed。

A2（移除左臂模仿后，把固定总 reward 预算重分给平衡/就绪）继续 blocked。它必须先从冻结 A0 rollout
估一次量级，再另建预注册；不得在本 A0/A1 run 中偷调 reward 比例。

## 结果

源码与机器预注册已完成；v1r1 recovery 专项 `12 passed`，新旧 runner 合跑 `30 passed`。A0 已有首个
绑定且 finite 的 `model_200.pt`；A1 已 ready 且 hard contract 绑定通过，但尚未收到 A1 milestone
checkpoint，也没有同卷 Isaac/MuJoCo 行为、配对成绩或真机结果。因此仍不能声称“不模仿左臂更好”或
继续购买 seed。

## 下一步

1. 不得重跑 v1、v1r1 `launch-a1` 或改动冻结 control；继续记录两臂 exact PID/PGID、GPU/RAM、日志与
   `+200/+500/+1000` checkpoint。
2. 两臂自然终档后用 v1r1 finalizer 验证 paired contract/checkpoint，再激活同一 immutable signed paper
   做 paired 早判；不因 A0 单臂曲线或最佳 seed 晋级。
3. external plan 路径问题只在新版本加负测修正，不得改写已绑定现场的 v1r1 bytes。
4. paired 行为门通过后才另开 A2 固定预算和 recovery/ready 交互实验。
