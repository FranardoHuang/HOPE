# 非击球臂模仿消融

- 状态：`preregistered`（源码与机器预注册已完成，Pod runtime 尚未验证/发射）
- 证据等级：[E1](../DEFINITIONS.md)（源码、静态计划与单元测试）
- 人类负责人：Franco
- 执行者：Codex
- 工作分支：`Franco_codex/non-striking-arm-a01-prereg-20260714`
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

## 判读与晋级规则

训练 checkpoint 先由 finalizer 验证 filename iteration、checkpoint 内 embedded iteration、finite tensor、
fresh lineage 与各自相邻 hard-contract SHA；A0/A1 的 hard SHA 必须不同，删除唯一预注册差异
`motion_imitation_body_names` 后的完整合同必须逐项相同。A0/A1 必须使用同一份另外激活的 immutable signed paper；本 launcher
不自动开卷。只有 A1 与 A0 都形成完整绑定且 A1 在同卷下回球/有符号几何不劣、平衡/恢复端点改善、没有
新的左臂振荡/自碰/间隙/关节/力矩违规，才允许给 A0/A1 匹配对照购买第二 seed。

A2（移除左臂模仿后，把固定总 reward 预算重分给平衡/就绪）继续 blocked。它必须先从冻结 A0 rollout
估一次量级，再另建预注册；不得在本 A0/A1 run 中偷调 reward 比例。

## 结果

源码与机器预注册已完成：reward override + manifest/runner 专项共 `71 passed`，静态 plan 能生成两条
除 run name/mask 外完全相同的命令。尚未在 Pod 执行 `validate-runtime` 或 `launch`，没有 trainer、
checkpoint、Isaac/MuJoCo 行为、成绩或真机结果；因此不能声称“不模仿左臂更好”。

## 下一步

1. 在空闲 Pod1 GPU0 安装 exact detached source 与 ignored A3 asset，运行 `validate-runtime`。
2. root 审阅后显式发射 A0/A1；记录 exact PID/PGID、GPU/RAM 和 +200/+500/+1000 checkpoint。
3. 激活同一 immutable signed paper 后做 paired 早判；不因单条曲线或最佳 seed 晋级。
4. A1 存活后才另开 A2 固定预算和 recovery/ready 交互实验。
