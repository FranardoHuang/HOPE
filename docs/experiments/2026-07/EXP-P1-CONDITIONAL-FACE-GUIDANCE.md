# EXP-P1-CONDITIONAL-FACE-GUIDANCE — 不逃离就绪区的固定预算 Reward

- 状态：`Partial`（E1 source gate 已合入 `main`；旧 source 的 1-env cache probe 通过，但 4096-env
  formal control 在第 0 次迭代前卡死；P1 新配对已绑定 exact source，等待全规模非科学探针）
- 阶段/轴：阶段 1 固定点；Reward 争抢机制
- 集成小目标：在不牺牲触点、拍速、完成率或安全的前提下，降低有符号拍面误差
- 人类负责人：franco
- 执行者：Codex
- 复核/决策负责人：franco
- 最高证据等级：`E2`（`main@61007e9` source/tests + Pod2 启动反例；尚无 Reward 训练结果）
- 创建日期/最后复核日期：2026-07-14

共享缩写见[术语与人话对照](../../DEFINITIONS.md)。本文的
[`racket_face_conditional_guidance_weight`](../../DEFINITIONS.md#conditional-face-guidance)
是“在击球窗内把固定成本从就绪缺口连续换成拍面误差”的 Reward 开关，不是另一张拍面题库。

## 问题与假设

已有静态 signed-face 线性罚在整段 `pre_strike | strike_window`（击球前或击球窗）持续收费。
最近的 `-0.4/-0.2` 单 seed 方向筛显示：角度可以改善，但位置、拍速或完成率同时付税；详细 runtime
receipt 进入 `main` 前，这里不抄成正式成绩。可证伪假设是：负结果主要来自“拍子尚未到位时多个 Reward
争同一自由度”，而不只是线性罚的权重不对。

朴素的 `readiness * face_error` 不能使用：配非正权重后，策略可故意离开就绪门，把罚金降成零。
本实验改问：若击球窗内始终保留同一最大成本，但随位置/拍速就绪把成本连续换成拍面误差，能否相对
匹配对照降低有符号拍面误差，同时不造成位置/拍速/完成率/安全退化？若不能，该结构假设被证伪；
不再扫更多静态权重。

## 冻结公式与单一因果轴

令 `e_theta` 为 [`raw-A`](../../DEFINITIONS.md) 实测拍面与题目目标面的有符号夹角，`e_p` 为当前
swing-through 目标点的位置误差，`e_v` 为完整拍速向量误差。紧支撑门为：

```text
compact(e; full, zero) = clamp((zero - e) / (zero - full), 0, 1)
face_fraction = clamp((e_theta - 0.262) / (pi - 0.262), 0, 1)
readiness = compact(e_p; 0.075 m, 0.095 m)
            * compact(e_v; 0.5 m/s, 1.0 m/s)
penalty = wide_strike_window * (1 - readiness * (1 - face_fraction))
```

`penalty` 严格在 `[0,1]`；Reward 权重必须非正，因此 `|weight|` 就是每个仿真 step 的最大罚金预算。
位置 `7.5 cm` 是 exact pass、`9.5 cm` 是 virtual-ball capture；拍速 `0.5 m/s` 是 exact pass，
`1.0 m/s` 是外门。击球窗外成本为零；击球窗内任一就绪外门之外成本固定为 1、拍面梯度为零；完全
就绪后成本等于拍面误差分数。因为 `d penalty / d readiness = -(1-face_fraction) <= 0`，位置或拍速
越就绪，成本绝不会增加；策略不能通过故意离开门来躲罚。15° 以内的拍面分数为零。

这不是“纯 face、对 readiness 完全无标量影响”的虚假声明：它会在拍面已有正确方向时给就绪改善
减免，但方向只可能与位置/拍速目标一致。首轮的单一因果轴是这整个固定预算联合机制的开关；门宽、
公式和预算不另扫。

首轮只有一个逻辑差异：

| 单元 | 人话 | 唯一差异 |
| --- | --- | --- |
| control | 同 source 固定点 fresh 对照 | static guidance `0`；conditional guidance `0` |
| treatment | 不逃离就绪区的固定预算纠面 | static guidance `0`；conditional guidance `-0.4` |

两格必须同 source commit、seed 3、v4rg 动作/bank/exam、零摩擦 plant、4096 environments、
1001 updates、PPO/观测/action/动作模仿/其他 Reward 和 `200/500/1000` checkpoint。由于新增字段进入
training hard contract，旧 source 的 control checkpoint 不能冒充匹配对照。首轮不扫门宽、不扫 weight、
不买第二 seed。

machine prereg 已冻结为 source `61007e93879f35677e4c7d38cf7f681f324f9571`、只在 Pod2 调度：

- control：`phase1_fresh_c_conditional_face_control_seed3_20260714`，conditional weight `0.0`；
- treatment：`phase1_fresh_c_conditional_face_w04_seed3_20260714`，conditional weight `-0.4`。

两格同时显式把 qdot hinge weight 固定为 `0.0` / margin `0.85`，避免继承差异；其余 delta 完全相同。
两格 `preferred_slot` 都是已过 warmup 的 Pod2 GPU1；修复后的短锁允许它们在同卡容量 3 内先后启动，
满载时才回退 round-robin。

该 source 的独立 Pod2 GPU1 cold-boot warmup 已自然退出：claim
`f10ed619e0b71a28a0b0b781d95f1e5508c9bcead5a6f82ca909affbfd527012` 明确
`purpose=boot_warmup_not_science`、1 env×2 updates/save1；`model_0/1` 均为 76 tensors / 1,762,715 floats
全 finite，embedded iter、schema3 contract `c39cf1ae...`、claim SHA 与 fresh lineage 全匹配，fatal0。
它只证明该 source/host/GPU 在 **1 environment** 的 cache/importer 最小路径能闭合，不代表
4096 environments 的正式 scene recipe。后续运行已经直接推翻“1-env 通过即可授权正式发射”这个假设。

发射 P1 full-scene probe 前又抓到一个控制面越界：原 execute 路径复用了普通 fill 的 all-Pod
`live_snapshot`，即使 `dispatch_pods=[pod2]` 仍会只读访问 reserved Pod1。该行为没有创建 probe，但违反本轮
Pod2-only 资源合同。P1.2 改为只读 selected dispatch Pod/GPU 的唯一 PID 数，未知输出或达到容量即
fail closed；远端 fd8 锁内容量复核保持不变，普通 fill 的跨 Pod claim 防重复语义也不改。源码回归通过前
probe 保持未发射，不能绕过专用 confirm 直接执行 dry-rendered SSH。

随后对 normal `fill` 的只读复核发现每臂会先建立一次 standalone doctor SSH，紧接着的 atomic launch 又
执行同一套检查。前者不占槽、不写 claim，也不能消除两次命令间的漂移，只会让不稳定网络或一次 Hydra
compose 暂态失败提前中断整批。P1.3 将 execute 收敛为每臂一次 launch SSH；source/assets/module/Hydra 门
仍在该原子 launch 内、并位于容量/claim/Kit 之前。该变更没有点火、没有改变 pair 配方或 probe 授权状态。

同一 launcher 红队还发现一个与本机制无关、但会影响失败臂安全收口的控制面缺口：旧 watchdog 只在
spawn 时验证 `PID=PGID`，signal 前没有重验 `/proc` starttime；TERM 后也只轮询 leader，可能漏掉仍活着的
同组 child。修复把 leader identity 与 TERM 前 exact member snapshot 写入 adjacent evidence；PID reuse、
leader 在 TERM 前消失、成员后来加入或读中漂移全部 no-signal/manual-review，KILL 只接受原成员的 exact
子集。该源码门没有连接 Pod、没有重放 probe、没有更改 conditional `0/-0.4` 配方或解锁状态。

P1.4 将“首迭代启动成功”和“full-scene probe 终档通过”拆成两个命令。probe trainer 现在在专属
`full_scene_probe_binding.json` 绑定 claim/log/source/GPU/PID/starttime；同 PGID supervisor 只自然等待并记录
normal rc 或 signal，不发 signal。launcher 仍只报告 first iteration。随后 selected-Pod-only finalizer 才核对
整个原 PGID 自然消失、current expected claim、phase 顺序、fatal0、finite/fresh-lineage1 `model_1.pt`、
schema-3 contract、exact supervisor argv、source-asset receipt 与 motion/bank binding，并
no-clobber 写 pass/fail `probe_result.json`。失败不自动 retry，仍 live 不提前冻结结果；普通 milestone attestor
拒绝 `attestable=false` probe。整合 focused `126 passed` 只是 source gate，尚未在 Pod 运行，所以 p1r1 pair 继续
blocked，也没有改变本实验的 `inconclusive` 结论。

旧 pair 保持不可修改的失败证据。新的 `p1r1` pair 已在结果前绑定同一 clean detached
`main@077e70c`（Pod2 checkout `/workspace/codexschema/nohope_p1_077e70c`），两格均
`runtime_binding=true`，仍只差 conditional weight `0/-0.4`。当前二者均为 `blocked`：先由 control 配方
在 Pod2 GPU1 的独立 `_full_scene_probes/` namespace 保留全部 scene argv 与 `4096 environments`，仅缩短为
2 updates/save1；只有看到首个 `Learning iteration`、随后自然退出并由 immutable terminal result 证明
fatal0/model1/contract/claim/source/asset 全闭合，才有资格交给未来显式 unlock consumer；当前不会自动解锁。

## 不可补偿安全边界

该项只返回非负 magnitude，再乘非正权重；它不会提供正安全信用，也不改 termination、自碰/自打、
joint/torque/qdot limit、观测、动作或 plant。以下任一项都独立判失败，不能拿拍面或 composite 改善抵消：

- 新出现 self-hit、桌网碰撞、非有限状态、hard-contract 漂移或 guard 类别；
- treatment 的 physical/root fall 或 guard reset 相对 control 恶化；
- checkpoint 文件名/嵌入 iteration、finite、fresh lineage 或相邻 hard-contract SHA 不一致。

## 运行漏斗与 `+200` 早判

1. **E1 source gate：** focused math、Hydra fail-loud translation、raw-A 共享配对、hard-contract
   bounds、就绪单调性反例与默认关闭测试全部通过；feature 合入 `main` 后才允许 machine prereg。
2. **E2 发射门：** lean queue 的 `doctor --live` 与 first-iteration marker 通过；control/treatment 必须在
   同一新 source 下成对启动，不能用已有旧-source control。
3. **`+200` 硬早判：** 两 checkpoint 均可审计；treatment 的
   `face_conditional_guidance_gate`、`face_conditional_guidance_error_fraction`、
   `face_conditional_guidance_cost_fraction` 和该 Reward contribution 必须 finite。若最后 21 updates 的
   gate 全零，说明机制从未产生拍面纠偏信号，判机制无效；若 position/velocity/
   completion 任一比 control 低超过 5 个百分点，或出现上述安全退化，立即保全日志并拒绝该格。
4. **`+500` 方向门：** treatment 的较差侧 signed normal error 至少比 control 低 10°，同时 position、
   velocity、completion、composite 较差侧均不低超过 5 个百分点；否则不跑第二 seed。
5. **`+1000` 决策：** 同一 immutable 每侧 50 题卷；只有 signed-face 改善、两侧 composite 不退、
   安全不退三者同时成立，才给胜者和匹配对照购买第二 seed。任何 Isaac 结果仍须过 vendor MuJoCo/
   Gate3，不能用解析上台冒充最终演示。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 同 source fresh 对照（`phase1_fresh_c_conditional_face_control_seed3_20260714`） | 基础设施失败，永久拒绝该 namespace | seed3；第 0 次迭代前 | E2 | claim `caffd19e...da52`、run log/launch sidecar | 4096-env 日志停在 URDF import；无 scene、contract 或 checkpoint，不是 Reward 失败 |
| 不逃离就绪区的固定预算纠面（`phase1_fresh_c_conditional_face_w04_seed3_20260714`） | 未发射、旧配对阻断 | seed3；无进程/claim | E1 | run directory 不存在 | control 失败后 serial fill fail closed；必须换 fresh source/namespace，不能单独补 treatment |
| P1 runtime-bound fresh 对照（`phase1_fresh_c_conditional_face_control_p1r1_seed3_20260714`） | blocked，等待 4096-env probe | seed3；200/500/1000 | E1 | exact source + machine prereg | 不得用 normal fill 绕过 probe |
| P1 runtime-bound 固定预算纠面（`phase1_fresh_c_conditional_face_w04_p1r1_seed3_20260714`） | blocked，等待 4096-env probe | seed3；200/500/1000 | E1 | exact source + machine prereg | 与新 control 唯一差异为 `-0.4` |

## 决定

- 决定：`inconclusive`
- 理由：公式、反向激励反例与 source gate 已进入 `main@61007e9`；1-env probe 虽通过，4096-env
  control 仍在 dynamic URDF import 后日志停止增长。它在 scene、hard contract、iteration 之前，不能评价机制。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：旧 source610 配对不再运行。新 pair 已绑定 fresh P1 source/namespace；下一步只运行
  同 source、GPU、`num_envs=4096` 和同 scene recipe 的非科学 full-scene probe。1-env 只作 cache probe，
  不再授权正式训练；probe receipt 审完前禁止手写 CLI 或 normal fill 点火。

## 复现与证据

源码入口是 `hope_rewards.racket_face_conditional_guidance`；训练只暴露一个开关，固定门参数由 hard
contract 记录。运行方法与 first-iteration/claim 纪律见[训练操作](../../operations/run_training.md)和
[lean queue 操作](../../operations/run_lean_training_queue.md)。本实验不授权 Pods、judge 或真机。

2026-07-14 的首个 P1 full-scene probe 又产生基础设施结论：它在 iter0 以 `rc=1` 自然退出，直接缺失
clean detached source 中 Git 忽略的 A3 `model.urdf`；这不是 conditional Reward 失败。新的 `p1r1` 两行现
显式绑定 donor `6d93bcb` 与接受的 46-file tree，且 science claim 会绑定完整 ignored-asset contract。
必须先走 selected-Pod-only `prepare-source-assets` 产生 source 外 exact receipt，再由 doctor 重算并消费；
本次源码工作不执行该水合、不重发 probe，pair 继续 blocked。

pre-probe 机器清单现改绑 exact `main@c7e1a90` 与
`/workspace/codexschema/nohope_p1_c7e1a90`；control/treatment 明确分到 Pod2 GPU1/GPU2，但两行仍
`blocked`，顶层 `launch_authorized=false`。普通 live snapshot 只访问 `dispatch_pods=[pod2]`，不会读取
reserved Pod1；terminal probe result 未通过前，`fill/launch-next` 会在任何 SSH 前拒绝。本次只闭合控制面，
没有水合、probe、checkpoint 或行为结论。

随后 P1.5 又把旧 probe 的 launcher-only 短终态变成可审计的 failure-only receipt，并补齐实际 4096-env、
物理球/桌实体、face179、31/31 零摩擦与正式 schema-3 validator 门。validator 直接从 exact checkout 载入，
不触发 Kit/Omni package import；旧 c7 hard contract 尚未由该门重新验证，不能追认通过。增量 focused
`100 passed` 仍只是 source gate，本卷 `inconclusive` 与两条 p1r1 `blocked` 状态不变。
