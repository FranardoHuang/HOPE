# EXP-ACTION-BALL-PHASED-READINESS-20260730 — ActionBall 分阶段训练准备账本

- 状态：`running`
- 阶段/轴：AgiBot A3，动作条件 Ball-first，N=1 诊断长训 → 正式 N=5 → N=73 → 部署
- 集成小目标：先产出一份可迭代的 N=1 policy，同时把不应阻塞首跑、但必须在后续阶段关闭的工程债写清楚
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E3`
- 创建日期/最后复核日期：2026-07-30 / 2026-08-01

共享缩写按[术语与人话对照](../../DEFINITIONS.md)解释。本文是依赖门和技术债账本，
**不是新的全项目优先级队列**；运行顺序、算力认领和当前采用 setting 仍只认
[`origin/main` 的 `NOW`](../../NOW.md)。本文只回答一件事：某项工作最迟应在首个 N=1
长训、1000 update 检查、正式 N=5、N=73 或部署中的哪一个边界前闭合。

## 0. 当前执行看板（本文唯一活跃 TODO）

### 0.1 维护规则

- 本节是本文唯一可领取、可更新状态的 TODO；[`origin/main` 的 `NOW`](../../NOW.md)
  仍是项目唯一优先级和算力队列，两者不得互相替代。
- 每次代码、合同、Pod 验证、发射状态或外部输入发生变化时，必须在同一操作批次、且不晚于
  下一次发射前先更新本节；不得只更新聊天、`PROGRESS` 或 Gate。不可替代的数字、SHA、失败原因
  和运行证据再移到后文附录。不要把事件时间线继续写进看板。
- 本轮训练决策的证据层级固定为：Franco 的显式裁定与 **2026-07-31 智元 A3 新训练 setting**
  是 DR、延迟、push 与训练流程的一手权威；逐关节 nominal 则以同一厂商的乒乓/标准 URDF、
  MJCF 与 deploy header 四份原件为更细粒度真源。Parkour regex 与原件冲突时不能把一组常数
  抹到不同腕轴上；[`dr_reward_external_diligence_20260731.md`](../../research/dr_reward_external_diligence_20260731.md)
  是本轮外部 DR/Reward/reset/PPO/吞吐/先例的主尽调；
  [`design_audit_and_speedup_20260729.md`](../../research/design_audit_and_speedup_20260729.md)
  只作上一版设计与加速历史底稿。这里采用的是厂商原件，不是用仓内自创旧值反压智元。
- 状态只用 `IN_PROGRESS`、`READY`、`BLOCKED`、`LATER`。完成项从本节移到
  [已完成证据附录](#2-已完成证据附录-a不参与调度)，不在看板长期堆积。
- `BLOCKED` 必须写清缺的输入；`IN_PROGRESS` 必须写清唯一下一动作和验收条件。任何后文边界、
  决策或历史记录若未在本节出现，都不是当前可领取任务。
- 测试期的 source-only/focused test、工件计算和相互独立的 Pod 证据可并行；不为了仪式串行。
  但上游 bytes 决定下游 SHA 的 identity→recipe→smoke 链不可倒置，关节硬边、finite checkpoint、
  no-clobber 和真机 no-publish 仍是硬门；“简化 gate”不等于跳过身份或安全证据。
- 高频问题直接入口：完整 194-D 顺序、177-D 前缀、`action_one_hot`、
  `base_ang_vel` 3-D / 姿态 6-D、task clocks 与球拍残差见[§1.2](#12-观测合同相对-task--绝对桌体上下文)；
  bang-bang 与“不新增 acceleration governor”见[§1.6](#16-bang-bang当前只用-action-change-惩罚不排产硬加速度-governor)；
  报告 §1–13 与智元设定的逐来源取舍见[§4.1](#41-本轮外部来源证据决策矩阵)和[§4.2](#42-跨来源综合决策)。

### 0.2 Now — 厂商 deploy nominal + 新智元训练 setting 重物化后 fresh N=1

**打开本文先看这里（2026-08-01 06:38 CST 当前快照）：**

- **四轴 Hctrl 机械门已在 exact clean source
  `62e0878ac52748373838850faf02c3be1c9f16bc` 关闭**；Pod 六文件 torch 组合
  `377 passed`，v7a receipt status=`PASS`。canonical/file/log SHA=
  `79e14853…/cb0fcfdc…/087028bf…`，已独立复算 canonical 一致。
- v7a 完成 8 组轴×方向差分：每组 OFF tick1 进 `[Hctrl,Hmech)` 且四 tick 内触/穿
  Hmech，全部 8 组 ON 的 32 ticks 严格位于 Hmech 内；64/64 contact=`0 N`、8/8
  full input tape exact、restore exact、four live proofs exact、public/run readback SHA 相等。
- 下游工件的 exact clean source 已固定为 `f0889ab12ca015d25207505dbf82a9811d9f31b0`；
  它相对机械 PASS source `62e0878a…` 只有文档变化，科学/运行时 bytes 不变。
- Pod1 GPU0/GPU2 的两条 identity recipe 与 `1 env × 2 update` smoke 已各自自然退出；
  四份 checkpoint finite，194-D contract 与 delay/ABI/std runtime marker 完整。loop/block live contract
  SHA=`912381de…7996c` / `19d4cfed…31ea`，required identity SHA=`a4360e39…5ebde` /
  `78267461…acde7`，四份 r4 工件已跟踪并回填 registry。Pod1 缺失的 reviewed private
  OpenGL 已从 Pod2 同内容只读同步，SHA=`9a0a6024…3a98d4a`，没有改系统库。
- required-identity/runtime-authority focused gate=`86 passed`；clean `C_RI`=
  `3c46a44d110bd7e6c1e4ebc63f535c0369065147` 已推送。
- exact `C_RI` 的 loop/block runtime authority 已并行物化：file SHA=
  `2b96d2ea…f458` / `226d3788…86ae`，三类 authorization 均为 false，已回填 registry。
- authority dependency-light validator/focused gate=`58 passed`；clean `C_AUTH`=
  `cd2375c7245df2cee2d9184b7eec4ae3ed32398b` 已推送。宿主 torch 组合因本机无 torch 未收集，
  将在 Pod 的 exact `C_AUTH` checkout 上补跑，不把未执行当 PASS。
- Pod torch focused gate=`72 passed`。loop/block dynamic-ready candidate 均已成功，file SHA=
  `a314f0b2…1b2a7` / `b0f92092…1c06f`，minimax hold utilization 均为 `0.931558…`。
- loop/block nominal hold 均为 PASS：各 `0.8 s / 160 physics steps / 40 policy steps`，
  plant contract exact、双脚接触率 `1.0`、无 terminal/truncation；receipt SHA=
  `5acee65a…a16a` / `298beec2…187c`。
- loop/block contact bundle 均 status=`PASS`，bundle SHA=`a57c3ca3…008f3` /
  `26931c76…ac673`；candidate/hold 和各四份 content-addressed bundle 子件已跟踪，registry
  已回填。
- bundle/consumer 宿主门=`165 passed`，同文件 `4` 个 full-scope preflight 因宿主无 torch
  未通过（不记 PASS）；clean `C0=ba1951657bdc22506e2011b922b5dbf48f75f711` 已推送。
- Pod exact `C0` bundle/consumer torch gate=`183 passed`，包含宿主未能执行的四个
  full-scope preflight；当前工件链的测试缺口已关闭。
- exact `C0` 三个 zero-PPO 任务均 accepted 且自然退出：A loop-static policy=
  `edfffec3…8714f`，B block-static policy=`44c20720…b870f`，C loop-adaptive effective Reward=
  `6520f153…3db26`；全部 0 PPO/0 checkpoint/未授权。C 与先前同科学配方 Reward SHA 一致，
  A/B 因 current r4 bundle/runtime identity 变更而必须更新。
- launcher/pin focused gate=`98 passed`；clean `C1=cc0020e21ed93731ce48399927af5a69e8b1368e`
  已推送。
- 三 lane `4096×5` immutable spec/plan 已在 exact `C1` 生成并通过 runtime-asset 合同：
  A loop-static claim=`6fbbd580…bfe5a`，B block-static=`19418caf…83334`，C loop-adaptive=
  `74fe2dd9…d7e64`。A/B 分别绑 Pod1 GPU0/GPU2；C 排 GPU0 在 A 自然退出后立即接跑。
- A 的首次 launch 在 PPO 前被 Hydra fail-loud：launcher 将新增 table attribution 键写成
  `task.table_contact_attribution_diagnostic=true`，而 task YAML 无该静态键，必须使用
  `+task.table_contact_attribution_diagnostic=true`。该 namespace 已 spent，无 PPO/无 checkpoint；B/C 未发。
- **Franco 最新权威裁定作废当前 C1 plant**：智元新 A3 表相比仓库旧 nominal 是更高权威；
  因此采用 waist-yaw Kp `80`、waist-pitch effort `115`、全部 wrist roll/pitch/yaw
  `Kp/Kd/effort/armature=30/2/24/0.004968`，并按 `0.25×effort/Kp` 重算 action scale。
  旧 `85/118/20/6/0.000810...` 不再作为今晚 N1 plant。
- **当前唯一下一动作**：合并智元 nominal + launcher 追加语义两个确定性修正，
  先产生 clean source，再在 Pod 跑并行 focused tests 与一次最小真实构造；后续是否重签
  Isaac plant/runtime identity 工件与 A/B/C pins，由直接 MuJoCo 尽调的工作量结论决定。
- **验收统一在 Pod，本地不记 PASS**：当前本地 `3 + 78` focused 只是快速语法/逻辑提示，
  不用于放行。两项修复先形成 clean commit，再在 Pod 的固定 Python/Torch/Isaac 环境并行跑
  source-focused suite + 一次最小真实构造。不为每个小修重复全套 recipe→smoke→probe；
  引擎选定后用一次 `4096×5` 最小安全门后直接进 `20000-iteration` long。
- **bang-bang 审计补漏（可迁移、直接修）**：vendor leaf 继承 `reward_pack=v2` 时会默认注入
  `action_acc_weight=-0.05`；这虽非硬 governor，仍是 raw action 二阶差分罚，与“只用
  action-change 惩罚调到不抖”冲突。今晚 vendor profile 显式钉 `action_acc_weight=0.0`，
  保留 `action_rate_clamped=-0.2`；不做学习 A/B，Isaac/MuJoCo 两引擎必须一致。
- 后续唯一流水线：三 lane `4096×5` →
  `4096×32` push evidence → 两条反手 static + 一条 loop adaptive long。
- 科学配方与 r4 identity→authority→candidate/hold→bundle→A/B/C pins 已全部固定；
  旧 pin 会因 source/plant bytes 变化而 fail-closed。
- Pod1 GPU1 的 PID `1259856` 属于既有任务，严禁触碰；GPU0/GPU2 在本次 plan 前分别仅
  `18/2 MiB`。Pod2 GPU0/1/2 全部有既有训练，本轮不使用。每次发车前
  重新核对 sidecar/NVML，只消费具名可用槽。

<details>
<summary>v5→v6→v7 与 r4 的证据时间线（默认折叠；不参与当前调度）</summary>

**2026-07-31 实时覆盖：**四轴 [`Hctrl`](../../DEFINITIONS.md#h-ctrl) source 候选、schema-3/authority/dynamic-ready/runtime consumer
与负例已闭合（scoped `288+7+47` tests PASS）；16-env v5 差分 stress 补全后的
`53` 项 probe tests 与 `141` 项组合复核全过，独立复核 P0/P1=`0`。table first-hit + exact
[OBB-vs-AABB](../../DEFINITIONS.md#obb) [SAT](../../DEFINITIONS.md#sat-collision-test)
候选已通过 Pod `51+3+70` 专项 tests，top-level post-init seam 与 code-owned probe-only launcher
已接好并通过 `88` tests；独立复核同样 P0/P1=`0`。两项均未改
Reward/PPO/`0.5%` table Gate，也尚未授权 long。
**独立红队动态：** probe-only 开关造成 probe/push scientific identity 不等的 P0，以及
16-env stress “只对账目标轴却声称整条 input tape exact”的 P1 均已关闭；
v5 现在对账 31-D 初态/每 tick qdes、origin-relative root 与隔离外部刚体，validator
独立重算 full-state/pair digest。**当前唯一共享前置已前移到 v7 receipt 可信链闭环：先关闭
joint/name→live `joint_index`、validated runtime→receipt tape、literal schema/kind/token、
runtime joint order→独立 PhysX public live-limit identity、identity→实际 live readback attestation 五个 P1，
再从 fresh clean commit 跑一次 live v7a；PASS 后立即并行重物化 two-action
required identity/authority/candidate/hold/bundle，再产 three policy pins 并重跑三 lane
`4096×5`。**
**Pod 实时进度：** clean detached `9819a8623a913d472fc764cef8d0c9f1a4f8ee83`
已建立，依赖完整的 CPU/torch/hydra 组合门新增 `259+313+196` tests PASS。首次 v5
调用传入 8 位缩写 SHA，被 source-identity 在 Kit/GPU 前正确拒绝，无 receipt/无模拟状态
副作用；失败 log 保留，现用完整 40 位 SHA 重跑。
40 位 SHA 重跑已到 vendor profile bind，又在 `gym.make`/物理前 fail-loud：
`robot_hit_table` 新的 `table_contact_attribution_diagnostic` 尚未进 `train.py` task-first exact
parameter set。spent FAIL=`ea42ce39…` 保留。strict set 已同步，且 top-level/term 均要求 exact
bool、command=`racket_target`、非 ActionBall 禁止开启；缺键/extra/非 bool/不一致/command
漂移/legacy 开启的行为负例 `10 passed`，相关组合 `13+336` PASS，独立复核确认
无第二个 full-param strict consumer。修复不改 diagnostic default、terminal、Reward 或 Gate；
现产新 clean commit 重跑，不修改/覆盖 spent receipt。
clean `8d0b8ba09ee50feb7883428d9d3bf4e91f618f74` 的 Pod 组合 `349` tests PASS；
v5c 已完成 scene/16 env/四轴 Hctrl live 安装与 readback，在 reset 前被旧 action manifest
的 solver profile SHA 正确拒绝（manifest=`af4f6f95…`，runtime=`f89587db…`），spent
FAIL=`a0f8d352…` 保留。这确认 source 变化后旧 profile/manifest/pins 不可复用，也修正了
原调度顺序：**formal profile pinner → 新 action profile/manifest 身份 → Pod live v5 →
其余 authority/candidate/hold/bundle/policy pins**。不在 stress 里绕过 solver gate。
formal pinner 已在 clean `8d0b8ba0…` 完成：物理数值画像 SHA 仍为
`aa5c9085…`，新 solver SHA=`f89587db…`，跟踪的 profile 文件
SHA=`509f3812…`。现正打开无覆盖 `r4` 制品纪元：保留两个动作的 stable
motion/source manifest，但 identity 以及后续 authority/candidate/hold/bundle pins 全部
fail-closed 置空。P1 只要求 profile/identity materializer 定向测试通过；此时
launcher/probe 因下游 pin 未产生而拒绝是预期状态，不得用旧 r2/r3 文件补空。
P1 source=`6a7587c05bc3fdb6c6070b72da12e251ce58795b` 已 push；两个独立 clean
worktree 并行产出 loop/block 各三件 r4 identity，两边 materializer 各 `19 passed`。
loop prototype/manifest/receipt=`d3458ee3…/e7531567…/3c79f266…`；block=
`19bb7aed…/7870a053…/9ffe90cc…`；producer 共同=`b90bac5f…`。当前
只回填这一层 registry，下游 pins 继续为空；定向回归过后提交 P2，
再用 clean P2 跑 Pod live v5。
P2 source=`e7917b1479980fb9c85e89b25c011ae9b1e52f38` 已 push，r4 消费端
`31/31`、identity+stress host 组合 `72/72` PASS。Pod v5d 已在 GPU0 用
16 env 自然执行并出具 spent FAIL receipt：content SHA=`2d10999c…`（已独立
重算一致），restore exact=`true`。行为轨迹上，腰 roll/pitch 四个 OFF 在 tick1 入
`[Hctrl,Hmech)` 且四步内均触/穿 Hmech，对应 ON 均留在 Hmech 内；但这还不是
可签署的因果 PASS：红队重读 failure evidence 发现 reset 后各 env 的 root x/y 不同，
validator 又在踝部 verdict 早抛错，尚未执行后置 root-pair exact 门。因此腰部
只记为行为观察，不能声称 ON/OFF 唯一差变量已证明。同时，左/右 ankle-roll
的 ON/OFF 在当前各自 tape 下都被 ground/contact 位置投影大幅弹回内侧，
tick1 未入正控带，故整体不放行。这是 **ankle OFF positive-control 未施加成功**，
不能写成 ankle Hctrl PASS/FAIL；当前唯一修复是在保持 ON/OFF 全 31-D 输入
exact 的前提下，先给 16 env 写入同一 origin-relative 空中 root pose、清零 root 6D
速度并证明四步外部 contact force 为零；同时把初始 full-input pair exact 检查前置到
任何动力学 verdict 之前。若离地后踝轴 OFF 仍不触 Hmech，再逐轴提高外向应力；宿主负测后
用新 source 和新 receipt 重跑；不改 2% Hctrl、不放宽 verdict、不覆盖 v5d。
v6 source candidate 已完成：16 env 共用 exact origin-relative root=`[0,0,3 m]`/
identity/零 6-D 速度，每 tick 从 `contact_forces` 强制全机器人外部接触力
`<=1e-6 N`，且所有 initial q/qdot/qdes/root/外物与 tick qdes/外物 pair parity
都在 outcome verdict 前先验。schema/kind/confirm token 全部升为 v6，host 正负测
`66 passed`；额外加入“ON/OFF 同步漂移但共同偏离声明 tape”的对抗用例与 live
write/step/update/contact-read 顺序断言；本机无 torch，提交 clean source 后在 Pod 补跑 torch
组合并用 fresh v6 receipt。与 r4 identity smoke 合跑 `97 passed`，两轮独立只读终审均为
P0/P1=`0`。
clean `ff41b12c…` 的 Pod 先过 torch 组合 `361 passed`，随后 v6a 自然产出并保留
spent FAIL：canonical/file/log SHA=`c9c56bde…/52ca07fa…/e6d4adf0…`，restore exact，
80 个 full-state seal 全部可复算。64/64 contact 样本精确 `0 N`，全部 8 对 initial/tick
input raw pair exact；FAIL 首因只是 PhysX 把写入的 identity quaternion 规范化为全 16 env
相同的 `[2.73e-11,2.87e-11,-6.62e-13,1]`，物理转角约 `7.9e-11 rad`，旧 validator
却要求 component 字面为零。failure telemetry 同时显示四个 ankle OFF 已不再被地面反弹，
tick4 距 Hmech 仅 `0.000308–0.000318 rad`，但仍未触边；腰四对则满足差分行为。
下一 source 版本化为 v7：root 声明门采用 quaternion 双覆盖下的物理角误差 `<=1e-9 rad`，
ON/OFF raw root 仍必须 byte-exact；只把 ankle 两轴的 kinematic outer fraction 从 `0.60R`
调至 `0.65R`（增加 `0.05R/5ms` 初始外向速度），腰保持 `0.60R`，四 tick/Hctrl/Hmech/
contact/verdict 全不变。先补复合负例和独立复核，再用 fresh v7a；不得覆盖 v6a。
v7 初轮 host 正负测为 `71 passed`、与 r4 identity 合跑 `102 passed`；新增逐行重算
q0/qdot/5-ms point/Hmech gap、quaternion norm=`1e-12` 与“物理同姿态但 raw pair 异号”负例。
独立红队三轮共抓到五个发车 P1，并严格按 **先更新本账本、再改代码** 的顺序全部关闭：
（1）每行 exact keyset 且 `joint_index == live_joint_order.index(joint)`；（2）receipt 组装前
用 runtime 内封存的 31 关节顺序、observations、diagnostic 与当前 tape 完整重跑 validator，
禁止“先验 runtime、后换 tape”；（3）literal pin v7 schema/kind/confirm 并拒绝旧 v6 token，
同时清除 Gate/操作文档中的 live-v6 旧指令；（4）runtime 的 31 关节顺序不得自证，必须与
`live_limit_identity.public_contract_joint_order` 及 selected names/indices exact 对账，并用该
独立顺序做 receipt-time revalidation；（5）上述 identity 自身必须证明 public/live Hctrl 与
articulation Hmech readback exact、OFF 只改目标轴，public/run-specific SHA 相等，setter/mixed/
joint-order digest 均可重算且为 canonical SHA-256。最新 focused=`82 passed`、
identity+probe=`113 passed`、`py_compile` 和 `git diff --check` PASS；独立终审确认
P0/P1=`0`，允许提交/push 后从 exact clean checkout 跑 Pod v7a。tape/runtime/identity 三方同时
恶意重签属于 clean-source attestation 信任边界，不伪称 receipt 内循环能防御。
踝部 `0.65R` 是为一次 fresh Pod 提高正控成功率的有界工程裕量，
不宣称数学或经验上的最小值。
clean successor `62e0878ac52748373838850faf02c3be1c9f16bc` 在 Pod 六文件 torch
组合 `377 passed`。no-clobber v7a 自然退出，receipt canonical/file/log=
`79e14853…/cb0fcfdc…/087028bf…`，status=`PASS`。独立复算 canonical 一致；
8/8 OFF 组进带并触/穿 Hmech，8/8 ON 组的 32 ticks strict Hmech，64/64
contact=`0 N`，8/8 initial+tick input tapes exact，restore/readback/four live proofs exact。因此
四轴 plant 机械 stress 门关闭，但它不代签 table attribution、fresh probe 或 push/long Gate。

</details>

| ID | 状态 | 当前交付 / 唯一下一动作 | 完成验收 | 阻塞输入 | 证据入口 |
| --- | --- | --- | --- | --- | --- |
| N1-DIAG-PROBE | `PAUSED-ENGINE` | exact C1 与三 claim 已因智元新 nominal 裁定作废；A 只在 PPO 前暴露 Hydra append 错误，无 checkpoint/PPO。旧 table/right-ankle 失败只作根因输入 | 只收口 launcher append 和智元 nominal/action-scale 两个确定性修正；不再为即将迁移的 Isaac/PhysX receipt 链新增 feature。等直接 MuJoCo 尽调裁定后，在选定引擎生成 fresh probe | 可迁移门保留：checkpoint finite、194/318 normalizer、small-σ LR/std、task-clock/strike denominator、table/fall/nonfinite 分账；不新增 acceleration/jerk governor | [本轮外部尽调](../../research/dr_reward_external_diligence_20260731.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| TABLE-GUARD-ATTRIBUTION | `DEFER-ENGINE` | 保留“first-hit body/obstacle/swing-phase + conservative/exact 分账”的引擎无关需求，但当前 PhysX contact view/world-AABB/SAT 实现不再扩展 | MuJoCo 尽调后用其 geom/contact API 重接同一账本语义；默认关闭时 Reward/observation/RNG 不变 | 继续拒绝用重复 `table_hit_penalty` 代替归因，也不放宽安全 Gate | [桌碰安全 smoke](../../operations/run_action_ball_table_safety_smoke.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| VENDOR-PUSH-EVIDENCE | `BLOCKED` | GPU2 旧 claim=`55fd8bd…` 的 update0--5 push application 全为 `0`，只能作 6% shared-safety 负证据；raw actual-hard/terminal 为 `0/0、33/15、3238/1303、418/169、2036/830、844/365`，禁止 resume 或冒充 push 结果 | final clean source/spec/policy/authority/bundle；checkpoint finite；ABI/delay/std 完整；push count≥4096 且 sampled delta finite/在界；qdes-hard/actual-hard/nonfinite 零容忍；table/fall 用推撞专用 per-env-step 有界率，且 strike/swing>0；中位吞吐≤8 s/update | 四轴机械 stress 已由 v7a PASS；现等 current three pins → fresh `4096×5` 行为/安全放行后重跑。Pod2 GPU0/2 只在不停他人进程前提下用于短诊断，GPU1 属于另一用户，严禁触碰。不得用空卡绕过 shared-safety 门 | [本轮外部尽调](../../research/dr_reward_external_diligence_20260731.md) |
| N1-TONIGHT-3LANE | `WAIT-MUJOCO-DECISION` | 三条**引擎无关科学配方**保持：A=`bh_loop_c` static，B=`bh_block` static，C=`bh_loop_c` monotonic adaptive-sigma；C 从 `0.20/1.0/0.52` 锁步单调收紧到 `0.075/0.5/0.262`，共同 coarse position=`0.30`；禁止恢复 `action_one_hot` | CC 直接 MuJoCo 训练尽调回来即裁发车引擎：短迁移则三条 `20000-iteration` 直接发 MuJoCo；若今晚不能闭合，则在完成新智元 nominal 最短门后先发两条 Isaac static，C 等第三槽 | 不以停止他人进程获取第三卡；不为等迁移报告继续增长 Isaac 专属开发队列 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[观测合同](../../interfaces/policy_observation_action.md) |
| N1-LONG-GATE | `WAIT-MUJOCO-DECISION` | `5/32 update` 只是启动/安全门，不足以暴露长期学习病理；本轮明确需要训到 `20000 iterations` 观察 std/LR、Reward hacking、课程锁死、strike/return 与策略塌缩 | 引擎选定后只跑最短启动门，随即发长训；MuJoCo 与 Isaac 共用 checkpoint finite、obs normalizer、mean std/LR、episode age、strike/contact/return、table/fall/nonfinite、push 相位分账指标 | 不把 Isaac 的 authority/USD/Hctrl/contact-view receipt 硬搬到 MuJoCo；不以引擎迁移为理由取消长训 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md) |
| RUNTIME-ASSET-LOADER-V2 | `READY` | **source + Pod live 验收已闭合**：schema/kind v2 pin OpenGL/GLU 固定目录、library SHA、direct SONAME、USD closure 与 exact `OpenGL:GLU` string；`ldd` 把 `libGLU.so.1.3.1` 的 `libOpenGL.so.0` 解到 exact private OpenGL，missing/reverse/tail 三个 plan 都在 namespace 前 rc2，正确 plan 已通过 loader 层并到达下一层 dynamic-ready path 检查 | loader bytes、claim helper 与 live 私库解析已满足；后续 C0/C1 继续复用同一 absolute runtime asset tree 和 threat model，不再修改 loader | threat model 是 `pathname_sha256_revalidated_immediately_before_exec_no_concurrent_local_writers_v1`；运行资产树须 quiescent，不宣称抵御恶意本地写者。当前 blocker 已下移到下一行，不得把该行重新解释为训练放行 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[本地同步](../../operations/setup_local_sync.md) |
| DYNAMIC-READY-PATH-IDENTITY | `READY` | **current r4 source + Pod plan/zero-PPO 门已关闭**：clean C0=`ba195165…`；loop/block claims=`6593710e…` / `b857f956…`，dynamic binding=`3c1e2c07…` / `103cd03e…`；两条均自然完成、0 PPO/0 checkpoint/authorization false，policy SHA=`edfffec3…` / `44c20720…` | 路径门不再排产新 feature；C1 只能消费本行 current pins | GPU1 仍属于另一用户 PID `1259856`；不触碰。boot marker/launcher return 不等于 materialization 完成；本行已核对产物和自然退出 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| LIVE-CONTRACT-MATERIALIZER | `READY` | **current r4 三输出已闭合**：C0=`ba195165…` 的 loop/block/adaptive 均 accepted、自然退出且 SHA 可复算；三 pin=`edfffec3…` / `44c20720…` / `6520f153…` | 回填三 pin 后产 clean C1；后续 smoke 必须反向验证 exact policy/Reward lineage | `launch_a3_vendor_identity_smoke.py` 不是 post-pin lane smoke；C1 必须用 vendor baseline diagnostic | [N=1 发射工序](../../operations/run_ablation_wave_launch.md) |
| VENDOR-DIAG-TEMPLATE | `READY` | code-owned A/B/C template、current r4 三 pin、launcher focused `98 passed` 和三个 exact C1 immutable claims 已齐 | 不再开发 launcher feature；三 lane 各自 `4096×5` 自然退出、finite checkpoint、exact lineage 和共同 safety/behavior gate PASS | 1-env identity smoke 已用于物化 contract，不代替 current 4096-env probe；adaptive-KL × small σ 只在 `4096×5` 数据中定谳 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md) |

adaptive-sigma hash-only 首次 clean-source plan 已真实执行并按预期 fail-closed：旧 r3 bundle
仍 pin `hope_commands.py=4c46d997…`，新文件 SHA=`a6ccf25e…`，因此在 Kit/PPO 前拒绝；这证明
hash stage 不能绕过旧工件身份。下一次只在两腰 safety source 冻结后，先重物化 loop/block 的
action-specific bundle/identity/authority，再跑 hash-only；不手填 SHA，不为临时 source 重烧两遍。

三 lane 所称“共同最强 vendor setting”精确指厂商 deploy nominal + `[0,2]` episode-fixed delay +
vendor 幅值且保留 PACE `5–15 s` 节奏的 push，以及当前 Reward/194-D/safety。stable-ready 首车暂关
gain/mass/CoM DR，待 healthy baseline 后按§4.2 机械门逐轴恢复；这是吸收智元一手配置后的阶段化
使用，不是继续沿用仓内旧设定。

**08-01 双位置包络首个 Pod 机械 smoke：** GPU0 的 unsealed candidate checkout 已自然完成
`1 env × 1 PPO update`，iteration=`2.27 s`，`model_0.pt` 为约 `6.9 MiB`；递归检查到
`88` 个 tensor（其中 `78` 个浮点/复数 tensor）全部 finite。checkpoint 真实含
`obs_norm_state_dict` 与 `privileged_obs_norm_state_dict`，两者均有 `_mean/_std/_var/count`，
因此§8 的“运行时是否保存 observation normalizer”已由当前 rsl_rl 运行证据定谳为 **是**。
本次 `120` 个 physics readback sample 中 mechanical actual-hard、qdes-hard、H_ctrl
ballistic-attempt/capture/penetration、table/fall/nonfinite 全为 `0`，两腰四侧 minimum
signed H_ctrl gap 均为正；但它是未封印 source 的机械 smoke、尚未跨 `t_hit`，不能替代
clean-source `4096×5`。红队发现 receipt readback digest 混入 startup 后可合法变化的 default-q；
当前先修正 digest 语义并补 host 回归，再合入统一 source，不能把旧 digest 写入正式 receipt。
### 0.3 Next — long 已运行后的判读与 formal N=5 前置

| ID | 状态 | 当前交付 / 触发条件 | 完成验收 | 阻塞输入 | 证据入口 |
| --- | --- | --- | --- | --- | --- |
| N5-INTENT | `LATER` | formal N=5 前把 v2 的 N1-only 无身份合同替换为**固定宽、内容生成的连续动作意图**；首版使用归一化 `q_ref_at_hit-q_ready` 与 `teacher_rate*qd_ref_at_hit`，共享 ready 有混叠才加中间相位 preview | actor/critic 同源；宽度不随 N 变化；跨 N tensor/order、动作间距离、混叠检查和 Pod 构造 parity 通过；禁止恢复 N 维 one-hot、UID 数值或 per-slot embedding | exact N=5 ordered motion/reference；N1 不被此项阻塞 | [ActionBall 合同](../../interfaces/action_conditioned_ball_first_contract.md)、[Policy 观测合同](../../interfaces/policy_observation_action.md) |
| N5-CURRICULUM-FIX | `LATER` | 正式 N5 curriculum 之前把§13 两个 critical 缺陷直接修成默认兼容的新合同：R1 单臂决定可休眠/到期重测，R2 评估窗独立配额提高 new-band 样本，R3 样本不足作废重测，R4 全域 safety hold 与当前 arm 锁定分层，R5 rho/center 滞回且 zero-tolerance 立即旁路 | 旧 key 缺席时行为逐字节不变；checkpoint/state SHA 新旧版本可验迁移；Wilson 算术、canary/heldout 冻结、全域 hold、new-band 归因与 zero-tolerance 负例通过 | 不阻塞 fixed-domain N1；需 N5 manifest 与专项 source 实现 | [本轮外部尽调§13](../../research/dr_reward_external_diligence_20260731.md) |
| N5-CURRICULUM-POWERON | `LATER` | 采纳 R6：N1 谱系继续明确为 fixed-domain；正式 N5 long 前先跑一条专门短 N5 curriculum 验收，不把整套机器首次通电押在 20k–25k long | 至少产生 1 次真实 marginal 决策、1 次 drain/reset、1 次 checkpoint→resume 且 curriculum SHA/state 不漂移；样本不足不误锁、全域 blocker 不误伤单臂 | 依赖 N5-CURRICULUM-FIX 与 exact N5 ordered manifest；N1 不被阻塞 | [formal 发射工序](../../operations/run_action_ball_curriculum_no_clobber.md)、[本轮外部尽调§13](../../research/dr_reward_external_diligence_20260731.md) |
| N5-CURRICULUM-SAMPLING | `LATER` | R8 已认证域内失败加权出题采纳为训练侧设计（认证窗仍冻结固定混合）；R9 形态1 的 2–3 臂同窗并行与 `cell_id + probed_arm` 行标签纳入后置设计 | R8 有 `≥10%` uniform 与 center 保底、per-cell + mix-standardized 双指标；R9 每 env 恰探一臂、每臂 Wilson 独立、族配额优先护台面积；certified 护台面积 KPI 可复算 | R8 不阻塞 N1；R9 硬依赖 R1+R2+R4，ADR 式跨窗队列形态2 不排产 | [本轮外部尽调§13.6](../../research/dr_reward_external_diligence_20260731.md) |
| N5-RECIPE-ID | `LATER` | formal N=5 前把 policy bootstrap identity 从绝对 checkout path 改为 candidate/hold 的 content/file SHA；路径只保留在 runtime hard contract | 同一 bytes 在不同 clean checkout 得到相同 policy SHA；任一 artifact bytes/binding 改变仍 fail-closed；旧 recipe 可审计但不静默重标 | 不阻塞当前 exact-path r4；需 recipe schema 迁移与负例 | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| N5-RECEIPT | `LATER` | formal N=5 前把正式审计从 per-env reset 仪式改为 device 紧凑 event tape + checkpoint/hourly 批量物化；同时清理残余 D2H、ledger clone、broker per-env Python | 旧式完整 receipt 可逐字段重建；checkpoint save/load/no-step、跨进程 exact resume、篡改负例与固定工作量吞吐验收通过；proposal/reject/action/domain/lifecycle/outcome 分母不丢 | 分段 profiler 和可用 Pod | [formal 发射工序](../../operations/run_action_ball_curriculum_no_clobber.md)、[设计与加速审计](../../research/design_audit_and_speedup_20260729.md) |
| N5-TEARDOWN | `LATER` | formal N=5 前让 `gym.make()` 之后的全部 pre-run/hard-contract 异常也在 `finally` 中关闭 env，避免未关闭 PhysX/Gym env 让 Kit teardown 自旋到 watchdog | 注入 pre-run 异常时 env close 被调用一次、进程及时退出、GPU/全 Pod boot lock 自然释放；正常 recipe-only 与训练行为不变 | 不阻塞当前 r4；需 Pod failure-path smoke | [N=1 发射工序](../../operations/run_ablation_wave_launch.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |
| N5-PHYSICS | `LATER` | formal landing 或 N=5 前显式选择 2026-07-30 OptiTrack ball-physics profile，重新物化 physics/solver/question bundle；zero-weight term 做结构裁剪，不能以“权重为零”冒充未解析 | physics/solver/question SHA 重钉；Isaac/solver 配置 parity；active Reward/term ledger 可复算；未充分辨识切向参数明确保持 prior/canary；首次 physical-ball-contact scoring 前把球拍排除出全身 restitution DR 并钉实测值 | 现有科学源可先工程化；最终 table/tangential 与 racket restitution 参数仍需 OptiTrack/物理接触补测 | [G05](../../gates/G05_isaac_training_first_loop.md)、[Reward 因果审计](../../operations/run_action_ball_reward_causal_prelaunch.md) |
| N5-LAUNCH | `BLOCKED` | 上述 N5-INTENT、N5-RECEIPT、N5-PHYSICS、N5-CURRICULUM-FIX 与 N5-CURRICULUM-POWERON 通过后，绑定 exact ordered N=5 manifest、逐动作 admission/ball support/new-forehand 安全证据并 fresh 发射 | formal receipt 绑定 continuous intent、motion bytes/order、Reward/PPO/plant/solver/physics/table/evaluator、已通电 curriculum 与 exact resume；clean/no-clobber lineage | Franco 确认 exact 五动作顺序；新正手采用/站位裁定 | [formal 发射工序](../../operations/run_action_ball_curriculum_no_clobber.md)、[G05](../../gates/G05_isaac_training_first_loop.md) |

### 0.4 Later — N=73 与部署边界

| ID | 状态 | 最迟边界 / 交付 | 完成验收 | 必须由人或硬件提供 | 证据入口 |
| --- | --- | --- | --- | --- | --- |
| N73 | `LATER` | N=73 发射前完成 exact ordered 73 manifest、逐件 compiler/safety/admission、动作专属 ready/ball center/support、full-body 与 fixed-width continuous intent 的任意 N 压力门 | 73 件 order/UID/motion bytes 全钉；逐动作 frontier 与 heldout 独立；broker/curriculum/checkpoint/exact resume 压力通过；不得由 N5 续成 | 若 repo/Pod 制品不完整，需提供缺失 bytes、证书、站位元数据和人类采用顺序 | [动作库终审](EXP-MOTION-CANONICAL-LIBRARY-20260723.md)、[ActionBall 合同](../../interfaces/action_conditioned_ball_first_contract.md) |
| DEP-OBS | `LATER` | 部署意图重训前落地 OptiTrack pose + gyro angular velocity + causal base linear velocity producer；用新合同加入 localization age/valid 和实测 noise/latency/dropout | capture/source/receive/consume timestamp 链、marker→`base_link`、venue→table、gyro frame、marker→COM 修正、hold-last/stale 语义及 Isaac/MuJoCo/C++ parity 全过；不得伪装成 194-D | OptiTrack v2 记录、Motive smoothing、遮挡/dropout、两组 SE(3) 外参、gyro bias/对齐 | [Policy 观测合同](../../interfaces/policy_observation_action.md)、[G07](../../gates/G07_mujoco_to_real.md) |
| DEP-PHYSICS | `LATER` | 任何正式 landing/真机前闭合 table effective restitution 与切向参数，并让训练/planner/MuJoCo/C++ 共用同一冻结接触模型 | 新测量 receipt、拟合 YAML、跨引擎 golden parity 和 binary metadata 拒载负例齐全 | OptiTrack 30 分钟落球/反弹补测及 Motive 配置 | [G03](../../gates/G03_data_processing_and_physics_calibration.md)、[G06](../../gates/G06_isaac_to_mujoco.md) |
| DEP-CONTROL | `LATER` | 当前只保留封顶的一阶 action-change 惩罚 `action_rate_clamped=-0.2` 和 q/qdes/projection 硬安全链；vendor leaf 显式将二阶 raw-action `action_acc_weight` 钉为 `0.0`。不排产 acceleration/jerk governor、EMA 或 Ruckig 工单；只有 A3 真机测量证明既有 q/effort/velocity 边界不足时才重开 | 按 1000-update 检查 executed-command 换向率、teacher phase delay、strike/return、拍速、imitation 分账；二阶差分只作零权重遥测，不进 total Reward | 只在重开 governor 时需 A3 真机 command acceleration/jerk 证据；常规部署仍需 PD/effort/torque-speed/delay 与 Franco 真机放行 | [G06](../../gates/G06_isaac_to_mujoco.md)、[G07](../../gates/G07_mujoco_to_real.md) |

## 1. 第一性原理裁定

### 1.1 先产 policy，但不能在错误合同上产

首个 N=1 训练只需要证明：

1. 动作身份在一拍内冻结；
2. 来球先采样，再由同一动作的 fixed-action solver 生成相对 task；
3. 出生、初始 `qdes`、上一动作、actor 观测和 teacher frame 0 描述同一 ready；
4. episode 能活过该动作 `t_hit`，checkpoint finite，table/fall/nonfinite/raw-hard 分账可信；
5. 训练设置、policy recipe、spec 和结果记录可复算。

它不需要等待正式 N=5 的完整审计基础设施、最大课程 support、N=73 动作 admission、
MuJoCo/C++ deploy consumer、full-body 对照或最终 command governor。反过来，观测列错位、
task 变绝对坐标、桌碰传感器恒零、出生姿态与控制目标冲突会直接改变 policy 学到的问题，
这些不能用“先跑再说”绕过。

### 1.2 观测合同：相对 task + 绝对桌体上下文

正在运行的 compatibility 合同是
[`action_ball_table_pose_twist_heading_task_n<N>`](../../DEFINITIONS.md#action-ball-table-pose-twist-heading-task-contract)；
fresh 首选合同是
[`action_ball_table_pose_twist_heading_task_teacher_start_v2`](../../DEFINITIONS.md#action-ball-teacher-start-contract)。
N=1 fresh actor 宽度为 **194**。下表就是 trainer 的 exact ordered terms，不再用
“177-D 本体状态”这种不准确简写；177 里同时有 teacher command、本体、上一动作和 task：

| 区间 | 项 | 维度 | 语义 |
| --- | --- | ---: | --- |
| `[0,62)` | `command` | 62 | 当前 teacher/reference phase 的 31 关节位置 + 31 关节速度；ready wait 期间冻结在 reference frame 0 |
| `[62,68)` | `motion_anchor_ori_b` | 6 | teacher 躯干相对当前 base 的姿态误差的连续 6D 表示 |
| `[68,71)` | `base_ang_vel` | 3 | pelvis/body frame 的 x/y/z 轴角速度（常叫 roll/pitch/yaw-rate channels，不是 Euler 角导数）；**角速度本来就是 3 值，不是 6** |
| `[71,102)` | `joint_pos` | 31 | 各关节 `q-default_q` |
| `[102,133)` | `joint_vel` | 31 | 各关节 `dq` |
| `[133,164)` | `actions` | 31 | 上一个 policy raw action，不是当前关节位置 |
| `[164,167)` | `projected_gravity` | 3 | base frame 中的重力方向 |
| `[167,169)` | `base_target_pos_b` | 2 | 当前 base 到目标站位的 yaw-heading XY 残差 |
| `[169,172)` | `racket_target_pos_b` | 3 | 实时球拍正向运动学（forward kinematics，FK）位置到目标击球点的 yaw-heading XYZ 残差 |
| `[172,175)` | `racket_target_vel_heading` | 3 | task 要求的球拍线速度，不是实际球拍速度 |
| `[175,176)` | `time_to_strike` | 1 | 当前 teacher phase 到 `t_hit` 的剩余秒数 |
| `[176,177)` | `swing_type` | 1 | reference 的正手 `+1` / 反手 `-1` 旗标；N=1 下仍是 reference 语义，不是 action slot ID |
| `[177,180)` | `base_position_table` | 3 | base 在桌面中心坐标系中的 XYZ |
| `[180,186)` | `base_orientation_table_6d` | 6 | base 相对桌体的完整 roll/pitch/yaw，连续 6D 旋转表示 |
| `[186,189)` | `base_lin_vel_heading` | 3 | base/root COM 在 base yaw-heading frame 的三轴线速度 |
| `[189,193)` | `racket_target_normal_cmd_heading` | 4 | fixed-action solver/planner 给的 raw-A 有符号目标拍面 normal(3)，加当前保留且填 0 的 `rho`(1) |
| `[193,194)` | `time_to_teacher_start_s` | 1 | 同一 Motion phase governor 中，老师离开 ready frame 前的剩余秒数 |

新鲜 actor 中**不存在 `action_one_hot`**。动作 UID/slot 只在 sampler/solver/curriculum/receipt
中冻结，不进入 policy。历史 N1 one-hot 恒为 `[1]`、严格不含信息；formal N5/N73 前用
固定宽 continuous action intent 补充未来动作内容，而不是恢复槽位标签。旧 `193+N` /
`194+N` one-hot 合同只保持历史读取。`time_to_strike` 已是 `t_hit` 时钟，
`time_to_teacher_start_s` 是 ready 等待时间；demanded face/velocity 都是 task 输入，不是实际拍状态。
实际球拍不在观测里重复放一份绝对 pose/twist：球拍相对 base 的 pose/twist 由
31-D `q/dq` + 冻结机器人运动学唯一决定，桌系绝对量再由 base pose/twist 唯一补齐；
`racket_target_pos_b` 又每步用同一份 live FK 算“目标减实际”，所以与本体状态是同一时刻、
同一物理状态的可复算约束，不是两套可以互相打架的拍状态。三个角度
没有被删掉：6D
旋转用旋转矩阵前两列表示完整
SO(3)，经正交化后第三轴唯一确定；它不是 yaw-only。采用 6D 而不是 Euler 三角或裸 quaternion
是表示合同修复，不是 Reward 假设，不做学习 A/B。依据是 Zhou 等人在 CVPR 2019 的
[连续旋转表示结果](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)。

task 位置继续保持机器人相对坐标：

- `base_target_pos_b` 是当前 base 到 base goal 的相对二维残差；
- `racket_target_pos_b` 是当前球拍 FK 到目标击球点的相对三维残差；
- 桌体 9 值回答“机器人相对球桌站在哪里、朝向如何”，不把 task 改回 world absolute。

同一个 actor 内的击球 task 不再混 frame：

```text
p_task_h = R_heading^T (p_target_table - p_racket_FK_table)
v_task_h = R_heading^T v_target_table
n_rawA_h = R_heading^T n_rawA_table
```

旧 `action_ball_table_pose_twist_n<N>` 同为 194-D，但 velocity/normal 仍在 world frame，
只保留兼容读取；同宽不代表同合同，旧 checkpoint 不得 resume 到新名称。planner wire、
fixed-action solver、Reward 和物理真值仍留在 canonical table/world frame，转换只发生在 actor
观测边界。这是确定性坐标合同修复，不做学习 A/B，只做 Pod tensor/构造 parity。

这种拆分同时满足泛化和可观测性：相同相对 task 可以在不同站位复用，而 policy 仍知道自己与桌边、
桌面的绝对几何关系。

### 1.3 传感器权威

部署侧按物理量拆分，不做“整套 OptiTrack 或整套 IMU”二选一：

| 量 | 维度 | 首选权威 | 原因 |
| --- | ---: | --- | --- |
| table-relative base position | 3 | 360 Hz OptiTrack rigid-body pose | 无积分漂移，直接给外部位置 |
| table-relative base orientation | 6 | OptiTrack orientation 经 marker→`base_link` 与 table SE(3) 标定 | 直接给全局姿态和 yaw，无 IMU yaw 漂移 |
| `projected_gravity` | 3 | 同一 OptiTrack orientation 派生 | 与绝对姿态使用同一权威，避免静默混源 |
| `base_ang_vel` | 3 | pelvis IMU 三轴陀螺仪 | 直接测角速度；mocap 姿态差分会放大噪声并引入滤波延迟 |
| `base_lin_vel_heading` | 3 | 因果状态估计器：OptiTrack 位置作无漂移锚，可融合 IMU 加速度 | 单帧 mocap 不直接给速度，纯差分噪声大；纯 IMU 积分会漂移；最终旋入与 task 相同的 yaw-heading frame |

所以 `base_ang_vel` 是 **3 值**，`base_lin_vel_heading` 也需要 **3 值**。线速度在当前无 history 的
feed-forward actor 中能区分“同一姿态、正在倒/正在平移”和“静止”，因此值得进入 actor；
但部署 producer 必须处理 marker 到 root/COM 的位移以及旋转刚体的
`omega × offset`，不能把 marker 速度冒充 COM 速度。

训练中这些量使用 simulator rigid-body truth 的同语义 counterpart。真实 producer 尚未闭合
marker rotational extrinsic、时间同步、丢帧/延迟和 causal velocity estimator，所以当前 v2
训练合同目前不授权真机。

2026-07-30 对近期分支的复核给出三条新的硬边界：

- 近期 OptiTrack producer 不在过期的 `origin/jiayi`，而在 `origin/hitter` /
  `origin/hitterobs`；当前 live P1 rigid-body 是 `360 Hz`、米制、完整 6-DoF；
- 当前消费的 `NamedPoseArray v1` 只带 ROS 到达时间，不能分解相机曝光、Motive 解算和网络延迟；
  驱动已有携带 camera mid-exposure 与 Motive latency 的 v2 消息，部署意图标定必须切到该
  timestamp 路径；
- `origin/hitterobs` 已有“只在收到新 pose 时按真实 `delta_t` 更新、EMA、age/stale、reset
  原子清状态”的二维线速度生产模式。ActionBall 应复用该**因果更新规则**并扩成三维
  root-COM heading velocity；重复的 held pose 绝不能被解释为零速度。

首个 194-D N1 运行保持其 exact 历史；fresh fixed-194 v2 用 teacher-start 时钟替换常量
one-hot，但仍不在没有
真实 dropout 分布时追加常量 freshness 列。最终
部署意图 actor 应在同一次自然断 warm-start 的版本迁移中增加至少
`base_localization_age_s(1) + base_localization_valid(1)`：短 dropout 时 policy 能区分
“新测量”和“hold-last”，超过 supervisor stale 门则直接安全停机。若只增加这两列，N=1
宽度将由 fresh 194 变为 196，必须使用新 contract 名并重钉 Python/C++/MuJoCo producer；
不能在 194-D 或 195-D 名称下静默追加。是否再加短 history 仍由实测延迟/别名 canary 决定。

旧青瞳（ChingMu）场地 profile 只可作数量级先验，不能直接冒充 OptiTrack 标定：旧位置模型是
`1.9 mm` white、`5.2 mm` AR(1) marginal、`rho=0.717`/50-Hz policy tick；旧场地记录还把
`40 ms` 作为保守整条 target 传输上界，而另一份实测说明传输 `<10 ms`、端到端 `<=20 ms`。
复核还确认这些数主要来自旧链的**球轨迹**，不是新 OptiTrack A3 base rigid-body；Jiayi V15
中的 `[3,3,5] mm`、`20 ms`、`1% dropout`、EMA `alpha=0.25` 也明确只是 guessed prior。
这些数字来自不同设备、对象和处理链。当前 N1 首波不把它们分别撒到 6-D rotation、
projected gravity 和 task residual 上；部署意图重训前必须重新记录 OptiTrack capture/source/
receive/consume timestamp、Motive smoothing、遮挡、gyro bias 和 marker→COM 线速度残差，并在
同一个因果 SE(3)+twist packet 上注入 episode extrinsic bias、tangent-space orientation noise、
colored position noise 与 hold-last latency。

### 1.4 近期球物理与 73 动作证据如何消费

`origin/main@ddfaaa02` 已加入 2026-07-30 OptiTrack 球物理：较可信的拟合量包括
`k_d=0.1253`、`k_m=0.00404`、球拍有效恢复系数 `0.646`，桌面恢复系数采用裁定值
`0.9215`；paddle speed decay、table/paddle tangential retention 仍欠充分辨识，只能作为
canary/prior。科学源 `ddfaaa02` 已以 `bed6661f` 进入当前分支，但正在运行的 exact
`f2c54fc3` N1 source 与已物化 bundle 仍绑定旧 physics profile，**不得声称该 policy 已经
使用新物理**。

首个 contact-only N1 的判据只是 task 击球位置/时间窗是否可达，新参数不阻塞 1000-update
diagnostic；任何正式 landing/return 结论、formal N=5 或部署意图训练前，必须显式合入新物理、
重新物化 physics/solver/question bundle 并重钉 SHA。参数改变不是 Reward A/B，但新旧 policy
表现也不能混成同一 lineage。

ChingMu-73 已不是“只有动作、来球要反推”的原料：exact N=73 manifest 已逐动作保存真实
`v_in/v_out`、击球点、station、yaw、contact phase 与 motion bytes。73 条 station 覆盖约
`0.260 m x 0.479 m`，yaw 约 `-4.76 deg .. 37.83 deg`。这进一步支持当前拆分：动作/task
保持 heading-relative 以复用，table-relative base pose 9 值保留绝对站位与三轴姿态；
每动作用 manifest 的真实来球作 ball center，而不是强行共享一个 spawn。

### 1.5 joint 和 task 信息是否够

本节只说明“为什么够”，不再维护第二份 term order；exact offset/dim 唯一以
[§1.2 的 17 项表](#12-观测合同相对-task--绝对桌体上下文)为准。fresh fixed-194 v2 合同中已经有：

- 31 个 joint position（相对 default）；
- 31 个 joint velocity；
- 31 个上一 policy action；
- 三轴 base angular velocity、三轴 base linear velocity、projected gravity；
- motion anchor orientation、62-D teacher command；
- 相对 base goal、相对 racket target、目标拍速、time-to-strike、teacher-start 倒计时；
- demanded signed face、完整 table-relative base pose。冻结 action identity 只在控制面，不在
  actor observation。

这里的 `HITTER` 是 177-D prefix 的来源；仓库没有一份独立、可替代上述布局的 “SMASH actor
contract”。SMASH 相关论文/实现可提供 reward、噪声或训练方法启发，但不能凭名字推断另一组列。
这也符合两篇论文的系统边界：
[HITTER](https://arxiv.org/abs/2508.21043)由上层 planner 生成击球位置、速度和时序，再让
whole-body controller 执行；[SMASH](https://arxiv.org/abs/2604.01158)把 onboard egocentric
perception 和 scalable whole-body skill 作为自己的系统组成。后者的感知输入不能在没有
frame、latency、deploy producer 和 checkpoint metadata 的情况下直接拼进 HITTER-derived prefix。
当前不再为首个 N=1 添加 raw ball state、角加速度、接触历史或 observation history：
fixed-action solver 已把每颗来球编译成当前相对 task；在真实 PPO 证明存在不可观测别名以前，
继续加列只会扩大接口和部署债。

单帧不是永远冻结的结论。若 OptiTrack/IMU 实测证明延迟或接触冲量导致同一当前帧对应不同最优
动作，先做关键通道 `0 vs 4/8` 帧的小预算 canary；ring buffer 必须进入 partial-reset 与
exact-resume state。它不是“纯拼列、无状态”的免费修改，GRU/蒸馏排在 feed-forward stack
之后，也不阻塞首个 N1 policy。

### 1.6 bang-bang：当前只用 action-change 惩罚，不排产硬加速度 governor

基础 v2 合同同时带 raw/processed action 一阶变化惩罚与 raw-action 二阶差分惩罚；
今晚 vendor leaf 显式将二阶项钉为 `0.0`，只保留 `action_rate_clamped=-0.2`。另保留实际 joint
position limit 账、finite qdes projection、投影前超出量 penalty 和 actual soft/hard 分账。
Franco 本轮裁定与击球经验一致：**bang-bang 先靠 action-change penalty 调到不抖，
不为此加执行命令 acceleration/jerk 硬限制**。首个 N=1 long 保持一阶惩罚和安全链，
不加 EMA、额外 temporal regularization、Ruckig 或新 governor，原因是：

1. 快速击球依赖短 contact window 内的高拍速；未按 A3 电机和 teacher 导数标定的硬限速会直接
   延迟 `t_hit` 或削掉拍速；
2. 更强软 penalty 可能压过模仿/击球收入，却不能给硬安全保证；
3. stateful governor 会改变 action→executed command 合同；若 actor 不看 executed state，
   同一个 actor action 会对应不同后继状态，制造新的部分可观测性；
4. 当前最大问题仍是先取得 healthy strike denominator，而不是凭视频印象调平滑权重。

1000 update 检查只用于定价现有 action-change 项：逐关节 executed-command 换向率、
归一化一阶差分、二阶差分零权重遥测、经验证的隐式-PD torque proxy（当前尚无可信 saturation 量测）、
qdes projection/saturation、实际关节 hard/soft 驻留、击球窗拍速和 table/fall。若抖动，只对当前
action-rate 剂量做最小 canary，同时比较 teacher phase delay、strike/return、racket speed
和 imitation，不以总 Reward 单独晋级。当前两个 dynamic-ready recipe 的 raw actor bias 约落在
`[-5.87, 16.30]`，所以不能套用 `[-1,1]` clip；那会让部分动作专属 ready 本身不可达。

acceleration/jerk governor 不再作为“1000 update 后自动轮到”的候选；只有真机测量证明
q/effort/velocity 现有硬边界仍不足以保护 A3，才重开该工单，并要求 Isaac/MuJoCo/C++ 同合同。

## 2. 已完成证据（附录 A；不参与调度）

本节只保存不可替代的运行数字、SHA、失败根因和完成边界。新工作不得从这里领取；状态变化先更新
[当前执行看板](#0-当前执行看板本文唯一活跃-todo)，完成后再把证据移入本节。

| 项 | 证据 | 当前边界 |
| --- | --- | --- |
| 四轴 Hctrl v7a 机械 stress | exact source=`62e0878ac52748373838850faf02c3be1c9f16bc`，Pod 六文件 torch 组合 `377 passed`。no-clobber receipt canonical/file/log=`79e14853b9a50b7b2ceb67ddf9b95f4bba9748edef95163f1190bba533559a98` / `cb0fcfdc0db2a5ec2d16b1eb096ed1b128415c012232a2a2e6ff961c5b55dd88` / `087028bf5d7ad6bfd159039278d9ec1b769a64f0e3d5c7e866f0cf614ea9966c`，status=`PASS`且 canonical 独立复算一致。8/8 OFF 组 tick1 进 `[Hctrl,Hmech)` 并四 tick 内触/穿 Hmech；8/8 ON 组的 32 ticks strict Hmech；64/64 contact=`0 N`，8/8 full input tape exact，restore exact，31-joint order/selected indices/four live proofs/public-run readback exact | 关闭 waist-roll/pitch + 双 ankle-roll 的 2% 双位置 plant 机械门；其余 27 轴仍是 Hmech，soft q-des/Reward/actor/observation 不变。本证据不代签 table attribution、fresh `4096×5`、push 或 long |
| 07-31 新尽调吸收闭环 | 已把报告 §1–13 逐类映射到[§4.1 来源决策矩阵](#41-本轮外部来源证据决策矩阵)：智元 A3、BeyondMimic、unitree_rl_lab、mjlab、HITTER、SMASH、PACE、PBHC/KungfuBot、IsaacLab Reach、Dextreme/ADR、DeepMind 乒乓与接触/reward/curriculum 先例都有“证据性质→真正证明什么→`ADOPT/DEFER/REJECT/RETAIN`→落点/最迟边界”。§1–5 的 DR/Reward/push、§7 误差核、§8 PPO、§9 reset、§10 吞吐、§11 厂商设定、§12 先例与§13 课程算法均已显式吸收；§6 历史排序被§11/§13 与 fresh vendor baseline 覆盖 | 尽调分类已完成；R1–R9 以 N5 专项 TODO/后置决策落账，N1 明确 fixed-domain。已实现的 legacy 真值/探针见下两行；§0.2 只批准 code-owned maximum→minimum 的 `bh_loop_c` adaptive-sigma C profile 进入 `IN_PROGRESS`，已拒绝从 live minimum 起步的 no-op 伪 canary。其余 canary 只在§4 保留后置边界，不因“已吸收”冒充交付 |
| C1 三 lane smoke / probe | final source=`f173fd6d…`。loop-static、block-static、loop-adaptive 的 `1 env×2` 均自然完成，checkpoint 全 finite，194/318 normalizer、exact policy/Reward/contract、`[0,2]` delay 与 adaptive maximum/live lockstep 均通过。随后三条 `4096×5` 自然完成：loop-static/adaptive actual-hard=`0`，但 table 均约 `1.110%`；block table约 `0.767%`，update4 另有 `right_ankle_roll_joint` actual-hard=`3`、min Hmech gap=`-0.0038333 rad`。loop static LR 在 update1--3 离开 floor，adaptive update4 LR=`2.25e-5`，故小 σ×adaptive-KL 不构成已证实的永久冻结。五轮总模拟时长约 `2.4 s`，未达到 `[5,15] s` push 首事件窗，push count=`0` 不是负结论 | smoke 身份链 PASS；probe 行为/机械 Gate FAIL，三条 long 与 push-evidence 不放行。先诊断 table body/phase 与 right ankle plant hard edge，不放宽 table/actual-hard 门，不用 soft reward 或 acceleration governor 猜修 |
| N1 long 收据 gate source | exactly-once 自然完成 marker、`weights_only=True` checkpoint replay、actor/critic normalizer `194/318`、policy ABI/std、完整 `[0,2]` control-step delay、push 六轴 extrema/count、actual/qdes/fatal/nonfinite 零容忍、table/fall 有界率与 strike/swing reachability 均已进 producer+consumer；claim-v2 还把同一已验证 claim 在 runner 构造前一次解析并同时写入 checkpoint、completion marker 与 namespace claim，普通训练不读取残留 ambient env；独立末审 PASS，`65 passed, 1 deselected` | source 闭合完成；artifact A smoke 已证明身份/自然完成链可运行。剩余阻塞是 probe/push 共享 actual-hard safety，不是 gate 逻辑或收据格式；不允许用旧 FAIL 运行或手写收据 |
| artifact A dynamic recipe + clean smoke | artifact A=`077ecb3f`，dynamic recipe=`f76df2…` 成功。Pod1 GPU0 `1 env×2` 自然完成，两份 checkpoint 均 finite 且含 `obs_norm_state_dict`；namespace、checkpoint metadata 与 natural-completion marker 均报 claim=`64697e…`，hard/nonfinite 全零 | identity→recipe→checkpoint→completion 机械链 PASS；只解锁同身份 diagnostic probe，不单独授权 receipt/long |
| artifact A probe / push safety 负证据 | GPU0 `4096×5` probe 自然完成，iteration=`3.07/2.07/8.70/3.75/6.79 s`、中位 `3.75 s`，吞吐门 PASS；但 update1 起有 waist-roll actual-hard，update4 同时有 waist-roll + waist-pitch，累积 `joint_actual` terminal 非零；strike=`1` 仍不得产生 receipt。compact worst actual penetration=`1.1343e-4 rad`。GPU2 push-evidence 在 update1 同样记到 31 个 waist-roll actual-hard，已对 exact-owned 进程精确 `TERM`，日志保留且 GPU2 释放 | long 暂阻塞于共享 hard-limit safety root cause。吞吐不再是本轮 blocker；`max_inward_until_nonoutward_v1` 仍只是 emergency containment，不是 acceleration/jerk governor；vendor-only 5%→6% 只是待 host + Pod 零门验证的候选，不是 PASS |
| legacy `reward_pack` 与配置真值 | 7 条 legacy + 2 条 ActionBall 真实 Hydra compose 均通过；legacy 显式钉 `reward_pack=v1`，ActionBall/vendor 保持 v2。DeployParity 与 full-observation `1 env×0 update` trainer dry-run 均自然 `rc=0`。同批将 `hope_env_cfg.py` 的 PD startup 注释改为实际 Kp/Kd 分键口径，删除 Hitter/DeployParity 的 IdealPD 时代 rationale，并把 implicit-A3 名义 `arm_torque_saturation_weight=-0.5` 改为有效真值 `0.0` | 兼容性与文档真值已闭合；不改 vendor N1 Reward 经济，不再作 TODO |
| weight-independent jerk / implicit-PD proxy | 已实现默认 `None` 的 `action_acc_jerk_probe` 与 `implicit_pd_post_step_effort_proxy_probe`；权重为零仍可显式打开记指标，但默认不构造 RewardTerm、不改 total Reward/policy 或 vendor N1 热循环。定向 `48 passed`，邻接 `384 passed` 中只余 4 个父提交已知 explicit-backend fixture 失败 | 可观测性 source 交付已闭合；真 `joint_acc_l2=-2.5e-7` 仍是 healthy baseline 后的独立学习消融 |
| Wave-P / `PROGRESS` 口径收口 | [PROGRESS](../../PROGRESS.md) 与 [Wave-P 实验记录](EXP-P1-PUSH-ROBUSTNESS-20260721.md) 已统一为 `14` 臂真实发射、`4` 臂 never-launched；前者均未到预注册 16700 终档，后者取消 | `closed_incomplete / superseded / no dose winner`；只保留 directional evidence，不补训、不补卷，不再作 TODO |
| 旧 N1 三条 milestone1000 终档 | Pod1 GPU0 block-fast seed1、GPU1 loop seed0、GPU2 block-fast seed0 均自然到达 update1000/1001；`model_1000.pt` SHA 分别为 `fdd9c156…e58c` / `000b275c…0fec` / `ca3167b7…16de`，均 80 tensor/76 浮点或复数且 finite。三条共经历约 797–1043 次 strike opportunity，但 capture/return 全为 0；update1000 拍位误差约 `0.30–0.42 m`、速度误差 `1.10–1.34 m/s` | 旧 stable-ready/legacy 物理 diagnostic 正式收口为负证据；不购买 20000-update，不续成新智元 setting |
| 旧 N1 runtime 零成本诊断 | GPU2 r4 checkpoint 的 `obs_norm_state_dict` 与 `privileged_obs_norm_state_dict` 均存在，update1000 count=`98,402,304`。scalar std 从 update0 均值 `0.0204` 升到 update1000 `0.6579`，范围 `0.168–0.966`，无穿零/NaN；LR 早期曾触 `1e-5`，update500=`1.5e-5`、update1000=`1.7086e-4` | obs normalization 定谳为真活；`noise_std_type=log` 不因“穿零”立即翻转，adaptive-KL 小 sigma 交互保留收据与运行时守卫 |
| 智元 host 运行合同与诊断 launcher | 29-DoF vendor nominal/armature replay 定向 `23 passed, 9 skipped`；PD split/startup 专项 `20 passed`、相关回归 `59 passed, 12 skipped`；control-step delay/runner/stage/training-contract/exact-resume 定向 `359 passed`；runtime guard `20+58+3 passed`；strike-window 直方图/exact-face/stage 定向 `34+13 passed`；旧+新 launcher `69 passed`；stage evidence v4 consumer/fixture `51 passed`；vendor eval+canonical+formal launcher 合跑 `128 passed`；identity-smoke launcher focused `15 passed` | E1 source mechanics、tracked authority 旧包拒绝负例、stage consumer 与双评测口径已验；actual authority/code pin/dynamic recipe 已由 artifact A 的 Pod E3 身份链闭合，当前剩余的是§0 actual-hard safety 阻塞 |
| 智元 identity recipe/smoke（Pod1 E3） | exact source `5665963e`、GPU0 recipe claim=`0ab91705…deb0`、smoke claim=`e413578f…9462` 均 fresh namespace 自然退出。policy contract=`27bf405e…e416`；schema-3 training contract SHA=`98fa3239…1366f`。`model_0.pt` / `model_1.pt` SHA=`93c6ea96…7d86` / `cfa408e8…80cd`，全部 tensor finite；delay/ABI 各 1 条、std/LR 2 条，恰好 2 个 PPO iteration，无 Traceback | vendor task、真实 USD articulation、PPO/checkpoint 与 runtime guard 已构造闭合。该 policy SHA 仅是 shared-ready identity bootstrap，不是后续 dynamic-ready recipe；run 仍是 `diagnostic_unauthorized` |
| 智元 live plant authority 顺序修复 | 第一份 live contract 证明真实顺序是 USD interleaved articulation order；旧 authority producer 错用 controller/CSV 分组顺序而拒绝合法合同。提交 `5665963e` 改为 exact 31-joint live order、按 joint name 映射 vendor 向量，并保留 action ids=`0..30`、delay/push/plant 全值 fail-closed；host 49 passed、Pod exact 同 49 passed，重跑 contract SHA 逐字不变 | 是 authority validator 合同修复，不改训练物理；旧 logical-order 反例继续拒绝。receipt 必须由含该 producer 的 clean commit 重新物化 |
| 智元 actual-authority 与 code pin 闭环 | 提交 `f948a150` 跟踪 schema-3 runtime contract SHA=`98fa3239…1366f`、actual-authority receipt SHA=`f66a9e59…5461a`、required-identity SHA=`240f3757…01ff`及 candidate/hold/bundle；launcher 同时固定两个 authority SHA。host 全量相关测试通过，随后在 clean `f948a150d6f64b02a3790c451d5f00fac09b761a` 直接重读 receipt、source blob 与 dynamic-ready candidate，验证 action=`bh_loop_c`、motion=`0fa46ad6…0c85b` 和 contract SHA 三方一致 | authority/pin 阻塞已闭合；只授权下一个 recipe-only 阶段，不授权 diagnostic/formal 训练、export、judge、deploy 或 hardware |
| vendor dynamic-ready recipe r1 受控拒绝 | exact source `2430fbb2`、GPU0 claim=`e37f8169…e32`、dynamic binding=`15f4f1e9…2a43` 已通过 clean source、actual authority、candidate/hold/bundle、GPU empty 与 pre-scene schema-v2 验证。env manager 构造时发现 `MotionCommand._configure_action_ball_dynamic_ready()` 仍将 runtime binding 写死为 schema-v1，因 `schema_version/kind mismatch` 拒绝。未生成 policy recipe、未进入 PPO；namespace 永久 spent，自有 leader PID/PGID=`1328514` 精确 `TERM` 后 GPU0=`18 MiB` | 这是 producer→runtime schema 断链的历史负证据，不是训练失败。后续修复已保留 schema-v1 兼容并对 v2 plant/delay 做完整验证；artifact A 已取代该 spent namespace |
| vendor dynamic-ready recipe 物化 | exact source `e7787e25`、GPU0 claim=`75f28f24…490c`、dynamic binding=`c49cf23e…b979b`。recipe-only 自然完成，marker 报 policy contract=`e408b84567f3d7c04496a2542d22080fbc9a9aed7ff64c807215db4834a7c65d`；materialized recipe file SHA=`86dcb3fd56fb1cf7c05f5946d65e2b3550f68d6a48696bab672b84f614a095e2`，0 PPO update、0 checkpoint，result 中 launch/export/judge/hardware 均为 false。GPU0 自然回到 `18 MiB` | dynamic-ready policy recipe 门已闭合，只解锁 vendor diagnostic `1 env×2` smoke；不授权 probe/long/formal/export/deploy/hardware |
| vendor diagnostic smoke | exact source `e7787e25`、GPU0 spec SHA=`c635f3a3…72fe`、plan SHA=`21f005c0…b8d`、claim=`be783ab7…ad54`、policy=`e408b845…c65d`。`1 env×2` 自然完成，runtime ABI/delay/std-LR marker=`1/1/2`，delay lag=`1`，std 两轮均 finite/positive，LR 均为 floor `1e-5`。`model_0.pt` / `model_1.pt` SHA=`ee587b84…1d06` / `daeb51f6…bad2`，各 83 tensors 且 finite。无 Traceback/table/fall/qdes-hard/nonfinite；update1 有1次 waist-roll actual-hard，episode age=`25`。smoke run training-contract SHA=`3e559a9a…ab00`，GPU0 自然回 `18 MiB` | 解锁同身份 `4096×5` probe；48 policy steps=`0.96 s` 小于 `t_hit_min=1.133 s`，zero strike/零入窗距离不用于改 Reward 或终止门 |
| vendor diagnostic probe（full-DR 断链负证据） | exact source=`e7787e25`、spec SHA=`da7f2b50…f55a`、plan SHA=`d0be2181…89e5`、claim=`819e6125…9ccf`、result SHA=`13b09b19…3628`、log SHA=`2787a742…8521`、policy=`e408b845…c65d`。`4096 env×5` 自然完成，五份 checkpoint 各 83 tensors 且 finite，SHA=`1945cc1c…521e/4b621751…9646/cbc1a485…3656/79b53290…3a27/dd915905…36bb`；wall=`4.74/9.26/8.81/8.58/8.74 s`；std 五轮均 finite/positive，LR 前两轮到 `1e-5` floor 后恢复。strike-window entry=`0/0/85/0/15`，100 个拍距中 `97% >0.20 m`、`83% >0.30 m`，均值 `0.434 m`；actual-hard terminal=`875/3684/3143/3193/3191`，waist-roll 主导、waist-pitch 次之，qdes-hard 全零，最小 hard gap=`-0.000618 rad`。本轮无进程残留，GPU0 compute-app 列为空 | 同时实锤精核死区和 vendor launcher 遗漏 `stable_ready_plant`；该 policy 不授权 long。直接补粗核与 launcher 机械键，重新物化 recipe/smoke/probe，不用放宽 actual hard edge 或用 R0 吞吐优化掩盖错误 plant stage |
| 智元 dynamic-ready / nominal-hold / N1 bundle | 基于 `98fa…1366f` contract 物化 schema-v2 candidate SHA=`c831a4e6…794`，hold LP 最大利用率 `0.93156`。Pod1 nominal-hold receipt SHA=`11c025dc…740`：`0.8 s/40` policy steps、160 physics steps、delay lag=1、双脚接触率 1.0、minimum root z=`1.0684 m`、maximum tilt=`0.02385 rad`，无 terminal/table/fall/qdes-hard/actual-hard/nonfinite。新 bundle SHA=`9881c52c…ae03`，solver=`146c4d6a…c248a`、physics=`aa5c9085…f85b7` | exact vendor plant 的出生/保持与 bundle 重物化已闭合；后续 dynamic recipe/smoke 已在 artifact A 完成，当前 probe/push 仅因 actual-hard safety 失败 |
| formal table pose-OBB v4 consumer 链 | 正式 producer→stage、producer→canonical admission、signed prelaunch per-action 与 generic curriculum launcher 已统一为 `isaac_action_ball_table_pose_obb_smoke_v4`；旧 v3/filtered keys、false pose gate、action tamper 全拒绝。stage+canonical+producer `165 passed`，formal launcher `58 passed` | formal consumer 迁移已闭合；历史 bank 字段名 `isaac_table_filtered_smoke_receipt_sha256` 仅是命名债，其内容只接受 v4；path-free runtime USD identity 仍属后续证据债 |
| N1 diagnostic 加速 Gate-B 校准 | exact source `7f77ae5c` 的 same-seed `4096×5` wall=`2.864/2.753/11.292/6.341/10.249 s`，均值 `6.700 s/update`、约 `14.67k environment-steps/s`；五份 checkpoint finite，joint-safety/exact-behavior 与 pose-OBB 基线逐轮相等 | 已进入 6–8 s 校准带，但 10–11 s reset 尾延仍存在；不再为新 vendor baseline 继续打磨旧身份 |
| fixed-194 v2 policy recipe / r4 specs | Pod1 recipe-only 真实构造物化 policy contract `165645f5…bd9`，tracked recipe raw SHA `4b81c74b…7fb1`；旧/新 recipe 的 PPO 字段相同，变化来自 dynamic-ready 绝对 artifact/receipt path 与派生 binding SHA。fresh r4 smoke/probe/milestone1000 spec raw SHA 依次为 `6fc4e7ca…c369`、`6e7caeb1…b200`、`533b50d2…36ea`；smoke canonical plan PASS，claim `257c6ccc…d80c` | recipe/spec/claim、r4 smoke/probe/milestone1000 均已闭合；终档负结论见本表首行 |
| fixed-194 v2 r4 smoke | Pod1 GPU2 exact source `8729104e`、claim `257c6ccc…d80c`、namespace `n1hr_smoke_fastball110_8729104e_block_gpu2_seed0_r4` 自然完成 iteration 0/1（`2.75/2.80 s`）。`model_0.pt` / `model_1.pt` 各有 80 个 tensor、其中 76 个浮点/复数 tensor，逐项全 finite；table/fall/qdes-hard/actual-hard/nonfinite/terminal reset 均为 0，双脚接触率 1.0 | 真实 ObservationManager、fixed-194 v2 actor、fresh dynamic-ready bootstrap 与 PPO/checkpoint 路径已闭合；48 steps 尚未进入击球窗，不能判断 strike/学习 |
| fixed-194 v2 r4 probe | Pod1 GPU2 claim `fca61705…b813`、PID/PGID `1133162` 自然完成 4096 env×5 updates，iteration wall=`10.09/10.48/26.63/17.16/24.85 s`，mean episode=`—/48.00/71.66/52.99/59.91`。五份 checkpoint 各 80 tensor/76 浮点或复数且全 finite，`model_4.pt` SHA=`0f925821…f2e7`；strike=`0/0/1985/0/643`，qdes-hard/fall 全零，table=`0/0/0/16/25`，actual-hard=`0/267/3103/861/2076` | 证明并行构造、击球窗和 denominator 可达；前五轮腰 hard 波动与历史早期谱系一致，只作为 milestone1000 的 update100/300/1000 趋势基线，不单独判死 |
| fixed-194 v2 r4 milestone1000 启动 | Pod1 GPU2 canonical claim=`2710fd6f…d4f4`，exact PID/PGID=`1134253`，namespace=`n1hr_milestone1000_fastball110_8729104e_block_gpu2_seed0_r4`；已进入真实 PPO，首两轮 wall 约 `9.42/12.05 s`。`model_0.pt` SHA=`1296e929…6bcf`，80 tensor/76 浮点或复数且全 finite | 已自然完成 update1000/1001；`model_1000.pt` SHA=`ca3167b7…16de`，全 finite，capture/return=0/0，不续成新 setting |
| r4 update-wall 取证与 compact candidate | r4 每轮固定 98,304 env-step；update 8 后 collection 均值 `23.79 s`、learning `0.299 s`，约 `4.13k steps/s`。全窗口 collection/reset 相关系数约 `0.84`，粗拟合 `12–14 s + 5.6–6.7 ms×reset`；稳态约 1670 reset/update，但同 reset 负载仍在 `21.7–29.9 s` 波动，15 秒 NVML 采样 SM 均值约 `10.8%`。源码审计确认剩余主项是 reset per-env Python/`.item()`、逐 step safety clone/identity、每 update 正式收据与残余同步。工作树 candidate 已批量化 reset identity D2H，并为 diagnostic 改用 update-scale device aggregate | 这些数字证明加速未收口且瓶颈不在 PPO/GPU 算力；candidate 尚未经过 Pod，不能写成已完成。formal checkpoint 粒度审计仍在 N5-RECEIPT |
| 第二批 update-wall candidate | exact `26c648d4` 把 diagnostic Motion timing 安装改为整批 handoff，并把 fixed-18-draw refill overlap 从 staged `O(K²)` 扫描改为 highwater + start-set `O(K)`；Pod focused suite `187 passed in 13.70 s`。首次 smoke 暴露 true-reset 的 previous swing sentinel 合法为 `-1`，修复后 fresh `1 env×2` 为 `2.67/2.45 s` 且 checkpoint finite。fresh `4096×5` wall=`9.00/10.11/25.63/16.81/23.89 s`；joint-safety、actual-joint 与 exact-behavior 三组 update JSON 均与第一批逐轮完全一致，五份 checkpoint 全 finite | wall 均值 `17.09 s` 只比第一批 `17.14 s` 改善约 `0.3%`，因此正确性 PASS、性能 FAIL。它反证 Motion scalar reset 读取和 refill `O(K²)` 是主要墙钟；下一唯一动作改为合并每步/击球步 host barriers，之后才轮到 table/joint kernel storm。当前 milestone1000 不热补丁 |
| P0a strict 1.1 倍热路径验收 | diagnostic 复用 Racket 已完成的 host identity/timing selection，普通零 pending 路径跳过被覆盖的全局 reduction；formal opaque resolver 未改。Pod focused motion/birth/runtime suite `75 passed`，event timing suite `6 passed, 1 deselected`。同一 strict 1.1 倍 bundle、同 seed 的旧 exact wall=`9.00/10.11/25.63/16.81/23.89 s`，新 exact `6557390f` wall=`2.90/3.65/18.38/9.70/16.40 s`，均值 `17.088→10.206 s/update`，改善 `40.3%`；五轮 joint-safety/actual-joint/exact-behavior JSON 逐轮相等，五份新 checkpoint 均 finite | 正确性与吞吐 PASS，但仍未稳定达到 `≤6.5 s`。一次 source `c38b25d0` 的 probe 名称虽含 fastball，实际绑定 1.0 倍 bundle，只作 spent 构造证据，永久排除于 strict 1.1 倍比较 |
| 第三批 update-wall candidate（Pod trainer 已验，性能 FAIL） | commits `d2ec91e9 / 5f85cc58 / dbb7ce04 / 60a8e219` 分别整批化 diagnostic birth callback、裁掉 diagnostic safety/receipt 冗余、收敛 broker/pool proof，并把 ordinary/strike metric host barriers 约从 `5/15` 降到 `1/2`。随后 `096afb7b / 4d631fb3` 分别移除 diagnostic 外围和内层 rollback 快照；formal 路径不变。Pod focused 回归通过，三组 update JSON 逐轮等价，所有 checkpoint finite | same-seed collection：基线 `10.0916 s`；第三批 `10.3804 s`；外围裁剪 `10.6618 s`；内层裁剪两个 replicate `10.7364/10.2878 s`。没有 wall 改善声明。reset 仪式仍约 `4.9--8.0 ms/env`，下一步是 opt-in 分段 profiler 与 compact batched reset；旧 active long 不热补，旧 spec/claim 不复用 |
| claim-pinned update profiler（Pod trainer 已验） | source `5e1443c4`，policy contract=`0acbbf02…6a57`，smoke/probe claim=`0b6cfc0c…e80b` / `83898200…8519`。Pod focused `136 passed`；smoke 2 行、probe 5 行 profile JSON，无 timing-scope mismatch。same-seed probe collection=`2.687/3.646/17.717/9.456/18.148 s`；solver `33.432 s` / 总 collection `51.654 s`，install 仅 `0.202 s`。三组 update JSON 与 `4d631fb3` 基线逐字相等，五份 checkpoint finite | profiler 已完成归因，不是性能 candidate。下一实现只针对 diagnostic producer→receipt 重复验证；formal/default 与 checkpoint receipt schema 不动 |
| compact receipt + batch sampler + update-boundary telemetry | source `ec6c1a6a`、policy contract=`a6bb7244…fd9c`，same-seed `4096×5` wall=`3.05/5.16/11.59/7.62/10.99 s`，均值 `7.682 s/update`、约 `12.80k environment-steps/s`；相对紧邻 candidate 再快 `10.7%`，相对旧 `12.341 s/update` 累计快 `37.8%`。五份 checkpoint 各 80 tensor/76 浮点且 finite，joint-safety、actual-joint、exact-behavior 三组 JSON 与前一 exact baseline 五轮逐字相等 | 数学等价 diagnostic 优化 PASS；formal receipt/state 未改。profile 进一步确认 pool/solver 包装仍占约一半 collection，不能再把剩余墙钟统称为 PhysX |
| pose-OBB v4 桌碰与 4096×5 | source `ff30f244` 的 1-env 真实 smoke 覆盖 top/keepout/net/two-post、wrist/elbow/ankle 和 substep 1/2/3/4；每次只有选中子步命中，legal stance clear，nonfinite fail-safe，raw reason/terminal 各计一次且 reset 零泄漏。训练 validator 同步迁移后的 source `3a857f63` focused `124 passed, 12 skipped`；recipe raw SHA=`04021d06…19b6`、policy contract=`4cb2c532…1f96`。`1 env×2` 两份和 `4096×5` 五份 checkpoint 均 80/76 且 finite；probe wall=`2.76/4.47/12.35/6.35/10.36 s`，均值 `7.258 s/update`、约 `13.54k environment-steps/s` | 相对 ec6 profiler 的 `7.582 s/update` 再快 `4.3%`。同 seed table reason `3984→138`（`-96.5%`），总 reset `6967→6346`；actual-hard 取代假桌碰成为早期主终止。formal v4 consumer 现已闭合，但 formal promotion 仍须其他 admission/evaluator/receipt 门 |
| host-only solver result 与 single-H2D | exact source `7f77ae5c`，profile pins=`da4bfd74…172e`，fast-ball bundle=`c5d8a01f…74b9d`。独立 review 先发现 field-major flatten 可吞掉异常字段宽度/尾随 NaN，已在 H2D 前逐行强制 `3/3/3/2/4` 并加反例；Pod focused `104 passed`。smoke `1 env×2` checkpoint 全 finite；profile `4096×5` wall=`2.864/2.753/11.292/6.341/10.249 s`，solver 累计 `16.367 s` / collection `32.924 s` | 五轮 reset、joint-safety、exact-behavior 与 pose-OBB 基线逐轮相等；table=`0/0/17/93/28`、actual-hard=`0/267/3095/801/2062`、fall=0。数学等价裁剪 PASS；后续不再继续磨 result clone，转 fixed-tape LM 迭代数数值验收 |
| P0a 反手拉 probe 与三卡 long | exact source `bd340479` 的反手拉原中心来球先完成 recipe、`1 env×2` 和 `4096×5`；probe wall=`2.90/2.87/12.51/5.97/9.83 s`，均值 `6.816 s/update`，五份 checkpoint finite。strike opportunity=`0/0/921/0/1`，table=`0/0/15/77/22`，fall=`0/0/0/0/6`，actual-hard=`0/264/3111/803/2059`。随后 GPU1 fresh milestone1000 已进入真实 PPO；GPU0 同时运行 `6557390f` strict 1.1 倍反手挡 seed1，GPU2 继续旧 `8729104e` 反手挡 seed0 | 三卡 long 均已自然完成 milestone1000，checkpoint finite 但 capture/return 全 0；作为旧物理负证据收口 |
| lean Motion timing validator focused | exact source `2c3a39fe` 在 Pod system Python/Torch 环境完成 handoff focused `22 passed in 2.76 s`；motion-birth/runtime-wiring/event 三套合并为 `66 passed, 1 failed`，唯一失败的 `motion_post_swing_replay` 旧 fixture 在父 source `bd340479` 单测同样失败，故不是本 candidate 回归。独立只读复审确认 formal 路径未改，协调篡改在任一 device write 前 fail-closed，最终结论 `MERGE` | focused 正确性 PASS；尚未在自然空闲 GPU 跑 `1 env×2` 与 same-seed `4096×5`，因此吞吐 replacement 仍是 pending，不热补当前三条 long |
| fixed-194 v2 r3 构造失败 | Pod1 GPU2 exact source `8729104e` 已真实构造并验证 `action_ball_table_pose_twist_heading_task_teacher_start_v2 (194D)` 与 dynamic-ready bootstrap；随后在 PPO runner 创建前 fail-closed：spec 沿用 policy recipe `b7209710…077f`，post-compose 实际为 `165645f5…bd9`。spent namespace=`n1hr_smoke_fastball110_8729104e_block_gpu2_seed0_r3`，无 PPO update/checkpoint；wrapper 最终由 watchdog exit `125`，GPU/boot lock 自然释放 | r3 永久只作失败证据且不复用；正确 recipe 的 r4 已取代它并通过 smoke |
| fixed-194 v2 fresh r3 specs / claim | smoke / probe / milestone1000 三份 canonical spec 均绑定 exact source `8729104e6c9a…46c4`、fast-ball bundle `3c1076e3…c32b`、Pod1 GPU2 UUID、seed0 与 fixed `current_low` Reward；raw JSON SHA 依次为 `e1b63f00…5b8d`、`3b200542…dd34`、`b0396fbe…d442`，namespace 全部 fresh `_r3`；smoke canonical plan PASS，claim `7f9d12ca…4002` | 历史 r3 被 policy-recipe 硬门否决；已由正确 SHA 的 r4 三段 spec 取代，无后续动作 |
| fixed-194 v2 profile / question repin | Pod1 exact source `17c7258a` 物化 profile pins `08c8f9c7…c6b4`、base bundle `ed9fa0f7…afef` 与 1.1 倍 fast-ball bundle `3c1076e3…c32b`；solver profile 为 `52777b36…9754`，physics profile 仍为旧诊断值 `aa5c9085…f85b7`。derivative 的 4096-proposal tape 保持 `2763/4096=67.46%` admitted 与逐原因拒绝分账；工件 commit A=`8729104e`，r4 spec/claim 与 smoke 已闭合 | 这是 diagnostic comparison，不是 formal 95% admission 或新 OptiTrack physics 证据；当前只进入同身份 r4 probe |
| fresh fixed-194 v2 source + focused suite | commits `291bc20e` / `0227cfe9` 已让当前 ActionBall trainer 只实例化固定 194-D v2，删除 `policy.action_one_hot`，N>1 fail-closed，并对 exact 17-term layout、旧同宽重标、teacher-start lazy bind 加回归；Pod1 exact `0227cfe9` focused suite 为 **391 passed, 12 skipped in 61.35 s** | dependency/contract 测试、r4 真实 ObservationManager、2 个 PPO update、4096×5 和 milestone1000 均已闭合；旧 lineage 负结论不转移到新 vendor baseline |
| dynamic-ready 出生与 hold | loop/block 在 Pod 各闭环保持 `0.8 s / 40` policy steps，双脚接触率 `1.0`，table/fall/hard/nonfinite 零；可见 raw-reset 图证明原生 reset 直立 | 证明出生与 nominal hold，不证明 teacher 全轨、strike 或 long |
| dynamic-ready trainer 接线 | candidate/hold receipt 双 pin；physical state、初始 qdes、last action、actor bias 和 motion frame 0 原子一致；Pod focused `63 passed` | 当前只授权 exact N=1 diagnostic |
| 170-update overflow | diagnostic joint-safety summary 已改为每 update 按事务排空 | 旧 overflow checkpoint 不续；fresh run 重新开始 |
| qdes 安全语义 | finite request 投影到合法包络，penalty 读取投影前超出量；nonfinite/actual hard 仍终止 | Reward 剂量仍需 healthy baseline 后 canary |
| reset 诊断热路径 | diagnostic per-reset 逐 env 完整转录已移出；不可变 receipt SHA、strike timing、metrics D2H 等已做确定性优化 | formal 路径仍保留完整 per-reset 仪式；每步 ledger/broker 税仍开放 |
| 194-D actor source | `203b2d92` 实现 table pose + base twist；Pod dependency-light 合同回归 `290 passed, 9 skipped, 1 deselected` | C++/ONNX/MuJoCo producer 未实现，旧 182/191-D checkpoint 不可复用；policy recipe 与 observation 分层，现有 dynamic-ready recipe 可复用 |
| explicit table-contact v3（历史，已被 v4 取代） | v3 曾尝试 5 个 table-source × 32 个显式 A3 body filter，并完成脚本层正负控与 on/off 定价；后续 Pod 原生日志证明 pinned GPU contact-filter backend 不支持该矩阵语义，不能把零/低 counter 当接触真值。中间 body-sphere + racket OBB 去掉了失效 sensor，但同题五轮产生 `3984` 个过宽 table reason | v3 与 sphere 中间态都不再是 active backend；现役 pose-OBB v4 的真实 smoke 与吞吐证据见上。历史 `0.22 s/update` 仅保留为当时 workload 定价，不再证明 v3 正确 |
| 旧 194-D loop/block 构造 probe | exact `c9682591` 各 `4096×5` 均持续产 checkpoint 且 finite；mean episode 已到约 29–32 步，birth age≤1 hard 为零 | 第一次 PPO 前 update0 已有 `860/864` 个 env 撞 `waist_roll` raw hard；两动作 hard-env Jaccard `0.982`，不能据此发 1000 |
| shared waist 根因定位 | qdes forbidden 始终 0；teacher waist/hard 最小余量 `0.272–0.303 rad`；hard 首发约 step18/0.365s，且两动作 substep hard Jaccard `0.992` | 指向共享 plant DR，不指向 solver/Reward/teacher 贴限；下一轮使用 stable-ready plant |
| frame-consistent stable-ready smoke | exact `f2c54fc3` 的 loop/block `1 env×2` 均验证 194-D actor，四份 checkpoint finite，qdes/actual-hard/table/fall/nonfinite 全零；log SHA 为 `db75ef49…49b` / `0a26cbee…69f` | 证明真实构造和初始 plant，不判断学习 |
| frame-consistent stable-ready probe | loop/block `4096×5` 各有五份 finite checkpoint；update 0 全安全，mean episode 随后约 `48–72` steps 并跨各自 `t_hit`；loop 在 update2/3 有 `783/84` 个 strike opportunity，block 在 update2/4 有 `1838/430` 个 | 第一次 PPO 后共同出现 waist-roll/pitch actual-hard，qdes forbidden 始终 0；五轮只证明窗口可达，不判恢复上限 |
| 历史恢复反例 | 旧 exact `4ff48b21` 在 update 1–5 同样有大量 reset；loop/block 的 actual-hard terminal 到 update100 已降到 `14/11`，update169 均为 `3` | 支持按预注册运行到 100/300/1000 再判，不支持把旧 policy 当新合同结果 |
| 三条 milestone1000 | Pod1 GPU1 的反手挡 seed0 已越过 update 300 且 `model_300.pt` 的 80 个 tensor 全 finite；GPU0 fresh 反手拉 seed0（claim `3c523fde…0196`）与 GPU2 fresh 反手挡 seed1（claim `7ac32418…e3f`）均已进入真实 PPO update。三条 exact source 都是 `f2c54fc3`、`4096×1001`、194-D、stable-ready plant | GPU1 已跨 `t_hit` 但 virtual capture/return 仍为 `0/0`，table/fall 与劣质动作局部解仍开放；GPU0/GPU2 提供动作差异与 seed 复现，不把早期波动写成晋级。旧 GPU0/GPU2 `4ff48b21` overflow 进程已按 exact PGID 响应 TERM 正常退出，日志/checkpoint 均保留 |
| 2026-07-30 19:39 CST live 快照 | loop seed0 / block seed0 / block seed1 分别到 update `219 / 574 / 186`，mean episode `104.88 / 481.52 / 105.90`，三条均持续写真实 PPO update、无新 Traceback；当前 update 的 strike opportunity 为 `945 / 965 / 962` | 三条 virtual capture/return 仍全为 `0/0`。block seed0 当前 table/fall/actual-hard=`2/3/6`，但 965 个 contact proposal 全被 face gate 拒绝；loop seed0 post-strike fall=`887/946`；block seed1 table/actual-hard=`590/325`。说明窗口与 denominator 已通，但动作质量、signed-face/contact 对齐和 seed 稳定性仍未闭合，不得写成可部署 policy |
| 2026-07-30 19:50 CST block seed0 update 600 里程碑 | exact PID/PGID=`1055080`、cwd=`n1dr_10069d3c/.../whole_body_tracking`、GPU UUID=`GPU-a8f7…e6cb6`、source/claim/namespace 均与发射绑定一致；update608 持续训练，`model_600.pt` 为 7,197,343 bytes、SHA=`11bee491…8470f`，80 个 tensor 全 finite。mean episode=`440.77`、iteration=`20.94 s`、strike opportunity=`951`，table/fall/actual-hard/qdes-forbidden=`4/5/14/0` | 仍是零 capture/return；`937/951` 被 face gate 拒绝，exact strike position/velocity/normal error=`0.2426 m / 1.3928 m/s / 86.31°`，实际/目标拍速=`0.2832/1.2793 m/s`。安全与 episode 长度已恢复，当前主要问题是未学到 teacher 的击球位置、拍速和拍面，不是出生或 qdes reset storm |
| 显式 teacher-start source contract | 历史 `194+N`/N1 195-D source 已通过 exact `020dc8d9` Pod focused suite `390 passed, 9 skipped`，但未进入 PPO；当前 fresh N1 改为固定 194-D v2，用真实 teacher-start scalar 替换常量 one-hot | 这是接口正确性直接修，不做学习 A/B。旧 194-D/195-D run 保持 exact 历史；v2 仍须真实 ObservationManager `1×2` 构造与 finite checkpoint |
| 反手挡 1.1 倍中心来球公式带 | 固定 `bh_block` 动作身份、原落点中心 `2.555 m`、初始速度双侧宽度 `0.15 m/s` 和现有 solver/physics/prototype，在 Pod1 GPU2 对 4096 个确定性 proposal 只把中心来球从 `4.2376948` 提到 `4.6614643 m/s`。同一逐球 solver 得到 target site speed 均值 `1.14907 m/s`、teacher-rate 均值/中位 `0.72055/0.71595`；`2763/4096=67.46%` admitted，拒绝严格分账为 `resid_gt_tol=1327`、`teacher_rate_below_min=6`，proposal tape SHA 为 `0335220d…2581` | 这证明“更快来球但仍落同一台面中心”会让挡球 task 卸力并降低主动老师拍速，不需要另写反向 solver。它只是 GPU2 的机制诊断：solver rejection 会把实际训练题条件化，且 admission 未达 formal `95%` 门，所以不能称严格单变量 A/B，也不可作为 curriculum 晋级或 N5 证据；训练 sampler 必须保留全部 proposal 分母与拒绝原因 |

当前 source 证据随本次集成提交进入 `main` 后才生效；旧 `f2c54fc3` 三条运行始终保持其原
commit、194-D observation 和旧 physics 身份，不因文档或 main 前移而重标。

上表的 1.1 倍比较也澄清了 ball→task 变量边界。每个环境先冻结动作，再采来球、到球时间、
击球点、base 与 landing aim；solver 随后保持该动作的拍速方向，逐球求拍速大小、有符号拍面和
精确击球位置。**landing aim 是 solver 输入，不是 solver 自己选择的输出**。所以来球变快后，
若仍要求落在原台面中心，公式会主动卸力；target 本身仍由同一公式逐球计算，训练中没有
selector 或换动作。若未来希望自然借力挡得更深，才另把 landing aim 作为显式课程轴，而不在
本比较里混入。

第一次 plan 正确拒绝了旧 r9 bundle：最新 source 的 `hope_commands.py` blob SHA 已从
`0e650b…` 变为 `e24190…`，而 derivative 仍引用旧 profile pins。随后在 Pod 以 exact source
blob map 重物化 profile pins `9ccb9854…5788`、current-source base bundle
`0daa5bce…ace53` 和 1.1 倍 bundle `f2be2331…1a491`；physics SHA 保持
`aa5c9085…f85b7`，solver profile 因实现 source pin 更新为 `bf255a78…f26e`。
这次 repin 不改变来球、落点、动作或 Reward 数值。三份 canonical spec 现绑定 exact source
`77f01deb`、fast-ball bundle `f2be2331…1a491`、seed `0` 和 Pod1 GPU2；仍按
1 env×2 update → 4096 env×5 update → 4096 env×1001 update 串行。每一阶段自然完成、
checkpoint finite 且无 fatal 后才发下一阶段；三段都从新随机初始化开始，不把前段 checkpoint
当作续训，也不覆盖旧 GPU2 seed1 目录。

source `77f01deb` 的 canonical plan 已通过并产生 claim `13dc15a2…8e86f`。第一次 smoke
随后在 namespace `n1hr_smoke_fastball110_77f01deb_block_gpu2_seed0_r1` 创建 simulator 前
fail-loud：stable-ready N=1 guard 读取了不存在的 `racket_cfg.clip_names`，而训练翻译层真实安装的
字段一直是 `clip_names_per_clip`，因此把合法的 `("bh_block",)` 误读为空。该 guard 是
`1d4b8a11` 后新增，旧 `f2c54fc3` long 尚未经过它；这解释了旧波能跑而新 source 失败。修复只把
guard 改读 canonical 字段并在拒绝信息打印 mode/diagnostic/action tuple，不改训练数值。失败
namespace 永久保留，修复提交后用 fresh r2 spec/source/namespace 重发。
fresh r2 的 smoke/probe/milestone specs 现绑定 source `319ae8ff`，分别使用
`n1hr_smoke/probe/milestone1000_fastball110_319ae8ff_block_gpu2_seed0_r2` namespace；未发射的
旧 r1 probe/milestone spec 已删除，失败 smoke r1 spec 留作 exact 证据。

## 3. 分阶段最迟闭合项（附录 B；边界参考，不参与调度）

本节保存各阶段的完整门槛语义，避免看板为容纳细节而再次变成长流水账。所有当前动作均须在
[当前执行看板](#0-当前执行看板本文唯一活跃-todo)有且只有一行；本节未出现在看板中的条目
不得被解释为当前算力或实现队列。

### 3.1 首个 vendor N=1 long 前（旧 1001-update 路线已由§0.2三 lane裁定取代）

直接修/验证，完成即开跑，不等待后续工程完美化：

1. table-contact Pod smoke 已在 `eb2799b1` 完成：五个 role 均有真实正控，32 个 matrix
   shape/order 正确，`robot_hit_table` 会触发，reset/settle 后零泄漏，日志无 unsupported
   filtered target；receipt 见
   [`table_smoke_eb2799b1_gpu1_r26.receipt.json`](../../../configs/n1_contact_dynamic_ready_20260730/table_smoke_eb2799b1_gpu1_r26.receipt.json)；
2. 本轮已先把 actor 迁移到 frame-consistent **194-D**
   `action_ball_table_pose_twist_heading_task_n1`；随后采用的 fresh successor 是同宽但新语义的
   fixed-194 `action_ball_table_pose_twist_heading_task_teacher_start_v2`，用老师开始倒计时
   替换 N1 常量 one-hot。
   recipe 只绑定 PPO/decoder/ready，observation name/width/term order 由训练 hard contract
   单独绑定，所以无需为该一维迁移重物化 recipe；旧 182/191-D、旧混合-frame 194-D 与当前
   compatibility 194-D checkpoint 一律不因同宽续成 v2；
3. 先完成 shared 4×5-ms 双包络 stress；PASS 后在 final clean source 上零 PPO 物化 loop/block
   policy contract 与 adaptive effective-Reward 三个 SHA，不手填 pin；
4. 三 lane 分别跑 code-owned `smoke 1×2 → probe 4096×5 → push-evidence 4096×32`，确认
   teacher-start、finite checkpoint、q/qdes/last-action/ready、delay/push 与 shared-safety 全闭合；
5. 每 lane 由自己的 probe+push 证据生成 gate receipt；三组 receipt + long scientific skeleton
   使用三个 sibling artifact commits，避免 exact-path 两文件规则互相污染；
6. 放行后 A=`bh_loop_c` static 与 B=`bh_block` static 先占 Pod2 GPU0/2 发 fresh
   `4096×20001`；C=`bh_loop_c` adaptive 等 GPU1 自然释放或调度另一 Pod。全部 namespace
   fresh/no-clobber，保存 exact spec、recipe、run record 和复现命令；
7. stable-ready 首车保留 joint-default/robot-material 既有设置，暂关直接改变弱腰平衡的
   gain/mass/CoM DR；healthy baseline 后按§4.2逐轴恢复，不把暂关误写成拒绝智元设置。

这批 N1 的物理身份是当前 bundle 已钉住的旧 profile，只授权 contact/学习可行性诊断；不得在
结果中写成 2026-07-30 OptiTrack 新球物理或部署候选。

下列项**不阻塞**首个 diagnostic N=1 long：

- formal per-reset receipt checkpoint compaction；
- generic formal N5 launcher；
- full-body、Reward、reference guard、课程 failure target、entropy/sigma/RSI 消融；
- 8192 env；
- MuJoCo/C++/真机 fixed-194 v2 producer；
- acceleration/jerk governor 或 EMA（本轮不排产，只有真机反例才重开）。

stable-ready plant 不是最终 sim-to-real 配方。它只把“先让 policy 学到动作”与“证明 ready
覆盖完整 DR support”拆开；robot-material DR 从未关闭，1000 后只按 base-CoM → PD → mass
的具名阶段重新加回，
每阶段先出 robust-hold/teacher-to-`t_hit` 证据再做短 canary，禁止一次性恢复整包后靠 Reward
解释 hard-limit。

### 3.2 1000 update 检查

千轮不是任意调参点。按以下顺序判读：

1. checkpoint 是否持续 finite、exact identity 是否未漂移、有无 Traceback/summary overflow；
2. 动作是否仍像 teacher：逐动作看 motion/pose error、击球窗相位、关节抽动、table/fall/hard；
3. 是否产生足够 strike opportunity；五轮没有 strike 不能判学习失败，1000 轮仍无 strike 才进入
   根因分支；
4. 长期 strike/return/landing 与四组 realized Reward income；
5. N1 必须保持 fixed-domain：promotion/proposed/admitted 应始终为 `0` 或明确禁用；frontier
   推进只在 formal N5 的 R6 专用短验收检查，不把 N1 的零推进误判成课程故障；
6. bang-bang 量尺是否显示持续的 executed-command 高频换向或饱和。

如果 dynamic-ready/episode 已健康跨 `t_hit`，但 200–1000 update 仍无 strike，下一嫌疑是
official low-gain waist plant 下 teacher 动态不可跟踪；先比较部署一致的腰 gain 或 reference
retiming，不回头反复改出生，也不先放宽 hard limit。

只有 activation denominator 非零后，才允许小预算 canary：

- tracking 权重/Reward 剂量；
- reference guard 的 metrics-only、phase-gated 或连续约束方案；
- qdes projection penalty 剂量；
- curriculum safe-policy failure 目标（默认约 10%，20% 仅对照）；
- upper/full-body；
- entropy/sigma/RSI；
- 现有 action-change penalty 的低剂量微调（不自动排 acceleration/jerk governor 或 EMA）。

这些是经验选择，必须 canary；数学等价、合同修复和删除重复同步不做科学 A/B，但仍做 Pod
parity/吞吐验证。
本轮 adaptive sigma 的精确状态是 **`REOPEN-CANDIDATE / DEFER DEFAULT / C PROFILE IN_PROGRESS`**。
第一性原理检查先否决了“仅打开 adaptive 三旗”：live position/velocity/normal sigma 本来就是
minimum `0.075/0.5/0.262`，而 `sigma ← min(sigma, EMA(error))` 不可能把它变大，因此该配置会
永久 no-op，不是可识别的 canary。

真正的 `bh_loop_c` C profile 必须是 code-owned schedule：position/velocity/normal 从 maximum
`0.20/1.0/0.52` 起步，additive 与 `racket_strike_success` 的三个 sigma 全程同值锁步，
只能单调收紧到 `0.075/0.5/0.262`。两臂共同的 coarse position sigma 继续是 `0.30`；
Reward 权重、DR、reset 和 seed 不变。maximum 初态是 adaptive schedule 能发生的必要初始条件，
不是另一个实验变量。C 只允许 fresh 发射、禁止 resume；launcher/compose/Pod 尚未通过前
保持 `IN_PROGRESS`，不写成已实现或 canary PASS。

这里不是静态 iteration 阶梯，而是已经钉死的 SMASH/PBHC 同族误差反馈律。每 `500` 个 control
steps，在 decayed exact-strike ledger（每个 control step 衰减 `0.99`，effective count 至少 `50`）上分别
计算位置、速度、拍面角误差均值；`sigma_ema_scale=1.0`，候选值为
`clamp(mean_error, sigma_min, sigma_max)`，实际值再取
`min(current_sigma, candidate_sigma)`，因此坏窗口只能暂停收紧、不能重新放宽。position/velocity/
normal 的 additive term 与 latent `racket_strike_success` 三个参数在任何写入前先验证 finite、positive、
exact lockstep，然后原子更新；缺 term、半更新、非 finite 全部 fail-loud。receipt/runtime 必须记录
profile=`monotonic_fresh_canary_v1`、update cadence/decay/min-count/scale、三组 current/min/max、
exact-strike denominator 和是否发生收紧；这正是重新采纳 adaptive sigma，而不是把已退役的静态
阶梯换个名字。

### 3.3 正式 N=5 前

以下可以晚于首个 N=1 policy，但**不得晚于 formal N=5**：

1. **正式收据粒度改造。** 目前只有 diagnostic 快路径；formal 仍在每个 env-reset 做完整
   Python transcript/receipt 仪式。改为热路径紧凑 device event/assignment tape，
   checkpoint/hourly 批量物化完整 receipt；保留 proposal 分母、reject reason、action/domain/
   birth/sample/task/lifecycle/outcome，不接受仅 `seed+config`；
2. **重建与 exact resume。** fixed tape 下从紧凑日志重建旧式完整 receipt，逐字段 parity；
   checkpoint save/load/no-step roundtrip、篡改负例、segment head 和外部 resume pin 全过；
3. **分段 profiler 与 all-mode 热路径。** 分开 physics rollout、metrics/D2H、termination、
   safety archive、birth/retire、sampling/solver、state write、PPO；据实修每步 ledger clone/
   string 重建、残余 D2H、broker per-env Python。`~7 ms/env-reset` 只作为混合上界，未经 profiler
   不写成精确归因；
4. **固定宽连续动作意图。** fresh N1 已从 actor 删除常量 one-hot；formal N5/N73 不能直接复用
   无 future intent 的 v2，也不得恢复 N 维 `action_one_hot`。用动作内容生成固定宽 descriptor，
   首版至少包含归一化 `(q_ref_at_hit-q_ready)` 与
   `teacher_rate*qd_ref_at_hit`，同时提供给 actor/critic；对 shared-ready 动作做两两混叠检查，
   必要时再加一帧中间相位 preview。action UID/slot 继续只在 sampler/solver/curriculum 内冻结和
   审计。该接口修复不做“one-hot 是否好学”的科学 A/B，但必须做 tensor/order、跨 N 固定宽、
   动作间距离和 Pod 构造 parity；
5. **generic formal launcher 余项。** launcher 改认固定宽连续动作意图合同，不让 policy input
   随 N 改宽；仍须把 dynamic-ready、table-pose-twist producer identity、真实
   Reward/PPO/plant/solver SHA 和 exact resume 全部绑定进 formal receipt；
6. exact ordered N=5 manifest、五件 motion bytes/admission、动作专属 ball center/support、
   new-forehand `t_hit/t_cycle/site speed/table clearance/recovery` 和非空 trust set；
7. formal Reward causal receipt、frozen evaluator canary/heldout、table smoke、stage evidence 和
   checkpoint identity 全闭合；
8. 运行规模压力证据只按 N=5 的 N/E/R 合同给 N=5，不让 N=73 缺口反向阻塞。
9. 已把 `origin/main@ddfaaa02` 以 `bed6661f` 合入，并在第一次 byte pin 前修正两份 YAML
   的旧曲线示例注释；formal N5 仍必须显式选择该 OptiTrack profile，重新物化并重钉
   physics/solver/question bundle。未充分辨识的切向参数保留 canary 身份，不静默写成
   verified constant。

正式路径的 per-reset receipt 改造是**有意留到这里而不是漏忘**：它不改变 N=1 学习问题，
却决定 N=5/N=73 是否能在短 episode 下接近正常吞吐和可用 checkpoint 大小。该改造不做学习
A/B；验收是 numerical/state parity、旧收据重建、exact resume 和固定工作量 Pod 吞吐。

### 3.4 N=73 前

1. exact ordered 73 manifest 与每件 motion/compiler/safety/admission 证书逐项闭合；
2. 每动作自己的 ready、`t_hit/t_cycle/site speed`、ball center/support 和 solver profile，
   不把 N=5 的 shared-ready 假设扩散到 73；
3. 直接消费 exact manifest 已保存的逐动作真实 `v_in/v_out`、击球点、station 与 yaw 作为中心；
   不再反推或把 73 件归一到同一绝对 spawn；
4. full-body actor/reference 与固定宽 continuous action-intent 的任意 N 路径通过；不得重新用
   one-hot、UID 数值或每动作 learned slot embedding 绑定动作库大小；
5. N=73 对应的 sampler/pool/broker/curriculum/checkpoint 压力、compaction 和 exact resume；
6. 逐动作独立 frontier、强制覆盖/starvation、center/interior/frontier 混合和 frozen heldout；
7. 若启用 base move，先证明真实 spawn 与 no-move/move goal 语义、preparation window 和移动恢复；
8. N=73 仍在 Pod 独立 smoke/canary，不能由 N=5 run 续成或更换 action order。

### 3.5 部署前

1. C++/ONNX/MuJoCo 全部支持最终版本化 ordered terms、normalization、metadata 和 stale/dropout
   语义；若采用 `localization_age/valid`，必须使用新名称/宽度，不能伪装成 194-D 或 195-D；
2. OptiTrack marker cluster→`base_link`、venue→table 的完整 SE(3) 外参、时间同步、Motive
   smoothing/端到端 latency 和遮挡处理进入部署合同；时间链使用 v2 capture/Motive timestamp，
   不再只看 ROS 到达时间；
3. 三轴 gyro 与 OptiTrack pose 的 frame/extrinsic 对齐；base linear velocity estimator 完成
   mocap anchor、可选 accelerometer 融合、marker→COM 刚体速度修正；
4. 用真实噪声、延迟和 dropout 重新做 Isaac/MuJoCo observation parity；不能把 simulator truth
   直接当部署证据；
5. 新 OptiTrack 球物理在 Isaac/MuJoCo/solver 中同值，question bundle 已按新 identity
   重物化；未充分辨识的切向参数不冒充部署硬真值；
6. 默认不采用 acceleration/jerk governor，必须保留 hard q/effort/velocity 安全边界；
   只有真机证据重开时，Isaac/MuJoCo/C++ 才需使用同一逐电机参数、executed state
   observation 和冲突处理；
7. 同一 checkpoint 在 MuJoCo 验证动作方向、teacher fidelity、table/ball/racket 物理和无桌碰；
8. G06/G07 dry-run、joint order、scale、stop 和 no-publish safety gate 通过前不得接真机。

### 3.6 Codex 无法自行生成、必须由人或硬件提供的输入

这些不是代码待办；在对应输入到位前只能保持 `OPEN/Partial`，不能用 simulator truth 或旧青瞳
数字代填：

| 最迟边界 | 必须提供的输入 | 为什么必须来自人/硬件 | Codex 收到后能做什么 |
| --- | --- | --- | --- |
| 1000-update 判读 | 对 loop/block teacher 动作是否“语义上像预期反手拉/挡”的人工视频裁定；如不接受，指出具体关节/相位 | 数值 imitation error 不能定义人的动作语义与真机可接受观感 | 绑定视频/checkpoint，定位 phase/joint/reward 冲突并设计最小 canary |
| formal N5 前 | 对 exact 五动作 ordered manifest 的最终人工确认，尤其新正手是否取代旧正手以及站位版本 | 动作集合是产品/运动学选择，不能由训练分数静默改写 | 重钉 manifest、action UID/order、motion/admission 与 fresh launch receipt |
| formal landing / N5 前 | OptiTrack 30 分钟落球/反弹补测及 Motive 配置，裁定 table effective restitution 和未充分辨识的切向参数 | 当前 tangential retention/paddle decay 仍只是 prior；软件不能从缺失实验恢复真值 | 重新拟合 physics YAML，生成 Isaac/MuJoCo golden parity 与新 bundle |
| 部署意图重训前 | OptiTrack v2 的 capture/source/receive/consume timestamps、Motive smoothing 档位、遮挡/dropout 记录、marker cluster→`base_link` 与 venue→table SE(3) 标定 | 延迟、外参和噪声是现场测量，不可由仓库推断 | 完成 196-D N1 freshness successor、因果 SE(3)+twist producer、噪声/延迟注入与 C++ parity |
| governor 重开时（默认不排产） | A3 逐电机真实 command acceleration/jerk 超出现有 q/effort/velocity 安全链的测量证据，以及实际 PD/torque-speed/delay 约束与厂商确认 | 没有真机反例就加硬 governor 可能削掉击球拍速或制造 sim-to-real gap | 只在重开后生成同一 Isaac/MuJoCo/C++ governor、executed-command observation 与干预率验收 |
| N=73 / full-body 前 | exact ordered 73 件与 full-body motion 的缺失 bytes、逐件证书、站位元数据和人类采用顺序（仅当 repo/Pod 现有制品不完整） | Agent 不能臆造缺失动作资产，也不能替人决定动作优先级 | 做 inventory/admission、任意 N 压力门、schema-v2 ready 和 fixed-width intent smoke/probe |
| canary 采用时 | Reward/curriculum/full-body 等经验 canary 的最终采用裁定与 GPU/停跑预算 | 代码可出证据，但产品权衡与算力授权归 Franco | 冻结 chosen recipe，更新 NOW/Gate 并发下一阶段 |
| 真机前 | 人类对 G06/G07 dry-run、急停链、场地净空和发布权限的明确放行 | 涉及真实机器人和现场安全，Agent 无权自授权 | 只在门通过后执行被批准的 no-publish/有限发布步骤 |

当前 merge、Pod fixed-194 v2 构造验证、旧三条运行守护、正式收据热路径重构、zero-weight term 结构裁剪、
exact-resume parity 与 N5 launcher 工程本身**不需要新的人工信息**；只需要可用的 Pod 槽和现有
仓库/制品访问。

## 4. 决策账本（附录 C；采用/拒绝依据，不参与调度）

本节说明为什么采用、推迟或拒绝某项设计，不承担 TODO 状态。当前状态只维护在
[当前执行看板](#0-当前执行看板本文唯一活跃-todo)。

### 4.1 本轮外部来源→证据→决策矩阵

本表是对
[`dr_reward_external_diligence_20260731.md` §1–13](../../research/dr_reward_external_diligence_20260731.md)
的可见吸收结果，不是只放一个链接。“一手”指厂商提供的同底盘表、官方 repo
或原论文；“工程证据”指 maintainer issue/profiling；无法打开原文或只有二手引用的内容
不做承重裁决。

08-01 逐节复核索引如下。它只证明报告没有“读过但没落账”的孤儿结论，不另建排期：

| 报告部分 | 吸收入口 | 当前边界 |
| --- | --- | --- |
| §1–2 结论/我方已覆盖，§4 全量对照 | 本节 BeyondMimic、unitree、mjlab 与智元四行，以及§4.2 的 action-change、q/qdes/projection、PD/mass/CoM 分账 | 保留已领先轴，不为“向外学习”重复改造 |
| §3 可借清单 | §4.2 的 push/delay/`joint_acc_l2`/RSI/armature/torque-limit/hip-anchor/racket-restitution/history 行 | 仅 vendor push+delay 进 fresh 身份；其余按 `DEFER/REJECT` 边界 |
| §5 侧发现 | §2 的 legacy `reward_pack`/config truth 已完成证据，§4.2 对应两行 | 已直修并测试，不再是 TODO |
| §6 旧排序 | §4.2 “报告§6的历史优先序” | 被智元新身份与本文§0 supersede，不用旧排序阻塞 N1 |
| §7 误差核 | mjlab、IsaacLab Reach、SMASH/KungfuBot 来源行，§4.2 的粗细核、normal/velocity、P2/P3/P5/P6、adaptive-sigma 行 | 粗+细正值核先行；负 guidance/终止不混入；拒绝 live-minimum no-op 伪 canary，adaptive C 只允许 code-owned maximum→minimum 独立 schedule |
| §8 PPO/探索/normalization | 小初始探索来源行与§4.2 的 `init_noise_std`、gamma/timeout、entropy/LR、normalizer、`noise_std_type`、max-iterations 行 | 保持 zero-head+0.02，用收据监控 std/LR/normalizer，不盲抄 1.0/0.005 |
| §9 reset/恢复 | PHC/ProtoMotions/recovery 与 DeepMind 来源行，§4.2 R0–R7/reset/RSI/post-swing 行 | 反 mid-swing airdrop 不翻案；先 reset 降本与保真 termination，后续只改“下回合练什么” |
| §10 循环/吞吐 | contact/Genesis/maintainer profiling 来源行与§4.2 reset O(term)、contact single-read、episode length、debug-vis/sync CI、inactive RewardTerm 行 | 先按 vendor fixed-workload profiler 定价；不用宣传 FPS 替代 same-tape 证据 |
| §11 智元同底盘 | 智元来源行和§4.2 nominal/gain split/delay/push/eval/noise/reward 行 | Franco 明确裁定智元新 A3 表高于仓库旧 nominal；因此新表同时统治训练流程和表内逐关节数值。仓内乒乓/URDF/MJCF/deploy header 只作旧 SKU 证据保留，不覆盖新表；AMP 权重仍不跨经济直抄 |
| §12 先例补遗/诚实空档 | HITTER/SMASH/PACE、IsaacLab Reach、Dextreme、接触、小 sigma、recovery/reward 等来源行 | 一手先例提升候选置信度，不把“找不到先例”伪造成默认采用 |
| §13 curriculum | ActionBall curriculum + 八家对照来源行、§0.3 三条 N5 TODO、§4.2 Curriculum-R1–R9 | N1 明确 fixed-domain；R1–R6/R8 按前置采纳，R7 不单开，R9 先形态1后观察形态2 |

| 来源与证据性质 | 原建议 / 它真正证明的事 | 本轮决策 | 具体落点 / 最迟边界 |
| --- | --- | --- | --- |
| **智元 instinct_mj A3 新表 + 仓内 deploy/URDF/MJCF 旧原件**（同底盘一手配置） | 新表提供逐组 nominal、Kp `(0.8,1.2)` / Kd `(0.7,1.3)` startup DR、`[0,2]` control-step delay、六轴 push、joint-zero/proprio noise、reset/history/mass/CoM/material；仓内旧原件在 waist-yaw、waist-pitch effort 和 wrist pitch/yaw 上与新表冲突 | **`ADOPT` 智元新表数值**：waist-yaw Kp `80`、waist-pitch effort `115`、全部 wrist roll/pitch/yaw `30/2/24/0.004968`，action scale 固定为 `0.25×effort/nominal_Kp`；`ADOPT` gain 拆键目标范围、delay/push；`DEFER` reset 噪声、history=8、腕+拍质量 `±20%`、摩擦 `(0.2,1.8)/(0.2,1.5)`；`NOT-APPLICABLE` depth/ray/terrain；`REJECT` 直拷 AMP 权重与 `freeze_upper_body` | 今晚 fresh N1 用新表 nominal + delay/push；当前 stable-ready 仍关 PD/mass/CoM DR，gain DR 只在 healthy baseline 后恢复。push 用智元幅值但保留 `5–15 s`；旧 `85/118/20/6` candidate、C1 与 checkpoint 都不 resume |
| **BeyondMimic / whole_body_tracking**（原论文 + 上游 repo 一手代码） | 固定宽 imitation 核、`action_rate_l2≈-0.1`、1–3 s push、失败 bin 加权的 clip 内起点采样；**没有** Kp/Kd 随机化 | `RETAIN` 我们已对齐的 imitation/action-change 骨架；`ADOPT` push 方向作独立佐证；`DEFER` 失败加权，`REJECT` mid-swing airdrop | push 已进 vendor profile；formal N5 只做回合起点的失败加权“选 action”，不复制 teacher 中段动量；不把该库当成 PD DR 依据 |
| **unitree_rl_lab**（Unitree 官方 repo 一手代码） | mimic push；全任务族 `joint_acc_l2=-2.5e-7`；部分任务手工 obs scale/history；没有 PD gain DR | `ADOPT` push 为第三个独立佐证；`DEFER` 真 `q̈` 罚和 scale/history；`RETAIN` 我们更强的 q/qdes/projection 分账 | `joint_acc_l2` 只在 healthy vendor probe 后做单变量消融，不默认进 N1；history 需改观测/部署合同，只在实测单帧别名后重开 |
| **mjlab**（官方 repo 一手代码） | task 中实际使用 push 和失败加权 reset；延迟原语含 hold-probability；manipulation 有粗/细双核；PD/armature 随机化只有原语、没接任务 | `ADOPT` 粗+细核形状（Pod 入窗分布已证明精核死区）；`DEFER` hold-probability/RSI；`REJECT` 把未接线原语说成运行先例 | vendor N1 位置核直接采用 `std=0.30 m` 粗项 + `0.075 m` 精项；不抄外部权重；hold/dropout 等真机丢包证据 |
| **HITTER**（arXiv v2 原论文一手；无公开 reward 权重/核宽） | 相对击球 task、50 Hz WBC、固定 PD、时间窗 reward；论文对 DR/push/delay **什么都没说**。Fig.3/4 另给出 0.5 s 前跨拍半径阈、偏移 `<0.75 m` 时 `<0.8 s` 收敛、`94.3%` 敏捷成功和真机 26 抛 24 回的评估量尺 | `RETAIN` relative task、PPO/WBC 骨架与接口语义；`SUPERSEDE` 固定 PD 物理，以智元同底盘新表为准；`REJECT` “HITTER 对齐所以不能 push”推论；Fig.3/4 只作 future evaluator 量尺 | fresh N1 保留 HITTER-derived 177-D term 语义，但不用 HITTER 沉默反对 vendor DR/push；不从论文臆造未发布权重。敏捷/偏移量尺 `DEFER` 到 base mobility / formal N5，不倒灌今晚 no-move N1 |
| **SMASH**（arXiv v1 原论文一手；无公开代码） | adaptive tracking sigma、adaptive region sampling、Motion-VAE；上身-only 与全身任务成功率打平、但动作质量差约 33%；642 连发真机的正/反手成功率约 `66.7%/38.9%`，是反手风险量尺；Motion-VAE 还含 KL 周期退火、phase-cycle loss 与 `0.035 m` 足穿地门 | adaptive sigma 定为 **`REOPEN-CANDIDATE / DEFER DEFAULT`**；`REJECT` 从 live minimum 起步的 no-op 伪 canary；`DEFER` full-body/Motion-VAE；`RETAIN` 上身-only 可先产 N1，且将反手不对称记为今晚判读先验而非调参依据 | 较难 `bh_loop_c` 只保留 code-owned `0.20/1.0/0.52 → 0.075/0.5/0.262` 单调收紧的 C profile；双 term 三参数锁步，共同 coarse position=`0.30`。Motion-VAE 三项进入 canonical motion / N73 战役，不阻塞今晚 |
| **PACE / TTRL**（论文一手定性 + shipped repo commit 一手数值） | 代码默认 vxy `±0.2 m/s`、`5–15 s` push；论文本身不公布这组数；actor 用 world-absolute base position | `ADOPT` `5–15 s` 为 ActionBall 节奏直接先例，幅值改用更权威的智元六轴表；`RETAIN` relative task + absolute table-context 拆分，不整体迁入 world task | 下一 fresh N1 已按该节奏接 push；PACE 数字引用必须指向 code commit，不张冠李戴到 HITTER |
| **KungfuBot / PBHC**（NeurIPS 2025 论文 + 官方 repo/default config 一手） | `sigma=min(sigma, EMA(error))` 单调收紧为论文主方法且出厂开启；不是固定 iteration 时间表 | **`REOPEN-CANDIDATE / DEFER DEFAULT`**；它提升了 adaptive-sigma 的先例强度，也同时证明“静态阶梯有外部共识”不成立 | 先例只支持 maximum→minimum 收紧，不支持从 minimum 起步。C 必须 fresh-only/no-resume，权重/DR/reset/seed 与主臂一致；主臂仍不开 adaptive |
| **IsaacLab Reach**（我们同框架的官方 task 代码一手） | 同一末端误差默认并联线性粗项 + tanh 细项，是粗/细分层的直接实装先例 | `ADOPT` 形状，不拷它的权重或 tanh 具体经济 | 与 mjlab 独立收敛到 vendor N1 `0.30 m + 0.075 m` 双 exp 核；粗项低于 hit 层，精项保验收语义；下一 probe 重记入窗分布 |
| **IsaacGymEnvs Dextreme / 自适应领域随机化（ADR）**（repo 一手代码；谱系论文本轮未全部重读，不承重） | 灵巧手栖有 effort-limit/armature DR 与动作/观测延迟队列；legged 社区并无同等默认 | `RETAIN` 延迟队列是可行设计的辅证；`DEFER` armature DR；`REJECT` 本轮 torque-limit DR | 延迟数值只信智元同底盘 `[0,2]`；armature 若在 healthy in-hold 后开，从小范围且同步 MuJoCo；torque DR 因余量约 `3.5%` 不排产 |
| **Isaac Gym / MJX / legged 接触简化与 Genesis 性能宣称**（官方模型、maintainer 复现与 issue profiling；不把宣传 FPS 当同工作量证据） | 简化腿部 mesh、单次读接触视图可降消费端开销；跨引擎 FPS 若 substep/自碰/工作量不同不可直比 | `ADOPT-CONDITIONAL` 先 profiler 和 same-tape oracle parity；`REJECT` 为速度盲换引擎/碰撞体 | 只有 vendor probe 证明 contact consumer 仍热才合并单次读；任何 solver/引擎提速必须钉相同 substep、self-collision、contact 和 env-step 工作量 |
| **小初始探索的 PPO / 模仿论文组**（报告§12.C 原论文与开源配置核查） | zero/near-zero residual head 有跨 DDPG/SAC/PPO 的直接先例，但多为“基策略+小残差”架构；DAWN 证明小方差会在 SAC 熵项下造成放大病理，**没有任一来源直接分析 PPO adaptive-KL × 我们 `σ=0.02` zero-head** | `RETAIN` 我们的 0.02 和运行时 std/LR 收据；`REJECT` 盲抄 1.0；`DEFER` log-std/熟化改动 | 新 probe 继续强制 std finite/positive 与 LR 轨迹；只有 floor-lock 或穿零证据才重开 noise parameterization |
| **PHC / ProtoMotions / MoCapAct / recovery curriculum**（报告§12.E 原文与 repo 一手机制） | recovery 与失败采样有强先例，但多数允许从 reference 中段或跌倒态重生 | `ADOPT` 失败加权的“下回合选 action/来球”思路；`DEFER` post-swing/recovery curriculum；`REJECT` mid-swing airdrop | N5 的 R3 只在 ready 边界选任务；R1/R2 与 reset 成本闭合后才评估 recovery buffer，不白拿 teacher 动量或新球 hit 收入 |
| **DeepMind 乒乓 Nature/arXiv 系统**（原论文一手） | 按 7 类来球的当前失败率反比重采样任务条件，类内再均匀采样；是真机多轮数据闭环 | `ADOPT` 回合起点的失败加权任务/action 采样思路；`REJECT` 把这外推成 mid-swing RSI | formal N5 的 R3 只改“下回合选哪个 action/来球类”，不改 ready 出生状态；N1 单 action 下无意义，不阻塞当前车 |
| **IsaacLab issue #5018 / Contact Sensor 文档**（maintainer profiling 工程证据，不是官方 benchmark） | 接触求解本身便宜，每步重复读 `.data` 刷新 filtered matrix 才是大头；单次缓存可回收大部分开销 | `ADOPT-CONDITIONAL` profiling/每步单读原则；`REJECT` 无证据就换引擎或盲降 solver iteration | vendor probe 若再证 contact consumer 是热点，以旧 exact 视图做 per-env/substep oracle parity 后缓存/合并；不再把它写成 6–8 s 的无条件前提 |
| **legged_gym / IsaacLab / humanoid-gym / FALCON / Booster 等 reward 实现**（官方或项目 repo 一手代码；理论论文强弱分开） | torque-tail 普遍依赖隐式-PD 解析代理；自碰多用子步/身体计数而非力值求和；角动量有论文先例但非本框架默认 | `DEFER` torque/angular-momentum/pelvis-torso 形状；`ADOPT` 若做 self-collision 只用计数形；`REJECT` 拷外部权重和力值自碰和 | healthy baseline 后先零权重可观测探针，再决定是否计价；隐式-PD torque 可观测性未闭合前不开 torque-tail reward |
| **ActionBall curriculum §13 专项 + 八家对照**（仓内算术/真实开关一手审计；Dextreme-ADR、DeepMind-TT、legged/IsaacLab/mjlab/PBHC/PHC/ProtoMotions 对照） | 我们的失败带、冻结 canary→heldout 与 state SHA/resume 强；但 154 行 Wilson 使实际扩张门只约 `3.25%`，单臂不可逆、样本不足误判太难、全域 blocker 误锁无关臂；N1 实际关掉 promotion，整机尚未通电 | `RETAIN` 指标/冻结双窗/state SHA；`ADOPT` R1–R6 设计方向与 R8 训练侧失败加权；`DEFER` R7 乐观扩张和 R9 并行实现到前置闭合；`REJECT` 在当前不可逆算法上单独开 R7、把首次通电押到 formal long | N1 明确 fixed-domain 并继续发车；formal N5 前必须修 R1–R5 并跑 R6 短验收；R8 可独立但不插队当前 N1；R9 形态1 需 R1+R2+R4，形态2 不排产 |

跨来源冲突的优先级是：**Franco 显式裁定 / 智元同底盘新表 > 官方运行代码 >
原论文已发布数值 > maintainer 工程证据 > 类比/二手推断**。这是为什么我们用
智元的 A3 幅值而用 PACE 的 `5–15 s` 节奏、用 IsaacLab/mjlab 的粗细形状但不拷权重，
也不用 HITTER 的沉默否决 push。

今晚三臂的 source→setting 合成固定如下；这是前面来源表的运行落点，不新增第四种配方：

| Lane | 共同吸收 | 本 lane 增量 | 明确不带 |
| --- | --- | --- | --- |
| A `bh_loop_c_static_v1` | 智元新表 nominal、`[0,2]` episode-fixed delay、智元六轴 push 幅值；PACE `5–15 s` cadence；mjlab/IsaacLab Reach 的 `0.30+0.075 m` 粗细位置核；fresh 194-D v2 | `bh_loop_c` 动作，static fine widths | action one-hot、hard acceleration/jerk governor、PD/mass/CoM DR、mid-swing RSI |
| B `bh_block_static_v1` | 与 A 完全相同 | `bh_block` 动作，static fine widths；必须拥有自己的 identity/pin/receipt | 不借 A 的 pass/pin，也不恢复 one-hot |
| C `bh_loop_c_monotonic_fresh_canary_v1` | 与 A 完全相同 | SMASH error-adaptive + PBHC monotonic contraction；三通道从 `0.20/1.0/0.52` 收紧到 `0.075/0.5/0.262`，算法见§3.2 | static iteration 阶梯、resume、operator 可调 cadence/bounds |

### 4.2 跨来源综合决策

| 项 | 分类 | 最迟边界 | 是否需要学习 A/B | 当前决定 |
| --- | --- | --- | --- | --- |
| A3 nominal Kp/effort/armature/action scale | **ADOPT 智元新表；新表覆盖仓库旧 nominal** | 下一条 fresh N1 | 否；做 authority/nominal-hold 门 | Franco 权威层级已明确：waist-yaw `80`、waist-pitch effort `115`、全部 wrist roll/pitch/yaw `Kp/Kd/effort/armature=30/2/24/0.004968`。action scale 固定为 `0.25×effort/nominal_Kp`，未来 PD DR 时也**不随随机 Kp 重算**。当前 C1/authority/hold/bundle/pins 因 plant bytes 变化作废，必须重物化 |
| Kp/Kd DR 拆键 0.8–1.2 / 0.7–1.3 | 采用配置合同/目标范围；运行时延后 | healthy stable-ready baseline 后的 robust-hold 恢复门 | 否；做 compose/robust-hold 机械门 | legacy `pd_gain_range` 只作兼容，与新键混用 fail-loud；**当前 diagnostic 的 `stable_ready_plant=true` 依然将 PD/mass/CoM DR 设为 `None`**。未来恢复按智元权威语义：startup 时每 env 独立 log-uniform scale 一次、episode 内及 reset 后固定，不沿用我方旧的 per-reset 重抽；action scale 仍按 nominal Kp。先做 in-hold/teacher-to-hit，不把“合同已采用”写成“本轮已随机化” |
| actuator delay `[0,2]` control steps | 直接采用 / 新物理身份 | 下一条 fresh N1 | 否；做 0/2-step、partial-reset、exact-resume 门 | 每 episode 一次、集内固定、31 关节共享 lag；不把 physics-step 死代码当 control-step |
| 智元 6-DoF push 幅值 | 直接采用 / v2.3 断链修复 | 下一条 fresh N1 | 否；做 event/safety 门 | `±0.25/±0.1/±0.26/±0.39`；首版保留 5–15 s 无 gate，formal launcher/profile 机械持有；recovery-hold gate 等资格与 skip/applied counters 接线后再启 |
| delay 与 push 的组合顺序 | **SUPERSEDE 尽调§11.3“先 push、后 delay 单独 A/B”调度** | 今晚三条 N1 | 不做传统 baseline A/B；各自机械门 + 运行时分账 | 用户明确要求今晚按最可能正确方案直接长训；两者都有同底盘数值与独立可观测 counter，且不改变 Reward 经济。故两动作主臂共同带 `[0,2]` episode-fixed delay + vendor 幅值 push；若出问题按 delay histogram 与 push applied/skip/event 分账定位，不用缺基线当借口 |
| 智元双评测口径 | 直接采用 / 报表合同 | 首个 vendor checkpoint 判读 | 否；做 compose-before-`gym.make` 与 receipt 门 | [`vendor_play_v1`](../../DEFINITIONS.md#a3-vendor-eval-profiles) 保留 obs 噪声+delay；`deterministic_ranking_v1` 再关 obs/delay/reset 噪声。两者不混报，host vendor-eval/canonical/formal 合跑 `128 passed`，尚无 Pod eval |
| Wave-P 历史波 / `PROGRESS` 口径 | **ADOPT 口径修复 / CLOSED** | 已完成 | 否 | `PROGRESS` 与 EXP-P1 已统一为 14 臂真实发射、4 臂 never-launched；`closed_incomplete/superseded`、`no dose winner`。原资产只作 directional evidence，不补训/不补卷，不以此冒充 vendor push dose 裁决 |
| strike-window 入口拍距 + 粗细核 | **ADOPT（诊断已命中）** | 下一条新 N1 probe | 否；这次有预注册运行量尺 | vendor full-DR probe 的 100 个入窗样本中 `97% >0.20 m`、均值 `0.434 m`，精核死区实锤；新 vendor recipe 直接使用 `std=0.30 m` 粗项 + `0.075 m` 精项，粗项默认在其他 task 中为零，下一 probe 复测分布与 strike |
| `racket_normal` / `racket_velocity` 死核 | **DEFER 自动改核，保留必须处理的证据门** | safety 健康后的首个 1000-update 前 | 运行分布命中门后不再重复做“是否死区” A/B；只比实现形状 | normal 是报告最重死核、velocity 较轻；当前只新增了入窗拍距直方图，**没有**冒充已有 face/velocity 入窗分布探针。先修 actual-hard，再用现有 at-strike/exact-strike normal/velocity 标量定谳；若它们仍落死区，直接采正值粗+细形状/静态宽核，不先加负 guidance |
| P2 线性 face guidance / P3 pos gate | **DEFER** | 粗+细正值核仍不足时 | 是，且 P2/P3 分开 | P2 是负项，最接近“软罚压制击球”故不默认开；P3 只减坏状态收入，不单跑以免掩盖粗核效果 |
| P5 mean-of-exp / P6 racket-error termination | **REJECT** | — | 否 | P5 共享基座高爆低益；P6 把 death penalty 砍在“没够到球”上，会直接教策略不击球 |
| `init_noise_std=0.02` + ready-bias zero head | **RETAIN** | 当前 N1 | 否；保留 4σ 逐关节安全证明 | 外部的 `1.0` 不适用于我们末层 weight=0/bias=ready 的 fresh bootstrap；继续记 std/LR，不盲抄共识 |
| timeout bootstrap / `gamma=0.99` | **RETAIN / 关闭质疑线** | 当前 | 否 | timeout 截断仍自举；virtual landing 触球步即发，0.76–1.27 s 因果窗被 gamma 半衰期 1.379 s 覆盖，不再用“信用饥饿”改 gamma |
| max-iterations / entropy / 变频 gamma / clamp 注释 / 未标定摩擦 | **RETAIN 有界 25000**；其余分类处理 | 当前或 formal sim-to-sim 前 | 只有 entropy 在 LR 风险实锤后向上 canary | 默认上限已改 25000；entropy 保留 0.01，不抄外部 0.005；decimation=4 固定，变频 gamma 只进检查单；clamp 旧注释做直修；关节摩擦未标定时 formal MuJoCo 依然 fail-closed |
| `joint_acc_l2=-2.5e-7` | 延后、只准具名消融 | healthy vendor probe 后 | 是 | unitree 跨任务同值是有效先例，但它最接近“软罚压制击球”的旧坑；不默认混入首跑 |
| weight-independent action-acc / implicit-PD probes | **ADOPT / IMPLEMENTED / 默认关** | 已完成 | 否；已有零权重不被裁剪的 counter 测试 | jerk 与 analytic post-step effort proxy 都可显式开启，默认不构造 term/不增 vendor 热循环；前者不冒充实际 `q̈`，后者不冒充 PhysX 实际力矩或子步峰值 |
| legacy `reward_pack` compose matrix | **ADOPT / IMPLEMENTED** | 已完成 | 否 | 7 legacy + 2 ActionBall compose 通过；legacy 显式钉 v1，ActionBall/vendor 保持 v2；两条 `1 env×0` trainer dry-run 自然完成 |
| 三处 legacy 配置真值清理 | **ADOPT / IMPLEMENTED** | 已完成 | 否；post-compose 已对账 | `hope_env_cfg.py` 的 PD startup 注释、Hitter/DeployParity 的 IdealPD 旧 rationale 与 implicit-A3 `arm_torque_saturation_weight` 名义值均已改成组装真值；详细证据在§2/G05/PROGRESS |
| reset 从 `O(env)` 改为 `O(term)` | probe 后条件性能直修 | vendor probe 仍显示 reset 占主导时 | 否；做相同 tape 与 checkpoint/receipt parity | 报告的 `60–75%` 是旧估算；旧身份后续 diagnostic 已达 `6.700 s/update`，新 vendor probe 先重新分段定价，再缓存 immutable SHA/批量转录，不删审计真值 |
| contact view 单次读取/缓存 | probe 后条件性能直修 | vendor probe 证明 contact 消费仍热时 | 否；做相同接触/termination tape parity | 外部 profiling 先例成立，但当前仓内主瓶颈已转为 ActionBall fixed-direction LM；不再把 contact merge 写成 6–8 s 的无条件前提 |
| fixed-direction LM 与 `6.700 s/update` | 已直接采用 / diagnostic 性能证据 | 当前 vendor probe | 否；保留 fixed-tape 数值验收 | 不把 task LM 误写成 PhysX solver，不用 TGS/Newton/Genesis 先例直接降 `cq_n_iters`；`6.700 s` 仅是 diagnostic，formal 仍需 receipt/exact-resume 门 |
| headless debug-vis + sync-budget CI | 无学习变量的防回退修复 | vendor long 前 | 否；做 composed-config 和每 policy-step `.item/.cpu/.tolist` 预算测试 | base contact/motion command 仍有 `debug_vis=True`，当前只有局部 sync guard；不阻塞 recipe/smoke |
| §10 热循环三个 NEW 项 | **RETAIN 地板 / ADOPT 结构裁剪 / DEFER metrics 降频** | formal N5 前，不阻塞当前 probe | 否；做 fixed-workload profiler 与数学等价测试 | vendored `rsl_rl` 每步 2 次无条件同步是预算地板，不冒充业务回归；36 个 inactive RewardTerm 应在 formal graph 不构造；`racket_target` 非安全 metrics 在开发诊断后移 logging/update 节拍，不与安全 counter 一起删 |
| episode `124→300` 免费乘数 | 诊断后直接配置或短 canary | reset 税仍主导时 | 先验明确时不做学习 A/B；必须验收 task horizon | 只在 termination/动作周期允许时延长，不能用 timeout 掩盖 actual-hard；先由 probe 的 episode age 分布定谳 |
| 失败加权“选 action”采样 | formal N5 前直接设计 | N5 | N1 无意义；N5 做 sampler parity/短 canary | 采用 **§9-R3**，不采用 mid-swing airdrop；只改变回合开始时的 action 条件，避免白拿 teacher 动量和新球收入 |
| Curriculum-R1 单臂休眠/重开 | **ADOPT 方向 / N5 前实现** | 首条短 N5 curriculum 前 | 否；做状态机/checkpoint 迁移门 | 旧默认 0 保持一次定终身；N5 profile 显式开定期重测，保留已认证 frontier 不缩域；这是 R9 与 R7 的硬前置 |
| Curriculum-R2 eval-only frontier 配额 | **ADOPT / N5 前实现** | Curriculum-R6 短验收前 | 否；做 canary/heldout 同混合与样本量门 | 不加回合数，只将已在跑的评估行重分配，new-band target `384+`/floor `256`；决策收据必须记录 eval mixture |
| Curriculum-R3 样本不足作废重测 | **ADOPT / 直接语义修复** | Curriculum-R6 短验收前 | 否 | NB 不足不更改 arm/frontier；独立 `insufficient_new_band`连续计数，默认 3 次仍不足才 fail-loud，防静默空转 |
| Curriculum-R4 safety blocker 分层 | **ADOPT / 安全语义修复** | Curriculum-R6 短验收前 | 否；做 center/interior 安全事件归因负例 | 全域 zero-tolerance 仍使全局 hold+报警且不推进任何臂；只有当前 probed new-band 事件才能决定该臂，不把“分层”误作“放宽” |
| Curriculum-R5 rho/center 滞回 | **ADOPT 方向 / 保守开启** | Curriculum-R6 短验收前 | 否；做 zero-tolerance 立即旁路 | rho 普通失败加 dwell；zero-tolerance 仍立即退档。center 默认仍保留旧 `stalled`，N5 专用 profile 才改为退回 certified + hold+报警 |
| Curriculum-R6 专用短 N5 通电验收 | **ADOPT / formal long 硬门** | 任何开 promotion 的 N5 long 前 | 否；它本身就是机械集成验收 | N1 发射器继续写死 fixed-domain；短 N5 必须完成 marginal 决策 + drain/reset + resume，不与 N1 对照 lineage 混用 |
| Curriculum-R7 放宽 `too_easy` | **DEFER / 当前不采用** | R1+R5+R6 通过后再裁 | 是，且只能独立评估 | 不可逆 + 乐观扩张会一步扩过宽并压制击球收入；前置不通就继续 UCB，宁可欠扩张 |
| Curriculum-R8 域内失败加权出题 | **ADOPT 训练侧设计 / N1 后实现** | formal N5 前 | 是，评估看 mix-standardized 口径 | 只加权已认证域内 task cell，不改 reset 状态；`≥10%` uniform、center 保底、action round-robin 不动；认证窗继续冻结固定混合 |
| Curriculum-R9 多臂并行 + 护台面积 KPI | **ADOPT 形态1 设计 / DEFER 形态2** | R1+R2+R4 与 R6 单臂验收后 | 是，先与串行算法同域对账 | 形态1 每窗 2–3 臂，每 env 恰探一臂，`cell_id+probed_arm` 可归因；按护台/速度/旋转/落点族保底轮转，记 `x×y×方向锥` certified 护台面积；跨窗 ADR 队列暂不排产 |
| R1 恢复保真 termination | **DEFER，但是 reset 性能健康后的直接修复** | formal N5 前；N1 long 结果若显示饱和尾可提前 | 先做触发率/reset 增量机械定价，不需“学不学”对照 | 当前 `metrics_only` 是为吞吐买的历史特例，不是学习上更正确；reset 税可承担后恢复 `phase_gated` 以砍掉饱和尾 |
| R2 拆 canonical-ready 双职能 | **ADOPT 接口方向 / DEFER 行为变更** | formal N5 的 R3/R4/R5 之前 | 否；先做默认字节等价与负例 | 将 hash/字节等价约束与 reset 分布裁定分键；当前仍 strict-ready，不借拆键静默开 RSI |
| adaptive RSI / post-swing buffer | 延后 | R0 reset 与保真 termination 健康后 | 是 | sampler 已在库内但现有 task 结构不可达；反 mid-swing 裁定不重开，优先 R0→R1→R3 |
| adaptive sigma / 静态 sigma 阶梯 | **`REOPEN-CANDIDATE / DEFER DEFAULT`**；C profile `IN_PROGRESS`，live-minimum 伪 canary `REJECT` | common probe 健康且入窗 denominator 非零后；launcher/compose/Pod 全过前不发 | 是，`bh_loop_c` 主臂 vs C 只差 adaptive schedule；maximum 初态是 schedule 所必需，不计第二变量 | 仅开三旗会从 live minimum `0.075/0.5/0.262` 起步并永久 no-op，故拒绝。真 C 由 code-owned profile 从 `0.20/1.0/0.52` 起，additive 与 weight=`0` 的潜在 `racket_strike_success` 的 position/velocity/normal 三参数锁步单调收紧到 `0.075/0.5/0.262`；共同 coarse position=`0.30`，权重/DR/reset/seed 不变，fresh-only 且禁止 resume。有效 Reward receipt 只列非零 additive 三项；success 锁步由 pinned train 的 receipt-before-write gate + runtime atomic scheduler 证明，receipt 若意外列出 success 反而 fail-closed，绝不为了审计激活它 |
| `noise_std_type=log` | 延后 | std/LR 证据显示守卫不足时 | 是，且需新 recipe | 旧 r4 std 无穿零并自 0.02 升至 0.66；先保留 scalar，只加 finite/positive 守卫和 LR 收据 |
| ready 噪声球 / history=8 / obs scale / mass/CoM/摩擦外扩 | 延后，但精确候选已记账 | 新 vendor baseline 已健康后 | 是 | 属于 reset/194-D/部署合同或多轴物理变更，不与本次发车捆绑。智元原候选为所有 body CoM xyz `±0.02 m`、torso+ankle+wrist mass/inertia scale `(0.8,1.2)`、摩擦 static/dynamic `(0.2,1.8)/(0.2,1.5)`、history=`8`、base-ang-vel scale=`0.25`、joint-vel scale=`0.05`；任务特化另保留 wrist+racket mass `±20%`。恢复顺序是 CoM → vendor body-set mass → racket 特化，逐层 in-hold；obs scale/history 会改变 194-D 时序/normalizer 与部署合同，必须 fresh canary，不静默塞进今晚三臂 |
| armature / passive-damping DR | **DEFER** | healthy in-hold baseline 后 | 是 | 当前 **RETAIN 厂商 MJCF 精确 nominal armature**，不等于采用其随机化；Dextreme/ADR 提供灵巧手先例，但不购买腿式直接迁移。若做从 startup `(0.9,1.1)` 起并同步 MuJoCo 合同 |
| motor-strength / torque-limit DR | **REJECT 本轮 / 不排产** | 重新获得足够加速度余量后才重审 | 是 | 当前加速度包络只约 `3.5%` 余量，且隐式-PD torque 不可靠可观；灵巧手 ADR 先例不能直接翻转腿臂计划 |
| hip roll/yaw deviation anchor | **DEFER** | healthy baseline 后 | 是 | 只做 `in_hold` 轻剂量具名 canary，起步权重按智元表用 `-0.01`，不再用旧提议 `-0.1`；不把 parkour 步态偏好默认压到击球步法 |
| racket restitution 从全身 material DR 拆出 | **DEFER，但保留硬边界** | 首次 physical-ball-contact scoring / formal landing 前 | 否；先标定并 pin | 当前解析球路 N1 不因此阻塞；进入物理接触得分前必须将球拍排除出全身 restitution 随机化，钉到实测值 |
| 一阶 action-rate + q/qdes/projection 安全链 | **RETAIN / 二阶 action-acc 归零** | 当前 | 只在抖动量尺异常时调一阶剂量 | 保留 `action_rate_clamped=-0.2`与三条限位分账：qdes `-40` 读 processed/executed 目标，actual-q `-40` 读真实关节，pre-clamp 请求超出由 projection-distance `-5` 计价；vendor leaf 钉 `action_acc_weight=0.0`，二阶项只可作零权重遥测，不叠新 governor |
| 报告§6的历史优先序 | **REJECT 作当前调度 / SUPERSEDED** | 立即 | 否 | §6“速度泛化优先、push/delay 在后、新臂默认不变”是 vendor identity 裁定前的历史语境；当前 fresh baseline 采用 deploy nominal，并吸收智元 gain-interface/delay/push，当前只认本账本§0 与尽调最新版§11/§13 |
| PACE 本地论文资产 | **DEFER 维护卡** | 下一次论文证据刷新 | 否 | 本地 PDF v3 比 arXiv v4 旧；不影响今晚数值（push cadence 引用 shipped code），但后续须版本化替换并重核引用页码 |
| 智元 AMP reward 权重、`freeze_upper_body`、mid-swing RSI airdrop | 不采用 | — | 否 | 收入经济/任务目标不同；mid-swing 还会白拿 teacher 动量和新球 hit 收入 |
| joint-zero + 非对称 actor/critic observation noise | **RETAIN（已对齐）** | 当前 | 否；做 compose/runtime receipt | joint default offset `±0.01 rad` 与 action offset 同源；actor 保留逐通道 uniform noise（`ang_vel ±0.2 / gravity ±0.05 / q ±0.01 / qdot ±0.5`），critic 保持干净+特权观测。这些已与外部/厂商骨架对齐，不重复改 |
| reset state noise | **RETAIN 现有 task 真值 / DEFER vendor N1 R4** | healthy vendor baseline 后 | 是，若为 vendor N1 开噪声则只做具名 R4 | legacy task 保留已注册且与参考栈对齐的 pose/velocity/joint reset noise；当前 canonical-ready vendor N1 继续全零，不在首车静默改分布。若恢复，按智元 A3 目标 x/y/yaw `±0.1`、base velocity `±0.2`、joint `±0.15`、joint velocity `0` 做独立 R4，并重物化 reset/recipe/checkpoint 身份 |
| 智元 depth/ray/camera/terrain/volume penetration | **NOT-APPLICABLE 本轮** | — | 否 | ActionBall 当前 actor 没有 parkour depth/ray 输入，没有地形越野或腿体穿障目标；不把无关 pipeline 塞进 194-D |
| 智元 feet/torso/pelvis/`feet_close_xy` 形状 | **RETAIN 已有足部骨架 / DEFER 新形状** | healthy baseline 后 | 是 | 不抄 AMP 权重；torso deviation 保留我们的 pre-strike 门控，pelvis/torso 拆分、feet-close 只做低优先单变量，防止 parkour 站距偏好压制击球步法 |
| torque-limit / angular-momentum / self-collision reward 形状 | 先零权重探针或延后 | healthy baseline 后 | 是 | 可借形状不抄 AMP 权重；隐式 PD torque 先闭可观测代理，自碰只用计数且先修腕/拍碰撞几何，不做力值求和 |
| 6D table-relative orientation | 直接修 | N=1 long | 否；做 tensor/recipe parity | 已实现，保留完整三角信息 |
| base linear velocity 3-D | 直接修 | N=1 long | 否；做 observation parity | 已进入 194-D actor |
| racket task velocity/normal 统一 heading frame | 直接修 | N=1 long | 否；做 tensor/构造 parity | 新合同名；旧同宽 194-D 不续 |
| OptiTrack pose + gyro angular velocity | deploy contract | 部署 | 否；做传感器延迟/噪声实测 | 采用分量级最优源，不整套弃用 IMU |
| OptiTrack v2 timestamp + localization age/valid | deploy contract | 部署意图重训前 | 否；做 producer/tensor parity | 当前 194-D 不加猜测常量；实测后以新合同迁移，长 stale 由 supervisor 停机 |
| teacher 开始倒计时 | 直接修 / fresh observation contract | 下一条 fresh launch | 否；做 tensor/reset/Pod 构造 parity | fresh N1 固定 194-D v2 用 `time_to_teacher_start_s` 替换常量 one-hot；getter 在 ObservationManager shape probe 先走既有 lazy runtime bind，reset 后仍读取 receipt 真值。当前三条旧 194-D N1 不停机、不重标、不 exact resume |
| 动作身份 actor 表示 | 直接修接口 / N5 前 | formal N5 | 不做 one-hot 学习 A/B；做 descriptor 混叠与 Pod parity | fresh N1 已删除 policy one-hot；UID/slot 仍冻结 sampler/solver/curriculum。formal actor/critic 改吃由 reference 内容生成的固定宽 contact intent，必要时加中间相位 preview。禁止 N 维 one-hot、UID 数值或 per-slot learned embedding 作为 arbitrary-N 正式接口 |
| 2026-07-30 OptiTrack 球物理 | identity/physics 直接修 | formal N=5 或任何正式 landing 结论前 | 否；重新物化和 Pod parity | 科学源已合入；当前 N1 bundle 仍是旧 profile，formal N5 前必须切换并重 pin |
| ChingMu ball/base 噪声直接复用 | 暂不采用 | 永不作为 OptiTrack 硬合同 | 否 | 只作数量级先验；新系统按对象、时间戳与 Motive 设置重测 |
| dynamic-ready 原子合同 | 直接修 | N=1 long | 否 | 已实现；现有 policy recipe 已复用，194-D hard-contract/smoke/probe 已过 |
| stable-ready plant（关 CoM/mass/PD DR） | 直接修后 Pod probe | N=1 long | 否；旧 full-DR 已给失败反例 | 防 shared waist raw-hard；1000 后逐轴恢复 DR |
| plant-state safety guard 5%→6%（vendor-only） | **REJECTED BY POD / source retained as failed evidence** | 已完成否决 | 否；这是安全修复机械验收 | source `fed55f55`、artifact `25400403`，host `412+377` PASS；smoke 也通过，但 4096-env GPU2 update0--5 的 raw event/terminal 为 `0/0、33/15、3238/1303、418/169、2036/830、844/365`，证明“只提前 emergency trigger”不足以治本。finite projection 的逐关节证据又显示：actual-hard 主因 joint05 waist-roll 在六个 update 的 projection trigger/count 全为 `0`；joint08 waist-pitch lower 约有 `1.2%--1.8%` sample 饱和，另有少量 joint19 left-ankle-roll。这否决“actor 在 waist-roll 顶 nominal projection 才穿边”的叙事，下一修复必须对账 transient guard endpoint 与 implicit-PD plant。按预注册停止加宽；不改 raw terminal，不加 acceleration/jerk governor |
| `H_mech/H_ctrl` 双位置包络 | **ADOPT / four-axis v7a mechanical gate PASS** | 已完成；后续只作 fresh probe/push 的 plant 输入 | 否；exact source `62e0878a…` 的 v7a 已过 | waist-roll/pitch + 左右 ankle-roll 四轴每侧内缩 2%，其余 27 轴 live constraint=`Hmech`；`Hmech` 继续驱动 terminal/ledger，soft/Q 逐字节不变。v7a 已证明 8/8 ON/OFF 差分、64/64 contact 为零、input/restore/live identity exact。它不是 acceleration/jerk governor；后续仍须用 `4096×5` 证明 actor 不长期靠 cage 墙 |
| table pose-OBB truth | 直接修 | N=1 long；formal consumer 在 N=5 前 | 否；做 Pod 正负控/吞吐 | diagnostic 已采用 43 collision-component + blade OBB、五桌体 AABB、四子步 sticky；filtered v3 与 sphere 中间态退役。formal producer/consumer schema 已收敛 v4；path-free USD identity 仍开放 |
| diagnostic receipt 快路径 | 直接修 | N=1 long | 否 | 已实现 |
| zero-weight RewardTerm 结构裁剪 | 数学等价直接修 | formal N=5 | 否；做 composed-config/Pod parity | N1 不删 table/contact sensors；formal ledger 仅遍历 active 非零列 |
| qdes weight-independent probe | 安全验收 | formal N=5；部署前冻结复测 | 否；冻结同 policy/task tape | 当前 penalty 权重归零会连遥测一起消失；需独立零值 probe 和 global residence `<0.005` 候选门 |
| 单一 physics/contact 真源 | 跨语言合同修复 | formal N=5 | 否；做 Python/MuJoCo/planner golden parity | 采纳 Jiayi 的模式，不搬其过期硬编码常量 |
| 部署 binary metadata 拒载 | deploy contract | 部署前 | 否；negative tests | 为 ActionBall 最终 obs/checkpoint/motion/physics 生成，不复用 V17 的 180-D 常量 |
| formal checkpoint 粒度 receipt | 直接修 | formal N=5 | 否；做 parity/吞吐 | 开放，不能拖到 N=73 |
| ledger/D2H/broker 热路径 | profiler 后直接修 | formal N=5 | 否；做固定工作量吞吐 | 开放 |
| Reward 权重与负项剂量 | canary | 1000 update 后 | 是 | 不阻塞首跑 |
| reference guard/CaT | canary | healthy baseline 后、formal N5 采用前 | 是 | 继续分账，不热改 |
| curriculum 10%/20% | canary | healthy baseline 后 | 是 | 10% 默认，20% 只作对照 |
| full-body | canary | N=73 前 | 是 | 不阻塞 upper N=1；现有 full bundle 仍是 schema-v1，须先完成 stable-full ready→core→ready、nominal hold、schema-v2 bundle/solver preflight 与 fresh fixed-width successor smoke/probe，不能冒充当前 upper 的可比对照 |
| EMA / 额外 temporal regularization | **REJECT 本轮 / 不排产** | 真机或 policy 反例到来才重审 | 重开时才需要 | 当前只调已有 action-change penalty；不用新的 stateful smoothing 制造相位延迟和部分可观测性 |
| acceleration/jerk governor / Ruckig | **REJECT 本轮 / 不排产** | 只有 A3 真机测量证明现有 q/effort/velocity 硬边界不足时重开 | 重开时需机械验收与学习 canary | 快速击球不默认需要 acceleration limit；不再把它写成“1000 update 后自动试一格”，部署仍保留 hard q/effort/velocity 与急停 |
| 8192 env | 性能 canary | formal N5 吞吐健康后 | 是 | 不用更多 env 掩盖热路径问题 |
| OptiTrack/IMU 噪声与延迟 | 实测后直接建模 | 部署意图重训前 | 不用猜值 A/B；估计器方案可 canary | 旧青瞳数字只作量级先验 |
| 关键通道 observation history | canary | 实测显示单帧别名后 | 是 | 首个 N1 不加；若做须绑定 reset/exact-resume state |

## 5. 收口判据（附录 D；验收参考）

### 可发下一条 fresh 194-D v2 N=1

- table smoke receipt 可信；
- 当前 dynamic-ready recipe 的 SHA 已验证，历史 194-D hard contract 已由 smoke 实例化；
  fresh fixed-194 v2 teacher-start 合同须另做 Pod 构造 smoke；
- `1 env×2` 与 `4096×5` 有真实 PPO update、finite checkpoint；
- stable-ready plant 下 episode 能跨动作 `t_hit`，没有 NaN/identity 漂移/持续
  table/fall/raw-hard 爆炸；
- exact spec、recipe、run record 已保存。

### 可从 1000 update 转 reviewed long

- checkpoint 和训练连续性正常；
- teacher imitation/strike denominator 可解释；
- unsafe 与 Reward income 分组未失效；
- 若 strike 仍为零，teacher dynamic feasibility 根因已判明，而不是靠临时改 Reward 掩盖。

### 可发 formal N=5

- formal receipt checkpoint compaction、重建、exact resume 与 fixed-workload throughput 通过；
- fixed-width continuous action-intent launcher、exact N=5 action/admission/ball/support、
  Reward/table/evaluator 全闭合；
- 使用新的 clean/no-clobber lineage，不把 N=1 diagnostic 结果冒充 formal 证据。

## 6. 证据入口（附录 E）

- 当前训练 Gate：[G05 Isaac training first loop](../../gates/G05_isaac_training_first_loop.md)
- ActionBall 语义：[按动作条件化 Ball-first 合同](../../interfaces/action_conditioned_ball_first_contract.md)
- actor/传感器接口：[Policy observation/action](../../interfaces/policy_observation_action.md)
- N=1 发射步骤：[消融与 dynamic-ready 发射工序](../../operations/run_ablation_wave_launch.md)
- formal N=5 工序：[no-clobber ActionBall 发射](../../operations/run_action_ball_curriculum_no_clobber.md)
- table truth：[ActionBall 桌体安全 smoke](../../operations/run_action_ball_table_safety_smoke.md)
- Reward truth：[ActionBall 发射前 Reward 因果审计](../../operations/run_action_ball_reward_causal_prelaunch.md)
- 本轮外部 DR/Reward/reset/吞吐/先例主审计：
  [`dr_reward_external_diligence_20260731.md`](../../research/dr_reward_external_diligence_20260731.md)
- 上一版历史底稿：[N1 设计背书审计与训练加速尽调](../../research/design_audit_and_speedup_20260729.md)

本文不自行声明 Gate `Done`，不更新 `NOW`，也不授权真机。
