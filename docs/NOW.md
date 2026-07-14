# NOW — 当前训练流程、课程阶段与下一步

最近复核：2026-07-15 CST。本页说明现在到底在训什么、整套训练怎样连起来、每个课程阶段在解决
什么问题，以及下一项工作为什么值得做。实验过程放在[实验登记册](experiments/README.md)，
复现命令和 Gate 结果放在对应 [Gate](gates/) 与操作文档。

`origin/main` 上的本页才是运行态权威。功能分支里的修改只是一份提案；合入前必须重新对账最新的
`NOW/TIMELINE/PROGRESS`。缩写和机器代号的完整释义见[术语与人话对照](DEFINITIONS.md)。

## 先看结论

- **当前仍在课程阶段 1：题目使用固定目标站位、固定击球位置和无旋正反手。** 来球速度会变化，
  policy 的下半身没有锁死；训练中动作可以不传送地重复，但这不等于连续不同来球。现有可信
  成绩只有每题重置的单拍，连续格仍未测。
- 最接近正式目标的零关节摩擦配方已在第 2000 与 4000 次迭代各跑完四个独立
  初始化同卷。2k 为 `83、100、100、20`；4k 为 `50、88、98、0`，稳定性在两个
  milestone 都失败。4k seed4 有 21 次物理 root fall，不是晚熟。
- 当前成绩卡是 **Python BankExam 单拍解析诊断，不是 Gate3**。4k seed2/3 的旧分虽是
  `88/98`，但正手 raw-A 法向误差为 `172.33°/174.35°`，signed composite 都是
  `0/50`；这证明旧 scorer 的正手符号盲区，所以没有新 baseline 晋级。fresh 广度池
  最近 24 个 K20 格子又全部正手 signed composite=0，因此 16 臂已分两波全部保留证据并
  停止；这不改写 q10/q50 合同。`n/-n` 负控和 signed-face source gate 已进入 main；下一项会改变
  阶段 1 判断的证据是一个 seed 的四格机制 canary，以及修正 scorer 后的同卷结果。
- 部署侧的 planner-policy exact tuple 源码已通过 portable Release 和 latest-main 本地回归；
  这解决了同 tick base/target 因果配对与源码可构建问题，但 ROS/Jazzy/AimRT、backend first tick、
  厂商 MuJoCo 行为和真机仍未运行。

## 当下状态与团队 focus

现在围绕“几天内做出可录屏、可解释的好球”并行推进三条最短链：把 Franco 五种用途动作和横移老师
做成安全可训练候选；把四个不同因果格先跨卡铺开以修正现役 policy 的拍面反号；把胜出 policy 连同我们的
planner 送进厂商 MuJoCo `Gate3`。Isaac 只负责训练/诊断，最终行为以厂商 MuJoCo 为准。

- **最新可分享结果：** 反手拉 B/C 的 rank-0 主选已在 Pod1 CPU-only runtime 完成 exact 整轨
  SE(2) 实体化；后续 schema-2/FK 源码门也已进入 `main` 并通过 `17` 项专项回归。逐资产 no-write
  runtime inspection 又在 exact donor/私有 PKL 下通过：B/C 分别 `91/98` 帧，未写输出。一次性 consume
  runner/activation 已进入 `main` 并通过 latest-main 全回归。2026-07-14 08:22 CST，B 在 fresh
  absence preflight 后花掉唯一一次 consume：`91` 帧 schema-2/FK NPZ 的 SHA-256 为
  `e2eb99e6...d28cc`，独立 `validate-result` 确认 `runner_lineage=true`、`npz_bound=true`，completion-last
  ledger SHA-256 为 `c0a25f2c...f4f8b`。这只解锁 B 的下一张 L0 静态证书；vendor L1、整轨桌网余隙、
  动力学与 RL 仍未跑。C 保持未消费，只在 B 后续外部安全门失败或证明有独有覆盖时作为后备，不复制
  一个 RL 槽。B 的历史绝对路径 portability 根因已修复并合入 `main`；Pod2 的 exact
  CPU-only full L0 `dry-run` 随后真正进入 151 帧 `mj_forward`，并在旧合同要求的
  float32 逐 bit 相等处 fail closed。实测最大差为 position `1.1920929e-7 m`、quaternion
  `5.9604645e-8`、COM velocity `2.9802322e-6 m/s`、angular velocity `5.9679151e-6 rad/s`；
  无证书、不占 GPU。这不是碰撞/自打/动作安全失败，而是 Pod1 producer → Pod2 重算的
  float32/50 Hz 差分可复现性合同不自洽。旧 v1 失败保留；v2 从 ULP 与差分误差解析推导，
  不放宽关节、地面、支撑脚或安全门。该 v2 已在 exact `main@cc1a2b1` 上通过 Pod2 full
  `dry-run` 与唯一 no-clobber formal audit，certificate SHA-256 为
  `60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`。vendor L1 随后在修复
  runtime→GMR 关节顺序 adapter 后完成 Pod2 full dry-run 与唯一 no-clobber audit：1201 个 400 Hz
  有限样本中自碰=`0`、拍/柄 `<5 mm` 自打=`0`、warning=`0`，最小余隙 `0.13829 m`；certificate
  SHA-256 `6840df34...db60`。B 只解锁下一张整轨桌网余隙门；动力学、RL 和真机仍未授权，C 保持后备。
  高点拍压 S0 与四条横移 M0 不仅通过 exact GVHMR 帧数/finite 审计，还已完成真实五条 PT 的
  canonical-beta `inspect/consume`，non-beta 内容逐 bit 不变。exact GMR runtime source gate 也已进入
  `main` 并通过全回归；Pod2 的 v2 runtime `inspect` 已尝试，但在 consumer 前因合同写死的
  `/workspace/yikang/.../python3.10` 整棵环境不存在而 rc127 fail closed。恢复审计又确认 Pod2 同时缺
  exact GMR tree/283 MB bundle、SMPLX/model/mapping 与 S0/M0 七份 canonical 输入；现有 Isaac venv 也只与
  234 行环境快照精确重合 87 行。两批 output root/lock 仍 absent，不是动作失败；必须先从权威备份恢复
  内容寻址资产，再建独立 v3 runtime，`consume`/schema-2/训练继续 blocked。
- **当前运行态：** 2026-07-15，V1+V2×base-decel fresh v4 两臂已收口。两份 `model_500` 都通过
  filename=embedded、finite、fresh lineage、claim 与同 hard-contract attestation，checkpoint SHA-256 为
  `22f78f88...a6a` / `a1735fbb...c14`。但冻结的 `480–500` 窗内 control post-swing
  eligible/selected/started=`0/0/0`，treatment 已为 `15087/3750/3750`；control 到 step519 才激活，不能
  倒灌 model-500。源码确认 buffer 只收 policy 自己活到自然 clip wrap 的状态，因此 base-decel 会反过来改变
  curriculum 何时 ready，这一对按预注册判 `activation-invalid`，不比较行为、不买第二 seed、不判卷。
  exact PGID=`380610/381237` 已停止；Pod2 GPU1/GPU2 当前空，GPU0 仍只归 Yikang。下一次同轴训练必须先用
  两臂共享、自然-wrap provenance 绑定的 immutable teacher-state receipt 做外生 cold-start，并在首 update
  前 fail closed。

  clean main-effect 也已自然终档：两臂关闭 post-swing replay、固定 V1+V2，只比较 base-decel `0/1`。
  `model_1000` 两份 filename=embedded、finite、fresh lineage、claim 与共同 hard contract exact，原
  PGID `385320/385948` 均已退出。980–1000 的 treatment/control raw base speed=`1.00882×`，按冻结
  `≤0.90×` 门正式 reject，不买第二 seed/judge。源码复核同时发现这个 Reward 实际追踪随目标距离变化的
  `v_des`，并非始终让 raw speed 更低；尾窗 raw-kernel-per-eligible 反而提升 `1.6003×`。因此不改写
  verdict，但后续必须先用 `|v_base-v_des|` 的近/中/远分桶重做量尺，不能复制当前 weight=`1`。

  连续恢复当前最短关键路径转为共享外生 teacher。v1 已在结果前绑定 fresh control `model_500`、动作、题库、
  A3 tree、Pod2 GPU1、4096 条 natural-wrap state 与 20000 inference-step 上限，但运行时派生器在 Hydra
  compose 阶段因遗留 train-only checkpoint 键 fail closed；capture directory/claim/process/GPU work 都未
  创建。源码复核还发现 play 没有实际应用冻结 seed。v1 不重发；当前先合 seed parity、删全 train-only 键，
  再用新 source/output namespace 预注册 v2。attestation、首 reset、科学 pair、第二 seed 与 judge 仍逐门
  fail closed。Pod2 GPU1/GPU2 当前空；GPU0 继续只归 Yikang。

  Fresh C 的五条单 seed 机制格均已越过 `+500` 且 checkpoint
  finite/contract/lineage 正确。V1+V2 出现当前最强击球精度信号（composite `0.0893`、normal pass
  `0.268`），但平衡债仍高；V2 单独格明显落后，已列为可替换且不复制 seed。base-decel、V1 和
  post-swing 保到 `+1000` 看权衡是否收敛。qdot-limit 第一次发射在第 0 update 的 A3 URDF import 阶段
  超时，launcher 只收口其 exact PGID；无 hard contract/model，因此是基础设施失败而非 reward 失败，
  全新 retry-v2 已通过 no-Kit compose 与真实 boot marker。qdot treatment/control 均已自然到
  `model_1000` 并退出；control 终档 SHA-256 `b6672869...12cb9`，filename/embedded iter=`1000`、76 tensors/
  1,762,717 elements finite、fresh lineage=`1`、schema-3 hard contract 与 claim 均 exact。两份 `model_500`
  的 checkpoint/claim/hard-contract/finite 也全过。`480–500` 的末 21 点里 treatment 的 qdot max
  `-16.4%`、near-limit `-20.1%`、torque saturation `-35.5%` 且 fall 改善，但 position pass
  `0.418→0.107`，当时是 mixed signal。`980–1000` 的同口径曲线随后翻转：treatment/control 的 position
  pass=`0.878/0.593`、position error=`4.74/9.62 cm`、signed composite=`0.310/0.146`、virtual
  return=`0.454/0.265`，而 pre/post fall 与 completion 基本持平。这把 `-5` 从“准备调低权重”改判为
  **晚熟候选**；仍不直接采用、不买第二 seed，也不启动交互，先对两份 finite `model_1000` 跑 immutable
  MuJoCo/vendor judge。
  qdot control 的首次冷启动随后在 scene creation 前卡住，iter0/无 contract；exact PGID 已收口并保全。
  仅允许相同配方的 retry-v2 再试一次；若同 phase 重复，停止 retry并转 importer 根因线。
  新 source 的正式 run 前现先走独立 `boot-warmup`（1 env×2 update、180 秒、非科学 namespace），避免再拿
  control/treatment 的证据目录承担冷缓存失败。
  通用 launcher 也已加入 180 秒日志无进展 watchdog；marker 优先，失败只收口自身 exact PGID并写 sidecar。
  conditional source 的 Pod2 GPU1 1-env warmup 虽自然退出并通过 2/2 updates，但随后 4096-env control
  在 dynamic URDF import 后停住，iter0、无 scene/contract/checkpoint；精确 PGID 收口并保全，treatment
  从未创建。这个反例推翻了“1-env warmup 可授权正式 run”。后续 `077e70c` 非科学 probe 又因 clean
  checkout 缺 Git-ignored A3 runtime tree 在 iter0 自然失败；它不是 4096-env 容量或 Reward 负结果，原
  attempt 永不重放。46-file A3 tree 的 YAML source closure/no-clobber hydrate 随后闭合，`main@c7e1a90`
  canary 已自然越过首迭代并终档：result/model/hard-contract SHA-256 分别为
  `02780b52...c4186` / `a813ea9b...38e68` / `c39cf1ae...df838`，76 个 tensor 的 1,762,715 个浮点元素
  全 finite，fatal=0，原 PGID 已自然为空。但它的 `unlock_authorized=true` 只符合 c7 旧终档语义，不能
  解锁科学训练。当前 conditional `0/-0.4` 与 `V1+V2 × base-decel` 两组单 seed pair 已改绑 strict
  `main@caeb9ad` checkout `/workspace/codexschema/nohope_p1_caeb9ad`。新 attempt
  `caeb_strict_terminal_pod2_gpu1_a1` 已真正通过：result/claim/model/hard-contract SHA-256 分别为
  `0d03bd03...9b1d1` / `7437db48...ad5` / `e1b79d14...3106` / `c39cf1ae...df838`，并绑定实际
  4096 environments、物理球/三实体、76 个 tensor 的 1,762,715 个浮点元素全 finite、fatal0 与自然空
  PGID。该非科学 receipt 已被显式队列变更消费：两对均为 ready、
  [`launch_authorized=true`](DEFINITIONS.md#launch-authorized)。conditional control/treatment 随后已分别在
  Pod2 GPU1/GPU2 越过 first iteration（PID=PGID `357023/357679`）；两份 `model_200.pt` 已写入 exact
  milestone receipt，checkpoint SHA-256 分别为 `b55b7d3b...b4b41` / `c07b1f12...bd51`，76 个 tensor、
  1,762,715 个浮点元素全 finite，fresh lineage/claim/schema-3 hard contract 均匹配。冻结
  step `180..200` 后，treatment 的 conditional gate/cost/reward mean/min/max 全为零，由源码公式可
  严格推出 eligibility 为零。这一 setting 已按预注册判 activation-invalid：不买第二 seed、
  不晋级，也不把方向差异写成 Reward 效果。
  interaction control PID=PGID `358331` 则在 first iteration 前的 dynamic URDF import 以
  `malloc(): invalid size (unsorted)`、`rc=134` 自然退出；treatment 未发射，claim/namespace 已保全。
  这不是 interaction Reward/行为失败，也不能写成 pair 已运行。旧 control 行已 rejected、禁止重发；
  逐字相同配方的 `control_retry_v2` 与从未 claim 的 treatment 均 ready，只允许同一 `fill --count 2` 事务
  先观察 retry first iteration 再发 treatment。该事务随后按序成功：retry-v2 PID=PGID `359240` 在 Pod2
  GPU1、treatment PID=PGID `359872` 在 GPU2，二者均已越过 first iteration。两份
  `model_200.pt` 也已通过 finite/iteration/lineage/claim/hard-contract receipt，checkpoint SHA-256
  为 `44a709ac...035a` / `b04e2338...e56b`。但 V1/V2/base-decel 的计数级 activation 仪表不全；
  V2 treatment 窗口 proxy 为零，base-decel Reward 虽非零也不能替代 denominator。该 pair 按
  activation-invalid/instrumentation-blocked 记账，不解释为 base-decel 负结果；旧 `358331`
  永久 rejected。
  “每卡只能发一条”的 launch-lock 根因已修：旧 `flock FILE command` fd 被 trainer 继承；新 controller
  用 fd8 持短锁并对子 launcher `8>&-`。conditional control/treatment 优先同落已 warm 的 Pod2 GPU1，
  正好验证容量不再退化为1；现役 qdot 的旧锁不做任何强制处理。

  2026-07-14 19:00 CST 起 Pod1 全部留给 Yikang 冲刺：Codex 的 V1/V2/V1+V2 只对精确 PGID 发出
  `TERM`，分别停在 iter `792/782/743`，三条 `model_700.pt` 与日志完整保留，未进入 `KILL`；复核 Pod1
  三卡无 Codex compute process。Codex 后续只使用 Pod2。qdot treatment/control 均已自然终档；c7/caeb
  canary 也已自然退出。最近可信 Pod2 快照中 GPU0 只有 Yikang 的外部 trainer，Codex 不管理；GPU1/GPU2
  分别运行 conditional control/treatment 与 interaction retry-v2/treatment，每卡两条 Codex trainer。机器可读
  [`dispatch_pods: [pod2]`](DEFINITIONS.md#dispatch-pods) 已使新 assignment 不可能
  落到 Pod1，普通 live snapshot 也不会读取 Pod1。

  2026-07-15 03:17 CST 的 Yikang 训练快照：纯 A 继续在 Pod1 GPU0（W&B `5nso93g0`）；旧的
  root-velocity B/AB/AB_FRESH 已在保全 finite checkpoint 后按 exact PGID 停止。物理
  `pelvis_link` force 替代组使用 branch `yikang-standhit-0714` 的 exact source
  `0db9a9a8bf27b3389080edef456551257c08a170`：B_FORCE 在 Pod1 GPU1（`e0jg8n35`）、AB_FORCE
  在 Pod1 GPU2（`qfn5xlnr`），两者从原始 `ayzxv1ma/model_10600` 严格恢复；AB_FORCE_FRESH
  在保留给 Yikang 的 Pod2 GPU0（`147kd92u`）从 iteration zero 启动。三条均为 8000 envs/seed 0，
  10 秒 timeout 与 fall termination 未改；更长 recovery horizon 仍是独立 C 实验，不混入本矩阵。

  发射 harness 已在 `main` 收紧：YAML recipe 先拒绝重复/owned key 和 Hydra 控制语法，真实最终 argv 在
  claim 前做 no-Kit compose，run directory 原子创建，canonical claim digest 自动绑定 source/argv/
  预算/motion/bank/exam 并进入 checkpoint provenance。trainer-owned
  [`run_binding.json`](DEFINITIONS.md#trainer-run-binding) 与
  [milestone attestor](DEFINITIONS.md#milestone-attestor) 已进入 main；代表性 full-scene probe 也已实现。
  full-scene finalizer 现在不再信任外层 wrapper 的旧 doctor 结论：它在 terminal 末端自行重算
  target/donor inventory、URDF mesh closure、donor clean commit 与 receipt，并把 `current_closure`
  写入 immutable result；direct-finalize 绕过、资产漂移与 bool iteration/lineage 均 fail closed。
  Pod2-only probe 的外层快照现在只读 selected dispatch slot，不再因普通 fill 的全 Pod 快照访问 reserved
  Pod1；`fill --execute` 每臂也已从“独立 doctor SSH + 内嵌 doctor launch SSH”收为一次原子远端调用。
  ignored asset prepare 与 strict caeb 自然终档已经闭合；两组 pair 的 `+200` 身份和冻结曲线也已复核。
  两组都因 activation 为零或仪表不完而不能产生机制结论；当前 exact PGID 在替换 source
  过门前仍 live，不会复制 seed。下一对 post-swing 概率消融已有 5 项整数计数源码，但红队发现
  共同 V1/V2 计数仍缺，故不跑 strict probe、不发射；先补成真正可早判的 source。

  非击球臂 A0/A1 已完成 checkpoint 层闭环。A1 自然退出；
  A0 的 `model_1000.pt` 写完后在 Kit/Python teardown 挂起近三小时，正式 failure regex 无命中，终档
  内嵌 iteration `1000`、`1,762,715` 个浮点元素全部 finite、fresh lineage `1`，hard-contract SHA
  与相邻文件一致。它对精确 PGID `1811464` 的 `TERM` 20 秒无响应后，只对同一个单成员 PGID 发出
  `KILL`；没有重启、重发或影响别的进程。既有 v1r1 finalizer 随后通过两臂
  `200/500/1000` 的 filename↔embedded iteration、finite、lineage、hard-contract 与唯一 body-mask
  差异，发布 paired result SHA-256 `30ba716b...d7d9`。结果仍写明 signed K100 尚未判卷、不得晋级或买
  第二 seed。两边归档 checkout 仍保持 clean exact `6d93bcb...480b`。

  C2 虽自然终档，但 2026-07-14 的 v1r2 runtime gate 发现它**不是所声明的零摩擦配方**：manifest 写
  `zero_joint_friction=true`，实际 launch argv 没有 `task.plant.zero_joint_friction=true`，相邻 hard
  contract 的 31 个系数全为非零 PhysX 默认值。v1r2 在任何 attestation/claim/D2 run 前 fail closed；
  D2 永久不续跑，C2 只保留为 nonconforming 根因证据。全新 C3/D3 已在 Pod1 GPU1/GPU2 各只发一次并
  自然到 `model_24.pt`；31/31 零摩擦、finite/iter24/lineage/claim/hard binding 与 paired receipt
  `bb3cd749...bbde` 均通过。它只证明 L1 provenance，下一步是同一 immutable K100，不得重跑或直接
  晋级 L2。Franco
  定为 Pod1 每卡 `4` 个我们的 trainer、Pod2 每卡 `3` 个，为 Yikang 的最多一张卡留动态余量。新增任务
  先跨六张可用 GPU 各放一条，再开始第二/第三轮，Pod1 才有第四轮。空槽只给已过前置门且有预注册
  早判的不同机制，不复制失败
  seed，也不拿未过动作安全门的任务凑数。
  signed-face exam bank 已过 E2：371 题 old/new replay 逐字节一致并发布新 bank/report；K100 paper
  已物化为 100 个唯一题、正反手各 50。generic checkpoint attestor 已进入 main，但 C3/D3 actual
  execution consumer 尚在收口，故 L2/judge/第二 seed/晋级仍阻断。
- **Franco focus：** 五种动作的用途、动作专属来球题族、空挥视觉锚点和横移终态站距语义；反手拉
  B/C 先补证，高点拍压作为第五动作，v12 只作后续 Jiayi 对照。
- **Jiayi focus：** v12/dang 路线与 planner-policy 契合；其候选必须在相同挡球专卷和厂商 MuJoCo 中
  与 Franco 主线对照，不能用录制版本号直接晋级。
- **Yikang focus：** Pod1 冲刺、历史 Gate3 谱系复核与独立 physics/reference oracle；现有证据不替代
  当前 179-D planner-policy tuple 的 vendor runtime 行为。
- **Codex 执行：** Pod2-only YAML 发射/里程碑早判 harness、单 seed 机制漏斗与失败槽替换；并行推进 exact
  动作 lineage、B 主选/C 后备的逐层证书、S0/M0 exact GMR 前置、C3/D3 零摩擦与 signed-face
  checkpoint execution contract 和 main 账本；
  每通过一层就合 main，不把多个未验层绑成一个大任务。

## 1. 当前一套训练是怎样完整跑起来的

动作、题目、观测、Reward、PPO 和判卷不是几根互不相关的配置轴。它们按下面的顺序组成一套
完整训练配方（setting）；其中任何一层改变，都会产生一套新配方。

1. **先出一道题。** 阶段 1 的题目规定用正手还是反手、来球速度，以及击球时球拍应达到的
   位置、速度、拍面方向和时刻。训练题与正式考试题分开。
2. **再给一段全身参考动作。** 当前正手和反手参考来自 `v4rg_runtime_order_v3` 动作对。它提供
   全身姿态和时序参照，不直接等于这道球题的答案；现役 Reward 只模仿其中的上半身。
3. **把“机器人现在怎样”和“本题要什么”一起给 policy。** 当前 179 维输入包括机器人状态、
   上一步动作、参考动作、目标球拍位置/速度/拍面、距击球时刻还有多久，以及正反手类型。
4. **policy 每 20 ms 输出一次动作。** 输出是 31 个目标关节位置，不是力矩或球拍轨迹；目标先过
   关节范围保护，再由 Isaac 的关节控制器执行。
5. **Isaac 同时推进 4096 个机器人。** 每个环境使用同一规则但不同采样；当前配方把 31 个关节
   摩擦设为零，以便和现有独立考卷合同对账。这不是标定后的真机物理。
6. **Reward 告诉 policy 哪种行为更好。** 一部分模仿上半身参考动作，另一部分分别检查击球时
   球拍的位置、速度和拍面；下半身可以按题调整，脚步正则与硬保护限制明显坏姿态。
7. **PPO 根据这些得分更新 policy。** 当前正式训练从随机初始化开始，计划 17000 次迭代；本页
   成绩卡考的是第 2000 次迭代保存的 checkpoint。
8. **训练外再判卷。** checkpoint 先进入独立 Python BankExam。质量晋级必须先有可信判分和
   跨初始化稳定性，再过厂商 `Gate3/Gate3B`；只检查运行链能否接通的 `Gate3-D0` 可并行推进，
   但不产生质量晋级。

一句话概括：**题目给出本拍答案，参考动作给出姿态和时序参照，179 维输入把两者交给 policy，
policy 输出 31 个关节目标，PPO 根据 Reward 改进它，独立考卷再检查实际行为。**

## 2. 课程、连续能力和部署验证是三条不同的线

### 2.1 课程阶段：机器人正在多学会哪一种球技

1. **阶段 1：固定点养成。** 固定站位、击球位置、目标落点和无旋条件，只改变来球速度。
2. **阶段 2：虚拟球的不同到达状态。** 球到达击球区域时的位置、速度、方向和高度开始变化；
   近球可以只用手臂，远球允许移动站位和脚步，也可调整整套动作的朝向、时序和拍面。
3. **阶段 3：物理球进入训练。** 球在仿真中真正从对面发出、飞行、弹台并与球拍接触；先做
   无旋或低旋，再加入旋转。旋转是否以后单列阶段 4，尚未决定。

所以，**阶段 2 大体是“适应不同到达状态”，“动起来”是阶段 2 的一种解法；阶段 3 的关键变化
是物理球真正进入训练。**

### 2.2 连续能力：每个课程阶段都要另考

阶段 1、2、3 都应分别报告单拍和连续成绩。连续能力必须在同一条不重置的序列中同时做到：

1. 上一拍结束后吸收身体的角动量和失衡，不摔、不滑、不撞桌；
2. 进入允许范围内的等待动作或准备姿态；
3. 下一球在任意有效时刻出现时，能在截止时间前启动合适动作。

组件消融只解释贡献。只收好上一拍、只模仿等待姿态，或只在固定时刻能启动下一拍，都不算完成。

### 2.3 部署验证：不是课程阶段

- Python BankExam 测的是 policy + Python 评估器 + BankExam 的 MuJoCo 机器人动力学配方，
  不包含 planner、消息、生产 C++ runner 或完整厂商运行时。
- `Gate3` 把 planner、消息、生产 C++ runner 和厂商 MuJoCo 串起来，先看整链能否稳定完成。
- `Gate3B` 复用同一完整运行链，再正式记录击球、回球和连续质量。
- 真机还要另过 [G07](gates/G07_mujoco_to_real.md) 的安全门。

## 3. 共用前提：先保证尺子可信

这不是课程阶段。所有阶段都依赖三个共同条件：

- **文件和接口对得上。** 当前动作、训练/考试题、179 维排列、checkpoint 和导出合同已经做
  内容绑定，不一致就拒绝继续；但合同一致只说明文件对上，不说明物理对上。
- **判分不骗人。** 2k/4k 的正手原始拍面误差多在 165–174°，旧解析判分器又会
  自动翻转法向；4k 已出现 parsed `48/50` 但 signed composite `0/50` 的同 checkpoint 反例。源码层
  `n/-n` 负控和有符号误差已通过；行为通过条件仍要求 fresh canary 和同一考卷重跑。详见
  [拍面判分复核](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- **机器人物理能外推。** 当前零关节摩擦只便于复现；历史非零摩擦数字存在单位/语义问题。
  通过条件是从真实数据拟合共享动力学参数，再分别用 Isaac、MuJoCo 适配器和厂商运行时验证。

当前只有第一项较完整；后两项未过。因此任何解析高分都不能直接称为物理回球或真机基线。

## 4. 阶段 1：固定点单拍基线（当前阶段）

### 4.1 目标和离理想还有多远

机器人站在固定位置，用指定正手或反手，在固定击球位置把不同速度的无旋来球打向固定落点。
它要在保持平衡的同时，让球拍在正确时刻达到正确位置、速度和朝向。

- **研究推进最低线：** 有效同卷中每个动作的解析回球率至少 50%，只用于判断训练管线是否值得
  继续扩题，不是部署质量线。
- **正式候选质量目标：** 跨独立初始化稳定达到每动作约 80%，同时拍面误差 p90 小于 15°、
  落点误差小于 0.3 m、零摔，并且判分尺可信。

现有结果只在平均解析回球率上越过最低线；初始化稳定性、拍面、落点毕业指标和连续能力都未过。

### 4.2 动作：身体应该怎样挥

**问题：** 只给球拍终点，机器人可能用不安全或不可连续的身体动作碰到目标；死跟参考动作，
又可能无法满足题目要求的拍面和拍速。

**当前解法：** 使用经过关节/刚体顺序整理的正反手动作对
（`v4rg_runtime_order_v3`）。触球时由题目给出的球拍目标主导，其余时段保留上半身动作模仿；
下半身不再被参考动作锁死。

**效果与差距：** 这套动作已进入当前训练和考卷合同，能产生可比较结果；但没有可信证据证明它
优于其他动作。Franco/v6/v7 新动作只完成离线安全和重定位检查；V5 专业动作有历史训练与诊断，
但没有在当前严格同卷下获胜。两类都未进入现行配方。

### 4.3 观测与输出：policy 知道什么、控制什么

**问题：** policy 看不到本题要求的拍面就无法按题调整；训练和部署的输入排列不同也无法迁移。

**当前解法：** 179 维输入在旧 175 维部署输入后增加目标拍面法向 3 维，并预留旋转参数 1 维；
输出固定为 31 个目标关节位置。严格导出器和加载器检查维度和来源。

**效果与差距：** 当前模型文件及负例已通过严格装载检查，但厂商运行链还没有当前 179 维模型的
首个有效控制周期；目标拍面通道的真实延迟/噪声也未与其他目标通道完全同处理。

### 4.4 Reward：每一组分数在教什么

**问题：** 只模仿动作可能“挥得像”却没达到题目要求；直接用旧解析回球分训练又可能鼓励 policy
钻拍面判分漏洞。

**当前解法：** 现役 Reward 分为四组：

- **回答这道球题：** 球拍位置匹配 `14`（宽度 `0.20 m`）、速度匹配 `10`
  （`1.0 m/s`）、拍面匹配 `5`（`0.3 rad`）；三个连续核的乘积另有击球奖金 `5`，不是三项
  过阈值后的二值分；接近目标的进度奖励为 `10`。
- **保持可用动作形态：** 只模仿上半身的参考位置、朝向和速度，各项权重 `1`；下半身模仿已解除，
  让腿可以按题调整，球拍手腕的朝向模仿也按现行开关释放。
- **等待和收拍：** 等待时接近准备状态的正奖励为 `2`，只在目标仍可原地够到时生效。
- **压制坏动作：** 包括脚滑/拖脚/过快、过度伸臂、腰部扭转、击球窗倾斜/角速度/脚速度/上下
  晃动、力矩饱和，以及直立、关节速度、动作变化、关节限位和异常接触等正则项；脚朝向惩罚由
  当前发射配方设为 `-0.3`。

此外，现役 `HOPEPingPongVirtualBall` 仍启用由**实际执行球拍状态**推演的解析 outcome Reward：
解析过网 `20`、解析落点 `30`、解析旋转 `5`。`vb_metrics_only=true` 不会关闭这三个 task 自带项；
它与本 task 已开启的 `virtual_ball=true` 是冗余 OR 条件。解析过网和落点都有完整合法回球前的稠密部分分，所以它们是训练
shaping，不是独立物理裁判。`physical_ball=true` 的 Phase-A engine-integrated 来球诊断目前才是
**只记指标、不进 Reward**：位置由 PhysX 积分，场馆 aero/table bounce 由代码驱动；它没有开启拍球冲量，
球会穿过机器人。减速入位、额外拍面引导和额外球拍引导当前都关闭。
目标关节范围裁剪和摔倒/跟踪终止另属硬保护，不靠其他 Reward 抵消。详见
[Reward 真值审计](experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md)。

**效果与差距：** 有的独立初始化达到 100/100，另一个只有 20/100。结果属于整套配方，不能归因
到某一个 Reward 项。拍面尺和稳定性解决前冻结 Reward；理想状态还需用单变量配对证明各项贡献，
并验证融合后不伤平衡和连续恢复。解除左侧非击球臂模仿、让它参与平衡的单-seed A0/A1 配对正在运行，
但尚无 paired milestone/同卷结果，不属于已采用 Reward；见
[非击球臂实验](experiments/non_striking_arm_imitation_ablation_20260713.md)。

### 4.5 时序与连续：一拍结束后怎么办

**问题：** 总从同一静止姿态开始，只会学到“摆好再挥”；动作结束就传送回起点，也学不会收拍。

**当前解法：** 每局 10 秒；25% 从站立等待开始，25% 从历史收拍状态开始；挥拍前随机等待
0–2 秒；动作循环时不传送机器人；目标有 2 个控制周期延迟，并加入白噪声和连续相关噪声。

连续实验按三层推进：`T0` 只在完整挥拍周期结束后换题；`T1` 在事件时刻揭示下一题，不传送、
不清历史且冻结 Reward；只有 `T1` 仍学不会时，`T2` 才增加平衡债和准备状态引导。

**效果与差距：** 当前训练见过更多起始状态，也能在完整动作后保留机器人状态；但 BankExam 每题
仍单独重置，没有连续成绩。旧机器预注册与上述次序不一致，在配置、校验器和操作文档同步前禁止
启动。详见[等待/恢复实验](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md)。

### 4.6 训练方法和机器人动力学

当前使用 `rsl_rl` 的 PPO、4096 个 Isaac 环境和 50 Hz 控制，从随机初始化训练。零关节摩擦只为
当前执行合同对账；训练配置虽打开来球轨迹真值仪，现有 Reward 和成绩卡都没有采用真实
球拍—球物理接触结果。Reward 中的过网/落台是 achieved 球拍状态驱动的解析推演，不能改名为物理结果。

原生 MuJoCo 微调是减少 Isaac→MuJoCo 迁移损失的候选训练引擎，不是 Gate3。现有候选仍有动作
保护、源码闭包、严格 JSON 和 MJCF 路径四个正确性缺口，没有可信训练环境、并行接口、PPO 冒烟
或微调结果，当前禁止合入。详见
[原生 MuJoCo 训练实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)。

### 4.7 当前成绩：Python BankExam 单拍解析诊断（不是 Gate3）

这张卡回答：**在每题重置的 Python MuJoCo 评估器中，机器人是否触发倾倒，以及击球时产生的
实际球拍状态被当前解析模型判为接触/合法落台的比例。** 它不等于球拍位置、速度、拍面三项全部
命中，也不回答完整部署链是否真的把球打回去。

实际链路是：四个独立初始化的第 2000 次迭代 checkpoint → 同一张 100 题考卷（正手 50、
反手 50，每题重置）→ Python 在 MuJoCo 中推进机器人 → 击球时读取球拍状态 → 解析模型推算球
结果。它虽然加载同一份厂商 MJCF 源文件，但使用不同的 Python runner 和 BankExam 动力学配方；
同一 XML 不等于 Gate3 运行时。

- **解析接触判定：** 拍中心距题目接触点小于 9.5 cm，且自动定向后的接近速度大于 0.3 m/s。
  自动定向正是当前拍面正负号风险所在。
- **解析合法落台推算：** 解析接触成立后，用数学接触冲量和 RK4 飞行推算球过网，并首次落在
  对面台内；全过程没有 MuJoCo 球碰撞。

| 动作 | 单拍：MuJoCo 机器人未触发倾倒 | 单拍：解析接触判定 | 单拍：解析合法落台推算 | 连续：未倾倒 | 连续：解析接触 | 连续：解析合法落台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | `200/200 = 100%` | `137/200 = 68.5%` | `133/200 = 66.5%` | **未测** | **未测** | **未测** |
| 反手 | `200/200 = 100%` | `170/200 = 85.0%` | `170/200 = 85.0%` | **未测** | **未测** | **未测** |

判读边界：

- “未触发倾倒”只检查 MuJoCo 中机器人根高度和倾角，没有球物理，更不是真机不摔。
- 四个初始化的解析合法落台数为 `83、100、100、20`；平均数掩盖最差初始化，稳定性结论为失败。
- 所有尝试最后都经过非倾倒的跟踪保护结束，不能据此证明收拍或连续稳定。
- 正手 66.5% 低于约 80% 的正式候选目标；反手平均 85% 也因最差初始化只有 40% 且拍面尺有问题
  不能晋级。当前也没有可信的拍面 p90、毕业用落点误差或连续成绩。

第 4000 次迭代的后续卷已完成：

| Seed | 解析合法落台 | 正手 / 反手 | 正手 signed composite | 物理 root fall |
| --- | ---: | ---: | ---: | ---: |
| 1 | `50/100` | `0/50` / `50/50` | `0/50` | 0 |
| 2 | `88/100` | `38/50` / `50/50` | `0/50` | 0 |
| 3 | `98/100` | `48/50` / `50/50` | `0/50` | 0 |
| 4 | `0/100` | `0/50` / `0/50` | 无正手 strike | 21 |

稳定门的 median/worst/spread/worst-side 全失败；详见
[稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。这张表同时说明为什么不能只看
旧解析落台数选 seed。

2026-07-13 的运行态是：16 条 fresh 广度臂已分两波全部按负责人运营决定停止。第二波前
重验最新 checkpoint 的迭代、76 tensor/`1,762,715` 浮点值 finite、schema-3、fresh lineage 和
相邻合同 SHA，且确认 24/24 最近 K20 格的正手 signed composite 都为 0。只向各自登记
PGID 发信号，无 broad kill/judge/真机命令；两 Pod GPU 已空。该动作不构成 q10 阈值正式判死或
setting 晋级；完整曲线、
checkpoint SHA 与信号边界见
[拍面×plant 广度矩阵](experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。

## 5. 阶段 2：虚拟球的不同到达状态

### 5.1 目标和实验结构

阶段 2 用虚拟来球分布生成题目，使球到达击球区域时的位置、速度、方向和高度变化，但只出当前
动作接口确实有可能完成的题。出题器和按题适配器是共同前置，不是一个实验臂。

正式比较应在同一批中做四臂：不额外适配的对照、站位钉死只用手臂补偿、站位随球平移并使用
脚步、整套动作绕击球锚点旋转。移动时序是正交消融，同批比较先移动后挥、边移动边挥和不规定
结构三种方案，而不是把这些方案预先排成固定串行课程。

### 5.2 动作与出题

**问题：** 直接放大目标框会生成动作根本做不到的题，也会把手臂、脚步、转身和重新定时混在一起。

**计划解法：** 用正向模拟生成不同来球的到达状态；每题先过动作接口可行性检查，再按题调整
整身朝向、时序和拍面。

**当前状态：** 整身旋转、时间律和动作静态安全筛已有部分工具。7 段私有新视频已精确登记；其中
高点拍压第五动作和左右横移四候选已通过 GVHMR 结构门，v12 未执行。既有 Franco 反手拉 B/C 的
signed 重定位产生 `19/3` 个 proposal，但还没有一个拿到 schema-2/L0/L1/桌网/动力学证书。
正式出题器、按题拍面/路径适配、站位与挥拍联合训练和留出卷均未完成，没有阶段 2 行为成绩。
详见[动作空间重定位实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)
和[新动作组合设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md)。

### 5.3 观测

当前 179 维输入已包含目标球拍位置、速度、拍面和击球倒计时，可以承接一部分变到达题。只有在
实验确认必须显式给出目标站位时，才考虑 181 维扩展；它不是当前配方。

### 5.4 Reward

正式对照使用同一份变到达题表，并冻结阶段 1 Reward、机器人动力学、训练预算和其他输入；只改变
预注册的手臂/站位/整身转向等动作补偿机制。减速入位只加入“站位随球移动”的对应臂；固定站位
臂不应得到同一项 Reward。

### 5.5 时序、连续和切真球条件

每个单拍臂都要另做不重置连续卷，检查上一拍收尾、等待和下一拍就绪是否一起成立。目前没有结果。

切到阶段 3 前，至少需要：训练内按全部挥拍机会计分的正反手解析回球率各 ≥50%、变到达状态
留出卷 ≥50%、虚拟预测与物理球落点差 <0.1 m，并且 `Gate3B` 评分链已建好。这里的训练内
回球率不是 `T1` 连续能力证据，两者必须分开报告。当前所有切换条件均未满足。

## 6. 阶段 3：物理球进入训练

### 6.1 目标

阶段 3 才让球从对面真实发出，在仿真中飞行、弹台并碰球拍；击球位置、预警时间和是否需要移动
由球的轨迹自然产生。阶段 2 先测清动作和站位边界，才能把阶段 3 的失败归因到球物理或接触。

### 6.2 动作与观测

动作适配和站位能力沿用阶段 2。179 维输入预留的旋转参数当前填零；先通过无旋物理球，再按
低旋 → 场馆旋转逐级加入旋转信息和随旋调整拍面/切向拍速的能力。

### 6.3 Reward 与物理成绩

先建立真实碰拍、过网和落台的物理结果，再决定落点、旋转等 Reward；不能继续用解析回球列冒充
物理训练信号。每个动作仍需分别报告单拍与连续的未倾倒、物理击球和物理上台。

### 6.4 当前状态

Isaac 的来球飞行真值仪和拍面接触源码已有材料，但拍面接触尚无被接受的 Pod 运行、训练结果、
完整物理成绩卡或跨引擎四格对账；球旋转传递也未完成实测标定。阶段 3 的行为训练尚未开始。

## 7. 部署验证线：当前卡在哪里

当前 179 维模型已经通过严格装载和负例拒绝。planner 与 C++ policy 之间的 formal tuple 现已把
shared epoch、command/base sequence、side、target 和最新 tick-start base 绑定到同一 actor gate；
exact 源码也已通过 portable Release。但这次构建明确关闭 ROS/AimRT，runner 没有执行，因此还没有
最终 ROS/Jazzy/AimRT 集成、厂商 MuJoCo 首个有效控制周期或固定考卷行为记录。当前 `Gate3-D0`
只要求先完成一份单拍全链演示，不冒充连续对打，详见
[Gate3-D0 实验](experiments/2026-07/EXP-GATE3-CURRENT179-D0.md)和
[exact build 卷宗](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md)。

2026-07-11 的 `13 PASS / 7 FAIL` 来自旧 110 维模型：3 次发球只有 1 次合法回球，出现 1 次摔倒
和明显漂移。它证明旧链曾执行，不是当前 179 维模型的 Gate3 结果。

## 统一工作队列（唯一优先级账本）

队列按主题组织，方括号内数字是全局执行顺序。人类责任人只能写人，Codex/Claude 只写执行者；
详细现状和命令留在对应实验记录。

### 动作主线

- **[1｜P0] Franco 五动作 + 横移老师。** 责任人 franco；执行者 Codex；下一证据：一次性物化反手拉
  B/C 的 schema-2/FK，并补 L0/L1/桌网/动力学证书；同时一次性完成已 finite 的高点拍压 S0 与横移 M0
  exact GMR；其 v2 Pod2 inspect 因绑定 runtime 与全部 ignored GMR/input 资产不存在而在 consumer 前拒绝，
  下一步先从权威备份恢复内容寻址资产并闭合独立 v3 runtime，再进入各自 schema-2。挡、拉、高点拍压
  各用自己的题族；先每个候选一个因果格，不把候选当
  seed 重复。只有离线证书通过后才分配 RL GPU。
  [旧动作实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)；
  [新动作设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md)

### 尺子与阶段 1

- **[2｜P0] 拍面正反与解析判分。** 责任人 franco；执行者 Codex；C2 的真实非零 plant 已证伪旧配对，
  D2 永久不发；下一证据是运行 main 上全新 C3/D3 显式零摩擦 L1 并形成 fresh paired receipt，然后用
  一个 seed 跑“热启动/从零 × 线性引导
  关/开”四个机制单元到相对 checkpoint
  `+200/+500/+1000`。只有胜者连同匹配对照才解锁第二 seed，不再给已失败配方复制四 seed。
  [量尺实验](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)；
  [机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)

### 部署验证

- **[3｜P0] 当前 179 维模型的 Gate3-D0 单拍全链。** 责任人 franco；执行者 Codex；exact source/build
  前置已闭合，下一证据是固定同卷完成 owned planner → C++ runner → 厂商 MuJoCo 的首个 no-publish
  有效周期和行为记录。[实验](experiments/2026-07/EXP-GATE3-CURRENT179-D0.md)
- **[7｜P1] Gate3 历史谱系复核。** 责任人 yikang；执行者 Codex/direct；分支
  `yikang-standhit-0714`。2026-07-15 已完成 mechanics 门并发射：以 W&B
  `ayzxv1ma/model_10600`（旧 Gate3 v4
  3× PASS、每轮 7/7 正手挥拍/恢复周期代理、0 摔；未测物理触球或落台）为唯一共同 warm-start，
  保持其 reward/观测/动作与 v12fix
  teacher 不变，运行三个因果分叉：纯 A=站姿可达击球点泛化、纯 B=每个 reset 独立
  Bernoulli `p=0.05` 随机推扰、A+B；另有同配方 fresh A+B origin 对照（禁止 checkpoint，保留
  原 yaw curriculum）。exact source `8c8cd53` 的 init/load 门与 B/AB 实际推扰门已过；Pod1 三卡
  正在运行 A [`5nso93g0`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/5nso93g0)、B
  [`4osh4ypc`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/4osh4ypc)、A+B
  [`jndof7jk`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/jndof7jk)，Pod2 GPU0 正在运行 fresh
  A+B [`xpiapvix`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/xpiapvix)；四条首个 checkpoint
  均已 finite/loadable。fresh 的短 mechanics 因初始策略 5 秒前摔倒只验证到 push selection，但正式
  run 到约 iter 322 已记录第一次真实 apply。下一证据：同绝对迭代 checkpoint 的泛化、推扰后存活与固定点回归；随后
  才做同运行链 Gate3 对照。启动健康不等于质量晋级；旧周期代理与球路泛化局限见
  [跨线审计](experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md)。2026-07-15 Codex 正在
  同一分支移植 `hitter@5f87a4f` 的 recipe-7/runtime-v2 训练、导出与部署合同；现役 V9 force
  作业保持原 exact source，不切 checkout、不重启。下一检查点是两项依赖提交完整合入、host 合同测试
  通过、空闲 GPU 复核，以及在独立 run namespace 发射一条 fresh RallyV10，而不是续训或替换现有矩阵。

### 训练引擎与机器人物理

- **[4｜P0] 原生 MuJoCo 训练候选。** 责任人 franco；执行者 Codex；下一证据：修正四个源码缺口
  后复核，再测单环境核心、并行吞吐和一次限预算 PPO 更新。[实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)

### 连续能力与后续接口
- **[5｜P1] 等待/恢复结构卷。** 责任人 franco；执行者 Codex；下一证据：同步机器合同后，用冻结
  Reward 跑 `T0/T1` 配对连续卷。[实验](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md)
- **[6｜P1] Hitter V3 规划器—policy 输入对齐。** 责任人 jiayi；执行者 direct；下一证据：旧观测
  排列、训练第 24100 次迭代 checkpoint 归属和第 7 版击球平面三项来源对齐。
- **[8｜P1] 110 维 RallyV10 左腕/恢复修复。** 责任人 dongc1；执行者 Codex；分支 `hitter`；
  实现已完成：从 RallyV9 冻结合同派生 V10，加入左腕参考 debt、修正反手 `tts=0.96` 边界，
  对齐 yaw-rate settle 与 whole-joint q_des 门；训练 host `87/87`、Gate3 report `12/12`。下一证据：
  dongc1 手动 fresh 训练后提交 deterministic eval、recipe-6 ONNX 和同 serve 表 Gate3；Codex 未启动训练。

队列排序与算力规则见[跑批作战手册](runbook.md#统一队列排序与算力纪律)。完整实验索引只在
[实验登记册](experiments/README.md)维护；本页不再复制实验索引或最近测试流水。
