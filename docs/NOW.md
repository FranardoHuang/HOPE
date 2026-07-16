# NOW — 当前训练流程、课程阶段与下一步

最近复核：2026-07-17 CST。本页说明现在到底在训什么、整套训练怎样连起来、每个课程阶段在解决
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

- **当前 task-revision 训练池：** formal 179-D 与训练现已统一为“一颗球一个 `task_id`、同球估计用递增
  `task_revision`”，挥拍中位置、速度、signed 拍面与剩余击球时间每个 policy tick 继续原子更新；phase
  governor 只接受可达的动作加速。准备时间不是以 `0.5 s` 为下限，而是同时采样 `<0.5 s` 压力、exact
  `0.5 s` 基线、`0.5–0.9 s` 快球和 `0.9–1.7 s` 宽分布。4096-env `A6` 已证明这些机制真实激活；随后
  22 个 delay-zero 格全部各发射一次。最终独立只读复核为 **19 条 `live_exact`、3 条首 iteration 前基础
  设施拒绝、0 条漏发**：Pod1 八条按 `3/3/2`，Pod2 十一条按 `3/4/4`；live 臂持续前进且未见 OOM/
  Traceback/Killed。两个 importer malloc `rc134` 与一个 boot stale-timeout 不算科学失败且不自动重跑；
  两个 positive-delay 格因 governor/actor transport 尚非同 tick 原子继续 NO-LAUNCH。0.5 秒 K100 paper
  已在两 Pod 物化但尚无 checkpoint 行为分数；现役 v4rg 正反手 TOPP 已在 Pod2 CPU-only 一次完成，
  两侧安全证书均通过，但当前搜索族只找到正手 `0.98 s`、反手 `0.78 s` 的可行 run-up 上界，没有
  0.5 秒动力学证书。这不证明 0.5 秒绝对不可能，却证明旧固定倍率不能冒充可行动作加速。
  `+200` 机制失活、`+500` 极端崩坏和 `+1000` 同父本容差 Pareto 的组合保护式淘汰 consumer
  已闭合：停臂必须同时有单臂行为 receipt 和同父本 portfolio receipt，且至少保留两条以及
  实际 exact-0.5 暴露与 broad 两类时间覆盖。修复 receipt-directory harness 后，Pod2 首个 write-side
  `+200` cycle 已对 9 条到档臂发布整数行为 receipt：quality 父本 6/6 的 revision/last-precontact/
  actor-visible 与 exact-0.5 暴露都激活，合法 stop 为 0；continuous 父本 3 条已取证、2 条仍等 checkpoint、
  1 条 infrastructure-terminal 排除。Pod1 最近只读为 4 条到档、4 条 live sibling 等待、2 条
  infrastructure-terminal 排除。这里零 stop 是机制正常的结论，不是默认让所有臂永生；行为优劣从
  `+500` 开始判。最新 Pod2 `+500` cycle 显示十一条 live 臂仍未到共同 `model_5000/5200`，故尚无合法
  `+500` 行为取证已在 Pod2 到档格上闭合：quality 父本六条 completion 为 `0.919–0.971`、virtual-return
  为 `0.278–0.395`，没有一条满足“连续两个窗口 completion<0.40”的崩坏门；ready 四项在本批日志中均为
  null，不能拿缺失量尺做淘汰。因此这轮合法 stop 仍为 0。最新 Pod2 只读动态复核为 11 条 live、
  1 条既有 importer malloc terminal，三卡 `3/4/4` 且利用率 `96–97%`；本轮 inspector 没有穿过实际
  checkpoint 制品路径，所以不把 `latest=null` 冒充“没有 +1000”。下一轮只用 reviewed
  `inspect-pruning-cycle --milestone-offset 1000` 闭合 checkpoint+behavior+portfolio 三联 receipt，之后
  才能合法 Pareto stop。自动 rolling 任务在本次 task-revision/TOPP 关键修复期间保持暂停；trainer 本身
  未因此停止。一条既有
  importer 失败继续排除且全程没有 signal。首个合格
  checkpoint 正在跑 K100，之后按 receipt 淘汰并把胜者送 vendor MuJoCo。详见
  [task-revision 卷宗](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

- **动作加速当前边界：** 完整旧 v4rg 只重定时得到 `0.98/0.78 s`，因此新增的 host-only
  ready-to-strike 生成器从动作第0帧零速准备态直接接入保真的触球前0.1秒。四元数符号跳变与
  joint-velocity 过度声明已经红队修复。Pod2 首次真实生成又在发布前暴露 source gate 假拒绝：正式
  v4rg 的完整 migration 三元组被当成额外字段，正反手均无候选、无 TOPP、无 GPU 行为；失败 namespace
  已冻结。修复版精确接受并逐位保留 canonical 三元组，仍拒绝残缺、坏值和未知字段，专项 `21 passed`。
  第二次新 namespace 已成功生成两条真实候选；production-FK TOPP hard gate 都通过，但正/反手可行
  run-up 上界为 `0.64/0.94 s`，仍没有 0.5 秒证书。下一轮已冻结 `6/12/17` 主梯、`9/14` 按需细化，
  并用两侧 minimax 区分 forehand/backhand frame0 哪个可作共同 ready。端点实测已显示 `d=17` 全面
  慢于 `d=6`；反手 own-ready 把 raw `0.94` 改善到 `0.78 s`，但正手自己的 ready 仍更好，故冻结规则
  选择两个 ready 的四个 `d=12` 中点。红队发现一次性 runner 未持久绑定 consumer 且 certificate parser
  不够严格后，独立 historical attestor 已全量重验并发布唯一 receipt
  `7cf1c7c9…c377f`：六个端点升级为可信的 screening evidence，分别为正手 `1.28/0.70/1.54 s`、反手
  `1.94/0.78/1.42 s`，仍全部高于 `0.5 s`。四个 `d=12` 中点的 v1 execute 中 generator 全部成功，
  但量尺把 candidate float32 producer-gradient 混成 TOPP float64 workspace-gradient，故四格在 TOPP 前
  fail closed；旧 summary `f92e6b8b…63c0e` 已冻结且不重放。v2 只修该量尺，绑定旧失败并使用唯一新
  CPU-only namespace，动作/join/预算/acceptance 不变。v2 唯一 execute 已证明四份 candidate 与 v1
  逐字节一致，随后因隔离 MJCF XML 漏复制其引用的 STL，四个 TOPP 都在算法前 missing-mesh rc1；
  summary `6910db28…f1476` 冻结且不得重放。v3 直接复验并复用这四份 candidate，零 generator 调用，
  唯一改变是从 frozen runtime Git objects 补齐 `1 XML + 74 mesh` 的 exact closure（75 文件、
  14,127,373 字节、manifest `e0381752…b962de`）。但 v3 唯一 Pod2 dry-run 在结果 root 前抓到 v1/v2
  contract SHA 账本混用，未启动 execute/TOPP。v4 用新 activation/namespace 绑定 v2 四份实际 contract，
  随后 dry-run 又发现 exact log SHA 后的英文文本猜测会假拒绝真实日志，同样未启动 execute/TOPP。v5
  删除重复文本解释、只保留 exact bytes，其他科学配方完全不变；本地相关回归 `70 passed`、独立红队 GO。
  在真实结果出现前仍没有
  production FK、TOPP≤0.5、L0/L1、桌网、动力学或行为通过，不能写成0.5秒动作已完成。见
  [短路径实验](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- **已停止的前代 rolling 池（只作背景）：** 旧 `24/24` 长曲线已按绑定身份停止，三份较强母本的 optimizer 被完整恢复到
  [24 格快速准备组合漏斗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。2026-07-16
  11:29 CST 的每 Pod 单连接只读审计确认 `24` 条都已消费唯一 claim，其中 `22` 条 live、fatal=`0`；两条在首迭代前由动态 URDF importer
  以 malloc `rc134` 退出，精确 PID/PGID 与 GPU context 均 absent，记为**基础设施拒绝**而不是 Reward
  失败，且不自动重跑。Pod1/Pod2 分别为三卡 `4/3/4` 与 `4/4/3`；GPU 利用率约
  `90–98% / 89–97%`，显存约 `17–23 GiB/卡`，主机可用内存约 `907/916 GiB`、swap=`0`。本轮比较约
  `1.0/0.7/0.5 s` 与随机准备时间、时间戳补偿、预测抖动、Reward 配比和脚朝向；全部是单 seed、热启动、
  demo-only 的工程候选，不能冒充正式因果结论。每 30 分钟继续检查结构/finite/恢复和 checkpoint；但
  11:29 CST 的量尺审计推翻了“现役 source 可自动执行 `+500/+1000` 行为淘汰”的假设：completion 与
  pre/post-fall 是跨历史 EMA，termination 混有非物理 guard，dense ready/balance 又缺 phase 分母与母本
  receipt，无法重建两个互不重叠的 100-update 窗。因此现役 22 条只能结构淘汰，行为状态统一为
  “量尺不完整，继续训练”；稀疏击球零值仍绝不算失败。12:45 CST 的 Pod1 单连接刷新仍是
  `11 live + 1 importer rejected`、fatal=`0`，11 份 latest `model_2100–2600` 的 embedded iteration、finite、
  hard/claim/binding 与 lineage 全过；旧 budget-v1 诊断臂现到 `model_2600`，`model_3600` 尚不存在。
  13:46 CST 的 Pod2 唯一只读 inspector 已闭合根因：第一份 `model_5200` 的 actual claim=`7878d92...`，
  自洽绑定修正 budget 的旧 runner=`428cbf...`；与现行 `aee7132.../90d7f26...` 的完整 content 唯一差异是
  `continuation_runner_script_sha256`，budget/题目/source/run/slot 全相同，绑定进程仍 `live_exact`、checkpoint
  为 regular、receipt absent。这不是 trainer 或 checkpoint 失败，也不授权直接重试。milestone consumer
  只精确白名单两份逐 job 完整重建的 corrected-budget claim，明确排除旧 budget-v1；任何第三 runner 或其他
  字段差异仍 fail closed。约 15:10 CST，reviewed consumer 对该 job 只连 Pod2 一次并成功 O_EXCL 发布首份
  `model_5200` receipt：filename/embedded=`5200`，74 个浮点 tensor、`1,762,715` 元素全部 finite，schema-3
  hard/claim/binding 对齐，lineage=`0`，取证时进程仍 live；receipt content SHA=`521910d...`。这只关闭
  checkpoint 完整性，不是行为领先。15:36 CST，第二个 job 的独立 dry-run 正确生成自身 per-job claim
  `691a52c.../0968d24...`（仍绑定 reviewed runner `428cbf.../90d7f...`），唯一 Pod2 SSH 随后也成功发布
  `model_5200` receipt：content SHA=`37d6bd2...`、checkpoint=`ff1b210...`、hard=`aa80162...`、binding=
  `7593d66...`，同为 embedded=`5200`、全部 finite、lineage=`0`、process live；第一份 receipt 未触碰。
  两条都继续、不排名、不停止。15:50 CST 的 Pod1 唯一只读连接仍为 `11 live_exact + 1 importer
  rejected`，GPU=`4/3/4`、util=`84/84/96%`、accepted fatal=`0`；11 份 latest checkpoint 均
  embedded/finite/schema-3 hard/claim/binding/lineage=`0` 且 optimizer 完整。budget-v1 PGID `2199057`
  仍 live、latest=`3300`、`model_3600` 不存在，未 signal；Pod2 同轮为 `11 live_exact + 1 importer
  rejected`、GPU=`4/4/3`、fatal=`0`，accepted checkpoints 全部通过同一完整性门。后续 source 必须增加 per-update 整数机会/
  完成/物理跌倒 union 与 ready-phase `sum+count`，现役模型若要提前排序只能另走绑定 checkpoint 的不可变
  同卷评估。`qdot` 是确定性的关节速度超限惩罚，不是随机外力；随机横向躯干推力正在
  补 trainer/物理响应门，过门后才占释放槽做同母本 no-force/force 配对。

- **V10 现场问题复核后的采用状态：** 旧 formal 179-D runner 的 active-swing target/TTS freeze、同球重复开
  task、VRPN capture age 未补偿和 planner/runner 盲盒 trace 已被 task-revision 协议取代：一颗物理球固定
  一个 `task_id`，触球前每个 policy tick 用递增 `task_revision` 原子更新目标、signed 拍面与剩余时间，
  capture age 进入外推，低频 trace 关联 ball-plane distance、source age、intercept、revision 与接受/拒绝
  原因。训练 actor 同样逐 tick 看同球修订，题库/Reward truth/critic 保持 immutable，避免把追踪噪声写成
  答案。尚未闭合的是行为层：Stage-1 来球位置仍不等于场馆 residual，positive-delay transport 仍
  NO-LAUNCH，0.5 秒 K100 和 vendor MuJoCo 尚未给出能否回球的成绩。详细边界见
  [G07 的八项现场审计](gates/G07_mujoco_to_real.md#audit-update-2026-07-16-rallyv10-field-test-timing-and-task-lifecycle-gaps)。

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
  SHA-256 `6840df34...db60`。下一张整轨桌网门也已在 clean Pod2 `main@c047ea7` 完成唯一 v4 audit：
  `1201 @ 400 Hz` 逐帧 `37×4` 对，hard/warning/unsafe=`0/0/0`；certificate `93fd5435...9b0e7`
  只声明诚实 saturated lower `0.099999999999 m`，pair/midpoint/time=null，并经独立复核接受。B 现只解锁
  vendor 动力学/平衡门；RL、回台、Gate3 和真机仍未授权，C 保持后备。
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
  exact PGID=`380610/381237` 已停止。2026-07-15 04:15 UTC 的只读快照中，Pod2 GPU0/1 分别仍是
  Yikang V9/V10（PID `379550/396374`），Codex 没有触碰；GPU2 已运行三条同源 10000-update 长曲线：
  关节速度边界惩罚 PGID `411519`、击球窗模仿放松 PGID `412204`、普通对照唯一重试 PGID `412899`。
  三条分别到 iteration `24/9/2`，fatal=`0/0/0`，claim/binding 都存在；GPU2 为 `97%`、17154 MiB。
  200/500/1000 只早筛，2000/3000 看中段，6000/10000 才看完整曲线，不买第二 seed。任何 Yikang
  进程仍在的 GPU 都保持保留。下一次同轴训练必须先用
  两臂共享、自然-wrap provenance 绑定的 immutable teacher-state receipt 做外生 cold-start，并在首 update
  前 fail closed。

  2026-07-15 15:49 UTC 的更新取代上面的 04:15 资源快照：Yikang 在 Pod2 GPU0/GPU1 的 compute PID
  已自然为空，因此 Codex 按 GPU0→GPU1 逐圈各一条补齐六格 10000-update 单 seed 长曲线。V1-only、
  `qdot=-1`、V2-only 唯一 retry、`qdot=-2.5`、脚部朝向惩罚 `0/-0.6` 的 exact PGID 分别为
  `419643/420298/423502/421479/422126/422783`；连同 GPU2 原三条，Pod2 三卡各恰三条 trainer，
  利用率 `97%/97%/91%`，九条 live 日志 fatal0。V2-only attempt-1 PGID `420947` 在首迭代前的动态
  URDF import 以 malloc rc134 退出，无 checkpoint；证据保全且 namespace 永不复用，不是 Reward 失败。
  这六格补齐“单独/组合放松动作模仿”、关节速度和脚部朝向三个因果问题，不复制 seed。200/500/1000
  只判结构；只有真实击球后才有意义的 Reward 若 eligible hit 样本不足必须继续，6000/10000 才看完整
  单 seed 曲线。[六格实验](experiments/2026-07/EXP-P1-LONG-SCALEOUT-SIX-ARM.md)

  2026-07-15 16:40 UTC，Pod1 也已重新授权并按 GPU0→GPU1→GPU2 四圈铺满 12 条不同问题的
  10000-update 单 seed 长曲线：非击球臂是否继续模仿 × 10/16/24 秒连续 episode，以及六种击球
  位置/速度/拍面 Reward 配比。两个首发 namespace 在动态 URDF import 的 180 秒 stale 门失败，均为
  0 iteration/0 checkpoint；各自唯一同配方 retry 已过首迭代。现为三卡各四条、利用率
  `97%/93%/97%`，12 条接受臂 PID=PGID、fatal0。Pod2 同时保持九条三卡满池；两 Pod 合计 21 条。
  当前只证明生产池已铺满，不把早期稀疏回球零值判成失败。
  [Pod1 十二格实验](experiments/2026-07/EXP-P1-POD1-LONG-BALANCE-REWARD-GRID.md)

  clean main-effect 也已自然终档：两臂关闭 post-swing replay、固定 V1+V2，只比较 base-decel `0/1`。
  `model_1000` 两份 filename=embedded、finite、fresh lineage、claim 与共同 hard contract exact，原
  PGID `385320/385948` 均已退出。980–1000 的 treatment/control raw base speed=`1.00882×`，按冻结
  `≤0.90×` 门正式 reject，不买第二 seed/judge。源码复核同时发现这个 Reward 实际追踪随目标距离变化的
  `v_des`，并非始终让 raw speed 更低；尾窗 raw-kernel-per-eligible 反而提升 `1.6003×`。因此不改写
  verdict，但后续必须先用 `|v_base-v_des|` 的近/中/远分桶重做量尺，不能复制当前 weight=`1`。

  连续恢复当前最短关键路径已完成共享外生 teacher 制品：v3 在 Pod2 GPU2 自然收满 4096 条
  `natural_clip_wrap` state；唯一授权 attestor attempt-2 从固定 `a38b7e9` rc0，发布 4103-byte receipt
  `e20a6989...d2aba4`。merged-main controller status 已确认 `teacher_receipt_binding_exact=true`。当前只差
  4096-env、首个 PPO rollout 前的 first-reset 采用率与 simulator root/joint readback probe；它未通过前，
  科学 pair、第二 seed 与 judge 继续 fail closed。Codex 唯一可用卡仍是 Pod2 GPU2，当前三槽已满。

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
  [跨线审计](experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md)。2026-07-15 已在同一
  分支完成 `hitter@5f87a4f` 的完整 recipe-7/runtime-v2 移植并推送 exact merge
  `8e0aaf885dc553ad727a19d1f106b25e98afa866`；现役 V9 force 作业保持原 exact source，不切
  checkout、不重启。合并后的 Python host 门、本机/Pod2 合同回归、C++ focused/runtime policy
  smoke 与 32-env×2-update 真 Isaac fresh smoke 均通过。独立 RallyV10 正式 run 已在 Pod2 GPU1
  以 8000 envs/seed0/10000 updates 发射：W&B
  [`qpl08mug`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/qpl08mug)，110-D actor/328-D critic，
  未加载 checkpoint。首个 progressed `model_100.pt` 已通过 embedded iteration、optimizer 与全部
  tensor finite 审计；下一证据是 deterministic core/planner/mix 选择、recipe-7 ONNX、MuJoCo 与
  V10 Gate3，不以启动曲线直接晋级。
- **[OPS｜P1] RallyV10 真机单命令部署与板载 VRPN。** 责任人 dongc1；执行者 Codex；分支
  `hitter`。把 CMTracker/VRPN、relay 与 planner 收敛到 HDU，Laptop 只保留完整 SSH/TTY
  入口，并为 MDU HAL、SHADOW、真机 runner 提供互不自动串联的单命令脚本。下一证据：脚本
  静态/打包检查、HDU preflight，以及 `run_pingpong_end_to_end.md` 9.9 的可复现终端矩阵；不由
  Codex 执行真机 MOTION。

### 训练引擎与机器人物理

- **[10｜P0] RallyV11 TOPP 加速动作 fresh 训练。** 责任人 yikang；执行者 Codex；分支
  `codex/rally-v11-topp-prestrike-20260716`（当前 baseline 分支
  `codex/rally-v11-topp-fast-20260716`），基于 `hitter`。2026-07-16 真机复核发现现役挥拍动作
  速度不足，当前优先级切换为：用 `main` 已验证的 TOPP v3 对 v12fix 正反手动作做触球锁窗的
  min-time 重定时，新建严格 hash/phase/receipt 绑定的 RallyV11 fast task，保持 station-relative
  击球平面 `0.58 m` 与 RallyV11 全身/下半身 Reward 合同。host gate、32-env smoke、8000-env
  显存探针与在线 W&B checkpoint/ONNX save smoke 已通过；commit `edf9e4e` 的唯一正式 run
  已在 GPU0 fresh 发射 seed 0：
  `num_envs=8000`、`max_iterations=20000`、`num_steps_per_env=24`；其余两张卡明确留给其他人。
  W&B [`i0jw4ohr`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/i0jw4ohr) 已越过
  `model_0` 保存并持续推进（首份看板快照到 iteration 49）；GPU1/GPU2 无 compute process。
  用户随后把优化目标收紧为“尽可能缩短第 0 帧到击球帧的时间”，而不是最短整段时长；新分支将
  保持触球关键窗与击球后时序，把全部压缩预算优先投到击球前，并按最快 MuJoCo-oracle PASS 候选
  重新绑定 receipt/phase/hash。当前 run 保留作 baseline，在新动作通过 audit/oracle/Isaac smoke
  前不停止；“已启动”仍不等于性能改善，后续只按 checkpoint 与冻结评估证据判断。
- **[4｜P0] 原生 MuJoCo 训练候选。** 责任人 franco；执行者 Codex；下一证据：修正四个源码缺口
  后复核，再测单环境核心、并行吞吐和一次限预算 PPO 更新。[实验](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)
- **[9｜P0] Hitter 实机 planner 时序与训练反应时间基线。** 责任人 yikang；执行者 Codex；分支
  `codex/hitter-lowerbody-mujoco-alignment`，基于 `hitter`。2026-07-16 用户将本轮顺序收敛为：
  先审计最新 `main`、`hitter` 与 frame0-wait-v2 已有实现，再补齐端到端球样本时间戳/短 TTS、
  planner 求解限频与回调阻塞隔离、逐球结构化可观测日志，以及训练来球时间分布与最大反应时间
  监控。原下半身 Reward/接口对齐材料保留但暂缓扩展，其余连续挥拍、可行性、mid-swing/TOPP
  等项进入后续 TODO。下一证据是可复现的 remote-reuse 台账、schema/runner/planner 单测、离线
  timing replay 与训练指标测试；这些证据完成前不启动长跑 PPO、不执行真机 MOTION。

### 连续能力与后续接口
- **[5｜P1] 等待/恢复结构卷。** 责任人 franco；执行者 Codex；下一证据：同步机器合同后，用冻结
  Reward 跑 `T0/T1` 配对连续卷。[实验](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md)
- **[6｜P1] Hitter V3 规划器—policy 输入对齐。** 责任人 jiayi；执行者 direct；下一证据：旧观测
  排列、训练第 24100 次迭代 checkpoint 归属和第 7 版击球平面三项来源对齐。
- **[8｜P1] 110 维 RallyV10 左腕/恢复修复。** 责任人 dongc1；执行者 Codex；分支 `hitter`；
  V10 fresh run 的训练曲线已显示击球精度成熟，但 yaw-rate 约 `0.46-0.52 rad/s`、左腕超差率约
  `0.42-0.54`，尚未证明修复通过。当前工作：在不改 V10 基线的前提下增加独立
  constrained-resume 任务，从当前 V10 checkpoint 严格续训，固定最终 planner/yaw 分布，对左腕
  position+velocity 做全阶段独立模仿并收紧 yaw-rate。下一证据：host 合同门、Isaac mechanics
  preflight 与用户手动发射的 resume run；Codex 不停止、不启动 PPO。

队列排序与算力规则见[跑批作战手册](runbook.md#统一队列排序与算力纪律)。完整实验索引只在
[实验登记册](experiments/README.md)维护；本页不再复制实验索引或最近测试流水。
