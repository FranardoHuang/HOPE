# 实验登记册

实验用来回答可证伪的问题。本目录管理假设、冻结变量、运行记录、结果以及采用/拒绝决定。
`NOW.md` 只链接每个特性的最新有效实验，不重复实验细节。
本目录的共享缩写统一按 [术语与人话对照](../DEFINITIONS.md) 解释。

## 必须按职责路由

| 信息 | 权威位置 |
| --- | --- |
| 当前采用的 setting 与阶段/特性状态 | [`NOW.md`](../NOW.md) |
| 单个实验的设计、运行、证据和决定 | 本目录 |
| 已合入 `main` 的重要能力/修复 | [`TIMELINE.md`](../TIMELINE.md) |
| 可复现的验收与 gate 状态 | [`gates/`](../gates/) |
| 命令与操作流程 | [`operations/`](../operations/) |
| 历史原始流水记录 | [`archive/`](archive/README.md) |

一项事实只在一处详细记录；其他文件只写一句摘要并链接到该记录。

## 状态与证据等级

实验状态：`proposed`（已提出）、`preregistered`（已预注册）、`ready`（输入齐、可启动）、
`running`（目标实验本身正在执行）、`completed`（已完成）、`invalidated`（结果无效）、
`blocked`（缺关键前置，不能启动/继续）、`superseded`（被新实验取代）。

索引“状态”列只填上述一个值；`prepared_not_started`、off-main、preflight、forensic 等属于
runtime/feature 细节，写进实验正文或“决定”列。全局 P0/P1 只看 NOW，不在实验登记册复制优先级。

证据等级：`E0` 设计；`E1` 源码/单元/静态检查；`E2` 运行时冒烟或模型加载；`E3` 受控训练；
`E4` 留出仿真器/Gate3 考卷；`E5` 真机。始终只记录实际达到的最高等级；不得用大量低等级测试推断高等级证据。

## 责任归属

- `人类负责人`：必填，且必须是具体的人。
- `执行者`：可选；使用 `direct`、`Claude`、`Codex` 或明确的组合。
- `复核/决策负责人`：需要时必须是具体的人。
- Git 作者、Claude 或 Codex 不会自动成为责任人。

## 当前索引

| ID | 问题 | 人类负责人 | 执行者 | 状态 | 证据 | 决定 |
| --- | --- | --- | --- | --- | --- | --- |
| [EXP-ACTION-BALL-PHASED-READINESS-20260730](2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md) | 首个 N1、1000 update、formal N5、N73 与部署各自最迟必须闭合哪些训练、观测、吞吐和安全合同？ | Franco | Codex | running | E2 | 三条 compatibility 194-D stable-ready `4096×1001` 已在 Pod1 运行；fresh N1 改为固定 194-D v2，以 teacher-start 替换常量 one-hot，尚缺 Pod 构造 smoke。formal N5/N73 在 fixed-width continuous future-motion intent 前 fail closed；MuJoCo/C++ 尚不支持 v2，不得部署 |
| [EXP-ROUGH-GROUND-FRICTION-FIX-20260729](2026-07/EXP-ROUGH-GROUND-FRICTION-FIX-20260729.md) | 能否在不移动桌子/动作坐标的前提下，用每环境零均值地垫和物理一致摩擦训练抬脚？ | Franco | Claude、Codex | blocked | E1 | host `379 passed`；fresh N5 首轮仍用平地/no-move，rough/move 须补 Isaac clone、接触、raycast、seed/mesh、初始穿插与 4096-env 性能门 |
| [EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728](2026-07/EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md) | 去掉旧正手、三反手独占一张卡并从 N4 热启动，能否提速且不触 physical-hard 关节包络？ | Franco | Codex | blocked | E1 | N3 bank/父本 pin/175D safe leaf 与 smoke-only launcher 已备；等待 clean commit、Pod Hydra/Isaac substep safety smoke 和小预算 canary，禁止长跑/真机 |
| [EXP-ACTION-CONDITIONED-BALL-FIRST-20260727](2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md) | 能否先冻结动作、采该动作来球，再 fixed-action 解 task，并把联合 safe-policy failure 调到 10%？ | Franco | Codex | blocked | E1 | manifest/sampler/curriculum/receipt host 合同在整合；正式 N5/N73 continuous intent、motion admission、新正手行为门、Pod smoke 与 frozen evaluator authority 仍缺 |
| [EXP-TASK-FIRST-N-ACTION-20260727](2026-07/EXP-TASK-FIRST-N-ACTION-20260727.md) | 每个动作能否从中心 task 开始，按位置→标量速度→拍面→base 独立泛化？ | Franco | Codex | superseded/ablation-only | E1 | 先采 task 不能保证存在匹配来球；保留为历史因果消融，不再是候选 executor |
| [EXP-ACTION-CAPABILITY-SELECTOR-20260727](2026-07/EXP-ACTION-CAPABILITY-SELECTOR-20260727.md) | 任意击球目标能否先过安全与支持域，再按成功率 LCB 和并列优先级选动作？ | Franco | Codex | blocked | E1 | pure-core 排序/receipt 候选不等于生产接线；能力卷、trusted producer、任意 N wire/C++ 与 Gate3 均缺 |
| [EXP-EFFECTIVE-REWARD-CAUSALITY-20260727](2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md) | 当前改善来自 task 自洽，还是名义高 Reward 权重？ | Franco | Codex | preregistered | E1 | 已证实实际 `4/0.5/0.5` 覆盖名义 `393.4/295.1/229.5`；task 自洽是当前最强解释，仍需同 task 分布的两 seed paired A/B |
| [EXP-MOTION-CANONICAL-LIBRARY-20260723](2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md) | 五动作 × upper/full 十件主资产怎样正式化，Franco 动作和全身学习是否分别有效？ | UNASSIGNED | Codex | blocked | E1/E2 probe | 十件按 direct shared-ready 路径重建，窗口允许持续加速，`adv2c3` 仅作历史比较；grounded torque/contact、行为/恢复和 registry adoption 未闭合，仍为 `0/10` training-authorized |
| [EXP-P1-INTEGRATED-UPGRADE-WAVE](2026-07/EXP-P1-INTEGRATED-UPGRADE-WAVE.md) | AR/action-acc/qbar/plant/push/脚部项的未测组合能否兼容工作，而不是哪一单项有效？ | Franco | Claude | blocked | E1 | action_acc 已接线但无行为证据；combo 仍绑定过时 B/strike phase，CGF 未发、交互未测；只认 Franco 明示解冻，不作单项归因 |
| [EXP-P1-CHATTER-GROUND-FOOT-WAVE](2026-07/EXP-P1-CHATTER-GROUND-FOOT-WAVE.md) | action_rate 剂量、地面摩擦/不平整覆盖、脚部落地塑形、软惩罚减负——哪些能降抖动/失稳而不伤击球？ | Franco | Claude | preregistered | E1 | 8 臂三件套+单测已落盘；渲染被占位 commit + groundfoot/kdpassive 双闸门锁死；对照=矩阵 w_c_s0 不重复买；rough 臂 fresh-from-random 铁律 |
| [EXP-P1-INTEL-WAVE-20260721](2026-07/EXP-P1-INTEL-WAVE-20260721.md) | 速度泛化混合/强 q_des 平滑/腿姿模仿方案包/全关节 qdes barrier 四条情报哪条值得进正式配方？ | Franco | Claude | inconclusive | E3 diagnostic | 多臂曾运行且部分续训；`v_qbar@10700` 到档，其余状态按 PROGRESS 对账；fullbody 同时改权重和窗宽，不能归因“全身学习” |
| [EXP-P1-PUSH-ROBUSTNESS-20260721](2026-07/EXP-P1-PUSH-ROBUSTNESS-20260721.md) | 随机推撞（速度推/同冲量力推）能否买到抗扰平衡？ | Franco | Codex | running | E3 | （07-22 补索引）18 臂扩容、12 条已上卡待判；pod 失联期间账面以 07-21 为准 |
| [EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720](2026-07/EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md) | 时序平滑（N/C/H）与稳定机制（S0 对照、S1 挥拍后安顿债务、S2 支撑包、S3 下肢软模仿）哪个组合能在高摔倒 parent 上降摔且不伤击球？ | Franco | Claude | running/closing | E3 | 24/24 已于 2026-07-20 发射；实测同卡算力分时后筛选终点前移到 +4000，收口进行中（S0/C 行 10 格已到终档）；动态清退（N 交互 6 格永久停、8 格暂停待续）以两 pod `scheduling_ledger_20260720.jsonl` 为准；取代 Wave A 科学位与 Wave B 六格 |
| [EXP-P1-LOWER-BODY-STABILITY-20260720](2026-07/EXP-P1-LOWER-BODY-STABILITY-20260720.md) | 静态下肢软模仿还是无参考支撑/腿速约束更能改善稳定且不伤击球？ | Franco | Codex | superseded | E1 | B1/B2 机制以 S3/S2 档并入 24 格矩阵（EXP-P1-BALANCE-TEMPORAL-MATRIX）；六格队列不再单独发射 |
| [EXP-MOTION-READY-TO-STRIKE-0P5](2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md) | 旧 two-ready/quintic-join 家族能否在不放松限制时接入保真窗并达到0.5秒？ | Franco | Codex | completed/rejected-family | E2 probe | stage2 v8 `0/4` 达标，旧家族按预注册停止；不能外推否定新的绝对 bridge，也不授权训练 |
| [EXP-P1-TASK-REVISION-CUTOVER](2026-07/EXP-P1-TASK-REVISION-CUTOVER.md) | 同一颗球能否在挥拍中实时修订 target/TTS，同时保持 exactly-once、宽准备时间、可达加速和可淘汰量尺？ | Franco | Codex | completed | E2 runtime | A6 机制门通过；22 格均已收口，19 格到终档、3 格为首迭代前基础设施拒绝而非科学负例；旧 ready/balance 分母为零，未产生行为胜者 |
| [EXP-P1-TIMING-EXAM-0P5](2026-07/EXP-P1-TIMING-EXAM-0P5.md) | 同一不可变双侧题表能否验证策略在仅 0.5 秒准备时间内真实完成击球？ | Franco | Codex | paper-ready/unknown | E2 artifact | 两 Pod 已物化同一 K100 paper；尚无 checkpoint 行为分数、TOPP 绑定卷或 vendor MuJoCo 结果 |
| [EXP-P1-HALF-SECOND-SPRINT](2026-07/EXP-P1-HALF-SECOND-SPRINT.md) | 从共同 model_5700 派生的二十三个单-seed问题能否找到半秒击球候选，并解释速度/平衡取舍？ | Franco | Codex | completed | E3 | [五臂](../DEFINITIONS.md#half-second-sprint-arms)全池已结束；W/Y 是稳定诊断前沿、V/X 是高回台高摔倒前沿；action-rate 证据已留存，仍无 vendor 行为结论 |
| [EXP-P1-BALANCE-ACTION-SLEW-20260720](2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md) | 腿腰恢复期 processed-q_des 突变铰链能否比全身 raw-action dense 平滑更好地降摔而不伤击球？ | Franco | Codex | preregistered | E3 mechanics / E1 result | [Wave A/B](../DEFINITIONS.md#balance-stability-waves)：probe9/v8 六收据齐全，但 authority `d5c08bb9` 的首个科学 W-N 在 `sim.reset` stale `180 s`，locked=`125`；stage-blind failure audit 误读 probe-only child evidence 后 outer=`121`。无 iteration/RSL/checkpoint/Reward，其他五格未发，v8 frozen，不作 N 机制结论。fresh v9/probe10 stage-aware source 已预注册，保持 W-N/GPU0→W-C/GPU1 顺序；六份 fresh receipt 齐全前禁止生成长训命令。结论仍 `inconclusive/not adopted` |
| [EXP-P1-FACE-PLANT-SCALEOUT](2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md) | 拍面×plant 广度矩阵哪些方向值得继续购买迭代？ | franco | Codex | completed/rejected | E4（诊断） | 16 臂已全部保留证据并停止；24/24 最近格的正手 signed composite=0，旧矩阵不能选 baseline |
| [EXP-P1-FRESH-SZ-STABILITY](2026-07/EXP-P1-FRESH-SZ-STABILITY.md) | 最接近正式 setting 的方案在不同 seed/checkpoint 间是否稳定？ | franco | Codex | completed/rejected | model-2000/4000：E4 diagnostic | 2k 与 4k 稳定性都失败；seed4 持续弱，旧 parsed 正手分被 signed-face 反例推翻 |
| [EXP-P1-FACE-SIGN-FORENSIC](2026-07/EXP-P1-FACE-SIGN-FORENSIC.md) | 高解析上台率是否隐去了拍面反号？ | franco | Codex | running | E4（旧卷诊断）+ E1（新源码） | `n/-n`/physical-B 源码门已实现；fresh canary 和修正后同卷未跑，旧分不晋级 |
| [EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715](2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md) | planner 给目标后，现役 Reward 是否还检查 achieved 击球能否上台？ | Franco | Codex | completed | E1 | 会用 achieved 球拍状态解析推演过网/落台并给分；Phase-A engine-integrated ball 不进 Reward，当前也没有拍球冲量 |
| [EXP-P1-SPARSE-REWARD-ELIGIBILITY](2026-07/EXP-P1-SPARSE-REWARD-ELIGIBILITY.md) | hit-conditioned Reward 尚未触发时，milestone 早筛能否诚实地继续？ | Franco | Codex | blocked | E1 | exact counter emitter/classifier 已实现；旧 live source 不可回填，等待新 source runtime receipt |
| [EXP-P1-SIGNED-FACE-RESCUE-FUNNEL](2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) | 拍面诚实修复后，线性角度引导能否用单 seed 脱离反面死区？ | franco | Codex | running/partial | E3 historical + E2 provenance | C3/D3 显式零摩擦 L1 已配对终档；同卷 K100/L2/第二 seed仍阻断 |
| [EXP-P1-CONDITIONAL-FACE-GUIDANCE](2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md) | 把击球窗固定成本从就绪缺口连续换成拍面误差，能否保住角度收益且不诱导逃离就绪区？ | franco | Codex | invalidated | E2 runtime | control/treatment 已各到 `model_200` 且 receipt 齐全，但冻结 `180..200` 窗内 treatment 的 conditional gate/cost/reward 全零、eligibility 为零；NOW 已按预注册判 activation-invalid：不买第二 seed、不晋级，也不把方向差异写成 Reward 效果 |
| [EXP-P1-V1V2-BASE-DECEL-INTERACTION](2026-07/EXP-P1-V1V2-BASE-DECEL-INTERACTION.md) | V1+V2 的击球精度能否与底座减速组合，同时降低平衡债？ | Franco | Codex | blocked | E1 | exact P1 source/单 seed配对已冻结；等待同 scene family full-scene probe |
| [EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN](2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md) | 补齐共同机制 activation 后，base-decel 权重 1 是否净改善平衡且不伤击球？ | Franco | Codex | completed / +500 invalid | E2 | model500 身份通过；配对无行为结论；共享 natural-wrap teacher source gate 已修，4096-env capture/readback 仍阻断 |
| [EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT](2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md) | 去掉内生 post-swing cold-start 后，base-decel 本身是否净改善？ | Franco | Codex | completed / treatment rejected | E2 | model1000 双臂 exact；冻结 raw-speed 门失败，不买 seed；同时发现 primary metric 未测实际 `v_des` tracking |
| [EXP-P1-LONG-NO-REPLAY-FUNNEL](2026-07/EXP-P1-LONG-NO-REPLAY-FUNNEL.md) | 早筛的 qdot 与 V1+V2 信号能否在无随挥回放的 10000-update 同源长曲线中保持？ | Franco | Codex | completed | E2 source/runtime | 三条曲线已在后续 task-revision cutover 中收口；早期 live 行只作历史快照，未形成可采用的完整行为胜者 |
| [EXP-P1-LONG-SCALEOUT-SIX-ARM](2026-07/EXP-P1-LONG-SCALEOUT-SIX-ARM.md) | 两张空卡能否用同一长曲线补齐模仿 `2×2`、关节速度与脚部朝向剂量曲线？ | Franco | Codex | completed | E2 runtime | 六格与唯一 importer retry 均已结束或在 cutover 收口；当前无 live trainer，未买第二 seed、未形成 vendor 结论 |
| [EXP-P1-POD1-LONG-BALANCE-REWARD-GRID](2026-07/EXP-P1-POD1-LONG-BALANCE-REWARD-GRID.md) | 非击球臂自由是否要到多拍后才显出平衡收益；击球位置/速度/拍面 Reward 怎样配比？ | Franco | Codex | completed | E3 runtime | 十二条 trainer 均曾越过首迭代并已收口；旧分母不足以合法选胜者，当前无 live trainer |
| [EXP-P1-DEMO-HOTSTART-PORTFOLIO](2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md) | 能否从三个 model-3500 母本严格续训六个组合，给次日演示准备多个候选？ | Franco | Codex | blocked | E1 | 专用 fail-closed queue 已冻结；等 parent finite/optimizer receipt 与 Pod2 GPU0/GPU1 release，后代永久 formal-ineligible |
| [EXP-P1-ROLLING-TIMING-SUPERCOMBO](2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md) | 同源 TTS 补偿与约 1.0/0.7/0.5 秒老师动作加速能否形成可部署准备时间的演示候选？ | Franco | Codex | superseded | E2 structural | 双 Pod 旧池已精确停止；active-swing target/TTS freeze 与 EMA-only 量尺使其不能回答部署时序或诚实淘汰，转 task-revision cutover |
| [EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1](2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md) | 显式零摩擦能否从 argv 到 checkpoint 完整闭合，并形成 guidance off/on 的 fresh L1 配对？ | Franco | Codex | L1 complete/behavior pending | E2 provenance | paired receipt `bb3cd749...bbde` 通过；不得重跑，等待同一 immutable K100 |
| [EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND](2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md) | 旧 exam bank 能否只改四个 metadata leaf 严格迁移到当前 signed-face family？ | Franco | Codex | completed/data-only | E2 | 真实 371 题 runtime replay 逐字节相同并发布 exact bank/report；新 schedule/paper activation/judge 仍阻断 |
| [EXP-P1-SIGNED-FACE-EXAM-PAPER](2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md) | 新 exact exam bank 能否冻结成不混旧 question ID、每侧 50 且全出手计分的 K100？ | Franco | Codex | paper_materialized_not_started | E2 | Pod1 clean detached `748b6d5` 已消费 exact private bank，物化 100 个唯一题、每侧 50 的不可变 schedule 与 paper-only activation；checkpoint attestor 仍只有 source/static gate，judge/L2/第二 seed/晋级全 false |
| [EXP-P1-HISTORICAL-SCHEMA3](2026-07/EXP-P1-HISTORICAL-SCHEMA3.md) | 新尺子能否区分历史候选？ | franco | Codex | completed | E4（诊断） | 尺子通过，候选仍为 inexact |
| [EXP-MUJOCO-NATIVE-TRAINING](2026-07/EXP-MUJOCO-NATIVE-TRAINING.md) | 原生 MuJoCo 微调能否减少留出集迁移损失？ | franco | Codex | blocked | E1 | off-main preflight 为 `NO-MERGE`；四个正确性缺口未修；无 trainer/backend/PPO |
| [EXP-MUJOCO-EVAL-FRAME-INTEGRATION](2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md) | latest main 能否同时保留 signed-face、parity guards 与 pelvis point/frame 合同？ | yikang | Codex | source integrated, pending main | E1 | focused `115 passed, 2 skipped`、root `647 passed, 9 skipped`；未授权 K100 |
| [EXP-MUJOCO-PELVIS-FRAME-PARITY](2026-07/EXP-MUJOCO-PELVIS-FRAME-PARITY.md) | MuJoCo 倒下是否来自 pelvis 点/轴坐标错位？ | yikang | Codex | completed | E2 | adopt 两处 evaluator frame 修复；主因与 ready-state 因果仍 inconclusive |
| [EXP-RECOVERY-TUPLE-ABC](2026-07/EXP-RECOVERY-TUPLE-ABC.md) | 哪种连贯的击球后命令语义能够安全恢复？ | franco | Codex | blocked | E1 旧结构；E0 新次序 | 新 reward 次序尚未同步 machine prereg；无行为结果 |
| [EXP-V9-YIKANG-CROSS-LEARNING-20260715](2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md) | Jiayi V9 / Yikang 支线哪些机制可复用，哪些结果受动作与来球分布限制？ | Franco | Codex | completed | E1 | 识别五类候选机制；旧 `7/7` 只测挥拍/恢复周期且未测物理触球/落台，不作为回球或泛化成绩 |
| [EXP-P1-LATERAL-BALANCE-PERTURBATION](2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md) | 恢复/等待窗内的有界横向躯干力能否增加可恢复平衡样本而不伤击球？ | franco | Codex | proposed | E1 | 两阶段 adapter 事务、伪造/异步/cache 攻击与中断冲量对账 source gate 通过；真实 Isaac adapter、GPU throughput、held-out paper 未闭合，禁止 launch |
| [EXP-MOTION-SPATIAL-RETARGET](2026-07/EXP-MOTION-SPATIAL-RETARGET.md) | 新空挥能否在不做不安全编辑的前提下放置到有效击球点？ | franco | Codex | running/partial | E2 | B 主选 schema-2/FK 一次性 consume 通过并解锁 L0；C 未消费后备，L1/桌网/动力学/RL 仍阻断 |
| [EXP-GATE3-CURRENT179-D0](2026-07/EXP-GATE3-CURRENT179-D0.md) | 当前 exact 179 policy/planner/runtime 能否完成一份固定考卷？ | franco | Codex | blocked | E2 | 仅通过模型预检；行为实验缺 runtime 前置 |
| [EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD](2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) | exact planner-policy 源码能否通过 portable Release，并无冲突地进入 latest main？ | franco | Codex | completed | E1 | adopt exact 源码；runtime/Gate3 行为仍未运行 |
| [v12/高点拍压/横移视频登记](motion_video_intake_v12_static_motion_20260713.md) | 7 段私有新视频是否能按精确字节和语义角色登记？ | franco | Codex | completed | E1 | 7/7 登记通过；不授予动作安全或训练资格 |
| [Franco 优先、static/motion GVHMR 预注册](motion_video_gvhmr_prereg_franco_static_motion_20260713.md) | 复用 Franco exact 结果后，static 与 motion 能否进入互不阻塞、no-clobber 的 GVHMR-only 小批？ | franco | Codex | completed | E2 | [S0](../DEFINITIONS.md#motion-s0)/[M0](../DEFINITIONS.md#motion-m0) `1/1 + 4/4` exact finite structural pass；后续 exact GMR 见专门卷宗 |
| [S0/M0 exact post-GVHMR handoff](motion_post_gvhmr_s0_m0_handoff_20260713.md) | 五条 exact GVHMR 结果能否无歧义进入 canonical-beta/GMR/schema-2 前置链？ | Franco | Codex | completed | E2 | S0/M0 runtime handoff exact SHA 已归档；后续 canonical-beta 与 exact GMR 均已完成诊断，正式门未开 |
| [S0/M0 exact donor canonical-beta](motion_canonical_beta_s0_m0_20260713.md) | 能否给新五条 exact PT 注入旧 donor，同时保证 beta 外 save/reload bit-exact？ | Franco | Codex | completed | E2 | 真实 `1+4` 条 PT 已 materialize 且 non-beta bit-exact；后续 exact GMR completions 已回收 |
| [S0/M0 exact GMR 与横移脚距](motion_exact_gmr_s0_m0_20260713.md) | 五条 canonical-beta PT 能否严格进入 A3 GMR，并拒绝横移末态收脚变窄？ | Franco | Codex | completed | E2 | S0 结构通过但需独立高球题族；M0 四条结构通过但 stance `0/4`，input-gate reject/no-RL；formal/schema2/training/hardware 全 false |
| [v12/高点拍压/横移组合设计](motion_v12_high_press_lateral_teacher_20260713.md) | 新动作和横移下肢老师怎样进入各自题族与全身组合？ | franco | Codex | proposed | E0 | S0/M0 exact GMR 诊断已完成；S0 需高球题族，M0 需末态 stance 修复，schema-2 未授权 |
| [非击球臂模仿消融](non_striking_arm_imitation_ablation_20260713.md) | 解除左臂模仿能否改善平衡且不破坏击球？ | Franco | Codex | preregistered | E1 | A0/A1 exact direct-mask source/runner 已冻结；Pod runtime/训练/同卷判读尚未运行，A2 固定预算继续 blocked |

新建记录使用 [TEMPLATE.md](TEMPLATE.md)。一个实验对应一个可证伪问题，不是一个 checkpoint；
各 checkpoint 应作为记录中的表格行。

## 写作规则

1. 发射前冻结假设、对照、自变量、固定变量、决策规则、负责人、commit 以及 asset/bank/checkpoint hash。
2. 原始 PID/SSH/重试输出放在产物日志中；本记录只保留会改变有效性或决定的事故。
3. 严格分开`已实现`、`机制已测`、`训练已跑`、`正式考卷已跑`、`已采用`。
4. 失败结果也是结果。`blocked` 表示实验没有运行到目标证据等级。
5. 结案必须使用 `adopt`、`reject`、`inconclusive` 或 `superseded`，并附一句理由。
6. 某个 setting 晋级时，把最终的分动作单球/连续球成绩表复制到 `NOW.md`，完整成绩表仍保留在此。
7. 先定义高于各组件消融的集成小目标。组件可以被采用，但小目标仍可能未完成。对等待/恢复问题，
   上一拍收尾、等待动作/姿态和任意时刻下一拍就绪必须在同一份 no-reset 考卷中一起通过。
8. run table 每行先写人话名，再附原始 `run_name`；不得用裸字母/缩写要求读者猜改了什么。
