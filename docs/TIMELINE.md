# TIMELINE — main 上值得记住的重要变化

Git 已经保存每个 commit 及其注释，因此本文件不复述 commit 流水。这里只记录以后团队必须知道的
重要根因修复、合同变化、方向翻案和共享能力。

规则：

1. 只有已经进入 `main` 的变化才能写；off-main WIP、排队状态和实验运行本身不写。若 main 的重要
   决策就是因明确缺口拒绝候选，可记录 `NO-MERGE` 原因，但不能把候选能力写成已合入。
2. 一个逻辑成果只写一行，即使红队过程中修了多个 commit。
3. 必须说清楚“原来错在哪、修了什么、还缺什么”；测试数量本身不是成果。
4. `人类责任人`只能写人；Claude/Codex 只能写在`执行者`。
5. 详细结果去[实验目录](experiments/README.md)，当前采用状态去 [NOW](NOW.md)，验收去 Gate 文档。

旧逐日流水完整保存在[历史 TIMELINE](experiments/archive/TIMELINE_legacy_through_2026-07-12.md)。
本文缩写的现行人话释义见 [术语与人话对照](DEFINITIONS.md)。

## 当前重要变化

| 日期 | 重要变化 | 人类责任人 | 执行者 | 修复/澄清了什么 | main 证据 | 仍然缺什么 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-14 | 每 GPU launch lock 不再被长训继承 | Franco | Codex | `flock FILE command` 的私有 fd 被 detached trainer 继承，使“Pod2 每卡3条”实际退化为每卡只能发1条；现由短命 controller 显式持 fd8，launcher child `8>&-`，锁只覆盖 doctor/容量/claim/boot。`preferred_slot` 让已过 cold-boot 的配对优先同卡，满载才回 round-robin。 | `4d492a1`、[队列操作](operations/run_lean_training_queue.md) | 现役旧 trainer 已继承的锁只等自然退出；仍需真实第二条同卡 first-iteration 验证修复。 |
| 2026-07-14 | 冷场景导入与科学训练 namespace 分离 | Franco | Codex | 动态 A3 URDF import 会在 reward/contract/PPO 前偶发卡死，过去污染正式 run 并等 900 秒；新 `boot-warmup` 只从 exact job 派生 1 env×2 update、独立 claim/namespace 与 180 秒上限，且专用确认 token、reserved-Pod 检查和 `not_science` 标记阻止它进入成绩。通用 launcher 对有内容但无增长的日志也在 180 秒留 sidecar并精确收口自身 PGID。 | `56535d5`、`9a115fe`、[队列操作](operations/run_lean_training_queue.md#新-source-先做冷启动探针) | 仍缺分阶段 scene-created marker；长期需 content-bound 预转换 USD。 |
| 2026-07-14 | 固定预算拍面/就绪联合塑形进入训练源码 | Franco | Codex | 朴素 `readiness*face_error` 配负权重会奖励策略故意离开就绪门；新公式在击球窗内保留固定成本，只允许位置/拍速就绪与拍面正确共同减免，并以标量单调反例锁住“误差变差不能少罚”。默认关闭，门宽与公式进入 hard contract。 | `61007e9`、[实验卷宗](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md) | 仍缺同 source control/treatment 的 Pod runtime、activation 分母与 `+200/+500/+1000` 行为早判。 |
| 2026-07-14 | 探索训练从手拼命令升级为 fail-closed YAML 发射合同 | Franco | Codex | 旧入口允许重复/owned Hydra key、未解析配方和复用 run directory，且 claim 不绑定真正执行 argv；现先编译单义 recipe、用最终 argv 做 no-Kit compose、原子创建 namespace，并让 canonical claim digest 自动进入 checkpoint launch provenance。显式 `dispatch_pods` 还把“Pod1 留给 Yikang、Codex 只用 Pod2”变成不可误发的机器合同。 | `169099c`、`97c341c`、[队列操作](operations/run_lean_training_queue.md) | 仍缺 trainer 写出的 exact RSL `run_binding.json`、milestone attestor、机制 activation 分母和 source-specific warmup phase marker。 |
| 2026-07-14 | 零摩擦声明从配置文字升级为贯穿发射与 checkpoint 的可验证合同 | Franco | Codex | 旧 C2 只在 manifest 写零摩擦，argv/recipe 漏传，真实 31 项全为非零；C3/D3 现把同一事实绑定到唯一 argv、recipe、claim、runtime marker、hard contract 与 checkpoint replay，并在 Pod1 发布 paired L1 receipt。 | `c2e81ba`、[C3/D3 卷宗](experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md) | 只闭合 provenance；同卷 K100、L2、第二 seed、MuJoCo/Gate3 行为仍未通过。 |
| 2026-07-14 | MuJoCo evaluator 的控制、自碰与 frame/provenance 语义进入同一严格合同 | Franco | Codex | 旧 evaluator 可把 P/D 分别截断而不是截总力矩、只在 control step 末看自碰、混用 pelvis link origin 与 COM/世界系 gyro，并让可覆写的 `partial` provenance 或旧 scoreboard 继续假绿；现 total-PD 后一次 clip、每个 physics substep 拒绝机器人自碰、COM/gyro frame 显式化，并让 mask/adapter/scoreboard 的来源与撤销 fail closed。 | `fef3eb4`、`c11551a`、`9aeb059`、[frame integration 卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md) | Pod2 tiny MuJoCo optional source gate 已 `10/10`；仍无 vendor MJCF policy rollout、同 policy 双引擎归因或 Gate3 行为通过。 |
| 2026-07-13 | schema-3 题库可做严格加法式物理合同重绑定 | franco | Codex | 目标源码只新增 signed-face helper 时，旧 loader 会正确拒绝整文件 SHA；新 consumer 不走 legacy，而以 Git/AST、全部问题数组 raw bytes、精确四-leaf metadata 和全题 old/new bitwise physics replay 发布新制品。 | `ecab785`、`62dfbbf`、[signed-face 漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) | v6 L1 尚未跑；新 train family 的 exam 也未重绑定，不能启动 exact judge/L2。 |
| 2026-07-13 | checkpoint 与 detached-worktree 发射闭包对齐实际运行布局 | franco | Codex | 首版误读 checkpoint；次版漏 source-first 环境；第三版过早 import Isaac；第四版又暴露 ignored A3 资产不随 worktree。现递归审计 `infos`，绑定确定性环境/模块 origin，并同时绑定 clean restore source 与 target 的完整 A3 tree；旧失败证据均保留。 | [signed-face 漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) | 仅修发射闭包；L1 行为、directional paper、MuJoCo/Gate3 仍未通过。 |
| 2026-07-13 | signed-face 身份门进入训练与解析共用量尺 | franco | Codex | 旧 Isaac virtual reward 与 NumPy/MuJoCo analytic scorer 会在 `orient_normal` 后把 `n/-n` 当同一平面；现先绑定 raw-A、逐 clip `[+1,-1]` physical-B 与 +X/hemisphere，负控证明错面不再得分，旧 unsigned 只能显式 inexact。 | `fdc3964`、`f4a2f3a`、[拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md) | fresh 训练 canary、修正后同卷和厂商 MuJoCo 行为仍未跑。 |
| 2026-07-13 | 私有动作 GVHMR queue 改为 exact batch + 不可变 child 输入 | franco | Codex | 旧通用 launcher 可把可变 staging 路径直接交给 child，也会混淆 Franco/v12/static/motion 优先级；现 S0/M0 只接受各自 prereg 和一次性 record，child 只读 O_EXCL 私有快照，并绑定 source/runtime/output/audit，旧入口只留历史证据。spatial validator 同时修正了外参声明字段层级的假拒绝。 | `882fea4`、`a4bbbaa`、[GVHMR 小批](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md) | 结构结果不等于机器人动作；仍缺 GMR/schema-2、足接触、桌网、自碰、动力学和行为卷。 |
| 2026-07-13 | model-4000 同卷推翻 `SZ` 稳定候选并实证拍面符号盲区 | franco | Codex | 四 seed 4k 为 `50/88/98/0`，稳定门全失败，seed4 非晚熟；同时 seed2/3 parsed 正手高分与 signed composite `0/50` 并存，因此旧分不再能选 baseline。 | [稳定性卷宗](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)、[aggregate](../configs/phase1_fresh_SZ_model4000_seed_stability_q50_aggregate_result_20260713.json) | signed 源码门已补；仍需 fresh canary、修正后同卷、厂商 Gate3/Gate3B 与 physical ball。 |
| 2026-07-13 | q50 一次性持久启动合同 | franco | Codex | 旧 activation consumer 的顶层进程会随 SSH shell 丢失，而 detached judge 仍可能继续；新 supervisor 只在核完进程身份与全 SHA 闭包后发布不可逆 token，token 可见后的任何异常都不得产生重试权。 | `2f2e705`、[执行子记录](experiments/phase1_fresh_sz_model4000_q50_20260713.md) | Linux `/proc` fake-runner、Pod 部署和真实 MuJoCo q50 尚未运行。 |
| 2026-07-13 | formal planner-policy causal tuple 与 exact portable Release 集成 | franco | Codex | 旧候选会让跨 topic 的旧 target/base 重新配对，或在读取本 tick base 前 engage；现用 shared epoch、command/base sequence、最新 tick-start base 与 revocation generation 统一 actor gate，并让 exact 源码通过 strict-finite Release。 | `6d6b778`、[实验卷宗](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) | ROS/Jazzy/AimRT、formal ONNX runtime、owned backend first tick、vendor MuJoCo 行为和硬件均未运行。 |
| 2026-07-13 | native MuJoCo feasibility/implementation 升为 P0，且不阻塞 `Gate3-D0` | franco | Codex | Isaac 高分与 MuJoCo 解析回球差反复背离，因此停止优先扩 Isaac-only sweep；首卷只做无球平衡/击球状态。红队又冻结 action trace/clamp、alias/exec、strict JSON、MJCF `strippath` 四项正确性缺口，以及 N=1/8/32/64、48 h 留 30% 余量的吞吐门。 | `abaea9b`、`68c0c2a`（`af803e9`）、`7c5f45b` | off-main preflight 候选仍为 `NO-MERGE`；无 trainer、backend、PPO smoke、physical return 或连续结果。 |
| 2026-07-13 | 文档总索引与实验登记骨架补回 main | franco | Codex | main 原本缺少统一 INDEX 和实验登记，任务路由分散在 Gate、operation 与聊天中；现补上一站式入口和实验最小字段。 | `3c7e507` | 该提交只有英文骨架，仍缺中文术语、完整实验卷宗和精简三本账。 |
| 2026-07-12 | strict 179-D face model contract | franco | Codex | 关闭两条假绿：把 actor raw face A 与对手向 physical face B 混淆；no-publish 顺便绕过模型校验。现在 export/runtime 绑定 per-clip sign、bank envelope 和 exact metadata。 | `8975043` | real-model preflight 通过；vendor behavior、self-hit、recovery 未测。 |
| 2026-07-12 | Gate3 source gate 与 runtime ownership 分层 | franco | Codex | legacy 脚本使用 broad process match，无法证明 fresh state。main 只接收 plan-only source gate 和明确 inexact 的 first-actor-candidate 诊断，进程启动/锁/信号所有权另立行为门。 | `b2067ba`、`7c0c385` | 无 owned vendor first tick 或 `Gate3-D0` behavior。 |
| 2026-07-13 | recovery A/B/C 结构合同与随机到球设计边界进入 main | franco | Codex | deploy idle 的混代 tuple 被定为 OOD，A bridge、B ready tuple、C previous tuple 分开；07-13 文档又将 T0/T1/T2、硬安全、随机到球环境轴和 reward 消融次序收紧。 | `e10922a`、`30484a2` | machine prereg/validator 仍固定旧三 reward/full `2^3`，待同步；无 recovery training 或连续 Gate3/Gate3B。 |
| 2026-07-13 | 新动作有效性改按“触球流形 × 适配题族 × 合法整轨站位”判断 | franco | Codex | 冻结站位 `0/64` 只能说明旧题未覆盖新动作，不能判动作无效；拉球、挡球必须用各自题型，离线安全只授予训练资格。 | `30484a2` | B/C 仍是重定位候选；无新 screen、训练或回球行为结果。 |
| 2026-07-12 | model-4000 matched paper 有了可审计 activation consumer | franco | Codex | 启动合同要求四 checkpoint 和两 Pod audit 同时满足，禁止单 Pod 或换题表绕过 barrier。 | `7bc6d1f` | consumer 只有机制证据；仍缺行为判卷和新分。 |
| 2026-07-12 | ball-physics 独立 reference oracle 恢复 | yikang | Codex | NumPy fit-lineage reference 加回 normal 校验与 source SHA，Torch contact/flight/landing 重新有独立对拍腿。 | `3df6ff5` | 只验证 physics code，不证明 policy 回球质量。 |
| 2026-07-12 | shadow solver 接入口 default-off 合入 | yikang | Claude | 加入 fail-closed backend interface、Echo 负控和 time-budget 诊断；关闭时不改变生产命令。 | `7e31819` | Torch solver body、warm start、omega 语义、runtime acceptance 未完成。 |
| 2026-07-12 | evaluator-owned same-paper contract 取代历史混分 | franco | Codex | immutable schedule、all-attempt denominator、exact motion/bank/checkpoint binding、train/exam 分离，防止不同题和幸存者分母互比。 | `83d7d56`（`f457ca3`） | 历史模型仍 inexact；physical instrument 仍不一致。 |
| 2026-07-11 | T1 event timing 变成 atomic state machine | franco | Codex | accepted strike 后原子揭题；miss 也消耗机会；robot/action/history/noise 全 carry。 | `be5d7cf`（随 Phase-1 infrastructure 合入） | 无 materialized schedule 或连续双引擎卷。 |
| 2026-07-11 | BankExam 统一 physical racket point 与 rigid-point velocity | franco | Codex | 旧 evaluator 对拍点和刚体点速度定义不一致，现把位置、角速度贡献和逐题合同绑定到同一物理拍点。 | `d4603d4` | 双引擎 physical instrument 和 mount 标定仍未闭合。 |
| 2026-07-11 | 部署 finite、fast-math 与 halt 安全修复 | franco | Codex | 移除会绕过 finite 检查的 fast-math，并让异常/停止路径 fail-closed。 | `a622997` | 不可抢占 backend send/ACK 和真机安全门仍是 G07 blocker。 |

## 重要未结论：Isaac–MuJoCo gap 还没有修好

同题工作已经定位出若干可复现差异和候选来源：racket-center tracking 不同、Isaac analytic face
normalization 会抹掉 signed-face 错误、缺 post-contact physical truth、历史 plant 单位错配。
目前只能写**部分差异已定位，整体因果归因与修复均未闭合**；没有 main 证据证明同一个修后
checkpoint 已在两个 physical instrument 上一致。真正关闭必须同时具备 semantics-correct、已标定的
plant，绑定一致的 runtime/observation/action/termination，以及冻结的两引擎 × physical/analytic 卷和
current Gate3/Gate3B 行为证据，并写明人类责任人、因果修复和 main commit。

## 基础里程碑

| 日期 | 里程碑 | 人类贡献者 | 价值 | main 证据 | 限制 |
| --- | --- | --- | --- | --- | --- |
| 2026-07-02 | A3 第一次 sim-to-real swing | yikang、jiayi | 证明 Route-A C++ 部署链能执行学习到的挥拍，真机行为接近 MuJoCo。 | `0bc9c53` | 仅正手、175-D legacy policy、scripted target；不是当前质量基线。 |
