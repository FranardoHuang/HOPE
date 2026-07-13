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
| [EXP-P1-FACE-PLANT-SCALEOUT](2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md) | 拍面×plant 广度矩阵哪些方向值得继续购买迭代？ | franco | Codex | completed/rejected | E4（诊断） | 16 臂已全部保留证据并停止；24/24 最近格的正手 signed composite=0，旧矩阵不能选 baseline |
| [EXP-P1-FRESH-SZ-STABILITY](2026-07/EXP-P1-FRESH-SZ-STABILITY.md) | 最接近正式 setting 的方案在不同 seed/checkpoint 间是否稳定？ | franco | Codex | completed/rejected | model-2000/4000：E4 diagnostic | 2k 与 4k 稳定性都失败；seed4 持续弱，旧 parsed 正手分被 signed-face 反例推翻 |
| [EXP-P1-FACE-SIGN-FORENSIC](2026-07/EXP-P1-FACE-SIGN-FORENSIC.md) | 高解析上台率是否隐去了拍面反号？ | franco | Codex | running | E4（旧卷诊断）+ E1（新源码） | `n/-n`/physical-B 源码门已实现；fresh canary 和修正后同卷未跑，旧分不晋级 |
| [EXP-P1-SIGNED-FACE-RESCUE-FUNNEL](2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) | 拍面诚实修复后，线性角度引导能否用单 seed 脱离反面死区？ | franco | Codex | proposed | E0 | 首轮只跑热启/从零 × 引导关/开四个机制单元；不复制 seed |
| [EXP-P1-HISTORICAL-SCHEMA3](2026-07/EXP-P1-HISTORICAL-SCHEMA3.md) | 新尺子能否区分历史候选？ | franco | Codex | completed | E4（诊断） | 尺子通过，候选仍为 inexact |
| [EXP-MUJOCO-NATIVE-TRAINING](2026-07/EXP-MUJOCO-NATIVE-TRAINING.md) | 原生 MuJoCo 微调能否减少留出集迁移损失？ | franco | Codex | blocked | E1 | off-main preflight 为 `NO-MERGE`；四个正确性缺口未修；无 trainer/backend/PPO |
| [EXP-RECOVERY-TUPLE-ABC](2026-07/EXP-RECOVERY-TUPLE-ABC.md) | 哪种连贯的击球后命令语义能够安全恢复？ | franco | Codex | blocked | E1 旧结构；E0 新次序 | 新 reward 次序尚未同步 machine prereg；无行为结果 |
| [EXP-MOTION-SPATIAL-RETARGET](2026-07/EXP-MOTION-SPATIAL-RETARGET.md) | 新空挥能否在不做不安全编辑的前提下放置到有效击球点？ | franco | Codex | preregistered | E1 | 无动作被推广 |
| [EXP-GATE3-CURRENT179-D0](2026-07/EXP-GATE3-CURRENT179-D0.md) | 当前 exact 179 policy/planner/runtime 能否完成一份固定考卷？ | franco | Codex | blocked | E2 | 仅通过模型预检；行为实验缺 runtime 前置 |
| [EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD](2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) | exact planner-policy 源码能否通过 portable Release，并无冲突地进入 latest main？ | franco | Codex | completed | E1 | adopt exact 源码；runtime/Gate3 行为仍未运行 |
| [v12/高点拍压/横移视频登记](motion_video_intake_v12_static_motion_20260713.md) | 7 段私有新视频是否能按精确字节和语义角色登记？ | franco | Codex | completed | E1 | 7/7 登记通过；不授予动作安全或训练资格 |
| [v12/高点拍压/横移组合设计](motion_v12_high_press_lateral_teacher_20260713.md) | 新动作和横移下肢老师怎样进入各自题族与全身组合？ | franco | Codex | proposed | E0 | 只有设计；没有处理后动作或行为证据 |
| [非击球臂模仿消融](non_striking_arm_imitation_ablation_20260713.md) | 解除左臂模仿能否改善平衡且不破坏击球？ | franco | Codex | proposed | E0 | 配对实验尚未运行 |

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
