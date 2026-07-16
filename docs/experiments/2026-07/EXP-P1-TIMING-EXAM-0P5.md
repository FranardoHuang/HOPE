# EXP-P1-TIMING-EXAM-0P5：0.5 秒不可变时序卷

## 状态

- 状态：`blocked`。逐题试卷合同、严格 materializer/validator 和 **Isaac inexact 诊断接线**已实现并通过
  本地测试；私有题表尚未消费，也没有对真实 checkpoint 跑出 100 题行为分数。vendor MuJoCo 仍未接入，
  formal judge 仍未授权。
- Human owner：Franco
- Executor：Codex
- 分支：`Franco_codex/rolling-task-revision-20260716`
- 日期：2026-07-16

这里的 [`K100`](../../DEFINITIONS.md) 指固定 100 道题（正手 50、反手 50）的不可变卷；
`tts_seconds` 是从题目揭示到预定触球的剩余秒数。本实验只建立独立评测合同，
不改变训练、planner、runner 或机器人接口，也不在本页建立另一份项目优先级队列。

## 问题与假设

0.5 秒应是部署基本能力门，而不是宣称的最短可接时间。现有 rolling 训练把完整动作统一加速到
0.5 秒，但旧 BankExam 题表没有逐题绑定准备时间、初态和 time law（动作时间律），因此旧成绩不能回答：

1. 同一批来球在 0.5 秒内是否真的接住；
2. 失败来自 policy，还是 planner 已判定不可行；
3. 当前 checkpoint 是正式通过，还是只在 inexact（合同不完整）条件下方向正确；
4. 低于 0.5 秒时，能力曲线在哪里开始失效。

假设：在同一 signed-face K100 题序上，把每一题显式绑定到零速度第 0 帧初态、25 个 50 Hz tick、
动作侧专属 time law 和预注册可行性标签，可以得到一个不删失败尝试、可复算且可跨 evaluator 复用的
0.5 秒能力卷。

## 冻结合同

详细机器真源是
[`configs/phase1_timing_exam_0p5_k100_20260716.json`](../../../configs/phase1_timing_exam_0p5_k100_20260716.json)，
当前 file SHA-256 为
`e4414cd9ba90f9b7170907dcc569a70fa364e6e4a68a36aab764cbfbd7eb389f`，
canonical content SHA-256 为
`8156d59b186fc0353259a4619e60e2996c02de06210d1afd7ee0798c28b9bc9d`。

| 项目 | 冻结值 |
| --- | --- |
| 原题表 | signed-face K100：file `f2777dcd…ec7`、semantic `3ca4bdba…3365`、question order `09f778f2…bd0` |
| 题数 | 正手 50 + 反手 50；所有 scheduled attempts 都在分母中 |
| 初态 | `nominal-frame0-zero-velocity-v1`：机器人本体为同一 nominal stand 且速度严格为零；老师参考为各 clip 精确第 0 帧姿态，参考关节/刚体速度严格为零 |
| 触球剩余时间 | 0.5 秒 = 25 个 50 Hz tick |
| 正手 time law | native contact 66 tick，统一相位倍率 2.64 |
| 反手 time law | native contact 45 tick，统一相位倍率 1.8 |
| 可行性状态 | 每题 `expected_feasible=null`、`feasibility_status=hypothesis_not_certified`；0.5 秒只是待证基本门，不冒充 planner/TOPP 结论 |
| 单侧通过线 | 每侧至少 31/50 composite success；对应单侧 95% 单边 Wilson 下界约 0.50369 |
| composite | 合法回台，位置误差 `<7.5 cm`、速度误差 `<0.5 m/s`、signed normal 误差 `<15°` |
| 安全门 | physical fall、自打、非法桌/网碰撞、reset/teleport、deadline shift 全部为 0 |
| 缺测/无效/不可行 | 保留该题并计失败；禁止 censor、换题或缩小分母 |

`expected_feasible=null` 明确表示“0.5 秒是要考的基本门，但尚未证明每题动力学可行”。两条统一相位
time law 的 `topp_or_dynamics_certified=false` 也被同一合同冻结；在
[`TOPP`](../../DEFINITIONS.md)/动力学证书出现前，既不得把
未通过直接写成 policy 的正式失败，也不得把通过写成 planner/动力学正式通过。

## 为什么 tracked spec 不伪造 100 个题号

准确的 100 个 `question_id` 只存在于私有 runtime schedule；Git 中只保存其 file、semantic 和
question-order 三个 SHA。工具
[`scripts/materialize_phase1_timing_exam_0p5.py`](../../../scripts/materialize_phase1_timing_exam_0p5.py)
（当前 SHA-256 `224122ba1cf4080ca18de40e2278a2825ad878f60bf7e477662df5ad1f043e90`）
必须先逐字节验证私有 schedule，随后才以 O_EXCL/no-replace 方式生成逐题 paper。最终每行都会包含：

- `question_id`、`side`、`schedule_index`；
- `initial_state_id`、`tts_seconds`、`tts_ticks`；
- `time_law_id`、`expected_feasible=null`、`feasibility_status=hypothesis_not_certified`；
- 原题的 `bank_row`、`attempt_seed`、`repeat` 与被替代的旧 `hold_steps`。

任何 SHA、题序、side 数量、重复题、题号前缀、bank row、hold/repeat 语义或未知字段变化都会 fail closed。

## 结果账本和门槛

工具的 `score-result` 不信任 evaluator 自报 summary，而是要求 schema-v2 的 100 行结果与 paper 逐行对齐，
并从原始字段重算。`convert-isaac-scorecard` 还会逐文件验证 evaluator source closure、checkpoint、相邻
hard-contract、paper 与 schedule；拍面误差必须从 `face_normal_signed_pre_orient_world` 的 raw physical signed
normal 重算，禁止把 orient 后的 normal 当作 signed 量尺而洗掉约 180 度反向。缺失某一行是合同错误；存在但
无观测的行仍留在 100 题分母中并按失败计算。

Isaac 当前没有完整 self-hit 和非法桌/网碰撞仪表，因此这两个字段必须保持 `null`，不能补成 `false`；
`safety_observation_complete=false` 时，即使其余诊断指标通过，`formal_gate_pass` 也必须为 false。result ledger
以 O_EXCL/no-replace 一次性发布，已有目的文件时拒绝运行且原字节保持不变。

每份结果必须精确绑定：paper file/semantic SHA、checkpoint SHA、相邻 hard-contract SHA、evaluator source
SHA 和 evaluation execution-contract SHA。`evaluation_contract_exact=false` 的历史或 hot-start checkpoint
可以得到 `diagnostic_performance_pass`，但永远不能得到 `formal_gate_pass`。

## Isaac 0.5 秒接线

[`isaac_bank_exam.py`](../../../hope_training/whole_body_tracking/scripts/isaac_bank_exam.py)
现在有一个默认关闭的 timing-paper rider；未提供 `+timing_paper=...` 时，旧 BankExam 的 hold、native clock、
记录字段与判分路径保持原样。打开 rider 后按以下顺序 fail closed：

1. 逐字节校验 paper 与原 K100 schedule 的 file/semantic/question-order/bank SHA；
2. 用原 schedule 的题号、bank row 和 attempt seed，但把旧 hold 替换为 0；
3. 安装各侧 paper time law（正手 `2.64x`、反手 `1.8x`），禁止已有随机/固定 retiming 叠加；
4. 只在 evaluator 本地 `MotionLoader` 中把两个 clip 的第 0 帧参考速度行归零；不改源 NPZ、训练配置或后续帧，
   并验证第一份 actor observation 的参考姿态仍是精确 motion frame 0；
5. 每题必须在第 25 个 policy tick 恰好产生 exact strike；提前或延后都记
   `deadline_shifted=true` 并判该题 composite 失败；
6. 终场再次验证现有浮点动作时钟与两侧 speed table 未退回 native clock。

JSON 的每个 attempt 直接保留 `eligible`、`infeasible` / `planner_infeasible`、`deadline_miss`、
`deadline_shifted`、`contact`、`returned`、`composite` 和 `physical_fall`；summary 固定使用全部 100 题分母，
不会删除提前摔倒、没有触球或被 guard reset 的题。

这里的 `infeasible` / `planner_infeasible` 当前输出为 `null`：固定题 policy exam 绕开了 production planner，
所以不能把纸面标签冒充在线可行性测量。Isaac 路径使用解析拍球/飞行结果，缺少完整 self-hit 与非法桌网碰撞仪表，且统一
相位 time law 未经过 TOPP/动力学认证，因此强制要求 `+allow_inexact_contract=true`，只允许方向性诊断，
不允许据此 stop/promote 或宣称 formal pass。

## 触球剩余时间 sweep

固定计划为 `0.90 / 0.70 / 0.50 / 0.40 / 0.33 / 0.25 s`：

1. 每档先做每侧 10 题，只画能力曲线，不停训练、不晋级；
2. 0.5 秒始终做每侧 50 题；
3. 找到首次失效转折后，只把转折两侧两档扩为每侧 50 题；
4. `<0.5 s` 永远先标为 late-ball / infeasible-boundary 诊断，当前合同不授权“最短可接时间”正式声明。

后续必须用经过动力学验证的 time law 或每题 `tau_min`（该动作、来球和初态的最短可行时间）替换统一相位加速，
才能把低于 0.5 秒的失败进一步归因为 policy 或 planner。

## 本地验证

只读校验 tracked spec：

```bash
python3 scripts/materialize_phase1_timing_exam_0p5.py validate-spec \
  --spec configs/phase1_timing_exam_0p5_k100_20260716.json \
  --expected-spec-file-sha256 e4414cd9ba90f9b7170907dcc569a70fa364e6e4a68a36aab764cbfbd7eb389f
```

定向测试：

```bash
python3 -m pytest -q \
  tests/test_materialize_phase1_timing_exam_0p5.py \
  hope_training/whole_body_tracking/tests/test_isaac_timing_exam_adapter.py
```

2026-07-16 本地结果：`24 passed`。覆盖 strict JSON、spec/source/paper/result SHA、逐题 100 行派生、
50/50 平衡、no-clobber、31/50 边界、全出手分母、精确第 25 tick、零速度 frame-0 reference、
retiming 终场保持、安全零容忍和 inexact 只能诊断。

2026-07-17 P0 扩展回归：`73 passed`。新增覆盖 TOPP lock-window 运动学硬限、production 禁止插值刚体路径、
完整输入/源码 provenance、三文件 no-replace 发布、Isaac JSON/CSV 成对不可覆盖、schema-v2 result ledger、
raw signed face 重算、未知安全状态不假绿和 scorecard→ledger 一次性转换。本地主机未安装 `torch`，因此
`test_isaac_bank_exam_phase_b.py` 在收集阶段缺依赖，未计入该通过数；这不是 simulator 行为通过。

恢复私有 schedule 后，先 no-clobber 物化 paper；下面仍是 **Isaac inexact 诊断**，不是 formal judge：

```bash
python3 scripts/materialize_phase1_timing_exam_0p5.py materialize \
  --spec configs/phase1_timing_exam_0p5_k100_20260716.json \
  --expected-spec-file-sha256 e4414cd9ba90f9b7170907dcc569a70fa364e6e4a68a36aab764cbfbd7eb389f \
  --source-schedule /abs/path/signed_face_exam_k100.schedule.json \
  --output /abs/path/timing_0p5.paper.json \
  --confirm SIM_ONLY_MATERIALIZE_ONE_PHASE1_TIMING_EXAM_PAPER

hope_isaac_py scripts/isaac_bank_exam.py \
  task=HOPEPingPongVirtualBall headless=true device=cuda:0 \
  +run_dir=/abs/path/to/training_run checkpoint=/abs/path/to/model_N.pt \
  +exam_bank=/abs/path/to/schema3_exam_bank.npz \
  +schedule_json=/abs/path/signed_face_exam_k100.schedule.json \
  +per_clip_quota=50 +schedule_seed=0 +noise_scale=0.0 \
  +allow_inexact_contract=true \
  +timing_paper=/abs/path/timing_0p5.paper.json \
  +expected_timing_paper_sha256=<PAPER_FILE_SHA256> \
  +expected_timing_paper_semantic_sha256=<PAPER_SEMANTIC_SHA256> \
  +output_dir=/abs/path/isaac_timing_0p5

python3 scripts/materialize_phase1_timing_exam_0p5.py convert-isaac-scorecard \
  --spec configs/phase1_timing_exam_0p5_k100_20260716.json \
  --expected-spec-file-sha256 e4414cd9ba90f9b7170907dcc569a70fa364e6e4a68a36aab764cbfbd7eb389f \
  --source-schedule /abs/path/signed_face_exam_k100.schedule.json \
  --paper /abs/path/timing_0p5.paper.json \
  --expected-paper-file-sha256 <PAPER_FILE_SHA256> \
  --scorecard /abs/path/isaac_timing_0p5/scorecard.json \
  --expected-scorecard-file-sha256 <SCORECARD_FILE_SHA256> \
  --checkpoint /abs/path/to/model_N.pt \
  --expected-checkpoint-file-sha256 <CHECKPOINT_FILE_SHA256> \
  --checkpoint-hard-contract /abs/path/to/model_N.pt.hard-contract.json \
  --expected-checkpoint-hard-contract-file-sha256 <HARD_CONTRACT_FILE_SHA256> \
  --output /abs/path/isaac_timing_0p5/timing_0p5.result.json \
  --confirm SIM_ONLY_CONVERT_ONE_ISAAC_TIMING_SCORECARD
```

结论边界：代码层已经可以对 **[`v4rg`](../../DEFINITIONS.md) motion-contract 匹配的 checkpoint**
做真正 25-tick Isaac 诊断；
但截至本记录尚未消费私有 paper、尚未启动 simulator，所以“0.5 秒能否接住”的行为答案仍是未知，不能写成通过。

## 未闭环项

1. 尚未在恢复的 exact private schedule 上运行一次 materialize，因此没有正式 paper file SHA。
2. Isaac adapter 已接线但尚未实际消费；vendor MuJoCo evaluator 仍会走 native/reset clock，不能保证保留逐题
   retiming，因此当前必须 fail closed，尚无同卷跨引擎 100 行结果。
3. 当前统一相位 time law 没有 TOPP/动力学证书；0.5 秒只能作为待测基本门。
4. 本卷初态是零速度第 0 帧，不等价于上一拍 carry-state；连续恢复需要另一张绑定 post-swing 初态的卷。
5. 本实现只授权显式 `allow_inexact_contract` 的 Isaac 诊断；没有授权 trainer、formal judge、
   production planner/runner、部署或真机命令。
