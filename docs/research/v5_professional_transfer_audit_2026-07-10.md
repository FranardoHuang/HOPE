# V5 专业动作迁移终审与 Phase 加速器

状态：预注册；代码正确性尺修复完成后进入机制冒烟。人类责任人：franco；执行者：Codex。

## 要回答的科学问题

不是“机器人能否逐关节复刻专业人”，而是：

> 专业人的击球路径、触球几何和近端到远端发力顺序，能否作为 A3 可迁移的软先验；哪些部分必须由机器人按自己的力矩、行程和时间预算重解？

人和 A3 的质量、惯量、关节布局与执行器预算不同。智元口头信息“肘约 24 Nm、肩约
60 Nm”提示最优分配可能更肩/腰主导，但在拿到连续/峰值、输出轴/电机轴、速度-温度降额和
固件限幅前，这两个数只作设计线索，不能直接充当训练或安全硬阈值。

## 三个互相正交的轴

1. **老师轴**：任务自主解（PACE 式 task-only）/ V4 软先验 / V5 专业动作软先验。
2. **时间轴**：视频原节奏 / A3 本位重定时（`topp_mintime.py`；保留联合路径中的启动顺序，重解绝对时间）。
3. **路径行程轴**：原路径 / 延长引拍（`extend_stroke.py`）/ 重写随挥（`rewrite_followthrough.py`）/ 两者组合。

另设一个不与老师轴混算的**接触点物理消融**：现役用 URDF 的统一球拍控制点
`pingpang_red_Link` 原点及该点的刚体线速度；高保真候选再用实际球-拍接触点
`v_contact = v_control + omega_racket × r_contact`。前者衡量控制策略，后者衡量接触模型；
不能把接触模型变准带来的分数变化误写成 V5 老师更好。

再设一个**拍速口径消融**：触球帧 `f-1/f/f+1` 与 `+-1`/`+-2` 帧差分。现役
`+-2` 在 50 Hz 下是 80 ms 平均，并非瞬时触球真值；V5hLs 实测正手两口径
`2.488/2.315 m/s`、反手 `3.404/3.533 m/s`，而两个触球帧仍 `unverified`。
触球帧与速度窗口未验证前，不允许把该数当“专业人真实拍速”做结论。

不要直接跑完整笛卡尔积。行程只在离线守卫证明受限的一侧进入训练；task-only 没有人类时间轴，
因此也不制造没有语义的空对照。

## 预注册假设与可证伪结论

- H1：若 V5 软先验在同题考卷上稳定超过 task-only，专业动作包含可迁移信息。
- H2：若 V5 只有 A3 重定时后胜出，则可迁移的是路径/顺序，不是真人绝对时间。
- H3：若 V4 胜 V5，先区分 V5 资产质量与动作风格；不能把重建噪声失败写成“专业人无用”。
- H4：若 task-only 不差于 V4/V5，则现阶段任务约束已足够，模仿不是必要路线。
- H5：延长引拍的收益应由 `a_min = v*²/(2L)` 与力矩余量改善解释；固定 ΔL 只改变反向段速度
  不应改变该下界。若改变，现有刚体行程解释被证伪。
- H6：随挥重写只应改善触球后的制动/平衡，不应改变锁死的触球位置、拍速或拍面；任何触球量漂移
  都是工具失败，不是实验收益。
- H7：若统一控制点速度与接触点速度的判分排序一致，现役近似足够用于筛选；若排序翻转，Phase
  的正式尺必须显式带球拍角速度和拍内接触位置，旧虚拟回球数只能标为中心点近似。

## 加速执行梯子

当前已落地的 `scripts/v5_ablation_accelerator.py` 只负责两件可安全自动化的事：把 M1
feasibility 结果收成内容寻址 manifest，以及对同题 all-attempt scorecards 做保守成对
halving。它**不会**生成可行性数据、展开训练命令、启动 RunPod 或把任意 evaluator CSV
猜成 scorecard；这几步在 adapter 与 schema 测试完成前仍需显式生产。因而当前状态是“决策/
合同加速器已落地”，不是端到端无人值守训练器。下一检查点是把现有行程/时间律守卫输出接成
feasibility producer，再做 BankExam summary→scorecard adapter；两者都必须保留原始逐题记录。

### M0：先修尺（零 GPU）

- 固定 schema-v3 题表，每题有 `question_id`、attempt seed 与 schedule SHA；不同模型/噪声必须
  完成同一 K 题。
- checkpoint、ONNX、动作解码、actuator integration/armature/friction 语义/velocity limits、
  soft q-des limits、dt 与观测合同全绑定。
- 一题一 reset 是单球能力正式尺；所有 V4/V5/task-only 都从 MJCF named `stand` 完整
  初态启动。各自 teacher-reference reset 只能做 within-lineage diagnostic。
- manifest 与每张 scorecard 强绑 schedule/ready-state/MJCF/execution SHA；任一候选混初态或
  plant 直接拒绝整轮 halving。
- 连续状态链另作 Gate 3B，不混分母。

尺不精确，实验不得点火。

### M1：离线可行性筛选（每资产分钟级）

对 V4/V5 正反手的原路径、引拍 +20%/+40%、随挥重写、组合路径运行：

- 触球窗口逐位锁定；
- URDF/soft q-des 限位；
- 自碰撞；
- CoM/CoP、摩擦、逆动力学力矩与速度；
- `a_min`、引拍/制动行程、肘/肩/腰归一化力矩余量；
- 最快可行时间。

任何硬守卫失败直接淘汰。只有在原路径确受行程约束且 morph 明确扩大 L/改善余量时，才保留
“延长行程”训练臂。由此先把十几种组合压到每侧最多 2 个路径候选。

### M2：机制冒烟（512 env × 25 iter，约 3 分钟/臂）

只查：环境能否启动、合同是否精确、奖励/梯度是否活、是否 NaN、是否出生即摔、目标通道是否被
策略消费。机制失败立即停，不用短跑绝对分排冠军。

### M3：配对信号档（4096 env × 2000 iter，约 1.2 小时/臂）

所有候选使用同题 manifest、同训练预算和同 checkpoint 抽查点。以成对 question outcome 做
sequential halving：每轮淘汰被同组最佳稳定支配、且没有独立机制价值的后半候选。每个轴保留一个
反面对照，避免只剩组合件后无法归因。

### M4：成熟档

只让前 2 个配方进入完整训练，至少 3 seeds；每 1000–2000 iter 固定题表后台抽考，峰值区加密。
研究升段线为同题全-attempt 回球率 50%；真机候选线为 80%，同时要求物理不摔、执行合同和部署
安全门全部通过。两条线用途不同，不再让 50%/80% 在文档里互相争“唯一标准”。

### M5：连续与真机

单题 reset 的赢家再进入连续来球 Gate 3B。连续卷保留固定 serve schedule，但允许状态历史自然
分叉；报告恢复时间、连打回球率、fall/attempt、controller ACK/watchdog，而不是拿它替代单题尺。

## 统一判读表

主指标：同题 `return_success / all_attempts`，正反手分列。

必报副指标：

- contact/attempt、exact reach/attempt；
- 物理摔倒/attempt、保护性重置与删失原因；
- 触球位置/速度/拍面误差；
- requested/measured torque 相对每关节可信包络；
- elbow/shoulder/waist 的峰值与累计负载；
- q-des soft/hard clamp 命中率；
- 统一控制点与估计接触点的速度差 `|omega × r_contact|` 及是否改变逐题回球判定；
- CoP/摩擦违规剂量、回到 ready 的时间；
- schedule SHA、checkpoint/ONNX/contract SHA。

## 结果怎么解释

| 结果 | 能下的结论 |
| --- | --- |
| V5 原时间 > task-only | 专业路径与节奏整体可迁移 |
| V5 重定时 > V5 原时间、且 > task-only | 专业路径/顺序可迁移；真人绝对时间不适合 A3 |
| V5 延长行程后才胜 | 专业意图有用，但原重建路径没有给 A3 足够加速/制动距离 |
| V4 > V5 | 当前 V5 资产或风格不适合；需资产质量对照后再谈“专业性” |
| task-only ≥ 所有老师 | 当前任务约束足够，模仿路线不应继续占主算力 |

## 硬边界

- 不把旧判分器数字用于新臂生死。
- 不用真人关节轨迹作为硬约束；触球任务强、参考正则软，触球外权重可退火。
- 不因肘 24/肩 60 的口头数值直接改安全包络；先向智元要完整力矩-速度-温度曲线和固件语义。
- 不把 controller-side timeout、完整四 topic sequence/ACK、物理 E-stop 伪装成用户态 C++ 已解决。
- 现有 A3 checkpoint 的 PhysX joint friction 是把 MuJoCo `frictionloss` Nm 数值误当无量纲
  载荷系数得到的未标定 proxy。非零值不得记 formal exact；另做零摩擦/标定摩擦臂。
- 此处 `exact` 只意味判卷执行协议已绑定，不宣称 PhysX 与 MuJoCo 的 mass/inertia/COM/
  contact/solver/DR 已完全等价。

## 依赖

- [机器人本位时间律](robot_centric_timing_2026-07-09.md)
- [行程接口综述](stroke_interface_survey_2026-07-09.md)
- [分阶段奖励设计](reward_staged_design_2026-07-08.md)
- `hope_training/whole_body_tracking/scripts/{topp_mintime,extend_stroke,rewrite_followthrough}.py`
