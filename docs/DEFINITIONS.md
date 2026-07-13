# 术语与人话对照

本文件是现行术语真源。新人和 agent 不需要去历史归档猜缩写。

## 先遵守这条：不用黑话

- `run_name`、`flag`、实验臂代号和缩写第一次出现时，必须同行写出：**它是什么、改了什么、用来回答什么**。
- 报告不许只写 `M3`、`R9c`、`SZ`这类裸代号；可以保留代号便于查路径，但后面必须跟一句人话。
- 新缩写先加到本表，再在 `NOW`、实验记录或 `Gate` 文档中使用。
- 代码里已存在的参数名不强行翻译，但文档必须解释开关开/关各会发生什么。

## 当前训练与判卷术语

| 术语 | 人话 |
| --- | --- |
| `setting` | 一整套可复现配方：动作、观测、reward、题库、plant、训练方法和裁决尺必须一起指定。只换一项就是新 setting。 |
| `arm` / 实验臂 | 一条具体训练或对照配置，不是机器人的手臂。每条臂必须说清与对照相比只改了什么。 |
| `run` | 某条实验臂的一次实际执行。同一实验可以有多个 run。 |
| `PPO` | Proximal Policy Optimization，本项目使用的批量强化学习策略优化算法。测试/合同通过不等于 PPO 已实际训练。 |
| `VecEnv` | vectorized environment，并行推进多个仿真环境的训练接口。只有配置或 preflight 时，不能写成 `VecEnv` backend 已实现。 |
| `seed` | 随机种子。配方不变、只换 seed，用来看训练是否稳定，不许只挑最好的 seed。机制尚未成立时先用一个阻断 seed；第二 seed 只给胜者和匹配对照，`3–4` seed 只给正式候选。所有已运行 seed 仍须全量报告。 |
| signed-face 漏斗 `L1 / L2 / L3` | 该实验内部的三层证据购买：L1=`512 env × 25 update` 四格发射/合同冒烟；L2=`4096 env × 1001 update` 单-seed 机制 canary；L3=胜者与匹配对照通过预注册门后才购买第二 seed。它们不是下方 `E1/E2/E3` 证据等级，也不是课程 Stage。 |
| signed-face `C2 / D2` | v9 失败证据之后新建的两条 **fresh L1 provenance 对照**：C2 的位置/拍面 guidance 都为 `0`；D2 只把有符号拍面 guidance 权重改为 `-0.4`。两者使用同 host/seed/source/PYTHONPATH/`16/16` Kit thread cap、同动作/题库/plant/预算；按团队广度优先调度分别绑定 Pod1 GPU1/GPU2，Kit boot 由 host lock 串行，boot 后训练可并发。每个 checkpoint 同时绑定各自含权重的 hard contract 和带 GPU lane 的外层原子 launch claim。`2` 表示全新 namespace/provenance 修正版，不是第二 seed，也不是 motion 文档里的 C2。 |
| signed-face `v6r1` | `v6 retry validator 1`：为原 v6 的 D 单格准备的第一版补跑 validator。真实 Pod 只安装了 config/script，没有 claim、training run、signal 或训练；runtime `validate` 揭示它错误要求一个本应不存在的旧 training dir，因此已被 `v6r2` 取代，禁止启动。 |
| signed-face `v6r2` | `v6 retry validator 2`：修正 `v6r1` prelaunch validator 的纯源码版本，只支持 `static-validate`（静态合同检查）；它没有 runtime preflight、命令重建、进程检查、launch 或 finalizer consumer，明确 **NOT LAUNCHED**。original-v6 与 foreign-v8 都是前三格完成后，串行 launcher 的第 4 格 D 卡在 Kit/scene-start boot、均未到 hard contract/checkpoint；根因未知，不能写成学习、配方或拍面 reward 失败，也不得自动重试。下一步是 boot root-cause 与全新 v6r3-or-later preregistration，不是启动或复用 v6r2。 |
| `checkpoint` / `ckpt` | 训练到某个迭代时保存的模型存档，例如 `model_2000.pt`。 |
| `lineage` / 谱系 | 从初始模型、代码、资产到 checkpoint 的来源链。来源混了就不能声称严格单变量。 |
| `PID / PGID` | `PID` 是单个进程编号；`PGID` 是进程组编号。管理长任务时只能从经核对的 launch sidecar 读取 exact 数值并检查组成员，不能用相似命令行模式猜所有权。 |
| `fresh` / `causal continuation` | `fresh`=从零开始训，才可能成为新正式谱系；`causal continuation`=从旧 checkpoint 继续训，可看改动方向，但谱系不纯。 |
| `v4rg` | 项目内部的第四版参考挥拍动作对（正手+反手）：已重新对齐到机器人坐标，未发现明显起始毛刺或支撑脚滑移。它是资产族名，不是训练算法名。 |
| `v4rg_runtime_order_v3` | 当前 formal fresh setting 实际绑定的 v4rg 版本：schema-2、50 Hz，已迁移到 runtime body order。只写“v4rg”时只表示资产族，不足以复现 formal setting。 |
| `legacy v4rg` | 迁移到 `v4rg_runtime_order_v3` 之前的旧资产/顺序；可用于 causal 历史诊断，不得冒充当前 fresh exact 动作。 |
| `schema-2 motion` | 动作资产的第 2 版数据合同，包含 runtime 所需的关节/刚体顺序与元数据。 |
| `schema-3 bank` | 题库和判卷的第 3 版合同：训练题与考试题分开，题序、分母、动作和 SHA 可绑定。它不是 schema-2 motion 的升级同一件事。 |
| `q10` | 每个动作/侧各 10 题的快速方向卷；只看有没有苗头，不许据此停训或晋级。 |
| <a id="q50-and-k100"></a>`q50` | 每个动作/侧至少 50 题的同卷决策考试。当本项目只考正手和反手时，合计通常是 100 次。 |
| `K100` | 当前一张具体的、100 行 immutable paper：正手 50 + 反手 50，共用固定 schedule/order，且不删失败尝试。`q50` 是考试协议类型，`K100` 是这次的具体卷，两者不是普遍同义词。`K100` 也不自动表示 exact，还必须核对题库 bytes、语义和分母。 |
| Python `BankExam` | 仓库内的独立 policy 考试：Python 在 MuJoCo 中物理推进机器人，每题单独重置，再从击球时的球拍状态用解析模型推算接触和落台。它没有真实球拍—球碰撞，也不包含 planner、生产 C++ runner 或完整厂商运行链，因此不是 `Gate3/Gate3B`。 |
| `readiness audit` | 开卷前的只读资格检查：核对 checkpoint、contract、题库/schedule 和本机路径，不启动 judge，不产生成绩。 |
| `all-four activation` | 四 seed 同卷的启动授权文件：只有 Pod1/Pod2 readiness audit 和四份 checkpoint 全对上才能生成。它只允许下一步 `prepare`，不是 judge 已启动，更不是新分数。`judges_started=0` 就是还没有子判卷进程启动。 |
| `prepared_not_started` | 两个 Pod 已按 activation 物化 no-clobber runtime contract 和 K100 路径，但 `jobs_started=0`、`auto_start=false`。这比 activation 多一步执行纸面，仍不是已开卷或有结果。 |
| <a id="persistent-supervisor"></a>`persistent supervisor` / 持久监督器 | 对一条已审过的长任务做内容绑定、脱离调用 shell、无覆盖启动并只读复核身份的窄封装。本项目 q50 版本只有一次 `launch` 和只读 `inspect`，没有重试、信号、远程登录、训练、部署或真机权限；详见[接口合同](interfaces/q50_persistent_supervisor_contract.md)。 |
| `no-clobber` | 只允许首次创建产物；目标路径已存在就拒绝，不会静默覆盖旧合同、日志或结果。 |
| `NO-MERGE` | 该候选当前禁止合入 `main`。通常表示即使部分测试通过，仍有会制造错误证据或越权的明确缺口；修复并复核前不得把它当现行能力。 |
| 解析击球/解析上台 | 没有在 simulator 里用真实球-拍-台-网接触重放，而是从触球时的拍位/拍速/拍面经解析接触模型推出击球和落台。它是诊断尺，不是 physical return。 |
| `raw-A / physical-B` | 179-D actor 消费球拍 mount 原始 `+Y` 面法向 A；planner/外部协议传对手向物理击球面 B。runner 按正/反手的 per-clip sign 把 B 还原成 A。只看无向平面或对法向做自动翻转，会隐去“用了反面”。 |
| `signed-face honesty gate` | 要求判分器保留拍面法向正负号，不得通过 `orient_normal`之类步骤把 `n` 和 `-n` 当成同一拍面。这条门未通过前，高解析上台率不得晋级 setting。 |
| `exact` | 训练与判卷的动作、题库、观测、动作输出和执行合同能逐项对上。它只说合同一致，不等于物理已对齐真机。 |
| `formal target` | 实验前预先指定、有资格进入正式决策卷的 setting。 |
| `accepted baseline` | 已通过预定稳定性、留出卷和必要部署门，团队可以正式往上比的基线。`formal target` 不会自动变成 accepted baseline。 |
| `plant` | 机器人与环境在仿真里的物理对象：质量、惯量、摩擦、驱动器和数值积分都在内。 |
| `SZ` | fresh factorial 中的一格：`S`=正反手共用同一拍面语义，`Z`=31 个关节摩擦置零。它是当前执行合同的 formal target，不是标定后的真机 plant。 |
| `SP / LZ / LP` | 同一 factorial 的其他格：`L`=旧的正反手异号拍面语义；`P`=历史非零摩擦数字直填。`P` 存在单位/语义问题，因此只作诊断。 |
| `SC` | 计划中的“共用拍面语义+正确标定摩擦” plant。必须先有物理潜变量模型和 PhysX/MuJoCo 独立 adapter，不能把 `SP` 改名当成 `SC`。 |
| `carry-state` | 下一拍直接继承上一拍结束的真实机器人状态，不 teleport、不 reset。 |
| `T0 / T1 / T2` | 连续恢复的三层实验：T0 只在完整周期结束后换题；T1 在任意允许时刻事件驱动揭题，但冻结 reward；T2 才增加 learned recovery shaping。三者是实验层级，不是三个 reward。 |
| `PhysicalBall Phase A / B` | 物理球仪器的实现层级，不是课程阶段：A 只让来球在引擎中飞行，禁用机器人碰撞且不施加拍面冲量；B 才加入受合同约束的球拍接触和碰后飞行。B 有源码材料不等于已有被接受的运行或训练成绩。 |
| `readiness critic / critic-gate q50` | readiness critic 是估计“当前状态能否及时接住下一题”的模型。它必须用独立训练/校准数据且不能偷看未揭题信息；critic-gate q50 是在正式 Gate3B 之前单独封存的一次性 50 题/侧诚实考试。 |
| `guard reset` | 判卷器因跟踪包络或其他保护条件提前结束，但未发生真实倾倒。“物理不摔”不能隐去 guard reset，更不能据此证明连续恢复。 |

## 动作库术语

| 术语 | 人话 |
| --- | --- |
| `GVHMR` | Global Video-based Human Motion Recovery：把单目人物视频恢复成 SMPL-X 人体动作的离线前处理器。结构输出通过只说明人体重建文件形状和有限数合法，不等于机器人动作、安全或击球有效。 |
| `GMR` | General Motion Retargeting：把 GVHMR 的人体动作重定向到 Agibot A3 关节/刚体。GVHMR 结果不会自动授权 GMR；每一代输入、body shape、源代码和输出都要另做内容绑定。 |
| `TOPP` | Time-Optimal Path Parameterization：在不改几何路径的前提下，按速度、加速度等约束重新分配动作时间。它可以压缩过长 clip 或对齐阶段，但不会自动修正碰撞、平衡、拍面或击球点。 |
| `SMPL-X` | 带身体、手和姿态参数的人体模型表示；本项目把它作为视频动作与机器人重定向之间的中间制品，不把它当作 A3 runtime-order 动作。 |
| `S0`（static high-press batch） | 2026-07-13 新视频的单条离线结构批，只处理 `static/pai.mp4` 高点拍压。通过只说明 GVHMR 结构输出合法，不是第 0 个随机种子、训练阶段或动作晋级。 |
| `M0`（motion lateral-teacher batch） | 同一代新视频的四条横移老师离线结构批，按 left-1/left-2/right-1/right-2 顺序处理。四条是动作候选，不是四个随机 seed；GVHMR 不验证机器人脚距。 |
| `High press` / 高点拍压 | 右手机器人用反手在较高击球点迎球，球拍向前且拍面朝下，把球压回台内的独立动作类型。它不是被动挡球或反手拉球，必须使用自己的高球来球考卷。 |
| `Lateral locomotion teacher` / 横移下肢老师 | 只描述准备迈步、击球支撑和恢复三段的下半身/根节点参考动作，并以有符号横移距离为条件。它不是正手或反手挥拍本身，和上半身动作组合后仍须重新过全身安全与动力学门。 |
| `Non-striking arm` / 非击球臂 | 当前右手 A3 动作库中的左臂。取消它的模仿 Reward 只表示允许左臂帮助平衡，不会关闭关节、力矩、自碰或安全停机约束。 |
| `A0/A1 non-striking-arm pair` / A0/A1 非击球臂配对 | A0 是当前上半身模仿对照；A1 只从位置、姿态、线速度、角速度四条 body-imitation Reward 中删除左 shoulder/elbow/wrist，躯干、右击球臂、权重、题库、seed、预算和所有安全项不变。它不是恢复实验里的 A/B/C，也不是传感延迟 A1。 |
| `SE(2)` / 平面刚体变换 | 在水平面内只做一次整体偏航旋转和 XY 平移；本项目的动作站位实体化把同一个 proper transform 原子地作用于整条 floating-root 轨迹，禁止镜像、Z、尺度、逐帧、关节或时间编辑。 |

## 部署与全链路术语

| 术语 | 人话 |
| --- | --- |
| `Gate3` | 上真机前的全链路彩排：在厂商 MuJoCo 中把 planner、真机同款 C++ runner、消息和机器人执行串起来，先看能否稳定走完。独立 BankExam 只测 policy 与自己的 Python 评估器/机器人动力学配方，不能替代 Gate3。 |
| `Gate3B` | Gate3 的回球评分版：用当前阶段来球分布，并正式记击球率/上台率。它比“能稳定跑完”更近真机质量门。 |
| `first tick` | vendor simulator、通信、planner 和 runner 真正启动后，第一个有效控制周期。只过源码检查或 model preflight 不算 first tick。 |
| `portable Release` | 在 Linux 上用优化编译、但明确关闭 ROS 2 与 AimRT backend 的 C++ source/binary gate。它能证明 exact 源码可编译、链接并通过 native suite，不会启动 transport、simulator 或硬件。 |
| `native suite` | 编进 C++ `run_tests` 可执行文件的测试集合。缺可选资产导致的 skip 必须单列，不能算 pass。 |
| `AimRT` | 厂商部署 runner 使用的 middleware/backend 路径。AimRT 关闭的 portable Release 明确弱于 AimRT-enabled Release、backend first tick 和 Gate3 行为。 |
| `Gate3-D0` | 本项目的“第 0 版最短部署仿真闭环”：固定同卷、planner + policy + C++ runner + vendor runtime 的单拍演示；不冒充连续对打。它是项目内部标签，不是行业通用术语。 |
| `Trainer-v0` | native MuJoCo 训练的首卷。因现役 vendor main sim loop 没有球/球台/网，目前只能练单拍平衡与击球状态，不是 physical return 结果，也不阻塞几天内 `Gate3-D0`。它是并行候选训练轨；产物若晋级，仍须独立通过 Gate3/Gate3B。旧草案曾把它也叫 `D0`，从现行文档起停止这种重名。 |
| `Recovery-D0` | recovery A/B/C 预注册的第 0 步：只用现有 179-D checkpoint 做 A bridge 与 C previous-tuple 的 zero-shot 诊断，不选型、不晋级。原 config 字段仍叫 `D0`，文档必须写全 `Recovery-D0`，避免与 `Gate3-D0` 混淆。 |

## 证据和文档术语

| 术语 | 人话 |
| --- | --- |
| `E0` | 只有设计或预注册，没有实现证据。 |
| `E1` | 源码、单测或静态检查证据。 |
| `E2` | 真实 runtime smoke、真实模型装载或 first-component 运行证据。 |
| `E3` | 受控训练已实际运行。 |
| `E4` | 留出仿真考试、Gate3 或 Gate3B 证据。 |
| `E5` | 真机证据。 |
| 课程阶段 / `Stage` | 只回答“机器人正在多学会哪种球技”。现行顺序是：阶段 1 固定点；阶段 2 虚拟球变到达状态，站位和脚步属于其中的解法；阶段 3 物理球进场，旋转在后段加入。连续恢复和部署验证不占课程阶段编号。 |
| 连续能力线 | 每个课程阶段都要另考的横向能力：上一拍安全收尾、等待动作/姿态合格、下一拍随时可启动。`T0/T1/T2` 描述它的实验层级。 |
| 部署验证线 | 独立于课程阶段的验收顺序：独立 BankExam → `Gate3` 全链稳定 → `Gate3B` 回球质量 → G07 真机安全。 |
| `Gate` | 可验收的项目里程碑。`Done` 必须有可复现命令、验证结果、输入/输出和已知限制；只有材料或代码时写 `Partial`。 |
| `P0 / P1` | `NOW` 唯一队列中的相对优先级：P0 是当前最高层，P1 是下一层。它们不表示证据等级，也不得复制到实验状态。 |
| `red-team P1` | 代码复核中的高优先级正确性缺口，不是 `NOW` 队列的 P1。文档首次使用时应写全“red-team P1 正确性缺口”。 |
| 吞吐继续门 | 在启动长训前，用 N=1/8/32/64 并行环境实测 sim-only 和完整 rollout+一次 PPO update 的速度、内存与扩展效率；只有预计两臂×两 seed 能在 48 小时内完成且留 30% 余量，才继续 CPU-Python 路线。 |
| `人类责任人` / `执行者` | 责任人只能是人；Claude/Codex 只是执行工具或 provenance。不知道人类责任人时写 `UNASSIGNED`。 |

## 基线目标

`HITTER-compatible baseline` 指保留 HITTER 风格的分层：基于模型的 planner
输出击球位置、拍速和时刻，RL 全身控制器消费 planner 目标与机器人状态，
输出关节位置目标。它不表示照搬 HITTER 的具体实现；硬件、动捕、simulator、
部署 runtime 和改进策略都以 A3 项目真实约束为准。

## 坐标系与资产类型

- `world`：HOPE 球台/世界坐标，遵循 ROS REP 103：X 朝对手，Y 向左，Z 向上。
- `base_link`：机器人机身参考坐标，精确物理位置以机器人模型和 SDK 为准。
- `racket`：由机器人 FK 和固定球拍安装关系推导的坐标，不是 mocap 直接跟踪点。
- `ball`：来自 mocap 或未来感知系统的球位置。
- `Source`：代码、launch、config、脚本、消息定义和文档。
- `Curated data`：用于测试/示例的小型 bag、CSV 或图表。
- `Runtime asset`：模型权重、ONNX/RKNN/TensorRT engine、sysroot、预编译二进制和 vendor 包。
- `External reference`：供研究或可选实现参考的上游仓库。

坐标系的完整合同见 [frames_and_coordinates](interfaces/frames_and_coordinates.md)。
