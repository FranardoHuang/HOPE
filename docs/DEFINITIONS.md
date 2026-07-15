# 术语与人话对照

- **signed-face A2/B2**：同一个已审计父 checkpoint 的跨 Pod 热启动探索 L1；A2 是 guidance `0.0`
  对照，B2 只把 signed-face guidance 改为 `-0.4`，两条 child 都是 lineage-inexact，不能当 fresh 证据。

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
| <a id="launch-authorized"></a>`launch_authorized` / 科学训练发射闩 | 轻量训练 YAML 顶层的显式布尔门。`false` 时 `launch-next` 与 `fill` 在 SSH、claim 或 Kit 启动前拒绝；`plan/status/doctor` 和独立的 source-asset/probe/finalizer 前置门仍可运行。只有 terminal probe 证据审完并把目标行改为 `ready` 后才可设为 `true`；它不授权 judge、第二 seed、晋级或真机。 |
| <a id="dispatch-pods"></a>`dispatch_pods` / 活跃 Pod 集合 | YAML 队列里唯一允许 live snapshot、claim 检查和接收**新** trainer 的 Pod 名单。未列入的 Pod 不被普通 status/doctor/fill/launch/attest 访问；交接前必须先把其历史行改成终态，并给新任务使用从未发射的 namespace。它表达当前资源归属，不改变每卡容量上限。 |
| <a id="preferred-slot"></a>`preferred_slot` / 优先槽 | 某条 YAML job 希望优先落到的 `pod/gpu`，用于同卡配对或复用该 source 的 cold-boot receipt；只在该槽仍低于容量时生效，满载后仍回到全局 round-robin，不会越过 `dispatch_pods` 或容量上限。 |
| <a id="required-slot"></a>`required_slot` / 硬绑定槽 | 某条 YAML job 只能落到的唯一 `pod/gpu`。该槽满载时本 job 保持等待、不会 fallback，但其他槽上的独立 job 仍可调度；不能和 `preferred_slot` 同时声明。它用于保护他人保留卡或固定证据所在 GPU，不提供多 job 原子发射语义。 |
| <a id="trainer-run-binding"></a>`run_binding.json` / trainer 运行绑定 | trainer 在自己确定真实 RSL log directory 后原子写出的不可覆盖 sidecar；把 queue claim、PID/PGID、进程 starttime、物理 GPU、source/argv 与真实日志/checkpoint 根绑定起来。外部脚本不得用时间戳或 stdout 猜目录。 |
| <a id="milestone-attestor"></a>`milestone attestor` / 里程碑取证器 | 只沿已验证 `run_binding.json` 查找预注册 checkpoint，核对迭代号、finite、hard contract 和 launch lineage，再以 no-clobber 方式写证据；它不启动训练、不判卷，也不自动 stop/promote。 |
| <a id="runtime-binding-flag"></a>`runtime_binding: true` / exact 运行绑定开关 | 轻量训练 YAML 的显式 capability：只有 pinned source 已包含 trainer callback 与 `lean_queue_runtime.py` 才能设为 `true`，此时科学 run 才注入 claim/binding path 并允许里程碑取证。默认 `false` 保持旧 source 与非科学 boot warmup 兼容，绝不追认历史 run。当前 P1 只支持 fresh run，`checkpoint_path` 必须为空。 |
| <a id="boot-warmup"></a>`boot-warmup` / 1-env 缓存探针 | 为一个 exact source/Pod/GPU 单独创建的非科学小运行：沿用动作、题库和配方，但强制 1 environment、2 updates、独立 namespace/claim 与 180 秒 boot 上限；它只回答最小 importer/cache 路径能否越过 first iteration，不能代表正式环境数的 scene。其 checkpoint/指标永远不能当实验结果；失败只管理 claim 中的 exact PGID。 |
| <a id="full-scene-probe"></a>`full-scene-probe` / 完整场景启动探针 | 从一条 ready/blocked exact job 派生的非科学两次 update 运行：保留 source、Pod/GPU、完整 scene recipe 与原 `num_envs`，只改独立 `full_scene_probe_not_science_*` run name、`max_iterations=2`、save interval `1`，并写入 `_full_scene_probes/` namespace。只有看到首个 `Learning iteration` 才算 boot ready；它不可取证、判卷、晋级或当训练成绩。 |
| `PPO` | Proximal Policy Optimization，本项目使用的批量强化学习策略优化算法。测试/合同通过不等于 PPO 已实际训练。 |
| `VecEnv` | vectorized environment，并行推进多个仿真环境的训练接口。只有配置或 preflight 时，不能写成 `VecEnv` backend 已实现。 |
| `seed` | 随机种子。配方不变、只换 seed，用来看训练是否稳定，不许只挑最好的 seed。机制尚未成立时先用一个阻断 seed；第二 seed 只给胜者和匹配对照，`3–4` seed 只给正式候选。所有已运行 seed 仍须全量报告。 |
| signed-face 漏斗 `L1 / L2 / L3` | 该实验内部的三层证据购买：L1=`512 env × 25 update` 四格发射/合同冒烟；L2=`4096 env × 1001 update` 单-seed 机制 canary；L3=胜者与匹配对照通过预注册门后才购买第二 seed。它们不是下方 `E1/E2/E3` 证据等级，也不是课程 Stage。 |
| signed-face `C2 / D2` | v9 失败证据之后新建的两条 **fresh L1 provenance 对照**：C2 的位置/拍面 guidance 都为 `0`；D2 只把有符号拍面 guidance 权重改为 `-0.4`。两者使用同 host/seed/source/PYTHONPATH/`16/16` Kit thread cap、同动作/题库/plant/预算；按团队广度优先调度分别绑定 Pod1 GPU1/GPU2，Kit boot 由 host lock 串行，boot 后训练可并发。每个 checkpoint 同时绑定各自含权重的 hard contract 和带 GPU lane 的外层原子 launch claim。`2` 表示全新 namespace/provenance 修正版，不是第二 seed，也不是 motion 文档里的 C2。 |
| signed-face `C3 / D3` | C2 的零摩擦声明与真实非零 plant 不一致后新建的 fresh L1 配对：C3 关闭有符号拍面引导，D3 只把该引导设为 `-0.4`；两格都把 `task.plant.zero_joint_friction=true` 唯一地贯穿 argv、optimization recipe、outer claim、runtime marker、31/31 hard-contract 摩擦值和 checkpoint replay。`3` 是全新 namespace，不是第三个 seed；L1 通过也只证明合同/发射闭合，不证明动作效果。 |
| one-shot continuation | 旧 trainer 已按原配方完成，但外层验证器假拒绝后使用的**一次性续接控制器**。它只消费内容绑定的旧 claim/log/hard-contract/checkpoint，并且只能 claim 尚未存在的配对臂；不能覆盖旧证据、重跑已完成臂、改变训练配方或自动 retry。具体版本仍须单独说明，不能把 source gate 自动写成 runtime 已执行。 |
| signed-face C2/D2 `v1r1` | 第一版 D2-only one-shot continuation：修复了 v1 对 `[1.0,-1.0]` 的 float/int 假拒绝，但又错误要求 trainer 的五键 compact `question_bank` 记录直含第六个 `physics_contract_sha256`。最后一次成功只读快照证明它没有安装 control、写 continuation/claim 或启动 D2；因此 bytes 只作冻结负例，禁止运行。 |
| signed-face C2/D2 `v1r2` | 第二版 D2-only one-shot continuation：它正确接受 trainer 的五键 compact `question_bank` 并独立绑定 physics contract，但真实 runtime 在写任何 continuation/attestation/D2 claim 前证明 C2 的 31 个关节摩擦系数全为非零，与 manifest 的零摩擦声明矛盾。因此 C2 只保留 nonconforming 证据，D2/v1r2 永久 **NO-LAUNCH**；历史 absence 不构成授权。 |
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
| <a id="motion-l0-static"></a>motion `L0 static audit` / 动作 L0 静态审计 | 对 exact schema-2 动作做的纯 CPU 离散静态门：核对字段/顺序/finite/形状/时间、四元数、vendor MJCF 关节范围、逐帧 FK 与 root-foot 接地，但不调用 `mj_step` 或推进动力学。source/static gate 通过只说明计划、validator 和合成反例闭环；必须另有 exact 资产的 runtime certificate 才能声称 L0 runtime 通过。它不包含 vendor L1 自碰/自打、桌网扫掠、动力学、RL、Gate3 或真机，也不是 signed-face 的 L1/L2/L3 或证据等级 E1/E2/E3。 |
| <a id="motion-vendor-l1-safety"></a>motion `vendor L1 safety audit` / 动作厂商 L1 安全审计 | 在 exact vendor MuJoCo 碰撞模型中，把一条已通过动作 L0 的 schema-2 整轨做有限密集插值并逐样本检查机器人自碰、球拍/拍柄碰机器人及关键部位余隙。任一硬失败都会否决整条动作，不能由 reward 或其他好成绩补偿。当前 B 合同把 151 个 50 Hz 原帧按每段 8 个子步扫成 1201 个 400 Hz 样本；`<5 mm` 的 hard 判定直接使用 MuJoCo 饱和谓词，不用距离二分的近似 midpoint。关键组包含右肩三轴和右肘，仅右腕/手/球拍安装链从 proximity pair 排除。这仍是有限采样，不是数学连续时间扫掠证明，也不含桌网、动力学、训练、Gate3 或真机。它与训练阶段的 L1/L2/L3 不是同一层级。 |
| <a id="float32-ulp"></a>float32 `ULP` / 相邻格宽 | `unit in the last place`：某个 float32 数附近相邻两个可表示数之间的距离。它随数值量级变化，不是固定物理容差；动作 L0 V2 用预注册的格数、近零 floor 和独立物理上限共同约束纯舍入差，不能用它掩盖关节、接地、支撑或安全失败。 |
| `schema-3 bank` | 题库和判卷的第 3 版合同：训练题与考试题分开，题序、分母、动作和 SHA 可绑定。它不是 schema-2 motion 的升级同一件事。 |
| `q10` | 每个动作/侧各 10 题的快速方向卷；只看有没有苗头，不许据此停训或晋级。 |
| <a id="q50-and-k100"></a>`q50` | 每个动作/侧至少 50 题的同卷决策考试。当本项目只考正手和反手时，合计通常是 100 次。 |
| `K100` | 当前一张具体的、100 行 immutable paper：正手 50 + 反手 50，共用固定 schedule/order，且不删失败尝试。`q50` 是考试协议类型，`K100` 是这次的具体卷，两者不是普遍同义词。`K100` 也不自动表示 exact，还必须核对题库 bytes、语义和分母。 |
| `signed-face K100 checkpoint attestor` | 给一份**显式指定**的 checkpoint 做判卷前一次性取证：核对 filename/embed iteration、finite、fresh lineage、相邻 hard contract、producer claim、评测源码/runtime、MJCF/plant 和 actual K100 activation，再在 checkpoint-SHA 唯一路径写 evidence/claim。C3/D3 v2 还在 claim 前绑定训练时 ignored Isaac A3 asset inventory、hydrate/verify 角色及 `libGLU.so.1` 存在性。它不运行 judge、不产生成绩，也不授权停止或晋级。 |
| `C3/D3 K100 paired execution consumer` | 只消费已分别通过 checkpoint attestor 的 C3/D3 终档，在独立 eval worktree/runtime 中用同一 immutable K100 顺序判卷，并发布两侧 raw count 与 `D3-C3`；v2 还要求两侧共享同一已验证 ignored A3 asset closure。它不重跑训练，也不自动授权 L2、第二 seed 或 stop/promote。 |
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
| <a id="qdot-limit-hinge"></a>`qdot-limit hinge` / 关节速度限位铰链惩罚 | 只在实际关节速度超过各自运行时速度上限的一定比例后开始收费：`mean(relu(abs(qd)/limit-margin)^2)`。它读取 31 个 articulation 关节的真实速度和同顺序真实上限，不是 action-rate 平滑的别名；权重为非正惩罚，默认 `0` 表示关闭。 |
| <a id="conditional-face-guidance"></a>`racket_face_conditional_guidance_weight` / 不逃离就绪区的固定预算 Reward | 默认关闭的非正拍面/就绪联合塑形。它只在击球时间窗收费：未进入触点/完整拍速紧支撑门时保持固定最大成本，进入后按就绪度把成本连续换成 15° 以上的 signed-face 误差。位置或拍速越就绪，成本绝不会更高；在外门之外拍面梯度为零，策略不能靠故意离开门来免罚。输出 `[0,1]`，权重绝对值是每个时间窗 step 的最大罚金。它不替代 signed-face honesty、碰撞/跌倒或 Gate3。 |
| <a id="v1-free-wrist-velocity"></a>`V1` / 持拍手腕线速度模仿释放 | 在 body linear-velocity imitation 中排除持拍手腕，让球拍速度主要由击球目标 Reward 决定；位置、姿态、角速度模仿和所有安全约束仍保留。`V1=true` 只表示该排除已配置，必须另以 eligible denominator 与 exclusion numerator 证明运行时真的作用。 |
| <a id="v2-strike-window-imitation"></a>`V2` / 击球窗动作模仿四分之一 | 仅在预注册击球时间窗把动作模仿总尺度设为 `0.25`，给击球目标更多控制预算；窗外模仿不变。`V2=0.25` 必须另以击球窗 eligible denominator 与 scaled numerator 证明运行时真的作用。 |
| <a id="post-swing-replay-start"></a>`post-swing replay start` / 随挥后状态重放起点 | 策略完成挥拍后把自身状态写进环形缓冲；后续真实 episode reset 在缓冲已达到最小填充量时，按 `post_swing_start_prob` 从这些状态起步，以训练吸收上一拍余势和恢复平衡。它不是 carry-state 连续来球，也不是 learned reset；比较概率前必须记录 buffer-ready reset denominator 与实际 replay-start numerator。 |
| `post-swing retry authorization` / 随挥教师重签授权 | 一份内容寻址、一次性的 JSON 授权：它把唯一可接受的原始 capture producer tuple、修复后 attestor tuple、v3 plan、capture、teacher checkpoint 和输出 namespace 固定在一起。trainer 必须从配置给出的 exact 文件与 SHA 派生两条 source tuple，不能相信 receipt 自述；它只授权 attestor attempt-2，不授权重跑 capture、首 reset、科学训练、第二 seed 或 judge。 |
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
| `FK` / forward kinematics / 正向运动学 | 给定 floating root 和关节位置，用机器人模型计算每个 link/刚体的世界位置与姿态。本项目的离线 MuJoCo FK 不推进动力学时间，也不等于 simulator、碰撞安全或动作有效性通过。 |
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
