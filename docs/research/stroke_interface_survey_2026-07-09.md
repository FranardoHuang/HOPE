# 行程接口文献侦察:谱系地图、直觉锚点、两层接口设计

**日期** 2026-07-09 · **作者** claude(pod2,工程卡 B)· **状态** 定稿候选(引用已逐条验真)
**前置** `inputs/research_result_{1,2,3}.md`(前任三路检索)· **过程** `B_research_log.md`
**范围** docs-only。不改代码、不改 watchdog、不改评估器默认行为。

---

## 0. 一页结论

1. **四路检索缺第二路(击球机器人)**,本卡补跑。另三路(TOPP 家族 / 生物力学 / 本库 papers 盘点)
   已在 inputs 里,本卡逐条验真。
2. **引用验真:49 条主引用,0 条纯编造;5 条须改写,2 条降级"待查",1 条删除。**
   最严重的一条是语义错(不是数字错):前任把 Serrien & Baeyens 的**峰值角速度时刻**当成了
   **启动次序**,并因此把肩肘顺序讲反了。详见 §5。
3. **谱系三族**(固定路径重定时 / 自由路径优化 / 基元参数化)俱全,开源实现基本齐备(§1)。
   **击球机器人域的空白是干净的**:没有任何先例把「行程 / 引拍深度」显式建成
   「所需触球速度」的函数。我们的 `extend_stroke` + 行程守卫正落在这个空处(§4)。
4. **franco 两直觉都有硬锚点,但都需要一次"翻译"才能用到刚体机器人上**(§2)。
   翻译的结论出人意料地统一:**对刚体机器人,引拍(countermovement)的收益主要不是生理性的
   「力建立时间」,也不是弹性储能——而就是把行程 L 加长**。于是两个直觉收敛到同一个分数式
   `a ≥ v*²/(2L)`。这给出一条可证伪的预注册(§2.4)。
5. **接口两层 = 松弛 / 加严的一对夹逼,不是"粗筛/精解"**(§3):
   - **快筛层**(松弛 ⇒ **必要条件** ⇒ 只能 **sound reject**):`a_min = v*²/(2L)`。
     `a_min > a_max` ⇒ **可证无时间律解**,回灌题库守卫。**PASS 不等于可行**。
   - **精解层**(加严 ⇒ **充分条件** ⇒ 可给 **sound accept**)。
6. **⚠️ 戒律:不得声称 TOPP-RA 直接可用。** 它每格点化为 LP 的前提是
   **约束集 𝒞ᵢ 为凸多面体、且对 (u,x)=(s̈, ṡ²) 仿射**。我们的 oracle 判据是**跨帧剂量**
   (违规帧时长占比)+ **腾空析取**,逐点可行集**根本不存在**。
   但障碍比"非凸"更具体、也更可修:四条障碍(剂量 / 腾空析取 / 关节黏性阻尼的 `√x` 项 / 摩擦锥是二阶锥)
   逐条给了修法与代价(§3.2)。**反直觉的一点**:被认定为主 binding 的**接触侧(CoP/摩擦锥/fz)恰恰是仿射的**,
   出问题的是力矩行——因为 31 个关节全带 `damping`。

---

## 1. 谱系地图

三族划分沿用 franco 的口径。开源列真实仓库;**⚠️** 标本卡改正的书目错误。

### 1.1 固定路径重定时(path 给定,只解时间参数化)

| 工作 | 出处 | 一句话 | 开源 |
| --- | --- | --- | --- |
| Bobrow, Dubowsky & Gibson 1985 | IJRR 4(3):3–17 | 开山:力矩限位映射为 (s,ṡ) 相平面加速度界,数值积分求 bang-bang | 无(思想被继承) |
| Shin & McKay 1985 | **IEEE TAC**(非 IJRR) | 同期姊妹工作 | 无 |
| Kant & Zucker 1986 | IJRR 5(3):72–89 | 提出 path-velocity decomposition | 无 |
| Pham 2014 | T-RO 30(6):1533–1540 · arXiv:1312.6533 | 数值积分法的鲁棒通用实现。**⚠️ 真题名是 "…of the Time-Optimal Path Parameterization Algorithm",非 "…of the TOPP Algorithm"** | ✅ `quangounet/TOPP`(2018-06 起停维,README 指向 TOPP-RA) |
| **TOPP-RA** Pham & Pham 2018 | T-RO 34(3):645–659 · arXiv:1707.07239 | 反向递推"可控集"(每格点一个小 LP)+ 前向贪心;比数值积分稳、比凸优化快 | ✅ `hungpham2511/toppra`;Drake `drake::multibody::Toppra` |
| Pham & Stasse 2015 | IEEE/ASME T-Mech 20(6):3257–3263 · hal-01138098 | 接触力当额外执行器,TOPP 扩到冗余驱动/多接触;HRP2-14(30 DoF)下 30 cm 台阶实证 | 部分(基于 TOPP) |
| Hauser 2014 | IJRR 33(9):1231–1250 | 两段式:接触流形上插值路径 → 沿路径时间优化;速度/加速度/力矩 + 多面体摩擦;**100 DoF、数十接触点、数秒** | ✅ Klamp't(`krishauser/Klampt`) |
| **AVP / AVP-RRT** Pham, Caron & Nakamura | RSS 2013(DOI 10.15607/RSS.2013.IX.052) | "路径也自由":沿路径段传播**全部可行末端速度区间**,嵌 RRT 在位形空间做 kinodynamic 规划 | ✅ `stephane-caron/avp-rrt`(2022-05 归档只读) |

**⚠️ 改正**:arXiv:1411.4045 **不是** AVP 的"期刊版",而是题名不同的预印本
(*Admissible Velocity Propagation: Beyond Quasi-Static Path Planning for High-Dimensional Robots*),
作者多一位 Lertkultanon(4 人)。

> **TOPP-RA 的两个性质与我们直接相关**(已由原文核实):
> ① **终端路径速度原生支持**:反向递推以 `𝒦ₙ := {ṡₙ²}` 初始化——这正是我们"触球帧拍速 = v*"的边界条件。
> ② **可控集 = 沿一维路径参数的反向可达集**,与题 3(反向可达集)口径同构。

### 1.2 自由路径优化(min-time / 终端速度硬约束)

| 工作 | 出处 | 一句话 | 开源 |
| --- | --- | --- | --- |
| Meier & Bryson 1990 | JGCD 13(5):859–866 | 间接法切换时间优化,两连杆 min-time,bang-bang 结构 | ❌ |
| **CPC** Foehn, Romero & Scaramuzza 2021 | Science Robotics · arXiv:2108.04537 | 时间做目标 + 途经点用**互补进度约束**绑定的直接法 NLP | ✅ `uzh-rpg/rpg_time_optimal` |
| **ALTRO** Howell, Jackson & Manchester | IROS 2019 | AL-iLQR + 投影 active-set;**min-time 用 τ=√Δt 当控制输入**(§III-C 原文,已核);终端等式约束原生 | ✅ Altro.jl / TrajectoryOptimization.jl |
| **Crocoddyl** Mastalli et al. | ICRA 2020 | 多接触 FDDP,接触序列给定;Box-FDDP 吃力矩盒。终端硬约束**靠罚函数**,自由末端时间**不原生** | ✅ `loco-3d/crocoddyl`(Opt2Skill arXiv:2409.20514 用它生成人形参考再 RL 跟踪) |
| **aligator / ProxDDP** Jallet, Bambade, Arlaud, El-Kazdadi, Mansard & Carpentier | **T-RO 2025, 41:2605–2624** · hal-04332348 | 近端增广拉格朗日约束 DDP:**硬等式/不等式原生**;Crocoddyl 系后继 | ✅ `Simple-Robotics/aligator` |
| **FATROP** Vanroye et al. | **⚠️ IROS 2023**(pp.10036–10043,最佳论文候选),**不是 ICRA 2023** · arXiv:2303.16746 | 结构利用的内点法 OCP,stagewise 硬约束,论文即演示 time-optimal | ✅ `meco-group/fatrop` |
| **OCS2** ETH leggedrobotics | — | SLQ/iLQR 实时 MPC;摩擦锥/自碰/gait 现成。min-time 不原生 | ✅ `leggedrobotics/ocs2` |
| **MuJoCo MPC** Howell et al. 2022 | arXiv:2212.00541(⚠️ arXiv 题名为 *Predictive Sampling…*;"MuJoCo MPC" 是软件名) | iLQG / 梯度 / Predictive Sampling 三规划器 | ✅ `google-deepmind/mujoco_mpc` |
| **Horizon** Ruscelli et al. 2022 | Frontiers in Robotics and AI | CasADi 框架,**dt 可作决策变量**(自由时间),接触约束内置 | ✅ `ADVRHumanoids/horizon` |
| **Koç, Maeda & Peters 2018** | Robotics and Autonomous Systems 105 | **乒乓专用 free-time OCP**:击球时刻自由 + 拍位姿/拍速终态**硬**约束;**明确抛弃固定虚拟击球平面** | ✅ `RobotLearning/traj-gen-and-tracking`(MATLAB) |

> **对 Koç 的一处重要澄清**:其目标泛函是 `min_{q̈,T} ∫₀ᵀ q̈ᵀq̈ dt`,即**自由末端时间的最小加加速度/能量**问题,
> **不是 min-time**。它证明了"自由击球时刻 + 触球终态硬约束"在线可解,但**没有**给出时间最优性。
> 谁要引它当 min-time 先例,是引错了。

### 1.3 基元参数化(带元参数的运动基元 / 学习策略)

| 工作 | 出处 | 一句话 | 开源 |
| --- | --- | --- | --- |
| Ijspeert DMP | — | 点到点,**终端速度恒为零** ⇒ 击球不可用 | ✅ 多社区实现 |
| **Kober, Mülling, Krömer, Lampert, Schölkopf & Peters 2010** | ICRA 2010, pp.853–858 | **击球 DMP 的源头**:引入**移动目标**,把期望末速 `ṫ` 写进变换系统 ⇒ 可在击球点指定**任意非零终速**,且时长与幅度可独立调;末速置零即退化为原始 DMP。7-DoF Barrett WAM | 未见代码 |
| **Mülling, Kober, Kroemer & Peters 2013**(MoMP) | IJRR 32(3):263–279 | 混合运动基元:门控网络选择/混合基元库;**元参数 = 期望击球点 + 期望击球速度**;**用固定虚拟击球平面**;7-DoF WAM | 未见代码 |

**反向可达集口径**(快筛层的理论支撑,归在此族的"解析核"下):

| 工作 | 出处 | 一句话 | 开源 |
| --- | --- | --- | --- |
| Webb & van den Berg 2013 | ICRA 2013 · arXiv:1205.5088(⚠️ 预印本题名不同:*…Optimal Motion Planning for Systems with Linear Differential Constraints*) | 线性系统 **fixed-final-state, free-final-time** 闭式最优控制(nilpotent A) | ✅ 多社区实现 |
| **Ruckig** Berscheid & Kröger | RSS 2021 · arXiv:2105.04830 | 多 DOF、**任意目标位置+速度+加速度**、jerk 受限的 time-optimal 在线轨迹生成,多轴时间同步。⚠️ **开源 Community(MIT)版已支持任意目标加速度**;Pro 版差异在本地中间路点/位置限位/tracking/硬实时。前身 Reflexxes(Kröger 2010),开源者为 Type II | ✅ `pantor/ruckig` |
| Haddad & Halder | ACC 2020 · arXiv:1909.12498 | 积分器链可达集边界的**精确凸几何刻画**(支撑函数)⇒ `L = v*²/(2a_max)` 的严格化与高阶推广 | 论文代码 |
| HJ 可达性 Bansal, Chen, Herbert & Tomlin 2017 | arXiv:1709.07523 | 反向可达管 = HJB PDE 粘性解;**精确但维度诅咒**(格点法 ≲5–6 维) | ✅ `StanfordASL/hj_reachability`;DeepReach `smlbansal/deepreach`(arXiv:2011.02082) |
| **Natherson & Scheeres 2024** | **JGCD 47(12):2560–2572** · doi:10.2514/1.G008118 | 指定方向可达集精确边界 + **终端速度约束**;含双积分器 rest-to-rest 与非零终速算例。**⚠️ 前任标"作者未核到",本卡三源(Crossref/Semantic Scholar/AIAA ARC)核定,非编造** | 未见 |

### 1.4 击球机器人(本卡补跑的第四路)

| 工作 | 年份/出处 | 触球终态怎么约束 | 行程/引拍怎么定 | 击球时刻 | 平衡 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| Kober et al. | ICRA 2010 | DMP **移动目标**指定非零末速 | 幅度 =(g−y₀) 由**击球点几何**定 | 外部给定 | 固定基座 WAM | 已验证 |
| Mülling et al. (MoMP) | IJRR 2013 | 元参数含击球点 **+ 击球速度** | 示教形状经空间/时间缩放泛化 | **固定虚拟击球平面** | 固定基座 WAM | 已验证 |
| Koç, Maeda & Peters | RAS 2018 | 拍位姿 + 拍速**硬终态** | 整条轨迹是优化的**隐式**结果 | **自由击球时刻** | 固定基座 WAM(7 DoF) | 已验证 |
| Büchler et al. | arXiv:2006.05935(T-RO) | 无显式终态;model-free RL | 摆幅 RL **涌现** | 端到端 | 固定基座**气动肌肉**臂(4 DoF) | 已验证 |
| D'Ambrosio et al.(DeepMind) | arXiv:2408.03906 | 无显式终态 | 摆幅**学习** | 事件触发 | **6-DoF ABB IRB 1100 + 双 Festo 直线导轨**,非全身平衡 | 已验证 |
| Dürr et al.(Sony "Ace") | **Nature 652:886–891 (2026)** | 无终态硬约束;SAC | 摆幅**学习** | 事件视觉 | **8 DoF = 2 移动 + 6 转动**,固定安装 + 2 轴直线台;**无腿** | 已验证 |
| **HITTER** Su et al. | arXiv:2508.21043 | planner 解析给拍速/拍面 | **两条**人类参考(正/反手),模仿 + RL 偏离 | **固定虚拟击球平面 x=−1.37 m**(作者自陈局限) | **全身**(臂+腿+腰) | 已验证(原文实抽) |
| **SMASH** Ren et al. | arXiv:2604.01158 | 任务命令(τ, v_hit, p_hit)进观测 | 库检索 + RL;**检索特征只含位置** | 无固定平面 | **全身**,自我中心视觉 | 已验证(原文实抽) |
| **PACE** Hu et al. | arXiv:2509.21690 | 零模仿,残差动作锚 ready 位形 | 完全 RL 涌现 | — | 全身 | 已验证 |
| Nguyen, Cancio & Kim(MIT) | arXiv:2505.01617 | OCP **硬约束**拍位置+朝向+速度 | 摆幅为 OCP **隐式**结果 | 固定时域 MPC | **5-DoF** 固定基座 | 已验证 |
| Ma, Cramariuc, Farshidian & Hutter(ETH) | **Sci. Robot. 10, eadu3922 (2025)** · arXiv:2505.22974 | 统一 RL 全身视觉运动策略 | 摆幅**涌现** | 无固定平面 | **腿式移动操作臂**(羽毛球) | 已验证 |
| Liu et al. | arXiv:2511.11218 | 三阶段课程(步法→精度引导挥拍→精修) | 挥拍由课程 RL 生成 | EKF 预测 | **人形全身**(羽毛球) | 已验证(⚠️ 项目名 "Phybot" 未证实) |
| Nguyen, Zaidi, Karol, Hodgins & Xie | arXiv:2510.08754 | **全身 MPC**,自旋感知 | MPC 隐式 | MPC | **四足 Spot,全身** | 已验证 |

**未验证(仅搜索摘要,不作论据)**:Saccon 组 impact-aware / reference spreading(⚠️ 该组在 **TU Eindhoven**,不是 TU Delft);
高尔夫挥杆机器人;棒球挥棒机器人。

### 1.5 我们在图上的位置

| 我们的件 | 落哪族 | 状态 |
| --- | --- | --- |
| `retime_motion_clip.py` | 固定路径重定时(保形缩放) | main |
| `synthesize_timing.py`(v1) | 固定路径重定时(匀加速单参族) | main(f923274) |
| `synthesize_timing_v2.py`(TOPP-lite) | 固定路径重定时(非均匀,oracle 在环) | main(f49e9db) |
| `topp_budget_search.py` | 外环预算搜索 | main(67ac959+ef7ae7f) |
| `topp_mintime.py`(v3) | 固定路径重定时(统一预算 min-time 双向) | **仅 `origin/topp-v3-mintime-0709`**,HEAD **fc39641**(非前任所记 e8efe5a),**未合 main**;单测 9+24 绿 |
| `scripts/feasibility_oracle.py` | 守卫的可行性发动机(A 层 mj_inverse) | main |
| `extend_stroke.py` | **行程 / 引拍轴(path morph)** | **未落仓**,原型在建 |
| 变体库 + SMASH 式检索 | 基元参数化 | 立项未实装 |

**覆盖**:时序轴全谱 + 变速语义 + 守卫 oracle。
**缺口**:①行程/引拍轴(在建);②拍面/侧向 morph v2 与变体库检索;③时间律工具与适配器统一接口
(`stage1_synthesis_2026-07-10.md` §5 就绪度表原文:"时间律合成 = 阶段 2 时序轴 … 缺口 = **与适配器接口对齐**");
④守卫的行程判据尚未接进 2-1 出题守卫。

---

## 2. franco 两直觉的学术锚点

### 2.1 直觉①「瓶颈(重的近端)关节提前开始动」= proximal-to-distal sequencing

**机制锚**:**Putnam, C.A. (1993)**, *Sequential motions of body segments in striking and throwing skills:
descriptions and explanations.* J Biomechanics 26(Suppl 1):125–135(PMID 8505347)。
近端段的前向加速通过**段间相互作用力矩 / 运动依赖力矩**(segment interaction / motion-dependent torques)
"逼"远端段先滞后、再被甩出。
> ⚠️ **诚实边界**:Putnam 摘要**未点名** Bunn,也未出现 "summation of speed principle" 字样。
> 把这两者绑定是二次文献的解读。定稿不写成 Putnam 明述。

**定量锚**:**Serrien, B. & Baeyens, J.P. (2018)**, *Systematic Review and Meta-Analysis on Proximal-to-Distal
Sequencing in Team Handball: **Prospects for Talent Detection?*** J Human Kinetics 63:9–21(PMC6162978)。

> ⚠️ **本卡改正了前任的一处语义错误,且改正后对我们更有利。**
> 前任把下面第二行当成了"启动次序"。实际上论文有**两张表**:

| | 骨盆旋转 | 躯干旋转 | 躯干屈 | 肩内旋 | 肘伸展 |
| --- | --- | --- | --- | --- | --- |
| **启动(onset)时刻**(Table 3) | **−330 ms** | −268 | −247 | **−94** | **−68** |
| **峰值角速度时刻**(Table 4) | −118 ms | −57 | −34 | **+6** | **−9** |

(单位 = 相对出手时刻的毫秒,罚球。)原文明述:
> "initiation of angular velocities follows a **strict proximal-to-distal sequence**,
> while for maximal velocities, **order of the shoulder and elbow is reversed**."

**对我们的三条读法**:
1. franco 的直觉说的是「**提前开始动**」= **onset**。正确的锚是 Table 3:
   **近端(骨盆)比最远端(肘)早启动约 260 ms**(−330 vs −68),而不是前任写的 118 ms。
   人类打击动作的次序展开尺度是**几百毫秒**量级。
2. **启动**严格近端→远端;**峰值**顺序里肩肘**翻转**。⇒ `extend_stroke` 只该约束
   **瓶颈关节提前启动**,**不该**顺带要求它也先达峰。过度约束峰值次序没有文献依据。
3. 别名对照(便于检索):kinetic link/chain principle、summation of speed principle、
   acceleration–deceleration principle、whip-like movement。

### 2.2 直觉②「加大引拍能打更快」= countermovement / SSC

**机制锚**:**Bobbert, Gerritsen, Litjens & Van Soest (1996)**, *Why is countermovement jump height greater
than squat jump height?* MSSE 28(11):1402–1412(PMID 8933491)。
结论:CMJ 比 SJ 高的原因**不是弹性能回收**(论文明确 "ruled out"),而是反向段让肌肉在**开始缩短之前**
就建立起高 active state 与高张力,于是缩短行程前段能多做功。

> ⚠️ **数字改正**:前任写"高约 18–20%"。原文报告的是 **CMJ 平均比 SJ 高 3.4 cm**(约 9–10%)。
> 且该对比是在**推蹬起始姿势相同**的条件下做的。

**配引**:van Ingen Schenau, Bobbert & de Haan (1997), *Does elastic energy enhance work and efficiency in
the SSC?* J Applied Biomechanics 13(4):389–415——同向结论。

### 2.3 ⚠️ 翻译到刚体机器人:两个直觉其实是同一个分数式

这是本卡认为最重要的一条综合结论。Bobbert 的机制有**两个生物学分支**,**对我们都不成立**:

- **弹性储能分支**:需要串联弹性(肌腱 / SEA / PEA / 闩锁-弹簧)。
  跳跃机器人的 catapult 路线正是它(Zhang et al. 2017 综述;froghopper 启发的可展开柔顺腿
  arXiv:2603.01128,PEBA 点阵储能,垂直跳高 **+17.1%**)。
  **A3 是刚性电驱 + PD,无串联弹性 ⇒ 此分支不可用。**
- **力建立时间分支**:肌肉 active state 的建立时间常数在 ~50–100 ms 量级(Ca²⁺ 动力学 / 横桥附着)。
  **电机电流环的力矩上升时间在毫秒量级 ⇒ 此分支对刚体机器人近似退化为零。**

那么"加大引拍"对**刚体**机器人还剩什么?只剩**运动学的那一份**:

> 功-能定理:`½ m v*² = ∫₀^L F(s) ds ≤ F_max · L` ⇒ `F_max ≥ m v*²/(2L)`,
> 等价地 `a_max ≥ v*²/(2L)`。**行程 L 变长,所需的力 / 加速度按 1/L 下降。**

**⇒ 对刚体机器人,countermovement 的收益 ≈ 把 L 加长。** 直觉②于是被直觉①的同一个分数式收编:
`extend_stroke` 的"反向加深"之所以对,不是因为 SSC,而是因为它**增大 L**;
"瓶颈关节提前启动"之所以对,是因为它给该关节**争取到更长的可用行程**。

这也解释了为什么 `docs/TIMELINE.md`(07-09 晚四)那句"**纯提前不加行程打不破界**"是对的:
提前启动若不换来 L 的增加,分数式右边不动。

### 2.4 我们自己的数据:直觉成立,且引拍空间不稀缺

行程账本(`stroke_result_1`,本卡**独立重算全部 14 个 a_min,逐位复现**):

| clip 族 | L_fh (m) | L_bh (m) | v*_fh | v*_bh | a_min,fh | a_min,bh | bh/fh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v4rg | 0.676 | 0.727 | 2.220 | 2.245 | 3.65 | 3.47 | 0.95 |
| swing | 0.877 | 0.425 | 2.640 | 1.725 | 3.98 | 3.50 | 0.88 |
| hopex | 0.767 | 0.751 | 2.274 | 2.249 | 3.37 | 3.37 | 1.00 |
| **v5hLs** | 1.130 | **0.497** | 2.488 | **3.405** | 2.74 | **11.67** | **4.26** |

- **不对称的分解**(本卡复核):`4.26 = (L_fh/L_bh = 2.274) × (v*_bh/v*_fh)² = 1.873`。行程短占 2.27×,拍速高占 1.87×。
- **单加行程只解一半**(本卡复核):L 加到正手同款 1.13 m,a_min 仍 **5.13**;
  要完全拉平到 2.74,需 **L ≈ 2.116 m**,或在 L=1.13 m 上把 v* 降到 **2.488 m/s**。⇒ 加行程 + 适度降拍速是组合拳。
- **时间律动不了它**:v5syn / v5topp / v5hLt 三个时间律变体(片长 ×1.95–3.45)a_min 纹丝不动
  (11.67 / 11.68 / 11.72)。**下界只认路径。**
- **引拍空间不稀缺,是 clip 没用**:v5 反手腰 pitch 触球前只用 0.908 rad 中的 **0.013 rad(1%)**;
  反向(前弯)有 **0.894 rad ≈ 51° 完全没用**。腰 roll(τ 瓶颈 No.1,峰 107–242%)只用 5–14%。
  对照 hopex 反手腰 pitch 用 32%(健康挥拍真用腰)。⇒ **franco 的 counter-movement 有真实物理空间。**

**📌 可证伪的预注册(由 §2.3 直接推出)**:
若 §2.3 的翻译正确,则 `extend_stroke` 的收益应当**只经由 ΔL 起作用**,按 `v*²/(2L)` 的规律标度;
而**不应**随"反向段速度峰值 / 反向加深的速率"变化(那才是 SSC 的签名)。
⇒ 消融:固定 ΔL,只改反向段的速度剖面 ⇒ **预测 a_min 与 oracle 剂量都不动**。若动了,§2.3 的翻译就错了。

---

## 3. 行程接口:两层设计

**两层不是"粗筛 / 精解",而是松弛与加严的一对夹逼**:

```
        松弛(丢约束)                          加严(加约束)
   T_relax   ≤        T_true(剂量制真值)        ≤   T_hard
   ─────────                                      ─────────
   快筛层:a_min = v*²/(2L)                       精解层:dose=0 硬闸
   给 必要条件 ⇒ 只能 sound REJECT                 给 充分条件 ⇒ 可 sound ACCEPT
                          ↑
              现状 v2/v3:oracle-在环贪心
              无最优性声明(工具自陈"真 min-time 上界")
```

### 3.1 快筛层:`a_min = v*²/(2L)` 进题库守卫

**推导(严格,不止匀加速)**:沿路径弧长 s,`v dv = a_t ds`,故
`v*² = v₀² + 2∫₀^L a_t(s) ds ≤ v₀² + 2·a_max·L`。⇒

> **`a_max ≥ (v*² − v₀²) / (2L)`,ready 静止起步时即 `a_max ≥ v*²/(2L)`。**

对**任意**切向加速度剖面成立(匀加速只是取等的那一支,也是峰值最小的那一支)。
用总加速度上界代替切向上界仍然是合法的(更弱的)必要条件,因为 `|a| ≥ |a_t|`。

**守卫签名(建议)**:

```python
def guard_stroke(clip, side, v_star, v0=0.0):
    L     = ledger[clip][side].L_deep          # 口径必须钉死,见下
    a_min = (v_star**2 - v0**2) / (2 * L)
    a_max = racket_point_tangential_envelope(clip, side)   # ← 今日缺口
    if a_min >  a_max:        return REJECT            # 可证无时间律解:不是"难",是"不可能"
    if a_min >  kappa * a_max: return ROUTE_EXTEND_STROKE   # kappa≈0.8,待标定
    return PASS                                        # ⚠️ PASS ≠ 可行
```

**四条必须写进契约的诚实边界**:

1. **守卫只做 sound reject,不做 accept。** `PASS` 只表示"没被这条必要条件证伪"。
   平衡、力矩耦合、接触可行性全被松弛掉了。
2. **`a_max` 的口径今日不成立(真缺口)。** `a_min` 是**拍点笛卡尔切向加速度**(m/s²);
   而现有预算是**关节空间** `|q̈|` 包络(v4 实证包络 ×1.5)。两者之间差一个 Jacobian。
   ⇒ 要么用 J 保守映射,要么**直接标定拍点切向加速度包络**。在此之前,守卫的 reject **不是 sound 的**。
3. **`L` 有两种口径**(hold 后首动帧 / 引拍最深点),v5 反手两口径差 0.21 m
   ⇒ a_min **8.2 vs 11.7**。守卫必须钉死一种(建议 `L_deep`,与 `extend_stroke` 的手术对象一致)。
4. **`a_min` 不是摔率预测器。** 全 10 clip 上 `ρ(a_min, CoP 剂量) ≈ −0.21 / −0.26 ≈ 0`。
   正确读法(沿用 `stroke_result_1`):**a_min 定"路径病"的地板,时间律定地板上方摔多少**。
   ⇒ 守卫里它只当**硬拒绝器**,**不当难度分**。
5. 口径警告:`hopex` 是干挥无球,其 `v*` 为速度峰约定帧,非真触球速度;`v4rg/swing/v5hLs` 相位 unverified。

**学术锚点**:`v*²/(2L)` 的严格化 = 双积分器可达集的凸几何(Haddad & Halder, ACC 2020);
多关节推广的闭式解 = Webb & van den Berg(fixed-final-state, free-final-time);
工程化在线内核 = **Ruckig**(多轴 jerk 受限、任意目标位/速/加速度、Community 版即可)。

### 3.2 精解层:算法选型 —— 与 `mj_inverse` 黑盒 oracle 的兼容性论证

#### 3.2.1 TOPP-RA 的前提(原文核实)

TOPP-RA 把约束离散成 `aᵢuᵢ + bᵢxᵢ + cᵢ ∈ 𝒞ᵢ`,其中 `u = s̈`(路径加速度)、`x = ṡ²`(路径速度平方);
并明述:
> "the set of admissible control-state pairs is `Ωᵢ = {(u,x) | u·ã(sᵢ) + x·b̃(sᵢ) + c̃(sᵢ) ≤ 0}`"
> "**Since 𝒞ᵢ is a polytope, Ωᵢ is a polygon … the above equations constitute two LPs.**"

⇒ 三条前提:**(P1) 逐点**(可行性只依赖 `(sᵢ, uᵢ, xᵢ)`);**(P2) 对 (u,x) 仿射**;**(P3) 𝒞ᵢ 是凸多面体**。

#### 3.2.2 我们的 oracle 实际是什么(读码,`scripts/feasibility_oracle.py`,928 行)

- 引擎 = MuJoCo **`mj_inverse` → `qfrc_inverse`**,逐帧;查 `|τ|/τ_max`、`fz≥0`、`CoP∈支撑凸包`、`|f_t| ≤ μ·fz`(μ=0.8)。
- **判据是剂量制(time-share)**:`FAIL when cop dose ≥ 0.35 or torque dose ≥ 0.10 or friction dose ≥ 0.05`,
  外加持续 `fz<0` / 腾空违规;逐帧阈值(`TORQUE_FAIL=1.0` 等)**只定义"什么算违规帧"**。
- docstring 自陈:**known-good floor(v4rg)在鞭打段带有 transient CoP excursions** ⇒ 单帧越界被**设计性容忍**。
- `COP_MIN_FZ = 1.0`:`fz` 低于此值时 **CoP / 摩擦锥无定义**。

#### 3.2.3 逐条对照:哪条前提破了

先把动力学沿路径展开。取 `u = s̈`、`x = ṡ²`(前向单调时间律 ⇒ `ṡ = +√x ≥ 0`),`q̇ = q′(s)ṡ`、`q̈ = q′(s)u + q″(s)x`:

```
τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + D q̇ + F_c·sign(q̇)
  = [M q′]·u  +  [M q″ + C̃(q,q′)]·x  +  g(q) + F_c·sign(q′)   +   [D q′]·√x
    └──────────────── 对 (u,x) 仿射 ────────────────┘            └── ✗ 非仿射 ──┘
```

- Coriolis/离心项对 `q̇` 二次 ⇒ 对 `x` **线性** ✅;`armature` 并入 `M` ⇒ 仍仿射 ✅;
- 干摩擦 `frictionloss`:`ṡ>0` 时 `sign(q̇_j) = sign(q′_j)`,在每个格点是**常数** ⇒ 并入 `c(s)` ✅
  (仅在 `q′_j` 变号处跳变);
- ⚠️ **黏性阻尼 `D q̇ = [D q′]·√x` 对 `x` 是 √x,不是仿射。**

**实地核对 MJCF**(`agi/…/a3_pingpong/a3_pingpong.xml`):**31 个关节全部带 `damping`**(0.5–2.0)与 `frictionloss`;
oracle docstring 明写 "**Damping / frictionloss / armature stay ON — they are real physics the motors must pay for**"。
而根关节是 `<freejoint name="pelvis_free_joint"/>`,**无 damping**,`<default>` 块也只设 geom、不设 joint damping。

⇒ **关键分割**:阻尼只污染**31 个关节力矩行**;**浮动基 6 行(→ GRF 旋量 → `fz` / CoP / 摩擦锥)仍然精确仿射**。

| 前提 | 我们这里 | 判 |
| --- | --- | --- |
| **(P2) 对 (u,x) 仿射** | **一半成立**。浮动基 6 行(接触可行性)**精确仿射** ✅;但 **31 个关节力矩行被黏性阻尼的 `√x` 项破坏** ❌。`robot_centric_timing` §七 说"τ 对 s̈ 仿射"——对 `s̈` 确实仿射,但 TOPP-RA 要的是**同时对 `(s̈, ṡ²)` 仿射**,阻尼在这里掉队 | ⚠️ 半破 |
| **(P3) 多面体** | **部分成立**。`|τ|≤τ_max` 是盒(✅);`CoP∈支撑凸包` 在 `fz>0` 下交叉相乘后对旋量**线性**(✅);`|f_t| ≤ μ·fz` 是**二阶锥**(凸但非多面体)⇒ **金字塔化**即可(保守) | ⚠️ 可修 |
| **(P1) 逐点** | ❌ **破了,而且这是致命的一条**。剂量 = 违规帧的**时长占比**,是跨帧的 **0-1 计数(cardinality)约束**:既非逐点、也非凸。一条轨迹可以在 30% 的帧越界仍然 PASS ⇒ **逐点可行集根本没有定义**。TOPP-RA 的反向可控集递推预设"给定 `(uᵢ,xᵢ)`,`sᵢ` 处的可行性与别处无关",剂量制直接摧毁这个前提 | ❌ |
| **(P1/P3) 第二处破坏** | ❌ **腾空析取**。`fz < COP_MIN_FZ` 时 CoP/摩擦无定义,腾空帧改判 `|f_req| < 0.05·weight` ⇒ 逐点可行集是「支撑凸集 ∪ 腾空凸集」的**并**,非凸 | ❌ |
| **黑盒不吐系数** | ⚠️ **不是本质障碍**。`D`(MJCF 已知)与 `q′(s)`(路径已知)⇒ **阻尼项可解析扣除**;扣除后对固定 `sᵢ` 是 `(u,x)` 的精确仿射映射,**3 次探针**(`(u,x)=(0,0),(1,0),(0,1)`)即可恢复 `(aᵢ,bᵢ,cᵢ)`。⚠️ 落地前必须先跑**仿射性残差检验**(mj_inverse 实测 − 仿射模型,残差应只剩 `D q′√x`),把 `frictionloss` 在 MuJoCo 里究竟走 efc 约束还是 passive 力这件事钉死 | ⚠️ 可修,须先验 |

**阻尼怎么救**:`√x` 在 `x ≥ 0` 上**凹**,故在 `x ∈ [0, x_max]` 上存在**仿射上界**(切线或割线);
把 `[D q′]·√x` 换成其仿射上界即得**保守多面体**(sound,但收紧了 τ 预算)。
另一条更粗的:把阻尼力矩当有界扰动 `|D q′|·√x_max` 并入**收紧后的 τ_max**。两者都保守。

#### 3.2.4 结论与戒律

> **⛔ 不得声称 "TOPP-RA 直接可用"。**
> 但要把话说准。**接触侧(CoP / 摩擦锥 / fz)本身恰好是仿射的、可多面体化的**——
> 而 §九.1 恰恰认定接触可行性是**主 binding**。真正的障碍有四条,按严重度排:
> **(i) 判据是跨帧剂量(0-1 计数约束,非逐点、非凸)——致命,毁掉逐点可行集本身;
> (ii) 支撑 / 腾空析取,使逐点集成为两片凸集的并;
> (iii) 关节黏性阻尼在力矩行引入 `√x` 项,破坏对 `(u,x)` 的仿射性(接触行不受影响);
> (iv) 摩擦锥是二阶锥而非多面体;黑盒不吐仿射系数。**
> 后两条是技术性的、可保守化的;前两条是**判据语义**层面的,必须先改判据。

**可恢复路径**(若要真 min-time 的最优性):
`R1` 剂量闸 → **dose = 0 逐点硬闸**(`robot_centric_timing` §九.4 的 TOPP v2 预算引擎規格恰是"不可行剂量为零");
`R2` **冻结接触模式**(挥拍段双脚支撑,无腾空)⇒ 消掉析取;
`R3` **阻尼项仿射上界**(`√x` 凹 ⇒ 切线/割线上界)或并入收紧的 `τ_max`;
`R4` **摩擦锥金字塔化**(内近似,保守);
`R5` **3 探针/格点** + 解析扣除阻尼,提取仿射系数(**先跑仿射性残差检验**)。
⇒ 得到真多面体 `𝒞ᵢ` ⇒ **TOPP-RA 可用**,且其终端路径速度边界条件(`𝒦ₙ := {ṡₙ²}`)**原生**就是我们的"触球帧拍速 = v*"。
注意 `R1`–`R5` **五条全是保守化**:每一条都只会让可行集更小、时长更长。

**但代价必须写明**:oracle 自陈 known-good floor 在鞭打段带 transient CoP excursion ⇒ `dose=0` 会把
**已知能跑的 clip(v4rg)判死**。所以 `R1–R4` 的 TOPP-RA 解答的是**另一个(更严的)问题**:
`T_min^{dose=0} ≥ T_min^{dose}`。它可以当**可行性证书 / sound accept** 用,**不能**当我们要的答案。

**选型建议(按最优性诉求)**:

| 诉求 | 选型 | 代价 / 注意 |
| --- | --- | --- |
| **现状,工程可用** | 维持 `topp_mintime` v3:外层 γ 扫 + 内层 oracle-在环贪心 | **无最优性声明**——工具 docstring 自陈"min-time = 本搜索族内最短 = **真 min-time 上界**"。诚实,推荐维持 |
| **要可行性证书** | `R1–R4` + **TOPP-RA**(`toppra` / Drake `Toppra`) | 过保守,会毙掉 v4rg。只用来**证明可行**,不用来定时长 |
| **要自由路径 min-time + 硬终态** | **aligator / ProxDDP**(硬约束原生)+ **Δt-as-control**(ALTRO 的 `τ=√Δt`);或 **Hauser**(Klamp't)两段式 | **前置必答题**:必须先把剂量判据换成**逐点、可微**的约束。这是立项前的设计问题,不是实现细节 |
| **做基线** | MuJoCo MPC | 终态只能软代价、无时间最优性。**做基线不做答案** |

**分工铁律不变**(`robot_centric_timing` §八):TOPP / 时间律只管"何时到、多快过"(幅值 + 时刻);
方向 / 空间不合归适配器的拍面 / 路径 morph 轴,不归时间律。

### 3.3 行程轴挂哪一环

- **归属已拍板**:`docs/TIMELINE.md`(07-09 晚四)原文——"**path morph 轴由此定形:引拍加深手术**
  (最小侵入:只动瓶颈关节、触球行逐位锁死、限位内)"。
  ⇒ **适配器的新轴(路径/行程 morph),与时序轴平行**;击球高度轴是"加轴"的先例(NOW.md 2-2)。
- **守卫不生成引拍,只消费行程派生判据**:每 clip 行程账本 `L` + 题目 `v*` → `a_min` → 对预算比对
  (或直接用 v3 min-time 表);超了才路由到 `extend_stroke` 或标不可行。
- **接口对齐的缺口**:`stage1_synthesis` §5 就绪度表把"时间律工具与适配器接口对齐"记为未做。
  **行程轴应在这次对齐里一并定签名**——`T_avail` 作为一等输入,两段式移挥时间律(NOW 2-3b)同批。

---

## 4. 空白与贡献切口

跨六个子域(乒乓 / 网球 / 羽毛球 / 棒球 / 高尔夫 / 冲击感知操作 + 击球 DMP)检索后:

1. **触球终态硬约束**:已被充分研究(Kober 移动目标 DMP;Koç free-time OCP;MIT MPC 硬终态)。
2. **击球时刻**:从固定虚拟击球平面(MoMP、HITTER 自陈局限)演进到自由击球时刻(Koç)与事件触发(DeepMind、Sony)。
3. **全身 / 平衡**:近两年爆发(HITTER、SMASH、ETH 羽毛球、四足全身 MPC 乒乓)。
4. **min-time 最优性**:在击球域**基本没有**。注意 Koç 是 *free-time* + *min 能量*,**不是 min-time**。
5. **⭐ 行程 / 引拍深度作为"所需触球速度"的显式函数:全域空白。**
   - **DMP 族**:幅度 `=(g − y₀)` 由**击球点几何**定;速度靠**移动目标末速 `ṫ` 与时间尺度 `τ`**。
     ⇒ **幅度与速度在参数化上正交**。这是空白的技术根源。
   - **OCP 族**:引拍是优化的**隐式**副产物,没有名为 stroke-length 的决策变量。
   - **RL 族**:摆幅**涌现**或由参考模仿而来(HITTER 只喂两条参考);从未被显式参数化或报告为
     "速度 → 引拍幅度"的映射。
   - 唯一系统讲清"引拍越大 → 击球越快"的是**人类生物力学**;机器人侧没有对应的显式建模。

> **贡献切口**:把 stroke length / backswing depth 抬升为**按 demanded contact speed 计算并规划**的一等量,
> 并给出它的**下界判据** `a ≥ v*²/(2L)` 与**手术工具** `extend_stroke`。
> 这条在击球机器人文献里没有先例。(限定:无法穷尽工程/专利文献;上表若干条目仅搜索摘要。)

**同时,原综述的"缺口结论"仍成立**:没有现成工作同时覆盖
「人形 + 平衡约束 + 触球终态硬约束 + min-time」。最近的组装路线:
(a) **aligator** 硬终端约束 DDP + Δt-as-control;
(b) 外层路径搜索 + 内层 **Hauser / TOPP-RA** 带接触时间压缩(AVP 思想补完备性);
(c) 解析下界与在线内核用 **Ruckig / Webb–van den Berg**。
乒乓域先例 Koç-Peters 2018 证明"自由击球时刻 + 触球终态硬约束"在线可解,缺的只是人形平衡那一层
(**以及 min-time 本身**)。

---

## 5. 引用验真台账

方法:每条主引用独立 WebSearch/WebFetch;争议项由本人复核原始页定案。
`scripts/feasibility_oracle.py`、四篇本库 PDF、仓库工具/分支/commit **全部实地核对**(pypdf 实抽原文 / pod1 ssh)。

**总计 49 条:已验证 41 · 有出入 6 · 待查 2 · 找不到(删)1 · 纯编造 0。**

### 5.1 须改写(6 条)

| # | 引用 | 出入 | 处置 |
| --- | --- | --- | --- |
| 1 | **Serrien & Baeyens 2018** | ①题名漏副标题 ": Prospects for Talent Detection?";②**−118/−57/−34/−9/+6 是「峰值角速度时刻」(Table 4),不是「启动次序」**;启动时刻是 −330/−268/−247/−94/−68(Table 3);③连带把肩肘顺序讲反 | 已改写(§2.1) |
| 2 | **Bobbert et al. 1996** | "高 18–20%" 错;原文为 **CMJ 比 SJ 平均高 3.4 cm**(≈9–10%) | 已改数字(§2.2) |
| 3 | **FATROP** Vanroye et al. | 是 **IROS 2023**,不是 ICRA 2023 | 已改(§1.2) |
| 4 | **Ruckig** | "开源版只支持目标速度"不成立:**Community(MIT)版已支持任意目标加速度** | 已改(§1.3) |
| 5 | **arXiv:2410.05681** | **+47% 投掷距离属实(正文)**;但「在前摆峰值附近释放」**无据**——全文唯一 wind-up 表述是 "**Arm-only** policies also generate momentum by utilising a winding-up motion",且为 RL **涌现**非"显式采用" | 删除释放时序主张 |
| 6 | **Pham 2014 / AVP 1411.4045 / Webb 1205.5088 / MuJoCo MPC** | 四处题名不精确(详见 `B_research_log.md` §4.2) | 已在 §1 表内标注 |

### 5.2 降级"待查"(2 条,不作论据)

| 引用 | 为什么 |
| --- | --- |
| **Kim 2011**, MMT 46(4):438–453(PII S0094114X10002168) | PII/期刊/作者/卷期真实(题名漏副标题 ": Sidearm and maximum distance");但所称结论「最优力矩解自发出现 初始 backswing + 肘弹簧蓄势 + 末端快速加速」**未能证实**(ScienceDirect 正文 403) |
| **Liu et al.**, arXiv:2511.11218 | 论文与三阶段课程属实;**项目名 "Phybot" 未证实** |

### 5.3 删除(1 条)

| 引用 | 为什么 |
| --- | --- |
| 网球「躯干旋转 ≈ 10% 拍速」 | **溯源失败**。Elliott 综述(Br J Sports Med 2006, PMC2577481)Table 2 **不单列躯干旋转百分比**——列的是"肩"(正手 15% / 发球 10%)。该数字疑为二手转述的近似说法,**不进定稿** |

### 5.4 前任标"未验证 / 存疑",本卡验为**真**(3 条)

| 引用 | 结论 |
| --- | --- |
| **Natherson & Scheeres 2024**(doi:10.2514/1.G008118) | **非编造**。Crossref + Semantic Scholar + AIAA ARC 三源一致:*Reachable Set Computation with Terminal Velocity Constraints*, JGCD **47(12):2560–2572, 2024** |
| **arXiv:2603.01128** | **非编造**。摘要原文 "emulates the energy-storage mechanism found in **froghopper** legs";**17.1% 垂直跳高**属实。全名 *A Deployable Bio-inspired Compliant Leg Design for Enhanced Leaping in Quadruped Robots* |
| **ISB 2013 乒乓海报** | 真实存在(ISB 官方服务器,海报 ps1-12d) |

### 5.5 本库盘点(research_result_2)的元陈述失真

| 前任声称 | 实地 | 判 |
| --- | --- | --- |
| 四篇 PDF 在 `/Users/Franco/Dropbox/乒乓/nohope/papers/` | 该路径**不存在**(pod 是 Linux);真实位置 = **`<repo>/papers/`**(仓库根,git 已追踪) | ❌ 臆造 |
| "本次直接 **pdftotext** 核对了原文" | `pdftotext` 在 **pod1/pod2 均未安装**;pypdf/PyPDF2/fitz/pdfminer 亦均无 | ❌ 不成立 |
| `topp_mintime.py` 在 `origin/topp-v3-mintime-0709`(**e8efe5a**),7 CPU 单测绿 | 分支 HEAD 已推进到 **fc39641**(对抗复核修复),单测 **9+24** 全绿;仍未合 main ✅ | ⚠️ 陈旧 |

> **但其内容主体经本卡用 pypdf 实抽原文复核,逐条为真**:HITTER 的 94 帧/1.88 s、第 43 帧(0.86 s)、
> `x=−1.37 m`、GVHMR→GMR、30→50 Hz、0.75 m/0.8 s;SMASH 的 400 条动捕、1.08 s(前后各 0.54 s)、
> Motion-VAE、双筛、**Eq.10 检索特征只含位置**(`p_rel_hit = p_hit + ε − p_anchor`,
> `i* = argmin‖p_rel_hit − p^i_target‖₂`)、`v_hit/τ` 仅作 task command 进观测、奖励窗 0.02 s / 0.1 s。
> ⇒ 其核心判断「四篇里"引拍多深、加速多久"从未被显式建模」**成立**。

---

## 6. 未决 / 待办

1. **`a_max` 口径(阻塞快筛的 soundness)**:`a_min` 是拍点笛卡尔量,预算是关节空间 `|q̈|` 包络。
   需 Jacobian 保守映射,或直接标定**拍点切向加速度包络**。**在此之前守卫的 reject 不是 sound 的。**
2. **`L` 口径钉死**:建议 `L_deep`(与 `extend_stroke` 手术对象一致);两口径在 v5 反手差 0.21 m ⇒ a_min 8.2 vs 11.7。
3. **精解层立项前必答**:剂量判据能否换成逐点可微约束?若不能,自由路径 min-time 一路(aligator / ALTRO)无法直接上。
4. **仿射性残差检验(廉价、应尽早做)**:对固定 `sᵢ` 三探针评估 `mj_inverse`,与仿射模型作差,
   核对残差是否**只剩** `D q′(s)·√x`。顺带钉死 `frictionloss` 在 MuJoCo 里走 efc 约束还是 passive 力
   (这决定它能否并入常数项 `c(s)`)。**此检验不做,§3.2 的一切修法都只是纸面推演。**
5. **`extend_stroke` 消融(§2.4 预注册)**:固定 ΔL、只改反向段速度剖面 ⇒ 预测 a_min 与 oracle 剂量都不动。
6. 待查两条(§5.2);Saccon 组 impact-aware 一族仅搜索摘要,若要引须补验(注意该组在 **TU Eindhoven**)。

---

## 附:主要来源

TOPP-RA arXiv:1707.07239 · toppra `hungpham2511/toppra` · Drake `Toppra` ·
TOPP arXiv:1312.6533 · `quangounet/TOPP` · Bobrow-Dubowsky-Gibson IJRR 4(3):3–17 ·
Kant & Zucker IJRR 5(3):72–89 · Pham & Stasse hal-01138098 · Hauser IJRR 33(9):1231–1250 · Klamp't ·
AVP RSS 2013 (10.15607/RSS.2013.IX.052) · `stephane-caron/avp-rrt` ·
CPC arXiv:2108.04537 · `uzh-rpg/rpg_time_optimal` · ALTRO IROS 2019 ·
Crocoddyl `loco-3d/crocoddyl` · Opt2Skill arXiv:2409.20514 ·
ProxDDP/aligator T-RO 2025 41:2605–2624 · `Simple-Robotics/aligator` ·
FATROP IROS 2023 · arXiv:2303.16746 · OCS2 · MuJoCo MPC arXiv:2212.00541 ·
Horizon (Frontiers Robotics AI 2022) · Koç-Maeda-Peters RAS 105 (2018) · `RobotLearning/traj-gen-and-tracking` ·
Webb & van den Berg arXiv:1205.5088 · Ruckig arXiv:2105.04830 · `pantor/ruckig` ·
Haddad & Halder arXiv:1909.12498 · HJ 综述 arXiv:1709.07523 · DeepReach arXiv:2011.02082 ·
Natherson & Scheeres JGCD 47(12):2560–2572 (2024) ·
Kober et al. ICRA 2010 pp.853–858 · Mülling et al. IJRR 32(3):263–279 ·
Büchler et al. arXiv:2006.05935 · D'Ambrosio et al. arXiv:2408.03906 ·
Dürr et al. Nature 652:886–891 (2026) · HITTER arXiv:2508.21043 · SMASH arXiv:2604.01158 ·
PACE arXiv:2509.21690 · Nguyen-Cancio-Kim arXiv:2505.01617 ·
Ma et al. Sci. Robot. 10, eadu3922 (2025) · Liu et al. arXiv:2511.11218 · Nguyen et al. arXiv:2510.08754 ·
Putnam J Biomech 26(S1):125–135 · Serrien & Baeyens J Hum Kinet 63:9–21 ·
Bobbert et al. MSSE 28(11):1402–1412 · van Ingen Schenau et al. J Appl Biomech 13(4):389–415 ·
Bańkosz & Winiarski JSSM 17(2):330–338 · Genevois et al. JSSM 14(1):194–202 ·
Zhang et al. Appl Bionics Biomech 2017:4780160 · Chen et al. arXiv:2603.01128 ·
Munn et al. arXiv:2410.05681 · Zeng et al. arXiv:1903.11239
