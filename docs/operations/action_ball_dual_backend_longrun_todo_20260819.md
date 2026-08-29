# ActionBall 双后端长跑：当前执行 TODO

> 状态：`corrected-dual-fresh-running / active-flight-profile-complete / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-29
>
> `origin/main:docs/NOW.md` 是全项目唯一优先级权威。本页只维护
> [FullMDP](../DEFINITIONS.md)（完整球路、击球、落点与恢复状态机）单动作双后端
> successor 的依赖顺序、运行事实和完成条件，不维护竞争性的优先级队列。旧的单动作执行页已转为
> [只读历史账](action_ball_single_action_dual_backend_todo_20260817.md)。

<a id="fullmdp-v6-todo-current"></a>

## 0.6 2026-08-29 current：同一 A3P0807、robust physical birth 与可解释的双后端差异

本节是唯一现役局部执行合同；`origin/main:docs/NOW.md`仍是项目优先级权威。当前V9两个长期namespace、
exact source和证据root继续只读，不hot-patch、不resume、不复用。R8只有在下列独立事实闭合后才替换它们；
有限probe、nominal hold、rate或文档状态均不单独授权长跑、promotion、physics parity、部署或真机。

### 已证根因，不再归为“Pod整体装坏”

- V9 MuJoCo实际选择的是legacy `a3_pingpong/a3_pingpong.xml`（SHA-256 `70c4fd65…36c0a`），而本轮
  teacher、runtime plant和Jiayi所谓0807对照需要`a3p_pingpong_0807/a3p_pingpong_0807.xml`
  （`7bbda723…bcae1`）。run name里的`0807/genesisfix`不是plant身份；当前probe已让错误root在import
  边界直接拒绝，并要求所选root在导入所有Mu合同前成为唯一环境绑定。
- pinned IsaacLab/Kit、sealed RSL、USD、mjlab和MuJoCo-Warp都能在Pod1构造、训练并连续写ACK；因此没有
  generic installation corruption证据。曾有ignored EPA48/RSL3恢复缺口，但它在首ACK前fail-closed，
  不能解释运行后的动作响应。
- R8 artifact中心导出的`512×H48×31` tape已经排除joint order、actor center、decoder和qdes：
  initial q/dq逐位相同，root/racket位置只差`7.8e-7/4.1e-6 m`量级，每tick qdes最大差
  `5.96e-8 rad`；但首个20 ms后q/dq已差`.00973 rad/.89084 rad/s`，tick47 joint/root/racket最大差
  `.13028 rad/.07872 m/.05263 m`。Isaac编译后clock为`.005 s×4=.020 s`，Mu为
  `.001 s×20=.020 s`、Euler/Newton/iterations100/noslip0。这把剩余原因收敛到编译后的
  plant/controller/integrator/contact响应，
  不是“动作输入对不上”；它仍只是first-divergence定位，不是physics parity裁决。
- 该响应层不是抽象未知：Isaac使用`ImplicitActuatorCfg`，由PhysX drive在`.005 s`物理步内求解位置/速度
  目标；Mu在每个`.001 s`子步显式计算并clamp `kp*(qdes-q)-kd*qd`。两边Kp/Kd与effort limit数值虽对齐，
  离散控制实现并不同。更直接的是Isaac当前`friction`为无量纲、随传递空间力缩放的PhysX系数，Mu的
  `frictionloss`是常值库仑关节力矩；`agibot_a3.py`已明确把这组相同数字标为未校准legacy port。
  因而同qdes后的首步分叉有具体plant/controller语义来源；Jiayi本机只有提交同asset、同actuator backend、
  同friction/contact/clock收据后，才可称与Pod实验匹配。

### 出生点合同的自查纠正

teacher frame0继续是不可改写的模仿目标，但当前0807 plant下它不可作为physical birth：仅
`waist_pitch`静态hold就需要约`-49.155 Nm`，而可执行position-control authority约`-21.704 Nm`。
正确结构是`immutable teacher reference + robust-feasible physical birth + policy-learned bridge`，不是
瞬移到老师，也不是让历史seed变成权威。

首版R7搜索虽名为`robust constrained`，实际fallback默认只要求数值上刚好大于零；新工件的右脚接触、
support与torque slack只有`6.98e-6 m / 5.68e-5 m / 1.263 Nm`，低于代码自己声明的robust reserve。
这是一项实现错误，不应靠训练时hard-edge Gate兜底。R8已改为：精确frame0只有满足全部具名reserve才直接
采用；否则旧ready只作确定性optimizer start，搜索每一项都达到同一reserve的固定可行域，再只在域内最小化
root/31q/racket到teacher的距离。fresh backend cache-miss重验和真实PhysX nonterminal prefix继续保留；
它们分别验证静态plant事实与有限运行事实，不代签学习成功。

### 学习与课程判读

课程仍为`balance → mimic → hit → landing`的自然重叠，不增加硬Stage。上一阶段开始基本成形时，下一阶段
必须已有非零eligible分母：mimic按playback行的teacher-achieved paddle误差，hit按
`selected contact / launch`及target误差，landing按`opponent landing / selected contact`。缺eligible写
`未测`，已有分母的零结果写`0/denominator`；return和安全计数都不能代替这些行为证据。

`observed_at=2026-08-29T10:47:30Z`的只读刷新已不是“太早看不出”。Mu ACK0..1610的first10→recent10
episode length/return=`135.289/15.644→165.950/18.878`，长期position/velocity改善但face/long-axis恶化，
且最近相邻10窗四项mimic都轻微变坏；累计launch/R03-valid/raw/selected/crossing/legal/recovery-success=
`217,463/178,009/7,637/662/596/0/0`，所以hit极稀疏，landing=`0/662 selected`、recovery=
`0/132,376 eligible`。Isaac ACK0..560的episode length/return=`97.255/10.946→151.409/15.860`，mimic
位置/速度/长轴长期改善但face仍坏于起始；`51,300 physical launch`后R03-valid/selected=`0/0`，故
mimic→hit交接已失败，landing因selected分母0仍为`未测`。两端fault/nonfinite/conservation均0：工程运行
健康不等于课程成功，自然重叠预期尚未满足。

R7弱裕量工件的MuJoCo 61-update profiler-off canary只达到rate/health证据：p50/p90=
`6.963/7.021 s/H48`，first10→recent10 episode mean=`131.66→129.20`，hard-edge=
`5.389%→2.323%`，但paddle四项误差均略恶化，contact=`0/launch`、landing=`未测`，所以不能称mimic成功。
Isaac同工件canary的p50/p90=`18.45/27.233 s/H48`，first10→recent10 episode mean=
`111.85→113.16`、hard-edge=`6.13%→6.79%`、contact=`0/1,242 launch`，terminal几乎全为
`base_fell_tilt/robot_hit_table`；这正是“仅正数裕量不是robust出生”的反例，只作冻结失败，不代表R8。

R8 MuJoCo 61-update profiler-off canary已自然完成：p50/p90=`6.962/7.018 s/H48`，first10→recent10
episode mean=`135.31→138.44`，actual-hard-edge rows=`4.96%→8.73%`；paddle position/velocity误差
`.1745→.1841 m / 1.156→1.217 mps`变差，face/long-axis `.3210→.3047 / .2605→.2548 rad`
略好，contact=`0/6,523 launch`。这是finite、可迭代、非灾难启动，但不是mimic成功或长期学习裁决。

R8 Isaac同口径61-update窗也已自然完成：p50/p90=`21.455/27.455 s/H48`，first10→recent10 episode
mean=`97.26→107.96 tick`；paddle position/velocity/long-axis误差`.2724→.2512 m / 1.1946→1.0959 mps /
.4755→.4548 rad`改善，face `.2893→.3721 rad`变差。全窗due/playback/launch/contact=
`14,221/13,555/461/0`，actual-hard-edge joint-sample fraction=`5.93%→6.69%`，terminal以
tilt/table=`8,064/5,868`为主。它证明mimic与physical launch入口已自然重叠，也明确证明当前Isaac墙钟
不可接受；`0/461`只作早期hit读数，不能用61 updates覆盖长期学习时间尺度。

旧V9 Mu的长窗也暴露了证据口径错误：先前写的generic `racket_contact_rows`不是课程合同的
`selected_contact_rows`。最终冻结的`2026-08-28T22:43:14Z` recent50 updates `7053..7102`中，
due/reveal=`11,384/11,384`且每项mimic误差有`182,295`个playback-active样本，但launch/raw/selected=
`0/0/0`；episode length/return=`108.07/9.99`且`11,380/11,380`个episode含tilt，四误差=
`.3677 m/1.5798 mps/.9754 rad/.9556 rad`。累计launch/raw/selected=`809,422/7,153/188`，opponent
landing=`0/188 selected`。该窗相对前窗生存、return、position/long-axis与launch都退化，只证明错误plant
谱系曾产生极稀疏selected hit后又遗忘，不能移签R8，也不能把generic接触冒充当前击球能力。

同一时刻Isaac recent50 updates `2119..2168`的scheduled/due/admitted/playback=
`8,048/8,046/8,039/8,018`，launch/raw actor-pair contact=`7,632/0`；episode length/return=
`153.05/15.07`，terminal tilt/table=`4,532/3,496`，每项mimic denominator=`535,048`，四误差=
`.3010 m/.9259 mps/1.2981 rad/1.0928 rad`。相邻窗只有约`.8%--1.8%`弱改善，不能称mimic基本成功；
累计raw contact仍为`0/262,249 launch`。当前schema不生产selected-rubber，故selected写`not_produced`、
landing写`未测`。两端入口已开而hit未形成；V9最终按精确进程身份停止并保留root，不再继续等待。

Observation V3 `215/231`保持不变：现有四组同clock teacher-minus-achieved heading residual已经是对直接
paddle目标的最小可观测闭环；本轮没有新增冗余obs，也没有向policy暴露不可观测未来量。当前阻塞在physical
birth/control response，不在缺少另一个观测字段。

结构审计确认当前核心6文件约`68k`行，且`check_table_obstacle_scene.py`同时保留legacy exact-frame0
`200/800`、endpoint equality与R8 learned-bridge两套consumer；这会放大修改与审计成本。当前R8只消费后者，
所以不在短训中混入大重构。replacement取得连续ACK后，下一项维护工作是把`physical birth / immutable
teacher / bounded prefix`提成一个dependency-light typed合同，删除迁移完毕的same-writer回声与legacy
procedural blockers；plant/file identity、fresh live recompute、真实write、finite/joint/table、WAL/fsync等
跨事实源边界必须保留。目标是减少owner和Gate，而不是放宽必要事实。

### 当前唯一执行顺序

1. [x] 固定V9只读学习前缀并核对实际plant，而不是相信namespace名称。
2. [x] 修正Isaac唯一reset owner、所选Mu root导入顺序、fixed tape单一actor-center真源与编译后clock/options遥测。
3. [x] Pod1已重物化全部13项达到具名reserve的R8 local candidate；真实PhysX
   `60 policy / 240 physics / 1.2 s` prefix双足接触率`1.0`、无terminal/当前或substep hard-edge，
   最小最终hard gap=`.21966 rad`。它只证明稳定learning prefix；frame0拍心误差最多`.23289 m`，不冒充mimic。
4. [x] 用R8工件重跑Isaac/Mu固定tape与两条`512×H48×61` profiler-off rate canary，逐端报告balance、
   mimic、launch/contact/landing分母、hard-edge、terminal mix和p50/p90；双fixed tape与双rate均已自然完成，
   短窗不判学习成败。
5. [x] 逐轮未决row压缩虽通过完整bitwise parity，但Pod profile2得到D05 question累计
   `98.72 s/12 updates`、collection中位`16.17 s`，相对旧dense profile的`18.16 s/8.05 s`严重回归；小batch、
   多次动态`nonzero`与kernel launch破坏GPU并行，已撤回。密度`3,584/290/22`保留为事实，不再推出该算法。
   首个非法shadow slotted Physical leaf的profiler也已删除，两个失败/归因root均只读且不作速度PASS。
6. [x] 保持单一mask-first compaction和三轮一次性dense compose；Pod exact CUDA已确认
   `virtual_ball.coarse_landing`的100-step Triton融合真实启用，`N=512`仅`.0526 ms/call`，不是当前墙。
   真正热点是Physical horizon：discovery固定`30 ticks×4`次eager reverse RK4，finalize又从同一contact state
   重算最多`30×4`次；`[512,3,3]`实测分别=`202.2/54.1 ms`，与旧dense D05 question约
   `245 ms/call`一一吻合。第一步trajectory cache/gather已在exact Kit/Torch2.7 RTX5090通过：`4,608`行
   全部admit、final batch全字段逐bit相等，matched reference/cache总耗时=`101.263/50.534 ms`，finalize=
   `51.471/.317 ms`，cache=`3.164 MiB`、该段peak增量=`7.321 MiB`，隔离测试=`201 passed,5 skipped`。
   cache候选的matched discovery仍约`50.2 ms/call`；第二步融合现已在exact Torch2.7 RTX5090闭合：actual
   `4,608`行、边界/非有限/重复identity
   probe及production record都逐bit相等，reference/fused/production=`49.591/.401/.416 ms`，约`124×`
   leaf speedup，fused peak增量=`3.551 MiB`；Pod组合门=`203 passed,5 skipped`。固定输入已否决降低
   solver12轮；`ef673014`只把数值叶放进单个有界`A=512` CUDA Graph，完整执行证据见
   [课程实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#single-a512-cuda-graph)。
   fresh profile已把D05 question从`61.18`降至`21.72 ms/call`；profiler-off p50/p90
   从`17.175/21.913`降至`15.135/19.779 s/H48`。随后`954200d5`把Reward28 dispatcher绑定与paddle/R06
   重复计算收敛到单cycle，Pod CPU/CUDA聚焦=`278/50 passed`；reward span从`28.03`降到`24.87 ms/call`
   （`-11.3%`），但整轮p50/p90=`15.280/19.654 s/H48`，与上一版基本持平，约6秒目标仍未达。
   `7ce9f120`又把regularization的construction-static joint geometry绑定一次，Pod组合=
   `123 passed,5 skipped`、exact CUDA lean Reward=`52 passed`；reward span继续降至`22.59 ms/call`。
   同GPU、同`512×H48×61` profiler-off的50个measured update为`14.880/19.364 s/H48`，相对`954200d5`
   只改善约`2.6%/1.5%`，仍远离方向目标。下一纯执行候选只把Reward热路需要的Epoch before-image收成窄
   immutable snapshot；public journal、14项付款chronology、carry、fault和durable边界全部保持，不缩
   RK4/solver、不增加per-call receipt或“成功即安全”Gate。
   `c53d3b31`已按该边界接线：`N=4096,K=1`理论写量从完整clone加late gather的`8.453 MiB/control step`
   降为19-tensor snapshot的`2.000 MiB`，没有删除14份payment image或公开账本。Pod exact
   Epoch+Reward=`160 passed`、CUDA Reward=`52 passed`，其余regularization/runtime/install分进程=
   `8/36/24 passed`；profile reward span=`22.59→21.22 ms/call`，collection中位只改善约`.8%`。整轮
   profiler-off p50/p90=`14.740/19.150 s/H48`，相对`7ce9f120`仅改善约`.9%/1.1%`；receipt/log
   SHA-256=`048b819b…5524`/`98d04235…e540`。故当前只接受结构/局部执行结果，不声称iteration目标已闭合，
   也不再沿同类静态微优化堆复杂度。
7. [x] 最终clean source=`954200d5beb770d9622e922aabff508b6181409a`通过exact测试、profile与rate后，按
   PID/startticks/PGID/cwd/source/namespace停止V9并保留旧root。正确0807的fresh长期Mu使用
   `fullmdp-r12-0807-mujoco-rewardpack-954200d5-20260828T2254Z-r2`（GPU2；首个同名root因ignored
   `meshes/`缺失在首ACK前fail-closed且未复用），fresh Isaac使用
   `fullmdp-r12-0807-isaac-rewardpack-954200d5-20260828T2245Z`（GPU0）。启动验收Mu ACK0..7明确绑定
   plant SHA `7bbda723…bcae1`。`observed_at=2026-08-29T05:18:37Z`时Isaac ACK0..1172、Mu ACK0..3421；
   两端Reward28、finite、conservation与attributed fault均clean。Isaac first50→recent50的episode
   length/return=`105.07/11.30→1311.79/125.64`，paddle position/velocity/face/long-axis最近误差=
   `.24872/.85594/.47811/.30615`；recent50 due/launch/contact=`4,920/4,882/0`，
   累计contact=`0/131,132 launch`，wall p50/p90=`19.425/19.832 s/H48`。生存和部分mimic量仍改善，
   但face比first50恶化、actual-hard-edge joint-sample已到`14.747%`，故不能称balance或mimic基本成功。
   Mu first50→recent50的episode length/return=`136.78/15.64→1310.24/132.50`，四项最近误差=
   `.15259/.87934/.38563/.20259`；recent50 due/launch/R03-valid/raw/selected/legal-landing=
   `5,248/5,096/5,066/3/0/0`，累计selected=`48/430,287 launch`、legal landing=`0/48 selected`，
   wall p50/p90=`6.264/6.392 s/H48`。Mu recent50 actual-hard-edge/qdes-guard rows已到
   `93.139%/93.380%`；episode/return几乎完全被边界使用污染。当前不是“太早看不出”，而是已有足够分母
   判定学习异常：balance/mimic未基本成功、hit仅约`.0112%`且最近窗为0、landing为`0/48`。继续跑只保留
   诊断时间序列，不提供“再等就会自然晋级”的理由。
8. [ ] 闭合仓库可部署边界；不再把历史Jiayi本机附件当仓内前置条件。第一层环境审计确认：
   `origin/build_4@324e60d1`没有环境lock，其path-autodiscovery在当前Pod1默认命中Python3.10、Isaac Sim4.5、
   `/workspace/IsaacLab@21f71363…`、RSL2.3.1；当前受控run则是Python3.11、Isaac Sim5.1、
   `/opt/IsaacLab-8320e0be`、RSL3.1.2。Pod1随后从远端clean clone `e3ef4e98…`，显式绑定exact
   Isaac/Kit/IsaacLab/Python-site/USD/RSL/GL/GPU后通过dry-run、`52 passed`和真实`512×H48×31` Kit/PhysX
   fixed-action probe，0 done/time-out，退出后GPU释放。receipt见
   [`action_ball_isaac51_fresh_clone_deployment_20260829.json`](../../configs/action_ball_isaac51_fresh_clone_deployment_20260829.json)。
   因此该版本“repo + 该Pod已合法准备的exact外部runtime”有历史PASS；纯`git clone`不包含EULA二进制和
   private assets，也不应伪装成自包含安装。后续复核判定`e9823e90…`把已在G05/现役Pod成立的一次性
   EULA/隐私setup事实错做成per-run显式flag，已删除这两个重复Gate。Jiayi原环境说明实际仍在Pod1，已按
   原文SHA吸收Isaac zip身份、安装顺序和83项非editable exact constraints；通用`setup_train_env.sh`也已
   删除静默path discovery，只接受显式runtime路径。Ubuntu Noble GL来自系统`libopengl0/libglu1-mesa`，
   launcher改为核canonical file/direct SONAME并记录观察SHA，不把单一发行版字节冒充跨机学习Gate。
   `4cd30d63`真实Kit fixed-action已0 done/timeout且action/state SHA保持，launcher本地36项通过。
   `e14b7141`又把Pod1 Python3.12.3的133项Mu基础解析闭包作为tracked lock；fresh venv dry-run精确解析
   `131 package + pip/setuptools`，lock SHA=`6e26d1e0…26bb1`，report SHA=`92021f56…05e47`。production仍由
   run-local EPA48/RSL3覆盖，不能让ambient RSL5冒充执行真源。Isaac launcher也不再替人硬编码隐私同意：
   machine-local provisioning必须给出EULA=`Y`与privacy=`Y|N`，映射到child后不泄回launcher环境。
   因此public软件配方和重建命令已闭合；合法Isaac下载、split USD、A3P0807的92个Mu mesh和private凭据仍是
   外部输入，item 8只为这些真实字节保持开放，不再笼统写成“Python/GL不可恢复”。
9. [ ] 做behavior-preserving瘦身：提取typed physical-birth consumer，删除已迁移的exact-only/self-echo
   procedural branches，并用相同artifact/固定tape/receipt反例证明语义未变。当前四个主文件合计约
   `13,167`行，问题不是行数本身，而是construction事实、runtime动态量、诊断身份与durable账本仍有交叉；
   每次只切一个owner边界，不做全文件重写。Epoch每step约`5.95 MiB`的44次record clone和14份payment image
   只有在journal bytes/chronology/carry逐项保持后才可收敛，不能为追求简洁删掉独立可消费证据。
   Build4的环境path-autodiscovery是同类结构债：procedural fallback同时决定Python、IsaacLab与ABI，却没有
   唯一运行身份；当前branch已用显式runtime输入替换该authority，而不是再加启动后Gate。
   第一刀已删除fresh FullMDP的R07→Motion ready self-echo：fresh Motion从自身可观测task/teacher event推进，
   没有读取该opaque capability；R07 publish/critic和legacy消费者保持。较大的R06 settle/retire融合与Epoch
   snapshot窄化暂不采纳，先等active-flight matched profile证明其收益并逐项保持journal/reason/carry。
   当前本地与Pod exact分进程聚焦测试均为`35/24/12 passed`（另`1 skipped`），Isaac fixed tape的
   action/state SHA与父版逐字一致。第一刀已验证，但完整typed owner清债尚未完成，所以item 9仍开。
   第二刀按全仓AST/callsite审计删除三个无生产consumer的兼容decoder/view及两个只为它们存在的facts类型：
   Physical/R06不再各自重读并clone完整Epoch slice，Epoch也不再暴露未消费的active D05 rows；源码净减约
   214行。该刀不在active调用图上，预期不改变墙钟；其价值是缩小owner/API面，仍须Pod exact相关套件确认。
10. [x] 已在GPU1隔离、严格加载Mu `model_2000.pt`做`512×240`随机policy轨迹诊断；optimizer未加载，当前
    GPU0/2长期run未改。`8,081/122,880=6.58%` policy rows有actual hard edge，`8,373/122,880=6.81%`
    有qdes guard，非零hard joint只有`waist_pitch/waist_roll/left_ankle_roll`三项；其中`77.47%` hard rows在
    outcome-settled/recovery phase。`waist_pitch`全部撞上限，但撞边时mean action=`-.551`、nominal qdes约
    `-.325 rad`，明确朝远离上限方向；因此“raw qdes越界/策略正向顶住”已被否决，首因更像显式PD、惯性、
    recovery目标或actuator response。receipt见
    [`action_ball_fullmdp_mu_model2000_jointdiag_20260829.json`](../../configs/action_ball_fullmdp_mu_model2000_jointdiag_20260829.json)。
    `qdes_limit_barrier=0`不是dead term：processed qdes已被soft inset投影，actual `joint_limit`仍有付款。
11. [x] 用同checkpoint与固定随机action tape只追三项关节的`q/dq/qdes/tau-before-clamp/tau-after-clamp`，覆盖
    launch→outcome→recovery边界，并与Isaac implicit-drive同字段对照。先判断是actuator mapping/符号、扭矩
    clamp/积分还是recovery reference过激，再选择controller、action scale或连续reward的最小修复；未闭合前
    不盲加权、不把hard edge改成Done，也不停止当前长期run。仓内probe现已接到**真实plant owner**：默认
    training path不分配trace，只在显式diagnostic模式记录每个`.001 s`子步的raw/clamped torque包络与真实
    q极值；首跑会同时落固定action tape，后续controller反事实和Isaac都复用该tape，避免闭环policy把plant
    差异重新混进输入。首次exact调用在首个真实plant step暴露诊断Tensor归并API错误且trace误接训练ledger；
    修正只用公开functional API并删除该非必要依赖，不改plant；失败root不计trajectory证据。随后fresh
    model2000 exact trace已排除torque clamp，并钉住旧guard把向内raw qdes改写成
    风险同侧endpoint、制动不足；v2只在共享纯tensor owner输出单边maximum-inward target。下一步必须复用
    NPZ SHA `4b843a4a…40382`内的action tape证明hard下降且不产生新故障，再做有限学习；否则不替换长跑。
    最终fresh reason replay继续复用同一action tape，三关节hard `8,715→171`（`-98.04%`）、torque
    clamp仍0；done `498→510`全部有生产reason解释（`base_fell_tilt=390`、`base_too_low=9`、
    `robot_hit_table=120`，可重叠），`joint_qdes_forbidden=0`、unknown bits=0、done-without-reason=0，
    NPZ SHA=`d09b650d…9ead9`。reason替换故障疑点已闭合，进入finite学习与Isaac同字段响应验证。
    修复后同源码Mu `512×H48×61`有限学习窗已自然完成：hard-edge=
    `19,816/1,499,136=1.322%`、qdes guard=`8.211%`，确认实现修复进入真实训练；但recent10 hard-edge
    `.739%→1.312%`的趋势回升，launch/raw/selected/legal-landing=`6,594/0/0/0`。所以controller首因item
    闭合，不把有限窗写成学习成功，也不据此盲调reward。
12. [x] 修正portable completion consumer仍期待旧Take061 UID的漂移：producer/live owner为当前0807
    `2552478955674699`，consumer旧值`5527597793770800`会让任何正确完成件末端失败。只更新独立consumer
    expected value，保留writer→consumer单侧漂移会红的测试，不用producer输出反写expected。
13. [x] 将本轮结构减法提交并在Pod1全新exact checkout分进程复跑lean-runtime、post-physics与Mu keepout；
    随后重跑同源码Isaac fixed tape和Mu profiler-off rate，只有source、receipt、测试与有限run一致后，才启动
    fresh双端长期replacement。长期判读按自然重叠课程分别报告episode/mimic、due/playback/launch、
    raw/selected contact与legal landing分母；上阶段基本形成时，下阶段必须已经出现入口增长，但不新增
    “成功后才允许开始学”的自动Gate。Observation V3保持`215/231`，除非证明有新的可观测、非冗余状态。
    新机恢复同时关闭HANDOFF遗留的`cryptography`手装欠账：package声明`>=44,<51`，兼容已验证的Isaac
    bundled `44.0.0`和Pod操作venv `50.0.0`；外部Isaac/EULA/private assets仍按显式SHA恢复，不伪装纯Git自包含。
    clean `75373daa` Pod exact=`35/24/12 passed`（另`1 skipped`）、signed-authority=
    `59 passed / 0 skipped`、Isaac fixed tape=0 done/timeout且SHA一致；Mu 61-update p50/p90=
    `6.671/6.740 s`、hard/guard=`.789%/5.727%`、launch/contact/landing=`6,001/0/0`。最终依赖source
    `d8fd8423`又在新exact checkout恢复并核过全部external input，重复上述测试；双端replacement已分别在
    GPU1/GPU2取得连续ACK，旧`954200d5`双run按exact进程身份停止并保留全部数据。到
    `2026-08-29T08:07:38Z`，新Mu/Isaac分别到ACK143/56，finite/conservation/fault clean，contact为
    `0/16,501`与`0/377 launch`；课程入口已开但学习不晋级。故item 13的source→test→run→replace闭环完成。
14. [ ] 将active-flight Physical→Epoch→R06逐物理子步事务收敛成最小typed状态转移。50-update exact profile
    （receipt SHA=`66d4a700…87bcf`）显示collection全窗p50/p90=`11.096/17.178 s`、recent10=
    `17.178/17.746 s`；无active flight时Physical postphysics均值`.014 s/update`，全H48 active时为
    `5.010 s/update`，高于同轮D05/reset的`2.650/1.863 s`。先用分段或fixed capture确定Epoch full-record
    clone/journal、R06 settle/retire与scene write各自份额，再只融合最大块；验收必须覆盖固定tape、
    first-observed/contact、settled/retired、reason/fault/counter、scene state、journal/WAL/ACK与profiler-off
    matched active strata。禁止删除engine事实、full-key/generation、finite/joint/table或用降低solver凑速度。
    第一轮只做同一owner内的masked mutation融合：`572a7080`用原地`masked_fill_`和显式拒绝storage alias的
    `torch.where(..., out=destination)`替换`full_like→where→copy_`临时张量；CPU/CUDA dtype/alias反例及
    exact Pod累计`210 passed`，`512×H48×31` fixed-action action/state SHA与父版逐字一致。matched
    50-update full-active两轮collection=`17.9318→17.6757 s`（`-1.43%`），Physical=
    `5.0127→4.7754 s`（`-4.73%`），R06 settle=`2.2841→2.0637 s`（`-9.65%`）；改动正确但远不足约6秒，
    不以继续堆同类微优化关闭本项。第二轮`a43aae3e`让R06 publish只消费本物理步产生的one-shot typed
    settlement delta，不再逐子步扫描历史mailbox；retained audit/current hot path共用唯一fact pack/projector。
    exact Pod相关套件`323 passed`；其中`25eb4c7a`删除Torch 2.7 CUDA event必须挂到CPU parent的错误测试
    假设，改验最小profile真实kernel数/名称与清零语义。三类完整CommitEntry canonical SHA新旧一致。
   Pod1空闲GPU同进程12组交替微基准median=`1045.170→771.048 us`（`-26.23%`），
   故采纳该结构减法。current-source真实Kit整轮profile在已建立Pod runtime authority下继续，未测前不得把
   局部收益外推为约6秒。其同源profiler-off `512×H48×61`现已自然完成，50个measured update的
   p50/p90=`14.720/19.377 s`（receipt SHA=`1d7508bc…415c`）；它复现上一版约14.74秒平台，证明仍远离约6秒。
   50-update full-active profile已自然完成，receipt/log SHA=`1070902b…d672` / `06fd6e72…41e3`；全窗
   collection p50/p90=`11.139/16.747 s`，recent10=`16.755/17.551 s`。recent10 inclusive
   `post_physics_publish=4.694/5.296 s`、`physical_epoch_postphysics=4.295/4.673 s`，其中
   `r06_postphysics_settle=1.939/2.108 s`、active capture=`.545/.596 s`、R06 facts publish=
   `.483/.526 s`、retire=`.362/.393 s`、Epoch refresh=`.309/.337 s`。本项继续开放：下一刀将四个子步的
   narrow PlantFacts保留在device并由一次有序typed scan结算，不能漏掉瞬时contact/crossing，也不能继续堆
   gate或用profile-on秒数代签整轮。
   `79efd71c`完成该刀的最窄版本：每个physics substep仍立即Scene capture、R06 sample与terminal park；只把
   score、mailbox copy、settlement prepare和retire收成control boundary一次，随后严格按
   `P0,R0,P1,R1…`回放四个CommitEntry。R06只新增cause/crossing不可推导事实和单一candidate replay，不镜像
   ordinal/observation/fault，也不缓存四份OutcomeRows。`255df4a1`同时修复测试harness在collection阶段替换
   canonical module、导致合并套件27个假失败的结构债。Pod1 final clean exact checkout分组为核心
   `286 passed,19 skipped`、lean+Scene合并`145 passed,8 skipped`、launcher/setup=`45 passed`、cfg=
   `13 passed,25 honest skips`。GPU fixed tape、journal/WAL逐字对拍与profiler-off matched active strata仍因三卡
   均被三条只读训练占用而未测，故item 14不关闭，也不先声称速度收益。
   当前相对`79efd71c^`的production净增为`460-21=439`行；本轮只抽出唯一candidate-grid换mask primitive并
   删除一个死参数，未合并语义不同的即时/deferred事务。虽没有增加外部Gate/owner，也不能仅凭语义测试称为
   最终优雅结构；先用GPU
   profile裁决收益，若不足以覆盖维护成本则撤回，若保留则再把legacy即时finalize与deferred finalize的重复数学
   收进同一内部primitive。后续性能依赖顺序已按profile固定为：(a) adaptive-sigma关闭时把command metric
   D2H从48次/update收为一次；(b) selected reset保留唯一preflight而让各owner只写`K`行；(c) Reward28同周期
   staged pack。三者整段硬上界分别`.978/1.863/1.066 s`，都不是预期收益且不能线性相加。已实测回归的逐轮
   unresolved D05 compaction继续拒绝，不因`3,584/290/22`稀疏度再次引入host/device控制流。
15. [ ] 修复direct-paddle经济轴并fresh双端验证。无条件`4/4/4/2` Mu到ACK623后已最终判负：matched
   ACK0..623相对旧baseline，raw/selected/crossing=`56/5/2` vs `1,665/334/281`；ACK614..623新run的
   position/velocity/face/long-axis误差=`.50920/1.31197/.74428/.61476`，四项都坏于旧baseline的
   `.28529/1.12387/.39936/.50190`。在`5,079,000` playback rows和`84,582` launch后，它同时破坏mimic
   与mimic→真实hit，因此不是“还早”或engineering fault，不能采用，也不再等待1000。其结构问题是同一个
   最高`14`也在容易的pre-playback ready hold持续支付，
   policy可在第一次动态teacher出现前先强化错误局部经济。

   当前最小后继为[`Reward28 playback-scaled paddle`](../DEFINITIONS.md#fullmdp-reward28-playback-scaled-paddle)：
   manager weight恢复baseline `1/1/1/.5`，只在Motion-owned playback-active行把同核乘`4`；H48、PPO V6、
   Observation V3、due cadence、plant、regularization、事件与安全事实均不变。它不增加Stage/Gate，仍是
   balance→mimic→hit→landing自然重叠。host静态验证后必须过Pod exact Reward/配置/fixed-action，再以fresh
   namespace做matched训练；三卡当前仍由三条只读run占用，禁止为取卡signal、同卡混跑或复用namespace。

   clean exact source=`ff7a6c4f`已完成Pod CPU/Torch组合门`444 passed,36 skipped`和launcher/setup
   `98 passed`。首个候选的旧weight断言与scaled-payment污染normalized kernel telemetry两项失败均已修复；
   validated training checkout=`8a57a522`又在fresh exact目录恢复并核过Mu mesh/EPA48/RSL3，Mu/Isaac完整
   launcher dry-run均RC0。current-source真实Kit/MuJoCo CUDA fixed-action因三卡占用仍为`未测`，不能据
   CPU门或dry-run勾选本项或发训练。

   Build4的mandatory actor warm-start仍是最强混杂项，但现有`model_2000.pt`明确
   `checkpoint_authority=false/resume_authority=false`。当前FullMDP也没有贯通env/owner/WAL的resume consumer，
   所以不写extractor偷用；未来若实现“全env重置后的numerical continuation”，必须另用新schema、新namespace
   和parent SHA，且不能冒充step-exact resume。本轮不为此再造一套臃肿状态机。

<a id="fullmdp-v9-superseded"></a>

## 0.5 2026-08-28 historical：V8反例与V9同源重叠课程替代源

本节已由§0.6取代，只保留历史证据；它不改变`origin/main:docs/NOW.md`的统一优先级。V5 source
`39f9481950a660e198dedac7fd402806d648906b`及其namespace保持只读，禁止hot-patch、resume或复用。
`9d333b0b`（语义`ba7225b2`）只是Observation/PPO最终裁决前的predecessor；V6 clean/pushed source
`caddecb76727ea55b0ce089453eea91cb5a9f8ea`的两个namespace已经被大分母证伪并在精确身份复核后停止；
run root与证据保持只读。V7 exact source=`1d33130ba07288918aa73d1323e1106303b7cad1`已经完成Pod聚焦验证并以
两个fresh namespace运行，但最新固定前缀已把它翻转为learning negative；它已在replacement ready后按
exact身份停止，root继续只读且不得resume。V8 clean source=
`0ad85ae1dfae13f617dc102a15bf99dba6b9ebf6`已完成Pod目标测试、双rate与fresh双端启动。后续只读大分母
再次得到两端合计`1,089,548`次launch、selected contact仍为0；同时源码/运行输入审计确认V8把Take058
teacher、Take061 ready、旧动态ready artifact和被YAML覆盖的split-ready混在一起，不能再用step数裁决
这条污染谱系。当前[`V9`](../DEFINITIONS.md#fullmdp-v9-candidate)只修同源ready→teacher、原子catalog绑定、
upper-only mimic与课程交接。clean exact source=`eb57233b4522d527455a0cbd7c547eb2ec49a68c`的MuJoCo/Isaac
fresh长期replacement已分别在GPU2/GPU0取得连续durable ACK；随后V8按exact身份停止，旧root/checkpoint只读
保留且未伪造completion。所有运行仍为
[`diagnostic_unauthorized`](../DEFINITIONS.md#diagnostic-unauthorized)，不得由
短验、单轮wall或文档状态推出promotion、physics parity、export、部署或真机授权。

### 为什么不能继续等V7

V7两端已有大量due、launch与R03 physically-valid分母，但`selected contact=0/launch`；
因此landing没有eligible分母，仍是`未测`。这说明balance/survival与task exposure已经改善，却没有形成
mimic→真实接触的桥；此时再把零接触解释成“step还不够”没有因果依据。冻结数字、正确分母与Build4对照只认
[课程实验§10](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#10-2026-08-25-currentv5反例收口与v6最小桥修复)。
V7只保留作当前learner失败的反例，不裁决固定LR后自然课程的成败。

课程顺序仍是**balance → mimic → hit → landing**，但交接判据不是硬Stage：上一阶段开始基本成形时，
下一阶段必须已经有自然非零分母。mimic开始成形而launch/contact仍无分母，或hit基本满足target而landing
仍无分母，才是交接实现故障；在此之前缺失格写`未测`，已有分母的零结果写`0/denominator`，不靠return
均值遮掉。

### V9 replacement唯一合同

| 面 | 当前合同 | 第一性原理理由 |
| --- | --- | --- |
| learner | [`PPO V6`](../DEFINITIONS.md#fullmdp-ppo-v6)：`512 env × H48 × U100000`、save2000、E5/MB1、fixed LR`1e-4`；`gamma=.99`、GAE`lambda=.98`、entropy0、learned`log_std`、fresh sigma`.05`不变 | 相对V5保持总transition、24,576-row minibatch、总optimizer step与按transition save cadence，但policy刷新快4倍；V7两端checkpoint均已卡LR=`1e-5`且mimic大分母零contact，所以移除per-minibatch adaptive KL。它是显式算法/速度取舍，不冒充等价热路优化 |
| Reward | [`Reward28`](../DEFINITIONS.md#fullmdp-reward28)：14项lifecycle + 6项common mimic + 4项measured-paddle composite prior + 4项action/joint连续成本 | Paddle固定50/50 precision-exp + coarse-Cauchy，物理宽度为`.075/.30 m`、`.50/2.0 mps`、`15/60 deg`、`10/40 deg`；另恢复`action_rate_l2=-.1`、qdes soft=`-10`、projection=`-1`、actual joint soft=`-10`，用连续目标修复硬边套利，不新增Done或安全Gate |
| Observation | [`Observation V3`](../DEFINITIONS.md#fullmdp-semantic-observation-v3)：actor/critic=`215/231`，在common prefix加入4×3同clock、全phase heading residual；V2 `203/219`只作旧checkpoint ABI和paired control | Reward28的paddle objective仍依赖独立measured Motion teacher；把同producer的teacher-achieved最小残差交给actor，关闭source未直接可见的representation gap，避免从q/dq重学FK/Jacobian；不加入raw ball/aim/rate/history/action ID或声称完整Markov |
| cadence | 六次真实due固定为tick `48/233/418/603/788/973`，相邻185 tick包含task close、77 tick recovery与2 tick hidden gap | 一整轮H48先提供balance梯度；首个mimic曝光仍位于既有60-tick nominal-hold实测窗内，不把1.2秒收据外推成5.9秒，也不新增Stage Gate |
| Isaac terminal overlap | `ResetTelemetry × D05 scheduled-due`只在既有CPU pre-optimizer drain做跨writer交集 | 保留独立事实源的证据；绝不新增每步Gate、D2H、owner或same-writer receipt |
| evidence | Isaac milestone schema8按具名字段解码；Mu update/completion/summary wire为`10/5/6` | 新wire增加paddle真实误差finite/sum/sumsq并显式升版；Isaac当前N1按aggregate报告，Mu由pinned action identity精确归属；不把新字段伪装成旧schema兼容 |
| Mu keepout | fused table-keepout直接消费`data.struct.xpos/xquat`的native Warp `wp.array` `vec3/quat`，不要求Torch中转或host sync | 真实mjwarp state就是事实源；adapter应贴合引擎表示，而不是要求环境迁就测试fixture |

### 对Build4的辩证裁决

mandatory Build1 `model21800` bit-exact actor warm-start是Build4相对fresh run最直接的已证配置差异；
因此Build4不能被当作fresh-from-zero证据。缺actual model0 receipt且配方混杂，早期行为也不能独立归因给
继承。V6只采用其
**直接连续拍面目标必须由policy可推断**的通用原则：Reward28直接实现拍面objective，Observation V3则是
本轮独立的最小correction-state实现，必要性仍待paired control。两者使用同一measured producer闭合
reward/action信息。Build4本身没有独立actor normal或achieved-residual字段，所以不能替V3证明必要性；
本轮采用Build4“任务应在早期连续出现”的原则，但不抄其混杂数值：首个due改为tick48来自本系统H48
rollout与60-tick nominal-hold收据，而不是从Build4的`1--1.3 s`反推。warm-start、replay、双learning-rate、
sigma`.19`和`14/14/5`数值均延后；formal/physical/73-action说法与checkpoint复用均拒绝。commit与证据
只认[课程实验§10.3](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#103-从build4学原则不抄证明与数值)。

### 结构减法与不可删除边界

结构目标是`PlantFacts → ActionBallState transition → StepTelemetry`。同一writer echo/hash、无人消费的
Gate和独立sealed immutable artifact后的重复source recheck只是**逐callpoint删除候选**：必须同批证明两个
操作数确由同一writer产生、seal后不可变、没有跨writer守恒价值，并保留mutation反例与必要telemetry，才可
删除；不能把“TOCTOU”三个字当成泛化豁免。plant finite/joint/table、full-key/generation、跨writer事实
守恒、optimizer成功边界、run-owned artifact no-clobber、WAL/fsync/ACK与GPU lifetime lock继续保留。
本轮已经删除无production consumer的`keyed_epoch_work`链和optimizer后same-writer identity echo。
旧V6热路的固定全量record clone静态为`714.375 MiB/H48`、10,560次`Tensor.clone()`；V7已经删除
Motion callback同writer重读并缩窄返回值，理论减少`285.75 MiB/H48`和4,224次clone。该payload不能
换算成秒；Pod active wall另行实测，不为这两份复制增加新Gate。

静态结构审计也确认债务不是抽象观感：`scripts/train.py`为22,426行，`_apply_task_overrides`单函数
3,401行、`_run_with_environment_close_owner`为1,778行；`ActionEpochOwner`单类3,743行。Reward24整包
替换漏掉四项连续成本就是跨层procedural override的真实反例。后续结构目标固定为code-owned typed learning
spec、pure transition/reward kernel、薄backend adapter和offline evidence consumer；先把逐shot JSON/Python
重放移出active ACK。active V8 exact checkout保持只读，不为整理文件热改训练，也不新造自证Gate。

V9又给出两个更直接的结构反例。其一，catalog attachment安装了motion/family/phase/sign，却漏装同一catalog
中的`clip_names_per_clip`，导致motion `N=1`而action identity `N=0`；正确修复是一个typed catalog对象一次性
原子安装并在安装边界核完整shape，不是在每个下游consumer再加相等性Gate。其二，dynamic-ready bootstrap
先正确桥接physical ready→teacher frame0，随后legacy config override又覆盖同一状态；正确修复是单一bootstrap
owner，而不是再加一个“覆盖后检查”。这两例都说明复杂procedural wiring本身会制造需要防守的错误。

### 当前Pod事实与执行顺序

`observed_at=2026-08-26T19:27Z`的V7固定前缀不再支持“继续等”：Mu ACK0..9815、Isaac update0..4459
分别约`965M/438M` transitions；累计launch约`1,748,621/430,393`，selected contact仍均为0。Mu最近50
episode均长`1499.39/1500`，Isaac约`494.91`，说明balance至少在Mu已形成、两端mimic/playback也已有
真实大分母；但teacher-achieved position error仍约`.568/.704 m`，远非接触尺度，landing因contact分母为0
继续写`未测`。Mu/Isaac最新可读checkpoint optimizer LR都精确为`1e-5`；Mu约update500起已长期贴下限，
时点与真实due/mimic开始重合。active-strata wall最近50约为Mu mean/p50=`9.46/9.40 s/H48`、Isaac
console=`25.64/25.51 s/H48`。这些数字只裁决V7学习与迭代设计，不代签physics或formal安全；详细分母只认
[课程实验§10.9](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#109-v7结论固定lr在mimic曝光前耗尽)。

V8 exact Pod目标矩阵=`305 passed, 4 skipped`。61-update profiler-off rate的Mu
p50/p90/throughput=`3.796/3.999 s/6,455 transitions/s`，Isaac=
`6.835/7.612 s/3,598 transitions/s`。V8 fresh namespace为
`fullmdp-a-h48-v8-mujoco-fixedlr512-0ad85ae1-20260826T203329Z`与
`fullmdp-a-h48-v8-isaac-fixedlr512-0ad85ae1-20260826T203329Z`；启动快照Mu ACK0..35、Isaac ACK0..51均
连续durable，optimizer LR精确为`1e-4`。`observed_at=2026-08-28T07:12Z`的后续只读累计已到两端合计
`1,089,548`次launch而selected contact
仍为0；源码/运行输入审计又确认Take058 teacher、Take061 ready与legacy bootstrap覆盖混用，因此该谱系不再
用更多step解释，保持只读等replacement。

V9 exact active-schedule聚焦测试=`53 passed`。predecessor tick295的Isaac同源rate probe为
p50/p90=`6.635/7.153 s/H48`，但61轮全程due=0，只能裁决速度。改为tick48后，真实Isaac在update4已得到
`due/selected/accepted=512/512/511`、Take061 action identity `511/511`、playback started=`248`。完整
61-update窗累计`due/selected/accepted/rejected=18,432/18,432/18,419/13`、playback=`9,120`，所有
`18,431`个terminal均为base tilt，recent10 episode mean length=`81.916`；physical launch=0。paddle
position误差first10→last10为`.464→.496 m`，face/long-axis则`.995→.914`、`1.282→1.080 rad`，短窗方向
混合，不能称mimic成功。active-task rate p50/p90=`10.470/13.863 s/H48`，说明旧空业务6秒附近测量掩盖
了task热路成本；wall与due row数相关系数约`.771`，满512-row due的p50=`12.35 s`、空due=`6.85 s`。
这支持“课程入口已修”，不支持“学习已闭合”或“Pod安装损坏”。

同源码MuJoCo有限窗已自然完成：p50/p90=`6.644/6.854 s/H48`，scheduled/reveal=
`10,861/10,860`、launch=`6,658`、missed=0、R03 valid=`5,107/5,107`、selected contact=`0/6,658`。
episode mean first10→last10=`135.78→139.98`，paddle位置/速度误差未降，mimic仍未成功。首个Mu root因
ignored EPA48/RSL3未恢复在首ACK前失败，按固定SHA恢复后r2成功；只归类为asset sync缺口。Isaac全tilt/
零launch与Mu table+tilt/大量launch的分叉必须做同初态固定action first-divergence parity，不能直接归因Pod
安装，也不能被“原生physics不同”豁免。

`observed_at=2026-08-28T09:48:33Z`的长期replacement快照：Mu namespace=
`fullmdp-a-h48-v9-mujoco-genesisfix-eb57233b-20260828T093350Z`，ACK0..83=`2,064,384` transitions；
Isaac namespace=`fullmdp-a-h48-v9-isaac-genesisfix-eb57233b-20260828T094243Z`，ACK0..18=`466,944`
transitions。Mu首10→近10 episode mean=`137.31→141.96 tick`，近10 launch/R03/contact=
`1,210/855/0`，wall mean=`6.57 s/H48`；Isaac为`67.34→73.55 tick`，近10 D05 due=`3,414`但
physical launch/R03/contact=`0/0/0`，wall mean=`16.76 s/H48`。两端finite/conservation/fact边界均clean。
Mu的零contact已有早期launch分母但仍远小于预注册学习窗；Isaac尚处balance且下游为`未测`。不得按该前缀
补写“基本成功”阈值或把Isaac慢速包装成安全Gate。

V6的历史闭合仍如下；它不再是待继续到25k的学习候选。上一候选`72b87100`在真实Pod揭示Isaac schema
slice和Mu native pose adapter两处实现错误；最终
`caddecb7`已完成exact Pod CPU与两端`2048×H48×61` rate。Mu p50/p90=`5.468/5.526 s`达到约6秒
方向；Isaac=`7.81/8.38 s`仍未达到严格6秒。该变化包含PPO迭代尺度取舍，不冒充纯热路径等价加速。
失败分类、receipt、rate window与唯一详细数字真源见
[热路实验§16](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md#fullmdp-v6-rate-current)。

- [x] 冻结旧V5因果快照；不再用持续增长的step数掩盖`launch→contact=0`。
- [x] 冻结Observation V3 `215/231`：只加入四个同clock measured-teacher-minus-achieved heading residual；
  V2保留为旧ABI/paired control，不得作为fresh fallback。
- [x] 实现并原子绑定PPO V5、Observation V3、Reward24、四次真实cadence/耗尽sentinel、terminal-overlap跨writer合成、
  milestone具名slice、native Warp keepout与canonical test harness。
- [x] 在Pod1 exact clean checkout复跑分环境CPU矩阵；最终为`1,036 passed, 11 skipped`，并把一次
  launcher-like import边界失败单列后精确重跑，未吞成PASS。
- [x] pre-V3、pre-PPO-V5 predecessor的Isaac真实Kit 4096×H48 probe自然完成；它只证明predecessor
  runtime/rate。
- [x] 用真实MuJoCo GPU Full-A覆盖keepout、returned observation/lifecycle与RSL H48；随后同卡、同strata、
  profiler-off重验PPO V5 2048×H48 rate，报告p50/p90与transitions/s。
- [x] 最终source冻结后重验Isaac PPO V5；连续完成测量窗且Reward24/V3 obs/done/reason/counters/WAL
  全finite一致，并用短学习canary排除刷新边界回归。不得沿用4096 predecessor的wall结论。
- [x] 验证通过后从最终clean SHA建立两个fresh root/namespace；只在replacement ready后按精确
  PID/startticks/PGID/cwd/source/namespace停止旧Mu，绝不广域kill或复用旧root。
- [x] fresh两端取得连续durable ACK，Reward24/V3/finite/conservation/fault启动边界已通过；按
  balance→mimic→hit→landing继续判读，不以早期0 hit停车。
- [x] 冻结V6结论翻转窗：balance/survival已经接近episode horizon，两端launch/R03均有数十万以上
  独立分母而selected contact仍为0，actual-hard-edge也从早期下降翻成末段持续上升；当前V6不再按
  “课程尚未打开”继续解释，也不再把25k自然完成当作科学上必须等待的结果。
- [x] 实现V7最小因果包：Reward28四项连续成本、固定composite paddle prior、同producer真实误差矩；
  Observation V3保持`215/231`，四次真实cadence和自然重叠课程不变。
- [x] 完成结构减法：milestone按term批处理；Motion只传四字段窄projection和caller消费的bool；保留独立
  plant/full-key、optimizer、WAL/fsync/ACK边界。新增误差使Isaac/Mu wire显式升为`8`与`10/5/6`。
- [x] Pod1 GPU2 fixed synthetic action-shaped tape拒绝`cq_n_iters=8/4`：两者都改变admission reason、
  selected identity和target residual；保持12 iterations，不把学习风险冒充等价加速。
- [x] 本地集成聚焦矩阵=`606 passed, 34 skipped`，changed tree compileall与`git diff --check`通过；Pod1把
  14个受影响文件逐进程隔离重跑为`706 passed, 32 skipped`。一次把所有fake-module测试塞进同一解释器的
  `212 failed`只证明test pollution，未吞成PASS；隔离后全部生产相关文件通过。
- [x] replacement ready后逐一复核PID/startticks/PGID/cwd/source/namespace，精确停止旧V6。Mu两PID在TERM
  后退出；Isaac wrapper退出但同一Kit child在60秒后仍占卡，复核原startticks/namespace后只对该PID做
  SIGKILL。旧root未删除、未伪造completion。
- [x] source `1d33130b`以fresh V7双端重启：Isaac namespace=
  `fullmdp-a-h48-v7-isaac-reward28-obsv3-1d33130b-20260825T180216Z`；Mu前两个namespace在首ACK前分别因
  未恢复ignored EPA48/RSL3 pinned bundle fail-closed且不复用，按既有`setup_local_sync`恢复后现役为
  `fullmdp-a-h48-v7-mujoco-reward28-obsv3-1d33130b-20260825T180216Z-r2`。两端首批ACK均为28项、finite、
  conservation fault 0，Isaac milestone schema8、Mu schema10。
- [x] 冻结V7 active-strata wall与学习结论：Mu/Isaac最近50约`9.46/25.64 s/H48`，两端累计约218万
  launch仍零contact，checkpoint LR均为`1e-5`；不再用启动balance-only的`5.36/8.42 s`代表现态。Pod
  `/sys`只读、governor仍为powersave；该infra限制独立记录，不绕过、不与算法混合归因。
- [ ] 在首次使用“基本成功”作结论前，先冻结可复算的**证据判读合同**，不能看完曲线再补阈值：balance按固定窗
  survival-to-due与terminal mix判；mimic按task/playback eligible行上的teacher-achieved paddle残差与行为趋势判，
  `income/sample`只能作收入诊断、不能代替梯度或行为成功；hit按`selected contact / launch`与target error判；landing按
  `opponent landing / selected contact`判。最小分母、连续窗口数和置信精度须在下一固定快照前预注册，正式阈值只
  消费预注册UTC/ACK之后的未来窗口，不得把已看过的ACK102+回填成首个判定窗。该合同只约束报告与停车解释，
  不进入runtime、不成为安全Gate；四阶段从首个自然eligible事件起始终重叠开放。
- [ ] 等上一阶段开始基本成形时核验下一阶段已自然出现非零分母；100000自然完成、
  formal physics parity、transfer与部署继续是独立未闭合项。
- [ ] replacement按一个可审计的最小因果包修复，顺序固定为：
  1. [x] 把已有纯tensor action/joint经济接回shared FullMDP Reward；现役24项图整块替换了旧ActionBall reward，
     却没有Build1已证明关键的`action_rate_l2`，也没有`qdes_projection_penalty`、`qdes_limit_barrier`或
     actual `joint_limit`。FullMDP又有意不把actual-hard-edge变成Done，导致抖动/硬边只有telemetry、没有
     可学习代价。修复应复用既有action/q/qdot/qdes/limits，不新增owner、receipt或安全Gate；action、命令、
     plant和projection四项可在一个shared cost里分别记component。
  2. [x] 在既有paddle producer中累计teacher-achieved position/velocity/face/long-axis真实误差及平方和，按
     playback/action/side给分母；`income/sample`不再代替mimic行为证据，也不把这些误差加进actor。
  3. [x] 用固定的coarse+precision连续paddle kernel关闭`.70 m / 4 mps / pi rad / 1 rad`宽核可长期取得收入、
     却不必达到物理接触精度的目标漏洞；数值先以现有strike-success物理尺度和Build4原则构造
     counterfactual，不直接抄Build4的`14/14/5`、warm-start、replay或sigma`.19`。
  4. [ ] Isaac把每update的逐CommitEntry Python重放、逐row `.item()/.tolist()`与完整shot JSON移出训练ACK；
     active长跑只保留device compact counts、per-action/per-side分层与bounded fault exemplars，完整逐shot
     transcript只留给显式诊断。保留optimizer→WAL/fsync→ACK顺序，不为删自证流再加Gate。
  5. [x] 对PPO的per-minibatch adaptive-KL做独立算法复核。V7两端最新checkpoint optimizer LR均为
     `1e-5`；Mu从约update500起长期贴底，而真实playback随后才大量出现。采用fixed LR`1e-4`，不改
     Reward28/Observation/plant，先隔离“学习步长被早期balance占用”的根因。
  6. [x] 验证PPO V6：`512×H48×U100000/MB1/save2000`保持总transition、minibatch、optimizer-step和
     transition-save cadence；先做两端61-update profiler-off rate与短学习未来窗。只有replacement ready后
     才精确停止V7并fresh双端重启。若paddle真实误差仍不降，再独立测试Build4启发的强direct-paddle权重，
     不把LR、reward和obs三轴混成一个不可归因版本。
  7. [x] 把N1 catalog的motion与action identity作为一个原子合同安装；拒绝`motion N=1 / action N=0`，
     并由Take061同源ready→frame0 bridge成为唯一bootstrap owner。
  8. [x] 首次due从tick295改为tick48，保留六次机会与185-tick完整task/recovery cadence；不新增Stage或
     learned-success Gate。
  9. [x] 收完tick48的Isaac/MuJoCo同源61-update有限窗；两端都证明自然task exposure与finite/identity，
     但mimic误差未降且行为明显分叉，因此不直接启动fresh长期namespace。
  10. [x] 用自然退出的`--diagnostic-profile-probe --profile-updates N`（有限host-wall归因；见
      [`diagnostic_unauthorized`](../DEFINITIONS.md#diagnostic-unauthorized)）定位Isaac满due路径约5秒增量；
      profiler span含嵌套且不做CUDA sync，不作为正式速度证据。exact source=`39569c49`的12轮自然完成；
      `D05 total/question compose`中位=`1.999/1.724 s`，累计=`22.394/18.444 s`，question compose占
      D05累计约`82.4%`。preview/build/epoch-settle累计仅`.001/.092/.719 s`，下一算法刀只审计固定三轮
      question bank，不再优化这些小段或增加Gate。
      round-density进一步得到真实construction attempted round1/2/3=`3,584/290/22`，固定三轮数值
      row-round中约`63.8%`没有消费者；采用未决行incremental compose候选，拒绝直接降成一轮。修正后的
      exact source=`34cd7af8`已在fresh 12-update namespace自然完成，直接active-row计数仍精确为
      `3,584/290/22`，不是由inactive row倒推；该profile-on诊断只授权密度事实，不授权速度或学习。
  11. [x] 从同一physical-ready、joint order、q/dq与固定31-D action tape逐tick比较Isaac/Mu的base、racket、
      terminal与first divergence。首轮exact Pod已完成：source clean `981327de`、joint order/action tape相同，
      首差却在`initial_joint_pos[0,12]`；Isaac写asset default，Mu写dynamic-ready，初始joint/root/racket位置
      max absolute差=`1.5199 rad/.1778 m/.5376 m`，tick7起terminal也分叉。exact Kit确认RSL wrapper已消费
      canonical genesis reset，唯一根因是reset Event仍写asset default。当前Event已改为消费Motion窄
      physical-ready projection；曾尝试的第二个`train.py` reset被runtime正确拒绝并已删除。修后
      `179148e3`双端同commit record已闭合birth：initial q/dq exact、root max=`4.1e-7 m`、racket position
      max=`0.467 mm`，done/time-out逐格相同。但首版raw-zero tape距live Take061 actor mean最大`16.32`，
      实际测的是default-qdes阶跃且每8 tick全量reset，不能裁决fresh policy动力学。tape已改为同一tracked
      Take061 live actor mean中心的`±.02`扰动。clean `3343fe90` centered record两端均无Done，但Isaac
      joint/root/racket相对初态最大漂移`.349 rad/.092 m/.170 m`，Mu仅`.012 rad/.008 m/.010 m`。v3
      `joint_qdes`对账已完成：tick0--34跨端最大只差`5.96e-8 rad`而q在首个20 ms步已分叉；tick35后
      Isaac waist-roll先触发hard-inner guard才改写q_des。故shared decoder/action order闭合，剩余差异是
      backend plant/controller response，不再把physics parity错设成长训Gate。旧nominal-hold PASS也降级为
      60-tick nonterminal prefix；其末端已是`waist_roll=-.3205 rad, dq=-1.19 rad/s`，不得写成稳定ready。

replacement已完成exact Pod target tests、双rate、fixed-action first-divergence与V9双fresh长期启动；
fixed-tape的`cq_n_iters=12`结论沿用同一未变physics合同。hard-edge、paddle误差单调性、active-strata wall
和短学习未来窗仍未闭合，现役namespace保持只读source且不resume/不hot-patch；不得把启动ACK或rate probe
升级成阶段成功。

`ACK 0..101`启动快照已冻结：两端各`10,027,008` transitions且仍持有exact GPU/CPU affinity/flock；
Mu first10→recent10 episode length=`104.337→159.622`并已有scheduled/public/overlap=
`14/13/1`，Isaac=`87.502→129.229`且尚无due。两端common mimic与paddle-prior income/sample均提高，
nonfinite/conservation/fault/qdes-terminal全0；Mu launch=0故contact/landing为`未测`，Isaac task及其
下游全部`未测`，仍属上一阶段尚未基本成形，不是hit
失败。完整分母、namespace与wall只在[课程实验§10.5](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#105-验收边界)
维护，避免本页复制流水账。

后续[首次双侧launch与关节遥测固定窗](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#106-首次双侧launch与关节遥测裁决固定诊断窗)
只证明两端自然打开playback/launch/R03；当时“hard-edge正在下降、继续只读训练”的前提已被
[结论翻转窗](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#107-v6结论翻转生存形成但mimichit桥与joint经济失败)
推翻。当前两端qdes nonfinite/terminal、Reward nonfinite/conservation与fact/attribution fault仍为0，
所以证据数字可信；失败是学习目标/热数据流问题，不是用新安全Gate包住即可解决的问题。formal、promotion、
deployment与真机安全保持NO-GO。

## HISTORICAL / SUPERSEDED — 0. 2026-08-22 学习阻塞修复与 fresh 重启 TODO

本节是当时 branch-scoped successor 的顺序清单，不改变 `origin/main:docs/NOW.md` 的项目优先级。
旧 MuJoCo/Isaac H48 长跑已经在successor验证后精确停止并只读封存；不得hot-patch、resume或复用namespace。
两条fresh successor各自从独立clean checkout运行，不共享可变source。

- [x] 保存两条现役 run 的连续 ACK、finite、速度、episode、阶段分母和 Reward14--19 趋势快照。
- [x] 查清两个 backend 的 `motion_body_ori` 都长期只有约理论最大值 `0.3%` 的根因；区分 reference/frame
  错配、body 集合污染和指数 Reward 饱和，不能凭 aggregate return 猜修法。
- [x] 按 `HANDOFF_TO_CODEX_20260808.md` §3 四问审计 D05/R07：真实 safety invariant 与
  balance→mimic→entry→strike→landing 课程推进必须分离；退役为别种诊断 run 设计或按构造恒真的门。
- [x] 冻结 adopt/defer/reject：课程采用自然事件 eligibility；上一阶段开始可用时下一阶段立即已有分母，不新增
  冗余或部署不可观测的 actor observation，不用 task Reward 调权掩盖零 eligible denominator。
- [x] 实现最小可学修复，并保留现有 13 项 readiness 分解的更新摘要；诊断 telemetry
  不得伪装成 policy observation、owner、receipt 或新的 safety Gate。
- [x] 复核 PPO action-noise/entropy/adaptive-KL 实际曲线，明确采用或拒绝显式有界 schedule；H48/
  `lambda=.98` 保持已采用，不能把 rollout 变化冒充性能修复。
- [x] 完成 host 聚焦回归、exact Pod 测试、有限短验和必要的固定条件数值/行为检查；学习语义变更必须
  使用 fresh checkpoint lineage。
- [x] successor 就绪后保存旧 run 最终证据，按精确 PID/PGID/namespace 收口旧 MuJoCo 与 Isaac；确认
  GPU/lock/process absent 后，用两个 fresh namespace 同时重启。
- [ ] 验收新 run 的 durable ACK、finite、wall/throughput、六项 mimic 梯度和逐阶段 denominator；
  正式判断仍按 balance→mimic→entry→strike→landing，不以早期 `ACCEPT=0` 单独停车。

### 本轮冻结的 adopt / defer / reject

- **Adopt — 课程推进：** episode 前 `295` 个 policy tick（约 `5.9 s`）只学习可持续站立；活到首个
  due tick 且仍是有限、未终止的 row 就进入 task reveal。随着能活到该 tick 的 row 增多，mimic 样本自然
  与 balance 样本重叠；mimic 稍有成形后，同一个 task 自带的 ball flight/contact/landing 奖励立即可学，
  不再另设一扇会把下游分母清零的课程门。
- **Adopt — reference：** hidden balance 阶段的 joint/body reference 必须来自同一份 reset 后静态 ready
  tuple；reveal 后才原子切到 action frame 0 和 bridge。不得再让 joint teacher 要求 runtime default、而
  14-body teacher 同时要求 measured action frame 0。
- **Adopt — safety 边界：** finite、joint envelope、跌倒/碰台等真实 plant invariant 继续硬失败；R07 的
  13 项 recovery 误差只做已发生 shot 后的恢复奖励和可读诊断，不再授权 task 是否可以出现。
- **Adopt — learner：** 保持已经采用的 `rollout=48`、`lambda=.98` 和 fresh lineage；先修零分母与错误
  reference，再判断 exploration schedule，不能用噪声或调 Reward 权重替代可学性修复。
- **Reject：** 不新增 actor observation；当前 semantic actor 已有 base pose/velocity/heading、teacher、
  task geometry、阶段和倒计时。13 项误差来自 critic 可观测的 plant/reference，只需要摘要 telemetry，
  再喂给 actor 会冗余并扩大部署合同。
- **Adopt — orientation梯度：** 先修正reference，再对同一14-body角误差使用`.4 rad` fine与`1.0 rad`
  coarse等权核；coarse只恢复大误差区梯度，不改变target或制造额外状态。
- **Reject：** 不通过单独放宽 `motion_body_ori` 的 `std` 掩盖 reference 错配；不把 R07 all-of 阈值改名成
  “安全”；不添加 receipt/owner/counter/gate 来证明按构造恒真的事实。
- **Defer：** `solver_solve_many` 的更低 `cq_n_iters` 和其他 MuJoCo 数值性能取舍独立做 fixed-tape parity
  canary；它们不与本轮学习语义修复捆成一个不可归因的改动。

实现、证据与fresh验收的详细真源见
[2026-08-22课程解阻实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md)。

## HISTORICAL / SUPERSEDED — 0.1 2026-08-22 active-path简化与第二次fresh重启

本节承接上节，不改变项目优先级。现役Isaac在durable ACK 257之后由匿名CUDA
`_assert_async`毒化context，最终在下一次PhysX scene write显错并卡住；它已按精确PID/start-ticks保全现场并
停止。MuJoCo继续只读运行。第二次successor必须先闭合下列顺序，不能hot-patch旧source或复用namespace。

- [x] 保全Isaac最后完整ACK、run log/WAL/checkpoint和进程身份；确认context已不可继续训练后，只对精确
  child执行TERM，TERM两次无效才KILL，并验证child/leader/GPU compute app均absent。
- [x] 把ActionBall active-path的匿名异步assert改为单一、具名、device-local fault状态：无效行不得写PhysX，
  在已有control/PPO边界统一同步并报告具体reason；保留真正的finite/identity/joint/contact安全，不保留
  同一writer按构造恒真的echo检查。
- [x] 让Isaac与MuJoCo共用同一纯tensor `q_des` guard语义；finite fallback、soft/hard inset、state-dependent
  brake和终止位必须有反例测试，不能继续把MuJoCo的简单hard clamp写成已对齐。
- [x] 用构造期确定性几何证明slot0的teacher strike pose与question ball/contact tuple闭合；该证明回答
  “mimic成功后是否已经能碰球”，不新增actor observation，也不用随机rollout冒充几何证明。
- [x] 收敛Isaac post-physics同一调用栈内的重复full-grid clone/同写者校验；只删除同步consumer完成前不会
  被修改的副本，并用alias/mutation反例与fixed-tape检查保护真实跨owner边界。
- [x] launcher加入有界profiler窗口和GPU-local NUMA放置的显式参数；profiler只作归因，profiler-off同卡
  `10 warmup + 50 measured`才记wall/throughput，且必须同时报告raw H48和H24-equivalent。
- [ ] host聚焦回归、exact Pod隔离测试、Reward20/actor203/critic219/done/reset/reason/RNG/fault parity全部通过；
  学习语义改变一律fresh lineage。
- [ ] successor就绪后保全仍在运行的MuJoCo最终证据并精确停止；两个backend用新clean detached source、
  fresh run root和fresh namespace同时重启，至少取得连续ACK `0..4`、finite/fault0后才宣布切换完成。
- [ ] 按balance→mimic→entry→strike→landing继续报告逐阶段分母：上一阶段基本成形时下一阶段应已有非零
  exposure；contact/landing为零就明确写零或`未测`，不靠aggregate return掩盖。

本轮冻结的取舍：采用active-path状态收敛、共享q_des guard、确定性几何闭合与有界诊断；延后resume、
CUDA Graph、solver iteration和物理步长变化；拒绝先写C++、增加actor observation、增加same-writer receipt，
以及用Reward调权掩盖zero denominator。100-step RK4 fused CUDA/Triton路径已经存在，本轮只验证真实绑定，
不重复实现。

## HISTORICAL / SUPERSEDED — 0.2 2026-08-22 `aa42418b`性能闭合与监测

这是上节的实测收口，不新增优先级。219-D critic的no-key support/dwell语义已经变化，因此旧`661ff84b`
checkpoint只保留历史，不是本轮exact-resume父本。

- [x] 用细分profile确认旧主墙是no-key阶段读取Isaac ContactSensor两脚net force，约`7.41--8.03 s/H48`；
  chronology snapshot/store各只有毫秒量级。
- [x] no-key路径删除contact读取，只保留独立source/reset-generation/Motion cadence chronology；
  critic `[216:219]`保持宽度但按N/A填零。keyed R07仍读取真实contact并计算support/slip/recovery。
- [x] 删除同writer自证：R07 bundle不再重验已绑定ActionEpoch `snapshot_idle_*` getter identity；LeanRuntime
  production component callpoint binding与callee owner核验仍保留。D05测试不再手造Motion true输入再断言
  ACCEPT；独立consumer chronology、catalog tick295/cadence293和真实keyed反例仍保留。
- [x] exact Pod分进程回归=`304 passed, 6 skipped`；no-key contact `.data` sentinel、keyed recovery、
  Epoch、profiler、Motion bridge、Physical hot lane、env install与D05 transaction均覆盖。
- [x] 同卡H48 profile后`r07_idle_support_read=0 calls`；profiler-off匹配5轮中位`6.346 s/H48`，
  相对旧`14.194 s/H48`约降55%，且fault/CENSOR/nonfinite/conservation全0。
- [x] 新Isaac连续ACK后精确停止旧Isaac；补齐ignored runtime三文件manifest后启动新MuJoCo，连续ACK后再
  精确停止旧MuJoCo。最终两条active run都绑定clean source `aa42418b…`且是fresh lineage。
- [ ] 继续只读监测balance→action mimic→launch/contact→landing。两端均已验证row活过tick295时task exposure
  立即打开；当前零分母只剩launch/contact/landing并记`未测`。只有action mimic已基本形成而后续仍无launch/
  contact exposure，或hit已基本形成而landing仍无exposure，才判对应课程交接故障。
- [ ] MuJoCo早期qdes forbidden已在update26基本归零，随后robot-table与fall占比继续波动；Isaac仍以fall为主。
  两者都属于policy可学习行为，先按原因/分母观察，不绕过真实joint/table termination，也不增加“安全”门。

## HISTORICAL / SUPERSEDED — 0.3 2026-08-23 V3审计与fresh V4

本节只记录当前branch-scoped successor的依赖顺序，不改变`origin/main:docs/NOW.md`的统一队列。
[`fullmdp-a-h48-v4-*`](../DEFINITIONS.md#fullmdp-correction-lineage-v4)表示第四批fresh实现纠错lineage，
仍消费PPO V3配方；名称中的V4不是新PPO算法、阶段或promotion等级。现役V3 source/run保持只读，所有修复只能
进入clean detached successor、新namespace和新run root。

- [x] 保存现役V3双后端的进程身份、continuous ACK、finite/fault、H48 wall、episode与
  `survive→reveal/ACCEPT→playback→launch→R03→contact→landing`逐阶段分母；零格不得用aggregate return
  或几何证明替代。
- [x] 重新对账本页既有TODO与
  [`EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802`](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)：
  hidden阶段reset-ready joint/body reference、reveal后原子切measured frame0与bridge、以及teacher/contact
  几何闭合都已经是已采用设计；当前零launch/contact本身不是“又有一个reference偏移”的证据。
- [x] 复核203-D actor / 219-D critic合同：actor已有deploy-observable robot、teacher、task geometry、phase与
  clocks；本轮没有构造出“同一现有actor状态却需要不同动作”的alias反例，故不增加冗余、不可部署观测。
- [x] 按`HANDOFF_TO_CODEX_20260808.md`§3裁决实现问题：R03/physical/R07只付fresh source step，R06只在
  settlement后完整付一次；missed launch不得catch-up；第五个episode内不可完成的shot从共享cadence删除；
  invalid contact退休shot而不伪造Gym reset；q-des真实终止语义双后端一致。
- [x] 把portable MuJoCo missed launch变成具名pre-optimizer ledger fault；完整rollout storage在同一次已有
  host reduction中检查policy/critic observation、action/value/logprob/mu/sigma/reward/return/advantage有限，
  并检查done严格二值，不新增逐step同步。
- [x] 分文件完成host聚焦回归、consumer/launcher、`py_compile`与`git diff --check`；混合import-alias造成的
  exact-type假失败须分进程验证，不能为让测试绿而放宽production type identity。
- [x] commit后在exact Pod clean detached checkout恢复固定SHA的EPA48/RSL3 ignored资产，运行隔离回归与两个
  one-shot dry-run；Isaac必须使用目标GPU的NUMA-local CPU set，并由ready child的`/proc/.../fdinfo`证明
  继承同一GPU lifetime flock。
- [x] 只在live lane owner/queue再次对账后，以fresh V4分别替换现役V3；先取得连续durable ACK、完整finite/
  fault0、run-owned cache与source clean，再宣布切换完成。空闲GPU不等于已获授权，不借用未协调lane。
- [ ] 继续按自然课程报告：balance基本成功前允许下游样本稀少；但一旦mimic基本形成，launch/contact必须已有
  非零分母；一旦hit基本形成，policy-settled R06/landing必须已有非零分母，否则才判对应交接故障。

结构裁决也冻结在本轮：真正独立的finite、full-key/generation、plant terminal、scene writer、optimizer与GPU
lock边界继续fail-stop；task成功、每update必须出事件、R07 readiness、同一writer回声/hash、late catch-up都不是
安全Gate。当前Physical/Epoch/R06等owner文件已经过大且存在多份sticky fact/payment副本与import/type耦合，
这确实提高了开发和审计成本；但V4前只做已经有因果证据的减法，不先造巨型state、C++重写或新owner。
V4取得运行证据后，下一轮结构目标是“一条lifecycle spine + 两个backend StepFeatures adapter + 纯fact/payment
kernels + 一个telemetry funnel”，每次删除self-proof都必须同时保留独立mutation反例和fixed-tape parity。

<a id="04-2026-08-23-v4最终冻结与v5第一性原理自查"></a>

## HISTORICAL / FROZEN — 0.4 2026-08-23 V4最终冻结与V5第一性原理自查

本节是V5历史合同；当前执行只认§0.6。V4两条run均已停止并最终冻结；
V5已从clean `39f9481950a660e198dedac7fd402806d648906b`在Pod1以fresh namespace启动Isaac与MuJoCo，
仍只是`diagnostic_unauthorized` implementation lineage，不是新PPO算法、课程Stage、checkpoint兼容或
晋级等级。学习配方继续是H48、GAE
`lambda=.98`、E5/MB8、entropy 0、fresh learned `log_std`/init sigma `.02`。H48是学习取舍；旧“6秒”只表示
需要大幅算法/数据流提速的量级方向，不是形式Gate。

**V4最终冻结（2026-08-23；本段替换临时前缀，不再追加运行流水）：**

- 两条均绑定clean source `9e26afd3342e1da8643b225c987d4a3c91a3ff2f`；exact Pod隔离回归为
  `589 passed, 12 skipped`（skip均为需真实GPU的用例），Isaac与MuJoCo one-shot dry-run都通过。Isaac
  namespace为`fullmdp-a-h48-v4-isaac-correctness-9e26afd3-20260823`，portable MuJoCo namespace为
  `fullmdp-a-h48-v4-mujoco-correctness-9e26afd3-20260823`；它们只是fresh运行身份，均保持
  `diagnostic_unauthorized=true`，不是算法版本或promotion标签。
- Isaac最终连续durable ACK=`0..748`（749轮、`147,259,392` transitions）；update749在optimizer前报
  `epoch drain decoded a terminal overflow`，无ACK749且进程已退出。V4只有generic bool，不能唯一恢复
  具体row/cause或把它归成EPA容量故障。recent50 episode均长/return=`179.802/12.157`，wall mean/median=
  `9.128/9.100 s/H48`；D05 accepted/due=`270/300`、playback=`61`、launch=`1`、contact/R03=`0/0`。
  R06一次settlement没有contact/net/cross/table/common，R07/retire仍为0。
- portable MuJoCo最终连续durable ACK=`0..4798`（4799轮、`943,521,792` transitions），evidence在精确
  PID/startticks/PGID/cwd/source/namespace/GPU核验后冻结，旧进程随后TERM并确认退出；未自然跑满12500，
  因而没有`completion.json`。recent20 wall mean/median=`14.146/13.994 s/H48`；Reward/storage finite，
  命名fault、nonfinite和conservation为0。累计public due/reveal/defer=`1,637,789/1,637,789/0`、
  launch=`527,957`、R03 physically-valid=`145,814`，racket/selected contact与landing均为0。完成episode
  `3,376,589`，tilt/table/base-low terminal bit=`1,521,819/1,860,583/383`；bit可重叠，不能相加伪装
  独占分类。旧schema-4字段
  `racket_contact_eligible`只是launch的同写者别名，不再作为接触机会或分母；当前缺的是独立durable
  playback denominator。actor已有Motion phase，但playback成功分母继续`未测`，不得由reveal代写。
- V4只证明survival-to-due 295可达且mimic/hit输入会自然打开，不证明balance已基本成形；它同时显示旧实现的
  mimic→hit交接不符合预期。由于下述reference、typed identity与transition chronology实现错误已污染这条
  lineage，该结果不能裁决corrected V5的课程设计。V5已经fresh训练；启动验收时两个后端都还没有due/
  action分母，随后MuJoCo已自然出现public due与launch，见下述阶段里程碑。mimic成功的独立分母仍`未测`；
  Mu hit/contact已有`0/6 launch` diagnostic negative，Isaac contact与两端landing/recovery仍为`未测`。

**V5 live-source Pod、fresh启动与学习轨迹审计（原位替换，不作live流水账）：**

- 两条live run的source为clean/pushed `39f9481950a660e198dedac7fd402806d648906b`。Pod broad CPU/ABI矩阵=
  `792 passed, 57 skipped, 0 failed`；另以无`PYTHONPATH`的fresh进程复跑plant/runtime/runner/consumer=
  `77 passed, 0 skipped`。两组有重复验证目的，不相加伪装unique总数。Mu真实GPU direct五节点与真实
  RSL H48一轮=`5+1 passed`；Isaac CUDA projection/selected-rubber/runner-drain=`32 passed`，其中历史命名的
  runner-drain只签CUDA/RSL callpoint，不冒充Isaac集成；当前真实Kit fresh训练提供更强的端到端路径。
  双launcher dry-run均在建root前通过，且两个real root此前不存在。
- Isaac fresh namespace=`fullmdp-a-h48-v5-isaac-chronology-39f94819-20260823T144237Z`，物理GPU1。
  启动验收前缀连续durable ACK=`0..63`（`12,582,912` transitions），每轮`196,608=4096×48`；最近五个
  完整console wall=`7.78/7.47/8.17/6.79/8.19 s`，但不是matched-strata稳态速度。该前缀139,264个episode
  全因tilt终止，全部D05/Motion/shot事件为0，Reward finite、attributed fault与conservation均为0。
  后续只读到ACK97时，10-update mean episode length/return从首窗`87.606/7.052`升到末窗
  `97.776/8.705`；这说明balance开始学习，不等于balance基本成功，且仍没有due。
- MuJoCo fresh namespace=`fullmdp-a-h48-v5-mujoco-chronology-39f94819-20260823T144237Z`，物理GPU0。
  启动验收前缀连续schema-6 ACK=`0..8`（`1,769,472` transitions）；post-warm recent5 wall
  mean/median=`8.727/8.777 s/H48`，加权`22,528.7 transitions/s`。该前缀16,378个episode均为
  `robot_hit_table`，mean length=`105.21 tick`；所有scheduled/public due、launch、R03、contact、outcome、
  landing、recovery与retire均为0，Reward/storage/domain、Mu四项fact-integrity与conservation全绿，
  std=`.020000→.019945`。这同样只是balance早期，不以零shot分母裁决击球实现。
- **固定学习轨迹审计：**不逐ACK追写，当前只认Isaac `0..450`（`88,670,208` transitions）与MuJoCo
  `0..385`（`75,890,688` transitions）两个冻结前缀。Isaac
  已从`201..220`的局部低谷`120.451/10.792`恢复；最新`431..450`共`23,081`个episode，mean
  length/return=`170.249/15.625`，超过先前`170..189`局部峰值`160.279/14.398`。该窗全部因tilt结束，
  qdes/table/low/timeout、fault、nonfinite与conservation均为0；这证明此前回退是可恢复的早期PPO非单调性，
  仍不等于balance基本成功。累计due/public=`20/20`、ACCEPT/reject=`17/3`，已有`2`次真实playback，
  launch及其下游仍为0。
  MuJoCo最新`366..385`共`21,033`个episode，mean length/return=`186.693/17.771`；terminal为
  tilt/table=`11,314/9,719`，base-low/qdes/timeout均0。累计scheduled/public/terminal-overlap=
  `307/297/10`、defer=0、natural launch/missed=`6/0`、R03 present/physically-valid=`1/1`，contact、
  outcome、landing与recovery仍为0，fault/nonfinite/conservation全0。slot0的tick45 playback、tick76 launch
  已由自然run走通launch边界；Mu racket/selected contact均为`0/6 launch`，是小样本diagnostic negative，
  不再写`未测`。Isaac因launch=0，contact仍`未测`；两端selected contact=0，故landing仍`未测`。现在的
  未闭合点是独立mimic/playback成功与Mu launch→contact，不是Stage或launch断链。
- 两条run均为fresh、无resume/hot-patch/namespace复用，且仍`diagnostic_unauthorized=true`。Isaac早期wall
  已出现`6.79 s`单点，Mu早期也比旧V4末段快，但状态未匹配，不能正式归因算法提速；约6秒方向尚未达到。
  与上述学习前缀对齐的recent20为Isaac stdout辅助total mean/median=`9.488/9.465 s`、Mu durable wall
  mean/median=`9.235/9.223 s`，进一步证明单个6.79秒不能代称稳态。正式rate window、独立playback分母、自然
  mimic→hit→landing、12500 completion与physics/transfer parity继续未闭合。

**对齐既有TODO与MuJoCo native readiness后的裁决：** reference只有三个权威切点：
`fresh_bound && policy_opportunities_created==0`的**首次ACCEPT前** balance行使用同一reset-ready
joint/body tuple；D05在post-transition ACCEPT边界原子安装selected measured action frame0 joint/body并把它
放进返回的`obs_{t+1}`，第一笔task-conditioned action/reward发生在下一transition；后续recovery/ready参考是
上一个completed action frame0，不是回到reset-ready。这正是V5修复的旧Isaac偏移：旧fresh路径用
generic `in_hold`把joint留在default，却让body读frame0。它是与既有
合同相反的实现回归，不是需要新offset、新课程或新actor Observation的证据。current
`mount_normal_sign=+1` slot0的selected-face球心位置几何仍闭合，稀少launch或`0/2`真实失败
也不构成新位置offset证据。

actor 203-D / critic 219-D已包含fixed-center family A所需的deployment-intended robot、teacher、task
geometry与clock；真实table/root/velocity producer仍由G07闭合，也没有构造出同观测却需不同动作的
alias反例。现役`history_length=0`；对照系统中的`history=8`只作deferred候选，在无alias反例前不接入。
同样不把contact/support/fault/reward账本、未来oracle或其他冗余/不可部署量加入actor。N73
negative-face normal另有现有合同实现错，V5以raw +Y mimic/R03与signed physical face分离修复，
不扩Observation。

phase也不再混称：**Epoch phase**是shot业务carry-state（`IDLE→REVEAL→LAUNCH→OUTCOME→RETIRED`），
回答“这一球的事实账走到哪”；**Motion phase**是teacher/reference的播放与可见性状态，回答
“policy此刻应跟哪段动作”。合法launch可在同tick进入outcome；旧shot的outcome/R06/R07/retire也可在同一
surviving边界后被新reveal原子替换，因此final phase不能单独解释该transition的全部事件。outcome与natural
recovery必须互斥，DEFER只允许结算后仍busy的row。actor的两个5-way one-hot分别表达Epoch与Motion，
`task_valid`由Motion当前可见性决定，不用Epoch中保留到RETIRED的payload-valid代替。

**当前采用/延后/拒绝（其余历史表不再覆盖本表）：**

| 裁决 | 当前合同 | 为什么 |
| --- | --- | --- |
| 采用 | PPO V3：H48、GAE `lambda=.98`、E5/MB8、entropy 0、fresh sigma `.02` | H48是学习取舍；性能另按真实wall/throughput验收 |
| 保持 | semantic actor/critic `203/219` | 203-D已含建议的8维root状态：`base_position_table(3)+base_heading_table_xy(2)+base_com_lin_vel_heading(3)`；没有证据再造237-D |
| 延后 | `history_length=0`保持，`history=8`仅作候选 | 尚无same-observation/different-action alias反例；先不扩大部署与normalizer合同 |
| 采用 | 修正首次ACCEPT前/ACCEPT/recovery三段reference selector | ACCEPT在post-transition边界安装frame0到`obs_{t+1}`；旧Isaac selector与此相反 |
| 拒绝 | 新offset、新stage、新balance Reward或用R07 ready授权task | V4只证明survival-to-due与下游曝光可达；尚无因果证据要新增这些机制，也不等于balance或安全存活毕业 |
| 采用 | scheduled/public due分账，DEFER只表示结算后仍busy | 同tickterminal不能假报actor看见过reveal；自然RETIRED可立即ACCEPT |
| 采用 | Epoch/Motion解耦、typed launch identity与具名row fault | launch同tickoutcome与retire后同边界reveal合法；只拦真实跨owner错配 |
| 保持 | finite、pure timeout、真实plant terminal、full-key、optimizer与durable WAL/ACK | 这些保护可独立击穿的可信事实；task成功、R07 ready、stdout显示与same-writer echo不是Gate |

**V5 adopt / defer / reject（只写现态，不写实现流水）：**

- **Adopt — 两个可归因的性能方向：**MuJoCo合并base metric与Full-A的contact-buffer遍历；Isaac热consumer
  读取ActionEpoch窄投影而不复制dense `current()`。保持H48，最终源码须用
  fixed-tape/RNG/reason/counter/safety parity与profiler-off matched-strata wall验收。理论调用数、LOC、
  host microbenchmark与旧Pod结果都不代签V5速度。冻结到Mu ACK385的recent20把
  collection/learning/wall mean分成`9.056/0.177/9.235 s`，collection约占`98%`；同期Isaac ACK450
  对齐的stdout辅助recent20 total/collection/learning mean为`9.488/8.625/0.862 s`，collection占`90.9%`。
  因此下一轮不能靠改日志、减少Gate或微调PPO声称
  达到6秒，必须先在matched state量出physics/step/solver热墙，再对该墙做算法或kernel级减法。
- **Adopt — 单一事实消费链：**Isaac R03/Physical/R06 fault先进入共享ActionEpoch owner的36项
  `row_fault_bits`，MuJoCo只保留R03 nonfinite、R06 source-invalid、R07 sequence、R07 nonfinite四项
  per-transition cause；两套编号不混计，且只进各自既有唯一pre-optimizer drain。direct Physical/R06只认
  typed 8-field key/publication，由Physical唯一推进ordinal与previous/current ball centre；legacy plane隔离，
  不造假digest。Motion exact-resume保留reset-ready/pending状态与三段reference；negative-face把raw +Y
  teacher/R03 normal与signed physical face分开。这些修复不增加Observation、owner或业务Gate。
- **Adopt — wire只记录可界定的事实：**MuJoCo update/evidence固定schema6、completion保持schema5、consumer
  summary为schema5。`scheduled_due_rows`记录schedule命中，`due_terminal_overlap_rows`记录actor不可见的
  due+terminal，`reveal_due_rows`只记录surviving public due；删除与launch按构造相等的
  `racket_contact_eligible_rows`。fresh-prefix检查`launch≤reveal`、racket/selected contact链，以及
  R03/outcome/landing/retire各自的因果上界；launch同tick outcome合法。R03 exact-strike与contact属于不同
  clock，`selected_contact/R03-valid`只报描述性ratio，正式contact rate以launch为分母。
  `r06_common_per_eligible`才是closed task-landing成功率；`opponent_landing_per_crossing`只是crossing
  条件比例，不能代称总成功率，也不新增event或Gate。`invalid_contact + Gym done`只reset、不retire；
  aggregate consumer对未知overlap只验可证明上下界。当前`business_chain_complete`仍是producer逐row检查后的
  attestation加consumer边际聚合一致性，不是独立same-env/same-epoch重放；晋级前须增加keyed carry-state重算或
  可重放trace，当前只能称`producer-attested + aggregate-consistent`。
- **Adopt — runtime/authority边界：**单一path-free `runtime_stack`绑定EPA48、RSL3.1.2与MJLab1.5.3；
  正交`source_plant/runtime_attach`绑定base receipt、actual geometry/clock/capacity/owner-frame与run-owned
  augmented `runtime.mjb`，consumer在ACK前独立复验。launcher把已核GPU UUID与同一flock
  open-file-description交给唯一child。optimizer→WAL/fsync→owner ACK→EPOCH_ACK/fsync才是authority。live
  `39f94819`的stdout为regular file且没有失败，但审计发现snapshot/completion后的两处裸print在闭管时会把已
  durable状态伪装成进程失败；clean/pushed post-launch successor
  `a3c528f1b4c9b0a60f5cd3aeec28a11e990044b3`已改成best-effort structured warning并由closed-pipe反例覆盖，
  exact Pod fresh checkout全文件=`52 passed, 1 skipped`（skip为显式real-GPU节点）。该修复不改学习语义，也不
  hot-patch或重启live。
- **Preserve：**backend physics separation、独立measurement/plant verification、finite与真实plant terminal、
  full-key/generation、optimizer成功边界、durable WAL/ACK与失败后sticky fail-stop。这些保护可被独立反例
  击穿的可信事实，不因追求简洁或速度删除。
- **Defer：**per-ordinal Reward snapshots、重复D2H、canonical/flat双import与多层stamp/receipt echo先等Pod
  matched-state profile，再逐项抽取并保持parity。CUDA Graph、solver iteration、物理步长、resume与actor
  Observation/history扩张也分别延后，不与当前P0捆绑。
- **Reject：**giant rewrite、新mega-state/C++先行、强行统一backend physics、live hot-patch/resume/namespace
  复用、same-writer receipt/hash/echo自证、无人消费的Gate，以及用task成功、R07 ready、stdout或旧
  microbenchmark冒充安全/速度证据。长期只增量收敛到
  `PlantFacts → ActionBallState transition → StepTelemetry`，每次抽取都做fixed-tape parity。

### HISTORICAL — V5执行清单（现役局部步骤只认§0.5；不改变`origin/main:docs/NOW.md`）

- [x] 代码合同已采用：三段reference、post-transition due结算、launch/outcome/recovery相位关系、Isaac 36项
  ActionEpoch fault、Mu四项fact-integrity、pure timeout/table reason与schema `6/5/5`。stdout在live 39f不是
  authority；下一source `a3c528f1…`另关闭了post-durable裸print的假失败路径。
- [x] 结构/性能边界已采用：single runtime stack、run-owned augmented `runtime.mjb`、contact census减法、Epoch
  窄投影、selected-rubber row-neutral fault、Motion exact-resume与legacy Reward lazy import；不新增业务Gate。
- [x] 最终代码逐fresh process完成受影响host矩阵、`py_compile`与`git diff --check`；重叠suite按各自矩阵
  报告而不相加，skip只标明确CUDA/Isaac/superseded边界，不能算green。
- [x] live训练使用已push clean commit `39f9481950a660e198dedac7fd402806d648906b`；post-launch只新增
  stdout nonauthority修复`a3c528f1…`，不把它伪装成当前live source或据此迁移checkpoint。
- [x] exact Pod使用独立`--no-hardlinks` clean checkout恢复固定SHA资产。Pod root overlay/`/tmp`容量有限，所有
  HOME/TMP/XDG/CUDA/Warp/pycache/pytest basetemp显式落到`/workspace`；逐文件fresh-process回归后再跑双launcher
  dry-run、Mu真实GPU direct/RSL H48一轮、Isaac projection/selected-rubber、fixed-tape和1/5 ACK consumer；
  broad与clean-runtime/GPU矩阵如上；此前任何旧commit结果都不代签最终源码。跨后端formal fixed-tape/physics
  parity仍是下项，不由这些单后端测试代签。
- [x] successor exact ready且live queue/lock重新对账后，V4 MuJoCo按精确进程身份收口；fresh Isaac/MuJoCo
  均已连续ACK `0..4`且finite/fault0，双后端切换完成。
- [ ] 继续按balance→mimic→hit→landing报告学习：上一阶段基本成形时下一阶段必须已有非零自然分母；zero
  denominator写`未测`。Isaac已从显著回退恢复并创新高，累计出现2次playback；Mu已自然出现6次launch与
  1次R03-valid。两者的balance仍未基本成功，继续到预注册窗口而不因早期非单调性改配方。下一fresh source
  只补playback/open-cohort与optimizer/KL等纯telemetry，不新增安全Gate；Mu contact当前为diagnostic `0/6`，
  Isaac contact与两端landing为`未测`；独立playback成功、recovery与12500 completion仍未闭合。
- [ ] 用next-source测试替换Isaac的fixture/self-echo证据：真实执行wrapped `alg.update()`并核
  fault→optimizer→WAL/ACK顺序；同batch覆盖pure-timeout/timeout+plant/plant-only；用两个同时eligible row核
  坏row隔离与健康peer；逐owner做foreign-bit拒绝矩阵。退役无production callpoint的direct-Reward与永久skip
  测试不得再计现役green。这些是审计债，不是当前live的新增安全Gate。
- [ ] 在不打断两条fresh长跑的前提下，后续以独立rate diagnostic或自然matched strata核正式50-update wall、
  p50/p90、throughput和H24-equivalent；再按profile决定下一项算法/数据流减法。单点低于6秒不算达到方向目标。

## 1. 历史运行事实（current只认§0.5）

### HISTORICAL / SUPERSEDED — 2026-08-22 `aa42418b`两条fresh lineage

下列namespace都是[仅全新训练、禁止续跑](../DEFINITIONS.md#fresh-only-no-resume)的一次性运行身份；名称中的
backend/commit用于回答“哪套源码在哪个后端训练”，不是算法版本或可复用checkpoint标签。

- Isaac：namespace `fullmdp-a-h48-v2-isaac-idle-zero-aa42418b-20260822`，物理GPU2，launcher/child=
  `2423802/2423818`。ACK176只读刷新中recent10 collection中位`8.238 s/H48`，Reward finite且fault/CENSOR/
  nonfinite/conservation均0；episode均长first10→recent10约`87.73→148.93 tick`。reset-ready imitation和
  survival均有上升，但balance/action mimic尚未成功。
- portable MuJoCo：namespace `fullmdp-a-h48-v2-mujoco-idle-zero-aa42418b-r3-20260822`，物理GPU1，
  launcher/child=`2426696/2426711`。update118只读刷新中recent10 collection中位`9.585 s/H48`，storage/
  Reward finite且全部lifecycle/conservation fault为0；episode均长从first10 `67.32`升到recent10
  `151.17 tick`。早期qdes forbidden从近全episode降为recent10仅1次；fall/table比例仍波动，balance未成功。
- 两端都严格是fresh、`diagnostic_unauthorized=true`；219 width只保留ABI，不给旧idle-foot-bit snapshot
  exact-resume语义。性能数字不授权physics promotion/export/deploy。
- 最新阶段穿越：Isaac累计`due/selected/construction/key=2/2/2/2`且CENSOR/fault=0；MuJoCo累计
  `due/reveal=15/15`、deferred=0，R03 present/physically-valid=`174/174`。说明上一阶段一有row活过tick295，
  task/action-mimic入口立即开放。Isaac仍未playback；两端launch=0，故contact/landing为`未测`，击球与上台
  继续`未测`。

### HISTORICAL / SUPERSEDED — 2026-08-22课程解阻后的上一批successor

- Isaac现役：commit `333f9490…`，namespace
  `fullmdp-a-h48-v2-isaac-unblock-333f9490-20260822`，GPU1，launcher/child=`2213515/2213532`；
  22:55 UTC durable ACK已到至少69且每轮Reward `196,608/196,608` finite。profiler-off最近20轮中位约
  `15.775 s/H48`（H24-equivalent `7.89 s`），仍未达目标；前5轮profile把主要墙定位到
  `post_physics_publish`。
- portable MuJoCo现役：commit `23c0f6c…`，namespace
  `fullmdp-a-h48-v2-mujoco-unblock-23c0f6c8-20260822`，GPU0，launcher/child=`2219700/2219718`；
  22:55 UTC durable ACK已到至少47且Reward/storage全finite。最近20轮wall中位约`9.634 s/H48`
  （H24-equivalent `4.82 s`），child cwd与全部运行输出均钉到run root，source clean。
- Isaac仍未有due；MuJoCo update45已首次出现`due=1/reveal=1/deferred=0`，随后27个phase-2 task row中
  R03 physically-valid为26且Reward0--9非零，证明mimic入口与balance自然重叠。contact/landing仍为0，
  击球/上台继续写`未测`。不能用早期`ACCEPT=0`停车；也不能在真实due出现后仍无reveal时继续盲等。
- 被替换的Isaac/MuJoCo分别精确停止于ACK631/3023，旧root、snapshot和admin-stop pre/post证据保留，
  completion均缺席且未伪造。下述`96f0ca69…`与`99405266…`段落是本次修复前的历史快照，已被以上两条
  fresh successor取代。

三条旧长跑均已停止，不可继续推进：

- Isaac `e8eef4fb…`止于 durable ACK（optimizer成功后已持久化的更新签收）`4603`，累计
  `452,591,616` transitions；
- Isaac `ddb1e7c4…`止于 durable ACK `3467`，累计 `340,918,272` transitions；
- portable MuJoCo r3 止于 durable ACK `10249`，终段 last-100 wall mean/median=
  `4.890/4.886 s/update`。

两条 Isaac 的退出因果仍是`未知`：现有日志没有足够证据把它归因于 OOM、外部停止或代码异常，
也不能因 `final_rc=0`把不完整 run 写成完成。MuJoCo r3 由真实 `EPA_HORIZON`（扩展多面体碰撞算法的
迭代深度上限）overflow fail-stop；
它同时消费了会在 IDLE 时随全局 step 漂移的旧 229-D observation，所以不允许 resume。

fresh successor已经从clean detached `96f0ca69887aba44c71983529d05e759e1a4cd2f`在Pod1真实发射：

- namespace=`fullmdp-a-h48-v2-96f0ca69-20260821`，run root=
  `/workspace/franco/runs/fullmdp-a-h48-v2-96f0ca69-20260821`；
- GPU2 UUID=`GPU-473a79f3-8736-6c7f-c3db-290c6be385b8`；发射前empty-app与nonblocking lock门通过，
  launcher PID=`2030437`持lock等待唯一child PID=`2030453`自然退出；
- exact argv为Full-A `4096×48×12500/save500`，fresh runtime site，无resume/retry/signal/`ACCEPT`门；
- 首个durable ACK为update `0`、`196,608` transitions，collection/learning/pre-ACK=
  `9.354775/0.284285/9.639704 s`；Reward20/storage finite，conservation/nonfinite fault均为0；
  `model_0.pt` SHA-256=`50ebc7c9…7b26`；
- 最近一次只读检查已见update `0..4`共5个连续durable ACK，child仍为`R`。这是运行态快照，不预测最终
  completion；source、namespace与PID只作为本次run身份，不为后续run复用。

Isaac successor也已从clean detached `9940526684a4ea068b08bf7a2627a6e07c1452f1`在Pod1真实发射：

- namespace=`fullmdp-a-h48-v2-isaac-99405266-20260822`，GPU0 UUID=
  `GPU-889b1712-8d89-0536-5c9e-e79aae30523d`，PID=PGID=`2095711`、Kit child=`2095727`；
- exact argv同为Full-A `4096×48×12500/save500`，fd16 runtime receipt精确为
  `trainer_runtime_attested_v2`，fd18 sealed RSL archive与GPU0 lock由live child继承；
- durable ACK已连续到`0..10`；已完整打印的10次wall范围=`16.02--22.52 s`、median=`18.445 s`，
  H24-equivalent median=`9.2225 s`。每个ACK均有`196,608/196,608` actual Reward finite，nonfinite与conservation
  fault均为0；早期`ACCEPT=0`不作停车门。

第一次Isaac real曾因launcher只复制单个USD、缺同目录sealed source bundle而在PPO前自然RC1；该root封存、
GPU/lock自然释放，未resume或复用。successor改为拒绝non-regular entry并复制完整61 MiB asset package，
训练侧enclosed-source reconstruction不放宽。

旧 run 的 `ACCEPT=0`只是课程 telemetry：当时 opportunity 仍停在 balance/mimic/readiness 阶段。
balance→mimic→entry→strike→landing 本来就可能需要很多 step；`ACCEPT>0`不是启动门、安全门或早期
学习成败门。业务 eligible denominator 为零时也不调整 Reward0--13。

## HISTORICAL / FROZEN — 2. V5已采用的 successor 合同（V6只认§0.5）

### 2.1 HISTORICAL — PPO（近端策略优化）V3（current见§0.5的PPO V5/V6 candidate）

FullMDP A/C 与 portable MuJoCo 统一使用 code-owned typed recipe：

- rollout horizon（每次策略更新、每个环境收集的control step数）`H=48`；
- `max_iterations=12500`、`save_interval=500`；
- `num_learning_epochs=5`、`num_mini_batches=8`；
- `gamma=.99`、GAE（广义优势估计）`lambda=.98`。
- `entropy_coef=0`，保留learned `log_std`、fresh init sigma `.02`与adaptive KL；旧V2永久
  `entropy=.01`只作std无界上升的历史反例，不得resume。

它保持旧 `H24/U25000/save1000/E5/MB4`的大致总 transition、minibatch size、optimizer step 和按
environment step 计的保存节奏，但改变 policy refresh、GAE 和每次 update 的优化分组，因此只允许
fresh launch。旧 H24或PPO V2 snapshot不能resume到V3。

H48 是学习算法取舍，不是性能修复或速度豁免。旧“约 6 秒”是 H24 尺度下要求大砍迭代时间的量级
信号；线性换算到 H48 约为 `12 s/update`，不是硬 Gate。性能必须同时报告：

- 原始 `wall_s/update`；
- `transitions/s`；
- `H24-equivalent = wall_s × 24 / H`。

旧 Isaac H24 稳态约 `22 s/update`；若吞吐不变，H48约为 `44 s/update`。这意味着仍需数量级明显的
算法/数据流升级，不能靠把 batch 翻倍掩盖吞吐不变。

### 2.2 semantic Observation V2

family A 冻结为 actor `203-D`、critic `219-D`。这是删冗余后的语义替换，不是把旧 229/399机械
扩成237/407：

- actor common `183-D`加入 table-relative root XYZ、continuous heading XY 与 heading-frame COM
  velocity，并保留重力/角速度、proprioception、last action、teacher、anchor 与 Motion phase；
- actor task tail `20-D`只给可部署观测的 delayed racket position/velocity/raw A/+Y normal residual、
  base-goal residual、三个 per-env countdown、learning phase 与 Motion-visible task mask；
- critic只追加 `16-D`未来因果训练事实：episode剩余时间、live ball position/velocity/spin、selected
  contact/net latch、双脚 support 与 cadence dwell；
- 删除 raw task45、owner fact blob、fault/age、Reward due/paid 等控制账本。

此前尽调没有否决 root state：历史 fixed-194 和 A211 已包含 table-relative root pose/velocity；遗漏发生
在 direct-lean 229-D 迁移。当前代码反例可构造 root translation、yaw 和 COM velocity不同但旧 actor
observation 相同的状态；这些量不能由关节 q/dq、base goal 或 torso anchor唯一恢复。因此本次增加的是
非冗余且在目标传感链可观测的状态，而不是“多给 policy 一切真值”。HITTER、SMaSH、BeyondMimic只提供
相对 base/anchor、history/base velocity 的设计方向，不能代替本地 alias 反例。

actor不读取 contact/support/spin/fault/reward ledger。部署侧仍需真实接通 IMU、table/root定位、
marker→COM 的因果速度估计、encoder/FK和planner tuple；这条 producer 尚未完成，所以 V2 当前仍是
simulation diagnostic，不是部署就绪或“单帧 Markov”声明。family C 将使用独立 `202/218`合同，不补零
凑成 A 的宽度。

R06（落点结果发布者）observation projection也同步瘦身：旧 broad projection复制81个 tensor，并可能从 legacy key plane
读取到全零 current flight；successor复用现有R06 owner，在owner内部按 canonical shot key八字段加
publication ordinal唯一选择 live INBOUND/OPEN row，只导出 `flight_slot + contact/net-crossed/net-clear`
四个 tensor。无匹配、EMPTY或SETTLED_RETAINED均返回无 live slot；slot只用于读取Physical行，不成为
新的业务identity。没有新增owner、receipt或Gate。

R07的support/dwell复用唯一真实post-physics plant read，不在Observation再次扫描全机器人。cold genesis
明确为zero；selected reset只把generation精确`+1`的行在当帧归零，same-generation peer仍严格对齐tick，
下一真实post-physics恢复。Phase-C1又把Motion broad observation view从34 tensor收成两个真实consumer的
10-field并集；publication只复制真实并集，validator保留窄consumer隔离clone。该结构债已在host闭合，但
不把静态payload减法写成Pod wall收益。

### 2.3 真安全边界与结构减法

继续保留能跨独立事实源失败的边界：nonfinite/overflow、真实contact和joint/table limit、selected-reset
generation、shot key/publication join、source/asset provenance、optimizer成功后的durable ACK，以及失败后的
sticky poison/fail-stop。

继续删除的不是这些边界，而是同一writer的digest/receipt互证、zero-callpoint gate、无事件仍写的journal、
每substep重验construction已固定的class/bound method、无人消费的counter和用zero policy表现作启动许可。
Phase-B已经物理删除zero-caller formal owner及专属适配层；下一步不为死接口补compatibility adapter。

### 2.4 portable Full-A runtime package binding

V5 branch candidate用唯一path-free `runtime_stack` v1原子替换旧EPA-only
`mujoco_warp_runtime`，不保留双wire。它同时绑定EPA48 exact wheel/runtime、`rsl-rl-lib==3.1.2` wheel和
`mjlab==1.5.3`选定193文件/`1,399,177` bytes/tree digest。runner在Torch/MJLab import前cold verify，env
构造后核所有loaded `mjlab.*` module；consumer在读run-owned `runtime.mjb`和ACK前独立执行同一验证。
V5 evidence/completion/summary wire为exact `6/5/5`；上一个已提交版本`5/5/4`以及V4
`4/4/3`、更早`3/4/3`只作历史并被新consumer拒绝，本轮未发布中间版不再二次升号。
legacy WAIT不绑定，checkpoint/resume authority仍false。完整合同与未授权边界只在
[portable Full-A实验§0](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md#epa48-fresh-runtime-binding-20260821)
维护；恢复与调用见[`setup_local_sync`](setup_local_sync.md#bind-the-exact-epa48--rsl-rl-312-site-for-portable-full-a)。

## HISTORICAL / SUPERSEDED — 3. 旧依赖顺序（现役只认§0.5清单）

每项必须独立提交、可单独回退；前一项闭合后才进入后一项。

1. **Observation V2原子闭合**：Isaac/MuJoCo、training hard contract、snapshot receipt与文档同时切到
   A203/219；两后端用独立native state正负扰动验证frame、符号、scale、mask和R06 live selection。
   V1只为明确的历史 WAIT consumer保留，不允许Full-A静默fallback。
2. **MuJoCo EPA48**：build chain与fresh dual-wheel actual import已在branch candidate闭合；tracked replay
   也已在Pod同卡完成stock24 mask256/contact0与fork48 zero-overflow/contact1+finite raw contact各10次，
   branch只保留fixed fixture+replay-only工具。H48 fixed-tape已在exact Pod同卡生成两份record：离散/
   reason/events与初态exact，连续repeat envelope已保存；五个自然business strata因due全DEFER仍`未测`。
   instrumented/ASan独立oracle继续阻塞physics promotion/transfer claim，但不再作为
   `diagnostic_unauthorized`长跑前的表现式准入门：fork只改project-pinned容量常量，runtime overflow/nonfinite
   fail-stop仍在，且真实Full-A 61-update已自然rc0。stock CPU同样硬编码24，不能充当golden；r3仍不resume。
3. **SUPERSEDED：不再造单一mega-state。** 现役中心链是single
   `PlantFacts → ActionBallState transition → StepTelemetry`；backend-local adapter与existing WAL/ACK留在
   两端。backend physics、独立plant/measurement、finite、full-key/generation、optimizer与fsync边界保持
   分离；只删除zero-live事务、same-writer echo与重复派生。
   Phase-C0已先退役zero-business D05事务，Phase-C1已收窄Motion publication，Phase-C2已删除五个无人消费的
   Epoch整record返回clone，Phase-C2b把Observation cache移到整record clone之前，并把R07同一事务三份
   Epoch snapshot收成一份栈内快照；stale caller与首写前推进仍由零clone version/head检查拒绝。剩余dense
   Epoch/R06/Physical state与host sync仍须按真实profile继续收敛，不能把这些局部刀称为single-state完成。
4. **matched H48性能验收**：clean exact source、同卡、profiler-off，按zero/mixed/active strata对比；
   同时验证fixed tape RNG/highwater/reason/done/reset/Reward20/203/219 parity。报告原始wall、transitions/s和
   H24-equivalent；首墙转移后重新profile，不继续堆零碎clone patch。MuJoCo的真实生产热路rate入口已固定
   61 update；Isaac只增加同样`10+50+1`的code-owned diagnostic budget，不开放任意短跑覆盖12500配方。
   MuJoCo actual p50/p90=`9.448/9.661 s/update`、throughput=`20,779.64/s`、H24-equivalent p50=`4.724 s`，
   因而本后端不继续堆微优化；Isaac同卡pre/post Phase-C wall仍`未测`。
5. **fresh namespace启动与训练**：clock-fixed、EPA-fixed的portable MuJoCo V2与Isaac V2均使用fresh
   namespace；matched性能是并行诊断，不是Isaac启动前置。两条都不复用旧runtime site或snapshot，不插
   `ACCEPT>0`门。MuJoCo caller必须传
   [`--mujoco-warp-runtime-site`](../DEFINITIONS.md#mujoco-fullmdp-longrun-flags)指向一个尚不存在的绝对路径；
   future launcher先从clean Git truth取得并传`source_commit`，binder不得自报。短验只回答构造、有限性和
   真实调用点，随后同一进程继续训练并按balance→mimic→entry→strike→landing分阶段报告分母。
   branch候选one-shot launcher只负责clean Git、fresh root、GPU UUID/空卡/lock与固定H48 argv，并等待child
   自然退出；它不监控、不重试、不发signal，也不以`ACCEPT>0`作为启动或停止条件。Pod1 clean detached
   `2e4279ba` dry-run已PASS且未建root、未查GPU、未改lock；随后clean detached `96f0ca69`已按上述fresh
   identity真实发射并取得update `0..4`连续durable ACK。61-update实际rate与fixed-tape关闭发射前的有限
   构造/吞吐证据；运行现在只允许自然推进，`ACCEPT=0`及五strata未出现不作为表现门。Isaac V2的同卡
   pre/post Phase-C测量仍在后续，不能由MuJoCo代签。Isaac one-shot候选已复用现有Kit boot owner并通过
   host双launcher回归`19 passed`；exact Pod dry-run及GPU0 fresh real均已闭合，source=`99405266…`、
   PID=PGID=`2095711`，durable ACK已有`0..10`连续且Reward finite/fault0。当前Isaac前10个完整H48 wall
   median=`18.445 s`、H24-equivalent median=`9.2225 s`；这是自然长跑中的早期观测，不是matched稳态
   测量或6秒GO。

## 4. 当前完成条件

- Observation：已冻结V3 actor/critic=`215/231`；两个backend由独立producer生成同一有序字段与
  static scale，四组全phase teacher-achieved residual、heading退化、task mask、R06 live selection、cold genesis、
  row-wise selected reset和critic-only边界均有可区分反例；snapshot receipt绑定同一training contract SHA。
- Physics：EPA24-fail/48-finite deterministic fixture、GPU复测、独立oracle和overflow fail-stop全部闭合。
- Performance：exact Pod的matched H48数据证明transitions/s显著提升；host测试或删行数不能代签。
- Training：fresh run有连续durable ACK、finite Reward/observation和可解释的per-stage/per-action/per-side
  denominator；缺失格写`未测`，不以总均值覆盖零格。
- Authority：所有运行保持`diagnostic_unauthorized`；未完成restore/physics/deployment Gate前，不授权
  resume、promotion、export、部署或真机安全结论。

热路径证据和结构设计详见
[FullMDP hot-path实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md)；
MuJoCo fork状态见[G06](../gates/G06_isaac_to_mujoco.md)，runtime资产与调用见
[`setup_local_sync`](setup_local_sync.md#bind-the-exact-epa48--rsl-rl-312-site-for-portable-full-a)，
其余训练操作边界见[训练工序](run_training.md)。

## HISTORICAL / SUPERSEDED — 2026-08-22 dd82之后的执行TODO

本节保留`661ff84b`到首次细分profile的执行轨迹；未完成项已由本页§0.2的`aa42418b`裁决取代，不是第二份
active队列。

- [x] 保留`dd82bb7b` Isaac fresh只读运行，确认连续finite ACK、fault/CENSOR为0；早期episode尚未到tick295时，
  mimic/strike/landing按零分母记`未测`。
- [x] 封存MuJoCo首次fresh自然RC1 root；根因为launcher遗漏显式ready-pose输入，不复用namespace。
- [x] 把ready pose变成required CLI + fixed-SHA child binding；host launcher `15 passed`。
- [x] 实现同一D2H内transport/keyed双activity事实；空业务跳R03与R07 keyed写，但保留R07 readiness/Motion。
- [x] exact Pod分进程运行fast-path、retired反例、R07 no-key与launcher focused tests：候选
  `8aa3…`合计`546 passed,32 skipped`；随后修复active-path noncontiguous fixed flight-slot后，最终
  `661ff84b…`重跑直接覆盖的Physical/Epoch两文件=`92 passed,7 skipped`。另一个未改区域的旧N2 landing
  fixture因缺`scene.cfg.replicate_physics=true`有`2 failed`，不混入本候选通过数，也不伪写成已修。
- [x] `661ff84b…` fresh Isaac已取得连续ACK0--10、fault/CENSOR/nonfinite为0；idle Epoch commit约
  `816--824/update`，低于`dd82bb7b`约`1104--1152/update`。但profiler-off ACK5--10 wall约
  `14.94--15.91 s/H48`，仍高于约`12 s/H48`量级目标，故只完成发车/正确性，不完成性能项。
- [x] 同一commit的新MuJoCo用required ready-pose fresh启动并连续取得ACK0--4以上；ACK1--28
  collection大致`9.11--9.61 s/H48`、Reward/storage finite且conservation fault为0。两个backend现同时运行。
- [x] 在空闲GPU用只增加host-clock、5 update后自动卸载的嵌套profile拆开R07。修复后的
  `a7ae7c6f…`显示全批无shot-key时`r07_readiness_no_key=7.00--8.32 s/update`，几乎等于整个
  post-physics wall；首次`8cbccad8…`因替换frozen实例方法自然RC1的root保留，不冒充训练失败。
- [x] 证伪`3e53f991…/19877d8d…`的845行bootstrap readiness旁路：同H48五轮collection仍为
  `13.84--14.59 s/update`，没有可用提速。它只缩窄Epoch clone，却保留Motion frame0、完整plant读取、
  13项R07误差和Motion install，因此不继续维护第二套bootstrap ABI。
- [x] **SUPERSEDED：**没有采用same-tick双脚support，因为它仍会触发昂贵ContactSensor读取；`aa42418b`
  采用更窄且可证的N/A-zero no-key语义，keyed post-shot recovery保持现状。
- [x] exact Pod按新语义完成`304 passed, 6 skipped`，包括contact `.data`禁读sentinel、selected reset/stale
  chronology、keyed recovery和203/219 Observation。
- [x] 同H48/GPU先跑5-update bounded profile再跑profiler-off matched 5轮；旧support-read归零，后者中位
  `6.346 s/H48`，因此启动fresh Isaac successor。
- [x] successor连续ACK/fault0后精确退役旧Isaac；随后同source启动fresh MuJoCo并在连续ACK后退役旧MuJoCo。
