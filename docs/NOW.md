# NOW — Active Work Board

短线作战板:现在在干什么、谁在干、下一个检查点是什么。长线路线图在
[gates/G08_blind_spot_improvements.md](gates/G08_blind_spot_improvements.md);历史在
[PROGRESS.md](PROGRESS.md)(2026-07-06 大重排:过期章节全部原样搬去那里);
发射/判卷/运维的操作手册在 [runbook.md](runbook.md)。

Rules:

1. **Claim before you code.** Add/update your row here (owner + branch) BEFORE starting a work
   item, in the same push as your branch. This exists because we already built the same feature
   twice (no-teleport wrap: `3eba347` on main vs the `rsi-on-wrap-progress-fix` branch).
2. One row per active item; move finished rows to the Done section with a date, and put the
   substance into PROGRESS.md / the gate doc.
3. Priority ordering is maintained by claude (franco's agent) and discussed with franco; anyone can
   edit their own row.
4. **This file lives on `main` ONLY.** Never edit NOW.md on a feature branch (it would fork the
   board and merge back stale). Claim/update flow from any branch, without switching:
   ```bash
   git fetch origin && git show origin/main:docs/NOW.md   # read the live board
   # edit + push a docs-only commit straight to main:
   git stash -q; git switch main && git pull --ff-only && $EDITOR docs/NOW.md \
     && git commit -am "now: <one line>" && git push && git switch - && git stash pop -q
   ```
   Docs-only commits to main need no PR/review; everything else goes through branches.
5. **不用黑话**:每个 run/flag 第一次出现必须带人话;新术语先进下面的术语表再用。

## 当下状态与团队 focus（2026-07-13 01:30 CST）

本节只做 roadmap 的当前入口；下面的实验结果、奖励/训练台账、长期路线和历史判决继续保留，
不能用这份短报替代可复现实验记录。

- **现在解决什么**：把“智元 vendor MuJoCo runtime/plant + 我们自己的 planner + 我们自己的
  policy”接成一条不会假绿、几天内可录屏的最短 demo 链。vendor 环境是裁判；planner 与 policy
  都可升级，但必须成对绑定 exact SHA 并过同一题表。179-D 拍面/模型装载合同已收口，当前主阻塞
  转到 planner-policy 接缝、发球时序所有权、安全的 vendor backend first tick 和 D0 单拍闭环。
- **刚得到什么**：16:04–16:22 两 Pod 的低频分块只读全扫成功。训练/eval 仍 clean exact
  `6d93bcb` / `46a0ce2`；28 条接受臂现在是 **16 条 fresh 在训练、12 条 continuation 已有
  `20998` 终档**。进程快照仍有 17 个 trainer PID，是因为 Pod1 最后一条 guidance 已写完终档但
  仍在自然清理；这不是第 17 条未终档臂，也没有被 signal/restart。28/28 最新 checkpoint 均满足
  文件名迭代=内嵌迭代、1,762,715 个浮点全 finite、内嵌 contract SHA=相邻文件 SHA、fresh
  lineage=1/continuation=0；接受日志无 NaN/Inf/Traceback/OOM/Killed/新 malloc/early exit。十个登记
  worker 也按动态状态复核：4 个仍有 pending job 且 PID/PGID/command/manifest sidecar 精确存活，
  6 个已清空各自队列后自然退出；累计 `68/68` hardened state（48 global + 20 causal）rc0、checkpoint/
  job/job-contract/report/std-sidecar SHA、K20 单题 reset 与 exactness 标签全部正确，无 child judge 或
  judge Kit lock。GPU util Pod1=`92/95/95%`、Pod2=`90/83/96%`，RAM available 约 `914/922 GiB`、
  swap 0。旧口径“10 workers 全活”已正式弃用。
- **4k matched 判卷状态**：四条 fresh SZ 的 `model_4000.pt` 已跨过训练里程碑；exact activation
  consumer 已随 main `7bc6d1f` 合入并通过顶层 `468 passed, 9 skipped`。它没有 SSH/信号面，固定
  `auto_start=false`，必须先由 Pod1(seed1/3)、Pod2(seed2/4) 各生成 readiness audit，再合成 all-four
  activation；每 Pod 串行拿共享 Kit lock，seed1 也重跑同一 K100，no-clobber 保存 PID=PGID/state/log/
  result。现在两 Pod 已在 train/eval 外放好同一绝对路径的 exact source bundle 与旧 K100 bytes；
  Pod1 seed1/3、Pod2 seed2/4 audit 已分别通过（file `3fc325e1...247b8` / `4f25786b...565f7`）；
  exact union 已生成 activation（file `9dea76c2...ce704`、content `eaa92ca2...aa4fb`），两 Pod 同路径
  同步并各自 `contract-check` PASS。三份 activation evidence 已入库。两 Pod 随后完成 no-clobber
  `prepare`：runtime file/content SHA 分别为 Pod1 `2b76a5a...8201e`/`36e878f0...5ba73`、Pod2
  `dbecc102...d1c9b`/`91a0070a...30794`；原始 JSON bytes 已入库并与 Pod 逐字节复核。两份合同仍是
  `prepared_not_started/jobs_started=0/auto_start=false`，重查 checkpoint/lineage/checkout/Kit 冲突均绿，
  **仍未 run/judge**，所以没有新分数。当前只差 reviewed persistent parent supervisor，防止 SSH 断开时
  丢掉两 seed 串行控制与最终 result；闭合后才启动并 aggregate。只判 seed4 是否晚熟，已知 seed1
  4k=`50/100` 使整族稳定门数学上不可能通过。
- **训练后端路线决策（团队 2026-07-12 21:30）**：纯 Isaac/解析回球指标不再作为继续扩展
  训练配方的依据；**原生 MuJoCo 训练/微调后端升为 P0**。第一步不是把单世界实时 AimRT vendor
  runtime 当采样器，而是独立实现 native CPU `rsl_rl VecEnv`，不能和判卷器共用 observation/action/
  reward 代码制造共因假绿。只读 preflight 已确认当前 main 的 vendor MJCF 没有球/台/网，且 BankExam
  会改 dt/摩擦/solver/clamp；因此先分 `isaac_bank_parity_v1` 与 1 ms explicit-PD `vendor_gate3_v1`
  两套 effective plant。D0 v0 只承诺一次挥拍的平衡/击球状态微调，不冒充物理回球。reset/固定 action
  tape 对拍冻结 Python evaluator；现 evaluator 没有训练 reward，逐项 reward 必须另写独立 replay oracle。
  warm start 只载 actor/distribution/actor-normalizer，critic/optimizer 必须 fresh；完整 engine/version/
  MJCF+mesh/effective-plant/runtime flags/obs/action/reward/reset/source checkpoint SHA 入新合同。
  **MJX/MJWarp 只在原生 A/B 同卷有效后扩吞吐**；其结果不能冒充 vendor exact plant，最终票仍是
  不参与训练的 vendor Gate3/Gate3B。现有 Isaac 臂按已冻结 milestone 留证，但不再新增纯
  Isaac-only reward/teacher 扫描，也不默认把全部诊断臂烧到终档。
- **现在谁在 focus 什么**：
  - franco/Codex：满池 checkpoint 早判、planner-policy 成对 Gate3、vendor first tick/D0、
    **MuJoCo 原生训练/微调 P0** + Isaac↔MuJoCo 分层归因；同时守住动作/TOPP/连续恢复队列，
    不让长期轴阻塞最短 demo。
  - jiayi/dongc1：`origin/hitter` 上积累 HitterPure/V3 policy、planner 与 vendor rally 编排；
    最新值得移植的是发球等待 MOTION 同步、marker→base 旋转、solve cadence 和预测时域修复，
    但该提交混有旧配方/资产，当前只做小块移植，不整支合入。
  - yikang：reference oracle、stand viewer 与 default-off 的 shadow solver 接线壳均已选择性合 main；
    head reward 没有硬塞进现役配方。当前继续做 vendor Gate3 谱系筛卷与课程侧诊断；v2 obs 顺序、
    24100 交接模型和 v7 击球平面仍需与 jiayi 对齐后才可判分。
  - 动作库轨：Franco/v6/v7 十段已过内容寻址、GMR/grounding/自碰与身体余隙；当前由
    spatial-retarget 击球点适配推进，TOPP、2-vs-4 和 RL 等离线安全/可行性门后再占 GPU。
- **最近值得团队同步的结果**：两轮红队先后抓出“反手 raw-A 符号被误判”和“no-publish 顺便
  免模型合同”两个会制造假绿的问题，现都已闭环；恢复/等待不把三种 reward 单独相加，先做 A/B/C
  结构轴，结构失败后才做 `2^3` 交互和固定总预算 mixture；fresh SZ 四 seed 的 2k 正式卷虽中位高，
  worst seed 仅 `20/100`，已正式判 seed-stability 失败。新 reference oracle 与 Torch 物理全表对拍
  通过，最大误差 `4.63e-9`；stand viewer 已纠正为生产 29-DOF PD、neck passive，不再伪称额外 head
  `40/2` 是生产合同。planner 第三版 `6aae7ac` 的 host `198 pass + 2 skip` 也被 fresh-clone 红队否决，
  再次证明 source-string/纯 helper 全绿不能代替真实 tick 与跨 topic 因果审计。D0 只读审计又确认
  旧 `pp_gate3_rally.sh`/conductor/run_sim 链不能修补复用：它先发球后起 runner、broad kill、复用
  RouDi/共享 `/tmp` 与 `/dev/shm`，且人类日志没有可证明的 fresh READY；新 D0 必须是私有
  namespace/cgroup+pidfd、machine status、DISARMED ball 一次性 arm 和 no-replace ledger。
- **当前新 blocker**：formal tuple/sequence、latest-base 全 actor 门和 serve-sync v4 的 source P1 已在
  候选 `c0a8e46` 收口，并与 main 的 default-off shadow solver 合并；host planner/source=
  `180 pass + 2 skip`、顶层=`517 pass + 9 skip`，serve design rc0、launch rc1 精确 `49 MISSING`。
  该候选尚未合 main，因为最终集成 C++ 字节仍需一次隔离 Linux/ROS/Jazzy Release compile+full native；
  Pod2 exact clone 已 clean，但 ROS/Jazzy 路径不存在，portable 预检又缺 ORT 1.19.2 头文件，均在 CMake
  前 fail-fast；旧 component `235/5` 不能冒充组合证据。即使源码合 main，49 个 runtime binding、vendor
  first tick 和 Gate3 行为仍是 OPEN。q50 readiness/prepare 已闭合，当前只差 reviewed persistent
  supervisor 后开卷。
- **正手拍面短环（Human owner: franco；Executor: Codex；branch:
  `Franco_codex/face-sign-forensic`）**：旧表述“所有 seed 正手都约 170°”不精确；model-2000
  seed1/2/3 的 raw-A 正手误差是 `171.10/172.94/173.39°`，seed4 没有正手 exact strike，不能补成第四个
  170°样本。现有同一击球态在 Isaac/MuJoCo 的有符号误差约 `170.72/171.09°`，而解析回球器会先
  `orient_normal` 抹掉正负号；当前任务是把“策略实际 raw-A 反面”与“回球尺对符号失明”分开落证，
  并给旧 q50 加不可忽略的 signed-face 诚实门。期间不改训练合同、不因高回球率晋级。
- **代码交流状态**：recovery A/B/C、plan-only Gate3 源码门和严格 face179 模型合同已分别随
  `e10922a`、`b2067ba`、`8975043` 进入 main；face179 的 vendor 行为证据仍明确为 `Partial`。
  yikang 的 oracle/stand 诊断随 `3df6ff5` 合入，shadow solver 接线壳随 `7e31819` 合入，head reward
  明确留在未验证队列。三版旧 planner 候选 `69418a9/71b0b23/6aae7ac` 均未合入；当前整合候选
  `c0a8e46` 已过 host 红队，等 exact Release 后才进 main。first-tick joined-source 诊断已随本轮合入 main，38/38
  host source 测试通过；它把 outer/payload 及 planner/native/source/runtime 五类 exactness 永久写成
  false，并由 formal-style consumer 拒绝。vendor pose/twist/racket 没有共同 MuJoCo sample sequence，
  因此这次合并只增加可观测性与防伪护栏，不是 vendor first tick 或 Gate3 行为通过。
  证据卫生修复随 `a08eb2e` 合入，旧 GMR ledger 不追写新 auditor SHA，plant v1 当前源码漂移按预期
  fail closed；model-4000 activation consumer 随 `7bc6d1f` 合入，all-four activation 证据随
  `06a65b4` 入库，但 judge 尚未启动。MuJoCo training-v0 只读 preflight 随 `68c0c2a` 合入，明确
  当前 v0 不是物理回球训练。
- **几天内 demo 的最短闭环**：D0 先做 vendor MuJoCo 中可录屏的固定同卷 demo——真实
  planner 发题、production C++ runner、fresh SZ 的最佳 finite checkpoint，正/反手各一组
  physical returns；每个模型/planner/runtime/MJCF/题表都带 SHA。D0 不冒充连续实战。
  D1 再要求同一进程内 3–5 球 no-reset；恢复/reward mixture、四动作、TOPP 与标定 plant
  不阻塞 D0，但继续并行排队。任何真机 demo 仍受 G07 独立安全门约束，本阶段不发真机命令。
- **未来 24 小时决策**：①给 Pod2 隔离验证树补 exact ORT 1.19.2 build dependency（或在已装 ROS 的
  Pod1 复用内容寻址缓存），对 `c0a8e46` 跑 exact Release/full-native，绿后立即合 main并刷新 source SHA；
  同时按已绿 activation 执行两 Pod prepare→重查锁→串行 matched K100→aggregate；
  ②在可用 MuJoCo 环境补 stand 10 秒数值诊断，但它不阻塞 D0；③按 serve v4 的 49 项缺口实现
  machine ACK/唯一 publisher/first-publish ledger，再只用新的精确进程所有权方案准备 vendor first tick；
  同步冻结 MuJoCo trainer v0 合同与
  单环境 parity canary（先验收，不启动长跑），明确 actor warm-start、critic/optimizer fresh 和
  独立 K100/Gate3 判卷边界；自然释放的 GPU
  槽只接受已过 schema2/L0/整轨安全与动力学门的动作/TOPP 任务，未过门就保持空闲而不制造无效训练；④能 first tick
  就立即跑 D0 小卷并录完整 ledger，不能则把失败精确归到 planner、policy、plant 或 runtime 一层。
  定期任务只做巡检；阶段结论统一更新本节，稳定时不刷聊天长报。

## 详细现场证据（04:15 CST 基线；后续以短报为入口）

- **Phase-1 仍是真满池**:四条原始续训自然终档后，四条独立预注册的因果缺边补进空槽；
  现在两 Pod 各 12 条 live trainer、每 GPU 恰好 4 条，加上 4 条 clean terminal，共 28 条
  接受臂。GPU 约 `23/32.6 GiB`、util `77–97%`，两机 host available RAM 约 `963 GiB`、无 swap。
  LZ-seed3 首次 `malloc` 失败已保留，同配方 retry PGID `1354525` 是唯一接受臂。
- 两 Pod 训练/eval checkout 均 clean 固定在 `6d93bcb` / `46a0ce2`；有训练存活就不对训练树
  pull/切换/修改。六个 global 里程碑 worker 已换成 hardened 版：Pod1
  `1432280/1432292/1432304`，Pod2 `200706/200718/200730`；四个因果补边 worker 为
  Pod1 `1416771/1416784`、Pod2 `198759/198771`。旧结果不洗账，五份已到存档重判
  rc0 并绑 manifest/job/job-contract SHA。禁止 broad kill，只管理记录的本臂/本 worker PGID。
- 01:44–01:52 CST 再逐一读取 28 条接受臂的最新 checkpoint：每个 1,762,715 个浮点元素全
  finite，文件名/内嵌 iteration、相邻 hard-contract SHA、fresh/causal lineage 全匹配；accepted
  日志 NaN/Inf/Traceback/OOM/malloc/Killed 为0。每 Pod 5个 hardened worker 全活、无 child judge、
  Kit lock 空闲；Pod1/2 当前 hardened 结果分别 16/16、15/15 保持正确 exactness 标签。
- 03:05–03:13 CST 又做了一轮独立全量复核：28/28 最新 checkpoint 仍各有 1,762,715 个浮点、
  non-finite=0、filename iter=embedded iter、checkpoint 内嵌 contract SHA=相邻 contract 文件 SHA、fresh
  lineage=1/causal=0；接受日志的 word-boundary NaN/Inf/Traceback/OOM/malloc/Killed 仍为0。
  两 Pod 各12 live+2 terminal、每卡4条，十个登记 worker 均按精确 PGID 存活、无 child judge/Kit；
  GPU util `80–97%`、显存约 `22.9–23.2/32.6 GiB`，available RAM `958/964 GB`、swap 0。
- 04:15 CST 再查仍为两 Pod 各 12 live+2 terminal、每卡 4 条；train/eval checkout 继续 clean
  `6d93bcb`/`46a0ce2`，十个 worker 全活、无 child judge/Kit。逐臂最新 checkpoint 的文件名/
  内嵌 iteration、finite、相邻 contract SHA 和 fresh/causal lineage 仍全匹配，最近 3 MB/臂异常词
  为 0；GPU util `74–94%`、显存约 `22.9–23.2 GiB`，available RAM `905/919 GB`。未发信号。
- Pod1 的 SSH 映射约一小时 connect timeout；RunPod 控制台也未登录/无法只读确认，因此没有
  重启或重复发任务。23:45 CST SSH 恢复后的全审计显示训练未中断：仍为 12 live+2 terminal、
  4/4/4，所有 iteration 连续前进，错误 0，训练树 clean，五个 worker 全活。
- **checkpoint 早判已改变选择**:fresh SZ seed1 q10 从 2k `0.90` 回落到 4k `0.50`，触发
  同一 K100 正式卷。2k 正/反/总=`33/50,50/50,83/100`，4k=`0/50,50/50,50/100`，因此
  保留 2k 峰值，但整臂继续原配方训练。两者都 fresh/exact、zero physical fall，但每题都由
  非物理 post-strike tracking guard 收尾，所以只证单拍 checkpoint 选择，不证连续恢复/
  部署稳定。同题 Isaac 对 2k/4k 都给 `99/100`，完全平局；它没复现 MuJoCo 排名，
  所以跨引擎 checkpoint 门仍开。
- M3 terminal 因果对在 MuJoCo K100 上 old/S1 总回球为 `.42/1.00`，但同题 Isaac 两者均
  `.99`。逐题 forensic 发现“同题”还不是“同判分仪器”：fresh 4k 前手拍心误差在 MuJoCo/
  Isaac 为 `13.15/2.48 cm`；M3-old 反手 `168.15°` signed-face 错误被 Isaac 的
  `orient_normal` 擦掉。旧 `9 physical falls` 也已纠正为 `1 physical fall+8 guard reset`。
  新 2 engine×physical/analytic 四格缺一即拒绝，阈值不改；当前等 Isaac post-contact truth。
- checkpoint 曲线继续不等 terminal：M3 seed2 old/S1 的 18k→19k 为 `.65/1.00→.85/.95`；
  M2 seed2 为 `.40/.60→.50/.40`，小卷甚至反转排序。两组都只保留曲线、继续训练，不停/
  晋级/q50。scale-out fresh 2k worker 已开始出 SP/LZ 等诊断格，SZ 正式格仍按同卷队列等待。
- plant 矛盾已正面记账:SZ 只是“零摩擦执行协议可精确重放”，不是部署 plant；SP/LP
  已定位为把 vendor MJCF 的 constant-Nm `frictionloss` 数字原样塞进 PhysX 无量纲/
  load-dependent friction 的历史 proxy，不能当标定摩擦对照。新 v1 合同拒绝任何非零跨单位
  数值转换，并强制最终 MuJoCo 腿绑定 vendor Gate3/Gate3B MJCF/runtime/31关节实例化报告；
  新 SC 仍严格卡在真实 calibration evidence，尚无训练臂。
- Franco/v6/v7 十段空挥已完成 intake、GVHMR 10/10、per-video-beta GMR 10/10+落地、
  “十段共用同一个中位体型参数”的 GMR 10/10。真实 loader 只取前 10 维，旧的“补六个零”
  猜测已撤销。共用体型版本的 CPU-only grounding 也已 10/10：只把固定 root-z 上移
  `6.73–9.74 cm`，其余 payload 逐 pickle bit-exact，最低点约留 `10 µm`。后续 654 源帧/
  5,162 个 240Hz 样本的地面危险、自碰、拍/柄距身体 `<5 mm` 危险和 `<20 mm` warning 都为0，
  最薄余隙 `40.25 mm`。现在 canonical HOPE +X counterfactual frame/mirror 合同已冻结并真消费
  原64题；所有 motion/library exact 都是 `0/64`、common support=0，所以 2-vs-4 仍无结论，
  也不等于动作无效。反手拉 B/C intrinsic 为 `32/32@.5444`、`27/32@.5155`，保留给 spatial
  retarget；当前瓶颈是击球点空间适配，不是 clip 长度，TOPP 暂停到 schema2/L0/L1/桌网/动力学门。
- **空间适配现在只开放安全的整轨 SE(2) proposal**：全十动作×同侧 64 题都在卷内；R0 只平移，
  R1 只允许 `[-10,-5,0,5,10]°` 小角度旋转再平移，严禁改 z/尺度/镜像/关节或逐帧扭曲。
  prereg/tool/scorer/source SHA 已自绑定，专项 7 tests 与旧 v5 合并回归通过；但本机没有 full v5
  大结果，certificate 也故意未预注册，所以尚无真实 proposal/accepted motion。任何候选必须再过
  exact schema2、L0、vendor L1、整轨桌网 `>=5mm` 和动力学，才可进入 TOPP/RL/Gate3。
- 连续时序缺口仍在：完整 clip-wrap+hold 同侧理论中位约 `3.75 s`；场馆 `1.903 s` 来自
  重叠 n=21、16/21 高球、2.5s 截尾，只证伪现役节奏，不作目标。T1 训练端核心已在
  `be5d7cf` 实现：仅 exact strike 后原子揭题、固定 deadline、miss 也消耗、全 carry-state。
  仍缺 materialized schedule、连续双引擎卷、自击门、fresh baseline 和标定 plant，未点火。
- 已有第一个 isolated exact MuJoCo checkpoint baseline(2k=`.83`)，但还没有跨引擎+连续球+
  标定 plant 都通过的 accepted quality/deployment baseline。只对能改决策的同 family/seed/milestone
  做 q10 方向卷和 q50 决策卷。
- **fresh 正式格四 seed 的 2k 小卷已齐，但只作方向信号**：同一 clean-q10 题表总回球率为
  seed1/2/3/4=`.90/1.00/1.00/.25`。seed4 是必须解释的稳定性离群点，不能据此停臂或挑 seed；
  四个 2k checkpoint、相邻 schema-v3 hard-contract 与 finite tensor 已复核；同一 K100
  （每侧50题）正式卷已完成：seed1/2/3/4=`83/100,100/100,100/100,20/100`，全部完整分母、
  exact、0 physical fall。预注册 median `.915` 过，但 worst `.20`、spread `.80`、worst-side
  `0` 均失败，正式结论 `fail_seed_stability_checkpoint_evidence`；2k 不能作稳定 baseline，
  不得用前三个强 seed 隐去 seed4。supervisor/child 已全退，trainer/worker 全活且零信号。
- **4k 四 seed 只排了严格同卷队列，还没有启动判卷**：继续逐字节复用 2k 的 K100、四 seed、
  SZ family 和原门槛；四个 checkpoint 未全部 finite/iter=4000/同 hard-contract/fresh lineage 前
  无法生成 activation，且 activation 本身没有 judge 入口。已知 seed1 4k=`.50`，所以 family
  stability 按原 worst-seed `.65` 门槛必然不过；该卷只区分 seed4 是晚熟还是持续弱，不能洗成
  稳定 baseline。源码护栏 20 tests 通过；当前 seed1/2 已越过4k，但 seed3/4 最新仅约
  `2850/3060`，四臂 barrier 未齐，尚无 Pod readiness union/activation/runtime。
- **最终裁决分两层，Isaac 不投最终票**：Gate3 绑定同一 C++ runner/vendor MJCF/标定 plant/model，
  先过 first tick 与连续稳定硬前置；Gate3B 必须复用同 runtime，再用 immutable random-arrival q50
  主判 first-strike non-regression、回球率和质量。候选要在智元 MuJoCo 里保持平衡、完成挥拍/恢复且
  不靠人工 reset，不能用更高 Isaac 分数覆盖失败。Isaac/MuJoCo 同 policy 的巨大差异另作归因轴：
  现已完成 PhysicalBall Phase-B
  源码门（attempt token、真实 served/contact/landing、完整 provenance，focused `63 pass/1 artifact skip`），
  但 clean-detached 100 题 runtime 和高速拍面 substep 几何尚未验证，所以不能拿当前 Isaac `.99`
  给 MuJoCo 塌陷洗白。
- 179-D 到 Gate3 的 loader 源码链已进 main：schema2 原子携带拍位/拍速/正X unit normal/
  zero-rho，严格绑定 face/bank/SHA metadata，坏包/无解显式 `valid=0` revoke，正式 flat 先于可选
  mirror 发布。修复通用 ONNX TypeInfo 生命周期 UB 后，Pod1 隔离 ROS/AimRT-enabled Release 三目标
  已全链接，formal SZ 179 loader 1/1、native 205 pass/9 optional skip；安全 preflight rc0 精确输出
  obs179/双 SHA，日志确认零 backend init/start。它只做零观测 ONNX prewarm，不是 backend first tick；
  但真实 formal train bank 暴露出尚未进 main 的拍面语义缺口：外部正 X normal 是 physical
  striking-face B，179 actor 尾部却是 raw mount +Y/A；反手 raw bank 724/724 行 `x<0`。当前 main
  尚缺按 clip `[+1,-1]` 的 B→A 转换，故旧 loader pass 不能外推为反手 engage。修复分支正在用真实
  bank 重导、球冠+参考半球双门和 vendor full build 复核；canonical recovery、first tick 和行为卷仍 open。
- **连续等待按一个耦合 phase 设计，不把三项单独过关相加**：A=显式 interruptible bridge、
  B=actor-visible canonical ready tuple、C=完整上一题 tuple 的结构轴已进 main；当前混代 idle 和
  static handoff 清零 last-action 都不是正式臂。A 还冻结了 executed-action→last-action 投影、逐 tick
  ownership、loss mask、duration-correct option return 与三臂同 actor-sample/update 预算。只有结构
  失败才把平衡债/ready potential/随机来球可接战性先归一化，做完整 `2^3`，再做固定总预算 mixture；
  safety/self-hit 不可补偿。`50 passed`，但 24 项执行绑定仍 null，launch-check 必须失败。
- 已验证基础设施、plant/q50 证据、4k 同卷护栏、动作 frame/空间 proposal screen、179 Gate3
  source+formal loader gate 与 recovery 设计/校验已收口到 `main@e10922a`；动作 2-vs-4、TOPP、plant 标定、backend
  first tick 与连续 Gate3 rollout 仍按 open gate 记账。NOW 只保留 main 活板版本，feature 分支的
  旧 NOW 不回灌。

### 14:00 CST 早期快照(仅历史，不用于进程管理)

- **Phase-1 已恢复满池(07-11 14:00 CST)**:两台 Pod 共六张 5090，每卡四条 4096-env
  训练，`4/4/4 + 4/4/4 = 24` 条预注册实验均达首 PPO iteration。实测每卡约
  `22.9–23.2/32.6 GiB`、GPU 利用率约 `87–97%`，host RAM 仍有 `840/904 GiB`
  available。唯一一次 LZ-seed3 scene-start `malloc` 失败已保留，完全同配方的精确单臂
  retry 成功；不能把失败臂算成第 25 条。
- 两 Pod 训练 checkout 均冻结在 clean `6d93bcb`，有任一训练存活就禁止 pull/改工作树；
  detached evaluator 仍冻结在 clean `46a0ce2`。原始六臂的旧存档续训/从零训练、
  新增 18 臂扩池的旧存档续训/从零训练已拆成独立后台判卷进程(worker)：
  Pod1 PGID `1394150/1394810/1380340/1397266`；Pod2 的原始续训队已完成，其余
  `194276/192815/195085`。新增 18 臂共 142 个每侧 10 题(clean q10)里程碑任务，
  不等终档才判；只按这些精确 PGID 管理，禁止 broad kill。
- 从零训练的 SZ(共用拍面+零摩擦、当前唯一合同精确格)clean-q10 已首次跑通增长曲线：
  seed1 的 `0/1k/2k=0.00/0.50/0.90`，seed2
  `0.00/0.50/1.00`；这只证明方向，不能停训/晋级，决定仍需同卷 50/侧。旧存档续训(causal)20k
  M3 old/S1=`0.45/1.00`，M2=`0.50/0.50`，全部仍是 inexact diagnostic。
- 非零摩擦 SP 格已纠正为 inexact 诊断：PhysX 系数与 MuJoCo `frictionloss`
  物理语义不同，不能因为共用拍面约定就冒充 exact。只有 SZ 是当前
  跨引擎执行合同的 formal target；SZ q50 照常做，部署 plant/Gate3B 另等标定摩擦 SC。
- Franco/v6/v7 十段空挥已完成内容寻址 intake 和 Pod1 GVHMR 10/10 结构重建；
  帧数/shape/finite 全过且工具、权重、环境、输入输出 SHA 已入库。这不是 A3
  安全验收；下一步是 GMR 诊断重定向、canonical body 形状、自碰/桌网余隙和逐帧回球可行性。
- 连续时序有明确缺口：现役虽不传送并在同一 episode 连续换题，但完整 clip-wrap+hold 的同侧
  击球间隔理论中位约 `3.75 s`；场馆保守 A-B-A 样本中位 `1.903 s`，新题往往在旧 clip 播完前
  已出现。当前 24 臂不改合同，也不能写成“随时来下一球已通过”；后续另开 T0/T1 event-driven
  timing/carry-state 配对。
- 还没有 accepted quality baseline。不给 30+ 历史臂盲补全表:先用 M3f/M2/G1
  known-good/common/known-bad canary 验尺,再用单球 `ns=0` 筛候选,只给存活臂补连续球、噪声和第二 seed。
- 本地旧 Phase-1 顺序题原型与现役 schema-v3 BankExam 不兼容,已保存在
  `codex/integrate-local-ablation-20260711@30f4652`,不会机械并入主线。

## 北极星与尺子(franco 2026-07-06 定版;先读这节再看任何数字)

- **北极星 = 回球率**(真球考试:场馆分布来球,虚拟回球合法落到对面台)。它动了才算真进步。
- **训练时主看 = 训练内上台率**(`virtual_return_rate`:每次到点击球算一次机会,球被打到、
  过网、落进对面台内才计 1;按机会数算,打不到球就是 0,不会被"打得少但打到就上台"虚高)。
  正手/反手各有分身(`virtual_return_rate_forehand/backhand`——反手先行的判读要看单侧)。
- **辅助 = 训练内击球率**(`virtual_hit_rate`:到点的击球机会里,真的碰到球的比例)。
- **三合格率(composite)降级为诊断尺**:只用来定位"哪一项不达标",不再判臂生死
  (07-06 实例:拍面 25° 误差照样 79% 上台)。
- **正式入账**:MuJoCo schema-v3 BankExam 是现役转移裁决腿;Isaac 只有在
  evaluator-owned 适配器消费同一内容 SHA/schedule 后才能作同题伴随腿。训练内曲线仍只作 proxy,必须标注"训练内"。
- **报数格式(2026-07-11,24 个报告切面)**:任何臂的读数一律报——
  **正手/反手 × 物理不摔/击球率/上台率 × Isaac/MuJoCo × 单球固定点/连续击球**;
  这些是同一批 rollout 的派生切面,不是 24 个无关实验。
  单元格没有数字就写"未测/不适用",不许合并平均后只报一个数(反手瘫痪被 0.998 的全局
  击球率盖住,就是这个格式缺失的直接代价)。连续击球的 12 个切面尚无正式入账卷;Isaac 连续击球版待物理发球时间线,
  MuJoCo 连续击球版=部署仿真考试/Gate 3B。
- **选存档 = 训练内上台率的峰值附近**(旧规矩"选 composite 峰值"同步换尺),入账前 MuJoCo 复核。
- **⚠ 口径补条(franco 07-09,fixE 终审教训)**:训练内上台率**报人一律用 rally 全分母口径**
  (`virtual_return_rate_rally_*`:分母=挥拍起手数,摔倒计败);`virtual_return_rate` 是幸存者
  分母(只算活到击球帧的挥拍),两者可差近一倍(fixE:0.8415 vs 0.5255)——幸存者口径只作
  诊断辅尺。**历史数字重读规则(franco 07-09 追认)**:rally 口径 07-08 才实装——此前
  一切"训练内击球率/上台率"都是幸存者分母,**高摔臂上系统性虚高**,重读近似=原值×(1−该侧
  触球前摔率)(例:M1 现测 vrr 正手 0.790 vs rally 0.309;s1w3 主攻期中 0.78 折全分母
  ≈0.62-0.65;fixE 0.8415→0.5255)。**旧 MuJoCo BankExam 也有幸存者分母病**:触球前摔倒/
  guard reset 曾在出题后从分母消失;NOW 旧版写的“摔=零击计败、历史无需追改”是
  错的。修后判卷器以 immutable schedule 的 **all attempts** 为分母,触球前失败留败卷;
  所有关键历史入账需用修后尺重跑,旧数只能标“旧判分器”。`--qdes-clamp`、共同 `stand`
  ready state、题表/MJCF/execution SHA 均是正式卷必填合同。
- **⚠ rally 口径 07-09 记账根治(merge 53d440e;判读纪律随之更新)**:旧 rally 曲线分子分母
  两本账衰减节拍不同,4096 env 同刻 resume 排成同步大队列时比值能冲破 1(0.31→1.48 振荡,
  不是真进步)。新记账=每拍打完起拍数+是否回球同刻入同一本账,**恒 ≤1 且=真实回球率**;
  旧算法曲线以 `_legacy` 后缀并行发一个过渡期做对照。**判读纪律:没开防同步旗标的存量/同步臂,
  一切 EMA 指标(摔率/完成率/上台率)取 ≥21 个迭代周期的均值再判,单点读数不作数;
  R4b/R5b/R6b 的摔率读数按此重审**。详见统一队列台账行(07-09 指标病定案)。
- **考卷节奏(franco 2026-07-06)= checkpoint 抽查制,两级**:平时追踪 Isaac 训练内曲线;
  每个存档落盘后后台自动"导出+**bank 考卷**"(筛选级,纯 CPU 无规划器,~25 分钟)攒低频
  曲线;**判卷点/候选存档再过 Gate 3B 全链路考**(入账级最高档:假球→真规划器→C++ 运行器
  →厂商 MuJoCo 整条真机链,jiayi 的 Gate 3 改造版,见阶段 1 表 1l)。信号档臂到线考一次;
  长跑每 1000-2000 步抽一个、峰值区加密。
- 落地状态:`virtual_return_rate` 已实装(2026-07-06 merge);虚拟球任务天生就有,
  DeployParity/Hitter 通过"只记分不发奖励"开关(`vb_metrics_only`)也有了——**全谱系训练时
  都看得到上台率曲线**。注意该开关会改变随机数流:开了之后的新跑与旧跑不逐位可比
  (分布相同,同批对照不受影响);S1/产品线走虚拟球任务,不受影响。

## 排序法则(2026-07-06 定;为什么队列长这样)

每个实验的位置 = **结论保质期**(后续改动会不会作废它)× **解锁量**(多少后续工作等它)。
奖励结构定义所有其他实验的尺子 → 最先;出题/课程其次;数据源(动作对)第三;
权重微调 = 结构定版后的抛光;抗噪/部署加固 = 预期不改学习结论,最后整包测。
推论:**结构未定前,别扫权重、别微调锚点**;新想法一律先过"机制检查→信号档"梯子。

## Runtime Estimates (RTX 5090, measured 2026-07-03)

| Job | Cost |
| --- | --- |
| DeployParity training, 4096 envs, solo GPU | **~2.0-2.2 s/iter → 2000 it ≈ 1.2 h · 4000 ≈ 2.4 h · 8000 ≈ 4.7 h · 12000 ≈ 7 h · 20000 ≈ 12 h** |
| Co-running 2 jobs on one GPU (measured) | each job ~20-25% slower, TOTAL throughput ≈ +37%; memory fine (2×~7 GB of 32 GB). **Stagger starts ≥60 s** |
| 算力手册 FINAL | 4096 envs 定版;关键路径独占卡;~~广度实验最多 2 任务/卡;3 任务/卡不增反降 ✗~~ **franco 2026-07-08 深夜改规:消融波拉满 3-4 任务/卡**——"测得多"优先于单臂墙钟(单臂慢 25-45% 接受);独占仍留给跑到底/判卷关键臂。**07-11 现批复验**:每卡 4 条约 22.9–23.2 GiB、util 87–97%，六卡都已实跑，不再把一卡一条叫“跑满” |
| Kit boot + env build (per run) | ~2 min |
| Mechanics check (512 envs, 25 it) | ~3 min |
| ONNX export (play.py 原生路,占 GPU 槽) | ~4 min |
| MuJoCo 考卷(mjeval venv,纯 CPU) | ~25 min/checkpoint(与训练并行) |
| Reference lineage | shipped models ≈ 9-25k it; treat <8k as immature for deploy protocols |

## Team

| Person | Focus |
| --- | --- |
| franco | Direction, priorities, arbitration |
| dongc1(jiayi) | Hitter 步法线 + 防摔修复栈;simtoreal2 谱系(已审计合并进 main)。**07-06 起在本地机器训练,不占云上算力池** |
| yikang | Deployment; sim/env alignment; 出题器 |
| claude (franco's agent) | Foundation: infrastructure, A/B experiments, doc/code hygiene |
| Codex | NOW 主线收口;V5 专业迁移、Phase 加速、正式判卷/部署合同与跨链安全 |

## Run-Name Legend(人话对照表 — 报告里不再用裸字母)

| run_name | 人话 |
| --- | --- |
| r3_P2_product | 「产品线」:虚拟球(奖励打到落点、压出球旋转)+ 动捕毛病仿真 + 奖励自动收紧 + 上拍收尾起手。composite 峰值 0.908@9540;现役打包模型 = model_9600 |
| s1w2_A_main | 「主攻臂」:标定动作对 + 反解题库 + 奖励锚定指令 + **拍面指令观测(179 维)** + 手腕不抄录像 |
| s1w2_B_noobs | 「减观测对照」:同主攻但不给拍面指令观测(175 维)——量指令通道净贡献(A−B) |
| s1w2_C_targets | 「只换题库」:题库+腕踢除,拍面奖励仍锚录像——量反解目标单独净贡献(C−D) |
| s1w2_D_base | 「产品线原样锚」:hopex 动作对+默认相位,无题库(=P2 真条件对照) |
| s1w2_E_weight | 「拍面奖励加权」:主攻 + 拍面奖励权重 ×3(0.5→1.5)——权重方向粗标定 |
| s1w2_F_bhf17 | 「反手换锚帧」:主攻但反手锚 = f17(训练最优帧 vs 真值帧,A−F) |
| s1w2_G_v4mo / s1w2_H_obmo | 「动作源对比」:v4 对 / 斜录对(标定版),各配自家题库 |
| s1w3_base(待发) | 「基准臂」:v5 标定动作对 + 现行配方(反解题库出题 + 拍面指令观测 + 拍面奖励锚定指令 + 手腕不抄录像 + 防摔硬化)。**一臂两用**:奖励轴的对照 + 动作组轴的 v5 成员 |
| s1w3_window(待发) | 「击球窗臂」:基准 + 击球奖励分通道给时间窗(触点窗紧 0.02s,拍面/拍速窗宽 0.1s)——"触点要准,挥向挥速给余量" |
| s1w3_wweight(待发) | 「击球窗加权臂」:击球窗 + 拍面奖励话语权加档(jiayi 线用过约 ×10 的旁证) |
| s1w3_course(待发) | 「课程臂」:基准 + 难度课程——题库按"需求拍面离老师拍面多少度"排序,先只出最容易的 30%,学动了滚动扩窗 |
| s1w3_v4teacher(待发) | 「v4 老师臂」:换 v4 标定动作对(hopex 视频重跑标定版;标定后轻重球全 100% 可回,最稳的一对)× 基准配方 |
| s1w3_oblique(待发) | 「斜录老师臂」:换斜录标定动作对(实战击球视频)× 基准配方;点火前置=clip 几何转正 |
| s1w3_main / blind / face3x / face10x / softteacher | 「第一波重跑五臂」(已停,正手成对差定型):主攻 / 蒙眼 / 拍面权重臂两档 / 模仿降权。**⚠ 07-08 标签更正**:face3x 实跑权重 1.5=主攻×0.3(实为降权臂)、face10x 实跑 5.0≡主攻(**纯复刻臂=免费噪声尺 0.01**)——live 基线是 yaml 的 5.0 不是类内 0.5 |
| s1w3_fixA / fixB / fixC | 「反手修复臂」:慢放 0.8×(**判死**,击球率 0)/ 换锚 f17(**判死**)/ **换 v4 反手(赢家,反手上台率 0.867/摔 0.006,收卷中)** |
| s1w3_fixE_bhtrim6(yikang) | 「掐头臂」:v5 反手掐前 0.12s(trim6)——反手上台率 0.8415/摔 0.270;与修C 对判定反手谱系 |
| J1-J6(jiayi) | 「jiayi 参数正名臂」:合并审计出的未消融参数逐个 A/B,见 jiayi 参数正名臂定义;排程走统一队列 |

## 术语人话表(第一次出现的术语以这张表为准;新术语先上表再使用)

| 术语 | 人话 |
| --- | --- |
| 回球率(北极星) | 真球考试:场馆分布来球,策略虚拟回球,合法落到对面台的比例。现状 0% |
| 上台率(训练内,virtual_return_rate) | 训练时每次"到点击球"算一次机会:打到球+过网+落对面台内才计 1。**训练主看这条曲线** |
| 击球率(训练内,virtual_hit_rate) | 到点的机会里真的碰到球的比例(辅助曲线) |
| composite(三合格率) | 击球帧位置/速度/拍面三项同时达标比例——**只作诊断**,不判生死 |
| 训练内 vs 考卷 | 训练内=Isaac 虚拟球快速估;考卷=MuJoCo 重考。**入账以考卷为准** |
| 信号档 | 短跑对比:2000-4000 步(1-3 小时),只回答"这个想法有没有苗头" |
| 跑到底 | 12000 步(约 7 小时)长跑,出能考门禁的成熟存档 |
| 机制检查(mech check) | 3 分钟小跑:确认开关真的生效、程序不崩、**奖励收入账目正常** |
| 热启(warm start) | 从练到一半的存档继续练;checkpoint(ckpt)=训练存档 |
| 题库/反解目标 | 每道题(来球)先用物理反解算出"应该用什么拍位/拍速/拍面接",拿这套可行解当训练目标——治"练了 1.3 万步落点奖励从没发过钱"的病根 |
| 拍面指令通道(179 维契约) | 观测里加 3 个数告诉策略"这颗球需要什么拍面朝向"(+1 个预留位)。以前策略根本看不见题目 |
| 击球窗分通道奖励(SMASH) | 击球奖励按通道给不同时间窗:位置窗紧(0.02s)、朝向/速度窗宽(0.1s)——"触点要准,挥向挥速给余量",防手腕尖峰 |
| 奖励收入记账 | 每项奖励的实际收入/占比进机制检查与判卷的固定输出——死通道(装了没通电的奖励)当场暴露 |
| q_des 限位剪切(qdes_clamp) | 训练时把超出关节物理范围的目标角剪掉,和真机一致——不许策略靠"喊超范围角度"多要力矩。**2026-07-06 起默认开**(jiayi 发现:不剪切的产品线在 MuJoCo 门禁里根本站不起来);关闭仅限老配方复现对照臂 |
| 脚部姿态惩罚(foot_orientation) | 不许拧着脚打球:髋偏航/髋侧滚/踝侧滚贴参考动作。jiayi 值 -0.3,默认 0(待消融) |
| 支撑脚滑移(foot skate,判炸器检查项 7) | 参考动作里"名义踩地的脚"在水平打滑——摔倒一等杀手(yikang 取证+复测定案 07-09):策略照着滑动的支撑脚学,蹬地窗口踩不实就摔。判级:支撑帧(脚离自己最低点 <3cm)滑速峰 >0.15 m/s FAIL / >0.05 WARN;头 10 帧单列走掐头路;中段 FAIL=腿姿病(遮挡幻觉/重定向),trim/慢放救不了,只能腿移植或重拍 |
| 挥拍段限定模仿(swing-only) | 等球期间不再模仿录像(等球参考帧是垃圾:半蹲瞬态)——防摔修复栈一部分,已全谱系生效 |
| 关节静摩擦(训练植体) | 训练物理里补上真机每关节 0.4-2.4 Nm 的静摩擦(以前是 0)——等球摔倒的根因修复,已全谱系生效 |
| 熵系数(entropy_coef) | 训练时"随机试新动作"的劲头。全局基线 0.01;jiayi Hitter 线钉 0.015(待消融) |
| 变体库/老师适配器 | 把老师动作按题目现场改造(转向/变速/拍面 morph)或预烤成库检索——让老师能示范题目要的动作 |
| 难度课程(curriculum) | 按"需求拍面离标定拍面多少度"排序题库,先练最易的,滚动扩窗 |
| 预警时间(target reveal lead) | 从"策略第一次看到有效目标"到"必须触球"的时间。真机上=规划器出有效计划→击球;训练里现状=整个起手时长(正手 1.3s/反手 0.9s,v5 对 0.75/0.42s) |
| 等待混合采样(hold mixture) | 等球时长大部分抽短档(保持击球密度)、小概率抽长档(覆盖长等待)——治"长等待稀释训练效率"(jiayi 均匀 [50,400] 把击球密度砍半的教训) |
| checkpoint 抽查(MuJoCo 曲线) | 每个训练存档落盘后,后台自动导出+MuJoCo 考卷,攒出一条低频"考卷曲线"与 Isaac 高频曲线并排看——平时追 Isaac,节点看 MuJoCo |
| q10 / q50 | q10=每侧 10 题的快速方向卷，只看有没有长进，不许停训/晋级；q50=每侧至少 50 题的同卷决策考试 |
| SZ / SP / LZ / LP | 四个从零训练格：S=共用正反手拍面约定，L=旧的正反手异号约定；Z=零关节摩擦，P=历史非零摩擦数字直填。只有 SZ 是当前执行协议 exact/formal；P 不是标定摩擦，其余只诊断 |
| SC / calibrated plant | 未开训的“共用拍面+语义正确标定摩擦”格；必须先有一个物理摩擦模型和 PhysX/MuJoCo 两个独立 adapter，不能把 SP 改名就算 SC |
| causal / fresh | causal=从旧存档接着练，能看方向但谱系不纯；fresh=从零训，才可能成为新正式谱系 |
| carry-state | 下一拍直接继承上一拍结束时的真实身体状态，不传送、不 reset |
| guard reset(非物理保护收尾) | 判卷器因跟踪包络等保护条件结束本题，但没发生倾倒/骨盆下沉这类 physical fall。单题卷允许记账，但不能用来证明连续恢复稳定 |
| TOPP 重定时 | 不改挥拍的空间路径，只在关节速度/加速度/力矩/平衡约束下重排每段时间，把太长 clip 压到最短可行时间 |
| 视频 intake / GVHMR / GMR | intake=按字节和 SHA 登记原视频；GVHMR=从单目视频估人体动作；GMR=把人体动作重定向成 A3 机器人关节轨迹 |
| exact / formal target | exact=训练与判卷执行合同能逐项对上，不等于真机物理已对齐；formal target=本轮预先指定可进决策卷的格 |
| Gate 3(jiayi 的全链路闭环) | **人话完整版:上真机前的"彩排考试"**。平时的判卷(bank 考卷)只考策略本身:直接把"这颗球要什么拍面/拍速"喂给策略,像开卷考。Gate 3 考的是**整条真机流水线**:在厂商官方 MuJoCo 仿真里,把真机上会跑的每个环节原样接起来——①发一颗(假)球 → ②**真正的规划器**看球算"该用什么拍面拍速" → ③**真机同款 C++ 运行器**把指令发给机器人 → ④仿真机器人执行。任何一环有 bug(时序、消息格式、坐标系)都会在这里暴露,而 bank 考卷看不见。jiayi 建的,现版判据="稳不稳"(接得上战、挥得完、不摔),已用 10 个固定落点各打一回合全过 |
| Gate 3B(上台考版,**建设中,2real 最高优先**) | Gate 3 只判"稳",不判"打得好不好"。Gate 3B = Gate 3 底座 + 两处改造:①发球不再是固定 10 个点,改成**按当前训练阶段的来球分布随机采样**(考卷和课本同分布);②外挂一个判分器算**击球率/上台率**(Gate 3 用的是假球、没有拍球接触物理,所以判分器在击球瞬间取拍面的真实位姿/速度,套场馆标定的接触模型把球的落点推算出来)。它是**入账级最高档**的尺子=离真机最近的一次全链路模拟考;分工 jiayi 底座/接线、claude 采样器/判分器 |
| 加固包 | 部署抗噪项打包测(相位噪声/感知毛病/减速塑形等):预期不涨分,判部署轴,整包过/整包拆 |
| HER 目标重放 | 把策略自己达成过的击球状态混进新题——目标永远"确实做得到" |
| eval-B / 真球考试 | `--target-source venue-balls`:场馆实测分布出题的 MuJoCo 考卷(自带反事实列) |
| 反事实回球率(cf\_\*) | 同样的位置/拍速,换上"应有拍面"再虚拟回球一次——量拍面单项损失多少 |
| 门禁 | 上真机前必须过的仿真考试(官方口径 = 厂商 AGI-MuJoCo)。⚠07-09 考证:真机≈implicit PD+力矩限幅(Isaac 训练配置反而最忠实);厂商 sim 是 explicit Euler=数值上偏严——implicit 挂才是大概率真问题,explicit 挂先疑数值伪影但仍拦上真机 |
| 契约日 | 改策略观测格式的大日子(175→179/181),训练端和 C++ 端要同步改、攒着一起改。**候选消融(franco 07-07):参考观测去下半身**——看全身但只奖励上半身可能没意义甚至有害(等球换代事故的结构根源就是策略看着下半身参考做平衡);腿该由任务驱动(站位指令),不该由"舞谱"驱动。契约日拼桌时做一臂:参考观测只留上半身。**07-09 第一性原理批次追加两候选**:①参考观测整体 actor→critic-only(HITTER 同构,比只遮腿更彻底——部署契约应从"控制器所需信息"出发定义,参考是训练脚手架不该泄漏进接口);②预警时间显式观测(time-to-contact 直给,不藏在 clip 相位里) |

## 🏆 当前胜利组合(常驻;每出一个裁决就更新这里,谁赢谁进)

**产品配方 v2(2026-07-05 定)** = 虚拟球·消旋 + 动捕毛病仿真(A1 延迟/噪声 + R8 感知三毛病)
+ 奖励自动收紧 + 上拍收尾起手 + 2% 中途改目标,从消旋臂 1500 步热启:

- composite 峰值 **0.908-0.910 @ ~9540**(r3_P2_product);现役打包模型 = **model_9600**(指纹已核)
- R8 感知三毛病:白送的抗噪(跑到底 0.905@10.6k 再证)→ 已并入
- R10 换种子:0.914@10.3k → 平台可信;"选峰值存档"铁律的来源(现在换尺:选**上台率**峰值)
- **⚠ 尺子警示**:以上全是 composite 时代的数字;上台率口径下产品线现状 = **0%**(结构病:
  拍面看不见题目 + 落点奖励从没发过钱)。四阶段计划就是治这个的
- **已拒绝**:R11 中途换招@0.002(命中率税真、抗摔收益无)、R6 剪短模板(无优势)、
  R16 手腕放开单独用(掰不动拍面;正确位置 = 配指令通道当执行机构,已进 S1 主攻臂)
- **simtoreal2 合并警示已关闭(2026-07-06 审计合并)**:未消融参数全部旗标关/退回基线,
  jiayi 谱系配方钉在他的任务 yaml 里;防摔修复栈(静摩擦/等球站姿参考/挥拍段限定模仿)按
  正确性修复全谱系生效——**跨波比较必须同批对照,老波数字只作方向参考**

## 当前在跑 + 第一波判读分支(预注册,2026-07-06)

**⚠ 本节为 wave-2 历史快照(07-08 现况:池上只剩修C 独跑收卷,其余全停——见统一队列"跑着"行)**

**在跑(gpu1/2,4 槽满)**:只换题库(C)/ 拍面加权(E)/ v4 对(G)/ 斜录对(H);
反手换锚(F)排队等 C 的槽。已到线待判:主攻(A,接力至 13600)/ 减观测(B)/ 产品线锚(D)。
jiayi 已转本地训练,云上 6 卡全归统一队列。
第一波共同配置、发射工程记录、止损复盘的原文在 PROGRESS.md 归档;考卷命令见 runbook。

**⚠ 第一波价值降级(franco 2026-07-06 判断,成立)**:第一波臂训练用的题库还带着
"挂网答案"缺陷(反解没查过网,同类扫描曾拦下 249/256 挂网解)+ 全部臂没开 q_des 剪切
(当时默认关)——**训练目标部分是错的,绝对分数不作数**。残余价值:①同批成对差仍可
方向性参考(主攻−减观测=指令通道、只换题库−产品线锚=反解目标——两边吃同一个坏题库,
差值大体公平);②工程机制已验证(179 维热启手术、自动队列、考卷链)。**正式的第一波
结论,等 yikang 修好球/网参数、题库重生成(v2)后按原设计重跑**——已进统一队列。

**判读用尺(都按 MuJoCo 考卷;训练内上台率并列参考;起飞线 ≥50%,分母一起记;
本表对题库 v2 重跑后的正式第一波生效,当前这波只看成对差)**:

| 第一波结果 | 判读 | 下一步 |
| --- | --- | --- |
| 主攻(A)回球率 ≥50% | 配方成立 | 巩固波提前:赢家跑到底+换种子+动作源复验;S2 准备加速 |
| A 未过线,但拍面-vs-应有误差比产品线锚(D)明显缩小 | 方向对、力度不够 | **奖励结构波(W2)= 主路**,照点火队列发 |
| A ≈ D 纹丝不动 | 先别烧卡 | 查账:拍面指令奖励发钱没有(奖励收入记账)?obs 接线对没有?机制层排查完再谈新臂 |

横比对凑齐一对判一对(A−B 指令通道净贡献、C−D 反解目标净贡献、A−F 锚帧、G/H 动作源),
不等整波;**动作源结论在奖励结构定版前只记"暂定"**。

## wave-2 训练侧收卷监测(2026-07-06 深夜,yikang;官方数只认 MuJoCo 考卷)

7/9 臂跑满 13600:face-command 臂群(A/B/E/F/G)击球帧拍面误差 **21-26°**、虚拟过网 **0.90-0.93**
(历史基线锁死 42-43°/回球 0);**E(face 权重 1.5×)训练侧领跑**(过网 0.928);B(无 +4 obs 对照)
训练侧与 A/E 持平——obs 通道的真价值等考卷的未见题泛化裁决。⚠ 口径警告:C/D 的 2-3° 是对 clip 面
的易口径,与 face-command 臂的对需求面口径**不可横比**。两件事要人:① **A_main_r2 挂起**
(12986/13600,进程活/GPU 0%,franco 处置);② **H 斜录臂异常——取证已结(2026-07-06 深夜,yikang):定性 = 斜录源结构性不适配,非配置漏、非发射事故**。
证据链:s1_obcal bank 闭环复检干净(反手 n=782 全落台全过网、err_med 1.8cm;meta grip/rally 旗全真);
H 训练侧捕获健康(virtual_hit 0.945、strike_pos_err 1.8cm、回合长 465)→"零捕获/配置漏"假说排除;
拍面误差稳在 40.9°≈该源 rally_yaw(±55-60°)→ 机制 = **rally-yaw 转正张力**:bank 按直线几何出题要拍面,
模仿项却在斜线旋转的 clip 上,冲突幅度∝|rally_yaw|(直线源 A/E 的 21-26° vs H 的 40.9° 与此一致)。
处置:判卷 H 照常进卷,但动作源结论必须标注该张力;若要救斜录源,方向是 clip 几何同步转正(而非重跑)。
~~判卷阻塞项仍是 1a-0~~ **已解除(07-07)**:bank 题源落地并在 pod 实跑通过(v2 考卷,守卫/分母报表全对,首考=减观测对照臂)。

## planner 延迟×精度基准(2026-07-06 深夜,yikang;分支 `planner-latency-bench` 8c1b34e,默认字节不变)

**0.4s 成因定论**:`StrikeSpecPlanner.solve` → 每次 LM 迭代全程重积分(1kHz 纯 python Euler)。
一次冷解 = 8 迭代 × 57 次全飞行 rollout = 3.6 万步 × 15.4µs = **451ms 中位**;单步 ~60% 在
`np.cross`(Magnus)。node.py 里它已被限流 1Hz 只做诊断——部署 tick 真正在跑的是镜像律管线。

**"jiayi planner" 考证**:不存在独立实现——部署三段管线(估计→预测→镜像律 plan)即 jiayi 6-17
所写(ed5cca9);"我随手做的" RacketTargetPlanner = 该管线第三段。C++ runner 只消费
`/racket/command_flat`,无 planner 数学。所以对比是**镜像律家族 vs 逆解家族**:

| 候选 | 中位 ms | p90 ms | 落点误差中位 |
|---|---|---|---|
| 镜像律(user_quick / jiayi 管线) | 18.6 / 22.7 | ~106 | **~75mm(盲旋+无切向,真实代价)** |
| StrikeSpec LM 基线 | 451 | 788 | 18.9mm(surface 口径 3.3mm;19 里 ~16 是 z=0 vs z=R 口径差) |
| **赢家:fastnp+远粗近细20ms+热启动+iter6** | **15.2** | **41.8** | 18.6mm(精度不掉,30-45×) |
| torch 批量 | batch1 1222 / batch64 摊销 24 | — | 1.9mm → **题库后端**(kernel-launch bound,非单题部署后端) |

N=200 场馆题,oracle=venue RK4@2ms;Mac CPU 是代理尺,pod/SoC 复跑:
`python3 hope_ws/src/hope_planner/benchmarks/benchmark_planner_latency.py --n 200 --seed 0`。
**部署建议**:tick 路径换 fast 变体(flag 门控已入库,默认关)+ Stage-2 predict 同 flag(3.9→0.96ms)
+ 重规划降频 30-50Hz;后续杠杆:手写 cross(单步 ~60%)、C++/torch 后端。含 87 tests 绿(已独立复验)。

## planner 延迟×精度基准已结(2026-07-07,yikang;分支 planner-latency-bench,commit 8c1b34e)

**0.4s 成因定死**:StrikeSpecPlanner.solve 一次冷解 = 8 LM 迭代 → 57 次全飞行 rollout → 3.6 万步
1kHz python-Euler(15.4µs/步,np.cross Magnus 项占大头)= 中位 451ms;灵敏度差分再 +87ms。
**"jiayi 的 planner"考古结论:不存在独立实现**——部署管线三段(估计→1kHz 预测→镜像律 plan)本就是
jiayi 6-17 所写(ed5cca9),C++ runner 只消费 /racket/command_flat;所以对比 = 镜像律家族 vs StrikeSpec 家族。
**N=200 场馆题实测(Mac CPU,oracle=venue RK4)**:镜像律家族(user_quick 18.6ms / jiayi 管线 22.7ms)
落点真误差 ~75mm(盲旋+无切向接触);StrikeSpec 基线 18.9mm 但 451ms。**胜者 = numpy 批量探针
+ 远粗近细 dt=0.02 + 上 tick 热启动 + 6 迭代 + 免每 tick 灵敏度:中位 15.2ms / p90 41.8ms,精度持平基线
(30-45×提速)**;油门到 dt=0.04 → 10.1ms。torch 批量 1.9mm 但单题 1.2s(kernel-launch 束缚)→ 训练侧
题库专用,部署单题别用。落地建议:部署 tick 走 FastStrikeSpec 组合 + Stage-2 预测同旗(3.9→0.96ms)
+ 30-50Hz 重规划间用缓存 spec;后续杠杆 = 手写 cross 替 np.cross(~60% 步成本)或 C++ 化。
诚实边界:Mac 为代理,上 SoC/pod 重跑 benchmark_planner_latency.py。旗默认全关,off 路径已验位等价。
**生产化已落地(07-07,commit b3961a2,已独立复验 97 tests)**:胜者配方进 hope_planner 包
(FastStrikeSpecPlanner 复用现有物理零复制;np.cross→手写 cross3 位级一致,5200 例过验,单换它就把
标量基线 451→165ms);node.py 逐 tick 快解旗控(40Hz 重规划间供缓存 spec)。生产实测:快解
**9.8/28.5ms @ 18.55mm**(比原型再快,cross3 惠及共享积分器);**整 tick S1+S2+快解 2.0/2.3ms**
(旧 S1+S2+S3 = 19.1/104.6ms,且落点还更好 48.9 vs 72.4mm)。off 指纹 SHA-256 与 8c1b34e 全等。

## 厂房硬化 × wave-2 兼容性(2026-07-07 凌晨,yikang;冻结探针实证,团队级)

**发现**:wave-2 全部 9 臂(00:07-11:24 从 b9d0eec 发射)训练于**软厂房**(零关节摩擦、无 q_des 剪切);
当天下午厂房被硬化并进 stage1(305d0e8 引入 f921c5b 关节静摩擦对齐 AGI MuJoCo:膝 2.43/髋 1.20/踝 1.4/
腰 1.7 Nm;56ebf08 qdes_clamp 默认开)。**冻结探针**(lr=0+fixed,E 臂 model_13599,逐项同配方):
在 b9d0eec 上 virtual_hit **0.9997**/摔 0.27;在 7e3be78 上 **0.63**/摔 0.87;qdes_clamp=false 份额探针
仅回 0.72/摔 0.85 → **摩擦主导,剪切次要**——jiayi f921c5b 提交语描述的故障("零摩擦策略 stiction 下
last_action 自激 3-5s 倒")反向重演。**推论**:① wave-2 臂在现行树上判卷/考试预演/warm-start 读数全失真,
评估环境必须钉 b9d0eec 同侧;② wave-2 臂本就是零摩擦产物,在 AGI MuJoCo 官方考卷上自带已 root-cause 的
hold-fall 缺陷——判卷预期照此校准;③ S1 wave-3 必须硬厂房发射(重训自然吸收摩擦+剪切)。
**Phase B 仪表连带结论**:实战策略下开/关旗对照再证无扰(0.526 vs 0.522);全链 26740 发球(3.9mm)/
**7972 真击球**/370 回球吃满;但今日 gap 读数(0.18m/0 过网)**隔离不采信**——病臂(0.63 命中、摔 87%)
打出的数据;富样本 gap 判卷搭 wave-3 硬厂房臂的车。方法论沉淀:冻结探针(lr=0+schedule=fixed)= 厂房
兼容性的标准试纸,一发 6 分钟。

## 四阶段总计划(2026-07-06 重排版;每阶段 = 开发任务 + 测试臂,过线才升段)

| 臂 | 配置(完整命令附后) | 回答什么 |
| --- | --- | --- |
| R0 基线 | `task=HOPEPingPongDeployParity`(main 默认) | 合并后的参照点 |
| R1 虚拟球·上旋 | `task=HOPEPingPongVirtualBall`(yikang 默认,落点30/过网20/旋转5,拍速法线降权) | 物理奖励栈是否成立 |
| R2 虚拟球·消旋 | R1 + `task.racket.vb_spin_mode=minimize`(franco 第一阶段:不奖励球质,奖励落点+出球旋转最小) | 两种旋转哲学谁先学会站稳打准 |
| R3 斜录数据 | R0 + `motion_file=/workspace/shared/motions/hope_forehand_oblique.npz motion_file_2=..._backhand_oblique.npz "task.racket.strike_phase_per_clip=[0.432,0.495]"` + 下方专属采样框(**⚠ 正手 0.368→0.432 人工纠错 2026-07-05:旧值=触球前加速段,球还差 3 帧才到拍;框按新帧重算,见 strike_annotations.yaml**) | 实战动作源是否更好(franco 预判:是) |
| R4 组合 | R1/R2 胜者 + adaptive_sigma + post_swing(可再叠 R3 若其胜) | 产品候选 |

**病根回顾(一句话)**:0% 回球 = ①策略看不见题目(拍面无指令通道)② 训练目标本身不是
"能回球的解"(完美跟踪也 0% 入界)③ 打球奖励从没发过钱(信号比 ≈1:5000)。
方向 = 把"教打球"从结果空间搬回控制空间:让已被证明有效的跟踪奖励指向逐球反解的应有拍
状态;落点奖励降级为验证奖金。阶段 0(相位校准/握拍标定/题库,0a-0k)已全部完成,
原文与结论存 PROGRESS.md 归档;关键遗产:反手先行、v4 对最稳、题库答案全在挥速锥外
(中位差 34°——老师没示范过答案,这正是奖励结构波要解的题)。

### 阶段 1 消融路径(人话一页,franco 2026-07-07 要求;波次代号只是编号,内容以这里为准)

**阶段 1 要证明的一句话**:给策略看题目(这颗球需要什么拍面)+ 给物理可行的答案当训练目标 +
奖励改按答案付钱,策略就能把固定点的无旋球打回去(判据:MuJoCo 考卷回球率 ≥50%)。

**已发生:昨天的第一波(八臂)大部分作废**。三个毒:①题库 v1 带挂网答案(训练目标部分
物理错误);②全部臂没开限位剪切(后来发现这是站立级缺陷);③发射时的相位坑(当场止损
重打过一次)。绝对分数全部不作数;成对差只当低置信度方向参考;留下的真收获是工程机制
(179 维热启手术、自动队列、判卷链)。

**接下来三波(每臂人话)**:

**第一波·重跑+奖励结构合体**(硬依赖:yikang 的题库 v2 = 球/网参数修好版,已生成):
| 臂(人话名) | 改什么 | 回答什么 |
| --- | --- | --- |
| 主攻臂 | 看得见题目(观测+3 个数=需要的拍面朝向)+ 题库可行答案当目标 + 拍面奖励按答案付钱 + 手腕不抄录像 | 整套方向成不成(主判据臂) |
| 蒙眼对照臂 | 同主攻,唯独**不给**拍面题目观测 | 策略看不见题目时差多少 = 观测通道的净价值(主攻−蒙眼) |
| 只换题库臂 | 题库可行答案当目标,但拍面奖励仍按录像老师付钱 | 只换题目值多少 = 反解目标的净价值(只换题库−原样) |
| 产品线原样臂 | 一切照旧的老配方 | 基准地板 + 同批对照 |
| 击球窗臂 | 主攻 + 击球奖励按通道分时间窗(位置窗紧 0.02s、拍面/速度窗宽 0.1s) | 窗口结构值多少(防手腕尖峰) |
| 比值臂族(franco 07-07:比值很重要)——现货三臂已发:话语权×3 / 话语权×10 / 模仿降权×0.5 | 同一个比值从两边逼近:**抬任务侧**(拍面奖励权重 ×3/×10)vs **压老师侧**(六项模仿整体 ×0.5)。三臂对上主攻即得"任务:模仿收入比"的粗曲线 | 老师话语权到底该多大;抬任务和压老师是否等效(不等效=交互项,有信息) |
| (期中读数 2026-07-08,it≈1.17万/1.36万;训练内虚拟球口径,入账以考卷为准) | 正手上台率:主攻 0.78 / 蒙眼 0.79 / 话语权×3 0.80 / ×10 0.77 / **只换题库 0.00 / 产品线原样 0.00**;全臂击球率 ~0.998;拍面误差:锚指令四臂 ~18°,只换题库 1.5°(**完美模仿老师拍面=0 回球,教科书级反证**)。初判:①题库+拍面奖励锚指令 = 正手 0→0.78 的整个跳变;②蒙眼≈主攻 → 固定点窄题距下"平均拍面"就够,指令通道的钱在阶段 2/3 才能赚;③×1/×3/×10 正手打平 → 固定点下比值敏感度低。**⚠ 判卷前置取证:反手全臂归零不是拍面——93% 反手挥拍触球前摔倒**(pre_strike_fall_rate_backhand 0.93;嫌疑:v5 反手参考猛[15.5 rad/s²=hopex 6 倍]×剪切新上线×帧0 +19.5° 未落地×热启断层;与考卷反手异常同源嫌疑)。模仿降权臂第二批漏发,已补排并点火(motion_scale 旧接线踩到"锚位置项已移除=None"崩溃,当场修掉)。**07-08 franco 决策落地**:①只换题库臂/产品线原样臂答案已到手(0 回球平台),**已停臂腾槽**(存档留在 ~1.17万 步可补考);②**反手归因完成**:上一波(剪切关)反手摔率就有 52-54%、完成率同样 0 → **病根=v5 反手参考本身不可平衡执行**(关节加速度 15.5 rad/s²=hopex 6 倍),剪切开后 52%→93% 只是加重;"重启"救不了。**修法梯子**(便宜→贵):a. 反手参考慢放(变速机制改 per-clip 固定档);b. **换反手动作=最快路**(v4 反手温和且轻重球标定全 100%;混对 v5正手+v4反手 需先合卷,同源同锚铁律);c. 制度修复=登记表 gate 加全身限位+速度余量审计 + SMASH 式跟踪可执行性过滤(重定向只保证位置限位合法,**速度/动力学可行性管线里没人查**——本次教训);d. 反手模仿容差放松起步。动作组三选波(v4 对在列)本来就是 b 的正规实验 | 期中止血/判卷准备 |
| **反手修复消融(franco 07-08 下令跑起来;归因已完成)** | **归因**:反手挥拍完成率两波都是 0——上一波(剪切关)摔率 52-54%,这波(剪切开)93%:**病根=v5 反手参考本身不可平衡执行**(加速度 15.5 rad/s²=hopex 6 倍),剪切只是加重。**"重启就好吗"——不是**,重启不改参考。修复臂(全部=主攻配方+单一修复,同批对照=在跑的主攻臂):**修B 反手换锚 f17**(相位 0.298,合并卷 v5 正手 818 题+f17 反手 791 题已过守卫门,零代码,已排队等槽)→ **修A 反手慢放 0.8×**(新 per-clip 慢放旗标已实装,参考慢放、题目答案不缩)→ **修C 反手换 v4 动作**(v5 正手+v4 反手混对;v4 卷 v2 生成→合并→点火链在跑。⚠ 顺手抓的坑:v4 登记项(烤入标记/相位/打向)困在 pod 上一个**未推送的本地提交**里(b9d0eec,只在 franco 的分支 worktree)——远端分支顶端没有,yikang 检出也没有,生成守卫两处同因拦截;已把登记项移植进一次性生成 worktree 解锁。**07-08 更正:该提交其实已在 origin(分支 s1-registry-v4cal),真残留=v4_cal 登记 16 行没进 main**,合 main 后守卫不再拦,见队列 0.6 行)→ 赢家组合(如 f17+慢放)。**制度学习(retargeting 为什么没拦住)**:重定向尊重位置限位(腕链审计零裁剪),但**速度/加速度与"边做边平衡"的动力学可行性无任何管线检查**——q_des 剪切剪的是策略不是参考,"合法但暴烈"的参考就这么漏进来;修法=登记表 gate 加全身限位+速度余量审计 + SMASH 式跟踪可执行性过滤(0i ④)用到原始 clip | 反手起死回生的最短路径 |
| **反手病根定案+修复裁决(07-08 深夜;详证见 TIMELINE 07-08)** | **归因三层收口(两轮对抗复核 CONFIRMED)**:主因=**GMR 重定向 IK 冷启动瞬态**(前 0.12s 四个腿关节超 URDF 速度硬限位——yikang 说法实锤;毛刺七条 clip 全有、仅 v5 反手超限;GVHMR 无辜:腿虽是遮挡下想象的但平滑合理,franco 目视"普通下蹲"没看错);次因=v5 反手全程深蹲过深(GMR 把人缩矮:骨盆 0.78 vs v4 0.97m)+掐头后挥拍仍激进(hopex 3.4×)。旧表述"v5 反手参考本身不可平衡执行"细化为上述两层;15.5 rad/s² 为前向差分口径,约 40% 来自瞬态段。**消融裁决(训练内)**:修A 慢放/修B f17 **判死**(击球率 0.000/摔 0.63-0.86);**修C 换 v4 反手=赢家**(反手上台率 0.867/摔 0.006 @11376,收卷后补 MuJoCo 考卷);fixE 掐头臂(yikang)0.8415/摔 0.270——**换温和底片 > 掐头激进底片**。**源头修复已落地**(pod GMR 分支 hope-frame0-warmup:帧 0 warm-up 到收敛+修"人帧 0 被丢"+逐关节 URDF 限位旗标;验证=首帧峰 15.89→0.81 rad/s、超限 5→0、触球峰保留 100.4%、基线逐字节复现;velocity-limit 裁决**默认关**——mink 限位是逐步信赖域会削合法挥拍 26%,限位检查放事后审计;⚠ 新工序坑两个:重跑管线必须 PYTHONPATH 指向 franco 的 GMR 检出、mink.solve_ik 传参 latent bug 已修)。**判炸器 L0 上线**(分支 `motion-feasibility-audit` 已推 origin,对抗复核 4 缺陷已修+测试 14/14:URDF 逐关节审位置/速度双口径/加速度/首帧健康;修复建议按 franco 语义"超了限位才掐、掐只对头尾、中段超速给慢放系数、都不适用整段拒收";实测 v5 反手 FAIL→建议 trim 前 6 帧与 trim6 吻合)。**"加速自杀"裁决=否**(被动摔:摔率开局即高不再升,每步净收入 +1.76/s 死了严格亏;但 99.9% 的"摔"实为跟踪包络终止而非倾倒——结构修法入增补臂 R-b) | 归因/修法全线收口 |
| **滑脚定案+一病两发(07-09,yikang 取证+franco 侧复测互证)** | 遮挡幻觉一个病根、两处发作:①头部毛刺(GMR 冷启动,warm-up 已治);②**蹬地窗支撑脚滑移**——想象腿的脚在"名义踩地"时以 0.30-0.35 m/s 滑动(v5rg/v5hA 反手实测),正好在蹬地窗口=摔倒的一等杀手;v5hL 钉脚移植 0.011、v4rg 真腿 0.010=干净("v4 脚真干净"=修C 高分的物理解释)。**正手遗留摔率(M1 0.60)不是滑脚**(实测 0.034),嫌疑=蹲深+热启过渡。fixE std 炸弹成因补全:被迫跟踪物理不可能的滑脚→策略学出高频抖动凑平衡→Isaac 容忍 MuJoCo 不容忍。**框架臂裁决**:R9b(遮腿观测+删缰绳 × v5hA)判死、R7(只遮腿 × v5rg)无效(fallB 0.819≈M1)——毒还从终止/RSI 通道进,资产侧修复(v5hL 族)是主战场。判炸器加检查项7(支撑脚滑移 FAIL>0.15/WARN>0.05 m/s)实装中;判炸器 v2=逆动力学力矩余量排队。**v5 救回现役梯队**:R9c(v5hLs 合体)watchdog 队首 → R9d(v5hLt 重定时,资产已产:bh acc 6.06→4.49)→ R9e(v5syn 时间律合成,在产)——机器人本位时序 A/B 消融见 docs/research/robot_centric_timing_2026-07-09.md | 病根四层全拆完 |
| **动作源 provenance 定论 + swing 第三源(franco 07-08 拍板)** | hopex 资产与 v4_cal **同底片**(真源=raw_video_hopex/*_v4.mp4,MD5+帧数对上;hope_backhand.npz 与 _hopex 逐位相同)——"v4=hopex 视频重跑标定版"口径为真,非登记事故,但**动作组消融里 v4 对 vs hopex 对不构成独立对照**。*_swing.mp4=同人另一段未使用录制 → **动作组消融成员定为 swing 对 / v4 对 / v5 对**(斜录候补,等几何转正)。**swing 试产完成**:修复版管线全链产出(98/108 帧,烤入残差 0.01°、腕限位零裁剪),**判炸器双 PASS=第一批完全干净过审资产**;正式入库差:触球帧登记(无球则走 hopex 约定帧惯例)、会话握拍复核(试产暂借 Rz5Rx40)、题库生成 | 动作组轴重定义 |
| **热启动假设裁决(franco 之问,07-08 深夜)** | "反手是不是热启动闹的?"——**否,已用对照证据裁决**:yikang 的从零谱系臂(硬植体、8192 环境、同一 v5 对)反手摔率 **0.97**,比热启臂的 0.93 还高;反手瘫痪与热启无关,**病根坐实在 v5 反手参考/任务本身**。修复消融照原计划(换锚 f17/慢放/换 v4),"从零"这格的证据已由他的臂免费提供 | 归因收口 |
| (3-4 任务/卡之争,07-08 实测) | yikang 说 3-4 个能跑——**"能跑"为真,"无代价"为假**:实测 3 任务卡上我们的 4096 臂各慢 25-45%(7-8 秒/迭代 vs 2 任务卡 5.6 秒),他的 8192 大臂自己很快;总吞吐大体持平,**但关键路径臂的墙钟被拖尾**。规矩修订:≤2 任务/卡仍是默认;第 3 个任务只允许"不赶时间"的后台臂且须在队列表标注,关键波期间禁止 | 算力手册增补 |
| **连续上台率长期追踪(franco 07-08 定)** | Isaac 侧:新指标 `virtual_return_rate_rally`(含正反手分身)已实装——**分母=挥拍起手数**,摔倒/没挥到击球帧都计失败,= 训练内的"连续对打"孪生数;MuJoCo 侧:checkpoint 抽查加跑部署仿真连打协议出曲线;bank 题源×连打协议评估器尚不支持(v1 限制)→ 开发项排队 | 长期曲线两条,新臂自动带上 |
| (已停臂台账 07-08) | 反手死的五臂已停(主攻/蒙眼/话语权×3/×10/模仿降权,存档留在 1.2-1.3 万步,正手成对差已定型可判);更早:只换题库臂/产品线原样臂已停(TERM→KILL,存档留在 ~1.2 万步可补考):它们的答案已定型——只换题库=0 上台(拍面被录像锚死),原样=0(旧世界地板);腾出的槽给了模仿降权臂与修复臂 | 省卡 |
| (比值现状读数,franco 07-08 求证;代码实值) | 现役配方**窗内**(击球窗 0.12s,窗外任务不付钱):任务合计:模仿 ≈ **2-3 : 1**(跟踪 4.0+0.5+0.5 × 核 + 三合格乘法奖金 5.0 + 触球帧落点 30/过网 20/消旋 5 一次性,对上模仿每步 ~3-4);**拍面单项:模仿 ≈ 1 : 9**(0.5 权重)——话语权×3→约 1:3、×10→约 1:1、模仿降权×0.5 再翻倍。⚠ 乘法奖金与落点奖金**打不回时收入=0**(老奖励沙漠),早期有效比≈1:1——常引导惩罚正是补这个;**判读一律按记账工具的实测"窗内任务:模仿收入比"读,不用解析估计** | 比值的地面真值 |
| **⚠ 比值读数与比值臂标签更正(07-08 审计,launch log applied 清单为证)** | 上行解析值与实跑不符:**所有 S1w3 臂 live 权重 = racket_position 14 / racket_velocity 10 / racket_normal 5**(DeployParity yaml 的 ARM A pin 经 Hydra 组合覆盖了 VirtualBall 类内 0.5/0.5 的重平衡设计);拍面单项:模仿实测收入比 ≈ **1:6** 不是 1:9。连带:比值臂按"基线 0.5"设计 → **face3x 实跑 1.5=主攻×0.3(实为降权臂)、face10x 实跑 5.0≡主攻(纯复刻臂)**;"×1/×3/×10 正手打平"正确读法="0.3×/1×/1× 打平"(1.5→5.0 区间不敏感),复刻对白送 **run 噪声尺 ≈0.01(正手上台率 0.78 vs 0.77)→ 判读信号门槛取 ≥0.03**。待 franco 裁决:基线权重本意=类 0.5 还是 yaml 5/10(差 10-20 倍,决定下轮比值档位;建议顺手给组合覆盖加 fail-loud) | 判读修正+防坑 |
| 常引导惩罚×2(需代码,随击球窗补发) | 位置/速度/拍面误差给**小而恒**的负分(挥不到也有梯度,exp 核远处饿死的解药)+ 击中/上台稀疏大奖金;两档罚强度 | 什么形状能"一直引导又说清要什么"(高击球率→高上台率→未来高球质);**比值铁律:引导罚每步 ≤ 模仿收入 10-20%,奖金稀疏而大** |
| 难度课程臂 | 主攻 + 题库按难度先易后难滚动开窗 | 先做易题能不能点亮学习 |

**第二波·动作组三选**(前置:等球锚旗标实现=半天;**成员 franco 2026-07-08 改版**):
上一波赢家配方**全部固定**,唯一变量 = 用哪套老师动作——**swing 对 / v4 对 / v5 对**,三臂同批,
各配自家题库、各用**自家首帧**当等球锚。回答:哪套老师最好学(先调 reward 再调动作组,franco 序)。
成员变更依据(07-08 provenance 定论):旧名单里 "v4 对 vs hopex 对"是同底片不构成对照;斜录对
候补(几何转正修好再进);swing 对=同人未使用录制、修复版管线试产已双 PASS(入库前置:触球帧
登记+会话握拍复核+题库生成)。⚠ 判读口径注脚:v4/swing/v5 都是同一个人,这组消融回答"同风格下
哪次示范更好学";**风格轴**的证据仍只有斜录(实战击球),留给候补臂。
**加两臂(franco 07-07:正好多做实验)**:v5 对 ± **遮蔽下半身参考观测**(遮蔽=下半身参考项喂
默认站姿常数,不改观测格式,零契约成本)——回答"看全身参考是不是没用甚至有害"(契约日正式
去维的前哨)。**发射前检查(franco):拍面指令随球更新重算已核**——题库写入在两条采样路径
末端原子更新(位置/速度/拍面一起换),且有 fail-loud 守卫(开指令没开题库直接报错);机制
检查摘要必须含 question_bank active 行。

**第三波·巩固**:动作组赢家跑到底(12k,独占卡)+ 换种子验稳 +(搭车)等待/预警起步档。
产出:能考门禁的成熟存档 → 阶段 1 过线判定。

**奖励和观测要不要一起调(franco 之问)**:分两层。**结构层是耦合的,必须一起定**——观测
通道(把题目给策略看)和奖励锚(按什么付钱)是一根通道的两半:奖励按答案付钱而策略看不见
答案=只能瞎摸。所以第一波的前四臂本来就是这两件事的 2×2 拆解(主攻=都换/蒙眼=只换奖励锚/
只换题库=只换目标/原样=都不换),一波跑完耦合就定死,以后不再动。**参数层是解耦的,分开调**
——窗口宽度、权重大小只改"付钱方式"不改"信息存在不存在",结构定死后单独扫(击球窗臂、
加权臂与主攻同批但各自独立)。

### 阶段 1「固定点养成」— 不同速度的无旋球,打回固定落点(进行中)

升段判据:MuJoCo 考卷回球率 ≥50%(分母一起记:可解率 85-94%、锥内 0%)。

| 顺序 | 类型 | 内容(人话) | 依赖 | 谁 |
| --- | --- | --- | --- | --- |
| 1a-0 | 开发 | **考卷工具补题源**(yikang 交接 07-06):`mujoco_eval_onnx.py` 加 `--target-source bank`——S1 判卷必须问 exam 卷的同源题(现有 boxes/venue-balls 都不是)。规格:装载走 `stage1_question_bank.load_question_bank`(强制校验烤入/转正标记,别绕过);逐题喂固定击球点 + 该题需求拍速 + **179 观测尾 4 维用该题需求拍面**(不是 clip 参考);输出连分母(可解率 kept/N + 锥内比例);网/界判据沿用现有常数。**✅ 已落地(07-07;07-10 重新加固)**:早版 BankExamSampler 是逐题出卷后“耗尽重洗”;现已由内容寻址的 immutable schedule、no-wrap、跨噪声档同题序和 all-attempt 分母替代。正式卷只收 schema-v3 exam split。附带坑:原生导出不产 obs 归一化与 std sidecar,判卷前先跑 make_std_sidecar(已进 runbook) | — | claude |
| 1a | 判卷 | 第一波 8 臂按预注册分支判读(上表)。**S1 臂契约名已注册**(`deploy_parity_face179`,发射不再传 contract=null;导出器元数据即刻可用,契约日 181 时与站位 2 维一起换名) | 1a-0 + 各臂到线 | claude |
| 1b | 开发 | **奖励收入记账**(半天;也是"A≈D"分支的排查工具) | 无 | claude |
| 1c | 开发 | **击球窗分通道奖励**(SMASH 整包,W2 门票;旗标默认关) | 无 | claude |
| 1d | 开发 | 难度课程 loader 窗口(确认现有 loader 能不能按难度开窗) | 无 | yikang+claude |
| 1e | 测试 | **W2 奖励结构波(4-5 臂,恰好一波)**:对照(W1 胜者续跑)/ 胜者+击球窗整包 / 胜者+击球窗+拍面权重(按 E 方向定档;jiayi 线拍面等效 ×10 是外部参照,梯子可搭高)/ 胜者+难度课程(先开最易 30% 滚动扩窗)/ J3 剪切采纳臂(搭车) | 1a+1b+1c(+1d) | claude |
| 1f | 测试 | **W3 巩固波**:W2 赢家跑到底(12k 独占卡)+ 换种子臂 + 动作源赢家复验臂(定版奖励下重验 G/H 结论) | 1e 判卷 | claude |
| 1g | 开发 | **适配器 v2 变体库骨架**(CPU):老师按题示范。**阶段 1 的保险**(若 W2 全输,这是唯一剩下的杠杆,且前置最长)+ 阶段 2 的长杆。含 SMASH 式最近邻检索 | 无(现在开工) | claude/franco |
| 1h | 抛光 | 权重细扫 2×2、f18/f19 备选锚(**只在 E/F 出大信号才跑**;结构定版前不扫) | 1e 定版 | claude |
| 1i | 顺手项 | 波次边界小活:torch 反解器 `coarse_landing` 的过网高度(net_z)没人消费——补同一行网 gate(numpy 侧已做,实测拦下 249/256 挂网答案) | 波次间隙 | claude |
| 1j | 开发 | **checkpoint 抽查流水线(筛选级)**:看门狗盯新存档 → GPU 空隙原生导出(~4 分钟)→ bank 考卷(CPU 池,~25 分钟)→ 结果追加进该 run 的考卷曲线文件 | 1a-0(bank 题源) | claude |
| 1n | 开发 | **等球锚训练端实现**(W3 动作组波的前置):`hold_ref_mode` 旗标——`clip0`(新标准:训练对首帧+零速度)/ `stand`(jiayi 谱系钉住,过门禁配方不动)。半天;C++/评估器侧同语义走 1m 的代际元数据 | 无 | claude |
| 1m | 开发 | **评估器代际旗标(07-07 回归倒查产物)**:部署仿真协议加 `--df-hold-ref stand|clip`(默认 stand=部署真值);导出器往 ONNX 元数据烤训练时等球语义,评估器 auto 按元数据选;判卷铁律补"对照必须同代际同语义"。背景:等球参考无旗标换代让老模型在部署仿真里全摔(三格定罪见 TIMELINE 07-07) | 无 | claude |
| 1l | 开发 | **⚡2real 提级(franco 07-09:"2real 要做的事都提优先级、开始工作";背景=jiayi 已于 ~07-07 在真机跑起老模型,还打不到球)** **Gate 3B 上台考(入账级,jiayi 提议以他的 Gate 3 为底座)**:①发球生成器——把现有"逐落点手解 vy"泛化成按分布采样 N 发(两种模式:阶段考卷模式=打到本阶段锚点/框+速度档;场馆综合模式=真实分布),解算对齐假球物理后输出 serves 列表;②判分器——击球瞬间取仿真里拍面位姿/速度真值 + 假球状态,走场馆接触模型+落点推演,报**击球率/上台率(分母=发球数)**,原有稳定性判据(接战/挥完/不摔)保留为并列列;③conductor 汇总行扩列。运行环境:jiayi 本地 distrobox,**或 pod 原生**(pod=Ubuntu 24.04,Jazzy 可直装,厂商仿真可直编——07-07 查实,不需要容器)。阶段 3 前假球要补旋转支持 | Gate 3 现成 + 1a-0 的判分函数复用 | jiayi(底座/接线)+ claude(采样器/判分器) |

**球物理 in-loop 验证收口(2026-07-06,yikang;Isaac 真球接入的物理前提)**:PhysX@训练 dt(5ms)
注入 venue 气动力 vs 解析 RK4——bank 档(场馆速度)落点差 **17mm 全通**;全包络(1-7 m/s、旋 0-95)
p90 30mm 超 20mm 线;**dt 减半误差精确减半**(30.1→15.0mm 全线 ×0.5,全包络 PASS)⇒ 差异=纯半隐式
欧拉一阶系统项,无建模/坐标错,旋钮=球物理 dt。判定:**S1(场馆速度、无旋)在训练 dt 下真球可用**;
S3/快球要么球子步减 dt、要么接受 <30mm(仍远低于 0.10m 考卷线)。工具:`scripts/isaac_ball_inloop_check.py`
(bank/synthetic 双模,日志 /workspace/yikang/inloop_{bank,syn,syn25}.log)。**Phase A 已落地并 mech 双向全绿**(stage1-fixed-point
9ed33b4):`++task.physical_ball=true` = 每 env 真球+真球桌,题目驱动反向积分发球,pod 实测
**发球精度 12.2mm / 速度误差 0.018 m/s**(正是 in-loop 预测量级);flag off 零痕迹。真球现为
纯真值仪表(不动奖励/观测);Phase B(拍面拟合冲量入引擎+CCD)排期时带上球子步决定。
附带物理发现:二次阻力反向积分在 ~1.3s 有限时间奇点(速度帽 40 m/s 规避,发球窗 0.6s,余量 >2x)。
**07-06 晚缺陷-修复闭环**:扩展 mech(seed=1)暴露发球误差 0.58m——根因=反向积分穿台面(该段
现实中在弹跳之前);修复 673cb53 = 截断在台面处只发"最后弹道段"(bank 触球高度下 tts_eff
~0.15-0.26s,截断是常态)。复测同配置:**发球误差 3.7-3.9mm**(比修前首跑 12mm 还好——段短漂移小),
迭代耗时降 44%。教训:mech 单 seed 全绿≠安全,扩展 mech(换 seed/加倍 envs)才暴露分布尾部。

### 阶段 2「虚拟球·变到达态」(franco 2026-07-08 重排拍板:原 2a 拍点框 + 2b 站位应变合并)

**为什么合并**:真球从对面发出后,"拍点变化"和"站位应变"都是来球分布的自然结果,不值得
各造一个一次性出题器——正向蒙特卡洛发球生成天生产出变到达态,一步取代 2a-1(point_mode
未写)与 2b-1(位移卷)。**franco 硬约束:阶段 2 必须对应动作接口**——出题器只出"老师动作
经适配器(**整身旋转 φ + 拍面 morph**)后接口仍能接"的题:旋转角/拍面角都要落在可行域内
(可行域判定与判炸器/腕限位审计同构,超域题不发货)。

| 顺序 | 类型 | 内容 | 依赖 | 谁 |
| --- | --- | --- | --- | --- |
| 2-1 | 开发 | **变到达态出题器**:正向蒙特卡洛(一跳回球族)按分布采样出手 → 到达态直接当题(取代 point_mode 框与位移卷);**接口守卫**:每题先验适配器可行性(φ/拍面角在域内)再入库 | 发球段重挖 ball_mocap_0703 当种子(或先用规则分布起步) | yikang/claude |
| 2-2 | 开发 | 适配器 1g 进训练循环:整身旋转轴(rally_yaw 机制,已验证)+ 变速(R14 机制,已实现)+ 拍面 morph v2/变体库(待实装)+ **击球高度轴(franco 07-08 补:接口还要能改来球/到达高度——后话,设计时留轴)**——老师侧接口,2-1 守卫的另一半 | 1g 骨架 | claude/franco |
| 2-3 | 测试 | 三臂同批:手补(站位钉死只伸手)/ 身补(站位随球平移;**复用 jiayi Hitter 站位驱动**,不重造;R12 减速塑形并入)/ 转补(整套动作绕锚旋转)+ 对照 = 4 臂 | 2-1(+2-2) | claude |
| 2-3b | 测试 | **移挥时序消融(franco 07-09)**:先移后挥(时间律两段+settle 边界)/ 边移边挥(重叠)/ 无结构(RL 自定)三臂——与 2-3 正交拼同波;"串行解排不下"=接口守卫高难判据 | 2-1+时间律工具(已有) | claude |
| 2-4 | 产出 | ~~分界表~~ **撤销(franco 07-09 批第一性原理清单)**:手/步/转的分工应从训好的策略行为里**统计涌现**,部署时提取,不人工设计 | 2-3 | claude |

### 阶段 3「真球进场」(franco 2026-07-08 重排拍板:球从对面发出;旋转入本阶段后段,是否独立成 4 待拍板)

**切真球预注册判据**:训练内 rally 上台率双侧 ≥50% + 变到达态考卷 ≥50% + 虚实落点 gap
<0.1m + **Gate 3B 先于臂发射建成**。真球纯仪表臂(physical_ball 旗标零代码)可即刻搭任意
臂的车,顺手补 Phase B 欠的富击球对账(待 franco 点头)。

| 顺序 | 类型 | 内容 | 依赖 | 谁 |
| --- | --- | --- | --- | --- |
| 3-0 | 开发 | **物理发球时间线**(正向蒙特卡洛两族:两跳发球+一跳回球;预警时间由球的物理自然给出)+ Isaac 真球从纯仪表转训练介质(Phase A 发球 3.9mm / Phase B 拍面冲量已收口) | 阶段 2 过线;发球种子重挖 | yikang/claude |
| 3-1 | 测试 | 真球臂:站位/拍点变化=来球分布自然结果,不再人工出题 | 3-0 | claude |
| 3-2 | 测试 | 旋转两档 × 消旋奖励有无 + 对照 = 5 臂(反解自动给随旋变化的拍面/切向目标;消旋奖励从此才有真效用;快球才需球子步,机制已可配) | 3-1(考卷加旋开关已有) | claude |
| 3-3 | 判读 | 重上旋"随旋收面"专项(阶段 0 遗留:开面拍接不住重上旋是物理自洽的,正是指令通道要教的) | 3-2 | claude |

### 等待与预警设计(franco 2026-07-06 立项,07-06 深夜按 franco 批注修订:粗解先行+等球姿态)

**规划器时延定案(franco 07-07 晚收口:延迟毫秒级后,"粗/精两档"作为计算概念作废)**:
实测已产品化——**整 tick(预测+反解)2.0-2.3ms**;一条全程精细正向推演(桌子一头到另一头,
0.7-1.5s 飞行,1ms 步长 RK4 = 700-1500 步,向量化)约 **1-2ms**。所以不再分粗解/精解:
**每个 tick 全精度重算一遍**(50Hz 策略 tick 20ms,只用掉 2-3ms,余量 ~10 倍)。
"早期计划粗、后期准"仍然成立,但只剩**信息**原因(球刚出手时估计本身不准、弹跳前不确定性大)
——计划每 tick 刷新,精度随球飞行自动变好,不需要任何双档机制。

**真机时间线(修订版;待 ③ 实测校正)**:对手触球 → 动捕锁球+速度拟合(31 帧@300Hz ≈
0.10s)+ 传输(≤0.02s)+ **粗解(毫秒)→ 触球后 ~0.12-0.15s 就有第一版(粗)目标**;
随球精化,弹跳后(约飞行 60% 处)收敛到精目标。场馆球全程飞行:快球 ~0.7-1.0s、中速
~1.2s、挑高 1.5s+。**推论:粗目标预警 ≈ 飞行时间 − 0.15s(快球 0.55-0.85s);精目标预警
≈ 快球 0.25-0.4s / 中速 0.4-0.7s。** 训练现状的预警=整个起手时长:hopex 对 1.31/0.88s
(正/反),v5 对 0.75/0.42s。

**设计(四件套,全部旗标默认关;和别的一样"慢慢扩大采样分布")**:

| 件 | 内容(人话) | 课程(分布扩张表) | 状态 |
| --- | --- | --- | --- |
| ①② 合并升级 → **物理发球时间线(franco 07-07 拍板;07-07 晚二修:正向蒙特卡洛,不做任何反解/求逆)** | 出题与发球一体化管线:**采样出手分布 → 批量正向模拟 → 到达态直接当题**。两族轨迹都要:**发球族(两跳:己方台一跳+对方台一跳)**与**回球族(一跳)**。到达锚点容差内的入库(题库本来就是样本集,不需要每球钉死一个点),应有拍状态用现有 torch 求解器按**实际到达态**解;**引擎里存出手态、按同一出手正向重放**——全程零反向积分、零弹跳求逆,上升段接触题(11%)顺带覆盖。预警时间由球的物理自然给出(快球=准备短);观测揭示=出手时刻+锁定时延(③实测)。虚拟球任务保留旧机制当近似 | 出手点课程:弹跳后 → 过网后 → **对面台侧全程(含两跳发球)**;生成批可直接用引擎跑(零生成器-引擎失配) | **数据更正(franco 07-07):发球其实采到过**——每回合第一球大多是发球,原始录里就有,只是拟合时没单列(样本少)。任务=从 ball_mocap_0703 原始录**重挖发球段**当出手分布种子,正向蒙特卡洛在其周围泛化(反正要接的就是各种球路);下次场地日再补量 |
| ③ 预警分布实测 | 从 ball_mocap_0703 + 规划器日志量真实"触球→粗目标/精目标"分布,替换上面的估算,数字直接进 ② 的课程档 | — | CPU 活,yikang+claude |
| ④ 等球姿态(franco 批注:目标不可见时,手要放在合理的等球位置、**随时准备快速击球**) | 两条腿:**演习驱动为主**——短预警题本身在教"什么等球姿态反应最快";**姿态锚为辅**——等球段贴 v5 首帧 ready 锚(小权重防漂移)。**终态定位(franco 07-07 晚二修):等球参考 = 该臂训练用的那套动作对的首帧 + 零速度**——不是统一 v5 首帧,更不是裸站姿(按哪套训,就用哪套的首帧)。史实澄清:半蹲瞬态摔因是 **hopex 对特有**(其首帧=半蹲中间帧);v5 对首帧本来就是准备姿态,冻结它+零速度即是正确等球锚——**毒在首帧速度(幽灵下蹲),不在"用首帧"本身**。下半身影响等球的通道=观测(command 62 维=全身关节参考),奖励侧模仿早已只管上半身。深层修法=所有模板以准备姿态开头(v5 已如此;进拍摄清单);hopex 谱系与 jiayi 站姿等球谱系为既成例外,评估器按代际元数据配对(1m),迁移周会对齐。三端(训练/C++/评估器)协同换,只换参考内容不动观测格式 | 姿态锚小权重起步 → 三端协同换"训练对首帧+零速度" | 设计好,随物理发球一起做 |

**归属与顺序(按排序法则)**:①②④改任务分布、与奖励结构交互 → **首臂挂 W3 巩固波搭车**
(起步档,先证明"开了不掉分"),阶段 2 起并入所有臂当默认结构(像分阶段出题一样),加固期
扩到真实下限档。**不进奖励结构波**(那一波只测奖励结构,一次只动一个自变量)。

### 跨阶段(不占主线判据)

| 类型 | 内容 | 时机 | 谁 |
| --- | --- | --- | --- |
| 加固包 | 相位依赖任务噪声(A1 叠成 v3)+ A1v2 感知毛病 + 换招低剂量 + **预警窗/等待档扩到真实下限(上面②③的末档)**——**整包**测部署轴(摔倒/存活),预期不涨分 | 回球率>0 的谱系出现后、上真机前 | claude |
| 部署债 | L1 平价校验 + L2 部署仿真指标 + MDU 打包链;S1 谱系 ONNX 过门禁 | W3 出成熟存档后 | claude 备 / yikang 运 |
| 采集 | 拍摄清单:拍面可测(拍上贴标记/棋盘格)+ "拍面朝前的正手"补拍 | 下次场地日 | franco |
| 物理 | 若真机回球系统性偏长/短:第一嫌疑 = 接触模型切向拟合 a_t(阶段 0 遗留) | 真机数据后 | yikang |

## jiayi 参数正名臂定义(合并审计产物,2026-07-06;方向对、参数待正名——每臂同批对照)

只是**臂的定义表**,排程一律走下面的统一队列。franco 2026-07-06:他的很多超参数可能是
顺手调的,**全部要数据支撑**——每臂同批对照;他现在本地训练,这些臂在他本地跑,结论走周会对表。

| 臂 | 内容(人话) | 现值 vs 基线 |
| --- | --- | --- |
| J1 | 熵系数:训练"乱试劲头"大一半有没有用(他已自退 0.01,降为记录性验证) | 0.015 vs 0.01 |
| J2 | 脚部姿态惩罚剂量:不许拧脚的代价 | -0.3 vs 0(他自己注释:窗口 [-0.5,-0.1]) |
| ~~J3~~ | ~~q_des 限位剪切采纳~~ **已裁决(07-06):默认全开**——jiayi 发现不剪切的产品线在 MuJoCo 里站不起来,这是训练=部署的正确性对齐,不是可调项 | 已定,不再消融 |
| J4 | 击球奖励权重包:位置/速度/拍面 = 14/14/5 vs 基线 8/6/3(拍面等效 ×10,和 E 臂互为参照) | 14/14/5 vs 8/6/3 |
| J5 | PD 增益域随机 ±15%(原 null → on,未消融) | on vs off |
| J6 | 等球就位门:等球奖励按"站位到位"发,不按"拍子够得着"发(防手臂钻空子) | station vs racket |
| J7 | 顺手调三件套(07-06 他随手改的):回合长度 16→10 秒、等球范围 [1-8 秒]→[1-4 秒]、站位奖励权重 2.0→1.5 | 新值 vs 前值(未消融,要数据) |

## 统一队列(唯一排程账本;franco 2026-07-06 定死:**不给卡/pod 分角色,只有一个队列**)

**调度规则(就这三条)**:
1. **任何空槽拉队列里最靠前的"就绪"项**(就绪=依赖列全满足);卡和 pod 无角色之分,
   全池 6 卡动态分(消融波 3-4 任务/卡 = 18-24 槽纸面容量)。
2. 任务自带属性,不是卡带属性:`独占`(跑到底类,单卡不共跑)/ `共跑`(消融臂,≤2/卡)/
   `CPU`(判卷/出题,不占槽);错峰 ≥60s、发射过 runbook 十条照旧。
   **CPU 任务也进池平衡(franco 07-09)**:两台 pod 各 128 核都在池里(mjeval venv/资产/题库
   两侧齐备),重 CPU 批(oracle 扫描/题库生成/补考)发射前看两侧 loadavg 挑空的;判卷
   (AUTOJUDGE)天然跟 run 所在 pod;池任务照旧 OMP_NUM_THREADS=1。
3. ~~两台 pod 共享同一个长期空间(/workspace 网络卷,数据自动同步)~~ **07-09 实况更正:
   两台 pod 各自独立本地盘,没有共享卷**——新臂的资产/题库/热启存档要先 rsync 到目标 pod
   (pod1→pod2 全量环境已于 07-09 搬运,后续增量用 /workspace/franco/pod2_sync.sh 模式);
   pod2 端点 `ssh root@162.43.172.181 -p 13146`(旧 pod2 74.2.96.x 判死=宿主机 GPU 卡死,
   见 runbook Known Quirks"no-PID 满载态";新 pod 点火前必查该项)。

**新认领(2026-07-06 深夜,yikang;先认领再动工)**:

| 项 | owner/分支 | 属性 | 要点 | 依赖 |
|---|---|---|---|---|
| **physical_ball Phase B:拍面冲量入引擎** | yikang / stage1-fixed-point | CPU 开发 + 共跑 mech | 拍面拟合冲量入 Isaac(**复用 spin_contact 现有实现,不写第四份**)+ per-pair 碰撞过滤 + CCD。**球子步决定当场闭环**:S1 场馆档 in-loop 17mm 已过线 → **不需要子步**;机制做成可配、默认关(S3 快球再开)——franco"排期时带子步决定"项就此关闭 | Phase A(已落地 9ed33b4/673cb53/274fa69)。**状态:已收口**——5d6d236 + 对抗验证轮(15 代理,10 报 9 确认 1 驳回)+ 九项修复 7e3be78(major:子步 FK 重绑一帧鲜度泄入奖励流→FK 纯函数化 _racket_fk,冲量扫描零属性写;另 8 minor:reset 步先算后清、撞点插值吸附、免同步热路径、presence 守卫、中挥重抽响亮守卫、回程弹跳分列计数)。**pod mech 全绿(GPU1,512envs×40iters×5 发)**:A≡A2 决定论成立;**A≡B 机器人流逐位一致(major 修复引擎内验证)**;A≡D Phase A 保存;pb 差异方向=修复本意(land_count 131 vs 118,reset 步事件不再漏计);serve_err 4.6mm;B 发随机策略即触发 2 hit/1 return(冲量路径引擎内实弹)。诚实边界:hit 通道仅 2 次实弹,富击球验证(pb_virt_phys_gap 大样本)搭下一个 warm-start 臂的车。**新增待办(franco 07-07 物理发球设计)**:弹跳感知发球(反向积分过弹跳)——升级为必做:①物理发球时间线要从对面台侧出手;②顺手解锁 11% 上升段接触题 |
| **MuJoCo 真球+真桌接线** | yikang / 新分支 mujoco-ball-wiring | CPU 开发;vendor 编译=交接件 | 把"编好没插电"的 C++ 球内核(venue 参数,已 4e-10mm 交叉验证)插上电:MJCF 球 mocap body + 桌/网 geom、**放置约定选边(vendor 竞技场系 vs 训练 env 系)+ drift-guard**、SimLoop 接线、发球注入/落点发布。vendor sim 编译实测需 jiayi distrobox(pod 封死)→ 产出做成**交接件**给 jiayi 本地跑 | C++ 内核(main 已有)+ 放置约定拍板。**状态:已交付 4607410(分支 mujoco-ball-wiring):训练系为准+单一平移到 vendor 系、drift-guard 七方一致、E2E 落点 vs 镜像 oracle 5.5e-12m、mujoco 3.10 smoke 过;交接件 docs/handoffs/mujoco_ball_wiring_jiayi.md,vendor 编译/QoS/GUI 只有 jiayi 能验** |


| R11 | `task.motion.clip_switch_prob=0.002` | **已停 2026-07-05(平台 0.762@10.7k,命中率税 0.12)**。剂量 0.002=每挥约 28% 被打断,远高于真机频率;回合长度 463≈469 说明没有多摔,掉的纯是命中率。~~真收益(抗切换摔倒)的量尺缺失~~ **量尺已建并量完(2026-07-05,`--switch-stress`):纯换招扰动压不出收益**——P2(没练过换招)在两种 PD 口径(Isaac 对照 implicit + 门禁 explicit clipped-PD)、24000 步 230 次换招(127 次在挥拍中段)下 **0 摔、换招后 2 秒存活率 100%、换招后命中率 0.97-1.00**;R11 同样 0 摔,唯一可见差别是它在门禁口径下命中率还略低(0.98-0.99 vs P2 的 0.99-1.00,税的延续)。**判读:MuJoCo 里换招离散跳变本身不构成摔倒威胁,switch 训练在这把尺子上零收益、税为真 → R11@0.002 维持拒绝**。尺子的适用边界:场馆真机摔倒可能是换招×感知毛病×延迟的交互(纯换招在干净仿真里不复现);若还要追这个方向,下一步是压力协议叠 A1 校准噪声或"击球窗口内定点换招",而不是继续扫剂量 |
| R11b | `task.motion.clip_switch_prob=0.0005`(剂量/4,**已点火 2026-07-05,o3_R11b_switch5e4**) | franco:继续调参找好参数。先扫剂量;若 0.0005 仍税重,下一轮动奖励侧(打断后 hold 补偿)。**⚠ 判读标准更新(2026-07-05 换招压力测试结果)**:抗摔轴上 P2 本来就满分(0 摔),R11b 在该轴不可能赢——它的存在意义只剩"更低剂量是否免税"(Isaac composite 对齐 P2)。**4000 步跑完,压力测试终值已量(exported_it4000_normbaked,2026-07-05)**:230 次换招 0 摔、换招后 2 秒存活 100%、两种 PD 口径命中率干净/换招后均 1.00(@2700 临时值相同;分布内不伤命中,好于 R11@0.002 的 0.98)。**信号档跑完(→5499,2026-07-05 判读):composite 峰值 0.839@5247,同批最低(R12 0.888 / R14 0.871 / R16 0.870)——低剂量税没免掉(~0.03-0.05),抗摔轴又已证明买不到东西(压力测试 0 摔)+ eval-B 0%/CF 98% 与全批一致 → 建议 clip_switch 方向整体关闭**(除非部署轴另有理由,franco 拍板) |
| R12 | `task.rewards.base_decel_weight=1.0`(o3_R12_basedecel,02:45 点火) | 减速塑形方向有没有信号(v1 P 律;v2 拟合包络等 6 套采集)。**信号档跑完(→5499):composite 峰值 0.888@4654,同批最高、与产品线持平——composite 无税**;eval-B(峰值存档 it4700):回球 0%/CF 100%/拍面 43.3°(结构现状不变,如预期——它本来就不是治拍面的)。终判看部署轴(减速入位行为、base_speed_xy_prestrike),candidate 进保险包 |
| R13 | 赢家叠加进产品线 | 产品候选 |
| R14 | `"task.motion.speed_scale_range=[0.8,1.2]"`(o3_R14_retiming) | **变速重定时**:同一 clip 变速播放+速度需求同步缩放,策略能否学会按需调节挥拍速度幅值 = 无新数据的连续强度 v0(franco"加减速改幅度";空间幅度另由 R6 裁剪臂回答,两臂合看)。**信号档跑完(→5499):峰值 0.871@4843,比产品线低 ~0.02——变速出题更难,小幅回落属预期**;eval-B(it4800):回球 0%/CF 96%/拍面 48.5°(比其他臂散 ~5°,变速让拍面更飘——判读时的一个减分项) |
| R15 | v5 clip 臂:`motion_file=/workspace/shared/motions/hope_forehand_v5.npz motion_file_2=.../hope_backhand_v5.npz` + 下方 v5 专属采样框(相位 [0.673,0.362],反手 2026-07-05 人工复核 +1 帧)(o3_R15_v5clips) | 新动作源(短条、首尾贴 ready);⚠与 hopex 差三个因子(数据源+clip 长度+参考噪声),与 R6 对照拆长度因子。**⚠ 发射异常(2026-07-05 06:57):看门狗把它发到了 DeployParity 任务、从零 2000 步(峰值 0.355,从零短跑正常水平)——与同批 vb 臂不可比,run 在 `agibot_a3_hope_deploy_parity/…o3_R15_v5clips`。按计划判读需在 vb 线重发(等 franco 定)** |
| R16 | `task.rewards.free_wrist_ori_mimic=true`(**已点火 2026-07-05,o3_R16_freewrist,franco 最高优先**) | **手腕解除姿态模仿**:把 right_wrist_yaw_Link 从 motion_body_ori/ang_vel 列表拿掉(位置/线速度模仿保留=挥拍路径照学)。理由:视频管线的手腕朝向不可靠(GVHMR),模仿它给拍面质量封顶。**在 vb 产品线上跑才有真学习信号**(落点/消旋奖励直接塑形拍面;纯 DeployParity 上只是去噪)。契约日它是法线指令通道的执行机构。**信号档跑完(→5499)+ eval-B 终判(2026-07-05):峰值 composite 0.870;真球考试拍面-vs-应有误差 43.5°(基线 P2 42.5°,纹丝不动)、回球率 0%、反事实 98% —— 手腕放开单独掰不动拍面**:落点/消旋奖励的间接梯度在 5.5k 步内没把拍面拉向可回球朝向。判定:**单独不进配方;它的正确位置是契约日的执行机构**(有了法线指令+奖励参照改跟指令,拍面才有直接梯度可爬) |

R15 v5 专属采样框(**franco 纠错后 2026-07-04 深夜版**;从击球帧提取,pos ±0.10 / vel(clean) ±0.50):
`"task.racket.strike_phase_per_clip=[0.673,0.362]"`(正手 0.768→0.673 franco;反手 0.345→0.362
claude 目视复核 2026-07-05,球 f014 贴拍,旧值早 1-2 帧;反手框已按帧 21 重算)
`"task.racket.pos_range_per_clip.forehand.x=[0.29,0.49]" ...y=[-0.63,-0.43] ...z=[0.74,0.94]`
`"task.racket.pos_range_per_clip.backhand.x=[0.68,0.88]" ...y=[0.13,0.33] ...z=[0.85,1.05]`
`"task.racket.vel_range_per_clip.forehand.x=[0.74,1.74]" ...y=[0.71,1.71] ...z=[1.20,2.20]`
`"task.racket.vel_range_per_clip.backhand.x=[2.30,3.30]" ...y=[0.06,1.06] ...z=[1.69,2.69]`

~~⚠ v5 正手 +Y 红旗~~ **已解除(franco 定位对了 2026-07-04 深夜)**:所谓 +Y 主导是**击球
相位钉错**——检测器选了拍速峰 0.768,那是触球后的甩鞭段;逐帧核查后真实触球 ≈ 0.67-0.69
(franco 给的"约 2/3 处"),该处速度 (+1.24,+1.21,+1.70)、法线 (+0.47,+0.84,−0.27),
方向健康。教训已**机制化(2026-07-05)**:`cfg/strike_annotations.yaml` 人工触球登记表 =
strike_phase 唯一可信源,analyze_strike_phase 注释优先、拍速峰降级为诊断候选并打不一致警告。
6/6 clip 已判定:v5 反手 0.345→**0.362**(claude 目视,球 f014 贴拍)、斜录正手 0.368→**0.432**
(触球前加速段,早 ~120ms)、斜录反手 0.495 确认;**hopex [0.47,0.333] 是速度峰约定值——
源视频(raw_video_hopex/ 已入库)为无球空挥,不存在触球真值**(正/反手约定帧在过击球平面后
18/33 cm,与真触球 clip 的"平面上升沿"语义不一致,列为 R15 动作源决策输入,不单独"修")。
视频源(GVHMR)clip 的拍面法线一律为手腕 +Y 代理(含 hopex),登记表已标 face_normal_reliable。
残留的真问题:**v5 参考噪声大**——mean|关节加速度| 5.9/15.5 rad/s²(正/反手),是 hopex
(2.5/2.7)的 2-6 倍,斜录(3.5/3.5)介于其间。短条+快挥拍给 GVHMR 的平滑上下文更少。
这是 R16(手腕解除姿态模仿)与参考滤波的直接论据;R15 判读时把"噪声"当第三个混杂因子。

| # | 项(owner) | 依赖(什么好了才能排) | 属性 | 状态 |
| --- | --- | --- | --- | --- |
| 跑着 | **s1w4 现况(07-09 白天刷新)——在训 12 臂**:M2(动作轴仅存,~15:05 UTC 收卷)+ 奖励轴 v4rg 底七臂 R1b/R2b/**R3c**(R3b 奖金窗 bug 修复重发)/R4b/R5b/R6b/R8b + R9a(删缰绳对照)+ v5 救回三档梯队 R9c(v5hLs)/R9d(v5hLt)/R9e(v5syn)。**已收卷/停臂**:M1(提前,病理对照判决定型)/ M3(judge.sh 已出卷:正手 1.000/4-5°,反手列锚病作废等 M3b)/ M4(收卷即 **AUTOJUDGE 自动出卷**)/ M5(噪声尺到手)/ R7(答案已出)/ R9b(判死)/ R3b(bug 臂)。**M3b(swing 反手换锚 0.495)队首等 M2 槽**;patrol_watchdog 常驻=掉臂复活+单一队列出队+收卷自动判卷。详账见 TIMELINE 07-09(白天)与 pod queue.md。——以下为点火期历史台账:**s1w4 消融第一波·动作轴五臂在训**(franco 07-08 深夜授权直接点火;机制检查全过、错峰 75s、热启 s1w3_main model_13000、physical_ball 仪表全开、4096×4000→17000):M1 v5rg 基准(gpu0)/ M2 v4rg(gpu1)/ M3 swing(gpu2)/ M4 混对 v5rg正手+v4rg反手(gpu0)/ M5 基准换种子(gpu1)。**奖励轴换底重发(07-09 晨,claude 调度决定)**:R1-R6 从 v5rg 底换到 **v4rg 底**(run 名 R{n}b_*_v4rg,对照=M2)——首班巡检+M4 查证发现 v5rg 底座反手完成率全线 0.0000(反手列不可读;v5rg 病根=深蹲过深+挥拍激进,warm-up 只治了首帧),而 M2(v4rg)反手=完美考场:击中 0.977、位置全过、**拍面误差 27.5°/速度 25% 不过=修C 考卷定位的短板本尊**,正是这批旗标要解的题;R7(腿遮蔽)留 v5rg(它测的就是病参考观测),R8 看守器同步改 v4rg 底。**优先级重排(franco 07-09:v5 救回+训练框架 > reward 塑形精修)**:R4b/R5b/R6b 已挂起让位(存档可续,恢复排看守器队尾);空出的 3 槽给 **R9a/R9b 下半身自由包**(anchor-z 缰绳删除+脚踝出包络+腿参考遮蔽;R9b 挂 v5hA=v5 救回的框架级测试)与 v5hL 臂。**v5 救回工程四线并行**:v5hA(尺度修,已成:骨盆 0.74→0.84,9% 身高错=遮挡下 GVHMR betas 估矮)/ v5hL(腿移植,在产:想象蹲腿换同人 v4 健康腿)/ v5hAs(触球窗外平滑+肩部毛刺诊断,在产)/ R9b(框架解耦)。病③定案:换尺度只降 11%(8.6→7.7),激进是动作/重建本身,集中右肩 11 rad/s。慢放判死降级为"带毛刺资产上判死,干净资产未测"。**保底牌设计升级(franco 07-09):不用匀速 0.8× 慢放,改"非均匀重定时"**——保触球帧速度逐位不动,重排时间轴把加速度剖面拟合得更均匀(数学上=对关节轨迹做时间重参数化、以 min峰值加速度 为目标、触球窗内时间刻度锁死);若 R9b/v5hLs 臂失败,以此产 v5hLt 候选重测,可再叠 R14 变速课程(慢→常速滚动)。**换底已完成(07-09 晨)**:R1b-R6b 六臂在训(布局 4/4/4,12 臂满池;R5b/R6b 引导罚在 v4rg 底重标定 W=-0.95/-2.0,实测占模仿收入 9.8%/21.0% 命中 10/20% 目标带——⚠与原 R5/R6 绝对权重不同、档位含义一致,读台账);R8b 看守器在岗。M4 混对查证=零错位,读数为真:v4rg 反手真好(完成 0.96/击中后上台 0.65),v5rg 正手被健康侧挤死(99.3% env 时间归反手)。原计划(八旗标合 main 9e08c38,39 测试绿,rewards/terminations 未知键已 fail-loud;八臂机制检查 8/8 过、逐臂核了旗标生效行;常引导按记账实测定档 R5=-0.4≈模仿收入 11%/R6=-0.75≈21%;R8=包络软化+RSI 站高,机制检查里回合长 13.5→313 步、净收入保持正)。三卡 4/4/4 拉满(22G/32.6G),R8 由 r8_slot_watcher 5 分钟一巡自动点火。臂表/PID/配置串=pod /workspace/franco/s1_wave4/queue.md;每 3h 自动巡检+掉卡看门狗已挂 | — | 3-4/卡新规 | 队列表=queue.md |
| **跑着·07-09 深夜批(claude 值守排产;franco §八 dependency 稳原则)** | **pod1**:掉卡孤儿四臂复活续跑(R9c 15500→/R9d 14400→/M1 16400→/M5 15100→17000,07-08 19:27 掉卡事件+watchdog 监护名单缺口所致,expected_arms 已补);GPU2 新发三臂=**R9g**(v5syn×标准配方,删缰绳单变量——R9a 训练内反手 0.73 vs 考卷接触 0.15/0.00 塌方,取证定性=删 anchor 缰绳后丢世界系 z 补偿,MuJoCo 全策略共性站高 +0.11m 下拍随躯干上浮 10cm 挥空)/ **M2 seed2** / **R1b seed2**(换种子验 ±0.01 噪声尺;R4b 种子臂撤销=其终档考卷塌方 0.071)。**pod2(162.43.172.181,07-09 上线)**:**R5b seed2**(常引导温和=考卷 ns.05 全侧最佳 0.800)/ **ST1**=M2+stagger 防同步旗标(指标修复验证对照)/ **G1 swingsyn** + **G2 v4rgsyn**(时间律合成泛化轴:最干净资产+最健康资产,判"合成时序更好还是无害";两资产判炸器 0 FAIL 0 WARN,G2 题库 92% 可解闭环 100% 落台)。**撤销两臂(工具物理不可达,理由入档)**:R9h(acc 3.9-4.1;synthesize 上确界 3.614——mean|acc|≈K/(T_out−1),减速段占片长 69% 无旋钮)、T2(acc 2.0-2.4;几何硬顶 2.63)——**副产品=时序轴到底的直接证据**:R9e(2.82)已是本族地板,地板上仍摔 0.20(10× v4rg)⇒ 残余是路径/接触病,与 oracle CoP 主 binding 定案互证。**oracle 对账台**:R9f 事前预测摔率 0.4-0.6,实测 ~0.20 @14.8k = **MISS**(排序仍对,阈值偏保守)。另发 **C1**=R1b×R5b 赢家组合臂(W4 交互确认,pod2 gpu1)。**TOPP v2 工具**(力矩余量×时间松弛,oracle 逐帧约束外环搜索)分支 `topp-budget-search-0709` 已推 origin 待审(对抗复核 3 缺陷已修,19 CPU 测试绿);**回测直接终结六档位臂**:v5hLs 上三档力矩预算全塌缩到唯一可行解 T_a=0.84s=ta_max(贴限=摊满),该解 oracle 打分与 v5syn 逐位一致(=R9e 已实测摔 0.21)——GPU 零消耗出裁决。~~时序轴收官:CoP 不可约地板 0.232 姿态绑定~~ **⚠ 07-09 午翻案:那只是匀加速单参族的地板**——Opus 平行分支 synthesize_timing_v2(非均匀时间律:逐路径点拉伸,oracle 在环,哪帧站不稳放慢哪帧)把 v5 反手 CoP 剂量压到 **0.0905 < v4rg 地板 0.167**(pod1 产线重跑逐位复现,f49e9db 已合;代价=反手时长 ×3.45)。**T3_v5topp_std 臂已排 pod2**(×标准配方,冒烟中——pod2 importer 挂死连中两发,杀组重试中;对照=R9g 同配方 v5syn=时间律 v1 vs v2 单变量;判读注意:bh clip 4.0s → 每集挥拍机会少,rally 分母小,预警语义 run-up 1.26s) | — | 双 pod | 值守=pod1 patrol_watchdog + pod2 patrol_watchdog_p2(revive-only) |
| **⚡2real 冲刺包(franco 07-09 提级:"2real 要做的事都提优先级、开始工作")** | 背景=jiayi ~07-07 真机跑通老模型(还打不到球,franco 转述待他本人补记录);他的 2real 修复在 `origin/hitter` 分支(审计中,正确性修复按 07-06 先例合入)。**清单(按性价比)**:①**Gate 3B 建成**(1l 行,最高优先;jiayi 底座+claude 采样器/判分器;pod 原生 Jazzy 直编为后备环境)②**考卷扩样本**:bank 考卷每格击数从 3-14 扩到 ≥50/侧(纯 CPU 加时长,判读方差立减——现在的考卷小样本是判读最大噪声源)③**富击球对账臂**:Isaac 拍面接触 Phase B(引擎内冲量)至今只有 2 次实弹击球对账;wave4 臂只开了 Phase A 发球仪表(已核:日志无 pb_hit/pb_virt_phys_gap)→ 下一臂带 Phase B 旗标点火(旗标名从 5d6d236/7e3be78 合并提交确认,先过机制检查)④**A3 执行器辨识**(需 franco 排真机时段——explicit PD/摩擦/延迟的最后一块)⑤L1 平价校验+L2 部署仿真指标+MDU 打包链(部署债,W3 成熟存档后)。**hitter 分支审计完毕(07-09,11 提交三档分类)**:第一档已合 main=真机 field 三件套 96040df(mocap 毫米→米 position_scale/规划器解算异常降级不死节点/policy_z_offset 0.76m——没它 engage 永不触发;参数化 sim 默认不变)+ 弹跳检测中心几何修复 850bfe0(真 mocap 跟球心,最低点 0.02m>旧阈值 0.005,旧检测器场馆数据上永不触发;99 planner 测试绿);第二档(SHADOW/MOTION 幽灵挥拍重置/恢复时钟/接管守卫等 planner 通用改动,每条有摔倒证据但改变已验证的 17400 行为)**待 g25/Gate3 回归后合**;第三档(训练回滚 entropy/hold_steps/Δ=0、默认模型 17400→15500、x_hit_follow_robot 默认翻转、闭环 harness 回滚)**与 main 审计结论冲突,周会对表**。⚠ main 现存活 bug:C++ obs 调试打印对 177 维模型索引越界(Eigen UB),修复在 hitter 分支 baf6215 但与 110 契约链耦合,随其合并或对表时单摘 | — | CPU+排真机 | 冲刺 |
| 台账更正(07-08 点火时核实) | ①**pod 实况=3 张 5090 不是 6 卡**(第二台 pod/D9 从未到位,算力池小节的"6 卡 12 槽"是纸面值,待 franco 定是否补第二台;**07-09 已补:新 pod2 上线,6 卡实况恢复**);②**wave3 七臂不是被停,是 07-07 两次全局掉卡事件团灭**(08:48 团灭只换题库/原样,10:26 团灭主攻/蒙眼/face3x/face10x/模仿降权;主攻止步 13089≈85%)——此 pod 掉卡有前科,wave4 巡检带自动续跑看门狗;③**修C 已收卷并判卷完毕(07-09 凌晨,入账级)**:反手谱系裁决=**修C 胜**(反手接触 1.00/回球 0.385-0.636/拍面 6.3°;正手 0.75/0.60-0.75)、**fixE 判死**(MuJoCo 上 469/469 秒摔、0 击球、learned_std 爆到 7.4——训练内 0.84 完全不折现);修C 反手下一个瓶颈=速度通道(vel_fail 11/13),正是 s1w4 奖励轴的题;详见 TIMELINE 07-09 | — | — | 实况为准 |
| 台账(07-09 指标病定案+根治,claude;merge 53d440e / fix b1e8867) | ①**指标病定案(取证)**:`virtual_return_rate_rally_*` 分子(`_vb_inb_acc`,击球帧入账、只在全场有击球的步衰减)与分母(`_swing_starts_acc`,拍首入账、每步衰减)**两本账衰减时刻表不同+入账相位差 ~116 步**;4096 env 同刻 resume+低摔率排成同步大队列(episode_length 锯齿 52→485、集体超时)时比值振荡 0.31→1.48 冲破 1。同病波及 swing_completion_rate(min(,1.0) 封帽长期激活)与 fall_rate 类 EMA 的同步态读数。**真值不受影响**(R5b/R6b 校正后 ≈0.72,与对照同水平)。②**根治(正确性修复,豁免消融)= per-swing 同刻入账**:击球帧只 latch"本拍已合法回球",拍尾(wrap/真 reset)起拍数+回球数**同刻入同一本账、同表衰减**→比值恒 ≤1 且=真实回球率(合成复现:旧口径同场景读 2.31→15,新口径全程=0.72 真值);全局+正反手分身同步换新;**旧口径原位保留 `_legacy` 后缀曲线一个过渡期**(开关 `task.racket.rally_legacy_metrics`,默认开;人话=旧算法对照曲线还要不要发),看板新旧对照满一个过渡期后关。③**防同步旗标 `task.motion.stagger_initial_clock`**(默认关=现役字节等价可比;**新点火臂建议开**;人话=把所有 env 的"到点超时+挥拍节拍"随机错开,EMA 曲线不再集体振荡):首个真 reset 给 hold 时钟加 U[0,150 步] 偏置+构造后首步给 episode 时钟加 U[0,回合长) 偏置,超时波从此永久错开——治所有 EMA 指标的同步读数病。④**判读纪律**:没开 stagger 的存量/同步臂,EMA 指标一律取 **≥21 个迭代周期均值**再判,单点不作数;**R4b/R5b/R6b 摔率读数按此重审**。13 个新单元测试全绿(test_metric_sync_fix.py) | — | 已合 main;pod 拉 main 后新点火臂自动吃到(不影响在跑臂) | 已入账 |
| 0 | 【判卷】**修C 收卷 MuJoCo 考卷 + fixE(yikang trim6 臂)补考对判**——反手谱系 fixE vs 修C 二选一的入账依据;五停臂顺手补考(判卷链 07-06 后欠账) | 修C 到线;mjeval CPU | CPU(导出占槽 4min) | **修C 到线即做** |
| 0.5 | 【资产】**swing 对正式入库**:触球帧人工登记(无球空挥则走 hopex 约定帧惯例)→ 会话握拍复核(试产暂借 Rz5Rx40)→ 题库 v2 生成 → 判炸器复审(试产已双 PASS);依赖=GMR 修复分支定版 | swing 试产已完成(regen_test_0708) | CPU | 动作组波门票 |
| 0.6 | 【制度】**判炸器接线**(分支 `motion-feasibility-audit` 已推 origin 待审):登记表加 feasibility_audit 块+首帧豁免旗(存量六 clip)、gen_stage1_questions fail-closed;**b9d0eec 更正:分支已在 origin(s1-registry-v4cal),真残留=v4_cal 登记 16 行合 main**,合了守卫就不再拦 | franco 审分支 | CPU | 防复发 |
| 0.7 | 【资产】**三套动作(swing/v4/v5)管线重跑=候选生成中**(franco 07-08 拍板"可以重跑";GMR 修复已合 master):六条候选 npz + 相位重标建议 + 判炸器全审 + 新旧对照,产物在 regen_0708_candidates/,**不动现役资产**;swap-in(含登记表 phase 人工复核、题库重出)等修C 收卷对判后执行 | 两分支已 merge(main 2187911 / GMR aabea2e;登记 16 行已 cherry-pick 1bcc083) | CPU 管线 | **进行中** |
| 1 | 【阶段1】第一波成对差初判(主攻/减观测/产品线锚已到线;**只看成对差,绝对分不作数**——题库缺陷见上) | 1a-0 bank 题源收尾(半成品在 `wip-bank-exam-source-0706`) | 导出占槽 4min+CPU | **今天** |
| 2 | 【阶段1】**六臂合体波(franco 07-07 定稿,07-08 两处更新:动作组成员改 swing/v4/v5、奖励轴扩员)**:基准臂 / 击球窗臂 / 击球窗加权臂 / 课程臂 / **swing 老师臂 / v4 老师臂**(斜录候补)。奖励轴另有 reward 提案臂族排队(V1 手腕速度剔除 / V2 窗内模仿让位 / 接近度门 / 常引导×2 / R-a 遮蔽腿参考 / R-b 终止软化 / R-c RSI 健康化——档位与取舍见提案,franco 拍板后编波)。各臂自家题库 v2 + 自家首帧等球锚 | 硬化植体已默认 + 题库 v2 + 奖励收入记账(1b)+ 击球窗代码(1c)+ 课程 loader(1d)+ 等球锚旗标(1n)+ swing 入库(0.5 行) | 共跑;6 卡下可每臂独占(一轮 ~2.5h) | **最高优先;门票=1b/1c/1d/1n 四件代码** |
| — | jiayi Hitter 主线长跑 + J1/J2/J4-J7 参数正名(jiayi) | 拉 main 即可 | **本地机器,不占云池** | 周会对表 |
| 3 | 【阶段1】W4 确认+巩固波:最优奖励 × 最优动作组的**交互确认臂**(若赢家不是对照配方/对照动作组)+ 赢家跑到底(独占)+ 换种子 + 等待/预警起步档搭车 | W3 判卷 | 混合 | 等 |
| 5 | 【阶段2a】拍点框 3 臂(claude) | 阶段1 过线 + 出题器 point_mode(yikang) | 共跑 | 等升段 |
| 6 | 【阶段2b】手补/身补/转补 4 臂(claude) | 阶段2a 过线 + 位移卷生成器 | 共跑 | 等升段 |
| 7 | 【阶段3】旋转 5 臂(claude) | 阶段2 过线(考卷加旋已有) | 共跑 | 等升段 |
| 8 | 【跨阶段】加固包 1-2 臂:感知毛病+相位噪声+预警末档 整包(claude) | 回球率>0 谱系 + 相位噪声代码 | 共跑 | 等 |
| 常驻 | checkpoint 抽查两级:每存档 bank 考卷(筛选)+ 判卷点 Gate 3B 全链路考(入账) | 1j 流水线;1l Gate 3B | 导出占槽 4min+CPU;3B 在 distrobox | 建好即常驻 |
| 已撤 | 权重细扫 / f18f19 锚 → 降级为 1h 抛光 | 结构定版才考虑 | — | 撤出队列 |

**尺子铁律(franco 2026-07-06)**:击球率/上台率**正式入账必须是 MuJoCo 版**;训练内
两球率只作过程监控,可并列报但必须标注(详见 runbook 判卷铁律)。

## 算力池(2026-07-09 修订:双 pod 实况上线;franco 拍板的第二台 3×5090 第三次落地成功)

- **池 = 2 pod × 3×5090 = 6 卡**(消融波 3-4 任务/卡),统一队列动态分;
  **无共享卷**(07-09 更正:各自本地盘,资产/存档 rsync 搬运)。jiayi 本地训练,云上 6 卡全归队列。
- **pod 台账**:pod1 = 162.43.172.171:18333(基座,watchdog 值守);
  **pod2 = 162.43.172.181:13146(2026-07-09 上线,体检干净 P8/0%)**;
  旧 pod2(74.2.96.48 / 74.2.96.37)**判死**——RunPod 5090 节点"无进程 99-100%/575W"
  宿主机级卡死,换端点复现,判 provider 侧;诊断与开机体检规程已入 runbook Known Quirks。
  旧 pod2 唯一派过的任务 R9f 已退回 pod1 重发,无其他遗留。
- **D9 现状**:①pod2 bring-up 已执行(pod1 全量 rsync:venv/IsaacLab/shared/bin/repo/银行/
  热启存档,~26G;`HOPE_URDF_IMPORTER_NO_UI=1` 随 env.sh 带过);②统一队列调度器仍是
  手动+watchdog 双轨(pod1 patrol_watchdog 自动复活+排队;pod2 暂手动发射,复用 fire 脚本模式)。
- 老规则不变:≤2 任务/卡、错峰 ≥60s(**每台 pod 各自错峰即可,跨 pod 无 Kit 锁冲突**)、
  独占属性的任务单卡、卡型钉死 5090-32GB。
- 扩缩容 = 池大小跟着队列走:就绪臂数持续 > 空槽+2 → 加卡;持续 < 4 卡占用 → 退一台 pod。
  全程纯算力 80-150 卡·时,6 卡下**墙钟 ≈ 2.5-3 天**,节拍器是判卷/拍板不是卡。
  钱:6 卡 × ~$1/h × 3 天 ≈ $400±。

## Codex 接手的近期主线(2026-07-10,franco 授权负责 NOW)

目标不是“把 V5 模仿得更像”,而是用可证伪实验回答:**专业人的路径、触球几何和
发力顺序里,哪些能迁移给 A3,哪些必须由机器人按自己的力矩、速度、行程和平衡重解。**
这条路走到底后再大力推 Phase;不用旧判分器、错初态或不同题表制造假结论。

### 迭代顺序

1. **M0 先修尺(正在收口)**:BankExam 同题 all-attempt 分母、同 `stand` 初态、一题一
   reset、schema-v3 ONNX/动作/执行合同与 SHA 全绑定;新模型未过合同不能出正式分。
2. **M1 离线先淘汰**:对原路径、引拍 +20%/+40%、随挥重写及组合只跑限位/自碰/
   CoP/摩擦/力矩/速度守卫。**延长行程只有在原路径受限、L 真增加、`a_min` 真下降、
   力矩余量真改善时才进 GPU**;否则不为“看起来更大”付训练费。
3. **M2 机制冒烟**:512 env × 25 iter,只查能否启动、梯度/奖励是否活、是否出生即摔;
   不拿短跑排冠军。
4. **M3 配对信号档**:4096 env × 2000 iter,同题串行消除,不稳定小样本保留;每个轴保留
   反面对照。
5. **M4 成熟档**:最多 2 个配方、至少 3 seeds。研究升段线=all-attempt 回球率 50%;
   真机候选线=80%+不摔+执行/安全门全过。两条线用途不同,不再争“唯一标准”。
6. **M5/Phase**:单题胜者才进连续球 Gate 3B;然后推 Phase 的位置、旋转、连续和任务驱动
   下半身,不让 V5 完美主义卡住 Phase,也不带病尺抢跑。

### 预注册消融轴(不跑无语义的全笛卡尔积)

| 轴 | 候选 | 回答什么 |
| --- | --- | --- |
| 老师 | task-only / V4 软先验 / V5 专业软先验 | 专业人是否提供任务约束外的可迁移信息 |
| 时间 | 真人原时序 / A3 重定时 | 可迁移的是路径/顺序,还是真人绝对节奏 |
| 行程 | 原路径 / 引拍 +20%/+40% / 随挥 / 组合 | 加速和制动距离是否真正的瓶颈 |
| 触球几何 | `site_colocated_v1` / `exact_face_contact_v2` | 旧的球心=拍心近似是否改变逐题排序 |
| 拍速口径 | 触球帧 × `+-1`/`+-2` 帧差分 | 80 ms 平均拍速是否在替代真正的瞬时接触速度 |
| 任务拍速 | 同路径/拍面下约 2.2 m/s / 3.4 m/s | V5 失败是专业路径不可迁移,还是当前 A3 被要求了过激拍速 |
| 公共外部卷 | V4/V5/task-only 同一来球/落点表 | 动作自生题库是否把教师质量与考题难度混在一起 |
| 来球谱 | 旧手工盒 / venue rebalanced 训练 / matchlike 考卷 | 结论能否覆盖真实场馆而不只是 2–5 m/s 人造题 |

拍心/接触口径见 [racket_contact_geometry.md](interfaces/racket_contact_geometry.md),实验纸面与加速器见
[v5_professional_transfer_audit_2026-07-10.md](research/v5_professional_transfer_audit_2026-07-10.md)。

### 当前硬边界

- 现有 A3 训练把 MuJoCo `frictionloss` 的 Nm 数字当成 Isaac/PhysX 无量纲、载荷相关的
  joint-friction coefficient;语义不等价。新 Formal BankExam 遇非零系数必须 fail-closed;
  直接数值 proxy 只能 diagnostic。旧 checkpoint 不能被洗白,下一批需做零摩擦/标定摩擦对照。
- `exact` 目前只表示“判卷执行协议完整绑定”,不表示 PhysX↔MuJoCo 跨引擎动力学完全
  等价;mass/inertia/COM、asset SHA、contact/solver 和 DR 分布尚未全部入合同。
- C++ 可以在用户态线性化模式/发送/停机,但无法抢占一个已经卡死的 backend `SendCommand`,
  也没有 controller ACK/timeout;真机门仍 `Partial`。
- 2026-07-11 franco 授权 Codex 接管两台 RunPod 的本主线实现、判卷与后续排程;
  但依赖顺序不变:先盘点 cc 现场并验收 schema-v3 canary,未过尺不盲恢复旧训练。

## Active

| Item | Priority | Owner | Branch | Status / next checkpoint |
| --- | --- | --- | --- | --- |
| **原生 MuJoCo 训练/微调后端 P0** | ★★★ | **Codex** | `codex/mujoco-training-preflight@6e5fce3` | **首票已实现并推送，待审合入**：完整 MJCF closure + 2 s parity trace 合同 + source/objective/warm-start/physics trust 边界，focused `63 passed`。静态请求准确保留 11 blockers；stored audit、caller trace、certificate 都不能授权，physics/objective/loader/isolated-producer 四个 trusted 位及 smoke/formal/long 三个授权位全为 false。binary contact/net/landing 只叫 diagnostic，不冒充 Isaac dense net-height/landing-error/spin reward。没有 VecEnv/PPO/Pod/长训。**下一票已由 Codex 接续**：trusted isolated runner + `vendor_gate3_v1` single-env core（禁 child-process escape、记 runtime module closure），再做 evaluator action-tape 与独立 same-shape reward oracle；D0 仅 balance/strike-state，physical return 仍等 `mujoco-ball-wiring@4607410` vendor 验收。门未过不 PPO。 |
| 全栈正确性尺+C++安全包+拍心/拍速合同收口 | ★★★ | **Codex** | `main` | 双 RunPod 源码验收已绿(portable/ROS C++、whole-body、planner);下一检查点=重出 fresh schema-v3 ONNX+修后考卷,旧判分器数字不入账 |
| V5 专业动作可迁移性+Phase 加速器 | ★★★ | **Codex** | `main` | manifest+保守 halving 已就绪;下一检查点=验证触球帧/拍速口径,把行程/时间律报告接成 feasibility producer,再做 BankExam→scorecard adapter;两者完成前不自动发训练 |
| 新动作库(Franco/v6/v7)+TOPP 最短可行时间+任意时刻下一拍恢复 | ★★★ | **Codex** | `codex/schema-v3-isaac-adapter@bf19fca` | 10 段完成 intake→canonical GMR/grounding→240Hz稠密安全屏；5,162样本地面/自碰/拍柄身体危险均0，最薄40.25mm。回球/phase/2-vs-4 因 frame/mirror 未证保持 null；正在内容寻址固定HOPE虚拟桌 counterfactual frame，过门才做64题、TOPP与RL。T1核心已实现但连续卷/自击/plant未齐，不点火、不真机。 |
| Phase-1 schema-v3 Isaac 同题 adapter + 候选重排 | ★★★ | **Codex** | `codex/schema-v3-isaac-adapter@bf19fca` | **24 live + 4 terminal，六卡每卡4条**。Pod1 SSH超时后恢复审计确认未中断；十个 hardened worker 正常。fresh SZ 2k(`.83`)>4k(`.50`)仍只是MuJoCo pair选择；forensic已定位Isaac虚拟判分/执行状态/拍面符号盲区，2×2 instrument门严格打开。M2/M3 18k/19k q10曲线已出且不裁决，fresh scale-out 2k正在自动判。训练/eval clean `6d93bcb`/`46a0ce2`；下一检查点=SZ seed3/4 2k曲线、Isaac physical truth、终档完整性。 |
| HitterPure RallyFinal clean-base task: x-lock/lunge, settle/slip, backhand clearance, front-facing constraints + Isaac/AGI rally gates | ★★★ | codex for dongc1 | `hitter` | PATCH COMPLETE / GATES PENDING 2026-07-10: clean-base Final task, native move-settle-arm readiness, strict metadata/eval/gate plumbing and docs implemented; host tests + x86 build pass. Next = Isaac smoke/train/ablation, Final ONNX MuJoCo scores, then no-rescue AGI closed-loop with physical contact/landing evidence |
| **加速度包络标定两件套(franco 07-09,时间律的下一层)**:①跟踪破裂标定(chirp/斜坡加压参考×现成跟踪策略,逐关节"边平衡边跟"真上限=判炸器 L1 升级);②贴限 vs 摊时消融(v5syn T_a 三档)——R9d 读数落地后一起排 | ★★★ | claude | — | 设计已入 research 时间律文档§六 |
| GMR 源头修复(pod GMR 分支 `hope-frame0-warmup`:warm-up/帧0/逐关节限位旗标)+ 判炸器(repo 分支 `motion-feasibility-audit` 已推 origin)——两分支待 franco 审;接线与 L6 重生成见队列 0.6/0.7 行 | ★★★ | claude | 两分支 | 已验证收口(TIMELINE 07-08);合入即防复发 |
| 1b 奖励收入记账 | ★★★ | claude | main | 半天;进机制检查+判卷固定输出 |
| 1c 击球窗分通道奖励(W2 门票) | ★★★ | claude | main(旗标默认关) | 半天-1 天;整包一臂 |
| 1g 适配器 v2 变体库骨架(CPU) | ★★ | claude/franco | — | S1 保险 + S2 长杆,现在开工 |
| 1d 难度课程 loader 窗口 | ★★ | yikang | `stage1-fixed-point` | **代码已落**(loader 难度开窗+train.py override 翻译层+推进速率修 bug,旗标默认关字节等价);yikang 07-10 转 vendor 链,后续臂搭车即可 |
| **head_discipline 奖励 + A3 stand 查看器**(公司 Linux 机改动移植,那台不让 push) | ★★ | **yikang** | `yikang-linux-port-0711` | 从公司机移回两件真改动(基 hitter@`5c346ea`=真实基线,不是 main):①**head_discipline 奖励**——head_yaw/head_pitch 是无奖励管辖 DOF(A3_UPPER_TRACKED 到腕即止、无 joint-space 项配 head_*、静态偏置在各正则项下付~0),njfc21an/model_9000 MuJoCo 诊断把头偏 park 在 -60° 软限(p5=p95=-60.6°)整轮=「机器人朝右看」病根(参考 271 帧恒 0);复用 foot_orientation_discipline(L1 \|q−ref\|)于头对,默认 0.0=**现役配方字节等价**,RallyFinalV2 白名单开 -0.5(同 07-07 pigeon-toe 同型病同型治)。②`scripts/view_a3_stand.py`——**Gate 3 厂商仿真线的本地诊断工具**:纯 MuJoCo 载**与 Gate 3 底座同一个厂商 `a3_pingpong` MJCF** + 部署 PD_STAND 增益(a3_policy_parameters.hpp)锁站姿(--check 10s 稳定门/骨盆 z 漂移 >0.15m=FAIL、--snapshot)。**Gate 3 侧做了的**=Mac 上**免 AimRT/iceoryx 编译**就能肉眼+量化看厂商站姿稳不稳;**没做/仍开的**=厂商 MJCF 无 integrator=显式 Euler(SIM_FIDELITY_NOTE 未落)、PD_STAND 塌陷本身**没修**——此工具只复现/诊断,行为级判读前仍须与 jiayi 对 MJCF 积分器配置(见下方 Gate 3 底座行知情项①)。③node.py 经核=origin/main 逐字节相同(公司机本地 main 落后 origin 的陈旧基线假象,非本人改)已剔除。已推;**07-12 满池实查→验证臂进统一队列(Queued 7)**,附 rebase 语义雷(V2Plus 派生键集)与 FinalV3 passive-head 同病两药定位。**✅ 07-12 Codex 审入裁决(TIMELINE「yikang 小改动选择性审入」)**:reference_oracle 按 main API 移植+三道门审入(contact/RK4/landing 对拍 max 4.63e-9);viewer 审入但**纠正增益口径**(生产 PD_STAND=29-DOF、neck 明确 passive,原脚本 head 40/2 不算生产增益;main 上已是 Codex 改写版=直接解析生产 pose/Kp/Kd+绑 SHA);**head_discipline 不合 main**(FinalV2 不在 main+V2Plus 派生键集+FinalV3 同病异治),采纳路径三步走见 Queued 7,现役 reward/config 回归 88 过、未发训练臂。本行使命完成,验证臂后续在 Queued 7 追踪 |
| D9 **双 Pod 终档/队列债** | ★★ | claude | — | 两台 pod 是独立磁盘,不共享 `/workspace`;pod2 十一臂有 16999 checkpoint 但旧导出/报告来自 16400。等 schema-v3 canary 绿后再补终档,不盲目恢复训练 |
| 等待混合采样 + 目标揭示预警窗 + 等球姿态锚(设计四件套的①②④,旗标默认关) | ★★ | claude | main | 约 2 天代码;首臂搭巩固波 |
| **拍面反解 torch 版上部署(franco 点名提上日程)**:训练内联的批量求解器搬进规划器,替换 0.3s LM;粗解先行+随球精化的节奏靠它 | ★★★ | claude(求解器)+ yikang(接线) | main(接线已合) | **07-12 yikang 接线侧影子骨架已落并合 main**(merge `7e31819`,合后 planner 零 rclpy 全集 **130 passed**;主线唯一动过 node.py 的 4a4ec76=发布点注释,零语义交叉;照 use_kalman 影子先例):`use_shadow_solver` 默认关=**字节等价**(node.py 纯增 +145/-0,False 短路于任何构造前);shadow_solver.py 零 rclpy——SolverBackend 接口+EchoBackend(重跑产品同款快解,对账管道自检 diff 恒 0)+TorchBackend fail-loud 占位;CSV 对账逐字段 diff+双路 wall-time;影子异常吞掉计数绝不影响发布命令,慢后端 wall_budget 0.25s×10 自禁;诊断逐键数值。**I/O 契约钉在模块 docstring**(ACCEPTANCE RULE=TOL_M 0.005/budget+1e-9、None 语义=后端自己执行验收、构造期 config 消费清单)。测试 12(含负控:两组真不同解 diff 必须>0.1m,防对账器恒零假绿)+邻居回归 25 全过;rclpy/colcon 级留 pod。对抗评审 8 发现(1 major)全核实修复。**等 claude:torch 求解器本体 + 拍板对账容差/warm-start 语义/omega 要不要接**(开问题清单在 shadow_solver.py docstring) |
| 预警分布实测(触球→粗/精目标) | ★★ | yikang+claude | — | CPU 活;数字回填预警窗课程档 |
| 1j checkpoint 抽查流水线(自动导出+考卷) | ★★ | claude | — | 1a-0 已落地,可开工 |
| pod 原生装 ROS2 Jazzy + 厂商仿真直编(pod=Ubuntu 24.04,不需要容器;Gate 3B/真球接线的 pod 后备) | ★★ | **yikang** | — | **✅ 冒烟全绿(07-10 晚收官,pod1)**:I/O 契约+推理路(shadow)+发布路/驱动/状态回流(摆动跟踪 0.02-0.08 rad 无熔断)+安全逻辑+reset 链路五关全过:Jazzy 直装 + vendor sim(iceoryx ON)+ pingpong runner 全链原生编译,headless 冒烟 rate=50Hz/sync_miss=0/halts=0——distrobox 不再是唯一路径。装法+学费清单见 runbook「pod 原生 Gate 3 底座」;脚本在 pod1 /workspace/yikang/gate3/。⚠ 三知情项:③**pingpong C++ 路径无 ±20 raw action clip**(仅 29-DOF decoder 有 kA3RawActionClip=20;出分布策略 raw 实测 48-52,仅靠 scale+关节限位兜底)——**⚠07-12 口径更正(对抗复核抓出)**:"训练侧已剪"只对 stage1-fixed-point 线成立(240fcd9 wrapper ±20);**hitter/Gate3 现役谱系训练侧 clip_actions=null=无剪**(A3_DEPLOYMENT_ACTUATOR_AUDIT.md:64 三处明文),C++ 无 clamp 与其训练一致——若补部署 clamp 是在 OOD 态引入 train/deploy 不对称,提案须按「部署安全界对齐 a3_action_decoder」口径谈而非 parity 修复,契约日议题;PD_STAND 直立保持在显式 Euler 配置下 1s 塌掉=知情项①的实证,行为级判读前必须对齐配置;①repo 内 vendor MJCF 无 integrator=显式 Euler(SIM_FIDELITY_NOTE 修复未落),行为级判读前与 jiayi distrobox 版对配置(**07-12 新增 `scripts/view_a3_stand.py`=Mac 纯 MuJoCo 免 AimRT 复现该站姿塌陷的诊断工具,分支 `yikang-linux-port-0711`;修复仍未落=待对积分器配置**);②**历史 179 不可载缺口已关闭**:main runner 现认 `110/175/177/179/180`，当前开放项是 no-publish preflight 仍会放宽 formal 179 模型合同，不能把其 rc0 当 production exact。**07-10 深夜:闭环底座已通**(假球→快解规划器→runner→vendor sim 全链 engage+tts 传递实证;colcon/移植学费 16-18 见 runbook)。判卷级复现 10/10 差 **model_17400_hitter177.onnx(仅存 jiayi 本地,交接项)**;175D 冒烟模型挥后摔=修复栈前代际档案病,非链路问题。**07-11:pod gate3 工作树已切 origin/hitter 并重建全绿**(rally 世代 runner+新规划器 Kalman/landing_mc+Gate3A/rally 编排;闭环编排实跑通过,110-D station 接战行实证)。**✅ 模型已从 wandb 自取自导(07-11)**:BerkeleyPingPong/hope_wbc 有 jiayi 全部本地 run 的 .pt;13200_footfix08 已导出 110-D ONNX(元数据全套)并跑通 pp_gate3_rally 正卷=**该模型 Gate 3 首读数**(13P/7F,3 发回 1,heading 类 FAIL=该世代已知开放问题;配方与学费 19-22 见 runbook)。原文:模型本体确认不在任何分支(setup_local_sync.md 定死=人工交接件):向 jiayi 要 `model_24100_v7.onnx`(cfg 指名)或现役基线 `model_13200_footfix08` 的 exported/policy.onnx;⚠ 110-D 世代硬性要求 external_base/oracle 定位(连挥拍都需要,比 177 更强)。模型到手= cp assets → 改 cfg → 重打包三步即判卷。**07-12 谱系筛卷首轮(满池全程 CPU:play.py 路不可用 → 新配方=克隆 13200 正品 ONNX 图只换 actor initializer+按 run config 改 metadata;数值对抗检查 .pt vs onnxruntime 128 组 max|diff|<4e-6 全过,并用正品自检管线本身)**:①**njfc21an(v2_fresh)model_9000 = 0P/12F**(0 engage:发球 z_w 全出 band[0.67,0.97]±0.05;走位侧 5 发全倒被 operator 救;max|last_action|=152.8>12 sanity)——**判死前有一个未排除混杂**:obs 项顺序继承自 13200 正品 metadata,若 v2 树改过 obs 排序=布局错位假读数(需 jiayi 给 v2 树 observation_names 顺序;另注意该 run 训练 hit_rate 本就 0.6-0.9% 且 crashed);model_7900(峰值档)已导出+对拍过未考(改 dist cfg 一条命令可考)。②**24100 谱系 wandb 全项目搜索不存在**=只能 jiayi 交接。③**v7 真身找到:nc9n1kt5(v3_v7approved_fresh,model_11500)**——但训练在 v6 动作对+reach 平面 x=0.70,考卷写死 0.51 平面几何 → 直接考=站位差 0.19m 假读数,需 harness 变体(x_hit≈1.22+重解发球瞄准)先对齐再考。④**⚠ 团队安全提示:原 pp_gate3_rally.sh 用 `pkill -f` 模式杀,共享 pod 上会误杀别人进程(当场差点命中 Codex 的 cmake 构建)——已建 pp_gate3_rally_safe.sh(预检 fail-loud+只杀自己记录的 PID),共享 pod 一律用 safe 版**。产物在 pod1 /workspace/yikang/gate3/(tools/swap_actor_export.py、adv_check.py、rally_njfc21an_9000_report.json 等;dist cfg 已还原 13200)。下一步=jiayi 三件(v2 obs 顺序/24100 交接/v7 平面对齐)→ Gate 3B 发球生成器/判分器 |
| Hitter 177-D 步法线(20k 从零 + 门禁) | ★★★ | dongc1(jiayi) | main(已合并) | 拉 main 后配方不变(yaml 钉住);新增:训练内上台率曲线可看 |
| Sim2real 部署路:C++ `pp_policy.hpp --planner` 唯一控制路 | ★★★ | dongc1(jiayi) | main | 177-D 对齐 + parity PASS;next:MDU 硬件门 |
| G07 mocap→runner 桥 + world→robot 变换设计(A2) | ★★ | dongc1(jiayi) | main | 设计文档 TODO |

## Queued (priority order)

1. S1 纯因果对照:同起点/同 +4000 的 old helper 和 S1-only 已在 M3/M2 两 family 配对训练;只有终档 S1-only 仍留死区才加 S1+guidance,现在不抢卡。
2. N1 三组:场馆真实时间相关且逐渐收敛的预测误差 / 同幅值打乱时序 / 显式 confidence 或历史输入。
3. R8 拆旗标:envelope-as-penalty 和 RSI stand-height 分开,不再用一臂猜两个因果。
4. W2/S2a/加固包只在 schema-v3 canary 和 `ns=0` shortlist 后排,不抢尺子前面的算力。
5. A5 挥拍视频 30-50 条 + 拍面可测拍摄(场地日顺手)。
6. A6 摔倒管理 + A3 执行器辨识(需专门真机时段)。
7. **head_discipline 验证臂(yikang 预注册提案,07-12 排队;先审后发,绝不自行点火)**:RallyFinalV2 白名单 −0.5 一臂(分支 `yikang-linux-port-0711` `407a443`,基 hitter@`5c346ea`;默认 0.0=现役字节等价——**不发臂=修复零通电**)。判据=head_yaw 不再 park −60° 软限(njfc21an 取证 p5=p95=−60.6° vs 参考恒 0)+ 常规卷不倒退。**排队原因(07-12 实查)**:两 pod 满池无空槽——pod1 三卡 23.0-23.2/32.6 GiB、util 84-98%、17 条训练进程实测(pod2 从 yikang 侧 SSH 不可达[kex reset,疑 IP/port 又变,谁有 console 请更新],板面审计其训练正常);禁 broad kill,空槽归 Codex 预注册回填。**franco 拍板两件**:①rebase 语义雷——origin/hitter 新推 V2Plus 键集从 V2 键集**减法派生**(train.py:447),rebase 后 head_discipline_weight 流进 V2Plus 白名单 → `test_hitter_rally_final_v2_plus_config.py:314` 穷举断言必挂;修法 A=V2Plus yaml 补 `head_discipline_weight: 0.0`(一行,但 V2Plus 头注明写"NOT FinalV3: no passive-head action change",往这个面加 head 键动你定义的 reward 面),修法 B=进排除表(还须同步改契约测试硬编码 CLEARANCE_KEYS[test 45-52 行],接触面更大)——两法都动你的面,你选。②定位——**FinalV3(0fccc3c)已用 passive-head 动作契约根治同一病**(passive_joint_names+raw-action 惩罚+C++ parity+契约测试)=同病两药,head_discipline(奖励层)增量价值仅剩 V2/V2Plus 等**不改动作契约**的谱系,审入按此评估。**附只报不动手的暴露面**:Phase-1 现役谱系(schema-v3 血统)机制层面同病——A3_UPPER_TRACKED 到腕无 head link、joint_pos 31 维全驱含 head、clamp 默认开使 joint_limit 对 park 软限的 head 永不触发(逐路径核过,未做数值复测);是否处置=franco lane。**→ 07-12 Codex 审入裁决已出(见 TIMELINE「yikang 小改动选择性审入」)**:head_discipline **不合 main**(FinalV2 不在 main/V2Plus 派生键集/FinalV3 同病异治),现役配方逐字不变+两道 guard 防静默引入;**采纳路径三步**=①先给现役命名配方显式 0.0 ②与 control/passive-head 做配对 ③与 balance/recovery/ready reward 当同 phase mixture interaction 判——本提案按此路径重定义,验证臂等配对设计后再排,不再按原 RallyFinalV2 单臂形态。

## Done

| Item | Owner | Landed | Where |
| --- | --- | --- | --- |
| **球物理 parity 对照重建落地(关闭 Active 风险"对照物丢失";用户拍板:数据非重点,算法用现有数据验证)**:repo 内 fit 血统参照 hope_training/ball_physics_fit/reference_oracle.py 接管两 parity 测试默认对照(复刻丢失 API,数学=07-03 已入库拟合移植 contact_model/ballcore,g/半径/桌面参数走 venue yaml 真源,先于且独立于 torch 端口=真跨实现对拍);RECORD_DIR 显式设置但缺失 → raise 拒绝静默跳过(fail-loud 制度补齐,skip 只剩 torch/yaml 能力缺失);首轮 pod 对拍即抓真雷:移植版 contact_model 不归一化法向(管线内恒单位 n 从未暴露),参照层修复。验证闭环(pod,44d3379e8680):physics 测试 contact 1e-10/flight+landing 0 误差 ALL PASSED;channel 测试 C++ 厂商栈全链 ALL PASSED(对参照腿 2 bounce 误差 3e-8mm);fail-loud 反证 exit=1;stage4 held-out 验证用 pod 实测数据复算 vs 07-03 定版工件=浮点噪声级一致(flight median 77.4085mm、strike→landing flight_only 0.060m、onoff 100%)——定版常数在补齐数据上完整复现 | yikang(claude) | 2026-07-10 | 分支 stage1-fixed-point bc86995+f0ac2fb |
| **venue 球数据安置对账+补传("原始数据只在 yikang 个人介质"单点拆除)**:pod /workspace/shared/ball_mocap_0703 现为完整主副本——9/9 take 的 C3D 到齐(正常/高球/颠球不转/颠球增旋 四个首次补传;**快速/kuaisu_000.c3d 修复 07-03 上传截断**[449MB→1.51GB,当时只传了 30%,extracted npz 不受影响]);/workspace/yikang/latest_data 镜像同步;07-09 新采集上架 shared/mocap_take_0709(Take_003 ulb+5×C3D,内容待审);11 个上传文件两端 md5 逐一核对全一致;yikang env.sh 接线 BALLFIT_DATA_ROOT,ballfit 管线 require_oracle pod 上全绿。遗留:原始 .tak/C3D 在 pod 之外仍只有 yikang 外接盘(6T 卷)一份 | yikang(claude) | 2026-07-10 | pod /workspace/shared |
| 单翻病定案+S1 face 约定修复:facesign 翻面只翻了实测侧,题库/obs/奖励仍 +Y 约定=反手 face 奖励最优点在错误平面(M3c/M2f 反手同收敛 ~34° 跨资产铁证);修复=_face_pair 统一 face 帧(exp 核/strike_success/face_guidance 实测侧改读未翻 raw 缓冲,S1≡S2 逐位等价、判卷链零改动)+双侧 B 卷卫兵+两道 mjeval 互斥守卫+face_cmd_normal_error_deg 指标+exporter 元数据;止损 R9t/R9u/M3d-live 三病灶臂;"题库按翻面重出"欠账否决关闭;232+11 tests 绿,双红队 APPROVE,现役配置字节等价 | claude(franco 拍板止损) | 2026-07-10 | main(face-frame-s1-0709);病因+设计=pod1 queue.md + s1_wave4/facesign_fix_design_0709.md;TIMELINE 07-10(凌晨二) |
| judge.sh 单命令判卷链(判卷标准入口):env.yaml 自动解析动作对/相位/题库(train→exam 同源推导,解析不到 fail-loud 拒默认值)→ play.py 导出+sidecar → 双侧×双噪声档 bank 考卷(--qdes-clamp/--hold-ref stand 默认)→ md 报告落 run_dir/judge/;--dry-run 供机制检查 | claude | 2026-07-09 | main `66aced9` + runbook 判卷链节 |
| watchdog AUTOJUDGE(收卷自动判卷):patrol_watchdog 巡到臂 DONE 即后台 judge.sh 出卷(M4 首单);顺带修终档 off-by-one(终档名 model_16999=0 起数,曾被当"未到 17000"反复复活 M3 四次) | claude | 2026-07-09 | pod patrol_watchdog.sh;patrol.log DONE/AUTOJUDGE 行 |
| rally 上台率记账根治:per-swing 同刻入账(比值恒 ≤1 =真实回球率;旧口径 `_legacy` 过渡曲线)+ stagger_initial_clock 防同步旗标(默认关)+ 判读纪律(同步臂 EMA 取 ≥21 迭代均值,R4b/R5b/R6b 摔率重审);13 tests 绿,正确性修复豁免消融 | claude | 2026-07-09 | main `53d440e`(台账 `02196d2`) |
| 判炸器检查项 7:支撑脚滑移(foot skate)+ 体序解析 `--body-order`(缺体序 fail-loud);pod 实跑分类:bh v5/v5rg/v5hA 中段 FAIL(滑移峰 0.20-0.35 m/s)→腿移植建议,v4rg/v5hL 干净(≤0.011)、fh v5rg 轻(0.034 PASS),v4/v5 现役头部毛刺归掐头路。⚠ bh v5rg=s1w4 现役跑臂底片:结论只入档不停臂(登记表已注) | claude | 2026-07-09 | main(audit-footskate-0709) |
| simtoreal2 审计合并:防摔修复栈(静摩擦/等球站姿参考/挥拍段限定模仿)全谱系生效;q_des 剪切/脚部惩罚/熵系数旗标化,jiayi 谱系 yaml 钉配方;177-D Hitter 契约/导出/评估/C++ 全链;规划器桌面系+死球修复 | claude(审计)/ dongc1(内容) | 2026-07-06 | main `305d0e8` |
| 训练主尺换装:virtual_return_rate(训练内上台率,全谱系,含正反手分身)+ vb_metrics_only 开关;击球率辅助、composite 降级诊断 | claude | 2026-07-06 | main `305d0e8` |
| NOW.md 大重排:四阶段重排 + 人话化 + GPU 舰队计划;过期章节原样归档 | claude | 2026-07-06 | main(docs)+ PROGRESS.md |
| GPU0 僵尸清理(原 s1w2_A_main 残进程,2.5GB) | claude | 2026-07-06 | pod |
| 阶段 0 收口:握拍标定+烤入+题库+反手 bank(0a-0k) | franco/claude/yikang | 2026-07-06 | PROGRESS.md 归档 |
| 16400_holdfix 三门禁全过 TRUE plant;177-D parity PASS | dongc1(jiayi) | 2026-07-05/06 | `simtoreal2`(已并) |
| Isaac-free 导出链 / eval-B 反事实列 / 换招压力测试 / 固定拍面反解(判死) | claude | 2026-07-04/05 | main;详见 PROGRESS.md |
| First sim-to-real (forehand-only) | yikang/dongc1 | 2026-07-02 | main |

## Update Rule

Update your row when: you start/finish an item, change branch, hit a blocker, or hand something
off. Keep rows one line; details go in the gate doc or PR description.
