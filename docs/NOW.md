# NOW — Active Work Board

Short-horizon board: what is being worked on RIGHT NOW, by whom, on which branch, and what the
next checkpoint is. The long-horizon roadmap lives in
[gates/G08_blind_spot_improvements.md](gates/G08_blind_spot_improvements.md); history lives in
[PROGRESS.md](PROGRESS.md).

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

## Runtime Estimates (RTX 5090, measured 2026-07-03)

| Job | Cost |
| --- | --- |
| DeployParity training, 4096 envs, solo GPU | **~2.0-2.2 s/iter → 2000 it ≈ 1.2 h · 8000 ≈ 4.7 h · 12000 ≈ 7 h · 20000 ≈ 12 h** (corrected 2026-07-03; earlier 4.7 s/iter figure had boot time baked in) |
| Co-running 2 jobs on one GPU (measured) | each job ~20-25% slower, TOTAL throughput ≈ +37%; memory fine (2×~7 GB of 32 GB). Fleet = 3 GPUs × 2 slots = **6 parallel experiments** for signal-tier runs; keep critical-path runs solo. **Stagger starts ≥60 s** (two Kits booting the same second can kill one: the B-ext incident) |
| 算力手册 FINAL (2026-07-03 晚封版) | 同 72 分钟墙钟命中率:**4096=0.42** · 8192=0.30 · 16384=0.07(单调劣化,批次换更新必输)。共卡:2 任务/卡 每任务-22% 总吞吐+37% ✅;**3 任务/卡 每任务慢至 ~4.9s/iter,总吞吐不增反可能低于串行 ✗**。**定版:4096 环境;关键路径独占卡;广度实验最多 2 任务/卡,错峰 ≥60s** |
| Kit boot + env build (per run) | ~2 min |
| Mechanics check (512 envs, 25 it) | ~3 min |
| ONNX export (play.py) | ~2 min |
| Scoreboard, 4 protocols × 2400 steps (CPU) | ~25 min/checkpoint (parallel to training) |
| Reference lineage | shipped models ≈ 9-25k it; treat <8k as immature for deploy protocols |

## Team

| Person | Focus |
| --- | --- |
| franco | Direction, priorities, arbitration |
| dongc1 | End-to-end training bring-up; reward tuning — the simtoreal2 lineage (HER achieved-replay, hold_ready, model_9000 training) |
| yikang | Deployment; sim/env alignment |
| claude (franco's agent) | Foundation: infrastructure, A/B experiments, doc/code hygiene |

## Run-Name Legend (人话对照表 — 报告里不再用裸字母)

| run_name | 人话 |
| --- | --- |
| r3_P2_product | 「产品线」:虚拟球(奖励打到落点、压出球旋转)+ 动捕毛病仿真 + 奖励自动收紧 + 上拍收尾起手,从消旋臂 1500 步续训。**13.5k 跑完,平台 0.89** |
| o2_R7_databoxes | 「真球分布盒」:训练时来球/目标的采样范围换成场馆实测分布(旧手设范围只盖住真球 20%)。**训练内分数不可与别的臂直接比**(它打的题更难),用统一评估裁决 |
| o2_R8_sensorv2 | 「感知三毛病」:模拟动捕在击球时的三种真实缺陷——①偶尔丢一帧(拿上一帧凑合);②击球后约 0.03 秒完全看不见目标(拍子挡住反光球);③每拍重新锁定后带一个小的固定偏移,整拍不变。只骗策略的眼睛,评分照旧。**结论:白送,不掉分** |
| o2_R9_oblique12k | 「斜录实战动作」:斜角机位拍的实战挥拍当模仿模板(替换现役摆拍数据) |
| o2_R10_seed2 | 「换种子重跑产品线」:验证 0.89 不是运气。**结论:同迭代差 0.004,稳** |
| o2_R11_clipswitch | 「中途换招练习」:每挥约 1/3 概率被中途喊停、立刻换一套动作从头打(模拟真机上规划器中途改主意的场景——场馆摔倒的根源之一) |
| day_R6_trimclips | 「剪短模板」:把 2.6-2.8 秒模板剪到约 1.4 秒(只留击球前后),看模仿更聚焦是否更好 |
| R14(待跑) | 「变速播放」:同一套模板随机快放/慢放(0.8-1.2 倍),要求的击球速度同步缩放——看策略能否"要多快给多快" |
| R15(待跑) | 「v5 新视频」:franco 07-04 下午拍的短条正反手当模板 |
| R16(待跑) | 「手腕放开」:拍子那节手腕不再照抄视频朝向(视频里手腕本来就测不准),只照抄挥拍路径;拍面朝向由打球效果的奖励自己教 |
| p21_A_noteleport / p21_A_ext12k | 「基线结构」：挥拍间不传送 + 25% 从站姿开始（main 默认配置），2k / 续跑到 14k |

## 术语人话表(报告与文档里第一次出现的术语,以这张表为准;新术语先上表再使用)

| 术语 | 人话 |
| --- | --- |
| composite(strike_composite_success_exact) | 「击球三合格率」:击球那一帧,位置/速度/拍面朝向三项同时达标的比例——训练里最主要的分数 |
| 信号档 | 短跑对比:2000-4000 步(1-3 小时),只回答"这个想法有没有苗头" |
| 跑到底 | 12000 步(约 7 小时)长跑,出能考门禁的成熟存档 |
| 机制检查(mech check) | 3 分钟小跑:只确认开关真的生效、程序不崩,不看成绩 |
| 热启(warm start) | 从练到一半的存档继续练,不从零开始 |
| checkpoint(ckpt) | 训练存档 |
| eval-B / 真球考试 | 用场馆实测来球分布出题,策略虚拟回球,看能否合法落到对面台(报"回球成功率") |
| deploy-faithful / 部署仿真考试 | 完全按真机流程(站姿起步、等球、一次挥完、不传送)在 MuJoCo 里考,主要看摔不摔 |
| 门禁 | 上真机前必须通过的仿真考试(官方口径=厂商 AGI-MuJoCo) |
| HER 目标重放 | 把策略自己打到过的击球状态混进新目标里练——要求的目标永远"确实做得到" |
| RSI(传送式起手) | 回合开始把机器人直接摆到模板某一帧上,与真机不符,正被站姿/上拍收尾起手替代 |
| 消旋(vb_spin_mode=minimize) | 虚拟球奖励口径:先不追求球质,奖励打到落点 + 出球旋转越小越好 |
| composite 不可比 | 改了训练题目难度的臂(换数据源/换采样分布),训练内分数不能和别人直接比,要用同一张考卷(eval-B/记分板)重考 |
| 采样框/盒(box) | 训练时随机出题的范围:击球点的位置框 + 要求拍速的速度框,按正/反手各一套 |
| 相位(phase) | 挥拍模板里的时间进度,0=起手,1=收尾;击球相位=触球在模板里的位置 |
| 反事实回球率(cf\_\* 列) | 真球考试的附加评分:击球那一刻,把实际摆出的拍面朝向换成"这颗球需要的朝向"再虚拟回球一次(位置、拍速都用实际达成值)——量"光是拍面朝向错,损失了多少回球" |
| 固定拍面反解(--venue-fixed-normal) | 真球考试变体:反过来让规划器迁就策略——拍面钉死在模板参考朝向,只解拍速去凑落点。答出的回球率=不重训练、只改规划器能到的上限 |
| 换招压力测试(--switch-stress p) | 部署仿真变体:每步以概率 p 强制"中途换招"(换模板跳回起手帧+换目标,机器人不动,和真机规划器改主意一个动作),报摔倒率、换招后 2 秒存活率、换招后命中率 vs 干净挥拍命中率 |
| 契约日 | 「改策略输入格式的大日子」:策略吃的观测(现在 175 个数)是训练端和真机 C++ 端共同的约定,改一次要两边同步+重训,所以攒着一起改。首个内容(franco 2026-07-05 拍板)=给观测加 3 个数:**这颗球需要的拍面朝向**(规划器已会算,策略现在根本看不见题目)——先只加角度,侧旋/强度以后按消融一点点加。现有目标信息只有:位置 3 + 拍速 3 + 击球倒计时 1 + 正反手 1 |
| 北极星指标(回球率) | franco 2026-07-05 定:主追踪指标升级为**真球考试回球率**(击球三合格率保留继续看)。理由:三合格率的"拍面项"按模板打分,而模板拍面本身不可信;回球率直接量"真球能不能合法打回去"。现状 0%——它动了才算真进步 |
| 分阶段出题(franco 2026-07-05 方针) | 训练/考试的出题都分阶段:**第一阶段只出"当前动作直接打得回去"的题**(确保奖励拿得到、梯度点得亮),之后按真实来球分布一点点扩散题库——涉及多轮消融,不许一步跳到全真题 |

## 🏆 当前胜利组合(常驻;每出一个裁决就更新这里,谁赢谁进)

**产品配方 v2(2026-07-05 定)** = 虚拟球·消旋 + 动捕毛病仿真(A1 延迟/噪声 **+ R8 感知三毛病,新采纳**)+ 奖励自动收紧 + 上拍收尾起手 + 2% 中途改目标,从消旋臂 1500 步热启:

- 峰值 **0.908-0.910 @ ~9540**(r3_P2_product,tensorboard 曲线复核 2026-07-05;旧记录
  "0.893@11-12k"不准。13.5k 终点回落到 0.885 → **选峰值附近的存档,别拿最后一个**)
- **现役打包模型出处已查明(2026-07-05 指纹匹配)**:`exported_deploy_normbaked` = **model_9600**
  (输出差 5.7e-6),恰好落在峰值区——无需重导;此前"出处未记录"的坑就此关闭(以后导出用
  standalone_onnx_export,--run-path 里写明迭代号)
- R8 感知三毛病:0.890 平台 ≈ 无代价 → **并入配方**(理由:白送的抗噪,真机全是这些毛病)。**跑到底终判(13.5k,2026-07-05):峰值 0.905@10.6k——白送再证**
- R10 换种子:0.885-0.890 → 平台可信,不是运气。**跑到底终判(13.5k):峰值 0.914@10.3k,比 P2 自己还高——稳上加稳;且所有长跑峰值都落在 ~10-11k、终点回落,"选峰值存档"铁律再+1**
- **候审**(信号档在跑,赢了进组合):R16 手腕放开 / R14 变速播放 / R12 减速塑形 / R11b 低剂量中途换招
- **等统一考卷**(训练内分数不可比):R7 真球盒(0.83@8.7k 还在爬)/ R9 斜录(0.71 到顶)/ R15 v5(排队)
- **已拒绝**:R11@0.002 剂量(命中率税 0.12,已停在 10.7k;抗摔收益已复核:换招压力测试 14 组全 0 摔,收益轴饱和、税为真)、R6 剪短模板@2k(无优势)
- **⚠ simtoreal2 合并警示(2026-07-05)**:DeployParity 任务 yaml 现在带着 jiayi ARM A 重训的
  实验默认值(回合 16s、等球 0-8s、A8=0.25、base_decel=ON、位置奖励 std 0.15)。**产品线
  (VirtualBall)已在自己的 yaml 里显式钉回基线值不受影响**;在 DeployParity 上开新臂的人注意
  基线已变,对照要自带。jiayi 的 std 0.15 理由(0.20 下模仿白拿 0.63 位置分=偷懒局部最优)
  值得产品线走梯子验证 → 排为候选臂 R17

## Plan To Saturday (2026-07-03 → 07-04, target: play at the venue with an improved policy)

Critical path (GPU 1/2) — ETAs corrected 2026-07-03 15:40 with the measured 2.15 s/iter:

1. **TODAY ~18:15** — 「基线结构」and「三合一」long runs hit 14k iters → 4-protocol scoreboard
   verdict (claude, ~40 min, CPU).
2. **TODAY 19:00-24:00** — explicit-clipped-PD fine-tune from the winner (~8000 it ≈ 4.7 h, GPU1);
   GPU2 = backup ft from the other run. (claude)
3. **TONIGHT/midnight** — export → MuJoCo explicit gate + deploy-faithful → build the MDU package
   (`build_a3_deploy_pkg.sh`). Saturday morning = verification margin, not critical path.
   (claude prepares, yikang ships/runs on the MDU)
4. **Saturday** — deploy & play: forehand first; **try backhand in SHADOW mode** —「三合一」and
   model_9000 both trained stand-entry, this is the potential headline. (yikang + franco)

Parallel tracks (no GPU conflict):

- **Ball physics v1 (P2.5 prerequisite)**: mocap ball-trajectory collection is happening NOW at the
  venue → fit drag/bounce from the fresh recordings (planner calibration path); spin-aware physics
  arrives with the simtoreal2 merge. Owner: **yikang** (franco 2026-07-03).
- **Teacher clips for Saturday: NO mass shoot needed** (franco 2026-07-03): the swings recorded
  during play sessions + orientation rotation (P2.2-lite, `reground_hope_frame.py` path) are
  sufficient; the 0703 clip uploads are that set. A5's 30-50-clip library and the dedicated
  ready-stance clip stay on the longer-horizon list (P2.0/A5 in G08), not on Saturday's path.
- simtoreal2 → main merge + doc updates (claude, in progress).

## 2026-07-04 白天台账(物理建模冲刺 + 白天测试/晚上跑满)

**入 main 的交付**(细节见 ball_physics_v2_roadmap / fit report §11):
- planner:12 态 EKF(有色噪声,影子模式)· Magnus 项 · Nakashima 自旋耦合反弹 · StrikeSpec
  逆解器(拍面角+v_n/v_t+落点敏感度;实测预算:v_n 0.85 m/(m/s) 是王,拍面 0.04-0.06 m/度)
  · venue 参数一致性守卫 —— 共 73 测试全绿
- 训练:A1v2 感知三缺陷(丢帧/击后盲窗/每击重锁偏差,默认关)
- configs/incoming_ball_venue.yaml:实测来球分布。**头条:手设 vb 盒只覆盖真球 20%**(训练
  一直在打"想象中的快平球")。采样原则(franco):**训练=按类重平衡的丰富度盒**(x −3.29~−0.10,
  z 到 −2.30,旋转 65 rad/s),**评估=综合分布**(答"真球接得住吗")
- 裁决:a_t=0.52 维持(自旋输入非病根;轨迹反推在本 rig 不可行 0/144;滚动上限被 CI 排除);
  双拍=打法混淆(§11.1);自旋衰减关闭(τ>5s)
- 实战互证:上旋不飞(接触抬升与 Magnus 下压线性抵消)、侧旋飞出台(无抵消)——**拍面补偿
  优先场景=侧旋**,已固化测试

**在飞**:P2 收尾(→自动导出终版)· eval 契约修复(P0,旧行为保留)+ 之后加"来球分布
驱动"的 B 模式评估 · R7+R8 机制检查 · 通宵舰队待命(R7 重平衡盒/R8 感知缺陷/R9 斜录
12k/R10 换种子,全部跑到底,P2 完成后自动点火)

## 10:00 判读(2026-07-04,自动流水线交付)

**判词:值得练到底。** P2 产品线(消旋虚拟球 + A1 校准噪声/延迟 + σ收紧 + 挥后起手,
R2@1500 热启)在 5500/12000 步时命中率 **0.858(峰值 0.884)**,已超 E 谱系终点 0.8128
—且带着 E 线没有的全部真实性约束。曲线在 0.85-0.88 平台微涨,无退化迹象。训练继续。

- 快照:`logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-03_22-32-36_r3_P2_product/exported_10am/policy.onnx`(iter 5500,Isaac-free 导出链)
- 信号档裁决(2000 步):R0=0.502 / R3 斜录=0.520(未过门槛,但 2 秒新 clip 首战打平)/
  R5 A1=0.491(几乎免费→进组合)/ **消旋 0.769 > 上旋 0.753(yikang 线)→ 组合采用消旋**
- 认证提醒:MuJoCo 评分器对连挂模型失效仍是 P0 —— P2 上真机前需修好评分器或走
  yikang 的厂商门禁

## Roadmap Scorecard (2026-07-03 傍晚版)

阶段定义(每项按同一条阶梯,标到已完成的最远一格):
**设计**=方案共识并写入文档 → **代码**=实现完毕(功能类默认关)→ **跑通**=机制验证(短跑无崩、开关生效)→
**有效**=信号档 A/B 有对照数字 → **门禁**=成熟版 checkpoint 过 MuJoCo 四协议 → **真机**=硬件验证。
数据类条目按:方案→管线→试样→拟合→入训练/规划器→真机。

| 项 | 设计 | 代码 | 跑通 | 有效 | 门禁 | 真机 |
| --- | --- | --- | --- | --- | --- | --- |
| P2.1 连续挥拍不倒 | ✓ | ✓ | ✓ | ✓(2k A/B) | **今晚终审** | 周六 |
| P2.2 朝向归一(现有 clip) | ✓ | ✓(re-ground 管线) | ✓(0703 clips 在训) | 隐含在主线里 | 随主线 | 周六 |
| P2.3a 自适应σ | ✓ | ✓ | ✓ | ✓(+48%) | **今晚(三合一)** | 周六 |
| P2.3b HER 目标重放(jiayi) | ✓ | ✓ | ✓(jiayi 线) | **今晚对照** | 未 | 未 |
| P2.3c 动作库检索(SMASH) | 未 | — | — | — | — | — |
| P2.4a 等球 hold+hold_ready | ✓ | ✓ | ✓ | 今晚对照 | 未 | 未 |
| P2.4b 减速命令/base 回归/ready pose | ✓(v3 doc) | 部分(base_decel v1) | ✓ | R12 待跑 | 未 | 未 |
| P2.5 物理建模 | ✓ | 部分(spin 代码在 main) | **数据采集中** | 未拟合 | 未入训练 | 未 |
| P2.6 smash | 未 | — | — | — | — | — |
| A1 延迟/时变注入 | ✓ | ✓ | ✓ | **今晚** | 未 | 未(数值等标定) |
| A2 坐标变换归属 | **未写**(已立项 yikang) | — | — | — | — | — |
| A3 执行器辨识 | 未 | — | — | — | — | — |
| A4 评估基建 | ✓ | ✓ | ✓ | ✓(已用于三次裁决) | 不适用 | 真机落盘制度未建 |
| A5 新视频 30-50 条 | ✓(主动推迟) | 管线已有 | — | — | — | — |
| A6 摔倒管理 | 部分(绝对终止在) | 部分 | — | — | — | — |
| A8 post-swing 起手 | ✓ | ✓ | ✓ | ✓(2k 中性) | **今晚(三合一)** | 未 |

读法:**一列看进度深度,一行看卡在哪一格**。今晚之后"门禁"列会填掉 4 格;"真机"列整列等周六。

## 动作源消融(franco 2026-07-04 拍板,今晚/明天必跑)

事实修正(2026-07-04 凌晨):registry 里两套(v3/v4 原始与 :latest 微调转正)**均为正面摆拍录制**;
此前 npz frame-0 yaw +82° 是 GVHMR 全局朝向产物,不代表机位。真正的**斜录(实战击球)视频**是新东西:
franco 已提供 forehand_new.mp4 / backhand_new.mp4,已上传 `/workspace/shared/motions/raw_video_oblique/`
(附 README 说明谱系)。franco 预期:实战斜录的动作质量会好很多。

| 消融臂 | 数据 | 状态 |
| --- | --- | --- |
| 正录(现役) | hopex 转正版(139/132 帧) | 所有既有结果就是它,无需重跑 |
| 斜录(新) | raw_video_oblique 两条视频 | **卡在 GVHMR→GMR→npz→转正→相位标定 管线**(pod 无 GVHMR 环境;dongc1 机器有)→ 产出后同配置 2000 步对照 |
| v5(更新) | raw_video_v5 两条视频(1.17/1.20s,franco 2026-07-04 下午提供) | **管线全通(2026-07-04 晚,claude,`/workspace/franco/v5_pipeline.sh` 复用斜录管线)**:npz 56/58 帧 @50Hz 已入 `/workspace/shared/motions/hope_{forehand,backhand}_v5.npz`,yaw 转正(+86.6°/+83.6°→0),相位 **[0.673, 0.345](franco 纠错版;检测器的 0.768 是甩鞭峰不是触球)** → 消融臂 R15 |

## 关键更新联合消融(Isaac 复活即发射;2000 步信号档,4096 envs,一卡两跑错峰)

| 臂 | 配置(完整命令附后) | 回答什么 |
| --- | --- | --- |
| R0 基线 | `task=HOPEPingPongDeployParity`(main 默认) | 合并后的参照点 |
| R1 虚拟球·上旋 | `task=HOPEPingPongVirtualBall`(yikang 默认,落点30/过网20/旋转5,拍速法线降权) | 物理奖励栈是否成立 |
| R2 虚拟球·消旋 | R1 + `task.racket.vb_spin_mode=minimize`(franco 第一阶段:不奖励球质,奖励落点+出球旋转最小) | 两种旋转哲学谁先学会站稳打准 |
| R3 斜录数据 | R0 + `motion_file=/workspace/shared/motions/hope_forehand_oblique.npz motion_file_2=..._backhand_oblique.npz "task.racket.strike_phase_per_clip=[0.368,0.495]"` + 下方专属采样框 | 实战动作源是否更好(franco 预判:是) |
| R4 组合 | R1/R2 胜者 + adaptive_sigma + post_swing(可再叠 R3 若其胜) | 产品候选 |

斜录臂专属采样框(从斜录击球帧提取,保持参考-目标一致性):
`--pos-range-per-clip 0.08 0.28 -0.72 -0.32 0.79 0.99 0.38 0.58 -0.51 -0.11 1.10 1.30`
`--vel-range-per-clip 1.52 2.52 -0.57 0.43 0.49 1.29 0.73 1.73 -0.30 0.70 0.06 0.86`
(yaml 键 pos/vel_range_per_clip 同值;斜录反手峰速仅 1.33 m/s 偏温和,视频重录候补)

新增部件状态:vb_spin_mode=minimize 已实现入 main(默认仍 topspin);斜录 npz 已产出
(`hope_{forehand,backhand}_oblique.npz`,96/106 帧,yaw 已转正,MuJoCo-FK 转换器 body 残差 0.00mm;
速度用有限差分,与 Isaac 雅可比法峰值差 ~0.18m/s——首个 mech check 时顺带确认无碍)。

## 基建现状(2026-07-04 凌晨)

- **pod 宿主机对 Isaac 判死**:裸 Kit 在重建容器上仍挂死;Stop→Start 不换宿主机(卷钉死机器)。
  训练全面阻塞。选项:RunPod 支持重置宿主机 / 新建 pod 迁移(venv 可重装,数据 ~20GB 可 rsync)。
- **导出与评分已绕开 Isaac**:`standalone_onnx_export.py`(纯 CPU 重建 actor+归一化器+动作缓冲,
  元数据从同配置旧 ONNX 拷贝)——两个 14k checkpoint 已导出成功,四协议评分进行中(纯 CPU)。
  周六候选的裁决不再依赖 Isaac。

## 周六早晨交接(2026-07-04 04:30 写)

**给今天打球的建议(按风险排序):**
1. **主用 07-02 已验证的部署包**(yikang 手上,正手已在真机成功过)——零新风险,保底能打
2. **E@14k 作为实验候选交给 yikang 过厂商门禁**:ONNX 已导出(位级忠实,`.../p21_E_sigma_postswing/exported/policy.onnx`,sha12=e2c19a01cd3e)。他的 AGI-MuJoCo 门禁环境在他自己机器上,不受 pod 影响。**过了门禁才上机,过不了就只用保底包**
3. Isaac 复活后第一件事:A/E 的正规 Isaac 评估补认证

**今晚查明的三件大事:**
- ~~**pod 宿主机对 Isaac 判死刑**(裸 Kit 两种缓存状态都起不来;裸 CUDA 正常)→ **需要:RunPod 支持票 或 新建 pod**。训练全阻塞,消融梯/斜录臂都在等这个~~
  **→ 已解决(2026-07-03 19:15, yikang):训练在这台 host 上复活了,不用迁移 pod。** 死刑判早了:barekit6 的 faulthandler 栈显示挂点在
  `isaacsim.asset.importer.urdf` 扩展 on_startup **无条件构建导入窗口 UI**——headless 也建,而第一个 `omni.ui.StringField` 控件创建在这台
  host 上永不返回(渲染栈 iray/RTX 确实坏了,franco 的判断对了一半)。修法:extscache 的该扩展打了**环境变量门控补丁**
  (`HOPE_URDF_IMPORTER_NO_UI=1` 跳过 build_ui,不设=原版行为;URDF→USD 转换 API 不依赖该窗口)。yikang 的 env.sh 已默认导出;**其他人的
  env.sh 也要加**。验证:HOPEPingPongVirtualBall 4096 envs 正常进入 PPO 循环(vb_smoke, GPU0)。注意:**需要 RTX 渲染/相机的任务在这台
  host 上仍然是死的**(_wait_for_viewport 挂,iray 插件加载失败)——headless 无相机训练/评估 OK,录像/render 类任务仍需迁移。
- **MuJoCo 评分器与连挥模型的契约不匹配**(P0,周日修):新旧两版评分器对 E@14k 给出一致的病理性全零(回合 ~1s 截断、只评反手、误差系统偏移 ~0.5m);真机验证过的 07-02 模型在同评分器下 0 击球窗口。评分器的回合/参考时钟按旧谱系(瞬移单挥)假设写死,对 wrap_teleport=False 的模型无效。证据:/workspace/franco/{premerge_check2,knowngood_check2}
- **Isaac-free 导出链已建成并验证**(standalone_onnx_export.py + harvest,位级零差)——导出永不再被 Isaac 绑架

**斜录(实战)动作:视频→CSV 全通**(GVHMR/GMR 在 pod 上直接跑通了,产物 /workspace/shared/motions/oblique/,58/64 帧)。剩 csv→npz 一步需要 Isaac FK(或我写 MuJoCo FK 版,已排队)。球轨迹数据已入共享区 ball_mocap_0703。

## 🔴 结构级发现(2026-07-04 傍晚,eval B 模式首跑)— 需要 franco 拍板

**策略的拍面朝向是"clip 锁死"的,175 维观测契约里没有任何法线指令通道。**
B 模式(真实来球分布 + StrikeSpec 反解应有拍状态)实测:位置/速度跟踪意外地好
(OOD 到 3.7cm / 0.18 m/s,双双满分)——但法线误差 36-76°,虚拟回球率 **0%**。
反事实归因:同样的达成质量,换上"应有法线"→ **25/25 合法回球**(中位 6.7cm);
换回错误法线 → 0/25。**法线是唯一短板,而且是架构性的**(策略只模仿 clip 的拍面,
无从得知"这个球需要什么拍面")。这正是 franco"拍面角度是接旋球的关键"的最终形态。

两条路(可并行):
- **A 短期(零训练,立即可部署)**:planner 反过来适配策略——StrikeSpec 加"固定法线求解"
  变体(法线钉在 clip 参考值,只解 v_r/落点),放弃部分落点自由度换取合法回球。claude 可即做。
  **→ 已做并判死刑(2026-07-05,claude)**:固定法线求解已实现(`solve_fixed_normal` +
  `--venue-fixed-normal`,16/16 测试过),复测结果 **0% —— 不是求解器不收敛,是物理不可行**。
  暴力可达性扫描(拍面钉在 clip 参考朝向,遍历 ≤6 m/s 全部拍速,~7000 落点/球)证明:
  **正手拍面(几乎朝正侧方 [0.41,0.90,-0.17])打出的球最远落 x≈1.4 m,连网(1.87 m)都
  过不了——任何拍速都救不了**;反手拍面只剩一条"贴网大斜线"小缝(落点 x≈1.9-2.0、
  y≈±0.3-0.67,需 2.5-4.3 m/s),全部在合法回球框(过网纵深 ≥0.3 m,即训练自己的
  dink 保护)之外。前提核实过:mode-A 里策略实际拍面 vs clip 参考只差 1.9°(达标率 100%),
  钉 clip 参考=钉策略真实拍面。**结论:规划器迁就救不了现役 clip 拍面,路 B(契约加法线
  通道)/ R16(手腕放开)/ 新拍面数据是唯一活路。**
  **→ 第三根钉子(2026-07-05 晚,R16 信号档 eval-B)**:手腕放开跑了 5.5k 步,拍面-vs-应有
  误差 43.5°(基线 42.5°)纹丝不动、回球率仍 0%——**落点奖励的间接梯度掰不动拍面,
  "契约日加法线指令通道"从三选一变成唯一剩下的路**。
  **→ 第四根钉子,也是机制解释(2026-07-05 晚,P2 训练曲线复盘,franco 追问"训练有落点
  奖励为什么没学会"逼出来的)**:P2 训练全程 13.5k 步,虚拟球**落点入界率 0.03%、过网率
  0.2%、落点误差 1.9m(和考试的 1.5-1.6m 同病)——三个打球奖励项(落点 30/过网 20/消旋 5)
  的实际收入全程 ≈ 0**。也就是说:0.89 的三合格率全部来自模仿+位置/速度跟踪,"打球效果"
  奖励装了却从没通电;R16 放开手腕也没用,因为一次正样本都没有、没有梯度可爬(奖励沙漠,
  yikang 07-03 已放宽过一轮核仍如此)。**训练分布 ≠ 考试分布不是主因——训练分布下它同样
  一颗都打不回**。治法两条腿(o4 轮):①契约日拍面指令 = 直接姿态梯度,不经过接触模型;
  ②franco 分阶段出题 = 把第一阶段题目限制在"物理上打得回"的域内,先点亮奖励再扩散
  (注意:可达性扫描已证正手拍面在任何拍速下无解,所以 curriculum 单独救不了正手,两条腿
  必须一起上)。
- **B 长期(下一代训练)**:观测契约 175→178(+目标法线 3 维),训练 racket_normal 奖励改跟
  指令法线,部署 pp_obs_builder 同步——一次契约变更,换来真正的"按需拍面"。需要全队排期。
  **路 A 死刑后升级为"必须做",不再是可选项。**

(附带发现:场馆来球接触点在人的击球高度 0.98-1.26m,高于训练框——采集分布是"人接球"
视角,机器人版分布尚缺;B 模式 v1 盒采样未强制相关性,升级路径已注明。)

→ 设计已固化并按代码逐条核实:[motion_and_contract_v3.md](motion_and_contract_v3.md)
(2026-07-04 晚,claude)。核实中的新发现:①`racket_target_vel_w` 是世界系裸直通——现役
契约里只有位置做自我中心化,速度不做(真机 identity-yaw 下退化一致);②部署线协议
`RacketCommand.normal` **已经在传法线**,缺口只在 obs 坑位与训练奖励参照;③critic 已有
特权 `racket_target_normal_w`,契约日训练侧只是把 racket_normal 奖励参照从 clip 换成指令。
迁移定为 **175→179**(法线 3 + ρ 1,可顺手把 vel 也改自我中心)。

## o4 轮计划 v4(2026-07-05 深夜,franco 八条批注后定稿;北极星=回球率)

**第一性原理(全部实锤)**:①训练目标不是"能回球的解"(诊断 A:完美跟踪入界率 0.00%/0.03%,
误差中位 1.84/1.92 m,与 P2 训练实测严丝合缝——0 回球 100% 是目标的锅);②策略看不见题目
(观测无拍面要求);③打球奖励信号比 ≈ 1:5000,PPO 看不见。
**方向 = 教打球从结果空间搬回控制空间**:让已被证明有效的跟踪奖励(位置 3.7cm/拍速 0.18m/s
就是它们教的)指向**逐球反解的应有拍状态**;落点奖励降级为验证奖金。

**franco 八条批注(07-05 深夜)落地**:相位消融最优先(斜录的失败可能就是击球帧没对上);
调参空间要标跑几次;契约要可扩展(侧向速度等后续还加);先改 doc 再开工;**反解直接上 torch
批量求解器**(训练内联提速,部署以后也能共用同一套解法;开关化,numpy 版留作对照测试);
位置放开拆成拍点(p_racket)和站位(p_base)两步 → **阶段升为 4 段**;判"动作行不行"的标准
不是拍面朝不朝前,而是**"这个动作在这个击球点、按它自己的挥速特征,能不能把球合法打到对面"**
(斜下拍面+斜上挥速是合法攻球组合,可回球性扫描按此重定义)。

### 阶段 0:出题与相位校准(纯 CPU,最优先)

| 项 | 内容 | 判据/产物 | 跑量 |
| --- | --- | --- | --- |
| 0a 目标不是解 | ✅ 已做:完美跟踪入界 0.00%/0.03%(2 万题/边) | 一锤定音 | 完成 |
| 0b 难度地图 | **✅ 已做(07-05,10 万题,可解率 94.5%)**。难度=应有拍面偏离模板角度:**正手中位 71°,98% 的题落在 60-90° 层,0-45° 层空的——从模板出发没有任何"近题"台阶**;**反手中位 37°,阶梯连续**(15-25° 3% / 25-35° 37% / 35-45° 51% / 45-60° 10%)。需求拍速中位两边都 ~1.9 m/s(速度不是瓶颈)。**设计结论:阶段 1 反手先行**(v5 反手 f17-18 窗口 + 连续难度阶梯,curriculum 有路可爬);**正手没有低难度起步区**——要么靠拍面指令硬拉 60-70°(最难臂,风险高),要么**下次拍摄清单加"拍面朝前的正手挥拍"**(franco 定)。产物 qbank_v0.npz 转作考卷/curriculum 刻度 | ✅ 完成 |
| 0c v5 拍面扫描 | ✅ 已做:v5 触球帧 0.47/0.46 vs hopex 0.41/0.39,v5 反手相位 0.18-0.23 有朝前段 | 按 0d 重新裁决(朝前分量只是粗代理) | 完成 |
| **0d 相位×可回球性扫描** | **✅ 已做并已并进 yikang 的工具(07-05 深夜)**。⚠ 语义修正(yikang):登记表 `phase` = 视频真值(人何时触球),扫描给的是**训练最优相位**(clip 运动学哪帧最适合机器人回球)——两者可合法不同,登记表新增独立字段 `train_phase_candidates` 存后者,**绝不回写 phase**。结果:六条 clip 五条全程 0%;**唯一窗口 = v5 反手 f17-18(相位 0.30-0.32)**,视频真值帧 21(0.362)两侧都是 0%;窗口分数随球速档变化——**场馆速度球 96%,阶段 1 默认 2-5 m/s 只有 17% → 阶段 1 起步球速用场馆档 [~1.0,2.5] 再放宽**。斜录换相位救不了(全程无窗,登记表已核 0.432/0.495 同样 0%)。实现已合一:yikang 的 gen_stage1_questions.py 加 --phase-scan 模式(分支 stage1-fixed-point,b36a008),我的草稿扫描器退役;他的 z 坐标陷阱教训(planner 残差全绿≠落点对,必须 torch 闭环复检)已内建 | ✅ 完成 |

| **0e 三假设翻案探针(franco 质疑"人明明稳定回球",07-05 深夜)** | ①**混帧 bug(franco Q1 猜中)**:分析器 normal_root 是骨盆系、速度/位置是世界系,挥拍中段骨盆转动把扫描/题库里的"拍面"转走几十度——已修(normal_w 世界系,b4825a0),修后 v5 反手窗口 f17-19=100%(场馆速度),两套实现完全对账,题库难度角改正为 正 69.6°/反 38.5°。②**斜线对拉(franco Q2 猜中)**:把球台绕击球点转 ψ 复测——**v5 反手在 ψ=-40°~-50° 时可回球性 88-100%(上旋球最优),ψ=0(我们假设的正对几何)是 0%**:人打的就是约 45° 斜线,视频真值帧 21 在正确几何下完全成立;f17-19 vs f21 之谜消解=两帧各自对应不同球台朝向。③**正手另有隐情**:方向指纹显示正手标注帧出球**射程中位仅 0.05 m(球直接砸脚下)**,转任何 ψ、±50° 握拍偏置都救不活——是 GVHMR 拍面重建在该帧不可用(登记表 face_normal_reliable:false 的实锤),不是"人的正手不行"。④握拍偏置(Q3):共用 +20°z/+10°x 能把合计从 0% 提到 50%(主要救反手),待叠加 ψ 修正后重测 | **管线修正提案:登记表加 rally_yaw 字段**(每 clip 的对拉轴向,视频/探针可测),重定向时按它转正——"转对方向"从此制度化;正手模板的拍面重建不可信,阶段 1 正手锚废弃,等新拍摄或纯指令拉动 | ✅ 探针完成,rally_yaw 待 yikang 接 |

**0f 定论(franco 2026-07-05 深夜):"拍面一直是算错的"——因为管线根本看不见拍子。**
GVHMR 只重建人体;所有视频 clip 的"拍面"= 腕系 +Y,是虚构量不是测量值(登记表
face_normal_reliable:false 的最终解释)。人的真实拍面 = 腕 × 握拍角,握拍角从未被观测。
**什么塌了、什么还立着**:
- 塌:一切"视频 clip 的拍面朝向"当物理事实的解读(正手"球砸脚下"= 人的腕姿态 × 机器人拍装
  几何的组合不成立,人自己的拍子当然不是这么装的);
- **立着(关键)**:机器人的拍面严格 ⊥ 腕+Y(STL 实测 0°)——所以"机器人复刻 clip 腕姿态时
  拍面指向哪里"是精确可算的,0b/0d/0e 全部结论按"机器人执行"口径依然成立:反手先行、
  正手腕姿态离任何解 ~70°、v5 反手 f17-19 锚点有效;
- 推论:①R18 拍面指令通道的理由从"三重"升为"四重"(视频拍面本就不可作为监督源);
  ②**采集升级进拍摄清单**:下次拍摄让拍面可测——拍上贴标记/棋盘格,或正对机位+同步球轨迹
  (场馆球轨迹已有,若有同场身体视频可反推每场的握拍角=把虚构量变成标定量);③在此之前,
  所有视频 clip 的腕朝向只当运动先验,不当拍面真值(R16 的直觉获得最终背书)。

### 阶段 1「固定点养成」:学会用合适的角度和速度,把不同速度的无旋球打回固定落点

开关(全部默认关,现状逐字节不变;合并安全):`题目模式(现状/反解)`、`击球点(固定/框/真球)`、
`旋转档位(无/低/场馆)`、`观测契约(175/178+版本号,侧向速度等预留扩展位——demanded 拍速矢量
本就在观测里,反解后自动携带侧向分量;新增仅拍面 3 维)`、`torch 求解器(开=训练内联反解)`。

**第一波 6 臂(一个舰队波次,~4-5.5k 步信号档,约 4 小时;判据=阶段 1 考卷的回球率+拍面误差)**:

| 臂 | 组合 | 回答 |
| --- | --- | --- |
| 1 主攻 | 反解目标+拍面观测+奖励改锚+手腕模仿解除;击球点固定、无旋、速度变 | 方向成不成 |
| 2 对照 | 主攻减拍面观测(175 维不动) | 观测通道的净贡献 |
| 3 相位A | 主攻 + clip 击球相位=0d 扫描最优 | 相位错位假设(franco 第 1 条) |
| 4 相位B | 主攻 + 相位=现标注(=臂 1,若 0d 说标注就是最优则臂 3/4 合并,腾出一臂给权重) | 同上 |
| 5 权重 | 主攻 + 拍面指令奖励权重 ×3(第一档粗探) | 新老师话语权粗标定 |
| 6 基线 | 产品线原样同种子 | 锚 |

**第二波(第一波出信号后,~6 臂)**:权重细扫 2 档 × 拍面奖励容差 2 档 + curriculum(难度窗口
自适应 vs 全开)2 臂。**阶段 1 预算合计 ≈ 2 波 12 臂 ≈ 1-2 天墙钟。**

### 阶段 2a「拍点放开」:击球点 固定→框→真球分布(站位基本不动,臂展内应变)

赢家热启;开关 `击球点` 逐档;配套=位置跟踪奖励(已有)。1 波 4 臂(两档分布 × 有无 curriculum
+ 对照)。**注意与 2b 分界:此段 p_base 仍钉住。**

### 阶段 2b「站位放开」:p_base 放开(步法进场)

球的到达点超出臂展 → base 目标由球驱动(不再是 ±0.1m 抖动);减速入位(R12,已证无税)在此段
并入配套;远球/跑动考卷同步建。1-2 波(base 分布逐档 + R12 有无消融)。

### 阶段 3「旋转进场」:无旋→低旋→场馆水平

反解自动给出随旋变化的拍面/切向速度目标;消旋奖励从此才有真效用。1 波 4 臂(旋转两档 × 消旋
奖励有无 + 对照)。

**总预算粗估:5-6 个舰队波次 ≈ 30-36 臂 ≈ 一周内(现有 3 卡×2 槽)。每阶段判据过线才升段,
考卷与训练同阶段同步(变速考卷并入阶段 1 考卷家族)。**

搭车项(不占主线):R12「刹得住」专项考(并入 2b)、R17 位置奖励收紧(阶段 1 第二波搭车)、
R15/斜录重赛(相位校准后并入阶段 1 相位臂)。

## 快速横向对比制度 + 新消融臂(2026-07-04 晚;franco:想法变多了,不再默认训久)

**制度(新想法一律走梯子,不再单独长训):**

1. 机制检查(512 envs × 25 it,~3 min):开关生效、无崩。
2. **信号档 2000 it @ 4096 envs**(独占 ~1.2h / 共卡 ~1.5h;3 卡 × 2 槽 = 6 臂/批,
   错峰 ≥60 s):同种子、同热启(当前 P2 产品线 ckpt),**每批必带同批对照臂**。
   裁决口径 = Isaac composite(2000 步,同 10:00 判读)+ eval-B 回球率(CPU,不占卡)双轴。
3. 只有信号档赢家进组合臂;组合赢家才升 12k(~7h)与门禁。输家记下数字后关闭。
4. **不急着跑全(franco 2026-07-05)**:新臂默认 4000 步(热启臂)/2000 步(从零臂),
   赢家再续跑(checkpoint_path 无缝续);**平台化的长跑当场停**(先记数字)——R11 在
   10.7k 平台 0.762 被停就是首例。每张卡最多 2 个任务、错峰 ≥60s 的定版继续有效。
5. **排队脚本必须"发射前查卡"(2026-07-05 事故规则)**:GPU1 曾出现 3 任务同跑,复盘 =
   两条发射链都把卡号写死在写脚本的时刻 + 一个手动加的臂(R11,06:27 临时发的 12k 长跑)
   没进任何账本 → 4 小时后舰队脚本按旧地图开火撞车。修法已落地:看门狗改用 wait_slot
   (发射瞬间数一遍卡上 >2GB 的进程,<2 才发,占满则等待并每 30 分钟报一次);**临时手动
   发臂必须当场进 NOW 消融表**(就是"先认领再动工"规则的 GPU 版)。加固版
   `o3_watchers2.sh` 已替换旧链。

**新臂(接 R 系列;R0-R10 已用):**

| 臂 | 配置(叠在当批对照/胜者上) | 回答什么 |
| --- | --- | --- |
| R11 | `task.motion.clip_switch_prob=0.002` | **已停 2026-07-05(平台 0.762@10.7k,命中率税 0.12)**。剂量 0.002=每挥约 28% 被打断,远高于真机频率;回合长度 463≈469 说明没有多摔,掉的纯是命中率。~~真收益(抗切换摔倒)的量尺缺失~~ **量尺已建并量完(2026-07-05,`--switch-stress`):纯换招扰动压不出收益**——P2(没练过换招)在两种 PD 口径(Isaac 对照 implicit + 门禁 explicit clipped-PD)、24000 步 230 次换招(127 次在挥拍中段)下 **0 摔、换招后 2 秒存活率 100%、换招后命中率 0.97-1.00**;R11 同样 0 摔,唯一可见差别是它在门禁口径下命中率还略低(0.98-0.99 vs P2 的 0.99-1.00,税的延续)。**判读:MuJoCo 里换招离散跳变本身不构成摔倒威胁,switch 训练在这把尺子上零收益、税为真 → R11@0.002 维持拒绝**。尺子的适用边界:场馆真机摔倒可能是换招×感知毛病×延迟的交互(纯换招在干净仿真里不复现);若还要追这个方向,下一步是压力协议叠 A1 校准噪声或"击球窗口内定点换招",而不是继续扫剂量 |
| R11b | `task.motion.clip_switch_prob=0.0005`(剂量/4,**已点火 2026-07-05,o3_R11b_switch5e4**) | franco:继续调参找好参数。先扫剂量;若 0.0005 仍税重,下一轮动奖励侧(打断后 hold 补偿)。**⚠ 判读标准更新(2026-07-05 换招压力测试结果)**:抗摔轴上 P2 本来就满分(0 摔),R11b 在该轴不可能赢——它的存在意义只剩"更低剂量是否免税"(Isaac composite 对齐 P2)。**4000 步跑完,压力测试终值已量(exported_it4000_normbaked,2026-07-05)**:230 次换招 0 摔、换招后 2 秒存活 100%、两种 PD 口径命中率干净/换招后均 1.00(@2700 临时值相同;分布内不伤命中,好于 R11@0.002 的 0.98)。**信号档跑完(→5499,2026-07-05 判读):composite 峰值 0.839@5247,同批最低(R12 0.888 / R14 0.871 / R16 0.870)——低剂量税没免掉(~0.03-0.05),抗摔轴又已证明买不到东西(压力测试 0 摔)+ eval-B 0%/CF 98% 与全批一致 → 建议 clip_switch 方向整体关闭**(除非部署轴另有理由,franco 拍板) |
| R12 | `task.rewards.base_decel_weight=1.0`(o3_R12_basedecel,02:45 点火) | 减速塑形方向有没有信号(v1 P 律;v2 拟合包络等 6 套采集)。**信号档跑完(→5499):composite 峰值 0.888@4654,同批最高、与产品线持平——composite 无税**;eval-B(峰值存档 it4700):回球 0%/CF 100%/拍面 43.3°(结构现状不变,如预期——它本来就不是治拍面的)。终判看部署轴(减速入位行为、base_speed_xy_prestrike),candidate 进保险包 |
| R13 | 赢家叠加进产品线 | 产品候选 |
| R14 | `"task.motion.speed_scale_range=[0.8,1.2]"`(o3_R14_retiming) | **变速重定时**:同一 clip 变速播放+速度需求同步缩放,策略能否学会按需调节挥拍速度幅值 = 无新数据的连续强度 v0(franco"加减速改幅度";空间幅度另由 R6 裁剪臂回答,两臂合看)。**信号档跑完(→5499):峰值 0.871@4843,比产品线低 ~0.02——变速出题更难,小幅回落属预期**;eval-B(it4800):回球 0%/CF 96%/拍面 48.5°(比其他臂散 ~5°,变速让拍面更飘——判读时的一个减分项) |
| R15 | v5 clip 臂:`motion_file=/workspace/shared/motions/hope_forehand_v5.npz motion_file_2=.../hope_backhand_v5.npz` + 下方 v5 专属采样框(相位 [0.673,0.345])(o3_R15_v5clips) | 新动作源(短条、首尾贴 ready);⚠与 hopex 差三个因子(数据源+clip 长度+参考噪声),与 R6 对照拆长度因子。**⚠ 发射异常(2026-07-05 06:57):看门狗把它发到了 DeployParity 任务、从零 2000 步(峰值 0.355,从零短跑正常水平)——与同批 vb 臂不可比,run 在 `agibot_a3_hope_deploy_parity/…o3_R15_v5clips`。按计划判读需在 vb 线重发(等 franco 定)** |
| R16 | `task.rewards.free_wrist_ori_mimic=true`(**已点火 2026-07-05,o3_R16_freewrist,franco 最高优先**) | **手腕解除姿态模仿**:把 right_wrist_yaw_Link 从 motion_body_ori/ang_vel 列表拿掉(位置/线速度模仿保留=挥拍路径照学)。理由:视频管线的手腕朝向不可靠(GVHMR),模仿它给拍面质量封顶。**在 vb 产品线上跑才有真学习信号**(落点/消旋奖励直接塑形拍面;纯 DeployParity 上只是去噪)。契约日它是法线指令通道的执行机构。**信号档跑完(→5499)+ eval-B 终判(2026-07-05):峰值 composite 0.870;真球考试拍面-vs-应有误差 43.5°(基线 P2 42.5°,纹丝不动)、回球率 0%、反事实 98% —— 手腕放开单独掰不动拍面**:落点/消旋奖励的间接梯度在 5.5k 步内没把拍面拉向可回球朝向。判定:**单独不进配方;它的正确位置是契约日的执行机构**(有了法线指令+奖励参照改跟指令,拍面才有直接梯度可爬) |

R15 v5 专属采样框(**franco 纠错后 2026-07-04 深夜版**;从击球帧提取,pos ±0.10 / vel(clean) ±0.50):
`"task.racket.strike_phase_per_clip=[0.673,0.345]"`(正手相位从 0.768 改钉 0.673!)
`"task.racket.pos_range_per_clip.forehand.x=[0.29,0.49]" ...y=[-0.63,-0.43] ...z=[0.74,0.94]`
`"task.racket.pos_range_per_clip.backhand.x=[0.60,0.80]" ...y=[0.12,0.32] ...z=[0.81,1.01]`
`"task.racket.vel_range_per_clip.forehand.x=[0.74,1.74]" ...y=[0.71,1.71] ...z=[1.20,2.20]`
`"task.racket.vel_range_per_clip.backhand.x=[2.60,3.60]" ...y=[0.51,1.51] ...z=[1.66,2.66]`

~~⚠ v5 正手 +Y 红旗~~ **已解除(franco 定位对了 2026-07-04 深夜)**:所谓 +Y 主导是**击球
相位钉错**——检测器选了拍速峰 0.768,那是触球后的甩鞭段;逐帧核查后真实触球 ≈ 0.67-0.69
(franco 给的"约 2/3 处"),该处速度 (+1.24,+1.21,+1.70)、法线 (+0.47,+0.84,−0.27),
方向健康。反手检测 0.345 与 franco 的"前 3/7 内"吻合,不改。教训已写入:**拍速峰 ≠ 触球,
选帧要用"前向速度峰/人工先验"交叉验证**(analyze_strike_phase 的已知坑)。
残留的真问题:**v5 参考噪声大**——mean|关节加速度| 5.9/15.5 rad/s²(正/反手),是 hopex
(2.5/2.7)的 2-6 倍,斜录(3.5/3.5)介于其间。短条+快挥拍给 GVHMR 的平滑上下文更少。
这是 R16(手腕解除姿态模仿)与参考滤波的直接论据;R15 判读时把"噪声"当第三个混杂因子。

**合并测试策略(找最优组合,不做全因子):**

1. **按裁决轴分组**。性能类(数据/奖励:R3/R9 斜录、R6 裁剪、R14 变速、R15 v5、R12)
   看 Isaac composite,期待提升,单臂 vs 同批对照,赢家贪心进产品线;交互只对"赢家对"
   补一次 2×2。保险类(部署加固:R11 clip_switch、A1、A1v2)**本来就预期 composite 微降、
   部署轴受益**——判它们用 eval-B 回球率 + deploy-faithful,不用 composite 一刀切。
2. **保险类整包测**:R11+A1+A1v2 合成一臂"部署加固包";包整体 composite 降幅 <3% 且部署轴
   不劣 → 整包采纳,不逐个消融(2^n → 1);包失败才 leave-one-out 二分。
3. **数据源类互斥合并**:hopex(对照)/斜录/v5/裁剪 是同一问题的四个臂,天然同批横比,
   一批出结论。
4. 每次产品线换代跑一次换种子臂(R10 惯例)验稳健,防止贪心叠加吃噪声。

**CPU 任务(不占 GPU,可立即做):**

- ~~StrikeSpec **固定法线求解**变体(结构级发现路 A)→ eval-B 复测~~ **已做 2026-07-05**:
  `StrikeSpecPlanner.solve_fixed_normal`(现有 solve() 一字未动,+5 个新测试,16/16 过)+
  `mujoco_eval_onnx.py --venue-fixed-normal`。**答案 = 0%(物理不可行,不是没解出来)**——
  详见上方 🔴 结构级发现的路 A 判决;证据存 pod `/workspace/franco/cf_eval/`
  (scan_reachability.py + modeB_fixed9600.log + verify_solver.out——网格可行点喂回求解器,
  内点可解 2.8 m/s/3 迭代/4.7 mm,判决不依赖求解器召回:合法框内无任何可行落点)。
- ~~eval-B 反事实 flag 化~~ **已固化 2026-07-05**:venue 模式每次击球自动多评一次
  "换上应有拍面朝向"(达成位置/速度不变),strikes CSV 追加 6 个 cf\_\* 列(原 14 列字节
  不变),汇总块报 CF 回球率。正式复现(P2 产品线,9600 步 seed 0,44 次击球):
  **实际 0/44,反事实 44/44,CF 落点中位差 0.10 m**——"拍面朝向是唯一短板"从手工分析
  升级为每次评估自动输出的常驻证据(当日 2400 步存档逐字节复现,前 43 列与 07-04 一致)。
- eval-B v2:截断 MVN 相关采样;机器人视角接触高度分布(等采集)。
- **门禁补 q_des 限位剪切(2026-07-05 新立,jiayi 部署发现的训练/部署不对齐)**:部署端
  (pp_joint_limits.hpp)发布前把每个关节的目标角硬剪到物理范围,训练(clip_actions=null)和
  MuJoCo 门禁都不剪——策略靠"超范围目标角"多要力的那部分,真机拿不到。**P2 暴露面已量
  (2026-07-05 探针,闭环加剪/不加剪各跑 4800 步)**:超限步数 10-18%,**全部在腰/踝等平衡
  关节(手臂 0 超限)**,最狠瞬间被剪掉该关节力矩上限的 35-56%;但闭环剪着打,击球三合格率
  1.000 不变、0 摔,拍速误差只 +0.02-0.03 m/s → **对 P2 产品线,仿真里看不出实质伤害;残余
  风险在真机平衡临界时刻(踝力矩恰好在最需要时被剪)**。待办:①门禁加同款剪切 flag(探针已
  验证一行 clip 即可,mirror pp_joint_limits);②训练侧剪切=jiayi 修复中(他的谱系上观察到
  的现象,各模型暴露面不同,他的存档修好后拿探针复量)。
- ~~**换招压力测试 eval 变体**(R11 收益的量尺,2026-07-05 立项)~~ **已建成并量完 2026-07-05**:
  `mujoco_eval_onnx.py --switch-stress p`(默认关,关=字节不变;训练侧 commands.py clip_switch
  同语义:均匀换模板、跳回起手帧、重新等球+换目标、机器人不动;开着时模仿守卫改成只看真摔)。
  量的结果见上方 R11 行:**两存档 × 两 PD 口径 × 三剂量全部 0 摔,收益轴饱和,税轴坐实**。
  R11b 跑完后一条命令补量(`--switch-stress 0.01 --pd-mode explicit --keep-passive`,
  先 standalone_onnx_export 带 --bake-obs-norm 导出;R11 的导出已放
  `o2_R11_clipswitch/exported_deploy_normbaked/`)。原始 12 组日志:pod
  `/workspace/franco/cf_eval/sw_*.log`。

## 计划未做全量清单(2026-07-04 franco 抓漏后建立;常驻,每日对账)

**P2.4 集群(本次漏账主体,franco 抓出):**
| 项 | 状态 | 归属/时机 |
| --- | --- | --- |
| P2.0 准备动作定义 | **franco 改判 2026-07-04 晚:不再专门拍 ready 视频**。v5 两条 clip 首尾均贴准备姿态(量化:fh/bh 起始帧互差 mean 0.15 rad——可直接当共享 ready 锚;首尾差 mean 0.24-0.27 rad,残差交给 RL 填充,clip_switch/post_swing/hold 机制都在)。ready 参考帧从 v5 clip 首帧提取,零采集成本 | claude(提取+接线 stand_start/hold) |
| PACE 减速命令(伪速度∝剩余误差→平滑减速入站位) | **v1 已实现 2026-07-04**(`rewards.base_decel_weight`,默认关,机制验证过)→ 信号档臂 R12;v2 = 拟合加减速包络+方向+时间预算+幅度耦合([motion_and_contract_v3.md](motion_and_contract_v3.md) §5,等 6 套采集) | claude |
| ready→strike→ready 拼接 | 未做(依赖 P2.0) | P2.0 后 |
| base-target 回归 ablation(HITTER 位置命令,论文背书) | 未做 | 通宵臂后的下一轮 |

**其余未做(按可行动性排序):**
| 项 | 状态 | 阻塞物 |
| --- | --- | --- |
| A4 后半:真机数据落盘制度 | 未建 | **今天场地就该全量落盘**(planner 日志/VRPN 流/视频) |
| A5 挥拍视频 30-50 条 | 主动推迟 | 场地顺手拍几条即赚 |
| eval B 模式(来球分布驱动) | **已交付 2026-07-04**(`--target-source venue-balls`;头条 = 法线 clip 锁死,见上方结构级发现;G06 有正式记录) | — |
| Ace 饱和 Magnus 形式接进 flight/virtual_ball | yaml 有键未消费 | 小活,高旋外推保护 |
| A6 摔倒管理(guard 包络、recover 行为) | 未做 | 下周 |
| A7 击球事件自动检测 | 未做 | 依赖 A4 落盘 |
| A2 坐标变换设计 | 未做 | yikang 下周 |
| A3 执行器辨识 | 未做 | 需专门真机时段 |
| P2.5-full 真球进 Isaac | 未做 | 下周主菜 |
| P2.6 扣杀 | 未动 | 长线 |
| Queued#6 G06 验收数字 | 半吊 | 新候选认证流程替代中 |
| Queued#2 clip 转正 | **其实已完成**(hopex=转正版),标记关闭 | — |

## Gap List 状态扫描(2026-07-04 中午重扫;原表保留在下方)

| # | 状态 |
| --- | --- |
| 1 终审+候选 | 候选=P2 线(0.884)已定;**双仿真认证仍开放**(eval 契约修复在飞,P0)|
| 2 explicit-PD 微调腿 | 作废(IdealPD 已否决;记分板显式门禁替代)|
| 3 门禁+MDU 打包 | **最大的真缺口**。认证分工定案(jiayi 提醒 2026-07-04):**官方门禁 = agi/A3_MuJoCo_Sim 厂商仿真**(部署保真,yikang 的门);wbt 的 mujoco_eval = 我们的指标工具(修复继续,但只服务指标)。门禁三层,claude 全接(franco 2026-07-04):L1 平价校验(pingpong_parity,无 ROS/AimRT,pod 可跑)→ L2 deploy-faithful 指标(修复后的 wbt eval × 厂商 MJCF)→ L3 完整 AimRT 仿真(需 distrobox 环境,pod 上立环境为后备,诚实标注工作量)。链条:P2 终版 ONNX(今晚自动)→ L1+L2 → MDU 打包 + 部署配置改指 |
| 4 部署配置悬空引用 | 仍未改指,打包时一并(依赖 #3)|
| 5 物理 v1 | ✅ 完成(yikang 场馆拟合已 merge)|
| 6 延迟/误差标定 | 噪声谱 ✅ 已接进训练;**延迟已有界**(franco rig ground truth 2026-07-04:传输稳定 <10ms;动捕处理本身支撑 300Hz 输出 → 端到端 ≈≤20ms ≈ ≤1 个 50Hz 策略步)→ 训练用 delay_steps=2(40ms)是保守上界,保留;时间戳精测降级为机会项 |
| 7 0703 clip 覆盖 | **franco/jiayi 团队接走**(2026-07-04);备胎:pod 上 GVHMR/GMR 管线已通,给原始视频即可出 clip |
| 8 算力 trade-off | ✅ 完成(手册封版)|
| 9 球进训练 | 虚拟球已实质替代(奖励层);真球入 sim 仍下周 |
| 10 A2 桥(yikang)| 下周 |

另扫出的未跑测试:R6 裁剪 clip 臂(工具/clip 齐,排 GPU0 空槽);HER 隔离 A/B(已被 P2
胜线包含,降级为学术项,正式放弃);KF 真流验证 + StrikeSpec 部署接线 + 机器人挥拍标定
= **下次场地日清单**。

## Gap List To Sunday (明确缺的活,截止周日 — added 2026-07-03)

| # | 缺什么 | 谁 | 何时 |
| --- | --- | --- | --- |
| 1 | 两条 14k 长跑的终审 + 选周六候选 | claude | 今天 18:15-19:00 |
| 2 | explicit-PD 微调腿(过 MuJoCo 硬门禁的配方) | claude | 今天 19:00-24:00 |
| 3 | MuJoCo 双门禁 + MDU 打包 | claude 备 / yikang 运 | 今晚-周六早 |
| 4 | ~~model_9000 同板对比~~ RESOLVED(jiayi:9000 为测试产物不提名;规则收窄为「提名上真机时才必须交 ONNX 过记分板」,训练期各用各的、以 Isaac 指标互比;main 部署配置的 model_9000 悬空引用今晚打包时改指获胜候选) | claude | 今晚 |
| 5 | 物理模型 v1:用今天采的球轨迹拟合 drag/bounce(→ planner 参数;训练侧下周) | yikang | 数据到即做,周六 planner 可用 |
| 6 | **延迟/误差标定**:从动捕录制的时间戳量真实延迟与噪声谱 → 填 A1 各 flag 的数值(franco 指出:这些本就该从物理建模数据算出,不拍脑袋)。顺带确认真实帧率(现有 300 与 320 两种说法,时间戳一算便知) | yikang(数据)+ claude(分析脚本) | 周六-周日 |
| 7 | 0703 打球录像 → 旋转归一 → 新参考 clip 验证(jiayi 的 re-ground 管线已做一版,确认覆盖) | jiayi | 周六前 |
| 8 | 训练速度 vs 并行数的 trade-off 终版报告(见下,搜索范围 4096/8192/16384 + 共卡) | claude | 今天 |
| 9 | 球进训练环境 + 落点奖励(P2.5-lite)— **周六前不可行,诚实排下周**;周六的增益来自策略改进+planner 物理参数,不来自训练内球 | claude/dongc1 | 下周 |
| 10 | mocap→runner 桥 + 坐标变换设计(A2) | yikang | 下周 |

## ~~Tonight's Test Slots (2026-07-03)~~ 已过期,由「关键更新联合消融」与通宵舰队替代

After the 18:15 finish + verdict, the freed slots run signal-tier (2000-it ≈ 1.5 h co-run) A/Bs.
"Winner" = tonight's better 14k run. All arms resume FROM the winner checkpoint:

| Slot | Run | 目的 |
| --- | --- | --- |
| GPU1-a | winner + 2000 it, plain (shared CONTROL) | 两组 A/B 的公共对照 |
| GPU1-b | winner + 2000 it, jiayi 默认捆绑（HER 30% + hold_ready 2.0,合并后已是 yaml 默认) | 验证新默认在我们谱系上不劣化——默认值影响所有人,必须有对照数字 |
| GPU2-a | winner + 2000 it, A1 延迟包（delay=2, jitter, 2% 中途更新) | 目标延迟训练在信号档稳定且不伤命中率(为 mocap 闭环) |
| GPU2-b | (备用) 若 backhand 在 deploy-faithful 明显弱于 forehand → backhand 加权微调 | 周六反手 SHADOW 测试的胜算 |
| GPU0-a/b | 留给 yikang 重启他的 20k(建议挂 adaptive_sigma)+ jiayi 任意 | 团队槽位 |

已关闭、不再测的问题:num_envs 扩缩(4096 定版)、一卡并行数(今晚 3 任务探针出数后定版)、显式执行器路线(团队已否决)。

## Active

| Item | Priority | Owner | Branch | Status / next checkpoint |
| --- | --- | --- | --- | --- |
| P2.1/P2.3/A8 ablation ladder (arms A-E) | ★★★ | claude | `p2-multiswing` | 2000-it round DONE (Isaac composite: A 0.42 / B-teleport 0.79 / **C+sigma 0.62** / D+postswing 0.40; adaptive sigma = +48% over A at equal budget). All arms too immature to survive MuJoCo deploy protocol (fall ~1.2 s from stand) — overnight: GPU1 A-ext→12k (control), GPU2 **E-ext = C-resume + post_swing + sigma →12k (product candidate)**; B-ext dropped (hardware already proved teleport-era fails stand-entry). 4-protocol scoreboard verdict when done |
| P2.3: adaptive tracking sigma (SMASH) | ★★★ | claude | `p2-multiswing` (flag `racket.adaptive_sigma`) | IMPLEMENTED + mech-verified 2026-07-03 (sigma live-updates within clamps); next: arm C after P2.1 A/B |
| A8: post-swing initial-state buffer (Ace) | ★★★ | claude | `p2-multiswing` (flag `motion.post_swing_start_prob`) | IMPLEMENTED + mech-verified 2026-07-03; next: arm D after P2.1 A/B |
| P2.0: ready-pose definition (see G08) | ★★ (foundation) | franco (拍摄) + claude (pipeline) | — | DECIDED 2026-07-03: option (a) — record a ready-stance video through GVHMR→GMR on the next site visit (bundle with A5's 30-50 new swing clips); claude processes + wires into stand_start/hold/clip re-entry |
| Legacy-task long run `merged_uniform_hopex` (20000 it, task=HOPEPingPong on own branch) | ? | yikang | `rsi-on-wrap-progress-fix` | RUNNING on pod GPU0. ⚠ branch duplicates main's wrap_teleport machinery and has LFS-pointerized CSVs — reconcile with main before merging; the unique progress-fix is already ported to `p2-multiswing` |
| simtoreal2 lineage: HER achieved-replay + hold_ready + model_9000_replane training (merged to main 2026-07-03) | ★★★ | dongc1 | `simtoreal2` (merged) | model_9000 backhand passed training gate; **needs**: drop `model_9000_replane.onnx` into `/workspace/shared/models/` so the scoreboard can grade it against tonight's candidates |
| Sim2real deploy bridge: `agibot_hardware_bridge` ROS pkg (`bridge_node`) + `world_frame` world→robot target transform (+`test_world_frame`) + `hope_pingpong_sim2real` launch + `wbc_runner` rebuild for the planner-driven deploy path; `HOPEPingPongDeployParity` hyperparameter tuning | ★★★ | dongc1 | `simtoreal2` | IMPLEMENTED 2026-07-03 (`a5016d43`) with `run_sim2real_bridge.md` ops doc; this is the concrete build of the G07 world→robot transform (row below). Next: wire mocap relay → planner → runner end-to-end on hardware |
| G07 mocap→runner bridge + world→robot target transform design (A2) | ★★ | dongc1 (impl started; see row above) | `simtoreal2` | `world_frame.py` + `test_world_frame` landed 2026-07-03; design doc still TODO — see G07 Next Steps and G08 audit item 2 |

## Queued (priority order, from G08)

1. A1 目标延迟/抖动/中途更新 — **MERGED to main** 2026-07-03 (flags `racket.target_delay_steps` / `target_jitter_pos_per_s` / `target_jitter_vel_per_s` / `midswing_resample_prob`, default off; mech-verified: delay=2 in effect, redraw rate 0.0201≈p=0.02). Next: stack on tonight's winner as the mocap-loop rehearsal arm.
2. P2.2-lite: orientation-normalize the existing two clips at retarget (`reground_hope_frame.py`).
3. A5: record 30-50 new reference swing videos (needs a human + camera; processing pipeline ready).
4. P2.5-lite: ball + drag/bounce + PACE at-contact landing reward (independent track; big).
5. A3: per-joint actuator ID on the real A3 (needs hardware time).
6. G06 acceptance numbers for the shipped checkpoint (needs `model_p4_deployparity.onnx` copied to
   `/workspace/shared/models/` or dongc1's machine).

## Done

| Item | Owner | Landed | Where |
| --- | --- | --- | --- |
| Standardize VRPN rigid-body object names (`P1`/`P2`/`Ball`) in the mocap relay config | dongc1 | 2026-07-03 | `simtoreal2` (`ac79a17a`) |
| wandb registry: fetch only the newest `motion.npz` | dongc1 | 2026-07-03 | `simtoreal2` (`3c9bc0cf`) |
| Merge `origin/main` (49 commits) into `simtoreal2` (local-first, doc+code) | dongc1 | 2026-07-03 | `simtoreal2` |
| Fixed-protocol sim2sim scoreboard (`scoreboard_eval.py`), validated end-to-end on pod | claude | 2026-07-03 | `p2-eval-harness` |
| 4 main-breaking merge casualties fixed (conflict markers; `motion_file` regression; `episode_time_left` probe crash; `play.py` `_wbt_tasks`) | claude | 2026-07-03 | main / `p2-multiswing` |
| `motion:` task-YAML/CLI plumbing for wrap_teleport / stand_start / hold | claude | 2026-07-03 | `p2-multiswing` |
| racket_progress exact-zero on resample (ported from yikang's `c7733db`) | yikang→claude | 2026-07-03 | `p2-multiswing` |
| RunPod multiuser provisioning + smoke suite | yikang (+team) | 2026-07-02 | pod `/workspace` |
| Doc realignment to simtoreal2 reality; Phase 2 roadmap into G08; papers | claude | 2026-07-03 | main |
| First sim-to-real (forehand-only, `model_p4_deployparity.onnx`) | yikang/dongc1 | 2026-07-02 | main |

## Update Rule

Update your row when: you start/finish an item, change branch, hit a blocker, or hand something
off. Keep rows one line; details go in the gate doc or PR description.
