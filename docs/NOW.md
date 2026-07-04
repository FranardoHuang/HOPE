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
| jiayi | End-to-end training bring-up; reward tuning — the simtoreal2 lineage (HER achieved-replay, hold_ready, model_9000 training) |
| yikang | Deployment; sim/env alignment |
| claude (franco's agent) | Foundation: infrastructure, A/B experiments, doc/code hygiene |

## Run-Name Legend (人话对照表 — 报告里不再用裸字母)

| run_name | 人话 |
| --- | --- |
| p21_A_noteleport / p21_A_ext12k | 「基线结构」：挥拍间不传送 + 25% 从站姿开始（main 默认配置），2k / 续跑到 14k |
| p21_B_teleport | 「旧传送式」对照：每拍开始把机器人瞬移到位（部署模型的旧训法），仅 2k，已停 |
| p21_C_sigma | 基线结构 + 「奖励自动收紧」（误差变小奖励口径跟着变严），2k |
| p21_D_postswing | 基线结构 + 「上拍收尾姿态起手」（一部分回合从上一拍打完的姿势开始练），2k |
| p21_E_sigma_postswing | 「三合一」＝基线结构＋奖励自动收紧＋收尾姿态起手，从 C 续跑到 14k —— **当前最优候选** |

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
- **B 长期(下一代训练)**:观测契约 175→178(+目标法线 3 维),训练 racket_normal 奖励改跟
  指令法线,部署 pp_obs_builder 同步——一次契约变更,换来真正的"按需拍面"。需要全队排期。

(附带发现:场馆来球接触点在人的击球高度 0.98-1.26m,高于训练框——采集分布是"人接球"
视角,机器人版分布尚缺;B 模式 v1 盒采样未强制相关性,升级路径已注明。)

→ 设计已固化并按代码逐条核实:[motion_and_contract_v3.md](motion_and_contract_v3.md)
(2026-07-04 晚,claude)。核实中的新发现:①`racket_target_vel_w` 是世界系裸直通——现役
契约里只有位置做自我中心化,速度不做(真机 identity-yaw 下退化一致);②部署线协议
`RacketCommand.normal` **已经在传法线**,缺口只在 obs 坑位与训练奖励参照;③critic 已有
特权 `racket_target_normal_w`,契约日训练侧只是把 racket_normal 奖励参照从 clip 换成指令。
迁移定为 **175→179**(法线 3 + ρ 1,可顺手把 vel 也改自我中心)。

## 快速横向对比制度 + 新消融臂(2026-07-04 晚;franco:想法变多了,不再默认训久)

**制度(新想法一律走梯子,不再单独长训):**

1. 机制检查(512 envs × 25 it,~3 min):开关生效、无崩。
2. **信号档 2000 it @ 4096 envs**(独占 ~1.2h / 共卡 ~1.5h;3 卡 × 2 槽 = 6 臂/批,
   错峰 ≥60 s):同种子、同热启(当前 P2 产品线 ckpt),**每批必带同批对照臂**。
   裁决口径 = Isaac composite(2000 步,同 10:00 判读)+ eval-B 回球率(CPU,不占卡)双轴。
3. 只有信号档赢家进组合臂;组合赢家才升 12k(~7h)与门禁。输家记下数字后关闭。

**新臂(接 R 系列;R0-R10 已用):**

| 臂 | 配置(叠在当批对照/胜者上) | 回答什么 |
| --- | --- | --- |
| R11 | `task.motion.clip_switch_prob=0.002` | 中途换 clip 平价能否救场馆切换摔倒,命中率代价多少 |
| R12 | `task.rewards.base_decel_weight=1.0` | 减速塑形方向有没有信号(v1 P 律;v2 拟合包络等 6 套采集) |
| R13 | R11/R12 赢家叠加进产品线 | 产品候选 |

**CPU 任务(不占 GPU,可立即做):**

- StrikeSpec **固定法线求解**变体(结构级发现路 A)→ eval-B 复测:法线钉在 clip 参考值、
  只解 (v_n, v_t) 时能换回多少合法回球率 = 现役策略 + 适配 planner 的真实上限。
- eval-B 反事实 flag 化(25/25 那次是未提交的手工分析,加输出列使其可复现)。
- eval-B v2:截断 MVN 相关采样;机器人视角接触高度分布(等采集)。

## 计划未做全量清单(2026-07-04 franco 抓漏后建立;常驻,每日对账)

**P2.4 集群(本次漏账主体,franco 抓出):**
| 项 | 状态 | 归属/时机 |
| --- | --- | --- |
| P2.0 准备动作定义(视频→GVHMR,方案已拍板) | **未做——但今天在场地拍 2 分钟就有原料**,管线已在 pod 跑通 | franco 场地拍,claude 处理 |
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
| 9 | 球进训练环境 + 落点奖励(P2.5-lite)— **周六前不可行,诚实排下周**;周六的增益来自策略改进+planner 物理参数,不来自训练内球 | claude/jiayi | 下周 |
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
| simtoreal2 lineage: HER achieved-replay + hold_ready + model_9000_replane training (merged to main 2026-07-03) | ★★★ | jiayi | `simtoreal2` (merged) | model_9000 backhand passed training gate; **needs**: drop `model_9000_replane.onnx` into `/workspace/shared/models/` so the scoreboard can grade it against tonight's candidates |
| G07 mocap→runner bridge + world→robot target transform design (A2) | ★★ | unassigned (natural fit: yikang) | — | design doc first; see G07 Next Steps and G08 audit item 2 |

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
