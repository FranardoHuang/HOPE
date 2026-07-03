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
| P2.4b 减速命令/base 回归/ready pose | 未(ready pose 已定方案) | — | — | — | — | — |
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

## 基建现状(2026-07-04 凌晨)

- **pod 宿主机对 Isaac 判死**:裸 Kit 在重建容器上仍挂死;Stop→Start 不换宿主机(卷钉死机器)。
  训练全面阻塞。选项:RunPod 支持重置宿主机 / 新建 pod 迁移(venv 可重装,数据 ~20GB 可 rsync)。
- **导出与评分已绕开 Isaac**:`standalone_onnx_export.py`(纯 CPU 重建 actor+归一化器+动作缓冲,
  元数据从同配置旧 ONNX 拷贝)——两个 14k checkpoint 已导出成功,四协议评分进行中(纯 CPU)。
  周六候选的裁决不再依赖 Isaac。

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

## Tonight's Test Slots (2026-07-03 evening, policy: 4096 envs × 2 jobs/GPU, stagger ≥60 s)

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
