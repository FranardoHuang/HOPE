# 简短进度记录

本文件只保留短日期摘要，不再做第三份实验真源。更新时只写几句话并链接到权威位置：

- 当前 setting/采用状态：[NOW](NOW.md)
- 实验设计与证据：[experiments/](experiments/README.md)
- `main` 上的重要变化：[TIMELINE](TIMELINE.md)
- 可复现验收：[gates/](gates/)
- 缩写与人话释义：[DEFINITIONS](DEFINITIONS.md)

旧 1700 行记录完整保存在
[历史 PROGRESS](experiments/archive/PROGRESS_legacy_through_2026-07-12.md)。

## 2026-07-15

- S0/M0 exact-GMR attempt-v2 在 Pod2 clean detached `b75204d` 上再次证明 source/static 合同正常：两份
  `static-v2` PASS、plan SHA exact，两个 `exact_gmr_v2` root 与 shared consume lock 执行前后均 absent。
  runtime `inspect` 在进入 consumer 前以 rc127 fail closed，因为合同绑定的
  `/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10` 连父环境都不存在。没有 GMR 输出，
  不是动作/脚距失败；两批均不得 consume，正在只读盘点 Pod2 可重建 runtime。见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)。

- clean base-decel 两份 `model_200` receipt 已闭合：checkpoint SHA-256 `6cb55718...94f1` /
  `d61998ac...6892`，76 tensors / 1,762,715 floats 全 finite、fresh lineage/claim/common schema-3 hard
  exact。step 0–200 两臂五个 post-swing counter 每点全零，raw base-decel 两边逐点为正且 weighted
  Reward 只在 treatment 非零。但 180–200 冻结窗底座速度 treatment/control=`0.75008/0.71340`
  （`1.05142×`），+200 `≤1.00×` 方向门失败；四项精度和解析回球均为零对零、pre-fall 约 100%，
  不能写成行为非劣。按预注册继续到 +500 只判晚熟，不买 seed/judge/晋级。详见
  [clean main-effect 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md)。

- B 的首次 vendor L1 CPU `dry-run` 在轨迹审计前因 private-name grounding helper 不能由 `sys.path`
  导入而 fail closed；没有 certificate，不是动作安全失败。harness 已改为按冻结 bytes/SHA 从 exact path
  事务式加载，执行前后复核 module origin，异常时恢复/清除 `sys.modules`，真实 helper alias 与
  SHA/stale/body-failure 负例通过。等待合入后 clean runtime 重跑，G08 保持 Partial。详见
  [B vendor L1 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)。

- clean base-decel 单变量 pair 已按同一次顺序事务在 Pod2 GPU1/GPU2 越过首迭代：control/treatment
  exact PID=PGID `385320/385948`，claim SHA-256 `a039226a...1746e` / `673bf6c6...9392`；GPU0 的 Yikang
  进程未触碰。04:27 CST 只读复核时 TensorBoard 到 step `106/89`，日志 fatal=0，两臂五个
  post-swing 计数在全部已写 update 严格为零；raw base-decel 两边均激活，weighted Reward 只在
  treatment 非零。尚未到 `model_200`，不比较行为、不买第二 seed。详见
  [clean main-effect 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md)。

- 新的 clean base-decel main-effect 已在结果前预注册：两臂固定 V1+V2、seed3、4096×1001 与同
  action/bank/plant，post-swing 明确关闭且五个 replay 计数必须逐 update 全零；唯一差异为 base-decel
  `0/1`。它不复用失败 pair 的行为，只复用 exact `2c2d70d...` 已通过的 source/scene boot 门；新 job/run
  namespace 硬绑 Pod2 GPU1/GPU2。详见
  [实验卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md)。

- fresh v4 两份 `model_500` receipt 已闭合（checkpoint SHA-256 `22f78f88...a6a` / `a1735fbb...c14`，
  finite/lineage/claim/hard exact）。但 control 的冻结 `480–500` 窗 post-swing 分母仍为零，treatment 已按
  24.86% 激活；control 到 step519 才 ready，不能倒灌。根因是 buffer 只收自然 clip-wrap 存活状态，
  base-decel 会内生改变共同 curriculum 的 cold-start 时刻。pair 按 `activation-invalid` 精确收口于日志
  `564/573`，不比较行为、不买 seed；下一版改用共享 immutable natural-wrap teacher receipt。
  详见 [replacement 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)。

- fresh v4 control/treatment 的 `model_200` 已分别发布 exact receipt：checkpoint SHA-256
  `d065441b...c77b` / `e1d2b43f...4fb7`，两边 filename=embedded `200`、76 tensors、1,762,715 floats
  finite、fresh lineage 与 hard contract 一致。V1 和 base-decel activation 闭合，V2 在有样本处相等；
  但 post-swing 两臂到 +200 的 eligible/selected/started 全为零，明确违反预注册正分母门。因此 +200
  当时记 `invalid/instrumentation-blocked`，不比较行为、不买 seed；随后 +500 的终局结论见本节首条。
  详见 [replacement 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)。

- exact source/config 审计纠正了现役 Reward 的错误表述：`task=HOPEPingPongVirtualBall` 同时保留
  目标位置/速度/拍面 `14/10/5` 与 achieved-state 解析过网/落点/旋转 `20/30/5`；
  `vb_metrics_only=true` 不会关闭 task 自带的 outcome Reward。真正 metrics-only 的是
  `physical_ball=true` Phase-A engine-integrated 诊断，当前又没有拍球冲量，所以没有真实物理回球
  Reward。解析过网与落点还会在完整合法回球前给稠密部分分；现役 pair 不改配方，未来先闭合
  Phase-B 物理 receiver，再做
  outcome-source 固定总预算单 seed 配对。详见
  [Reward 真值审计](experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md)。

- 完成 Jiayi V9 与 Yikang 部署支线的 exact-commit 只读审计：定向 recovery debt、二维 station settle、
  动作首帧上肢准备态、外生随机长等待及 per-side planner metadata 可作为 current-main 单变量候选；
  直接写 root velocity、旧 broad-kill harness 与 checkpoint 专用 soft clamp 不整体移植。旧三次 `7/7`
  的成功条件只是 engage→挥拍→恢复，结果明确未测 physical contact/landing，且只覆盖固定正手区，
  所以不能作为物理回球、球路泛化或选档成绩。详见
  [跨线审计](experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md)。

- fresh v4 measurement control/treatment 已由同一次 `fill --count 2` 顺序发射：Pod2 GPU1/GPU2 exact
  PGID=`380610/381237`，claim content SHA-256=`576724de...a49d` / `1a529430...4c5`，两臂均绑定 clean
  `2c2d70d...`、4096 env、schema-3 hard contract并越过首迭代。GPU0 的 Yikang trainer 保持原样；
  本节上方 `model_200` 条目已经取代这条启动快照。

- exact `2c2d70d...` 的唯一 4096-env full-scene probe 已在 Pod2 GPU1 完成两个 update 并自然退出；finalizer
  复核 actual env、物理球/桌三实体、face179、31/31 零摩擦、schema-3、76 tensors 全 finite、fatal0、
  source/asset closure 与空 PGID，result file SHA-256 `4b12854c...0b27`。queue 已显式消费 receipt，fresh v4
  control/treatment 变为 ready 且 `launch_authorized=true`；这不授权 judge、第二 seed或晋级。

- inference-counter 修复后的 replacement pair 已改绑 clean exact `2c2d70d...`，并换用从未发射的 `v4`
  namespace；control/treatment 由 main `8b0a084...` 分别 hard-bound 到 Pod2 GPU1/GPU2，Pod1 与一康 GPU0
  均不在发射路径。其 source/asset/strict probe 门现已闭合；两臂 ready，但本条记账时尚未创建科学 claim。

- 轻量训练 harness 新增 `required_slot` 硬绑定：目标 GPU 满载时本 job 不 fallback，同时不饿死其他槽的
  独立任务；与 `preferred_slot` 互斥，science claim、warmup、probe/finalizer 都在 SSH 前执行检查，防止
  Codex 作业落到一康保留的 GPU0。该字段不冒充 matched pair 原子性；replacement queue 已重绑但仍
  blocked，尚未 probe 或启动。

- Same-phase activation successor `0f3900a...` 的 4096-env Pod2 strict probe 抓到离线测试遗漏：
  RewardManager 在 `torch.inference_mode()` 内创建 ledger，normal-mode runner 第一次 `zero_()` 即 fatal；因此
  该 source/attempt 永久不解锁科学 pair。修复把私有 counter reset 放回 inference mode，并新增跨 mode 的
  create/consume/reuse 回归，专项 `10 + 2 + 11 passed`；尚待新 source 重绑、全新 probe 自然终档和显式
  receipt consumer，G05 仍 Partial。probe 前另以旧 inode 硬链接保全 + canonical atomic replace 解除了
  一康旧 launcher 的 lock-fd 代际泄漏，全程未触碰一康 GPU0 进程。

- Yikang 的 `ayzxv1ma/model_10600` 四臂矩阵已从功能分支 exact source `8c8cd53` 发射：Pod1
  三卡分别运行 A 泛化 [`5nso93g0`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/5nso93g0)、
  B 推扰 [`4osh4ypc`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/4osh4ypc)、A+B
  [`jndof7jk`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/jndof7jk)，Pod2 GPU0 运行 fresh A+B
  [`xpiapvix`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/xpiapvix)。四条 init/load 门与首个
  finite/loadable checkpoint 已过；B/AB mechanics 已实际施加推扰，fresh 短 smoke 因随机策略未活到
  5 秒只验证 selection，但正式 run 到约 iter 322 已记录第一次真实 apply。训练质量、
  matched-iteration 对照和 Gate3 仍未判。

- V1+V2×底座减速旧仪表 pair 已自然到 `model_1000` 并退出，不是中途失败。control/
  treatment 的 model SHA-256 为 `ad69bc70...9f75` / `dcfb9599...00e8`，两边 filename/embedded
  iter=`1000`、76 tensors / 1,762,715 浮点元素全 finite、fresh lineage=`1`、fatal0；
  no-clobber receipt 为 `8c0b3750...415d` / `050f2657...5f00`。终点 21 点的底座击球前
  速度 treatment/control=`0.15364/0.16714`，但旧 source 缺 activation denominator/numerator，
  仍判 instrumentation-blocked，不买第二 seed/不 judge/不晋级。见
  [interaction 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-INTERACTION.md)。

- Franco 反手拉 B 的 [`vendor L1 safety audit`](DEFINITIONS.md#motion-vendor-l1-safety) 已完成
  source-only 预注册/validator：绑定 exact L0 certificate、B schema-2 NPZ、vendor MJCF closure 与
  MuJoCo 3.10 runtime，复用既有 shortest-arc/linear 插值将 `151 @ 50 Hz` 有限密扫为
  `1201 @ 400 Hz`；自碰穿透或球拍/拍柄 `<5 mm` 自打均为不可补偿 hard fail。红队后续把 5 mm
  决策改为 exact saturation predicate（4.99/5.00/5.01 mm 反例闭环）、补齐右肩三轴/右肘，并令
  dry-run 在 runtime 前强制 parent 存在、target absent 且非 symlink；明确不声称连续时间。
  专项连同 L0 回归 `23 passed`；本任务没有连接 Pod、没有运行 runtime 或写证书，G08 仍 Partial，
  桌网/动力学/训练继续 blocked。见
  [L1 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)与
  [操作](operations/run_motion_backhand_loop_b_vendor_l1_safety.md)。

- 稀疏平衡失败的横向扰动消融已完成 source-only 红队修正版：`torso_link` 质心 WORLD-Y 有界脉冲按
  随机化后整机总质量缩放，`L0/L1` 用 domain-separated Philox4x32-10 共同随机题并暴露 potential draw/
  schedule SHA；episode reset 截断会记录 sampled/commanded/applied/abandoned 冲量且当步禁止重启。
  后续红队新增不可由配置放大的 `0.15 m/s` 冲量、`2.0 m/s²` 加速度、`0.02--0.20 s` 时长和 `200 N`
  WORLD-Y force 硬包络，并把 scheduler→adapter 改成 source-token 绑定的无副作用 preflight + 原子/no-throw
  commit：删除公开 acknowledgement，mass/cast/final wrench/receipt/cache 全在写前 host-visible 校验；坏
  receipt/stale token 不写 backend、不 cache、不解锁，同 step token 可换全新 preflight nonce 重试。commit 进入后异常或非
  `None` 返回会永久标成 `DIRTY/UNKNOWN`；cache 还绑定 live backend token，不能重放给同 SHA 新实例。
  strike/window 现在和 reset 一样逐环境保存 sampled/commanded/applied/abandoned 恒等式并在中断 tick 真正
  写零。源码专项增至 `36 passed`。torso COM 仅意味着 zero explicit/link-local lever-arm torque，不代表
  整机无 `r×F` 角冲量。
  GPU throughput 门和不可变 ball×action-family held-out paper 仍 pending，故继续
  `launch_authorized=false`，未连接 Pod。见
  [实验卷宗](experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md)与
  [G05](gates/G05_isaac_training_first_loop.md)。最新 `origin/main@107102f` 整合重放为
  `847 passed, 22 skipped, 3 failed`；三项失败均在未改动路径且已在 main 原样复现，不是本分支新增回归。

- Franco 反手拉 B 的 L0 V1 portable dry-run 已登记为数值合同负结果，而非动作失败：schema-2 只存
  post-FK normalized float32 root body pose，V1 再把它当原 free-joint qpos 注入并要求 byte equality；
  position/quaternion/COM velocity/angular velocity 最大差分别为 `1.1920929e-7 / 5.9604645e-8 /
  2.9802322e-6 / 5.9679151e-6`，未写证书。新 V2 冻结 V1 原字节及全部 lineage、joint/ground/support/
  safety 门，只对不可重构 pose 使用 two-ULP + physical cap、对 COM velocity 使用 exact `body_ipos` 与
  50 Hz 误差传播，angular/joint velocity 仍 byte exact；两份专项 `29 passed`。本任务没有连接 Pod、
  没有运行 V2 runtime/audit，G08 继续 Partial。见
  [L0 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)与
  [操作](operations/run_motion_backhand_loop_b_l0_static.md)。

## 2026-07-14

- 反手拉 B 的 V2 L0 已在 Pod2 exact detached `main@cc1a2b1` 闭环：full `dry-run` 通过后，
  独立只读复核 source/plan/validator/四输入与证书 absence；随后唯一 `O_EXCL` formal audit
  发布 certificate SHA-256 `60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`。
  certificate 仅令 `l0_static_complete/vendor_l1_authorized=true`，桌网、动力学、训练、formal motion 和
  hardware 仍 false；下一门是 vendor L1 自碰/球拍自打。见
  [L0 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)与
  [G08](gates/G08_blind_spot_improvements.md)。

- Pod2 四条科学臂的 paired `model_200` 身份和 step `180..200` 曲线已冻结。conditional
  treatment 的 gate/cost/reward 全零，可严格推出 eligibility=0，故当前 `-0.4` setting
  在 `+200` 判 activation-invalid，不买 seed/不晋级。V1+V2×base-decel 的 checkpoint SHA-256 为
  `44a709ac...035a` / `b04e2338...e56b`，receipt `ad47c826...4d1f` / `49234348...7748`；
  V1/V2/base-decel 的 count-level denominator/numerator 不完整，只记 instrumentation-blocked，不写成
  Reward 负结果。见 [conditional 卷宗](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)与
  [interaction 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-INTERACTION.md)。

- Post-swing replay 的真实 reset 路径已新增 buffer-not-ready、eligible、random-not-selected、
  selected 与 started 五组 per-update 整数计数。后续 exact successor `0f3900a...` 又补齐
  V1/V2 仪表和两臂同 RewardManager phase 的 base-decel raw kernel 计数：独立红队确认
  probe 返回严格零、参数同源、treatment 同 step 去重，聚焦套件 `222 passed`，另有两个
  未被本改动触及的 main `MotionLoader/PosixPath` 基线失败。新 post-swing pair 与
  base-decel measurement-complete replacement 队列仍保持 `launch_authorized=false` / `blocked`；
  exact source ignored-asset hydration 与 strict full-scene terminal probe 尚缺，不得据源码门点火。见
  [post-swing 卷宗](experiments/2026-07/EXP-P1-V1V2-POST-SWING-INTERACTION.md)与
  [base-decel replacement](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)。

- 反手拉 B 的 portable full L0 `dry-run` 已在 Pod2 CPU 真实执行，旧 v1 合同在跨节点
  float32 逐 bit 重算门 fail closed：position/quaternion/COM velocity/angular velocity 最大差为
  `1.1920929e-7` / `5.9604645e-8` / `2.9802322e-6` / `5.9679151e-6`，证书仍 absent。
  诊断绕过只用于定位、不是 formal pass；旧失败保留，v2 从 float32 ULP 与 50 Hz 差分误差
  独立推导，不改关节/地面/支撑脚/安全门。见
  [L0 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)。

- conditional P1 control/treatment 的 `model_200.pt` 已由 source-pinned attestor 写入 no-clobber receipt：
  checkpoint SHA-256 `b55b7d3b...b4b41` / `c07b1f12...bd51`，各 76 tensors、1,762,715 浮点元素全 finite，
  fresh lineage、claim 与 schema-3 hard contract 匹配；receipt content SHA-256 为
  `08c7731a...03df` / `e7dcb7cc...c2c9`。这只闭合身份门，`+200` trailing-21 activation/方向屏尚待复核，
  不停臂、不晋级。见 [conditional 卷宗](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)。

- Franco 反手拉 B 的 L0 portability 根因已做成 fail-closed source 修复：历史 Pod1 checkout 只保留为
  claim/source provenance，当前 detached-clean commit、runner、source validator 与 runtime body order
  另行内容绑定，且无旧绝对路径 fallback；原生 consume loader 仍拒绝当前 runner 接管旧 activation，C
  不消费。新增 full `dry-run` 会跑完整只读 L0 而不写证书，两个专项 `51 passed`；Pod2 尚未运行，G08
  继续 Partial。见 [L0 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)与
  [操作](operations/run_motion_backhand_loop_b_l0_static.md)。

- Full-scene terminal authority 现在自行重算 ignored A3 target/donor 当前库存、URDF mesh 闭包与
  donor clean commit，将实测 SHA 写入 immutable `current_closure`；直接绕过 queue wrapper 后的两侧
  资产漂移与 boolean iteration/lineage 负测均 fail closed。full-scene 专项 `39 passed`，整合
  harness/source-asset 回归 `146 passed`。caeb 旧 probe 的 wrapper doctor 当时通过，但旧 result 没有
  `current_closure`，故不追认新能力；新 Pod result 尚未运行，G05 保持 `Partial`。见
  [G05](gates/G05_isaac_training_first_loop.md)、[queue 操作](operations/run_lean_training_queue.md)与
  [运行绑定接口](interfaces/lean_training_run_binding.md)。

- strict receipt 解锁后的 conditional control/treatment 已分别在 Pod2 GPU1/GPU2 越过 first iteration，
  PID=PGID 为 `357023/357679`；尚无 checkpoint 早判或 Reward 结论。紧接着的 interaction control
  PID=PGID `358331` 在 first iteration 前 dynamic URDF import 报
  `malloc(): invalid size (unsorted)`、`rc=134` 并自然退出，treatment 未发射；claim/namespace 保全，
  不能写成 interaction pair 已运行或 Reward 失败。旧 control 行已 rejected/no-relaunch；逐字同配方
  `control_retry_v2` 与从未 claim 的 treatment 均 ready，只允许同一 `fill --count 2` 事务先等 retry first
  iteration 再发 treatment。该事务随后按序成功：retry-v2 PID=PGID `359240`（Pod2 GPU1）、treatment
  PID=PGID `359872`（GPU2）均越过 first iteration，interaction pair 现 live；尚无 checkpoint/早判。见
  [conditional 卷宗](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)与
  [interaction 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-INTERACTION.md)。

- c7 非科学 full-scene canary 已闭合旧语义的基础设施路径：result/model/hard-contract SHA-256 分别为
  `02780b52df27255eea096f34dda9a26e806ae3a196c233a46a2af1cde16c4186`、
  `a813ea9ba8c058cf5ed2f9a9a8f8fe3b95ec0903cd3702831b99736736738e68`、
  `c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`；76 个 tensor 的
  1,762,715 个浮点元素全 finite、fatal0、trainer/supervisor 原 PGID 自然为空。旧结果中的
  `unlock_authorized=true` 不满足 `main@caeb9ad` 新增的实际 4096-env、物理球/三实体与完整 schema-3
  终档门，不能解锁。

- strict caeb probe `caeb_strict_terminal_pod2_gpu1_a1` 随后通过：result/claim/model/hard-contract SHA-256
  分别为 `0d03bd0305a56e56440b14e1f41278a26c0cad3a84cc1245325faed1ef29b1d1`、
  `7437db488d8aa062aba8de91fb517362cc609a81900f0e953f80e15174c36ad5`、
  `e1b79d142c13bc2df513b2a7311fbeb7b610fc64047e095c1a54c76571fe3106`、
  `c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`。它绑定 clean caeb source、实际
  4096 environments、physical ball/三实体、76 tensors / 1,762,715 浮点元素全 finite、fatal0 与自然空
  PGID。两份 P1 队列已显式
  [`launch_authorized=true`](DEFINITIONS.md#launch-authorized)；后续点火、一次 importer abort 与
  unchanged retry 的实际状态见本节首条。probe 仍非科学且不可晋级。见
  [G05](gates/G05_isaac_training_first_loop.md)与[操作](operations/run_lean_training_queue.md)。

- Lean queue 加入显式 `launch_authorized` 发射闩：false 时
  `fill/launch-next` 会零 SSH 拒绝；live snapshot
  只访问 Pod2。历史七条 ready 行已按既有证据终态化，新 conditional 与 V1+V2×base-decel 配对现改绑
  `main@caeb9ad`、分列 Pod2 GPU1/GPU2，并在上述 strict receipt 后显式解锁；当前运行态见本节首条。
  见 [G05](gates/G05_isaac_training_first_loop.md)与[操作](operations/run_lean_training_queue.md)。

- full-scene probe P1.5 关闭短跑终态与假绿缺口：launcher-only pre-marker/watchdog/timeout 只能冻结失败；
  pass 新增实际环境数、物理球/桌实体、face179、31/31 零摩擦和 direct-file schema-3 validator 门，并修正
  PID reuse/并发 finalizer race。该源码合入时的增量 focused 为 `100 passed`，当时未重跑 Pod、未追认旧
  c7 contract；后续 strict caeb receipt 见上，G05 仍 `Partial`。见
  [G05](gates/G05_isaac_training_first_loop.md)与
  [运行绑定接口](interfaces/lean_training_run_binding.md)。

- Kit watchdog 的 marker-priority 测试移除亚秒 sleep 调度竞态：现在于第二次 marker probe 同步注入
  marker，仍直接验证 timeout/stale 已到边界时 marker 优先，且不改生产 launcher 语义；相关专项
  `15 passed`。见 [G05](gates/G05_isaac_training_first_loop.md)。

- qdot `-5/0` terminal 曲线推翻 `+500` 的 mixed-only 读法：updates `980–1000` 中 treatment/control 的
  position pass=`0.878/0.593`、error=`4.74/9.62 cm`、signed composite=`0.310/0.146`、virtual
  return=`0.454/0.265`，fall/completion 基本持平；两份 `model_1000` 均 finite/lineage/contract/claim exact。
  `-5` 改判为晚熟候选，但仍不采用、不买 seed，先过 immutable MuJoCo/vendor judge。

- qdot matched control 已自然终档并释放 Pod2 GPU0：`model_1000.pt` SHA-256 `b6672869...12cb9`，
  filename/embedded iter=`1000`、76 tensors/1,762,717 elements finite、fresh lineage `1`、schema-3
  contract SHA `25faa6f5...da12` 与 claim `c73ac441...8a959` 均匹配，fatal `0`。这只闭合配对终档身份；
  `-5` mixed-signal 仍不采用、不买第二 seed。

- Pod2-only full-scene probe 已实际创建唯一 PID=PGID `353107`，随后自然 `rc=1`：fresh `077e70c`
  checkout 缺 Git-ignored A3 URDF/mesh tree，故无首 iter/model/hard contract；不是 4096-env 或 Reward
  失败，原 attempt 不重放。控制器同时修掉 reserved Pod1 的越界快照和每臂重复 doctor SSH；下一门是
  完整 46-file source-asset hydrate/receipt。见 [conditional 卷宗](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)
  与 [G05](gates/G05_isaac_training_first_loop.md)。

- lean queue P1/P1.1 已进入 main：trainer-owned `run_binding` 与 exact milestone attestor 不再靠 glob 猜
  checkpoint；新 `full-scene-probe` 保留正式 `4096 env` scene recipe、只用独立 2-update 非科学 namespace。
  Pod2 clean detached `077e70c` source 与外部动作/bank/exam已核对；conditional 和 V1+V2×base-decel 新 pair
  在该 source-gate 合入时已绑定但仍 blocked、probe 尚未执行；后续 strict caeb 结果见本节顶部。见
  [运行接口](interfaces/lean_training_run_binding.md)与
  [G05](gates/G05_isaac_training_first_loop.md)。

- qdot 同源 `-5/0` pair 的 `model_500` 身份/finite/contract/claim 全过；末 21 updates 显示 qdot max
  `-16.4%`、near-limit `-20.1%`、torque saturation `-35.5%` 且 fall 改善，但 position pass
  `0.418→0.107`。判 mixed signal，不采用、不买第二 seed；缺 activation denominator/per-joint tail，等待
  immutable judge。见 [Fresh-C 机制卷宗](experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)。

- conditional-face source610 的 1-env warmup 不能代表正式启动：Pod2 GPU1 的 4096-env control 在 dynamic
  URDF import 后停住，iter0、无 scene/hard contract/checkpoint；精确 PGID `332786` 的 TERM 30 秒无响应后
  对同一 PGID KILL，证据保留。serial fill 未创建 treatment，因此不是“两条 Reward 都失败”。旧 pair 撤销，
  新 pair 必须绑定 source-pinned watchdog/runtime binding，并过同规模 full-scene 非科学 probe。见
  [实验卷宗](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)与
  [G05](gates/G05_isaac_training_first_loop.md)。

- qdot retry-v2 已在 Pod2 GPU2 通过 no-Kit doctor 与真实 `Learning iteration` boot marker；只读复核到
  iter `79`，schema-2 claim digest、96 项实际 argv、`model_0.pt` finite、hard contract 与 fresh/claim
  lineage 全匹配，fatal `0`。第一次 0-update 超时因此维持“基础设施失败”，不是 reward 失败；下一步补
  同 source/seed 的 weight `0` 匹配对照。

- 结果出现前已把三条必要对照写入 Pod2-only active YAML：qdot 同-source weight `0` control，以及
  conditional-face 同-source `0/-0.4` 配对；三条都是 seed3 的不同因果单元，不是复制失败 seed，均有
  `+200/+500/+1000` 早判。当前仍为预注册、未 launch。

- qdot control attempt-1 在 iter0 的动态 URDF import 返回前停住，无 contract/checkpoint；成功 treatment
  有同样 warning 而能完成 scene creation，故排除 reward 与 warning 字面为差异根因。exact PGID 收口并
  保全后，只登记一次 unchanged retry-v2；重复则停止 retry，转 boot watchdog/预转换 USD。

- lean harness 新增独立 `boot-warmup`：从 exact job 派生 1 env×2 update、独立 claim/namespace、180 秒
  boot 上限的非科学冷启动探针，reserved Pod 与科学确认 token 均 fail closed；queue suite `23 passed`。
  尚未在 Pod 执行，不能写成 runtime 通过。

- conditional source 的 Pod2 GPU1 `boot-warmup` 已自然退出并通过：2/2 updates，`model_0/1` 各 76 tensors/
  1,762,715 floats 全 finite，embedded iter、schema3 contract、claim、fresh lineage 匹配，fatal0；明确
  `not_science`，不进入成绩或晋级。

- 通用 Kit launcher 新增默认 180 秒 content-bearing stale-log watchdog：增长重置、marker 优先，只精确
  收口自己的已验证 PGID并以 rc125/sidecar 留证；空日志仍走 hard timeout，stat 异常 fail closed。
  专项 `9 passed`、相关 retry/queue `50 passed`；它缩短卡死，不冒充 importer 根因修复。

- 找到“容量写3/4但每卡只能发一条”的根因：`flock FILE command` 的 fd 被 detached trainer 继承并持锁
  到终档。lean harness 现让短命 controller 持 fd8、对子 launcher `8>&-`，并加入容量内 preferred-slot/
  满载回退；queue suite `24 passed`。现役旧锁不剥离、不重启。

- Lean queue P0 已把重复/owned Hydra override、control flag/interpolation、run-dir 覆盖与未解析配方挡在
  claim 前；doctor 用真实最终 argv 做 no-Kit compose，schema-2 canonical claim 自动绑定 source、argv、
  预算和 motion/bank/exam identity。五机制 `+500` 中 V1+V2 出现 composite `0.0893` / normal pass
  `0.268`，V2 单独格判可替换；qdot 首次发射在第 0 update 的 A3 URDF import 超时，exact PGID 已由
  launcher 收口，无 model，按基础设施失败保全并排全新 retry-v2。见
  [实验](experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)与
  [操作](operations/run_lean_training_queue.md)。

- Pod1 已全部移交 Yikang 冲刺：Codex 三条 trainer 精确 `TERM` 于 iter `792/782/743`，未发 `KILL`，
  `model_700.pt`/日志保留且复核无剩余 compute process。active queue 新增机器可检验的
  `dispatch_pods: [pod2]`；新任务只在 Pod2 三卡轮转，同时只读 Pod1 旧 claim 防重复。

- 轻量训练队列的发射前 P0 合同已收紧：recipe 重复/越权/Hydra 控制语法在 SSH 前拒绝，`run_dir`
  全局唯一且只能原子首次创建；standalone doctor 与 launch 共用最终 argv 做 no-Kit
  `train.py --cfg job --resolve`。canonical claim 绑定 source、完整 caller argv、run/预算和三类 input identity，
  digest 自动写入真实 trainer argv。focused `19 passed`；本条没有 Pod 写入、新 trainer/checkpoint 或行为结论，
  G05 仍为 Partial。见[操作](operations/run_lean_training_queue.md)与
  [实验](experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)。

- `main@61007e9` 新增默认关闭的“不逃离就绪区”固定预算 Reward source gate：击球时间窗内未就绪时保持最大成本，
  就绪后才把成本连续换成有符号拍面误差；位置/拍速改善绝不会增加成本，门外没有拍面梯度，也不能
  通过故意退到外门免罚。首轮只允许同新 source 的 `0/-0.4` 配对、单 seed 与
  `+200/+500/+1000` 早判；focused `6+78+34+62 passed`，未合 main、未跑 Pod，不改变当前 setting。
  见 [实验](experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)、
  [G05](gates/G05_isaac_training_first_loop.md)与[训练操作](operations/run_training.md)。

- Fresh C queue 的五条 `retry-v2` 已全部越过真实 first iteration，现场到 `103–160/1001` 且无
  NaN/Inf/Traceback/OOM/Killed；五份 `model_100.pt` 均 filename=embedded iter、76 tensor finite、
  schema-3 hard-contract SHA 与 fresh lineage 绑定通过。第六个 actual qdot-limit tail 格冻结为 fresh
  seed3、4096×1001、weight `-5.0`/margin `0.85`，只作 +200 direction screen；同 source weight0 control
  尚未跑，故不得作因果采用或买第二 seed。见
  [实验](experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)与
  [操作](operations/run_lean_training_queue.md)。

- 第二圈第六机制的 31 关节 qdot-limit hinge 已完成 E1 source gate：VirtualBall 默认关闭，normalized
  tail 公式直接消费 actual articulation qdot/velocity limits，Hydra 只接受非正 weight 和 `(0,1)` margin，
  applied marker 与 hard-contract/outer-claim 边界已写清；错序、零/非有限 limit fail closed。qdot-focused
  `30 passed`、override 全文件 `76 passed`、schema-3/claim suite `62 passed`；没有 machine prereg/Pod
  run/checkpoint/行为结论，不授权点火。
  见 [G05](gates/G05_isaac_training_first_loop.md)、
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [训练操作](operations/run_training.md)。

- C3/D3 K100 v1 在 C3 ONNX 导出前暴露 ignored Isaac A3 asset 打包缺口并永久冻结，未产生行为成绩。
  v2 新 namespace 在 claim/judge 前绑定训练 checkout 的递归 canonical inventory、一次 hydrate/二次
  verify 角色和 `libGLU.so.1` 存在性；focused `56 passed`，static/source-plan rc0。hydrate 后的
  C3/D3 inexact diagnostic 均成功导出并进入 MuJoCo，asset blocker 已关闭；日志诚实记录
  `evaluation_contract_exact=false`，两侧又都在第 0 题前因
  articulation `[8]` PhysX velocity-limit braking 无 MuJoCo 等价约束而 fail closed。无 attempt/score，
  `asked=0`、方向分不存在；K100 behavior 仍 OPEN，L2/第二 seed/promote 继续阻断。
  见[实验](experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)与
  [v2 操作](operations/run_phase1_signed_face_c3d3_k100_v2.md)。

- C3/D3 v2 快审已把新 evaluator bytes 逐级绑定到 attestor 与 paired manifest，并把 ignored asset
  hydrate 从可覆盖 child 的 `rename(2)` 改成 exclusive root/directory + `link(2)` 原子 no-replace；并发
  sentinel 攻击 fail closed 且保留证据。focused `57 passed`、static/source-plan rc0；未新增 Pod runtime。

- Fresh C 五机制 attempt-1 均因队列未把 `HOPE_WBT_PYTHONPATH` 传给 raw Python，在第 0 update、
  first marker 前以 `ModuleNotFoundError: whole_body_tracking` 退出；五目录/claim/log 已保全，无 model，
  不能解释为机制失败。旧 namespace 已 `rejected`，同 recipe 的全新 `retry-v2` 是唯一一次基础设施重试。
  doctor/trainer 现共用 child env，exact module probe 在 claim 前；新增无写 `doctor --live` 与单进程
  `fill`（逐条等 first iteration 后重采）。focused `17 passed`；尚未启动 retry-v2，G05 仍为 Partial。
  见[实验](experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)与
  [操作](operations/run_lean_training_queue.md)。
  随后两 Pod 五格 `doctor --live` 全部 `DOCTOR_OK`，六 GPU live occupancy 为 0；没有 retry-v2 claim/
  trainer，Hydra compose 仍明确未运行。

- 动作专属轻量 YAML 训练队列完成 E1 source gate：一行绑定 motion、专属 train bank/exam、source、
  base+delta、seed、预算、`+200/+500/+1000` milestone 与六卡资源；默认 dry-run，blocked 永不启动，
  Pod1/Pod2 每卡容量 `4/3` 且先铺满六卡一圈。runner 入口源码固化、ready placeholder 在 SSH 前拒绝，
  全局 scheduler flock 内重采六卡再选槽；`nvidia-smi` 同 PID 重复行按每 GPU unique PID 去重。
  探索入口不做逐文件/pip/receipt hash；当前示例仍 blocked，
  没有 Pod trainer 或行为结果。见[操作](operations/run_lean_training_queue.md)。

- C3/D3 同卷 K100 one-shot consumer source gate 已绑定 paired L1 receipt、两份终档 exact attestation、
  immutable schedule/activation 与 float `[1.0,-1.0]`；focused `28 passed`，static/source-plan rc0。尚未 SSH/
  attest/judge，L2、第二 seed、stop/promote 仍为 false。见[操作](operations/run_phase1_signed_face_c3d3_k100.md)。

- C3/D3 在 Pod1 GPU1/GPU2 各只 claim 一次并自然到 `model_24.pt`；两条 hard-contract marker 与
  31/31 实例化零摩擦 marker 均唯一，finite/iter24/fresh-lineage/outer-claim binding 通过。paired L1
  receipt SHA `bb3cd749...bbde` 只闭合 provenance，不判 guidance 效果；不得重跑，K100/L2/第二 seed
  继续阻断。见[实验](experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)。

- A2/B2 plan-only gate 已收口为全新 v2 跨 Pod one-shot L1 runtime：Pod1 GPU0 跑 A2 对照，Pod2 GPU0
  跑 B2 guidance；两条均为同父模型热启动、`512 env × 25 update`，并显式绑定零摩擦 argv/runtime/hard
  contract、空 GPU、fresh namespace 与 no-retry。focused `27 passed`，static/plan rc0；尚未连接 Pod 或
  启动 trainer，不授权 judge/L2/第二 seed。见[实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)
  与[操作](operations/run_phase1_signed_face_a2b2_l1.md)。

- C2 的 31/31 非零摩擦根因已转成全新 C3/D3 显式零摩擦 L1 source gate：两格 fresh seed3 只差
  signed-face guidance `0/-0.4`，同一 zero-friction leaf 被唯一绑定到 argv、optimization recipe、outer
  claim、runtime marker、hard contract 和 checkpoint replay。专项 `38 passed`、完整回归
  `972 passed, 10 skipped`；此条只表示 main source 可执行，Pod runtime/行为仍未通过。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)与
  [操作](operations/run_phase1_signed_face_c3d3_l1.md)。

- 非击球臂 A0/A1 的 checkpoint 层已闭环：A1 自然退出；A0 在稳定写完 `model_1000.pt` 后发生近三小时
  Kit/Python teardown hang。终档 iteration/finite/fresh-lineage/hard binding 与正式 failure regex 先通过，
  精确单成员 PGID `1811464` 对 `TERM` 无响应后才被同 PGID `KILL`，未重启或重发。冻结 v1r1 finalizer
  随后验过两臂 `200/500/1000` 并发布 paired result SHA `30ba716b...d7d9`；signed K100 仍未判，第二 seed/
  晋级仍阻断。见[实验](experiments/non_striking_arm_imitation_ablation_20260713.md)与
  [操作](operations/run_phase1_non_striking_arm_imitation_a01.md)。

- B 反手拉在 fresh no-write preflight 后花掉唯一 schema-2/FK consume，`91` 帧 NPZ SHA
  `e2eb99e6...d28cc`；独立 `validate-result` 得到 `runner_lineage=true`、`npz_bound=true`，completion-last
  ledger SHA `c0a25f2c...f4f8b`。只解锁 B 的 L0 静态证书；C 保持未消费后备，L1/桌网/动力学/RL/真机
  仍未授权。见[动作实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- B 反手拉的 [L0 静态审计](DEFINITIONS.md#motion-l0-static)首次 runtime 调用在运动学与 certificate 写入前
  暴露历史 runner 的 checkout-path portability bug，只创建输出父目录。修复保持旧 runner/claim 字节，
  用 activation bytes/SHA、canonical path 和 inspected source commit 进入原完整 lineage/NPZ 校验；新
  prereg/validator SHA 为 `7118b9cd...595a6` / `ee6ccd46...c171`，专项连同上游 schema-2 为 `58 passed`。
  没有重跑 runtime、没有 certificate，子门仍为 Partial，L1/桌网/动力学/RL/真机继续 blocked。见
  [实验](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)与
  [操作](operations/run_motion_backhand_loop_b_l0_static.md)。

- signed-face K100 的 generic checkpoint attestor 已完成 E1 source/static gate：每个 request 必须显式绑定
  checkpoint SHA/filename+embedded iteration/finite/fresh lineage、相邻 hard contract、producer claim、
  evaluator source+runtime、MJCF/plant 和 actual schedule/activation；同 checkpoint 只能写一个
  SHA-derived no-clobber evidence/claim namespace，且 claim 不授权 judge、停止或晋级。旧 runtime receipt
  摘要的 integer `[1,-1]` 被 versioned correction pointer 保留并降级；consumer 直接严格验证 actual
  activation 的 float `[1.0,-1.0]`；路径通配符/穿越、symlink ancestry、checkpoint 替换、request TOCTOU、
  dangling namespace 与 evidence-only partial 都 fail closed。focused `21 passed`、rebase 后仓内 `tests/`
  `956 passed, 9 skipped`，且
  `py_compile`/`static-validate` rc0；未连接 Pod、
  未创建 runtime claim 或运行判卷。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)、
  [G05](gates/G05_isaac_training_first_loop.md)与
  [操作](operations/run_phase1_signed_face_k100_checkpoint_attestor.md)。

- signed-face fresh C2 已在 Pod1 自然产生 finite/iter24/lineage1 terminal bytes，但 v1 用整数
  `[1,-1]` 假拒绝训练端合法 float `[1.0,-1.0]`；冻结 v1r1 又把 trainer 实际五键 compact bank
  record 错当成应直含第六个 physics SHA。最后一次成功只读快照证明 v1r1 从未安装/运行且 D2 从未
  claim；后续 SSH unknown，历史 absence 不授权 launch。v1r2 保持 v1/v1r1 bytes 冻结并禁止运行旧
  mode，只接受 exact 五键，再从 NPZ metadata/source-family 独立绑定 physics；旧 v1r1 evidence/pair、
  D2 arm/exact run 任一存在都 fail closed。六文件外部 mini-tree static/plan 与专项攻击测试
  `52 passed`，重复 JSON key 也 fail closed；三代聚焦回归 `111 passed`，完整仓内 `tests/` 为
  `934 passed, 10 skipped`。本分支未连接 Pod、安装 control、写 attestation 或启动 D2，L2/judge/
  第二 seed 仍未授权。见
  [face-sign 实验](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)与
  [操作](operations/run_phase1_signed_face_cd_l1.md)。

- S0/M0 exact-GMR v1 在真实 runtime `inspect` 中于写 root 前 fail closed：hash-only pip 证据
  `97c66009...18ff` 没有保留 234 行输入，且不能由 exact Python 的实际规范化快照
  `56b0f8af...c694` 复现；M0 未重复同一 blocker，两份 v1 root 均 absent，v1 永久 **NO-CONSUME**。
  新 attempt v2 使用新 consumer/plan/runtime/root，跟踪完整 4,702-byte snapshot，并绑定 v1 base consumer、
  五个直接 import 的 version/origin/METADATA/RECORD 与 post-converter 重验；S0/M0 `consume` 还共用
  exact marker 的 exclusive flock，只串行而不互相设成功依赖。runtime SHA `a55c52cc...b7b2`，S0/M0 plan
  SHA `0746291e...f2f2` / `a810ee01...41f3`；两份 host `static-v2` 通过，v2 专项 `15 passed`、
  新旧 focused `28 passed`、仓内回归 `949 passed, 10 skipped`。v2 runtime 未执行，
  GMR/schema-2/训练/真机仍 blocked。见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)与
  [操作](operations/run_motion_s0_m0_exact_gmr.md)。

- S0 高点拍压与 M0 横移老师的 shared exact-GMR source/static blocker 已闭环：16 项低频只读证据绑定
  clean GMR tree、七个 import module、mapping、Python/pip、direct 31-joint/32-body order 与显式 qpos
  bijection。direct retarget XML 的 site inventory 精确为空、左右 foot site absent；consumer 不再错误要求
  canonical vendor 足点出现在 retarget XML，M0 stance 仍只用 vendor MJCF 做 FK。shared runtime SHA
  `cb9b01b9...0d45`，两份 host `static` 均 `PASS`；canonical-site 冒充、新 site/非 absent、runtime drift
  负测在专项 `13 passed`，基于最新 main 的仓内 `tests/` 为 `867 passed, 10 skipped`。未连接 Pod、读取私有 PT、运行
  `inspect/consume`、GMR、仿真、训练或真机；见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)与
  [操作](operations/run_motion_s0_m0_exact_gmr.md)。

- Franco 将共享算力调度改为“先铺满卡、再叠并发”：保留已绑定运行原位不动，新任务先跨 Pod1/Pod2
  六张可用 GPU 各放一个有独立科学问题和早判合同的单元，再开始第二、第三轮，Pod1 才有第四轮；
  被他人占用、前置门未过或会破坏严格配对的卡跳过，不用重复 seed/失败配方补位。操作真源已同步到
  [跑批作战手册](runbook.md#rtx-5090-实测算力手册)与
  [RunPod 操作约束](operations/run_on_runpod.md#hard-rules-summary--full-list-in-the-pod-readme)。

- B/C schema-2/FK 的两次真实 no-write runtime inspection 已入严格 receipt：Pod1 detached
  `748b6d5` 前后 clean；默认 Python 因缺 `onnxruntime` rc=2 fail closed，现成
  `hope_mjeval_venv` 绑定 Python/NumPy/ONNX Runtime/MuJoCo `3.12.3/2.5.0/1.27.0/3.10.0` 后，
  B/C `91/98` 帧分别 rc=0，donor/MJCF/name domain exact，两个 output root 仍不存在。可绕过且失败
  不永久花预算的 v1 activation 已否决；v2 一次性 runner 源码门现以 atomic pre-child claim、B/C
  shared flock、permanent failure/completion-last ledger、runtime/input 重验和 NPZ 内容级 lineage
  validator 闭合，bypass/concurrency/failure-spends/runtime-drift 等专项 `28 passed`、连同 prereg
  `45 passed`，latest-main `tests/` 回归 `850 passed, 10 skipped`。runner 尚未在 Pod 执行、attempts
  仍为 0；L0/L1/simulator/训练/真机均未授权。见
  [实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)、
  [G04](gates/G04_sim_modeling_mujoco_isaac.md)、[G08](gates/G08_blind_spot_improvements.md)和
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- B/C 独立 schema-2/FK prereg 的 source gate 已闭合：两份计划绑定 exact 私有 SE(2) PKL/report、
  不重叠 no-clobber 输出与 `91/98@30 Hz -> 151/163@50 Hz`；共享合同绑定 restricted pickle、formal
  donor SHA/三行 metadata 期望、vendor `1 XML + 74 mesh` closure、31-joint/32-body order，以及
  link-origin pose/COM velocity。consumer 只接受 `--hope_frame off`。两份 `static` 与专项
  `17 passed`，基于最新 `origin/main@7679b30` 的仓内回归 `782 passed, 10 skipped`；没有读取私有 PKL/ONNX、没有
  FK/schema-2/L0/L1/simulator/RL/真机。下一步仅为逐资产
  no-write runtime `inspect`。见[实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)和
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- Pod2 CPU 补跑了 MuJoCo evaluator 两个此前因本机缺依赖而 skip 的 optional 模块。首次真实收集
  `2 failed, 8 passed`，定位为 synthetic fixture 把非等价执行路径当对照、把 welded child 当可碰
  articulation；只修夹具后，同一 production evaluator bytes 在 Python `3.12.3` / MuJoCo `3.10.0`
  得到 `10 passed`。失败与通过日志/source SHA 已冻结在
  [runtime result](../configs/mujoco_eval_optional_runtime_test_results_20260714.json)；该结果不包含 policy、
  vendor MJCF、Gate3、GPU 训练或真机，G04/G06 仍为 `Partial`。

- signed-face E2 rebound exam bank 的下一层 immutable K100 source gate 已冻结：严格复用现有 schema-v3
  schedule 算法，从 exact bank SHA 重建 question ID，seed0/hold0–100/每侧无放回50/全100次分母；raw-A
  `[+1,-1]` physical-B 身份、旧纸拒绝、no-replace 和 activation-last 均 fail closed。专项攻击回归
  `14 passed`、latest-main root `747 passed, 10 skipped`、`static-validate` rc0。随后 Pod1 的 clean detached
  `748b6d5` source 完成单次 exact-bank consume：100 unique、50/侧；schedule file/semantic/order SHA
  `f2777dcd...1ca` / `3ca4bdba...3365` / `09f778f2...bd0`，activation file/content SHA
  `e0125b0e...bb4` / `533beb03...3d8`。这只把 paper 升为 E2 materialized；checkpoint execution contract、
  L2/judge/第二 seed/晋级仍全阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)与
  [操作](operations/run_phase1_signed_face_exam_k100.md)。

- 非击球臂 A0/A1 直接 mask 已从设计升级为 E1 machine prereg：训练 override 同时只从位置/姿态/
  线速度/角速度四条模仿 Reward 删除左 shoulder/elbow/wrist，并用负测证明右击球臂/躯干、reward
  参数、关节/动作/力矩/接触/自碰/终止安全均不变；四项 post-override body list 已进入 checkpoint
  hard contract，A0/A1 各绑不同 SHA，去掉该唯一字段后必须完全相同。两条 fresh seed17 长臂绑定同 motion/bank/
  `4096 env × 1001 update`，默认 plan-only、root token 点火、no-clobber runtime/finalizer、
  `+200/+500/+1000` 早判；A2 固定预算继续 blocked。Pod1 A0 已以 exact PID=PGID `1811464` 运行，
  `model_200.pt` 的 iter/finite/fresh lineage/hard-contract SHA 绑定通过；旧 outer verifier 因错误要求 compact
  bank record 直含 metadata physics SHA 而假拒绝，A1 当时从未 claim。一次性 v1r1 continuation 已补
  `12 passed` 的 source gate：绑定 old+new control、复现旧错误、独立解析 bank metadata、先 attest 既有 A0，
  再且仅再 claim A1；禁止重跑 A0，A0/A1 漂移或预存在均 fail closed。external `validate-runtime` 全绿后
  A1 已以 PID=PGID `1816234` 越过 Kit ready，hard-contract SHA `c85b52a...6b146`；A0 `1811464`
  untouched，judge 未启动。external plan 的相对路径 bug 在任何 write/claim 前失败且不影响绝对路径
  runtime/launch；冻结 v1r1 bytes 不得修改，只在后续新版本修。尚无 A1 milestone、配对终档、同卷判读
  或真机。见[实验](experiments/non_striking_arm_imitation_ablation_20260713.md)与
  [操作](operations/run_phase1_non_striking_arm_imitation_a01.md)。

- MuJoCo frame/evaluator integration 的独立红队 `NO-MERGE` 阻塞已逐项关闭并合入 main：bound implicit
  改为每 substep 执行 Isaac `clip(P-D)`；被动/无 effort-limit 代理 formal fail closed；自碰只认 pelvis
  机器人子树且 formal 首次即拒绝，动态球不误报；mask 供证只接受 canonical/严格空 partial；旧
  Phase-B rider direct loader 按内容 SHA 撤销；旧 scoreboard header 不再错列追加。合入后 focused
  `147 passed, 2 skipped`、当前 main 仓内 `tests/` 为 `714 passed, 9 skipped`；两项 focused skip 都因
  本机无 `mujoco`，不是 physics 通过。本机也无 `torch`，Phase-B Torch 套件未收集。重要合同修复已记入
  [TIMELINE](TIMELINE.md)；没有运行 Pod、Isaac、vendor backend、Gate3/Gate3B 或真机。测试和剩余
  optional-runtime 边界见
  [集成卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md)；G04/G06 仍为 `Partial`。

- 第二轮独立红队又抓到两个残余假绿并在候选分支修正：可覆写 `__call__` 的 partial subclass 曾能以
  canonical `.func` 洗出 epoch 1，现逐层仅接受 exact built-in partial；自碰曾只看 control step 末态，
  现每个 MuJoCo physics substep 后 formal 首碰即拒绝、diagnostic 完整累计。两项均有 dependency-free
  攻击复现与负测；未运行 MuJoCo/Isaac/vendor/Gate3/真机，G04/G06 继续 `Partial`。见
  [集成卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md)。

- v6/v8 D 两次 pre-contract timeout 的三次低频只读审计已机器入账：两份 D 都以加载 byte-identical
  table USD（`683,433` bytes，SHA `c6fc99a8...996`）为 Kit 最后一行且未到 PhysX；相邻 C 在
  `2.339/3.031 s` 越过同一边界，v8 D 在 C clean shutdown 后 `44 s` 才启动。事后 GPU/RAM/disk/shm
  非饱和只排弱持续容量耗尽；Carbonite 残留只记相关，`dmesg` 未获权限，根因仍未证明。已冻结
  [结果 ledger](../configs/phase1_signed_face_boot_root_cause_results_20260714.json)与 design-only
  `D-first/ordinal-4 × host/private IPC` [诊断 prereg](../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)；
  无 Pod/process/signal/training/retry/judge/部署/真机权限。专项 `8 passed`，最新 main 基线 host
  `tests/` 回归 `722 passed, 9 skipped`。

- B/C schema-2 前置审计纠正了关节列序合同：GMR `dof_pos` 与 Isaac/runtime `joint_pos` 的 31 个
  名字相同但顺序不同。新增两份内容绑定的 order 真源、双向 permutation、旧 mirror 与完整 ONNX metadata
  fail-closed validator；converter 改读合同，历史 L0 auditor 保持已被运行账本绑定的 byte-exact 源码、
  由 validator AST 复核其 target mirror。重复/缺失/额外/错序/错误长度/partial
  metadata/duplicate JSON key/NaN 负测专项 `12 passed`，基于 `origin/main@5734dc8` 的 repo 回归
  `733 passed, 10 skipped`。未读私有 B/C 资产、未跑
  FK/schema-2/simulator/RL/真机，证书仍
  为 0；见[空间重定向实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)、
  [关节接口](interfaces/joint_order_and_robot_state.md)与
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- 反手拉 B/C 的 rank-0 主选已各有独立 no-clobber 整轨站位实体化 prereg（SHA
  `e016ca74...51aee` / `27f938cd...9d454`）和 restricted-pickle consumer
  `21ebbe68...87375`。consumer 只做冻结的 proper [SE(2)](DEFINITIONS.md)，验证 xyzw 左乘、
  Z/fps/dof/non-spatial exact、可选 world velocity 同转、save/reload 逆变换、刚体距离和 report-last；
  专项 `10 passed`、全仓 host tests `656 passed, 9 skipped`；两份 exact 私有源先 inspect，后在 Pod1
  CPU-only runtime `consume`。B motion/report SHA 为 `27827912...ad6` / `a238c077...df3`，C 为
  `0dd981a6...f48b` / `b3b93d2c...f67`，最大逆误差 `<2.23e-16`。没有 simulator/RL/真机，
  schema-2/L0/vendor L1/桌网/动力学仍未跑、证书仍为 0，只解锁 schema-2 prereg。见
  [实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)与
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- signed-face exam bank 已在 Pod1 目标 runtime 完成 no-write validate 与独立 E2 发布：新 bank/report SHA
  为 `60e1a7ad...d1ca` / `dd4332ed...ad0`，24 个非 metadata 数组未变，正/反手 `183/188` 题 old/new
  output bytes 一致且 landing/net 全过。它只通过数据门；新 bank 绑定的 immutable schedule、paper
  activation、L2/judge/formal score 仍阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)。

- signed-face foreign v8 使用新 source/manifest/launcher 串行跑过 A/B/C 前序，D 作为第四格又在
  900 秒内未到 hard contract/runtime verified；exact-PGID wrapper cleanup 后 rc=124，没有学习、checkpoint
  或 NaN/Inf/Traceback/OOM。继旧 v6 D 后这是第二次独立 pre-contract timeout，自动重试已停止，转入
  boot 根因；四格 activation/L2/judge/第二 seed 全 false。最终 Pod1 审计为 0 trainer/worker/judge、
  三张 GPU 空。见[机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)。

- v6r1 的首次真实 `validate` 在写 claim/训练前发现合同自相矛盾：immutable audit 明确 D 的
  `run_dirs=[]`，但 validator 却要求旧 would-be training path 必须存在。团队没有伪造目录；v6r1
  从未 claim、launch、signal 或训练。新 [v6r2](DEFINITIONS.md) 只发布静态源码修正：旧 path 必须
  absent，任何目录/file/symlink/special entry 都 fail closed；它只支持 `static-validate`，没有 runtime
  preflight、命令重建、launch 或 finalizer。专项 `14 passed`，合入当前 main 后仓内 `tests/` 为
  `713 passed, 9 skipped`；v6r2 明确未启动，下一步仍是第 4 格 Kit boot 根因与独立新 prereg。

- S0/M0 的下一层 exact GMR 已形成两份独立 no-clobber plan 与共享 consumer：五条 canonical-beta PT、
  converter argv、Python/pip、A3 model tree、两套 joint/body order 和 31-joint bijection 都是 required；M0
  预冻结 exact 30 Hz ready sample、足点 FK、前后/横向二维脚距、3 cm component band 与独立 5 mm 防收窄门。
  07-14 只读回执补齐 clean tree、model/mapping、关键 import 与 Python/pip SHA，但 direct retarget XML
  order/site 段被传输截断；共享 runtime 以 16 项机器清单继续 blocked，两份 batch 已预注册且真实
  `static` 均 rc=2。专项 `12 passed`、全仓 `645 passed, 9 skipped`；未运行 GMR/仿真/RL/真机。见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)。

## 2026-07-13

- 从现场 `50c49e5` 选择性移植 evaluator parity guard、pelvis COM→link-origin、XBODY gyro 与
  `actor_leg_ref_mask` epoch 供证到最新 main 基线；没有吞入旧分支的 `NOW`/实验状态。combined focused
  `115 passed, 2 skipped`，root suite `647 passed, 9 skipped`。这是 E1 source integration；没有新
  K100、vendor backend、Gate3 或真机结果，跨引擎 gap 仍 inconclusive。见
  [集成卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md)。

- 反手拉 B/C 的 22 条 signed 整轨 proposal 已收敛为 exactly one primary per asset：只把 3 组
  `yaw=0` 的 R0/R1 逐字段同义项合并，随后按平移范数、偏航、回球余量、身体余隙、frame 和 ID
  冻结完整备选顺序。主选 B=`98e7b883...f3c14`、C=`aa0c86fd...f299`；只有桌/网外部几何失败可换
  下一位，schema-2/L0/vendor L1/内部动力学失败必须停止资产。专项 `13 passed`；没有物化、GMR、
  simulator、训练或真机，证书仍为 0；全仓回归 `646 passed, 9 skipped`。见
  [实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)与
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- S0/M0 canonical-beta 已从 E1 计划升为 E2 runtime 结果：Pod1 的 clean detached `c3f58be` 用冻结
  Python `3.10.20` 在 CPU 上依次完成两批 `static/inspect/consume`。S0/M0 completion manifest SHA 为
  `964a7333...f1be3` / `5cef05f7...71a65`，共 `1+4` 条，五条 non-beta 内容 bit-exact，donor copy SHA
  均为 `f405ba45...4cbf2`；formal/training/hardware 仍全 false，M0 脚位/初末脚距/容差/pass 仍全 null。
  未运行 GMR、GPU trainer 或真机；下一步仅解锁独立 exact GMR prereg。见
  [canonical-beta 卷宗](experiments/motion_canonical_beta_s0_m0_20260713.md)。

- signed-face exam bank 的独立严格重绑定已完成 E1 预注册：原 train-v2 manifest 保持 byte-exact，
  generalized consumer 以封闭 profile 另行冻结旧 exam path/`63,968` bytes/SHA、split、`183/188` 题、
  旧/目标 family 与独立 no-clobber output；mutation、source-byte receipt 和双 profile synthetic rebind 为
  `18 passed`。本分支未访问 Pod 或目标 runtime，未生成 bank/report；真实 371 题 replay、从新 bank
  重建 schedule 与 judge 仍阻断，G06 保持 `Partial`。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)与
  [操作](operations/run_phase1_signed_face_exam_bank_rebind.md)。

- epoch-1 signed-face v6 的 A/B/C 已到终档，D 在 `runtime_verified`/checkpoint 前 Kit boot timeout；
  旧 D launch/state/log SHA 与 dead PID/零 checkpoint 诊断、B 终档后 exact-PGID cleanup、`50c49e5`
  source bundle 与 A/B/C checkpoint audit `62076758...d354` 都已冻结。当日新增的
  [v6r1](DEFINITIONS.md) D-only validator 后续被真实 `validate` 证明错误要求一个本应不存在的旧
  training dir；它从未 claim、launch、signal 或训练，现只作 superseded evidence，修正见 07-14 条目。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- signed-face v5 在 scene 构建后、第一次学习前被旧 schema-3 train-bank physics contract 正确拒绝；
  A claim/log 保留，B/C/D 未创建，没有 checkpoint。新增严格 no-clobber 重绑定 consumer：只允许一个
  冻结 helper 的加法式源码变化，要求所有问题数组 raw bytes 不变、metadata 精确四 leaf，并在目标
  runtime 重跑 exact motion contract 与 1481 题 old/new bitwise physics replay。v1 no-write Pod
  preflight 又抓到 Python 小版本相关的 `ast.dump` SHA 假拒绝；v2 改用 helper 原始源码片段 SHA、仍
  保留同 runtime AST 等价门。v2 已发布 bank/report SHA `3a9d8851...5b71` / `9fffed03...bb37`，24 数组
  未变，两侧 landing/net 全过；v6 launcher 绑定完整 report 及父旧 bank→当前新 bank 的唯一精确
  common-field transition。专项 `32 passed`；v6 L1 尚未启动，旧 exam family 也未重绑定，故 L2/judge
  继续阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- signed-face 单-seed 漏斗的首次 Pod v1 preflight 在创建 run 前抓到 checkpoint 审计假拒绝：旧代码
  只看顶层 tensor/合同键，实际 RSL-RL 权重嵌套且 provenance 在 `infos`。父 `model_13800.pt` 递归
  `74` 个浮点 tensor、`1,762,715` 元素、nonfinite `0`；v2 改为递归扫描并绑定 `infos`，保留 v1
  证据且不改四格/seed/预算/L2 blocker。v2 随后在首格学习前因 exact worktree `PYTHONPATH` 未传给
  child 退出；失败 claim/log 保留，其他三格未创建。v3 绑定 tracked setup、拒绝 local override，并
  在 claim 前解析模块来源。v3 因在 `SimulationApp` 前真正 import IsaacLab 而假拒绝；v4 改用
  `find_spec` 只验 exact module origin。v4 再在 scene 构建时发现 ignored A3 资产缺失；失败 claim
  保留，v5 从 clean `6d93bcb` 恢复并绑定 source/target `46` files、`15,378,264` bytes、tree SHA
  `0137f59b...26c6`。专项 `23 passed`；v5 Pod launch 尚未记为完成。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- 有符号拍面修复后的首轮消融已从 E0 设计升级为 machine prereg：同卡只跑 seed3 的
  hot/fresh × face-guidance-off/on 四个因果格；热启动明确保持 lineage0，fresh 必须 lineage1，半写
  claim/no-clobber/缺失 Git checkout 均 fail closed。focused `23 passed`；L1 尚无 Pod 行为结果，L2
  在 signed directional checkpoint paper 的 path/SHA 冻结前硬阻断，也没有 judge/真机授权。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- S0/M0 exact post-GVHMR handoff 已在证据机完成，分别为 4,970/9,242 bytes、SHA-256
  `d57a93e0...a1054` / `60c55150...088ef`。下一层 canonical-beta 已做成两份独立 no-clobber prereg：
  复用旧 materializer 的 PT/save-reload 审计，只注入旧 Franco exact donor，不重算新 cohort。host static
  与新旧专项为 `15 passed, 1 skipped`，最新 main 重放回归 `620 passed, 9 skipped`；真实 PT 的后续
  consume 已按本节首条完成，GMR/schema-2/安全/效果/训练仍未授权。
  M0 的 foot sites、初末二维脚距、容差和 pass 全保持 null，必须由未来 exact GMR 产生。详见
  [canonical-beta 卷宗](experiments/motion_canonical_beta_s0_m0_20260713.md)。

- S0/M0 的五条 exact GVHMR 结果已增加 post-GVHMR no-clobber consumer：两份 prereg 同时绑定 tracked
  summary、execution record、queue state、每条 binding/audit/PT 和 canonical-beta donor，host static
  两批通过，专项 `8 passed`；后续 runtime handoff 与 canonical-beta consume 已按本节其他条目完成，
  GMR/schema-2 仍未运行；S0
  禁止借用拉球题，M0 后续必须恢复含前后错位的初始二维脚间向量，双脚并拢不算成功。详见
  [实验卷宗](experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)与
  [操作文档](operations/run_motion_post_gvhmr_exact.md)。

- Franco 动作主线第一次从“排队”进入 runtime：Pod1 上 S0 高点拍压 `88/88` 帧、M0 四条横移
  `105/105、97/97、82/82、96/96` 帧全部通过 exact GVHMR finite structural audit；输入、execution
  record、queue、output、binding 和 audit SHA 已进入
  [`motion_video_gvhmr_s0_m0_results_20260713.json`](../configs/motion_video_gvhmr_s0_m0_results_20260713.json)。
  同时 signed spatial-retarget 对真实 v5 输入完成 640-cell screen，反手拉 B/C 分别产生 `19/3` 个
  bounded proposal，但 certificate 仍是 `0`，所以只解锁物化/安全门，不解锁 TOPP、RL、Gate3 或真机。
  详见[GVHMR 小批](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)和
  [空间重定位](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- 反手拉 B/C 的 signed spatial-retarget 首次对真实 v5 输入点火，在生成 proposal 前抓到验证器
  schema 假拒绝：`capture_table_pose_observed=false` 位于 `frame_contract`，而旧代码误从只含
  path/bytes/SHA 的 `frame_contract_evidence` 读取。修复后仍同时绑定 evidence SHA，且缺失/true
  fail closed；新 prereg/tool SHA 为 `0f757c8c...af66a` / `d053dd50...5259b`。这只解除输入验证阻塞，
  尚不是动作晋级。详见[动作空间重定位实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- 复盘 Phase-1 的 GPU 证据购买方式：`SZ` 在 2k 已失去稳定性资格后，把四 seed 都继续买到 4k
  对拒绝 baseline 属于过量复现。新制度改为一个阻断 seed 先跑四个不同机制单元，固定
  相对 `+200/+500/+1000` checkpoint；只有胜者和匹配对照补第二 seed，`3–4` seed/terminal 只给正式候选。
  第一张新纸是“热启动/从零 × 线性拍面引导关/开”的四格；当时只有 E0 设计，尚未启动 Pod、训练、
  judge 或真机。详见[机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)和
  [算力制度](research/phase1_ablation_acceleration_2026-07-11.md#seed-是晋级税不是首轮并发单位)。

- signed-face 诚实门已在 feature source 闭合：Isaac virtual reward 与 NumPy/MuJoCo analytic
  scorer 都在 `orient_normal` 前绑定 raw-A、每 clip `[+1,-1]` physical-B 和严格 +X/hemisphere
  门；`n/-n` 负控证明同一冲量/落点下错面不再记分，旧 unsigned 路径只能显式 inexact。seed3
  TensorBoard 九个 milestone 的 content-bound 摘要又显示正手误差 `174.02°`/normal pass `0` 时
  训练回台仍 `.965`；实际 `env.yaml` 绑定启用的 face-blind reward 及 `20/30/5/5` 权重。step13800
  的 `2.961×` 只是跨环境/正反手的全局 reward-tag 比值，不能量化正手错面支付份额；准确结论收紧为
  “wrong-face FH states were treated as reward-eligible by the active face-blind reward path”，而非
  “已量化错面支付”或单因素因果。focused 为 `38 passed, 1 skipped`，
  顶层 broad 为 `546 passed, 9 skipped`；
  没有 simulator/Pod/真机行为结果，fresh canary 与同卷复判仍待做，G05/G06 保持 `Partial`。详见
  [拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- Fresh `SZ model_4000` 四 seed 同一 K100 已通过 Linux fake-runner 冒烟、两 Pod 一次性持久
  启动和正式 aggregate 完成：`50/88/98/0`，median `.69`、worst `.00`、spread `.98`、
  worst-side `.00`，四项稳定门全失败；seed4 有 21 次 root fall，判为持续弱而非晚熟。
  seed2/3 正手 parsed `38/50,48/50` 但 signed composite 均 `0/50`，法向误差
  `172.33°/174.35°`，所以旧高分不晋级。aggregate file/content SHA 为
  `1ba88e39...d195` / `226e6050...648d`；详见
  [稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
- 动作离线顺序纠正为 Franco 主线优先：六段旧素材复用 exact GVHMR/GMR，不再重跑；反手拉 B/C 的
  frame 49/50 只登记为空挥名义视觉锚点。新视频拆成互不阻塞的 [S0/M0](DEFINITIONS.md) 离线结构批：高点拍压单条与四条横移候选，
  v12 本轮不授权。两批绑定 GVHMR/权重/Python/`nvidia-smi`/validator/argv，并用 batch-only source
  fd 生成私有只读快照供 child 消费，再以 inode/mtime/ctime/SHA 复核；同时拒绝 symlink、原子 claim
  不相交 state/output。07-11 旧 launcher 已压成仅证据 gzip，不再提供通用入口；M0 未来终点必须恢复初始、
  朝向对齐且含前后错位的双脚分离向量。Host 聚焦套件 `50 passed`，仓库 `tests/` 为
  `573 passed, 9 skipped`；本分支未复制 Pod、未启动
  GVHMR/GMR/simulator/RL/真机。见
  [实验卷宗](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)与
  [操作文档](operations/run_motion_video_gvhmr_prereg.md)。
- Phase-1 fresh 广度池分两波完成负责人批准的运营收口：16 臂已全部在保留日志并验证最后
  checkpoint 的迭代、`1,762,715` 个浮点元素 finite、schema-3、fresh lineage 与相邻合同 SHA 后，
  只按各自登记 PGID 停止。第二波前又确认 24/24 最近 K20 格的正手 signed composite 都为 0；
  TERM 未退出时仅在确认无 live child/Kit-lock holder 后对同一
  exact PGID 使用 KILL，没有 broad kill、worker/judge 信号或真机命令。这不是预注册 q10/q50
  阈值停止结论，旧 `screen_only`/`whole_arm_stop_allowed=false` 语义不变；完整曲线、PGID 和 checkpoint
  SHA 见[拍面×plant 广度实验](experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。
- MuJoCo pelvis 点/轴 frame 审计在 `codex/mujoco-com-reset-frame` 修正两处源码合同：每个
  motion clip 显式声明 COM/link-origin 线速度点，teacher-reference 只对 COM 做 rigid-point
  转换，含糊的旧 inexact 包拒绝 reset；actor `base_ang_vel` 从 MuJoCo inertia-principal axes
  改为 pelvis link/IMU axes，并对 pelvis 自身恰好一个、零地址 freejoint fail-loud（不禁止球等
  其他 free body）。真实 A3 MJCF 的 formal CPU group 为 `115 passed, 0 skipped`，完整合同 union
  为 `183 passed, 0 skipped`，支持的根目录 `tests/` 为 `554 passed`；10 秒 plain-MuJoCo PD stand
  为 `1.816 mm` z 漂移、`0.311 deg` 最大倾角、
  双脚接触 `100%`。没有在 Pod/vendor backend/真机上运行 policy rollout；ready-state 四格仍未运行。
  两轮独立 review 复核公式、MuJoCo BODY/XBODY/freejoint 语义、mixed/count 负控和 standalone
  old-donor 兼容后均无 P0/P1/P2。
  另登记 vendor ROS 非零 `SimReset` world-angular→body-qvel 的潜伏接口 bug，当前全零 keyframe
  路径不触发。详见 [G06](gates/G06_isaac_to_mujoco.md) 和
  [frame 合同](interfaces/frames_and_coordinates.md)。
  同日只读复核用户给的两个 Pod：一台 SSH 握手连续 reset；另一台 3 张 RTX 5090 全空闲、无
  train/eval 进程，`/workspace/franco/nohope` 停在 `16a94b1`，其未刷新的 `origin/main` 也仅到
  `7b85546`。所以这两台当前都没有运行或验证本 ticket，不能把本地源码通过当成云上训练结果。
- exact planner-policy tuple 源码已在 latest-main 集成候选中闭合：23 项有效源码/配置逐字节匹配
  `c0a8e46`，portable Release 为 focused `40/40`、native `233 passed + 5 optional skips`，主线本地
  回归为 planner `180 passed, 2 skipped`、serve `39 passed`、root `521 passed, 9 skipped`。这只关闭
  source/binary merge blocker；ROS/Jazzy/AimRT、formal ONNX runtime、backend first tick、vendor
  MuJoCo 和真机都未运行。详见
  [实验卷宗](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md)。
- 最新 main 登记并在本地逐字节核验了 7 段私有新视频：v12 正反手挡球、高点拍压第五动作，以及
  左右横移各两段下肢老师。新版 intake 合同能区分挥拍与横移动作，拒绝重复 JSON 键、非有限数和
  角色/动作错配；7/7 文件与 11 项专项测试通过，仓库测试为 `472 passed, 9 skipped`。同时建立了
  [动作组合设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md)、
  [视频 intake 记录](experiments/motion_video_intake_v12_static_motion_20260713.md)和
  [非击球臂消融](experiments/non_striking_arm_imitation_ablation_20260713.md)。这只证明素材登记和
  设计已落地；没有复制到 Pod，也没有 GVHMR/GMR、仿真、RL 或真机行为结果。
- 按负责人阅读路径重写 [NOW](NOW.md)：先解释题目、参考动作、179 维输入、31 维关节目标、
  Reward、PPO 和独立判卷怎样组成一套完整训练，再按现行课程逐段写问题、解法、效果与差距。
  同时纠正阶段编号：阶段 2 是虚拟球变到达状态，站位/脚步是其中的解法；阶段 3 才是物理球
  进场；连续恢复和 `Gate3/Gate3B` 分别是横向能力线和部署验证线。成绩卡明确为 Python
  BankExam 单拍解析诊断，不是 Gate3。本次只改文档，没有新增训练、仿真行为或真机结果。
- Fresh `SZ model_4000` 四 seed 同卷的 Pod1/Pod2 readiness audit 与 all-four activation
  已物化（activation file `9dea76c2...ce704`，content `eaa92ca2...aa4fb`），两 Pod
  `contract-check` 通过。随后两份 no-clobber runtime contract 已完成 `prepare`；Pod1/Pod2
  file SHA 分别为 `2b76a5a...8201e`、`dbecc102...d1c9b`。当前仍是
  `prepared_not_started/jobs_started=0/auto_start=false`，没有 run、judge、新分或真机动作；该
  readiness/prepare 事务当时未发 trainer signal，后续 8 臂运营停止是本节首条记录的独立决定。
  持久监督器 source gate 后续已审绿，仍缺 Linux fake-runner smoke 与正式 job。详见
  [Fresh SZ 稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
- `model_4000` 同卷启动新增一次性、无覆盖的持久监督器：父进程只在核对 PID=PGID、procfs 身份、固定环境和完整 SHA 闭包后发布不可逆 token；token 可见后的超时、证据 `stat` 或临时清理异常都只能报告 committed-pending，不能产生重试权限。supervisor+queue+consumer 为 `64 passed`；这仍是 host 源码门，Linux/Pod 与 MuJoCo judge 尚未运行。详见[执行卷宗](experiments/phase1_fresh_sz_model4000_q50_20260713.md)。
- Native MuJoCo feasibility/implementation 已确认为 P0，但不阻塞几天内 `Gate3-D0`。off-main
  preflight `6e5fce3` 的 63 项 focused test、顶层 `468 passed, 9 skipped` 和七个 false 授权位
  证明 fail-closed；red team 同时抓出 action trace、source alias/exec、strict JSON、MJCF
  `strippath` 四个高优先级正确性缺口，所以当前 `NO-MERGE`。single-env core 未来还必须过
  N=1/8/32/64 与 48 小时留 30% 余量的吞吐继续门。它不是 trainer、`VecEnv`、PPO smoke 或训练结果，详见
  [实验卷宗](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)。
- 正手拍面复核纠正了“所有 seed 都约 170°”的旧说法：model-2000 seed1/2/3 raw-A 误差为
  `171.10/172.94/173.39°`，seed4 没有正手 exact strike；解析回球器的 `orient_normal`
  可能抹掉正负号。signed-face 诚实门通过前，旧解析高分不用于晋级；详见
  [拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- 连续拍等待/恢复设计经原文和现役代码复核后收紧：T0 按周期换题，T1 只改事件驱动结构并冻结
  reward，T2 才允许 learned shaping；随机到球先作为环境轴。若 T1 失败，先做平衡债/ready
  potential 的配对 `2^2`，第三 critic 只有独立校准并通过隔离 q50 后才能进入 `2^3`。
  这次只收紧文档设计边界；现有 machine prereg/validator/operation 仍固定旧三 reward/full `2^3`，
  必须另做内容寻址同步后才能点火。没有训练、simulator、Pod 或真机行为结果；详见
  [连续时序审计](research/phase1_continuous_rally_timing_2026-07-11.md)。
- 新动作的冻结站位 `0/64` 不再解释为“动作无效”：正式问题是动作自身安全触球流形 × 适配来球/动作题族
  × 合法整轨 `SE(2)` 站位。反手拉 B/C 仍只到重定位候选，挡球需另出题；没有重跑 screen。
- `main@3c7e507` 先补回了缺失的 INDEX 和实验账骨架；本分支把它升级为中文一站式路由、
  术语人话表、逐实验卷宗、精简 NOW/TIMELINE/PROGRESS、唯一队列和算力纪律。合入 `main`
  前，新版 NOW 仍只是一份提案。本次文档迁移没有运行训练、simulator、Pod 进程或真机。

## 2026-07-12

- 完成 native MuJoCo `Trainer-v0` 只读 preflight：现役 vendor main 的 sim loop 没有球/球台/网，
  所以首卷只做单拍 balance/strike-state fine-tune；reward 用独立 replay oracle；warm start 只载 actor，
  critic/optimizer 全新。没有启动 backend、sim、Pod 或真机，详见
  [preflight](research/mujoco_training_v0_preflight_2026-07-12.md)。
