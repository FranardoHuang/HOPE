# ActionBall 双后端长跑：当前执行 TODO

> 状态：`r36-v5-dual-learning-live / model600-visual-negative / contact-prior-candidate-validated / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-31
>
> `origin/main:docs/NOW.md` 是全项目唯一优先级权威。本页只维护
> [FullMDP](../DEFINITIONS.md)（完整球路、击球、落点与恢复状态机）单动作双后端
> successor 的依赖顺序、运行事实和完成条件，不维护竞争性的优先级队列。旧的单动作执行页已转为
> [只读历史账](action_ball_single_action_dual_backend_todo_20260817.md)；本页过去的 superseded 章节已移到
> [双后端 TODO 历史归档](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md)。

<a id="fullmdp-v6-todo-current"></a>

## 0.7 2026-08-31 current：R36 v5 直接修根因与双端学习闭环

本节是唯一现役局部执行合同。当前 GPU0 Isaac 严格绑定 `5dd39786`，GPU2 Mu 绑定
`e5c02ea6`，都不热补。GPU1 已从 fresh clean exact `67109ec2` 启动后继 Isaac：它只合入
已做完整等价对拍的 Reward ledger 批量转录和 `300 update` 存档频率，不改 PPO、Reward
经济或自然课程。所有结论限定为 `diagnostic_unauthorized`。

### 已采用合同

- 课程是自然重叠的 `balance → mimic → hit → landing/recovery`，不新增硬 Stage 或 success Gate。
  上一层基本形成时，下一层必须已有真实 eligible 入口。
- physical-ready 与 teacher truth 分层：前者是闭环可站稳的 A3 姿态，后者保留 motion 真实
  frame0/速度。准备窗内 Motion owner 冻结 teacher frame0，不改写 motion 假装零速。
- D05 到期即发布 frozen ActionEpoch；physical launch/TTC 相对 public reveal；R03 唯一读取同一
  ActionEpoch 的击球位置、速度和有符号拍面。
- Phase1/2 做几何与连续解，Phase3 只排 seed，Phase4 才用最终 A3 plant、qdes/torque/support、
  racket-site、桌网球物理与 recovery 准入。v5 bundle 同时绑定 teacher、physical-ready、nominal-hold、
  exact ball physics 和 compiled Mu plant。
- Observation V3 保持 actor/critic `215/231`；本轮没有新增 actor 字段、Gate 或 oracle。
- checkpoint 固定机位回放必须安装训练时的同一份 dynamic-ready binding，并把影片写到
  fresh no-clobber 目录；否则画面属于另一个出生 MDP，不能用来判断学习。
- 四项 measured-paddle prior 在全动作段保留原始 `1x`，只在真实 playback 的接触邻域按同一
  time-to-contact 钟连续升到 `4x`：`|TTC|>=0.12 s` 为 `1x`、`0.06 s` 为 `2.5x`、接触为
  `4x`。这不是新 Stage/Gate；它防止长动作中大量容易的非接触帧淹没真正决定击球的接触窗。

### 已验证的当前真值

- Phase4 nominal hold 在真 Isaac 完成 `1.2 s / 240 physics / 60 policy`，双脚接触持续，无
  hard-limit/table/fall，root-z 下界 `1.066056 m`，最大 tilt `0.032642 rad`。
- 固定视角已导出 raw reset、atomic physical-ready、teacher start、contact 和 recovery。raw env reset
  姿态可见不是训练起点，atomic write 后才是实际反手 ready；contact 帧的球/拍心/教师挥拍目标
  对齐。这是工作流证据，不是新 Gate。
- `5dd39786` 的 Isaac/Mu 真 CUDA fixed-action 都完成 `512×48×31`，done/timeout=0，同一
  action tape。Isaac update 276--305 在三卡并行负载下 p50/p90=`19.55/19.76 s`；这是当前运行
  wall-time，不把启动期 `~7 s` 或单卡早窗 `~10.7 s` 冒充现态。
- `e5c02ea6` fresh Mu 61-update diagnostic 自然完成，RC=0；每轮 `24,576/24,576`
  UID/identity rows 正确，storage finite、reward conservation/fact-integrity 无故障，p50/p90=
  `6.667/7.153 s`。同 source 已从 fresh root 进入 100,000-update Mu long。
- GPU0 Isaac 的 update 849--948 持续 finite；该 100 轮 episode 平均 `464.011 tick`，
  physical observed=`4,735`，timeout/tilt/table=`4,717/439/140`；但 R03/contact/crossing/landing
  仍=`0/0/0/0`。同窗 playback 误差为
  `0.5910 m / 0.3327 m/s / 0.3364 rad / 0.2399 rad`。从 update 500 起多数窗口已接近
  `500 tick` horizon，故这已不是“太早看不出”：当前旧配方明确卡在接触窗位置/姿态质量。
- GPU1 fresh Isaac 的 update 412--511 持续 finite；该 100 轮 episode 平均 `347.648 tick`，
  physical observed/R03/contact=`4,000/0/0`，tilt/table/timeout=`2,228/899/3,808`，四项误差=
  `0.5304 m / 0.3731 m/s / 0.3087 rad / 0.2812 rad`。它相对 update 0--299 的
  `119→147 tick` 已明显学会更多 balance；但该臂与 GPU0 同 seed，且 update 0--499 的学习账逐窗相同，
  所以它是实现等价/耐心对照，不是独立 seed 的科学复现。
- GPU1 到 update 700 的单轮 mean episode 已到 `500 tick`，`99.8%` 是 timeout，
  physical observed=`55`，R03/contact=`0/0`。`model_600` 的同合同 400-step 固定机位回放
  RC=0，影片 SHA256=`0dd2b23d460fa1fdaa88d7d9e33d2a328bf1bc1f141f80e703c66c9b5b08b7a8`，
  Pod 证据目录为 `/workspace/franco/evidence/r36-policy-u600-fixedcam-6Ao1oR`。画面前段不是
  完全静止，但没有形成老师反手挡的接触窗动作，后段明显向桌面倾覆且无恢复。
  单 env 回放不代签 512-env 分布；二者合读证明批量 balance 已形成、轨迹稳定性仍脆弱，
  mimic→hit 负例更强，不支持继续用“太早”解释。
- GPU2 Mu 的 update 1879--1978 持续 finite；该 100 轮 episode 平均 `369.536 tick`，
  launch/R03/raw/selected/crossing=`4,938/4,890/3,283/465/464`，selected/launch 约 `9.42%`，
  profiler-off wall p50/p90=`5.902/6.336 s`。hit 入口继续真实形成；但 legal landing/recovery
  仍=`0/0`，所以只能称 mimic→hit 在推进，不能称 landing 已学会。三条均为
  nonfinite/conservation=`0/0`。

### “早期少动”与 Build4 的公平解释

- fresh Gaussian policy 在前几百 update 先把动作幅度压小、减少倾倒并延长 episode，是这个自然课程的合理
  暂态：只有活过 teacher/contact 时钟，mimic 和 hit 才有稳定样本。当前 Isaac 0--399 的 episode
  `119→171 tick`、速度误差 `0.817→0.520 m/s` 同时改善，支持“先学 balance”，不支持“策略什么都没学”。
- 该解释有截止条件：episode 已长期接近 horizon、due/observed 分母充分后，R03/contact 仍为零就是
  mimic→hit 负例。GPU0 已越过该截止，不能继续用耐心替代 treatment；GPU1 尚用于确认同一学习轨迹，
  下一条 contact-prior 才是 matched treatment。
- `origin/build_4@324e60d1` 不能作为 fresh 早期动作对照：其 manifest/YAML 强制从 Build1
  `model_21800.pt` 位级热启动全部 8 个 actor tensor，只重置 sigma/critic/optimizer。仓库与 Pod 当前均无
  原始 Build1 run 的 model0/早期 checkpoint 序列，所以“Build4 一开始就会挥”只证明热启动 actor 会挥，
  不证明 fresh policy 应当立刻挥。后续只吸收 direct-paddle 与连续暴露原则，不兼容 Build4 checkpoint ABI。

### 已直接修复

- [x] reveal-relative launch、ActionEpoch 单一 target、Reward 同写者自证减法、filtered-net plant repin。
- [x] Phase2/3/4 seed/admission/final-plant authority 分层，safe follow-through、physics 绑定与 20 ms deadline。
- [x] physical-ready/teacher truth 分离；删除旧 MotionLoader 232 行重复字节/前缀仪式。
- [x] 允许真实 moving teacher frame0；时钟冻结由 Motion owner 表达，不伪造 motion 数据。
- [x] 零宽课程多个 layout slot 可共享同一物理 landing center；binding 仍对 slot+semantic 绑定。
- [x] Mu runner/consumer 不再各自手抄 action UID；`e5c02ea6` 直接读同一 portable catalog，
  并继续在每个 update 检测真正的 row-wise 漂移。Pod focused=`275 passed, 1 skipped`。
- [x] 删除从未进入 runtime graph 的 Racket observation token/registry：它只有测试自产自销，
  production binder 永久 HOLD，却残留 capability/record/clone/stale 校验和 semantic exclusion。保留真正
  在线的 ActionEpoch selected-rubber、cold reference table 和 D05/R05/generation truth。
- [x] Reward28 每个 control step 的 28 次独立 row 转录收敛为一次 batched close；61 个
  ACK JSON 全量递归逐字节相等。Isaac GPU1 profiler-off p50/p90 从
  `11.060/11.291 s` 降到 `10.755/10.943 s`（`-2.76%/-3.08%`）；这是真实小收益，
  不冒充主墙已消失。
- [x] checkpoint cadence 改为 `300 update`，专用于尽早固定机位看真实 policy；
  `206 passed, 2 skipped`，不增 Gate，不改学习数学。
- [x] 修正真实 policy 视频工作流（`d8fc5d12..5b32f42e`）：`play.py` 复用训练的 FullMDP
  owner factory、typed PPO identity、dynamic-ready resolver/loader 和代码拥有的 motion catalog；
  capture 路径不再误触发 ONNX export，且只接受当前 RSL3 grouped observation，不保留旧 actor-tensor
  兼容支路。显式 `video_dir` 只能创建 fresh no-clobber 目录。Pod focused=`7 passed`，真实
  `model_300` 400-step 回放 RC=0；影片 SHA256=
  `3e5c3a58fe831b3fd362f4ece02fab3bcbb4d1c95540569ef343f134bc6da13e`，证据目录为
  `/workspace/franco/evidence/r36-policy-u300-fixedcam-gDJxLV`。
- [x] `dad61048..00814042` 将同一四项 paddle prior 从 playback 全段固定 `4x` 改成接触窗连续
  `1x→4x→1x`，Isaac 和 Mu 共用纯 tensor 核与各自同义 TTC；不增 observation、Stage 或 Gate。
  Pod exact `00814042`：共享/Isaac `368 passed, 26 skipped`，Mu
  `214 passed, 6 skipped`。数学与接线已闭合；学习收益仍须 fresh 小预算 canary，不能移签给旧 run。

### 当前唯一执行队列

- [x] v5 bundle/consumer、nominal hold、固定视角帧和双端 CUDA fixed-action。
- [x] Isaac fresh O_EXCL long：
  `/workspace/franco/runs/fullmdp-r36-v5-5dd39786-isaac-h48-20260831T0939Z`。
- [x] Mu 精确诊断为 `uid_rows=0/24576, identity_rows=0/24576`；根因是 runner/consumer 还手抄
  0807 旧 UID，不是 reset row 破坏 identity。`e5c02ea6` 已改从 portable catalog 取唯一真源。
- [x] `e5c02ea6` fresh Mu diagnostic 61-update rate window：RC=0，p50/p90=`6.667/7.153 s`。
- [x] Mu fresh long：
  `/workspace/franco/runs/fullmdp-r36-v5-e5c02ea6-mujoco-h48-20260831T1034Z`。
- [x] 等价加速与 `300 update` 可视存档的 fresh GPU1 Isaac long：
  `/workspace/franco/runs/fullmdp-r36-67109ec2-isaac-h48-gpu1-20260831T1153Z`。
- [ ] 双端按 update 100/300/1000 读同一自然链。
- [x] GPU1 `model_300.diagnostic_nonresumable.pt` 已导出 400-step 固定视角真实 policy 视频。
  画面显示策略不是出生即倒，也有周期性动作与存活/重置；但没有形成清晰的老师反手挡接触窗动作或可信
  击球，和同窗 R03/contact=`0/0`一致。确定性 teacher 视频仍证明中心题的球/拍心/contact target 对齐，
  因此下一步修接触窗学习信用，不再次改 task/球公式。
- [x] GPU1 `model_600.diagnostic_nonresumable.pt` 已用同一固定机位和 dynamic-ready 绑定回放。
  它排除“policy 全程不动”，但直接暴露了无清晰老师挥拍、单轨迹向桌面倾覆且无恢复；
  与 update 700 的 512-env timeout 分布并列记录，不用一个视频代签总体成功率。
- [ ] 任一 GPU 自然形成可用窗口后，从 `00814042` 或其文档后继的 clean exact source 发一条 fresh
  contact-prior canary；与旧配方只比较 matched update/window 的接触窗 p/v/face 误差、R03/contact
  和 episode/safety 分母，不用 total return 裁决。现有 GPU0/GPU1/GPU2 都是有用的自然学习对照，
  不为抢 treatment 人为截断；待独立 lane 可用再发。
- [x] 每小时守护 `task-first` 已从过期 R35 root 更新为上述三条 R36 live root；它按
  balance→mimic→hit→landing 的截止条件、固定机位 checkpoint 视频和删除式结构审计工作，不再把
  early stillness、Build4 热启动或新 Gate 当结论。

### 结构减法（不阻塞已启动 long）

- hot path 只保留数值真值和跨 owner 边界；删同 writer postcondition、每步 full-manager snapshot、可离线
  重建的转录和死 registry。完整字节证明下沉 launch/checkpoint，不常驻 reset/physics substep。
- 下一性能刀只针对 profile 已显示的 D05/reset/command 重复数据搬运：跨 owner 事实保留，
  same-writer 回声、per-env Python 转录和每步 D2H 下沉。monolith 拆分按真正 owner 边界渐进落地，
  不做一次性重写，也不用新 Gate 补偿臃肿结构。
- `hope_commands.py` 当前约 `32k` 行，已经成为审计和改动放大器。新 FullMDP 数学/状态所有权不得再
  塞回该文件；优先把纯 paddle/task 数学、D05 construction、metric materialization 按真实 owner
  抽成小模块。每次只迁移一个已有调用面并做等价对拍，不另造 adapter/gate 层。
- 性能减法保留 reason/counter/safety/durable truth，但不用“安全”为名保留同写者仪式或重复身份验证。
- dead Racket observation authority 删除后的 exact Pod 聚焦回归为 `171 passed`；历史
  semantic-surface fixture 的父提交假红已用显式 13-symbol source-evolution map 修复，未改 production digest。
- 当前 production 已无每步 full-manager snapshot，死 Racket registry 已删除，policy playback 也只保留
  当前 RSL3 grouped observation。继续删除真正无消费者的兼容与同写者仪式；不做兼容旧 FullMDP/checkpoint
  的 adapter 链。`build_4` 只吸收 direct-paddle、早期自然暴露等第一性原理，不复制其旧 runtime、混合
  warm-start/replay/sigma 或 checkpoint ABI。

### 继续与停止

- 继续：finite ACK，fault/nonfinite/conservation=0，自然链的真分母逐层出现。零分母写`未测`。
- 学习层未改善时先看 balance、mimic、launch、contact、landing 分母和画面，不以 total return 粉饰，
  不增 success Gate 掩盖时钟/target/plant 根因。
- NaN/nonfinite、真身份漂移、counter/WAL 破坏时保留 root 停止；早期 table/fall/hard-limit 只按预注册
  update 窗看趋势，不因一次早期均值停训练。

### 视觉复核是工作流，不是 Gate

修改 reset/reference/task/contact frame 后，在同一固定相机下导出 `raw reset`、`post atomic ready`、
`teacher start`、`contact window`、`recovery`。它用来直接发现坐标、姿态和球拍时钟错配；不要求
发射前新增一套安全审批，也不用截图代替真实 contact/landing denominator。

## 历史链接兼容入口

旧章节不再留在现役 TODO，避免历史流水遮住唯一可执行清单。以下 anchor 只用于保持旧文档链接可解析；
内容和证据都在[历史归档](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md)，不构成当前
运行 authority。

<a id="fullmdp-v9-superseded"></a>

- [V8/V9 及后续 superseded 章节](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md#fullmdp-v9-superseded)

<a id="04-2026-08-23-v4最终冻结与v5第一性原理自查"></a>

- [V4/V5 第一性原理自查历史](../archive/action_ball_dual_backend_longrun_todo_20260819_history.md#04-2026-08-23-v4最终冻结与v5第一性原理自查)
