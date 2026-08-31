# ActionBall 双后端长跑：当前执行 TODO

> 状态：`r36-v5-dual-learning-live / diagnostic_unauthorized`
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
- GPU0 Isaac 到 update 513 仍 finite；近 100 轮 episode 平均已到 `353.37 tick`，
  launch=`4,041`，timeout/tilt/table=`3,862/2,093/878`，证明 balance/survival 与 hit 入口持续
  打开；但 R03/contact 仍=`0/0`。同窗 playback 误差为
  `0.534 m / 0.370 m/s / 0.310 rad / 0.282 rad`。这个窗的样本已更多进入晚期 playback，
  不能用原始均值直接断言 mimic 退化；可确定的是 hit 仍未学会。
- GPU2 Mu 到 update 690 持续 finite；近 100 轮 episode 平均 `154.78 tick`，
  launch/R03/contact=`2/2/0`，tilt/table=`8,192/7,709`，仍主要败在 balance/mimic。两端都不能
  称 hit 或 landing 已成功；landing 仍为`未测`。

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
- [ ] GPU1 `model_300.diagnostic_nonresumable.pt` 出现后立即导出固定视角
  reset/pre-teacher/contact-window/recovery 视频，并将画面与同窗 balance/mimic/launch/contact 分母对齐。

### 结构减法（不阻塞已启动 long）

- hot path 只保留数值真值和跨 owner 边界；删同 writer postcondition、每步 full-manager snapshot、可离线
  重建的转录和死 registry。完整字节证明下沉 launch/checkpoint，不常驻 reset/physics substep。
- 下一性能刀只针对 profile 已显示的 D05/reset/command 重复数据搬运：跨 owner 事实保留，
  same-writer 回声、per-env Python 转录和每步 D2H 下沉。monolith 拆分按真正 owner 边界渐进落地，
  不做一次性重写，也不用新 Gate 补偿臃肿结构。
- 性能减法保留 reason/counter/safety/durable truth，但不用“安全”为名保留同写者仪式或重复身份验证。
- dead Racket observation authority 删除后的 exact Pod 聚焦回归为 `171 passed`；历史
  semantic-surface fixture 的父提交假红已用显式 13-symbol source-evolution map 修复，未改 production digest。

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
