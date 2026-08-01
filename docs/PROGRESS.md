# 简短进度记录

本文件只保留短日期摘要，不再做第三份实验真源。更新时只写几句话并链接到权威位置：

- 当前 setting/采用状态：[NOW](NOW.md)
- 实验设计与证据：[experiments/](experiments/README.md)
- `main` 上的重要变化：[TIMELINE](TIMELINE.md)
- 可复现验收：[gates/](gates/)
- 缩写与人话释义：[DEFINITIONS](DEFINITIONS.md)

旧 1700 行记录完整保存在
[历史 PROGRESS](experiments/archive/PROGRESS_legacy_through_2026-07-12.md)。

## 2026-08-01（ActionBall 双动作与自适应 σ source candidate）

- exact `db647517…` 的首轮 Pod plant suite 为 `65 passed, 9 skipped, 12 failed`；12 项均由
  required-identity 测试夹具仍写旧 full-precision armature 触发，新的生产 authority 正确
  fail-loud。当前窄修只把夹具同步到已冻结 hybrid plant，不放宽校验；successor 将在 Pod
  并行重跑 plant、push、integrated-gate 三组，全部结果返回前不记 PASS。实时下一动作见
  [ActionBall 准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。
- successor `ac64553c…` 的 Pod plant 已为 `77 passed, 9 skipped`，push 组在项目 venv +
  隔离 pytest-only shim 下为 `165 passed`。integrated focused 唯一失败是 Hydra append 形式的
  table diagnostic 未从 scientific argv 排除；工作树仅修该 exact prefix，提交后只重跑
  integrated 组，不为无关代码重复 plant/push。
- exact `42be696e…` 只重跑 integrated focused=`47 passed`；本轮 source-focused 三门全部闭合。
  下一步按身份 SHA 顺序重签双动作工件链与 A/B/C pins，然后只跑一次综合 `4096×5`，不再增加
  standalone push probe 或学习 baseline。
- r5 no-clobber 物化 epoch 已在工作树开启：双动作 registry planned paths 切到
  `20260801_r5`，identity source 与所有派生产物 SHA 置空。该中间态只能供 materializer 使用，
  launcher 必须 fail-closed；loop/block 身份 repin 将从同一 exact commit 两个 clean checkout 并行。
- Pod exact `173cb72d…` 已生成 r5 formal profile pins，SHA=`509f3812…f34c1`，与 r4 bytes
  相同但使用 fresh tracked path。identity-smoke 固定入口已切到该 r5 path；下一 commit 仍是
  fail-closed producer source，随后才允许双动作 identity repin。
- exact `cf3e07c2…` 的 loop/block identity repin 已在两个 Pod checkout 并行完成，三输出 SHA
  分别为 loop `365eb07b…/141062e2…/5ae2117a…`、block
  `c5a0ab79…/4f09af08…/93f591e4…`。registry 只回填 identity 层；runtime/authority/bundle
  继续为空，因此 successor 仍只能进入 recipe→smoke，不能训练。
- exact `8f7fcd91…` 的 loop recipe 在 env/PPO 前正确拒绝旧 r4 Reward SHA：当前 fully composed
  vendor leaf（coarse position + `action_acc=0`）产出 `71358fd4…`，不是 `8220f339…`。
  code-owned identity/static constants 与定向测试已同步；必须先在 Pod 验证后才用 fresh namespace
  重发 recipe，旧失败 namespace 永久 spent。
- **SUPERSEDE（current vendor N1 操作面）：**今晚 plant 已冻结为非冲突 parkour 新表 +
  task/SKU 三处 fallback：
  `waist-yaw Kp85 / waist-pitch effort118 / wrist-pitch,yaw Kp20 effort6
  armature0.0008100893338`，wrist-roll 仍 `30/24/0.004968`。当前只差 Pod；未来
  exact-SKU 直接确认 24 只产生下一版 plant，不热改今晚 run。
- 今晚固定 A=`bh_loop_c` static、B=`bh_block` static、C=`bh_loop_c` monotonic
  adaptive-sigma，全部 fresh-only。共同采用智元 `1–3 s` 六轴 velocity-only
  push，`force_push=false`、`combined_exclusive=false`。live stage 只有 `1×2` smoke、
  `4096×5×save1` integrated probe 与 `4096×20001×save100` long；每 lane 只生成
  一份 `stages.probe` receipt v2。standalone `4096×32 push_evidence` 已退役，旧
  spec/receipt 仅作 spent history。pre-long 只硬门 source/plant、194/318 real-runner
  normalizer roundtrip、finite checkpoint、std-LR、delay、joint actual-hard、qdes、
  nonfinite、completion 与 push event/applied>0+六轴 extrema finite/in-range；table/fall
  频率、strike-window、recovery 改为 20k 前 100 update telemetry，不再是
  5-update blocker。验收只在 Pod，本地不跑测试；MuJoCo P0 与架构
  P1/P2 不混入今晚 prelaunch source。详见
  [G05](gates/G05_isaac_training_first_loop.md) 和
  [发射工序](operations/run_ablation_wave_launch.md)。
- 本日下方其余条目按发生时间保留 historical/superseded 候选与运行事实。
  其中 `80/115/腕30-24`、旧 C1 以及旧 push-evidence blocker 不再是 current 操作面；
  不改写它们当时的运行结果，但也不用它们为新 plant 代签。
- 首次 exact `cc0020e2…` loop-static probe 在 PPO 前因 table-attribution Hydra 键缺少 `+`
  而 fail-loud，无 checkpoint/PPO；B/C 未发。同时 Franco 明确智元新 A3 表高于仓库
  旧 nominal，故当前 C1 及三 claim 作废。下一 clean source 将合并 launcher append 修正和
  `waist_yaw Kp=80 / waist_pitch effort=115 / wrists=30,2,24,0.004968`，随后从头重签双动作工件链与三 lane pins。
- 可迁移性审计发现 vendor leaf 被 `reward_pack=v2` 隐式注入
  `action_acc_weight=-0.05`；本轮改为显式 `0.0`并保留 `action_rate_clamped=-0.2`。
  MuJoCo 路线裁定前只收口可迁移科学合同和已发现的确定性错误，不新增
  Isaac/PhysX 专属 feature；新 source 的验收统一在 Pod，本地只作快速提示。
- clean `b57a9685…` 已采用智元新 nominal、修复 Hydra append 并钉死 bang-bang 配方；Pod
  nominal/task=`8 passed, 14 skipped`、launcher=`78 passed`。唯一 reward 失败是审计断言夹具，
  successor `ca365126…` 修正后 Pod reward 全文件=`249 passed in 1.96 s`。现只盘点六项跨引擎
  golden contracts 并等直接 MuJoCo/mjlab 尽调；旧 C1/r4 工件不得恢复或代签今晚长训。
- 六项只读盘点已收口：所选后端只补 31-D decoder+lag 联合 golden；直接 MuJoCo 再补完整
  194-row/frame parity；20k 前在唯一 `4096×5` Pod gate 中做当前 `rsl_rl` training-runner
  normalizer save/load roundtrip。sampler/adaptive-sigma 的完整 resume 延后，三 lane 明确 fresh-only。
  14:26 CST NVML 显示 Pod1 GPU0/GPU2 空闲、GPU1 有既有任务，Pod2 三卡均占用；不停止他人进程。
- MuJoCo 今晚路线只读复核确认：场景/单环境 evaluator 基础后来已有，但 native Trainer-v0 与
  mjlab A3 backend 均未接通 fixed-194/critic-318、Reward、VecEnv/PPO、batched reset、delay/push
  和 4096 吞吐；今晚默认 Isaac，只有 CC 交付 runnable `4096×5` 分支才翻转。同期发现先前只改
  腰/腕冲突仍未完整吸收智元 armature 表，现按全 29-DoF literal 并行修 robot、Isaac authority、
  MuJoCo replay 与 31-D decoder/delay golden；不做学习 A/B，head 保持具名 HOPE fallback。

- clean r4 `C0=ba195165…` 三个 zero-PPO 物化任务均 accepted 且自然退出：
  loop/block policy=`edfffec3…/44c20720…`，loop adaptive effective Reward=`6520f153…`；
  全部 0 PPO/0 checkpoint/未授权。三 pin 正回填以生成 clean C1，旧 C1 证据不代签 r4。
- `C_AUTH=cd2375c7…` 的 loop/block dynamic-ready 与 Pod nominal hold 均通过；两条 hold
  各执行 `0.8 s/160 physics steps`，无 terminal/truncation。接触 bundle 随后均 PASS，
  SHA=`a57c3ca3…/26931c76…`，已跟踪 candidate/hold/bundle 并回填 registry。待 focused
  gate 后提交 clean `C0`，再物化 A/B/C 三 pin。
- clean `C_RI=3c46a44d…` 的双动作 runtime authority 已并行物化，loop/block file
  SHA=`2b96d2ea…/226d3788…`，三类 authorization 均为 false，已回填 r4 registry。
  下一步是 validator/focused gate 和 clean `C_AUTH`，再进入 candidate/nominal-hold。
- r4 exact source `f0889ab1…` 在 Pod1 GPU0/GPU2 各自完成 loop/block recipe 与
  `1 env×2 update` identity smoke；四份 checkpoint finite，两份 live contract SHA=
  `912381de…/19d4cfed…`。两份 required identity SHA=`a4360e39…/78267461…`，已与
  runtime contract 一起跟踪并回填 r4 registry。下一步是 focused gate + clean `C_RI`，
  然后从该 successor 物化双 authority；这些仍是 diagnostic identity，不授权 long。
- Pod2 exact `fdc43396…` 已用 loop/block 两个 detached clean worktree 各完成
  recipe 和 `1 env×2 update` identity smoke；四份 checkpoint 均 finite，194-D/schema-3/
  31 关节/action-only、`[0,2]` 延迟和 ABI/std marker 均闭合。两份 live contract
  与两份 required identity 已在 `205a0c52…` 跟踪；registry 现回填四个 exact
  file SHA，A3 vendor 工件链回归 `118 passed`。它们仅完成 runtime 身份层，
  authority/dynamic-ready/nominal-hold/contact bundle 与共享 safety gate 仍 fail-closed；
  [权威账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)持续更新。
- clean `f0a949bc…` 已从上述双动作 exact runtime contract 物化 runtime-authority：
  loop/block file SHA 分别为 `04418ced…/648182df…`，已合入并回填 action
  registry；source/action/contract 与三类 authorization=false 均经 validator 复验，
  A3 vendor 联合回归仍为 `118 passed`。下一门是两动作 dynamic-ready 候选和
  Pod nominal-hold，未通过前不产生 contact bundle 或 long 授权。
- 本轮三卡发射的两个工件工具与一个机械压测已完成 source/host
  验收：live schema-3 contract + 12 组/31 关节 required-identity 固定路径原子
  物化器 focused `38 passed`；三条 code-owned N1 lane 模板 focused `59 passed`；
  waist-roll/pitch × lower/upper × Hctrl ON/OFF 的 8-env/5 ms 压测 focused
  `13 passed`；红队补齐 exact 12 组 consumer、发布前 source re-attest、exact task 和
  物化期测试 fixture 后整合 `113 passed`、无剩余 P0/P1。下一步是提交 clean source，再进入
  registry `sha256=None` 的显式物化 epoch；该中间态只能 fail-closed，不用假 SHA
  让 launcher “先跑起来”。Pod 上仍需真实 stress、loop/block 身份物化与
  `4096×5 → 4096×32` 共享安全门；G05 保持 `Partial`。
- 可测工具 source `5028500c…` 推送后，action registry 已切换为新 `_r2`
  identity/runtime fixed paths，loop/block 新产物 SHA、identity source commit 与 contact
  bundle 全部留 `None`。此中间态 materializer `38 passed`，训练 launcher 在
  contract SHA 缺席时预期 fail-closed；不得恢复旧 loop pin 或用假 SHA 让测试变绿。
- `_r2` profile pins 已从 clean `9a7429f1…` 产生并重算一致，SHA-256
  `df7fe0f038d79e3a89feebc638eea48290caa7e8cf85c4ddefe76ac310b9d3fe`。loop/block
  identity-bootstrap 已从共同 clean source `a2882d68…` 产生六个跟踪工件并回填
  registry；消费者已改钉新 profile/solver SHA，A3 vendor 六组工件链回归
  `115 passed`。runtime contract/required identity 仍为 `None`，训练仍 fail-closed。
- Pod2 首轮 runtime 物化抓到 identity launcher 解释器 P0：已钉住的 venv
  `/workspace/hope_isaac_venv/bin/python` 被 `Path.resolve()` 退化为裸系统 Python，导致
  formal pinner 在 Kit/PPO 前缺 `yaml` 拒绝。修复保留 claim 内 venv entry path，定向
  回归 `89 passed`。同轮 8-env stress 在 simulation start 后无输出占满 CPU 5 分钟，
  exact group TERM 后自然清空；只记 harness hang，不记 PhysX 结论。
- P0 nominal 对质反转：固定 plant 恢复智元 deploy/URDF/MJCF 原件（waist-yaw
  Kp `85`、waist-pitch effort `118`、wrist-pitch/yaw `Kp20 / effort6 /
  armature 0.0008100893338`），并将其余 29-DoF armature 从 parkour 四舍五入组
  恢复为 `a3_pingpong.xml` 全精度值。Isaac config、motion replay 与 runtime
  authority 现同表 fail-loud；Kp/Kd DR 接口、`[0,2]` delay、push 和 eval
  口径保留。这使先前 parkour-nominal 物化件失效，下一步必须按 corrected
  nominal 重物化所有 action-specific 身份链；G04/G05 仍为 `Partial`。
- 合并前红队抓到并修复两条身份 P0：probe-gate 已从退役 identity label 切到真实
  action-registry source，并封印 action-specific bundle/identity/authority/contract/sigma；identity
  repin producer 也改为 per-action pin，loop/block 可共存且 cross-action/drift fail-closed。
  独项/组合为 `62`、`48`、`128 passed`，全链合并回归 `382 passed`；仍待 clean-source/Pod。
- `bh_block` 的动作专属 registry、identity、authority、dynamic-ready 与 gate 链已形成
  **source-only candidate**：每层按动作选择自己的 motion/bundle/identity，缺少该动作的新物化
  工件时机械 fail-closed，不会借用 `bh_loop_c` 的旧 pin。host 定向回归 `117 passed`；尚未在
  Pod 物化或运行，未采用，也不改变当前 long gate。
- `bh_loop_c` 增加 fresh-only 单调自适应 σ canary 候选：位置/速度/拍面法向的宽核
  `0.20 m / 1.0 m/s / 0.52 rad` 只允许按双 term 锁步收紧至
  `0.075 m / 0.5 m/s / 0.262 rad`，static 路径保留 `0.30 m` 粗位置核；禁止 resume。
  host 定向回归 `322 passed`。其 Reward SHA 仍须从 clean source 走零 PPO 的 hash-only
  物化，再到 Pod 重钉完整身份链；当前同样是未采用、未 Pod 的 candidate。
- 上述 host PASS 只证明源码级 fail-closed/合同回归，不等于 Pod PASS。`bh_loop_c` 与
  `bh_block` 的 long 仍被 shared actual-hard 安全失败共同阻塞；不得用这两项候选绕过
  smoke/probe/push-evidence 门。权威运行顺序与状态见
  [滚动准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。
## 2026-07-31（ActionBall vendor N1 containment 与 long gate）

- replacement smoke 在 PPO 前正确 fail-closed：配置 policy SHA `9fbc61ad…`、实际
  `f76df202…`。用 exact smoke argv 做零 PPO 配方物化后，diff 只落在 dynamic-ready
  identity：recipe wrapper 仍固定 r2/v3 bundle，smoke 已正确使用 r3/v4。现将
  r3 bundle 提升为 vendor launcher 唯一 code-owned pin，recipe wrapper 直接复用同一常量；
  stale bundle 在 spec compose 阶段即拒绝。联合回归 `49 passed`，py_compile/diff-check 通过；
  该失败为零 checkpoint/零 PPO update，不构成训练证据。运行态与下一动作见
  [滚动准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#02-now--新智元训练物理身份重物化后-fresh-n1)。
- r3/v4 canonical recipe 在 Pod1 精确产生预期 `f76df202…`。后续 fresh smoke
  又在 PPO 前发现 completion payload 无法从 scientific argv 取到自身 claim SHA；
  该 namespace 仍为零 update/零 checkpoint，已核对 sidecar PID/PGID/starttime 后 exact `TERM`。
  修复使已验证 internal exec 用受控 env 传 claim SHA，trainer 拒绝非 64-hex 或
  cfg/env 冲突，避免把 claim 塞回它自己认证的 argv 形成 hash cycle。相关回归
  `121 passed, 1 deselected`；deselect 是 clean source 后按设计必须重物化的 tracked identity。
- 独立审查又拦住 completion claim v1：marker 有 claim，但 runner/checkpoint 仍收到
  `None`，后续 gate 必然拒绝。claim-v2 现在只在 vendor stage/contract 成对存在时
  于 runner 构造前解析一次 effective claim，同一值同时进 checkpoint 和 completion；
  普通训练及 vendor 半配置不读 ambient env。host 回归 `109 passed, 1 deselected`，
  主线联合回归 `124 passed, 1 identity-rematerialization deselected`；独立对抗复审
  PASS（core `90 passed`）。现可提交 clean source，然后只重签 authority/required identity。
- 旧 `89082b7c` replacement probe / push-evidence 都自然完成且 checkpoint finite，但分别有
  `4,873` / `37,417` 次 actual-hard，只作失败根因证据，不得放行 N1 long。
  新 source 用 `max_inward_until_nonoutward_v1` 收口 5% guard：危险方向跨 policy/substep
  latch，只到回到安全 envelope 且速度不再向外后先写一次 `q_hold`，下一 policy
  才恢复 nominal；raw actor、delay queue、Reward 和 actual-hard terminal 不变。exact-resume 已覆盖
  `delay OR containment`；广义/affected 回归为 `157/110 passed`。
- push 诊断 wrapper 已验 public binding 与 `asset_cfg` 透传，物理 writer/RNG/结果语义不变；
  device 端累计 event call/env coverage/nonfinite/六轴 min/max/bounds，focused `137 passed`。
- N1 long producer/consumer 已经独立对抗末审：自然完成 marker 只在 learn+env close+cleanup
  全成功后 exactly once 输出；checkpoint 以 `weights_only=True` 重放，机械验 actor/critic
  normalizer `194/318`、scalar std×31、RSL-RL origin/distribution、完整 `[0,2]` delay 合同、
  push 运行证据、安全/任务失败率与 no-clobber lineage。聚焦回归 `65 passed, 1 deselected`；
  deselect 只是旧 tracked identity 必须随 clean source 重物化。
- `dr_reward_external_diligence_20260731.md` 最新 §13 已全量落到
  [滚动准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)：N1 继续 fixed-domain；
  formal N5 前采纳 R1–R6；R8 作训练侧失败加权设计，R7/R9 只在前置闭合后再启用。

## 2026-07-31（ActionBall P0a 与三卡 N1）

- 尽调侧发现的 legacy `reward_pack` 发车雷已定谳并修复：
  `HOPEPingPong` / DeployParity / Hitter / HitterPure / Rally / RallyV3 都是不含
  v2 virtual-ball 直接项的旧 Reward 谱系，现显式钉 `reward_pack=v1`；
  RealSensor alias 同源继承 v1，ActionBall 及智元 vendor leaf 仍显式为 v2。
  真实 Hydra compose 覆盖 `7` 条 legacy + `2` 条 ActionBall 全过，
  DeployParity 与旧 full-observation task 的 `1 env×0 update` trainer dry-run 均自然
  `rc=0`、各仅一条 v1 marker且无 runtime error；余下谱系不再占 Pod Kit，
  由 compose + pack-expansion 回归守住。详见 [G05](gates/G05_isaac_training_first_loop.md#2026-07-31legacy-reward_pack-发车兼容修复)。
- 补上两条默认零开销的诊断通道：
  [`action_acc_jerk_probe`](DEFINITIONS.md#action-acc-jerk-probe) 让 DeployParity/HitterPure
  在不改 Reward 时仍能记 raw/封顶 jerk；
  [`implicit_pd_post_step_effort_proxy_probe`](DEFINITIONS.md#implicit-pd-post-step-effort-proxy-probe)
  只在显式开启时记 live-gain 的步末解析 PD demand，并明确不冒充
  PhysX 实际力矩或子步峰值。两个 cfg 槽位缺席/`false` 都保持 `None`，
  不构造 RewardTerm、不增加当前 vendor N1 热循环。定向（含 effective-Reward
  taxonomy）`48 passed`；邻接回归 `384 passed` 外仅余父提交已知的 `4` 个
  explicit arm-torque mock/backend fixture 失败。
- 关闭三处会误导发射者的配置真值债：HITTER-pure 源码注释不再把
  Kp/Kd 合并写成 `±15%`，现显式写智元 startup Kp `(0.8,1.2)` / Kd
  `(0.7,1.3)`；Hitter/DeployParity 不再沿用 IdealPD 时代的 DR 理由；两个
  production implicit-A3 YAML 的 `arm_torque_saturation_weight` 从名义 `-0.5`改为真实
  `0.0`。组装后 active Reward 仍为零，不改科学行为；显式执行器研究叶仍可
  override，backend compatibility receipt 仍记录 requested/effective 差异。两个相关
  test files 全量回归 `223 passed`；详见
  [effective Reward 因果账本](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md#2026-07-31implicit-a3-名义-reward-与组装真值对齐)。
- 智元 07-31 训练权威的 Stage A 已在 exact source
  `5665963e96bf75c677e7669efc58c449e0c04876` 完成 recipe-only 和 `1 env×2`
  identity smoke：schema-3 training contract SHA 为
  `98fa3239daba825f07d3997fb28f4564c92967536f2552e6bdc0f8772781366f`，
  `model_0.pt`/`model_1.pt` finite，delay/ABI/std marker 计数为 `1/1/2`，且
  authority 的 live-order bug 已修复。recipe 生成的 shared-ready policy SHA
  `27bf405e5677fe2e7bab6fcc15c166901734048dd334b8b0abc3a8ffef3ce416`
  只证明 shared-ready，不得直接充当 dynamic-ready recipe。新 `bh_loop_c`
  dynamic-ready candidate/nominal-hold/bundle SHA 依次为 `c831a4e6…`、
  `11c025dc…`、`9881c52c…`；hold 在 `0.8 s / 40` 步内 PASS，双脚接触为
  `1`，无 terminal。actual-authority receipt SHA 为 `f66a9e59…`，
  `required_identity.v1.json` 已物化为 `240f3757…`，本批 launcher
  同时 pin 这两个 SHA；物化/pin 已过 `90` 个非 Torch 定向测试并随本批跟踪。
  clean `f948a150` 已双验 authority/candidate；dynamic-ready recipe 与 vendor
  diagnostic smoke 亦已在后续 clean source 闭合，当前唯一紧距动作是
  `bh_loop_c`/seed0 的 `4096 env×5` vendor probe。`bh_block` 与当前 revision 的
  `long` 仍机械拒绝。
  G04/G05 仍为 `Partial`。新的 2026-07-31
  [外部尽调](research/dr_reward_external_diligence_20260731.md) 是当前口径，与旧常数绑定的早期审计只作历史证据。
  Pod1 GPU2 旧残留已按 exact sidecar `TERM`，GPU0/GPU2 可用；GPU1 缺 sidecar，保持未动。
- dynamic-ready recipe-only wrapper 已于 `2430fbb2` 跟踪，Pod 相关回归
  `83 passed`。首个 fresh namespace claim=`e37f8169…e32` 通过 authority 与
  schema-v2 pre-scene 验证，但 env 构造暴露 `MotionCommand` 只消费 schema-v1；
  因 kind mismatch fail-closed，零 recipe/零 PPO。自有 PGID=`1328514` 已精确 `TERM`，
  GPU0 回到 `18 MiB`。消费端修复 `e7787e25` 同时验证 schema-v1/v2 和
  v2 plant/delay，Pod `54 passed`。fresh recipe claim=`75f28f24…490c` 自然产生
  policy=`e408b845…c65d`；随后 vendor smoke claim=`be783ab7…ad54` 完成
  `1 env×2`，model0/1 各 83 tensors 且 finite，ABI/delay/std marker=`1/1/2`，
  无 Traceback/table/fall/qdes-hard/nonfinite。update1 有1次 waist-roll actual-hard（age=25），
  交由 `4096×5` probe 定价；GPU0 自然回 `18 MiB`。
- Wave-P 历史 push robustness 波终档统一为 `14` 条已真实发射、`4` 条
  never-launched；由于无统一终档裁决，整波标记为
  `closed_incomplete/superseded`、`no dose winner`。不补训、不补卷，仅保留
  directional evidence；本条是对 07-21 当日“12 条已上卡”运行快照的最终覆盖口径，权威终档见
  [Wave-P 实验记录](experiments/2026-07/EXP-P1-PUSH-ROBUSTNESS-20260721.md)。新 vendor
  `5–15 s` 逐轴 6-DoF push 另走运行收据门，不续接或重命名旧臂。
- 旧三条 stable-ready milestone1000 均已自然完成，checkpoint finite，但约
  `797–1043` 次 strike opportunity 下 capture/return 全为零；证据升为 E3 负结果，不买
  20000-update、不 resume 成智元 setting。新三卡路径只是 seed `0/1/2` 的单卡
  diagnostic；formal N1 仍因 action-set registry/trust/receipt 缺失 blocked。唯一运行看板见
  [分阶段准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)。
- update 热路最新 exact source `7f77ae5c` 已在 Pod1 GPU0 完成 host-only
  solver result + single-H2D 验收：focused `104 passed`，`1 env×2` 和
  same-seed `4096×5` 的 7 份 checkpoint 全 finite。五轮 wall 均值
  `6.700 s/update`（约 `14.67k environment-steps/s`），相对旧
  `12.341 s/update` 累计快 `45.7%`；reset、joint-safety 与 exact-behavior
  和 pose-OBB 基线逐轮相等。下一刀改为 fixed-tape `cq_n_iters=4/6/8/12`
  数值验收；formal per-reset receipt 迁到 checkpoint 粒度仍是 N5 前工作。
- diagnostic update profiler 已在 Pod1 GPU2 以 exact source `5e1443c4` 收口：`1 env×2`
  恰好输出 2 行，same-seed `4096×5` 恰好输出 5 行且三组逐 update JSON 与
  `4d631fb3` 基线逐字相等，五份 checkpoint 全 finite。五轮 collection 均值
  `10.3308 s`；总 collection `51.654 s` 中 `solver_solve_many` 占 `33.432 s`
  （约 `64.7%`），`pool_request_many` 占 `34.724 s`，而 install 仅 `0.202 s`。
  下一刀已据此改为 diagnostic-only compact/prevalidated task receipt，formal 不动；
  不先优化 packet upload、PPO 或桌碰。
- 第三批 update-wall 与两层 diagnostic rollback 裁剪已在 Pod exact source 收口数值门：
  commits `d2ec91e9 / 5f85cc58 / dbb7ce04 / 60a8e219 / 096afb7b / 4d631fb3`
  的三组逐 update JSON 均与同 seed 基线逐字相等，全部 checkpoint finite；但
  same-seed collection 均值 `10.3804/10.6618/10.7364/10.2878 s` 均未优于基线
  `10.0916 s`。相同五轮 reset 数 `0/267/3103/875/2101`，剩余粗税仍约
  `4.9--8.0 ms/reset-env`。下一步只做严格 opt-in 分段 profiler，再按最大段实现
  compact batched reset；不再磨 rollback clone、PPO 或 exact table。
  当前 TODO 与 SHA 只看[分阶段准备看板](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)。
- strict 1.1 倍同题带 P0a 已在 Pod 以 deep-counter parity 和 finite checkpoint 收口：
  same-seed `4096×5` 均值从 `17.088` 降到 `10.206 s/update`，改善 `40.3%`；反手拉同实现
  probe 为 `6.816 s/update`，随后 GPU1 milestone1000 已与 GPU0 的 1.1 倍来球反手挡、GPU2
  的旧谱系反手挡一起进入三卡有用 long。精确桌碰后端实测固定税仅约 `0.22 s/update`，因此
  保留 exact contact，box/prism 只作后端失效降级。diagnostic-only lean Motion timing
  validator 已在 exact Pod source `2c3a39fe` 通过 focused `22 passed`；相关三套为
  `66 passed`，另有 1 个在父提交同样失败的旧 event-contract fixture。当前性能 TODO 是等
  自然空闲槽补 `1 env×2`/same-seed `4096×5` parity/吞吐，同时继续批量化 reset
  broker/receipt 的逐 env Python；formal checkpoint 粒度 receipt 仍在 N5 前单独闭合。实时
  状态见[分阶段准备看板](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)，
  可复现证据见 [G05](gates/G05_isaac_training_first_loop.md)。

## 2026-07-30（A3 stable-upper successor）

- 第二批 update-wall candidate 已在 Pod 收口：exact `26c648d4` 将 diagnostic Motion timing
  安装改成整批 host handoff，并把 fixed-18-draw refill overlap 从 `O(K²)` 降为 `O(K)`；
  focused suite `187 passed`，fresh `1 env×2` 为 `2.67/2.45 s` 且 checkpoint finite。
  same-seed `4096×5` 的三组深层 update JSON 与第一批逐轮完全一致，但 wall 均值只由
  `17.14` 降到 `17.09 s`（约 `0.3%`），故正确性 PASS、性能 FAIL。当前加速明确没有收口；
  唯一下一动作是合并每个普通 step/strike step 的 host validation barriers，再按实测决定是否
  批量化 table/joint kernel storm。当前 milestone1000 不热补；实时 TODO 见
  [分阶段准备看板](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)。
- update-wall 取证确认 r4 稳态 collection=`23.79 s`、learning=`0.299 s`，仅约
  `4.13k environment-steps/s`；约 `9–11 s` 来自 reset 热路径，另有 `12–14 s` 固定
  ledger/同步税。fresh candidate 已把 diagnostic safety 收敛为 device update aggregate，并把
  reset identity 的循环内 `.item()` 合成一次批量 D2H；exact `c0747d59` 的 Pod focused suite
  已 `134 passed`。旧 solver pin 在真实 recipe 构造中正确 fail-closed，并已按 current source
  重钉 profile/bundle、物化 policy `569431a5…d0c0`。fresh smoke 为 `3.83/1.81 s` 且
  checkpoint finite；新 `4096×5` wall=`9.25/10.95/25.44/16.17/23.91 s`，与旧波的
  termination/strike/table/题带统计逐轮一致，但均值只改善约 `3.9%`。第一批补丁确认数值
  等价却只削掉小头；当前 long 不热补，下一批直接处理 VirtualBall/command per-step host
  barriers 与 reset broker/receipt 的逐 env Python。实时状态只看
  [分阶段准备看板](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)。
- Pod1 exact source `17c7258a` 已为 fixed-194 v2 重物化 profile pins
  `08c8f9c7…c6b4`、base bundle `ed9fa0f7…afef` 和 1.1 倍 fast-ball bundle
  `3c1076e3…c32b`，并以工件 commit `8729104e` 生成三份 canonical fresh r3 spec；spec SHA
  为 smoke `e1b63f00…5b8d`、probe `3b200542…dd34`、milestone1000
  `b0396fbe…d442`。第一版缩进 JSON 已在 namespace 创建前被严格 canonical-byte 门拒绝；
  规范化 smoke spec 随后由 exact `8729104e` launcher canonical plan PASS，claim
  `7f9d12ca…4002`。r3 已在 Pod1 GPU2 真实构造并验证 fixed-194 v2 与 dynamic-ready，但在
  PPO 前发现 spec 沿用旧 policy recipe `b7209710…077f`，实际 composed recipe 为
  `165645f5…bd9`；该 namespace spent。recipe-only 构造已物化 raw SHA
  `4b81c74b…7fb1`，fresh r4 三段 spec 也已生成；r3 已由 watchdog 自然收口并释放 GPU/锁。
  r4 smoke（claim `257c6ccc…d80c`）已自然完成两个真实 PPO update，iteration 约
  `2.75/2.80 s`；`model_0.pt` / `model_1.pt` 各 80 个 tensor，浮点/复数 tensor 全 finite，
  table/fall/qdes-hard/actual-hard/nonfinite/terminal reset 均为 0。随后 r4 `4096×5` probe
  也已自然完成，五份 checkpoint 全 finite；update 2/4 有 `1985/643` 个 strike opportunity，
  证明完整 preparation+击球窗可达。qdes-hard/fall 全零；前五轮 actual-hard
  `0/267/3103/861/2076` 只作为预注册的 update100/300/1000 趋势基线。同一 exact setting 的
  fresh `4096×1001` milestone1000 已用 claim `2710fd6f…d4f4` 在 Pod1 GPU2 启动并进入真实
  PPO；`model_0.pt` 的 80 个 tensor 全 finite。当前唯一下一动作是守护到 update100；状态只看
  [分阶段准备看板](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)。
- fresh ActionBall actor 合同切为固定 194-D
  `action_ball_table_pose_twist_heading_task_teacher_start_v2`，当前 trainer 不再创建
  `action_one_hot`；UID/slot 只留控制面，N5/N73 在 fixed-width continuous future-motion
  intent 前 fail-closed。Pod1 exact `0227cfe9` focused suite 为
  `391 passed, 12 skipped in 61.35 s`；current-source repin、fresh
  `1 env × 2 updates` smoke 与 4096-env probe 均已闭合，下一步只跑上面的千轮里程碑。
  当前 TODO 只看
  [分阶段准备看板](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)。
- exact `f2c54fc3` 的 frame-consistent 194-D stable-ready loop/block 已在 Pod1 各完成
  `1 env×2` 与 `4096 env×5`：14 份 checkpoint 全部 finite，smoke 全安全，probe mean episode
  约 `48–72` steps 并跨 `t_hit`；loop/block 已分别出现 `867/2268` 个 strike opportunity。
  第一次 PPO 后两者仍共同出现 waist-roll/pitch actual-hard，但旧 `4ff48b21` 对照到
  update100/169 曾降到 `14/11` 与 `3/3`，故不以五轮否决学习。Pod1 三卡现分别运行
  loop seed0 / block seed0 / block seed1 的 `4096×1001`；2026-07-30 19:39 CST 快照到
  update `219/574/186`，mean episode `104.88/481.52/105.90`，都持续产生 PPO update 且无新
  Traceback。三条当前 update 各有约 950 个 strike opportunity，但 virtual capture/return
  仍全为 `0/0`：block seed0 已把 table/fall/actual-hard 降到 `2/3/6`，却有 965 个 proposal
  全被 face gate 拒绝；loop seed0 post-strike fall 为 `887/946`；block seed1 table/actual-hard
  为 `590/325`。因此“窗口与 denominator 可达”已证明，但动作质量、signed-face/contact
  对齐和 seed 稳定性仍开放，不热改旧 run，也不写成可部署 policy。身份、传感器/物理后续边界见
  [分阶段准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。
- `origin/main@ddfaaa02` 的 OptiTrack 球物理拟合管线与双份 Isaac/MuJoCo YAML 已以
  `bed6661f` 合入当前分支，并在首次 byte pin 前纠正速度衰减曲线的旧示例注释。当前运行的
  exact `f2c54fc3` N1 bundle 仍绑定旧 profile，只作 contact/学习可行性诊断；formal N5
  前须显式选择新 YAML，并重物化 physics/solver/question bundle。Pod fresh worktree
  `9fdb909a` 的 observation/launcher/training-contract focused suite 为
  `314 passed, 9 skipped`。
- 首个 N1 actor 候选已收口为 194-D
  `action_ball_table_pose_twist_heading_task_n1`：177-D HITTER-derived 前缀内的 racket
  position/velocity 与尾部 signed face 统一到 yaw-heading frame，另加入相对桌体 XYZ、连续
  SO(3) 6D、yaw-heading 三轴 root-COM 线速度与冻结动作身份；base/racket task 继续是机器人
  相对 residual。table-hit reset 只隔离上一 episode final-substep 碰桌行的
  首份 PhysX stale report，非桌碰 reset 与 persistent 新碰撞保持可见。Pod 194-D actor
  smoke、`1 env×2`、`4096 env×5` 与 fresh finite checkpoint 已完成；formal receipt 热路径、
  部署 producer 和 bang-bang canary 的最迟边界见
  [分阶段准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。
- fresh successor 改为固定 194-D
  `action_ball_table_pose_twist_heading_task_teacher_start_v2`：用真实老师启动倒计时替换 N1
  恒为 `[1]` 的 one-hot；UID/slot 只留在控制面。历史 195-D source 的 exact `020dc8d9`
  focused suite 为 `390 passed, 9 skipped`，但未进入 PPO；formal N5/N73 在发射前另加固定宽
  content-derived future-motion intent。旧三条同宽 194-D 运行不重标、不 exact resume；v2
  真实 Isaac `1 env×2` 仍是下一构造门。
- Pod1 block seed0 已到 update608 并自然产出 finite `model_600.pt`（80 tensors；
  SHA `11bee491…8470f`）。mean episode=`440.77`，table/fall/actual-hard/qdes-forbidden
  为 `4/5/14/0`，说明出生与安全风暴基本恢复；但 951 个 strike opportunity 仍零 capture，
  其中 937 个被 face gate 拒绝，实际拍速 `0.2832 m/s` 对目标 `1.2793 m/s`。当前主问题已收窄
  为 teacher 击球位置/拍速/拍面学习，而非 reset 合同；继续到 update1000，不热改。
- Pod1 GPU2 的 fixed-action exact tape 验证了反手挡的 1.1 倍来球比较：保持原台面落点中心
  `2.555 m` 和原初始宽度，只把中心球速提到 `4.6615 m/s`，同一 ball→task solver 把老师速率
  降到均值 `0.7206`；4096 题中 `2763` 题 admitted，`1333` 题按 residual/teacher-rate 下界
  分账拒绝。该结果只授权 fresh diagnostic comparison，不满足 formal `95%` admission 门；语义和后续
  发射边界见[分阶段准备账本](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。
- 首次 fast-ball plan 因旧 r9 solver blob pin 与最新 source 不一致而 fail-closed，未创建 run。
  Pod 随后把 current-source profile pins、base bundle 和 1.1 倍 derivative 重物化为
  `9ccb9854…5788` / `0daa5bce…ace53` / `f2be2331…1a491`；physics 与科学参数均未变，
  下一提交再用包含新工件的 exact source commit 生成 canonical 三阶段 spec。
- current-source fast-ball plan 已通过；首个 smoke 在 PPO 前暴露 stable-ready guard 字段名
  错位：guard 读 `clip_names`，实际 canonical cfg 是 `clip_names_per_clip`，因而把合法 N=1
  错拒。失败 namespace 保留不复用；代码改为读取 canonical 字段并补充 fail-loud 明细，下一
  exact source 用 fresh r2 namespace 在 Pod 重发。
- r2 smoke 又在 PPO 前暴露 ObservationManager shape probe 早于 ActionBall timing lazy bind：
  `time_to_teacher_start_s` 读取未绑定 Motion tensor 并 fail-loud。getter 现先走既有 runtime
  bind；失败 namespace 保留。因同时删除 actor one-hot，旧 claim/spec 作废，需重新 pin exact
  `hope_commands.py` solver source、生成 fresh v2 spec/claim 后再发。
- action identity observation 收口：当前 N1 的 one-hot 恒为 `[1]`，不影响本轮来球域泛化，
  因此不再延误首个 policy；formal N5/N73 前废弃随 N 扩宽的 actor one-hot，改为 actor/critic
  共用、由 contact reference 内容生成的固定宽连续意图，并对 shared-ready 动作做混叠检查。
- Pod1 clean `eb2799b1` 的 table smoke 已完成 E2：32 个 A3 body×top/keepout/net/左右 post
  五列 matrix 全构造，五 role 均有真实 PhysX 正控，四个子步覆盖；五次 automatic reset 后
  table raw reason/ledger/force 全零。log SHA 为 `15c52d29…26aac`，unsupported/Traceback/FAIL
  均为 0；receipt 已入
  [`configs/n1_contact_dynamic_ready_20260730/`](../configs/n1_contact_dynamic_ready_20260730/table_smoke_eb2799b1_gpu1_r26.receipt.json)。
- nominal-hold 截图器现将未经 artifact 覆盖的 `raw_env_reset` 与
  `physical_ready_after_reset_write` 分开记录，避免把手工 ready 冒充原生 reset；同时修复
  `test_metric_sync_fix.py` 的旧 `__new__` fixture，补齐生产初始化必有的非 task-first /
  非 ActionBall / 非 planner 默认旗标及 exact-attempt buffer，未改生产逻辑。两轮 clean Pod
  run 依次暴露旧夹具缺失的 exact-attempt buffer（`12 passed / 9 failed`）与 planner 默认旗标
  （`19 passed / 2 failed`）；第三轮确认最后两项还缺 inactive actor-view 的既有 metric
  buffer。补齐后的完整复测与真实 Isaac 图像验收待完成。
- clean Pod 已确认旧 metric fixture `21 passed`、nominal-hold focused fixture
  `4 passed`。fresh checkout 的首次真实 Isaac probe 还暴露了两项纯启动依赖：绝对 URDF
  路径变化会重复转换同一 A3，且 renderer 需要 `libGLU.so.1`。现已物化 Franco-owned、
  no-clobber 的 preconverted A3 USD 与 private GLU 副本并按逐层 SHA 核对；诊断器也增加
  构造/reset/probe stage marker。首个 stage run 证明进程自然完成 gym make 与初始 reset，
  静默关闭发生在 nominal-hold 之前的 32-sensor `force_matrix_w` spawned-receipt 枚举。该枚举
  已留给独立 formal table smoke；hold 仍逐步保留 table/fall/hard term，真实截图/hold verdict
  待复跑。
- clean `4c870e94` Pod numeric hold 已让 `bh_loop_c` 与 `bh_block` 各保持 `0.8 s / 40`
  policy steps，均 `PASS`、双脚接触率 `1.0`、零 terminal；minimum root z 均为
  `1.0684000 m`，maximum tilt 分别 `0.00983/0.01029 rad`。首轮 PNG 的 post-write/step/final
  均显示直立稳定，但原生 reset 图是 RTX 首次 render 的全黑 warm-up 帧；截图器现先丢弃同一
  物理状态的首帧再保存第二帧，待 Pod 复截后才判断原生 reset 姿态。
- clean `22890ea2` Pod 复截已让两动作 `raw_env_reset` 均得到可见 PNG，原生 reset 是直立、
  双脚着地的动作专属 frame 0，不是歪倒/failure-buffer 状态；两件 screenshot hold 再次
  `PASS`。loop/block receipt SHA-256 分别为 `e0abbfe6…` / `53e8950c…`，远端证据目录为
  `/workspace/franco/n1dr_nominal_22890ea2_{loop,block}_frames_r1/`。因此下一训练 blocker 已从
  “出生是否站得住”收窄为 dynamic-ready 的 qdes/last-action/observation/reference/preparation
  接线，以及 teacher 在 official low-gain 腰 plant 下能否走到击球窗。
- CC 复核发现 teacher-rate consumer 的 geometry 模块绑定位于 `try` 内、异常类型却在
  `except` 上引用该局部变量；属性读取本身失败时会用 `UnboundLocalError` 掩盖根因。绑定现已
  移到 `try` 前，并新增缺失 geometry 时保留原始 `AttributeError` 的回归；这是异常完整性修复，
  不改变有效 task 的 teacher-rate 数值或训练目标。
- reset 收据治理采纳“checkpoint/hourly 物化、热路径紧凑事件日志”的方向，但把
  `~7.1 ms/env-reset` 明确降为 profiler 前上界；仅 `seed+config` 不足以离线重建，日志还必须
  保留 env/action/generation、domain、birth/sample、proposal reason、exact task、生命周期与
  outcome。该改动不做学习 A/B，但须 Pod fixed-tape、旧收据重建、exact-resume 与分段吞吐验收。
- reset 审计排除 failure-buffer 提前启用：ActionBall canonical path 固定
  `stand=1/post-swing=0` 并写动作 frame 0；当前没有“35% strike 后混入失败姿态”的实现。
  source 已加入 action-specific static-hold minimax 和 dynamic-ready candidate producer；
  独立复核发现并修正 MuJoCo actuator row 与 A3 runtime joint order 的非恒等排列。
  下一步只在 Pod 跑 focused regression、物化 loop/block 候选，并截取真实 Isaac reset 后
  `0/1/10/final` 帧与 hold telemetry；尚未授权 trainer/long。详见
  [设计/加速审计 §6.7](research/design_audit_and_speedup_20260729.md#67-reset-语义复核与-dynamic-ready-实现状态2026-07-30)。
- block stable-v2 的 1024×100 recovery 在 update 77 前始终 mean episode `21--22`、
  strike=`0`、actual raw-hard `47--49 events/rollout`，否定当前 ready 可在 100 updates 内
  自救；`model_20/40/60.pt` 已写出。update 77 后暴露 teacher-rate producer/consumer 的 float32
  边界复验不一致并按 Traceback 停止；`194e9786` 已改为复用 canonical 容差、继续禁止
  clipping，Pod1 focused test `2 passed`。Pod MuJoCo replay 同时表明 static LP 不等于动态 hold，下一步先物化
  action-specific hold qdes，再谈 preparation window 和 long。
- stable-upper v2 loop 的 4096×5 probe 仍在击球前由 actual raw-hard 大量 reset：
  mean episode `21.01--24.20`、strike=`0`、吞吐约 `2.3--3.7k environment-steps/s`。
  block 4096 构造在 PPO 前由 launcher 的 900 秒静默 stale 门自然停止；其 1024×100 recovery
  已到 update 22，`model_20.pt` finite，但 episode 仍约 `21--22`、strike=`0`。老师腰轨迹
  远离 hard limits 且 q_des 未越界，下一直接修转为 unified A3 dynamic-ready/qdes/preparation
  合同；Reward、CaT、full-body 与 curriculum 剂量比较继续等待健康 strike baseline。
- stable-upper loop/block `1 env×2` smoke 均自然完成，4 个 checkpoint finite，q_des projection、
  table、fall、nonfinite 与腿/踝 hard 均为零；但两动作都在 episode age `16--17` 唯一触发
  `waist_pitch_joint` 上侧 raw mechanical edge，mean episode 仍仅 `16--17`、strike 为零。
  两动作同位同龄反例定位到 successor 漏项：A3 官方 stand 的腰 ready 为零，而 v1 保留了旧
  深蹲动作 frame-0 `waist_pitch=+0.103 rad` 绝对偏置。v2 将三腰轨迹整体重基准到 runtime
  ready 零位，同时逐帧保持相对 frame-0 增量和 qd。Pod focused test `2 passed`；新
  loop/block motion SHA 为 `0fa46ad6…` / `cc9bbccd…`，双脚 `3+3`、static LP
  `feasible=true`，击球帧拍速为 `1.8181/1.6422 m/s`。需重绑 contact 后重跑 smoke。
  N1 contact producer 已切到 v2 exact bytes，并把 runtime-site finite-difference 速度更新为
  `1.8083/1.5947 m/s`。Pod N1 focused regression `11 passed, 2 deselected`；v2 loop/block
  bundle SHA 为 `85c7a276…` / `09d0dea3…`，均 materialize PASS。
  真实 Isaac scene 物化的 loop/block policy contract 为 `03f833e1…` / `3442881f…`；
  12 腿与三腰 normalized bias 全零。materialization 在写出 exact recipe 后因无
  `Learning iteration` marker 被 boot wrapper 记 rc=1，但两份 no-clobber recipe 完整存在；
  下一步生成绑定该 contract 的 fresh smoke spec。
- 历史早期恢复审计没有找到 fresh 0--300 update 对照；唯一近邻 `s1w4_M2_v4rg` 是从
  `model_13000` 连 optimizer warm-resume，恢复后第 2--12 update 才跨击球窗。故 `4096×5`
  只否决当前 setting 直接 long 的资格，不证明 fresh policy 长线永不可恢复。stable-ready
  probe 健康后先跑 `100--300` update fresh recovery，再据 strike/raw-hard 趋势续 long。
- N1 launcher 的 contact receipt validator 已扩展为同时接受两种 upper 合同：历史
  corrected-Z receipt，以及 stable-upper 把整块 contact box 重绑到 pinned strike-frame
  selected rubber-face center 的 retargeted receipt。两条路径按互斥 exact keyset 和 authority
  校验；这只是让发射器理解已验证的新工件，不改变球题、Reward、PPO 或训练语义。待 Pod focused
  test 后串行发 loop/block `1 env×2` smoke。
- qvel-fixed 反手拉/挡已完成 exact `4096 env × 5 update`，但两者 mean episode 仅约
  `23/12` 步、strike 恒零，actual raw-hard 分别约 `2.5k--4.2k/update` 与末轮
  `7.7k/update`，吞吐约 `2--3k environment-steps/s`。finite checkpoint 正常且
  q_des/table/fall 不是主因，故短 probe 已足够否决当前 long，不再用 Reward 或更多 env
  掩盖 plant birth。
- exact A3 复核定位到 upper 出生合同：两动作共享 root `z=0.920683 m`、pitch
  `-11.19°` 与深蹲腿位；几何接触不等于 implicit-PD 闭环可保持。下一资产保留腰以上动作，
  将 12 腿、三腰 frame-0 ready 与 root 改到 runtime default stand（upright、`z=1.0684 m`），
  三腰只做常量重基准而保留动作增量/速度，重建 FK 并重绑
  ball/task。稳定 stand 配置与 no-clobber materializer 已在 Pod `12 passed`；两条新 motion
  SHA 为 `4343a85e…` / `08aeafaf…`，exact A3 双脚 `3+3` 接触且 static-ground LP
  `feasible=true`。击球帧拍速保持，世界拍位变化后 N1 contact box 正在整体重绑；仍待
  `1 env×2` 与 `4096×5`。两动作 stable N1 bundle 已物化 PASS，SHA 分别为
  `054be7f2…` / `6973f1a3…`；真实 Isaac scene 物化的 loop/block policy contract 为
  `80c70eb3…` / `359d4e97…`，腿 normalized bias 均为零。旧 qvel bundle/policy receipt
  不得跨 motion bytes 复用。详见
  [设计/加速审计 §6.5](research/design_audit_and_speedup_20260729.md#65-4096-probe-反证与-a3-stable-upper-successor2026-07-30)。

## 2026-07-30（A3 upper qvel-only 资产物化）

- Pod1 已让 qvel-fixed 反手拉/挡分别自然完成 `1 env × 2 update`；iteration 为
  `4.65/3.18 s` 与 `4.67/2.92 s`，四个 `model_0/1.pt` 均可载入且逐 tensor finite。
  q_des/table/fall terminal 为零，但 N=1 已见踝关节 actual raw-hard，故不据小样本调
  Reward。launcher 新增唯一 fixed `4096 env × 5 update × save1` 的 `probe` stage，
  下一步用同 setting 验证吞吐、episode 是否跨过 `t_hit`、strike 与 hard/table/fall 分账。
  详见[设计/加速审计 §6.4](research/design_audit_and_speedup_20260729.md#64-决策账本与执行进度2026-07-30)。
- Pod1 用 exact `A3T2.5_pingpong_0519` 模型完成两条 upper 的 qvel-only
  no-clobber 物化。反手拉/挡 motion SHA-256 分别为
  `3b7cabdec864db09cf3124557b0f79e9f81b4e5cdb28b67a019df11471d307e0` 与
  `a228e5695a70d19e0153317fd2124d8c6db1c800f3d00ca9bb3c5ee3eb944e0`；
  只把 12 个恒定腿 qpos 对应的 stale qvel 归零，所有 qpos/root/timing/strike 以及
  `right_racket` 全帧位姿、线/角速度 bitwise 不变。输入姿态双脚接触、joint/collision
  检查通过；零速度 static-contact LP 不可行被保留为遥测，因为它不是动态挥拍 feasibility
  证明。资产与 N1 bundle 已纳入 Git。
  决策账本见
  [设计/加速审计 §6.4](research/design_audit_and_speedup_20260729.md#64-决策账本与执行进度2026-07-30)。

## 2026-07-29（N1 ActionBall 真实 smoke 与 physical-crossing 拆账）

- `eaf55fba` 已在 Pod1 排除“2%-inner soft-band Done 仍是全部根因”：recoverable inner
  occupancy 不再 reset 后，4096-env updates 0--4 仍有
  `2,549/3,986/4,225/4,188/4,162` 次 raw-hard terminal，episode 约
  `22--24<t_hit`、strike 为零，而 q_des projection/penalty/nonfinite 全零。进一步的 A3
  资产对账推翻了“现役 upper qpos 未接地”的初判：两条 upper 的 12 个腿关节位置已全片恒定并
  等于 A3 grounded-ready candidate（其中 `candidate_id=G1` 只是候选代号，不是 G1
  机器人），exact A3 MuJoCo frame 0 均为双脚 `3+3` 接触；真实不一致是这些恒定腿位置仍携带
  非零腿 `joint_vel`。Pod 原型将 12 列腿速度归零并重建 schema-2 后，每帧
  `right_racket` 位姿、线/角速度和 strike identity 均 bitwise 不变。下一步先把该 q/qd
  合同修复做成内容寻址资产，再跑 1-env/4096 smoke；raw-hard/table/fall/nonfinite 终止保持。
  full compiler 的 ready/face 证据链另行补齐，不阻塞 upper 首发。详见
  [Reward 因果实验](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)、
  [G05](gates/G05_isaac_training_first_loop.md)和
  [设计/加速审计独立裁定](research/design_audit_and_speedup_20260729.md#6-codex-独立裁定与执行顺序2026-07-29)。
- 第一批不改学习问题的 hot-path 候选已收拢：immutable receipt SHA 外部缓存、manager 同一步
  strike timing 去重、global+per-action `10+8×N` 标量
  [device-to-host transfer（D2H，设备到主机传输）](DEFINITIONS.md#device-to-host-transfer) 合成一次、以及
  `fired_valid` device mask。它们不做学习 A/B，但必须在 Pod 通过数值/状态/exact-resume
  parity 与 profiler 后才进入 grounded-ready replacement；Reward、reference/CaT、
  death/entropy/sigma/RSI、8192 env 仍留健康 baseline 后做单变量 canary。
- curriculum 审计也更正了 CC 报告的一个事实前提：recent-100/rolling-30 只负责候选调度，
  不是 formal 晋级门；schema-4 已经产出 action×axis×side 的 `NB/NB_F` 新增带计数，但
  marginal formal consumer 仍错误读取全域 `F/(L+F)`。候选修复直接把 marginal 判定接到
  `NB_F/NB` Wilson 区间并保留全域 admission/unsafe blockers；这是统计合同修复，不做学习
  A/B，须在 Pod 跑反例回归。
- `8d2a1bcd` 的 diagnostic-only joint×side 计数已在 Pod1 真实 Isaac 闭合。1-env 两轮都在
  age `17/19` 由 `left_ankle_pitch_joint` 下侧 inner band 触发；4096-env updates 0--2 的
  actual event 为 `3,187/4,457/5,087`，其中左脚踝 pitch 下侧
  `2,304/2,416/3,112` 次，另有腰 pitch 上侧、右脚踝 roll 上侧和少量腰 roll。绝大多数只是
  进入 hard limit 内侧 `2%`，不是 raw mechanical hard edge；没有首步事件，计数分母与旧
  `joint_actual_forbidden` 完全对账。run 在完整 PPO boundary 保存 `model_2.pt` 后停止。
  这证明旧 Done 把可恢复软约束和硬碰限混在一起。下一候选按既定设计让 2%-inner 只进强
  actual-q barrier/遥测，只有 nonfinite/current-or-substep raw hard edge 才 reset；table/fall
  不变。source 已改，尚待 fresh Pod smoke/4096 行为验收。
- `478f485b` 的额外 `5%` finite-q_des 内缩已在 Pod1 给出反例：1-env 两轮自然完成，
  但 4096-env updates 0--6 仍为 `33.98--41.66 s/update`（首轮 `27.11 s`）、
  episode 约 `19--24` steps、`4,664--5,087 joint_actual_forbidden/update` 且 strike
  opportunity 始终为零；q_des termination/projection penalty 均为零。该 run 已在完整 PPO
  boundary 保存 `model_6.pt` 后停止，候选不晋级。正确 runtime joint order 下，老师全片和
  `q+0.02*qdot` 都没有越过实际 hard-limit 内缩 `2%`，因此下一步只加 diagnostic-only 的
  joint×side×episode-age GPU 计数，在 update 边界一次同步，先分清地面/初态 plant drift、
  当前 q inner-band 与真实 substep hard edge；不再靠改 Reward 或继续猜 margin。详见
  [Reward 因果实验](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)和
  [G05](gates/G05_isaac_training_first_loop.md)。
- `5dbb4e58` 的 Pod1 反手拉 1-env smoke 自然完成，但 4096-env update 0--17 仍约
  `4.7k joint_actual_forbidden/update`、episode length `19--24<t_hit≈31`，fall 仅
  `0--23/update`，说明慢速主因不是倒地或 q_des clamp 失效，而是真实关节进入 hard-limit
  内缩 2% 的安全带。update 3--8 对旧 source 只快约 2.5%，不能晋级。候选修复不动
  Reward/Done/margin，只让每个 5-ms substep 保持滚动 20-ms crossing/brake horizon；host
  joint-safety focused `81 passed`、联合 runtime wiring `125 passed`，尚待 Pod A/B。详见
  [Reward 因果实验](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)和
  [G05](gates/G05_isaac_training_first_loop.md)。
- `curr-launch-fix@7a14b0b9` 已在 Pod1 自然完成反手拉 upper 的
  `1 env × 2 update`：两轮 `2.85/2.02 s`，`model_0/model_1` 共审计
  `1,775,488` 个 tensor 元素且全部 finite。随后 4096-env 诊断首轮真实跑出
  `28.36/39.82 s`，有限 q_des 投影、nonfinite 与投影罚均为零，但旧名
  `joint_qdes_forbidden` 实际混入 q/qdot 预测 crossing；实际 hard-limit 也单独触发，仍造成
  reset storm。候选修复保留有限 brake target，只让 q_des term 终止 nonfinite，预测 crossing
  不再重复 reset，真实/substep hard edge 仍由 actual term 终止；Pod host 联合测试
  `80 passed`。exact spec、证据与后续 replacement 见
  [Reward 因果实验](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)和
  [G05](gates/G05_isaac_training_first_loop.md)。
## 2026-07-28 深夜(舰队满编与合 main)

- **合 main(Franco 裁定)**:ballfirst 谱系快进合入(62 提交),动作集入库 `assets/motions/`。
  旧合并阻断由 admission v2 解除(门限定到 canonical 消费路径)。
- **14 臂 ball-first banked 矩阵在训**(五动作组 8 + 101 全库组 6,+5000 淘汰制),
  table/death −1800 双默认;pod2 曾遭一次横扫误停(00:44Z 六臂同灭),已 fresh 重建并给巡检
  加了名册对账。逐动作读数首次可用(clip_names 桶):反手挡/拉在学,正手拉 0(待 combo 臂
  确诊拍面死区),s0 解在挥速锥外 → 匹配带 v2 题库排队。
- **动态课程代码全绿未发射**:uniform 采样两次裁定、球出生门、新增带 rolling-30 环、pin 工具
  全落地(462 测试);A 路线在信任根触发诚实停点(五类构建器缺失 + 73 件不合 ready 端点合同 +
  receipt 三字段无定义),两信任集保持空集待 Franco 深层裁定;环 n=30 使 f10 的 expand 统计
  不可达,环长待裁。
## 2026-07-28(Fable 接力 Codex 断粮现场)

- **桌碰 no-touch 合同补全**：ActionBall 从 broad/body-origin + 单腕 filter 改为 exact 32 个
  articulation body × 五件桌体的逐体 pair-filter（含双脚；拍面/拍柄归右腕），四个 physics
  substep 全部做 freshness + sticky latch，接触阈值收紧到 `1e-6 N` 数值零容差。host focused
  `83 passed in 2.19s`；构造门与 Pod smoke 已同步要求 32 个 `[env,1,5,3]` live tensors，并将
  4096-env throughput/memory 与整周期 `>=5 mm` continuous teacher clearance 保持为开跑前硬门。
  复现见 [G05](gates/G05_isaac_training_first_loop.md) 与
  [桌体安全工序](operations/run_action_ball_table_safety_smoke.md)。
- **ActionBall 桌体安全链 E1 初版（已被上条 32-body no-touch 合同取代）**：初版建立五件桌体、
  四子步 latch 与 Pod actor-contact 工序，但只覆盖 broad/body-origin + 单腕 filter，故其
  `74 passed` 不能作为当前 admission 证据。G05 继续 `Partial`，不授权训练/真机。
- **physical-hard joint 账接入 runner**：ActionBall/UpperSafe 现采用 zero-copy
  `prepare → device-side validate/sparsify → durable prepared sidecar → optimizer →
  durable commit marker → exact-token ack`。保护任务的旧 one-shot destructive consume 已禁用；
  actual hard edge/non-finite q 会先落 fatal sidecar、阻止 optimizer 并保留账本。缺 4+1 readback、
  identity/transcript 漂移、磁盘/parent-directory fsync、容量/预算失败均 fail closed。host focused
  `70 passed`；4096×24×31 safe-case prepare `0.0443 s`、Python peak `15,183 B`、完整边界
  `0.3649 s`、sidecar `956,704 B`。尚无 Pod Isaac 真实 4×0.005 s readback、两次 update 或
  Pod filesystem fsync 证据，因此不授权长跑/真机。详见
  [实验](experiments/2026-07/EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md)与
  [工序](operations/run_upper_n3_backhand_safe.md)。
- **三反手 upper 专卡候选**：旧 N4 upper 在完整 PPO boundary 收口到 finite
  `model_10809`；N3 确定性题库只保留 `bh_loop_c/bh_block/s0_highpress`，并新增 175D
  `HOPEPingPongUpperSafe` 叶子、physical-hard 双 joint termination、pre-physics guard、统一
  terminal 罚与 GPU1-exclusive smoke-only launcher。当前 E1 为 `14 passed, 1 skipped`，旧 source
  probe 在 Hydra/Kit 前拒绝；clean commit、substep hard readback、Pod 两次 update 与 canary
  未过，故没有长跑。详见
  [实验](experiments/2026-07/EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md)与
  [工序](operations/run_upper_n3_backhand_safe.md)。
- **发射还账**:07-27 00:09 Codex 在两 pod 发射的题库化 bh_loop_c 六臂波此前无文档,现立账
  [EXP-BANKED-CGROUP-BHLOOPC-20260727](experiments/2026-07/EXP-BANKED-CGROUP-BHLOOPC-20260727.md)。
  核心读数:bank 基线 legal/strike **0.488** vs uniform 对照 **0.000**(同 seed);bank+seed1 也崩塌
  (崩塌臂 reward 反而更高——"不打球"仍是收入最优,种子脆弱性是真风险);ep20 臂查实为 base 的
  逐位重复(episode_length 覆盖未生效),白跑作废。v2 reward-scale 队列 4k 终读:上台 1.2× 全面
  领先、延付有害,该档进下一代默认。
- **现场恢复**:Codex 断粮时的未提交 action-conditioned ball-first 层(+14k 行:action_ball_*
  课程/采样/manifest/runtime + hope_commands 接线 + 合同/实验文档)已从 task-first worktree
  抢救为提交 `6b13e0ff`,新工作分支 `Franco_claude/ballfirst-curriculum-20260728` 已推 origin。
  本地 v2-reward 工作树的过期拷贝收进 stash(`stale-worktree-copies-before-taskfirst-switch-20260728`)。
- **101 动作库**:pod2 `/workspace/yikang/chingmu_retarget/chingmu_a3_units_v2` 实为 74 个
  unit pkl(15 FH + 59 BH)+ 26 frozen take;`ball_ext/` 已含逐 unit 真球轨迹(击球前后各 1s,
  覆盖率中位 100%)——**匹配球是实测的,不用反推**。已有 2 条验收样例(058FH/060BH,含
  v_in 拟合与实测拍面),批量转换进行中。

## 2026-07-27（按动作条件化 Ball-first / 任意 N 动作预开跑）

- 裁定训练顺序为 `action → incoming ball/base/aim → fixed-action solve → atomic install`；训练期
  selector 关闭。新增
  [训练合同](interfaces/action_conditioned_ball_first_contract.md)与
  [实验预注册](experiments/2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md)。旧
  task-first 只保留为消融。
- 已形成 strict arbitrary-N manifest/adapter、逐动作 deterministic sampler、fixed-action proposal
  solver、marginal→joint 课程、single-use birth/task broker 与 exact-resume host 候选。schema v3
  新增到球时间，并把位置、速度/方向、旋转大小/方向、落点、base spawn/travel 的两侧拆成 exact
  32-arm catalog；`no_move` 有效 28 arm。teacher rate 由 required/reference physical
  racket-site speed 决定，额外 ready wait 上限 1 秒。课程把
  solver/安装/启动/闭合、safe policy failure、table/fall/collision 和 infrastructure invalid 分账；
  rolling-100 只排下一候选方向，frozen checkpoint generation 全局单调，正式扩域仍需互斥
  canary/heldout。sampler+manifest+adapter v3 中间验收为 `184 passed`，加
  curriculum/evaluation 后 domain-core 为 `210 passed in 8.35s`。N93/E4096 单轮 sampler
  state 为 `160,906 B`，但 4096 个 birth/sample pair 已到 `6,070,936 B`，100 轮线性外推约
  `607 MB`；因此跨 sampler/broker/pool/provider/Racket 的退休前缀 segment compaction 已升为
  长跑硬门。runtime/Motion/train union 尚在收口，不能把局部数字当发射通过。
- 首轮仅允许 `no_move`；新正手仍缺 upper/full、动作特定 `t_hit/t_cycle`、physical racket-site
  speed、全周期无撞桌和 opaque motion admission。Pod1/2 GPU 占用快照下未启动新 trainer；N93
  ordered bytes/certificate 也尚未闭合。只读复核 Pod3 后确认 exact `out_refined` 路径不存在，
  但另一路径已有 74 组动作/球 sidecar 同名配对、108 个击球单元 metadata；没有 exact ordered
  93 件集合，不能用 74/108 冒充 N93。G05/G08 状态不变。动作条件
  Ball-first 守护已改为每小时一次；它只读检查并在全部 Gate 通过时发起新 namespace，不打扰现役
  训练。

## 2026-07-27（task-first / 任意 N 动作预开跑合同，已被上节取代）

- 曾把 executor 训练边界改写成不依赖来球的
  [task-first 任意 N 动作合同](interfaces/task_first_n_action_contract.md)：每个动作先做零增量中心
  warm-up，再按位置、标量速度、拍面锥和 base residual 独立扩域；站位中心平移同时移动动作、
  球拍 task 和 base 中心。新五动作视图淘汰旧正手 `fh_loop`，但新正手仍缺正式 `t_hit`、
  `t_cycle`、击球速度、无桌碰与 Pod Isaac 证据，所以保持 `training_authorized=false`。
- 新建 [动作能力 selector 合同](interfaces/action_capability_selector_contract.md)与两份因果预注册：
  [训练生产顺序 / task 泛化](experiments/2026-07/EXP-TASK-FIRST-N-ACTION-20260727.md)、
  [实际 Reward 配方](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)。selector
  顺序固定为硬安全 → support/OOD → 校准成功率下界 → 同带宽 priority → 明确 abstain；stable
  action UID 是身份，dense slot 只作本地索引。
- 该方案因 task 与来球可能无物理解而被取代；记录保留为历史消融，不再发射。

## 2026-07-27（虚拟球 rollout 融合:45 ms → 0.067 ms,逐 bit 不变）

- `virtual_ball.coarse_landing`(训练侧落点预测的 100 步 RK4)原来每步发 86 个 GPU kernel、
  一次调用 8600 次发射、45 ms,且**与球数无关**(64 球和 16384 球一样贵),这是"连续画球 +
  内联逆解"路线唯一的成本瓶颈。现在整段 rollout 融合进一个 Triton kernel(一次发射),
  RTX 5090 上 0.067 ms/次,约 680 倍;逆解一次(n=4096, 12 迭代)从 13.7 s 降到 0.19 s。
- 数值**逐 bit 相同**:120 万颗随机球(常规/出界/擦网擦线/地板以下/NaN-Inf)零 bit 差;
  整条 12 迭代 Gauss-Newton 逆解与连续题生成器输出零 bit 差(对照 git HEAD 实现)。
  运行时还有 per-configuration 逐 bit 闸门,不一致就自动退回 eager 参考实现并告警;
  关闭开关 `HOPE_VIRTUAL_BALL_FAST=0`。回归测试:
  `hope_training/whole_body_tracking/tests/test_virtual_ball_fused_rollout.py`。
- **合并前必须处理**:`physics_contract_sha256` 是 `virtual_ball.py` 等文件的内容哈希,
  文件一改哈希就变,`stage1_question_bank.load_question_bank` 会在启动时拒绝所有既有题库
  (物理其实一个 bit 都没变)。离散考试题库需要走 rebind 流程重新绑定后才能合入。

## 2026-07-24（承接 07-23 动作库终审）

- 五动作 × upper/full 十件正在按 `shared ready → 击球 core/window → shared ready` 直接路径重建；
  击球窗口允许持续加速，`adv2c3` 仅保留为历史比较项。十件当前都只是 compiler candidate，
  grounded torque/contact、行为与恢复、registry adoption 等门尚未闭合，因此仍为 `0/10`
  training-authorized；详见[终审实验](experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)
  和[预处理合同](interfaces/motion_preprocessing_contract.md)。
- 晚间续建（Fable 接力）：把 Codex 断线时改到一半的 marker-authority v2 迁移续完（compiler/
  bank gate/neutral CLI 统一从 v2 权威取 ge80 窗口 seed 与 nominal anchor，事件不混叠成窗口）；
  排序补窗前零加速度平台票数、加 `t_hit<=0.5s` 编译筛门；实现 §5.1 受保护窗口摘要模块；
  对全部近日产出跑了多代理独立审计（9 维度 + 逐条对抗复核）。审计确认一条**合并阻断**：
  空信任集的 legacy motion admission 接进了默认训练路径，合 main 前必须先补 v4rg digest 或
  限定作用域（详见 NOW 顶部合并警告）。ready v2 = 中立右臂挑战者 + G1 平足重解，pod2 求解中；
  角度中立 vs 五动作可达时间对称、0.5s 门对拉球是否过紧，两问挂 Franco 裁决。

<details>
<summary>2026-07-22/23 的历史推演流水（保留证据，不与上方当前结论并列）</summary>

## 2026-07-22

- 两 pod SSH 全失联（本机网络已排除＝停机/换端口），`hope-pods-watch` 暂停；账面以 07-21 为准。
  最大欠账不变：已收口终档（model_10700-13300）一份判卷成绩都没有，pod 恢复后判卷第一优先。
- 分支修复归账：jiayi YAML-null 删参（8ee2e82a）、yikang 归一化 2x2 预检（635be7cf）、jiayi
  精确续训包（9f684ae5，长训刚需）三项搬进 main 血统并各配单测；tanh 合同只搬思想（写成"默认
  姿态局部斜率=原 action scale"规范）；被动阻尼折 kd 转 kdpassive 消融臂；V15 反向情报
  （action_rate 降到 -0.01/模仿翻倍）记录不搬。裁决表+三人分支看板
  （scripts/branch_dashboard.py）见[分支修复审计](research/branch_fix_audit_20260722.md)。
- yikang 反手"结构性死亡"（44-57°/门限 15°/134 帧 0%）经三路对抗复核改判**应复扫**：符号翻号
  与题库复用被排除；伪影机制＝投影钉根删掉反手 ~40-60° 转体 + stationary 剖面取消 -40° 归一化，
  仓库内有同 clip 100%→0% 的 A/B 反证；三步零改码复扫命令已写入审计文档。复扫前反手死刑不作决策输入。
- plant 地面覆盖接线：task.plant 五新键（地面摩擦/机器人材质范围/随机凹凸地形，默认全=现状）+
  schema-3 `ground_plant` 合同块指纹（平地谱系与改动 plant 互相拒绝静默续训）；mjlab 脚部
  reward（落地冲击/抬脚高度,默认 0）+ 击球窗下肢模仿衰减键落盘。量纲要点：foot_soft_landing
  输出是无量纲超阈倍数，mjlab 等效起步剂量 -3e-3（别抄 -1e-4/-0.1）。host 测试 396+47 全绿。
- 预注册抖动-地面-脚部消融波 8 臂（ar02/ar05/ar10/grip/rough/footrw/penlight/kdpassive，父本
  只用 W，对照=矩阵 w_c_s0，续训 13301 到 ~20001;rough fresh 20001）；渲染被占位 commit +
  groundfoot/kdpassive 双闸门锁死。详见[波预注册](experiments/2026-07/EXP-P1-CHATTER-GROUND-FOOT-WAVE.md)。
- 部署侧站立四欠账最小修复（只动 agi/，默认行为逐字节不变）：两条站立路径增益来源启动摘要 +
  static handoff 时间戳事件 + obs/trace CSV 尾部增 31 列实测力矩（SDK 不暴露电流/温度）+
  build git 指纹。macOS 无 vendor 栈未整体编译，待 Linux 构建机 portable Release 回归。
- 发现并修复 main 既有欠账：`test_reward_flags_mdp.py` 在 origin/main 上单独跑就有 12 项失败
  （c102b9e3/ca078850 家族改 hope_commands/commands 后 host stub 未同步：缺
  `planner_revision_enabled`/`_clip_names`/假 motion、MotionLoader 收 PosixPath、teacher 收据
  规范名漂移）——四类全是测试侧欠账，纯测试文件修复后 97 绿，源码零改动。
- v7 终表落账(07-23):11 段终审表齐(切片/直达 t_hit、t_recover、t_cycle、速度/方向边界、
  窗宽、逐段管线距离,全人话列+2c/3 名词卡);主选 C 的 t_cycle 0.266 家族最快(直达 t_hit
  与 s0 并列 0.109);syn 腕拧 0.212 含在 t_recover、正式版锥回后 ≈0.17;旋转轴未测入 caveat
  (v1 阶梯随口径作废),s0 方向轴欠专卷。双对抗复核 CONFIRMED(时间偏差 ≤0.5ms)。
  全身版 fb_adv2c3 七件战役在跑,产物与同款表格待落。
- 变体谱系终审拍板(07-23,Franco):**adv2c3 是主训练件**(模仿参考=各动作 2c/3 切片;
  共同 ready→2c/3 起点的过渡不写死、不进时间账,RL 自动学);**loop 版废弃**(构造回位尾加时
  太久);**原版只当对照**。主选已换 bh_loop_c。正式化按此谱系:每动作 2 件=adv2c3(主)+原版(对照)。
- 设计 ready 验证+时间账重算(07-23,v6,替代作废的 v5):Franco 指正后数据验证成立——
  franco 六段起始单簇(L∞ 均值 16.6°)、正反手同一 ready、收尾闭环;canonical ready =
  bh_loop_c 帧 0(medoid)。从设计 ready 重算:取小口径 11/11 过 0.5s 门(承接口径单看
  fh_loop/bh_loop_b 超门 +0.035/+0.011 须带限定);B"零税"论据失效,C 成时间最优
  (0.422/0.109)且 ready 即其帧 0,建议主选换 C(待 Franco 终审)。双对抗复核 CONFIRMED。
- 首帧穷举+主选终审包(07-23):注册首帧定帧——正手 fh_loop f40(唯一 ≤0.5 候选)/反手
  bh_loop_b f62(worst+mean 双冠);双站姿全库 0.280-0.474s。C 肩 roll 贴限裁决=有风险需
  morph 不必换件(钳位后 t_hit 不变)。B/A/C 真部署口径:C 仅快 B 0.023s 而窗宽差 2.2 倍,
  建议维持 B 主选,换将条件三条已列。正式化战役按 Franco 打断暂停,待其确认首帧+主选后重开。
- 五动作合集定案+正式化开工(07-23):合集=fh_loop/bh_loop_b/fh_block_syn/bh_block/s0;
  bh_loop 终判 B 主选 A 挑战者 C 存档;分工路由(高球→压/近身→挡/远慢→拉/快→挡)阈值由
  扫描几何数据定;每段三变体(原/adv2c3/loop)canonical 三证重产后进 main 供直接训练。
- 第四轮三格收官(07-23):①**部署真口径 t_hit**:正/反手两个共同前置站姿下全库 11 段
  全 ≤0.5s(FH worst 0.474/BH worst 0.434);单一全集 ready 仅 B/fh_loop 超 0.035-0.039s。
  ②**S0 高压理论证实**:高球带 × 1-8 m/s 全档 100%——"够高就能扣"成立,旧"边界 1-2.5"系
  考卷伪影注废。③**bh→fh 挡合成成立**:单轴 wrist_roll −180° 平移(轨迹 [26,44]°→[−154,−136]°,
  ±160° 限内零裁剪),同面 sign,fh_block_syn 三门全净、七轴 92-100%——正手挡位白得健康资产,
  原 franco_fh_block 降级重定向修复案。canonical 正式化 prereg 草案入库
  (research/PREREG_canonical_formalization_draft_20260723.md):五工具解绑/11+1 段三证重产/
  三变体(原/adv2c3/loop)/40.3° 裁决项/挡族同面 sign 规范/两张新卷。pod3 已静默可归还。
- FINAL 11 段终表 **v3** 定稿(07-23 晨,URDF 拍面口径):grip 自由度取消后 9/11 段全谱
  96-100%(速度边界全 5-8、方向 vy2.4);三个站位 yaw 修正(-147/-80/-72°)翻案 v2 的
  "fh_loop 弱/bh_block 锁死"伪影;仅剩两真病灶=fh_block GMR 腕轨迹拍面病(唯一全零,曾被
  解得 grip 掩盖)+ v12_fh_block 方向锥窄;s0 边界 1-2.5=慢高球分工正确;共享就绪税
  0.07-0.16s;**在册 v4rg bake 拍面偏 URDF 40.3° 挂账待裁**;B URDF 锚复测 2c/3 0.42✓。
  pod3 分流净赚 ~1.5h 已空载可归还。表见 final_11_motion_table_20260723.md v3 节。
- pod3 借用入档(07-23,Franco 提供):`ssh root@74.2.96.37 -p 14176 -i ~/.ssh/id_ed25519_runpod`
  ——128 核/3×RTX 5090/基本空载;有 IsaacLab+hope_isaac_venv(一条安装中断日志,完整性待验),
  无 mjeval venv 与 codexschema 资产,借用需先 bootstrap。用途:扫描/认证类 CPU 战役分流与
  临时加速;**注意旧档"74.2.96.x 是已判死旧 pod2 别试"作废**——IP 被 RunPod 回收复用,本端口
  是 Franco 亲自提供的新 pod3。训练发射仍限 pod1/pod2 现役队列纪律。
- FINAL 11 段终表 **v2**(07-23 晚,解拍面口径):v1 适用率半张确系"钉帧法线"伪影——解拍面后
  11 段速度边界全推 5-8 m/s、反手拉三段+v12 反手挡方向 vy2.4 全 100%、窗宽回到运动直觉
  (v4rg 240-400ms;挡球的窗在快球侧 300-420ms,慢档才 0-20ms→挡球考卷应按快球定档);
  残余 0% 仅 4 格且经 60° 宽锥反证为速度锥合同真边界。未标定段的 grip 全部解出。
  详见 final_11_motion_table_20260723.md v2 节。
- FINAL 11 段终表交付(07-23):t_hit 门原就绪 3/11、c/3 前置 8/11、**2c/3 前置 11/11 全过**
  (0.22-0.38s);回位富余 ×3.7-10.8;binding 只剩肩(8)/肘(3)两堵墙;适用率短板在"面"不在时间
  (grip/面标定+挡球快球卷+S0 高压卷是下一步)。全表见
  [final_11_motion_table_20260723](research/final_11_motion_table_20260723.md)。
- v12 补链+11 段全集总账终版(07-23):v12 两段原件在本机 ~/Downloads/v12/(pod 上 yikang
  keep_take4 是另一 take,哈希已分辨),影子 exact-batch 走通 GVHMR→GMR→落地修正→三门→grip→
  锚窗。**v12_bh_block 是惊喜**:近恒等 grip 96% 宽平台、快档 100%。**挡球家族规律实锤**:挡在
  快球轴得分(fh_block 快轴 83% vs 泛用卷 29%)——泛用 default 考卷系统性低估挡球,需专属快球
  题族;franco_bh_block 维持速度+拍面双锁死原判。注册草案升级:grip_session 一等公民+每段就绪
  前置切片。正式化最短路四项列账(工具解绑/C activation/pkl 允许清单/v12 prereg)。11 段表的
  t_hit 空格由 boundary_cert 终表任务补齐中。账:pod2 franco_pipeline HARVEST.md+drafts/。
- t_recover/t_cycle 补全量尺(07-23):随挥改逐关节恒扭矩 bang-bang(acc=τ/M 逐点)后回位仅
  0.16-0.25s——参考随挥 1.0-1.8s 比硬件需要长 5-9 倍,**回位不是瓶颈**;全周期 t_cycle(2c/3
  前置)4/5 动作 <0.5s(fh_block 0.47/bh_block 0.42/bh_v4rg 0.40/fh_v4rg 0.48),仅 B 0.57s 且差在
  hit 段右肩地板(与其题族 0/9 同根)。hit/recover 两段同墙=右肩 pitch,卸肩一处见效两处受益。
  解析下界口径(忽略耦合),CoP 债照旧待回放验证。恒扭矩 bang-bang 正版(α=(τ−|bias|)/M 逐点 41 姿态、双积分+切换点二分、τ×1.0/0.9 两档)复核:
  判定全维持且更快(回位 0.13-0.23s;τ0.9 档只贵 0.01-0.02s);**新发现:recover 段 binding 出现
  肘(τ 仅 24 Nm)——卸肩方案须连带核对肘余量,别把肩债搬到肘**。正版账 TRECOVER_v2_bangbang.json,
  旧运动学版作废留档。
- Franco 全套管线战役收官(07-23 凌晨,pod2 franco_pipeline_20260722/,34M):canonical 五工具
  只认 B/C(拒收行已取证)→ 用影子管线(B 上逐位验证 max_abs_diff=0.0)推其余段全过几何三门。
  **grip 标定是最大杠杆**:反手拉 A/B/C 全部拉回 100% 锚窗(B 烤后六轴 100% 共窗 0.547-0.560,
  拍面残差 0.01°);"B 上旋 0%"系探针读错 grip 的假报警(已记档)。fh_loop 原站位穿桌 10.3cm,
  dx=-0.20 过桌网门但烤 grip 后 33-58% 未达线(需 station×grip 联合设计);bh_block 拍速锁死
  (1.94 m/s),是题族问题并入 v12 挡球线;S0 影子三门全过+专属题族设计草案已出;TOPP 四段
  window_locked 全库同病入 morph 队列不阻注册。注册表草案与 S0 题族草案在 pod drafts/ 待落库;
  正式化第一断点=五工具的 B/C 窄绑定需泛化 prereg。M0 按令 blocked。
- t_hit 终审矩阵收官(07-23 凌晨):硬件真源包络+击球链最小绑定(C3)下 **5/5 动作可行**,
  挡 0.42-0.50s、bh_v4rg 0.44s 原就绪即达标,B 加就绪前置 0.36s——旧"0/5"判决四分之三是尺子病
  (包络过保守 2-20 倍/×0.85 无依据/稳定器误绑)。唯一硬件墙=右肩 pitch 力矩惯量比;三笔债如实
  入账(CoP 转嫁策略稳定器待验证/挡回球性未证/IK 重解未及)。详见认证账本附3。
- 全库四轴可行性认证收官(07-22 夜):**合法上桌 16/16 全过**(速度三档/方向/上旋);
  **纯扭矩 16/16 全过**(τ 剂量最高 0.0166 vs 闸 0.02,没有一条被力气卡死;腕三关节全库饱和
  是共性);统一预算 TOPP 只有 v4rg 家族 4 条 PASS,12 条卡锁窗包络/CoP(非扭矩);投影件
  补跑全矩阵同过(快档 96% 略降如实记)。全表与边界见
  [认证账本](research/motion_feasibility_certification_20260722.md)。
- 投影修复终审通过(07-22 夜):root yaw 折进 waist_yaw + 剖面恢复 rally_yaw 后,反手
  stationary 扫描从 134 帧全 0% → f45 两速度档 100%,**死刑正式撤销**;被删 non-yaw 实测
  ~19-22°(源反手真实侧倾)另记债待裁决。分支 Franco_codex/stationary-projection-yaw-fix-20260722
  已推(未合 main);全库四轴可行性认证战役(球速/来向/旋转/扭矩)进行中。
- 全动作库击球帧扫描矩阵执行完毕(07-22 晚):**反手死刑撤销**——钉根删 yaw 15.7° +
  stationary 剖面吃掉 rally_yaw 40° ≈ 55.7° 正是死刑报告的 44-57° 拍面差;钉根复现件换 generic
  剖面即回 100%(对照 134 帧全 0%)。**全库 16 条现役 clip 无第二个死刑候选**;观察名单
  bh_v5rg(注册帧 38-42%,腿病)与三条窄 band 资产;投影资产锚帧须重选(最优窗前移 f40-43)。
  投影工具修复(保 yaw+fail-loud)+ 重投影 + stationary 终审重扫进行中(执行 Claude)。
  详见[分支修复审计](research/branch_fix_audit_20260722.md)。
- 判卷链修通+首批 11 臂出报告(07-22 傍晚):两个炸点都是旗标级修复(--exam-bank 手传同源考卷;
  --export-extra 补 episode_length_s=16.0 对齐导出合同)。但成绩全 0 接触且**暂不可解读**:
  每题 ~26 步(~0.5 s)被评估器的 reference 相对终止收题(不是摔、不是 deadline guard),
  正手 11 臂全部活不到击球帧、反手能活到但打不准——疑似评估器未实现 task-revision 代际的
  揭题/hold 协议(7-13 老代际同链正常:~140 步/题、接触率 1.0)。取证已闭环:伪影坐实且方向反转——策略按 governor 节奏 0.1s 冲到 6 m/s,评估器参考 1x 慢放,
  ee_body_pos 守卫 100% 收题、零摔倒;本批 MuJoCo 分数作废,诊断档 B(松 reference 阈值)在跑,
  正式修复 A(评估器补 phase-governor 协议)执行中。诊断档 B 已出:守卫伪影坐实(反手
  活到击球帧 0→188/188),但抬守卫后**全臂 100% 物理摔倒、零接触**(V 臂 ~0.9s 倾倒、W 臂
  ~1.15s 高度塌,W 比 V 稳 +0.25s)——老代际协议下的摔倒不能直接判"不会打",终审等 A 协议
  重判;若仍全摔则升级为正式 sim-to-sim gap 证据。全档见
  [judge_results_20260722](experiments/2026-07/judge_results_20260722.md)。
  另:这批 checkpoint 的 lineage_exact 全为 0,判卷走 --allow-inexact-contract 诊断档。
- **修复 A 落地 + 终审重判完成(07-22 晚,main `9d22dc38`+`a5dbfdfb`)**:MuJoCo 评估器按
  ONNX metadata `planner_task_revision` 门控补齐 task-revision 揭题协议(numpy governor 推参考帧、
  tts obs=任务 deadline 倒数、判分帧=governor 到达帧、参考 jv 按 governed 帧速缩放;老代际
  逐字节不变;C++/python 参考/torch 真源码三方对拍全绿)。11 臂重判 11/11 rc=0:**反手全臂
  接触率 100%**,V 臂反手回球率四臂 1.00(拍位 0.02-0.06m、拍面 11-17°);正手两种真实失败——
  V 臂 0.1s 爆发离包络被收、W 臂 183/183 全到帧但拍面反 102-144°(用背面迎球),与 Isaac
  "V摔但回球好/W稳但不回球"吻合。协议伪影结论终审坐实,成绩仍带 lineage-inexact+plant 双
  caveat 但臂间排名可用。Round 6 全档见
  [judge_results_20260722](experiments/2026-07/judge_results_20260722.md)。
- pod 成果验收与复活(07-22 下午):账面核对完毕——25k 续训八格全部被重启杀停
  (w_c_s0@13900/s1@11800/s2@11500/s3@11900;v_c_s0@10700/s1@14300/s2@13700/s3@14000,总目标
  25001);push 波 12 臂到档待判(W 侧 10300-12900、V 侧 11400-13900);Wave Q 账实不符坐实:
  w/v_fullbody@9200、w_hstrong@9700、v_hstrong@9900、w_qbar@7200、v_spdmix/w_spdmix@7000 已
  发射被杀,**v_qbar@10700 已到终档**;全部判卷成绩为零(07-21 唯一一次 w_c_s1 判卷 exam 目录
  是空的——judge.sh 对 rebound 题库名推不出同源考卷,rc=1 静默欠账)。处置:两 pod 各起编排器,
  11 条续训按"原 argv 逐字 + 重算剩余步数(总目标=原起点+原相对 max_iterations)"复活,8 条已
  越过首迭代在训,3 条(w_c_s0/w_hstrong/v_c_s2)冻结在 Starting the simulation 等 1800s 裁决;
  判卷修正队列(--exam-bank 手传 schema3_exam_882fea4_rebound)两 pod 串行在跑,首批 11 个终档;
  hope-pods-watch 在新账号重建(每小时:停滞裁决/收口/按回填队列补位/判卷看护/WARN 摘要),
  回填队列=w_fullbody/w_qbar(pod1)+v_hstrong/v_fullbody(pod2)。
- 蹭滑/拖脚剂量键接线：`foot_slip_sq_weight`/`foot_drag_weight`（此前源码常开 -1.0/-0.5、CLI
  够不着）进 train.py 白名单；penlight 减负臂扩到六项软惩罚（新增 -0.33/-0.17），随
  groundfoot 闸门锁。pod 于 07-22 下午恢复（旧端口未变，重启过）：pod1 三卡=yikang 四条
  训练（legfreeze ft6 + stationary-v2 x3），pod2 三卡全空；checkpoint 卷完好（W 父本在）。

</details>

<details>
<summary>2026-07-12 至 07-21 的历史进度（默认折叠；采用状态以顶部与 NOW 为准）</summary>

## 2026-07-21

- Wave P push 鲁棒性波扩到 18 臂并上卡 12 条（速度 ±0.2/0.35/0.5/0.8、方向 yaw/ang、频率 fast、
  同冲量力推 f035/f08 @ pelvis link 原点）；Franco 拍板 push 是平衡的希望、力推与速度推同冲量配对
  比较。RunPod 两次重启 Pod1 容器（07-20 19:43Z、07-21 04:35Z），`hope-pods-watch` 值班 routine
  自动完成里程碑收口（`w_h_s0`/`v_c_s0`/`v_p035`）、补位（`v_f035`/`v_p08`）与整机重建；
  `v_h_s0` 由人工收口于 model_10900。详见 [push 实验](experiments/2026-07/EXP-P1-PUSH-ROBUSTNESS-20260721.md)。
- 按 2026-07-21 情报预注册 Wave Q 四对臂（速度混合最优先/强 q_des 铰链/全身模仿加强/全关节
  qdes barrier 去 top-k）；速度泛化定为最高价值泛化轴，拍面/引拍/脚步轴降级为评测工具。
  三本账（NOW/INDEX/TIMELINE/PROGRESS）与两份实验记录运行态同步至当日实况。

## 2026-07-20

- 24 格矩阵全部发射后按实测算力（同卡 SM 分时，4 条/卡 ≈15 s/iter）把筛选终点前移到 +4000，并按
  Franco 授权动态清退：N 交互 6 格永久停、8 格暂停待续（台账 `scheduling_ledger_20260720.jsonl`）。
  发射日修复 4 个真机 bug：BFS 关节序断言三层、inference-mode 计数器复位、每卡并发预检、
  `kit_boot_lock.sh` flock 泄漏（子进程继承锁 fd 导致后续发射永久阻塞——两 pod 已打补丁）。
- 预注册并实现 24 格平衡×时序广度矩阵 {W,V}×{N,C,H}×{S0,S1,S2,S3}：新增挥拍后安顿债务包
  `post_swing_settle_debt`（Jiayi V13 思想在 main 重做，77 单测）、24 格 lean 队列渲染器（63 单测，
  修正 v8/v9 的 180 s stale_timeout 死因）、动作语义唯一真源 `motion_role_catalog.json` + 校验器
  （30 单测）、首个真 `mj_step` 动作动力学重放 `motion_dynamic_replay.py`（22 单测）。Wave A 科学位与
  Wave B 六格队列标记 superseded。详见
  [矩阵实验](experiments/2026-07/EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md)与
  [组合语义合同](interfaces/stroke_footwork_composition.md)。

- Wave A v8 的第一条科学长训只发 W-N（Pod1 GPU0）：current-main 命令制品 SHA-256
  `be346f94...4b42a`，`04:45:02Z–04:48:25Z` 停在 `sim.reset`，无首个 iteration、Reward 或
  checkpoint；locked launcher rc=`125`。外层 rc=`121` 来自 train stage 错要 probe-only child evidence，
  不是第二个 trainer 故障。exact PID/PGID `2728928`、starttime `335835722` 已在四次快照中闭合，GPU 与
  locks 为空，其余五格未发；v8 immutable，本次是 infrastructure-only / non-science，结果 JSON
  SHA-256=`ac09b70a...1a75`。详见 [Wave A 实验](experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md)
  与 [NOW](NOW.md#统一工作队列唯一优先级账本)。

- fresh v9/probe10 替换候选已冻结但尚未发射：root
  `/workspace/codexschema/phase1_balance_action_slew_v9_20260720`，config/runner SHA-256=
  `3bf5085e...e59e` / `0fff4515...89ce`，manifest content/file SHA-256=`36ceb3c7...456c` /
  `664375cb...0c4a`。它修正 probe/train stage-aware failure audit，并把 leader/child/failure evidence
  收紧为 exact schema、严格类型与前后同一 snapshot；保持 W-N GPU0/W-C GPU1 swap；
  六格必须全局串行取得 fresh `6/6` receipt，之后才可渲染科学 train。状态继续 inconclusive / not
  adopted；M0 moving teacher 仍因 stance `0/4` 拒绝，不因本次基础设施轮换而解锁。

- 回收的 Pod1 exact evidence 纠正了“两个 S0/M0 `exact_gmr_v2` root 全局 absent / 未 consume”的旧推断。
  S0 在 `2026-07-14T05:05:55.085040Z` 完成，manifest SHA=`a762d6df...d1a23`；唯一 88-frame
  高点拍压输出 finite/30 Hz/31 DoF structural pass，但 ball contact/effectiveness 为 `null`，下一门是独立
  高球题族。M0 在 `2026-07-14T05:06:21.749762Z` 完成，manifest SHA=`fdd60fcf...396e`；四份 moving
  输出结构通过但 stance `0/4`，故 input-gate rejected、不得占 RL GPU。较后的 Pod2 rc127 仍是另一处真实
  失败 location，不删除也不再作为全局 current state。详见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)。

- W/Y 的真实零写入 ONNX `--plan` 均在 exact `origin/main@a0c1284` 通过，随后 fresh `179→31`
  ONNX 也通过独立 checker 与 CPU ONNX Runtime 有限值推理；W/Y SHA-256 分别为
  `ee0e2e83...d970` / `72da43d9...f995`。但两份 checkpoint 的
  `training_contract_lineage_exact=0`，两份 ONNX 的 `training_contract_exact=0`，所以只能诊断，
  production/vendor 必须拒绝。本分支在 NOW 提议把最短 P0 改为 exact-lineage remediation → 同卷 vendor adapter；
  G05/G06 保持 `Partial`、`Gate3-D0` 保持 `Open`。见
  [0.5 秒操作](operations/run_phase1_task_revision_0p5_exam.md)与[G06](gates/G06_isaac_to_mujoco.md)。

- 双 Pod 各三张 GPU 的 NVML compute process 和显存占用均为零，当前没有训练作业；这不是永久卡位
  归属，发射前仍要核验具体 PID/PGID 与 Kit lock。半秒冲刺、Pod1 十二格 long-grid 及相邻长曲线已
  结束/收口。B 已通过 schema-2/FK、L0、vendor L1、桌网门，下一门为动力学/平衡；C 为后备。
  C3/D3 L1 与 A0/A1 checkpoint 配对均已闭合，下一证据分别为 immutable K100/signed K100，不能再写成
  待发 L1 或运行中。权威队列见 [NOW](NOW.md#统一工作队列唯一优先级账本)。

- 最近支线审计未产生整体 merge：Jiayi V13 无 checkpoint/行为证据且分支含无关破坏性历史；Yikang
  V9 force 未传 `position_data`，所谓 COM force 实际施于 link origin，故现有 force runs 无平衡因果
  结论；V10 终档 `9999` 但无 MuJoCo/Gate3，V11 fast/prestrike 停在 `2816/18274`，只有代理材料。
  可用思想只能在当前 `main` 选择性重做并重测。证据入口见
  [G05 支线审计](gates/G05_isaac_training_first_loop.md)。

- 稳定机制拆成两波：Wave A 是 W/V processed-qdes action-slew 六格单拍诊断，不是完整稳定方案；
  Wave B 的 M0 moving teacher 已因 stance `0/4` 被 input gate 拒绝；实现支线把候选收敛为 upper-only
  control、静态 v4rg 十二腿关节软模仿与无参考脚距/qdot 稳定约束，合入前仍不改变现役 setting。Wave A
  probe2 W-C 自然跑完 `6700/6701`，但旧 verifier 错误要求首个 `0.48 s` rollout 也有触球后 `0.20 s`
  recovery sample，故在合法 `eligible=0` 处假拒绝。probe3 随后自然完成 W-C/W-N/V-C；red-team 发现其
  verifier 只验分母能被 4096 整除、未绑定 `24 steps/env`，所以 W-C 旧 receipt 与三格 runtime 均不能
  解锁长训，其余三格未启动。probe4 W-C 随后在建 run-dir/Kit 之前被真实 Hydra compose 拒绝：
  `algo.num_steps_per_env=24` 不是现有 key；v3 root、log、checkpoint、PID/PGID 和 GPU compute 均不存在。
  probe5 改为 [`algo.runner.num_steps_per_env=24`](DEFINITIONS.md#ppo-num-steps-per-env)，保持 qdes/qdot 每 update exact
  `98304`，并转入 v4 no-clobber namespace；新 manifest 为 `6bfa7358…1f51bc`。Pod1 已用 W-C 完整
  exact argv 运行零训练 `--cfg job --resolve`，exit `0`且解析为 runner `24`、顶层无死字段。两波都不得绕过
  连续恢复的 `T0 → T1 → T2` 顺序。见
  [Wave A 实验](experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md)。

- probe5 随后自然完成 W-C、V-C、W-N、V-N、V-H 五格并通过五份
  [exact probe receipt](DEFINITIONS.md#balance-probe-receipt-set)；W-H 的 fork→exec 身份竞态是基础设施拒绝，
  五份旧收据不可续用。probe6/v5 的 W-C 又在 `2026-07-20T03:01:10Z` 于 trainer/probe supervisor 内、
  进入 trainer 前失败：transaction wrapper 给 shlex-quoted multiline Python 参数逐行加两个空格，81 B
  `run.log` 只有 `IndentationError: unexpected indent`。locked launcher 未绑定快速退出的 child，也未发
  signal；leader/child evidence、binding、terminal、checkpoint、receipt 与 RSL 均 absent，PID=PGID
  `2712318` 双扫稳定 absent，GPU0=`0 MiB / 0%`、locks free；其他五格未发。v5 因而永久只作失败历史，
  不是机制负例。probe7/v6 的 W-C、V-C、V-N 随后 natural exit `0` 并各发布 exact receipt；两步均为
  `98304` samples，进程/GPU 闭合。W-N 于 `03:22:50Z` 到达 scene config 后冻结在
  `Starting the simulation`，无 Learning iteration/binding/terminal/receipt；180 s locked watchdog exact
  TERM→KILL 后 rc=`125`，组/GPU1/locks 全空。V-N 在失败被确认前 6 s 已发并自然验证；W-H/V-H 未发。
  W-C/V-N 成功排除 W parent 与 `action_rate=0` 各自为必要失败原因，故不改 Reward 结论或 timeout；v6
  immutable，禁止重试或混收据。[probe8 替换批](DEFINITIONS.md#balance-probe-generations) 随后只发 v7
  W-N：`2026-07-20T03:44:19.857Z–03:44:32.112Z` 在 Pod1 GPU1 的 `sim.reset` 阶段 SIGABRT
  （trainer exit=`-6`、外层 rc=`134`），`run.log` 末行为 `malloc(): invalid size (unsorted)`，没有首个
  Learning iteration、binding 或 receipt；其余五格未发。manifest/claim/launch-spec/run-log/launch-log/
  leader/terminal/child 文件 SHA-256 依次为 `887c0b9e…a7231`、`8472ecf9…fa8b`、`334ed262…c23c`、
  `b3437c87…f49c`、`edc4782f…ce8`、`b7f981c8…8815`、`ad5c46a7…6268`、`c2d6c31b…26f`。事后
  Pod1 v7 只有 `probes/w_n`、Pod2 v7 root 不存在；`03:49:31–03:49:33Z` 两轮 closure 均见六张 GPU、
  相关进程和 Kit/cache lock holder 全空。这是 pre-RewardManager infrastructure 失败，不是 Reward 结果；
  v7 immutable、禁止重试。[probe9 替换批](DEFINITIONS.md#balance-probe-generations) 已在
  `2026-07-20T04:14:32Z–04:22:42Z` 严格按 W-N Pod1 GPU0→closure→W-C Pod1 GPU1→closure→W-H→V-C
  →V-N→V-H 完成；六格均 natural exit=`0`、normal=`true`、first iteration=`true`、exact verifier passed，
  且每格 closure 后才启动下一格。receipt file/content 为 W-N `ee8c5378…8c5ff` / `afce94d7…80f9`、W-C
  `b948a4d8…18d5` / `e5daf19c…d3f0`、W-H `a80502c9…8111` / `32abf562…7e68`、V-C
  `c3db6c38…edc1` / `ae30d1b5…3446`、V-N `06919a60…4bc7` / `bc99acd0…979c`、V-H
  `b7a24015…1ec2` / `0905ad8f…32a7`，共同 verifier=`d736a205…0ebc`。本地 receipt-set 重验通过，
  SHA-256=`cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`；当前六张 GPU、相关进程和
  locks 全空。W-N GPU0/W-C GPU1 通过不证明 GPU 等价，也不抹掉 probe7/8。same-swapped 科学长训在
  `origin/main=16263be5` 的合入前历史 render 为 `/tmp/phase1_balance_slew_train_commands.json`，SHA-256=
  `fc6f1ea38a5a823016d83675d56fc41b50b70dbde1bba60602b26d6c743802df`；它未执行 SSH，本文档合入后
  authority 过期，禁止执行，须在最新 `origin/main` 重新 render 并审计。结果仍 inconclusive / not
  adopted。运行 authority 只来自合入后的 `origin/main` NOW 条目。详见
  [Wave A 实验](experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md)。
- 新增默认关闭的 Wave-B 下肢稳定机制与 W/V×B0/B1/B2 六格 probe/long 队列预注册：B1 是静态
  `v4rg` 十二腿关节软模仿，B2 是无参考脚距下界与实际腿速尾部；每格显式声明双 weight，并共用
  不看成功结果的 pre `0.30 s` / same-attempt post `0.40 s` inclusive gate。source/reward/schema-3/queue
  focused 回归为 `269 passed`，默认不生成 SSH，六条 long 仍被六份自然退出 probe receipt 锁住；尚无 RunPod probe、
  科学训练或行为结果。M0 当前 0/4，不进入本轮。详见
  [实验记录](experiments/2026-07/EXP-P1-LOWER-BODY-STABILITY-20260720.md)与
  [操作页](operations/run_phase1_lower_body_stability_wave.md)。

## 2026-07-19

- 一次 Pod1 只读全域精确查找已为 W/Y 各唯一定位 `model_6700.pt`。两份 checkpoint 均为
  iteration `6700`，各含 `74` 个浮点 tensor / `1,762,715` 个元素 / non-finite `0`，actor 均为
  `179→31`，导出所需四份 `params` 材料齐全。standalone exporter 已新增真正零写入的
  `--plan`：它以 `weights_only` 加载、完成 finite/donor/全材料验证后在首次写入前退出，
  JSON 含 `checkpoint_iteration`、`artifact_written=false` 与 `graph_export_not_executed=true`。
  本地聚焦回归为 `97 passed in 0.38s`，且普通导出 fake smoke 已通过；真实 W/Y plan 尚未在 Pod 运行，
  这些都不是 vendor 行为分。G05/G06 仍为 `Partial`、`Gate3-D0` 仍为 `Open`。详见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 在上述全域精确查找之前，三轮 Pod1 只读定位均 exit `0` 且未重连。第二轮已为 W/Y 各找到唯一完整 `run_name` 的 wrapper
  `run.log`，但日志没有输出可唯一解析的 RSL/checkpoint 绝对路径；cached-source 受限根中也没有可由
  相邻材料精确归属的 `model_6700.pt`。本地源码复核确认日志根由 launcher 的启动工作目录决定，而
  sprint 配置没有保存该目录。第三轮确认每臂有唯一 regular `run.sh`，但仍无法静态闭合绝对 cwd。
  当时 W/Y 因此保持 `UNKNOWN`；后续全域精确查找已按上一条闭合，不倒写前三轮的失败结论。详见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 约 10:26 CST，Pod1 单次只读 SSH 确认 L2 PGID `2457829` 成员数为 `0`，NVML compute app
  为空，GPU0/1/2 的利用率与显存占用均为零；L2 已从待确认状态闭合为进程组与计算进程完全
  absent。Pod1 V/L2/Z3 和 Pod2 D2/F 至此全部收口，双 Pod Isaac 训练池结束；V/L2/D2/F
  是终档 teardown，不是自然终档。下一步为 `W`（拍心优先 × 自由非击球臂）/`Y`（拍心优先 ×
  触球窗老师静音）准备同卷厂商 MuJoCo，`U`（拍心优先 × 强准备）保留为稳定备选；
  G05 保持 `Partial`。详见[半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 本地只读源码定位确认 W/Y 仍是准备态，尚不能行为运行：现有
  [0.5 秒时序卷](DEFINITIONS.md#timing-exam-0p5)使用
  [K100（100 道固定同卷题）](DEFINITIONS.md#q50-and-k100)，但直接驱动 policy、绕过生产 planner；Python MuJoCo
  评估器虽支持 179 维与固定题库，却不消费逐题 25 周期时序卷；Gate3 假球入口只接六元发球列表。
  下一能力是同一 100 题（每侧 50、第 0 帧零速度、25 周期、正手倍率 `2.64`、反手倍率 `1.8`）经
  同一生产规划器（planner）、MuJoCo XML 场景模型（MJCF）和执行 plant 的适配器，并逐题输出
  attempt/completion/hit/return/fall/deadline。现阶段只允许一次只读定位两份 `model_6700` 与导出
  preflight（导出前置检查）；旧连续演练脚本保持隔离禁用。
  G05/G06 保持 `Partial`、`Gate3-D0` 保持 `Open`，没有厂商演示结果。详见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)与
  [G06](gates/G06_isaac_to_mujoco.md)。

## 2026-07-18

- 20:52 CST，修正四条终档臂的审计条件：用 NUL 分隔的完整 `run_name` token 核身份，并按完整
  日志行识别真正 fatal；V/L2/D2/F 都通过唯一日志、iteration `6700`、10 秒稳定、fatal=`0` 与
  配方指纹检查。V、D2、F 已仅对各自数值进程组完成 TERM→KILL，最终组 absent/成员为零且 NVML
  absent；L2 也完成精确 TERM→KILL 且 NVML absent、Pod1 三卡显存/利用率归零，但短等待后
  `/proc` 仍有一个组成员，所以最终状态保持 `UNKNOWN`，以后只读确认 absent/zombie，绝不再 signal。
  四条都属于终档 teardown 收尾，不能写成自然终档。W/Y 的下一步仍是同卷 vendor MuJoCo；G05
  保持 `Partial`。详见[半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 约 19:48 CST，新五臂 `+1000` 已完整：`5701–6700` 的 1000-update 窗和 `6201–6700`
  最近 500-update 窗均为 missing=`0`、duplicate=`0`。U/V/W/X/Y 累计“完成率/合法回台率/fall 率”
  为 `93.31/31.94/0.87%`、`48.40/70.14/22.05%`、`94.14/31.52/0.74%`、
  `49.68/68.08/22.00%`、`93.57/32.31/0.82%`；最近 500 update 为
  `94.99/32.46/0.30%`、`48.14/79.06/23.40%`、`95.28/32.27/0.24%`、
  `48.49/77.33/23.61%`、`95.18/33.10/0.31%`。最近 `<0.5 s` 的“完成/回台”为 U
  `96.42/26.15%`、V `48.88/63.26%`、W `96.61/26.39%`、X `49.22/60.31%`、Y
  `96.53/26.34%`。五维均非支配，不按单指标停止；W/Y 为 demo 优先双候选，U 为稳定备选，
  V/X 因 `22%–24%` fall 尚非 demo-ready。下一步直接给 W/Y 跑同卷 vendor MuJoCo，不继续盲加
  Isaac step。Z3 已精确收口；Pod1 V/L2 与 Pod2 D2/F 仍 live，因 checkpoint/log 路径门不完整未
  signal。G05 保持 `Partial`。详见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 19:23 CST，Pod1 唯一连接重验 Z3 身份/启动时间/命令/source 一致且长时间无
  首个 `Learning iteration`；只精确处置该数值进程组，最终 trainer `/proc` absent，证据
  保留且不重放。V/L2 仍 NVML live，因完整终档门输出被截断而未 signal。Pod2 唯一
  连接确认 D2/F 身份仍 live；本轮误找 stdout 路径，实际日志在
  `simple_half_second_sprint_20260718/<run>/run.log`，所以当前 iteration/fatal 条件为
  `UNKNOWN`，fail-closed 不 signal；其余 8 条 exit 仍 `UNKNOWN`，A/C2 已确认自然终档不变。
  五臂 `+1000` 聚合脚本已跑，但整数输出中段截断，数字和胜者仍 `UNKNOWN`。见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 18:51 CST，Pod1 仅 3 个 NVML 训练侧 compute process（GPU `0/1/2`，util `0/0/1%`）：V 与 L2 已到
  iteration `6700` 但进程仍 live，U/W/X/Y 的 `model_6700` 存在且进程 absent。
  Z3 唯一启动约 11 小时 37 分仍无 `rsl_rl` 日志/首个 `Learning iteration`，只能记为
  启动挂起，不写 fatal=`0` 或 accepted。Pod2 仅 D2/F 两个 trainer 仍 live（GPU
  `1/0/1`），均到 iteration `6700`且日志 fatal 扫描为 `0`；除已确认自然终档的
  A/C2 外，其余 8 条 absent 作业的 terminal/exit 仍为 `UNKNOWN`。U/V/W/X/Y 的
  `model_6700` 都已存在，`+1000` 账本可读但尚未聚合/判定。详见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 07:42 CST，Pod1 的 K2/P2/W 与 Pod2 的 A/C2 均在 iteration `6700`、fatal=`0`
  自然终档。Pod1 当时有 9 个 trainer 进程（8 条 accepted + 1 条 Z3 boot pending），
  GPU `4/3/2`；Pod2 为 10 条 accepted，GPU
  `3/4/3`；两 Pod 其余 live trainer 均 fatal=`0`。Pod1 GPU2 的 Z3 唯一启动仍在
  boot/import，未出现第一个 `Learning iteration`，本轮未重放，因此不记为已接受训练。
  详细 PID 与 run 映射见[半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 06:42 CST，Pod1/Pod2 仍为 `11`/`12` 条 live trainer、GPU `4/4/3` 与 `4/4/4`，fatal=`0`；
  `5701–6200` 的 `+500` 累计窗完整。稳定位置组 U/W/Y 的“完成率/合法回台率/fall 率”为
  `91.68/31.42/1.432%`、`93.02/30.78/1.241%`、`92.00/31.52/1.322%`；激进拍速组 V/X
  为 `48.66/61.38/20.71%`、`50.91/59.08/20.35%`。最近 `5901–6200` 五臂依次为
  U `94.47/31.98/0.612%`、V `46.93/70.82/22.84%`、W `94.92/31.47/0.552%`、X
  `47.49/69.41/23.14%`、Y `94.56/32.15/0.572%`。Y/W 是当前稳定 demo 前沿；V/X 尚非
  demo-ready，但构成唯一高回台前沿。无全维支配，按规则 stop=`0`。结果仍是训练内 virtual
  outcome，不是 vendor MuJoCo；Z3 条件仍为 false。见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 05:42 CST，Pod1/Pod2 保持 `11`/`12` 条 live trainer、GPU `4/4/3` 与 `4/4/4`，fatal=`0`；
  Z3 条件仍为 false。新五臂的 `5801–5900` 第二个独立 100-update 窗已完整：U/V/W/X/Y 的总体
  “完成率/合法回台率/总 fall（pre/post）”分别为 `93.80/31.29/0.956% (0.143/0.814%)`、
  `44.19/62.59/21.80% (20.17/1.62%)`、`94.37/30.81/0.901% (0.150/0.751%)`、
  `46.38/59.55/21.39% (19.61/1.77%)`、`93.58/31.52/0.855% (0.126/0.729%)`。位置优先组
  完成/回台改善且 fall 下降；速度优先组回台约增 `27.56` 个百分点，但 V/X 完成率分别下降
  `16.07/24.01` 个百分点，pre-fall 约 `20%`。Y 安全最好、W 完成最好、V 回台最高，没有
  return、completion、safety 全维支配，故不淘汰。结果仍是训练内 virtual outcome，不是 vendor
  MuJoCo。见[半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 05:13 CST，Pod1/Pod2 分别为 `11`/`12` 条 accepted trainer，GPU 分布 `4/4/3` 与
  `4/4/4`，fatal=`0`，iteration 范围为 `5929–6286` 与 `6018–6238`。Pod1 新五臂的
  `5701–5800` 完整 `+100` 窗已覆盖全部四个初始准备时间桶且分母足够：拍速优先×强准备姿态
  （V）四桶合法回台率 `21.96/36.03/38.21/27.01%`、完成率
  `62.29/61.14/59.40/55.10%`；拍心优先×自由非击球臂（W）完成率
  `87.43/87.16/85.92/82.41%`、合法回台率 `19.41/29.90/30.75/18.97%`。拍心优先×强准备姿态
  （U）与拍心优先×触球窗老师静音（Y）为折中，拍速优先×自由非击球臂（X）居中；无臂全维
  被支配，故暂不淘汰。`<0.5 s` 已有非零训练内能力，但结果仍是 virtual outcome，
  不是 vendor MuJoCo。GPU2 仍为三路，Z3 条件未满足。见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 04:04 CST 直接训练冲刺为 Pod2 `12/12`、Pod1 `11/12`，共 23 条 accepted trainer。
  `HOPE_AGIBOT_A3_USD_PATH` 直接 `UsdFileCfg` 的 5 条 Pod1 新作业均已越过首迭代，
  并真实输出 `<0.5`/`=0.5`/`(0.5,0.9]`/`>0.9 s` 四个初始 TTS 桶的整数
  机会、完成、触球和合法回台计数。GPU2 第四路 `Z/Z2` 在 env/reward 前的 Kit
  shader discovery 处两次同点 allocator abort；显存/RAM 充足且同 USD 其他作业正常，
  所以保留日志、不判配方失败、不做第三次盲试。等 GPU2 自然降到两路后再以
  第三路补该格。旧 18 臂不能事后追溯 TTS×outcome；新 5 臂从 `+100` 开始比较。
  自动 rolling 任务仍暂停。见
  [半秒冲刺](experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)。

- 0.5 秒 K100 已改成无人工 SHA/activation/receipt 的直接 evaluator 路径，并在 Pod2 完成第一份真实
  100 题结果：`model_5700` 正反手都为触球 `0/50`、回台 `0/50`，总计 `0/100`，但物理摔倒 `0/100`。
  运行前修掉 planner 两端半配置和 MotionLoader 高级索引副本导致的假清零；聚焦回归 `18 passed`。
  该 checkpoint 的严格半秒能力被否定，下一批训练转向更宽/更短准备时间、动作加速和拍内 target/TTS
  更新；G05 仍保持 `Partial`。见
  [实验](experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md)与
  [操作](operations/run_phase1_task_revision_0p5_exam.md)。

## 2026-07-17

- ready 量尺 successor 已冻结为 source-only、plan-only 的 Pod2 四格：先跑 4096-env×2-update
  full-scene probe，严格验证 task-entry/ready/foot-unavailable/legacy-hold/nonfinite/task-revision
  整数守恒；probe 和 equal-Reward `model_5700` parent receipt 写回新 activation commit 后，才允许比较
  baseline/strong ready Reward × qdot-limit hinge `-5/0`。qdot 是关节超速惩罚，不是随机力；未来
  `+200/+500/+1000` 各需两个完整 100-update 窗，稀疏合法回台 eligible 不足时零值不得淘汰。
  最小 runner SHA=`2cf2f3dd…5c8f`、专项 `32 passed`，当前只实现 validate/plan、read-only parent
  inspector 与 probe/finalize；Pod2 parent 只读语义检查已通过（inspection `e17cedb1…ade4`，evidence
  `85967393…1096`），fill、behavior/portfolio/stop 是下一迭代 blocked 接口。未运行
  probe/trainer、未排名或 signal，保持 `Partial / NO-LAUNCH`。见
  [实验](experiments/2026-07/EXP-P1-TASK-REVISION-READY-SUCCESSOR.md)与
  [操作](operations/run_phase1_task_revision_ready_successor.md)。

- Stage2 TOPP v8 在 Pod2 唯一自然结束：四格均 generator/TOPP `rc=0`、MJCF closure 通过，但最短时间
  `0.80/0.86/1.10/0.94 s`，`0/4 <=0.5 s`；summary `ac880412…b7030c`。解释器闭包假拒绝已闭环，
  当前 join-ladder family 则按预注册负结果停止；这仍是 screening，不是行为或部署通过。见
  [动作卷宗](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- exact-0.5 K100 v1 的输入失败保持冻结。资产恢复版 v2 已在 Pod2 唯一启动，却在首题前以
  `timing rider requires a native-clock command before activation` 失败；日志 `f8c3be8b…a9e28`，没有
  scorecard，因此是“0 题执行/能力未测”，不是 0/100。后续 2026-07-18 复核已确认它自然 D0 终止，旧
  supervisor/evaluator/guardian/cgroup absent，人工 stop 未发 signal 且不得重放。v3 把 native command、
  零速第0帧验证放到 retiming activation 之前，并绑定自然终档闭包；最新合并专项为
  `88 passed, 1 skipped`。G05 保持 `Partial`。见
  [实验](experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md)与
  [操作](operations/run_phase1_task_revision_0p5_exam.md)。

- planner-mode ready 分母的结构性零值已定位并补 source candidate：旧量尺只认已被 task-revision
  协议清零的 legacy hold，所以 19 份 `+1000` receipt 不可回填、不可排名或淘汰。新量尺在新
  `(control_epoch, task_id)` 安装后的首次 metrics sample 恰记一次，同球 revision 不重复；四个显式
  witness 区分总样本、新 task、非法 legacy hold 和脚传感器不可用。最终 exact commit
  `0ebd14a6…a8dd` 已在 Pod2 的 clean、CPU-only worktree 通过四个 focused 函数（`4/4`）；此前缺 pytest、
  过严浮点 stub 和 source materialization 的 harness 拒绝均保留为基础设施证据，未冒充源码失败或通过。
  full-scene 与两个完整 100-update 窗仍未跑，G05 保持
  `Partial`。见
  [task-revision 实验](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)和
  [训练操作](operations/run_training.md)。

- Historical v1 exact-0.5 K100 source milestone: harness `c2ce2784…1b63`,
  activation `996775d6…7cfb6`, focused `35 passed, 1 skipped`. The skip is the delegated cgroup-v2
  runtime probe; the later v1 Pod launch failure and v2 successor are recorded above. This source-only
  checkpoint did not claim a behavior score. Pod2's 13:05Z read-only snapshot was zero trainers/all three GPUs free; Pod1 was unknown.
  See the [experiment](experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md) and
  [operation](operations/run_phase1_task_revision_0p5_exam.md).

- Formal arena/task-revision planner now ingests every qualified 300 Hz mocap sample but sends only
  the latest immutable snapshot to a one-slot worker capped at 50 Hz; there is no FIFO/catch-up, and
  stale completions cannot cross source/no-ball/close-rearm/epoch/base-authority boundaries. Optional
  strike-spec diagnostics use a separate worker. Full planner source regression is `225 passed,
  2 skipped`; ROS/Jazzy 300 Hz stress remains open. The double-Pod `+1000` cycle also completed:
  all 19 ready checkpoints have receipts, but the old pool cannot legally eliminate because four
  ready/balance denominators are zero. See [planner operation](operations/run_planner.md) and the
  [task-revision experiment](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md).

- main `8b371eb7` 的 ready×join Stage-2 v6 已完成唯一远端 dry-run/execute：dry-run 全绿，execute natural
  terminal，summary=`b5209bc7…`。四格都保持 generator=`0`，TOPP 均 rc=`1`、没有 timing；冻结的
  `75 files / 74 mesh` closure 与退出后无残留均通过。只读 forensics 证明四份日志同 SHA
  `f1d5088e…`，共同首错为 `/usr/bin/python3` 缺 `mujoco`，故结论是 runtime dependency closure 失败，
  不是动作失败。项目实际 TOPP 依赖仅 `numpy+mujoco`，此前 `scipy` 硬门属于过度检查并废除；targeted
  probe 已证明 `/workspace/hope_mjeval_venv/bin/python` 在清空 `PYTHONPATH` 后可加载 `numpy 2.5`、
  `mujoco 3.10` 和 exact MJCF（`nq=38,nv=37,nbody=33,ngeom=79,nmesh=74`）。v7 只绑定该解释器、包
  closure 和 preflight，科学配方不变。后续把完整 RECORD、实际 native ELF/`DT_NEEDED` 解析、canonical
  `ldd/readelf` 与 MJCF pre/post snapshot 补齐。唯一远端 v7 dry-run 随后在 root/child 前 fail closed：
  `readlink()` 字面 target 被过度当作解释器身份，在 binary/包闭包核验前便因文本不同拒绝；实际 binary
  是否漂移在该次尝试中仍未知，execute=`0`。v8 改用 canonical realpath+binary SHA+Python version+
  venv prefix+RECORD/ELF closure，科学
  四格不变；runner/activation=`40e89c6a…ae09/e878de11…0447`，专项 `91 passed`，远端尚未执行。因此
  timing/TOPP≤0.5/L0/L1/行为仍未知，G08 保持 Partial。

- ready×join Stage-2 v2 唯一 execute 已保全 summary `6910db28…f1476`：四份 candidate/contract 与 v1
  逐字节一致，四个 TOPP 均 rc1、无 timing、无重试；这是全部正式结论，`run.log` 只作诊断，不能据其
  文本宣称 rc1 根因。v3 零 generator 调用并精确复用这四份 candidate，只从 frozen Git objects 提供
  `1 XML + 74 mesh`
  closure（75 文件、14,127,373 字节、manifest `e0381752…b962de`）；wrong prior/log/blob/tree/mode 与 XML
  include/path 反例均 fail closed。v3 唯一远端 dry-run 又在结果 root 前发现 expected contract 误绑 v1
  而非 v2，execute/TOPP 未启动；v4 使用新 activation/namespace 绑定四份真实 v2 contract，其余配方不变。
  v4 相关回归 `68 passed`、独立红队 GO，但唯一 dry-run 又在结果 root 前暴露 exact log SHA 后的脆弱
  英文文本猜测；真实日志格式不同，execute/TOPP 未启动。v5 删除重复文本解释却仍因一份 log SHA 手抄
  一字符错误而 pre-root fail closed。v6 把旧 V1 summary、generator 副本与全部日志移出科学输入，
  只复验 v2 四份 candidate/contract，并在 Git-object 完整闭包中各跑一次 TOPP；本次真实运行结果与
  runtime dependency 根因见上一条。

- ready×join Stage-2 v1 dry-run 通过后唯一 execute 自然终止并保全失败 summary
  `f92e6b8b…63c0e`：四个 generator 均 rc0，但 runner 重复了历史已知量尺错误，把 generator 的 float32
  producer-gradient 当成 TOPP float64 workspace-gradient，四格全在 TOPP 前拒绝；无 timing、无重试、
  无 signal。v2 保持动作/join/预算/acceptance 不变，固定新 namespace 并精确绑定 v1 失败 summary；
  candidate 与 TOPP 分别按两条 producer 合同验证，missing/tampered prior 均 pre-root fail，组合回归
  `55 passed`。

- Stage-2 远端执行前的真实 source-root 对账发现 tracked runner 仍错误地从 `b1f5a38` 训练 checkout
  寻找后置 `66f93559` generator；该 checkout 按合同本来就不含它。未创建 Stage-2 namespace、未启动 child。
  runner 已改为读取并冻结 Stage-1 receipt 认证的 immutable generator copy，旧 checkout 只提供
  TOPP/MJCF/URDF/body-order；missing/tampered copy 均在 namespace 前 fail closed，组合回归 `51 passed`。

- ready×join Stage-1 historical attestor 已在 Pod2 唯一 no-clobber consume 中成功发布 receipt
  `7cf1c7c9…c377f`。六格 candidate、完整 schema-2、production-FK TOPP 输出/provenance、frame0 零速、
  protected window 与 source closure 均重验通过；可信 screening 时间为 `1.28/0.70/1.54/1.94/0.78/1.42 s`，
  全部仍高于 `0.5 s`。这关闭了端点证据假绿，不等于动力学或行为通过；四个预注册 `d=12` 中点的
  tracked CPU-only runner 已经独立红队，绑定冻结输入、contact/fps timing、唯一 namespace 和 3600 秒
  child timeout；随后双 source-root 修复后的组合回归为 `51 passed`，下一步仅为一次性执行。

- Pod2 task-revision `+500` 已为 quality 父本六格和 continuous 父本一格发布行为 receipt。quality 六格
  completion=`0.919–0.971`、virtual-return=`0.278–0.395`，没有任何一格达到预注册 dense-collapse 门；
  ready 四项缺测，禁止据 null 排名，故合法 stop=`0`。后续唯一 Pod2 `+1000` pruning cycle 自然 rc0，
  为 `p2_equal_reward@5700`、`p2_no_joint_speed_penalty@5700`、`p2_fast_equal_reward@5500` 发布三份
  behavior receipt；其余 8 条 live 等待 checkpoint，既有 `p2_combo_high_noise_free_medium` importer
  terminal 被排除。两个 parent 均为 `waiting_for_all_live_cells`，没有 portfolio receipt、signal 或合法
  stop。这表示同父本比较尚未凑齐，不表示所有臂表现都好，也没有行为胜者。

- Stage-1 historical attestor 的首次 production dry-run 在写 receipt 前 fail closed：旧 task-revision
  checkout `b1f5a38` 不含来自独立 `generator_source_commit=66f93559` 的后置生成器，首版 attestor 错把
  两个 source root 合成一个。失败只留下 content-addressed attestor source、无 receipt/候选重跑/训练 signal。
  source gate 已改为核 Stage-1 中实际执行的 immutable generator copy 对注册 SHA。第二次 dry-run 又在
  receipt 前抓到审计器把 schema-2 generator 的 float32 producer-gradient 误当成 TOPP 的 float64-workspace
  gradient；现已分开精确重算两条生产链。第三次 dry-run 又抓到 historical TOPP 使用冻结工具的默认
  budget scale `1.5`，而 attestor 错写成 `1.0`；已固定为 source-pinned `1.5`，专项仍为 `47 passed`。

- ready×join Stage 1 六个新端点格都自然产生 candidate/TOPP 原始制品；raw 数值显示 `d=17` 全面更慢，
  反手 own-frame0 ready 将最近 join 从 `0.94` 改善到 `0.78 s`，正手则 own-ready `0.64` 优于
  backhand-ready `0.70 s`，形成 side crossover。独立红队随后抓到一次性 summary 未绑定 consumer 源码、
  parser 未全量重验 certificate provenance，故结果曾降级为 pending historical attestation；上方 receipt
  已关闭该阻塞。六格不重跑，四个 `d=12` 中点转由唯一 tracked runner activation 消费。见
  [短路径实验](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- 新源码 `66f93559` 的 Pod2 CPU-only attempt2 已对真实 v4rg 成功生成正/反手 0.5秒 host 候选，随后
  production-FK TOPP 两侧 hard acceptance 全过，但当前 join 的可行 run-up 上界为 `0.64/0.94 s`，故
  没有 0.5秒动力学证书、未送 L0/L1/桌网/训练。下一轮预注册保持安全合同不变，先跑 join 端点
  `delta=6/17` 的 side×ready 小因子阵，再按冻结规则跑中点 `12` 和细化点 `9/14`；已跑格不重放。见
  [短路径实验](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- ready-to-strike 首次 Pod2 CPU-only 真实生成在发布前 fail closed：正式 v4rg 的三项 canonical migration
  provenance 被 v1 严格字段表误当成 unexpected，正反手均无候选 NPZ/contract、无 TOPP、无 GPU 行为，
  因此不是动作或动力学失败。失败 namespace 永久保留。修复版只新增完整 canonical-v2 三元组这一精确
  变体，逐位继承 primary source、不混入 ready-source，并继续拒绝 partial/坏 SHA/point/tool/未知字段；
  也明确不声称已重算 legacy ancestor bytes。专项 `21 passed`，下一次只用新源码和新 namespace。见
  [短路径实验](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- 动作加速从“只给完整旧 clip 做 TOPP”扩成可审计的 ready-to-strike 空间路径候选：严格取动作第0帧
  姿态、显式零速度、解析 quintic 接入，触球前0.1秒/触球行保持逐字节。独立红队抓到并修复了 join
  `q/-q` 符号跳变和 joint-velocity suffix 过度声明；专项 `8 passed`。输出仍固定训练/部署权限为 false，
  必须继续过 production FK、TOPP≤0.5、L0/L1/桌网/动力学与行为卷。见
  [短路径实验](experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

- Pod2 的 `+500` write-side cycle 使用唯一 SSH 完成只读/receipt 判定：十一条 live 臂均尚未到共同
  `model_5000/5200`，一条既有 importer 失败被排除；没有 behavior/portfolio receipt、没有 signal。
  因此当前零淘汰是“未到共同 +500 门”而不是默认保活，后续到档后才比较 dense collapse。

- Pod2 首次 `+200` write-side pruning cycle 在第一条 behavior receipt 前 fail closed：reviewed atomic writer
  正确要求父目录已存在，而新 consumer 漏建绑定 run_dir 下的 `behavior_milestones/`。没有 behavior/
  portfolio receipt、signal 或 retry；可能新建的 checkpoint receipt仍是独立合法制品。修复版只允许在
  已验证真实 run_dir 下用单级 `mkdir` 建固定目录名，缺 parent、文件或 symlink 均拒绝；专项增至
  `65 passed`。旧失败输出保留；修复进入 `main@85ab36df` 后的唯一 Pod2 consume 成功为 9 条到档臂发布
  behavior receipt。quality 父本 6/6 机制激活、exact-0.5 暴露且 `+200` 合法零淘汰；continuous 父本
  3 条已取证、2 条仍等 checkpoint、1 条 infrastructure-terminal 排除。全程 signal=`0`。见
  [task-revision cutover](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

- task-revision 行为淘汰闭环已补齐：`+200` 只在两个完整整数窗都证明 revision/ledger 机制未激活时淘汰，
  `+500` 的 dense-collapse 还要过同父本组合保护，`+1000` 按 YAML 容差 Pareto，并至少保留两条、一个
  实际记录过 exact-0.5 样本的候选和一个 broad 候选；exact stop 必须同时消费单臂与 portfolio 两份 no-clobber receipt，
  signal 前再验 PID/PGID/starttime/argv。专项 `57 passed`，现役 24 格 claim 重建逐字节无漂移。首次
  每 Pod 单 SSH 的只读 `+200` 扫描中，Pod1 `4 ready + 4 live waiting + 2 infra excluded`；Pod2 quality
  父本 `4 ready + 2 waiting`，continuous 父本 `5 waiting + 1 infra excluded`，因此正确地产生 0 个 stop，
  等同父本 live sibling 到齐后再发布组合决定。红队另抓到并修复了三处合入前漏洞：fast curriculum
  不再替代 exact-0.5 暴露、`+500` 不再使用 YAML 未声明的隐藏改善容差、exact stop 在 intent 后与
  signal 前再次对拍 PID/PGID/starttime/argv，防止 PID reuse/exec 漂移误伤。见
  [task-revision cutover](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

- task-revision successor 的 22 个 delay-zero 格已各消费唯一 launch claim：最终只读复核为 `19 live_exact`
  与 `3` 个首训练 iteration 前的基础设施拒绝，未漏发、未自动重试。Pod1 八条按 `3/3/2` 分布，Pod2
  十一条按 `3/4/4` 分布；live 臂的 PID=PGID、claim/binding、`/proc`、NVML 与首步均一致，未见 OOM、
  Traceback 或 Killed。两条 positive-delay 格继续因 governor/actor 非原子 transport 保持 NO-LAUNCH。
  三个失败 namespace（两个 importer malloc `rc134`、一个 boot stale-timeout）不是 Reward 结论。0.5 秒
  K100 尚无 checkpoint 分数；v4rg 正反手 TOPP 证书已转入 CPU-only 实跑。见
  [task-revision cutover](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

- Pod2 CPU-only 一次性跑完现役 v4rg 正反手 TOPP run-up：两侧证书均通过 production FK、finite、静止
  frame0、触球行逐位、拍速/拍面、关节/CoP/摩擦/力矩门，进程自然退出且未占 GPU。当前搜索族找到的
  最佳可行上界为正手 `0.98 s`、反手 `0.78 s`，因此没有 0.5 秒动力学证书；这不能反推 0.5 秒绝对
  不可能，仍须 K100 实际回球。NPZ SHA 为 `64f34305…9a6da` / `3a09894b…1f5f7`。见
  [0.5 秒卷](experiments/2026-07/EXP-P1-TIMING-EXAM-0P5.md)。

- task-revision `A6` 已通过 4096-env 两-update generic + specialized full-scene 门：finite
  model-1、fatal0、schema-3/lineage 正确、进程/NVML 自然清空；四个准备时间分层全覆盖（exact 0.5 秒
  `2,406` 样本），同球 revision `176,387=165,417+10,970`，最后触球前接受且 actor 可见 `839`。
  specialized receipt content SHA=`77db7925…d54a`；队列现为 22 格 ready、2 格 transport NO-LAUNCH。
  这只解锁训练，不是行为或半秒回球结论。见
  [task-revision cutover](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

- task-revision full-scene `A5` 已越过 A4 的 CUDA metric-shape 根因、进入真实 PPO iteration 并自然
  完成两次 update，但 finalizer 正确拒绝其 malformed hard contract：Hydra mixture dict 被旧通用
  converter 写成 key list。修复版改从已验证 runtime object 生成 canonical object，并在 sidecar/runner
  前自验新 schema-3；A5 永久 rejected，须新 `A6` 运行门。聚焦回归 `203 passed`。见
  [task-revision cutover](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

- 03:30 CST，task-revision full-scene `A4` 首次越过 4096-env scene import 与 schema-3
  hard-contract 写入，却在 iteration 1 前触发 CUDA env-id 越界。根因是 planner revision 将两个
  `[num_envs]` command metric 错误重绑成 eligible 子集短张量；0.36 秒支路第 18 步缩集而首 rollout
  为 24 步，和现场完全吻合。源码已改为固定全长、逐步清零、按原 id scatter，并补 4096-env/high-id
  partial-reset 回归；`A4` 无 checkpoint 且进程/NVML context 已 absent，修复版仍须 fresh full-scene
  通过后才能点火新池。见 [task-revision cutover](experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

## 2026-07-16

- 17:00 CST，按“先停自动任务、再改训练协议”的要求完成 rolling task-revision cutover：双 Pod 22 条
  接受臂与两条既有 importer rejected job 均已逐项确认进程/NVML absent，Pod1/Pod2 no-clobber receipt
  SHA 分别为 `e6b2480a...8263e`、`4c370431...949`。旧池没有可重建的独立行为窗口，且 formal 179-D
  active swing 冻结目标/TTS，所以不再继续训练或从 EMA 假造淘汰。下一池先闭合同球递增 revision、受限
  phase governor、真正的 0.5 秒卷和 consume-once 整数事件账。见
  [rolling 组合卷宗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。

- 完成 V10 现场八项问题的源码/合同复核：训练侧 rolling TTS 同源补偿、统一 motion retiming 与 Python
  planner 逐样本重规划已经存在；但 VRPN 仍用 host receipt stamp，formal 179 active swing 冻结 target/clock，
  真实击球位置/trajectory residual 未进入训练，0.5 秒行为量尺不完整，调试输出未形成单条关联 trace，且
  command sequence 没有“一颗球/一个 task 只消费一次”的语义。没有运行真机，G07 保持 `Partial`。见
  [G07 八项审计](gates/G07_mujoco_to_real.md#audit-update-2026-07-16-rallyv10-field-test-timing-and-task-lifecycle-gaps)、
  [planner 操作](operations/run_planner.md)与[ROS topic 合同](interfaces/ros_topics.md)。

- 15:36 CST，第二份 registered checkpoint 也完成唯一 Pod2 SSH/no-clobber attestation：
  `rolling_p2_trange_comp2_j0_equal_f03@5200` 的 job-specific exact variants 为
  `691a52c.../428cbf...` 与 `0968d24.../90d7f...`，不能复用第一 job 的 `7878/aee` 摘要。remote actual
  精确匹配 `691a52c...`；receipt content SHA=`37d6bd2...`，checkpoint SHA=`ff1b210...`，
  filename/embedded=`5200`，74 个浮点 tensor / `1,762,715` elements、nonfinite=`0`，hard schema-3
  SHA=`aa80162...`、binding=`7593d66...`、lineage=`0`、process=`live`。第一份 receipt 未触碰，也没有
  judge/stop/retry。同轮 Pod1 仍为 11 live_exact、GPU `4/3/4`、fatal0，latest `model_2600–3200` 全过；
  budget-v1 latest=`3200`、`model_3600` 不存在，未 signal。

- 约 15:10 CST，reviewed historical-claim attestor 对
  `rolling_p2_t05_comp2_j0_equal_f03@5200` 只连 Pod2 一次并成功 no-clobber 发布 receipt
  （content SHA=`521910d...`）：checkpoint SHA=`72dbcb9...`，filename/embedded=`5200`，74 个浮点 tensor、`1,762,715`
  元素 nonfinite=`0`；schema-3 hard SHA=`4e84c51...`、claim=`7878d92...`、binding=`4b9c5b2...`、
  lineage=`0`，取证时 process=`live`。它只证明 checkpoint 身份/finite/合同，不提供行为排序。15:14 CST
  Pod1 唯一只读审计用 `/proc` 专用双读闭合上轮 UNKNOWN：11 条 live_exact、GPU `4/3/4`、accepted
  fatal0，latest `model_2600–3100` 全部 finite/合同/optimizer 正确；budget-v1 PGID `2199057` 只到
  `model_3100`，未出现 `model_3600`，无 signal。既有 importer malloc 失败保持 rejected、未重试。

- 13:46 CST，Pod2 唯一只读 inspector 证明首份 `model_5200` 的 actual immutable claim=`7878d92...`、
  launcher runner=`428cbf...`，claim/binding/process=`live_exact`、checkpoint regular、receipt absent；相对
  当前 `aee7132.../90d7f26...` 的完整 content 唯一差异是 continuation runner SHA，corrected budget、
  题目/source/run/slot 全同。milestone attestor 因此改为从独立 YAML contract 对每个 job 完整重建并精确
  匹配 `428cbf...` 与 `90d7f26...` 两个 reviewed corrected-budget 变体，再把 actual digest 交给 runtime；
  旧 budget-v1、第三 runner 或任何其他字段漂移仍拒绝。Pod1 同轮 source/claim/GPU/fatal 健康，但自定义审计
  对 `/proc` 伪文件误用 regular-file size/mtime 门，identity/checkpoint 刷新安全标 UNKNOWN，未停止任何臂。

- 12:45 CST，Pod1 单连接仍为 `11 live + 1 importer rejected`、fatal0，11 份 latest `model_2100–2600`
  均 embedded/finite/schema-3 hard/claim/binding/lineage 一致；budget-v1 只到 `model_2600`，未触发
  `model_3600` stop。Pod2 的唯一连接按注册命令消费第一份 `model_5200` attestation，但在任何 receipt
  发布或 checkpoint load 前因 actual immutable claim digest 不等于当前 YAML 重建值而 fail closed；未重试，
  训练继续。历史静态复算表明三代 launcher 会因 runner SHA/budget 语义生成 `b639160.../7878d92.../
  aee7132...` 三种摘要，现有错误未返回 actual，不能猜是哪一代。runner 因此新增严格只读、单 SSH 的
  `inspect-milestone-binding`：只稳定自校验 actual claim/binding、进程身份、checkpoint/receipt presence
  并报告字段差异，不物化 runtime、不写 receipt、不 signal；下一轮先诊断，不重复 attestation。详见
  [组合卷宗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)与
  [运行操作](operations/run_lean_training_queue.md#rolling-timing-双-pod-严格续训2026-07-16)。

- 11:29 CST 的 rolling 审计仍为双 Pod `22 live/2 importer rejected`、fatal0；Pod2 两条已出现
  `model_5200`，但尚无 milestone receipt。source/event-schema 复核证明现役 completion/fall 是重叠历史
  EMA，physical-fall union、ready-phase `sum+count` 与母本机器基线均缺失，不能物化冻结的两个
  100-update 行为窗。因此当前 22 条只允许结构淘汰，统一记为“量尺不完整，继续训练”；不得从
  TensorBoard EMA 假造自动 Pareto。rolling runner 已补默认 dry-run、YAML/immutable-claim/source-bound 的
  no-clobber checkpoint attestor；content-addressed runtime snapshot、post-preflight swap、symlink/race/
  mismatch 负测和 generic 回归共 `126 passed`。它只证明 checkpoint，不运行 judge 或 signal。详见
  [组合卷宗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)与
  [运行操作](operations/run_lean_training_queue.md#rolling-timing-双-pod-严格续训2026-07-16)。

- 固定 Isaac Lab 2.1/PhysX 源码审计纠正了球空气动力的作用点说明：`position_data=None` 是 link
  transform origin，不是 COM。现有三类球均为原点居中的单一 `SphereCfg`，所以行为仍等价；standalone
  in-loop 检查新增 exact-zero local COM offset 门，未来资产一旦偏置就 fail closed，而不是静默产生
  `r×F` 转矩。详见 [G04](gates/G04_sim_modeling_mujoco_isaac.md)。

- 10:20 CST 的 rolling timing 单连接/Pod 只读审计确认已真实消费 `24/24` 个唯一 claim：`22` 条 live/fatal0，
  Pod1/Pod2 分别三卡 `4/3/4` 与 `4/4/3`；另外两条在首迭代前因动态 URDF importer malloc `rc134`
  退出，精确进程和 NVML context 均 absent，按基础设施拒绝保全且不自动重跑。随机横向躯干推力不再
  由 `qdot` 冒充；待 trainer hard-contract 与 full-scene dynamics-response 门通过后，优先用释放槽做
  同母本 no-force/force 配对。旧 budget-v1 诊断臂当前 `model_2200`，未到 exact-stop 的 `model_3600`；
  Pod2 最快两条为 `model_5000`，未到本母本 `+500/model_5200`，故当前无合法行为淘汰。详见
  [组合卷宗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)与
  [横向扰动卷宗](experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md)。

- 随机横向躯干推力已补齐 default-off trainer E1 接线：Hydra 只允许冻结的 L0 零推力同调度对照或
  L1 `0.04–0.08 m/s` recovery/hold treatment 与 uint32 题种子；启用时 checkpoint hard contract 绑定
  schedule/safety/Isaac explicit-COM backend，以及 active EventManager term 的 exact、typed、JSON-safe 参数值
  manifest/SHA；pinned `SceneEntityCfg` 会绑定 selector 与 resolved ids，EventTermCfg 全行为字段和 plain function
  source identity 也入账；未知/非有限/callable 参数、decorated/method func 与 interval writer 都 fail closed，
  每步前后重验可抓 attach 后漂移。训练日志有 opportunity/
  command/backend-accepted/abandoned/zero-write、质量与冲量标量，且不无界保留 4096-env receipt；这里的
  backend-accepted 只证提交边界，不证 solver consumed。聚焦与相邻回归 `173 + 107 passed`；没有 Pod/full-scene/
  solver-response/throughput/checkpoint，`launch_authorized=false` 不变。见
  [实验卷宗](experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md)与
  [接口合同](interfaces/lateral_perturbation_adapter_contract.md)。

- rolling fill 的本地等待由全局逐条串行改为每批 Pod1/Pod2 各至多一条并发，同 Pod 仍由 host Kit lock
  串行。两 future settle 后才继续；部分失败保留 sibling 成功 claim 并停止后续批次，绝不自动 retry。
  同一进程 attempted overlay 还拒绝 snapshot 短暂漏 claim 时重提交 job；rolling+generic runner
  `92 passed`。该改动只缩短点火墙钟时间，不改变训练 recipe。

- rolling continuation 首条真实点火抓到 RSL resume budget 语义：parent `1600` 下 CLI
  `max_iterations=3601` 实际日志为 `1601/5201`，字段表示追加 updates 而非绝对终点。本地等待已在 remote
  watchdog 前退出，健康 trainer/证据保留且未重发。runner 修为 trainer arg `2001` + claim absolute
  exclusive bound `3601` / 最后 checkpoint `3600`；首条仅作 schedule-v1 inexact 诊断并计划在
  `model_3600` 精确收口，其余格使用修正合同。
  详见[组合卷宗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。

- 24 格 rolling timing 组合有了独立 continuation runner：相对 parent 的
  `+200/+500/+1000/+2000` 会转为绝对 checkpoint，三份 parent 必须在原 Pod 通过 checkpoint/hard/claim/
  binding、actor+critic、finite 与完整 optimizer 只读核验；激活状态、full-scene evidence 和 runner bytes
  都有 fail-closed allowlist/SHA 门。runner 与 generic queue 共 `88 passed`；三份 parent 已用每 Pod 一条
  只读 SSH 通过，24 条 dry-run 精确为四轮×六卡且每卡四条；该条记录的是点火前 source 门，真实运行态
  以上方最新条目为准。
  详见[组合卷宗](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。

- 真实测试触发的 training-critical 修复已进入 `main@704bf3a2`：actor 可显式消费同源延迟的
  position/velocity/face/side/TTS 元组，并按已知 step delay 更新剩余击球时间；schema-3 题库现在可与
  约 `1.0/0.7/0.5 s` 老师动作 retiming 合用，但绝对物理出球答案不被错误缩放。Pod2 新增 10 个专项
  cases 全过，全集 6 个失败与父提交逐项相同。该能力只过 source gate，下一步是 4096-env probe 和
  [24 条单 seed 工程组合](experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。

- rolling source 的 Pod2 4096-env×2-update probe 已自然 rc0：0.7 秒 compensated 配方写出 finite
  `model_1`、schema-3/fresh lineage 且 fatal0，结束后进程/GPU 为空。strict finalizer 另抓到 supervisor
  Popen 后 `/proc` starttime 首读竞态并 fail closed；训练本体证据接受为工程点火门，probe 不自动重跑，
  identity capture 修复并行处理。

- 05:29 CST 的全池只读审计逐条覆盖双 Pod `24/24`，而非只看新到里程碑的候选：两边均三卡
  `4/4/4`、全部 live/fatal0，24 份 latest checkpoint 的 embedded iteration、finite、hard/claim/binding
  与 lineage 均通过。Pod1 12 条约到 `model_1000–1200`；16 秒自由臂有最强 matched 方向，但 10 秒近似
  打平、24 秒无优势，Reward 近似均分最均衡，单项重押和双倍总强度都有跨项或 fall 代价。Pod2 七组合中
  五条已过 `+500/model_4000` 完整性门、两条在 `model_3900`；五条保留线已到
  `model_1400/1500/4500/4500/4600`。eligible/activation 分母缺失且 Pod1 exact-hit 仅约
  `0.47%–0.54%`，故只记方向、不排名、不停臂。07:47 的下一轮双 Pod 单连接均未在限时内返回输出，
  记 `UNKNOWN` 而非训练失败。见[Pod1 十二格](experiments/2026-07/EXP-P1-POD1-LONG-BALANCE-REWARD-GRID.md)
  与[演示组合](experiments/2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md)。

- 04:32 CST 的最后一份完整可信快照为两台 Pod 各三卡四路、共 `24/24` 条 trainer。该快照的 Pod1 为 `12/12`
  live、每卡 `4/4/4`、fatal=`0`；12 条 latest checkpoint 均 finite、hard-contract/claim 与 fresh
  lineage=`1` 匹配，且全部至少到 `model_800`，其中 16 秒普通对照已到 `model_1000`。本轮 Pod1
  单连接刷新在远端检查开始前因本地审计程序 `SyntaxError` 退出，故当前状态记 `UNKNOWN`；没有远端
  写入或 signal，不记训练失败。Pod2 的七个
  model-3500 演示续训候选均从
  policy/value/optimizer 完整恢复并真实越过首迭代，PID 为
  `426506/427190/428347/431061/431910/432838/433601`；三卡 `4/4/4`、fatal0。七条均已写
  `model_3700`，冻结的 `+200` checkpoint 完整性门全部 `PASS`；机制激活与行为仍待后续仪表判读。原自由臂/保守模仿两次首迭代前基础设施失败
  继续保留为 rejected，唯一 recipe-identical retry 均已成功。前四条 PID
  `426506/427190/428347/431061` 的 `+500/model_4000` 已通过 embedded iteration、finite、
  hard-contract/claim 与 lineage 完整性门；后三条尚未到该点，不是失败。当前没有 activation/eligible
  计数，不能排名或停臂，fall-rate 只作诊断；仍无行为胜者，后三条继续到 `+500`、七条再到 `+1000`，
  且不以稀疏零值误杀。
  见[实验卷宗](experiments/2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md)。

- Demo hot-start 的自由非击球臂行与普通母本保守模仿行分别在首迭代前以 malloc `rc134`、content-bearing
  stale timeout `rc125` 结束；exact PID/PGID/starttime 均已确认 absent，完整 claim/binding/log/launch/
  identity SHA 已绑定，旧行标 `rejected` 且不再占调度槽。新增两个 recipe-identical 的一次性人工
  `retry_v2`，使用新 namespace、硬绑 GPU1→GPU0 并按 claim 顺序错峰；`automatic_retry=false`，本提交
  未点火。点火前同一 GPU 锁会重核旧 5/7 个证据 SHA、旧 PGID/成员 PID 与 NVML context；leader 退出但
  child 仍活的攻击测试 fail closed。前七条 claim digest 不变，`32` 个专项测试通过。见
  [实验卷宗](experiments/2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md)。

- 在不改变前六条续训 recipe/claim digest 的前提下，队列增加 GPU2 第四槽的第七条 16 秒长回合候选：
  从 qdot model-3500 snapshot 继续，组合 V1/V2、强速度/拍面引导、脚朝向与自由非击球臂，专门观察
  单 episode 连续 3–4 拍累积的平衡债。episode 长度在独立 base 中唯一设为 16 秒；claim 绑定
  `+200` 结构/激活、`+500` 安全/平衡、`+1000` 候选排序，稀疏命中为零不得早停。`19` 个专项测试通过；
  尚未远端启动，G05 仍为 `Partial`。见
  [实验卷宗](experiments/2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md)。

- 为次日演示冻结六条 Pod2 model-3500 严格续训组合：三类母本、两档拍面引导、两档 qdot、自由非击球臂
  与脚朝向以组合方案而非伪因果格运行。v2 先只读审原始 claim/binding 与非空全 optimizer，再唯一消费
  `O_EXCL` 只读 parent snapshots；launch 必须从日志证明 iteration-3500/optimizer strict resume、显式合同
  mismatch 和新 qdot/conditional-face hard binding。v3 又在 trainer 前同 GPU lock 内重验四个 snapshot SHA，
  并要求 binding 的 `/proc` PID/PGID/starttime/cmdline 存活到真实 `Learning iteration >3500`；失败只记 exact
  identity、不 signal。generic fresh-only queue 未
  放宽。Pod2 今晚采用实测四路/卡：六个 model500 保全且 GPU0/GPU1 各 `<=3` 时先上前两组合，其余四条
  等四个弱臂精确退出后补齐并保留 V1-only/foot-`-0.6`。02:10 CST 唯一 parent inspect/attest 已通过，
  v2 receipt file SHA `fd200bd6...f2f34` 与三套 checkpoint/hard/claim/binding 已回填，六行现为 ready；
  尚无后代 Pod 行为结果。见
  [实验卷宗](experiments/2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md)。

## 2026-07-15

- Pod1 获重新授权后已按三卡四圈铺满 12 条不同问题的单 seed 长曲线：非击球臂模仿开关 ×
  10/16/24 秒连续 episode，以及六种击球位置/速度/拍面 Reward 配比。两个 attempt-1 在动态 URDF
  import stale 门按 exact PGID 收口；各自唯一同配方 retry 均过首迭代。16:40 UTC 三卡各四条、
  GPU `97/93/97%`，12/12 accepted PID=PGID/fatal0；尚不作行为结论。见
  [实验卷宗](experiments/2026-07/EXP-P1-POD1-LONG-BALANCE-REWARD-GRID.md)。

- 15:33 UTC 只读复核确认 Pod2 GPU0/GPU1 已完全空闲；GPU2 的 qdot/V1+V2/control 三条 10000-update
  长训约到 `3193/3201/3205`，latest `model_3100.pt` 的 finite/schema-3/fresh lineage/contract/claim
  均 exact，fatal0。真实 exact-hit 仍稀疏，当前不能定论。已预注册六条共同 seed3 的长曲线来补齐
  模仿 `2×2`、关节速度 `0/-1/-2.5/-5` 与脚部朝向 `0/-0.3/-0.6`；按 GPU0/GPU1 逐圈发射，
  稀疏击球样本不足时不早停。见
  [实验卷宗](experiments/2026-07/EXP-P1-LONG-SCALEOUT-SIX-ARM.md)。

- 六格首次铺池已有五条越过真实首迭代；只降低击球窗模仿的 attempt-1（PGID `420947`）在动态 URDF
  import 以 `malloc(): invalid size` / rc134 自然退出，无 checkpoint。证据与 namespace 保留，它不是
  Reward 负结果；只给逐字相同配方一个全新 namespace 的唯一重试，同 phase 再失败则转 importer 根因线。

- V2-only 唯一 retry-v2 PGID `423502` 已越过首迭代。15:49 UTC，Pod2 GPU0/GPU1/GPU2 均恰有三条
  trainer，利用率 `97%/97%/91%`；六条新格 exact PGID 为
  `419643/420298/421479/422126/422783/423502`，除已归档 attempt-1 外 fatal0。当前只是满池启动证据，
  新格尚无 model-200，不作行为胜负。

- 连续等待/恢复新增独立 frame-0 v2 design contract：揭题前用上一公开动作自己的第 0 帧零速度参考，
  原子揭题后才切新动作自己的第 0 帧零速度参考；XY 只在阶段入口捕获一次，连续 episode 不
  teleport/reset/清 history/action/delay。Ready 仍是全部安全与可达容差合取。旧 A/B/C prereg 保持
  `17008` bytes / SHA-256 `ca7806df...0616` 不变；CPU validator 红队 `25 passed`，`launch-check`
  按设计 rc1。现役 hold 的 default-stand、未 hold-zero 的 anchor velocity 与 live per-tick XY reanchor
  尚未修，所以没有 Isaac/Pod/行为结论。见 [T1 接口](interfaces/t1_event_training_contract.md)、
  [恢复实验](experiments/2026-07/EXP-RECOVERY-TUPLE-ABC.md)和
  [操作](operations/run_phase1_recovery_tuple_prereg.md)。

- 稀疏 Reward milestone 早筛已补 E1 资格账本：同一步整数链覆盖 exact strike→virtual
  capture→解析 net/landing/legal return（分动作）与 qdot observed→active→excess；只写 receipt 的五态
  classifier 固定总 `100`、每动作 `50`、连续两个 milestone，任何状态都不自动停 trainer。focused
  `14+4+18 passed`；旧 live source 不可回填，PhysicalBall Phase B 仍未测。见
  [实验](experiments/2026-07/EXP-P1-SPARSE-REWARD-ELIGIBILITY.md)与
  [接口](interfaces/sparse_reward_eligibility_ledger.md)。

- Pod2 GPU2 的无随挥回放长曲线已真正点火：qdot/V1+V2/control-retry exact PGID
  `411519/412204/412899` 均返回 `KIT_BOOT_READY`，04:15 UTC 分别到 iter `24/9/2`，fatal0 且
  claim/binding present；GPU2 `97%`、17154 MiB。GPU0/1 的 Yikang PID `379550/396374` 未触碰。
  当前无 model-200，不作 Reward 结论。见[实验卷宗](experiments/2026-07/EXP-P1-LONG-NO-REPLAY-FUNNEL.md)。

- 新的 10000-update 无随挥回放漏斗已在结果前冻结：普通对照、关节速度边界惩罚 `-5`、击球窗模仿放松
  三格同 source/题库/seed，只落 Pod2 GPU2。exact source `2c2d70d...607e` 的 4096-environment
  full-scene terminal probe 与三格 no-Kit Hydra compose 均通过；host 队列/调度回归 `54 passed`。
  200/500/1000 只用于机制与趋势早筛，2000/3000 看中段，6000/10000 看完整曲线；暂不授权第二
  seed/judge/晋级。见
  [实验卷宗](experiments/2026-07/EXP-P1-LONG-NO-REPLAY-FUNNEL.md)。

- 上述漏斗的普通对照 attempt-1 在首迭代前停在动态 URDF import：日志 180 秒无进展，watchdog 只按
  exact PGID `410589` 保全 pre-TERM/pre-KILL identity 后收口，`rc=125`、无 checkpoint；另两格未 claim。
  这不是 Reward 失败。旧 namespace 已 rejected；队列先发两个未消费 treatment，再允许普通对照逐字相同
  配方的唯一 retry-v2，以减少基础设施故障造成的 GPU 空等。见同一[实验卷宗](experiments/2026-07/EXP-P1-LONG-NO-REPLAY-FUNNEL.md)。

- post-swing teacher v3 的唯一 attestor attempt-2 已在 Pod2 从 clean detached `a38b7e9` 自然 rc0：固定
  authorization 来自 clean `main@ff9a253`，4096-state receipt 为 4103 bytes / SHA-256
  `e20a6989...d2aba4`。PGID `403786` 已 absent；merged-main controller status 对 immutable v3 plan、原
  capture producer、attestor source 与授权复核后给出 `teacher_receipt_binding_exact=true`。这只解锁独立
  first-reset full-scene probe；科学 pair、第二 seed、judge/promotion 仍 blocked。见
  [机器结果](../configs/phase1_post_swing_teacher_capture_attempt_v3_result_20260715.json)、
  [实验卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)和
  [操作](operations/run_post_swing_teacher_capture.md)。

- Franco 反手拉 B 的 v4 桌网门已在 clean Pod2 `main@c047ea7` 完成 full dry-run 与唯一 audit：1201 个
  400 Hz 样本逐帧 `37×4` 对，hard/warning/unsafe=`0/0/0`；15064-byte certificate
  `93fd5435...9b0e7` 只声明诚实 saturated lower `0.099999999999 m`，pair/midpoint/time=null，并通过独立
  只读复核。B 现只解锁 vendor 动力学/平衡门；连续时间、RL、回台、Gate3 与真机仍未证明。见
  [桌网卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)和[G08](gates/G08_blind_spot_improvements.md)。

- B 桌网 v3 虽在 clean `main@b9b011b` 产出 hard/warning=`0/0` 的 certificate `39d6cc38...79a19`，
  exact-semantics review 仍正式 REJECT：reporting cap predicate 只证明 `>=0.1-1e-12`，旧 aggregator 却把
  saturated 默认值 `0.1` 写成 certified lower bound。旧文件保持 immutable 但不算 `table_net_complete`，
  dynamics 继续 unauthorized。schema-v4 把 pair/全轨 saturated lower 统一为 `0.099999999999`，null
  pair/midpoint/time 与 saturation flag 语义闭环，并换到独立 v4 namespace/name。初版 v4 commit
  `7241157` 又在合入/运行前被红队判 NO-MERGE：hard/warning 复用 reporting epsilon，会接受
  nextafter/half-epsilon 的门槛下方值。修正版把 5 mm/20 mm 安全门改为无 epsilon 的 finite
  `distance>=threshold`（non-finite fail closed），epsilon 只保留给 reporting cap/bisection；两处边界反例
  均闭环。focused `47 passed`、完整 B chain `148 passed`、static PASS。见
  [机器拒绝记录](../configs/motion_backhand_loop_b_table_net_v3_rejected_result_20260715.json)、
  [实验卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)和
  [运行操作](operations/run_motion_backhand_loop_b_table_net_clearance.md)。

- B 桌网 schema-v2 在 clean `main@f214a80` 的 Pod2 CPU `dry-run` 又于几何循环前 rc2 fail closed、
  certificate absent：bound 724-byte runtime joint-order 是一行 `#` 说明加 31 个唯一关节，upstream L0
  明确过滤 blank/comment，table/net snapshot reader 却只过滤 blank，误计成 32。schema-v3 复用 exact
  upstream comment 语义且不改文件/顺序；未标注 metadata 与 duplicate 仍拒绝。log SHA-256
  `5c9a5940...f92d`；focused `39 passed`、完整 B chain `140 passed`、static PASS。合入 review 前不重跑，
  G08 仍 Partial。见[实验卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)和
  [运行操作](operations/run_motion_backhand_loop_b_table_net_clearance.md)。

- B 桌网门的首次 Pod2 CPU `dry-run` 在 1201 帧循环前因 geom ID 假设 rc2 fail closed、没有输出：MuJoCo
  会把新增的四个 worldbody geom 编为 `1..4`，故 child-body robot geom 只发生确定性整体 `+4`，不是
  37 个碰撞体漂移。schema-v2 仅归一化这个精确 shift，并继续逐项绑定 robot 顺序/名字、topology、qpos0、
  collision row/mesh 与 frozen collision SHA；`1e-9` 漂移和非 `1..4/+4` 反例均拒绝。focused
  `36 passed`、完整 B lineage chain `137 passed`、source/static PASS；合入并 review 前不重跑，G08 仍
  Partial。见
  [实验卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)和
  [运行操作](operations/run_motion_backhand_loop_b_table_net_clearance.md)。
- post-swing teacher capture v3 已在 Pod2 GPU2 自然收满 `4096/4096` 条 finite `natural_clip_wrap` 状态并
  自然退出；claim/states/result SHA-256 为 `81126b27...244e` / `8d07668e...95d8` /
  `0aa2f37f...d641`。但 one-shot attestor attempt-1 在 receipt 写入前因 canonical content 与 JSON
  document 末尾换行混用而 rc2 假拒绝，`teacher_receipt.json` absent。源码修复已拆分无换行 content
  digest 与单换行 document bytes；attempt-1 在 `_claim` 停止，后续 checkpoint/lineage/source/motion/速度门
  都尚未执行。修订后的 attestation schema 2 又把原始 producer `capture_source` 与修复后 consumer
  `attestor_source` 分开，status 分别对回 immutable v3 plan 和 tracked retry authorization；交换、重绑、dirty
  均负测；另加 main-tracked one-shot retry authorization，把唯一 attestor commit/SHA 绑定 v3
  plan/capture/checkpoint/output，拒绝任意 clean HEAD 自签自验。授权固定 attestor `a38b7e9e...293cf` /
  `03611b56...310f` 与 authorization `87fd1c71...dfda`；attempt-2 尚未执行。补齐 consumer 后六文件 host suite
  `181 passed`（一个既有
  duplicate-ZIP warning）。修复合入 main 前禁止重跑
  capture/attestor，首 reset 与科学训练仍 blocked。见[机器结果](../configs/phase1_post_swing_teacher_capture_attempt_v3_result_20260715.json)、
  [实验卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)和
  [操作文档](operations/run_post_swing_teacher_capture.md)。

- post-swing trainer consumer 红队补出一处 NO-MERGE：旧 loader 只检查 source tuple 的 hex 形状与 clean，
  合法 40/64-hex 重绑仍可通过。successor 现在要求训练配置同时提供 tracked retry authorization 路径/SHA，
  从该 exact byte snapshot 派生 capture/attestor tuples 并完整比较，且把规范化 authorization 内容纳入
  schema-3 hard contract；四类合法 hex 重绑与缺失配对项均有负测。仅闭合 source consumer，attempt-2、
  首 reset 和科学训练仍未执行；见[接口](interfaces/post_swing_teacher_artifact.md)与
  [操作](operations/run_post_swing_teacher_capture.md)。

- Franco 反手拉 B 的桌网整轨门已通过独立 source/static 红队：冻结 validator/plan、runtime-order 名字双射、
  四个碰撞障碍和 `1201×37×4` 有限密扫均内容绑定，`<5 mm` 为不可补偿 hard fail；focused `29 passed`、
  完整 lineage chain `130 passed`。这只允许进入 Pod2 CPU 的只读 dry-run，尚无 runtime certificate，
  也不证明连续时间、动力学、平衡、TOPP、回台或 RL。见
  [实验卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)和
  [运行操作](operations/run_motion_backhand_loop_b_table_net_clearance.md)。

- post-swing teacher capture schema-v2 已从 prelaunch 推进到真实 runtime，但在零 inference step 因
  `get_observations()` 返回 `(actor observation, extras)`、旧 play 直接 `.to()` 而失败；v2 只有 bound claim，
  states/result/receipt absent，exact teardown 后永久不重发。successor source 已统一到 actor-only adapter，
  拒绝 critic-only/坏结构，并保证 wrapper 在正常/初始 observation/step 异常均 exactly once close；focused
  source/Hydra tests 通过，Pod 重验仍未做。见[机器结果](../configs/phase1_post_swing_teacher_capture_attempt_v2_result_20260715.json)、
  [实验卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)和
  [操作文档](operations/run_post_swing_teacher_capture.md)。

- 横向躯干扰动的 source-only Isaac adapter 已通过独立红队：每个 physics substep 以当前
  `torso_link` 显式 WORLD COM 提交力，same-tick/reset 竞争 writer、异常后的 terminal zero、motion inode
  替换和 output no-clobber 均有反例；focused `65 passed`。它尚无真实 full-scene、solver response、
  direct-setter 独占或 throughput 证据，`launch_authorized=false` / `training_authorized=false` 不变。见
  [实验卷宗](experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md)和
  [运行操作](operations/run_lateral_perturbation_runtime_probe.md)。

- post-swing capture schema-v2 controller/builder 已闭合九类 pre-launch blocker：历史 teacher lineage、
  Pod2 physical GPU2 UUID/共享 lease、absolute byte-bound tools、safe env、timeout compose、same-PID handoff
  和 status 防重绑均有负测；`plan` 现与 `launch` 共用 exact cwd/env/argv/timeout 的只读 Hydra compose，
  compose 前后复核且失败不消费 namespace，成功绑定 output digest/bytes/elapsed；按 operation 所列四文件
  在可导入 Hydra 的本地环境复现为 `41 passed`。只完成 host source gate，未连接 Pod、未 capture；
  详见[实验卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)和
  [操作文档](operations/run_post_swing_teacher_capture.md)。

- post-swing capture 的 seed-parity source blocker 已闭环：`play.py` 现在拒绝 bool/float/string、负数与
  uint32 越界 seed，并在创建环境前把同一个冻结值写入 env 与 PPO runner；真实 Hydra compose 负测也
  逐项拒绝三个 train-only checkpoint 键。该提交不运行 Pod、不追认失败 v1，也不单独授权 successor；
  schema-2 prereg、4096-environment capture 与首 reset 仍保持 fail closed。见
  [producer operation](operations/run_post_swing_teacher_capture.md)和
  [measurement rerun 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)。

- clean base-decel 两臂已自然终档，`model_1000` filename/embedded/finite/fresh/claim/common hard exact；
  980--1000 的 raw base speed 比为 `1.00882x`，按冻结 `<=0.90x` 门正式 reject，不买第二 seed/judge。
  同步源码语义审计发现 Reward 实际追踪随 racket-target 距离变化的 `v_des`，现有 primary 却只测未分桶
  raw speed；尾窗 raw-kernel-per-eligible 提升 `1.6003x`。冻结 verdict 不变，后续先另立
  `|v_base-v_des|` 分桶量尺。见
  [clean main-effect 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md)。

- post-swing 外生 teacher 的首个 inference-only capture 已冻结为机器预注册
  [`phase1_post_swing_teacher_capture_prereg_20260715.json`](../configs/phase1_post_swing_teacher_capture_prereg_20260715.json)：
  exact main source、fresh measurement-control `model_500`、两条动作、schema-3 bank、ignored A3 tree、
  Pod2 GPU1、4096 条 natural-wrap 状态、20000 inference-step 上限和 root 速度上限全部在数据前绑定。
  只授权一次 capture；attestation、首 reset、科学训练、第二 seed 与 judge 仍逐级 fail closed。
  随后的 v1 Hydra compose 在任何 capture directory/claim/process/GPU work 前 rc1 fail closed：派生器遗漏
  三个 train-only checkpoint 键；源码复核还发现 play 未实际应用冻结 seed。v1 证据保全且不重发；上述
  seed parity 与 controller/builder 已关闭源码缺口，但 v2 仍必须使用全新 namespace 并逐级过 runtime 门。

- clean base-decel 的 `model_500` 两份 receipt/finite/fresh/claim/common hard exact，step 0–500 activation
  全过且 480–500 尾窗两臂都有真实 V2/exact-strike 分母。treatment/control 底座速度=`1.13669×`
  （FAIL `≤0.90×`）、signed-face pass 差=`−0.16609`、composite 差=`−0.06942`，解析回球降到
  `0.49583×`；虽 pre-fall `−0.03287`、velocity pass `+0.10617`，当前 weight=`1.0` treatment 仍按
  单 seed screen reject。按冻结合同 trainer 继续 +1000，只收终档，不买 seed/judge/晋级。见
  [clean main-effect 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md)。

- Franco 反手拉 B 的 vendor L1 已在 Pod2 exact `main@7dec698` 闭环：runtime→GMR/MJCF 31-joint
  name permutation 修掉第二个 harness 假拒绝后，full `dry-run` 与唯一 `O_EXCL` audit 通过。1201 个
  400 Hz 有限样本自碰/`<5 mm` 自打/warning=`0/0/0`，最小余隙 `0.1382918358 m`；certificate
  SHA-256 `6840df34...db60`。这不是连续时间或动力学证明，只解锁独立桌网整轨门，训练仍 blocked。
  详见 [B vendor L1 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)。

- S0/M0 exact-GMR attempt-v2 在 Pod2 clean detached `b75204d` 上再次证明 source/static 合同正常：两份
  `static-v2` PASS、plan SHA exact，两个 `exact_gmr_v2` root 与 shared consume lock 执行前后均 absent。
  runtime `inspect` 在进入 consumer 前以 rc127 fail closed，因为合同绑定的
  `/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10` 连父环境都不存在。后续只读恢复审计
  又确认 exact GMR tree/283 MB bundle、SMPLX/model/mapping 与 S0/M0 七份 canonical 输入全部 absent；
  最接近的 Isaac venv 只与 234 行冻结环境精确重合 87 行，不能无猜测重建 v2。没有 GMR 输出，
  不是动作/脚距失败；在该 Pod2 location 两批均不得 consume，须先权威恢复资产再建隔离 v3。2026-07-20
  后来回收的 Pod1 S0/M0 completions 已取代“全局 absent/未 consume”推断，但不改写这次 rc127 事实。见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)。

- B vendor L1 的第二次 CPU `dry-run` 在 dense 704 报 ankle 超限 `0.656861334 rad`，只读复算证实是
  runtime-order column 23 elbow 被按 GMR-order column 23 ankle 解释的 adapter 假拒绝；真正 ankle
  在 column 14 且合法，L0 按名字得到 max excess 0。L1 已在 densify/range/qpos 前加入冻结名字表的
  byte-preserving 31-joint 双射并报告 permutation，duplicate/missing/drift 负例通过；不改 range、B/C
  或动作字节。等待合入后 clean runtime 重跑，certificate 仍不存在、G08 保持 Partial。详见
  [B vendor L1 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)。

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
  SHA/stale/body-failure 负例通过。合入后 clean runtime 已越过 import，并暴露本节上方的 joint-order
  adapter 假拒绝；G08 始终保持 Partial。详见
  [B vendor L1 卷宗](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)。

- clean base-decel 单变量 pair 已按同一次顺序事务在 Pod2 GPU1/GPU2 越过首迭代：control/treatment
  exact PID=PGID `385320/385948`，claim SHA-256 `a039226a...1746e` / `673bf6c6...9392`；GPU0 的 Yikang
  进程未触碰。04:27 CST 只读复核时 TensorBoard 到 step `106/89`，日志 fatal=0，两臂五个
  post-swing 计数在全部已写 update 严格为零；raw base-decel 两边均激活，weighted Reward 只在
  treatment 非零。尚未到 `model_200`，不比较行为、不买第二 seed。详见
  [clean main-effect 卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md)。

- post-swing 外生 teacher cold-start 的首版及其伪 private-capability 修订均被红队否决后已完成 source 修复：
  receipt/claim/NPZ/raw result 改为单 fd/单 bytes 消费；capture 收回 `MotionCommand` live-state 路径并由
  `O_EXCL` claim 占有 namespace，不再暴露 arbitrary-array writer，也不把 callback label 当证明；独立 attestor
  仅以 `weights_only=True` 实查 checkpoint、
  schema-3 lineage/claim、相邻 hard contract、两份 clean source、motion/joint order 与速度 limits；首 reset
  另绑 adopted count/fraction、概率偏差和 state readback。dependency-light 攻击专项 `13 passed`，但尚未跑
  4096-env Isaac probe，故仍 `Partial` / `launch_authorized=false`。详见
  [接口](interfaces/post_swing_teacher_artifact.md)、
  [操作](operations/run_post_swing_teacher_capture.md)与
  [实验卷宗](experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)。

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
  schedule SHA；episode reset 截断会记录 sampled/commanded/backend-accepted/abandoned 冲量且当步禁止重启。
  后续红队新增不可由配置放大的 `0.15 m/s` 冲量、`2.0 m/s²` 加速度、`0.02--0.20 s` 时长和 `200 N`
  WORLD-Y force 硬包络，并把 scheduler→adapter 改成 source-token 绑定的无副作用 preflight + 原子/no-throw
  commit：删除公开 acknowledgement，mass/cast/final wrench/receipt/cache 全在写前 host-visible 校验；坏
  receipt/stale token 不写 backend、不 cache、不解锁，同 step token 可换全新 preflight nonce 重试。commit 进入后异常或非
  `None` 返回会永久标成 `DIRTY/UNKNOWN`；cache 还绑定 live backend token，不能重放给同 SHA 新实例。
  strike/window 现在和 reset 一样逐环境保存 sampled/commanded/backend-accepted/abandoned 恒等式并在中断 tick 真正
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
  新旧 focused `28 passed`、仓内回归 `949 passed, 10 skipped`。本条形成时 v2 runtime 未执行，
  GMR/schema-2/训练/真机仍 blocked；当前以顶部 2026-07-20 completion 回收记录为准。见
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
  consume 已按本节首条完成；本条形成时 GMR/schema-2/安全/效果/训练仍未授权，
  M0 的 foot sites、初末二维脚距、容差和 pass 全保持 null。后来 exact-GMR 诊断已回收，
  M0 stance gate 为 `0/4`；当前详情见 [exact-GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)。本层详见
  [canonical-beta 卷宗](experiments/motion_canonical_beta_s0_m0_20260713.md)。

- S0/M0 的五条 exact GVHMR 结果已增加 post-GVHMR no-clobber consumer：两份 prereg 同时绑定 tracked
  summary、execution record、queue state、每条 binding/audit/PT 和 canonical-beta donor，host static
  两批通过，专项 `8 passed`；后续 runtime handoff 与 canonical-beta consume 已按本节其他条目完成。
  本条形成时 GMR/schema-2 仍未运行；当前 exact-GMR 已有诊断结果、schema-2 仍未授权。S0
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

</details>
## 2026-07-29

- Pod1 对 `5e94f21b` 滚动 20 ms 预测刹车做了 4096-env 同 seed 反例：updates
  `1–16` 平均 `36.48 s/update`、episode `20.19` steps、actual-joint
  `4,791.6 reset/update`，只有 `2` 次 strike opportunity；相对 `5dbb` 既未降低 actual
  reset，最近十窗还约慢 `4%`，故该单变量不再解释为修复。reference-only shadow breach 仅
  `43.3/98,304=0.044%` transitions/update，也排除 reference mode 是当前 mass-reset 主因。
  下一 fresh 候选只把 ActionBall finite executed q_des 在 soft limits 内每侧再留 `5%`，
  四条 loop/block × upper/full 老师均未越出，最小剩余余量约 `0.046 rad`；新增比例已绑定
  runtime/schema-3 training contract，旧 checkpoint 不得 exact resume。source-level 检查不作
  Isaac 证据；下一证据只认 clean Pod `1 env × 2 update` 与 4096 同 seed。
- Reference termination 原文复核否定了“像 q_des 一样 clamp reference error”的说法。
  BeyondMimic 保留 reference hard ET 并以失败段采样；DeepMimic 动态技能的 no-ET 消融明显
  退化；PHC/Stubborn 的改进是选择性/概率化终止并另建恢复/采样机制，不是截断 reference
  error。当前 `metrics_only` 仅为 N1 diagnostic：现役 ActionBall broker 已绕过 BeyondMimic
  failed-bin sampler，且本轮 raw breach 极低，所以暂不热改；最终默认必须在 actual-joint
  主因清除后做 fixed-seed `phase_gated` / `metrics_only` / hybrid A/B。
- `curr-launch-fix` 已把 finite q_des 投影、投影前超出量 Reward（主线 weight=`-5`）、
  reference metrics-only、shared-ready fresh actor bootstrap 与 full-scope post-solver
  预飞合成 exact `b1d299e1` 并推送；590 项整合回归只暴露一个测试 import 笔误，修正后的相关
  107 项全过。随后从该 commit 生成 profile pins（文件 SHA-256
  `47a00a6a...30488`）和 loop/block 的 upper/full 四份 N=1 bundle。upper 均 PASS；full
  loop 为 `511/512=99.80%`，full block 为 `443/512=86.52%`（diagnostic PASS、formal
  canary 阈值未过）。Pod 首次真实 compose 随即抓到 N1 launcher 对未声明键少写 Hydra `+`；
  已修成 `+task.racket.reference_guard_mode=metrics_only`，同时把 bootstrap/std/reference 三项
  接入 N5 formal launcher，并让 full launcher 拒绝无 solver preflight PASS 的旧 bundle，相关
  launcher 回归 `80 passed`。这些只解锁 Pod 真实 smoke，不是训练效果或 Gate 晋级结论；精确工件与边界见
  [Reward 因果实验记录](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)。
- `curr-launch-fix` 功能分支完成 ActionBall reset/吞吐复盘，并形成下一 fresh wave 候选；这不改变
  `origin/main` 的 `docs/NOW.md` 权威状态。旧 v2 也曾在早期由 `ee_body_pos` 产生
  `1,690–2,301 reset/update`，但后期学到只剩 `1–7/update`，说明“早期 reset 多”不是新现象。
  被引用的 `6.4 s collection/update` 来自 mean episode length=`1`、恰好
  `98,304 reset/update` 的失败 probe；同代码修正 stand hold 后代表值为 `4.49 s`，不能把
  `6.4 s` 当健康基线。当前 ActionBall 稳态快照约为反手拉 `27 s/update`（主要
  `ee_body_pos`）和反手挡 `48 s/update`（主要有限 `q_des` 请求终止）；取消后者的有限请求
  reset 预计可省 `14–17 s/update`，仍须 fresh 实测确认。
  候选语义是[有限 q_des 投影执行](DEFINITIONS.md#finite-qdes-execution-projection) +
  [投影前超出量惩罚](DEFINITIONS.md#qdes-projection-penalty)，首发 weight=`-5`、`-20` 只消融；
  reference guard 改为[只记指标](DEFINITIONS.md#reference-metrics-only)，而 nonfinite/实际越限/
  子步 crossing/table/fall 仍 hard reset。四件老师轨迹的全片 hard/soft/2%-inner crossing 都为
  `0`；block 的 normalized hard/soft margin（upper `0.115081/0.072312`、full
  `0.115081/0.072312`）不小于 loop（upper `0.111954/0.068838`、full
  `0.113493/0.070548`），排除“block 老师贴限”作为两动作 qdes 差 180 倍的根因。小时巡检新增
  [`collection_vector_step_wall_s`](DEFINITIONS.md#collection-vector-step-wall-s)、
  [`amortized_e2e_vector_step_wall_s`](DEFINITIONS.md#amortized-e2e-vector-step-wall-s)、
  [`collection_environment_step_us`](DEFINITIONS.md#collection-environment-step-us) 和
  [`collection_environment_steps_per_s`](DEFINITIONS.md#collection-environment-steps-per-s)；
  CaT 连续约束终止与 PPO bound loss 留作后续，不阻塞今晚发射。完整证据见
  [Reward 因果实验记录](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)。
- Pod1 已在 exact `4ff48b21` 上保留三条 `4096 env` N=1 upper 长跑：反手拉
  `current_low`（现行低拍位/拍速/拍面权重）、反手拉 `mimic_x2`（动作模仿两倍）和反手挡
  `current_low`。每个 PPO update 固定含 24 个
  [`vector policy steps`](DEFINITIONS.md#vector-policy-step)，即 `98,304`
  [`environment steps`](DEFINITIONS.md#environment-step-throughput)。11:20 UTC 快照分别为
  `61.69/149.60/271.99 s per update`，PPO learning 仅约 `0.1 s`，其余几乎全是 collection；
  第一批五轮的 q_des hard-limit reason 分别为 `148267/148651/221278`，三臂击球机会均为零。
  11:31 UTC 的最后完整 update 已改善到 `25.38/31.69/136.90 s`，即每个 vector policy step
  `1.0575/1.3204/5.7042 s`；三者仍无 exact strike，故这只是重置风暴逐步减弱后的吞吐改善，
  不是动作质量晋级。
  `mimic_x2` 的三项 raw mimic Reward 已约精确翻倍，但重置行为没有分离，证明该 Reward 不是
  “假接线”，而是在当前死亡/重置尺度下太小。full 反手拉已越过 Isaac 导入后暴露 solver
  admission 下界错配，task-strong direct 则在 PhysX start 活锁后按 exact PGID 留证停止；均未
  伪装成有效 run。逐 run 身份、失败和 step 时序见
  [`n1_live_wave_4ff48b21.v1.json`](../configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json)。
  下一 fresh wave 改为显式 opt-in 的 shared-ready actor 初始化（末层零权重、ready bias、
  `init_noise_std=0.02`），仍保留软/硬限位与 table/fall 保护；N=73 不得被该常量 bias 路径误伤。
- N=1 fresh `_r8` 已把“先产 policy”与 formal 证明层拆开：exact `e469d85b` 的
  `training_authorized=false` diagnostic 保留实际 Reward、q_des clamp、软/硬限位惩罚与
  hard-limit/table/fall termination，但冻结 level-0 curriculum 并跳过 formal Reward/joint
  receipt 和 rollout-end advancement。Pod1 反手拉与 Pod2 反手挡均自然完成 `1 env × 2 update`、
  零 Traceback、`model_0/1` finite，现已各启动 1024-env upper `current_low` canary；x2/x4
  Reward 臂按同 Pod 首 iteration 后串行 boot。Pod1 smoke 的 episode length 约 1，24/24 policy
  steps 为 `joint_qdes_forbidden`，因此前 20–50 updates 重点看是否学会脱离硬限位请求。exact
  六臂历史尝试已由当前
  [`4ff48b21` 运行记录](../configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json)
  接管；复跑规则见[消融发射工序](operations/run_ablation_wave_launch.md#未变配方的诊断续跑快线)。
  这些运行不主张 formal curriculum/Gate 证据。
- N=1 `_r4` 两动作已越过 quaternion receipt seam，但固定 mixture 第 4 个 birth 进入 frontier
  时，level-0 `current width == initial center width` 被旧 sampler 误判成无合法 arm；两条仍是
  `0 iteration / 0 checkpoint`。现改为优先 promoted frontier、否则采当前非零 support 的 outer
  band，并保留 stratum/arm/quota/receipt/exact replay；全零仍原子拒绝。联合回归
  `171 passed, 14 skipped`，pins/bundle 内容不变，下一步 fresh `_r5`。
- N=1 `_r3` 反手拉在真实 Pod 穿透 scene/runtime/Reward/obs/q_des clamp 后，首个 PPO update 前
  暴露 receipt 四元数二次归一化造成约 `1e-16` canonical SHA 漂移；Pod2 未重复执行同一确定性
  失败。修复保留已是单位四元数的 binary64 tuple，非单位输入和符号规则不变；核心回归
  `119 passed, 14 skipped`，新 pins 为 `52000401...f465`，反手拉/挡 bundle 为
  `baad5b95...acbf` / `0d3c80f4...92ab`。下一步 fresh `_r4` 双 Pod 两更新 smoke，通过即发
  upper Reward canary；full-body 不阻塞首批 policy。
- N=1 upper Reward 首次双 Pod 真实构造 smoke 已到 scene/Reward/obs/q_des clamp，但两动作同在
  diagnostic motion payload 生命周期处失败，未进入 PPO；失败目录与 exact 进程证据保留。修复
  diagnostic bytes snapshot、unauthorized receipt、hard-contract formal/diagnostic 分支和
  Motion↔Racket 初始化后 digest probe，host 相关可运行回归 `236 passed`。因 solver source
  改变，已重物化 profile pins。`_r2` 又穿透到 hard contract/Reward receipt，随后在首 true reset
  捕获 birth/task receipt 漏传 broker registry SHA；修复后最终 pins 为 `26eb1ff2...6804d`，
  反手拉/挡 N=1 bundle 为 `c2399571...05d0` / `c53d1669...41a2`；待 fresh `_r3` Pod smoke。
- 新增 ActionBall 1-env formal Reward 因果发射门：真实 post-Hydra/live RewardManager exact
  recipe 对账后，逐 active objective 用权威 tensor 做单轴 worsening，按四组报告 signed
  per-step/per-event 剂量；unknown term、方向错误、clean/HEAD producer/identity 漂移均 fail
  closed。host focused `15 passed`、Reward 相关联合回归 `433 passed`；尚无 clean Pod Isaac
  receipt，仍是 E1。详见
  [实验](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)与
  [工序](operations/run_action_ball_reward_causal_prelaunch.md)。
- 接管 CC 的 Isaac ground-plant 修复：旧 generator 会改写 env origins、让机器人和克隆桌子错位，
  现改为每环境零均值地垫并提供显式 `robot_material_make_consistent=true`。Python 3.8 联合回归
  `379 passed`；同时纠正文档中会把有效摩擦额外乘 1.5 的错误 CLI。证据仍为 E1，未跑 Pod
  Isaac/4096-env，故 fresh N5 首轮冻结为平地 upper/no-move，rough/move 保持 blocked。详见
  [rough ground 实验](experiments/2026-07/EXP-ROUGH-GROUND-FRICTION-FIX-20260729.md)。
- 2026-07-30：N=1 A3 动作专属 dynamic-ready 已完成 source 接线：candidate/Isaac PASS 双 pin、
  motion frame-0 physical/teacher、nominal-hold qdes、reset action buffers 与 fresh actor bias
  进入同一 schema-2 合同，旧路径保持兼容；等待 Pod focused test 与 `1 env×2 → 4096 env×5`。
  同时确认旧 loop/block diagnostic long 都在 update 169 后因无人消费 joint-safety summary 而
  确定性 overflow，不是在继续训练；fresh successor 每 PPO update 排空但仍无 formal
  Reward/curriculum 权，actual-hard/nonfinite 保持 fail-closed。五轮只诊断跨击球窗，学习判断
  至少观察到约 1000 updates。详见
  [实验记录](experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)和
  [发射工序](operations/run_ablation_wave_launch.md#n1-动作专属-dynamic-ready-fresh-发射2026-07-30)。
- 2026-07-30：clean Pod focused suite `63 passed`；唯一初轮失败是父提交也稳定复现的 full
  `bh_block` admission fixture `443→447`，仅刷新测试期望、未改 solver。exact commit profile
  复算与 tracked qvel profile bytes 相同，两动作 dynamic-ready bundle v2 已物化 PASS：
  loop `22672c3d…`、block `69b3b78d…`。recipe-only 入口同时补齐 dynamic-ready 支持，下一门是
  clean Pod 真实 scene 物化各自动作的新 schema-2 policy contract，再写 smoke spec。
- 2026-07-30：loop/block 旧 194-D `4096×5` probe 均产出五份 finite checkpoint，但 update0
  已分别有 `860/864` 个 env 撞 `waist_roll` raw hard；两动作 hard-env Jaccard=`0.982`，
  qdes forbidden=`0`，teacher waist 余量 `0.272–0.303 rad`，故定位为 shared plant DR
  超出 ready 支持而非 Reward/solver/teacher 贴限。fresh actor 已版本化为
  `action_ball_table_pose_twist_heading_task_n1`，把 racket position/velocity/normal 统一到
  yaw-heading frame；N1 launcher 改用 stable-ready plant，暂关 torso CoM/link-mass/PD DR。
  下一步只在 Pod 做 focused parity、两动作 `1×2 → 4096×5`，健康即发 `4096×1001`。
- 2026-07-30：Pod1 三卡已从“一条有效、两条 overflow 后挂住”切换为三条 exact
  `f2c54fc3` N1 milestone：GPU0 反手拉 upper seed0、GPU1 反手挡 upper seed0、GPU2 反手挡
  upper seed1，均为 `4096×1001`、194-D、schema-v2 dynamic-ready 与 stable-ready plant。
  旧 GPU0/GPU2 进程只在复核 owner/PID/PGID/cwd/claim/checkpoint 后精确 TERM，证据未删；新
  GPU0/GPU2 claims 为 `3c523fde…0196` / `7ac32418…e3f`，均已出现真实 PPO update。当前
  这批 194-D observation 没有独立 teacher-start 倒计时，但可由 TTS、目标拍速和 action identity
  精确重建；随后采用的显式 successor 见下一条。full-body 因仍是 schema-v1 且缺 stable-full
  dynamic-ready/hold，不以旧 bundle 冒充对照。
- 2026-07-30：先实现显式 `time_to_teacher_start_s` 的历史 195-D ActionBall actor source
  `action_ball_table_pose_twist_heading_task_teacher_start_n<N>`；随后 fresh N1 已由顶部记录的
  fixed-194 v2 取代，不再喂 one-hot。历史 scalar 直接读取 Motion phase governor；formal
  reset 由 Racket 发布 receipt 后立即复用既有 timing validator，避免首个 actor observation
  假零。旧三条 `f2c54fc3` 194-D long 保持 exact 历史并继续训练，不停机、不重标、不 resume。
  source/依赖轻量合同验证后仍须 Pod `1 env×2 update` 真实构造，action-set/source/claim 必须 fresh。
- 2026-07-31：ActionBall 桌碰后端改成 5 个 table source × 32 个显式有序 A3 body filter；
  Pod focused `214 passed`，五 role/四 physics substep 真实正控与 reset 零泄漏通过，异常进程
  shell rc 不再被 Kit teardown 改写为零。两次 4096-env 短稳态 on/off 定价显示 exact table
  后端平均只增加约 `8.97 ms/policy-step`（约 `0.22 s/24-step update`），不是当前
  `17–25 s/update` 的主瓶颈，因此保留精确接触；table-frame geometry prism 仅作为后端失效的
  保守降级。当前 GPU2 fixed-194 milestone 已到 update 558，`model_500.pt` 的 80 个 tensor
  全 finite；strike-window hit rate 首次到 `0.0042`，但 capture/return 仍为零。继续守护到
  600/1000，同时性能主线转向 per-step host validation packet 与 reset broker 批量化。
- 2026-07-31：新智元 vendor `bh_loop_c` `4096×5` probe 的五份 checkpoint 全 finite，
  但因 launcher 丢失 stable-ready 而产生 `14,086` 次 actual-hard terminal，不放行 long。
  `100` 个入窗拍距中 `97% >0.20 m`，因此下一身份保留 `std=0.075 m`
  精核并叠加 `std=0.30 m`、低收入粗核；同时强制 stable-ready、把 plant-state
  guard 从 2% 提前到 5%，新增 `4096×32` push-evidence 阶段。当前等 clean
  source 重物化后串行 recipe→smoke→probe→push-evidence；权威账本见
  [ActionBall 分阶段准备](experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。
- 2026-08-01：vendor-only 双位置包络完成 source 与 Pod 1-env 活体验证：只给
  waist-roll/pitch 的 PhysX constraint 每侧内缩 hard span 2%，其余 29 轴及
  H_mech/soft/Q/actor/delay/Reward 不变；setter 后与 startup 首 reset 两次 exact getter
  readback 均通过。host 定向 `364 passed`；Pod GPU0 `1 env×1 update` 自然完成，
  actual-hard/terminal=`0/0`，并产出 control/mechanical gap、attempt/capture、dwell、
  side-flip、Δqdot 遥测。该 smoke 不是 long admission；双侧 5 ms ON/OFF stress、
  schema-3/authority/recipe 重签及 fresh `4096×5` probe/push 仍开放，G05 保持 `Partial`。
- 2026-07-31：智元 A3 vendor 的双动作工件链完成到 contact bundle：`bh_loop_c` 与
  `bh_block` 均从 action-specific runtime authority 生成 dynamic-ready candidate，在 Pod2
  自然完成 `0.8 s / 160 physics step / 40 policy step` nominal hold，双脚接触率 1.0、无
  terminal，随后物化 `landing_claim=false` contact bundle。loop 的
  candidate/hold/bundle SHA 为 `847ffe78…/22010d5d…/bd91b652…`，block 为
  `3692f4c3…/49948c41…/bbfb612a…`。两次 loop 前置失败分别是 private OpenGL 与 fresh
  checkout `PYTHONPATH` 缺失，均未创建 scene/receipt；已把 exact private runtime 恢复路径写入
  setup/runbook。contact pins 关闭后仍须通过同源 VendorV1 双包络 stress 与
  `4096×5 → 4096×32` shared-safety 门，才可发今晚三条 long。
- 2026-07-31：8-env 双位置 stress 的旧 Pod hang 定位在 `gym.make`，发生于 reset、live-limit
  写入和唯一 5 ms sim step 之前；同时确认旧 harness 绕过了 long 的 VendorV1 Hydra leaf，故
  不归类为 PhysX FAIL。探针现改为 compose `HOPEPingPongActionBallA3VendorV1` 并复用
  `train.py::_apply_task_overrides`，从 action registry 解析 N=1 motion/manifest，保留 `[0,2]`
  delay、2% Hctrl、6% guard 与原 ON/OFF stress 数学，新增逐阶段 marker。host focused
  `20 passed`、独立红队无 P0/P1；下一步是新 clean commit 的 no-clobber Pod receipt。
- 2026-07-31：loop/block contact bundle 的 registry pins 已对 tracked bytes 逐层重算，八个
  runtime-contract / required-identity / runtime-authority / contact-bundle SHA 全一致。block
  旧“未物化”测试改成 exact pin 正向校验，并保留 loop↔block 交叉和篡改 contract 拒绝。
  host 集成回归 `238 passed`；upper contact bundle `11 passed, 6 full-body deselected`，后六项
  因本机 Python 无 Torch，留到 Pod Torch 环境补跑，不将其误记为代码失败。
- 2026-07-31：clean `07ba61cf` 的 8-env stress 已越过 VendorV1 bind、`gym.make`、reset、
  Hctrl/mixed readback 与唯一 5 ms sim step，证明旧 manager 构造 hang 已解除；但旧生命周期在
  `_run_live.finally` 先调用 Isaac `simulation_app.close()`，其 `os._exit(0)` 抢在外层
  validate/receipt publication 之前，故 v2 只有完整 marker log、无 receipt，不作 PASS。
  lifecycle 已改为先 restore/validate/re-attest/no-clobber write/flush，再由 main 关闭 Kit；FAIL
  或未发布路径用 nonzero hard exit，focused `21 passed`。v2 路径 spent，待新 clean source v3。
- 2026-07-31：clean `9a08d979` stress v3 首次产出真实 FAIL receipt `ec1eebc…`；FAIL 是
  diagnostic 聚合预期 `2/1/1`、实际 `3/1/1`，没有形成 mechanical verdict。两条件初态的
  pre readback 正确贡献 attempt=2，post readback 又诚实记录 OFF 持续 attempt=1、ON capture=1、
  OFF penetration=1。修复不放宽为 3，而是 pre/post 分相消费并分别要求 `2/0/0` 与 `1/1/1`；
  validation FAIL 同时保留真实 restore、raw observation/diagnostic failure evidence。host focused
  仍为 `21 passed`，v3 receipt/log spent，待 clean v4。
- 2026-08-01：clean `cf79d84f` stress v4 产出可验签 FAIL receipt `d5e0fc4b…`，且 finally
  restore exact。8/8 行 qdes=q0、全 finite；Hctrl ON 四行在首个 5 ms 后均严格位于控制
  包络内，OFF 四行均进入 `[Hctrl,Hmech)`，最小机械余量仍为 `3.57 mm`。FAIL 仅因旧
  20-ms ballistic proxy 把“上一 readback 尝试且当前已回内、速度非外向”记作 capture：
  waist-pitch ON 两侧首步仍余 `-0.0348/+0.0456 rad/s` 外向速度，尽管已分别消掉约
  `98.6%/98.2%`。下一门改为完整 4×5-ms policy horizon 逐子步验证 ON 不越 Hctrl、全部
  不越 Hmech、qdes/restore exact；旧 proxy 原样保留作遥测，不以放宽安全边界造 PASS。
- 2026-08-01：4×5-ms full-policy-horizon stress v2 source 已实现：8 个 env 各记录四个
  q/qdot/qdes 子步，ON 每步 strict Hctrl、ON/OFF 每步 strict Hmech、OFF 首步进入两包络间、
  qdes=q0/finite/restore exact；旧 20-ms ballistic/capture proxy 仍按首步原点采样但仅作 telemetry。
  schema/kind/confirm token 均版本化，host focused `24 passed`、`py_compile` 与 `diff --check`
  通过；待独立 review 后提交 clean source 并在 Pod2 生成 no-clobber v2 receipt。
- 2026-08-01：clean `861a7842` 的首个 v2 Pod 尝试 v5 在 `gym.make` 前因登记的 ignored
  preconverted USD 路径缺失而生成 canonical FAIL receipt `5c6d09de…`；没有 env/restore/机械
  轨迹，不能作 cage 结论。已按 setup runbook 从团队保留副本恢复完整 6-file bundle，核对
  21,897,893 bytes 与 model/base/physics/sensor 四层 SHA。另确认 `kit_boot_lock.sh` 只异步释放
  boot lock，wrapper rc 不代表 detached child；后续以 child completion + receipt status 验收。
- 2026-08-01：同一 clean source 的 Pod2 v6 已完成完整 4×5-ms 差分轨迹并 exact restore/qdes。
  ON 最大 Hctrl solver penetration 仅 `6.06e-5 rad`，但最小 Hmech gap 仍 `0.01392266 rad`；
  关闭唯一 live Hctrl 后，相同 q0/qdot/qdes 的四组 OFF 全在 tick2 穿过 Hmech，最大 penetration
  `3.27e-4 rad`。因此 v6 的 schema-v2 FAIL 是 validator 把“strict Hctrl”错当约束有效性的
  必要条件，并同时错误要求 positive-control OFF 不越 Hmech，不是 plant 负例。下一 source
  版本化差分 v3：ON 四 tick 对 Hmech 零容忍，Hctrl penetration 只能小于 cage reserve；OFF
  首 tick 进入两包络间且后续至少一 tick 触/穿 Hmech；两组同带、finite、qdes/restore exact。
  v6 canonical/file/log SHA=`da977d6…/eb93a9f…/8eb17b8c…`，namespace spent。
- 2026-08-01：差分 stress v3 已在 clean `dff36ad4` 收口。schema/kind/confirm token 升版；ON
  对 Hmech 零容忍且 Hctrl penetration 必须小于 reserve，OFF 首 tick 进入两包络间并在四 tick
  内触/穿 Hmech。独立红队抓到并修掉“输入 tensor 冒充 live 初态”：最终 q0/qdot 直接来自
  PhysX DOF position/velocity getter，再与 tape 和 ON/OFF exact 对账。旧 20-ms proxy 数值完全
  退出 verdict，只原样记 telemetry。专项 `28 passed`、两轮 review 无 P0/P1；下一步用 Pod2
  GPU0 产 fresh v7 canonical receipt，PASS 前不进入 recipe/pin。
- 2026-08-01：clean `956a7a3a` 的 Pod2 GPU0 v7 已 natural rc=0，canonical PASS
  `06da2c91…`；receipt/log SHA=`1dd6ef2f…/49f8c3f7…`。live PhysX q0/qdot、逐 tick qdes、
  ON/OFF same-tape、restore exact；ON 最大 Hctrl penetration `6.0558e-5 rad` 且最小 Hmech
  gap `0.01392266 rad`，OFF 四组共 10 tick 触/穿 Hmech、最大 penetration `3.27498e-4 rad`。
  双包络机械差分门正式 PASS，但该 receipt 仍 training/deployment/hardware unauthorized。
  下一项已先登记 EXP：正式三个 exec 入口把 `LD_LIBRARY_PATH` 重建为仅 GLU，遗漏 v7 真启动所需
  private OpenGL；先把 runtime-asset claim 升 v2、exact pin `OpenGL:GLU`，再物化三个 final pins。
- 2026-08-01：RUNTIME-ASSET-LOADER-V2 已在 clean `42277708` 实现：base/identity/dynamic 三个
  exec 入口共享 claim-owned helper，nested v2 exact pin private OpenGL/GLU 的固定路径、bytes SHA、
  direct SONAME、USD closure 与无 ambient tail 的 `OpenGL:GLU`。missing/reverse/tail/tamper/v1
  全 fail-closed；四组 focused `170 passed`、独立红队最终无 P0/P1。TOCTOU 口径诚实限定为
  exec 前 pathname SHA 重验且 launch window 无并发本地写者，不宣称恶意写者下 immutable。
  C0 三个 zero-PPO 输出与 C1 smoke 必须复用同一绝对 checkout；切 C1 前要等 exact PGID 与 GPU
  lock 释放，不能把 boot marker 当 natural completion。
- 2026-08-01：loader-v2 Pod live 门已关闭：GLU 的 `libOpenGL.so.0` 解析到 exact private
  OpenGL，missing/reverse/ambient-tail 三个 plan 均在 namespace 前 rc2。clean `eef4d61e` 的首个
  correct loop policy plan 随后在 Kit/GPU/namespace 前继续 fail-closed：tracked dynamic-ready
  candidate 内部 `sources.stable_motion.path` 是产物生成时的旧 checkout 绝对路径，而 action registry
  是 repo-relative logical path。未启动训练、未创建目标 namespace。EXP 已先登记
  `DYNAMIC-READY-PATH-IDENTITY`：保留 action/motion/runtime-contract/artifact SHA 全闭包，先判明
  absolute path 是 runtime identity 还是旧 provenance 表示，再选择固定 root 重物化或安全的
  repo-relative+tracked-SHA 迁移；不得直接删除 path 检查。
- 2026-08-01：dynamic-ready path identity 采用可移植 logical-source 语义并完成 source：absolute
  provenance 只接受完整 registry relative component suffix，relative 只接受 exact registry path；
  dot/dotdot、重复/尾斜杠、双根、控制字符、relative prefix 与 same-basename/wrong-dir 均拒绝。
  action/frame0/motion/runtime-contract SHA 仍先验，current commit blob + worktree motion SHA 再双验；
  training runtime 的 candidate/hold canonical absolute path + file/content SHA binding 未改。整合
  `139 passed`、pycompile/diff-check PASS，独立红队 P0/P1=0，真实 tracked loop/block r2 正例通过。
  下一步提交 clean C0，在固定 Pod checkout 重做 fresh plans；不复用旧空 plan/spec/namespace。
- 2026-08-01：clean `0670ad1f` 的 loop/block zero-PPO recipe 已在 Pod2 自然完成，分别得到
  policy SHA `ddcc1a7c…a09f` / `73d9de68…1e51`，0 PPO、0 checkpoint 且全 unauthorized。
  adaptive-sigma r1 随后正确生成 31 项非零 effective-Reward receipt `6520f153…`，但 wrapper
  错把 weight=0 的潜在 `racket_strike_success` 当成有效图必需项而 false-negative。修复保持 v2
  success 权重为 0，显式校验四个非零核宽 `0.20/0.30/1.0/0.52` 并拒绝 receipt 意外激活
  success；潜在三宽锁步继续由 train receipt-before-write gate 与 runtime 原子 scheduler 保证。
  host focused `121 passed`、独立审计 P0/P1=0；下一步在新 clean C0、fresh namespaces 同源重做
  loop/block/adaptive 三 pin，旧 r1 永久 spent。
- 2026-08-01：修复后的 clean C0 `7587124d` 已在固定 Pod checkout 产出同源三 pin：loop/block
  policy `ddcc1a7c…a09f` / `73d9de68…1e51`，adaptive effective Reward
  `6520f153…63db`（31 active terms，file SHA `fbf1c09c…2960`）。三个 child/PGID 均自然退出、
  GPU0/2 lock 释放，结果均 0 PPO/0 checkpoint/全 unauthorized，source clean；Pod 依赖相关组
  `403 passed`，第二轮独立 review 仍 P0/P1=0。三 SHA 已原子写入 code-owned vendor launcher；
  下一步提交窄 C1 并用该 exact checkout 运行三 lane `1×2` smoke。

## 2026-07-31（N1 共享安全门）

- 三 lane smoke 与 `4096×5` probe 均自然完成，但 table 率为 `0.767%–1.110%`，且 block
  有 3 次右 ankle-roll raw-hard；既定 `0.5%` table Gate 不变，三条 long 继续 blocked。
  下一候选把 2% PhysX 控制位置包络扩到两腰+双 ankle roll，以 schema-3 显式绑定并改用
  16-env v5 全系统机械 stress；同时加入 default-off exact
  [OBB](DEFINITIONS.md#obb) [SAT](DEFINITIONS.md#sat-collision-test) 桌体诊断，不改 terminal/Reward/Gate。
  可复现口径见 [G05](gates/G05_isaac_training_first_loop.md) 与
  [三卡发射工序](operations/run_ablation_wave_launch.md)。
- 两个候选的独立复核均已收口为 P0/P1=`0`。四轴 v5 stress 已把初态
  31-D q/qdot/qdes、每 tick 31-D qdes、origin-relative root、隔离外部刚体与
  finally restore 全部纳入成对身份及 validator 重算 digest；`53` probe tests 与
  `141` scoped tests 通过。table 候选的 first/category/phase/cell 与 raw table terminal
  双重守恒已进 PPO 边界和 gate materializer。当前唯一共享前置是在 reviewed
  exact commit 上跑 clean Pod live v5；旧两腰 v7 receipt 不为新增双踝代签。
- clean `9819a862…` 在 Pod 的依赖完整测试新增 `259+313+196` PASS。live v5
  首次在 Kit 前正确拒绝短 SHA；使用完整 SHA 后在 vendor profile bind 抓到
  `robot_hit_table` task-first exact-key consumer 未吸收新的 `attribution_diagnostic` /
  `attribution_command_name`。spent FAIL=`ea42ce39…` 保留。最小修复只同步 strict
  key set 并验证 bool/command/ActionBall 语义，不改 diagnostic default、terminal、Reward
  或 Gate；独立复核确认无第二个 full-param strict consumer，并提升 env cfg 为 exact bool。
  host 行为负例 `10 passed`、runtime-contract `13 passed`、launcher/override 组合
  `336 passed`；下一步提交 exact 修复并重跑 Pod live v5。
- clean `8d0b8ba0…` 的 Pod 组合 `349 passed`；v5c 已完成 scene/16 env/四轴
  Hctrl live 安装与 readback，在 reset 前被旧 action manifest solver SHA 拒绝：
  manifest `af4f6f95…` vs runtime `f89587db…`，spent FAIL=`a0f8d352…`。下一顺序修正为
  formal profile pinner → 新 action profile/manifest 身份 → live v5 → 其余 pins；不绕过
  solver gate，旧 profile/manifest 不复用。
- formal pinner 已在 clean `8d0b8ba0…` 完成：physics SHA=`aa5c9085…` 不变，
  solver SHA=`f89587db…`，tracked profile SHA=`509f3812…`。现打开无覆盖
  `r4` artifact epoch，保留 stable motion/source manifest，下游 identity/authority/bundle
  pins 全部先置空 fail-closed；下一步是 P1 定向测试后并行物化 loop/block identity。
- P1=`6a7587c0…` 已 push；loop/block r4 identity 在两个独立 clean worktree 并行
  物化，两边 materializer 各 `19 passed`。manifest=`e7531567…/7870a053…`，
  receipt=`3c79f266…/9ffe90cc…`；当前回填这一层 registry，下游 pins 仍为空。
- clean P2=`e7917b14…` 的 Pod v5d 已运行：spent FAIL content=`2d10999c…`，
  restore exact。腰部 OFF/ON 轨迹表面满足行为，但 ON/OFF root x/y 不同，
  因而不是因果 PASS；4 个 ankle OFF/ON 又都被 ground/contact 位置投影弹回
  内侧。下一步统一 16 env 空中 root/零 6D 速度、证明零外部 contact，并把
  full-input pair parity 移至动力学 verdict 之前；不改 2% Hctrl 或 verdict。
- v6 stress source candidate 已实现 exact airborne root=`[0,0,3 m]`/零 6-D 速度、
  每 tick 外部 contact force `<=1e-6 N`，并把全部 pair-input parity 前置到任何
  outcome verdict 之前；schema/kind/confirm token 升为 v6，host 正负测 `66 passed`，含同步
  q_des 漂移与更早动力学失败并存时仍须先报 input 污染的复合反例；与 r4 identity
  smoke 合跑 `97 passed`，两轮独立只读终审均为 P0/P1=`0`。
- clean `ff41b12c…` Pod torch 组合 `361 passed`；v6a 产出 spent FAIL，canonical/file/log
  SHA=`c9c56bde…/52ca07fa…/e6d4adf0…`，restore exact、64/64 contact=`0 N`、8/8
  input pair exact、80 seals 可复算。首因仅是 PhysX identity quaternion 的 `7.9e-11 rad`
  规范化被字面等值误杀；四个 ankle OFF tick4 另只差 `0.000308–0.000318 rad`。v7
  保持 raw pair exact，以双覆盖物理角 `<=1e-9 rad` 验声明 identity，并只把 ankle outer
  stress `0.60→0.65R`；腰/qdes/horizon/Hctrl/contact/verdict 不变。tape 固定公式逐行重算，
  quaternion norm=`1e-12` 且 raw pair 仍 exact；host `71 passed`、identity+probe `102 passed`。
- v7 receipt-integrity 独立红队三轮暴露的 joint/index、runtime/tape、literal-version、
  live-limit identity 与 live-readback attestation 五个 P1 已全部关闭；exact keyset/
  31-joint order/receipt-time revalidation/public selected names+indices/四个 live proof 和所有
  readback/setter/mixed/order SHA 均 fail-closed。最新 host focused=`82 passed`、
  identity+probe=`113 passed`，`py_compile`/`git diff --check` PASS，独立终审 P0/P1=`0`。
  下一步是 clean commit/push 后在 Pod 用 exact checkout 跑六文件 torch 门和 no-clobber v7a；
  机械 stress 仍待 live receipt，未写成 PASS。
- v7 receipt-integrity 实现、负例、EXP/G05/运行工序已合入并 push
  `04b50343a1455914c79bcbf6f8080551864ab289`。本次纯进度账本 successor 推送后，
  Pod 只允许在该 exact clean checkout 上跑六文件 torch 门与 fresh v7a；不修改/覆盖
  v5d/v6a spent evidence。
- exact clean `62e0878ac52748373838850faf02c3be1c9f16bc` 的 Pod 六文件 torch
  组合 `377 passed`；16-env no-clobber v7a receipt canonical/file/log=
  `79e14853…/cb0fcfdc…/087028bf…`，status=`PASS`且 canonical 独立复算一致。
  8/8 OFF 组 tick1 进 `[Hctrl,Hmech)` 并触/穿 Hmech，8/8 ON 组 32 ticks
  strict Hmech；64/64 contact=`0 N`，8/8 full input tape exact，restore/four live proofs/
  public-run readback exact。四轴 plant 机械门关闭；下一步改为并行重物化 loop/block
  工件链与 A/B/C pins，再跑 table-attributed fresh `4096×5`。

- r5 identity recipe 在当前 composed Reward SHA 上 fail-closed 后，`f9b80cd9…` 已同步 code-owned
  Reward pin。Pod 三文件定向组为 `354 passed, 3 failed in 4.03 s`：一条测试漏列
  `force_push`，另两条是 contact bundle 尚为 `None` 时的预期 fail-closed。下一窄提交只补
  `force_push=false` 测试断言并在 Pod 重跑 identity/reward 两文件；不伪造 bundle 或放宽 launcher。

- exact `223a5b4a…` 已在 Pod 通过 identity/reward 定向组：`280 passed in 3.99 s`。下一步使用
  fresh `a3vendor-identity-recipe-r5-loop-223a5b4a-r2` 运行 loop recipe，待 policy-contract SHA
  自然产出后再运行 smoke；旧失败 namespace 永不复用。
