# EXP-P1-BALANCE-ACTION-SLEW-20260720 — 腿腰恢复期执行目标突变是否比全身 raw-action 平滑更适合乒乓

- 状态：`superseded`（2026-07-20：fresh v9/probe10 科学长训位被 [24 格平衡×时序矩阵](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md) 的 {N,C,H}×S0 六格取代，永久不再单独发射；probe9/v8 六份 receipt（set SHA `cc9ff591…d9c8`）作为 runtime mechanics 证据被该矩阵引用。本记录保留 C/N/H 机制定义与全部 probe 历史）
- 运行态：[`W/V × C/N/H`](../../DEFINITIONS.md#balance-action-slew-matrix) 六格中，`probe2 W-C false reject；probe3 W-C/W-N/V-C verifier contract rejected；probe4 W-C Hydra compose rejected；probe5 five receipts passed / W-H pre-trainer identity reject；probe6 W-C transaction-wrapper pre-trainer reject；probe7 W-C/V-C/V-N receipts passed / W-N scene-boot watchdog teardown / W-H,V-H not launched；probe8 W-N sim.reset SIGABRT / other five not launched；probe9 6/6 completed；v8 scientific W-N sim.reset stale / other five not launched；fresh v9/probe10 preregistered / not launched`
- 阶段/轴：Phase 1 / 单拍后的平衡恢复与动作平滑
- 集成小目标：降低高回台候选的摔倒与腿腰突变，同时不压低稳定候选的击球完成和回台
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：Wave A runtime mechanics=`E3`（probe9 六格 exact 两步探针）；机制结论=`E1`（科学长训未发射）；外部 M0 moving input gate=`E2`
- 创建日期/最后复核日期：2026-07-20 / 2026-07-20

本记录使用的 [`raw action-rate`](../../DEFINITIONS.md#raw-action-rate-l2) 是每个 50 Hz tick
连接当前与上一 policy 输出的全 31 维二次差；[`processed-q_des slew hinge`](../../DEFINITIONS.md#processed-qdes-slew-hinge)
是只看执行前已变换、已 clamp 的 15 个腿腰目标的恢复期阈值惩罚。
[`W/V × C/N/H`](../../DEFINITIONS.md#balance-action-slew-matrix) 是本轮六格的完整人话定义。
本实验是[稳定性 Wave A](../../DEFINITIONS.md#balance-stability-waves)，不是唯一平衡方案；下文另预留
Wave B 的下半身老师消融，但不猜尚未审计的 flag。

## 问题与假设

问题：对需要快速挥拍、但击球后容易姿态混乱的乒乓策略，现役只看相邻 raw action 的全身 dense
惩罚是否已经有效；如果需要更强的下肢约束，能否只在同一拍恢复窗、只对腿腰执行目标的极端变化收费，
而不抑制击球臂的快速动作？

可证伪假设：把现役全身 `action_rate_l2=-0.10` 换成恢复期 processed-q_des 腿腰铰链，会在高摔倒
`V` parent 上显著降摔，并在稳定 `W` parent 上保持完成、回台和摔倒非劣；如果只让动作变平滑却过不了
下文的击球与硬摔倒门，或根本没有激活预注册尾部，则拒绝这一 replacement。

### 50 Hz 下“只连上一 action”是否太弱

不是“只产生一次联系”。每一个 tick 都收费，因此相邻边沿串成覆盖整段轨迹的离散平滑先验；对频率为
`f` 的单维正弦 action，其每步均方差分按 `sin^2(pi*f/50)` 增长，所以 50 Hz 下它优先压高频抖动，
而不是慢漂。现有历史对照支持“它并非小到完全没有作用”，但不是 formal 因果证明。它有三个明确边界：

1. 它只记一个相邻样本，没有更长时域状态，也不是二阶差分或恢复策略。
2. 它对 31 维 raw policy action 全程一视同仁，可能同时压住需要快速变化的击球臂。
3. 它量的是 affine transform 和 q_des clamp 之前的 action；真正送给比例微分控制器的执行目标可能不同。

因此 Wave A 不直接把全身 dense 权重做得更大。“腿部可以大一些”须拆成两种不同意思：腿腰的异常突变
可以使用更强、更有针对性的惩罚；合法迈步的每帧免罚变化量也应随各关节速度许可变大。这里同时采用
相位化、阈值化、按实际关节速度许可归一化的 processed-q_des 项；击球前、触球窗、手臂和下一拍揭题
都不收费。

## 已有因果线索（不是本轮结果）

[半秒冲刺记录](EXP-P1-HALF-SECOND-SPRINT.md#2026-07-20-action-rate-证据回收)保存了 runtime 配置差分只剩
action-rate weight、同 `qdot=0` 的 `-0.05` 与 `0` 两格完整 5701–6700 指标、119 MB 证据副本和
checkpoint SHA；历史 hard contract 未绑定该轴，所以只作方向性诊断。
关闭惩罚后 raw action delta、最大关节速度和 base pitch 都明显增大，显示这两条历史轨迹中的相邻项
与高频平滑有关；但
completion、return 与 fall 有交叉取舍，不能据此把全局 `-0.05` 或更强全身权重宣布为平衡答案。

## BeyondMimic 给出的边界

[BeyondMimic 项目](https://beyondmimic.github.io/)、[论文 v4](https://arxiv.org/abs/2508.08241v4)和
[官方代码的冻结 reward 配置](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L198-L278)
采用的是组合平衡学习：全身 tracking、`action_rate_l2=-0.1`、关节限位和接触约束、低阻抗比例微分
控制与 action scale/armature、失败终止、随机初态、1–3 秒外扰，以及按失败区间自适应采样。其直接
消融最支持的是自适应采样，不是“action-rate 单项足以解决平衡”。

所以本实验只隔离 Wave A 的 action-slew 机制，不声称复现 BeyondMimic。下半身老师与失败采样等机制
必须另开 matched ablation，不能把整套上游组合的效果归给本轮一个 Reward。

## Wave A：冻结的 setting

机器真源是
[`phase1_balance_action_slew_20260720.yaml`](../../../configs/phase1_balance_action_slew_20260720.yaml)；
[`run_phase1_balance_action_slew_queue.py`](../../../scripts/run_phase1_balance_action_slew_queue.py)默认只验证并
打印计划，不 SSH、不写远端、不发 signal。远端训练 source 必须是 clean、detached 的 exact commit
`54c9a62656f0e60e5bb41cbcfa0e5a972b793906`；不能以启动时的任意 HEAD 代替。
即使 source 已冻结，命令生成仍被独立复核的 [`launch manifest`](../../DEFINITIONS.md#balance-launch-manifest)
硬门阻断；它必须绑定 source、queue 与全部远端输入的 SHA-256，其中 preconverted `model.usd` 还要连同
依赖的完整 6-file sibling bundle 做 tree hash。probe9/v8 清单现只作历史；当前 fresh
probe10/v9 预注册清单是
[`phase1_balance_action_slew_launch_manifest_20260720.json`](../../../configs/phase1_balance_action_slew_launch_manifest_20260720.json)，
文件 SHA-256=`664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a`、content
SHA-256=`36ceb3c77dc056f4565378a92b03da58865378d86c5849085ba066631cea456c`。它只把 probe command
从“缺清单”解锁为“可渲染”，不表示已经 SSH 或启动。train command 仍必须消费六份
[`probe receipt`](../../DEFINITIONS.md#balance-probe-receipt-set)。

| 字段 | 冻结值/SHA |
| --- | --- |
| source checkout | `/workspace/codexschema/nohope_balance_action_slew_20260720`；clean detached `54c9a62656f0e60e5bb41cbcfa0e5a972b793906` |
| queue bytes | config SHA-256 `3bf5085ea8396513d162b9cce249dfb761b39b2827ec722959343c953683e59e`；runner SHA-256 `0fff4515cbe7e62798e8c39f701851c46e68287c7321e7618161fa9dde4789ce` |
| 动作 | `v4rg_runtime_order_v3` 正/反手，路径固定在 YAML |
| 观测/action | `deploy_parity_face179`，actor `179→31`；`qdes_clamp=true` |
| plant/engine | Isaac `HOPEPingPongVirtualBall`，`dt=0.005`、decimation `4`，即 50 Hz；零关节摩擦 |
| parent/seed | `W@model_6700` 与 `V@model_6700`；六格均 `seed=3`，完整恢复 policy/value/optimizer/normalizer |
| 课程/题库 | short-focus `0.10/0.45/0.40/0.05`；同一 schema-3 train bank |
| 其他 Reward | 每个 parent 内保持原击球/准备配方；qdot-limit hinge=`0`，post-swing replay=`0` |
| 谱系 | [`checkpoint_allow_contract_mismatch=true`](../../DEFINITIONS.md#checkpoint-contract-mismatch)；所有后代只作诊断、永久 formal-exact-ineligible |

### processed-q_des 腿腰铰链

对 exact 3 个腰关节和 12 个腿关节逐关节计算：

```text
u_j    = abs(q_des[t,j] - q_des[t-1,j]) / (qdot_limit[j] * 0.02)
tail_j = 1 - exp(-(relu(u_j - 0.85) / 0.15)^2)
value  = mean(tail_j over the exact 15 joints)
```

`q_des` 已经过 action affine transform 与 train=deploy joint clamp。只有同一拍触球后
`0.20 <= age <= 1.55 s`、且 previous q_des 有效时才返回 `value`；reset 后第一步强制为零，不能把 episode
边界伪造成大突变。`tail_j` 位于 `[0,1)`，weight=`-0.25`；RewardManager 再乘 `0.02 s`，因此满激活时
每 tick 的惩罚幅值小于 `0.005`、每连续 eligible 秒小于 `0.25`。50 Hz 漂移、关节重排、腿腰集合不完整
或非有限/非正速度上限都 fail closed。这里既把惩罚集中到腿腰，又没有给所有腿关节同一个绝对阈值：
速度上限更大的关节会自然得到更大的每帧免罚变化量，合法快速迈步不会仅因绝对角度变化较大就被收费。
按当前 pinned A3 nominal limits，`0.85` margin 对执行目标的免罚变化量约为 hip `0.204`、knee `0.248`、
ankle-pitch `0.184`、ankle-roll `0.328 rad/tick`；runtime 仍以实际 articulation limit 为准。这已经是较宽的
腿部 allowance，先看 tail activation、fall 与 tilt，再决定是否另开 `1.25×/1.5×` allowance 消融，不能
仅凭“腿应该更快”修改本轮冻结值。

### 六格矩阵与唯一 run names

`C`、`N`、`H` 三格都显式写入 raw/processed 权重与 processed margin/window，避免“默认值”冒充单变量。
下面每行先给人话，再给唯一 `run_name`：

| 运行（人话 + `run_name`） | parent | raw action-rate | processed-q_des | 固定槽 |
| --- | --- | ---: | ---: | --- |
| W 当前 dense 对照 — `phase1_balance_slew_science_retry2_w_c_dense_m0p10_seed3_20260720` | W 稳定 parent | `-0.10` | `0` | Pod1 GPU1 |
| W 无平滑负控 — `phase1_balance_slew_science_retry2_w_n_none_seed3_20260720` | W 稳定 parent | `0` | `0` | Pod1 GPU0 |
| W 恢复期腿腰 replacement — `phase1_balance_slew_science_retry2_w_h_processed_qdes_seed3_20260720` | W 稳定 parent | `0` | `-0.25` | Pod1 GPU2 |
| V 当前 dense 对照 — `phase1_balance_slew_science_retry2_v_c_dense_m0p10_seed3_20260720` | V 高摔倒 parent | `-0.10` | `0` | Pod2 GPU0 |
| V 无平滑负控 — `phase1_balance_slew_science_retry2_v_n_none_seed3_20260720` | V 高摔倒 parent | `0` | `0` | Pod2 GPU1 |
| V 恢复期腿腰 replacement — `phase1_balance_slew_science_retry2_v_h_processed_qdes_seed3_20260720` | V 高摔倒 parent | `0` | `-0.25` | Pod2 GPU2 |

`W` 与 `V` 的不同 parent 配方不构成 W-vs-V 单变量比较；机制只在各自 parent 内比较 `C/N/H`。

### 2026-07-20 W-C probe2 假拒绝与 probe3 修订

Pod1 GPU0 的 `phase1_balance_slew_probe2_w_c_seed3_20260720` 是唯一实际启动的旧探针。它从 W
`model_6700.pt` 自然跑完两个 update，`2026-07-20T01:08:47.435447Z` 开始、
`2026-07-20T01:09:24.372852Z` 退出，terminal status 为 `exit_code=0`、`normal_exit=true`，并生成
`model_6700.pt`、`model_6701.pt`、hard contract 和完整 TensorBoard ledger。进程组与 GPU 已自然释放；
没有人工 signal。

旧 outer verifier 没有发布 receipt，因为它错误要求**每个** update 都有非零恢复期分母。第 6700 步
包含 `24 × 4096 = 98304` 个样本，其中 previous-q-des valid/invalid=`89899/8405`，但
recovery-eligible=`0`；初始准备时间最短 `0.36 s`，再加触球后 `0.20 s` 的窗口起点，最早资格时刻
`0.56 s` 已超出首个 `24 × 0.02 = 0.48 s` rollout。第 6701 步已有 recovery-eligible=`31459`、
tail-active=`31456`、above-margin joints=`189191`、gated-tail sum=`11670.224609375`，证明通道实际激活。
故这次拒绝是量尺错误，不是 trainer 崩溃，也不能写成 W-C 机制失败。

probe3 的修订仍逐 update 强制 observed=`valid+invalid`、tail/joint/value 守恒；它只允许某一 update 的
recovery-eligible 为零，并新增**两步合计必须大于零**的硬门。为保持 no-clobber，全部六格转到新根
`/workspace/codexschema/phase1_balance_action_slew_v2_20260720`，旧 probe2 目录原样保留且不得补写 receipt。
probe3 的 config、runner、manifest 进入 `origin/main` 前，不允许重发任何一格；旧 manifest 也不再提供
发射权限。

### 2026-07-20 probe3 完整-rollout 漏洞与 probe4 修订

probe3 在新 v2 根实际自然跑完 W-C、W-N、V-C 三格，均到 `model_6701.pt`、exit `0`，GPU/进程组自然
释放；W-C 还被当时 verifier 发布了 receipt（file SHA=`4914c558…fceca`）。独立 red-team 随即发现，
remote/local verifier 只要求 observed 能被 `4096` 整除，测试 fixture 甚至把 `4096` 当作一整个 update；
它没有证明冻结 PPO 的 `24 steps/env` 全部落账。因此只含 1/24 rollout 的伪造 receipt 理论上也能通过。
W-C 的真实两行恰好都是 `98304`、合计 `196608`，所以 runtime 数据可保留作 mechanics evidence；但旧
verifier identity 不够严格，W-C receipt 不能解锁 train。W-N/V-C 在告警到达时已经启动，允许其自然退出，
没有运行旧 verifier、没有发布 receipt；其余三格没有启动，没有发送任何 signal。

probe4 意图把 24-step rollout 写进命令和 claim，在 verifier spec/receipt 中绑定
`expected_samples_per_update=4096×24=98304`，并要求每个 update 的 processed-q-des observed 与 qdot
observed 都**精确等于** `98304`；两步 totals 因而必须为 `196608`。测试新增 `4096`、`98304±4096`
及 forged expected-denominator 攻击。全部六格再次转到全新 no-clobber 根
`/workspace/codexschema/phase1_balance_action_slew_v3_20260720`，run name 使用 `probe4`；v2 的三格与 receipt
保持历史只读，不能补写、替换或复用。

### 2026-07-20 probe4 Hydra 组合门拒绝与 probe5 修订

probe4 W-C 的唯一次 SSH 尝试在远端 manifest/source/checkpoint 预检通过后，被真实
Hydra `--cfg job --resolve` 组合门拒绝：命令错写了不存在的 `algo.num_steps_per_env=24`。
真实消费路径是 [`algo.runner.num_steps_per_env=24`](../../DEFINITIONS.md#ppo-num-steps-per-env)；Hydra 建议的
`+algo.num_steps_per_env=24` 只会添加训练不读的死字段，不可采用。该 compose guard 位于建 run
directory 和调用 Kit launcher 之前，所以 v3 root、claim、log、checkpoint、TensorBoard、PID/PGID
全部不存在；GPU0 为 `0 MiB / 0% / no compute app`，boot lock 仍是 probe3 时间且无 holder。
这是 fail-closed 基础设施拒绝，不是 trainer natural exit 或科学负结果。

probe5 改用真实 runner key，增加“配置层级必须存在”单测和 Hydra 可用时的真实 compose 测试，
并转到新 no-clobber 根 `/workspace/codexschema/phase1_balance_action_slew_v4_20260720`。probe run name
使用 `probe5`；probe4 命令、manifest 和 v3 根只作失败历史，不可再发。Pod1 上用 probe5
W-C 的**完整 exact argv**只运行 `--cfg job --resolve`，已得到 exit `0`：解析后
`algo.runner.num_steps_per_env=24`、`cfg.algo` 顶层无死字段、`max_iterations=2`、run name 为
`phase1_balance_slew_probe5_w_c_seed3_20260720`。这是零训练 compose 证据，不是 probe 已启动。

### 2026-07-20 probe5 五收据与 W-H exec 身份竞态；probe6 修订

probe5 在双 Pod 完成了 W-C/W-N/V-C/V-N/V-H：五格均 natural exit=`0`、normal exit=`true`，
dedicated verifier 均通过，receipt content SHA 依次为 `8cd90351…8f62c`、`3480cc2d…148a2`、
`c9a18fcf…58dd24`、`fb0c26df…075cd5`、`3ba6f291…3f91ba`。W-H 则在首个 training iteration 之前被
supervisor 拒绝：旧实现在 `Popen(argv)` 后只读一次 `/proc/<child>/cmdline`，把 fork→exec 瞬间的
暂时 argv 当成身份不一致。现场只有 50 B 错误日志与 launcher evidence；binding、terminal status、
RSL run-dir、checkpoint 和 receipt 全部不存在。leader PID=PGID `2708248`已自然消失，
双扫该 PGID members=`0`、GPU2=`0 MiB / 0% / no compute`、Kit lock 无 holder；未发任何 signal。
这是启动身份采样竞态，不是 H 机制训练失败。同一 manifest 必须六收据齐全，因此五份旧收据
不能解锁 train，也不能混入新 identity。

probe6 在 `Popen` 后、首次 `/proc` 读取前先不可覆盖地保存 child PID；随后对同一 child 最多等待
`5 s`、每 `10 ms` 复核。每次身份读取都要求两份 `/proc/stat` 的
PID/PGID/starttime 完全一致并与 `getpgid` 相符，首个可读 starttime 后也必须始终不变；只有与
supervisor 同 PGID 的 exact argv 出现才接受。early exit、PGID/starttime 漂移或
timeout 会不可覆盖地写最后身份证据；所有身份读取异常也必须走该 failure path。外层把 launcher 与
state/marker/binding/fatal postcheck 包进同一 transaction，任一失败后都两次稳定扫描 exact PGID 和 child
PID，且用 `lstat`/`O_NOFOLLOW` 拒绝 dangling symlink 或非普通失败证据；只有都为空才返回原错误，
否则 rc=`121` 要求人工身份审计。身份等待与外层审计本身都不发 signal；既有 locked launcher 的
boot watchdog 仍只能按其原合同向已绑定 exact PGID 发信号。任何失败后
仍须确认 exact PGID/child/GPU/locks 全空，否则停止全批。行为测试覆盖了 exec 过渡、瞬时 `/proc`
absent、child exit、PGID 漂移、starttime 变化和 timeout。新 root 为
`/workspace/codexschema/phase1_balance_action_slew_v5_20260720`，run name 使用 `probe6`。

### 2026-07-20 probe6 transaction-wrapper 拒绝与 probe7 修订

probe6/v5 只发了 Pod1 GPU0 的 W-C。它在 `2026-07-20T03:01:10Z` 于 trainer/probe supervisor 内、进入
trainer 前 fail closed；81 B `run.log` 只报告 `IndentationError: unexpected indent`。根因不是 Hydra、
模型或 Reward，而是 `_failure_audited_transaction_shell` 为 transaction 的每一行加两个空格，改变了
已经由 `shlex.quote` 保护的 multiline Python 参数字节。locked launcher 无法绑定已快速退出的 child，
也没有发 signal。leader evidence、child evidence、binding、terminal status、checkpoint、receipt 与
RSL run-dir 均不存在；PID=PGID `2712318` 已双扫稳定 absent，GPU0=`0 MiB / 0% / no compute`，
Kit/cache locks free。整批立即停止，W-N/W-H/V-C/V-N/V-H 均未发。故 v5 永久只作基础设施失败历史，
不能写成 C 或任一 action-slew 机制的负结果，也不得在该 namespace 补发。

probe7 把 transaction body 逐字节原样拼接，不再对 body 做行级缩进；新增回归直接比较 multiline shell
payload 在 wrapper 前后的字节一致性，防止再次破坏嵌入式 Python。child exact-identity wait、不可覆盖
evidence 与失败后 PGID/child 残留审计保持不变。新 no-clobber root 为
`/workspace/codexschema/phase1_balance_action_slew_v6_20260720`，六个 probe run name 为
`phase1_balance_slew_probe7_{w_c,w_n,w_h,v_c,v_n,v_h}_seed3_20260720`。config/runner SHA-256 分别为
`912bd8d212791d99ce6a6851a8f05c12d182cdfa9d5566e02381f1b4703b8f3c` /
`3fbaf23f97fdb40e05a448f9f769267b21c7cca3bd767aa082c0d5b965ecd7d7`；manifest file/content SHA-256
分别为 `4552fe23abd551d8959a9de05cc5f9d761d0da25eed88138d61fa45cc6558e9e` /
`6e3518d97d48fad550e7971a5178b1f11c15895696f03d30d5a62d1e27741640`。该 identity 冻结时 probe7 尚未
发射；下一节记录其后续 runtime，不能倒写或补用 v6 收据。

### 2026-07-20 probe7 三收据、probe8 W-N scene-reset 失败与 probe9 crossover 替换批

probe7/v6 的 W-C、V-C、V-N 均 natural exit=`0`、normal exit=`true`，dedicated verifier 证明 6700/6701
两步的 processed-q_des/qdot observed 各为 `98304`，并在退出后闭合 exact process group 与 assigned GPU。
三份 receipt 的 file/content SHA-256 为：

- W-C：`9e36cdb52691383c40c9659a1c9120328a3bd9297bd3e69601301f6a97504438` /
  `38b360ddff1811107ba3c13a039081d6b5ead3cbd52cd2513ffd8c127af669fc`；
- V-C：`85320e1ef03b9f6bf9af7c67aadff8f274e1500e47e4bc3a31260b76490de07f` /
  `06c93fea66eb8a8ed19d30b098ce2727253870ca9129a80e796b136beef48c43`；
- V-N：`31a7d1cdc476832190f54f37413cfe6d07d81a7237ee0423ddea4e1724c61dbd` /
  `74797a461f3dc1fe780fc702319952e68423ab2963337e0793ab13b6a3dbb527`。

W-N 于 `2026-07-20T03:22:50Z` 发射，已进入 RSL/scene config，却冻结在 `Starting the simulation`；没有
Learning iteration、binding、terminal status 或 receipt。`run.log` 为 22601 B、SHA-256
`9c46896bbc2a12324a209635374556ab8a4100cd96acd5e9ad01595cbfaa0e3b`，最后 mtime
`03:23:06.408757Z`；180 s locked watchdog 只对已经绑定的 exact process group 先 TERM 再 KILL，最终
rc=`125`。launch ledger SHA-256=`7b087a6f54c78ab83d070bb96aab2152dda10cd294d7dee08df2f3dacc74d1b3`；
事后 exact groups 稳定 absent、GPU1 empty、Kit/cache locks free。V-N 在 W-N 失败被确认前 6 s 已经发出，
因此允许它自然完成并验证；W-H/V-H 从未发射。

同一 v6 identity 下 W-C 已证明 W parent 可完成，V-N 已证明 `action_rate=0` 可完成；结合 exact argv 对比，
W parent 与无 action-rate 都不是 W-N 冻结的必要原因。故 W-N 只记 infrastructure-only transient，不是 N
机制负例，也不修改 Reward 结论或 180 s timeout。v6 namespace/manifest 永久 immutable：三份收据不能与
W-N 重试或下一代混成六份，v6 任何格都不得补发。

probe8 使用 fresh no-clobber root `/workspace/codexschema/phase1_balance_action_slew_v7_20260720`，六个
run name 为 `phase1_balance_slew_probe8_{w_c,w_n,w_h,v_c,v_n,v_h}_seed3_20260720`。config/runner
SHA-256 分别为 `0c84613f05439237f6e36d37e0c9210984465d928b9c0cba50999bd8995145f9` /
`2bc9d59e21413a812a742529c6f3291f5710c384e0f4af7ee7098f33b25ba17d`；manifest file/content SHA-256
分别为 `887c0b9e097e50300d83eef27e587d112f70132958e6e8d9b68af74437fa7231` /
`13f92d5eda71e90abd6a14a1498c2afd98d3cb825cb26e2f5b74958b5a795f84`。它只发射了 Pod1 GPU1 的 W-N：
`2026-07-20T03:44:19.857Z`开始，在 4096 个 environment 已创建后进入 `sim.reset`，随后于
`2026-07-20T03:44:32.112Z` 报 `malloc(): invalid size (unsorted)`。trainer 以 `-6`/`SIGABRT`
退出，outer transaction 返回 `134`；没有 first learning iteration、binding、RSL run directory、checkpoint
或 receipt。其他五格均未发射。manifest/claim/supervisor spec/run log/launch ledger/leader/terminal/child
evidence 的 SHA-256 依次为 `887c0b9e…a7231`、`8472ecf9…fa8b`、`334ed262…c23c`、
`b3437c87…f49c`、`edc4782f…ce8`、`b7f981c8…8815`、`ad5c46a7…6268`、`c2d6c31b…26f`。
事后两个 Pod 的 exact process groups、assigned GPU 与 Kit/cache locks 均已闭合。失败发生在
managers 和 RewardManager 建立之前，只记 infrastructure-only evidence，不是 N 机制或 Reward 负例。
v7 namespace/manifest 永久 immutable，不得重试、补格或与下一代混 receipt。

probe9 是使用 fresh
no-clobber root `/workspace/codexschema/phase1_balance_action_slew_v8_20260720`，六个 run name 为
`phase1_balance_slew_probe9_{w_n,w_c,w_h,v_c,v_n,v_h}_seed3_20260720`。config/runner SHA-256 为
`c7ec75a9917b8bdcf7976186633b021c69fb82e591898dad6b8d5c93cfdb37d5` /
`24b5f7831ad49c2b88266fed65c37e6e4bcdddaacab28ffa917fce66ef918db1`，manifest content/file SHA-256 为
`97c36e471fb8fc6b93fe212f20846de6697db518192e7a45c6618e5924947e28` /
`688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353`。该批于
`2026-07-20T04:14:32Z–04:22:42Z` 严格按 W-N（Pod1 GPU0）→receipt+closure→W-C（Pod1
GPU1）→receipt+closure→W-H→V-C→V-N→V-H 全局串行完成。六格都在下一格启动前
natural exit=`0`、normal=`true`、first iteration=`true`、exact verifier passed，并完成
process/GPU/lock closure。共同 verifier SHA-256 为 `d736a205…0ebc`；六份 receipt 的 file/content
SHA-256 分别为：

- W-N：`ee8c5378…8c5ff` / `afce94d7…80f9`；
- W-C：`b948a4d8…18d5` / `e5daf19c…d3f0`；
- W-H：`a80502c9…8111` / `32abf562…7e68`；
- V-C：`c3db6c38…edc1` / `ae30d1b5…3446`；
- V-N：`06919a60…4bc7` / `bc99acd0…979c`；
- V-H：`b7a24015…1ec2` / `0905ad8f…32a7`。

六份远端 verifier bytes 已复制到本地并通过同一 manifest 的完整重验，receipt-set SHA-256 为
`cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`。crossover 双方通过只排除
“W-N 在 GPU0 必然失败”与“W-C 在 GPU1 必然失败”；它不推翻 probe7/8 的 immutable 失败历史，
不证明 GPU 等价，也不是 action-slew 的科学结果。

六收据 gate 曾解锁 same-swapped 科学长训的**命令生成**；
`/tmp/phase1_balance_slew_train_commands.json` 在 `origin/main=16263be5` 上渲染，SHA-256 为
`fc6f1ea38a5a823016d83675d56fc41b50b70dbde1bba60602b26d6c743802df`。它是从未执行 SSH 的历史
command-generation artifact，authority 过期后没有被发射。

### 2026-07-20 v8 科学长训首格 sim.reset stale、failure-audit 分层缺陷与 fresh probe10/v9

在 `origin/main=d5c08bb91728edfa75801a630531527aeb2ae06c` 上重新渲染并复核的 v8 科学长训 command artifact
是 `/tmp/phase1_balance_slew_train_commands_d5c08bb9.json`，SHA-256=
`be346f94cf6bf738da36804bf59f6a60bc5249f3c6bf5474abf617358db4b42a`。只发射了第一格 W-N（Pod1
GPU0）：`2026-07-20T04:45:02Z` 启动，`04:45:11.250Z` 发布 exact binding，`04:45:15.781Z`
后 `run.log` 不再增长，locked launcher 在 `sim.reset` 前后持续无进展满 `180 s` 后按 exact process-group
合同 TERM/KILL，并于 `04:48:25.971Z` 以 terminal kind=`stale_timeout`、exit=`125` 结束。没有出现
first learning iteration、RSL run directory、checkpoint 或 Reward/机制账；其他五格均未发射。

外层最终 rc=`121` 不是另一项 trainer 结果：旧 post-launch failure audit 对 `train` stage 仍强行读取只有
`probe` supervisor 才会生成的 `trainer_child_evidence.json`。科学长训中 trainer 自身就是 launch-group
leader，现场既没有 trainer-child evidence，也没有 identity-failure evidence；因此 audit 错误覆盖了 locked
launcher 的 `125`。人工复核后 leader PID/PGID、exact group、assigned GPU compute、Kit/cache lock holder
均为零，并在 `04:49:40–04:52:51Z` 得到稳定闭合快照；没有自动 retry 或后续发射。

逐字段真源是
[`phase1_balance_action_slew_train_v8_attempt1_result_20260720.json`](../../../configs/phase1_balance_action_slew_train_v8_attempt1_result_20260720.json)，
文件 SHA-256=`ac09b70a1df89a501165504f4c07158858687127172a8d9d5a6bdf1473e61a75`。其中 launch-spec/leader/
pre-TERM/pre-KILL SHA-256 依次为
`0604c6eab0d48a2f147ab626933f84c6b9655b7db8c9b6873c5ec33b760896ca` /
`a4c6b8fdf66fb7d98e39e024e915835366cb61db73df7e8c3b9d89bf3d31a2c4` /
`1c415a8b53d027cd67cf27ceaada420e0eaf380bafd72dcac9597f036898881d` /
`5c0095f48db3cc2aebc2447d28af4dea8afe2f3e792c3940de5c95136eeecc33`；queue-claim
file/content=`a4ff22dfcd835324e780f24809226cfcfe877c4f9da4b867ab67dff76fb5622f` /
`0a5a56b80ab1b8e43bc94bb09608ce54329dce4240310846fa4dca15afcd4a01`；run-binding
file/content=`e0723f916d6a299d874b3b58b9c03cb5f723b07d95d24300b65a3cfd6613a03c` /
`0cb8fb07f156c5163f30cadfe97b82b77201c17b26040662204418290571d497`；run-log/launch-ledger=
`a2db80ad7f093a9f3eaf5927d972d7f07098482a16aec0c615fb58292baa0ff2` /
`5aca7d4786e9ed158aacc5a78f4c41d2702305117f61d97805d095bf116687c1`。v8 root、run 与 manifest
永久冻结；这只是 infrastructure-only evidence，不是 N 机制、Reward 或 action-slew 的负结果。

fresh v9 把 no-clobber root 改为
`/workspace/codexschema/phase1_balance_action_slew_v9_20260720`，并把 failure audit 显式绑定 `probe/train`
stage：probe 继续强制读取并验证 child evidence，train 则强制该两类 probe-only evidence 不存在，再复核
exact leader group 为空。config/runner SHA-256 分别为
`3bf5085ea8396513d162b9cce249dfb761b39b2827ec722959343c953683e59e` /
`0fff4515cbe7e62798e8c39f701851c46e68287c7321e7618161fa9dde4789ce`；manifest content/file
SHA-256 分别为 `36ceb3c77dc056f4565378a92b03da58865378d86c5849085ba066631cea456c` /
`664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a`。probe10 保持 W-N→Pod1
GPU0、W-C→Pod1 GPU1、W-H→Pod1 GPU2、V-C/V-N/V-H→Pod2 GPU0/1/2 的同一 crossover 顺序；六个
probe run name 为 `phase1_balance_slew_probe10_{w_n,w_c,w_h,v_c,v_n,v_h}_seed3_20260720`。
该 fresh 批目前只完成 source preregistration，尚未发射；必须在同一 v9 identity 下收齐并重验六份
probe10 receipt，才允许生成任何 `science_retry2` 长训命令。不得复用 probe9 receipt 或 v8 科学 binding。

## 预算、量尺与停止规则

1. 六格先各跑 `4096 env × 24 step/env × 2 update` 的完整场景合同探针。每格必须自然退出到独立
   `model_6701.pt`（absolute milestone=`[6701]`，exclusive iteration upper bound=`6702`）；dedicated exact
   verifier 须证明 6700/6701 两步 processed-q_des、completion/fall/legal-return、ready-tilt、qdot tag 的
   分母与守恒账；processed-q_des 恢复资格允许单个 update 为零，但两步合计必须非零，其余预注册行为
   分母仍须逐 update 非零；processed-q-des/qdot observed 必须逐 update 精确等于 `98304`。另须证明 finite
   policy/value/full optimizer/two normalizers、C/N/H exact
   weight-margin-window-applied markers、lineage=`0`、fatal scan，以及 leader/PGID/GPU 已释放并发布不可覆盖
   收据。不能借用 lean fresh-probe 的相对 `[1]` 语义。只有本地按 `DIR/JOB_ID/probe_receipt.json` 收齐并
   重验全部六份，才允许生成科学训练命令；人工写一个“probe 已通过”布尔值无效。
2. 科学 continuation 为每格最多 `+1001 update`，预注册相对 parent 的 `+200/+500/+1000`
   （absolute `6900/7200/7700`）里程碑。`+200` 只看激活/明显崩坏，`+500` 看方向，`+1000` 决策。
3. 单 seed 只筛机制；只有 surviving replacement 与其 matched control 才可另开至少 3 seed 的 formal-fresh
   预注册。本 queue 的 descendants 永远不能因追加 seed 变成 formal exact。

每个里程碑固定使用**截至该 absolute milestone 的最后 100 个完整 PPO update**：`6900` 用
`6801..6900`，`7200` 用 `7101..7200`，`7700` 用 `7601..7700`。窗口内 100 个 step 必须逐个存在，
不得插值、改成累计窗或事后挑最近若干点。行为账统一读 `Live/racket_target/`，q_des 账统一读
`Live/processed_qdes_slew/`，qdot 账统一读 `Live/qdot/`；先逐 tag 求和 exact ledger counters，再做下列
除法，不能先对每 update 的比例作平均：

- completion=`sum(swing_completion_count)/sum(swing_outcome_count)`；physical fall=
  `sum(physical_fall_count)/sum(swing_outcome_count)`；legal return=
  `sum(virtual_legal_return_count)/sum(strike_opportunity_count)`；
- q_des tail mean=`sum(gated_tail_value_sum)/sum(recovery_eligible_sample_count)`，tail-active rate=
  `sum(tail_active_sample_count)/sum(recovery_eligible_sample_count)`，above-margin joint fraction=
  `sum(above_margin_joint_count)/(15*sum(recovery_eligible_sample_count))`；
- ready tilt mean=`sum(ready_tilt_rad_sum)/sum(ready_tilt_eligible_sample_count)`；qdot normalized excess mean=
  `sum(Live/qdot/normalized_excess_square_sum)/sum(Live/qdot/observed_sample_count)`，qdot excess rate=
  `sum(Live/qdot/excess_sample_count)/sum(Live/qdot/observed_sample_count)`。

processed-q_des 的 observed/previous-valid/reset-invalid/recovery-eligible/tail-active/above-margin 整数账仍须逐
update 守恒。base angular velocity、strike composite 和训练标量可作为同一固定窗的辅助诊断报告，但当前
instrumentation 没有 q_des-tail p95/max，因此它们不进入硬接受式，也不得由外部脚本事后伪造。任一硬指标
分母为零、100-step 不完整、probe 未激活或计数不守恒时结果无效，不把零写成成功。

`H` 相对同 parent `C` 的最终接受门：

- V：fall 相对下降至少 `25%` 且绝对下降至少 `5` 个百分点；completion 下降不超过 `2` 个百分点，
  legal return 下降不超过 `3` 个百分点。
- W：fall 不高于 `max(0.5%, C + 0.2 个百分点)`；completion/return 分别不劣于 C 超过 `2/3` 个百分点。
- 两个 parent：q_des tail mean 或 above-margin joint fraction 至少下降 `20%`，并且 ready tilt mean 或
  qdot normalized excess mean 至少下降 `10%`；这些比例只能由上面的固定 100-step exact ledger 计算。
- physical-fall 硬失败不能由更高 return、较小 action delta 或其他平均分抵消。

`N` 是机制负控：它用于确认完全无平滑时尾部是否放大，不是预先允许晋级的候选。任何自动 stop、自动 retry、
第二 seed、judge、部署或真机命令都不在 queue 权限内；停止只能在证据落盘后重验 exact PID/PGID/starttime/argv，
再按 [RunPod 操作页](../../operations/run_on_runpod.md#已登记-phase-1-实验臂的算力释放)处置数值进程组。

## Wave B：下半身 matched ablation（moving-teacher 输入已拒绝）

Wave A 只回答“怎么对执行目标突变收费”。它不回答机器人是否应模仿静态下半身准备姿态、左右移动老师，
或改用不依赖 demo 的下身稳定约束。下一波的 matched control 是现役 upper-only imitation；treatment 只能从
当前静态 v4rg 下半身参考或 non-demo stability constraint 中选择，上半身动作、击球 Reward、题库、parent、
seed、预算和判读尺保持匹配。

2026-07-20 对 live [`M0`](../../DEFINITIONS.md#motion-m0) 输入的只读回收已经闭合“左右移动老师能否
立即使用”：exact manifest 的 `completed_utc=2026-07-14T05:06:21.749762Z`，文件位于
`/workspace/codexschema/motion_video_intake_20260713_m0/exact_gmr_v2/completion_manifest.json`，SHA-256
`fdd60fcfdc7290677aa51ec7804278568a267e239de548cdb623d0565dac396e`。四份 exact-GMR moving 输出都通过
30 Hz、31-DoF、finite 结构审计，但 `stance_passed=false` 为 `4/4`：left1/left2/right1 的末态脚间横向
分量相对各自初始值偏离超过 `3 cm`；left1/right1/right2 的站距收窄超过 `5 mm`。right2 只通过横向分量
band，仍因 no-narrowing 失败。manifest 顶层和每个结果
均为 `formal_eligible=false`、`schema2_authorized=false`、`training_authorized=false`、
`hardware_authorized=false`。因此本轮把 M0 moving-teacher input gate 判为 **reject / no-launch**；“文件存在”
不能越过后续独立 schema2、L0/L1、桌网、动力学预注册。

当前不为 Wave B 发明 flag 或 run name。先审计 repo 中实际 body-mask/phase 接口、静态 v4rg 的 exact
下半身语义与 non-demo constraint，再单独冻结 machine-readable contract、run table 和安全门。Wave B 不得
借用 Wave A 的 launch authority，也不能与六格 action-slew 矩阵混为一个多变量结论。

## 运行表

| 运行 | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| Wave A probe2 W-C | `natural exit / verifier rejected` | W `model_6700` / seed3 | E3 mechanics only | `model_6701.pt` 与两步 ledger；无 receipt | 首步 recovery 分母合法为零；不作机制结论，不重用旧目录 |
| Wave A probe3 W-C/W-N/V-C | `natural exit / verifier contract rejected` | W/V `model_6700` / seed3 | E3 mechanics only | 三份 `model_6701.pt`；W-C 有旧 receipt，另两格无 receipt | verifier 未绑定 24-step rollout；全部不可解锁 train |
| Wave A probe4 W-C | `Hydra compose rejected before run-dir` | W `model_6700` / seed3 | E1 fail-closed preflight | 无 run dir/log/checkpoint/receipt | 错误 Hydra key；未进 Kit、未占 GPU，不是机制结果 |
| Wave A probe5 六格 | `five verified receipts / W-H pre-trainer identity reject` | W/V `model_6700` / seed3 | E3 mechanics only | W-C/W-N/V-C/V-N/V-H receipts；W-H 无 binding/checkpoint/receipt | 五收据不能解锁；W-H 是 fork→exec 采样竞态，不是 H 负例 |
| Wave A probe6 六格 | `W-C transaction-wrapper rejected / other five not launched` | W `model_6700` / seed3 | E1 fail-closed infrastructure | 81 B `run.log`；无 evidence/binding/terminal/checkpoint/receipt/RSL | multiline Python 参数被逐行缩进破坏；残留/GPU/locks 全空，不是机制负例 |
| Wave A probe7 六格 | `W-C/V-C/V-N verified；W-N watchdog teardown；W-H/V-H not launched` | W/V `model_6700` / seed3 | E3 mechanics / E1 result | 三份 exact receipt；W-N 只有 run log/ledger、无 binding/terminal/receipt | W-N 是 scene-boot infrastructure transient；v6 immutable，不作 Reward 结论 |
| Wave A probe8 六格 | `W-N sim.reset SIGABRT / other five not launched` | W `model_6700` / seed3 | E1 infrastructure only | W-N `malloc(): invalid size (unsorted)`，trainer `-6`/outer `134`；无 first iteration/binding/RSL/checkpoint/receipt | pre-managers/pre-RewardManager 失败；两 Pod 闭合，v7 immutable，不作 Reward 结论 |
| Wave A probe9 六格 | `6/6 verified receipts / completed` | W/V `model_6700` / seed3 | E3 mechanics / E1 result | 六份 exact receipt；set `cc9ff591…d9c8`；`16263be5` 历史 train artifact `fc6f1ea3…02df` 未执行 | crossover 与六收据不证明 GPU 等价，也不是科学结果 |
| Wave A v8 科学长训 | `W-N sim.reset stale / other five not launched` | W `model_6700` / seed3 | E1 infrastructure only | authority `d5c08bb9`；train artifact `be346f94…b42a`；locked `125`，outer audit `121`；result `ac09b70a…a75` | 无 iteration/RSL/checkpoint/Reward；stage-blind audit 误读 probe-only evidence；exact closure，v8 frozen，不作 N 机制结论 |
| Wave A fresh probe10/v9 | `preregistered / not launched` | W/V `model_6700` / seed3 | E1 source | fresh root；stage-aware failure audit；config/runner `3bf5085e…59e` / `0fff4515…89ce`；manifest content/file `36ceb3c7…456c` / `664375cb…0c4a` | 保持 crossover/顺序；必须六份 fresh receipt 齐全后才能生成 `science_retry2` 长训命令 |
| Wave B 下半身 matched ablation | `design pending / M0 moving rejected` | 未冻结 | E2 input gate | M0 manifest `fdd60fcf…396e` | moving teacher no-launch；只继续静态 v4rg 或 non-demo constraint 设计 |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

Wave A 是 single-swing continuation 诊断，不能声称完成 T0/T1/T2 连续恢复；即使 Isaac fall 改善，仍须
独立 vendor MuJoCo/Gate3 行为卷。

## 决定

- 决定：`inconclusive / not adopted`（v8 科学首格在 first iteration 前停滞，其他五格未发；没有 C/N/H 机制结果）
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：先把 stage-aware failure audit 与 fresh v9 identity 合入最新 `origin/main`，严格串行完成并重验六格 probe10 receipt；只有同一 v9 六收据齐全后才能生成和复核六格 `science_retry2` 的 `1001`-update 长训命令，再按 `+200/+500/+1000` 里程碑判读。Wave B 另行审计与预注册。

## 复现与证据

只读 plan、命令生成和 Pod 启动纪律见
[Run Training](../../operations/run_training.md#恢复期腿腰-processed-q_des-slew-wave-a)与
[RunPod](../../operations/run_on_runpod.md#2026-07-20-action-slew-wave-a-启动前状态与发射纪律)。
当前记录还启动过 probe7 W-C/V-C/V-N/W-N；其中前三格验证通过，W-N 由 locked watchdog 精确拆除，
W-H/V-H 未发。probe8 只启动 W-N 并在 `sim.reset` 以 `SIGABRT` 退出，其他五格未发。probe9
六格已串行完成并全部验证。v8 科学长训只启动 W-N，并在 first iteration 前由 locked stale watchdog
精确拆除；其他五格未发。fresh v9/probe10 尚未发射。至今没有科学机制结果，也没有启动 judge、部署或真机。
