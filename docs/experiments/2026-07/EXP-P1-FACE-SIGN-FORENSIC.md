# EXP-P1-FACE-SIGN-FORENSIC — 高解析上台率是否隐去了拍面反号？

- 状态：source fix implemented；C2 L1 terminal evidence frozen、v1r1 attestation/D2 与同卷复判 pending
- 工作类型：forensic（只做取证复核，不改变训练配方）
- 阶段/轴：共用判分基础 + 课程阶段 1 / 拍面符号
- 人类负责人：franco
- 执行者：Codex
- 工作分支：`Franco_codex/signed-face-honesty-20260713`、`Franco_codex/signed-face-cd-l1-v1r1-20260714`
- 最高证据等级：E4 diagnostic + source/unit gate；C2 有 terminal runtime bytes，但尚未形成成对结果
- 最后复核：2026-07-14

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

## 问题

现有 fresh `SZ model_2000` K100 报出较高的解析击球/上台率，但正手 actor raw-A
拍面误差接近 180°。要区分两件事：

1. policy 是否真在用球拍反面；
2. 解析回球器是否因 `orient_normal` 把 `n/-n` 当成同一无向平面，从而把错拍面判绿。

## 已有诊断证据

| 格子 | 正手 raw-A 有符号误差 | 人话解释 |
| --- | ---: | --- |
| model-2000 seed1 | `171.10°` | 接近完全反面 |
| model-2000 seed2 | `172.94°` | 接近完全反面 |
| model-2000 seed3 | `173.39°` | 接近完全反面 |
| model-2000 seed4 | `未测` | 没有正手 exact strike，不得补成“第四个 170° 样本” |
| model-4000 seed1 | `164.86°` | 正手 signed composite `0/50`，旧 parsed return `0/50` |
| model-4000 seed2 | `172.33°` | 正手 signed composite `0/50`，旧 parsed return 却是 `38/50` |
| model-4000 seed3 | `174.35°` | 正手 signed composite `0/50`，旧 parsed return 却是 `48/50` |
| model-4000 seed4 | `未测` | 没有正手 strike；不得把缺测补成 180° |

现有同一击球状态的 Isaac/MuJoCo 有符号误差约为 `170.72/171.09°`，而解析回球路径
会先对法向做方向归一。这已构成“分数可能对符号失明”的可复现反例，但还不是
修正后的新考卷结果。

model-4000 配对 K100 把反例收紧为同 checkpoint/同题同卷的直接矛盾：seed2/3 的解析
正手回台为 `38/50` 与 `48/50`，但保留 raw-A 符号的位置+速度+法向复合命中均为
`0/50`，法向差为 `172.33°/174.35°`。因此旧 parsed return 在正手上已不具备
checkpoint 晋级资格；这不需要等新训练终档才能判断。

### 训练信用冲突的原始曲线

seed3 的 TensorBoard event 另给出一条独立于终档判卷的机制一致性证据。content-bound 小摘要是
[`phase1_fresh_SZ_seed3_training_face_reward_forensic_20260713.json`](../../../configs/phase1_fresh_SZ_seed3_training_face_reward_forensic_20260713.json)：
文件 SHA `6b6ff8e5a98c38ee9ea3856820fa6876aa9a6422eaa931be408a14a18859147a`、canonical
content SHA `07186e1a6d70371b4d038705b7ae3f068d68e17cf4cb377ad18333cc5bae8181`；它绑定原始 event
SHA `c1578922...25e`、训练合同 SHA `3a3b3d95...b9972` 与实际 `params/env.yaml` SHA
`4dfb829e...a70051`，没有把约 518 MB 的 event 文件提交进 Git。冻结的 `env.yaml` 明确记录
`commands.racket_target.virtual_ball=true`、`vb_metrics_only=true`，以及
`virtual_pass_net/virtual_landing/virtual_spin/racket_normal` 权重 `20/30/5/5`。同一 run 的 launch
SHA `20fbb3cc...7cbf0` 还绑定了选臂命令、`vb_metrics_only=true`、`racket_normal=5` 与
`face_guidance=0`；当时 267-byte `git/nohope.diff` SHA `8d873bdd...8c65f2` 记录 05:40 工作树
clean/no diff。launch 不含 HEAD，因此 `6d93bcb...480b` 只能记为 scheduler/saved-lineage claim；
no-diff 证据不能独立证明运行时 HEAD 或进程实际加载的源码字节。
摘要用 TensorBoard `EventFileLoader` 解码 tensor/simple value，九个预定 iteration 都是 exact-step
命中，不使用 nearest fallback。

| iteration | 正手有符号法向误差 | 正手 normal pass | 正手训练内解析回台 | 反手法向误差 / pass / 回台 |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | `120.17°` | `.000` | `≈0` | `9.15° / .885 / .686` |
| 2000 | `167.49°` | `.000` | `.692` | `6.45° / .976 / .938` |
| 4000 | `172.44°` | `.000` | `.897` | `7.63° / .955 / .943` |
| 8000 | `173.04°` | `.000` | `.940` | `6.33° / .995 / .969` |
| 13800 | `174.02°` | `.000` | `.965` | `5.86° / .996 / .967` |

iteration 13800 时，全局 `Live/Reward/*` tag 中
`virtual_landing + virtual_pass_net + virtual_spin = .4615195`，而 `racket_normal = .15587743`，
由冻结值复算为 `2.960784637×`。这些 tag 汇总所有环境和正反手，**不能**把这笔 reward 归因到
正手错面样本，也不能量化正手错面贡献了多少。结合有符号正手指标、冻结运行配置和旧 face-blind
源码路径，支持的最窄结论是：**wrong-face FH states were treated as reward-eligible by the active
face-blind reward path**（正手错面状态被现役 face-blind reward 路径视为可得奖）。这不是“回球奖励
单因素导致反面”的因果估计；reward、policy 状态和训练 iteration 同时变化，fresh 配对 canary 才能
回答修门后的学习行为。

## 源码修正与边界

本分支把“无向冲量平面”和“有向物理拍面”拆开：

1. NumPy 解析 scorer 在任何 `orient_normal` 前，先比较 achieved/target raw mount `+Y`（A）法向，
   再用每 clip `[+1,-1]` 映射到外部 physical face B；只有 `dot(A_achieved,A_target)>0`
   且 achieved/target physical-B 都严格 `x>1e-6` 才能接触记分。
2. `n/-n` 负控锁定旧病：两者的冲量、出球和落点仍逐值相同，但只有正确拍面可 `contacted/landed_ok`；
   错面必须在 plane orientation 前失败。
3. Isaac `_vb_evaluate` 用同一 convention-matched face pair 和 physical-B `+X` 门生成
   `vb_fired`；`virtual_pass_net/landing/spin` 因而不能再消费错面样本。源码门不是训练行为通过，必须
   从该源码 fresh 起跑。
4. MuJoCo formal analytic scorer 必须从 ONNX metadata 读取完整
   `mount_normal_sign_per_clip`；缺失/非法/长度错误 fail closed。只有显式
   `--allow-inexact-contract` 能跑旧 unsigned-plane 诊断，而且结果写
   `signed_face_exact=false` / `evaluation_contract_exact=false`，不得晋级。
5. 动作 phase/spatial screen 的调用点已迁到 raw-A + target-A + clip-id 接口。历史 v5 和旧 q50
   仍绑定旧 scorer，不能倒改成新证据；未来 screen 要新 prereg/source SHA。

本地 focused 回归为 `38 passed, 1 skipped`，顶层 broad 为 `546 passed, 9 skipped`；另一个排除
Torch/Hydra import-bound 文件的 training dependency-light 组合为 `381 passed, 21 skipped`。focused
skip 与未收集的运行时模块都源于宿主没有 Torch/Isaac/Hydra，不是行为通过；Isaac canary 尚未执行。
没有访问 Pod、启动 judge/simulator、改训练或运行真机。

## 2026-07-14 C2 outer-verifier 假拒绝与 D2-only 续接

[`signed-face C2/D2`](../../DEFINITIONS.md) 的 C2 在 Pod1 GPU1 已按冻结 source/seed/recipe 自然跑到
`model_24.pt`，但 v1 outer verifier 没有写 `runtime_verified.json`。根因是表示层而不是 trainer
配方：manifest 的摘要字段使用整数 `[1,-1]`，实际 Hydra 命令使用 `[1.0,-1.0]`，训练端又显式
`float(x)` 后把 `[1.0,-1.0]` 写入 hard contract；v1 `require_exact` 同时要求 Python 类型相等，于是
把合法 float wire value 误报成 bool/int confusion。

冻结的 terminal 证据为：canonical claim `37fe2443...86e5`、final log `abffd457...6dc3`、hard
contract `83f47ae6...2772`、`model_24.pt` `dbbc7a28...6f6`；launch contract/state 分别是
`26bf204d...0e96` / `2bcc5656...beb8`。旧 runtime/failure/result 三文件必须 absent，D2 arm 与 exact
run name 也必须 absent。C2 checkpoint 仍需由一次性 consumer 重算 canonical claim、finite/iter24/
lineage1 和 checkpoint↔hard-contract/claim binding 后才可进入 pair；这里不把缺失的 v1 runtime sidecar
补写成“当时已验证”。

修复采用 [`v1r1` one-shot continuation](../../DEFINITIONS.md)，而不是重跑 C2。新 manifest
`8d893009...6e232` 冻结上述六个 SHA 和 absence boundary；新 verifier 只接受 exact float
`[1.0,-1.0]`，显式拒绝 `[True,-1.0]` 与 `[1,-1]`。C2 attestation 写入独立
`continuations/v1r1/` evidence root，不向 preserved C2 arm 增加文件。唯一 launch mode 是
`launch-d2`：它要求 C2 attestation 可逐值 replay、D2 未 claim、GPU2 空，并让 D2 checkpoint 绑定
v1r1 manifest/launcher、原 v1 source/recipe 和 C2 attestation。最终 pair 必须显式承认 mixed outer
control（C2=v1、D2=v1r1），同时证明规范化 trainer recipe 与 hard contract 都只差 signed-face
weight。activation/judge/L2/第二 seed/stop-promote/真机继续为 false。

本分支只完成本地 source/contract gate，没有连接 Pod 或启动 D2；运行步骤见
[C2/D2 L1 操作文档](../../operations/run_phase1_signed_face_cd_l1.md)。

## 预注册决定规则

- 为旧 K100 结果生成 content-bound signed-face 诊断表，保留 raw-A 有符号误差。
- 增加 `n` 和 `-n` 必须得到不同判定的负控；对符号无感就 fail-closed。
- 不改训练合同，不因旧高上台率晋级 checkpoint。
- 最终要与同一 K100 的 Isaac/MuJoCo × physical/analytic 2×2 仪器表对账；只修解析 scorer
  不能证明跨引擎 gap 关闭。

## 当前决定

Fresh `SZ model_2000` 的已有成绩仍保留为“解析诊断卡”，但在 signed-face honesty gate
通过前，不得称为 accepted baseline，也不得用来证明 physical return。
同样，model-4000 seed2/3 的 `.88/.98` 只保留为旧 scorer 的失真证据，不是好 policy 候选。

下一步闭环顺序固定为：

1. 按[单-seed 机制漏斗](EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)用带新 hard-contract/source SHA
   的小型 fresh 双侧 canary 验证错面不会获得 `vb_fired` 和三项 virtual reward；必须同时看到正手
   signed normal/return 的学习曲线，源码测试不能代替，也不要先复制四个 seed。
2. 用同一 immutable K100、同 checkpoint、带 exact face metadata 的新 scorer 复判历史模型；旧
   scorer 结果保留为 paired legacy column，不覆盖。
3. 再做 Isaac/MuJoCo physical/analytic 2×2 归因；analytic 即便修正也只可诊断，最终仍由 vendor
   Gate3/Gate3B 判定。

证据入口：[Fresh SZ 稳定性实验](EXP-P1-FRESH-SZ-STABILITY.md)、
[model-4000 aggregate](../../../configs/phase1_fresh_SZ_model4000_seed_stability_q50_aggregate_result_20260713.json)、
[G05](../../gates/G05_isaac_training_first_loop.md) 和 [G06](../../gates/G06_isaac_to_mujoco.md)。
