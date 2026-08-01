# G05 Isaac Training First Loop

Status: Partial (the base training-loop mechanics are proven; the current-candidate promotion sub-gate is open)

**2026-08-01 SUPERSEDE — current vendor N1 集成放行合同：**今晚 plant authority 已冻结为
非冲突 parkour 新表 + task/SKU 三处 fallback：waist-yaw Kp=`85`、
waist-pitch effort=`118`、wrist-pitch/yaw=`Kp20 / effort6 /
armature 0.0008100893338`，wrist-roll 仍为 `Kp30 / effort24 / armature0.004968`。
当前只差 Pod 对拍；智元未来直接确认 24 N·m 只产生下一版 plant，不热改今晚 run。

放行合同只跑 A=`bh_loop_c` static、B=`bh_block` static、C=`bh_loop_c`
monotonic adaptive-sigma 三条 fresh-only lane。三条共用智元 `1–3 s` 六轴
velocity-only push，`force_push=false`、`combined_exclusive=false`；live stage 只有
`smoke=1×2×save1`、综合 `probe=4096×5×save1` 和
`long=4096×20001×save100`。每 lane 只由一份
[`n1_vendor_probe_gate_receipt_v2`](../DEFINITIONS.md#n1-vendor-probe-gate-receipt-v2)
放行 long；其 `stages` 只有 `probe`，硬门 source/plant、194/318 real-runner
normalizer roundtrip、finite checkpoint、std-LR、delay、joint actual-hard、qdes、
nonfinite、natural completion，以及 push event/applied>0 与六轴 extrema finite/in-range。
table/fall 频率、strike-window 和 recovery 是 20k 前 100 update 持续 telemetry，
不是 5-update blocker。standalone `push_evidence=4096×32` 已退役；旧 spec/receipt/
namespace 只作 spent history。本页下方与此冲突的 `probe → push_evidence → long`、
`5–15 s` 或 5-update behavior blocker 均显式作历史运行记录。测试和真实验收只在
Pod 进行；本地只跑 `git diff --check`。MuJoCo P0 与架构 P1/P2 不混入今晚
prelaunch source。

**以下全部是按时间保留的 historical/superseded Gate 证据，不参与 current
调度。**其中 `80/115/腕30-24`、`5–15 s`、standalone `4096×32`、旧多 stage
receipt 和 table/fall/strike/recovery 的 5-update blocker 都只表示当时候选或当时裁决；
如与上文 SUPERSEDE 冲突，一律以上文为准。

2026-08-01 权威与发车更正：Franco 明确裁定智元新 A3 setting 比仓库旧
nominal 更权威。因此 clean `C1=cc0020e2…` 及其三个 probe claim 不再是今晚
plant identity，即使 launcher 修复也禁止恢复。新 source 必须采用 waist-yaw Kp `80`、
waist-pitch effort `115`、全部 wrist roll/pitch/yaw `30/2/24/0.004968`，并按
`0.25×effort/nominal_Kp` 重算 action scale。首次 A probe 另在 PPO 前揭示 Hydra
append 接线错误：动态 table-attribution 键必须使用
`+task.table_contact_attribution_diagnostic=true`；失败 namespace 无 PPO/无 checkpoint。两项确定性
修正、focused tests 和从新 clean source 开始的 identity→authority→hold/bundle→A/B/C
pins 重签未闭合前，G05 继续 `Partial`。
另一项引擎无关补漏是 vendor leaf 继承 `reward_pack=v2` 后实际会得到
`action_acc_weight=-0.05`。这与“只保留 action change penalty”冲突，因此 leaf 必须
显式钉 `action_acc_weight=0.0`且保留 `action_rate_clamped=-0.2`。新修改的验收
只认 Pod 固定环境；本地结果不记 Gate PASS。直接 MuJoCo 尽调完成前暂停新增
USD/Hctrl/PhysX contact-view/Isaac receipt 专属 feature，但 194-D、Reward/adaptive-σ、
task clocks、DR schema 与长训 metrics 作为可迁移合同继续前进。

上述确定性修正在 clean `b57a9685f5a10e5c2c7485705368eac8324f5a3e` 落地；Pod exact
checkout 的 nominal/task focused suite 为 `8 passed, 14 skipped`，launcher 为 `78 passed`。
Reward suite 初跑的行为断言已经证明 effective
`action_acc=0 / action_rate_l2=0 / action_rate_clamped=-0.2`，唯一失败是测试把两条审计日志误当成
同一字符串。successor `ca36512670ffe994af7ab020a021603201735288` 修正测试后，Pod exact
checkout 重跑 reward 全文件为 `249 passed in 1.96 s`。这关闭本次 source-focused 门，但不授权
沿用旧 C1 工件或发 Isaac long；若 MuJoCo 尽调选择短迁移，后续只实现薄适配层，若今晚回退 Isaac，
才更新仍含旧 nominal 的 runtime-authority/replay producer 并从头重签工件链。

同一 clean source 的六项跨引擎 golden-contract 只读盘点确认：fixed-194 schema/禁 one-hot、
table/heading 数学、`[0,2]` delay、sampler replay、Reward/adaptive-sigma 和 normalizer 组件覆盖均强，
不应再复制测试。选定后端仍必须补 31-D ordered action scale→lag0/2→qdes 联合 golden；若选
MuJoCo，另补完整 194-row 数值与 Torch frame parity。任何 20k long 前，还必须在 Pod 当前
`rsl_rl` 真实 training runner 上证明 checkpoint save→第二 runner load 后 194/318 normalizer
tensors 全等、count 不回退。sampler command-level resume 和 adaptive-normal resume 可延后，
所以本轮只授权 fresh-only，不宣称 formal exact-resume。

MuJoCo 今晚路线的后续独立复核把旧 preflight 分成两半：table/net/keepout、native-ball teacher
diagnostic 和单环境 PD/reset/evaluator 已存在，故“场景/评测未实现”已过时；但 fixed-194/
critic-318 trainer、per-term Reward、batched reset/timeout、delay/push、VecEnv/PPO、training
checkpoint hooks 与 4096-env 吞吐仍未实现，mjlab 也没有已接入的 A3 task/backend。因此今晚默认
继续 Isaac；除非 CC 交付可直接通过 `194/318 + PPO + 4096×5` 的运行分支，不因架构报告改发射
引擎。另发现 `b57a9685…` 尚保留非冲突关节的旧 MJCF armature 精度；Franco 已裁定整套智元
29-DoF 表更权威，故 robot cfg、runtime-authority 与独立 MuJoCo replay 现按全表并行修正并由
31-D decoder/delay golden 对拍。head 不在厂商 29-DoF 表内，继续具名 HOPE fallback。

2026-07-31 C1 probe 与机械/桌体候选补记：三条 `1 env×2 update` smoke 均自然完成，
三条 `4096×5` probe 也都自然完成且 checkpoint finite，但共享安全门未过。loop static/
adaptive 的 actual-hard 为零，但 table 分别为 `5,456/491,520=1.110%` 与
`5,455/491,520=1.110%`；block table 为 `3,769/491,520=0.767%`，并有 `3` 次
`right_ankle_roll_joint` actual-hard。三者均超过既定 table Gate `0.5%`，因此 loop static、
loop adaptive-σ 和 block static 三条 long 全部继续 fail-closed，不用 natural completion 或 finite
checkpoint 绕过门。

本轮两个可分别归因的修复中，四轴 plant 机械门已由 v7a Pod PASS，table
attribution 与 fresh 三 lane 仍未 PASS：（1）把 2% PhysX
控制位置包络从两腰对称扩到腰 roll/pitch 与双 ankle roll，其他 27 轴、
[`Hmech`（机械硬边界）](../DEFINITIONS.md#h-mech)、
soft q-des、Reward/actor/observation 不变；新 schema-3 显式绑定四轴顺序、`0.02`、完整
`31×2` Hmech/[`Hctrl`（PhysX 控制保护边界）](../DEFINITIONS.md#h-ctrl)
与不变式，必须用 16-env v7 全系统同带 ON/OFF stress 出 clean receipt；
（2）table attribution 默认关闭且只做诊断，在完全保留现役 conservative terminal mask 和
`0.5%` Gate 的前提下，逐 component/blade×五件桌体计算 exact
[OBB-vs-AABB](../DEFINITIONS.md#obb) [SAT](../DEFINITIONS.md#sat-collision-test)，分开
exact overlap、broad-only 与 nonfinite，再按 body/obstacle/swing phase 记首中账本。这一诊断不改
Done、Reward 或 Gate。四轴 stress 现已完成，但 table diagnostic 定价、fresh
`4096×5` 和 push/long 还未完成，因此 G05 仍保持 `Partial`。

四轴 plant/diagnostic 候选本身通过了早期复核，v7 receipt producer 的独立终审又发现
joint/index 绑定、validated runtime→receipt tape 绑定、literal version closure、
runtime order→独立 PhysX public live-limit identity，以及 identity→实际 live readback
attestation 五个发车 P1。它们已按下列合同全部关闭，但在 fresh Pod receipt 前
仍不能把机械 stress 写成 PASS。修复给出 exact row keyset、
`joint_index == live_joint_order.index(joint)`、receipt 前完整重验、旧 v6 confirm token
拒绝，以及同步伪造 runtime order/tape 时仍由 public selected names/indices 拒绝的负例；
四个 readback/target-only proof 必须 exact true，joint-order digest 可重算，public/run-specific
readback SHA 必须相等。四轴
v7 stress 目标仍是 4 轴×2 侧×ON/OFF=`16 env` 的唯一变量证明，
对账完整 31-D 初态、每 tick 31-D qdes、origin-relative root、被移至远处的 external rigid
objects 与 finally exact restore；validator 不信任 producer 布尔摘要，会重算
full-state/pair digest。最新 host focused=`82 passed`、identity+probe=`113 passed`，
`py_compile` 与 `git diff --check` PASS；独立终审确认 P0/P1=`0`。因此当前只授权
提交/push 这组 exact bytes。实现+文档 commit=`04b50343a1455914c79bcbf6f8080551864ab289`
已 push；下一步从本次纯进度账本后的 clean successor checkout 跑 Pod 六文件 torch 组合门与
no-clobber live v7a。旧 clean `956a7a3a…` 的两腰 v7 PASS 不能为新增左/右 ankle-roll 代签。

clean successor `62e0878ac52748373838850faf02c3be1c9f16bc` 在 Pod 六文件 torch
组合 `377 passed`，后经 pod 级 Kit boot lock 在 GPU0 共存发射 16-env no-clobber v7a。
receipt canonical/file/log SHA=`79e14853…/cb0fcfdc…/087028bf…`，status=`PASS`且
canonical 独立复算一致。8/8 OFF 组 tick1 进 `[Hctrl,Hmech)` 并四 tick 内触/穿
Hmech；8/8 ON 组 32 ticks strict Hmech；64/64 contact=`0 N`，8/8 full input tape exact，
restore exact，four live proofs exact，public/run readback SHA 相等。因此四轴 plant 机械门
已关闭；G05 的当前阻塞下移到当前双动作工件/pins、table attribution、fresh
`4096×5` 与 push/long。

r4 下游身份正在从该 PASS 后的 clean source `f0889ab1…` 重物化。Pod1
GPU0/GPU2 的 loop/block recipe 与 `1 env×2 update` identity smoke 均自然退出，四份
checkpoint finite，live contract SHA=`912381de…/19d4cfed…`；required identity
SHA=`a4360e39…/78267461…`。四份文件已回填 r4 action registry，待 focused gate 与
clean successor 后才允许物化 runtime authority。这一层仅关闭 runtime identity 循环，不授权
candidate/contact bundle/long，G05 继续 `Partial`。

required identity 与 tracked runtime contract 已经 focused `86 passed` 并以 clean
`C_RI=3c46a44d…` 推送。从该 exact successor 生成的 loop/block runtime authority file
SHA=`2b96d2ea…/226d3788…`，两份都明确记录 training/hardware/deployment authorization=false。
回填后仍必须过 validator 并产生 clean `C_AUTH`；未提交的 authority 不能被 candidate 消费。

clean `C_AUTH=cd2375c7…` 的 Pod torch focused gate=`72 passed`；loop/block dynamic-ready
candidate file SHA=`a314f0b2…/b0f92092…`。两条 nominal hold 均在真实 Isaac plant 上完成
`0.8 s/160 physics steps/40 policy steps`，plant contract exact、双脚接触率 `1.0`、无
terminal/truncation；receipt SHA=`5acee65a…/298beec2…`。随后物化的 loop/block contact
bundle 均 status=`PASS`，SHA=`a57c3ca3…/26931c76…`。这些仍须在 clean `C0`
跟踪并通过 consumer gate；未过 fresh `4096×5` 与 push evidence 前仍不授权 long。

bundle 已以 clean r4 `C0=ba195165…` 推送，Pod exact C0 的 bundle/consumer 组合
`183 passed`。同一 C0 上三个 zero-PPO 任务均 accepted 且自然退出：loop/block
policy SHA=`edfffec3…/44c20720…`，loop adaptive effective Reward=`6520f153…`；全部
0 PPO/0 checkpoint 且 authorization=false。三 pin 回填成 clean C1 后才能发 current
`4096×5`；旧 C1 smoke/probe 不为 r4 代签。

clean `9819a8623a913d472fc764cef8d0c9f1a4f8ee83` 在 Pod 的依赖完整
CPU/torch/hydra 回归新增 `259+313+196` PASS。v5 首次在 Kit 前因输入短 SHA
正确 fail-closed；完整 SHA 重跑在 vendor profile bind 暴露 task-first exact-key consumer
尚缺 `attribution_diagnostic` 与 `attribution_command_name`，spent FAIL content SHA
`ea42ce39…` 保留。候选修复只将两键纳入 strict set，并证明 diagnostic 为 exact
bool、与 top-level cfg 相等、command 为 `racket_target`且不能在非 ActionBall 开启。
该修复不改默认路径、terminal、Reward 或 Gate。独立复核确认无第二个 full-param
strict consumer，并要求 top-level cfg 也是 exact bool；正例与缺键/extra/非 bool/不一致/
command 漂移/legacy 开启负例 `10 passed`，runtime-contract `13 passed`，launcher/override
组合 `336 passed`。新 clean source live v5 通过前 G05 仍为 `Partial`。

clean `8d0b8ba09ee50feb7883428d9d3bf4e91f618f74` 的 Pod 组合 `349 passed`。
v5c 已完成 scene 创建、16 env 组装及四轴 Hctrl live install/readback，但在 reset 前
被旧 action manifest 的 solver profile 身份正确拒绝：manifest `af4f6f95…` vs runtime
`f89587db…`，spent FAIL content SHA `a0f8d352…` 保留。这是 source 变化后旧
profile/manifest/pins 不可复用的身份门，不是 Hctrl verdict。最短正确依赖改为
formal profile pinner → 新 action profile/manifest 身份 → live v5 → 其余 pins；禁止在 stress
内绕过 solver gate。G05 仍为 `Partial`。

formal pinner 已在 clean `8d0b8ba0…` 上闭合：physics SHA=`aa5c9085…` 不变，
solver SHA 更新为 `f89587db…`，profile 文件 SHA=`509f3812…`。当前进入
`r4` no-clobber artifact epoch，identity 及其后所有 action-specific pins 先置空
fail-closed，新 identity materialization 接受后再逐层回填；禁止用旧 r2/r3 pin 填补。
P1=`6a7587c05bc3fdb6c6070b72da12e251ce58795b` 已 push；loop/block 在两个独立
clean worktree 并行物化了 r4 identity 三件套，materializer 两边各 `19 passed`。
manifest SHA 分别为 `e7531567…` / `7870a053…`；receipt 分别为
`3c79f266…` / `9ffe90cc…`。当前只回填 identity 层，下游 pin 仍 fail-closed。

clean P2=`e7917b1479980fb9c85e89b25c011ae9b1e52f38` 的 Pod v5d 已在 16 env
自然执行，spent FAIL content SHA=`2d10999c…` 独立重算一致，restore exact。
腰 roll/pitch 的 4 个 OFF/ON 轨迹表面满足行为要求，但红队发现 ON/OFF
root x/y 并非 exact，且 validator 在后置 root parity 前已因踝部 verdict 早抛错；
因而腰部只能记为行为观察，不是因果 PASS。左/右
ankle-roll 的 ON/OFF 都被当前踝部动力学弹回内侧，OFF tick1 未入
`[Hctrl,Hmech)`，因而踝轴正控无效。下一步统一空中 root/零速度、证明零外部
contact，并把 pair parity 前置；若离地后仍不触界才逐轴增加外向应力。不改 2%
Hctrl、不覆盖 v5d。G05 仍 `Partial`。
新 v6 source candidate 已将 16 env root 写成同一 origin-relative `[0,0,3 m]`/
identity/零 6-D 速度，每 tick 从 `contact_forces` 证明全机器人外部接触力
`<=1e-6 N`，并将全部 pair-input parity 前置到动力学 verdict 之前。schema/kind/
confirm token 均升为 v6，host 正负测 `66 passed`（含同步 q_des 漂移不能靠 pair equality
蒙混、且 input gate 必须先于 outcome 的复合反例），与 r4 identity smoke 合跑
`97 passed`；两轮独立只读终审均为 P0/P1=`0`。本机无 torch；更广的 Pod 组合随后在
clean `ff41b12c…` 得到 `361 passed`；v6a receipt canonical/file/log
SHA=`c9c56bde…/52ca07fa…/e6d4adf0…`，restore exact、64/64 contact=`0 N`、8/8 input
pair exact、80 seals 可复算。首因是全 env 相同 identity quaternion 的 PhysX 数值规范化
产生约 `7.9e-11 rad` 物理角，被 component 字面等值误杀；四个 ankle OFF tick4 另尚余
`0.000308–0.000318 rad` 未触 Hmech。v7 因此只把声明姿态改成双覆盖物理角
`<=1e-9 rad`（raw pair 仍 exact），并把 ankle outer stress `0.60→0.65R`；腰、qdes、
四 tick、2% Hctrl/contact/verdict 不变。host v7 正负测 `71 passed`、与 r4 identity 合跑
`102 passed`；tape 固定公式逐行重算、quaternion norm=`1e-12`，raw pair 异号即使物理同姿态
也 fail。fresh Pod v7a 已按上文证据 PASS。

2026-08-01 identity source-gate 补记：probe/push→long gate 不再读取已退役 runtime-source
label，而是绑定真实 action registry 及 action-specific bundle/required identity/authority/contract/sigma。
identity-repin producer 同时改为 per-action pin，loop/block 的 receipt 不再被单个全局 SHA 串绑。
真实跨组件、双动作共存、cross-action 与 worktree drift 回归已覆盖；全链 `382 passed`。
这些只是 host source gate，clean-source compose/Pod 未过前 G05 仍为 `Partial`。

2026-08-01 source-candidate 补记：当前未提交集成已把 `bh_block` 扩为动作专属
registry/identity/authority/dynamic-ready/gate 链；缺任一新动作工件就 fail-closed，不能回落到
`bh_loop_c` pin。另为 `bh_loop_c` 加入 fresh-only 单调自适应 σ canary：宽核
`0.20 m / 1.0 m/s / 0.52 rad` 以位置/速度/法向双 term 锁步，只收紧到
`0.075 m / 0.5 m/s / 0.262 rad`，static 粗位置核仍为 `0.30 m`，且禁止 resume。
两组 host 定向回归分别为 `117 passed` 与 `322 passed`，但它们都还是 source-only、未 Pod、
未采用。自适应 canary 的 Reward SHA 还必须由 clean source 的零 PPO hash-only 阶段物化并重钉
完整身份链。shared actual-hard 失败未闭合，因此两动作 long 继续 fail-closed；host PASS 不构成
本 Gate 的 Pod 验收证据。

2026-07-31 vendor-A3 checkpoint：旧三条 fixed-194 stable-ready milestone1000 已全部自然
完成，`model_1000.pt` 均 finite，但约 `797–1043` 次 strike opportunity 下
capture/return 全为 `0/0`；它们是旧 plant 的 E3 负证据，不再续跑也不 resume 成新
setting。智元新权威 nominal/scale、`[0,2]` control-step delay、六轴 `5–15 s` push 与粗细拍距核
已在 `89082b7c` 上完成 Pod smoke/probe/push-evidence，checkpoint 均 finite，但 probe/push 分别有
`4,873/37,417` 次 actual-hard，不具备 long 资格。新 source 已用
`max_inward_until_nonoutward_v1` 修复跨 policy/substep 的最大向内 emergency containment，并完成
push 六轴运行 counter、exact-resume 状态、自然完成 marker 及 probe+push→long 的严格
producer/consumer gate；独立末审 PASS。Gate 仍为 `Partial`，现只等 clean source 重物化后的
smoke、`4096×5` probe 和 `4096×32` push-evidence 同时过 actual-hard/qdes/nonfinite 与有界任务失败门；
在此之前 formal N1 与 `4096×20001` long 仍 fail closed。
## Goal

Run the first end-to-end Isaac training loop that produces a policy artifact, even if the policy is weak.

This gate should prove that the training stack can consume A3 assets and produce a deployable policy format.

## Inputs

- Isaac-ready A3 asset from G04.
- Motion references or placeholder task references.
- BeyondMimic/whole-body tracking scaffold.
- Policy observation/action contract.

## Outputs

- First accepted training run.
- Training config.
- Logs and metrics.
- Exported policy artifact path or metadata.
- Initial policy evaluation notes.

## Related Directories

- `hope_training/whole_body_tracking`
- `docs/interfaces/policy_observation_action.md`
- `docs/operations/run_training.md`
- `vendor_assets/` for generated heavy policy artifacts if needed
- `external_repos/TTRL-ICRA2026` as an auto-synced reference if first-loop failures need comparison

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_phase1_lower_body_stability_wave.md](../operations/run_phase1_lower_body_stability_wave.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

Core first-loop mechanics (already demonstrated):

- Isaac environment starts with the A3 asset.
- A random rollout works.
- A first PPO or equivalent training loop runs.
- Policy export path is documented.
- First-loop results are recorded, even if poor.

Current-candidate promotion sub-gate (required before this Gate can close):

- The selected checkpoint has exact training-contract lineage rather than an allowed warm-resume mismatch.
- A fresh deploy ONNX preserves the exact contract and passes finite, dimension and inference checks.
- The checkpoint and ONNX are bound to an immutable behavior paper and handoff contract. This closes the G05
  artifact sub-gate; vendor MuJoCo behavior belongs to downstream [G06](G06_isaac_to_mujoco.md) and is not a
  G05 closure criterion.

## Current State

Follow-up note (2026-07-30, qvel-fixed 4096 probe rejected; Gate remains `Partial`):

- stable-upper v1 loop/block 的 Pod `1 env×2` 都自然完成且 checkpoint finite，但每轮都在
  age `16--17` 唯一由 `waist_pitch_joint` 上侧 raw mechanical edge reset；q_des/table/fall/
  nonfinite 和腿/踝 hard 均为零。两动作相同事件证明 lower/root replacement 生效，但 v1
  错把旧深蹲动作 frame-0 `waist_pitch=+0.103 rad` 当成 A3 stable ready。v2 直接把三腰
  q 轨迹按 frame-0 常量平移到 `AGIBOT_A3_CFG.init_state` 的零位，保持每帧相对增量和 qd；
  这是同一 A3 birth/reference 修复的补全，不做学习 A/B。Pod producer focused test
  `2 passed`；loop/block 新 motion SHA 为 `0fa46ad6…` / `cc9bbccd…`，双脚 `3+3`、
  static LP `feasible=true`，重建 contact 后重新 smoke。
- N1 contact producer 已钉到 stable-upper v2 exact bytes；声明的 runtime-site
  finite-difference task-contact speed 为 loop `1.8083184461 m/s`、block `1.5947043195 m/s`。
  Pod focused regression 为 `11 passed, 2 deselected`；v2 loop/block bundle SHA 为
  `85c7a276…` / `09d0dea3…` 且 materialize PASS。policy receipt 尚未物化，故当前仍不可发
  probe/long。
- 真实 Isaac scene 已物化 v2 loop/block policy contract `03f833e1…` / `3442881f…`；
  lower+waist normalized bias 全零。recipe-only 子进程在写出工件后退出且未产生
  `Learning iteration` marker，boot wrapper 因而报告 rc=1；这不是 PPO 失败，下一步由正式
  launcher 绑定 recipe 进行 smoke。
- 历史最接近的“早期能恢复”run 是 warm-resumed `s1w4_M2_v4rg`，不是 fresh N1：它从
  `model_13000` 恢复 optimizer，约第 2--12 update 才出现 strike。保留日志没有 fresh
  0--300 update。因此五轮 probe 的结论严格限定为“当前发射不具备 long 资格/存在击球前样本
  饥饿”，不得写成“policy 永远学不会”；stable probe 通过后用 fresh 100--300 update recovery
  决定是否续 long。
- Pod1 已完成 `bh_loop_c` 与 `bh_block` 的 exact `4096 env × 5 update × save1`。两者 mean
  episode 约 `23/12` 步、strike opportunity 均为零；actual raw-hard 分别约
  `2.5k--4.2k/update` 与末轮约 `7.7k`。checkpoint 均 finite，q_des
  projection/nonfinite 与 table 不是主因。当前约 `2--3k environment-steps/s`，故不授权
  qvel-fixed long。
- actual A3 birth 是 root `z=0.920683 m`、pitch `-11.19°` 加深蹲腿位。它虽有 exact MuJoCo
  双脚几何接触，但不是训练 implicit-PD plant 的闭环 stable hold；tracked static-ground
  receipt 此前为 `feasible=null / missing scipy`，不得写成 PASS。qvel-only 只修了 schema，
  已被行为 probe 证明不是 reset 风暴根因。
- successor 固定为 stable-upper：head/arm q/qd、腰轨迹增量/qd、frame/timing/strike 不变；
  三腰 frame-0 改 runtime ready 零位，12 腿改 runtime
  default stand，root 保 X/Y/source yaw 并改 upright、`z=1.0684 m`；exact A3 重建 FK 后
  重新物化 ball/task binding。它是正确性修复，不做学习 A/B，但仍须 Pod deterministic
  hold、`1 env×2` 与 `4096×5` 行为门。Gate 在 episode 跨过 `t_hit`、strike 非零、
  raw-hard/table/fall/nonfinite 不爆炸和 finite checkpoint 全齐前保持 `Partial`。
- stable-upper producer 已在 Pod focused regression `12 passed`；反手拉/挡 motion SHA 为
  `4343a85e…` / `08aeafaf…`，exact A3 均为双脚 `3+3` 接触且 static-ground LP
  `feasible=true`。击球帧拍速保持，但 selected face world position 改变最多
  `0.138/0.064 m`，故旧 contact task 作废；新 N1 bundle 必须把原 profile 宽度整体平移到
  新 face center 后再进入 smoke。该重绑已 materialize PASS，loop/block bundle SHA 为
  `054be7f2…` / `6973f1a3…`；Isaac scene 物化的 policy contract 为
  `80c70eb3…` / `359d4e97…`，stable lower normalized bias 全零。尚未获得 PPO smoke 行为证据。
- launcher contact receipt gate 现按 exact alignment keyset 区分 legacy upper corrected-Z 与
  stable-upper retargeted-center；后者必须绑定
  `a3_stable_upper_selected_rubber_face_center_at_pinned_strike_frame`、明确不保留旧 center，
  且 world-Z 与 ready-root + B-yaw task-Z 闭合。该兼容修复不改变训练 setting；Pod focused
  regression 和两动作 PPO smoke 仍是下一证据。

Follow-up note (2026-07-29, `eaf55fba` raw-hard counterexample and A3 upper q/qd root cause;
Gate remains `Partial`):

- Pod1 exact `eaf55fba5e201d76153162ab2f7f482bb66b3f22` has already separated recoverable
  2%-inner occupancy from `joint_actual_forbidden`: only nonfinite/current raw mechanical edge or a
  physics-substep raw-hard latch remains terminal. Its fresh 4096-env upper `bh_loop_c` diagnostic
  completed updates 0--4, but actual hard terminal counts were
  `2,549/3,986/4,225/4,188/4,162`; mean episode age stayed about `22--24<t_hit≈31` and strike
  opportunity stayed zero. q_des projection count, projection penalty and nonfinite count were all
  zero. This invalidates “soft-band Done is still the whole reset storm” and leaves the run
  ineligible for Reward/curriculum comparison.
- Those events happen before useful PPO learning and are concentrated on the left ankle pitch lower
  edge, waist pitch upper edge and right ankle roll upper edge. The reference clip and
  `q+20 ms*qdot` remain away from the hard boundary, and the fresh actor is exact-ready bias plus
  `0.02` exploration. Therefore reward tuning, q_des margin and PPO optimization are not the next
  lever.
- The initial “current upper qpos is ungrounded” diagnosis was too broad. `canonical_ready_v1`
  itself is an uncertified donor with an old zero-foot-contact result, but the actual two upper
  fivebind NPZs already carry constant 12-leg positions equal to the existing AgiBot A3
  grounded-ready candidate. That candidate has SHA-256
  `585bbd7d643857abd08108eac7b4dd997b228d0df1a9921334ca845cd931d71e`, receipt file SHA-256
  `ee7dea1aec81169e1d002bbe0b2cfa75c793a97a3f89e1e740d0064dc8be7c46` and binds exact model
  `A3T2.5_pingpong_0519`; `candidate_id=G1` is only the A3 construction-candidate label, not a G1
  robot. Exact A3 MuJoCo replay of both upper frame-0 states yields two feet / six contact points,
  zero unsupported/self collision and feasible static-ground dynamics.
- The actual schema defect is narrower: all 12 leg `joint_pos` columns are constant, while their
  `joint_vel` columns contain stale nonzero samples. A Pod qvel-only prototype zeroed those columns
  and rebuilt schema-2 body FK/velocity; all-frame `right_racket` position/orientation/linear/angular
  velocity stayed bitwise identical, as did frame count and strike frame. The next artifact repair
  must therefore leave every qpos/root/timing byte unchanged and only fix the inconsistent leg qd
  plus derived schema-2 channels. Full-body clips still require a complete `ready→core→ready`
  recompile.
- This is an A3 motion-schema correctness repair, so it does not require a scientific old/new learning
  A/B. It does require a no-clobber Pod `1 env × 2 update` construction smoke and a fresh 4096-env
  five-update behavior test. Promotion requires episode length to cross `t_hit`, nonzero strike
  opportunity, finite exact checkpoint, and no increase in raw-hard/table/fall/nonfinite rates.
- Independently, behavior-equivalent hot-path changes may be accepted by numerical/state parity and
  profiler rather than learning A/B: immutable receipt SHA cache external to dataclass state,
  one-shot same-step strike-timing handoff, batched `10+8×N` scalar
  [device-to-host transfer (D2H)](../DEFINITIONS.md#device-to-host-transfer) with unchanged error
  reductions, exact float32 counts for supported `N<=8192`, unchanged per-step Python EMA and
  historic accumulator-update order, and removal of the state-preserving `fired_valid.any()` host branch. No result
  is claimed until focused Pod tests and a fixed-workload timing sample pass.

Follow-up note (2026-07-29, ActionBall actual-joint reset follow-up; Gate remains `Partial`):

- `8d2a1bcd` 已在 Pod1 真实 Isaac 把旧 scalar actual reason 拆到 joint×side。1-env 两轮均为
  `left_ankle_pitch_joint` 下侧、episode age `17/19`；4096-env updates 0--2 的 exact event
  分母为 `3,187/4,457/5,087`，与旧 `joint_actual_forbidden` reason 对账。左脚踝 pitch 下侧
  分别占 `2,304/2,416/3,112`，其余主要是腰 pitch 上侧与右脚踝 roll 上侧；没有 age<=1
  事件。只有少数行带 substep raw-hard overlap，多数只是当前 q 进入 2%-inner band。
- 因此旧 DoneTerm 的语义反了：recoverable soft-band occupancy 被立刻 reset，PPO 看不到恢复
  transition；而现役 actual-q barrier 已从 soft envelope 内侧 `8%` 开始以 weight=`-40`
  持续收费。下一候选让 2%-inner 继续进入 barrier 和 diagnostic，但不 Done；nonfinite、
  current/substep raw mechanical hard edge 仍 hard reset，table/fall 与 q_des projection
  不变。这符合“软限位不能蹭、硬限位仍终止”，不是删除关节安全。
- telemetry schema v2 将 `total_safety_event_count` 与
  `total_hard_terminal_count` 分开；后者必须与 fresh run 的
  `joint_actual_forbidden` reason 对账。当前只有 source candidate，尚未取得该新语义的 Pod
  smoke/4096 结果，Gate 继续 `Partial`。
- `478f485b` 的 finite executed q_des 额外内缩 `5%` 也没有降低 mass reset。Pod1
  1-env × 2-update smoke 自然完成；4096-env updates 0--6 的
  `joint_actual_forbidden=3,187--5,087/update`、mean episode `18.67--23.74`，strike
  opportunity 始终为零，而 q_des termination、nonfinite 和 projection penalty 均为零。
  graceful boundary checkpoint 为 `model_6.pt`，run 随后停止，不能当作训练结果。
- 用 `configs/a3_runtime_articulation_joint_order.txt` 的真实 articulation order 重算后，
  `bh_loop_c_upper_fivebind` 老师全片以及 `q+0.02*qdot` 都没有进入当前 hard-limit 内缩
  `2%` 带；frame 0 也合法。故“老师 bytes 本身贴限”已被排除，但当前日志把 31 个关节和上下侧
  OR 成一个 bit，尚不能区分 shared-ready/ground transient、PD/接触动态或某个关节的 inner-band
  漂移。
- 该次短诊断只增加 device-side joint×side×episode-age 计数；rollout 内不得 `.item()`、
  `.cpu()` 或 JSON，PPO update 边界只允许一次小批量 D2H 并输出
  `HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON`。它不是新的发射门，也不改变 Reward、Done、
  action、physics 或 curriculum。先用 1-env smoke 验证构造，再用 fresh 4096-env 一轮定位；
  在此之前不得继续改 safety-band 宽度或把 actual-limit Done 删除。
- `5e94f21b` 的 20 ms receding-horizon brake 已在 Pod1 跑到 4096-env update 16。相对同 seed
  `5dbb`，actual-joint reset 没降（`4,791.6` vs `4,728.8/update`），mean episode 仍只有
  `20.19` steps、短于约 31-step 的击球帧，strike opportunity 合计仅 `2`；最近十窗吞吐反而约慢
  `4%`。它是行为反例，不晋级。
- 同窗 reference-only breach 约 `43.3/98,304=0.044%` transitions/update，只相当于当前
  reset 数的约 `0.9%`。所以 `reference_guard_mode=metrics_only` 不是本轮 reset storm 的解释；
  但官方 BeyondMimic/DeepMimic 也不支持把它未经 A/B 设为最终默认。reference error 不可像
  q_des 一样投影；后续只比较 hard/metrics/hybrid termination，物理 fall/table/actual-limit
  始终 hard。
- 下一候选将 finite ActionBall execution envelope 在现有 soft limits 内每侧额外保留
  [`5%`](../DEFINITIONS.md#finite-projection-soft-inset)，raw action/log-prob、actual hard
  `2%` termination band、Reward 权重和非 ActionBall 路径不变。四条 N1 teacher 轨迹均在新包络
  内，最小余量约 `0.046 rad`；比例已进入 runtime/schema-3 training contract。当前仍只有
  source-level 证据，必须 fresh clean Pod smoke/canary 后才能判断 reset 是否下降。

Follow-up note (2026-07-29, ground-plant handoff; Gate remains `Partial`):

- 旧 `TerrainImporterCfg(terrain_type="generator")` 会把 `env_origins` 移到 terrain tile，
  而克隆桌体仍在 GridCloner 网格，因而 rough arm 实际把机器人与自己的桌子拆开。候选修复改为
  每环境 `robot_side_zero_mean_patch`：机器人侧围绕 z=0 起伏、桌近沿起整块严格为平面；另以
  `robot_material_make_consistent=true` 显式约束逐桶 `dynamic≤static`。相关 schema-3 plant
  指纹和 host 联合回归为 `379 passed`。
- 这只达到 E1。尚缺真实 Isaac 的 2-env clone/contact isolation、兜底地板 drop、桌 footprint
  raycast、seed/mesh/pickle 内容绑定、ready 脚初始穿插检查与 4096-env VRAM/吞吐。故首轮 fresh
  N5 固定为平地 upper/no-move；corrected-friction 与 rough/move 均为独立后续实验，不能通过自由
  Hydra override 混进 smoke→canary→long 的同一 scientific recipe。

Follow-up note (2026-07-29, live Reward causality prelaunch; Gate remains `Partial`):

- 新增 1-env formal Isaac 审计器：从真实 post-Hydra `RewardManager` 重建并逐字闭合 effective
  recipe，再按 MJLab 平衡稳定、BeyondMimic 模仿、HOPE 击球/上台、immutable safety 四组，对每个
  active objective 做权威输入 tensor 的单轴 worsening；生产 callable 的
  `weight × raw × policy_dt` 不严格下降、没有复核 mutation 或运行异常都 fail closed。
- root quaternion/world velocity、motion hold gate 等派生 property 不再被当作可写状态；审计修改
  权威源并显式读回生产 getter。clean 门包含 untracked，且 producer 必须是 HEAD-tracked exact
  blob；同时复用训练 physical-validity source guard、manifest/motion/policy-contract identity
  与 Isaac 异常非零退出模式。
- host focused `15 passed`、Reward 相关联合回归 `433 passed`，只证明源码/receipt 合同。
  clean commit 的 Pod 真实 1-env receipt 尚未
  生成，故不能写成四组 Reward 已通过 E2，也不授权 canary/长训。操作与证据边界见
  [Reward 因果审计工序](../operations/run_action_ball_reward_causal_prelaunch.md)和
  [实验卷宗](../experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)。

Follow-up note (2026-07-28, ActionBall table safety chain; Gate remains `Partial`):

- ActionBall 的解析球场景改用
  [`ActionBall table safety assembly`](../DEFINITIONS.md#action-ball-table-safety)：真实桌板、
  floor→slab-underside 保守 robot keep-out、球网与左右网柱，五件各自是明确 cuboid collider；
  tracked USD 只供显示，因为其 whole-mesh convex hull 不能代表真实桌腿/网几何。keep-out 不与桌板
  体积重叠，也不宣称是腿模型；physical/shadow 动力学球组合在 cfg mutation 前 fail closed。
- `robot_hit_table` 的 ActionBall 路径不再用 broad force + body-origin AABB 归因；它按 runtime
  articulation order 配置 exact 32 个 one-body pair-filter sensors，每个都过滤五件桌体。双脚也在
  内；fixed-merge 的拍面/拍柄由 `right_wrist_yaw_Link` 覆盖。ActionBall action term 在
  decimation=4 的每个 5 ms physics substep 读取全部 sensor，episode-sticky OR；首帧以前一 policy
  sensor timestamp 为 baseline，重复/跳步/缺失/错序/陈旧读数 fail closed。只在指定 reset env 清
  latch，不能用新 policy step 擦掉证据。接触阈值为 `1e-6 N` 数值零容差，不允许轻蹭套利。
- host focused 命令与输出见
  [桌体安全 smoke 工序](../operations/run_action_ball_table_safety_smoke.md)：当前
  `83 passed in 2.19s`，覆盖 32-body exact order、top/edge/keep-out/net/posts、五个 filter
  index、四个单子步 pulse、compound reason、partial reset、2000 次 float32 5 ms 累加与重复/
  漏步负例。
- 这仍是 E1 source/host 证据。本轮没有启动 Isaac 或 Pod；真实五件 CollisionAPI、四子步真实
  32-body actor pulse、raw termination/generic terminal 各一次、reset 零泄漏和 4096-env
  throughput/memory 均待跑。teacher 还必须另过整个 prep→hit→recovery、all robot/racket geoms
  对桌面/边缘/桌底/keep-out/网/网柱连续至少 `5 mm` 的 swept-clearance 门；runtime sensor 不能
  替代这张证书。
  在这些命令留下结果前不得启动 ActionBall canary/长训，G05 不晋级。旧
  `motion_backhand_loop_b_table_net_clearance_prereg_20260715.json` 绑定的是修改前 source bytes，
  不能认证当前 scene builder；正式引用须新 prereg/receipt。

Follow-up note (2026-07-28, three-backhand upper baseline safety cutover; Gate remains `Partial`):

- 新候选是
  [`upper N3 safe warm-start`](../DEFINITIONS.md#upper-n3-safe)：从四动作
  `model_10809` 的同形 `175 → 31` PPO 状态 non-exact 热启动，只保留
  `bh_loop_c / bh_block / s0_highpress`。N3 题库 strict schema-3 SHA-256 为
  `6d61fda0...df22`，父 checkpoint/contract 为 `74d48177...ff23 / 75f46c92...9bc`。
  Pod1 physical GPU1 独占该候选，不与 73/93、MuJoCo 或视频混卡；专卡只解决资源分时，不是
  行为晋级。
- 父本最后 100 窗三动作 rally return mean 分别为 `44.394% / 46.907% / 37.369%`，
  legal/strike 为 `53.57% / 54.99% / 42.47%`；但 near-limit mean `25.916%`、physical fall
  `440/85646`，且没有 raw q_des/substep hard-gap 账，不能直接部署或继续照旧长训。
- 新 `HOPEPingPongUpperSafe` 叶子保持 VirtualBall 175D/static-bank 与
  `4.0/0.5/0.5 + virtual_landing 1648.8` 首跑配方；两个 joint DoneTerm 只在 physical
  `joint_pos_limits` hard edge 终止，processed q_des 仍走 soft clamp、qbar `-0.65/0.08` 和
  joint-limit `-10`。所有 termination 共用一次 `death=-3600`，table/joint specific 为零。
- runner 已新增 protected-task 两阶段消费者：optimizer 前 zero-copy freeze，并在 simulator device
  深校验/稀疏化逐 policy-step joint-safety ledger；prepared sidecar 经 file/parent-directory
  `fsync` 后才允许 optimizer，optimizer commit marker durable 后才 exact-token ack。actual hard
  edge/non-finite q 先写 fatal sidecar再阻止 optimizer；公开 destructive one-shot consume 已禁用。
  缺 4+1 readback、identity/action term 漂移、持久化/容量/预算失败均拒绝继续。
- 当前只有 host/source 候选：N3 bank/launcher/safe-leaf AST tests
  `14 passed, 1 skipped`；joint safety focused `70 passed`，4096×24×31 host prepared sidecar
  `956,704 B`。旧 `8ba15f38` probe 在 Hydra
  compose 前被拒绝，没有 Kit/run/checkpoint，只作 infrastructure-invalid 证据。必须先形成 clean
  commit，再在 GPU1 完成 `1 env × 2 update`、NaN/extreme pre-physics、两个 joint reason 和
  physics-substep hard-limit readback；这些证据缺一都不允许小 canary 或长跑。完整操作和证据见
  [实验](../experiments/2026-07/EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md)与
  [工序](../operations/run_upper_n3_backhand_safe.md)。无真机授权。

Follow-up note (2026-07-27, action-conditioned ball-first/N-action prelaunch source candidate; Gate remains
`Partial`):

- 新候选采用[按动作条件化的 Ball-first](../interfaces/action_conditioned_ball_first_contract.md)：
  balanced schedule 先冻结 action，再按该动作 profile 采 time-to-contact/incoming
  ball/base/landing aim，最后由 fixed-action solver 解 task 与认证 teacher rate；训练期不运行
  selector。`task-first` 保留为历史消融，不再是候选 executor。
- `level=0` 是 manifest 的 non-zero initial std；schema v3 把所有 lower/upper 和方向 tangent
  正负侧拆成 32 个有序 arm（`no_move` 有效 28 个）。各动作先找 per-arm marginal frontier，再用
  joint `rho` 把 **safe closed policy non-return** 控制到 10% 目标带。rolling-100 只排下一个
  canary 候选，不能批准扩域。solver reject、
  install/start/close 缺口、table/fall/collision 和 infrastructure invalid 分账；所有动作的 table hit
  都是零容忍安全门，新正手还须单独完成 `0/-5/-10 cm` 站位对照与整轨 clearance 证书。
- 动作身份不再只靠正反手 sign。manifest 文件 SHA、stable action UID、motion bytes/order、
  per-action RNG、single-use birth/task receipt、课程状态、balanced sampler、effective Reward receipt
  与 checkpoint 恢复都必须精确对账。冻结 eval window 另绑定 policy contract、checkpoint SHA 与
  monotonic generation；在线训练 rollout 不能冒充晋级证据。
- 首轮五动作 training view 候选排除旧 `fh_loop`，使用
  `bh_loop_c, fh_block_syn, bh_block, s0_highpress, fh_loop_high`。旧正手只退出新 view，历史
  bytes 保留。新正手目前缺 upper/full 正式输出、grounded collocation trace、动作特定
  `t_hit/t_cycle`、physical racket-site speed、无撞桌证书和 Isaac filtered-contact smoke；
  `training_authorized=false`。
- 新正手 `0/-5/-10 cm` 站位只作 upper/full 对照，取共同过门的最近档。当前 certifier 只产未授权
  reference checks，不能授权 simulator 或训练。
- Reward 审计证明名义 v2 quality 权重 `393.4/295.1/229.5` 没有进入现役 composed task；task YAML
  的 `4/0.5/0.5` 后写覆盖。当前“学得好”的最强解释是 task 可行、自洽、低熵，而不是高权重调对；
  必须把 producer-order A/B 与同 task 分布的 Reward A/B 分开。以后每个 run 用
  [effective Reward recipe](../DEFINITIONS.md#effective-reward-recipe) receipt 绑定实际 callable/
  weight/params。
- 训练后的动作选择另属
  [capability selector](../interfaces/action_capability_selector_contract.md)：hard safety →
  support/OOD → calibrated LCB → `delta_tie` 内 priority → abstain。当前 production planner/
  schema-4/C++ 仍是二动作；pure core 不能冒充接线完成。
- 目前最高证据仍是 E1 source/host contract。schema v3 的局部中间验收为
  sampler+manifest+adapter `184 passed`，加 curriculum/evaluation 后 domain-core 为
  `210 passed in 8.35s`。N93/E4096 单轮 sampler state 为 `160,906 B`，但 4096 个
  birth/sample pair 已到 `6,070,936 B`，100 轮约 `607 MB`；跨组件退休前缀 segment
  compaction 与 external resume pin 尚未闭合。旧 `254 passed`
  属于已废弃的对称 7-axis schema。这些都不包含最终
  train/runner/commands/Isaac union。红队仍在收口 compact lifecycle ledger、solver exact replay、
  Racket→Motion time/rate 驱动和 frozen evaluator authority，故局部 pass count 不构成 launch
  authorization。
  code-rooted arbitrary-N motion admission、N5/N93 ordered assets、完整 strict-resume 与 Pod Isaac
  smoke 也尚未闭合。Pod3 exact `out_refined` 路径不存在；可见的是 74 组 action/ball sidecar
  配对与 108 个击球 metadata，不是 exact ordered 93 件。2026-07-27 只读资源快照显示
  Pod1/Pod2 六卡均占用；没有 checkpoint 或新训练
  成绩。资源占用不是永久归属或行为失败。G05 因而保持 `Partial`。

详细假设与停止条件见
[action-conditioned ball-first 实验](../experiments/2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md)、
[Reward 因果审计](../experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)和
[selector 实验](../experiments/2026-07/EXP-ACTION-CAPABILITY-SELECTOR-20260727.md)。

Follow-up note (2026-07-20, W/Y export mechanics pass but exact lineage blocks promotion; Gate remains `Partial`):

- Both real W/Y zero-write `--plan` executions passed against exact `origin/main@a0c1284`: checkpoint iteration
  `6700`, actor dimensions `179→31`, all formal/material/train-bank checks true,
  `artifact_written=false`, and `graph_export_not_executed=true`.
- Fresh diagnostic ONNX artifacts were then generated. W has SHA-256
  `ee0e2e83c8f3dc8302fcef609fe13b2feaf69e247e39f405d1ea6c30b652d970`; Y has SHA-256
  `72da43d96ab9dd95e1da6aba2ed548ad26e61863b70cf8120c120132b7b8f995`. Both are `179→31`, contain
  `94` metadata keys, and pass an independent checker plus finite CPU ONNX Runtime inference.
- Both source checkpoints report `training_contract_lineage_exact=0`, and both ONNX artifacts report
  `training_contract_exact=0`. They are diagnostic only and must be rejected by the production runner. The
  local artifact blocker is exact-lineage remediation; global execution order remains solely in
  [`docs/NOW.md`](../NOW.md). The vendor adapter and behavior paper remain downstream.
- A 2026-07-20 NVML audit found no compute process and zero allocated memory on all three GPUs of both Pods.
  This is an availability snapshot, not permanent GPU ownership and not proof that non-GPU stale processes are
  absent.
- The proposed W/V processed-qdes action-slew matrix is Wave A: a default-off, single-swing diagnostic, not the
  complete stability program. Wave B's M0 moving-teacher input is rejected at stance `0/4`; the remaining design
  space is an upper-only matched control versus static-v4rg lower-body imitation or a non-demo stability
  constraint. Wave B exact flags and matrix remain under audit and must not be guessed. Neither wave can promote a
  policy or replace the ordered continuous-recovery gates `T0 → T1 → T2`.
- The Wave A probe2 W-C cell naturally exited at `model_6701.pt` with finite checkpoint and
  complete two-update ledgers. The outer verifier rejected step 6700 only because it required a positive recovery
  denominator inside the first 0.48 s rollout, although the earliest possible eligibility is 0.56 s; step 6701 had
  `31459` eligible samples. Probe3 then naturally completed W-C/W-N/V-C, but its verifier only required denominators
  divisible by 4096 and did not bind the frozen 24 steps/env; the old W-C receipt and all three runtimes cannot unlock
  training. Probe4 bound the exact `98304` ledger but its only W-C attempt used the nonexistent Hydra key
  `algo.num_steps_per_env`; the real compose guard rejected it before run-directory creation or Kit launch, leaving
  no process, checkpoint, log, receipt, or GPU allocation.
- Probe5 (the fifth non-scientific two-update contract-probe attempt) corrected the override to
  [`algo.runner.num_steps_per_env=24`](../DEFINITIONS.md#ppo-num-steps-per-env) and produced exact receipts for
  W-C, V-C, W-N, V-N and V-H under its v4 manifest (file/content SHA-256
  `6bfa73587968f8f0af71b5617e8c324f75114b304bbe1452d0b0e4617d1f51bc` /
  `093a4cc7a0ce91aad74948ed39b581e5f4a0693ba114f3144062b6cc4386a462`). W-H failed closed before trainer
  binding because the old single-read supervisor sampled the `Popen` child during its fork-to-exec transition and
  compared that transient argv with the final trainer argv. It produced no first iteration, trainer binding, RSL
  checkpoint, receipt or GPU training compute; exact post-failure checks found its leader group and child absent.
  This is a supervisor-evidence race, not a negative result for the H mechanism. The five v4 receipts remain
  immutable history and cannot be mixed with a later manifest.
- Probe6 (the sixth non-scientific contract-probe attempt) launched only W-C. At
  `2026-07-20T03:01:10Z` its generated supervisor Python exited with
  `IndentationError: unexpected indent` before the locked launcher child could bind. The failure-transaction helper
  had prefixed two spaces after every newline in an already shlex-quoted multiline program, changing the program
  bytes. There was no leader/child evidence, binding, terminal marker, checkpoint or receipt. Exact stable
  double-scans found PID=PGID `2712318` absent; Pod1 GPU0 was empty, the Kit/cache locks were free, and the audit sent
  no signal. The other five cells were not launched. This is a launcher-generation defect, not a mechanism negative.
  Its v5 no-clobber root
  `/workspace/codexschema/phase1_balance_action_slew_v5_20260720` and manifest file/content SHA-256
  `718bbee0a556cc3640ee636e20f8eb2adb293cee8d0bb1820afceebf5ce1a267` /
  `3cbb019e9d315abdc687d1635c9deffc13eae9577ee2d820b0e3c30ba9b7cfd8` are immutable history and cannot be retried.
- Probe7 (the seventh non-scientific contract-probe attempt) produced natural exits and exact verifier receipts for
  W-C, V-C and V-N. W-N reached RSL at `2026-07-20T03:22:50Z` and then froze at `Starting the simulation`, before
  any learning iteration, trainer binding, terminal checkpoint or receipt. Its locked 180 s watchdog used exact
  TERM/KILL and returned rc `125`; subsequent checks closed the exact groups, assigned GPU and Kit/cache locks.
  V-N had already been launched six seconds before the W-N failure became known and later verified successfully;
  W-H and V-H were never launched. This is transient launch infrastructure evidence, not a negative result for
  `action_rate=0` or either parent policy/mechanism. The v6 root
  `/workspace/codexschema/phase1_balance_action_slew_v6_20260720`, its three receipts, and its config/runner and
  manifest file/content SHA-256 values
  `912bd8d212791d99ce6a6851a8f05c12d182cdfa9d5566e02381f1b4703b8f3c` /
  `3fbaf23f97fdb40e05a448f9f769267b21c7cca3bd767aa082c0d5b965ecd7d7` /
  `4552fe23abd551d8959a9de05cc5f9d761d0da25eed88138d61fa45cc6558e9e` /
  `6e3518d97d48fad550e7971a5178b1f11c15895696f03d30d5a62d1e27741640` are immutable history: no retry or mixing.
- Probe8 (the eighth non-scientific contract-probe attempt) launched only W-N on Pod1 GPU1 from fresh v7. It ran
  from `2026-07-20T03:44:19.857Z` to `03:44:32.112Z`; after all 4096 environments were created, `sim.reset`
  aborted with `malloc(): invalid size (unsorted)`. The trainer returned `-6`/`SIGABRT` and the outer transaction
  returned `134`, before any first learning iteration, binding, RSL run directory, checkpoint or receipt. The
  manifest/claim/supervisor-spec/log/launch/leader/terminal/child-evidence SHA-256 values are respectively
  `887c0b9e…a7231`, `8472ecf9…fa8b`, `334ed262…c23c`, `b3437c87…f49c`, `edc4782f…ce8`,
  `b7f981c8…8815`, `ad5c46a7…6268` and `c2d6c31b…26f`. The other five cells were not launched; exact
  process groups, assigned GPUs and Kit/cache locks closed on both Pods. Because the abort preceded managers and
  RewardManager construction, this is infrastructure-only evidence, not an N-mechanism or Reward result. The v7
  root and manifest are immutable: no retry, completion or cross-generation receipt mixing is allowed.
- Probe9 (the ninth non-scientific contract-probe attempt) completed from `2026-07-20T04:14:32Z` through
  `04:22:42Z`.
  Its fresh no-clobber root is `/workspace/codexschema/phase1_balance_action_slew_v8_20260720`, with run names
  `phase1_balance_slew_probe9_{w_n,w_c,w_h,v_c,v_n,v_h}_seed3_20260720`. Config/runner SHA-256 values are
  `c7ec75a9917b8bdcf7976186633b021c69fb82e591898dad6b8d5c93cfdb37d5` /
  `24b5f7831ad49c2b88266fed65c37e6e4bcdddaacab28ffa917fce66ef918db1`; manifest content/file SHA-256 values are
  `97c36e471fb8fc6b93fe212f20846de6697db518192e7a45c6618e5924947e28` /
  `688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353`. Its first two cells form a crossover gate
  inside the same full six-cell identity. The runtime followed the frozen order exactly: W-N on Pod1 GPU0 exited
  naturally, published its exact receipt and closed process/GPU/locks before W-C launched on Pod1 GPU1; after the
  same W-C closure, W-H, V-C, V-N and V-H ran globally serially. All six returned natural exit `0`,
  `normal_exit=true`, crossed the first iteration, passed the exact verifier and closed before the next cell.
  Receipt file/content SHA-256 values are W-N `ee8c5378…8c5ff` / `afce94d7…80f9`, W-C
  `b948a4d8…18d5` / `e5daf19c…d3f0`, W-H `a80502c9…8111` / `32abf562…7e68`, V-C
  `c3db6c38…edc1` / `ae30d1b5…3446`, V-N `06919a60…4bc7` / `bc99acd0…979c`, and V-H
  `b7a24015…1ec2` / `0905ad8f…32a7`; their common verifier SHA-256 is `d736a205…0ebc`.
- The six copied receipts passed local validation under the same manifest, with receipt-set SHA-256
  `cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`. The crossover has a narrow
  interpretation boundary: the two passes exclude only deterministic failure of W-N on GPU0 and W-C on GPU1.
  They neither erase probe7/8 nor prove GPU equivalence or any action-slew benefit. This remains a non-scientific
  qualification probe.
- The verified receipt set first produced a command-generation-only artifact at `origin/main=16263be5`, SHA-256
  `fc6f1ea38a5a823016d83675d56fc41b50b70dbde1bba60602b26d6c743802df`; it never executed SSH. A fresh v8
  scientific command artifact rendered under `origin/main=d5c08bb91728edfa75801a630531527aeb2ae06c` has
  SHA-256 `be346f94cf6bf738da36804bf59f6a60bc5249f3c6bf5474abf617358db4b42a`. Only W-N on Pod1 GPU0 launched,
  from `2026-07-20T04:45:02Z` through `04:48:25.971Z`. It published an exact binding, then stopped writing its
  log at `04:45:15.781Z` while still in `sim.reset`; the locked 180 s watchdog closed the exact process group with
  terminal kind `stale_timeout`, rc `125`. There was no first learning iteration, RSL run directory, checkpoint or
  Reward result, and the other five cells did not launch.
- The outer rc `121` was a second infrastructure defect, not a trainer result: the old failure audit required
  probe-only `trainer_child_evidence.json` even at train stage, where the trainer itself is the group leader. Its
  bad read overrode rc `125`; stable manual checks then proved zero exact-PGID members, leader, GPU compute and
  Kit/cache-lock holders. The immutable result is
  [`phase1_balance_action_slew_train_v8_attempt1_result_20260720.json`](../../configs/phase1_balance_action_slew_train_v8_attempt1_result_20260720.json),
  SHA-256 `ac09b70a1df89a501165504f4c07158858687127172a8d9d5a6bdf1473e61a75`; launch-spec/leader/run-log/
  launch-ledger/run-binding-content SHA-256 values are respectively `0604c6ea…96ca`, `a4c6b8fd…a2c4`,
  `a2db80ad…0ff2`, `5aca7d47…687c` and `0cb8fb07…d497`. v8 is frozen and cannot be retried or completed. This is
  infrastructure-only evidence, not an N-mechanism or action-slew result.
- Fresh v9 pre-registers a stage-aware failure audit: probe still requires and fully validates child evidence;
  train requires probe-only child/identity evidence to be absent before accepting exact group closure. Its root is
  `/workspace/codexschema/phase1_balance_action_slew_v9_20260720`; config/runner SHA-256 values are
  `3bf5085ea8396513d162b9cce249dfb761b39b2827ec722959343c953683e59e` /
  `0fff4515cbe7e62798e8c39f701851c46e68287c7321e7618161fa9dde4789ce`, and manifest content/file SHA-256 values
  are `36ceb3c77dc056f4565378a92b03da58865378d86c5849085ba066631cea456c` /
  `664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a`. Probe10 preserves the reviewed
  W-N/GPU0 → W-C/GPU1 → W-H → V-C → V-N → V-H order. It is not launched; all six fresh same-identity receipts
  must pass local revalidation before any `science_retry2` train command may be generated. G05 remains `Partial`,
  and the mechanism decision remains inconclusive/not adopted.
- Reproducible probe10 command-render gate (it emits JSON but does not SSH or start a trainer):

  ```bash
  BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
  cd "$BALANCE_REPO_ROOT"
  BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
  BALANCE_LAUNCH_MANIFEST_SHA256="664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a"
  git fetch origin main
  test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
  git cat-file -e origin/main:configs/phase1_balance_action_slew_launch_manifest_20260720.json
  python3 scripts/run_phase1_balance_action_slew_queue.py \
    --stage probe --authorize-launch \
    --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
    --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
    > /tmp/phase1_balance_slew_probe10_commands.json
  ```

  Inputs are a clean current-main authority checkout, the fresh v9 manifest above and its manifest-bound remote
  assets and parents. The output is the six-job probe10 command plan only. No probe10 receipt exists yet, so train
  command rendering and execution remain blocked. No long-run mechanism result has been accepted, and G05 remains
  `Partial`.

Follow-up note (2026-07-20, recent Jiayi/Yikang branches audited; no branch-wide merge or behavior promotion):

- Jiayi's V13 branch contains a bundled post-swing balance/recovery proposal but no checkpoint or behavior
  evidence. Its history also includes destructive/unrelated changes, so the branch is not mergeable as a unit;
  any useful idea must be reimplemented selectively on current `main` with its own tests and paper.
- Yikang's V9 force implementation called the external-force API without `position_data`, so the advertised
  pelvis-COM force was applied at the link origin. The V9 runs are mechanics/throughput evidence only and yield
  no accepted balance conclusion. V10 finished iteration `9999` but has no MuJoCo/Gate3 behavior result. V11
  fast stopped at iteration `2816`, prestrike stopped at `18274`; the available proxy checkpoints do not close
  a formal behavior Gate. None of these branches should be merged wholesale.

Follow-up note (2026-07-20, S0/M0 exact-GMR evidence recovered; no training authorization):

- Pod1 report-last manifests prove both v2 batches reached `complete_exact_gmr_diagnostic`; the later Pod2 rc127
  remains a separate failed location rather than global absence. S0's 88-frame output is finite/30 Hz/31 DoF but
  ball contact/effectiveness remain null, so it needs an independent high-ball paper.
- All four M0 moving outputs pass finite/30 Hz/31-DoF structure but the frozen stance gate is `0/4`; M0 is
  input-gate rejected and must not consume an RL GPU. Formal/schema2/training/hardware are false for both batches.
  G05 remains `Partial`; exact hashes and per-row failures are in
  [the motion experiment](../experiments/motion_exact_gmr_s0_m0_20260713.md).
Follow-up note (2026-07-20, lower-body Wave-B source/queue preregistered; Gate remains `Partial`):

- Inputs are bound, but not yet runtime-consumed: exact source `5db7366a...e39e`；W/V `model_6700` parent
  checkpoints and adjacent hard contracts；static `v4rg_runtime_order_v3` forehand/backhand；schema-3 train bank；
  ignored A3 runtime tree；six-file preconverted USD bundle；and the reviewed queue manifest.
- Implemented: default-off twelve-leg soft pose imitation and reference-free signed-stance/qdot bundle；every
  explicit Wave-B B0/B1/B2 cell declares both weights, enables both weight-independent ledgers, and writes both
  schema-3 hard-contract blocks. The common inclusive gate is pre-strike `0.30 s` or same-attempt post-strike
  `0.40 s`, without success conditioning.
- Queue discipline: W/V×B0/B1/B2 uses six unique GPU slots, `4096 env × 24 steps × 2 update` probe namespaces, natural
  exit and full policy/value/optimizer/two-normalizer checkpoint validation. Long SSH argv cannot be generated
  until all six canonical receipts verify against one manifest; each update must contain exactly `98304` observed
  samples. Authorized rendering also requires HEAD/bytes and the owner/executor/two-branch/queue-id claim in one
  latest `origin/main` NOW entry; default invocation emits no SSH command.
- Validation on this branch: Wave-B source/reward/schema-3/queue focused suites jointly report
  `269 passed` (`35 + 124 + 68 + 42`). Queue compile and default no-launch plan pass；checked-in
  config/runner/manifest digests bind exactly；12 probe/train remote bodies and 6 verifier bodies pass `bash -n`.
- Outputs pending: six RunPod full-scene probe receipts, then six long science claims/bindings/checkpoints and
  matched behavior metrics. There is no Isaac full-scene, PPO behavior, vendor MuJoCo or real-robot evidence；
  descendants intentionally retain `training_contract_lineage_exact=0`. M0 moving teachers are 0/4 at their
  separate stance gate and are absent from this queue. G05 therefore remains `Partial`.

Follow-up note (2026-07-19, W/Y static export inputs closed; Gate remains `Partial`):

- One read-only, exact full-filesystem search located exactly one W and one Y `model_6700.pt`. Both load on CPU
  with embedded iteration `6700`, `74` floating tensors / `1,762,715` floating elements / zero non-finite
  elements, an actor shape of `179→31`, and all four expected `params` materials present:
  `training_contract.json`, `env.pkl`, `agent.pkl`, and `env.yaml`.
- `standalone_onnx_export.py` now has a genuinely zero-write `--plan` source path. It uses a weights-only
  checkpoint load, requires a non-negative integer `checkpoint_iteration`, validates finite actor/normalizer,
  donor, motions, harvest, train bank, contract and the formal face-179 envelope, then exits before the first
  output-side effect. Its JSON explicitly reports `artifact_written=false` and
  `graph_export_not_executed=true`. The focused local suite is `97 passed in 0.38s`, including the unchanged
  normal-export fake smoke.
- Neither real W/Y plan has run on a Pod, no ONNX has been produced, and no vendor scene or policy behavior has
  run. These are static artifact/source gates only; G05 remains `Partial`.

Follow-up note (2026-07-19 about 10:26 CST, Isaac pool closed; Gate remains `Partial`):

- One read-only Pod1 SSH confirmed that L2 PGID `2457829` has zero members, no NVML compute application, and
  zero utilization and memory use on GPU0/1/2. L2 is therefore fully absent at both process-group and compute
  levels rather than `UNKNOWN`.
- Pod1 V/L2/Z3 and Pod2 D2/F are now all closed, so the two-Pod Isaac pool has ended. V/L2/D2/F remain terminal
  teardowns rather than natural terminals. W/Y advance as the vendor MuJoCo same-exam demo pair, with U retained
  as the stable reserve. Deployment behavior is still unverified, so G05 remains `Partial`.

Follow-up note (2026-07-18 20:52 CST, four terminal teardowns closed; Gate remains `Partial`):

- The four-arm audit now matches a NUL-complete `run_name` token and scans semantic fatal conditions over full
  log lines. V, L2, D2, and F each passed trainer identity, unique-log, last-iteration `6700`, ten-second
  stability, fatal-zero, and recipe-fingerprint checks before any action.
- Pod1 V was handled only through its numeric process group with exact TERM then KILL, and the group is absent.
  L2 received the same exact-group teardown and is NVML-absent; all Pod1 GPUs ended at zero memory and zero
  utilization. One `/proc` group member remained after the short wait, so L2's final state is `UNKNOWN`; future
  work may only verify absent/zombie read-only and must never signal it again.
- Pod2 D2 and F each received exact TERM then KILL; both groups ended with zero members and no NVML process.
  GPU memory was zero on all cards. GPU2's instantaneous 51% utilization had no NVML process and is not evidence
  of a live trainer. These four closures are terminal teardowns, not natural terminals; existing model/result
  evidence is unchanged. W/Y still proceed to the same vendor MuJoCo exam, and G05 remains `Partial`.

Follow-up note (2026-07-18 about 19:48 CST, complete `+1000` comparison; Gate remains `Partial`):

- All five new cells have complete ledgers for both updates `5701–6700` and the recent `6201–6700` window:
  missing and duplicate updates are zero in every cell and window. Cumulative completion/legal-return/fall rates
  were U `93.31/31.94/0.87%`, V `48.40/70.14/22.05%`, W `94.14/31.52/0.74%`, X
  `49.68/68.08/22.00%`, and Y `93.57/32.31/0.82%`. Recent-500 rates were U
  `94.99/32.46/0.30%`, V `48.14/79.06/23.40%`, W `95.28/32.27/0.24%`, X
  `48.49/77.33/23.61%`, and Y `95.18/33.10/0.31%`.
- In the recent `<0.5 s` bucket, completion/legal-return rates were U `96.42/26.15%`, V
  `48.88/63.26%`, W `96.61/26.39%`, X `49.22/60.31%`, and Y `96.53/26.34%`. All five cells remain
  non-dominated across completion, legal return, fall, sub-half-second completion, and sub-half-second return;
  no cell is stopped on one metric alone.
- W/Y are the two demo-priority candidates and U is the stable reserve. V/X preserve the high-return frontier,
  but `22%–24%` fall rates make them not demo-ready. The next behavior gate is the same vendor MuJoCo exam for
  W/Y, not more blind Isaac updates. Training-time virtual outcomes are not deployment evidence, so G05 remains
  `Partial`.
- Z3 has been closed exactly and will not be replayed. Pod1 V/L2 and Pod2 D2/F remain live; their checkpoint/log
  path gate was incomplete, so none was signaled.

Follow-up note (2026-07-18 19:23 CST, Z3 closed exactly while other incomplete gates remain untouched; Gate remains `Partial`):

- Pod1's only connection revalidated Z3's numeric process group/trainer identity, start time, command line and
  source. It still had no first `Learning iteration`, so only that recorded process group was handled. The trainer
  is now absent from `/proc`; its evidence directory is preserved and it must not be replayed. V and L2 remain
  NVML-live. The complete terminal-gate evidence was truncated from this review's output, so neither was signaled.
- Pod2's only connection revalidated D2 and F as live. The review looked for stdout in the source timestamp
  directory, but their actual logs are under `simple_half_second_sprint_20260718/<run>/run.log`. Current
  iteration/fatal conditions are therefore `UNKNOWN`, and neither process was signaled. The other eight exits
  remain `UNKNOWN`; the earlier confirmed A/C2 natural terminals are unchanged.
- The remote `+1000` aggregation script ran for U/V/W/X/Y, but its long middle section containing the integer
  output was truncated. Exact metrics, ranking and stop decisions remain `UNKNOWN`. Detailed evidence is in the
  [half-second sprint experiment](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md). G05 remains `Partial`.

Follow-up note (2026-07-18 18:51 CST, most processes exited and `+1000` is readable but unjudged; Gate remains `Partial`):

- Pod1 had only three training-side NVML compute processes, distributed `0/1/2` across GPU0/1/2 at `0/0/1%`
  utilization. U/W/X/Y
  each had `model_6700` with the process absent. V also had `model_6700`, but its trainer remained live; L2
  remained live at iteration `6700`. Z3 had spent about 11 hours 37 minutes in its single startup without an
  `rsl_rl` log or first `Learning iteration`. It is a startup hang, not an accepted trainer, and its fatal status
  cannot be claimed as zero.
- Pod2 had only two NVML trainers at `1/0/1`: D2 and F were both live at iteration `6700`, with zero fatal log
  matches. A/C2 retain their earlier confirmed natural-terminal status. Eight other Pod2 processes were absent,
  but their terminal material was not individually bound in this review, so their terminal/exit status remains
  `UNKNOWN` rather than natural-terminal.
- All five U/V/W/X/Y `model_6700` files exist, making their `+1000` integer ledgers available to read. They have
  not yet been aggregated or judged, so no `+1000` winner or stop decision is recorded. Detailed state is in the
  [half-second sprint experiment](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md). G05 remains `Partial`.

Follow-up note (2026-07-18 07:42 CST, five natural terminals and one boot pending; Gate remains `Partial`):

- Pod1 K2, P2 and W naturally reached terminal iteration `6700` with zero fatal events. Pod1 then had eight
  accepted trainers plus one boot-pending Z3 process at `4/3/2` across its GPUs. Pod2 A control and C2 deadline-plus-qdot0 also naturally reached
  iteration `6700` with zero fatal events; Pod2 then had ten live trainers at `3/4/3`. All remaining live
  trainers on both Pods had zero fatal events at this snapshot.
- The single Z3 launch on Pod1 GPU2 was still in boot/import with no first `Learning iteration` and no fatal
  event. It was not replayed in this cycle and is not accepted as a successful trainer. Detailed run identity
  and runtime state are recorded in the
  [half-second sprint experiment](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md). G05 remains `Partial`.

Follow-up note (2026-07-18 06:42 CST, completed `+500` window; Gate remains `Partial`):

- Pod1 remains at 11 live trainers (`4/4/3`) and Pod2 at 12 (`4/4/4`), with zero fatal events. GPU2 still
  has three trainers, so Z3 remains ineligible.
- Across cumulative updates `5701–6200`, completion/legal-return/fall rates were U
  `91.68/31.42/1.432%`, V `48.66/61.38/20.71%`, W `93.02/30.78/1.241%`, X
  `50.91/59.08/20.35%`, and Y `92.00/31.52/1.322%`. In the recent `5901–6200` window they were U
  `94.47/31.98/0.612%`, V `46.93/70.82/22.84%`, W `94.92/31.47/0.552%`, X
  `47.49/69.41/23.14%`, and Y `94.56/32.15/0.572%`.
- U/W/Y form the stable position-priority group, with Y/W on the current stable-demo frontier. V/X form the
  aggressive velocity-priority group: they are not demo-ready because of low completion and high fall rates,
  but they are the only high-return frontier. No cell dominates across completion, return, and safety, so the
  registered decision is zero stops. These remain Isaac training-time virtual outcomes, not vendor MuJoCo
  behavior; G05 stays `Partial`.

Follow-up note (2026-07-18 05:42 CST, second independent 100-update window; Gate remains `Partial`):

- Pod1 remains at 11 live trainers (`4/4/3`) and Pod2 at 12 (`4/4/4`), with zero fatal events. GPU2 still
  has three trainers, so Z3 remains ineligible to start.
- All five new cells completed the independent `5801–5900` window. Overall completion/legal-return/total-fall
  rates, with pre/post fall in parentheses, were U `93.80/31.29/0.956% (0.143/0.814%)`, V
  `44.19/62.59/21.80% (20.17/1.62%)`, W `94.37/30.81/0.901% (0.150/0.751%)`, X
  `46.38/59.55/21.39% (19.61/1.77%)`, and Y `93.58/31.52/0.855% (0.126/0.729%)`.
- The position-priority cells improved completion and legal return while reducing falls. The velocity-priority
  cells gained about `27.56` percentage points in legal return, but V/X completion fell by `16.07/24.01`
  percentage points and their pre-fall rates are about `20%`. Y is currently safest, W has the highest
  completion, and V the highest legal return. No cell dominates another across return, completion, and safety,
  so none is stopped. Legal return remains an Isaac training-time virtual outcome, not vendor MuJoCo behavior;
  G05 stays `Partial`.

Follow-up note (2026-07-18 05:13 CST, first TTS-by-outcome `+100` window; Gate remains `Partial`):

- Pod1 has 11 accepted trainers at `4/4/3`, iterations `5929–6286`; Pod2 has 12 at `4/4/4`,
  iterations `6018–6238`. Fatal count is zero. GPU2 still has three trainers, so the predeclared Z3
  third-process condition is false.
- All five new Pod1 cells completed the common-parent `5701–5800` window with sufficient denominators in
  each initial time-to-strike bucket: `<0.5`, `=0.5`, `(0.5,0.9]`, and `>0.9 s`. The velocity-priority plus
  strong-ready cell (V) recorded legal-return rates `21.96/36.03/38.21/27.01%` and completion rates
  `62.29/61.14/59.40/55.10%`. The position-priority plus free non-striking arm cell (W) recorded the highest
  completion rates, `87.43/87.16/85.92/82.41%`, with legal-return rates
  `19.41/29.90/30.75/18.97%`. Position-priority plus strong-ready (U) and position-priority plus a muted
  contact-window teacher (Y) trade completion against return, while velocity-priority plus a free non-striking
  arm (X) is intermediate.
- No cell is dominated on every recorded dimension, so no cell is stopped at `+100`. The `<0.5 s` bucket is
  no longer zero-capability in the training ledger, but all reported returns are Isaac training-time virtual
  outcomes, not vendor MuJoCo behavior. G05 therefore remains `Partial`.

Follow-up note (2026-07-17, ready-ruler successor queue source paper; Gate remains `Partial`):

- A new plan-only queue freezes one Pod2 4096-environment/two-update full-scene probe followed by a
  four-cell same-parent comparison: baseline/strong ready Reward ×
  [`qdot-limit hinge`](../DEFINITIONS.md#qdot-limit-hinge) `-5/0`. Qdot hinge is a joint-speed
  penalty, not a random lateral push. All four cells resume complete state from the Pod2 equal-Reward
  `model_5700` parent and use one seed as a mechanism funnel.
- The minimal source gate exposes local `validate/plan`, one read-only Pod2 parent inspector and one
  full-scene probe/finalizer; runner SHA is `2cf2f3dd…5c8f` and its focused suite is `32 passed`.
  Fill, behavior/portfolio consumers and exact stop are deliberately blocked next-iteration interfaces,
  not hidden commands.
- Launch remains false until a no-clobber probe receipt proves positive and conserved planner task-entry
  ready denominators, explicit foot-sensor unavailability, zero legacy-hold violations/nonfinite values,
  complete task-revision counters, a finite checkpoint and exact source/hard/claim/binding/lineage. The
  parent has passed the separate Pod2 read-only semantic binding inspection (inspection content
  `e17cedb1…ade4`, evidence content `85967393…1096`); both evidence documents are then
  frozen into a new activation commit rather than patched on a Pod.
- Each future `+200/+500/+1000` decision requires two disjoint complete 100-update integer windows.
  Sparse legal-return zero with insufficient eligible opportunities cannot stop a cell, and same-parent
  portfolio review retains at least two cells. At this source milestone no full-scene probe, trainer,
  behavior receipt, ranking or signal has run. See the
  [experiment](../experiments/2026-07/EXP-P1-TASK-REVISION-READY-SUCCESSOR.md) and
  [operation](../operations/run_phase1_task_revision_ready_successor.md).

Follow-up note (2026-07-18, direct exact-0.5 K100 completed; Gate remains `Partial`):

- The versioned launch harness was removed from the active workflow. A direct Pod2 evaluator run now needs only
  the checkpoint, question bank, timing paper, an unused output directory and an idle GPU; operators no longer
  provide or compare SHA values before launch.
- Two real runtime bugs were fixed: both planner-revision owners are disabled before `gym.make`, and frame-0 body
  velocities are zeroed in the MotionLoader backing tensors instead of an advanced-index copy. The focused suite
  is `18 passed`, including asymmetric planner state and property-copy regressions.
- `taskrev_p2_equal_reward@model_5700` completed all 100 scheduled questions from exact zero-velocity frame 0.
  It reached/ hit/returned `0/100`, had `0/100` physical falls, and ended all 100 attempts by deadline guard.
  Both sides were `0/50`. This rejects this checkpoint for the strict 0.5-second requirement; it is still an
  Isaac diagnostic and not vendor MuJoCo evidence. See the
  [experiment](../experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md) and
  [direct operation](../operations/run_phase1_task_revision_0p5_exam.md).
- At the 2026-07-18 04:04 CST snapshot, Pod2 had `12/12` accepted trainers at `4/4/4`; Pod1 had `11/12`
  at `4/4/3`, for 23 live trainers. Five new Pod1 jobs loaded a complete pre-converted A3 composition through
  `HOPE_AGIBOT_A3_USD_PATH` and `UsdFileCfg`, crossed their first training iteration, and never initialized the
  dynamic URDF importer. The importer bypass is therefore runtime-verified; G05 remains `Partial` because policy
  quality and vendor MuJoCo are still open.
- The GPU2 fourth-slot `Z` and one unchanged infrastructure retry `Z2` both aborted during Kit USD shader discovery,
  before env or Reward construction, with `malloc(): invalid size`. Other jobs using the same USD remained live and
  both device and host memory had ample headroom. They are infrastructure failures, not policy-setting failures.
  No third blind retry is allowed; the cell may start as a third process only after GPU2 naturally drops to two.
- The original 18 trainers use the old source. They record and activate same-ball target/TTS revisions, but
  do not emit outcome counts cross-tabulated by initial TTS bucket, and those missing counts cannot be reconstructed
  retrospectively. The new source on this branch implements integer opportunity, completion, hit and legal-return
  counts for four initial-TTS buckets: `<0.5`, `=0.5`, `(0.5,0.9]` and `>0.9 s`. The five cached-USD jobs now emit
  these counters directly. Consequently `+100` after the common parent is the first fair early comparison point;
  earlier observations may only reject boot, stability, or activation failures, and the old 18 arms cannot identify
  which preparation-time bucket has learned to hit or return. The latest completed forehand timing exam remains
  hit/return `0/50`. The current sprint and its single-seed questions are recorded in the
  [half-second sprint experiment](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md).

Follow-up note (2026-07-17, planner ready-ruler source repair; Gate remains `Partial`):

- Root cause is now explicit: planner mode correctly sets legacy hold clocks to zero because
  time-to-strike (TTS, the planner's remaining time before contact) owns preparation, but the old
  ready ruler admitted only `motion.in_hold`. The 19 existing `+1000` receipts therefore have
  structurally zero ready denominators. A frozen checkpoint cannot be retroactively given counters
  its source never emitted, so those receipts cannot be backfilled, ranked or used to stop arms.
- The successor ruler samples on the first metrics sample after each active new
  `(control_epoch, task_id)` has been installed. Same-ball `task_revision` updates do not count
  again, and an unexpected planner-mode legacy hold is a violation, never an alternative
  eligibility path. It exposes four witnesses: `ready_phase_sample_count`,
  `ready_planner_task_entry_sample_count`, `ready_planner_legacy_hold_violation_count` and
  `ready_foot_sensor_unavailable_sample_count`.
- Pod2 checked final exact source `0ebd14a6…a8dd` CPU-only in a clean detached worktree; all four
  focused functions passed (`4/4`), including the illegal-hold negative case and unavailable-sensor
  accounting. Earlier missing-pytest, temporary-float-stub and source-materialization failures remain
  recorded as harness failures, not source verdicts. This is not full-scene evidence. A clean full-scene probe plus two
  complete disjoint 100-update integer windows are still required before behavior pruning, so G05
  stays `Partial`.

Follow-up note (2026-07-17, historical v1 exact-0.5 K100 source gate; superseded by the v1 failure note above):

- The checkpoint-bound [0.5-second timing exam](../DEFINITIONS.md#timing-exam-0p5) first froze a v1
  one-launch/read-only-inspect supervisor. Harness SHA is `c2ce2784…1b63`, activation SHA is
  `996775d6…7cfb6`, and the focused suite is `35 passed, 1 skipped`. It binds
  `taskrev_p2_equal_reward@model_5700`, the fixed exact-25-tick K100, checkpoint/hard/claim/binding,
  runtime, resource floors, no-retry commit chain, guardian and owned-cgroup cleanup.
- At this checkpoint it was source evidence only. The skipped host test was the delegated cgroup-v2
  runtime probe; the later v1 Pod launch and its input failure are recorded above. At 2026-07-17 13:05Z Pod2 had zero trainers and
  all three GPUs free; Pod1 is `UNKNOWN`. See the [experiment](../experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md)
  and [operation](../operations/run_phase1_task_revision_0p5_exam.md).

Follow-up note (2026-07-17, task-revision pool terminal refresh; Gate remains `Partial`):

- The old statement below that 19 trainers were live is historical launch-time evidence. By the
  `+1000` cycle, all 19 checkpoint-ready cells had behavior receipts; most had naturally exited,
  Pod2 later reached zero trainers, and no scientific stop signal was issued. Four ready/balance
  denominators were zero, so the preregistered portfolio correctly produced no winner or elimination.
  Future long runs must prove nonzero ready denominators in a full-scene probe first.

Follow-up note (2026-07-17, task-revision pool uniquely launched; Gate remains `Partial`):

- All 22 delay-zero cells consumed exactly one immutable launch claim. An independent read-only
  recomputation after the final batch found 19 `live_exact` trainers and no unsubmitted launchable
  cell. Pod1 task-revision occupancy is `3/3/2` and Pod2 is `3/4/4`; the independent RallyV11 on
  Pod1 GPU0 was not touched. Every live row had `PID=PGID`, matching claim/binding, `/proc` and
  NVML, had crossed its resumed first iteration and had no OOM, Traceback or Killed marker.
- Three rows terminated before the first training iteration: two dynamic-URDF importer malloc
  exits (`rc134`) and one reviewed boot stale-timeout. Their processes and NVML contexts are absent;
  they are infrastructure rejections, not Reward or hypothesis failures, and are not automatically
  replayed. The two positive-delay rows remain intentional NO-LAUNCH because governor and actor
  transport are not atomic at non-zero delay.
- This closes the launch/runtime part of G05 only. No checkpoint has yet completed the 0.5-second
  K100. The CPU-only v4rg TOPP run-up certificates passed their safety/fidelity gates but found
  only `0.98 s` forehand and `0.78 s` backhand feasible upper bounds, not a 0.5-second certificate.
- The same-parent pruning consumer is now complete: `+200` checks revision/ledger activation,
  `+500` combines dense-collapse with a portfolio guard, and `+1000` applies the YAML-bound
  tolerance Pareto while retaining at least two cells plus an actually observed exact-0.5 sample
  and broad timing
  coverage. The first read-only `+200` scan produced zero stops because some still-live siblings
  had not yet reached the shared milestone; infrastructure-terminal rows were excluded. Actual
  portfolio receipts, the 0.5-second K100 result and vendor MuJoCo remain open.
- The first Pod2 write-side `+200` cycle then found a missing-directory harness bug before the first
  behavior receipt: the no-clobber runtime correctly refused to publish into an absent
  `behavior_milestones/` parent. No behavior/portfolio receipt or signal was produced. The source
  successor creates only that fixed direct child below the already bound real run directory and
  rejects missing parents, files and symlinks. After the repair entered `main@85ab36df`, one Pod2
  cycle successfully published nine behavior receipts. The six-cell quality parent had all
  revision/exact-0.5 mechanisms active and therefore correctly produced zero +200 eliminations;
  the continuous parent still has two live siblings before checkpoint. This closes the Pod2 +200
  runtime mechanism consume, not the +500 behavior comparison, so G05 remains `Partial`.

Follow-up note (2026-07-17, task-revision `A6` full-scene gate passed; Gate remains `Partial`):

- The clean `b1f5a380` source passed one 4096-env/two-update generic and specialized probe on Pod1
  GPU1. The checkpoint is finite (74 floating tensors / 1,762,715 elements), schema-3 hard contract
  and lineage are bound, fatal count is zero, and exact PID/NVML state is naturally absent.
- All four preparation-time strata were observed; exact 0.5 seconds has 2,406 samples. Same-ball
  revision accounting is exact (`176,387 = 165,417 accepted + 10,970 rejected`), including 839
  accepted and actor-visible last-precontact updates. The specialized receipt content SHA is
  `77db7925…d54a`; the queue is activation-ready for 22 delay-zero cells. This proves runtime
  mechanics only, not 0.5-second return ability or a winning policy.

Follow-up note (2026-07-17, task-revision `A5` reached rollout but hard-contract producer failed;
Gate remains `Partial`):

- `A5` crossed the previous CUDA metric-shape failure, observed a real PPO iteration and naturally
  completed its two-update full-scene process. Its generic finalizer correctly refused activation:
  the producer serialized the weighted initial preparation-time mixture as a two-key list rather
  than the required object. No behavior result, queue activation or promotion follows from `A5`.
- The successor sources the mixture from the parsed runtime authority and validates a newly built
  schema-3 contract before writing its sidecar or constructing the runner. The old generic converter
  is unchanged, so planner-OFF historical contracts do not drift. Focused dependency-light
  regression is `203 passed`; a fresh `A6` full-scene runtime pass remains mandatory.

Follow-up note (2026-07-15, schema-v3 capture complete but attestor newline false rejection; Gate remains `Partial`):

- Pod2 GPU2 attempt `control_model500_v3_obsfix_gpu2_20260715a` naturally produced 4096 finite
  `natural_clip_wrap` states with `wrap_teleport=false` and no clip-switch abort. PID=PGID `399423`
  naturally disappeared after the bound result was published; claim/states/result SHA-256 are
  `81126b27...244e` / `8d07668e...95d8` / `0aa2f37f...d641`. This closes the capture runtime gate only.
- One-shot attestor attempt-1 then failed before receipt publication with rc2 `training launch claim
  canonical digest mismatch`; `teacher_receipt.json` is absent. The queue producer hashes compact canonical
  content without a trailing newline, while the attestor incorrectly reused document serialization with a
  newline for that embedded digest. This is only the first observed blocker: attempt-1 stopped in `_claim`, so
  checkpoint/lineage/hard-contract/source/motion/velocity gates remain unexecuted and unproven; capture-array
  finiteness is separate evidence. The source fix separates content bytes from newline-terminated document
  bytes and splits original capture-producer lineage from the later fixed-attestor lineage. Receipt attestation
  schema 2 and controller status reject swapped/rebound/dirty sources while permitting producer commit `906a3c3`
  to differ from the post-fix attestor commit. A tracked one-shot retry authorization binds the only accepted
  attestor commit `a38b7e9e693db407795d9a5f3af144b8f8e293cf` / script SHA
  `03611b56...310f` to the immutable v3 plan/capture/checkpoint/output; the authorization SHA is
  `87fd1c71...dfda`. This prevents an arbitrary clean HEAD
  from signing and validating itself. After the authorization-backed trainer consumer was added, the
  six-file host suite is `181 passed` with one existing
  duplicate-ZIP warning; this is source evidence, not full attestation.
  Capture remains permanently no-rerun; the one authorized attestor attempt is recorded below.
  First-reset, replacement training, second seed, judge and promotion remained unauthorized at this source stage. See the
  [machine result](../../configs/phase1_post_swing_teacher_capture_attempt_v3_result_20260715.json),
  [experiment](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md), and
  [operation](../operations/run_post_swing_teacher_capture.md).
- A follow-up consumer red-team found that the trainer initially checked only source tuple shape and
  `clean=true`; another valid 40/64-hex tuple could therefore survive loader validation. The successor
  requires a paired retry-authorization path/SHA in training config, derives both accepted tuples from
  that exact immutable file, compares them byte-for-byte with the receipt, and records the normalized
  authorization in the schema-3 hard contract. Valid-hex capture/attestor commit/SHA rebound negatives
  now fail closed.
- The only authorized attempt-2 then ran from clean detached `a38b7e9` against authorization from clean
  `main@ff9a253`, exited rc0 and published a 4103-byte, 4096-state receipt SHA-256
  `e20a6989...d2aba4`. Exact PGID `403786` is naturally absent. Merged-main controller status re-bound the
  receipt to the immutable v3 plan, original producer, fixed attestor and tracked authorization and returned
  `teacher_receipt_binding_exact=true`. This completes artifact attestation only; the 4096-environment first-reset
  pre-rollout adoption/readback probe, scientific pair, second seed, judge and promotion remain unauthorized.

Follow-up note (2026-07-15, schema-v2 capture reached runtime then failed before the first inference step; Gate remains `Partial`):

- Plan `control_model500_v2_schema2_gpu2_20260715a` passed exact read-only compose and launch-side
  environment/179-D actor/checkpoint hard-contract checks, then failed at the initial observation because this
  IsaacLab wrapper returned `(actor_observation, extras)` while `play.py` called `.to()` on the tuple. The spent
  namespace contains only the 1243-byte bound claim; states, capture result and teacher receipt are absent.
  The exact PID/PGID/SID was terminated after the failed process remained in Kit teardown for more than 99 s.
  See the [machine result](../../configs/phase1_post_swing_teacher_capture_attempt_v2_result_20260715.json).
- The successor source now routes both initial and stepped observations through the existing actor-only adapter,
  accepts only a tensor, exact `(observation, extras mapping)`, or a mapping with `policy`, and rejects critic-only
  or ambiguous structures. It also closes the final RSL-RL wrapper exactly once on normal, initial-observation and
  step failures without replacing the primary exception. Focused source/Hydra tests pass, but no new Pod capture
  has run; v2 is not retried and attestation, first reset, training, judge and hardware remain unauthorized.

Follow-up note (2026-07-15, post-swing schema-v2 launch controller hardened; Gate remains `Partial`):

- The successor controller now closes the known pre-launch false-green paths: direct no-follow namespaces,
  exact historical claim/binding/model-500 milestone lineage, plain-uint32 seed, absolute byte-bound Git and
  `nvidia-smi`, fixed hostname/machine-id/boot-id, physical Pod2 GPU2 UUID plus shared lease, safe exact
  environment allowlist, resolved Hydra compose with timeout, and same-PID exec handoff. A separate offline
  builder produces a controller-validated schema-v2 plan outside the immutable source tree.
- The read-only `plan` path now executes the exact same absolute cwd/environment/argv/timeout Hydra
  `--cfg job --resolve` as `launch`, binds output digest/bytes/elapsed, and re-verifies runtime after compose
  before any launch/capture namespace, claim or capture process can exist. `launch` repeats the same helper
  and post-compose drift check before consuming the capture namespace.
- Focused host verification with Hydra available is `41 passed` using the four files listed in the operation;
  no Pod launch, Isaac capture, attestation, first-reset
  probe, science trainer, judge or hardware command occurred. The threat model is trusted root plus accidental
  drift/concurrent legitimate jobs; the five inventoried source/Isaac import roots are not claimed as a full
  venv/stdlib/native/rootfs dependency graph. Runtime remains `Partial` until a new GPU2 plan passes on Pod2.

Follow-up note (2026-07-15, clean base-deceleration terminal and teacher capture preregistration; Gate remains `Partial`):

- Both clean-pair model-1000 checkpoints naturally terminated and independently pass filename/embedded
  iteration, finite, fresh-lineage, claim/binding and common schema-3 hard-contract checks. In steps
  980--1000, treatment/control raw pre-strike base speed is `1.00882x`, failing the frozen `<=0.90x`
  primary gate, while each non-degradation metric passes narrowly. The formal treatment verdict remains
  reject; no second seed, judge or promotion is authorized.
- Source review shows the implemented reward tracks `v_des=clamp(2*planar target error,0,1.6)` rather than
  minimizing raw speed everywhere. The frozen raw-speed metric does not measure `|v_base-v_des|` or
  exclude holds; its verdict is retained, but any successor must preregister distance-bucketed target-
  tracking metrics before launch.
- The external natural-wrap teacher capture now has an exact machine preregistration binding the main
  source, one fresh model-500 teacher, motion/bank/A3 bytes, Pod2 GPU1, 4096 states, a 20000-step limit
  and root-velocity limits. This authorizes only the one-shot inference capture. Attestation, first-reset
  readback and scientific training remain separate fail-closed gates. See the
  [clean result](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md) and
  [capture operation](../operations/run_post_swing_teacher_capture.md). Its v1 runtime derivation then
  failed closed during Hydra compose, before any capture directory, claim, process or GPU work: three
  train-only checkpoint compatibility keys were retained, and a follow-up source audit found that play
  did not apply the frozen seed. V1 is not retried; a new source and namespace are required.
- A successor-side one-shot controller now has a dependency-light source gate: it derives play argv
  from the exact run binding, removes every frozen train-only/ownership key, retains and checks seed,
  runs Hydra compose, re-hashes all inputs, and only then creates output and a numeric process group.
  It has no SSH, stop, retry or trainer command. This interim step was not runtime evidence; the newer
  schema-v2 note above records the completed seed parity and independent source red-team.

Follow-up note (2026-07-15, clean base-deceleration `+500` treatment rejected; Gate remains `Partial`):

- Both model-500 checkpoints are finite and exact for filename/embedded iteration, fresh lineage,
  claims and common schema-3 hard contract. Across steps 0--500 all disabled post-swing counters stay
  zero; V1/V2 count equalities and raw/weighted base-deceleration activation pass, including active
  V2 denominators in the frozen 480--500 window.
- Treatment/control pre-strike base speed is `0.545428/0.479838 = 1.13669x` (fails `<=0.90x`), signed
  face pass delta is `-0.16609`, composite pass delta `-0.06942`, and analytic return falls from
  `0.24771` to `0.12282`. Lower pre-fall and better velocity pass do not compensate these three
  preregistered failures. The current weight-1 treatment is rejected by the single-seed screen; the
  frozen queue nevertheless continues both trainers to +1000 only for terminal diagnostics. No
  second seed, judge, interaction or promotion is authorized. See
  [the clean main-effect record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md).

Follow-up note (2026-07-15, clean base-deceleration `+200` direction gate failed; Gate remains `Partial`):

- Control/treatment model-200 SHA-256 are `6cb55718...94f1` / `d61998ac...6892`; both receipts bind
  filename=embedded iteration 200, 1,762,715 finite floating elements, fresh lineage, claims and the
  common schema-3 hard contract `ca57a94f...cc2e`.
- Across every step 0--200, all five disabled post-swing counters are exactly zero. Raw
  base-deceleration activation is positive on both arms and weighted reward is nonzero only on
  treatment, so the execution contract passes. In the frozen 180--200 window, however, treatment /
  control pre-strike base speed is `0.75008/0.71340 = 1.05142x`, failing the preregistered `<=1.00x`
  direction gate. All four exact-strike pass metrics and analytic return are zero on both arms, while
  pre-fall is approximately 100%, so numeric non-degradation is vacuous rather than behavior success.
  Per preregistration the trainers continue to +500 only to test late reversal; no stop, second seed,
  judge or promotion is unlocked. See
  [the clean main-effect record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md).

Follow-up note (2026-07-15, clean base-deceleration pair running; Gate remains `Partial`):

- The ordered queue transaction launched control/treatment on Pod2 GPU1/GPU2 with exact PID=PGID
  `385320/385948` and claim SHA-256 `a039226a...1746e` / `673bf6c6...9392`. GPU0 remains owned only
  by Yikang. Both arms crossed their first iteration; a 04:27 CST read-only audit found no fatal log
  signature and TensorBoard through steps `106/89`.
- On every emitted update, all five post-swing replay counters are exactly zero. Raw base-deceleration
  eligible/nonzero/sum counters are positive on both arms, while weighted base-deceleration reward is
  zero only on control and nonzero on treatment. This closes the initial execution/activation gate but
  is not a behavior result. No comparison, second seed, judge or promotion is allowed before exact
  model-200 receipts. See
  [the clean main-effect record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md).

Follow-up note (2026-07-15, post-swing teacher source gate repaired; Gate remains `Partial`):

- The first cold-start candidate was rejected because it re-opened receipt/NPZ paths after hashing,
  allowed duplicate NPZ ZIP keys to collapse, and trusted self-reported checkpoint/natural-wrap
  provenance. The replacement uses one `O_NOFOLLOW` descriptor and one immutable byte buffer for
  hash/parse, a MotionCommand-owned `O_EXCL` capture namespace with no arbitrary-array writer API,
  and a separate no-clobber attestor. Callback labels are not treated as cryptographic evidence.
- The attestor uses the restricted `weights_only=True` loader and verifies actual checkpoint bytes,
  embedded schema-3/exact lineage and launch claim,
  the adjacent hard contract, clean checkpoint/capture sources, ordered motion bytes, articulation
  joint order and root/joint velocity bounds. Trainer consumption re-binds the raw capture result;
  a standalone receipt JSON cannot unlock replay. First-reset acceptance now supports a frozen
  adopted count/fraction, probability tolerance and simulator state readback.
- Dependency-light negative tests and the immutable receipt attestation pass, but no 4096-environment first-reset
  trainer-consumer Pod probe has run.
  `launch_authorized=false`; no replacement science pair, second seed, judge or promotion is
  unlocked. See the [artifact contract](../interfaces/post_swing_teacher_artifact.md),
  [producer operation](../operations/run_post_swing_teacher_capture.md), and
  [experiment record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md).

Follow-up note (2026-07-15, clean base-deceleration main effect preregistered; Gate remains `Partial`):

- The invalid pair mixed base deceleration with a post-swing buffer whose readiness depends on policy
  survival. A fresh single-seed pair now disables post-swing replay in both arms and changes only
  `base_decel_weight=0/1`; all five replay counters must remain zero on every update before behavior
  may be compared. New job, run and claim namespaces are hard-bound to Pod2 GPU1/GPU2.
- The queue reuses the exact `2c2d70d...` 4096-env terminal probe only as source/scene/checkpoint-wiring
  evidence, not as scientific recipe or behavior evidence. Launch still requires fresh final-argv
  compose, clean source, absent run directories and each arm's own first iteration. No second seed,
  judge or promotion is authorized. See
  [the clean main-effect record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-CLEAN-MAIN-EFFECT.md).

Follow-up note (2026-07-15, fresh v4 `+500` matched-activation failure; Gate remains `Partial`):

- Both model-500 checkpoints pass filename/embedded iteration, finite, fresh-lineage, claim and common
  schema-3 hard-contract attestation. Control/treatment SHA-256 are `22f78f88...a6a` /
  `a1735fbb...c14`; receipt content SHA-256 are `67d76a2b...e6d0` / `e8cdfc87...1cf7`.
- In the frozen 480--500 window, control post-swing eligible/selected/started remain `0/0/0`, while
  treatment closes at `15087/3750/3750` with selected fraction `0.248558`. Control first becomes ready
  only at step 519, which cannot be backfilled into the milestone. The buffer is populated only by
  policy-survived natural clip wraps, so base deceleration itself changes when the purportedly common
  curriculum becomes available. This is an endogenous-curriculum design failure, not a base-deceleration
  behavior result.
- Exact process groups were stopped after receipts; GPU1/GPU2 are free and Yikang's GPU0 process was not
  touched. A replacement must consume the same immutable, natural-wrap-provenance teacher-state receipt
  in both arms and fail before the first scientific update if it is unavailable. No arbitrary timeout
  capture, second seed, judge or promotion is authorized. See
  [the replacement record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md).

Follow-up note (2026-07-15, fresh v4 `+200` activation audit; Gate remains `Partial`):

- Both live arms published exact model-200 receipts. Control/treatment checkpoint SHA-256 are
  `d065441b...c77b` / `e1d2b43f...4fb7`; filename and embedded iteration are 200, all 1,762,715
  floating elements are finite, and fresh lineage, claim and the common schema-3 hard contract match.
- V1 and raw base-deceleration count closures pass, and V2 equality holds wherever samples occur.
  However, both post-swing buffers report only `buffer_not_ready` through step 200; eligible,
  selected and started counts are all zero. This violates the preregistered positive-denominator gate,
  so +200 is instrumentation-invalid and no behavior comparison is allowed. Trainers continue to +500
  only to distinguish late buffer activation from an execution-path defect; no second seed, judge or
  promotion is unlocked. See
  [the replacement record](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md).

Follow-up note (2026-07-15, Reward/physical-truth semantic audit; Gate remains `Partial`):

- The live fresh v4 recipe selects `HOPEPingPongVirtualBall`: direct achieved-FK position/velocity/
  signed-face shaping is overridden to `14/10/5`, while the task's analytic pass-net/landing/spin
  terms remain active at `20/30/5`. `vb_metrics_only=true` does not disable those task-owned terms.
  The analytic contact gate reads achieved racket state, but pass-net and landing include dense partial
  credit before a fully legal return, so training reward is not an independent physical verdict.
- `physical_ball=true` is a separate Phase-A engine-integrated diagnostic: PhysX integrates position,
  while code applies the venue aero and table-bounce models. The current recipe does not enable racket
  impulse, and the ball passes through the robot; no physical hit/net/landing metric is
  consumed by reward or observation. The running pair remains frozen. A future outcome-source ablation
  requires Phase-B receiver closure first. Phase-B still reuses the analytic paddle-contact law, so it
  tests contact detection and engine-integrated post-contact flight rather than an independent contact
  model. See
  [the Reward truth audit](../experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md).

Follow-up note (2026-07-15, Jiayi/Yikang selective cross-learning audit; Gate remains `Partial`):

- Exact-commit source/config review keeps directional recovery debt, vector station-settle, clip-frame-0
  upper-body ready, exogenous long-hold sampling and per-side planner metadata as separately testable
  candidates. Because several act in the same recovery/ready phase, none is adopted by simply adding
  weights; surviving terms require single-variable activation first and then a fixed-total-budget
  interaction test. Direct root-velocity writes, broad process-kill harnesses and checkpoint-specific
  soft clamps are not adopted as defaults.
- The historical V9/v12fix `7/7` counter was computed from engage, ready, swing completion and recovery;
  its own result marks physical contact and landing as unmeasured, and the repeated cases cover only a
  fixed forehand region. It is therefore a deployment-cycle diagnostic, not a physical-return,
  held-out-ball or multi-action generalization result. Full evidence and proposed falsifiers are in
  [the cross-learning audit](../experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md).

Follow-up note (2026-07-15, Yikang RallyV9 reach/balance matrix; Gate remains `Partial`):

- Feature source `yikang-standhit-0714@8c8cd53` passed host/pod contract checks and four Isaac
  init/load smokes. A/B/A+B strictly resume the common `ayzxv1ma/model_10600` parent; fresh A+B
  starts from iteration zero. All four first periodic checkpoints loaded with optimizer state and
  finite floating tensors.
- Production is active on A
  [`5nso93g0`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/5nso93g0), B
  [`4osh4ypc`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/4osh4ypc), resumed A+B
  [`jndof7jk`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/jndof7jk), and fresh A+B
  [`xpiapvix`](https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/xpiapvix). B/AB mechanics reached a
  real push application; fresh selected push episodes but its random policy did not survive the
  minimum 5-second delay in the short smoke, then logged its first production application by about
  iteration 322. Launch health is not a generalization, balance, or Gate3 result; matched-iteration
  evaluation and formal fork export remain pending.

Follow-up note (2026-07-16, lateral-balance trainer E1 binding; Gate remains `Partial`):

- The previously probe-only recovery/hold lateral-force candidate now has a narrow, default-off
  `train.py` binding.  Hydra exposes only `enabled`, frozen cell `L0/L1`, and an exact uint32
  counter seed.  It does not expose body/frame/XYZ/torque/duration/magnitude knobs; L0 and L1 share
  the same Philox schedule while L0 commands zero impulse and L1 remains the preregistered
  `0.04--0.08 m/s` treatment.
- An enabled run's schema-3 hard contract binds the resolved integer-step schedule, cell/seed,
  common-random schedule and hard-safety identities, direct-COM Isaac backend/transform identities,
  every active EventManager term's exact typed parameters plus manifest SHA, including the pinned
  `SceneEntityCfg` selector and resolved ids, every EventTermCfg behavior field, and plain function
  source identity. Unknown config types, decorated/method callables, non-finite/callable/opaque
  parameter values and any interval term are rejected before a force submit; pre/post-step hashes
  catch post-attach drift. The historical absent/disabled path attaches no env-cfg field,
  constructs no hook and emits no lateral hard-contract key.  Disabled cell/seed, unknown fields,
  T1 event timing, non-torso semantics and competing writers fail closed.
- Trainer mode prevents unbounded 4096-environment receipt retention and instead returns a copied
  `extras['log']` row with integer opportunity/selected/commanded/backend-accepted/zero-overwrite
  counts, abandoned and sampled/commanded/backend-accepted impulse totals, and actual randomized
  total-mass min/mean/max.  “Backend accepted” means scheduler commit plus synchronous setter/scene
  submission succeeded; it is never solver-consumed evidence.  Metric collisions or malformed Gym
  output terminally clear the private wrench.  Focused scheduler/adapter/translation regression is
  `173 passed` (`40 + 37 + 96`).
- This is E1 source/mock evidence only.  No full-scene trainer, solver-response, throughput,
  checkpoint or behavior was run; correctness-first host synchronizations remain.  Therefore
  `launch_authorized=false`, the current rolling portfolio must not depend on it, and G05 remains
  `Partial`.

Follow-up note (2026-07-15, lateral-balance Isaac adapter candidate; Gate remains `Partial`):

- A default-off, explicit-probe-only candidate now binds the merged scheduler transaction to the
  exact Isaac Lab `v2.1.0@21f7136325136ca3f6ca4e0a8125edffe5c24f7e` articulation buffers.
  It is not registered in an existing task or trainer.  Disabled mode directly delegates
  `env.step(action)` and does not inspect the environment.
- Red-team source audit found that the first candidate's COM claim was false: pinned Isaac Lab
  passes `position_data=None`, which applies at the link transform/origin rather than the COM.  That
  BODY-buffer path is rejected.  The corrected candidate leaves Isaac's built-in wrench buffers
  fully zero/unowned, reads current link poses plus PhysX local COM offsets before every substep,
  and directly calls `apply_forces_and_torques_at_position` with explicit WORLD torso-COM
  `position_data` and `is_global=true`.  A non-zero COM-offset mock proves the offset is rotated into
  WORLD rather than silently using the origin.
- Strike/window interruption writes zero in the same policy step.  When a subset reset causes the
  extra full-scene write in `ManagerBasedRLEnv`, the candidate clears the **whole batch** before
  that write so non-reset environments cannot receive an extra off-decimation wrench; any valid
  continuing pulse is reconstructed on the next policy step.  Receipts reconcile episode
  index/step, eligibility, strike window, application ledger, every physics substep and reset.
- Isaac Lab exposes no getter for the wrench actually consumed by the PhysX solver.  The candidate
  therefore advertises `solver_execution_readback_available=false`; synchronized direct-setter
  evidence is not called solver-execution evidence.  The strict full-scene probe exists but was
  not run in this source change, and the correctness-first implementation still contains hot-path
  host synchronizations.  Full-scene lifecycle, independent dynamics response, same-GPU
  throughput/no-host-sync redesign, runner/hard-contract binding and held-out behavior all remain
  open.  The API also has no owner/readback for a second direct setter inside the same scene write;
  exact source closure must exclude that path before runtime ownership can be claimed.
  `launch_authorized=false` and `runtime_adapter.implemented=false` are unchanged.
- Adapter red-team also requires every private command copy/reset and every substep boundary to
  prove both built-in buffer identities, all body/environment bytes and the owner flag.  Same-tick
  non-torso writes fail closed.  Any scene-write exception, wrong return type or post-dispatch
  validation or scene-hook restoration failure enters a terminal zero-overwrite guard and can never continue.  A
  successful rollout also cleanly terminalizes and submits a full zero command before receipt validation,
  source re-attestation, output creation, print or environment close; a failed terminal zero cannot publish.  The probe now
  loads motion through stable kernel-fd paths, rechecks public path identity/SHA before output, and creates its
  receipt through stable-parent-dirfd `openat(O_EXCL|O_NOFOLLOW)`.  Python tensor copies/CUDA sync are
  no longer self-declared atomic/noexcept: any commit exception or malformed return writes no
  scheduler ledger, permanently dirties the backend and forbids retry/advance/another simulator
  step.  Probe aggregate reset/strike claims are non-vacuous; missing natural reset or active-pulse
  strike interruption yields an explicit lifecycle-uncovered status and null reset evidence.
- Focused scheduler plus corrected-adapter verification is `65 passed`: 36 scheduler/transaction
  cases plus 29 dependency-light adapter/artifact cases, including non-zero COM and derived-COM overflow,
  offset, substep/reset/direct-setter competing writers, scene/direct-setter exceptions, malformed returns,
  hook-restore failure, clean/failing terminal zero, post-zero publication failure, motion replacement and
  output-parent symlink swap.  Source compilation, JSON parsing and
  whitespace checks are part of the source handoff.  No Pod, simulator, trainer or hardware was touched.  Reproduction and the exact
  probe boundary are in
  [EXP-P1-LATERAL-BALANCE-PERTURBATION](../experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md)
  and [run_lateral_perturbation_runtime_probe](../operations/run_lateral_perturbation_runtime_probe.md).

Follow-up note (2026-07-15, lateral-balance perturbation source gate; Gate remains `Partial`):

- The proposed sparse-balance-data ablation now has a source-only, recovery/hold-exclusive pulse
  scheduler.  Its treatment samples a bounded left/right WORLD-Y impulse-equivalent force at the
  `torso_link` COM and scales force by the articulation's post-randomization total mass; the
  matched control has zero impulse but shares the same potential random schedule.  It never writes
  root velocity, X/Z force or torque.
- Red-team review rejected the first linear counter streams and silent reset truncation.  The
  successor uses Random123-compatible domain-separated Philox4x32-10 (known-vector and
  cross-stream/cross-seed distribution tests), exposes the potential draws plus a common-random
  schedule SHA, and records sampled/commanded/backend-accepted/abandoned impulse when reset interrupts a
  pulse.  Reset cannot immediately restart a pulse.  Hot-path tensor validity/application
  accounting no longer calls `.item()` or `bool(torch.any(...))`.
- A second red-team pass found that finite-but-extreme config values, float64-to-float32 casts and
  mass multiplication could emit an infinite or arbitrarily large force, and that the typed
  receipt did not bind the randomized mass or commanded force.  The successor freezes an
  immutable `0.15 m/s`, `2.0 m/s^2`, `0.02--0.20 s`, `200 N` envelope; validates config, derived,
  cast and final wrench layers; and binds actual total mass, WORLD force/impulse, scheduled mask and
  transform identity into the receipt/application ledger.  Applying at torso COM only removes an
  explicit/link-local lever-arm torque; it does not imply zero whole-articulation `r x F` angular
  impulse or contact response.
- The fourth source successor closes tensor-ownership holes: an adapter receives isolated clones
  of mass/force/torque, so mutation followed by rejection or exception leaves caller tensors bit
  exact; the scheduler keeps a private deep-cloned application ledger and every first/cached/
  duplicate public return is another deep clone.  Adversarial mutation tests cover both paths.
- Final reviews then closed four source-state-machine holes.  Public receipt acknowledgement has
  been removed; only a non-public dispatch identity capability can prepare/commit bookkeeping, and
  every expected tensor/mask/count is derived from the scheduler-private canonical step.  The
  adapter seam is now two-phase: a source-token-bound, side-effect-free typed preflight is fully
  validated before a full-buffer commit/readback whose successful result is `None`.  Rejected preflight
  is discarded with backend/cache/counters unchanged and permits a canonical retry on the same
  step token with a fresh preflight nonce.  Python/CUDA copies are not self-certified as atomic or
  noexcept: commit exceptions or non-`None` results permanently mark the backend `DIRTY/UNKNOWN`
  before any ledger and block retry/advance/next simulator step.  Same-step cache also binds
  transform SHA, backend SHA and the live backend object token, so another same-SHA adapter cannot
  borrow an old application ledger.
- Mass/cast/final-wrench, cached duplicate and preflight receipt/mask safety predicates are all
  host-visible before backend commit; monkeypatched async-assert attacks produce zero writer calls.
  This correctness-first implementation now has multiple hot-path host completions, not one, and
  therefore still fails the preregistered no-host-sync runtime gate.  Strike and safe-window
  interruptions now preserve per-environment sampled/commanded/backend-accepted/abandoned impulse just like
  reset, satisfy both conservation identities, and issue an actual full-buffer zero commit on the
  interruption tick in the runtime-ack test.
- Focused source verification is `36 passed`; this is E1 only.  The Isaac adapter, WORLD-to-BODY
  wrench transform, post-randomization total-mass reader, every-step full-batch zero overwrite,
  runtime ledger/logger, hard-contract/runner integration and content-addressed held-out papers are
  absent.  A same-GPU throughput comparison must retain at least `0.95x` environment-steps/s with
  no more than `1.05x` p95 step time and no hot-path host sync; promotion also requires an immutable
  held-out ball-arrival-bin by action-family paper with all-bin/worst-bin reporting and vendor
  MuJoCo reuse.  Both are pending.  The machine prereg remains `launch_authorized=false`; no Pod,
  trainer or simulator was touched.  See
  [EXP-P1-LATERAL-BALANCE-PERTURBATION](../experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md).
- Replayed on current `origin/main@107102f`, the focused suite is `36 passed`; the full 57-file
  tracking suite is `847 passed, 22 skipped, 3 failed`.  All three failures reproduce unchanged on
  `origin/main` (two existing MotionLoader `PosixPath` cases and one virtual-scorer tolerance case),
  so no integration regression is attributed to this source gate.

Follow-up note (2026-07-14, Pod2 `+200` activation audit; Gate remains `Partial`):

- Four science trainers produced exact `model_200` receipts with finite tensors, matching embedded
  iteration, fresh lineage, launch claim, and schema-3 hard contract. Conditional face guidance had
  zero gate/cost/reward throughout frozen steps 180--200, which proves zero eligible samples under
  its current formula; that setting is activation-invalid rather than a learned-policy failure.
- The V1+V2 x base-deceleration pair also passed checkpoint identity, but its preregistered V1, V2,
  and base-deceleration integer denominators/numerators were not emitted. Its curves are therefore
  instrumentation-blocked, not evidence for or against the reward. Neither pair authorizes a second
  seed, judge, or promotion. Replacement queues remain fail-closed until the execution path exports
  all required counters and a new exact source passes the strict full-scene probe. See the
  [conditional](../experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md) and
  [interaction](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-INTERACTION.md) records.

Follow-up note (2026-07-13, seed-budget correction; Gate remains `Partial`):

- The rejected `SZ` family used four from-scratch seeds through model-4000. That was enough to
  establish instability and expose a signed-face measurement defect, but continuing to replicate a
  rejected recipe bought more evidence than the baseline decision needed.
- Future mechanism screens use one blocking seed first. One 5090's four breadth slots hold four
  distinct causal cells, with relative checkpoints at `+200/+500/+1000`; only a surviving cell together with its
  matched control receives a second seed. Three to four seeds and terminal training are reserved for
  a candidate that could actually become the accepted baseline.
- The first application is the proposed hot-start/fresh × face-guidance-off/on signed-face funnel.
  It has no source SHA, machine prereg or Pod run yet, so this note does not authorize launch and does
  not add accepted training evidence. See
  [the experiment record](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) and
  [the acceleration policy](../research/phase1_ablation_acceleration_2026-07-11.md#seed-是晋级税不是首轮并发单位).

Follow-up note (2026-07-05, R15 v5 correction):

- Reverted the mistaken v5 default switch in `/workspace/yikang/nohope`: `cfg/train.yaml` and `cfg/play.yaml` keep `motion_file: null`, and the product/default deploy-parity task remains on the hopex lineage (`strike_phase_per_clip: [0.47, 0.333]` with the existing hopex target boxes). v5 is an R15 ablation arm only, passed with explicit `motion_file=` / `motion_file_2=` and `task.racket.*` CLI overrides.
- Added `hope_training/whole_body_tracking/cfg/strike_annotations.yaml` as the source of truth for hand-aligned contact phases. `scripts/analyze_strike_phase.py` now treats speed-peak picks as diagnostic candidates and applies annotations first. The checked v5 output selects forehand frame 37 / phase 0.673, not the frame 43/44 post-contact whip (`vx` near zero, +Y-heavy); backhand phase 0.345 remains unverified until a frame-by-frame scrub.
- R15 v5 sampling boxes are generated from the annotated frames: forehand pos x/y/z `[0.29,0.49]`, `[-0.63,-0.43]`, `[0.74,0.94]`; forehand vel x/y/z `[0.74,1.74]`, `[0.71,1.71]`, `[1.20,2.20]`; backhand pos x/y/z `[0.60,0.80]`, `[0.12,0.32]`, `[0.81,1.01]`; backhand vel x/y/z `[2.60,3.60]`, `[0.50,1.50]`, `[1.66,2.66]`. Face normals from video/GVHMR clips are wrist +Y proxies and marked unreliable.

Follow-up note (2026-07-02/03, after the simtoreal2 merge):

- The training default is now `task=HOPEPingPongDeployParity` (`cfg/train.yaml`;
  `HOPEPingPongRealSensor` is a backward-compat alias): the 175-D deploy-parity actor contract
  (see [../interfaces/policy_observation_action.md](../interfaces/policy_observation_action.md)),
  base-free footwork rewards (`base_position` removed; dense `racket_progress` + pre-strike
  slip/twist/upright penalties + `arm_torque_saturation`), per-clip blade-centered 3-D target
  pos/vel boxes, `strike_phase_per_clip: [0.47, 0.333]` on the re-grounded `_hopex` (v3) clips,
  `racket_velocity_std: 1.0` (plan 1.0 → 0.8 → 0.5), and PD-gain DR re-enabled at ±15%
  (2026-07-02 sim2real fine-tune; documented HITTER departure). The 180-D `task=HOPEPingPong` is a
  legacy comparison path and is not deploy-honest.
- This path produced the first hardware-deployed policy: `model_p4_deployparity.onnx` (175-D /
  31-act), sim2sim-validated in MuJoCo and run on the real A3 on 2026-07-02 (forehand only). The
  newest lineage is the explicit-clipped-PD fine-tune (`launch_explicitpd_ft.sh`, model_25700).
  Contract checks: `hope_isaac_py scripts/verify_realsensor.py --check layout|rollout|onnx`.
- Exact accepted run IDs/metrics for a quality baseline are still pending (see Not done).

Follow-up note (2026-07-01, `main` after the unified HITTER audit — values superseded above):

- The active `HOPEPingPong` config then defaulted to unified forehand+backhand training (`registry_name_2` enabled), `target_mode: uniform`, fixed strike plane `x=0.4`, `strike_phase_per_clip: [0.36, 0.74]` (v1 clips), actor `swing_type`, and no actor `racket_target_normal_w`.
- The `registry_for_runner` blocker and local-motion regression found during the audit were fixed in the training entry: local `motion_file=<forehand.npz> motion_file_2=<backhand.npz>` now bypasses WandB, while registry-backed runs still link the used registry artifact(s).
- The 2026-06-26 first-loop result remains useful as pipeline history, but the unified HOPEPingPong path still needs a fresh Isaac run before it can count as an accepted baseline.

Done:

- Training scaffold is present under `hope_training/whole_body_tracking`.
- Existing docs describe BeyondMimic-style training assumptions.
- The branch adds Hydra training/eval entrypoints, HOPEPingPong task config, racket-target command logic, and A3-specific robot config.
- `reimplement.md` records that `TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied A3 URDF asset, including wandb logging, checkpoint save, and `policy.onnx` export.
- Commit `42489cd` adds `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Earlier `reference_perturbed` / PATH B-C experiments (commit `c951d9d`) are present in history, but the current default is the 2026-07-01 unified HITTER path: direct uniform target sampling, no success-gated perturbation curriculum, and no `ref_vel_scale` ramp. `reference_perturbed` remains available as a non-default `target_mode` and now uses per-clip reference strike centers.
- `RacketTargetCommand` logs conditional exact-strike pass rates: `strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`, `strike_composite_success_exact`, and `exact_strike_sample_count_decayed`.
- `RacketTargetCommand` also supports optional debug reward logging (`debug_reward_logging`) for swing-through sign checks and raw-vs-gated reward kernels. Keep it off for production runs unless diagnosing reward scale.
- The HOPE actor observation includes `swing_type` and desired runtime targets, but not `racket_target_normal_w`; racket pose/velocity/normal remain critic/reward-only simulation state.
- `scripts/train.py` logs import provenance, env-cfg source, every applied task override, and post-override racket knobs; YAML keys that target missing env-cfg attributes raise instead of silently no-oping.
- `scripts/train.py` keeps registry defaults available from `cfg/task/*.yaml`, while `motion_file=<local.npz>` and optional `motion_file_2=<local.npz>` take precedence for no-WandB smoke tests or locally generated references.
- Local unified-policy training can use Step 9-12 video-generated motions directly with `motion_file=../motions/preprocessed/hope_forehand.npz motion_file_2=../motions/preprocessed/hope_backhand.npz logger=tensorboard`.
- Generated ONNX policy artifacts remain ignored by asset policy unless a gate records an external artifact path.
- Merged from `train_1` (2026-06-26) and superseded by the unified HITTER alignment: paddle-contact timing is per clip, expressed as `strike_phase_per_clip` (then `[0.36, 0.74]` on v1 clips; current default `[0.47, 0.333]` on the `_hopex` v3 clips); `episode_length_s: 3.0` caps each episode to about one swing; `scripts/train.py` / `cfg/train.yaml` keep the `checkpoint_path` knob for staged resume.
- 2026-07-02: `HOPEPingPong.yaml` and `HOPEPingPongRealSensor.yaml` were synchronized for unified forehand+backhand training. Old single-swing wording and old backhand positive-y/high-z coordinate notes were removed from the YAML comments; both observation-route candidates now point at the same re-grounded target boxes.
- 2026-07-02: Clip wrap no longer performs mid-episode RSI for HOPE ping-pong (`motion.rsi_on_wrap: false` in both task YAMLs; the knob was renamed to main's equivalent `motion.wrap_teleport` on 2026-07-03). Episode reset still initializes from the reference, but wrap only advances the reference clip/time and target, forcing the policy to learn physical between-swing recovery.
- 2026-07-02: `racket_progress` now resets its previous-distance baseline and emits zero on motion/target resample steps, removing the fixed wrap/reset reward spike from the base-free footwork signal.
- 2026-07-02: `HOPEPingPong` and `HOPEPingPongRealSensor` were switched to `racket.target_mode: reference_perturbed`: the initial target center is the imitated clip's own strike-frame racket FK state, and the long-run distribution widens through success-gated perturbations (`ref_perturb_pos=[0.15,0.20,0.15]`, `ref_perturb_vel=[1.0,1.0,0.8]`). [Pre-merge branch configuration: after merging main's unified HITTER redesign, the default is `target_mode: uniform` with a fixed strike plane; `reference_perturbed` and these perturbation ranges remain available as a non-default option.]
- 2026-07-02: The shared RunPod now has an independent `hope-motion-py310` Conda env for GVHMR/GMR motion preparation, separate from Isaac Lab. GVHMR/GMR import checks pass on RTX 5090 with PyTorch `2.7.0+cu128`; PyTorch3D `0.7.9` was built from source for `sm_120`; GVHMR non-body checkpoints are present; and the local ignored GMR clone has verified `agibot_a3` MJCF/IK registration with 31 hinge joints matching `joint_order_agibot_a3.yaml`.
- 2026-07-02: Uploaded forehand/backhand MP4s were converted through GVHMR -> GMR -> `scripts/csv_to_npz.py` into local ignored motions: `hope_training/motions/preprocessed/hope_forehand.npz` (`joint_pos=(139,31)`, `body_pos_w=(139,32,3)`, `fps=50`) and `hope_training/motions/preprocessed/hope_backhand.npz` (`joint_pos=(132,31)`, `body_pos_w=(132,32,3)`, `fps=50`).
- 2026-07-02: WandB setup is verified for `WANDB_ENTITY=BerkeleyPingPong`, `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org`, `WANDB_PROJECT=hope_wbc`, and `WANDB_MOTION_PROJECT=csv_to_npz`. The registry aliases `dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_forehand:latest` and `.../hope_backhand:latest` resolve to `BerkeleyPingPong/csv_to_npz/hope_forehand:v4` and `BerkeleyPingPong/csv_to_npz/hope_backhand:v4`; both contain `motion.npz`.
- 2026-07-02: Registry-backed `HOPEPingPong` WandB smoke passed with `source setup_train_env.sh && hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=32 max_iterations=1 logger=wandb run_name=smoke_registry_wandb_finish`. W&B run `6xus13ga` finished at https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga, used both motion artifacts, and synced `model_0.pt`, `2026-07-02_11-56-04_smoke_registry_wandb_finish.onnx`, config, diff, output log, and summary. Local outputs are under `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-07-02_11-56-04_smoke_registry_wandb_finish/`.
- 2026-07-02: The same registry `hope_forehand/backhand:v4` artifacts used for the smoke were later verified as directionally wrong for real training: frame-0 pelvis yaw is 82.03/85.92 deg and strike velocity is +Y-dominant. Corrected local ignored clips were generated at `hope_training/motions/preprocessed/hope_forehand_hopex.npz` and `hope_backhand_hopex.npz`. [Pre-merge, both task YAMLs defaulted to these local files; post-merge the YAMLs default to the WandB registry aliases again — pass `motion_file=`/`motion_file_2=` to train on the corrected `_hopex` clips until a future verified registry artifact replaces them. The local v5 clips are R15-only, not that replacement.]
- 2026-07-02: `scripts/csv_to_npz.py --robot agibot_a3` now auto-aligns exported world-frame arrays into HOPE +X before saving/uploading, and `scripts/check_motion_target_alignment.py` provides a no-Isaac gate. Verification: `python -m py_compile ...` passed; `python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml` and `--yaml cfg/task/HOPEPingPongRealSensor.yaml` passed; the same check fails on old v4 as expected.
- 2026-07-02 (later): Fixed an `UnboundLocalError` in `RacketTargetCommand._resample_command` (`hope_commands.py`; the base-XY coupling branch read `motion` before assignment) that crashed EVERY `HOPEPingPong*` env reset — the working tree could not start training at all until this fix. After the fix: local-clip smoke passed (`num_envs=32 max_iterations=2 logger=tensorboard`), and a bounded verification training run passed: `hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=4096 max_iterations=300 run_name=e2e_verify_train seed=1` -> W&B run `wuj6ds9u` (https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/wuj6ds9u), mean reward -1.37 -> ~25, mean episode length 5 -> ~340 steps, `strike_success` 0 -> 0.006 in 300 iters, `model_{0,100,200,299}.pt` + ONNX exported and synced. Pipeline viability evidence on the corrected `_hopex` clips, still not a quality baseline. [This run used the pre-merge branch defaults, i.e. `target_mode: reference_perturbed`.]
- 2026-07-02 (later): Full fresh MP4 -> npz rerun in an isolated dir reproduces the shipped artifacts bit-for-bit (GMR pkl and retargeted CSV byte-identical; npz equal to `hope_forehand_hopex.npz` within 2e-7 float noise), and `check_motion_target_alignment.py --clip` passes on the regenerated clip. One env caveat found and documented: GVHMR's YOLO load needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` under torch 2.7 (see `docs/operations/setup_environments.md`).
- 2026-07-04: `motion.clip_switch_prob` (default 0.0, try 0.002) adds deploy-parity MID-swing clip
  switches through the existing wrap-resample path: per-step random abort to a different clip's
  frame 0 + fresh pre-swing hold + fresh target, robot untouched. Parity for
  `pp_reference_clock.hpp`, which flips `clip_id` at arbitrary tts when the planner re-sides the
  target — the root cause of the venue falls at 准备/正手/反手 switches. A8 post-swing capture
  stays wrap-only (aborted swings are not captured). Mech-verified via `clip_switch_count`.
- 2026-07-04: P2.4 `base_decel` reward landed default-off (`rewards.base_decel_weight: 0.0`;
  PACE-style pre-strike pseudo-speed tracking on racket→target planar distance — formula and v2
  plan in `docs/operations/run_training.md` Reward Shaping and `docs/motion_and_contract_v3.md`
  §5). Mech-verified on/off. Same commit hardened `scripts/train.py`: `task.motion`/`task.racket`
  yaml keys go through explicit whitelists and unknown keys RAISE — new yaml keys must extend the
  whitelist in the same commit (the 018467a startup-crash lesson).
- 2026-07-04 (evening): `motion.speed_scale_range` (R14 retiming) landed default-off — per-swing
  reference playback speed with the full consistency cascade (clock ×s via float shadow clock,
  reference velocities ×s, tts ÷s, racket velocity target ×s incl. HER clamp box); train-only
  (play/eval force `[1.0, 1.0]`); flag docs in `docs/operations/run_training.md`, design in
  `docs/motion_and_contract_v3.md` §2. MECH-VERIFIED on pod (2026-07-04 late): OFF run 25 it
  clean; ON run `[0.8,1.2]` 25 it clean with `Live/motion/playback_speed` per-iter env-mean
  fluctuating 0.981-1.012 — the U(0.8,1.2)/√512 signature (a dead flag would sit at 1.0000).
  Arm R14 is launch-ready.
- 2026-07-04 (late): `rewards.free_wrist_ori_mimic` (R16, franco's wrist idea) landed default-off
  and MECH-VERIFIED on pod (25 it clean; startup override log shows both
  `motion_body_ori.body_names-=right_wrist_yaw_Link` and `motion_body_ang_vel.body_names-=...`).
  Config-level: this codebase has no joint-level mimic rewards — body-level orientation tracking
  on the racket-mount link IS the face mimic; the flag filters it out of the two orientation
  terms while keeping position/linear-velocity mimic (swing path). Face is then shaped by
  `racket_normal` / ball-outcome rewards; at contract v3 the freed wrist becomes the actuator of
  the commanded-normal channel.
- 2026-07-04 (evening): v5 clips processed end-to-end ON the pod (`v5_pipeline.sh`, reusing the
  oblique pipeline + `csv_to_npz_mujoco.py`): `/workspace/shared/motions/hope_{forehand,backhand}_v5.npz`
  (56/58 frames @50 Hz, yaw re-grounded +86.6°/+83.6°→0). Strike phases CORRECTED late-night by
  franco's prior: forehand 0.673 (detector's speed-peak 0.768 is the post-contact whip — at the
  true ~2/3 contact the velocity/normal are direction-healthy, retiring the "+Y-dominant" flag),
  backhand 0.345 (matches franco's "within the first 3/7"). Lesson: speed peak != contact;
  cross-check `analyze_strike_phase` picks against the forward-velocity peak and a human prior.
  Remaining data flag: v5 reference jitter is 2-6× hopex (mean joint |acc| 5.9/15.5 vs 2.5/2.7
  rad/s²; oblique 3.5 sits between) — evidence for R16 / reference filtering, and a third
  confounder for R15 verdicts.

Done (2026-06-26 — first loop reproduced in this harness):

- The Isaac WBC training loop now runs end-to-end on this machine. Run: task `TrackingFlat`, `num_envs=1024`, `max_iterations=60`, `algo.runner.save_interval=25`, `logger=tensorboard`, `run_name=stand_bootstrap`. Mean reward improved monotonically `-4.08 -> -0.24`. Artifacts under `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_flat/2026-06-26_13-13-07_stand_bootstrap/`: checkpoints `model_0.pt`, `model_25.pt`, `model_50.pt`, `model_59.pt`, and `exported/policy.onnx` (exported via `scripts/play.py task=TrackingFlat ... checkpoint=model_59.pt motion_file=../motions/a3_stand.npz`). These prove pipeline viability only; the reference clip is a static stand, not a swing.
- Two blockers were resolved to get here, beyond EULA acceptance:
  1. **Blackwell GPU incompatibility (the real blocker).** The RTX 5090 is sm_120; Isaac Sim 4.5.0's bundled `torch 2.5.1+cu124` has no sm_120 kernels and a real CUDA matmul fails with `no kernel image is available for execution on the device`. Fixed by upgrading `hope-isaac-py310` to `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` and pinning `numpy==1.26.4` (Isaac needs `numpy<2`). Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`. `isaaclab*` keep a `torch==2.5.1` pin in metadata but are editable installs imported at runtime, so the upgrade does not break them.
  2. **WandB-only motion loading.** `scripts/train.py` fetched the motion clip only from the WandB registry. Added a local `motion_file=` override (skips WandB, mirrors `play.py`) and `motion_file: null` in `cfg/train.yaml`.
- EULA accepted non-interactively via `OMNI_KIT_ACCEPT_EULA=YES`.
- Bootstrap motion: `scripts/make_static_motion.py` generates `hope_training/motions/a3_stand.npz` (static default-pose clip, `fps=50`, `joint_pos[600,31]`, `body_pos_w[600,32,3]`) so the loop runs without the GMR/GVHMR pipeline or WandB. Placeholder reference only.

Not done:

- No accepted quality baseline is set yet; the recorded local and WandB smoke runs and the bounded `wuj6ds9u` verification run prove pipeline viability, not policy strength.
- The verified WandB smoke used only 32 envs for 1 PPO iteration and used later-rejected +Y-facing registry motions; no long unified policy training run or post-train evaluation has been accepted.
- The post-merge unified uniform-target configuration has not been run in Isaac yet; the 2026-07-02 verification runs used this branch's pre-merge `reference_perturbed` defaults, so the unified HOPEPingPong path still needs a fresh run before it can count toward a baseline.
- Uniform reachable target ranges, reward tuning, exact-strike pass rates, stable recovery metrics, and first usable baseline thresholds still need formal acceptance.
- Exact accepted run IDs, checkpoint paths, ONNX paths, and first quality metrics still need to be recorded in this gate for an accepted run.
- The corrected `_hopex.npz` motion clips are ignored local artifacts; new machines must restore or regenerate them through `setup_local_sync.md` before reproducing the recorded local-clip `HOPEPingPong*` training runs.
- The v5 R15 ablation clips have no accepted smoke, training run, or quality baseline; forehand is hand-verified at phase 0.673, while backhand phase 0.345 is still unverified.

## Current Verification Commands

Account-free reproduction (no WandB, no motion data — the 2026-06-26 path). On a Blackwell GPU apply the
torch-cu128 fix first (see [run_training.md](../operations/run_training.md#blackwell-rtx-50-series-sm_120-torch-fix)):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
  --output_file ../motions/a3_stand.npz --frames 600 --fps 50
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=1024 max_iterations=60 algo.runner.save_interval=25 \
  logger=tensorboard run_name=stand_bootstrap \
  motion_file=$(pwd)/../motions/a3_stand.npz
```

GPU/Isaac environment, after `source setup_train_env.sh` and restoring/generating Step 9-12 local motions:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke

hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=hope_smoke
```

Note: if the local Step 9-12 clips predate the 2026-07-02 HOPE +X alignment fix in `csv_to_npz.py`, use the
corrected `hope_forehand_hopex.npz` / `hope_backhand_hopex.npz` clips (or regenerate);
`scripts/check_motion_target_alignment.py --clip <npz>` verifies alignment without Isaac.

Record the startup lines from `scripts/train.py` showing source provenance and applied overrides. For
an accepted run, also record the registry artifact or local `motion_file`, WandB run ID when logging to
WandB, checkpoint path, exported ONNX path, and exact-strike pass metrics.

### Local Harness Check 2026-06-25

Commands run from the repo root:

```bash
command -v conda
command -v distrobox
command -v docker
nvidia-smi
python3 hope_training/whole_body_tracking/tests/test_table_tennis_geometry.py
bash -lc 'cd hope_training/whole_body_tracking && source setup_train_env.sh && hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"'
```

Results: the ignored A3 Isaac asset was restored locally from tracked Agibot materials and checked for
mesh references (`86` references, `0` missing). The host table-tennis geometry test passed (`6/6`;
torch aerodynamics skipped because host torch is unavailable). Later checks created `grasping` from
`docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` and verified it sees the RTX 5090 and `nvcc`.
`external_repos/IsaacLab` was cloned at tag `v2.1.0` / commit `21f7136`, and the local
`hope-isaac-py310` env now imports `torch 2.5.1+cu124`, `hydra 1.3.3`, `onnx 1.16.1`, and
`onnxscript 0.3.2` from inside `grasping`. `pip check` reports no broken requirements. Importing
Isaac/Kit reaches the NVIDIA Omniverse EULA prompt; no Isaac smoke test was run before explicit EULA
acceptance.

Additional local motion-environment checks from 2026-06-25:

```bash
cd hope_training/GMR
PYTHONNOUSERSITE=1 /home/agiuser/miniconda3/bin/conda run --no-capture-output \
  -n hope-motion-py310 python -c "import general_motion_retargeting, mujoco, smplx, torch"
```

Result: GMR commit `bb1bbe4` imports in `hope-motion-py310` with `torch 2.12.1+cu130` and
`mujoco 3.10.0`. GVHMR commit `6ec3ca3` is cloned but not installed because its requirements pin
CUDA 12.1-era `torch==2.3.0+cu121`/`pytorch3d` wheels that are not accepted for this RTX 5090 host
without a compatibility pass.

## Risks

- Training may fail before policy quality can be evaluated because of asset, observation, or reset issues.
- A weak first policy is still useful, but only if metrics and failure modes are recorded.
- Copying HITTER reward assumptions blindly may hide A3-specific limitations.
- TTRL can change upstream; record the source commit if it informs a training change.

## Next Steps

1. The fresh `SZ` line is closed as completed/rejected: four-seed stability failed at both
   milestones (model-2000 `83/100/100/20`, model-4000 `50/88/98/0` with 21 physical root falls on
   seed 4), so no fresh SZ checkpoint is handed to G06. The next handoff path is the
   balance-temporal 24-cell matrix winners (diagnostic lineage; see
   `docs/experiments/2026-07/EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md`), with the winning
   mechanism then re-run for rehabilitation on the exact-lineage chain (the qdot treatment
   `model_1000` family: fresh lineage=1, exact schema-3 hard contract).
2. Re-run the unified `HOPEPingPongDeployParity` smoke and then the real unified forehand+backhand training run under the post-merge uniform-target defaults, from the corrected local `_hopex.npz` clips (`motion_file=`/`motion_file_2=`) or after uploading future verified registry artifacts and recording their aliases. Do not use the v5 R15 clips as product defaults.
3. Set measurable acceptance metrics for first usable baseline: fall rate, racket error at strike, physical recovery after clip wrap, and command latency assumptions.
4. Record exact local motion paths or registry artifacts, WandB run IDs when used, checkpoint paths, and ONNX export paths; evaluate the trained checkpoint/ONNX from the W&B run and record exact quality metrics and failure modes here.
5. Watch the exact-strike pass rates (`strike_pos/vel/normal_pass_exact`, `strike_composite_success_exact`) during long training under the uniform default; if a run opts into `target_mode=reference_perturbed`, also watch `ref_perturb_scale`, since that mode widens the target distribution only through its success-gated curriculum.
6. Run `scripts/sync_external_repos.sh` before using TTRL for comparison, and record the source commit for any extracted idea or config.

## Audit update 2026-07-10: formal training lineage

- Schema-v3 checkpoints/ONNX now bind instantiated action/joint order,
  decoder, PD, actuator integration, armature, effort/velocity limits, PhysX
  friction semantics, q-des limits, timing, observation/body layout, motion
  lineage and the exact racket control point. Legacy or override resumes
  cannot acquire an exact lineage merely by being re-exported.
- Motion kinematics schema 2 now declares rigid-point semantics and binds the
  complete articulation `body_names` column order. Isaac-compatible references
  store link-origin positions and COM-point linear velocities; old V5/MuJoCo
  files whose velocity is `d(link position)/dt` are rejected for formal use
  until explicitly migrated. Schema 1 is exact-ineligible because it lacked
  body order. `make_static_motion.py` also writes the full contract, because an
  all-zero clip cannot reveal its point semantics from content.
- Each clip FPS must be a finite positive scalar, every clip in a unified run
  must match, and the result must equal `1/env.step_dt`; schema-v3 records the
  per-clip FPS, full articulation order and selected-body index/name mapping.
- Actual racket speed uses the link-origin channel and target speed is derived
  from the same site-position path. `clean_reference_strike_velocity=false`
  is rejected rather than falling back to a COM/site mixture.
- A1 position/velocity/face/sign now traverse one atomic delay/drop message and
  reset clears held/drop/bias state. VirtualBall explicitly pins the intended
  `4/0.5/0.5` position/velocity/normal shaping and zero foot-orientation term;
  historical 10/5 and `-0.3` runs retain their saved config provenance.
- V5hLs target speed is still an `80 ms` (`+-2` frame at 50 Hz) average, not
  verified instantaneous contact truth. Contact frame and `+-1/+-2` window are
  preregistered ablations before a professional-transfer conclusion.
- Existing A3 checkpoints inherit uncalibrated PhysX joint-friction
  coefficients. The next formal wave needs zero-friction vs calibrated-
  friction controls; do not silently rewrite old checkpoints.

Fresh formal artifacts must be exported from the current schema and current
motion assets. Old ONNX files remain diagnostic only.

## Phase-1 local integration audit 2026-07-11

The local Phase-1 snapshot was preserved at
`codex/integrate-local-ablation-20260711@30f4652`, but its formal Isaac
evaluator was not merged.  It encodes question identity as packed
`(clip,row)` integers, consumes separate per-side sequential cursors and tries
to inject an exam bank into a command that the current schema-v3 stack
correctly constrains to the train split.  That conflicts with the production
content-addressed question IDs, immutable schedule and per-attempt seed.  With
many vector environments it can also exhaust a small side bank during one
reset.  These are protocol differences, not merge-conflict cosmetics.

The production MuJoCo BankExam remains the only **bookable** score path until
the new Isaac companion leg passes its runtime canary.  The companion adapter
is now implemented in `scripts/isaac_bank_exam.py`: it restores the saved
`env.pkl`/`agent.pkl`, verifies termination and checkpoint contracts before
`gym.make`, applies a content-addressed nominal evaluation profile, and assigns
one independently validated schema-v3 exam row to each environment through an
explicit runtime seam.  The saved training command and its train-split bank are
not replaced.  Both simulators can consume the same balanced schedule JSON,
content IDs, hold values and per-attempt action-noise seeds.  Its first runtime
acceptance test remains a fixed M3f/M2/G1 canary
of ten questions per side.  Every report cell uses a fixed question prefix;
if any prefix attempt is censored, the cell is invalid rather than filled by a
later question.  Only `noise_scale=0` survivors proceed to 50 questions per
side, continuous play, noise and a second seed.

The dependency-light review also closed four first-action/runtime hazards
before Pod launch: the nominal default-joint capture startup event is preserved
with randomization disabled; external injection refreshes actual racket FK and
strike timing before actor action 0; command resampling clocks are frozen; and
the shared hold contract is exactly `H` stand actions then raw frame 0. The
termination parser is schema v3 and records current hold-aware tracking guards
with `ignore_hold=true`. Actor observation histories greater than one sample
remain unsupported and fail before rollout rather than being double-advanced
by the release-frame refresh.

Three independent utilities were retained from the snapshot:

- `scripts/audit_runpod_terminal_runs.py` inventories historical terminal
  checkpoints and prints judge commands without executing them;
- `scripts/termination_contract.py` freezes timing and termination semantics
  from a saved run and is now checked before Isaac environment construction;
- `scripts/virtual_return_scorer.py` specifies the Isaac 10 ms RK4 scorer and
  ball-centre contact plane; MuJoCo now delegates its actual and counterfactual
  return scores to this implementation without changing the physics-hash-bound
  `venue_ball_sampler.py`.

The shared schedule and adapter tests are reproducible from
`docs/operations/build_and_test.md`. Local verification on 2026-07-11 passed
`67` adapter/audit tests with one optional Torch parity skip, `85` formal CPU
contract tests, and `141` unique tests in their combined contract run with the
same one optional skip. M2's Isaac leg is now verified below; Pod M3f/G1 and
the companion same-paper MuJoCo legs are still required, so this gate remains
`Partial` and historical checkpoints remain diagnostic-only.

The first M2 q1/side Kit smoke on Pod1 reached `gym.make` and then correctly
failed on the current MotionLoader's link-origin-vs-COM guard: the historical
v4rg `_cal` files are untagged and their `body_lin_vel_w` is numerically the
finite difference of link-origin position. The exact path remains unchanged
and still refuses those files. The historical evaluator now has a separate
default-off `allow_legacy_link_origin_velocity` seam, enabled only after the
explicit inexact preflight detects that content signature; the profile records
motion paths/SHA and the resulting legacy command semantics. This preserves
the checkpoint's old input meaning for ruler diagnosis without making the
motion eligible for fresh training or a bookable score. The next retry passed
the motion and legacy-bank loaders, then found a normal Python pickle-evolution
gap: the old `RacketTargetCommandCfg` predates `rally_legacy_metrics`. The
inexact path now fills only dataclass/configclass fields that have an explicit
current default and records every filled field; exact evaluation refuses any
such hydration. The subsequent retries are recorded below.

That rerun reached complete environment and policy construction before the
historical checkpoint exposed four zero observation-normalizer `_std` entries.
This is valid for the saved rsl_rl implementation because inference uses
`(x - mean) / (std + eps)` and the configured `eps` is `1e-2`. The inference-
only compatibility loader now accepts finite non-negative std values only when
epsilon is finite and positive; it continues to reject negative/non-finite
scales, zero/missing epsilon, dimension mismatch and missing actor state. This
is covered by a CPU regression. The next two retries below close the remaining
writer issue and regenerate the same q1 cell.

The next identical-paper retry completed both Isaac attempts and failed only
while assembling source provenance: a positional `Path.parents` index treated
`.../hope_training` as the checkout root, producing a duplicated
`hope_training/hope_training/.../virtual_ball.py` path. Repository discovery
now uses the checkout's venue-physics and whole-body-tree markers, and a CPU
test requires both hashed source paths to exist. The scorecard must still be
regenerated; no partial in-memory result from the failed writer is accepted.

Retry 4 regenerated the q1 cell successfully at commit `a619aa4`. The result is
valid and uncensored, with the exact bank SHA, schedule SHA
`7809555811788675a26705deb9495159210c6f449b17aeb96161d73ecc34160a`
and ordered question IDs from the supplied paper. Both `hold_steps=0` attempts
were retained as guard-reset failures (0/2); nothing was replaced. The
quota-10 paper then completed all 20 attempts with deterministic nonzero holds,
20/20 exact reaches and hits, no falls/guards/censoring, and 16/20 returns
(forehand 6/10, backhand 10/10). The JSON SHA is
`e625a09c31931a5c4cdcd8118f96ddc351b9eb0f2cad59cf265f26107eb787fc`.
This is successful runtime acceptance evidence for the historical M2 Isaac
leg, but `evaluation_contract_exact=false`; M3f/G1 and the same-paper MuJoCo
legs remain required before this gate can move beyond `Partial`.

M2 then advanced to the fixed 50-per-side slice. The first execution was
discarded because a concurrent task fast-forwarded the checkout during the
cell, so its end-of-run source hashes could not prove the code that had been
loaded. The rerun held `c69ff13` fixed and produced 100 finalized uncensored
rows with the supplied schedule SHA
`9d1a1d601b098f93ab151a9d9dfabf6a92a81b4b2896230bfebf7483e20324cd`:
100/100 exact reaches and hits, 86/100 returns (forehand 36/50, backhand 50/50),
no fall and no guard reset. Valid JSON SHA:
`723322b469b105282506fd7b2536e79bf6cd24e338cecd50536296f773015a01`.

M3f and G1 completed the same quota-10 paper contract at fixed checkout
`c69ff13`: M3f returned 20/20, while G1 returned 10/20 with backhand 0/10.
Both had 20/20 exact reaches/hits, no physical falls or guards, complete
uncensored ledgers, and exact bank/schedule/order equality. G1 stopped at the
known-bad canary. M3f's 50/side clean result was 99/100 returns with one
zero-hold forehand tracking guard; 5% action noise remained 99/100 and schedule
seed 1 was 100/100. M2 was 86/100 clean, 85/100 at 5% noise and 91/100 on
schedule seed 1. Full artifacts and hashes are recorded in
`docs/PHASE1_SCHEMA3_RESULTS_2026-07-11.md`. These cells validate the runtime
adapter but remain historical/inexact and cannot close this gate.

### Phase-1 main-matrix closure and fresh-lineage preflight (2026-07-11)

The historical main matrix now uses the accepted same-paper adapter rather
than old scorecards. R1b (two policy seeds), R5b (two policy seeds), C1, G2
and M2f3 completed ten clean questions per side in both simulators. R5b seed
1/2, G2 and M2f3 stopped because MuJoCo forehand returned 0/10. R1b reached
q50 but both policies collapsed to 3/50 MuJoCo forehand returns, versus 45/50
and 40/50 in Isaac. C1 completed q50 clean, noise and second-schedule cells:
Isaac clean was 46/50 forehand and 50/50 backhand; MuJoCo clean was 40/50 and
10/50. All ledgers were complete and uncensored. The full tables and hashes
are in `docs/PHASE1_SCHEMA3_RESULTS_2026-07-11.md`.

C1 is therefore an additional historical diagnostic survivor, not a formal
winner. M3f remains stronger in MuJoCo and carry-state continuity. Every
checkpoint above still lacks schema-v3 training lineage and records
`evaluation_contract_exact=false`.

Fresh training also needs an explicit plant control. The default A3 config
continues to preserve its historical, uncalibrated PhysX joint-friction
coefficients. A new default-off `task.plant.zero_joint_friction=true` override
zeros every actuator before `gym.make`, so the saved env and schema-v3
training contract bind the actual all-zero vector that formal MuJoCo can
reproduce. False/absent is a no-op, unknown keys and malformed booleans fail
loud, and legacy warm-starts remain exact-ineligible. An isolated Pod worktree
passed the override and schema-3 contract tests (`60 passed`); Hydra composition
accepted the declared leaf and rejected a misspelled parent. The true Kit
smoke additionally requires the post-`gym.make` 31/31-zero assertion to pass.
The fresh
migrated-motion/bank runtime smoke and training launch are still required, so
this gate remains `Partial`.

The motion preflight subsequently found and fixed a real body-column mismatch.
The old source order cannot be reused as the live Isaac articulation order;
the migration now takes `--target-body-order`, reorders all four body arrays by
name, and only then converts link-origin velocity to COM velocity. The first
incorrect outputs are quarantined. The corrected v4rg pair is schema 2 at
50 Hz with the tracked 32-body runtime order. Its schema-v3 train/exam banks
share family `b21c161a...28ad5`, have disjoint question IDs and passed every
strict Torch/physics/loader check. Exact paths, counts and full hashes are in
`docs/PHASE1_FRESH_LINEAGE_2026-07-11.md` and the tracked asset manifest.

Pre-launch contract review also closed a diagnostic-export trap: a legacy
continuation still writes a complete schema-3 execution sidecar, but its
inexact motion must not be rejected before it can produce a diagnostic ONNX.
The exporter now requires structural validation plus the checkpoint/sidecar
SHA binding and emits `training_contract_exact=0`; a checkpoint claiming exact
lineage still goes through the stronger schema-2 motion gate. The hard contract
also binds whether face command is enabled and which face pairing is selected.
`scripts/launch_phase1_20260711.sh` hashes every parent/motion/bank and pins the
matched M3/M2 controls plus two fresh seeds. The full 179-D Kit construction
smoke has now passed with hard-contract SHA `3a3b3d95...b9972`. All four
causal continuations and both fresh seeds reached their first PPO iteration at
checkout `6d93bcb`; first checkpoints bind the matching schema-3 sidecar, with
inexact lineage for resumes and exact lineage for fresh runs. Within each M3
or M2 pair, recursive contract diff reports only `face_command_pairing`.
Training is still running and no terminal checkpoint or score exists, so G05
remains `Partial`.

### Phase-1 breadth and checkpoint-curve correction (2026-07-11)

The first launch occupied six GPUs but ran only one 4096-env process per GPU.
That is not the established breadth policy. Historical measured operation is
four jobs per 5090 (about 22--23 GiB total), with serialized Kit boots and a
75-second stagger. The reviewed scale-out is now 24 arms: a second paired
continuation seed for each M3/M2 old-vs-S1 family plus a four-seed fresh 2x2
factorial over face pairing and zero/non-zero plant. Only the fresh
`shared_plus_y + zero-friction` (`SZ`) cell is the pre-registered formal target;
the other cells are causal diagnostics. The exact assignment is tracked in
`configs/phase1_scaleout_matrix_20260711.json`.

Waiting for the terminal checkpoint was also contrary to the recorded
checkpoint policy. The first curve workers target causal `17000/18000/19000`
and fresh `0/1000/2000` checkpoints, with one Isaac export at a time per Pod
and CPU MuJoCo exams allowed to overlap. Their first preflight found a missing
ignored A3 asset link in each detached worktree; after linking the frozen
training asset, a second preflight wrote ONNX but exposed a buffered/missing
success handshake. The next preflight reached the normalizer sidecar and found
four saved `_std=0` constant features. Runtime already evaluates them with
`eps=0.01`; the writer now matches that contract (`std>=0`, `std+eps>0`) while
negative/non-finite values remain fatal. All failed batches are retained and
none is a checkpoint score. `judge.sh` now forces unbuffered export output for the retry. Later points follow
the 1000--2000 iteration schedule and densify around a measured peak. The
worker records checkpoint/evaluator hashes and never signals a training
process. Both scale-out roles (three layers of three arms on each Pod) and all
18 initial checkpoint jobs passed dry-run input/hash/path checks. Actual curve
results, the corrected retry and the remaining 18 first-iteration contracts are still pending, so
the gate remains `Partial`.

The following retries reached the CPU evaluator and separated two more evaluator faults from
training quality. Both Pod venvs had `onnxruntime` but not the `onnx` package required for formal
graph checking; they now pin `onnx==1.22.0`, and checker plus runtime inference pass. Fresh exact
checkpoints then failed before rollout on only `2.71e-9` armature disagreement caused by float32
metadata versus float64 MJCF parsing. A later retry found the analogous `3.0517578e-6` residue at
the `118.2` ankle effort limit. The formal gate now compares exact float32 grid identity instead of
using a fixed tolerance, with midpoint/next-grid regressions that reject material plant differences. The six
original trainers remained live and the frozen checkouts stayed clean. None of these evaluator
failures is counted as a model result, and G05 remains `Partial` pending the corrected fresh curve
and layer-by-layer scale-out proof.

The scale-out proof is now complete, although training/results are not. Both Pods reached four
4096-env trainers on each of their three RTX 5090s. The full-pool snapshot used
`22.9--23.2/32.6 GiB` per card at `87--97%` utilization, with `840/904 GiB` host RAM still
available. All 24 accepted arms reached a first PPO iteration; every first checkpoint is finite and
its embedded contract SHA matches the adjacent schema-3 sidecar. Pod 1 LZ seed 3 had one
pre-contract scene-start `malloc` abort; its log/launch-state SHAs are retained, the process exited
itself, and an unchanged single-arm retry (PGID `1354525`) passed. The failed boot is not a 25th
experiment. G05 remains `Partial` because periodic curves and terminal verification are incomplete.

The first corrected formal curve is now real training evidence rather than an evaluator preflight.
At clean q10 on the same immutable paper, fresh `SZ` seed 1 returned
`0.00 -> 0.50 -> 0.90` at checkpoint `0/1000/2000`, and seed 2 returned
`0.00 -> 0.50 -> 1.00`. All six jobs completed `rc=0` with exact schema-v3
evaluation. This proves why checkpoints are tested during training, but the
10-per-side prefix remains direction-only and cannot authorize stop/promotion.
The original causal 20000 screens also completed: M3 old/S1 was `0.45/1.00`,
while M2 old/S1 was `0.50/0.50`; causal results remain inexact diagnostics.

Periodic coverage now extends to all 18 newly launched arms. Deterministically
generated per-Pod causal/fresh manifests cover 142 additional clean q10 jobs,
with milestone-major barriers and separate queues so a causal terminal does not
block a fresh 2000-point screen. Generation and current worker compatibility
pass locally (`7 passed` including the venue timing and existing worker tests).

A separate audit found that the live pool is continuous only in the slow
complete-clip sense. The bound motions plus hold produce same-player strike
intervals around `2.90/3.75/4.60 s` (q10/median/q90), versus a conservative
venue A-B-A sample at `1.757/1.903/3.356 s`. The next target currently appears
only at clip wrap, later than the measured opponent-hit event. Therefore these
24 arms cannot satisfy the arbitrary-time continuous-play acceptance item.
They remain unchanged; an event-driven `T0/T1` timing pair, longer/opportunity-
count episode and new hard-contract timing fields are specified in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`. G05 remains
`Partial` pending terminal/paired q50 evidence and that separate continuity lane.

### New motion-library intake, TOPP and recovery lane (2026-07-11)

Ten private Franco/v6/v7 air-swing videos are now registered by the tracked,
content-addressed `configs/motion_video_intake_20260711.json`. Local and Pod1
copies passed byte/hash/media validation. `scripts/audit_motion_video_intake.py`
the structural `scripts/audit_gvhmr_result.py`, and the memory-gated
byte-exact gzip source archive `docs/experiments/archive/run_motion_video_gvhmr_queue_20260711.py.gz`
and the tracked result binding are covered by dependency-light
tests. Pod1 queue PID/PGID `1383735` completed all ten reconstructions after
GPU1 naturally crossed the `19000 MiB` launch gate. It processed the Franco
forehand-block item first, then the remaining Franco and v6/v7 clips, and was pinned to GVHMR `6ec3ca3`, a
clean worktree, the full checkpoint/body-model tree and motion-Python freeze;
each result matches the source frame count and finite SMPL tensor shapes.
The queue ran outside the frozen Phase-1 checkout. Its queue-state SHA and all
ten result/audit hashes are tracked in
`configs/motion_video_gvhmr_results_20260711.json`.

The next CPU-only stage is also complete as a **diagnostic**, not as a motion
promotion. A repo-owned serial GMR queue required a clean source checkout at
`aabea2eee4be4bc16d4be17dac5ffa85e5a31539` plus a verified recovery bundle,
kept frame-zero warm-up enabled, and retargeted all 10/10 items to finite
30 Hz, 31-DoF A3 pickles in 52 s. Every item converged below
`max|dq| < 1e-4`; the exact source/output/log/audit/tool/environment bindings
are in `configs/motion_video_gmr_results_20260711.json`. These outputs retain
per-video GVHMR body betas and therefore carry
`body_shape_contract=diagnostic_video_betas` and
`formal_eligible=false` by construction.

A deeper read-only replay of the Franco forehand-block pilot found useful and
blocking evidence at once. Joint order matches the canonical 31-joint MJCF,
all joint samples are inside limits, 30 Hz finite-difference speed stays below
the URDF limits, and 641 sampled/interpolated canonical-MuJoCo poses reported
zero robot self-contact. That finite-substep test is not a continuous
collision certificate and the MJCF lacks table/net geometry. More
importantly, all 65 frames penetrate the floor, with the lowest collision geom
roughly `7.7--8.4 cm` below zero. The current root trajectory is therefore
blocked on ground/root calibration followed by repeat collision, dynamics and
table/net clearance gates. A repo-owned `scripts/ground_gmr_pkl.py` now
implements the first step without directory scans or in-place edits: it binds
the exact input and canonical MJCF SHAs, computes the lowest world-z support
over enabled robot collision geoms at each source frame, applies one constant
root-z translation, and emits a no-clobber pickle plus SHA-bound report. Real
canonical-MJCF mesh tests pass. Pod1 then ran the tool no-clobber on all ten
diagnostic GMR outputs. Original discrete-frame minima were
`-0.08072..-0.08716 m`; every per-motion fixed root-z shift left a global
minimum of about `10 um`. The ledger
`configs/motion_video_gmr_ground_results_20260711.json` binds all ten
input/output/report SHAs to tool `db5bd167...`, canonical MJCF
`2ab1cd31...` and compiled collision digest `18e7f6ff...`. These remain
per-video-betas diagnostics: inter-frame ground/self collision, dynamics,
table/net clearance, canonical betas and all later gates remain open, so none
is eligible for schema-2 promotion or RL yet.

The upstream body-shape normalization is now materialized separately. A
CPU-only no-clobber tool gives each of the ten videos one equal vote and writes
one shared 10-D beta vector (SHA `a03f1642...9cc6`) into ten new GVHMR PTs;
all non-beta semantic digests remain bit-exact after save/reload. The result is
bound in `configs/motion_video_canonical_betas_result_20260711.json`. There is
no measured performer height: GMR's `1.73066 m` value is explicitly only its
beta heuristic, so the artifacts remain diagnostic, non-calibrated and
formal-ineligible. The historical preregistration's unbound guess that the
loader padded six zeros has been revoked: clean GMR `aabea2e` loader SHA
`2737f472...5de2` actually selects
`betas[0].detach().cpu().numpy()[:10]` with no padding.

A body-shape-aware CPU-only no-clobber queue then retargeted all ten
canonical-beta PTs in 48.7 s (PID/PGID `1442090`). All outputs are 30 Hz,
31-DoF and finite; frame-zero warm-up converged in 16--29 rounds with final
max `|dq|=6.88e-5..9.76e-5`. GMR remained clean and no GPU was allocated.
Exact source/output/log/audit bindings are in
`configs/motion_video_canonical_gmr_results_20260711.json`. These remain
diagnostic: the ten new outputs still need independent no-clobber grounding
and dense collision, racket/handle-to-body, dynamics and table/net gates.

The canonical-GMR grounding prerequisite has since completed independently for
all ten inputs. Its content-addressed ledger is
`configs/motion_video_canonical_gmr_ground_results_20260711.json`; every
30 Hz source-row minimum is about `10 um` above the bound floor after one
per-asset constant root-z translation. A separate CPU-only screen then replayed
654 source rows as 5,162 finite samples at 240 Hz. It found zero ground-danger
samples, zero robot self-interpenetrations, zero racket/handle-to-critical-body
samples below the 5 mm hard threshold, and zero below the 20 mm warning
threshold. The smallest observed body clearance was `40.2466 mm` in
`franco_backhand_loop_a`. This is finite dense sampling, not a mathematical
continuous-time certificate, and the canonical MJCF still has no table/net
collision geometry.

The same screen extracts the official vendor-MJCF `right_racket` site centre
and local +Y face plus `mj_differentiatePos`/`mj_objectVelocity` site speed.
It deliberately does **not** report a hit phase or motion-library coverage yet.
Canonical grounding changed root z only; it did not bind GMR world to the HOPE
+X virtual-table frame, and intake mirror status remains unverified. A first v2
result that scored venue questions anyway is preserved but all of its return,
phase, selector and two-vs-four fields are revoked; only its safety subtree is
accepted, and that subtree equals v3/v4 for all ten inputs. Accepted v4 freezes a
64-question paper but records `consumed_for_returnability=false`, with every
phase/coverage/selector field null and blocked. The preregistration and compact
ledger are `configs/motion_video_gmr_phase_safety_prereg_20260711.json` and
`configs/motion_video_gmr_phase_safety_results_20260711.json`. Schema-2 plus
HOPE +X reground (or an independently verified proper-rigid transform) and mirror
semantics must land before that paper can run. No motion entered RL or hardware.

That diagnostic frame prerequisite has now passed without claiming a capture-table
extrinsic.  Ten content-bound midpoint crops show upright, unreflected Chinese
background labels; independently, every canonical GMR is right-arm dominant by
at least `9.98x` versus the left arm (`5x` preregistered threshold).  Each clip's
proper-rigid matrix is derived only from frame-0 pelvis XY/heading and the audited
ground plane, mapping the root to the HOPE origin/+X while preserving z.  The
matrix set was frozen before scoring.  The target is explicitly the standard
counterfactual HOPE virtual table, not the room in which the air swings were
recorded.  Evidence is in
`configs/motion_video_gmr_frame_contract_results_20260711.json`.

The v5 CPU runtime then consumed the unchanged 64-question paper (result SHA
`c299b7a0...`) and reproduced every v4 dense-safety subtree.  Exact zero-retarget
coverage is `0/64` for all motions and libraries, with zero common support, so it
cannot choose two versus four actions and must not be paraphrased as "all motions
are ineffective."  Intrinsic relocation-only evidence retains Franco backhand
loop B (`32/32`, phase `0.5444`) and C (`27/32`, phase `0.5155`) as spatial-retarget
candidates; A is `1/32`, all others zero.  None is TOPP-eligible yet.  TOPP remains
paused until explicit spatial retarget, schema-2/L0/L1, table/net and dynamics
gates; final motion/library acceptance belongs to AgiBot vendor MuJoCo
Gate3 runtime/stability first and Gate3B no-reset behavior scoring second.  Compact ledger:
`configs/motion_video_gmr_phase_counterfactual_results_20260711.json`.

The 2026-07-13 interpretation is stricter: motion effectiveness is the motion's
own safe contact-time manifold crossed with a compatible incoming-ball/stroke
question family and a legal whole-trajectory `SE(2)` stance. B (`frame 49`,
`32/32`, nearest old question `0.165 m`) and C (`frame 50`, `27/32`, `0.237 m`)
fit only the `0.30 m` translation-norm bound and remain candidates. Forehand waits on
the roughly `170 deg` face-sign ambiguity, and block motions need a block-specific
paper. Schema-2/L0/vendor-L1 self-hit/full table-net swept clearance `>=5 mm`
grants training eligibility, not evidence of return effectiveness.

The next spatial step is now preregistered and mechanically checked, but has
not been promoted or run against restored private evidence.  Plan
`configs/motion_video_spatial_retarget_prereg_20260712.json` (SHA
`d8c918ac...5a9f`) keeps all ten motions on every matching immutable question;
the B/C intrinsic result affects ranking only.  R0 permits translation only;
R1 permits the frozen yaw grid `[-10,-5,0,5,10] deg` plus translation.  Each is
one ground-preserving proper SE(2) transform applied atomically to the entire
motion: no z, scale, reflection, joint or per-frame edit, and no capture-table
extrinsic claim.  The station envelope is norm `0.30 m`, `|x|<=0.20 m`,
`|y|<=0.30 m`.  The CPU tool/test contract passes `7` tests and rejects skipped
assets, unsafe/wrong-side frames, out-of-envelope stations, clobbering and
incomplete certificates.  The laptop lacks the exact 792,241-byte full v5
result, and the current manifest deliberately records
`certificate_bundle_preregistered=false`; therefore it can produce only
proposals after exact evidence restore.  Promotion requires candidate-bound
runtime-order schema-2 materialization, L0 PASS, vendor-MJCF L1 PASS and a
whole-trajectory table/net swept-clearance PASS with at least `5 mm` margin.
Dynamics/balance and TOPP remain downstream; Gate3 runtime/stability must precede Gate3B no-reset
behavior scoring, and RL/hardware remain blocked.
No GPU, Pod, trainer or hardware was touched.  Reproduction and the restore
boundary are in `docs/operations/run_motion_spatial_retarget_screen.md`.

This is not yet a training result. The videos contain no ball/table/contact
truth; final-pixel mirror status is now verified but monocular depth/capture-table
extrinsic remain unverified, and the three Franco
backhand-loop recordings are candidates for one semantic action rather than
three new action classes. Every candidate must pass A3 schema-2 conversion,
finite/limit/endpoint checks, vendor-MJCF self-collision and racket/handle-to-
body/table/net swept-clearance gates before returnability phase scanning.
Native/TOPP v3 assets are then compared on the same spatial path and strike
constraints, with both outputs re-audited.

Formal two-vs-four-action training is blocked on a dynamic clip catalog and a
shared global-question axis: current upper layers still encode two clips and
sample clip before question. The preregistered comparison therefore has two
separate fairness papers, equal total transitions and equal per-action
exposure, plus common-action non-regression and a train-fitted frozen stable-
action selector. Between-shot work uses strike/absorb/recover/ready states and
a tolerant ready set rather than a dense reward back to an exact frame 0.
Repository Hitter evidence warns that post-strike brake/ready rewards can
propagate through GAE and damage the hit; recovery is first isolated as an
option/bridge, then evaluated on event-driven T0/T1 timing. Full rationale,
literature and stop/promote gates are in
`docs/research/motion_library_topp_recovery_2026-07-11.md`. G05 remains
`Partial`; no new motion is authorized for hardware.

The first real continuation terminal also corrected a queue-index bug. Pod2
M2-S1 exited normally with `model_20998.pt` (`iter=20998`), not 20999; all
1,762,715 floating checkpoint elements are finite, checkpoint SHA is
`574ff640...0049`, and embedded/adjacent contract SHA is
`7268eb38...28f2`. Its schema-3 legacy-motion lineage is correctly inexact.
The cadence and scale-out causal manifests now use 20998, with a generator
regression. Only the affected waiting-worker PGIDs were replaced; fresh
scale-out workers and training arms received no signal. After the later split
and global hardening transaction, the six current original/scale-out worker
PGIDs are Pod1 `1432280/1432292/1432304` and Pod2
`200706/200718/200730`. Only their recorded legacy worker PGIDs received
TERM; trainers and judges received no signal. Five available old states were
rejudged rc=0 under manifest/job/job-contract bindings rather than silently
reused. The full transaction ledger is
`configs/phase1_global_curve_worker_hardening_result_20260711.json`.

Pod1 M3-S1 has now reached the same terminal integrity point. Its accepted
`model_20998.pt` has SHA-256
`a924048810aebda864bbf1f7b156ef4c4aa2c60ec4d65da6ae8977833deaa21e`,
contains `iter=20998`, and has zero non-finite values across 1,762,715 floating
elements. The embedded schema-3 contract SHA matches the adjacent contract
(`d3ff715e...29d9ce`), and the run log contains the contiguous 4,000 records
`16999..20998` with no NaN/Inf/traceback/OOM/malloc/killed signature. The
launcher did not persist an OS exit code, so the terminal checkpoint,
contiguous log, absent process and zero error signatures are the recorded
completion evidence. Its legacy-motion parent keeps this result causal and
inexact. M3-old has since naturally produced its own finite
`model_20998.pt` and exited. Its checkpoint SHA is `320b77c9...417a`,
embedded/adjacent contract SHA is `7542c59b...d941b`, and lineage remains
causal/inexact; the complete audit is
`configs/phase1_M3_old_terminal_audit_20260711.json`. The paired immutable
terminal q10 then completed on schedule `7a908142...d614`: M3-old
FH/BH/aggregate=`0.50/0.40/0.45`, M3-S1=`1.00/1.00/1.00`, aggregate delta
`+0.55`. This is a 10-per-side direction screen, not a stop/promotion
decision. Its ledger is `configs/phase1_M3_terminal_q10_pair_20260711.json`.

The triggered shared-schedule MuJoCo q50 has since completed. Both terminal
checkpoints consumed the same K=100 schedule semantic SHA
`949eb196...8fc0`, 50 attempts per side, seed 0, no noise and no censored
attempts. M3-old returned FH/BH/aggregate `31/50,11/50,42/100` and contacted
`89/100`. The raw ledger resolves the summary's legacy `fell=9` union into
**one physical fall plus eight non-physical guard resets**, not nine physical
falls. M3-S1 returned `50/50,50/50,100/100`, contacted `100/100` and had zero
such terminations. The aggregate paired delta is `+0.58`.
This selects M3-S1 only inside this legacy swing-family causal diagnostic; both
lineages and evaluations remain inexact and no formal/deployment/hardware
promotion follows. The first runner attempt judged M3-old but rejected its
own wrong result-schema assumption before starting S1; that attempt is
preserved. The corrected v2 reproduced the identical schedule bytes and reran
both cells successfully. Full hashes and limitations are in
`configs/phase1_M3_terminal_q50_result_20260711.json`. The same-paper Isaac
companion then scored both old and S1 at FH/BH/aggregate
`0.98/1.00/0.99`, delta zero. It does not reproduce MuJoCo's `+0.58`
ranking, so the cross-engine causal gate stays open and S1 can be selected
only inside that MuJoCo family/evaluator. Full companion hashes are in
`configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

The question-level forensic in
`docs/research/phase1_cross_engine_saturation_forensic_2026-07-11.md` further
shows that the two runs did not yet share an outcome instrument. Fresh model
4000's FH racket-center error is `13.15 cm` in MuJoCo versus `2.48 cm` in
Isaac, and Isaac's analytic face orientation erases M3-old's signed BH face
error (`168.15 deg` before orientation). The cross-engine gate therefore stays
open. A fail-closed 2x2 instrument-parity prereg now requires both physical and
analytic cells from both engines without changing the frozen thresholds; its
current blocker is missing Isaac post-contact physical truth.

The independent continuous-timing axis is now content-addressed but remains
launch-blocked. Its venue aggregate binds raw strikes SHA `6ad3c459...` and
records the overlapping-window, 16/21 high-ball and 2.5 s right-censor limits;
`1.903 s` is not used as a target. T0/T1 instead share a balanced engineering
event grid and freeze motion/TOPP/plant/face/reward/2-vs-4. Design validation
passes; launch validation must fail until the post-strike scheduler,
materialized schedules, continuous Isaac/MuJoCo judges, self-hit instrument,
fresh exact checkpoint and semantics-correct plant are all bound. Prereg SHA
is `2e7c4a34...2289c`; no timing arm has launched.

The training-side core is now implemented at `be5d7cf`: only an accepted exact
strike can arm the absolute event clock; reveal atomically installs the bank
row/native clip/exact hold/fixed deadline, and misses or infeasible rows still
consume the original opportunity without teleport/reset/history-noise reset.
Every timing-changing field is in the hard contract. This does not mutate the
frozen prereg or authorize launch; materialized schedules, continuous judges,
self-hit instrumentation, fresh baseline and calibrated plant remain open.

Natural terminal release also opens a separate second-wave causal paper rather
than permission to mutate the frozen 24-arm matrix. The preregistration
`configs/phase1_causal_followups_20260711.json` completes two
old-helper/S1-only/S1+guidance triangles with four arms: M3 gets missing
S1-only guidance-0 seed 1/2; M2 gets missing S1+guidance-`-0.95` seed 1/2.
All remain 4,000-update causal/inexact continuations. Their external
content-addressed launcher refuses dirty/wrong train or eval checkouts,
wrong assets/tools, a fourth occupied GPU slot, reused run names or a live
M3-old predecessor; it verifies each emitted hard-contract before starting an
independent `17000/18000/19000/20000/20998` q10 worker. The original 16999
parent is never copied beside the new sidecar or judged as the new contract.
q10 remains direction-only and q50 is an inactive separate template. After a
read-only duplicate-PID fix and a second clean validation, all four followups
launched without signalling an existing trainer. Accepted trainer/worker PGIDs
are Pod1 M3 seed1 `1409914/1410648`, M3 seed2 `1411167/1412047`; Pod2 M2
seed1 `196177/196753`, M2 seed2 `197146/197939`. All four emitted the expected
family hard-contract SHA, reached 17000 with zero logged bad signatures, and
restored the full pool to four trainers per GPU (24 live). Their first q10
screens were M3 S1-only seed1/2 `0.60/0.55` aggregate and M2 S1+guidance
seed1/2 `0.30/0.30`; these are inexact 10-per-side direction points only.

The first adjacent scale-out pair reinforces that rule. On identical K20
schedule `75aca567...51d7`, M2 seed2 old/S1 changed from `.40/.60` at 18k to
`.50/.40` at 19k, reversing the small-paper order. Both 19k checkpoints are
finite and bind iteration, adjacent hard-contract SHA and causal lineage.
They continue unchanged; no stop, promotion or q50 trigger follows. The
content-addressed curve ledger is
`configs/phase1_M2_seed2_18k_19k_q10_curve_result_20260711.json`.

The first 17k run also exposed a provenance gap before later milestones: the
launcher had deliberately pinned eval checkout `46a0ce2`, whose worker SHA
`8b980359...` predates the checked-in screen-policy/job-contract state fields.
The four judge commands and rc=0 results match the immutable manifests, but
their state JSON lacks manifest/job/job-contract SHAs. A separate replacement
contract now requires both workers on one Pod to be alive, exact-PGID,
childless and manifest-bound before any signal; it then TERM-signals only those
workers, preserves the legacy evidence, starts standalone hardened worker
`21e30153...` with a fresh state dir, and rejudges 17k before accepting the
correction. Trainers/judges are out of scope. Both Pod transactions have now
completed. Old childless workers were TERM-signalled only at the four exact
PGIDs above; hardened worker PGIDs are Pod1 `1416771/1416784` and Pod2
`198759/198771`. The correction-sidecar SHAs are respectively
`2faf88de...ffe3`, `1d6f8ba3...bae9`, `0dd02fae...d165` and
`45f4334d...0ad`. Rejudged 17k states returned rc=0 and bind manifest, job
spec and job contract SHAs plus checkpoint, judge, clean training `6d93bcb...`
and eval `46a0ce2...`; legacy state/log bytes remain preserved.

`SZ`'s target label is now explicitly scoped: it is the only current fresh
cell whose zero-friction plant can be replayed with the same schema-v3
cross-engine execution semantics. It is **not** a deployment-plant or hardware
candidate. The 2026-07-07 frozen probe already showed a zero-friction policy
degrading from virtual hit `0.9997` to `0.63` and fall `0.27` to `0.87` when
moved into the non-zero-friction plant. `SP/LP` do not resolve this: their
PhysX values are the historical unit-mismatched copy of MuJoCo constant-Nm
`frictionloss`, not calibrated friction. A new from-scratch `SC` cell requires
measured friction semantics plus separate PhysX/MuJoCo adapters and hard-
contract hashes before deployment-plant, continuous-practical or Gate3B
promotion. This does not block the same-schedule SZ q50 required for current
execution-contract model selection. `SP` is explicitly judged inexact so its
non-zero plant cannot fail the formal profile and block later SZ milestones.
The current 24-arm training recipes remain unchanged. The separately
validated repair preregistration is in
`docs/research/phase1_plant_semantics_repair_2026-07-11.md` and
`configs/phase1_plant_semantics_repair_prereg_20260711.json`; it is currently
`blocked_on_calibration_evidence`.

The 2026-07-12 provenance recheck corrected that v1 manifest's declared
repository snapshot from training-only ancestor `612f54d` to `d4ca566`, the
first commit containing all eight already-recorded source hashes. Current main
has since changed `training_contract.py` for strict face179, so current-checkout
verification now deliberately returns exit 2. This is an additional fail-closed
prelaunch blocker: a new reviewed preregistration must bind current source bytes
before any `SC` arm. It does not change the running `SZ/SP/LZ/LP` recipes, and
G05 remains `Partial`.

The 2026-07-12 follow-up adds an offline plant-contract v1 compiler without
changing any current trainer. It binds explicit units, the 31-joint order, one
latent physical model, separate engine fit/probe evidence and a calibrated
support envelope. Non-zero `dimensionless <-> N*m` conversion is impossible by
construction; only exact zero crosses that helper boundary. Runtime preparation
also rejects an out-of-support load/speed/temperature/pose request. The final
MuJoCo adapter is required to target the Agibot vendor Gate3/Gate3B runtime and
bind its MJCF, runtime source and 31-joint instantiation report; a generic
MuJoCo wrapper cannot close this gate. Tests exercise the compiler only. No
calibration artifact, runtime wiring or `SC` training arm exists, so G05 stays
`Partial` and current `SZ/SP/LZ/LP` recipes are unchanged. Interface and
commands are in `docs/interfaces/plant_semantics_contract.md` and
`docs/operations/prepare_semantics_correct_plant.md`.

That current execution-contract selection has already produced one early
checkpoint result. Fresh SZ seed1 regressed on q10 from `0.90` at 2000 to
`0.50` at 4000, then consumed one exact K=100 paper. Model 2000 returned
FH/BH/aggregate `33/50,50/50,83/100`; model 4000 returned
`0/50,50/50,50/100`. Model 2000 is retained; the arm continued unmodified at
that paper's decision time and was only later stopped by the separate 2026-07-13 operational
resource decision. Both cells were fresh/exact and had zero physical falls, but all
questions ended through the non-physical post-strike guard; this is isolated
checkpoint selection, not recovery or deployment evidence. Bindings are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`.

The fresh/exact Isaac companion reused the byte-identical K=100 schedule and
scored both checkpoints FH/BH/aggregate `0.98/1.00/0.99`, with one guard
reset and no physical falls. It does not reproduce MuJoCo's `0.83` versus
`0.50` ranking. The final earlier-checkpoint tie-break is only a rule for a
complete Isaac tie, not cross-engine support. Model 2000 remains retained
inside the MuJoCo pair; no cross-engine/formal deployment gate closes. See
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

The original cadence no longer serializes fresh milestones behind causal terminal.
Each Pod now has independent original-causal and original-fresh manifests and
state directories, matching the scale-out split. Current q10 manifests carry
both a fail-closed top-level screen policy and per-job `screen_only=true`; the
checked-in worker rejects omissions/contradictions, verifies `schedule_k`, and
binds a canonical screen-policy-plus-job contract SHA before a completed state
can be reused (while recording the full manifest SHA for audit). This closes
the documented silent parameter-reuse path while preserving the rule that only
a separate q50 paper may stop or promote an arm.

The split itself exposed a Pod-specific historical gap: seed1 `model_4000.pt`
had not existed when the old combined Pod1 worker was replaced, whereas Pod2
seed2 4000 had already been judged. The Pod1 fresh queue therefore starts at
4000 and Pod2 at 6000. Only the childless Pod1 fresh worker was precisely
restarted; no trainer or judge child received a signal.

Before copying or launching any checked-in queue, run
`python3 scripts/validate_phase1_queue_governance.py`. It validates all 142
scale-out jobs and 24 cadence plan slots, requires K20/10-per-side q10
screen-only semantics and milestone/barrier continuity, and rejects q50 from
the generic worker. For one milestone-major runtime manifest, use
`--manifest /absolute/path.json --require-readiness-barrier`.

The corrected Pod2 terminal q10 has now finished: M2-old/S1 aggregate return
is `0.40/0.35`, with FH `0/10` for both and BH `8/10` versus `7/10`. It is a
causal/inexact 20-attempt direction screen, not evidence to kill S1 or select
old; the immutable q50 follow-up remains required. Machine-readable hashes and
the paired delta are in `configs/phase1_M2_terminal_q10_pair_20260711.json`.

### 2026-07-12 reward-composition and PhysicalBall source boundary

Post-strike balance recovery, convergence to a shared ready set and arbitrary-time next-task
readiness occupy one phase and may interact, but the 2026-07-13 primary-source audit does not
classify all three as rewards. G05 does not accept three independently positive reward ablations
as evidence that their sum is optimal. Safety remains a non-compensable constraint; ready shaping
is potential progress to a set; random arrival is first a frozen environment/question/deadline
axis scored by the actual next strike. If T1 still needs shaping, run scale-matched balance/ready
`2^2` with paired seed blocks. A full `2^3` is allowed only after an independent readiness critic
is locked on separate train/calibration splits and passes a one-shot preregistered critic-gate q50
that is disjoint from sealed formal Gate3B q50, without hidden-future leakage. Surviving terms use a
constant-budget mixture plus a second total-budget level. The exact design is recorded in
`docs/research/phase1_ablation_acceleration_2026-07-11.md`; no recovery arm has launched yet.

Isaac PhysicalBall Phase-B now has a contract-bound source implementation at `612f54d` with
attempt-generation tokens, served/contact/landing validity, held publication and strict
training/T1 isolation. Focused host verification passed (`63 passed, 1 artifact-gated skip`).
This is only a source gate: no Pod Isaac runtime result exists, a clean-detached 100-row ledger
has not been produced, and moving-racket substep geometry is not yet quantified. G05 therefore
remains `Partial`; existing analytic Isaac scores cannot be relabelled physical.

The fresh SZ model-2000 four-seed exact MuJoCo q50 is also complete. Seeds 1/2/3/4 return
`83/100`, `100/100`, `100/100`, and `20/100`; seed 4 has FH/BH=`0/50,20/50`, with zero
physical falls. Median `.915` passes, but the preregistered worst-seed, spread and worst-side
criteria fail, so the checkpoint evidence is not seed-stable. At this q50 decision time all trainers
continued unchanged; the result authorized neither stopping nor promotion. The separate 2026-07-13
owner resource decision later stopped seed1/2/4 without rewriting this paper. Later same-milestone curves determine
whether seed 4 is delayed or persistently sensitive.

### Model-4000 four-seed matched paper preregistered (2026-07-12)

The next fresh `SZ` milestone is now queued without touching a Pod. It freezes
seed1/2/3/4 at `model_4000.pt`, the exact model-2000 K100 file bytes
`66e89986...71cb3`, semantic SHA `7dc6af82...ff3e`, question order
`b87e81a3...1f91`, and the same four stability thresholds. It cannot prepare
or start a q50 judge: the queue has no runtime entrypoint, and its validator
contains no SSH, process launch or signal surface. Pod1 must first produce a
content-bound finite/iteration/contract/lineage readiness audit for seed1/3,
Pod2 for seed2/4; only their exact four-seed union can create the mandatory
activation artifact. Any absent/non-finite/wrong-iter/wrong-contract arm keeps
the whole paper runtime-ineligible.

The interpretation is deliberately narrower than “family stable.” Seed4 was
`.20` at 2k; it supports delayed learning at 4k only if it reaches the unchanged
`.65` aggregate and `.50` on both sides. But seed1 4k was already known before
this preregistration to be `.50` aggregate (`FH=.00`, `BH=1.00`). Therefore the
four-seed 4k stability gate is mathematically unable to pass its unchanged
worst-seed threshold, even if seed4 recovers. This matched paper can diagnose
the seed4 trajectory and full distribution; it cannot launder the family into
a stable baseline or change training. Source validation passed `20` tests.
Queue/prereg/validator SHAs are `d4e69d91...d3909`, `ca5ea90f...bff0`, and
`e763ecb9...6cd3`; commands and the future-runner boundary are in
`docs/operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md`. G05
remains `Partial`; no Pod audit, activation, runtime or hardware action has run.

### Recovery tuple is now a structural axis, not a reward bundle (2026-07-12)

A read-only audit of the implemented T1 source and the current vendor Gate3 runner found a concrete
train/deploy mismatch. Natural wrap training installs one complete new
`position/velocity/normal/side` question tuple; T1 keeps the complete previous tuple until reveal
and then installs the complete next tuple atomically. Actor latency/dropout likewise carries all
four fields as one generation. The current 179-D deploy recovery path instead feeds a **new,
live-base-anchored position** together with the **previous strike velocity and previous strike
normal/rho**. No bound training transition creates that mixed-generation moving target. It is now
classified OOD and is not a formal arm; tuning its anchor does not repair the missing semantics.

`configs/phase1_recovery_tuple_abc_prereg_20260712.json` freezes the replacement comparison at SHA
`ca7806df...d810616`:

- A: an interruptible, content-bound safe PD/trajectory bridge into the ready set;
- B: the same actor receives an atomic canonical tuple consisting of a ready-set-selected racket
  position, zero desired velocity, neutral ready normal, rho zero and ready-phase semantics;
- C: the actor retains the complete previous tuple until the atomic next-question reveal.

Ready is a safety-and-reachability **set**, not equality to clip frame 0. It jointly requires station
and upright tolerances, low base/joint/racket speed, stable contacts and slip, joint/torque/q-des
margin, self/table/net/ground clearance, and a bounded safe start to every enabled next motion,
question family and random-arrival deadline. If the global intersection is empty, the design must
declare family-specific ready sets and an explicit transition graph rather than silently deleting
hard questions.

Existing finite, lineage-bound 179-D atomic-question checkpoints have only two narrow uses: A may
reuse one as a frozen **swing diagnostic** after bridge/handoff certification; C may run a zero-shot
coherent-tuple diagnostic. B requires fresh training because current checkpoints never saw the
zero-velocity/neutral-normal canonical tuple. A fair A/B/C causal comparison is fresh exact and
paired; C also needs fresh training before any learned random-arrival recovery claim. No old model
may be relabelled T1-trained.

A also needs a fresh checkpoint for the formal comparison: an external bridge changes who owns the
executed action and therefore changes the PPO data contract. Every tick must bind
`actor_control_mask`, executed action, shadow action, last-action observation and loss masks. Only
an actor-owned, actually executed sample has a valid policy logprob. Bridge ticks have zero policy,
entropy and value loss masks; shadow actions are diagnostic only. The actor's last-action channel
must be the exact content-bound projection of the executed bridge action, never shadow/zero/stale.
Actual bridge rewards use duration-correct `gamma^k`, collapse into the preceding actor option
transition and use `gamma^duration` to bootstrap at the next actor-controlled state unless the
simulator truly terminates; miss/infeasible rows are not fake
terminals. A prebound common budget fixes rollout env-steps, scheduled opportunities, updates,
actor-controlled samples, minibatches and epochs across A/B/C. B/C surplus samples are
deterministically downsampled without seeing outcomes; an A shortfall fails the whole paired update
instead of padding/reusing samples or running extra A steps. Evaluation keeps the full scheduled
denominator regardless of ownership masks.

The first structural paper freezes reward source/weights, total reward budget, motion, bank, face,
plant, network, observation/action schema, seeds, optimizer and random-arrival rows. It prohibits
mid-sequence robot/last-action/history/noise reset and deadline shifts. A's handoff remains blocked
until the exact executed-bridge-action projection into actor history is content-bound; shadow,
zero and stale action substitution are prohibited. Only if B/C fail ready-set acquisition without
single-strike regression may reward work begin. Random arrival remains an immutable environment
axis and actual next-task objective. Normalize balance-absorption debt and ready-set potential on
frozen rollouts, then run paired `2^2` presence/absence. A third readiness potential and full
`2^3` require separate critic train/calibration splits, a one-shot disjoint critic-gate q50 and proof
that no future tuple leaks before reveal; formal Gate3B q50 remains sealed. A
constant-total-budget simplex may follow only for surviving components, and must include a second
total-budget level because fixed total alone identifies proportions, not PPO reward magnitude.
Positive hold income is still prohibited, and safety/self-hit can never be offset by another reward.

The literature boundary is now explicit in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`: ACE supports an interruptible
near-time-optimal reset bridge and conditioned prepare posture but has no free-standing humanoid
balance debt; HITTER changes tasks only after a swing completes; SMASH's phase/recovery clips feed
offline library generation and runtime motion matching; PACE's five serves are not arbitrary
mid-followthrough reveal. These systems motivate T0/T1/T2, but none proves this A3 policy, vendor
MuJoCo plant or random-arrival contract. G05 therefore remains `Partial`.

2026-07-13 的新 reward 次序目前仍是文档级设计：现有
`phase1_recovery_tuple_abc_prereg_20260712.json` 与 validator 继续强制旧的三 reward/full `2^3`。
它们未被追写，也不得冒充新设计的 E1 证据；必须生成新的内容寻址 prereg、validator 和测试后，
才允许按“先 `2^2`、校准后可选 `2^3`”点火。这个同步缺口使 G05 继续为 `Partial`。

The pure-contract validator passes `50` red-team tests, including nested duplicate-key, non-finite
JSON, strict type identity, exact identity/time/scope and unknown-key rejection; `launch-check`
deliberately fails because
the schedules, checkpoint inventory, bridge and trajectory certificate, canonical tuple selector,
fresh checkpoints, A ownership/PPO contracts, full numeric ready contract, Isaac continuous judge,
vendor Gate3 runtime/stability judge, Gate3B behavior judge, shared Gate3/Gate3B runtime, self-hit
instrument and calibrated plant are not bound. Gate3 is the exact-C++/MJCF/plant/model first-tick and
continuous-stability hard prerequisite; Gate3B reuses that runtime with the immutable random-arrival
q50 and is the final first-strike/return-quality behavior arbiter. Commands are in
`docs/operations/run_phase1_recovery_tuple_prereg.md`. No simulator, Pod, GPU, C++, Gate3 worktree
or robot was changed. G05 remains `Partial`.

### 2026-07-12 formal train-normal envelope export

The 179-D export chain now consumes the demanded-normal rows of the exact schema-3 train bank
instead of binding only its file SHA. Native Isaac export requires the live validated
`QuestionBank`; the Isaac-free standalone path requires `--train-bank`, runs the same strict bank
and motion-anchor loaders, and refuses to inherit any `stage1_*` envelope from a donor ONNX. For
each clip independently (clip0 forehand, clip1 backhand), rows must be unit within `2e-4` and stay
strictly more than `1e-6` inside the raw +Y/A-frame reference hemisphere, exactly matching the C++
runtime gate. A merely positive but `<=1e-6` row now fails export. The checkpoint contract and both
exporters must also carry the exact sign table `[+1,-1]`; representability is
`sign[clip] * raw_A.x > 1e-6`, not `raw_A.x > 0`. Therefore forehand raw A is positive-x and
backhand raw A is negative-x, while the external schema-2 physical striking-face B remains
opponent-facing positive-x for both. The normalized per-clip raw-A row sum defines the
spherical-cap center and the minimum row dot defines its boundary. The exporter records the
frame, convention, algorithm, tolerances, exact sign table, centers, references, thresholds, row
counts, train-bank SHA and source-family SHA in a canonical payload with its own SHA-256.

The external B normal is converted to A only after clip selection; position and velocity are not
changed. Dependency-light Python verification covers all bank rows at the exported boundary,
independent forehand/backhand signs, a physical-B positive-X but raw-A out-of-support normal,
opposite-sign poisoning, non-unit rows, wrong clip order, bank/family mismatch and payload content
binding. A subprocess smoke imports the standalone exporter without the package `__init__`, Isaac,
ONNX or Torch. The prospective real-bank fixture
`configs/phase1_face179_real_bank_envelope_expectations_20260712.json` binds bank
`2da2bd12...a0700`, family `b21c161a...28ad5`, `757/724` rows and expected cap minima
`0.974278/0.972078`; those read-only statistics are export expectations, not behavior evidence.
The standalone path now validates checkpoint binding, donor, both motion files, harvested buffers,
bank and derived envelope before creating any graph output. It exports to an owned same-directory
temporary file, checks the graph and metadata round trip, fsyncs it, then atomically replaces
`policy.onnx`; validation/export failure preserves an existing final model and removes the temp.
Behavioral tests cover successful replacement, injected failure and empty output. Host verification
is `41 passed, 1 optional real-runner integration skip` for the focused contract/export/preflight
group and `11 passed` for the planner wire. Pod1 subsequently produced an envelope-bearing formal
SZ seed3 model-2000 ONNX (`0c428ddf...b7b155`) from the exact train bank and passed the full
ROS/AimRT Release suite plus strict positive/negative production preflight; see
`configs/gate3_face179_strict_preflight_evidence_20260712.json`. This closes an export/model-load
prerequisite only. No policy has yet passed vendor MuJoCo backend first tick or behavior with the
envelope, self-hit gate or recovery contract. G05 remains `Partial`.
### Yikang branch changes were integrated by current-main semantics (2026-07-12)

Three small changes were audited against `origin/main@b2067ba`; neither old-base branch was merged
whole. The fit-lineage NumPy oracle from `stage1-fixed-point@bc86995/f0ac2fb` was accepted and
hardened. The existing Torch parity test now defaults to that in-repo reference, emits the exact
reference/contact-model/venue-YAML SHA tuple, and fails on an explicitly missing `RECORD_DIR`.
Normal handling is scale-invariant and rejects zero/NaN/Inf instead of propagating NaNs. Seven
dependency-light oracle tests pass. The full current-source Torch CPU parity gate also reports
`ALL PASSED`: table/paddle contact errors are below `4.63e-9`, flight RK4 is exact at reported
precision, and first-landing error is `0.000 mm`.

The `head_discipline` diagnosis from `407a443` is retained as a candidate, but its code and `-0.5`
weight were deliberately not ported. That commit is based on old `hitter@5c346ea` and enables a
`HOPEPingPongHitterPureRallyFinalV2` recipe that does not exist on current main; FinalV2Plus also does
not exist here. Importing the term/whitelist alone would create a stale, unnamed reward surface.
Moreover, `origin/hitter@0fccc3c` has a passive-head FinalV3 action contract for the same symptom,
and FinalV2Plus derives an exact reward-key set from FinalV2. A reward term must therefore be an
explicit named-recipe decision, not accidental inheritance or silent stacking with passive-head.

Two dependency-light guards verify that current main has neither the old FinalV2/FinalV2Plus recipe
nor a silently exposed/enabled `head_discipline_weight`. No training was launched. If reward-side
head discipline is adopted later, start from an explicit `0.0` current-line interface, compare it
paired against unchanged control and passive-head, and treat its overlap with balance/recovery/
ready-state rewards as a mixture interaction under fixed total budget. The old `-0.5` is a hypothesis,
not a validated default. In the local Torch/Hydra environment, the current reward/config tests pass
`88` relevant cases. The unfiltered suite has two additional failures in pre-existing
`MotionLoader` handling of a single `Path` as an iterable; neither touches this diff or head rewards.
Because no head code/config was changed, current reward bytes and behavior remain unchanged. Full
audit and commands are in `docs/research/yikang_selective_integration_20260712.md`. G05 stays
`Partial`.

### 2026-07-12 inexact first-actor-candidate observability boundary

The deploy runner's new `--first-tick-json` diagnostic does not change training, the 179-D actor,
actions, normalization, reward or any checkpoint. It records the first observed planner-engaged
actor candidate; idle/wait/recovery rows cannot consume the capture. This is not an atomic planner
snapshot or Gate3 certificate.

Backend `RobotState` has joint q/dq and IMU state but no root linear velocity. The diagnostic reads
the vendor pelvis-twist topic through a subscription-only sim sidecar instead of fabricating zero,
and records observation base separately from the joined vendor-world base. Missing/stale/nonfinite,
regressed headers and odd generations fail closed. ONNX Runtime loads the same stable bytes whose
digest enters the JSON.

The vendor topics have asynchronous publish-time stamps and no common MuJoCo sample sequence; the
current planner also lacks merged same-tick snapshot/shared payload epoch semantics. Therefore the
outer document and payload fix `evaluation_contract_exact=false`, with planner/native/source/runtime
exactness fields all false. Dependency-light source tests pass `6` cases. No actor/model weights
were evaluated and no Isaac or MuJoCo rollout ran. G05 remains `Partial`; this instrumentation
cannot promote a checkpoint or repair the four-seed stability failure.

### 2026-07-12 model-4000 activation consumer source gate

The previously runtime-free fresh `SZ` model-4000 queue now has a separate reviewed execution
contract and activation-consuming runner without changing the frozen queue, preregistration or
validator bytes. Every command requires the exact all-four activation file and caller-supplied SHA,
then revalidates both content-addressed Pod audits and all four finite/iteration-4000/schema-3/
exact-lineage/hard-contract records. A Pod also rehashes and re-audits its own two checkpoint files
and adjacent contracts before it may prepare or run.

Preparation is no-clobber and only copies the already-materialized K100 bytes; there is no schedule
generation path. Pod1 is fixed to seed1 then seed3 and Pod2 to seed2 then seed4, serially. Seed1 is
conservatively rerun on the same paper rather than reusing its previous score. Each judge uses the
pinned `judge.sh`, the shared `/workspace/.kit_boot.lock`, a new session with observed PID=PGID and
preserved state/log on failure. The runner has no SSH or signal API and cannot stop a trainer or
worker. Result validation binds exactness, 50 attempts per side, question order, MJCF,
execution/ready-state, checkpoint/contract and report/summary/attempt-ledger SHAs before a Pod
result can exist.

The aggregate cannot return a family-stable PASS: known-before-prereg seed1 model-4000 was `.50`,
below the unchanged `.65` worst-seed rule. It only classifies seed4 as delayed learning when 4k is
at least `.65` aggregate and `.50` on both sides; otherwise weakness is persistent through 4k.
Either outcome keeps all training unchanged and authorizes no promotion/deployment/hardware.
Queue plus runner focused source tests pass `40` cases. At source merge no Pod readiness audit,
activation, preparation or judge had run, so that result was not training evidence.

The barrier was subsequently materialized outside both frozen checkouts on 2026-07-13 local time.
Pod1/Pod2 audit file SHAs are `3fc325e1...247b8` and `4f25786b...565f7`; their exact union produced
activation file SHA `9dea76c2...ce704`, content SHA `eaa92ca2...aa4fb`. The immutable source,
schedule, both audits and activation now occupy identical absolute paths on both Pods. Both
activation-consuming `contract-check` calls passed, and the immediate snapshots contained no
child judge, Kit process or Kit-lock holder. No `prepare`, rollout, score, trainer mutation or
signal occurred. This closes only the all-four readiness barrier; G05 remains `Partial`.

Both Pod runtime contracts were subsequently created by the no-clobber `prepare` command. Pod1's
file/content SHAs are `2b76a5a...8201e` / `36e878f0...5ba73`; Pod2's are
`dbecc102...d1c9b` / `91a0070a...30794`. Direct runtime-binding validation rehashed the local two
checkpoints per Pod and confirmed iteration 4000, finite tensors, exact lineage and the common
hard-contract SHA; both train/eval checkouts remained exact and clean. Each contract still records
`prepared_not_started`, `jobs_started=0`, `auto_start=false`, and the post-prepare process/lock
snapshot was empty. No judge, score, rollout, signal or hardware action ran. This is execution
paper preparation, not training evidence; G05 remains `Partial`.

### 2026-07-12 training-backend boundary

G05 continues to own the Isaac first loop and its checkpoint lineage, but a higher Isaac training
score or another Isaac-only reward/teacher sweep no longer counts as resolving G06. Native MuJoCo
training/fine-tuning is now a P0 implementation and promotion track recorded in G06. Shared
observation, action, reward, reset and export contracts must still be updated here when they change,
and the exact fresh `SZ model_2000` checkpoint is the first handoff candidate. No backend code or
training run exists yet, so G05 remains `Partial`.

### 2026-07-12 MuJoCo-v0 handoff and warm-start correction

The native-MuJoCo preflight keeps the 179-D Isaac actor as a source candidate but corrects the
handoff semantics. `load_actor_tolerant()` is not suitable for the planned causal paper: its strict
path restores the runner, while its fallback loads every shape-matching state tensor, including any
matching critic tensor. The v0 loader must construct a seeded fresh critic/optimizer first, then
load only the complete actor, action-distribution state and actor normalizer; tests must prove the
critic is unchanged, optimizer state is empty and iteration is zero. The actor normalizer remains
frozen for the initial v0 paper.

The current schema-3 hard contract also deliberately omits reward weights, termination thresholds
and optimizer settings. That remains valid for its existing curriculum purpose but is insufficient
to identify a MuJoCo causal fine-tune. The new backend experiment contract must additionally bind
those fields, reset/timeout semantics, effective MuJoCo profile, runtime action post-processing and
source-checkpoint SHA. Its first one-shot `Trainer-v0` optimizes balance/strike-state only; the vendor MJCF has
no physical ball/table/net, so it cannot book formal return evidence. Full reasoning and read-only
commands are in [the MuJoCo training-v0 preflight](../research/mujoco_training_v0_preflight_2026-07-12.md).
No code, config or training changed; G05 remains `Partial`.

### 2026-07-13 MuJoCo preflight 红队暂缓合入

franco 已明确批准 native MuJoCo feasibility/implementation 作为 P0 能力轨，但它不是几天内
`Gate3-D0` vendor planner+policy 演示的前置。当前 matched paper 证明解析回球/击球执行跨引擎
塌陷，而 physical-fall 计数接近零，不能夸成“平衡也已证明退化”。候选 `6e5fce3` 的七个授权位
保持 false，focused 63 项与顶层 `468 passed, 9 skipped` 通过；但 action tape/trace 未覆盖
clamp/runtime adapter、静态 source closure 有 alias/exec 逃逸、JSON 接受 duplicate/nonfinite、
MJCF `compiler strippath` 解析错误。因此当前 `NO-MERGE`，四项必须先补负测并修正。

第一个 single-env core 还必须对 N=1/8/32/64 分别报告 sim-only 和完整 rollout+一次 PPO update
的 step/s、RTF、RSS/CPU 与扩展效率，并按预注册 transition budget 推算两臂×两 seed 墙钟。
只有能在 48 小时内完成且留 30% 余量，才继续 CPU-Python 长训；否则转 C++/OpenMP 或另行过
parity 门的 MJX/MJWarp。没有 `VecEnv`、PPO、训练、Pod、simulator 或真机行为证据，G05 仍为
`Partial`。

### 2026-07-13 v12、高点拍压、横移老师与非击球臂设计

`configs/motion_video_intake_20260713.json` 已逐字节绑定 7 段新的私有视频：v12 正反手挡球、
一个反手高点拍压第五动作，以及左右横移各两个下肢老师候选。7/7 文件核验与 11 项专项测试通过。
其中 Franco 主线的 S0 高点拍压和 M0 四条横移又在 Pod1 完成 exact GVHMR：帧数分别为
`88/88` 与 `105/105、97/97、82/82、96/96`，所需 tensor 全 finite，输入、输出、queue、binding
和 audit SHA 已进入 `configs/motion_video_gvhmr_s0_m0_results_20260713.json`。这仍只是人体结构证据；
五条都没有完成 GMR、运行顺序 schema-2、L0、厂商 L1、桌网余隙、动力学或匹配题目的回球门，
也没有候选进入 RL 队列。v12 本轮未执行。

横移素材被定义为“以横移距离为条件的下肢老师”，不是另一种挥拍。上下半身先按准备/击球支撑/
恢复事件对齐，再明确根节点、骨盆、躯干和足接触的所有权，由受约束的全身求解处理耦合；TOPP
只能给已接受路径重定时，不能把错误足接触变稳定。恢复终态要回到该素材初始的水平双脚相对向量，
同时保留站距和前后错位。v12 必须在挡球专用考卷上赢过旧安全候选，高点拍压必须先过独立高球卷，
之后才允许讨论四动作对五动作。

另一个配对设计测试是否解除左侧非击球臂的模仿，让它参与平衡；“直接移除 Reward”和“固定总预算
重新分配”必须分开，所有硬安全保持开启。三份记录见 [实验登记册](../experiments/README.md)。
后续 E1 source gate 已把 A0/A1 直接 mask 物化：四条 body-imitation Reward 都显式列出
`body_names`，A1 只删左 shoulder/elbow/wrist，并保持 A0 的躯干/右击球臂、所有权重和硬安全不变；
contract drift 与错误布尔值 fail closed。machine prereg 固定 fresh seed17、`4096 env × 1001 update`、
`+200/+500/+1000`，默认 plan-only，Pod launch 需要 root 显式 token，claim/checkpoint/result 都
no-clobber。checkpoint 内嵌 hard contract 还逐臂绑定 post-override 四项 body list；两臂 hard SHA
必须不同，而删除该唯一字段后合同必须完全相同。源码/runner 共 `71 passed`，但 Pod
runtime 已形成一个受控 partial：A0 于 `2026-07-13T19:48:35Z` 以 PID=PGID `1811464` 启动，
`19:49:15Z` ready；其 `model_200.pt` 已验证 embedded iter `200`、finite、fresh lineage `1` 并绑定
hard-contract SHA `14ef410b...29f1`。旧 outer verifier 随后因把 schema-3 bank metadata 的 physics SHA
错当 compact hard-record direct leaf 而精确假拒绝，故 A1 从未 claim。v1r1 source gate 现改为独立解析
bank file/metadata、复现旧错误、先 attest 既有 A0 三份稳定 SHA，再且仅再 claim A1；A0 dead/drift、
A1 预存在或 bank drift 都 fail closed，且禁止重跑 A0。v1r1 专项 `12 passed`，新旧 runner 合跑
`30 passed`。现场 `validate-runtime` 全绿后，唯一一次 `launch-a1` 已成功：A1 PID=PGID `1816234`、
Kit ready，emitted hard-contract SHA 为
`c85b52a28ad64a667a7b522562842466270b3741591f6daf09afc1d0f7c6b146`；A0 PID=PGID `1811464`
untouched。recovery/runtime receipt 已 no-clobber 写入，judge 未启动。external `--mode plan` 另暴露只读
相对路径 bug：它在 external control 下误找 `control/configs` 并在任何写/claim 前失败；runtime/launch
使用冻结绝对 v1 路径，不经过该分支。已绑定的 v1r1 bytes 不得修改，路径 bug 只能在新版本修。
A1 milestone、配对终档和同卷判读仍未发生；A2 固定预算继续 blocked。详见[实验](../experiments/non_striking_arm_imitation_ablation_20260713.md)
与[操作](../operations/run_phase1_non_striking_arm_imitation_a01.md)。S0/M0 有 Pod 离线结构结果，但没有
Isaac/MuJoCo 训练、仿真行为或真机动作，G05 仍为 `Partial`。

S0/M0 的 post-GVHMR machine handoff 已完成 exact runtime `consume`，输出分别是 4,970/9,242 bytes，
SHA-256 `d57a93e0...a1054` / `60c55150...088ef`。下一层 canonical-beta 已拆成两份独立 no-clobber
计划；consumer 只注入旧 exact donor，其他 PT leaf 必须 save/reload bit-exact。host 新旧专项为
`15 passed, 1 skipped`；真实 canonical-beta `inspect/consume` 也已在绑定 Pod1 CPU runtime 完成，S0/M0
completion manifest SHA 为 `964a7333...f1be3` / `5cef05f7...71a65`，五条 non-beta 内容 bit-exact。
这只解锁另建 exact GMR prereg，不直接解锁 schema-2 或 RL。S0 仍不得借用拉球题或声称击球有效；M0 的
canonical foot-site 与容差现已由 exact-GMR prereg 冻结，初末二维脚间向量和 pass 仍为 null，必须由
robot-coordinate GMR 产生，双脚
并拢不能通过。这是 canonical-beta-time 快照；2026-07-20 回收结果已填充并判 stance `0/4`。详见
[handoff 记录](../experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)与
[canonical-beta 记录](../experiments/motion_canonical_beta_s0_m0_20260713.md)。

### 2026-07-14 S0/M0 exact GMR prereg boundary

The five canonical-beta PTs now have separate S0/M0 exact-GMR machine plans, but they do not yet
authorize a training asset. The CPU consumer freezes the old GMR argv, exact source/runtime/model
closure and report-last publication. M0 additionally freezes exact 30 Hz ready-window sample lists,
canonical A3 foot FK and both components of `right_foot_xy-left_foot_xy`; a finish narrowed by more
than 5 mm fails independently of the 3 cm component band. S0 contact/effect remains null and cannot
borrow a loop paper.

The 2026-07-14 follow-up bound all 16 exact source/runtime facts. Direct retarget XML has the bound
31-joint/32-body order but an exactly empty site inventory; canonical vendor foot sites cannot be
substituted into retarget evidence and remain exclusive to M0 stance FK. Both batch plans and shared
runtime are `preregistered_not_executed`; both host static validations pass. Runtime
`inspect/consume` and GMR outputs still do not exist. Schema-2, L0/L1, dynamics and RL remain blocked;
no trainer or hardware command ran. G05 remains `Partial`; details are in
[the experiment](../experiments/motion_exact_gmr_s0_m0_20260713.md).

This is the preregistration-time snapshot. The recovered 2026-07-20 Pod1 completions in Current State supersede
only its global “outputs do not exist” inference; all downstream blocks remain.

### 2026-07-12 文档路由与当前成绩表

训练状态现在按职责拆分，不再复制到三份流水中：

- [`docs/NOW.md`](../NOW.md) 负责当前完整训练流程、现行课程阶段、各主题的问题/解法/效果/差距，
  以及最接近正式目标的逐动作单拍/连续成绩表；
- [`docs/experiments/`](../experiments/README.md) 负责假设、冻结变量、run 表和决定；
- [`docs/TIMELINE.md`](../TIMELINE.md) 只解释已经进入 `main` 的重要逻辑变化。

全局优先级只看 [`docs/NOW.md`](../NOW.md) 的统一工作队列；GPU/Pod 不再各建影子队列。
排序、Kit boot lock、关键路径独占与广度波 3–4 条/卡的适用条件统一见
[跑批作战手册](../runbook.md#统一队列排序与算力纪律) 和
[`run_on_runpod.md`](../operations/run_on_runpod.md)。本次只收口文档规则，没有改发射命令。

当前 `SZ model_2000` 成绩表汇总四个 exact seed，没有隐藏 seed 4：正手单拍的
无物理摔倒/解析击球/解析回球为 `200/200, 137/200, 133/200`；反手为
`200/200, 170/200, 170/200`。连续格仍为 `未测`，因为每道 K100 题都通过非物理
tracking guard/reset 路径结束，其中包括 seed-4 的击球前 guard。这是同一份既有结果，不是新实验；
这里的击球/回球来自 `VirtualReturnScorer` 对拍状态的推演，不是 simulator 球-拍-台接触。
seed-stability 判决仍为失败。本次文档迁移没有运行 Pod、模拟器、训练或真机动作；G05 仍为
`Partial`。

### 2026-07-13 formal tuple source integration boundary

The accepted deployment repair changes planner transport and actor engage safety only: no reward,
training recipe, observation/action dimension, checkpoint or active Pod arm changed. Formal racket
schema 3 now references one exact formal base sequence, while closed-loop gates and the actor
observation use latest tick-start base. Latest stale/low/implausible/epoch-changed or
revocation-changed base blocks actor inference during active swing and recovery; ordinary valid
same-epoch refresh remains legal.

Exact source passed planner tests `180 passed, 2 optional skipped`, Pod2 portable Release focused
`40/40` and native `233 passed, 5 optional skips, 0 failed`. These are source/binary results, not a
new training setting or behavior score. The 179 actor still needs ROS/AimRT first tick and vendor
MuJoCo behavior; G05 remains `Partial`.

### 2026-07-13 persistent q50 top-level startup source gate

The model-4000 activation consumer now has a separate
[persistent-supervisor contract](../interfaces/q50_persistent_supervisor_contract.md) for the one
remaining process-lifetime gap: the consumer's top-level Python process could disappear with its
invoking SSH shell while an already-detached judge child continued. The new wrapper exposes only a
manual no-clobber `launch` and read-only `inspect`. It neither changes nor replaces the existing
consumer, execution config, all-four activation, prepared runtime contracts, [q50/K100 paper](../DEFINITIONS.md#q50-and-k100),
checkpoints, trainers, or workers.

Before execution, the child creates a new session, redirects fixed stdio, closes inherited file
descriptors and publishes a hello with `PID=PGID`, Linux boot id/procfs start ticks, bound executable
SHA, exact argv/fixed-environment digest and the complete source/config/activation/runtime SHA
closure. The parent publishes an immutable ledger and commit token only after independently
validating that identity. Without the token the child times out and exits by itself; after the token
it revalidates all bytes and identity/token/ledger/result before acknowledgment and again before
`execve`. First possible visibility of the token's final link, not acknowledgment timing or the
following directory fsync, is the irreversible no-retry point; the startup deadline only governs
token absence. Independent acknowledgment and exec observation
windows return `token_published_pending_ack` or `committed_pending_exec` with return code zero when
progress is not yet visible, while every second launch remains no-clobber rejected. Inspection
rejects PID reuse and delegates terminal acceptance to the original runner's full result validator.

The focused supervisor suite passes `24` cases; queue+consumer+supervisor together pass `64`. The
suite includes tokenless deadline expiry, post-token delayed rehash, a 1.15-second acknowledgment
atomic-publication stall, post-ack delayed exec and terminal-result A-to-B replacement. The three
post-token stalls preserve no-retry authority and converge without a fatal-before-later-runner
sequence. Post-link token-directory-fsync plus evidence-stat failure, token temporary-cleanup
failure and parent-observation-write failure also return committed pending, reject restart and later
inspect as exact running; none can escape as a retryable launch error. The host is macOS, so procfs behavior is covered through an injected
identity seam and still needs one Linux fake-runner source smoke before any real q50 process. No Pod
deployment, judge, simulator, training mutation, process-control action or hardware command ran.
G05 remains `Partial`.

### 2026-07-13 Phase-1 运行池运营裁剪

负责人明确批准把已显示持续塌陷的 fresh 运行停止，以便把算力换给
[NOW 唯一队列](../NOW.md#统一工作队列唯一优先级账本)中更靠前且前置已满足的工作。16 条
fresh 广度臂中首先精确停止 8 条：formal `SZ` seed1/2/4，以及诊断格 `SP` seed1/4、`LZ` seed1、
`LP` seed1/2；其余 8 条当时继续，后来按本节下方的 signed-face 取证再停止。详细 q10 曲线、
已知 q50、PGID、最后 checkpoint 与 SHA 见
[拍面×plant 广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。

这是负责人在结果出现后作出的**算力运营决定**，不是发射前预注册的统计停止规则。历史 manifest
仍保持 q10 K20/每侧 10 题 `screen_only=true`，不能晋级；model-2000/model-4000 q50 合同中的
`whole_arm_stop_allowed=false` 也不改写。因此不得把这次停止写成“q10 正式 reject 三个 formal
seed”，不得隐藏已停止 seed，也不得用它给任何 setting 晋级。

每臂信号前均保留最新日志/checkpoint，并验证文件名迭代号等于内嵌迭代号、`1,762,715` 个浮点
元素且 `nonfinite=0`、schema-3、fresh lineage=1，以及 checkpoint ↔ 相邻 hard-contract SHA
一致。TERM 未使这些 trainer 退出；确认没有 live child 或 Kit-lock holder 后，只向各臂 `.launch`
登记的 PGID 发送 KILL，再核对该组消失和其余接受臂仍存活。没有使用 broad `pkill/killall`，没有
向 worker/judge 或真机发信号。formal 四 seed 的 model-4000 checkpoint 早已内容绑定并通过
readiness，所以已准备 K100 后续卷输入不变。这个运行处置不新增质量成绩，也不关闭训练稳定性、
signed-face 或连续能力门；G05 保持 `Partial`。

model-4000 与剩余臂的 signed 切面随后使这个运营决定扩展到全部 16 臂：剩余臂最近
24/24 K20 格的正手 signed composite 均为 0，无论 shared 还是 legacy face 都不分离。第二波
也在 no-clobber checkpoint/log/contract 审计后只按精确 PGID 停止；两 Pod 无 trainer、GPU 已空。
这仍是负责人事后算力决定，不是 q10 阈值的预注册 stop rule。详见
[广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。

### 2026-07-13 model-4000 四 seed matched q50 稳定性失败

已准备的 fresh `SZ model_4000` 同卷在 Linux fake-runner 冒烟后，经过内容绑定的一次性
supervisor 在两 Pod 完成 seed1→3 与 seed2→4 串行判卷。两份 Pod result 均为
`terminal_result_validated`，最终无残留 supervisor/child judge/q50 Kit-lock holder。正式
aggregate file/content SHA 为 `1ba88e39...d195` / `226e6050...648d`，独立 canonical 复算通过。

四 seed 旧 parsed return 为 `.50/.88/.98/.00`，median `.69 < .75`，worst `.00 < .65`，
spread `.98 > .20`，minimum-side `.00 < .50`，四项冻结门全失败。seed4 为
`0/100`且有 21 次 `fall_root_z`，因此归类为“持续弱到 4k”而非晚熟。该失衡是
seed4 特定结果，其他三 seed 物理 root fall 为 0。

同一结果又证明旧解析分不能作 baseline selector：seed2/3 正手 raw-A signed normal 误差
`172.33°/174.35°`，signed strike composite 都是 `0/50`，但 parsed return 仍为
`38/50` 与 `48/50`。所以不会晋级最佳 seed，也不再用相同 `SZ` 续训买晋级证据；
下一步是 `n/-n` 负控、signed-face scorer 修正和同卷复判。这仍是每题重置的
Python BankExam，不是 physical ball、连续恢复或厂商 Gate3/Gate3B；G05 保持 `Partial`。
详细证据见 [Fresh SZ 稳定性实验](../experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。

### 2026-07-13 signed-face 训练信用门：源码闭合，行为仍待 canary

seed3 的 content-bound TensorBoard 摘要把“旧解析尺有病”推进到训练信用本身：正手法向误差从
iteration 2000 的 `167.49°` 继续到 13800 的 `174.02°`，signed normal pass 一直为 `0`，但训练内
virtual return 同期从 `.692` 升到 `.965`；反手在 13800 为 `5.86°/.996/.967`。同一 step 的三项
全局 virtual reward tag 合计 `.4615195`，是全局 normal reward tag `.15587743` 的
`2.960784637×`；但这些 tag 汇总所有环境和正反手，不能归因或量化正手错面的 reward 份额。实际
`env.yaml` 已绑定 `virtual_ball/vb_metrics_only=true` 与 `20/30/5/5` 四项权重。结合 face-blind
源码路径，只支持“wrong-face FH states were treated as reward-eligible by the active face-blind reward
path”，不支持“正手错面实际领取了多少”或把反面行为归因于单一 reward；完整 tag、step/value、
event/training-contract/env.yaml/launch/nohope.diff SHA 与 source-commit claim 证据边界见
[拍面符号卷宗](../experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。

当前 feature source 已把 `hope_commands._vb_evaluate` 的 `vb_fired` 改为：先用统一
`face_tracking_pair` 比较有符号拍面，再要求 achieved/target physical-B 都严格朝 `+X`，最后才允许
`orient_normal` 进入冲量计算。所有 `virtual_pass_net/landing/spin` 都消费这个门后的 one-shot
mask。Torch oracle 同步增加 finite/non-degenerate、strict hemisphere 和 physical-B 门；NumPy
`n/-n` 负控证明出球/落点仍相同而错面不能触发记分。

focused 回归为 `38 passed, 1 skipped`，顶层 broad 为 `546 passed, 9 skipped`；另一个排除
Torch/Hydra import-bound 文件的 training dependency-light 组合为 `381 passed, 21 skipped`。
这些不是 Isaac 行为结果。本节没有启动 Isaac、Pod、trainer、judge 或真机。下一个 fresh 双侧
canary 必须绑定新 source/hard-contract SHA，
验证错面样本不再得到 `vb_fired`/virtual reward，并观察正手 signed normal 与 return 是否共同学习；
它按[单-seed 机制漏斗](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)先买机制证据，
不先复制四个 seed。在此之前不能写“训练行为已修复”，G05 继续 `Partial`。

该漏斗的 machine prereg 与 launcher 后续已物化：训练 source 固定 clean detached
`882fea4285f0cf9a97ba79d79ae8af31d26ea1ed`，一张卡只允许 seed3 的 A/B/C/D 四个不同因果格；L1
为 `512 env × 25 update` launch-integrity smoke。热启动 A/B 因旧父合同缺少当前
event-timing/target-cadence 字段，固定为显式 inexact transfer 并要求 lineage `0`；fresh C/D 必须
lineage `1`，四格 emitted contract SHA 一致。四格终档 finite/contract/lineage 全通过后只写
no-clobber completion 证据；它不能单独授权 L2。L2 的 `4096 env × 1001 update` 设计因 immutable
signed directional checkpoint paper 的 path/SHA 未冻结而在 runtime preflight 顶部 fail closed。
focused 静态/攻击回归 `23 passed`。首次 Pod `control/v1` preflight 在任何 run claim 前暴露审计器
只看 checkpoint 顶层的假拒绝；只读递归复核证明父模型 `74` 个浮点 tensor、`1,762,715` 个元素均
finite，合同三元组实际位于 runner 的 `infos`。`control/v1` 保留未覆盖，v2 改为递归 finite 扫描并
强制从 `infos` 绑定 schema/SHA/lineage。v2 随后在 A 格首次 learning iteration 前暴露 source-first
环境未传给 child；失败 claim/log 保留，B/C/D 未创建。v3 绑定 tracked setup 脚本、拒绝 local override，
v3 因在 `SimulationApp` 前真正 import IsaacLab 而假拒绝；v4 改用 `find_spec` 只验证模块 origin 位于
exact `882fea4` worktree，正式 import 留给 Kit boot。当前仍没有新 Isaac checkpoint 行为结果，
不能把 E1 写成训练修复。v4 随后进入 Kit/scene 后发现 detached worktree 缺 Git-ignored A3
URDF/mesh/config；失败 A claim 保留。v5 从 clean `6d93bcb` 恢复并同时绑定 source/target tree 的
`46` files、`15,378,264` bytes 与 canonical SHA `0137f59b...26c6`，claim 前拒绝缺失/额外/symlink。
复现命令、SSH 中断恢复和半写 claim 的 fail-closed 处置见
[操作文档](../operations/run_phase1_signed_face_rescue_funnel.md)。

v5 随后在 scene 构建完成后被 schema-3 loader 正确拒绝：旧 train bank 的 physics contract 记录旧
`virtual_ball.py`，而 `882fea4` 新增 signed-face helper。失败发生在 hard-contract/first iteration/
checkpoint 前，A claim/log 保留，B/C/D 未创建；不能以 legacy load 绕过。main 现加入严格 no-clobber
bank rebind consumer：先证明七个 physics 文件只有一个 helper 定义新增、移除它后 executable AST
相同，且 generator/loader 不变；再要求全部非-meta 数组 raw bytes 不变、metadata 只有四个 leaf、目标
runtime 的 exact motion contract 和 1481 题 old/new contact/flight bytes 相同。Pod1 已正式发布并复核
bank SHA `3a9d8851...5b71` 与 report SHA `9fffed03...bb37`：24 数组未变，正/反手 `757/724` 题的
旧/新输出 raw bytes 相同，landing/net 全过。v6 又把 report closure 及唯一允许的父旧 bank→当前新
bank common-field transition 写入 preflight；其他父/新共同字段仍逐值相同。

实际 epoch-1 v6 后续在 clean `50c49e5` source 上启动：A/B/C 到终档，checkpoint 迭代分别为
`13824/13824/24`，lineage `0/0/1`，共同 hard-contract SHA `dfc583d4...888a5`；D 在
`runtime_verified`/checkpoint 前 Kit boot timeout。其旧 launch/state/log 与 timeout 诊断已按 exact SHA
冻结，PID 已死且旧 claim 不覆盖。原始 checkpoint audit `62076758...d354` 绑定 A/B/C 的 exact
checkpoint/finite/lineage，并明确 D `run_dirs=[]`。后续 [v6r1](../DEFINITIONS.md) 首次真实
`validate` 在任何 claim/训练前暴露 validator 自相矛盾：它错误要求旧 would-be D training path 存在，
而冻结 audit 与 filesystem 都证明该 path 应 absent。团队没有伪造目录；v6r1 从未 claim、launch、
signal 或训练。新 [v6r2](../DEFINITIONS.md) 只做 source contract correction：旧 path 必须 absent，
任何 entry kind 都 fail closed；只支持 `static-validate`，没有 runtime preflight、命令重建、进程检查、
launch、signal 或 finalizer，且明确 NOT LAUNCHED。
后续 foreign v8 使用 clean `72418fff` 与全新 manifest/launcher，`v6_artifacts_adopted=false`，按 terminal
barrier 串行发射 A/B/C/D。A/B/C 前序已终档；D 是第四格，900 秒内再次未出现 hard-contract marker、
runtime verified、learning iteration 或 checkpoint。locked wrapper 只对 `PID=PGID=1782834` 做精确
cleanup 并返回 124；日志无 NaN/Inf/Traceback/OOM/malloc/Killed。因为这已是继 v6 D 后第二次独立
pre-contract timeout，自动 retry 已停止，转入 boot 根因。没有四格 activation；L2/judge/第二 seed
仍固定 false，所以 G05 仍为 `Partial`。操作仍见上面的 signed-face 漏斗运行手册。

后续三次只读审计把 v6/v8 D 的共同失败点收窄到 identical table USD 的 load→PhysX 交界；两份 D
normalized argv 除 versioned run name 和 v8 launch-claim provenance 外相同，且都从未出现 hard contract
或 learning。相邻 C 分别在 `2.339/3.031 s` 越过该边界，v8 D 又是在 C clean shutdown 后 `44 s`
启动，所以这仍不是 reward/seed 学习结论，也不能用配方相同 retry 获得新信息。事后容量非饱和只
降低持续资源耗尽的可能性；Carbonite residue、瞬时 driver/filesystem stall 与 ordinal-4 累积状态仍未
分离。机器账见 [boot 结果 ledger](../../configs/phase1_signed_face_boot_root_cause_results_20260714.json)。
下一份 `D-first × ordinal-4`、`host IPC × private IPC` 的 scene-only 诊断只有
[design-only prereg](../../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)，无 launcher/Pod/
signal/training 权限。G05 继续 `Partial`。

### 2026-07-14 C2/D2 provenance-complete L1 source gate

v9 的只读证据复核发现旧 hard contract `dfc583d4...888a5` 没有绑定 positional/signed-face guidance
weight，因而旧 C/D checkpoint 即使有 outer launch claim，也不能只靠 checkpoint+相邻合同区分
`0.0/-0.4`。新训练 source `4467d79f1ed425a4263f0caaad2f661e1ec737ad` 把两项 post-Hydra
guidance 的 weight/command/bound 写入 schema-3 hard contract；checkpoint `infos` 另绑定非自引用的原子
launch-claim SHA，claim 覆盖 exact source、优化配方、host/GPU lane、seed、终档迭代和 claim directory
identity。Kit carb/TBB `16/16` 与 `useOmniJob=false` 也由启动后 runtime marker fail closed。

新的 [`signed-face C2/D2`](../DEFINITIONS.md) manifest/launcher 只包含 fresh seed3 的两条
`512 env × 25 update` L1。为遵守团队先铺满六卡的调度，C2 固定 Pod1 GPU1、D2 固定 Pod1 GPU2；每条
claim 前本卡必须空。host-wide Kit boot lock 串行，但 C2 `runtime_verified` 后继续训练，D2 可立即在
另一卡 boot/并发。两条 command 的 local device 都是 `cuda:0` 且 source/PYTHONPATH/runtime 相同；
physical GPU 是 outer execution lane，不进入优化配方。每个 `model_24.pt` 必须 finite、iter24、lineage1，
绑定各自相邻含 guidance weight 的 hard contract 与 outer claim；pair finalizer 要求两合同去掉该唯一
nested weight 后逐值相同。

manifest/launcher SHA 为 `785ad96d...9895` / `0fa25020...03ba`；专项测试 `28 passed`，source
launch-claim/thread-cap `28 passed`，reward/hard-contract override `58 passed`，仓内 `tests/` 回归见同次
实验卷宗。本任务没有 Pod/runtime/trainer/checkpoint；旧 v9 artifact 不采用，activation/judge/L2/第二
seed/stop-promote/真机全为 false。复现见
[运行手册](../operations/run_phase1_signed_face_cd_l1.md)和
[实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)。因此这里只是可审阅的 E1
source gate，G05 保持 `Partial`。

### 2026-07-14 C2 两层 outer 假拒绝与 v1r2 continuation source gate

C2 已在 Pod1 GPU1 按上述 v1 claim 自然到达 `model_24.pt`；final log/hard contract/checkpoint SHA 为
`abffd457...6dc3` / `83f47ae6...2772` / `dbbc7a28...6f6`，canonical outer claim 为
`37fe2443...86e5`。trainer hard contract 正确写出 exact float
`mount_normal_sign_per_clip=[1.0,-1.0]`，但 v1 outer verifier 用整数 `[1,-1]` 作为 strict-type
期望，导致合法合同在 post-boot 被假拒绝。旧 `runtime_verified`、failure、terminal result 均 absent；
不得补造旧 runtime sidecar，也不得重跑 C2。

冻结 [`v1r1`](../DEFINITIONS.md) 接受 exact float 并拒绝 bool/int，却又错误要求 trainer compact
`question_bank` 直含第六个 `physics_contract_sha256`。source `4467d79` 实际只发
`sha256/schema_version/split/source_family_sha256/exact` 五键；physics 只在 exact NPZ metadata 与
source-family contract 内。v1r1 因而精确假拒绝合法 C2 hard contract。最后一次成功只读快照
`2026-07-13T22:32:07Z` 证明 v1r1 control/evidence/pair、D2 arm/exact run 全 absent 且没有
write/claim/launch；后续 SSH 状态 unknown，所以历史 absence 不构成当前授权。v1/v1r1 bytes 均冻结并
禁止运行。

新的 [`v1r2`](../DEFINITIONS.md) 是 D2-only source gate：manifest/launcher SHA 为
`4e202589...8c638` / `2b53c865...45a12`。它精确复现 v1r1 错误，严格接受 trainer 实际五键 shape，
再独立解析 exact NPZ `meta_json` 绑定 file/schema/split/source-family/physics SHA，并复算
source-family contract。伪造第六键、metadata drift 或 v1r1 evidence root/C2 receipt/pair receipt、
D2 arm/exact run 任一存在都 fail closed。C2 attestation 只能进入独立 `continuations/v1r2/`
no-clobber root：先写 content-bound absence receipt 并立即重查，再只允许未 claim 的 D2；preserved
C2 arm 不增加文件。实际写任何 v1r2 byte 前必须已完整通过 C2 terminal、exact bank、v1r1 假拒绝、
两张绑定 GPU 与 live absence 复核；首次 exclusive write 后的失败会保留 receipt 并永久阻断同 namespace
重试。v1r2 自有 JSON/NPZ metadata 的重复 key 也 fail closed。

外部 control 只接受 `scripts/ + configs/` 六文件 mini-tree，同时绑定 v1r2 与冻结 v1/v1r1 的
helper/manifest；safe relative paths 拒绝绝对/`..`/symlink。临时外部树 subprocess
`static-validate/plan` 已通过，缺任一文件、旧扁平布局和重复 JSON key 均失败；v1r2 专项攻击测试
`52 passed`，三代聚焦回归 `111 passed`，受支持的完整仓内 `tests/` 为 `934 passed, 10 skipped`。
本分支没有连接 Pod、安装 control、写 attestation 或启动 D2；因此这是 source gate，不是 C2/D2 成对 runtime
通过。activation/judge/L2/第二 seed/晋级/真机仍全部为 false，G05 保持 `Partial`。复现见
[操作文档](../operations/run_phase1_signed_face_cd_l1.md)与
[face-sign 卷宗](../experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。

### 2026-07-14 C2 零摩擦声明/发射不一致：D2 永久停止

v1r2 control 合入并精确安装后，`static-validate/plan` 通过；fresh `validate-runtime` 在任何
`continuations/v1r2`、attestation、D2 claim 或同名 training run 写入前，以
`hard contract is not 31/31 zero-friction` fail closed。现场相邻 hard contract SHA
`83f47ae6...2772` 的 31 个摩擦系数均为非零 PhysX 默认值。根因不是第三个 outer 假拒绝：冻结 manifest
声明 `zero_joint_friction=true`，但 C2 launch argv 与 optimization recipe 都没有
`task.plant.zero_joint_friction=true`，所以 trainer 正确记录了真实非零 plant。

C2 只保留为 nonconforming 根因证据，不能与正式零摩擦谱系混用；C2 不重跑，D2/v1r2 永久
`NO-LAUNCH`。下一次训练必须使用全新 C3/D3 namespace，并把同一零摩擦值同时绑定到 argv、optimization
recipe、claim 和 emitted hard contract。该新 source/runtime 门通过前，G05 保持 `Partial`。

### 2026-07-14 C3/D3 显式零摩擦 L1 source gate

全新 C3/D3 prereg 不复用 C2/D2 的 run、claim、environment receipt 或 artifact root。两条都是 fresh
seed3、`512 env × 25 update`；C3 guidance 为 `0`，D3 只把 signed-face guidance 改为 `-0.4`。manifest
和 launcher 要求 `task.plant.zero_joint_friction=true` 在每条 argv 恰好出现一次，并逐层绑定到 outer
optimization recipe、atomic claim、唯一 `ZERO_FRICTION_RUNTIME_OK` marker、31/31 finite-zero hard
contract 以及 terminal checkpoint replay。任一层不一致永久停止该 namespace，不自动 retry。

manifest/launcher SHA 为 `eefc8023...5dc2` / `19214890...a628`；`static-validate`、plan 与专项
`38 passed`，完整 `tests/` 为 `972 passed, 10 skipped`。本节仍只有 E1 source 证据：Pod runtime、
checkpoint、activation/judge/L2/第二 seed/晋级/真机全为 false。复现见
[实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)与
[操作文档](../operations/run_phase1_signed_face_c3d3_l1.md)，G05 继续 `Partial`。

Runtime 随后在 Pod1 GPU1/GPU2 分别一次性启动 C3/D3。两条各自 hard marker 与 31/31
`ZERO_FRICTION_RUNTIME_OK` marker 唯一，均自然到 finite/iter24/lineage1 `model_24.pt`；terminal SHA 为
`8c579386...e8ef` / `ccb9933c...7f0e`。finalizer 证明两份 hard contract 除
`racket_guidance_reward.signed_face.weight` 外逐值相同，并发布 paired receipt SHA
`bb3cd749...bbde`。这把 L1 provenance 从 E1 提升到 E2，但仍无 K100 行为、L2、第二 seed 或晋级，
所以 G05 保持 `Partial` 且 C3/D3 禁止重跑。

### 2026-07-14 A0/A1 paired checkpoints complete，行为仍未判

非击球臂 A1 自然退出；A0 在 `model_1000.pt` 稳定写完后发生 terminal teardown hang。终档 embedded
iteration、finite、fresh lineage、相邻 hard SHA 和正式 failure regex 均先通过；精确 PGID `1811464`
对 `TERM` 20 秒无响应后，只向同一个单成员 PGID 发 `KILL`。冻结 v1r1 finalizer 随后验证 A0/A1
两臂 `model_200/500/1000.pt` 与唯一 `motion_imitation_body_names` 差异，paired result SHA 为
`30ba716b...d7d9`。该结果明确 `same_immutable_signed_paper_judged=false`，所以它只完成 checkpoint
证据，不回答“不模仿非击球臂是否更好”，也不授权第二 seed。见
[实验卷宗](../experiments/non_striking_arm_imitation_ablation_20260713.md)。G05 仍为 `Partial`。

### 2026-07-14 signed-face K100 paper runtime materialized

Pod1 使用 clean detached `748b6d5` source 和 exact rebound exam bank `60e1a7ad...d1ca` 完成单次
CPU-only consume。新 schedule 是 `100` 个唯一题、正反手各 `50`，file/semantic/question-order SHA 为
`f2777dcd...1ca` / `3ca4bdba...3365` / `09f778f2...bd0`；最后写出的 paper-only activation
file/content SHA 为 `e0125b0e...bb4` / `533beb03...3d8`。完整 receipt 见
[`phase1_signed_face_exam_k100_runtime_receipt_20260714.json`](../../configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json)。
它没有消费 checkpoint，也明确不授权 trainer、judge、L2、第二 seed、停止/晋级或部署；所以这不是
Isaac 行为结果，G05 保持 `Partial`。

### 2026-07-14 signed-face K100 checkpoint attestor source gate

paper 后的 generic [`checkpoint attestor`](../DEFINITIONS.md) 已完成 source/static gate。它不自动挑“最新”
模型；每个候选 request 必须显式绑定 checkpoint path/bytes/SHA、filename/embed iteration、相邻
`params/training_contract.json` SHA、fresh lineage integer `1` 和 producer claim canonical SHA。hard contract
必须逐类型保留 `deploy_parity_face179`、`shared_plus_y` 与 exact float
`mount_normal_sign_per_clip=[1.0,-1.0]`，并把 31-joint plant 字段算入 semantic SHA。

同一 request 还冻结 clean source commit/tree、judge/evaluator/scorer/schedule source closure、checkpoint 与
evaluator Python runtime fingerprints、MJCF bytes/SHA，以及 actual immutable schedule 和 activation 的
file/content SHA。consumer 必须直接读取 actual activation；旧 runtime receipt 摘要中的 integer `[1,-1]`
已由 versioned correction pointer 保留并降级，不能作为 signed numeric type 权威，也不能用 Python 数值
等价放行。

全部 no-write 检查通过后，consumer 才在由 checkpoint SHA 唯一导出的 no-clobber namespace 先写 evidence、
最后写 claim；partial 或已有 root 均不可复用。claim 仍标记
`attested_not_executed_no_decision`，judge/trainer/L2/第二 seed/stop-promote/formal score/部署/真机全 false。
路径通配/穿越、symlink ancestry、checkpoint/contract/request 中途替换、dangling namespace 与
evidence-only partial 都 fail closed。focused 攻击回归为 `21 passed`，rebase 后仓内 `tests/` 为
`956 passed, 9 skipped`，`py_compile` 与 `static-validate` rc0；没有 Pod 连接、runtime request、
checkpoint evidence/claim、judge 或训练结果。因此只是 E1 source gate，G05 继续 `Partial`。见
[实验](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)与
[操作](../operations/run_phase1_signed_face_k100_checkpoint_attestor.md)。

### 2026-07-14 C3/D3 signed-face K100 paired execution source gate

最小 one-shot consumer 已 exact 绑定 paired L1 receipt `bb3cd749...bbde`、C3/D3 两份终档四键
（checkpoint/hard/producer claim/terminal）、generic checkpoint attestor 和同一 K100
schedule/activation。任一侧 attestation、actual float `[1.0,-1.0]`、独立 eval worktree/runtime、MJCF/plant、
env.yaml 或 no-clobber namespace 缺失都拒绝；现有 judge 只允许在 distinct empty GPU 上顺序运行，不写训练
run、不发 signal。focused `28 passed`，static/source-plan rc0；尚未 runtime attest/judge，因此仍无行为结果，
G05 保持 `Partial`。见[实验](../experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)与
[操作](../operations/run_phase1_signed_face_c3d3_k100.md)。

### 2026-07-14 A2/B2 跨 Pod 热启动 L1 runtime source gate

全新 v2 consumer 把旧 plan-only A2/B2 收口为两条独立 one-shot 探索 L1：Pod1 GPU0=A2 对照、Pod2
GPU0=B2 signed-face guidance。两条都绑定同一父 checkpoint、clean source commit/tree、`512×25`、唯一
`task.plant.zero_joint_friction=true` argv、outer claim、runtime zero-friction marker、31/31 zero hard
contract、fresh namespace/empty GPU 和 no-retry；terminal 必须是 finite `model_13824.pt`、lineage0。
focused `27 passed`，static/plan rc0；actual-host 由两 Pod GPU0 UUID 而非 CLI 自报绑定；跨 Pod pair
finalizer 完整比较两 hard contracts，并把 current-only 值锁到预注册值。尚无 Pod runtime/checkpoint/行为结果，不授权 judge/L2/第二 seed/
晋级/真机，G05 保持 `Partial`。复现见[操作](../operations/run_phase1_signed_face_a2b2_l1.md)。

### 2026-07-14 动作专属轻量 YAML 队列 source gate

新增探索训练用的单一 YAML 入口：每条 job 把 action/motion、专属 train bank/immutable exam、source
commit、base recipe/唯一 delta、seed、训练预算、checkpoint milestone 和资源策略放在一起。默认
`plan/status/launch-next` 都是 dry-run；只有显式 simulation-only token 才能启动一条 `ready` job，
`blocked` 永不调度。当轮六卡先各放一条再进入下一圈，Pod1/Pod2 的每卡 `4/3` 只是 2026-07-14
队列的临时容量合同，不是永久 GPU 归属；后续队列必须按当前占用重新审计并明确自身容量。

探索入口只查 clean commit、必需资产存在、GPU 容量、重复 claim 与 Kit boot lock，不引入逐文件 SHA、
pip/import closure 或 receipt；三个 runner 入口固定为 canonical repo-relative 路径，ready placeholder
在 SSH 前拒绝，并用全局 scheduler flock 包住六卡重采样/round-robin 选槽/launch；GPU 占用按唯一
numeric PID 计数，拒绝 `nvidia-smi` 重复行导致的假满。正式晋级/Gate3 仍用严格合同。focused 测试与静态检查见
[操作文档](../operations/run_lean_training_queue.md)。当前示例全部 blocked，尚无 Pod/训练/行为结果，
G05 保持 `Partial`。

#### Fresh C 五机制 attempt-1 基础设施失败与 harness 修复

active queue 的五个 attempt-1 都在 Pod1 GPU0 创建 claim 并启动过子进程，但均为 0 update、
pre-marker rc1、无 model，PID/PGID 已退出。根因是 setup 只导出 `HOPE_WBT_PYTHONPATH`，旧 launcher 的
raw Python 没有收到 `PYTHONPATH`，触发 `ModuleNotFoundError: whole_body_tracking`；因此不能据此拒绝任何
机制。旧 namespace/log/claim 已保全并标为 `rejected`，五个完全同 recipe 的 `retry-v2` 是唯一允许的
基础设施重试。

harness 现让 doctor/trainer 共用 CUDA+source-first PYTHONPATH builder，在 `mkdir/claim` 前验证 clean exact
source、assets 和 exact `find_spec` origin；SSH 错误保留 phase/stdout/stderr，launcher 等第一个
`Learning iteration`。`doctor --live` 不写 run 状态，并明确不声称无 Kit Hydra compose；`fill` 由单个
scheduler 进程逐条 doctor→claim→first iteration→重采。focused `17 passed`；retry-v2 尚未启动，
无 checkpoint/行为结果，G05 继续 `Partial`。见
[实验](../experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)与
[操作](../operations/run_lean_training_queue.md)。

只读 runtime doctor 随后在五个 retry-v2 分配槽全部返回 `DOCTOR_OK`：两 Pod source/assets/exact module
origin 通过，实际 child Python 为 `/workspace/hope_isaac_venv/bin/python`；live 六 GPU occupancy 为 0。
该命令没有创建 retry-v2 claim 或 trainer，Hydra compose 仍明确未运行，所以只把 harness 从 E1 提升为
module-runtime preflight 通过，不改变 G05 `Partial`。

随后单一 `fill` scheduler 把五条 retry-v2 依次铺到 Pod1 GPU0/1/2 与 Pod2 GPU0/1，并全部越过第一个
`Learning iteration`。2026-07-14T10:09:52Z 五条仍存活于 update `103–160/1001`，无 fatal；五份
`model_100.pt` 的 filename=embedded iteration、finite、相邻 schema-3 hard-contract SHA 与 fresh lineage
均通过。第六格 qdot-limit tail 已冻结为同 fresh-C 配方、weight `-5.0`/margin `0.85`，只授权单 seed
`+200` direction screen；同 source weight0 matched control 前不得作因果采用。G05 仍为 `Partial`。

#### Lean queue 发射前 P0 合同收紧

后续 source gate 把高频误发模式挡在 claim 前：recipe 的 `key/+key/++key` 统一后必须无重复，也不能覆盖
seed、预算、run、device、motion/bank 或 launch-claim 等 harness-owned key；Hydra flag、删除语法与
interpolation fail closed。`run_dir` 在整份 YAML 内唯一且不能位于 ready source 内，远端只能以不带 `-p`
的原子 `mkdir` 创建全新 namespace，已有目录/文件/symlink 都拒绝。

standalone doctor 和 launch 内置 doctor 现在都用同一条最终 override 向量执行
`train.py --cfg job --resolve`，在 claim 前完成 no-Kit Hydra compose。canonical claim content 绑定 source、
完整 caller argv、run name、预算/milestones、motion/bank/exam identity 和 Pod/GPU，其 digest 自动加入真实
trainer argv 的 `training_launch_claim_sha256`，claim envelope 同时保存完整执行 argv。focused
`19 passed`；本变更没有连接 Pod、启动 trainer 或产生 checkpoint/行为结果，且尚未增加 source-specific
asset/cache warmup 的 phase marker，因此 G05 保持 `Partial`。复现见
[操作](../operations/run_lean_training_queue.md)与
[实验](../experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)。

五机制 `+500` 随后证明并非全失败：V1+V2 出现 composite `0.0893` / normal pass `0.268` 的强精度信号，
但 completion/fall 仍差；V2 单独格 completion `0.176` / pre-fall `0.751`，成为唯一可替换格。五份
`model_500.pt` 均 finite/contract/lineage 通过。qdot attempt-1 则在第 0 update 的 A3 URDF import 阶段
停住，900 秒后由 launcher 精确终止 PGID `323083`；没有 hard contract/model，故只作基础设施失败，
同 recipe 的全新 retry-v2 namespace 才允许重试。G05 仍为 `Partial`。

qdot retry-v2 随后在 Pod2 GPU2 通过 no-Kit compose 和真实 boot marker，并到 iter `79`；schema-2 claim
digest、96 项 `/proc` argv、`model_0.pt` finite、hard contract 与 fresh/claim lineage 全匹配，fatal `0`。
它只关闭“机制能否真正进入训练”的运行门；未到 `+200`，且同 source/seed 的 weight `0` 对照未运行，
不构成 reward 结论或晋级。

qdot weight `0` 同-source control 与 conditional-face `0/-0.4` 配对已在结果前写入 active YAML；三条均
只允许 Pod2 dispatch、seed3、4096×1001、save100 与 `+200/+500/+1000`，且各自绑定 exact source、
motion/bank/exam/plant。它们尚未 launch，不能把 machine prereg 写成 runtime 通过。

qdot control attempt-1 随后也在 iter `0` 的动态 URDF importer 返回前停住，无 contract/checkpoint；
成功 treatment 有相同 `libGLU`/malformed-axis warning 却完成 scene creation，因此 warning 与 qdot weight
均非差异根因。exact PGID `327651` 已保全日志后收口；只允许完全相同配方的 retry-v2 fresh namespace
再试一次，若同 phase 重复则停止 retry。G05 仍为 `Partial`。

lean harness 因此新增默认不执行的 `boot-warmup` source gate：只从预注册 job 派生 1 env×2 update、独立
claim/namespace、180 秒 boot 上限的非科学探针，reserved Pod 与科学确认 token 均不能授权它。聚焦 queue
suite `23 passed`；尚未在 Pod 执行，所以这里只是 E1，不证明 importer 已稳定。

随后 conditional source 在 Pod2 GPU1 的首个 warmup 自然退出并通过 2/2 updates；`model_0/1` finite，
embedded iter、schema3 contract、claim 与 fresh lineage 匹配，fatal0，且 claim 明确 `not_science`。这把该
source/host/GPU 的 cold-boot 门提升到 E2，但不构成 conditional control/treatment runtime 或行为证据。

通用 Kit launcher 同时加入默认 180 秒 stale-log watchdog：只在日志非空后跟踪 size/mtime，增长即重置；
marker 同 poll 优先，stale 时只收口已验证 pid=pgid 的自身进程组并写 sidecar，rc125；空日志保持 hard
timeout rc124，stat 异常 rc126。专项 `9 passed`、相关 retry/queue `50 passed`。这缩短卡死占槽时间，不是
URDF importer 根因修复或 runtime 成绩。

后续红队发现“spawn 时验证过 PGID”仍不足以授权数分钟后的 signal：PID/PGID 可能复用，且 TERM 后只看
leader 会把仍存活的 child 漏掉。watchdog 现把 leader PID=PGID、双读一致的 Linux starttime 与
`getpgid` 写入 adjacent evidence；TERM 前再验证并冻结整组成员，TERM wait 枚举整组，KILL 前只接受该成员
集合的 exact 子集。leader 在 TERM 前消失、PID reuse、读中漂移或新成员加入都 no-signal + manual-review；
leader 在 TERM 后退出、已绑定 child 残留则仍可按其原 PID/starttime/PGID 安全收口。该项只是 E1 source
安全门，未连接 Pod、未 signal 任何远端进程，也不改变训练配方，G05 仍为 `Partial`。

对应的 marker-priority 回归不再用 `sleep 0.5` 与一秒 watchdog 竞争调度；测试 shim 在第二次 marker probe
同步写入 marker，从而稳定覆盖“同一轮 watchdog 已到期时 marker 仍优先”的语义。生产 launcher、timeout
和 signal 路径均未改变，专项 launcher/process-group 回归为 `15 passed`。

并发发射的另一根因也已定位：`flock FILE command` 的 lock fd 被 detached trainer 继承，导致每 GPU
名义容量 3/4 实际只能再发第一条。lean harness 现由短命 controller 持 fd8，launcher child 显式
`8>&-`；新增 preferred-slot 容量/回退测试后 queue suite `24 passed`。现役 qdot 两条仍持旧锁，不做信号，
只让新发射使用修复。

资源边界随后切换为 Pod2-only：Pod1 的三条 Codex trainer 在 iter `792/782/743` 由 exact PGID `TERM`
收口，`model_700.pt` 与日志保留，未发 `KILL`；Pod1 复核无 Codex compute process并全部交给 Yikang。
active queue 的 `dispatch_pods: [pod2]` 是可执行合同，不依赖聊天记忆；旧 Pod1 claim 仍只读防重复。

### 2026-07-14 31 关节 qdot-limit hinge 源码门

VirtualBall reward stack 新增默认关闭的
[`qdot-limit hinge`](../DEFINITIONS.md#qdot-limit-hinge)：它计算
`mean(relu(abs(qd)/joint_velocity_limits - margin)^2)`，默认 `margin=0.85`，只接受非正
Reward 权重。实现直接消费 `robot.data.joint_vel` 与 `robot.data.joint_vel_limits` 的同一 31 关节
runtime order；任意关节子集/重排、零值、非有限上限或不同 environment 的 limit 漂移都会 fail closed，
不能退化成 `action_rate` 代理。

Hydra 的 `joint_velocity_limit_hinge_weight` / `joint_velocity_limit_hinge_margin` 已走 fail-loud
translation，并把 applied marker 写入启动日志。post-override weight、margin、公式、31-joint identity
order 与 runtime limit 来源同时进入 training hard contract；未来 outer launch claim 必须再绑定 exact
argv/manifest。reward-layer focused qdot tests 为 `21 passed`，整个 dependency-light override 文件为 `76 passed`；
actual reward math 的 Torch/Isaac-stub focused tests 为 `3 passed`，schema-3/launch-claim suite 为
`62 passed`。合计 qdot-focused selection 为 `30 passed`。这仍是 E1 source gate：没有 Pod
训练、runtime marker、checkpoint 或行为成绩，也没有授权第二 seed/judge/晋级，G05 保持 `Partial`。

### 2026-07-14 不逃离就绪区的固定预算 Reward 源码门

为区分“静态 signed-face 权重不对”与“拍面 Reward 在触点/拍速尚未就绪时争自由度”，新增默认关闭的
[`racket_face_conditional_guidance_weight`](../DEFINITIONS.md#conditional-face-guidance)。它只在 wide
strike window 内花固定成本；位置误差用 `9.5→7.5 cm`、完整拍速向量误差用 `1.0→0.5 m/s`
形成就绪度。未就绪时成本为 1，进入门后连续换成 `15°→180°` 的拍面误差分数。故就绪度提高不会
增加成本，策略也不能靠退出门来免罚；门外拍面梯度为零。输出仍在 `[0,1]`，非正 weight 的绝对值是
每个时间窗 step 的最大罚金预算。拍面对仍强制走 raw-A/target-A 的共享 `_face_pair`。

Hydra 只暴露一个非正 weight 轴；门宽和公式随 source 固定并进入 training hard contract。默认 off、
数值/compact support、无效 bounds、raw-A 接线和 override/hard-contract 负例已写入 focused tests。
机制 math/梯度/单调性专项 `6 passed`，override 全文件 `78 passed`，raw-A face suite `34 passed`，schema-3/
launch-claim `62 passed`，`py_compile` 与 `git diff --check` 通过。
源码与反向激励反例已合入 `main@61007e9`；当前没有 Pod runtime、checkpoint 或行为结果。后续
control/treatment 必须同新 source、同
seed/动作/bank/plant，只改 conditional weight `0/-0.4`，并按 `+200/+500/+1000` 早判。安全/self-hit/
fall/guard 退化不可由拍面收益补偿。G05 保持 `Partial`；见
[实验卷宗](../experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)与
[训练操作](../operations/run_training.md)。

#### Conditional 正式启动推翻 1-env warmup 授权语义

同 source 的 Pod2 GPU1 `1 env × 2 updates` warmup 通过后，4096-env matched control 仍在 dynamic
URDF import 阶段停止日志增长：PID=PGID `332786`，没有 scene-created marker、hard contract、checkpoint
或 `Learning iteration`；claim digest 为 `caffd19e...da52`，实际 argv 与 claim 匹配。2026-07-14T12:10:05Z
只向该精确 PGID 发出 `TERM`，30 秒未退出后只向同一 PGID 发出 `KILL`；最终 pre-marker rc137，日志、
claim 和 launch sidecar 全部保留。serial fill 没有创建 treatment directory/claim，故没有第二个失败 arm。

这不是 conditional Reward 负结果，而是启动 harness 反例：1-env 只可回答最小 cache/import 路径，不能
代表 4096-env scene recipe。旧 source610 配对永久撤销；下一次 formal pair 必须换 fresh source/namespace，
让 stale-log watchdog 随 source 固定，并先以相同 source/GPU/4096 environments/scene recipe 的非科学
full-scene probe 越过 first iteration，同时写 trainer-owned runtime binding。上述能力和新运行证据闭合前，
G05 继续 `Partial`。

#### Trainer-owned binding、milestone attestor 与全规模非科学 probe

`main@0b632c7` 让 opted-in trainer 在真实 RSL log directory 选定后原子发布 `run_binding.json`，绑定
queue claim、source actual HEAD/clean、PID=PGID、`/proc` starttime/argv、物理 GPU 与预注册 milestones；
attestor 不再 glob/latest 猜 checkpoint，只沿 binding 打开 exact `model_N.pt`，复核 filename/embed iter、
floating/complex finite、相邻 schema-3 contract SHA、fresh lineage 与 launch claim，再 no-clobber 写 receipt。
旧 source 默认关闭，不能被追认。

1-env warmup 被真实 4096-env 卡死反例推翻后，`main@077e70c` 又加入独立 `full-scene-probe`：保留 exact
source/动作/bank/plant/GPU 和正式 `num_envs`，只改为 2 updates/save1；输出位于按 job/source/Pod/GPU/attempt
隔离的非科学 namespace，明确 `attestable=false/promotable=false`，专用确认词、reserved Pod、重复路径、
zero/placeholder source 与环境数漂移均 fail closed。source focused/扩展套件分别 `73/109 passed`。

最初的 Pod2 clean detached exact `077e70c` source 已准备 motion、train bank 与 K100 exam；它后续因缺
Git-ignored A3 tree 在 iter0 自然 fail closed。当前 conditional P1 pair 与 V1+V2×base-decel pair 已改绑
strict `main@caeb9ad` source；该 source 的 full-scene probe 已通过并显式解锁两对，但 scientific trainer
尚未点火。因此这里已有 E2 启动/终档运行证据、没有 Reward 行为结果，G05 维持 `Partial`。

full-scene probe 首次 dry/capacity preflight 又发现 execute 原先会复用 all-Pod `live_snapshot`：Pod2-only
probe 也会访问 reserved Pod1。P1.2 将该路径收窄为 selected dispatch Pod/GPU 唯一 PID 计数，未知输出、
空 dispatch 或达到容量均 fail closed，远端 fd8 二次容量检查不变；普通 fill 仍使用 all-Pod claim 快照。
这是 source gate 修复，不是 probe runtime 通过，修复合入前没有创建 run directory/claim/process。

同一轮控制面复核还发现 `fill` 每臂先通过独立 SSH 跑 standalone doctor，随后的 `_launch_script` 又在远端
短锁内重复完全相同的 source/assets/module/Hydra compose。第一遍既不保留容量也不写 claim，不能提供额外
安全，却多出一个网络与 compose 失败面。P1.3 删除 execute 路径的前置重复调用；每臂现在只剩一次原子
launch SSH，且内置 doctor 仍严格位于容量、namespace/claim 与 Kit spawn 之前。standalone doctor/dry-run
保持不变；这是 source/control-plane gate，不是 Pod runtime 或行为结果，G05 仍为 `Partial`。

P1.4 再关闭“看到 first iteration 就误当终档”的缺口。full-scene probe 现在预注册唯一内部
`milestones=[1]`，trainer 在独立非科学路径发布 claim/binding；source-pinned supervisor 只自然 `wait` 并
no-clobber 记录 normal rc 与 signal 的区别，绝不发 signal。selected-Pod-only finalizer 要求绑定的 trainer/
supervisor 和原 PGID 全部自然消失，再核对 current expected claim、scene→hard-contract→first-iteration phases、
fatal0、finite/embedded-iter1/fresh-lineage1 `model_1.pt`、adjacent schema-3 SHA、exact supervisor argv、
claim/source-asset receipt 与 motion/train-bank binding。still-live/orphan 不写终档；其他终态
失败写 immutable `unlock_authorized=false` 结果且禁止自动 retry。普通 milestone attestor 明确拒绝该
`attestable=false` binding，queue 也没有自动 unlock consumer。dependency-light 整合 focused `126 passed`；尚无用
合入 source 产生的远端 claim/binding/exit/result，因此这里只是 E1 source gate，G05 继续 `Partial`。

#### qdot matched pair `+500` mixed signal

同 source/seed 的 qdot weight `-5/0` 两份 `model_500.pt` 已通过 filename/embed iter、finite、fresh lineage、
hard-contract 与 queue-claim binding。updates `480–500` 配对均值中 treatment 的 raw qdot max、near-limit
fraction、torque saturation 分别下降 `16.4%/20.1%/35.5%`，pre/post fall 也改善；但 position pass 从
`0.418` 降到 `0.107`，position error 从 `0.219 m` 升到 `0.311 m`，exact composite 两边均为零。日志还缺
activation denominator 与 normalized/per-joint tail，所以只能判 mixed signal：不采用、不买第二 seed，
等待 terminal checkpoint 的 immutable judge；G05 不因此晋级。

matched control 后续自然完成到 `model_1000.pt` 并退出。该文件 SHA-256 为 `b6672869...12cb9`，
filename/embedded iter=`1000`、76 tensors/1,762,717 elements finite、fresh lineage `1`；内嵌与相邻
schema-3 contract SHA 同为 `25faa6f5...da12`，queue claim 为 `c73ac441...8a959`，fatal regex 为 `0`。
对应 treatment `model_1000.pt` 也为 iter `1000`、76 tensors/1,762,717 elements finite、fresh lineage `1`；
model/contract/claim SHA 分别为 `8814debb...556e` / `3f6a532a...9091` / `3910e3e2...8fb6`，fatal `0`。
updates `980–1000` 的 21 点均值已出现晚熟翻转：treatment/control 的 position pass=`0.878/0.593`、
error=`0.0474/0.0962 m`、signed composite=`0.310/0.146`、virtual return=`0.454/0.265`，而 fall 与
completion 基本持平。因此停止低剂量/interaction 扩展，把 `-5` 保留为晚熟候选；同题 immutable
MuJoCo/vendor judge 尚未执行，G05 仍不晋级。

#### P1 full-scene probe 暴露 ignored A3 source closure 缺口

Pod2 首个 4096-environment P1 probe 在 `Learning iteration` 前自然 `rc=1`：exact detached
`077e70c` source 缺少 Git 忽略的 `assets/agibot_a3/urdf/model.urdf`，因此没有 hard contract、checkpoint 或
Reward 结论。archive donor 的既有接受树为 46 regular files、15,378,264 bytes、canonical SHA
`0137f59b...26c6`；其中 URDF 实际闭包有 43 个唯一 mesh 引用。source `git status` clean 不能证明 ignored
runtime asset 存在。

P1.4 source gate 因此让 YAML source 显式绑定 target/donor/commit/完整 tree 合同；新增 selected-Pod-only
`prepare-source-assets`，在 source 无 trainer 时用 source-specific lock、source 外 no-clobber staging、
`renameat2(RENAME_NOREPLACE)` 与 no-clobber receipt 水合。声明者的 doctor 在 Hydra/run-dir/claim/Kit 前
重算 donor/target、43/43 URDF closure、Git-ignore 并消费 exact receipt；science claim 自动绑定完整 source
mapping。旧行不声明时兼容。该 source-gate 提交当时不远程水合、不重发 probe、不改变 blocked 状态；
后续 strict caeb 结果见下文。G05 保持 `Partial`。

#### Pod2-only pre-probe 发射闩

P1 source 现改绑 exact `main@caeb9ad` checkout。
[`launch_authorized=false`](../DEFINITIONS.md#launch-authorized) 时 `fill/launch-next` 会在任何
SSH 前拒绝；status/doctor 与 probe 前置门只读取 `dispatch_pods=[pod2]`，不再访问 reserved Pod1。历史七条
ready 行同步改为 complete/rejected，新 conditional 与 V1+V2×base-decel 两对预分 Pod2 GPU1/GPU2。strict
receipt 通过后，两份队列已显式切为 `launch_authorized=true`、四条科学行 ready；尚未点火或产生科学
checkpoint，G05 继续 `Partial`。

#### Full-scene probe P1.5 终档诚实门

P1.5 收口首个 probe 的“短跑结束但没有 supervisor exit receipt”问题。launcher 现在只在精确 PGID 已按
既有 identity helper 收口后写 `pre_marker_exit/watchdog_error/stale_timeout/boot_timeout` 终态；finalizer 可把
该证据冻结为 **failure-only** 结果，绝不解锁或自动 retry。普通 exit-receipt 路径新增实际 scene telemetry：
`num_envs` 必须等于 claim 的 4096、物理球开关与 `pb_ball/pb_table/pb_table_visual` 必须都真实存在；claim/
hard contract 还必须分别证明 `deploy_parity_face179` 与 31/31 PhysX 零摩擦。schema-3 正式 validator 从
claim-bound clean checkout 的 dependency-light `training_contract.py` 直接载入，禁止经过会启动 Kit/Omni 的
package `__init__`。PID 已复用只证明原 identity 已退出，仍由双扫描 PGID closure 阻止 orphan；并发 finalizer
仍以 atomic no-replace 胜者为准，只接受 byte-identical 重放。增量 focused `100 passed`；源码门本身不等于
Pod runtime，G05 继续 `Partial`。

旧 `main@c7e1a90` 随后完成一次非科学基础设施 canary。其 `probe_result.json` 内容 SHA-256 为
`02780b52df27255eea096f34dda9a26e806ae3a196c233a46a2af1cde16c4186`，finite `model_1.pt` SHA-256 为
`a813ea9ba8c058cf5ed2f9a9a8f8fe3b95ec0903cd3702831b99736736738e68`，相邻 hard-contract SHA-256 为
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`；76 个 tensor 的
1,762,715 个浮点元素全 finite，fatal 命中为 0，trainer/supervisor 的原 PGID 自然为空。hard contract 也
独立通过正式 schema-3 validator，但 c7 `probe_result.json` 的 `unlock_authorized=true` 只属于旧终档语义；
它没有受 P1.5 结果绑定证明实际 4096 environments、物理球与 `pb_ball/pb_table/pb_table_visual`，不能追认
或解锁科学训练。

clean exact `main@caeb9ad` 随后用全新 attempt `caeb_strict_terminal_pod2_gpu1_a1` 通过严格门。result/claim/
model/hard-contract SHA-256 分别为
`0d03bd0305a56e56440b14e1f41278a26c0cad3a84cc1245325faed1ef29b1d1`、
`7437db488d8aa062aba8de91fb517362cc609a81900f0e953f80e15174c36ad5`、
`e1b79d142c13bc2df513b2a7311fbeb7b610fc64047e095c1a54c76571fe3106`、
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`。result 绑定 actual
`num_envs=4096`、`physical_ball=true`、三实体全存在、76 个 tensor / 1,762,715 个浮点元素全 finite、
fatal0、自然空 PGID 与 clean caeb source，故 `unlock_authorized=true`。该 receipt 已被显式队列变更消费，
两对当时均变为 ready。probe 仍 `not_science=true / attestable=false / promotable=false`，不产生 Reward
结论、不授权第二 seed、judge、晋级或部署；G05 保持 `Partial`。

#### Full-scene 终档的 ignored A3 当前闭包

红队复核发现：上述 caeb 时代的 queue shell 会在调用 finalizer 前重算 source-asset closure，但
runtime `finalize()` 自身只校验 immutable hydration receipt。因此直接调用 runtime 可以绕过这层当前资产
证明，而 wrapper doctor 与终档结果之间的 target/donor 漂移也没有观测摘要。

新 source gate 让 terminal authority 自身从 claim 导出 target、donor 和 receipt，用 no-follow 稳定读重算两棵
完整文件树库存，解析 `urdf/model.urdf` 并复核所有 mesh 引用闭包，同时核 donor exact clean
commit。任一当前闭包与 claim/receipt 不同即冻结 `unlock_authorized=false`；pass 结果在
`source_asset_receipt.current_closure` 分别保存 target/donor 的实测 tree SHA、文件数/字节数、URDF
闭包和 donor source 状态。checkpoint 内嵌 iteration 与 fresh lineage 也都只接受 plain integer，不再让
JSON boolean 借 `True == 1` 通过。直接 finalizer 绕过 wrapper 后分别篡改 target/donor、boolean
iteration/lineage 的负测均 fail closed；full-scene 专项 `39 passed`，整合 harness/source-asset 回归
`146 passed`。

caeb attempt 当时的 wrapper doctor 有通过记录，因此继续作为原语义的 E2 启动/终档证据；但它的
result 不含 `current_closure`，不追认新能力。本项目前只是 E1 source gate，新 exact source 尚未产生带该
字段的 Pod result，所以 G05 仍为 `Partial`。

显式 unlock 后，conditional control/treatment 已分别在 Pod2 GPU1/GPU2 越过 first iteration，PID=PGID
`357023/357679`；尚无 checkpoint 早判，不能形成 Reward 结论。随后 interaction control PID=PGID
`358331` 在 first iteration 前的 dynamic URDF import 报 `malloc(): invalid size (unsorted)`、`rc=134` 并
自然退出，treatment 未发射；claim/namespace 保全。这是新的启动基础设施失败，不是 V1+V2×base-decel
Reward 或行为失败，也不能把单边 attempt 记成 matched pair。旧 control 行已 rejected/no-relaunch；逐字
同配方 `control_retry_v2` 与从未 claim 的 treatment 均 ready，只允许同一 `fill --count 2` 事务先等 retry
first iteration 再发 treatment。该事务已按序完成：retry-v2 PID=PGID `359240`（Pod2 GPU1）和 treatment
PID=PGID `359872`（GPU2）均越过 first iteration，pair 现 live；尚无 checkpoint/早判。G05 仍为
`Partial`。

#### V1+V2×底座减速旧仪表 pair 自然终档

Pod2 的 retry-v2 PID=PGID `359240` 与 treatment PID=PGID `359872` 后续均自然跑到
`model_1000` 并退出。no-clobber 终档验身确认两边 filename/embedded iter=`1000`，76 tensors /
1,762,715 个浮点元素全 finite，fresh lineage=`1`，schema-3 hard-contract SHA 均为
`451cda47...2291`，精确 fatal 扫描为 0。control/treatment checkpoint SHA 为
`ad69bc70...9f75` / `dcfb9599...00e8`，receipt content SHA 为
`8c0b3750...415d` / `050f2657...5f00`。

updates `980–1000` 的描述性 treatment/control 均值：底座击球前速度
`0.15364/0.16714`，pre-fall=`0.06359/0.06569`，post-fall=`0.02390/0.02239`，legacy virtual
return=`0.35220/0.34512`。但旧 source 仍缺 V1/V2/base-decel 的 activation denominator/numerator，
且速度比 `0.9193` 未过预注册 `<=0.90`。因此这只是 finite 终档+弱方向证据，不是
Reward 采用结论；不买第二 seed、不 judge、不晋级。新 fresh 复测必须先让同 phase 仪表
source 过独立 review 和 exact full-scene probe。G05 仍为 `Partial`。

#### 同 phase activation successor 已合入源码门

history-reachable exact source `0f3900a612863faf326dca6ad3e8d38bfe8df3c9` 关闭了旧 pair 的仪表缺口。
control 和 treatment 都在 Isaac RewardManager 的 reward→reset→command 顺序中同一 reward stage
执行 `base_decel_activation_probe`；manager weight=`1.0`，但每个 environment 返回严格零，不改总
Reward。它逐项复用真实 `base_decel` 的 `v_gain/v_max/std`；treatment 在同一
`common_step_counter` 上通过共享 signature 去重，参数漂移则 fail loud。runner 只在 probe 为 active
RewardTerm 时每个 PPO update 消费一次 raw total，不对 environment 取均值。V1/V2 和 post-swing
的 count-level denominator/numerator 也与两份 replacement queue 一起绑定 exact source。

独立红队结论为 source MERGE，聚焦套件 `222 passed`；两个 `MotionLoader/PosixPath` 失败已在
origin/main 基线复现，本 source 未改相关文件。两份 queue 仍为 Pod2-only、`launch_authorized=false`、
jobs `blocked`。下一门是为 exact checkout 水合 content-bound ignored A3 资产，再在 Pod2 空槽跑自己的
4096-environment strict full-scene terminal probe；probe pass 也只能由显式 queue consumer 解锁单 seed
fresh pair。在 receipt 存在前不授权 trainer/judge/第二 seed/晋级，G05 继续 `Partial`。

#### Same-phase counter 的真实 InferenceMode 假绿已抓获

`0f3900a...` 的首个 Pod2 strict full-scene attempt 在 trainer 前被一康旧 launcher 继承的全局 Kit lock inode
阻塞。operator 只精确收口自己的 wait PGID，以 hardlink 保全旧 inode，并用 atomic replace 把 canonical lock
换到空闲新 inode；一康 GPU0 的 PID、启动时刻与旧 inode holder 全程不变。该 attempt 明确记为 preboot、
non-science、不可复用。

新 attempt 随后在 exact 4096-env scene 越过首个 `Learning iteration`，但 runner 第一次记录 activation total 时
因 normal-mode `zero_()` 修改 RewardManager 在 `torch.inference_mode()` 中创建的 scalar 而 fatal。故场景启动通过
不能解锁 scientific pair，`0f3900a...` 永久 NO-LAUNCH。本 source fix 只在 reset 三个私有 device counter 时重新
进入 inference mode，并新增 inference-create → normal-consume → next-step-reuse 回归；专项
`10 + 2 + 11 passed`。尚需新 exact commit/checkout、source asset hydration、全新 full-scene terminal result，
且 queue 必须显式消费 pass receipt 后才能从 blocked 改 ready；G05 继续 `Partial`。

#### GPU hard-slot 调度源码门

Pod2 GPU0 被一康占用时，旧 `preferred_slot` 只保证“优先”，目标槽满后会 round-robin fallback，不能作为
资源隔离合同。queue schema 新增 [`required_slot`](../DEFINITIONS.md#required-slot)：与 preferred 互斥、必须
属于 dispatch slot；该槽无容量时本 job 等待，但其他槽的独立 job 不被饿死。science claim、boot warmup、
full-scene probe 与 finalizer 均在 SSH 前强制同一 hard slot。泛化负测覆盖“GPU1 满载时该 job 不落
GPU0/GPU2、GPU0/GPU2 独立任务仍可调度”、非 dispatch required slot、ambiguous 双字段以及底层 claim
绕过。该字段不提供 matched pair 原子发射。V1+V2×base-decel replacement 已以 fresh `v4` namespace 改绑
exact `2c2d70d...`，control/treatment 分别 required Pod2 GPU1/GPU2。唯一 fresh 4096-env probe 已完成两个
update 并自然退出；finalizer 绑定 actual env、物理球/桌三实体、face179、31/31 零摩擦、schema-3、finite
model、source/asset closure 与空 PGID，result file SHA-256 `4b12854c...0b27`。queue 已显式消费 receipt，
`launch_authorized=true`。同一次 `fill --count 2` 随后顺序发射 control/treatment：Pod2 GPU1/GPU2 exact
PGID=`380610/381237`、claim=`576724de...a49d` / `1a529430...4c5`，均已越过首迭代，最近到 `25/11` 且
fatal0；GPU0 仍只有 Yikang。尚无 milestone checkpoint 或 activation/行为结果，G05 继续 `Partial`。

#### 10000-update 无随挥回放三格漏斗

为避免把 1000-update 早筛误写成最终消融，新增三条同源、同题库、同 seed 的长曲线：普通对照、
[`qdot-limit hinge`](../DEFINITIONS.md#qdot-limit-hinge) 权重 `-5`、以及 V1+V2 击球窗模仿放松。
三格共同关闭随挥状态回放，两个 treatment 分别只相对共享对照改变一个机制 bundle；不比较两个
treatment，也不声称交互。exact source `2c2d70d...607e` 已有 4096-environment full-scene terminal
result `4b12854c...0b27`，三条最终 argv 的远端 no-Kit compose 均返回
`hydra=exact-no-kit-compose`。机器队列强制三条 `required_slot=pod2/gpu2`，不会发到 Pod1 或 Pod2
GPU0/GPU1。200/500/1000 只可停损/看方向，2000/3000 看中段，6000/10000 才形成完整训练曲线；
仍需 immutable MuJoCo/vendor 同卷，不能直接采用、买第二 seed 或晋级。训练尚未产生 checkpoint，
G05 保持 `Partial`；详见
[实验卷宗](../experiments/2026-07/EXP-P1-LONG-NO-REPLAY-FUNNEL.md)。

普通对照 attempt-1 随后在首迭代前的 dynamic URDF import 日志静止 180 秒；launcher 写完 exact
PGID `410589` 的 pre-TERM/pre-KILL identity evidence 后只收口该组，`rc=125`、无 checkpoint，另两格
没有 claim。该 namespace 已 rejected，不能解释为 Reward 负结果。队列现先发 qdot/V1+V2 两个未消费
treatment，再允许普通对照逐字相同配方的唯一 retry-v2；任一新 attempt 仍须先越过真实
`Learning iteration`。G05 继续 `Partial`。

重排后的同一次 `fill --count 3` 已逐条拿到 `KIT_BOOT_READY`：qdot/V1+V2/control-retry exact PGID
`411519/412204/412899`。2026-07-15 04:15 UTC 只读复核分别为 iter `24/9/2`，fatal0、claim/binding
present；Pod2 GPU2 为 `97%` / 17154 MiB，GPU0/1 的 Yikang PID 未变化。尚无 model-200 或 activation
receipt，故只把训练状态从 ready 改为 running，G05 仍为 `Partial`。

#### 两张新空卡的六格长曲线补全

2026-07-15 15:33 UTC 的独立只读复核确认 Pod2 GPU0/GPU1 已无任何 compute PID，GPU2 三条现役长训
仍正常到约 `3.2k`、fatal0，latest `model_3100.pt` 均为 finite/schema-3/fresh lineage 且 contract/claim
exact。当前 exact-hit 仍只有约 `0–1%`，所以不能用 3.2k 的瞬时指标定胜负。

新增六条同 source/动作/题库/seed 的 10000-update 单 seed 格，按 GPU0→GPU1 逐圈发射：单独放开手腕
线速度模仿、`qdot=-1`、单独把击球窗模仿降为四分之一、`qdot=-2.5`、脚部朝向惩罚 `0`、脚部朝向
惩罚 `-0.6`。它们与 GPU2 普通对照/`qdot=-5`/两项模仿同时放松组成三个不同问题，不互相冒充
对照，也不复制 seed。200/500/1000 只能判结构与激活；只有真实击球后才有意义的指标在达到最少
eligible hit 样本前必须继续。队列/host 回归通过后才允许点火，G05 保持 `Partial`。详见
[实验卷宗](../experiments/2026-07/EXP-P1-LONG-SCALEOUT-SIX-ARM.md)。

首次铺池中五条已越过首迭代；“只把击球窗模仿降为四分之一”attempt-1 的 exact PGID `420947` 在
动态 URDF import 阶段以 `malloc(): invalid size` / rc134 自然退出，未到首迭代且没有 checkpoint。
claim/log/leader identity 保留，namespace 永不复用；这是已知 importer 基础设施故障，不是该模仿方式
失败。只授权逐字相同配方、全新 namespace 的一次 retry-v2；若同 phase 再失败，不再重试。

retry-v2 随后以 exact PGID `423502` 越过真实首迭代。2026-07-15 15:49 UTC，GPU0 的
V1-only/foot-orientation-0/V2-retry 为 `419643/422126/423502`，GPU1 的
qdot-1/qdot-2.5/foot-orientation--0.6 为 `420298/421479/422783`；两卡各三条。连同 GPU2 现役
`411519/412204/412899`，Pod2 三卡利用率 `97%/97%/91%`，九条日志除已归档 V2 attempt-1 外均
fatal0。新格尚未到 model-200，不作行为结论，G05 继续 `Partial`。

#### Pod1 连续挥拍平衡与 Reward 配比十二格

负责人重新授权 Pod1 后，三卡 live snapshot 均为 0 compute PID。新的单 seed 长曲线把 12 个槽分成
两个问题：非击球臂模仿开关 × 10/16/24 秒 episode 的 `2×3`，以及位置/速度/拍面跟踪的六种配比或
总强度。现役 `wrap_teleport=false` 在动作片段结束后保留机器人物理状态并进入下一拍，所以更长
episode 确实增加同一连续状态内的挥拍次数。队列按 GPU0→GPU1→GPU2 四圈发射，不复制 seed。

4096-env probe 已越过首迭代并 rc0；finalizer 仅因退出后 `/proc` starttime 变成 null 而把同一
PID/PGID/argv 判作 identity mismatch，该 harness 假拒绝另行修。科学池两个 attempt-1 分别以 exact PGID
`2152129/2157392` 卡在 dynamic URDF import 的 180 秒 stale 门；两者无 iteration/checkpoint、原 namespace
保留。各自唯一逐字同配方 retry `2158534/2159350` 均已越过首迭代。

2026-07-15 16:40 UTC，12 条接受臂全部 live、PID=PGID、exact source `2c2d70d...607e`、fatal0，Pod1
GPU0/GPU1/GPU2 各四条，利用率 `97%/93%/97%`，显存约 `23.2/23.1/23.1 GiB`，无外部 compute PID。
这只关闭满池发射门；尚无配对行为结论，稀疏回球机会不足时继续，G05 仍为 `Partial`。详见
[实验卷宗](../experiments/2026-07/EXP-P1-POD1-LONG-BALANCE-REWARD-GRID.md)。

### 2026-07-15 selected-action frame-0 等待 v2（design-only）

等待/恢复参考现已另建 machine-readable v2，而没有改写旧 A/B/C prereg bytes。连续 episode 在揭题前
只公开上一拍动作，并以该动作自己的第 0 帧姿态、全零 root/joint/body 线角速度作为恢复参考；原子揭题
后才切到新动作自己的第 0 帧零速度参考。XY 在参考阶段入口从 live station 捕获一次并冻结；参考切换
不得 teleport、reset 或清 observation history、executed last action、action/target delay、noise/dropout
与 per-swing bias。Ready 另按全部安全/可达容差合取，不能把 frame-0 等式或 Reward 加权分冒充 ready。

源码审计确认现役 hold 仍用 `default_joint_pos`，root/anchor 速度未 hold-zero，body XY 又逐 tick 跟随
live robot。只切 joint reference 会制造 mixed reference，因此本次未声称 source adapter；合同的 adapter、
immutable XY snapshot、atomic non-leakage、carry-state receipt、ready 数值阈值和 full-scene probe 都是
null。CPU design validator + `25` 个红队测试通过，`launch-check` 按设计 rc1。可复现命令见
[恢复操作](../operations/run_phase1_recovery_tuple_prereg.md)，详细语义见
[T1 接口](../interfaces/t1_event_training_contract.md#selected-action-frame-0-waiting-contract-v2)。
没有 Isaac、Pod、训练或真机行为证据，G05 保持 `Partial`。

#### 稀疏 Reward eligibility ledger 源码门

为防止 `+200/+500/+1000` 把“尚未击中球，所以 outcome Reward 没资格出现”误写成 setting 失败，
训练源码新增[`稀疏 Reward 资格账本`](../DEFINITIONS.md#sparse-reward-eligibility-ledger)。解析球路径在
`_vb_book_strike_step` 同一步记录 exact-strike opportunity、virtual capture、net clear、landing valid 与
legal return 的非衰减整数，并按 forehand/backhand 分账；不从 warm-up 抑制和 decay 混合的 EMA rate
反推。qdot 的 RewardManager-stage 零返回探针与真实 hinge 复用 runtime 31-joint math，分别记录
observed/hinge-active/excess；control active 必须为零，treatment active 必须等于 observed。

只写 receipt 的 classifier 固定总机会至少 `100`、每动作至少 `50`、每动作至少一次 capture，并要求
连续两个预注册 milestone 才给 `DECISION_ELIGIBLE`。零机会为 `NO_OPPORTUNITY_CONTINUE`，有机会但
hit-conditioned 分母不足为 `CENSORED_CONTINUE`，身份或计数闭包错误为 `MEASUREMENT_INVALID`，单个
完整 milestone 仅 `DIRECTION_ONLY`。所有状态都写
`automatic_trainer_action=CONTINUE_UNCHANGED`，绝不 stop/restart/promote/买 seed。

host focused 回归为 classifier `14 passed`、qdot/virtual ledger `4 passed`、Hydra translation `18 passed`。
这只是 E1：当前在跑 source `2c2d70d` 没有这些 exact counter，禁止用旧 EMA 回填；也没有真实 Isaac
logger/sidecar receipt。virtual 结果仍是解析 Phase A，PhysicalBall Phase B 的真实触球/过网/落台未测。
因此不改变现役训练格，G05 保持 `Partial`。见[实验](../experiments/2026-07/EXP-P1-SPARSE-REWARD-ELIGIBILITY.md)、
[接口](../interfaces/sparse_reward_eligibility_ledger.md)与
[操作](../operations/run_sparse_reward_milestone_classifier.md)。

#### 次日演示七组合的 inexact 严格续训门

为避免最后一晚再从零等待，新增 Pod2-only 专用续训 runner；generic lean queue 的 fresh-only 拒绝规则没有
放宽。七条都从 qdot、V1+V2 或普通对照的 `model_3500.pt` 严格加载 policy/value/optimizer，追加 5001
updates，并显式设置 `checkpoint_tolerant=false`、不允许缺 hard contract、允许合同变化。合同变化意味着所有
后代永久 formal-ineligible；它们只能用于演示候选排序，不能当正式因果/fresh 证据。

机器清单先以 `launch_authorized=false`、六行 blocked 合入。v2 用不落盘的 `parent-inspect` 逐字验证三个原始
claim/binding、完整 argv/source/run/process、checkpoint 的 actor/critic、非空 optimizer state/param_groups、
finite、embedded iter=3500 与 schema-3 hard/claim 双绑定；通过后唯一一次 attest 才把 checkpoint/hard/claim/
binding `O_EXCL` 复制到只读 fixed snapshots，并只对 snapshot 重验、写新路径 receipt。旧 v1 receipt 不可复用；
2026-07-16 02:10 CST 的唯一 inspect/attest 已通过并发布 receipt file SHA
`fd200bd65ee00d33fb50a73f5de8d011cd810498ef626a3ca9d3a63b5bff2f34`；checkpoint/hard/claim/binding
SHA 已回填，前六行显式切到 ready。六个旧 scaleout 的 model500
证据都保全且 GPU0/GPU1 occupancy 各 `<=3` 时，v2 可先用第 4 槽发前两条，不要求先停；其余四条只在
精确停止四个弱臂后补入，并保留 GPU0 V1-only/GPU1 foot-`-0.6`，最终每卡四条。focused host 回归
`17` 个专用测试及相邻队列回归通过；parent snapshot provenance 与七条后代首迭代现均已运行通过，
第七条不修改前六条 recipe 或 claim：它只用 qdot snapshot，在独立的 16 秒 base recipe 中把
`task.env.episode_length_s` 从 10 秒替换一次（Hydra key 不重复），并组合 V1/V2、qdot `-5`、拍面
`-0.4`、脚朝向 `-0.3` 和自由非击球臂。它硬绑 GPU2 第四槽，专门测同 episode 连续 3–4 拍累积的平衡债；
claim 绑定 `+200` 只判结构/激活、`+500` 判安全/平衡、`+1000` 排演示候选，稀疏命中为零不可早停。
前六条 canonical claim digest 的回归逐项不变；专项测试扩为 `19` 个。当前运行证据仍不足以判断行为，故 G05 保持
`Partial`。详见
[实验卷宗](../experiments/2026-07/EXP-P1-DEMO-HOTSTART-PORTFOLIO.md)。

后续首迭代门抓到两条基础设施失败：V1+V2 自由臂行 `pre_marker_exit/rc134`（malloc invalid size），
普通母本保守模仿行 `stale_timeout/rc125`。两组 exact PID/PGID/starttime 均已确认 absent，原 claim、binding、
log、launch/identity evidence SHA 已冻结，旧行改为 `rejected`，因此旧 claimed namespace 不再占 effective
slot。只新增各一次 recipe-identical `retry_v2`：新 id/name/dir，分别硬绑 GPU1/GPU0，loader 机器比较
parent/recipe/seed/budget/milestones 与 predecessor 相同；claim 绑定终态证据、`retry_of`、
`manual_retry_limit=1`、`automatic_retry=false` 和 `recipe_equal=true`。两条按 GPU1→GPU0 顺序错峰，第二条
只有第一条已消费新 claim 后才可选择。点火锁内还会在创建新目录前重算 predecessor 的 5/7 个证据 SHA，
双次扫描旧 PGID、解析绑定成员并逐 PID 检查 absent、拒绝残留 NVML context；leader 已退但同 PGID child
仍活的反例必须 fail closed。两条 retry-v2 已分别在 GPU1/GPU0 消费唯一新 namespace 并越过首迭代；旧
PID `429116/429974` 仍为 `/proc` 与 NVML 双重 absent。`32` 个专项测试通过，G05 仍为 `Partial`。

每条 launch 的 `phase=first_iter` 现在还要求日志同时证明 explicit hard-contract mismatch、从 snapshot
model-3500 恢复到 iteration 3500 和 `optimizer=resumed`，并从新 hard contract 核对本行 qdot/conditional-face
权重。v3 还在同一 GPU lock、trainer 调用前重算 checkpoint/hard/claim/binding 四个 snapshot file SHA；
FIRST_ITER 从 binding 复核 `/proc` PID=PGID/starttime/cmdline，等到真实 `Learning iteration >3500` 才成功。
自然退出、只打印 resume/3500 或 PID reuse 都只写 exact identity 处置证据；wrapper 不发 signal、不自动 replay。
首个 checkpoint 的 lineage=0 仍由专用 milestone receipt fail-closed。最终只读审计中，七条后代均从
`model_3500` 完整恢复并在 `3501` 首次观察到真实训练迭代；exact PID 为
`426506/427190/428347/431061/431910/432838/433601`，全部 live、fatal0。七个 `model_3700` 均通过
stable load、filename=embedded `3700`、74 个浮点 tensor / 1,762,715 元素全 finite（nonfinite `0`）、
schema-3 hard contract、checkpoint↔hard/claim/binding 与 `lineage_exact=0`。PID `426506/427190` 已出现
`model_4000`，但本轮未审其内容。七条继续到 `+500`；当前证据只关闭首迭代和首个 `+200` checkpoint 的
provenance 门，没有给出行为赢家、正式因果结论或 vendor MuJoCo 结果，G05 保持 `Partial`。

2026-07-16 05:02 CST（UTC 21:02）只读复核时七条仍 live、fatal0；PID
`426506/427190/428347/431061` 的 `model_4000` 全部通过 stable load、filename=embedded `4000`、
74 个浮点 tensor / 1,762,715 元素全 finite（nonfinite `0`）、schema-3 hard contract、
checkpoint↔hard/claim/binding 与 `lineage_exact=0`。PID `431910/432838/433601` 尚未到该里程碑，记
`UNKNOWN` 而非失败；七条均未出现 `model_4500`。日志末窗没有明确 activation count 或 eligible
sparse-hit count，instrumentation 因此保持 `UNKNOWN`，不得据零值排名或停臂；fall-rate 只作诊断，不是
正式安全结论。仍无行为赢家；后三条继续到 `+500`，七条再继续到 `+1000`，G05 保持 `Partial`。

2026-07-16 05:29 CST 的下一次全量审计覆盖双 Pod 全部 `24/24` 条：两边均三卡 `4/4/4`、全部
live/fatal0，24 份 latest checkpoint 的 embedded iteration、finite、hard/claim/binding 与 lineage 均通过。
Pod1 十二格约在 `model_1000–1200`，Pod2 七组合中五条已通过 `model_4000`、两条在 `model_3900`，另五条
保留长曲线在 `model_1400/1500/4500/4500/4600`。训练仪表仍没有 activation/eligible sparse-hit 整数
分母，Pod1 exact-hit 也只有约 `0.47%–0.54%`；因此当前只能保留自由臂时长、Reward 配比和 qdot/全栈的
方向诊断，不能形成行为赢家、停臂或正式因果结论，G05 保持 `Partial`。

#### 2026-07-16 rolling target/TTS 与题库 retiming 源码门

真实测试暴露出一个训练/部署共同缺口：planner 会持续更新目标，但 actor 可能看到上一份位置、速度和
拍面，旁边却是本 tick 的剩余击球时间。`main@704bf3a2` 为此新增
[`atomic planner tuple timing`](../DEFINITIONS.md#atomic-planner-tuple-timing)：显式
`source_timestamp_compensated` 模式让位置、速度、拍面、动作侧和 TTS 经过同一 delay/drop ring，并从
source TTS 扣除已知 step delay；`uncompensated` 只作 matched stale-time 负控；默认 `live` 保持历史
checkpoint 行为。只有 policy 改读 actor TTS，critic、Reward 和真值指标继续使用 live TTS。reset/dropout
也按整元组 backfill/hold，mode 与 delay 进入 schema-3 hard contract。

同一提交同时修正了 question-bank retiming 的假冲突：motion retiming 可以改变老师路径的时钟和参考
速度，但 schema-3 bank 的最终 demanded racket velocity 是对同一来球反解出的绝对物理答案，不随老师
动作一起缩放。仍然拒绝真正不相容的 `hitter_pure` target ownership，以及不重排同一来球却 mid-swing
换 bank row。Pod2 exact Isaac 环境中新增 10 个 timing/retiming cases 全过；两文件全集与父提交具有完全
相同的 6 个既有环境/fixture 失败，因此没有把它们冒充本次回归。该门只授权下一轮单 seed、formal-ineligible
工程组合；尚无长训练曲线、vendor MuJoCo 或真实行为结论，G05 保持 `Partial`。详见
[rolling supercombo 卷宗](../experiments/2026-07/EXP-P1-ROLLING-TIMING-SUPERCOMBO.md)。

随后 Pod2 以 0.7 秒/40 ms compensated 格直接跑 4096 environments × 2 updates：训练自然 rc0，first
iteration 已观察，`model_1.pt` embedded iteration=1，74 个浮点 tensor / 1,762,715 元素 nonfinite=0，
schema-3 hard contract、fresh lineage=1、fatal0 与自然空进程组均通过。严格 finalizer 没有假绿：Popen
后第一次 `/proc` 读取竞态令 exit receipt 的 trainer starttime 为 null，而 trainer-owned binding 有真实
starttime，故 immutable result 按身份不一致失败。该失败属于 supervisor 取证基础设施，不推翻已经独立
复核的 scene/training/checkpoint；不自动重跑 probe，另修 bounded identity capture。G05 仍为 `Partial`。

24 格 rolling 续训另有专用 fail-closed runner；generic fresh queue 不因本轮而放宽。runner 将每个 parent 的
`+200/+500/+1000/+2000` 物化为绝对 checkpoint，绑定原始 checkpoint/hard/claim/binding、完整最终 Hydra
argv 和自身 bytes，并强制同 Pod parent、六卡四轮、每卡四条。激活门只接受 exact
`activated_demo_only_inexact`，还会逐字段复核上述 full-scene 训练证据及 reviewed runner SHA；父模型只读门要求
`actor.*`、`critic.*`、全部浮点 tensor finite 和非空 optimizer state/groups。`88` 个 runner/generic queue
专项测试通过。随后每 Pod 一条只读 SSH 完成三份 parent 核验：embedded iteration 为 `1600/4700/4500`，
三者均 actor/critic 各 `8` keys、74 个浮点 tensor / 1,762,715 elements、nonfinite `0`、optimizer state
`17` entries / `1` group。队列静态 dry-run 为四轮各六条、每槽四条；随后 24 个唯一 claim 已全部消费，
最近完整可信状态为 22 条 live/fatal0，Pod1/Pod2 三卡分别 `4/3/4` 与 `4/4/3`。另外两条在首迭代前
由动态 URDF importer malloc `rc134` 退出，精确进程与 NVML context 均 absent；按基础设施拒绝保全且
不自动重跑，不构成 Reward 负结论。现只授权本轮 demo-only inexact 仿真续训，因此 G05 保持 `Partial`。操作见
[rolling timing 双 Pod 严格续训](../operations/run_lean_training_queue.md#rolling-timing-双-pod-严格续训2026-07-16)。

2026-07-16 10:20 CST 的下一次每 Pod 单连接只读审计仍为 22 live/2 importer rejected，22 条
PID=PGID/starttime/binding、source=`704bf3a` 与 fatal=`0` 全部一致；六卡利用率 `94–97%`。旧 budget-v1
诊断臂只到 `model_2200`，Pod2 最快两条只到 `model_5000`，尚未分别越过 exact-stop `model_3600` 和
本母本 `+500/model_5200`，故当前没有可按预注册合同执行的行为淘汰。

2026-07-16 11:29 CST 的只读刷新仍有 22 条 live/fatal0，Pod1/Pod2 三卡为 `4/3/4` 与 `4/4/3`，
利用率 `89–98%`。Pod1 budget-v1 只到 `model_2400`，所以 `model_3600` exact-stop 尚未解锁；Pod2 两条
quality-parent 后代已写 `model_5200`，但尚无 no-clobber milestone receipt。更重要的是，source/logger
审计确认现役 completion/pre/post-fall 是跨历史 EMA，pre/post numerator 混入所有 termination；ready/balance
又缺 phase eligible denominator/sum 和 content-bound parent baseline。这些字段不能重建预注册的两个独立
100-update 窗，当前 22 条因此只能做结构/finite/合同淘汰，行为状态必须为
“量尺不完整，继续训练”。专用 runner 新增的 source-bound milestone attestor 只闭合 checkpoint
身份，不把描述性 EMA 升格成行为分数；未来必须用 consume-once per-update 整数账或 checkpoint-bound
immutable exam 才能恢复自动行为淘汰。G05 保持 `Partial`。

2026-07-16 12:45 CST，Pod1 的单连接刷新仍为 `11 live + 1 importer rejected`、accepted fatal0；11 份
latest `model_2100–2600` 均通过 embedded/finite/schema-3 hard/claim/binding/lineage，budget-v1 尚无
`model_3600`。Pod2 的唯一连接按注册入口尝试第一份 `model_5200` checkpoint attestation，却在任何
runtime materialization、checkpoint load 或 receipt 发布前发现 actual immutable claim canonical digest 与
当前 YAML 重建值不一致并 fail closed；没有 retry、stop 或 trainer 变更。三代 launcher 的 claim digest
可在本地分别复算，但该错误未返回 actual，当前不能把差异归因写成事实。main 新增独立
`inspect-milestone-binding` source gate：它从 YAML 派生 job/Pod/path/registered milestone，在唯一 SSH 内
只读稳定自校验 actual claim/binding、进程身份和 checkpoint/receipt presence，并报告 runner SHA、budget 与
字段差异；不物化 attestor runtime、不写 receipt、不 signal。该 inspector 尚未在 Pod2 执行，原 attestation
也不得重试；G05 保持 `Partial`。

2026-07-16 13:46 CST，Pod2 唯一只读 inspector 已把该差异闭合为 producer lineage：actual claim/self
digest=`7878d92...`、runner=`428cbf...`，binding/process=`live_exact`、`model_5200` regular、receipt absent；
相对当前 `aee7132.../90d7f26...` 的完整 claim 只有 continuation runner SHA 不同，corrected budget 与
其余字段逐字相同。新的 attestation contract 只登记这两个 reviewed corrected-budget runner，并对每个 job
完整重建两个 exact claim 候选；remote actual 必须逐字段等于恰好一个候选，actual digest 再交给 runtime
复核 binding。它不是任意 diff 容差：旧 budget-v1、第三 runner 或 budget/recipe/source/parent/run/slot/argv
任一漂移仍 fail closed。Pod1 同轮 source/claim/GPU/fatal 健康，但临时审计脚本错误使用 regular-file
size/mtime 检查 `/proc`，所以本轮 identity/checkpoint 为 UNKNOWN，未停止任何臂。兼容 source gate 即使合入
也只解锁后续一次 no-clobber checkpoint attestation，不改变“量尺不完整，继续训练”或 G05=`Partial`。

约 15:10 CST，上述兼容门第一次真实消费通过：`rolling_p2_t05_comp2_j0_equal_f03@5200` 的 production
dry-run 只列两个 exact variant，唯一 Pod2 SSH 随后 O_EXCL 发布 receipt（content SHA=`521910d...`）；checkpoint
filename/embedded=`5200`、74 个浮点 tensor / `1,762,715` elements、nonfinite=`0`，schema-3 hard
SHA=`4e84c51...`、actual claim=`7878d92...`、binding=`4b9c5b2...`、lineage=`0`，取证时进程 live。没有
judge/stop/retry/第二 job，因此只把 historical-claim checkpoint source gate 从未实测改为实测通过，G05
整体仍 `Partial`。同轮 Pod1 `/proc` 专用双读也恢复 11 条 `live_exact` 与 latest `model_2600–3100` 的
finite/合同/optimizer 证据；budget-v1 只到 `model_3100`，`model_3600` 不存在且未 signal。

15:36 CST，第二个 registered job 也通过同一 source gate：其 per-job exact claim 摘要是
`691a52c.../0968d24...`，runner 仍严格为 `428cbf.../90d7f...`；唯一 Pod2 SSH 发布 receipt content
SHA=`37d6bd2...`，checkpoint=`ff1b210...`、filename/embedded=`5200`、全部浮点元素 finite，hard
SHA=`aa80162...`、actual claim=`691a52c...`、binding=`7593d66...`、lineage=`0`、process live。没有访问
第一 receipt、judge/stop/retry。两份 checkpoint source gate 均实测通过，但行为量尺仍缺，G05 继续
`Partial`。同轮 Pod1 latest 扩至 `model_2600–3200`；budget-v1 只到 `3200`，未 signal。

首次真实 continuation 又抓到 budget 字段语义错误：parent `1600` 配 CLI `max_iterations=3601` 时，RSL
实际报告 `1601/5201`，说明该值是追加 update 数而非绝对终点。首 trainer 健康且 binding 正确；本地等待
在 remote watchdog 前退出，未重发或 signal trainer。runner 已改为 CLI 传 `2001`，同时在 claim 中分别绑定
trainer arg=`2001`、absolute exclusive bound=`3601` 与最后 checkpoint=`3600`，并保留 first
marker=`1601/3601`。首条旧 schedule 只能作
inexact 方向诊断，到 `model_3600` 精确收口；剩余格使用修正合同。G05 仍为 `Partial`。

为不把两个独立 Pod 的首迭代等待串成一条长链，续训 runner 每批最多并发 Pod1/Pod2 各一条；同 Pod 永不
并发且仍经过 host Kit boot lock。两 future 全 settle 后才取下一份 live snapshot；一边失败时保留另一边
成功证据并立即停止后续批次，不 retry/replay；本进程 attempted overlay 又防止下一 snapshot 短暂漏 claim
时重提交相同 job。`92` 个 rolling/generic runner 测试通过；这是调度能力，
不改变任何训练配方或 G05 行为结论。

2026-07-16 17:00 CST，rolling 池因训练/部署共同的 task-revision 缺口被整体停止，不再等待无法由现役
EMA 日志产生的行为淘汰。专用 cutover 工具先逐条复核 claim/binding/source/PID=PGID/starttime/argv/cwd/
checkpoint/hard，再只 signal 记录的精确进程组；最终双 Pod 的 24 个注册 leader、22 个接受进程组与 NVML
compute context 均 absent。Pod1/Pod2 no-clobber recovery receipt SHA-256 分别为 `e6b2480a...8263e` 与
`4c370431...949`；finalizer 本身没有发 signal。最后稳定 checkpoint 约为 Pod1 `2900–3500`、Pod2
`5800–6500`，只保留为 inexact 结构证据。正式 179-D active swing 冻结目标/TTS、训练不消费同一物理球
递增 revision、0.5 秒卷尚未真正绑定 time-law，故旧 queue 标为 superseded，不能 resume。G05 继续
`Partial`；新池必须先过 task identity/revision、受限 phase governor、0.5 秒卷和整数行为窗口四道 source/
full-scene 门。

2026-07-16 task-revision replacement source 把这四道门实现为一个不可拆开的训练协议。每颗新球先
从显式加权 [`initial TTS mixture`](../DEFINITIONS.md#initial-tts-mixture) 抽样；分布同时包含低于 0.5 秒
压力层、精确 0.5 秒点质量、0.5–0.9 秒部署层和更长来球层，且每层/总数写入 hard training contract。
同一物理球只修改 actor-visible position/velocity/signed-normal/TTS；question-bank row、物理球、Reward
和 critic truth 均不可变。motion 与 racket command 共用 SHA-bound
[`phase governor`](../DEFINITIONS.md#phase-governor)，旧 hold clock 被强制清零，避免两个 preparation clock
重复扣时。runner 每个 PPO update 输出整数事件账，淘汰必须对两个互不重叠窗口先求和再重算比例；
eligible 分母为零的稀疏 Reward 永远不能据零值早停。

当前仍只有 E1：本地没有 Torch/Isaac full scene，K100
[`0.5-second timing exam`](../DEFINITIONS.md#timing-exam-0p5) 尚未实际运行，TOPP 也没有 0.5 秒动力学
证书。任何 successor queue 必须等 clean detached Pod probe 同时证明 finite checkpoint、schema-3
contract、mixture count partition、same-task revision activation 和 exact behavior ledger 后才可点火；
否则 G05 保持 `Partial` 且旧池不恢复。完整边界见
[task-revision cutover 卷宗](../experiments/2026-07/EXP-P1-TASK-REVISION-CUTOVER.md)。

首次真实 4096-environment successor probe `A4` 已越过 scene import 与 hard-contract 写入，但在
iteration 1 前因 planner revision 的 accept/reject command metric 被 eligible 子集短张量重绑而触发
CUDA 全局 env-id 越界。0.36 秒支路第 18 步缩集、首 PPO 窗 24 步，与现场时序吻合；同构 CPU
反例还能复现相同的最后三个失败位置。源码修为固定 `[num_envs]`、逐 tick 清零和按原 env id scatter，
并新增 4096-env/high-id partial-reset 回归。`A4` 无 checkpoint，进程与 GPU context 已 absent；修复版
尚未 fresh full-scene 通过，因此 successor queue 仍不得点火，G05 保持 `Partial`。

### 2026-07-18 exact-0.5 K100 直接行为结果

旧的版本化启动门反复在题目执行前制造假拒绝，已从现行操作路径删除。现在 evaluator 直接读取
checkpoint、题库和 timing paper；操作者不再提供 SHA 或维护 activation/receipt 版本。首次完整直跑暴露并
修复两项真实程序问题：planner 的 motion/racket-target 两端必须在环境构造前同时关闭旧 revision owner；
MotionLoader 的 body velocity property 是高级索引副本，必须写其 backing tensor 才能得到真正的第 0 帧
零速度参考。

修复后 Pod2 完成 100/100 题且结果文件状态为 `valid`：正反手各 50 题，触球与回台均为 `0`，物理摔倒
为 `0`，100 题都因 0.5 秒 deadline guard 结束。该 checkpoint 因此不能满足严格半秒要求；下一轮训练要
直接覆盖更短且更宽的准备时间、动作加速和同一拍 target/TTS 更新。Isaac 结果仍不是 vendor MuJoCo 门，
所以 G05 保持 `Partial`。

### 2026-07-19 W/Y checkpoint 静态制品闭合

前三轮定位因 launcher cwd 未入账而失败；后续一次 Pod1 只读全域精确查找不再推断 cwd，
按完整 `run_name` 为 W/Y 各得到唯一 regular `model_6700.pt`。两份 checkpoint 均为
iteration `6700`，均有 `74` 个浮点 tensor、`1,762,715` 个浮点元素且 non-finite `0`，
actor 形状均为 `179→31`；`params/training_contract.json`、`env.pkl`、`agent.pkl` 与
`env.yaml` 全部存在。这只是静态 checkpoint/导出材料闭合，不是 vendor 行为。

standalone exporter 的 `--plan` 源码门现会验证完整材料，并且在第一次写入前退出；
`checkpoint_iteration` 必须是非负整数，成功 JSON 明确记录没有写 artifact 或执行 graph export。
五文件聚焦回归为 `97 passed in 0.38s`，包含普通导出 fake smoke。真实 W/Y plan、ONNX、
vendor MuJoCo 和行为卷均未运行，因此 G05 保持 `Partial`。

### 2026-07-29 N=1 ActionBall Reward smoke：构造到 runtime bind，PPO 尚未开始

Pod1 的反手拉和 Pod2 的反手挡各运行一次 `1 env / 2 update` 构造 smoke。两边都在 exact clean
`bbefa277` 上通过 scene import、182D actor observation、effective Reward 回读与 q_des clamp
激活，但在 birth broker 绑定 diagnostic motion bytes 时同因 Traceback；没有出现
`Learning iteration`，因此不得写作 smoke PASS 或训练结果。运行日志、canonical argv、source commit、
PID/PGID 和 pre-TERM identity 均保留在各自 no-clobber namespace。

修复提交把 diagnostic motion bytes、unauthorized admission receipt、evaluator hard-contract 分支和
Motion↔Racket 初始化后 shared-state probe 合成一个无重入生命周期；host 可运行的相关回归
`236 passed`，Torch 行为套件仍待 Pod。solver/profile pins 与两个 N=1 contact bundle 已按新
`hope_commands.py` 重新物化。下一门是 fresh `_r2` 自然跑完两次 update、零 Python
Traceback、checkpoint finite、episode length 非恒 1，并回读 exact Reward/motion/manifest；
因此 G05 状态仍为 `Partial`。

`_r2` 两动作进一步通过 runtime bind、hard contract 与 effective Reward receipt，在首个 true reset
因 birth/task receipt 调用未传新必填的 broker registry SHA 同因停止；仍未出现
`Learning iteration`。调用点与 API 已对齐并重新物化 pins/bundle，下一门改为 fresh `_r3`，G05
仍为 `Partial`。

`_r3` 的 Pod1 反手拉已越过上述 registry 缺口，但在首个 update 前由 receipt
serialize→deserialize 自检拒绝：已是单位长度的 binary64 四元数被二次归一化后产生约 `1e-16`
表示差，导致 canonical SHA 不同；Pod2 因该确定性构造失败未启动 trainer。修复只让 unit-input
canonicalization 幂等，非单位归一化、符号规则和全部物理/安全门不变；核心回归
`119 passed, 14 skipped`，新 pins 与反手拉/挡 bundle 分别为 `52000401...f465`、
`baad5b95...acbf` / `0d3c80f4...92ab`。fresh `_r4` 仍须自然完成两次 update、产出 finite
checkpoint 且无 Python Traceback 才能解锁 canary；当前 G05 仍为 `Partial`。

`_r4` 两动作已证明 quaternion receipt seam 消失，但固定 mixture 在 level-0 第 4 个 birth 进入
frontier 时，旧 sampler 把 `current width == initial center width` 误判成没有合法 arm；两边均在
首个 update 前停止，`0 iteration / 0 checkpoint`。修复仍优先真正 promoted frontier；尚未扩张时
改采当前非零合法 support 的 outer band，并保留 stratum/arm/quota/receipt/exact replay；全零 scope
仍在 draw 前原子拒绝。联合回归 `171 passed, 14 skipped`，pins/bundle 内容不变。fresh `_r5`
仍须自然完成两次 update 和 finite checkpoint，G05 保持 `Partial`。

fresh `_r5` 又证明两边能完成 rollout，但 formal fingerprint 对 inference tensor 读取不可用的
`_version`，在 optimizer 前停止；修复只区分普通 tensor 与 inference tensor 的证据路径。`_r6`
的 Pod1 被 formal Reward terminal-edge conservation 审计挡在 optimizer 前；`_r7` 的 Pod1 已完成
一次 optimizer，却被 rollout-end curriculum receipt 的旧字段挡在打印/checkpoint 前。它们都是
新增证明层的失败，不是 PPO、动作或 Isaac 本体不能运行。

为尽快得到可判读 policy，exact
`e469d85b5c9f493e5c1fbb6861eefe84b0926a32` 只对
`training_authorized=false` 的 N=1 diagnostic 路径关闭 formal Reward/joint evidence fence，并
冻结 level-0 curriculum、跳过 rollout-end advancement；真实 Reward、`q_des` clamp、
soft/hard-limit penalty 和 hard-limit/table/fall termination 保持开启。由此产生的 run 只具有
diagnostic Reward-screen 权限，不能晋级正式 curriculum 或 Gate。

该 commit 的 fresh `_r8` 已在 Pod1 反手拉、Pod2 反手挡各自然完成 `1 env × 2 update`，零
Traceback，并产出 finite `model_0.pt/model_1.pt`，因此“能进入真实 PPO 并保存 checkpoint”的
smoke 门已过。Pod1 mean episode length 仍约 `1.0`，第二个 rollout 24/24 policy steps 为
`joint_qdes_forbidden`，故当前仅解锁固定 level-0 的 1024-env Reward canary；前 20–50 updates
必须观察 episode length 与逐关节 hard-limit 请求是否改善。G05 仍为 `Partial`，因为动态课程、
formal receipt、held-out 晋级、MuJoCo 与真机均未由这些 diagnostic run 证明。

#### 2026-07-29 `4ff48b21` 4096-env 运行证据

Pod1 当前保留三条固定 level-0、formal-ineligible 的 N=1 upper 诊断长跑；exact 身份与动态快照见
[`n1_live_wave_4ff48b21.v1.json`](../../configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json)。
每个 PPO update 固定是 `4096 × 24 = 98,304` 个环境步。11:20 UTC 的 update 墙钟为：

- 反手拉现行 Reward：`61.69 s`，即每个
  [`vector policy step`](../DEFINITIONS.md#vector-policy-step) `2.57 s`；
- 反手挡现行 Reward：`271.99 s`，即 `11.33 s`；
- 反手拉两倍动作模仿：`149.60 s`，即 `6.23 s`。

每轮 PPO learning 仅约 `0.1 s`，瓶颈是 collection 中的大量提前安全 reset。三臂最初五轮均无
strike opportunity；`mimic_x2` 的 raw mimic 项约精确翻倍，但 terminal/qdes 与同动作 control
没有分离，故“接线生效”已证明，“能改变早期行为”尚未证明。

反手拉 control 的 qdes reason 从 update 2 的 `48,681` 降到 update 35 的 `909`，mean episode
已接近/越过动作击球帧，说明 trainer 在学习避开限位；但截至该快照 strike 仍为零。下一 fresh
lineage 使用显式 opt-in 的 shared-ready actor 初始化与 `init_noise_std=0.02`，不改变 decoder，
不放宽 soft/hard-limit、table 或 fall 安全。该 bootstrap 只允许 exact N=1/N=5 shared-ready
动作；N=73/N=93 必须保持未启用，不能因常量 bias 实现而回归为不可启动。

full 反手拉的 scene/Reward 构造已过，但 runtime solver 在 face-center 到 official-site 的
teacher-rate 映射后存在低于 `0.6` 的接纳缝，导致至少一个 birth 零 receipt、未到 update 0。
这是物化器下界和缺少 post-solver admission 预飞的问题；修复必须保持 runtime
`teacher_rate_min=0.6`，重物化 full bundle 后重新 smoke。task-strong upper direct 又在 PhysX
simulation start 活锁，已按 exact process identity 留证停止且未重试。以上都不能计入 Gate
通过；G05 继续为 `Partial`。

#### 2026-07-29 finite q_des / reference-reset 候选切换

这是 `curr-launch-fix` 功能分支候选；只有合入 `main` 后才可能改变运行态，当前
`origin/main/docs/NOW.md` 仍是唯一主板。旧 v2 早期也有 `ee_body_pos=1,690–2,301
reset/update`，到 update 4181–4187 已学到 `1–7/update`。被当作旧性能基线的 `6.4 s`
collection 实际来自 mean episode length=`1`、`98,304 reset/update` 的失败 probe；同代码修正
stand hold 后代表值为 `4.49 s`。当前 ActionBall 反手拉约 `27 s/update`、主要受
`ee_body_pos` reset 影响；反手挡约 `48 s/update`、主要受 finite q_des reset 影响。后者切换后
预计节省 `14–17 s/update`，但这只是预算，不能代替 fresh 4096-env 证据。

候选合同将有限请求改为
[`finite q_des execution projection`](../DEFINITIONS.md#finite-qdes-execution-projection)，并以
weight=`-5` 的
[`qdes_projection_penalty`](../DEFINITIONS.md#qdes-projection-penalty)读取**投影前**归一化超出量；
`-20` 只允许作为明确消融。reference anchor/body/ee guard 使用
[`metrics_only`](../DEFINITIONS.md#reference-metrics-only)，保留逐步指标但不 reset。raw q_des
nonfinite、实际/physics-substep hard edge、table hit 和 fall 仍是 hard reset。仅由当前 q/qdot
预测出来的 crossing 继续触发 finite brake，但不在实际越界前 reset；真实 hard edge 仍由
`joint_actual_forbidden` 的 sticky substep 证据终止，所以这不是关闭物理安全。

老师贴限也不是 block 风暴的解释：loop/block upper/full 四件动作的 hard/soft/2%-inner crossing
均为 `0`；block 全片 normalized hard/soft margin 为 `0.115081/0.072312`，不小于 loop upper
`0.111954/0.068838` 或 loop full `0.113493/0.070548`。fresh smoke/长跑必须逐关节记录投影前
超出量、触发率、正负侧与贴边饱和占比，并同时记录：

- [`collection_vector_step_wall_s`](../DEFINITIONS.md#collection-vector-step-wall-s)；
- [`amortized_e2e_vector_step_wall_s`](../DEFINITIONS.md#amortized-e2e-vector-step-wall-s)；
- [`collection_environment_step_us`](../DEFINITIONS.md#collection-environment-step-us)；
- [`collection_environment_steps_per_s`](../DEFINITIONS.md#collection-environment-steps-per-s)。

在新合同下自然完成 smoke、finite checkpoint，并取得 4096-env reset/吞吐与 Reward ledger 前，
不能把预计提速或“policy 已学会限位”写成 Gate PASS。CaT 或 PPO bound loss 留给后续单变量候选，
不作为本轮 G05 前置；G05 保持 `Partial`。

#### 2026-07-29 `b1d299e1` source/config 证据

finite q_des、reference metrics-only、shared-ready bootstrap 与 full-scope solver preflight 已固定
为 exact `b1d299e1e57bd0909aa402ca2701b3901975337b`。ActionBall schema-3 hard contract 现强制
运行时投影 fact 为 exact `true`；legacy `false` 仍编码为字段缺席，保持旧合同字节。host
整合验证在修正一个测试 import 后为相关 `107 passed`，语法与 diff 检查 PASS。

该 commit 的 profile-pins 文件 SHA-256 为
`47a00a6a35ea4709603634deeb062febc3a6e7bb2b9f57aab5c573781d330488`。upper loop/block
bundle SHA 分别为 `29adc3cf...c85c4`、`fb1ed6ee...b6c5a`；full loop/block 为
`d94c7f0a...223a2`、`ca13d958...2f0f`。full 512-proposal solver preflight 分别接纳
`511/512` 与 `443/512`；后者只过 diagnostic 门，未过 formal canary rate 门。

这些仍是 E1 source/config 证据。尚未取得 clean Pod 的 live Reward/PPO/hard-contract receipt、
真实两 update checkpoint 或 4096-env reset/吞吐，因此 G05 继续 `Partial`。

Pod 第一次真实 compose 还发现 N1 reference override 少了 Hydra `+`；follow-up 已修正，并把
bootstrap/std/reference 三项补入 N5 formal launcher、把 full solver-preflight PASS 变成 N1
launcher 硬门。launcher 联合回归 `80 passed`，但尚未重跑 Pod smoke，所以证据等级不变。

#### 2026-07-29 `7a14b0b9` live smoke / 4096 反例

Pod1 clean checkout 已恢复与旧训练相同且 Git-ignored 的 A3 46-file 生成树，tracked 状态保持
clean。反手拉 upper 的 policy contract SHA 为 `8e07609d...0f4d`，effective Reward SHA 为
`c2f13419...6c11`。exact
[`smoke spec`](../../configs/n1_contact_20260729/smoke_loop_upper_gpu1_7a14b0b9.json)
自然完成 `1 env × 2 update`：iteration `2.85/2.02 s`，`model_0/model_1` 共
`1,775,488` 个 tensor 元素全部 finite，零 Traceback/OOM。

同 source 的
[`4096-env diagnostic spec`](../../configs/n1_contact_20260729/long_loop_upper_gpu1_7a14b0b9_r1.json)
进入真实 PPO，但首两轮为 `28.36/39.82 s`。update 6 的投影 sample、nonfinite 与投影罚均为零，
而 live 旧名 `joint_qdes_forbidden=0.05249/env-step`、actual hard
`joint_actual_forbidden=0.02490/env-step`。代码/teacher 复核证明前者混入 predicted/physical
crossing，且老师全片和 `q+qdot×5/20 ms` 都没有 2%-inner crossing；这是未训练 plant 动量和
重复 termination 所致，不是 finite proposal 或老师贴限。

successor 候选在 projection mode 令 q_des DoneTerm 只拥有 nonfinite raw request，predicted
crossing 仍生成 finite brake target 但不 reset，实际/子步 hard edge 仍由 actual DoneTerm 终止；
legacy 行为不变。Pod host joint-safety suite `80 passed`。fresh 4096 replacement 仍须证明
actual-hard 比率和 update wall time 可接受；G05 继续 `Partial`。

#### 2026-07-29 `5dbb4e58` 4096 replacement 与滚动刹车候选

exact 1-env smoke 两轮自然完成（`4.72/3.00 s`，q_des termination=`0`）。同 source 的
4096-env replacement 在 update 0--17 的 mean episode length 仍只有
`19.30--23.81` steps，约 `4,658--5,092 joint_actual_forbidden/update`，而 fall 只有
`0--23/update`；只在 update 12 产生一次 strike opportunity。update 3--8 对同序号旧 source
只快约 `2.5%`，所以 predicted reset 所有权修正正确但不足以解锁健康长跑。

这里的 `joint_actual_forbidden` 同时拥有当前真实 q 进入 hard-limit 内缩 `2%` 安全带和
physics-substep 真实 hard-edge sticky latch；它不能被误写成每次都已撞机械硬挡。q_des
projection/nonfinite/penalty 全为零，说明 clamp 目标合法，但不证明 implicit PD、重力和接触下的
真实关节没有动态超调。早期慢 update 本身不判死；必须观察 episode length 是否越过约
31-step `t_hit` 并稳定出现 strike。

单变量 successor 候选保持 Reward、Done、2% safety band 与 nominal safe q_des 不变，只让每个
fresh 5-ms substep readback 继续用完整 20-ms policy/control horizon 做滚动 crossing 预测和
brake；安全行 target 要求 bitwise 不变。host joint-safety focused suite `81 passed`，与
ActionBall runtime wiring 联合回归为 `125 passed`。尚未产生 Pod A/B，所以 G05 保持
`Partial`。

#### 2026-07-30 A3 qvel-fixed upper 双动作 smoke

tracked qvel-fixed motion、动作专属 bundle/manifest 和实际 composed Reward/PPO receipt 已在
Pod1 串行穿过完整 Isaac scene/runtime。反手拉
`n1_qvelfix_smoke_5ecf0e06_loop_gpu1_r3` 与反手挡
`n1_qvelfix_smoke_5ecf0e06_block_gpu1_r4` 均自然完成 `1 env × 2 update`，iteration 分别为
`4.65/3.18 s` 与 `4.67/2.92 s`；四个 checkpoint 均可载入且逐 tensor finite。

两条 run 的 q_des/table/fall termination 为零。N=1 小样本仍观察到 actual raw-hard：
loop 第二轮 `2` 次，block 每轮 `1` 次，主要为左右踝。该证据既不授权调 Reward，也不证明
4096-env 会形成 mass reset。launcher 因此增加唯一 exact
`probe = 4096 env × 5 update × save1` 的运行验收 budget；它不改变 setting、不产生科学 A/B
结论。G05 继续 `Partial`，直到同 source 的 probe 报告吞吐、episode 是否跨过 `t_hit`、
strike opportunity、hard/table/fall/nonfinite 分账及 finite checkpoint。

#### 2026-07-30 stable-upper v2 probe / recovery

stable-upper v2 两动作 `1 env × 2 update` 均自然完成并持续产出 finite checkpoint；但 loop
exact 4096 probe 的五轮 mean episode 仍只有 `21.01--24.20` steps、strike 恒零，每轮
`3741--4654` 次 `joint_actual_forbidden`，iteration 为 `26.73--42.63 s`。q_des、table 与
nonfinite 均为零，fall 仅偶发。它不能通过 `t_hit`/strike 门，因此不得发 loop long。

block exact 4096 probe 没有进入 PPO：URDF/scene 构造在 CPU 高占用、GPU 约 1% 下静默约
900 秒，由 launcher 自身 `KIT_BOOT_STALE_TIMEOUT_S=900` 自然停止；该 run 只能作为启动性能
反证，不能写作训练失败。相同 motion/Reward/solver 的 `1024 env × 100 update` recovery 运行到
update 77：iteration 通常 `10--18 s`，mean episode 始终约 `21--22`，strike 始终为零，
actual raw-hard 始终约 `47--49` events/rollout；`model_20/40/60.pt` 已写出且
`model_20.pt` 逐 tensor finite。这足以否定当前 ready 在 100 updates 内自然跨过击球窗。
run 随后命中 producer/consumer 不一致的 teacher-rate float32 边界检查而 Traceback；该 run
按预登记停止，不能把 77--100 写成训练结果。`194e9786` 已统一 producer/consumer 的 canonical
边界函数且保持 no-clipping；Pod1 focused test 对合法边界与越界篡改正/负控为 `2 passed`。

老师腰部轨迹远离 A3 hard limits，q_des projection 也为零，故当前不授权调整 Reward、CaT 或
真实 hard-edge 终止。下一门改为 nominal A3 plant 的 unified dynamic-ready/initial-qdes/
observation/reference/preparation-window hold diagnostic。Pod MuJoCo 动态 replay 又显示
block/loop teacher 约在 `1.24/1.26 s` 后失衡，说明 static LP 不等于动态 hold；旧 receipt
没有保存 LP actuator solution。必须先物化 action-specific hold qdes并稳定跨过
`preparation + t_hit + margin` 后再重跑规模 probe。G05 保持 `Partial`。

#### 2026-07-30 dynamic-ready 候选生成器（待 Pod）

reset 路径复核确认 stable-v2 从未抽取 post-swing/failure buffer：canonical ActionBall 强制
`stand_start_prob=1`、`post_swing_start_prob=0`，true reset 写动作 frame 0 后提前返回。
因此 Jiayi 描述的“35% 后混入失败姿态”不是当前击球前死亡原因；该门也尚未接入现役代码。

source 已新增 opt-in static-hold minimax LP 与动作专属 dynamic-ready candidate producer。
独立复核发现并已修正 MuJoCo actuator row 与 A3 runtime joint order 的非恒等排列；所有
runtime qdes/PD 边界先 scatter 到 LP order，求解力矩再 gather 回 runtime order。
dependency-light 回归已写但按约定未在本地运行。下一证据是 clean Pod focused pytest、
loop/block 两份 exact 候选、Isaac 的 `raw_env_reset`、artifact-ready、`1/10/final` 截图与
闭环 hold telemetry。截图器已把 raw reset 与候选写入后的 ready 分账，并补上旧 metric
fixture 的两个生产默认 mode flag；当前仍只有 source，需 clean commit 后在 Pod 运行 focused
pytest 和真实 renderer。连续若干次 renderer 在 scene creation 后零退出且无 PNG/receipt，
均明确不算证据。这些通过前不接 trainer、不发 long，G05 继续 `Partial`。

后续 clean Pod 已把 fixture 完整收口为 `test_metric_sync_fix.py: 21 passed`、
nominal-hold focused `4 passed`，未改生产逻辑。renderer/hold 仍未出 receipt；已定位到 fresh
checkout 的绝对 URDF 路径触发重复 A3 conversion，以及 headless Kit 缺 private GLU。按
[`setup_local_sync.md`](../operations/setup_local_sync.md) 物化并逐层核对了 Franco-owned
preconverted USD/GLU 副本；工具增加 `HOPE_TABLE_DIAGNOSTIC_STAGE`，下一次零退出将能精确
区分 `gym_make / reset / spawn check / nominal hold`。在真实 PNG 与 hold receipt 出现前仍不
声称 dynamic-ready 通过，G05 保持 `Partial`。

首个 stage run 已越过 `gym_make_done` 和 `initial_reset_done`，零退出发生在 nominal hold 前的
32-sensor `force_matrix_w` spawned-receipt 枚举；不是 A3 reset、hold、Reward 或 qdes 失败。
nominal-hold 现不再物化这份独立 formal table receipt，但 live `robot_hit_table`、fall、
qdes-nonfinite、actual-hard term 原样保留并逐步判定。

clean `4c870e94` Pod numeric hold 已让 loop/block 两件各保持 `0.8 s / 40` policy steps；
两件均 `PASS`、双脚接触率 `1.0`、零 terminal，minimum root z 均为 `1.0684000 m`，
maximum tilt 分别 `0.00983/0.01029 rad`。post-write、step 1/10 与 final PNG 均显示直立且未
继续倾倒；首个 raw-reset PNG 是 headless RTX 首次 render 的全黑填充帧，不可判姿态。producer
现不推进 physics、只丢弃同一 reset 状态的首次 render 后再保存，真实 raw-reset 仍待 Pod 复截。

clean `22890ea2` Pod 复截已让 loop/block 的 `raw_env_reset` 产生可见 PNG；两件原生 reset
均为直立、双脚着地的动作专属 frame 0，不是歪倒或 failure-buffer 状态。两件 screenshot hold
再次 `PASS`，receipt SHA-256 为 `e0abbfe6…` / `53e8950c…`，远端 no-clobber 根目录为
`/workspace/franco/n1dr_nominal_22890ea2_{loop,block}_frames_r1/`。这通过了出生静态 hold，
尚未证明完整 teacher 动态、policy strike 或 long；G05 继续 `Partial`。

CC 非阻断复核另发现 teacher-rate consumer 的 contact-geometry 属性读取若失败，旧异常处理会
因局部变量尚未绑定而掩盖根因。绑定已移到 `try` 前，并有回归要求缺失 runtime geometry 时保留
原始 `AttributeError`；有效 receipt 的计算与判定路径不变，故不需要学习 A/B，仍需 Pod focused
回归。

teacher-rate focused Pod 回归现为 `3 passed`，其中包括合法 float32 边界、篡改拒绝和缺失
geometry 时保留原始异常。reset 收据粒度方向也已裁定：完整 JSON/replay/hash-chain 移向
checkpoint/hourly，热路径保留可 exact-resume 的紧凑事件日志与全部安全/solver/curriculum
真值；`seed+config` 单独不足以重建历史。实现前先用 segmented profiler 把当前
`~7.1 ms/env-reset` 上界拆开，再做 fixed-tape、旧收据重建和 exact-resume parity，不做学习 A/B。

#### 2026-07-30 dynamic-ready trainer 接线与 diagnostic 170-update 生命周期修复

source 已实现[动作专属动态准备合同](../DEFINITIONS.md#action-specific-dynamic-ready)：
候选与 nominal-hold PASS receipt 在 gym 前双 pin，Motion 再按实际载入 bytes 验证 action
顺序和 frame 0；true reset 把 simulator physical state 与 action-manager/raw/processed/
pre-clamp/previous/nominal qdes 状态作为同一可回滚事务，fresh actor schema-2 bias 解码到同一
`hold_qdes`。旧 shared-ready schema-1 仍可读，二者互斥，dynamic-ready 当前只授权 exact N=1。
这是 E1 source 状态；Pod focused test、真实构造、`1 env×2` 与 `4096 env×5` 尚待执行，
因此 G05 继续 `Partial`。

Pod1 两条 exact `4ff48b21` long 并未继续学习：loop/block 均在 update 169 后由
`joint-safety policy-step summary overflow` 退出，日志中此前没有 joint-safety consume record。
4096 summary capacity 与 24 steps/update 精确解释该边界。successor 让 diagnostic 仍无 formal
Reward/promotion authority，但每 update 沿原有安全事务排空摘要；actual-hard/nonfinite 继续在
optimizer 前 fail-closed。已溢出 run 不允许清 sticky latch 续跑。fresh probe 过门后先发
1000-update canary；五轮只判是否跨 `t_hit`，不判学习上限。

clean Pod source 验证现为 `63 passed`。exact commit profile 复算与 tracked qvel profile
bytes 相同；upper loop/block bundle v2 materialize 均 PASS，SHA 分别为 `22672c3d…` /
`69b3b78d…`。初轮 full-block fixture `443` 对父提交同一 asset 也实算 `447`，已作为旧 fixture
纠正，生产 solver 未改。尚缺真实 A3 scene 的 schema-2 policy recipe、两动作 `1 env×2`、
`4096 env×5` 和 finite checkpoint，G05 继续 `Partial`。

#### 2026-07-30 194-D table-pose-twist 原候选与剩余边界

该轮原始 fresh N1 候选曾升级为
[`action_ball_table_pose_twist_n1`](../DEFINITIONS.md#action-ball-table-pose-twist-contract)：
完整 `hitter_footwork(177)` 前缀后依次追加 table-relative position `3`、连续完整 SO(3)
orientation `6`、yaw-heading root-COM linear velocity `3`、signed face/rho `4` 与冻结 action
identity `1`，总宽 **194**。三个角度没有被简化成 yaw；task 的 base/racket 位置仍是机器人
相对 residual，绝对 9 值只补“机器人相对桌体在哪里、朝向如何”的几何上下文。部署权威按物理量
拆分为 OptiTrack position/orientation、pelvis IMU 三轴 gyro，以及以 OptiTrack position 为
无漂移锚的因果三轴线速度估计器。旧 182/191-D checkpoint 不可复用；dynamic-ready policy
recipe 与 observation hard contract 分层，现有两份 recipe 无需重物化。后续 probe 发现该
同宽合同混用了 heading position 与 world velocity/normal，已由下文版本化的 frame-consistent
合同取代；本段只保留当时证据。

当前 source 同时修正 table-hit 后 reset 的 PhysX stale force report：只对上一 episode
**最终 physics substep** 确有 table hit 的 env，隔离新 episode 第一份不可区分的旧 report；
非 table reset 不隔离，persistent 新碰撞在下一 substep 仍会触发，decimation 小于 2 时拒绝。
该机制依赖已经通过 hold 的 table-clear dynamic-ready 出生姿态。Pod1 clean `eb2799b1`
table smoke 已让五个 collider role 分别得到真实 PhysX 正控，32 个 body×5 列 matrix 全构造，
四个 physics substep 均覆盖；五个 probe 的 post-reset raw reason/ledger/force 均零泄漏。
log SHA-256 为 `15c52d29…26aac`，unsupported/Traceback/FAIL 均为 0，且出现
`main_completed`；机器可读 receipt 见
[`table_smoke_eb2799b1_gpu1_r26.receipt.json`](../../configs/n1_contact_dynamic_ready_20260730/table_smoke_eb2799b1_gpu1_r26.receipt.json)。
dependency-light 194-D 回归为 `257 passed, 9 skipped`。真实 `1 env×2 → 4096 env×5` 尚未完成，
因此 G05 继续 `Partial`，也尚未授权长训或真机。

首个 N1 不等待 formal per-reset receipt 的 checkpoint 粒度重构，也不临时添加更强 action
penalty、EMA 或 command governor；这些分别在 formal N5 前和健康 `model_1000.pt` 后按
[分阶段准备账本](../experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)
闭合。当前两份 actor bias 超出 `[-1,1]`，直接 raw clip 会使动作专属 ready 不可达。首发前
唯一剩余运行门已收窄为：194-D 真实 actor 构造、finite checkpoint、episode 跨 `t_hit`，
以及 table/fall/raw-hard/nonfinite 不持续爆炸。

#### 2026-07-30 frame-consistent 194-D 与 shared-waist DR 收口

`c9682591` 的 loop/block 旧混合-frame 194-D `4096 env × 5 update` probe 均有真实 PPO
update 和 finite `model_0..4`。mean episode 从 `24` 到约 `29–32` steps，birth age `<=1`
actual-hard 为零；但第一次 PPO 更新前仍分别有 `860/864` 个 env 触发
`waist_roll_joint` raw mechanical hard。两动作 hard-env Jaccard=`0.982`、substep
Jaccard=`0.992`，qdes forbidden 始终为零；teacher 的 waist raw-hard 最小余量为
`0.272–0.303 rad`。因此不能把旧 probe 续成长跑，也不能靠改 Reward、删除 actual-hard 或
放宽机械限位掩盖问题；证据指向共享 torso-CoM/link-mass/PD DR 超出当前 ready 稳定域。

fresh successor 改用
[`action_ball_table_pose_twist_heading_task_n1`](../DEFINITIONS.md#action-ball-table-pose-twist-heading-task-contract)：
racket position residual、demanded velocity 与 raw-A face normal 全部统一到 yaw-heading frame；
table-relative base position `3` + continuous orientation `6` 仍保留完整 6-DoF，宽度保持
`194`。旧同宽 checkpoint 因列语义不同不能 resume。N1 launcher 同时选择
[`stable-ready plant`](../DEFINITIONS.md#stable-ready-plant)，只暂关 torso CoM、link mass
与 PD-gain DR，保留 material DR 和 recipe-bound joint-default offset。下一证据是 clean Pod
focused parity → 两动作 `1×2` → `4096×5`；任一动作跨 `t_hit` 且 raw-hard 不再爆炸即发
`4096×1001`。当前仍未授权真机，G05 保持 `Partial`。

#### 2026-07-30 frame-consistent stable-ready 双动作实跑与首条 1000-update

exact `f2c54fc3` 已在 Pod1 通过 dependency-light focused suite
`338 passed, 9 skipped`。loop/block 两条 `1 env × 2 update` smoke 均真实验证
`action_ball_table_pose_twist_heading_task_n1 (194D)`，四份 checkpoint 逐 tensor finite，
qdes/actual-hard/table/fall/nonfinite 全零；log SHA-256 分别为
`db75ef49…49b` / `0a26cbee…69f`。stable-ready plant 日志明确只关闭 torso CoM、
link-mass 与 PD-gain DR，保留 material 与 recipe-bound joint-default offset。

随后两动作各自然完成 `4096 env × 5 update`。十份 checkpoint 全部 finite，update 0
全安全；mean episode 随后约 `48–72` steps，已跨 loop `t_hit≈31` steps 与 block
`t_hit≈24` steps。loop 在 update2/3 分别产生 `783/84` 个 strike opportunity；block 在
update2/4 分别产生 `1838/430` 个，说明 task 击球位置/时间窗已对上。两条从第一次 PPO 更新后
都出现共享 waist-roll/pitch actual-hard，qdes forbidden 始终为零；probe log SHA-256 为
`5c5ce9d4…420` / `a93b39c4…e7a`。这证明出生修复有效、动态学习仍需观察，不能用五轮判定恢复
上限或调 Reward。

历史 exact `4ff48b21` 给出直接反例：它在 update1–5 同样有大规模 reset，但 loop/block 的
actual-hard terminal 到 update100 已降到 `14/11`，到 update169 均为 `3`。因此按预注册不被
五轮早期失稳拦截。Pod1 GPU1 已接受 fresh 反手挡
`n1hr_milestone1000_f2c54fc3_block_gpu1_r1`（`4096×1001`，claim
`8dc4dcb2…ff080`）并进入真实 `Learning iteration`。2026-07-30 快照已到 update 250；
`model_100.pt` / `model_200.pt` 均有 80 个 tensor、全部 finite。update231–250 的
iteration time 为 `21.19–27.99 s`，平均 `23.48 s`、中位 `22.12 s`。mean episode 已到
`124.11` steps，显著跨过 `t_hit`；该 update 有 `882` 个 strike opportunity、
`843/1585` swing completion（`53.19%`），qdes forbidden=`0`。

actual-hard 从 update81 的 `1420` 降到 update200/250 的 `265/116`；腰部已基本退出 hard
来源，update250 主要是 left ankle pitch/roll 与少量 right ankle roll，而且全部发生在
episode age>1，不是出生瞬间。与此同时学习质量出现相反趋势：update100→200→250 的
exact strike hit=`1.73%→0→0.23%`，strike-window hit=`26.46%→0.20%→5.01%`；
update250 的 exact strike position/velocity/normal error 为
`0.2646 m / 1.4014 m/s / 81.66°`，racket speed 仅 `0.2627 m/s`，目标为
`1.3807 m/s`。table/fall=`481/227`，base height/upright 已降到 `0.9668/0.8935`，
virtual capture/return 仍为 `0/0`。因此目前不是“仍被初始 hard-reset 饿死”，而是
“episode 已足够长、hard 在恢复，但 policy 通过倾倒/撞桌走向劣质局部解”。它尚未触发
NaN、identity、receipt、Traceback 或 counter invariant 的立即停止门，故不在 update250
热改配方，继续到预注册 update300 做同口径裁决。

update250 income ledger 中 motion imitation 五项均非零；racket position 只有 `0.0001`，
velocity/normal/strike-success 为 `0.0000`，death=`-7.2`、qdes barrier=`-0.1790`、
qdes projection=`0.0000`。`table_hit_penalty` 为 `0.0000` 是 pinned task config
明确设置 `table_hit_penalty_weight=0.0` 的结果；桌碰仍由 `robot_hit_table` 终止并进入
generic death penalty，不应把零列误称为“额外桌碰罚已生效”。反手拉在
该槽自然完成后排队。该谱系仍是 diagnostic、绑定旧 physics profile，只授权 contact/学习可行性，不授权 formal
landing、export 或真机。G05 继续 `Partial`。

其后 feature-branch head `9fdb909a` 仅增加 fresh/formal observation guard、阶段账本与
OptiTrack ballfit 科学源；Pod fresh worktree 对 observation、N1/generic launcher 与
training-contract 的 dependency-light focused suite 为 `314 passed, 9 skipped`。该验证不改变
正在运行的 exact `f2c54fc3` bytes，也不把其旧 physics bundle 重标为 OptiTrack profile。

2026-07-30 后续只读核对发现 Pod1 GPU0/GPU2 的旧 `4ff48b21` 进程均已在 update169 抛出
`joint-safety policy-step summary overflow`，却仍持有约 `7.7 GiB` 显存且不再产出 update。
在 Franco 明确授权替换后，按 exact owner/PID/PGID、cwd、claim 和有限 `model_100.pt`
复核，只对 PGID `844989/848759` 发 TERM；两者正常退出，未发 KILL、未删除日志/checkpoint/
namespace、未触碰 GPU1。

随后 GPU0 fresh 接受反手拉 upper seed0
`n1hr_milestone1000_f2c54fc3_loop_gpu0_r1`（claim `3c523fde…0196`），GPU2 fresh 接受反手挡
upper seed1 `n1hr_milestone1000_f2c54fc3_block_gpu2_seed1_r1`（claim
`7ac32418…e3f`）。二者均为 exact `f2c54fc3`、`4096×1001`、194-D、schema-v2
dynamic-ready、stable-ready plant，并已越过首个真实 `Learning iteration`；发射 specs 在
`configs/n1_contact_heading_stable_ready_20260730/`。GPU1 seed0 同时已越过 update300，
`model_300.pt` 80 个 tensor 全 finite。三卡现在分别回答动作差异与 block seed 复现；仍是旧
physics diagnostic，不授权 formal/landing/部署。

full-body 没有直接占用 GPU2：仓内 full bundle 仍是 schema-v1，而该 launcher 明确要求 exact
motion/nominal-hold 双 pin 的 schema-v2 dynamic-ready。可比 full-body 仍须完成
stable-full `ready→core→ready`、动作专属 dynamic-ready/nominal hold、full solver preflight、
schema-v2 bundle 与 fresh fixed-width successor `1×2→4096×5`；运行旧 bundle 只会同时改变 ready、observation 与
teacher bytes，不能作为科学对照。G05 保持 `Partial`。

同日先实现了历史 195-D teacher-start one-hot 合同，随后按动作泛化审计改为 fresh N1 固定
194-D `action_ball_table_pose_twist_heading_task_teacher_start_v2`：用
`time_to_teacher_start_s` 替换恒为 `[1]` 的 N1 one-hot。Racket getter 在
ObservationManager shape probe 时先调用既有 ActionBall lazy bind，reset 后仍直接读取 Motion
phase governor 的 receipt 真值。动作 UID/slot 保持在 sampler/solver/curriculum/receipt，
不进入 policy；formal N5/N73 须另加固定宽 content-derived future-motion intent。旧
`f2c54fc3` 三条同宽 194-D run 不停机、不重标、不允许以新合同 exact resume。新 source 在 Pod
N1 v2 `1 env×2 update` 构造 smoke 前保持
`Partial`，且 action-set source SHA、training contract 与 launch claim 均须 fresh repin。
exact `020dc8d9` 已在 Pod1 跑过 teacher-start/observation/action-set/launcher/schema focused
suite：`390 passed, 9 skipped in 71.26 s`。这只证明历史 195-D source 接线，不冒充 v2
ObservationManager 的真实 Isaac 构造或 finite PPO checkpoint。

2026-07-30 19:39 CST 只读快照：loop seed0 / block seed0 / block seed1 分别到 update
`219 / 574 / 186`，mean episode=`104.88 / 481.52 / 105.90`，均继续产生真实 PPO update，
无新 Traceback。三条当前窗的 strike opportunity=`945 / 965 / 962`，但 virtual
capture/return 仍全为零。block seed0 的 table/fall/actual-hard 已降到 `2/3/6`，但 965 个
proposal 全被 face gate 拒绝；loop seed0 的 post-strike fall 为 `887/946`；block seed1 的
table/actual-hard 为 `590/325`。因此窗口与 denominator 已通，动作质量、signed-face/contact
对齐及 seed 稳定性仍未通过；这些不是 teacher-start source merge 的阻断理由，也不能作为 N5、
landing 或部署证据。下一条 fresh fixed-194 v2 source 仍须独立 Pod `1 env×2` 构造验证。

2026-07-30 19:50 CST，GPU1 block seed0 已自然产出 `model_600.pt`：7,197,343 bytes，
SHA-256=`11bee4911f54d9d43e0a112f843009f5811a746475b82d5fe050ca5fffb8470f`，80 个 tensor
全部 finite；exact PID/PGID、cwd、GPU UUID、source、claim 与 namespace 均匹配。update608
iteration=`20.94 s`、mean episode=`440.77`、strike opportunity=`951`，
table/fall/actual-hard/qdes-forbidden=`4/5/14/0`，无 Traceback/RuntimeError/NaN/identity/
receipt/counter drift。但 capture/return 仍为零，`937/951` 被 face gate 拒绝，exact strike
position/velocity/normal error=`0.2426 m / 1.3928 m/s / 86.31°`，实际拍速仅
`0.2832 m/s`（目标 `1.2793 m/s`）。因此出生/episode/safety 恢复已形成新证据，学习质量仍未
通过；按预注册继续到 update1000，不热改超参。

2026-07-30 后续对 GPU2 比较臂先做了 fixed-action 公式带，而不是先改 Reward。反手挡动作身份、
motion、solver、physics、拍速方向、原落点中心 `2.555 m` 与初始速度宽度都保持不变；只把
中心来球从 `4.2376948` 提到 `4.6614643 m/s`（1.1 倍），再由现有 solver 对每球重算拍速、
拍面和击球位置。4096 个确定性 proposal 中 `2763` 个 admitted（`67.46%`），teacher-rate
均值/中位为 `0.72055/0.71595`；`1327` 个 residual 超容差、`6` 个低于 teacher-rate 下界，
均保留在 proposal 分母。该证据证明快球在同一落点约束下会让挡球卸力并形成更慢的 teacher
task，但 solver rejection 使实际题分布条件化，且未达到 formal `95%` admission 门，只允许 fresh no-clobber
diagnostic comparison，不改变 G05=`Partial`，也不冒充 curriculum 晋级。

同日第一次 fast-ball plan 在创建 namespace 前按预期 fail-closed：derivative 引用的 r9 profile
把 `hope_commands.py` 钉为 `0e650b…`，最新 source 实际为 `e24190…`。没有绕过该门；Pod 以
exact source blob map 重物化 profile pins `9ccb9854…5788`、base bundle
`0daa5bce…ace53` 和 fast-ball bundle `f2be2331…1a491`。physics SHA 和全部科学参数未变，
solver profile 更新为 `bf255a78…f26e`。新 canonical spec 必须绑定包含这组工件的 exact
source commit；旧 `8bd480…` bundle 不再用于发射。

current-source canonical plan 随后通过（claim `13dc15a2…8e86f`），但首个 smoke 在真实 PPO
前 fail-loud：`1d4b8a11` 新增的 stable-ready N=1 guard 误读
`racket_cfg.clip_names`，训练翻译层实际安装的字段是 `clip_names_per_clip`，所以合法
`("bh_block",)` 被看成空 tuple。该 namespace 已 spent 且不复用。直接修正 guard 的字段名并把
mode/diagnostic/action tuple 写入拒绝信息；这不改变 Reward、ball→task、plant 或 policy，
修复提交后仍需 fresh 1-env×2 Pod smoke。

第二次 fresh smoke（exact `319ae8ff`，claim `691dc1ac…344`，namespace
`n1hr_smoke_fastball110_319ae8ff_block_gpu2_seed0_r2`）又在 PPO 前暴露独立构造次序缺口：
ObservationManager 为探测 term shape 读取 `time_to_teacher_start_s` 时，CommandManager 已完成，
但 ActionBall cross-command timing 仍按设计等待 lazy bind，getter 直接读 Motion 因而报
`action-ball task timing is not bound`。失败 namespace 保留不复用。直接修为 getter 先调用
既有 `_ensure_action_ball_runtime_initialized()`；不把未绑定 timing 静默伪造成运行时零。
同时按 Franco 决定删除 fresh actor 的 `action_one_hot`，切固定 194-D v2；因此旧 r2 claim/spec
全部作废，须以新 source、profile pin 和 fresh namespace 重发。

fixed-194 v2 的新工件已在 Pod1 exact `17c7258a` 上按 no-clobber 物化：profile pins
`08c8f9c7…c6b4`、base bundle `ed9fa0f7…afef`、1.1 倍 fast-ball bundle
`3c1076e3…c32b`，solver profile `52777b36…9754`；旧诊断 physics profile
`aa5c9085…f85b7` 未变。4096-proposal tape 仍为 `2763/4096=67.46%` admitted，故只授权
diagnostic comparison。下一门是先把工件提交为 source commit A，再让三份 fresh r3 spec
精确指向 A；未完成前不得复用旧 r1/r2 namespace 或 claim。

工件现已进入 exact source commit `8729104e6c9a…46c4`。fresh r3 smoke/probe/milestone1000
spec 都指向该 commit、bundle `3c1076e3…c32b`、Pod1 GPU2 UUID 与 seed0；三份 canonical
raw JSON SHA 依次为 `e1b63f00…5b8d`、`3b200542…dd34`、`b0396fbe…d442`。第一版缩进 JSON
已在 namespace 创建前被 canonical-byte 门拒绝并保留原 operator-control 目录；它们尚未 launch：
下一门是把 tracked spec 复制到独立 operator-control 目录，用 exact source A 的 launcher
生成 canonical claim，并在真实 GPU/lock/namespace no-clobber admission 后才可发 smoke。

规范化 smoke spec 已在新 operator-control 目录由 exact `8729104e` launcher canonical
plan PASS，launch claim=`7f9d12cac0de0e0bbe645c34cd556ef585e1469c9794634bd48d7b7a57084002`。
plan 期间 Pod1 GPU2 UUID 正确、显存仅 3 MiB、无 compute process，目标 namespace 不存在。
下一门收窄为用该 claim 发 `1 env × 2 updates`，取得真实 PPO update 与 finite checkpoint。

r3 随后已真实构造 scene，并通过 fixed-194 v2 ObservationManager 与 dynamic-ready bootstrap：
log 明确打印 `action_ball_table_pose_twist_heading_task_teacher_start_v2 (194D)`。但在 PPO
runner 创建前，policy-recipe 硬门发现 spec 的旧 SHA `b7209710…077f` 与实际 post-compose
`165645f5…bd9` 不同而 fail-closed；无 PPO update/checkpoint。namespace
`n1hr_smoke_fastball110_8729104e_block_gpu2_seed0_r3` spent 且不复用。下一门是物化实际
schema-2 recipe，生成 fresh r4 spec/claim，并等待旧 PID 自然退出后再占自然空闲槽。

recipe-only 真实构造现已物化 policy contract `165645f5…bd9`（artifact raw SHA
`4b81c74b…7fb1`）。差异不是 PPO 或 fixed-194 观测字段，而是 schema-2 policy initialization
把 dynamic-ready candidate/hold 的绝对 checkout path 及派生 binding SHA 纳入 identity；
因此换 checkout 必须重物化 recipe。fresh r4 smoke/probe/milestone1000 spec raw SHA 依次为
`6fc4e7ca…c369`、`6e7caeb1…b200`、`533b50d2…36ea`。r3 pre-run exception 暴露另一个非首发
阻断：`gym.make()` 后 hard-contract 异常未先 `env.close()`，Kit teardown 会自旋到 launcher
watchdog；当前不手工 signal/kill，待 exact-PG 收口后才发 r4，异常 finally-close 修复列入
formal N5 前 TODO。r4 smoke canonical plan 已 PASS，claim=`257c6cccdcbef47b3154536ab19b8f8434711e9f614c3bece4a146026fded80c`；
它只等 r3 watchdog 收口和全 Pod Kit boot lock 自然释放，不再有 recipe/spec/claim 缺口。

Superseding evidence（2026-07-30）：r3 wrapper 已由 launcher watchdog 按 exact PG 自然收口，
exit=`125`；GPU2 回到 `3 MiB`、无 compute process，GPU lock 可非阻塞获取。fresh r4 随后在
namespace `n1hr_smoke_fastball110_8729104e_block_gpu2_seed0_r4` 自然完成两个真实 PPO update，
iteration 0/1 分别约 `2.75/2.80 s`。日志回读的 actor contract 是
`action_ball_table_pose_twist_heading_task_teacher_start_v2 (194D)`，fresh policy bootstrap
生效；`model_0.pt` 与 `model_1.pt` 各含 80 个 tensor，其中 76 个浮点/复数 tensor 逐项全
finite。两轮 table/fall/qdes-hard/actual-hard/nonfinite/terminal reset 均为 0，首轮 ready
双脚接触率为 1.0；48 个 environment steps 尚未到击球窗，strike=0 符合 smoke 预算而不能判学习。
下一门仅为同一 exact setting 的 `4096 env × 5 updates` r4 probe。

Probe evidence（2026-07-30）：claim=`fca61705…b813` 的 fresh r4 probe 已自然完成
`4096 env × 5 updates`，五份 checkpoint 各含 80 个 tensor、其中 76 个浮点/复数 tensor，
逐项全 finite；`model_4.pt` SHA=`0f925821…f2e7`。iteration wall 依次为
`10.09/10.48/26.63/17.16/24.85 s`，mean episode 在 update 1--4 为
`48.00/71.66/52.99/59.91`；update 2/4 分别出现 `1985/643` 个 strike opportunity，故完整
preparation+击球窗已可达。qdes-hard/fall 始终为 0，table 在 update 3/4 为 `16/25`，
actual-hard 为 `0/267/3103/861/2076`。历史同谱系在 update 1--5 也有早期腰 hard、到
update100/169 显著下降，故该五轮只作趋势基线，不单独否决学习；下一门是同身份 fresh
`4096 × 1001` milestone1000，并按 100/300/1000 判读。

Milestone launch evidence（2026-07-30）：fresh r4 milestone1000 已用 canonical
claim=`2710fd6f…d4f4` 在 Pod1 GPU2 发射，exact PID/PGID=`1134253`，namespace 为
`n1hr_milestone1000_fastball110_8729104e_block_gpu2_seed0_r4`。它已进入真实 PPO；
首两轮 wall 约 `9.42/12.05 s`。首份 `model_0.pt` SHA=`1296e929…6bcf`，80 个 tensor 中
76 个浮点/复数 tensor 逐项全 finite。该 run 是 fresh diagnostic，不从 smoke/probe resume，
下一次 Gate 判读在 update100；此前腰 actual-hard 波动只记录，不临时改超参。

Update-wall forensics / candidate（2026-07-30，focused tests 已过，真实构造待验）：

- r4 每轮固定 `4096×24=98,304` env-step；update 8 后 collection 均值 `23.79 s`
  （约 `4.13k steps/s`），learning 只有 `0.299 s`。全窗口 collection/reset 相关系数约
  `0.84`；粗拟合固定逐步/同步税 `12–14 s`，另有 `5.6–6.7 ms/env-reset`，稳态约
  1670 reset/update 即 `9–11 s`。同 reset 数下仍见 `21.7–29.9 s` 抖动，NVML 15 秒采样
  SM 均值约 `10.8%`，故瓶颈不是 PPO 或 GPU 算力。
- 源码审计确认只完成了 SHA cache、strike timing 去重、部分 metrics D2H 合批和 diagnostic
  terminal transcript bypass；逐 reset Python/`.item()`、逐 step dense safety clone/identity、
  每 update formal-style receipt 与残余同步仍开放。
- fresh candidate 直接消除两类确定性开销：diagnostic 保留 clamp/brake/q-qdot freshness/
  actual-hard Done 和逐关节 aggregate，但不再保留 dense substep/per-step identity 或每 update
  formal 文件；reset 的 env/slot/UID/reset/swing/previous-swing/active 只做一次 host-row D2H，
  后续 broker/pool/receipt 检查复用同一行。formal 路径不改。
- exact `c0747d59` 在 Pod 的 joint-safety/action-ball focused suite 为 `134 passed in
  7.52 s`。首次 recipe-only 真实构造正确拒绝旧 solver pin
  `52777b36…9754` 与 runtime `a7b120f7…09ec` 的不一致；因此没有复用旧 bundle，而是从
  exact source 重钉 profile raw SHA `3c978844…762f`、base bundle
  `d9de51e8…b03d` 与 1.1 倍 fast-ball bundle `398287f7…bdf`。
- artifact commit `886f42a7` 的 recipe-only 真实构造物化 policy SHA
  `569431a5…d0c0`。fresh smoke claim=`a16dc172…fd72d`，两轮 wall=`3.83/1.81 s`；
  两份 checkpoint 均为 80 tensor 且全 finite。fresh probe claim=`5ffabcbe…ab17e`，
  五轮 wall=`9.25/10.95/25.44/16.17/23.91 s`；五份 checkpoint 均为 80 tensor 且全
  finite。旧/新同 seed 的 actual-hard reset=`0/267/3103/861/2076`、table
  reset=`0/0/0/16/25`、strike opportunity=`0/0/1985/0/643` 以及完整 behavior counters
  逐轮一致，固定 solver 诊断的 counts/distribution 也完全一致。
- 第一批 candidate 的 wall 均值只从 `17.84` 降到 `17.14 s`（约 `3.9%`），没有达到
  `≥15k environment-steps/s` 或等 reset 负载 collection `≤6.5 s`。Gate 仍为 `Partial`；
  下一批只做剩余 VirtualBall/command host barriers 与 reset broker/receipt 批量化，继续
  Pod parity/吞吐验收，不动 PPO，也不热补当前 milestone1000。

Second update-wall candidate（2026-07-30，正确性 PASS、性能 FAIL）：

- exact `26c648d4` 的 diagnostic Motion timing handoff 复用 Racket 已有 host identity rows，
  消除了逐 reset env 的 timing scalar device read；fixed-18-draw refill overlap 从 staged
  `O(K²)` 区间扫描改为 highwater + start-set `O(K)`。formal 路径未改。
- Pod focused suite 为 `187 passed in 13.70 s`。首次 smoke 揭示 true reset 的 previous
  swing generation 合法 sentinel 是 `-1`；修复后 fresh `1 env×2` wall=`2.67/2.45 s`，
  两份 checkpoint 均为 80 tensor、其中 76 个浮点/复数 tensor，逐项全 finite。
- fresh same-seed `4096×5` wall=`9.00/10.11/25.63/16.81/23.89 s`。五轮
  `HOPE_JOINT_SAFETY_UPDATE_JSON`、`HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON` 与
  `HOPE_EXACT_BEHAVIOR_UPDATE_JSON` 均与第一批 candidate 逐字相同；actual-hard
  `0/267/3103/861/2076`、table `0/0/0/16/25`、strike opportunity
  `0/0/1985/0/643` 未漂移，五份 checkpoint 全 finite。
- wall 均值 `17.09 s` 只比第一批 `17.14 s` 快 `0.06 s`（约 `0.3%`），相对旧版累计只改善
  约 `4.2%`。这反证 Motion scalar reset 读取与 refill `O(K²)` 是剩余主瓶颈；Gate 继续
  `Partial`。下一唯一 candidate 是把 diagnostic 普通 step/strike step 的 invariant、
  finite、partition 与 EMA scalar 收敛为少量 validation packet；若仍慢，再批量化
  table-contact 和 joint-safety 小 kernel。PPO 与正在运行的 milestone1000 均不改。

Explicit table-contact v3 / backend 定价（2026-07-31）：

- 旧 32-source×5-filter 方向会触发 pinned PhysX filter 警告；单个 Robot wildcard aggregate
  也不能提供 32 个有序 filter slot，均已拒绝。当前实现为 5 个 table source × 32 个显式
  A3 body filter，矩阵 `[E,1,32,3]`；
- Pod focused suite `214 passed`；source `f068555b` 的 1-env 真实 PhysX smoke 覆盖
  top/keepout/net/左右 post、wrist/elbow/ankle 与四个 physics substep，五 probe 自动 reset 后
  零泄漏，无 unsupported/did-not-match/FAIL/Traceback，出现 `main_completed` 且 shell rc=0。
  独立故意失败进程返回 rc=1，Kit teardown 不再能伪装 PASS；
- source `2bddb440` 两次 4096-env 短稳态 A/B 的 table on/off 为
  `67.577/60.233` 与 `72.000/61.400 ms/policy-step`，平均差 `8.972 ms/step`，折算
  24-step rollout 约 `0.215 s/update`。因此 exact table 后端不是现役 `17–25 s/update`
  的主因，不切 analytic box；box/prism 只保留为后端失效的降级设计，且必须覆盖真实 collision
  geometry/racket offset，禁止只查 body origin。

这批证据闭合 table sensor 正负控与固定税归因，但没有闭合完整 trainer 的
`≥15k environment-steps/s` 健康线；G05 继续 `Partial`。下一性能 candidate 仍是合并
diagnostic 普通/strike step 的 host validation packet 与 reset broker 的逐 env Python，
随后才做同 seed `4096×5` trainer parity/吞吐。

P0a validation packet / strict 1.1× evidence（2026-07-31）：

- diagnostic 路径复用 Racket 已完成的 host identity/timing selection，普通零 pending step
  不再做随后会被覆盖的全局 reduction；formal opaque resolver、Reward、Done、solver 与 RNG
  语义未改。Pod focused motion/birth/runtime suite 为 `75 passed`，event timing suite 为
  `6 passed, 1 deselected`；
- strict 1.1 倍同一 bundle/seed 的旧 exact `4096×5` wall 为
  `9.00/10.11/25.63/16.81/23.89 s`，新 exact source `6557390f` 为
  `2.90/3.65/18.38/9.70/16.40 s`，均值 `17.088→10.206 s/update`，改善 `40.3%`。
  五轮 `HOPE_JOINT_SAFETY_UPDATE_JSON`、`HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON`
  与 `HOPE_EXACT_BEHAVIOR_UPDATE_JSON` 逐轮相等，五份新 checkpoint 均 finite；
- source `c38b25d0` 的早先 probe 虽在 run name 中写了 fastball，实际绑定的是 1.0 倍 bundle，
  永久只作 spent 构造证据，不进入 strict 1.1 倍性能比较；
- exact source `bd340479` 的反手拉依次通过 recipe、`1 env×2` 与 `4096×5`。probe wall 为
  `2.90/2.87/12.51/5.97/9.83 s`，均值 `6.816 s/update`，五份 checkpoint finite；
  strike opportunity=`0/0/921/0/1`，table=`0/0/15/77/22`，
  fall=`0/0/0/0/6`，actual-hard=`0/264/3111/803/2059`。随后 Pod1 GPU1 的 fresh
  milestone1000 已进入 PPO；GPU0 同时运行 `6557390f` strict 1.1 倍反手挡 seed1，GPU2
  继续旧 `8729104e` 反手挡 seed0。三卡都在产不同动作、来球或 seed 证据，不跨 source 合并报数。

P0a 证明 host validation 合批是实质杠杆，但没有证明所有动作/reset 负载都稳定满足
`≤6.5 s collection` 或 `≥15k environment-steps/s`。diagnostic-only lean Motion timing
validator 在 exact source `2c3a39fe` 保留 generation、canonical teacher-rate/拍速向量、
scaled-time、episode horizon、pending-wait、篡改负例与零 partial-write；Pod handoff focused
为 `22 passed in 2.76 s`。相关三套回归为 `66 passed, 1 failed`，唯一旧 event-contract
fixture 在父 source `bd340479` 同样失败，不是本 candidate 回归。它仍须自然空闲 GPU 的
`1 env×2`、same-seed `4096×5` 深层 parity/吞吐后才能进入 replacement。之后的主项仍是
reset broker/receipt 的逐 env Python。精确 table 后端固定税只有约 `0.22 s/update`，继续保留；
box/prism 仅在 pinned backend 失效或完整 trainer 证明非线性放大时作为 table-frame 保守降级，
不能退化成 world-frame 无限半空间或只查 body origin。formal checkpoint 粒度 event journal
仍是 N5 前独立工作，不由 diagnostic fast path 代替；G05 保持 `Partial`。

Third update-wall candidate（2026-07-31，数值 parity PASS，trainer 性能 FAIL）：

- diagnostic birth callback 已由每个 reset env 两次 Python claim/provide 改为每批一次；
  joint-safety 不再为 diagnostic 做 formal-only dense summary、二次 gather 与 CUDA
  `unique().item()`；broker/pool 又把 ledger 构造从约 `3R` 降至不超过 `3A`，空 transcript
  SHA 与 formal-only assignment 在 diagnostic 归零。formal receipt/state schema 均未改；
- metric/validation packet 把已知 ordinary step host barriers 从 `5` 降至 `1`，strike step
  约从 `15` 降至 `2`；formal/default 仍同步 fail-fast，exact contact/capture/landing、
  sparse ledger、EMA 与 hold/recovery recurrence 保持原顺序。对应 commits 为
  `d2ec91e9 / 5f85cc58 / dbb7ce04 / 60a8e219`；
- Pod 分组证据：metric focused `126 passed`、joint ledger `88 passed`，runtime/wiring/highwater
  focused 均通过。组合测试里两个 module-reload/pickle fixture 在父提交同样失败，故不冒充
  candidate 回归；CUDA `_assert_async` 好/坏谓词已分别验证正常通过与同步时非零退出；
- `hope_commands.py` 变化已触发按 exact source 重钉，而非沿用旧工件。profile raw SHA 为
  `2c1c91c…9b2c`，base bundle=`d28a5b12…4246`，strict 1.1× fast-ball bundle=
  `81dee53f…0351`；固定 proposal tape 仍为 `2763/4096` admitted，故仍只授权 diagnostic。
  工件 config commit 为 `056625be`。

真实 wall 已补齐。第三批 `a91b4686`、外围 rollback 裁剪 `096afb7b` 的 same-seed
collection 均值为 `10.3804/10.6618 s`；内层 simulator/dynamic-ready rollback 裁剪
`4d631fb3` 的两个 replicate 为 `10.7364/10.2878 s`。三组 update JSON 均与
`6557390f` 基线逐字相等，所有 smoke/probe checkpoint finite，但相对基线
`10.0916 s` 均无可测收益。相同五轮 reset env 数为 `0/267/3103/875/2101`，扣除
无-reset 粗底座后仍约 `4.9--8.0 ms/reset-env`。

因此 rollback clone、PPO 与 exact table 不再列为下一主墙钟。下一门是严格 opt-in、
仅 diagnostic 可开启且默认零开销的 update profiler；按 birth reserve、broker/pool、
solver、task install 与未归因 reset wall 分段后，直接把最大段改为 compact batched
reset。formal receipt/checkpoint schema 在 N5 前另行闭合，不由 diagnostic compact path
代替。旧 long 不热补，旧 bundle/spec/claim 不重标；G05 继续 `Partial`。

该 profiler 门现已由 exact source `5e1443c4` 在 Pod1 GPU2 完成。focused suite
`136 passed`；`1 env×2` 恰好 2 行 profile JSON 且 checkpoint finite；same-seed
`4096×5` 恰好 5 行，collection=`2.687/3.646/17.717/9.456/18.148 s`，五轮
reset=`0/267/3103/875/2101`。总 collection `51.654 s` 中
`solver_solve_many=33.432 s`、`pool_request_many=34.724 s`，而 task install
只有 `0.202 s`。三组逐 update JSON 与 `4d631fb3` 基线逐字相等，五份 checkpoint
全 finite，无 timing-scope mismatch、Traceback 或身份漂移。

下一 Gate 输入因此收窄为 diagnostic-only compact/prevalidated task receipt；它必须保留
producer 的 solve/admission、exact-face/timing、sample draw、高水位、reason/conservation
和下游 birth/action/install 检查，只能裁掉相同字段在 receipt `__post_init__` 中的重复证明。
formal/default constructor 与 exact-resume schema 不得改变。该 replacement 未完成 Pod
parity 与 wall 复测前，G05 仍为 `Partial`。

后继 compact receipt/batch sampler/update-boundary telemetry、pose-OBB v4 和 host-only
solver result 已将同题五轮均值从 `12.341` 降到 `6.700 s/update`。
最新 source=`7f77ae5c`，Pod focused `104 passed`；`1 env×2` 两份及
`4096×5` 五份 checkpoint 均 80 tensor/76 浮点且 finite。same-seed
reset=`0/267/3110/884/2085`，table=`0/0/17/93/28`，actual-hard=
`0/267/3095/801/2062`，fall=0；joint-safety 与 exact-behavior JSON 和
pose-OBB 基线逐轮相等。profile 仍显示 solver 占 `16.367/32.924 s`
collection，因此下一门是 fixed-tape `cq_n_iters=4/6/8/12` 残差、admit、
reason 与 task 稳定性验收。G05 继续 `Partial`：这些是 diagnostic 训练证据，
formal checkpoint-granularity receipt 与 full exact-resume 仍未闭合。pose-OBB v4 的
stage/canonical/signed-prelaunch/generic-launcher consumer 已在 2026-07-31 同一批修复收敛，
不再列为运行 blocker；path-free runtime USD identity 仍是后续证据债。

### 2026-07-31 latest Agibot vendor baseline host gate

The adopted fresh-training identity now contains these independently auditable pieces:

- the full latest-vendor 29-DoF nominal/armature table in the shared A3 articulation and replay;
  head nominal values remain repository-owned because the vendor table has no head rows;
- gain DR split into startup Kp `log_uniform(0.8,1.2)` and Kd
  `log_uniform(0.7,1.3)`. The current inherited selector is `joint_names=[".*"]`, so it also
  randomizes the two repository-owned head joints; that extension is a HOPE choice, not a vendor
  31-DoF claim;
- immutable task profile
  [`HOPEPingPongActionBallA3VendorV1`](../DEFINITIONS.md#a3-vendor-v1-profile), with ungated
  [`axis_box_6d_v2`](../DEFINITIONS.md#axis-box-6d-v2) every `5–15 s` and one per-episode
  `[0,2]` [control-step action delay](../DEFINITIONS.md#control-step-action-delay);
- fresh normalizer ABI validation, per-update positive/finite realized-std and learning-rate
  receipt, once-per-swing strike-window-entry racket-distance bins, and finite default PPO budget;
- a separate single-GPU
  [`N1 vendor baseline diagnostic`](../DEFINITIONS.md#n1-vendor-baseline-diagnostic) that allows
  only seeds `0/1/2` and exact `smoke/probe/long` stages. It remains
  `diagnostic_unauthorized=true` and cannot mint formal evaluator, promotion, resume, export, or
  judge authority.

Focused host evidence includes vendor articulation/replay `23 passed, 9 skipped`; PD split/startup
`20 passed` plus `59 passed, 12 skipped`; delay/runner/stage/training-contract/exact-resume
`359 passed`; runtime guards `20+58+3 passed`; window-entry/exact-face/stage subsets
`34+13 passed`; the final old+new launcher suite `69 passed`; stage-evidence v4 `51 passed`; and
vendor eval + canonical admission + formal launcher `128 passed`. The latest materialization/pin
change additionally passes `90` focused non-Torch tests. These source checks do not authorize a
training launch by themselves.

Stage A has now run on Pod at exact source `5665963e96bf75c677e7669efc58c449e0c04876`.
The recipe-only stage and `1 env×2`
[`A3 vendor identity smoke`](../DEFINITIONS.md#a3-vendor-identity-smoke) passed, emitted schema-3
training-contract SHA `98fa3239daba825f07d3997fb28f4564c92967536f2552e6bdc0f8772781366f`,
and saved finite `model_0.pt`/`model_1.pt`. Runtime delay/ABI/std marker counts were `1/1/2`.
The policy recipe SHA
`27bf405e5677fe2e7bab6fcc15c166901734048dd334b8b0abc3a8ffef3ce416` is shared-ready only;
using it as a dynamic-ready recipe would silently collapse two different contracts and is forbidden.
The authority live-order bug exposed by Stage A is fixed.

The cross-bound `bh_loop_c` evidence set is now:

- dynamic-ready candidate SHA
  `c831a4e6d1c03519181efb090120a881702d113e95ebcf22f745a3a2ca4fc794`;
- nominal-hold receipt SHA
  `11c025dc25cba93c7d0d9894bac75da05a1a7aff11f797e9a35f9b2906f67740`, PASS for
  `0.8 s / 40` steps with feet-contact `1` and no terminal;
- bundle SHA `9881c52ca035bbdee0a3e1d0c0689eb7592b2a73b5442866a9a6e9480cbaae03` at
  `configs/n1_contact_vendor_a3_20260731/bh_loop_c.bundle.v2.9881c52ca035.json`;
- actual-authority receipt SHA
  `f66a9e59f441c22c465d3236d717c95354393d04c5975f58ece3e7612a65461a` at
  `configs/a3_vendor_runtime_authority_20260731/bh_loop_c.vendor_runtime_authority.v1.json`;
- materialized required-identity SHA
  `240f3757e45006de9dc5f4ecabcfc40071058009751fd1f0b8eb92656e1801ff`, binding the
  `98fa3239…` contract and naming only `bh_loop_c` as dynamic-ready action.

The launcher in this batch pins both required-identity and actual-authority SHAs, and the same
batch tracks their materialized files. Clean `f948a150` revalidated the authority chain. The
distinct **dynamic-ready recipe-only** path at `e7787e25` materialized policy `e408b845…c65d`, and
vendor diagnostic smoke claim `be783ab7…ad54` then completed `1 env×2` with two finite
checkpoints and ABI/delay/std-LR markers `1/1/2`. Same-seed `probe` (`4096×5`) is now the next
runtime gate. `bh_block` remains rejected. `long` remains
mechanically rejected until an actual probe produces a named `vendor_probe_gate_receipt` with the
runtime ABI/std/LR/delay receipts, safety denominators, `t_hit`, and entry-distance decisions.
Formal training, promotion, export, judge, deployment and hardware remain unauthorized. G05 stays
`Partial`.

Recipe r1 at exact source `2430fbb2` and claim `e37f8169…e32` passed clean authority, bundle,
GPU and schema-v2 pre-scene gates, then failed closed before recipe/PPO because MotionCommand's
runtime consumer still required schema-v1. The spent namespace produced no policy SHA. Its own
PGID `1328514` was terminated and GPU0 returned to 18 MiB. The consumer fix keeps v1 compatibility,
adds strict v2 plant/delay validation. The fresh rerun succeeded as described above; r1 remains a
permanently spent negative receipt.

The 2026-07-31 diligence and vendor setting are the current fresh-training authority. Earlier
audits bound to repository constants are historical evidence only; they do not regain authority
over this plant or prove compatibility with the old deployment decoder.

Evaluation now has two explicit, receipted profiles linked to the
[policy/action interface](../interfaces/policy_observation_action.md): `vendor_play_v1` disables
startup plant DR and interval push but retains policy observation corruption and the trained
episode-sampled delay; `deterministic_ranking_v1` additionally removes observation corruption,
delay and reset-state noise. Results must name the profile and cannot be averaged across them.

### 2026-07-31 vendor probe 与下一身份修正

Exact source `e7787e25`、policy `e408b845…c65d` 的 `bh_loop_c` `4096 env × 5 update`
probe 已自然完成，五份 checkpoint 各含 `83` 个 tensor / `2,056,100` 个元素且
all-finite；ABI/delay/std marker 为 `1/1/5`，4096 个 env 的延迟直方图为
`0:1357 / 1:1360 / 2:1379`。但这不是可放行的 long gate：五轮
`joint_actual_forbidden` 为 `875/3684/3143/3193/3191`，总计 `14,086`，主要是
waist-roll，其次是 waist-pitch；qdes-hard 始终为零。根因之一是 vendor
adapter 漏了已裁的 stable-ready override，使 full CoM/mass/PD DR 在首轮重新打开。

同一 probe 首次给出 `100` 个 strike-window entry 拍距：`97/100 > 0.20 m`、
均值 `0.4339 m`。因此下一 vendor leaf 直接保留精核
`weight=4,std=0.075 m`，并在同一 swing-through target/TIGHT window 上叠加低收入粗核
`weight=1,std=0.30 m`；其他 ActionBall task 默认为零。新 vendor effective-Reward
SHA 预注册为 `8220f339…54dc3`，identity recipe 会在 scene 之前重算拒绝假 pin。

safety 时序审计同时证明 brake 在 control-step delay queue 之后、每个 `5 ms`
physics substep 立即写入；`lag=2` 不是 brake 的 40 ms 额外延迟。最小 containment
把 plant-state trigger 从硬行程 `2%` 提前到 `5%`，不改 `20 ms` horizon、brake
公式或 raw mechanical-edge DoneTerm；新 lag-2 集成测试证明 brake 首个 physics write
生效且不污染 delay history。新 probe 仍必须以 raw-hard 为零或不再持续成风暴
验收；若失败，下一层是 runtime-PD-aware inward brake，不是放宽 terminal。

为关闭 push “静态接线即当作跑过”的证据缺口，vendor launcher 新增 exact
`push_evidence=4096 env × 32 update × save8`，覆盖 `15.36 s` policy time，并把
Pod 实际 IsaacLab interval scheduler 和 velocity-push 两份源码 SHA 封入 claim。在当前
`[5,15) s` timer 且普通 episode reset 不重置 timer 的实现下，自然完成才能机械
证明每个 env 至少执行一次 push。当前 S0 仍等 clean source commit→新
identity/runtime authority→dynamic recipe→smoke/probe/push-evidence；G05 保持 `Partial`。

### 2026-07-31：implicit-A3 Reward/config 真值清理

E1 source/config 复核确认，Hitter/DeployParity 的 production arms/waist 都是
`ImplicitActuatorCfg`，因而 `arm_torque_saturation` 没有已证的 explicit pre-clip demand。
旧 YAML 写 `-0.5`、组装期又强制清零，active Reward 虽始终为零，但名义配置会
误导人。两个 implicit 基线现显式写 `0.0`；explicit actuator 研究叶仍可 override，
backend compatibility receipt 仍保留。因组装后 effective term 原本就是零，该修正不改
训练收入或科学 setting。

同批把 HITTER-pure 源码注释的单一 `PD ±15%` 改为当前智元 startup Kp
`(0.8,1.2)` / Kd `(0.7,1.3)`，并将 Hitter YAML 的 IdealPD 旧理由替换为
position-target + implicit-PD 的实际 drive-gain 语义。相关两个 test files 全量联合回归
`223 passed`，py_compile 和 `git diff --check` 通过；
未生成 Pod/Isaac 行为证据，G05 继续 `Partial`。局部真源见
[effective Reward 因果账本](../experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md#2026-07-31implicit-a3-名义-reward-与组装真值对齐)。

### 2026-07-31：jerk 与 implicit-PD effort 的 probe-only 可观测性

DeployParity/HitterPure 现有 `action_acc_l2` 零权重时会被 RewardManager 剪掉，
所以没有二阶 action jerk 运行证据。新
[`action_acc_jerk_probe`](../DEFINITIONS.md#action-acc-jerk-probe) 只在显式布尔开关为真时
动态安装：manager weight=`1`，函数返回严格零，设备端每 update 累加
history-valid 分母、nonfinite、raw 二阶差分平方和/最大值、`36.0` 封顶和与超封顶数。
它不改 total/per-term Reward，也不把复位后历史不齐的前两步算成 jerk。

新
[`implicit_pd_post_step_effort_proxy_probe`](../DEFINITIONS.md#implicit-pd-post-step-effort-proxy-probe)
只在能证明 action joints 全部归 implicit actuator 所有时运行；显式或缺归属直接
fail-loud。代理量固定为 `Kp_live*(q_des_sent-q_post)-Kd_live*qdot_post`，使用 DR 后
live gain、真正下发的最终 `q_des` 与 policy step 末状态，记 `>0.9×limit` / `>1.0×limit`
的 joint sample 分母、和与峰值。这是 **analytic/post-step/non-actual/substep-blind** 诊断：
PhysX implicit drive 不暴露实际力矩，RewardManager 步末读取也看不到四个 physics substep
内的峰值；因此这些数不能叫 actual torque、actual saturation 或 substep peak，也尚未被定价为 Reward。

性能语义是硬边界：两个 rewards cfg 槽位默认都是 `None`，开关缺席或为假时
train translator 不构造 RewardTerm，runner 也只在 exact active-term name 出现后才消费标量账。
因此当前 vendor N1 的 RewardManager active graph 与每步热循环不增项。host 定向
`test_probe_only_observability.py + test_action_acc_smoothing.py + test_effective_reward_recipe.py =
48 passed`；邻接 reward/翻译/因果回归 `384 passed`，其余 `4` 个失败是父提交已稳定复现的
explicit `arm_torque_saturation`
mock 缺 backend ownership，与新 probe 无关。`py_compile` 和 `git diff --check` 通过。
尚无 Pod/full-scene 运行证据，这两个 probe 默认仍关闭，G05 继续 `Partial`。

### 2026-07-31：vendor dynamic-ready recipe/bundle 单一身份修复

new vendor replacement smoke 在进入 PPO 前因 policy recipe SHA 不一致正确
fail-closed：配置 `9fbc61ad…`，实际 `f76df202…`。它没有生成 checkpoint，
也没有执行 PPO update。用原 smoke argv 加 no-clobber recipe output 跑了一次
`1 env` 零 PPO 物化；递归 diff 证明 runner/policy/algorithm、completion stage
与 vendor contract 全部一致，唯一差异是 recipe wrapper 仍绑 r2 bundle 的
v3 dynamic-ready candidate/nominal-hold，而 smoke 已绑 r3 bundle 的 v4 资产。

修复不改 PPO、Reward、DR 或 safety setting：

- vendor launcher 新增唯一 code-owned canonical r3 bundle pin；
- smoke/probe/push/long 任一 spec 的 bundle path/SHA 不等即在 compose 阶段拒绝；
- dynamic-ready recipe wrapper 不再维护第二份常量，直接引用 vendor launcher 的 pin。

### 2026-07-31：双动作 vendor authority、nominal hold 与 contact bundle

`bh_loop_c` 与 `bh_block` 已各自从 exact runtime authority 生成 action-specific
dynamic-ready candidate，并在 Pod2 自然完成 `0.8 s / 160 physics step / 40 policy step`
nominal hold。两动作双脚接触率均为 1.0、无 terminal，plant contract 与 `[0,2]` episode
delay receipt 一致；相应 contact bundle 均为 `landing_claim=false`。精确 SHA 见
[ActionBall 分阶段准备账本](../experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md)。

loop 的前两次基础设施失败分别发生在缺少 private `libOpenGL.so.0` 与 fresh checkout 未进入
`PYTHONPATH`，均在 scene/receipt 之前；retry 修正 exact runtime 环境后自然通过。该证据只关闭
动作专属 artifact 链，不替代 shared-safety：同源 VendorV1 双包络 stress、fresh
`4096×5` actual-hard/nonfinite 零门及 `4096×32` push evidence 仍为 long 的硬前置，故 G05
保持 `Partial`。

同日对旧 8-env stress hang 的证据边界完成纠正：hang 位于 `gym.make` manager/command 构造，
尚未执行 reset、live-limit 写入或 sim step，因此不能写成 PhysX stress 失败。旧 harness 还绕过
了 long 使用的 VendorV1 Hydra profile。修复后探针 compose exact VendorV1 leaf、复用
`train.py::_apply_task_overrides`、从 code-owned registry 解析动作 identity，并在 vendor bind、
`gym.make`、reset、Hctrl/mixed readback 与 sim step 前后输出唯一阶段 marker。host focused
`20 passed`，但 live compose/PhysX 仍必须由新 clean source 的 Pod receipt 验收；G05 状态不变。

同批 registry 接入后，loop/block 的 runtime contract、required identity、runtime authority 与
contact bundle 共八个 pin 均逐文件 SHA 重算一致；host vendor 集成 `238 passed`，upper contact
bundle `11 passed`。本机无 Torch 的 6 个 full-body bundle 用例未参与该数字，必须在 Pod 补跑，
但不阻塞当前 upper N=1 stress；G05 仍只在 Pod shared-safety evidence 后推进。

新 stale-bundle 负例加入联合回归；recipe/launcher 回归 `49 passed`，
`py_compile` 与 `git diff --check` 通过。Pod 上仍须以这个 clean source
fresh 物化 r3/v4 recipe，再跑 smoke→probe/push；因而 G05 保持 `Partial`。

### 2026-07-31：vendor completion claim 的无自引用传递

r3/v4 policy recipe 在 Pod1 已精确物化为 `f76df202…`，闭合了上一节的
bundle 身份问题。后续 fresh smoke 在 `runner.learn` 前构造 natural-completion
payload 时拒绝：formal 路径的 `training_launch_claim_sha256` 为 null。这不是
训练失败；无 learning marker、无 checkpoint、无 PPO update。残留 Kit 进程已按
namespace 的 sidecar PID/PGID/starttime 三元组复核后对 exact process group 发 `TERM`，
GPU0 释放，GPU1 旧进程未动。

claim digest 不能写入它自己覆盖的 scientific argv，否则会形成不可解的
hash cycle。修复因此放在已完成 claim/source/bundle/GPU 双重验证的 internal-exec
边界：launcher 只向它 exec 的训练进程注入
`HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256`；trainer 只用这个值构造 vendor
completion marker，若 formal cfg 也存在则必须与 env 完全相等。非 64 位小写
hex、cfg/env 冲突都 fail-loud；普通训练仍 strict no-op。相关回归
`121 passed, 1 deselected`，后者仅表示 source 变动后必须重物化 tracked identity。
还没有 fresh smoke 运行 PASS，G05 继续 `Partial`。

### 2026-07-31：vendor claim 在 checkpoint/completion 的同一性

completion-claim v1 虽然避免了 argv 自引用，但独立审查发现 runner 构造
仍使用 formal cfg 的 null claim。这会导致 completion marker 含正确 claim，
checkpoint `infos` 却缺失 claim，而 probe-gate producer 要求 checkpoint、marker 和
namespace claim 三者精确相等；因此 v1 被判为 P0，未用来重签 authority。

claim-v2 将解析点移到 runner 构造前，且只有
`n1_vendor_diagnostic_stage` 与 `vendor_runtime_training_contract_sha256` 两个 key 都存在
时才读取 internal-exec env。解析后的同一 effective SHA 同时传给 runner
和 natural-completion payload。ordinary training 以及 vendor 半配置即使遇到坏 ambient
env 也不读它；半配置仍由原 builder fail-loud。新回归同时检查
complete-vendor 正例、ordinary/half-config no-read 和 runner/completion 两个 callsite 共用
单一变量；主线 focused `124 passed, 1 identity-rematerialization deselected`。
独立对抗复审 PASS（core `90 passed`），确认 checkpoint/completion 共用同一 effective
claim 且普通/半配置训练不读 ambient env。还待 authority 重签和 Pod fresh
smoke，G05 保持 `Partial`。

claim-v2 独立复审通过后已切 clean source S=`2bc020f8…`。该 source 不改变
schema-3 contract bytes，所以使用已跟踪的 `38974f1b…` contract 最小重签：
runtime-authority file SHA=`651b2dc6…d08`，required-identity file SHA=`21e33743…f6d`，
launcher 两个固定 pin 已同步。v4 candidate/hold 和 r3 bundle 内容未变且继续由
contract/motion/content seal 约束，没有冒充新 nominal-hold 运行。artifact-focused
`77 passed`，py_compile/diff-check 通过；仍待提交 A 和 Pod fresh recipe/smoke，
G05 保持 `Partial`。

### 2026-08-01：vendor-only 6% actual-joint guard 身份重签

5% 身份的 4096-env probe/push 在首次 PPO 更新后出现 waist-roll/pitch
raw actual-hard，但 worst penetration 仅 `1.1343e-4 rad`。source `fed55f55`
只将 vendor task 的 `pre_apply_guard_margin_fraction` 从 5% 提至 6%；全31关节
算术证明 executable `q_des` endpoint 逐项不变，waist-roll trigger 提前
`6.981e-3 rad`（约为已观测穿透的 61.6 倍）。这是 emergency
containment 阈值校准，不是 acceleration/jerk governor；raw terminal、PPO
action、delay/push 与 action-rate/action-acc Reward 均未改。host 组合回归
`412 passed`，独立复核 `377 passed, 1 expected deselected`。

Pod1 GPU0 的 exact clean `fed55f55` identity recipe/smoke 自然完成 2 次 PPO；
checkpoint finite 且含 `obs_norm_state_dict`，policy-contract SHA 仍为
`27bf405e…`，schema-3 training-contract SHA 仍逐字节为 `38974f1b…`。
因此 6% 不改变 runtime plant contract，旧 dynamic-ready candidate、nominal-hold
与 contact bundle 无需重烧；只重物化源码封印的 authority（file SHA
`05ed320e…`）、required identity（file SHA `a4c71e3c…`）并更新 launcher
pins。尚未运行 fresh `4096×5` probe/push，所以 actual-hard 零门仍未
通过，long 仍不得发射，G05 保持 `Partial`。

### 2026-08-01：6% Pod 零门否决与 projection 反证

artifact `25400403` 的 GPU2 exact run 正常进入 PPO，但 update0--5 的 push
application 均为 `0`，故只用于 shared-safety 判读。raw actual-hard/terminal 依次为
`0/0、33/15、3238/1303、418/169、2036/830、844/365`，明确否决零容忍门。
相对旧 5% 的 update0--2，`6% + max_inward_until_nonoutward_v1` bundle 的事件与终止
约减 `48.5%/48.7%`、最深穿透约缩浅 `66%`；但 margin 与 brake mode 同时变化，不能把
减量单独归因于 1% margin，也不能称为修复。owned 进程在 update5 完整 PPO/checkpoint
边界后停止，日志保留且不生成 gate receipt。

finite projection 的逐关节 TensorBoard 证据显示，actual-hard 主因 joint05
`waist_roll_joint` 在 update0--5 的 projection trigger/count 全为 `0`；joint08
`waist_pitch_joint` 的 lower saturation sample ratio 约 `1.2%--1.8%`，另有少量 joint19
`left_ankle_roll_joint`。因此当前证据不支持“raw actor 顶 waist-roll nominal projection
边界导致穿透”，下一修复必须对账 emergency guard 的 executed transient target、latch 与
implicit-PD plant。保持 raw mechanical hard-edge 零容忍；不继续加宽 guard，不增加
acceleration/jerk governor，G05 继续 `Partial`。

下一 source 的唯一授权候选改为双位置边界：URDF `H_mech` 保持 raw hard terminal 与
ledger 的不可放宽真值；只对 Pod 已实锤故障的 waist-roll/pitch 把 PhysX constraint 每侧
内缩 hard span 的 2% 成为 `H_ctrl`，其余 29 轴 live constraint 仍等于 `H_mech`，避免无证据
扩展 solver crutch。nominal executable `Q`、actor、delay、Reward、qdes projection endpoint
均不得变化，并须对两腰证明 `Q ⊂ 6% guard ⊂ H_ctrl ⊂ H_mech`。这不是
acceleration/jerk governor。候选只有在 runtime constraint readback、两腰双侧 5 ms ON/OFF
stress、fresh `4096×5` 机械边零门均过后
才能采用；只改 metadata、soft limits 随 `H_ctrl` 重算或给 actual-hard 加 tolerance 均
fail-closed。该改动属于 plant contract 变化，schema-3、authority 与 recipe 必须重签。

### 2026-08-01：双位置包络 source 与 1-env live readback 通过

vendor-only source 已实现上述候选，普通 ActionBall 的默认值仍为 `0.0`，不会触碰 PhysX
position limits。vendor leaf 的唯一启用值为 exact float `0.02`；运行时先逐轴核对 31 关节
identity、原始 `H_mech`、soft/Q 与 6% guard，再只对 `waist_roll_joint`、
`waist_pitch_joint` 计算 `H_ctrl`。写入使用与当前 Isaac Lab articulation backend 相同的
`root_physx_view.set_dof_limits(control_cpu, indices=_ALL_INDICES.cpu())`，避免 public writer
同步改写 data/default/soft buffers。setter 后立即 getter exact 对账；startup events 完成后的
首个 ActionManager reset 再做一次 exact fail-loud readback。这里没有声称每 PPO update 重验。

身份摘要分成两个语义：`setter_no_mutation_sha256` 包含 setter 时刻的 default-q 不变证明；
`run_specific_live_limit_sha256` 只绑定 joint order、live `H_ctrl`、`H_mech`、default hard 与
soft limits，允许合法 startup default-q calibration，且明确不是 recipe identity。host 定向回归
`364 passed`，`py_compile` 与 `git diff --check` 通过。

最终 source 在 Pod1 GPU0 完成 diagnostic-unauthorized `1 env × 1 PPO update`，自然退出：

- ACTIVE 与首 reset VERIFY 的 live SHA 同为 `aed12687...dbaa`，getter exact；setter no-mutation
  SHA 为 `4be7933b...ae7c`；
- update-0 共 `120` 个 env-readback samples，mechanical actual-hard/terminal=`0/0`；
- waist-roll 的 control min-gap 为 lower/upper `0.302961280/0.330007392 rad`，mechanical
  min-gap `0.316923904/0.343970016 rad`，最大 `|Δqdot|=0.188334320 rad/s`；
- waist-pitch 对应 control min-gap `0.449917696/0.395893760 rad`，mechanical min-gap
  `0.468069152/0.414045184 rad`，最大 `|Δqdot|=0.068935920 rad/s`；
- near/penetration/ballistic-attempt/capture/dwell/side-flip 均为零。所有这些量是复用既有
  q/qdot readback 的 kinematic proxy，不是 PhysX constraint impulse。

该 smoke 只关闭 API、setter 语义、startup/首 reset readback 与遥测接线风险，不关闭正式
机械门。下一步仍是两腰×双侧 5 ms H_ctrl ON/OFF stress，再重签 schema-3/authority/recipe，
运行 fresh `4096×5` probe 与 push-evidence；在 mechanical actual-hard/nonfinite 零门和任务
指标通过前不得发 long，G05 保持 `Partial`。

### 2026-07-31：legacy `reward_pack` 发车兼容修复

07-31 外部尽调的侧发现已由真实 compose 定谳：2026-07-25 将缺席
`task.rewards.reward_pack` 翻为 v2 后，旧 HOPE/Hitter/Rally YAML 仍全部缺席该键。
这些 env reward class 不声明 v2 的非零 `virtual_landing` 等直接项，所以不是
“自动升级为 v2”，而是会在 trainer 的 `_expand_reward_pack` 边界 fail-loud。
这会让默认 DeployParity 以及历史 HitterPure/Rally 命令不能发车，属兼容性
bug，不是一个应当保留的科学闸。

最小修复是由各 legacy task YAML 显式钉 `reward_pack: v1`：
`HOPEPingPong`、`HOPEPingPongDeployParity`、`HOPEPingPongHitter`、
`HOPEPingPongHitterPure`、`HOPEPingPongHitterPureRally` 和
`HOPEPingPongHitterPureRallyV3`。`HOPEPingPongRealSensor` alias 组装
DeployParity 后同样是 v1。`HOPEPingPongActionBall` 在 Hitter v1 之后显式覆写
v2，智元 `HOPEPingPongActionBallA3VendorV1` 继承的仍是 v2；因此修复不改
ActionBall vendor 身份、Reward 收入或发射 pin。

验证结果：

| task | 真实 Hydra compose | pack expansion |
| --- | --- | --- |
| HOPEPingPong / DeployParity / RealSensor | v1 PASS | 原 rewards node 原样返回，仅 v1 marker |
| Hitter / HitterPure | v1 PASS | 原 rewards node 原样返回，仅 v1 marker |
| Rally / RallyV3 | v1 PASS | 原 rewards node 原样返回，仅 v1 marker |
| ActionBall / A3VendorV1 | v2 PASS | 显式 v2 保持 |

另外，DeployParity 与 legacy full-observation `HOPEPingPong` 各完成一次
`1 env × 0 update` 真实 trainer dry-run：两者均自然 `rc=0`，均恰好输出一条
`rewards.reward_pack=v1 (legacy baseline)`，无 `ERROR during run`；DeployParity 还完成
175-D actor contract、hard-contract 和 effective-Reward receipt 构建。为不和正在进行的 N1
Pod Kit 相互阻塞，余下 legacy class 不再反复启动 simulator，由上述真实 compose +
trainer pack-expansion 回归覆盖。该修复只恢复旧 task 发车兼容性，不能作为 N1
vendor runtime gate 或任何 Gate Done 证据；G05 继续 `Partial`。

### 2026-08-01：三 lane 物化工具与双包络压测 host 收口

为了不在 Pod operator 层手写 loop/block 身份、scientific spec 或 SHA，本轮已实现
两个 source-only producer：

- `materialize_a3_vendor_required_identity.py` 从真实 Isaac 产生的 live schema-3
  training contract 安装原字节，再从 clean commit 的 exact deploy nominal 派生
  12 组/31 显式关节 required identity。两个固定路径一起保留、写入和
  fsync；任一失败就回滚本次保留，不接受 operator output/SHA。
- `launch_n1_vendor_baseline_diagnostic.py template` 只产生三条 code-owned
  scientific lane：反手拉 static、反手挡 static、反手拉 fresh-only monotonic
  adaptive sigma。long 先跟踪不含 source/GPU/namespace/log 的 scientific skeleton，再在仓外
  合并 runtime placement，避免 tracked JSON 对它所在 commit 形成自引用。

双位置包络的真实 PhysX 门不再借用训练日志猜测：新工具
`probe_a3_vendor_dual_position_envelope.py` 固定构造 8 env，即
waist-roll/pitch × lower/upper × Hctrl ON/OFF。每个 ON/OFF pair 的 q0、qdot、qdes
相同，只有 live limit 不同；理论 5 ms 点跨 Hctrl `0.6R`但仍在 Hmech 内
`0.4R`。它在 finally 中恢复全 env Hctrl 并 exact readback，恢复失败无法发布
PASS；20 ms 旧 diagnostic 继续只称 kinematic capture proxy，不冒充 constraint
impulse。

host 证据：红队后六工具 focused `113 passed`，stress focused `13 passed`；
三个脚本的 `py_compile` 和 worktree `git diff --check` 通过。本轮已额外覆盖
exact 12 组 consumer、发布前 clean/source re-attest、fchmod/write/fsync rollback、
exact vendor task 与 `sha256=None` 物化期的非跳过 unit fixture；独立复审无剩余 P0/P1。
旧 identity-repin suite 的 19 个 setup 错误已通过新 profile pins 和共同 clean
source `a2882d68…` 的 loop/block identity-bootstrap 真实物化关闭；没有修测试去接受
旧 SHA。回填 action registry 后，六组 A3 vendor 工件链 host 回归共 `115 passed`，
cross-action/source/profile/producer 闭包继续 fail-closed。

Pod2 首轮 `_r2` runtime 物化还暴露 identity launcher 的解释器路径 bug：
`Path(sys.executable).resolve()` 将 venv entry 退化为裸 `/usr/bin/python3.10`，formal pinner
在 Kit/PPO 前因缺 `yaml` 拒绝。新 source 保留 spec 已验证并封印的 venv entry
path，回归 `89 passed`。该失败 namespace 永久 spent，零 recipe/零 PPO。

8-env dual-envelope 真实 Pod 工具进入 scene/simulation start 后在 MotionCommand 初始化警告后
5 分钟无日志/无 receipt，leader 持续约 111% CPU；按 exact PID/PGID/starttime 绑定
TERM，12 s 内整组自然退出。该运行只判 harness hang，不得冒充 PhysX PASS/FAIL。

当前仍是 `Partial`。Pod2 exact `fdc43396…` 已分别产生 loop/block live
contract 和 required identity，并在 `205a0c52…` 跟踪；四个 file SHA 已回填
action registry。clean `f0a949bc…` 又生成 loop/block runtime-authority，file SHA
`04418ced…/648182df…` 也已回填 registry，全链 host 回归 `118 passed`。
identity/runtime/authority 层因而脱离 `None`，但 dynamic-ready/nominal-hold/contact
bundle 仍未重签。同时 8-env
stress 仍是 harness hang，没有 PhysX PASS/FAIL。必须先完成上述工件链与新
stress receipt，再跑 `4096×5 → 4096×32`，最后才能发三条 `4096×20001`
long。

### 2026-07-31：stress publication 生命周期修复

Pod Torch 已补跑 contact bundle 全量 `17 passed`。clean `07ba61cf` 的 stress v2 也真实越过
VendorV1 bind、`gym.make`、reset、Hctrl/mixed readback 与唯一 5 ms sim step，但没有 receipt：
`_run_live.finally` 中过早的 `simulation_app.close()` 在 Isaac 4.5 下 `os._exit(0)`，抢在外层
validate/publication 之前。该 run 只证明旧构造 hang 已解除，不能证明 stress PASS。

修复把 restore/validate/source re-attestation/no-clobber publication/flush 移到 Kit hard-exit
之前，FAIL 与未发布路径保持非零退出；host focused `21 passed`，v2 路径永久 spent。G05 仍为
`Partial`，必须由新 clean source 的 v3 receipt 裁定真实 ON/OFF PhysX 结果。

v3 随后生成 canonical FAIL receipt `ec1eebc…`，错误为每 joint/side 的 aggregate
`attempt/capture/penetration=3/1/1`，而旧 harness 预期 `2/1/1`。这里 3 不是额外 PhysX
attempt：pre readback 对 ON/OFF 两个 env 贡献 2，post readback 对仍在 Hctrl 外的 OFF env 再贡献
1；capture/penetration 各 1。下一 source 按采样相位拆账，pre 必须 exact `2/0/0`、post 必须
exact `1/1/1`，不通过放宽 aggregate 阈值制造 PASS。同时 validation FAIL receipt 将保留 finally
完成的 restore 与 raw observations/diagnostic，避免 v3 默认 `restore.attempted=false` 丢证据。

### 2026-08-01：v4 首步机械结果与完整 policy-horizon 复验

clean `cf79d84f`、producer SHA `1b77c9bd…` 的 Pod2 v4 已自然生成 no-clobber FAIL
receipt `d5e0fc4b…`；canonical JSON 的尾换行计入 SHA，重算一致。receipt 的 source、motion、
VendorV1 profile、`[0,2]` delay、2% Hctrl、6% guard、mixed live limit 和 finally restore 均
闭合，restore exact。

首个 5-ms substep 的机械位置结果不是失败：8/8 行 qdes 与 float32 q0 exact、全部 finite；
Hctrl ON 四行均严格位于 Hctrl 内，OFF 四行均进入 `[Hctrl,Hmech)`，所有行仍严格位于
Hmech 内，最小 mechanical gap 为 `0.003570497 rad`。严格 FAIL 来自旧 diagnostic 的
`capture_proxy`：它要求上一 readback 有 20-ms ballistic attempt，且当前 readback 已回内并且
速度非外向。waist-pitch ON lower/upper 在一个 substep 后 qdot 为
`-0.03480399/+0.04559414 rad/s`，相对 tape 初值 `-2.5412/+2.5412 rad/s` 已分别减少约
`98.6%/98.2%`，但还未反号，因此 capture 为零；这不是 Hmech 穿透、readback 漂移或恢复失败。

下一 source 不改 Hctrl/Hmech、guard、qdes、plant 或旧 20-ms proxy。probe 改为覆盖真实一个
policy horizon 的 4×5-ms trajectory，每个子步记录 q/qdot/qdes，并严格要求：ON 每步都在
Hctrl 内；ON/OFF 每步都在 Hmech 内；qdes=q0、finite 与最终 restore exact；OFF 首步仍须
进入 `[Hctrl,Hmech)` 证明 A/B stress 有效。只有该轨迹 receipt PASS 后才进入 fresh
`4096×5`。v4 namespace 永久 spent，G05 继续 `Partial`。

4×5-ms successor 已在 source 实现并版本化为 schema/kind v2。运行时不再只保存首步末态，
而是逐子步把 q/qdot/qdes 追加到 failure-safe trajectory；任一中途异常也保留已完成前缀。
validator 对每个 ON env 要求四步均 strict Hctrl，对 ON/OFF 全部八行要求四步均 strict Hmech，
并保留 OFF 首步进入两包络间、qdes=q0、finite 与 finally restore exact。旧 20-ms proxy 的
pre/post 首步采样和原始字段不变，但 receipt 明写 telemetry-only。host focused `24 passed`，
`py_compile` 与 `git diff --check` 通过；仍须 clean-source Pod receipt，G05 保持 `Partial`。

clean `861a7842` 的 v5 首次 Pod 尝试在 `gym.make` 前因
`/workspace/franco/runtime_assets/a3_preconverted_usd_1b3fecd7/model.usd` 缺失而生成 schema-v2
canonical FAIL receipt `5c6d09de…`。该收据没有 env、restore 或 trajectory，故只记基础设施失败。
完整 6-file USD bundle 已从 `simple_half_second_sprint_20260718` 登记副本恢复，文件数、总 bytes
及 model/base/physics/sensor 四层 SHA 均与 setup runbook 一致。v5 路径 spent；同一 clean source
须用 fresh v6 路径重跑，G05 保持 `Partial`。

v6 随后在同一 clean source 上完成 gym/reset 与全部 4×5-ms 轨迹，canonical content SHA
`da977d6…`、receipt file SHA `eb93a9f…`、log SHA `8eb17b8c…`，qdes 和 finally restore 都 exact。
它揭示 v2 validator 的验收方向需要修正，而非 plant 失败：Hctrl ON 的 solver 级最大控制包络
penetration 只有 `6.06e-5 rad`，且四 tick 全部严格留在 Hmech 内，最小机械余量
`0.01392266 rad`；相同 q0/qdot/qdes 的 Hctrl OFF 四组都在 tick2 穿过 Hmech，最大机械
penetration `3.27e-4 rad`。因此“ON 每 tick strict Hctrl”会误杀真实有效的约束，而“OFF 每 tick
strict Hmech”会否定本来就应越过机械边界的 positive control。

successor 版本化为差分 schema/kind v3：ON 的所有 tick 必须 finite、qdes exact 且 strict Hmech；
若有 Hctrl penetration，它必须严格小于该侧 Hctrl→Hmech 的完整 cage reserve，并同时记录
`max_on_ctrl_penetration_rad` 与 `min_on_mech_gap_rad`。OFF tick1 必须进入 `[Hctrl,Hmech)`，且
后续至少一个 tick 必须触/穿 Hmech；两组的 q0/qdot/qdes 必须相同、最终 restore exact。OFF 穿
Hmech 只证明同带差分因果，不是安全接受；Hmech 对 ON 仍零容忍。旧 20-ms proxy 继续 telemetry-only，
v6 namespace 永久 spent；只有 fresh v3 receipt PASS 才进入 recipe pin 和 `4096×5`，G05 仍为
`Partial`。

差分 v3 已在 clean source `dff36ad4…` 实现。初版被独立红队拒绝，因为把刚写入的 q0/qdot
input tensor 当成 live 初态会留下错误 PASS 缺口；最终版在 `write_data_to_sim()` 后、首个
physics tick 前直接调用 `root_physx_view.get_dof_positions()` 与 `get_dof_velocities()`，逐行对
float32 tape 并对 ON/OFF exact。旧 20-ms diagnostic 的 horizon、semantics、shape 和 counters
全部退出 verdict；pre/post 只需是可 canonical 记录的 Mapping，best-effort 指标解析不到写 `null`。
focused `28 passed`，实现复核和独立红队均无 P0/P1。下一门是 Pod2 GPU0 的 fresh v7
no-clobber schema-v3 receipt；未取得 canonical PASS 前 G05 继续 `Partial`。

fresh v7 已在 clean `956a7a3a488cbda316875f31f08b5c2bb55678ae`、Pod2 GPU0 natural rc=0
完成。canonical content SHA `06da2c9109d272c60be6049dc333f75a6512e0013936220a1eed9fc622c4e6eb`，
receipt file SHA `1dd6ef2f5ade3533e7ab94e3e933382b33f4f980517f638dde78b3773ad1b67e`，log SHA
`49f8c3f7e1a262aa76a887c887954942aa81a8a20c7780d0bcc3dd1d97c57642`。live PhysX q0/qdot、
逐 tick qdes、ON/OFF same-tape 和 finally restore 全 exact。ON 最大 Hctrl penetration
`6.0558319e-5 rad`，最小 Hmech gap `0.0139226615 rad`；OFF 四个 joint/side 都触/穿
Hmech，共 10 tick，最大 penetration `0.0003274977 rad`。机械差分门 PASS，且 receipt 明确
training/deployment/hardware unauthorized；G05 仍为 `Partial`，因为 `4096×5`、push `4096×32`
和 long gate 尚未完成。

v7 同时再次暴露 recipe 前基础设施 P0：manual launch 只有 exact private OpenGL + private GLU
loader path 才能启动，而三个正式 exec 入口目前的 runtime-asset claim 只 pin GLU 并重建为单一
GLU 路径。下一代码动作已先登记到 EXP 为 `RUNTIME-ASSET-LOADER-V2`：nested schema/kind v2
同时封存 OpenGL/GLU 固定目录、library SHA、direct SONAME 和 exact `OpenGL:GLU` loader string，
三个入口共享 claim-owned helper；旧 v1 claim fail-closed。完成该门后才物化 final recipe pins。

loader-v2 source 已在 clean `422777080165d420fe7ffc2773f68edb780b71b1` 完成：四组 focused
`170 passed`，三个脚本 py_compile 与 diff-check 通过；missing/reverse/tail/tamper/旧 v1 都在
exec 前 fail-closed。独立红队指出 pathname recheck 到 ELF open 不能抵御并发本地写者，最终合同
未虚称 immutable，而是强制 claim 字段
`pathname_sha256_revalidated_immediately_before_exec_no_concurrent_local_writers_v1`；plan→exec
窗口 runtime tree 必须 quiescent。复核后无 P0/P1。Pod 门还需 `ldd` exact、plan 正负控与同一
绝对 checkout 上的三条 zero-PPO C0 物化，G05 仍为 `Partial`。

Pod live 已进一步证明 loader-v2 本身闭合：`ldd` 将 private GLU 依赖的 `libOpenGL.so.0` 解析到
exact private OpenGL，missing/reverse/ambient-tail 三个 plan 都在 namespace 前拒绝。首个 correct
loop plan 随后在下一层发现 dynamic-ready path identity 断链：registry 声明 repo-relative stable
motion，而 tracked candidate `847ffe78…` 保留其生成 worktree 的旧绝对 source path。拒绝发生在
Kit boot、GPU claim、namespace 与 PPO 之前，没有训练副作用。下一门不是放宽 SHA 或 action 检查，
而是先核实该字段的 producer/consumer 语义：若是 runtime identity，则在固定 final checkout 重物化
candidate/hold/bundle 并 repin；若只是 provenance，则须迁移为完整 repo-relative logical suffix +
exact current tracked blob SHA，同时继续把 candidate/hold/bundle 的绝对 runtime pin 纳入 policy
binding。该决策与负例进入 EXP 的 `DYNAMIC-READY-PATH-IDENTITY` 后才允许改代码；G05 仍为
`Partial`。

该断链已按同仓 vendor authority 的跨-checkout 语义修复，而非重物化或删除检查。launcher 现在
只接受 exact repo-relative path，或 normalized absolute provenance 的完整 repo-relative component
suffix；同时显式拒绝 dot/dotdot、重复或尾斜杠、双根、控制字符、relative prefix 和
same-basename/wrong-directory。通过逻辑路径后仍用 registry exact pin 调 `_verify_tracked_file`，对
source commit blob 与 current worktree motion bytes 双重 SHA；action、frame0、artifact SHA 与 runtime
contract SHA 检查均保留。training contract 的当前 artifact/receipt canonical absolute path、无 symlink
和 file/content SHA binding 未改。相关整合 `139 passed`，pycompile/diff-check PASS；独立红队
P0/P1=0，真实 tracked loop/block r2 validator 正例通过。下一门是 clean C0 的 Pod correct plan 与
zero-PPO materialization；G05 仍为 `Partial`。

clean `0670ad1f86cf8f7f8b8e1810fe98442933be6892` 的 correct loop/block zero-PPO
materialization 已分别自然完成并产出 policy SHA
`ddcc1a7cc36f9c42098ca90473d199b74e1f7be51b26cf543badf872f6b9a09f` 与
`73d9de685f35e32a99e5d1098a67e8e1524d47835679779f5796a22d30b71e51`；两者均为
0 PPO、0 checkpoint、全 authorization false。adaptive-sigma hash r1 则在生成 canonical
effective-Reward receipt 后被 wrapper 拒绝。逐字段审计证明 receipt 的 schema-1 语义本来就只含
weight 非零项：三个 adaptive additive 核为 `0.20/1.0/0.52`，共同 coarse position 为 `0.30`，
weight=0 的 `racket_strike_success` 正确省略；train 在写 receipt 前另行校验 additive/success
三宽锁步、三旗和完整 schedule，runtime scheduler 也先验四份配置再原子更新。因此本次是
validator false-negative，不是 Reward producer 或训练 compose 漂移。wrapper 已改为校验四个
非零有效核，并明确拒绝 receipt 意外激活 success；host focused `121 passed`，独立审计无 P0/P1。
旧 adaptive r1 namespace 永久 spent；下一门是在修复后的同一 clean source 上以 fresh namespaces
重做 loop/block/adaptive 三份 zero-PPO pin，三者同源后才允许回填 launcher。G05 仍为 `Partial`。

修复后的 clean C0 `7587124db729a86867e74e48f2e0c6a7d0c5acb2` 已在固定
`/workspace/franco/a3vendor_final_pin` 复跑全部三份 zero-PPO materialization。loop/block policy
SHA 仍为 `ddcc1a7c…a09f` / `73d9de68…1e51`，证明 validator-only 修复没有科学漂移；adaptive
effective Reward SHA 为 `6520f153…63db`、receipt file SHA `fbf1c09c…2960`，31 个 active term
且不含 zero-weight strike-success。claims=`555aed65…` / `e512c38a…` / `310464fc…`；三个结果均
accepted、0 PPO、0 checkpoint、全 authorization false，child/PGID 自然退出、GPU lock 释放且
source clean。Pod 依赖相关测试 `403 passed`，第二轮独立 review P0/P1=0。三 SHA 已原子回填
code-owned launcher；提交窄 C1 后必须用 vendor baseline diagnostic 做三 lane `1×2`，不能用
shared-ready identity smoke 代替。G05 仍为 `Partial`。

2026-08-01 的 hybrid plant successor `db64751767cccfb665e7d57b64100758d301135c` 首轮只取得
`65 passed, 9 skipped, 12 failed`，因此本 Gate 仍为 `Partial`。12 个失败全部从
`test_materialize_a3_vendor_required_identity.py::_joint_values()` 的旧 full-precision armature
夹具开始，生产 required-identity authority 按设计拒绝；没有证据支持放宽 production literal
校验。本轮先同步测试夹具，再在同一个 clean successor 的 Pod checkout 并行重跑 plant、push 和
integrated-gate focused suites；三组未全绿前不得物化新的 `4096×5` claim。

successor `ac64553c117ebc1938bf3722ea5bfc222639bb77` 的 Pod plant focused 已闭合为
`77 passed, 9 skipped in 9.58 s`，且 production authority 未放宽。push focused 的系统 pytest
因缺 Hydra 未收集；改用项目 venv + `/tmp` 隔离 pytest-only shim 后为 `165 passed in 2.97 s`，
没有修改共享 venv/系统包。integrated focused 为 `46 passed, 1 failed`：唯一失败是
`+task.table_contact_attribution_diagnostic=true` 没被旧无 `+` 的 scientific-argv prefix 排除。
下一 successor 只修该 exact prefix 并保留 `...diagnostic_extra` / `...other_diagnostic` 反例，Pod
只重跑 integrated 组；通过后才能重签三 lane pins 和发起真实 `4096×5`。
