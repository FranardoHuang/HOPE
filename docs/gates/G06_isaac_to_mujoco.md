# G06 Isaac-To-MuJoCo Parity

Status: Partial (parity procedure operational and used to gate the 2026-07-02 sim-to-real; formal per-checkpoint acceptance thresholds still to be recorded)

**2026-08-02 successor 提案（Gate 仍 `Partial`）：**下一版将 MuJoCo 设为 N73 主训练引擎；Isaac
只提供 N1 最小可学证据和冻结 handoff。G06 未来应拆成 portable contract/plant/reward/reset parity、
可选 Isaac checkpoint replay diagnostic、MuJoCo native VecEnv/PPO 三个子门；本页下方的 mandatory
Isaac-trained ONNX 与 reset-first 179-D 条款是旧版接受条件，尚未由 successor 实现取代。新依赖、
真球 matched benchmark 和验收草案见
[MuJoCo 原生下一版准备账](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)。
在代码/合同和 `main` 主板切换前，Gate 状态不晋级。

当前 MuJoCo 实现状态已不再只是 single-env：
`deec4a52c758b1f173436d4522e3e13e7ccb7bfd` 已在 native physical-ball core 外增加一条
CPU sequential diagnostic `VecEnv`，具有 deterministic batched reset、76-D purpose-group observation、
finite no-reward rollout、strict physics-substep contact-event ledger 和 exact tape-timeout latch。
`41411c3b6a6ef3ad03c2cba41370e84709066d8d` 又从
`HOPEDeployParityTerminationsCfg` 绑定了两个 exact base termination subset：
`base_fell_tilt := pelvis_up_world_z < cos(0.7)` 与
`base_too_low := pelvis_link_origin_height_w_m < 0.5`，都是严格小于、control step 后取样、
sticky latch，子集内的 reason order 为 tilt 优先于 height。其 Isaac 源配置字节 SHA 被固定；
源语义漂移时 fail closed。termination blocker receipt 只在首次校验源后缓存，
4096 次 cache-hit 调用合计 `.446 ms`，receipt SHA-256=`353382b4…3789`。

2026-08-03 当前 successor 又加入 `joint_actual_forbidden` exact diagnostic predicate：每个
control step 后，以 runtime joint order 将实际 `q` 对照 MuJoCo `model.jnt_range`，固定
exact-zero bounds tolerance，并 sticky 保留同一 tick 内任一 physics substep 触边；非有限/无效区间或到达任一 raw hard edge 即触发。它与 tilt/height
共享 sticky latch，reason order 为 tilt→height→joint actual。Isaac 配置及 termination callable
源码分别 SHA pin，漂移即拒绝。Host 三组聚焦回归为 `45 passed, 8 skipped`；skip 是缺少
MuJoCo/SciPy 的真引擎用例。该增量仍是 `diagnostic_unauthorized`，robot/table、qdes、phase/recovery、
compact reset、Reward/PPO/save/resume/export 继续 fail closed，G06 保持 `Partial`。

同日 current-worktree successor 再关闭 `joint_qdes_forbidden` 这一具名子项：绑定 Isaac
`pre_clamp_qdes_forbidden_zone`、`joint_pos_limits`、`margin_rad=0`、`margin_fraction=0.02`
与 `finite_preclamp_qdes_projection_enabled=true`。因此 finite pre-clamp 越界仍由 projection+
penalty 保留 transition，只有有效 affine qdes 含 NaN/Inf 才触发 hard termination；reason order
扩为 tilt→height→joint qdes→joint actual，sticky latch 不变。receipt 继续双源码 SHA pin，
host 三组聚焦回归 `45 passed, 10 skipped`（新增 skip 仍来自 host 缺 MuJoCo/SciPy）。PPO
`step()` 继续在 physics 前 fail closed；robot/table、phase/recovery、compact reset、Reward、
save/resume/export 仍未闭合，Gate 不晋级。
同一 exact `0d1d641e` 随后已传入 Pod1 clean checkout
`/workspace/franco/actionball_mujoco_0d1d641e_20260803`，用 Pod 现有
`/workspace/hope_isaac_venv/bin/python` 执行完整 native suite=`55 passed in 13.59 s`，
所有 host optional skip 在 Pod 均实际执行。这只证实 current diagnostic
scene/VecEnv/termination subset 能在 Pod runtime 运行，不改变上述 PPO 阻塞。

current successor 继续移植 `robot_hit_table`：新 exact guard 重开 Isaac
config/callable/`hope_actions.py` latch、43-component collision artifact 与五实体 table
geometry 的 SHA，并只接受 canonical root MJCF。它还将 verified base model 与实际
augmented/precompiled live model 的 32 个 owner body 按 name、selected parent、local
position/local quaternion 逐项比较，同名但 owner-frame 漂移会 fail closed。随后按 component
world-AABB + live racket OBB broadening 对加 `0.02 m` margin 的桌体 AABB 做 inclusive
overlap。每个 physics substep 取样，control step 内 sticky 并保留首次 substep；只接受
decimation=4，reason order 为 tilt→height→table→qdes→actual。Host 完整 native
suite=`60 passed, 12 skipped`，skip 是 host 无 MuJoCo 的真引擎路径；该增量尚待 exact Pod
重验。随后 current-worktree successor 已实现 diagnostic lane 的 per-env episode done latch 与
terminated-batch compact reset：`episode_dones=exact_hard_terminations OR time_outs`，只 reset
命中的 core/episode length/hard latch/ledger，未命中行连续推进。返回 observation 是 reset 后
next state；另以 mask 绑定 reset 前 terminal observation，terminal ledger 也是 reset 前 caller-owned
deep copy。`robot_hit_table` 的首次 physics substep 保留在 terminal snapshot，新 episode latch 清空。
rollout v4 receipt 按 env 冻结完整 question source SHA 列表，并公开 semantic 与 digest-only terminal
trace descriptor，使 receipt 加返回 trace 可独立重算总 digest。Host 四组聚焦回归=`62 passed,
13 skipped`；skip 是 host 缺 torch/MuJoCo 的集成路径，不是 PASS。

后续 current-worktree 最小切片已从 Isaac authority 精确绑定 reference-envelope 三项：
`anchor_pos`、`anchor_ori`、`ee_body_pos`。阈值分别为 anchor z 误差严格 `>0.25 m`、
reference/robot projected-gravity body-z 绝对差严格 `>0.8`、四个 feet/hands body z 误差任一
严格 `>0.25 m`；等于阈值不触发。`recovery_hold` 的 `in_hold=true` 会屏蔽三项，且仍受
ActionBall episode-frozen `reference_terminations_enabled` gate 约束。hard reason order 现固定为
anchor pos→anchor ori→end-effector body pos→tilt→height→table→qdes→actual。
所用 termination class inheritance 与 direct term assignments（含源码顺序）、raw predicates、
hold/gate helpers、command gate 与 A3 feet/hands assignment 采用 selected-AST semantic SHA pin；
相关语义改变会 fail closed，无关 A225/C225 或
reward WIP 不会伪造漂移。既有 table/base guard 也已收窄到实际消费的 term factory、termination
classes、`robot_hit_table` callable、physics-substep latch class/method 与 qdes/actual callable AST；
阈值、term/order 或 latch 语义变化会拒绝，无关 config class append 不漂移，不再依赖易碎的整文件 SHA。

production `MujocoN1BallCore` 现已提供显式、不能猜 phase 的安装缝：只有加载外部 SHA-bound
`a3_mujoco_phase_fidelity_reference_tape_v1` 后才广告 sample contract。reference tape 绑定
plant/scene/robot-tape/sample-contract SHA、`pelvis_link`、四个 feet/hands body order、逐 tick
post-control-step MotionCommand reference、hold context 和 episode-frozen gate；row 数必须与 robot
tape 完全相等。core 从 live MuJoCo pelvis link-origin z、pelvis rotation 推导 projected-gravity-z，
并从四个真实 body `xpos.z` 计算误差，未读取或推断 `time_to_contact`。文件/content seal、authority
source SHA、binding、gate 恒定性、hold/context、finite/range 任一不符均 fail closed。

VecEnv 仍要求全部 core 同时广告相同 contract SHA 且每 tick 返回完整 sample；mixed advertisement、
SHA 不同、漏样本或未广告却返回样本都会使整批失效。默认未安装 external tape 的 core 保持不广告，
receipt 写 `exact_phase_fidelity_runtime_sample_available=false`，当前 formal blocker 是
`native_core_phase_fidelity_reference_tape_not_installed`；安装合法 tape 后 termination receipt 才变为
`FORMAL_TERMINATION_AVAILABLE_DIAGNOSTIC_ONLY`，training/promotion 权限仍全 false。rollout receipt 现为
v4，并把 runtime availability、contract SHA、reference-tape SHA lineage、每 env canonical phase sample
与 native physical-event facts transcript 纳入 digest。当前 phase sample contract
SHA=`e33568f5…f1d2596`；host 五组扩展回归=`89 passed, 18 skipped`，其中指定的
core/termination/vec/reward 四组=`72 passed, 13 skipped`。这是重验前的 host 口径；其中
真实 MuJoCo core emission 与部分 torch VecEnv runtime 当时因缺依赖而 skip。
完整 Reward/PPO/save/resume/export 仍 fail closed，G06 保持 `Partial`。

当前 successor 又修复了两个跨 runtime 问题：selected-AST pin 不再受 Python 3.12+
新增空 `type_params` 影响，并对 `Ellipsis/bytes/complex` 做显式可移植编码；
runner exact-resume tensor digest 先把 scalar reshape 成 1-D 再 view bytes，不改任何 tensor
内容。host native+plant 联合回归=`115 passed, 18 skipped`。exact Pod detached clean
`299145e9` 又分别通过 native=`110`、plant=`26`、runner guards=`25`，合计
`161 passed, 0 skipped, 0 failed`。这些只关闭 diagnostic core/runtime guard 的
Python 3.10/MuJoCo 3.10/Torch 2.7 复核，不是 Reward/PPO 或 normal-step 授权。
同一 exact `7135d5ce` 随后已传入 Pod1 clean checkout
`/workspace/franco/actionball_7135d5ce_20260803`，上述四组完整 native 回归=
`72 passed in 17.44 s`；这关闭当时 table-guard successor 的 host optional skip，但早于上述
compact-reset/lineage successor；该 successor 的最新 exact Pod 复核是上述 `299145e9/161 passed`，
两者都不改变 Reward/PPO blocker 或授权状态。

single-env 底层仍绑定 schema-3 31-D action、implicit total-PD、episode-fixed delay、
immutable teacher reference + 独立 sealed physical reset/hold 和 100-tick fixed tape。
首轮 tick9 hand↔hip/wrist↔table 失败的根因是把动态 v5 teacher frame0 当成静态出生状态；
teacher reference 没有被改写，physical reset 现使用在当前 exact MJCF 重审的 shared
root/leg + v5 非腿关节，并由 LP 求 envelope 内 hold qdes/history。修复后 d0/d1/d2 各跑满
`100 ticks / 400 substeps`，qdes clamp、velocity、自碰和桌碰事件全为0，因此状态更新为
`IN_PROGRESS / BIRTH-HOLD-SAFETY-PASS`，仍不是 trainer ready。三条 effort clip 分别为
`1108/1098/1084`，不得外推为机械准入或 learnability。clean Pod
`/workspace/franco/actionball_mujoco_41411c3b_20260803` 上三个聚焦测试集为
`48 passed in 15.71 s`。但正常 `step()` 仍在 physics 前 fail closed：剩余的
Isaac-equivalent robot/table collision termination、phase fidelity、
terminated-batch compact reset、teacher/official-racket-site p/v/face/long-axis、完整 reward 与
PPO/save/resume/export 仍未闭合。formal canonical N1 authorization 也仍因最终
ABI/reward/scheduler/measured authority 未冻结而 `BLOCKED`。
详见 [MuJoCo native single-env 运行账](../operations/run_mujoco_native_single_env.md)。

2026-08-03 的并行增量已将 single-env 推进为**native physical-ball plumbing probe**：新 scene
绑定 table/ball/racket contact pair、portable/backend asset closure、immutable-question/external expected SHA，
并在每个physics substep上 latch首次接触、recontact/同时接触invalid及contact-end outgoing state。
Host为`30 passed, 7 skipped`，Pod MuJoCo 3.10.0为`37 passed`；一次真immutable authority演练
运行400 substeps、仅触发一次table edge，跨reset/fresh-core trace确定。但explicit launch还没重现
immutable tape的aero/table-bounce轨迹，收据正确写`incoming_question_parity=false`；没有racket hit、
reward、PPO、checkpoint或trainer授权；现在已有的是上述 no-reward diagnostic VecEnv，
不能把它写成 trainer。因此 Gate 不晋级。

MuJoCo 拍面几何使用 2026-08-03 v2 exact identity：`right_racket` site/FK 不变，只修正
collision proxy 的 Y 厚度，新 root MJCF SHA-256=`70c4fd65…36c0a`。旧 v1 identity 仍
保存历史收据；formal lane 须新建 v2 的 L0/vendor-L1/table-net successor 链，禁止
把旧证书原地 repin 到新 MJCF。

portable parity 还必须绑定两项新权威：同一份实测 racket teacher，以及同一批 fixed
swings 上的 reward landscape/实际收入收据。旧 schema-v3 长轴错了45°，已 revoked；本地
schema-v4 已用 URDF/MJCF 正确轴完成 exact `73/73` full-phase 运动学重定向、50 Hz 物化与
独立 FK 反算，并生成 receipt-bound 73-action manifest。完整机械口径仍是
`0/73 admitted`：`57/73` 有已观测 hard failure，另 `16/73` 仍为 `UNKNOWN`；
37/73 超速、58/73 近限位是较早窄口径机理反例。prototype/source capsule/final ABI 也未闭合，
因此尚不是 formal N73 teacher。Isaac 与
MuJoCo 必须分别从自己的 achieved FK 对同一 measured teacher 计误差，不得用 retargeted q 自生
teacher 再宣称 parity。

V2 reward 已在实际 profile 上改动；冻结历史误差上收紧/初始 adaptive sigma 的
window 收入为 `2.664360/2.872667`，且对实际73 catalog 的静态会计是73/73满足
max motion `3.6575` < target `4.0296/4.3104` < landing `6`。迁移前两端仍须逐项匹配 `e/sigma`、有限差分
改善、eligible denominator 与 discounted per-swing 训练收入，并验证
`动作模仿 < 目标击球 < 上台结果`，不能用 static/counterfactual 结果代签已学会。

## Goal

Test whether a policy learned in Isaac can be replayed or approximated in MuJoCo.

This gate is the sim-to-sim bridge before real deployment.

## Inputs

- Isaac-trained policy ONNX from G05 (exported with the full metadata contract).
- MuJoCo A3 model from G04 (`a3_pingpong.xml`).
- Shared joint order and observation/action contract (`docs/interfaces/policy_observation_action.md`).

## Outputs

- Replay/evaluation procedure with Isaac-exact metrics.
- Cross-sim metrics and known mismatch list.
- Decision on which MuJoCo configuration is deploy-faithful.

## Related Directories

- `hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py` — the parity evaluator.
- `agi/a3_deploy_example/` — active deploy tree: `MUJOCO_VALIDATION_RUNBOOK.md`, `SIM_DEPLOY_REHEARSAL.md`, `SIM_FIDELITY_NOTE_FOR_AGI.md`.
- `agi/A3_MuJoCo_Sim/` — vendor AimRT MuJoCo sim (the explicit-PD subscriber lives here).
- `agi/code_deployment/a3_deploy_example/` — older vendor reference subset.

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/run_phase1_task_revision_0p5_exam.md](../operations/run_phase1_task_revision_0p5_exam.md)
- [../operations/run_phase1_signed_face_exam_k100.md](../operations/run_phase1_signed_face_exam_k100.md)
- [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)
- [../operations/run_gate3_first_tick_harness.md](../operations/run_gate3_first_tick_harness.md)

## Acceptance Criteria

- The same action ordering is verified in both simulators.
- The exported deploy ONNX (not a re-export) runs in MuJoCo with the training observation rebuilt exactly.
- Divergence sources are documented: contact, latency, actuator, timestep, observation delay, model mismatch.
- Exact-strike metrics from Isaac are reproduced in MuJoCo and recorded per accepted checkpoint.
- Before MuJoCo training starts, each explicit effective-plant profile reproduces a byte-frozen
  Python MuJoCo evaluator on a reset-first 179-D observation and a fixed short 31-D action tape.
  Per-term rewards must separately match an independent reward-replay oracle; the current evaluator
  has no training-reward API.
- The MuJoCo `VecEnv` completes deterministic reset, finite rollout, at least one PPO update,
  checkpoint resume and deploy export while recording measured throughput and a complete engine-bound
  training contract.
- Formal return training/scoring uses physical ball-racket-table/net contact and landing state;
  analytic virtual return remains diagnostic and cannot promote a policy.
- The first fine-tune paper is preregistered before launch: same source checkpoint, frozen control
  versus warm-start fine-tune, equal budget, multiple seeds and an immutable held-out
  [K100](../DEFINITIONS.md#q50-and-k100) (100 fixed questions, 50 per side) with
  per-side fall/hit/return. Final promotion still requires independent vendor Gate3/Gate3B.
- The selected 0.5-second vendor paper preserves all 100 question rows and order, starts every question
  from zero-velocity motion frame 0, and reaches contact at tick 25 of 50 Hz control. Each row must
  traverse the same production planner, C++ runner, vendor MJCF and effective plant, then emit
  attempt/completion/hit/return/fall/deadline fields without deleting failures.

## Current State

### 2026-07-20 W/Y export is diagnostic-only; lineage precedes the vendor adapter

Both real W/Y zero-write plans passed on exact `origin/main@a0c1284`, and fresh `179→31` ONNX artifacts passed
an independent structural check plus finite CPU ONNX Runtime inference. W's ONNX SHA-256 is
`ee0e2e83c8f3dc8302fcef609fe13b2feaf69e247e39f405d1ea6c30b652d970`; Y's is
`72da43d96ab9dd95e1da6aba2ed548ad26e61863b70cf8120c120132b7b8f995`.

The promotion gate nevertheless fails closed before vendor behavior: both checkpoints record
`training_contract_lineage_exact=0`, and both ONNX exports record `training_contract_exact=0`. They are useful
diagnostic artifacts, not deployable policies. The local prerequisite chain is:

1. remediate or retrain W/Y with exact checkpoint lineage and re-export an exact-contract ONNX;
2. implement the frozen K100-to-serve/`task_revision`/production-planner/C++-runner/vendor-MJCF adapter;
3. run and preserve all 100 vendor rows before comparing or promoting either candidate.

This is a dependency description, not a competing priority list; global execution order is owned only by
[`docs/NOW.md`](../NOW.md).

The native MuJoCo trainer remains valuable, but it follows this lineage-and-adapter chain rather than replacing
it. A separate processed-qdes action-slew matrix may run as a single-swing diagnostic; it cannot satisfy this
Gate or bypass the continuous-recovery order `T0 → T1 → T2`. G06 remains `Partial`, and `Gate3-D0` remains
`Open`.

### 2026-07-19 demo-priority vendor same-paper is preparation-only

A local read-only source audit found no runnable production path from the 0.5-second timing paper to
the vendor chain. The two demo-priority candidates are `W` (racket-position priority with the
non-striking arm free) and `Y` (racket-position priority with imitation muted in the strike window);
`U` (racket-position priority with a stronger ready pose) remains the stable fallback.

The completed Isaac K100 drives the policy directly and bypasses the production planner. The Python
`mujoco_eval_onnx.py` path supports 179-D observations and a fixed bank, but it does not consume the
per-question timing paper or retime every row to 25 control ticks. Gate3's fake-ball input accepts a
flat `N × 6` serve list (initial position plus velocity), and no adapter currently maps timing-paper
rows through serve generation, same-ball
[`task_revision`](../DEFINITIONS.md#planner-task-revision), the production planner and the vendor
runner. The old `pp_gate3_rally.sh` / `pp_rally_conductor.py` path remains quarantined and forbidden.

One exact, read-only filesystem-wide Pod1 search has now located one W and one Y `model_6700.pt`.
Both checkpoints load with embedded iteration `6700`, `74` floating tensors / `1,762,715` floating
elements / zero non-finite elements, and actor dimensions `179→31`. Each run also contains
`params/training_contract.json`, `env.pkl`, `agent.pkl`, and `env.yaml`. This closes only the static
training/export-input check; it is not vendor behavior or parity evidence.

The standalone exporter now has a genuinely zero-write `--plan`. It uses a weights-only checkpoint
load, requires a non-negative integer `checkpoint_iteration`, validates finite checkpoint materials,
the donor, motions, harvest, train bank, contract and formal face-179 envelope, and exits before the
first directory/temp/graph/artifact write. Its JSON reports `artifact_written=false`,
`graph_export_not_executed=true`, dimensions and formal-material status. The five-file focused suite
passes `97` tests in `0.38s`, including the unchanged normal-export fake smoke. Neither real W/Y plan
has run on a Pod and no ONNX artifact has been created.

The next runtime capability is the adapter described in
the acceptance criteria above, with the exact same 100 questions (50 per side), frame-0 zero
velocity, 25 ticks, forehand time scale `2.64`, backhand time scale `1.8`, and the same planner,
MuJoCo XML model (MJCF) and effective plant. Until that adapter and its per-question output exist,
W/Y are candidates rather than a successful demo. G05 and G06 remain `Partial`; `Gate3-D0` remains
`Open`. Detailed evidence and the frozen output contract are in the
[half-second sprint record](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md).

### 2026-07-17 exact-0.5 Isaac K100 remains upstream of MuJoCo parity

The first checkpoint-bound [0.5-second timing exam](../DEFINITIONS.md#timing-exam-0p5) launch failed
closed before evaluator creation because its immutable bank was absent; v1 is permanently consumed.
The bank/report were then restored with exact size/SHA and no-clobber permissions. The asset-restored
v2 supervisor binds harness `be17289c…cc59`, activation `2b91248b…0626`, the v1 failure receipt,
`taskrev_p2_equal_reward@model_5700`, all 100 exact-25-tick attempts and a fresh state/output namespace.
Its focused source suite is `41 passed, 1 skipped`; the skipped delegated-cgroup probe and the v2
RunPod behavior exam are still open.

G06 therefore remains `Partial`. Even a completed Isaac result is explicitly inexact; the same
checkpoint and immutable questions must still run in vendor MuJoCo before any parity or deployment claim. See the
[experiment](../experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md) and
[operation](../operations/run_phase1_task_revision_0p5_exam.md).

### 2026-07-15 analytic Reward is not the physical referee

The current VirtualBall task does use achieved racket FK state to analytically predict contact, net
crossing and landing, but those outcome terms remain a training model with dense partial credit. The
separate Isaac Phase-A engine-integrated ball diagnostic is metrics-only, and the live recipe leaves
racket impulse off; there is therefore no current physical-return reward or policy result. Before comparing analytic versus
physical outcome reward, Phase-B hit/net/landing events and all-serves denominators must close; Phase-B's
paddle impulse still reuses the analytic contact law. The same
actor/racket trajectory must be replayed against the Agibot vendor MuJoCo referee. Detailed exact-source
semantics are in [the Reward truth audit](../experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md).
No Gate3/Gate3B score changes, and G06 remains `Partial`.

Done (2026-06-27 → 2026-07-02, recorded 2026-07-03):

- The parity procedure exists and is battle-tested: `scripts/mujoco_eval_onnx.py` loads the exact
  exported deploy ONNX, reads the whole actuator contract from ONNX metadata (joint_names,
  default_joint_pos, action_scale, kp/kd, body_names — fails loudly if missing), auto-detects the
  175-D deploy-parity vs 180-D legacy obs contract, rebuilds the Isaac actor observation in MuJoCo
  (same frame math; the deploy-honest racket-target reframe is verified by
  `scripts/realsensor_obs_reference.py`), and reproduces Isaac's exact-strike metrics
  (pos/vel/normal pass, composite, hit-speed error, velocity attainment) with per-clip
  forehand/backhand breakdowns and per-step CSVs.
- Historical diagnostics implicated actuator PD integration: with the same ONNX and
  byte-identical `a3_pingpong.xml`, MuJoCo with `implicitfast` + kd in `dof_damping` was stable with
  clean swings, while the AGI deploy sim's
  explicit-Euler PD path (`joint_actuator_subscriber.cc`, MJCF without an integrator attribute,
  passive damping not zeroed) diverges within ~0.1 s. Switching only the PD integration moved
  hit-speed error 0.61 → 0.31 m/s and velocity attainment 0.35 → 0.88. This comparison did **not**
  prove Isaac equivalence because passive kd bypassed the total effort clip; the 2026-07-14 audit
  below revokes that exactness interpretation while preserving the numbers as diagnostics.
  Historical one-flag reproduction:
  `--pd-mode implicit` vs `--pd-mode explicit --keep-passive`. See
  `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md`.
- Historical verdict stance (2026-07-02): implicit PD was treated as the Isaac-faithful cross-check,
  but that label is superseded by the 2026-07-14 total-effort correction below. The
  binding pre-hardware gate is the AGI explicit clipped-PD MuJoCo run ("falls in MuJoCo = falls on
  the real robot"). The deployed policy was fine-tuned to survive it
  (`launch_explicitpd_ft.sh`, exported via `export_onnx_explicitpd.sh`).
- A deploy-faithful episode protocol exists: `--deploy-faithful` mirrors the C++ runner
  (nominal-stand start, windup hold with pinned time_to_strike, one full clip per swing, rest
  between swings, no teleports, absolute fall terminations only), reporting swing completion rates
  and time-to-fall.
- Eval mode B exists (2026-07-04): `--target-source venue-balls` (`mujoco_eval_onnx.py` +
  `scripts/venue_ball_sampler.py`) samples fitted venue incoming balls (with spin), StrikeSpec-
  inverts the demanded racket state (pos/vel/normal, sign-matched to the swing side's reference
  face), drives the unchanged target pipeline, and scores a virtual return at the exact-strike
  frame (capture gate → venue contact model → drag+Magnus flight → bounds + net clearance).
  Headline reported as `return_success_rate` per strike; mode-A (`boxes`) output stays
  byte-identical. First run: pos/vel tracking survives the OOD venue distribution (3.7 cm /
  0.18 m/s) but the face normal is clip-locked (36-76° err, 0% legal returns) — the 175-D
  contract has no normal channel (`docs/motion_and_contract_v3.md`). v1 caveats: uncorrelated
  box sampling, human-receiver contact heights (0.98-1.26 m vs trained 0.72-1.13 m —
  intentional realism, expect pos_pass to drop), incompatible with `--deploy-faithful`.
- The normal counterfactual is a committed output (2026-07-05; was an ad-hoc uncommitted
  analysis on 07-04): every venue strike is auto-rescored with the DEMANDED face normal swapped
  into the achieved kinematics — `cf_*` columns after the 14 venue columns + a CF summary
  block. Committed record (P2 product line, 9600 steps seed 0, 44 strikes): actual 0/44 vs
  counterfactual 44/44, CF median landing error 0.10 m; the 07-04 2400-step run reproduces
  byte-identically (first 43 CSV columns). The face-orientation channel alone fails the return.
- Fixed-normal inversion exists and delivered a verdict (2026-07-05): `--venue-fixed-normal`
  pins the StrikeSpec normal at the clip reference face (`solve_fixed_normal`, velocity-only
  LM; free `solve()` untouched; 16/16 planner tests). Result: the path-A ceiling is ~0% — a
  brute-force reachability scan (face pinned, all |v_r| ≤ 6 m/s, ~7k landings/ball) shows the
  forehand face ([0.41,0.90,-0.17], near-sideways) lands x ≤ 1.4 m at ANY racket velocity
  (never clears the net at 1.87 m) and the backhand face only reaches a net-hugging cross-court
  sliver (x≈1.9-2.0, |y|≈0.3-0.67) outside the legal landing box (≥0.3 m depth guard =
  training's own dink rule). Premise verified: mode-A achieved normal is within 1.9° of the
  clip reference, so the pinned face IS the policy's face. Planner adaptation cannot rescue the
  clip-locked face; the normal-channel contract change (175→179) is the only path.
  Evidence: pod `/workspace/franco/cf_eval/` (scan_reachability.py, modeB_*.log).
- A deploy-parity mid-swing switch stress protocol exists (2026-07-05): `--switch-stress P`
  (multiswing only; default off = byte-identical) aborts the swing each step with probability P
  exactly like the deploy runner's planner re-decides (training `clip_switch` semantics:
  uniform new clip, windup frame, fresh hold + target, robot untouched; tracking guards off —
  balance falls + timeout only). Reports switches, falls, 2 s post-switch survival, post-switch
  vs clean-swing hit rates. First matrix ({P2, R11} × {implicit, explicit+keep-passive} ×
  {~0, 0.002, 0.01}/step, 24000 steps each): zero falls in all 12 runs, 100% post-switch
  survival, post-switch hit rate ≈ clean — the switch discontinuity alone does not topple even
  the non-switch-trained P2 in MuJoCo; R11's in-distribution hit-rate tax remains visible on
  the explicit gate (0.98-0.99 vs P2's 0.99-1.00). Logs: pod `/workspace/franco/cf_eval/sw_*`.
- A documented validation flow with an acceptance-criteria table exists:
  `agi/a3_deploy_example/MUJOCO_VALIDATION_RUNBOOK.md` (rate ~50 Hz, sync stable, infer < 20 ms,
  projected gravity sanity, bounded actions, neck passive).

Not done:

- Formal per-checkpoint acceptance: the metric thresholds and the numbers for the currently shipped
  checkpoint (`model_p4_deployparity` / explicitpd_ft `model_25700`) are not yet pasted into this
  gate as an accepted record.
- (Fixed 2026-07-03, branch `audit-leftover-fixes`.) `eval_realsensor_hopex.sh` /
  `export_onnx_explicitpd.sh` now resolve their own location and take `HOPE_EVAL_*` /
  `HOPE_EXPORT_*` env overrides, and `mujoco_eval_onnx.py` resolves strike phases as CLI >
  ONNX `clip_strike_phases` metadata > built-in legacy `(0.36, 0.50)` (plus a
  `clip_seg_lengths`-vs-npz mismatch warning). The `--onnx`/`--motion-files` defaults still point
  at a legacy run — pass current artifacts explicitly.
- **Decision recorded 2026-07-12:** native MuJoCo training/fine-tuning is now a P0 implementation
  track; start with native CPU MuJoCo and measure the A3 workload before choosing an accelerated
  backend. The current code remains validation/dry-run only, and vendor Gate3/Gate3B remains an
  independent final arbiter.

## Risks

- A policy can appear valid in Isaac but fail in MuJoCo because of actuator/contact mismatch — this
  happened (explicit-PD divergence) and cost significant time before the root cause was isolated.
- Evaluating with the script's stale defaults silently tests the wrong contract; always pass the
  checkpoint's own clips/phases.

## Next Steps

1. Remediate W/Y checkpoint lineage and produce a fresh ONNX with both checkpoint lineage and exported
   training contract exact; keep the current inexact artifacts diagnostic-only.
2. Implement the frozen 100-row vendor adapter, then run the exact same paper through serve generation,
   same-ball `task_revision`, production planner, C++ runner, vendor MJCF and effective plant.
3. Implement the frozen evaluator semantics independently in a native MuJoCo `rsl_rl VecEnv`; keep
   trainer and evaluator imports separate, pass reset/action-tape parity plus an independent reward
   replay canary, then require one finite PPO smoke before any long run.
4. Preregister and run the same-checkpoint frozen-control versus warm-start-fine-tune multi-seed
   held-out K100 paper; do not let the training environment grade itself.
5. Record the accepted sim2sim numbers for the shipped checkpoint (implicit cross-check + explicit
   clipped-PD gate + `--deploy-faithful` protocol) in this gate.
6. When the mocap→planner bridge lands, extend the MuJoCo rehearsal to consume live
   `/racket/command` targets instead of sampled planner-equivalents
   (`docs/operations/run_shared_interface_rehearsal.md`).

## Audit update 2026-07-10: formal BankExam ruler

The old headline scores are not a trustworthy promotion ruler. The evaluator
had an exact-strike one-step offset, omitted pre-strike failures from its
denominator, compared different question slices across noise columns and did
not enforce the held-out split. These are now closed:

- one immutable schedule with stable question IDs and per-attempt seeds;
- all scheduled attempts remain in the denominator;
- every noise/model column receives the same ordered questions;
- train/exam split, motion SHA/order/frame and physics-source lineage are
  fail-closed;
- every formal attempt starts from the MJCF named `stand` keyframe with all
  hidden state and last action reset; teacher-reference reset is diagnostic;
- schedule, ready-state, MJCF and resolved execution-contract SHA are emitted
  in summaries and attempt CSVs;
- actuator integration, armature, ctrl/velocity limits and q-des contract come
  from schema-v3 rather than observation width guesses.

Non-zero PhysX joint friction has no exact MuJoCo `frictionloss` equivalent.
Formal BankExam therefore refuses it. `--allow-inexact-contract` may run a
direct-number proxy, but the result is stamped
`evaluation_contract_exact=false` and cannot be booked. Here `exact` means the
listed execution protocol is bound; it does not claim complete cross-engine
dynamics equivalence.

All key historical scores must be rerun after fresh export; retain old values
only with an explicit `old scorer` label.

The 2026-07-11 local Phase-1 snapshot also contained a NumPy
`virtual_return_scorer.py` and a saved-run `termination_contract.py`.  They
were initially retained as simulator-independent specifications.  The current
schema-v3 adapter branch now closes both production seams without modifying
the physics-hash-bound `venue_ball_sampler.py`:

- `mujoco_eval_onnx.py` delegates actual and counterfactual returns to the
  NumPy 10 ms RK4/ball-centre-plane scorer and binds scorer source, venue YAML,
  parameters and score spec into the execution contract;
- `bank_exam_schedule.py` materializes a balanced, canonical JSON paper with
  an exact per-clip quota, immutable content IDs, deterministic hold values and
  per-attempt noise seeds. Its hashed release rule defines `H` ready-stand
  actions followed by raw clip frame 0. MuJoCo accepts it with
  `--exam-schedule-json`; Isaac consumes the same artifact;
- `isaac_bank_exam.py` keeps the saved train bank untouched, installs one
  evaluator-owned exam row per environment after a nominal-stand reset, emits
  raw all-attempt JSON/CSV, and invalidates the whole cell on truncation.
  Exact cells additionally verify the runtime train-bank schema/family/SHA;
  historical legacy banks are allowed only in the explicit inexact canary lane
  and are recorded as an inexact reason.

Dependency-light verification on 2026-07-11 passed `67` adapter/audit tests
with one optional Torch parity skip, `85` formal CPU contract tests and `141`
unique tests in the combined contract run with the same optional skip. This is
implementation evidence, not a gate pass:
the shared-paper Pod canary and question-order/hash equality across both
simulators are still pending.  M3f/M2/G1 predate exact schema-3 checkpoint
binding, so their canary cells must say `evaluation_contract_exact=false`; only
a fresh exact-lineage model can produce a bookable score.

The M2 Isaac quota-10 leg has now passed runtime artifact validation: all 20
scheduled rows are present and uncensored, its bank/schedule SHA and ordered
IDs match the supplied paper, and its diagnostic return rate is 16/20. The
matching MuJoCo q1 leg initially stopped before rollout because the historical
`obs_norm.npz` has four zero std dimensions. They are valid constant features
under the saved `(obs-mean)/(std+eps)` implementation with `eps=1e-2`.
MuJoCo now accepts finite non-negative std only when every `std+eps` divisor is
strictly positive; negative/non-finite scales and unprotected zeros remain
fatal. A rerun is required, and cross-simulator canary status remains pending.

The next MuJoCo pre-rollout attempt exposed a main/rollout scope error in
`training_hold_protocol` and an avoidable dependency on the shell variable
`HOPE_STAGE1_QB`. One pure helper now derives hold-aware guard semantics in
both scopes. BankExam also resolves the current checkout's dependency-light
`stage1_question_bank.py` directly and records its SHA in the execution
contract, rather than importing the Isaac task package or trusting ambient
shell state. Both failures occurred before rollout and produced no score; the
same-paper rerun remains required.

That rerun and the full single-question diagnostic matrix are now complete.
At quota 10, M3f/M2/G1 MuJoCo return was 17/20, 10/20 and 9/20; G1 backhand was
0/10 in both engines. At quota 50, M3f returned 91/100 in MuJoCo versus 99/100
in Isaac, while M2 returned 51/100 versus 86/100. Both survivors also completed
the same-paper 5% action-noise and second evaluation-seed cells. Every ledger
was complete and uncensored, and every cross-engine bank/schedule SHA and
ordered ID check passed. All MuJoCo `fell` rows were tracking guards rather
than absolute physical falls. The detailed per-side table and result hashes
are in `docs/PHASE1_SCHEMA3_RESULTS_2026-07-11.md`.

MuJoCo carry-state BankExam remains a separate inexact diagnostic. Its summary
now includes `return_and_recover_rate`: among paper rows that have a scheduled
next opportunity, a row counts only if it legally returns and naturally
completes its swing. A post-strike guard preserves the return result but fails
recovery. The final paper row is excluded from this product denominator.

The completed q50 carry-state cells produced return-and-recover rates of
70/99 for M3f and 30/99 for M2. Overall returns were 82/100 and 40/100; no
absolute physical fall occurred, while tracking guards/timeouts remained
failed opportunities. Summary SHAs are `091bd045...0e6ea` and
`5658b7cc...b8774`. This is useful candidate ranking but remains an inexact
continuity diagnostic; Isaac continuous and a fresh exact-lineage policy are
still required for gate completion.

The historical main-matrix extension is also complete. At clean q50, R1b
seed 1/2 returned only 15/100 and 17/100 in MuJoCo (both 3/50 forehand),
despite 95/100 and 90/100 in Isaac, and stopped before robustness. C1 returned
50/100 in MuJoCo versus 96/100 in Isaac and advanced. Its MuJoCo noise and
second-schedule cells returned 48/100 and 55/100; its carry-state cell returned
42/100 and both returned+recovered on 26/99 next-opportunity rows. No C1 cell
had an absolute physical fall. M3f therefore remains the historical diagnostic
leader (`91/100` clean and `70/99` continuity product), while all of these
cells remain `evaluation_contract_exact=false`.

The formal friction gap now has a training-side, fail-loud control rather than
an undocumented source edit. Fresh runs may set
`task.plant.zero_joint_friction=true`; `train.py` then zeros every actuator
friction field before environment construction, and the existing schema-v3
runtime fact collector records the expanded zero vector. The checked-in
non-zero plant remains unchanged by default and is still diagnostic-only in
this gate. Override/contract unit tests passed `60` tests in an isolated Pod
worktree; the training entry also refuses to continue unless the instantiated
contract contains exactly 31 aligned zero coefficients. This does not complete
G06: a from-scratch schema-v3 checkpoint on
migrated schema-2 motion, a bound train bank, export, and exact BankExam are
still pending.

The export/judge replay path now preserves the two new runtime controls instead
of composing the default plant/layout after training. For a schema-3
checkpoint, `judge.sh` reads the adjacent hard contract: exactly 31 zero
friction coefficients restore `task.plant.zero_joint_friction=true`; the
declared non-zero default remains false; partial-zero, malformed, negative or
non-finite vectors fail closed. The same sidecar supplies the validated
175/179/181 actor contract and is cross-checked against saved face/station
flags. Face-command enabled state and pairing, legacy-motion permission and
motion exactness flow into ONNX metadata. Thus a legacy causal export remains
explicitly inexact while a future fresh zero-friction export can reach the
formal MuJoCo plant check without a compose mismatch. The dependency-light
contract/judge regression now passes `38` tests. No terminal fresh checkpoint
or exact BankExam result exists yet, so this gate remains `Partial`.

The 179-D exact-construction smoke now proves the export inputs can coexist in
one live contract: schema-2 runtime-order motion, schema-v3 bank, shared face
pairing and a 31-zero plant. Both fresh seeds wrote `model_0.pt` with schema-3
contract SHA `3a3b3d95...b9972` and embedded lineage exact `1`; the four causal
`model_17000.pt` files bind their own sidecars with lineage `0`. `judge.sh`
dry-run resolves the canonical adjacent exam banks and now adds
`--allow-inexact-contract` only for diagnostic motion/pairing contracts, while
fresh exact candidates receive no escape. It also resets `PYTHONPATH` from the
current checkout's `setup_train_env.sh`, preventing another user's Pod checkout
from supplying export code. Terminal export and same-paper Isaac/MuJoCo cells
are still pending, so G06 remains `Partial`.

The evaluation cadence is no longer terminal-only. Two checkpoint-curve
workers attempted the missing causal `17000/18000/19000` and fresh
`0/1000/2000` immutable BankExams. The first two attempts are preserved as
evaluator preflight failures (missing ignored A3 asset link, then buffered
export-success handshake despite an ONNX file), not booked model results. The
links now resolve only to the frozen training assets and the retry uses
unbuffered export output. A third preflight correctly reached sidecar creation
and exposed the known four constant observation dimensions. The sidecar writer
now preserves finite zero std only under its bound `eps=0.01` and still rejects
negative/non-finite or non-positive divisors; both Pods reproduced the same
four zeros with valid SHA-bound output. Each Pod serializes the Isaac export phase;
after an export reaches MuJoCo, CPU exams may overlap with
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Every result directory
is checkpoint-specific, so no old ONNX or normalizer is reused. The workers
run from the detached evaluator while both training checkouts remain clean at
`6d93bcb`. Diagnostic pairing/motion still receives the explicit inexact
escape; the `SZ` target cell may not. No curve result is booked yet, and G06
therefore remains `Partial`.

The inexact escape is now a one-way result downgrade in both simulators. An
exact-provenance fresh checkpoint evaluated with legacy face pairing (`LZ/LP`)
is allowed only as a diagnostic and must emit
`evaluation_contract_exact=false`; MuJoCo applies this when assembling the
bank contract, and Isaac records the pairing as an inexact reason before its
scorecard. `SP` remains a non-target plant ablation even when its bytes are
fully reproducible. This prevents the 2x2 diagnostic grid from laundering a
formal target label.

The next checkpoint preflights closed two more evaluator-only blockers. The Pod CPU venv contained
`onnxruntime` but not the `onnx` graph package required by formal inspection; both Pods now pin
`onnx==1.22.0`, and the generated 179-D graphs pass checker and runtime. Fresh exact models then
stopped before rollout because Isaac's float32 metadata representation of the same MJCF armature
decimals differed by at most `2.71e-9`, while the comparison threshold was `1e-10`. Passing that
field exposed the same representation issue at the `118.2` ankle effort limit: float32 metadata is
`118.199996948...` (`3.0517578e-6`, about 0.4 ULP). Formal plant comparison now requires exact
float32-grid identity rather than a field-specific tolerance and tests both sides of the 0.5-ULP
boundary plus next-grid rejection. A separate report fix propagates final artifact/escape exactness into the denominator
section, so a legacy causal report can no longer display `true` while its summary JSON says `false`.
These preserved attempts are not model scores. A corrected exact fresh BankExam is still required,
so G06 remains `Partial`.

Formal retry then proved why report code must not live in a physics-hashed module: changing only
`BankExamSampler.denominator_report()` changed the complete `venue_ball_sampler.py` SHA, and the
schema-v3 bank refused export before rollout. The sampler is restored byte-identically
(`00e28e85...30cc`), while final artifact exactness is now substituted by the outer MuJoCo
evaluator. Only the recorded judge PGIDs were terminated after Isaac's failed shutdown hung; no
training process was signalled. This retained attempt is not a score.

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_bank_exam_schedule.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py
```

The corrected float32-grid plant gate has now produced the first exact fresh
MuJoCo checkpoint curve. On the same clean q10 paper, `SZ` seed 1 scored
`0.00/0.50/0.90` and seed 2 `0.00/0.50/1.00` at `model_0/1000/2000`;
all six reports say `evaluation_contract_exact=true`. At 2000 the side splits
were FH/BH `0.80/1.00` and `1.00/1.00`. These are successful formal direction
screens, not q50 acceptance cells. The 20000 causal rows stayed explicitly
inexact: M3 old/S1 `0.45/1.00`, M2 old/S1 `0.50/0.50`.

The current single-question BankExam also does not certify real continuous
timing. Live training preserves state across natural clip wraps, but the
complete-clip schedule is materially slower than the conservative venue A-B-A
sample and installs the next target after the observed opponent-hit event.
The future continuity gate must use the same immutable question **and interval**
schedule in Isaac and MuJoCo, report per-opportunity carry-state failures, and
retain zero resets/teleports. The reproducible timing audit and required metrics
are in `docs/research/phase1_continuous_rally_timing_2026-07-11.md`. G06 remains
`Partial` pending q50, Isaac same-paper companion results, terminal lineage
verification, and event-driven continuity evaluation.

The causal terminal cadence no longer waits for an impossible filename. The
first normal M2-S1 completion proved that a continuation resumed at 16999 for
4000 updates finishes/saves at iteration 20998. Its terminal checkpoint is
finite and SHA-bound. The later paired terminal q10 judged M2-old/S1 at
`0.40/0.35` aggregate (both forehands zero), but remains an inexact,
non-decisive direction screen. Cadence and scale-out causal manifests now
target `model_20998.pt`; the exact waiting-worker PGIDs were replaced without
signalling trainers or fresh workers. This changes only checkpoint discovery,
not the immutable exam or causal `evaluation_contract_exact=false` rule.

Cross-engine exactness for `SZ` is deliberately narrow. All-zero friction is
byte/semantics reproducible, but prior frozen-plant evidence says it is not a
safe proxy for the deployment plant. Conversely, current `SP/LP` non-zero
coefficients cannot be made exact by feeding the same numbers into MuJoCo
`frictionloss`, because the physical meanings differ. G06 therefore has no
deployment-qualified plant cell yet. Closure requires a measured, versioned
friction model with engine-specific adapters, a fresh `SC` training cell and
the full train-plant x eval-plant transfer matrix; until then, `SZ` scores can
validate the evaluation contract but cannot clear sim-to-real parity.
`SP` is consequently an inexact diagnostic, with an explicit evaluator escape;
it cannot be booked and cannot block the later formal SZ jobs in the same
milestone-major queue.

The offline plant-contract v1 boundary now implements the fail-closed half of
that closure plan. It refuses non-zero cross-unit numeric conversion, requires
one content-addressed latent model plus independent PhysX/MuJoCo fit and probe
reports, checks the canonical 31-joint order and rejects requested runtime
envelopes outside calibrated load/speed/temperature/pose support. Crucially,
the final MuJoCo leg is not a generic standalone evaluator: it must bind the
Agibot vendor `a3_pingpong.xml`, the Gate3/Gate3B runtime source and a raw
31-joint adapter-instantiation report. Current BankExam remains useful
development/selection evidence but cannot substitute for that vendor-runtime
cell. No calibration bytes, passed runtime probe, vendor instantiation report
or fresh `SC` checkpoint exists; the compiler is not wired to either engine,
so G06 remains `Partial`.

Original causal terminal and original fresh exams now run in separate
workers/state directories, so neither checkpoint-availability order can block
the other. Q10 manifests declare the screen-only/no-promotion policy at both
manifest and job level, and the checked-in `phase1_checkpoint_curve_worker.py`
rejects omissions or contradictions, checks the schedule, and requires the
same canonical screen-policy-plus-job contract SHA before reusing a completed
state. This is an
operational guard, not permission to
book q10; q50 and the same-paper Isaac/MuJoCo pair remain the decision gate.
Pod1 fresh starts at 4000 because that checkpoint had not existed when the old
combined worker was replaced; Pod2 4000 was already handled and starts at 6000.

The first corrected terminal MuJoCo q10 pair is preserved at M2-old/S1
`0.40/0.35` aggregate (FH both `0/10`; BH `8/10`/`7/10`). Both are inexact
diagnostics and the prefix is too small to decide. Full result/checkpoint/report
hashes are tracked in `configs/phase1_M2_terminal_q10_pair_20260711.json`;
neither cell advances or stops without q50 and its Isaac companion.

The matching Pod1 M3 terminal pair is now also complete and finite. M3-old's
`model_20998.pt` has SHA `320b77c9...417a`, matches adjacent contract
`7542c59b...d941b`, and carries causal/inexact lineage. On immutable schedule
`7a908142...d614`, M3-old returned FH/BH/aggregate
`0.50/0.40/0.45`, while M3-S1 returned `1.00/1.00/1.00`; paired aggregate
delta is `+0.55`. This triggered the separately frozen K=100 q50 paper. On
that shared 50-per-side schedule, M3-old returned FH/BH/aggregate
`0.62/0.22/0.42`; the raw ledger has one physical fall plus eight guard resets
(the legacy summary's `fell=9` is their union), while M3-S1 returned
`1.00/1.00/1.00` with zero such terminations. Aggregate delta is `+0.58`, so M3-S1 wins
the MuJoCo terminal selection inside this same legacy swing-family causal
paper. Both results remain `evaluation_contract_exact=false`. The same-paper
Isaac companion then scored both cells `0.98/1.00/0.99` FH/BH/aggregate,
delta zero, on the identical question order. It does not reproduce the MuJoCo
ranking, so cross-engine selection, continuity and calibrated plant remain
open. Full terminal and paired bindings are in
`configs/phase1_M3_old_terminal_audit_20260711.json` and
`configs/phase1_M3_terminal_q10_pair_20260711.json`; q50 execution/result
hashes are in `configs/phase1_M3_terminal_q50_result_20260711.json`; the Isaac
ledger is `configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

The four newly refilled causal workers have also been corrected from eval
`46a0ce2`'s legacy state schema without changing their judge paper. Only the
four exact, childless legacy worker PGIDs were TERM-signalled; hardened PGIDs
are Pod1 `1416771/1416784` and Pod2 `198759/198771`. Each rejudged 17k state
returned zero and now binds manifest, job spec, job contract, checkpoint,
judge and both clean commits. Old state/log bytes remain immutable beside a
content-addressed correction sidecar. This closes provenance for future
milestones but does not change their causal `evaluation_contract_exact=false`
status.

The six older original/scale-out workers were independently hardened as well.
Current PGIDs are Pod1 `1432280/1432292/1432304` and Pod2
`200706/200718/200730`; no trainer or judge was signalled. Five available old
states were rejudged rc=0 and now bind manifest, job and job-contract SHA.
`configs/phase1_global_curve_worker_hardening_result_20260711.json` preserves
the exact signal scope and all transaction hashes.

Fresh SZ seed1 also closed its first exact checkpoint-selection q50. On one
K=100, 50-per-side paper, the analytic virtual-return scorer gave model 2000 FH/BH/aggregate
`0.66/1.00/0.83`, while model 4000 returned `0.00/1.00/0.50`; model 2000 is
retained. The whole arm continued at that paper's decision time and was only later stopped by the
separate 2026-07-13 operational resource decision. Both evaluations are exact/fresh, but
all attempts finalized through a non-physical post-strike guard, so this is
not a continuous or deploy-stability gate. The result is bound in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`. Its fresh
same-paper Isaac companion gave both checkpoints `0.98/1.00/0.99`
FH/BH/aggregate, delta zero. The MuJoCo ranking is therefore not reproduced;
the cross-engine checkpoint gate stays open. Companion hashes are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

Question-level forensics localize reproducible state differences but do not yet establish their
causes. Fresh model 4000's mean FH racket-center error is `13.15 cm` in
MuJoCo, beyond the frozen `9.5 cm` analytic contact margin on all 50 questions, versus
`2.48 cm` in Isaac; model 2000 is `9.03/3.03 cm`. M3-old BH has mean signed
normal error `168.15 deg`, and the later face-sign audit shows that the analytic
`VirtualReturnScorer` path may erase `n/-n` through `orient_normal`.
The earlier wording “MuJoCo physical outcome” was wrong: this evaluator has no simulated ball
contact and its incoming ball is visual-only. Both reported return cells are analytic outcomes
derived from racket state. Thus same question bytes/order are necessary but not sufficient, while
the exact engine/trajectory/scorer contribution remains unresolved. The forensic result is bound in
`configs/phase1_cross_engine_saturation_forensic_result_20260711.json`.

The next gate is preregistered as a strict 2x2: Isaac/MuJoCo x physical
truth/analytic counterfactual, with the original K100 order and capture/speed
thresholds frozen. Missing/duplicate/non-finite cells, changed order, or a
virtual-only physical cell all fail closed. Numeric Isaac ready/base/racket,
signed-face-before-orient and analytic state instrumentation is implemented.
Isaac PhysicalBall Phase-B source implementation exists at `612f54d`, but it has no accepted Pod
runtime, post-contact K100 ledger or content-addressed four-cell evidence manifest. Until those
runtime cells exist, G06 remains `Partial`.

Run `python3 scripts/validate_phase1_queue_governance.py` before any curve
manifest is copied or launched. The validator enforces the 142-job/24-slot
q10 screen contract and refuses q50 through the generic worker. Plant parity
remains separate: SZ is only zero-friction protocol exact, while SP/LP are
historical direct-number proxies. The repair contract is
`docs/research/phase1_plant_semantics_repair_2026-07-11.md`, status
`blocked_on_calibration_evidence`.

The v1 plant preregistration's source snapshot is now explicitly historical at
`d4ca566`; current strict-face179 `training_contract.py` bytes differ and the
current-checkout verifier fails closed. No current Gate3/SC result may consume
that stale snapshot. Re-preregistration of the complete current training,
adapter, judge and vendor-runtime closure is required before this plant leg can
advance; G06 remains `Partial`.

### 2026-07-12 final-engine priority

The final behavioral arbiter is the Agibot-provided A3 MuJoCo deploy chain called Gate 3 in
`docs/operations/run_pingpong_end_to_end.md`: fake ball -> real planner -> production-equivalent
C++ runner -> vendor MuJoCo. Isaac remains a fast training/diagnostic engine and native MuJoCo
training/fine-tuning is now P0; a win inside either training engine cannot promote a checkpoint that
fails Gate 3 balance, completion or recovery.
Continuous candidates must run without between-serve simulation reset and eventually satisfy
zero falls, zero operator rescues and complete recovery after every engaged swing. Gate 3B adds
the immutable stage distribution and hit/return scoring, but it does not weaken Gate 3 stability.

The current Isaac/MuJoCo gap is an open causal problem, not evaluator noise to average away.
The preregistered engine x physical/analytic 2x2 now has its Isaac PhysicalBall Phase-B source
mechanism, but the runtime gate remains closed until one clean-detached K100 ledger and
moving-blade substep audit exist. Plant semantics, ready-state, termination, observation/action
runtime and signed racket-face measurements must remain separately bound so a score difference
can be localized rather than hidden in one aggregate return rate.

The exact model-2000 SZ paper now adds a separate transfer warning: MuJoCo K100 return across
seeds 1/2/3/4 is `.83/1.00/1.00/.20`. This is not a plant fall signature (all physical-fall
counts are zero); it is checkpoint/learning seed instability on the current single-strike paper.
It blocks a stable Phase-1 checkpoint baseline before Gate 3. Do not average away seed 4, and do
not attribute the variance to Isaac/MuJoCo until the same checkpoints have the registered physical
instrument cells.

### 2026-07-12 MuJoCo training/fine-tuning P0

The project has promoted native MuJoCo training/fine-tuning from undecided/evaluation-only to P0.
This responds to repeated evidence that Isaac training metrics can stay high while held-out MuJoCo
strike execution and analytic return degrade. Matched physical-fall counts are zero, so balance
degradation is not established. The decision does not make the training environment the final judge.

The first backend must independently implement the frozen meanings from
`scripts/mujoco_eval_onnx.py`, must wrap batched native MuJoCo state as an `rsl_rl VecEnv`, and must load the
vendor A3 MJCF while bypassing the single-world real-time AimRT/ROS/GUI loop. It must not import a
shared observation/action/reward implementation from the evaluator, because shared mistakes would
create a common-mode false green. Its training contract binds engine/version, MJCF plus mesh
closure, resolved plant/PD/integrator/dt, runtime action post-processing, observation/action,
reward/termination/reset, question bank and source checkpoint hashes.

The preflight identifies two non-equivalent profiles. `isaac_bank_parity_v1` reproduces the current
schema-3 BankExam/Isaac profile; `vendor_gate3_v1` preserves the resolved 1 ms vendor plant, explicit
per-step PD, hard joint limits, neck override and frozen runtime flags. Loading the same source MJCF
does not make them equal because BankExam mutates the in-memory model. Reset/observation/action-tape
parity is judged against the frozen Python evaluator for the named profile. Per-term reward is
judged by a separate replay oracle: the evaluator records metrics and analytic virtual return but
does not implement PPO training rewards, and no independent C reward evaluator was found.

The first causal paper uses one exact source checkpoint: frozen control versus actor warm-start
fine-tune, fresh critic/optimizer, equal budget, at least two training seeds and an immutable held-out
K100. Formal return learning/scoring requires physical ball-racket-table/net contact and landing;
analytic virtual return remains diagnostic. A future MJX/MJWarp path is throughput work with its own
parity burden, not an exact-vendor label. Final promotion remains the unchanged vendor Gate3/Gate3B.

The tracked vendor MJCF currently has no ball, table or net and the existing analytic `BallPhysics`
driver is not wired into `MujocoSimModule::SimLoop()`. The 2--3 day `Trainer-v0` is therefore explicitly a
one-shot balance/strike-state fine-tune, not a physical-return or continuous-rally claim. Actor warm
start must load only actor/distribution/actor-normalizer state into a newly initialized critic and
optimizer; current `load_actor_tolerant()` does not enforce that boundary.
The separately tracked `mujoco-ball-wiring@4607410` handoff is not merged into this audited main and
has not passed its vendor build/runtime acceptance; formal physical-return work cannot consume it
until that independent gate closes.

No MuJoCo `VecEnv`, PPO smoke, training run or result exists yet. This decision adds an implementation
and acceptance path but does not close the engine gap; G06 remains `Partial`. The audited file
boundary, canary tapes, evaluator isolation and `Trainer-v0` sequence are recorded in
[the MuJoCo training-v0 preflight](../research/mujoco_training_v0_preflight_2026-07-12.md).

### Gate 3 face-command wire and engine-gap localization

The 179-D Phase-1 policies cannot be tested by adding `179` to a shape whitelist. Their last four
columns require the actor's raw mount-A world-frame normal and a zero rho placeholder atomically
paired with position/velocity. The external planner wire deliberately carries the physical,
opponent-facing striking face B instead. A versioned flat schema-2 publisher/receiver and exact
`deploy_parity_face179` ONNX metadata path are now implemented in source. The loader additionally
requires `face_command_enabled=1`, `shared_plus_y`, `mount_plusY_A`, an exact schema-3 train bank,
train split and lowercase content/source-family SHA-256 bindings; width and term names alone are
not enough. Schema-2 rows require a world-frame opponent-facing unit B normal (`B.x>1e-6`) and zero
rho. After clip selection the runner applies exact `[+1,-1]` to the normal only to recover raw A;
position and velocity are unchanged. Any malformed/unknown row after an active face tuple records
`invalid_after`; the publisher turns a bad solve or payload into an explicit finite `valid=0` row
on both wires, so silence cannot
keep an old swing eligible for the longer command timeout. Schema 1 remains the default for
existing models and cannot engage a 179 actor. This is not yet a gate result: the
vendor-source offline x86 build is recorded below, while a ROS/AimRT-enabled build, no-publish
first-tick parity trace, and full Gate 3 MuJoCo run are pending.
Active-swing fields are atomic. Post-swing recovery is not yet exact: the current Gate 3 runner
combines a synthesized base-anchored hold position with the previous swing's velocity/normal,
and no Phase-1 contract proves that hybrid tuple is on-distribution. A canonical recovery tuple
or separately accepted vendor-MuJoCo recovery paper is required before continuous promotion.
The physical-B positive-X invariant is also only a minimum sign/frame guard. Source now exports
and enforces a content-bound per-clip normal envelope from the exact training bank, as described below. A new
envelope-bearing formal ONNX, self-hit evidence and vendor behavior gate are still absent, so this
source guard must not be promoted into a Gate 3 result.

The same-policy Isaac/MuJoCo gap is localized in stages rather than one aggregate score:

1. replay identical joint/racket trajectories kinematically to isolate geometry, frames and scorer;
2. replay identical open-loop actions from a bound initial state to expose actuator/plant/integrator drift;
3. run closed loop with identical externally supplied observation rows to isolate policy/runtime timing;
4. only then compare each engine's native observation and physical contact in the full closed loop.

Each stage binds joint order, action scale/clamp, PD, dt/decimation, initial/ready state, signed face,
contact/termination and vendor MJCF SHA. Gate 3/Gate 3B is the final behavioral leg; Isaac remains a
training/diagnostic leg even if its score is higher.

#### 2026-07-11 isolated vendor-source build evidence

Source commit `8d56ea86f6450c198836969360bc133146934617` was archived into the isolated
Pod1 path `/workspace/codexschema/gate3_face179_8d56ea8`; neither the live training checkout nor
the eval checkout was changed. The local ONNX Runtime 1.19.2 archive used by the build has SHA-256
`eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33`. A Release configure with
ROS messages and AimRT disabled built both `run_tests` and the actual
`a3_deploy_onnx_ref_pingpong` executable. Focused `PpPlannerInput.*:PpFace179Wire.*` was 10/10;
the full native suite was 195 passed / 4 skipped (only absent optional fixture/asset tests).
The test binary SHA-256 was
`1349038f5a3bd057026630f1fdcc9636cf68d5acef1041712911e2808140a1fe`; all 78 compile commands
contained the finite-safety flags and none contained `-ffast-math` or `-ffinite-math-only`.

This closes the offline vendor-source compile/test leg only. It does not exercise ROS/AimRT,
load a formal 179 ONNX, tick the production backend, instantiate the vendor MuJoCo, or score a
ball. Therefore G06 remains Partial and Gate 3/Gate 3B remains open.

The next matched fresh checkpoint paper is also preregistered at model 4000,
but it does not weaken this cross-engine gate. It reuses the **same K100 file
bytes**, semantic schedule, question order, exact-family bank and 2k stability
thresholds for all four `SZ` seeds. The offline queue cannot invoke a judge;
it can only combine two read-only Pod checkpoint audits after all four
`model_4000.pt` files are finite, embed iter 4000, bind the same adjacent
schema-3 hard-contract SHA and retain exact fresh lineage. A future runner must
consume the content-addressed activation artifact and still bind the current
MuJoCo evaluator. Source verification is `20 passed`; no Pod/runtime action has
occurred.

This is seed/checkpoint evidence, not an engine-parity result. Known seed1 4k
already returns only `.50` on this MuJoCo paper and scores `.99` in the analytic
Isaac companion, so the four-seed stability gate cannot pass and the existing
instrument disagreement remains. Seed4 at 4k can support “delayed learning”
only against the unchanged `.65` aggregate/`.50` each-side thresholds; it
cannot close family stability, physical Isaac truth, calibrated plant, or the
Agibot vendor MuJoCo Gate3/Gate3B final gate. The frozen paper and barrier are
documented in
`docs/operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md`; G06
remains `Partial`.

### Production-root-bound stand viewer is diagnostic, not Gate3 (2026-07-12)

`yikang-linux-port-0711@6b10998` supplied a useful plain-MuJoCo stand viewer, but the original
version retyped gains and assigned head gains `40/2` while describing them as production PD_STAND.
The tracked production header instead defines a 29-DOF policy view and explicitly leaves the two
neck slots passive. The selective port now parses the production pose/Kp/Kd arrays from
`a3_policy_parameters.hpp` at runtime, requires every 29-DOF joint exactly once, leaves head yaw/
pitch passive, and never modifies the vendor MJCF or integrator.

Dependency-light `--identity-only` binds vendor `a3_pingpong.xml=2ab1cd31...3feb97` and the
production parameter header `df73e3f6...c5c8d8`; four host-only parser/identity tests and pycompile
pass. Those are root-source identities only; the MJCF's 74 transitive mesh assets are not yet an
asset-closure hash and remain part of the formal runtime binding. With MuJoCo installed, `--check`
records actual timestep/integrator; finite status for
qpos/qvel/qacc/ctrl/actuator-force arrays; pelvis-z range/drift, maximum pelvis tilt and per-foot/
both-foot floor-contact fractions. Its default
thresholds are diagnostic tripwires, not Gate3 acceptance. The current Mac lacks the MuJoCo binding,
so no 10-second numerical result, snapshot or plant claim is recorded.

This tool starts no planner, policy, AimRT/backend, first tick, Gate3/Gate3B or hardware. It can
localize a base MJCF/PD/integrator problem before policy evaluation, but it cannot promote a policy
or explain the same-policy Isaac/MuJoCo gap alone. Instructions are in
`docs/operations/run_deploy_dryrun.md`; selective-port evidence is in
`docs/research/yikang_selective_integration_20260712.md`. G06 remains `Partial`.

The production runner now also has a fail-closed `--model-preflight-only` path. It requires
`--no-publish` or `--dry-run`, constructs `PpPolicy` before any backend object is created, and on
success emits parsed `publishable_model_contract=true`, `training_contract_exact=1`, the accepted
observation width, and training-contract/source-checkpoint SHA-256 before exiting. No-publish is a
transport/runtime diagnostic state, not permission to relax model metadata. This safely separates
“the formal 179 export is loadable under the production
metadata contract” from “the vendor backend has started.” The former still requires an isolated
binary run with the formal candidate; the latter, first actor tick, normal-envelope/recovery
contracts and Gate 3/Gate 3B behavior all remain open.

The first full-dependency probe found a common loader defect before any score was produced:
`PpOnnxPolicy` chained `GetInputTypeInfo(...).GetTensorTypeAndShapeInfo()` through a temporary
owner. The borrowed tensor-info handle was already dangling when its shape was read and real 175-
and 179-D models could throw `length_error`/`bad_alloc`. The source now retains both input
`TypeInfo` owners through all shape/type reads and adds an optional real-ONNX regression. This
finding invalidates the failed loader attempt, not the model; isolated Release rebuild plus the
formal-ONNX test and production preflight are required before marking the repair verified.

#### 2026-07-11 formal 179 production-loader gate

The repair and model-only preflight are now verified in a second isolated archive,
`/workspace/codexschema/gate3_face179_a82eba6`, from exact source
`a82eba6c7dbfad0c6750b2ca5684f3f2f7b6ea6e` (tree `7d0452ea...354a`, archive SHA
`7553dde0...c58`). The configure enabled both ROS messages and the AimRT backend; Release built
`run_tests`, `a3_deploy_onnx_ref_pingpong` and `a3_policy_runtime_probe`. Their binary SHAs were
`0aef44d2...3440c`, `1f0e13de...20cc` and `8cf9b300...36e0`. The formal SZ seed2 model-2000 ONNX
was copied read-only into the archive and retained SHA `350b51cc...34cc2`.

With `A3_PP_ONNX_PATH` bound to that model, the lifetime regression passed 1/1; the full suite was
205 pass, 9 optional-asset skips and 0 failures (214 total). Without no-publish,
`--model-preflight-only` exited 2 before model/backend initialization. With
`--planner --no-publish --model-preflight-only`, it exited 0 and printed
`backend_not_initialized=true`, `obs_dim=179`, training contract `3a3b3d95...b9972` and source
checkpoint `d920...5e22`. Both stdout/stderr searches found no `backend cfg`, backend initialized or
backend started line. Accepted/preflight and full-suite logs have SHAs `2962d653...b5f4` and
`eb15d603...f64e`.

The direct-CMake executable needed its build-tree TBB directory on `LD_LIBRARY_PATH`; the packaged
runner stages TBB. `PpPolicy` construction performs one intended zero-observation ONNX prewarm
inference, but no policy driver, backend tick, transport, simulator, Kit or command path started.
The live training/eval checkouts remained clean at `6d93bcb...`/`46a0ce2...`, and no isolated
process remained. This closes the pre-envelope formal-model production loading proof only. The
same ONNX is intentionally rejected by the stricter source below because it lacks the new envelope
metadata. Re-export/rebuild, first backend tick, canonical recovery tuple and full vendor MuJoCo
Gate 3/Gate 3B behavior remain open, so G06 stays Partial.

Red-team follow-up downgrades the 2026-07-11 loader proof to lifetime/backend-order evidence only:
at that source revision `diagnostic_no_publish` was also passed as the loader's legacy-contract
escape, and the optional real-model test explicitly enabled it. The inspected model happened to
declare exact lineage, but the run did not prove that no-publish and live-publish enforced the same
parsed contract. The stricter source below removes that coupling; a new envelope-bearing model and
rebuilt production binary must rerun the proof before publishable-model loading is closed again.

### Recovery tuple and named-ready mismatch are now explicit Gate3 blockers (2026-07-12)

The recovery A/B/C preregistration binds the read-only Gate3 policy blob at commit
`1d46ef2cbb915efc135251f9b32f4ec25d0342ab`, SHA `8c9814c...0eba4`, and rejects its current idle
179-D tuple as a formal train/deploy match: idle position is newly anchored to the live base while
velocity and face normal/rho remain from the previous strike. Training produces only an all-old
tuple before reveal or an atomically installed all-new tuple. The same runner also zeroes the
actor's last-action observation during static-stand handoff; that intervention is not T1's
carry-state contract and must be replaced or explicitly isolated before a no-reset score is valid.

A second static audit prevents the word `stand` from hiding a different initial state. The bound
Isaac reset pelvis is `(0,0,1.0684) m`; the vendor MJCF stand key is
`(-0.0416378,0.000359049,1.06839) m` with approximate roll/pitch/yaw
`(-0.030,0.249,0.042) deg`. The full 31-joint vectors differ by `0.171845 rad` L2, dominated by
head-yaw `-0.169416 rad`; excluding the head still leaves `0.028789 rad`. Stage-1 contact positions
are environment-origin absolute and the 179-D actor observes target minus current racket FK, so the
`4.16 cm` root-x offset does not automatically cancel. These numbers define a causal hypothesis,
not a proven root cause of the Isaac/vendor discrepancy. Formal A/B/C evaluation therefore blocks
until one content-addressed numeric contract binds the exact ready base, joint vector, racket FK,
target position/velocity/normal/rho and observation result in both engines.

All arms must consume the same immutable random-arrival rows, question order and deadlines, with no
physical reset, teleport, last-action/history/noise reset, or replacement of infeasible rows. q10
remains directional only; q50 is the decision paper. The final MuJoCo path has two distinct gates:

1. Gate3 is a hard runtime prerequisite. It binds the exact C++ runner, vendor MJCF, calibrated
   plant and model, and must pass first-tick parity plus continuous stability.
2. Gate3B may run only after Gate3 and must reuse the same runtime contract. It consumes the
   immutable random-arrival q50 schedule and is the final behavior arbiter for first-strike
   non-regression and return quality.

Isaac remains the development/cross-engine precheck, not the final behavior vote. A discrepancy is
blocked and root-caused, never averaged. The design-only validator is green (`50 passed`, prereg SHA
`ca7806df...d810616`), while launch remains intentionally blocked on separate Gate3 runtime/
stability and Gate3B scoring judges, their shared runtime contract, exact A policy-ownership/PPO
accounting, calibrated plant and safety bindings. See
`docs/operations/run_phase1_recovery_tuple_prereg.md`; G06 remains `Partial`.

The 2026-07-13 primary-source audit adds no runtime credit. ACE's near-time-optimal reset MPC is
evidence for an interruptible bridge/prepare architecture, not for free-standing humanoid balance;
HITTER samples the next task after swing completion; SMASH uses strike-centred recovery clips and
cyclic phase but does not publish a mid-followthrough random-reveal comparison; PACE's five-serve
episode is likewise not that treatment. Consequently G06 defines `T0` as cycle-bound install,
`T1` as event-driven structure with frozen rewards, and `T2` as a later learned-shaping increment.
Random arrival is first an immutable environment axis. Balance/ready shaping starts with paired
`2^2`; a third readiness potential and `2^3` require separate critic train/calibration splits and a
one-shot preregistered critic-gate q50 disjoint from sealed formal Gate3B q50, without hidden-future
leakage. Any self-hit, reset/teleport/history clear, deadline shift/censoring, per-transition-cell
collapse, fifth-and-later opportunity decay, one-shot regression or Isaac/vendor direction reversal
fails promotion. Full sources and DOE boundaries are in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`. No paper substitutes for the exact
Gate3/Gate3B runtime result, so G06 remains `Partial`.

上述 2026-07-13 reward 顺序只完成文档设计收紧。当前 machine prereg/validator 仍固定旧三项
reward 和 full `2^3`；因此它们不能作为新顺序的机制证据，也不能用于 launch。需要新的内容寻址
config、validator、测试和 operation 对账后才可解除这一阻塞，G06 仍为 `Partial`。

### 2026-07-12 Gate3 first-tick static plan gate (red-team corrected)

The historical `pp_gate3_rally.sh` launch command is no longer an approved formal launcher.
Content-bound audit `configs/gate3_legacy_process_audit_20260712.json` records 14 concrete risks:
eleven fuzzy `pkill -9` calls, conductor `pgrep -f` SIGSTOP/SIGCONT, no PID/PGID/starttime/token
ledger or trap, hard-coded unbound workspaces, inherited ROS graph, destructive fixed `/tmp` and
shared-memory cleanup, no formal-loader-first gate, publish-capable free-form runner args, a boot
loop that proceeds after timeout, partial direct-PID cleanup, and no concurrency lock. The old
scripts remain historical result provenance; do not invoke their cleanup to make a new run pass.

Red-team review rejected feature commit `1fc69d1` as mergeable runtime shape: it carried an armed
future supervisor before dependency closure or a safe startup handshake existed. The corrected
`scripts/run_gate3_first_tick_harness.py` is **plan-only**. It has no runtime option/arming phrase,
direct process launcher, signal path, process scan, runtime lock or trace consumer. Old
`--mode run`/arming arguments fail in argparse before any contract/Git work. Its only child commands
are read-only Git queries with `GIT_OPTIONAL_LOCKS=0`; therefore “starts no process” is too broad,
but it starts no sim/Kit/transport/planner/runner and sends no signal.

Schema-2 validation binds core absolute path+SHA pairs, but does not call that set an exact runtime
closure. Every path must equal its resolved spelling and every component is checked with `lstat`,
so symlink ancestors fail. Training/eval paths must be clean exact-commit Git top-levels. Proposed
argv arrays are fixed and passive/no-publish; `--flag=/abs`, unbound absolute paths, relative
payloads and extra flags fail. The optional plan output uses fsynced temporary bytes plus atomic
hard-link create and directory fsync; it never uses overwrite-capable `os.replace`. It is rejected
under the recorded source/train/eval worktrees or any Git dir/common dir, then all three clean Git
identities are revalidated before an external write. The ledger's runtime block is permanently
`not_run`, with no components, signals, lock, behavior result or ownership token. Source tests pass
`32` cases; no runtime was launched.

The plan explicitly keeps five runtime blockers null: the production C++ full
`--first-tick-json` needs verified same-sample runtime output (the source diagnostic below is
structurally inexact and has not run); exact process ownership still needs pidfd plus a cgroup/reviewed supervisor
startup handshake; PATH/LD/Python/AMENT directory manifests and AimRT/transitive `.so`/plugin
closure are absent; separate vendor config/MJCF hashes do not prove parser-resolved semantics; and
the atomic runtime ledger/exact lock transaction is undesigned. String containment in a config is
not accepted as MJCF binding. Filling or deleting a blocker invalidates the static contract. A
separate reviewed runtime implementation must close all five; this source never becomes runtime
eligible by changing a flag.

The ledger also freezes a ready-state hypothesis without turning it into a result. Fresh training
starts at pelvis `(0,0,1.0684)` plus default q; vendor `stand` is
`(-0.0416378,0.000359,1.06839)` with about `(-0.030,0.249,0.042) deg` rpy. Mapped joint L2 is
`0.171845 rad`, dominated by head-yaw `-0.169416 rad`; excluding the head still leaves
`0.028789 rad`. Because Stage-1 bank contact positions are env-origin absolute while 175/179 target
position is relative to current racket FK, the `-4.16 cm` root-x shift need not cancel. It may
contribute to the engine gap, but is not yet causal evidence. The preregistered same-K100
vendor/root-only/joints-only/full-match four-cell diagnostic remains inexact and unrun; the formal
vendor stand is unchanged.

Every plan records the four-stage engine-gap ladder as not run with no inference authority:
kinematic replay, open-loop action replay, external-observation closed loop, then native closed
loop. Isaac stays training/diagnostic-only. A future first tick would close only a runtime
prerequisite; only Agibot vendor MuJoCo Gate3/Gate3B behavior can promote a checkpoint. Full static
operation and remaining blockers are in `docs/operations/run_gate3_first_tick_harness.md`. G06
remains `Partial`.

#### 2026-07-12 content-bound per-clip demanded-normal source gate

Formal 179 export and loading now bind the raw-A normal distribution rather than accepting every
opponent-facing physical-B unit vector. The train NPZ must be exact schema 3, split `train`, ordered
`forehand,backhand`, `shared_plus_y` and `mount_plusY_A`, with its bytes and source family matching
the checkpoint contract. The contract and both exporters must carry the exact
`mount_normal_sign_per_clip=[+1,-1]`. Each clip is processed alone: normalize raw-A rows only after
the bank's `2e-4` unit check; require `sign[clip] * raw_A.x > 1e-6` for physical-B wire
representability and `raw_A_row · reference_A > 1e-6`, the same open-hemisphere margin enforced at
runtime; normalize the row-vector sum; and save
the minimum row-to-center dot. This
`per_clip_sign_preserving_spherical_mean_cap_v1` construction never averages forehand with
backhand or opposite face signs.

The ONNX carries envelope schema/frame/convention/pairing/algorithm, bank/runtime tolerances,
clip order, the exact sign table, two centers, references, dot thresholds, row counts, and
duplicated bank/family SHA
bindings. A dependency-free C++ SHA-256 implementation recomputes the canonical metadata payload;
the loader then rejects missing keys, a stale payload hash, bank/family mismatch, malformed or
non-unit vectors, flipped centers, invalid thresholds/counts and wrong clip order. `PpPolicy`
converts only the physical-B wire normal to A after selecting the clip, then checks both the raw-A
reference hemisphere and selected cap before its engage transaction commits clock, position,
velocity, side or normal. A positive-X physical-B unit vector whose converted raw A is outside the
selected support sets `face_command_out_of_train_envelope` and cannot start a swing. Older 179
ONNX files lack these
mandatory keys and therefore fail closed even under model-only/no-publish loading. Other registered
110/175/177/180 models retain their prior loader behavior.

Host verification currently covers the Python derivation suite, Python exporter/contract suite,
an Isaac-free standalone-import subprocess, locale-independent standard SHA-256/numeric parsing
and a compiled dependency-light C++ parse/accept/reject smoke. The Python results are `34 passed`
for contract/export and `11 passed` for the planner wire. The prospective real-bank fixture
binds bank `2da2bd12...a0700`, source family `b21c161a...28ad5`, `757/724` rows, raw-A/B sign
ranges and cap minima `0.974278/0.972078`; it is a read-only source-contract expectation, not a
behavior result. The fixture does not contain the ignored NPZ: restore and SHA-check it using
[setup_local_sync.md, Phase-1 fresh and causal bundle](../operations/setup_local_sync.md#phase-1-fresh-and-causal-bundle-2026-07-11).
A full vendor-dependency Release build and freshly re-exported real 179 model now pass the strict
model gate described below. A spherical cap is still only an on-distribution envelope, not a
collision or behavior proof:
self-hit instrumentation, the recovery tuple and no-reset vendor MuJoCo Gate 3/Gate 3B remain
open. G06 stays `Partial`.

#### 2026-07-12 publishable-model and atomic-export hardening

`PpPolicyConfig::diagnostic_no_publish` no longer reaches the ONNX contract escape. Plain
`--no-publish`, `--dry-run` and `--model-preflight-only` now load with the same publishable
schema-2 packaging, exact/complete schema-3 execution contract, normalization, effort envelope,
registered layout and (for 179) full content-bound normal envelope required by a live publisher.
The only legacy escape is `--allow-legacy-model-diagnostic`; CLI validation requires no-publish
and rejects its combination with model-preflight before loading a model or constructing a backend.
The optional real-model GTest now uses the strict constructor and requires parsed exact/schema-3/
publishable flags plus the 179 envelope when applicable.

The production preflight certificate is also derived rather than asserted by mode: it checks the
parsed booleans, prints `publishable_model_contract=true training_contract_exact=1`, and prints the
parsed envelope sign table. `verify_face179_preflight_failclosed.py` runs one valid model then
creates temporary metadata-stripped, missing-envelope and `training_contract_exact=0` variants;
each must exit nonzero with no backend marker, while legacy-diagnostic plus preflight must exit 2.
The helper/unit source gate passes locally; the real runner integration result is recorded below.

The standalone exporter now completes every checkpoint/donor/motion/harvest/bank/envelope check
before creating a graph, writes only an owned same-directory temp, checks the ONNX and metadata
round trip, fsyncs, and atomically replaces the destination. Failure tests prove an existing
`policy.onnx` remains byte-identical and no temp remains. Focused host results are `41 passed` plus
one optional real runner/model integration skip; planner wire remains `11 passed`.

The missing integration was then run on Pod1 from an isolated tree whose 19 changed files match
exact source `2fa35340b63f98c04c67c8b29c80939610fd86e9` (tree
`299a2907229c1aaa4b581007c0ebe46cd914a011`). ROS/Jazzy + AimRT 1.6 Release rebuilt all three
targets. Fresh SZ seed3 model-2000 was re-exported from checkpoint `11f3a288...e77a5`, motions
`f2cb2d9f...1687`/`17225533...7534` and train bank `2da2bd12...a0700`; the new ONNX is
`0c428ddf...b7b155`, with envelope payload `df3fd8ae...08502e`. The native suite reports 219
tests, 210 pass, 9 optional-asset skips and 0 failures. Strict production preflight exits 0 and
prints parsed publishable/exact/179/sign/bank/family values with no backend marker. Graph-identical
metadata-stripped, missing-envelope and exact=0 variants each exit 3 before backend; legacy plus
preflight exits 2. An exam-bank-as-train failure preserves the existing ONNX SHA and leaves zero
temp files. All 824 compile commands are free of fast/finite-only math flags.

The content-addressed ledger is
`configs/gate3_face179_strict_preflight_evidence_20260712.json`. This closes the corrected export,
full Release build and strict model-only preflight gates. It did not start the vendor simulator,
transport or backend, so first tick, planner-policy closed loop, self-hit, continuous stability and
Gate3/Gate3B behavior remain open; G06 remains `Partial`.

### Planner proposal held out by coordinate and clock pairing counterexamples (2026-07-12)

The selective planner proposal at `69418a9` is **not merged** despite its green host suite. Its
explicit schema-2 side was selected from `intercept_y_world - base_y_world`, whereas the C++ policy
forms `tgt_b = R_yaw^-1 * (target_world - base_world)`. This changes decisions inside the allowed
heading range: with base yaw 10 degrees and target delta `(0.67, 0.02) m`, the proposed selector
chooses BH while policy-frame Y is `-0.09665 m` and requires FH. A wrong side also chooses the wrong
clip before the formal face179 `[+1,-1]` physical-B-to-raw-A conversion. The correction must use
the same normalized base-yaw frame and fail closed if orientation is missing or invalid.

The proposal's `2.6 s` prediction horizon exposed a second policy-clock mismatch. Current formal
179 clips have approximately 1.30 s FH and 0.88 s BH maximum in-training windup. Unlike 110, 179
does not wait while a valid command is earlier than that clip window; it clamps and engages, so an
approximately 1.89 s Gate3 arrival would put the FH/BH strike clocks about 0.59/1.01 s early. A
formal fix needs selected-clip, metadata-bound waiting with continued freshness/revocation checks,
or a preregistered serve schedule already inside the relevant windup window. Merely widening the
planner horizon is not a demo fix.

Finally, future serve release must bind both exact-owned runner fresh MOTION and exact-owned
planner fresh readiness, including executable/argv/config and PID/PGID/start-ticks/log inode. A
runner marker alone cannot prove the first planner command exists. The reviewed source remains
outside main until these pairings and C++ tests close. No vendor runtime ran; G06 remains `Partial`.

#### Planner correction still held by revoke and yaw-ownership counterexamples (2026-07-12)

The follow-up candidate `71b0b23` fixes the original pure helper geometry and adds selected-clip
windup waiting plus a dual runner/planner readiness preregistration, but it is still **not merged**.
Independent state-machine review found that a base sample ageing past `0.2 s`, or a newly malformed
base sample, clears only the planner's in-process corrected-base tuple. Neither path immediately
publishes a schema-2 invalid flat command. A previously valid racket tuple can consequently remain
inside the C++ subscriber's `0.5 s` command timeout and become eligible again if base freshness
returns before another admitted ball solve.

Publishing an invalid row alone is insufficient for formal 179: the current runner applies the
legacy `planner_invalid_grace_s=0.25` to every actor. A tuple explicitly revoked while waiting can
therefore still be classified fresh during the grace interval. Formal schema 2 must revoke
immediately; the historical grace may remain only for explicitly registered legacy contracts.

The proposed Python selector also consumes the corrected mocap/pelvis quaternion, while
`LocMode::kExternalBase` deliberately keeps the runner's boot-yaw-aligned IMU orientation and
ignores the quaternion carried by `/a3/base_pose_flat` for the policy target frame. Passing the same
quaternion to two pure helpers does not prove those runtime authorities coincide. Before merge, the
runner must fail closed whenever the planner's proposed side disagrees with the runner's actual
policy-frame geometry (with an explicit boundary rule), and dynamic tests must cover stale revoke,
malformed-base revoke, revoke during windup waiting, recovery before timeout, and mismatched yaw
authorities. No simulator, backend, Pod process or robot ran; G06 remains `Partial`.

#### Third planner correction held by same-tick and cross-topic causality failures (2026-07-12)

Candidate `6aae7ac` added dual flat revocation, immediate formal invalid, a base-receive epoch,
runner-frame side consistency, sample-driven READY and stronger source/environment bindings. Its
host suite is genuinely green at `198 passed, 2 skipped`, but it is still **not merged**.

`ComputeCommand` invokes `PlannerEngageStep_` before it samples the current IMU/base mailbox. A
previous tick with base age `0.199 s` can therefore leave `base_fresh=true`; on the next tick the
sample may be stale or explicitly invalid, yet engage can latch level 1 before current localization
is read. The same ordering makes the side check use `last_base_quat_w_` while the observation later
uses the current IMU. A small yaw change can move `tgt_b.y` from the ambiguous band to the exclusive
side region after the wrong clip has already latched.

The receive-time epoch is also insufficient across two DDS topics. Per-topic ordering permits
base invalid then base valid, followed by a delayed pre-revoke racket valid and only later the
racket invalid. Because the old valid is received after the base invalid, a local timestamp
comparison misclassifies it as post-epoch. Formal closure requires a single source epoch/sequence
carried in both base and racket payloads (or one atomic combined topic), exact equality at engage,
and a single same-tick localization snapshot shared by engage, side/face gates, windup wait and
the policy observation. The immutable prereg must also bind and parse the mailbox/wire/frame helper
bytes it currently omits. No runtime, simulator, Pod mutation or robot ran; G06 remains `Partial`.

#### Shared-epoch candidate still held by source-age and active-revoke counterexamples (2026-07-12)

The next unmerged worktree adds schema-3 racket rows, schema-2 base rows, a shared source epoch and
sequence, source-header-to-monotonic mapping, a common mailbox transaction mutex and a single
localization snapshot reused by engage and observation. Its host suite reaches `155 passed, 2
skipped`; an isolated ROS/Jazzy Release build reaches `220 passed, 5 optional skips`. These are
useful source results, not merge authority.

A second fresh review still reproduces five P1 groups. Once a formal stream is established, a
recognized legacy-schema packet can downgrade state without poisoning pre-barrier recovery.
Formal invalid state still has one receive-wall-time `>` dependency, so valid and invalid events
sharing a clock tick need not revoke deterministically. The Python base lease is keyed to receive
time rather than mapped source time, and expiry occurs after current-sample admission; an already
old base can therefore calculate a command or print READY, then be rejected by the runner.

Active formal 179 swings also fail to latch both the engage epoch and a base-revocation generation.
An epoch change followed by fast recovery, or a local malformed base followed by same-epoch valid,
can be hidden between two policy ticks while an E1 frozen target continues on newer localization.
Base epoch/revocation changes are localization safety revokes and must abort/rearm even if ordinary
racket flutter remains frozen in flight. The latter behavior must be stated once: current comments
conflict over whether malformed racket input aborts an active swing.

Finally, the planner advertises world/table frame codes but does not compare the incoming ball/base
ROS `frame_id` to a configured formal authority. A fresh finite sample in a different frame can be
relabeled as formal world. Exact schema-3 must bind and enforce both frame ids, or remain inexact
behind a runtime publisher/header gate. The serve preregistration still omits wire/mailbox/frame
helpers, merged-YAML parser semantics, same-host monotonic authority, unique publisher/domain and
hot-restart session closure. The candidate remains **NO-MERGE** and G06 remains `Partial`.

#### 2026-07-12 joined-source first-tick diagnostic

The production runner now implements no-publish-only `--first-tick-json` instrumentation for a
strict 179-D model. PASSIVE waits; SHADOW records the first observed planner-engaged actor candidate;
idle/wait/invalid/recovery rows do not consume it. The output is canonical mode 0600, fsynced atomic
hard-link no-replace and contains joined qpos38/qvel37/base7/racket7, target candidate, obs179,
action31, layouts, clocks and content SHAs. It does not emit a source-commit claim.

`RobotState` lacks root linear velocity, so a subscription-only sim sidecar reads the vendor pelvis
pose/twist and right-racket pose topics without publisher/reset/command or estimation. Kernel
`flock` plus whole-record `pwrite/pread`, freshness, finite/unit checks, strictly advancing stamps,
positive even generations, 20 ms native-header skew and a 30 ms RobotState/sidecar receipt join are
enforced. The observation base is recorded separately from joined vendor-world base and the native
racket point must agree with formal FK within 5 mm.

This is deliberately not a native same-tick snapshot. The tracked vendor publishers stamp messages
asynchronously at publish time and expose no common MuJoCo sample sequence. The current planner also
has known same-tick snapshot/shared payload epoch blockers. Both outer document and payload fix
`evaluation_contract_exact=false`; planner/native/source-binary/source-semantics/runtime-closure
exactness are fixed false with non-empty reasons. Gate3/Gate3B and promotion consumers must reject
this v1 schema.

The model path has no load/hash TOCTOU: stable canonical ONNX bytes are hashed and passed directly
to ONNX Runtime. The checked-in ledger hashes only a reviewed source subset and fixes
`source_semantics_closure_exact=false`; it is not parser-backed closure. Vendor config→MJCF parser
resolution, publisher binary/config/transitive membership, planner/wire/frame/backend closure,
owned supervisor/timeout, runtime ledger and actual backend first tick remain OPEN/null.

Host source checks are `6 passed`; the combined static-plan+diagnostic tests are `38 passed`. No
simulator, transport, backend, Kit, Pod/GPU or hardware ran. Full ROS/Jazzy/AimRT Release build and
native GTest are also unrun. G06 remains `Partial`.

### 2026-07-12 model-4000 matched-q50 execution source gate

The fresh `SZ` model-4000 matched MuJoCo paper now has a source-reviewed consumer for the existing
all-four activation barrier. It strictly pins queue, preregistration, queue validator, fresh exact
result helper, itself and the four evaluation-tool files. It accepts no queue-only or one-Pod
authorization: every command revalidates the activation content hash, exact barrier id, both Pod
audit hashes, four checkpoint audit records and the immutable K100 file/semantic/order hashes.
At runtime each Pod additionally rehashes its two local model-4000 checkpoints and reruns the
finite/embedded iteration/contract/lineage audit.

The execution contract does not call the schedule materializer and rejects a different path even
when its bytes happen to match the paper audited into the activation. `prepare` creates an
activation-bound no-clobber contract. `run` executes two serial pinned `judge.sh` children per Pod,
sets the common Kit boot lock, verifies each new-session PID equals its PGID, waits for completion
and preserves state/log on every failure without exposing any signal API. Seed1 is rerun, not
reused. A result must reproduce `evaluation_contract_exact=true`, K100/50-per-side, schedule/order,
vendor-development MJCF, execution/ready-state and checkpoint/hard-contract bindings; report,
summary and raw attempt ledger are rehashed before the Pod result is written.

Aggregation retains the unchanged model-2000 stability thresholds but fixes
`family_stable_claim_allowed=false` because seed1 `.50` was known before preregistration. Seed4's
only permitted conclusion is delayed learning versus persistent weakness through 4k. This paper
still does not answer the Isaac/MuJoCo physical-instrument gap, calibrated plant, recovery,
continuous stability or Agibot vendor Gate3/Gate3B behavior. Focused queue+consumer tests pass
`40`; at source merge no Pod audit, activation, MuJoCo judge or simulator had run.

The all-four barrier was then materialized outside train/eval on 2026-07-13 local time. Pod1's
seed1/3 audit is `3fc325e1...247b8`; Pod2's seed2/4 audit is `4f25786b...565f7`. Their exact union
created activation file SHA `9dea76c2...ce704` with content SHA `eaa92ca2...aa4fb`, covering all
four seeds and retaining `judges_started=0`. Source, K100, both audits and activation are present
at the same absolute paths on both Pods. Both runner `contract-check` calls passed; immediate
pre-run snapshots found no child judge, MuJoCo evaluator, play/Kit process or shared-lock holder.

Both Pod runtime contracts were subsequently created by no-clobber `prepare`. Pod1's file/content
SHAs are `2b76a5a...8201e` / `36e878f0...5ba73`; Pod2's are
`dbecc102...d1c9b` / `91a0070a...30794`. Direct binding validation rehashed both local checkpoints
per Pod and confirmed iteration 4000, finite tensors, exact lineage, the shared hard-contract SHA,
clean exact train/eval checkouts and an empty post-prepare process/lock snapshot. Both contracts
remain `prepared_not_started`, `jobs_started=0`, `auto_start=false`; no `run`, judge, aggregate,
score, signal or hardware action occurred. The remaining launch blocker is a reviewed persistent
parent supervisor that retains serial two-seed ownership and final-result materialization after an
SSH disconnect. This is execution-paper preparation rather than behavior evidence, so G06 remains
`Partial`.

### 2026-07-14 signed-face v8 再次在合同前阻断，不授权判卷

Pod1 epoch-1 v6 的 A/B/C 已到终档，但 D 在产生 runtime verified/checkpoint 前 Kit boot timeout；因此
不存在完整四格 L1 activation。后续 [v6r1](../DEFINITIONS.md) 首次真实 `validate` 在任何 claim/训练前
发现 expected-absent 合同错误：checkpoint audit 明确 D `run_dirs=[]`，但 validator 却要求旧 would-be
training path 是 directory。团队没有伪造目录；v6r1 从未启动。新 [v6r2](../DEFINITIONS.md) 只发布
source-only 静态修正，要求旧 path absent 且任何 entry kind fail closed；它没有 runtime、命令重建、
launch、signal 或 mixed finalizer，明确 NOT LAUNCHED，不能补出 L1 activation。

后续 foreign v8 不采用 v6 artifact，而以新 source/manifest/launcher 按 A/B/C/D terminal barrier 串行
运行。A/B/C 前序已终档；D 作为第四格又在 900 秒内未产生 hard contract/runtime verified/checkpoint，
exact-PGID wrapper cleanup 后 rc=124。它是继 v6 D 后第二次独立 pre-contract Kit boot timeout；自动
retry 已停止，必须先做 boot root-cause。v8 没有四格 activation，且 L2/judge/第二 seed 均为 false，
所以不能进入 Isaac/MuJoCo 同卷，更不能成为 Gate3/Gate3B 或部署证据。G06 保持 `Partial`。

只读 postmortem 现已证明两个 D 的最后 Kit 语义操作都是加载 byte-identical table USD，且都未进入
PhysX context；相邻 C 则分别在 `2.339/3.031 s` 越过同一边界。它只把 failure boundary 从泛称的
“boot timeout”收窄，未证明第四进程、Carbonite cleanup、driver 或 filesystem 中的任一项是根因。
结果 ledger 明确把 fact/inference/unknown 分开；`dmesg` 不可读，共享内存残留只记 correlation。
计划中的 D-first/ordinal-4 与 fresh private IPC 对照仍是 [design-only prereg](../../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)，
不能运行训练、不能生成 L1 activation，也不能授权 Isaac/MuJoCo 判卷。G06 保持 `Partial`。

### 2026-07-13 MuJoCo trainer preflight 红队：授权安全，源码门暂缓

独立复核确认 `codex/mujoco-training-preflight@6e5fce3` 的 focused `63 passed`、顶层
`468 passed, 9 skipped`、`valid_but_blocked`，以及 `--require-ready`/certificate 输出均 rc=2。
七个授权布尔值全为 false，也没有 consumer/launcher，所以当前分支不能启动或批准训练。
但它仍因四个 red-team P1 正确性缺口被标为 `NO-MERGE`：trace 缺 action clamp/runtime adapter
状态且 action tape 太小；静态 source independence 可经 alias/exec 绕过；JSON 未拒 duplicate key/NaN；
MJCF `compiler strippath` 语义建错。vendor scene 仍无可碰撞球/台/网，因此 v0 只允许无球
balance/strike-state diagnostic，不能记 physical return。

修复这四个源码门后，第一个 single-env core 还要通过预注册的 N=1/8/32/64 吞吐继续门，并证明
两臂×两 seed 在 48 小时内完成且留 30% 余量。这个门不替代独立 exact vendor Gate3/Gate3B，
也不阻塞几天内 `Gate3-D0`。G06 仍为 `Partial`。

### 2026-07-12 文档路由

引擎/迁移现状只在 [`docs/NOW.md`](../NOW.md) 汇总一次，详细实验记录放在
[`docs/experiments/`](../experiments/README.md)。经过筛选的
[`docs/TIMELINE.md`](../TIMELINE.md) 只记录已经进入 `main` 的重要能力和根因修复，并明确说明
Isaac–MuJoCo gap 只有部分可复现差异和候选来源已定位，整体因果归因与修复均未闭合。
Legacy Gate3 diagnostic 与 current exact-179 Gate3
属于两份独立实验记录，不能互相填充结果单元。本次纯文档迁移没有运行 evaluator、模拟器、backend、
Pod 或真机，也不改变 G06 的 `Partial` 状态。

### 2026-07-15 legacy V9 `7/7` proxy correction

The branch-local V9/v12fix `7/7` value is not a physical-return result. Its success predicate combines
planner engage, ready, at least one completed swing and recovery, while the result explicitly records
`physical_contact_measured=false` and `landing_measured=false`; the fake-ball publisher supplies planner
poses rather than a MuJoCo contact/flight event chain. Repetition covered a fixed forehand region, so it
also cannot establish unseen-ball, backhand or multi-action generalization. It may be retained only as an
exact planner-policy cycle-stability proxy. Physical selection still requires a vendor-MuJoCo receiver
with all-serves denominators and a disjoint held-out paper. The source audit and candidate mechanisms are
recorded in [the Jiayi/Yikang cross-learning experiment](../experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md);
no runtime, score or Gate status changes here, and G06 remains `Partial`.

#### 2026-07-13 exact formal tuple and portable Release integrated

The earlier planner red-team candidates above are superseded by exact source `c0a8e46`. Formal
racket schema 3 carries shared epoch, command sequence and exact `base_sequence_ref`; a bounded
base history proves that causal reference, while side/target/yaw, base-low, observation, active
abort and recovery use one latest tick-start base. Fixed-latency barriers no longer chase receive
time, and Python/C++ share the same workspace and source-time continuity limits. Every formal actor
path checks latest finite/fresh/plausible/base-low state and preserves the latched engage epoch plus
base revocation generation across recovery.

The 23 effective source/config/test paths were transplanted byte-for-byte onto latest main with
manifest SHA-256 `8af1a2fc37dc912f41cb5609a687b481fbadbddc531ff4f430d6294796665fd3`;
no source conflict was resolved semantically. Local planner/source is `180 passed, 2 optional
skipped`; serve preregistration is `39 passed`; full root tests are `521 passed, 9 skipped`.
Serve design-check passes only with all 49 runtime bindings blocked, and launch-check fails with 49
`MISSING` lines.

The same exact source passed isolated Pod2 Ubuntu 24.04/GCC 13 portable Release: focused
`PpPlannerInput.*:PpFirstTickJson.*` `40/40`, complete native `233 passed + 5 optional skips + 0
failed`, both test and production runner binaries linked, and all 80 compile commands retained
strict finite math. This closes the exact source/binary merge blocker only. ROS/Jazzy/AimRT were
disabled, the production runner was not executed, and formal ONNX runtime, backend first tick,
vendor MuJoCo behavior, continuous stability and hardware remain unrun.

The v2 joined-source ledger remains deliberately inexact: it has no common native MuJoCo sample
sequence, executed binary/runtime closure or owned supervisor. The separate serve v4 design keeps
49 runtime bindings null and cannot arm a publisher. Detailed hashes and reproduction are in
[the exact build experiment](../experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) and
[serve operation](../operations/run_gate3_serve_sync_prereg.md). G06 remains `Partial`.

### 2026-07-13 persistent model-4000 q50 startup source gate

A detached two-phase [persistent supervisor](../DEFINITIONS.md#persistent-supervisor) now closes
the activation consumer's top-level SSH-lifetime gap without changing any evaluated bytes. Its
caller-SHA-pinned config binds the existing consumer/config/activation and a distinct prepared
runtime contract plus Python realpath/binary SHA for each Pod. A fixed no-clobber state directory
contains child hello, immutable launch ledger, commit token, commit acknowledgment and combined log.
The token is withheld until the parent verifies `PID=PGID`, Linux boot id/procfs start ticks,
executable SHA, exact argv/fixed-environment digest and every artifact SHA; parent loss or stall
before that token makes the child self-exit rather than start an unowned judge. Deadline, process
identity and token/ledger binding are checked before token publication. Atomic token final-link
visibility is irreversible even if the following directory fsync fails; after it, slow rehash,
acknowledgment publication and exec are pending committed work rather than deadline failure. The
child still rechecks all bytes and
identity/token/ledger/result before acknowledgment and before `execve`. Separate acknowledgment and
exec-observation windows return `token_published_pending_ack` or `committed_pending_exec` with return
code zero instead of creating retry authority when progress is not yet visible.

Read-only `inspect` rehashes the complete closure. A live result requires the preserved PID, PGID,
start ticks, executable, command line and environment digest to match; terminal acceptance is
delegated to the unchanged runner's complete schedule/arms/lineage/count/report validator. A
pre-existing result prevents launch. No retry, remote login, process-control, trainer/worker,
simulator, deployment or robot surface exists. The detailed contract and commands are in
[the model-4000 q50 operation](../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md#persistent-top-level-launch-source-gate).

Supervisor tests pass `24`; combined queue/consumer/supervisor tests pass `64`. Tokenless deadline
expiry cannot execute; post-token delayed rehash, a 1.15-second acknowledgment atomic-publication
stall and post-ack delayed exec all reject restart and later converge without a
fatal-before-later-runner sequence. Terminal validation also freezes bytes/SHA and rejects an A-to-B
replacement. Post-link token-directory-fsync plus evidence-stat failure, token temporary-cleanup
failure and parent-observation-write failure are separately covered: all return committed pending,
reject restart and later inspect as exact running without a fatal. This is host source evidence only: Linux procfs has not yet been smoke-tested, the wrapper is
not deployed, and no MuJoCo judge or score ran. It therefore closes neither the matched
[q50/K100](../DEFINITIONS.md#q50-and-k100) result nor vendor Gate3/Gate3B. G06 remains `Partial`.

### 2026-07-13 Phase-1 trainer 裁剪不改变 MuJoCo 证据门

16 条 fresh 广度臂中的前 8 条已按负责人后续运营决定精确停止并保留最新 finite、schema-3、
fresh-lineage checkpoint；其余 8 条在后续 signed-face 取证后也已停止。formal `SZ` seed1/2/4 trainer
虽在首波停止，但 model-4000
四 seed checkpoint 在此之前已经通过 all-four readiness 并进入两 Pod 的
`prepared_not_started` K100 runtime contract，因此后续 matched q50 输入没有变化。

这次动作没有运行 MuJoCo judge、没有新 q50/physical/Gate3 分，也没有修改 q10 screen-only 或
q50 `whole_arm_stop_allowed=false` 合同。停止运行不能替代 signed-face 诚实门、同一 checkpoint
跨引擎归因、厂商 runtime `Gate3/Gate3B` 或标定 plant。运行与证据边界详见
[拍面×plant 广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)；G06 保持 `Partial`。

model-4000 取证后，剩余 8 臂的最近 24 个 K20 格又全部出现正手 signed composite=0，
法向误差 `164.4°–175.2°`，而 parsed return 可达 1.0。负责人因此批准第二波精确停臂；
现在 16 条 fresh 广度 trainer 均已保留证据并停止，无残留 judge/Kit 且两 Pod GPU 为空。
这只使得“修 signed-face 后再训”成为明确顺序，不会关闭跨引擎或 Gate3/Gate3B；G06 保持
`Partial`。

### 2026-07-13 model-4000 matched q50 结果与拍面仪器失真

内容绑定的 Linux supervisor 冒烟、两 Pod 判卷和单次 aggregate 已完成。Pod1/Pod2 result
file SHA 为 `02d0e58d...645d` / `d31323a6...4e6f`；aggregate file/content SHA 为
`1ba88e39...d195` / `226e6050...648d`。四 seed parsed rate `.50/.88/.98/.00` 使预注册稳定门
的 median/worst/spread/worst-side 四项全失败，seed4 为 21 次物理 root fall 且 `0/100`，
因此不支持 4k 晚熟。

这份卷更重要的 G06 证据是同 checkpoint 内部的仪器矛盾：seed2/3 解析正手 return
`38/50` / `48/50`，但 raw-A 有符号法向差 `172.33°/174.35°`，位置+速度+法向复合
命中均为 `0/50`。因此旧 `orient_normal` 解析回台分已被实证为正手符号盲区，
不得作为跨引擎或部署晋级证据。需先用 `n/-n` 负控修表、重跑同卷，再以同一
checkpoint 进行 kinematic replay → open-loop action → external-observation closed-loop → native
closed-loop 归因。该 Python BankExam 仍不是 Agibot vendor Gate3/Gate3B runtime，所以 G06 保持
`Partial`。详见 [稳定性实验](../experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md) 和
[拍面符号取证](../experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。

### 2026-07-13 signed-face analytic ruler source gate

feature source 已把 analytic scorer 升为 schema 2：formal 路径必须从 ONNX metadata 读取完整
`mount_normal_sign_per_clip`，把 achieved/target raw-A 通过每 clip sign 绑定到 opponent-facing
physical-B，并在 `orient_normal` 前要求 strict signed hemisphere 与 achieved/target B.x `>1e-6`。
结果与 strike CSV 同时写 signed-face exactness、dot/error 和 physical-B-facing 字段；scorer source、
venue config、参数、sign table 与门限进入 execution-contract SHA。缺 metadata、非法 sign 或长度不符
均 fail closed。

只有显式 `--allow-inexact-contract` 可保留旧 unsigned-plane 诊断；它必须写
`signed_face_exact=false` 与 `evaluation_contract_exact=false`，不能晋级。phase/spatial 动作筛卷调用点
也已迁移到 raw-A achieved/target + clip-id；历史 v5 和 q50 仍绑定旧 scorer，只能作为 paired legacy
列，不能被新源码追认。`n/-n` 负控锁定了根因：冲量与落点逐值相同，但新门只让正确 physical face
contact/return。

本地 focused source/unit 回归为 `38 passed, 1 skipped`，顶层 broad 为 `546 passed, 9 skipped`；
没有运行 MuJoCo/Isaac/Pod/judge/vendor
Gate3/Gate3B。下一步是在同一 immutable K100/同 checkpoint 下生成新 execution contract 和不覆盖的
paired result，再按 kinematic replay → open-loop action → external-observation closed-loop → native
closed-loop 分层归因。analytic scorer 即便通过也仍是 diagnostic，不替代 physical return 或 vendor
runtime；G06 继续 `Partial`。

训练侧随后为同一问题物化了单-seed A/B/C/D 漏斗，但它没有改变 G06 的裁判边界。A/B 从旧
`model_13800.pt` 进入当前源码时，因为 hard-contract 新增 event-timing/target-cadence 字段，只能是
`training_contract_lineage_exact=0` 的显式表示迁移；C/D 才允许 fresh lineage `1`。L1 是
25-update launch-integrity smoke，其四格 completion 文件不能授权 L2 或 judge。immutable signed-face
directional checkpoint paper path/SHA 尚未冻结，manifest 明确 `l2.launch_authorized=false` 且
`automatic_judge_launch=false`。源码/攻击回归 `23 passed` 不构成 Isaac/MuJoCo 行为或 Gate3 结果。
Pod 首次 v1 preflight 的 checkpoint 假拒绝已根因到“顶层扫描/顶层 provenance”错误；v2 递归审计
嵌套 tensor，并只接受 runner 写在 `checkpoint["infos"]` 的合同字段。v2 首格又在学习前暴露 detached
worktree 的 source-first 环境没有传入 child；v3 又因在 `SimulationApp` 前真正 import IsaacLab 而
假拒绝。v4 用确定性环境 SHA 与不执行包的 exact-worktree `find_spec` 在 claim 前关闭发射缝，随后
又在 scene 构建时发现 ignored A3 资产不会随 worktree 出现。v5 绑定并验证 exact restore/target 资产树。
上述修复都没有缩小 Isaac–MuJoCo 行为差，也没有启动 judge；
操作见 [signed-face 漏斗运行手册](../operations/run_phase1_signed_face_rescue_funnel.md)，G06 保持
`Partial`。

v5 又在第一次 learning iteration 前揭示旧 train bank 的 physics-contract SHA 与 `882fea4` 不同。
main 的严格 rebind consumer 不放宽 schema-3 loader：它只在 source/AST、全部问题数组 raw bytes、四个
metadata leaf、exact motion contract，以及全部 1481 道题 old/new contact/flight bitwise replay 同时
通过时发布新路径 train bank。Pod1 已发布 bank/report `3a9d8851...5b71` / `9fffed03...bb37`，两侧
landing/net 全过；这只是 E2 runtime data gate。train rebind 后 source-family SHA 已变为
`9603a178...a9db`；旧 immutable exam
不能与它组成 exact 同 family 证据。对应 exam bank 还未同法重绑定/重生成，L2 directional paper 与
judge 仍阻断，所以这项能力不构成 G06 或 Gate3 结果。

### 2026-07-14 signed-face exam bank E2 数据门完成

exam 对应的严格数据门已完成 E2 runtime replay 与 no-clobber 发布。generalized consumer 仍 byte-exact
接受历史 train-v2 manifest，同时新增一个封闭 exam-v1 profile：旧 exam path、`63,968` bytes、SHA
`d7db2568...f5096`、split `exam`、正/反手 `183/188`、旧 family `b21c161a...8ad5` 和独立 no-clobber
输出都不可替换；目标 physics/family 与已发布 train-v2 同为 `09dfe899...afb95` / `9603a178...a9db`。
mutation/source-receipt/profile 回归为 `18 passed`。Pod1 目标 runtime 的 24 个非 metadata 数组未变，
正/反手 `183/188` 道题 old/new output bytes 全相同并通过 landing/net；发布 bank/report SHA 为
`60e1a7ad...d1ca` / `dd4332ed...ad0`。

新 bank SHA 改变 question ID，旧 schedule 不得复用；必须从新 bank 重新冻结独立 schedule/paper
activation 后才能启动 judge。详见
[实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)与
[运行手册](../operations/run_phase1_signed_face_exam_bank_rebind.md)。当前 L2、signed-face paper、G06 与
Gate3 状态均不变，G06 保持 `Partial`。

### 2026-07-14 signed-face K100 materializer/activation source gate

E2 rebound bank 的下一层 paper source gate 已预注册并通过 host 静态/攻击回归。consumer 不接受旧
schedule 输入；它只从 exact `63,643`-byte bank SHA `60e1a7ad...d1ca` 重新生成包含新 bank SHA 的原子
question ID，并复用现有 schema-v3 deterministic schedule 算法冻结 seed `0`、hold `[0,100]`、每侧
无放回 50 的 K100。所有 100 个 scheduled attempt 保持在分母；missing/invalid/reset 不能删除。

paper 同时冻结 raw-A/physical-B signed 身份：clip order `forehand,backhand`、sign `[+1,-1]`，每个 target
raw-A normal 必须 finite/unit，映射后的 opponent-facing physical-B 必须严格 `x>1e-6`；unsigned 或先
`orient_normal` 再判身份的路径拒绝。旧 paper file/semantic/question-order receipt 均列入禁用表。
output root 必须不存在，schedule 与 activation 都 no-replace，activation 最后写；partial root 不能续写。

manifest/consumer SHA 为 `e401305d...e556` / `4e094bbe...ac6e`；mutation、旧 schedule、unsigned、
重复题、单侧不足和 partial no-reuse 回归共 `14 passed`，latest-main root `747 passed, 10 skipped`，
`static-validate` rc0。随后 Pod1 用 clean detached `748b6d5` source 成功执行单次 exact-bank consume：
schedule 为 100 个唯一题、正反手各 50，file/semantic/question-order SHA 为 `f2777dcd...1ca` /
`3ca4bdba...3365` / `09f778f2...bd0`；activation file/content SHA 为 `e0125b0e...bb4` /
`533beb03...3d8`，并在 schedule 落盘复核后最后写入。runtime receipt 见
[`phase1_signed_face_exam_k100_runtime_receipt_20260714.json`](../../configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json)。
activation 固定 trainer/judge/L2/第二 seed/晋级/部署/真机全 false；后续还需独立 reviewed execution
contract。详见
[实验](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)与
[操作](../operations/run_phase1_signed_face_exam_k100.md)。paper 已物化但没有 checkpoint/judge 行为，G06
继续 `Partial`。

### 2026-07-13 pelvis point/axis frame correction

A focused frame audit found two concrete MuJoCo evaluator errors without finding a gross
`xyzw/wxyz`, Z-up, gravity-sign, joint-axis or joint-name permutation error.

- Diagnostic `teacher-reference` reset copied the motion schema's pelvis-COM world linear velocity
  directly into the freejoint translation, which is the pelvis link-origin world velocity. The A3
  pelvis origin-to-COM offset is about `0.1273 m`; the corrected path applies
  `v_origin = v_com - omega_world x (R_world_body * body_ipos)` only to a clip explicitly bound as
  `center_of_mass`. Checkpoint-bound schema-3 native and standalone exports now carry
  `motion_body_lin_vel_points` for every clip; explicitly bound `link_origin` clips retain direct
  assignment and remain exact-ineligible. Schema 1/2 or contractless standalone re-exports strip
  the field rather than inheriting an unproved donor claim.
  Old exact schema-2 exports have one narrow all-COM compatibility rule. Missing/inexact aggregate
  metadata cannot identify a point and now fails loudly before teacher-reference reset instead of
  being guessed.
- `base_ang_vel` used `mjOBJ_BODY` with local output. In MuJoCo that is expressed in the compiled
  inertia-principal axes, not the pelvis link/IMU axes required by the actor and used for projected
  gravity. The corrected read uses `mjOBJ_XBODY` with local output. The vendor A3 pelvis inertia
  axes differ from the link axes by about `0.3315 deg`, so this is a real every-step observation
  mismatch but not, by magnitude alone, evidence for the observed cross-engine strike gap.
- The evaluator requires exactly one freejoint owned by `pelvis_link`, at qpos/dof address zero.
  Other free bodies, such as a dynamic ball, remain permitted.

A separate read-only audit found the analogous latent bug in the vendor ROS `SimReset` nonzero
base-twist subscriber: its published twist is world/odom link-origin twist, but world angular
velocity is copied directly into body-local freejoint qvel. Existing keyframe scripts and formal
K100 send zero velocity, so this does not explain their behavior and is not changed in this Python
evaluator ticket. The open G04/G07 interface contract is recorded in
[`frames_and_coordinates.md`](../interfaces/frames_and_coordinates.md).

The real `a3_pingpong.xml` regression uses nonzero orientation, three-axis angular velocity and COM
velocity; it checks freejoint origin velocity, COM world velocity and the actor gyro frame. The
focused reproduction is:

```bash
/Users/yyk956614/anaconda3/envs/backend/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_mujoco_reference_reset_com_frame.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py
```

This focused command passes `51` tests on the local CPU environment with the real MJCF test
executed and zero skips; the formal CPU contract group passes `115` with zero skips, the complete
contract union passes `183` with zero skips, and the repository's supported root `tests/` suite
passes `554`.
No policy rollout, Pod job, vendor backend or hardware ran. The reset correction does not change
formal `stand-keyframe` K100 because that path starts with zero qvel; the gyro correction does
affect its actor observation. The separately preregistered
vendor/root-only/joints-only/full-match ready-state four-cell remains unrun, so these source fixes
do not close the causal Isaac-to-MuJoCo gap. Full forensic scope and limitations are in
[`EXP-MUJOCO-PELVIS-FRAME-PARITY`](../experiments/2026-07/EXP-MUJOCO-PELVIS-FRAME-PARITY.md);
G06 remains `Partial`.

### 2026-07-14 evaluator parity red-team closure (source gate only)

Independent review found that the first implicit-effort guard could miss saturation-equivalence
errors when P and D cancel, and that the first self-contact counter treated every non-world dynamic
body as robot. The corrected evaluator executes Isaac's total `clip(P-D)` law for bound zero-passive
implicit joints, rejects passive/unbound proxies from formal status, classifies only pelvis-subtree
robot geom pairs and fails formal BankExam on any such pair. A dynamic-ball negative control is
explicitly excluded.

The same review closed three evidence-publication bypasses: command-mask provenance now accepts only
canonical callables or strict empty built-in partials; the revoked model-2000 Phase-B rider is denied directly
by content SHA even when a caller bypasses the 2x2 validator; and cumulative scoreboard CSVs refuse
an old header rather than appending misaligned wider rows. The historical Phase-B launch command is
now documented as forbidden and a replacement requires a post-epoch checkpoint/new rider/all four
cells rerun. No new policy rollout, immutable K100, vendor backend, Gate3/Gate3B or robot result was
produced, so G06 remains `Partial`. Full scope is in
[the integration experiment](../experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md).

A second independent review found two residual fail-open paths before merge. First, `isinstance`
also accepted a `functools.partial` subclass whose overridden `__call__` executed different command
semantics while `.func` still named the canonical function; provenance now unwraps only exact
built-in partial objects at every layer. Second, self-contact was classified only after all physics
substeps in a control step, so an earlier transient collision could affect dynamics and disappear
before grading. Classification and formal refusal now happen after every `mj_step`, with diagnostic
substep aggregates retained. Dependency-free negative tests reproduce both attacks; the optional
real MuJoCo contact/frame modules remain separate runtime evidence rather than being inferred from
those tests. G06 remains `Partial` pending an immutable behavior paper.

The two optional MuJoCo modules were then collected on isolated Pod2 CPU. Invalid synthetic
controls failed first and were preserved; correcting only their execution-equivalence and
articulation assumptions produced `10/10` with the same production evaluator bytes. This is E2
source/runtime evidence for the helpers, not an immutable policy paper, vendor MJCF run, Gate3 or
Isaac-to-MuJoCo gap closure. Exact SHA are in
[`mujoco_eval_optional_runtime_test_results_20260714.json`](../../configs/mujoco_eval_optional_runtime_test_results_20260714.json),
and G06 remains `Partial`.

### 2026-07-14 C2/D2 L1 不授权跨引擎判卷

新的 [C2/D2 L1 source gate](G05_isaac_training_first_loop.md)修复的是训练 checkpoint 的证据身份：
guidance weight 进入相邻 hard contract，outer claim/source/GPU lane 进入 checkpoint `infos`。它只准备在
Pod1 GPU1/GPU2 买两条 fresh 25-update provenance smoke；当前没有 runtime contract、`model_24.pt`、
ONNX export、immutable signed-face **policy** paper execution 或 MuJoCo 结果。K100 schedule/paper-only
activation 已另行物化，但该 activation 明确把 trainer/L2/judge 全设为 false。

launcher/finalizer 源码中不存在 activation、judge、L2、第二 seed 或 stop/promote mode；pair result 也把
这些授权固定为 false。即使未来两个 L1 terminal 通过，也只能说明 source/claim/contract/finite/lineage
闭合，不能说明 guidance 有效，更不能作为 Isaac/MuJoCo、vendor Gate3/Gate3B 或真机成绩。进入 L2
之前还需独立、no-clobber 的 L1 result consumer，并把已物化 paper 的 exact receipt/SHA 纳入一个新的
L2-only activation；当前 C2/D2 pair result 自己不能翻转该 paper activation 的 false 位。启动 judge
又需另一个 reviewed execution contract。G06 保持 `Partial`。

### 2026-07-14 C3/D3 K100 v1 asset-packaging failure 与 v2 source gate

C3/D3 paired K100 v1 在 C3 ONNX 导出前自然失败：独立 eval checkout 缺少 Git-ignored
`agibot_a3/urdf/model.urdf` 及同闭包 meshes/config。tracked-clean gate 因上游
`assets/.gitignore:*` 没有覆盖这些 runtime bytes。v1 output/attestation 已消费并永久冻结；没有 ONNX、
MuJoCo attempt 或 K100 成绩，因此不能判 C3/D3 行为。

v2 在全新 attestor/pair namespace 中把训练 checkout 的 ignored A3 递归 canonical inventory、required
URDF、C3 一次 hydrate/D3 exact verify 角色与 `libGLU.so.1` 可加载性放到 claim/judge 前。focused
`56 passed`，static/source-plan rc0。用同一训练 asset closure 和 exact checkpoint/bank/plant binding
做的 C3/D3 inexact diagnostic 均成功导出并进入 MuJoCo，故 asset packaging blocker 已关闭；日志明确
为 `evaluation_contract_exact=false`，且两侧均在第 0 题前因
`formal BankExam reached bound PhysX joint-velocity limit on articulation indices [8]; MuJoCo lacks same
braking constraint` fail closed。两侧均 `asked=0`，没有 attempt/score/方向分；新的 open blocker 是
articulation `[8]` 的
velocity-limit braking parity。可复现合同见
[v2 操作](../operations/run_phase1_signed_face_c3d3_k100_v2.md)。G06 仍为 `Partial`：parity 修复前不得
执行 formal paired K100；明确 allow-inexact 的方向筛不能授权 L2、第二 seed、stop/promote、部署或真机。
evaluator→attestor→paired manifest 的 SHA 级联与 hydrate 并发 sentinel no-replace 攻击已在 source
层通过（focused `57 passed`、static/source-plan rc0）；没有新的 Pod runtime 或行为结论。

### 2026-07-15 frame-0 等待 v2 的跨引擎边界

新的等待设计合同冻结了同一条连续序列在两个引擎都必须满足的状态边界：揭题前未来 action/clip/frame0/
target/deadline 全部不可见；揭题时一次原子切换；切换仅改 reference，不能写 root/joints、teleport、
reset 或清 history/action/delay/noise；selected public action 的 frame-0 pose 使用 phase-entry station XY，
所有 root/joint/body reference velocity 为零。Ready 是全部数值安全/可达 tolerance 的 fail-closed 合取。

当前 v2 只过 exact-byte CPU design check。现役 `commands.py` 的 default-stand hold、未清零 anchor velocity
与 live per-tick XY reanchor 和该设计冲突；source adapter、numeric tolerance、carry-state runtime receipt、
Isaac full-scene probe 与 vendor MuJoCo continuous gate 均未绑定。因而不能把 v2 写成 Isaac/MuJoCo parity、
Gate3/Gate3B 或部署证据；`launch-check` 必须失败，G06 保持 `Partial`。复现入口见
[恢复操作](../operations/run_phase1_recovery_tuple_prereg.md)。

### 2026-08-03 N1 reward/event kernel 与 native physical-facts integration

新增 `mujoco_native/n1_reward_event_kernel.py`，仅把调用方已 source-bound 的事实映射为 motion-mimic、A
target-window、closed-swing/hit、achieved outgoing flight、predicted outcome 与 observed net/legal-landing 的
独立布尔分母和付款资格。它不读取 MuJoCo state、不从 target/window 推断 contact、不预测落点，
也不分配 reward value。predicted outcome 严格要求 selected-rubber actual contact 后的 finite
outgoing flight；未命中 timeout 只保留 closed-swing 分母，不能付款。

production core 现在每 tick 返回 source-bound `a3_mujoco_n1_physical_event_facts_v1`：累计
racket contact edge count、首个 contact stamp、simultaneous/recontact invalid reasons，以及带
`(policy_tick, physics_substep)` 的首个 contact-free outgoing position/linear velocity/spin。VecEnv 要求
全部 core all-or-none 广告同一 contract，逐行重验 source/contract SHA、finite vector和严格事件顺序；
缺行或坏行会使整批失效。validated facts 进入 `DiagnosticBatchStep` 和 rollout v4 digest，并在
compact reset 前冻结 terminal tick。generic racket geom 命中继续只证明 blade contact；新增的
versioned selected-rubber classifier 只有在该 contact edge 已发生后，才用 official site frame、URDF
red/black outer planes 与 exact STL 派生的 strict inscribed safe disk 分类。safe disk 外为
`edge_or_rim_ambiguous`，球心在两 outer planes 之间为 `between_outer_planes_ambiguous`，两类都 fail
closed。classifier/question lineage 精确绑定 `action_id/action_uid/mount_normal_sign`、manifest/motion/
geometry/physics SHA、scene/assembled XML/backend/classifier SHA；legacy/manual question 仍无 authority。
source/dependency-light host focused tests=`73 passed, 28 skipped`。exact Pod detached clean
`4b43ac52` 再对 selected-rubber classifier、native ball core、reward event kernel 与 VecEnv 跑
`81 passed, 0 skipped, 0 failed`；这证明 current source 在目标依赖环境可 import/执行所覆盖路径，
但该 suite 没有发起实际球拍击球 rollout，所以 contact emission 仍为 `未测`；
`reward_authorized=false` 不变。

因此 normal `step()` 仍在 physics 前 fail closed。selected-rubber 的 source/classifier 子门已闭合，
但还需 exact Pod 真 MuJoCo ball-racket contact rollout；其余接口是 desired-contact/window、
outgoing-flight predictor、observed net/legal-landing resolver、swing-closure 和
per-term reward magnitude/weights receipt；全部齐备且能独立 replay sum-closure 后才能打开 normal reward。
PPO/save/cold-load 仍未实现；exact resume 还缺 MuJoCo/core/ledger/delay/RNG state hooks。G06 保持
`Partial`。

action-specific hold 的候选身份也从 path-bound v1 升级为 portable v2。generator 只把 repo-relative
POSIX logical path 和 source SHA 写入 canonical payload；consumer 固定从 repo root 解析，拒绝旧 v1、
绝对路径、`.`/`..`、repo 外来源及 symlink escape，run-tape 的 root-MJCF identity 使用同一 resolver。
host focused=`18 passed,6 skipped`；exact Pod detached clean current source 的真 MuJoCo d0/d1/d2 focused=
`24 passed,0 skipped,0 failed`。这关闭了换 checkout 路径就改变 hold SHA 的工程 blocker，但还没有
进入 ball-racket contact、Reward 或 PPO。

portable hold 通过后，exact Pod `592835dc` 用同一 immutable question 得到一次真实 generic racket
edge、零 table edge、valid actual contact/outgoing flight；runtime selected-rubber sidecar 报正号红面，
`policy_tick=1/physics_substep=3`，球心切向距拍心 `0.007168732 m` 小于 strict safe radius
`0.044263876 m`，无 ambiguity/invalid reason。旧 CLI receipt 没有序列化这份分类，所以 successor
把 ball-core receipt 升到 v2，加入完整 classification seal、classifier/question/scene/backend SHA、
stamp、face sign 和半径；无 generic contact 或 generic contact 无 classifier 时均写 explicit unknown、
`fail_closed=true`。host Python3.14 focused=`15 passed,3 skipped`；exact Pod v2 receipt replay 尚待新
commit 复核，Reward/PPO 仍未授权。
