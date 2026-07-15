# Validate The Phase-1 Recovery-Tuple A/B/C Preregistration

状态：旧 A/B/C 结构设计与 frame-0 等待 v2 均可校验；launch 明确 blocked

> 2026-07-13 阻塞说明：本文件绑定的 prereg/validator 仍把 random-arrival readiness 当第三项
> reward，并强制 full `2^3`。最新文档设计已改为“随机到球先作为环境轴；平衡债/ready potential
> 先做 `2^2`；第三 critic 独立校准并过隔离 q50 后才可选 `2^3`”。因此下面的 `design-check`
> 只验证旧结构记录没有漂移，不证明新 reward 次序已经物化，也不授权实验。不得改写旧文件；
> 必须另建并绑定新的 config、validator、测试和 SHA 后，才能更新本操作并考虑 launch。

> 2026-07-15 边界：新的 frame-0 等待 v2 已作为**独立设计合同**落库，且没有改写旧 prereg；
> 它仍缺 source adapter、数值 ready 容差和双引擎 runtime receipt，所以不是 launch manifest，
> 也不解除上面的 reward 次序阻塞。

## Frame-0 Waiting V2 Design Check

人话语义见 [T1 接口](../interfaces/t1_event_training_contract.md#selected-action-frame-0-waiting-contract-v2)：
揭题前只使用上一公开动作自己的第 0 帧零速度参考；原子揭题后才使用新动作自己的第 0 帧零速度参考。
XY 在阶段入口从当前站位捕获一次；连续 episode 只切参考，不传送、不 reset、不清历史/动作/delay。
Ready 是全部安全/可达容差的合取，不是 Reward 分数。

绑定文件：

- v2 design contract：
  `configs/phase1_frame0_wait_recovery_contract_v2_20260715.json`
  (`sha256 cc05d63fa4e4ffd9515f369f176ba032ca2a46d8996431a7b1e7d34e2b1bf28e`)；
- v2 validator：`scripts/validate_phase1_frame0_wait_recovery_contract_v2.py`；
- v2 tests：`tests/test_validate_phase1_frame0_wait_recovery_contract_v2.py`；
- immutable v1 parent 仍为 `17008` bytes / SHA-256
  `ca7806df83b650546cf4406963bb231622a248c8e04e944991a371e44d810616`。

从仓库根目录运行 CPU-only design check：

```bash
python3 scripts/validate_phase1_frame0_wait_recovery_contract_v2.py \
  --contract configs/phase1_frame0_wait_recovery_contract_v2_20260715.json \
  --expected-contract-sha256 cc05d63fa4e4ffd9515f369f176ba032ca2a46d8996431a7b1e7d34e2b1bf28e \
  --mode design-check

python3 -m pytest -q \
  tests/test_validate_phase1_frame0_wait_recovery_contract_v2.py
```

已签入结果为 `25 passed`。它覆盖 duplicate key、non-finite、bool-as-int、v1 parent/source SHA、
frame0/default-stand 偷换、缺 root/body/joint 零速度、live per-tick XY 重锚、future-action 泄漏、
teleport/reset/history/action/delay clear，以及把 ready 改成可补偿分数等负测。

`launch-check` 必须退出 `1`：

```bash
python3 scripts/validate_phase1_frame0_wait_recovery_contract_v2.py \
  --contract configs/phase1_frame0_wait_recovery_contract_v2_20260715.json \
  --expected-contract-sha256 cc05d63fa4e4ffd9515f369f176ba032ca2a46d8996431a7b1e7d34e2b1bf28e \
  --mode launch-check
```

现役 `commands.py` 的 hold 使用 `default_joint_pos`，root/anchor 速度未 hold-zero，body XY 又是 live
per-tick reanchor。只补一个 joint switch 会产生混合参考，所以 v2 明确没有 source adapter。
未来 adapter 必须默认关闭，并在一个变更中同时覆盖 selected-action frame0 pose、root/joint/body
全零速度、phase-entry immutable XY、atomic reveal non-leakage 与 carry-state receipt；再跑 strict
full-scene Isaac 和 vendor MuJoCo 连续门，才可另建 launch manifest。

## Purpose

Use this procedure to audit the command semantics between one completed swing and the next
randomly revealed task. It does not launch Isaac, MuJoCo, training, a Pod process, or a robot.
The frozen structural choices are:

- `A_explicit_bridge`: a content-bound, interruptible safe PD/trajectory bridge;
- `B_canonical_tuple`: the 179-D actor receives a complete canonical recovery tuple;
- `C_previous_tuple`: the actor retains the complete previous question tuple until reveal.

The current deploy hybrid is not a fourth formal arm. It combines a new live-base-anchored
position with the previous strike velocity and normal, a tuple generation absent from the bound
training source.

## Bound Files

- preregistration:
  `configs/phase1_recovery_tuple_abc_prereg_20260712.json`
  (`sha256 ca7806df83b650546cf4406963bb231622a248c8e04e944991a371e44d810616`)
- validator:
  `scripts/validate_phase1_recovery_tuple_prereg.py`
  (`33854` bytes,
  `sha256 c6d0e615dd1f356c28da701c8b13c937faa13f70f429c2550e26cb339e2e6991`)
- tests: `tests/test_validate_phase1_recovery_tuple_prereg.py`

The preregistration also verifies the immutable T0/T1 design/schedule files, exact training git
blobs at `d3cdbdfc2f6e30726aa197b7bca66496ee0d39e5`, and the read-only vendor Gate3 policy and
MJCF blobs at `1d46ef2cbb915efc135251f9b32f4ec25d0342ab`.

## Design Check

From the repository root:

```bash
python3 scripts/validate_phase1_recovery_tuple_prereg.py \
  --prereg configs/phase1_recovery_tuple_abc_prereg_20260712.json \
  --expected-prereg-sha256 ca7806df83b650546cf4406963bb231622a248c8e04e944991a371e44d810616 \
  --mode design-check
```

Pass output includes:

```json
{"current_179_B_usable": false, "current_hybrid_formal": false, "formal_arms": ["A_explicit_bridge", "B_canonical_tuple", "C_previous_tuple"], "launch_authorized": false, "status": "pass_design_only", "vendor_gate3_hard_prerequisite": true, "vendor_gate3b_final_behavior": true}
```

Run the red-team regression:

```bash
python3 -m pytest -q tests/test_validate_phase1_recovery_tuple_prereg.py
```

The checked-in result is `50 passed`. It includes raw-JSON red teams for duplicate keys at top and
nested levels, `NaN`/`Infinity`/overflowing numeric constants, exact preregistration identity/time/
scope, strict JSON type identity, unknown top-level fields and unknown fields in each critical outer
map.

## Expected Launch Failure

```bash
python3 scripts/validate_phase1_recovery_tuple_prereg.py \
  --prereg configs/phase1_recovery_tuple_abc_prereg_20260712.json \
  --expected-prereg-sha256 ca7806df83b650546cf4406963bb231622a248c8e04e944991a371e44d810616 \
  --mode launch-check
```

This command must exit `1` with `LAUNCH BLOCKED`. Do not change null fields in this immutable
design file. A future launch requires a new content-addressed manifest binding, at minimum:

- immutable q10/q50 random-arrival and question schedules;
- individual finite/iteration/lineage-bound 179-D checkpoints;
- A bridge source, whole-trajectory safety certificate, fresh checkpoint set, actor handoff,
  per-tick policy-ownership and PPO-rollout contracts;
- B canonical tuple source, ready-set selector, and fresh checkpoints;
- C coherent-tuple source and fresh checkpoints for any learned random-arrival claim;
- a complete numeric ready/base/racket/target contract, not just a named `stand` pose;
- same-family Isaac continuous judge, Agibot vendor Gate3 same-C++/MJCF/plant/model first-tick and
  continuous-stability judge, Gate3B random-arrival q50 behavior judge, and a shared Gate3/Gate3B
  runtime contract;
- racket/handle self-hit instrumentation and a semantics-correct calibrated plant.

## Checkpoint Interpretation

- Existing 179-D atomic-question checkpoints may be used in A only as a frozen swing diagnostic
  after the bridge and handoff are certified.
- They may be used in C only for a zero-shot coherent-tuple diagnostic. Extended final-frame dwell
  and random reveal were not trained, so this is not a learned recovery result.
- They cannot be used in B: the current training distribution contains neither the canonical
  zero-velocity tuple nor its neutral ready normal. B requires fresh training.
- A fair A/B/C causal comparison is fresh exact and paired, including a fresh A checkpoint produced
  under the bound bridge ownership/PPO contract. No existing checkpoint may be renamed `T1-trained`
  after the fact.

## A Bridge Policy Ownership And PPO Accounting

Every A tick must log `actor_control_mask`, executed action, shadow-policy action, the action written
to the actor's last-action observation, logprob validity and policy/entropy/value loss masks.

- `actor_control_mask=1` only when the sampled policy action is the action actually executed. Its
  PPO logprob must be for that exact action.
- On a bridge-owned tick, the bridge output is executed and the shadow action is diagnostic only.
  It has no valid logprob, and policy, entropy and value loss masks are all zero. Reconstructing a
  policy logprob for either the bridge or shadow action is prohibited.
- The last-action observation is a content-bound projection of the **executed** action into the
  actor action coordinates. It is never silently the shadow action, zero or a stale actor action;
  an inexact/unavailable projection fails closed.
- Bridge rewards are real. They use the duration-correct `gamma^k` sum across the bridge segment and
  collapse into the preceding actor-owned option transition. If the sequence continues, return/GAE
  uses `gamma^duration` to bootstrap at the
  next actor-controlled state; only a true simulator termination bootstraps zero. A deadline miss
  or infeasible scheduled row is not a fake terminal. A bridge segment before the first actor tick
  remains metric-only.

All three arms use the same prebound simulator env-steps and scheduled rows per update, optimizer
update count, actor-controlled sample count, minibatch size and epoch count. B/C surplus actor
samples are selected by a prebound deterministic schedule that cannot read outcomes. If A cannot
produce the fixed actor-sample count, the entire paired update fails: no padding, reuse or extra A
environment steps. Raw actor ticks, bridge ticks, env-steps and opportunities are reported for every
arm. Evaluation always retains every scheduled opportunity regardless of ownership/loss masks.

## Ready And Cross-Engine Boundary

Ready is a conjunction of station, uprightness, low velocity, support/slip, actuator margin,
clearance and next-task/deadline reachability. It is never equality to motion frame 0. The bound
static audit already finds different named stands: Isaac pelvis `(0,0,1.0684)` versus vendor MJCF
`(-0.0416378,0.000359049,1.06839)`, with 31-joint L2 difference `0.171845 rad`. Because Stage-1
contact position is environment-origin absolute while the actor sees target minus current racket
FK, the `4.16 cm` root-x difference does not automatically cancel. This is a hypothesis to isolate,
not a proven cause of engine divergence.

Final decisions use three ordered layers. Isaac is the development/cross-engine precheck. Agibot
vendor MuJoCo Gate3 then hard-gates the exact C++/MJCF/plant/model at first tick and across continuous
stability. Only after that pass may Gate3B use the **same runtime contract** and immutable random-
arrival q50 schedule to score first-strike non-regression and return quality; Gate3B is the final
behavior arbiter. q10 is directional screen only. Any disagreement blocks and triggers root-cause
work rather than averaging scores.

## Prohibited Actions

This runbook does not authorize:

- editing or switching a live training checkout;
- starting, stopping or signalling any Pod process;
- changing the Gate3 worktree or C++ runtime;
- running a real-robot command;
- adding recovery rewards before the A/B/C structural result.
- treating the frame-0 v2 design check as source/runtime/behavior evidence or filling its null
  bindings in place.

若 T1 结构仍失败，现行文档设计要求先在冻结 rollout 上归一化平衡债与 ready-set potential，
跑配对 seed 的 `2^2`。随机到球继续作为环境/题目轴；只有 readiness critic 独立训练/校准、
不泄露未揭题信息，并一次性通过与 sealed Gate3B q50 隔离的 critic-gate q50，才可扩成 `2^3`。
存活项的固定总预算 mixture 还必须补第二个总 reward 量级。安全与 self-contact 始终是硬约束，
不能被其他 reward 抵消。由于本文件绑定的 machine contract 尚未同步，这一段仍不授权 launch。
