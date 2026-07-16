# EXP-P1-TASK-REVISION-CUTOVER

Status: `activation-ready` — `A6` passed the 4096-env, two-update generic and task-revision
full-scene gates with a finite checkpoint, exact integer ledger and live final-precontact revisions.
The 22 delay-zero cells are unblocked; the two non-atomic delayed-transport cells remain NO-LAUNCH.
This is runtime/harness acceptance, not a behavior winner or 0.5-second return result.

- Human owner: Franco
- Executor: Codex
- Date: 2026-07-16
- Scope: simulator and source only; no real-robot command is authorized or was run.

## Problem

The old formal 179-D runner froze the target and countdown after engage, while the old training
pool did not expose same-ball revisions. That choice protected an atomic one-shot command from
mid-swing discontinuities, but it is wrong for a continuously re-estimated ball: the policy can
aim at a stale intercept, and a producer that keeps publishing the old ball can be consumed as a
second action after recovery. The same rolling pool also used timing buckets without a true
0.5-second exam and only historical EMA behavior logs, so it could neither answer the deployment
question nor perform honest behavior pruning.

## Cutover decision and preserved evidence

Before changing the protocol, the recurring automation was paused and both Pods' old rolling pool
was stopped by an exact PID/PGID/starttime/argv/claim/binding/checkpoint verifier. The no-clobber
receipts are:

- Pod1: `e6b2480abfbc8a8b6437c7c8a1baacaac14c6b50537f1c8654eed0a74348263e`
- Pod2: `4c370431b0501ee6ed01e374c059290607499790a757eb12ffd9e5d9948cc949`

The old queue is superseded and must not be resumed. Its final checkpoints remain diagnostic
evidence; they cannot be relabelled as task-revision, 0.5-second or exact-pruning results.

## Replacement contract

1. **One ball, one task.** A physical inbound ball receives one `(control_epoch, task_id)`.
   Re-estimation increments `task_revision`; solver valid/invalid jitter does not allocate another
   task. Contact/plane/deadline closure plus a trustworthy new inbound track allocates the next
   task.
2. **Atomic live target.** Before reference contact, each accepted revision atomically changes
   position, velocity, physical-B normal and TTS. Side and clip remain immutable inside a task.
   Stale, fractional, cross-task, post-contact and out-of-envelope rows fail closed.
3. **Bounded action acceleration.** A checkpoint-bound
   [`phase governor`](../../DEFINITIONS.md#phase-governor) advances reference phase monotonically
   under explicit rate and acceleration limits. It may slow for a later deadline or accelerate for
   an earlier reachable one; an unreachable earlier deadline is rejected rather than silently
   commanding an impossible stroke.
4. **Wide preparation-time curriculum.** Training samples an explicit weighted
   [`initial TTS mixture`](../../DEFINITIONS.md#initial-tts-mixture) spanning sub-0.5 stress,
   exact 0.5, 0.5–0.9 and longer arrivals. The exact 0.5 component is separately counted and is a
   required baseline, not a lower bound.
5. **Truth separation.** The actor sees revisions; the immutable question-bank row, physical ball,
   Reward target and critic target never move. This prevents a convenient new prediction from
   rewriting the question being scored.
6. **Exactly-once behavior ruler.** Every PPO update emits integer opportunity/outcome,
   completion, physical-fall, guard/reset and ready counters. Two disjoint windows are summed as
   integers and ratios are recomputed; sparse zero with zero eligible denominator is never a
   failure. A cross-window close-out counter is required because swing start and outcome can fall
   in different 100-update windows.
7. **Coupled transport or no launch.** `phase_governor_v1` currently permits only
   `target_delay_steps=0`. The older latency ring delays only actor observations, so enabling it
   would let the motion governor react before the actor receives the matching tuple. Positive-delay
   timestamp pairs remain `NO-LAUNCH` until one atomic ring delivers the full revision to both in
   the same tick.

## Timing and TOPP evidence boundaries

The tracked contract for the paper K100
[`0.5-second timing exam`](../../DEFINITIONS.md#timing-exam-0p5) is
`configs/phase1_timing_exam_0p5_k100_20260716.json`: 50 questions per side, zero-velocity frame 0,
25 policy ticks at 50 Hz and all attempts in the denominator. On 2026-07-16 it was consumed once
on each Pod from the same immutable source schedule; both copies are 48,963 bytes with file SHA
`6f5f1526…672d` and semantic SHA `fa7e3c21…3b66`. No checkpoint has taken the exam yet, so
0.5-second ability remains **unknown**.

`topp_mintime.py` now distinguishes total-motion and run-up objectives, prepends a static
zero-velocity frame when required and never lets an oracle bypass CoP/friction/torque gates. Its
reported result is a feasible upper bound in the searched family, not a global optimum. The
current forehand heuristic is about 0.94 s and no backhand final-static certificate or 0.5-second
dynamics certificate exists. This is source progress, not completion of action acceleration.

## Source verification so far

- planner Python suite: `215 passed, 2 skipped` in the local non-ROS environment;
- timing exam/export/TOPP/materializer focused suites: `80 passed`;
- training contract/stage-1 focused suites: `75 passed`;
- dependency-light launch/ledger suites: `14 passed` before the final ledger red-team;
- direct C++ phase-governor/task-revision smoke compiled and ran under Apple clang with the
  repository Eigen headers.

These tests prove internal source semantics only. Local macOS lacks Torch/Isaac, ROS 2 Jazzy,
VRPN and the vendor runtime, so the counts are not an accepted runtime gate.

## Runtime harness findings before acceptance

The first two successor probe attempts stopped before any trainer or iteration. `A2` correctly
rejected the absent private 0.5-second paper. After the paper was materialized, `A3` exposed a
second harness bug: the queue stored `task.planner_revision` as JSON with quoted mapping keys,
which the Hydra override grammar rejects. The queue now stores one canonical Hydra-native typed
mapping directly, and the validator rejects the old JSON spelling; dependency-light queue and
launch regression is `108 passed`.

`A4` then reached a 4096-environment scene, wrote the schema-3 hard contract, instantiated all
three physical-scene entities and entered the first PPO rollout. Before iteration 1 it hit
`IndexKernel.cu:93` on reset. The failure is fully explained by one source invariant violation:
`MotionCommand.submit_planner_revision()` replaced two registered `[num_envs]` command metrics with
the compact `[eligible_envs]` accept/reject tensor. The 0.36-second curriculum rows begin leaving
the eligible set at policy step 18, while the first rollout contains 24 steps; the next partial
reset therefore indexed a shortened tensor with global environment ids. A CPU reconstruction
reproduces the same three failing reset positions (`68/69/70`). The fix preserves both metric
buffers at `[num_envs]`, clears them every policy step and scatters compact decisions through their
original environment ids; a 4096-environment regression includes high global reset ids. `A4` was
classified `stale_timeout/125` after the CUDA cascade, both recorded process ids and the GPU context
are absent, and it produced no checkpoint. This is a failed probe, not a training result or retry
authorization.

`A5` used the fixed metric buffers and reached the first PPO iteration, completed the two-update
probe and wrote a normal-exit marker. The generic finalizer then rejected its schema-3 sidecar
before task-revision activation: `_contract_value()` treated the raw Hydra mapping as an iterable
and serialized `initial_tts_mixture` as `["contract_version", "components"]`. The validator was
correct; `A5` is permanently rejected and cannot be repaired by a receipt. The successor takes the
canonical document from the already validated `InitialTtsMixture` runtime object and, before any
sidecar write or runner creation, validates every newly produced schema-3 structure. It deliberately
does not change the legacy generic converter, preserving planner-OFF contract bytes. The exact A5
malformed shape is now a fail-closed regression. A fresh source commit plus a unique no-clobber
`A6` attempt are still required.

`A6` then passed both no-clobber finalizers. Its model-1 checkpoint contains 74 floating tensors
and 1,762,715 floating elements with zero non-finite values; schema-3 hard contract SHA is
`74d7a884…cf45`, fatal count is zero, and the exact process group plus NVML context are naturally
empty. Across exact update ids 0 and 1 it sampled all four preparation-time components, including
2,406 exact-0.5 samples and 1,772 sub-0.5 stress samples. It recorded 176,387 revision attempts,
165,417 accepts, and 839 accepted/actor-visible revisions on the last precontact interval. The
specialized receipt file/content SHAs are `524b6923…f46c` / `77db7925…d54a`; queue validation now
returns `activation_ready=true`, 22 launchable cells and no blockers. These two short updates prove
mechanism activation only; their one virtual capture and zero legal returns are not a behavior
verdict.

## Acceptance sequence

1. Launch the 22 activated delay-zero cells in cross-GPU rounds; keep the two transport cells
   blocked and verify every child resumes full policy/value/optimizer state.
2. At +200 verify structure/mechanism only; at +500/+1000 consume two complete integer windows and
   prune only under the registered rules, preserving exact-0.5 and stochastic-timing candidates.
3. Run the K100 0.5-second exam and the no-clobber TOPP safety certificate. Record failures as
   failures; do not replace either with training Reward or a fixed clock multiplier.
4. Vendor MuJoCo Gate3/Gate3B remains the final judge; no real-robot command is allowed by this
   experiment.
