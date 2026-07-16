# EXP-P1-TASK-REVISION-CUTOVER

Status: `blocked` — replacement source is implemented and under red-team; no full-scene or behavior
result yet.

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

The paper K100 [`0.5-second timing exam`](../../DEFINITIONS.md#timing-exam-0p5) is materialized at
`configs/phase1_timing_exam_0p5_k100_20260716.json`: 50 questions per side, zero-velocity frame 0,
25 policy ticks at 50 Hz and all attempts in the denominator. It has not been run; 0.5-second
ability is therefore **unknown**.

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

## Acceptance sequence

1. Finish independent ledger and whole-chain producer→wire→C++→training→export red-team.
2. Merge source and synchronized interface/operation/Gate docs to `main` only after the full local
   regression is green.
3. In a clean detached Linux source, run one no-clobber full-scene Isaac probe and verify natural
   exit, finite checkpoint, schema-3 hard contract, exact mixture partition, accepted same-task
   revisions and per-update integer ledger.
4. Run the K100 0.5-second exam and the no-clobber TOPP safety certificate. Record failures as
   failures; do not replace either with training Reward or a fixed clock multiplier.
5. Only then launch the 22 scientifically authorized delay-zero rows from the 24-row, one-seed
   preregistration round-robin across the six GPUs and reactivate one recurring monitor. The two
   delay-two rows remain NO-LAUNCH until coupled transport exists. Pruning uses two complete exact
   outcome windows, never legacy EMA.
6. Vendor MuJoCo Gate3/Gate3B remains the final judge; no real-robot command is allowed by this
   experiment.
