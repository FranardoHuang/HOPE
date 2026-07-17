# EXP-P1-TASK-REVISION-CUTOVER

Status: `running / no behavior verdict` — `A6` passed the 4096-env, two-update generic and
task-revision full-scene gates with a finite checkpoint, exact integer ledger and live
final-precontact revisions. All 22 delay-zero cells have now consumed exactly one launch claim:
19 crossed their first training iteration and remain live, while three failed before the first
training iteration in the importer/boot path. The two non-atomic delayed-transport cells remain
NO-LAUNCH. This is runtime/harness acceptance and a live engineering search, not a behavior
winner or 0.5-second return result.

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

`topp_mintime.py` distinguishes total-motion and run-up objectives, prepends a static zero-velocity
frame when required and never lets an oracle bypass CoP/friction/torque gates. On Pod2 the exact
main-matching tool was consumed once, CPU-only and in parallel, against the current v4rg forehand
and backhand paths. Both no-clobber certificates passed finite, production-FK, frame-0-zero,
contact-bitwise, signed-face, joint, CoP, friction and torque gates. The best feasible run-up found
was `0.98 s` forehand and `0.78 s` backhand; neither side obtained a 0.5-second dynamics
certificate. The result is a feasible upper bound in the searched family, not a global optimum or
a proof that 0.5 seconds is physically impossible.

The output root is
`/workspace/codexschema/phase1_task_revision_supercombo_20260716/topp_v4rg_runup_36e42103`.
Forehand NPZ/certificate SHAs are `64f34305…9a6da` / `fe295146…c16`; backhand are
`3a09894b…1f5f7` / `3da9dde9…7531`. Both processes exited naturally with no GPU use or signal.

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

## Activated pool launch result

The activated queue was launched in cross-GPU rounds from the clean source
`b1f5a3803efa14f91e594912959cec2de473a5a6`. A final read-only audit after all launch attempts
recomputed claims, process identity and NVML state rather than trusting launcher output:

- `22/22` launchable cells have one immutable claim; there are no unsubmitted launchable cells;
- `19` are `live_exact`, with numeric `PID=PGID`, matching claim/binding, `/proc` and NVML;
- Pod1 has eight live task-revision trainers distributed `3/3/2`; the independent RallyV11
  process on GPU0 was neither counted nor touched;
- Pod2 has eleven live task-revision trainers distributed `3/4/4`;
- all live logs were fatal-free at the audit boundary and had advanced beyond the first resumed
  iteration;
- three cells are terminal before the first training iteration and have no surviving `/proc` or
  NVML process: `p1_free_non_striking_arm` and `p2_combo_high_noise_free_medium` exited with
  importer malloc `rc134`; `p1_fast_curriculum_high_noise` hit the reviewed boot stale-timeout.

Those three rows are **infrastructure rejections**, not evidence against the free-arm, fast/high-
noise or high-noise combo hypotheses. Their namespaces and logs are preserved and are not
automatically retried. The two positive-delay rows remain intentionally unclaimed because the
governor and actor would observe different revision ticks; they are not idle omissions.

The behavior consumer now binds exact two-window integer ledgers and rejects sparse-zero
misclassification. `+200` can stop only if revision/last-precontact/actor-visible mechanism
counters are absent or zero in the complete windows; `+500` requires the registered clear dense
collapse; `+1000` uses YAML-bound tolerance-aware Pareto dominance. Every stop additionally needs
a no-clobber same-parent portfolio receipt that retains at least two cells plus actual exact-0.5 exposure and
broad-arrival coverage. The public pruning cycle batches all ready checkpoints in at most one SSH
per Pod and never signals by itself.

The first real read-only `+200` cycle found four ready Pod1 `model_1800` checkpoints and four live
rows still waiting; Pod2 had four ready quality-parent `model_4900` checkpoints while its two later
quality rows and all continuous-parent rows were still waiting for their registered milestone.
The three infrastructure-terminal rows were explicitly excluded. No row was stopped or ranked;
the portfolio correctly waits for every still-live sibling rather than rewarding early starters.

The next Pod2 write-side cycle exposed one harness defect before its first behavior receipt: the
atomic writer requires an existing parent, but the consumer had not created the bound run's
`behavior_milestones/` directory. It produced no behavior/portfolio receipt and sent no signal.
The successor creates only that fixed direct child with a single-level `mkdir`, after validating
the bound run directory; missing parents, ordinary files and symlinks fail closed. This is a
consumer infrastructure failure, not a training or Reward result, and the failed invocation is
retained rather than silently replayed.

After that repair entered `main@85ab36df`, one new explicit Pod2 cycle published nine behavior
receipts. All six quality-parent cells had positive revision, last-precontact, actor-visible and
exact-0.5 exposure counts, so the registered +200 rule retained all six; no portfolio elimination
receipt was published and no signal was sent. Three continuous-parent cells were also attested,
while two still-live siblings remained before their checkpoint and one infrastructure-terminal
cell stayed excluded. This is a positive mechanism-activation result, not a behavior winner claim;
the first behavior comparison remains +500.

The subsequent unique Pod2 `+1000` pruning cycle exited naturally with rc0. It published three
checkpoint-bound behavior receipts: `p2_equal_reward@5700` (equal Reward weights),
`p2_no_joint_speed_penalty@5700` (no joint-speed penalty), and
`p2_fast_equal_reward@5500` (fast-arrival equal Reward weights). The other eight live cells were
still waiting for their registered checkpoints, while the existing
`p2_combo_high_noise_free_medium` importer terminal remained explicitly excluded. Both parent
portfolios therefore returned `waiting_for_all_live_cells`: no portfolio receipt, signal, or legal
stop was produced. In plain language, the comparison is not complete yet; zero eliminations does
not mean every cell is good and does not identify a winner.

A later read-only all-Pod `+1000` inspection found that this wait had ended at the checkpoint layer:
every non-infrastructure cell on both Pods had reached its registered checkpoint. Pod1 had eight
checkpoint-ready cells, of which seven were still live and one had exited after writing the
checkpoint; two importer/boot failures remained excluded. Pod2 had eleven checkpoint-ready live
cells and one excluded importer failure. Only the three earlier Pod2 behavior receipts existed;
all remaining ready rows were `checkpoint_ready_behavior_receipt_absent`. The inspector made zero
SSH signals and wrote no receipt. Therefore the next legal action is one no-clobber `+1000`
attestation cycle, followed by portfolio analysis; it is no longer scientifically correct to say
that the pool is waiting for checkpoints, but it is still too early to stop a row before those
behavior and portfolio receipts exist.

The following no-clobber `+1000` attestation was then executed exactly once per Pod. It published
behavior receipts for all 19 checkpoint-ready cells, kept the same three infrastructure terminals
excluded, and sent zero signals. At attestation time Pod2's eleven valid cells had all exited
naturally; on Pod1 only `p1_strong_foot_ready` remained live. Neither parent produced an elimination
receipt. The blocking reason is now evidence, not scheduling: all four ready/balance derived metrics
were `null` because their eligible denominators were zero. The preregistered Pareto consumer correctly
refused to impute them or silently drop those axes. This pool therefore cannot yield a formal winner;
future long runs must prove nonzero ready denominators in the full-scene probe before launch.

## Acceptance sequence

1. Launch the 22 activated delay-zero cells in cross-GPU rounds; keep the two transport cells
   blocked and verify every child resumes full policy/value/optimizer state.
2. At +200 verify structure/mechanism only; at +500/+1000 consume two complete integer windows and
   prune only through the same-parent portfolio receipt, preserving at least two cells and both
   actual exact-0.5 exposure and broad-arrival coverage.
3. Run the K100 0.5-second exam. The first no-clobber TOPP run-up certificates are complete and
   show `0.98/0.78 s`, so the old fixed 0.5-second multiplier is not certified; do not replace the
   missing behavior result with training Reward or reinterpret this upper bound as impossibility.
4. Vendor MuJoCo Gate3/Gate3B remains the final judge; no real-robot command is allowed by this
   experiment.
