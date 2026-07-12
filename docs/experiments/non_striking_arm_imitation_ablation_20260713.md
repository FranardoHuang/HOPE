# Non-striking-arm imitation ablation

Status: Design
Human owner: Franco
Executor: Codex (design); runtime executor unassigned
Branch: Franco_codex/new-motion-batch-20260713

## Question and decision scope

Test whether removing imitation from the non-striking left arm lets the right-handed A3 use that arm
for balance without reducing strike/return performance or creating unsafe arm motion. This is a
reward-mask question, not permission to remove self-collision, joint-limit, torque or halt guards.

## Inputs and immutable bindings

No checkpoint/config is selected yet. Before launch, bind one accepted motion family, exact
planner/policy/plant/runtime/MJCF, task-matched immutable paper, reward source, body/joint mask,
same seeds and equal transitions. Vendor MuJoCo Gate3/Gate3B is the decision environment; Isaac
curves are diagnostic.

## Design and controls

Stage A separates the direct mask effect from reward-budget reallocation:

- A0: current upper-body imitation baseline (right striking arm plus left non-striking arm); the
  existing lower-body imitation remains off;
- A1: left-arm imitation removed, every other reward weight unchanged;
- A2: left-arm imitation removed and the lost **measured baseline reward magnitude** restored to a
  fixed balance/ready budget, without increasing right-arm imitation.

A1 answers the literal question; A2 asks whether the same shaping budget is better spent on
balance/readiness. The lost magnitude must be estimated once from frozen A0 rollouts, not tuned per
seed. Keep position/orientation/linear/angular imitation masks explicit; do not accidentally free
only orientation and call the whole arm unimitated.

Always-on non-compensable terms remain: joint/torque/action bounds, self-collision, racket/handle to
body/table/net clearance and fall/safe-halt logic. Log left-arm pose/action/jerk and clearance so a
quiet reward cannot hide uncontrolled motion.

Only if A1 or A2 survives does the mask enter the already designed balance-debt/ready-potential
study. Use a small paired `mask x recovery-mixture` interaction first; do not launch a full weight
grid. For any surviving mixture, keep total shaping budget fixed during the ratio simplex and test a
second total-budget level so PPO reward-scale effects are not mistaken for component interaction.

## Acceptance and failure rules

Use paired seeds and checkpoint cadence; no best-seed selection. A candidate needs vendor-MuJoCo
non-inferiority on task-matched return and signed strike geometry, plus improvement in a frozen
balance endpoint such as fall/guard-reset rate, pelvis/torso stability, support/contact margin or
fifth-and-later-shot decay. Any new self-hit, hard-clearance breach, joint/torque violation or
unstable free-arm oscillation rejects the candidate regardless of return score.

Exact denominators and numeric non-inferiority margins remain open until an accepted baseline and
vendor paper are bound; therefore this record stays `Design`.

## Reproduction

No run command exists yet. The eventual preregistration must materialize the left-arm body/joint
mask and prove in a dependency-light unit test that only the intended imitation components change.

## Results

None. No config, training, simulator, Pod or hardware action has run.

## Limitations and claims not made

The free left arm may help angular-momentum control, but imitation removal alone does not provide a
balance objective. Conversely, a balance benefit can interact with recovery/ready shaping, which is
why the direct mask is screened before the small interaction paper.

## Decision and next action

Queue after one task-matched vendor-MuJoCo baseline exists. If the direct mask has no balance gain or
harms return, stop; do not spend GPUs on reward-ratio sweeps.
