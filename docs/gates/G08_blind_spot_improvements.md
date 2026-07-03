# G08 Blind-Spot Improvements

Status: Research track (Phase 2 roadmap recorded 2026-07-03)

## Goal

Improve beyond the HITTER-compatible baseline by targeting weaknesses that can decide matches.

The baseline exists as of 2026-07-02: the unified swing policy transferred to the real A3
(forehand only, scripted targets). Phase 2 is about performance; the reference papers in `papers/`
are the evidence base: HITTER (`2508.21043`), PACE (`2509.21690`, the TTRL-ICRA2026 paper), SMASH
(`2604.01158`), Sony Ace (`s41586-026-10338-5`).

## Phase 2 Performance Roadmap (team, 2026-07)

Each item lists the failure being targeted and the paper-backed mechanism to try first.

### P2.1 Balance across consecutive swings

Failure: the robot falls after several swings; it does not recover weight/posture into a robust
ready stance. Status correction (2026-07-03): the structural machinery already exists on main
(`3eba347`) — 10 s multi-swing episodes, `wrap_teleport=False` (no teleport between swings),
`stand_start_prob=0.25` (deploy-entry resets), pre-swing hold — but no shipped checkpoint was
trained with it (the deployed model predates it, hence the teleport-entry backhand gap). The task
is therefore to train and validate under this structure (P2.1 A/B on branch `p2-multiswing`), not
to build it.

- HITTER: 10 s episodes chain swings; after each swing the next swing type and targets are
  resampled; the base-position reward is active only pre-strike, which *is* the ready phase
  (Sec V-B). Restoring multi-swing episodes is the primary fix.
- PACE: up to 5 consecutive serves per episode + fall termination + residual actions anchored to a
  nominal ready pose → recovery emerges without a dedicated reward.
- SMASH: every reference clip embeds a ~0.54 s recovery segment after contact plus a cyclic strike
  phase variable, making cycles chainable.
- Ace (cheap alternative/complement): train single swings but sample episode initial states from
  the distribution of the policy's own post-swing/recovery states instead of always default-stand.

### P2.2 Reference-motion orientation normalization

Failure: collected swing videos were not rotated to face the incoming-ball direction, so imitated
stance and swing are skewed.

- HITTER: base orientation command is always "face forward"; references are anchored
  pelvis-relative — heading is never inherited from the video.
- SMASH: strike-target features and motion matching are anchor-relative; the racket-orientation
  objective uses `|n·v̂|` (driven by commanded strike velocity, not the clip's facing).
- Action: normalize orientation at the GMR retarget/anchor step and keep the base-heading command
  fixed forward.

### P2.3 Target distribution vs teacher motion consistency

Failure: the sampled target's center (hit position, racket velocity) differs from what the teacher
motion represents, forcing the policy to learn two things at once.

- HITTER: targets are sampled conditioned on swing type in non-overlapping per-swing regions;
  workspace coverage comes from moving the base command, not widening the racket-target box.
- Ace: hindsight experience replay — relabel each transition with the achieved state as the target
  (removes the two-objectives conflict at near-zero cost) + event-table stratified replay to keep
  rare high-quality strikes represented.
- SMASH (systematic solution): expand the clip library (Motion-VAE 400 → 5k), retrieve the
  nearest-reference per sampled target, region-adaptive sampling, and adaptive tracking sigma
  (their ablation: removing adaptive sigma collapses success 86.4 → 22.6).
- PACE: tolerance-clamped tracking rewards (stop rewarding once inside tolerance) preserve
  exploration freedom around the target.

### P2.4 Locomotion aggressiveness / ready motion / clip stitching

Failure: the robot moves too violently toward targets; no preparation motion; needs
target-conditioned stitching/scaling of ready+strike references and eventually rallies.

- HITTER: position-style base commands (with fixed forward heading) produced faster AND calmer
  single-step reaches than velocity commands. Note: the deploy-parity redesign removed the base
  target/reward entirely — with the mocap base pose available at play time, re-adding the base
  command mechanism (as an ablation at minimum) is paper-supported.
- PACE: pseudo-velocity command proportional to remaining position error → smooth deceleration
  into the strike stance; predictor-based proactive targets (reactive control is the root cause of
  late/violent motion).
- SMASH: autoregressive Motion-VAE generates the next reference segment from the robot's *current*
  state (built-in stitching); strike rewards gated to short windows around impact avoid sharp wrist
  accelerations; phase-dependent target noise trains tolerance to late target updates.

### P2.5 Ball-flight physics modeling (landing point, spin, ball-quality rewards)

Failure: no ball model at hit time → cannot reward landing point, pace, angle, or spin; cannot
switch strategy objective (win the point vs sustain the rally). Note: the trained task currently
contains NO ball at all; ball spin is not yet measured by the rig (a patterned/rigid-body ball and
a relay change are prerequisites — see `docs/interfaces/frames_and_coordinates.md`).

- HITTER: drag+gravity flight + diagonal restitution bounce, identified from 15 trajectories; its
  Eq. 5-6 run forward from racket state at contact predict landing/pace — directly a reward.
- PACE (the key reward template): at the instant of paddle contact, physics-predict the landing
  point and net-crossing height and reward them immediately; their ablation shows sparse
  landed/not-landed feedback alone never learns returning.
- SMASH: AEKF with bounce handling (removing bounce handling explodes prediction error
  3.5 → 12.7 cm); analytic inversion of a linearized drag model — chosen flight time T_f is the
  pace knob.
- Ace (ceiling): velocity-dependent Magnus model, data-fit table/racket contact + residual MLP,
  spin measured at 400-700 Hz; policy-sampler modes are exactly the win-vs-rally objective switch.
- In-repo head start: `hope_planner` already has drag+bounce with traj01 calibration, the
  `tasks/table_tennis` Isaac scene has drag/Magnus hooks, and the unmerged branch
  `ball-physics-realistic` (commit `0098c43`) contains mocap-calibrated spin-aware ball physics
  fitted from 300 Hz recordings.

### P2.6 Smash (on top of P2.1-P2.5)

- SMASH's lesson: train smash as a SEPARATE policy with dedicated smash reference data (longer
  execution horizon than the shared strike policy handles). Their smash deployment was limited by
  egocentric-camera FOV during aggressive postures — our external mocap does not have that
  weakness.
- Pace maps analytically: shorter chosen flight time → larger commanded racket velocity
  (HITTER Eq. 5-6 / SMASH Eq. 28-29); the WBC already tracks commanded racket velocity, so smash
  training is target-velocity-distribution + dedicated references + P2.5 rewards.

## Audit-Derived Additions (2026-07-03)

High-leverage items the papers treat as first-class but the roadmap above does not cover:

1. **Latency and target-time-variance modeling.** Training currently assumes a perfect, constant
   target from t=0; the real loop delivers late, jittery, converging targets (SMASH quantifies
   ~10 cm error at 0.6 s-to-strike → ~1 cm at contact and injects tts-decaying target noise; PACE
   injects delays/noise at real-sensor magnitudes; Ace models latency/dropout). Measure end-to-end
   mocap→tick latency and train with target delay + mid-swing re-sampling before the mocap bridge
   lands.
2. **Planner→policy frame-transform ownership.** The 175-D actor is base-position-free, but
   converting a HOPE-world planner target into the robot-relative frame requires the mocap base
   pose at the interface boundary (plus the engage-time yaw alignment). This transform currently
   has no owner and is masked by scripted targets. Design it before/with the mocap bridge (G07
   Next Steps).
3. **Actuator system identification.** PACE's sim-to-real gap (94 → 61%) came from one
   mis-modeled transmission; our implicit/explicit-PD incident is the same class. Per-joint
   step/chirp ID (arms + waist first), fit delay/bandwidth (Ace uses delayed-LTI joints), feed
   both sims. The elbow-at-6.7× -torque-limit trace shows the policy currently exploits actuator
   fictions.
4. **Evaluation infrastructure + real-data flywheel.** Fixed serve corpus (recorded real serves
   replayed in sim), fixed target grids, per-checkpoint scoreboard on `mujoco_eval_onnx.py`,
   mandatory full mocap/obs/action logging for every hardware session (the MDU captures already
   found the yaw drift and the elbow saturation — make it policy), automatic hit detection from
   ball-velocity discontinuities for real hit/return/landing stats.
5. **Reference-motion scale.** 2 clips vs SMASH's 400(+5k generated). The GVHMR→GMR pipeline
   already works; 30-50 orientation-normalized clips across the target workspace is the cheapest
   attack on P2.2/P2.3. MVAE generation can come later (SMASH: 5k→10k was already marginal).
6. **Fall management as a subsystem.** Guard thresholds must be derived from the trained state
   envelope (the 0.6 rad squat-guard incident), take-over should damp rather than snap, and a
   recover-to-ready behavior is the long-term answer (Ace always keeps a verified-safe reset
   trajectory). Fewer falls = more data per hardware session.

## Candidate Tracks (longer-term)

1. Spin perception and spin-aware ball dynamics (P2.5 extension; `ball-physics-realistic` branch).
2. Double-bounce and short-ball handling.
3. Deep-ball and non-fixed hitting-plane handling.
4. Serve generation (Ace: demonstrated toss stitched to an optimized strike).
5. Topspin, backspin, sidespin, block, chop, and loop stroke repertoire.
6. Opponent intent modeling and tactical adaptation (Ace policy sampler; HITTER cites Latte-MV).
7. Multi-agent training.
8. Vision-based perception to reduce mocap dependency (SMASH is the humanoid existence proof).

## Related Directories

- `hope_ws/src/hope_planner`
- `hope_training/whole_body_tracking`
- `agi/A3_MuJoCo_Sim/`
- `external_repos/TTRL-ICRA2026` (PACE's public training code)
- `papers/` — HITTER, PACE, SMASH, Ace
- branch `ball-physics-realistic` (spin-aware ball physics, unmerged)

## Operation Docs

Pick the operation doc based on the selected mini-spec. Common starting points:

- [../operations/run_planner.md](../operations/run_planner.md)
- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)

## Acceptance Criteria

Each track needs its own mini-spec before implementation:

- Failure case being targeted.
- Measurable improvement metric.
- Required data.
- Required simulator changes.
- Deployment risk.
- Owner and expected demo.

## Current State

Done:

- HITTER paper limitations identified: fixed hitting plane, external mocap dependency, ignored
  spin, limited stroke repertoire, no autonomous serving, no opponent adaptation.
- All four reference papers read and mapped to the roadmap (2026-07-03, see above).
- Baseline exists: first sim-to-real of the unified swing policy (2026-07-02, forehand only).
- TTRL/PACE code locally available via `scripts/sync_external_repos.sh`;
  `HOPE-TableTennis-AgibotA3-v0` provides the candidate Isaac scene for ball/serve experiments;
  `ball-physics-realistic` branch holds spin-aware ball physics.

Not done:

- No Phase 2 track has a written mini-spec yet.
- The audit-derived items 1-4 are not yet scheduled; item 2 must land before/with the mocap
  bridge.

## Risks

- Stacking new rewards before restoring the two dropped HITTER structures (multi-swing episodes,
  target-reference consistency) treats symptoms; fix structure first.
- The highest-impact blind spot may shift once mocap-in-the-loop play produces real failure modes.
- TTRL may drift upstream; mini-specs must record the TTRL commit when extracting ideas or code.

## Next Steps

1. Write the P2.1 mini-spec first (multi-swing episodes + initial-state-distribution variant) —
   highest evidence, smallest change.
2. Schedule audit item 2 (frame-transform design) into the G07 mocap-bridge work.
3. Start the P2.5 prerequisite chain: patterned ball + relay orientation forwarding + serve-corpus
   recording, and evaluate the `ball-physics-realistic` branch for merging.
4. Run `scripts/sync_external_repos.sh` before using TTRL/PACE as a reference and record the
   commit.
