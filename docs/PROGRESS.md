# Progress Log

Use this file for short project-state updates that future humans and agents need to see. Keep detailed reasoning in the relevant gate doc.

## 2026-07-11

- Implemented the evaluator-owned schema-v3 same-question path without
  changing the physics-hash-bound `venue_ball_sampler.py`: balanced canonical
  schedule JSON, strict content IDs/per-attempt seeds, Isaac post-reset atomic
  exam injection, all-attempt ledger, and MuJoCo delegation to the common
  10 ms RK4/ball-centre-plane return scorer. Historical checkpoint support is
  an explicit diagnostic-only escape hatch and remains stamped
  `evaluation_contract_exact=false`; formal installation stays fail-closed.
- Added bank hash-before/load/hash-after guards, strict integer runtime IDs and
  tuple/mapping observation normalization after cc's two old evaluators failed
  on the Isaac wrapper tuple. The pre-Pod review also preserved nominal-joint
  startup capture, refreshed racket FK/TTS before action 0, froze command
  resampling, versioned the `H`-stand-actions-then-frame-0 release rule, added
  hold-aware termination-contract v3, explicitly gated legacy train-bank
  loading/hold-guard overrides to historical diagnostics, and made tolerant
  checkpoint loading fail on missing actor keys. Local verification passed 66
  adapter/audit tests with one optional Torch skip, 81 formal CPU tests, and
  136 unique tests in
  the combined run with the same optional skip. Pod same-paper runtime canary
  remains the next acceptance step; no result has been promoted from old
  scorecards.
- Both clean Pod checkouts passed the adapter/audit suite with Torch enabled
  (`64 passed` on each, so scorer parity did not skip). M2's new schema-v3 exam
  bank passed 183/183 forehand and 188/188 backhand Torch closed-loop land/net
  checks. The first q1/side Kit smoke then exposed the expected historical
  motion provenance gap at `gym.make`: old `_cal` files store untagged
  link-origin finite-difference velocity, while current exact MotionCommand
  requires COM velocity. Added a content-detected, inexact-only loader escape
  that preserves the old checkpoint input semantics and leaves formal training
  fail-closed; the failed Kit PGID was terminated cleanly and its GPU/lock were
  released before retry.
- M2 q1 retry 1 passed the new diagnostic motion/v2-bank loaders and exposed
  the next historical pickle field gap (`rally_legacy_metrics`, added after the
  run). Added inexact-only hydration from declared dataclass defaults with a
  complete field list in the nominal profile; exact cells refuse hydration.
  The second failed Kit PGID was also terminated cleanly before the next retry.
- M2 q1 retry 2 completed full environment, command, observation and actor/
  critic construction, then stopped at checkpoint-normalizer validation. The
  historical runner state has four zero `_std` entries, but its
  `EmpiricalNormalization` divides by `std + eps` with finite positive
  `eps=1e-2`, so those entries are valid constants rather than a divide-by-zero
  artifact. The tolerant inference loader now accepts finite non-negative std
  only when the configured epsilon is finite and positive; negative std,
  non-finite values and missing/zero epsilon still fail closed. A dependency-
  light regression covers both the accepted zero-std state and rejected zero
  epsilon. The Kit cell itself remains pending until the identical q1 paper is
  rerun after deploying this change.
- Moved the stale local `main` to `origin/main@ba998c4`, then rebased the
  selective port onto the newer `origin/main@caf4a4e` Gate-3 update before
  publication.  The previously uncommitted Phase-1 work is preserved intact on
  `codex/integrate-local-ablation-20260711@30f4652`; no local work was lost.
- Rejected a mechanical merge of that snapshot into the formal evaluator.  Its
  packed row IDs, per-side sequential cursors and exam-bank injection conflict
  with the current schema-v3 content IDs, immutable schedules and train-split
  command contract.  Mixing the two would either fail at startup or compare
  different questions across cells.
- Selectively ported the historical RunPod evidence documents and three
  simulator-independent utilities: a read-only terminal-checkpoint inventory,
  a fail-closed termination-contract parser and an Isaac-compatible virtual
  return scorer specification.  The latter two remain library-only until the
  evaluator-owned schema-v3 Isaac adapter is implemented; they do not change
  the production MuJoCo BankExam path.
- Local verification passed: 35 standalone utility tests (3 declared skips),
  74 formal BankExam/motion/racket/schema/V5 tests, and 105 planner tests
  (2 optional skips).  A repository-wide whole-body collection is not a valid
  host command because this Mac environment lacks the documented optional
  Torch/Hydra stack; no dependency-only collection error was attributed to
  this change.
- Recorded the next experiments in `NOW.md`: common external questions,
  2.2/3.4 m/s task-speed separation, guarded stroke extension, S1 continuation
  controls, realistic temporal-error controls and split R8 flags.  No training
  or judging job was running, so the next execution step is a small
  M3f/M2/G1 schema-v3 canary rather than bulk historical reruns.

## 2026-07-10

- Post-merge acceptance stayed green after incorporating the latest `main`:
  81 formal geometry/exam/V5 contract tests, 105 planner tests (2 optional
  skips), and 39 RunPod source-only Hydra override tests passed. Removed the
  final strict-finite header whitespace defect caught by repository-wide
  `git diff --check`; no runtime semantics changed.
- Completed the full-stack audit closure and made the formal score path
  fail-closed: BankExam now uses an immutable content-addressed question
  schedule, identical questions across noise columns, one MJCF `stand` reset
  per attempt, all-attempt denominators, exact-strike frame alignment, exam
  split/motion/plant/execution hashes and non-zero exit on evaluation failure.
  Historical scores from the old survivor denominator/late frame are not
  formal and must be rerun.
- Re-audited the racket point from the assets. URDF `pingpang_red_Link`, MuJoCo
  `right_racket`, Python, ONNX and C++ now use wrist-local
  `[0.210210, 0.032078, 0.032036] m`. The red rubber area centre is 1.264 mm
  away. Fixed the material Isaac speed bug: link-origin position had been
  combined with COM velocity, causing about 0.401/0.598 m/s error on the
  audited forehand/backhand poses. Actual speed now uses link-origin velocity
  and `omega x r`; target speed is same-site finite difference. Exact ball
  contact remains a versioned experiment because current banks co-locate ball
  centre and site (20.040 mm red / 33.232 mm black approximation).
- Upgraded motion kinematics to schema 2: every NPZ binds full articulation
  body-column order, link-position/COM-velocity point semantics and per-clip
  FPS. Loaders verify body index/name mapping, equal clip rates and
  `fps=1/env.step_dt`; legacy link-velocity clips require explicit MJCF/body-
  order migration and cannot acquire exact lineage by re-export.
- Closed deploy-side safety races and unsafe defaults: independent deadline
  supervisor, linearized authorization/send/zero barrier, finite/effort/q-des
  envelopes, localization/yaw/re-entry gates, strict CLI and schema-v3/racket-
  point loader checks. A real GCC 13 Release build exposed that global
  `-ffast-math` optimized away NaN/Inf checks; it is removed and compile-time
  strict-finite guards are now mandatory. Controller ACK/timeout and a backend
  call already blocked inside `SendCommand` remain external G07 blockers.
- Corrected the planner table collision plane from ball-centre `z=0` to
  `z=radius`, without inventing an infinite ground plane outside the table.
  Added the V5 professional-transfer/Phase accelerator and preregistered
  teacher, robot retiming, stroke-extension, exact-contact, speed-window and
  venue-spectrum ablations in `NOW.md`; offline feasibility gates reduce each
  side to at most two GPU candidates before paired signal runs.
- Source-only verification used both RunPods without inspecting or controlling
  training: portable C++ Release 188 passed/4 skipped, ROS 2 Jazzy Release 202
  passed/4 skipped, whole-body Python 435 passed/4 optional-asset skips and
  planner 107 passed. Local geometry/motion tests were 22 passed, formal
  BankExam/training/V5 tests 53 passed and planner 105 passed/2 skipped.
  Detailed decisions and boundaries are in
  `research/full_stack_audit_closure_2026-07-10.md` and
  `research/v5_professional_transfer_audit_2026-07-10.md`.

## 2026-07-09

- RunPod pod2 GPU incident diagnosed and recorded in `docs/operations/run_on_runpod.md`: endpoint
  `74.2.96.48` moved from port `16389` to `10473` after restart, and a newly opened endpoint
  `16442` still had the same GPU UUIDs and host boot id with only a fresh container hostname. After
  killing the only visible Isaac/Python training processes, `nvidia-smi` and `pmon` showed no
  running processes and no container process held `/dev/nvidia*` fds, yet all three RTX 5090s
  remained `P1`, 99-100% util, ~575 W, and ~1538 MiB used. GPU reset from inside the container is
  unsupported (`nvidia-smi --gpu-reset -i 0,1,2` returns "not supported"). Conclusion:
  host/GPU-driver no-PID full-load state, not project training load or a visible in-container
  miner. Remedy is fresh host migration or RunPod host-side reset; another pod on the same
  host-local volume can reproduce the same bad allocation.
- Follow-up fresh endpoint `74.2.96.37:14746` was checked: it has a different host boot id,
  different GPU UUIDs, and driver `590.48.01`, but still reports all three RTX 5090s at `P1`,
  99-100% util, ~575 W, and no visible `nvidia-smi`/`pmon` processes or `/dev/nvidia*` fd owners.
  This suggests a broader RunPod host/provider isolation problem on these RTX 5090 nodes, not just
  one stale pod on `74.2.96.48`.

## 2026-07-05

- Strike alignment closed out (yikang-driven round): corrected the RunPod v5 config mistake
  (product/default configs stay on the hopex/registry route; v5 is R15-CLI-only), landed
  `cfg/strike_annotations.yaml` as the contact-phase source of truth with ALL 6 clips
  adjudicated — v5 [0.673 franco / 0.362 claude visual frame scrub of the source video], oblique
  [0.432 (old auto 0.368 was the pre-contact acceleration peak, ~120 ms early; boxes regenerated)
  / 0.495 confirmed], hopex [0.47 / 0.333] KEPT: the source videos (raw_video_hopex/) are DRY
  SWINGS with no ball — the values are the forward-swing speed-peak convention (18/33 cm past the
  x=0.40 plane), and the semantic mismatch vs true-contact clips is filed as R15 decision input.
  `analyze_strike_phase.py` is annotation-first (speed peak demoted to a diagnostic candidate —
  known whip/pull-up trap) and tags GVHMR face normals UNRELIABLE on all video-derived clips incl.
  hopex. `scripts/play.py` resolves local `motion_file`/`motion_file_2` the same way as training
  (R15 replay/export parity).

## 2026-07-04 (day/evening, main) — deploy-parity robustness flags; eval mode B finds the face normal is clip-locked; contract v3 design

- **`motion.clip_switch_prob`** (018467a, default 0.0, try 0.002): deploy-parity MID-swing clip
  switch through the wrap-resample path (random abort → other clip's frame 0 + fresh hold +
  fresh target, robot untouched). Root-caused the venue falls at 准备/正手/反手 switches to
  untrained mid-clip reference jumps (`pp_reference_clock` flips clip_id at arbitrary tts). A8
  capture stays wrap-only.
- **P2.4 `base_decel` reward** (74c129e, default off): PACE-style pre-strike pseudo-speed tracking
  `v_des = clamp(2.0·planar_dist(racket→target), 0, 1.6)` on ‖v_base_xy‖, `std 0.4`, dead at/after
  the strike frame. Mech-verified on/off. v1 is a deliberate proxy — the P-law critique (no fitted
  accel/decel envelope, magnitude-only, no time budget) and the v2 spec live in
  `docs/motion_and_contract_v3.md` §5.
- **train.py override layer hardened** (1181c74 + 74c129e hotfix): `task.motion`/`task.racket`
  yaml keys translate through explicit whitelists; unknown keys RAISE at startup. Lesson recorded
  in run_training.md: a new task-yaml key must extend the whitelist in the same commit (018467a
  briefly broke every task-yaml startup).
- **Eval mode B landed** (f56f9c4): `--target-source venue-balls` — sample fitted venue incoming
  balls (with spin), StrikeSpec-derive the demanded racket state, score the virtual return
  (contact model → drag+Magnus flight → bounds+net). HEADLINE: the policy tracks venue balls
  pos/vel OOD (3.7 cm / 0.18 m/s) **but the face normal is clip-locked** (36-76° err, 0% legal
  returns; counterfactual with the demanded normal: 25/25, median 6.7 cm) — the 175-D contract has
  NO normal-demand channel. Two paths logged in NOW.md (planner-side fixed-normal solve now /
  175→179 contract extension next gen); decision doc: `docs/motion_and_contract_v3.md`.
- **Contract & motion-library v3 design committed** (`docs/motion_and_contract_v3.md`): planner
  output ↔ policy egocentric input table verified against code (finding: `racket_target_vel_w` is
  a WORLD-frame passthrough — only position is egocentric; the deploy wire `RacketCommand.normal`
  already carries the face normal, it dies at `pp_obs_builder` for lack of an obs slot; the critic
  already has privileged `racket_target_normal_w`), 175→179 migration plan, continuous-intensity
  motion library q_ref(φ,ρ) with cost-based selector, PACE-decel v2, 6-clip capture plan.
- **R14 retiming implemented** (evening): `motion.speed_scale_range` — per-swing reference
  playback speed (clock ×s / ref velocities ×s / tts ÷s / target velocity ×s, one consistency
  cascade; train-only, default off). The "can accel/decel modulate stroke amplitude" experiment:
  retiming modulates VELOCITY amplitude; SPATIAL amplitude is the clip-trim axis (R6) — the two
  arms together are the no-new-data continuous-intensity v0.
- **v5 clips processed to npz on the pod** (evening): `hope_{forehand,backhand}_v5.npz` (56/58
  frames, re-grounded) → ablation arm R15. P2.0 REVISED (franco): no dedicated ready video — v5
  first frames agree to 0.15 rad mean (shared ready anchor), last frames 0.24-0.27 rad from first
  (RL fills the gap). CORRECTION (late night, franco): the forehand strike phase is 0.673, NOT
  the detector's 0.768 — the speed peak is the post-contact whip; at the true contact (~2/3, per
  franco) the velocity is (+1.24,+1.21,+1.70) and the face normal healthy, so the earlier
  "+Y-dominant" flag was a mis-pinned phase, not a pipeline direction failure. Lesson recorded:
  speed peak != contact; cross-check with forward-velocity peak / human prior. Remaining real
  issue: v5 reference jitter is 2-6x hopex (mean joint |acc| 5.9/15.5 vs 2.5/2.7 rad/s^2) —
  supports R16 and reference filtering; third confounder for R15 verdicts.

## 2026-07-03 (night, branch `rsi-on-wrap-progress-fix`) — venue data on RunPod; F10/full-state/self-check ran on real data

- **Venue dataset copied to `/workspace/yikang/latest_data`** (extracted npz ×9 + segments + qa + analysis artifacts; the `.tak` files are unused raw Avatar projects). All report-§10 tools now have REAL-data results (report §10.4–10.6 updated):
- **F10**: the two paddles are genuinely DIFFERENT — p1 e(u_n) nearly flat (0.682·exp(−0.0093·u_n)) vs p2 steep (0.975·exp(−0.1093·u_n)), Δg2 CI [0.015,0.201]; at u_n=7 the pooled yaml curve sits 0.08–0.10 e away from either paddle → **identify which physical paddle the robot uses and feed the planner that paddle's e(u_n)**. Face (正反面): p1 unmeasurable (93/7 one-sided face usage); p2 point estimate Δe=+0.054 — right at the "it matters" threshold — but p=0.145 at n=19/26 → UNDERPOWERED; needs ≥50 strikes/face with the robot's racket + a bench note pairing mocap face label to rubber color. Blade position: no detectable trend.
- **Full-state validation (n=82)**: flight model tracks whole arcs — H1 3D position 19→74 mm across 0→450+ ms horizons; velocity 0.24–0.27 m/s ≈ measurement floor; spin error ≈ the 2 rev/s quat floor; net-plane dz p90 49 mm; **H1q ≈ H1** (spin from first 100 ms of flight quats costs nothing → the match-realistic spin source works). H0 through-paddle: 48→269 mm (paddle model form, as §9).
- **NEW `flight_selfcheck.py`** (前推后/后推前): on 192 arcs, forward 55–85 mm / backward 55–108 mm vs matched noise floors 36–57/43–79 mm → **error is ~half variance, half model-form bias in quadrature (excess 42–62 mm, grows with horizon)** — independently reproduces the §9.3 noise-MC model-form estimate by a different method; direction asymmetry within the drag-expected floor (no direction-dependent bias). Heavy-topspin takes under-covered (short/gappy tracks).

## 2026-07-03 (late evening, branch `rsi-on-wrap-progress-fix`) — ball-physics open questions quantified; full-state + paddle-split tools shipped

- **Quantified the team's open questions on the shipped venue fit** (simulation on the fitted constants + MC with the forensics noise model, adversarially reviewed; full numbers in `docs/ball_physics_fit_report.md` §10): spin is tiered (2 rev/s ≈ noise-floor 31 mm; 15 rev/s = 0.1–0.5 m flight + ±0.7 m/s bounce kick; deployed spin-blind planner misses 2–6× racket radius vs 15 rev/s balls, sidespin ~19–31 mm/rev/s, topspin mostly timing, up to 166 ms early); prediction right after the strike vs at net crossing differs ~2× (90 vs 45 mm anchored, ~70% of the gain available from a 150 ms early window; landing-time error 5–11 ms is never the problem); the stack is net-blind and σ_z at the net is 34–64 mm ⇒ clearance <2σ is a coin flip, cord clips (0.06–0.49 m) can pass the pairing gates into H1/H0 tails, and the return-side drag-only net check is 42–84 mm optimistic vs 10–15 rev/s topspin (>30 mm margin) — recommendations in §10.3.
- **Two runnable-on-data tools shipped** (smoke-tested on synthetic trees + adversarial review; NOT yet run on the venue data — that needs yikang's Mac): `predict_check.py --full-state` (position-vs-horizon bins, velocity/spin checkpoints, net-plane crossing state; H1q variant re-estimates spin from the first 100 ms of flight quats since the at-strike quat channel reads ~0.22×) and `falsify/f10_paddle_split.py` (per-paddle / per-FACE / blade-position splits of e(u_n) and a_t; face identity needs the npz body-normal fallback because strikes.json `pad_n` is flipped toward approach; verified to recover a planted 10% face difference).
- **Robot will play with the SAME racket as the venue-capture paddles (team info)** → the F10 face split is deploy-relevant: sensitivity ruler says face Δe 0.02 is ignorable (39 mm return shift), 0.05 matters (~98 mm + net margin), 0.10 forces per-face planner constants. F10's detection floor at n≈82 is ~0.05 — run it next Mac session.

## 2026-07-03 (evening, branch `rsi-on-wrap-progress-fix`) — venue ball-physics fit v1 (Stage 0–5 complete)

- **Processed the 2026-07-03 venue dataset end-to-end and shipped the venue physics fit v1**: `configs/ball_physics_venue.yaml` + `docs/ball_physics_fit_report.md` + the full pipeline in `hope_training/ball_physics_fit/` (Stage 0–4 scripts, the F1–F8 falsification battery under `falsify/` + `stage3_falsify.py`, `predict_check.py`, `requirements-ballfit.txt`; paths via `BALLFIT_DATA_ROOT`; `test_oracle_present.py` fails loudly). 9 C3D takes @300 Hz → 1286 flights / 218 bounces / **154 racket strikes (first-ever racket data)**. QA PASS (g=9.825 @0.15°).
- **Constants (taped 3.4 g ball)**: k_d 0.1261 (C_d 0.569), k_m 0.00444 (inside old-rig CI → quat spin scale validated), table e_n **0.9215** (a +0.010 contact-time bias was found by forensics and FIXED in stage1 — e>1 tail 17%→4%, F3 upgraded to PASS-in-coverage), paddle first real fit: e(u_n)=0.759·exp(−0.0441·u_n) (F4 KILL, adversarially verified), a_t 0.52 velocity-channel (strike spin channel reads 0.22× → joint fits untrustworthy), μ_safety 0.5.
- **Falsification**: F4 KILL (velocity-dependent paddle e; needs ~3-line spin_contact.py + C++ change), F2 INCONCLUSIVE by coverage (SR>0.7 effectively empty — Magnus saturation stays OPEN; recommend the saturating C_L form), F1/F6 PASS, F3 PASS after fix, F5/F8 noise-limited, F7 borderline.
- **Honest validation vs the shipped yaml** (observed-bounce/terminal-window landing ground truth): bounce speed **2.32% PASS** (was 5.96% pre-fix), direction 4.6° (>3° bar, noise-limited), flight 77/147 mm vs 25/60 bar. Two-horizon `predict_check` (n=82): **H2 ~100 ms before landing = 5 mm**, H1 at-contact from measured out-state = 67 mm, H0 through the paddle model = **0.25 m median** — the flight model is excellent; the at-strike ceiling is paddle model FORM + racket-state (noise-floor MC: only 4–15% of the landing miss is noise). Deploy freebies: 120 ms init windows (flight 76→~60 mm), colored-noise observer tuning (rig = 1.9 mm white + AR(1) ρ≈0.94).
- Next data wants: heavy-spin serves (SR 1–4), shallow bounces, denser strike-zone cameras, ITTF drop test on the venue table.

## 2026-07-03 (branch `rsi-on-wrap-progress-fix`)

- **Merged `main` into `rsi-on-wrap-progress-fix` (union merge).** New from main, kept in full: the p2 fixed-protocol sim2sim scoreboard (`scoreboard_eval.py` + the G06 harness) with the `deploy_faithful_implicit` fourth protocol, in-distribution per-clip eval boxes in `mujoco_eval_onnx.py`, the `train.py` fix that honors `motion_file` again (no-WandB local path) and the `play.py` `_wbt_tasks` NameError fix, portable eval/export launchers, ONNX-metadata strike phases and the widened relay pose-topic forms, the git-lfs normalization, `docs/operations/run_on_runpod.md`, the `NOW.md` work board + ablation results, and the new reference papers. Branch invariants preserved through the merge: motion source stays local-first (`motion_file=` / `motion_file_2=` bypass the WandB registry entirely; local `.npz` paths passed as `registry_name`/`registry_name_2` remain back-compat), wrap teleport is controlled solely by main's `wrap_teleport` knob (`rsi_on_wrap` stays removed), and the restored `a3_runtime` vendor CSVs and analysis PNGs remain real content.
- **Branch reconciliation cleanup (pre-push).** (1) Restored the 13 `agi/code_deployment/.../a3_runtime` vendor CSVs that c7733db had silently rewritten into git-LFS pointers (the vendored `.gitattributes` declares `*.csv filter=lfs` and this clone has the lfs filter active; main stores the same bytes as real blobs) — those files are now byte-identical to main again, and a local-only `.git/info/attributes` guard prevents re-pointerization on this clone. (2) Dropped this branch's `rsi_on_wrap` knob in favor of main's equivalent `wrap_teleport` (from 3eba347): both flags independently implemented "no RSI teleport at clip wrap", and the 478f1a2 merge had left a teleport requiring BOTH to be true. Effective behavior is unchanged (wraps never teleport under every in-repo config). `train.py`'s `motion:` plumbing now exposes the Phase-A swing-entry knobs (`wrap_teleport`, `stand_start_prob`, `hold_steps_range`, `stand_start_min_hold`), matching franco's `p2-multiswing` commit 087e304 so the eventual merge back is clean; `HOPEPingPongDeployParity.yaml` documents those defaults and `HOPEPingPongRealSensor.yaml` now purely inherits it.

## 2026-07-03 (later, branch `audit-leftover-fixes`)

- Fixed the audit's leftover technical items: `eval_realsensor_hopex.sh` / `export_onnx_explicitpd.sh` are now portable (self-locating + `HOPE_EVAL_*`/`HOPE_EXPORT_*` env overrides instead of `/home/dongc1/...` paths); `mujoco_eval_onnx.py` now resolves `strike_phase_per_clip` as CLI > ONNX `clip_strike_phases` metadata > built-in legacy fallback (same keys the C++ runner uses) and warns when the motion npz `seg_len` disagrees with the ONNX `clip_seg_lengths`; the VRPN relay topic matcher now accepts all observed pose-topic forms (`pose`, `pose<idx>`, `pose_id_<idx>`, `pose_<idx>`), covering the vendored tracker.hpp naming.
- Normalized the git-lfs mismatch: the repo has NO actual LFS objects, yet vendor/IsaacLab-copied `.gitattributes` marked 13 `a3_runtime/*.csv` + 5 `analysis/*.png` raw blobs as lfs, making them show permanently "modified" on lfs-configured machines (found via the RunPod smoke). Added deepest-level `.gitattributes` unset rules for exactly those trees.
- Hotfixed `main` (`8d2af53`): the simtoreal2 merge had left raw conflict markers in `my_on_policy_runner.py`, breaking every training import — caught by the RunPod train smoke. Also provisioned/verified the shared RunPod (see `docs/operations/run_on_runpod.md`): franco's clone synced + git-lfs configured locally, full smoke incl. 10-iteration training run.

## 2026-07-03

- Ran a full doc-vs-implementation audit of `main` at `0bc9c53` (multi-agent, 68 verified findings) plus a structured read of the four reference papers now in `papers/` (HITTER `2508.21043`, PACE `2509.21690` = the TTRL-ICRA2026 paper, SMASH `2604.01158`, Sony Ace `s41586-026-10338-5`). Realigned the docs that had gone stale after the simtoreal2 merge: gate statuses (G06/G07 → Partial), the actor observation contract (175-D deploy-parity), the deploy action path (31-act pingpong runner), the mocap interface contract (play vs data-collection phases), PROJECT_MAP zones, operations runbooks, and `reimplement.md` steps 13-18. Note: merge `73297db` took "docs from main" and silently clobbered the a197bde doc updates (G07, policy_observation_action, run_deploy_dryrun); this pass restored and extended them.
- Recorded the mocap runtime contract from the team (see `docs/interfaces/frames_and_coordinates.md`): ChingMu/VRPN; during PLAY the rig streams at 300 Hz robot base (pelvis) pose + ball position; ball rotation/spin is planned for the physics-modeling phase (not yet measured; the relay currently publishes the ball as a position-only point); the data-collection/physics-calibration phase additionally captures racket pose, table 4 corners, and net 2 corners (the racket-tracking prohibition is a play/competition rule). Aligned the VRPN bridge launch default `update_freq` to 300 Hz.
- Pinned deploy facts that were undocumented: the C++ runner currently consumes NO mocap topics (targets scripted; the ROS chain vrpn_mocap → relay → hope_planner → /racket/command → hope_wbc_runner exists but is not bridged into the runner); deployment is FOREHAND-ONLY (backhand has a stand-entry training gap); the shipped policy is `model_p4_deployparity.onnx` (175-D / 31-act).
- Wrote the Phase 2 performance roadmap into G08: the five team items (multi-swing balance, reference-orientation normalization, target-reference consistency, locomotion aggressiveness/ready motion, ball-flight physics modeling incl. spin) plus audit-derived additions (latency/target-time-variance modeling, planner→policy frame-transform ownership, actuator system ID, evaluation infrastructure/data flywheel, reference-motion scale, fall management), each mapped to the paper evidence.

## 2026-07-02

- **Merged `main` (unified-swing uniform-target redesign + simtoreal2 deploy fixes) into `rsi-on-wrap-progress-fix`.** Post-merge defaults follow MAIN's training design: `target_mode: uniform` (fixed hit-plane sampling), WandB-registry motion defaults with the local-first `motion_file=` / `motion_file_2=` overrides, main's swing accounting (`swing_completion_rate`, `pre_strike_fall_rate`), pre-swing HOLD + stand-start resets, and the reworked auto-detecting `setup_train_env.sh`. Preserved from this branch: `rsi_on_wrap` (kept as the HOPE YAML knob; teleporting at wrap now needs BOTH `wrap_teleport=True` AND `rsi_on_wrap=True`), the per-clip `reference_perturbed` upgrade incl. the live-verified `motion` hoist fix (reference_perturbed stays available but non-default), the `racket_progress` resample-spike fix, `csv_to_npz.py` HOPE +X alignment (+ `hope_frame_utils.py`, `check_motion_target_alignment.py`), and the 2026-07-02 verification records. Note: upload is now opt-in via `--upload_wandb` (the branch's `--no_upload` flag is gone), and main's committed `my_on_policy_runner.py` contained unresolved conflict markers (broken on main) — repaired here by synthesizing both sides. Earlier entries below describing `reference_perturbed`/local-clip YAML defaults refer to the PRE-merge branch configuration.
- **MP4 -> training pipeline verified end-to-end and two training-blocking bugs fixed**. (1) Fixed `UnboundLocalError` in `RacketTargetCommand._resample_command` (`hope_commands.py` base-XY coupling branch used `motion` before assignment) — this crashed every `HOPEPingPong*` env reset, so no training could start on the working tree. (2) Fixed `train.py` so the `registry_name_2='none'` disable sentinel is normalized to `None`; previously it leaked into `runner_registry_names` as `none:latest` and `wandb.run.use_artifact` killed single-clip wandb runs at the first checkpoint save. Verification: fresh isolated rerun of forehand MP4 -> GVHMR -> GMR -> ground -> CSV -> `csv_to_npz` reproduces shipped artifacts bit-for-bit (pkl/CSV identical; npz within 2e-7); `check_motion_target_alignment.py` passes; post-fix smoke (32 envs x 2 iters) and a bounded 4096-env x 300-iter run passed — W&B `wuj6ds9u`, reward -1.37 -> ~25, episode length 5 -> ~340, checkpoints + ONNX synced. Env caveat now documented in `setup_environments.md` + `reimplement.md`: GVHMR needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` on torch>=2.6 (ultralytics 8.2.42 YOLO load). Also noted for wandb hygiene: the motions registry version numbers are independent from `csv_to_npz` project versions (today's registry v5 == project v4 == the +Y-facing rejected clip; registry links v3-v5 were later deleted, `latest` is back to the 2026-06-30 v2), and the corrected `_hopex` clips remain local-only.

- **HOPE +X motion and teacher-centered target fix**: the local/WandB `hope_forehand:v3/v4` and `hope_backhand:v3/v4` motions were verified to still face world +Y (`frame0_yaw` 82.03/85.92 deg; strike velocity Y-dominant), so they are not accepted for long training despite the earlier registry smoke. Corrected local ignored clips were generated at `hope_training/motions/preprocessed/hope_forehand_hopex.npz` and `hope_backhand_hopex.npz`, both with `frame0_yaw=0` and +X-dominant strike velocity. `scripts/csv_to_npz.py --robot agibot_a3` now applies HOPE +X alignment by default before local save/upload, `HOPEPingPong*.yaml` default to the corrected local clips, and `racket.target_mode` is now `reference_perturbed` with per-clip reference strike centers plus success-gated long-run perturbation. `scripts/check_motion_target_alignment.py` passes on both task YAMLs and fails on the old v4 artifacts as intended.
- **WandB motion registry and training smoke verified**: the generated `hope_forehand.npz` and `hope_backhand.npz` clips were uploaded to the org-scoped motions registry. `hope_forehand:latest` and `hope_backhand:latest` resolve to `BerkeleyPingPong/csv_to_npz/hope_forehand:v4` and `BerkeleyPingPong/csv_to_npz/hope_backhand:v4`; both manifests contain canonical `motion.npz`. `scripts/train.py` now calls `wandb.finish()` before Isaac `simulation_app.close()`, and the registry-backed `HOPEPingPong` W&B smoke run finished cleanly: https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga. It used both motion artifacts and synced `model_0.pt`, ONNX, config, diff, and logs from `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-07-02_11-56-04_smoke_registry_wandb_finish`. This is pipeline verification, not a quality baseline.
- **RunPod motion-processing environment provisioned for GVHMR -> GMR -> CSV**: installed Miniforge under `/workspace/yikang/miniforge3` and created independent Conda env `hope-motion-py310` (Python 3.10), separate from `/workspace/hope_isaac_venv` / Isaac Lab. The env uses `torch==2.7.0+cu128` / `torchvision==0.22.0+cu128` for RTX 5090 (`sm_120`), builds `pytorch3d==0.7.9` from source with CUDA 12.8, and installs GVHMR (`6ec3ca3`) + GMR (`bb1bbe4`) editable. GVHMR non-body checkpoints were restored under `hope_training/GVHMR/inputs/checkpoints/` from the public Hugging Face mirror after Google Drive quota blocked `gdown`: `dpvo`, `gvhmr`, `hmr2`, `vitpose`, and `yolo` are present and hash-checked. GMR now has local ignored `agibot_a3` support in its clone (`assets/agibot_a3/a3_mocap.xml`, `smplx_to_a3.json`, registration, and CLI choice); the generated MJCF is verified as free-root + 31 hinge joints matching `config/joint_order_agibot_a3.yaml`. The current RunPod has the required license-gated body-model files for the completed forehand/backhand MP4 -> motion run; new machines must restore them manually because they are not redistributable.
- **Continuous multi-swing training fix**: HOPE ping-pong now exposes `MotionCommandCfg.rsi_on_wrap` and both `HOPEPingPong*.yaml` set it to `false`, so clip wrap no longer teleports the robot mid-episode. `racket_progress` now emits zero and resets its previous-distance baseline on motion/target resample steps, removing the wrap/reset spike from the base-free footwork reward. Target sampling now starts teacher-centered and widens through the `reference_perturbed` curriculum rather than using the old active uniform per-clip boxes.
- **Unified forehand+backhand training YAMLs synchronized**: `HOPEPingPong.yaml` and `HOPEPingPongRealSensor.yaml` now both describe the default route as one unified policy (`registry_name` forehand + `registry_name_2` backhand + `swing_type`) and remove stale single-swing / old backhand positive-y/high-z coordinate notes. The current per-clip boxes remain centered on the re-grounded HOPE +X strike points: forehand `(0.48,-0.38,0.87)`, backhand `(0.52,-0.04,1.05)`. The no-Isaac per-clip position/velocity sampling check scripts now use the same reference points and speed intent.

## 2026-07-02 (recorded 2026-07-03)

- First sim-to-real success: the unified swing policy was deployed on the real A3 via Route A (`agi/a3_deploy_example` C++ runner `a3_deploy_onnx_ref_pingpong`, reusing AGI `A3AimrtBackend`/`A3PolicyDriver` unmodified) with real behavior matching MuJoCo. Deployed artifact: `model_p4_deployparity.onnx` — 175-D deploy-parity actor obs / 31 actions / 50 Hz, forehand only. Merged as `merge simtoreal2 to main` (`0bc9c53`).
- The s2r observation alignment is commit `a197bde` "new observation design": the deploy-parity actor obs (175-D) removes `motion_anchor_pos_b` and `base_target_pos_b` and reframes `racket_target_pos_b` racket-FK-relative so no world base position is consumed; verified numerically by `scripts/realsensor_obs_reference.py` / `scripts/verify_realsensor.py` and mirrored by `scripts/mujoco_eval_onnx.py` and the C++ `pp_obs_builder`. Training default switched to `task=HOPEPingPongDeployParity` (`HOPEPingPongRealSensor` is an alias; 180-D `HOPEPingPong` kept as legacy comparison).
- Deploy-chain fixes from real bring-up (`d17cd57`/`dd32a29`): exporter zero-kp/kd guard, clip clock read from ONNX metadata (fixes the stale-clock bug that fired the forehand ~0.6 s early), squat guard raised 0.6 → 1.4 rad plus 0.35 tilt guard (the old threshold sat inside the trained swing envelope and its snap-back catapulted the robot), and IMU yaw-align capture at engage (boot-to-boot yaw drift 130-165°).
- Sim-fidelity finding fed back to Agibot in a one-off letter (deleted after delivery; permanent record: `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md`): the AGI deploy MuJoCo applies explicit-Euler PD while the real backend and Isaac are implicit; the same ONNX is stable under implicit PD and diverges in ~0.1 s under the deploy sim's explicit path; switching only the PD integration moved hit-speed error 0.61 → 0.31 m/s and velocity attainment 0.35 → 0.88. Follow-up: `launch_explicitpd_ft.sh` fine-tunes under explicit clipped PD with ±15% PD-gain DR, and the AGI explicit-PD MuJoCo run is now the binding pre-hardware gate.

## 2026-07-01

- Pulled `main` from `8a4ae7f` to `4ff5f52` (`Merge train_1 into main: unified swing training and HITTER alignment`) and audited the training/documentation state. This was an audit only; no Isaac/ROS gate was newly verified.
- Found a G05 training blocker in current `main`: `pyflakes hope_training/whole_body_tracking/scripts/train.py` reports `undefined name 'registry_for_runner'` at the runner construction site, so the Hydra training entry will fail after env setup unless this is fixed.
- Found a reproducibility regression relative to the 2026-06-26 docs: `cfg/train.yaml` and docs still advertise `motion_file=<local.npz>` as a no-WandB training override, but current `scripts/train.py` ignores `cfg.motion_file` and always downloads the primary clip from the WandB registry.
- Recorded that G05 docs were stale after the unified HITTER merge: code/config now default to forehand+backhand unified training with `registry_name_2`, `strike_phase_per_clip`, `target_mode: uniform`, actor `swing_type`, and no actor racket-normal observation, while several docs still described one-policy-per-swing `reference_perturbed` curriculum training.
- Recorded non-training drift: `setup_train_env.sh` now contains fixed `/workspace/...` paths and real WandB identity defaults instead of the documented local-override placeholders, and `avatar_pro_vrpn.yaml` now defaults P1/P2 input rigid-body names to `ppp2`/`ppp3` while live mocap verification is still required.
- Restored a first-class local Step 9-12 motion path: `scripts/train.py` now accepts `motion_file=<forehand.npz>` plus optional `motion_file_2=<backhand.npz>` (or list-valued `motion_file`) and skips WandB entirely when local files are set; `registry_for_runner` is defined only for registry-backed runs. `scripts/play.py` and `scripts/replay_npz.py` also accept local motion files.
- Restored `setup_train_env.sh` to the portable contract: source-first working tree path, optional `setup_train_env.local.sh`, overridable `HOPE_ISAAC_*` paths, placeholder WandB defaults, and auto-detection for the current `/workspace/...` Isaac/IsaacLab layout when present.
- Updated the training runbooks to make the video-to-motion workflow local-first: `reimplement.md` Step 12 now writes and replays local `motions/preprocessed/*.npz` files before any optional `--upload_wandb`, and Step 14 / `run_training.md` use `motion_file` + `motion_file_2` for local unified training.
- Final-aligned the documentation entry points for the video-to-motion workflow: root `README.md`, `docs/START_HERE.md`, `run_training.md`, `setup_environments.md`, `setup_local_sync.md`, `whole_body_tracking/README.md`, and `reimplement.md` now all point to the same local-first Step 9-12 flow, with WandB explicitly optional.
- Updated the exact-strike probe to share the same local/registry motion resolver and to use the command term's per-clip `time_to_strike`, avoiding a stale single-`strike_phase` assumption for unified forehand/backhand policies.
- Removed tracked temporary training artifacts: `hope_training/cmp.py`, `reward_share.py`, `scan2.py`, and `whole_body_tracking/.hitter_align_backup/*.bak`; added `.hitter_align_backup/` to `.gitignore`.

## 2026-06-26

- Clarified setup documentation after the selective backport: new machines should start from `docs/START_HERE.md` and the relevant `docs/operations/*` doc, while `reimplement.md` is only supplemental historical command detail when a gate/operation doc cites a specific step. Also corrected the README local no-WandB smoke command to include an explicit `motion_file=...`; internal full training remains WandB-default.
- Merged `origin/train_1` into `main` (3 commits: strike-velocity fix, forehand/backhand split, doc update). Auto-merge resolved cleanly; the two overlapping files (`scripts/train.py`, `cfg/train.yaml`) combined the local-`motion_file` override with train_1's additions without conflict, verified by `py_compile` + YAML load. New from train_1: per-clip strike phase (`strike_phase_by_motion`: `hope_forehand: 0.46` / `hope_backhand: 0.59`, replacing the wrong shared 0.46 that stalled the backhand), `clean_reference_strike_velocity` (denoise the strike target velocity via a centered finite difference instead of the ~1 m/s-noisy stored `body_lin_vel_w`), `episode_length_s: 3.0` (one swing per episode), `motion_scale: 0.50`, and a `checkpoint_path` resume knob in `scripts/train.py` / `cfg/train.yaml` for curriculum hand-off.
- Ran the first end-to-end Isaac WBC training loop on this machine (G05 first-loop now reproduced here, not just inherited from `reimplement.md`). Task `TrackingFlat`, `num_envs=1024`, 60 iterations, `logger=tensorboard`, `run_name=stand_bootstrap`. Mean reward improved monotonically `-4.08 -> -0.24`. Artifacts: `logs/rsl_rl/agibot_a3_flat/2026-06-26_13-13-07_stand_bootstrap/{model_0,25,50,59}.pt` and `exported/policy.onnx`. These are pipeline-viability artifacts, NOT an accepted quality baseline (the reference is a static stand, not a real swing).
- Fixed the real G05 training blocker: the **RTX 5090 is Blackwell (sm_120)** and Isaac Sim 4.5.0's bundled `torch 2.5.1+cu124` has no sm_120 kernels (a real CUDA matmul failed with `no kernel image is available for execution on the device`; this is the deeper blocker the EULA note in G05 hid). Upgraded `hope-isaac-py310` to `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` (sm_120 kernels) and pinned `numpy==1.26.4` (Isaac Sim 4.5 needs `numpy<2`). After the fix Isaac Kit boots on the 5090 and a CUDA matmul succeeds. Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`.
- Accepted the NVIDIA Omniverse EULA non-interactively via `OMNI_KIT_ACCEPT_EULA=YES` (the remaining gate from the 2026-06-25 G05 note).
- Made training runnable without a WandB account/registry: `scripts/train.py` previously fetched the motion clip ONLY from the WandB registry (`wandb.Api().artifact(...)`). Added a local `motion_file=` override (mirrors `scripts/play.py`) that skips WandB when set, plus `motion_file: null` in `cfg/train.yaml`. `registry_name` is only consumed by the runner under `logger=wandb`, so the local/tensorboard path needs no WandB identity.
- Added `scripts/make_static_motion.py`: generates a valid BeyondMimic motion `.npz` (a static "stand at default pose" A3 clip) without the GMR/GVHMR retargeting pipeline or WandB, so the WBC loop can run with no motion data. Output `hope_training/motions/a3_stand.npz` (`fps=50`, `joint_pos[600,31]`, `body_pos_w[600,32,3]`, zero velocities). Schema matches `scripts/csv_to_npz.py`. This is a placeholder reference for first-loop/pipeline proof only.
- Created the `hope-motion-py310` conda env (`python=3.10`) and accepted the conda default-channel ToS; GMR/GVHMR clones from 2026-06-25 remain in place.
- Selectively backported internal-safe engineering fixes from the public starter work while preserving main's gate/progress workflow: internal training still defaults to WandB registry/logging, `motion_file=...` can explicitly override registry reads for local/no-WandB runs, `csv_to_npz.py` saves local `.npz` files by default and uploads only with `--upload_wandb`, and docs now describe both internal-default and local-smoke paths.
- Added `scripts/prepare_a3_isaac_asset.py` plus a tracked `whole_body_tracking.assets` path helper; the generated A3 Isaac asset remains ignored and is rebuilt from the tracked Agibot ping-pong URDF package. Asset policy now distinguishes ping-pong URDF source, standard non-racket A3 source, MuJoCo/AimRT materials, generated Isaac copies, local-only artifacts, and third-party notices.
- Fixed planner return consistency and tests: Stage 3 outgoing velocity/net clearance now use the same quadratic-drag-plus-gravity free-flight model as Stage 2, racket normals are guarded to face HOPE `+x`, `/racket/command` uses reliable keep-last depth 10 QoS, and planner tests now cover degenerate normals, drag landing, bounce-then-cross, and quaternion local-`+x` alignment.
- Hardened `RacketTargetCommand`'s current MotionLoader private-field coupling behind `_reference_body_state(...)` and fixed the table-tennis geometry harness so ball-aerodynamics failures affect the process exit code while torch-unavailable hosts still skip that subtest.

## 2026-06-25

- Removed the obsolete root `Dockerfile.hope-ros2-jazzy` and updated README, setup/build operations, G00, and `reimplement.md` so ROS work points to the distrobox or otherwise pre-provisioned ROS 2 Jazzy environment instead of the old Docker path.
- Continued environment setup using the `reimplement.md` names `hope` and `grasping`: installed user-local `distrobox 1.8.2.5` and `lilipod v0.0.3`, created `~/workspace/HOPE -> ~/workspace/nohope`, pulled the `osrf/ros:jazzy-desktop-full` image, and confirmed the host GPU is visible (`RTX 5090`, driver `570.153.02`, CUDA `12.8`). Creating the `hope` distrobox is blocked until host `podman uidmap` or equivalent rootless mapping helpers are installed with sudo.
- Unblocked rootless containers after local sudo access was provided: installed host `podman` and `uidmap`, created the `hope` distrobox from `docker.io/osrf/ros:jazzy-desktop-full`, verified `ROS_DISTRO=jazzy` and `colcon`, created the NVIDIA-enabled `grasping` distrobox from `docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, and verified it sees the RTX 5090 plus `nvcc`.
- Built an ignored local Isaac training Python env `hope-isaac-py310` with `torch 2.5.1+cu124`, `isaacsim-rl 4.5.0.0`, `isaacsim-app 4.5.0.0`, editable Isaac Lab source from `external_repos/IsaacLab` tag `v2.1.0` (`21f7136`), editable `whole_body_tracking`, and `setup_train_env.local.sh` pointing `hope_isaac_py` at that env. Python checks pass from inside `grasping`; the first Kit/training launch is paused until the NVIDIA Omniverse EULA is explicitly accepted.
- Restored ignored motion repos: `hope_training/GMR` at `bb1bbe4` and `hope_training/GVHMR` at `6ec3ca3`. Installed GMR editable into `hope-motion-py310` and verified imports (`torch 2.12.1+cu130`, `mujoco 3.10.0`); GVHMR was cloned but not installed because its requirements pin CUDA 12.1-era wheels that need a Blackwell compatibility pass.
- Checked the G05 training environment in this harness. Restored the ignored package-local A3 Isaac asset from tracked Agibot URDF materials, verified `86` mesh references with `0` missing, and passed the host table-tennis geometry test (`6/6`; torch aerodynamics skipped because host torch is unavailable).
- Clarified the new-computer/agent startup index in `README.md` and `docs/START_HERE.md`: start from `docs/START_HERE.md`, use `docs/operations/run_training.md` for Isaac training setup, use `docs/operations/setup_local_sync.md` for ignored/private assets, and treat `reimplement.md` as a long-form runbook only when referenced by a gate or operation doc.
- Integrated the latest `origin/train_1` updates on `main` while keeping the main-branch docs structure: training code/config now follows the PATH B/C slower-hit defaults (`ref_vel_scale: 0.6`, narrower reference perturbations, updated reward stds/weights), table-tennis physics adopts Purdue PACE material defaults and a tracked visual USD overlay, and the docs now point new training machines through `docs/START_HERE.md` -> `docs/operations/run_training.md` / `setup_local_sync.md` rather than using `reimplement.md` as the primary entry point.
- Merged `origin/train_1` into `main` and updated the docs/reimplementation rhythm so `reimplement.md` is gate-indexed rather than a separate phase plan. G04 now records the new `HOPE-TableTennis-AgibotA3-v0` Isaac scene, and G05 records the updated `HOPEPingPong` exact-strike metrics, reward/target defaults, source-provenance logging, and success-gated reference perturbation.
- Preserved source/config/test changes while excluding generated or local artifacts (`.codex-tmp/`, `.vscode/`, generated `*.onnx`) under the asset policy. Scrubbed private training paths and WandB identities from `reimplement.md`; machine-specific values belong in `setup_train_env.local.sh`.

## 2026-06-24

- Integrated `origin/jiayi` commit `c951d9d` into this harness branch while preserving the open-source placeholders and A3 support-material docs. The branch now includes `reference_perturbed` racket target sampling for `HOPEPingPong`, training override plumbing for the new parameters, and Avatar-Pro relay `ball_tracking_mode` support (`rigid_body` preferred, `auto` fallback).

## 2026-06-23

- Open-source documentation pass for this branch: rewrote `README.md` (About-this-branch, Repository Layout, Quickstart, Assumptions & Limitations; corrected A3 31/29-DOF and ChingMu/VRPN framing), indexed `reimplement.md` from `START_HERE`, and filled env-setup gaps across `docs/operations/*` (environment creation, GMR/GVHMR + SMPL-X/checkpoint restore, WandB team-vs-org + offline `logger=tensorboard`, the `Live/...` telemetry and the reward-std `strike_success=0` fix) and `docs/interfaces/*` (31-DOF list, 31-vs-29 deploy, real actor/critic observations + action dims, world-Z frame, RacketCommand fields).
- Scrubbed maintainer-private values for public release: `setup_train_env.sh` now reads overridable `HOPE_ISAAC_*` paths and placeholder WandB identity (real values via a git-ignored `setup_train_env.local.sh`); added `ros-jazzy-vrpn-mocap` + `python3-vcstool` to `Dockerfile.hope-ros2-jazzy`.
- Integrated A3 Isaac/BeyondMimic training updates (through commit `42489cd`): working joint-order YAML, updated A3 robot config, deploy-transcribed PD/action-scale values, HOPEPingPong task updates, train/play fixes, `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Added `scripts/sync_external_repos.sh` so TTRL remains an auto-synced ignored reference instead of a pinned dependency; docs now require recording the TTRL source commit when material is extracted from it.

## 2026-06-22

- Added HITTER paper under `papers/2508.21043v2.pdf`.
- Added ChingMu VRPN ROS 2 package under `hope_ws/src/vrpn_mocap` and supplied a minimal ROS 2 `package.xml`.
- Added tracked source/config subset of Agibot `a3_deploy_example` under `agi/code_deployment/a3_deploy_example`.
- Moved the complete Agibot deploy payload to ignored local storage under `vendor_assets/agibot/a3_deploy_example_full`.
- Added TTRL as an ignored local reference under `external_repos/TTRL-ICRA2026`.
- Removed root `tmp/` after moving useful materials.
- Verified `hope_planner` package tests from `hope_ws/src/hope_planner`: 20 passed.
- Could not run ROS workspace build in the current shell because `colcon` is not installed.

---

# ══ 2026-07-06 归档:NOW.md 大重排时移入的历史章节(内容原样,只挪家)══

背景:franco 2026-07-06 下令按"奖励结构优先"重排四阶段计划并清黑话,NOW.md 回归"只放现役"。
以下章节均为已完成/已过期的历史记录,原样搬入本文件;现役计划见 NOW.md。

## Plan To Saturday (2026-07-03 → 07-04, target: play at the venue with an improved policy)

Critical path (GPU 1/2) — ETAs corrected 2026-07-03 15:40 with the measured 2.15 s/iter:

1. **TODAY ~18:15** — 「基线结构」and「三合一」long runs hit 14k iters → 4-protocol scoreboard
   verdict (claude, ~40 min, CPU).
2. **TODAY 19:00-24:00** — explicit-clipped-PD fine-tune from the winner (~8000 it ≈ 4.7 h, GPU1);
   GPU2 = backup ft from the other run. (claude)
3. **TONIGHT/midnight** — export → MuJoCo explicit gate + deploy-faithful → build the MDU package
   (`build_a3_deploy_pkg.sh`). Saturday morning = verification margin, not critical path.
   (claude prepares, yikang ships/runs on the MDU)
4. **Saturday** — deploy & play: forehand first; **try backhand in SHADOW mode** —「三合一」and
   model_9000 both trained stand-entry, this is the potential headline. (yikang + franco)

Parallel tracks (no GPU conflict):

- **Ball physics v1 (P2.5 prerequisite)**: mocap ball-trajectory collection is happening NOW at the
  venue → fit drag/bounce from the fresh recordings (planner calibration path); spin-aware physics
  arrives with the simtoreal2 merge. Owner: **yikang** (franco 2026-07-03).
- **Teacher clips for Saturday: NO mass shoot needed** (franco 2026-07-03): the swings recorded
  during play sessions + orientation rotation (P2.2-lite, `reground_hope_frame.py` path) are
  sufficient; the 0703 clip uploads are that set. A5's 30-50-clip library and the dedicated
  ready-stance clip stay on the longer-horizon list (P2.0/A5 in G08), not on Saturday's path.
- simtoreal2 → main merge + doc updates (claude, in progress).

## 2026-07-04 白天台账(物理建模冲刺 + 白天测试/晚上跑满)

**入 main 的交付**(细节见 ball_physics_v2_roadmap / fit report §11):
- planner:12 态 EKF(有色噪声,影子模式)· Magnus 项 · Nakashima 自旋耦合反弹 · StrikeSpec
  逆解器(拍面角+v_n/v_t+落点敏感度;实测预算:v_n 0.85 m/(m/s) 是王,拍面 0.04-0.06 m/度)
  · venue 参数一致性守卫 —— 共 73 测试全绿
- 训练:A1v2 感知三缺陷(丢帧/击后盲窗/每击重锁偏差,默认关)
- configs/incoming_ball_venue.yaml:实测来球分布。**头条:手设 vb 盒只覆盖真球 20%**(训练
  一直在打"想象中的快平球")。采样原则(franco):**训练=按类重平衡的丰富度盒**(x −3.29~−0.10,
  z 到 −2.30,旋转 65 rad/s),**评估=综合分布**(答"真球接得住吗")
- 裁决:a_t=0.52 维持(自旋输入非病根;轨迹反推在本 rig 不可行 0/144;滚动上限被 CI 排除);
  双拍=打法混淆(§11.1);自旋衰减关闭(τ>5s)
- 实战互证:上旋不飞(接触抬升与 Magnus 下压线性抵消)、侧旋飞出台(无抵消)——**拍面补偿
  优先场景=侧旋**,已固化测试

**在飞**:P2 收尾(→自动导出终版)· eval 契约修复(P0,旧行为保留)+ 之后加"来球分布
驱动"的 B 模式评估 · R7+R8 机制检查 · 通宵舰队待命(R7 重平衡盒/R8 感知缺陷/R9 斜录
12k/R10 换种子,全部跑到底,P2 完成后自动点火)

## 10:00 判读(2026-07-04,自动流水线交付)

**判词:值得练到底。** P2 产品线(消旋虚拟球 + A1 校准噪声/延迟 + σ收紧 + 挥后起手,
R2@1500 热启)在 5500/12000 步时命中率 **0.858(峰值 0.884)**,已超 E 谱系终点 0.8128
—且带着 E 线没有的全部真实性约束。曲线在 0.85-0.88 平台微涨,无退化迹象。训练继续。

- 快照:`logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-03_22-32-36_r3_P2_product/exported_10am/policy.onnx`(iter 5500,Isaac-free 导出链)
- 信号档裁决(2000 步):R0=0.502 / R3 斜录=0.520(未过门槛,但 2 秒新 clip 首战打平)/
  R5 A1=0.491(几乎免费→进组合)/ **消旋 0.769 > 上旋 0.753(yikang 线)→ 组合采用消旋**
- 认证提醒:MuJoCo 评分器对连挂模型失效仍是 P0 —— P2 上真机前需修好评分器或走
  yikang 的厂商门禁

## Roadmap Scorecard (2026-07-03 傍晚版)

阶段定义(每项按同一条阶梯,标到已完成的最远一格):
**设计**=方案共识并写入文档 → **代码**=实现完毕(功能类默认关)→ **跑通**=机制验证(短跑无崩、开关生效)→
**有效**=信号档 A/B 有对照数字 → **门禁**=成熟版 checkpoint 过 MuJoCo 四协议 → **真机**=硬件验证。
数据类条目按:方案→管线→试样→拟合→入训练/规划器→真机。

| 项 | 设计 | 代码 | 跑通 | 有效 | 门禁 | 真机 |
| --- | --- | --- | --- | --- | --- | --- |
| P2.1 连续挥拍不倒 | ✓ | ✓ | ✓ | ✓(2k A/B) | **今晚终审** | 周六 |
| P2.2 朝向归一(现有 clip) | ✓ | ✓(re-ground 管线) | ✓(0703 clips 在训) | 隐含在主线里 | 随主线 | 周六 |
| P2.3a 自适应σ | ✓ | ✓ | ✓ | ✓(+48%) | **今晚(三合一)** | 周六 |
| P2.3b HER 目标重放(jiayi) | ✓ | ✓ | ✓(jiayi 线) | **今晚对照** | 未 | 未 |
| P2.3c 动作库检索(SMASH) | 未 | — | — | — | — | — |
| P2.4a 等球 hold+hold_ready | ✓ | ✓ | ✓ | 今晚对照 | 未 | 未 |
| P2.4b 减速命令/base 回归/ready pose | ✓(v3 doc) | 部分(base_decel v1) | ✓ | R12 待跑 | 未 | 未 |
| P2.5 物理建模 | ✓ | 部分(spin 代码在 main) | **数据采集中** | 未拟合 | 未入训练 | 未 |
| P2.6 smash | 未 | — | — | — | — | — |
| A1 延迟/时变注入 | ✓ | ✓ | ✓ | **今晚** | 未 | 未(数值等标定) |
| A2 坐标变换归属 | **未写**(已立项 yikang) | — | — | — | — | — |
| A3 执行器辨识 | 未 | — | — | — | — | — |
| A4 评估基建 | ✓ | ✓ | ✓ | ✓(已用于三次裁决) | 不适用 | 真机落盘制度未建 |
| A5 新视频 30-50 条 | ✓(主动推迟) | 管线已有 | — | — | — | — |
| A6 摔倒管理 | 部分(绝对终止在) | 部分 | — | — | — | — |
| A8 post-swing 起手 | ✓ | ✓ | ✓ | ✓(2k 中性) | **今晚(三合一)** | 未 |

读法:**一列看进度深度,一行看卡在哪一格**。今晚之后"门禁"列会填掉 4 格;"真机"列整列等周六。

## 动作源消融(franco 2026-07-04 拍板,今晚/明天必跑)

事实修正(2026-07-04 凌晨):registry 里两套(v3/v4 原始与 :latest 微调转正)**均为正面摆拍录制**;
此前 npz frame-0 yaw +82° 是 GVHMR 全局朝向产物,不代表机位。真正的**斜录(实战击球)视频**是新东西:
franco 已提供 forehand_new.mp4 / backhand_new.mp4,已上传 `/workspace/shared/motions/raw_video_oblique/`
(附 README 说明谱系)。franco 预期:实战斜录的动作质量会好很多。

| 消融臂 | 数据 | 状态 |
| --- | --- | --- |
| 正录(现役) | hopex 转正版(139/132 帧) | 所有既有结果就是它,无需重跑 |
| 斜录(新) | raw_video_oblique 两条视频 | **卡在 GVHMR→GMR→npz→转正→相位标定 管线**(pod 无 GVHMR 环境;dongc1 机器有)→ 产出后同配置 2000 步对照 |
| v5(更新) | raw_video_v5 两条视频(1.17/1.20s,franco 2026-07-04 下午提供) | **管线全通(2026-07-04 晚,claude,`/workspace/franco/v5_pipeline.sh` 复用斜录管线)**:npz 56/58 帧 @50Hz 已入 `/workspace/shared/motions/hope_{forehand,backhand}_v5.npz`,yaw 转正(+86.6°/+83.6°→0),相位 **[0.673, 0.345](franco 纠错版;检测器的 0.768 是甩鞭峰不是触球)** → 消融臂 R15 |

## 关键更新联合消融(Isaac 复活即发射;2000 步信号档,4096 envs,一卡两跑错峰)

| 臂 | 配置(完整命令附后) | 回答什么 |
| --- | --- | --- |
| R0 基线 | `task=HOPEPingPongDeployParity`(main 默认) | 合并后的参照点 |
| R1 虚拟球·上旋 | `task=HOPEPingPongVirtualBall`(yikang 默认,落点30/过网20/旋转5,拍速法线降权) | 物理奖励栈是否成立 |
| R2 虚拟球·消旋 | R1 + `task.racket.vb_spin_mode=minimize`(franco 第一阶段:不奖励球质,奖励落点+出球旋转最小) | 两种旋转哲学谁先学会站稳打准 |
| R3 斜录数据 | R0 + `motion_file=/workspace/shared/motions/hope_forehand_oblique.npz motion_file_2=..._backhand_oblique.npz "task.racket.strike_phase_per_clip=[0.368,0.495]"` + 下方专属采样框 | 实战动作源是否更好(franco 预判:是) |
| R4 组合 | R1/R2 胜者 + adaptive_sigma + post_swing(可再叠 R3 若其胜) | 产品候选 |

斜录臂专属采样框(从斜录击球帧提取,保持参考-目标一致性):
`--pos-range-per-clip 0.08 0.28 -0.72 -0.32 0.79 0.99 0.38 0.58 -0.51 -0.11 1.10 1.30`
`--vel-range-per-clip 1.52 2.52 -0.57 0.43 0.49 1.29 0.73 1.73 -0.30 0.70 0.06 0.86`
(yaml 键 pos/vel_range_per_clip 同值;斜录反手峰速仅 1.33 m/s 偏温和,视频重录候补)

新增部件状态:vb_spin_mode=minimize 已实现入 main(默认仍 topspin);斜录 npz 已产出
(`hope_{forehand,backhand}_oblique.npz`,96/106 帧,yaw 已转正,MuJoCo-FK 转换器 body 残差 0.00mm;
速度用有限差分,与 Isaac 雅可比法峰值差 ~0.18m/s——首个 mech check 时顺带确认无碍)。

## 基建现状(2026-07-04 凌晨)

- **pod 宿主机对 Isaac 判死**:裸 Kit 在重建容器上仍挂死;Stop→Start 不换宿主机(卷钉死机器)。
  训练全面阻塞。选项:RunPod 支持重置宿主机 / 新建 pod 迁移(venv 可重装,数据 ~20GB 可 rsync)。
- **导出与评分已绕开 Isaac**:`standalone_onnx_export.py`(纯 CPU 重建 actor+归一化器+动作缓冲,
  元数据从同配置旧 ONNX 拷贝)——两个 14k checkpoint 已导出成功,四协议评分进行中(纯 CPU)。
  周六候选的裁决不再依赖 Isaac。

## 周六早晨交接(2026-07-04 04:30 写)

**给今天打球的建议(按风险排序):**
1. **主用 07-02 已验证的部署包**(yikang 手上,正手已在真机成功过)——零新风险,保底能打
2. **E@14k 作为实验候选交给 yikang 过厂商门禁**:ONNX 已导出(位级忠实,`.../p21_E_sigma_postswing/exported/policy.onnx`,sha12=e2c19a01cd3e)。他的 AGI-MuJoCo 门禁环境在他自己机器上,不受 pod 影响。**过了门禁才上机,过不了就只用保底包**
3. Isaac 复活后第一件事:A/E 的正规 Isaac 评估补认证

**今晚查明的三件大事:**
- ~~**pod 宿主机对 Isaac 判死刑**(裸 Kit 两种缓存状态都起不来;裸 CUDA 正常)→ **需要:RunPod 支持票 或 新建 pod**。训练全阻塞,消融梯/斜录臂都在等这个~~
  **→ 已解决(2026-07-03 19:15, yikang):训练在这台 host 上复活了,不用迁移 pod。** 死刑判早了:barekit6 的 faulthandler 栈显示挂点在
  `isaacsim.asset.importer.urdf` 扩展 on_startup **无条件构建导入窗口 UI**——headless 也建,而第一个 `omni.ui.StringField` 控件创建在这台
  host 上永不返回(渲染栈 iray/RTX 确实坏了,franco 的判断对了一半)。修法:extscache 的该扩展打了**环境变量门控补丁**
  (`HOPE_URDF_IMPORTER_NO_UI=1` 跳过 build_ui,不设=原版行为;URDF→USD 转换 API 不依赖该窗口)。yikang 的 env.sh 已默认导出;**其他人的
  env.sh 也要加**。验证:HOPEPingPongVirtualBall 4096 envs 正常进入 PPO 循环(vb_smoke, GPU0)。注意:**需要 RTX 渲染/相机的任务在这台
  host 上仍然是死的**(_wait_for_viewport 挂,iray 插件加载失败)——headless 无相机训练/评估 OK,录像/render 类任务仍需迁移。
- **MuJoCo 评分器与连挥模型的契约不匹配**(P0,周日修):新旧两版评分器对 E@14k 给出一致的病理性全零(回合 ~1s 截断、只评反手、误差系统偏移 ~0.5m);真机验证过的 07-02 模型在同评分器下 0 击球窗口。评分器的回合/参考时钟按旧谱系(瞬移单挥)假设写死,对 wrap_teleport=False 的模型无效。证据:/workspace/franco/{premerge_check2,knowngood_check2}
- **Isaac-free 导出链已建成并验证**(standalone_onnx_export.py + harvest,位级零差)——导出永不再被 Isaac 绑架

**斜录(实战)动作:视频→CSV 全通**(GVHMR/GMR 在 pod 上直接跑通了,产物 /workspace/shared/motions/oblique/,58/64 帧)。剩 csv→npz 一步需要 Isaac FK(或我写 MuJoCo FK 版,已排队)。球轨迹数据已入共享区 ball_mocap_0703。

## 🔴 结构级发现(2026-07-04 傍晚,eval B 模式首跑)— 需要 franco 拍板

**策略的拍面朝向是"clip 锁死"的,175 维观测契约里没有任何法线指令通道。**
B 模式(真实来球分布 + StrikeSpec 反解应有拍状态)实测:位置/速度跟踪意外地好
(OOD 到 3.7cm / 0.18 m/s,双双满分)——但法线误差 36-76°,虚拟回球率 **0%**。
反事实归因:同样的达成质量,换上"应有法线"→ **25/25 合法回球**(中位 6.7cm);
换回错误法线 → 0/25。**法线是唯一短板,而且是架构性的**(策略只模仿 clip 的拍面,
无从得知"这个球需要什么拍面")。这正是 franco"拍面角度是接旋球的关键"的最终形态。

两条路(可并行):
- **A 短期(零训练,立即可部署)**:planner 反过来适配策略——StrikeSpec 加"固定法线求解"
  变体(法线钉在 clip 参考值,只解 v_r/落点),放弃部分落点自由度换取合法回球。claude 可即做。
  **→ 已做并判死刑(2026-07-05,claude)**:固定法线求解已实现(`solve_fixed_normal` +
  `--venue-fixed-normal`,16/16 测试过),复测结果 **0% —— 不是求解器不收敛,是物理不可行**。
  暴力可达性扫描(拍面钉在 clip 参考朝向,遍历 ≤6 m/s 全部拍速,~7000 落点/球)证明:
  **正手拍面(几乎朝正侧方 [0.41,0.90,-0.17])打出的球最远落 x≈1.4 m,连网(1.87 m)都
  过不了——任何拍速都救不了**;反手拍面只剩一条"贴网大斜线"小缝(落点 x≈1.9-2.0、
  y≈±0.3-0.67,需 2.5-4.3 m/s),全部在合法回球框(过网纵深 ≥0.3 m,即训练自己的
  dink 保护)之外。前提核实过:mode-A 里策略实际拍面 vs clip 参考只差 1.9°(达标率 100%),
  钉 clip 参考=钉策略真实拍面。**结论:规划器迁就救不了现役 clip 拍面,路 B(契约加法线
  通道)/ R16(手腕放开)/ 新拍面数据是唯一活路。**
  **→ 第三根钉子(2026-07-05 晚,R16 信号档 eval-B)**:手腕放开跑了 5.5k 步,拍面-vs-应有
  误差 43.5°(基线 42.5°)纹丝不动、回球率仍 0%——**落点奖励的间接梯度掰不动拍面,
  "契约日加法线指令通道"从三选一变成唯一剩下的路**。
  **→ 第四根钉子,也是机制解释(2026-07-05 晚,P2 训练曲线复盘,franco 追问"训练有落点
  奖励为什么没学会"逼出来的)**:P2 训练全程 13.5k 步,虚拟球**落点入界率 0.03%、过网率
  0.2%、落点误差 1.9m(和考试的 1.5-1.6m 同病)——三个打球奖励项(落点 30/过网 20/消旋 5)
  的实际收入全程 ≈ 0**。也就是说:0.89 的三合格率全部来自模仿+位置/速度跟踪,"打球效果"
  奖励装了却从没通电;R16 放开手腕也没用,因为一次正样本都没有、没有梯度可爬(奖励沙漠,
  yikang 07-03 已放宽过一轮核仍如此)。**训练分布 ≠ 考试分布不是主因——训练分布下它同样
  一颗都打不回**。治法两条腿(o4 轮):①契约日拍面指令 = 直接姿态梯度,不经过接触模型;
  ②franco 分阶段出题 = 把第一阶段题目限制在"物理上打得回"的域内,先点亮奖励再扩散
  (注意:可达性扫描已证正手拍面在任何拍速下无解,所以 curriculum 单独救不了正手,两条腿
  必须一起上)。
- **B 长期(下一代训练)**:观测契约 175→178(+目标法线 3 维),训练 racket_normal 奖励改跟
  指令法线,部署 pp_obs_builder 同步——一次契约变更,换来真正的"按需拍面"。需要全队排期。
  **路 A 死刑后升级为"必须做",不再是可选项。**

(附带发现:场馆来球接触点在人的击球高度 0.98-1.26m,高于训练框——采集分布是"人接球"
视角,机器人版分布尚缺;B 模式 v1 盒采样未强制相关性,升级路径已注明。)

→ 设计已固化并按代码逐条核实:[motion_and_contract_v3.md](motion_and_contract_v3.md)
(2026-07-04 晚,claude)。核实中的新发现:①`racket_target_vel_w` 是世界系裸直通——现役
契约里只有位置做自我中心化,速度不做(真机 identity-yaw 下退化一致);②部署线协议
`RacketCommand.normal` **已经在传法线**,缺口只在 obs 坑位与训练奖励参照;③critic 已有
特权 `racket_target_normal_w`,契约日训练侧只是把 racket_normal 奖励参照从 clip 换成指令。
迁移定为 **175→179**(法线 3 + ρ 1,可顺手把 vel 也改自我中心)。


### 阶段 0:出题与相位校准(纯 CPU,最优先)

| 项 | 内容 | 判据/产物 | 跑量 |
| --- | --- | --- | --- |
| 0a 目标不是解 | ✅ 已做:完美跟踪入界 0.00%/0.03%(2 万题/边) | 一锤定音 | 完成 |
| 0b 难度地图 | **✅ 已做(07-05,10 万题,可解率 94.5%)**。难度=应有拍面偏离模板角度:**正手中位 71°,98% 的题落在 60-90° 层,0-45° 层空的——从模板出发没有任何"近题"台阶**;**反手中位 37°,阶梯连续**(15-25° 3% / 25-35° 37% / 35-45° 51% / 45-60° 10%)。需求拍速中位两边都 ~1.9 m/s(速度不是瓶颈)。**设计结论:阶段 1 反手先行**(v5 反手 f17-18 窗口 + 连续难度阶梯,curriculum 有路可爬);**正手没有低难度起步区**——要么靠拍面指令硬拉 60-70°(最难臂,风险高),要么**下次拍摄清单加"拍面朝前的正手挥拍"**(franco 定)。产物 qbank_v0.npz 转作考卷/curriculum 刻度 | ✅ 完成 |
| 0c v5 拍面扫描 | ✅ 已做:v5 触球帧 0.47/0.46 vs hopex 0.41/0.39,v5 反手相位 0.18-0.23 有朝前段 | 按 0d 重新裁决(朝前分量只是粗代理) | 完成 |
| **0d 相位×可回球性扫描** | **✅ 已做并已并进 yikang 的工具(07-05 深夜)**。⚠ 语义修正(yikang):登记表 `phase` = 视频真值(人何时触球),扫描给的是**训练最优相位**(clip 运动学哪帧最适合机器人回球)——两者可合法不同,登记表新增独立字段 `train_phase_candidates` 存后者,**绝不回写 phase**。结果:六条 clip 五条全程 0%;**唯一窗口 = v5 反手 f17-18(相位 0.30-0.32)**,视频真值帧 21(0.362)两侧都是 0%;窗口分数随球速档变化——**场馆速度球 96%,阶段 1 默认 2-5 m/s 只有 17% → 阶段 1 起步球速用场馆档 [~1.0,2.5] 再放宽**。斜录换相位救不了(全程无窗,登记表已核 0.432/0.495 同样 0%)。实现已合一:yikang 的 gen_stage1_questions.py 加 --phase-scan 模式(分支 stage1-fixed-point,b36a008),我的草稿扫描器退役;他的 z 坐标陷阱教训(planner 残差全绿≠落点对,必须 torch 闭环复检)已内建 | ✅ 完成 |

| **0e 三假设翻案探针(franco 质疑"人明明稳定回球",07-05 深夜)** | ①**混帧 bug(franco Q1 猜中)**:分析器 normal_root 是骨盆系、速度/位置是世界系,挥拍中段骨盆转动把扫描/题库里的"拍面"转走几十度——已修(normal_w 世界系,b4825a0),修后 v5 反手窗口 f17-19=100%(场馆速度),两套实现完全对账,题库难度角改正为 正 69.6°/反 38.5°。②**斜线对拉(franco Q2 猜中)**:把球台绕击球点转 ψ 复测——**v5 反手在 ψ=-40°~-50° 时可回球性 88-100%(上旋球最优),ψ=0(我们假设的正对几何)是 0%**:人打的就是约 45° 斜线,视频真值帧 21 在正确几何下完全成立;f17-19 vs f21 之谜消解=两帧各自对应不同球台朝向。③**正手另有隐情**:方向指纹显示正手标注帧出球**射程中位仅 0.05 m(球直接砸脚下)**,转任何 ψ、±50° 握拍偏置都救不活——是 GVHMR 拍面重建在该帧不可用(登记表 face_normal_reliable:false 的实锤),不是"人的正手不行"。④握拍偏置(Q3):共用 +20°z/+10°x 能把合计从 0% 提到 50%(主要救反手),待叠加 ψ 修正后重测 | **管线修正提案:登记表加 rally_yaw 字段**(每 clip 的对拉轴向,视频/探针可测),重定向时按它转正——"转对方向"从此制度化;正手模板的拍面重建不可信,阶段 1 正手锚废弃,等新拍摄或纯指令拉动 | ✅ 探针完成,rally_yaw 待 yikang 接 |

**0f 定论(franco 2026-07-05 深夜):"拍面一直是算错的"——因为管线根本看不见拍子。**
GVHMR 只重建人体;所有视频 clip 的"拍面"= 腕系 +Y,是虚构量不是测量值(登记表
face_normal_reliable:false 的最终解释)。人的真实拍面 = 腕 × 握拍角,握拍角从未被观测。
**什么塌了、什么还立着**:
- 塌:一切"视频 clip 的拍面朝向"当物理事实的解读(正手"球砸脚下"= 人的腕姿态 × 机器人拍装
  几何的组合不成立,人自己的拍子当然不是这么装的);
- **立着(关键)**:机器人的拍面严格 ⊥ 腕+Y(STL 实测 0°)——所以"机器人复刻 clip 腕姿态时
  拍面指向哪里"是精确可算的,0b/0d/0e 全部结论按"机器人执行"口径依然成立:反手先行、
  正手腕姿态离任何解 ~70°、v5 反手 f17-19 锚点有效;
- 推论:①R18 拍面指令通道的理由从"三重"升为"四重"(视频拍面本就不可作为监督源);
  ②**采集升级进拍摄清单**:下次拍摄让拍面可测——拍上贴标记/棋盘格,或正对机位+同步球轨迹
  (场馆球轨迹已有,若有同场身体视频可反推每场的握拍角=把虚构量变成标定量);③在此之前,
  所有视频 clip 的腕朝向只当运动先验,不当拍面真值(R16 的直觉获得最终背书)。

**0g 拍面标定成功——六条 clip 全部翻案(franco 的推理程序,2026-07-06)。**
按"v5 是标准对拉,触球帧必然合法回球"的先验,用拍面点运动学(修正后的分析器口径,不再用
腕点近似)联合反解(共享握拍角,每 clip 一个斜线方向):
**握拍角 = 拍面相对腕 +Y 绕腕x 上仰 40°、绕腕z 转 5°**(腕局部拍面方向 [-0.07,0.76,0.64],
符合真实握拍)。三重验证:①v5 正/反手在触球真值帧双 100%(ψ=+40°/-40°,正反手两条对角线
方向相反,完全符合球理);②**同一角度零调参迁移到斜录:双 100%**(ψ=+55°/-60°);③hopex
空挥修正后,可回球窗口恰好落回约定相位旁(正 0.46-0.49 / 反 0.32-0.35)。
**推论:此前一切"这个动作打不回球"的判决(0b 正手 71° 无台阶、0d 五条全零、0e 球砸脚下)
全部是"未观测的握拍角 + 对拉方向"两个缺失量所致,动作本身没有问题。**
**franco 拍板(2026-07-06):烤进重定向**——把标定握拍角写进 csv→npz 管线,腕关节逐帧
调整使机器人拍面对齐人的真实拍面,一次修好全部模板先验;拍面指令通道照上(两条奖励通路
——模仿 与 回球——本来就并行,先验修对后指令学的只是随球微调)。**动作大升级一步到位
(franco 授权,claude 执行)**:①csv_to_npz 加握拍旋转(腕 3 关节逐帧反解,FK 全身一致重算,
顺带查机器人腕关节限位够不够 +40° 上仰——不够就是真硬件发现);②rally_yaw 转正进重定向
(登记表"打直线"约定落地到 npz 本体);③六条 clip 重生成 → 难度地图/相位扫描/题库全部重算
(正手预期复活);④标定求解器产品化(gen_stage1_questions --grip);⑤登记表 v2(标定 clip
标 face_normal_reliable: true-calibrated)。
**→ 执行进度(2026-07-06,claude)**:①⑤已完成——`csv_to_npz_mujoco --grip-rot` 落地(腕三
关节逐帧反解、FK 全身一致、关节速度重差分;**限位审计:四条 clip 零裁剪,机器人手腕全程够得
到标定拍面**,面残差 0.01°);四条标定 clip 已产(`hope_{fh,bh}_{v5,oblique}_cal.npz`)+
登记表条目(分支 00f1530)。**端到端验证(真值帧、机器人原生拍面口径、rally_yaw 转正)**:
斜录正/反 **100%/100%**、v5 反 **100%**、v5 正 **69%**(烤入后拍面点位置随腕转移动 ~13cm,
接触点变了——这是机器人真实可执行的诚实数字)。②按 franco 口径改为登记表字段+用题时旋转
(数学等价);hopex 无源 CSV 走 npz 后处理(待办);③地图/扫描/题库在 _cal 集重算(下一步)。

**「老师动作适配器」接口(franco 2026-07-06 修正版——之前我把③想窄了)**:三个轴按人打球
的真实机构设计:①**整身朝向旋转**(全套动作绕锚点旋 φ,打向适配;rally_yaw 机制复用);
②**加减速**(R14 变速重定时机制);③**拍面/侧向速度 morph——小臂带动为主**(肩+肘+腕全链、
从引拍起全程渐变;franco:侧向速度人也是小臂带动的)。**v1(触球窗 ±0.24s 拧腕)初衷即错,
且被包络实验否决**(限位裁剪 96-100%、腕速需 8-10 倍、拍位漂移 33cm、适配后回球 0-27%);
v2 = 全程链级 morph 或**预烤变体库 + SMASH 最近邻检索**(优先)。与指令通道关系不变:通道告诉
策略"要什么",适配器让老师"示范要的";阶段 1 第二波消融净贡献。完整体系见
[motion_pipeline.md](motion_pipeline.md)(视频→标准老师→泛化接口,全链路人话说明书)。

**0h 协同状态(2026-07-06 凌晨,yikang 审计二轮 + 合流)**:yikang 抓到 5 个产品化问题并已推
分支修复,其中概念级一条全队要记:**"旋转 mount offset ≠ 烤入腕链"**(offset 几乎平行握拍轴,
只挪 1.6cm;烤入是重解整条腕链,挪 ~13cm)→ **_cal npz 是题库唯一合法输入源**,raw+登记表旋转
会把击球锚点放到离机器人可执行拍位 ~11cm 的地方。其余:烤入检测 fail-closed 化、训练 loader
强制读 grip_applied、单 session 钉错报错、守卫弃裸 assert。我这边对应动作:①筛选/权威验证
分层(raw 拍位只做筛选,定稿必在 _cal 上复验);②**179 维评估器支持已交付**(拍面指令尾 4 维,
175/180 逐字节回归过,main bec7673)——他的接线+我的评估器=S1 反手臂机制检查就绪;③顺序:
握拍终值(四 clip×轻/重球联合标定)→ 重烤 _cal → 他重出 bank;**反手先行不等终值**(v5 反手
在候选握拍角下都 100%,现有 _cal 即可出反手 bank)。教训台账:旋转/坐标系已连抓五个真 bug
(z 偏移、骨盆系混用、腕点≠拍面点、旋向符号、offset≠腕链)——全是"solver 全绿、物理全错"型,
**对抗验证轮固定进流程**。

**0i SMASH 动作泛化对表(franco 让研究;2026-07-06 读原文 §IV,VAE 之外全部可借)**:
SMASH 的泛化 = 库(400 真实条 → VAE 扩 5k,**造库工具我们换成物理反解+烤入,不用 VAE**)+
**最近邻检索当老师**(特征=锚相对击球点,加小扰动 ε 模拟部署预测误差;检索到的 clip 进观测当
motion command)——**这正是我们"变体 clip 库"路线的正名**,检索特征我们再加拍面维度。五件
直接可抄:①**击球窗分通道任务奖励**(位置窗 0.02s 紧、朝向/速度窗 0.1s 宽,gated exponential
——"触点要准、挥向挥速给时间余量",防手腕尖峰、降 sim2real;比我们现在的单帧打分先进,抄);
②**手腕从模仿踢除**(原文原话:防参考过约束拍子、让手腕服从任务——**R16 作为组合件被论文
背书**);③**相位依赖任务噪声**(噪声 ∝ 距击球时间,远大近小=规划收敛的真实样子;和 A1 的
测量噪声互补,可叠成 A1v3);④**tracker 可执行性过滤**(生成的动作先让跟踪策略在仿真里跑,
跟不稳的丢弃——我们的变体库同样要过这道筛);⑤**区域自适应采样**(按滚动成功率优先采弱区
——franco 扩圈 curriculum 的兄弟版,进消融第三臂)。
**一个值得记的分歧**:SMASH **不给 actor 拍面指令**——他们的拍面误差 e_ori=arccos|n·v̂|(对称
双面),称"拍面由目标击球速度隐含"。他们场景旋转轻(面≈速度方向的函数);我们要消旋/补旋时
面≠f(v)。→ 阶段 1 加一条消融臂:**拍面显式指令 vs 仅速度隐含**,让数据裁决(若隐含够用,
契约日可以只加 ρ 不加法线,省 3 维)。

**0j 握拍终值 + 权威复验(2026-07-06 收口)**:四 clip × 轻/重上旋联合标定终值
**Rg=(0°,+40°)——八个组合全 100%,唯一 worst-case 满分解**;与现烤值 (5°,40°) 同在 100% 高原,
**六条 _cal 免重烤**。权威复验(烤后运动学、机器人原生拍面、rally_yaw 转正):**轻上旋
(=考试/场馆档)六条全 100%**;重上旋(对拉档)v5反/v4正/v4反 100%、斜录反 69%、v5正/斜录正
12%(开面 +24/+39° 接不住重上旋——物理自洽;阶段 3 的题,也正是拍面指令通道要教的"随旋收面")。
意外收获:**v4(hopex 视频重跑)是最稳的一对(轻重全 100%)**——阶段 1 锚点备选。立拍张力
留给物理线:模型坚持 ~24-40° 开面才全通吃,若真机回球系统性偏长/短,第一嫌疑=接触模型切向
拟合 a_t。


## 快速横向对比制度 + 新消融臂(2026-07-04 晚;franco:想法变多了,不再默认训久)

**制度(新想法一律走梯子,不再单独长训):**

1. 机制检查(512 envs × 25 it,~3 min):开关生效、无崩。
2. **信号档 2000 it @ 4096 envs**(独占 ~1.2h / 共卡 ~1.5h;3 卡 × 2 槽 = 6 臂/批,
   错峰 ≥60 s):同种子、同热启(当前 P2 产品线 ckpt),**每批必带同批对照臂**。
   裁决口径 = Isaac composite(2000 步,同 10:00 判读)+ eval-B 回球率(CPU,不占卡)双轴。
3. 只有信号档赢家进组合臂;组合赢家才升 12k(~7h)与门禁。输家记下数字后关闭。
4. **不急着跑全(franco 2026-07-05)**:新臂默认 4000 步(热启臂)/2000 步(从零臂),
   赢家再续跑(checkpoint_path 无缝续);**平台化的长跑当场停**(先记数字)——R11 在
   10.7k 平台 0.762 被停就是首例。每张卡最多 2 个任务、错峰 ≥60s 的定版继续有效。
5. **排队脚本必须"发射前查卡"(2026-07-05 事故规则)**:GPU1 曾出现 3 任务同跑,复盘 =
   两条发射链都把卡号写死在写脚本的时刻 + 一个手动加的臂(R11,06:27 临时发的 12k 长跑)
   没进任何账本 → 4 小时后舰队脚本按旧地图开火撞车。修法已落地:看门狗改用 wait_slot
   (发射瞬间数一遍卡上 >2GB 的进程,<2 才发,占满则等待并每 30 分钟报一次);**临时手动
   发臂必须当场进 NOW 消融表**(就是"先认领再动工"规则的 GPU 版)。加固版
   `o3_watchers2.sh` 已替换旧链。

**新臂(接 R 系列;R0-R10 已用):**

| 臂 | 配置(叠在当批对照/胜者上) | 回答什么 |
| --- | --- | --- |
| R11 | `task.motion.clip_switch_prob=0.002` | **已停 2026-07-05(平台 0.762@10.7k,命中率税 0.12)**。剂量 0.002=每挥约 28% 被打断,远高于真机频率;回合长度 463≈469 说明没有多摔,掉的纯是命中率。~~真收益(抗切换摔倒)的量尺缺失~~ **量尺已建并量完(2026-07-05,`--switch-stress`):纯换招扰动压不出收益**——P2(没练过换招)在两种 PD 口径(Isaac 对照 implicit + 门禁 explicit clipped-PD)、24000 步 230 次换招(127 次在挥拍中段)下 **0 摔、换招后 2 秒存活率 100%、换招后命中率 0.97-1.00**;R11 同样 0 摔,唯一可见差别是它在门禁口径下命中率还略低(0.98-0.99 vs P2 的 0.99-1.00,税的延续)。**判读:MuJoCo 里换招离散跳变本身不构成摔倒威胁,switch 训练在这把尺子上零收益、税为真 → R11@0.002 维持拒绝**。尺子的适用边界:场馆真机摔倒可能是换招×感知毛病×延迟的交互(纯换招在干净仿真里不复现);若还要追这个方向,下一步是压力协议叠 A1 校准噪声或"击球窗口内定点换招",而不是继续扫剂量 |
| R11b | `task.motion.clip_switch_prob=0.0005`(剂量/4,**已点火 2026-07-05,o3_R11b_switch5e4**) | franco:继续调参找好参数。先扫剂量;若 0.0005 仍税重,下一轮动奖励侧(打断后 hold 补偿)。**⚠ 判读标准更新(2026-07-05 换招压力测试结果)**:抗摔轴上 P2 本来就满分(0 摔),R11b 在该轴不可能赢——它的存在意义只剩"更低剂量是否免税"(Isaac composite 对齐 P2)。**4000 步跑完,压力测试终值已量(exported_it4000_normbaked,2026-07-05)**:230 次换招 0 摔、换招后 2 秒存活 100%、两种 PD 口径命中率干净/换招后均 1.00(@2700 临时值相同;分布内不伤命中,好于 R11@0.002 的 0.98)。**信号档跑完(→5499,2026-07-05 判读):composite 峰值 0.839@5247,同批最低(R12 0.888 / R14 0.871 / R16 0.870)——低剂量税没免掉(~0.03-0.05),抗摔轴又已证明买不到东西(压力测试 0 摔)+ eval-B 0%/CF 98% 与全批一致 → 建议 clip_switch 方向整体关闭**(除非部署轴另有理由,franco 拍板) |
| R12 | `task.rewards.base_decel_weight=1.0`(o3_R12_basedecel,02:45 点火) | 减速塑形方向有没有信号(v1 P 律;v2 拟合包络等 6 套采集)。**信号档跑完(→5499):composite 峰值 0.888@4654,同批最高、与产品线持平——composite 无税**;eval-B(峰值存档 it4700):回球 0%/CF 100%/拍面 43.3°(结构现状不变,如预期——它本来就不是治拍面的)。终判看部署轴(减速入位行为、base_speed_xy_prestrike),candidate 进保险包 |
| R13 | 赢家叠加进产品线 | 产品候选 |
| R14 | `"task.motion.speed_scale_range=[0.8,1.2]"`(o3_R14_retiming) | **变速重定时**:同一 clip 变速播放+速度需求同步缩放,策略能否学会按需调节挥拍速度幅值 = 无新数据的连续强度 v0(franco"加减速改幅度";空间幅度另由 R6 裁剪臂回答,两臂合看)。**信号档跑完(→5499):峰值 0.871@4843,比产品线低 ~0.02——变速出题更难,小幅回落属预期**;eval-B(it4800):回球 0%/CF 96%/拍面 48.5°(比其他臂散 ~5°,变速让拍面更飘——判读时的一个减分项) |
| R15 | v5 clip 臂:`motion_file=/workspace/shared/motions/hope_forehand_v5.npz motion_file_2=.../hope_backhand_v5.npz` + 下方 v5 专属采样框(相位 [0.673,0.345])(o3_R15_v5clips) | 新动作源(短条、首尾贴 ready);⚠与 hopex 差三个因子(数据源+clip 长度+参考噪声),与 R6 对照拆长度因子。**⚠ 发射异常(2026-07-05 06:57):看门狗把它发到了 DeployParity 任务、从零 2000 步(峰值 0.355,从零短跑正常水平)——与同批 vb 臂不可比,run 在 `agibot_a3_hope_deploy_parity/…o3_R15_v5clips`。按计划判读需在 vb 线重发(等 franco 定)** |
| R16 | `task.rewards.free_wrist_ori_mimic=true`(**已点火 2026-07-05,o3_R16_freewrist,franco 最高优先**) | **手腕解除姿态模仿**:把 right_wrist_yaw_Link 从 motion_body_ori/ang_vel 列表拿掉(位置/线速度模仿保留=挥拍路径照学)。理由:视频管线的手腕朝向不可靠(GVHMR),模仿它给拍面质量封顶。**在 vb 产品线上跑才有真学习信号**(落点/消旋奖励直接塑形拍面;纯 DeployParity 上只是去噪)。契约日它是法线指令通道的执行机构。**信号档跑完(→5499)+ eval-B 终判(2026-07-05):峰值 composite 0.870;真球考试拍面-vs-应有误差 43.5°(基线 P2 42.5°,纹丝不动)、回球率 0%、反事实 98% —— 手腕放开单独掰不动拍面**:落点/消旋奖励的间接梯度在 5.5k 步内没把拍面拉向可回球朝向。判定:**单独不进配方;它的正确位置是契约日的执行机构**(有了法线指令+奖励参照改跟指令,拍面才有直接梯度可爬) |

R15 v5 专属采样框(**franco 纠错后 2026-07-04 深夜版**;从击球帧提取,pos ±0.10 / vel(clean) ±0.50):
`"task.racket.strike_phase_per_clip=[0.673,0.345]"`(正手相位从 0.768 改钉 0.673!)
`"task.racket.pos_range_per_clip.forehand.x=[0.29,0.49]" ...y=[-0.63,-0.43] ...z=[0.74,0.94]`
`"task.racket.pos_range_per_clip.backhand.x=[0.60,0.80]" ...y=[0.12,0.32] ...z=[0.81,1.01]`
`"task.racket.vel_range_per_clip.forehand.x=[0.74,1.74]" ...y=[0.71,1.71] ...z=[1.20,2.20]`
`"task.racket.vel_range_per_clip.backhand.x=[2.60,3.60]" ...y=[0.51,1.51] ...z=[1.66,2.66]`

~~⚠ v5 正手 +Y 红旗~~ **已解除(franco 定位对了 2026-07-04 深夜)**:所谓 +Y 主导是**击球
相位钉错**——检测器选了拍速峰 0.768,那是触球后的甩鞭段;逐帧核查后真实触球 ≈ 0.67-0.69
(franco 给的"约 2/3 处"),该处速度 (+1.24,+1.21,+1.70)、法线 (+0.47,+0.84,−0.27),
方向健康。反手检测 0.345 与 franco 的"前 3/7 内"吻合,不改。教训已写入:**拍速峰 ≠ 触球,
选帧要用"前向速度峰/人工先验"交叉验证**(analyze_strike_phase 的已知坑)。
残留的真问题:**v5 参考噪声大**——mean|关节加速度| 5.9/15.5 rad/s²(正/反手),是 hopex
(2.5/2.7)的 2-6 倍,斜录(3.5/3.5)介于其间。短条+快挥拍给 GVHMR 的平滑上下文更少。
这是 R16(手腕解除姿态模仿)与参考滤波的直接论据;R15 判读时把"噪声"当第三个混杂因子。

**合并测试策略(找最优组合,不做全因子):**

1. **按裁决轴分组**。性能类(数据/奖励:R3/R9 斜录、R6 裁剪、R14 变速、R15 v5、R12)
   看 Isaac composite,期待提升,单臂 vs 同批对照,赢家贪心进产品线;交互只对"赢家对"
   补一次 2×2。保险类(部署加固:R11 clip_switch、A1、A1v2)**本来就预期 composite 微降、
   部署轴受益**——判它们用 eval-B 回球率 + deploy-faithful,不用 composite 一刀切。
2. **保险类整包测**:R11+A1+A1v2 合成一臂"部署加固包";包整体 composite 降幅 <3% 且部署轴
   不劣 → 整包采纳,不逐个消融(2^n → 1);包失败才 leave-one-out 二分。
3. **数据源类互斥合并**:hopex(对照)/斜录/v5/裁剪 是同一问题的四个臂,天然同批横比,
   一批出结论。
4. 每次产品线换代跑一次换种子臂(R10 惯例)验稳健,防止贪心叠加吃噪声。

**CPU 任务(不占 GPU,可立即做):**

- ~~StrikeSpec **固定法线求解**变体(结构级发现路 A)→ eval-B 复测~~ **已做 2026-07-05**:
  `StrikeSpecPlanner.solve_fixed_normal`(现有 solve() 一字未动,+5 个新测试,16/16 过)+
  `mujoco_eval_onnx.py --venue-fixed-normal`。**答案 = 0%(物理不可行,不是没解出来)**——
  详见上方 🔴 结构级发现的路 A 判决;证据存 pod `/workspace/franco/cf_eval/`
  (scan_reachability.py + modeB_fixed9600.log + verify_solver.out——网格可行点喂回求解器,
  内点可解 2.8 m/s/3 迭代/4.7 mm,判决不依赖求解器召回:合法框内无任何可行落点)。
- ~~eval-B 反事实 flag 化~~ **已固化 2026-07-05**:venue 模式每次击球自动多评一次
  "换上应有拍面朝向"(达成位置/速度不变),strikes CSV 追加 6 个 cf\_\* 列(原 14 列字节
  不变),汇总块报 CF 回球率。正式复现(P2 产品线,9600 步 seed 0,44 次击球):
  **实际 0/44,反事实 44/44,CF 落点中位差 0.10 m**——"拍面朝向是唯一短板"从手工分析
  升级为每次评估自动输出的常驻证据(当日 2400 步存档逐字节复现,前 43 列与 07-04 一致)。
- eval-B v2:截断 MVN 相关采样;机器人视角接触高度分布(等采集)。
- **门禁补 q_des 限位剪切(2026-07-05 新立,jiayi 部署发现的训练/部署不对齐)**:部署端
  (pp_joint_limits.hpp)发布前把每个关节的目标角硬剪到物理范围,训练(clip_actions=null)和
  MuJoCo 门禁都不剪——策略靠"超范围目标角"多要力的那部分,真机拿不到。**P2 暴露面已量
  (2026-07-05 探针,闭环加剪/不加剪各跑 4800 步)**:超限步数 10-18%,**全部在腰/踝等平衡
  关节(手臂 0 超限)**,最狠瞬间被剪掉该关节力矩上限的 35-56%;但闭环剪着打,击球三合格率
  1.000 不变、0 摔,拍速误差只 +0.02-0.03 m/s → **对 P2 产品线,仿真里看不出实质伤害;残余
  风险在真机平衡临界时刻(踝力矩恰好在最需要时被剪)**。待办:①门禁加同款剪切 flag(探针已
  验证一行 clip 即可,mirror pp_joint_limits);②训练侧剪切 **✅ 已落地(simtoreal2
  `f8c166e6`+`df64b9d8`:train==deploy q_des clamp 进训练;修复谱系 model_16400_holdfix
  已在 TRUE plant 三门禁全过)**——各模型暴露面不同,其它谱系存档仍可拿探针复量。
- ~~**换招压力测试 eval 变体**(R11 收益的量尺,2026-07-05 立项)~~ **已建成并量完 2026-07-05**:
  `mujoco_eval_onnx.py --switch-stress p`(默认关,关=字节不变;训练侧 commands.py clip_switch
  同语义:均匀换模板、跳回起手帧、重新等球+换目标、机器人不动;开着时模仿守卫改成只看真摔)。
  量的结果见上方 R11 行:**两存档 × 两 PD 口径 × 三剂量全部 0 摔,收益轴饱和,税轴坐实**。
  R11b 跑完后一条命令补量(`--switch-stress 0.01 --pd-mode explicit --keep-passive`,
  先 standalone_onnx_export 带 --bake-obs-norm 导出;R11 的导出已放
  `o2_R11_clipswitch/exported_deploy_normbaked/`)。原始 12 组日志:pod
  `/workspace/franco/cf_eval/sw_*.log`。

## 计划未做全量清单(2026-07-04 franco 抓漏后建立;常驻,每日对账)

**P2.4 集群(本次漏账主体,franco 抓出):**
| 项 | 状态 | 归属/时机 |
| --- | --- | --- |
| P2.0 准备动作定义 | **franco 改判 2026-07-04 晚:不再专门拍 ready 视频**。v5 两条 clip 首尾均贴准备姿态(量化:fh/bh 起始帧互差 mean 0.15 rad——可直接当共享 ready 锚;首尾差 mean 0.24-0.27 rad,残差交给 RL 填充,clip_switch/post_swing/hold 机制都在)。ready 参考帧从 v5 clip 首帧提取,零采集成本 | claude(提取+接线 stand_start/hold) |
| PACE 减速命令(伪速度∝剩余误差→平滑减速入站位) | **v1 已实现 2026-07-04**(`rewards.base_decel_weight`,默认关,机制验证过)→ 信号档臂 R12;v2 = 拟合加减速包络+方向+时间预算+幅度耦合([motion_and_contract_v3.md](motion_and_contract_v3.md) §5,等 6 套采集) | claude |
| ready→strike→ready 拼接 | 未做(依赖 P2.0) | P2.0 后 |
| base-target 回归 ablation(HITTER 位置命令,论文背书) | 未做 | 通宵臂后的下一轮 |

**其余未做(按可行动性排序):**
| 项 | 状态 | 阻塞物 |
| --- | --- | --- |
| A4 后半:真机数据落盘制度 | 未建 | **今天场地就该全量落盘**(planner 日志/VRPN 流/视频) |
| A5 挥拍视频 30-50 条 | 主动推迟 | 场地顺手拍几条即赚 |
| eval B 模式(来球分布驱动) | **已交付 2026-07-04**(`--target-source venue-balls`;头条 = 法线 clip 锁死,见上方结构级发现;G06 有正式记录) | — |
| Ace 饱和 Magnus 形式接进 flight/virtual_ball | yaml 有键未消费 | 小活,高旋外推保护 |
| A6 摔倒管理(guard 包络、recover 行为) | 未做 | 下周 |
| A7 击球事件自动检测 | 未做 | 依赖 A4 落盘 |
| A2 坐标变换设计 | 未做 | yikang 下周 |
| A3 执行器辨识 | 未做 | 需专门真机时段 |
| P2.5-full 真球进 Isaac | 未做 | 下周主菜 |
| P2.6 扣杀 | 未动 | 长线 |
| Queued#6 G06 验收数字 | 半吊 | 新候选认证流程替代中 |
| Queued#2 clip 转正 | **其实已完成**(hopex=转正版),标记关闭 | — |

## Gap List 状态扫描(2026-07-04 中午重扫;原表保留在下方)

| # | 状态 |
| --- | --- |
| 1 终审+候选 | 候选=P2 线(0.884)已定;**双仿真认证仍开放**(eval 契约修复在飞,P0)|
| 2 explicit-PD 微调腿 | 作废(IdealPD 已否决;记分板显式门禁替代)|
| 3 门禁+MDU 打包 | **最大的真缺口**。认证分工定案(jiayi 提醒 2026-07-04):**官方门禁 = agi/A3_MuJoCo_Sim 厂商仿真**(部署保真,yikang 的门);wbt 的 mujoco_eval = 我们的指标工具(修复继续,但只服务指标)。门禁三层,claude 全接(franco 2026-07-04):L1 平价校验(pingpong_parity,无 ROS/AimRT,pod 可跑)→ L2 deploy-faithful 指标(修复后的 wbt eval × 厂商 MJCF)→ L3 完整 AimRT 仿真(需 distrobox 环境,pod 上立环境为后备,诚实标注工作量)。链条:P2 终版 ONNX(今晚自动)→ L1+L2 → MDU 打包 + 部署配置改指 |
| 4 部署配置悬空引用 | 仍未改指,打包时一并(依赖 #3)|
| 5 物理 v1 | ✅ 完成(yikang 场馆拟合已 merge)|
| 6 延迟/误差标定 | 噪声谱 ✅ 已接进训练;**延迟已有界**(franco rig ground truth 2026-07-04:传输稳定 <10ms;动捕处理本身支撑 300Hz 输出 → 端到端 ≈≤20ms ≈ ≤1 个 50Hz 策略步)→ 训练用 delay_steps=2(40ms)是保守上界,保留;时间戳精测降级为机会项 |
| 7 0703 clip 覆盖 | **franco/jiayi 团队接走**(2026-07-04);备胎:pod 上 GVHMR/GMR 管线已通,给原始视频即可出 clip |
| 8 算力 trade-off | ✅ 完成(手册封版)|
| 9 球进训练 | 虚拟球已实质替代(奖励层);真球入 sim 仍下周 |
| 10 A2 桥(yikang)| 下周 |

另扫出的未跑测试:R6 裁剪 clip 臂(工具/clip 齐,排 GPU0 空槽);HER 隔离 A/B(已被 P2
胜线包含,降级为学术项,正式放弃);KF 真流验证 + StrikeSpec 部署接线 + 机器人挥拍标定
= **下次场地日清单**。

## Gap List To Sunday (明确缺的活,截止周日 — added 2026-07-03)

| # | 缺什么 | 谁 | 何时 |
| --- | --- | --- | --- |
| 1 | 两条 14k 长跑的终审 + 选周六候选 | claude | 今天 18:15-19:00 |
| 2 | explicit-PD 微调腿(过 MuJoCo 硬门禁的配方) | claude | 今天 19:00-24:00 |
| 3 | MuJoCo 双门禁 + MDU 打包 | claude 备 / yikang 运 | 今晚-周六早 |
| 4 | ~~model_9000 同板对比~~ RESOLVED(jiayi:9000 为测试产物不提名;规则收窄为「提名上真机时才必须交 ONNX 过记分板」,训练期各用各的、以 Isaac 指标互比;main 部署配置的 model_9000 悬空引用今晚打包时改指获胜候选) | claude | 今晚 |
| 5 | 物理模型 v1:用今天采的球轨迹拟合 drag/bounce(→ planner 参数;训练侧下周) | yikang | 数据到即做,周六 planner 可用 |
| 6 | **延迟/误差标定**:从动捕录制的时间戳量真实延迟与噪声谱 → 填 A1 各 flag 的数值(franco 指出:这些本就该从物理建模数据算出,不拍脑袋)。顺带确认真实帧率(现有 300 与 320 两种说法,时间戳一算便知) | yikang(数据)+ claude(分析脚本) | 周六-周日 |
| 7 | 0703 打球录像 → 旋转归一 → 新参考 clip 验证(jiayi 的 re-ground 管线已做一版,确认覆盖) | jiayi | 周六前 |
| 8 | 训练速度 vs 并行数的 trade-off 终版报告(见下,搜索范围 4096/8192/16384 + 共卡) | claude | 今天 |
| 9 | 球进训练环境 + 落点奖励(P2.5-lite)— **周六前不可行,诚实排下周**;周六的增益来自策略改进+planner 物理参数,不来自训练内球 | claude/dongc1 | 下周 |
| 10 | mocap→runner 桥 + 坐标变换设计(A2) | yikang | 下周 |

## ~~Tonight's Test Slots (2026-07-03)~~ 已过期,由「关键更新联合消融」与通宵舰队替代

After the 18:15 finish + verdict, the freed slots run signal-tier (2000-it ≈ 1.5 h co-run) A/Bs.
"Winner" = tonight's better 14k run. All arms resume FROM the winner checkpoint:

| Slot | Run | 目的 |
| --- | --- | --- |
| GPU1-a | winner + 2000 it, plain (shared CONTROL) | 两组 A/B 的公共对照 |
| GPU1-b | winner + 2000 it, jiayi 默认捆绑（HER 30% + hold_ready 2.0,合并后已是 yaml 默认) | 验证新默认在我们谱系上不劣化——默认值影响所有人,必须有对照数字 |
| GPU2-a | winner + 2000 it, A1 延迟包（delay=2, jitter, 2% 中途更新) | 目标延迟训练在信号档稳定且不伤命中率(为 mocap 闭环) |
| GPU2-b | (备用) 若 backhand 在 deploy-faithful 明显弱于 forehand → backhand 加权微调 | 周六反手 SHADOW 测试的胜算 |
| GPU0-a/b | 留给 yikang 重启他的 20k(建议挂 adaptive_sigma)+ jiayi 任意 | 团队槽位 |

已关闭、不再测的问题:num_envs 扩缩(4096 定版)、一卡并行数(今晚 3 任务探针出数后定版)、显式执行器路线(团队已否决)。


## ══ 同批归档:阶段1 第一波详情/发射工序/阶段2-3 原文(2026-07-06 重排前版本)══

## 🏆 当前胜利组合(常驻;每出一个裁决就更新这里,谁赢谁进)

**产品配方 v2(2026-07-05 定)** = 虚拟球·消旋 + 动捕毛病仿真(A1 延迟/噪声 **+ R8 感知三毛病,新采纳**)+ 奖励自动收紧 + 上拍收尾起手 + 2% 中途改目标,从消旋臂 1500 步热启:

- 峰值 **0.908-0.910 @ ~9540**(r3_P2_product,tensorboard 曲线复核 2026-07-05;旧记录
  "0.893@11-12k"不准。13.5k 终点回落到 0.885 → **选峰值附近的存档,别拿最后一个**)
- **现役打包模型出处已查明(2026-07-05 指纹匹配)**:`exported_deploy_normbaked` = **model_9600**
  (输出差 5.7e-6),恰好落在峰值区——无需重导;此前"出处未记录"的坑就此关闭(以后导出用
  standalone_onnx_export,--run-path 里写明迭代号)
- R8 感知三毛病:0.890 平台 ≈ 无代价 → **并入配方**(理由:白送的抗噪,真机全是这些毛病)。**跑到底终判(13.5k,2026-07-05):峰值 0.905@10.6k——白送再证**
- R10 换种子:0.885-0.890 → 平台可信,不是运气。**跑到底终判(13.5k):峰值 0.914@10.3k,比 P2 自己还高——稳上加稳;且所有长跑峰值都落在 ~10-11k、终点回落,"选峰值存档"铁律再+1**
- **候审**(信号档在跑,赢了进组合):R16 手腕放开 / R14 变速播放 / R12 减速塑形 / R11b 低剂量中途换招
- **等统一考卷**(训练内分数不可比):R7 真球盒(0.83@8.7k 还在爬)/ R9 斜录(0.71 到顶)/ R15 v5(排队)
- **已拒绝**:R11@0.002 剂量(命中率税 0.12,已停在 10.7k;抗摔收益已复核:换招压力测试 14 组全 0 摔,收益轴饱和、税为真)、R6 剪短模板@2k(无优势)
- **⚠ simtoreal2 合并警示(2026-07-05)**:DeployParity 任务 yaml 现在带着 jiayi ARM A 重训的
  实验默认值(回合 16s、等球 0-8s、A8=0.25、base_decel=ON、位置奖励 std 0.15)。**产品线
  (VirtualBall)已在自己的 yaml 里显式钉回基线值不受影响**;在 DeployParity 上开新臂的人注意
  基线已变,对照要自带。jiayi 的 std 0.15 理由(0.20 下模仿白拿 0.63 位置分=偷懒局部最优)
  值得产品线走梯子验证 → 排为候选臂 R17


### 阶段 1「固定点养成」:学会用合适的角度和速度,把不同速度的无旋球打回固定落点

开关(全部默认关,现状逐字节不变;合并安全):`题目模式(现状/反解)`、`击球点(固定/框/真球)`、
`旋转档位(无/低/场馆)`、`观测契约(175/178+版本号,侧向速度等预留扩展位——demanded 拍速矢量
本就在观测里,反解后自动携带侧向分量;新增仅拍面 3 维)`、`torch 求解器(开=训练内联反解)`。

**0k 反手 bank 落地 + 三层语义对账(yikang 2026-07-06 凌晨,机械全绿)**:
两个锚点的权威数字——f21 真值帧:可解率 85%/84%(train/exam),难度(需求面 vs 标定面)中位
33.6°,锥内 0%,torch 闭环 2.0cm ✓;f17 候选帧:94%/91%,38.4°,锥内 0%,2.1-2.2cm ✓。
**意外(如实报)**:候选帧没有点亮锥内可答率、难度反而更高;且 raw 代理面下难度 25°、烤入
标定面后 38°——早前"f17-19 100%"与 bank 数字**三层语义并存、互不矛盾**:
①我的 100% = **存在性 × 任意合法落点**(锥内扫出过合法回球,落哪都行);
②bank 的 33-38° / 锥内 0% = **最省力解 × 指定落点**(LM 默认取全局最省力答案,不测锥内是否
另有解);③难度必须以**烤后标定面**为基准(raw 代理面的 25° 是旧语义)。
bank 的任务(指定落点)才是真训练任务;**答案合法地落在挥速锥外,正是拍面指令臂要证明
"学得动"的东西**。可选优化:LM 初值改 clip 挥速/加锥约束(锥内优先解,半小时,不阻塞)。
**判卷铁律(yikang):回球率必须连分母一起记**(可解率 91-94%、锥内 0%),防止被误读成
场馆水平。推荐决策:**直接点火,用 f21 真值帧版 bank**(难度低 5°,锚语义与考卷/登记表一致)。

**✅ 已点火(2026-07-06 深夜,franco 下令)——S1 第一波打了 40 分钟止损重打,
现役 = 修正版六臂全部在跑 + 两臂动作源对比排自动队列(共 8 臂,3 卡 6 槽占满)**。

**止损原因(两个新发现,都已进制度)**:
1. **题库只管目标(位置/速度/拍面),不管时机**——击球时机走 `strike_phase_per_clip`,
   而默认值 (0.47, 0.333) 是 **hopex/v4 的真值**(产品线一直练 hopex 对,所以默认没错过);
   v5 真值是 (0.673, 0.362)。第一波四臂正手时机错了 0.2 个 clip(几十厘米的拍位差),
   正手题基本不可学。**铁律:换动作对必须显式传相位**。
2. 六条 _cal 都没做 +X 重落地(帧 0 朝向偏 +17.5°/+19.5°)。判定:**本波不动资产**——
   当时 rally_yaw 是在未落地的 clip 上反解的,倾斜已被吸收进登记值,题库/登记表/npz
   三者自洽;但「产品线原样」锚改用 hopex 对+默认相位 = 和 P2 逐位同条件。落地债进管线
   (新 clip 必须先重落地再标定)。
3. 附带制度:警告行(WARN)必须进机制检查摘要——这次的落地警告在日志里躺着,
   我的摘要 grep 没抓到,是第一波带病出发的直接原因。

| 臂(run_name) | 卡 | 组合 | 回答 |
| --- | --- | --- | --- |
| s1w2_A_main | GPU0 | v5cal 对+题库+奖励锚指令+**拍面观测 179**+腕踢除 | 方向成不成(主攻) |
| s1w2_B_noobs | GPU1 | A 减观测(175,SMASH 式"速度隐含") | 观测通道净贡献(A−B) |
| s1w2_C_targets | GPU2 | 题库+腕踢除,拍面奖励仍锚 clip | 反解目标单独净贡献(C−D) |
| s1w2_D_base | GPU0 | **hopex 对+默认相位**(=P2 真条件),无题库 | 产品线锚 |
| s1w2_E_weight | GPU1 | A + 拍面奖励权重 ×3(0.5→1.5) | 新老师话语权粗标定 |
| s1w2_F_bhf17 | GPU2 | A 但反手锚=f17(相位 0.298,可回球性最优帧;合并卷) | 相位臂:真值锚 vs 训练最优锚(A−F) |
| s1w2_G_v4mo | 队列 | 主攻旗标 + **v4 对**(hopex 视频重跑标定版)+v4 卷,相位 [0.47,0.333] | 动作源对比:最稳一对 |
| s1w2_H_obmo | 队列 | 主攻旗标 + **斜录对** + 斜录卷,相位 [0.432,0.495] | 动作源对比:斜录能否被相位/标定救回 |

共同配置:4096 envs、同种子、P2 配方、热启 P2 model_9600(179 臂用扩列存档)、4000 新迭代
(计数 9600→13600)。题库:A/B/C/E=`/workspace/yikang/s1_v5cal_train.npz`(822 正+707 反);
F=`/workspace/franco/s1_banks/s1_fhann_bhf17_train.npz`(正手真值卷+反手 f17 卷合并);
G/H 卷由 gen_stage1_questions 现场生成(v4 登记项当场补齐并推了小分支 `s1-registry-v4cal`
给 yikang 拣)。**自动队列** `s1_queue.sh`:等题库+空槽 → 先 512×25 冒烟(摘要含 WARN)→
干净才发真臂。日志:`/workspace/franco/s1_wave1/`,运行目录在 nohope_s1 worktree 的 logs 下。

**发射工程记录(躲过/踩掉的坑,复用给以后每一波)**:
1. **175→179 热启靠"观测扩列手术"**:现成 tolerant 加载器明确拒绝 actor 形状变化;
   解法=把 P2 存档的 actor 首层权重、优化器动量、obs 归一化器共 6 个张量在末尾补 4 列
   (权重/动量补 0,归一化 mean 补 0/var 补 1)→ 第 0 步行为与 P2 逐位相同,严格恢复成功
   (`pad179.py`,产物 `/workspace/franco/s1_ckpt/model_9600_pad179.pt`)。
2. hydra 对 yaml 未声明的新键要求 `++` 前缀(question_bank/face_command/腕踢除三旗都是)。
3. worktree 没有机器人资产(assets/agibot_a3 被 .gitignore 全量忽略)→ 从主检出软链。
4. **两个 Isaac 同秒启动会撞 CUDA 枚举**(报"no suitable CUDA GPU")——错峰 ≥60s
   铁律的又一实证;机制检查也必须错峰。
5. Isaac 退出码不可信(异常后仍 exit=0),判活/判死只能看日志签名。

**判卷(各臂 ~4-6 小时后落地,双卡共跑会拉长)**:各臂 model_13600 → ONNX 导出(179 维臂
注意导出器输入维自检)→ 阶段 1 考卷:**按侧各跑一趟固定锚点、锚点用各臂自己的考卷**
(v5 卷:正手 env(0.379,−0.547,0.856)、反手 env(0.772,0.227,0.971);F 反手用 f17 锚;
G/H 用 v4/斜录卷各自锚)、无旋、场馆速度档。北极星=回球率,**起飞线 ≥50%**(反事实上限
100%);**分母一起记**(v5 卷可解率 85%/84%,锥内 0%——超锥拍面正是拍面指令臂要证明
学得会的东西)。**尺子铁律(franco 2026-07-06):击球率/上台率正式入账必须是 MuJoCo 版**;
训练内 Isaac 虚拟球的两球率只作过程监控,可并列报但必须标注(详见 runbook 判卷铁律)。

### 发射工序(2026-07-06 立;**完整核对单与全部运维坑固化在 [runbook.md](runbook.md)**,此处只留队列)

发射前过 runbook 的十条核对单(相位显式传/登记五件套/卷动作同源/契约匹配/++语法/错峰/
先 ps 认领卡上进程/日志目录先建/摘要含 WARN/run_name 入表);判卷链(原生导出→mjeval 环境
→按侧锚点)与"判卷链也要机制检查"同在 runbook。

**点火队列(跨阶段统一;所有想跑的臂都进这张表,不再散落各节靠记忆)**:

**⚠ 算力变更(2026-07-06 凌晨,franco 指令)**:**gpu0 整卡让给 jiayi 测试**,我方收缩到
gpu1/2 双卡运作。执行:D 臂自然到线退出;A 臂在 model_12500 检查点停下、在 gpu2 空槽上
严格恢复续跑 1100 步到 13600 对齐点(run 名 s1w2_A_main_r2,只丢 2 步+约 20 分钟墙钟);
B 导出/考卷改走 gpu1/2 空隙 + CPU;F 顺延到 C 退出后的槽。调度器全部改为只认 gpu1/2。

| # | 臂 | 依赖(什么好了才能排) | 状态 |
| --- | --- | --- | --- |
| 到线 | s1w2_B(隐含)/ s1w2_D(锚) | — | 判卷中 / 已完成 |
| 跑着 | s1w2_C/E/G/H + A 接力(_r2) | — | gpu1/2 双卡 |
| 队 1 | s1w2_F_bhf17(相位臂) | 双卡编排器守着 C 退出的槽 | 冒烟后自动上 |
| 队 2 | SMASH 击球窗分通道奖励臂 | **代码实现**(认领:franco/claude) | 实现完即插 |
| 队 3 | 相位依赖任务噪声臂(A1 叠成 v3) | 代码实现 | 同上 |
| 队 4 | 权重细扫 2 档 × 容差 2 档 | E 臂出信号定方向 | 等判卷 |
| 队 5 | curriculum(难度窗口 vs 全开) | loader 难度窗口支持(待确认,yikang) | 等确认 |
| 队 6 | f18/f19 备选锚臂 | F(f17)出信号再定 | 等判卷 |
| S2a | p_racket 放开臂 ×2(题点 box 两档) | S1 过线(回球率 ≥50%)+ 出题器 point_mode box(yikang) | 升段判据 |
| S2b | 手补/身补/转补三臂(Δ 分档) | S2a 过线 + 位移球卷生成器;R12 减速并入身补臂 | 升段判据 |
| S3 | 旋转两档 × 消旋奖励有无 | S2 过线 + 考卷加旋(评估器开关已有) | 升段判据 |
| 平行 | 适配器 v2 变体库(CPU)/ L1+L2 门禁 / 连续挥拍终审 | 不占训练槽 | 可随时做 |

**阶段 1 预算合计 ≈ 2 波 ≈ 1-2 天墙钟;每阶段判据过线才升段。**

### 阶段 2「位置应变」(franco 半圆模型重设计,2026-07-06;做成消融回答"手补还是身补")

**架构升级(franco 2026-07-06:题目从同一分布随机取,接口按题出老师)**:阶段 2 起,适配器
进训练循环——每道题(来球方向/位置/速度)由接口现场改造老师(整身旋转 φ=题目方位、变速
s=题目节奏、拍面/侧向 morph=小臂带动全链渐变或变体库检索),策略学的是"题-老师-指令"三元组。
阶段 1 是它的退化版(老师固定);接口三轴在阶段 2 消融各自净贡献(转/速/面 逐轴开关)。

**模型**:对人/机器人,可击球面近似**以身体为心的半圆**;应对来球位置变化有三个通道——
A 手补(p_racket 在臂展内伸,尤其可沿 x 往前多探一点)、B 身补(p_base 平移;人实际走斜线,
可能因为横移难/需要面朝前,采样先验按斜线设计)、C 转补(整套动作绕身旋转适配方向——
rally_yaw 机制的免费红利:同一条 clip 旋转后天然服务任何打球方向,幅度限制在两侧小角度)。

**核心实验(franco:同样的球,用手的移动弥补 vs 身体的移动弥补,哪个更容易)**:
同一批"位移球"(来球接触点相对锚点横向/纵深偏 Δ,Δ 分档 5/10/20/30cm),三臂同批:
- 臂 A 手补:p_base 钉死,题目的拍点=位移后接触点,反解目标照常——策略只能伸手;
- 臂 B 身补:base 目标随球平移(斜线先验),拍点保持在身前"甜区半圆"内——策略必须挪步;
- 臂 C 转补:题目整体绕锚点旋 φ=球方位角,动作参考同旋(rally_yaw 机制复用)——策略"转身打";
- 判据:各 Δ 档的回球率 + 三合格率 + 摔倒/力矩代价 → 得到"每厘米位移的最优通道"分界表,
  直接变成部署时 planner 的分工规则(多少厘米内用手、多少用步、多少用转)。
赢家热启自阶段 1;R12 减速入位在臂 B 并入配套;远球/跑动考卷同步建。预算 1-2 波(3 臂×Δ 课程
+ 对照)。

### 阶段 3「旋转进场」:无旋→低旋→场馆水平

反解自动给出随旋变化的拍面/切向速度目标;消旋奖励从此才有真效用。1 波 4 臂(旋转两档 × 消旋
奖励有无 + 对照)。

**总预算粗估:5-6 个舰队波次 ≈ 30-36 臂 ≈ 一周内(现有 3 卡×2 槽)。每阶段判据过线才升段,
考卷与训练同阶段同步(变速考卷并入阶段 1 考卷家族)。**

搭车项(不占主线):R12「刹得住」专项考(并入 2b)、R17 位置奖励收紧(阶段 1 第二波搭车)、
R15/斜录重赛(相位校准后并入阶段 1 相位臂)。
