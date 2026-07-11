# G06 Isaac-To-MuJoCo Parity

Status: Partial (parity procedure operational and used to gate the 2026-07-02 sim-to-real; formal per-checkpoint acceptance thresholds still to be recorded)

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
- [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)
- [../operations/run_gate3_first_tick_harness.md](../operations/run_gate3_first_tick_harness.md)

## Acceptance Criteria

- The same action ordering is verified in both simulators.
- The exported deploy ONNX (not a re-export) runs in MuJoCo with the training observation rebuilt exactly.
- Divergence sources are documented: contact, latency, actuator, timestep, observation delay, model mismatch.
- Exact-strike metrics from Isaac are reproduced in MuJoCo and recorded per accepted checkpoint.

## Current State

Done (2026-06-27 → 2026-07-02, recorded 2026-07-03):

- The parity procedure exists and is battle-tested: `scripts/mujoco_eval_onnx.py` loads the exact
  exported deploy ONNX, reads the whole actuator contract from ONNX metadata (joint_names,
  default_joint_pos, action_scale, kp/kd, body_names — fails loudly if missing), auto-detects the
  175-D deploy-parity vs 180-D legacy obs contract, rebuilds the Isaac actor observation in MuJoCo
  (same frame math; the deploy-honest racket-target reframe is verified by
  `scripts/realsensor_obs_reference.py`), and reproduces Isaac's exact-strike metrics
  (pos/vel/normal pass, composite, hit-speed error, velocity attainment) with per-clip
  forehand/backhand breakdowns and per-step CSVs.
- The dominant divergence source was isolated to actuator PD integration: with the same ONNX and
  byte-identical `a3_pingpong.xml`, MuJoCo with `implicitfast` + kd in `dof_damping` (Isaac
  `ImplicitActuator` equivalent) is stable with clean swings, while the AGI deploy sim's
  explicit-Euler PD path (`joint_actuator_subscriber.cc`, MJCF without an integrator attribute,
  passive damping not zeroed) diverges within ~0.1 s. Switching only the PD integration moved
  hit-speed error 0.61 → 0.31 m/s and velocity attainment 0.35 → 0.88. One-flag reproduction:
  `--pd-mode implicit` vs `--pd-mode explicit --keep-passive`. See
  `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md`.
- Current verdict stance (2026-07-02): implicit PD remains the Isaac-faithful cross-check, but the
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
- No decision recorded on MuJoCo as a training backend (currently it is a validation/dry-run stage
  only).

## Risks

- A policy can appear valid in Isaac but fail in MuJoCo because of actuator/contact mismatch — this
  happened (explicit-PD divergence) and cost significant time before the root cause was isolated.
- Evaluating with the script's stale defaults silently tests the wrong contract; always pass the
  checkpoint's own clips/phases.

## Next Steps

1. Record the accepted sim2sim numbers for the shipped checkpoint (implicit cross-check + explicit
   clipped-PD gate + `--deploy-faithful` protocol) in this gate.
2. When the mocap→planner bridge lands, extend the MuJoCo rehearsal to consume live
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
K=100, 50-per-side paper, model 2000 returned FH/BH/aggregate
`0.66/1.00/0.83`, while model 4000 returned `0.00/1.00/0.50`; model 2000 is
retained and the whole arm continues. Both evaluations are exact/fresh, but
all attempts finalized through a non-physical post-strike guard, so this is
not a continuous or deploy-stability gate. The result is bound in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`. Its fresh
same-paper Isaac companion gave both checkpoints `0.98/1.00/0.99`
FH/BH/aggregate, delta zero. The MuJoCo ranking is therefore not reproduced;
the cross-engine checkpoint gate stays open. Companion hashes are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

Question-level forensics now localize both disagreements and prevent a false
gate closure. Fresh model 4000's mean FH racket-center error is `13.15 cm` in
MuJoCo, beyond the frozen `9.5 cm` contact margin on all 50 questions, versus
`2.48 cm` in Isaac; model 2000 is `9.03/3.03 cm`. M3-old BH has mean signed
normal error `168.15 deg`, but Isaac's analytic `orient_normal` removes the
sign before scoring, while MuJoCo physical contact keeps the consequence.
Thus same question bytes/order are necessary but not sufficient: current Isaac
virtual outcome and MuJoCo physical outcome are different instruments. The
forensic result is bound in
`configs/phase1_cross_engine_saturation_forensic_result_20260711.json`.

The next gate is preregistered as a strict 2x2: Isaac/MuJoCo x physical
truth/analytic counterfactual, with the original K100 order and capture/speed
thresholds frozen. Missing/duplicate/non-finite cells, changed order, or a
virtual-only physical cell all fail closed. Numeric Isaac ready/base/racket,
signed-face-before-orient and analytic state instrumentation is implemented,
but Isaac PhysicalBall currently has incoming-flight Phase A only and no
racket impulse/post-contact truth. Until Phase B and one content-addressed
four-cell evidence manifest exist, G06 remains `Partial`.

Run `python3 scripts/validate_phase1_queue_governance.py` before any curve
manifest is copied or launched. The validator enforces the 142-job/24-slot
q10 screen contract and refuses q50 through the generic worker. Plant parity
remains separate: SZ is only zero-friction protocol exact, while SP/LP are
historical direct-number proxies. The repair contract is
`docs/research/phase1_plant_semantics_repair_2026-07-11.md`, status
`blocked_on_calibration_evidence`.

### 2026-07-12 final-engine priority

The final behavioral arbiter is the Agibot-provided A3 MuJoCo deploy chain called Gate 3 in
`docs/operations/run_pingpong_end_to_end.md`: fake ball -> real planner -> production-equivalent
C++ runner -> vendor MuJoCo. Isaac remains the training engine and a diagnostic companion; an
Isaac win cannot promote a checkpoint that fails Gate 3 balance, completion or recovery.
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

### Gate 3 face-command wire and engine-gap localization

The 179-D Phase-1 policies cannot be tested by adding `179` to a shape whitelist. Their last four
columns require the planner's demanded world-frame normal and a zero rho placeholder atomically
paired with position/velocity. A versioned flat schema-2 publisher/receiver and exact
`deploy_parity_face179` ONNX metadata path are now implemented in source. The loader additionally
requires `face_command_enabled=1`, `shared_plus_y`, `mount_plusY_A`, an exact schema-3 train bank,
train split and lowercase content/source-family SHA-256 bindings; width and term names alone are
not enough. Schema-2 rows require a world-frame opponent-facing unit normal (`x>1e-6`) and zero
rho. Any malformed/unknown row after an active face tuple records `invalid_after`; the publisher
turns a bad solve or payload into an explicit finite `valid=0` row on both wires, so silence cannot
keep an old swing eligible for the longer command timeout. Schema 1 remains the default for
existing models and cannot engage a 179 actor. This is not yet a gate result: the
vendor-source offline x86 build is recorded below, while a ROS/AimRT-enabled build, no-publish
first-tick parity trace, and full Gate 3 MuJoCo run are pending.
Active-swing fields are atomic. Post-swing recovery is not yet exact: the current Gate 3 runner
combines a synthesized base-anchored hold position with the previous swing's velocity/normal,
and no Phase-1 contract proves that hybrid tuple is on-distribution. A canonical recovery tuple
or separately accepted vendor-MuJoCo recovery paper is required before continuous promotion.
The positive-X invariant is also only a minimum sign/frame guard. A content-bound per-clip normal
envelope from the exact training bank is not yet exported; until it exists, arbitrary positive-X
normals and their self-collision consequences remain a Gate 3 runtime blocker.

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

The production runner now also has a fail-closed `--model-preflight-only` path. It requires
`--no-publish` or `--dry-run`, constructs `PpPolicy` before any backend object is created, and on
success emits the accepted observation width plus training-contract and source-checkpoint SHA-256
before exiting. This safely separates “the formal 179 export is loadable under the production
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
process remained. This closes formal-model production loading only. First backend tick, per-clip
normal envelope, canonical recovery tuple and full vendor MuJoCo Gate 3/Gate 3B behavior remain
open, so G06 stays Partial.

#### 2026-07-12 Gate3 first-tick static plan gate (red-team corrected)

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

The plan explicitly keeps five runtime blockers null: current C++ lacks full
`--first-tick-json`; exact process ownership still needs pidfd plus a cgroup/reviewed supervisor
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
