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
