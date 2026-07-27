# RunPod Training Audit — 2026-07-10

Status: historical live-fleet/training-evidence audit plus Phase-1 measurement requirements. Initial fleet snapshot:
2026-07-10 07:44–07:56 UTC; follow-ups: 14:35–14:36 UTC and 15:46 UTC (23:46 CST). No training
process was stopped or modified. Execution contracts and current authorization are governed by
`research/full_stack_audit_closure_2026-07-10.md` and `NOW.md`, not by this historical snapshot.

## Bottom Line

There is real Stage-1 progress, but there is still no accepted policy-quality baseline.

- The project has moved from the old `0%` fixed-point return regime to several policies that return
  balls in MuJoCo. The strongest new evidence is the end-to-end S1 face-frame repair on two motion
  families: M3f backhand return `0.929 / 0.917` and M2f3 backhand return `0.750 / 0.538`
  (`ns=0 / 0.05`).
- These numbers are still small-sample fixed-point, single-ball bank results. The corresponding
  per-noise cells contain only 2–14 hits, continuous hitting is unmeasured, and Gate 3B is not built.
- Pod2's eleven completed arms reached `model_16999`, but their only judge reports and shared
  `exported/` artifacts are from the earlier `model_16400`. Their terminal policies have not been
  exported or judged.
- By the 14:35 UTC follow-up all six GPUs were idle; the 15:46 UTC recheck was unchanged. G1b had
  finished at `model_20998` and produced
  a terminal old-style judge report; the eleven older Pod2 arms still have the terminal-export debt
  above.

Accordingly, use the terms **fixed-point single-ball candidate** and **mechanism evidence**. Do not
call any current arm a product-quality winner or a completed Stage-1 baseline. A later red-team of
the evaluators also found that the old Isaac bank path paired an answer with an independently
sampled virtual incoming ball, and the old MuJoCo headline denominator contained only policies
that survived to the exact-strike frame. Historical percentages below therefore remain useful for
mechanism screening, but they are not the requested full-attempt two-simulator scorecard.

## Live Fleet Snapshot

| Pod | Endpoint | GPU state | Training state | Queue/judge state |
| --- | --- | --- | --- | --- |
| pod1 | `162.43.172.171:18333` | GPU0/1/2 P8, 0%, 0 MiB at 15:46 UTC | no training process | watchdog healthy; 26 unique expected arms all have terminal checkpoints and `DONE+JUDGED`; no ready item was missed |
| pod2 | `162.43.172.181:13146` | GPU0/1/2 P8, 0%, 0 MiB at 15:46 UTC | no training process; G1b finished at `model_20998` | G1b terminal old-style report exists; eleven older terminal checkpoints remain only `EARLYJUDGED` at 16400, with deleted recovery metadata still missing |

Pod1 being idle is a normal completed queue. Pod2 is now also idle, but the eleven-arm terminal
export/judge debt remains because the watchdog cannot reconstruct the deleted helper and command
records.

Execution note: the 17:12 CST follow-up was initially blocked by the Codex execution allowance, but
read-only SSH resumed after the window and produced the 14:35 and 15:46 UTC status above. This is
an evidence record, not a request to restart, export, judge, or train any arm.

## Evidence Rules Used In This Audit

1. Historical Isaac `virtual_return_rate_rally*` values are now labelled **training proxies**.
   Their per-swing ledger repair is real, but the pre-remediation Stage-1 bank path sampled the
   answer row and incoming virtual ball independently. They cannot fill a public-exam two-sim cell.
2. Formal return claims use MuJoCo judge output with the checkpoint named explicitly. Training
   metrics are diagnostic only.
3. The judge report's sentence `CF 显著高于实测=失分在拍面通道` is ignored. It is emitted
   unconditionally by `scripts/judge.sh`, even when CF equals or is below the measured result.
4. A result with fewer than 50 attempts per side and noise level is a screening result, not a
   stable rate estimate. Current reports are below that bar.
5. Different motion families' self-generated banks are not assumed to have equal difficulty.
6. Formal no-fall means no absolute balance event (`base_fell_tilt` / `base_too_low` in Isaac;
   `fall_tilt` / `fall_root_z` in MuJoCo). Tracking-envelope resets are reported separately and are
   not renamed as physical falls.
7. The three headline rates share one denominator: eligible, ended attempts. Timeout, evaluator
   stop, or aborted in-flight attempts are explicit censoring, never silent no-fall successes.

## What Each Phase-1 Training Arm Tests

This table states the scientific question before looking at the score. `Continue` here means
evaluate or run a clean control; it does not automatically authorize more PPO iterations.

| Arm | Direct parent / actual change | What the training tests | Current decision |
| --- | --- | --- | --- |
| M2 | healthy v4rg baseline | Whether the Stage-1 recipe transfers on the clean reference/action foundation | Freeze as the common parent; expanded exam only |
| R1b | M2 minus wrist linear-velocity imitation | Whether task-demanded racket velocity should control the wrist instead of teacher wrist speed | Shortlist; expand both policy seeds |
| R5b | M2 plus gentle linear face guidance (`-0.95`) | Whether a non-dead-zone face gradient improves legal returns | Shortlist; seed1/seed2 paired exam |
| R8b | M2 plus envelope-as-penalty and RSI stand-z hold | Whether the two-flag stability package reduces early loss without sacrificing return | Package test, not one-variable proof; terminal seed2 exam first |
| R10c | R9a plus 2-D world station-error observation; 179→181 pad | Whether a world station anchor can replace the deleted reference tether and expose the interface later needed for movement | Current fixed-point implementation failed transfer; **retain the station-anchor interface** and split it from lower-body flags in the Hitter stationary→moving route |
| M3f | M3c terminal + shared S1 face-frame helper +4000 | Whether one face convention across bank→obs→reward→judge rescues swing backhand | Strong mechanism signal; needs old-helper same-duration continuation control |
| M2f3 | M2f terminal + shared S1 helper +4000 | Whether the same convention repair transfers to v4rg | Positive cross-asset mechanism signal; same continuation confound |
| ST1 | M2 plus staggered initial clock | Whether desynchronizing env clocks fixes measurement artifacts without changing performance | Measurement-validation arm, not a product candidate |
| G1 | corrected-anchor swing → swingsyn timing family | Whether the synthetic timing law generalizes to the swing asset | Backhand negative control; terminal record then stop |
| G2 | M2 v4rg → v4rgsyn timing family | Whether synthetic timing helps or is at least harmless on the healthy asset | High-priority candidate, but timing/phase/length/bank changed together |
| C1 | R1b × R5b | Whether wrist-speed decoupling and gentle face guidance are complementary | High-priority interaction screen |
| C2 | R1b × R8b package | Whether wrist-speed decoupling complements the two-flag stability package | Expand only if terminal result is balanced by side |
| C3 | R5b × R8b package | Whether face guidance complements the stability package | Promising balanced screen; expanded exam |
| C4 | R1b × R5b × R8b package | Whether the full winner package beats its singles and pairs | Treat as a package/factorial cell, never a single-variable arm |
| N1 | M2 plus an actor-visible target-position error that is large when the ball is first seen and shrinks as the strike approaches | Whether the policy keeps adjusting its swing as real ball-position estimates become progressively more accurate | The real problem is valid; this `0.15 m`, independent-per-step approximation hurt performance. Re-test with measured, time-correlated prediction-error traces rather than abandon the direction. |
| T3 | R9g v5syn → v5topp nonuniform timing | Whether oracle timing reduces falls while preserving returns | Current backhand failed; archive after terminal record, redo only after S1 on a clean single-variable setup |
| G1b | G1 terminal + S1 face-coordinate repair + face guidance `-0.4` +4000 | Whether that combined repair can recover any G1 backhand return | Terminal small screen partially recovered backhand (`0.222/0.286`), but the three simultaneous changes prevent single-cause attribution |

Necessary parents/controls are M1 (v5 negative), M3/M3b/M3c, M2f, R9a, R9g, both R5b seeds,
and both M2/R1b seeds. G1/G2/T3 also change timing, phase, clip length, and bank; M3f/M2f3 add
training duration; R8b/C2/C3/C4 contain multi-flag packages. Rankings must respect those confounds.

## Current End-Checkpoint Evidence On Pod1

Rates below are return rates. MuJoCo cells are `ns=0 / ns=0.05`; `asked` is forehand/backhand.

| Run | Checkpoint | Isaac last-21 rally | MuJoCo forehand | MuJoCo backhand | Asked | Audit verdict |
| --- | --- | --- | --- | --- | --- | --- |
| R10c station observation | `model_16999` | total `0.711±0.006`; FH `0.698`, BH `0.724` | `0.250 / 0.286` | `0.250 / 0.200` | `12 / 25` | Failed its preregistered goal of restoring M2-level transfer. Large Isaac→MuJoCo gap remains. |
| M3f swing + S1 face frame | `model_20998` | total `0.684±0.016` | `0.667 / 0.875` | `0.929 / 0.917` | `15 / 31` | Strong face-frame rescue signal; BH face error fell to `7.67° / 5.50°`. Needs matched continuation control and larger exam. |
| M2f3 v4rg + S1 face frame | `model_20998` | total `0.723±0.060` | `0.667 / 0.200` | `0.750 / 0.538` | `9 / 29` | Cross-motion repair signal is positive, but noisy forehand/noise robustness is not accepted. |

M3f improves over M3c's backhand return `0.308 / 0.333` and roughly `33–34°` face error. M2f3
improves over M2f's backhand return `0.417 / 0.400` and roughly `29–30°` face error. Both repair
runs also add training iterations, so a pure-continuation control is still required for clean
causal attribution.

R9t, R9u, and M3d failed together during the 2026-07-09 18:28 UTC CUDA event and were deliberately
not revived because they used the superseded face convention. They have no judge report and must
not be used as performance evidence.

## Pod2 Screening Results And Missing Terminal Exams

All rows below are **early `model_16400` judge reports**, not the existing terminal
`model_16999` checkpoints. Rates are MuJoCo return `ns=0 / ns=0.05`.

| Run | Asked FH/BH | Forehand | Backhand | All-side | Screening interpretation |
| --- | --- | --- | --- | --- | --- |
| R5b seed2 | `13 / 25` | `0.000 / 0.167` | `1.000 / 0.900` | `0.750 / 0.625` | It worked on backhand but almost never returned forehand; this does not validate a two-sided R5b winner. |
| ST1 stagger | `15 / 24` | `0.000 / 0.111` | `0.846 / 0.625` | `0.733 / 0.353` | No demonstrated performance gain from staggering. |
| G1 swingsyn | `11 / 19` | `0.500 / 0.667` | `0.000 / 0.000` | `0.154 / 0.308` | Backhand failed. |
| G2 v4rgsyn | `12 / 17` | `0.500 / 0.667` | `0.900 / 1.000` | `0.833 / 0.833` | Best training-side arm and a promising screen; terminal exam required. |
| C1 R1b×R5b | `13 / 26` | `1.000 / 1.000` | `1.000 / 1.000` | `1.000 / 1.000` | Highest screen, but far too few attempts and not terminal. |
| C2 | `13 / 26` | `0.333 / 0.000` | `1.000 / 1.000` | `0.867 / 0.588` | Backhand returned every scored ball, while forehand returned only one third at zero noise and none at `0.05`. |
| C3 | `13 / 26` | `0.667 / 0.571` | `0.833 / 0.900` | `0.800 / 0.765` | Balanced promising screen; CF is not higher than measured. |
| C4 three-way combination | `14 / 24` | `0.667 / 0.500` | `0.923 / 0.750` | `0.875 / 0.611` | Promising screen; terminal exam required. |
| N1 progressively improving target estimate | `13 / 27` | `0.500 / 0.167` | `0.000 / 0.000` | `0.125 / 0.067` | This `0.15 m` independent-per-step approximation failed, especially on backhand; it does not reject the real progressively improving-data requirement. |
| T3 v5 timing | `12 / 24` | `1.000 / 1.000` | `0.000 / 0.000` | `0.333 / 0.286` | Backhand failed; Isaac fall rate was about `0.26`. |
| R8b seed2 | `11 / 27` | `0.667 / 0.429` | `0.333 / 0.455` | `0.400 / 0.444` | Not a confirmed winner. |

Training-side last-21 rally means were approximately `0.70–0.72` for most R5b/ST1/C1–C4/R8b
runs, `0.768` for G2, `0.340` for G1, and `0.247` for T3. This confirms that similar Isaac rates
can map to very different MuJoCo outcomes and that a training-only ranking is unsafe.

G1b contained both the S1 face-coordinate repair and a `-0.4` linear face-guidance reward, and also
continued G1 for 4000 more iterations. It is therefore not a single-variable experiment. Its
terminal `model_20998` old-style MuJoCo screen returned forehand `1.000 / 1.000` from only `3 / 5`
hits and backhand `0.222 / 0.286` from `9 / 7` hits (`ns=0 / 0.05`); contact was `1.000` in all four
survivor-denominator cells. This partially rescues G1's old backhand `0 / 0`, but cannot separate
the coordinate repair, guidance reward, and extra training. The counterfactual score exactly
equalled the measured score, so the historical report's automatic sentence blaming the remaining
loss on the face channel is false for this run.

## Required 24-Cell Coverage Status

| Side × metric | Isaac fixed-point single ball | MuJoCo fixed-point single ball | Isaac continuous hitting | MuJoCo continuous hitting |
| --- | --- | --- | --- | --- |
| FH physical no-fall | Old `1-pre-post` conflates every non-timeout termination with a fall | Old judge has no full-attempt physical-fall denominator | Not formally recorded | Not formally recorded |
| FH hit rate | Training proxy; old question answer/incoming ball were not atomic | Strike-survivor screen; pod2 terminal checkpoints untested | Training proxy only | Not measured |
| FH return rate | Training proxy; not the public exam | Strike-survivor screen; no common full-attempt exam | Training proxy only | Not measured |
| BH physical no-fall | Old `1-pre-post` conflates every non-timeout termination with a fall | Old judge has no full-attempt physical-fall denominator | Not formally recorded | Not formally recorded |
| BH hit rate | Training proxy; old question answer/incoming ball were not atomic | Strike-survivor screen; pod2 terminal checkpoints untested | Training proxy only | Not measured |
| BH return rate | Training proxy; not the public exam | Strike-survivor screen; no common full-attempt exam | Training proxy only | Not measured |

No arm currently has the full twenty-four-cell report required by `NOW.md`. This is the direct reason
the audit does not name an accepted winner.

## Audit Of Prior NOW Conclusions

| Prior conclusion | Audit disposition | Reason |
| --- | --- | --- |
| The core fixed-point direction works | **Keep, with scope** | Healthy-reference policies now produce nonzero and sometimes >50% MuJoCo single-ball returns. This is mechanism progress, not a product baseline. |
| R1b and R5b are winners | **Candidate only** | Existing judge cells are tiny; in R5b seed2 the forehand was near zero while the backhand was high, and same-recipe small exams have varied by >0.4. |
| fixC/v4 backhand beats fixE/trim6 | **Keep direction** | fixC transferred; fixE fell in all MuJoCo trials with zero hits. Exact rates are still bank-difficulty and sample-size dependent. |
| GMR cold-start was the main v5 cause | **Downgrade** | Warm-up fixed a real first-frame defect, but later v5rg backhand still failed. Deep squat, path/contact load, strike speed, and follow-through remain. |
| Support-foot skate is the sole/primary cause | **Downgrade** | The skate measurement is real, but clean-foot v5 variants still fall and low-skate M1 forehand also fell. It is one cause, not a sufficient explanation. |
| Timing showed a smooth monotonic dose law and oracle prediction was validated | **Retract precision claim** | R9f was predicted at `0.4–0.6` fall but measured near `0.20`; R9e and R9f were nearly tied. Timing helps across the large 3.5→4.49 regime but does not explain the remaining floor. |
| Removing the anchor tether is free or beneficial | **Retract** | R9a looked healthy in Isaac but lost MuJoCo contact; R9g restored transfer without the three free-lower-body flags. R10c station observation did not repair the gap. |
| Every backhand must simply flip the racket-face sign | **Retract** | The question-bank→observation→reward→judge chain used a self-consistent +Y gauge. A one-sided sign flip broke the gauge. The S1 shared face-frame helper is the supported repair. |
| R10c station observation is the live route to restore the anchor | **Current recipe failed; interface retained** | Terminal MuJoCo all-side return was only `0.250 / 0.235`, despite Isaac proxy rally `0.711`. This rejects the fixed-point R10c implementation, not the station-anchor direction needed by jiayi's Hitter for stationary→moving control. |
| Isaac repaired rally rate is already the public-exam full-attempt rate | **Downgrade to proxy** | Its same-ledger fix prevents numerator/denominator drift, but the old bank path independently sampled incoming-ball physics. |
| MuJoCo bank return already counts pre-strike falls as zero | **Retract** | The old headline was `landed_ok / n_strikes`; an attempt that terminated before exact strike never entered `n_strikes`. |
| Old composite ≈0.91 or wave-2/3 absolute rates prove quality | **Retract as quality evidence** | Composite is diagnostic, old rates used invalid targets/semantics or survivor denominators, and product-line return was zero. |
| Stage 1 is complete / v5 is rescued | **Not supported** | No accepted common exam, no terminal Pod2 exams, no continuous protocol, no Gate 3B, and current v5 timing/face-rescue arms still have zero BH return. |

## Next Tests, In Order

### P0 — Repair The Measurement And Recovery Path Before More Broad Training

1. Restore Pod2's `queue.md`, `patrol_helpers.py`, and arm command ledger. Move terminal-checkpoint
   recognition ahead of any revive-only dependency so a completed run can still auto-judge.
2. Explicitly judge all relevant `model_16999` checkpoints. Regenerate ONNX, observation norm, and
   learned-std sidecars from that exact checkpoint; do not reuse each run's current `exported/`.
3. Fix the unconditional CF interpretation sentence in `judge.sh` and compute the actual measured
   vs counterfactual difference before assigning a face-channel cause.
4. Use one paired bank and fixed question order, at least 50 attempts **per side and noise level**,
   at least two policy seeds, and report Wilson/bootstrap intervals. Start with M3f, M2, G1/N1,
   then G2, C1, R1b, R5b, and ST1; defer guard-package arms as below.
5. Run the Pod unit/runtime check for the new per-run termination contract before formally scoring
   R8b/C2/C3/C4 or any R9-derived recipe. Read-only copies of real M2/R8b/C3/G1b configs already
   pass locally. The contract mirrors each enabled/disabled condition, threshold, affected body
   list, control frequency, episode length, and motion anchor; unsupported values fail closed.
6. The exact-strike-plus-same-step-termination order is resolved locally: both formal evaluators
   decide the endpoint first, then reject a new same-step contact. Verify this through the Pod unit
   gate; do not promote from local tests alone.

The formal scorecard must additionally pass all of the following before six GPUs are filled:

- Every attempt has one raw row. `eligible = ended and not censored`; physical no-fall, hit, and
  legal return all use that denominator. An evaluator stop or episode timeout is censoring.
- Physical balance fall, tracking guard reset, timeout, and protocol completion are separate
  columns. A legal `hit=1, return=1, physical_fall=1` row is possible when recovery fails after a
  successful shot.
- The selected bank row atomically carries `question_id`, contact position, incoming velocity/spin,
  demanded racket velocity/normal, and intended landing. Both simulators consume the same row.
- Question/order, hold/rest, and action-noise RNG streams are independent. Each noise cell rewinds
  bank order and ledger state; changing noise may not silently change the exam paper.
- The result manifest hashes `params/env.yaml`, checkpoint, ONNX, normalization/std sidecars,
  motions, bank, observation contract, station offset, evaluator/scorer, and git commit.
- Each side reaches at least 50 eligible attempts; hitting the max-step cap first fails the cell.
  Summary counts must be recomputable from raw rows, with `returns <= hits <= eligible`.
- The bank contains the quota plus vector/in-flight prefetch reserve; formal repeats are forbidden.
- The 181-D station offset is taken from ONNX metadata or an explicit manifest/CLI value and printed.
  A legacy zero fallback is allowed only for the already-known zero-offset R10c export.

The first smoke is `M3f` (known-good mechanism check), `G1` (known-bad backhand check), and `M2`
(common baseline), with 10 eligible attempts per side in both protocols and both simulators. Only
after that smoke agrees should the
50-per-side main cells start.

Integration status (2026-07-11): `origin/main` now owns the production schema-v3 bank validation,
content-addressed immutable MuJoCo BankExam schedule, one-question reset, per-attempt seeds and
all-attempt denominator. The earlier local dual-sim prototype used packed `(clip,row)` ids and a
global sequential cursor; it is preserved at
`codex/integrate-local-ablation-20260711@30f4652` but is **not** production-compatible and must not
issue formal percentages. In particular, it tried to feed an exam split through a command that the
production contract correctly restricts to train split, and a large vector reset could exhaust a
small bank atomically.

Two simulator-free specifications from that work remain useful: `termination_contract.py` parses
and freezes the saved per-run termination/timing surface, and `virtual_return_scorer.py` reproduces
Isaac contact + RK4 + ball-centre landing semantics (including the measured 16.196 mm error of the
old bare-plane/Euler approximation). They are library/audit tools until explicit adapters bind them
to the production evaluators. The intended Isaac adapter must be evaluator-owned, consume the same
schema-v3 content IDs/schedule as BankExam, verify `env.pkl` against `env.yaml` before construction,
and fail the whole cell if any question in the fixed prefix is censored. Later completed questions
may not replace it. The pairing claim is about scored questions, not identical hold/reset physics
histories.

Exact read-only `params/env.yaml` inputs used for that check:

| Run directory suffix | SHA-256 |
| --- | --- |
| `2026-07-07_22-48-20_s1w4_M2_v4rg` | `470618c4900f64947652429c37b15ab302245a0933df3b9e5d846fecb8c518cb` |
| `2026-07-09_04-48-39_s1w4_R8b_envpen_v4rg_seed2` | `f9bb86a59d3939686c07f4244d394275fd296bd49b0523d6f0ea67fb47578e0a` |
| `2026-07-09_09-00-48_s1w4_C3_guide_envpen` | `64c194c4edbb96c5ff97d054d2d2069e81250e999a816d76637d36a4d1c77011` |
| `2026-07-10_07-31-02_s1w4_G1b_swingsyn_facerescue` | `800d4e1474c8f26cf56047699cea052066f98ff7aa7150ebf2336e8a0944adb8` |

### P1 — Confirm The S1 Face-Frame Repair Cleanly

From the same M3/M2 starting checkpoints, run matched `+4000` continuation controls:

- old helper, no guidance;
- S1 shared face helper only;
- S1 helper + linear guidance only if S1-only leaves a measurable dead zone.

Keep the question bank, seed, sampling balance, and wall-clock opportunity count fixed. G1b cannot
answer this because it changes helper, reward, and training duration together.

### P2 — Re-rank The Current Shortlist

Only after terminal expanded exams, compare G2, C1, R1b/R5b, and M3f; add C3/C4 after their
saved termination settings pass the Pod smoke. Require balanced
forehand/backhand opportunity quotas; do not accept an all-side average that hides a dead side.
Then run the winner and its direct parent under a second seed.

### P3 — Make Motion And v5 Tests Fair

- Compare M1/v5, M2/v4, and M3/swing on the same external arrival/required-racket bank rather than
  each motion's self-generated bank.
- After the S1 helper is fixed, compare one variable at a time: standard v5 vs v5ftopp/T3, then
  required strike speed around `2.2 m/s` vs the recorded `3.4 m/s`. Normalize by swing opportunity,
  not wall time, because retimed clips have different lengths.
- Do not repeat N1's current `0.15 m` independent-per-step approximation. Preserve its real goal:
  replay measured prediction errors that start large, remain time-correlated, and shrink as more
  ball observations arrive. Do not treat T3 as a candidate until these prerequisites are met.

### P4 — Complete The Acceptance Surface

Before promoting a policy, add the missing continuous-hitting cells, Gate 3B per-serve scoring,
Phase-B physical-contact rich-hit comparison (the current engine check had only two true hits), and
A3 actuator identification. These are deployment gates, not optional polish.

## Candidate Phase-1 Closeout Schedule

This is a dependency order, not current launch authorization. `Run full` means use available
capacity only after the current dependency passes; it never means start six new PPO arms while the
ruler is still changing.

1. **Measurement smoke (known-good/known-bad checks):** after the schema-v3 Isaac adapter exists, use
   `M3f`, `G1`, and `M2` single+continuous smoke cells. In parallel, use pod CPU lanes for the
   matching MuJoCo cells. Require 10 eligible attempts per side, identical manifest/question order,
   and raw-row invariant checks. Do not start the main matrix if any invariant fails.
2. **Pod2 terminal debt in parallel:** all three GPUs are now free. Regenerate exact `model_16999`
   exports/sidecars and terminal quick screens for the eleven completed arms. G1b already finished
   and produced its one old-style terminal screen; re-run it only under the repaired common exam.
   This is artifact recovery, not more training.
3. **Primary deterministic main卷 (`ns=0`, 50/side):** six Isaac lanes cover
   `M2`, `R1b-s1/s2`, `R5b-s1/s2`, `G2`, `C1`, `M3f`, `M2f3`, plus one negative control;
   guard-package arms join after their saved termination settings pass the Pod smoke. Each arm runs fixed-point and continuous
   protocols from its exact saved env manifest.
   MuJoCo cells run concurrently on CPU with `qdes_clamp`, generation-correct hold semantics, and
   the same exam manifest.
4. **Robustness卷 (`ns=0.05`):** only the deterministic shortlist and direct parents advance. This
   avoids spending half the fleet on noisy cells for arms already eliminated at `ns=0`.
5. **Causal cleanup training:** only after the public ruler is stable, run matched old-helper
   continuation and S1-only controls for M3/M2. Replace N1 with measured progressively improving
   target estimates; do not repeat the current N1 approximation or T3. Split R8's two flags
   only if seed2 terminal data retains a signal.
6. **Promotion:** require both sides, both sims, both protocols, two policy seeds, confidence
   intervals, then Gate 3B and physical-contact checks. A high all-side average cannot hide a dead
   forehand or backhand.

The station anchor remains on the critical path: Phase 1 validates stationary anchoring as a
contract/interface; jiayi's Hitter then advances the same interface through stationary hold,
bounded target-box movement, and finally continuous footwork. The failed fixed-point R10c recipe
does not authorize deleting or bypassing that channel.

Contract caveat: R10c's 181-D tail and Hitter's deployed 177-D `base_target_pos_b` station channel
are not yet the same tensor layout. MuJoCo can evaluate 181 and now records its offset, but the
current C++ runner only accepts 175/177/180. Before movement promotion, explicitly map or unify the
181 world-anchor semantics with jiayi's 177 route and pass C++/Isaac/MuJoCo observation parity;
do not assume that sharing the word “station” makes the contracts interchangeable.

## Reproduction Commands

Representative read-only fleet check:

```bash
ssh root@162.43.172.171 -p 18333 -i ~/.ssh/id_ed25519_runpod \
  'date -u; nvidia-smi --query-gpu=index,pstate,utilization.gpu,memory.used,power.draw --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv; ps -eo pid,lstart,etime,args | grep -E "patrol_watchdog|train.py|python.*Isaac"'

ssh root@162.43.172.181 -p 13146 -i ~/.ssh/id_ed25519_runpod \
  'date -u; nvidia-smi --query-gpu=index,pstate,utilization.gpu,memory.used,power.draw --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv; ps -eo pid,lstart,etime,args | grep -E "patrol_watchdog|train.py|python.*Isaac"'
```

Pod1 terminal report examples:

```text
/workspace/franco/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/
  2026-07-09_18-55-42_s1w4_R10c_stationobs_v4rg/judge/judge_report_model_16999_20260710_003029.md
  2026-07-09_19-13-09_s1w4_M3f_swing_faceframe_fix/judge/judge_report_model_20998_20260710_041533.md
  2026-07-09_19-14-32_s1w4_M2f3_v4rg_faceframe_fix/judge/judge_report_model_20998_20260710_043033.md
```

Known limitation: the ignored RunPod logs, checkpoints, and exported policies remain external
runtime artifacts. Reproducing an exact rate on another machine requires manually restoring the
recorded motions, banks, checkpoint, observation normalization, and learned standard deviation.
