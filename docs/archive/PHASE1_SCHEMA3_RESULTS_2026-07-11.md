# Phase-1 Schema-v3 Same-Paper Results — 2026-07-11

## Status and interpretation

The evaluator-owned schema-v3 paper now runs end to end in Isaac and MuJoCo.
The known-good/common/known-bad canary separated M3f, M2 and G1 as expected,
and M3f/M2 completed the 50-question-per-side clean, 5% action-noise and second
evaluation-seed slices.  The subsequent historical main matrix also completed:
R1b (two policy seeds), R5b (two policy seeds), C1, G2 and M2f3 all ran the
same ten-per-side clean ruler; the non-zero-side survivors R1b and C1 then ran
50 questions per side.  C1 additionally completed noise, second schedule seed
and MuJoCo carry-state continuity.

Every model in this report is historical. None has a schema-v3
`training_contract.json` or exact checkpoint/train-bank/ONNX binding, so every
cell correctly records `evaluation_contract_exact=false`. These results accept
the ruler and rank diagnostic candidates; they are not bookable formal model
scores and do not create an accepted quality baseline.

All percentages below use every scheduled attempt. Physical falls, tracking
guard resets and pre-strike failures remain in the denominator. `Physical
fall` means the absolute tilt/root-height fuse; `guard` means a tracking guard,
and is reported separately.

## Immutable inputs

| Candidate | Checkpoint SHA | Exam bank SHA | Bank kept (FH/BH) |
|---|---|---|---:|
| M3f | `f25f274f...3d61` | `750f1df4...cb0e` | 188 / 200 |
| M2 | `182064d9...a017` | `10917148...e5bb2` | 183 / 188 |
| G1 | `dde4d37a...53c5` | `96c98296...b5c8c` | 188 / 199 |

All three banks are schema 3, split `exam`, have unique atomic content IDs,
and passed Torch closed-loop landing/net verification on every kept row. M3f
uses the declared `train-candidate` strike anchor; M2/G1 use `annotated`.

## Ten-per-side canary

| Candidate | Isaac return FH / BH / all | Isaac physical / guard | MuJoCo return FH / BH / all | MuJoCo composite FH / BH / all | MuJoCo physical / guard | Decision |
|---|---:|---:|---:|---:|---:|---|
| M3f | 10/10 / 10/10 / 20/20 | 0 / 0 | 10/10 / 7/10 / 17/20 | 10/10 / 5/10 / 15/20 | 0 / 2 | advance |
| M2 | 6/10 / 10/10 / 16/20 | 0 / 0 | 2/10 / 8/10 / 10/20 | 1/10 / 6/10 / 7/20 | 0 / 0 | advance |
| G1 | 10/10 / 0/10 / 10/20 | 0 / 0 | 9/10 / 0/10 / 9/20 | 7/10 / 0/10 / 7/20 | 0 / 0 | stop at canary |

The bad G1 backhand is reproduced in both simulators: Isaac returned 0/10 and
MuJoCo returned/composited 0/10. G1 therefore did not consume a 50-per-side
slice.

Canary schedule SHAs are M3f
`8468e205037d125d026aa36a789664df999cee016b8a2f85fd7b7049379ee1a6`,
M2 `29c590ae9c6171255bc8e317bd3338c661759e9d9114502c43f6dc83c4c4d99c`
and G1 `1ea88d83f50bd537507c461372de8846491089b6c26f4243c091f9797509332c`.
For each candidate, both simulators emitted the same bank SHA, semantic
schedule SHA and ordered question IDs.

## Fifty-per-side survivor slices

### Clean, schedule seed 0

| Candidate | Isaac return FH / BH / all | Isaac physical / guard | MuJoCo return FH / BH / all | MuJoCo composite FH / BH / all | MuJoCo physical / guard |
|---|---:|---:|---:|---:|---:|
| M3f | 49/50 / 50/50 / 99/100 | 0 / 1 | 50/50 / 41/50 / 91/100 | 50/50 / 33/50 / 83/100 | 0 / 8 |
| M2 | 36/50 / 50/50 / 86/100 | 0 / 0 | 7/50 / 44/50 / 51/100 | 1/50 / 30/50 / 31/100 | 0 / 1 |

M3f's sole Isaac failure was a forehand row with `hold_steps=0`; it guard-reset
before exact strike and remained in the denominator. All MuJoCo `fell` rows in
these cells were `ee_body_pos` tracking guards, not physical falls.

### Robustness slices

| Candidate | Slice | Isaac return FH / BH / all | Isaac physical / guard | MuJoCo return FH / BH / all | MuJoCo composite all | MuJoCo physical / guard |
|---|---|---:|---:|---:|---:|---:|
| M3f | same paper, action noise 0.05 | 49/50 / 50/50 / 99/100 | 0 / 1 | 50/50 / 39/50 / 89/100 | 84/100 | 0 / 9 |
| M3f | clean, schedule seed 1 | 50/50 / 50/50 / 100/100 | 0 / 0 | 50/50 / 37/50 / 87/100 | 84/100 | 0 / 12 |
| M2 | same paper, action noise 0.05 | 35/50 / 50/50 / 85/100 | 0 / 0 | 8/50 / 40/50 / 48/100 | 28/100 | 0 / 2 |
| M2 | clean, schedule seed 1 | 41/50 / 50/50 / 91/100 | 0 / 0 | 16/50 / 38/50 / 54/100 | 21/100 | 0 / 1 |

The second-seed semantic schedule SHAs are M3f
`59a09efd5bdbb796673888553ea26fc58abd7d2f477b781aedf339d2c7735a96`
and M2 `3ccce2c6392f9d786f3a3dc109bb735d6b8de4e29dfcfe01d34db9b11a630feb`.

## Historical main-matrix rerank

These cells use the same all-attempt ruler but remain inexact historical
diagnostics.  R1b, R5b, C1 and M2f3 share the M2 v4rg exam bank and schedules.
G2 uses its own v4rgsyn family bank, SHA
`0840afa68741664dd80f0480f878dbbe7bac172a048894d43e3fe1fd35c75423`,
with 181/190 forehand and 184/206 backhand rows kept and verified.  Its q10
schedule SHA is
`f977e1e2ba47e541a473f71eb45dce411150b529865649563fb50b57176e19bf`.

### Clean ten-per-side screen

The operational screen was side-preserving: a candidate with zero legal
returns on either side in either simulator stopped at q10.  This is the same
rule that stopped G1 above; an all-side average was not allowed to hide a dead
side.

| Candidate | Isaac return FH / BH | MuJoCo return FH / BH | MuJoCo composite FH / BH | MuJoCo physical / guard | Decision |
|---|---:|---:|---:|---:|---|
| R1b seed 1 | 10/10 / 10/10 | 1/10 / 1/10 | 0/10 / 0/10 | 0 / 0 | advance to q50 |
| R1b seed 2 | 7/10 / 10/10 | 1/10 / 4/10 | 0/10 / 1/10 | 0 / 1 | advance to q50 |
| R5b seed 1 | 6/10 / 10/10 | 0/10 / 6/10 | 0/10 / 6/10 | 0 / 12 | stop: MuJoCo FH zero |
| R5b seed 2 | 8/10 / 10/10 | 0/10 / 10/10 | 0/10 / 6/10 | 0 / 2 | stop: MuJoCo FH zero |
| C1 | 10/10 / 10/10 | 7/10 / 2/10 | 2/10 / 2/10 | 0 / 1 | advance to q50 |
| G2 | 7/10 / 10/10 | 0/10 / 6/10 | 0/10 / 3/10 | 0 / 0 | stop: MuJoCo FH zero |
| M2f3 | 7/10 / 10/10 | 0/10 / 10/10 | 0/10 / 5/10 | 0 / 10 | stop: MuJoCo FH zero |

Every Isaac cell reached exact and hit on all 20 attempts.  R5b seed 1 reached
exact on only 8/10 MuJoCo forehands because two attempts guard-reset before
strike; every other MuJoCo q10 cell reached exact on all attempts.  All
reported falls in this table are `ee_body_pos` tracking guards, not absolute
physical falls.

R8b/C2/C3/C4 did not join this comparison.  Their saved runs disable the
anchor/EE tracking guards used by the matrix, so substituting the M2
termination manifest would no longer reproduce their policy runtime while
using their easier saved termination would make the cells incomparable.  They
remain excluded pending a preregistered matched-termination retrain; no result
was silently filled in.

### Clean fifty-per-side confirmation

| Candidate | Isaac return FH / BH / all | Isaac physical / guard | MuJoCo return FH / BH / all | MuJoCo composite FH / BH / all | MuJoCo physical / guard | Decision |
|---|---:|---:|---:|---:|---:|---|
| R1b seed 1 | 45/50 / 50/50 / 95/100 | 0 / 0 | 3/50 / 12/50 / 15/100 | 0/50 / 0/50 / 0/100 | 0 / 2 | stop: transfer collapse |
| R1b seed 2 | 40/50 / 50/50 / 90/100 | 0 / 0 | 3/50 / 14/50 / 17/100 | 0/50 / 4/50 / 4/100 | 0 / 11 | stop: transfer collapse |
| C1 | 46/50 / 50/50 / 96/100 | 0 / 0 | 40/50 / 10/50 / 50/100 | 16/50 / 10/50 / 26/100 | 0 / 9 | robustness slices |

The larger paper shows that the two R1b q10 cells were not transferable
survivors: both have only 3/50 MuJoCo forehand returns despite 40--45/50 in
Isaac.  They stop after clean q50.  C1 preserves non-zero performance on both
sides in both simulators and therefore receives the remaining diagnostic
slices, but its MuJoCo backhand is only 10/50 and it does not displace M3f.

### C1 robustness and continuity

| Slice | Isaac return FH / BH / all | MuJoCo return FH / BH / all | MuJoCo composite all | MuJoCo physical / guard |
|---|---:|---:|---:|---:|
| same paper, action noise 0.05 | 46/50 / 50/50 / 96/100 | 37/50 / 11/50 / 48/100 | 22/100 | 0 / 6 |
| clean, schedule seed 1 | 48/50 / 50/50 / 98/100 | 42/50 / 13/50 / 55/100 | 34/100 | 0 / 12 |

C1's seed-1 schedule SHA is the same M2-family
`3ccce2c6392f9d786f3a3dc109bb735d6b8de4e29dfcfe01d34db9b11a630feb`.
All four single-question C1 ledgers contain exactly 100 uncensored rows and
match the supplied order, bank row, content ID, hold and attempt seed.

The C1 MuJoCo carry-state cell reached exact on 44/50 forehands and 48/50
backhands, returned 23/50 and 19/50, and composited 11/50 and 17/50.  There
were no physical falls; tracking guards were 23/0 and episode timeouts 9/8.
Among the 99 rows with a scheduled next opportunity, 42 returned, 59
recovered, and 26 did both: `26/99 = 26.26%`, or `26/42 = 61.90%`
conditional on return.  This is materially below M3f's `70/99` product.

## Result artifact identities

| Cell | Isaac JSON SHA | MuJoCo summary JSON SHA |
|---|---|---|
| M3f q10 clean | `e1e1b5b6...3091` | `efca6937...4fef` |
| M3f q50 clean | `be13c36b...10a5` | `45280ca6...ce20` |
| M3f q50 noise 0.05 | `ab733faf...f3fa` | `7134c2e8...db1b` |
| M3f q50 seed 1 | `ff454ae0...f395` | `6f15b0fd...6366` |
| M2 q10 clean | `e625a09c...87fc` | `4d7393c5...4c88` |
| M2 q50 clean | `723322b4...5a01` | `c119c7de...d264` |
| M2 q50 noise 0.05 | `04770a49...7f11` | `6446fb6c...dc0d` |
| M2 q50 seed 1 | `3bd40d09...bc6f` | `4e76f03c...3b54` |
| G1 q10 clean | `88587e35...f20e` | `e4243d93...93ef` |
| R1b seed 1 q10 clean | `d719577d...1a6a` | `10befacb...7a32` |
| R1b seed 2 q10 clean | `59bd6030...d6e7` | `7e75c3f...8e9e` |
| R5b seed 1 q10 clean | `28b4c9cd...4793` | `9b9c5b9a...be3a` |
| R5b seed 2 q10 clean | `d05f639d...7504` | `4c1dbbbf...93702` |
| C1 q10 clean | `bca7abc1...bf03` | `21dc3214...04a` |
| G2 q10 clean | `3595f684...b264` | `29940648...094f` |
| M2f3 q10 clean | `3d27d05f...af04` | `6bcba7eb...eca4` |
| R1b seed 1 q50 clean | `cfad6b12...43ad` | `787de3f2...333c` |
| R1b seed 2 q50 clean | `9f2c26a1...f0f0` | `cb9975ac...af14` |
| C1 q50 clean | `99fb2fba...1887` | `b2f9a1cc...871f` |
| C1 q50 noise 0.05 | `d810424f...79dd` | `15688b5c...fb4` |
| C1 q50 seed 1 | `5671847d...959b` | `fa05c3af...47a3` |
| C1 continuity q50 | not applicable | `59c130e6...20df` |

Full paths live under
`/workspace/codexschema/phase1_schema3_20260711/{M3f,M2,G1,shortlist}` on the
corresponding Pod. JSON/CSV ledgers were independently reloaded and checked for
complete schedule indices, no censoring, exact order, content ID, bank row,
hold and attempt seed. An early M2 q50 result was discarded and rerun because a
parallel task moved the checkout during that cell; only the stable-checkout
`723322b4...5a01` result above is evidence.

## MuJoCo carry-state continuity diagnostic

This is not the one-question ruler: `--exam-continuity-diagnostic` starts from
the common stand state once, then carries physical state and last action across
the same 100-row seed-0 paper. Episode timeout/guard recovery may reset an
episode, but the failed opportunity stays in the paper denominator. The cell
is always `evaluation_contract_exact=false`.

| Candidate | Exact reach | Return FH / BH / all | Composite / attempt | Physical / guard | Episode timeouts | Recovered to next | Returned and recovered to next |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3f | 82/100 | 46/50 / 36/50 / 82/100 | 76/100 | 0 / 9 | 20 | 70/99 | 70/99 = 70.71% |
| M2 | 86/100 | 6/50 / 34/50 / 40/100 | 25/100 | 0 / 3 | 30 | 66/99 | 30/99 = 30.30% |

M3f recovered after 70 of the 81 returns that had a scheduled next row
(`86.42%` conditional); M2 recovered after 30 of 39 (`76.92%`). The terminal
paper row is intentionally absent from the 99-row product denominator. M3f
summary SHA is `091bd04564970353f7fe2d38f7020b7b4f16d82cecbcbca1174db5e70f30e6ea`;
M2 summary SHA is
`5658b7cc1288bcbad456f61d3863e6a7eaa546a51ab0a711bf80bc7bf3db8774`.

## Current decision and remaining work

M3f remains the clear historical diagnostic winner; C1 is the only additional
main-matrix candidate that survives q50 on both sides and both simulators, but
its MuJoCo clean/continuity result is far below M3f.  M2 remains the common
baseline; G1 is a useful known-bad backhand control.  R1b's two independent
policies reproduce an Isaac-to-MuJoCo forehand collapse, while R5b, G2 and
M2f3 reproduce a zero-MuJoCo-forehand stop at q10.  Cross-engine absolute rates
therefore remain an open parity issue until a fresh exact-lineage export binds
the real execution/plant contract.

That fresh lane is now live, not yet scored. The 179-D zero-friction/schema-2
motion/schema-v3 bank construction gate passed, and two from-scratch seeds plus
four matched face-pairing continuations reached their first PPO iteration at
fixed checkout `6d93bcb`. This does not alter any historical table above.
Launch contracts, paths and status are recorded in
`PHASE1_FRESH_LINEAGE_2026-07-11.md`; terminal exports and same-paper results
must be added before any fresh candidate is promoted.

The fixed-point single-question, noise, evaluation-seed and MuJoCo carry-state
diagnostic slices are complete. Isaac continuous play still requires
the physical serve/next-ball timeline identified in `NOW.md`; the independent
one-environment-per-question adapter must not be relabeled continuous.

Reproduction commands and exact/inexact boundaries are in
[operations/run_training.md](../operations/run_training.md#shared-schema-v3-bankexam-isaac--mujoco.
