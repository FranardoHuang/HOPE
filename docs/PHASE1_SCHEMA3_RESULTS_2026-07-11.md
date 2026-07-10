# Phase-1 Schema-v3 Same-Paper Results — 2026-07-11

## Status and interpretation

The evaluator-owned schema-v3 paper now runs end to end in Isaac and MuJoCo.
The known-good/common/known-bad canary separated M3f, M2 and G1 as expected,
and M3f/M2 completed the 50-question-per-side clean, 5% action-noise and second
evaluation-seed slices.

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

Full paths live under
`/workspace/codexschema/phase1_schema3_20260711/{M3f,M2,G1}` on the
corresponding Pod. JSON/CSV ledgers were independently reloaded and checked for
complete schedule indices, no censoring, exact order, content ID, bank row,
hold and attempt seed. An early M2 q50 result was discarded and rerun because a
parallel task moved the checkout during that cell; only the stable-checkout
`723322b4...5a01` result above is evidence.

## Current decision and remaining work

M3f is the clear historical diagnostic winner; M2 remains the common baseline;
G1 is a useful known-bad backhand control. Cross-engine absolute rates differ,
especially for M2 forehand, which is expected to remain an open parity issue
until a fresh exact-lineage export binds the real execution/plant contract.

The fixed-point single-question, noise and evaluation-seed slices are complete.
MuJoCo carry-state continuity is diagnostic-only and is being run separately
with a product `return_and_recover_rate`. Isaac continuous play still requires
the physical serve/next-ball timeline identified in `NOW.md`; the independent
one-environment-per-question adapter must not be relabeled continuous.

Reproduction commands and exact/inexact boundaries are in
[operations/run_training.md](operations/run_training.md#shared-schema-v3-bankexam-isaac--mujoco).
