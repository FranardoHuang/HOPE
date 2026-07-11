# Phase-1 Cross-Engine Saturation Forensic (2026-07-11)

## Outcome

The two same-question papers are byte-aligned on `question_id` and order, but they are **not the
same outcome instrument**. The present Isaac score is an analytic virtual-ball result with
`physical_ball=false`; the MuJoCo decision score is physical paddle contact plus physical flight.
The split must not be repaired by changing a threshold.

The content-addressed full report is
`configs/phase1_cross_engine_saturation_forensic_result_20260711.json` (SHA-256
`aff8f4e6...45c9`). Its frozen input ledger is
`configs/phase1_cross_engine_saturation_forensic_inputs_20260711.json` (SHA-256
`e074a7c4...747e`). The analyzer is
`scripts/analyze_phase1_cross_engine_forensics.py` (SHA-256 `077e3e48...6bf5`). It is read-only:
it verifies every source SHA and joins raw ledgers; it does not run a simulator, rescore an
attempt, or touch a training process.

| Paper | MuJoCo physical returns | Isaac virtual returns | Localized reason |
| --- | ---: | ---: | --- |
| fresh SZ model 2000 | 83/100 | 99/100 | 13 MuJoCo no-contacts and 4 contact/net failures become virtual returns in Isaac |
| fresh SZ model 4000 | 50/100 | 99/100 | all 49 aligned discrepant FH questions are MuJoCo no-contact; Isaac records a virtual return |
| causal M3 old | 42/100 | 99/100 | 7 no-strike, 4 no-contact, and 47 physical contact/net failures become virtual returns in Isaac |
| causal M3 S1 | 100/100 | 99/100 | one Isaac guard reset; otherwise the instruments agree |

## What Is Actually Separating The Fresh Checkpoints

Fresh SZ does not fail because of question ordering: all 100 IDs, attempts and side assignments
match, and neither ledger is censored. It also does not reduce to ball-flight parameters. The
policies arrive at materially different strike states in the two engines:

- model 4000 FH mean racket-position error is `0.13153 m` in MuJoCo versus `0.02478 m` in Isaac;
  the complete MuJoCo FH range is `0.1002..0.1519 m`, outside the frozen `0.095 m` capture radius,
  so it makes `0/50` FH contacts;
- model 2000 FH mean error is `0.09032 m` in MuJoCo versus `0.03029 m` in Isaac; MuJoCo retains
  `37/50` FH contacts and `33/50` returns;
- Isaac places both checkpoints comfortably inside its virtual capture gate and returns `49/50`
  FH for each. It therefore has no ranking resolution on this paper.

This localizes the first-order fresh failure to **engine-specific policy execution at the capture
margin**, followed by an Isaac virtual-score ceiling. Candidate causal axes are ready-state
mapping and articulation/actuator/contact dynamics. The current artifacts do not distinguish
those candidates: MuJoCo starts from `mjcf_named_keyframe:stand:v1` / `3cbfe700...7fc2`, while
Isaac starts from `isaac-default-stand` / `3120e130...177e`, and the Isaac scorecard exports no
numeric base state for cross-engine subtraction. The hash schemas are engine-specific, so unequal
hashes prove that a shared ready-state contract is absent, not by themselves that initialization
is the whole cause.

Fresh SZ is already zero-friction protocol exact in the current contract. That prevents the known
non-zero friction-unit mismatch from being used as a sufficient explanation here; it does not
prove that every other plant parameter is cross-engine equivalent.

## Why M3 Saturates In Isaac

M3 is a second, more direct mechanism. Isaac's accepted evaluator defines `hit` as
`vb_fired`; `returned` is `vb_fired & vb_net_clear & vb_on_opponent`. The virtual capture path
orients the face normal toward the incoming ball before analytic contact. That makes opposite
signed face normals equivalent for this scorer.

The raw M3-old BH metric exposes exactly what the scorer hides: Isaac records mean normal error
`168.15 deg` for old versus `3.66 deg` for S1, yet both score `50/50` BH virtual returns. In
MuJoCo, M3-old has 89 counterfactual analytic legal landings but only 42 physical returns; 47 of
the aligned Isaac-success/MuJoCo-failure questions physically contact and then fail the net.
M3-S1 physically returns all 100. Therefore the current Isaac metric is structurally unable to
reproduce the signed-face pairing ranking; this is not evidence that the two checkpoints are
physically equivalent.

M3 remains causal/inexact. The forensic also corrects a ledger-reading ambiguity: M3-old has one
attempt classified `physical_fall` and eight classified `guard_reset`; the run-level `fell=9` is
their termination union. It must not be restated as nine physical falls.

## Next Diagnostic, Without Moving A Threshold

1. Keep the two cross-engine gates open and retain MuJoCo's physical result for the current
   within-pair checkpoint decision.
2. Add a same-instrument 2x2 diagnostic: physical-ball truth in both engines and the existing
   analytic counterfactual in both engines. The already-recorded MuJoCo `cf_*` channel shows why
   this matters: it separates trajectory/capture failure from physical-contact/flight failure.
3. Export the complete numeric Isaac ready state plus per-question base pose, racket position,
   velocity, face normal, and physical-ball outcome. Bind the same fields and conventions in
   MuJoCo before attributing the fresh gap to ready state or actuator dynamics.
4. Run a fixed-state signed-normal probe through both physical contacts and both analytic scorers.
   It should test convention parity, not tune success thresholds to recreate a desired ranking.

## Reproduction

Restore the five bound files for each arm beneath one ignored artifact root using the relative
layout in the input ledger. The accepted source bytes live on Pod1 under the two MuJoCo judge
`exam/` directories and the two Isaac q50 result directories named by
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json` and
`configs/phase1_M3_terminal_q50_isaac_result_20260711.json`. Then run:

```bash
python3 scripts/analyze_phase1_cross_engine_forensics.py \
  --config configs/phase1_cross_engine_saturation_forensic_inputs_20260711.json \
  --artifact-root /path/to/phase1_cross_engine_forensic_inputs_v1 \
  --output /tmp/phase1_cross_engine_saturation_forensic_result_20260711.json
shasum -a 256 /tmp/phase1_cross_engine_saturation_forensic_result_20260711.json
```

The expected report SHA is `aff8f4e665d20bb76a56e079735f32b6766388ee05f61c51e93adeb568be45c9`.
