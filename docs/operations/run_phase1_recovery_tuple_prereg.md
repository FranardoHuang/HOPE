# Validate The Phase-1 Recovery-Tuple A/B/C Preregistration

Status: design validation available; launch deliberately blocked

## Purpose

Use this procedure to audit the command semantics between one completed swing and the next
randomly revealed task. It does not launch Isaac, MuJoCo, training, a Pod process, or a robot.
The frozen structural choices are:

- `A_explicit_bridge`: a content-bound, interruptible safe PD/trajectory bridge;
- `B_canonical_tuple`: the 179-D actor receives a complete canonical recovery tuple;
- `C_previous_tuple`: the actor retains the complete previous question tuple until reveal.

The current deploy hybrid is not a fourth formal arm. It combines a new live-base-anchored
position with the previous strike velocity and normal, a tuple generation absent from the bound
training source.

## Bound Files

- preregistration:
  `configs/phase1_recovery_tuple_abc_prereg_20260712.json`
  (`sha256 39b97915bbb37bcf69e9a8a5eb87cb928bc1e7a6425c1f1555dd61c128b71e1a`)
- validator:
  `scripts/validate_phase1_recovery_tuple_prereg.py`
  (`26034` bytes,
  `sha256 3de8bf3039ba603d7171fb5956d824f7ba1a24f773bbae648604fdf19f9682d0`)
- tests: `tests/test_validate_phase1_recovery_tuple_prereg.py`

The preregistration also verifies the immutable T0/T1 design/schedule files, exact training git
blobs at `d3cdbdfc2f6e30726aa197b7bca66496ee0d39e5`, and the read-only vendor Gate3 policy and
MJCF blobs at `1d46ef2cbb915efc135251f9b32f4ec25d0342ab`.

## Design Check

From the repository root:

```bash
python3 scripts/validate_phase1_recovery_tuple_prereg.py \
  --prereg configs/phase1_recovery_tuple_abc_prereg_20260712.json \
  --expected-prereg-sha256 39b97915bbb37bcf69e9a8a5eb87cb928bc1e7a6425c1f1555dd61c128b71e1a \
  --mode design-check
```

Pass output includes:

```json
{"current_179_B_usable": false, "current_hybrid_formal": false, "formal_arms": ["A_explicit_bridge", "B_canonical_tuple", "C_previous_tuple"], "launch_authorized": false, "status": "pass_design_only", "vendor_gate3_final": true}
```

Run the red-team regression:

```bash
python3 -m pytest -q tests/test_validate_phase1_recovery_tuple_prereg.py
```

The checked-in result is `20 passed`.

## Expected Launch Failure

```bash
python3 scripts/validate_phase1_recovery_tuple_prereg.py \
  --prereg configs/phase1_recovery_tuple_abc_prereg_20260712.json \
  --expected-prereg-sha256 39b97915bbb37bcf69e9a8a5eb87cb928bc1e7a6425c1f1555dd61c128b71e1a \
  --mode launch-check
```

This command must exit `1` with `LAUNCH BLOCKED`. Do not change null fields in this immutable
design file. A future launch requires a new content-addressed manifest binding, at minimum:

- immutable q10/q50 random-arrival and question schedules;
- individual finite/iteration/lineage-bound 179-D checkpoints;
- A bridge source, whole-trajectory safety certificate, and an executed-action versus shadow-policy
  action/history handoff contract;
- B canonical tuple source, ready-set selector, and fresh checkpoints;
- C coherent-tuple source and fresh checkpoints for any learned random-arrival claim;
- a complete numeric ready/base/racket/target contract, not just a named `stand` pose;
- same-family Isaac and Agibot vendor MuJoCo Gate3 continuous no-reset judges;
- racket/handle self-hit instrumentation and a semantics-correct calibrated plant.

## Checkpoint Interpretation

- Existing 179-D atomic-question checkpoints may be used in A only as a frozen swing diagnostic
  after the bridge and handoff are certified.
- They may be used in C only for a zero-shot coherent-tuple diagnostic. Extended final-frame dwell
  and random reveal were not trained, so this is not a learned recovery result.
- They cannot be used in B: the current training distribution contains neither the canonical
  zero-velocity tuple nor its neutral ready normal. B requires fresh training.
- A fair A/B/C causal comparison is fresh exact and paired. No existing checkpoint may be renamed
  `T1-trained` after the fact.

## Ready And Cross-Engine Boundary

Ready is a conjunction of station, uprightness, low velocity, support/slip, actuator margin,
clearance and next-task/deadline reachability. It is never equality to motion frame 0. The bound
static audit already finds different named stands: Isaac pelvis `(0,0,1.0684)` versus vendor MJCF
`(-0.0416378,0.000359049,1.06839)`, with 31-joint L2 difference `0.171845 rad`. Because Stage-1
contact position is environment-origin absolute while the actor sees target minus current racket
FK, the `4.16 cm` root-x difference does not automatically cancel. This is a hypothesis to isolate,
not a proven cause of engine divergence.

Final decisions use the same immutable random-arrival no-reset exam in Isaac first and the Agibot
vendor MuJoCo Gate3 implementation last. q10 is directional screen only; q50 decides. Any engine
disagreement blocks and triggers root-cause work rather than averaging two scores.

## Prohibited Actions

This runbook does not authorize:

- editing or switching a live training checkout;
- starting, stopping or signalling any Pod process;
- changing the Gate3 worktree or C++ runtime;
- running a real-robot command;
- adding recovery rewards before the A/B/C structural result.

If structure later shows that learned shaping is necessary, normalize the three component scales
on frozen rollouts, run a full paired `2^3` presence/absence design, and only then consider a
constant-total-budget mixture. Safety and self-contact remain hard constraints and can never be
compensated by another reward.
