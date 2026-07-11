# Run The Phase-1 Cross-Engine Instrument-Parity 2x2

## Current Status

`Preregistered / runtime blocked`. Do not launch this paper yet. The prerequisite contract is
`configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json`; its validator must
continue to report `instrument_parity_gate_closed=false` until all four real cells exist.
The current preregistration SHA is `dd8fb0b9...6818`; the bound validator SHA is
`e688873c...fb5c`.

This operation is simulator-only. It authorizes no real-robot command or deployment test and does
not change, stop, promote, or restart a training arm.

## Why This Paper Exists

The same 100 question IDs produced different rankings because the accepted MuJoCo score used a
physical paddle/ball outcome while the Isaac score used analytic virtual contact and flight. A
same schedule is not enough when the measurement instruments differ. The frozen 2x2 separates
engine execution from instrument behavior:

| Engine | Physical truth | Analytic counterfactual |
| --- | --- | --- |
| Isaac | required; currently blocked | required; instrumentation prepared, not run |
| MuJoCo | required; old physical result exists but needs normalized cell output | required; old `cf_*` exists but needs normalized cell output |

No row or column may substitute for another. In particular, an Isaac analytic return relabeled as
"physical" fails closed.

## Frozen Paper

- target: fresh exact SZ seed1 `model_2000`, checkpoint SHA
  `99e82659...ae4c`, hard-contract SHA `3a3b3d95...b9972`;
- bank SHA `d7db2568...5096`;
- schedule file SHA `66e89986...1cb3`, semantic SHA `7dc6af82...ff3e`, 100 attempts,
  50 per side, one question per reset, no wrap/no censoring;
- question-order SHA `b87e81a3...1f91`;
- analytic capture radius remains `0.095 m` and minimum approach speed remains `0.3 m/s`.

Changing a threshold to reproduce the MuJoCo ranking creates a different paper and is forbidden.

## Isaac Instrumentation Extension

The evaluator keeps the accepted `hope.isaac-bank-exam.v1` score fields intact and adds the
JSON-only `hope.cross-engine-state-instrumentation.v1` extension. It exports:

- the complete numeric ready state: environment-local root state, explicit joint order, joint
  position and joint velocity, plus both the legacy ready-state digest and a content SHA;
- one numeric state snapshot for every question, at exact strike or at termination before exact:
  base root state, racket position/velocity, target state, incoming ball velocity/spin;
- all three face lanes: signed striking-face normal **before** `orient_normal`, raw mount `+Y`,
  and the analytic oriented normal;
- the analytic capture/net/opponent/landing outcome and an explicit physical-truth capability.

The extension does not enter observations, rewards, commands, actions, resets, or scoring. The
CSV schema remains unchanged; nested numeric state lives in the scorecard JSON.

Current Isaac `PhysicalBallManager` is Phase A only: it realizes incoming flight, disables robot
collision and applies no racket impulse. The evaluator therefore emits
`available=false`, capability `incoming_flight_only_no_paddle_contact_phase_a`. This is deliberate
and is why the physical Isaac cell cannot yet exist. Full capability must be exactly
`physical_paddle_contact_and_post_contact_flight_v1` with per-question contact, net, landing and
return booleans.

## Validate The Preregistration

This command is local and starts no simulator:

```bash
python3 scripts/validate_phase1_cross_engine_instrument_parity_2x2.py \
  --config configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json
```

The accepted pre-runtime output is `valid_preregistered_runtime_blocked` and
`instrument_parity_gate_closed=false`. Any other current claim is an error.

## Runtime Evidence Contract

After Isaac Phase B and the matching MuJoCo state export exist, run each cell from one clean,
detached evaluation checkout containing the exact source hashes bound by the preregistration.
Do not mutate a live training checkout. Normalize each output to
`hope.cross-engine-instrument-cell.v1`, preserving the raw source artifact SHA. Every cell must
contain:

1. the frozen checkpoint, bank, schedule and question-order bindings;
2. a numeric ready state and 100 ordered, uncensored numeric question-state snapshots;
3. exact/fresh lineage;
4. for physical cells, actual physical paddle contact and post-contact flight outcomes;
5. for analytic cells, the frozen capture/contact/flight outcomes.

Create a content-addressed `hope.cross-engine-instrument-parity-evidence.v1` manifest that lists
the four cell files relative to one external artifact root. Validate it with:

```bash
python3 scripts/validate_phase1_cross_engine_instrument_parity_2x2.py \
  --config configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json \
  --evidence /path/to/instrument_parity_evidence.json \
  --artifact-root /path/to/instrument_parity_cells
```

The validator verifies every file SHA and every row. It emits
`instrument_parity_gate_closed=true` only after accepting exactly four cells. A missing cell,
duplicate cell, virtual-only physical cell, changed question order, mismatched same-engine ready
state, non-finite state, or incomplete physical outcome raises an error and emits no closed gate.

## Runtime Blockers

1. Implement and validate Isaac Phase-B racket impulse plus post-contact physical ball truth. The
   current incoming-flight probe is insufficient by construction.
2. Run the new Isaac numeric instrumentation on the frozen q50 schedule; no accepted runtime
   scorecard contains it yet.
3. Export/normalize MuJoCo's signed face-normal vector and complete state schema. The old strike
   ledger contains physical and `cf_*` outcomes but predates this state contract.
4. Produce all four immutable cell artifacts and their evidence manifest.

Until all four are resolved, this operation remains a diagnostic prerequisite and cannot close a
cross-engine model-selection, plant, continuity, deployment, or real-robot gate.
