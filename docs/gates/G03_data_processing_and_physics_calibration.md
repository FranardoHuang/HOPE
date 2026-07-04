# G03 Data Processing And Physics Calibration

Status: Partial

## Goal

Turn raw real-world ball data into calibrated physical parameters and testable planner inputs.

This gate supports real-to-sim and planner validation.

## Inputs

- Raw bag data from G02.
- Processed CSV trajectories.
- HITTER-compatible ball flight and bounce assumptions.

## Outputs

- Clean segmented trajectories.
- Calibrated drag and restitution parameters.
- Planner tests and regression data.
- Known limitations for spin, bounce, and outlier handling.

## Related Directories

- `calib_bags/`
- `calib_csv/`
- `hope_ws/src/hope_planner`
- `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`

## Operation Docs

- [../operations/build_and_test.md](../operations/build_and_test.md)
- [../operations/run_planner.md](../operations/run_planner.md)

## Acceptance Criteria

- Raw bag to CSV conversion is reproducible.
- Trajectory segmentation is reproducible.
- Calibration procedure records fitted constants and units.
- Planner tests cover state estimation, trajectory prediction, racket target planning, and calibration utilities.

## Current State

Done:

- `hope_planner` tests pass from the repo root with `PYTHONPATH=hope_ws/src/hope_planner`: 26 tests on 2026-06-26.
- Existing tests cover ball state estimation, trajectory prediction, calibration, quaternion utilities, racket target planning, and CSV splitting.
- Processed calibration CSVs and chunk manifests exist.
- Stage 3 racket target planning now uses the same quadratic-drag-plus-gravity free-flight model as Stage 2 for outgoing velocity shooting and net-clearance interpolation, instead of solving outgoing returns ballistically while inbound prediction used drag.
- Pure Python planner tests now cover opponent-facing racket normals, degenerate/sideways normal cases, drag-aware outgoing landing, bounce-then-cross hit-plane prediction, and quaternion local-`+x` alignment.

Done (2026-07-03, venue ball-physics fit v1):

- **Accepted physical constants are now recorded in `configs/ball_physics_venue.yaml`**
  (flight k_d/k_m, table/paddle contact blocks; every value carries provenance, CI, and
  the (speed, SR, v_n) validity envelope). Methodology + F1–F8 falsification verdicts:
  `docs/ball_physics_fit_report.md`; pipeline of record: `hope_training/ball_physics_fit/`.
- Spin is measured (ball quaternion channel, scale validated aerodynamically, ≤15 rev/s
  coverage) and modeled in flight (Magnus) and at contacts (tangential-impulse spin
  equation). The paddle block got its FIRST real-racket fit (150 strikes); paddle
  restitution is velocity-dependent (F4 KILL — consume the yaml `e_exp_*` keys).
- Outlier rejection and QA gates are implemented and documented in the pipeline
  (Stage 0: sampling / units-frame / gravity magnitude+tilt gates; robust losses and
  quality gates throughout; `test_oracle_present.py` fails loudly when data is absent).

Not done:

- Double-bounce behavior is not modeled (landing predictor is first-bounce only).
- Magnus saturation above SR ≈ 0.7 is unvalidated — the venue data never reaches it
  (F2 inconclusive by coverage); needs a dedicated high-spin capture.
- The venue rig's 9 mm noise leaves the table tangential block at the v0 values and
  paddle a_t identified only through the velocity channel (0.52, CI [0.46, 0.61]);
  Stage-4 absolute landing bars not met against observed/terminal-window ground
  truth (through-paddle median ≈ 0.25 m on full coverage n=82, vs the 0.10 m
  target — error budget dominated by paddle model form + racket-state accuracy;
  flight model is fine: 67 mm at contact from measured out-state, 5 mm at ~100 ms
  before landing. See `predict_check.py` H0/H1/H2 decomposition).

## Risks

- HITTER assumes negligible spin; this can fail against skilled opponents.
- Bounce parameters fitted on one ball coating may not transfer.
- A model that looks good on curated samples can fail on live noisy mocap.

## Next Steps

1. Record the current fitted parameters and source dataset.
2. Add a small planner regression dataset if needed.
3. Define explicit tests for short balls, deep balls, spin, and double-bounce cases.
