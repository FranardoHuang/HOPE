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

- `hope_planner` tests pass from the package directory: 20 tests.
- Existing tests cover ball state estimation, trajectory prediction, calibration, quaternion utilities, racket target planning, and CSV splitting.
- Processed calibration CSVs and chunk manifests exist.

Not done:

- Accepted physical constants are not recorded in a single canonical interface/config document.
- Spin and double-bounce behavior are not modeled.
- Outlier rejection and timestamp quality checks are not fully documented.

## Risks

- HITTER assumes negligible spin; this can fail against skilled opponents.
- Bounce parameters fitted on one ball coating may not transfer.
- A model that looks good on curated samples can fail on live noisy mocap.

## Next Steps

1. Record the current fitted parameters and source dataset.
2. Add a small planner regression dataset if needed.
3. Define explicit tests for short balls, deep balls, and double-bounce cases.
