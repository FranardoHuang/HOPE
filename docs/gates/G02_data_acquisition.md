# G02 Data Acquisition

Status: Partial

## Goal

Create repeatable procedures for recording real ball, table, and robot-base data.

This gate is about collecting data that can support calibration, planner validation, and later sim-to-real comparison.

## Inputs

- Working mocap stream from G01.
- Arena frame definitions.
- Recording scripts or ROS bag commands.
- Metadata schema.

## Outputs

- Raw recordings with metadata.
- A small curated sample set committed to git when useful.
- A larger local dataset stored outside normal git when needed.

## Related Directories

- `calib_bags/`
- `calib_csv/`
- `mocap/`
- `hope_ws/src/hope_planner/hope_planner/bag_to_csv.py`
- `docs/operations/run_mocap.md`

## Operation Docs

- [../operations/run_mocap.md](../operations/run_mocap.md)

## Acceptance Criteria

- Every recording has metadata: date, arena, ball type, table pose, robot pose if applicable, mocap settings, operator, and notes.
- Recording commands are documented.
- Raw data and processed data can be traced to each other.
- At least one sample can be replayed or converted by documented commands.

## Current State

Done:

- `calib_bags/traj01/` contains a sample `.mcap` and metadata.
- `calib_csv/` contains processed CSVs, chunks, manifests, and overview plots.
- Planner package includes bag-to-CSV and split utilities.

Not done:

- General recording protocol is not yet finalized.
- Dataset naming and metadata schema are not yet enforced.
- Large evolving mocap datasets are not yet assigned a local sync path.

## Risks

- Data without metadata becomes hard to use for physics calibration.
- Multiple ball coatings or camera settings can silently change dynamics.
- Repeated temporary data can bloat the repo if not routed through asset policy.

## Next Steps

1. Define a metadata template for new recordings.
2. Add a local-only path for large raw recordings if needed.
3. Record one fresh ChingMu dataset after G01 live stream verification.
