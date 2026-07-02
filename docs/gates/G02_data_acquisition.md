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
- The Avatar-Pro relay can now track the ball either as a named rigid body (`ball_tracking_mode:=rigid_body ball_object:=Ball`) or by motion-based auto-detection (`ball_tracking_mode:=auto`).

Not done:

- General recording protocol is not yet finalized.
- Dataset naming and metadata schema are not yet enforced.
- Large evolving mocap datasets are not yet assigned a local sync path.
- Live arena confirmation is still needed for the real `PPT`/`P1`/`P2` labels and whether the ball is best exposed as a named rigid body or auto-detected marker.

## Calibration-Phase Capture Set (Planned)

For the physics-modeling / calibration phase, the capture set extends beyond ball trajectories: the racket pose, the table's 4 corners, and the net's 2 corners are also mocap-tracked for model fitting (racket tracking is prohibited only during competition/play). The only in-repo artifact today is ball-only: `calib_bags/traj01` contains just `/ball/point`; the corner/racket capture set is planned for the physics-modeling phase.

## Risks

- Data without metadata becomes hard to use for physics calibration.
- Multiple ball coatings or camera settings can silently change dynamics.
- Repeated temporary data can bloat the repo if not routed through asset policy.

## Next Steps

1. Define a metadata template for new recordings.
2. Add a local-only path for large raw recordings if needed.
3. Record one fresh ChingMu dataset after G01 live stream verification, including the chosen `ball_tracking_mode` and `ball_object` in metadata.
