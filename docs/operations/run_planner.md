# Run Planner

Status: Draft

## Task Setup

Use the ROS environment. You need either live mocap from [run_mocap.md](run_mocap.md) or replay data that publishes the planner input topics.

```bash
distrobox enter hope
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install
source install/setup.bash
```

No Agibot deploy payload is required. For calibration-data work, use the curated files in `calib_bags/` and `calib_csv/` or document any local dataset path in the relevant G02/G03 gate doc.

## Launch

After building and sourcing `hope_ws`:

```bash
ros2 launch hope_planner hope_planner.launch.py
```

Current config:

- `hope_ws/src/hope_planner/config/hope_planner.yaml`

Runtime parameters that must travel with a formal 179 planner/policy pairing:

- `max_predict_time`: forward horizon. The arena default is `2.0 s`; the simulator profile binds
  `2.6 s`. An early prediction does not authorize an early swing: the runner waits for the selected
  clip's metadata windup window and rechecks the tuple each policy tick.
- `solve_period_s`: expensive solve cadence. `0.0` preserves every-measurement solving; the Gate3
  simulator profile binds `0.033 s` while all 300 Hz samples still enter the estimator.
- `base_pose_max_age_s`: formal base receive-age limit, `0.2 s` in both profiles. Expiry or invalid
  recovery revokes both flat rows; READY stdout remains diagnostic-only.
- `swing_side_split_y` / `swing_side_hysteresis_y`: choose side in corrected base-yaw coordinates.
  The C++ runner additionally binds `--planner-side-split-y 0.0` and
  `--planner-side-hysteresis-y 0.04` for its own policy-frame consistency check.
- `marker_to_base_xyz`: marker-local offset, rotated by the normalized marker quaternion before
  addition. Missing or non-finite orientation cannot define a formal side.
- `use_shadow_solver`: both formal profiles explicitly set `false`. Enabling it changes timing and
  dependency closure and needs a new preregistration.

## Inputs

The planner subscribes to `/poses` (`geometry_msgs/PoseArray`) with best-effort, keep-last depth 1. The configured `ball_pose_index` selects the ball pose from the array.

The planner publishes `/racket/command` (`hope_msgs/RacketCommand`) with reliable, keep-last depth 10 because it is a control setpoint.

For the native C++ runner it also publishes reliable `/racket/command_flat` and
`/a3/base_pose_flat`. Formal racket schema 3 is exact20 and carries shared epoch, racket sequence,
exact base-sequence reference and source time; formal base schema 2 is exact12. Side `0` is a
revocation, never permission for the runner to guess.

Confirm live topic wiring before relying on outputs.

Relevant source:

- `hope_ws/src/hope_planner/hope_planner/node.py`
- `hope_ws/src/hope_msgs/msg/RacketCommand.msg`

## Verification

1. Start mocap or replay data.
2. Launch planner.
3. Echo planner output:

```bash
ros2 topic echo /racket/command
```

4. Record latency and prediction sanity checks in G03.

Host-only source checks, without ROS, a simulator or a runner:

```bash
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest -q \
  hope_ws/src/hope_planner/test \
  tests/test_planner_side_contract_source.py
```

Accepted latest-main integration result: `180 passed, 2 optional skipped`. Runtime verification
still requires the separately owned first-tick path.
