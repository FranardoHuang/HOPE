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

## Inputs

The planner subscribes to `/poses` (`geometry_msgs/PoseArray`) with best-effort, keep-last depth 1. The configured `ball_pose_index` selects the ball pose from the array.

The planner publishes `/racket/command` (`hope_msgs/RacketCommand`) with reliable, keep-last depth 10 because it is a control setpoint.

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
