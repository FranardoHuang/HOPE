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
- `hope_ws/src/hope_planner/config/hope_planner.task_revision.yaml` is the schema-4 same-ball
  revision overlay. It must be paired with a checkpoint whose ONNX metadata binds the identical
  [`phase governor`](../DEFINITIONS.md#phase-governor); never enable only one side.

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
exact base-sequence reference and source time. Task-revision schema 4 is exact22 and adds
`task_id/task_revision`; formal base schema 2 is exact12. Side `0` is a revocation, never permission
for the runner to guess.

Confirm live topic wiring before relying on outputs.

Current source semantics (implemented 2026-07-16; runtime gate still open): the Python node
recomputes trajectory, target and `time_to_strike` from each admitted sample. In schema 4, one
physical ball keeps one task id and every valid/invalid refresh increments the revision. A matching
179-D C++ policy consumes the latest complete target/TTS tuple on every policy tick before contact,
while side/clip stay immutable; it never treats a refresh as a second swing. The task closes only on
an explicit ball boundary/contact/deadline transition and consumed task ids cannot replay.

The runner must be started with its task-revision option and must load ONNX metadata whose
`planner_task_revision` document exactly matches the producer/training profile. A schema-4 producer
with an old frozen-target model, or a revision-trained model with schema 3, fails closed. These are
source gates only until the clean Linux Release/full-scene/vendor tests pass.

VRPN still defaults to host receipt time. Capture-stamp experiments must explicitly set
`source_timestamp_mode=vrpn_packet`, prove the VRPN and ROS clocks are synchronized within
`vrpn_source_max_abs_skew_s`, and retain a latency trace. Invalid/skewed packet stamps suppress the
sample instead of silently falling back.

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

The planner now emits a throttled 10 Hz correlated diagnostic including source age, ball distance to
the strike plane, predicted intercept position/velocity, planner TTS, runner-effective TTS, task
state/id/revision and command epoch/sequence. Save it together with runner status/accept reason; the
two streams still need a shared runtime capture before they count as a field trace. Do not infer a
task-lifecycle bug from only one log.

Host-only source checks, without ROS, a simulator or a runner:

```bash
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest -q \
  hope_ws/src/hope_planner/test \
  tests/test_planner_side_contract_source.py
```

Current feature integration result: `215 passed, 2 optional skipped`. This is not yet an accepted
latest-main/runtime result; ROS/Jazzy Release, first tick and vendor behavior remain separate gates.
