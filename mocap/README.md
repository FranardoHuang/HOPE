# Motion capture interface

HOPE drives its planner from an external motion-capture system that streams rigid-body
poses into ROS 2. During competition the arena streams the named rigid bodies `Ball`,
`P1`, and `P2` (the shipped bringup aggregates only `Ball` into `/poses` by default). A
`Table` asset is used for setup/calibration only and appears only in training-data
recordings — it is not streamed during competition. This document defines the generic
frame and topic contract
the rest of the stack expects. It is deliberately vendor-neutral — any optical
motion-capture rig that can publish the topics below will work. Configure your own rig's
network address in the launch files (see `hope_ws/`).

## Coordinate frame

A single right-handed world frame is shared by mocap, planner, training, and the ball
physics model:

| Axis | Direction | Range over the table |
|------|-----------|----------------------|
| +x   | forward (toward the opponent half of the table) | `[0, length]` |
| +y   | left      | `[-width, 0]` |
| +z   | up        | `0` **is the table surface** |

The **origin is the near-side left corner of the table _surface_**, from the robot's
(P1's) perspective. Because `z = 0` is the playing surface, the floor sits at
`z = -0.76 m`.

Units are SI: metres and seconds. Timestamps are on a single shared clock.

These dimensions and landmarks are not duplicated by hand anywhere: the single source
of truth is
`hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py`,
which derives everything from [`configs/ball_physics.yaml`](../configs/ball_physics.yaml)
so the simulator, planner, and evaluator share one world.

The competition stream contains vendor-defined **6-DOF rigid bodies** (`Ball`, plus `P1`/`P2`
where used). Conceptually, the ball
pose may be inspected as `(x, y, z, pitch, yaw, roll)`, but the ROS 2 wire contract uses
`geometry_msgs/Pose`: position `(x, y, z)` plus quaternion orientation
`(qx, qy, qz, qw)`. Euler angles are derived using an explicitly documented axis and rotation
order; never write pitch/yaw/roll values directly into `Pose.orientation`.

The current no-spin planner consumes only the ball position, so preserving orientation does
not change its input behavior. The orientation remains available for validation and future
spin-aware estimation. The robot's control-facing base orientation (yaw) is taken from the
robot IMU, not from mocap — this is why the policy observation includes an IMU-derived
`base_forward_xy` term (see [POLICY_INTERFACE.md](../docs/POLICY_INTERFACE.md)). Treat a
mocap base-yaw estimate as advisory unless a robot integration contract says otherwise.

## Topics

| Topic | Type | Rate (typical) | Meaning |
|-------|------|----------------|---------|
| `/poses` | `geometry_msgs/PoseArray` | ~300 Hz | Full tracked pose(s) in the world frame. `Ball` first (the shipped bringup aggregates only `Ball`, a single-pose array); add `P1`, `P2` after it when aggregating robot bases. The planner reads `Ball` at `ball_pose_index` and currently consumes only its position. |
| `/tf` (optional) | `tf2_msgs/TFMessage` | ~300 Hz | Named transforms for `world → Ball` (and `world → P1`, `world → P2` where streamed). The shipped VRPN path does **not** publish `/tf`; add a `tf2_ros` broadcaster if your deployment needs named transforms. |
| `<robot_base_pose>` | `geometry_msgs/PoseStamped` | ~300 Hz (optional) | Full robot-base pose in the world frame, used for the fixed-station recentring term. Topic name is a launch parameter. |

The planner consumes every incoming mocap sample for its estimator but runs its (more
expensive) trajectory solve at **at most 50 Hz**. Source timestamps are propagated where the
source provides them; the vendored VRPN path (both vendors) uses receipt time by default and
requires `use_vrpn_timestamps: true` plus synchronized clocks (NTP/PTP) for server capture
timestamps.

## Bringing up mocap

Both vendors use the **same uniform path**: the vendor application solves the named
rigid bodies and serves them as **VRPN** trackers; the vendored ROS 2 client
(`hope_ws/src/vrpn_mocap`, MIT licensed) publishes
`/vrpn_mocap/<sender>/pose_id_<sensor_id>` as `geometry_msgs/PoseStamped` (with
`multi_sensor: true`); and the shipped `hope_bringup/pose_to_posearray` adapter copies the
complete pose — including its quaternion — into `/poses`. Use the vendor-native rigid-body
stream; do not reconstruct the ball from an unlabeled point cloud on the ROS host.

- **OptiTrack:** enable Motive's **VRPN Streaming Engine** (default port 3883; rigid bodies
  only — markers/skeletons are not carried over VRPN). Two caveats: Motive's *Up Axis*
  setting is not clearly documented as applying to the VRPN stream, whose frame can differ
  by Motive version and installation (Y-up output is commonly observed) — verify the streamed
  frame at surveyed table landmarks first, and only if it is Y-up apply a full-pose
  (position + quaternion) Y-up → Z-up conversion on the ROS 2 side before `/poses`
  (required engineering, not included in this repository; the preserved arena design
  document, Section 6.2, specifies the transform and the validation);
  and disable *Zero When Untracked* so occlusions surface as dropouts rather than
  all-zero poses.
- **Chingmu:** CMTracker/MCServer serves the named rigid bodies as VRPN trackers directly;
  configure it to stream Z-up so no software frame conversion is needed.

Topic and sender names are case-sensitive. Configure them for the actual asset name shown
by Motive or CMTracker instead of assuming that `Ball` and `ball` are interchangeable.
A generic bringup that wires the VRPN path into the planner lives in
`hope_ws/src/hope_bringup/`.

For testing without a physical rig, `hope_ws/src/hope_bringup/scripts/fake_ball_publisher`
publishes synthetic `/poses` trajectories.

## What is intentionally not here

This is a generic interface description, not a venue setup guide. Rig-specific hardware,
camera counts, network addresses, and calibration recordings are deployment details you
supply for your own environment.

For a worked example of one such environment, see the preserved arena design document —
[HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md](HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md) ([中文](HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup_ZH.md)). It
covers OptiTrack/Motive and Chingmu/CMTracker configuration, camera layout,
tracked-object taxonomy, `base_link` marker placement, and 6-DOF ball tracking.
It predates this stack, so treat the contract above as authoritative where the two differ.
