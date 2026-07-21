# Motion capture interface

HOPE drives its planner from an external motion-capture system that streams rigid-body
poses into ROS 2. During competition the arena streams the named rigid bodies `Ball`,
`P1`, and `P2` (`Ball` is first in `/poses`; the default VRPN bringup aggregates only it). A
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
| `/poses` | `geometry_msgs/PoseArray` | ~300 Hz | Full tracked pose(s) in the world frame. `Ball` is always first; `P1` and `P2` may follow when the selected adapter is configured to aggregate robot bases. The planner reads `Ball` at `ball_pose_index` and currently consumes only its position. |
| `/tf` (optional) | `tf2_msgs/TFMessage` | ~300 Hz | Named transforms for `world → Ball` (and `world → P1`, `world → P2` where streamed). The OptiTrack relay publishes these; the shipped Chingmu/VRPN path does **not**, so add a `tf2_ros` broadcaster if that deployment needs named transforms. |
| `<robot_base_pose>` | `geometry_msgs/PoseStamped` | ~300 Hz (optional) | Full robot-base pose in the world frame, used for the fixed-station recentring term. Topic name is a launch parameter. |

The planner consumes every incoming mocap sample for its estimator but runs its (more
expensive) trajectory solve at **at most 50 Hz**. The shipped Chingmu/VRPN client and the
OptiTrack/NatNet driver both use ROS receipt time by default. For a VRPN capture timestamp,
set `use_vrpn_timestamps: true` and synchronize server and ROS clocks (NTP/PTP). The NatNet
driver can instead be configured with `topics.header_time: camera` only when the Motive and ROS
clocks are similarly synchronized.

## Bringing up mocap

HOPE ships two source-specific paths that converge at the identical planner interface:

| Venue system | Vendor transport | Raw ROS 2 message | HOPE adapter |
|---|---|---|---|
| **OptiTrack Motive** | **NatNet UDP** (not VRPN) | `/optitrack/poses`, `motion_capture_tracking_interfaces/NamedPoseArray` | `optitrack_mct_relay` → `/poses` |
| **Chingmu CMTracker/MCServer** | **VRPN** | `/vrpn_mocap/<sender>/pose_id_<sensor_id>`, `geometry_msgs/PoseStamped` | `pose_to_posearray` → `/poses` |

### OptiTrack / Motive: NatNet

Use the `optitrack` backend for Motive. Enable NatNet, set **Up Axis = Z**, prefer unicast,
and stream rigid bodies named exactly `Ball`, `P1`, and `P2`. NatNet uses the Motive command
port (normally UDP 1510); the driver obtains the data-port and unicast/multicast details from
the server response. Motive's legacy VRPN stream on port 3883 is **not used** by this backend.

```text
Motive NatNet → motion_capture_tracking_node (namespace /optitrack)
             → /optitrack/poses (NamedPoseArray)
             → optitrack_mct_relay → /poses (PoseArray, Ball at index 0)
```

`NamedPoseArray` carries one header plus entries of the form `{name, Pose}`. The relay maps
the case-sensitive Motive asset names into the HOPE topics, preserves the position and
quaternion, and only publishes `/poses` on a frame that contains `Ball`; it never repeats a
stale ball pose during an occlusion. The raw topic is intentionally namespaced because its
message type differs from the HOPE `/poses` `PoseArray` contract.

Launch it with:

```bash
ros2 launch hope_bringup hope_bringup.launch.py \
  mocap_backend:=optitrack mocap_server:=<MOTIVE_PC_IP>
```

The `Table` asset may be recorded through this relay during a separate setup/calibration
session, but must be disabled or omitted from Motive's competition stream. Consequently no
live `/table/pose`, table TF, or table entry reaches the competition `/poses` stream. See the
full operational guide in [`docs/OPTITRACK.md`](../docs/OPTITRACK.md).

### Chingmu / CMTracker: VRPN

CMTracker/MCServer serves the named rigid bodies as VRPN trackers directly. Configure it to
stream Z-up so no software frame conversion is needed. The vendored ROS 2 client
(`hope_ws/src/vrpn_mocap`, MIT licensed) publishes one `PoseStamped` topic per tracker (with
`multi_sensor: true`), and `hope_bringup/pose_to_posearray` copies the complete pose —
including its quaternion — into `/poses`.

```text
CMTracker/MCServer VRPN → /vrpn_mocap/<sender>/pose_id_<sensor_id> (PoseStamped)
                         → pose_to_posearray → /poses (PoseArray, Ball at index 0)
```

Topic and asset names are case-sensitive. Configure them for the actual name shown by Motive
or CMTracker instead of assuming that `Ball` and `ball` are interchangeable.

For testing without a physical rig, `hope_ws/src/hope_bringup/scripts/fake_ball_publisher`
publishes synthetic `/poses` trajectories (`fake_optitrack_publisher` does the same at the
OptiTrack driver level).

## What is intentionally not here

This is a generic interface description, not a venue setup guide. Rig-specific hardware,
camera counts, network addresses, and calibration recordings are deployment details you
supply for your own environment.

For a worked example of one such environment, see the preserved arena design document —
[HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md](HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md) ([中文](HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup_ZH.md)). It
covers OptiTrack/Motive and Chingmu/CMTracker configuration, camera layout,
tracked-object taxonomy, `base_link` marker placement, and 6-DOF ball tracking.
For the general frame and topic contract, treat this README as authoritative.
