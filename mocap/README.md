# Motion capture interface

HOPE PingPong drives its planner from an external motion-capture system that streams
**ball** (and optionally **robot base**) positions into ROS 2. This document defines the
generic frame and topic contract the rest of the stack expects. It is deliberately
vendor-neutral — any optical/VRPN motion-capture rig that can publish the topics below will
work. Configure your own rig's network address in the launch files (see `hope_ws/`).

## Coordinate frame

A single right-handed world frame is shared by mocap, planner, training, and the ball
physics model:

| Axis | Direction |
|------|-----------|
| +x   | forward (toward the opponent half of the table) |
| +y   | left      |
| +z   | up        |

Units are SI: metres and seconds. Timestamps are on a single shared clock.

The motion-capture system provides **positions only**. The robot's base orientation (yaw)
is taken from the robot IMU, not from mocap — this is why the policy observation includes an
IMU-derived `base_forward_xy` term (see [POLICY_INTERFACE.md](../docs/POLICY_INTERFACE.md)).
If your rig also produces a base yaw estimate, treat it as advisory only.

## Topics

| Topic | Type | Rate (typical) | Meaning |
|-------|------|----------------|---------|
| `/poses` | `geometry_msgs/PoseArray` | ~300 Hz | Tracked ball position(s) in the world frame. Only `position` is used; orientation is ignored. |
| `<robot_base_pose>` | `geometry_msgs/PoseStamped` | ~300 Hz (optional) | Robot base position in the world frame, used for the fixed-station recentring term. Topic name is a launch parameter. |

The planner consumes every incoming mocap sample for its estimator but runs its (more
expensive) trajectory solve at **at most 50 Hz**. Source timestamps are propagated so the
planner can extrapolate for capture latency.

## Bringing up mocap

The repository vendors a third-party VRPN ROS 2 client (`hope_ws/src/vrpn_mocap`, MIT
licensed) that bridges a VRPN motion-capture server to ROS 2. Point it at your server and
map your tracked rigid bodies to the topics above. A generic bringup that wires mocap into
the planner lives in `hope_ws/src/hope_bringup/`.

For testing without a physical rig, `hope_ws/src/hope_bringup/scripts/fake_ball_publisher`
publishes synthetic `/poses` trajectories.

## What is intentionally not here

This is a generic interface description, not a venue setup guide. Rig-specific hardware,
camera counts, network addresses, and calibration recordings are deployment details you
supply for your own environment.
