# ROS Topics

Status: Draft

## Current Mocap And Bringup Topics

Configured in `hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml`:

| Topic | Meaning |
| --- | --- |
| `/table/pose` | Table pose in HOPE frame |
| `/P1/pose` | Player 1 robot base or mocap rigid body pose |
| `/P2/pose` | Player 2 robot base or mocap rigid body pose |
| `/ball/point` | Ball position as point |
| `/poses` | PoseArray ordered as `["ball", "PPT", "P1", "P2"]` |

VRPN client namespace:

- `/vrpn_mocap`

## Current Planner Topics

The exact planner subscriptions/publications should be confirmed from source and launch before live use.

Relevant files:

- `hope_ws/src/hope_planner/launch/hope_planner.launch.py`
- `hope_ws/src/hope_planner/config/hope_planner.yaml`
- `hope_ws/src/hope_planner/hope_planner/node.py`
- `hope_ws/src/hope_msgs/msg/RacketCommand.msg`

## QoS Notes

High-rate mocap data should prefer low latency over reliable delivery when the stream is continuous. Confirm QoS in live tests and record changes here.

## Update Rule

Any new topic, renamed topic, changed message type, QoS change, or launch parameter change must update this file and the affected operation doc.
