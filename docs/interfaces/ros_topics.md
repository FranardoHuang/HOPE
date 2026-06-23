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

The mocap object names `PPT`/`P1`/`P2` and the auto-detected `ball` are TODO(confirm) placeholders per [avatar_pro_vrpn.yaml](../../hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml).

VRPN client namespace:

- `/vrpn_mocap` (raw client topics live here, e.g. `/vrpn_mocap/<server>/pose_<id>`)

## Current Planner Topics

The exact planner subscriptions/publications should be confirmed from source and launch before live use.

Relevant files:

- `hope_ws/src/hope_planner/launch/hope_planner.launch.py`
- `hope_ws/src/hope_planner/config/hope_planner.yaml`
- `hope_ws/src/hope_planner/hope_planner/node.py`
- `hope_ws/src/hope_msgs/msg/RacketCommand.msg`

### RacketCommand message

Defined in [RacketCommand.msg](../../hope_ws/src/hope_msgs/msg/RacketCommand.msg). The carrying topic name should be confirmed from [node.py](../../hope_ws/src/hope_planner/hope_planner/node.py) (see also [hope_planner.launch.py](../../hope_ws/src/hope_planner/launch/hope_planner.launch.py) and [hope_planner.yaml](../../hope_ws/src/hope_planner/config/hope_planner.yaml)): TBD.

| Field | Type | Meaning |
| --- | --- | --- |
| `header` | `std_msgs/Header` | Stamp and frame |
| `position` | `geometry_msgs/Point` | Target racket position |
| `velocity` | `geometry_msgs/Vector3` | Target racket velocity |
| `normal` | `geometry_msgs/Vector3` | Target racket-face normal |
| `strike_time` | `float64` | Absolute strike time |
| `time_to_strike` | `float64` | Time remaining until strike |
| `ball_velocity_outgoing` | `geometry_msgs/Vector3` | Desired outgoing ball velocity |
| `valid` | `bool` | Whether the command is valid |
| `clears_net` | `bool` | Whether the shot clears the net |
| `bypasses_net_posts` | `bool` | Whether the shot bypasses the net posts |
| `predicted_bounces` | `int32` | Predicted number of bounces |

## QoS Notes

High-rate mocap data should prefer low latency over reliable delivery when the stream is continuous. Confirm QoS in live tests and record changes here.

## Update Rule

Any new topic, renamed topic, changed message type, QoS change, or launch parameter change must update this file and the affected operation doc.
