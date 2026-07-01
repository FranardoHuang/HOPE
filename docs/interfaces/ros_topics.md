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

Current relay config defaults the input object names to `PPT` for the table and `ppp2`/`ppp3` for the
two robot rigid bodies, while publishing the normalized output topics `/P1/pose` and `/P2/pose`. These
input names are still TODO(confirm); G01 must record the live CMTracker names before deployment or data
collection.

Ball tracking is selected by the relay/launch parameter `ball_tracking_mode`:

- `rigid_body`: `ball_object` names a CMTracker rigid body such as `Ball`; the relay publishes `/vrpn_mocap/<ball_object>/pose` position to `/ball/point`.
- `auto`: `ball_object` is ignored and the relay locks onto the moving non-rigid marker. Use this fallback only when the ball cannot be exposed as a named rigid body.

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

Defined in [RacketCommand.msg](../../hope_ws/src/hope_msgs/msg/RacketCommand.msg). The planner publishes it on `/racket/command` from [node.py](../../hope_ws/src/hope_planner/hope_planner/node.py) (see also [hope_planner.launch.py](../../hope_ws/src/hope_planner/launch/hope_planner.launch.py) and [hope_planner.yaml](../../hope_ws/src/hope_planner/config/hope_planner.yaml)).

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

Current planner QoS:

| Topic | Direction | QoS |
| --- | --- | --- |
| `/poses` | Planner subscription | best-effort, volatile, keep-last depth 1 |
| `/racket/command` | Planner publication | reliable, volatile, keep-last depth 10 |
| `/planner/diagnostics` | Planner publication | default integer depth 1 |

High-rate mocap data prefers low latency over reliable delivery because fresh samples replace old ones. `/racket/command` is a control setpoint, so it uses reliable delivery with a small keep-last queue. Confirm live compatibility with downstream controllers before hardware use.

## Update Rule

Any new topic, renamed topic, changed message type, QoS change, or launch parameter change must update this file and the affected operation doc.
