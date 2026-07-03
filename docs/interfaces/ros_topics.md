# ROS Topics

Status: Draft

## Current Mocap And Bringup Topics

Configured in `hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml`:

| Topic | Meaning |
| --- | --- |
| `/table/pose` | Table pose in HOPE frame (setup/calibration anchoring; optional during play) |
| `/P1/pose` | Player 1 robot base rigid-body pose (6-DOF, streamed at play time) |
| `/P2/pose` | Player 2 robot base rigid-body pose (6-DOF) |
| `/ball/point` | Ball position as point — the relay deliberately drops the VRPN pose orientation; when ball spin arrives (physics-modeling phase) the relay must forward orientation or add a spin topic |
| `/poses` | PoseArray ordered as `["ball", "PPT", "P1", "P2"]` |

Rates: the rig streams at 300 Hz during play (team contract, 2026-07); the bridge launch default
`update_freq` is aligned to 300 Hz. The vendored client's own `client.yaml` default is 100 Hz —
always launch through `avatar_pro_hope_bridge.launch.py`.

Current relay config defaults the input object names to `PPT` for the table and `ppp2`/`ppp3` for the
two robot rigid bodies, while publishing the normalized output topics `/P1/pose` and `/P2/pose`. These
input names are still TODO(confirm); G01 must record the live CMTracker names before deployment or data
collection.

Ball tracking is selected by the relay/launch parameter `ball_tracking_mode`:

- `rigid_body`: `ball_object` names a CMTracker rigid body such as `Ball`; the relay publishes `/vrpn_mocap/<ball_object>/pose` position to `/ball/point`.
- `auto`: `ball_object` is ignored and the relay locks onto the moving non-rigid marker. Use this fallback only when the ball cannot be exposed as a named rigid body.

VRPN client namespace:

- `/vrpn_mocap` (raw client topics live here). Naming is currently inconsistent across the stack:
  the vendored client builds multi-sensor topics as `/vrpn_mocap/<sender>/pose_id_<n>`
  (`tracker.hpp`), the relay matcher accepts `pose` / `pose<digits>` (live-rig example
  `/vrpn_mocap/MCAvatar/pose31664`), and older docs showed `pose_<id>`. Confirm the live form on
  the rig and align the relay matcher/vendored client before relying on multi-sensor topics
  (TODO, tracked in G01).

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

Planner ball selection note: `hope_planner` identifies the ball by PoseArray index 0 on `/poses`,
not by rigid-body name; its `ball_rigid_body_name` parameter is currently unused.

## Deploy Runtime Topics

- `/racket/command` — consumed by `hope_wbc_runner` (the legacy 180-D Python runner); the C++
  deploy runner (`a3_deploy_onnx_ref_pingpong`) does not subscribe to ROS mocap/planner topics yet
  and runs scripted targets (see G07).
- `/hope/estop` — hope_wbc_runner safety gate.
- `/body_drive/*` — AGI backend command/state interface; on the MDU these run over iceoryx and are
  invisible to the ros2 CLI. A ros2/iceoryx transport mismatch presents as `rate=0` + safe-halt
  (a known transport-mismatch symptom; see `agi/a3_deploy_example/README_robot_io_backend.md`).

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
