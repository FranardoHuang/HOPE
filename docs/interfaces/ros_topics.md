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

Planner cadence is deliberately different from sensor cadence: every qualified `/poses` sample is
ingested and replaces the sole pending immutable solve snapshot. Production arena/schema-4 profiles
bind `use_latest_only_solve_worker=true` and start expensive Stage 2/3 at most at 50 Hz
(`solve_period_s=0.02`), with no FIFO and no catch-up burst. Publication/task lifecycle stay on the
ROS executor; stale completions are rejected across source revoke, no-ball/close/rearm, epoch and base
authority boundaries. A newer valid base sequence or ordinary same-ball revision does not invalidate
the captured completion. Source regressions cover bounded latest-value and lifecycle behavior;
ROS/Jazzy 300 Hz load plus injected slow-solve latency remains a runtime Gate.

Timestamp boundary (source implemented 2026-07-16, runtime still `Partial`): the vendored VRPN
tracker now has an explicit `source_timestamp_mode`:

- `receipt` is the default and preserves the historical host-receipt stamp;
- `vrpn_packet` preserves each pose/twist/accel callback's VRPN `msg_time` only when its absolute
  difference from the local ROS clock is at most `vrpn_source_max_abs_skew_s` (default `0.1 s`).

Packet mode rejects negative/overflowing seconds, invalid microseconds, non-finite time, bad clock
configuration and excessive skew by suppressing that sample; it never falls back to receipt time.
It therefore requires recorded PTP/NTP-equivalent synchronization between the VRPN producer and ROS
host. The source helper has a dependency-light C++ pass, but ROS/Jazzy package build and an actual
capture→planner→runner latency report remain open. Until those pass, production must keep `receipt`
and must not claim camera-capture latency compensation.

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

### Native deploy flat topics

| Topic | Type | Contract |
| --- | --- | --- |
| `/racket/command_flat` | `std_msgs/Float64MultiArray` | Schema 1 is legacy 12-value position/velocity. Face schema 2 is exact16 with physical face-B normal and zero rho. Formal schema 3 is exact20 and adds shared control epoch, racket sequence, exact base-sequence reference and source-monotonic time. Task-revision schema 4 is exact22. Valid and task-scoped invalid rows both require positive `task_id` and positive monotonic `task_revision`; only a global invalid revoke may use `(0,0)`, and mixed-zero pairs are malformed. Valid rows require explicit `swing_sign=+1/-1`. |
| `/a3/base_pose_flat` | `std_msgs/Float64MultiArray` | Legacy schema 1 is exact9. Formal schema 2 is exact12 and adds shared control epoch, base sequence and source-monotonic time. Marker-local `marker_to_base_xyz` is quaternion-rotated before addition. |

Schema 4 has exactly three identity cases: positive/positive is one task-scoped valid or invalid
revision; zero/zero is an anonymous global revoke; either mixed-zero pair is malformed and globally
revokes. The ordinary `/racket/command` message has no task identity fields and remains a
legacy/diagnostic output. Formal task-revision execution is authorized only by
`/racket/command_flat` schema 4.

Formal schema 3 selects side from `(R_yaw(base)^-1 * (intercept_w-base_w)).y`, not raw world Y.
Missing, stale, malformed or implausible base state publishes finite invalid rows on both topics;
only a newer causal racket row may re-arm. Planner mocap yaw proposes side, while the C++ runner
rejects a proposal inconsistent with its boot-aligned-IMU target geometry outside the explicit
±`0.04 m` overlap. Human-readable READY logs are diagnostic only and cannot authorize a serve.

The formal command sequence remains a transport ordering field, not a ball identity. Schema 4 closes
that gap with `(control_epoch, task_id, task_revision)`: one physical inbound ball owns one task id;
each solver refresh, including task-scoped invalid rows with a positive pair, advances its revision.
A sustained no-ball gap plus a clearly inbound next track, plane/deadline close, or contact-sized outbound transition
closes the old task. The C++ consumer linearizes the task id exactly once, accepts a latest-value
mailbox that first exposes revision `N>1`, and never replays a consumed task after idle/rearm.
Position, velocity, physical face-B normal and remaining strike time may update atomically before
contact while side/clip stay frozen. Source tests pass, but vendor runtime behavior remains open.

## Deploy Runtime Topics

- `/racket/command` — consumed by `hope_wbc_runner` (the legacy 180-D Python runner). This mirror has
  no task id/revision and is diagnostic/legacy only; it cannot satisfy task-revision exactly-once.
- `/racket/command_flat` and `/a3/base_pose_flat` — consumed by the C++
  `a3_deploy_onnx_ref_pingpong --planner` path through AimRT ROS 2 subscribers. Source wiring is
  implemented. Formal schema-4 task revision is carried only by `/racket/command_flat`; vendor
  first-tick/behavior evidence and hardware use remain separate open gates.
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
| `/racket/command_flat` | Planner publication | reliable, volatile, keep-last depth 10 |
| `/a3/base_pose_flat` | Planner publication | reliable, volatile, keep-last depth 10 |
| `/planner/diagnostics` | Planner publication | default integer depth 1 |

High-rate mocap data prefers low latency over reliable delivery because fresh samples replace old ones. `/racket/command` is a control setpoint, so it uses reliable delivery with a small keep-last queue. Confirm live compatibility with downstream controllers before hardware use.

## Update Rule

Any new topic, renamed topic, changed message type, QoS change, or launch parameter change must update this file and the affected operation doc.
