# Running with OptiTrack (Motive / NatNet)

HOPE supports two motion-capture backends that feed the planner the **same
`/poses` contract** (`geometry_msgs/PoseArray`, ball at index 0 — see
[interfaces/ros_topics.md](interfaces/ros_topics.md)); everything downstream is
backend-agnostic:

| Backend | Venue software | Driver → adapter |
|---|---|---|
| `vrpn` (default) | Any VRPN server (e.g. ChingMu/Avatar Pro CMTracker, VRPN TCP 3883) | vendored [`vrpn_mocap`](../hope_ws/src/vrpn_mocap/README.md) → `pose_to_posearray` |
| `optitrack` | OptiTrack Motive (NatNet UDP, cmd port 1510) | vendored [`motion_capture_tracking`](../hope_ws/src/motion_capture_tracking/PIN.md) → `optitrack_mct_relay` |

This document covers the `optitrack` backend. The VRPN path is unchanged and
documented in [interfaces/ros_topics.md](interfaces/ros_topics.md) and
[mocap/README.md](../mocap/README.md).

## How the OptiTrack path works

```text
Motive (NatNet UDP)
      |  motion_capture_tracking_node        (vendored driver, namespace /optitrack)
      v
/optitrack/poses                             motion_capture_tracking_interfaces/NamedPoseArray
      |  optitrack_mct_relay (hope_bringup)  (one message per camera frame, objects by name)
      v
/poses (ball at index 0), /ball/point, /P1/pose, /P2/pose, TF
  (`/table/pose` only when a separate calibration session deliberately streams `Table`)
      |  hope_planner
      v
/racket/command
```

The driver's raw topics stay under `/optitrack/*` **on purpose**: its `poses`
topic is a `NamedPoseArray` — on the bare `/poses` name it would collide with
the HOPE `/poses` contract (`geometry_msgs/PoseArray`) as a DDS type mismatch —
and its raw `/tf` (body names verbatim) would fight the relay's transforms.
`optitrack_hope_bridge.launch.py` enforces both remaps; the relay is the only
`/poses`/TF authority.

The vendored driver is [IMRCLab
motion_capture_tracking](https://github.com/IMRCLab/motion_capture_tracking)
pinned at v1.0.9 with its `libmotioncapture`/`librigidbodytracker` submodules
materialized, non-OptiTrack vendor SDKs removed, and NatNet unicast fixes
applied — the complete provenance and patch list is in
[`hope_ws/src/motion_capture_tracking/PIN.md`](../hope_ws/src/motion_capture_tracking/PIN.md).
It uses the **open-source NatNet depacketizer**, so it runs on any platform
(including aarch64) with no closed-source NatNet SDK.

## Build

The two vendored packages build with the workspace:

```bash
cd hope_ws
rosdep install --from-paths src --ignore-src -r -y   # PCL, Eigen, fmt, Boost, ...
colcon build
source install/setup.bash
```

Additional system dependencies beyond the VRPN-only build: the vendored
manifests declare PCL and Eigen, so the `rosdep install` line resolves those.
Two build requirements are **not** declared upstream — `libfmt-dev` (the
driver package's own CMake does `find_package(fmt REQUIRED)`) and
`libboost-program-options-dev` (a `librigidbodytracker` CMake requirement) —
install them via apt if the build reports a missing `fmt` or
`Boost::program_options`. Message generation also needs the rosidl generator
pythons (`python3-empy`, `python3-lark`), part of a standard ROS 2 dev
install.

Building with a uv/conda Python shim earlier on `PATH` can make CMake's
`FindPython3` pick a non-system interpreter, failing the
`motion_capture_tracking_interfaces` message generation with
`No module named 'em'`; in that case build with
`--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3`.

VRPN-only builds stay possible: `hope_bringup` intentionally declares **no**
manifest dependency on the vendored driver (the OptiTrack scripts import its
message lazily and fail with an actionable error), so
`colcon build --packages-skip motion_capture_tracking motion_capture_tracking_interfaces`
builds everything else unchanged.

## Motive-side checklist

In Motive's Data Streaming pane (see the full table in the
[mocap reference §6.2](../mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md)):

- NatNet **enabled**, Up Axis = **Z** (critical — REP 103 Z-up), Unicast
  preferred; the driver queries the UDP command port 1510 and discovers the
  server-selected data port from Motive's response.
- Rigid Bodies **ON**; competition assets named exactly `Ball`, `P1`, and
  `P2`. `Table` is setup/calibration-only (older notes call it `PPT`) and must
  be disabled or omitted for competition. The driver streams Motive asset
  names verbatim and the relay maps by name.
  Assets created/renamed while the bridge runs self-heal in ~1–2 s (the
  vendored driver re-requests the model definition when an unnamed body
  streams, PIN.md patch #6); restart the bridge only as a fallback.
- **Ball, MODE A — rigid-body asset:** the ball is a Motive rigid-body asset
  named exactly `Ball` (≥3 markers). Set the asset **pivot to the sphere
  center** (the planner's bounce geometry assumes ball-center positions).
  Marker streaming is not consumed in this mode. Occlusion clears the asset's
  tracking-valid bit and the driver drops it from the frame, so `/poses`
  pauses exactly like the VRPN path; expect dropouts on fast/spinning balls if
  fewer than 3 markers stay co-visible. The librigidbodytracker block in
  [`optitrack_mct.yaml`](../hope_ws/src/hope_bringup/config/optitrack_mct.yaml)
  must stay COMMENTED OUT (an enabled same-named tracker body would fight the
  vendor body and grab stray points).
- **Ball, MODE B — single unlabeled marker** (the
  [mocap reference §5.1/5.2](../mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md)
  design): fully coated retroreflective ball = one point at the ball center.
  Uncomment the tracker block in `optitrack_mct.yaml`; then Unlabeled Markers
  **ON** and Labeled Markers **OFF** become load-bearing (the tracker
  re-acquires by nearest-neighbour with an occlusion-growing search radius —
  stray points WILL be grabbed), and place the ball near `initial_position`
  when the driver starts.
- Units stream in **metres** → `position_scale:=1.0` (default). Sanity check:
  `/P1/pose` reading hundreds means a millimetre feed → `0.001`.

Note: the "VRPN port 3883" Motive also exposes is its legacy VRPN broadcast —
NOT used by this backend (NatNet cmd 1510; the data port and
unicast-vs-multicast are auto-negotiated from the server response).

## Bringup

One command for mocap + planner (`mocap_server` = the Motive PC IP):

```bash
ros2 launch hope_bringup hope_bringup.launch.py \
  mocap_backend:=optitrack mocap_server:=<MOTIVE_PC_IP>
```

Or start the mocap side alone (also publishes the static HOPE world frame):

```bash
ros2 launch hope_bringup optitrack_hope_bridge.launch.py \
  hostname:=<MOTIVE_PC_IP> position_scale:=1.0
```

`hostname` is a REQUIRED argument with no default — venue values are passed
explicitly, never baked in. Driver config (ball tracker modes, body names):
[`config/optitrack_mct.yaml`](../hope_ws/src/hope_bringup/config/optitrack_mct.yaml).
Relay config (name → topic mapping, scale):
[`config/optitrack_relay.yaml`](../hope_ws/src/hope_bringup/config/optitrack_relay.yaml).

### Verify

```bash
ros2 topic echo --once /optitrack/poses   # raw driver frames (names must match config)
ros2 topic echo --once /poses             # HOPE contract (ball at index 0)
ros2 run hope_bringup mocap_rate_probe.py --topic /P1/pose --min-hz 180
```

`mocap_rate_probe.py` is a one-shot pass/fail rate gate (NatNet is UDP — unlike
the VRPN TCP port there is nothing to `connect()` to before launch, so mocap
liveness can only be proven by data). Counting published messages can read
lower than the camera rate under receive-side drops; that is normal for a
best-effort sensor stream.

## No-hardware smoke test

`fake_optitrack_publisher` replaces Motive + driver with a synthetic
`/optitrack/poses` feed (alternating serves; the ball entry is omitted between
serves, exercising the relay's `/poses` gating):

```bash
ros2 run hope_bringup fake_optitrack_publisher
ros2 launch hope_bringup optitrack_mct_relay.launch.py   # relay under test
ros2 topic echo /poses
```

A driver-level no-hardware test also exists: `mocap_type:=mock` on
`optitrack_hope_bridge.launch.py` runs the real driver code with a static mock
backend instead of NatNet. Caveat: mock streams only the rigid bodies defined
in `optitrack_mct.yaml`'s `rigid_bodies` block, which the shipped (MODE A)
config leaves commented out — with the default config mock emits empty frames.
Uncomment the MODE B block first (mock uses only each body's
`initial_position`), or just use `fake_optitrack_publisher` above.

For bag replay, record `/optitrack/poses` at a live session
(`ros2 bag record /optitrack/poses`) and replay it against
`optitrack_mct_relay.launch.py` (`start_mocap_node:=false` on the full bridge).

## Camera rate and the planner's `fit_window`

The planner's velocity fit uses `fit_window` **samples** (default 31 ≈ 103 ms
at a 300 Hz rig). The window is rate-coupled: keep it at ≥ ~100 ms of samples,
i.e. `round(31 × rate / 300)`. OptiTrack rigs commonly stream **360 Hz** →
set `fit_window: 37` in
[`hope_planner.yaml`](../hope_ws/src/hope_planner/config/hope_planner.yaml)
(or pass `-p fit_window:=37`). The camera rate is a venue fact — read it from
Motive, don't infer it from `ros2 topic hz` (receive-side drops read low).

## Multi-machine DDS (laptop bridge topology)

A common venue topology puts the NatNet driver on a laptop that bridges the
Motive LAN to the robot's network. Where DDS multicast discovery does not work
(venue Wi-Fi, segmented LANs), wrap each side's command with
`with_fastdds_unicast.sh` to use explicit unicast peers:

```bash
# Laptop (runs the bridge; peers with the robot host):
./hope_ws/src/hope_bringup/scripts/with_fastdds_unicast.sh --peer <ROBOT_HOST_IP> -- \
  ros2 launch hope_bringup optitrack_hope_bridge.launch.py hostname:=<MOTIVE_PC_IP>

# Robot host (peers with the laptop):
./hope_ws/src/hope_bringup/scripts/with_fastdds_unicast.sh --peer <LAPTOP_IP> -- \
  ros2 launch hope_bringup hope_bringup.launch.py mocap_backend:=optitrack ...
```

The wrapper generates a Fast DDS profile (unicast-only transport, interface
whitelist derived from the route to each peer) and sets
`ROS_STATIC_PEERS`/`RMW_IMPLEMENTATION` for the wrapped command only.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Driver starts but 0 Hz on `/optitrack/poses` | Wrong `hostname`, firewall blocking NatNet UDP (command port 1510 or the server-advertised data port), or not on the Motive LAN. `ping` the Motive PC first. |
| Objects stream but nothing relayed | Motive asset names don't match `optitrack_relay.yaml` (`Ball`/`P1`/`P2`, case-sensitive; `Table` only for calibration). Check `ros2 topic echo --once /optitrack/poses`. |
| Rigid bodies stream with empty names | Fixed by vendored patch #6 (self-heals in ~1–2 s); if persistent, restart the bridge. |
| `/P1/pose` positions in the hundreds | Millimetre feed → `position_scale:=0.001`. |
| `/poses` pauses while `/P1/pose` keeps updating | By design: the ball left the volume / lost tracking; the relay never re-emits a stale ball (protects the planner's velocity fit). |
| `optitrack_mct_relay` exits with an import error | The vendored interfaces package isn't built/sourced — build the workspace, or use the VRPN backend. |
| Planner predictions lag/noisy at 360 Hz | Scale `fit_window` with the camera rate (see above). |
