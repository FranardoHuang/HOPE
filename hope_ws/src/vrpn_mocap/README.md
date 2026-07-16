# vrpn_mocap (HOPE)

VRPN client for ROS 2 that publishes ChingMu/VRPN motion-capture poses. This
package originates from the ChingMu VRPN ROS 2 plugin and is vendored into
`hope_ws` for the HOPE motion-capture pipeline.

Tested target: Linux + ROS 2 Jazzy.

## Install

Install the distro package, then let `rosdep` resolve the remaining
dependencies during the workspace build:

```
sudo apt install ros-jazzy-vrpn-mocap
```

## Build (inside hope_ws)

From the `hope_ws` workspace root:

```
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run

```
ros2 launch vrpn_mocap client.launch.yaml server:=<MOCAP_SERVER_IP> port:=3883
```

`client.launch.yaml` arguments:

- `server` -- VRPN server IP/hostname (default: `localhost`)
- `port` -- VRPN server port (default: `3883`)

`config/client.yaml` parameters:

- `update_freq` (double) -- frequency of the motion-capture data publisher (default: `100.`)
- `refresh_freq` (double) -- frequency of dynamically adding newly tracked objects (default: `1.`)
- `sensor_data_qos` (bool) -- use best-effort QoS for the VRPN data stream; set to `false` for the reliable system-default QoS (default: `true`)
- `multi_sensor` (bool) -- set to `true` if more than one sensor (frame) reports on the same object (default: `false`)
- `source_timestamp_mode` (string) -- `receipt` preserves the historical behavior and stamps each pose, velocity, and acceleration sample with the local ROS clock at callback receipt. `vrpn_packet` is a strict opt-in that preserves the VRPN callback's packet/capture timestamp (default: `receipt`).
- `vrpn_source_max_abs_skew_s` (double) -- maximum absolute difference, in seconds, between a VRPN packet timestamp and the local ROS clock when `source_timestamp_mode=vrpn_packet` (default: `0.1`).

`vrpn_packet` mode assumes the VRPN server/capture clock and ROS host clock are
synchronized (for example with PTP or NTP). It validates seconds,
microseconds, integer conversion, and absolute clock difference for every pose,
velocity, and acceleration callback. A malformed, too-old, or too-far-in-the-
future sample is suppressed so downstream freshness logic can stale/revoke; the
node never silently falls back to receipt time. Invalid mode names or invalid
skew configuration fail node construction. Keep `receipt` unless clock
synchronization has been independently measured.

## Inspect the data stream

```
ros2 topic list
ros2 topic echo /vrpn_mocap/<server>/pose_<id> --once
```

Raw topics live under the `/vrpn_mocap` namespace (e.g.
`/vrpn_mocap/MCServer/pose_0`). In HOPE these raw topics are mapped by a relay
to the `/P1`, `/P2`, `/ball`, and `/poses` topics; see
[../../../docs/interfaces/ros_topics.md](../../../docs/interfaces/ros_topics.md).

## Runtime documentation

The authoritative runtime procedure for bringing up motion capture is
[../../../docs/operations/run_mocap.md](../../../docs/operations/run_mocap.md).

## License

See [LICENSE](LICENSE).
