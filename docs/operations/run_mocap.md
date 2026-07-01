# Run Mocap

Status: Draft

## Task Setup

Use the ROS environment plus the live mocap network. You do not need `vendor_assets/` for this task.

```bash
distrobox enter hope
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install
source install/setup.bash
```

Before launching, know the mocap server IP/port and the live object names configured in ChingMu/CMTracker. Record confirmed names in G01 and [../interfaces/ros_topics.md](../interfaces/ros_topics.md).

## Discover the Mocap Server + Object Names

The `server:=PLACEHOLDER_MOCAP_SERVER_IP` in the launch below is a stand-in for the real ChingMu/CMTracker host. Fill it as follows:

- Server IP: the address of the machine running ChingMu/CMTracker on the mocap LAN. Read it from the CMTracker UI/config (VRPN streaming output), or confirm reachability with `ping <ip>`. The VRPN port is `3883` unless changed in CMTracker.
- Object (rigid-body) names: the VRPN tracker/rigid-body labels configured in CMTracker. They become the raw topics `/vrpn_mocap/<server>/pose_<id>`, so you can enumerate what is actually streaming:

```bash
ros2 launch vrpn_mocap client.launch.yaml server:=PLACEHOLDER_MOCAP_SERVER_IP port:=3883
ros2 topic list | grep vrpn_mocap
```

The HOPE relay maps these to `PPT` (table), `P1`, `P2`, and `/ball/point`. `PPT`/`P1`/`P2` are TODO(confirm) placeholders until verified against the live rig. For the ball, choose one mode:

- Preferred: `ball_tracking_mode:=rigid_body ball_object:=Ball` when CMTracker exposes the ball as a named rigid body.
- Fallback: `ball_tracking_mode:=auto` when the ball is only an unnamed moving marker; `ball_object` is ignored.

Record the confirmed labels, ball mode, and ball name/id in G01 and [../interfaces/ros_topics.md](../interfaces/ros_topics.md).

## VRPN Client

After building and sourcing `hope_ws`, launch the VRPN client directly only when debugging raw mocap:

```bash
ros2 launch vrpn_mocap client.launch.yaml server:=PLACEHOLDER_MOCAP_SERVER_IP port:=3883
```

Confirm live topics:

```bash
ros2 topic list | grep vrpn_mocap
```

## HOPE Relay

For normal bringup, launch the full bridge so the VRPN client, relay, and world frame start together:

```bash
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py \
  server:=PLACEHOLDER_MOCAP_SERVER_IP \
  port:=3883 \
  update_freq:=180.0 \
  ball_tracking_mode:=rigid_body \
  ball_object:=Ball
```

Use `ball_tracking_mode:=auto` and omit `ball_object` if the ball is not a named rigid body.

Expected HOPE-standard topics are listed in [../interfaces/ros_topics.md](../interfaces/ros_topics.md).

Current limitation for G07 ping-pong deploy: these mocap topics are not yet
consumed directly by `a3_deploy_onnx_ref_pingpong`. Today they help planner and
future localizer work, but the current deploy runner still uses
`perfect_tracking` unless a separate hardware pose bridge is added.

## Verification

```bash
ros2 topic echo /ball/point --once
ros2 topic echo /poses --once
ros2 run tf2_ros tf2_echo world P1_base_link
```

Replace frame names based on live object names and measured transforms.

## Replay-Bag Fallback

You can exercise the mocap -> relay -> planner path without the live rig by replaying a recording from [../../calib_bags](../../calib_bags) instead of running the VRPN client:

```bash
ros2 bag play calib_bags/<recording>
```

While the bag plays, run the HOPE Relay and Verification steps above (skip the VRPN Client launch). The bag supplies the `/vrpn_mocap/...` (or already-relayed) topics, so the rest of the pipeline behaves as if the rig were live. Object/topic names in old recordings may predate the confirmed labels — cross-check against [../interfaces/ros_topics.md](../interfaces/ros_topics.md).

## Update Rule

After the first successful live mocap run, record the real server IP convention, exact topic names, object names, ball tracking mode, ball rigid-body name or auto-detection behavior, and QoS notes here and in G01.
