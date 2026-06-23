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

## VRPN Client

After building and sourcing `hope_ws`, launch the VRPN client:

```bash
ros2 launch vrpn_mocap client.launch.yaml server:=PLACEHOLDER_MOCAP_SERVER_IP port:=3883
```

Confirm live topics:

```bash
ros2 topic list | grep vrpn_mocap
```

## HOPE Relay

Launch the relay:

```bash
ros2 launch hope_bringup avatar_pro_vrpn_relay.launch.py
```

Expected HOPE-standard topics are listed in [../interfaces/ros_topics.md](../interfaces/ros_topics.md).

## Verification

```bash
ros2 topic echo /ball/point --once
ros2 topic echo /poses --once
ros2 run tf2_ros tf2_echo world P1_base_link
```

Replace frame names based on live object names and measured transforms.

## Update Rule

After the first successful live mocap run, record the real server IP convention, exact topic names, object names, and QoS notes here and in G01.
