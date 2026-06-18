# HOPE Reimplementation Guide

This guide explains how to reimplement the HOPE reference system from the documentation in this repository. The repo is a design-document repository, not a complete runnable software stack. A successful reimplementation means you will create your own ROS 2 packages, training configuration, policy export, and deployment integration while following the reference architecture and competition constraints.

Read this file from top to bottom. Do not skip the verification gates. Each later phase assumes the earlier phase is already working.

## How To Read Placeholders And Terms

This guide uses two kinds of values:

1. **Fixed values**: numbers defined by the HOPE table frame or the rules, for example table length `2.74 m` or table height `0.76 m`. Use these directly unless the official competition setup tells you otherwise.
2. **PLACEHOLDER values**: values that depend on your robot, your network, your motion-capture PC, or your training run. Do not copy these blindly. Every placeholder is marked with a name such as `PLACEHOLDER_A3_ROBOT_IP`, `PLACEHOLDER_WANDB_ENTITY`, `PLACEHOLDER_A3_ASSET_DIR`, or `AVATAR_PRO_PC_IP`.

Placeholder rule:

1. If a value comes from Agibot, the mocap vendor, WandB, or your local hardware, this guide must say where to get it.
2. If a value comes from measurement, this guide must say what to measure, what units to use, and where to write the result.
3. If a value cannot be known from public documentation, it is marked as an external blocker. Do not invent it.
4. After replacing a placeholder, run the verification command in that section before moving on.

Plain terminology:

- **Frame**: a coordinate system. Example: `world` is the table coordinate system.
- **Pose**: position plus orientation. Position is `x, y, z`; orientation is usually quaternion `qx, qy, qz, qw`.
- **Transform**: the pose of one frame relative to another frame. Example: `P1 -> P1_base_link`.
- **Rigid body**: a tracked object made from several mocap markers that move together, such as the table or robot torso marker plate.
- **base_link**: the main robot body frame used by the robot model. It is usually near the pelvis or torso, but the exact location must come from the robot URDF or SDK.
- **URDF / MJCF / USD**: robot model file formats. URDF is common in ROS, MJCF is common in MuJoCo, and USD is common in Isaac Sim / Isaac Lab.
- **Joint order**: the exact order of robot joints in an array. This must be identical in training, ONNX export, and hardware control.
- **PD gains**: controller stiffness/damping numbers. `Kp` pulls a joint toward a target; `Kd` damps motion so it does not oscillate.
- **FK, forward kinematics**: computing a link or racket pose from robot joint angles.
- **WBC, whole-body controller**: the learned controller that moves the robot body and arm together.
- **ONNX**: the exported neural-network policy file used at deployment time.
- **QoS**: ROS 2 message delivery settings. High-rate mocap topics should usually use Best Effort and small queue depth to reduce latency.
- **AimRT / AimDK**: Agibot's runtime and robot development kit. Public X1/X2 examples exist, but A3-specific files and APIs must come from Agibot.

## 0. What You Are Reimplementing

HOPE is a humanoid robot table-tennis system with four major parts:

1. Motion capture system
   - Tracks the ping-pong table frame, robot base_link poses, and ball position.
   - Does not track the racket.

2. Model-based planner
   - Receives ball position.
   - Estimates ball state.
   - Predicts the ball trajectory.
   - Publishes a desired racket command.

3. Whole-body controller training
   - Uses human swing motion references.
   - Retargets them to a humanoid robot.
   - Trains a policy using BeyondMimic-style reinforcement learning.
   - Exports the trained policy to ONNX.

4. Hardware deployment
   - Runs the planner and WBC controller in ROS 2.
   - Runs ONNX inference at 50 Hz.
   - Sends joint position commands to the humanoid's low-level controller.

The reference architecture is adapted from HITTER, but HITTER's source code and trained weights are not available. Treat this project as a clean-room blueprint.

## 1. Study The Required Source Documents

Before writing code, read these documents in this order:

1. `README.md`
   - Understand the complete architecture and supported robots.

2. `HOPE_AI_Challenge_2026_Rules_EN.docx`
   - Understand the competition constraints.
   - Pay special attention to robot requirements, communication requirements, safety rules, and the no-racket-tracking rule.

3. `mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`
   - Understand the world frame, tracked objects, table setup, and ROS 2 mocap interface.
   - The Chinese version is `mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup_ZH.md`.

4. `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`
   - This is your planner specification.
   - It contains enough algorithmic detail to implement Stages 1-3.

5. `HOPE_WBC_Simulation_Training_Reference_Setup.md`
   - This is your WBC training specification.
   - It explains motion acquisition, retargeting, preprocessing, RL training, and ONNX export.

6. `HOPE_Hardware_Deployment_Reference_Setup.md`
   - This is your deployment specification.
   - It explains the ROS 2 node graph, launch order, vendor-specific deployment examples, and safety workflow.

Verification gate:

- You can explain why the racket is never tracked by motion capture.
- You can draw the data flow from motion capture to planner to WBC to robot.
- You know the target robot is the Agibot Expedition A3, and you know which of its model files, SDK (AimDK/AimRT), robot description files, and safety interface you already have versus must still request from Agibot.

## 2. Decide Your Target Platform

Do this before creating software packages.

The target robot for this reimplementation is the **Agibot (Zhiyuan / 智元) Expedition A3** (远征 A3).

### 2.0 Confirmed Agibot Expedition A3 facts

These were verified against Agibot's official announcement and corroborating sources (June 2026). Treat any number not in this list as unconfirmed until Agibot's own documentation states it.

- Manufacturer: Agibot / Zhiyuan Robotics (智元机器人). English brand "AgiBot".
- Height: 1.73 m (1730 mm). Mass: 55 kg. Both comfortably satisfy the HOPE rule (humanoid, 1.0–1.9 m, ≤80 kg) with large margin.
- Active DOF (excluding hands): **31** — neck 2, each arm 7, waist 3, each leg 6 (`2 + 7×2 + 3 + 6×2 = 31`). The 7-DOF arms plus a 3-DOF flexible waist give the orientation redundancy and torso rotation a table-tennis swing needs.
- Arm end payload ~3 kg; advertised TCP end speed up to 2 m/s; daily walking speed up to ~1.8 m/s. No published wrist-flick speed, end-effector acceleration, or control-loop rate, so adequacy for a fast rally is plausible but **not proven** from public data.
- Optional dexterous hand (Agibot OmniHand O10: 10 active / 16 total DOF). For HOPE the hand is largely irrelevant — the racket is on a fixed mount (see step 11), so a fixed paddle bracket or the silicone-fist configuration is simpler than a dexterous hand.
- The A3 is a real, shipping commercial product (unveiled Feb 2026, first customer deliveries Apr 2026), positioned for entertainment/retail/exhibition use rather than as an open developer platform.

### 2.1 What you MUST obtain from Agibot (not public)

This is the single biggest dependency in the whole project. **No A3 URDF/MJCF/USD, joint name list, joint order, actuator parameters, PD gains, collision meshes, or SDK is published anywhere public** (verified across agibot.com, the AgibotTech GitHub org, and the AimDK docs site). The only fully open-source Agibot humanoid is the **X1** (a different 29-DOF robot), and the documented developer SDK is **AimDK_X2** for the X2 (whose URDF is obtained by contacting Agibot after-sales support). Request from Agibot, under whatever developer/partner channel applies to your A3:

- Robot model: URDF (most likely), and MJCF or USD if available.
- Exact `base_link` name, its physical location, and standing `base_link` height.
- Full joint name list and joint order (the order your controller and the exported ONNX policy must agree on).
- Joint limits, gear ratios, link inertial parameters, and collision meshes.
- Default PD gains / impedance settings and the joint command interface.
- The A3 control SDK / AimDK / ROS 2 interface, low-level joint command API, and safety (E-stop/standby) interface.
- Redistribution terms for any of the above (this affects whether you can publish a build guide that references them).

Until these arrive, develop against the **X1 open model as a stand-in** (see steps 10–16) and keep everything A3-specific behind the bridge package so swapping in the real A3 model is a config change, not a rewrite.

How to get and verify the A3 placeholders:

| Placeholder | How to get it | How to verify it |
| --- | --- | --- |
| `A3_URDF`, `A3_MJCF`, or `A3_USD` | Request from Agibot support, the A3 partner portal, or the hardware vendor contact. Ask for meshes, inertial values, joint limits, and actuator parameters in the same package. | Open the model in ROS/Isaac/MuJoCo and confirm the robot height is about `1.73 m` and the joint count matches the A3 documentation. |
| `base_link` name | Read the root or pelvis frame name in the A3 URDF, or ask Agibot for the official control-frame name. | Run `ros2 topic echo /joint_states --once` and confirm the SDK documentation uses the same body frame in its examples. |
| `base_link` height | Put the robot in the official standing calibration pose on level ground. Measure from the floor to the `base_link` origin if the origin is physically marked. If it is not marked, compute it from the robot model by FK in the standing pose. | In simulation, publish `world -> base_link`; the Z value should match the measured standing height within a few centimeters. |
| Joint names and joint order | Get the ordered joint list from the A3 SDK/API or from the A3 URDF plus SDK command message definition. | Compare the order in four places: `/joint_states`, the training config, ONNX metadata, and the hardware command message. They must match exactly. |
| PD gains / impedance | Get default safe gains from Agibot. Do not copy X1/X2 gains onto A3 unless Agibot says they are valid. | In low-gain standby, command tiny joint motions and confirm there is no buzzing, overshoot, or unexpected motion. |
| E-stop and standby API | Get the exact hardware E-stop wiring, software stop service/topic, and standby command from the A3 manual. | Time a stop test with logs or high-speed video. The required upper-body and gait stop time is below `200 ms`. |

If one of these values is missing, write it into your local `A3_BLOCKERS.md` and do not proceed to hardware deployment for that item. Public X1/X2 files are useful only for building the software path; they are not proof that the A3 hardware path is correct.

### 2.2 Middleware: AimRT + AimDK

- Agibot's robots run on **AimRT**, Agibot's open-source C++20 runtime framework. AimRT interoperates with ROS 2 (and HTTP/gRPC/MQTT/Zenoh), so a ROS-2-native HOPE stack can coexist with it.
- **AimDK** is Agibot's robot development kit; `AimDK_X2` (docs at `x2-aimdk.agibot.com`) is the published example and exposes ROS 2 interfaces. Assume the A3 has an analogous SDK and confirm it.
- Distro caveat: Agibot's published X1 deployment stack (`agibot_x1_infer`) is **ROS 2 Humble + AimRT (C++)**, while HOPE's reference is **ROS 2 Jazzy**. Verify AimRT/AimDK compatibility with your chosen ROS 2 distro early — this can force a distro decision.

### 2.3 Simulation backend

- Agibot precedent: the open X1 locomotion stack trains in **Isaac Gym (Preview 4)** and validates in **MuJoCo (sim2sim)**; Agibot's newer flagship simulator `genie_sim` is **Isaac Sim 5.1 / Isaac Lab / PhysX**. HOPE's WBC reference (BeyondMimic / `whole_body_tracking`) is also Isaac Lab based.
- Recommendation: use **Isaac Lab + PhysX** as the primary training backend (aligns with BeyondMimic and `genie_sim`), and **MuJoCo** for sim-to-sim verification (aligns with the X1 stack). If Agibot ships only a URDF, both Isaac Lab (URDF→USD) and MuJoCo (URDF→MJCF) are viable; either way the actuator params, PD gains, and collision meshes must come from Agibot.

### 2.4 Deployment path

- Keep the HOPE planner and WBC interface in **plain ROS 2 first**.
- Add an Agibot bridge (`agibot_hardware_bridge` / `agibot_bringup`) that maps HOPE joint commands to the A3 low-level API via AimDK/AimRT. Mirror the X1 deployment pattern: ONNX Runtime inference, subscribe `/joint_states`, publish `/joint_cmd`, with a ~1 kHz PD loop. Note HOPE's WBC policy runs at **50 Hz** (vs the X1 walking policy's 100 Hz).
- Consider a native AimRT-only path only after the ROS 2 bridge is working and safe.

Verification gate:

- You have (or have formally requested) the A3 robot model file and know the joint order expected by your controller.
- You know how to send joint position commands safely to the robot through AimDK/AimRT.
- You know how to stop the robot immediately (hardware E-stop and software soft-stop).

## 3. Create A Workspace Layout

Use separate repositories or folders for each major subsystem. Do not mix external upstream repositories into this documentation repo.

Recommended layout:

```text
~/workspace/HOPE/hope_ws/
  src/
    hope_msgs/
    hope_planner/
    hope_bringup/
    hope_monitoring/
    # mocap: ros-jazzy-vrpn-mocap (apt) + the avatar_pro relay in hope_bringup
    motion_tracking_controller/      # for WBC deployment
    agibot_bringup/                  # Agibot A3 launch/bringup (AimDK adapter)
    agibot_hardware_bridge/          # Agibot A3 low-level bridge (AimRT/AimDK)

~/workspace/HOPE/hope_training/
  GVHMR/
  GMR/
  whole_body_tracking/
  mjlab/                             # MuJoCo backend / sim2sim path only
  motions/
    raw_video/
    smplx/
    retargeted/
    preprocessed/
  policies/
```

Install baseline system tools:

```bash
sudo apt update
sudo apt install git curl build-essential cmake python3-pip python3-venv
```

Install ROS 2:

- HOPE's reference environment is Ubuntu 24.04 LTS plus ROS 2 Jazzy.
- If your host is Ubuntu 26.04, use Distrobox or Docker for the HOPE ROS 2 workspace instead of mixing 24.04/Jazzy apt packages into the host OS.
- Use a native install only when the machine itself is Ubuntu 24.04.
- ROS 2 Humble on Ubuntu 22.04 is allowed by the rules for external communication, but the reference docs use Jazzy.

Distrobox path for an Ubuntu 26.04 host:

Use this path for day-to-day development because it feels close to a normal host shell while keeping ROS 2 Jazzy inside an Ubuntu 24.04 container.

This machine has a suitable Distrobox named `hope`. It is Ubuntu 24.04; the block below verifies or installs the ROS 2 Jazzy desktop stack inside it. Copy and paste this whole block into your host terminal:

```bash
cd ~/workspace/HOPE

mkdir -p \
  ~/workspace/HOPE/hope_ws/src/hope_planner \
  ~/workspace/HOPE/hope_ws/src/hope_bringup \
  ~/workspace/HOPE/hope_ws/src/hope_monitoring \
  ~/workspace/HOPE/hope_ws/src/agibot_bringup \
  ~/workspace/HOPE/hope_ws/src/agibot_hardware_bridge \
  ~/workspace/HOPE/hope_training/GVHMR \
  ~/workspace/HOPE/hope_training/GMR \
  ~/workspace/HOPE/hope_training/whole_body_tracking \
  ~/workspace/HOPE/hope_training/mjlab \
  ~/workspace/HOPE/hope_training/motions/raw_video \
  ~/workspace/HOPE/hope_training/motions/smplx \
  ~/workspace/HOPE/hope_training/motions/retargeted \
  ~/workspace/HOPE/hope_training/motions/preprocessed \
  ~/workspace/HOPE/hope_training/policies

HOPE_ROS_BOX=hope

if ! distrobox list | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' | grep -qx "$HOPE_ROS_BOX"; then
  sudo apt update
  sudo apt install -y distrobox podman
  distrobox create --yes \
    --name "$HOPE_ROS_BOX" \
    --image docker.io/osrf/ros:jazzy-desktop-full
fi

distrobox enter "$HOPE_ROS_BOX" -- bash -lc '
set -e

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  git \
  ros-jazzy-desktop-full \
  ros-jazzy-vrpn-mocap \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  python3-venv
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ERROR: ROS 2 Jazzy did not install correctly; /opt/ros/jazzy/setup.bash is missing."
  exit 1
fi
rosdep update || true
if ! grep -q "HOPE ROS 2 Jazzy environment" ~/.bashrc; then
  {
    printf "\n# HOPE ROS 2 Jazzy environment.\n"
    printf "if [ -f /opt/ros/jazzy/setup.bash ]; then\n"
    printf "    source /opt/ros/jazzy/setup.bash\n"
    printf "fi\n\n"
    printf "if [ -f \"\\$HOME/workspace/HOPE/hope_ws/install/setup.bash\" ]; then\n"
    printf "    source \"\\$HOME/workspace/HOPE/hope_ws/install/setup.bash\"\n"
    printf "fi\n"
  } >> ~/.bashrc
fi
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE
echo "ROS_DISTRO=$ROS_DISTRO"
ros2 --help >/dev/null
rviz2 --help >/dev/null
echo "ros2-ok"
exec bash -l
'
```

After entering the Distrobox shell, `~/workspace/HOPE`, `~/workspace/HOPE/hope_ws`, and `~/workspace/HOPE/hope_training` are the same host folders, so files you edit inside the container remain visible on the host. Later, enter the environment with `distrobox enter hope`.

Docker path for an Ubuntu 26.04 host:

Use this path if you want an explicit Docker image, need to debug low-level Docker run flags, or need a more conventional container setup for hardware integration.

Copy and paste this whole block into your host terminal:

```bash
cd ~/workspace/HOPE

sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker

DOCKER_CMD="docker"
if ! docker info >/dev/null 2>&1; then
  DOCKER_CMD="sudo docker"
fi

cat > Dockerfile.hope-ros2-jazzy <<'EOF'
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep update || true

WORKDIR /root/workspace/HOPE
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
EOF

$DOCKER_CMD build -t hope-ros2-jazzy -f Dockerfile.hope-ros2-jazzy .
mkdir -p \
  ~/workspace/HOPE/hope_ws/src/hope_planner \
  ~/workspace/HOPE/hope_ws/src/hope_bringup \
  ~/workspace/HOPE/hope_ws/src/hope_monitoring \
  ~/workspace/HOPE/hope_ws/src/agibot_bringup \
  ~/workspace/HOPE/hope_ws/src/agibot_hardware_bridge \
  ~/workspace/HOPE/hope_training/GVHMR \
  ~/workspace/HOPE/hope_training/GMR \
  ~/workspace/HOPE/hope_training/whole_body_tracking \
  ~/workspace/HOPE/hope_training/mjlab \
  ~/workspace/HOPE/hope_training/motions/raw_video \
  ~/workspace/HOPE/hope_training/motions/smplx \
  ~/workspace/HOPE/hope_training/motions/retargeted \
  ~/workspace/HOPE/hope_training/motions/preprocessed \
  ~/workspace/HOPE/hope_training/policies

$DOCKER_CMD run --rm -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v "$PWD":/root/workspace/HOPE \
  -w /root/workspace/HOPE \
  hope-ros2-jazzy
```

This block creates `Dockerfile.hope-ros2-jazzy`, builds the `hope-ros2-jazzy` image, creates your host workspace folders inside `~/workspace/HOPE`, and opens a shell inside the ROS 2 Jazzy container with that same repository mounted at `~/workspace/HOPE`.

For GPU simulation or training, install NVIDIA Container Toolkit on the host and add `--gpus all` to the `docker run` command. For real hardware, keep `--net=host` so ROS 2 DDS discovery and the Agibot A3 robot network (AimRT/AimDK) are visible inside the container.

Verification gate:

- `ros2 --help` works inside the Ubuntu 24.04/Jazzy environment and `echo "$ROS_DISTRO"` prints `jazzy`.
- You can create and build an empty colcon workspace.
- You can source your workspace with `source install/setup.bash`.

## 4. Implement The Shared ROS 2 Messages

Create a package:

```bash
cd ~/workspace/HOPE/hope_ws/src
ros2 pkg create hope_msgs --build-type ament_cmake
mkdir -p hope_msgs/msg
```

Create `msg/RacketCommand.msg`:

```text
std_msgs/Header header
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 normal
float64 strike_time
float64 time_to_strike
geometry_msgs/Vector3 ball_velocity_outgoing
bool valid
bool clears_net
bool bypasses_net_posts
int32 predicted_bounces
```

Update `package.xml` dependencies:

```xml
<depend>std_msgs</depend>
<depend>geometry_msgs</depend>
<buildtool_depend>rosidl_default_generators</buildtool_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

Update `CMakeLists.txt`:

```cmake
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RacketCommand.msg"
  DEPENDENCIES std_msgs geometry_msgs
)
```

Build:

```bash
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_msgs
source install/setup.bash
ros2 interface show hope_msgs/msg/RacketCommand
```

Verification gate:

- `ros2 interface show hope_msgs/msg/RacketCommand` prints the message definition.

## 5. Set Up The HOPE World Frame

Use the canonical HOPE frame everywhere:

```text
Origin: near-side left corner of the table surface from P1 perspective
X: toward P2 along the table length
Y: left from P1 perspective, along table width
Z: up
Table surface: z = 0
Floor below origin: z = -0.76
```

Table landmarks:

```text
Origin:              [0.0,   0.0,     0.0]
Net center line:     [1.37, -0.7625,  0.0]
P1 half center:      [0.685,-0.7625,  0.0]
P2 half center:      [2.055,-0.7625,  0.0]
Virtual hit plane:   x ~= 0.0
```

How these numbers are calculated:

1. Standard table length is `2.74 m`; the net is halfway along the length, so `net_x = 2.74 / 2 = 1.37`.
2. Standard table width is `1.525 m`; the center line in the HOPE Y convention is `center_y = -1.525 / 2 = -0.7625`.
3. P1 half center is halfway between the P1 edge and the net: `x = 1.37 / 2 = 0.685`, `y = -0.7625`, `z = 0`.
4. P2 half center is halfway between the net and the far edge: `x = 1.37 + 0.685 = 2.055`, `y = -0.7625`, `z = 0`.
5. Table surface is defined as `z = 0`. The floor is about `z = -0.76` because a standard table is `0.76 m` tall.

Implementation rules:

1. Use meters.
2. Use ROS 2 REP 103.
3. Use Z-up.
4. Do not silently convert to table-center coordinates.
5. If you must support another frame internally, write explicit conversion functions and unit tests.

Implement the world frame in a bringup package:

```bash
cd ~/workspace/HOPE/hope_ws/src
ros2 pkg create hope_bringup --build-type ament_cmake
mkdir -p hope_bringup/config hope_bringup/launch
```

Create `hope_bringup/config/hope_world_frame.yaml` with:

- Frame names for `world`, `PPT`, `P1`, `P2`, `P1_base_link`, `P2_base_link`, and the landmark helper frames.
- Table dimensions and HOPE landmarks in meters.
- `planner.x_hit`.
- `mocap_to_base_link` offsets for P1 and P2.

Create `hope_bringup/launch/hope_world.launch.py` so it publishes:

- `world -> table_center`
- `world -> p1_half_center`
- `world -> p2_half_center`
- `world -> net_center`
- `world -> floor_origin`
- `world -> virtual_hit_plane`
- `P1 -> P1_base_link`
- `P2 -> P2_base_link`

`mocap_to_base_link` is a required measured transform, not a guess.

What it means:

```text
P1 frame          = the rigid-body frame reported by motion capture for the P1 marker plate
P1_base_link     = the robot model's base_link frame
mocap_to_base_link.p1_xyz = translation from P1 marker frame to P1_base_link, in meters
mocap_to_base_link.p1_rpy = roll, pitch, yaw from P1 marker frame to P1_base_link, in radians
```

How to get `mocap_to_base_link`:

1. Best method: define the mocap rigid-body pivot and axes directly at the robot `base_link`. If Avatar-Pro lets you set a pivot and local axes, set `P1` to match `P1_base_link`. Then the offset is `[0, 0, 0]` and `[0, 0, 0]`.
2. Physical measurement method: mount the mocap marker plate rigidly on the torso or pelvis. Mark the marker-frame origin on the plate. Measure the vector from that origin to the robot `base_link` origin with a ruler, calipers, or a laser measure. Write the result in meters in the marker frame axes. Example: `12 cm forward, 3 cm left, 8 cm up` becomes `[0.12, 0.03, 0.08]` if the marker frame axes use forward/left/up.
3. Orientation method: align the marker plate axes with the robot base axes during mounting. If the plate is aligned, `p1_rpy` starts as `[0, 0, 0]`. If it is rotated, measure roll/pitch/yaw with a digital level/protractor or solve it by comparing a known standing pose in mocap and in the robot model.
4. ROS check method: publish both `world -> P1` from mocap and `world -> P1_base_link` from the robot/model. The transform `P1 -> P1_base_link` is the offset to put in `hope_world_frame.yaml`.

Write the values here:

```yaml
mocap_to_base_link:
  p1_xyz: [PLACEHOLDER_MEASURED_X_M, PLACEHOLDER_MEASURED_Y_M, PLACEHOLDER_MEASURED_Z_M]
  p1_rpy: [PLACEHOLDER_MEASURED_ROLL_RAD, PLACEHOLDER_MEASURED_PITCH_RAD, PLACEHOLDER_MEASURED_YAW_RAD]
  p2_xyz: [PLACEHOLDER_MEASURED_X_M, PLACEHOLDER_MEASURED_Y_M, PLACEHOLDER_MEASURED_Z_M]
  p2_rpy: [PLACEHOLDER_MEASURED_ROLL_RAD, PLACEHOLDER_MEASURED_PITCH_RAD, PLACEHOLDER_MEASURED_YAW_RAD]
```

Zero is allowed only when you intentionally made the mocap rigid-body frame equal to `base_link`. Otherwise zero is a temporary placeholder and must be replaced before hardware tests.

Build and verify:

```bash
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_msgs hope_bringup
source install/setup.bash
ros2 launch hope_bringup hope_world.launch.py
```

In another terminal:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/setup.bash
ros2 run tf2_ros tf2_echo world table_center
'
```

Verification gate:

- A table corner marker at the origin reports approximately `[0, 0, 0]`.
- A point at the opponent half center reports approximately `[2.055, -0.7625, 0]`.
- The ball height above the table is positive.
- `ros2 run tf2_ros tf2_echo world table_center` reports approximately `[1.37, -0.7625, 0.0]`.
- `ros2 run tf2_ros tf2_echo P1 P1_base_link` reports the measured `p1_xyz` and `p1_rpy`, not an accidental zero offset.

## 6. Set Up Motion Capture (Avatar Pro / Chingmu VRPN)

HOPE on the Agibot Expedition A3 uses a single motion-capture path: **Avatar Pro**
(Chingmu / 青瞳) streamed over **VRPN** into ROS 2. The OptiTrack / Vicon /
`motion_capture_tracking` alternatives from the reference documents are out of
scope for this build; their per-vendor details remain in
`mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md` if you ever
need them.

Hardware requirements:

1. Use at least 6 cameras.
2. Prefer 8-12 cameras.
3. Use at least 120 Hz.
4. Prefer 240-360 Hz for fast ball tracking.
5. Cover the full table plus at least 1.5 m margin on each player's side.

Create the table rigid body:

1. Attach at least 4 asymmetric reflective markers to the outer table frame.
2. Do not place markers on the playing surface.
3. Name the table rigid body `PPT`.
4. Set the rigid body pivot to the near-side left corner of the table surface.
5. Align the PPT local frame with the HOPE world frame.
6. Confirm the stationary PPT pose is approximately identity.

How to set the table pivot and axes:

1. Put a small temporary mark on the near-side left table-surface corner from P1's perspective. This is the HOPE origin.
2. In the mocap software, set the rigid-body pivot/origin of `PPT` to that corner. If the software cannot set the pivot exactly, record the offset from the software pivot to that corner and apply it in the bridge.
3. Align the `PPT` local X axis along the table length toward P2.
4. Align the `PPT` local Y axis across the table width so table points have negative Y values inside the table, matching this guide.
5. Align Z upward.
6. Test with a marker or calibration wand: the origin corner should read close to `[0, 0, 0]`; the far-right/far-left signs should match the Y convention before you continue.

Create robot base_link rigid bodies:

1. Add at least 4 asymmetric markers to a rigid torso or pelvis plate.
2. Name the P1 robot rigid body `P1`.
3. Name the P2 robot rigid body `P2`.
4. Measure the static transform from the marker cluster frame to the robot URDF base_link.
5. Put those offsets into `hope_bringup/config/hope_world_frame.yaml`.
6. Launch `ros2 launch hope_bringup hope_world.launch.py` to publish the `P1 -> P1_base_link` and `P2 -> P2_base_link` transforms.

How to check the robot rigid body:

1. The marker plate must not flex. If the plate flexes, the rigid-body pose will jump even when the robot is still.
2. The markers must be asymmetric. If the pattern is symmetric, mocap can rotate the body by 180 degrees without warning.
3. Stand the robot in the official zero or calibration pose. Record `world -> P1` for 10 seconds.
4. The position noise should be small compared with the ball radius. If it drifts centimeters while the robot is still, fix the marker placement or camera calibration before using the data.
5. Measure `P1 -> P1_base_link` using the `mocap_to_base_link` recipe in Step 5.

Prepare the ball:

1. Use an official 40+ table-tennis ball.
2. Add one small retroreflective marker or reflective coating.
3. Keep added mass as low as possible.
4. Do not use multiple ball markers.
5. Track it as a single unlabeled marker.

The mocap hardware steps above (cameras, calibration, rigid bodies, ball coating)
require the physical Avatar Pro / Chingmu rig and are done in the arena, not on a
development laptop.

Stream, do not export animation. Avatar Pro can emit several formats; for live
HOPE control use the real-time stream, not an animation export. Prefer, in order:

1. Real-time `VRPN` stream, preferred.
2. Real-time `LiveStream`, SDK stream, or ROS-compatible stream, acceptable if a bridge can convert it.
3. `C3D`, `TRC`, `TS`, or CSV-like marker/rigid-body export for offline debugging only.
4. `FBX` or `BVH` only for human reference motion/training clips, not for live ball/table/robot tracking.

Avatar-Pro setup for this project:

In Avatar-Pro:

1. Use rigid-body / marker tracking, not only avatar skeleton solving.
2. Set the streaming coordinate frame to Z-up if the software exposes that option.
3. Calibrate the world origin at the P1 near-side left table corner.
4. Align +X toward P2 along the table length.
5. Create the rigid bodies and name them exactly `PPT`, `P1`, and `P2`. Track
   each as a 6-DOF rigid body. These names DO appear in the topic path, so
   spelling and capitalization matter.
6. The ball is different, and this is the part that most often goes wrong. A
   ping-pong ball carries one marker and cannot form a rigid body (a rigid body
   needs >=3 markers), so CMTracker cannot give it an asset name like `ball` the
   way it does for the rigid bodies. You only get whatever the VRPN stream emits
   for that single point. So:
   - Make the single ball marker a tracked point with a unique, recorded id. A
     free/unlabeled marker is usually NOT streamed at all, so it must be defined
     as a tracked object/marker, not just a stray reflection on the table.
   - Enable individual-marker (single-point) output in the VRPN/stream settings,
     not only rigid-body output, or the point never enters the stream.
   - Use a fully coated reflective ball if possible (one bright blob at the true
     ball center; see the mocap reference doc section 5.4).
7. Enable VRPN streaming, usually on port `3883`.
8. Record the Avatar-Pro PC IP address.

Because the ball usually arrives under an opaque marker id (not the name
`ball`), the HOPE relay does NOT rely on a ball name. It auto-detects the ball by
motion: among every streamed object that is not `PPT`/`P1`/`P2`, the ball is the
one whose position actually moves. So you do not have to discover and hard-code
the ball's id before you can run — wave the ball and the relay locks onto it.

How to get the Avatar-Pro IP address:

1. On the Avatar-Pro Windows PC, open PowerShell.
2. Run `ipconfig`.
3. Use the IPv4 address of the wired adapter connected to the robot/ROS network.
4. Confirm from the ROS machine:

```bash
ping AVATAR_PRO_PC_IP
```

If the rigid-body names in Avatar-Pro are not exactly `PPT`, `P1`, `P2`, either
rename them in Avatar-Pro or edit `ppt_object`/`p1_object`/`p2_object` in
`hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml` and rebuild `hope_bringup`.
The relay matches rigid bodies by their VRPN sender name, so spelling and
capitalization matter for those three. The ball is matched by motion, not name,
so `ball_object` is left empty (auto). Only set `ball_object` to a concrete id if
you have confirmed it and want to pin it instead of auto-detecting.

On the ROS 2 machine, install the VRPN client and rebuild the HOPE bringup package:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
sudo apt-get update
sudo apt-get install -y ros-jazzy-vrpn-mocap
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_bringup
source install/setup.bash
'
```

Hands-on quick path for this code version:

1. Open `hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml`.
2. Check only these 4 lines first:
   - `ppt_object: "PPT"`
   - `p1_object: "P1"`
   - `p2_object: "P2"`
   - `ball_object: ""`
3. Important:
   - `PPT`, `P1`, and `P2` must match CMTracker exactly.
   - `ball_object` should normally stay empty. This code auto-detects the ball by motion.
4. Before starting, stop old ROS nodes from previous tests:

```bash
distrobox enter hope -- bash -lc '
pkill -f avatar_pro_vrpn_relay || true
pkill -f client_node || true
pkill -f static_transform_publisher || true
ros2 daemon stop || true
ros2 daemon start
'
```

5. Start the full bridge:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/setup.bash
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py \
  server:=192.168.1.100 \
  port:=3883 \
  update_freq:=360.0
'
```

6. What you should see in that launch terminal:
   - one line saying the relay is in auto mode, similar to:
     `ball auto-detect: will lock onto the moving non-rigid marker`
   - several discovery lines, similar to:
     `discovered /vrpn_mocap/PPT/.../pose -> PPT`
     `discovered /vrpn_mocap/P1/.../pose -> P1`
   - after you move the real ball by hand, one lock line, similar to:
     `ball -> /vrpn_mocap/.../pose (moving marker ...)`
7. Very important:
   - `/ball/point` and `/poses` are not expected to publish before the relay locks onto the ball.
   - In this code version, `/poses` is published on BALL updates only, not on table/P1/P2 updates.
   - So if the ball is standing still, `/poses` may appear quiet even though the relay is healthy.

Start the whole mocap bringup with one command. `avatar_pro_hope_bridge.launch.py`
starts the VRPN client, the relay, and the static HOPE world frame together.
Replace `192.168.1.100` with the IP found by `ipconfig` on the Avatar-Pro PC:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/setup.bash
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py \
  server:=192.168.1.100 \
  port:=3883 \
  update_freq:=360.0
'
```

This launches three things:

1. `vrpn_mocap client_node` — Avatar-Pro VRPN -> `/vrpn_mocap/<sender>/<sensor>/pose`.
   The bridge runs it with `multi_sensor:=true`, so every VRPN sensor channel
   gets its own topic. This matters because the ball is a single marker, not a
   nameable rigid body, so it usually shows up only as a sensor channel (an extra
   `/<sensor>` path segment), never as a tidy `/vrpn_mocap/ball/pose`.
2. `avatar_pro_vrpn_relay` — discovers `/vrpn_mocap/*` pose topics at runtime,
   matches `PPT`/`P1`/`P2` by sender name (the extra sensor-index segment is
   ignored), auto-detects the ball as the moving non-rigid marker, and republishes
   HOPE `/poses`, `/tf`, `/ball/point`, `/{table,P1,P2}/pose`.
3. `hope_world.launch.py` — the static world-frame landmarks and `P -> P_base_link` offsets.

Set `update_freq` to your camera rate (≥240–360 Hz for the ball). The default
`vrpn_mocap` poll rate is only 100 Hz, which is too slow for fast ball tracking.

Alternative: if you already run the VRPN client elsewhere (or want to share it with
other tools), start it yourself and tell the bridge to skip its own client:

```bash
# Terminal 1: the VRPN client
ros2 launch vrpn_mocap client.launch.yaml server:=192.168.1.100 port:=3883
# Terminal 2: relay + world frame only
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py start_vrpn_client:=false
```

Verify that HOPE sees the Avatar-Pro data:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/setup.bash
ros2 topic list | grep -E "ball|P1|P2|table|poses"
ros2 topic echo /ball/point --once
ros2 topic hz /poses
ros2 run tf2_ros tf2_echo world P1
'
```

What to expect from those checks:

1. `ros2 topic list | grep -E "ball|P1|P2|table|poses"`:
   - you should at least see `/P1/pose`, `/table/pose`, `/ball/point`, `/poses`
   - topic names can exist before data is flowing, so this command alone is not enough
2. `ros2 run tf2_ros tf2_echo world P1`:
   - should print numbers if P1 is being tracked
3. `ros2 topic echo /ball/point --once`:
   - may wait until you move the real ball
   - if it waits forever, do not panic yet; first physically wave the ball in front of the cameras
4. `ros2 topic hz /poses`:
   - only becomes non-zero after the ball has been locked and starts updating
   - this is expected for this code version

The relay discovers `/vrpn_mocap/*` pose topics at runtime (it does not assume
fixed names). It matches each on the VRPN sender name — so both
`/vrpn_mocap/PPT/pose` and `/vrpn_mocap/PPT/0/pose` map to `PPT` — and treats any
non-rigid-body marker as a ball candidate:

```text
/vrpn_mocap/PPT/<sensor>/pose    -> PPT   (rigid body, by name)
/vrpn_mocap/P1/<sensor>/pose     -> P1    (rigid body, by name)
/vrpn_mocap/P2/<sensor>/pose     -> P2    (rigid body, by name)
/vrpn_mocap/<anything else>/pose -> ball candidate -> the moving one is the ball
```

and publishes:

```text
/table/pose
/P1/pose
/P2/pose
/ball/point
/poses
/tf
```

When the relay locks onto the ball it logs a line like `ball -> /vrpn_mocap/7/pose
(moving marker ...)`. If it never logs that, the ball marker is not reaching ROS
— see "If you cannot find the ball topic" below.

Required object names:

1. `PPT`: table rigid body, 6-DOF pose, near-side left table corner as pivot.
2. `P1`: player-one robot mocap rigid body, 6-DOF pose.
3. `P2`: optional opponent robot mocap rigid body, 6-DOF pose.
4. `ball`: ping-pong ball, single marker, 3-DOF position only; orientation may be
   identity and should be ignored. It is NOT a rigid body and is NOT matched by
   name — the relay auto-detects it by motion.

If you cannot find the ball topic:

The ball marker is upstream of the relay, so first check what the stock VRPN
client actually publishes, with the ball physically moving:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
# 1) list every object the VRPN server is streaming
ros2 topic list | grep vrpn_mocap
# 2) wave the ball by hand and watch which topic position changes
ros2 topic echo /vrpn_mocap/<suspect>/pose --field pose.position
'
```

Three outcomes:

1. Nothing under `/vrpn_mocap/` at all -> the client is not connected: wrong
   server IP/port, different subnet, firewall, or VRPN streaming is off in
   Avatar-Pro. Fix the connection first.
2. You see `PPT`/`P1`/`P2` but no extra moving object -> the ball is an
   Avatar-Pro problem: the single marker is not defined as a tracked marker, or
   individual-marker streaming is off. Set it up per step 6 above.
3. You see an extra topic whose position tracks the ball -> good. The relay's
   auto-detect will lock onto it; you do not need to copy its id anywhere. (If
   you prefer to pin it, put that sender name/id in `ball_object`.)
4. You see an extra moving topic, but the relay still does not lock -> lower
   `ball_lock_speed_mps` a little in `avatar_pro_vrpn.yaml`, rebuild
   `hope_bringup`, relaunch, and try again.

Required coordinate convention:

1. Units are meters.
2. Parent frame is `world`.
3. Use REP 103 Z-up.
4. Origin is the P1 near-side left corner of the table surface.
5. X points toward P2 along table length.
6. Y follows the HOPE table-width convention, with table points in `[-1.525, 0]`.
7. Z is table height, so the table surface is `z = 0` and the floor is about `z = -0.76`.

Recommended live ROS 2 mapping:

```text
/ball/point    geometry_msgs/PointStamped    ball center in world frame
/P1/pose       geometry_msgs/PoseStamped     P1 mocap rigid body in world frame
/P2/pose       geometry_msgs/PoseStamped     P2 mocap rigid body in world frame
/table/pose    geometry_msgs/PoseStamped     PPT pose in world frame
/poses         geometry_msgs/PoseArray       combined tracked objects
/tf            tf2_msgs/TFMessage            world -> PPT/P1/P2/ball
```

Verification commands:

```bash
ros2 topic list
ros2 topic hz /poses
ros2 topic hz /tf
ros2 topic echo /tf --once
```

Verification gate:

- `/poses` or `/tf` is publishing at the camera rate.
- `PPT`, `P1`, `P2`, and the ball are visible.
- The relay logged `ball -> /vrpn_mocap/<id>/pose` and `/ball/point` moves only
  when the real ball moves (not when a robot or the table moves).
- The ball is not confused with table or robot markers.
- No racket, wrist, hand, or paddle marker exists.

## 7. Implement The Planner Package

> Reference-implementation status: this package is already implemented in this
> repository under `hope_ws/src/hope_planner`, built with `colcon`, and verified
> in the `hope` distrobox (17 unit tests pass; the node publishes valid
> `hope_msgs/RacketCommand` for a simulated incoming ball). The algorithm code is
> taken verbatim from `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`
> with two corrections, noted at the end of this section. Use the steps below to
> understand or rebuild it.

Practical execution order note:

- The section numbers here describe the build-up of the system, not the only valid day-to-day working order.
- For a real beginner doing live bringup, the most practical order is usually:
  1. Step 6: make sure mocap topics are real and stable.
  2. Step 7 quick smoke test: confirm the planner node starts and can read `/poses`.
  3. Step 8: record ball trajectories and calibrate `drag_k`, `restitution_h`, and `restitution_v`.
  4. Step 7 real validation again: re-run the planner with the calibrated values.
- So yes: if your goal is "get accurate physics first", then Step 6 -> Step 8 -> Step 7 is a reasonable path.
- The only reason to touch Step 7 before Step 8 is to do a quick health check with the default values and make sure the whole data path is alive.

Create the package:

```bash
cd ~/workspace/HOPE/hope_ws/src
ros2 pkg create hope_planner --build-type ament_python \
  --dependencies rclpy geometry_msgs std_msgs diagnostic_msgs hope_msgs
```

Files (as implemented):

```text
hope_planner/
  hope_planner/
    __init__.py
    constants.py                 # TableParams, BallPhysics, PlannerConfig
    ball_state_estimator.py      # Stage 1
    ball_trajectory_predictor.py # Stage 2 (+ StrikeTarget)
    racket_target_planner.py     # Stage 3 (+ RacketCommand dataclass)
    planner.py                   # HOPEPlanner pipeline (Stages 1-3)
    quaternion_utils.py          # normal_to_quaternion
    calibration.py               # calibrate_ball_physics + hope_calibrate CLI (step 8)
    node.py                      # ROS 2 node -> publishes hope_msgs/RacketCommand
  config/
    hope_planner.yaml
  launch/
    hope_planner.launch.py
  test/
    test_ball_state_estimator.py
    test_ball_trajectory_predictor.py
    test_racket_target_planner.py
    test_quaternion_utils.py
    test_calibration.py
```

Implement constants:

```text
Table length: 2.74 m
Table width: 1.525 m
Table height: 0.76 m
Net x: 1.37 m
Net height: 0.1525 m
Net overhang: 0.15 m
Ball radius: 0.02 m
Ball mass: 0.0027 kg
Gravity: [0, 0, -9.81]
Default drag k: 0.5
Default horizontal restitution C_h: 0.75
Default vertical restitution C_v: 0.85
Default racket restitution C_r: 0.88
Planner integration dt: 0.001 s
Mocap rate: 360 Hz
Polynomial fit window: 31 samples
```

Plain-language note about `restitution_racket`:

- This is not the racket pose.
- This is not tracked by motion capture.
- It is just one number that says how strongly the ball bounces off the racket face.
- Bigger number: the ball keeps more speed after the hit.
- Smaller number: the ball loses more speed after the hit.
- If you do not have the robot yet, keep `0.88` for now and calibrate later.
- Easy first calibration: hold the racket still, measure ball speed just before and just after impact, then use:

```text
restitution_racket ~= rebound_speed / incoming_speed
```

This simple formula is for a still racket and should use the speed component normal to the racket face.

Implement Stage 1, ball state estimation:

1. Keep a sliding buffer of recent timestamps.
2. Keep a sliding buffer of recent ball positions.
3. Use at least 6 samples before estimating.
4. Fit a second-order polynomial independently for x, y, and z.
5. Normalize timestamps around the newest sample before fitting.
6. Use the polynomial value at the newest sample as smoothed position.
7. Use the polynomial derivative at the newest sample as velocity.
8. Detect table bounce with a three-sample z pattern:
   - previous previous z is above threshold,
   - previous z is at or below threshold,
   - current z is above threshold.
9. Clear the fit buffer immediately after a detected bounce.

Unit tests:

1. Constant velocity input returns the correct velocity.
2. Parabolic z input returns the correct vertical velocity.
3. Bounce pattern clears the buffer.
4. Fewer than 6 samples does not produce a command.

Implement Stage 2, trajectory prediction:

1. Input current estimated ball position, velocity, and timestamp.
2. Integrate forward using explicit Euler at 1 kHz.
3. Use flight acceleration:

```text
a = -k * norm(v) * v + g
```

4. Detect table contacts when the next z crosses below 0 and velocity is downward.
5. Only apply bounce if the contact point is within table bounds expanded by ball radius.
6. Apply restitution:

```text
v_plus = diag(C_h, C_h, -C_v) * v_minus
```

7. Interpolate within the timestep to estimate the exact bounce moment.
8. Continue the remaining timestep after the bounce.
9. Detect crossing of the virtual hit plane `x_hit`.
10. Only produce a strike target when the ball is moving toward P1.
11. Return predicted ball position, velocity, strike time, bounce count, and validity.

Unit tests:

1. A known incoming trajectory crosses `x_hit`.
2. A ball moving away from P1 does not produce a valid command.
3. A table bounce reverses z velocity.
4. A bounce outside table bounds is not treated as a valid table bounce.

Implement Stage 3, racket target planning:

1. Use the Stage 2 strike target.
2. Set the intercept position equal to predicted ball position at the hit plane.
3. Aim at the center of the opponent half:

```text
target_land = [2.055, -0.7625, 0.0]
```

4. Compute desired outgoing ball velocity:

```text
v_out = (p_land - p_intercept) / delta_t_flight + 0.5 * g * delta_t_flight
```

5. Compute racket face normal:

```text
u = normalize(v_out - v_in)
```

6. Compute desired racket velocity:

```text
v_racket = ((dot(v_out, u) + C_r * dot(v_in, u)) / (1 + C_r)) * u
```

7. Check net clearance at `x = 1.37`.
8. Check whether the ball bypasses the net posts.
9. If net clearance fails, try adjusted flight times such as 0.4, 0.6, 0.35, 0.7, and 0.3 seconds.
10. Return `RacketCommand`.

Unit tests:

1. A normal incoming ball produces a valid command.
2. The normal vector has unit length.
3. The command reports net clearance correctly.
4. Invalid strike targets produce `valid = false`.

Implement quaternion conversion:

1. Convert the racket normal to a quaternion.
2. Use a stable fallback when the normal is nearly parallel to the chosen up vector.
3. If `constrain_up` is true, align the handle direction with world negative Z as much as possible.
4. Unit test that the quaternion is normalized.

Implement the ROS 2 node:

1. Subscribe to `/poses` or `/tf`.
2. Extract the ball position.
3. Feed each ball position into the planner.
4. Publish `/racket/command` as `hope_msgs/msg/RacketCommand`.
5. Publish diagnostics at 10 Hz.
6. Use Best Effort QoS with depth 1 for high-rate mocap topics.

Config (`config/hope_planner.yaml`, as implemented):

```yaml
hope_planner:
  ros__parameters:
    ball_rigid_body_name: "pingpong_ball"
    ball_pose_index: 0          # which slot in /poses PoseArray is the ball
    x_hit: 0.0
    target_land_x: 2.055
    target_land_y: -0.7625
    delta_t_flight: 0.5
    drag_k: 0.5
    restitution_h: 0.75
    restitution_v: 0.85
    restitution_racket: 0.88
```

Verification commands:

```bash
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_msgs hope_planner
source install/setup.bash
colcon test --packages-select hope_planner   # or: python3 -m pytest src/hope_planner/test
ros2 launch hope_planner hope_planner.launch.py
ros2 topic hz /racket/command
ros2 topic echo /racket/command --once
```

Verification gate:

- Tossing a ball toward P1 causes `/racket/command` to publish valid commands.
- A ball moving away from P1 produces no valid strike command.
- `time_to_strike` is positive and decreases as the ball approaches.
- Planner runtime is below 5 ms per update.

Two corrections vs. the reference-doc skeleton (already applied in this repo):

1. The node publishes the full `hope_msgs/RacketCommand` (position, velocity,
   normal, strike_time, time_to_strike, ball_velocity_outgoing, valid, clears_net,
   bypasses_net_posts, predicted_bounces). The skeleton in the planner reference
   doc published a `geometry_msgs/PoseStamped`, which dropped most fields.
2. `time_to_strike` is the time *remaining* (`t_strike - latest_sample_time`),
   so it is positive and decreases as required. The skeleton's `time_to_strike`
   property returned the absolute strike time.

The `/poses` `PoseArray` carries no names, so `ball_pose_index` selects the
ball's slot; for a name-keyed setup, replace the `/poses` subscription with a
`/tf` lookup on `ball_rigid_body_name`.

## 8. Calibrate Ball Physics

Do not rely permanently on default drag and restitution constants.

1. Record at least 15 ball trajectories with the mocap system.
2. Include free-flight arcs and table bounces.
3. Segment each trajectory into flight and bounce sections.
4. Fit drag using:

```text
|a - g| = k * |v|^2
```

5. Fit horizontal restitution:

```text
C_h = median(|v_horizontal_after| / |v_horizontal_before|)
```

6. Fit vertical restitution:

```text
C_v = median(|v_z_after| / |v_z_before|)
```

7. Store calibrated values in `hope_planner.yaml`.
8. Re-run planner tests with calibrated values.

Plain meaning of the fitted values:

- `drag_k`: how strongly air drag slows the ball. Larger values make predicted flight shorter.
- `restitution_h`: how much horizontal speed remains after a table bounce.
- `restitution_v`: how much vertical speed remains after a table bounce.

This fitter is implemented as `hope_planner/calibration.py` (`calibrate_ball_physics`,
taken from the planner reference doc) with a CLI. Record the ball with the mocap
system, export each trajectory to a CSV with columns `t,x,y,z` in the HOPE frame
(e.g. from a `ros2 bag` of `/poses` or `/ball/point`), then:

```bash
ros2 run hope_planner hope_calibrate traj1.csv traj2.csv ... traj15.csv
```

Important for this repository version:

- The current `hope_calibrate` command fits only:
  - `drag_k`
  - `restitution_h`
  - `restitution_v`
- It does NOT fit `restitution_racket`.
- `restitution_racket` still needs a separate ball-vs-racket test later, or a temporary default such as `0.88`.

Practical recording flow for this repository version:

1. Record one toss per bag from `/ball/point`.
2. Before recording, confirm `/ball/point` is really publishing and changes when the real ball moves.
2. Convert each bag to one CSV with:

```bash
ros2 run hope_planner hope_bag_to_csv \
  --bag traj01 \
  --topic /ball/point \
  --output traj01.csv
```

3. Then run:

```bash
ros2 run hope_planner hope_calibrate traj01.csv traj02.csv ... traj15.csv
```

Recommended bag-record command:

```bash
ros2 topic echo /ball/point --once
ros2 bag record -s mcap /ball/point -o traj01
```

Important:

- Start the recording only after `/ball/point` is alive.
- After one toss, stop with `Ctrl+C` and wait for rosbag to finish writing metadata.
- If the bag folder contains a `0-byte` `.mcap` file or no `metadata.yaml`, the recording is incomplete; delete that folder and record again.

How to create each calibration CSV:

1. Start mocap and the Avatar-Pro bridge.
2. Record one toss at a time. Each file should contain one continuous ball flight, not a whole practice session.
3. Save the ball center position in the HOPE `world` frame. Units must be seconds and meters.
4. Use this header:

```text
t,x,y,z
```

5. Example rows:

```text
t,x,y,z
0.000000,1.850000,-0.650000,0.420000
0.002778,1.846100,-0.650400,0.418700
0.005556,1.842200,-0.650800,0.417200
```

6. Keep at least 20 percent of the recorded throws as held-out test trajectories. Do not use those files for fitting; use them only to check prediction error.

If you record `/ball/point` with `ros2 bag`, export the bag to CSV with any ROS bag tool you prefer, but keep only these four columns: timestamp seconds, `point.x`, `point.y`, and `point.z`. Rename them to `t,x,y,z` before running `hope_calibrate`.

It prints fitted `drag_k`, `restitution_h`, `restitution_v` ready to paste into
`config/hope_planner.yaml`. Note the three-sample bounce detector is phase-
sensitive: record at the full mocap rate (≥240–360 Hz) so each bounce yields a
clean descend→contact→rise pattern.

Verification gate:

- Predicted hit-plane crossing is close to measured crossing on held-out trajectories.
- Predicted bounce count matches observed bounce count.
- Net-clearance checks match visual inspection or high-speed video.

## 9. Acquire Human Swing Motions

You need at least two reference motions:

1. Forehand.
2. Backhand.

Recommended recording setup:

1. Record from a side view, 90 degrees to the table long axis.
2. Place the camera 3-5 m from the player.
3. Use camera height around 1.0-1.4 m.
4. Use clean background and even lighting.
5. Record at 1080p or better.
6. Record at 30 fps unless you know how to resample higher frame rates correctly.
7. Capture 3-5 seconds per swing.
8. Include ready stance, backswing, strike, follow-through, and recovery.

Do not:

1. Use slow motion without resampling.
2. Record directly from behind or in front.
3. Include multiple people unless you crop or choose the correct tracked person.
4. Use copyrighted broadcast footage without checking the license for your use case.

Install GVHMR:

```bash
cd ~/workspace/HOPE/hope_training
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR
# Follow the GVHMR repository installation instructions.
```

Run GVHMR:

```bash
python tools/demo/demo.py --video=../motions/raw_video/forehand_swing.mp4 -s
python tools/demo/demo.py --video=../motions/raw_video/backhand_swing.mp4 -s
```

Expected output:

```text
GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt
GVHMR/outputs/demo/backhand_swing/hmr4d_results.pt
```

Verification gate:

- GVHMR viewer shows plausible pelvis, torso, shoulder, elbow, and wrist motion.
- The strike timing is visible.
- The motion does not contain severe foot sliding, broken arms, or swapped left/right limbs.

## 10. Retarget Human Motions To The Robot

Install GMR:

```bash
cd ~/workspace/HOPE/hope_training
git clone https://github.com/YanjieZe/GMR.git
cd GMR
pip install -e .
```

Download SMPL-X body models:

1. Go to `https://smpl-x.is.tue.mpg.de/`.
2. Download the required model files.
3. Put them under:

```text
GMR/assets/body_models/smplx/
```

Required files:

```text
SMPLX_NEUTRAL.pkl
SMPLX_FEMALE.pkl
SMPLX_MALE.pkl
```

For the Agibot Expedition A3 robot:

1. Create or obtain a GMR target for the A3, for example `agibot_a3`. If you do not yet have the A3 URDF/MJCF from Agibot, retarget against the open Agibot X1 model (`agibot_x1`) as a stand-in and re-run once the A3 model arrives.
2. Confirm the output joint order matches the A3 controller joint order (the same order the exported ONNX policy will use). Until Agibot provides it, mark the joint order as `PLACEHOLDER_EXTERNAL_A3_JOINT_ORDER` and keep it in one shared config.
3. Retarget forehand.
4. Retarget backhand.
5. Export to CSV or PKL as expected by the preprocessing script.

How to create or verify the GMR target:

1. Locate the robot model file. For A3, this is `PLACEHOLDER_A3_URDF_OR_MJCF` from Agibot. For the temporary stand-in, use the X1 model from `agibot_x1_train/resources/robots/x1`.
2. List the robot joints from the model. For URDF, use `grep '<joint' robot.urdf` or a URDF parser. For MJCF, list the `<joint>` elements.
3. Remove fixed joints from the controllable joint list. Keep only joints that the controller can command.
4. Create one shared file such as `hope_training/config/joint_order_agibot_a3.yaml` with the exact order.
5. Use the same file for GMR retargeting, Isaac/MuJoCo training, ONNX metadata, and the hardware bridge.
6. If the A3 SDK publishes `/joint_states`, compare the topic order and names with the YAML file:

```bash
ros2 topic echo /joint_states --once
```

If the names or order differ, fix the YAML and rebuild/re-export before training. Do not fix joint-order mismatch in only one script.

`PLACEHOLDER_EXTERNAL_A3_JOINT_ORDER` means an external blocker, not an optional cleanup. Replace it only after Agibot provides the A3 joint order or after you verify it from the real SDK.

For a custom robot:

1. Add the robot URDF or MJCF.
2. Define the robot joint list.
3. Define key body matching from SMPL-X joints to robot links.
4. Define joint limits.
5. Define collision or self-contact constraints if supported.
6. Run retargeting.
7. Inspect the output visually.

Important racket note:

- SMPL-X hand pose is not the racket pose.
- The HOPE system uses a fixed physical racket mount.
- The robot racket pose is computed by forward kinematics:

```text
world -> base_link -> waist -> shoulder -> elbow -> wrist -> T_mount -> racket
```

Verification gate:

- Retargeted forehand and backhand play back without impossible joint angles.
- The right wrist trajectory roughly matches the human hitting motion.
- The robot remains balanced in the reference motion.
- Joint limits are not violated.

## 11. Model The Fixed Racket Mount

This step is mandatory. Do not approximate it by tracking the racket externally.

Before you start, prepare these things:

1. A robot model file for simulation:
   - URDF, MJCF, or USD.
   - If the real A3 model is not available yet, use the open Agibot X1 model as a temporary stand-in.
2. The wrist link name in that model:
   - this is the link the racket will attach to.
3. A simple racket model:
   - at minimum, a paddle-face center,
   - a face-normal direction,
   - a radius or width/height,
   - and a simple collision shape.
4. A mounting idea:
   - where the racket center should sit relative to the wrist,
   - and which way the racket face should point.
5. A place to record `T_mount`:
   - translation `x y z` from wrist to racket center,
   - rotation `roll pitch yaw` from wrist axes to racket axes.
6. A first contact guess for simulation:
   - `restitution_racket` for how strong the rebound is,
   - friction if your simulator requires it.

If you do not have the real robot or final bracket yet, that is okay. Start with a temporary racket mount in simulation, make the planner and training pipeline work, then replace `T_mount` with measured values later.

1. Design or obtain the 3D-printed bracket that attaches the racket to the robot wrist.
2. Define a fixed transform `T_mount` from the robot wrist link to the racket frame.
3. Add `T_mount` to the robot model as a fixed joint.
4. Use the same `T_mount` in:
   - simulation,
   - training reward computation,
   - deployment forward kinematics,
   - hardware calibration.
5. Physically measure the mount:
   - translation from wrist frame to racket center,
   - orientation of racket face normal,
   - handle or blade roll convention.
6. Write a small FK test that computes racket pose from known joint angles.

How to measure `T_mount`:

1. Print or assemble the racket bracket.
2. Identify the robot wrist link frame from the URDF. If the wrist frame is not physically obvious, ask Agibot or inspect the URDF link/joint origin.
3. Mark the racket frame:
   - origin: center of the racket face,
   - +X or normal axis: choose the direction the planner uses as the racket face normal,
   - another axis: along the handle or blade top edge, so roll is not ambiguous.
4. Measure translation from wrist frame origin to racket-face center in meters. Use calipers for short offsets or a ruler/tape for larger offsets.
5. Measure orientation:
   - easiest: design the bracket so the racket normal is aligned with a known wrist-link axis, then orientation is a simple 0/90/180 degree rotation;
   - otherwise measure the three angles with a digital angle gauge, or solve the transform by touching known points on the racket with a tracked calibration pointer.
6. Add this as a fixed joint in URDF/MJCF/USD, and also put the same transform in any FK code used by deployment.

Minimum check:

```text
Known robot standing pose + measured T_mount -> computed racket center should match the real racket center within 1-2 cm.
```

Verification gate:

- The simulated racket frame matches the physical mount.
- FK from base_link and joint states returns the expected racket center.
- Racket normal direction is consistent with the planner normal.

## 12. Preprocess Motions For BeyondMimic

For the Agibot Expedition A3 Isaac Lab path (with MuJoCo for sim-to-sim), install BeyondMimic:

```bash
cd ~/workspace/HOPE/hope_training
git clone https://github.com/HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
```

Install Isaac Sim and Isaac Lab:

1. Install Isaac Sim 4.5.0.
2. Install Isaac Lab 2.1.0.
3. Use Python 3.10.
4. Prefer a machine with an NVIDIA RTX 4090 or better.

Add the Agibot A3 robot assets used by training:

```bash
mkdir -p source/whole_body_tracking/whole_body_tracking/assets/agibot_a3

# PLACEHOLDER_A3_ASSET_DIR is the folder containing the robot model and meshes.
# For real A3 hardware, this folder must come from Agibot.
# For temporary software bringup, use the open X1 asset folder instead.
PLACEHOLDER_A3_ASSET_DIR=/absolute/path/to/agibot_a3_assets

cp -r "$PLACEHOLDER_A3_ASSET_DIR"/* \
  source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/
```

Reminder: there is **no public A3 robot description**. Agibot's only open humanoid model is the X1
(`github.com/AgibotTech/agibot_x1_train`, which ships URDF + MJCF + meshes under `resources/robots/x1/`).
Use X1 to build and debug the pipeline, then swap in the real A3 assets from Agibot.

How to fill `PLACEHOLDER_A3_ASSET_DIR`:

1. If Agibot gave you a zip/tar package, extract it under `~/workspace/HOPE/hope_training/vendor_assets/agibot_a3`.
2. The folder should contain at least one robot model file (`.urdf`, `.mjcf`, `.xml`, or `.usd`) and mesh files (`.stl`, `.dae`, `.obj`, or `.usd`).
3. Use an absolute path, for example:

```bash
PLACEHOLDER_A3_ASSET_DIR=$HOME/workspace/HOPE/hope_training/vendor_assets/agibot_a3
```

4. Verify the folder before copying:

```bash
find "$PLACEHOLDER_A3_ASSET_DIR" -maxdepth 3 -type f | sed -n '1,40p'
```

If the folder has only meshes but no robot model, or only a robot model but no meshes, it is incomplete. Ask Agibot for the complete asset package.

After copying the assets:

1. Register the robot asset path in the training config.
2. Verify joint names and order against the real A3 controller joint order (the same order used for ONNX export and deployment).
3. Verify inertial values, actuator limits, and default PD gains.
4. Add the fixed racket mount link or fixed joint used by `T_mount`.

Install the training package:

```bash
python -m pip install -e source/whole_body_tracking
```

Set up WandB:

1. Create a WandB account.
2. Create or choose a WandB organization.
3. Create a Registry collection named `Motions`.
4. Set the entity:

```bash
export WANDB_ENTITY=PLACEHOLDER_WANDB_ENTITY
```

How to get `PLACEHOLDER_WANDB_ENTITY`:

1. Open WandB in the browser.
2. Look at the workspace URL. In `https://wandb.ai/my-lab-name/...`, the entity is `my-lab-name`.
3. Put that exact value into `WANDB_ENTITY`.
4. Confirm login:

```bash
wandb login
wandb whoami
```

Some BeyondMimic/WandB registry paths include `-org` after the entity. Use the exact registry name shown by the WandB UI after upload. In the examples below, replace `PLACEHOLDER_WANDB_ENTITY-org` with that registry owner string.

Convert retargeted motions:

```bash
python scripts/csv_to_npz.py \
  --input_file ~/workspace/HOPE/hope_training/motions/retargeted/forehand_swing.csv \
  --input_fps 30 \
  --output_name hope_forehand \
  --headless

python scripts/csv_to_npz.py \
  --input_file ~/workspace/HOPE/hope_training/motions/retargeted/backhand_swing.csv \
  --input_fps 30 \
  --output_name hope_backhand \
  --headless
```

Replay preprocessed motions:

```bash
python scripts/replay_npz.py \
  --registry_name=PLACEHOLDER_WANDB_ENTITY-org/wandb-registry-motions/hope_forehand

python scripts/replay_npz.py \
  --registry_name=PLACEHOLDER_WANDB_ENTITY-org/wandb-registry-motions/hope_backhand
```

Verification gate:

- The Agibot A3 (or X1 stand-in) model replays each motion.
- The wrist and racket mount follow a plausible stroke.
- Feet, pelvis, torso, and arms do not jitter badly.
- WandB artifacts are registered and accessible.

## 13. Implement HOPE-Specific WBC Training Extensions

The base BeyondMimic policy tracks a reference motion. HOPE needs more than motion tracking: it must hit a commanded racket target at the strike time.

Add observations to the training environment:

```text
desired racket position relative to base: 3 dims
desired racket velocity in world frame: 3 dims
desired racket face normal: 3 dims
time remaining until strike: 1 dim
desired base XY position in world frame: 2 dims
swing type, forehand or backhand: 1 dim
```

Add reward terms:

1. Imitation reward.
   - Tracks human-like upper-body swing style.
   - Active throughout the episode.

2. Base position reward.
   - Encourages the robot to step to a useful base position.
   - Active before strike time.

3. Racket target reward.
   - Tracks racket position, velocity, and normal.
   - Active near strike time, for example inside a 0.2 s window.
   - Compute actual racket state in simulation by FK through `T_mount`.

4. Regularization reward.
   - Penalizes excessive torque, contact force, action changes, and unstable motion.

Add domain randomization:

```text
PD gains
link mass
friction
external pushes
observation noise
motor strength
racket target position
base starting lateral offset
```

Episode design:

1. Initialize the robot in a standing pose.
2. Randomize the robot lateral offset.
3. Sample a reachable racket target.
4. Select forehand or backhand based on target Y relative to robot center.
5. Load the matching reference motion.
6. Give the policy target observations.
7. Reward base repositioning before strike.
8. Reward racket target tracking near strike.
9. Continue briefly after strike for balance recovery.
10. Reset on fall or episode end.

How to define "reachable racket target":

1. Start with the real or simulated robot in the ready stance.
2. Sample candidate racket positions near the expected hit plane, for example around `x = 0.0` in the HOPE world frame.
3. Use inverse kinematics or dense FK sampling to check whether the racket center can reach the candidate without violating joint limits.
4. Reject targets that require foot penetration, self-collision, table collision, or a racket normal outside the wrist range.
5. Save the accepted target bounds in the training config, for example:

```text
racket_target_x_range: [PLACEHOLDER_MIN_REACHABLE_X, PLACEHOLDER_MAX_REACHABLE_X]
racket_target_y_range: [PLACEHOLDER_MIN_REACHABLE_Y, PLACEHOLDER_MAX_REACHABLE_Y]
racket_target_z_range: [PLACEHOLDER_MIN_REACHABLE_Z, PLACEHOLDER_MAX_REACHABLE_Z]
```

Measure these ranges in simulation first. Confirm slowly on hardware only after the no-ball dry test passes.

Verification gate:

- The environment runs headless.
- Observation dimensions match the policy config.
- Reward terms are finite.
- FK-computed racket pose is correct.
- Random sampled targets are reachable.

## 14. Train The WBC Policy

Baseline motion tracking command:

```bash
python scripts/rsl_rl/train.py \
  --task=PLACEHOLDER_TRACKING_TASK_NAME \
  --registry_name PLACEHOLDER_WANDB_ENTITY-org/wandb-registry-motions/hope_forehand \
  --headless \
  --logger wandb \
  --log_project_name hope_wbc \
  --run_name forehand_tracking
```

HOPE target-tracking command:

```bash
python scripts/rsl_rl/train.py \
  --task=PLACEHOLDER_HOPE_TASK_NAME \
  --registry_name PLACEHOLDER_WANDB_ENTITY-org/wandb-registry-motions/hope_forehand \
  --headless \
  --logger wandb \
  --log_project_name hope_wbc \
  --run_name hope_forehand_racket_tracking
```

Train separate policies first:

1. Train forehand.
2. Evaluate forehand.
3. Train backhand.
4. Evaluate backhand.
5. Implement runtime switching later.

Expected HITTER/BeyondMimic settings:

```text
Algorithm: PPO
Actor: MLP [512, 256, 128]
Critic: MLP [512, 256, 128]
Control frequency: 50 Hz
Parallel environments: around 4096
```

Check the current upstream config before training:

```bash
cat source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/agents/rsl_rl_ppo_cfg.py
```

How to fill the training placeholders:

1. `PLACEHOLDER_TRACKING_TASK_NAME` is the task name registered for plain motion tracking. If you follow the naming in this guide, it will be `Tracking-Flat-AgibotA3-v0`, but it only works after you register that task in the training code.
2. `PLACEHOLDER_HOPE_TASK_NAME` is the task name registered for the HOPE racket-target environment. If you follow this guide, use `HOPE-PingPong-AgibotA3-v0`.
3. To find registered task names, search the training repo:

```bash
rg -n "AgibotA3|HOPE-PingPong|Tracking-Flat|gym.register|register" source scripts
```

4. `PLACEHOLDER_WANDB_ENTITY-org/wandb-registry-motions/hope_forehand` must match the registry path printed by the upload command or shown in the WandB UI.

Evaluate:

```bash
python scripts/rsl_rl/play.py \
  --task=PLACEHOLDER_HOPE_TASK_NAME \
  --num_envs=2 \
  --wandb_path=PLACEHOLDER_WANDB_ENTITY/hope_wbc/PLACEHOLDER_RUN_ID
```

`PLACEHOLDER_RUN_ID` is the run ID from the WandB run URL. Example: in `https://wandb.ai/my-lab/hope_wbc/runs/abc123`, the run ID is `abc123`.

Target metrics:

```text
Tracking success rate: > 90 percent
Racket position error at strike: < 7.5 cm
Racket velocity error at strike: < 0.5 m/s
Racket normal error at strike: < 15 degrees
Base repositioning time: < 0.8 s
```

Verification gate:

- Policy does not fall during evaluation.
- Racket reaches the target near strike time.
- Forehand and backhand policies both work.
- Recovery after swing is stable.

## 15. Export The Policy To ONNX

Clone deployment code:

```bash
cd ~/workspace/HOPE/hope_ws/src
git clone https://github.com/HybridRobotics/motion_tracking_controller.git
```

Export ONNX:

```bash
cd ~/workspace/HOPE/hope_ws/src/motion_tracking_controller
python scripts/export_onnx.py \
  --wandb_path=PLACEHOLDER_WANDB_ENTITY/hope_wbc/PLACEHOLDER_RUN_ID \
  --output_path=~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx
```

Export both policies:

```bash
python scripts/export_onnx.py \
  --wandb_path=PLACEHOLDER_WANDB_ENTITY/hope_wbc/PLACEHOLDER_FOREHAND_RUN_ID \
  --output_path=~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx

python scripts/export_onnx.py \
  --wandb_path=PLACEHOLDER_WANDB_ENTITY/hope_wbc/PLACEHOLDER_BACKHAND_RUN_ID \
  --output_path=~/workspace/HOPE/hope_training/policies/hope_backhand_policy.onnx
```

Run ID lookup:

1. Open the successful training run in WandB.
2. Copy the run ID from the URL or from the run overview page.
3. Use the forehand run ID for `hope_forehand_policy.onnx` and the backhand run ID for `hope_backhand_policy.onnx`.
4. After export, keep the ONNX file and the exact joint-order YAML together. They are a matched pair.

The ONNX should include:

1. Actor network.
2. Observation normalization metadata.
3. Joint names.
4. Joint order.
5. PD gains.
6. Action scale.
7. Reference motion metadata.

Verification gate:

- ONNX export completes.
- The ONNX model loads in ONNX Runtime.
- Metadata contains the expected joint order.

## 16. Run Sim-To-Sim Verification

Before touching hardware, verify the ONNX policy in simulation.

```bash
ros2 launch motion_tracking_controller mujoco.launch.py \
  policy_path:=~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx
```

Then test backhand:

```bash
ros2 launch motion_tracking_controller mujoco.launch.py \
  policy_path:=~/workspace/HOPE/hope_training/policies/hope_backhand_policy.onnx
```

Check:

1. The policy loads.
2. Joint order is correct.
3. PD gains are reasonable.
4. The robot stays upright.
5. The swing resembles training.
6. The racket mount moves as expected.

Verification gate:

- Sim-to-sim behavior matches training playback.
- No hardware deployment until this passes.

## 17. Implement Runtime Forehand/Backhand Switching

The reference deployment code may load one ONNX policy. HOPE needs forehand and backhand switching unless you train a single multi-skill policy.

Implement:

1. Load both ONNX sessions at startup.
2. Subscribe to `/racket/command`.
3. Compute ball or target Y relative to robot center.
4. Select forehand or backhand:
   - Use the sign convention from your training environment.
   - The HITTER paper uses ball Y relative to the robot center.
5. Build the observation vector exactly as during training.
6. Run the selected ONNX session.
7. Publish joint position commands.
8. Keep policy switching deterministic and logged.

How to compute the side for switching:

1. Use the same sign convention used during training.
2. In the HOPE world frame, get the commanded racket target Y from `/racket/command.position.y`.
3. Get the robot center Y from `P1_base_link` or the base pose used by training.
4. Compute:

```text
relative_y = racket_command.position.y - p1_base_link_world_y
```

5. If your training labels positive `relative_y` as forehand, use that. If your training labels it as backhand, invert the rule. The important part is consistency, not the sign name.
6. Log the values on every switch:

```text
relative_y, selected_policy, command_time, time_to_strike
```

Important:

- Do not change observation ordering after export.
- Do not change joint ordering after export.
- Do not switch policies in the middle of a strike unless your training supports that.

Verification gate:

- A target on the forehand side selects the forehand model.
- A target on the backhand side selects the backhand model.
- Switching does not cause a discontinuous joint command spike.

## 18. Deploy On The Agibot Expedition A3

Use this path for the A3. Keep the HOPE ROS 2 interfaces stable and put all Agibot-specific logic (AimDK/AimRT) behind a bridge package so the rest of the stack does not depend on the vendor API.

Study Agibot's own deployment pattern first. Agibot's open X1 inference stack is the closest public reference for how an Agibot humanoid runs a learned policy:

- `github.com/AgibotTech/agibot_x1_infer` — C++ deployment node built on **AimRT** middleware, **ROS 2 Humble**, running the policy via **ONNX Runtime**. It subscribes `/joint_states`, publishes `/joint_cmd`, runs the **PD loop at ~1 kHz** and the **policy at ~100 Hz** (decimation 10), and is driven by a state machine (`pd_idle`, `pd_stand`, `rl_walk_*`, ...). HOPE's WBC policy runs at **50 Hz**, so adjust the decimation accordingly.
- `github.com/AgibotTech/agibot_x1_train` — the matching training repo (Isaac Gym + MuJoCo sim2sim; PT→ONNX/JIT export).
- `x2-aimdk.agibot.com` — the published **AimDK_X2** developer SDK docs (ROS 2 interfaces, robot specs). Assume the A3 has an analogous SDK and confirm with Agibot.

Caveat: the X1 stack is **locomotion-only** and uses **ROS 2 Humble**, while HOPE targets ROS 2 Jazzy and needs whole-body motion tracking. Reuse its **plumbing** (AimRT + ONNX Runtime, `/joint_states`/`/joint_cmd`, the 1 kHz-PD/decimated-policy split, the state machine), not its walking algorithm.

Create or add Agibot deployment packages:

```bash
cd ~/workspace/HOPE/hope_ws/src

# If Agibot provides an A3 ROS 2 / AimDK bridge or SDK wrapper, clone it here.
# Otherwise create a thin bridge package and keep the AimDK/AimRT low-level API isolated.
ros2 pkg create agibot_hardware_bridge --build-type ament_cmake
ros2 pkg create agibot_bringup --build-type ament_python

git clone https://github.com/HybridRobotics/motion_tracking_controller.git
```

The Agibot A3 bridge must provide:

1. Joint encoder feedback as `sensor_msgs/msg/JointState` (mapped from the AimDK/AimRT joint feedback).
2. A safe command path from policy action to A3 joint position targets (HOPE action → `/joint_cmd` via AimDK).
3. The A3 joint name list and joint order used by the ONNX policy (obtain from Agibot — see step 2.1).
4. PD gains or impedance settings for each controlled joint.
5. A hard stop, soft stop, and standby mode (wired to the A3 hardware E-stop and AimDK soft-stop).
6. Network setup for the A3 controller (AimRT/DDS discovery).
7. Launch files that start the bridge, safety monitor, and WBC controller in a predictable order.

Bridge placeholders and how to replace them:

| Placeholder | Meaning | How to get it |
| --- | --- | --- |
| `PLACEHOLDER_A3_ROBOT_IP` | The robot controller IPv4 address. | Read the Agibot network setup page or the robot controller UI. If DHCP is used, ask the router or run `arp -a` after connecting the robot. |
| `PLACEHOLDER_A3_NETWORK_INTERFACE` | The Linux network device connected to the robot, such as `enp4s0` or `eth0`. | Run `ip -br addr` on the control PC and choose the wired interface on the robot subnet. |
| `PLACEHOLDER_A3_JOINT_ORDER_FILE` | YAML file listing commandable joints in controller order. | Create it from the A3 SDK/URDF and verify against `/joint_states`. |
| `PLACEHOLDER_A3_PD_GAINS_FILE` | Safe startup PD or impedance values. | Get defaults from Agibot, then tune only after no-ball tests. |
| `PLACEHOLDER_A3_STOP_SERVICE_OR_TOPIC` | Software stop command. | Read the A3 SDK/AimDK docs or Agibot manual; verify with the robot supported off the ground or in low-power mode. |

Install dependencies and build:

```bash
cd ~/workspace/HOPE/hope_ws
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  --packages-up-to agibot_hardware_bridge agibot_bringup motion_tracking_controller

source install/setup.bash
```

Connect to the Agibot A3:

1. Connect the control PC to the A3 robot network using the Agibot hardware guide.
2. Set the static IP, subnet, and firewall rules required by the AimDK SDK.
3. Identify the network interface:

```bash
ip -br addr
```

4. Find or set the robot IP:

```bash
# Show neighbors on the local Ethernet network after the robot is connected.
ip neigh

# Or, if your network uses ARP tools:
arp -a
```

5. Confirm that the A3 controller is reachable:

```bash
ping PLACEHOLDER_A3_ROBOT_IP
```

Run real control only after sim-to-sim passes:

```bash
ros2 launch agibot_bringup agibot_a3.launch.py \
  robot_ip:=PLACEHOLDER_A3_ROBOT_IP \
  network_interface:=PLACEHOLDER_A3_NETWORK_INTERFACE

ros2 launch motion_tracking_controller real.launch.py \
  policy_path:=~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx
```

Record the A3 safety controls before any active swing test. Fill every line from the Agibot manual or SDK docs; do not leave a line blank:

```text
Standby command or button: PLACEHOLDER_A3_STANDBY_METHOD
Activate WBC command: PLACEHOLDER_A3_WBC_ACTIVATE_METHOD
Soft stop service/topic/button: PLACEHOLDER_A3_SOFT_STOP_METHOD
Hard E-stop physical button location: PLACEHOLDER_A3_ESTOP_LOCATION
Power cut method: PLACEHOLDER_A3_POWER_CUT_METHOD
Recovery procedure after stop: PLACEHOLDER_A3_RECOVERY_STEPS
```

How to measure E-stop response:

1. Start logging joint states and the stop command timestamp.
2. Record video at 120 fps or faster if available.
3. Trigger the E-stop while the robot is moving slowly in a safe test.
4. Measure time from the stop trigger to the time upper-body and gait motion stops.
5. The competition requirement is below `200 ms`; if your measured value is higher, fix the safety path before continuing.

Verification gate:

- The A3 enters standby safely.
- The A3 emergency stop works from hardware and software, and stops upper-body and gait motion within 200 ms (competition requirement).
- Joint states are streaming with the expected names and order.
- The bridge rejects commands outside joint, velocity, torque, and workspace limits.
- The policy can be activated in a controlled, no-ball test.

## 19. Port To Another Humanoid

Use this path only after the Agibot A3 deployment path works or if the target robot changes.

Recommended path:

1. Keep the HOPE planner in ROS 2.
2. Bridge `/racket/command`, `/P1/pose`, and joint feedback into the vendor middleware.
3. Run ONNX inference in a controller adapted to the target joint names and joint order.
4. Send joint position commands through the target robot's actuator bus.
5. Keep safety interlocks outside the learned policy.

Porting tasks:

1. Confirm `base_link` convention.
2. Convert or obtain the robot MJCF, URDF, or USD model.
3. Add the robot to GMR.
4. Add the robot to the chosen simulation backend.
5. Define joint names and order.
6. Define PD gains or impedance settings.
7. Define `T_mount`.
8. Adapt the ONNX observation builder.
9. Adapt the action-to-joint-command mapping.
10. Add vendor middleware or ROS 2 bridge launch files.

Verification gate:

- The new robot sim policy works before real hardware.
- Bridge latency is measured.
- E-stop works through the target robot safety stack.

## 20. Integrate The Full ROS 2 Pipeline

Launch order:

```bash
# Terminal 1: motion capture bringup (Avatar Pro VRPN client + relay + world frame)
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py \
  server:=PLACEHOLDER_AVATAR_PRO_PC_IP port:=3883 update_freq:=360.0

# Terminal 2: planner
ros2 launch hope_planner hope_planner.launch.py

# Terminal 3: WBC controller
ros2 launch motion_tracking_controller real.launch.py \
  network_interface:=PLACEHOLDER_A3_NETWORK_INTERFACE \
  policy_path:=~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx

# Terminal 4: monitoring
ros2 topic hz /poses
ros2 topic hz /tf
ros2 topic hz /racket/command
ros2 topic hz /joint_states
```

Use `PLACEHOLDER_AVATAR_PRO_PC_IP` from Step 6 (the Avatar Pro / CMTracker PC IP). Use `PLACEHOLDER_A3_NETWORK_INTERFACE` from Step 18.

Expected topic map:

```text
/poses             mocap -> planner/WBC                 120-360 Hz
/tf                mocap -> planner/WBC                 120-360 Hz
/ball/point        mocap -> planner                     optional
/P1/pose           mocap -> planner/WBC                 optional
/P2/pose           mocap -> opponent                    optional
/table/pose        mocap -> drift monitor               optional
/racket/command    planner -> WBC                       50 Hz or higher
/joint_states      robot -> WBC                         around 500 Hz
```

Use low-latency QoS:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

mocap_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
```

Verification gate:

- Mocap topics are live.
- Planner topics are live.
- WBC receives base_link, joint states, and racket commands.
- Total perception-to-actuation latency is below 20 ms.

How to measure total latency:

1. Time-stamp the incoming ball message with the mocap timestamp.
2. Time-stamp the outgoing joint command right before it is sent to the robot bridge.
3. Subtract: `latency = joint_command_send_time - mocap_sample_time`.
4. Log at least 200 samples during soft tosses and report median and 95th percentile.
5. If 95th percentile is above `20 ms`, check network, CPU load, ROS QoS, and whether any node is using Reliable QoS on high-rate mocap data.

## 21. Add Monitoring And Safety Nodes

Implement `hope_monitoring` with:

1. Mocap health monitor.
   - Warn if `/poses` or `/tf` drops below required rate.
   - Warn if the ball disappears for 3 or more consecutive frames.

2. Table drift monitor.
   - Warn if PPT pose deviates from identity beyond tolerance.

3. Robot workspace monitor.
   - Warn if robot base_link leaves allowed side.
   - For exhibition mode, enforce virtual safety wall at `x = 1.37`.

4. Racket speed monitor.
   - Use FK and joint states to estimate racket-grip speed.
   - For human exhibition, keep peak resultant speed below 6 m/s.

5. E-stop monitor.
   - Verify stop response is below 200 ms when tested.

6. Diagnostics publisher.
   - Publish `diagnostic_msgs/DiagnosticArray`.

Suggested starting thresholds:

```text
mocap_min_rate_hz: 120
mocap_preferred_rate_hz: 240
ball_missing_warn_frames: 3
table_translation_drift_warn_m: 0.01
table_rotation_drift_warn_rad: 0.02
base_allowed_x_min_m: -1.5
base_allowed_x_max_m: 1.37
racket_speed_exhibition_limit_mps: 6.0
estop_required_stop_time_s: 0.200
```

How to tune thresholds:

1. Record 30 seconds with the table and robot completely still.
2. Compute normal noise for `PPT`, `P1`, and ball position.
3. Set warning thresholds at least 3 times larger than normal still-scene noise.
4. Keep the E-stop threshold fixed at `0.200 s`; that comes from the competition rule, not from tuning.

Verification gate:

- Unplugging or stopping mocap causes a visible warning.
- Moving PPT causes a table drift warning.
- Crossing the safety wall triggers the intended safety response.
- E-stop behavior is measured, not assumed.

## 22. Run A No-Ball Dry Test

Before any ball is tossed:

1. Power the robot.
2. Start mocap.
3. Start planner.
4. Start WBC in standby.
5. Confirm `/tf` includes PPT and P1.
6. Confirm `/joint_states` is active.
7. Confirm FK-computed racket pose is reasonable.
8. Activate policy with no ball.
9. Confirm robot remains stable.
10. Press E-stop.
11. Confirm robot stops within the required time.

Verification gate:

- Robot is stable in standby and active modes.
- E-stop is verified.
- No unexpected arm swing occurs without a valid command.

## 23. Run A Soft Toss Test

Use a slow, controlled ball toss.

1. Keep all people outside the robot arm reach envelope.
2. Start recording logs.
3. Toss the ball gently onto the table.
4. Watch `/racket/command`.
5. Do not activate full-speed striking yet.
6. Confirm the planner predicts the incoming ball.
7. Confirm the WBC receives the command.
8. Confirm the selected forehand/backhand policy is correct.
9. Activate low-gain or reduced-speed mode if available.
10. Stop immediately if the robot behaves unexpectedly.

Verification gate:

- Planner command matches the observed ball.
- Strike time is plausible.
- Selected policy is correct.
- Robot remains balanced.

## 24. Tune Sim-To-Real Details

Common symptoms and fixes:

```text
Robot falls immediately:
  Check inertial parameters, base_link offset, initial pose, PD gains.

Arm oscillates:
  Lower Kp or increase Kd.

Arm is too slow:
  Increase Kp gradually.

Racket misses despite good command:
  Re-measure T_mount and check FK.

Forehand/backhand wrong:
  Check Y-axis convention and robot center offset.

Planner predicts wrong bounce:
  Recalibrate drag and restitution.

Latency feels too high:
  Check QoS, CPU load, ROS 2 topic rates, and network.
```

PD gain tuning procedure:

1. Start with simulation-trained gains.
2. For hardware, begin around 70 percent of simulation gains.
3. Increase gradually.
4. Watch torque, position error, and oscillation.
5. Change one joint group at a time.
6. Retest in sim-to-sim after major config changes.

Verification gate:

- Robot can repeatedly swing without falling.
- Racket reaches a static or slow target.
- No joint oscillation or buzzing is present.

## 25. Prepare For Competition Rules

Competition requirements that affect implementation:

1. Robot must be humanoid.
2. Robot height must be between 1.0 m and 1.9 m.
3. Robot mass must not exceed 80 kg.
4. Robot must run autonomously during match play.
5. External power is prohibited during a match.
6. Remote operation and remote brain are prohibited.
7. Robot may communicate only through the organizer-provided network with the HOPE mocap/referee systems.
8. Racket, grip, hand, and wrist must not have mocap markers or active tracking devices.
9. The robot must have a clearly marked emergency stop.
10. E-stop must stop upper-body and gait motion within 200 ms during the official test.
11. The robot must subscribe to and parse the HOPE standard ROS 2 topics.
12. For the qualification video, the robot must demonstrate:
    - independent standing and ready stance,
    - 3 consecutive stable serves,
    - at least 5 consecutive rally strokes with a partner,
    - live subscription logs for HOPE mocap topics,
    - safe recovery or removal after E-stop or fall.

Qualification video deadline from the English rules:

```text
2026-07-31 23:59 Beijing time
```

Video requirements:

```text
Resolution: at least 1920x1080
Frame rate: at least 30 fps
Duration: no more than 10 minutes
Note: no more than 1 page
```

Verification gate:

- You can produce the qualification video without exposing private model weights, source code, URDF, training data, or internal algorithms.

## 26. Final End-To-End Test

Run this complete sequence:

1. Start mocap.
2. Confirm PPT, P1, P2, and ball tracking.
3. Start planner.
4. Confirm `/racket/command`.
5. Start WBC in standby.
6. Confirm joint states and base_link pose.
7. Confirm FK racket pose.
8. Run no-ball active test.
9. Run soft toss.
10. Run controlled rally with a human or robot partner.
11. Record logs.
12. Review misses and classify the cause:
    - mocap,
    - planner,
    - retargeting,
    - WBC policy,
    - deployment control,
    - mechanical mount,
    - latency.
13. Fix one cause at a time.
14. Repeat until the robot can rally consistently.

Minimum success criteria:

```text
Mocap publishes at required rate.
Planner publishes valid commands for incoming balls.
WBC receives commands and joint states.
Robot stays balanced.
Racket reaches commanded strike region.
E-stop works.
No racket markers are used.
```

## 27. Suggested Implementation Order Summary

Use this as the master checklist:

1. Read all docs.
2. Choose robot and backend.
3. Create workspace.
4. Implement `hope_msgs`.
5. Configure HOPE world frame.
6. Set up mocap.
7. Verify mocap topics.
8. Implement planner constants.
9. Implement ball state estimator.
10. Test state estimator.
11. Implement trajectory predictor.
12. Test trajectory predictor.
13. Implement racket target planner.
14. Test racket target planner.
15. Implement planner ROS 2 node.
16. Test planner with recorded or live ball data.
17. Calibrate ball physics.
18. Record forehand and backhand videos.
19. Run GVHMR.
20. Verify SMPL-X output.
21. Install GMR.
22. Retarget motions to robot.
23. Verify retargeted motions.
24. Model `T_mount`.
25. Verify FK to racket frame.
26. Install Isaac Lab or mjlab.
27. Install BeyondMimic training stack.
28. Preprocess motions.
29. Replay preprocessed motions.
30. Implement HOPE-specific WBC observations.
31. Implement HOPE-specific WBC rewards.
32. Add domain randomization.
33. Train forehand policy.
34. Evaluate forehand policy.
35. Train backhand policy.
36. Evaluate backhand policy.
37. Export ONNX.
38. Run sim-to-sim verification.
39. Implement forehand/backhand runtime switching.
40. Set up hardware deployment.
41. Test standby and E-stop.
42. Test no-ball active mode.
43. Test soft toss.
44. Tune PD gains and `T_mount`.
45. Add monitoring and safety nodes.
46. Run end-to-end rally.
47. Record qualification video.

## 28. Do Not Violate These Constraints

1. Do not add motion-capture markers to the racket, grip, hand, or wrist.
2. Do not use measured racket pose as WBC feedback.
3. Do not mix table-center and HOPE corner-origin coordinates without explicit transforms.
4. Do not deploy to hardware before sim-to-sim verification.
5. Do not activate full-speed swings before E-stop testing.
6. Do not assume Agibot robot assets from different sources are interchangeable (the X1 model is NOT the A3 model).
7. Do not assume any A3-specific detail (joint order, DOF, PD gains, control API, ROS 2 distro) until you confirm it from Agibot documentation or hardware. The public sources cover X1/X2, not the A3.
8. Do not change observation or joint ordering after ONNX export.
9. Do not use Reliable QoS for high-rate real-time mocap topics unless you have measured that latency remains acceptable.
10. Do not tune multiple sim-to-real variables at once.

## 29. References In This Repository

- `README.md`
- `HOPE_AI_Challenge_2026_Rules_EN.docx`
- `mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`
- `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`
- `HOPE_WBC_Simulation_Training_Reference_Setup.md`
- `HOPE_Hardware_Deployment_Reference_Setup.md`

External Agibot / Expedition A3 references (verified June 2026):

- Agibot Expedition A3 product page: `https://www.agibot.com/` (product announcement; 1.73 m, 55 kg).
- AimRT middleware (Agibot's open-source C++20 runtime, ROS 2 interop): `https://github.com/AimRT/AimRT`.
- AimDK_X2 developer SDK docs (closest published Agibot humanoid SDK; ROS 2 interfaces, robot specs): `https://x2-aimdk.agibot.com/en/latest/`.
- Agibot X1 training stack (only open Agibot humanoid model; URDF/MJCF/meshes, Isaac Gym + MuJoCo, PPO, ONNX export): `https://github.com/AgibotTech/agibot_x1_train`.
- Agibot X1 inference/deployment stack (AimRT + ROS 2 Humble + ONNX Runtime; `/joint_states` → `/joint_cmd`, 1 kHz PD / 100 Hz policy): `https://github.com/AgibotTech/agibot_x1_infer`.
- Agibot OmniHand O10 dexterous hand (optional; SDK `github.com/AgibotTech/agillink_omnihand_sdk`).
- Third-party ROS 2 robot_description packaging template (Unitree G1 + Agibot X2 URDF/MJCF; not the A3): `https://github.com/ioai-tech/robot_description`.

Note: at the time of writing there is **no public robot description, SDK, or joint specification for the Expedition A3 itself**. The A3-specific URDF/MJCF, joint order, actuator parameters, PD gains, and control/safety API must be obtained directly from Agibot (see step 2.1).

## 30. Values You Must Fill In (TODO) And How To Get Them

Everything below is a placeholder in the committed code/config. Each row says
where the value lives, what is there now, and exactly how to obtain the real one.
Search the repo for `TODO` to find these inline. Nothing here can be guessed from
a laptop — each value comes from the arena network, a physical measurement, a
calibration run, or Agibot.

### 30.1 Mocap network and stream (do at the arena)

| Value | Where | Now | How to get it |
|-------|-------|-----|---------------|
| Avatar Pro server IP | `avatar_pro_hope_bridge.launch.py` arg `server` | `192.168.1.100` | Run `ipconfig` on the Avatar-Pro / CMTracker PC; use the wired-adapter IPv4 on the robot LAN. Confirm with `ping <ip>` from the ROS host. |
| VRPN port | launch arg `port` | `3883` | Chingmu/Avatar-Pro default is 3883; confirm in the CMTracker streaming settings. |
| Camera / poll rate | launch arg `update_freq` | `360.0` | Set to the mocap system's frame rate (≥240–360 Hz for the ball). |
| Object names | `avatar_pro_vrpn.yaml` (`ppt_object`, `p1_object`, `p2_object`, `ball_object`) | `PPT`/`P1`/`P2`/`ball_object=""` | `PPT`/`P1`/`P2` must match the rigid-body labels in CMTracker exactly. Leave `ball_object` empty unless you want to pin the ball to one confirmed VRPN sender id instead of using auto-detect. |
| Stream frame is REP-103 Z-up | mocap software | assumed | Set the mocap Up Axis to Z and calibrate the origin at the P1 near-side left corner. If it can only stream Y-up, add the fixed rotation in the relay (mocap doc §6.5.3). |

### 30.2 World frame ↔ robot (physical measurement)

| Value | Where | Now | How to get it |
|-------|-------|-----|---------------|
| `mocap_to_base_link` P1/P2 `xyz` (m), `rpy` (rad) | `hope_bringup/config/hope_world_frame.yaml` | all `0.0` | Measure the rigid transform from each robot's marker-cluster frame to its URDF `base_link` (CAD or tape-measure + level). Verify: stand the robot still and compare FK-predicted `base_link` to the mocap reading; adjust until they agree. |

### 30.3 Ball physics (calibration run)

| Value | Where | Now | How to get it |
|-------|-------|-----|---------------|
| `drag_k`, `restitution_h`, `restitution_v`, `restitution_racket` | `hope_planner/config/hope_planner.yaml` | HITTER defaults | Record ≥15 ball trajectories with the mocap, export each to CSV (`t,x,y,z`), run `ros2 run hope_planner hope_calibrate traj*.csv`, paste the printed values (Section 8). `restitution_racket` is not a pose and not a marker-tracking result. It is the bounce-strength number for ball vs racket contact. Easiest first measurement: keep the racket still, measure ball speed before and after impact, and estimate `restitution_racket ~= rebound_speed / incoming_speed`. |
| `ball_pose_index` | `hope_planner/config/hope_planner.yaml` | `0` | With the avatar_pro relay it is 0 (ball is first in `pose_array_order`). Confirm with `ros2 topic echo /poses --once` and count the ball's slot. |
| `x_hit`, `delta_t_flight` | `hope_planner/config/hope_planner.yaml` | `0.0`, `0.5` | Tune to your robot's reach and preferred return arc; start with the defaults and adjust during soft-toss tests (Section 23). |

### 30.4 Agibot Expedition A3 robot (request from Agibot — see step 2.1)

| Value | Where it will be used | How to get it |
|-------|----------------------|---------------|
| A3 URDF/MJCF/USD + meshes | GMR retarget, sim training, FK | Request from Agibot (no public A3 model). Develop against the open X1 model meanwhile. |
| Joint name list + joint order | planner→WBC, ONNX export, bridge | From the A3 SDK/URDF. Must be identical everywhere (training, export, deploy). |
| Joint limits, link inertials, gear ratios | sim fidelity, safety limits | From Agibot datasheet/URDF. |
| Default PD gains / impedance | WBC training + `agibot_hardware_bridge` | From Agibot; start hardware at ~70% of sim gains (Section 24). |
| `base_link` name + standing height | world-frame FK chain | From the A3 URDF. |
| Control SDK / AimDK / ROS 2 API, joint command + feedback topics | `agibot_hardware_bridge` | From Agibot (analogous to AimDK_X2). Confirm ROS 2 distro (X1 uses Humble; HOPE targets Jazzy). |
| E-stop / soft-stop / standby interface | safety, `agibot_bringup` | From the Agibot manual; verify the 200 ms stop requirement. |

### 30.5 Racket mount and WBC/deploy (later phases)

| Value | Where | How to get it |
|-------|-------|---------------|
| `T_mount` (wrist→racket fixed transform) | robot model + FK + reward (step 11) | Physically measure the 3D-printed bracket (translation to racket center, face-normal orientation, blade roll); add as a fixed joint. |
| WandB entity / run ids | training + ONNX export (steps 12–15) | Your own WandB org/run after training. |
| ONNX policy paths | sim2sim + deploy (steps 16–18) | Output of `export_onnx.py` after a successful training run. |
| Robot IP + network interface | deploy (step 18) | From the Agibot network setup; `ip addr` on the control PC. |
