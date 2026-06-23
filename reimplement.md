# HOPE Reimplementation Guide

This guide explains how to reimplement the HOPE reference system from the documentation in this repository. The repo is a design-document repository, not a complete runnable software stack. A successful reimplementation means you will create your own ROS 2 packages, training configuration, policy export, and deployment integration while following the reference architecture and competition constraints.

Read this file from top to bottom. Do not skip the verification gates. Each later phase assumes the earlier phase is already working.

## Guide Structure By Phase

Use the steps in order, but think of them as eight larger workstreams:

1. **Phase 1 — Scope, Platform, and Workspace Foundation**: steps `0-5`
2. **Phase 2 — Motion Capture and Planner Bringup**: steps `6-8`
3. **Phase 3 — Human Motion Data and Retargeting**: steps `9-11`
4. **Phase 4 — WBC Training Pipeline**: steps `12-16`
5. **Phase 5 — Runtime Policy and Hardware Deployment**: steps `17-18`
6. **Phase 6 — Portability and Full-System Integration**: steps `19-21`
7. **Phase 7 — Testing, Safety, and Competition Readiness**: steps `22-26`
8. **Phase 8 — Summaries, Constraints, References, and Open Values**: steps `27-30`

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

## Execution Scope And Shell Conventions

This guide uses three execution scopes. Do not mix them casually.

1. **Host terminal**
   - Use the normal shell on the machine itself.
   - Use this for `distrobox create`, `distrobox enter`, Docker/Podman setup, copying vendor zip/tar files into `~/workspace/HOPE`, and physical-lab tasks such as mocap calibration or robot-network checks.

2. **ROS distrobox: `hope`**
   - Use this box for the ROS 2 Jazzy workspace in `~/workspace/HOPE/hope_ws`.
   - Run all `ros2`, `colcon`, `rosdep`, mocap, planner, monitoring, and deployment commands here unless a later step explicitly says otherwise.
   - Practical rule: steps `4-8` and `16-21` are ROS-box work.

3. **GPU / Isaac distrobox: `grasping` or another NVIDIA-enabled box**
   - Use this box for Isaac Sim, Isaac Lab, BeyondMimic, WandB motion preprocessing, policy training, evaluation, and ONNX export.
   - Practical rule: steps `9-15` are GPU-box work. In particular, Step 9 (GVHMR) and Step 10 (GMR) are Python/ML toolchain work and are usually easier in `grasping` than in `hope`.
   - Do not install Isaac Sim or Isaac Lab into `hope`; keep the ROS box and the training box separate.

Shared-path rule:

- `~/workspace/HOPE` is visible from the host and from the distroboxes on this machine, so files created in one environment appear in the others.
- Example: you may clone `motion_tracking_controller` from the host or from `hope`, but the `python` commands that depend on Isaac Lab still belong in the GPU distrobox.

## Phase 1 — Scope, Platform, and Workspace Foundation

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

### 2.1 A3 materials now available, and what still must be verified

The project originally treated A3-specific assets as the largest external blocker because they were not public. That blocker is now partially resolved by Agibot-provided materials already placed in this repo or local asset area. Treat these as the current project source of truth, not as X1 stand-ins:

- A3 URDF and meshes: `agi/URDF/`.
- A3 MuJoCo/AimRT simulation materials: `agi/A3_MuJoCo_Sim/`.
- A3 deploy documentation and source: `agi/code_deployment/` and `agi/code_deployment/a3_deploy_example/`.
- Full local deploy payload, including heavy runtime assets: `vendor_assets/agibot/a3_deploy_example_full/`.
- Current working joint order for training/export alignment: `hope_training/config/joint_order_agibot_a3.yaml`.
- Isaac/BeyondMimic A3 robot config and deploy-transcribed PD/action-scale values: `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py`.

What remains blocked is not "do we have any A3 model"; it is hardware and cross-runtime verification:

- Exact `base_link` physical interpretation and measured mocap-marker to `base_link` transform.
- Confirmation that the hardware SDK joint-state and joint-command order matches the project YAML.
- A3 control SDK / AimDK / ROS 2 runtime behavior on the actual robot.
- Hardware E-stop, soft-stop, standby, and recovery procedure.
- Safe low-gain command path from exported policy targets to A3 joint commands.

Keep all A3-specific assumptions behind explicit config files and bridge packages. If hardware verification changes an order, frame, gain, or runtime topic, update the single source of truth first, then propagate from there.

How to get and verify the A3 placeholders:

| Placeholder | How to get it | How to verify it |
| --- | --- | --- |
| `A3_URDF`, `A3_MJCF`, or `A3_USD` | Current project source: `agi/URDF/` and `agi/A3_MuJoCo_Sim/`; request updates from Agibot if these change. | Open the model in ROS/Isaac/MuJoCo and confirm the robot height is about `1.73 m` and the joint count matches the A3 documentation. |
| `base_link` name | Read the root or pelvis frame name in the A3 URDF, or ask Agibot for the official control-frame name. | Run `ros2 topic echo /joint_states --once` and confirm the SDK documentation uses the same body frame in its examples. |
| `base_link` height | Put the robot in the official standing calibration pose on level ground. Measure from the floor to the `base_link` origin if the origin is physically marked. If it is not marked, compute it from the robot model by FK in the standing pose. | In simulation, publish `world -> base_link`; the Z value should match the measured standing height within a few centimeters. |
| Joint names and joint order | Current working order: `hope_training/config/joint_order_agibot_a3.yaml`; verify against SDK/API before hardware. | Compare the order in four places: `/joint_states`, the training config, ONNX metadata, and the hardware command message. They must match exactly. |
| PD gains / impedance | Current training values are transcribed from Agibot deploy materials in `robots/agibot_a3.py`; still verify safe hardware startup behavior. | In low-gain standby, command tiny joint motions and confirm there is no buzzing, overshoot, or unexpected motion. |
| E-stop and standby API | Get the exact hardware E-stop wiring, software stop service/topic, and standby command from the A3 manual. | Time a stop test with logs or high-speed video. The required upper-body and gait stop time is below `200 ms`. |

If one of the hardware-verification values is missing, write it into your local `A3_BLOCKERS.md` and do not proceed to real hardware deployment for that item.

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

Execution scope from this point:

- Unless a later step says otherwise, ROS 2 commands belong inside `distrobox enter hope`.
- Physical mocap setup, robot cabling, and network inspection still happen outside the container in the real arena.
- The Isaac Sim / Isaac Lab training steps later in this guide do **not** run in `hope`; they switch to a separate GPU distrobox.

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

## Phase 2 — Motion Capture and Planner Bringup

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
source install/local_setup.bash
'
```

### 6.1 Step-By-Step Commands

Use this exact flow for the current repository version.

1. Check the relay config first:

```bash
distrobox enter hope -- bash -lc '
sed -n "1,80p" ~/workspace/HOPE/hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml
'
```

2. Confirm these four lines:
   - `ppt_object: "PPT"`
   - `p1_object: "P1"`
   - `p2_object: "P2"`
   - `ball_object: ""`

3. If the rigid-body names in CMTracker are different, edit the file:

```bash
distrobox enter hope -- bash -lc '
nano ~/workspace/HOPE/hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml
'
```

Important:

- `PPT`, `P1`, and `P2` must match CMTracker exactly.
- `ball_object` should normally stay empty. This code auto-detects the ball by motion.
- If `P1` and `P2` do not exist yet, Step 6 is still useful for `PPT` plus the ball. In that case, skip the `/P1/pose` and `/P2/pose` checks later.

4. Rebuild `hope_bringup` after any config change:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_bringup
'
```

5. Before starting, stop old ROS nodes from previous tests:

```bash
distrobox enter hope -- bash -lc '
pkill -f avatar_pro_vrpn_relay || true
pkill -f client_node || true
pkill -f static_transform_publisher || true
source /opt/ros/jazzy/setup.bash
ros2 daemon stop || true
ros2 daemon start
'
```

6. Open Terminal A and start the full bridge. Replace `192.168.1.100` with the
   IP found by `ipconfig` on the Avatar-Pro PC:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py \
  server:=192.168.1.100 \
  port:=3883 \
  update_freq:=180.0
'
```

Keep Terminal A open.

What that launch starts:

1. `vrpn_mocap client_node` — Avatar-Pro VRPN -> `/vrpn_mocap/<sender>/<sensor>/pose`.
2. `avatar_pro_vrpn_relay` — discovers `/vrpn_mocap/*` topics, matches `PPT`/`P1`/`P2` by sender name, auto-detects the ball as the moving non-rigid marker, and republishes HOPE topics.
3. `hope_world.launch.py` — the static world-frame landmarks and `P -> P_base_link` offsets.

What you should see in Terminal A:

- one line like `ball auto-detect: will lock onto the moving non-rigid marker`
- discovery lines like `discovered /vrpn_mocap/PPT/.../pose -> PPT`
- after you move the real ball by hand, one lock line like `ball -> /vrpn_mocap/.../pose (moving marker ...)`

Very important:

- `/ball/point` and `/poses` are not expected to publish before the relay locks onto the ball.
- In this code version, `/poses` is published on ball updates only, not on table/P1/P2 updates.
- So if the ball is standing still, `/poses` may appear quiet even though the relay is healthy.

7. Open Terminal B and check that the topic names exist:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 topic list | grep -E "vrpn_mocap|ball|P1|P2|table|poses"
'
```

You should at least see some names like:

```text
/vrpn_mocap/...
/table/pose
/P1/pose
/ball/point
/poses
```

8. Still in Terminal B, check the table and optional P1 rigid body:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 topic echo /table/pose --once
ros2 topic echo /P1/pose --once
'
```

If `P1` is not set up yet, skip the second command. If `position` and
`orientation` print for `/table/pose`, the VRPN -> ROS path is already partly working.

9. Move the real ball in front of the cameras, then check the ball topic:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 topic echo /ball/point --once
'
```

If it works, you should see:

```text
point:
  x: ...
  y: ...
  z: ...
```

10. Check `/poses`:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 topic hz /poses
'
```

Remember: `/poses` only becomes active after the ball has been locked and starts updating.

11. If the ball never appears, inspect the raw VRPN topics directly:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep vrpn_mocap
'
```

If you see many `/vrpn_mocap/...` topics but no `/ball/point`, go back to Terminal A
and look for a line like `ball -> /vrpn_mocap/.../pose`.

If you still cannot find the ball topic, interpret the result like this:

1. Nothing under `/vrpn_mocap/` at all -> the client is not connected: wrong server IP/port, different subnet, firewall, or VRPN streaming is off in Avatar-Pro.
2. You see `PPT`/`P1`/`P2` but no extra moving object -> the ball is an Avatar-Pro problem: the single marker is not defined as a tracked marker, or individual-marker streaming is off.
3. You see an extra topic whose position tracks the ball -> good. The relay auto-detect will lock onto it; you do not need to copy its id anywhere. If you prefer to pin it, put that sender name/id in `ball_object`.
4. You see an extra moving topic, but the relay still does not lock -> lower `ball_lock_speed_mps` a little in `avatar_pro_vrpn.yaml`, rebuild `hope_bringup`, relaunch, and try again.

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

### 8.1 Step-By-Step Commands

Use this exact flow for the current repository version.

1. Step 8 requires Step 6 to stay alive. Keep the Step 6 launch terminal open while recording.

2. Build the planner tools first:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_planner
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 pkg executables hope_planner
'
```

You should see at least:

```text
hope_planner hope_bag_to_csv
hope_planner hope_calibrate
hope_planner hope_planner_node
```

3. Create the working folders:

```bash
mkdir -p ~/workspace/HOPE/calib_bags
mkdir -p ~/workspace/HOPE/calib_csv
```

4. Before recording anything, confirm `/ball/point` is alive:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 topic echo /ball/point --once
'
```

If this waits forever, stop and fix Step 6 first.

5. Record the first trajectory. If an old bad folder exists, delete it first:

```bash
rm -rf ~/workspace/HOPE/calib_bags/traj01
```

Then record:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
cd ~/workspace/HOPE/calib_bags
ros2 bag record -s mcap /ball/point -o traj01
'
```

During recording:

- keep the recorder running
- toss the ball once
- try to include one clean free-flight arc
- ideally include one table bounce
- press `Ctrl+C` after that one toss
- wait for rosbag to finish writing metadata before closing the terminal

6. Check that the bag was written correctly:

```bash
distrobox enter hope -- bash -lc '
cd ~/workspace/HOPE
ls -lah calib_bags/traj01
'
```

You want:

- `metadata.yaml`
- `traj01_0.mcap`
- the `.mcap` file must not be `0 bytes`

If the `.mcap` file is `0 bytes` or `metadata.yaml` is missing, delete the folder and record again:

```bash
rm -rf ~/workspace/HOPE/calib_bags/traj01
```

7. Convert that first bag to CSV:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
cd ~/workspace/HOPE
ros2 run hope_planner hope_bag_to_csv \
  --bag calib_bags/traj01 \
  --topic /ball/point \
  --output calib_csv/traj01.csv
'
```

You should see something like:

```text
Wrote 123 rows to calib_csv/traj01.csv
```

8. Check that the CSV looks right:

```bash
head ~/workspace/HOPE/calib_csv/traj01.csv
```

You should see the header:

```text
t,x,y,z
```

Each calibration CSV must contain one continuous ball trajectory in the HOPE
`world` frame, with time in seconds and position in meters.

9. Repeat the same flow for `traj02` through `traj15`. One toss per bag, one CSV per bag.

Example for `traj02`:

```bash
rm -rf ~/workspace/HOPE/calib_bags/traj02

distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
cd ~/workspace/HOPE/calib_bags
ros2 bag record -s mcap /ball/point -o traj02
'
```

Then convert it:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
cd ~/workspace/HOPE
ros2 run hope_planner hope_bag_to_csv \
  --bag calib_bags/traj02 \
  --topic /ball/point \
  --output calib_csv/traj02.csv
'
```

10. If all `traj01` to `traj15` bags already exist and look healthy, you can batch-convert them:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
cd ~/workspace/HOPE
for i in $(seq -w 1 15); do
  ros2 run hope_planner hope_bag_to_csv \
    --bag calib_bags/traj$i \
    --topic /ball/point \
    --output calib_csv/traj$i.csv
done
'
```

11. Run the calibration:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
cd ~/workspace/HOPE
ros2 run hope_planner hope_calibrate calib_csv/traj*.csv
'
```

You should get values like:

```text
drag_k = ...
restitution_h = ...
restitution_v = ...
```

12. Paste those values into:

```bash
distrobox enter hope -- bash -lc '
nano ~/workspace/HOPE/hope_ws/src/hope_planner/config/hope_planner.yaml
'
```

Update:

```yaml
drag_k: ...
restitution_h: ...
restitution_v: ...
```

Do not change this yet:

```yaml
restitution_racket: 0.88
```

13. Rebuild `hope_planner`:

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install --packages-select hope_planner
'
```

Recording notes:

- start recording only after `/ball/point` is alive
- one toss per bag is best; do not mix a whole practice session into one file
- keep at least 20 percent of the throws as held-out test trajectories for later validation
- if values in the CSV are obviously not table-scale numbers, stop and fix the upstream units before fitting

It prints fitted `drag_k`, `restitution_h`, `restitution_v` ready to paste into
`config/hope_planner.yaml`. Note the three-sample bounce detector is phase-
sensitive: record at the full mocap rate (≥240–360 Hz) so each bounce yields a
clean descend→contact→rise pattern.

Verification gate:

- Predicted hit-plane crossing is close to measured crossing on held-out trajectories.
- Predicted bounce count matches observed bounce count.
- Net-clearance checks match visual inspection or high-speed video.

## Phase 3 — Human Motion Data and Retargeting

## 9. Acquire Human Swing Motions

### 9.0 New-Video Pipeline: Step 9 -> Step 11

For the current repo, the simplest way to process a newly recorded clip is:

1. Put the new clip under `~/workspace/HOPE/hope_training/motions/raw_video/`.
2. Rename it to either `forehand_swing.mp4` or `backhand_swing.mp4`.
3. Re-run the same Step 9 -> Step 11 commands below.

Current limitation:

- `assets/agibot_a3/postprocess_ground.py` currently assumes the motion names `forehand_swing` and `backhand_swing`.
- `assets/agibot_a3/render_inspect.py` currently assumes the same names.
- Under the current helper scripts, keep both motion files present:
  `forehand_swing.pkl` and `backhand_swing.pkl`.
  If you only replace one clip, leave the other one in place or update the helper scripts first.

Use this command-and-output flowchart for one new forehand clip. For a backhand clip,
replace every `forehand_swing` token with `backhand_swing`.

```text
new video
  ~/workspace/HOPE/hope_training/motions/raw_video/forehand_swing.mp4
    |
    | Step 9: GVHMR
    | command:
    |   cd ~/workspace/HOPE/hope_training/GVHMR
    |   source ~/workspace/HOPE/hope_training/.venv-motion/bin/activate
    |   python tools/demo/demo.py --video=../motions/raw_video/forehand_swing.mp4 -s
    | writes:
    |   GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt
    v
human motion in SMPL-X form
  GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt
    |
    | Step 10: GMR retarget
    | command:
    |   cd ~/workspace/HOPE/hope_training/GMR
    |   source ~/workspace/HOPE/hope_training/.venv-motion/bin/activate
    |   python scripts/gvhmr_to_robot.py \
    |     --gvhmr_pred_file ../GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt \
    |     --robot agibot_a3 \
    |     --save_path ../motions/a3_gmr/forehand_swing.pkl
    | writes:
    |   motions/a3_gmr/forehand_swing.pkl
    v
robot joint motion
  motions/a3_gmr/forehand_swing.pkl
    |
    | Step 10: ground the feet
    | command:
    |   python assets/agibot_a3/postprocess_ground.py
    | writes:
    |   motions/a3_gmr/forehand_swing.pkl
    |   motions/a3_gmr/forehand_swing_raw.pkl
    v
grounded robot joint motion
  motions/a3_gmr/forehand_swing.pkl
    |
    | Step 10: inspect the body motion
    | command:
    |   PYTHONPATH=. ~/workspace/HOPE/hope_training/.venv-motion/bin/python \
    |     scripts/vis_robot_motion.py \
    |     --robot agibot_a3 \
    |     --robot_motion_path ../motions/a3_gmr/forehand_swing.pkl
    | writes:
    |   no new files
    v
checked robot body motion
    |
    | Step 11: build the fixed racket model
    | command:
    |   python assets/agibot_a3/build_racket_xml.py
    | writes:
    |   assets/agibot_a3/a3_racket.xml
    v
robot + fixed racket model
  assets/agibot_a3/a3_racket.xml
    |
    | Step 11: validate FK
    | command:
    |   MUJOCO_GL=egl python assets/agibot_a3/racket_fk.py
    | writes:
    |   no new files
    |
    | Step 11: render inspection PNGs
    | command:
    |   python assets/agibot_a3/render_inspect.py
    | writes:
    |   motions/a3_gmr/forehand_swing_contact.png
    |   motions/a3_gmr/racket_pose.png
    v
checked robot + racket motion
```

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

Execution shell for this step:

1. Enter the GPU distrobox first:

```bash
distrobox enter grasping
```

2. Run the rest of Step 9 inside `grasping`, not inside the ROS box `hope`.
   The videos live under `~/workspace/HOPE`, which is shared with the host and the other distroboxes.

Organize the recorded videos:

1. Put the raw clips here:

```text
~/workspace/HOPE/hope_training/motions/raw_video/
```

2. Use these filenames:

```text
forehand_swing.mp4
backhand_swing.mp4
```

Create a dedicated Python environment for the motion-processing tools before installing
GVHMR or GMR. Do not use the host system `pip`; on Ubuntu 24.04/26.04 it can fail with
the PEP 668 `externally-managed-environment` error, and Ubuntu 26.04's default
Python 3.14 is newer than the Python 3.10 stack these repos target.

Preferred setup:

```bash
# Install Miniforge or another conda-compatible Python distribution first if needed.
source /opt/conda/etc/profile.d/conda.sh
conda create -n hope-motion-py310 python=3.10 -y
conda activate hope-motion-py310

python --version
python -m pip install --upgrade pip setuptools wheel
```

If you already have a Python 3.10 virtual environment for GVHMR, reuse it for GMR instead
of creating a new one.

If `grasping` already has a reusable Python 3.10 environment for motion tools, activate it
instead of creating another one. The important requirements are:

1. Python `3.10.x`
2. GPU-visible PyTorch support
3. The same environment stays active for both GVHMR and GMR

Create the GVHMR working folders if they do not already exist:

```bash
cd ~/workspace/HOPE/hope_training/GVHMR
mkdir -p inputs outputs inputs/checkpoints
```

Install GVHMR:

```bash
cd ~/workspace/HOPE/hope_training
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR
# Follow the GVHMR repository installation instructions inside the active
# `hope-motion-py310` environment.
```

Minimum install commands:

```bash
cd ~/workspace/HOPE/hope_training/GVHMR
python -m pip install -r requirements.txt
python -m pip install -e .
```

If `python -m pip install -r requirements.txt` fails while building `chumpy` with an error like
`ModuleNotFoundError: No module named 'pip'`, install `chumpy` once without build isolation and then rerun
the requirements command:

```bash
cd ~/workspace/HOPE/hope_training/GVHMR
python -m pip install --no-build-isolation chumpy
python -m pip install -r requirements.txt
```

This is a packaging quirk in `chumpy`, not evidence that the Python 3.10 environment itself is broken.

Download the GVHMR prerequisites before running the demo:

1. Download SMPL and SMPL-X body models from their official sites:
   - `https://smpl.is.tue.mpg.de/`
   - `https://smpl-x.is.tue.mpg.de/`
2. Download the other pretrained checkpoints referenced by `GVHMR/docs/INSTALL.md`.
3. Arrange the files under:
   For the current HOPE Step 9 demo path, use the neutral body models.
   You do not need male/female here unless you are extending GVHMR beyond this demo flow.
```text
GVHMR/inputs/checkpoints/
  body_models/smplx/SMPLX_NEUTRAL.npz
  body_models/smpl/SMPL_NEUTRAL.pkl
  dpvo/dpvo.pth
  gvhmr/gvhmr_siga24_release.ckpt
  hmr2/epoch=10-step=25000.ckpt
  vitpose/vitpose-h-multi-coco.pth
  yolo/yolov8x.pt
```

4. If your camera was not static, also prepare:

```text
GVHMR/inputs/checkpoints/dpvo/dpvo.pth
```

Do not download the GVHMR training and evaluation support archives for Step 9 demo inference:

```text
AMASS_hmr4d_support.tar.gz
BEDLAM_hmr4d_support.tar.gz
H36M_hmr4d_support.tar.gz
3DPW_hmr4d_support.tar.gz
EMDB_hmr4d_support.tar.gz
RICH_hmr4d_support.tar.gz
```

Those archives are for GVHMR dataset preparation, training, and benchmark evaluation, not for running
`tools/demo/demo.py` on your own forehand and backhand videos.

Run GVHMR:
 two clips one at a time in this order:

1. Forehand:

```bash
cd ~/workspace/HOPE/hope_training/GVHMR
python tools/demo/demo.py --video=../motions/raw_video/forehand_swing.mp4 -s
```

2. Backhand:

```bash
python tools/demo/demo.py --video=../motions/raw_video/backhand_swing.mp4 -s
```

Use `-s` only when the camera is static, for example on a tripod or fixed mount.
If the camera moved during recording, omit `-s` and install the optional DPVO dependency
plus its checkpoint first.

Expected output:

```text
GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt
GVHMR/outputs/demo/backhand_swing/hmr4d_results.pt
```

Quick check:

```bash
find ~/workspace/HOPE/hope_training/GVHMR/outputs/demo/forehand_swing -maxdepth 2 -type f | sort
find ~/workspace/HOPE/hope_training/GVHMR/outputs/demo/backhand_swing -maxdepth 2 -type f | sort
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
python -m pip install -e .
```

If `python -m pip install -e .` still prints `externally-managed-environment`, the conda
or virtual environment is not active yet. Do not use `--break-system-packages` here.

Download SMPL-X body models:

1. Go to `https://smpl-x.is.tue.mpg.de/`.
2. Register or sign in, because the downloads page is license-gated.
3. Download the main **SMPL-X model** bundle from the downloads page.
   Do not download the Blender add-on, Unity package, UV maps, or segmentation map for this step.
4. Extract the archive and locate the `smplx/` model files inside it.
   The official `smplx` loader supports a folder that contains both `.npz` and `.pkl` variants.
5. Put them under:

```text
GMR/assets/body_models/smplx/
```

Required files:

```text
SMPLX_NEUTRAL.pkl
SMPLX_FEMALE.pkl
SMPLX_MALE.pkl
```

Why GMR asks for three genders while GVHMR Step 9 only uses neutral:

- The current `GVHMR -> GMR` path in this project feeds GVHMR output into GMR using the neutral body model, so GVHMR Step 9 only needs `SMPLX_NEUTRAL.npz` and `SMPL_NEUTRAL.pkl`.
- GMR's standalone SMPL-X loading utilities are more general and can read motion files labeled as `male`, `female`, or `neutral`, so its documentation asks you to prepare all three `.pkl` files.
- For the current HOPE reimplementation flow, GVHMR using neutral does not hurt the later GMR step.

You may also see matching `.npz` files in the extracted archive:

```text
SMPLX_NEUTRAL.npz
SMPLX_FEMALE.npz
SMPLX_MALE.npz
```

That is normal. If the archive contains both `.pkl` and `.npz`, keep both in
`GMR/assets/body_models/smplx/`.

Quick check:

```bash
find ~/workspace/HOPE/hope_training/GMR/assets/body_models/smplx -maxdepth 1 -type f | sort
```

Step 10 result:

- `assets/agibot_a3/a3_mocap.xml`
- `general_motion_retargeting/ik_configs/smplx_to_a3.json`
- GMR registration for `agibot_a3`
- `hope_training/motions/a3_gmr/forehand_swing.pkl`
- `hope_training/motions/a3_gmr/backhand_swing.pkl`
- `hope_training/config/joint_order_agibot_a3.yaml`

Step 10 primary outputs are PKL motion files. PNG inspection images are generated
later by `assets/agibot_a3/render_inspect.py` after Step 11 builds
`assets/agibot_a3/a3_racket.xml`.

Run every command in this step from:

```bash
cd ~/workspace/HOPE/hope_training/GMR
source ~/workspace/HOPE/hope_training/.venv-motion/bin/activate
```

### 10.1 Inputs

1. Use `agi/URDF/a3_t2d5/` as the robot body model for retargeting.
2. Do **not** use `agi/URDF/A3T2.5-URDF-std-pingpang/` in Step 10. That model is only for Step 11 because it includes the fixed racket links.
3. Use the Step 9 GVHMR outputs:
   - `../GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt`
   - `../GVHMR/outputs/demo/backhand_swing/hmr4d_results.pt`

Quick check:

```bash
test -f ~/workspace/HOPE/agi/URDF/a3_t2d5/urdf/model.urdf
test -f ../GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt
test -f ../GVHMR/outputs/demo/backhand_swing/hmr4d_results.pt
```

### 10.2 Create and update the A3 target in GMR

Create these files once:

- `assets/agibot_a3/convert_urdf_to_mjcf.py`
- `assets/agibot_a3/postprocess_ground.py`
- `assets/agibot_a3/render_inspect.py`
- `general_motion_retargeting/ik_configs/smplx_to_a3.json`
- `hope_training/config/joint_order_agibot_a3.yaml`

Update these files once:

- `general_motion_retargeting/params.py`
- `scripts/gvhmr_to_robot.py`

Required code changes:

1. `convert_urdf_to_mjcf.py` must read `agi/URDF/a3_t2d5/urdf/model.urdf`, add a free joint to `pelvis_link`, and write `assets/agibot_a3/a3_mocap.xml`.
2. `smplx_to_a3.json` must map SMPL-X joints onto A3 links. Start from `smplx_to_g1.json` and rename the robot links to A3.
3. `params.py` must register `agibot_a3` in:
   - `ROBOT_XML_DICT`
   - `IK_CONFIG_DICT["smplx"]`
   - `ROBOT_BASE_DICT`
   - `VIEWER_CAM_DISTANCE_DICT`
4. `gvhmr_to_robot.py` must accept `--robot agibot_a3`.
5. `joint_order_agibot_a3.yaml` must store the 31 hinge-joint order from `a3_mocap.xml`. Reuse this same order for training, ONNX export, and the hardware bridge.

Sanity check:

```bash
python -c "from general_motion_retargeting import ROBOT_XML_DICT, IK_CONFIG_DICT; print('agibot_a3' in ROBOT_XML_DICT, 'agibot_a3' in IK_CONFIG_DICT['smplx'])"
# expected: True True
```

### 10.3 Clean old outputs before a fresh rerun

Skip this if you want to keep the previous outputs.

```bash
rm -f ../motions/a3_gmr/forehand_swing.pkl \
      ../motions/a3_gmr/backhand_swing.pkl \
      ../motions/a3_gmr/forehand_swing_raw.pkl \
      ../motions/a3_gmr/backhand_swing_raw.pkl
```

### 10.4 Run Step 10

1. Build the MuJoCo model:

```bash
python assets/agibot_a3/convert_urdf_to_mjcf.py
```

Writes:

```text
assets/agibot_a3/a3_mocap.xml
```

Expected output:

```text
nbody=33  njnt=32  nq=38  nv=37
has free joint: True
root body: pelvis_link
actuated (hinge) joints: 31
```

2. Retarget forehand:

```bash
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file ../GVHMR/outputs/demo/forehand_swing/hmr4d_results.pt \
  --robot agibot_a3 \
  --save_path ../motions/a3_gmr/forehand_swing.pkl
```

Writes:

```text
../motions/a3_gmr/forehand_swing.pkl
```

3. Retarget backhand:

```bash
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file ../GVHMR/outputs/demo/backhand_swing/hmr4d_results.pt \
  --robot agibot_a3 \
  --save_path ../motions/a3_gmr/backhand_swing.pkl
```

Writes:

```text
../motions/a3_gmr/backhand_swing.pkl
```

4. Ground the feet:

```bash
python assets/agibot_a3/postprocess_ground.py
```

Writes:

```text
../motions/a3_gmr/forehand_swing.pkl       # overwritten with grounded root_pos
../motions/a3_gmr/backhand_swing.pkl       # overwritten with grounded root_pos
../motions/a3_gmr/forehand_swing_raw.pkl   # one-time backup of the pre-ground PKL
../motions/a3_gmr/backhand_swing_raw.pkl   # one-time backup of the pre-ground PKL
```

Expected output pattern:

```text
forehand_swing: ... AFTER min=0.0000 max=0.0000
backhand_swing: ... AFTER min=0.0000 max=0.0000
```

At this point you have the Step 10 PKL outputs only. Do not expect PNGs yet.

### 10.5 Check Step 10 outputs

Required output files:

```text
assets/agibot_a3/a3_mocap.xml
../motions/a3_gmr/forehand_swing.pkl
../motions/a3_gmr/backhand_swing.pkl
../motions/a3_gmr/forehand_swing_raw.pkl
../motions/a3_gmr/backhand_swing_raw.pkl
```

Check them:

```bash
find assets/agibot_a3 -maxdepth 1 -name 'a3_mocap.xml'
find ../motions/a3_gmr -maxdepth 1 \( -name 'forehand_swing*.pkl' -o -name 'backhand_swing*.pkl' \) | sort
```

Visual check with a viewer:

```bash
PYTHONPATH=. ~/workspace/HOPE/hope_training/.venv-motion/bin/python \
  scripts/vis_robot_motion.py \
  --robot agibot_a3 \
  --robot_motion_path ../motions/a3_gmr/forehand_swing.pkl
```

Writes:

```text
no new files
```

Headless check with rendered PNGs:

```bash
# run this only after Step 11 creates assets/agibot_a3/a3_racket.xml
python assets/agibot_a3/render_inspect.py
```

Step 10 passes when:

- no impossible joint angles and no joint-limit violations,
- the right wrist follows the swing,
- both feet stay on the ground after grounding,
- the torso stays upright and the robot remains balanced.

### 10.6 If Step 10 needs edits

1. If GMR cannot find the robot, fix `params.py` and `gvhmr_to_robot.py`.
2. If the model fails to compile, fix `convert_urdf_to_mjcf.py` or the source URDF path.
3. If a limb is twisted, edit the quaternion offset in `smplx_to_a3.json`, then rerun 10.4 and 10.5.
4. If joint order later mismatches the SDK, update only `hope_training/config/joint_order_agibot_a3.yaml` and propagate that one shared order everywhere else.

### 10.7 Handoff to Step 11

Do not treat the retargeted hand pose as the racket pose. The racket pose must be:

```text
world -> pelvis_link -> waist -> torso -> shoulder -> elbow -> wrist -> T_mount -> racket
```

Step 11 result:

- `assets/agibot_a3/a3_racket.xml`
- a reusable wrist-to-racket transform `T_mount`
- a reusable FK helper for racket center and normal
- `hope_training/motions/a3_gmr/forehand_swing_contact.png`
- `hope_training/motions/a3_gmr/backhand_swing_contact.png`
- `hope_training/motions/a3_gmr/racket_pose.png`

## 11. Model The Fixed Racket Mount

Run every command in this step from:

```bash
cd ~/workspace/HOPE/hope_training/GMR
source ~/workspace/HOPE/hope_training/.venv-motion/bin/activate
```

### 11.1 Inputs

Use the `std-pingpang` URDF only in this step:

- `agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf`

Copy the three real paddle meshes from that URDF's `meshes/` into the A3 asset
mesh dir once, so the racket model can render the actual hand+paddle (not a
synthetic disk):

```bash
cp ../../agi/URDF/A3T2.5-URDF-std-pingpang/meshes/{pingpang_red_Link,pingpang_black_Link,right_hand_pingpang_Link}.STL \
   assets/agibot_a3/meshes/
```

Use these mount constants:

- `T_mount = [0.21021, 0.032078, 0.032036]` m
- rotation = identity
- racket face normal = `+Y`
- blade diameter `≈ 160 mm`
- blade center in the racket link frame `≈ [-0.004, -0.0015, -0.004]`

### 11.2 Create and update the fixed-racket files

Create these files once:

- `assets/agibot_a3/build_racket_xml.py`
- `assets/agibot_a3/racket_fk.py`

Required code behavior:

1. `build_racket_xml.py` must read `a3_mocap.xml`, attach a fixed `racket_root` body to `right_wrist_yaw_Link` at `T_mount`, and write `a3_racket.xml`.
2. The new model must:
   - mount the real paddle meshes at their `std-pingpang` transforms — the fist+handle (`right_hand_pingpang_Link`) on the wrist at offset 0, and the red/black blade faces (`pingpang_red_Link`/`pingpang_black_Link`) on `racket_root` — so the racket is held on the handle and visually matches the real robot,
   - hide a3's open `right_hand_Link` geoms (transparent, no collision) so they do not poke through the paddle,
   - add a `racket_center` site at the blade center (use this for the reward / ball-contact target) and a `racket_normal` site marking the `+Y` face normal.
3. `racket_fk.py` must expose FK for racket center and face normal, and it must validate against the original `std-pingpang` URDF.

Sanity-check the placement against the real model by rendering the A3 racket next
to the `std-pingpang` URDF (same pose, zoomed on the right hand); the fist, handle,
and blade should coincide.

### 11.3 Clean old output before a fresh rerun

Skip this if you want to keep the previous result.

```bash
rm -f assets/agibot_a3/a3_racket.xml
```

### 11.4 Run Step 11

1. Build the racket model:

```bash
python assets/agibot_a3/build_racket_xml.py
```

Writes:

```text
assets/agibot_a3/a3_racket.xml
```

Expected output:

```text
nq=38 nv=37 hinge_dof=31
racket_root body present: True
sites present: racket_center=True racket_normal=True
```

2. Validate FK:

```bash
MUJOCO_GL=egl python assets/agibot_a3/racket_fk.py
```

Writes:

```text
no new files
```

Expected output pattern:

```text
max racket-center residual a3 vs pingpang   : 0.0005 mm
VALIDATION PASS (<1 mm)
```

3. Render the inspection PNGs from the existing PKLs:

```bash
python assets/agibot_a3/render_inspect.py
```

Writes:

```text
../motions/a3_gmr/forehand_swing_contact.png
../motions/a3_gmr/backhand_swing_contact.png
../motions/a3_gmr/racket_pose.png
```

Expected output pattern:

```text
wrote .../forehand_swing_contact.png
wrote .../backhand_swing_contact.png
wrote .../racket_pose.png
```

### 11.5 Check Step 11 outputs

Required output files:

```text
assets/agibot_a3/a3_racket.xml
assets/agibot_a3/build_racket_xml.py
assets/agibot_a3/racket_fk.py
../motions/a3_gmr/forehand_swing_contact.png
../motions/a3_gmr/backhand_swing_contact.png
../motions/a3_gmr/racket_pose.png
```

Check them:

```bash
find assets/agibot_a3 -maxdepth 1 \( -name 'a3_racket.xml' -o -name 'build_racket_xml.py' -o -name 'racket_fk.py' \) | sort
find ../motions/a3_gmr -maxdepth 1 \( -name 'forehand_swing_contact.png' -o -name 'backhand_swing_contact.png' -o -name 'racket_pose.png' \) | sort
```

Step 11 passes when:

- `a3_racket.xml` builds with unchanged DOF,
- the racket is held on the handle on the right wrist (matches the `std-pingpang` model, no overlap with the open hand),
- `racket_fk.py` passes with sub-mm residual,
- the planner and simulator both use the same `+Y` racket-normal convention.

### 11.6 If Step 11 needs edits

1. If the racket is misplaced, update the constants in `build_racket_xml.py`.
2. If FK fails against the URDF, fix the same constants in one place and rerun 11.4.
3. If the real printed bracket differs from the URDF, update `T_mount`, rebuild `a3_racket.xml`, rerun `racket_fk.py`, and then reuse the new constants everywhere else.

## Phase 4 — WBC Training Pipeline

## 12. Preprocess Motions For BeyondMimic

Execution scope for Phase 4:

- **Host terminal**: use it only to enter the GPU distrobox and to unpack or copy vendor assets into `~/workspace/HOPE/hope_training`.
- **GPU distrobox**: run every command in steps `12-15` inside `grasping` or another NVIDIA-enabled Isaac distrobox.
- **ROS distrobox `hope`**: do not install Isaac Sim, Isaac Lab, or BeyondMimic here.

If you already finished steps `10` and `11`, then step `12` on this repo means:

1. Turn the final GMR `.pkl` motions into retargeted `.csv`.
2. Enter the GPU / Isaac environment.
3. Install `whole_body_tracking` and set up WandB.
4. Copy the A3 ping-pong URDF assets into the training repo.
5. Turn each retargeted `.csv` into a BeyondMimic `.npz` and upload it to the WandB `motions` registry.
6. Replay the uploaded motions to verify the robot and joint order are correct.

Do the following **in order**. Do not skip ahead.

### 12.1 Check that step 10 / 11 outputs exist

From any shell:

```bash
cd ~/workspace/HOPE/hope_training
ls -lh motions/a3_gmr/forehand_swing.pkl
ls -lh motions/a3_gmr/backhand_swing.pkl
ls -lh motions/a3_gmr/forehand_swing_contact.png
ls -lh motions/a3_gmr/backhand_swing_contact.png
```

You should already have:

- `motions/a3_gmr/forehand_swing.pkl`
- `motions/a3_gmr/backhand_swing.pkl`
- the corresponding inspection `.png` files from step `11`

If either `.pkl` file is missing, go back to step `10` first.

### 12.2 Convert the final GMR `.pkl` files into retargeted `.csv`

Run this in the same motion environment you used for steps `9-11`:

```bash
cd ~/workspace/HOPE/hope_training/GMR
source ~/workspace/HOPE/hope_training/.venv-motion/bin/activate
python scripts/batch_gmr_pkl_to_csv.py --folder ../motions/a3_gmr
```

This writes CSV files to:

```text
~/workspace/HOPE/hope_training/motions/a3_gmr/csv/
```

Now copy only the **final** motions into the retargeted folder used by step `12`:

```bash
mkdir -p ~/workspace/HOPE/hope_training/motions/retargeted
cp ~/workspace/HOPE/hope_training/motions/a3_gmr/csv/forehand_swing.csv \
  ~/workspace/HOPE/hope_training/motions/retargeted/
cp ~/workspace/HOPE/hope_training/motions/a3_gmr/csv/backhand_swing.csv \
  ~/workspace/HOPE/hope_training/motions/retargeted/
ls -lh ~/workspace/HOPE/hope_training/motions/retargeted
```

Expected files:

- `motions/retargeted/forehand_swing.csv`
- `motions/retargeted/backhand_swing.csv`

Note: `batch_gmr_pkl_to_csv.py` also converts `*_raw.pkl` if those files are in the folder. That is normal. Ignore the `*_raw.csv` files for training.

### 12.3 Enter the GPU / Isaac distrobox

Open a host terminal and enter `grasping`:

```bash
distrobox enter grasping
```

Now inside `grasping`, activate the Isaac / training Python:

```bash
source /opt/drone_venv/bin/activate
cd ~/workspace/HOPE/hope_training/whole_body_tracking
python -c "import sys; print(sys.executable)"
```

Important on this machine:

- `wandb` and the Isaac training dependencies are expected to come from `/opt/drone_venv`.
- Do **not** use plain `/usr/bin/python3` for the training commands in this phase.

### 12.4 Install `whole_body_tracking`

```bash
python -m pip install -e source/whole_body_tracking
python -m pip show whole_body_tracking
```

If `pip show` cannot find `whole_body_tracking`, the editable install did not land.
The actual runtime check for this machine is the `hope_isaac_py ...` check below,
not plain `python -c "import whole_body_tracking"`.

If `python -m pip install -e source/whole_body_tracking` fails with a permission
error like:

```text
Permission denied: '/opt/drone_venv/lib/python3.11/site-packages/...'
```

then do **not** stop here. On this machine, `/opt/drone_venv` may be owned by
`root`, so editable install can fail even though the runtime itself is usable.
Use this no-install fallback for the rest of steps `12-15`:

```bash
cd ~/workspace/HOPE/hope_training/whole_body_tracking

# Source the training env (once per terminal). It sets HOPE_WBT_PYTHONPATH so Isaac's bundled python
# can see hydra/omegaconf (in /opt/drone_venv) + isaaclab_rl, defines the `hope_isaac_py` launcher,
# and exports the wandb team/org/project. The script header documents each piece. It MUST be SOURCED
# (not `./setup_train_env.sh`, which would set everything in a subshell that exits). Re-source every
# new terminal. (Equivalent to exporting HOPE_WBT_PYTHONPATH + defining hope_isaac_py by hand.)
source setup_train_env.sh
```

From this point on, on **this machine**, use `hope_isaac_py` for every Isaac /
Isaac Lab command in steps `12-15`. Do **not** use plain `python scripts/...`
for `csv_to_npz.py`, `replay_npz.py`, `train.py`, or `play.py`.

Quick check:

```bash
hope_isaac_py scripts/csv_to_npz.py --help
hope_isaac_py scripts/replay_npz.py --help
```

Expected behavior:

- both commands print their usage text,
- they may still end with `There was an error running python` because `--help`
  exits through Isaac Sim's wrapper; that is fine,
- this confirms the fallback launcher is wired correctly.

If Isaac / Kit startup prints many lines like:

```text
Failed to create change watch ... errno=28/No space left on device
```

that usually means the Linux `inotify` watch limit is too low, **not** that the
disk is full. On this machine, increase the kernel limits from the host shell:

```bash
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=1024
sudo sysctl -w fs.inotify.max_queued_events=32768
```

To make that persistent:

```bash
cat <<'EOF' | sudo tee /etc/sysctl.d/99-isaac-inotify.conf
fs.inotify.max_user_watches=524288
fs.inotify.max_user_instances=1024
fs.inotify.max_queued_events=32768
EOF

sudo sysctl --system
```

Then restart Isaac / replay processes and rerun the command:

```bash
pkill -f isaacsim || true
pkill -f 'replay_npz.py' || true
```

### 12.5 Configure WandB

WandB has **two different scopes** that you must not confuse — getting this wrong
is the most common Step-12 failure:

- **Runs** (training, smoke tests, checkpoints) are logged under an **entity** —
  your personal entity or a **team** (e.g. `your-wandb-team`). This is `WANDB_ENTITY`.
- The **motions registry** is owned by an **organization**, not a team. Registry
  artifact paths must start with the **org** name (e.g.
  `your-wandb-org`). Passing a *team* name there
  fails with `CommError: Unable to find organization for entity '<team>'`. This is
  `WANDB_REGISTRY_ORG`.

A team usually lives inside an org, so the two values differ. Find your org name at
`https://wandb.ai/registry/` (it appears in the URL / org switcher), or query it:

```bash
curl -s --netrc https://api.wandb.ai/graphql -H "Content-Type: application/json" \
  -d '{"query":"query { viewer { organizations { orgEntity { name } } } }"}'
```

In the browser, open `User Settings -> API Keys` and create or copy an API key.

Back in `grasping`:

```bash
wandb login --relogin --verify
wandb whoami
```

Persist the defaults for this workspace (replace the two names with your own team
and org):

```bash
cat >> ~/.bashrc <<'EOF'

# HOPE / BeyondMimic WandB defaults
export WANDB_ENTITY=your-wandb-team                                     # runs/checkpoints -> your team/entity
export WANDB_REGISTRY_ORG=your-wandb-org  # motions registry -> your ORG (not the team)
export WANDB_PROJECT=hope_wbc
export WANDB_DIR=$HOME/workspace/HOPE/hope_training/wandb
EOF

source ~/.bashrc
mkdir -p "$WANDB_DIR"
echo "$WANDB_ENTITY"
echo "$WANDB_REGISTRY_ORG"
echo "$WANDB_PROJECT"
```

Expected output:

```text
your-wandb-team
your-wandb-org
hope_wbc
```

### 12.6 Create and smoke-test the WandB `motions` registry

In the browser:

1. Open `https://wandb.ai/registry/`.
2. Create a registry named `motions`.
3. Do not rename it. The code expects the registry name to be exactly `motions`.

Run a smoke-test WandB run:

```bash
python - <<'PY'
import os
import wandb

run = wandb.init(
    entity=os.environ["WANDB_ENTITY"],
    project=os.environ["WANDB_PROJECT"],
    name="setup-smoke-test",
)
wandb.log({"setup_ok": 1})
print("Run URL:", run.url)
run.finish()
PY
```

Run a smoke-test artifact upload:

```bash
python - <<'PY'
import os
import numpy as np
import wandb

entity = os.environ["WANDB_ENTITY"]
project = "csv_to_npz"
collection = "smoke_motion"
tmp = "/tmp/smoke_motion.npz"

np.savez(tmp, joint_pos=np.zeros((2, 3)), joint_vel=np.zeros((2, 3)))

with wandb.init(entity=entity, project=project, name=collection) as run:
    art = run.log_artifact(artifact_or_path=tmp, name=collection, type="motions")
    run.link_artifact(artifact=art, target_path=f"wandb-registry-motions/{collection}")
    print("Run URL:", run.url)

org = os.environ.get("WANDB_REGISTRY_ORG", entity)  # registry is org-scoped, not team-scoped
print(f"Expected registry path: {org}/wandb-registry-motions/{collection}:latest")
PY
```

If both smoke tests succeed, WandB is ready.

### 12.7 Copy the A3 ping-pong URDF assets into the training repo

On this repo, use the existing A3 ping-pong URDF package here:

```text
~/workspace/HOPE/agi/URDF/A3T2.5-URDF-std-pingpang
```

Create the training asset folder and copy the files:

```bash
mkdir -p source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf
mkdir -p source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/meshes
mkdir -p source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/config

cp ~/workspace/HOPE/agi/URDF/A3T2.5-URDF-std-pingpang/package.xml \
  source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/
cp -r ~/workspace/HOPE/agi/URDF/A3T2.5-URDF-std-pingpang/meshes/. \
  source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/meshes/
cp -r ~/workspace/HOPE/agi/URDF/A3T2.5-URDF-std-pingpang/config/. \
  source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/config/
cp ~/workspace/HOPE/agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf \
  source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

The copied URDF still points at meshes using `package://0000014503_A3T2.5-URDF-std-pingpang-0409/...`.
Rewrite those mesh paths to local relative paths so Isaac Lab can find them from
`assets/agibot_a3/urdf/model.urdf`:

```bash
sed -i 's#package://0000014503_A3T2.5-URDF-std-pingpang-0409/meshes/#../meshes/#g' \
  source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

Remove the glued ball link. The std-pingpang URDF carries a fixed, massless
`pingbang_ball_Link` rigidly attached to the racket's +Y (red) face — a CAD decoration,
not a target. It has a 40 mm **collision** sphere that gets merged into the racket body and
would interfere with the real dynamic ball you spawn later (and it confusingly sits on the
"back" of the paddle during a backhand). Strip the link and its fixed joint (keep the
red/black blades — those are the hitting surface):

```bash
python - <<'PY'
import xml.etree.ElementTree as ET
p = "source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf"
t = ET.parse(p); r = t.getroot()
for tag, name in (("joint","pingbang_ball_joint"), ("link","pingbang_ball_Link")):
    e = r.find(f"{tag}[@name='{name}']")
    if e is not None: r.remove(e)
t.write(p)
print("removed glued ball link+joint")
PY
```

This does **not** change the 31 DOF or the 32 tracked bodies (the ball was a massless
merged link), so any motions you already converted/uploaded stay valid — no need to redo
Steps 12.8–12.9.

Verify the copied training asset:

```bash
ls -lh source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
ls -lh source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/meshes/pingpang_red_Link.STL
find source/whole_body_tracking/whole_body_tracking/assets/agibot_a3 -maxdepth 2 -type f | sort | sed -n '1,40p'
```

### 12.8 Convert the retargeted `.csv` files into BeyondMimic `.npz`

`scripts/csv_to_npz.py` takes:

- one retargeted CSV
- the robot name `agibot_a3`
- an output collection name such as `hope_forehand`

It does **two** things:

1. It writes a temporary local file to `/tmp/motion.npz`.
2. It uploads that file to the WandB `motions` registry.

Prepare the local folder for keeping a copy of each generated `.npz`:

```bash
mkdir -p ~/workspace/HOPE/hope_training/motions/preprocessed
```

Convert the forehand motion:

```bash
hope_isaac_py scripts/csv_to_npz.py \
  --robot agibot_a3 \
  --input_file ~/workspace/HOPE/hope_training/motions/retargeted/forehand_swing.csv \
  --input_fps 30 \
  --output_name hope_forehand \
  --headless

cp /tmp/motion.npz ~/workspace/HOPE/hope_training/motions/preprocessed/hope_forehand.npz
ls -lh ~/workspace/HOPE/hope_training/motions/preprocessed/hope_forehand.npz
```

Success marker:

```text
[INFO]: Motion saved to wandb registry: motions/hope_forehand
```

On the current repo version, the script exits automatically after this line.
If you are running an older copy of the script and it keeps rendering after this
success line, `Ctrl+C` is safe after the upload has completed.

Convert the backhand motion:

```bash
hope_isaac_py scripts/csv_to_npz.py \
  --robot agibot_a3 \
  --input_file ~/workspace/HOPE/hope_training/motions/retargeted/backhand_swing.csv \
  --input_fps 30 \
  --output_name hope_backhand \
  --headless

cp /tmp/motion.npz ~/workspace/HOPE/hope_training/motions/preprocessed/hope_backhand.npz
ls -lh ~/workspace/HOPE/hope_training/motions/preprocessed/hope_backhand.npz
```

Success marker:

```text
[INFO]: Motion saved to wandb registry: motions/hope_backhand
```

Expected local files:

- `motions/preprocessed/hope_forehand.npz`
- `motions/preprocessed/hope_backhand.npz`

Expected WandB registry entries:

- `$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand:latest`
- `$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand:latest`

If the robot pose looks scrambled during conversion, check `AGIBOT_A3_JOINT_NAMES` in
`source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py`. That list must match
the DOF column order written by your GMR A3 retargeting CSV.

### 12.9 Replay the uploaded motions

Replay forehand:

```bash
hope_isaac_py scripts/replay_npz.py \
  --robot agibot_a3 \
  --registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand"
```

Replay backhand:

```bash
hope_isaac_py scripts/replay_npz.py \
  --robot agibot_a3 \
  --registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand"
```

If you instead see `CommError: Unable to find organization for entity '<name>'`,
you passed a **team** (or the wrong name) where the registry expects your **org** —
set `WANDB_REGISTRY_ORG` to the org name from 12.5 and rerun. The registry prefix
is always the org, never the team. (The `Could not parse xyz string '0.0' / xyz not
specified for axis` lines from the URDF importer are harmless and unrelated.)

### 12.10 Step-12 done checklist

Step `12` is complete only if all of the following are true:

- `motions/retargeted/forehand_swing.csv` exists.
- `motions/retargeted/backhand_swing.csv` exists.
- `motions/preprocessed/hope_forehand.npz` exists.
- `motions/preprocessed/hope_backhand.npz` exists.
- WandB registry contains `hope_forehand` and `hope_backhand`.
- `replay_npz.py` shows the A3 replaying both motions without obviously broken joint order.

Backend note: this build trains the A3 on the **Isaac Lab + URDF** path (reimplement.md step 2.3 /
step 12). The `README.md` and WBC reference doc Section 4A also describe an alternative **mjlab +
MJCF** path for the A3 — that scaffolding is **not** implemented here; only the Isaac Lab tasks are
registered. Use Isaac Lab unless you deliberately switch to mjlab.

Verification gate:

- The Agibot A3 (or X1 stand-in) model replays each motion.
- The wrist and racket mount follow a plausible stroke.
- Feet, pelvis, torso, and arms do not jitter badly.
- WandB artifacts are registered and accessible.

## 13. Implement HOPE-Specific WBC Training Extensions

> Reference-implementation status: this step is already coded in the training repo under
> `hope_training/whole_body_tracking/`. You should treat step 13 as **task design and tuning**,
> not as a prompt to write a fresh PPO loop.

The HOPE training stack now has one clear split:

1. **Task design** lives in the Agibot A3 environment code and task YAMLs.
2. **PPO hyperparameters** live in one shared YAML file.
3. **Training / evaluation entrypoints** are `scripts/train.py` and `scripts/play.py`.

Use these files as the source of truth:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py
  HOPE-specific observations, rewards, and extra domain randomization

source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py
  racket/base target sampling, strike timing, FK-computed racket state

source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py
  HOPE reward formulas and strike-time gating

cfg/task/HOPEPingPong.yaml
  the main task-tuning file: reward weights/stds, racket target ranges, strike timing

cfg/base/{env_base,sim_base,randomization_base}.yaml
  shared num_envs, 50 Hz timing, and shared DR defaults

cfg/algo/ppo.yaml
  single source of truth for PPO hyperparameters
```

What the HOPE task adds on top of plain BeyondMimic tracking:

1. **A sampled racket target command**:
   - desired racket position, velocity, and face normal
   - desired base XY position
   - strike timing (`time_to_strike`, `pre_strike`, `strike_window`)
2. **Actor observations** that contain only runtime-available desired targets.
3. **Privileged critic observations** that additionally include the FK-computed actual racket state.
4. **Goal rewards** for base repositioning before strike and racket tracking near strike.

Reward design is split cleanly:

1. **Inherited imitation reward** stays in `tracking_env_cfg.py` as the `motion_*` terms.
2. **HOPE goal reward** lives in `hope_rewards.py`.
3. **Reward weights and kernel widths** are tuned in `cfg/task/HOPEPingPong.yaml`, not by editing the Python formulas every time.

The reward formulas in `hope_rewards.py` are:

```text
racket_position_tracking_exp
racket_velocity_tracking_exp
racket_normal_tracking_exp
base_position_tracking_exp
```

These are multiplied by the timing masks from `RacketTargetCommand`, so:

1. `base_position` is active only before strike.
2. `racket_position`, `racket_velocity`, and `racket_normal` are active only in the strike window.

Domain randomization is also split cleanly:

1. Base BeyondMimic-style DR stays in the base environment config:
   - friction
   - center of mass
   - joint default position offsets
   - external pushes
2. Extra HOPE/A3 DR is controlled from `cfg/base/randomization_base.yaml` and applied by `scripts/train.py`:
   - link mass range
   - PD gain range

Important paper-alignment note:

- HITTER states PD gains are fixed / heuristic, not randomized.
- If you want to match that behavior more closely, set `task.domain_rand.pd_gain_range=null`.

Fixes already applied (verified end-to-end — the env builds headless on 16 envs and steps with finite
rewards):

1. **Racket face normal = +Y** (not the placeholder +Z). Step 11 confirmed the blade is thin along Y,
   so `mount_normal_axis=1` is set in `hope_commands.py`, `hope_env_cfg.py`, and
   `cfg/task/HOPEPingPong.yaml` (the YAML is what `train.py` applies). This resolves the old
   "confirm the face-normal axis" TODO; `sign=+1` is the red/forehand face.
2. **Self-collision OFF** (`enabled_self_collisions=False` in `robots/agibot_a3.py`). With it on, PhysX
   aborted at sim start (`free(): corrupted unsorted chunks`) because the merged wrist body carries
   overlapping collision meshes (wrist + hand + red/black blades) on the copied A3 ping-pong URDF
   asset. Re-enable only after the Isaac collision geometry is cleaned or replaced.

What you still need to do:

1. Produce or receive a clean Isaac collision asset if self-collision should be re-enabled.
2. Verify the current Agibot-transcribed PD gains, standing pose, action scale, and joint order against the real SDK/hardware.
3. Measure realistic reachable racket target ranges and write them into `cfg/task/HOPEPingPong.yaml`.
4. Verify joint-order consistency across preprocessing, training, ONNX export, and deployment.

Verification gate:

- `hope_isaac_py -c "import whole_body_tracking"` works. ✅
- The Agibot A3 tasks register before launch. ✅
- The env launches headless, rewards are finite, and sampled targets are reachable. ✅ (verified with the
  copied A3 ping-pong URDF asset; policy quality still depends on collision cleanup, hardware validation, and tuning.)

## 14. Train The WBC Policy

Training uses the Hydra entrypoint `scripts/train.py` (`task=` selects the env, `algo=ppo` selects
the PPO config). Run everything inside the GPU distrobox (`grasping`), same Python environment as
Step 12. The commands below are copy-paste: 14.1 sets up the shell, 14.2 is a ~1-minute smoke test,
14.3–14.4 are the real runs, 14.5 evaluates. (Verified end-to-end on 2026-06-22 — see the status
note at the end of this step.)

### 14.1 Set up the training shell (run once per terminal)

```bash
# from the host:
distrobox enter grasping

# inside grasping:
cd ~/workspace/HOPE/hope_training/whole_body_tracking

# Source the training env. This sets HOPE_WBT_PYTHONPATH (so Isaac's bundled python sees
# hydra/omegaconf in /opt/drone_venv + isaaclab/isaaclab_rl), defines the `hope_isaac_py` launcher,
# and exports the wandb team/org/project (WANDB_ENTITY / WANDB_REGISTRY_ORG / WANDB_PROJECT — the
# team and org are DIFFERENT; using the team for the registry fails, see Step 12.5).
source setup_train_env.sh
```

It MUST be **sourced**, not executed (`./setup_train_env.sh` runs in a subshell that exits, leaving
your shell unchanged), and re-sourced in every new terminal. On success it prints `[hope] training
env ready`. The script (`setup_train_env.sh`, in this directory) is the single source of truth for
the training env — edit it there, not inline. If you prefer not to source a file, its three pieces
(the `HOPE_WBT_PYTHONPATH` export, the `hope_isaac_py` function, and the three wandb exports) can
still be pasted by hand; the function falls back to a built-in PYTHONPATH if the export is missing.

### 14.2 Smoke test (~1 min) — confirm the whole pipeline runs

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke
```

Success = the env builds on 32 envs, then PPO iterates with finite rewards:

```text
[INFO] Task: Tracking-Flat-AgibotA3-v0 | experiment: agibot_a3_flat | log: .../logs/rsl_rl/...
...
                        Learning iteration 0/3
                       Mean reward: -0.62
...
                        Learning iteration 2/3
```

If it instead drops back to the shell with no `Learning iteration` lines, go to 14.7.

### 14.3 Train the baseline (plain motion tracking)

Train the plain tracker first — it is easier and confirms the robot/asset before adding the racket
objective. Omitting `num_envs`/`max_iterations` uses the config defaults; logs to wandb by default.

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  run_name=forehand_tracking
```

### 14.4 Train the HOPE racket task (one policy per swing)

Train forehand and backhand as separate policies (the HOPE default), each from its own clip:

```bash
# forehand
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  run_name=hope_forehand

# backhand
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand" \
  run_name=hope_backhand
```

Useful overrides (append to any command): `num_envs=4096 max_iterations=20000 seed=1`,
`task.rewards.racket_position_weight=5.0`, `task.racket.pos_x_range=[0.3,0.6]`,
`task.domain_rand.pd_gain_range=null`. **Record the wandb run ID of each good run** — Step 15 needs it.

### 14.5 Evaluate a trained policy

Run all of these **from inside `~/workspace/HOPE/hope_training/whole_body_tracking`** — the
`checkpoint=`, `motion_file=`, and `logs/` paths are relative to that directory. `play.py` always
exports `policy.onnx` next to the checkpoint (that is Step 15).

Pick the checkpoint source (in precedence order): `checkpoint=<local .pt>` > `wandb_path=<run>` >
newest local run. Pick the reference motion: `motion_file=<.npz>` (fully offline) > `registry_name=`
(downloads from wandb) > the task default. The local motion clips that training downloaded are cached
at `artifacts/hope_forehand:v0/motion.npz` and `artifacts/hope_backhand:v0/motion.npz`.

**Watch live in the Isaac Sim window** (needs a local display; the window stays open until you close it):

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="artifacts/hope_forehand:v0/motion.npz" \
  headless=false
```

**Record a video instead** (headless, no display needed; writes
`logs/rsl_rl/agibot_a3_hope/<RUN>/videos/play/play.mp4` via imageio — the terminal prints
`captured N frames` then `wrote video -> …`):

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="artifacts/hope_forehand:v0/motion.npz" \
  headless=true video=true        # video_length frames; tune in cfg/play.yaml
```

**From a wandb run** (instead of a local checkpoint; entity = your TEAM where runs are logged):

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  wandb_path="$WANDB_ENTITY/hope_wbc/<RUN_ID>" headless=false
```

`<RUN>` is the run folder under `logs/rsl_rl/agibot_a3_hope/` and `<N>` the checkpoint number
(e.g. `model_16000.pt`). `<RUN_ID>` is the last segment of the run URL
(`wandb.ai/<team>/hope_wbc/runs/<RUN_ID>`).

Two `play.py` gotchas already fixed in this repo (re-apply if you re-clone upstream BeyondMimic):
this rsl_rl/IsaacLab returns a **TensorDict** from `get_observations()` (not an `(obs, extras)`
tuple), so the rollout loop uses `obs = env.get_observations()` + `env.step(actions.to(env.device))`
(the old `obs, _ = …` unpack silently produced a 1-D action → `IndexError` in `process_action`); and
video uses a manual `env.render()` + `imageio` capture rather than `gym.wrappers.RecordVideo` (which
needs `moviepy` and was masked by Isaac's hard-exit). If video errors with `ModuleNotFoundError`, run
`hope_isaac_py -m pip install imageio imageio-ffmpeg`.

### 14.6 Where to tune

| File | What |
| --- | --- |
| `cfg/task/HOPEPingPong.yaml` | HOPE reward weights/stds, racket target ranges, strike timing |
| `cfg/task/TrackingFlat.yaml` | baseline plain-tracking task |
| `cfg/base/{env_base,sim_base,randomization_base}.yaml` | shared env count, 50 Hz timing, DR defaults |
| `cfg/algo/ppo.yaml` | rollout length, learning rate, entropy, mini-batches, epochs, network sizes |

Global CLI overrides: `num_envs max_iterations registry_name run_name seed logger log_project_name`.
Everything else is `task.<group>.<key>` or `algo.<group>.<key>` (e.g. `task.rewards.racket_position_weight=5.0`).
The legacy `scripts/rsl_rl/train.py --task=HOPE-PingPong-AgibotA3-v0 --registry_name … --headless`
entrypoint still works if you prefer argparse over Hydra.

### 14.7 Troubleshooting

- **`train.py` returns to the shell with no error, log stops right after Isaac startup.** Isaac's
  `simulation_app.close()` hard-exits the process and used to swallow the exception, making a real
  failure look like a clean exit. `scripts/train.py` now prints the traceback (flushed) before
  closing and exits non-zero, so just re-run and read the error. The usual cause is the next item.
- **`CommError: Unable to find organization for entity '<name>'`.** The `registry_name` prefix must
  be your wandb **org** (`$WANDB_REGISTRY_ORG`), never your **team** (`$WANDB_ENTITY`) — registries
  are org-scoped (Step 12.5). The `cfg/task/*.yaml` `registry_name` defaults now read
  `WANDB_REGISTRY_ORG`, so you may also omit `registry_name=...` entirely.
- **`ModuleNotFoundError: No module named 'hydra'`** (at `import hydra` near the top of `train.py` /
  `play.py`, run via `hope_isaac_py`). Your shell's `HOPE_WBT_PYTHONPATH` was empty, so `hope_isaac_py`
  launched Isaac's bundled python with an **empty `PYTHONPATH`**. hydra/omegaconf live in
  `/opt/drone_venv` and are visible only through that PYTHONPATH (Isaac's python has neither). Usually
  you opened a new terminal and never sourced the env in it. Fix: `source setup_train_env.sh` (14.1)
  in *this* terminal — it must be re-sourced per terminal. Check with `echo "$HOPE_WBT_PYTHONPATH"`
  (blank = not sourced). (Sanity check: `hope_isaac_py -c "import hydra, omegaconf;
  print(hydra.__version__)"` should print `1.3.2`.)
- **`free(): corrupted unsorted chunks` / Aborted at "Starting the simulation."** PhysX heap
  corruption from the copied A3 ping-pong URDF asset's overlapping wrist collision meshes
  (wrist + hand + red/black blades). `robots/agibot_a3.py` sets `enabled_self_collisions=False`
  to avoid it; re-enable only after the Isaac collision geometry is cleaned or replaced (Step 13).
- **`AttributeError: '…OnPolicyRunner' object has no attribute 'obs_normalizer'`** (at the first
  checkpoint save, only with `logger=wandb` — `logger=tensorboard` skips the ONNX export and hides it).
  This rsl_rl version moved obs normalization onto the policy; `my_on_policy_runner.py` and `play.py`
  now read `policy.actor_obs_normalizer` (fixed). If you re-clone the upstream BeyondMimic code and
  hit this, apply the same change.

Target metrics (a fully trained policy should reach):

```text
Tracking success rate: > 90 percent
Racket position error at strike: < 7.5 cm
Racket velocity error at strike: < 0.5 m/s
Racket normal error at strike: < 15 degrees
Base repositioning time: < 0.8 s
```

Verification gate:

- **Pipeline gate.** The env builds, PPO iterates, rewards are finite, checkpoints save, and the ONNX
  export runs. ✅ Verified 2026-06-22 — `TrackingFlat` and `HOPEPingPong` (forehand) train end-to-end
  on the copied A3 ping-pong URDF asset, including the **default `logger=wandb` path** (checkpoint save +
  `policy.onnx` export) after the `actor_obs_normalizer` fix.
- **Trained-policy gate (after a full run).** Policy does not fall, the racket reaches the target near
  strike time, forehand and backhand both work, recovery after the swing is stable.

Status / what still needs validation:

1. Training **runs today** on the copied A3 std-pingpong URDF with self-collision off (Step 13).
   Reaching target metrics still needs collision cleanup, measured reachable target ranges, and
   hardware validation of PD gains / standing height / action scale.
2. The forehand and backhand reference clips must be in the `motions` registry (Steps 9–12) so
   `registry_name=.../hope_forehand` and `.../hope_backhand` resolve.
3. Set the reachable-target and reward-tuning values in `cfg/task/HOPEPingPong.yaml` and re-train
   until the target metrics are met.
4. Record the wandb run IDs of the good forehand and backhand runs — Step 15 needs them.

### 14.8 Disk usage and cleanup

Each training run is large — the bulk is checkpoints and the tensorboard event file:

| Item | Size | Note |
| --- | --- | --- |
| `logs/rsl_rl/<exp>/<run>/` | **~300 MB/run** | the big consumer |
| └ `model_*.pt` | ~7 MB each (~224 MB at `save_interval=500`, 16k iters) | most of it |
| └ `events.out.tfevents.*` | ~75 MB | grows with iterations |
| └ `*.onnx` | ~2 MB | the exported policy |
| `hope_training/wandb/` | ~230 MB | local cache, already synced to the wandb cloud |
| `motions/`, `artifacts/`, `outputs/` | ~4 MB / ~0.5 MB / <1 MB | small |

**If a run went badly, these are safe to delete** (run from `~/workspace/HOPE/hope_training/whole_body_tracking`):

```bash
# 1) drop the whole failed run (you are re-training from scratch) — frees ~300 MB
rm -rf logs/rsl_rl/agibot_a3_hope/<RUN>

# 2) wandb local cache — frees ~230 MB; safe because runs are synced to the wandb cloud
rm -rf wandb/run-* ../wandb/run-*

# 3) Hydra job logs — a few hundred KB, optional
rm -rf outputs/*
```

**If instead you want to warm-start the next run from a good checkpoint**, keep that one `.pt` and
delete only the redundant checkpoints + the event file:

```bash
RUN=logs/rsl_rl/agibot_a3_hope/<RUN>
find "$RUN" -maxdepth 1 -name 'model_*.pt' ! -name 'model_<N>.pt' -delete   # keep model_<N>.pt
rm -f "$RUN"/events.out.tfevents.*
```

**Never delete** `motions/` (the Step 9–12 retargeting products) or `artifacts/hope_*:v0/motion.npz`
(the reference clips) — small, but expensive to regenerate / re-download. All of `logs/`, `outputs/`,
`wandb/`, `artifacts/`, `motions/` are gitignored, so deleting them never affects git.

To produce fewer files in the first place, raise `save_interval` in `cfg/algo/ppo.yaml`
(`500 → 2000` cuts the checkpoints ~4×; the final policy is unaffected).

## 15. Export The Policy To ONNX

Execution scope:

- Stay inside the GPU distrobox and the same Python environment used for training.

The preferred export path is now built into `scripts/play.py`. When you evaluate a trained run,
the script loads the checkpoint and exports `policy.onnx` next to it under the run's `exported/`
directory. You do not need a separate export script for the normal workflow.

Export and evaluate:

```bash
# inside the GPU / Isaac distrobox
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  wandb_path=$WANDB_ENTITY/hope_wbc/PLACEHOLDER_RUN_ID
```

If you want a stable deployment filename, copy the exported file into `hope_training/policies/`
after each export:

```bash
mkdir -p ~/workspace/HOPE/hope_training/policies
cp /path/to/exported/policy.onnx \
  ~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx
```

Run ID lookup:

1. Open the successful training run in WandB.
2. Copy the run ID from the URL or from the run overview page.
3. Run `scripts/play.py` once for the forehand run and once for the backhand run.
4. Copy or rename the exported files to stable names such as `hope_forehand_policy.onnx` and `hope_backhand_policy.onnx`.
5. Keep each ONNX file and the exact joint-order YAML together. They are a matched pair.

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

## Phase 5 — Runtime Policy and Hardware Deployment

## 16. Run Sim-To-Sim Verification

Before touching hardware, verify the ONNX policy in simulation.

Execution scope:

- From this step onward, switch back to the ROS distrobox: `distrobox enter hope`.
- The launch commands below are ROS 2 commands, so they belong in `hope`, not in the GPU distrobox.

```bash
# inside the ROS distrobox: hope
ros2 launch motion_tracking_controller mujoco.launch.py \
  policy_path:=~/workspace/HOPE/hope_training/policies/hope_forehand_policy.onnx
```

Then test backhand:

```bash
# inside the ROS distrobox: hope
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

Execution scope:

- Run the software commands in this step inside `distrobox enter hope`.
- The robot cabling, Ethernet setup, E-stop access, and physical safety checks happen outside the container on the real hardware setup.

Study Agibot's own deployment pattern first. Agibot's open X1 inference stack is the closest public reference for how an Agibot humanoid runs a learned policy:

- `github.com/AgibotTech/agibot_x1_infer` — C++ deployment node built on **AimRT** middleware, **ROS 2 Humble**, running the policy via **ONNX Runtime**. It subscribes `/joint_states`, publishes `/joint_cmd`, runs the **PD loop at ~1 kHz** and the **policy at ~100 Hz** (decimation 10), and is driven by a state machine (`pd_idle`, `pd_stand`, `rl_walk_*`, ...). HOPE's WBC policy runs at **50 Hz**, so adjust the decimation accordingly.
- `github.com/AgibotTech/agibot_x1_train` — the matching training repo (Isaac Gym + MuJoCo sim2sim; PT→ONNX/JIT export).
- `x2-aimdk.agibot.com` — the published **AimDK_X2** developer SDK docs (ROS 2 interfaces, robot specs). Assume the A3 has an analogous SDK and confirm with Agibot.

Caveat: the X1 stack is **locomotion-only** and uses **ROS 2 Humble**, while HOPE targets ROS 2 Jazzy and needs whole-body motion tracking. Reuse its **plumbing** (AimRT + ONNX Runtime, `/joint_states`/`/joint_cmd`, the 1 kHz-PD/decimated-policy split, the state machine), not its walking algorithm.

Create or add Agibot deployment packages:

```bash
# inside the ROS distrobox: hope
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
# inside the ROS distrobox: hope
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
# host terminal or inside the ROS distrobox; use whichever shell is attached to the robot network
ip -br addr
```

4. Find or set the robot IP:

```bash
# host terminal or inside the ROS distrobox; use whichever shell is attached to the robot network
# Show neighbors on the local Ethernet network after the robot is connected.
ip neigh

# Or, if your network uses ARP tools:
arp -a
```

5. Confirm that the A3 controller is reachable:

```bash
# host terminal or inside the ROS distrobox; use whichever shell is attached to the robot network
ping PLACEHOLDER_A3_ROBOT_IP
```

Run real control only after sim-to-sim passes:

```bash
# inside the ROS distrobox: hope
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

## Phase 6 — Portability and Full-System Integration

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

Execution scope:

- Open four terminals, and in each one run `distrobox enter hope` first.
- Then launch the ROS 2 commands below inside that `hope` shell.
- The Avatar-Pro software itself still runs on the separate mocap PC, outside this machine.

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

## Phase 7 — Testing, Safety, and Competition Readiness

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

## Phase 8 — Summaries, Constraints, References, and Open Values

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

### 30.4 Agibot Expedition A3 robot (current sources and remaining verification)

| Value | Where it will be used | How to get it |
|-------|----------------------|---------------|
| A3 URDF/MJCF/USD + meshes | GMR retarget, sim training, FK | Current source: `agi/URDF/` and `agi/A3_MuJoCo_Sim/`; request updated assets from Agibot when they change. |
| Joint name list + joint order | planner->WBC, ONNX export, bridge | Current working source: `hope_training/config/joint_order_agibot_a3.yaml`; verify against hardware SDK before real commands. |
| Joint limits, link inertials, gear ratios | sim fidelity, safety limits | Current source: Agibot URDF/MJCF materials; verify when model revisions arrive. |
| Default PD gains / impedance | WBC training + `agibot_hardware_bridge` | Current training source: Agibot deploy transcription in `robots/agibot_a3.py`; verify on hardware before increasing gains. |
| `base_link` name + standing height | world-frame FK chain | Current sim source: A3 URDF and `robots/agibot_a3.py`; still measure/verify against the real robot. |
| Control SDK / AimDK / ROS 2 API, joint command + feedback topics | `agibot_hardware_bridge` | From Agibot (analogous to AimDK_X2). Confirm ROS 2 distro (X1 uses Humble; HOPE targets Jazzy). |
| E-stop / soft-stop / standby interface | safety, `agibot_bringup` | From the Agibot manual; verify the 200 ms stop requirement. |

### 30.5 Racket mount and WBC/deploy (later phases)

| Value | Where | How to get it |
|-------|-------|---------------|
| `T_mount` (wrist→racket fixed transform) | robot model + FK + reward (step 11) | Physically measure the 3D-printed bracket (translation to racket center, face-normal orientation, blade roll); add as a fixed joint. |
| WandB entity / run ids | training + ONNX export (steps 12–15) | Your own WandB org/run after training. |
| ONNX policy paths | sim2sim + deploy (steps 16–18) | Output of `export_onnx.py` after a successful training run. |
| Robot IP + network interface | deploy (step 18) | From the Agibot network setup; `ip addr` on the control PC. |
