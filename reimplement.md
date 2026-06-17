# HOPE Reimplementation Guide

This guide explains how to reimplement the HOPE reference system from the documentation in this repository. The repo is a design-document repository, not a complete runnable software stack. A successful reimplementation means you will create your own ROS 2 packages, training configuration, policy export, and deployment integration while following the reference architecture and competition constraints.

Read this file from top to bottom. Do not skip the verification gates. Each later phase assumes the earlier phase is already working.

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

3. `HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`
   - Understand the world frame, tracked objects, table setup, and ROS 2 mocap interface.

4. `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`
   - This is your planner specification.
   - It contains enough algorithmic detail to implement Stages 1-3.

5. `HOPE_WBC_Simulation_Training_Reference_Setup.md`
   - This is your WBC training specification.
   - It explains motion acquisition, retargeting, preprocessing, RL training, and ONNX export.

6. `HOPE_Hardware_Deployment_Reference_Setup.md`
   - This is your deployment specification.
   - It explains ROS 2 node graph, launch order, G1 deployment, A3 deployment, and safety workflow.

Verification gate:

- You can explain why the racket is never tracked by motion capture.
- You can draw the data flow from motion capture to planner to WBC to robot.
- You know whether you are targeting Unitree G1, G1 EDU, Agibot A3, or another humanoid.

## 2. Decide Your Target Platform

Do this before creating software packages.

1. Choose the robot.
   - Recommended first target: Unitree G1.
   - Alternative target: Agibot A3, if you have access to its robot model, SDK, joint names, and low-level control API.

2. Record the robot facts you will need.
   - Robot model.
   - Total controlled DOF.
   - Arm DOF.
   - Waist DOF.
   - Exact URDF or MJCF file.
   - Exact base_link name.
   - Physical location of base_link.
   - Standing base_link height.
   - Joint names and joint order.
   - Joint limits.
   - Default PD gains.
   - Control interface.
   - Middleware: ROS 2, AimRT, vendor SDK, or bridge.

3. Choose the simulation backend.
   - For Unitree G1: use Isaac Lab plus PhysX and USD assets.
   - For Agibot A3 or robots with MJCF assets: use mjlab plus MuJoCo Warp.

4. Choose the deployment path.
   - For Unitree G1: use `motion_tracking_controller`, `legged_control2`, and `unitree_bringup`.
   - For Agibot A3: use AimRT with a ROS 2 bridge first, then consider native AimRT only after the ROS 2 path works.

Verification gate:

- You have a robot model file and know the joint order expected by your controller.
- You know how to send joint position commands safely to the robot.
- You know how to stop the robot immediately.

## 3. Create A Workspace Layout

Use separate repositories or folders for each major subsystem. Do not mix external upstream repositories into this documentation repo.

Recommended layout:

```text
~/hope_ws/
  src/
    hope_msgs/
    hope_planner/
    hope_bringup/
    hope_monitoring/
    motion_capture_tracking/        # installed by apt or cloned if needed
    motion_tracking_controller/      # for WBC deployment
    unitree_bringup/                 # G1 only

~/hope_training/
  GVHMR/
  GMR/
  whole_body_tracking/
  mjlab/                             # A3 or MJCF path only
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

- Use ROS 2 Jazzy on Ubuntu 24.04 if possible.
- ROS 2 Humble on Ubuntu 22.04 is allowed by the rules for external communication, but the reference docs use Jazzy.

Verification gate:

- `ros2 --version` works.
- You can create and build an empty colcon workspace.
- You can source your workspace with `source install/setup.bash`.

## 4. Implement The Shared ROS 2 Messages

Create a package:

```bash
cd ~/hope_ws/src
ros2 pkg create hope_msgs --build-type ament_cmake
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
cd ~/hope_ws
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

Implementation rules:

1. Use meters.
2. Use ROS 2 REP 103.
3. Use Z-up.
4. Do not silently convert to table-center coordinates.
5. If you must support another frame internally, write explicit conversion functions and unit tests.

Verification gate:

- A table corner marker at the origin reports approximately `[0, 0, 0]`.
- A point at the opponent half center reports approximately `[2.055, -0.7625, 0]`.
- The ball height above the table is positive.

## 6. Set Up Motion Capture

Hardware requirements:

1. Use at least 6 cameras.
2. Prefer 8-12 cameras.
3. Use at least 120 Hz.
4. Prefer 240-360 Hz for fast ball tracking.
5. Cover the full table plus at least 1.5 m margin on each player's side.

For OptiTrack Motive:

1. Open Motive.
2. Go to Data Streaming or Edit -> Settings -> Streaming.
3. Enable NatNet.
4. Set transmission to unicast if possible.
5. Set Up Axis to `Z Axis`.
6. Enable rigid bodies.
7. Enable unlabeled markers.
8. Disable labeled markers unless you need them.
9. Disable skeletons.
10. Confirm command port 1510 and data port 1511.

Create the table rigid body:

1. Attach at least 4 asymmetric reflective markers to the outer table frame.
2. Do not place markers on the playing surface.
3. Name the table rigid body `PPT`.
4. Set the rigid body pivot to the near-side left corner of the table surface.
5. Align the PPT local frame with the HOPE world frame.
6. Confirm the stationary PPT pose is approximately identity.

Create robot base_link rigid bodies:

1. Add at least 4 asymmetric markers to a rigid torso or pelvis plate.
2. Name the P1 robot rigid body `P1`.
3. Name the P2 robot rigid body `P2`.
4. Measure the static transform from the marker cluster frame to the robot URDF base_link.
5. Publish that transform with `static_transform_publisher`.

Prepare the ball:

1. Use an official 40+ table-tennis ball.
2. Add one small retroreflective marker or reflective coating.
3. Keep added mass as low as possible.
4. Do not use multiple ball markers.
5. Track it as a single unlabeled marker.

Install the recommended ROS 2 bridge:

```bash
sudo apt install ros-jazzy-motion-capture-tracking
```

Configure `motion_capture_tracking`:

```yaml
type: "optitrack"
hostname: "MOTIVE_PC_IP"

robot_types:
  ball:
    motion_capture:
      tracking: "librigidbodytracker"
      initial_position: [1.37, -0.7625, 0.2]
      dynamics:
        max_velocity: 10.0
```

Expected topics:

```text
/poses    geometry_msgs/PoseArray    120-360 Hz
/tf       tf2_msgs/TFMessage         120-360 Hz
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
- The ball is not confused with table or robot markers.
- No racket, wrist, hand, or paddle marker exists.

## 7. Implement The Planner Package

Create the package:

```bash
cd ~/hope_ws/src
ros2 pkg create hope_planner --build-type ament_python \
  --dependencies rclpy geometry_msgs std_msgs diagnostic_msgs hope_msgs
```

Recommended files:

```text
hope_planner/
  hope_planner/
    __init__.py
    constants.py
    ball_state_estimator.py
    ball_trajectory_predictor.py
    racket_target_planner.py
    quaternion_utils.py
    node.py
  config/
    hope_planner.yaml
  launch/
    hope_planner.launch.py
  test/
    test_ball_state_estimator.py
    test_ball_trajectory_predictor.py
    test_racket_target_planner.py
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

Example config:

```yaml
hope_planner:
  ros__parameters:
    ball_rigid_body_name: "pingpong_ball"
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
cd ~/hope_ws
colcon build --symlink-install --packages-select hope_msgs hope_planner
source install/setup.bash
ros2 launch hope_planner hope_planner.launch.py
ros2 topic hz /racket/command
ros2 topic echo /racket/command --once
```

Verification gate:

- Tossing a ball toward P1 causes `/racket/command` to publish valid commands.
- A ball moving away from P1 produces no valid strike command.
- `time_to_strike` is positive and decreases as the ball approaches.
- Planner runtime is below 5 ms per update.

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
cd ~/hope_training
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
cd ~/hope_training
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

For Unitree G1:

1. Use GMR's `unitree_g1` target if available.
2. Confirm the output joint order matches BeyondMimic's G1 order.
3. Retarget forehand.
4. Retarget backhand.
5. Export to CSV or PKL as expected by the preprocessing script.

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

Verification gate:

- The simulated racket frame matches the physical mount.
- FK from base_link and joint states returns the expected racket center.
- Racket normal direction is consistent with the planner normal.

## 12. Preprocess Motions For BeyondMimic

For the G1 Isaac Lab path, install BeyondMimic:

```bash
cd ~/hope_training
git clone https://github.com/HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
```

Install Isaac Sim and Isaac Lab:

1. Install Isaac Sim 4.5.0.
2. Install Isaac Lab 2.1.0.
3. Use Python 3.10.
4. Prefer a machine with an NVIDIA RTX 4090 or better.

Download the G1 assets used by BeyondMimic:

```bash
curl -L -o unitree_description.tar.gz \
  https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz

tar -xzf unitree_description.tar.gz \
  -C source/whole_body_tracking/whole_body_tracking/assets/

rm unitree_description.tar.gz
```

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
export WANDB_ENTITY=your-org-name
```

Convert retargeted motions:

```bash
python scripts/csv_to_npz.py \
  --input_file ~/hope_training/motions/retargeted/forehand_swing.csv \
  --input_fps 30 \
  --output_name hope_forehand \
  --headless

python scripts/csv_to_npz.py \
  --input_file ~/hope_training/motions/retargeted/backhand_swing.csv \
  --input_fps 30 \
  --output_name hope_backhand \
  --headless
```

Replay preprocessed motions:

```bash
python scripts/replay_npz.py \
  --registry_name=your-org-name-org/wandb-registry-motions/hope_forehand

python scripts/replay_npz.py \
  --registry_name=your-org-name-org/wandb-registry-motions/hope_backhand
```

Verification gate:

- The G1 model replays each motion.
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
  --task=Tracking-Flat-G1-v0 \
  --registry_name your-org-org/wandb-registry-motions/hope_forehand \
  --headless \
  --logger wandb \
  --log_project_name hope_wbc \
  --run_name forehand_tracking
```

HOPE target-tracking command:

```bash
python scripts/rsl_rl/train.py \
  --task=HOPE-PingPong-G1-v0 \
  --registry_name your-org-org/wandb-registry-motions/hope_forehand \
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
cat source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py
```

Evaluate:

```bash
python scripts/rsl_rl/play.py \
  --task=HOPE-PingPong-G1-v0 \
  --num_envs=2 \
  --wandb_path=your-org/hope_wbc/run_id
```

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
cd ~/hope_ws/src
git clone https://github.com/HybridRobotics/motion_tracking_controller.git
```

Export ONNX:

```bash
cd ~/hope_ws/src/motion_tracking_controller
python scripts/export_onnx.py \
  --wandb_path=your-org/hope_wbc/run_id \
  --output_path=~/hope_training/policies/hope_forehand_policy.onnx
```

Export both policies:

```bash
python scripts/export_onnx.py \
  --wandb_path=your-org/hope_wbc/forehand_run_id \
  --output_path=~/hope_training/policies/hope_forehand_policy.onnx

python scripts/export_onnx.py \
  --wandb_path=your-org/hope_wbc/backhand_run_id \
  --output_path=~/hope_training/policies/hope_backhand_policy.onnx
```

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
  policy_path:=~/hope_training/policies/hope_forehand_policy.onnx
```

Then test backhand:

```bash
ros2 launch motion_tracking_controller mujoco.launch.py \
  policy_path:=~/hope_training/policies/hope_backhand_policy.onnx
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

Important:

- Do not change observation ordering after export.
- Do not change joint ordering after export.
- Do not switch policies in the middle of a strike unless your training supports that.

Verification gate:

- A target on the forehand side selects the forehand model.
- A target on the backhand side selects the backhand model.
- Switching does not cause a discontinuous joint command spike.

## 18. Deploy On Unitree G1

Install G1 deployment dependencies:

```bash
# Install ROS 2 Jazzy first.

echo "deb [trusted=yes] https://github.com/qiayuanl/unitree_buildfarm/raw/noble-jazzy-amd64/ ./" \
  | sudo tee /etc/apt/sources.list.d/qiayuanl_unitree_buildfarm.list

echo "yaml https://github.com/qiayuanl/unitree_buildfarm/raw/noble-jazzy-amd64/local.yaml jazzy" \
  | sudo tee /etc/ros/rosdep/sources.list.d/1-qiayuanl_unitree_buildfarm.list

sudo apt-get update
sudo apt-get install ros-jazzy-unitree-description
sudo apt-get install ros-jazzy-unitree-systems
```

Clone packages:

```bash
cd ~/hope_ws/src
git clone https://github.com/qiayuanl/unitree_bringup.git
git clone https://github.com/HybridRobotics/motion_tracking_controller.git
```

Install dependencies and build:

```bash
cd ~/hope_ws
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  --packages-up-to unitree_bringup

colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  --packages-up-to motion_tracking_controller

source install/setup.bash
```

Connect to G1:

1. Connect PC to G1 via Ethernet.
2. Set PC static IP to `192.168.123.11`.
3. Find network interface:

```bash
ifconfig
```

4. Ping robot:

```bash
ping 192.168.123.1
```

Run real controller only after sim-to-sim passes:

```bash
ros2 launch motion_tracking_controller real.launch.py \
  network_interface:=enp3s0 \
  policy_path:=~/hope_training/policies/hope_forehand_policy.onnx
```

G1 joystick controls from the reference doc:

```text
L1 + A: standby controller
R1 + A: activate motion tracking controller
B: emergency stop
```

Verification gate:

- The robot enters standby safely.
- The emergency stop works.
- Joint states are streaming.
- The policy can be activated in a controlled, no-ball test.

## 19. Deploy On Agibot A3

Use this path only if you have access to A3 hardware, robot model, SDK, and control interface.

Recommended path:

1. Use ROS 2 bridge through AimRT.
2. Keep the HOPE planner in ROS 2.
3. Bridge `/racket/command`, `/P1/pose`, and joint feedback into AimRT.
4. Run ONNX inference in a controller adapted to A3 joint names and joint order.
5. Send joint position commands through the A3 actuator bus.

Install AimRT:

```bash
cd ~/hope_ws/src
git clone https://github.com/AimRT/aimrt.git
cd aimrt
mkdir build
cd build
cmake .. -DAIMRT_BUILD_WITH_ROS2=ON
make -j$(nproc)
sudo make install
```

A3-specific implementation tasks:

1. Confirm base_link convention.
2. Convert or obtain the robot MJCF/URDF.
3. Add A3 to GMR.
4. Add A3 to mjlab if training there.
5. Define A3 joint names and order.
6. Define A3 PD gains.
7. Define A3 `T_mount`.
8. Adapt the ONNX observation builder.
9. Adapt the action-to-joint-command mapping.
10. Add AimRT or ROS 2 bridge launch files.

Verification gate:

- A3 sim policy works before real hardware.
- AimRT bridge passes timestamps and frame IDs correctly.
- E-stop works through the A3 safety stack.

## 20. Integrate The Full ROS 2 Pipeline

Launch order:

```bash
# Terminal 1: motion capture bridge
ros2 launch motion_capture_tracking optitrack.launch.py \
  server_ip:=192.168.1.100

# Terminal 2: planner
ros2 launch hope_planner hope_planner.launch.py

# Terminal 3: WBC controller
ros2 launch motion_tracking_controller real.launch.py \
  network_interface:=enp3s0 \
  policy_path:=~/hope_training/policies/hope_forehand_policy.onnx

# Terminal 4: monitoring
ros2 topic hz /poses /tf /racket/command /joint_states
```

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
6. Do not assume G1 USD assets from different sources are interchangeable.
7. Do not assume A3 details until you confirm them from Agibot documentation or hardware.
8. Do not change observation or joint ordering after ONNX export.
9. Do not use Reliable QoS for high-rate real-time mocap topics unless you have measured that latency remains acceptable.
10. Do not tune multiple sim-to-real variables at once.

## 29. References In This Repository

- `README.md`
- `HOPE_AI_Challenge_2026_Rules_EN.docx`
- `HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`
- `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`
- `HOPE_WBC_Simulation_Training_Reference_Setup.md`
- `HOPE_Hardware_Deployment_Reference_Setup.md`

