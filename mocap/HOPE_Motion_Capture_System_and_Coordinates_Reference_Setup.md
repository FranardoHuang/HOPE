# Motion Capture System Reference Setup for HOPE Ping-Pong Arena

**v0.6** — 2026-07-20

> **Preserved reference document.** This predates the current HOPE
> stack and is kept for arena-build background and provenance. The authoritative
> frame and topic contract is [`mocap/README.md`](README.md); the live driver
> shipped in this repo is the vendored
> [`hope_ws/src/vrpn_mocap/`](../hope_ws/src/vrpn_mocap). Index:
> [`REFERENCE_DOCS.md`](../REFERENCE_DOCS.md).

---

## 1  Compatible Motion Capture Systems

This reference design document creates a reference system compatible with several mainstream motion capture brands — principally **OptiTrack**, **Vicon**, and **青瞳视觉 (CHINGMU)** — and is expected to extend to the other marker-based brands supported by the `motion_capture_tracking` library, including Qualisys, NOKOV, VRPN, FZMotion, and Motion Analysis. These systems differ in their cameras and vendor software — OptiTrack pairs Motive with the NatNet protocol, Vicon uses Vicon Tracker, and Chingmu uses CMTracker/CMAvatar streaming over VRPN, TrackD, DTrack, OpenVR, and its native LiveStream, each shipping C/C++, Python, and ROS SDKs — but this design unifies them under a single ROS 2 REP 103 coordinate frame and `/poses` + `/tf` topic interface. During competition the arena streams the named **6-DOF rigid bodies** `Ball`, `P1`, and `P2` (the shipped bringup aggregates only `Ball` into `/poses` by default, so the default `/poses` array has a single pose). A fourth asset, `Table`, is defined for arena setup/calibration only and appears only in training-data recordings — it is **not** tracked or reported during competition. In the current reference path, **both OptiTrack and Chingmu stream these rigid bodies over VRPN** into the same vendored ROS 2 client; Section 6 covers the uniform path (OptiTrack specifics in Section 6.2, Chingmu in Section 6.3).

For the HOPE reference design, the minimum specification is:

- At least **8 cameras**, arranged to cover the full table volume plus a 1.5 m margin on each player's side
- Camera frame rate **≥ 300 Hz** (competitive ball tracking at speeds exceeding 5 m/s)
- Sub-millimeter reconstruction accuracy within the tracking volume
- The ball **must** be provided with stable rigid-body modeling and tracking — a vendor-qualified `Ball` rigid-body asset (Section 5) with verified high-speed tracking, occlusion recovery, and ID stability. Single-point / unlabeled-marker ball tracking does not meet this reference design.

---

## 2  Setup of the Environment Markers and Coordinate Frames

To avoid calibration error and potential platform movements, the most straightforward approach is to anchor the motion capture system origin directly on the `Table` rigid body (older notes may call it PPT). However, a common source of confusion is that the default coordinate frame in OptiTrack (Y-up) differs from both ROS 2 (Z-up, REP 103) and Vicon (Z-up). **In this reference design, we adopt the ROS 2 REP 103 convention as the canonical world frame.**

### 2.1  Canonical World Frame (ROS 2 REP 103)

The world frame origin is placed at the **near-side left corner of the table surface**, from Player One's (P1's) perspective:

| Axis | Direction | Range on table surface |
|------|-----------|------------------------|
| **X** | Forward — toward Player Two (P2) along the table length | 0 → +2.74 m |
| **Y** | Left — along the table width, from P1's perspective | 0 → −1.525 m |
| **Z** | Up — vertical | 0 = table surface |

This convention is **identical** to the frame used in the companion document *HOPE 7DOF Racket Model-based Planner Reference Setup*, ensuring that all ball trajectory predictions, racket target computations, and ROS 2 topic messages share a single consistent coordinate system.

Key landmarks in this frame:

| Landmark | X (m) | Y (m) | Z (m) |
|----------|-------|-------|-------|
| Origin (P1 near-side left corner) | 0.0 | 0.0 | 0.0 |
| Net center line | 1.37 | −0.7625 | 0.0 |
| P1 half center | 0.685 | −0.7625 | 0.0 |
| P2 half center | 2.055 | −0.7625 | 0.0 |
| Floor directly below origin | 0.0 | 0.0 | −0.76 |
| Virtual hitting plane (planner) | x = x_hit ≈ 0.0 | — | — |

The table surface occupies the region: `x ∈ [0, 2.74]`, `y ∈ [−1.525, 0]`, `z = 0`.

### 2.2  Correcting OptiTrack's Default Coordinate Frame

OptiTrack Motive defaults to a **Y-up** coordinate system, which is incompatible with ROS 2's Z-up convention. To correct this:

1. In Motive, navigate to **Edit → Settings → Streaming** (or open the Data Streaming pane).
2. Under **Advanced Network Options**, change **Up Axis** from "Y Axis" to **"Z Axis"**. **Note:** whether this setting also governs Motive's **VRPN** stream (the current HOPE reference path, Section 6.2) is not clearly specified by OptiTrack's documentation and can differ by Motive version and installation — Y-up VRPN output is commonly observed, but must not be assumed. Determine the VRPN stream's actual frame empirically at surveyed table landmarks (Section 6.5) and apply the full-pose frame conversion of Section 6.4 **only if** the stream is verified Y-up (required engineering — see Section 6.2).
3. Orient the calibration ground plane so that the calibration square's long edge aligns with the desired X-axis direction (toward P2). This sets the world frame orientation during the calibration wand procedure.

Vicon Tracker defaults to Z-up and generally requires no axis correction. However, verify during ground-plane calibration that the X-axis points along the table length toward P2.

For **青瞳 (Chingmu) CMTracker**, the world frame is fixed by the L-frame / calibration-square placement during the ground-plane calibration step, and the up axis is configurable in the streaming/export settings. Set the up axis to **Z** so that streamed data matches the ROS 2 REP 103 convention, and place the calibration square so its long edge points along the table length toward P2. If a particular CMTracker installation can only stream in a Y-up or otherwise non-REP-103 frame, do **not** attempt to re-calibrate around it — instead apply the fixed axis conversion in the ROS 2 bridge node described in Section 6.4.

### 2.3  Table Rigid Body Definition (`Table`; legacy `PPT` name)

Reflective markers or retroreflective patches (at least 10 mm × 10 mm) are attached to the **outer frame** of the table. Collectively, these markers form one rigid body defined in Motive or CMTracker as the asset **`Table`**. Older arena notes may call this same asset `PPT` (Ping-Pong Table); `Table` is the canonical asset name in setup sessions and training-data recordings. **The `Table` asset is a setup/calibration tool only — it is not streamed or reported during competition.**

Placement requirements:

- Attach **at least 4 markers** in an asymmetric configuration on the table frame's outer edges.
- Place markers where they are visible from the majority of camera positions and will not be occluded by players, the net, or the ball during play.
- **Do not place markers on the playing surface** — they would interfere with ball bounce dynamics and may degrade rigid-body identification.

The `Table` rigid body's pivot point must be set to the **near-side left corner of the table surface** (the origin), with the body's local frame aligned with the world frame axes defined above. After calibration, the `Table` rigid body should report identity pose (position ≈ [0, 0, 0], orientation ≈ [0, 0, 0, 1]) when the table is stationary and properly aligned.

The `Table` rigid body serves two purposes:

1. **Origin anchor** — It defines the world frame origin for all other tracked objects.
2. **Movement verification between sessions** — during setup or verification sessions, a `Table` pose deviating from identity indicates the table was bumped or shifted and the arena needs re-calibration. During competition the table is treated as a static, surveyed world origin: no live `Table` stream exists, so any suspected shift is handled by re-running the verification, not by a runtime topic.

---

## 3  Tracked Object Taxonomy

During competition the motion capture system streams the named rigid bodies **`Ball`, `P1`, and `P2`**. The `Table` asset exists only for setup/calibration and in training-data recordings (Section 2.3). The racket/paddle is explicitly tracked by **nothing**, ever.

### 3.1  Racket Exclusion Policy — Paddle Is NOT Tracked by Motion Capture

**The motion capture system must not track the ping-pong racket (paddle).** No reflective markers or tracking assets should be placed on or attached to the racket. This is a deliberate architectural decision aligned with the HOPE competition design:

**Rationale:**

1. **Forward kinematics inference.** The humanoid must infer its paddle's 6-DOF pose (position and orientation) from its own proprioceptive state — joint encoder readings plus the tracked `base_link` position — using forward kinematics through its arm kinematic chain. This tests the robot's internal body model accuracy, which is a core competency for any real-world manipulation task.

2. **No external sensing of end-effector.** In this architecture, the whole-body controller (WBC) receives a desired racket state `(p_intercept, v_racket, n_racket, t_strike)` from the planner and uses its RL policy to drive the 7-DOF arm to achieve that state. The controller never receives measured racket pose from the motion capture system. The racket's actual position is an emergent property of the robot's joint configuration, not an externally measured quantity.

3. **Competition fairness.** Tracking the racket externally would provide closed-loop feedback that bypasses the robot's control challenge. The HOPE competition requires each team's humanoid to demonstrate autonomous paddle control through its own kinematic model.

4. **Practical reliability.** Markers on a rapidly swinging paddle (arm speeds exceeding 3 m/s) suffer from severe occlusion, motion blur, and centripetal marker detachment. Excluding the paddle from tracking eliminates a fragile sensing link.

**Enforcement:** During competition setup, referees verify that no retroreflective material is present on the racket, the robot's hand, or the wrist link beyond the last tracked rigid-body marker on the robot's torso/pelvis.

**Cross-references:** The companion *HOPE 7DOF Racket Model-based Planner Reference Setup* (Section 0.1) documents that the planner outputs a desired racket state without any racket pose feedback. The companion *HOPE WBC Simulation Training Reference Setup* (Section 2.8 — Racket Mount Kinematics) documents the complete FK chain from `base_link` through the 7-DOF arm to the 3D-printed fixed racket mount, including the `T_mount` calibration procedure that ensures the simulation model matches the physical bracket.

### 3.2  Tracked Objects Summary

| Object ID | Asset type | What is tracked | Markers | Tracking mode |
|-----------|-----------|-----------------|---------|---------------|
| **Table** | Rigid body (setup/calibration only — **not streamed in competition**; poses appear only in training data) | Ping-pong table frame and world origin | ≥ 4 asymmetric on table outer frame | Vendor 6-DOF |
| **P1** | Rigid body (vendor-tracked) | Player 1 humanoid `base_link` | ≥ 4 asymmetric on torso/pelvis plate | Vendor 6-DOF |
| **P2** | Rigid body (vendor-tracked) | Player 2 humanoid `base_link` | ≥ 4 asymmetric on torso/pelvis plate | Vendor 6-DOF |
| **Ball** | Rigid body (vendor-tracked) | Ping-pong ball center pose | Vendor-qualified rigid-body pattern/constellation | Vendor 6-DOF |

No other objects should carry unregistered retroreflective patterns within the tracking volume during play. Give every rigid body a unique asymmetric signature and stable asset name so the vendor solver cannot swap asset identities.

---

## 4  Setup of the Humanoid base_link Markers

In this reference design, the humanoid infers its paddle's 6-DOF pose using **forward kinematics from `base_link`** through the arm's kinematic chain. Therefore, the only spatial anchor the motion capture system provides for each robot is its `base_link` location.

### 4.1  base_link Convention — General Principles

There is no universal standard for where a humanoid robot's `base_link` is defined. The convention varies by manufacturer, URDF authoring choices, and the robot's intended control architecture. However, three common patterns have emerged across the industry:

**Pattern A — Pelvis root (most common for bipedal locomotion).** The `base_link` is the pelvis link, located at the center of the hip plate where the leg kinematic chains branch downward and the torso chain branches upward. This is a common choice for RL-trained locomotion controllers because the pelvis is the most stable reference during walking — it is the floating-base frame in whole-body dynamics. The Unitree G1, Unitree H1, Boston Dynamics Atlas, and Agility Digit are examples of this pattern.

**Pattern B — Torso/chest root.** Some platforms place `base_link` at the upper torso or chest, above the waist joint(s). This is less common for bipedal locomotion (the pelvis is more dynamically stable) but can appear in manipulation-focused configurations where the arms are the primary concern and the legs are treated as a mobile base subsystem.

**Pattern C — Waist joint root.** A compromise where `base_link` sits at the waist joint itself — the interface between legs and torso. In many simple designs this is co-located with the pelvis origin (Pattern A). In robots with multi-DOF waist articulation, the waist joint is above the pelvis, and choosing it as `base_link` places the root between the two subsystems.

**For the HOPE competition, the critical requirement is:**

> The `base_link` must be the root of the forward kinematics chain that reaches the paddle-holding hand. The planner outputs a desired racket state in the world frame; the robot's WBC must compute the arm joint trajectory from `base_link` to the paddle that achieves it.

This means the complete FK chain is: `world → base_link (from mocap) → waist joints → shoulder → elbow → wrist → paddle tip (from joint encoders)`. Every joint between `base_link` and the paddle must be instrumented with encoders whose readings are available to the robot's control software.

### 4.2  Unitree G1

The Unitree G1 is one robot-specific integration example; the same registration and frame requirements apply to every participating humanoid.

| Property | Value |
|----------|-------|
| `base_link` location | **Pelvis** — center of lower torso at the waist, approximately at the intersection of the two hip yaw joint axes |
| Pattern | A (pelvis root) |
| Standing pelvis height | ~0.78 m above floor (z ≈ +0.02 m in HOPE frame) |
| Robot overall height | 1.27–1.32 m |
| Weight | ~35 kg with battery |
| Total DOF | 23 (base) to 43 (EDU with dexterous hands) |
| Arm DOF | 7 per arm |
| Waist DOF | 1 (yaw) |
| URDF source | `github.com/unitreerobotics/unitree_ros` → `robots/g1_description` |
| Middleware | ROS 2 natively supported |

The kinematic tree branches from the pelvis:

```
pelvis (base_link)
├── left_hip_yaw_joint  → left leg (6 DOF)
├── right_hip_yaw_joint → right leg (6 DOF)
└── waist_yaw_joint     → torso → shoulder → elbow → wrist (7 DOF per arm)
```

**Marker placement:** Attach a 4-marker asymmetric cluster on a rigid plate secured to the pelvis shell. Set the rigid body pivot point in Motive to the pelvis origin (center of the hip plate). If markers are on the outer shell surface, calibrate a static TF offset of a few centimeters in Z.

### 4.3  Agibot Expedition A3

The Expedition A3 is an athletic humanoid by Agibot (Zhiyuan Robotics).

| Property | Value |
|----------|-------|
| `base_link` location | **To be confirmed** — likely pelvis (Pattern A), but the flexible waist may warrant Pattern C |
| Standing height | Full-size (~1.75 m, estimated from video) |
| Weight | Not publicly disclosed |
| Total DOF | Not publicly disclosed; described as "highly anthropomorphic full-body degrees of freedom" |
| Arm DOF | Not publicly disclosed (7 DOF per arm expected, based on Agibot platform lineage) |
| Waist DOF | **Multi-DOF flexible waist** — a key distinguishing feature engineered to mirror the human range of motion, enabling rotation and swaying for complex whole-body movements |
| URDF source | Not publicly available as of March 2026 |
| Middleware | **AimRT** (Agibot's native C++20 runtime); supports ROS 2 protocol bridging |

**Key considerations:**

1. **Flexible waist implications.** The A3's multi-DOF flexible waist is specifically engineered for the kind of torso rotation and weight transfer that table tennis demands. However, if the waist has 2–3 DOF (pitch, roll, yaw), the choice of where `base_link` sits relative to the waist joints significantly affects the FK chain length. For ping-pong, the waist DOFs contribute directly to racket positioning (waist rotation extends the arm's effective reach and angle), so `base_link` should ideally be **below** the waist (Pattern A) to include waist DOFs in the paddle FK chain.

2. **Vendor coordination.** Teams planning to use the A3 for HOPE should coordinate directly with Agibot to obtain the URDF and confirm the `base_link` convention, `base_link` height, and the complete joint chain from `base_link` to the paddle-holding hand. The open-source Agibot X1 training repository (`github.com/AgibotTech/agibot_x1_train`) contains URDF files under `resources/robots/` and may serve as a reference for Agibot's kinematic tree conventions.

3. **Middleware bridging.** The A3 runs on AimRT natively, not ROS 2. AimRT supports ROS 2 as one of several communication protocols (alongside HTTP, gRPC, MQTT, and Zenoh). For the HOPE architecture, two integration approaches are available:
   - **Approach 1 (recommended):** Run the HOPE planner as a ROS 2 node; bridge the `RacketCommand` topic into AimRT where the A3's native WBC consumes it. The `base_link` pose from motion capture still flows through ROS 2 → AimRT.
   - **Approach 2:** Run the planner within AimRT directly, subscribing to the motion capture data via AimRT's ROS 2 protocol support.

### 4.4  Competition Registration Requirements

Each team must declare the following during HOPE competition registration. This information is needed to verify that the motion capture system, planner, and WBC are correctly integrated for their specific humanoid platform.

| Item | Description | Example (Unitree G1) |
|------|-------------|---------------------|
| **Robot model** | Manufacturer and model designation | Unitree G1 EDU |
| **`base_link` URDF link name** | The exact link name in the URDF that corresponds to `base_link` | `pelvis` |
| **`base_link` physical location** | Description of where the link origin sits on the physical robot | Center of hip plate, at intersection of hip yaw axes |
| **`base_link` pattern** | Which convention (A/B/C from Section 4.1) | Pattern A (pelvis root) |
| **Standing `base_link` height** | Height of `base_link` origin above the floor when standing in nominal pose | 0.78 m (z ≈ +0.02 m in HOPE frame) |
| **Mocap-to-URDF static offset** | Translation [dx, dy, dz] from the mocap marker cluster centroid to the URDF `base_link` origin | [0.0, 0.0, −0.03] m (markers on outer shell, 3 cm above pelvis origin) |
| **Arm DOF count** | Number of actuated joints from `base_link` to paddle grip, including waist | 1 waist + 7 arm = 8 DOF |
| **Middleware** | ROS 2 native, AimRT with ROS 2 bridge, or other | ROS 2 native |
| **URDF availability** | Public URL or "provided to organizers under NDA" | `github.com/unitreerobotics/unitree_ros` |

The static offset (mocap-to-URDF) is published as a `static_transform_publisher` in the team's launch file:

```python
Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=[
        '--x', '0.0', '--y', '0.0', '--z', '-0.03',
        '--roll', '0', '--pitch', '0', '--yaw', '0',
        '--frame-id', 'P1_mocap',
        '--child-frame-id', 'P1_base_link'
    ],
)
```

### 4.5  What the Robot Knows vs. What Motion Capture Provides

| Information | Source | Used by |
|-------------|--------|---------|
| Ball 6-DOF pose: position `[x, y, z]` + quaternion `[qx, qy, qz, qw]` at the capture rate | Motion capture → ROS 2 topic | Planner uses position (Stages 1–3); orientation is preserved for validation and future spin-aware estimation |
| Humanoid `base_link` 6-DOF pose | Motion capture → ROS 2 topic | WBC (Stage 4) for base position commands |
| `Table` rigid-body pose | Setup/calibration sessions and training-data recordings only — **no competition stream** | Arena calibration (world-origin verification) |
| Paddle 6-DOF pose | **Forward kinematics** from joint encoders + `base_link` | WBC internal state; **not** from motion capture |
| Paddle desired state | Planner output (Stage 3) | WBC (Stage 4) as tracking target |

---

## 5  Ball Rigid-Body Tracking Configuration

Both OptiTrack Motive and Chingmu CMTracker now solve the ping-pong ball as a named **rigid-body asset**. The measured state is a full pose: translation `(x, y, z)` and orientation. The asset pivot must coincide with the ball's geometric center; if that is not possible in the vendor tool, record and apply a fixed asset-to-center transform.

### 5.1  Ball Preparation and Asset Definition

- Use the vendor-qualified ball preparation and marker pattern/constellation for rigid-body tracking. Do not infer a working marker count or layout from the retired single-point setup.
- Minimize changes to the ball's mass, center of mass, diameter, surface friction, and aerodynamics, and validate the prepared ball against competition rules.
- Make the pattern asymmetric and distinguishable from `Table` and robot patterns throughout the camera volume.
- Define a stable rigid-body asset name and ID (recommended logical name: `Ball`) in Motive or CMTracker. Topic and sender names are case-sensitive.
- Set the rigid-body pivot to the geometric center and document its local axes. Validate high-speed tracking, occlusion recovery, and ID stability before recording data.

> **Legacy ball preparations are incompatible.** Earlier revisions of this document (≤ v0.4,
> single-marker tracking) recommended a *fully coated* retroreflective ball. That
> recommendation is now **inverted**: a uniformly coated sphere presents no distinguishable
> marker constellation and cannot be identified as a rigid body. Rigid-body tracking requires
> a patterned preparation. Do not reuse balls prepared for the retired single-point setup.

### 5.2  Pose Representation

Operators may inspect the state as `(x, y, z, pitch, yaw, roll)`, but ROS 2 should never carry Euler angles as the canonical orientation representation. VRPN tracker reports carry a quaternion; ROS 2 stores it in `geometry_msgs/Pose.orientation` as `(x, y, z, w)`:

```text
geometry_msgs/Pose
  position:    x, y, z
  orientation: x=qx, y=qy, z=qz, w=qw
```

Normalize each quaternion and reject NaN, zero-norm, or stale poses. If Euler angles are needed for display or analysis, state the frame, handedness, and rotation order. Motive displays X as Pitch, Y as Yaw, and Z as Roll in its documented right-handed local-axis convention; do not assume that another vendor's Euler display uses the same convention.

### 5.3  Orientation, Angular Velocity, and Spin

The current HOPE planner remains a **no-spin** planner: it consumes `(x, y, z)` and models translational drag, but it does not use ball orientation or Magnus force. The ROS 2 bridges nevertheless preserve the measured quaternion so recordings remain 6-DOF and future estimators can use it.

A rigid-body quaternion is not itself angular velocity. A spin-aware extension must unwrap quaternion sign, differentiate rotations using source timestamps, filter noise, and verify that the tracked rigid-body pattern is mechanically locked to the ball. If the marker carrier slips relative to the shell, the reported attitude is not physical ball spin.

### 5.4  Acceptance Checks

Before an arena session:

1. Place the Ball asset at surveyed table landmarks and confirm the reported pivot is the ball center.
2. Rotate the prepared ball through known attitudes and confirm the quaternion is normalized and the displayed pitch/yaw/roll change on the intended axes.
3. Launch, bounce, and strike the ball across the full volume; measure dropouts, latency, reacquisition behavior, and any asset-ID swaps.
4. Confirm the ROS 2 `frame_id`, source timestamps, units, and complete position-plus-orientation fields with `ros2 topic echo`.
5. Record the vendor asset name/ID, marker preparation, pivot transform, local-axis definition, and software versions with the session metadata.

---

## 6  Streaming Rigid-Body Poses to ROS 2

The vendor application performs camera reconstruction and rigid-body solving. A ROS 2 bridge receives the vendor's native stream and maps every solved pose to standard ROS messages. Position is in metres; orientation remains a quaternion from the vendor stream through ROS 2.

### 6.1  Confirmed Transport Path (VRPN, both vendors)

Both vendors stream over **VRPN**, into the same vendored ROS 2 client, through the same
adapter, onto the same planner topic — one uniform path:

```text
OptiTrack cameras → Motive (VRPN Streaming Engine): Ball/P1/P2 rigid bodies ───────┐
                                                                                   │ VRPN tracker
Chingmu cameras → CMTracker/MCServer: Ball/P1/P2 rigid bodies ─────────────────────┤ reports
                                                                                   ▼
  vendored vrpn_mocap → /vrpn_mocap/<sender>/pose_id_<N> (geometry_msgs/PoseStamped)
    → hope_bringup/pose_to_posearray → /poses (geometry_msgs/PoseArray)
```

| System | Vendor payload | ROS 2 bridge | ROS 2 result |
|--------|----------------|--------------|--------------|
| **OptiTrack** | VRPN tracker report per Motive rigid-body asset (VRPN Streaming Engine, rigid bodies **only**): sender name, sensor index, position vector, and quaternion | vendored `vrpn_mocap` | `/vrpn_mocap/<sender>/pose_id_<sensor_id>` as `geometry_msgs/PoseStamped` with `multi_sensor: true`; shipped adapter copies the complete pose to `/poses` |
| **Chingmu** | VRPN tracker report for a named CMTracker rigid body: sender name, sensor index, position vector, and quaternion | vendored `vrpn_mocap` | identical to the OptiTrack row — same client, same topics, same adapter |

This is the important wire-level fact: `(pitch, yaw, roll)` is an operator-facing representation, while a VRPN tracker report delivers the orientation as a quaternion, which the ROS 2 messages preserve end to end. The shipped `pose_to_posearray` adapter assigns `out.poses[i] = input.pose`, so it does **not** discard ball orientation. `PoseArray` has no per-pose name field; keep the configured input order stable (Ball is normally `ball_pose_index: 0`) and use recorded asset metadata (or an optional `tf2` broadcaster) when names are required.

> **Why not NatNet?** Motive's NatNet stream remains available for diagnostics or other
> consumers, but the HOPE reference path does not use it: the NatNet-based ROS 2 bridges
> publish their own message types (e.g. `motion_capture_tracking`'s `/poses` is a custom
> `NamedPoseArray`, not the `geometry_msgs/PoseArray` the shipped planner subscribes to),
> so they do not connect to the shipped stack without an extra conversion node. The VRPN
> path needs no such conversion on either vendor.

### 6.2  OptiTrack / VRPN Path

In Motive, define the competition assets `Ball`, `P1`, and `P2` as rigid bodies, set each pivot (`Ball` at the ball center, `P1`/`P2` at each robot's declared `base_link`), and enable the **VRPN Streaming Engine** in the Data Streaming settings. Each rigid body is served as a VRPN tracker under its asset name. The `Table` asset is used during calibration sessions only (Section 2.3) — disable or delete it before competition streaming so no `Table` tracker is served.

The expected Motive settings are:

| Setting | Required value | Notes |
|---------|----------------|-------|
| VRPN Streaming Engine | ✅ Enabled | Serves each rigid body as a VRPN tracker named by asset |
| VRPN Broadcast Port | 3883 (default) | Must match the `port` parameter of the ROS 2 client |
| Rigid Bodies | Defined for `Ball`, `P1`, `P2` (competition); `Table` for calibration sessions only | VRPN streams **rigid bodies only** — markers and skeletons are not carried over VRPN |
| Zero When Untracked | **Disabled** (recommended) | If enabled, an occluded asset streams an all-zero pose and the downstream consumer sees the ball teleport to the origin. Prefer dropout plus rejection of stale/identity poses (Section 6.5). |

Two OptiTrack-specific caveats:

1. **Frame conversion is configuration-specific — verify before enabling.** OptiTrack documents *Up Axis* as selecting the up axis of streamed data, but does not clearly specify whether it applies to the VRPN Streaming Engine; deployments commonly observe Y-up VRPN output regardless of the setting, and behavior can differ by Motive version. Do **not** assume either frame: place the `Ball` asset at surveyed table landmarks (Section 6.5) and read the streamed coordinates. Enable the conversion below **only if** the stream is verified Y-up — applying Rx(+90°) to a stream that is already Z-up double-rotates every pose and silently corrupts the world. **Required engineering (not included in this repository):** a frame-conversion stage on the ROS 2 side, between the VRPN client and `/poses` — either an option added to the `pose_to_posearray`-style aggregation step, or a small standalone relay node subscribing to each `PoseStamped` topic and republishing it converted. It must apply the fixed Y-up → Z-up **full-pose** transform of Section 6.4 exactly once per sample: rotate the position by Rx(+90°) — `(x, y, z) → (x, −z, y)` — and left-multiply the orientation quaternion by the same fixed rotation (`q_R = Rx(+90°)`, i.e. `(x=√½, y=0, z=0, w=√½)` in `(x, y, z, w)` order), preserving the header stamp and `frame_id`. It must be switchable per deployment (a boolean parameter, **off by default**), because Z-up sources — Chingmu configured per Section 2.2, and any Motive installation whose VRPN stream is verified Z-up — must pass through unconverted.
2. VRPN carries no marker data, so any marker-level diagnostics must use NatNet side by side; this does not affect the reference path.

Then run the same vendored client as for Chingmu, pointed at the Motive host:

```bash
ros2 launch vrpn_mocap client.launch.yaml server:=MOTIVE_PC_IP port:=3883
```

Verify that all four assets appear as `/vrpn_mocap/<asset>/pose_id_0` and that the full `Ball` pose appears at the configured `/poses` index.

### 6.3  Chingmu / VRPN Path

In CMTracker/MCServer, define the ball as a rigid body and assign a stable VRPN sender name such as `Ball`. The Ball is no longer handled as an unlabeled marker under a shared sender. Set the streaming up axis to **Z** (Section 2.2) so no software frame conversion is needed. Run the vendored native ROS 2 VRPN client against the Chingmu server:

```bash
ros2 launch vrpn_mocap client.launch.yaml server:=CHINGMU_SERVER_IP port:=3883
```

```yaml
/vrpn_mocap_client:
  ros__parameters:
    server: "CHINGMU_SERVER_IP"
    port: 3883
    frame_id: "world"
    multi_sensor: true
    use_vrpn_timestamps: false  # set true only when server and ROS clocks are synchronized
    update_freq: 100.0
    refresh_freq: 1.0
```

The client auto-discovers VRPN tracker senders. Its pose callback maps `vrpn_TRACKERCB.pos[0:3]` directly to `PoseStamped.pose.position` and `quat[0:4]` directly to `PoseStamped.pose.orientation.{x,y,z,w}`. With `multi_sensor: true`, typical single-sensor rigid bodies appear as:

```text
/vrpn_mocap/P1/pose_id_0     geometry_msgs/PoseStamped
/vrpn_mocap/P2/pose_id_0     geometry_msgs/PoseStamped
/vrpn_mocap/Ball/pose_id_0   geometry_msgs/PoseStamped
```

Actual names and capitalization come from CMTracker and are case-sensitive. If CMTracker assigns a sensor index other than zero, use that published index rather than rewriting it. `multi_sensor: true` is a safe default and prevents collisions if a sender exposes more than one sensor.

Configure `hope_bringup/pose_to_posearray` with the Ball topic first to preserve the planner's default `ball_pose_index: 0`. The adapter publishes `/poses` but does not create `/tf`; add a `tf2_ros` broadcaster if the deployment also requires named transforms.

### 6.4  Coordinate and Orientation Conversion

Both vendor outputs must arrive in the canonical REP 103 Z-up frame of Section 2.1. CMTracker can stream Z-up natively (configure it, Section 2.2); the frame of Motive's VRPN stream is installation-dependent and must be verified at landmarks (Sections 6.2 and 6.5) — the conversion is applied only when the stream is verified Y-up, per the engineering requirement described in Section 6.2. Whenever a conversion is applied, transform the **entire pose**, not just its three position values:

```text
p_HOPE = R_HOPE_FROM_MOCAP · p_mocap + t_HOPE_FROM_MOCAP
R_HOPE_BODY = R_HOPE_FROM_MOCAP · R_mocap_body
```

For the commonly encountered right-handed Y-up to HOPE Z-up rotation, translation alone maps as `x_HOPE=x_mocap`, `y_HOPE=-z_mocap`, `z_HOPE=y_mocap`. Apply the same fixed rotation to the orientation using `tf2` or quaternion/matrix composition. Component-wise edits to pitch/yaw/roll are not a valid general pose transform.

Verify the source handedness and axis directions at surveyed table landmarks before trusting any conversion. A mirrored source frame needs an installation-specific correction; do not guess it from the vendor name.

### 6.5  ROS 2 Validation Checklist

```bash
ros2 topic list | grep -E 'poses|vrpn_mocap'
ros2 topic echo /vrpn_mocap/Ball/pose_id_0 --once   # either vendor
ros2 topic echo /poses --once                       # the planner input
```

Confirm all of the following:

- `Ball` (and `P1`/`P2` where streamed) are distinct, stable rigid bodies; no asset ID swaps occur after occlusion. Confirm **no `Table` topic is being streamed** during competition.
- `position` is in metres, `orientation` is finite and unit length, and the Ball pivot is at its geometric center.
- The message `frame_id` and axes match the HOPE world frame. **Landmark validation is mandatory before play** for every vendor and installation: place the `Ball` asset at surveyed landmarks (e.g. the net-center line `x = 1.37, y = −0.7625, z = 0.02`) and confirm the streamed coordinates match Section 2.1 — this determines whether the Y-up → Z-up conversion (Sections 6.2/6.4) must be ON or OFF, and catches both a missing conversion and a double rotation.
- Occlusion produces a **dropout**, not a frozen or all-zero pose: disable Motive's *Zero When Untracked*, and reject stale, identity, or zero-norm poses in the consumer.
- Capture timestamps are preserved where supported. If `use_vrpn_timestamps: true`, synchronize the mocap server and ROS host with NTP/PTP; otherwise use receipt time and characterize its jitter.
- `/poses` index order matches the planner configuration. The current no-spin planner reads the Ball position while the full quaternion remains in the message and bag recording.

---

## 7  Integration with the HOPE Planner

The companion planner document (*HOPE 7DOF Racket Model-based Planner Reference Setup*) consumes ball position data from the `/poses` stream described in Section 6 and produces racket target commands. The data flow through the complete system is:

```
Motion Capture System (360 Hz)                         Humanoid (proprioceptive)
  │                                                      │
  ├── Ball 6-DOF rigid-body pose ──▶ HOPE Planner      │
  │      (planner currently uses xyz)  Stages 1–3       │
  │                                        ▼              │
  └── P1 base_link 6-DOF ──────────▶ WBC (Stage 4) ◀── RacketCommand
                                           │              (p_intercept,
                                           │               v_racket,
                                           ▼               n_racket,
                                     Joint commands        t_strike)
                                     (varies by platform)
                                           │
                                           ▼
                                     Paddle pose
                                     (inferred via FK from
                                      base_link + joint encoders,
                                      NOT measured by mocap)
```

The planner operates entirely in the HOPE canonical world frame defined in Section 2.1. The OptiTrack and Chingmu bridges deliver complete rigid-body poses in this frame after the configuration or transform in Section 6.4. The current planner uses the Ball translation; the orientation quaternion remains available to other consumers and recordings.

---

## 8  Summary

The HOPE motion capture reference system publishes exactly four named rigid bodies:

1. **`Ball`** — the ping-pong ball as a vendor-tracked 6-DOF rigid body. ROS 2 receives position plus quaternion orientation; the current no-spin planner uses position only.
2. **`Table`** — a setup/calibration-only asset anchoring the world frame origin (legacy arena notes may call this `PPT`); it appears in training-data recordings but is **not** streamed during competition.
3. **`P1`** — the Player 1 humanoid `base_link` rigid body.
4. **`P2`** — the Player 2 humanoid `base_link` rigid body.

The `base_link` definition varies by manufacturer (Section 4); each team declares its robot-specific mapping at registration.

**The paddle/racket is never tracked by the motion capture system.** Each humanoid must infer its own paddle pose through forward kinematics from joint encoders and the tracked `base_link`. This is the fundamental sensing architecture: external perception (ball trajectory) feeds the model-based planner, while internal proprioception (joint states + `base_link`) drives the whole-body controller that positions the paddle. See the companion *HOPE WBC Simulation Training Reference Setup* (Section 2.8) for the complete forward kinematics chain from `base_link` through the 7-DOF arm to the 3D-printed racket mount.

---

## References

- Su, Z., Zhang, B., Rahmanian, N., Gao, Y., Liao, Q., Regan, C., Sreenath, K., & Sastry, S. S. (2025). HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning. *arXiv:2508.21043v2*.
- HITTER project page: https://humanoid-table-tennis.github.io/
- motion_capture_tracking (NatNet-based alternative; publishes a custom `NamedPoseArray`, not used by the HOPE reference path): https://github.com/IMRCLab/motion_capture_tracking
- OptiTrack Motive VRPN Streaming Engine (rigid bodies only; default port 3883): https://docs.optitrack.com/motive-ui-panes/settings/settings-streaming
- VRPN protocol: https://github.com/vrpn/vrpn
- 青瞳视觉 (CHINGMU) motion capture: https://www.chingmu.com/ (EN: https://en.chingmu.com/) — VRPN/LiveStream streaming, C/C++/C#/Python/ROS SDKs
- ChingMuVrpnRos (official Chingmu ROS/VRPN reference): https://github.com/ChingMuVisionTech/ChingMuVrpnRos
- vrpn_mocap (ROS 2 VRPN client): https://index.ros.org/p/vrpn_mocap/
- Agibot X1 training code (reference for Agibot kinematic conventions): https://github.com/AgibotTech/agibot_x1_train
- Companion document: *HOPE 7DOF Racket Model-based Planner Reference Setup, v0.1*
- Companion document: *HOPE WBC Simulation Training Reference Setup, v0.5*
- Companion document: *HOPE Hardware Deployment Reference Setup, v0.1*
