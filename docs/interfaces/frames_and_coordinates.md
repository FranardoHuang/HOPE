# Frames And Coordinates

Status: Draft

## Canonical World Frame

Use the HOPE table `world` frame unless a gate explicitly names another frame. A frame with the same
axis directions but a different origin is not `world`.

- Origin: near-side left corner of the table surface.
- X: toward opponent along table length.
- Y: left along table width. The table extends along `-Y` (its center is at `y = -0.7625`).
- Z: up. **World `z = 0` IS the table surface.** The floor is at `z = -0.76` (`floor_origin`). The `0.76 m` value is the surface-to-floor height, NOT the surface world Z.
- Table length: `2.74 m`.
- Table width: `1.525 m`.
- Table surface-to-floor height: `0.76 m` (the surface itself is at world `z = 0`).
- Net height: `0.1525 m`.

Landmark coordinates (from `hope_world_frame.yaml`):

- `table_center`: `[1.37, -0.7625, 0.0]`
- `net_center`: `[1.37, -0.7625, 0.0]`
- `p1_half_center`: `[0.685, -0.7625, 0.0]`
- `p2_half_center`: `[2.055, -0.7625, 0.0]`
- `floor_origin`: `[0, 0, -0.76]`

Current config source:

- [`hope_ws/src/hope_bringup/config/hope_world_frame.yaml`](../../hope_ws/src/hope_bringup/config/hope_world_frame.yaml)
- [`hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py) for the Isaac table-tennis scene.

For `HOPE-TableTennis-AgibotA3-v0`, the Isaac environment-local world frame is intentionally the HOPE
frame: the table surface is `z = 0`, the floor is `z = -0.76`, and table/net landmarks match the values
above. Do not introduce a table-center origin in that task without updating this file and G04.

<a id="canonical-motion-hope-world-bridge"></a>

## Canonical Motion 与 HOPE `world` 的桥

Canonical 五动作的 recipe 使用另一个明确 frame ID：

| frame ID | 原点与用途 |
| --- | --- |
| `world` | HOPE 球台/ROS frame；P1 近端左桌面角为原点，桌面 `z=0` |
| `a3_robot_origin_ground_z0` | schema-2/vendor-MJCF 动作 frame；机器人局部地面原点，地面 `z=0` |

两者都是 `+X` 朝对手、`+Y` 朝机器人左侧、`+Z` 向上，但绝不是同一 frame。现行 canonical
counterfactual station（机器人原点到近台边 `0.5 m`、对准桌宽中心）的唯一桥是无旋转纯平移：

```text
p_a3_robot_origin_ground_z0 = p_world + [0.5, 0.7625, 0.76]
p_world = p_a3_robot_origin_ground_z0 + [-0.5, -0.7625, -0.76]
R_a3_from_world = I
```

所以 `world` 的桌中心 `[1.37,-0.7625,0]` 映到 motion/MJCF 的
`[1.87,0,0.76]`。位置使用上式；速度、角速度、法向和姿态只应用旋转（这里为恒等），不加 translation。
compiler output 保持 `a3_robot_origin_ground_z0`，不能仅因轴相同就标成 `world`。桌网/planner verifier
必须在 manifest 中写明 source/target frame ID、变换方向和 transform SHA，并从 exact bytes 复核。

这座桥只定义固定 canonical counterfactual station，不是录制现场外参、mocap
`world -> base_link` 标定或 locomotion 后的动态站位。locomotion 改变机器人站位后必须使用其内容绑定
的 root/station transform，并重跑桌网、接触、动力学和行为门；不得继承上述固定平移的证书。vendor
sim 的 `odom` 也不能凭名字视为这两个 frame 之一；其当前受限合同见下方
[Vendor MuJoCo `SimReset` Base Twist](#vendor-mujoco-simreset-base-twist)。

<a id="task-first-station-center"></a>

## Task-first station center

[`station_center_shift_xy_m`](../DEFINITIONS.md#station-center-shift) 位于
`a3_robot_origin_ground_z0` 的 XY 平面，并平移**整套动作/任务中心**，不是只改 base 的 offset。
设动作 reference 拍点为 `p_ref`、base 中心为 `b_ref`、位置扰动为 `delta_p`、最后一轴的 base
residual 为 `delta_b`：

```text
racket_target = p_ref + [station_x, station_y, 0] + delta_p
base_target_xy = b_ref + [station_x, station_y] + delta_p_xy + delta_b
```

motion 和 HOPE table 两套 frame 都以 `+X` 朝对手/球台，所以 station X 为负表示机器人、动作和
要求拍点一起远离近台边。新正手比较 `[0,0]`、`[-0.05,0]`、`[-0.10,0] m`，它们是三个
候选中心，不是课程中同时采样的三个值；只有 upper/full 都通过桌网、身体、地面、动力学和行为门
后，才取离原站位最近的一档。

station center 属于 level-0 task identity。level 0 只做中心 warm-up，position perturbation 过门后
才开始；base residual 是最后一轴。改变 station center 会改变 manifest/task identity，并使旧碰撞、
能力与 selector 证据失效。

## Historical ChingMu Runtime Contract (superseded for FullMDP semantic V2/V3)

The following 300-Hz ChingMu/VRPN contract documents the old P1/P2 consumer and remains relevant
only when reproducing that lineage. The venue has since switched to 360-Hz OptiTrack; FullMDP
semantic V2/V3 use the OptiTrack+IMU split stated in
[FullMDP semantic V2/V3 shared table and heading frame](#action-ball-table-pose-frame).
Isaac and MuJoCo implement the simulator row, but the real C++ producer does not; this sentence does
not authorize deployment.
Do not copy the old object names, rate, sensor authority or noise profile into the fresh contract.

The historical rig was ChingMu streaming over VRPN. The tracked-object set differed by phase:

- **PLAY (live table tennis), 300 Hz**: robot base (pelvis) pose and ball position. Ball
  rotation/spin is planned for the physics-modeling phase — the ball is currently tracked as a
  point and the relay publishes a position-only `PointStamped` (`/ball/point`), so spin work will
  also require a patterned/rigid-body ball and a relay change to forward orientation.
- **DATA-COLLECTION / physics-calibration only**: additionally racket pose, the table's 4 corners,
  and the net's 2 corners. The racket-tracking prohibition in the HOPE challenge materials applies
  to competition play, not to this internal calibration phase.

Current expected mocap object names:

- Table rigid body: `PPT` (setup/calibration anchoring; optional during play)
- Player 1 robot rigid body: current config default `ppp2` (publishes `/P1/pose`)
- Player 2 robot rigid body: current config default `ppp3` (publishes `/P2/pose`)
- Ball: named rigid body (preferred) or auto-detected moving marker

Current relay config:

- `hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml`
- VRPN bridge launch default `update_freq` is aligned to the rig's 300 Hz
  (`hope_ws/src/hope_bringup/launch/avatar_pro_hope_bridge.launch.py`).

Consumption status (updated 2026-07-13): the ROS chain (relay → `hope_planner`) consumes mocap.
The C++ deploy runner does not subscribe to raw mocap, but its `--planner` source now subscribes to
the planner's `/racket/command_flat` and `/a3/base_pose_flat` through AimRT ROS 2. This is source
wiring only; the current exact-179 ROS/AimRT first tick and vendor behavior are unrun. The deployed
175-D actor observation deliberately consumes no mocap term (see
[policy_observation_action.md](policy_observation_action.md)); when the mocap→planner bridge
is used, the HOPE-world → robot-frame target transform still depends on the corrected base pose at
the interface boundary.

The historical base flat wire already carried position plus a normalized quaternion, but the old
`LocMode::kExternalBase` C++ consumer deliberately uses only mocap position and retains the
yaw-aligned pelvis-IMU quaternion. That mixed historical producer is not semantic V2/V3. Their shared prefix requires
one time-consistent packet: table-relative root position and unit heading from calibrated
OptiTrack/localization, causal root inertial-COM linear velocity from the same pose lineage, and
projected gravity plus body-frame angular velocity from the pelvis IMU. Explicit latency,
stale/dropout and marker-extrinsic handling remain mandatory. The current complete ordered V3 215/231 layout is
defined only in [policy_observation_action.md](policy_observation_action.md#current-portable-fullmdp-semantic-observation-v3-actor-215--critic-231);
V2 203/219 is historical/control.

<a id="action-ball-table-pose-frame"></a>

## FullMDP semantic V2/V3 shared table and heading frame

Semantic V2 and V3 intentionally share a frame contract narrower than a general movable-table interface. The current Isaac and
MuJoCo ActionBall scenes fix every table to the environment-local world axes; environment origins
translate rows but never rotate them. In that fixed frame:

```text
p_base_table = p_root_world - env_origin - p_table_surface_center_scene
heading_xy = normalize(project_xy(R_world_root * [1, 0, 0]))
```

The table-surface centre is `(near_x + TABLE_LENGTH/2, 0, surface_z)` in scene coordinates. A future
rotatable table needs a measured table pose and a new versioned contract; it must not reinterpret
these columns in place. A root pose whose forward X axis is nearly vertical has no stable yaw heading
and fails explicitly instead of publishing a shrinking or fabricated vector.

For `heading_xy=[c,s]`, every world-axis vector uses one inverse-yaw transform while preserving
physical vertical Z:

```text
heading([x,y,z]) = [c*x + s*y, -s*x + c*y, z]
position_error_h = heading(target_position_world - achieved_position_world)
vector_error_h   = heading(target_vector_world - achieved_vector_world)
```

Therefore motion-paddle residuals, task position/velocity/raw-A-normal residuals and root COM velocity share the table yaw
convention without rotating through transient root roll/pitch. `base_position_table` is absolute
table context, `base_goal_error_heading_xy` is a task residual, and `motion_anchor_pos_b` is a
teacher alignment residual; equal dimensions do not make them interchangeable. Their exact network
order belongs only to
[policy_observation_action.md](policy_observation_action.md#current-portable-fullmdp-semantic-observation-v3-actor-215--critic-231).

Isaac's `root_lin_vel_w` is the linear velocity of the root body's inertial center of mass. MuJoCo's
`cvel` linear component is instead expressed at the kinematic root subtree COM, so the portable
MuJoCo producer converts it to the pelvis body's inertial COM before heading rotation:

```text
v_body_com_world = cvel_linear_world
                   - (xipos_body - subtree_com_root) cross omega_world
```

Using raw freejoint `qvel`, link-origin velocity, or unconverted `cvel` would change the observation
point and violate cross-backend semantics.

The real producer must derive table/root position and heading from calibrated OptiTrack table and
pelvis marker bodies plus a measured marker-cluster→root-inertial-COM transform. Root-COM velocity
must be causal: update only on a genuinely new capture timestamp, transform the marker sample to the
COM point, estimate from present/past samples, and surface age/dropout instead of turning held poses
into zero velocity. Pelvis IMU supplies projected gravity and calibrated body-frame gyro directly;
joint encoders and FK supply achieved robot/racket state. Capture-time synchronization,
OptiTrack↔IMU offset, marker/COM extrinsics, estimator error under dropout and the exact real 215-D
builder have not been accepted. V3's additional measured-teacher/achieved paddle values use sources
already required here, but source availability does not prove synchronization or builder parity.
Semantic V3 is therefore simulator-trainable in design but **not deploy-ready**.

## Base Link

`base_link` must come from the robot model and SDK. Do not assume the mocap marker cluster equals `base_link` unless the transform has been measured.

Historical ChingMu measurements (do not satisfy the OptiTrack contract):

- venue/table frame -> ChingMu mocap world
- `P1 mocap frame -> P1_base_link`
- `P2 mocap frame -> P2_base_link`
- A3 standing `base_link` height

Fresh OptiTrack measurements required before deployment-intent retraining:

- venue/table frame -> OptiTrack world;
- OptiTrack rigid marker cluster -> A3 `base_link`/root COM;
- OptiTrack v2 camera mid-exposure/Motive/source/receive/consume timestamps and Motive smoothing setting;
- time offset and rotational extrinsic between OptiTrack and pelvis IMU gyro;
- causal root-COM velocity error, dropout and hold-last distributions.

The live 360-Hz P1 rigid body already supplies metric position and full orientation, but the
currently consumed v1 message timestamps ROS arrival, not capture. Noise or delay values from the
old ChingMu ball pipeline therefore cannot close this list. The causal velocity producer updates
only on a genuinely new pose using its real `delta_t`; a repeated hold-last pose preserves the
previous estimate and increases localization age instead of fabricating zero velocity.

## Racket Frame

The racket is not tracked by motion capture **during play** (competition rule); during the
data-collection/physics-calibration phase the racket pose IS captured for model fitting. At play
time it must be inferred from:

1. `world -> base_link`
2. robot joint state
3. robot model FK
4. fixed racket mount transform

The canonical control point is the origin of MuJoCo site `right_racket` / URDF link
`pingpang_red_Link`. Its fixed offset in `right_wrist_yaw_Link` is:

- `RACKET_SITE_OFFSET_WRIST_M = (0.210210, 0.032078, 0.032036) m`.
- Raw mount normal A = racket-local `+Y` (`mount_normal_axis = 1`, `mount_normal_sign = +1`). The
  train bank and 179-D actor face tail remain in this A convention.
- Flat-wire schema 2 carries the physical opponent-facing striking face B in world/table frame.
  After clip selection, `n_A = s[clip] * n_B` with frozen `s=[+1,-1]` for
  forehand/backhand; the inverse is identical. This conversion applies only to the normal, never
  racket position or velocity. Consequently a valid backhand raw-A normal can have negative world
  x while its physical-B wire normal must have positive world x.

This site offset is the canonical engineering control point, not a non-final approximation. Exact
ball-centre contact, red/black face-centre offsets and the older `1.49 um`-different Python point are
separate concerns whose single truth is
[Racket Control Point And Contact Geometry](racket_contact_geometry.md); do not copy either older
offset back into this page.

For motion-reference construction, “measured racket” and “FK racket” are distinct sources in this
same frame. The measured marker rigid body becomes the teacher only after one frozen
marker-to-`official_racket_site` transform, including signed face; retargeted A3 FK is the prediction
to be fitted against it. That measured site constraint must participate in retargeting, and the
position/orientation/point-velocity residual must be reported per action. During live play the
achieved state still follows the four-step FK chain above because competition play has no racket
mocap. See the measured-racket alignment contract in
[Racket Control Point And Contact Geometry](racket_contact_geometry.md).

## Vendor MuJoCo `SimReset` Base Twist

The vendor simulator publishes `/sim/a3/pelvis_twist` from a zero-offset `pelvis_site` with
`frame_id=odom`. Its current numeric meaning is the pelvis **link-origin** linear velocity and
angular velocity, with both vectors expressed in odom/world axes. This is not the motion NPZ's
pelvis-COM linear-velocity convention.

Nonzero `MODE_ABSOLUTE` twist reset is currently unsafe as a frame-exact interface:
`SimResetRos2Subscriber::ApplyBaseTwist` copies the world angular vector directly into MuJoCo's
body-local freejoint angular qvel and does not validate `header.frame_id` or state the linear-velocity
point. The tracked keyframe reset path is unaffected because it writes an all-zero twist. Until a
separate G04/G07 ticket freezes and tests the ROS contract, formal flows must use the named
keyframe/zero-velocity reset and must not replay a nonzero `/sim/a3/pelvis_twist` row as raw qvel.

## Update Rule

Any frame, axis, origin, static transform, or measured calibration change must update this file and the relevant gate doc.
