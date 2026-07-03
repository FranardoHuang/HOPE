# Sim2Real Bridge: mocap → planner → WBC runner → robot

**Status: implemented + dry-run-verified 2026-07-03.** This is the ROS 2 chain that
connects the AvatarPro/CMTracker mocap through the real `hope_planner` to the
`hope_wbc_runner` (and, in hardware mode, through `agibot_hardware_bridge` to the
AGI backend). It replaces the earlier `planner_imitate`-only bring-up.

```
AvatarPro / CMTracker (MCServer, VRPN, 300 Hz)
  └─ vrpn_mocap client_node        /vrpn_mocap/<obj>/pose_id_<n>      [mocap frame == HOPE table frame]
     └─ avatar_pro_vrpn_relay      /poses (ball@idx0, ball-synced), /P1/pose, /ball/point
        └─ hope_planner_node       /racket/command  frame_id="world"  [TABLE frame]
           └─ wbc_runner           table→policy transform, swing side, gate, clock, ONNX
              └─ /a3/joint_command (ONLY mode=hardware + enable + !estop)
                 └─ agibot_hardware_bridge   /body_drive/*_joint_command → AGI backend
```

One-line launch (dry-run, safe — never publishes joint commands):

```bash
distrobox enter hope -- bash -lc '
source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 launch hope_bringup hope_pingpong_sim2real.launch.py \
  server:=192.168.10.100 ball_tracking_mode:=rigid_body ball_object:=Ball \
  onnx_path:=/abs/path/policy.onnx mode:=dry_run'
```

## The two "world" frames (the core sim2real gap this bridge closes)

| | HOPE table frame (planner/mocap "world") | Policy frame (training env origin) |
|---|---|---|
| origin | P1 near-side LEFT table corner, ON the surface | ground under robot base_link at boot |
| +x | toward opponent (P2) | robot boot heading |
| z=0 | table surface (floor = −0.76) | the FLOOR |

`wbc_runner` converts every table-frame input (world-frame `/racket/command`,
`/P1/pose`) into the policy frame via a rigid yaw-only transform
(`hope_wbc_runner/world_frame.py`), built from `robot_start_xy_table` +
`robot_start_yaw_table` and refined by averaging the first `/P1/pose` samples
(`origin_autocapture`). `frame_id="base_link"` targets (planner_imitate) pass
through untouched.

Swing side (forehand/backhand) is picked from the **base-relative** target Y —
the raw table-frame Y is always negative (table spans y 0..−1.525) and would
always pick forehand.

## What was fixed on 2026-07-03 (why it didn't work before)

1. **Relay heard nothing**: the launch forces `multi_sensor:=true`, which makes the
   vendored vrpn_mocap publish `.../pose_id_<n>`; the relay's topic matcher only
   accepted `pose`/`pose<n>`. Fixed in `hope_bringup/scripts/avatar_pro_vrpn_relay`.
2. **No table→policy transform anywhere**: world-frame targets and `/P1/pose` fed the
   obs raw (wrong origin, wrong z-datum, base-target obs ~1 m off). Fixed in
   `wbc_runner` + new `world_frame.py`.
3. **Always-forehand bug**: swing sign came from raw world Y. Fixed (base-relative).
4. **Mocap orientation garbage**: mocap is position-only; the runner used its
   quaternion for `projected_gravity`. Now: IMU orientation with boot yaw-align
   (`imu_yaw_align`, same idea as the C++ runner fix), mocap position only.
5. **Follow-through truncation**: the planner flips to `valid=False`/goes quiet the
   moment the ball passes the hit plane → the runner snapped to stand mid-swing.
   Now: committed swings (tts ≤ 0.35 s) latch through the clip end (`latch_swing_completion`).
6. **Stale clip layout**: the runner now reads `clip_seg_lengths`/`clip_strike_phases`
   from the ONNX metadata and overrides the yaml (old exports fall back + warn).
7. **OOD lunge protection**: base-relative target gate (`target_gate_*`) rejects
   unreachable targets → stand instead of an out-of-distribution lunge.
8. **Hardware bridge**: command topics now RELIABLE (AGI's ros2 config subscribes
   RELIABLE; best-effort pub never matches), merged state withheld until ALL four
   joint groups reported (was: zeros for missing limbs), timer mode no longer
   double-publishes.

Verified end-to-end in dry run: synthetic 300 Hz ball serves → planner → runner
alternating forehand (ball right) / backhand (ball left), strike frames aligned to
tts=0, latch riding follow-throughs to the clip end, 0 gate rejections in-box.

## Model compatibility (updated 2026-07-03 evening)

The runner now supports BOTH obs contracts, auto-detected from the ONNX input dim
(mirrors the C++ runner): **180-D** full (`model_15200` era) and **175-D
`deploy_parity`** (`model_9000_replane` / `model_7500_hopex` generation — racket
target relative to the live racket FK, world-base terms dropped). The FK port
(`hope_wbc_runner/racket_fk.py`) is cross-checked against the C++
`pp_racket_fk.hpp` to 4e-13. Verified end-to-end in dry run with
`model_7500_hopex.onnx`: forehand + backhand swings on the v4-lineage clip
frames (metadata seg 139/132, phases 0.47/0.333 auto-override the yaml).

## ⚠ Ball-physics calibration blocks real-ball play

`hope_planner.yaml drag_k: 0.8781` (salvaged single-recording fit) is ~8x the
physical value (k ≈ 0.11 for a 40 mm ball). With it, the planner's drag-aware
return solve demands 4.6–10 m/s racket speeds — ALL outside the policy's trained
envelope (max ~3.0 m/s) — so the runner's target gate correctly stands on every
ball. With a plausible k (0.11–0.15) demands drop to ~1 m/s. Re-fit from the
2026-07-03 venue recordings (`hope_calibrate`), or override `-p drag_k:=0.15`
for bring-up. `delta_t_flight` is now 0.65 s (0.5 s also pushed demands OOD).

## MANUAL FILL-IN CHECKLIST (in order, before hardware)

Mocap / CMTracker side:
- [ ] `server:=` — MCServer PC IP on the arena LAN (launch arg, no default that works).
- [ ] CMTracker rigid-body names must match the relay config exactly
      (`hope_bringup/config/avatar_pro_vrpn.yaml`): `ppt_object` (table, "PPT"?),
      `p1_object` (currently "ppp2" TODO), `p2_object` ("ppp3" TODO), and the ball
      (`ball_object:=Ball` with `ball_tracking_mode:=rigid_body`). Check with
      `ros2 topic list | grep vrpn_mocap` after the client is up.
- [ ] Mocap world calibration: meters, Z-up, origin at P1 near-side LEFT table
      corner ON the surface, +x toward P2. Verify with the net-center landmark:
      a marker at the net center must read ≈ (1.37, −0.7625, 0).
- [ ] `update_freq:=` — set to the camera rate (300). At 180 Hz the planner's
      31-sample velocity fit window doubles to ~170 ms (sluggish estimates).

Robot placement / runner (`hope_wbc_runner/config/wbc_runner.yaml`):
- [ ] `robot_start_xy_table` — measure the boot base_link ground XY in the table
      frame (tape measure from the origin corner is fine; mocap autocapture refines
      XY but a sane initial value keeps early commands correct).
- [ ] `robot_start_yaw_table` — PLACEMENT CONVENTION: put the robot down facing
      +x (the opponent) and leave 0.0, or measure the yaw. Position-only mocap
      cannot recover yaw; this value is trusted.
- [ ] `marker_to_base_xyz` — measure the P1 marker-cluster → base_link offset
      (also update `hope_bringup/config/hope_world_frame.yaml: mocap_to_base_link`).
- [ ] Boot procedure: robot standing STILL for the first ~2 s (origin capture:
      60 samples of /P1/pose; IMU yaw-align: 100 IMU samples).
- [ ] `target_gate_*` — tighten to the deployed model's validated box
      (model_15200: x[0.30,0.50] z[0.65,1.30]; model_9000: x[0.56,0.78] z[0.72,1.13]).

Planner (`hope_planner/config/hope_planner.yaml`):
- [ ] `x_hit` — must match robot placement: `x_hit ≈ robot_start_x + 0.65`
      (robot at −0.5 → x_hit ≈ 0.15). Constant `idle_reason=target_gate` in
      `/wbc_runner/diagnostics` means these disagree.
- [ ] `restitution_racket` — still the HITTER default 0.88 (TODO calibrate).
- [ ] Re-fit `drag_k`/`restitution_h`/`restitution_v` from ≥15 real trajectories
      (`ros2 run hope_planner hope_calibrate traj*.csv`) — current values come from
      ONE salvaged recording.

Hardware path (shadow/hardware modes):
- [ ] Source the AGI overlay for `joint_msgs` before launching the bridge/runner.
- [ ] Run the AGI backend with a ros2-enabled AimRT config for `/body_drive/(.*)`
      (pattern in `a3_deploy_example/.../a3_aimrt_config.ros2.yaml`). The default
      sim AND rockchip configs are iceoryx-ONLY — the ROS bridge hears silence.
- [ ] Kill any stray `planner_imitate` before using the real planner — BOTH publish
      `/racket/command` (`ps aux | grep planner_imitate`). A leftover
      `planner_imitate dry_run:=false` was found running from June 30.
- [ ] Clock sync between boards if headers cross machines.

## Debugging

- `ros2 topic echo /planner/diagnostics` — poses_received / valid_commands counters.
- `ros2 topic echo /wbc_runner/diagnostics` — idle_reason (`no_command` = nothing
  from the planner; `target_gate` = reachability rejections; `planner_invalid` =
  ball never crosses the hit plane validly), origin_source, imu_yaw_offset_rad.
- Runner CSV (`csv_path:=...`): per-tick `base_x/y/z`, `tgt_x/y/z` (both POLICY
  frame), `latched`, swing type, time_step. With the robot at its start pose,
  `base_*` ≈ (0, 0, pelvis height) and a hit-plane target ≈ (0.5, ±0.2..0.4, 0.7..1.1).
- Startup log `[frames]` line prints the full transform/gate/latch config; the
  `[obs verify]` block prints projected gravity + origin/yaw-align state on the
  first real-state obs.
