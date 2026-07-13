# Ping-Pong End-to-End Runbook: fresh machine → train → gates → MuJoCo → robot

**Audience:** you have a new robot (or a new machine), a full clone of this repo, and
you want the WHOLE process — which distrobox, which directory, which command, in order.
Written 2026-07-03; C++-only control path since 2026-07-04 (§0 retirement note).

> **CURRENT BASELINE (2026-07-08): `model_13200_footfix08` — the 110-D `hitter_pure` generation**
> (task `HOPEPingPongHitterPure`, run `2026-07-07_22-31-59_footfix`, iter 13200). Direct descendant of
> `model_12200_hitterpure` (identical faithful-HITTER MDP: independent station + station-relative fixed
> 0.70 m plane + learned velocity-normal), warm-resumed with `foot_orientation_weight=-0.8` to cure the
> pigeon-toe (hip_yaw internal rotation was reward-free with no lower-body imitation → toe-in while
> stepping). Verified: `eval_deterministic` composite **0.9925** (fh 0.996 / bh 0.978); C++⇔Python
> parity **9.5e-07**; Gate 2.5 **oracle 10/10** (incl. 20 s hold, post-swing hold, P7 cycles — needs
> the 110 idle-anchor fix in pp_policy.hpp, §6; pre-fix Δ=0 idle was 8/9 P7 creep, a runner artifact);
> **landing MC 97.8% on-table** (§7.5); **pigeon-toe FIXED** — moving-frame hip_yaw p95 **0.27 rad**
> (12200 was 0.94; envelope 0.41). Cost of the −0.8 penalty: pre-strike fall 0.4%→1.3% (still <2%).
> Lineage: model_12200 (foundation, strike 0.994) → 12900 (−0.3 footfix, toe p95 0.85, partial) →
> **13200 (−0.8 footfix, toe FIXED) = current**. 12200 remains a valid fallback (0.4% fall, toe-in).
> ⚠ **HARD REQUIRES `external_base` localization (live mocap in the arena / `--oracle-pelvis` in sim)
> — even to STRIKE**: its obs is world-frame + station-relative, so under `perfect_tracking` (the Δ=0
> dropout fallback) the swing itself diverges (Gate 2.5 perfect_tracking only **2/5** — stand + hold
> pass, swings fall). This is a STRONGER mocap requirement than the prior 177-D baseline (which could
> at least strike-in-place). 2026-07-11 更新：该 checkpoint 此后跑过一次 **legacy diagnostic**
> Gate3 planner 闭环（`13 PASS / 7 FAIL`；3 次发球中 1 次合法回球，并出现 1 次摔倒和明显漂移），
> 因此只能说明 Gate3 链路被执行过，不能写成 Gate3 已通过。Gate 1 的 110-D 分支、rockchip build
> 和真机发布仍未完成。该结果不得填入独立的 current exact-179 Gate3 单元；详见
> `docs/experiments/2026-07/EXP-GATE3-CURRENT179-D0.md`。
> Prior baseline (177-D `model_17400_hitter177`, 2026-07-06) is retained below as the last
> hardware-shipped generation and for the gate history.
Companion docs: [run_sim2real_bridge.md](run_sim2real_bridge.md) (the RETIRED python
ROS control chain — historical), `agi/a3_deploy_example/PINGPONG_NEW_CHECKPOINT_TUTORIAL.md`
(AGI-side checkpoint sync in depth), [run_training.md](run_training.md).

## 0. The big picture — ONE control path (C++); ROS only feeds it inputs

```
                       TRAINING (grasping box, Isaac)
                                   │ .pt
                                   ▼  export (grasping box)
                       policy.onnx (110-D hitter_pure — baseline model_12200)
                                   │
                                   ▼
        C++ deploy runner (a3_deploy_onnx_ref_pingpong) — THE control path
          scripted mode:  keys f/b/1/0 (built-in test targets; gates + bring-up)
          --planner mode: LIVE racket targets from hope_planner over two
                          low-rate ros2 topics (/racket/command_flat,
                          /a3/base_pose_flat) — the autonomous path
          /body_drive stays iceoryx in BOTH modes
                                   │ same binary
                       ┌───────────┴───────────┐
                       ▼                       ▼
                 AGI MuJoCo sim            robot MDU

        input side (plain ROS 2, laptop for sim / HDU for arena — carries NO control code):
          fake_ball_publisher (sim) / mocap vrpn relay (arena) → hope_planner
          [arena: on the HDU, ROS_DOMAIN_ID=232 to reach the MDU runner — §9.-1]
```

- **Scripted mode** (built-in test targets) is for policy validation in sim and first
  hardware swings (§6 Gate 2, §8).
- **`--planner` mode (2026-07-04) is the official autonomous path** for BOTH the sim
  closed loop (§7 Gate 3) and the hardware demo (§9): the SAME binary subscribes the
  live racket target (`/racket/command_flat`) + base pose (`/a3/base_pose_flat`) over
  the AimRT ros2 backend and drives the proven swing machinery through the C++ engage
  state machine (§9.6). This follows AGI's RobotIOBackend adaptation guide (own
  executable + backend subscription) — no ROS workspace, no python onnxruntime, no
  ros2-on-/body_drive needed on the MDU.
- The ROS side is INPUT ONLY: mocap / fake ball + `hope_planner` (pure python; the
  flat topics are `std_msgs`). It contains no policy runner.
- The ONNX auto-detects the obs contract by input dim (110 hitter_pure / 175 deploy_parity /
  177 hitter_footwork / 179 face-command / 180) and the clip layout + hitter_pure geometry boxes come from ONNX
  metadata. The baseline `model_13200_footfix08` is 110-D (as is its foundation `model_12200_hitterpure`).
  The 179 path has passed exact source/portable-build gates: it additionally requires exact
  `deploy_parity_face179` metadata, `--planner`, and formal planner parameter
  `racket_flat_schema:=3`; it has not yet passed a ROS/AimRT first tick or vendor Gate 3 runtime
  and is not a deploy baseline.

Before any 179-D Gate 3 attempt, run the production binary with
`--planner --no-publish --model-preflight-only`. This mode validates the ONNX graph and the full
metadata/lineage contract, prints parsed `publishable_model_contract=true`,
`training_contract_exact=1`, the accepted observation width and bound SHA values, then exits
before AimRT/backend initialization. `PpPolicy` does execute its deliberate zero-observation ONNX
prewarm inference; no policy-driver/backend tick or transport starts. It is intentionally weaker
than a no-publish first tick and does not replace the vendor MuJoCo Gate 3/Gate 3B run.
No-publish alone never relaxes model metadata. `--allow-legacy-model-diagnostic` is the only legacy
escape; it requires no-publish and is forbidden with model preflight.

> **RETIRED 2026-07-04 — the python control chain.** The old "Path B" (hope_wbc_runner
> + agibot_hardware_bridge driving `/body_drive` over ros2) is ABANDONED: closed-loop
> testing exposed a post-swing hold runaway (base-anchored hold carrot, no hold
> timeout → the robot chased its own hold target 12 m downrange) that the C++ runner's
> bounded-hold design (box-center hold seed + `hold_recover_s` → sticky static stand)
> does not have, and maintaining two engage state machines double-spends every fix.
> This runbook documents ONLY the C++ path. The python chain survives in git history
> (`git log -- hope_ws/src/hope_wbc_runner`) and in
> [run_sim2real_bridge.md](run_sim2real_bridge.md) (historical).

## 1. One-time machine setup

Three distroboxes (create once):

| box | for | inside it |
| --- | --- | --- |
| `grasping` | Isaac training + ONNX export | `/workspace/isaacsim` (Isaac Sim install) |
| `hope` | ROS 2 Jazzy: hope_ws, AGI sim, x86 deploy build | ⚠ broken `.bashrc`: run `source /opt/ros/jazzy/setup.bash` in EVERY shell |
| host (no box) | rockchip cross-build (Docker) | do NOT source ROS on the host |

**Not in git — copy from the team bundle / an existing machine** (a fresh clone alone
is NOT sufficient):

- `agi/a3_deploy_example/thirdparty/` big bundles: `onnxruntime/` (118 MB — parity +
  runner builds), `unitree_sdk2/`, `rknn_runtime/`, the `.deb`. (`joint_msgs/` IS now
  tracked in git — .gitignore fixed 2026-07-03.)
- `agi/a3_deploy_example/assets/a3_runtime/models/` — the deployed `.onnx` models.
- `agi/A3_MuJoCo_Sim/` sim **binaries** (`aimrt_main`, `.so` plugins next to
  `src/models/bin/`) — configs are tracked, binaries are not.
- `hope_training/.venv-motion/` (python env with mujoco+onnxruntime) — or rebuild it.
- `/workspace/isaacsim` inside `grasping`, plus
  `hope_training/whole_body_tracking/setup_train_env.local.sh` (machine-local paths,
  intentionally untracked).
- wandb login inside `grasping` (registry access for motion clips).

**Build the ROS workspace** (hope box):

```bash
distrobox enter hope
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/hope_ws
colcon build --packages-up-to hope_bringup hope_planner
source install/local_setup.bash
# sanity: pure-python tests need no ROS
cd src/hope_planner && python3 -m pytest test/ -q   # 73 passed
```

**Build the deploy package** (hope box):

```bash
distrobox enter hope
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example
bash scripts/build_a3_deploy_pkg.sh --arch x86_64 \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
# -> dist/a3_deploy_x86_64/  (runner binary, models/, config/, setup_ros2_msgs.bash)
```

## 2. ⚠ TODO / manual-calibration master list

Fill these before the corresponding stage; the chain tells you when one is missing
(planner diagnostics, the runner's `PLANNER: <status>` line, `[pp gate] REJECT` values).

**Planner ball physics + hit plane**

- [x] ~~zeroed placeholders~~ → **VENUE FIT LANDED (2026-07-03 recordings, merged
  from main 2026-07-04)** in `hope_ws/src/hope_planner/config/hope_planner.yaml`:
  `drag_k: 0.1261`, `restitution_h: 0.64` (no-spin grip equiv), `restitution_v: 0.9215`,
  `restitution_racket: 0.654` — matching node.py defaults; consistency-guard tests
  police drift. Re-fit ONLY if the venue/ball changes:
  `ros2 run hope_planner hope_calibrate traj1.csv ... traj15.csv` → paste values.
  (The old `drag_k 0.8781` fit was ~8× physical — do NOT restore it; it makes the
  planner demand 4.6–10 m/s racket speeds and the runner gate stands on every ball.)
- [x] `x_hit` is now **ADAPTIVE** (2026-07-04): tracks the live robot pose from
  `robot_pose_topic` (`/P1/pose` arena, `/sim/a3/pelvis_pose` sim) as
  `clamp(robot_x + x_hit_offset 0.67, [x_hit_min 0.0, x_hit_max 0.35])`. The static
  `x_hit` is only the fallback while no robot pose arrived. The [0.0,0.35] clamp is
  the table-collision protection (§9.3). Retune the window if robot placement changes.

**Mocap / arena — the VENUE FILL-IN SHEET (exact file : key for every manual value)**

All paths below are the **src** trees — edit src, then `colcon build --packages-select
hope_planner hope_bringup` (NEVER hand-edit `install/`, it is overwritten).
✎ = must fill at the venue; ✓ = correct default, verify only.

*(1) `hope_ws/src/hope_planner/config/hope_planner.yaml`* — ball physics + hit plane
+ the C++ runner's flat feed:

| key (line) | now | fill with |
|---|---|---|
| ✓ `drag_k` (~59) | `0.1261` venue fit | re-fit only if venue/ball changes: ≥15 trajectories → CSV (t,x,y,z) → `ros2 run hope_planner hope_calibrate traj*.csv` → paste. ⚠ if the fit comes out ≥0.5 it is WRONG (the rejected 0.8781 class) — keep 0.11-0.15 |
| ✓ `restitution_h` / `restitution_v` | `0.64`/`0.9215` venue fit | same calibrate run prints both (h = no-spin grip equivalent) |
| `restitution_racket` | `0.654` venue fit (paddle e const) | re-fit only with a new racket-bounce recording |
| ✎ `marker_to_base_xyz` (bottom, flat block) | `[0,0,0]` | SAME number as (2)'s `mocap_to_base_link.p1_xyz` — the planner applies it to `/P1/pose` before publishing `/a3/base_pose_flat` for the C++ `--planner` runner |
| ✓ `policy_z_offset` (bottom, flat block) | `0.76` (field 2026-07-07) | mocap/table-surface frame → policy/floor frame on ALL planner outputs; 0.76 = table height whenever the G5 calibration puts z=0 ON the surface (it does). Sim yaml keeps 0.0 |
| ✓ `position_scale` (avatar_pro_vrpn.yaml, relay) | `0.001` (field 2026-07-07) | this venue's CMTracker streams MILLIMETRES over VRPN; relay converts to metres. Set 1.0 if the export is switched to metres |
| ✓ `robot_pose_topic` | `/P1/pose` | arena default; sim overlay overrides |
| ✓ `x_hit` static fallback (~18) | `0.17` | only matters before the first `/P1/pose`; MUST equal `robot_start_x + 0.67` — if you place the robot at −0.8 (recommended, §9.3) set this to **−0.13** |
| ✓ `x_hit_offset/min/max` | `0.67 / 0.0 / 0.35` | retune the clamp only if robot placement changes (table-collision protection) |

*(2) `hope_ws/src/hope_bringup/config/hope_world_frame.yaml`* — the shared frame
definitions:

| key (line) | now | fill with |
|---|---|---|
| ✎ `mocap_to_base_link.p1_xyz` (~41) | `[0,0,0]` | the SAME marker→base_link translation as (1)'s `marker_to_base_xyz`; measurement procedure is in the file's comment block (CAD or tape-measure + level, refine by FK-vs-mocap while standing still) |
| `mocap_to_base_link.p1_rpy` (~42) | `[0,0,0]` | rotation only if the marker plate is mounted tilted |
| ✓ `landmarks_m` / `table_m` | ITTF geometry | fixed; used for the calibration checks below |

(The old item (2), `wbc_runner.yaml`, is RETIRED with the python runner (§0). Its
load-bearing values moved: robot placement → the physical convention in §9.3 + item
(1)'s `x_hit` fallback; the reachability gate → compile-time in `pp_policy.hpp`,
item (5).)

*(3) NOT yaml — launch args of `avatar_pro_hope_bridge.launch.py`* (defaults in the
launch file, override per run):

| arg | default | fill with |
|---|---|---|
| ✎ `server:=` | `192.168.10.100` (confirmed IP, but 2026-07-07 **UNROUTABLE** from HDU+MDU — 100% loss, no `192.168.10.x` route; G3/§9.-1) | the MCServer/CMTracker PC IP; the HDU (planner host) needs a route/NIC onto `192.168.10.x` before VRPN connects |
| ✓ `port:=` | `3883` | Chingmu VRPN default |
| ✓ `update_freq:=` | `300.0` | match the camera rate (ball needs ≥240) |
| ✓ `ball_tracking_mode:=` / `ball_object:=` | `rigid_body` / `Ball` | matches the standardized naming |

*(4) NOT ours to edit — CMTracker/AvatarPro side (procedures, no files):*

- Rigid bodies named EXACTLY `Ball`, `P1`, `P2` (+ `PPT` if the table is tracked) —
  these are the defaults in `hope_bringup/config/avatar_pro_vrpn.yaml`
  (`p1_object/p2_object/ball_object`), so nothing to edit if the names match; verify
  `ros2 topic list | grep vrpn_mocap`.
- World calibration IN CMTracker: metres, Z-up, origin = P1 near-side LEFT table
  corner ON the surface, +x toward the opponent. VERIFY with a marker at the net
  center → `/poses` must read ≈ `(1.37, −0.7625, 0)`.
- Boot ritual (no file): robot stands STILL, square, FACING +x, for ~2 s at every
  MOTION entry (`m`) — the C++ runner yaw-aligns the IMU there; entering MOTION
  while the robot moves/leans rotates every target transform (Gate 3 bug #9).

*(5) C++ `--planner` runner — CLI flags, not yaml* (defaults are the verified values;
touch only if the venue forces it): `--engage-min-tts 1.0`, `--cmd-timeout 0.5`,
`--invalid-grace 0.25`, `--swing-rest` (0.5 default). **110-D flags (2026-07-08, Gate-3 rally
campaign — see §7):** `--vel-box-center` (DEMO config: command the trained box-center velocity;
planner keeps WHERE+WHEN — use this on hardware), `--vel-gate-margin 0.30` (per-clip trained
vel-box gate slack; REJECT(110)+OUT-OF-BAND prints the demand vs box), `--stream-target`
(mid-swing streaming, default OFF — enable ONLY for a model trained with
midswing_resample_prob > 0; 13200 trained 0.0). Compile-time in PpPolicyConfig:
`engage_yaw_max_deg 20` (engage heading gate, status `yawed`), `static_handoff_yaw_max_deg 10`
+ near-station 0.3 m (static-stand handoff guards), `engage_settle_s 1.0` (MOTION-entry settle). **For the 110-D `hitter_pure` baseline the
per-clip engage gate (z-band, y-band, station geometry, plane_x=0.70) is METADATA-DRIVEN — read
from the ONNX `hitter_pure_pos_range_per_clip` / `_vel_range_per_clip` / `_base_target_range` at
boot** (the runner prints `[pp] 110 hitter_pure: station geometry from ONNX boxes: plane_x=0.70
fh y[-0.65,-0.15] z[0.67,0.97] bh y[-0.05,0.45] z[0.88,1.18]`), so a correctly-exported checkpoint
needs NO hpp edit for its trained envelope. Only the WIDE outer safety gate (`gate_x_lo/hi 0.20/0.90`,
`gate_y_abs 0.85`, `gate_z_lo/hi 0.55/1.40`, `gate_speed_max 3.5`) and the SCRIPTED Gate-2/2.5 test
target (`racket_pos_w_clip` / `racket_vel_w_clip`, re-synced 2026-07-07 to the hitter_pure 0.70 plane
centers fh (0.70,-0.40,0.82) / bh (0.70,0.20,1.03)) are compile-time in `pp_policy.hpp`. (Legacy
175/177 models still use the compile-time scalar box on the non-110 branch.) Rebuild rockchip if the
hpp changes or the model generation changes.

**Training / eval**

- [ ] A1 latency/jitter flag values (`racket.target_delay_steps` etc.) from timestamped
  venue recordings — currently default-off.
- [ ] Per-model reachability-gate box (compile-time `gate_*` in `pp_policy.hpp`, sheet
  item (5)) when the training boxes change — rebuild BOTH dists.
- [ ] **Next-retrain riders (2026-07-06, from the model_17400 gate campaign)** — none
  block the current baseline: (a) hold-gated ARM default-pose term (arms twist
  left-down/right-up during holds — upper-body imitation is `swing_only` and
  `hold_ready` std 1.5 only constrains the racket point, so hold arm posture is
  reward-free; q_des clamp protects the limits meanwhile); (b) hold-gated knee
  soft-limit (chronic `right_knee` clamp at ~86% of ticks, max viol 1.29 rad — safe
  but habitual); (c) if station range is widened toward HITTER ±0.75 m, revisit
  entropy 0.01 / hold_steps_range [50,200] / base_position 1.5 / episode 16 s (the
  2026-07-06 audited knob set).
- [ ] **HitterPure successor rider (ported from jiayi `5e97504`)** — `arm_hold_discipline`
  now exists as a default-off hold-gated L1 term, and the native runner can re-arm the nominal-arm
  hold on a fresh policy entry. Treat this as a candidate recipe until the exact ONNX, resolved
  reward manifest, x86/rockchip build and Gate 2.5 record are attached; it does not change the
  hardware-shipped 177-D default.

**Robot / deploy**

- [ ] Rockchip sysroot tarball (`scripts/export_rockchip_sysroot.sh` output) present in
  `thirdparty/` for the cross-build.

## 3. Train (grasping box)

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh        # defines hope_isaac_py
# BASELINE generation = HOPEPingPongHitterPure (110-D faithful HITTER repro):
hope_isaac_py scripts/train.py task=HOPEPingPongHitterPure algo=ppo headless=true
# resume from a checkpoint (the 2026-07-07 backhand-normal-sign fix was applied as a warm-resume):
hope_isaac_py scripts/train.py task=HOPEPingPongHitterPure algo=ppo headless=true \
  checkpoint_path=logs/rsl_rl/agibot_a3_hope_hitter_pure/<run>/model_XXXX.pt
```

Runs land in `logs/rsl_rl/agibot_a3_hope_hitter_pure/<date_time>/`. Health signals (tensorboard):
`strike_composite_success_exact` climbing with fh/bh balanced (⚠ the DETERMINISTIC eval is far
higher than the stochastic W&B rollout — model_12200 read 0.75 rollout but 0.994 deterministic;
the deployed ONNX is the mean), `pre/post_strike_fall_rate` < a few %, and WATCH
`Policy/mean_noise_std` — late-run std inflation makes reward curves lie; judge checkpoints by
`eval_deterministic` + the gates below, not W&B. **hitter_pure gotcha:** a single global
`mount_normal_sign` pins the BACKHAND face-normal ~137° off velocity (composite 0 for bh) — the
baseline uses per-clip `mount_normal_sign_per_clip=(1.0,-1.0)` so both faces strike (2026-07-07).

## 4. Export ONNX (grasping box)

```bash
distrobox enter grasping
# 110-D hitter_pure generation (HOPEPingPongHitterPure, CURRENT BASELINE):
bash ~/workspace/HOPE/hope_training/whole_body_tracking/scripts/export_onnx_hitter_pure.sh \
  logs/rsl_rl/agibot_a3_hope_hitter_pure/<run>  [model_XXXX.pt]     # omit ckpt = newest
#   e.g. the baseline: ... logs/rsl_rl/agibot_a3_hope_hitter_pure/2026-07-07_13-28-13 model_12200.pt
# 177-D hitter_footwork generation (HOPEPingPongHitter, prior baseline):
bash ~/workspace/HOPE/hope_training/whole_body_tracking/scripts/export_onnx_hitter.sh \
  logs/rsl_rl/agibot_a3_hope_hitter/<run>  [model_XXXX.pt]
# 175-D deploy_parity generation (legacy lineage):
bash ~/workspace/HOPE/hope_training/whole_body_tracking/scripts/export_onnx_deploy_parity.sh \
  logs/rsl_rl/agibot_a3_hope_deploy_parity/<run>  [model_XXXX.pt]
# -> <run>/exported/policy.onnx  (auto-kills the hung Isaac after the file appears)
# 110 exports MUST carry actor_obs_contract=hitter_pure + clip_seg_lengths/clip_strike_phases +
# hitter_pure_pos_range_per_clip / _vel_range_per_clip / _base_target_range metadata (the C++ runner
# reads the engage gate boxes + station geometry from them; the export script verifies the keys and
# WARNs if any are missing — a missing box drops the engage gate to the wide gate_z_lo/hi fallback).
# (177 exports similarly MUST carry actor_obs_contract=hitter_footwork + ref_reach_offset_xy.)
```

⚠ **The export must bake the SAME motion clips the run trained on.** The script pins
the hopex re-grounded clips (== registry sources `hope_forehand:v2`/`hope_backhand:v1`)
that the 2026-07-03 lineage uses. For any other lineage, verify first:

```bash
grep -oE "'motion_file': \[[^]]*\]" \
  ~/workspace/HOPE/hope_training/wandb/wandb/run-*<wandb_id>/logs/debug.log | sort -u
# then override:  FH=/path/fh.npz BH=/path/bh.npz bash scripts/export_onnx_deploy_parity.sh ...
```

(A v4-clip export of a hopex-trained checkpoint failed the deploy gate with 2.6 m/s
velocity errors — same weights, wrong baked references.)

## 5. Gate 1 — training-side MuJoCo (host, no distrobox needed)

The deploy-faithful behavioral gate: nominal-stand start, no teleports, fall-only
termination. **PASS = falls 0, completion 1.0 per clip, high composite.**

> **[110-D baseline] MuJoCo support is now implemented.** The historical `model_12200`
> result below was nevertheless produced only by Isaac `eval_deterministic` and must not be
> relabelled as a MuJoCo score. Fresh formal 110-D grading uses the immutable schedule, common
> MJCF `stand` ready state, all-attempt denominator and schema-v3 model/plant contract. The
> historical command was:
> ```bash
> hope_isaac_py scripts/eval_deterministic.py task=HOPEPingPongHitterPure algo=ppo headless=true \
>   num_envs=256 +steps=1200 +tail=400 '+noise_scales=[0.0,0.05,0.1]' \
>   checkpoint=logs/rsl_rl/agibot_a3_hope_hitter_pure/2026-07-07_13-28-13/model_12200.pt
> ```
> `model_12200` (baseline, 2026-07-07): composite **0.9936** (fh 0.996 / bh 0.992), pos err 2.4 cm,
> vel err 0.18, pos/vel/normal pass all > 0.99, pre-strike fall 0.5 %, post 0.1 % — the best of the
> 12000–12400 window (all clustered 0.982–0.994; it won on composite AND lowest fall rate). TODO:
> re-export a fresh schema-v3 110-D artifact and run BankExam before promoting this lineage.
>
> **[x-LOCKED generation, 2026-07-08] The task itself is now x-plane-locked** —
> `HOPEPingPongHitterPure.yaml` (+ the mirrored env-cfg default) pins `base_target_x_range: [0,0]`,
> so the station x is FIXED at spawn and the absolute striking plane no longer wobbles ±0.10 (it now
> equals the deploy planner's fixed `x_hit=1.03` exactly — a train↔deploy alignment fix, not just a
> constraint). The G1 command above is UNCHANGED (it inherits the lock), but two things change:
> * baseline sanity: `model_13200_footfix08` scores **0.9935** on the locked task with zero
>   retraining (fh 0.998 / bh 0.990) — the lunge was never load-bearing for striking.
> * NEW x-lock criteria for x-locked candidates (report rows already in `eval_deterministic`):
>   `base_drift_fwd_per_swing` ≤ **0.03 m** (was ~0.10 tolerance in the x-free world) and
>   `base_dist_from_origin` tail-mean ≤ **0.20 m**; watch `base_pos_error_x` ≪ `base_pos_error_y`
>   (the x-locked / y-footwork signature). Footwork budget lives in y ONLY (station y ±0.40).
>
> The 177/175 MuJoCo gate below is retained for those generations:

```bash
cd ~/workspace/HOPE/hope_training/whole_body_tracking
../.venv-motion/bin/python scripts/mujoco_eval_onnx.py \
  --onnx logs/rsl_rl/agibot_a3_hope_hitter/2026-07-06_00-56-46/exported/model_17400_hitter177.onnx \
  --motion-files artifacts/hope_forehand_hopex/motion.npz artifacts/hope_backhand_hopex/motion.npz \
  --noise-scales 0.0 --pd-mode implicit --deploy-faithful --df-clips both --steps 6000
# cross-checkpoint comparison on fixed protocols (appends to logs/scoreboard/scoreboard.csv):
../.venv-motion/bin/python scripts/scoreboard_eval.py --onnx <...> --label <name> \
  --motion-files artifacts/hope_forehand_hopex/motion.npz artifacts/hope_backhand_hopex/motion.npz
```

`scoreboard_eval.py` now fails closed before launching a protocol when an existing CSV header is
not exactly the current schema. Use a fresh output root or explicitly migrate an older scoreboard;
never append wider current rows beneath a historical header.

Reference: `model_17400_hitter177` (baseline, 2026-07-06) scored 0 falls × 3 seeds at
6000 steps, completion 1.0, composite 0.966, pos err ~6-7 cm; the prior 175-D
`model_7500_hopex` reference was composite 1.0 both clips, 7/7 swings, 0 falls at
1500 steps. ⚠ 177 models: the harness feeds a LIVE station through holds (training
semantics). `PP_DF_HOLD_DZERO=1` restores the old Δ=0-during-hold pinning for A/B
only — with it the policy free-wanders meters during holds and falls off-station
(that is the localization-DROPOUT behavior, not the nominal deploy path).

## 6. Sync to the AGI side + Gate 2 (policy-only in the official sim)

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example
# stage asset + rewrite runtime cfg + C++<->Python parity (obs-contract-agnostic; expect PASS ~1e-6).
# The sync script auto-repoints config/a3_runtime_config.pingpong.yaml model_path to the new onnx:
PYBIN=~/workspace/HOPE/hope_training/.venv-motion/bin/python \
  bash scripts/sync_pingpong_model.sh \
   ~/workspace/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_hitter_pure/2026-07-07_22-31-59_footfix/exported/model_13200_footfix08.onnx \
    model_13200_footfix08
# (baseline model_13200_footfix08 = model_12200 warm-resumed with foot_orientation_weight=-0.8 for the
#  gait fix; run 2026-07-07_22-31-59_footfix iter 13200. parity: max|delta| seed 9.5e-07 -> PASS.)
# 110-D note: the engage gate boxes come from ONNX metadata — no hpp box edit needed for the trained
# envelope. Only the SCRIPTED Gate-2/2.5 test target racket_pos_w_clip in pp_policy.hpp needs to match
# the generation (re-synced 2026-07-07 to the hitter_pure 0.70 plane; §2 item 5). Then rebuild:
bash scripts/build_a3_deploy_pkg.sh --arch x86_64 \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml

# Gate 2: free-base swings in the OFFICIAL AGI MuJoCo sim (C++ runner, scripted targets).
# ⚠ 110-D hitter_pure HARD-REQUIRES external_base — it FALLS on the swing under perfect_tracking
# (world-frame/station obs degrades), so use the --oracle-pelvis (real base feedback) path:
bash scripts/pp_freebase_watch.sh --single-swing --oracle-pelvis  # + run scripts/run_oracle.sh in a 2nd terminal
# (perfect_tracking `bash scripts/pp_freebase_watch.sh --single-swing` will stand + hold but fall on
#  the swing for 110-D — that is the mocap hard-requirement, NOT a regression.)
# The HEADLESS automated qualifier is Gate 2.5 (§6 blockquote below) — the recommended way to
# qualify a new checkpoint (no viewer/keyboard). For model_12200: pp_gate25.sh --oracle = 10/10
# (with the 2026-07-07 110 idle-anchor fix; 8/9 with the legacy Δ=0 idle — see the blockquote below).
```

PASS per swing direction: ≥5 clean cycles, 0 falls, no guard trips, sync_miss 0.
Boot log must show `obs_dim=110 act_dim=31` and `[pp] clip layout from ONNX metadata:
seg_len={139,132} strike_phase={0.470,0.333}` — the legacy `{95,105}` (or `obs_dim=175/177`)
means a stale binary or wrong model. A 110-D `hitter_pure` model must ALSO print
`[pp] 110 hitter_pure: station geometry from ONNX boxes: plane_x=0.70 fh y[-0.65,-0.15]
z[0.67,0.97] bh y[-0.05,0.45] z[0.88,1.18]` (a `[pp WARN] 110 hitter_pure: ONNX lacks
hitter_pure box metadata` line = a bad export → the engage gate drops to the wide
gate_z_lo/hi fallback; re-export with `export_onnx_hitter_pure.sh`, do NOT edit the hpp).
⚠ This generation walks to its station AND its whole obs is world-frame/station-relative, so
run `--oracle-pelvis`: `perfect_tracking` runs the loops open (station obs → Δ=0 dropout) and
the 110-D swing DIVERGES (stand + hold still pass). (A 177 model instead prints `[pp] 177 hitter:
reach offsets from ONNX metadata: fh=(+0.700,-0.409) bh=(+0.706,+0.185)` and can strike-in-place
under perfect_tracking.)

> ⚠ **Gate 2 coverage limits (2026-07-04 audit)** — know what a PASS does NOT prove:
> the AUTOMATED portion drives exactly ONE path (PD_STAND warmup → MOTION at level 1
> → one swing → level-0 hold); every repeat swing, every f/b switch, and every hold
> longer than the operator's pause is HUMAN-KEYED — coverage depends on what the
> operator happened to press. Known holes, confirmed by the automated transition
> matrix (Gate 2.5 below):
> 1. **Long level-0 hold is NOT covered and DOES fall**: parked at level 0 the policy
>    hold sinks and falls at **~5 s** (measured, scripted mode, perfect_tracking) —
>    Gate 2 never sees it because '1' is pressed within seconds. Do not park at level
>    0; drop to 's' (PD_STAND) when idling.
> 2. **f/b switches are manual-only**: the mid-swing QUEUE latch and the at-hold
>    switch have no systematic coverage in Gate 2.
> 3. **Walked-position re-swings are masked under perfect_tracking** (base obs pinned
>    to the reference → the footwork loop runs open); only `--oracle-pelvis` runs
>    exercise the real-base version — manually.
>
> **Gate 2.5 — automated transition matrix** (headless, deterministic; no viewer, no
> keyboard):
>
> ```bash
> distrobox enter hope -- bash ~/workspace/HOPE/agi/a3_deploy_example/scripts/pp_gate25.sh            # perfect_tracking
> distrobox enter hope -- bash ~/workspace/HOPE/agi/a3_deploy_example/scripts/pp_gate25.sh --oracle   # + real base feedback
> ```
>
> It owns the runner on a pty (headless keys), stands the robot itself
> (reset-into-armed-PD_STAND), then drives: stand→m0 entry, 20 s m0 hold, m0→m1 fh
> swing+complete, 8 s post-swing hold, f→b at hold, bh swing from the walked
> position, mid-swing queued dir switch, latched swing, alternating cycles. Each
> phase prints PASS/FAIL individually (`[g25] ... MATRIX`) so the fragile transition
> is IDENTIFIED, not averaged away. Run BOTH loc modes when qualifying a new
> checkpoint.
>
> **Expected with `model_13200_footfix08` (CURRENT baseline; = 12200 lineage + gait fix)**: `--oracle` = **10/10 phases PASS**
> (2026-07-07 evening, WITH the 110 idle-anchor fix below) — including both historic hold-killers
> (P2 m0 hold 20 s, P3b post-swing hold 8 s) AND the P7 continuous alternating cycles, z≈1.05
> throughout.
> **⚠ The 110 IDLE-ANCHOR fix (pp_policy.hpp, 2026-07-07 evening) is LOAD-BEARING for this score.**
> The first build fed station Δ=0 at 110-D level-0 idle ("hitter_pure trains no hold") and scored
> **8/9 — P7.1 fell**. That fall was MISDIAGNOSED as a training gap (continuous-rally retrain was
> attempted: model_18000 — strike regressed 0.994→0.866 AND it diverged outright in the Δ=0 idle,
> walking +0.94 m through the world-fixed target). Actual cause: Δ=0 idle has NO pull-back, so
> follow-through displacement accumulates across swings against world-fixed scripted targets until
> a swing starts from untrained geometry (hitter_pure trains NO forward locomotion — station
> x ±0.10, drift 0.01-0.02 m/swing; the observed forward pigeon-toed creep is OOD behavior, not a
> trained gait). Fix: 110 level-0 now uses the 177-style FIXED-WORLD hold anchor (idle actively
> station-keeps; `idle_station_dzero_110` keeps the legacy Δ=0 as a compile-time A/B). Zero
> retraining needed. If a 110 model creeps forward between swings in this gate, check that flag /
> rebuild before blaming the checkpoint.
> `perfect_tracking` = **2/5** (P1 stand + P2 20 s hold PASS; **P3a fh swing FALLS** at z=0.15, and the
> later swing phases with it): for 110-D this is NOT the mild 177 P3b-only failure — the whole obs is
> world-frame/station-relative, so WITHOUT real base localization the SWING itself diverges. Consequence:
> **qualify 110-D checkpoints on `--oracle` ONLY; on HARDWARE live mocap (`external_base`) is a HARD
> requirement for the strike, not just the footwork.** Serve/point discipline (§9.3) is unchanged.
>
> **Prior baseline — expected with model_17400_hitter177 (177-D, verified 2026-07-06)**:
> `--oracle` = **10/10 phases PASS** — including the two historic killers: P2 m0
> hold 20 s (the 175-era models fell at ~5 s; the ARM-A hold training + the 177
> station anchor fixed it) and P3b post-swing hold 8 s. perfect_tracking = 3/4
> (aborts at `P3b post-swing hold`): **that FAIL is BY DESIGN**, not a regression —
> without real base localization the 177 station channel degrades to the Δ=0
> dropout fallback, and the post-swing recovery hold has no world anchor to lean
> on (the policy free-wanders and tips; same mechanism, measured in the training
> harness: falls at |torso| 1–2 m from ±0.1 m stations). Consequences: (a) qualify
> 177 checkpoints on the `--oracle` matrix; perfect_tracking checks only the
> strike-in-place + dropout-degradation path; (b) on HARDWARE, live mocap
> (`external_base`) is REQUIRED for the 177 generation — it feeds both the footwork
> loop and the hold anchor. Known cosmetic behavior: during the policy hold the
> ARMS drift/twist (left down / right up, occasionally kissing joint limits —
> q_des clamp protects): training gates upper-body imitation OFF during holds
> (`motion_body_pos_swing_only`) and `hold_ready` (std 1.5) constrains only the
> racket point, so hold arm posture is reward-free. Harmless in `--planner` mode
> (the static-stand handoff covers idle); NEXT-RETRAIN rider: a hold-gated
> arm-default-pose term (see §2 TODO).
>
> *(2026-07-06 harness fix: P6 "latched fh swing" used to read `applied=False`
> on a HEALTHY latch — the queued flip legally applies while the swing clock still
> sits at the windup clamp, i.e. BEFORE "swing complete", and the old scan baseline
> missed it. The 175-era "6/7" scores had the same false FAIL. pp_gate25.py now
> scans from the pre-`f` offset.)*
>
> **Historic note (175-era, model_11400, 2026-07-04)**: `P2 m0 hold 20s` failed at
> ~5 s in BOTH loc modes while the SAME onnx held 20 s in the training-side harness
> — the AGI sim's actuation model is stricter than Isaac's on marginal holds, and
> AGI calls their sim predictive of hardware. That gap drove the ARM-A hold
> retrain; it is CLOSED in the 177 baseline (oracle P2/P3b PASS above). The
> mitigations remain shipped and still apply to any future marginal model:
> `--planner` mode never parks on the policy hold (static stand except the bounded
> recovery window, §9.6); in scripted mode drop to `s` when idling.

## 7. Gate 3 — planner + control closed-loop in MuJoCo (C++ `--planner`, sim)

> **2026-07-12 launcher quarantine:** the tracked `pp_gate3_rally.sh` command below is retained
> only as historical behavior evidence. Do not use it for a new formal run. Static audit found
> broad `pkill -9`, conductor `pgrep -f` signalling, no owned PID/PGID/trap, hard-coded unbound
> paths, fixed `/tmp`/wildcard shared-memory cleanup, no formal-loader-first gate and a runner boot
> loop that proceeds after timeout. Use the **plan-only** static source procedure in
> [run_gate3_first_tick_harness.md](run_gate3_first_tick_harness.md). It has no runtime/arming,
> launch, signal, process-scan or runtime-lock path; it only runs read-only Git helpers with
> `GIT_OPTIONAL_LOCKS=0`. Any plan output must be outside source/train/eval worktrees and their Git
> dirs/common dirs; those clean identities are rechecked before the external atomic write. It
> explicitly leaves full `--first-tick-json`, pidfd+cgroup/supervisor
> startup ownership, complete PATH/AimRT `.so`/plugin closure, parser-backed config→MJCF binding
> and the runtime ledger/lock transaction blocked. It does not authorize a first tick or hardware,
> and a future first tick will not replace the no-reset Gate3/Gate3B behavior paper.

The closed loop runs the DEPLOY binary itself: fake_ball → REAL hope_planner (sim
profile, publishes the flat topics) → C++ runner in `--planner` mode
(`external_base` localization from `/a3/base_pose_flat`) → AGI MuJoCo sim (iceoryx
body-drive). This is the exact hardware wiring — the only sim-specific pieces are
the fake ball and the ground-truth base pose source. Input contract + engage
semantics: §9.6.
> **GATE 3 REDESIGNED for the 110-D generation (2026-07-08): `pp_gate3_rally.sh` — the
> CONTINUOUS-RALLY deploy rehearsal.** The per-point harness below (`pp_planner_closedloop.sh`,
> serve → ONE return → operator `p` → sim reset) was designed around the 177 walk-and-strike
> contract and structurally cannot test what the station-keeping hitter_pure generation must do
> in the demo: return serve after serve from its station with NO reset — post-swing recovery
> INTO the next engage, station-drift accumulation, fh↔bh alternation from the walked (not
> reset) pose. The rally gate runs the same exact hardware wiring, stands the robot ONCE, then
> counts PP_SERVES (default 12) consecutive serves with per-serve verdicts:
>
> ```bash
> distrobox enter hope -- bash ~/workspace/HOPE/agi/a3_deploy_example/scripts/pp_gate3_rally.sh
> # PP_VIEWER=1 to watch; PP_SERVES=12 PP_PAUSE_S=4.0 PP_RESET_Y=0.0
> # PP_EXTRA_ARGS="--vel-box-center"   <- the 110-D demo-verified runner flags (see below)
> # PP_DROPOUT_AT=5 injects a 1 s planner freeze right after serve 5's engage (dropout stress)
> # outputs: per-serve table + /tmp/pp_rally_report.json + per-tick 110-D obs /tmp/pp_obs.csv
> # deep-dive: python3 scripts/pp_rally_report.py /tmp/pp_obs.csv /tmp/pp_rally_report.json
> # conductor mirrors the demo operator: a robot standing but refusing to engage for 2 serves
> # (e.g. PLANNER: yawed) gets an operator re-stand (counted as a RESCUE, not a fall).
> ```
>
> **What its first runs on `model_13200_footfix08` found (2026-07-08 debug campaign — each fixed
> in pp_policy.hpp / the harness same night):**
> 1. **Backhand engage was mathematically impossible in planner mode** — the pre-side late gate
>    used the windup MAX (cutoff 1.0 s) which sat above the whole bh engage window [0.78, 0.87] s.
>    Fixed (windup MIN). The legacy per-point gate never caught it because its 177-era serves
>    arrived 20 cm below the 110 bh z-band and were rejected before the timing gate.
> 2. **The 110 idle hold target rode the base** (0.70 m ahead of wherever the robot is, re-anchored
>    per tick) — a moving carrot with no positional feedback; the untrained-hold policy charged
>    +0.83 m off-station in ~1 s on it. Fixed: hold target is WORLD-FIXED at the hold-station
>    anchor, per-side geometry, trained box-center velocity (the Gate-2.5-proven scripted-hold
>    obs family).
> 3. **The planner demanded out-of-band racket velocities** (vy +0.18 vs the trained fh box
>    [0.96, 1.96]; only |v|≤3.5 was gated) — the swing executed OOD commands. Fixed twice over:
>    engage/stream now gate against the per-clip `hitter_pure_vel_range_per_clip` ONNX boxes
>    (±`--vel-gate-margin` 0.30), and the physically-solvable planner velocities barely intersect
>    the trained boxes (they meet only near the low-vz corner), so the DEMO config commands the
>    trained box-center velocity instead: **`--vel-box-center`** (planner keeps WHERE + WHEN;
>    aim precision of the return is given up — returns go where the human demo's returns went).
>    A single `target_land_y` also cannot satisfy both sides (fh returns cross-court LEFT,
>    bh RIGHT by training); hope_planner now supports per-side aim
>    (`target_land_y_fh/_bh`, `delta_t_flight_fh/_bh`) — the rally harness sets
>    fh +0.70/0.40 s, bh −0.30/0.35 s (offline-solved 10/10 in-band for the sweep serves).
> 4. **Mid-swing target streaming defaulted ON but the model trained with
>    `midswing_resample_prob: 0.0`** — every stream update was an untrained obs transition (and
>    moved the derived STATION mid-swing, which even training's resample contract forbids).
>    stream_target now defaults OFF (`--stream-target` re-enables it, only for a model actually
>    trained with resample > 0).
> 5. **Swing execution in the AGI sim leaves the robot yawed 30–55° after some follow-throughs**
>    (the reference clips yaw ≤ ±21° mid-swing and END at ~0–6°; this is execution divergence,
>    fh and bh both, stochastic). Engaging from a yawed stand is far outside the trained start
>    distribution (measured: engage at −30° → 2 m sprint, violent fall) and the STATIC official
>    stand tips ~3 s after freezing onto a yawed/staggered stance (even +17°). Fixed with three
>    guards: an engage HEADING gate (reject while >20° off the engage heading, status `yawed`),
>    a MOTION-entry settle (1 s, no engage), and the static handoff now requires near-station
>    (<0.3 m) AND near-heading (<10°) — off those bounds the POLICY hold keeps balancing (never
>    fell in any run; 20 s g25-proven) until the operator re-stands. ⚠ hardware discipline:
>    re-enter MOTION only with the robot physically re-squared to table +x — EVERY 'm', not just
>    the first (yaw-align re-captures there, and the 110 world-frame obs make heading load-bearing).
> 6. **SHADOW→MOTION clock-domain corruption** (a swing engaged during a SHADOW preview latched
>    a phantom level-1 into MOTION) — planner-mode swing state is now fully reset at every
>    SHADOW/MOTION entry; also: guard-tripped mid-swing aborts now restart the recovery clock
>    (no more instant stiff-stand on a tilted robot), swing-speed keys are ignored in planner
>    mode, and the last-action obs is zeroed while the static stand owns the robot.
>
> **Result progression (12-serve protocol, PP_MIN_RETURN_RATE=0.5):** run 1 (pre-fix) fell on
> serve 1's follow-through charge; run 4 (idle anchor + vel gates) returned fh+bh back-to-back
> with no reset — the first-ever 110-D closed-loop rally pair — then tipped on a yawed static
> handoff; run 6 (`--hold-recover 9999`, policy hold forever) had **ZERO falls across all 12
> serves** (min z 1.011) but starved engages (the untrained policy-hold idle wobbles through the
> narrow engage windows) — confirming the architecture: static stand for engage-readiness,
> policy hold for recovery, 10° handoff gate between them.
> **FINAL (runs 7+8, `--vel-box-center`, all gates, reproducible): 12 serves, 0 falls, min z
> ≈1.00, station drift 0.15 m, 3 engages (fh/bh/fh) all swings completed safely — 2 fully
> returned+recovered, the 3rd still balancing on the policy hold at run end; 2 operator
> rescues.** That is the CERTIFIED demo behavior for `model_13200_footfix08`: ~1 clean return
> per stand at a 5.7 s serve cadence, `yawed`-rejects (not falls) after divergent
> follow-throughs, operator re-stand to continue. Nothing fell in 4 consecutive
> gated-configuration runs (~45 rally-minutes equivalent).
>
> **The model-level finding this gate surfaces (retrain rider):** hitter_pure never trained the
> post-swing regime (episodes end after the strike), so follow-through heading/stance quality in
> the stricter AGI sim is chaotic even at box-center targets. The demo protocol that is safe
> TODAY: serve → return → if the next serves are `yawed`-rejected, operator re-stand ('p', square
> the robot to +x, 's', 'm') → continue. Expect roughly one clean return per stand, sometimes
> two-plus. The durable fix is the rally/hold retrain done RIGHT (station-anchored hold + heading
> restoration + post-strike brake — model_18000's regression shows it must be a careful warm
> resume, not a from-scratch rerun).
>
> **[x-LOCKED generation, 2026-07-08 — the retrain rider above is being executed, and this gate's
> PASS bar changes with it.]** Progress so far: (a) HEADING solved in training — `hold_heading=1.0`
> (hold-gated, GAE-safe; NOT post_strike_brake, which is the proven precision killer) taught
> self-squaring; `model_15500_rallyv3` ran this gate with **0 rescues** (engages self-cleared, no
> `yawed` re-stands). (b) That unmasked FORWARD DRIFT as the next blocker: with nobody re-standing
> the robot, base-x creep accumulates — 3-run evidence: drift 2.37 m → FELL, 1.49 m → FELL,
> 0.38 m → survived. Root cause: the task never locked the x-plane (station x wobbled ±0.10, soft
> symmetric base_position, no x termination). Now fixed at the SOURCE (x-locked task yamls) and the
> x-locked RallyV3 retrain (warm from 13200) is the candidate pipeline.
> **PASS bar for x-locked candidates on this gate:** falls == 0 **AND rescues == 0** (no operator
> re-stands — heading recovery is trained now, so a rescue is a regression) **AND station drift ≤
> ~0.4 m over 12 serves** (the measured fall threshold sits ≥ 1.5 m; 0.38 m survived; the certified
> 13200 run drifted 0.15 m) **AND every engaged swing completes + recovers**. Return rate remains
> reported-not-gated while `--vel-box-center` trades aim for in-band velocities.

Closed-loop VERIFIED 2026-07-04 (§9.6 status); re-verified
2026-07-06 with the PRIOR 177-D baseline `model_17400_hitter177` on the full serve sweep:
`PP_POINTS=10` → **10/10 points PASS** (every point engage → swing complete →
recovery done, fall_guard=0, min pelvis z 1.010) across 10 distinct placements
(fh arrivals y −0.65..−0.20, bh y +0.02..+0.34 — the trained-box sweep).

**ONE COMMAND, headless (the standard way to run this gate):**

```bash
distrobox enter hope -- bash ~/workspace/HOPE/agi/a3_deploy_example/scripts/pp_planner_closedloop.sh
# add PP_VIEWER=1 before `bash` to WATCH it (opens the MuJoCo viewer, same test)
# Runs the WHOLE chain: AGI sim (iceoryx body-drive) + REAL hope_planner (sim profile,
# publishes the flats) + fake_ball + the C++ runner in --planner mode, conducted by
# scripts/pp_planner_conductor.py (owns the runner pty, stands the robot via
# reset-into-armed-PD_STAND, presses m only when verifiably standing).
# The conductor plays PER-POINT, mirroring the §9.3 hardware demo discipline: serve →
# return → post-swing recovery → 'p' (operator abort, point over) → reset to the start
# spot → 's' → 'm' → next point, for 3 points. The model holds ONE clean return per
# placement (known margin: the post-lunge static stand tips after ~5-7 s, and the
# walked-forward robot puts later serves out of reach) — parking it after the return
# is NOT part of the contract, on hardware OR in sim.
# PASS: conductor SUMMARY "points_ok=3/3 ... -> PASS" (each point: engage → swing
#   complete → recovery done, fall_guard=0; the limp collapse after each deliberate
#   'p' is the operator catch, NOT a fall). Logs: /tmp/pp_runner.log (runner),
#   /tmp/pp_planner.log, /tmp/pp_ball.log, /tmp/pp_sim.log.
# The [obs] line shows "PLANNER: <status>" (no_command/stale/planner_invalid/too_late/
# base_low/target_gate/rest/engage/swinging); "[pp gate] REJECT ..." prints the values
# behind every reachability rejection (z_w≈0.00 = the ball died before the plane —
# EXPECTED for serves arriving after the robot has lunged forward within a point).
```

Verification physics inside that harness (arena values DON'T work here — expected):
drag_k **0.05** on BOTH planner and fake_ball (the arena 0.15 quadratic drag kills
every slow lob at floor height before the plane — physically correct, and exactly why
§9.3 wants close-bounce high-pop serves); robot reset to x=0.33 so the adaptive x_hit
(robot_x+0.67) sits at the hop apex; serve cadence `pause_s 4.0` (the 4-6 s demo
cadence). **Serves (2026-07-06 coverage sweep, replaces the old fixed pair)**: 10
placements (5 fh + 5 bh, alternating) with per-placement vy solved against the exact
publisher physics — fh `[3.2,-0.24,0.5,-2.0,vy,3.6]` arrives y −0.65..−0.20 at z=0.725
(trained fh box y [−0.74,−0.14], z [0.72,0.92]); bh **retuned** to
`[3.2,0.24,0.5,-1.6,vy,5.0]` (slower vx + higher toss) so it arrives at z=1.008 —
the old vz=3.6 backhand serve arrived at z=0.725, **20 cm below** the trained bh box
[0.93,1.13]. The old fh `vy=+0.10` (arrival y=−0.13 = box edge; the −0.10 fix flagged
2026-07-03 had never landed here) is gone with the sweep. bh arrivals stay ≥ +0.02:
the runner classifies fh/bh by base-rel target-y sign, so negative-y backhands would
engage as forehands. The sim overlay also sets `target_land_y: 0.0` (centered sim
table) — the arena aim (−0.7625) pointed every return at the sim table's right edge
and skewed the demanded forehand vy to the WRONG SIGN vs the trained velocity box.
Default conductor = 3 points (3 distinct placements); `PP_POINTS=10` sweeps all 10,
and `PP_RESET_Y=<m>` offsets the start spot so the 177-D station command is nonzero
from the first serve. NOTE: the sim overlay's x_hit clamp is deliberately wide
([0.55,2.20]) so the plane follows the robot's walk-and-strike drift across serves —
expected in sim; the ARENA clamp ([0.0,0.35], hope_planner.yaml) is the
table-collision protection and must stay tight in real life.

**Manual / raw three-terminal variant** — ⚠ prefer `PP_VIEWER=1` above: the manual
stand dance below has a **~0.5 s** reset→`s` window that is genuinely hard to hit by
hand across two terminals, and missing it produces a state that LOOKS completely dead:
`s` on a lying robot trips the fall guard → PASSIVE, after which every reset collapses
limp and no key seems to do anything (recovery: press `s` again within 0.5 s of the
NEXT reset — or just use the conductor, which retries this loop for you). Keep this
variant for debugging individual pieces:

Terminal A — the sim, with viewer:

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example
A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh   # no MUJOCO_GL=egl → viewer opens
```

Terminal B — planner + fake ball (the drag-0.05 verification physics):

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
ros2 run hope_planner hope_planner_node --ros-args \
  --params-file ~/workspace/HOPE/hope_ws/src/hope_planner/config/hope_planner.yaml \
  --params-file ~/workspace/HOPE/hope_ws/src/hope_planner/config/hope_planner.sim.yaml \
  -p drag_k:=0.05 &
ros2 run hope_bringup fake_ball_publisher --ros-args \
  -p serve_forehand:="[3.2, -0.24, 0.5, -2.0, -0.10, 3.6]" \
  -p serve_backhand:="[3.2, 0.24, 0.5, -1.6, -0.10, 5.0]" \
  -p drag_k:=0.05 -p pause_s:=4.0
# fh vy=-0.10 (NOT the old +0.10: that arrived y=-0.13 = box edge; -0.10 -> y=-0.35 mid-box).
# bh retuned vx=-1.6 vz=5.0 -> arrives z=1.008 inside the trained bh box (old: 20 cm low).
# For the full 10-placement coverage sweep use pp_planner_closedloop.sh (serves:=... list).
# sanity: ros2 topic hz /racket/command_flat /a3/base_pose_flat   # both alive
```

Terminal C — the C++ runner (read the STAND ORDER WARNING first):

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --planner --start passive --official-stand
# boot log MUST show: target_src = PLANNER, localization mode = external_base,
# "racket target subscriber enabled" + "base pose subscriber enabled", and the
# ONNX-metadata clip-layout line (seg_len={139,132}; {95,105} = stale binary).
# then: run  bash ~/workspace/HOPE/agi/a3_deploy_example/scripts/reset_sim.sh  in
# terminal B and press 's' HERE within ~0.5 s (official gains catch the keyframe
# stand inside the limp-buckle window); wait until visibly stable, press 'm'.
# Keys: p = PASSIVE (operator abort), h = SHADOW (full engage+swing pipeline,
# publishes nothing), q = quit. 0/1/f/b are IGNORED in --planner mode (the planner
# drives engage + side).
```

**STAND ORDER WARNING (the "falls immediately" trap):** the sim robot is PASSIVE
until commanded — if you reset and wait, it collapses limp within ~0.5 s and no
later key picks it up off the floor. Reset INTO the catch: `reset_sim.sh`, then `s`
immediately. And NEVER reset after MOTION entry — it teleports the robot mid-hold
(the reset-during-MOTION trap). The one-command conductor handles all of this,
which is why it is the standard way to run this gate.

Each POINT looks like: `[pp engage] ... locked ... (clock tts0=...)` → full swing
with a forward lunge (~0.7 m, the trained walk-and-strike) → `swing complete` →
bounded post-swing policy hold → `post-swing recovery done -> STATIC official
stand` → conductor `p` + reset (point over). PASS = 3/3 points with `fall_guard=0`.
`PLANNER:` status cycling no_command/stale/planner_invalid/too_late between serves
is NORMAL, as are `z_w≈0.00` gate REJECTs late in a point (the robot lunged forward,
later serves die before the walked plane). Constant REJECTs from the FIRST serve of
a point = real geometry mismatch — read the printed values.

> **Why per-point (2026-07-04 evening finding):** a 65 s "park and rally" run fell
> ~6 s after the first return — the post-lunge stance is feet-staggered, and the
> STATIC official stand cannot actively rebalance it (the policy hold that could is
> itself only good for ~5 s, Gate 2.5). Not an input-chain bug: engage, swing,
> recovery and the sticky-stand handoff all executed cleanly, and every later serve
> was correctly gate-rejected (adaptive plane ahead of the dead ball). One clean
> return per placement IS the model's contract (§9.3); the durable fix is the
> post-strike homing / long-hold TRAINING item.

## 7.5 Landing analysis — does the strike actually put the ball ON THE TABLE? (host, CPU, no sim)

⚠ **A high `strike_composite_success_exact` does NOT by itself mean the return lands.** That metric
scores how well the racket matches the COMMANDED pos/vel/normal at contact; whether the ball then
clears the net and lands in the opponent half is the PLANNER's job (it solves the racket command
from a drag+impact model aimed at `target_land`). Nothing in Gates 1–3 closes that loop — Gate 2/2.5
feed scripted targets with no ball, and Gate 3's fake_ball only drives the planner (the MuJoCo sim
has no ball to hit back). This gate closes it in expectation.

`hope_planner.landing_mc` (2026-07-07) Monte-Carlos the EXACT deploy chain per sample: sample a lob
arrival in the trained per-clip bands → `RacketTargetPlanner.plan()` (the deploy drag shooting-solve
→ racket command) → **perturb the command by the policy's MEASURED execution error** (vel-norm σ,
normal-angle σ, strike-point σ from its `eval_deterministic` exact-error rows) → forward
`predict_paddle_contact` (the SAME venue-fitted impact law as training's virtual_ball, bit-for-bit)
→ drag free-flight → net-clearance + landing classification. All physics are venue fits (drag k
0.1261; paddle e(u_n) from 150 real strikes; table restitution from 101 bounces).

```bash
# host, no distrobox; pure-python + the planner's own physics
cd ~/workspace/HOPE/hope_ws/src/hope_planner
python3 -m hope_planner.landing_mc        # (venv-motion python if numpy is missing)
# to score a NEW checkpoint: edit the "champion" run() line to its eval_deterministic exact errors
#   run(vel_err, normal_err_deg, pos_err, label="model_XXXX")
```

**Results (per-model with measured errors + reference levels; arena robot placement, target_land
(2.055,−0.7625)). The four footfix-lineage checkpoints all land ~97–98% — the foot/stance gait
differences are cosmetic and don't touch the ball; on-table is set by strike accuracy, which is
near-identical (pos 0.024–0.026 m / vel 0.17 / normal ~4°) across them:**

| model (foot_orientation weight) | measured vel/nrm°/pos | ON-TABLE | dominant failure |
|---|---|---:|---|
| perfect execution (planner-solve sanity) | 0 / 0 / 0 | **100.0%** | — (lands at target, σ 8–10 cm) |
| `model_12200_hitterpure` (0, foundation) | 0.177 / 4 / 0.024 | 98.1% | net-clip 1.9% |
| `model_12900` (−0.3, toe partial) | 0.177 / 4 / 0.026 | 97.2% | net-clip 2.8% |
| `model_13000` (−0.4, mid) | 0.177 / 4 / 0.024 | 97.2% | net-clip 2.7% |
| **`model_13200_footfix08` (−0.8, BASELINE)** | 0.172 / 4 / 0.025 | **98.3%** | net-clip 1.7% |
| at the TRAINING pass thresholds (landing-insufficient) | 0.5 / 15 / 0.075 | **68.7%** | net 13% + short 14% |

**Takeaways:**
1. Champion-accuracy strikes land **~98%** — comparable to the HITTER paper's 92.3% hardware return
   rate; current `eval_deterministic` accuracy DOES translate to on-table.
2. **The planner solve is sound** (perfect execution → 100%, centered on the opponent half).
3. **⚠ The training pass thresholds are landing-INSUFFICIENT**: a policy scoring 100% "success"
   right at `strike_success_vel_thresh 0.5` / `normal 15°` lands only **67%**. So **rank checkpoints
   by the exact-error MEANS (pos/vel/normal), never by pass rates.** A landing-calibrated threshold
   set is ≈ vel 0.25 / normal 6° (→94%); tightening the thresholds is a safe METRIC recalibration
   (reward uses stds, not thresholds — no training impact).
4. **Net-clip is the dominant failure at every level** — planner-side levers (higher net margin,
   longer `delta_t_flight`) come before any training change if on-table rate is short.

**Honest bounds (not a substitute for the paper's 26-ball hardware protocol):** the arrival
distribution is synthesized (plausible descending lobs, not venue-recorded); spin is mild (0–30
rad/s); contact is assumed (pos err 2.5 cm ≪ paddle 7.5 cm radius); outgoing flight omits Magnus
(the planner's own model does too). Ground truth still needs a contact-ball sim or on-robot balls.

## 8. Deploy to the robot

> **[110-D baseline `model_13200_footfix08`: this is the pending promotion step.]** The x86_64 dist
> is synced + built + Gate 2.5-verified (§6). To put it on the robot: run the rockchip build below
> (the runtime cfg already points at `model_13200_footfix08.onnx` after the sync; the 2026-07-07
> idle-anchor edits AND the 2026-07-08 Gate-3 campaign fixes — bh engage window, vel gate,
> heading gates, hold anchor, stream-off default, `--vel-box-center` — mean the rockchip dist
> MUST be rebuilt; also `colcon build --packages-up-to hope_planner` wherever the planner runs,
> for the per-side-aim params), then
> rsync to the MDU and re-run hardware G2 (§9.-1). ⚠ 110-D has NO mocap-less mode — confirm live mocap at the venue
> before making it the on-robot default; keep `model_17400_hitter177` as the fallback until then.

Build the rockchip package — **on the HOST (Docker), not in a distrobox**:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
bash scripts/build_a3_deploy_pkg.sh --arch rockchip \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
# rebuild rockchip after ANY pp_*.hpp / model change — dist is never auto-rebuilt
```

Ship + run (scripted-target bring-up, the hardware-proven flow):

```bash
HDU=<hdu_wifi_ip>
rsync -azP -e "ssh -J agi@$HDU" dist/a3_deploy_rockchip/ agi@10.42.10.12:/agibot/a3_deploy/
ssh -J agi@$HDU agi@10.42.10.12
# on the MDU:
source /agibot/software/v0/entry/env/env.sh
export A3_TRANSPORT=iceoryx
cd /agibot/a3_deploy
taskset -c 4-7 ./run_a3_pingpong.sh --start passive --legs-passive --gain-scale 0.4 --single-swing
# staged: passive -> (hoist checks) -> stand -> motion; gain-scale 0.4 first, then raise.
# keys: 1/0 swing/hold, f/b forehand/backhand, q quit.
# boot log MUST show the metadata clip-layout line + the loc-mode line (stale-binary check).
```

**Planner-driven variant (`--planner`, 2026-07-04):** same binary, same staged bring-up,
plus live inputs. The run script auto-switches the AimRT cfg to
`a3_aimrt_config.pingpong_ros2body.yaml` (body-drive iceoryx + the two planner topics on
ros2). Boot log must show `target_src = PLANNER`, `localization mode = external_base`,
and BOTH `racket target subscriber enabled` + `base pose subscriber enabled` lines:

```bash
taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand \
  --gain-scale 0.4 --leg-gain-scale 1.0
# --official-stand + --leg-gain-scale 1.0 MANDATORY free-standing (knee gains, §9.7 STEP 5).
# 0/1/f/b keys are IGNORED (the planner drives engage + side); p/s/h/m still work — 'p'
# (PASSIVE) is the operator abort. Full bring-up + input contract: §9.6.
```

## 9. Planner + control in REAL LIFE (hardware) — the demo runbook

> **★★ To RUN the mocap policy test: §9.8 — the bare copy-paste version** (4 terminals, in
> order, no checks). **§9.7** is the same flow WITH verification at every step (use it when §9.8
> doesn't behave). The subsections below (§9.-1 topology, §9.0 gap checklist, §9.1b builds,
> §9.2b/c tests + body-drive) are the reference detail.

Scenario: HUMAN serves, robot returns. The robot does not serve. Baseline policy:
`model_13200_footfix08` — the 110-D `hitter_pure` generation (2026-07-08; = model_12200 lineage +
`foot_orientation_weight=-0.8` gait fix). Verification chain: Isaac `eval_deterministic` composite
**0.9925** (fh 0.996 / bh 0.978); C++⇔Py parity **9.5e-07**; x86_64 dist built + Gate 2.5 **ORACLE
10/10** (incl. the 20 s hold + post-swing hold + P7 cycles — needs the 110 idle-anchor pp_policy.hpp
fix, §6; pre-fix Δ=0 idle = 8/9); landing MC **97.8% on-table** (§7.5); pigeon-toe FIXED (hip_yaw p95
0.27 vs 12200's 0.94). Foundation `model_12200_hitterpure` (composite 0.9936, toe-in p95 0.94) is the
fallback lineage root.
**[PENDING for 110-D: Gate 1 MuJoCo (110 branch incomplete), Gate 3 closed loop, rockchip build,
hardware ship/G2 — see those sections.]** ⚠ **110-D HARD REQUIREMENT: live mocap** (`external_base`)
— STRONGER than the 177 generation: the whole obs is world-frame + station-relative, so without a
fresh base pose the SWING ITSELF diverges (Gate 2.5 perfect_tracking only 2/5 — stand + hold pass,
swings fall), not merely the hold anchor. Do NOT arm MOTION on hardware without the mocap relay
alive — there is NO mocap-less strike-in-place mode for this generation. **Prior baselines:**
`model_17400_hitter177` (177-D, 2026-07-06 — last HARDWARE-shipped + full Gate 1/2.5/3 chain; can
strike-in-place under perfect_tracking); `model_11400_hopex` (175-D, 2026-07-04 — the mocap-less
strike-in-place fallback). Until the 110-D rockchip ship + hardware G2 land, `model_17400_hitter177`
remains the on-robot baseline (§9.-1); `model_13200_footfix08` is the sim-verified successor.

### 9.-1 Verified robot topology + first hardware session (2026-07-07)

First on-robot bring-up (read-only recon + the G2 dry-run). **Two-box robot, both aarch64:**

| box | hostname | interfaces | reaches | runs |
|-----|----------|-----------|---------|------|
| **HDU** (head unit) | `hdu` | `eth_hdu` 10.42.10.10/24 · `wifi_hdu` 192.168.120.249/24 | the MDU (10.42.10.x, 0-loss L2) + the laptop (wifi). **NOT** the mocap LAN. | mocap relay + `hope_planner` (the flats' publisher) |
| **MDU** (motion unit) | `mdu` | `eth_mdu` 10.42.10.12/24 (gw 10.42.10.10 = HDU) | the HDU only. | the C++ `--planner` control runner (iceoryx body-drive + the two ros2 flats) |

Laptop→HDU = `ssh agi@192.168.120.249`; laptop→MDU = `ssh -J agi@192.168.120.249 agi@10.42.10.12` (password `1`, both hops). (An alternate jump `172.16.238.4` routes via **ProtonVPN** and may time out from some hosts — the wifi jump `192.168.120.249` is the reliable path.)

**Session results:**
- ✅ **Runner fully shipped + fresh on the MDU** — `/agibot/a3_deploy` has `model_17400_hitter177.onnx`
  (+ `model_p4_deployparity.onnx` 175 fallback), the aarch64 binary + run script (Jul 7 00:19), runtime
  cfg → the 177 model, and `a3_aimrt_config.pingpong_ros2body.yaml`.
- ✅ **G2 PASSES on the real MDU** (`--planner --dry-run`, no-publish, 25 s clean, no lingering proc). Boot log:
  `clip layout seg_len={139,132}`, `177 hitter reach offsets fh=(+0.700,-0.409) bh=(+0.706,+0.185)` (from ONNX
  metadata, no "computed from baked refs" warn), BOTH `racket target subscriber enabled` + `base pose subscriber
  enabled`, `backend started`, `target_src = PLANNER`, `localization mode = external_base(mocap)`. **The aarch64
  AimRT ros2 plugin loads on hardware — the historical G2 blocker is CLOSED.**
- ✅ **HDU→MDU DDS interop VERIFIED** — the runner runs on **`ROS_DOMAIN_ID=232`** (set by the MDU's
  `/agibot/software/v0/entry/env/env.sh`; both boxes `rmw_fastrtps_cpp`, `ROS_LOCALHOST_ONLY=0`). From an
  HDU shell with `export ROS_DOMAIN_ID=232`, `ros2 topic info /racket/command_flat` + `/a3/base_pose_flat`
  each show `Subscription count: 1` (the runner) across 10.42.10.x. ⚠ A fresh HDU jazzy shell is domain 0 =
  INVISIBLE to the runner — the planner shell MUST set 232. The hand-fed-flats path (§9.2b) is fully wired.
- ✅ **Planner builds on the HDU** — `colcon build --packages-up-to hope_planner` (69 passed / 4 skipped);
  `--packages-select` fails on the missing `hope_msgs` build-dep.
- ⛔ **hope_ws (mocap relay + planner) is on NEITHER box.** Build tooling IS present on both
  (colcon/cmake/g++, python3-numpy 1.24.2) → build it on the **HDU** (§9.1b). `hope_planner` is pure-python
  (builds now via `--packages-up-to hope_planner`; `hope_msgs` is a build-time dep, the runtime import is optional); **`vrpn_mocap` is C++ and needs libVRPN, installed
  nowhere and absent from the apt cache** — the one real build dep to source.
- 🟡 **Live-mocap ingress via the LAPTOP (2026-07-07 — corrects the earlier "network-blocked" claim):** the
  HDU/MDU can't route to the MCServer `192.168.10.100` (100% loss), BUT the **laptop is ON the mocap LAN**
  (`enx…` = `192.168.10.82/24`, reaches `192.168.10.100` 0-loss) AND on the HDU's wifi
  (`192.168.120.133` ↔ HDU `192.168.120.249`, 0-loss). So the mocap chain is: **laptop** runs `vrpn_mocap`+relay
  (already built in the `hope` box; ingests `192.168.10.100`) → publishes `/P1/pose`+`/ball` on **domain 232**
  over wifi → **HDU** runs `hope_planner` (subscribes those, publishes the flats to the MDU over `10.42.10.x`;
  the HDU bridges its two NICs on one domain) → **MDU** runner. **Laptop→HDU DDS on domain 232 VERIFIED
  2026-07-07** (HDU discovered + read a laptop-published probe; the laptop's ProtonVPN did NOT break it) — and
  HDU→MDU was already verified, so the full mocap transport is proven end-to-end. Remaining are venue items only:
  the `vrpn_mocap`↔MCServer VRPN connection (rigid bodies named `Ball`/`P1`/`P2`, world calibration) + the
  hope_planner params (G4/G5/G8). (The earlier "mocap blocked" conclusion only checked HDU/MDU — not the laptop,
  which is the natural ingress.)

**Corrected host assignment (supersedes the earlier "run the planner ON the MDU"):** runner on the **MDU**;
mocap relay + `hope_planner` on the **HDU** (it sees the MDU's DDS on 10.42.10.0/24 and is the box that will
route to the mocap LAN). The laptop cannot publish the flats to the MDU (different subnet, no DDS discovery
across the jump). The MDU is disqualified as the planner host: no mocap route + no libVRPN.

**What is unblocked right now** (no mocap needed): drive the runner from a **hand-crafted flat publisher on the
HDU** (decision-tree item 2) — publish `/racket/command_flat` + `/a3/base_pose_flat` (std_msgs/Float64MultiArray)
and exercise the real engage→swing pipeline on hardware. The 177 hold anchor still wants a live base pose, so
keep such tests to single strikes + `s`-to-stand between them (the perfect_tracking/Δ=0 caveat, §6).

### 9.0 Gap checklist — fill EVERY row before arming on hardware

Legend: ✅ = filled in the repo (2026-07-04), 🏟 = venue-day measurement/procedure,
⛔ = BLOCKING engineering gap (not a venue measurement — needs real work).
**For every 🏟 row, the EXACT file : key : measurement procedure is in the §2
VENUE FILL-IN SHEET** — G3/G4/G5 → sheet items (3)+(4), G6/G7 → §9.3 placement +
(1)'s `x_hit` fallback, G8 → (1)+(2) (the `--planner` flat feed needs
`marker_to_base_xyz` too), G9 → (1).

| # | item | where | status |
|---|------|-------|--------|
| G1 | **Planner inputs DDS-visible to the MDU** *(2026-07-04 REFRAMED — was: full hope_ws chain on the MDU)*. The `--planner` C++ runner (§9.6) eliminates wbc_runner, the hw bridge, and the python-onnxruntime wheel from the MDU entirely; the control loop is the native aarch64 binary. What remains: the two flat input topics must reach the MDU's DDS, and the laptop cannot see the MDU's DDS → run **mocap relay + hope_planner ON the MDU**. hope_planner is pure python (rclpy/numpy/core msgs — the AGI robot env already ships a ros2+rclpy runtime; the flats are `std_msgs`, no custom typesupport needed on the receive side). Remaining build items: **numpy on the MDU** (pip/apt, aarch64) and **vrpn_mocap** (one small C++ pkg: 3 .cpp + libVRPN — cross-compile with the rockchip sysroot or build on the MDU), plus the IP route MDU → MCServer LAN (VRPN is plain TCP/UDP — works if routable). **[2026-07-07 SUPERSEDED: build on the HDU, not the MDU — the MDU has no route to the mocap LAN and lacks libVRPN; the HDU sees the MDU's DDS on 10.42.10.0/24. Tooling present on both boxes; hope_planner builds now (pure-python), vrpn_mocap awaits libVRPN. See §9.-1 + §9.1b.]** | HDU | 🟡 |
| G2 | **aarch64 AimRT ros2_plugin loads on the MDU** *(2026-07-04 SCOPED DOWN — was: ros2-enabled `/body_drive`)*. `/body_drive` **stays iceoryx** in `--planner` mode; only the two low-rate flat topics ride the ros2 backend (`a3_aimrt_config.pingpong_ros2body.yaml`, same dual-plugin pattern as AGI's own teleop cfg). The aarch64 `libaimrt_ros2_plugin.so` ships in the rockchip dist; it loads clean on x86-in-box but has NEVER been exercised on the MDU (the x86 HOST hits an rclcpp ABI break — box works, host doesn't; the MDU is a third environment). Verify FIRST THING: `./run_a3_pingpong.sh --planner --dry-run` must reach "backend started" with both `subscriber enabled` lines and no undefined-symbol abort; then `ros2 topic hz /racket/command_flat` (after `setup_ros2_msgs.bash`) sees the planner. If the plugin won't load → fallback §9.5. **[2026-07-07 ✅ VERIFIED on the real MDU — plugin loads, both subscribers enabled, 177 metadata parsed, `backend started`, 25 s no-publish clean. Boot log in §9.-1.]** | MDU | ✅ |
| G3 | MCServer (CMTracker PC) IP on the arena LAN → `server:=` launch arg. No working default. **[2026-07-07: IP is `192.168.10.100`, but NEITHER robot box can route to `192.168.10.x` (100% loss from HDU + MDU) — this is now a real routing/interface gap, not just a value to paste. The HDU (the planner host) needs a route or NIC onto the mocap LAN before vrpn_mocap can connect. See §9.-1.]** | venue | 🏟 ⛔route |
| G4 | CMTracker rigid bodies named EXACTLY `Ball`, `P1` (robot), `P2`, `PPT` (2026-07-03 convention, avatar_pro_vrpn.yaml). Verify: `ros2 topic list \| grep vrpn_mocap`. | venue | 🏟 |
| G5 | Mocap world calibration: meters, Z-up, origin at P1 near-side LEFT table corner ON the surface, +x toward the opponent. VERIFY with a marker at the net center → must read ≈ (1.37, −0.7625, 0). | venue | 🏟 |
| G6 | Robot placement: **0.8 m behind the table edge on the forehand half** (physical placement, see §9.3 why). No yaml key anymore — the C++ runner localizes from the live mocap base pose; just update (1)'s `x_hit` static fallback to `robot_x + 0.67` (−0.8 → **−0.13**). | venue | 🏟 |
| G7 | **Placement convention**: the robot FACES +x (the opponent), square. The C++ runner yaw-aligns the IMU at every MOTION entry (`m`) — keep it still ~2 s there. There is no yaml yaw override anymore: place it square. | venue | 🏟 |
| G8 | `marker_to_base_xyz` (hope_planner.yaml, sheet (1)) + `mocap_to_base_link.p1_xyz` (hope_world_frame.yaml, sheet (2)): P1 marker-cluster → base_link offset. Measure per the procedure in hope_world_frame.yaml. `[0,0,0]` is usable if the cluster sits on the pelvis. | venue | 🏟 |
| G9 | Ball physics: **VENUE FIT IN THE YAML** (2026-07-03 recordings via main: drag_k 0.1261, restitution 0.64/0.9215/0.654; consistency-guard tests police drift vs node defaults). Spot-check on venue day; full re-fit only if venue/ball changed. The salvaged Jun-23 fit (0.8781) stays REJECTED (8× physical). | venue | ✅ (spot-check 🏟) |
| G10 | Planner adaptive x_hit: `robot_pose_topic:=/P1/pose`, `x_hit_offset 0.67`, clamp `[0.0, 0.35]` — **already the yaml defaults** (2026-07-04). The clamp is the TABLE-COLLISION protection: it stops the demanded plane (and the lunge endpoint ≈ plane − 0.67) short of the table edge. | — | ✅ |
| G11 | Runner engage-safety set (active-swing lock + frozen target, bounded post-swing hold → sticky static stand, engage-tts clock seed + clamp, invalid-flutter grace, base_low guard, **MOTION-entry yaw align**) — in the C++ runner (§9.6). | — | ✅ |
| G12 | Baseline ONNX on the robot: `model_17400_hitter177.onnx` staged in `assets/a3_runtime/models/` + runtime cfg points at it (2026-07-06 sync); **rockchip dist rebuilt AFTER the 2026-07-06 C++ changes** (177-D obs builder, reach-offset metadata parse, hold-station anchor — §8 build; x86_64 rebuilt + gate-verified same day). ⚠ 177 needs mocap (see baseline note above); fallback without mocap = `model_11400_hopex.onnx` (175-D). **[2026-07-07: rockchip dist shipped to the MDU + G2-verified on hardware (§9.-1). Note the shipped mocap-less fallback is `model_p4_deployparity.onnx` (175 forehand), NOT `model_11400_hopex` — stage 11400 too if you want its in-place hold.]** **[2026-07-08 successor: `model_13200_footfix08` (110-D, = 12200 + gait fix) synced + x86_64-built + Gate 2.5 oracle 10/10 (with the 110 idle-anchor fix — rockchip MUST be rebuilt to carry it) + landing MC 97.8% + pigeon-toe fixed (hip_yaw p95 0.27); TODO to promote it on-robot = rockchip build (§8) + rsync to the MDU + hardware G2. ⚠ 110-D has NO mocap-less mode, so keep `model_17400_hitter177` as the on-robot default until live mocap is confirmed at the venue.]** | laptop→MDU | ✅ 177 shipped+G2; 🟡 110-D built, ship pending |

**G1/G2 were the blockers; the 2026-07-04 `--planner` port (§9.6) shrank both** from
"build a full aarch64 ROS chain + re-plumb /body_drive" to "get two topics onto the MDU
+ confirm one plugin loads". **2026-07-07: G2 is now CLOSED (verified on the real MDU,
§9.-1); G1 remains open and reframed — the relay+planner host is the HDU, and the current
hard sub-blocker is the mocap-LAN route (G3) + libVRPN, not the plugin.** The demo decision
tree:

1. ~~`--planner --dry-run` on the MDU loads the ros2 plugin (G2 ✓)~~ **DONE** AND relay+planner
   run **on the HDU** (G1 — pending libVRPN + mocap route) → **autonomous demo via §9.6** (official path).
2. G2 ✓ but no live mocap yet → no live ball, but §9.6 can still be driven by a hand-crafted
   `/racket/command_flat` (+ `/a3/base_pose_flat`) publisher **on the HDU** (scripted serves,
   real engage logic). **← this path is UNBLOCKED today** (G2 passed; the HDU sees the MDU's DDS).
3. ~~ros2 plugin fails on the MDU~~ (didn't) → **fallback demo §9.5** (scripted keys) — still
   available on the proven scripted-mode binary.

### 9.1 One-time prep (laptop)

```bash
# HOST (no distrobox, no ROS sourced): rebuild rockchip after the C++ safety fixes
cd ~/workspace/HOPE/agi/a3_deploy_example
bash scripts/build_a3_deploy_pkg.sh --arch rockchip \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml

# ship (HOST):
HDU=<hdu_wifi_ip>
rsync -azP -e "ssh -J agi@$HDU" dist/a3_deploy_rockchip/ agi@10.42.10.12:/agibot/a3_deploy/
```

### 9.1b Build the planner chain on the HDU (2026-07-07 — the G1 procedure)

The mocap relay + `hope_planner` run on the **HDU** (`192.168.120.249` wifi / `10.42.10.10`
on the robot net) — the box that sees the MDU's DDS on `10.42.10.0/24`. Both boxes carry
colcon/cmake/g++ + `python3-numpy 1.24.2`, so build on-robot. `hope_planner` is pure-python
and runs flat-only at RUNTIME (its `hope_msgs` import is optional, `node.py:28`), but colcon
still needs `hope_msgs` BUILT — so use `--packages-up-to hope_planner` (NOT `--packages-select`,
which skips the dep and fails on a missing `hope_msgs/package.sh`). `vrpn_mocap` is C++ and needs **libVRPN** (absent
on both boxes, not in the apt cache) — source it first (apt if a Debian mirror is reachable,
else cross-build with the rockchip sysroot and ship the `.so`) AND give the HDU a route to
the mocap LAN (G3, §9.-1).

```bash
# LAPTOP → HDU: copy the overlay source only (no build products)
cd ~/workspace/HOPE
rsync -azP --exclude build --exclude install --exclude log \
  -e ssh hope_ws/ agi@192.168.120.249:~/hope_ws/

# ON THE HDU:
ssh agi@192.168.120.249
source /opt/ros/jazzy/setup.bash
cd ~/hope_ws

# the planner + its hope_msgs build-dep — pure-python, ALL the HDU needs:
colcon build --packages-up-to hope_planner          # NOT --packages-select (skips hope_msgs → fails)
source install/local_setup.bash
python3 -m pytest src/hope_planner/test -q          # sanity (expect all pass); import needs the build above
```

> ⚠ Do NOT run `colcon build --packages-up-to hope_bringup ...` on the HDU — it fails on
> `vrpn_mocap` (`FindVRPN.cmake` missing: no libVRPN on the HDU) and that is FINE: the mocap
> bridge (`hope_bringup` + `vrpn_mocap`) runs on the **LAPTOP** (§9.7 STEP 1, the only box that
> reaches the MCServer), never on the HDU. The HDU's entire job is `hope_planner`. (An earlier
> revision of this section suggested the full build on the HDU — superseded 2026-07-07.)

With (a) alone you can already drive the MDU runner from the HDU with a hand-crafted flat
publisher (decision-tree item 2) — the whole engage→swing pipeline on hardware, minus the
live ball. With (b) + the G3 mocap route, the full autonomous chain (§9.6) comes up.

### 9.2 Staged bring-up (each stage safe by construction)

⚠ **2026-07-07 host correction:** the mocap + planner steps (1, 2) run **on the HDU**
(`ssh agi@192.168.120.249`), NOT the MDU — see §9.-1. Only the runner steps (3, 4, 5)
run on the MDU. Until G1 (hope_ws on the HDU + libVRPN + mocap route) is closed, rehearse
on the laptop against the AGI sim, §7 (identical wiring).

Runner shells (MDU) first:

```bash
ssh -J agi@$HDU agi@10.42.10.12        # $HDU = 192.168.120.249
source /agibot/software/v0/entry/env/env.sh
```

Planner/mocap shells (HDU) first:

```bash
ssh agi@192.168.120.249
source /opt/ros/jazzy/setup.bash
source ~/hope_ws/install/local_setup.bash   # after §9.1b builds the overlay
export ROS_DOMAIN_ID=232                     # ⚠ MUST match the MDU runner (env.sh sets 232); domain 0 = invisible
export ROS_LOCALHOST_ONLY=0                  # ⚠ the HDU login env sets 1 → localhost-trapped even on 232
```

1. **Mocap sanity** — the bridge runs on the **LAPTOP**, inside the `hope` distrobox (⚠ NOT the HDU): the
   laptop is the only box that reaches the MCServer `192.168.10.100` (via its `192.168.10.82` mocap-LAN NIC; the
   HDU/MDU have no route), and its `hope_ws` already has `hope_bringup`+`vrpn_mocap` built. Laptop→HDU DDS on
   domain 232 VERIFIED 2026-07-07 (§9.-1), so the mocap topics reach the HDU planner.
   ```bash
   distrobox enter hope
   source /opt/ros/jazzy/setup.bash
   cd ~/workspace/HOPE/hope_ws && source install/local_setup.bash
   export ROS_DOMAIN_ID=232                     # MUST — so /P1/pose+/ball reach the HDU planner + MDU
   ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py server:=192.168.10.100
   # sanity (laptop): ros2 topic list | grep vrpn_mocap ; ros2 topic hz /poses  (ball ≥240 Hz)
   #                  ros2 topic echo /P1/pose --once   # net-center marker ≈ (1.37, −0.7625, 0)
   ```
2. **Planner sanity** — on the **HDU** (`ROS_DOMAIN_ID=232`, no robot motion); it receives `/P1/pose`+`/ball`
   from the laptop bridge over domain 232 and republishes the flats to the MDU:
   ```bash
   ros2 run hope_planner hope_planner_node --ros-args \
     --params-file src/hope_planner/config/hope_planner.yaml \
     --params-file src/hope_planner/config/hope_planner.hitter_pure.yaml \
     -p robot_pose_topic:=/P1/pose -p marker_to_base_xyz:="[<G8 values>]"
   # ⚠ hitter_pure.yaml = FIXED plane (x_hit_follow_robot:=false, x_hit 0.20) — same as the armed
   #   deploy (§9.7). Startup log MUST read "x_hit FIXED", NOT "FOLLOW-MODE" (follow-mode deletes
   #   the x anchor for the x-locked policy → drift-fall). Needs a rebuilt install (see note below).
   ros2 topic hz /racket/command_flat /a3/base_pose_flat   # both alive
   ros2 topic echo /racket/command_flat  # data[1]=1.0 (valid) on good serves,
                                         # data[3] (px) ≈ constant x_hit 0.20 (FIXED plane, not a range),
                                         # data[9] (tts) counting down 2.0 → 0
   ```
3. **G2 gate — the ros2 plugin loads** (first run of the binary; publishes nothing):
   ```bash
   cd /agibot/a3_deploy && export A3_TRANSPORT=iceoryx
   taskset -c 4-7 ./run_a3_pingpong.sh --planner --dry-run
   # must reach "backend started" with BOTH "subscriber enabled" lines and no
   # undefined-symbol abort; boot log shows target_src = PLANNER,
   # localization mode = external_base. Plugin won't load → fallback demo §9.5.
   ```
4. **Shadow** — staged start, then `h`:
   ```bash
   taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand \
     --gain-scale 0.4 --leg-gain-scale 1.0
   # --official-stand: the 's' stand needs the official knee gains to bear weight (§9.7 STEP 5).
   # passive → hoist checks → 's' (stand). Then 'h' (SHADOW): the FULL engage+swing
   # pipeline runs on real serves, publishing nothing. Verify [pp engage] fires on
   # good serves, the PLANNER status cycles sanely, [pp gate] REJECT only on
   # genuinely bad serves, projected gravity ≈ [0,0,−1] on the [obs] line.
   ```
5. **Hardware** — the ARM ritual below, then `m` (MOTION); gain-scale 0.4 first,
   raise with `]` once stable.

### 9.2b Hand-fed-flats hardware test (no mocap) — the unblocked path (wiring VERIFIED 2026-07-07)

Decision-tree item 2: with the mocap chain still blocked (G3 route to `192.168.10.100` + libVRPN),
drive the MDU runner from a hand-crafted flat publisher on the HDU. This exercises the REAL
engage→swing pipeline on hardware as a **strike-in-place** (the fed base pose is static → no
footwork; that needs live mocap). Transport is verified end-to-end (§9.-1).

⚠ **Every HDU shell must `export ROS_DOMAIN_ID=232`** (the MDU `env.sh` sets it; a fresh jazzy shell
is domain 0 = invisible to the runner). ⚠ **`--dry-run` and SHADOW publish NOTHING** (the robot is NOT
driven), so keep it HOISTED/supported until the MOTION step.

**Step 1 — iterate the target with ZERO motion (`--dry-run`, robot hoisted/safe):**

MDU:
```bash
ssh -J agi@192.168.120.249 agi@10.42.10.12
source /agibot/software/v0/entry/env/env.sh          # sets ROS_DOMAIN_ID=232
cd /agibot/a3_deploy && export A3_TRANSPORT=iceoryx   # ⚠ EXACTLY iceoryx — a typo like 'iceory' STICKS in the shell (re-export or unset)
taskset -c 4-7 ./run_a3_pingpong.sh --planner --dry-run --start shadow
# --start shadow runs the policy + engage machine from t=0 with NO key-press and NO motion
# (--dry-run never publishes). Equivalent: plain --dry-run then press 'h'.
```
HDU term 1 — base pose, ≥5 Hz continuous (engage drops if base stale >0.2 s):
```bash
ssh agi@192.168.120.249
source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=232
ros2 topic pub -r 30 /a3/base_pose_flat std_msgs/msg/Float64MultiArray \
  "{data: [1, 1, 0.0, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0]}"   # [schema,valid,x,y,z,qw,qx,qy,qz]: robot at origin, standing
```
HDU term 2 — fire ONE forehand engage (20 msgs @4Hz ≈ 5 s window, then exits):
```bash
ssh agi@192.168.120.249
source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=232
ros2 topic pub -t 20 -r 4 /racket/command_flat std_msgs/msg/Float64MultiArray \
  "{data: [1, 1, 0, 0.70, -0.41, 0.82, 2.0, 0.0, 0.5, 1.5, 0.0, 0]}"
#  [schema,valid,swing_sign,px,py,pz,vx,vy,vz,tts,strike_time,frame_code=0(world)]
#  fh target base-rel x0.70 y-0.41 z0.82 (station≈base → strike in place, no walk), tts1.5
```
Expect on the MDU (**VERIFIED on hardware 2026-07-07**, both MDU-loopback and HDU→MDU):
`[pp engage] forehand locked: tgt base-rel (+0.70,-0.41,-0.13) tts=1.45s (clock tts0=1.30s)` → PLANNER
`racket`→`swinging` → `[pp] swing complete -> level 0 (held stand) (auto re-arm after rest)` → re-engage,
all with ZERO motion. (base-rel z −0.13 = world 0.82 − pelvis 0.95; tts0 clamps to the ~1.30 s clip windup.)
If instead `[pp gate] REJECT ...`, read the printed values and adjust the target
(gate: base-rel x∈[0.20,0.90], |y|≤0.85, z∈[0.55,1.40], speed≤3.5; base z≥0.7; tts≥1.0). Side is by
base-rel y sign (y<0 = forehand). `PLANNER: no_command/stale` between bursts is normal.

**Step 2 — a real strike (MOTION), operator hands-on.** ⚠ FIRST bring up the body-drive backend
per **§9.2c** (stop the default `agibot_pm` controller, keep the HAL) — otherwise the AGI default
controller fights our runner over the motors. Robot supported per §9.2c's safety gate.
```bash
# MDU: restart WITHOUT --dry-run (AFTER the §9.2c body-drive bring-up):
# (2026-07-08, 110-D: add --vel-box-center — the Gate-3-rally-verified demo config; the
#  planner's solved velocities sit outside the trained boxes and are now gate-rejected.
#  Re-square the robot to +x before EVERY 'm', not just the first — heading is load-bearing
#  for the 110 world-frame obs, and `PLANNER: yawed` rejects until square. §7 findings.)
taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand \
  --gain-scale 0.4 --leg-gain-scale 1.0 --hold-recover 1.2 \
  --vel-box-center
# --official-stand + --leg-gain-scale 1.0 are MANDATORY free-standing (knee gains — §9.7 STEP 5 notes)
# stage: passive -> hoist checks -> 's' (STAND, real) -> keep still ~2 s (IMU yaw-align) -> 'm' (MOTION)
# HDU: keep term-1 base stream running; fire term-2's racket burst -> ONE real forehand strike.
# hand on 'p' (PASSIVE = abort). 's' to re-stand between strikes — the static fed base pose gives no
# walk-forward recovery anchor, so single strikes only until live mocap (the §9 177 mocap requirement).
```

### 9.2c Real-motion body-drive bring-up — stop the default stack, keep the HAL (⚠ 2026-07-07)

The dry-run/engage checks (§9.2b) publish nothing → no body-drive backend needed. A REAL strike
(`m` = MOTION) publishes body-drive over iceoryx, which needs the low-level HAL EtherCat up AND the
AGI default controller stopped — otherwise TWO controllers fight the motors. Recon 2026-07-07 on the
demo robot (model **A3_P1D0**): the boot service `agibot_pm` runs the default stack that OWNS the
motors — `process_manager` → `motion_player` + `start_motion_control` + `hal_ethercat`. Good news:
`iox-roudi` (the iceoryx broker our runner needs) runs INDEPENDENTLY (ppid 1) and SURVIVES stopping
agibot_pm; the HAL publishes joint + IMU state over iceoryx (no separate estimator process seen).
`start_hal_ethercat.sh` launches `aimrt_main_hal` with the model config, refuses if one is already
running, and its RouDi launch is commented out (assumes RouDi already up — it is).

⚠⚠ **SAFETY — stopping agibot_pm can DROP the robot.** It kills the controller currently HOLDING the
robot; until our runner takes over in MOTION the robot is uncommanded. **HOIST / support the robot
(feet off-load) BEFORE stopping agibot_pm.** Never on a free-standing, weight-bearing robot. These
are `sudo` + motion-enabling commands — operator-run (the automated harness never runs them).

Sequence (robot SUPPORTED; two MDU terminals):
```bash
# --- terminal HAL (MDU) ---
sudo systemctl stop agibot_pm                 # stops the default controller (RouDi survives)
pgrep -a iox-roudi                            # MUST still be running (our runner's iceoryx broker)
pgrep -a hal_ethercat && sudo pkill -TERM aimrt_main_hal   # drop the OLD HAL so the next line owns EtherCat
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_ethercat/start_hal_ethercat.sh   # A3_P1D0 -> hal_ethercat_a3_p1d0_cfg.yaml; BLOCKS (own terminal)

# --- terminal RUNNER (MDU) --- our runner is now the SOLE controller (then §9.2b Step 2 s->m + flats):
cd /agibot/a3_deploy && export A3_TRANSPORT=iceoryx
taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand \
  --gain-scale 0.4 --leg-gain-scale 1.0 --hold-recover 1.2
```
Restore the robot's normal stack afterwards: `sudo systemctl start agibot_pm`.

### 9.3 The ARM ritual + demo constraints (READ BEFORE ENABLING)

The C++ runner has NO ROS estop — the operator's hand stays on the runner keyboard;
**`p` (PASSIVE) is the abort** (robot goes limp: hoist/catch ready), and the
always-on fall guard drops to PASSIVE automatically if gravity-z says fallen for
>0.5 s.

```text
ARM sequence (runner keyboard, robot at its marked spot):
  1. robot standing STILL at the start spot, FACING the opponent (+x), square;
  2. 's'  — PD_STAND (official gains, pose-blended entry); verify weight-bearing;
  3. 'm'  — MOTION. Keep the robot untouched for ~2 s: the runner yaw-aligns the
            IMU at MOTION entry; entering MOTION while it moves/leans rotates
            EVERY target transform (Gate 3 bug #9).
  abort = 'p' at ANY moment (PASSIVE, zero gains).
```

- **Robot placement**: 0.8 m behind the table edge (`robot_start_x = −0.8`). The
  trained swing is a WALK-AND-STRIKE that lunges ~0.5–0.8 m forward — from −0.8 the
  lunge ends ~0.15–0.35 m short of the table. Starting at −0.5 puts the lunge INTO
  the table.
- **One clean return per rally point.** After the return the robot recovers on the
  bounded policy hold, then parks on the STATIC stand at its NEW position; the
  adaptive x_hit clamp ([0.0,0.35]) rejects serves it would have to lunge into the
  table for. Walk the robot back to the start spot between points (`p` → reposition
  → `s` → `m`; yaw re-aligns at each MOTION entry). Continuous rally without
  repositioning needs the post-strike homing behavior — a TRAINING item, not
  deployed.
- **Serving instructions for the human** (the policy needs ≥1.0 s of warning):
  - slow, HIGH lobs — apex ≥ 1.3 m, total flight to the robot ≥ 1.3 s. Hard/flat
    serves are stood out as `too_late` by design.
  - serve into the FOREHAND half first (robot's right); backhand works in the gates
    but has less closed-loop mileage.
  - aim the bounce mid-table; balls that would cross the robot's plane outside the
    planner window x∈[0.0,0.35] or the runner's reachability gate (base-rel
    x∈[0.20,0.90], |y|≤0.85, z∈[0.55,1.40], speed≤3.5) are gate-rejected (the
    robot just stands — that is the SAFE outcome, not a bug).
- **Failure = stand.** Every rejection path (late ball, unreachable target, planner
  flutter, fallen-robot guard) converges to a held stand, not a wild swing.

### 9.4 Honest status on hardware

The `--planner` C++ path is code-complete and closed-loop-verified in the AGI sim
(2026-07-04, §7/§9.6: engage → full swing → bounded recovery → sticky static stand
→ re-engage, zero falls). Still not exercised anywhere: the ros2 plugin on the
actual MDU (G2) and real mocap noise/latency through the engage logic — treat the
first hardware session as shadow-heavy (`h` before `m`, §9.2). The python
wbc_runner chain that used to back this section was RETIRED 2026-07-04 (§0); the
C++ runner reuses its planner and engage semantics.

### 9.5 Fallback demo (runs TODAY): scripted-mode choreography

If G1/G2 aren't closed by demo day: the proven C++ runner in SCRIPTED mode with an
operator on the keys, human serving for THEATER (ball not tracked):

```bash
# MDU (after 9.1 ship):
cd /agibot/a3_deploy
export A3_TRANSPORT=iceoryx
taskset -c 4-7 ./run_a3_pingpong.sh --start passive --gain-scale 0.4 --swing-rest 1.5
# stage up: p -> s (stand) -> m (MOTION); raise gain [ / ] once stable.
# operator presses f/b to match the server's side, 1 to swing; the 2026-07-04 fixes
# make the keys safe: f/b mid-swing is QUEUED to the next windup (no more OOD snap),
# a fall drops to PASSIVE automatically (fall guard), 's' mid-swing blends.
```

Honest framing if asked: scripted-target swings, not ball tracking. The mocap →
planner stack can run alongside on the laptop (stages 1–2 above work laptop-side —
only the ROBOT link needs the MDU) to show live `/racket/command` on a screen.

### 9.6 The official C++ `--planner` path (2026-07-04) — autonomous demo without the ROS chain

**What it is.** The deploy binary (`a3_deploy_onnx_ref_pingpong`) gained a live-input
mode built exactly per AGI's RobotIOBackend adaptation guide: two extra AimRT
subscriptions in `A3AimrtBackend` (registered only when `--planner` sets the callbacks,
mirroring the stock teleop hook) feed a C++ port of the (retired) python wbc_runner engage state machine
inside `PpPolicy`. The swing itself is executed by the SAME clip-clock/completion/latch
machinery Gate 2 proved. Everything below the target source is unchanged — obs build,
ONNX, decode, joint map, PD, watchdog, safety guards.

**Input contract** (both `std_msgs/Float64MultiArray`, RELIABLE, over the AimRT ros2
backend; hope_planner publishes both when `publish_flat_cmd:=true`, the default):

```text
/racket/command_flat  [schema=1, valid(0/1), swing_sign(ignored — side derives from
                       base-rel y), px,py,pz, vx,vy,vz, time_to_strike, strike_time,
                       frame_code(0=world/table, 1=base_link)]        (≥11 doubles)
                       schema=2 face179 uses exactly 16 doubles: the same 12-value
                       prefix + physical_striking_face_B_normal_w[3] + rho[1].
                       Phase-1 requires frame_code=0 (world/table), B.x>1e-6,
                       a unit B normal and rho exactly zero.
                       schema=3 formal179 uses exactly 20 doubles: schema-2 prefix
                       + shared control epoch + racket sequence + exact base-sequence
                       reference + mapped source-monotonic time.
/a3/base_pose_flat    [schema=1, valid(0/1), x,y,z, qw,qx,qy,qz]      (≥9 doubles)
                       ← the robot base in the SAME frame as the racket target
                       (arena: /P1/pose + marker_to_base_xyz; sim: /sim/a3/pelvis_pose)
                       schema=2 formal base uses exactly 12 doubles and adds the
                       shared epoch, base sequence and mapped source time.
```

Schema 1 is the default and remains unchanged for current 110/175/177/180 baselines. A 179-D
Gate 3 source rehearsal must launch `hope_planner` with `-p racket_flat_schema:=3`; the 179 runner
then refuses schema 1, missing/non-unit/non-opponent-facing face commands, nonzero rho, scripted
mode, or mismatched ONNX term/face/bank metadata. A bad/no-solution publisher row is converted to
an exact finite `valid=0` schema-2 revocation; the receiver also marks malformed/unknown traffic
after a live face tuple as `invalid_after`. Diagnostics report planner solves separately from
control-valid rows and count flat-contract rejects. Do not use this on hardware: the vendor MuJoCo
build and a content-addressed no-publish runtime ledger must pass first.

The schema-2 normal is physical striking face B, not the actor/bank's raw mount +Y/A normal. Once
the runner has selected clip0 forehand or clip1 backhand, it computes
`normal_A = [1,-1][clip] * normal_B`. It never applies that sign to position or velocity. Formal
metadata must bind the same exact sign table in the checkpoint contract and normal-envelope
payload; any disagreement fails model load.

A formal 179 ONNX must also carry the content-bound per-clip train-normal envelope described in
`docs/interfaces/policy_observation_action.md`. The loader verifies its payload SHA and exact
train-bank/source-family bindings. At engage, the runner selects clip0 forehand or clip1 backhand,
uses its sign to convert physical B to raw A, and requires raw A to remain in both that clip's
reference hemisphere and spherical cap. The check occurs before the atomic target/clock/side
commit. A positive-X physical-B unit normal whose converted raw A is outside trained support is
reported as `face_command_out_of_train_envelope` and does not start a swing. Models
exported before this contract intentionally fail to load and must be re-exported from the exact
schema-3 train bank. Model-only preflight prints the envelope payload/bank/family SHA triplet for
the audit ledger; it still does not initialize a backend or prove behavior.

The active swing uses one frozen schema-2 tuple. Post-swing recovery is still a known blocker:
the current rally runner synthesizes a base-anchored hold position while carrying the last swing's
velocity/normal, and the 179 training contract has not proven that hybrid observation. Until a
canonical recovery tuple or matching vendor-MuJoCo recovery paper is accepted, use the 179 path
only for wire/first-tick/single-swing diagnostics and do not report a continuous Gate 3 pass.
The envelope source gate is not a Gate 3 behavior pass: it neither proves all points inside the cap
collision-free nor closes self-hit/recovery/vendor-MuJoCo stability. Keep those gates open.

**Engage semantics** (port of the retired python runner's `_tick`; defaults, all
CLI-tunable): a fresh
VALID command engages only if it passes `--cmd-timeout 0.5` staleness,
`--invalid-grace 0.25` planner-flutter, `--engage-min-tts 1.0`, base-standing
(`base z ≥ 0.7`), and the reachability gate (base-rel x∈[0.20,0.90], |y|≤0.85,
z∈[0.55,1.40], speed≤3.5). On engage: side locks, target freezes, the ENGAGE tts
seeds the swing clock (`clock tts0=min(tts,max)` in the log — the strike meets the
ball), the clip runs once to the end (single-swing). Planner flutter/invalid
mid-swing is IGNORED (frozen target); the aborts are the safety guards and the
operator's `p` key. Localization defaults to `external_base` (live mocap base
position + yaw-aligned IMU attitude); a stale base stream (>0.2 s) blocks engage and
falls back to perfect-tracking obs with a loud warn — failure = stand, never a wild
swing.

**Level-0 policy: static stand except a bounded recovery window.** The model's
level-0 policy hold has only **~5 s of margin** (Gate 2.5, §6) — the runner therefore
NEVER parks on it: before the first engage it publishes the STATIC official stand
(the Python `_stand` design); after a completed swing it runs the policy hold just
long enough to actively balance out of the follow-through (`hold_recover_s` 2.5 s —
a static stand cannot do this part), then, quiescence-gated (upright + still, forced
by +3 s), blends to the STATIC official stand and stays there (sticky) until the
next engage. Verified: two full cycles incl. a re-engage FROM the static stand.
**Demo cadence**: serve every ~4-6 s (the policy is most stable swinging or briefly
holding), or estop (`p`) + reposition after each return per §9.3 — never rely on
more than ~3 s of autonomous policy hold.

**Rehearsal in the AGI sim (do this before hardware):** this is exactly Gate 3 —
§7 (one-command headless conductor + manual viewer variant, incl. the drag-0.05
verification physics and the reset-into-armed-stand mechanics).

**Hardware bring-up (MDU)** — after §9.1 ship: follow §9.2 (mocap → planner+flats
→ `--planner --dry-run` G2 gate → staged start + shadow `h` → ARM ritual §9.3 →
`m`). The G1 items (numpy [present] + vrpn_mocap [needs libVRPN] on the HDU, MCServer IP-routable) are in
§9.0.

**Status (2026-07-04, evening — closed-loop VERIFIED headless):** the full chain
fake_ball → REAL hope_planner (sim profile + flats) → C++ `--planner` runner
(external_base from the flats) → AGI MuJoCo sim ran end-to-end, repeatedly:
`[pp engage] forehand locked ... tts=1.25s (clock tts0=1.25s)` → full swing →
`swing complete` → post-swing policy hold → (in the faster-cadence run) a SECOND
engage from the walked position. Verified: both flat subscriptions, external_base
localization, every engage gate (no_command/stale/planner_invalid/too_late/base_low/
target_gate — the `[pp gate] REJECT` line prints the offending values), engage-time
tts transfer into the swing clock, side selection, single-swing completion, rest.
Bugs found+fixed during verification: engage originally ignored the planner tts
(strike always fired 1.30 s after engage — now `clock tts0=min(tts,max)`); the
pre-FIRST-engage hold now publishes the STATIC official stand (running the policy's
level-0 hold cold was never a validated regime and knelt the robot within ~2 s; the
retired python runner's `_stand` did the same); base_link-frame commands rotate velocity too;
stale-oracle engage is blocked. Residual (model-margin, NOT input-path): the
post-swing policy hold degrades after ~5-10 s and a second swing from the walked
position can fall — consistent with §9.3's demo spec (ONE clean return per placement,
reposition between points; continuous rally = the known post-strike-homing TRAINING
item). Still pending: ros2_plugin load on the actual MDU (the G2 gate above) and real
mocap noise/latency.

**Headless-harness mechanics** (baked into `scripts/pp_planner_conductor.py` /
`scripts/pp_gate25.py` — read this only when modifying them): the robot must be stood
by resetting INTO an armed PD_STAND — per attempt: publish a `mode:1` SimReset
(keyframe + `set_base` to place the robot), wait ~0.15 s, press `s` (official gains
catch it inside the ~0.5 s limp-buckle window), verify standing ~6 s, then `m`.
Arming `s` on a LYING robot instead trips the fall guard → PASSIVE → every later
reset collapses limp. The runner's keys need a real pty headless (`isatty` gate) —
`pty.openpty()` + `Popen(stdin=slave)`. Never let a reset fire after MOTION entry
(teleports the robot mid-hold = the reset-during-MOTION trap; use ONE persistent
rclpy publisher — per-call `ros2 topic pub --once` cold discovery can delay a reset
into the MOTION window, observed).

### 9.7 ★ FULL MOCAP POLICY TEST — copy-paste walkthrough (2026-07-07)

The end-to-end autonomous test: a human serves → mocap tracks the ball + robot base → the planner
computes a racket target → the C++ runner engages and swings → the robot returns. **Four roles, all
on `ROS_DOMAIN_ID=232`.** The whole DDS transport (laptop→HDU→MDU) was verified 2026-07-07; the
runner engage/swing was verified via hand-fed flats (§9.2b). What remains is the venue mocap config
and the physical bring-up.

```
  MCServer 192.168.10.100                LAPTOP (hope box)          HDU                    MDU
  (Ball/P1/P2 rigid bodies) --VRPN-->  vrpn_mocap + relay  --DDS-->  hope_planner --DDS-->  a3_deploy runner
                                       (only box that            (flats over 10.42.10.x)  (--planner)
                                        reaches the MCServer)                              --iceoryx--> HAL --> motors
```

**Pre-flight** (one-time; ✅ = verified this session, 🏟 = venue-day):
- ✅ MDU: `model_17400_hitter177.onnx` runner shipped; G2 ros2-plugin load verified (§9.-1).
- ✅ HDU: `hope_planner` built (`colcon build --packages-up-to hope_planner`, §9.1b).
- ✅ Laptop: `hope_bringup`+`vrpn_mocap` built in the `hope` box; laptop reaches the MCServer + HDU.
- 🏟 MCServer streaming rigid bodies named **`Ball`/`P1`/`P2`**, world-calibrated (net center ≈ (1.37,−0.7625,0)) — G4/G5.
- 🏟 Robot **0.8 m behind the table on the forehand half, facing +x, square** (§9.3); `marker_to_base_xyz` measured — G8.

---

**STEP 1 — Mocap bridge (LAPTOP, in the `hope` distrobox — ⚠ NOT the HDU)**
```bash
distrobox enter hope
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/hope_ws && source install/local_setup.bash
export ROS_DOMAIN_ID=232
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py server:=192.168.10.100
```
Verify in a 2nd laptop shell (`distrobox enter hope`; source; `export ROS_DOMAIN_ID=232`):
```bash
ros2 topic list | grep vrpn_mocap        # vrpn topics present = VRPN connected
ros2 topic hz /poses                      # ball ≥240 Hz (publishes ONLY while ball data flows)
ros2 topic echo /P1/pose --once           # MUST be METRES + mocap frame: |x|,|y| < 3 and z ≈ 0.15
                                          #   (marker height ABOVE THE TABLE SURFACE — the G5 mocap
                                          #   origin is ON the surface); net-center marker ≈ (1.37,−0.7625,0)
# Position values in the HUNDREDS = the mocap streams MILLIMETRES (this venue does) — covered by
# avatar_pro_vrpn.yaml `position_scale: 0.001` (relay-side mm→m, field 2026-07-07). Without it the
# planner's solver diverges on the first ball (FloatingPointError) — now crash-guarded, but blind.
> `/vrpn_mocap/...` present but `/P1/pose` empty → the MCServer isn't streaming rigid bodies named
> `Ball`/`P1`/`P2` (fix on the mocap software — G4).
> Notes (field 2026-07-07): (a) run ONE bridge instance — a duplicate launch doubles every topic
> (two publishers on /P1/pose) and muddies rates; `pgrep -f avatar_pro_hope_bridge` before launching.
> (b) vrpn topics carry SENSOR-SUFFIXED names here (`/vrpn_mocap/P1/pose8`, `/vrpn_mocap/Ball/pose5`,
> plus a huge `/vrpn_mocap/MCAvatar/pose#####` flood) — that is normal, the relay discovers suffixed
> topics. (c) THE BALL CHECK: with the ball VISIBLY on the table,
> `ros2 topic hz /vrpn_mocap/Ball/pose5` and `/ball/point` must both tick. `/poses` is published ONLY
> when ball data flows (relay `_emit_ball` → `_publish_poses`) and the planner takes the ball FROM
> `/poses` — no ball stream ⇒ planner NEVER emits a racket command (runner `ts=-1` forever), even
> though `/P1/pose` is fine. Silent `Ball/pose5` while the ball is visible = CMTracker not tracking
> the Ball rigid body (re-enable it / check markers) — G4.

**STEP 2 — Planner (HDU)**
```bash
ssh agi@192.168.120.249                    # password 1 (wifi jump; NOT the ProtonVPN 172.16.238.4)
source /opt/ros/jazzy/setup.bash
source ~/hope_ws/install/local_setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0                # ⚠ the HDU LOGIN env (aima load-env) sets this to 1 —
                                           # localhost-only traps the planner on the HDU even with
                                           # the right domain (field 2026-07-07, planner log said
                                           # "'localhost_only' is enabled")
echo $ROS_DOMAIN_ID $ROS_LOCALHOST_ONLY    # ⚠ MUST print "232 0" BEFORE starting the planner — wrong
                                           # domain OR localhost_only=1 = INVISIBLE to laptop+MDU and
                                           # the runner spams "[pp EXT-BASE] NO FRESH mocap base sample"
ros2 run hope_planner hope_planner_node --ros-args \
  --params-file ~/hope_ws/src/hope_planner/config/hope_planner.yaml \
  --params-file ~/hope_ws/src/hope_planner/config/hope_planner.hitter_pure.yaml \
  -p robot_pose_topic:=/P1/pose \
  -p marker_to_base_xyz:="[0.0, 0.0, 0.0]"     # G8: marker→base_link offset ([0,0,0] ok if cluster on the pelvis)
# ⚠ hope_planner.hitter_pure.yaml = FIXED plane (x_hit_follow_robot:=false, x_hit 0.20). MANDATORY
#   for the x-locked HITTER: follow-mode chases the robot, deletes the obs x-anchor, and the policy
#   drifts +x until it FALLS. The shared base profile deliberately retains follow-mode for 175/177;
#   load this overlay to select both HitterPure's fixed-plane semantics and venue x_hit/offset.
#   Startup log MUST read "x_hit FIXED".
```
**MANDATORY verify** in a 2nd HDU shell (also `export ROS_DOMAIN_ID=232` + `export ROS_LOCALHOST_ONLY=0`)
— do NOT arm the robot until both pass:
```bash
ros2 topic hz /a3/base_pose_flat          # continuous ~15-40 Hz (robot pose alone drives it) — silence
                                          #   = planner not on 232 OR /P1/pose not reaching the HDU
ros2 topic echo --once /a3/base_pose_flat # data[4] (z) MUST read ≈0.91 (mocap 0.15 + policy_z_offset
                                          #   0.76 → floor frame; <0.7 = the runner base_low-blocks ALL
                                          #   engage). x,y in metres. Verified 2026-07-07: z=0.9125 ✓
ros2 topic info /poses                    # Subscription count ≥1 = the planner is really subscribed
# then toss a real ball THROUGH the air:
ros2 topic hz /racket/command_flat        # bursts while the ball flies
ros2 topic echo /racket/command_flat --once   # data[1]=1.0 valid, data[3] px∈[0.0,0.35], data[9] tts→0
# cross-check the planner's own env if in doubt:
#   tr "\0" "\n" < /proc/$(pgrep -f hope_planner_node | head -1)/environ | grep ROS_DOMAIN_ID
```
> **Frame contract (2026-07-07 fixes)**: the planner INGESTS the mocap frame (z=0 at the TABLE
> SURFACE — its bounce plane is z=0) and PUBLISHES the policy/floor frame (`policy_z_offset: 0.76`
> in hope_planner.yaml shifts both flats + /racket/command; sim yaml keeps 0.0 — sim feeds are
> already floor-origin). The planner is also crash-guarded: a diverging solve (garbage/mm feed)
> logs a throttled warning and degrades to no-command instead of killing the node.

**STEP 3 — Runner in SHADOW: verify engage on REAL serves, ZERO motion (MDU)**
```bash
ssh -J agi@192.168.120.249 agi@10.42.10.12
source /agibot/software/v0/entry/env/env.sh          # sets ROS_DOMAIN_ID=232
cd /agibot/a3_deploy && export A3_TRANSPORT=iceoryx   # ⚠ EXACTLY iceoryx
taskset -c 4-7 ./run_a3_pingpong.sh --planner --dry-run --start shadow
```
Serve balls. Expect `[pp engage] forehand/backhand locked …` on good serves and `[pp gate] REJECT …`
(with values) on unreachable ones — the WHOLE pipeline minus motion. Tune serve style / robot
placement here until engage fires reliably. This is the safe gate before any motion.

**STEP 4 — Body-drive backend for real motion (⚠ robot HOISTED — see §9.2c)**
```bash
# MDU terminal HAL (robot SUPPORTED — stopping agibot_pm drops the controller holding it):
sudo systemctl stop agibot_pm
pgrep -a iox-roudi                                   # must stay alive (our iceoryx broker)
pgrep -a hal_ethercat && sudo pkill -TERM aimrt_main_hal
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0 && bash scripts/hal_ethercat/start_hal_ethercat.sh   # A3_P1D0; BLOCKS here
```

**STEP 5 — Runner in MOTION: real returns (MDU, 2nd terminal)**
```bash
cd /agibot/a3_deploy && export A3_TRANSPORT=iceoryx
taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand \
  --gain-scale 0.4 --leg-gain-scale 1.0 --hold-recover 1.2
# ⚠ --official-stand is MANDATORY for free-standing: without it 's' uses --stand-kp 60
#   (vs the official knee ~2000) → the KNEES BUCKLE and the robot cannot stand
#   (field-confirmed 2026-07-07; main.cpp:709 picks the gain set on this flag).
# ⚠ --leg-gain-scale 1.0 is MANDATORY on the ground with --gain-scale 0.4: gain-scale
#   alone also softens the POLICY leg kp to 0.4× → legs sink during the swing
#   (main.cpp:551 comment). 0.4 keeps only arms/waist gentle; legs stay full strength.
# ARM ritual (§9.3): 's' (stand — VERIFY it bears weight, knees firm) → keep still ~2 s
# (IMU yaw-align) → 'm' (MOTION). 'p' = PASSIVE abort. Raise --gain-scale with ']' once stable.
```
**Human serving** (§9.3): slow HIGH lobs (apex ≥1.3 m, flight ≥1.3 s), into the **forehand half first**,
bounce mid-table. Hard/flat serves are stood out as `too_late` by design. **One clean return per
placement**, then `p` → walk the robot back to the start spot → `s` → `m` for the next point (no
post-strike homing yet). Restore the normal robot afterwards: `sudo systemctl start agibot_pm`.

> ⚠ **A STATIC ball does nothing — BY DESIGN.** Holding/placing a ball in front of the robot will
> NEVER trigger a swing: the planner fits a FLIGHT trajectory and predicts its crossing of the hit
> plane; a ball with ~zero velocity never crosses it → `valid=0` → the runner stands
> (`PLANNER: planner_invalid`/`no_command`). Engage also needs ≥1.0 s time-to-strike and a standing
> robot (`base z ≥ 0.7`). The ONLY trigger is a real served lob per the rules above.

**If it's not working:**
| symptom | cause → fix |
|---|---|
| STEP 2 `/racket/command_flat` never valid | mocap not flowing to the HDU → recheck STEP 1 `/P1/pose`, and `ROS_DOMAIN_ID=232` on BOTH laptop + HDU |
| **runner spams `[pp EXT-BASE] NO FRESH mocap base sample`** (field 2026-07-07) | the PLANNER shell missed `export ROS_DOMAIN_ID=232` → planner on domain 0, invisible to the runner. Decisive check: `tr "\0" "\n" < /proc/$(pgrep -f hope_planner_node\|head -1)/environ \| grep ROS_DOMAIN` — no line = domain 0. Restart the planner after the export. Engage is BLOCKED (by design) until the base stream returns |
| planner on 232 but STILL invisible / base flat silent (field 2026-07-07 #2) | `ROS_LOCALHOST_ONLY=1` from the HDU login env (`aima em load-env`) — planner boot log prints `'localhost_only' is enabled`. Fix: `export ROS_LOCALHOST_ONLY=0` before starting (STEP 2). The correct boot log says `'localhost_only' is disabled` |
| **runner `ts=-1` forever, robot stands through serves** (field 2026-07-07) | no racket command ever received → the BALL isn't streaming: `/vrpn_mocap/Ball/pose*` silent → `/ball/point`+`/poses` never publish → planner has no ball. Verify with the ball VISIBLE on the table (STEP 1 ball check); fix = CMTracker Ball rigid body tracking (G4) |
| **planner crashes `FloatingPointError: outgoing velocity solve diverged`** (field 2026-07-07) | garbage-scale measurements — the venue mocap streams MILLIMETRES. Fix = `position_scale: 0.001` in avatar_pro_vrpn.yaml (relay mm→m). The node is now crash-guarded (degrades to no-command), but the feed must still be fixed for real solutions |
| every engage `base_low`-rejected though the robot stands | base flat z < 0.7 → the mocap-frame z (surface origin, robot ≈0.15) reached the runner unshifted. Fix = `policy_z_offset: 0.76` in hope_planner.yaml (mocap/table frame → policy/floor frame); verify `data[4]≈0.91` per STEP 2 |
| STEP 3 no `[pp engage]`, only `no_command` | HDU planner not publishing / domain mismatch → `ros2 topic echo /racket/command_flat` on the HDU; confirm 232 |
| STEP 3 constant `[pp gate] REJECT z_w≈0.00` | ball dies before the plane → higher/slower serves (§9.3); or robot placement vs `x_hit` |
| STEP 3 engage only forehand, never backhand | runner splits fh/bh by base-rel target-y **sign** (y<0=fh); backhand needs the ball to arrive y≥0 |
| STEP 5 robot sags when you stop agibot_pm | it wasn't hoisted — that stop removes the holding controller (§9.2c safety) |
| **'s' but KNEES BUCKLE, cannot stand** (field 2026-07-07) | `--official-stand` missing → PD_STAND uses `--stand-kp 60` instead of the official knee ~2000. Re-run with `--official-stand`. If it buckled, fall guard likely tripped → PASSIVE (looks totally dead): support robot upright, then `s` again |
| stands, but legs SINK during the swing / tiny swing | `--gain-scale 0.4` also softened the policy LEG kp → add `--leg-gain-scale 1.0` (arms stay gentle, legs full); raise `--gain-scale` with `]` for full swings |
| ball placed/held in front → nothing | **by design** — static ball never crosses the hit plane, planner never emits valid. Serve a real lob (apex ≥1.3 m, flight ≥1.3 s, bounce mid-table) |
| boot log `seg_len={95,105}` or no 177 line | stale binary → re-ship the rockchip dist (§9.1) |

### 9.8 ★★ BARE COPY-PASTE VERSION — 4 terminals, in order, no checks

No verification steps. Four terminals, start them in this order, leave each one running.
Every password is `1`. Robot: HOISTED until Terminal D says otherwise.

**Terminal A — LAPTOP (mocap bridge). Leave running.**
```bash
distrobox enter hope
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/hope_ws
source install/local_setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
pkill -INT -f "avatar_pro_hope_bridge.launch"; sleep 5
ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py server:=192.168.10.100
```

**Terminal B — HDU (planner). Leave running.**
```bash
ssh agi@192.168.120.249
source /opt/ros/jazzy/setup.bash
source ~/hope_ws/install/local_setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
pkill -f "hope_planner_nod[e]"; sleep 2
ros2 run hope_planner hope_planner_node --ros-args \
  --params-file ~/hope_ws/src/hope_planner/config/hope_planner.yaml \
  --params-file ~/hope_ws/src/hope_planner/config/hope_planner.hitter_pure.yaml \
  -p robot_pose_topic:=/P1/pose \
  -p marker_to_base_xyz:="[0.0, 0.0, 0.0]"
# ⚠ hope_planner.hitter_pure.yaml = FIXED plane (x_hit_follow_robot:=false). MANDATORY for the
#   x-locked HITTER: follow-mode chases the robot, deletes the obs x-anchor, and causes drift-fall.
#   Startup log MUST read "x_hit FIXED", not "FOLLOW-MODE".
```

**Terminal C — MDU (HAL). ⚠ robot MUST be hoisted/supported BEFORE the first line. Leave running.**
```bash
ssh -J agi@192.168.120.249 agi@10.42.10.12
sudo systemctl stop agibot_pm
sudo pkill -TERM aimrt_main_hal; sleep 2
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_ethercat/start_hal_ethercat.sh
```

**Terminal D — MDU (runner).**
```bash
ssh -J agi@192.168.120.249 agi@10.42.10.12
source /agibot/software/v0/entry/env/env.sh
cd /agibot/a3_deploy
export A3_TRANSPORT=iceoryx
taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand \
  --gain-scale 0.4 --leg-gain-scale 1.0 --hold-recover 1.2
```

**Then, in Terminal D (keyboard):**
1. Put the robot on the floor at its spot: **0.8 m behind the table edge, forehand half, facing the table, square**. Hand on `p` (= PASSIVE abort) from here on.
2. Press `s` — robot stands.
3. Keep it untouched **2 seconds**.
4. Press `m` — MOTION.
5. **Serve**: slow HIGH lob (apex ≥ 1.3 m), bounce mid-table, into the robot's right half. Robot swings.
6. After each return: press `p` → carry the robot back to its spot → `s` → wait 2 s → `m` → next serve.
7. Done for the day: `q` in Terminal D, Ctrl-C in Terminal C, then on the MDU: `sudo systemctl start agibot_pm`. Ctrl-C Terminals A/B.

## 10. Quick reference

| task | box | directory | command |
| --- | --- | --- | --- |
| train / resume | grasping | `hope_training/whole_body_tracking` | `hope_isaac_py scripts/train.py task=HOPEPingPongHitterPure ...` |
| export onnx | grasping | same | `bash scripts/export_onnx_hitter_pure.sh <run> [ckpt]` (110-D baseline) |
| Gate 1 (110-D) | grasping | same | `hope_isaac_py scripts/eval_deterministic.py task=HOPEPingPongHitterPure ... checkpoint=<ckpt>` (MuJoCo 110 branch pending) |
| sync + parity | hope | `agi/a3_deploy_example` | `bash scripts/sync_pingpong_model.sh <onnx> <name>` |
| build x86 / rockchip | hope / HOST | same | `bash scripts/build_a3_deploy_pkg.sh --arch x86_64\|rockchip --runtime-cfg ...pingpong.yaml` |
| AGI sim policy-only | hope | same | `bash scripts/pp_freebase_watch.sh --single-swing [--oracle-pelvis]` |
| AGI sim rally **Gate 3 (x-locked 110-D)** | hope | same | `bash scripts/pp_gate3_rally.sh` — FIXED plane (`-p x_hit_follow_robot:=false`), continuous rally, per-serve verdicts. **Use this for any x-locked / 110-D hitter_pure ONNX.** |
| AGI sim + planner loop (**follow-mode, 175/177 only**) | hope | same | `bash scripts/pp_planner_closedloop.sh` — adaptive plane required by the current 175/177 rally. ⚠ Do NOT run an x-locked ONNX through this: follow-mode deletes the x anchor → drift-fall (the planner logs a `FOLLOW-MODE` compatibility warning). Viewer variant in §7. |
| landing analysis (§7.5) | host | `hope_ws/src/hope_planner` | `python3 -m hope_planner.landing_mc` — on-table % from a checkpoint's exact errors (champion ≈ 98%; training thresholds only 67%) |
| mocap + planner sanity (arena) | **HDU** (dom 232) | `~/hope_ws` | `export ROS_DOMAIN_ID=232` + `avatar_pro_hope_bridge.launch.py server:=192.168.10.100` + `hope_planner_node` (§9.2 steps 1-2; libVRPN + mocap route pending) |
| ship to robot | host→MDU | `agi/a3_deploy_example` | `rsync ... dist/a3_deploy_rockchip/ agi@10.42.10.12:/agibot/a3_deploy/` |
| run on robot (scripted) | MDU | `/agibot/a3_deploy` | `taskset -c 4-7 ./run_a3_pingpong.sh --start passive --legs-passive --gain-scale 0.4 --single-swing` |
| run on robot (planner) | MDU | `/agibot/a3_deploy` | `taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --official-stand --gain-scale 0.4 --leg-gain-scale 1.0 --hold-recover 1.2` (§9.7 STEP 5; `--official-stand`+`--leg-gain-scale 1.0` mandatory free-standing — knees buckle without them, field 2026-07-07) |
| transition matrix (Gate 2.5) | hope | `agi/a3_deploy_example` | `bash scripts/pp_gate25.sh [--oracle]` (§6: shipped `model_17400_hitter177` = 10/10 oracle / 3/4 perfect_tracking; HitterPure candidates use the separate 110-D gate and must not replace the default without artifact provenance + hardware G2) |
| hardware planner+control | HDU planner + MDU runner | §9.2 | (HDU dom 232) mocap → planner+flats → (MDU) `--planner --dry-run` (G2 ✅) → shadow `h` → ARM ritual + `m` (§9.2-9.3) |
| hand-fed flats test (no mocap) | HDU pub + MDU runner | §9.2b | (HDU dom 232) `ros2 topic pub` the 2 flats → (MDU) `--planner --dry-run`+`h` iterate → real `s`→`m` strike; wiring verified 2026-07-07 |
