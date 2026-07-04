# Ping-Pong End-to-End Runbook: fresh machine → train → gates → MuJoCo → robot

**Audience:** you have a new robot (or a new machine), a full clone of this repo, and
you want the WHOLE process — which distrobox, which directory, which command, in order.
Written 2026-07-03; C++-only control path since 2026-07-04 (§0 retirement note).
Companion docs: [run_sim2real_bridge.md](run_sim2real_bridge.md) (the RETIRED python
ROS control chain — historical), `agi/a3_deploy_example/PINGPONG_NEW_CHECKPOINT_TUTORIAL.md`
(AGI-side checkpoint sync in depth), [run_training.md](run_training.md).

## 0. The big picture — ONE control path (C++); ROS only feeds it inputs

```
                       TRAINING (grasping box, Isaac)
                                   │ .pt
                                   ▼  export (grasping box)
                             policy.onnx (175-D deploy_parity)
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

        input side (plain ROS 2, laptop or MDU — carries NO control code):
          fake_ball_publisher (sim) / mocap vrpn relay (arena) → hope_planner
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
- The ONNX auto-detects the 175/180-D obs contract and the clip layout comes from
  ONNX metadata.

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
| ✎ `server:=` | `192.168.10.100` (placeholder!) | the MCServer/CMTracker PC IP on the arena LAN — there is NO working default |
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
`--invalid-grace 0.25`, `--swing-rest` (0.5 default). The reachability gate box is
compile-time in `pp_policy.hpp` (`gate_x_lo/hi` etc.) — matches the model_11400
boxes; rebuild rockchip if the model generation changes.

**Training / eval**

- [ ] A1 latency/jitter flag values (`racket.target_delay_steps` etc.) from timestamped
  venue recordings — currently default-off.
- [ ] Per-model reachability-gate box (compile-time `gate_*` in `pp_policy.hpp`, sheet
  item (5)) when the training boxes change — rebuild BOTH dists.

**Robot / deploy**

- [ ] Rockchip sysroot tarball (`scripts/export_rockchip_sysroot.sh` output) present in
  `thirdparty/` for the cross-build.

## 3. Train (grasping box)

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh        # defines hope_isaac_py
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true
# resume from a checkpoint:
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  checkpoint_path=logs/rsl_rl/agibot_a3_hope_deploy_parity/<run>/model_XXXX.pt
```

~2.1 s/iter at 4096 envs on the 5090; runs land in
`logs/rsl_rl/agibot_a3_hope_deploy_parity/<date_time>/`. Health signals (tensorboard):
`strike_composite_success_exact` climbing with fh/bh balanced, `pre/post_strike_fall_rate`
< a few %, and WATCH `Policy/mean_noise_std` — late-run std inflation makes reward curves
lie; judge checkpoints by the gates below, not W&B.

## 4. Export ONNX (grasping box)

```bash
distrobox enter grasping
bash ~/workspace/HOPE/hope_training/whole_body_tracking/scripts/export_onnx_deploy_parity.sh \
  logs/rsl_rl/agibot_a3_hope_deploy_parity/<run>  [model_XXXX.pt]   # omit ckpt = newest
# -> <run>/exported/policy.onnx  (auto-kills the hung Isaac after the file appears)
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

```bash
cd ~/workspace/HOPE/hope_training/whole_body_tracking
../.venv-motion/bin/python scripts/mujoco_eval_onnx.py \
  --onnx logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-03_13-32-07/exported/policy.onnx \
  --motion-files artifacts/hope_forehand_hopex/motion.npz artifacts/hope_backhand_hopex/motion.npz \
  --noise-scales 0.0 --pd-mode implicit --deploy-faithful --steps 1500
# cross-checkpoint comparison on fixed protocols (appends to logs/scoreboard/scoreboard.csv):
../.venv-motion/bin/python scripts/scoreboard_eval.py --onnx <...> --label <name> \
  --motion-files artifacts/hope_forehand_hopex/motion.npz artifacts/hope_backhand_hopex/motion.npz
```

Reference: `model_7500_hopex` scored composite 1.0 both clips, 7/7 swings, 0 falls,
pos err 4.1 cm, vel err 0.20 m/s.

## 6. Sync to the AGI side + Gate 2 (policy-only in the official sim)

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example
# stage asset + rewrite runtime cfg + C++<->Python parity (expect PASS ~1e-6):
PYBIN=~/workspace/HOPE/hope_training/.venv-motion/bin/python \
  bash scripts/sync_pingpong_model.sh \
   ~/workspace/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-03_13-32-07/exported/policy.onnx \
    model_11400_hopex
# if the training target boxes moved: update racket_{pos,vel}_w_clip in
# src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_policy.hpp (compile-time!) then:
bash scripts/build_a3_deploy_pkg.sh --arch x86_64 \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml

# Gate 2: free-base swings in the OFFICIAL AGI MuJoCo sim (C++ runner, scripted targets)
bash scripts/pp_freebase_watch.sh --single-swing                  # perfect_tracking
bash scripts/pp_freebase_watch.sh --single-swing --oracle-pelvis  # + run scripts/run_oracle.sh in a 2nd terminal
```

PASS per swing direction: ≥5 clean cycles, 0 falls, no guard trips, sync_miss 0.
Boot log must show `[pp] clip layout from ONNX metadata: seg_len={139,132} ...` —
the legacy `{95,105}` means a stale binary or wrong model. Test BOTH loc modes: this
generation walks to its target; `perfect_tracking` runs that loop open.

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
> **Expected with model_11400 + what the m0-hold FAIL means** (differential run
> 2026-07-04): `P2 m0 hold 20s` FAILS at ~5 s in BOTH loc modes — but the SAME onnx
> holds the SAME 20 s windup hold INDEFINITELY in the training-side harness
> (`mujoco_eval_onnx.py --deploy-faithful --df-hold-steps 1000`: sinks to ~0.96,
> recovers, oscillates 0.98–1.01 for the full 20 s). The C++ obs at hold are verified
> healthy (upright gravity, nominal joints, sane FK-relative target, tts clamped) —
> so this is NOT a pp_policy obs bug and NOT a pure model failure: the policy's hold
> posture is MARGINAL (it always dips to ~0.95), training-side physics lets it
> recover, the AGI sim's actuation/joint model does not (same class as the known
> explicit-PD lean artifact — but AGI calls their sim predictive of hardware, so
> treat the long policy hold as UNSAFE on the robot). Mitigations already shipped:
> `--planner` mode never parks on the policy hold (static stand except the bounded
> recovery window, §9.6); in scripted mode drop to `s` when idling. Note the
> training-side run ALSO fell during the swing AFTER its 20 s hold (drift
> accumulates) — keep holds short everywhere. NEXT-RETRAIN item: train the hold
> (hold(0,100)→(0,500+) steps, stillness reward, friction/armature DR) — repro =
> Gate 2.5 P2 and `--df-hold-steps 1000`.

## 7. Gate 3 — planner + control closed-loop in MuJoCo (C++ `--planner`, sim)

The closed loop runs the DEPLOY binary itself: fake_ball → REAL hope_planner (sim
profile, publishes the flat topics) → C++ runner in `--planner` mode
(`external_base` localization from `/a3/base_pose_flat`) → AGI MuJoCo sim (iceoryx
body-drive). This is the exact hardware wiring — the only sim-specific pieces are
the fake ball and the ground-truth base pose source. Input contract + engage
semantics: §9.6. Closed-loop VERIFIED 2026-07-04 (§9.6 status).

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
§9.3 wants close-bounce high-pop serves); serve `[3.2, ∓0.24, 0.5, -2.0, ±0.10, 3.6]`
(probed: crosses the adaptive plane at z≈0.72-0.85, tts≈1.25); robot reset to x=0.33
so the adaptive x_hit (robot_x+0.67) sits at the hop apex; serve cadence `pause_s 4.0`
(the 4-6 s demo cadence). NOTE: the sim overlay's x_hit clamp is deliberately wide
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
  -p serve_forehand:="[3.2, -0.24, 0.5, -2.0, 0.10, 3.6]" \
  -p serve_backhand:="[3.2, 0.24, 0.5, -2.0, -0.10, 3.6]" \
  -p drag_k:=0.05 -p pause_s:=4.0
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

## 8. Deploy to the robot

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
taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --gain-scale 0.4
# 0/1/f/b keys are IGNORED (the planner drives engage + side); p/s/h/m still work — 'p'
# (PASSIVE) is the operator abort. Full bring-up + input contract: §9.6.
```

## 9. Planner + control in REAL LIFE (hardware) — the demo runbook

Scenario: HUMAN serves, robot returns. The robot does not serve. Baseline policy:
`model_11400_hopex` (2026-07-04 verification: C++⇔Py ONNX parity 1e-7; training-side
deploy-faithful MuJoCo gate 7/7 completions fh+bh ZERO falls; AGI-sim free-base
continuous swings; Gate 3 closed-loop planner-driven rally in sim, 2 engages, zero
falls).

### 9.0 Gap checklist — fill EVERY row before arming on hardware

Legend: ✅ = filled in the repo (2026-07-04), 🏟 = venue-day measurement/procedure,
⛔ = BLOCKING engineering gap (not a venue measurement — needs real work).
**For every 🏟 row, the EXACT file : key : measurement procedure is in the §2
VENUE FILL-IN SHEET** — G3/G4/G5 → sheet items (3)+(4), G6/G7 → §9.3 placement +
(1)'s `x_hit` fallback, G8 → (1)+(2) (the `--planner` flat feed needs
`marker_to_base_xyz` too), G9 → (1).

| # | item | where | status |
|---|------|-------|--------|
| G1 | **Planner inputs DDS-visible to the MDU** *(2026-07-04 REFRAMED — was: full hope_ws chain on the MDU)*. The `--planner` C++ runner (§9.6) eliminates wbc_runner, the hw bridge, and the python-onnxruntime wheel from the MDU entirely; the control loop is the native aarch64 binary. What remains: the two flat input topics must reach the MDU's DDS, and the laptop cannot see the MDU's DDS → run **mocap relay + hope_planner ON the MDU**. hope_planner is pure python (rclpy/numpy/core msgs — the AGI robot env already ships a ros2+rclpy runtime; the flats are `std_msgs`, no custom typesupport needed on the receive side). Remaining build items: **numpy on the MDU** (pip/apt, aarch64) and **vrpn_mocap** (one small C++ pkg: 3 .cpp + libVRPN — cross-compile with the rockchip sysroot or build on the MDU), plus the IP route MDU → MCServer LAN (VRPN is plain TCP/UDP — works if routable). | MDU | 🟡 (was ⛔) |
| G2 | **aarch64 AimRT ros2_plugin loads on the MDU** *(2026-07-04 SCOPED DOWN — was: ros2-enabled `/body_drive`)*. `/body_drive` **stays iceoryx** in `--planner` mode; only the two low-rate flat topics ride the ros2 backend (`a3_aimrt_config.pingpong_ros2body.yaml`, same dual-plugin pattern as AGI's own teleop cfg). The aarch64 `libaimrt_ros2_plugin.so` ships in the rockchip dist; it loads clean on x86-in-box but has NEVER been exercised on the MDU (the x86 HOST hits an rclcpp ABI break — box works, host doesn't; the MDU is a third environment). Verify FIRST THING: `./run_a3_pingpong.sh --planner --dry-run` must reach "backend started" with both `subscriber enabled` lines and no undefined-symbol abort; then `ros2 topic hz /racket/command_flat` (after `setup_ros2_msgs.bash`) sees the planner. If the plugin won't load → fallback §9.5. | MDU | 🟡 (was ⛔) |
| G3 | MCServer (CMTracker PC) IP on the arena LAN → `server:=` launch arg. No working default. | venue | 🏟 |
| G4 | CMTracker rigid bodies named EXACTLY `Ball`, `P1` (robot), `P2`, `PPT` (2026-07-03 convention, avatar_pro_vrpn.yaml). Verify: `ros2 topic list \| grep vrpn_mocap`. | venue | 🏟 |
| G5 | Mocap world calibration: meters, Z-up, origin at P1 near-side LEFT table corner ON the surface, +x toward the opponent. VERIFY with a marker at the net center → must read ≈ (1.37, −0.7625, 0). | venue | 🏟 |
| G6 | Robot placement: **0.8 m behind the table edge on the forehand half** (physical placement, see §9.3 why). No yaml key anymore — the C++ runner localizes from the live mocap base pose; just update (1)'s `x_hit` static fallback to `robot_x + 0.67` (−0.8 → **−0.13**). | venue | 🏟 |
| G7 | **Placement convention**: the robot FACES +x (the opponent), square. The C++ runner yaw-aligns the IMU at every MOTION entry (`m`) — keep it still ~2 s there. There is no yaml yaw override anymore: place it square. | venue | 🏟 |
| G8 | `marker_to_base_xyz` (hope_planner.yaml, sheet (1)) + `mocap_to_base_link.p1_xyz` (hope_world_frame.yaml, sheet (2)): P1 marker-cluster → base_link offset. Measure per the procedure in hope_world_frame.yaml. `[0,0,0]` is usable if the cluster sits on the pelvis. | venue | 🏟 |
| G9 | Ball physics: **VENUE FIT IN THE YAML** (2026-07-03 recordings via main: drag_k 0.1261, restitution 0.64/0.9215/0.654; consistency-guard tests police drift vs node defaults). Spot-check on venue day; full re-fit only if venue/ball changed. The salvaged Jun-23 fit (0.8781) stays REJECTED (8× physical). | venue | ✅ (spot-check 🏟) |
| G10 | Planner adaptive x_hit: `robot_pose_topic:=/P1/pose`, `x_hit_offset 0.67`, clamp `[0.0, 0.35]` — **already the yaml defaults** (2026-07-04). The clamp is the TABLE-COLLISION protection: it stops the demanded plane (and the lunge endpoint ≈ plane − 0.67) short of the table edge. | — | ✅ |
| G11 | Runner engage-safety set (active-swing lock + frozen target, bounded post-swing hold → sticky static stand, engage-tts clock seed + clamp, invalid-flutter grace, base_low guard, **MOTION-entry yaw align**) — in the C++ runner (§9.6). | — | ✅ |
| G12 | Baseline ONNX on the robot: `model_11400_hopex.onnx` staged in `assets/a3_runtime/models/` + runtime cfg points at it; **rockchip dist rebuilt AFTER the 2026-07-04 C++ changes** (dir-flip latch, PD_STAND blend, yaw guard, fall guard — §8 build). | laptop→MDU | ✅ code / ⛔ rebuild+ship pending |

**G1/G2 were the blockers; the 2026-07-04 `--planner` port (§9.6) shrank both** from
"build a full aarch64 ROS chain + re-plumb /body_drive" to "get two topics onto the MDU
+ confirm one plugin loads". The demo decision tree:

1. `--planner --dry-run` on the MDU loads the ros2 plugin (G2 ✓) AND relay+planner run
   on the MDU (G1 ✓) → **autonomous demo via §9.6** (official path).
2. G2 ✓ but no time for vrpn_mocap/numpy on the MDU → no live ball, but §9.6 can still
   be driven by any hand-crafted `/racket/command_flat` publisher on the MDU (scripted
   serves with real engage logic).
3. ros2 plugin fails on the MDU → **fallback demo §9.5** (scripted keys) — runs TODAY
   on the proven scripted-mode binary.

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

### 9.2 Staged bring-up (each stage safe by construction)

All commands below run **on the MDU** once G1 is closed (until then rehearse on the
laptop against the AGI sim, §7 — the identical wiring). Every MDU shell first:

```bash
ssh -J agi@$HDU agi@10.42.10.12
source /agibot/software/v0/entry/env/env.sh
# + source the MDU-side hope_ws overlay (planner) where a ros2 command needs it
```

1. **Mocap sanity** — `cd <hope_ws on MDU>`:
   ```bash
   ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py server:=<MCSERVER_IP>
   ros2 topic hz /poses            # ball ≥240 Hz
   ros2 topic echo /P1/pose --once # robot pose alive, sane table-frame numbers
   ```
   Landmark check: marker at net center reads ≈ (1.37, −0.7625, 0).
2. **Planner sanity** (no robot motion): start the planner, toss real balls:
   ```bash
   ros2 run hope_planner hope_planner_node --ros-args --params-file hope_planner.yaml \
     -p robot_pose_topic:=/P1/pose -p marker_to_base_xyz:="[<G8 values>]"
   ros2 topic hz /racket/command_flat /a3/base_pose_flat   # both alive
   ros2 topic echo /racket/command_flat  # data[1]=1.0 (valid) on good serves,
                                         # data[3] (px) inside [0.0,0.35],
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
   taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --gain-scale 0.4
   # passive → hoist checks → 's' (stand). Then 'h' (SHADOW): the FULL engage+swing
   # pipeline runs on real serves, publishing nothing. Verify [pp engage] fires on
   # good serves, the PLANNER status cycles sanely, [pp gate] REJECT only on
   # genuinely bad serves, projected gravity ≈ [0,0,−1] on the [obs] line.
   ```
5. **Hardware** — the ARM ritual below, then `m` (MOTION); gain-scale 0.4 first,
   raise with `]` once stable.

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
/a3/base_pose_flat    [schema=1, valid(0/1), x,y,z, qw,qx,qy,qz]      (≥9 doubles)
                       ← the robot base in the SAME frame as the racket target
                       (arena: /P1/pose + marker_to_base_xyz; sim: /sim/a3/pelvis_pose)
```

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
`m`). The G1 items (numpy + vrpn_mocap on the MDU, MCServer IP-routable) are in
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

## 10. Quick reference

| task | box | directory | command |
| --- | --- | --- | --- |
| train / resume | grasping | `hope_training/whole_body_tracking` | `hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity ...` |
| export onnx | grasping | same | `bash scripts/export_onnx_deploy_parity.sh <run> [ckpt]` |
| MuJoCo gate | host | same | `../.venv-motion/bin/python scripts/mujoco_eval_onnx.py ... --deploy-faithful` |
| sync + parity | hope | `agi/a3_deploy_example` | `bash scripts/sync_pingpong_model.sh <onnx> <name>` |
| build x86 / rockchip | hope / HOST | same | `bash scripts/build_a3_deploy_pkg.sh --arch x86_64\|rockchip --runtime-cfg ...pingpong.yaml` |
| AGI sim policy-only | hope | same | `bash scripts/pp_freebase_watch.sh --single-swing [--oracle-pelvis]` |
| AGI sim + planner loop (Gate 3) | hope | same | `bash scripts/pp_planner_closedloop.sh` — one command; viewer variant also in §7 |
| mocap + planner sanity (arena) | hope/MDU | `hope_ws` | `avatar_pro_hope_bridge.launch.py server:=<IP>` + `hope_planner_node` + `ros2 topic hz /racket/command_flat` (§9.2 steps 1-2) |
| ship to robot | host→MDU | `agi/a3_deploy_example` | `rsync ... dist/a3_deploy_rockchip/ agi@10.42.10.12:/agibot/a3_deploy/` |
| run on robot (scripted) | MDU | `/agibot/a3_deploy` | `taskset -c 4-7 ./run_a3_pingpong.sh --start passive --legs-passive --gain-scale 0.4 --single-swing` |
| run on robot (planner) | MDU | `/agibot/a3_deploy` | `taskset -c 4-7 ./run_a3_pingpong.sh --planner --start passive --gain-scale 0.4` (§9.6; G2 gate first) |
| transition matrix (Gate 2.5) | hope | `agi/a3_deploy_example` | `bash scripts/pp_gate25.sh [--oracle]` (§6: per-phase PASS/FAIL; m0-hold FAIL expected on model_11400) |
| hardware planner+control | MDU | §9.2 | mocap → planner+flats → `--planner --dry-run` (G2) → shadow `h` → ARM ritual + `m` (§9.2-9.3) |
