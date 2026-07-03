# Ping-Pong End-to-End Runbook: fresh machine → train → gates → MuJoCo → robot

**Audience:** you have a new robot (or a new machine), a full clone of this repo, and
you want the WHOLE process — which distrobox, which directory, which command, in order.
Written 2026-07-03. Companion docs: [run_sim2real_bridge.md](run_sim2real_bridge.md)
(the ROS chain in depth), `agi/a3_deploy_example/PINGPONG_NEW_CHECKPOINT_TUTORIAL.md`
(AGI-side checkpoint sync in depth), [run_training.md](run_training.md).

## 0. The big picture — two control paths

```
                       TRAINING (grasping box, Isaac)
                                   │ .pt
                                   ▼  export (grasping box)
                             policy.onnx (175-D deploy_parity)
                                   │
              ┌────────────────────┴────────────────────┐
              ▼ PATH A (hardware-proven)                ▼ PATH B (autonomous, planner-driven)
   C++ deploy runner (a3_deploy_onnx_ref_pingpong)   ROS chain (hope_ws):
   scripted targets, keys f/b/1/0, iceoryx           mocap → hope_planner → wbc_runner
   ← what has hit the real robot so far              → agibot_hardware_bridge → /body_drive
              │                                          │
              ▼                                          ▼
   AGI MuJoCo sim  ──same binary──▶  robot MDU       AGI MuJoCo sim (ros2body cfg) / robot
```

- **Path A** is the deploy vehicle: the exact binary that ships to the robot, driven by
  built-in test targets. Use it for policy validation in sim and first hardware swings.
- **Path B** is the competition loop: real ball → planner → policy. It runs against the
  AGI MuJoCo sim today (§6) and on hardware once the robot-side ROS bridge is up (§8).
- The same ONNX feeds both; both auto-detect the 175/180-D obs contract and read the
  clip layout from ONNX metadata.

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
colcon build --packages-up-to hope_bringup hope_planner hope_wbc_runner agibot_hardware_bridge
source install/local_setup.bash
# sanity: pure-python tests need no ROS
cd src/hope_wbc_runner && python3 -m pytest test/ -q   # 32 passed
cd ../hope_planner    && python3 -m pytest test/ -q   # 41 passed
```

**Build the deploy package** (hope box; also produces the joint_msgs python overlay):

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
(planner diagnostics, wbc_runner `idle_reason`, gate warnings).

**Blocking the planner (RIGHT NOW: values are zeroed in the config!)**

- [ ] `hope_ws/src/hope_planner/config/hope_planner.yaml`: `drag_k`, `restitution_h`,
  `restitution_v`, `restitution_racket` are **0 = placeholder**. The planner CANNOT
  work with zeros (0 restitution = every bounce prediction dies). Fit from ≥15 mocap
  ball recordings: `ros2 run hope_planner hope_calibrate traj1.csv ... traj15.csv`
  and paste the printed values. Interim plausible values until the fit lands:
  `drag_k: 0.15`, `restitution_h: 0.85`, `restitution_v: 0.85`, `restitution_racket: 0.88`.
  (The old `drag_k 0.8781` fit was ~8× physical — do NOT restore it; it makes the
  planner demand 4.6–10 m/s racket speeds and the runner gate stands on every ball.)
- [ ] `x_hit` must track robot placement: `x_hit ≈ robot_start_x + 0.67` (current
  policy strikes 0.56–0.78 m in front of base). Default 0.17 assumes robot at −0.5.

**Mocap / arena (before Path B on real data)**

- [ ] MCServer IP → launch arg `server:=` (no working default).
- [ ] Rigid bodies in AvatarPro/CMTracker named EXACTLY `Ball`, `P1`, `P2` (standardized
  2026-07-03; these are now the config defaults).
- [ ] Mocap world calibration: meters, Z-up, origin = P1 near-side LEFT table corner ON
  the surface, +x toward the opponent. Verify: net-center marker ≈ (1.37, −0.7625, 0).
- [ ] `wbc_runner.yaml`: `robot_start_xy_table` (tape-measure the boot base XY),
  `robot_start_yaw_table` (place the robot facing +x and leave 0 — position-only mocap
  can't measure yaw), `marker_to_base_xyz` (P1 marker-cluster → base_link offset; also
  mirror into `hope_bringup/config/hope_world_frame.yaml`).
- [ ] Boot ritual: robot stands STILL ~2 s after launch (origin capture 60 samples +
  IMU yaw-align 100 samples).

**Training / eval**

- [ ] A1 latency/jitter flag values (`racket.target_delay_steps` etc.) from timestamped
  venue recordings — currently default-off.
- [ ] Per-model `target_gate_*` in `wbc_runner.yaml` when the training boxes change.

**Robot / deploy**

- [ ] Rockchip sysroot tarball (`scripts/export_rockchip_sysroot.sh` output) present in
  `thirdparty/` for the cross-build.
- [ ] For Path B on hardware: an AimRT backend config with ros2 enabled for
  `/body_drive/(.*)` on the robot (pattern: `a3_aimrt_config.ros2.yaml`) — the stock
  hardware config is iceoryx-only, the ROS bridge hears silence without it.

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
  --onnx logs/rsl_rl/.../exported/policy.onnx \
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
  bash scripts/sync_pingpong_model.sh <run>/exported/policy.onnx model_XXXX_<tag>
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

## 7. Gate 3 — planner + control closed-loop in MuJoCo (Path B, sim)

Recipe status: wired 2026-07-03, dry-run chain smoke-verified; first full sim run
pending. Nothing in the official simulator is modified — a NEW config file
(`a3_pingpong_ros2body_cfg.yaml`, tracked in git) routes `/body_drive/(.*)` over
**both** iceoryx and ros2; the original config and binary are untouched, and the C++
runner still works against it.

Terminal A — the sim with the ros2-bridged config:

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example
A3_SIM_CFG=a3_pingpong_ros2body_cfg.yaml ./scripts/run_sim.sh
./scripts/reset_sim.sh    # settle the robot
```

(The tracked source of the overlay is
`agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/a3_pingpong_ros2body_cfg.yaml`;
run_sim.sh reads `build/install/bin/cfg/` — a copy is staged there, and `A3_SIM_CFG`
also accepts a full path to the src file. On a fresh sim install, re-copy it.)

Terminal B — the full ROS chain (planner → runner → bridge → sim):

```bash
distrobox enter hope; source /opt/ros/jazzy/setup.bash
source ~/workspace/HOPE/hope_ws/install/local_setup.bash
source ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64/setup_ros2_msgs.bash  # joint_msgs
ros2 launch hope_bringup hope_pingpong_sim2real.launch.py \
  start_mocap:=false start_hw_bridge:=true \
  onnx_path:=$HOME/workspace/HOPE/agi/a3_deploy_example/assets/a3_runtime/models/model_7500_hopex.onnx \
  mode:=hardware state_source:=ros
# the runner needs the sim ground-truth pelvis pose instead of arena mocap:
#   base_pose_topic:=/sim/a3/pelvis_pose  base_pose_frame:=policy
# (set in wbc_runner.yaml or pass through a param override)
```

Terminal C — synthetic serves + arm the runner:

```bash
# same sourcing as B
ros2 run hope_bringup fake_ball_publisher            # table-frame serves, planner-matched physics
ros2 topic pub /hope/hardware_enable std_msgs/msg/Bool "data: true" -1   # ARM (publishes to sim)
# watch:  ros2 topic echo /wbc_runner/diagnostics    (swing_type, idle_reason, latched)
# estop:  ros2 topic pub /hope/estop std_msgs/msg/Bool "data: true" -1
```

Key facts making this work: the sim already publishes ground truth over ROS2
(`/sim/a3/pelvis_pose`, policy frame — `base_pose_frame:=policy` uses it verbatim);
`fake_ball_publisher`'s drag/restitution params MUST match `hope_planner.yaml`; all
frames stay in the arena convention, so this rehearses the exact real-life config.
"mode:=hardware" publishes only while `/hope/hardware_enable` is true — in this setup
it drives the SIM, not a robot.

## 8. Deploy to the robot

Build the rockchip package — **on the HOST (Docker), not in a distrobox**:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
bash scripts/build_a3_deploy_pkg.sh --arch rockchip \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
# rebuild rockchip after ANY pp_*.hpp / model change — dist is never auto-rebuilt
```

Ship + run (Path A — scripted-target bring-up, the hardware-proven flow):

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

## 9. Planner + control in REAL LIFE (Path B, hardware)

Staged bring-up — each stage is safe by construction (`dry_run`/`shadow` can never
publish; `hardware` needs enable AND not-estop):

1. **Mocap sanity** (laptop or MDU, hope box): launch the mocap bridge and verify data:
   ```bash
   ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py server:=<MCSERVER_IP>
   ros2 topic hz /poses ; ros2 topic echo /P1/pose --once   # ball ≥240 Hz, P1 alive
   ```
   Landmark check: net center ≈ (1.37, −0.7625, 0).
2. **Planner sanity** (no robot): `ros2 launch hope_planner hope_planner.launch.py`,
   toss real balls, `ros2 topic echo /racket/command` — valid=true commands with
   sensible intercepts, `time_to_strike` counting down. Requires §2 ball physics fitted.
3. **Dry-run** (no robot output): full chain with `mode:=dry_run`, CSV on; check
   swing side, gate rejections, latch in the CSV.
4. **Shadow on the robot**: robot held/hoisted or standing with Path A idle; run the
   chain with `mode:=shadow state_source:=ros` ON THE MDU (the laptop cannot see the
   MDU's DDS) with the ros2-enabled AimRT backend config (§2 last item) so
   `agibot_hardware_bridge` hears real joints/IMU. Verify `[obs verify]` block:
   projected gravity ~[0,0,−1], base_pos ≈ (0,0,pelvis) at the start pose, yaw-align
   offset logged.
5. **Hardware**: same launch with `mode:=hardware`, gain-scale ramp on the backend,
   operator on `/hope/estop`, then `ros2 topic pub /hope/hardware_enable ... true`.
   Start with forehand-side serves only.

**Honest current status of Path B on hardware:** the chain is code-complete and
sim-verified, but the robot-side link (`/body_drive` over ros2 on the MDU) has not
been exercised — the fully deploy-faithful alternative is to add a `/racket/command`
subscriber to the C++ runner (documented as PINGPONG_DEPLOY_ALIGNMENT §10.7) so the
planner drives the exact Path-A binary. That C++ front-end input is the single
missing piece for competition play; everything upstream (mocap→planner→targets) is
identical either way.

## 10. Quick reference

| task | box | directory | command |
| --- | --- | --- | --- |
| train / resume | grasping | `hope_training/whole_body_tracking` | `hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity ...` |
| export onnx | grasping | same | `bash scripts/export_onnx_deploy_parity.sh <run> [ckpt]` |
| MuJoCo gate | host | same | `../.venv-motion/bin/python scripts/mujoco_eval_onnx.py ... --deploy-faithful` |
| sync + parity | hope | `agi/a3_deploy_example` | `bash scripts/sync_pingpong_model.sh <onnx> <name>` |
| build x86 / rockchip | hope / HOST | same | `bash scripts/build_a3_deploy_pkg.sh --arch x86_64\|rockchip --runtime-cfg ...pingpong.yaml` |
| AGI sim policy-only | hope | same | `bash scripts/pp_freebase_watch.sh --single-swing [--oracle-pelvis]` |
| AGI sim + planner loop | hope | same + `hope_ws` | §7 three terminals |
| ROS chain (arena) | hope | `hope_ws` | `ros2 launch hope_bringup hope_pingpong_sim2real.launch.py server:=<IP> onnx_path:=<onnx> mode:=dry_run` |
| ship to robot | host→MDU | `agi/a3_deploy_example` | `rsync ... dist/a3_deploy_rockchip/ agi@10.42.10.12:/agibot/a3_deploy/` |
| run on robot | MDU | `/agibot/a3_deploy` | `taskset -c 4-7 ./run_a3_pingpong.sh --start passive --legs-passive --gain-scale 0.4 --single-swing` |
