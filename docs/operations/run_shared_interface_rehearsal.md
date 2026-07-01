# Run Shared-Interface Rehearsal

Status: Draft

## Goal

Use the MuJoCo sim that shares the same `/body_drive/*` interface as the real
A3 to validate the exact deploy runner path before any hardware motion.

All commands below assume the `distrobox enter hope` Jazzy environment. On this
machine the host only exposes `/opt/ros/lyrical`, which is not sufficient for
this build/runtime path.

## Current Materials

- Local HOPE ping-pong deploy source: `agi/a3_deploy_example`
- Ping-pong ONNX: `agi/a3_deploy_example/assets/a3_runtime/models/model_15200.onnx`
- Shared-interface sim helpers:
  - `agi/a3_deploy_example/scripts/run_sim.sh`
  - `agi/a3_deploy_example/scripts/run_mode.sh`
  - `agi/a3_deploy_example/scripts/reset_sim.sh`
  - `agi/a3_deploy_example/scripts/run_oracle.sh`

## Sim Variants

- `source sim`:
  `agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install`
  Uses the shared `/body_drive/*` topics and also publishes `/sim/a3/*`
  oracle/reset topics.
- `standalone sim`:
  `agi/a3_deploy_example/mujoco_sim_standalone/`
  Uses the shared `/body_drive/*` topics only. No `/sim/a3/*`.

Use `./scripts/run_sim.sh --print-paths` to confirm what the local machine can
currently see.

## Build The Ping-Pong Runner

```bash
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example
source /opt/ros/jazzy/setup.bash

bash scripts/build_a3_deploy_pkg.sh \
  --arch x86_64 \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
```

Expected packaged artifacts:

```bash
ls dist/a3_deploy_x86_64/run_a3_pingpong.sh \
   dist/a3_deploy_x86_64/config/a3_runtime_config.pingpong.yaml \
   dist/a3_deploy_x86_64/models/model_15200.onnx
```

## Shared-Interface Rehearsal Order

Terminal 1, start the sim:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
./scripts/run_sim.sh --print-paths

# recommended: source sim, supports oracle/reset too
A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh

# optional fallback: standalone
A3_SIM_FLAVOR=standalone ./scripts/run_sim.sh a3_t2d5_cfg.yaml
```

Terminal 2, run the exact ping-pong deploy path:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --dry-run
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --reference-playback
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --start shadow --perfect-tracking --level 1
```

Optional sim-only closed-loop check on the same shared interface:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
./scripts/run_mode.sh B 10 0.4
```

Optional oracle mode, source sim only:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
./scripts/run_oracle.sh
./scripts/run_mode.sh C 10 0.4
```

Optional source-sim reset:

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
./scripts/reset_sim.sh
```

## Expected Outcomes

1. `--dry-run` proves the six input topics, sync path, and safe-halt behavior.
2. `--reference-playback` proves joint order, sign, scaling, and the command
   publish path without ONNX in the loop.
3. `shadow` proves the 180-D observation builder, `time_step`, ONNX front-end,
   and `level 0/1` clock without command output.
4. `run_mode.sh B ...` proves the same `/body_drive/*` closed-loop path against
   MuJoCo before moving to hardware.

## Known Limitations

- `perfect_tracking` is still a placeholder world-position mode, not a real
  mocap-fed localizer.
- `oracle` is sim only. It depends on `/sim/a3/pelvis_pose` and cannot be used
  on hardware.
- If `level 1` is fine on the hoist but ground support fails, treat that as a
  released-leg support problem first, not as evidence that the shared interface
  is broken.
- `standalone sim` is useful for A/B shared-interface validation, but it does
  not provide `/sim/a3/*` reset/oracle topics.

## Do Not Skip Before Hardware

Only move to the real robot after the shared-interface rehearsal is recorded and
the hardware bring-up order remains:

`--dry-run` -> `--reference-playback` -> `shadow` -> low-gain `MOTION`
