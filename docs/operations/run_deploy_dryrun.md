# Run Deploy Dry-Run

Status: Draft

## Current Materials

- Source/config subset: `agi/code_deployment/a3_deploy_example`
- Ignored local ping-pong package observed on this machine:
  `agi/a3_deploy_example`
- Full local payload: `vendor_assets/agibot/a3_deploy_example_full`
- Deployment docs: `agi/code_deployment/A3 deploy example.md`
- Backend guide: `agi/code_deployment/RobotIOBackend 架构与策略适配指南.md`

## Task Setup

Use the target Agibot deploy/MuJoCo environment, not the ROS planner environment and not the Isaac training environment.

On this machine, that means `distrobox enter hope`; the host environment only
has `/opt/ros/lyrical`, so host-shell builds do not reproduce the Jazzy deploy
path.

This task does require ignored local assets. Before running deploy or standalone MuJoCo dry-runs, restore:

```text
vendor_assets/agibot/a3_deploy_example_full/
```

Check:

```bash
test -d vendor_assets/agibot/a3_deploy_example_full && echo "Agibot full deploy payload present"
```

The tracked source subset in `agi/code_deployment/a3_deploy_example/` is enough for code review and integration planning, but the full ignored payload is needed for runtime assets such as models, sysroots, prebuilt libraries, and standalone runtime files.

For ping-pong hardware recovery specifically, use
[run_pingpong_recovery_audit.md](run_pingpong_recovery_audit.md) before any
dynamic ONNX motion. The ignored `agi/a3_deploy_example/` package contains a
local ping-pong runner and model artifact, but it is not a reproducible tracked
source package by itself.

Because MuJoCo and the real A3 share the same `/body_drive/*` interface, also
record the sim-first rehearsal in
[run_shared_interface_rehearsal.md](run_shared_interface_rehearsal.md) before
any hardware command test.

## Safety Order

Do not jump directly to hardware motion.

1. Build deploy code in the target environment.
2. Start backend/sync without loading policy and without publishing commands.
3. Run inference latency probe without command output.
4. Verify joint order and command scaling.
5. Verify reference playback through the exact hardware command path, without
   ONNX.
6. Verify safe halt.
7. Only then plan low-gain bounded hardware command tests.

## Ping-Pong `model_15200` Fast Path

For the current HOPE ping-pong runner, use the custom executable and runtime
config instead of AGI's native `a3_deploy_onnx_ref` front-end:

```bash
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example
source /opt/ros/jazzy/setup.bash

bash scripts/build_a3_deploy_pkg.sh \
  --arch x86_64 \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml

cd dist/a3_deploy_x86_64
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --dry-run
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --reference-playback
A3_SOURCE_ROBOT_ENV=0 ./run_a3_pingpong.sh --start shadow --perfect-tracking --level 1
```

Use that order intentionally:

1. `--dry-run` proves the six input topics, sync, and safe-halt path.
2. `--reference-playback` proves joint order, signs, command scaling, and
   latency without ONNX in the loop.
3. `shadow` proves the 180-D observation, `time_step`, ONNX front-end, and
   `level 0/1` clock without publishing commands.

On this machine, the rehearsal should not stop there. Because sim and hardware
share the same `/body_drive/*` interface, the next gate is to run the same
runner against MuJoCo through
[run_shared_interface_rehearsal.md](run_shared_interface_rehearsal.md), then
carry the validated flags onto hardware.

Current field limitation to keep in mind:

- The ping-pong runner does not yet consume the HOPE mocap/VRPN topics
  directly. `perfect_tracking` is still a front-end placeholder for world
  position, not a real localizer.
- If `level 1` looks good on the hoist but the robot cannot stand without the
  support, treat that as a released-leg support issue first. Start with
  `--official-stand --auto-leg-hold --leg-gain-scale 0.5 --ankle-gain-scale 1.0`.
  If the knees still sink under reduced hoist load, escalate to
  `--leg-stand-gains --leg-clamp-rad 0.15` and, if stiff legs twitch,
  `--leg-smooth-alpha 0.2`.

## Required Documentation Before Hardware

- [../interfaces/joint_order_and_robot_state.md](../interfaces/joint_order_and_robot_state.md) must be filled.
- G07 must list exact build command, dry-run command, latency result, and safe halt result.
- Ping-pong recovery work must also record the exact source/config/model
  fingerprint and the destructive-audit result from
  [run_pingpong_recovery_audit.md](run_pingpong_recovery_audit.md).
