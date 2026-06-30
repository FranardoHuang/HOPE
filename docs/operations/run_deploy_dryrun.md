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

## Required Documentation Before Hardware

- [../interfaces/joint_order_and_robot_state.md](../interfaces/joint_order_and_robot_state.md) must be filled.
- G07 must list exact build command, dry-run command, latency result, and safe halt result.
- Ping-pong recovery work must also record the exact source/config/model
  fingerprint and the destructive-audit result from
  [run_pingpong_recovery_audit.md](run_pingpong_recovery_audit.md).
