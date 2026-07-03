# Run Deploy Dry-Run

Status: Draft

## Current Materials

- Active tracked ping-pong deploy tree: `agi/a3_deploy_example/` — the C++ runner `a3_deploy_onnx_ref_pingpong` (used for the first successful sim-to-real, 2026-07-02), build script `scripts/build_a3_deploy_pkg.sh`, and runbooks `PINGPONG_RUN.md`, `PINGPONG_DEPLOY_ALIGNMENT.md`, `HARDWARE_BRINGUP_CHECKLIST.md`, `MUJOCO_VALIDATION_RUNBOOK.md`
- Older vendor reference subset: `agi/code_deployment/a3_deploy_example`
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

The active tracked ping-pong deploy tree is `agi/a3_deploy_example/`; the older vendor reference subset in `agi/code_deployment/a3_deploy_example/` remains useful for code review and integration planning. The full ignored payload is needed for runtime assets such as models, sysroots, prebuilt libraries, and standalone runtime files.

## Ping-Pong Dry-Run

For the ping-pong runner, the actual dry-run sequence starts from the package built by `agi/a3_deploy_example/scripts/build_a3_deploy_pkg.sh`; from the build output directory:

```bash
./run_a3_pingpong.sh --dry-run
```

then the inference/latency probe, shadow mode, and the staged bringup PASSIVE -> PD_STAND -> SHADOW -> MOTION. Follow [../../agi/a3_deploy_example/PINGPONG_RUN.md](../../agi/a3_deploy_example/PINGPONG_RUN.md) and [../../agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md](../../agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md) §0.

## Safety Order

Do not jump directly to hardware motion.

1. Build deploy code in the target environment.
2. Start backend/sync without loading policy and without publishing commands.
3. Run inference latency probe without command output.
4. Verify joint order and command scaling.
5. Verify safe halt.
6. Only then plan low-gain bounded hardware command tests.

## Required Documentation Before Hardware

- [../interfaces/joint_order_and_robot_state.md](../interfaces/joint_order_and_robot_state.md) must be filled.
- G07 must list exact build command, dry-run command, latency result, and safe halt result.
