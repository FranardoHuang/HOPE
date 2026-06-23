# Run Deploy Dry-Run

Status: Draft

## Current Materials

- Source/config subset: `agi/code_deployment/a3_deploy_example`
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
