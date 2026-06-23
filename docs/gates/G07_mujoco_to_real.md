# G07 MuJoCo-To-Real Deployment

Status: Not complete

## Goal

Move from MuJoCo/deploy dry-run to safe real A3 execution.

This gate is about safety, timing, message layout, and controlled rollout before performance.

## Inputs

- Deployment source under `agi/code_deployment/a3_deploy_example`.
- Full local runtime assets under `vendor_assets/agibot/a3_deploy_example_full`.
- Policy artifact from G05/G06.
- A3 runtime and hardware access.

## Outputs

- No-command backend/sync verification.
- Inference latency probe.
- Low-gain command test.
- Safe halt and emergency procedure.
- Real-run logs.

## Related Directories

- `agi/code_deployment/`
- `agi/code_deployment/a3_deploy_example`
- `vendor_assets/agibot/a3_deploy_example_full`
- `docs/operations/run_deploy_dryrun.md`
- `docs/interfaces/joint_order_and_robot_state.md`

## Operation Docs

- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

- Backend and sync can run without publishing commands.
- Inference can run without command output.
- Command layout is verified against A3 SDK/runtime.
- Safe halt works before any dynamic motion.
- First real command is low-gain and bounded.

## Current State

Done:

- Agibot deployment documents exist.
- RobotIOBackend adaptation guide exists.
- Deployment source subset is tracked.
- Full runtime package is preserved locally.

Not done:

- No dry-run has been verified in this environment.
- No hardware command test is recorded.
- No real policy has been deployed.

## Risks

- Joint order or command scaling mistakes can be dangerous.
- Latency and dropped messages can destabilize control.
- Generated policy artifacts should not be treated as safe until dry-run and low-gain tests pass.

## Next Steps

1. Build deploy code in the intended target environment.
2. Run backend/sync with command output disabled.
3. Run latency probe.
4. Only then plan a low-gain hardware test.
