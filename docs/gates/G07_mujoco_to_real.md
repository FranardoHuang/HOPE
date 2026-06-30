# G07 MuJoCo-To-Real Deployment

Status: Partial

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
- `docs/operations/run_pingpong_recovery_audit.md`
- `docs/interfaces/joint_order_and_robot_state.md`

## Operation Docs

- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/run_pingpong_recovery_audit.md](../operations/run_pingpong_recovery_audit.md)
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
- Local ignored ping-pong deploy package was inspected on 2026-06-30 at
  `agi/a3_deploy_example/`. It contains a custom C++ ping-pong runner that
  reuses AGI `A3AimrtBackend` and `A3PolicyDriver`, plus the HOPE
  `model_15200.onnx` artifact.
- Local fingerprints recorded in
  [../operations/run_pingpong_recovery_audit.md](../operations/run_pingpong_recovery_audit.md):
  `model_15200.onnx` SHA256
  `87ea1db28183b00556d469a89d7e91070c27bf28901f20dd41c74e4bd3743a09`;
  `a3_deploy_onnx_ref_pingpong` SHA256
  `8a82fef237c78122b7c566d95a3245203f1a15afd7040ad6719eaa18c0de01ae`.

Not done:

- No dry-run has been verified in this environment.
- No hardware command test is recorded.
- No real policy has been deployed.
- The ping-pong runner and model are currently ignored/local, not reproducible
  from tracked source alone.
- `onnxruntime` was not available in this shell, so the local ONNX contract was
  not inspected here.
- The packaged x86_64 ping-pong binary reported missing runtime shared
  libraries from this shell until the target environment/package layout is
  fixed.
- No independent reference-playback test has verified SDK joint order, signs,
  command scaling, and group latency without ONNX.
- Hoist behavior has not been accepted as policy-transfer evidence.

## Risks

- Joint order or command scaling mistakes can be dangerous.
- Latency and dropped messages can destabilize control.
- Generated policy artifacts should not be treated as safe until dry-run and low-gain tests pass.
- Hoist support changes contact and balance dynamics; poor hoist swing quality
  can expose command-path bugs but does not prove policy failure by itself.
- Tuning the policy before reference playback passes can hide joint-order,
  state-estimation, or command-scaling defects.

## Next Steps

1. Promote or document the exact ping-pong source/config/model provenance used
   for hardware tests.
2. Run the destructive audit in
   [../operations/run_pingpong_recovery_audit.md](../operations/run_pingpong_recovery_audit.md).
3. Verify ONNX contract, joint map/scatter, observation parity, backend sync,
   safe halt, and reference playback before ONNX motion.
4. Only then plan a low-gain bounded hardware test.
