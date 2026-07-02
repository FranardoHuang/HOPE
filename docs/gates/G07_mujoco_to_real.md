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
- [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)
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
- The training/export side now records a first-class 175-D deploy-parity actor
  observation contract (`task=HOPEPingPongDeployParity`) and exports that
  contract metadata with the ONNX artifact, so deploy-side ports have one
  tracked source of truth to match.
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
- The local ping-pong runner already includes a separate reference-playback
  mode and a staged `PASSIVE -> PD_STAND -> SHADOW/MOTION` bring-up flow,
  rather than going directly from ONNX export to dynamic hardware motion.
- The ping-pong packager path can now preserve the ping-pong runtime-config
  basename and generate a `run_a3_pingpong.sh` wrapper when built with
  `--runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`.
- The local helper scripts now target the actually present shared-interface sim
  layout on this machine: `scripts/run_sim.sh` auto-detects the source-built
  `agi/A3_MuJoCo_Sim/.../build/install` package first and falls back to
  `mujoco_sim_standalone/`, while `run_mode.sh`/`reset_sim.sh`/`run_oracle.sh`
  no longer depend on the stale `aimrt_mujoco_sim_source/...` path.
- The x86_64 ping-pong package build was reproduced on 2026-07-01 inside
  `distrobox hope` with:
  `bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`.
  Result: `dist/a3_deploy_x86_64/` now contains `run_a3_pingpong.sh`,
  `config/a3_runtime_config.pingpong.yaml`,
  `config/a3_aimrt_config.pingpong_iceoryx.yaml`, and
  `models/model_15200.onnx`.

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
- No recorded reference-playback result has yet verified SDK joint order,
  signs, command scaling, and group latency on this machine or on hardware.
- Hoist behavior has not been accepted as policy-transfer evidence.
- Current hardware issue remains open: `level 1` swing quality alone is not the
  blocking factor; the unresolved gap is ground weight-bearing support when the
  released legs must stand without the hoist.
- The current ping-pong runner still does not consume the HOPE mocap/VRPN
  topics directly; hardware uses `perfect_tracking` for world position unless a
  separate localizer bridge is added.
- The currently observed local ping-pong runner is still a 180-D front-end with
  base-position placeholders / `perfect_tracking`, so it is not yet a verified
  implementation of the new 175-D deploy-parity actor contract.

## Risks

- Joint order or command scaling mistakes can be dangerous.
- Latency and dropped messages can destabilize control.
- Generated policy artifacts should not be treated as safe until dry-run and low-gain tests pass.
- A deploy-parity-trained ONNX does not by itself fix the front-end mismatch if
  the runner still builds the older 180-D observation.
- Hoist support changes contact and balance dynamics; poor hoist swing quality
  can expose command-path bugs but does not prove policy failure by itself.
- Tuning the policy before reference playback passes can hide joint-order,
  state-estimation, or command-scaling defects.
- A live mocap system can help diagnose or future-proof localization, but by
  itself it will not fix the current released-leg support problem unless it is
  actually bridged into the deploy front-end.

## Next Steps

1. Promote or document the exact ping-pong source/config/model provenance used
   for hardware tests, including the `--runtime-cfg` build command that
   produces the packaged ping-pong wrapper/config aliases.
2. Run the destructive audit in
   [../operations/run_pingpong_recovery_audit.md](../operations/run_pingpong_recovery_audit.md).
3. Record a reproducible `--reference-playback`, `--dry-run`, and `shadow`
   result before further ONNX tuning.
4. Record the shared-interface MuJoCo rehearsal from
   [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)
   before moving back to hardware tuning.
5. Re-test ground support with the current bring-up set
   `--official-stand --auto-leg-hold --leg-gain-scale 0.5 --ankle-gain-scale 1.0`,
   then escalate to `--leg-stand-gains --leg-clamp-rad 0.15` and
   `--leg-smooth-alpha 0.2` if the knees still sink under reduced hoist load.
6. Port and verify the deploy-side ping-pong obs builder against the exported
   175-D deploy-parity contract and golden capture before treating a hardware
   run as an honest front-end parity test.
7. If the field mocap is meant to replace `perfect_tracking`, define and
   document the actual bridge/topic/frame contract first; do not assume the
   current ping-pong runner is already using it.
