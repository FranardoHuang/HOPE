# G07 MuJoCo-To-Real Deployment

Status: Partial (first real-A3 deployment demonstrated 2026-07-02, forehand only; acceptance evidence and the mocap-in-the-loop chain still open)

## Goal

Move from MuJoCo/deploy dry-run to safe real A3 execution.

This gate is about safety, timing, message layout, and controlled rollout before performance.

## Inputs

- Active deploy tree `agi/a3_deploy_example` (tracked): C++ ping-pong runner
  `a3_deploy_onnx_ref_pingpong` reusing AGI `A3AimrtBackend`/`A3PolicyDriver` (Route A), build and
  packaging scripts, runbooks.
- Older vendor reference subset under `agi/code_deployment/a3_deploy_example`; full local runtime
  assets under `vendor_assets/agibot/a3_deploy_example_full`.
- Policy artifact from G05/G06 (currently `model_p4_deployparity.onnx`, 175-D / 31-act, exported
  with the full metadata contract).
- A3 runtime and hardware access (MDU).

## Outputs

- No-command backend/sync verification, inference latency probe, low-gain command test, safe halt.
- Real-run logs and MDU captures.

## Related Directories

- `agi/a3_deploy_example/` — runner source (`src/a3/a3_deploy_onnx_ref/include/a3_pingpong/`),
  `PINGPONG_RUN.md`, `PINGPONG_DEPLOY_ALIGNMENT.md`, `PINGPONG_NEW_CHECKPOINT_TUTORIAL.md`,
  `HARDWARE_BRINGUP_CHECKLIST.md`.
- `docs/operations/run_deploy_dryrun.md`, `docs/operations/run_pingpong_recovery_audit.md`,
  `docs/interfaces/joint_order_and_robot_state.md`, `docs/interfaces/policy_observation_action.md`.

## Operation Docs

- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)
- [../operations/run_pingpong_recovery_audit.md](../operations/run_pingpong_recovery_audit.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

- Backend and sync can run without publishing commands; inference can run without command output.
- Command layout is verified against the A3 SDK/runtime.
- Safe halt works before any dynamic motion; first real command is low-gain and bounded.
- A real swing run is recorded with its checkpoint, config, and behavior notes.

## Current State

Done (2026-06-30 → 2026-07-02, recorded 2026-07-03):

- **First sim-to-real transfer succeeded (2026-07-02).** The unified swing policy ran on the real
  A3 through the Route A runner with AGI's backend/`A3PolicyDriver`/100 Hz sync/watchdog/safe-halt
  reused unmodified, and real behavior matched MuJoCo. Deployed artifact:
  `model_p4_deployparity.onnx` (175-D deploy-parity obs / 31 actions / 50 Hz). See
  `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md` and `agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md` §0.
- **Forehand only.** Backhand is NOT deploy-ready on the current models (training gap:
  teleport-entry clips, no stand-entry coverage) — do not trigger backhand on hardware until
  stand-entry backhand training closes the gap (`PINGPONG_DEPLOY_ALIGNMENT.md` §0).
- The runner validates the whole contract from ONNX metadata at load (joint_names bijection onto
  the 31 backend slots, default_joint_pos, action_scale, kp/kd with a zero-gain guard, obs input
  [1,175] or [1,180], optional `clip_seg_lengths`/`clip_strike_phases`). Actions are 31-DOF
  (`MakeA3Layout31`); neck outputs are overridden post-decode to q=0/kp=40/kd=2; q_des is clamped
  to MJCF joint limits; commands publish `{q_des, kp, kd}` to the implicit-PD backend.
- Hardware-derived fixes are in code (`d17cd57`/`dd32a29`): exporter zero-kp/kd guard; clip clock
  taken from ONNX metadata (the v1-layout default fired the forehand ~0.6 s early on v2-baked
  models); squat guard raised 0.6 → 1.4 rad plus a 0.35 tilt guard (the old threshold sat inside
  the trained swing envelope and its kp-2000 snap-back catapulted the robot); IMU yaw-align capture
  at every SHADOW/MOTION engage (boot-to-boot yaw drift 130-165°).
- Staged bring-up used throughout: PASSIVE → PD_STAND → SHADOW (no publish) → MOTION, with
  `run_a3_pingpong.sh --dry-run` / probe first (`PINGPONG_RUN.md`).
- The x86_64 package build is reproduced from the tracked tree
  (`scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg
  src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`); earlier fingerprints for the
  model_15200 era are in
  [../operations/run_pingpong_recovery_audit.md](../operations/run_pingpong_recovery_audit.md).

Not done:

- **No mocap in the loop.** The runner consumes zero VRPN/mocap topics; racket targets are
  scripted, world/torso pose is synthesized (`loc_mode: perfect_tracking`; only IMU orientation is
  real). The ROS chain (vrpn_mocap → relay → hope_planner → `/racket/command` → hope_wbc_runner)
  exists but is not bridged into the C++ runner. Closing this loop also requires deciding who owns
  the HOPE-world → robot-frame target transform (it needs the mocap base pose at the interface
  boundary even though the 175-D actor obs itself does not).
- Acceptance evidence for the successful runs (dates, checkpoint SHAs, MDU capture paths, observed
  behavior vs MuJoCo) is not yet recorded in this gate.
- No quality baseline on hardware (hit-rate style metrics need the mocap/ball loop).
- Backhand deploy readiness (see above).

## Risks

- Joint order or command scaling mistakes are dangerous — mitigated by the metadata bijection check
  and staged bring-up; keep both mandatory for new checkpoints
  (`PINGPONG_NEW_CHECKPOINT_TUTORIAL.md`).
- Safety guards themselves can cause incidents when their thresholds intersect the trained motion
  envelope (the 0.6 rad squat-guard lesson). Derive guard thresholds from the trained state
  envelope, and prefer damped take-over behaviors to stiff position snaps.
- IMU yaw drift silently corrupts any world-referenced quantity; the engage-time yaw-align must be
  re-verified whenever the boot sequence changes.

## Next Steps

1. Record the 2026-07-02 run evidence (checkpoint SHA, configs, captures) in this gate.
2. Bridge mocap into the deploy chain (VRPN → planner target in the runner), including the
   world→robot target transform design; rehearse in the shared-interface MuJoCo first.
3. Close the backhand stand-entry training gap before enabling backhand on hardware.
4. Define the first hardware quality metrics (hit rate on served balls) once the ball loop exists.

## Audit update 2026-07-10: runtime safety package

The publish path now fail-closes on mode races, missed deadlines and invalid
state instead of relying on the policy thread to stop itself:

- mode generation, command send and zero-gain barrier are linearized;
- an independent deadline supervisor faults and retries the safe halt;
- NaN/Inf, missing measured effort, requested/measured effort envelope breach,
  soft/hard q-des breach and localization/yaw readiness block publishing;
- MOTION/SHADOW re-entry resets clocks/latches; planner owns timing and forces
  `swing_speed=1`; unsafe scripted hotkeys are blocked during a live swing;
- formal ONNX must bind the exact wrist-to-racket point and schema-v3
  execution contract; 179/181 still fail closed pending contract day.

A RunPod Release build found that the former global `-ffast-math` optimized
away `std::isfinite` safety checks. It has been removed, GNU/Clang builds force
`-fno-fast-math -fno-finite-math-only`, and a compile-time header rejects any
target that re-enables finite-only math. Release tests must remain part of the
gate; debug-only tests are insufficient. Ubuntu 24.04/GCC 13 verification:
portable Release `188 passed/4 skipped`, ROS 2 Jazzy Release `202 passed/4
skipped`; the ping-pong runner and runtime probe linked. Reproduction commands
and the direct-CMake `joint_msgs` library path are in
[build_and_test.md](../operations/build_and_test.md).

Torque mapping clarification: shoulder pitch/roll are `60 Nm`; shoulder yaw,
elbow and wrist roll are `24 Nm`; wrist pitch/yaw are `6 Nm`. Future
robot-centric strokes should preferentially use waist and shoulder
pitch/roll, but these model limits are not substitutes for vendor continuous
torque-speed-temperature curves.

Still not solved:

- a userspace watchdog cannot preempt a backend `SendCommand` already blocked
  inside the call, and there is no downstream controller sequence ACK/timeout;
- external-base orientation ownership (mocap world quaternion vs engage-
  relative IMU yaw) is unresolved;
- real mount calibration and 179/181 wire contract remain open.

No real-robot command test was run in this audit. G07 remains `Partial`.

## Audit update 2026-07-13: exact planner-policy bytes pass portable Release

Exact planner-policy source `c0a8e46` compiled and linked in a clean isolated Pod2 Ubuntu
24.04/GCC 13 portable Release after restoring the documented private Unitree SDK. Focused
planner/first-tick tests passed `40/40`; the complete native suite passed 233 tests with 5 explicit
optional-asset skips and 0 failures. Both test and production runner binaries are content-addressed,
and all 80 compile commands preserve strict finite math.

This closes only the source/binary blocker. The production runner was not executed; ROS/AimRT
Release, formal ONNX runtime loading, backend first tick, vendor simulator behavior and hardware
remain open. See
[the experiment record](../experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md). No real
robot command ran; G07 remains `Partial`.
