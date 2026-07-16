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

- **No accepted mocap-in-the-loop behavior.** The successful 2026-07-02 runner used scripted racket
  targets and synthesized world/torso pose (`loc_mode: perfect_tracking`; only IMU orientation was
  real). Formal ROS flat-topic source wiring into the C++ runner now exists, but has no exact
  ROS/AimRT first tick, vendor MuJoCo behavior or hardware result. Closing the loop also requires
  validating ownership of the HOPE-world → robot-frame target transform (it needs the mocap base
  pose at the interface boundary even though the 175-D actor obs itself does not).
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
- Vendor `SimReset` nonzero base twist is not frame-exact: `/sim/a3/pelvis_twist` publishes
  link-origin linear and angular velocity in odom/world axes, while the subscriber copies world
  angular velocity directly into MuJoCo body-local freejoint qvel and ignores `header.frame_id`.
  Current named-keyframe flows use all-zero velocity and do not trigger it. Do not add a nonzero
  absolute-twist reset to Gate3 until the ROS point/frame contract and a round-trip test are fixed.

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

## Audit update 2026-07-16: RallyV10 field-test timing and task-lifecycle gaps

The RallyV10 field observations (Jiayi's 110-D policy/planner route) were audited against the actual
Python planner, ROS transport and C++ runner. This is a source-and-contract audit only; no robot command was run. The eight requested
questions are not all closed end to end:

1. **Capture timestamp / latency — Partial.** The planner validates a source stamp, maps it to
   monotonic time, publishes it in formal schema 3, and the C++ consumer subtracts source age from
   the remaining strike time. However, the vendored VRPN client currently discards the VRPN packet
   timestamp and stamps the ROS message with host receipt time. The relay preserves that receipt
   stamp. Camera-to-host latency therefore cannot yet be compensated or measured absolutely.
2. **Continuously revised target and time-to-strike — Partial.** The Python planner re-estimates the
   ball, predicts the trajectory and republishes target plus countdown for each admitted sample.
   The formal 179-D runner rechecks the latest tuple before engage, but freezes target and clock
   during the active swing. The optional 110-D streaming path updates position and velocity during the swing but
   still freezes the strike clock. Neither path has a vendor Gate3 behavior result.
3. **Trajectory prediction — Partial.** The ROS planner really does call the repository's
   `BallTrajectoryPredictor` before target planning; the C++ runner only consumes the result. There
   is no evidence that the field V10 launch used this exact source/config, and measured venue
   residuals have not been replayed into training or passed through vendor Gate3.
4. **Training distribution parity — Partial.** Current Stage 1 still uses a fixed reference station
   and fixed per-clip target-sampling boxes. The rolling timing jobs use engineering timing bins, not a fitted venue
   distribution; real strike-position coverage has not been compared quantitatively.
5. **Fast preparation / retiming — Partial.** Training now has approximately 1.0/0.7/0.5-second and
   clip-dependent random retiming, while the question-bank demanded absolute racket target remains
   unchanged. This is uniform clip retiming, not [TOPP](../DEFINITIONS.md#topp); new-motion TOPP and
   an equivalent deployed reference/target-speed interface remain open.
6. **Readable debugging — Partial.** Existing 10 Hz planner diagnostics expose counts, validity and
   time-to-strike, while the runner throttles some gate warnings. There is still no single low-rate
   correlated record containing ball distance to strike plane, source age, predicted intercept,
   command sequence/epoch, runner state and accept/reject reason.
7. **Minimum catchable preparation time — Partial.** Multiple 0.5-second training jobs are live and
   checkpoint-finite, but the running source lacks the complete per-update eligible/ready/physical-
   fall ledger needed for a behavior decision. No 0.5-second ability claim is allowed yet.
8. **Repeated task consumption — Open.** Formal sequence numbers reject stale/out-of-order ROS 2
   DDS transport rows
   and the mailbox keeps the latest row, but there is no consumed ball/rally/task identifier. After
   a swing and the default rest period, a still-fresh valid producer stream may engage again. This
   is not equivalent to consuming one planned strike exactly once, and the field repetition report
   has not been reproduced or fixed in the vendor runtime.

Before the next field attempt, the shortest closure order is: preserve the real VRPN capture stamp;
add a compact correlated planner/runner trace; add a ball/rally task identity with exactly-once
consume semantics; then test rolling target/countdown revisions in shared-interface MuJoCo and
vendor Gate3. G07 remains `Partial`.

Reproduction evidence: planner/runner causal source `6d6b778a`; focused host source tests passed
`66/66` during this audit. These tests do not replace the open runtime and behavior gates above.
