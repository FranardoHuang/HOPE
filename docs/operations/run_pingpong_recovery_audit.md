# Ping-Pong Deployment Recovery Audit

Status: Draft

## Purpose

Use this runbook when the A3 ping-pong deployment behaves badly on the real
robot or hoist. The goal is to decide whether the current custom deployment
path is valid, salvageable, or should be replaced.

This is a destructive audit in the engineering sense: every unverified
assumption must either pass a reproducible check or be removed from the trusted
path. Do not treat hoist behavior, visual similarity, or partial joint motion as
proof of sim-to-real transfer.

## Current Verdict

Salvage, not keep.

The AGI backend, body-drive command path, `A3PolicyDriver`, safe halt behavior,
and the custom 180-D observation front-end pieces are worth preserving. The
current ping-pong motion path is not accepted for hardware performance claims
until the baseline-aligned audit below passes.

If the audit shows joint order/sign, timing, or command scattering problems,
replace the custom front-end around the AGI runner before any more policy
tuning. If the audit passes and the policy still fails on ground, the likely
failure moves to policy training, state estimation, or hoist-to-ground mismatch.

## Baseline Facts

- The official AGI deploy example is the trusted baseline for backend lifecycle,
  synchronized robot state, body-drive command layout, watchdog behavior, and
  safe halt.
- The official deploy policy front-end is not a drop-in path for the current
  HOPE ONNX artifact. The official front-end expects a 29-DOF policy and a
  large `obs_dict` tokenizer-style input; the HOPE `model_15200.onnx` path uses
  `obs[1,180]` plus `time_step[1,1]` and returns 31-DOF actions plus reference
  tensors.
- The intended command semantics remain position targets with backend PD:
  `q_des`, `kp`, and `kd` are meaningful; `dq_des` and `tau_ff` are zero for
  the policy command path.
- Head and neck joints are not policy-controlled in the AGI 29-DOF baseline and
  should be pinned or held explicitly when scattering into the 31-slot backend
  layout.

## Local Evidence Observed On 2026-06-30

The ignored local package below exists in this workspace but is not tracked by
git:

```text
agi/a3_deploy_example/
```

It contains a ping-pong C++ runner that reuses AGI runtime components:

```text
src/a3/a3_deploy_onnx_ref/src/a3_deploy/a3_pingpong_main.cpp
src/a3/a3_deploy_onnx_ref/include/a3_pingpong/
dist/a3_deploy_x86_64/a3_deploy_onnx_ref_pingpong
dist/a3_deploy_x86_64/models/model_15200.onnx
```

Recorded local fingerprints:

```text
model_15200.onnx
sha256 87ea1db28183b00556d469a89d7e91070c27bf28901f20dd41c74e4bd3743a09

a3_deploy_onnx_ref_pingpong
sha256 8a82fef237c78122b7c566d95a3245203f1a15afd7040ad6719eaa18c0de01ae
```

The packaged x86_64 binary did not have all runtime libraries discoverable from
this shell. `ldd` reported missing ROS 2 and ONNX Runtime shared libraries until
the target runtime environment is sourced or the package layout is fixed. This
blocks accepting the package as reproducible from a fresh checkout.

The local source config also differs from the dist config: the ping-pong runtime
YAML was observed in `dist/a3_deploy_x86_64/config/`, not in the tracked source
subset. Treat that as a packaging and reproducibility gap.

## Minimal Destructive Audit

Run these checks in order. Stop at the first failure and fix that layer before
testing motion again.

### A0 Asset And Source Provenance

```bash
git status --short
git ls-files agi/a3_deploy_example
sha256sum agi/a3_deploy_example/dist/a3_deploy_x86_64/models/model_15200.onnx
sha256sum agi/a3_deploy_example/dist/a3_deploy_x86_64/a3_deploy_onnx_ref_pingpong
```

Pass condition: the source used to build the runner, the config used to run it,
and the model artifact are named in this doc or promoted to tracked source plus
ignored binary policy locations. An ignored package alone is not enough.

### A1 ONNX Contract

```bash
python3 - <<'PY'
import onnxruntime as ort
p = "agi/a3_deploy_example/dist/a3_deploy_x86_64/models/model_15200.onnx"
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
print("inputs", [(i.name, i.shape, i.type) for i in s.get_inputs()])
print("outputs", [(o.name, o.shape, o.type) for o in s.get_outputs()])
PY
```

Pass condition: inputs are `obs[1,180]` and `time_step[1,1]`, and action output
is 31-D. If the runtime environment lacks `onnxruntime`, install or source the
target environment before using this result.

### A2 Joint Map And Scatter

Run the local joint-map test or its equivalent in the target environment:

```bash
cd agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref
cmake --build build --target pp_jointmap_test
./build/pp_jointmap_test
```

Pass condition: every policy joint maps exactly once to the intended SDK slot,
head and neck handling is explicit, and signs/defaults match
`docs/interfaces/joint_order_and_robot_state.md`.

### A3 Observation Parity

Run the C++ vs Python golden observation parity check:

```bash
cd agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref
cmake --build build --target pp_parity_test
./build/pp_parity_test include/a3_pingpong/test/golden.txt
```

Pass condition: the C++ observation builder matches the Python/export contract
for the 180-D observation vector, including `swing_type`.

### A4 No-Publish Backend Sync

Run the AGI backend with command publishing disabled.

Pass condition: all required state channels arrive at the expected rate, skew is
bounded, joint names resolve consistently, and safe halt can be built without
publishing commands.

### A5 Reference Playback, No ONNX

Add or run a mode that bypasses ONNX and plays bounded reference targets through
the same scatter, gains, and AGI body-drive publisher.

Pass condition: with low gains and small amplitudes, each group follows the
commanded reference with correct sign, range, and latency. Test waist, left arm,
right arm, legs, and neck/head hold separately before combined motion.

### A6 ONNX Shadow, Then Motion

Only after A0 through A5 pass:

1. Run ONNX in shadow mode and log `q`, `dq`, `q_des`, action, reference tensors,
   mode, latency, and backend skew.
2. Run level-0 hold or wind-up with low gains.
3. Run one bounded level-1 swing with a hard stop condition.

Pass condition: shadow commands are finite, bounded, correctly scattered, and
consistent with offline replay before any published dynamic swing.

## Hoist-Specific Interpretation

Slow or static hoist behavior is not proof that the policy is wrong. Hoist
support changes contact, base dynamics, torso loading, and balance feedback.

Treat hoist failures as useful only for these checks:

- command signs and joint order;
- range limits and saturation;
- backend timing and dropped messages;
- obvious state-estimation problems;
- safe halt behavior.

Do not use hoist swing quality as the final acceptance metric for a whole-body
table-tennis policy.

## Replacement Architecture

If the current path fails the audit, rebuild around this smaller architecture:

1. Keep AGI `A3AimrtBackend`, state sync, command topics, `A3PolicyDriver`,
   watchdog, and safe halt.
2. Add one HOPE front-end module only:
   `RobotState -> 180-D obs + time_step -> ONNX -> 31-D action -> target_q`.
3. Keep one explicit joint-order table with unit tests for policy order,
   Isaac/MuJoCo order, and SDK order.
4. Add reference-playback mode beside ONNX mode, not inside the ROS planner.
5. Add CSV/MCAP logging at the driver boundary before publishing commands.
6. Keep ROS planner integration upstream of racket targets only; do not let ROS
   introduce a second hardware command path until the C++ baseline path passes.

## Ground Gate

After hoist checks pass, ground testing still needs its own gate:

- low-gain stand and safe halt;
- bounded reference playback on ground;
- ONNX shadow on ground;
- one bounded swing with operator stop;
- comparison against MuJoCo replay using the exact logged observations.

## Decision Tree

- If A2 fails: replace joint map/scatter before any policy discussion.
- If A3 fails: replace observation builder or exported model contract.
- If A4 fails: fix AGI runtime/backend setup, not the policy.
- If A5 fails: fix command semantics, gains, signs, or hardware limits.
- If A6 fails in shadow: fix ONNX front-end, normalization, timing, or reference
  clock.
- If A6 passes on hoist but ground fails: investigate base estimation,
  balance/contact mismatch, and policy robustness.
- If ground reference playback passes but ONNX fails: retrain or adapt the
  policy; do not keep tuning deploy glue.

## Definition Of Success

Success for this recovery pass is not a good rally. Success is proving the
deployment contract:

- official AGI backend path reused for hardware commands;
- exact model, config, source, and binary provenance recorded;
- 180-D observation and 31-D action contract verified in the target runtime;
- joint order, sign, scaling, and scatter tested independently of ONNX;
- reference playback works before policy playback;
- safe halt works from every mode;
- first ONNX hardware motion is bounded, logged, and replayable offline.
