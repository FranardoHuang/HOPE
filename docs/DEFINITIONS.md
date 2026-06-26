# Definitions

This file keeps shared language stable across humans, agents, documents, and code.

## Baseline Target

`HITTER-compatible baseline` means:

- The system keeps the HITTER-style hierarchy: model-based planner plus RL whole-body controller.
- The planner emits striking position, racket velocity, and strike time.
- The WBC consumes planner targets plus robot state and outputs joint position targets.
- Evaluation uses planner prediction error, hit rate, return rate, reaction time, and rally readiness.

It does not mean blindly copying the HITTER implementation details. Hardware, mocap, simulator, deployment runtime, and improvement strategy should follow the actual A3 project constraints.

## Stage Versus Gate

`Stage` describes the physical pipeline:

1. Real preparation
2. Real-to-sim
3. Sim-to-sim
4. Sim-to-real
5. Real evaluation and improvement

`Gate` describes a verifiable project milestone. Gates can share folders and can overlap in time, but each gate needs its own target, inputs, outputs, and acceptance criteria.

## Core Frames

- `world`: HOPE table/world frame. Use ROS REP 103: X forward toward opponent, Y left, Z up.
- `base_link`: robot body reference frame. Its exact physical location must come from the robot model and SDK.
- `racket`: inferred frame from robot FK and fixed racket mount. It is not tracked by mocap.
- `ball`: measured ball position from mocap or future perception.

See [interfaces/frames_and_coordinates.md](interfaces/frames_and_coordinates.md).

## Artifact Types

- `Source`: code, launch files, config, scripts, message definitions, docs.
- `Curated data`: small sample bag/CSV/plot files used for tests and examples.
- `Runtime asset`: model weights, ONNX/RKNN/TensorRT engine files, sysroots, prebuilt binaries, robot vendor packages.
- `External reference`: upstream repos used for study or optional implementation paths.

## Done

A gate is `Done` only when it has:

1. Reproducible commands.
2. Passing verification.
3. Recorded inputs and outputs.
4. Updated docs.
5. Known limitations listed.
