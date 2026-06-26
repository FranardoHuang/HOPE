# G04 Sim Modeling In MuJoCo And Isaac

Status: Partial

## Goal

Build consistent A3 simulation assets in MuJoCo and Isaac so training and deployment can share robot semantics.

This gate is about robot model correctness, not RL performance.

## Inputs

- A3 URDF and meshes under `agi/URDF/`.
- Agibot MuJoCo/AimRT simulation materials under `agi/A3_MuJoCo_Sim/`.
- Isaac training scaffold under `hope_training/whole_body_tracking`.
- A3 deployment configs under `agi/code_deployment/`.

## Outputs

- Validated MuJoCo model.
- Validated Isaac asset.
- Shared joint names and joint order.
- Shared racket mount definition.
- Sim model limitations.

## Related Directories

- `agi/URDF/`
- `agi/A3_MuJoCo_Sim/`
- `agi/code_deployment/a3_deploy_example/mujoco_sim_standalone`
- `hope_training/whole_body_tracking`
- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis`
- `docs/interfaces/joint_order_and_robot_state.md`
- `docs/interfaces/frames_and_coordinates.md`

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

- A3 model loads in MuJoCo.
- A3 model loads in Isaac.
- URDF/MJCF/Isaac asset agree on joint names, limits, and ordering.
- Standing pose, base height, and racket FK are checked.
- Contact and table/ball parameters are documented.

## Current State

Done:

- A3 URDF and MuJoCo support materials exist.
- Agibot MuJoCo sim source exists.
- Tracked deploy subset includes standalone MuJoCo configs.
- On 2026-06-25, this harness restored the ignored package-local A3 Isaac asset under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` from tracked `agi/URDF/A3T2.5-URDF-std-pingpang/` materials and rewrote URDF mesh paths to local `../meshes/` references. Host verification found `86` mesh references and `0` missing files.
- The branch now includes an A3 Isaac/BeyondMimic robot config using the Agibot-provided ping-pong URDF path, official joint/body names, deploy-transcribed PD gains, standing pose, and action-scale logic.
- `scripts/prepare_a3_isaac_asset.py` now prepares the generated Isaac asset from `agi/URDF/A3T2.5-URDF-std-pingpang/` and verifies the prepared `model.urdf` by parsing all mesh references. The check rejects stale `package://.../meshes` references, verifies every `../meshes/...` file exists, and requires `right_hand_pingpang_Link.STL`, `pingpang_red_Link.STL`, `pingpang_black_Link.STL`, and `pingbang_ball_Link.STL`.
- A working 31-DOF joint-order YAML exists at `hope_training/config/joint_order_agibot_a3.yaml`.
- `reimplement.md` records that the A3 task registers and the env launches headless with finite rewards on the copied A3 ping-pong URDF asset.
- `origin/train_1` adds `HOPE-TableTennis-AgibotA3-v0`, a HOPE-frame Isaac Lab table/net/ball/A3 scene with modular geometry constants, optional drag and Magnus force hooks, table/net/floor contact materials, 400 Hz physics, CCD enabled, ball serve reset, and placeholder returner rewards.
- The table-tennis scene now includes a tracked Purdue PACE table/net USD visual overlay under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`. Physics still comes from invisible cuboid colliders; the USD is visual-only.
- Table-tennis ball/table contact now follows Purdue PACE materials by default: ball mass `3.4 g`, ball restitution/friction `0.9/0.1`, table restitution/friction `0.95/0.4`, multiplicative combine for an effective ball-table normal restitution of `0.855`. HOPE-calibrated aero drag is available but off by default for Purdue parity.
- `tests/test_table_tennis_geometry.py` covers table/frame geometry and pure drag/Magnus math; the drag/Magnus tests skip automatically if host `torch` is missing.

Not done:

- This Codex shell has not independently run Isaac because the required GPU/Isaac environment is not active here.
- The table-tennis scene has not yet been verified in-sim in this Codex shell with `scripts/play_table_tennis.py`.
- Self-collision is disabled in the Isaac config due to overlapping wrist/racket collision meshes; a cleaner Isaac collision asset is still needed before re-enabling it.
- Sim parity between MuJoCo and Isaac is not established.
- Hardware SDK parity for joint order and command/state layout is not established.
- The table-tennis scene is not yet a trained returner or accepted sim-to-real baseline; it is a G04/G08 candidate scene.
- The internal main branch intentionally keeps multiple A3 asset layers: ping-pong URDF source for WBC, standard non-racket `agi/URDF/a3_t2d5/` for comparison, and Agibot MuJoCo/AimRT ping-pong MJCF/collision materials for parity. Do not delete the standard `right_hand_Link.STL` or MuJoCo collision assets without a recorded replacement.

## Current Verification Commands

Plain host checks:

```bash
python3 scripts/prepare_a3_isaac_asset.py --check
python3 hope_training/whole_body_tracking/tests/test_table_tennis_geometry.py
python3 -m py_compile hope_training/whole_body_tracking/scripts/play_table_tennis.py
python3 -m py_compile hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/*.py
```

GPU/Isaac checks, inside the sourced training environment:

```bash
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
hope_isaac_py scripts/play_table_tennis.py --fix_base
hope_isaac_py scripts/play_table_tennis.py --enable_aero --headless --steps 300
```

## Risks

- Differences in actuator model, contact, timestep, or joint order can break sim-to-sim transfer.
- URDF import can lose inertial or collision fidelity.
- Racket mount errors directly corrupt planner-to-WBC training.

## Next Steps

1. Use `hope_training/config/joint_order_agibot_a3.yaml` as the current working joint-order source and verify it against the real SDK before deployment.
2. Load the same standing pose in MuJoCo and Isaac and compare FK.
3. Clean or replace Isaac collision geometry so self-collision can be revisited.
4. Document MuJoCo/Isaac parity metrics before policy transfer work.
