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
- `docs/interfaces/joint_order_and_robot_state.md`

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
- The branch now includes an A3 Isaac/BeyondMimic robot config using the Agibot-provided ping-pong URDF path, official joint/body names, deploy-transcribed PD gains, standing pose, and action-scale logic.
- A working 31-DOF joint-order YAML exists at `hope_training/config/joint_order_agibot_a3.yaml`.
- `reimplement.md` records that the A3 task registers and the env launches headless with finite rewards on the copied A3 ping-pong URDF asset.

Not done:

- This Codex shell has not independently run Isaac because the required GPU/Isaac environment is not active here.
- Self-collision is disabled in the Isaac config due to overlapping wrist/racket collision meshes; a cleaner Isaac collision asset is still needed before re-enabling it.
- Sim parity between MuJoCo and Isaac is not established.
- Hardware SDK parity for joint order and command/state layout is not established.

## Risks

- Differences in actuator model, contact, timestep, or joint order can break sim-to-sim transfer.
- URDF import can lose inertial or collision fidelity.
- Racket mount errors directly corrupt planner-to-WBC training.

## Next Steps

1. Use `hope_training/config/joint_order_agibot_a3.yaml` as the current working joint-order source and verify it against the real SDK before deployment.
2. Load the same standing pose in MuJoCo and Isaac and compare FK.
3. Clean or replace Isaac collision geometry so self-collision can be revisited.
4. Document MuJoCo/Isaac parity metrics before policy transfer work.
