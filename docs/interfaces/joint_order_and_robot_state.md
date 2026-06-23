# Joint Order And Robot State

Status: Working source exists; hardware verification pending

## Goal

Maintain one accepted joint and robot-state contract across:

- A3 URDF
- MuJoCo model
- Isaac asset
- training config
- exported policy
- deployment runtime
- hardware SDK messages

## Current Working Joint Order

The current working 31-DOF A3 order is stored in:

- `hope_training/config/joint_order_agibot_a3.yaml`

It is used as the current training/export alignment source. Before real hardware commands, verify it against:

1. A3 SDK command message order.
2. A3 runtime joint-state order.
3. ONNX metadata/export order.
4. MuJoCo and Isaac articulation order mappings.

## Required Hardware Verification Table

Fill this table before real commands:

| Index | Joint name | YAML source | SDK command field | State field | Limits source | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | See YAML | `hope_training/config/joint_order_agibot_a3.yaml` | TBD | TBD | A3 URDF/MJCF/SDK | Verify before deployment |

## Robot State Contract

The deployment and policy runtime must agree on:

- joint positions
- joint velocities
- previous action
- base angular velocity
- projected gravity
- base pose or base forward vector
- policy clock or time-to-strike fields

See [policy_observation_action.md](policy_observation_action.md).

## Current Materials

- A3 URDF materials: `agi/URDF/`
- A3 deploy source: `agi/code_deployment/a3_deploy_example`
- RobotIOBackend guide: `agi/code_deployment/RobotIOBackend 架构与策略适配指南.md`
- Isaac robot config and PD/action-scale source: `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py`

## Blocking Rule

Do not run real joint commands until this document is filled from verified model and SDK sources.
