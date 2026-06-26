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

### Explicit 31-DOF Order

The 31 actuated DOF, in the exact order of [`joint_order_agibot_a3.yaml`](../../hope_training/config/joint_order_agibot_a3.yaml) (every name is suffixed `_joint`):

| Index | Joint name |
| --- | --- |
| 0 | `waist_yaw_joint` |
| 1 | `waist_roll_joint` |
| 2 | `waist_pitch_joint` |
| 3 | `head_yaw_joint` |
| 4 | `head_pitch_joint` |
| 5 | `left_shoulder_pitch_joint` |
| 6 | `left_shoulder_roll_joint` |
| 7 | `left_shoulder_yaw_joint` |
| 8 | `left_elbow_joint` |
| 9 | `left_wrist_roll_joint` |
| 10 | `left_wrist_pitch_joint` |
| 11 | `left_wrist_yaw_joint` |
| 12 | `right_shoulder_pitch_joint` |
| 13 | `right_shoulder_roll_joint` |
| 14 | `right_shoulder_yaw_joint` |
| 15 | `right_elbow_joint` |
| 16 | `right_wrist_roll_joint` |
| 17 | `right_wrist_pitch_joint` |
| 18 | `right_wrist_yaw_joint` |
| 19 | `left_hip_pitch_joint` |
| 20 | `left_hip_roll_joint` |
| 21 | `left_hip_yaw_joint` |
| 22 | `left_knee_joint` |
| 23 | `left_ankle_pitch_joint` |
| 24 | `left_ankle_roll_joint` |
| 25 | `right_hip_pitch_joint` |
| 26 | `right_hip_roll_joint` |
| 27 | `right_hip_yaw_joint` |
| 28 | `right_knee_joint` |
| 29 | `right_ankle_pitch_joint` |
| 30 | `right_ankle_roll_joint` |

This order originates from the `<joint>` hinge order in the GMR `a3_mocap.xml` (the order GMR writes `dof_pos`, which training, ONNX export, and the SDK must all share). It is confirmed byte-for-order identical to `AGIBOT_A3_JOINT_NAMES` in [`robots/agibot_a3.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py). VERIFY against the real A3 SDK (`ros2 topic echo /joint_states --once`) before deployment.

### 31-DOF Training vs 29-DOF Deploy

Training emits 31 actions (one per actuated DOF above). The A3 deploy policy I/O view is 29: the 2 neck joints (`head_yaw_joint`, `head_pitch_joint`, indices 3 and 4) are excluded from the deploy policy I/O and driven separately at deploy `kp=40` / `kd=2` via `ExpandToBackend`. The remaining 29 joints keep their relative order.

- Per-joint `action_scale = 0.25 * effort_limit / stiffness`, exactly matching the deploy `a3_action_scale`.
- Decoder target = `action * action_scale + default_angle`.
- PD gains in [`robots/agibot_a3.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py) (`a3_kps` / `a3_kds` / `a3_default_angles`) are the official Agibot values from `a3_policy_parameters.hpp`.

## Required Hardware Verification Table

Fill this table before real commands:

The YAML source order is confirmed (training/export, identical to `AGIBOT_A3_JOINT_NAMES`). The SDK command field, `/joint_states` state field, and limits columns remain hardware-TBD until verified on a real A3.

| Index | Joint name | YAML source | SDK command field | State field | Limits source | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `waist_yaw_joint` | confirmed (`joint_order_agibot_a3.yaml` row 0) | TBD | TBD | A3 URDF/MJCF/SDK | Verify before deployment |
| 1 | `waist_roll_joint` | confirmed (row 1) | TBD | TBD | A3 URDF/MJCF/SDK | Verify before deployment |
| 2 | `waist_pitch_joint` | confirmed (row 2) | TBD | TBD | A3 URDF/MJCF/SDK | Verify before deployment |
| 3 | `head_yaw_joint` | confirmed (row 3) | TBD (deploy-excluded; kp=40/kd=2) | TBD | A3 URDF/MJCF/SDK | Not in 29-DOF deploy I/O |
| 4 | `head_pitch_joint` | confirmed (row 4) | TBD (deploy-excluded; kp=40/kd=2) | TBD | A3 URDF/MJCF/SDK | Not in 29-DOF deploy I/O |
| 5-30 | See [explicit 31-DOF order](#explicit-31-dof-order) | confirmed (rows 5-30) | TBD | TBD | A3 URDF/MJCF/SDK | Verify before deployment |

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
