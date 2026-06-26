# G01 Real Preparation

Status: Partial

## Goal

Prepare the real-world interfaces before relying on simulation or training results.

This gate proves that the project can observe the arena and robot state in a consistent coordinate system, and that the A3 deployment path has a safe dry-run plan.

## Inputs

- Motion-capture setup docs under `mocap/`
- `hope_ws/src/vrpn_mocap`
- `hope_ws/src/hope_bringup`
- A3 URDF and deployment materials under `agi/`
- Full local A3 deploy assets under `vendor_assets/agibot/a3_deploy_example_full`

## Outputs

- Verified mocap stream.
- Verified `world` frame convention.
- Verified robot `base_link` interpretation.
- A3 joint order and state layout document.
- Safe deploy dry-run checklist.

## Related Directories

- `mocap/`
- `hope_ws/src/vrpn_mocap`
- `hope_ws/src/hope_bringup`
- `agi/URDF/`
- `agi/code_deployment/`
- `docs/interfaces/`
- `docs/operations/`

## Operation Docs

- [../operations/run_mocap.md](../operations/run_mocap.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/build_and_test.md](../operations/build_and_test.md)

## Acceptance Criteria

- `ros2 topic list` shows expected VRPN topics.
- Ball and robot base pose are in the HOPE `world` frame or have a documented transform into it.
- A3 `base_link`, joint names, joint order, and command/state layout are documented.
- A3 dry-run can start backend/sync without publishing unsafe commands.
- Emergency stop and safe halt behavior are documented before any real command test.

## Current State

Done:

- ChingMu VRPN source package is present.
- Existing `hope_bringup` launch/config already expects `vrpn_mocap`.
- Motion-capture reference docs define frame conventions and ChingMu conversion notes.
- A3 deploy docs and source support are present.
- Current working A3 joint order exists at `hope_training/config/joint_order_agibot_a3.yaml`.

Not done:

- Live VRPN stream has not been verified in this repo.
- `colcon build` for the ROS workspace has not been verified in the intended environment.
- A3 joint order has a working training/export source, but hardware SDK/runtime state layout has not been verified.
- A3 safe dry-run has not been executed.

## Risks

- Frame mismatch can invalidate downstream planner and training data.
- Joint order mismatch can make a good policy unsafe on real hardware.
- Mocap timestamps and QoS can dominate reaction-time error.

## Next Steps

1. Build `hope_ws` inside the ROS environment.
2. Run VRPN against ChingMu and record exact topic names.
3. Fill [../interfaces/frames_and_coordinates.md](../interfaces/frames_and_coordinates.md) with measured transforms.
4. Fill [../interfaces/joint_order_and_robot_state.md](../interfaces/joint_order_and_robot_state.md) from A3 SDK/model/runtime messages.
