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
- A3 GMR/controller and runtime articulation orders now have distinct content-bound tables and an
  explicit bijection. The legacy `joint_order_agibot_a3.yaml` mirrors only GMR/controller order;
  see `docs/interfaces/joint_order_and_robot_state.md` before labeling runtime columns.
- `avatar_pro_vrpn.yaml` currently defaults the robot input rigid-body labels to `ppp2`/`ppp3` after one observed rig, while the relayed HOPE topics remain `/P1/pose` and `/P2/pose`; live G01 verification must record the actual CMTracker labels before relying on them.
- Mocap play-time contract confirmed with the team (2026-07): ChingMu over VRPN streams the robot base (pelvis) pose plus the ball position at 300 Hz during play; the relay publishes the ball as position-only `PointStamped` (spin measurement is planned for the physics-modeling phase).
- `colcon build --packages-up-to hope_planner hope_wbc_runner` verified inside the `hope` distrobox; the x86_64 deploy package builds via `agi/a3_deploy_example/scripts/build_a3_deploy_pkg.sh`.
- Hardware SDK/runtime state layout verified: the `pp_joint_map` backend order was checked slot-for-slot against AGI `robot_io::MakeA3Layout31()` — a checked bijection (`agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md:137-139`) — with joint limits clamped from the MJCF.
- A3 safe dry-run executed: staged PASSIVE→PD_STAND→SHADOW→MOTION bring-up on the MDU preceded the first real joint commands (2026-07-02 sim-to-real run).

Not done:

- Live VRPN stream into the deploy chain: the rig streams during play, but the mocap link is not yet bridged into the deploy front-end (targets were scripted for the 2026-07-02 run).

## Risks

- Frame mismatch can invalidate downstream planner and training data.
- Joint order mismatch can make a good policy unsafe on real hardware.
- Mocap timestamps and QoS can dominate reaction-time error.

## Next Steps

1. Run VRPN against ChingMu, record exact topic names, and bridge the mocap stream into the deploy chain.
2. Fill [../interfaces/frames_and_coordinates.md](../interfaces/frames_and_coordinates.md) with measured transforms.
