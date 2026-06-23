# Build And Test

Status: Draft

## Task Setup

For package-local Python tests, no ROS environment or ignored local asset is required. The planner pytest needs a Python with `rclpy`, `numpy`, and the `hope_planner` deps available (e.g. a sourced ROS 2 Jazzy environment).

For ROS workspace build, use the ROS environment:

```bash
distrobox enter hope
cd ~/workspace/HOPE/hope_ws
```

No `vendor_assets/` payload is required for planner tests or ROS package discovery.

## Planner Unit Tests

Run from the package directory:

```bash
cd hope_ws/src/hope_planner
python3 -m pytest test
```

Current known result:

- 2026-06-22: 20 passed.

Running the same command from the repo root currently fails unless `hope_planner` is on `PYTHONPATH`.

## ROS Workspace Build

Run inside the intended ROS environment:

```bash
cd hope_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The `rosdep install` step resolves `vrpn_mocap` VRPN/eigen dependencies and must run before `colcon build`.

Current status:

- Target environment is Linux + ROS 2 Jazzy. The build is to be verified inside the ROS environment (see [Dockerfile.hope-ros2-jazzy](../../Dockerfile.hope-ros2-jazzy)).

## Deploy Source Build

The tracked deploy source subset and its docs live under [agi/code_deployment/a3_deploy_example/](../../agi/code_deployment/a3_deploy_example/) (see also [agi/code_deployment/A3 deploy example.md](../../agi/code_deployment/A3%20deploy%20example.md)). Note that the full ~1.7 GB private Agibot handoff (ONNX/RKNN/TensorRT/sysroots/.deb) is vendor-gated and not redistributed here.

Full deploy build commands are pending hardware (gate G07): they have not been run/verified yet and must be recorded in G07 before hardware use. This section is intentionally not yet populated with build steps rather than silently empty.
