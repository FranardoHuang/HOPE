# Build And Test

Status: Draft

## Task Setup

For package-local Python tests, no ROS environment or ignored local asset is required.

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
colcon build --symlink-install
source install/setup.bash
```

Current known limitation:

- `colcon` is not installed in the current macOS shell used for this documentation pass.

## Deploy Source Build

Use the Agibot deploy docs:

- `agi/code_deployment/A3 deploy example.md`
- `agi/code_deployment/a3_deploy_example/README.md`

Record successful build commands in G07 before hardware use.
