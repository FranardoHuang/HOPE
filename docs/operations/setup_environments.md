# Setup Environments

Status: Draft

## Purpose

This is a reference matrix, not a required first stop for every task. Most work should start from the relevant gate doc and operation doc. Use this file when you are unsure which execution scope a task belongs to.

This project uses different execution scopes. Do not mix them casually; many failures come from running a command in the wrong environment.

There is no single project environment. Setup is a matrix:

| Gate / Work | Environment | Main Setup Doc |
| --- | --- | --- |
| G00 materials and repo harness | Host terminal | this file, [setup_local_sync.md](setup_local_sync.md) |
| G01 real preparation | ROS environment plus host hardware/network setup | this file, [run_mocap.md](run_mocap.md), [run_deploy_dryrun.md](run_deploy_dryrun.md) |
| G02 data acquisition | ROS environment plus mocap vendor software | [run_mocap.md](run_mocap.md) |
| G03 data processing and planner calibration | ROS environment or package-local Python for tests/tools | [build_and_test.md](build_and_test.md), [run_planner.md](run_planner.md) |
| G04 MuJoCo/Isaac model work | GPU/Isaac environment for Isaac; deploy/MuJoCo target environment for Agibot sim | this file, [setup_local_sync.md](setup_local_sync.md) |
| G05 Isaac training | GPU/Isaac environment | [run_training.md](run_training.md) |
| G06 Isaac-to-MuJoCo parity | GPU/Isaac plus MuJoCo/deploy target environment | [run_training.md](run_training.md), [run_deploy_dryrun.md](run_deploy_dryrun.md) |
| G07 MuJoCo-to-real | Target deployment environment plus robot network | [run_deploy_dryrun.md](run_deploy_dryrun.md), [setup_local_sync.md](setup_local_sync.md) |
| G08 blind-spot improvements | Depends on chosen track | relevant gate mini-spec |

## Host Terminal

Use the host terminal for:

- git operations
- file organization
- local asset sync under `vendor_assets/`
- entering distroboxes or containers

The repo is assumed to be visible at the same path from the host and project environments. Existing long-form commands in `reimplement.md` use `~/workspace/HOPE` as the shared path.

## ROS Environment

Use the ROS environment for:

- `ros2`
- `colcon`
- `rosdep`
- mocap
- planner
- bridge/dry-run commands that are ROS-native

Typical entry:

```bash
distrobox enter hope
cd ~/workspace/HOPE/hope_ws
colcon build --symlink-install
source install/setup.bash
```

Current Codex shell limitation:

- `colcon` is not installed in the current macOS shell, so ROS workspace build verification must happen inside the ROS environment.

## GPU / Isaac Environment

Use the GPU/Isaac environment for:

- Isaac Sim / Isaac Lab
- BeyondMimic training
- motion replay
- policy evaluation
- ONNX export from training/eval

Typical entry:

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

`setup_train_env.sh` is the source of truth for:

- `HOPE_WBT_PYTHONPATH`
- `hope_isaac_py`
- `WANDB_ENTITY`
- `WANDB_REGISTRY_ORG`
- `WANDB_PROJECT`

It must be sourced in every new terminal.

Quick check:

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## Local Assets

Use [setup_local_sync.md](setup_local_sync.md) for ignored assets such as:

- `vendor_assets/agibot/a3_deploy_example_full/`
- `external_repos/TTRL-ICRA2026/`

These ignored paths are not fully restored by `git clone` or `git pull`. If a command depends on vendor payloads, copy them manually before running that gate. If a task uses TTRL as a reference, run `scripts/sync_external_repos.sh` first.

## Update Rule

If an environment name, path, setup script, Python launcher, or required environment variable changes, update the relevant operation doc first, then update this reference matrix.
