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

Build and verify inside the ROS environment:

- `colcon` and `ros2` come from `ros-jazzy-desktop-full` and are only available inside the ROS 2 Jazzy box. Run `colcon build --symlink-install` and `source install/setup.bash` there; do not attempt ROS workspace builds from the host terminal.

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

`setup_train_env.sh` is the source of truth for the `hope_isaac_py` launcher and the WandB exports, and must be sourced in every new terminal. On the current shared RunPod it exports:

- `HOPE_ISAAC_VENV=/workspace/hope_isaac_venv`
- `ISAACLAB_PATH=/workspace/IsaacLab`
- `HOPE_WBT_PYTHONPATH=<repo>/hope_training/whole_body_tracking/source/whole_body_tracking`
- `WANDB_ENTITY=BerkeleyPingPong`
- `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org`
- `WANDB_PROJECT=hope_wbc`
- `WANDB_MOTION_PROJECT=csv_to_npz`
- `WANDB_DIR=/workspace/yikang/nohope/hope_training/wandb`

The legacy `/workspace/isaacsim/python.sh`, `/opt/drone_venv`, and `hope-motion-py310` paths are not used for Isaac training on this machine. If another machine differs, update `setup_train_env.sh` and this operation doc together.

Quick check:

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## Local Assets

Use [setup_local_sync.md](setup_local_sync.md) for ignored assets such as:

- `vendor_assets/agibot/a3_deploy_example_full/`
- `external_repos/TTRL-ICRA2026/`

These ignored paths are not fully restored by `git clone` or `git pull`. If a command depends on vendor payloads, copy them manually before running that gate. If a task uses TTRL as a reference, run `scripts/sync_external_repos.sh` first.

## Creating the environments

The sections above describe how to *enter* the environments. This section covers how to *create* them on a fresh machine. `hope` and `grasping` are the maintainer's example distrobox names — substitute your own local names throughout.

### ROS box

Linux + ROS 2 Jazzy. Two ways to build it:

- distrobox: the full `distrobox-create` recipe lives in `reimplement.md`.
- container: build from the reference recipe [Dockerfile.hope-ros2-jazzy](../../Dockerfile.hope-ros2-jazzy), which is based on `docker.io/osrf/ros:jazzy-desktop-full` and now includes `ros-jazzy-vrpn-mocap` and `python3-vcstool`.

Apt packages to install in the box:

- `build-essential`, `cmake`, `curl`, `git`
- `ros-jazzy-desktop-full`
- `ros-jazzy-vrpn-mocap`
- `python3-colcon-common-extensions`, `python3-pip`, `python3-rosdep`, `python3-vcstool`, `python3-venv`

Then build the workspace:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### GPU / Isaac (grasping) box

This is currently the biggest reproducibility gap. The box is assumed to be **pre-provisioned** with:

- Isaac Sim 4.5.0
- Isaac Lab 2.1.0
- Python 3.10
- an NVIDIA CUDA GPU

A from-scratch Isaac install is **not yet documented here** — follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to provision the GPU box, then point `setup_train_env.sh` at it.

`source setup_train_env.sh` provides the `hope_isaac_py` launcher and the `WANDB_*` exports. On the current shared RunPod this launcher uses `/workspace/hope_isaac_venv` plus `/workspace/IsaacLab/isaaclab.sh`; see [GPU / Isaac Environment](#gpu--isaac-environment) above. `hydra` and `omegaconf` must be importable in that Isaac Lab Python. Sanity check:

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

### Motion retargeting (GMR + GVHMR)

The motion pipeline uses two separate conda envs (`python=3.10`). Both are git-ignored clones and are absent on a fresh clone. The full procedure is in `reimplement.md` steps 9-11.

- On the current shared RunPod with RTX 5090 / Blackwell (`sm_120`), the working motion env is **independent from Isaac Lab**:

  ```bash
  source /workspace/yikang/miniforge3/etc/profile.d/conda.sh
  conda activate hope-motion-py310
  ```

  Do not reuse `/workspace/hope_isaac_venv` for GVHMR/GMR. The env uses PyTorch `2.7.0+cu128` and a source-built `pytorch3d 0.7.9`; GVHMR's upstream `cu121` pins are not used on this GPU.

- **GMR** ([YanjieZe/GMR](https://github.com/YanjieZe/GMR.git), local pin `bb1bbe4`): `pip install -e .`. Needs license-gated SMPL-X body models (`SMPLX_NEUTRAL/MALE/FEMALE.pkl` from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de)).
- **GVHMR** ([zju3dv/GVHMR](https://github.com/zju3dv/GVHMR.git), local pin `6ec3ca3`):

  ```bash
  conda create -y -n gvhmr python=3.10
  conda activate gvhmr
  pip install -r requirements.txt
  pip install -e .
  ```

  Optional DPVO needs CUDA 12.1; the `cu121` pins fix `torch==2.3.0` and `pytorch3d 0.7.6 py310_cu121_pyt230`. Blackwell (sm_120) GPUs are incompatible with the `cu121` pins — see `hope_training/GVHMR/.hope-motion-py310-freeze-before-blackwell-fix.txt`. GVHMR also needs license-gated checkpoints into `inputs/checkpoints/`.

  Current RunPod note: the five non-body GVHMR checkpoint files were restored under `hope_training/GVHMR/inputs/checkpoints/` from a public Hugging Face mirror after the upstream Google Drive folder hit quota, and the local license-gated body-model files were sufficient for the completed forehand/backhand MP4 -> motion run. New machines must still restore SMPL/SMPL-X manually because those files are not redistributable.

  Required before `tools/demo/demo.py` on torch>=2.6 envs (this pod's hope-motion-py310 uses torch 2.7.0+cu128): `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. ultralytics 8.2.42 loads the YOLO checkpoint with a plain `torch.load`, which fails under the new `weights_only=True` default (`UnpicklingError: ... DetectionModel was not an allowed global`). The checkpoints are local and hash-checked, so opting out is safe. Verified 2026-07-02: a fresh shell without the variable fails at `[Preprocess]`; with it, the full mp4 -> csv -> npz rerun reproduces the existing artifacts bit-for-bit.

## Update Rule

If an environment name, path, setup script, Python launcher, or required environment variable changes, update the relevant operation doc first, then update this reference matrix.
