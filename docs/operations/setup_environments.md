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
- entering distroboxes or the already-provisioned project environments

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
- table-tennis physics/visualization scene checks (`scripts/play_table_tennis.py`)
- policy evaluation
- ONNX export from training/eval

Typical entry:

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

`setup_train_env.sh` is the source of truth for the `hope_isaac_py` launcher and the WandB exports, and must be sourced in every new terminal. The script reads overridable env vars with placeholder defaults, auto-sources an optional git-ignored local override, and auto-detects the current `/workspace/...` Isaac/IsaacLab layout when present:

- `HOPE_ISAAC_PYTHON` — path to the Isaac Lab Python interpreter (`hope_isaac_py` wraps it)
- `HOPE_ISAACLAB_ROOT` — Isaac Lab install root
- `HOPE_ISAAC_VENV_SITE` — extra `site-packages` to inject (e.g. so `hydra`/`omegaconf` import in the Isaac Lab Python; they are not in the package `install_requires`)
- `WANDB_ENTITY` — your run-logging team (`your-wandb-team`)
- `WANDB_REGISTRY_ORG` — your motion-registry org (`your-wandb-org`); it must differ from `WANDB_ENTITY` or registry reads fail with "Unable to find organization for entity ..."
- `WANDB_PROJECT`
- `setup_train_env.local.sh` — git-ignored local override, auto-sourced if present; put your real machine paths and WandB identities here instead of editing the tracked script

Replace the placeholders with your own values (never commit a private WandB identity to this public branch).

On the current shared RunPod (paths verified 2026-07-02) the sourced environment resolves to:

- Isaac venv `/workspace/hope_isaac_venv` with the Isaac Lab checkout at `/workspace/IsaacLab`
- `HOPE_WBT_PYTHONPATH=<repo>/hope_training/whole_body_tracking/source/whole_body_tracking` (working-tree source first)
- `WANDB_ENTITY=BerkeleyPingPong`
- `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org`
- `WANDB_PROJECT=hope_wbc`
- `WANDB_MOTION_PROJECT=csv_to_npz`
- `WANDB_DIR=/workspace/yikang/nohope/hope_training/wandb`

The legacy `/workspace/isaacsim/python.sh`, `/opt/drone_venv`, and `hope-motion-py310` paths are not used for Isaac training on this machine. If another machine differs, set the machine-specific values in the git-ignored `setup_train_env.local.sh`, and update this operation doc if the shared layout changes.

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

The sections above describe how to *enter* the environments. This section covers how to *create* them on a fresh machine. This operation doc is the current setup entry; the old long-form runbook uses the same distrobox names, `hope` for ROS/Jazzy work and `grasping` for GPU/Isaac work.

### Host container prerequisites

On an Ubuntu 22.04 host checked on 2026-06-25, `distrobox` was installed in `~/.local/bin` and the NVIDIA driver was healthy (`RTX 5090`, driver `570.153.02`, CUDA `12.8` reported by `nvidia-smi`). Creating the `hope` distrobox initially failed before container startup because the host did not have a working container backend with rootless UID/GID mapping helpers:

```text
failed to find dependency getsubids
newuidmap ... I/O error
```

The `osrf/ros:jazzy-desktop-full` image was pulled into the local `lilipod` store, but no `hope` container was created until host sudo was available. Install a normal backend and mapping helpers with host sudo before rerunning the distrobox creation command:

```bash
sudo apt update
sudo apt install -y podman uidmap
```

If the packaged distrobox is preferred over the user-local script, also install `distrobox`. Keep the container names `hope` and `grasping` so current docs and old command snippets stay aligned.

Current local status: `podman 3.4.4` and `uidmap` are installed, `hope` was created from
`docker.io/osrf/ros:jazzy-desktop-full`, and `grasping` was created with `--nvidia` from
`docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`.

### ROS box

Linux + ROS 2 Jazzy. The supported path is a distrobox or otherwise pre-provisioned ROS 2 Jazzy environment. The old root `Dockerfile.hope-ros2-jazzy` has been removed because this repo no longer uses that Docker path.

- distrobox: use this doc for the supported package list and ROS commands; if you need the exact historical `distrobox-create` invocation, see `reimplement.md`.

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

The reference path assumes this box is pre-provisioned with:

- Isaac Sim 4.5.0
- Isaac Lab 2.1.0
- Python 3.10
- an NVIDIA CUDA GPU

A local bringup on 2026-06-25 used an ignored conda env and source checkout instead of a system
`/opt/isaacsim` install:

- `grasping`: `docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` with `--nvidia`; verified RTX 5090 visibility and `nvcc`.
- `hope-isaac-py310`: conda Python `3.10`, `isaacsim-rl 4.5.0.0`, `isaacsim-app 4.5.0.0`, editable Isaac Lab source packages, and editable `whole_body_tracking`. **On the Blackwell RTX 5090 the bundled `torch 2.5.1+cu124` was replaced with `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` + `numpy==1.26.4`** — 2.5.1 has no sm_120 kernels and crashes with `no kernel image is available for execution on the device`. See [run_training.md](run_training.md#blackwell-rtx-50-series-sm_120-torch-fix).
- `external_repos/IsaacLab`: ignored clone at tag `v2.1.0` / commit `21f7136`.
- `hope_training/whole_body_tracking/setup_train_env.local.sh`: ignored local override pointing `HOPE_ISAAC_PYTHON` at `/home/agiuser/miniconda3/envs/hope-isaac-py310/bin/python` and `HOPE_ISAACLAB_ROOT` at the ignored Isaac Lab checkout.

First Isaac/Kit launch prompts for the NVIDIA Omniverse EULA. Do not set `OMNI_KIT_ACCEPT_EULA=YES`
unless the user has explicitly accepted that EULA.

A from-scratch Isaac install is otherwise **not fully documented here** — follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to provision the GPU box, then point `setup_train_env.sh` at it.

`source setup_train_env.sh` provides the `hope_isaac_py` launcher and the `WANDB_*` exports. The scrubbed script reads the overridable `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` / `HOPE_ISAAC_VENV_SITE` env vars (or your git-ignored `setup_train_env.local.sh`) — see [GPU / Isaac Environment](#gpu--isaac-environment) above; on the current shared RunPod the launcher resolves to `/workspace/hope_isaac_venv` plus `/workspace/IsaacLab/isaaclab.sh`. `hydra` and `omegaconf` are not in the package `install_requires`; they must be importable in the Isaac Lab Python (inject them via `HOPE_ISAAC_VENV_SITE` if needed). Sanity check:

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

### Motion retargeting (GMR + GVHMR)

The motion pipeline uses separate Python/conda environments for GVHMR/GMR and Isaac preprocessing. The clones and generated motions are git-ignored and absent on a fresh clone. Use [setup_local_sync.md](setup_local_sync.md) for current restore points; `reimplement.md` steps 9-12 provide the supplemental command detail through local `.npz` generation and replay.

Current local status from 2026-06-25:

- Miniconda exists at `/home/agiuser/miniconda3`.
- `hope-motion-py310` exists and uses Python `3.10.20`.
- `hope_training/GMR` is restored at commit `bb1bbe4`; editable install into `hope-motion-py310` imports successfully with `general_motion_retargeting`, `mujoco 3.10.0`, `smplx`, and `torch 2.12.1+cu130`.
- `hope_training/GVHMR` is restored at commit `6ec3ca3`, but its `requirements.txt` pins `torch==2.3.0+cu121` and a `pytorch3d` CUDA 12.1 wheel. Do not blindly install those pins on the RTX 5090 / Blackwell host; resolve the CUDA compatibility issue first.
- License-gated SMPL-X body models and GVHMR checkpoints were not present in this checkout.

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
