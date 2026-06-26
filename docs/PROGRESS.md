# Progress Log

Use this file for short project-state updates that future humans and agents need to see. Keep detailed reasoning in the relevant gate doc.

## 2026-06-26

- Ran the first end-to-end Isaac WBC training loop on this machine (G05 first-loop now reproduced here, not just inherited from `reimplement.md`). Task `TrackingFlat`, `num_envs=1024`, 60 iterations, `logger=tensorboard`, `run_name=stand_bootstrap`. Mean reward improved monotonically `-4.08 -> -0.24`. Artifacts: `logs/rsl_rl/agibot_a3_flat/2026-06-26_13-13-07_stand_bootstrap/{model_0,25,50,59}.pt` and `exported/policy.onnx`. These are pipeline-viability artifacts, NOT an accepted quality baseline (the reference is a static stand, not a real swing).
- Fixed the real G05 training blocker: the **RTX 5090 is Blackwell (sm_120)** and Isaac Sim 4.5.0's bundled `torch 2.5.1+cu124` has no sm_120 kernels (a real CUDA matmul failed with `no kernel image is available for execution on the device`; this is the deeper blocker the EULA note in G05 hid). Upgraded `hope-isaac-py310` to `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` (sm_120 kernels) and pinned `numpy==1.26.4` (Isaac Sim 4.5 needs `numpy<2`). After the fix Isaac Kit boots on the 5090 and a CUDA matmul succeeds. Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`.
- Accepted the NVIDIA Omniverse EULA non-interactively via `OMNI_KIT_ACCEPT_EULA=YES` (the remaining gate from the 2026-06-25 G05 note).
- Made training runnable without a WandB account/registry: `scripts/train.py` previously fetched the motion clip ONLY from the WandB registry (`wandb.Api().artifact(...)`). Added a local `motion_file=` override (mirrors `scripts/play.py`) that skips WandB when set, plus `motion_file: null` in `cfg/train.yaml`. `registry_name` is only consumed by the runner under `logger=wandb`, so the local/tensorboard path needs no WandB identity.
- Added `scripts/make_static_motion.py`: generates a valid BeyondMimic motion `.npz` (a static "stand at default pose" A3 clip) without the GMR/GVHMR retargeting pipeline or WandB, so the WBC loop can run with no motion data. Output `hope_training/motions/a3_stand.npz` (`fps=50`, `joint_pos[600,31]`, `body_pos_w[600,32,3]`, zero velocities). Schema matches `scripts/csv_to_npz.py`. This is a placeholder reference for first-loop/pipeline proof only.
- Created the `hope-motion-py310` conda env (`python=3.10`) and accepted the conda default-channel ToS; GMR/GVHMR clones from 2026-06-25 remain in place.

## 2026-06-25

- Removed the obsolete root `Dockerfile.hope-ros2-jazzy` and updated README, setup/build operations, G00, and `reimplement.md` so ROS work points to the distrobox or otherwise pre-provisioned ROS 2 Jazzy environment instead of the old Docker path.
- Continued environment setup using the `reimplement.md` names `hope` and `grasping`: installed user-local `distrobox 1.8.2.5` and `lilipod v0.0.3`, created `~/workspace/HOPE -> ~/workspace/nohope`, pulled the `osrf/ros:jazzy-desktop-full` image, and confirmed the host GPU is visible (`RTX 5090`, driver `570.153.02`, CUDA `12.8`). Creating the `hope` distrobox is blocked until host `podman uidmap` or equivalent rootless mapping helpers are installed with sudo.
- Unblocked rootless containers after local sudo access was provided: installed host `podman` and `uidmap`, created the `hope` distrobox from `docker.io/osrf/ros:jazzy-desktop-full`, verified `ROS_DISTRO=jazzy` and `colcon`, created the NVIDIA-enabled `grasping` distrobox from `docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, and verified it sees the RTX 5090 plus `nvcc`.
- Built an ignored local Isaac training Python env `hope-isaac-py310` with `torch 2.5.1+cu124`, `isaacsim-rl 4.5.0.0`, `isaacsim-app 4.5.0.0`, editable Isaac Lab source from `external_repos/IsaacLab` tag `v2.1.0` (`21f7136`), editable `whole_body_tracking`, and `setup_train_env.local.sh` pointing `hope_isaac_py` at that env. Python checks pass from inside `grasping`; the first Kit/training launch is paused until the NVIDIA Omniverse EULA is explicitly accepted.
- Restored ignored motion repos: `hope_training/GMR` at `bb1bbe4` and `hope_training/GVHMR` at `6ec3ca3`. Installed GMR editable into `hope-motion-py310` and verified imports (`torch 2.12.1+cu130`, `mujoco 3.10.0`); GVHMR was cloned but not installed because its requirements pin CUDA 12.1-era wheels that need a Blackwell compatibility pass.
- Checked the G05 training environment in this harness. Restored the ignored package-local A3 Isaac asset from tracked Agibot URDF materials, verified `86` mesh references with `0` missing, and passed the host table-tennis geometry test (`6/6`; torch aerodynamics skipped because host torch is unavailable).
- Clarified the new-computer/agent startup index in `README.md` and `docs/START_HERE.md`: start from `docs/START_HERE.md`, use `docs/operations/run_training.md` for Isaac training setup, use `docs/operations/setup_local_sync.md` for ignored/private assets, and treat `reimplement.md` as a long-form runbook only when referenced by a gate or operation doc.
- Integrated the latest `origin/train_1` updates on `main` while keeping the main-branch docs structure: training code/config now follows the PATH B/C slower-hit defaults (`ref_vel_scale: 0.6`, narrower reference perturbations, updated reward stds/weights), table-tennis physics adopts Purdue PACE material defaults and a tracked visual USD overlay, and the docs now point new training machines through `docs/START_HERE.md` -> `docs/operations/run_training.md` / `setup_local_sync.md` rather than using `reimplement.md` as the primary entry point.
- Merged `origin/train_1` into `main` and updated the docs/reimplementation rhythm so `reimplement.md` is gate-indexed rather than a separate phase plan. G04 now records the new `HOPE-TableTennis-AgibotA3-v0` Isaac scene, and G05 records the updated `HOPEPingPong` exact-strike metrics, reward/target defaults, source-provenance logging, and success-gated reference perturbation.
- Preserved source/config/test changes while excluding generated or local artifacts (`.codex-tmp/`, `.vscode/`, generated `*.onnx`) under the asset policy. Scrubbed private training paths and WandB identities from `reimplement.md`; machine-specific values belong in `setup_train_env.local.sh`.

## 2026-06-24

- Integrated `origin/jiayi` commit `c951d9d` into this harness branch while preserving the open-source placeholders and A3 support-material docs. The branch now includes `reference_perturbed` racket target sampling for `HOPEPingPong`, training override plumbing for the new parameters, and Avatar-Pro relay `ball_tracking_mode` support (`rigid_body` preferred, `auto` fallback).

## 2026-06-23

- Open-source documentation pass for this branch: rewrote `README.md` (About-this-branch, Repository Layout, Quickstart, Assumptions & Limitations; corrected A3 31/29-DOF and ChingMu/VRPN framing), indexed `reimplement.md` from `START_HERE`, and filled env-setup gaps across `docs/operations/*` (environment creation, GMR/GVHMR + SMPL-X/checkpoint restore, WandB team-vs-org + offline `logger=tensorboard`, the `Live/...` telemetry and the reward-std `strike_success=0` fix) and `docs/interfaces/*` (31-DOF list, 31-vs-29 deploy, real actor/critic observations + action dims, world-Z frame, RacketCommand fields).
- Scrubbed maintainer-private values for public release: `setup_train_env.sh` now reads overridable `HOPE_ISAAC_*` paths and placeholder WandB identity (real values via a git-ignored `setup_train_env.local.sh`); added `ros-jazzy-vrpn-mocap` + `python3-vcstool` to `Dockerfile.hope-ros2-jazzy`.
- Integrated A3 Isaac/BeyondMimic training updates (through commit `42489cd`): working joint-order YAML, updated A3 robot config, deploy-transcribed PD/action-scale values, HOPEPingPong task updates, train/play fixes, `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Added `scripts/sync_external_repos.sh` so TTRL remains an auto-synced ignored reference instead of a pinned dependency; docs now require recording the TTRL source commit when material is extracted from it.

## 2026-06-22

- Added HITTER paper under `papers/2508.21043v2.pdf`.
- Added ChingMu VRPN ROS 2 package under `hope_ws/src/vrpn_mocap` and supplied a minimal ROS 2 `package.xml`.
- Added tracked source/config subset of Agibot `a3_deploy_example` under `agi/code_deployment/a3_deploy_example`.
- Moved the complete Agibot deploy payload to ignored local storage under `vendor_assets/agibot/a3_deploy_example_full`.
- Added TTRL as an ignored local reference under `external_repos/TTRL-ICRA2026`.
- Removed root `tmp/` after moving useful materials.
- Verified `hope_planner` package tests from `hope_ws/src/hope_planner`: 20 passed.
- Could not run ROS workspace build in the current shell because `colcon` is not installed.
