# HOPE: Hitch Open Ping-Pong Embodied AI Challenge

HOPE is an open platform for humanoid robot table tennis, developed by [Hitch Interactive](https://hitchinteractive.com) (Intelligent Racing Inc.) in collaboration with the [ROAR Platform](https://roar.berkeley.edu) at UC Berkeley. The challenge invites teams to deploy whole-body humanoid controllers that can rally a ping-pong ball against human opponents or other robots, using off-the-shelf humanoid hardware and an open-source perception and planning stack.

This repository contains the **reference design documents**, early implementation code, and support materials for the HOPE system architecture, covering motion capture setup, model-based racket planning, reinforcement learning training, simulation, and deployment.

For developer and agent navigation, start with [docs/START_HERE.md](docs/START_HERE.md). It indexes the current project gates, folder roles, asset policy, and update rules.

## About This Branch (`codex/integrate-support-materials`)

This branch turns the HOPE repo from a set of public reference-design documents into a **working, gate-driven reimplementation harness**. Its implementation target is the **Agibot Expedition A3** humanoid (31 actuated DOF), reproducing the HITTER planner + whole-body-control (WBC) stack on the actual A3 hardware, ChingMu/VRPN motion capture, MuJoCo, and Isaac.

On top of `main`, it adds:

1. **A documentation harness under [docs/](docs/)** — a hub rooted at [docs/START_HERE.md](docs/START_HERE.md): gated milestones ([docs/gates/](docs/gates/) G00–G08), per-task how-tos ([docs/operations/](docs/operations/)), interface contracts ([docs/interfaces/](docs/interfaces/)), plus [PROJECT_MAP](docs/PROJECT_MAP.md), [DEFINITIONS](docs/DEFINITIONS.md), [PROGRESS](docs/PROGRESS.md), [ASSET_POLICY](docs/ASSET_POLICY.md), and the [AGENTS.md](AGENTS.md) agent rules.
2. **Integrated A3 support materials** under [agi/](agi/): the A3 ping-pong URDF, a MuJoCo/AimRT sim example, and a tracked subset of the Agibot deployment package (the ~1.7 GB full payload is git-ignored — see *Assumptions & Limitations*).
3. **A WBC training scaffold** under [hope_training/whole_body_tracking/](hope_training/whole_body_tracking/): a new A3 ping-pong task (`HOPE-PingPong-AgibotA3-v0`), a Hydra config tree, the `setup_train_env.sh` launcher, richer Weights & Biases telemetry, reference-coupled racket target sampling, and the reward-shaping `strike_success=0` fix.
4. **External-reference auto-sync** ([scripts/sync_external_repos.sh](scripts/sync_external_repos.sh)) and an expanded `.gitignore` so heavy/vendor artifacts stay out of git.

**Status:** this is an in-progress reimplementation. The full sim-to-real loop is **not** complete — every gate is `Partial` or `Not started` (see the live table in [docs/START_HERE.md](docs/START_HERE.md)). Treat the A3 training pipeline as *demonstrated end-to-end (pipeline viability)*, not as an accepted quality baseline. New contributors should read [docs/START_HERE.md](docs/START_HERE.md) and the **Quickstart** below.

## Documents

| Document | Description | Version |
|----------|-------------|---------|
| [Developer and Agent Start Here](docs/START_HERE.md) | Current repo map, gate status, documentation update rules, and implementation entry points | living |
| [Motion Capture System Reference Setup](mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md) | OptiTrack/ROS 2 arena configuration, coordinate frames, tracked object taxonomy, humanoid base_link marker setup, ball tracking, and streaming pipeline | v0.3 |
| [7DOF Racket Model-based Planner Reference Setup](HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md) | Ball state estimation, trajectory prediction, and racket target planning (Stages 1–3 of the HITTER framework), reimplemented in the HOPE canonical frame | v0.1 |
| [WBC Simulation Training Reference Setup](HOPE_WBC_Simulation_Training_Reference_Setup.md) | SMPL-X motion acquisition, GMR retargeting, BeyondMimic RL training pipeline for whole-body control (Stage 4), with dual-backend support for Isaac Lab and mjlab | v0.5 |
| [Hardware Deployment Reference Setup](HOPE_Hardware_Deployment_Reference_Setup.md) | Real-robot deployment via `legged_control2` (G1) or AimRT (A3): ONNX inference, ROS 2 node graph, PD gain tuning, safety procedures, and competition workflow | v0.1 |

Each document contains a **Section 0 prologue** listing all implementation differences from the original HITTER paper (Su et al., arXiv:2508.21043v2).

## System Architecture

```
                    ┌─────────────────────────────┐
                    │     OptiTrack Cameras        │
                    │     (9×, 360 Hz)             │
                    └──────────┬──────────────────┘
                               │ NatNet
                               ▼
                    ┌─────────────────────────────┐
                    │  motion_capture_tracking     │
                    │  (ROS 2 Jazzy)               │
                    │                              │
                    │  Publishes:                  │
                    │   • Ball position (3D)       │
                    │   • P1/P2 base_link pose     │
                    │   • Table origin frame       │
                    └───┬──────────────┬──────────┘
                        │              │
                        ▼              ▼
              ┌──────────────┐  ┌──────────────────┐
              │  HOPE Planner │  │  Whole-Body       │
              │  (Stages 1-3) │  │  Controller       │
              │               │  │  (Stage 4)        │
              │  Ball state   │  │                    │
              │  estimation   │  │  BeyondMimic RL    │
              │  → trajectory │  │  policy (50 Hz)    │
              │  prediction   │──│                    │
              │  → racket     │  │  Receives:         │
              │  target       │  │   • Racket target  │
              │  planning     │  │   • base_link pose │
              └──────────────┘  │   • Joint encoders  │
                                │                    │
                                │  Outputs:          │
                                │   • 29-DOF joint   │
                                │     position cmds  │
                                └────────┬───────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  Humanoid Robot   │
                                │  (Unitree G1 /    │
                                │   Agibot A3)      │
                                │                   │
                                │  PD controller    │
                                │  → joint torques  │
                                └──────────────────┘
```

> The diagram above shows the **general** HOPE challenge architecture. *This branch* implements the Agibot A3 path with **ChingMu/VRPN** motion capture (the `vrpn_mocap` ROS 2 package), not only the OptiTrack/NatNet path. See *About This Branch*.

## Repository Layout

| Path | Role |
| --- | --- |
| [docs/](docs/) | Project harness: gates, interfaces, operations, asset policy — start at [docs/START_HERE.md](docs/START_HERE.md) |
| [reimplement.md](reimplement.md) | Long-form, machine-specific historical runbook; use only when a gate or operation doc cites a specific step |
| [hope_ws/](hope_ws/) | ROS 2 (Jazzy) integration workspace: `hope_bringup`, `hope_msgs`, `hope_planner`, `vrpn_mocap` |
| [hope_training/](hope_training/) | Isaac Lab / BeyondMimic WBC training (`whole_body_tracking`) plus the GMR + GVHMR motion tooling |
| [agi/](agi/) | Agibot A3 support: URDF (`agi/URDF/`), MuJoCo sim (`agi/A3_MuJoCo_Sim/`), deploy example (`agi/code_deployment/`) |
| [mocap/](mocap/) | Motion-capture setup and coordinate reference |
| [scripts/](scripts/) | Repo maintenance (`sync_external_repos.sh`) |
| [calib_bags/](calib_bags/), [calib_csv/](calib_csv/) | Small curated calibration recordings and processed CSVs |
| `vendor_assets/`, `external_repos/` | Git-ignored: heavy vendor payloads and auto-synced reference repos (absent on a fresh clone) |

The four `HOPE_*_Reference_Setup.md` documents and the `HOPE_AI_Challenge_2026_Rules_*.docx` files (see *Documents* above) are the original public competition/design references; they describe the broader system, not this branch's implementation specifics.

## Quickstart: Build & Run

This repo has **no single environment** — work happens in three scopes (full matrix: [docs/operations/setup_environments.md](docs/operations/setup_environments.md)). A new contributor, new computer, or agent should start from [docs/START_HERE.md](docs/START_HERE.md), then follow the relevant operation doc. Use [reimplement.md](reimplement.md) only as the long-form runbook when a gate or operation doc points to a specific step.

For a training machine, the practical index is:

1. [docs/START_HERE.md](docs/START_HERE.md) for the current gate map and task entry points.
2. [docs/operations/run_training.md](docs/operations/run_training.md) for Isaac/BeyondMimic setup, smoke tests, and training commands.
3. [docs/operations/setup_local_sync.md](docs/operations/setup_local_sync.md) for ignored/private assets missing from a fresh clone.

**1. ROS 2 workspace** (planner, mocap, bringup) — ROS 2 Jazzy; see [docs/operations/build_and_test.md](docs/operations/build_and_test.md):

```bash
cd hope_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Use the ROS environment described in [docs/operations/setup_environments.md](docs/operations/setup_environments.md); the old root Dockerfile has been removed because the project no longer uses that Docker path.

**2. WBC training** (Isaac Lab + BeyondMimic) — needs a GPU plus a pre-provisioned Isaac Sim 4.5.0 / Isaac Lab 2.1.0 (Python 3.10); see [docs/operations/run_training.md](docs/operations/run_training.md):

```bash
cd hope_training/whole_body_tracking
source setup_train_env.sh          # put machine paths/IDs in setup_train_env.local.sh
# local training-loop smoke test (no WandB account needed; requires a local .npz):
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke \
  motion_file=../motions/hope_forehand.npz
```

**3. Motion retargeting** (optional — to make your own reference clips) — GVHMR (video→SMPL-X) → GMR (SMPL-X→A3) → `scripts/csv_to_npz.py`; see [docs/operations/setup_local_sync.md](docs/operations/setup_local_sync.md).

> The first real run needs assets that are **not** in a fresh clone (the A3 Isaac asset, reference motions, vendor payloads). See *Assumptions & Limitations* and [docs/operations/setup_local_sync.md](docs/operations/setup_local_sync.md).

## Assumptions & Limitations

- **The repo is not self-contained.** A fresh `git clone` does **not** include the git-ignored `vendor_assets/`, `external_repos/`, the `hope_training/GMR` and `hope_training/GVHMR` clones, reference motions, or any `*.onnx/*.pt/*.ckpt` binaries. Restore them per [docs/operations/setup_local_sync.md](docs/operations/setup_local_sync.md).
- **Isaac is assumed pre-provisioned.** This branch does not document a from-scratch Isaac Sim / Isaac Lab install; `setup_train_env.sh` expects an existing install (target versions: Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10). Use the [official Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). This is the single biggest reproducibility gap.
- **The A3 Isaac asset is generated from tracked source** [agi/URDF/A3T2.5-URDF-std-pingpang/](agi/URDF/A3T2.5-URDF-std-pingpang/) into `assets/agibot_a3/urdf/model.urdf` with `package://` mesh paths rewritten; use [docs/operations/run_training.md](docs/operations/run_training.md) and `scripts/prepare_a3_isaac_asset.py` as the current procedure. `reimplement.md` Step 12.7 is historical background.
- **A3 = 31 actuated DOF** for training; the A3 **deploy** policy I/O is **29** (the 2 neck joints `head_yaw`/`head_pitch` are excluded from the policy and driven at fixed kp=40/kd=2). The "29-DOF" figure in the architecture diagram is the original HITTER **G1** number.
- **Only the Isaac Lab + URDF backend is runnable today.** The mjlab / MuJoCo-Warp A3 backend referenced in the design docs is not yet wired up in this branch.
- **WandB is the internal default for full training** (reference clips load from a WandB registry); `logger=tensorboard motion_file=<local.npz>` covers local training-loop smoke tests without an account. Use your **own** WandB team (`WANDB_ENTITY`) and org (`WANDB_REGISTRY_ORG`) — they must differ (see [docs/operations/run_training.md](docs/operations/run_training.md)).
- **Some inputs are private / vendor-gated** and cannot be redistributed: the full Agibot A3 deploy payload (`vendor_assets/agibot/.../a3_deploy_example_full`, ~1.7 GB), the maintainer's WandB motion registry, and possibly the TTRL reference repo. External users must produce their own equivalents ([docs/operations/setup_local_sync.md](docs/operations/setup_local_sync.md)).
- **Hardware safety:** do not send real joint commands until the joint-order, command-scaling, and safe-halt checks in [docs/gates/G07_mujoco_to_real.md](docs/gates/G07_mujoco_to_real.md) and [AGENTS.md](AGENTS.md) pass.
- **Local environment names** (`distrobox` boxes `hope` / `grasping`, working tree at `~/workspace/HOPE`) are the maintainer's concrete examples; substitute your own.

## Key Design Decisions

**Racket tracking is prohibited.** The motion capture system tracks exactly three categories of objects: the ping-pong table origin frame (PPT), each humanoid's `base_link` (P1, P2), and the ball. No reflective markers may be placed on the racket, the robot's hand, or the wrist link. Each robot must infer its paddle's 6-DOF pose through forward kinematics from its own `base_link` + joint encoders. This is a deliberate competition constraint that tests autonomous paddle control through the robot's internal body model.

**Multi-platform support.** The reference design supports both the Unitree G1 (via Isaac Lab + PhysX with USD assets) and the Agibot Expedition A3 (via mjlab + MuJoCo Warp with MJCF assets). Both backends share the same BeyondMimic MDP formulation and export to ONNX for deployment.

**Open-source training stack.** The WBC training pipeline is built entirely on open-source code: [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) (MIT license) for motion tracking RL, [GMR](https://github.com/YanjieZe/GMR) (MIT license) for SMPL-X to robot retargeting, and [GVHMR](https://github.com/zju3dv/GVHMR) for monocular video-to-SMPL-X extraction. The HITTER paper's trained weights are not released; all training starts from scratch.

## Supported Robots

| Robot | DOF | Simulation Backend | Model Format | Role in this branch |
|-------|-----|-------------------|--------------|---------------------|
| Agibot Expedition A3 | 31 actuated (29 in deploy policy) | Isaac Lab + PhysX (from URDF) | URDF → USD | **Active implementation target.** `TrackingFlat` + `HOPEPingPong` train and export ONNX end-to-end (pipeline viability — see [G05](docs/gates/G05_isaac_training_first_loop.md)). mjlab / MuJoCo-Warp backend planned, not yet wired. |
| Unitree G1 / G1 EDU | 29 (+ hands, unused) | Isaac Lab + PhysX | USD | Original HITTER reference platform; upstream BeyondMimic baseline retained for reference |

## Coordinate Frame Convention

All three documents share a common world frame (ROS 2 REP 103):

- **Origin**: Near-side left corner of the table surface
- **X**: Toward opponent (along the 2.74 m table length)
- **Y**: Left (along the 1.525 m table width)
- **Z**: Up
- **Table surface height**: 0.76 m above floor

The OptiTrack system must be configured with **Up Axis → Z** in Motive to match this convention.

## Prerequisites

The reference documents assume familiarity with:

- ROS 2 Jazzy
- Python 3.10+
- NVIDIA Isaac Lab 2.1.0 or mjlab
- OptiTrack Motive (or compatible motion capture system)
- PyTorch and Weights & Biases (WandB)

## Related Repositories

| Repository | Purpose |
|-----------|---------|
| [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | BeyondMimic training code (Isaac Lab) |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | BeyondMimic training code (MuJoCo Warp) |
| [HybridRobotics/motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) | ROS 2 deployment (ONNX inference) |
| [qiayuanl/legged_control2](https://qiayuanl.github.io/legged_control2_doc/) | Low-level controller framework for legged robots |
| [qiayuanl/unitree_bringup](https://github.com/qiayuanl/unitree_bringup) | Unitree robot bringup utilities |
| [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | Unitree official mjlab integration |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | General Motion Retargeting (SMPL-X → robot) |
| [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | Video-to-SMPL-X pose estimation |
| [IMRCLab/motion_capture_tracking](https://github.com/IMRCLab/motion_capture_tracking) | ROS 2 motion capture bridge |
| [purdue-tracelab/TTRL-ICRA2026](https://github.com/purdue-tracelab/TTRL-ICRA2026) | Table-tennis RL reference, auto-synced into `external_repos/` by `scripts/sync_external_repos.sh` (ignored, unpinned) |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | GPU-accelerated MuJoCo |
| [AimRT/aimrt](https://github.com/AimRT/aimrt) | Agibot's lightweight robotics middleware |

## References

- Su, Z., Zhang, B., Rahmanian, N., Gao, Y., Liao, Q., Regan, C., Sreenath, K., & Sastry, S. S. (2025). HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning. *arXiv:2508.21043v2*. [Project page](https://humanoid-table-tennis.github.io/)
- Liao, Q., et al. (2025). BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion. *arXiv:2508.08241v4*. [Project page](https://beyondmimic.github.io/)
- Araújo, J. P., Ze, Y., Xu, P., Wu, J., & Liu, C. K. (2025). Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking. *arXiv:2510.02252*.
- Ze, Y., et al. (2025). LATENT: Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data. *arXiv:2603.12686*.
- mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning. *arXiv:2601.22074*.
- Peng, X. B., et al. (2024). SMPLOlympics: Sports Environments for Physically Simulated Humanoids. *arXiv:2407.00187*.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE). See [LICENSE](LICENSE) for the full terms.

The reference documents in this repository describe system architectures and training pipelines that depend on third-party open-source software. Each dependency carries its own license (MIT or Apache 2.0). See the Related Repositories table above for links to each project and its respective license.

## Contact

**Allen Yang**, Co-founder and CTO, Hitch Interactive (Intelligent Racing Inc.); Chair, AI Racing ROAR Platform, UC Berkeley; Founding Executive Director, VIVE AR Center, UC Berkeley

### HITTER Authors (UC Berkeley)

The HOPE reference design is adapted from the HITTER framework. The original HITTER authors are:

Zhi Su, Boren Zhang, Navid Rahmanian, Yuchen Gao, Qiayuan Liao, Colin Regan, Koushil Sreenath, S. Shankar Sastry

Hybrid Robotics Group and ROAR Platform, University of California, Berkeley
