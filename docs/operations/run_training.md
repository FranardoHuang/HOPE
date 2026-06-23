# Run Training

Status: Draft

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

Latest `jiayi` records that `TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied A3 URDF asset, including wandb logging, checkpoint save, and ONNX export. This proves the pipeline can run, but it is not yet an accepted quality baseline.

`origin/jiayi` through `42489cd` adds:

- `setup_train_env.sh` as the training shell setup source of truth.
- richer live metrics in WandB from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.

## Entry Files

- `hope_training/whole_body_tracking/README.md`
- `hope_training/whole_body_tracking/scripts/train.py`
- `hope_training/whole_body_tracking/scripts/play.py`
- `hope_training/whole_body_tracking/cfg/train.yaml`
- `hope_training/whole_body_tracking/cfg/play.yaml`
- `hope_training/whole_body_tracking/setup_train_env.sh`

## Environment Setup

Run inside the GPU/Isaac environment, not the ROS environment:

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

The script must be sourced, not executed. It defines `hope_isaac_py`, sets `HOPE_WBT_PYTHONPATH`, and exports the WandB team/org/project variables.

Quick sanity check:

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## Local Assets Needed For This Task

Before smoke tests or training, the A3 Isaac asset must exist at the path expected by `robots/agibot_a3.py`:

```text
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

If it is missing, create it from the tracked Agibot URDF package:

```bash
cd ~/workspace/HOPE
mkdir -p hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/{urdf,meshes,config}
cp -r agi/URDF/A3T2.5-URDF-std-pingpang/meshes/. \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/meshes/
cp -r agi/URDF/A3T2.5-URDF-std-pingpang/config/. \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/config/
cp agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

Then rewrite mesh references in the copied URDF if they still use `package://...` paths. The long-form version of this setup is in `reimplement.md` Step 12.7.

Motion references are also task-local setup. Use one of:

- WandB registry paths such as `$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand`.
- Local ignored `.npz` motion files under `hope_training/motions/` or `hope_training/whole_body_tracking/artifacts/`, with the exact path recorded in G05.

Do not commit generated logs, checkpoints, WandB caches, or motion artifacts unless the asset policy changes.

## Smoke Test

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke
```

Success means the env builds, PPO prints learning iterations, and rewards remain finite.

## Baseline Training Commands

Plain tracking first:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  run_name=forehand_tracking
```

HOPE racket task, one policy per swing:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  run_name=hope_forehand

hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand" \
  run_name=hope_backhand
```

Useful overrides:

```bash
num_envs=4096 max_iterations=20000 seed=1
task.rewards.racket_position_weight=5.0
task.domain_rand.pd_gain_range=null
```

## Evaluate And Export

`play.py` exports `policy.onnx` next to the checkpoint.

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="artifacts/hope_forehand:v0/motion.npz" \
  headless=false
```

Headless video:

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="artifacts/hope_forehand:v0/motion.npz" \
  headless=true video=true
```

From a WandB run:

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  wandb_path="$WANDB_ENTITY/hope_wbc/<RUN_ID>" headless=false
```

## First-Loop Rule

Before setting a baseline quality target, record:

1. Isaac asset path.
2. Environment start command, including `source setup_train_env.sh`.
3. Random rollout result.
4. First training command.
5. WandB run ID, checkpoint path, and ONNX export path.
6. Failure mode or first metric.
7. Whether the run is only pipeline viability or an accepted quality baseline.

Write the result to G05 and [../PROGRESS.md](../PROGRESS.md).
