# Run Training

Status: Draft

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

`TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied Agibot A3 URDF asset (31 actuated DOF), including WandB logging, checkpoint save, and ONNX export. This proves the pipeline can run; it is NOT an accepted quality baseline. Every gate is still Partial/Not started.

This branch adds:

- a scrubbed `setup_train_env.sh` as the training shell setup source of truth (site paths are now overridable env vars).
- richer live `Live/...` telemetry in WandB/TensorBoard from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.

## Entry Files

- `hope_training/whole_body_tracking/README.md`
- `hope_training/whole_body_tracking/scripts/train.py`
- `hope_training/whole_body_tracking/scripts/play.py`
- `hope_training/whole_body_tracking/cfg/train.yaml`
- `hope_training/whole_body_tracking/cfg/play.yaml`
- `hope_training/whole_body_tracking/setup_train_env.sh`

## Environment Setup

This runs in the GPU/Isaac environment (Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10, CUDA GPU), not the ROS environment. `grasping` is the maintainer's EXAMPLE distrobox name — substitute your own box.

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

The script must be SOURCED (not executed) in every new GPU/Isaac terminal. It defines the `hope_isaac_py` launcher, sets `HOPE_WBT_PYTHONPATH`, and exports the WandB variables.

The script is now scrubbed of site-specific paths. It reads overridable env vars with placeholder defaults:

- `HOPE_ISAAC_PYTHON` — the Isaac Lab Python interpreter `hope_isaac_py` wraps.
- `HOPE_ISAACLAB_ROOT` — your Isaac Lab checkout.
- `HOPE_ISAAC_VENV_SITE` — optional extra `site-packages` to inject (e.g. to provide `hydra`/`omegaconf`).

Set these for your machine in a git-ignored `setup_train_env.local.sh` next to the script; `setup_train_env.sh` auto-sources it if present.

A from-scratch Isaac Sim 4.5.0 / Isaac Lab 2.1.0 / Python 3.10 install is NOT documented here and is the single biggest reproducibility gap. Follow the official [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) first, then point `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` at it.

`hydra`, `omegaconf`, and `rsl_rl` are NOT in the package `setup.py` `install_requires`; they must be importable from the Isaac Lab Python (provide via Isaac Lab itself or `HOPE_ISAAC_VENV_SITE`). Install the package into that Python:

```bash
hope_isaac_py -m pip install -e source/whole_body_tracking
```

Quick sanity check (expect `hydra` 1.3.2):

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## WandB Setup

WandB needs two DISTINCT identities. They MUST differ or motion-registry reads fail with `Unable to find organization for entity ...`.

- `WANDB_ENTITY` — your team, used for run logging.
- `WANDB_REGISTRY_ORG` — your org, used for the motion registry.
- `WANDB_PROJECT=hope_wbc` — the run project.

Replace any placeholder with your own; never commit a private identity. This branch ships `your-wandb-team` / `your-wandb-org` as placeholders.

```bash
wandb login
export WANDB_ENTITY=your-wandb-team
export WANDB_REGISTRY_ORG=your-wandb-org
export WANDB_PROJECT=hope_wbc
```

No WandB account? Pass `logger=tensorboard` for smoke tests and local runs; it is the no-WandB fallback and needs no login or registry.

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

`TrackingFlat` needs no motion registry and no WandB account, so it is the cleanest smoke test:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
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
```

`HOPEPingPong` trains ONE swing style per policy (forehand or backhand), chosen entirely by the `registry_name` reference clip.

### ppo.yaml deltas on this branch

- `max_iterations: 300000000000` is a train-FOREVER sentinel. Always pass `max_iterations=` on the CLI and stop manually when `strike_success` plateaus.
- `save_interval` 500 -> 100.
- `entropy_coef` 0.005 -> 0.004.

### Racket Target Sampling

`HOPEPingPong.yaml` now defaults to `racket.target_mode: reference_perturbed`. In this mode the command term computes the reference motion's racket position, velocity, and face normal at the strike frame using the same FK path as the actual racket, then samples a curriculum-scaled perturbation around that reachable reference state:

```yaml
target_mode: reference_perturbed
ref_perturb_pos: [0.15, 0.20, 0.15]
ref_perturb_vel: [1.0, 1.0, 0.8]
ref_perturb_normal: 0.30
ref_perturb_curriculum_steps: 30000
ref_perturb_curriculum_start: 0.15
```

This is the current first-loop default because the old independent uniform box could sample targets the imitated swing never reaches, keeping `strike_success` at 0 even with reward shaping.

The legacy uniform ranges (`pos_x [0.25,0.55]`, `pos_y [-0.45,0.45]`, `pos_z [0.70,1.15]`, `vel_x [1.5,4.0]`, ...) are still present but ignored unless `target_mode: uniform`. Treat them as PLACEHOLDER until validated against A3 right-arm IK reachability.

## Live Training Telemetry

`MotionOnPolicyRunner` (`utils/my_on_policy_runner.py`) logs a `Live/...` dashboard to WandB/TensorBoard every PPO iteration. Namespaces:

- `Live/<command_term>/<metric>` — per-axis command tracking (reference vs robot anchor pos/vel per x/y/z, joint error mean/max, `motion_phase`, racket pos/vel/normal per axis, `time_to_strike_s`, `pre_strike_flag`, `strike_window_flag`, `racket_speed`, ...).
- `Live/Reward/<term>` — per-reward-term contributions.
- `Live/Termination/*`, `Live/Action/*`, `Live/Env/*`.

The real "is it learning to hit" signal is `strike_success` (fraction of strikes with racket position error < `strike_success_pos_thresh` = 0.075 m) and the `*_at_strike` metrics. Episode-wide errors are DILUTED by the long non-strike phase, so do not judge progress from them.

## Reward Shaping (strike_success=0 fix)

The reward kernel is `exp(-||err||^2 / std^2)`. With `std` set to the final acceptance tolerance, the reward is ~0 for any early error (a 50 cm error gives `exp(-44) ~ 0`), so there is no gradient and `strike_success` stays stuck at 0. The target-sampling fix above handles unreachable targets; the reward shaping here handles too-narrow early rewards.

This branch widens the stds in `HOPEPingPong.yaml` so the reward gives a gradient from tens of cm out:

- `racket_position_std` 0.075 -> 0.35
- `racket_velocity_std` 0.5 -> 1.2
- `racket_normal_std` 0.262 -> 0.5

These stds are DECOUPLED from `strike_success_pos_thresh` = 0.075: the acceptance metric still reports true success only below 7.5 cm.

Optional later precision pass: once `strike_success` is non-trivial, tighten the stds back toward `0.075 / 0.5 / 0.262` and resume from the checkpoint.

## Evaluate And Export

`play.py` exports the policy to `<checkpoint_dir>/exported/policy.onnx`.

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
