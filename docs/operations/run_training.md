# Run Training

Status: Draft

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

`TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied Agibot A3 URDF asset (31 actuated DOF), including WandB logging, checkpoint save, and ONNX export. This proves the pipeline can run; it is NOT an accepted quality baseline. G04/G05 remain Partial, and G06/G07 are not accepted until sim-to-sim and dry-run deployment gates record verification.

This branch adds:

- a scrubbed `setup_train_env.sh` as the training shell setup source of truth (site paths are now overridable env vars).
- source-first `HOPE_WBT_PYTHONPATH` ordering, so local `whole_body_tracking` edits beat stale installed copies.
- richer live `Live/...` telemetry in WandB/TensorBoard from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.
- `HOPE-TableTennis-AgibotA3-v0`, a first-pass Isaac Lab table/net/ball/A3 physics scene for G04 visualization and future G08 returner/spin experiments, now with a tracked Purdue PACE USD table/net visual overlay and Purdue-style table/ball contact materials.
- updated `HOPEPingPong` target/reward defaults with conditional exact-strike metrics, success-gated reference perturbation, debug reward logging hooks, and slow-to-fast velocity staging through `ref_vel_scale`.

## Entry Files

- `hope_training/whole_body_tracking/README.md`
- `hope_training/whole_body_tracking/scripts/train.py`
- `hope_training/whole_body_tracking/scripts/play.py`
- `hope_training/whole_body_tracking/scripts/play_table_tennis.py`
- `hope_training/whole_body_tracking/scripts/probe_metric.py`
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

### Blackwell (RTX 50-series, sm_120) torch fix

Isaac Sim 4.5.0 ships `torch 2.5.1+cu124`, which has **no sm_120 kernels**. On a Blackwell GPU
(e.g. RTX 5090) `import torch` "works" and `torch.cuda.is_available()` is `True`, but any real CUDA op
fails with `no kernel image is available for execution on the device`, so training crashes after Kit
startup. Fix (verified 2026-06-26 on RTX 5090):

```bash
# inside the Isaac env (hope-isaac-py310)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4    # Isaac Sim 4.5 requires numpy<2; the torch upgrade may pull numpy 2.x
```

`isaaclab*` carry a `torch==2.5.1` pin in metadata but are editable installs imported at runtime, so the
upgrade does not break them. Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124
--index-url https://download.pytorch.org/whl/cu124`. Verify with a real kernel, not just `is_available()`:

```bash
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item())"
```

### EULA

The first Kit launch needs the NVIDIA Omniverse EULA. Accept it non-interactively for headless runs:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### No-WandB training (local motion override)

`scripts/train.py` can load the motion clip from a local `.npz` instead of the WandB registry. Pass
`motion_file=<path>` (mirrors `scripts/play.py`); when set it skips WandB entirely, so use it with
`logger=tensorboard` for an account-free run. If you have no motion data at all, generate a placeholder
"stand at default pose" clip (pipeline proof only, not a real swing):

```bash
hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
  --output_file ../motions/a3_stand.npz --frames 600 --fps 50

hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=1024 max_iterations=60 algo.runner.save_interval=25 \
  logger=tensorboard run_name=stand_bootstrap \
  motion_file=$(pwd)/../motions/a3_stand.npz
```

`hydra`, `omegaconf`, and `rsl_rl` are NOT in the package `setup.py` `install_requires`; they must be importable from the Isaac Lab Python (provide via Isaac Lab itself or `HOPE_ISAAC_VENV_SITE`). Install the package into that Python:

```bash
hope_isaac_py -m pip install -e source/whole_body_tracking
```

Quick sanity check (expect `hydra` 1.3.2):

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## Fresh Machine Entry

For a new computer, start from `docs/START_HERE.md`, then use this operation doc for Isaac training setup and [setup_local_sync.md](setup_local_sync.md) for ignored/private assets. Use [../../reimplement.md](../../reimplement.md) only as the long-form runbook when a gate or operation doc points at a specific step, such as the A3 URDF copy in Step 12.7 or the motion pipeline in steps 9-11.

Minimum order for a training machine:

1. Install Isaac Sim/Lab and set `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` in `setup_train_env.local.sh`.
2. Source `hope_training/whole_body_tracking/setup_train_env.sh`.
3. Restore or create the A3 Isaac URDF asset and motion references listed below.
4. Run the smoke commands in this doc before starting a long training run.

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

## Table-Tennis Physics Scene Smoke Test

This is a G04 scene/physics check, not the accepted G05 WBC baseline. It loads
`HOPE-TableTennis-AgibotA3-v0`, serves a ball from the P2 half toward the P1-side A3, and verifies the
table/net/ball frame plus drag/Magnus hooks.

```bash
hope_isaac_py scripts/play_table_tennis.py
hope_isaac_py scripts/play_table_tennis.py --num_envs 9
hope_isaac_py scripts/play_table_tennis.py --fix_base
hope_isaac_py scripts/play_table_tennis.py --enable_aero
hope_isaac_py scripts/play_table_tennis.py --magnus 0.1
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
```

Expected behavior: by default the ball arcs under PhysX gravity/contact only, bounces on the table, and travels toward the robot. With `--enable_aero` or `--magnus`, the HOPE drag/Magnus callback also affects flight. With `--fix_base`, the pelvis stays pinned for a stable visualization. Without it, the robot may drift or fall because no balance/return policy exists yet.

Default table-tennis scene behavior follows Purdue PACE parity: the ball uses PhysX gravity plus contacts with ball mass `3.4 g`, ball/table restitution/friction `0.9/0.1` and `0.95/0.4`, and aero drag off. Pass `--enable_aero` to use the HOPE-calibrated drag callback; `--magnus` also enables aero and adds spin.

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

Resume / curriculum hand-off (added on `train_1`): `checkpoint_path=<model.pt>` loads weights + optimizer
from a prior run and CONTINUES training (the iteration counter resumes). Use it to apply a staged config
change — e.g. tightening `racket_velocity_std` — without throwing away progress:

```bash
checkpoint_path=logs/rsl_rl/agibot_a3_hope/<run>/model_2000.pt
```

`HOPEPingPong` trains ONE swing style per policy (forehand or backhand), chosen entirely by the `registry_name` reference clip.

At startup, `scripts/train.py` prints:

- the imported `whole_body_tracking` path,
- the composed env-cfg source file,
- every applied task override from `cfg/task/<name>.yaml`,
- the post-override racket reward stds and target-sampling knobs.

If a YAML key targets a missing env-cfg attribute, training now raises instead of silently ignoring the
override. Treat these printed lines as part of the G05 verification record.

### ppo.yaml deltas on this branch

- `max_iterations: 300000000000` is a train-FOREVER sentinel. Always pass `max_iterations=` on the CLI and stop manually when `strike_success` plateaus.
- `save_interval` 500 -> 100.
- `entropy_coef` 0.005 -> 0.004.

### Racket Target Sampling

`HOPEPingPong.yaml` now defaults to `racket.target_mode: reference_perturbed`. In this mode the command term computes the reference motion's racket position, velocity, and face normal at the strike frame using the same FK path as the actual racket, then samples a curriculum-scaled perturbation around that reachable reference state:

```yaml
target_mode: reference_perturbed
ref_perturb_pos: [0.06, 0.08, 0.06]
ref_perturb_vel: [0.3, 0.3, 0.25]
ref_perturb_normal: 0.15
ref_perturb_curriculum_steps: 30000
ref_perturb_curriculum_start: 0.05
ref_perturb_advance_threshold: 0.25
ref_perturb_advance_rate: 1.0e-5
ref_vel_scale: 0.6
debug_reward_logging: false
```

This is the current first-loop default because the old independent uniform box could sample targets the imitated swing never reaches, keeping `strike_success` at 0 even with reward shaping.

The command cfg defaults keep `ref_perturb_success_gated: true`, `exact_success_decay: 0.99`, `exact_success_min_count: 50.0`, `strike_success_vel_thresh: 0.5`, and `strike_success_normal_thresh_deg: 15.0` even when those keys are not present in the YAML. `ref_vel_scale: 0.6` deliberately trains a slower 3-4 m/s racket target first; raise it toward `1.0` after the slow strike clears the exact-success gates.

In `reference_perturbed` mode, `base_target` is coupled to the racket target:
`base_target_xy = racket_target_xy - reference_base_to_racket_xy`, then the YAML
`base_target_x_range`/`base_target_y_range` add only small jitter (`[-0.10, 0.10]`). This avoids a base
reward that fights arm reachability.

The legacy uniform ranges (`pos_x [0.25,0.55]`, `pos_y [-0.45,0.45]`, `pos_z [0.70,1.15]`, `vel_x [1.5,4.0]`, ...) are still present but ignored unless `target_mode: uniform`. Treat them as PLACEHOLDER until validated against A3 right-arm IK reachability.

## Live Training Telemetry

`MotionOnPolicyRunner` (`utils/my_on_policy_runner.py`) logs a `Live/...` dashboard to WandB/TensorBoard every PPO iteration. Namespaces:

- `Live/<command_term>/<metric>` — per-axis command tracking (reference vs robot anchor pos/vel per x/y/z, joint error mean/max, `motion_phase`, racket pos/vel/normal per axis, `time_to_strike_s`, `pre_strike_flag`, `strike_window_flag`, `racket_speed`, ...).
- `Live/Reward/<term>` — per-reward-term contributions.
- `Live/Termination/*`, `Live/Action/*`, `Live/Env/*`.

The real "is it learning to hit" signal is the exact-strike metric group:
`strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`,
`strike_composite_success_exact`, and `exact_strike_sample_count_decayed`. These are conditional
sample-weighted pass rates at the exact strike frame. `strike_success` and the `*_at_strike` metrics are
still useful, but episode-wide errors are diluted by the long non-strike phase, so do not judge progress
from them.

## Reward Shaping (strike_success=0 fix)

The reward kernel is `exp(-||err||^2 / std^2)`. With `std` set to the final acceptance tolerance, the reward is ~0 for any early error (a 50 cm error gives `exp(-44) ~ 0`), so there is no gradient and `strike_success` stays stuck at 0. The target-sampling fix above handles unreachable targets; the reward shaping here handles too-narrow early rewards.

The current `HOPEPingPong.yaml` PATH B/C values try to keep useful gradients in the observed error band while preventing the motion imitation reward from dominating long episodes:

- `racket_position_weight: 8.0`, `racket_position_std: 0.15`
- `racket_velocity_weight: 4.0`, `racket_velocity_std: 1.2`
- `racket_normal_weight: 3.0`, `racket_normal_std: 0.30`
- `base_position_weight: 4.0`, `base_position_std: 0.25`
- `motion_scale: 0.30` over the six inherited motion-imitation rewards
- regularization tightened to `joint_torques_weight: -0.00003`, `action_rate_weight: -0.15`, `joint_limit_weight: -15.0`, and `undesired_contacts_weight: -0.2`

These stds are DECOUPLED from acceptance thresholds: the position metric still reports true success only
below `strike_success_pos_thresh = 0.075 m`, velocity below `0.5 m/s`, and racket-normal error below
`15 deg`.

Optional later precision pass: once the exact-strike pass rates are non-trivial, raise `ref_vel_scale` toward full clip speed, then tighten the stds back toward `0.075 / 0.5 / 0.262` and resume from the checkpoint.

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
