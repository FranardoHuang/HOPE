# Run Training

Status: Draft

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

The current shared RunPod has verified local two-clip `HOPEPingPong` smoke and a registry-backed WandB pipeline smoke on the copied Agibot A3 URDF asset (31 actuated DOF). The registry smoke run `6xus13ga` finished at https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga, but the `hope_forehand:v4` / `hope_backhand:v4` motion artifacts used by that smoke were later verified to face world +Y rather than HOPE +X. Treat that run as pipeline-only. Current training defaults use corrected local `_hopex.npz` clips until corrected v5+ registry artifacts are uploaded.

This branch adds:

- a current-machine `setup_train_env.sh` as the training shell setup source of truth (`/workspace/hope_isaac_venv`, `/workspace/IsaacLab`, and the verified WandB team/org/project).
- richer live `Live/...` telemetry in WandB/TensorBoard from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.
- canonical WandB motion uploads where every motion artifact contains `motion.npz`, regardless of the source filename.
- HOPE +X motion alignment in `scripts/csv_to_npz.py --robot agibot_a3` before local save/upload.
- `scripts/check_motion_target_alignment.py`, a no-Isaac gate for frame-0 yaw, +X-dominant strike velocity, and target/reference center alignment.
- explicit `wandb.finish()` before Isaac `simulation_app.close()`, so WandB runs finish and sync before Isaac can hard-exit the process.

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

On the current shared RunPod, `setup_train_env.sh` points at the actual Isaac install: `HOPE_ISAAC_VENV=/workspace/hope_isaac_venv` and `ISAACLAB_PATH=/workspace/IsaacLab`. The legacy `/workspace/isaacsim/python.sh`, `/opt/drone_venv`, and `hope-motion-py310` paths are not used for Isaac training. If another machine has different paths, update the script and this doc together.

A from-scratch Isaac Sim 4.5.0 / Isaac Lab 2.1.0 / Python 3.10 install is NOT documented here and is the single biggest reproducibility gap. Follow the official [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) first, then update `HOPE_ISAAC_VENV` and `ISAACLAB_PATH` in `setup_train_env.sh` to point at that install.

`hydra`, `omegaconf`, and `rsl_rl` are NOT in the package `setup.py` `install_requires`; they must be importable from the Isaac Lab Python used by `hope_isaac_py`. Install the package into that Python:

```bash
hope_isaac_py -m pip install -e source/whole_body_tracking
```

Quick sanity check (expect `hydra` 1.3.2):

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## WandB Setup

WandB needs two DISTINCT identities. They MUST differ or motion-registry reads fail with `Unable to find organization for entity ...`.

Current shared RunPod values are exported by `setup_train_env.sh`:

- `WANDB_ENTITY=BerkeleyPingPong` — team/entity for run logging.
- `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org` — org for the motion registry.
- `WANDB_PROJECT=hope_wbc` — training project.
- `WANDB_MOTION_PROJECT=csv_to_npz` — motion upload project.
- `WANDB_DIR=/workspace/yikang/nohope/hope_training/wandb` — local W&B cache.

Run `wandb login` before registry-backed training. The API key is stored outside git (observed in `/root/.netrc`); never write it into repo files. No WandB account? Pass `logger=tensorboard` and local `.npz` paths for smoke tests and local runs.

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

Motion references are also task-local setup. Current `HOPEPingPong*.yaml` defaults expect the corrected local ignored clips:

- `hope_training/motions/preprocessed/hope_forehand_hopex.npz`
- `hope_training/motions/preprocessed/hope_backhand_hopex.npz`

Use registry paths only after `scripts/check_motion_target_alignment.py --clip ...` passes for those downloaded artifacts. Do not commit generated logs, checkpoints, WandB caches, or motion artifacts unless the asset policy changes.

## Smoke Test

`TrackingFlat` needs no motion registry and no WandB account, so it is the cleanest smoke test:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke
```

Success means the env builds, PPO prints learning iterations, and rewards remain finite.

Before any HOPE task smoke, run the no-Isaac motion/target gate:

```bash
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPongRealSensor.yaml
```

Both commands passed on 2026-07-02 using `hope_forehand_hopex.npz` / `hope_backhand_hopex.npz`. The same check intentionally fails on the old v4 registry downloads because frame-0 yaw is 82.03/85.92 deg and strike velocity is +Y-dominant.

Local corrected-clips smoke for the current unified HOPE task:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  num_envs=32 max_iterations=1 logger=tensorboard run_name=smoke_hopex_local
```

Registry + WandB smoke `6xus13ga` on 2026-07-02 finished and synced `model_0.pt`, ONNX, config, diff, output log, and summary, but used the later-rejected v4 +Y-facing motions. Keep it as pipeline evidence only, not motion-quality evidence.

## Baseline Training Commands

Plain tracking first:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  run_name=forehand_tracking
```

HOPE racket task, unified forehand+backhand policy:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  registry_name_2="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand" \
  run_name=hope_unified
```

Useful overrides:

```bash
num_envs=4096 max_iterations=20000 seed=1
```

`HOPEPingPong` now defaults to a unified policy: clip 0 comes from `registry_name`, clip 1 comes from `registry_name_2`, and the actor receives `swing_type`. The task YAML defaults `registry_name` / `registry_name_2` to corrected local `_hopex.npz` files. It also sets `motion.rsi_on_wrap: false`, so clip wrap resamples the reference clip/time and racket target without teleporting the simulated robot. Episode reset still uses RSI.

### ppo.yaml deltas on this branch

- `max_iterations: 300000000000` is a train-FOREVER sentinel. Always pass `max_iterations=` on the CLI and stop manually when `strike_success` plateaus.
- `save_interval` 500 -> 100.
- `entropy_coef` 0.005 -> 0.004.

### Racket Target Sampling

`HOPEPingPong.yaml` and `HOPEPingPongRealSensor.yaml` currently use `racket.target_mode: reference_perturbed` for the unified forehand+backhand run:

```yaml
strike_phase_per_clip: [0.47, 0.33]
ref_perturb_curriculum_start: 0.05
ref_perturb_pos: [0.15, 0.20, 0.15]
ref_perturb_vel: [1.0, 1.0, 0.8]
```

The initial target center is computed from each imitated clip's own strike-frame racket FK state, so the teacher action and training target start aligned. The long-run distribution widens only when the success-gated exact-strike metric advances `ref_perturb_scale`. The old `pos_range_per_clip` / `vel_range_per_clip` boxes remain fallback/debug settings for `target_mode=uniform`, not the active distribution.

For the real-sensor footwork variant, `racket_progress` is zeroed on motion/target resample steps and its previous-distance baseline is reset. This prevents clip wrap or reset from contributing a fixed progress penalty/reward that the policy cannot control.

## Live Training Telemetry

`MotionOnPolicyRunner` (`utils/my_on_policy_runner.py`) logs a `Live/...` dashboard to WandB/TensorBoard every PPO iteration. Namespaces:

- `Live/<command_term>/<metric>` — per-axis command tracking (reference vs robot anchor pos/vel per x/y/z, joint error mean/max, `motion_phase`, racket pos/vel/normal per axis, `time_to_strike_s`, `pre_strike_flag`, `strike_window_flag`, `racket_speed`, ...).
- `Live/Reward/<term>` — per-reward-term contributions.
- `Live/Termination/*`, `Live/Action/*`, `Live/Env/*`.

The real "is it learning to hit" signal is `strike_success` (fraction of strikes with racket position error < `strike_success_pos_thresh` = 0.075 m) and the `*_at_strike` metrics. Episode-wide errors are DILUTED by the long non-strike phase, so do not judge progress from them.

## Reward Shaping (strike_success=0 fix)

The reward kernel is `exp(-||err||^2 / std^2)`. With `std` set to the final acceptance tolerance, the reward is ~0 for any early error (a 50 cm error gives `exp(-44) ~ 0`), so there is no gradient and `strike_success` stays stuck at 0. The target-sampling fix above handles unreachable targets; the reward shaping here handles too-narrow early rewards.

This branch uses wider-than-acceptance stds in `HOPEPingPong.yaml` so the reward gives a gradient from tens of cm out:

- `racket_position_std` 0.20
- `racket_velocity_std` 1.0
- `racket_normal_std` 0.30

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
