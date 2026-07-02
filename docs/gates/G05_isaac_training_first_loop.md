# G05 Isaac Training First Loop

Status: Partial

## Goal

Run the first end-to-end Isaac training loop that produces a policy artifact, even if the policy is weak.

This gate should prove that the training stack can consume A3 assets and produce a deployable policy format.

## Inputs

- Isaac-ready A3 asset from G04.
- Motion references or placeholder task references.
- BeyondMimic/whole-body tracking scaffold.
- Policy observation/action contract.

## Outputs

- First accepted training run.
- Training config.
- Logs and metrics.
- Exported policy artifact path or metadata.
- Initial policy evaluation notes.

## Related Directories

- `hope_training/whole_body_tracking`
- `docs/interfaces/policy_observation_action.md`
- `docs/operations/run_training.md`
- `vendor_assets/` for generated heavy policy artifacts if needed
- `external_repos/TTRL-ICRA2026` as an auto-synced reference if first-loop failures need comparison

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

- Isaac environment starts with the A3 asset.
- A random rollout works.
- A first PPO or equivalent training loop runs.
- Policy export path is documented.
- First-loop results are recorded, even if poor.

## Current State

Follow-up note (2026-07-02/03, after the simtoreal2 merge):

- The training default is now `task=HOPEPingPongDeployParity` (`cfg/train.yaml`;
  `HOPEPingPongRealSensor` is a backward-compat alias): the 175-D deploy-parity actor contract
  (see [../interfaces/policy_observation_action.md](../interfaces/policy_observation_action.md)),
  base-free footwork rewards (`base_position` removed; dense `racket_progress` + pre-strike
  slip/twist/upright penalties + `arm_torque_saturation`), per-clip blade-centered 3-D target
  pos/vel boxes, `strike_phase_per_clip: [0.47, 0.333]` on the re-grounded `_hopex` (v3) clips,
  `racket_velocity_std: 1.0` (plan 1.0 → 0.8 → 0.5), and PD-gain DR re-enabled at ±15%
  (2026-07-02 sim2real fine-tune; documented HITTER departure). The 180-D `task=HOPEPingPong` is a
  legacy comparison path and is not deploy-honest.
- This path produced the first hardware-deployed policy: `model_p4_deployparity.onnx` (175-D /
  31-act), sim2sim-validated in MuJoCo and run on the real A3 on 2026-07-02 (forehand only). The
  newest lineage is the explicit-clipped-PD fine-tune (`launch_explicitpd_ft.sh`, model_25700).
  Contract checks: `hope_isaac_py scripts/verify_realsensor.py --check layout|rollout|onnx`.
- Exact accepted run IDs/metrics for a quality baseline are still pending (see Not done).

Follow-up note (2026-07-01, `main` after the unified HITTER audit — values superseded above):

- The active `HOPEPingPong` config then defaulted to unified forehand+backhand training (`registry_name_2` enabled), `target_mode: uniform`, fixed strike plane `x=0.4`, `strike_phase_per_clip: [0.36, 0.74]` (v1 clips), actor `swing_type`, and no actor `racket_target_normal_w`.
- The `registry_for_runner` blocker and local-motion regression found during the audit were fixed in the training entry: local `motion_file=<forehand.npz> motion_file_2=<backhand.npz>` now bypasses WandB, while registry-backed runs still link the used registry artifact(s).
- The 2026-06-26 first-loop result remains useful as pipeline history, but the unified HOPEPingPong path still needs a fresh Isaac run before it can count as an accepted baseline.

Done:

- Training scaffold is present under `hope_training/whole_body_tracking`.
- Existing docs describe BeyondMimic-style training assumptions.
- The branch adds Hydra training/eval entrypoints, HOPEPingPong task config, racket-target command logic, and A3-specific robot config.
- `reimplement.md` records that `TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied A3 URDF asset, including wandb logging, checkpoint save, and `policy.onnx` export.
- Commit `42489cd` adds `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Earlier `reference_perturbed` / PATH B-C experiments are present in history, but the current `main` default is the 2026-07-01 unified HITTER path: direct uniform target sampling, no success-gated perturbation curriculum, and no `ref_vel_scale` ramp.
- `RacketTargetCommand` logs conditional exact-strike pass rates: `strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`, `strike_composite_success_exact`, and `exact_strike_sample_count_decayed`.
- `RacketTargetCommand` also supports optional debug reward logging (`debug_reward_logging`) for swing-through sign checks and raw-vs-gated reward kernels. Keep it off for production runs unless diagnosing reward scale.
- The HOPE actor observation includes `swing_type` and desired runtime targets, but not `racket_target_normal_w`; racket pose/velocity/normal remain critic/reward-only simulation state.
- `scripts/train.py` logs import provenance, env-cfg source, every applied task override, and post-override racket knobs; YAML keys that target missing env-cfg attributes raise instead of silently no-oping.
- `scripts/train.py` keeps registry defaults available from `cfg/task/*.yaml`, while `motion_file=<local.npz>` and optional `motion_file_2=<local.npz>` take precedence for no-WandB smoke tests or locally generated references.
- Local unified-policy training can use Step 9-12 video-generated motions directly with `motion_file=../motions/preprocessed/hope_forehand.npz motion_file_2=../motions/preprocessed/hope_backhand.npz logger=tensorboard`.
- Generated ONNX policy artifacts remain ignored by asset policy unless a gate records an external artifact path.
- Merged from `train_1` (2026-06-26) and superseded by the unified HITTER alignment: paddle-contact timing is per clip, expressed as `strike_phase_per_clip` (then `[0.36, 0.74]` on v1 clips; current default `[0.47, 0.333]` on the `_hopex` v3 clips); `episode_length_s: 3.0` caps each episode to about one swing; `scripts/train.py` / `cfg/train.yaml` keep the `checkpoint_path` knob for staged resume.

Done (2026-06-26 — first loop reproduced in this harness):

- The Isaac WBC training loop now runs end-to-end on this machine. Run: task `TrackingFlat`, `num_envs=1024`, `max_iterations=60`, `algo.runner.save_interval=25`, `logger=tensorboard`, `run_name=stand_bootstrap`. Mean reward improved monotonically `-4.08 -> -0.24`. Artifacts under `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_flat/2026-06-26_13-13-07_stand_bootstrap/`: checkpoints `model_0.pt`, `model_25.pt`, `model_50.pt`, `model_59.pt`, and `exported/policy.onnx` (exported via `scripts/play.py task=TrackingFlat ... checkpoint=model_59.pt motion_file=../motions/a3_stand.npz`). These prove pipeline viability only; the reference clip is a static stand, not a swing.
- Two blockers were resolved to get here, beyond EULA acceptance:
  1. **Blackwell GPU incompatibility (the real blocker).** The RTX 5090 is sm_120; Isaac Sim 4.5.0's bundled `torch 2.5.1+cu124` has no sm_120 kernels and a real CUDA matmul fails with `no kernel image is available for execution on the device`. Fixed by upgrading `hope-isaac-py310` to `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` and pinning `numpy==1.26.4` (Isaac needs `numpy<2`). Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`. `isaaclab*` keep a `torch==2.5.1` pin in metadata but are editable installs imported at runtime, so the upgrade does not break them.
  2. **WandB-only motion loading.** `scripts/train.py` fetched the motion clip only from the WandB registry. Added a local `motion_file=` override (skips WandB, mirrors `play.py`) and `motion_file: null` in `cfg/train.yaml`.
- EULA accepted non-interactively via `OMNI_KIT_ACCEPT_EULA=YES`.
- Bootstrap motion: `scripts/make_static_motion.py` generates `hope_training/motions/a3_stand.npz` (static default-pose clip, `fps=50`, `joint_pos[600,31]`, `body_pos_w[600,32,3]`) so the loop runs without the GMR/GVHMR pipeline or WandB. Placeholder reference only.

Not done:

- Re-run the current Hydra training entry in Isaac after the local-motion fix; only lightweight static checks have run in this shell.
- No accepted quality baseline is set yet; the recorded first loop proves pipeline viability, not policy strength. A real forehand/backhand baseline requires the GMR/GVHMR video pipeline or equivalent local/registry `.npz` motions.
- Forehand/backhand reference availability, uniform reachable target ranges, reward tuning, exact-strike pass rates, and stable recovery metrics still need formal acceptance.
- Exact accepted run IDs, checkpoint paths, ONNX paths, and first quality metrics still need to be recorded in this gate.

## Current Verification Commands

Account-free reproduction (no WandB, no motion data — the 2026-06-26 path). On a Blackwell GPU apply the
torch-cu128 fix first (see [run_training.md](../operations/run_training.md#blackwell-rtx-50-series-sm_120-torch-fix)):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
  --output_file ../motions/a3_stand.npz --frames 600 --fps 50
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=1024 max_iterations=60 algo.runner.save_interval=25 \
  logger=tensorboard run_name=stand_bootstrap \
  motion_file=$(pwd)/../motions/a3_stand.npz
```

GPU/Isaac environment, after `source setup_train_env.sh` and restoring/generating Step 9-12 local motions:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke

hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=hope_smoke
```

Record the startup lines from `scripts/train.py` showing source provenance and applied overrides. For
an accepted run, also record the registry artifact or local `motion_file`, WandB run ID when logging to
WandB, checkpoint path, exported ONNX path, and exact-strike pass metrics.

### Local Harness Check 2026-06-25

Commands run from the repo root:

```bash
command -v conda
command -v distrobox
command -v docker
nvidia-smi
python3 hope_training/whole_body_tracking/tests/test_table_tennis_geometry.py
bash -lc 'cd hope_training/whole_body_tracking && source setup_train_env.sh && hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"'
```

Results: the ignored A3 Isaac asset was restored locally from tracked Agibot materials and checked for
mesh references (`86` references, `0` missing). The host table-tennis geometry test passed (`6/6`;
torch aerodynamics skipped because host torch is unavailable). Later checks created `grasping` from
`docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` and verified it sees the RTX 5090 and `nvcc`.
`external_repos/IsaacLab` was cloned at tag `v2.1.0` / commit `21f7136`, and the local
`hope-isaac-py310` env now imports `torch 2.5.1+cu124`, `hydra 1.3.3`, `onnx 1.16.1`, and
`onnxscript 0.3.2` from inside `grasping`. `pip check` reports no broken requirements. Importing
Isaac/Kit reaches the NVIDIA Omniverse EULA prompt; no Isaac smoke test was run before explicit EULA
acceptance.

Additional local motion-environment checks from 2026-06-25:

```bash
cd hope_training/GMR
PYTHONNOUSERSITE=1 /home/agiuser/miniconda3/bin/conda run --no-capture-output \
  -n hope-motion-py310 python -c "import general_motion_retargeting, mujoco, smplx, torch"
```

Result: GMR commit `bb1bbe4` imports in `hope-motion-py310` with `torch 2.12.1+cu130` and
`mujoco 3.10.0`. GVHMR commit `6ec3ca3` is cloned but not installed because its requirements pin
CUDA 12.1-era `torch==2.3.0+cu121`/`pytorch3d` wheels that are not accepted for this RTX 5090 host
without a compatibility pass.

## Risks

- Training may fail before policy quality can be evaluated because of asset, observation, or reset issues.
- A weak first policy is still useful, but only if metrics and failure modes are recorded.
- Copying HITTER reward assumptions blindly may hide A3-specific limitations.
- TTRL can change upstream; record the source commit if it informs a training change.

## Next Steps

1. Run the account-free `TrackingFlat` smoke and the local unified `HOPEPingPong` smoke in the GPU/Isaac environment with Step 9-12 generated `.npz` files.
2. Record exact local motion paths or registry artifacts, WandB run IDs when used, checkpoint paths, ONNX export paths, and first quality metrics for the next successful run.
3. Set measurable acceptance metrics for first usable baseline: fall rate, racket error at strike, recovery, and command latency assumptions.
4. Run `scripts/sync_external_repos.sh` before using TTRL for comparison, and record the source commit for any extracted idea or config.
