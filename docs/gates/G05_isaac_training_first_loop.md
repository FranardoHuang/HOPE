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

Done:

- Training scaffold is present under `hope_training/whole_body_tracking`.
- Existing docs describe BeyondMimic-style training assumptions.
- The branch adds Hydra training/eval entrypoints, HOPEPingPong task config, racket-target command logic, and A3-specific robot config.
- `reimplement.md` records that `TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied A3 URDF asset, including wandb logging, checkpoint save, and `policy.onnx` export.
- Commit `42489cd` adds `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Commit `c951d9d` is integrated here and added `target_mode: reference_perturbed`; the current unified forehand+backhand task YAMLs now use it by default with per-clip reference strike centers and `strike_phase_per_clip: [0.47, 0.33]`.
- 2026-07-02: `HOPEPingPong.yaml` and `HOPEPingPongRealSensor.yaml` were synchronized for unified forehand+backhand training. Old single-swing wording and old backhand positive-y/high-z coordinate notes were removed from the YAML comments; both observation-route candidates now point at the same re-grounded target boxes.
- 2026-07-02: Clip wrap no longer performs mid-episode RSI for HOPE ping-pong (`motion.rsi_on_wrap: false` in both task YAMLs). Episode reset still initializes from the reference, but wrap only advances the reference clip/time and target, forcing the policy to learn physical between-swing recovery.
- 2026-07-02: `racket_progress` now resets its previous-distance baseline and emits zero on motion/target resample steps, removing the fixed wrap/reset reward spike from the base-free footwork signal.
- 2026-07-02: `HOPEPingPong` and `HOPEPingPongRealSensor` now use `racket.target_mode: reference_perturbed`: the initial target center is the imitated clip's own strike-frame racket FK state, and the long-run distribution widens through success-gated perturbations (`ref_perturb_pos=[0.15,0.20,0.15]`, `ref_perturb_vel=[1.0,1.0,0.8]`). The old per-clip boxes remain only as fallback/debug settings for `target_mode=uniform`.
- 2026-07-02: The shared RunPod now has an independent `hope-motion-py310` Conda env for GVHMR/GMR motion preparation, separate from Isaac Lab. GVHMR/GMR import checks pass on RTX 5090 with PyTorch `2.7.0+cu128`; PyTorch3D `0.7.9` was built from source for `sm_120`; GVHMR non-body checkpoints are present; and the local ignored GMR clone has verified `agibot_a3` MJCF/IK registration with 31 hinge joints matching `joint_order_agibot_a3.yaml`.
- 2026-07-02: Uploaded forehand/backhand MP4s were converted through GVHMR -> GMR -> `scripts/csv_to_npz.py` into local ignored motions: `hope_training/motions/preprocessed/hope_forehand.npz` (`joint_pos=(139,31)`, `body_pos_w=(139,32,3)`, `fps=50`) and `hope_training/motions/preprocessed/hope_backhand.npz` (`joint_pos=(132,31)`, `body_pos_w=(132,32,3)`, `fps=50`).
- 2026-07-02: WandB setup is verified for `WANDB_ENTITY=BerkeleyPingPong`, `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org`, `WANDB_PROJECT=hope_wbc`, and `WANDB_MOTION_PROJECT=csv_to_npz`. The registry aliases `dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_forehand:latest` and `.../hope_backhand:latest` resolve to `BerkeleyPingPong/csv_to_npz/hope_forehand:v4` and `BerkeleyPingPong/csv_to_npz/hope_backhand:v4`; both contain `motion.npz`.
- 2026-07-02: Registry-backed `HOPEPingPong` WandB smoke passed with `source setup_train_env.sh && hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=32 max_iterations=1 logger=wandb run_name=smoke_registry_wandb_finish`. W&B run `6xus13ga` finished at https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga, used both motion artifacts, and synced `model_0.pt`, `2026-07-02_11-56-04_smoke_registry_wandb_finish.onnx`, config, diff, output log, and summary. Local outputs are under `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-07-02_11-56-04_smoke_registry_wandb_finish/`.
- 2026-07-02: The same registry `hope_forehand/backhand:v4` artifacts used for the smoke were later verified as directionally wrong for real training: frame-0 pelvis yaw is 82.03/85.92 deg and strike velocity is +Y-dominant. Corrected local ignored clips were generated at `hope_training/motions/preprocessed/hope_forehand_hopex.npz` and `hope_backhand_hopex.npz`; both task YAMLs default to these local files until corrected v5+ registry artifacts are uploaded.
- 2026-07-02: `scripts/csv_to_npz.py --robot agibot_a3` now auto-aligns exported world-frame arrays into HOPE +X before saving/uploading, and `scripts/check_motion_target_alignment.py` provides a no-Isaac gate. Verification: `python -m py_compile ...` passed; `python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml` and `--yaml cfg/task/HOPEPingPongRealSensor.yaml` passed; the same check fails on old v4 as expected.
- 2026-07-02 (later): Fixed an `UnboundLocalError` in `RacketTargetCommand._resample_command` (`hope_commands.py`; the base-XY coupling branch read `motion` before assignment) that crashed EVERY `HOPEPingPong*` env reset — the working tree could not start training at all until this fix. After the fix: local-clip smoke passed (`num_envs=32 max_iterations=2 logger=tensorboard`), and a bounded verification training run passed: `hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=4096 max_iterations=300 run_name=e2e_verify_train seed=1` -> W&B run `wuj6ds9u` (https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/wuj6ds9u), mean reward -1.37 -> ~25, mean episode length 5 -> ~340 steps, `strike_success` 0 -> 0.006 in 300 iters, `model_{0,100,200,299}.pt` + ONNX exported and synced. Pipeline viability evidence on the corrected `_hopex` clips, still not a quality baseline.
- 2026-07-02 (later): Full fresh MP4 -> npz rerun in an isolated dir reproduces the shipped artifacts bit-for-bit (GMR pkl and retargeted CSV byte-identical; npz equal to `hope_forehand_hopex.npz` within 2e-7 float noise), and `check_motion_target_alignment.py --clip` passes on the regenerated clip. One env caveat found and documented: GVHMR's YOLO load needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` under torch 2.7 (see `docs/operations/setup_environments.md`).

Not done:

- No accepted quality baseline is set yet; the recorded local and WandB smoke runs prove pipeline viability, not policy strength.
- The verified WandB smoke used only 32 envs for 1 PPO iteration and used later-rejected +Y-facing registry motions; no long unified policy training run or post-train evaluation has been accepted.
- Target-range widening, reward tuning, stable recovery metrics, and first usable baseline thresholds still need formal acceptance.
- The corrected `_hopex.npz` motion clips are ignored local artifacts; new machines must restore or regenerate them through `setup_local_sync.md` before reproducing current default `HOPEPingPong*` training.

## Risks

- Training may fail before policy quality can be evaluated because of asset, observation, or reset issues.
- A weak first policy is still useful, but only if metrics and failure modes are recorded.
- Copying HITTER reward assumptions blindly may hide A3-specific limitations.
- TTRL can change upstream; record the source commit if it informs a training change.

## Next Steps

1. Start the real unified forehand+backhand training run from the corrected local `_hopex.npz` clips, or first upload corrected v5+ registry artifacts and record their aliases.
2. Set measurable acceptance metrics for first usable baseline: fall rate, racket error at strike, physical recovery after clip wrap, and command latency assumptions.
3. Evaluate the trained checkpoint/ONNX from the W&B run and record exact quality metrics, checkpoint path, ONNX path, and failure modes here.
4. Watch `ref_perturb_scale` and exact-strike pass rates during long training; the target distribution should widen only through the success-gated `reference_perturbed` curriculum.
5. Run `scripts/sync_external_repos.sh` before using TTRL for comparison, and record the source commit for any extracted idea or config.
