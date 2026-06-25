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
- Commit `c951d9d` is integrated here: `HOPEPingPong` now defaults to `target_mode: reference_perturbed`, which samples racket targets around the reference motion's strike-frame racket pose/velocity/normal with a curriculum perturbation. This keeps the first RL target distribution reachable by construction while preserving the legacy uniform box as an explicit placeholder mode.
- Latest `origin/train_1` training updates are integrated here: `HOPEPingPong.yaml` now uses the PATH B/C slower-hit defaults (`racket_position_weight: 8`, `racket_position_std: 0.15`, `racket_velocity_weight: 4`, `racket_velocity_std: 1.2`, `racket_normal_weight: 3`, `base_position_weight: 4`, `motion_scale: 0.30`), shrinks reference perturbations to `[0.06, 0.08, 0.06]` position and `[0.3, 0.3, 0.25]` velocity, narrows the strike reward window to `0.12 s`, and sets `ref_vel_scale: 0.6` to train a controllable 3-4 m/s hit before returning to the full-speed reference.
- `RacketTargetCommand` now logs conditional exact-strike pass rates: `strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`, `strike_composite_success_exact`, and `exact_strike_sample_count_decayed`. The success-gated perturbation curriculum advances from these exact-strike rates, not from diluted episode-wide errors.
- `RacketTargetCommand` also supports optional debug reward logging (`debug_reward_logging`) for swing-through sign checks and raw-vs-gated reward kernels. Keep it off for production runs unless diagnosing reward scale.
- The HOPE actor observation now includes desired `racket_target_normal_w`; actual racket pose/velocity/normal remains critic/reward-only simulation state.
- `scripts/train.py` logs import provenance, env-cfg source, every applied task override, and post-override racket knobs; YAML keys that target missing env-cfg attributes raise instead of silently no-oping.
- Generated ONNX policy artifacts remain ignored by asset policy unless a gate records an external artifact path.

Not done:

- This Codex shell has not independently reproduced the Isaac training run because the GPU/Isaac environment is not active here.
- No accepted quality baseline is set yet; the recorded first loop proves pipeline viability, not policy strength.
- Forehand/backhand reference availability, reference-perturbation ranges, optional uniform reachable target ranges, reward tuning, exact-strike pass rates, and stable recovery metrics still need formal acceptance.
- Exact accepted run IDs, checkpoint paths, ONNX paths, and first quality metrics still need to be recorded in this gate.

## Current Verification Commands

GPU/Isaac environment, after `source setup_train_env.sh`:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke

hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=hope_smoke
```

Record the startup lines from `scripts/train.py` showing source provenance and applied overrides. For
an accepted run, also record the WandB run ID, checkpoint path, exported ONNX path, and exact-strike
pass metrics.

## Risks

- Training may fail before policy quality can be evaluated because of asset, observation, or reset issues.
- A weak first policy is still useful, but only if metrics and failure modes are recorded.
- Copying HITTER reward assumptions blindly may hide A3-specific limitations.
- TTRL can change upstream; record the source commit if it informs a training change.

## Next Steps

1. Record exact wandb run IDs, checkpoint paths, and ONNX export paths for the first successful runs.
2. Set measurable acceptance metrics for first usable baseline: fall rate, racket error at strike, recovery, and command latency assumptions.
3. Train and evaluate both forehand and backhand references with `target_mode: reference_perturbed`, then compare against legacy uniform sampling only after reachable ranges are measured.
4. Run `scripts/sync_external_repos.sh` before using TTRL for comparison, and record the source commit for any extracted idea or config.
