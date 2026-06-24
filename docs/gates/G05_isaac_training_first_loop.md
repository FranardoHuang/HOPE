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

Not done:

- This Codex shell has not independently reproduced the Isaac training run because the GPU/Isaac environment is not active here.
- No accepted quality baseline is set yet; the recorded first loop proves pipeline viability, not policy strength.
- Forehand/backhand reference availability, reference-perturbation ranges, optional uniform reachable target ranges, reward tuning, and stable recovery metrics still need formal acceptance.
- Exact accepted run IDs, checkpoint paths, ONNX paths, and first quality metrics still need to be recorded in this gate.

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
