# G00 Materials And Harness

Status: Partial

## Goal

Make the repository usable as a stable human-and-agent workspace before deeper reproduction work begins.

This gate is about project structure, support materials, documentation, ignored assets, external references, and basic verification commands.

## Inputs

- HITTER paper: `papers/2508.21043v2.pdf`
- HOPE reference documents in the repo root and `mocap/`
- Agibot support materials under `agi/`
- Local complete Agibot payload under `vendor_assets/`
- Local external references under `external_repos/`

## Outputs

- Clear repo map.
- Clear asset policy.
- Gate documents.
- Agent documentation update rules.
- Basic verification commands.

## Related Directories

- `docs/`
- `papers/`
- `agi/`
- `hope_ws/`
- `hope_training/`
- `vendor_assets/`
- `external_repos/`

## Operation Docs

- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)
- [../operations/setup_environments.md](../operations/setup_environments.md)

## Acceptance Criteria

- `docs/START_HERE.md` exists and points to all important docs.
- Every gate has a living document.
- Git does not see heavyweight local assets by default.
- Each local-only required asset has a documented path.
- The repo has agent rules that require docs to be updated with progress and target changes.

## Current State

Done:

- HITTER paper is in `papers/`.
- ChingMu VRPN package is in `hope_ws/src/vrpn_mocap`.
- Agibot deploy source/config subset is in `agi/code_deployment/a3_deploy_example`.
- Full Agibot deploy payload is preserved locally in `vendor_assets/agibot/a3_deploy_example_full`.
- TTRL is available as an ignored local reference in `external_repos/TTRL-ICRA2026`.
- `tmp/` has been cleared.
- Planner package tests pass when run from `hope_ws/src/hope_planner`.

Not done:

- ROS workspace build has not been verified because `colcon` is not available in the current shell.
- TTRL has not been promoted to a pinned submodule or fork.
- Git LFS policy has not been decided.

## Risks

- Local-only assets can become invisible to future contributors if operation docs are not updated.
- External repos can drift if they stay as ignored clones instead of pinned dependencies.

## Next Steps

1. Verify ROS workspace in the intended ROS environment.
2. Decide whether TTRL becomes a submodule, fork, or stays local-only.
3. Decide whether heavyweight model artifacts should remain local-only or move to Git LFS later.
