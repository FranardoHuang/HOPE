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
- `scripts/sync_external_repos.sh`

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
- TTRL is available as an ignored auto-synced local reference in `external_repos/TTRL-ICRA2026`.
- `scripts/sync_external_repos.sh` clones or fast-forwards the TTRL reference clone before G05/G08 use.
- `tmp/` has been cleared.
- Planner package tests pass when run from `hope_ws/src/hope_planner`.
- `origin/train_1` was integrated while preserving source/config/test changes and excluding generated or local editor/debug artifacts from the index.
- `.gitignore` covers `.codex-tmp/`, `.vscode/`, training logs/caches, and generated policy/model artifacts such as `*.onnx`.
- Generated `hope_training/policies/hope_forehand_policy.onnx` is treated as an ignored artifact, not a tracked source file.
- The obsolete root `Dockerfile.hope-ros2-jazzy` has been removed; ROS work now relies on the distrobox or otherwise pre-provisioned ROS 2 Jazzy environment described in `docs/operations/setup_environments.md`.
- On 2026-06-25, `~/workspace/HOPE` was created as a symlink to this checkout so `reimplement.md` commands resolve to the current repo. User-local `distrobox 1.8.2.5` and `lilipod v0.0.3` were installed, and the `osrf/ros:jazzy-desktop-full` image was pulled.
- Host `podman` and `uidmap` are installed, `hope` exists from `docker.io/osrf/ros:jazzy-desktop-full`, and the first `hope` entry verified `ROS_DISTRO=jazzy`, Python `3.12.3`, and `/usr/bin/colcon`.
- `.gitignore` now excludes `pw.txt` so local password scratch files do not enter git.

Not done:

- ROS workspace build has not been verified inside `hope`; only the ROS/Jazzy tools and container entry were verified.
- TTRL has not been promoted to a pinned submodule or fork because it is currently reference-only.
- Git LFS policy has not been decided.

## Risks

- Local-only assets can become invisible to future contributors if operation docs are not updated.
- Auto-synced external references can drift by design; extracted work must record the source commit.

## Next Steps

1. Verify ROS workspace in the intended ROS environment.
2. Record TTRL source commits whenever code, config, or experiment ideas are extracted from it.
3. Decide whether heavyweight model artifacts should remain local-only or move to Git LFS later.
