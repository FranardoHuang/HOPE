# Asset Policy

This repo should stay useful without becoming a dump of generated files or opaque binaries.

## Track In Git

Track:

- Source code.
- Build scripts and small config files.
- ROS messages, launch files, and package manifests.
- Small curated calibration samples.
- Source papers and reference documents when redistribution is allowed.
- Small metadata files that explain how to obtain or verify local assets.
- Small runtime visual assets that are required for a task to load after a fresh clone, when license permits and size is modest.

## Do Not Track In Normal Git

Keep these ignored unless the project explicitly adopts Git LFS or another artifact system:

- `*.onnx`
- `*.rknn`
- `*.engine`
- `*.trt`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.tar.gz`
- `*.deb`
- `*.so`
- `*.a`
- Large generated logs, videos, raw mocap dumps, and training runs

Current ignored local asset roots:

- `vendor_assets/`
- `external_repos/`
- `tmp/`
- `.codex-tmp/`
- `.vscode/`
- `pw.txt` or other local secret scratch files; never commit passwords, tokens, or private identities.
- `external_repos/IsaacLab/` when a local Isaac Lab source checkout is used for the training environment; record the tag/commit in the relevant gate doc.
- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` and other package-local copied/generated training assets; keep only the tiny `assets/__init__.py` path-helper and local `.gitignore` tracked so `whole_body_tracking.assets.ASSET_DIR` remains importable after a fresh clone.

## Agibot Deploy Assets

The tracked deploy source is under:

- `agi/code_deployment/a3_deploy_example/`

The complete original payload is kept locally under:

- `vendor_assets/agibot/a3_deploy_example_full/`

If a command depends on a file inside `vendor_assets/`, the operation doc must say so and give the expected path.

## Agibot A3 Robot Assets

Tracked source assets:

- `agi/URDF/A3T2.5-URDF-std-pingpang/` is the current internal source URDF package for the A3 ping-pong variant. It includes the racket hand, red/black paddle faces, and the racket-center ball marker meshes: `right_hand_pingpang_Link.STL`, `pingpang_red_Link.STL`, `pingpang_black_Link.STL`, and `pingbang_ball_Link.STL`.
- `agi/URDF/a3_t2d5/` is the non-ping-pong standard A3 URDF package. Keep it as a source reference for non-racket comparisons; do not delete it just because the WBC training asset uses the ping-pong variant.
- `agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/` is the Agibot-provided MuJoCo/AimRT ping-pong model layer, including collision-oriented MJCF/mesh materials used for sim parity work.

Generated Isaac training asset:

- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` is generated from the tracked ping-pong URDF package by `scripts/prepare_a3_isaac_asset.py`. It is ignored to avoid committing duplicate copied meshes.
- The tracked `whole_body_tracking/assets/__init__.py` only defines `ASSET_DIR`; it is source code, not a generated asset.
- Verify the generated asset with `python3 scripts/prepare_a3_isaac_asset.py --check`. The check parses `model.urdf`, rejects stale `package://.../meshes` references, verifies every `../meshes/...` reference exists, and requires the ping-pong/racket meshes listed above.

Do not remove the standard `right_hand_Link.STL`, non-racket URDF, MuJoCo MJCF, or collision mesh materials unless a gate records that they are obsolete. Main is the internal solution branch and keeps these layers for training, planner, mocap, MuJoCo parity, and sim-to-real work.

## Tracked Small Runtime Visual Assets

The table-tennis Isaac scene tracks a small Purdue PACE table/net USD visual overlay under:

- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`

This folder is intentionally committed as ordinary git blobs, not Git LFS pointers, because it is a hard runtime dependency for `HOPE-TableTennis-AgibotA3-v0` visualization and is small enough for normal git. Its local `.gitattributes` disables the repo-root USD LFS filter for that folder. The included license file records the Purdue PACE MIT license. Heavy generated USDs, converted robot assets, logs, checkpoints, videos, and policy exports still belong in ignored local roots unless the team adopts an artifact system.

## External Repos

External repos start in `external_repos/` while they are only references. Reference repos may auto-update because their contents are used for reading and extraction, not direct runtime imports.

Use:

```bash
scripts/sync_external_repos.sh
```

to clone or fast-forward the local TTRL reference clone.

Promote external repos when they become real dependencies:

- Use a submodule for pinned upstream code.
- Use a fork plus submodule for code we need to modify.
- Use Git LFS only after the team agrees to track heavy binary artifacts.

TTRL is currently an auto-synced reference clone, not a pinned dependency. If code, config, or an experiment result is extracted from TTRL, record the upstream commit in the relevant gate doc or operation doc. Its future dependency state should be decided during G05/G08 planning.

## Third-Party And Vendor Provenance

Keep a short notice file at [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for bundled source/vendor drops and ignored runtime payloads. At minimum it must describe Agibot A3 URDF/deploy materials, BeyondMimic/whole_body_tracking provenance, VRPN/mocap source, and AimRT/MuJoCo runtime policy.

Public-release branches may omit private or redistribution-restricted materials, but `main` should not remove internal runbooks, gates, or source/vendor layers only for public cleanliness. Instead, document which materials are source in git, generated, local-only ignored, external references, future-open candidates, or not currently public.

## Local Sync Rule

If a local-only asset is required to reproduce a result:

1. Put it under `vendor_assets/` or another ignored root.
2. Record the expected path in [operations/setup_local_sync.md](operations/setup_local_sync.md).
3. Record the consuming gate in the relevant `docs/gates/G*.md`.
4. Never rely on a private chat message as the only description of the asset.

A fresh clone is not complete for deployment or training-result reproduction until the required ignored assets have been manually restored or linked from the agreed external artifact system.

For TTRL-only reference work, a fresh clone becomes current after `scripts/sync_external_repos.sh`; reproducible extracted work still needs the source commit recorded.

## Branch Integration Rule

When merging feature/training branches into `main`, keep source, config, tests, and docs tracked, but remove generated/debug artifacts from the index. Current examples are `.codex-tmp/`, `.vscode/`, training logs, WandB caches, checkpoints, and generated `*.onnx` policy files. If a generated artifact is needed to reproduce a gate result, record the restore path and metadata in [operations/setup_local_sync.md](operations/setup_local_sync.md) and the relevant gate doc instead of committing it directly.
