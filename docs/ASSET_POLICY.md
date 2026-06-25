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

## Agibot Deploy Assets

The tracked deploy source is under:

- `agi/code_deployment/a3_deploy_example/`

The complete original payload is kept locally under:

- `vendor_assets/agibot/a3_deploy_example_full/`

If a command depends on a file inside `vendor_assets/`, the operation doc must say so and give the expected path.

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
