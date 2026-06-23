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

## Agibot Deploy Assets

The tracked deploy source is under:

- `agi/code_deployment/a3_deploy_example/`

The complete original payload is kept locally under:

- `vendor_assets/agibot/a3_deploy_example_full/`

If a command depends on a file inside `vendor_assets/`, the operation doc must say so and give the expected path.

## External Repos

External repos start in `external_repos/` while they are only references. Promote them when they become real dependencies:

- Use a submodule for pinned upstream code.
- Use a fork plus submodule for code we need to modify.
- Use Git LFS only after the team agrees to track heavy binary artifacts.

TTRL is currently a reference clone. Its future state should be decided during G05/G08 planning.

## Local Sync Rule

If a local-only asset is required to reproduce a result:

1. Put it under `vendor_assets/` or another ignored root.
2. Record the expected path in [operations/setup_local_sync.md](operations/setup_local_sync.md).
3. Record the consuming gate in the relevant `docs/gates/G*.md`.
4. Never rely on a private chat message as the only description of the asset.

A fresh clone is not complete for deployment or training-result reproduction until the required ignored assets have been manually restored or linked from the agreed external artifact system.
