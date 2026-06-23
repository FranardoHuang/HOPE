# Setup Local Sync

Status: Draft

## Purpose

Some required assets are too large or too environment-specific for normal git. They live under ignored local folders and must be documented here.

These files and folders are not provided by `git clone`, `git pull`, or normal branch checkout. A new machine needs manual restore/copy steps for any gate that depends on them.

## Current Local Assets

| Path | Purpose | Required By |
| --- | --- | --- |
| `vendor_assets/agibot/a3_deploy_example_full/` | Complete Agibot deploy payload, including heavy runtime assets | G07 |
| `external_repos/TTRL-ICRA2026/` | Local TTRL reference clone | G05, G08 |

## Manual Restore Checklist

Run from the repo root.

1. Create ignored roots if missing:

```bash
mkdir -p vendor_assets/agibot external_repos
```

2. Restore the full Agibot deploy payload when working on deploy, MuJoCo runtime, or hardware dry-run:

```bash
# Copy or move the Agibot-provided full deploy package here:
vendor_assets/agibot/a3_deploy_example_full/
```

Expected contents include heavy runtime assets that should not be committed in normal git:

- ONNX/RKNN/TensorRT models
- sysroot tarballs
- `.deb` packages
- prebuilt libraries
- full standalone MuJoCo/deploy runtime files

The tracked source/config subset lives separately at:

```text
agi/code_deployment/a3_deploy_example/
```

3. Restore or clone TTRL only when a gate needs it:

```bash
# If the ignored local clone is absent:
git clone https://github.com/purdue-tracelab/TTRL-ICRA2026 external_repos/TTRL-ICRA2026
```

If a result depends on TTRL, record the commit hash in the relevant gate doc. If TTRL becomes a stable dependency, promote it to a submodule or fork instead of relying on an ignored clone.

4. Keep generated training artifacts out of git:

```text
hope_training/whole_body_tracking/logs/
hope_training/whole_body_tracking/outputs/
hope_training/whole_body_tracking/artifacts/
hope_training/whole_body_tracking/wandb/
hope_training/motions/
```

If a generated policy, motion, or dataset is required to reproduce a result, record where it lives, how it was produced, and whether it is in WandB, local ignored storage, or a future artifact store.

## Expected Policy

- Keep source and small config in git.
- Keep large generated/runtime artifacts in `vendor_assets/`.
- Promote external repos to submodules only after a project decision.

## Rebuild Local Assets

If `vendor_assets/agibot/a3_deploy_example_full/` is missing, restore it from the Agibot-provided deploy package and keep the same relative path. Then update this file with the source and date.

If `external_repos/TTRL-ICRA2026/` is missing, clone or copy the repo again and record the commit if it becomes part of a result.

## Verification

Check local-only assets:

```bash
test -d vendor_assets/agibot/a3_deploy_example_full && echo "Agibot full deploy payload present"
test -d external_repos/TTRL-ICRA2026 && echo "TTRL local reference present"
```

Check that ignored assets are still ignored:

```bash
git status --ignored --short vendor_assets external_repos
```

Expected output uses `!!` for ignored roots, not `??`.

## Documentation Rule

Whenever a gate depends on an ignored file, the gate doc must name the file or folder and this setup doc must say how to restore it.
