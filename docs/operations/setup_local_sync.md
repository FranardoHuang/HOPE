# Setup Local Sync

Status: Draft

## Purpose

Some required assets are too large or too environment-specific for normal git. They live under ignored local folders and must be documented here.

These files and folders are not provided by `git clone`, `git pull`, or normal branch checkout. A new machine needs restore/copy/sync steps for any gate that depends on them.

## Current Local Assets

| Path | Purpose | Source | Required By |
| --- | --- | --- | --- |
| `agi/a3_deploy_example/` | Local full-ish Agibot deploy working package with ping-pong C++ runner, dist configs, binaries, and `model_15200.onnx` | Local developer/vendor-derived package, ignored by git; observed on 2026-06-30 | G07 ping-pong recovery audit |
| `vendor_assets/agibot/a3_deploy_example_full/` | Complete Agibot deploy payload, including heavy runtime assets | Private/vendor-gated: Agibot handoff (~1.7 GB, no public URL) | G07 |
| `external_repos/TTRL-ICRA2026/` | Auto-synced local TTRL reference clone | Public but may be access-gated: [purdue-tracelab/TTRL-ICRA2026](https://github.com/purdue-tracelab/TTRL-ICRA2026.git) | G05, G08 |
| `hope_training/GMR/` | Motion retargeting (SMPL-X -> robot) clone | Public: [YanjieZe/GMR](https://github.com/YanjieZe/GMR.git) (observed pin `bb1bbe4`) | Motion pipeline |
| `hope_training/GVHMR/` | Video -> SMPL-X motion recovery clone | Public: [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR.git) (observed pin `6ec3ca3`) | Motion pipeline |
| GMR body-models dir (`SMPLX_NEUTRAL/MALE/FEMALE.pkl`) | SMPL-X body models for retargeting | License-gated: [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) | Motion pipeline |
| `hope_training/GVHMR/inputs/checkpoints/` | GVHMR model checkpoints | License-gated: per GVHMR instructions | Motion pipeline |
| `hope_training/GVHMR/inputs/checkpoints/{dpvo,gvhmr,hmr2,vitpose,yolo}/` | GVHMR public pretrained weights restored on the current RunPod | Upstream GVHMR Google Drive hit quota; restored from public Hugging Face mirror `camenduru/GVHMR` on 2026-07-02 | Motion pipeline |
| `hope_training/motions/preprocessed/hope_forehand_hopex.npz` and `hope_backhand_hopex.npz` | Corrected HOPE +X reference swing clips for current `task=HOPEPingPong*` defaults | Generated locally from the 2026-07-02 v4 motions with `scripts/reground_hope_frame.py`; ignored, not redistributable | Training (HOPEPingPong) |
| WandB motion registry (`hope_forehand`/`hope_backhand`) | Optional reference swing source after corrected artifacts are uploaded | Private/org-scoped WandB registry. The 2026-07-02 `hope_forehand:v4` / `hope_backhand:v4` artifacts contain canonical `motion.npz` but were later rejected because they still face +Y | Training (HOPEPingPong) |

## Manual Restore Checklist

Run from the repo root.

1. Create ignored roots if missing:

```bash
mkdir -p vendor_assets/agibot external_repos
```

2. Restore the Agibot deploy payloads when working on deploy, MuJoCo runtime, or hardware dry-run:

```bash
# Copy or move the Agibot-provided full deploy package here:
vendor_assets/agibot/a3_deploy_example_full/

# If reproducing the current ping-pong recovery audit exactly, restore the
# local working package here as well:
agi/a3_deploy_example/
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

For the ignored ping-pong package observed on 2026-06-30, the relevant local
artifacts were:

```text
agi/a3_deploy_example/dist/a3_deploy_x86_64/a3_deploy_onnx_ref_pingpong
agi/a3_deploy_example/dist/a3_deploy_x86_64/models/model_15200.onnx
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/a3_pingpong_main.cpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/
```

The fingerprints and audit sequence are recorded in
[run_pingpong_recovery_audit.md](run_pingpong_recovery_audit.md). If this
package becomes the official deployment path, promote source/config changes to
tracked `agi/code_deployment/a3_deploy_example/` and keep only heavy binaries or
models ignored.

3. Sync TTRL when a gate needs it:

```bash
scripts/sync_external_repos.sh
```

The script clones TTRL if absent, then fetches and fast-forwards the local reference clone to the upstream default branch. It leaves the worktree untouched if local uncommitted changes exist.

If a result depends on TTRL, record the source commit hash printed by the script in the relevant gate doc. If TTRL becomes a stable dependency, promote it to a submodule or fork instead of relying on an ignored clone.

4. Restore the motion-retargeting clones (both git-ignored, absent on a fresh clone) when working on the motion pipeline:

```bash
# Video -> SMPL-X (observed pin 6ec3ca3)
git clone https://github.com/zju3dv/GVHMR.git hope_training/GVHMR
# SMPL-X -> robot (observed pin bb1bbe4)
git clone https://github.com/YanjieZe/GMR.git hope_training/GMR
# Each installs into its own conda env (python=3.10):
#   pip install -e .
```

See [reimplement.md](../../reimplement.md) steps 9-11 for the full per-env install procedure and pins.

5. Add the license-gated model assets the clones depend on (you must accept each license yourself; not redistributable):

```text
# SMPL-X body models from smpl-x.is.tue.mpg.de into the GMR body-models dir:
SMPLX_NEUTRAL.pkl  SMPLX_MALE.pkl  SMPLX_FEMALE.pkl

# GVHMR checkpoints into hope_training/GVHMR/inputs/checkpoints/:
gvhmr/gvhmr_siga24_release.ckpt
hmr2/epoch=10-step=25000.ckpt
vitpose/vitpose-h-multi-coco.pth
yolo/yolov8x.pt
dpvo/dpvo.pth            # optional, only for the DPVO path

# GVHMR body models into hope_training/GVHMR/inputs/checkpoints/body_models/:
smplx/SMPLX_NEUTRAL.npz
smpl/SMPL_NEUTRAL.pkl

# GMR body models into hope_training/GMR/assets/body_models/smplx/:
SMPLX_NEUTRAL.pkl  SMPLX_MALE.pkl  SMPLX_FEMALE.pkl
```

6. Provide the reference swing clips for `task=HOPEPingPong`:

Current `HOPEPingPong.yaml` and `HOPEPingPongRealSensor.yaml` defaults use local ignored corrected clips:

```text
hope_training/motions/preprocessed/hope_forehand_hopex.npz
hope_training/motions/preprocessed/hope_backhand_hopex.npz
```

They were generated on 2026-07-02 from the local v4 artifacts:

```bash
cd hope_training/whole_body_tracking
python scripts/reground_hope_frame.py --in artifacts/hope_forehand:v4/motion.npz \
  --out ../motions/preprocessed/hope_forehand_hopex.npz --strike-phase 0.47
python scripts/reground_hope_frame.py --in artifacts/hope_backhand:v4/motion.npz \
  --out ../motions/preprocessed/hope_backhand_hopex.npz --strike-phase 0.33
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml
```

The private registry aliases observed earlier are not accepted for long training until corrected v5+ artifacts are uploaded:

```text
dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_forehand:latest
  -> BerkeleyPingPong/csv_to_npz/hope_forehand:v4, manifest: motion.npz, REJECTED: frame0_yaw=82.03 deg, +Y-dominant strike velocity
dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_backhand:latest
  -> BerkeleyPingPong/csv_to_npz/hope_backhand:v4, manifest: motion.npz, REJECTED: frame0_yaw=85.92 deg, +Y-dominant strike velocity
```

> Make your own motions instead: run GVHMR (video -> SMPL-X) -> GMR (`--robot agibot_a3`) -> `scripts/csv_to_npz.py --robot agibot_a3`. The converter now applies HOPE +X alignment by default for Agibot A3 before local save/upload. Run `scripts/check_motion_target_alignment.py` before training or uploading the resulting `.npz` to your own WandB Motions registry. Full steps: [reimplement.md](../../reimplement.md) steps 9-11 and [run_training.md](run_training.md).

7. Keep generated training artifacts out of git:

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
- Keep reference-only external repos synced through `scripts/sync_external_repos.sh`.
- Promote external repos to submodules only after a project decision.

## Rebuild Local Assets

If `vendor_assets/agibot/a3_deploy_example_full/` is missing, restore it from the Agibot-provided deploy package and keep the same relative path. Then update this file with the source and date.

If `external_repos/TTRL-ICRA2026/` is missing or stale, run:

```bash
scripts/sync_external_repos.sh
```

Record the printed commit if TTRL becomes part of a result.

## Verification

Check local-only assets:

```bash
test -d vendor_assets/agibot/a3_deploy_example_full && echo "Agibot full deploy payload present"
test -d agi/a3_deploy_example && echo "Local ping-pong deploy package present"
scripts/sync_external_repos.sh
```

Check that ignored assets are still ignored:

```bash
git status --ignored --short vendor_assets external_repos
```

Expected output uses `!!` for ignored roots, not `??`.

## Documentation Rule

Whenever a gate depends on an ignored file, the gate doc must name the file or folder and this setup doc must say how to restore it.
