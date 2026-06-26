# Setup Local Sync

Status: Draft

## Purpose

Some required assets are too large or too environment-specific for normal git. They live under ignored local folders and must be documented here.

These files and folders are not provided by `git clone`, `git pull`, or normal branch checkout. A new machine needs restore/copy/sync steps for any gate that depends on them.

Small tracked runtime assets, such as the Purdue PACE table/net USD visual overlay under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`, do not need a manual restore step; they should appear after a normal clone or pull.

## Current Local Assets

| Path | Purpose | Source | Required By |
| --- | --- | --- | --- |
| `vendor_assets/agibot/a3_deploy_example_full/` | Complete Agibot deploy payload, including heavy runtime assets | Private/vendor-gated: Agibot handoff (~1.7 GB, no public URL) | G07 |
| `external_repos/TTRL-ICRA2026/` | Auto-synced local TTRL reference clone | Public but may be access-gated: [purdue-tracelab/TTRL-ICRA2026](https://github.com/purdue-tracelab/TTRL-ICRA2026.git) | G05, G08 |
| `external_repos/IsaacLab/` | Local Isaac Lab source checkout used by `setup_train_env.sh` when a pre-provisioned Isaac Lab checkout is absent | Public: [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab.git), observed tag `v2.1.0` / commit `21f7136` | G05 |
| `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` | Package-local Isaac A3 URDF, meshes, and config copied from tracked Agibot materials; ignored by upstream `**/assets/` rule | Tracked `agi/URDF/A3T2.5-URDF-std-pingpang/`, with URDF mesh paths rewritten from `package://.../meshes/` to `../meshes/` | G04, G05 |
| `hope_training/GMR/` | Motion retargeting (SMPL-X -> robot) clone | Public: [YanjieZe/GMR](https://github.com/YanjieZe/GMR.git) (observed pin `bb1bbe4`) | G05 motion references |
| `hope_training/GVHMR/` | Video -> SMPL-X motion recovery clone | Public: [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR.git) (observed pin `6ec3ca3`) | G05 motion references |
| GMR body-models dir (`SMPLX_NEUTRAL/MALE/FEMALE.pkl`) | SMPL-X body models for retargeting | License-gated: [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) | G05 motion references |
| `hope_training/GVHMR/inputs/checkpoints/` | GVHMR model checkpoints | License-gated: per GVHMR instructions | G05 motion references |
| WandB motion registry (`hope_forehand`/`hope_backhand` `.npz`) | Reference swing clips for `task=HOPEPingPong` | Private/org-scoped WandB registry (not redistributable) | Training (HOPEPingPong) |
| Generated policy artifacts such as `hope_training/policies/*.onnx` | Exported policies for local eval/deploy handoff | Produced by `scripts/play.py` or training/eval export; store metadata in G05/G07 | G05/G07 when a specific policy is accepted |

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

3. Sync TTRL when a gate needs it:

```bash
scripts/sync_external_repos.sh
```

The script clones TTRL if absent, then fetches and fast-forwards the local reference clone to the upstream default branch. It leaves the worktree untouched if local uncommitted changes exist.

If a result depends on TTRL, record the source commit hash printed by the script in the relevant gate doc. If TTRL becomes a stable dependency, promote it to a submodule or fork instead of relying on an ignored clone.

4. Restore Isaac Lab source when a local training env needs source packages:

```bash
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git external_repos/IsaacLab
git -C external_repos/IsaacLab rev-parse --short HEAD
```

Record the commit in G05 before accepting a training result. On 2026-06-25 the observed commit was
`21f7136`.

5. Restore the package-local A3 Isaac asset when working on G04/G05 training:

```bash
mkdir -p hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/{urdf,meshes,config}
cp -r agi/URDF/A3T2.5-URDF-std-pingpang/meshes/. \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/meshes/
cp -r agi/URDF/A3T2.5-URDF-std-pingpang/config/. \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/config/
cp agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
perl -0pi -e 's#package://0000014503_A3T2\.5-URDF-std-pingpang-0409/meshes/#../meshes/#g' \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

Then verify all copied URDF mesh references resolve:

```bash
python3 -c 'import pathlib, re, sys; urdf=pathlib.Path("hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf"); refs=re.findall(r"filename=\"([^\"]+)\"", urdf.read_text()); missing=[r for r in refs if not (urdf.parent / r).resolve().exists()]; print(f"mesh_refs={len(refs)} missing={len(missing)}"); sys.exit(1 if missing else 0)'
```

6. Restore the motion-retargeting clones (both git-ignored, absent on a fresh clone) when working on the motion pipeline:

```bash
# Video -> SMPL-X (observed pin 6ec3ca3)
git clone https://github.com/zju3dv/GVHMR.git hope_training/GVHMR
# SMPL-X -> robot (observed pin bb1bbe4)
git clone https://github.com/YanjieZe/GMR.git hope_training/GMR
# Each installs into its own conda env (python=3.10):
#   pip install -e .
```

See [reimplement.md](../../reimplement.md) steps 9-11 for the full per-env install procedure and pins.

2026-06-25 local status: both ignored clones were restored at the observed pins above. GMR was installed
editable into the existing `hope-motion-py310` conda env and passed an import check. GVHMR was cloned
but not installed because its tracked requirements pin CUDA 12.1-era PyTorch/PyTorch3D wheels; resolve
that compatibility issue before installing on an RTX 5090 / Blackwell host.

7. Add the license-gated model assets the clones depend on (you must accept each license yourself; not redistributable):

```text
# SMPL-X body models from smpl-x.is.tue.mpg.de into the GMR body-models dir:
SMPLX_NEUTRAL.pkl  SMPLX_MALE.pkl  SMPLX_FEMALE.pkl

# GVHMR checkpoints into hope_training/GVHMR/inputs/checkpoints/:
gvhmr_siga24_release.ckpt
hmr2/epoch=10-step=25000.ckpt
vitpose-h-multi-coco.pth
yolov8x.pt
dpvo.pth            # optional, only for the DPVO path
```

8. Provide the reference swing clips for `task=HOPEPingPong`:

The maintainer `hope_forehand`/`hope_backhand` `.npz` live in a private, org-scoped WandB "Motions" registry and cannot be redistributed. External users should make their own instead (see below) and point `registry_name` at their own registry collection.

> Make your own motions instead: run GVHMR (video -> SMPL-X) -> GMR (`--robot agibot_a3`) -> `scripts/csv_to_npz.py --robot agibot_a3`, then upload the resulting `.npz` to your own WandB Motions registry. This unblocks training without any private artifact. Full steps: [reimplement.md](../../reimplement.md) steps 9-11 and [run_training.md](run_training.md).

9. Keep generated training artifacts out of git:

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
test -d external_repos/IsaacLab && git -C external_repos/IsaacLab rev-parse --short HEAD
test -f hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf && echo "A3 Isaac asset present"
scripts/sync_external_repos.sh
```

Check that ignored assets are still ignored:

```bash
git status --ignored --short vendor_assets external_repos
```

Expected output uses `!!` for ignored roots, not `??`.

## Documentation Rule

Whenever a gate depends on an ignored file, the gate doc must name the file or folder and this setup doc must say how to restore it.
