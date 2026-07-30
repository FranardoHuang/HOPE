# Setup Local Sync

Status: Draft

## Purpose

Some required assets are too large or too environment-specific for normal git. They live under ignored local folders and must be documented here.

These files and folders are not provided by `git clone`, `git pull`, or normal branch checkout. A new machine needs restore/copy/sync steps for any gate that depends on them.

Small tracked runtime assets, such as the Purdue PACE table/net USD visual overlay under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`, do not need a manual restore step; they should appear after a normal clone or pull.

`vendor_assets/` 必须是当前 checkout 内的真实 ignored 目录，不是仓库跟踪项，也不能是绝对、
跨 checkout 或自指 symlink。新 checkout 先做：

```bash
test ! -e vendor_assets
mkdir vendor_assets
test -d vendor_assets
test ! -L vendor_assets
git check-ignore -q vendor_assets
```

如果这个路径已存在，先只读核对 `lstat`、内容来源和使用者；不要删除、替换或跟随未知 symlink。
恢复某个子树时应先复制到 checkout 外的新 staging 目录，核对逐文件清单后，仅在目标子树完全
不存在时用同一文件系统的 rename 发布。不得用 `rsync --delete` 清理一个共享
`vendor_assets/` 根，也不得恢复进正在训练的 checkout。

## Current Local Assets

| Path | Purpose | Source | Required By |
| --- | --- | --- | --- |
| `vendor_assets/agibot/a3_deploy_example_full/` | Complete Agibot deploy payload, including heavy runtime assets | Private/vendor-gated: Agibot handoff (~1.7 GB, no public URL) | G07 |
| `external_repos/TTRL-ICRA2026/` | Auto-synced local TTRL reference clone | Public but may be access-gated: [purdue-tracelab/TTRL-ICRA2026](https://github.com/purdue-tracelab/TTRL-ICRA2026.git) | G05, G08 |
| `external_repos/IsaacLab/` | Local Isaac Lab source checkout used by `setup_train_env.sh` when a pre-provisioned Isaac Lab checkout is absent | Public: [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab.git), observed tag `v2.1.0` / commit `21f7136` | G05 |
| `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` | Package-local Isaac A3 URDF, meshes, and config; ignored by upstream `**/assets/` rule | Normally rebuild from tracked `agi/URDF/A3T2.5-URDF-std-pingpang/`. For C3/D3 K100 v2, do **not** rebuild or hand-copy: the attestor inventories the exact training checkout source under `/workspace/codexschema/nohope_signed_face_c3d3_l1_4467d79/.../assets/agibot_a3` and hydrates a fresh eval checkout per [v2 operation](run_phase1_signed_face_c3d3_k100_v2.md) | G04, G05, G06 |
| Stable A3 pre-converted runtime directory containing `model.usd` and `configuration/` | Optional Isaac robot-scene cache selected by `HOPE_AGIBOT_A3_USD_PATH`; it avoids converting the same robot description again in every training process | Copy the complete directory produced by one successful conversion to an ignored, stable runtime path; do not commit it | G05 training sprint |
| `hope_training/GMR/` | Motion retargeting (SMPL-X -> robot) clone | Public: [YanjieZe/GMR](https://github.com/YanjieZe/GMR.git) (observed pin `bb1bbe4`) | G05 motion references |
| `hope_training/GVHMR/` | Video -> SMPL-X motion recovery clone | Public: [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR.git) (observed pin `6ec3ca3`) | G05 motion references |
| GMR body-models dir (`SMPLX_NEUTRAL/MALE/FEMALE.pkl`) | SMPL-X body models for retargeting | License-gated: [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) | G05 motion references |
| `hope_training/GVHMR/inputs/checkpoints/` | GVHMR model checkpoints | License-gated: per GVHMR instructions | G05 motion references |
| `hope_training/GVHMR/inputs/checkpoints/{dpvo,gvhmr,hmr2,vitpose,yolo}/` | GVHMR public pretrained weights restored on the current RunPod | Upstream GVHMR Google Drive hit quota; restored from public Hugging Face mirror `camenduru/GVHMR` on 2026-07-02 | G05 motion references |
| `${HOME}/Downloads/{Franco,v6_dang,v7_dang}/*.mp4` | Ten private air-swing recordings for the 2026-07-11 shoulder-dominant/block/loop motion-library study | User recording; content-addressed metadata only in `configs/motion_video_intake_20260711.json`; do not publish raw bytes | G05/G08 motion-library, TOPP and recovery study |
| `${HOME}/Downloads/{v12,static,motion}/*.mp4` | Seven private v12 block, backhand high-press and lateral-locomotion teacher recordings from 2026-07-13 | User recording; schema-2 content-addressed metadata only in `configs/motion_video_intake_20260713.json`; do not publish raw bytes | G05/G08 motion library, fifth-action and lateral composition studies |
| RunPod `/workspace/codexschema/motion_video_intake_20260711/` | Private content-addressed copy, GVHMR outputs, logs and queue bindings for the same ten recordings | Manually copied from the exact local videos; ignored runtime evidence, not a durable artifact service | G05/G08 motion preprocessing |
| RunPod `/workspace/codexschema/motion_video_intake_20260713_{s0,m0}/` | Accepted no-clobber runtime evidence for [S0/M0](../DEFINITIONS.md), including GVHMR, exact handoff, canonical-beta and exact-GMR v2 completions | Created on Pod1 on 2026-07-13/14. Copy each complete root, including `exact_gmr_v2/{outputs,logs,audits,bindings,completion_manifest.json}`; S0/M0 completion manifest SHA-256 is `a762d6df...d1a23` / `fdd60fcf...396e`. Preserve the five bound inputs, both handoffs, donor, runtime closure and GMR bundle together. These paths are not a durable artifact service and must never be reconstructed from the later Pod2 rc127 absence | G04/G08 offline motion preprocessing |
| `vendor_assets/motion_finalize_20260724/` | canonical 五动作终审的当前本地 ignored 输入根：ready、四个 SHADOW 源、四个 GMR 源和 scope-specific 换面证据 | 从下列 Pod2 内容寻址路径恢复；逐文件哈希见本页恢复命令。它是 compiler candidate 输入/证据根，不是训练资产库 | G08 canonical motion compiler and audit |
| Pod2 `/workspace/codexschema/firstframe_20260723/designed_ready/` | shared-ready 的 NPZ 与说明 JSON | `canonical_ready_v1.npz` SHA-256 `cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0`；JSON SHA-256 `95be5bc32150f7dea8a4eed41bf591acd95588624e2f6f3f508f39f2c6c9e227` | G08 ready lineage |
| Pod2 `/workspace/codexschema/franco_pipeline_20260722/fk/SHADOW_{fh_loop_yaw147,franco_bh_loop_c,bh_block_yaw80,s0_yaw72}.npz` | 四个 50 Hz SHADOW 运动学源 | SHA-256 依次为 `faa8df8c...3dcc1`、`d5338168...c1fba`、`55870b98...a5fe0`、`2cd32da1...78c01`；均为 probe-grade source，不直接授权 registry/training | G08 source inputs |
| Pod2 `/workspace/codexschema/franco_set_cert_20260722/inputs/` | 与四个 SHADOW 源绑定的 GMR PKL | 文件名与完整 SHA-256 固定在下列恢复命令；不得用同名重跑件替换 | G08 source lineage |
| RunPod `/workspace/codexschema/motion_spatial_retarget_signed_a4bbbaa_v1/` | Full 640-cell signed spatial-retarget proposal result and launch evidence | Created on Pod1 on 2026-07-13; result is 225,920 bytes, SHA-256 `69c3db16fa78f526aef49f20eeafe0d7e5e3004c4ed27f5e2823bb3574e2465c`; tracked summary is `configs/motion_video_spatial_retarget_signed_results_20260713.json` | G08 B/C candidate materialization |
| RunPod `/workspace/codexschema/motion_video_intake_20260711/gmr_spatial_retarget_primary_v1/` | Exact B/C rank-0 whole-motion SE(2) outputs and report-last receipts | Created on Pod1 CPU on 2026-07-14; restore exact paths/bytes/SHA from `configs/motion_backhand_loop_bc_se2_materialization_results_20260714.json`. Private PKLs are not tracked; do not rerun into an existing no-clobber root | G04/G08 B/C schema-2 preregistration |
| RunPod `motion_video_intake_20260711/gmr_provenance/GMR_aabea2e.bundle` | Recovery bundle for the clean five-local-commit GMR diagnostic source used on 2026-07-11; 282,953,810 bytes, SHA-256 `5b94af15f4a367dff8d7dc6c1cf14d26be6a649a25df6e1c1046b0e6ab72e2de` | Copy from the exact Pod1 ignored evidence root or another verified backup; `git bundle verify` and require advertised commit `aabea2eee4be4bc16d4be17dac5ffa85e5a31539` | Reproducing the Franco/v6/v7 diagnostic GMR outputs |
| WandB motion registry (`hope_forehand`/`hope_backhand` `.npz`) | Optional shared/internal reference swing clips for `task=HOPEPingPong` | Private/org-scoped WandB registry (not redistributable). The 2026-07-02 `hope_forehand:v4` / `hope_backhand:v4` artifacts contain canonical `motion.npz` but were rejected because they still face +Y; future verified uploads are pending | G05 registry-backed training |
| `hope_training/motions/preprocessed/*.npz` or `hope_training/whole_body_tracking/artifacts/**/*.npz` | Local reference swing clips passed with `motion_file=...` / `motion_file_2=...` | Generated by GVHMR/GMR/`csv_to_npz.py` or restored from an agreed internal artifact store | No-WandB smoke tests or local G05 training |
| `hope_training/motions/preprocessed/hope_forehand_v5.npz` and `hope_backhand_v5.npz` | Optional R15 ablation clips only; not product/default train or replay inputs | Team RunPod shared asset path `/workspace/shared/motions/hope_{forehand,backhand}_v5.npz` | G05 R15 v5 ablation |
| `hope_training/motions/preprocessed/hope_forehand_hopex.npz` and `hope_backhand_hopex.npz` | Corrected HOPE +X reference swing clips for `task=HOPEPingPong*`, passed with `motion_file=...` / `motion_file_2=...` | Generated locally from the 2026-07-02 v4 motions with `scripts/reground_hope_frame.py`; ignored, not redistributable | Training (HOPEPingPong) |
| `hope_training/whole_body_tracking/artifacts/bank_exam/<family>/` | Family-specific schema-v3 train/exam banks, immutable schedule JSON and raw evaluator scorecards | Regenerate from the exact ignored motion pair with `gen_stage1_questions.py` and `materialize_bank_exam_schedule.py`; never substitute another family's bank | G05/G06 BankExam |
| RunPod `/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/` | Audited runtime-order schema-2 v4rg motions and same-family schema-v3 fresh train/exam banks | Generated on Pod 1 from the recorded legacy source motions using `migrate_motion_kinematics.py --target-body-order configs/a3_runtime_body_order.txt`; tracked manifest: [`configs/phase1_fresh_v3_asset_manifest_20260711.json`](../../configs/phase1_fresh_v3_asset_manifest_20260711.json) | G05 fresh formal training |
| RunPod `/workspace/codexschema/phase1_fresh_20260711/assets/legacy/{v4rg,swing}/` | Recovered train banks for the controlled M2f/M3c face-pairing continuations only | Reproduced from each exact historical motion family; manifests prove byte-identical exam-parameter recovery and train/exam disjointness | G05/G06 legacy causal diagnosis |
| RunPod `/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/` | Checkpoint-specific judge worker state and preserved failed preflights | Generated by `phase1_checkpoint_curve_worker.py`; restore together with each run's ignored `judge/` directory | G05/G06 milestone curves |
| RunPod `/workspace/codexschema/phase1_signed_face_rescue_20260713/source_50c49e5.bundle` and `source_50c49e5_git_evidence.txt` | Disaster-recovery source and clean-worktree proof for the historical epoch-1 signed-face v6 checkout | Private runtime evidence; bundle/evidence SHA-256 `2a794e2c...0a39e` / `12dc839f...c99`, bundle advertises `50c49e58a9413ec6ac1c3ed2565d9a78acdb5e64` and requires base `882fea4285f0cf9a97ba79d79ae8af31d26ea1ed` | G05 signed-face boot diagnosis; not retry permission |
| RunPod `/workspace/codexschema/phase1_signed_face_rescue_20260713/{control/v6,control/v6r1,runs/l1,l1_checkpoint_audit.jsonl,b_kill_action.txt,d_timeout_diagnostic.txt}` | Immutable foreign v6 control, A/B/C terminal, D timeout and never-launched v6r1 validator evidence | Copy the complete exact evidence root from Pod1 or the reviewed local snapshot `/private/tmp/pod1_v6_foreign_20260713/`; foreign config/launcher SHA-256 `97779cee...eebf2` / `9463f228...85052`, checkpoint-audit SHA-256 `62076758...d354`. Do not edit or flatten paths; v6r1 is superseded and v6r2 is source-only | G05 signed-face boot diagnosis; no finalizer/retry |
| RunPod `/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_exam_bank_rebind_v1/` | Accepted signed-face exam E2 rebound bank and report-last receipt | Bank/report are 63,643/18,795 bytes with SHA `60e1a7ad...d1ca` / `dd4332ed...ad0`; tracked ledger `configs/phase1_signed_face_exam_bank_rebind_results_20260714.json` binds full receipts. Restore both together before `run_phase1_signed_face_exam_k100.md`; this does not restore a schedule or judge activation | G06 signed-face paper input |
| RunPod `/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/` | Exact-bank-bound K100 schedule and paper-only activation | Materialized once on Pod1. Copy the complete no-clobber root and verify against `configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json`: schedule file/semantic/order `f2777dcd...1ca` / `3ca4bdba...3365` / `09f778f2...bd0`, activation file/content `e0125b0e...bb4` / `533beb03...3d8`. Never delete/re-consume v1; a schedule without the activation is partial | G06 signed-face paper activation; never judge/training permission by itself |
| RunPod `/workspace/codexschema/{staging/nohope_signed_face_source_72418ff_adf1c0be8a1e066f80dc96011c799d6eab99cc5e610a08d9234d6a6af4f1efc3.bundle,phase1_signed_face_rescue_20260713/control/v8,phase1_signed_face_rescue_20260713/runs/l1/phase1_signed_face_l1_v8_D_fresh_guidance_seed3}` | Foreign v8 source/control and D second pre-contract boot-timeout evidence | Bundle is 2,640,446 bytes SHA `adf1c0be...f1efc3`; exact config/launcher/artifact receipts are in `configs/phase1_signed_face_v8_d_boot_failure_20260714.json`. Restore no-clobber and keep the D failure untouched; it is diagnostic evidence, not retry permission | G05/G06 signed-face boot root-cause |
| Future Pod1 `/workspace/codexschema/phase1_signed_face_cd_l1_20260714/` plus detached source `/workspace/codexschema/nohope_signed_face_cd_l1_4467d79/` | New C2/D2-only provenance-complete signed-face L1 claims, logs, adjacent contracts and `model_24.pt` on GPU1/GPU2 | Not materialized at source-gate time. Restore/copy the ignored A3 asset from exact `6d93bcb...480b` as 46 files / 15,378,264 bytes / tree SHA `0137f59b...26c6`, never as symlinks. If launched, preserve the entire no-clobber namespace and both read-only control files together; see [`run_phase1_signed_face_cd_l1.md`](run_phase1_signed_face_cd_l1.md) | G05 L1 provenance smoke only; no judge/L2/robot |
| Venue `$BALLFIT_DATA_ROOT/analysis/segments/strikes.json` (current Pod copy: `/workspace/yikang/latest_data/analysis/segments/strikes.json`) | Detected real racket contacts used for the conservative A-B-A next-task timing audit | Generated from the 2026-07-03 venue mocap pipeline; current file SHA-256 `6ad3c45959c94b6fdd4033130403c32e0f1b612a138738c12afa43a58f752841` | G05 continuous-timing design; G03 ball fit |
| RunPod historical M3c/M2f `model_16999.pt` checkpoints | Warm starts for the four-arm face-pairing comparison; never fresh-formal inputs | Existing ignored run trees under `/workspace/franco/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/` | G05/G06 legacy causal diagnosis |
| `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` | Generated Isaac A3 ping-pong URDF asset | Rebuilt from tracked `agi/URDF/A3T2.5-URDF-std-pingpang/` using `scripts/prepare_a3_isaac_asset.py` | G04/G05 |
| Generated policy artifacts such as `hope_training/policies/*.onnx` | Exported policies for local eval/deploy handoff | Produced by `scripts/play.py` or training/eval export; store metadata in G05/G07 | G05/G07 when a specific policy is accepted |

## 2026-07-24 动作终审输入根恢复

当前本地真路径是 `vendor_assets/motion_finalize_20260724/`。以下命令只恢复 Pod2 上的 ready、
四个 SHADOW 源和四个 GMR 源；`evidence/face_scopes/` 是本轮换面求解审计在该输入根上生成的
scope-specific 证据，不从旧 full-body/syn probe 目录继承。目录尚未进入正式制品系统，恢复时
必须 no-clobber：

```bash
MOTION_SYNC_LOCAL=vendor_assets/motion_finalize_20260724
POD2='root@162.43.172.181'
POD2_SSH='ssh -p 13146 -i ~/.ssh/id_ed25519_runpod'

test ! -e "$MOTION_SYNC_LOCAL"
mkdir -p "$MOTION_SYNC_LOCAL/ready" \
  "$MOTION_SYNC_LOCAL/sources" \
  "$MOTION_SYNC_LOCAL/gmr_sources"

rsync -a --rsync-path='nice -n 19 rsync' -e "$POD2_SSH" \
  "$POD2:/workspace/codexschema/firstframe_20260723/designed_ready/canonical_ready_v1.npz" \
  "$POD2:/workspace/codexschema/firstframe_20260723/designed_ready/canonical_ready_v1.json" \
  "$MOTION_SYNC_LOCAL/ready/"

rsync -a --rsync-path='nice -n 19 rsync' -e "$POD2_SSH" \
  "$POD2:/workspace/codexschema/franco_pipeline_20260722/fk/SHADOW_fh_loop_yaw147.npz" \
  "$POD2:/workspace/codexschema/franco_pipeline_20260722/fk/SHADOW_franco_bh_loop_c.npz" \
  "$POD2:/workspace/codexschema/franco_pipeline_20260722/fk/SHADOW_bh_block_yaw80.npz" \
  "$POD2:/workspace/codexschema/franco_pipeline_20260722/fk/SHADOW_s0_yaw72.npz" \
  "$MOTION_SYNC_LOCAL/sources/"

rsync -a --rsync-path='nice -n 19 rsync' -e "$POD2_SSH" \
  "$POD2:/workspace/codexschema/franco_set_cert_20260722/inputs/franco_forehand_loop.diagnostic_cohort_median_betas.grounded.pkl" \
  "$POD2:/workspace/codexschema/franco_set_cert_20260722/inputs/franco_backhand_loop_c.aa0c86fd3509.se2.gmr.pkl" \
  "$POD2:/workspace/codexschema/franco_set_cert_20260722/inputs/franco_backhand_block.diagnostic_cohort_median_betas.grounded.pkl" \
  "$POD2:/workspace/codexschema/franco_set_cert_20260722/inputs/static_backhand_high_press.exact_franco_donor_betas.gmr.pkl" \
  "$MOTION_SYNC_LOCAL/gmr_sources/"

(cd "$MOTION_SYNC_LOCAL" && shasum -a 256 -c <<'SHA256'
cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0  ready/canonical_ready_v1.npz
95be5bc32150f7dea8a4eed41bf591acd95588624e2f6f3f508f39f2c6c9e227  ready/canonical_ready_v1.json
faa8df8c552e4bd99134cefe5457f86e646499ff12737db160fa43eec763dcc1  sources/SHADOW_fh_loop_yaw147.npz
d5338168e692c8a2c19fbfac8aeb56653fa79a1f45cebc6803a460835fbc1fba  sources/SHADOW_franco_bh_loop_c.npz
55870b981584a458bfd479171046445845cb74171618b71338fd9dc9f66a5fe0  sources/SHADOW_bh_block_yaw80.npz
2cd32da1864fa686aff544d29a84e988b91911503ae7f7680601f93345378c01  sources/SHADOW_s0_yaw72.npz
d75d8f17e7b7cad3f06a47fbf01bdec194116996a155c6ab61e9d3eb0d84f6c8  gmr_sources/franco_forehand_loop.diagnostic_cohort_median_betas.grounded.pkl
0dd981a6d29c0c5321c905d1591a59fbb79763de6e43d92d4d76aefdc29ff48b  gmr_sources/franco_backhand_loop_c.aa0c86fd3509.se2.gmr.pkl
a4d92b68b9d1fc1185d2cfc87a233bf5ef974538a2d2a3ee7faa58593aace9d4  gmr_sources/franco_backhand_block.diagnostic_cohort_median_betas.grounded.pkl
2dbe61e80af7187e9524b63095887287d2fd6aa615cbe9b712f68ea4dfc70edc  gmr_sources/static_backhand_high_press.exact_franco_donor_betas.gmr.pkl
SHA256
)
```

恢复成功只证明输入字节一致，不证明十件 candidate 已构建或通过 Agibot MuJoCo、grounded
torque/contact、行为、恢复和 registry 门。不得从该根直接启动训练。

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

For a direct source build, restore the vendor SDK under the active tree (the
destination is git-ignored) and let the setup script fetch/locate ONNX Runtime:

```bash
rsync -a \
  vendor_assets/agibot/a3_deploy_example_full/thirdparty/unitree_sdk2/ \
  agi/a3_deploy_example/thirdparty/unitree_sdk2/
source agi/a3_deploy_example/setup_a3_env.sh
```

Verified 2026-07-13 from this standard handoff: the complete SDK canonical manifest contains 863
entries and has SHA-256 `e8f808e92b9b73cbcde2803b34ef48bb329941e40cd20f7b73282a1195588c13`;
`lib/x86_64/libunitree_sdk2.a` is `27,351,696` bytes with SHA-256
`93ebabb2eca346892f23b9f78ece974a48091b44d745053e6911d3e294f74ec7`. A remote copy is accepted
only after a no-clobber staging manifest matches the local source. Preserve an incomplete
destination under another ignored name instead of overwriting it in place.

If the handoff uses a different internal layout, locate its `unitree_sdk2`
root and preserve that directory name at the destination. Do not copy the
vendor bundle into git.

Tracked deploy code lives separately at:

```text
agi/a3_deploy_example/                  # active working ping-pong deploy example (tracked)
agi/code_deployment/a3_deploy_example/  # older vendor reference subset
```

The deployed policy artifact (`models/model_p4_deployparity.onnx` under the deploy package build output `dist/`) is generated and git-ignored; rebuild it via `agi/a3_deploy_example/scripts/build_a3_deploy_pkg.sh`.

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

Detached training worktrees also omit this ignored tree. For the Pod1 signed-face L1 funnel, the
reviewed restore source is the clean `6d93bcb16c422a2f42748c2dc99432559653480b` checkout and the
destination is the clean detached `882fea4285f0cf9a97ba79d79ae8af31d26ea1ed` checkout. Only while
no trainer uses either worktree, restore without deleting or overwriting an existing destination:

```bash
SOURCE=/workspace/codexschema/nohope/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3
DEST_PARENT=/workspace/codexschema/nohope_signed_face_rescue_882fea4/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets
test ! -e "$DEST_PARENT/agibot_a3"
mkdir -p "$DEST_PARENT"
cp -a "$SOURCE" "$DEST_PARENT/"
test -z "$(git -C /workspace/codexschema/nohope_signed_face_rescue_882fea4 status --porcelain)"
```

The accepted tree contains exactly `46` regular files and `15,378,264` file bytes; its canonical
`{relative_path,bytes,sha256}` manifest SHA is
`0137f59b1fe45e7d5f8fa731bedca905f5466bc98e8d1354081fe071d60426c6`. The v6 launcher recomputes
that digest for both source and destination, requires the destination to remain Git-ignored, and
rejects symlinks, special files, missing files and extra files before creating a run claim.

For the lean P1 source `/workspace/codexschema/nohope_p1_077e70c`, do not use the manual `cp` recipe
above. The conditional and V1+V2/base-decel queue rows bind the same accepted tree and donor
`/workspace/codexschema/nohope@6d93bcb16c422a2f42748c2dc99432559653480b`. Use the reviewed
`prepare-source-assets` command in [the lean queue operation](run_lean_training_queue.md#新-source-的-ignored-a3-资产先显式水合).
It verifies all 46 accepted files plus the 43 unique URDF-referenced meshes, publishes with
no-replace semantics, and writes a source-external receipt consumed before any science claim. A
timeout or preserved staging directory is not permission to replay the copy.

### Optional: reuse one successful A3 conversion during a training sprint

`HOPE_AGIBOT_A3_USD_PATH` may point directly to `model.usd` from a successful Isaac conversion.
That file depends on sibling files under `configuration/`, so restore the **entire conversion
directory**; copying `model.usd` alone is incomplete. Keep this generated runtime cache outside Git
at a stable path shared by the training processes:

```bash
SOURCE=/path/to/one/successful/a3_conversion
DEST=/workspace/codexschema/runtime_assets/a3_preconverted

test -f "$SOURCE/model.usd"
test -d "$SOURCE/configuration"
mkdir -p "$DEST"
cp -a "$SOURCE/." "$DEST/"
export HOPE_AGIBOT_A3_USD_PATH="$DEST/model.usd"
```

With the variable set, training loads the pre-converted scene and avoids starting a separate robot
converter in every process. If the variable is unset or empty, the existing robot-description
importer remains the fallback. This cache is an ignored runtime optimization, not a Git asset; this
simple restore path does not require a hash or receipt.

### Signed-face epoch-1 v6 and superseded v6r1 private evidence

The epoch-1 source commit and production v6 control were created after the tracked `882fea4` source
gate and are not currently reachable from a normal clone. The reviewed local backup is
`/private/tmp/pod1_v6_foreign_20260713/`; the canonical RunPod restore root is
`/workspace/codexschema/phase1_signed_face_rescue_20260713/`. Preserve the directory layout and copy
without replacing an existing file. Minimum hashes are:

```text
source_50c49e5.bundle
  2a794e2c0f9c4adefd5194d94c404bbdf137cf5368f9c2c2aedf2bc50cc0a39e
source_50c49e5_git_evidence.txt
  12dc839fc76217cd714cfd8ef8f61c42c7e8231cce2b218f34fd42da4a008c99
control/v6/phase1_signed_face_rescue_funnel_prereg_v6_20260713.json
  97779cee50819ae6ff34d62f6f3c2aed6b13c360b1bf7f0d075aec1f07feebf2
control/v6/run_phase1_signed_face_rescue_funnel.py
  9463f228b26e0a2af548dc749b42428cc3dd1a6379c9d11448e854cfa9d85052
l1_checkpoint_audit.jsonl
  620767581cb47dda23843822129b09c66507b0cdc887e283d619d4b51fb0d354
b_kill_action.txt
  cf619541487f6fb87182df0dd33b73e09c79ab9b1a30037a40901208bcddcafe
d_timeout_diagnostic.txt
  ae7de7a37329eddfaa264adb99f9a38c0b7ead33579d5b1dac0d63f4a74b5a0c
```

The preregistered bundle path is immutable. If
`/workspace/codexschema/phase1_signed_face_rescue_20260713/source_50c49e5.bundle` is missing, restore
that exact file from the reviewed local evidence (transfer the backup to the restore host first if
needed), then verify it; **do not edit the manifest to point at another convenient path**:

```bash
LOCAL_EVIDENCE=/private/tmp/pod1_v6_foreign_20260713/source_50c49e5.bundle
EXPECTED_BUNDLE=/workspace/codexschema/phase1_signed_face_rescue_20260713/source_50c49e5.bundle
EXPECTED_SHA=2a794e2c0f9c4adefd5194d94c404bbdf137cf5368f9c2c2aedf2bc50cc0a39e

test "$(sha256sum "$LOCAL_EVIDENCE" | awk '{print $1}')" = "$EXPECTED_SHA"
if test ! -e "$EXPECTED_BUNDLE"; then
  mkdir -p "$(dirname "$EXPECTED_BUNDLE")"
  cp --no-clobber --preserve=mode,timestamps "$LOCAL_EVIDENCE" "$EXPECTED_BUNDLE"
  chmod 0444 "$EXPECTED_BUNDLE"
fi
test -f "$EXPECTED_BUNDLE"
test ! -L "$EXPECTED_BUNDLE"
test "$(sha256sum "$EXPECTED_BUNDLE" | awk '{print $1}')" = "$EXPECTED_SHA"
```

Run `git bundle verify source_50c49e5.bundle`; it must advertise
`50c49e58a9413ec6ac1c3ed2565d9a78acdb5e64` and require
`882fea4285f0cf9a97ba79d79ae8af31d26ea1ed`. The bundle is recovery material, not permission to
rewrite an active checkout. If `/workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5`
already exists, require that exact HEAD and an empty `git status --porcelain`; never fetch/switch or
restore files while a trainer uses it. The historical v6/v6r1 evidence and every outer-evidence SHA are frozen in
[`run_phase1_signed_face_rescue_funnel.md`](run_phase1_signed_face_rescue_funnel.md).

Tracked v6r2 is only a source validator correction and needs no private runtime restore. It must not be copied to
the Pod as a launcher: it has no runtime preflight, command reconstruction, process inspection, signal, launch or
finalizer consumer. Any future runtime attempt requires a new boot-root-cause preregistration.

The production and tracked controls intentionally collide on the same v6 `manifest_id` while binding
different source/config/launcher bytes (`50c49e5` + `97779cee...` / `9463f228...` versus tracked
`882fea4`). Restore and consume them by full SHA and path, never by manifest ID alone.

### Signed-face foreign v8 and exam E2 private evidence

The later v8 lane is independent of v6/v6r1. Restore its 2,640,446-byte source bundle only at the frozen
`/workspace/codexschema/staging/nohope_signed_face_source_72418ff_adf1c0be8a1e066f80dc96011c799d6eab99cc5e610a08d9234d6a6af4f1efc3.bundle`
path and require SHA `adf1c0be8a1e066f80dc96011c799d6eab99cc5e610a08d9234d6a6af4f1efc3`; it must advertise
`72418fff817d2d9beb9f764562b5a28e82a13044`. Preserve `control/v8` and the complete D claim directory.
The exact failure/state/launch-contract/run-log receipts and the directly bound C predecessor receipt are in
`configs/phase1_signed_face_v8_d_boot_failure_20260714.json`. This restore is for boot diagnosis only: automatic
retry is forbidden, and no file in the old claim may be deleted or overwritten.

Restore the accepted exam E2 bank and `rebind_report.json` together under
`.../assets/schema3_exam_bank_rebind_v1/`, using the full receipts in
`configs/phase1_signed_face_exam_bank_rebind_results_20260714.json`. The report must remain the completion
marker. This root does not contain the still-missing immutable schedule or paper activation.

### Signed-face v6/v8 read-only boot postmortem evidence

The tracked result ledger
`configs/phase1_signed_face_boot_root_cause_results_20260714.json` depends on four ignored local
postmortem files. They are preserved under the Dropbox-backed ignored `vendor_assets/` root rather
than ephemeral `/private/tmp`. Restore them at the exact repository-relative no-clobber paths below;
verify size and SHA before using the ledger to reproduce any byte-level statement:

| Local path | Bytes | SHA-256 |
| --- | ---: | --- |
| `vendor_assets/phase1_signed_face_boot_root_cause_20260714/pod1_v6v8_readonly_diagnosis_20260714.md` | 5,955 | `b54cb06a50bfa5f0994b1768beb995577b03a360eb4dfaefca13959c1c2d76af` |
| `vendor_assets/phase1_signed_face_boot_root_cause_20260714/pod1_v6v8_attempt1_inventory_20260714.txt` | 157,048 | `b18935129364cb342a4d3989caf56821bc0f5cb3dbae79c9a409e26d0e21cc1d` |
| `vendor_assets/phase1_signed_face_boot_root_cause_20260714/pod1_v6v8_exact_evidence_20260714.tar` | 8,509,440 | `29dabc9e23fc7f4d4f1713a75bb9bc3be20009b19f90f51838a439d65e0283a6` |
| `vendor_assets/phase1_signed_face_boot_root_cause_20260714/pod1_v6v8_attempt3_system_20260714.txt` | 60,056 | `02b78e2d4db982145e57d9bcbe82768799b2756b21636a031052c7a1b30d1e25` |

The tar has exactly 47 entries: all eight v6/v8 A/B/C/D run evidence directories plus their eight
Kit logs. Extract only to a new local directory and preserve paths. This material is read-only
postmortem evidence. It does not authorize Pod access, shared-memory cleanup, process launch,
signal, retry training, judge, deployment or hardware. The paired diagnostic prereg is design-only.

6. Restore the motion-retargeting clones (both git-ignored, absent on a fresh clone) when working on the motion pipeline:

```bash
# Video -> SMPL-X (observed pin 6ec3ca3)
git clone https://github.com/zju3dv/GVHMR.git hope_training/GVHMR
# SMPL-X -> robot (observed pin bb1bbe4)
git clone https://github.com/YanjieZe/GMR.git hope_training/GMR
# Each installs into its own conda env (python=3.10):
#   pip install -e .
```

See [reimplement.md](../../reimplement.md) steps 9-12 for the full per-env install procedure, pins, CSV conversion, local `.npz` generation, and replay checks.

2026-06-25 local status: both ignored clones were restored at the observed pins above. GMR was installed
editable into the existing `hope-motion-py310` conda env and passed an import check. GVHMR was cloned
but not installed because its tracked requirements pin CUDA 12.1-era PyTorch/PyTorch3D wheels; resolve
that compatibility issue before installing on an RTX 5090 / Blackwell host.

7. Add the license-gated model assets the clones depend on (you must accept each license yourself; not redistributable):

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

8. Provide the reference swing clips for `task=HOPEPingPong`:

Registry defaults exist for internal shared runs, but no-WandB smoke tests and locally generated references should use explicit `.npz` paths under an ignored folder. The current accepted local clips are the corrected HOPE +X versions:

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

Then pass them explicitly:

```bash
cd hope_training/whole_body_tracking
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo \
  motion_file=../motions/preprocessed/hope_forehand_hopex.npz \
  motion_file_2=../motions/preprocessed/hope_backhand_hopex.npz \
  logger=tensorboard
```

The optional R15 v5 clips are restored separately and must stay command-line only:

```bash
mkdir -p hope_training/motions/preprocessed
scp -P 18333 -i ~/.ssh/id_ed25519_runpod \
  root@162.43.172.171:/workspace/shared/motions/hope_forehand_v5.npz \
  root@162.43.172.171:/workspace/shared/motions/hope_backhand_v5.npz \
  hope_training/motions/preprocessed/
```

On the RunPod itself, copy from `/workspace/shared/motions/` into your own `/workspace/<name>/nohope/hope_training/motions/preprocessed/`. Use the R15 override command in `run_training.md`; do not edit `cfg/train.yaml`, `cfg/play.yaml`, or task YAML defaults to point at v5.

The maintainer `hope_forehand`/`hope_backhand` registry artifacts live in a private, org-scoped WandB "Motions" registry and cannot be redistributed. Override `registry_name=...` only when selecting a different registry clip. As of 2026-07-02 the registry aliases below are not accepted for long training until future verified artifacts are uploaded; the local v5 R15 clips are not registry/default replacements:

```text
dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_forehand:latest
  -> BerkeleyPingPong/csv_to_npz/hope_forehand:v4, manifest: motion.npz, REJECTED: frame0_yaw=82.03 deg, +Y-dominant strike velocity
dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_backhand:latest
  -> BerkeleyPingPong/csv_to_npz/hope_backhand:v4, manifest: motion.npz, REJECTED: frame0_yaw=85.92 deg, +Y-dominant strike velocity
```

> Make your own motions instead: run GVHMR (video -> SMPL-X) -> GMR (`--robot agibot_a3`) -> `scripts/csv_to_npz.py --robot agibot_a3 --output_file ../motions/preprocessed/<name>.npz`. The converter now applies HOPE +X alignment by default for Agibot A3 before local save/upload; run `scripts/check_motion_target_alignment.py` before training. Add `--upload_wandb` only if you also want to upload the resulting `.npz` to your own WandB Motions registry. Full steps: [reimplement.md](../../reimplement.md) steps 9-12 and [run_training.md](run_training.md).

7. Regenerate a family-specific schema-v3 BankExam bundle when the ignored
motion pair or bank artifact is absent. Use the same clip paths, sampling
support, anchor, stroke budget and generator source for both splits; only
`--split` and `--out` differ. Example:

```bash
WBT=hope_training/whole_body_tracking
OUT="$WBT/artifacts/bank_exam/<family>"
mkdir -p "$OUT"
python "$WBT/scripts/gen_stage1_questions.py" \
  --clip forehand:/abs/path/forehand.npz \
  --clip backhand:/abs/path/backhand.npz \
  --stroke-budget-clips /abs/path/v4rg_forehand.npz /abs/path/v4rg_backhand.npz \
  --n 512 --seed 0 --split train --out "$OUT/s1_<family>_v3_train.npz"
python "$WBT/scripts/gen_stage1_questions.py" \
  --clip forehand:/abs/path/forehand.npz \
  --clip backhand:/abs/path/backhand.npz \
  --stroke-budget-clips /abs/path/v4rg_forehand.npz /abs/path/v4rg_backhand.npz \
  --n 512 --seed 0 --split exam --out "$OUT/s1_<family>_v3_exam.npz"
python "$WBT/scripts/materialize_bank_exam_schedule.py" \
  --exam-bank "$OUT/s1_<family>_v3_exam.npz" \
  --per-clip-quota 10 --schedule-seed 0 --hold-range 0 100 \
  --output "$OUT/canary_q10_seed0.schedule.json"
```

Do not copy an exam bank between M2/v4rg, M3f/swing or G1/swingsyn: schema-v3
motion SHA/order/frame validation must fail on that substitution. For a fresh
exact-lineage model, retain both split files and record their common
`source_family_sha256`; a historical canary may use the exam split only but is
permanently diagnostic.

8. Keep the generated schedule and every raw JSON/CSV ledger beside the bank
or in another ignored artifact directory. A schedule is small but result-bound;
do not hand-edit it. Restore by copying the complete bank/schedule/ledger
bundle from the recorded artifact location or by rerunning the commands above.

9. Rebuild the generated A3 Isaac asset when missing or stale:

```bash
python3 scripts/prepare_a3_isaac_asset.py --force
python3 scripts/prepare_a3_isaac_asset.py --check
```

The generated directory is ignored because it duplicates tracked Agibot source meshes.

10. Keep generated training artifacts out of git:

```text
hope_training/whole_body_tracking/logs/
hope_training/whole_body_tracking/outputs/
hope_training/whole_body_tracking/artifacts/
hope_training/whole_body_tracking/wandb/
hope_training/motions/
```

If a generated policy, motion, or dataset is required to reproduce a result, record where it lives, how it was produced, and whether it is in WandB, local ignored storage, or a future artifact store.

### Franco/v6/v7 private motion-video intake (2026-07-11)

Restore the three original folders exactly under the manifest's default root:

```text
${HOME}/Downloads/Franco/
${HOME}/Downloads/v6_dang/
${HOME}/Downloads/v7_dang/
```

The ten MP4 files are user recordings and must not be added to git or a public
artifact registry. Verify the complete byte and media contract before any
conversion:

```bash
python3 scripts/audit_motion_video_intake.py \
  --manifest configs/motion_video_intake_20260711.json \
  --source-root "$HOME/Downloads"
```

The verified private Pod1 staging root is
`/workspace/codexschema/motion_video_intake_20260711/raw/`. Re-run the same
audit there before processing. The serial GVHMR queue writes only under the
separate staging/GVHMR output trees; it does not edit the Phase-1 training
checkout. The 2026-07-11 run used PID/PGID `1383735`, GPU physical index 1,
memory gate `19000 MiB`, GVHMR commit `6ec3ca3`, and the ordered worklist
Franco pilot/family first, then v6/v7. It completed 10/10 structural outputs;
the durable small evidence manifest is
`configs/motion_video_gvhmr_results_20260711.json`. The exact historical source is retained only as
`docs/experiments/archive/run_motion_video_gvhmr_queue_20260711.py.gz`; decompressing it reproduces
the tool bytes/SHA recorded by the result ledger, but it is deliberately not an active `scripts/`
entrypoint. Re-running the old queue is not currently authorized. A future reproduction must create a
new versioned preregistration and no-clobber evidence root instead of directly executing the archived source.

The queue validates all inputs, requires a clean GVHMR worktree, hashes the
complete checkpoint/body-model tree and motion Python environment, and records
manifest/tool/auditor/GVHMR/input/output SHA bindings. It refuses output-name
aliases and stops on the first failure. The `19000 MiB` memory check is a
pre-launch sampling gate, not a reservation; keep monitoring concurrent jobs.
Every `hmr4d_results.pt` must also pass expected-frame, SMPL-shape and finite
structural checks. That pass permits the next preprocessing item but is not a
visual-quality or safety promotion. A successful `hmr4d_results.pt` is
only a reconstruction artifact; do not promote it to an A3 motion until the
GMR/schema-2/L0/self-collision/table-net gates in
`docs/research/motion_library_topp_recovery_2026-07-11.md` pass.

The gzip file is the byte-exact source archive bound by that result manifest.
The only active legacy entrypoint rejects every schema-version 2 intake, while the secure queue accepts only
the two exact committed S0/M0 preregistrations; changing an intake id cannot bypass those gates.

### v12/static/lateral private motion-video intake (2026-07-13)

Restore these folders without renaming or moving the source files:

```text
${HOME}/Downloads/v12/
${HOME}/Downloads/static/
${HOME}/Downloads/motion/
```

Verify all seven exact videos before any copy or processing:

```bash
python3 scripts/audit_motion_video_intake.py \
  --manifest configs/motion_video_intake_20260713.json \
  --source-root "$HOME/Downloads"
```

Schema 2 records v12 forehand/backhand blocks, one backhand high-press fifth
action and four lateral-locomotion teacher candidates. It rejects a locomotion
teacher mislabeled as a stroke. The two GVHMR-only queues have now completed on Pod1:
S0 contains only `static/pai.mp4`; M0 contains only the four lateral candidates.
They bind nominal air-swing event/ready
windows and the exact runtime closure in separate versioned private roots. Follow
[`run_motion_video_gvhmr_prereg.md`](run_motion_video_gvhmr_prereg.md) for the
Pod copy, second byte/media audit and no-clobber execution record. Never append
these files to the accepted 2026-07-11 evidence tree, and do not interpret the
GVHMR authorization as permission for GMR, simulator, RL or hardware. The six Franco
motions already have exact GVHMR/GMR ledgers and are not rerun; v12 remains a later
Jiayi-route comparison and is not authorized by S0/M0. S0 and M0 are independent
batches, while the four lateral assets are candidates rather than repeated seeds. Restore the complete
S0/M0 state roots plus the five output PT files and verify every SHA in
`configs/motion_video_gvhmr_s0_m0_results_20260713.json`; copying only the PT files loses the accepted lineage.
The separate [post-GVHMR consumer](run_motion_post_gvhmr_exact.md) also needs the exact 2026-07-11
`canonical_betas.json` and its `materialization_manifest.json`. Its two preregistrations bind every
output/binding/audit plus the tracked summary and donor artifacts; runtime handoff and canonical-beta have both
been consumed. Missing evidence must fail closed rather than be reconstructed from matching basenames.

The 2026-07-11 CPU-only GMR diagnostic then completed 10/10 under a separate
serial queue. Restore and verify the exact source bundle before reproducing it:

```bash
git -C /path/to/GMR bundle verify \
  /private/ignored/motion_video_intake_20260711/gmr_provenance/GMR_aabea2e.bundle
git -C /path/to/GMR bundle list-heads \
  /private/ignored/motion_video_intake_20260711/gmr_provenance/GMR_aabea2e.bundle
test "$(git -C /path/to/GMR rev-parse HEAD)" = \
  aabea2eee4be4bc16d4be17dac5ffa85e5a31539
test -z "$(git -C /path/to/GMR status --porcelain --untracked-files=all)"

python3 scripts/run_motion_video_gmr_queue.py \
  --manifest configs/motion_video_gvhmr_results_20260711.json \
  --gmr-root /path/to/GMR \
  --gmr-bundle /private/ignored/motion_video_intake_20260711/gmr_provenance/GMR_aabea2e.bundle \
  --python /path/to/hope-motion-py310/bin/python \
  --output-root /private/ignored/motion_video_intake_20260711/gmr_outputs \
  --state-dir /private/ignored/motion_video_intake_20260711/gmr_state
```

The queue forces CPU execution, keeps frame-zero warm-up enabled, requires a
clean GMR checkout, binds the bundle/commit/entrypoint/Python/tool/input/output
hashes and stops on the first failure. The tracked small result ledger is
`configs/motion_video_gmr_results_20260711.json`. All outputs intentionally
retain per-video GVHMR betas and therefore remain
`body_shape_contract=diagnostic_video_betas`, `formal_eligible=false`.
Do not use them as canonical-betas motion assets. The first deeper replay also
found about 8 cm of floor penetration, so root/ground calibration and repeated
collision/dynamics/table-net gates are mandatory before schema-2 conversion.

The body-shape-normalized rerun is a separate queue and lineage. The ten
materialized cohort-median-beta PTs are bound by
`configs/motion_video_canonical_gmr_prereg_20260711.json`; the exact GMR loader
at clean commit `aabea2e` has SHA-256 `2737f472...5de2` and consumes
`smpl_params_global.betas[0][:10]` without zero padding. The neutral SMPL-X
NPZ used by that loader has SHA-256 `37602144...992`. Do not point the legacy
`diagnostic_video_betas` queue at these PTs. The dedicated queue command shape
is:

```bash
PLAN=configs/motion_video_canonical_gmr_prereg_20260711.json
PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
CUDA_VISIBLE_DEVICES= PYTHONPATH=scripts \
  /path/to/hope-motion-py310/bin/python \
  scripts/run_motion_video_canonical_gmr_queue.py \
  --plan "$PLAN" --expected-plan-sha256 "$PLAN_SHA"
```

The committed v1 plan is already consumed: its no-clobber output/state roots
exist on Pod1, so the command above is documentary and must fail if repeated.
A true reproduction must preregister new versioned output/state roots and bind
the new plan SHA; never delete or overwrite the v1 evidence. The accepted v1
run was CPU-only, completed 10/10 in 48.7 s, and every result passed 30 Hz,
31-DoF finite structure plus frame-zero warm-up (`16--29` rounds, final
`max|dq| < 1e-4`). The small binding ledger is
`configs/motion_video_canonical_gmr_results_20260711.json`. These outputs are
still uncalibrated diagnostics and must take their own no-clobber grounding,
dense collision, dynamics and table/net gates before schema-2.

The new S0/M0 canonical-beta outputs use a separate exact-GMR consumer; do not
append them to the consumed ten-asset v1 root. Restore the two completion
manifests (`964a7333...f1be3` / `5cef05f7...71a65`), their exact five PTs and
both byte-identical donor `canonical_betas.json` copies at the absolute paths
in `configs/motion_exact_gmr_s0_prereg_20260714_v2.json` and
`configs/motion_exact_gmr_m0_prereg_20260714_v2.json`. The shared runtime
contract additionally requires the ignored GMR commit **and tree OID**, exact
`a3_mocap.xml`, exact SMPL-X-to-A3 mapping, neutral SMPL-X NPZ, recovery bundle,
Python/pip closure and independently parsed retarget joint/body/site order. A
2026-07-14 follow-up recovered all 16 exact file/runtime facts. Direct
`a3_mocap.xml` has 31 hinge joints and 32 bodies in the separately bound order,
but its site inventory is exactly empty and `left_foot/right_foot` are absent;
never copy canonical vendor sites into that retarget inventory. Both batch-plan
`static` calls now pass. Restore every absolute binding from
`configs/motion_s0_m0_exact_gmr_runtime_20260714_v2.json` and its package snapshot
`configs/motion_s0_m0_exact_gmr_pip_freeze_56b0f8af_v2.txt`, then follow
[`run_motion_s0_m0_exact_gmr.md`](run_motion_s0_m0_exact_gmr.md) for evidence verification.

The authoritative Pod1 exact-GMR v2 roots are already consumed. S0/M0
`completion_manifest.json` SHA-256 is respectively
`a762d6df22d4ffdcfc323425c234a0d3b910022d17a1541fa48ab7fe700d1a23` and
`fdd60fcfdc7290677aa51ec7804278568a267e239de548cdb623d0565dac396e`. Restore each complete
`exact_gmr_v2` root, including outputs, logs, audits and bindings; verify all bytes before accepting the report-last
manifest. Never rerun v2 `consume` into these roots or delete them to make a command pass. New S0/M0 action
versions require new preregistrations and no-clobber roots.

2026-07-15 Pod2 restore audit found **none** of those ignored bindings: the
GMR worktree and 282,953,810-byte bundle, neutral SMPL-X/model/mapping, S0
manifest/betas/PT and M0 manifest/betas/four PTs are all absent. The old
`/workspace/yikang/.../hope-motion-py310` environment is absent too; the
nearest Isaac venv matches only 87 of the 234 frozen package lines. Do not
recreate v2 by guessing or modifying the shared Isaac venv. Restore the bundle
and both complete intake roots from the authoritative Pod1/backup copy, verify
every SHA in the v2 plans and completions, and keep the Pod2 rc127 as a separate failed-location record. It does
not mean the Pod1 exact-GMR roots are absent. Only a new action version may preregister a new isolated runtime
with wheel hashes and actual import origins before its own `inspect` or `consume`.

Ground exactly one accepted diagnostic GMR pickle with explicit no-clobber
paths. The command below is the Franco forehand-block pilot shape; use each
row's own input SHA/frame count for other assets:

```bash
python3 scripts/ground_gmr_pkl.py \
  --input /private/ignored/motion_video_intake_20260711/gmr_queue_outputs/franco_forehand_block.diagnostic_video_betas.pkl \
  --expected-input-sha256 0e7e674ecba2459b4db6d2c49fb8498a35db2fe8291782eadb7214933be39be5 \
  --output /private/ignored/motion_video_intake_20260711/gmr_grounded/franco_forehand_block.grounded.pkl \
  --report /private/ignored/motion_video_intake_20260711/gmr_grounded/franco_forehand_block.grounding.json \
  --mjcf agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
  --expected-mjcf-sha256 2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97 \
  --expected-frames 65
```

The output and report parents must already exist, and neither target may
exist. The tool never scans a directory or overwrites an input. It uses enabled
canonical collision geometry rather than body origins/COMs, applies one fixed
root-z translation, verifies a near-ground non-penetrating source frame and
records all relevant SHAs. A pass covers only the discrete source frames. Run
the later dense/inter-frame collision, dynamics and table/net gates before
schema-2; never treat grounding as robot authorization.

The accepted canonical-beta grounding and dense safety evidence is retained
under the same private Pod1 root. Restore/copy the complete versioned trees,
never a directory assembled by basename:

```text
/workspace/codexschema/motion_video_intake_20260711/gmr_canonical_betas_grounded_v2/
/workspace/codexschema/motion_video_intake_20260711/phase_safety_control_v4/
/workspace/codexschema/motion_video_intake_20260711/phase_safety_v4/
```

Cross-check every grounded PKL and grounding report against
`configs/motion_video_canonical_gmr_ground_results_20260711.json`. The full
accepted safety result is
`phase_safety_v4/phase_safety_result.json` (571,252 bytes, SHA-256
`d2518f766720dabf979e2a95e5044fdf7fcc7d85b0622471b476888febf301d8`);
the durable small summary is
`configs/motion_video_gmr_phase_safety_results_20260711.json`. The earlier
`phase_safety_v2` tree must also remain if auditing history: only its safety
subtrees are accepted, while every virtual-return/phase/coverage field is
revoked because no GMR-world to HOPE table transform was bound. Raw videos,
GMR pickles and full result JSON remain private ignored artifacts; do not add
them to git or a public registry. Schema-2 +X reground/explicit transform and
mirror evidence remain required before using the frozen question paper.

The follow-up counterfactual frame/phase evidence is versioned separately:

```text
local ignored: vendor_assets/motion_video_intake_20260711/gmr_frame_contract_v1/
Pod1 control:  /workspace/codexschema/motion_video_intake_20260711/phase_safety_control_v5/
Pod1 output:   /workspace/codexschema/motion_video_intake_20260711/phase_safety_v5/
```

Restore the exact ten grounded PKLs before reproducing the local frame audit;
raw video and decoded mirror crops remain private.  The accepted full v5 result
is `phase_safety_v5/phase_safety_result.json` (792,241 bytes, SHA-256
`c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53`).
Tracked small ledgers are
`configs/motion_video_gmr_frame_contract_results_20260711.json` and
`configs/motion_video_gmr_phase_counterfactual_results_20260711.json`.
Do not restore v5 over v4 or rerun into either accepted output directory.  The
scored table is only the canonical HOPE virtual table; capture-table pixels,
extrinsic and real-returnability truth remain absent.  Reproduction commands
and no-clobber checks are in
`docs/operations/run_motion_gmr_counterfactual_screen.md`.

The spatial-retarget proposal screen does not need another copy of the videos
or GMR pickles, but it does require the **full** v5 JSON because the tracked
compact ledger intentionally omits the 64 questions and per-source-frame racket
states. Restore that one file into a new ignored local root, for example:

```text
/private/ignored/motion_video_intake_20260711/phase_safety_v5/phase_safety_result.json
```

It must be exactly 792,241 bytes with SHA-256
`c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53`.
Pass it through `--predecessor-result`; do not change the absolute historical
path inside the tracked preregistration and do not overwrite accepted v5.
Candidate-specific schema-2/L0/L1/table-net certificate artifacts do not yet
exist and must not be fabricated from the compact ledger. See
`docs/operations/run_motion_spatial_retarget_screen.md`.

The accepted 2026-07-13 signed run is stored on Pod1 under
`/workspace/codexschema/motion_spatial_retarget_signed_a4bbbaa_v1/`. Its full proposal JSON is
225,920 bytes with SHA-256 `69c3db16fa78f526aef49f20eeafe0d7e5e3004c4ed27f5e2823bb3574e2465c`.
Restore the whole directory when materializing B/C certificates; the tracked summary intentionally does not
duplicate all 640 cells or 22 proposal payloads.

The accepted rank-0 materializations now live under
`/workspace/codexschema/motion_video_intake_20260711/gmr_spatial_retarget_primary_v1/`. Restore both candidate
subdirectories exactly as listed in
`configs/motion_backhand_loop_bc_se2_materialization_results_20260714.json`: the B/C motion SHA values are
`27827912...ad6` / `0dd981a6...f48b`, and report SHA values are `a238c077...df3` / `b3b93d2c...f67`.
The report is the completion marker. A restored motion without its exact report is partial and must not be consumed.
These payloads only unlock schema-2 preregistration; they are not certificates or training assets yet.

### Venue strike timing table

The continuous-rally timing audit uses the processed strike event table, not
the raw `.tak` projects. Restore the complete venue `analysis/` tree from the
team's 2026-07-03 dataset handoff, set `BALLFIT_DATA_ROOT` to that dataset
root, and verify the exact input before reproducing the published numbers:

```bash
test -f "$BALLFIT_DATA_ROOT/analysis/segments/strikes.json"
sha256sum "$BALLFIT_DATA_ROOT/analysis/segments/strikes.json"
python3 hope_training/whole_body_tracking/scripts/analyze_rally_intervals.py \
  "$BALLFIT_DATA_ROOT/analysis/segments/strikes.json" \
  --max-leg-s 2.5 --summary-only
```

The current canonical processed copy is
`/workspace/yikang/latest_data/analysis/segments/strikes.json`, SHA-256
`6ad3c45959c94b6fdd4033130403c32e0f1b612a138738c12afa43a58f752841`.
Do not replace it with a different detector run without recording the new
source SHA and recomputing the timing report.

### Phase-1 fresh and causal bundle (2026-07-11)

The canonical working copy is currently on Pod 1 at
`/workspace/codexschema/phase1_fresh_20260711/`. It is ignored runtime evidence, not a durable
artifact service. While that Pod copy is retained, restore the reproducibility subset into the
normal ignored artifact root as follows:

The launch-safe, read-only parent copies live at
`/workspace/codexschema/phase1_fresh_20260711/parents/{M3c,M2f}/`; matching legacy motions are
under `assets/legacy/{swing,v4rg}/motions/`. Pod 2 carries the M2f/v4rg subset plus the complete
fresh v3 asset directory with identical hashes. The original historical run trees remain source
evidence, but launchers use these canonical copies so a result directory cannot be mutated.

Detached evaluation worktrees do not receive the git-ignored A3 asset directory. Before an Isaac
export, verify the frozen training copy and create only this local symlink (never copy an ambient
user checkout):

```bash
TRAIN_ASSET=/workspace/codexschema/nohope/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3
EVAL_ASSET=/workspace/codexschema/nohope_eval_08e438e/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3
test -f "$TRAIN_ASSET/urdf/model.urdf"
mkdir -p "$(dirname "$EVAL_ASSET")"
test ! -e "$EVAL_ASSET" && ln -s "$TRAIN_ASSET" "$EVAL_ASSET"
```

The link is ignored and must be recreated on another Pod/worktree. A missing link causes Isaac
scene creation to fail before export; do not classify that as a checkpoint failure.
The detached evaluator also depends on the Pod-local
`/workspace/hope_mjeval_venv`; restore/verify both `onnx==1.22.0` and
`onnxruntime==1.27.0` with the command in `run_training.md`. Having only
`onnxruntime` allows inference import but fails formal graph inspection after
an otherwise successful GPU export.

```bash
REMOTE='root@162.43.172.171'
SSH='ssh -p 18333 -i ~/.ssh/id_ed25519_runpod'
LOCAL_ROOT='hope_training/whole_body_tracking/artifacts/phase1_fresh_20260711'

mkdir -p "$LOCAL_ROOT/assets" "$LOCAL_ROOT/smokes" \
  "$LOCAL_ROOT/checkpoints/M3c" "$LOCAL_ROOT/checkpoints/M2f"
rsync -a --rsync-path='nice -n 19 rsync' -e "$SSH" \
  "$REMOTE:/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/" \
  "$LOCAL_ROOT/assets/v4rg_runtime_order_v3/"
rsync -a --rsync-path='nice -n 19 rsync' -e "$SSH" \
  "$REMOTE:/workspace/codexschema/phase1_fresh_20260711/assets/legacy/" \
  "$LOCAL_ROOT/assets/legacy/"
rsync -a --rsync-path='nice -n 19 rsync' -e "$SSH" \
  "$REMOTE:/workspace/codexschema/phase1_fresh_20260711/smokes/runtime_order_v2_motion/" \
  "$LOCAL_ROOT/smokes/runtime_order_v2_motion/"
rsync -a --rsync-path='nice -n 19 rsync' -e "$SSH" \
  "$REMOTE:/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/" \
  "$LOCAL_ROOT/checkpoint_curves/pod1/"

REMOTE2='root@162.43.172.181'
SSH2='ssh -p 13146 -i ~/.ssh/id_ed25519_runpod'
rsync -a --rsync-path='nice -n 19 rsync' -e "$SSH2" \
  "$REMOTE2:/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/" \
  "$LOCAL_ROOT/checkpoint_curves/pod2/"
scp -P 18333 -i ~/.ssh/id_ed25519_runpod \
  "$REMOTE:/workspace/codexschema/phase1_fresh_20260711/assets/BUILD_RECORD.json" \
  "$REMOTE:/workspace/codexschema/phase1_fresh_20260711/assets/v4rg/OBSOLETE_DO_NOT_USE.json" \
  "$LOCAL_ROOT/assets/"
scp -P 18333 -i ~/.ssh/id_ed25519_runpod \
  "$REMOTE:/workspace/franco/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-09_07-28-14_s1w4_M3c_swing_facesign/model_16999.pt" \
  "$LOCAL_ROOT/checkpoints/M3c/"
scp -P 18333 -i ~/.ssh/id_ed25519_runpod \
  "$REMOTE:/workspace/franco/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-09_07-34-05_s1w4_M2f_v4rg_facesign/model_16999.pt" \
  "$LOCAL_ROOT/checkpoints/M2f/"
```

If the Pod copy is gone, the legacy banks may be regenerated only from the exact motions and
commands recorded in their `asset_manifest.json`; restore the checkpoints from the retained
historical run archive. The fresh v3 family may be regenerated only by repeating the explicit
source-order-to-runtime-order migration and then generating both splits. Do not restore or
regenerate from `assets/v4rg/`: that directory is quarantined. The suffixed v2 filenames were a
fail-closed generator diagnostic and are not accepted bank inputs.

Verify the restored payload before use:

```bash
cd hope_training/whole_body_tracking/artifacts/phase1_fresh_20260711
shasum -a 256 \
  assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz \
  assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz \
  assets/v4rg_runtime_order_v3/s1_v4rg_runtime_order_schema3_train.npz \
  assets/v4rg_runtime_order_v3/s1_v4rg_runtime_order_schema3_exam.npz \
  assets/legacy/v4rg/s1_v4rg_schema3_train.npz \
  assets/legacy/v4rg/s1_v4rg_schema3_exam.npz \
  assets/legacy/v4rg/asset_manifest.json \
  assets/legacy/swing/s1_swing_schema3_train.npz \
  assets/legacy/swing/s1_swing_schema3_exam.npz \
  assets/legacy/swing/asset_manifest.json \
  assets/BUILD_RECORD.json assets/OBSOLETE_DO_NOT_USE.json \
  checkpoints/M3c/model_16999.pt checkpoints/M2f/model_16999.pt
```

Expected SHA-256 values, in that order:

```text
f2cb2d9f5d27cefbcee0b790000fcd979abaf02894d4fcad061ebca27f141687
1722553375cd28f9b2d567c01b1a5fc6bcd149fa12cadb20e5202a9153367534
2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700
d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096
dc67326f5cf0e1a3e3a6ae89f43cbd8a9a3785c341ab2ae998e3edbd40f8d30a
10917148ef251a4dabe387ea418b8907c26fe320b95d9ff874380d09f73e5bb2
6ff25a94c2a5abb590e84415458a8958f749bb62f34f0890d7b7922891571a74
96329c79f13e659c035bf65bafedf84123d23a3219b02ac78ae654aed930cf60
750f1df4ebaf851b96495c53ebbd083c06275f254ce19862f2e84183ec45cb0e
04751e0424458a85f033519ee033176169f5479013abe5a85a46bc1488970fbe
95b9727489d1aa298e411e1ec589505b2c3da1720569fa0da33ddd10176c6032
59ce1bcbe60e3ee594f3ccd4061690ee324de963b8fe84208a463190e03a3388
46f0050589f3343d96f2e5c261b92224079b379e4da473be342b1bd0f0cf7ff1
0ab05144ec1792db91d6e1e3c2ce79f46dae9507ed267b1470838bf998f0f012
```

Also verify tracked `configs/a3_runtime_body_order.txt` has SHA-256
`1cdae4ba7c8d604428ee69ed4a3059e67fb195b22e1d0e294d509c4325809a3a`. The fresh train/exam
banks must both report source-family SHA-256
`b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5`, physics SHA-256
`2e58221442665ddad7cc6dcc18d5c811dec1b0c47439b81c1c744b5148169a27`, physics-contract SHA-256
`70242d798f5b97e1405df7dedfd22a5f81421c9c03127e71c254982236cfad35`, and disjoint train/exam
content IDs. See [PHASE1_FRESH_LINEAGE_2026-07-11.md](../archive/PHASE1_FRESH_LINEAGE_2026-07-11.md) for
the incident record, smoke evidence, and exact-vs-diagnostic restrictions.

The canonical tracked audit is
[`configs/phase1_fresh_v3_asset_manifest_20260711.json`](../../configs/phase1_fresh_v3_asset_manifest_20260711.json),
SHA-256 `0c2a565d7b7040afdda97baecdaf2cea923beaf3cf9c45a574d218bb82386e46`. Compare the restored NPZs
against it rather than copying hashes from a shell transcript. Its `validation.frozen_mode=0444`
matches the four audited remote NPZs.

## Wave-B 下肢稳定队列的 ignored/remote 输入

[`run_phase1_lower_body_stability_wave.md`](run_phase1_lower_body_stability_wave.md) 的 source checkout 是
`/workspace/codexschema/nohope_lowerbody_wave_20260720`。Git 不携带 A3 runtime asset tree；在每个 Pod 的
clean detached checkout 中，从团队已审计的 shared asset 副本恢复到：

```bash
WAVE_B_SOURCE=/workspace/codexschema/nohope_lowerbody_wave_20260720
WAVE_B_ASSET_DST="$WAVE_B_SOURCE/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3"

test -d /workspace/shared/assets/agibot_a3
test ! -L "$WAVE_B_ASSET_DST"
mkdir -p "$WAVE_B_ASSET_DST"
rsync -a /workspace/shared/assets/agibot_a3/ "$WAVE_B_ASSET_DST/"
```

renderer 的 remote preflight 会重新计算全树，不以复制成功作为验收。受理值是 46 个 regular files、
15378264 bytes、tree SHA-256 `071640ea68a4b3d51e8d11154af3098b42b79f356901588d110b5f49e7e6b070`，
且禁止 symlink。

其余大制品保留在已登记的 `/workspace/codexschema/` 根，不复制进 Git：

- six-file USD bundle：`simple_half_second_sprint_20260718/assets/a3_preconverted_usd/`，6 files、
  21897893 bytes、tree `716487dfdf02a5973f78263f0ae8a09e4680c04159e57dbe20796b7825dbeb4d`；
- static v4rg forehand/backhand：`f2cb2d9f...1687` / `17225533...7534`；
- schema-3 train bank：`3a9d8851...5b71`；
- W/V parent checkpoints：`2caab3dd...fcce` / `ad901910...2716`，相邻 hard contracts：
  `e208b682...8551` / `274cb3bd...aee0`。

这些输入任一缺失时必须从保留该 exact SHA 的团队 Pod/制品备份按同一绝对路径恢复，并重新生成 reviewed
manifest；禁止用同名重生成文件、M0 diagnostic GMR 输出或别的 checkpoint 代替。manifest preflight
未通过前只能本地 plan，不能 launch。

## Expected Policy

- Keep source and small config in git.
- Keep large generated/runtime artifacts in `vendor_assets/`.
- Keep reference-only external repos synced through `scripts/sync_external_repos.sh`.
- Promote external repos to submodules only after a project decision.

## Pod A3 preconverted USD 与 headless GLU

ActionBall 的 fresh detached checkout 不应因 URDF 绝对路径变化重复转换同一台 A3。Pod1 当前
Franco-owned 的 ignored/runtime 副本是：

- `/workspace/franco/runtime_assets/a3_preconverted_usd_1b3fecd7/`
- `/workspace/franco/runtime_assets/libglu_af791d1e/`

USD 四层 SHA-256（`model / base / physics / sensor`）依次为
`1b3fecd7685cd98ca80de226fbf89985b77b8a8cfc6a36f18fcc22e65080693c`、
`8e521141bfee4274b8a2369d382cdd8aac9bb1cfcae5bfa480666a1935a7fb42`、
`5b5fc00b96566be295a0cd4eb6b0cd276e360d9cca189057cef452ad0bfc7981`、
`c76c5bdd9e9b5434d72b45c9001858a9c80363656272011ed50d1419149ca60a`；
`libGLU.so.1.3.1` 为
`af791d1ee2acf25417f612290e634248fd716cf5da0374ba21160fb264eaeab4`。

若副本丢失，只能从团队保留的 exact six-file USD bundle
`/workspace/codexschema/simple_half_second_sprint_20260718/assets/a3_preconverted_usd/`
和已审计 private GLU 恢复到新的 no-clobber 目录，逐文件复算上述 SHA 后再设置
`HOPE_AGIBOT_A3_USD_PATH` / `LD_LIBRARY_PATH`；不能按同名或目录存在判定等价。

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
