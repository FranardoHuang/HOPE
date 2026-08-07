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
- `hope_training/whole_body_tracking/.hitter_align_backup/` and other one-off backup/debug scratch folders.
- `pw.txt` or other local secret scratch files; never commit passwords, tokens, or private identities.
- `external_repos/IsaacLab/` when a local Isaac Lab source checkout is used for the training environment; record the tag/commit in the relevant gate doc.
- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` and other package-local copied/generated training assets; keep only the tiny `assets/__init__.py` path-helper and local `.gitignore` tracked so `whole_body_tracking.assets.ASSET_DIR` remains importable after a fresh clone.

`vendor_assets/` 本身必须是每个 checkout 内新建的真实目录：不得把这个根作为普通文件、
Git blob、绝对 symlink、指回 checkout 自身的 symlink，或指向另一个正在运行的 checkout。
只允许按下文和
[`setup_local_sync.md`](operations/setup_local_sync.md)
把明确的内容寻址子树恢复进来。这样 fresh checkout 的 Git 状态与 ignored 资产彼此独立，
训练前也能分别证明 source clean 和 local asset bytes 一致。

## User-Recorded Motion Videos

Raw user-recorded motion videos are private, local-only inputs. Do not commit
or publish the video bytes. Track only a small content-addressed manifest with
the relative filename, byte count, SHA-256, media properties and intended
semantic action. A private processing-Pod copy is allowed for this project,
but it remains an ignored runtime asset and is not an artifact publication.

The 2026-07-11 Franco/v6/v7 intake is bound by
`configs/motion_video_intake_20260711.json`; the local restore and private Pod
staging paths are recorded in `docs/operations/setup_local_sync.md`. Derived
SMPL-X, CSV, NPZ, checkpoint and policy files follow the same ignored-heavy-
artifact rule. A derived action is not deployable merely because its source
video is present: motion, self-collision, table/net clearance and simulator
gates must still pass, and no raw video grants permission for a real-robot run.

The 2026-07-13 v12/static/motion intake is separately bound by
`configs/motion_video_intake_20260713.json`. Its schema distinguishes stroke
videos from lower-body footwork clips, so a footwork clip cannot be silently
treated as a block clip. The intake file's `lateral_locomotion_teacher` /
`lateral_step_teacher` labels are frozen historical fields: since 2026-07-20
the four `motion/` dang clips are semantically the shared lateral footwork
module, and `configs/motion_role_catalog.json` (checked by
`scripts/validate_motion_role_catalog.py`) is the single semantic source of
truth for motion roles; the intake bytes themselves must not change. The seven
raw MP4s stay under the user's private `${HOME}/Downloads/{v12,static,motion}`
folders until an explicitly versioned private staging copy is made; intake
verification grants no compute or simulator authorization.

The diagnostic GMR run also depends on five local source commits not present
in the observed public upstream. Its 282,953,810-byte recovery bundle remains
under the ignored private processing root; git tracks only the verified commit,
bundle byte count/SHA and small per-result bindings. Do not add the bundle or
generated GMR pickles to normal git. Reproduction must `git bundle verify`,
confirm that the bundle advertises the recorded commit, and require a clean
checkout before processing; the exact restore path is in
`docs/operations/setup_local_sync.md`.

## Agibot Deploy Assets

The tracked deploy source is under:

- `agi/a3_deploy_example/` — the active ping-pong deploy tree (Route A runner, build scripts,
  runbooks); generated build outputs under its `dist/` and policy `.onnx` artifacts stay ignored.
- `agi/code_deployment/a3_deploy_example/` — the older vendor reference subset.

The complete original payload is kept locally under:

- `vendor_assets/agibot/a3_deploy_example_full/`

If a command depends on a file inside `vendor_assets/`, the operation doc must say so and give the expected path.

## Agibot A3 Robot Assets

Tracked source assets:

- `agi/URDF/A3T2.5-URDF-std-pingpang/` is the current internal source URDF package for the A3 ping-pong variant. It includes the racket hand, red/black paddle faces, and the racket-center ball marker meshes: `right_hand_pingpang_Link.STL`, `pingpang_red_Link.STL`, `pingpang_black_Link.STL`, and `pingbang_ball_Link.STL`.
- `agi/URDF/a3_t2d5/` is the non-ping-pong standard A3 URDF package. Keep it as a source reference for non-racket comparisons; do not delete it just because the WBC training asset uses the ping-pong variant.
- `agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/` is the Agibot-provided MuJoCo/AimRT ping-pong model layer, including collision-oriented MJCF/mesh materials used for sim parity work.

2026-08-03 收到的新交付 `A3-P1-32dof-0803-BerkeleyPingpang-90deg` 先按原字节保存在
ignored `vendor_assets/agibot/A3-P1-32dof-0803-BerkeleyPingpang-90deg/`，其小型内容清单是
[`configs/a3_p1_0803_raw_intake_v1.json`](../configs/a3_p1_0803_raw_intake_v1.json)。它是后续
A3-P1 successor model 的原始输入权威，但**尚不是**现役 Isaac/MuJoCo/runtime canonical：交付无
license/README，URDF 实际含 `40` 个 movable joints（旧身体31轴加9个未耦合夹爪轴），并有 Linux
大小写 mesh 失配、20个缺失夹爪 collision mesh、非有限颜色、夹爪安装姿态冲突和旧身体 plant
参数变化。不得用它原地覆盖 `agi/URDF/A3T2.5-URDF-std-pingpang/` 或改写历史模型 SHA。

2026-08-04 项目裁决 raw URDF（不是冲突 workbook）为该交付的 mount/几何 ground truth，并
版本化拥有左夹爪 training projection：九个非 policy joint 固定在 URDF coordinate `q=0`。这不
声称 `q=0` 是 vendor neutral 或硬件 home。生成器保留 `0.76626209416 kg` 夹爪质量/inertial；
20 个未随包交付的 gripper collision mesh 不得伪造，只有对应的 20 个 collision element 被显式
disabled，并保留为 collision/sim-to-real promotion 风险。可复现 candidate receipt 是
[`configs/a3_p1_0803_31d_v1.json`](../configs/a3_p1_0803_31d_v1.json)：31 movable joints，
URDF SHA `2f15df8a…2535`，closure `73a47e85…8f08`。它仍为 ignored future-primary candidate；
`current_runtime_pointer_changed=false / training_authorized=false`，不得热覆盖现役
`agibot_a3/`。不允许把 56 MB mesh 复制品加入 Git。

0803 raw/normalized 的 official paddle-centre local transform 与四个右拍 mesh bytes 均和现役
exact；但新 `right_elbow_joint` origin 令共同 `q=0` 的 world paddle centre 相对现役移动
`9.013878 mm`。所以“local right racket unchanged”只授权 candidate construction，不授权复用旧
动作的 world-FK/retarget receipt；切换前必须在 successor 上重跑 full-phase audit。

2026-08-06 项目裁决：那 `9.013878 mm` 里的 `9.0 mm` 是交付自身的左右不对称缺陷，不是设计
改动。**人话**：同一份交付里左肘还是 `x=0.01`，只有右肘变成 `0.001`；而右上臂和右肘的 mesh
与现役逐字节相同（44/44 SHA-256 一致），零件根本没改。因此
[`configs/a3_p1_0803_31d_v2.json`](../configs/a3_p1_0803_31d_v2.json) 在 v1 基础上只加一条
**声明式项目覆盖**：`right_elbow_joint` 的 `x` 恢复为 `0.01`，保留交付修好的 `z=-0.1325`
（那一条是真修复：旧右 `-0.133` vs 旧左 `-0.1325`）。其余全部沿用交付，包括
`right_hip_roll_joint` 的 `1.1 mm` 对称化和五个 link 的质量/质心/惯量改动。

该覆盖是 **provisional**：交付自带的 joint workbook 也写 `0.001`，说明缺陷在上游 CAD，必须
向厂商上报，manifest 里 `mirror_symmetry_correction_vendor_confirmed=false` 且
`reproduces_delivered_joint_origins_exactly=false` 自陈这一点。覆盖后 successor 相对现役的
paddle centre 偏移从 `9.013878 mm` 降到 `0.500000 mm` —— 仍超 `1e-4 m` 的 racket FK 门，
**动作库照样要重跑 audit**，只是从"确定作废"变成"要量一下"。v1 目录与其 Pod receipt 冻结不
动，producer 会拒绝写入。

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

恢复完成后至少验证：

```bash
test -d vendor_assets
test ! -L vendor_assets
git check-ignore -q vendor_assets
test -z "$(git status --porcelain --untracked-files=no)"
```

最后一行只证明 tracked source 没被恢复动作污染；每个具体 Gate 仍须重算所消费子树的
逐文件 SHA/receipt，不能把“被 Git ignore”当成内容证明。

For TTRL-only reference work, a fresh clone becomes current after `scripts/sync_external_repos.sh`; reproducible extracted work still needs the source commit recorded.

## Branch Integration Rule

When merging feature/training branches into `main`, keep source, config, tests, and docs tracked, but remove generated/debug artifacts from the index. Current examples are `.codex-tmp/`, `.vscode/`, `.hitter_align_backup/`, ad hoc comparison scripts, training logs, WandB caches, checkpoints, and generated `*.onnx` policy files. If a generated artifact is needed to reproduce a gate result, record the restore path and metadata in [operations/setup_local_sync.md](operations/setup_local_sync.md) and the relevant gate doc instead of committing it directly.

## Exception: curated training motion sets (Franco 2026-07-28)

`assets/motions/fivebind_20260727/`(五动作 fivebind 编译件 + 注册行)和
`assets/motions/chingmu73_20260728/`(ChingMu 真动捕 73+1 件 50Hz npz + 逐件 manifest +
CLIP_ORDER)**入库跟踪**:它们是现役训练臂的直接输入,小体积、内容寻址(manifest 带逐件
sha256),丢失即无法复现任何在跑 run。题库(bank npz)仍不入库——由
`gen_stage1_questions.py` 按注册行可再生。

同目录中的 `bh_loop_c_upper_qvel_fix_v1.npz`、
`bh_block_upper_qvel_fix_v1.npz` 及各自 `*.receipt.json` 也属于该例外：它们是 exact A3
上对 two upper 动作做的 qvel-only 一致性修复，体积小且直接作为 N=1 训练输入；receipt
固定输入、A3 模型与输出 SHA，不能只保留在 Pod 临时目录。


## 2026-08-07 A3P-P1 双引擎模型集

厂商在 0807 重发了 `A3P-P1-32dof-0807-OP3+pingpang`，**采纳了项目对 `right_elbow_joint` 的镜像
对称判定**（`x` 由 `0.001` 改回 `0.01`），并额外修掉 5 个 fixed 关节上的非法 `<axis>` 与两个
`ankle_pitch` 的 ±1.5 mm 侧向不对称。**人话**：0803 那条项目补丁（v2）就此退役——它的值被源头
确认了，两者拍心只差 5 µm。v2 仍然保留在仓库里作为历史记录，不删。

这次交付**不是一个自洽的包**：URDF 不含任何网格，`OmniPicker3-T1-0324-T1.5-close-ROS2` 只含
20 个夹爪视觉网格，124 个引用里 104 个只能回到 0803 包里取（且 82 个大小写不符）。因此
[`intake_a3p_p1_0807_bundle.py`](../scripts/intake_a3p_p1_0807_bundle.py) 拼装出一个
**project-assembled bundle**（不是 vendor closure），逐文件记录来源包与 SHA-256，并把网格落成
URDF 要求的精确大小写，使其在 Linux 上零失配。OmniPicker3 的 20 个网格与 0803 逐字节相同，
**不新增任何几何**。

厂商口头确认"夹爪碰撞几何等同视觉几何"。这条**已写进 intake 收据**，并显式标注
`written_evidence_on_file=false` / `channel=relayed_by_project_owner_from_vendor`。据此，0803 的
collision-disabled 合同被 `GRIPPER_COLLISION_EQUALS_VISUAL_CONTRACT` 取代：20 个碰撞网格按字节
复制自对应视觉网格，复制后逐个 SHA 校验相等。**这仍不是书面确认**，收据自陈这一点。

[`prepare_a3p_p1_0807_model_set.py`](../scripts/prepare_a3p_p1_0807_model_set.py) 从同一个 bundle
同时产出两套：Isaac 资产 `agibot_a3p_p1_0807_v1/`，以及新版本化 MJCF
`model/a3p_pingpong_0807/a3p_pingpong_0807.xml`。现役 `a3_pingpong.xml` **一个字节都不改**（它被
4 处 SHA 钉死），派生前先校验其 SHA 未漂。MJCF 只动几何：8 个 `<body pos>`（4 个真改动 + 4 个
取整）与 7 个 `<inertial>`；armature/damping/frictionloss/31 个 actuator/33 个凸包/球拍面代理/
site/sensor/keyframe/contact-exclude **全部逐字保留**。左夹爪不新增 body 或 joint——质量并入
`left_wrist_yaw_Link`（`0.280678 → 0.846940 kg`），20 个网格作为该 body 自身的 geom 挂在 q=0
位姿上，因此 32-body / 31-actuator ABI 与 keyframe 宽度全部不变。

**未验证边界（收据自陈）**：本机既无 Isaac Lab 也无 `mujoco`/`mujoco_warp`，两套输出都只做了
结构与 ABI 校验。`isaac_import_verified` / `mujoco_compile_verified` / `mujoco_warp_load_verified` /
`cross_engine_parity_verified` / `training_authorized` 全为 `false`。GPU lane 入口是
`A3_PINGPONG_XML` 环境变量（`mjlab_lane/a3_plant_env.py` 的 `default_xml()` 不做 hash 校验），
所以不需要 identity v3 就能导入；identity v3 只有在要重新进入 CPU 侧四个 fail-closed 证据门时才必须。
