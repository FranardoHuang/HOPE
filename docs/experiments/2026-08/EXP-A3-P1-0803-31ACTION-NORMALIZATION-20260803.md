# EXP-A3-P1-0803-31ACTION-NORMALIZATION-20260803

Status: completed

Evidence: E1 host/static; Isaac/PhysX import `未测`

## Question

能否不覆盖现役 `agibot_a3/`，把 2026-08-03 A3-P1 交付确定性地投影成与现有
31-action policy ABI 兼容的独立 Isaac candidate，同时保留新的 body origin/inertial 和右拍？

## Decision before materialization

- **Adopt:** 全部 9 个未耦合左夹爪 movable joint 固定在各自 URDF `q=0`；保留
  21 个夹爪 link 的 `0.76626209416 kg` 质量、COM、惯量和 joint origin；恢复现有
  runtime body 的 mixed-case `_Link` 名称；把 78 个大小写错的 mesh 路径显式改到
  交付文件的真实 basename。
- **Defer:** 左夹爪 mount 的 URDF/workbook 冲突、Isaac GPU import、standing hold、
  31-joint ordered importer ABI、拍心 world-FK、collision/dynamics parity 和 MuJoCo identity v3。
- **Reject:** 原地覆盖现役 canonical；猜测 1-driver+8-coupling；伪造缺失的夹爪 collision
  mesh；剪掉整个夹爪而静默丢掉其质量；从旧 plant 抄回 origin/inertial。

## Materialized candidate

- Raw authority: `configs/a3_p1_0803_raw_intake_v1.json`，112 files / 57,803,270 bytes，
  closure SHA-256 `b1da6430fb20901ffd4fedbf60ee1cda452b12d25bd02f3816f359c24a47818f`。
- Tracked producer: `scripts/prepare_a3_p1_0803_31d_asset.py`。
- Tracked exact diff/closure receipt: `configs/a3_p1_0803_31d_v1.json`。
- Ignored output: `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3_p1_0803_31d_v1/`。
- Output: 63 unique links / 62 joints / 31 movable joints / 104 mesh references / 100 unique meshes；
  URDF SHA-256 `d2b8c8359a405f1a1506c153889ee6052aa524f0edee8e173c7de5bf91300bb3`；
  101-file closure SHA-256 `a8f6bd6971d69e6b1ce17e6b85e5aefaea075abd088ab199068a6df362ae6411`。

归一化 diff 精确记录：9 个 movable gripper joint 转 fixed；重复的第一个
`imu_in_pelvis_link` 删除、保留含专用 collision mesh 的第二个；删除 20 个确认缺文件的
gripper collision element；删除只含 `nan nan nan nan` 颜色且无 geometry 的 gripper-base
visual；保留其余 link inertial 与 joint origin/axis/limit。Raw intake 中原先写错的
red/black racket mesh SHA 也已用 raw/current 三方实际字节更正。

## Host verification

```bash
python3 scripts/prepare_a3_p1_0803_31d_asset.py --check
python3 -m pytest -q tests/test_prepare_a3_p1_0803_31d_asset.py
python3 scripts/prepare_a3_isaac_asset.py --check
```

Observed 2026-08-03:

- normalized check: `PASS`, `31 movable / 63 links / 62 joints`, output/URDF SHA exact;
- focused tests: `4 passed`;
- existing current asset: `86 mesh refs`, `0 missing`, still passes and was not rewritten;
- a second prepare attempt exits `1` before writing because the versioned output already exists.

## Result and replacement boundary

**Adopt the producer + candidate as the E1 normalized successor source; do not adopt it as current runtime.**

It is structurally capable of replacing the current asset only after the Pod-side gates below. It cannot yet replace
the current plant before a pre-long run because no real Isaac importer/PhysX evidence exists for these bytes, fixed
gripper links may be merged differently, and the left-gripper mount authority remains conflicting. The new right-racket
local contract is preserved, but changed right-elbow/right-hip origins mean world FK must be remeasured.

Required Pod receipt, all against the exact URDF/closure above:

1. Isaac URDF import completes without allocator/importer failure and reports exactly 31 movable joints.
2. Imported `joint_names` equals the content-bound runtime articulation order, and the required 32 runtime body names
   all resolve after fixed-joint merge; extra/merged gripper bodies are explicitly inventoried.
3. Nominal standing reset/hold passes finite pose, no initial/table/self collision, actual-hard/qdes safety and support.
4. Official right-racket site position/orientation/Jacobian and fixed-gripper left-arm mass properties are compared with
   the bound source; no silent inertia loss during import.
5. Same-state old/new plant parity report explains expected deltas from the new torso/arm inertials and two joint origins;
   fixed tape/RNG/reason/counter/safety parity is rerun before any learning comparison.
6. Vendor resolves or the project explicitly owns the left-gripper URDF-vs-workbook mount choice; then mint MuJoCo
   identity v3 and rerun collision/dynamics parity. Until then `training_authorized=false`.
