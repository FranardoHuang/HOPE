# Joint Order And Robot State

Status: two distinct 31-DOF order domains are explicit and content-bound, and the separate B/C
[schema-2 motion](../DEFINITIONS.md)/FK preregistration now passes its tracked source gate. Runtime
donor/pickle/model inspection and schema-2 materialization have not run; no simulator or hardware
command ran. Backend command scattering was previously verified on hardware (2026-07-03) against
AGI `robot_io::MakeA3Layout31()`.

## Goal

Preserve the joint *names* across GMR, MuJoCo, Isaac, policy export and deployment while mapping each
ordered column domain explicitly. Equal name sets do not imply equal column order.

The old statement that GMR `dof_pos`, training/ONNX and SDK “must share” one array order was false.
GMR writes controller/MJCF order; Isaac runtime and schema-2 `joint_pos` use the interleaved
articulation order. Silent copying between them scrambles all 31 joints.

## Two source-of-truth tables

Both tables are newline-delimited exact URDF joint names. Their raw file bytes and canonical name-list
SHA-256 values are bound by
[`a3_joint_order_bijection_v1.json`](../../configs/a3_joint_order_bijection_v1.json).

| Domain | Human meaning | Tracked source of truth | Columns |
| --- | --- | --- | --- |
| `gmr_dof_pos` | [GMR](../DEFINITIONS.md) A3 pickle/CSV `dof_pos`; also the legacy controller/backend layout | [`a3_gmr_dof_pos_joint_order.txt`](../../configs/a3_gmr_dof_pos_joint_order.txt) | 31 |
| `runtime_articulation_joint_pos` | Isaac articulation, schema-2 NPZ `joint_pos`/`joint_vel`, exact ONNX policy/action order | [`a3_runtime_articulation_joint_order.txt`](../../configs/a3_runtime_articulation_joint_order.txt) | 31 |

The legacy [`joint_order_agibot_a3.yaml`](../../hope_training/config/joint_order_agibot_a3.yaml) and
`AGIBOT_A3_JOINT_NAMES` remain mirrors of **only** `gmr_dof_pos`. The validator parses both mirrors
without importing Isaac and rejects drift. The YAML's old embedded “training/ONNX/SDK must share”
comment is known false but is retained byte-exact because completed content-bound GMR ledgers bind
that file as `1,099` bytes / SHA `7748a134...b205`; this interface and the new contract supersede
the prose without rewriting historical evidence.

`audit_motion_npz.py::ISAAC_JOINT_NAMES` is a byte-frozen mirror of the runtime target order because
an executed v4 safety prereg binds that whole tool at SHA `ec25fac2...47b1b`. Rewriting it would
falsify history, so the new validator parses its literal with AST and rejects disagreement instead.

### GMR source order (`dof_pos`)

| Index | Joint | Index | Joint |
| --- | --- | --- | --- |
| 0 | `waist_yaw_joint` | 16 | `right_wrist_roll_joint` |
| 1 | `waist_roll_joint` | 17 | `right_wrist_pitch_joint` |
| 2 | `waist_pitch_joint` | 18 | `right_wrist_yaw_joint` |
| 3 | `head_yaw_joint` | 19 | `left_hip_pitch_joint` |
| 4 | `head_pitch_joint` | 20 | `left_hip_roll_joint` |
| 5 | `left_shoulder_pitch_joint` | 21 | `left_hip_yaw_joint` |
| 6 | `left_shoulder_roll_joint` | 22 | `left_knee_joint` |
| 7 | `left_shoulder_yaw_joint` | 23 | `left_ankle_pitch_joint` |
| 8 | `left_elbow_joint` | 24 | `left_ankle_roll_joint` |
| 9 | `left_wrist_roll_joint` | 25 | `right_hip_pitch_joint` |
| 10 | `left_wrist_pitch_joint` | 26 | `right_hip_roll_joint` |
| 11 | `left_wrist_yaw_joint` | 27 | `right_hip_yaw_joint` |
| 12 | `right_shoulder_pitch_joint` | 28 | `right_knee_joint` |
| 13 | `right_shoulder_roll_joint` | 29 | `right_ankle_pitch_joint` |
| 14 | `right_shoulder_yaw_joint` | 30 | `right_ankle_roll_joint` |
| 15 | `right_elbow_joint` |  |  |

### Runtime target order (`joint_pos`)

This is the order independently tabulated from exported ONNX metadata and the Isaac action contract
in `agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md` section 4.

| Target index | Joint | GMR source index |
| --- | --- | --- |
| 0 | `left_hip_pitch_joint` | 19 |
| 1 | `right_hip_pitch_joint` | 25 |
| 2 | `waist_yaw_joint` | 0 |
| 3 | `left_hip_roll_joint` | 20 |
| 4 | `right_hip_roll_joint` | 26 |
| 5 | `waist_roll_joint` | 1 |
| 6 | `left_hip_yaw_joint` | 21 |
| 7 | `right_hip_yaw_joint` | 27 |
| 8 | `waist_pitch_joint` | 2 |
| 9 | `left_knee_joint` | 22 |
| 10 | `right_knee_joint` | 28 |
| 11 | `head_yaw_joint` | 3 |
| 12 | `left_shoulder_pitch_joint` | 5 |
| 13 | `right_shoulder_pitch_joint` | 12 |
| 14 | `left_ankle_pitch_joint` | 23 |
| 15 | `right_ankle_pitch_joint` | 29 |
| 16 | `head_pitch_joint` | 4 |
| 17 | `left_shoulder_roll_joint` | 6 |
| 18 | `right_shoulder_roll_joint` | 13 |
| 19 | `left_ankle_roll_joint` | 24 |
| 20 | `right_ankle_roll_joint` | 30 |
| 21 | `left_shoulder_yaw_joint` | 7 |
| 22 | `right_shoulder_yaw_joint` | 14 |
| 23 | `left_elbow_joint` | 8 |
| 24 | `right_elbow_joint` | 15 |
| 25 | `left_wrist_roll_joint` | 9 |
| 26 | `right_wrist_roll_joint` | 16 |
| 27 | `left_wrist_pitch_joint` | 10 |
| 28 | `right_wrist_pitch_joint` | 17 |
| 29 | `left_wrist_yaw_joint` | 11 |
| 30 | `right_wrist_yaw_joint` | 18 |

The machine permutation means `runtime[..., i] = gmr[..., target_from_source_indices[i]]`. Its
inverse is also stored and checked. Duplicate, missing, extra, malformed or wrong-length names;
non-bijective indices; mirror drift; and wrong-order or partial ONNX metadata all fail closed.

## Source gate and conversion boundary

Run the tracked source gate from repository root:

```bash
python3 hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py
python3 -m pytest -q tests/test_a3_joint_order_contract.py
```

The MuJoCo converter now consumes the same contract instead of a private hard-coded list. The
byte-frozen L0 auditor remains unchanged but its literal target table is checked as a contract
mirror. A donor ONNX must expose all three fields: `joint_names`, `articulation_joint_names`, and identity
`action_joint_ids=0..30`; partial legacy metadata is rejected. `audit_motion_npz.py` resolves its
default NPZ joint labels from that validated byte-frozen mirror.

The separate preregistration now binds all of those inputs in
[`motion_backhand_loop_bc_schema2_fk_runtime_v1.json`](../../configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json)
and one independent plan per B/C asset. The consumer maps GMR columns through the checked
permutation before FK; schema-2 `joint_pos/joint_vel` are therefore runtime-order. The body file has
32 unique columns that form a bijection over the 32 named vendor MJCF bodies. `body_pos_w` comes from
link-origin `xpos`, while `body_lin_vel_w` differentiates center-of-mass `xipos`.

Passing the tracked `static` gate still authorizes **no immediate materialization**. The next command
is a no-write runtime inspection: restricted-load the exact private pickle; hash the exact donor
ONNX and re-extract all three required metadata rows; load the content-bound MJCF closure; and verify
the output root is absent. The CLI only accepts `--hope_frame off`, because the B/C root has already
been transformed into the HOPE frame. Only a successful per-asset inspection unlocks its one
no-clobber 30→50 Hz FK materialization; output report is published last.

## Policy/runtime/backend scattering

Training emits 31 actions in runtime target order. The HOPE deploy runner decodes those actions and
name-scatters to the AGI backend layout; it does not depend on positional equality. Neck joints are
overridden after decode to `q=0`, `kp=40`, `kd=2`. The previous 2026-07-03 slot-for-slot check proved
the backend map against `MakeA3Layout31`; it did not prove that GMR and Isaac columns were equal.

The runtime robot state must additionally agree on joint positions/velocities, previous action,
base angular velocity, projected gravity, base pose/heading and policy clock. See
[`policy_observation_action.md`](policy_observation_action.md).

## Hardware boundary

No real command is authorized by this source correction. Any change to either ordered table, the
permutation, ONNX action metadata or backend layout re-triggers dry-run and hardware verification.
The `/joint_states` field order remains hardware-TBD; do not infer it from command order.
