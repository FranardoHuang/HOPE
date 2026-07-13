# Policy Observation And Action

Status: Implemented for 110/175/177/180. The 179 training/evaluation contract and versioned
flat-wire/C++ source path are implemented, but vendor Gate 3 runtime evidence is still pending.
The 181 deploy wire remains intentionally blocked pending the station/order contract day.

## HITTER-Compatible Contract

The baseline policy keeps the HITTER-style separation:

- Planner provides target racket position, target racket velocity, target racket normal, and time to strike.
- WBC policy combines planner target, robot state, and previous action.
- Policy outputs desired joint positions.

## Actor Observation (implemented): 175-D deploy-parity

Source of truth: `HOPEPingPongDeployParityAgibotA3EnvCfg` in `hope_env_cfg.py`
(`HOPEPingPongRealSensor` is a backward-compat alias). The canonical layout with per-term hardware
sources is codified and asserted in
`hope_training/whole_body_tracking/scripts/realsensor_obs_reference.py` and checked by
`scripts/verify_realsensor.py`; the C++ deploy runner builds the identical layout
(`pp_obs_builder.hpp`) and auto-detects 175 vs 180 from the ONNX input dim.

| # | Term | Dim | On-hardware source |
| --- | --- | --- | --- |
| 0 | `command` | 62 | reference clip future joint pos/vel (baked into the ONNX) |
| 1 | `motion_anchor_ori_b` | 6 | relative orientation ref-vs-robot (IMU + clip); no base position |
| 2 | `base_ang_vel` | 3 | pelvis link/IMU-frame gyro; never the compiled inertia-principal axes |
| 3 | `joint_pos` | 31 | joint encoders (q − default_q) |
| 4 | `joint_vel` | 31 | joint encoders (dq) |
| 5 | `actions` | 31 | previous action |
| 6 | `projected_gravity` | 3 | pelvis IMU (gravity in base frame), Unoise ±0.05 |
| 7 | `racket_target_pos_b` | 3 | REFRAMED: `yaw(base)⁻¹ · (target_w − racket_FK_w)`, Unoise ±0.02 |
| 8 | `racket_target_vel_w` | 3 | planner desired racket velocity (world) |
| 9 | `time_to_strike` | 1 | swing clock (from clip strike phase) |
| 10 | `swing_type` | 1 | forehand `+1` / backhand `−1` (planner) |

Total = 175. Removed vs the legacy 180-D `full` layout:

- `motion_anchor_pos_b` (3) — reference torso position error; needs the world base pose.
- `base_target_pos_b` (2) — base-repositioning target; needs the world base pose.

The racket-target reframe makes the term base-position-free: because `quat_rotate_inverse` is linear
in its vector argument, `R⁻¹(target − racket) = (target rel base) − (racket rel base)` for ANY base
position — verified numerically in `realsensor_obs_reference.py`.

Notes:

- Desired racket normal is NOT an actor observation (HITTER Table I: normal is a reward target
  only). `base_lin_vel` is critic-only. `swing_type` is included because the default task trains
  one unified forehand+backhand policy.
- The MuJoCo implementation of `base_ang_vel` must use the pelvis link frame (`mjOBJ_XBODY` with
  local output, or the numerically equivalent `R_pelvis^T * omega_world`). `mjOBJ_BODY` with local
  output uses MuJoCo's compiled inertia-principal axes when `body_iquat` is non-identity; that is a
  different frame from both the pelvis IMU and `projected_gravity`.
- `task.racket.face_command_pairing` does not change the 175-D actor. For the 179/181 layouts,
  `racket_target_normal_cmd` always remains the delayed atomic bank command in the raw mount
  +Y/A convention. The external schema-2 wire carries the physical striking face B; the 179
  runner converts only that normal to A after clip selection. The selector never flips or
  relabels the actor command itself.
- Rationale for base-position freedom: the mocap DOES stream the robot base pose during play
  (300 Hz, `/P1/pose` — see the deploy-available signal set below), but that VRPN link is not
  bridged into the deploy runner, and independence from it is a deliberate robustness choice. The
  earlier "no localizer on the real A3" wording was inaccurate.
- The legacy 180-D `full` contract (`task=HOPEPingPong`, model_15200 lineage) is kept for
  comparison only; it depends on world-base-position terms and is not deploy-honest. The deploy
  runner still accepts 180-D ONNX for legacy checkpoints.

### Other registered actor layouts

| Dim | Contract | Delta / source | C++ publish status |
| --- | --- | --- | --- |
| 177 | `hitter_footwork` | 175 layout with `base_target_pos_b(2)` inserted after projected gravity; requires fresh external/oracle base localization. | Supported, but publication fails closed without fresh localization. |
| 179 | `deploy_parity_face179` | Exact 175 prefix + tail `racket_target_normal_cmd(3), rho(1)`; actor tail is raw mount +Y/A after the runner converts the physical-B wire normal with the selected clip sign. | Flat wire schema 2 plus exact metadata/content-bound train-normal envelope/planner mode are mandatory. One envelope-bearing formal model has passed strict Release model preflight (`configs/gate3_face179_strict_preflight_evidence_20260712.json`), but no backend first tick or vendor Gate 3 behavior, so it is not yet a behavioral candidate. |
| 181 | `deploy_parity_station181` | Exact 179 prefix + tail `station_anchor_err_b(2)`. | Blocked: wire and the unique station/normal term order are not frozen. |
| 110 | `hitter_pure` | HITTER Table-I style: 99-D proprio prefix + base forward(2), station delta(2), racket target rel base(3), target velocity(3), tts(1); no reference command or swing flag. | Supported; requires fresh localization and metadata-bound per-side station geometry. |

Do not infer a contract from width alone. Formal consumers require the registered name, mode,
ordered term names/dims and total dimension to agree. The 179 C++ path accepts only
`deploy_parity_face179` metadata and flat wire schema 2; it also binds face enabled,
`shared_plus_y`, `mount_plusY_A`, exact schema-3 train split, train-bank SHA and source-family SHA.
Schema 1 never fabricates the tail.
Merely accepting 181 in an input-shape whitelist would still create a right-width/wrong-columns
command, so 181 remains rejected until its unique station/normal order is frozen.

### Flat racket-command wire

Schema 1 remains the backward-compatible position/velocity wire for 110/175/177/180. Schema 2 is
an explicit 16-double row: the same 12-value prefix with mandatory `frame_code`, followed by the
physical, opponent-facing striking-face-B `normal_cmd[3]` and `rho`. Phase-1 schema 2 accepts only
`frame_code=0` world/table rows: the old
schema-1 code1 path is a yaw-heading transform, not a frozen full-3D base-link normal contract.
The Phase-1 contract also requires `normal_B.x > 1e-6`, a unit B normal, and exactly-zero rho. Once
the forehand/backhand clip is selected, the C++ runner computes
`normal_A = mount_normal_sign_per_clip[clip] * normal_B` with the exact table `[+1,-1]`. Only the
normal is transformed; position and velocity remain untouched in the world/table frame. A
malformed schema-2 row retains the last good tuple for diagnostics but records
`invalid_after`, so the engage grace blocks it rather than letting the old tuple live for the full
command timeout. Unknown/fractional rows received after an active schema-2 command do the same;
schema-1 keeps its historical ignore-and-age behavior when no formal face command is active. A 179
actor refuses to engage on schema 1. The planner publishes schema 1 by default for compatibility;
schema 2 remains the face-only prefix, while a reviewed formal 179 Gate3 launch must set
`racket_flat_schema:=3` to add epoch/sequence/base-reference causality.
The formal flat row is published before the optional `hope_msgs/RacketCommand` mirror; mirror
conversion/DDS failures are counted but cannot suppress a new formal row or revocation.

The positive-X check is only the physical-B wire invariant. Formal 179 exports add a per-clip
spherical-cap envelope in raw A, derived from the exact schema-3 train-bank bytes. Clip 0 is always
`forehand`, clip 1 always `backhand`; their rows are never pooled. For each clip, every raw-A
demanded normal must already be unit within `2e-4`, lie in the same open hemisphere as that clip's
raw `mount_plusY_A` reference with `dot(row_A, reference_A) > 1e-6`, and satisfy
`mount_normal_sign_per_clip[clip] * normal_A.x > 1e-6` so it has a representable opponent-facing B
wire value. Thus forehand raw-A x is positive while backhand raw-A x is negative. The exporter
normalizes those rows, forms the normalized vector sum
(`per_clip_sign_preserving_spherical_mean_cap_v1`), and records the minimum row-to-center dot as
the cap boundary. This avoids the invalid operation of averaging opposite racket-face signs.

The ONNX binds all of the following metadata into one canonical newline-delimited payload and
recomputable SHA-256: envelope schema `1`, frame `world_table_frame0`, face convention
`mount_plusY_A`, pairing `shared_plus_y`, algorithm, bank-row unit tolerance `0.0002`, runtime
unit/dot tolerances `0.000001`, exact clip order, exact sign table `1,-1`, two centers, two
reference normals, two minimum dots, two row counts, train-bank SHA and source-family SHA. The C++
loader requires every field,
recomputes the payload SHA, checks both embedded bank/family hashes against the existing formal
179 metadata, and rejects malformed/non-unit/flipped caps or a sign-table disagreement. At planner
engage it selects the clip, converts B to raw A, then requires both
`dot(reference_A, normal_A) > 1e-6` and
`dot(center_A, normal_A) + 1e-6 >= min_dot` before any target/clock/side/normal state is committed.
Missing envelope metadata therefore makes earlier 179 ONNX files unloadable under the new source;
they must be re-exported from the exact train bank. The other 110/175/177/180 layouts do not read
these fields.

This closes a source-level load/safety prerequisite only. A cap contains all train rows but is not
a proof that every point inside the cap is dynamically safe, collision-free or successful in the
vendor MuJoCo. Self-hit instrumentation, a canonical recovery tuple, a new envelope-bearing formal
export and Gate 3/Gate 3B behavior evidence remain mandatory.

The active swing tuple is atomic, but the existing Gate 3 post-swing policy-recovery path
synthesizes a base-anchored hold position while retaining the previous velocity/normal. Current
Phase-1 training has no frozen contract proving that hybrid tuple is on-distribution. Therefore
179 source support does not make recovery exact; a canonical recovery tuple or an independently
accepted vendor-MuJoCo recovery gate is required before continuous or deploy claims.

The prospective real-bank fixture
`configs/phase1_face179_real_bank_envelope_expectations_20260712.json` binds train bank
`2da2bd12...a0700`, source family `b21c161a...28ad5`, row counts `757/724`, and the observed raw-A
sign/range and cap statistics. It is a source-contract expectation for the next export, not a
formal ONNX, Isaac result, vendor-MuJoCo result, collision proof or recovery result.

Model publication state and model-contract strictness are separate. Plain `--no-publish`,
`--dry-run`, and `--model-preflight-only` still require the same schema-2 packaging, exact complete
schema-3 execution metadata and (for 179) envelope as live publication. Only explicit
`--allow-legacy-model-diagnostic` relaxes legacy model loading; it requires no-publish and cannot
be combined with model preflight. Therefore an accepted preflight certificate always reports
parsed `publishable_model_contract=true` and `training_contract_exact=1`.

## Critic (privileged) Observation (implemented)

The critic group is unchanged between full and deploy-parity (~318-D) and is never deployed. Base
`PrivilegedCfg` terms: `command` 62, `motion_anchor_pos_b` 3, `motion_anchor_ori_b` 6, `body_pos`
42 (14 tracked bodies × 3), `body_ori` 84 (14 × 6), `base_lin_vel` 3, `base_ang_vel` 3,
`joint_pos` 31, `joint_vel` 31, `actions` 31 — all noise-free. HOPE additions: `base_target_pos_b`
2, `racket_target_pos_b` 3, `racket_target_vel_w` 3, `racket_target_normal_w` 3, `time_to_strike`
1, plus sim-only actual racket FK state `racket_pos_b` 3, `racket_lin_vel_w` 3, `racket_normal_w`
3, and `episode_time_left` 1. Privileged sim-only info is allowed here by the training contract.

When `face_command=true`, one validated selector keeps the face reward, privileged face
observations, and exact/composite face metrics on the same measured/target pair:

- `face_command_pairing: shared_plus_y` (default) compares raw mount +Y with the demanded
  `target_normal_cmd` in the bank's +Y/A frame.
- `face_command_pairing: legacy_signed_vs_A` compares the per-clip signed measured normal with the
  same A-frame target. It intentionally reproduces the historical signed-measurement/A-target
  mismatch and is diagnostic only; report such results with
  `evaluation_contract_exact=false`.

This selector changes training reward/critic/metric pairing, not actor observation meaning or the
deploy wire. Unknown values fail during command construction. The selected value is bound into the
schema-3 hard contract and ONNX/export metadata so old and corrected continuations cannot be mixed.

## Action

- `ActionsCfg.joint_pos` with `joint_names=[.*]`, `use_default_offset=True`.
- Dimension = **31** in training AND in the deployed ping-pong runner: the ONNX outputs 31 actions
  (incl. `head_yaw_joint`/`head_pitch_joint`); the runner scatters them to the 31 SDK slots
  (`MakeA3Layout31`) and overrides the neck slots post-decode to `q=0, kp=40, kd=2`. The 29-DOF
  `ExpandToBackend` view belongs to AGI's official reference runner, not the HOPE ping-pong path.
  See [joint_order_and_robot_state.md](joint_order_and_robot_state.md).
- Decoder target = `action * action_scale + default_angle`, per-joint
  `action_scale = 0.25 * effort_limit / stiffness`. The policy target first reproduces the
  schema-3 training soft q-des limits in actor order; a separate outer SDK-order hard clamp protects
  stand/reference/blend and future command sources. The runner publishes `{q_des, kp, kd}` to the
  implicit-PD backend (`dq_des = tau_ff = 0`) only while its requested/measured effort envelope and
  authorization generation remain valid.
- Formal MuJoCo BankExam realizes an Isaac implicit joint's bounded drive as one total operation,
  `tau = clip(kp*(q_des-q) - kd*qdot, -effort_limit, +effort_limit)`, recomputed every physics
  substep. It does **not** clip P before D and does not place actuator kd in MuJoCo passive damping:
  passive damping lies outside motor `ctrlrange` and cannot share Isaac's total-effort limit.
  Retained passive damping or missing effort limits therefore makes an implicit replay explicitly
  diagnostic; formal construction fails before rollout.

## ONNX Metadata Contract

The whole obs/action contract travels with the model. `scripts/play.py` bakes into the ONNX:
`joint_names`, `default_joint_pos`, `action_scale`, `joint_stiffness` (kp), `joint_damping` (kd),
`body_names`, and the clip clock keys `clip_seg_lengths` / `clip_strike_phases`. Consumers
hard-validate at load and fail on any missing key, non-bijective joint map, obs input other than
[1,175]/[1,180], or any kp/kd ≤ 0 (zero-gain guard): the C++ runner (`pp_onnx_policy.hpp`), the
MuJoCo evaluator (`mujoco_eval_onnx.py`), and the contract diff tool
(`scripts/inspect_a3_deploy_contract.py`).

### Formal execution provenance (training-contract schema 3)

`hope_metadata_schema_version=2` remains the ONNX packaging/layout schema. A distinct immutable
`training_contract_schema_version=3` now binds the checkpoint and export to execution facts read
from the instantiated environment, not copied from YAML comments:

- articulation and action joint order (identity is required), default q, action scale and
  `use_default_offset`;
- per-joint actuator integration type, kp/kd, armature, effort/velocity
  limits, PhysX friction backend/semantics/units, the 31 soft q-des limit pairs
  and whether q-des clamp was active;
- physics dt, policy dt and decimation (`policy_dt = physics_dt * decimation`);
- exact actor term names/dims/history, articulation and tracked body
  order/indices, anchor and motion segment lengths, per-clip FPS and the
  schema-2 motion-kinematics contract;
- task timing/target/bank facts needed to prevent exporting an old actor under a new evaluation
  recipe.
- the canonical racket-point identity and wrist-local offset. See
  [racket_contact_geometry.md](racket_contact_geometry.md) for the distinction
  between the URDF site, physical face centre and ball centre at contact.

A command-bearing 62-D actor term additionally requires
`actor_leg_ref_mask_provenance_epoch=1`. Only the exact canonical masked/unmasked callable, or a
strictly empty `functools.partial` around it, may mint the epoch. Any bound positional/keyword
argument, wrapper or copied marker is non-authoritative because it can change the configured
command semantics. Epoch 1 plus absent `actor_leg_ref_mask` proves unmasked; epoch 1 plus the
true-only `actor_leg_ref_mask=1` proves masked. The ONNX binding hashes that fact together with the
training-contract and source-checkpoint SHA; old artifacts cannot acquire it by metadata backfill.

The adjacent `params/training_contract.json` is content-addressed; its SHA and schema are embedded
in every newly saved checkpoint. Only a fresh schema-3 run or a resume from an exact SHA-bound
schema-3 checkpoint can keep `training_contract_lineage_exact=1`. Legacy/missing/mismatch overrides
remain diagnostic forever and cannot be “washed” into formal status by one continuation save.
Native and standalone exporters verify checkpoint↔JSON binding, write
`source_checkpoint_sha256`, and derive normalization truth from the actual graph/checkpoint.
For a non-baked normalized actor, the sidecar/runtime transform is exactly
`(obs - mean) / (std + eps)`. Constant features may have `std=0`; validity
requires finite non-negative std, finite non-negative epsilon, and a strictly
positive `std+eps` divisor in every dimension. This numeric guard is shared by
the Isaac checkpoint compatibility loader and MuJoCo sidecar path.

Publish-capable C++ and formal BankExam require: metadata schema 2, exact training schema 3,
baked empirical normalization, self-consistent dt/decimation, `qdes_clamp=1`, 31 soft-limit pairs,
and the source checkpoint/contract hashes. Older artifacts may only be opened under process-wide
no-publish diagnostics.

Formal evaluation additionally binds runtime facts that do not belong in the
training ONNX: immutable question schedule, bank/source-family SHA, per-attempt
noise seed and hold, ready-state hash, scorer/physics source and the resolved
simulator execution contract. The schedule artifact contains content-addressed
atomic question IDs and an exact per-clip quota; both simulator legs must emit
the same schedule SHA and ordered IDs. `hold_steps=H` means exactly `H` policy
actions on the ready-stand reference followed by one action on raw clip frame
0; this release-frame rule is hashed into the schedule artifact.
For MuJoCo BankExam the execution contract also records the SHA of the exact
standalone `stage1_question_bank.py` loader selected from the current checkout.
This keeps exam validation independent of Isaac package imports and ambient
`HOPE_STAGE1_QB` shell state.

Carry-state BankExam is explicitly diagnostic. Its product continuity metric
uses only scheduled rows that have a following paper item. A legal return plus
natural clip completion counts as `returned_and_recovered_to_next`; a
post-strike fall or tracking guard may retain `returned=true` but fails the
recovery conjunct. The terminal paper row has no next opportunity and is not
in this metric's denominator.

The Isaac companion evaluator does not add an actor term or change the action
contract. It restores the saved train-split command, performs one nominal
stand reset, then installs the evaluator-owned exam timing and the complete
atomic row (contact, incoming velocity/spin, demanded velocity/normal) before
re-reading the first actor observation. One environment owns one schedule
item; there is no cursor, wrap or replacement question. Physical falls, guard
resets and timeouts remain rows in the original denominator, while any
external truncation invalidates the whole cell. The MuJoCo leg resets from the
common MJCF `stand` keyframe and consumes the same artifact. A
teacher-reference reset, direct PhysX-friction-number proxy or historical
checkpoint without exact train-family binding forces
`evaluation_contract_exact=false`.

For the diagnostic teacher-reference reset, the embedded motion's pelvis position is the link
origin while a checkpoint-bound schema-3 export carries each clip's linear-velocity point in ONNX
metadata as `motion_body_lin_vel_points`. A `center_of_mass` clip needs the rigid-point conversion recorded in
`racket_contact_geometry.md`; a declared `link_origin` clip is assigned directly and remains
exact-ineligible. Pre-field exact schema-2 exports are narrowly interpreted as all-COM. An old
inexact/missing aggregate flag does not identify the velocity point and must fail before this reset,
not silently select either branch. The formal `stand-keyframe` path does not consume motion qvel.

## Joined-source first-tick diagnostic JSON

`a3_deploy_onnx_ref_pingpong --first-tick-json ABS_PATH` is a no-publish-only diagnostic interface,
not a new actor observation and not Gate3 evidence. It observes the first 179-D actor candidate for
which the current implementation reports planner engaged/hold; PASSIVE/no-command/wait/invalid/
recovery rows do not count. Because the corrected same-tick planner snapshot and shared payload
epoch are not merged, this predicate is not an atomic planner certificate.

Both outer document and payload fix `evaluation_contract_exact=false`. The payload also fixes
`planner_snapshot_exact=false`, `native_sample_alignment_exact=false`,
`source_binary_binding_exact=false`, `source_semantics_closure_exact=false` and
`runtime_artifact_closure_exact=false`, with non-empty
reasons. Gate3/Gate3B or promotion consumers must reject v1. The diagnostic vector layout is:

| field | size | frame/layout/source |
| --- | ---: | --- |
| `qpos` | 38 | closest-receipt joined vendor-world pelvis `xyz,wxyz`, then RobotState 31-joint positions |
| `qvel` | 37 | closest-receipt joined vendor-world pelvis linear/angular velocity, then RobotState joints |
| `base_pose` | 7 | joined vendor-world pelvis `xyz,wxyz`; byte-value equal to `qpos[0:7]` |
| `racket_pose` | 7 | joined vendor `right_racket` site `xyz,wxyz` |
| `target` | tuple | world-table position/velocity, raw mount +Y/A actor normal, rho=0, time, side and valid |
| `obs` | 179 | exact `deploy_parity_face179` actor row |
| `policy_action` | 31 | raw actor action before command decoding/neck/leg overrides |

The actor row uses a different base orientation semantic from native qpos: external-base position
plus yaw-aligned pelvis IMU. The JSON therefore carries `policy_observation_base_pose` separately,
with its payload SHA, source age, distance to native base, projected-gravity difference and
quaternion dot. Current source gates require <=3 cm position and <=0.02 tilt-vector disagreement;
yaw can differ by design and is recorded, not silently equated. Joint q/dq come from the
`RobotState` consumed by that actor candidate.

Root linear velocity is never reconstructed from acceleration or filled with zero. It must come
from vendor `pelvis_framelinvel`; missing/stale/skewed native pose/twist/racket input aborts without
an artifact. Native topic source stamps must advance strictly during one sidecar lifetime and the
C++ consumer accepts only positive even committed generations; replay/regression/reset cannot
refresh old state. The tracked `right_racket` site offset equals the formal wrist control point, and its
native position must match formal FK within 5 mm. The vendor ROS publishers use asynchronous
publish-time stamps and expose no shared MuJoCo sample sequence, so the bounded 20/30 ms joins prove
proximity only, never same-tick alignment.

Every vector/target and the full canonical payload has a SHA-256. The envelope/model/training/
checkpoint, joint names, frames, sync/receipt clocks and reviewed native-source subset are recorded;
no exact source-commit/binary claim is emitted. ONNX Runtime parses the same stable file bytes whose
SHA is recorded. Output is canonical-path, mode-0600, fsynced atomic no-replace. Parser-resolved
config→MJCF, publisher/transitive artifacts and actual backend tick remain OPEN. Full details are in
`docs/operations/run_gate3_first_tick_harness.md`.

## Deploy-Available Signal Set

What the real system can observe (team contract, 2026-07):

- Robot-side (always): joint encoders (q, dq), pelvis IMU (quat, gyro), torso IMU (orientation),
  previous action.
- Mocap during PLAY (ChingMu/VRPN, 300 Hz): robot base (pelvis) pose, ball position. Ball
  rotation/spin is planned for the physics-modeling phase — not yet measured.
- Mocap during DATA-COLLECTION/physics-calibration only: additionally racket pose, table 4
  corners, net 2 corners.

Rules: actor observations must be built only from deploy-available signals; rewards run in sim
only and may use privileged state; the critic may be privileged. The current 175-D actor uses only
robot-side signals + planner targets (no mocap terms at all). NOTE: when the mocap→planner loop is
bridged, the HOPE-world → robot-frame target transform will need the mocap base pose at the
interface boundary even though the actor obs does not — that transform currently has no owner and
must be designed with the bridge (see G07 Next Steps).

## Table-Tennis Physics Scene Observation (experimental)

`HOPE-TableTennis-AgibotA3-v0` is a G04/G08 physics/visualization scene, not the accepted deployment
WBC policy. Its current actor observation is robot proprioception plus ball state in the base frame:
`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel`,
`last_action`, `ball_pos_b`, `ball_vel_b` (position/velocity only — no spin). The critic mirrors
these without noise. If this scene becomes a returner baseline, record dimensions, normalization,
reward targets, and deploy compatibility here first.

## Contract Knobs

- Control rate is model metadata, currently 50 Hz (physics dt 0.005 × decimation 4). Formal
  consumers reject a runtime rate that disagrees with the model instead of assuming 50 Hz.
- Training target source: `target_mode: uniform` with per-clip 3-D position and velocity boxes
  centered on the reference blade strike state (`pos_range_per_clip` / `vel_range_per_clip` in
  `cfg/task/HOPEPingPongDeployParity.yaml`); `strike_phase_per_clip: [0.47, 0.333]` on the
  re-grounded `_hopex` clips. The fixed `x=0.4` hit plane with (y,z)-only sampling is superseded.
- `racket.target_mode: reference_perturbed` remains a non-default option (it was
  the default on the pre-merge `rsi-on-wrap-progress-fix` branch): the target
  center is computed from each imitated clip's own strike-frame racket FK state
  (position, velocity, face normal), and the long-run distribution widens
  through success-gated perturbations. Either mode changes target generation
  only, not the observation/action tensor contract.
- Face-command grading: `racket.face_command_pairing: shared_plus_y` is the production convention.
  The explicit `legacy_signed_vs_A` value exists only for controlled historical diagnosis. In both
  modes the actor's demanded normal, when present, remains the shared +Y/A-frame command; only the
  reward/privileged-observation/metric pairing changes.
- Clip wrap for HOPE ping-pong does not teleport the robot mid-episode: the
  target and reference clip/time resample, but the policy must physically carry
  the body between swings (`MotionCommandCfg.wrap_teleport` defaults to false;
  the HOPE task YAMLs keep it explicit as `motion.wrap_teleport: false`). True episode resets
  still use reference-state initialization, except for the `stand_start_prob`
  fraction of envs that start from the default stand pose.
- Normalization: per-term Unoise with `enable_corruption` on the policy group (values in the actor
  table above).

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency, or joint-order change must update this file and the relevant gate docs.

## 2026-07-13 formal-179 planner/base transaction contract

This changes deployment transport and engage safety, not the 179-D actor tensor, 31-D action
tensor, normalization or checkpoint bytes.

- Formal racket schema 3 has exact length 20. Besides valid/time/target/physical face-B/side, it
  carries a shared `control_epoch`, strictly increasing racket command sequence, exact
  `base_sequence_ref` and mapped source-monotonic time. After clip selection only the normal is
  converted by frozen `[+1,-1]` into raw mount A; position and velocity are unchanged.
- Formal base schema 2 has exact length 12 and carries valid pose, the same `control_epoch`, a
  strictly increasing base sequence and mapped source time. Legacy sizes cannot follow a formal
  stream; downgrade poisons the lease.
- The C++ mailbox keeps at most 128 exact formal base rows. The referenced historical row proves
  causal pairing only; latest tick-start base independently owns policy-frame side/target/yaw,
  base-low, first observation, active abort and recovery safety.
- Python and C++ share the finite workspace and source-time continuity bounds: x/y `[-3,+3] m`, z
  `[0.4,1.5] m`, translation `0.05 + 8*dt` metres and quaternion shortest-angle
  `0.15 + 12*dt` radians. Proven-old delayed rows are rejected before continuity comparison; a new
  implausible row revokes without replacing the last good baseline.
- Before every formal-179 actor call, including level-0 recovery/static hold, latest base must be
  finite, fresh, plausible and at/above `base_low`. The latched engage epoch and base revocation
  generation must remain usable after a swing; failure returns zero gain and re-arms.

The guarantee is sampled at each actor call, not an asynchronous stop inside an already executing
20 ms compute interval. These hard bounds still need vendor-trajectory validation, and this source
contract is not a Gate3 runtime result.
