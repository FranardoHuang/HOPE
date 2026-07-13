# G04 Sim Modeling In MuJoCo And Isaac

Status: Partial

## Goal

Build consistent A3 simulation assets in MuJoCo and Isaac so training and deployment can share robot semantics.

This gate is about robot model correctness, not RL performance.

## Inputs

- A3 URDF and meshes under `agi/URDF/`.
- Agibot MuJoCo/AimRT simulation materials under `agi/A3_MuJoCo_Sim/`.
- Isaac training scaffold under `hope_training/whole_body_tracking`.
- A3 deployment configs under `agi/code_deployment/`.

## Outputs

- Validated MuJoCo model.
- Validated Isaac asset.
- Shared joint-name set plus explicit content-bound mappings between GMR, runtime articulation and
  backend order domains.
- Shared racket mount definition.
- Sim model limitations.

## Related Directories

- `agi/URDF/`
- `agi/A3_MuJoCo_Sim/`
- `agi/code_deployment/a3_deploy_example/mujoco_sim_standalone`
- `hope_training/whole_body_tracking`
- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis`
- `docs/interfaces/joint_order_and_robot_state.md`
- `docs/interfaces/frames_and_coordinates.md`

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

- A3 model loads in MuJoCo.
- A3 model loads in Isaac.
- URDF/MJCF/Isaac asset agree on joint names, limits, and ordering.
- Standing pose, base height, and racket FK are checked.
- Contact and table/ball parameters are documented.

## Current State

Done:

- A3 URDF and MuJoCo support materials exist.
- Agibot MuJoCo sim source exists.
- Tracked deploy subset includes standalone MuJoCo configs.
- On 2026-06-25, this harness restored the ignored package-local A3 Isaac asset under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/` from tracked `agi/URDF/A3T2.5-URDF-std-pingpang/` materials and rewrote URDF mesh paths to local `../meshes/` references. Host verification found `86` mesh references and `0` missing files.
- The branch now includes an A3 Isaac/BeyondMimic robot config using the Agibot-provided ping-pong URDF path, official joint/body names, deploy-transcribed PD gains, standing pose, and action-scale logic.
- `scripts/prepare_a3_isaac_asset.py` now prepares the generated Isaac asset from `agi/URDF/A3T2.5-URDF-std-pingpang/` and verifies the prepared `model.urdf` by parsing all mesh references. The check rejects stale `package://.../meshes` references, verifies every `../meshes/...` file exists, and requires `right_hand_pingpang_Link.STL`, `pingpang_red_Link.STL`, `pingpang_black_Link.STL`, and `pingbang_ball_Link.STL`.
- The two distinct 31-DOF column domains are explicit: GMR `dof_pos` and runtime/schema-2
  `joint_pos` have content-bound tables and a fail-closed bijection. The legacy YAML mirrors only
  the GMR source order; see the [joint-order interface](../interfaces/joint_order_and_robot_state.md).
- `reimplement.md` records that the A3 task registers and the env launches headless with finite rewards on the copied A3 ping-pong URDF asset.
- `origin/train_1` adds `HOPE-TableTennis-AgibotA3-v0`, a HOPE-frame Isaac Lab table/net/ball/A3 scene with modular geometry constants, optional drag and Magnus force hooks, table/net/floor contact materials, 400 Hz physics, CCD enabled, ball serve reset, and placeholder returner rewards.
- The table-tennis scene now includes a tracked Purdue PACE table/net USD visual overlay under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`. Physics still comes from invisible cuboid colliders; the USD is visual-only.
- Table-tennis ball/table contact now follows Purdue PACE materials by default: ball mass `3.4 g`, ball restitution/friction `0.9/0.1`, table restitution/friction `0.95/0.4`, multiplicative combine for an effective ball-table normal restitution of `0.855`. HOPE-calibrated aero drag is available but off by default for Purdue parity.
- `tests/test_table_tennis_geometry.py` covers table/frame geometry and pure drag/Magnus math; the drag/Magnus tests skip automatically if host `torch` is missing.
- A historical diagnostic found stable rollouts when MuJoCo used `implicitfast` with kd in
  `dof_damping`, while the AGI explicit-Euler path diverged in about `0.1 s`. That did **not**
  establish actuator parity: passive kd sits outside motor effort clipping. The 2026-07-14 source
  correction below replaces bound execution with Isaac's total `clip(P-D)` law and keeps the old
  passive-kd mode diagnostic-only. The older observations remain in
  `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md` as historical evidence, not an exactness claim.
- Hardware SDK parity for joint order is established: the `pp_joint_map` backend order was verified slot-for-slot against AGI `robot_io::MakeA3Layout31()` — a checked bijection (`agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md:137-139`).

Not done:

- This Codex shell has not independently run Isaac because the required GPU/Isaac environment is not active here.
- The table-tennis scene has not yet been verified in-sim in this Codex shell with `scripts/play_table_tennis.py`.
- Self-collision is disabled in the Isaac config due to overlapping wrist/racket collision meshes; a cleaner Isaac collision asset is still needed before re-enabling it.
- The table-tennis scene is not yet a trained returner or accepted sim-to-real baseline; it is a G04/G08 candidate scene.
- The internal main branch intentionally keeps multiple A3 asset layers: ping-pong URDF source for WBC, standard non-racket `agi/URDF/a3_t2d5/` for comparison, and Agibot MuJoCo/AimRT ping-pong MJCF/collision materials for parity. Do not delete the standard `right_hand_Link.STL` or MuJoCo collision assets without a recorded replacement.

## Current Verification Commands

Plain host checks:

```bash
python3 scripts/prepare_a3_isaac_asset.py --check
python3 hope_training/whole_body_tracking/tests/test_table_tennis_geometry.py
python3 -m py_compile hope_training/whole_body_tracking/scripts/play_table_tennis.py
python3 -m py_compile hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/*.py
```

GPU/Isaac checks, inside the sourced training environment:

```bash
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
hope_isaac_py scripts/play_table_tennis.py --fix_base
hope_isaac_py scripts/play_table_tennis.py --enable_aero --headless --steps 300
```

## Risks

- Differences in actuator model, contact, timestep, or joint order can break sim-to-sim transfer.
- URDF import can lose inertial or collision fidelity.
- Racket mount errors directly corrupt planner-to-WBC training.
- The vendor ROS `SimReset` nonzero base-twist interface is not frame-exact: its odom/world angular
  velocity is copied into body-local freejoint qvel. Named-keyframe zero-velocity reset is safe;
  nonzero absolute-twist replay remains blocked pending a versioned point/frame contract and test.

## Next Steps

1. Load the same standing pose in MuJoCo and Isaac and compare FK.
2. Clean or replace Isaac collision geometry so self-collision can be revisited.

## Audit update 2026-07-10: racket point and plant semantics

- The canonical racket control point is now one literal point across URDF,
  MuJoCo, Python, ONNX metadata and C++: `pingpang_red_Link` / `right_racket`
  at wrist-local `[0.210210, 0.032078, 0.032036] m`. The red rubber area
  centroid is only `1.264 mm` away in-plane.
- Isaac Lab 2.1 `body_pos_w` is a link-origin position but
  `body_lin_vel_w` is a COM-point velocity. The old racket calculation mixed
  them, producing about `0.401 m/s` forehand and `0.598 m/s` backhand error on
  the audited V5/hopex poses. Tracking and the Phase table-tennis environment
  now require `body_link_lin_vel_w`; wrist fallback uses the official offset
  and `omega x r`. Missing link-point channels fail closed.
- The Phase environment also reconstructs the site from
  `right_wrist_yaw_Link` when the URDF importer merges the zero-mass fixed
  paddle links, instead of silently disabling paddle contact.
- Existing `AGIBOT_A3_CFG` friction values are not the documented constant
  MuJoCo torques: Isaac Lab interprets them as dimensionless, load-dependent
  PhysX coefficients. They are uncalibrated proxies. Do not claim
  PhysX/MuJoCo plant equality from equal-looking numbers.
- Exact physical ball contact is not yet the compatibility contract. Current
  `site_colocated_v1` co-locates the ball centre with the site; centred rigid
  contact would offset the ball centre by `20.040 mm` on the red face and
  `33.232 mm` on the black face. Promote this only through the versioned
  `exact_face_contact_v2` experiment described in
  [racket_contact_geometry.md](../interfaces/racket_contact_geometry.md).

Verification:

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_racket_geometry_contract.py \
  hope_training/whole_body_tracking/tests/test_motion_kinematics_contract.py \
  hope_training/whole_body_tracking/tests/test_table_tennis_geometry.py
```

Remaining G04 limits include rigid-body mass/inertia/COM and asset SHA parity,
contact/solver parameters, DR distributions and calibrated joint friction.

## Audit update 2026-07-12: fit reference and stand diagnostic

- The existing Torch ball-physics parity test no longer depends on a lost default Mac path. Its
  default NumPy reference is reconstructed in-repo from the committed fit-lineage contact kernel
  and venue YAML, and every run prints their content SHA tuple. Explicit missing `RECORD_DIR` and
  zero/non-finite normals fail loudly. Seven dependency-light tests pass. A local Torch CPU
  environment also reports `ALL PASSED`: table/paddle contact maximum error is below `4.63e-9`,
  RK4 position error is zero at reported precision, and first-landing error is `0.000 mm`.
- `scripts/view_a3_stand.py` now provides a root-source-bound plain-MuJoCo diagnostic for the vendor
  MJCF. It parses production default pose/Kp/Kd from the tracked header, leaves neck joints passive
  per the 29-DOF PD_STAND contract, and can report finite state, pelvis tilt/z and foot contacts.
  Source/identity tests pass. A 2026-07-13 local CPU rerun completed 10 seconds with finite state,
  `1.816 mm` maximum pelvis-z drift, `0.311 deg` maximum tilt and both feet in contact for `100%`
  of samples. This confirms the static vendor MJCF/axis/gravity/PD stand can remain upright; it
  does not run the policy, change the MJCF/integrator, or constitute a Gate3 result.

See `docs/research/yikang_selective_integration_20260712.md` and
`docs/operations/run_deploy_dryrun.md`. G04 remains `Partial`.

## Audit update 2026-07-12: MuJoCo training-v0 plant boundary

A read-only P0 preflight found that the tracked vendor `a3_pingpong.xml` is a robot-dynamics scene,
not a physical table-tennis scene: it contains the floor, A3 and `right_racket` site but no ball,
table or net. The analytic `BallPhysics` source expects a `ball` mocap body, yet the current
`MujocoSimModule::SimLoop()` neither owns nor steps that driver. A first native MuJoCo trainer can
therefore target balance and strike-state tracking, but cannot claim physical hit, landing or return
training until a separate instrumented scene/runtime closes those assets and contacts.
The `mujoco-ball-wiring@4607410` item shown in `docs/NOW.md` is a separate handoff candidate, not
current main: at this audited base it is not a main ancestor, and vendor build, QoS/transport, GUI
and runtime behavior remain open. It can close the missing-scene prerequisite only after reviewed
merge and independent contact/landing verification.

The audit also separates two effective plants that can load the same MJCF bytes. Current formal
BankExam may overwrite timestep, passive damping/frictionloss, actuator integration and q-des
semantics to reproduce an Isaac schema-3 profile. Vendor Gate3 keeps the resolved 1 ms MJCF plant,
recomputes explicit PD every simulator step and includes production hard-clamp/neck/runtime action
post-processing. MJCF SHA alone does not prove equality; future contracts must bind resolved plant
facts, mesh closure and runtime flags. Reproducible source commands and the two-profile table are in
[the MuJoCo training-v0 preflight](../research/mujoco_training_v0_preflight_2026-07-12.md). No
simulator ran and G04 remains `Partial`.

## Audit update 2026-07-13: formal base plausibility source gate

Planner and runner now share one source-bound gross-workspace and source-time continuity check
before formal-179 inference: x/y `[-3,+3] m`, z `[0.4,1.5] m`, translation
`0.05 + 8*dt` metres and quaternion shortest-angle `0.15 + 12*dt` radians. A new ordered
implausible sample revokes the base lease, clears exact history and cannot replace the last good
continuity baseline; proven-old delayed packets are discarded before comparison.

These constants block finite teleport/glitch inputs but are not vendor-plant validation. They must
be compared with content-addressed vendor MuJoCo pelvis trajectories before claiming they are
neither too loose nor liveness-breaking. No Isaac asset, MJCF, contact parameter, trainer,
checkpoint or hardware changed; G04 remains `Partial`.

## Audit update 2026-07-13: S0/M0 post-GVHMR model handoff boundary

The completed S0/M0 GVHMR structural outputs now have two independent, content-addressed
post-GVHMR preregistrations. The consumer binds the tracked result summary, execution record,
final queue state, every per-asset binding/audit/PT and the existing exact canonical-beta donor.
Host static validation and eight red-team tests pass, including lineage mutation, extra binding,
duplicate-key/non-finite JSON and no-clobber publication controls.

This does not run GMR or create schema-2 motion. The next stage is only a separately
preregistered canonical-beta materialization. GMR remains blocked on its ignored/private source,
loader, SMPL-X and runtime closure; schema-2 remains blocked on exact GMR output plus runtime body
order, link-origin positions and COM-point velocities. M0 robot-coordinate foot stance and S0
strike behavior are also unmeasured. Commands are in
[`run_motion_post_gvhmr_exact.md`](../operations/run_motion_post_gvhmr_exact.md); G04 remains
`Partial`.

## Audit update 2026-07-13: detached Isaac A3 asset closure

An exact Git worktree does not contain the ignored package-local A3 URDF/mesh/config tree. The
signed-face L1 v4 launch exposed this before its first learning iteration. The v5 preflight now
binds both the clean `6d93bcb` restore source and clean detached `882fea4` target to exactly 46
regular files, 15,378,264 file bytes and canonical tree SHA `0137f59b...26c6`; symlinks, special,
missing or extra files fail closed. Restore steps are in
[`setup_local_sync.md`](../operations/setup_local_sync.md). This proves asset byte closure only,
not Isaac/MuJoCo model equivalence or behavior; G04 remains `Partial`.

## Audit update 2026-07-14: repeated Kit boundary is table USD load before PhysX

The byte-bound v6/v8 postmortem narrows both failed D launches to the same Isaac scene boundary:
each Kit log ends on loading the same 683,433-byte table USD (SHA-256 `c6fc99a8...996`) and never
prints the following PhysX-context line. The adjacent C controls cross the same boundary in
2.339/3.031 seconds and reach iteration 24. This is evidence about scene composition order, not a
model, reward or learning result. Postmortem capacity was ample, but historical transient driver or
filesystem delay remains unknown; Carbonite shared-memory residue is correlation, not a proven
cause, and `dmesg` was unreadable. The tracked [result ledger](../../configs/phase1_signed_face_boot_root_cause_results_20260714.json)
and [design-only diagnostic prereg](../../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)
keep all launch and training permissions false. G04 remains `Partial`.

## Audit update 2026-07-13: S0/M0 exact donor canonical-beta materialization

The two runtime handoffs have since completed at exact `4,970/9,242` bytes and SHA-256
`d57a93e0...a1054` / `60c55150...088ef`. A separate canonical-beta consumer is now preregistered:
it reuses the audited historical PT replacement/save-reload primitives but injects the old exact
same-performer donor instead of recomputing a new cohort median. Host static plus synthetic
save/reload tests are `15 passed, 1 skipped`, with latest-main repository regression
`620 passed, 9 skipped`; the bound Pod1 CPU runtime has now completed
`static -> inspect -> consume` for the real one-plus-four PTs. S0/M0 completion manifests are
`964a7333...f1be3` / `5cef05f7...71a65`; all five outputs preserve every non-beta leaf bit-exact
and copy donor `canonical_betas.json` SHA `f405ba45...4cbf2`.
For M0, all A3 foot sites, initial/terminal `d_xy`, tolerances and pass result remain null until a
separately preregistered exact GMR produces robot-coordinate evidence. G04 remains `Partial`.

## Audit update 2026-07-14: B/C exact whole-motion SE(2) materializer

Franco backhand-loop B/C rank-0 proposals now have two independent no-clobber preregistrations and
one restricted-pickle CPU consumer. It binds the exact main selection ledger, then resolves each
source path/SHA/bytes through the exact canonical-beta GMR registry. The only permitted spatial
change is one proper ground-preserving [SE(2)](../DEFINITIONS.md) left action on the full floating
root trajectory; root xyzw orientation is yaw-left-multiplied, root Z/fps/frame count/joints and
non-spatial fields remain exact, and explicit world root velocities would rotate without
translation. Unknown payload fields and non-NumPy pickle globals fail closed.

Ten focused tests and the `656 passed, 9 skipped` host repository suite pass, including
save/reload inverse, pairwise rigid-distance, grounding,
mirror, unsafe-pickle, report-last and no-clobber negative controls. Read-only inspect of the two
exact B/C private sources passed, then Pod1 CPU-only `consume` published both motion/report pairs.
B motion/report SHA is `27827912...ad6` / `a238c077...df3`; C is
`0dd981a6...f48b` / `b3b93d2c...f67`; maximum inverse error is below `2.23e-16` and pairwise-distance
error below `4.17e-17 m`. This only unlocks a separate schema-2 preregistration. L0, vendor L1,
table/net clearance, dynamics, simulator, training and hardware remain unrun; certificate count is
still zero. G04 remains `Partial`.

## Audit update 2026-07-14: GMR-to-runtime joint-order source gate

The pre-schema-2 audit found and removed a false shared-order claim. GMR pickle/CSV `dof_pos` uses
the controller/MJCF order, while Isaac articulation, schema-2 NPZ `joint_pos` and exact ONNX actions
use an interleaved runtime order. Two tracked order tables plus
`configs/a3_joint_order_bijection_v1.json` now bind file/name SHAs and both 31-element permutations.
The validator also parses the legacy YAML and Python source-order mirrors without importing Isaac,
and requires complete ONNX `joint_names`, `articulation_joint_names` and identity `action_joint_ids`.
Duplicate/missing/extra/wrong-length names, declared permutation drift, partial/wrong-order metadata,
bad array shape, duplicate JSON keys and NaN/Inf all fail closed; focused tests are `12 passed`, and
the dependency-light repo suite on `origin/main@5734dc8` is `733 passed, 10 skipped`.

`csv_to_npz_mujoco.py` now consumes this contract directly. The historical L0 auditor remains
byte-exact because an executed prereg binds its source SHA; the validator AST-checks its target-order
literal instead. This is source-only: no B/C private asset, MuJoCo forward kinematics, schema-2
output, L0, simulator, training or hardware ran. A separate exact B/C prereg must still bind the
vendor MJCF closure, runtime body order, 30→50 Hz resampling, link-origin/COM-velocity convention and
no-second-HOPE-rotation rule. G04 remains `Partial`.

## Audit update 2026-07-14: B/C schema-2/FK source gate

The previously open preregistration is now mechanically closed at source level. Independent B/C
plans (`3d71cc02...d33ae` / `662b8c4c...57e31`) bind the accepted private SE(2) pickle and report
paths, bytes and SHA, distinct no-clobber output roots, `91/98` input frames at 30 Hz and expected
`151/163` schema-2 frames at 50 Hz. Their shared runtime contract (`3d32b146...ed6e8`) binds the
restricted NumPy-pickle loader, GMR-to-runtime joint bijection, exact 32-body column order, schema-2
point semantics, converter helpers and a canonical hash over the vendor MJCF plus all recursively
included/external assets. The current model has one XML, zero includes and 74 unique referenced
meshes; aggregate closure SHA is `e0381752...962de`.

The consumer (`33cf23ee...caebd`) admits only `--hope_frame off`: both private roots already carry
the accepted HOPE-frame SE(2), so a second rotation has no CLI representation. It freezes linear
root/joint interpolation plus shortest-path quaternion SLERP, then records link origins from MuJoCo
`xpos` and center-of-mass velocities from the gradient of `xipos`. The exact formal donor ONNX SHA
is bound to the required three-row metadata expectation, but this source gate honestly records that
the metadata was not re-extracted from the ONNX here. A later read-only `inspect` must hash that ONNX,
extract and compare the rows, restricted-load each private pickle and load the vendor model before
one no-clobber materialization may run.

Focused dependency-light tests are `17 passed`; against `origin/main@7679b30`, the repository
`tests/` suite is `782 passed, 10 skipped`. Both tracked `static` commands pass without reading private assets or importing
MuJoCo/ONNX runtime. Negative tests cover closure/metadata/order/time/
point/frame drift, B/C namespace overlap, duplicate JSON, wrong plan SHA and `--hope_frame on`.
No FK, schema-2 output, L0/L1, simulator, training or hardware ran; certificate count stays zero and
G04 remains `Partial`. See the [experiment](../experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)
and [operation](../operations/run_motion_spatial_retarget_screen.md).

## Audit update 2026-07-14: B/C no-write runtime inspection and consume activation

The two next-gate inspections have now run in the clean detached Pod1 checkout
`748b6d5fe24bfe58915c34d8dfe09f254f8e4957`. The default Python 3.12.3 environment failed closed
with rc=2 because `onnxruntime` was absent; that attempt wrote nothing and is not counted as a pass.
The existing `/workspace/hope_mjeval_venv` environment bound Python 3.12.3, NumPy 2.5.0,
ONNX Runtime 1.27.0 and MuJoCo 3.10.0. With the same exact tool, plans, donor, private PKL/report and
MJCF closure, B and C passed separately at 91 and 98 input frames. Both output roots remained absent
and the source checkout remained clean.

The tracked receipt is
`configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json` SHA
`8e2d2d2d...3fb61`. Inspection loaded and name-checked the vendor model but did not evaluate the
151/163-frame FK trajectories, step dynamics, or write schema-2. The reviewed-next-step activation
`366d59d5...d6337` is source-only and still records zero attempts started. It authorizes at most one
serial no-clobber consume attempt per existing B/C output root, requires report-last publication,
and forbids automatic retry after failure. The activation itself authorizes no L0/L1, table/net,
dynamics, simulator, training, formal-motion or hardware action. No consume ran in this change;
activation-focused tests are `28 passed`, combined prereg/activation tests are `45 passed`, and the
latest-main repository suite is `822 passed, 10 skipped`. Schema-2 materialization and certificate
count remain zero, so G04 stays `Partial`.

For M0, the canonical-beta materialization still has null A3 stance fields. The downstream exact-GMR
plan now freezes canonical foot sites and tolerances, while initial/terminal `d_xy` and pass remain
null until robot-coordinate evidence exists. G04 remains `Partial`.

## Audit update 2026-07-14: bounded implicit effort and robot-only self-contact semantics

The MuJoCo evaluator no longer represents a bound Isaac implicit drive as clipped P plus unbounded
passive D. It computes and sends `clip(P-D,-L,L)` per substep; cancellation, same-direction, pure-D
and exact-boundary tests cover both signs. A retained MuJoCo passive damping column or missing effort
limit cannot realize this law and is diagnostic-only; formal construction fails closed.

Self-contact classification is now the intersection of two geoms in the explicit `pelvis_link`
articulation subtree. Dynamic balls, table/net bodies, mocap helpers and unrelated free bodies are
excluded even though their MuJoCo body IDs are nonzero. Because Isaac training has self-collision
disabled, classification now runs after every MuJoCo physics substep: formal BankExam fails on the
first classified robot pair, including a transient pair gone by the final substep, while diagnostic
protocols aggregate all substeps without mutating physics. The first control-rate-only implementation
was rejected by a second red team because it could miss a short collision after that collision had
already changed the trajectory. These are source/contract corrections, not evidence that Isaac and
vendor MuJoCo contacts or integration are behaviorally equivalent. G04 remains `Partial`; see
[the integration experiment](../experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md).

Pod2 subsequently collected the two previously skipped optional MuJoCo modules. The first real
collection rejected two invalid synthetic controls; after correcting only the test fixture, the
unchanged evaluator source passed `10/10` on MuJoCo Python `3.10.0`. This closes the optional
total-PD/self-contact/reference-reset source gate, not the vendor-plant or cross-engine behavior
gate; machine evidence is
[`mujoco_eval_optional_runtime_test_results_20260714.json`](../../configs/mujoco_eval_optional_runtime_test_results_20260714.json).
G04 remains `Partial`.

## Audit update 2026-07-14: S0/M0 exact GMR source/model gate

The next offline step now has two independent no-clobber batch plans and one shared runtime
contract. The consumer binds each of the five exact canonical-beta PTs, clean ignored GMR
commit/tree/bundle, converter/import/model closure, Python/pip, both retarget and canonical A3
joint/body orders, and an explicit 31-name/index qpos bijection. The canonical vendor A3 FK model is
bound as a 76-file tree, including `left_foot/right_foot` site identity and local positions. `inspect`
does not create an output root; `consume` publishes completion last. Focused unit coverage is
`12 passed`; repository regression is `645 passed, 9 skipped`.

A 2026-07-14 read-only receipt recovered the exact GMR tree, retarget MJCF/mapping bytes and SHA,
critical import SHA, Python binary SHA and normalized pip-freeze SHA. Direct retarget XML
joint/body/site parser output was truncated, and several absolute import/Python paths remain
unobserved. The two batch plans are preregistered, while the shared runtime keeps a machine-readable
16-item `required_unresolved_evidence` list and remains
`blocked_pending_exact_ignored_gmr_source_closure`. Both real `static` calls validate that receipt and
then fail rc=2 at the same enumerated gaps. No GMR, FK result, schema-2, simulator, training or
hardware ran; G04 remains `Partial`. See [the experiment](../experiments/motion_exact_gmr_s0_m0_20260713.md) and
[operation](../operations/run_motion_s0_m0_exact_gmr.md).
