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
- Shared joint names and joint order.
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
- A working 31-DOF joint-order YAML exists at `hope_training/config/joint_order_agibot_a3.yaml`.
- `reimplement.md` records that the A3 task registers and the env launches headless with finite rewards on the copied A3 ping-pong URDF asset.
- `origin/train_1` adds `HOPE-TableTennis-AgibotA3-v0`, a HOPE-frame Isaac Lab table/net/ball/A3 scene with modular geometry constants, optional drag and Magnus force hooks, table/net/floor contact materials, 400 Hz physics, CCD enabled, ball serve reset, and placeholder returner rewards.
- The table-tennis scene now includes a tracked Purdue PACE table/net USD visual overlay under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`. Physics still comes from invisible cuboid colliders; the USD is visual-only.
- Table-tennis ball/table contact now follows Purdue PACE materials by default: ball mass `3.4 g`, ball restitution/friction `0.9/0.1`, table restitution/friction `0.95/0.4`, multiplicative combine for an effective ball-table normal restitution of `0.855`. HOPE-calibrated aero drag is available but off by default for Purdue parity.
- `tests/test_table_tennis_geometry.py` covers table/frame geometry and pure drag/Magnus math; the drag/Magnus tests skip automatically if host `torch` is missing.
- Sim parity between MuJoCo and Isaac is established via `hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py`: MuJoCo with `implicitfast` integration and kd placed in `dof_damping` matches Isaac's `ImplicitActuator` (stable rollout), while the AGI deploy sim's explicit Euler PD diverges in ~0.1 s. Documented in `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md`.
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
locally synced exact B/C private sources also passed with maximum inverse errors below `2.23e-16`
and pairwise-distance error below `4.17e-17 m`. No output was consumed or published in this change;
schema-2, L0, vendor L1, table/net clearance, dynamics, simulator, training and hardware remain
unrun. G04 remains `Partial`.

For M0, the canonical-beta materialization still has null A3 stance fields. The downstream exact-GMR
plan now freezes canonical foot sites and tolerances, while initial/terminal `d_xy` and pass remain
null until robot-coordinate evidence exists. G04 remains `Partial`.

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
