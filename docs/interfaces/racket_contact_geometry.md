# Racket Control Point And Contact Geometry

Status: canonical control point fixed; exact ball-contact geometry is a versioned research migration.

## Canonical control point

The deployed policy controls the origin of `pingpang_red_Link`, expressed in
`right_wrist_yaw_Link` coordinates as:

```text
r_site/wrist = [0.210210, 0.032078, 0.032036] m
```

This is the same point in all active implementations:

- URDF fixed joint `pingpang_red_joint`;
- MuJoCo site `right_racket`;
- Python `racket_geometry_contract.RACKET_SITE_OFFSET_WRIST_M`;
- C++ `racket_control_point_offset_wrist_m()`;
- schema-v3 ONNX metadata `racket_control_point_offset_wrist_m`.

The C++ formal loader rejects a model whose embedded point differs by more
than `1e-9 m`. The older Python offset came from `pingbang_ball_joint`; it is
only `1.49 um` away, but is no longer used as the canonical value.

The site is near the geometric centre of the red rubber. STL integration puts
the red outer-face area centroid at this site plus
`[0.000893694, 0, 0.000893694] m`, a `1.264 mm` in-plane difference. It is
therefore a valid engineering control point, but it is not literally the
centre of the ball at contact.

## Measured-racket alignment contract

For canonical motion training, a measured racket rigid body is the teacher authority; the
URDF/MJCF site produced by retargeted joints is the model prediction. A mocap marker frame `M`
therefore cannot be relabelled as the official site `S`. Each calibrated content/session must bind
one frame-constant rigid transform `T_M_S`:

```text
T_W_S_measured(t) = T_W_M(t) T_M_S
E_site(t) = inverse(T_W_S_measured(t)) T_W_S_FK(q_retarget(t))
v_S_measured = v_M + omega_M x R_W_M r_S/M
```

The transform must be obtained from an explicit paddle calibration (including face sign and site
location), stored with source bytes/SHA, and reused unchanged over the clip. Per-frame or
per-action residual fitting after retargeting is prohibited because it would hide a wrong mount or
coordinate contract. The same fixed mount/site/normal must agree in URDF, MJCF, Isaac, Python,
C++ and export metadata.

For this training migration Franco designates the current URDF/MJCF racket as geometric ground
truth. `T_M_S` is therefore the content-bound map from the measured blade/face channel into that
official model site; it must not be used to move or refit the URDF to each clip. The unit/BVH bytes,
face sign and transform receipt remain mandatory because otherwise measured and FK teachers could
be cross-wired. A future real-racket mass/CoM/inertia calibration is separate sim-to-real physics
evidence and does not block this motion-retarget authority decision.

Retargeting must consume `T_W_S_measured(t)` as an end-effector constraint, then report the
measured-vs-FK residual per action: position, SO(3) geodesic orientation, signed-face normal and
point-consistent linear velocity p50/p95. Comparing only the wrist, marker origin or unsigned
plane is insufficient. The canonical full-phase solver preregisters these dynamic gates:

- full-phase position p95 `<= 0.05 m` and signed-face p95 `<= 10 deg`;
- hit-frame position `<= 0.05 m` and signed-face `<= 5 deg`;
- hit-frame point-velocity direction `<= 15 deg` and relative speed error `<= 20%`.

The ChingMu source was located on 2026-08-02, correcting the earlier “ball-only / raw missing”
claim. Its unit NPZ provides same-clock physical blade, butt and normal channels and its hit JSON
provides the signed face. `solve_chingmu_canonical_racket_full_phase.py` now constrains the current
MJCF `right_racket` site for every source frame, solves the striking-face sign independently per
action, and writes an admitted output only when all six gates pass.
`materialize_measured_racket_motion_npz.py` resamples the repaired qpos and measured paddle to the
same 50 Hz hit-aligned clock, rebuilds every robot body channel through the current MJCF, and writes
the measured teacher plus source/receipt SHA and solver-selected mount sign. `MotionLoader` requires
the complete bank contract; measured-teacher mode never falls back to FK and rejects a configured
or manifest face sign that differs from the retarget receipt.

The 2026-08-02 schema-v3 `73/73` conclusion is revoked. That solver/auditor used the site's local
`+X` column as the robot butt-to-blade axis. The URDF/MJCF site has identity orientation, but the
rigid visual racket inside `right_hand_pingpang_Link.STL` does not point along local `+X`: its visual
handle-butt to blade-centre axis is `(local +X + local +Z)/sqrt(2)`. The connected visual component's
PCA direction differs from this diagonal by only `0.066 deg`; the pinned rigid mesh SHA-256 is
`442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd`.

Re-auditing all 73 old schema-v3 files with the URDF-visual axis gives long-axis error p50/p95/max
`45.042/45.719/47.770 deg`; `0/73` pass the full or hit long-axis gate and `0/73` pass the full SO(3)
gate. Thus the old bank proved site position, point velocity and signed face only; it did **not**
prove wrist twist or full rigid-racket alignment and must not be used as canonical motion authority.

The repaired pipeline now binds both the diagonal axis and mesh SHA into the geometry payload,
solver report, schema-v4 measured NPZ and strict runtime loader. The corrected local sibling
`assets/motions/chingmu73_measured_v4_20260803` contains 73/73 solver outputs, 73/73 schema-v4 NPZs
and 73/73 independent FK audits over 5,107 frames. Its bank receipt SHA-256 is
`e6f0283f87401d004249689fbef30729fa7744ff6076a62c89996a945b727a82`. Independent worst-case
full-phase p95 position/face/long-axis/SO(3) is `49.31 mm / 6.769 deg / 7.920 deg / 9.521 deg`;
hit-frame worst-case is `.879 mm / .174 deg / .126 deg / .197 deg`, with velocity direction/relative
error `4.320 deg / 12.33%`. This is a complete **kinematic** admission, not mechanical admission.

The high-weight re-solve is mechanically rejected in its current form. Against the original GMR,
optimized-joint absolute delta p95/p99/max is `1.409/2.212/3.108 rad`; 55/73 clips hit the `.12 rad`
solver step cap, 58/73 contain near-limit frames, and 37/73 exceed tracked URDF velocity limits at
95 intervals. Finite-difference maxima reach `14.4 rad/s` and `1122 rad/s^2`. Therefore this sibling
remains `diagnostic_unauthorized`, with `mechanical_admission=false`; it must not be promoted as the
N73 training teacher until a soft-limit/velocity/acceleration/torque-speed re-solve and reference-
tracking rollout pass per action.

FullMDP on the 0807 A3P runtime plant does not consume that older-plant bank. Its diagnostic catalog is
`assets/motions/chingmu73_measured_a3p0807_20260808` via
`configs/action_ball_chingmu73_measured_a3p0807_f10_20260819.json`; changing motion bytes also changes the
canonical `action_uid`. All 73 files have frame-0 `pelvis_link` yaw within `1e-6 rad`; torso yaw is not a
grounding fact. This plant-consistent bank is still mechanically unadmitted (`0/73`) and therefore cannot
grant formal motion, export, deployment or robot authority.

Runtime tracking of measured site position, point velocity, signed face and the corrected long axis
is the intended wrist teacher, while all three right-wrist joints remain policy actions. “Free wrist
from body mimic” means generic wrist body position/orientation/velocity imitation is removed; it does
not mean the wrist is untrained. The direct measured rigid-paddle reward remains low-weight over the
full clip, including the strike window, so it teaches preparation, impact, follow-through and twist.
The strike-window ball-task target is the much higher-weight master on shared coordinates. Real
racket mass/CoM/inertia is a separate physics-calibration gate.

## Signed face identity is not an oriented plane

The canonical naming follows [raw-A / physical-B](../DEFINITIONS.md):

- **raw A** is the racket mount's local `+Y` normal transformed to world;
- **physical B** is the selected rubber face presented toward the opponent;
- for clip `c`, `physical_B = mount_normal_sign_per_clip[c] * raw_A` and the current
  forehand/backhand table is `[+1,-1]`.

`orient_normal(n, incoming, racket_velocity)` is allowed only inside the contact impulse law. It
chooses a convenient direction for the same geometric plane, so `n` and `-n` produce the same
impulse and cannot prove which rubber face struck the ball. Any score/reward that carries face
identity must therefore gate **before** that operation:

```text
dot(achieved_raw_A, target_raw_A) > 0
achieved_physical_B.x > 1e-6
target_physical_B.x > 1e-6
```

Normals must be finite and non-degenerate; offline formal scorers additionally require unit normals
within `2e-4`. Missing/invalid per-clip signs fail closed. The strict hemisphere test only says the
correct physical face is presented; the tighter normal-error threshold remains a separate strike
quality metric.

The analytic scorer's explicit legacy escape is diagnostic-only: it may orient an unsigned plane
when `--allow-inexact-contract` is supplied, but must emit `signed_face_exact=false` and
`evaluation_contract_exact=false`. Such a result cannot select a checkpoint, promote a motion, or
stand in for physical ball contact.

## Three points that must not be conflated

For world-from-racket rotation `R`, selected face sign `s` and ball radius
`R_ball`, exact centred contact obeys:

```text
p_ball_center = p_site + R (r_face_center/site + R_ball n_face)
v_face_center = v_site + omega_racket x R r_face_center/site
```

The tracked A3 meshes give:

| Face | `r_face_center/site` | site-to-ball-centre distance at contact |
| --- | --- | ---: |
| red (`+Y`) | `[+0.000894, 0, +0.000894] m` | `20.040 mm` |
| black (`-Y`) | `[+0.000894, -0.013208, +0.000894] m` | `33.232 mm` |

The black number is larger because the canonical site lives on the red side
of the paddle. Merely flipping the face normal for a backhand does not move
the controlled point to the black face and is not exact geometry.

### 2026-08-03 URDF-grounded collision-thickness correction

The `official_racket_site`, wrist-to-site transform, racket frame and geom centre did not move.
The MuJoCo collision mesh alone was `0.396240 mm` too thick on each side relative to the URDF
rubber outer surfaces; its local-Y scale is now `0.943396221367`. The current root MJCF SHA-256 is
`70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a`, bound by
[`a3_mujoco_identity_v2_20260803.json`](../../configs/a3_mujoco_identity_v2_20260803.json).
This changes collision/model identity but not the full-phase retarget FK values. Historical v1
identity and collision certificates remain valid only for their old bytes; a v2 L0 -> vendor-L1 ->
table/net successor chain is required instead of repinning old receipts.

## Current `site_colocated_v1` contract

Existing question banks, the online planner and BankExam historically set the
incoming ball centre equal to `p_site`. This is a virtual point-contact model,
not rigid-body contact geometry. It remains the compatibility default so old
checkpoints and banks are not silently redefined.

An `exact_face_contact_v2` experiment must change the tuple atomically:

1. store both ball-centre intercept and racket-site target;
2. bind the selected physical face and full racket orientation;
3. grade velocity at the selected face centre;
4. propagate the schema through generator, trainer, MuJoCo exam, planner,
   ONNX metadata and C++ runtime;
5. compare v1/v2 on the same immutable questions before promotion.

## Velocity point contract

Isaac Lab 2.1 deliberately mixes points in its legacy body state:

- `body_pos_w`: link/actor origin position;
- `body_lin_vel_w`: centre-of-mass linear velocity;
- `body_link_lin_vel_w`: link-origin linear velocity.

The old racket computation added `omega x r_site` to `body_lin_vel_w`. That
double-counted the link-origin-to-COM term. On the current V5/hopex strike
poses the error is about `0.401 m/s` forehand and `0.598 m/s` backhand.

The active Isaac measurement is now point-consistent:

```text
v_site = body_link_lin_vel_w[wrist] + omega_wrist x R r_site/wrist
```

MuJoCo already obtains the world linear velocity of the `right_racket` site
with `mj_objectVelocity`, so it did not have this COM/link bug.

Motion NPZ kinematics schema 2 records the intentionally mixed reference
convention (`body_pos_point=link_origin`,
`body_lin_vel_point=center_of_mass`) and binds every body-array column to its
articulation `body_names` order. Legacy MuJoCo/V5 files whose velocity is
`d(body_pos_w)/dt` must be explicitly migrated before formal training;
loaders reject the decisive link-velocity signature rather than guessing from
a filename. Schema 1 remains readable only as exact-ineligible legacy input
because it did not bind body order.

That same point contract applies when a diagnostic teacher-reference state is installed in
MuJoCo. If `O` is the pelvis link origin, `C` its rigid-body centre of mass,
`r_W = R_WB * body_ipos[pelvis]`, and the motion supplies `v_C^W` and `omega^W`, the freejoint must
receive:

```text
v_O^W = v_C^W - omega^W x r_W
omega^B = R_WB^T * omega^W
```

The translation uses `body_ipos`; `body_iquat` only orients the inertia-principal axes and must not
rotate the origin-to-COM offset. A checkpoint-bound schema-3 ONNX binds one
`motion_body_lin_vel_points` entry per clip. A clip explicitly identified as link-origin velocity
uses `v_O^W` directly and remains diagnostic; an old
inexact/missing aggregate contract is ambiguous and cannot enter teacher-reference reset. The only
pre-field compatibility is an exact schema-2 lineage, whose clips were all COM-valued. This reset
rule does not affect the formal BankExam `stand-keyframe` path, whose initial qvel is zero.

## Reference contact-speed caveat

`clean_reference_strike_velocity=true` differentiates the same site position
over `+-2` frames. At 50 Hz this is an `80 ms` average, not a measured
instantaneous ball-contact velocity. On the currently used V5hLs assets:

| Clip | `+-2` target | `+-1` diagnostic | vector difference |
| --- | ---: | ---: | ---: |
| forehand frame 39 | `2.488 m/s` | `2.315 m/s` | `0.248 m/s` |
| backhand frame 23 | `3.404 m/s` | `3.533 m/s` | `0.267 m/s` |

Both contact annotations are still marked `unverified`. Formal V5 transfer
must therefore treat contact frame and velocity window as an ablation, and
must not call the `+-2` value physical ground truth.

## Reproducible checks

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_racket_geometry_contract.py \
  hope_training/whole_body_tracking/tests/test_motion_kinematics_contract.py

PYTHONPATH=hope_training/whole_body_tracking/scripts \
python3 hope_training/whole_body_tracking/scripts/analyze_strike_phase.py \
  --motion-file /path/to/motion.npz --clip-name forehand
```
