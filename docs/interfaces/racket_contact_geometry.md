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
