# Policy Observation And Action

Status: Implemented for 110/175/177/180 and the historical/fresh ActionBall layouts described
below. The no-ball motion-prior pretraining contract `stage1_natural_clip_site_v1` has passed its
Pod ObservationManager/normalizer Gate, but it is not the first segment of the pending canonical
same-run phased ABI. The 179 training/evaluation contract, versioned
flat-wire/C++ source path and opt-in same-ball task-revision source are implemented, but the new
schema-4 path has not yet passed an Isaac full-scene run, ROS/Jazzy Release, vendor Gate 3 or
hardware behavior. It therefore remains `Partial`, not the currently accepted deployment path.
The 181 deploy wire remains intentionally blocked pending the station/order contract day.
Dynamic `task_first_n<N>` remains a historical training-only candidate. Dynamic
`action_ball_n<N>` is implemented in the Isaac trainer and has Pod construction/training evidence,
but it still has no production arbitrary-N flat wire or C++ observation consumer. The feature-branch
historical `action_ball_table_pose_twist_n<N>` added table-relative base 6-DoF context and
three-axis root-COM linear velocity, but left racket velocity/normal in world coordinates while
position was heading-relative. Historical compatibility runs used the versioned,
frame-consistent `action_ball_table_pose_twist_heading_task_n<N>` successor. Current vendor fresh
N1 uses fixed-194 `action_ball_table_pose_twist_heading_task_teacher_start_v2`: the former constant
one-hot slot is replaced by the exact Motion phase-governor countdown. Pod smoke/probe have
materialized the 17-term layout and finite fresh checkpoints; what remains open is the corrected
safety-gated long result and production deploy-consumer parity.
The current portable FullMDP successor is the semantic Observation V2 contract:
actor=`203`, critic=`219`. It restores the minimum floating-base state needed to distinguish
balance-relevant states, removes raw task/control ledgers, and uses one shared ordered/scaled ABI in
Isaac and MuJoCo Full-A. It is a fresh, warm-start-breaking lineage. The old raw `229/399` V1 remains
only for intentional legacy WAIT compatibility; neither V1 nor the historical `211/319` fixed-question
lineages is a fallback for a Full-A V2 launch.

## HITTER-Compatible Contract

The baseline policy keeps the HITTER-style separation:

- Planner provides target racket position, target racket velocity, target racket normal, and time to strike.
- WBC policy combines planner target, robot state, and previous action.
- Policy outputs desired joint positions.

## Historical V1 no-ball motion-prior contract: 170-D actor / 296-D critic

The code-owned 2026-08-02 identifier remains `stage1_natural_clip_site_v1`, but the project-level
meaning has been corrected: this is an isolated no-ball motion-prior pretraining/diagnostic ABI,
not Stage 1 of the canonical same-run three-stage training. Its actor contract is:

| Slice | Term | Dim | Meaning |
| --- | --- | ---: | --- |
| `[0:62]` | `command` | 62 | current natural clip phase's 31-D reference joint position plus 31-D reference joint velocity |
| `[62:65]` | `motion_anchor_pos_b` | 3 | reference torso position relative to current torso |
| `[65:71]` | `motion_anchor_ori_b` | 6 | reference torso orientation residual in continuous 6-D form |
| `[71:74]` | `base_ang_vel` | 3 | base/pelvis angular velocity; three axes, not a six-value angular velocity |
| `[74:105]` | `joint_pos` | 31 | encoder position relative to default |
| `[105:136]` | `joint_vel` | 31 | encoder joint velocity |
| `[136:167]` | `actions` | 31 | previous normalized actor output, before the episode-fixed actuator delay |
| `[167:170]` | `projected_gravity` | 3 | gravity direction in the base frame |

The corresponding critic uses the ordinary 14-body privileged motion-tracking layout and totals
296 values: the same 62-D command, anchor pose, 14 body positions and 6-D orientations, base
linear/angular velocity, joint position/velocity and previous action.

This motion-prior actor intentionally contains no `action_one_hot`, action UID, ball state, demanded
face/reserved scalar, actor-visible `time_to_strike`, teacher-start clock or current/target racket
block. This does **not** remove the strike time from the task. Each code-owned lane still binds its
motion SHA to an exact `strike_frame/strike_phase`; the runtime derives signed `time_to_strike`
from the current clip phase and uses it for the tight/wide Reward windows, exact-strike metrics and
adaptive reward sigma. Each policy is fixed to one original-speed clip, while `command` includes
both reference position and velocity at the current phase. Therefore remaining time to strike is a
deterministic function of the actor-visible reference stream and the fixed lane, rather than hidden
state; another scalar would duplicate the same clock. The clip-derived official-site
position/normal/velocity are training targets in Reward/critic-side metrics, not extra actor truth.

That is a statement about Markov sufficiency, **not** an endorsement of the representation. It asks
the MLP to reconstruct both teacher and achieved official-site position/velocity/normal from
`q_ref/dq_ref` and `q/dq`, including robot FK, the racket mount offset, the Jacobian and the signed
face convention. The update10k result shows this was needless sample-complexity on top of an already
sparse strike-window reward. The canonical successor therefore makes both site-state tuples
actor-visible instead of relying on implicit reconstruction.

This argument stops applying as soon as ball arrival/contact time can vary independently of clip
phase. The canonical three training stages are one PPO run and therefore must use one fixed,
versioned ball-conditioned actor/critic ABI from the first stage onward. That ABI must expose
explicit `time_to_contact` from its first rollout, must preserve every column's physical meaning
through all stage transitions, and must continue with the same optimizer/normalizer/checkpoint
lineage. Stage is only a chronological label within that run, not an operator-controlled switch:
all reward callables and weights are installed
once at run start, and event/eligibility masks make contact, hit and outcome terms fire only when
their physical denominators exist. It may use teacher-consistent values and explicit validity masks
in the first stage, but must not grow 170-D into a different network later. The current 170-D checkpoints may be evaluated
as pretraining donors only; transfer into a wider network is a new lineage, not continuous staged
training.

The canonical phased actor excludes both `action_one_hot` and `swing_type`. `swing_type` is a
constant in an N=1 policy and a degenerate action one-hot in an N>1 policy. No currently proven
partial-observability condition requires it: continuous reference motion plus ball/task state is
the intended source of behavior. A future style latent may be reconsidered only after a measured
ambiguity test, and it must be content-derived rather than a slot/UID label.

`base_lin_vel` is not actor-visible in this contract.  It is replaced by `projected_gravity`,
matching the current A3 deploy-facing signal choice while avoiding an unverified causal base-speed
estimator.  The three-value `motion_anchor_pos_b` remains because full-body imitation must expose
root translation error; its production authority is the table/base pose provider. Observation
history remains one frame for this first motion-prior launch. An eight-frame history changes the ABI,
reset buffer and exact-resume state and therefore belongs to the subsequent MuJoCo contract batch,
not an unversioned edit to this layout.

## Current ball-conditioned fresh N1 compatibility contract: fixed 194-D, no action one-hot

The retained ball-conditioned fresh-N1 source contract is
`action_ball_table_pose_twist_heading_task_teacher_start_v2`.  It is not safe to call its first
177 columns “177-D proprioception”: that prefix mixes **68 reference/teacher values**, **99
robot/runtime values** and **10 task/clock values**.  The remaining 17 values add table-relative
base pose/twist, demanded face/rho and the teacher-start clock.  The exact slices are:

| Slice | Term | Dim | Kind | Exact actor meaning / frame |
| --- | --- | ---: | --- | --- |
| `[0:62]` | `command` | 62 | reference/teacher | selected motion's 31 joint positions + 31 joint velocities in actor joint order; held at ready frame 0 during the pre-swing wait |
| `[62:68]` | `motion_anchor_ori_b` | 6 | reference + robot | reference-torso versus current-torso orientation residual, continuous 6-D representation |
| `[68:71]` | `base_ang_vel` | 3 | robot | pelvis angular-velocity vector `(wx, wy, wz)` in the pelvis/body frame; simulator truth plus configured corruption in training, bias-corrected pelvis IMU gyro at deploy |
| `[71:102]` | `joint_pos` | 31 | robot | measured `q - default_q` in actor joint order |
| `[102:133]` | `joint_vel` | 31 | robot | measured joint velocity in actor joint order |
| `[133:164]` | `actions` | 31 | runtime | previous normalized actor output; with vendor actuator-delay DR this is still the actor output, not the delayed drive row |
| `[164:167]` | `projected_gravity` | 3 | robot | gravity direction expressed in the base frame |
| `[167:169]` | `base_target_pos_b` | 2 | task | demanded base XY minus current base XY, rotated into the base yaw-heading frame |
| `[169:172]` | `racket_target_pos_b` | 3 | task + robot FK | demanded racket position minus **current racket FK position**, in the same yaw-heading frame |
| `[172:175]` | `racket_target_vel_heading` | 3 | task | demanded racket linear velocity, yaw-heading frame |
| `[175:176]` | `time_to_strike` | 1 | task clock | signed `time_to_contact-task_age`: the receipt-owned task deadline, including ready wait plus scaled clip time to `reference_t_hit` |
| `[176:177]` | `swing_type` | 1 | historical task/reference | forehand `+1` or backhand `-1`; this is still a live term in current fixed-194, but is rejected from the pending canonical same-run phased actor |
| `[177:180]` | `base_position_table` | 3 | robot/table | current root XYZ relative to the table-surface centre |
| `[180:186]` | `base_orientation_table_6d` | 6 | robot/table | current full table-to-base orientation as `[R00,R01,R10,R11,R20,R21]`; this is an orientation encoding, not angular velocity |
| `[186:189]` | `base_lin_vel_heading` | 3 | robot | current root-COM linear velocity `(vx, vy, vz)`, yaw-heading frame |
| `[189:193]` | `racket_target_normal_cmd_heading` | 4 | task | demanded raw-A face normal `(nx,ny,nz)` in yaw-heading frame + legacy zero placeholder formerly named `rho`; the fourth value is always zero and has no current physical semantics |
| `[193:194]` | `time_to_teacher_start_s` | 1 | task/teacher clock | seconds until the selected teacher leaves ready frame 0; zero throughout swing/recovery |

Dimension check: `68 + 99 + 10 = 177`; then `table pose/twist 12 + face/rho 4 +
teacher-start 1`, giving `177 + 17 = 194`.  An angular-velocity vector has three components.
The complete spatial base twist has six components here only after combining the separate
`base_ang_vel(3)` and `base_lin_vel_heading(3)` terms; they intentionally use different frames and
deployment authorities, so they are not concatenated under one misleading “6-D angular velocity” name.

`action_one_hot` still exists in the source registry and observation helper for exact parsing of
historical contracts and for the separate historical `task_first_n<N>` path.  **The fresh N1
trainer does not attach it.**  It admits only the fixed-v2 contract, appends
`time_to_teacher_start_s`, and its schema-3 hard contract rejects any `action_one_hot` term.  The
2026-07-31 Pod identity smoke independently materialized the 17 names/dimensions above with total
194 and file SHA-256 `38974f1bc5da8140aec24e07dd2d59d9b7cc90ed52acdd20f54564dd70368fba`.
The remaining action UID/local slot is control-plane state used by the sampler, solver, curriculum
and receipts, not a neural-network input.

This fixed-194 actor with the existing 318-D privileged critic is the **implemented contract for
the legacy fixed-question target-mask diagnostic**, not the final N73 ABI. The historical recipes keep
the network width and term order unchanged: `current_lm` makes target
`position/velocity/face = 111` valid, `analytic_no_velocity` uses `101`, and
`outcome_dense_only` uses `000`. In this compatibility contract validity is receipt/Reward state,
not an actor column: an invalid target component is zeroed at the final observation boundary. That
is sufficient to compare the three diagnostic recipes, but it does not close the final superset ABI,
which must expose component validity explicitly so that a physical zero cannot be confused with a
missing target. The live `swing_type` value is `-1` for this one-backhand N1, hence constant and
information-free. It remains only for compatibility and must be removed, rather than expanded into
an action ID or renamed motion-intent field, in canonical N73.

Three clocks must not be conflated. `reference_t_hit` is a fixed landmark inside the selected clip
and is not an independent actor column. `time_to_strike` is the variable solved ball-task contact
deadline remaining and stays signed after contact. `time_to_teacher_start_s` is only
`max(pre_swing_wait-task_age,0)`, so it becomes zero as soon as reference playback leaves ready.
The fixed-194 correction therefore removed the final constant `action_one_hot(1)` and used that
same slot for wait time; it did **not** remove the live historical `swing_type(1)` in the middle.

The demanded racket tuple is also not duplicated incoherently. Position residual, demanded
velocity and demanded raw-A normal all use the same actor-visible task and yaw-heading rotation;
`time_to_strike` is that task's strike clock, while `time_to_teacher_start_s` answers the distinct
question of when reference playback begins.  The actor has no explicit full “current racket
state” block: current racket position enters the position residual through joint-encoder FK, and
joint `q/dq` plus base state carry the robot state. Current code places explicit
`racket_pos_b/racket_lin_vel_w/racket_normal_w` only in critic observations, but that is an
implementation/design debt, not a privilege requirement: the same achieved official-site state is
causally constructible on hardware from joint encoders, the frozen robot/racket geometry, IMU and
the base-state estimator. Requiring the actor to learn that FK/Jacobian/sign mapping again adds
sample complexity without adding information.

The fourth “face rho” column must not be confused with three unrelated quantities elsewhere in the
project: curriculum joint-domain scale `rho`, AR(1) noise correlation `rho`, or torque-envelope
utilization `rho`. The face-command producer always writes zero, and no consumer assigns it a unit,
frame or physical meaning. It is therefore a dead compatibility placeholder, not spin, confidence,
restitution or face state.

## Ball-free explicit-paddle canary: 225-D ordered actor

The implemented 225/318-D canary is a **ball-free explicit-paddle diagnostic**. It validates the
ordered paddle fields, frames, signed-face convention and measured-teacher geometry, but it has no
incoming-ball receipt and does not implement a ball-conditioned contact target. Its
`racket_contact_desired_at_t_hit_heading` row is currently a teacher-copy placeholder. It must not
be reported as evidence that a policy learned from incoming ball or adapted a professional motion
to a different landing task.

This historical `H225` contract retains four distinct blocks at the same
`official_racket_site` and with the same signed-face convention. Teacher-now state and future
contact demand remain separate: they coincide only in this ball-free placeholder or when a later
ball question deliberately asks for the teacher's nominal impact. This is not a canonical A/B/C
shared superset. The fixed-midpoint successor below deliberately gives A and C different contract
names and different `[212:221]` meanings.

For every block below, “heading” is fully defined rather than naming only a rotation:

```text
position_heading = R_heading^T (position_world - current_base_position_world)
linear_velocity_heading = R_heading^T linear_velocity_world
signed_normal_heading = R_heading^T signed_normal_world
```

All four use the current tick's base origin and yaw-heading rotation.

| Block | Dim | Meaning / source |
| --- | ---: | --- |
| `racket_site_achieved_now_heading` | 9 | current actual `position(3) + linear_velocity(3) + signed_face_normal(3)`, computed causally by the shared FK/Jacobian producer |
| `racket_site_teacher_now_heading` | 9 | current 73 reference phase's aligned `position(3) + linear_velocity(3) + signed_face_normal(3)`; remains observable and receives a low-weight full-phase paddle reward, while the strike-window task target has much larger weights |
| `racket_site_teacher_at_reference_hit_heading` | 9 | the selected reference's nominal official-site contact state at immutable `reference_t_hit`; it is the teacher baseline used to expose the teacher-to-task correction, not an action ID or a separate motion-intent code |
| `racket_contact_desired_at_t_hit_heading` | 9 | historical reserved shape only: `H225` writes the teacher-contact copy; `A225-proto` replaces this slice with a real contact demand, while `C225-proto` replaces it with incoming-ball state and therefore cannot share this contract or checkpoint |

Source authority is part of this layout, not an implementation detail. In the final proposal,
`racket_site_achieved_now_heading` comes from simulator/live robot FK; both `teacher` blocks come
from the same-clock measured racket channel after the frozen marker-to-`official_racket_site`
transform; in the final A proposal only, `racket_contact_desired_at_t_hit_heading` comes from the
current ball/task planner. C instead uses its separately named incoming-ball slice.
The implemented ball-free 225/318-D canary instead derives its teacher blocks from retargeted joint
FK/Jacobian and copies teacher contact into the nominal desired-contact slot. The
2026-08-02 schema-v3 measured bank was later found to use site-local `+X` instead of the URDF-visual
45-degree butt-to-blade axis; its claimed long-axis/SO(3) admission is revoked. The local schema-v4
sibling now binds the corrected axis and rigid visual mesh SHA, refuses FK fallback or stale face
signs, and closes the kinematic denominator at 73/73. It is not mechanically admitted: the complete
auditor admits 0/73; 57/73 have an observed position/velocity hard failure and the remaining 16/73
still lack acceleration, torque-speed and inverse-dynamics authority. Its three ball-free lanes remain
diagnostic-only. Strict referenced-asset loading also rejects the legacy schema-v1 prototype because
`velocity_contract` is missing; the source capsule/compiler must preserve all measured channels and
pass the residual and mechanical Gates in
[Racket Control Point And Contact Geometry](racket_contact_geometry.md).

Adding only teacher-at-hit would be sufficient for the isolated fixed-clip motion prior, but not for
an A contact-guidance run: teacher-at-hit says what the nominal motion can do, while
desired-at-contact says what the current ball requires. An A arm may make them equal on the first
narrow teacher-consistent distribution. C has no desired-contact tuple: it receives incoming-ball
state under a different ABI and learns through dense achieved-outcome feedback. B is deferred until
it has its own executable ABI or an explicit, receipt-bound partial-field validity mechanism.

The three continuous error families are therefore different physical questions. For any two
official-site tuples `X=(p_X,v_X,n_X)` and `Y=(p_Y,v_Y,n_Y)`, define the componentwise error

```text
d(X,Y) = ( ||p_X-p_Y||_2,
           ||v_X-v_Y||_2,
           acos(clamp(n_X dot n_Y, -1, 1)) )

e_motion(t) = d(achieved_now(t), teacher_now(t))
e_adapt     = d(desired_at_contact, teacher_contact_nominal)
e_task      = d(achieved_at_actual_contact, desired_at_contact)
```

`e_motion` is dense full-phase professional-motion tracking; `e_adapt` tells the policy how far the
current ball task asks it to depart from that professional stroke; `e_task` scores whether the
actual contact realized the requested departure. Invalid A/B components must be omitted by their
versioned eligibility rule, not treated as zero error. C does not compute `e_adapt` or `e_task`
because its ABI has no desired-contact tuple. The actor receives paired raw tuples in A so an MLP
can form these differences without redundant residual columns; Reward and metrics compute the same
errors explicitly. In the ball-free 225-D implementation `e_adapt` is identically zero by
construction and `e_task` is not a ball-conditioned training denominator, which is exactly why that
canary cannot answer the A/B/C learning question.

No additional fixed-width motion-intent descriptor is part of the canonical design. The teacher
trajectory already changes with the selected motion: current `q_ref/dq_ref`, body reference and
`racket_site_teacher_now_heading` are the actual professional control target at that phase, not
features asking the policy to classify an action. The nominal-contact block says what that same
stroke would naturally do at impact, and its difference from `desired_at_contact` is the requested
task correction. Different professional strokes may therefore retain different natural impact
effects without any extra intent/ID. In the measured
73 bank, same-relative-phase `q_ref+dq_ref` has no exact cross-action collision; inventing another
18-D pre-hit/hit identifier would duplicate teacher content rather than close a demonstrated
observability gap. If a future audit finds two references with indistinguishable current teacher
state but materially different required futures, the remedy is a short causal future-teacher
preview, not UID, slot, one-hot, PCA pseudo-ID or a collision-avoidance code.

For the final ball-conditioned contract, ordered terms are grouped by purpose rather than by which
backend happened to expose them first:

```text
actor  = robot/achieved -> teacher/reference -> incoming-ball/task target -> clocks/validity -> causal history
critic = privileged robot/teacher -> the same exogenous ball/task target -> achieved outcome/eligibility
```

Here **input width** means the total number of scalar columns presented to that network. It is not
the policy MLP hidden-layer shape (currently `512,256,128`). Status must stay explicit:

| Contract | Implemented meaning | What it does not prove |
| --- | --- | --- |
| `L194`: actor `194` / critic `318` | legacy fixed-question target-mask diagnostic; its `000` proxy has no incoming-ball actor state | not a true A225/C225 comparison or final N73 ABI |
| `H225`: actor `225` / critic `318` | historical ball-free explicit-paddle canary; desired contact is a teacher copy | not a ball task and not an A/B/C result |
| `A225-proto`: actor `225` / critic `318` | fixed-midpoint contact-oracle diagnostic contract | dedicated producer/config/normalizer/Gym/launcher and runtime Reward materialization exist; exact Pod oracle32 is blocked by deterministic ready-to-teacher table intrusion, so PPO remains `未测` |
| `C225-proto`: actor `225` / critic `318` | fixed-midpoint incoming-ball-direct diagnostic contract | dedicated critic/normalizer/Gym/task/trainer admission now exist; no-clobber launcher and exact Pod learning remain blocked, so no learning/rate result |
| `FINAL-N1/N73` | **proposal** ordered by the purpose groups above | actor/critic widths are not yet frozen |

The repeated critic width `318` does not make the two canaries semantically or checkpoint
compatible. Final width remains intentionally unfrozen until the A/C contact-target decision and
the two-step delay-history decision close. Reordering terms, replacing a source or changing a
validity rule changes the contract even when total width stays the same.

The final field groups and their current implementation gap are:

| Group | Human meaning | Implemented canaries versus final proposal |
| --- | --- | --- |
| `incoming ball` | predicted contact-time ball `position3/velocity3/spin3`, contact time, validity and estimate age | absent from `L194/H225`; fixed-tape `C225-proto` now has an independently admitted 225/318-D trainable consumer, but no launcher/Pod learning result. Final N73 adds only causal predictions; critic may use truth without leaking it to actor |
| `achieved paddle` | what the robot currently achieved at the official site: `position3/point-velocity3/signed-face3` | explicit in actor-225; only implicit through robot state/FK in actor-194; final proposal makes it explicit |
| `teacher_now` / `teacher_contact_nominal` | what the selected professional reference is doing now and what it naturally does at impact | joint teacher stream exists in both; explicit paddle blocks exist only in actor-225 today; final proposal uses the measured, same-clock racket source |
| `desired_at_contact` | A planner's requested contact `position/velocity/face`; its difference from teacher contact is task adaptation | `L194` has a compatibility target tuple, `H225` only a teacher copy, and fixed-tape `A225-proto` is wired for the current diagnostic. B has no executable successor ABI; C intentionally has no such tuple |
| `landing/spin` | desired outgoing result; actual landing/spin is outcome truth | current ActionBall question owns `aim_xy` outside these actor layouts and first N1 has no authoritative desired-spin observation; final proposal adds desired result fields while actual result stays critic/Reward/evaluator-only |
| clocks/validity/history | when ball contact and teacher start occur, whether/when estimates are usable, and causal delayed state | `L194` implements two clocks but not explicit target validity/age; `H225` contact clock is a reference landmark. A/C clock producers are not yet authorized; final proposal keeps both clocks distinct and versions any history |

"One SHA across Isaac/MuJoCo/export" is a **canonical admission proposal** for one traceable lineage,
not one literal digest for unlike bytes: a portable-semantics SHA binds the ordered
observation/action/reward/task/curriculum meaning;
each backend adds its own binding SHA for assets, indices, `dt` and contact settings; actor and critic
normalizers each have a SHA over ordered `(mean, variance, count)` state; the checkpoint SHA covers
model/optimizer/normalizers/curriculum/delay queues/RNG; and the export SHA covers exported bytes plus
its source checkpoint, portable-semantics SHA and normalizer-baking declaration. Any missing or
mismatched parent fails closed. Existing canaries implement subsets of this provenance chain; they
must not be described as having final Isaac/MuJoCo/export parity until every parent and backend
binding is emitted and checked.

In plain terms, the normalizer is the learned per-column scale/offset applied before the network; it
is part of the policy because applying column 17's statistics to a reordered column 17 changes the
function even when tensor width matches. A checkpoint is the complete resumable learning state, not
weights alone. An Isaac or MuJoCo backend binding proves where every portable body/site/contact term
was read and which plant/assets produced it. An export is the deployable graph plus an explicit
choice that normalization is either baked into that graph or supplied by an exact sidecar. The
parent SHAs make it impossible to silently pair valid-looking weights with a different observation
order, teacher source, simulator mapping or normalization state.

The **historical implemented ball-free 225-D canary order**, not the current fixed-194 diagnostic or final N73 order, is frozen below. `world` means the canonical HOPE venue frame: origin at the
near-side left table-surface corner, `+X` toward the opponent, `+Y` left from Player One and table
surface `z=0`.  Actual and teacher joint positions both mean `q - default_q`; teacher values must not
silently use absolute joint angle while actual values are relative.

| Slice | Term | Dim | Exact meaning |
| --- | --- | ---: | --- |
| `[0:15]` | `actual_base_now_world` | 15 | actual root `position3 + orientation6 + linear_velocity3 + angular_velocity3` in HOPE world |
| `[15:30]` | `teacher_base_now_world` | 15 | current-phase teacher root in the identical world representation |
| `[30:61]` | `joint_pos` | 31 | actual `q-default_q` |
| `[61:92]` | `teacher_joint_pos` | 31 | current teacher `q_ref-default_q` |
| `[92:123]` | `joint_vel` | 31 | actual `dq` |
| `[123:154]` | `teacher_joint_vel` | 31 | current teacher `dq_ref` |
| `[154:185]` | `actions` | 31 | previous normalized actor output |
| `[185:194]` | `racket_site_achieved_now_heading` | 9 | actual site state now |
| `[194:203]` | `racket_site_teacher_now_heading` | 9 | aligned teacher site state now |
| `[203:212]` | `racket_site_teacher_at_reference_hit_heading` | 9 | aligned nominal reference-hit site state |
| `[212:221]` | `racket_contact_desired_at_t_hit_heading` | 9 | ball-free teacher-contact copy placeholder; reserved shape only, not a current ball-task desired state |
| `[221:223]` | `desired_base_xy_world` | 2 | desired base station in HOPE world XY |
| `[223:224]` | `time_to_contact` | 1 | ball-free signed seconds until the reference strike landmark; not a receipt-owned incoming-ball deadline |
| `[224:225]` | `time_to_teacher_start` | 1 | Stage1 MotionCommand-owned seconds until teacher playback next leaves ready hold; zero after start |

### Historical fixed-midpoint ActionBall A225/C225 contracts

**2026-08-06: the A225/C225 family is retired as a runnable lane** — no gym task id, no EnvCfg, and two
independent fail-closed refusals (`train.py` on `actor_obs_contract`, `MotionOnPolicyRunner.__init__` on
`obs_mode`). Its launchers, materializer and trainability validators were deleted; see
[successor ledger §8.1](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md).
The two contract objects below are deliberately **kept as an ABI ledger**: they record why a 225-wide
normalizer or checkpoint must never be loaded into a 211-wide network. They describe history, not a lane
you can launch.

Two 225-D contracts remain registered as superseded history. They preserve rows `[0:212]` (robot state,
teacher/mimic state and achieved/teacher paddle state) and `[221:225]` (desired base station plus the
two independent clocks). They intentionally give `[212:221]` different physical meanings and
therefore use different contract names, normalizers and checkpoint lineages even though the width is
the same:

| Contract | Slice | Ordered terms | Purpose / frame |
| --- | --- | --- | --- |
| `action_ball_a225` | `[212:221]` | task desired contact position3, velocity3, signed-face3 | contact-oracle A; position uses current-base origin and heading, vectors use current-base heading |
| `action_ball_c225` | `[212:221]` | incoming ball-at-contact position3, velocity3, spin3 | direct ball-state C; the fixed table midpoint is an environment constant and is not repeated as a task input; position uses current-base origin and heading, vectors use current-base heading |

The 212-D common prefix groups as `robot/achieved=117` and `teacher/mimic=95`:

- robot/achieved: actual base `position3 + orientation6D + linear velocity3 + angular velocity3`,
  actual joint position31, joint velocity31, previous action31 and achieved paddle
  `position3 + point-velocity3 + signed-face3`;
- teacher/mimic: in the current A225/C225 prototype, teacher base uses the same absolute 15-D world
  representation, followed by teacher joint position31/velocity31, teacher paddle-now9 and teacher
  paddle-at-reference-hit9. This raw teacher-base representation is not the final canonical choice.

### Current portable FullMDP semantic Observation V2: actor 203 / critic 219

The current FullMDP identity is
`action_ball_full_mdp_semantic_observation_v2`, paired with actor contract
`action_ball_full_mdp_semantic_actor_v2` and critic contract
`action_ball_full_mdp_semantic_critic_v2`. This section is the **only complete ordered-layout and
scale truth source**; operation and Gate documents may summarize it but must not copy a second table.
The critic is asymmetric: its first 203 columns are the exact already-scaled actor row, followed by
16 privileged values. Actor and critic use static nondimensionalization at the observation boundary;
`empirical_normalization=false`, and the V2 packer performs no observation clipping.

The multiplier column below means `network_value = raw_physical_value * multiplier`.

| Slice | Ordered term | Dim | Exact meaning / frame | Static multiplier |
| --- | --- | ---: | --- | ---: |
| `[0:3]` | `projected_gravity_b` | 3 | unit gravity direction in pelvis/body coordinates | `1` |
| `[3:6]` | `base_ang_vel_b` | 3 | pelvis/body-frame angular velocity | `0.25` |
| `[6:9]` | `base_position_table` | 3 | root XYZ minus fixed table-surface centre, in table/world axes | `1` |
| `[9:11]` | `base_heading_table_xy` | 2 | normalized `[cos(yaw), sin(yaw)]` in fixed table axes | `1` |
| `[11:14]` | `base_com_lin_vel_heading` | 3 | root inertial-COM linear velocity, inverse-yaw heading frame with world Z preserved | `0.5` |
| `[14:45]` | `joint_pos_rel` | 31 | measured joint position minus default, actor joint order | `1` |
| `[45:76]` | `joint_vel` | 31 | measured joint velocity, actor joint order | `0.05` |
| `[76:107]` | `last_action` | 31 | previous normalized actor output | `1` |
| `[107:138]` | `teacher_joint_pos_rel` | 31 | current teacher joint position minus default | `1` |
| `[138:169]` | `teacher_joint_vel` | 31 | current teacher joint velocity | `0.05` |
| `[169:172]` | `motion_anchor_pos_b` | 3 | teacher anchor position relative to the current robot anchor, current anchor frame | `10/3` |
| `[172:178]` | `motion_anchor_ori_b6` | 6 | teacher-versus-current anchor orientation residual, continuous 6-D form | `1` |
| `[178:183]` | `motion_phase_one_hot` | 5 | existing Motion phase (`pre-reveal hidden/active opportunity/post-deadline suffix/recovery hidden/ready hold`) | `1` |
| `[183:186]` | `racket_target_pos_error_heading` | 3 | actor-visible target position minus achieved racket-site position, heading frame | `5` |
| `[186:189]` | `racket_target_vel_error_heading` | 3 | actor-visible target point velocity minus achieved racket-site velocity, heading frame | `1` |
| `[189:192]` | `racket_target_normal_error_heading` | 3 | actor-visible raw A/+Y target normal minus achieved raw mount +Y normal, heading frame；per-action mount sign只选择胶面/contact identity，不翻转该残差 | `2` |
| `[192:194]` | `base_goal_error_heading_xy` | 2 | desired base XY minus current root XY, heading frame | `5` |
| `[194:195]` | `time_to_contact_s` | 1 | signed task contact time remaining; zero when task-hidden | `1/2.42` |
| `[195:196]` | `time_to_teacher_start_s` | 1 | teacher playback start time remaining; zero when task-hidden | `1` |
| `[196:197]` | `time_to_next_opportunity_s` | 1 | time until cadence can next reveal a task; visible also outside a task | `1/5.86` |
| `[197:202]` | `epoch_learning_phase_one_hot` | 5 | compact public phase (`idle/reveal/launch/outcome/retired`) | `1` |
| `[202:203]` | `task_valid` | 1 | Motion-owned actor visibility bit, not Epoch payload retention | `1` |
| `[203:204]` | `episode_time_remaining_s` | 1 | privileged finite-episode time remaining | `1/30` |
| `[204:207]` | `live_ball_center_rel_root_heading` | 3 | current live flight ball centre minus root, heading frame | `1` |
| `[207:210]` | `live_ball_lin_vel_heading` | 3 | current live flight ball linear velocity, heading frame | `0.1` |
| `[210:213]` | `live_ball_ang_vel_heading` | 3 | current live flight ball angular velocity, heading frame | `1/60` |
| `[213:214]` | `selected_rubber_contact_latched` | 1 | privileged current-flight selected-rubber contact latch | `1` |
| `[214:215]` | `net_crossed_latched` | 1 | privileged current-flight net-crossed latch | `1` |
| `[215:216]` | `net_clear_latched` | 1 | privileged current-flight net-clear latch | `1` |
| `[216:218]` | `foot_supported_lr` | 2 | privileged left/right support-contact bits from the existing R07 plant sample | `1` |
| `[218:219]` | `cadence_ready_dwell_fraction` | 1 | privileged ready streak divided by required dwell, clamped to `[0,1]` | `1` |

This is not `229 + 8 = 237` or `399 + 8 = 407`. V1 exposed a 45-value raw task payload plus
Epoch/control bookkeeping to the actor and broad owner-fact/Reward due-paid ledgers to the critic.
V2 adds the nonredundant floating-root block `base position3 + unit heading2 + COM velocity3`, but
deletes those raw ledgers and replaces them with the semantic residuals, clocks, phases and compact
privileged suffix above. V1 `229/399` is retained only by the deliberately legacy WAIT runner/model;
Full-A must bind V2 `203/219` and cannot silently fall back by matching an old checkpoint width.

The three position-like terms are not duplicates. `base_position_table` answers where the robot is
relative to the table; `base_goal_error_heading_xy` answers where the current task wants the base to
move from its present location; `motion_anchor_pos_b` answers how the current robot anchor differs
from the time-varying teacher anchor. Holding either the goal residual or teacher residual fixed while
moving the whole robot relative to the table changes clearance, reach and balance. Likewise,
instantaneous 31-D joint `q/dq` cannot recover floating-root translational COM velocity: joint
velocities omit the free root. These eight restored root values reduce a measured
balance-relevant observation alias; they do **not** prove that the complete process is Markov.

Task visibility has one owner. Motion exposes the task only in its task-active phases and supplies
the actor `task_valid` mask; Epoch intentionally retains the last payload through `RETIRED` until a
later accept, so Epoch payload validity must not be reused as actor visibility. Task residuals and
the two task clocks are zero under that Motion mask. On Isaac, the three racket residuals use
Racket's public actor-visible target accessors, including configured observation delay, never live
truth or a private target tensor. A backend must supply the corresponding causal target view rather
than granting the policy an undelayed simulator oracle.

The internal Isaac Motion publication feeding this ABI is one typed ten-tensor snapshot: row-wise
`control_tick`, five-state `phase`, `reset_generation`, `swing_generation`, `action_uid`,
`task_identity`, `task_valid`, and the three float64 countdowns used above. Observation and Physical
validate the same opaque publication token, then receive isolated clones of only these ten fields;
they do not receive the old 34-field identity/timing ledger. The isolation is intentional because a
frozen dataclass does not make Tensor bytes immutable. Float64 countdowns are converted to float32
only at this policy boundary. This internal narrowing does not change the 203/219 model ABI.

R07 support and cadence dwell are one narrow direct observation state from the same real
post-physics plant sample; Observation must not call the broad plant adapter a second time or consume
the Motion-only admission capability under a forged consumer name. Before the first real
post-physics edge, genesis returns explicit zero support/dwell. After a row-wise selected reset,
only the exact `reset_generation + 1` row returns zero for that observation; same-generation peers
retain their published bytes and still require exact control-tick alignment. The next real
post-physics publication restores the reset row. Generation rollback, jumps and integer wrap fail.

The critic's current-flight latches and ball row are selected by R06 against the complete current
`ActionEpochShotKey` **and** publication ordinal. R06 performs that identity match internally and
returns only a physical storage slot plus the three latches; the slot is used immediately to gather
the Physical `[N,K,13]` state and is never concatenated into actor or critic observations. Thus a
slot/UID remains an implementation locator, not an action identity or learnable feature.

Actor fields are intended to be deploy-observable, but the real producer is not yet closed. Joint
state comes from encoders; projected gravity and gyro come from the pelvis IMU; table-relative root
position/heading and causal marker-to-COM velocity require calibrated OptiTrack table/root rigid
bodies and marker extrinsics; achieved racket state uses encoder FK plus that base estimate; teacher,
planner targets and clocks are controller-owned causal state. Ball truth, selected-contact/net
latches, support and ready dwell remain critic-only. This source list is a design constraint, not a
deployment claim; calibration, synchronization, stale/dropout behavior, real builder parity and
hardware evidence remain open in G07.

### Historical fixed-question A211/C211 contracts (not current FullMDP)

This historical fixed N1 successor deletes the actor-only raw `teacher_base_now_world(15)` block. It does
not replace it with a residual. Its actor-side actual-base information remains 15 scalars but is a
new source-homogeneous v2 layout: localizer world `position3 + orientation6D + linear_velocity3`
followed by the pelvis/body-frame IMU gyro3. It contains neither `projected_gravity` nor a second,
world-frame angular-velocity copy. The actor also keeps teacher q/dq, achieved paddle9,
teacher-now paddle9 and teacher-at-hit paddle9. Teacher-to-task adaptation is expressed by the
teacher nominal contact versus A's desired contact or C's incoming ball state. Physical reset uses
the tracked split-ready artifact with literal-zero joint velocity. The same-gate direct measured-
frame0 physical-birth screen passed `0/73`, so measured frame 0 is retained only as teacher authority
and must not be repurposed as reset authority. The split-ready state has a `60 policy tick / 240
physics substep / 1.2 s` PhysX nominal-hold receipt, which covers the hidden WAIT maximum of 25
control ticks. Each episode privately samples 5--25 WAIT ticks; physical state and teacher both
remain split-ready, while the not-yet-active ball is held in a no-contact parked state rather than
reverse-integrated through table/floor collisions. The reveal event atomically installs the sealed
incoming-ball launch state and complete task tuple; on that same tick the teacher changes to measured frame 0
and the actor sees the original `time_to_teacher_start` (about
`.712376 s` for the historical N1), so dense mimic learns the split-ready-to-frame0 bridge. No hand-
authored robot connector, robot teleport or robot reset rewrite is part of this contract; installing
the ball launch state is the task-start event itself. The separate `200/800`
durability soak and a 4 s passive hold are not birth or first-learnability prerequisites. The policy
therefore never needs the raw teacher-base block: it receives the teacher/reference stream and task-
relative contact information that explain the transition and professional swing adaptation.

Both actor and critic append one atomic `task_valid` scalar. Therefore actor width is
`225 - 15 + 1 = 211`; the historical 318-D critic did not own that actor teacher-base block, so its
then-fresh width was `318 + 1 = 319`. A and C retain separate normalizer/checkpoint lineages:

| Contract | Task block | Final rows | Meaning |
| --- | --- | --- | --- |
| `action_ball_a211` | desired contact `position3/velocity3/signed-face3` | `desired_base_xy2 + time_to_contact1 + time_to_teacher_start1 + task_valid1` | contact-oracle A |
| `action_ball_c211` | incoming ball-at-contact `position3/velocity3/spin3` | same five rows | direct ball-state C; fixed table midpoint is not repeated |

The complete actor v2 order is frozen below. Equal width is not compatibility: a pre-IMU A211 row
with world angular velocity in `[12:15]` is a different ABI and must fail closed.

| Slice | Ordered term | Dim | Source / exact meaning |
| --- | --- | ---: | --- |
| `[0:12]` | `actual_base_pose_lin_vel_world` | 12 | localizer/fused root: HOPE-world position3, orientation6D and linear velocity3 |
| `[12:15]` | `base_ang_vel_body` | 3 | simulator root angular velocity in pelvis/body coordinates; deployment maps the calibrated pelvis IMU gyro into the same frame |
| `[15:46]` | `joint_pos` | 31 | actual `q-default_q` |
| `[46:77]` | `joint_vel` | 31 | actual encoder velocity |
| `[77:108]` | `actions` | 31 | previous normalized actor output |
| `[108:117]` | `racket_site_achieved_now_heading` | 9 | achieved official-site position/point-velocity/signed-face |
| `[117:148]` | `teacher_joint_pos` | 31 | current teacher `q_ref-default_q` |
| `[148:179]` | `teacher_joint_vel` | 31 | current teacher `dq_ref` |
| `[179:188]` | `racket_site_teacher_now_heading` | 9 | measured-racket teacher site now |
| `[188:197]` | `racket_site_teacher_at_reference_hit_heading` | 9 | measured-racket nominal site at reference hit |
| `[197:206]` | A desired contact / C incoming ball | 9 | A=`position3/velocity3/signed-face3`; C=`position3/velocity3/spin3` |
| `[206:208]` | `desired_base_xy_world` | 2 | task base station |
| `[208:209]` | `time_to_contact` | 1 | public ball/contact clock |
| `[209:210]` | `time_to_teacher_start` | 1 | public teacher-start clock |
| `[210:211]` | `task_valid` | 1 | atomic WAIT/TASK_ACTIVE validity bit |

The actor trainability and empirical-normalizer identities are v2
(`action_ball_a211_fixed_question_learnability_v2` /
`action_ball_c211_fixed_midpoint_learnability_v2` and
`action_ball_{a,c}211_actor_norm_v2`). The 319-D privileged A/C critic content and normalizer keep
their existing v1 identities across this actor-side IMU repair; it remains the paired 318-D A/C
critic layout plus final `task_valid`, not an authorization to consume any unrelated historical
318-D critic. Consequently all pre-IMU 211 actor normalizers/checkpoints are rejected even though
their tensor shape matches, while the critic is not needlessly renumbered.

During `task_valid=0`, the task block, base goal and both public clocks are exactly zero. All A/C task,
contact and outcome rewards and their opportunity/closed-swing/outcome denominators are ineligible
and therefore contribute neither numerator nor denominator; balance, safety and non-task whole-body
mimic remain active. At reveal, the complete tuple and both clocks become valid atomically. Once a
valid swing is active, a completed miss is reported honestly as zero successes over a non-zero
closed-swing/opportunity denominator; only the achieved-outgoing-flight outcome denominator can stay
zero until such a valid flight exists. The private random wait countdown is not an observation.
The runner uses the raw final validity bit to re-mask actor `[197:210]` and critic `[305:318]`
**after empirical normalization** on the fresh initial observation, every rollout next observation,
and the bootstrap/terminal value observation. Therefore a normalizer's learned non-zero mean cannot
turn a raw WAIT zero into hidden task information; the normalized validity indicator itself remains
present as the state bit. This post-normalizer mask does not add parameters or change the 211/319
state-dict layout.

For C211, “paddle centre” in the nominal-strike proximity reward is exactly the live URDF
`official_racket_site` (`pingpang_red_Link` origin), which is also the retargeting target of the
measured `physical_blade_center` channel. The reward must not add the red/black rubber-area-centroid
offset to that point. Selected-rubber offsets remain part of the exact contact classifier and the
achieved-outgoing-flight calculation, where the contacted surface actually matters. Thus the dense
distance and measured retargeting use one paddle-centre ground truth, while contact physics may use
a distinct face centre and ball-centre offset.

The exact frame/packing contract is:

```text
position_heading = R_heading^T (position_world - current_base_position_world)
velocity_heading = R_heading^T velocity_world
spin_heading     = R_heading^T spin_world
face_heading     = R_heading^T signed_face_world
orientation6D    = [R00,R01,R10,R11,R20,R21]
```

Every actor term and its paired critic/exogenous term must come from one atomic same-control-tick
snapshot/generation. Mixing a new ball packet with an old base heading or teacher phase fails closed.
Base orientation uses the continuous 6-D rotation representation above. A paddle face uses a signed 3-D
unit normal, not the old 194-D compatibility layout's `normal3 + legacy rho0` four-vector. Measured
long-axis remains a full-phase paddle-mimic reward/metric together with position, velocity and signed
face; it is not silently inserted as another actor column because that would change the versioned
ABI. A consumes no separate incoming-ball row because desired contact is the ball/question summary;
C consumes no contact target or fixed-midpoint task row. Formal A keeps `online_solver` as its
question source and may reuse only an exact answer keyed by the complete question semantics: sampler,
curriculum and RNG advance on every reset; the first cold Q and each changed Q' really solve once;
replay/assert/checkpoint paths do not re-solve. Formal C uses `direct_ball`, performs zero inverse
calls and has no answer cache. `immutable_tape` is a historical target-information ablation fixture,
not an A211/C211 lineage input. An offline `banded_question_bank` was only an optional future
producer optimization and is not a prerequisite for the first expanding long.

The following paragraph applies only to historical `L194/H225`, not to the historical A211/C211 row.
`L194` reads its receipt-owned continuous pre-swing deadline. `H225` has no ball receipt and reads
`max(MotionCommand.hold_counter_after_current_update, 0) * policy_dt`: the counter is the number
of future frozen-reference control steps visible to the next action.  Consequently the final
already-executed held step reports zero even though `in_hold` remains true for that step's
reward/termination accounting.  The current VendorV2 recipe has zero hold, so the value is
currently zero without becoming an action identity placeholder; a later same-ABI ready wait can
activate it without changing the column or reviving `action_one_hot`.

The four paddle positions all subtract the **current actual base position** before heading rotation.
Their linear velocities are absolute site velocities rotated into heading; base velocity is not
subtracted, because contact physics depends on absolute paddle speed.  Normals are rotation-only.
Paddle teacher-minus-achieved and task-minus-achieved residuals are deliberately omitted: an MLP can
form those subtractions from adjacent paired fields, while duplicating every paddle residual inflates
the versioned ABI. No teacher-base block is present in A211/C211. `projected_gravity` is omitted because actual base orientation
already determines it. `action_one_hot`, `swing_type` and zero `rho` are excluded.
Paddle angular velocity is deferred until off-centre/an-isotropic contact makes it independently
necessary; one-frame delay history is a separate Markov audit, not a hidden addition to this repair.
A measured butt-to-blade long axis is now a low-weight full-window reward channel. “Free wrist from
body mimic” means the opposite of adding the old generic wrist-body reward: the striking wrist is
removed from generic body position/orientation/velocity imitation so that it cannot fight the
paddle task. Outside contact, measured official-site position, point velocity, signed face and
butt-to-blade long axis replace it with a more direct rigid-paddle teacher; inside contact, the first
three coordinates yield to the ball-task target and long axis remains only a low-weight nullspace
pin. All three right-wrist joints remain in the 31-D action, so this reward can teach their motion.
This is not yet proof of dynamic learnability or validated off-centre spin/contact.
A uses contact-target income in a valid strike window and achieved-outcome income only after actual
contact. C has no contact target and its task reward is deliberately minimal: one nominal-strike
Cauchy paddle-centre-to-ball-centre distance term, plus one landing outcome gated by actual
selected-rubber contact and a valid achieved outgoing flight. It has no desired-contact reward,
independent hit bonus or additional dense outcome term. Full-phase measured-paddle/body mimic remains
a shared non-task prior, not a hidden C contact oracle. A C run must never regain an oracle target
through the critic, Reward, normalizer or checkpoint side channel.

For the five-update scale gate, only qdes-hard, actual-hard and nonfinite are implementation
strict-zero accounts. Fall, base-too-low and robot-hit-table remain real terminations, but are initial-
policy behavioral evidence: report them separately for hidden-WAIT, revealed-pre-strike and
post-strike phases with conserved denominators instead of requiring zero occurrences. The historical
MuJoCo A211/C211 host paths preserve the same split-ready/WAIT/reveal contract, but each family's
`1 env x 2 PPO update + save/fresh-process cold-load` remains exact-Pod `未测` and cannot authorize
canonical N1 or 4096-native training.
A future outgoing-spin target is an explicit 3-D outcome block with frame/unit/validity, not a
revival of the scalar placeholder.

The historical ball-free `H225` canary's paired critic is an exact ordered 318-D contract:

`command62, motion_anchor_pos_b3, motion_anchor_ori_b6, body_pos42, body_ori84,
base_lin_vel3, base_ang_vel3, joint_pos31, joint_vel31, actions31,
teacher_at_reference_hit9, desired_at_contact9, desired_base_xy_world2,
time_to_contact1, time_to_teacher_start1`.

The first ten terms are the historical 296-D privileged body/reference stream; the final 22 values
are exogenous teacher/task conditions needed to predict returns. Withholding them from the critic while exposing
them to the actor would create an avoidable partially observed value function. Current
achieved/teacher-now paddle states need not be duplicated into the critic because its privileged
body/reference state already determines them. Schema-3 records and checks ordered names, every
per-term dimension and total width from the instantiated ObservationManager; width 318 alone is not
an identity. In particular, this critic is not the same contract as the fixed-194 diagnostic critic
merely because both total 318 scalars. It is also not an authorized critic for `A225-proto` or
`C225-proto`: both prototypes now have separately registered 318-D critic ABI and normalizer
identities, and must not silently reuse this historical `318` or one another's checkpoint lineage.

This historical A211/C211 V2 preserved vendor-scale joint observation noise (`q ±0.01 rad`, `dq ±0.5`) and a
clean privileged critic. The new 15-D `actual_base_now_world` block is currently noise-free: the
legacy `base_ang_vel ±0.2` and `projected_gravity ±0.05` knobs do not define physically valid
noise for a mixed position/orientation/linear-velocity/angular-velocity block. Base pose/twist noise
must be specified per mocap/IMU component instead of applying one uniform perturbation to all 15
columns.

This was a warm-start-breaking contract migration. It is retained as ABI history, not as permission
to relabel a `211/319` checkpoint as current semantic V2. Isaac and MuJoCo had to independently construct
the same ordered row and pass fixed-tape parity; actor, critic and Reward had to read the same
site/frame/sign truth. Existing 170-D and fixed-194 checkpoints remain historical inputs and are
never relabelled as this contract.

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

### Actor-visible planner tuple timing

`racket_target_pos_b`, `racket_target_vel_w`, the optional 179-D face tail, swing identity and
`time_to_strike` describe one atomic estimate and must not mix different source times. The legacy
delay ablation exposes three explicit modes through `task.racket.target_delay_tts_mode`:

- `live` preserves the historical contract and keeps the live countdown beside a delayed target;
- `source_timestamp_compensated` delays the complete tuple and converts the source countdown to
  current remaining time by subtracting the configured transport delay;
- `uncompensated` delays the complete tuple without compensation and exists only as an engineering
  negative control.

Only the actor sees the delayed/compensated countdown. The critic, Reward eligibility and truth
metrics keep the current simulator countdown. Reset and dropout operate on the whole tuple, not
individual fields.

The opt-in [`planner task revision`](../DEFINITIONS.md#planner-task-revision) path supersedes the
old mid-swing freeze without turning every update into a new stroke. One physical ball owns one
`(control_epoch, task_id)`; each newer estimate increments `task_revision` and atomically revises
position, velocity, physical face-B normal and TTS. The runner may consume the newest revision at
any actor tick before reference contact. A bounded [`phase governor`](../DEFINITIONS.md#phase-governor)
changes playback rate without reversing phase or exceeding the checkpoint-bound rate,
acceleration, target-displacement or deadline envelope. Side and clip identity remain immutable
within one task, and contact/plane/deadline closure makes that task consume-once. Training keeps
the physical question, Reward truth and critic truth immutable while exposing the same revision
protocol to the actor. This is the required answer to the old target/TTS freeze; it is source-level
only until the full-scene and vendor gates below run.

In the current `phase_governor_v1` cutover, `target_delay_steps` must be `0`:
the legacy delay ring delays actor observations but would let the motion governor consume a source
revision earlier. Any positive delay therefore fails closed until a coupled provider/transport
ring can submit the complete `(epoch, task, revision, target, TTS)` to governor and actor in the
same policy tick. Timestamp-compensation pairs with positive delay are `NO-LAUNCH`, not evidence.

### Other registered actor layouts

| Dim | Contract | Delta / source | C++ publish status |
| --- | --- | --- | --- |
| 177 | `hitter_footwork` | 175 layout with `base_target_pos_b(2)` inserted after projected gravity; requires fresh external/oracle base localization. | Supported, but publication fails closed without fresh localization. |
| 179 | `deploy_parity_face179` | Exact 175 prefix + tail `racket_target_normal_cmd(3), rho(1)`; actor tail is raw mount +Y/A after the runner converts the physical-B wire normal with the selected clip sign. | Legacy face-only input uses schema 2; the existing formal atomic transport uses schema 3. The opt-in live-revision path requires exact schema 4 plus matching `planner_task_revision` ONNX metadata and `--planner-task-revision`. One older envelope-bearing model passed strict Release preflight, but no task-revision model has passed a backend first tick or vendor Gate 3 behavior. |
| 181 | `deploy_parity_station181` | Exact 179 prefix + tail `station_anchor_err_b(2)`. | Blocked: wire and the unique station/normal term order are not frozen. |
| `181+N` | `task_first_n<N>` | Exact `hitter_footwork(177)` prefix + `racket_target_normal_cmd(4)` + `action_one_hot(N)`；manifest/action/motion order 必须逐项相同。 | Training-only source candidate；无 ball、无 production planner wire，Pod Isaac 未测。 |
| `181+N` | `action_ball_n<N>` | 与 `task_first_n<N>` 逐列同构，但 checkpoint 身份另名：动作先冻结，再由该动作的来球与 fixed-action solver 产生 task。当前 N=1 总宽度为 `182`。 | Isaac trainer 已实现并有 Pod 证据；无 production arbitrary-N wire/C++ consumer。 |
| `190+N` | `action_ball_table_pose_n<N>` | `hitter_footwork(177)` + `base_position_table(3)` + `base_orientation_table_6d(6)` + demanded signed face / reserved spin `(4)` + `action_one_hot(N)`。当前 N=1 总宽度为 `191`。 | Feature-branch Isaac source candidate；Pod evidence 未完成，production arbitrary-N/191-D flat wire 与 C++ consumer 均不存在。 |
| `193+N` | `action_ball_table_pose_twist_n<N>` | 历史 194-D 候选；position residual 在 heading frame，但 velocity/normal 仍在 world frame。 | 只作旧 checkpoint/诊断兼容，不得用于 fresh launch。 |
| `193+N` | `action_ball_table_pose_twist_heading_task_n<N>` | table pose/twist 与上一行同宽；racket position residual、demanded velocity、raw-A normal 全部统一到 yaw-heading frame。 | Pod N1 `1×2/4096×5` 已过且 1000-update 正在运行；production arbitrary-N flat wire、C++/MuJoCo consumer 与真实速度估计器尚未闭合。 |
| 194 | `action_ball_table_pose_twist_heading_task_teacher_start_v2` | frame-consistent table pose/twist/task 后显式加入 `time_to_teacher_start_s(1)`，不再把 `action_one_hot`/UID/slot 喂给 actor。 | fresh N1 首选；旧同宽 194-D 与历史 195-D 都不重标、不 exact resume。formal N73 由所选 teacher trajectory 本身区分动作，不另加 motion-intent code；仍须 Pod 194-D v2 构造 smoke，production consumer 未闭合。 |
| `194+N` | `action_ball_table_pose_twist_heading_task_teacher_start_n<N>` | 历史 teacher-start + final N-wide one-hot 合同；N1=195-D。 | 只保留旧 checkpoint/receipt/schema 解析，不再用于 fresh ActionBall launch。 |
| 110 | `hitter_pure` | HITTER Table-I style: 99-D proprio prefix + base forward(2), station delta(2), racket target rel base(3), target velocity(3), tts(1); no reference command or swing flag. | Supported; requires fresh localization and metadata-bound per-side station geometry. |

Do not infer a contract from width alone. Formal consumers require the registered name, mode,
ordered term names/dims and total dimension to agree. The 179 C++ path accepts only
`deploy_parity_face179` metadata and the wire schema named by that runtime profile; it also binds
face enabled, `shared_plus_y`, `mount_plusY_A`, exact schema-3 train split, train-bank SHA and
source-family SHA. A schema-4 launch additionally double-keys the runner flag, exact task-revision
metadata and wire width. Schema 1 never fabricates the tail.
Merely accepting 181 in an input-shape whitelist would still create a right-width/wrong-columns
command, so 181 remains rejected until its unique station/normal order is frozen.

### Historical N=1 ActionBall actor: exact 182-D layout

Historical upper backhand loop/block runs used `action_ball_n1`: the exact
`hitter_footwork(177)` prefix followed by demanded signed face plus reserved spin scalar `(4)` and
the frozen one-action identity `(1)`. The prefix is:

`command(62), motion_anchor_ori_b(6), base_ang_vel(3), joint_pos(31), joint_vel(31),
last_action(31), projected_gravity(3), base_target_pos_b(2), racket_target_pos_b(3),
racket_target_vel_w(3), time_to_strike(1), swing_type(1)`.

There is **no absolute base world position and no absolute world yaw/quaternion column**. World
base pose is used internally to form the two relative base-target coordinates and the
base-yaw-frame racket-target residual. In Isaac, the orientation/angular-velocity/gravity inputs
come from simulator rigid-body truth plus configured observation noise; this is not an integrated
IMU drift model. The historical A3 consumer used pelvis-IMU orientation-derived terms and external
mocap position only; the fresh contract below supersedes that source split.

Fresh N=1 ActionBall runs may opt into the
[action-specific dynamic ready](../DEFINITIONS.md#action-specific-dynamic-ready) reset contract.
The physical robot state and teacher frame 0 remain the exact motion bytes, while the action manager,
`last_action`, processed/pre-clamp qdes buffers and fresh actor output-layer bias all start from the
same nominal-hold-certified qdes. Raw policy-history validity remains false across reset, so this
initialization cannot masquerade as a sampled transition. The policy action remains the same 31-D
unbounded Gaussian output followed by the existing affine decoder and finite qdes projection; no
observation or action width changes.

### N=1 ActionBall frame and table-state semantics

The exact current order and slices are frozen in the truth table above.  The first twelve terms
retain the 177-D HITTER-footwork shape, but the versioned contract replaces the historical
world-frame velocity slot with `racket_target_vel_heading` without moving its offset.  The fresh
v2 tail replaces the old constant `action_one_hot(1)` with `time_to_teacher_start_s(1)`, so N1
remains exactly `194` dimensions even though the semantic contract changes.  The historical
same-width `action_ball_table_pose_twist_heading_task_n1` instead ended in `action_one_hot(1)` and
must not be inferred or resumed by width.
`base_position_table` is the current root XYZ in the table-surface-center frame, not an arbitrary
venue-world coordinate. `base_orientation_table_6d` is the full table-to-base rotation
`R_table_base`, encoded from its first two columns in row-major tensor order:

```text
R00, R01, R10, R11, R20, R21
```

This is a complete 3-DoF orientation, **not yaw-only**: after orthonormalization, the two encoded
axes uniquely determine the third axis by a cross product. The repository uses the first two
matrix columns because of its rotation convention; libraries that transpose the convention may
spell the same construction with rows, so byte ordering must follow this contract rather than a
generic example.

The representation choice follows Zhou et al.,
[“On the Continuity of Rotation Representations in Neural Networks,” CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html):
Euler angles and quaternions introduce discontinuities when exposed as Euclidean network features,
whereas 3-D rotations admit continuous 5-D/6-D representations. The 6-D construction therefore
keeps roll, pitch and yaw information without Euler wrap/gimbal singularities or quaternion
`q/-q` sign ambiguity. This is a representation/contract correction, not a Reward hypothesis;
it needs Pod tensor/recipe parity, not a learning A/B.

Task geometry remains relative for generalization. In particular, `base_target_pos_b` remains the
base-goal residual from the current base in the robot yaw frame, and `racket_target_pos_b` remains
the target-racket FK residual in that frame. The extra table-pose channels answer the different
question “where and how is the robot standing with respect to the table?” They must not be used to
turn the task target back into an absolute world coordinate.

All actor-visible racket-task vectors use the same heading rotation:

```text
p_task_h = R_heading^T (p_target_table - p_racket_FK_table)
v_task_h = R_heading^T v_target_table
n_rawA_h = R_heading^T n_rawA_table
```

`rho` is a scalar and is unchanged. The planner wire, fixed-action solver, critic, Reward and
physics remain canonical world/table-frame quantities; conversion happens only at the actor
observation boundary. This change removes a deterministic coordinate mismatch and therefore does
not need a learning A/B, but it does require Pod tensor/construction parity and a new contract name.
An old `action_ball_table_pose_twist_n1` checkpoint is not resume-compatible even though it is also
194-D.

`base_lin_vel_heading` is root-rigid-body COM velocity rotated into the base yaw-heading frame.
It is neither a raw single-frame mocap difference nor an IMU acceleration integral. Deployment
uses a causal estimator with OptiTrack position as the drift-free anchor, optional IMU
accelerometer propagation, marker→COM offset correction and the `omega × offset` term. This
explicit velocity closes a real Markov-state omission in the feed-forward actor; its empirical
noise/latency model must still be calibrated before deployment.

For real A3 consumption, pose and velocity use different authorities by physical quantity:
calibrated OptiTrack owns table-relative position, full orientation and projected gravity;
the pelvis IMU three-axis gyroscope owns `base_ang_vel`; and the causal fused estimator owns
`base_lin_vel_heading`. `motion_anchor_ori_b` combines OptiTrack base orientation with
joint-encoder FK. The current C++ deploy builder cannot construct either the compatibility 194-D
layout or the fresh fixed-width 194-D v2 successor observations,
and the marker-cluster rotational extrinsic, estimator delay/dropout and frame parity are not yet
closed, so this training contract does not authorize a real-robot run.

The first N1 wave intentionally remains single-frame. A frame-history ring is stateful at the
environment level and changes reset/exact-resume semantics; it is not a free 72-column append.

### Fresh N1 teacher-start and reference semantics

Fresh launches after this interface cut use
`action_ball_table_pose_twist_heading_task_teacher_start_v2`. It replaces the N1 predecessor's
constant `action_one_hot(1)` tail with `time_to_teacher_start_s(1)`, so the width remains 194-D
while term identity changes. N-wide one-hot does not encode motion similarity and changes network
width with the bank size, so fresh ActionBall no longer attaches it. UID/slot remains mandatory in
the control plane for receipts and accounting, but it is not a policy observation. Multi-action
training uses the different selected teacher trajectories already present in the observation; an
extra content-identity vector is neither required nor authorized without a measured observability
failure.

`time_to_teacher_start_s` is the live Motion phase-governor clock:

```text
time_to_teacher_start_s = max(pre_swing_wait_s - action_ball_task_age_s, 0)
```

It resets to the selected task receipt's full wait, decreases by one policy `step_dt` after each
physical tick, clamps at zero and remains zero through swing/recovery. During the positive interval
the 62-D teacher command stays at frame 0 with zero reference velocity; afterwards it advances at
`teacher_rate`. Motion timing is resolved immediately after Racket publishes the task receipt so
the first post-reset observation cannot report a false zero. Although the scalar could be derived
from TTS, target-speed magnitude and action constants, making it explicit avoids asking an MLP to
learn a norm, division and per-action lookup. Existing live 194-D runs keep their exact old bytes
and are not relabelled.

The new fixed-194 v2 N1 contract still excludes localization freshness because the current OptiTrack
dropout/latency distribution has not yet been measured. A deployment-intent successor should add
at least `base_localization_age_s(1)` and `base_localization_valid(1)` in one versioned,
warm-start-breaking migration after the live producer uses capture/Motive timestamps. Held poses
must retain increasing age and must never be converted into zero velocity merely because no new
sample arrived. Adding those two fields would make the N1 contract 196-D; neither historical nor
current 194-D/195-D checkpoints may be reused as exact resume. Mocap coordinate conversion, filtering,
timestamp alignment, sensor fusion, dropout admission and hard-stale stopping remain outside the
policy. The actor sees age/valid only when the supervisor intentionally permits a short hold-last
interval; if the supervisor instead guarantees fresh localization on every control tick and stops
before policy evaluation, freshness can remain supervisor-only.
History is reconsidered only after the measured OptiTrack/IMU pipeline shows a residual
partial-observability or delay problem that cannot be represented by the coherent pose/twist
packet. At that point the first canary is a small critical-channel stack, not an unbounded history
or an immediate recurrent-policy migration.

Noise and delay are injected at the coherent sensor-packet producer, not independently into
`base_position_table`, 6-D orientation, projected gravity and derived task vectors. For deployment-
intent retraining, each episode samples one calibrated table/base extrinsic bias, tangent-space
SO(3) noise, causal packet age/hold-last behavior and gyro bias; derived terms consume the same
packet. The older ChingMu profile (`1.9 mm` white position noise, `5.2 mm` AR(1) marginal noise,
`rho=0.717` per 50-Hz policy tick) is only an order-of-magnitude prior. OptiTrack has a different
camera/solver/filter pipeline, so its live timestamped error spectrum and latency must replace
those numbers before deployment-intent retraining. Guessed independent `±0.01 m`, 6-D component
noise or blanket `20–40 ms` delay does not enter the first N1 wave.

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
Schema 4 is an exact 22-double extension of schema 3. It adds `task_id` and `task_revision`; both
are non-negative integers exactly representable as doubles. For one active ball, task identity is
stable and revision strictly increases. A missing, fractional, stale, cross-task, post-contact or
out-of-envelope revision fails closed. Solver valid/invalid jitter does not create a new task.
Only a lifecycle-proven closure followed by a new inbound ball may allocate the next task. This
separates “refresh the same strike” from “start another action” and closes the repeated-task bug at
the same interface boundary.

Schema 4 仍只传二动作 side/sign，不传
[stable action UID](../DEFINITIONS.md#stable-action-uid)。任意 N 动作的候选生产接口是 schema 5
exact23：在 exact22 后用 `[22]` 传 stable `action_uid`，并在同一
`(control_epoch, task_id)` 内冻结；C++ 通过内容绑定 catalog 把 UID 映射到本地 slot。该 schema
目前只是待冻结设计，Python selector core、right-width tensor 或文档本身都不能把它写成已实现。

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
`--dry-run`, and `--model-preflight-only` still require the same selected wire-schema packaging,
exact complete schema-3 execution metadata and (for 179) envelope as live publication. Only explicit
`--allow-legacy-model-diagnostic` relaxes legacy model loading; it requires no-publish and cannot
be combined with model preflight. Therefore an accepted preflight certificate always reports
parsed `publishable_model_contract=true` and `training_contract_exact=1`.

For a task-revision-trained export, the checkpoint hard contract owns two deliberately separate
records. `planner_task_revision` is the runtime contract and has exactly four fields: enabled,
revision schema version, complete governor document and enclosing initial-TTS range. The ONNX
embeds the canonical form of this runtime record and its SHA. `planner_task_revision_training`
records the explicit weighted [`initial TTS mixture`](../DEFINITIONS.md#initial-tts-mixture) and
training-only revision noise/counters; it is not copied into the runtime metadata. Native and
standalone export both reconstruct the four-field runtime payload from the source checkpoint,
reject missing/extra/tampered fields and never borrow it from a donor ONNX. The C++ runner requires
this exact metadata only when `--planner-task-revision` and schema 4 are both selected; any partial
combination fails closed.

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
- Active Isaac and portable MuJoCo do not clip the normalized policy action before that affine
  decoder.  Isaac fixes `clip_actions=null`; portable MuJoCo requires `TaskCfg.action_clip=None` and
  rejects every non-`None` setting.  Policy action, current-state observation, teacher and PD vectors
  all use the tracked runtime/schema-2 joint order; the sole runtime-to-MJCF actuator permutation is
  the final plant `ctrl` write.  For portable FullMDP, a true Gym reset writes
  `runtime_plant.default_joint_pos_rad`, the configured default root translated by `env_origin`,
  zero joint/root velocity, and zero current/previous action history; `take061/q_ready` is provenance
  only and is not the physical reset pose.  Natural shot retirement is not a reset and does not change
  `reset_generation`.  The affine decoder is
  `q_raw = runtime_plant.default_joint_pos_rad + action_scale * action`, followed in MuJoCo by its
  compiled mechanical hard-range clamp.
  If one affine q-des element is NaN/Inf, that element uses the preceding finite action for this one
  safe transition (zero raw action after reset, hence the configured action offset), while the raw nonfinite request is latched
  into termination after physics.  Thus nonfinite data never enters the plant.  This elementwise
  fallback and the decoded q-des clamp are not a raw-policy clip; the historical MuJoCo-only
  `[-4,+4]` clip diverged from Isaac and must not be restored.
  This closes raw proposal/order/scale/offset, not the executable target guard: active Isaac also
  applies its soft inset, finite projection and state-dependent q/qdot brake.  Until MuJoCo consumes
  that same owner, the guard axis is `DIVERGENT_DECLARED`; MuJoCo-only diagnostics are allowed, but
  checkpoint transfer, promotion and matched cross-engine causal claims are not.
- The vendor ActionBall plant additionally distinguishes the runtime mechanical position ledger
  [`Hmech`](../DEFINITIONS.md#h-mech) from a PhysX control-position envelope
  [`Hctrl`](../DEFINITIONS.md#h-ctrl).  The selected runtime-ordered joints are
  exactly `waist_roll_joint`, `waist_pitch_joint`, `left_ankle_roll_joint` and
  `right_ankle_roll_joint`; each Hctrl side is inset by exact float `0.02` of that joint's Hmech
  span.  The other 27 control rows equal Hmech byte-for-byte.  This does not change the articulation
  mechanical ledger, the existing soft q-des envelope, the 31-D actor action, Reward or observation.
  A source implementation without a fresh 16-environment differential Pod receipt is only a
  candidate plant, not training admission.
- Formal MuJoCo BankExam realizes an Isaac implicit joint's bounded drive as one total operation,
  `tau = clip(kp*(q_des-q) - kd*qdot, -effort_limit, +effort_limit)`, recomputed every physics
  substep. It does **not** clip P before D and does not place actuator kd in MuJoCo passive damping:
  passive damping lies outside motor `ctrlrange` and cannot share Isaac's total-effort limit.
  Retained passive damping or missing effort limits therefore makes an implicit replay explicitly
  diagnostic; formal construction fails before rollout.

For fresh training, the 2026-07-31 latest Agibot table is the nominal authority for the 29 non-head
degrees of freedom, including armature and the action scale derived from
`0.25 * effort_limit / stiffness`; the head keeps repository nominal values because the vendor
table has no head rows. This is a training identity break from old URDF/MJCF/deploy constants, not
a claim of hardware or cross-engine parity. Every old checkpoint/bundle must retain its old
metadata rather than being relabelled.

The vendor ActionBall profile adds
[`control-step action delay`](../DEFINITIONS.md#control-step-action-delay) at the normalized
policy-action boundary: actor output is delayed **before** `JointPositionAction` performs the
affine q_des conversion. One scalar lag is sampled per environment at true episode reset and
selects a complete 31-D action row, so the five actuator groups cannot receive different lags in
one episode. Reset fills every queue age with normalized zero/default q_des; dynamic-ready replaces
that fill with the action-specific normalized hold in the same rollback transaction without
resampling. PPO log-prob, `ActionManager.action`, and the actor-visible `last_action` remain the
current actor output, not the delayed drive row. A zero maximum lag uses the original no-queue/no-RNG
path. Delay is training DR and is not baked into the ONNX action decoder.

### Vendor evaluation profiles

The exact vendor task has two explicit
[`A3 vendor eval profiles`](../DEFINITIONS.md#a3-vendor-eval-profiles), applied after task
composition and before `gym.make`:

- `vendor_play_v1` mirrors the vendor Play table: startup plant randomization and interval push are
  disabled, while policy observation corruption and the episode-sampled `[0,2]` action delay stay
  enabled;
- `deterministic_ranking_v1` starts from that profile and additionally removes policy observation
  corruption, action delay and physical reset-state noise for repeatable checkpoint ranking.

Both preserve the 194-D actor and 31-D normalized action contract and emit
`VENDOR_A3_EVAL_PROFILE_JSON`. A result must name the profile; deterministic ranking cannot be
reported as vendor Play robustness. The host vendor-eval/canonical/formal suite passes
`128 passed`; no Pod evaluation is claimed here.

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
- when the vendor Hctrl envelope is enabled, a required `physx_control_position_limits` block with
  backend `physx_root_view_dof_limits`, exact four-joint ordered selection, exact per-side fraction
  `0.02`, full `31×2` Hmech/Hctrl matrices, `unselected_joint_count=27`, and explicit true invariants
  that unselected rows equal Hmech while the mechanical and soft-q-des ledgers remain unchanged.
  Runtime contract construction verifies composed config against the live PhysX getter and omits
  only the environment-count-dependent readback SHA from the hard semantic block, so `1 env` smoke
  and `4096 env` probe can share one scientific identity.  Missing, reordered, cross-environment
  divergent or numerically inconsistent rows fail closed; old schema-3 bytes remain historical and
  cannot be relabelled.
- physics dt, policy dt and decimation (`policy_dt = physics_dt * decimation`);
- exact actor term names/dims/history, articulation and tracked body
  order/indices, anchor and motion segment lengths, per-clip FPS and the
  schema-2 motion-kinematics contract;
- task timing/target/bank facts needed to prevent exporting an old actor under a new evaluation
  recipe.
- when enabled, a `control_step_action_delay` block with unit `policy_control_step`, inclusive
  discrete-uniform range, once-per-episode sampling, shared-31-joint identity and queue-fill
  semantics. Disabled contracts preserve total absence of the block and the legacy schema-3 bytes.
- for a racket task, the post-override positional and signed-face guidance term identity
  (`weight`, canonical `racket_target` command name, and distance/angle cap). Most Reward weights
  remain curriculum-mutable and outside the hard contract; these two are the preregistered causal
  identity of the C2/D2 signed-face comparison, so copying a checkpoint cannot relabel `0.0` as
  `-0.4`.
- the canonical racket-point identity and wrist-local offset. See
  [racket_contact_geometry.md](racket_contact_geometry.md) for the distinction
  between the URDF site, physical face centre and ball centre at contact.

Fresh learning also validates actor/critic empirical normalizer state before the first rollout and
emits the three [vendor runtime JSON markers](../DEFINITIONS.md#vendor-runtime-json-markers). The
delay runtime receipt binds the training-contract SHA and ordered action-term identity; its
initialized count and lag histogram must conserve the environment denominator. These runtime
receipts prove the live ABI/state used by a run, not policy quality or deployment authority.

A command-bearing 62-D actor term additionally requires
`actor_leg_ref_mask_provenance_epoch=1`. Only the exact canonical masked/unmasked callable, or a
strictly empty **exact built-in type** `functools.partial` around it, may mint the epoch. The
exact-type rule is applied at every unwrap layer: a `partial` subclass can override `__call__`
despite exposing canonical `.func` and empty args, so it is a wrapper and must be rejected. Any
bound positional/keyword argument, wrapper or copied marker is likewise non-authoritative because
it can change the configured command semantics. Epoch 1 plus absent `actor_leg_ref_mask` proves
unmasked; epoch 1 plus the true-only `actor_leg_ref_mask=1` proves masked. The ONNX binding hashes
that fact together with the training-contract and source-checkpoint SHA; old artifacts cannot
acquire it by metadata backfill.

The adjacent `params/training_contract.json` is content-addressed; its SHA and schema are embedded
in every newly saved checkpoint. Only a fresh schema-3 run or a resume from an exact SHA-bound
schema-3 checkpoint can keep `training_contract_lineage_exact=1`. Legacy/missing/mismatch overrides
remain diagnostic forever and cannot be “washed” into formal status by one continuation save.
An optional fail-closed launcher may also write `training_launch_claim_sha256` into checkpoint
`infos`. That operational claim is deliberately outside the scientific hard-contract SHA; its
consumer must reconstruct the non-self-referential claim from exact control/source, optimization
recipe, host/GPU lane, seed/run/terminal identity and atomic claim-directory identity. Absence is
allowed for ordinary training, but a preregistered run that requires the claim must reject a
missing/mismatched field.
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
- Mocap during PLAY (OptiTrack rigid body, 360 Hz): robot base (pelvis) pose, ball position. Ball
  rotation/spin is planned for the physics-modeling phase — not yet measured.
- Mocap during DATA-COLLECTION/physics-calibration only: additionally racket pose, table 4
  corners, net 2 corners.

Rules: actor observations must be built only from deploy-available signals; rewards run in sim
only and may use privileged state; the critic may be privileged. The current 175-D actor uses only
robot-side signals + planner targets (no mocap terms at all). The historical fixed-194 contract and
the current FullMDP semantic V2 both require absolute table-relative context in addition to
robot-relative tasks. Semantic V2's actor uses calibrated table/root pose, a unit root heading and a
causal root inertial-COM velocity estimate, while projected gravity and angular rate remain direct
pelvis-IMU quantities. Training must use each backend's exact counterpart of those same physical
terms. Deployment must close table and marker→root/COM extrinsics, gyro extrinsic, timestamp
alignment, velocity-estimator noise/latency/stale/dropout and the exact 203-D real builder first;
until then semantic V2 remains simulator-only (see G07 Next Steps).

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
- `racket.target_mode: task_first` is the ball-free, manifest-bound arbitrary-N candidate. It
  requires `task_first_n<N>`, balanced action sampling and the fixed per-action
  position→scalar-speed→face→base curriculum. Level 0 is center warm-up, not a small default
  perturbation. Full semantics and launch blockers are in
  [task-first contract](task_first_n_action_contract.md); it is not an adopted default.
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
