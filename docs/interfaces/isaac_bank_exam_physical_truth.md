# Isaac BankExam Physical Truth Interface

## Status And Authority

The interface is implemented locally but has **no currently authorized exact rider**. It is
simulator-only and evaluator-owned. The historical content-addressed rider
`configs/phase1_isaac_bank_exam_physical_truth_phase_b_contract_20260711.json` is revoked by
`configs/phase1_cross_engine_instrument_parity_2x2_revocation_20260713.json`: its checkpoint
predates explicit actor leg-reference mask provenance. The direct loader rejects that rider by
content SHA before source/profile validation, so bypassing the 2x2 validator cannot revive it.
The mechanism changes no
training recipe, frozen analytic threshold, actor observation, reward, reset, action, deployment
message, or real-robot command.

Phase B is opt-in through all three evaluator arguments:

- `instrument_physical_truth_phase_b=true`;
- `phase_b_contract=/absolute/path/to/contract.json`;
- `expected_phase_b_contract_sha256=<64 hex>`.

Missing, mismatched or revoked arguments fail before simulator construction. The checkout must
also be clean and detached. With the flag absent/false, the evaluator retains Phase A and
virtual-v1 behavior. Formal recovery needs a post-provenance-epoch checkpoint and a newly frozen
rider; the historical rider cannot be edited or metadata-backfilled.

The frozen fresh checkpoint predates the four T1 event-timing config fields now present in the
evaluation class. Generic exact-mode hydration still rejects all missing fields. The Phase-B
contract has one narrow compatibility seam: it requires `event_timing_mode=disabled` and may
materialize only the neutral values (`disabled`, empty schedule/path SHA, `repeat=false`) before
that generic audit. A present non-neutral value fails; this rider never evaluates Phase B and T1
at the same time.

## Contact And Rollout Contract

The dynamic ball collider is disabled. Code is the single ball/robot/table contact authority.
Each physics callback samples the same pure racket FK used by the command path, without writing
the command's reward/observation-visible racket buffers. The sample contains blade position,
orientation, link-point linear velocity, raw local-`+Y` normal, and signed striking-face normal.

One contact per swing is accepted when the ball center is inside the `0.075 m` blade disc and
either enters the ball-radius slab while approaching or crosses the signed blade plane between
physics samples. The crossing branch is the explicit anti-tunneling mechanism. At contact:

1. the ball position is snapped to the interpolated blade-plane crossing, or projected onto that
   plane for a slab-only contact;
2. outgoing velocity and spin are obtained by the exact
   `virtual_ball.predict_paddle_contact` implementation;
3. the ball enters return mode, with gravity integrated by PhysX, deterministic venue drag/Magnus
   applied through the existing physics callback, and table bounce applied by the fitted
   code-driven venue model;
4. the first positive-x net-plane crossing records net height, and the first descending
   `surface_z + ball_radius` crossing records landing.

`physical_ball_substep=1` is frozen. A missing physics callback may retain a non-formal
control-rate diagnostic internally, but its capability becomes
`phase_b_degraded_control_rate_no_physics_callback`; BankExam refuses it.

## Capability Metadata

The manager publishes `cross_engine_truth_metadata` with at least:

- `available`, `capability`, `physics_callback_active`, `racket_impulse_enabled`;
- `aero_substep`, `contact_authority`, `post_contact_rollout`, `collision_authority`;
- racket and ball radii plus a human-readable reason.

The only formal available capability string is
`physical_paddle_contact_and_post_contact_flight_v1`.

## Per-Question Outcome

At one-question finalization,
`cross_engine_physical_truth(env_id, expected_attempt_token=schedule_index, final=True)` returns:

```json
{
  "available": true,
  "capability": "physical_paddle_contact_and_post_contact_flight_v1",
  "attempt_token": 17,
  "served": true,
  "exact_seen": true,
  "contacted": true,
  "net_clear": true,
  "landed_ok": true,
  "returned": true,
  "landing_xy_env_m": [2.55, 0.03],
  "contact_authority": "code_driven_blade_disc_and_venue_paddle_impulse",
  "post_contact_rollout": "physx_gravity_plus_deterministic_venue_aero_and_code_table_bounce"
}
```

`returned` must equal `contacted && net_clear && landed_ok`. A completed no-contact attempt is a
valid all-false outcome with `landing_xy_env_m=null`. A contacted ball without a recorded landing
is unavailable and invalidates the entire physical cell; it is never silently scored false. The
attempt token must equal the immutable schedule index, and both `served` and `exact_seen` must be
true. Thus a reverse-serve failure cannot be mislabeled as a policy miss.

The result replaces only `attempts[*].instrumentation.physical_truth`, and the nested
instrumentation SHA is recomputed. Existing `hope.isaac-bank-exam.v1` `hit`/`returned` columns
remain the analytic virtual-v1 result for backward compatibility. The CSV schema is unchanged.

## Lifecycle And Failure Boundaries

After reset and atomic external motion/question installation, the evaluator calls
`begin_external_exam_attempt(env_ids, schedule_indices)`. That seam clears reset-time/train-row
truth and binds each env to one immutable generation. `on_resample` then consumes physics events,
publishes the just-ended swing before clearing its live
latches, and then rearms one-contact state. This protects clip-completion ordering when command
resampling happens before the evaluator sees the end of the step. The first unconsumed publication
wins, so a repeated empty resample cannot overwrite a just-ended hit/landing. Termination before
resampling uses the live latches with `final=True`.

The physical cell fails closed on any of the following:

- wrong checkpoint, training contract, bank, schedule SHA, schedule size, or source SHA;
- attached branch, dirty checkout, or HEAD change during evaluation;
- absent manager, disabled impulse, callback degradation, non-default substep;
- censored attempt, duplicate/missing attempt, pending contacted flight, non-finite landing;
- physical `returned` inconsistent with its three defining booleans.

For this instrument-parity paper, any pre-exact fall, guard reset, timeout, or missing physical
serve invalidates the whole cell; it is not entered as an ordinary all-attempt policy miss. This
is intentionally stricter than the legacy BankExam denominator because the 2x2 requires the same
instrument opportunity on every row. On such a rejected path, Isaac may already have reset before
the evaluator's fallback diagnostic snapshot is read, so that snapshot is not admissible evidence
and may describe the reset state. No scorecard is written from that path.

## Runtime-Unvalidated Geometry Limitation

The anti-tunnel test compares signed ball-to-blade distance at successive physics samples, so it
accounts for a moving blade in the crossing decision. The current contact snap interpolates the
ball segment against that sign change and evaluates disc radius at the current blade pose; it does
not interpolate the full blade pose/orientation across the substep. The clean-detached Isaac smoke
must therefore report contact-plane/disc residuals before this physical cell can be accepted. This
is a recorded runtime blocker, not evidence that the 2x2 gate is closed.

The operational command and simulator-dependent acceptance test are in
`docs/operations/run_phase1_cross_engine_instrument_parity_2x2.md`.
