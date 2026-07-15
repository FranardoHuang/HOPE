# T1 Post-Strike Event Training Contract

Status: **core training implementation plus frame-0 waiting design; not launch-ready**
(2026-07-15).

This note defines the runtime seam added after the frozen T0/T1 preregistration at commit
`1913861`. It does not mutate that preregistration or fill any of its null launch bindings.
Materialized schedules, continuous Isaac/MuJoCo judges, self-hit instrumentation, a fresh exact
baseline, and the semantics-correct plant contract remain blockers.

## Materialized schedule bytes

`MotionCommandCfg.event_timing_mode: post_strike_t1` requires both
`event_timing_schedule` and `event_timing_schedule_sha256`. The loader verifies the exact UTF-8
JSON bytes before parsing, rejects duplicate keys, non-finite constants, unknown executable row
fields, and any schema other than version 1.

```json
{
  "schema_version": 1,
  "schedule_id": "content-owner-defined-id",
  "policy_rate_hz": 50,
  "sequences": [
    {
      "sequence_id": "sequence-000",
      "rows": [
        {
          "question_id": "0123456789abcdef",
          "clip_id": 0,
          "bank_row": 12,
          "reveal_ticks_after_prior_strike": 35,
          "next_strike_ticks_after_prior_strike": 70,
          "available": true
        }
      ]
    }
  ]
}
```

`question_id` is the existing lowercase `blake2b64` Stage-1 question ID. Every row is checked
against the exact schema-v3 train-bank `(clip_id, bank_row)` before the scheduler can arm.
Environments use deterministic `env_id % sequence_count` assignment. Rows never repeat inside an
episode; a true sequence-boundary reset restarts the same environment's sequence.

## State transition contract

The scheduler is dormant until `RacketTargetCommand` observes an accepted exact-strike control
frame. Advancing time alone can never reveal a row. That exact frame becomes clock zero for the
first row. Thereafter each row is anchored to the previous **scheduled deadline**, regardless of
contact, return, or miss.

At a reveal tick, one command-manager compute installs the immutable schedule-row identity,
`clip_id`, train-bank row, native clip start, exact hold, target position/velocity/normal,
incoming-ball state, and absolute deadline. Motion publishes first and racket publishes second;
no reward, observation, or next policy action can observe a mixed command.

The installed clip always runs at one native frame per policy step. If `K` is its positive native
start-to-strike tick count and `N = deadline - reveal`, installation is feasible only when
`K <= N`; the exact hold is `N - K`. An unavailable row or `K > N` is marked on the ledger and is
not installed, replaced, censored, or shifted. Its original deadline still consumes one
opportunity. A due deadline must be finalized after same-step racket metrics; the next scheduler
advance fails closed if it was not.

Once armed, natural clip completion clamps at the clip's last frame. It cannot trigger random
wrap/resample. Event mode rejects `wrap_teleport=true`, clip switching, stagger, retiming,
within-episode schedule repeat, and RSI settle-frame skipping.

## Carry-state and history boundary

An event install does not call motion adaptive/resample logic, write root/joint simulator state,
write/reset the last action, or call `_reset_actor_target_state`. The robot pose/velocity, action,
observation history, delay ring, AR(1) noise, dropout countdown, hold-last message, and per-swing
bias state carry across. The existing A8 post-swing ring remains a true-reset initial-state
mechanism only; it is not an event-time teleport.

## Checkpoint hard contract

The training hard contract now binds the event mode and exact schedule SHA/byte length/sequence
shape, plus every existing field that can change strike/reveal/target timing: strike phases and
windows, episode horizon, hold/stand/post-swing entry, random clip switching, RSI frame skip,
stagger, native speed settings, adaptive motion sampling, midswing redraw, target delay,
jitter/noise/dropout, and post-strike dropout/bias state.

## Local verification

No Isaac, Pod, evaluator worktree, or robot command is required for the deterministic scheduler
tests:

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_event_timing_scheduler.py
```

The tests prove post-strike-only arming, miss consumption, fixed deadlines, explicit unavailable
and infeasible rows, exact-byte SHA binding, and the absence of reset/teleport/history-reset calls
from the two event-install methods. They do **not** prove continuous Isaac/MuJoCo parity, self-hit
safety, or launch readiness.

## Selected-action frame-0 waiting contract v2

The machine-readable design is
[`phase1_frame0_wait_recovery_contract_v2_20260715.json`](../../configs/phase1_frame0_wait_recovery_contract_v2_20260715.json).
Its human meaning is deliberately narrower than “ready equals frame 0”:

- the **waiting reference** is the currently public action's exact frame-0 pose, with root, joint
  and body linear/angular reference velocities all exactly zero;
- the frame-0 body is translated once in XY to the live station at reference-phase entry. That XY
  anchor stays fixed until the next atomic reveal or a true episode boundary; it does not follow
  the drifting robot every tick;
- after a swing and before the next reveal, the only public action is the just-completed action, so
  recovery points to that action's zero-velocity frame 0. The future action ID, clip, frame 0,
  target and deadline remain hidden;
- one atomic reveal makes the next action public and switches the reference to **that** action's
  zero-velocity frame 0. This only switches a reference: it does not write simulator root/joints,
  teleport, reset an episode, or clear observation/action/delay/noise history;
- “ready” remains a fail-closed conjunction of safety and reachability tolerances. It is not exact
  pose equality and not a weighted score whose positive terms may offset a failed safety conjunct.

This phase split resolves the apparent conflict between “use each selected action's own frame 0”
and “do not leak the future action”: before reveal, the selected/public identity is the completed
action; only at reveal does the new action become the selected/public identity.

### Current source conflict and launch boundary

At audited `main@6c3e47d`, `MotionCommand.joint_pos` substitutes
`robot.data.default_joint_pos` during hold, while the v2 contract requires the selected clip's
frame-0 joint pose. Joint/body velocity references are zeroed, but root/anchor velocity references
are not; body XY is also re-anchored to the live robot every command update rather than captured
once. Changing only `joint_pos` would therefore create a mixed reference. No runtime adapter is
claimed in v2.

The design validator passes while `launch-check` must fail:

```bash
python3 scripts/validate_phase1_frame0_wait_recovery_contract_v2.py \
  --contract configs/phase1_frame0_wait_recovery_contract_v2_20260715.json \
  --expected-contract-sha256 cc05d63fa4e4ffd9515f369f176ba032ca2a46d8996431a7b1e7d34e2b1bf28e \
  --mode design-check
```

A later, default-off source adapter must change pose, root/body velocity, immutable XY anchoring,
atomic reveal and carry-state observation together, then pass a strict full-scene Isaac receipt
and the vendor MuJoCo continuous gate. Until those bindings and numeric ready tolerances exist,
the v2 contract is `NO-LAUNCH`.
