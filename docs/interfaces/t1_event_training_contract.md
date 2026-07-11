# T1 Post-Strike Event Training Contract

Status: **core training implementation only; not launch-ready** (2026-07-11).

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
