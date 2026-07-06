# Mocap Random Delay — measured on the latest venue dataset (2026-07-05)

**Question (yikang):** our mocap has some random delay — quantify it on the latest dataset;
measure it if no study exists.

**TL;DR.** There are two independent random-delay sources; only one is recoverable from the
recorded takes.

1. **Occlusion-dropout staleness (MEASURED here).** When markers are lost, a hold-last consumer
   sees the freshest sample age by a random amount. On the 9 latest venue takes this lives almost
   entirely on the **ball**, not the racket: in-play ball dropout is **median 10 ms, p90 47 ms,
   p99 166 ms, max 180 ms (~0.5 / 2.4 / 8.3 / 9 policy steps @50 Hz)**; the racket (paddles p1/p2)
   is essentially clean in-play (90–100 % tracked). This is a **ball-estimator (KF) coast-horizon**
   number, not a racket-target-delay number.
2. **Transport / packet-arrival jitter (NOT recoverable here — needs a live log).** The recorded
   takes are resampled onto a fixed 300 Hz grid, so arrival jitter is gone at export. The only
   real transport-timestamped sample we have is a *legacy* relay log showing an effective
   **~44 Hz (median inter-arrival 22.6 ms, p90 41.6 ms)** — an order of magnitude worse than the
   occlusion side, and unmeasured on the current rig. Closing this is **NOW.md Gap #6**; the
   `mocap_stream_logger` venue tool below captures it.

---

## Why only one source is in the data

- **Fixed-grid export.** `hope_training/ball_physics_fit/extract_canonical.py:192` builds
  `t = np.arange(nf) / rate` and `:254` `frame = np.arange(a, b+1)`. Every take reads back with
  `dt_cv ≈ 0` and contiguous frames (verified below) — a dense 300 Hz grid with **NaN holes** on
  occlusion. Motive already resampled onto this grid, so packet-arrival jitter is thrown away.
  What survives is *which frames have a usable sample* → occlusion dropout.
- **Arrival-time stamps.** The vendored vrpn client stamps `header.stamp = get_clock()->now()`
  (`hope_ws/src/vrpn_mocap/src/tracker.cpp:110`), i.e. the **arrival** time on the deploy host, not
  the mocap capture time. So even a live log gives sync-free inter-arrival **jitter** (the random
  delay), but **not** absolute source→host latency without a client patch or NTP/PTP sync (see
  "Closing Gap #6").

## Method

`hope_training/ball_physics_fit/mocap_timing.py` (numpy-only, runs in the `pingpong` env):

- `dropout` — per take, gap = a maximal run of `~ball_present` (usable sample absent) on the fixed
  grid, length × (1000/rate) ms. Split at **200 ms**: ≤200 ms = in-play occlusion (mid-rally
  blips), >200 ms = ball out of play (between rallies / on the floor / picked up) → gated at
  deploy, not modeled as sensor delay. Same for paddles via `p1_present`/`p2_present`.
- `stream` — per object, inter-arrival `dt` of a live-log CSV (or the legacy `t,x,y,z`): effective
  rate, jitter percentiles, dropout gaps (`dt > 1.5×` nominal period), duplicate-position (held)
  fraction. Validated: it reproduces the legacy log's **median 22.59 ms / p90 41.60 ms** exactly.

## Results — occlusion dropout (9 takes, ~724 s, ChingMu 300 Hz)

| take | dur s | dt_cv | contig | ballV % | p1 % | p2 % | n gap | med ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GAO_QIU (高球) | 74.9 | 0.0 | ✓ | 89.0 | 100 | 100 | 28 | 8.3 | 662 | 2779 | 3080 |
| TANTIAO (弹跳) | 51.1 | 0.0 | ✓ | 85.7 | 94.4 | 100 | 44 | 10.0 | 35.7 | 3252 | 4747 |
| ZHENGCHANG (正常) | 114.4 | 0.0 | ✓ | 78.0 | 94.0 | 96.0 | 75 | 10.0 | 343 | 6599 | 8560 |
| buzhuandianqiu (颠球不转) | 23.1 | 0.0 | ✓ | 99.5 | 100 | 97.8 | 4 | 8.3 | 61.3 | 81.1 | 83.3 |
| cexuan (侧旋) | 112.0 | 0.0 | ✓ | 66.5 | 100 | 93.8 | 78 | 16.7 | 2450 | 4503 | 6053 |
| dainqiu (颠球转) | 52.0 | 0.0 | ✓ | 75.2 | 98.2 | 100 | 13 | 30.0 | 4035 | 5719 | 5850 |
| shangxuan (上旋) | 72.9 | 0.0 | ✓ | 54.3 | 90.1 | 98.7 | 56 | 35.0 | 1567 | 6921 | 8373 |
| xiaxuan (下旋) | 50.6 | 0.0 | ✓ | 70.0 | 100 | 100 | 28 | 11.7 | 712 | 8914 | 11600 |
| 快速 (fast) | 173.1 | 0.0 | ✓ | 62.4 | 96.5 | 88.0 | 96 | 13.3 | 2765 | 8467 | 12587 |

`dt_cv = 0.0` and contiguous frames on every take ⇒ the grid is a perfectly uniform 300 Hz —
**no transport jitter is present in these files**, by construction.

**Pooled ball dropout (n=422 events):**

| class | n | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| **in-play (≤200 ms)** | 339 | **10.0 ms** | 47.3 ms | 83.3 ms | 166.2 ms | 180.0 ms |
| in-play, in 50 Hz steps | | 0.50 | 2.37 | 4.17 | 8.31 | 9.00 |
| out-of-play (>200 ms) | 83 | 1.27 s | — | — | — | 12.59 s |

In-play CDF: **56 % ≤10 ms · 78 % ≤20 ms (1 step) · 91 % ≤50 ms · 96 % ≤100 ms.** 80 % of dropout
*events* are the short in-play blips; the 83 multi-second gaps are between-rally "ball not in play".

**Paddles / racket (actor-target analog):** p1 tracked 90–100 %, p2 88–100 %; only 9 / 15 gaps total
and every one is a multi-second "paddle set aside" period, not an in-play loss. **In-play racket
tracking has no meaningful random dropout.**

## Deployment implications

- **The random delay is on the BALL, and it feeds the planner's ball estimator, not the policy.**
  The 175-D policy contract consumes a *racket target* (clean); the ball KF (the shadow AR(1)-EKF
  in main) is what must tolerate stale ball samples. Calibration target: **coast/hold-last up to
  ~180 ms (~9 frames @300 Hz / ~9 steps @50 Hz)** before declaring the estimate stale, and **gate
  out gaps >200 ms** as out-of-play rather than coasting through them.
- **A1 target-delay flags are mostly modeling the wrong surface.** `racket.target_delay_steps`,
  `midswing_resample_prob`, and the A1v2 target-dropout flags act on the racket target — which is
  clean in-play. Their fixed `delay_steps=2` (40 ms) is a conservative *transport* bound (franco rig
  ground truth: <10 ms transmission, ≤20 ms e2e), a different quantity from occlusion dropout. Keep
  the transport bound on the racket; put the occlusion-dropout distribution above into the **ball
  KF**, not the racket-target injector.

## Closing Gap #6 — the transport-jitter piece

The recorded takes can never show it. Log a **live stream** at the next venue session:

```bash
# in the ROS distrobox, after colcon build + source install/setup.bash
ros2 run hope_bringup mocap_stream_logger --ros-args \
    -p out_dir:=~/mocap_timing_logs -p duration_s:=120.0
# analyse (Mac pingpong env):
python3 hope_training/ball_physics_fit/mocap_timing.py stream ~/mocap_timing_logs/*.csv
```

`mocap_stream_logger` timestamps every incoming sample with its own monotonic clock and subscribes
as far upstream as possible (discovers raw `/vrpn_mocap/<sender>/pose*`; also logs relay outputs).
It writes a `stream`-ready CSV plus an on-exit per-topic summary (effective rate, dt percentiles,
jitter, dropout gaps) so the operator gets an instant readout at the venue.

**Optional — absolute source→host latency.** As-is the client overwrites source time with `now()`.
To recover true transport latency, patch `hope_ws/src/vrpn_mocap/src/tracker.cpp` to stamp with the
VRPN callback `t.msg_time` instead of `get_clock()->now()` (behind a `use_vrpn_stamp` param), and/or
NTP/PTP-sync the ChingMu host and the deploy host. Then `recv − stamp` in the logger CSV becomes the
latency. Until then, only inter-arrival **jitter** is trustworthy (it is sync-free).

## Reproduce

```bash
BALLFIT_DATA_ROOT=/path/to/Record/latest \
  python3 hope_training/ball_physics_fit/mocap_timing.py dropout
python3 hope_training/ball_physics_fit/mocap_timing.py stream calib_csv/traj01.csv  # legacy 44 Hz
```
