#!/usr/bin/env python3
"""Offline replay of the deploy planner (hope_planner Stages 1-2) against the
2026-07-03 venue ball recordings — hit-plane crossing accuracy vs lead time.

This is the "step 2" planner validation: before wiring the real planner to a
live VRPN stream, replay recorded ball trajectories sample-by-sample through
the exact deploy code path (BallStateEstimator / BallKalmanEstimator ->
BallTrajectoryPredictor with venue-fit constants) and score the predicted
hitting-plane crossing (y, z, t) against the crossing the recorded ball
actually made.

Frames: recordings are in the venue table frame (origin = table center,
X = length, Z = up). The planner works in the HOPE frame (origin = near-side
left corner): HOPE = table + (+1.37, -0.7625, 0). Both table ends carry rally
traffic, so each take is replayed twice — pass "near" maps the table frame
directly, pass "far" rotates 180 deg about the table-center vertical so
approaches to the far end also become approaches to x_hit = 0.

Ground truth: the interpolated recorded crossing of x_HOPE = 0 (raw mocap,
~2-6 mm noise — negligible vs cm-level prediction error). A prediction is
matched to the next actual crossing within a time gate; unmatched predictions
are tallied separately (they are failures of a different kind, not silently
dropped).

Usage:
  BALLFIT_DATA_ROOT=/workspace/shared/ball_mocap_0703 \
    python3 scripts/planner_replay_eval.py [--takes ZHENGCHANG_000 ...]
    [--stride 10] [--max-lead 1.5] [--out-dir <dir>]

Outputs: printed summary + planner_replay.json (+ .png if matplotlib) under
<BALLFIT_DATA_ROOT>/analysis/planner_replay/ by default.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "hope_ws", "src", "hope_planner"))

from hope_planner.ball_kalman_estimator import BallKalmanEstimator
from hope_planner.ball_state_estimator import BallStateEstimator
from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams

HOPE_SHIFT = np.array([1.37, -0.7625, 0.0])
LEAD_BINS = [0.05, 0.15, 0.25, 0.35, 0.50, 0.80, 1.20]
MATCH_TIME_GATE = 0.25   # |predicted t_strike - actual t_cross| must beat this
GAP_RESET_S = 0.05       # tracking gap that forces an estimator reset
VX_GATE = -0.5           # only predict for balls approaching the plane
RACKET_RADIUS = 0.075    # context line in the summary


def data_root():
    root = os.environ.get("BALLFIT_DATA_ROOT", "/workspace/shared/ball_mocap_0703")
    ext = os.path.join(root, "analysis", "extracted")
    if not os.path.isdir(ext):
        raise SystemExit(f"no extracted takes under {ext!r}; set BALLFIT_DATA_ROOT")
    return root, ext


def to_hope(pos_t, mirror):
    """Table frame -> HOPE frame; mirror rotates 180 deg about table-center z."""
    p = pos_t.copy()
    if mirror:
        p[:, 0] = -p[:, 0]
        p[:, 1] = -p[:, 1]
    return p + HOPE_SHIFT


def actual_crossings(t, pos, present):
    """Interpolated recorded crossings of x = 0 with the ball moving -x.

    Returns list of dicts with t_cross, y, z.
    """
    out = []
    x = pos[:, 0]
    dt_med = np.median(np.diff(t))
    for i in range(len(t) - 1):
        if not (present[i] and present[i + 1]):
            continue
        if t[i + 1] - t[i] > 2.5 * dt_med:
            continue
        if x[i] > 0.0 >= x[i + 1]:
            frac = x[i] / (x[i] - x[i + 1])
            out.append(dict(
                t_cross=float(t[i] + frac * (t[i + 1] - t[i])),
                y=float(pos[i, 1] + frac * (pos[i + 1, 1] - pos[i, 1])),
                z=float(pos[i, 2] + frac * (pos[i + 1, 2] - pos[i, 2])),
                idx=i,
            ))
    return out


def actually_bounced(t, pos, present, t0, t1, table):
    """Did the recorded ball touch the table surface inside [t0, t1]?"""
    m = present & (t >= t0) & (t <= t1)
    if not np.any(m):
        return False
    p = pos[m]
    low = p[:, 2] < 0.05
    on = (p[:, 0] > -0.02) & (p[:, 0] < table.length + 0.02) & \
         (p[:, 1] < 0.02) & (p[:, 1] > -table.width - 0.02)
    return bool(np.any(low & on))


def replay_take(stem, npz, mirror, stride, max_lead, records, tallies):
    t = npz["t"].astype(float)
    present = npz["ball_present"].astype(bool)
    pos = to_hope(npz["ball_pos_t_m"].astype(float), mirror)
    crossings = actual_crossings(t, pos, present)
    tag = f"{stem}/{'far' if mirror else 'near'}"
    if not crossings:
        return

    cfg = PlannerConfig()
    cfg.max_predict_time = max_lead
    table = TableParams()
    predictor = BallTrajectoryPredictor(BallPhysics(), cfg, table)
    estimators = {
        "polyfit": BallStateEstimator(cfg),
        "ekf": BallKalmanEstimator(cfg),
    }
    cross_times = np.array([c["t_cross"] for c in crossings])

    last_t = None
    for i in range(len(t)):
        if not present[i]:
            continue
        if last_t is not None and t[i] - last_t > GAP_RESET_S:
            for est in estimators.values():
                est.reset()
        last_t = t[i]
        for est in estimators.values():
            est.push(t[i], pos[i])

        if i % stride:
            continue
        for name, est in estimators.items():
            if not est.ready:
                continue
            p_est, v_est, t_est = est.estimate()
            if v_est[0] >= VX_GATE or not (0.05 < p_est[0] < 3.0):
                continue
            tallies[name]["attempted"] += 1
            strike = predictor.predict(p_est, v_est, t_est)
            if not strike.valid:
                tallies[name]["invalid"] += 1
                continue
            # match to the next actual crossing, nearest to the predicted time
            ahead = cross_times[cross_times > t_est + 0.02]
            if len(ahead) == 0:
                tallies[name]["unmatched"] += 1
                continue
            j = int(np.argmin(np.abs(ahead - strike.t_strike)))
            c = crossings[int(np.searchsorted(cross_times, ahead[j]))]
            if abs(strike.t_strike - c["t_cross"]) > MATCH_TIME_GATE:
                tallies[name]["unmatched"] += 1
                continue
            lead = c["t_cross"] - t_est
            if lead <= LEAD_BINS[0] or lead > max_lead:
                continue
            records.append(dict(
                take=tag, estimator=name,
                lead=float(lead),
                dy=float(strike.p_ball[1] - c["y"]),
                dz=float(strike.p_ball[2] - c["z"]),
                dt=float(strike.t_strike - c["t_cross"]),
                pred_bounces=int(strike.num_bounces),
                actual_bounced=actually_bounced(
                    t, pos, present, t_est, c["t_cross"], table),
            ))
            tallies[name]["matched"] += 1


def summarize(records, tallies):
    lines = []
    lines.append(f"racket radius (tolerance context): {RACKET_RADIUS*100:.1f} cm")
    for name in ("polyfit", "ekf"):
        tl = tallies[name]
        lines.append(
            f"\n=== {name} ===  attempted {tl['attempted']}  matched {tl['matched']}"
            f"  unmatched {tl['unmatched']}  invalid {tl['invalid']}")
        recs = [r for r in records if r["estimator"] == name]
        if not recs:
            continue
        lines.append(f"{'lead (s)':>12} {'n':>6} {'planar med':>11} {'planar p90':>11}"
                     f" {'|dt| med':>9} {'bounce agree':>13}")
        for lo, hi in zip(LEAD_BINS[:-1], LEAD_BINS[1:]):
            sel = [r for r in recs if lo < r["lead"] <= hi]
            if not sel:
                continue
            planar = np.array([np.hypot(r["dy"], r["dz"]) for r in sel])
            dts = np.array([abs(r["dt"]) for r in sel])
            agree = np.mean([
                (r["pred_bounces"] > 0) == r["actual_bounced"] for r in sel])
            lines.append(
                f"{lo:>5.2f}-{hi:<6.2f} {len(sel):>6}"
                f" {np.median(planar)*100:>9.1f}cm {np.percentile(planar, 90)*100:>9.1f}cm"
                f" {np.median(dts)*1000:>6.1f}ms {agree*100:>12.0f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", nargs="*", default=None,
                    help="npz stems (default: every take under analysis/extracted)")
    ap.add_argument("--stride", type=int, default=10,
                    help="predict every Nth mocap frame (10 = 30 Hz at 300 Hz)")
    ap.add_argument("--max-lead", type=float, default=1.5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root, ext = data_root()
    stems = args.takes or sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(ext, "*.npz")))
    out_dir = args.out_dir or os.path.join(root, "analysis", "planner_replay")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    tallies = {n: dict(attempted=0, matched=0, unmatched=0, invalid=0)
               for n in ("polyfit", "ekf")}
    for stem in stems:
        npz = dict(np.load(os.path.join(ext, f"{stem}.npz")))
        for mirror in (False, True):
            replay_take(stem, npz, mirror, args.stride, args.max_lead,
                        records, tallies)
        n_take = sum(1 for r in records if r["take"].startswith(stem))
        print(f"[replay] {stem}: {n_take} matched predictions so far", flush=True)

    summary = summarize(records, tallies)
    print("\n" + summary)

    with open(os.path.join(out_dir, "planner_replay.json"), "w") as fh:
        json.dump(dict(records=records, tallies=tallies,
                       stride=args.stride, max_lead=args.max_lead,
                       takes=stems), fh, indent=1)
    print(f"\n[replay] wrote {out_dir}/planner_replay.json"
          f" ({len(records)} matched predictions)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, color in (("polyfit", "tab:orange"), ("ekf", "tab:blue")):
            recs = [r for r in records if r["estimator"] == name]
            if not recs:
                continue
            ax.scatter([r["lead"] for r in recs],
                       [np.hypot(r["dy"], r["dz"]) * 100 for r in recs],
                       s=4, alpha=0.25, color=color, label=name)
        ax.axhline(RACKET_RADIUS * 100, color="k", ls="--", lw=1,
                   label="racket radius 7.5 cm")
        ax.set_xlabel("prediction lead time (s)")
        ax.set_ylabel("planar crossing error |dy,dz| (cm)")
        ax.set_ylim(0, 40)
        ax.legend()
        ax.set_title("Deploy planner hit-plane prediction error vs lead "
                     "(venue 0703 replay)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "planner_replay.png"), dpi=120)
        print(f"[replay] wrote {out_dir}/planner_replay.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
