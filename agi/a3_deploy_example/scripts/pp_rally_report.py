#!/usr/bin/env python3
"""Deep-dive analyzer for the Gate-3 rally rehearsal (110-D hitter_pure obs CSV).

Reads the runner's --obs-csv capture (and optionally the conductor's report JSON)
and prints a per-swing table + health checks, so a failed/odd rally can be
debugged LOCALLY from numbers instead of re-running with a viewer.

110-D obs layout (pp_obs_builder.hpp build_obs_110 / LogFirstTick blks110):
  [0:3]    base_ang_vel        [3:34]   joint_pos_rel     [34:65]  joint_vel
  [65:96]  last_action         [96:99]  projected_gravity [99:101] base_forward_xy
  [101:103] base_target_delta_xy (world station - base)
  [103:106] racket_target_rel_base (world)   [106:109] racket_target_vel_w
  [109]    time_to_strike

Swing segmentation: at idle the runner pins tts at the engaged clip's windup max
(fh ~1.30 s / bh ~0.87 s); during a swing tts DECREASES every tick down to the
clip-end clamp. A swing = a maximal run of strictly-decreasing tts spanning >0.5 s.

Usage: python3 pp_rally_report.py /tmp/pp_obs.csv [/tmp/pp_rally_report.json]
"""
import csv
import json
import math
import sys

B = {"ang_vel": (0, 3), "jpos": (3, 34), "jvel": (34, 65), "act": (65, 96),
     "grav": (96, 99), "fwd": (99, 101), "dstation": (101, 103),
     "rkt_rel": (103, 106), "rkt_vel": (106, 109), "tts": 109}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pp_obs.csv"
    rows = []
    with open(path) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        n_obs = sum(1 for c in hdr if c.startswith("obs_"))
        if n_obs != 110:
            print(f"FAIL: obs CSV has {n_obs} obs columns, not the 110-D contract")
            sys.exit(1)
        o0 = hdr.index("obs_0")
        for r in rd:
            if len(r) < o0 + 110:
                continue
            rows.append({
                "tick": int(r[0]), "ts": float(r[1]), "mode": r[2],
                "sync_miss": int(r[hdr.index("sync_miss")]),
                "o": [float(v) for v in r[o0:o0 + 110]],
            })
    if not rows:
        print("FAIL: empty obs CSV")
        sys.exit(1)

    def seg(o, k):
        lo, hi = B[k]
        return o[lo:hi]

    # ---- global health ----
    bad = ok = 0
    checks = []
    nan_ticks = sum(1 for r in rows if any(math.isnan(v) or math.isinf(v) for v in r["o"]))
    checks.append(("no NaN/Inf in obs", nan_ticks == 0, f"{nan_ticks} ticks affected"))
    sm = rows[-1]["sync_miss"]
    checks.append(("sync_miss == 0", sm == 0, f"final sync_miss={sm}"))
    gz_bad = sum(1 for r in rows if seg(r["o"], "grav")[2] > -0.7)
    checks.append(("upright (grav_z<-0.7) except transients", gz_bad < 0.02 * len(rows),
                   f"{gz_bad}/{len(rows)} ticks tilted"))
    dmax = max(math.hypot(*seg(r["o"], "dstation")) for r in rows)
    checks.append(("|station delta| <= 0.85 (engage-gate bound)", dmax <= 0.85,
                   f"max |dstation|={dmax:.3f} m"))
    amax = max(max(abs(v) for v in seg(r["o"], "act")) for r in rows)
    checks.append(("|last_action| < 12 (sane policy output)", amax < 12.0,
                   f"max |action|={amax:.2f}"))
    jvmax = max(max(abs(v) for v in seg(r["o"], "jvel")) for r in rows)
    checks.append(("|joint_vel| < 25 rad/s", jvmax < 25.0, f"max |joint_vel|={jvmax:.1f}"))

    # ---- swing segmentation from the tts channel ----
    tts = [r["o"][B["tts"]] for r in rows]
    swings = []
    i, n = 1, len(rows)
    while i < n:
        if tts[i] < tts[i - 1] - 1e-6:          # decreasing -> in a swing
            j = i
            while j + 1 < n and tts[j + 1] < tts[j] - 1e-6:
                j += 1
            if rows[j]["ts"] - rows[i - 1]["ts"] > 0.5:
                swings.append((i - 1, j))
            i = j + 1
        else:
            i += 1

    print(f"== {path}: {len(rows)} ticks, {len(swings)} swings segmented ==")
    print("swing  t_start  tts0   side   strike_rel_base(xyz)      tgt_vel(xyz)"
          "        |dstation|@strike  peak|angvel|")
    for k, (a, b) in enumerate(swings, 1):
        tts0 = tts[a]
        side = "fh" if tts0 > 1.1 else "bh"     # windup max: fh 1.30 / bh 0.87
        # strike tick = tts closest to 0 inside the swing
        st = min(range(a, b + 1), key=lambda t: abs(tts[t]))
        o = rows[st]["o"]
        rr = seg(o, "rkt_rel")
        rv = seg(o, "rkt_vel")
        ds = math.hypot(*seg(o, "dstation"))
        pav = max(math.sqrt(sum(v * v for v in seg(rows[t]["o"], "ang_vel")))
                  for t in range(a, b + 1))
        print(f"  {k:2d}  {rows[a]['ts']:7.1f}  {tts0:4.2f}   {side}   "
              f"({rr[0]:+.3f},{rr[1]:+.3f},{rr[2]:+.3f})  "
              f"({rv[0]:+.2f},{rv[1]:+.2f},{rv[2]:+.2f})   {ds:6.3f}            {pav:5.2f}")
        # per-swing checks: strike should happen AT the fixed plane, station kept
        checks.append((f"swing {k}: strike x_rel_base in [0.45,0.95]",
                       0.45 <= rr[0] <= 0.95, f"{rr[0]:+.3f}"))
        checks.append((f"swing {k}: |station delta| at strike < 0.45",
                       ds < 0.45, f"{ds:.3f}"))

    print("\n== checks ==")
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}  ({detail})")
        ok, bad = ok + passed, bad + (not passed)

    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            rep = json.load(f)
        print(f"\n== conductor report: serves={rep['serves']} returned={rep['returned']} "
              f"falls={rep['falls']} drift={rep['station_drift_m']}m "
              f"-> {'PASS' if rep['pass'] else 'FAIL'} ==")
    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {ok} checks passed, {bad} failed")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
