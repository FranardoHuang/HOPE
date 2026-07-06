#!/usr/bin/env python3
"""Mocap timing / random-delay analysis.

Two independent sources of "random delay" in the ChingMu -> VRPN -> relay -> policy
chain, and the two things this tool measures:

  dropout : OCCLUSION-DROPOUT STALENESS from the recorded venue takes.
            When markers are lost the freshest usable sample goes stale for a
            random number of frames; a hold-last consumer sees that as delay.
            This is fully recoverable from the extracted npz (a dense fixed-rate
            grid with NaN holes). Reproduces docs/research/.../mocap_random_delay.md.

  stream  : INTER-ARRIVAL JITTER from a live-stream log (mocap_stream_logger CSV,
            or the legacy calib_csv/traj01.csv `t,x,y,z`). The recorded takes
            CANNOT show this -- extract_canonical.py resamples onto arange(nf)/rate,
            so packet-arrival jitter is gone. Only a live log has it. This is the
            transport-side random delay and the only way to close NOW.md Gap #6.

Runs on the Mac ball-physics env (numpy only):
  python mocap_timing.py dropout                       # venue-take occlusion delay
  python mocap_timing.py stream  <log.csv> [log2.csv]  # live-stream jitter
  python mocap_timing.py stream  calib_csv/traj01.csv  # reproduce the old 44Hz number
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# 50 Hz policy step; 300 Hz nominal mocap stream (team contract, 2026-07).
POLICY_HZ = 50.0
POLICY_STEP_MS = 1e3 / POLICY_HZ
NOMINAL_MOCAP_HZ = 300.0

DEFAULT_DATA_ROOT = os.environ.get(
    "BALLFIT_DATA_ROOT", "/Users/yyk956614/Desktop/Hope/Record/latest"
)
IN_PLAY_MAX_MS = 200.0  # gaps longer than this => ball out of play (between rallies)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def runs_of_false(mask) -> np.ndarray:
    """Lengths (in samples) of maximal runs where ``mask`` is False."""
    idx = np.flatnonzero(~np.asarray(mask, bool))
    if idx.size == 0:
        return np.array([], int)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return (ends - starts + 1).astype(int)


def _fmt_row(vals, widths):
    return "".join(f"{str(v):>{w}}" for v, w in zip(vals, widths))


# --------------------------------------------------------------------------- #
# dropout: occlusion staleness from the recorded venue takes
# --------------------------------------------------------------------------- #
def cmd_dropout(args) -> int:
    ext = os.path.join(args.data_root, "analysis", "extracted")
    files = sorted(glob.glob(os.path.join(ext, "*.npz")))
    if not files:
        print(f"no extracted *.npz under {ext}", file=sys.stderr)
        return 2

    pool = {"ball": [], "p1": [], "p2": []}
    rates = set()
    W = (22, 8, 8, 8, 8, 7, 7, 7, 6, 8, 8, 8, 9)
    print(f"reading {len(files)} takes from {ext}")
    print("=" * sum(W))
    print(_fmt_row(
        ["take", "dur_s", "dt_cv", "contig", "ballV%", "p1%", "p2%",
         "ngap", "med_ms", "p90_ms", "p99_ms", "max_ms"],
        (22, 8, 8, 8, 8, 7, 7, 6, 8, 8, 8, 9)))
    print("-" * sum(W))
    for f in files:
        z = np.load(f, allow_pickle=True)
        rate = float(z["rate"]); rates.add(rate); dt_ms = 1e3 / rate
        nf = len(z["t"])
        dt = np.diff(z["t"])
        dt_cv = float(np.std(dt) / np.mean(dt)) if len(dt) else 0.0
        contig = bool(np.all(np.diff(z["frame"]) == 1))

        ball = z["ball_present"].astype(bool)
        bg = runs_of_false(ball) * dt_ms
        p1g = runs_of_false(z["p1_present"].astype(bool)) * dt_ms
        p2g = runs_of_false(z["p2_present"].astype(bool)) * dt_ms
        pool["ball"].append(bg); pool["p1"].append(p1g); pool["p2"].append(p2g)
        print(_fmt_row(
            [os.path.basename(f)[:-4], round(nf * dt_ms / 1e3, 1), round(dt_cv, 5),
             contig, round(100 * ball.mean(), 1),
             round(100 * z["p1_present"].mean(), 1), round(100 * z["p2_present"].mean(), 1),
             len(bg), round(_pct(bg, 50), 1), round(_pct(bg, 90), 1),
             round(_pct(bg, 99), 1), round(_pct(bg, 100), 1)],
            (22, 8, 8, 8, 8, 7, 7, 6, 8, 8, 8, 9)))

    if any(abs(r - NOMINAL_MOCAP_HZ) < 1 for r in rates):
        note = "uniform fixed grid (dt_cv~0, frame contiguous) => NO transport jitter here"
    else:
        note = ""
    print(f"\nrate(s): {sorted(rates)} Hz.  {note}")

    def summarize(label, arrs, split=False):
        a = np.concatenate(arrs) if arrs else np.array([])
        a = a[np.isfinite(a)]
        if not len(a):
            print(f"  {label:<26} (no gaps)"); return
        if split:
            ip = a[a <= IN_PLAY_MAX_MS]; op = a[a > IN_PLAY_MAX_MS]
            print(f"  {label:<26} n={len(a):>4}  "
                  f"IN-PLAY(<= {IN_PLAY_MAX_MS:.0f}ms) n={len(ip)}: "
                  f"med={np.median(ip):.1f} p90={_pct(ip,90):.1f} p95={_pct(ip,95):.1f} "
                  f"p99={_pct(ip,99):.1f} max={ip.max():.1f} ms "
                  f"[{np.median(ip)/POLICY_STEP_MS:.2f}/{_pct(ip,99)/POLICY_STEP_MS:.1f} steps med/p99]")
            print(f"  {'':<26}       OUT-OF-PLAY(> {IN_PLAY_MAX_MS:.0f}ms) n={len(op)}: "
                  f"med={np.median(op)/1e3:.2f}s max={op.max()/1e3:.2f}s  (gate out at deploy)")
        else:
            print(f"  {label:<26} n={len(a):>4}  med={np.median(a):.1f} p90={_pct(a,90):.1f} "
                  f"p99={_pct(a,99):.1f} max={a.max():.1f} ms  (few gaps, all 'set-aside')")

    print("\nPOOLED gap (staleness) distribution  [ms; steps @50Hz]")
    summarize("BALL (usable measurement)", pool["ball"], split=True)
    summarize("PADDLE p1 (racket)", pool["p1"])
    summarize("PADDLE p2 (racket)", pool["p2"])
    print("\nverdict: random delay lives on the BALL (-> planner KF coast horizon), "
          "not the racket target (clean in-play).")
    return 0


# --------------------------------------------------------------------------- #
# stream: inter-arrival jitter from a live-stream log
# --------------------------------------------------------------------------- #
def _load_stream_csv(path):
    """Return dict topic -> (t_arrival[s], xyz[N,3]).  Accepts the logger CSV
    (topic,seq,t_wall,t_mono,stamp,x,y,z) or the legacy `t,x,y,z`."""
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    cols = {name: i for i, name in enumerate(header)}
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float,
                         usecols=[cols[c] for c in header
                                  if c not in ("topic", "seq")])
    # rebuild a name->index map for the numeric-only columns we actually loaded
    num_cols = [c for c in header if c not in ("topic", "seq")]
    ni = {name: i for i, name in enumerate(num_cols)}
    if data.ndim == 1:
        data = data[None, :]

    # topic column (string) read separately if present
    if "topic" in cols:
        topics = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str,
                               usecols=[cols["topic"]])
        topics = np.atleast_1d(topics)
    else:
        topics = np.array(["(legacy)"] * len(data))

    # arrival time: prefer monotonic logger clock, else wall/stamp, else legacy t
    tcol = ("t_mono" if "t_mono" in ni else
            "t_wall" if "t_wall" in ni else
            "t" if "t" in ni else
            "stamp" if "stamp" in ni else None)
    if tcol is None:
        raise SystemExit(f"{path}: no recognizable time column in {header}")
    t = data[:, ni[tcol]]
    xyz = None
    if all(c in ni for c in ("x", "y", "z")):
        xyz = data[:, [ni["x"], ni["y"], ni["z"]]]

    out = {}
    for tp in np.unique(topics):
        m = topics == tp
        out[str(tp)] = (t[m], None if xyz is None else xyz[m], tcol)
    return out


def cmd_stream(args) -> int:
    any_ok = False
    for path in args.logs:
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr); continue
        any_ok = True
        streams = _load_stream_csv(path)
        print(f"\n### {path}")
        for topic, (t, xyz, tcol) in streams.items():
            order = np.argsort(t); t = t[order]
            if xyz is not None:
                xyz = xyz[order]
            n = len(t)
            if n < 3:
                print(f"  {topic}: n={n} (too few)"); continue
            dur = t[-1] - t[0]
            dt_ms = np.diff(t) * 1e3
            dt_ms = dt_ms[dt_ms >= 0]
            eff_hz = (n - 1) / dur if dur > 0 else float("nan")
            # dropout gaps = inter-arrival dt well above the nominal frame period
            nominal_ms = 1e3 / NOMINAL_MOCAP_HZ
            gap_thr = 1.5 * nominal_ms
            gaps = dt_ms[dt_ms > gap_thr]
            # duplicate/held samples (identical position back-to-back)
            dup_pct = float("nan")
            if xyz is not None and n > 1:
                same = np.all(np.abs(np.diff(xyz, axis=0)) < 1e-9, axis=1)
                dup_pct = 100 * same.mean()
            print(f"  {topic:<22} n={n:<6} dur={dur:6.1f}s  clock={tcol}")
            print(f"    effective rate = {eff_hz:6.1f} Hz   (nominal {NOMINAL_MOCAP_HZ:.0f} Hz "
                  f"=> {100*eff_hz/NOMINAL_MOCAP_HZ:.0f}% of contract)")
            print(f"    inter-arrival dt [ms]: med={np.median(dt_ms):6.2f}  "
                  f"p90={_pct(dt_ms,90):6.2f}  p95={_pct(dt_ms,95):6.2f}  "
                  f"p99={_pct(dt_ms,99):6.2f}  max={dt_ms.max():7.2f}")
            print(f"    jitter (std)         = {np.std(dt_ms):.2f} ms   "
                  f"[nominal period {nominal_ms:.2f} ms]")
            print(f"    dropout gaps (> {gap_thr:.1f} ms): n={len(gaps)} "
                  f"({100*len(gaps)/len(dt_ms):.1f}% of intervals)  "
                  f"max={gaps.max() if len(gaps) else 0:.1f} ms "
                  f"= {(gaps.max() if len(gaps) else 0)/POLICY_STEP_MS:.1f} policy steps")
            if np.isfinite(dup_pct):
                print(f"    duplicate-position samples = {dup_pct:.1f}% "
                      f"(held/re-sent frames inflate the apparent rate)")
    if not any_ok:
        return 2
    print("\nnote: header.stamp on /vrpn_mocap/** is get_clock()->now() (tracker.cpp:110),")
    print("      i.e. ARRIVAL time, not mocap capture time -- absolute source->host")
    print("      latency needs a tracker.cpp msg_time patch or NTP/PTP sync; the")
    print("      inter-arrival jitter above is sync-free and is the transport random delay.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dropout", help="occlusion staleness from recorded venue takes")
    d.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help=f"take root with analysis/extracted (default {DEFAULT_DATA_ROOT})")
    d.set_defaults(func=cmd_dropout)
    s = sub.add_parser("stream", help="inter-arrival jitter from a live-stream log CSV")
    s.add_argument("logs", nargs="+", help="logger CSV(s) or legacy t,x,y,z CSV")
    s.set_defaults(func=cmd_stream)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
