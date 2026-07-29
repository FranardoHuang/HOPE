# optitrack/ — OptiTrack (Motive) ball-take pipeline

Turns a Motive C3D export of a coated-ball session into the canonical take npz
that `ballcore` consumes, plus a stage-0-style QA + content report. Built and
validated on `Take 2026-07-21` (360 Hz, 21,550 columns, truncated export);
designed to stream takes of ANY size (a 5-min take ≈ 36 GB, a 25-min take
≈ 180 GB — memory use is O(columns), time is linear in file size).

## Which extractor do I want?

| export contains | use | notes |
| --- | --- | --- |
| ball only (unlabeled columns) | `ot_scan.py` → `ot_extract.py` | take 0721 |
| ball + rackets + table, all labeled assets | **`ot_extract_full.py`** | 2026-07-30 session |
| Avatar-Pro venue export (mm, `b_*` labels, rigid ball) | `../extract_canonical.py` | 2026-07-03 venue |

`ot_extract_full.py` is a single streaming pass that writes the full canonical
schema (ball centre, both racket poses + face normals, exact table frame from
the 4 corner markers, net posts as a QA cross-check). It exists because the
2026-07-30 session fell between the other two readers: the venue extractor is
mm-hardcoded, expects `b_PPP1_*` labels, loads whole files through
`c3d.Reader.read_frames()` (hopeless at 2.3 GB / 3255 columns) and — the real
blocker — runs a rigid-body Kabsch solve on the BALL, which on a coated ball
fabricates quaternions and a centre offset; `ot_extract.py` streams correctly
but has no table or racket channel at all.

```bash
python ot_extract_full.py <take.c3d> <out_dir> [--center sphere|centroid]
                          [--max-fill-gap 3] [--fill-any-gap]
```

Ball centre is a FIXED-RADIUS (20 mm) sphere solve through the visible wandering
points when ≥ 4 are present, centroid at 3. On the 07-30 session that leaves a
2.9 mm per-frame residual — the points really are on the ball surface, so the
solve is well posed even though the ball has no rigid template.

### Two things this format does that will bite you

- **Empty export frames.** ~11.5% of frames on the 07-30 session are written
  blank: ball, both rackets AND the bolted-down table markers vanish together
  (99.3% co-occurrence), about every 11th frame with a drifting phase. That is
  the exporter, not occlusion — real sample rate is ~317 Hz on an unchanged
  360 Hz grid. Holes ≤ 3 frames are repaired by local quadratic fit through 6
  real samples per side and MARKED (`ball_interpolated`,
  `frame_blank_in_export`). Without the repair, tracked runs never reach the
  15-frame minimum and arc extraction returns **zero** arcs.
- **The venue quality gates do not transfer.** A coated-ball centre is a
  per-frame sphere solve, not a solved rigid body, so stage-1 window-fit RMS is
  ~11 mm against the venue's ~5 mm. `stage2_fits.py`/`predict_check.py` now take
  `--rms-gate-bounce/strike/paddle` (venue defaults preserved); at the venue
  values 100% of 07-30 bounces are rejected. Every gated fit also reports a
  strict-quartile sensitivity so a loose gate cannot silently carry a result.

### Running the rest of the pipeline on a non-venue session

`ballcore` resolves its take registry from the DATA ROOT, so no stage needs
editing. Drop `analysis/takes.json` next to `analysis/extracted/`:

```json
{"takes": {"tui": "Tui", "xuan": "xuan"},
 "roles": {"lowspin": ["tui", "xuan"], "spin": []}}
```

Then `qa_stage0.py → stage1_segments.py → stage2_fits.py → predict_check.py`
run unchanged. Stages that need a ball orientation channel detect its absence
(`ballcore.has_spin_channel`) and degrade loudly rather than fitting NaNs.

### Contact detection

`ballcore.detect_contacts` infers strikes from pointwise `d(vel)/dt`; at 360 Hz
with a ~6 mm ball-centre noise floor its 1 m/s threshold is ~2.5σ, and on the
07-30 session it fired 1035–1860 times per take against 41–95 real table
bounces, cutting every flight into 0.09 s fragments. When racket poses exist,
`extract_arcs` automatically switches to `detect_contacts_racket`, which
proposes a strike at a ball–racket distance minimum and confirms it with a
two-sided velocity jump fitted on ~12 real samples per side (median arc
duration 0.094 s → 0.24–0.33 s, median parabola-g 3.5–8.6 → 9.4–10.2,
ballistic arcs 48 → 274).

## Why the venue extractor doesn't apply here

| | venue Avatar-Pro (extract_canonical.py) | Motive export (this pipeline) |
|---|---|---|
| ball | 15 SOLVED rigid-body points, stable template | 8 asset markers, RIGID constellation on a ~18.6 mm sphere, but **relabelled between frames** so column identity is meaningless |
| orientation / spin | Kabsch quats, spin usable < 75 rev/s | recoverable, but needs a permutation-robust CHAINED solve (`ball_orientation.py`); usable < ~23 rev/s at 360 Hz |
| table | 6 labeled markers → exact frame | usually none → surface auto-calibrated from bounce minima |
| units | mm | often meters, POINT:UNITS missing → auto-detected |
| labels | all columns labeled | thousands of `Unlabeled_NNNN` transient columns (one per re-acquisition) |

## Run order

```bash
PY=<python with numpy/scipy/matplotlib/c3d>       # c3d needed by scan/extract only
$PY ot_scan.py    <take.c3d> <out_dir>            # pass 1: column stats, ball asset,
                                                  #   static refs, truncation report
$PY ot_extract.py <take.c3d> <out_dir>            # pass 2: centroid + unlabeled gap-fill
                                                  #   -> <take>.npz (ballcore keys)
$PY ot_analyze.py <out_dir>/<take>.npz            # QA + bounces/rallies/strokes + plots
```

Each pass streams the whole file once (~2.5 min per 20 GB on a USB3 exFAT
drive). Both scan and extract survive transient drive I/O errors by
reopen-and-retry (a memmap dies with SIGBUS when an external drive hiccups —
learned the hard way).

## Key design decisions (validated on take 0721)

- **Ball center = centroid of visible asset markers** (≥3). Fixed-radius
  sphere fitting is slightly cleaner per frame (4.5 vs 5.0 mm arc RMS) but
  needs ≥4 points and fragments coverage (97 → 14 ballistic arcs). The
  centroid also matches the venue convention (centroid = rigid virtual point).
- **Gap fill from unlabeled columns**: when the asset solve drops, coast a
  ballistic prediction (≤100 ms) and attach unlabeled clusters within 45 mm;
  forward + backward sweeps. On 0721 this lifted coverage 64.6% → 77.7%
  (86.5% after ≤5-frame interpolation) without changing arc RMS.
- **Speed-spike guard**: fill frames implying |v| > 30 m/s are rejected
  (mis-associations otherwise create absurd velocity outliers).
- **Surface height without table markers**: mode of bounce local-minima minus
  R_BALL. On 0721 the calibration ground plane was ON the table (surface_z ≈ 0,
  origin ≈ a table corner, x along the table length).
- **Stroke count = vx-reversal**, not `detect_contacts` hits: at ~5 mm
  surface-wander noise the hit detector fires ~20× too often (1900 vs ~100).

## What this data can and cannot support

- OK: arc inventory, ballistic-arc noise floor, bounce/landing distributions,
  rally/stroke statistics, trajectory-prediction gates, table-plane checks.
- OK **on a marker ball** (2026-07-30 session): spin, and therefore k_m / Magnus
  and the tangential contact blocks — but ONLY via the chained orientation solve
  in `ball_orientation.py`, and only below the ~23 rev/s aliasing limit. Take
  0721's fully-coated ball genuinely has no orientation; check which you have
  with `ball_rigidity_probe.py` before assuming either way.
- CAUTION: a spin-blind joint kd+g fit reads |g| a few % high on topspin-heavy
  play (measured 2026-07-30: 9.70 → 10.45 m/s² across arc-speed terciles) —
  that is Magnus absorption, not a data defect; the venue ±0.05 gravity gate
  does not apply to a spin-blind fit.
- **Do NOT judge ball rigidity by pairwise-distance scatter.** By column index it
  is destroyed by relabelling; sorted (permutation-invariant) it is destroyed by
  rank swapping when the distance spectrum is dense relative to the noise. Both
  read "non-rigid" on a ball that is provably rigid. Use consecutive-frame
  alignment against a wandering-points control instead.
- Restitution from ±11 ms window velocities is unreliable at this noise level;
  use ballcore cross-gap arc pairing if e is needed.

## Truncation check (do this before waiting on a long export)

`ot_scan.py` prints `[TRUNCATED: ...]` when the data section holds fewer frames
than the header claims (bytes ÷ (4·(4·npts + analog_floats)) vs header count).
Take 0721 was cut at 10.4% (Motive export interrupted). Frame counts land on a
clean frame boundary, so partial files remain fully usable up to the cut.

## Capture recommendations for the next session

1. ~5 min of pure play per take (≈ 36 GB C3D) — big files export slowly and
   truncate silently; several short takes beat one giant one.
2. Stick 4 markers on the table corners (minutes of work) — exact table frame
   instead of bounce-cloud inference.
3. If spin will ever be needed: export the rigid-body SOLVED data too, or use
   a ball with discrete markers; the coated ball alone cannot give orientation.
4. After export, run `ot_scan.py` immediately — it verifies completeness in
   seconds-per-GB and catches truncation before the session ends.
