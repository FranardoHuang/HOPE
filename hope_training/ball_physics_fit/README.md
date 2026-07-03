# ball_physics_fit — venue ball-physics fitting pipeline

Fits the flight + contact model of `docs/ball_physics_low_speed_validation.md`
(Sony-Ace family: `a = g − k_d|v|v + k_m(ω×v)`, angle-dependent tangential-impulse
contacts) on the 2026-07-03 venue mocap dataset (Avatar Pro, 300 Hz, C3D exports:
15-marker ball rigid body + two 4-marker paddles + table markers). This replaces the
Mac-local-only `~/Desktop/Hope/Record` workspace as the fitting toolchain of record.

Deliverables produced by this pipeline live at:
- `configs/ball_physics_venue.yaml` — fitted constants with provenance + validity envelope
- `docs/ball_physics_fit_report.md` — full FIT_REPORT (stages, F1–F8 falsification verdicts,
  held-out metrics)

## Data layout

Point `BALLFIT_DATA_ROOT` at the venue recording folder (default: yikang's Mac,
`~/Desktop/Hope/Record/latest`). Expected tree:

```
$BALLFIT_DATA_ROOT/
  <take folders with .c3d exports>       # raw (not required once extracted)
  analysis/
    extracted/<take>.npz                 # extract_canonical.py output
    qa_stage0.json                       # qa_stage0.py output
    segments/{flights,bounces,strikes,meta}.json   # stage1_segments.py output
    fits/stage2_fits_{all,train}.json    # stage2_fits.py output
    fits/stage4_validation.json          # validate_stage4.py output
    falsification/F*_verdict.json + .png # F1–F8 battery
```

## Pipeline (run in order; python needs numpy/scipy/matplotlib + `c3d`)

```bash
python extract_canonical.py <take_dir_or_c3d> analysis/extracted   # per take
python qa_stage0.py           # sampling/units/gravity gates — STOP if it fails
python stage1_segments.py     # flights / table bounces / racket strikes
python stage2_fits.py --split all     # ordered fits: k_d → k_m → table e → table tan → paddle
python stage2_fits.py --split train   # for held-out validation
python validate_stage4.py --fits .../stage2_fits_train.json --split test
python test_oracle_present.py # loud-fail oracle check (never skips)
```

## Conventions

- Table frame: origin = table center, X = length (2.740), Y = width (1.525), Z = up;
  the playing surface sits at `meta.surface_z` (≈ −14 mm: corner markers stand ~14 mm
  proud of the surface). Ball-center at table contact = surface_z + 0.020.
- SI units throughout; spin = rad/s expressed in the table frame
  (`spin_from_quats(..., R_table=take["table_R"])` — quats are template→world).
- The venue ball is coated for mocap: m = 3.4 g (clean ITTF ball 2.70 g). Fitted k_d/k_m
  are acceleration coefficients for THIS ball; scale by m_taped/0.0027 for a clean ball.
- The 15 exported ball "markers" are Avatar-Pro solved-model points, NOT physical
  sphere-surface positions (max pairwise span 56 mm > ball diameter). Treat the centroid
  as a rigidly-attached virtual point; the true-center offset is handled dynamically
  (QA wobble check), not geometrically.
- Never finite-difference accelerations for fitting — RK4 shooting fits only
  (`ballcore.fit_arcs_global`), g frozen at 9.81.

## Gotchas learned on this dataset

- Contacts are usually OCCLUDED (racket blocks cameras) → the contact falls between
  tracked runs. Pair arcs ACROSS gaps (extrapolate both sides to a meeting point);
  in-run contact detection alone finds almost nothing.
- Parabola-g arc gates must be wide ([5, 16]) or heavy-topspin arcs get systematically
  excluded (vertical Magnus biases apparent g) — which would blind the falsification tests.
- Quaternion spin is trustworthy below ~75 rev/s at 300 Hz; its SCALE was cross-validated
  aerodynamically (venue k_m 0.0044 inside the old rig's CI [0.0035, 0.0049]).
- Venue position noise ≈ 9 mm shooting-fit RMS (old OptiTrack rig: 0.4 mm) — widen
  outlier tolerances accordingly; contact windows are ≤25 frames with ±3/±5-frame
  exclusion zones at table/racket contacts.
