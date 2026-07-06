# Run Ball-Physics Fit And Trajectory Gates

Status: Partial

This operation doc records the reproducible commands for the venue ball-physics
pipeline under `hope_training/ball_physics_fit/`. It is the G03 entry point for
re-running the 2026-07-03 venue fit and the deploy-style ball-only trajectory
prediction / strike-factor gates.

## Task Setup

Use the venue dataset root:

```bash
cd hope_training/ball_physics_fit
export BALLFIT_DATA_ROOT=/Users/yyk956614/Desktop/Hope/Record/latest
```

Install dependencies in the Python environment used for fitting:

```bash
python -m pip install -r requirements-ballfit.txt
```

The `c3d` package is required only for raw C3D extraction. Everything after
`analysis/extracted/*.npz` exists can run without `c3d`.

## Full Pipeline

```bash
python extract_canonical.py "$BALLFIT_DATA_ROOT/<take>" analysis/extracted  # repeat per take
python qa_stage0.py
python stage1_segments.py
python stage2_fits.py --split all
python stage3_falsify.py
python stage2_fits.py --split train
python validate_stage4.py --yaml ../../configs/ball_physics_venue.yaml --paddle-e exp
python predict_check.py --yaml ../../configs/ball_physics_venue.yaml --split all
python trajectory_prediction_gate.py --yaml ../../configs/ball_physics_venue.yaml
python strike_factor_audit.py --yaml ../../configs/ball_physics_venue.yaml
python test_oracle_present.py
```

Stop after Stage 0 if sampling, units/frame, or gravity gates fail.

## Current Known Result

Verified on 2026-07-03 venue data from this Mac:

- `stage3_falsify.py --only F3,F6`: F3 `PASS` in coverage after the contact-time fix; F6 `PASS`.
- `trajectory_prediction_gate.py`: ball-only median landing error from post-hit windows:
  post30 ms `0.250 m`, post60 ms `0.181 m`, post100 ms `0.099 m`,
  post150 ms `0.074 m`, net+20 ms `0.030 m`.
- The same gate reports future trajectory position/velocity errors and
  net/centerline crossing height/time, not just landing.
- `strike_factor_audit.py`: current low-spin data has spin landing shift
  `25 mm` median / `78 mm` p90 if spin is ignored, but high spin is still out of
  coverage; front/back rubber side is not labeled; contact position is only a
  marker-centroid proxy; the rigid paddle-contact residual remains dominant
  (H0 through-paddle `0.250 m` vs H1 measured-out `0.067 m`).

Outputs are written under:

```text
$BALLFIT_DATA_ROOT/analysis/fits/
$BALLFIT_DATA_ROOT/analysis/falsification/
```
