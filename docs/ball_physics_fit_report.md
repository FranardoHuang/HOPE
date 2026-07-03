# Ball Physics v1 — Venue Fit Report (2026-07-03 dataset)

> Produced 2026-07-03 by the Stage 0–5 pipeline of
> `docs/ball_physics_low_speed_validation.md` §4, on the venue mocap dataset.
> Fitting code: `hope_training/ball_physics_fit/` (this replaces the Mac-local
> Record workspace). Constants: `configs/ball_physics_venue.yaml`.
> Raw analysis artifacts (segments, per-event tables, falsification plots):
> `~/Desktop/Hope/Record/latest/analysis/` on yikang's Mac.

## 0. Dataset

9 takes, Avatar Pro C3D exports, 300 Hz, ~800 s total: 上旋/下旋/侧旋 (spin strikes),
弹跳 (bounce trials), 快速 (fast strikes), 正常 (rally), 颠球不转/颠球增旋 (juggling),
高球 (lobs). Rigid bodies: ball (15-marker solved body), two paddles (4 markers each,
b_PPP1/b_PPP2 — **first time racket data exists**), table (4 corners + 2 net posts).

Preprocessing (`extract_canonical.py`): per-take npz with ball 6-DoF (centroid +
Kabsch quaternion), both paddle poses + face normals, canonical table frame
(origin table center, X = length, Y = width, Z = up); first/last 2 s trimmed;
below-table samples (ball center z < 0 in table frame) invalidated **with the time
axis preserved**; original frame indices kept.

Data quality: ball tracked 62–99.5% per take (fast/spin takes lower — the ball
leaves the capture volume between rallies); gaps concentrate AT contacts
(racket occlusion); no interpolation artifacts (0 frozen frames); Kabsch RMS
≈ 1e-4 mm (solved body). **Venue position noise ≈ 9 mm shooting-fit RMS — 20×
noisier than the old OptiTrack rig (0.4 mm). All fits use robust losses; contact
states use ≤25-frame shooting windows with ±3/±5-frame exclusion zones.**

A finding to keep in mind: the 15 exported ball "markers" are solved-model points,
not physical surface positions (max pairwise span 56 mm > 40 mm ball diameter;
sphere-fit residual 17 mm). The centroid is a rigidly-attached virtual point — fine
for dynamics; the geometric center offset is unknowable from geometry alone and was
handled dynamically (Stage 0 wobble check found spins too low for it to matter).

## 1. Stage 0 — QA gates: **PASS**

| Gate | Requirement | Measured | Verdict |
| --- | --- | --- | --- |
| Sampling | ≥250 Hz effective | 300 Hz uniform, gap p50 ≈ 10 ms (occlusions, not drops) | PASS |
| Units/frame | table = 2.740 × 1.525 m | 2.745 × 1.523 m (marker centers) | PASS |
| Gravity | 9.81 ± 0.05, tilt < 0.2° | **9.825**, tilt vs world z **0.15°** (60 low-spin arcs, joint shooting fit) | PASS |
| Time sync | ball/paddles same clock | same C3D stream | PASS (trivially) |
| Ball mass | record taped mass | **3.4 g** (coated; clean 2.70 g) — from team memory, no session metadata file | PASS w/ note |

Notes: (a) the table-corner-marker plane is tilted 0.40° vs gravity while the mocap
world z is gravity-true to 0.15° — corner markers stand 12–40 mm proud (uneven
mounts); the playing surface was therefore self-calibrated from 179 bounce minima:
**surface z = −14.1 mm** below the marker plane. Normal-direction error ≤0.4° affects
e_n by <0.7‰ — negligible. (b) Quaternion spin reads low overall (median 2 rev/s,
max 15 rev/s across ballistic arcs). Adjudicated aerodynamically in Stage 2: the
sidespin-channel k_m came out **inside the old rig's 95% CI**, so the spin *scale*
is right — this venue session simply contains modest spin (amateur play, no
high-spin serves).

## 2. Stage 1 — segmentation

1286 flight arcs (514 clean ballistic, 337 s), **217 table bounces**, **151 racket
strikes**, produced by pairing ballistic arcs ACROSS tracking gaps (extrapolating
both sides to a meeting point < 60 mm, requiring |Δv| > 0.8 m/s) — contacts are
occluded in this rig, so in-run contact detection alone finds almost nothing.
Racket state at contact is bridged over the ±30 ms occlusion window by a
constant-acceleration fit on [−120, −15] ms (never post-contact frames), with the
ω_racket × r contact-point term included in v_r (the v0 code gap #1).

## 3. Stage 2 — fitted constants (all data; g frozen at 9.81)

| Param | Venue value | Old value | Notes |
| --- | --- | --- | --- |
| k_d | **0.1261** 1/m (C_d 0.569 taped) | 0.1222 (v0), 0.136 (OptiTrack) | 100 low-spin arcs, joint shooting fit; rms 9.1 mm |
| k_m | **0.00444** | 0.0042, CI [0.0035, 0.0049] | 66 sidespin arcs (axis_z > 0.6); inside precedent CI → spin channel validated |
| table e_n | **0.934** (MAD 0.044) | 0.908 (old venue table) | 92 gated bounces, v_n 1.0–4.5 m/s; different physical table |
| table a_t/b_t/μ | see F5 | 0.369/0/2.0 | tangential fit is the weakest block on this data — see §4 F5 |
| paddle e_eff | **0.654 const** / e(u_n) = 0.77 − 0.032·u_n (OLS), −0.013…−0.018 robust | 0.4632 (never fit on racket) | **first real racket fit**, 150 strikes, u_n 1.4–7.2 m/s |
| paddle a_t | **0.399** | 1.205 | large move, as predicted |
| paddle b_t | −0.035 (≈0) | 0.304 | cosθ term not supported by venue data |
| paddle μ | unidentified (cap never binds) | 2.5 | keep safety value |

Mass scaling (doc Stage 0.5): k_d,clean = k_d·(3.4/2.7) = **0.159**, k_m,clean =
**0.00559** — these remove the mass effect ONLY; a clean ball is also smoother
(C_d likely ~0.44 vs 0.57 coated), so treat clean-ball extrapolation as an upper
bound on drag/Magnus. The v0 yaml's 0.1222 remains the best clean-ball estimate.

## 4. Stage 3 — falsification verdicts (doc §3)

| # | Assumption | Verdict | Key numbers |
| --- | --- | --- | --- |
| **F2** Magnus saturation | **INCONCLUSIVE — by coverage (high conf.)** | the trusted channel never reaches the saturation region: only 1 kept arc with SR > 1.5 (KILL test needs ≥5/bin). Low-SR sanity PASSES (bin medians ~0.0052 ∈ [0.003, 0.006] → not a rig problem). Suggestive: sidespin bin medians fall 0.0054 → 0.0028 across SR 0.16 → 0.35 (wide CIs), roughly consistent with saturation onset (saturating law predicts 0.68×). **The R1 risk stays open for high-spin play** — see §7.2 |
| **F4** paddle e const | **KILL (medium conf.) — adopt e(u_n)** | 7-estimator ensemble Δe(2–8 m/s) = −0.14 (quality-gated re-derivation: −0.11), all 7 negative, beyond the −0.05 KILL line; within-take fixed-effects slope p=0.0015 (survives excluding the suspicious dianqiu_spin take and controlling fit quality); high-quality u_n≥5 tail: median e 0.628 vs 0.675 bulk (MW p=0.017); best slope −0.013…−0.018/m/s ≈ literature −0.017…−0.021. Racket-ω confound ruled out (partial r=−0.05). Caveats: pooled Theil-Sen alone straddles 0; trend carried by the u_n>4.3 tail (n=17); paddle p2 shows it robustly, p1 weakly; e also correlates with \|ω_out\| at fixed u_n → e(u_n)-only is an incomplete form |
| **F3** table e const | **INCONCLUSIVE (flat where measurable)** | over v_n 2.0–4.5: flat within ±0.023 of 0.934. Apparent slope −0.021/m/s is manufactured by noise-inflated e>1 events below v_n 1.5 (16/92 bounces e>1, 10 of them at v_n<2; scatter 1.7× larger in low half). No CI excludes \|slope\|<0.01. v_n>4.5 (shell-buckling region) NOT covered |
| **F5** grip-only tangential | **INCONCLUSIVE — venue refit degenerate; v0 grip block retained** | the venue tangential refit collapsed to zero tangential impulse (Coulomb cap never binds, μ unidentified) because the Δω noise floor alone is 2.0 rev/s (vs the 3 rev/s PASS bar) at 9 mm position noise. Residuals DO grow with \|u_t\|/\|u_n\| (Spearman 0.45, p=1.5e-6, coverage to ratio 2.8) but not in the specific slip signature. Diagnostic: across contacts, quaternion Δω reads ~0.65× the value implied by the position-derived tangential impulse (contact-window blur) — the two channels disagree, so this dataset cannot beat the old 101-bounce/0.4 mm fit; a_t=0.369 grip block retained in the yaml |
| F1 k_d const | **PASS** (medium) | bins within ±6% of global over 2–6.8 m/s (n=100); Spearman p=0.52; >6.8 m/s uncovered |
| F6 no inverse Magnus | **PASS (high conf.)** | worst cell (highest Re 12.6–28k × lowest SR): median km = +0.0065, 1/13 arcs negative, sign-test p=1.0. The one negative-median cell sits at LOWEST Re × HIGHEST SR (the physical opposite of inverse-Magnus) and dissolves under a deflection-≥-noise gate (n=7, median −0.0004, p=0.5). Certifies the envelope only — Re tops out at 2.8e4, far below the ~1e5 inverse-Magnus onset |
| F7 no spin decay | **borderline KILL (low conf.)** | median decay 12.2%/flight but n=14, IQR [−3%, +21%], bootstrap CI includes 0, Wilcoxon p=0.086; alt normalization gives 5%. Not actionable alone — see recommendation |
| F8 instantaneous paddle | **INCONCLUSIVE → PASS-leaning** | raw Spearman 0.21 (p=0.01) between direction residual and \|ω_racket\| collapses to 0.11 (p=0.19) after controlling u_n and to −0.03 controlling (u_n,u_t,v_out): the correlation is speed confounding. Direct dwell rotation ≈ 0.15° median — well under 2° |

### 4.5 Adversarial verification (independent re-derivations; all AGREE, severity "minor")

- **F2 verified** by a fit-free path (per-arc cubic-polynomial acceleration, gravity+drag
  subtracted, projected on unit(ω×v)): per-arc k̂_m agreement with the shooting fits
  Spearman 0.905. Sharpened finding: effective SR > 0.7 coverage is actually **zero**
  (the single "kept" high-SR arc flips sign between methods = noise; all 4 raw SR>0.7
  arcs also have near-degenerate Magnus geometry). The SR 0.3–0.45 dip is slightly
  deeper independently (0.46×) and remains non-monotonic — neither PASS for constant
  k_m nor confirmation of the saturating law inside coverage.
- **F4 verified**: independent re-derivation from strikes.json reproduces the pipeline
  e/u_n to machine precision (max diff 2e-16) — the dispute is purely statistical.
  Under stricter quality gates the pooled OLS weakens (p=0.088) but the within-take
  fixed-effects evidence (p=0.0015) and the u_n≥5 tail keep the KILL; magnitude
  tempered to Δe(2–8) ≈ −0.107.
- **Constants audit**: kd convention/value PASS (C_d taped 0.569, clean-mass 0.452);
  reproducibility PASS (6 random per-arc refits from raw npz bit-identical). Three
  flags adopted into the yaml: (1) **naive mass-scaled k_d,clean = 0.159 exceeds all
  clean-ball precedents — do not use** (tape changes C_d, not just mass); (2) **table
  e_n = 0.934 is above the ITTF band and 17% of bounces read e>1 (unphysical) —
  likely 2–5% high from 9 mm contact noise; adopt after an ITTF drop-test cross-check**;
  (3) k_m implies lift coefficient ≈ 1.00 (physical ceiling) — weakly corroborates
  quaternion spin reading ~6% low (inside the prior CI).

## 5. Stage 4 — held-out validation (fit on 70%, score on 30%, split by trial)

| Metric | Target (doc §4.4) | Measured (test split) | Verdict |
| --- | --- | --- | --- |
| Flight: pos error at +0.4 s from first 60 ms | med ≤ 25 mm, p95 ≤ 60 mm | **med 76 mm, p95 147 mm** (n=31) | FAIL vs absolute bar |
| Bounce: outgoing speed | ≤ 5% | **4.2%** (n=38) | PASS |
| Bounce: outgoing direction | ≤ 3° | **6.1°** | FAIL vs bar |
| Bounce: ω⁺ | ≤ 15% | 148% | not meaningful (see below) |
| **Strike → landing (money metric)** | med ≤ 0.10 m, p90 ≤ 0.25 m, on/off ≥ 90% | **med 0.164 m, p90 0.549 m, on/off 95%** (n=20) | on/off PASS; distances FAIL vs bar |

**Interpretation — the absolute bars were written for a 0.4 mm rig; this rig is 9 mm.**
The flight miss is dominated by initial-state noise, not model form: a 60 ms /
18-frame window at 9 mm noise yields ~0.15–0.3 m/s velocity error → 60–120 mm at
+0.4 s, exactly the observed band. Same mechanism inflates bounce direction (6°) and
landing distances (the "measured" landing itself is reconstructed from a noisy
post-strike state). ω⁺ validation is not meaningful on this dataset: spins are ≤15
rev/s (denominator ~ noise) and the F5 diagnostic shows the quaternion Δω channel
reads ~0.65× across contacts. What the venue data DOES establish: on/off-table
classification 95% ✓ (the reward-gating quantity), bounce speed 4.2% ✓, and
**fit stability across the 70/30 split: k_d ±1.7%, k_m ±3.6%, table e ±0.01%,
paddle e ±0.2%** — the constants are solid; paddle a_t is NOT (0.40 vs 0.83
across splits — wide uncertainty, expect refinement on a cleaner rig).

Per doc §4.4.4: flight+bounce failing on noise (not model form) with landing at
1.6× the bar means the deficit attribution to the paddle block is NOT confirmed
here; the noise floor must come down first (better camera coverage of the strike
zone, or the old OptiTrack rig for Set-D-style captures).

## 6. Validity envelope — nothing outside this box is validated

- Ball speed: **1–7 m/s** (p95 5.6, thin to 10.5); Re 2.5e3–2.8e4
- Spin: **0–15 rev/s** quaternion-validated (scale cross-checked aerodynamically)
- Spin ratio SR: **≤ 0.5 well covered; 0.5–1.6 sparse (20 arcs); > 1.6 EMPTY**
- Table impact v_n: **1.0–4.5 m/s** (buckling regime > 5 m/s not probed)
- Paddle u_n: **1.4–7.2 m/s**
- NOT covered: pro-level spin (40–100+ rev/s), SR > 1.6 (exactly where Magnus
  saturation lives), smashes > 8 m/s, shallow-incidence slip bounces (ratio > 2.3).
  Sony's own high-speed extrapolation failure applies verbatim: do not silently
  extrapolate.

## 7. Recommendations for the sim (v1)

1. **Adopt the venue yaml for Isaac/MuJoCo low-speed work now**; the paddle block is
   the big win (first real fit, e_eff 0.463 → 0.654 const or the e(u_n) linear form —
   ±0.05 e ≈ 15 cm landing, so this correction is worth ~50+ cm of realism at speed).
2. **Magnus saturation (F2/R1) stays OPEN** — the venue data never reaches SR > 1.6.
   Options: (a) keep constant k_m and enforce the SR ≤ 0.5 envelope in reward
   rollouts; (b) adopt the Ace saturating form now (calibrated to match k_m = 0.00444
   at SR ≤ 0.5) so behavior is already correct if policies discover high spin. (b) is
   ~5 lines in flight.py + C++ mirror and strictly safer.
3. Next capture session, if saturation matters: record deliberate heavy-spin serves
   (target 30–60 rev/s, SR 1–4) and shallow-incidence bounces; both boxes are empty.
4. F7 spin decay: not actionable (n=14, CI includes 0). Re-test on the next dataset;
   an exponential decay term costs one line if it firms up.

## 8. Reproduction

```bash
cd hope_training/ball_physics_fit
export BALLFIT_DATA_ROOT=~/Desktop/Hope/Record/latest
python extract_canonical.py "$BALLFIT_DATA_ROOT/<take>" analysis/extracted  # ×9
python qa_stage0.py && python stage1_segments.py
python stage2_fits.py --split all && python stage2_fits.py --split train
python validate_stage4.py --fits "$BALLFIT_DATA_ROOT/analysis/fits/stage2_fits_train.json"
python test_oracle_present.py
```
