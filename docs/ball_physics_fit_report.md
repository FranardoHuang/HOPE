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
states use ≤35-frame shooting windows with ±3/±5-frame exclusion zones.**

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

1286 flight arcs (514 clean ballistic, 337 s), **218 table bounces**, **154 racket
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
| table e_n | **0.9215** (forensics estimator over all 218: 0.925, CI [0.920, 0.937]) | 0.908 (old venue table) | v_n 1.0–4.5 m/s. The first-pass 0.934 carried a +0.010 contact-time bias (fixed, §9.1); remaining +0.013 vs the old table is plausibly real |
| table a_t/b_t/μ | see F5 | 0.369/0/2.0 | tangential fit is the weakest block on this data — see §4 F5 |
| paddle e_eff | **0.654 const**; shipped form e(u_n) = 0.759·exp(−0.0441·u_n) (F4 KILL) | 0.4632 (never fit on racket) | **first real racket fit**, 150 strikes, u_n 1.4–7.2 m/s |
| paddle a_t | **0.52** (velocity channel, CI [0.46, 0.61]) | 1.205 | joint fits are dragged to ~0.32–0.38 by the noise-dominated strike spin channel (§9.2); μ_safety → 0.5 |
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
| **F3** table e const | **PASS within coverage (upgraded by forensics)** | the e>1 tail and the apparent low-v_n slope were a PIPELINE BUG, mechanism found (forensics `g_e_gt1_dissection`): the assumed contact height sat ~9 mm above the true pre/post-fit intersection, making t_c early and inflating e by ≈0.17/v_n² (95% of v_n<1 bounces read e>1). With the fix (t_c at the two-sided meeting point, 35-frame windows): **e_n = 0.920, MAD 0.03, e>1 down to 3–4%, slope vs v_n ≈ −0.003/m/s ≈ 0** over v_n 1–4.5. v_n>4.5 (shell-buckling) still NOT covered |
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

Two parameter sources are scored: the train-split refit (methodology check) and
**THE SHIPPED `ball_physics_venue.yaml`** (`validate_stage4.py --yaml`, paddle e =
exponential; table block = the retained v0 grip params). Landing ground truth is the
**observed first-bounce contact point p_c** (continuity-gated pairing); the earlier
reconstructed label (measured out-state through the same flight model) read
optimistic and is only reported for comparison.

| Metric | Target | train-fit | **shipped yaml** | Verdict (yaml) |
| --- | --- | --- | --- | --- |
| Flight +0.4 s from first 60 ms | med ≤ 25 / p95 ≤ 60 mm | 76 / 147 mm (n=31) | 77 / 147 mm | FAIL vs bar |
| Bounce outgoing speed | ≤ 5% | 4.2% (degenerate tangential) | **2.32%** (n=22, after the t_c fix) | **PASS** |
| Bounce outgoing direction | ≤ 3° | 6.1° | 4.6° | FAIL vs bar (noise-limited) |
| Bounce ω⁺ | ≤ 15% | — | 217% | not meaningful (spins ≤15 rev/s ≈ noise; Δω channel reads 0.65×) |
| **Strike → landing vs observed/terminal GT** | med ≤ 0.10 / p90 ≤ 0.25 m / on-off ≥ 90% | — | **med 0.25 m, p90 0.72 m, on/off 100%** (n=82 all-split incl. 32 recovered landings; test-split n=15: 0.26/0.60) | on/off PASS; distances FAIL (model-form limited, §9) |

Notes: (i) the previously-reported bounce-speed 4.2% PASS came from the train-fit's
*degenerate* tangential block, not the shipped yaml — with the yaml's v0 grip block the
same metric is 5.96% and FAILS; the yaml is scored honestly above. (ii) the
reconstructed-label landing numbers (0.151/0.556 on the same pairs) are optimistic vs
observed-p_c ground truth, as expected for a label that shares the flight model.

### 5.1 Error budget — two-horizon prediction check (`predict_check.py`, n=48 pairs)

Prediction of the observed landing point made at three moments:

| Horizon | Median | p90 | Timing (med) |
| --- | --- | --- | --- |
| **H2 — ~100 ms before touchdown** (state refit near landing → flight) | **5.3 mm** | 15 mm | −0.7 ms |
| **H1 — at racket contact, measured outgoing state → flight** | **67 mm** | 144 mm | −5 ms |
| **H0 — at racket contact, through the paddle model** | **250 mm** | 725 mm | −96 ms |

Reading: the flight model itself is excellent (H2 ≈ mm-level; H1's 67 mm over
0.3–0.9 s of flight is consistent with outgoing-state estimation noise at 9 mm/300 Hz).
**The paddle contact model dominates the at-strike error** (H0 ≈ 3–4× H1, and its −96 ms
median timing bias says the modeled outgoing ball is systematically "hot"). Fit
stability across the 70/30 split: k_d ±1.7%, k_m ±3.6%, table e ±0.01%, paddle e ±0.2%
— but paddle a_t is NOT stable (0.40 vs 0.83). Improvement without new data therefore
concentrates on the paddle block (coherent refit with e(u_n) frozen, better racket
contact-point handling) and on contact-window state estimation — see the forensics
addendum (§9).

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

1. **Adopt the venue yaml for Isaac/MuJoCo low-speed work** — with eyes open: the
   paddle block is the big win (first real racket fit; e(u_n) exponential form,
   ±0.05 e ≈ 15 cm landing) and the flight block is solid (H2 = mm-level, model
   residual ~45–66 mm @0.4 s). Know the honest at-strike ceiling: through-paddle
   landing prediction is ~0.17–0.22 m median with this model form (§5.1, §9) — good
   enough for on/off-table reward gating (95%), NOT yet for 0.10 m-grade shaping.
2. **Magnus saturation (F2/R1) stays OPEN** — the venue data never reaches SR > 0.7
   effectively. Options: (a) keep constant k_m and enforce the SR ≤ 0.5 envelope in
   reward rollouts; (b) adopt the Ace saturating form now (calibrated to match
   k_m = 0.00444 at SR ≤ 0.5) so behavior is already correct if policies discover
   high spin. (b) is ~5 lines in flight.py + C++ mirror and strictly safer.
3. **Deploy-side freebies from the noise-floor MC**: 120 ms init windows
   (flight prediction 76→~57–68 mm), spin from longer quat windows, and observer
   tuning that models the colored noise (1.9 mm white + AR(1) ρ≈0.94, 5.2 mm marginal).
4. **The remaining wall is paddle model FORM / racket-state accuracy** (tangential
   velocity error 0.68 m/s median; face-normal ±5° swings worst cases by ±0.3 m).
   Parameter tuning is exhausted (§9.2). Improvements need either dwell modeling /
   better face-normal estimation on existing data, or cleaner racket capture.
5. Next capture session: heavy-spin serves (30–60 rev/s, SR 1–4), shallow-incidence
   bounces, denser cameras on the strike zone (racket normal), and an ITTF drop test
   on the venue table (adjudicates e_n 0.92 vs the old 0.908).
6. F7 spin decay: not actionable (n=14, CI includes 0). Re-test on the next dataset;
   an exponential decay term costs one line if it firms up.

## 8. Reproduction

```bash
cd hope_training/ball_physics_fit
export BALLFIT_DATA_ROOT=~/Desktop/Hope/Record/latest
python extract_canonical.py "$BALLFIT_DATA_ROOT/<take>" analysis/extracted  # ×9
python qa_stage0.py && python stage1_segments.py
python stage2_fits.py --split all && python stage2_fits.py --split train
python stage3_falsify.py                         # F1–F8 + adversarial verifiers
python validate_stage4.py --yaml ../../configs/ball_physics_venue.yaml --paddle-e exp
python predict_check.py  --yaml ../../configs/ball_physics_venue.yaml --split all
python test_oracle_present.py
```

## 9. Forensics addendum (2026-07-03 late) — active bug hunt + improvement ceiling

A second adversarial pass ("it ran" ≠ "it is right") over the finished pipeline.
Raw verdicts: `<venue data>/analysis/forensics/`.

### 9.1 Pipeline bugs found and FIXED

1. **Table-bounce contact-time bias (the e>1 mechanism).** The assumed contact height
   (bounce-minima surface + R) sits ~9 mm above where the pre/post window fits actually
   intersect; refining t_c to that height made t_c early, and gravity back-extrapolation
   inflated e by ≈0.17/v_n² — hence 95% of v_n<1 bounces reading e>1 and the fake
   F3 slope. FIX in `stage1_segments.py`: t_c stays at the two-sided meeting point,
   35-frame windows. Corrected **table e_n = 0.9215 (was 0.934)**; e>1 tail 17%→~4%;
   e-vs-v_n slope ≈ 0. F3 upgraded INCONCLUSIVE → PASS-within-coverage.
2. **Landing ground truth.** Stage-4 originally reconstructed the "measured" landing
   through the same flight model (optimistic, as flagged in review). Now: observed
   bounce p_c, continuity-gated. Additionally, the bounce extractor misses ~2/3 of
   first bounces (post-bounce arc occluded/short) — `predict_check.py` recovers those
   landings from the TERMINAL WINDOW of the incoming arc (H2-grade, mm–cm accurate,
   identity-gated so juggle re-hits can't masquerade as landings): landing-GT n 48→82 (identity-gated).
3. **Wrong strike→bounce pairings.** 21/69 naive pairings grabbed a later bounce
   (median continuity error 6.5 m — unambiguous). All landing metrics use the
   continuity gate now; 1 of the original 20 stage-4 pairs was wrong.

### 9.2 Paddle block — resolved (this is where the improvement budget went)

- The a_t "instability" (0.40↔0.83 across splits) is **collinearity, not noise**: all
  strikes are near-normal (cosθ median 0.98), so only a_t + b_t·cosθ is identified.
  With b_t≡0 the subsample spread collapses (a_eff = 0.375 ± 0.031 joint).
- **The strike spin channel is junk on this rig**: quaternion Δω at strikes reads
  ~0.22× the position-implied impulse (IQR −0.01…0.46; table bounces read 0.65×).
  Joint fits let it drag a_t down to ~0.38. The velocity channel and three
  model-free tangential-impulse estimates agree on **a_t ≈ 0.58** (0.576 / 0.601 /
  0.622; stat std 0.023, systematic floor 0.49). Adopted in the yaml with μ→0.5
  (cap never binds; measured impulse-ratio p90 = 0.27).
- Landing error is FLAT over a_t 0.2–0.6 and sharply optimal at the F4 e-calibration
  (±10% e ≈ ±5 cm landing) — so the a_t choice follows the direct tangential
  measurement, and the landing-relevant error is elsewhere:
- **Dissection of the through-paddle landing error** (48 clean pairs): swapping the
  model's outgoing VELOCITY for the measured one removes nearly all excess error
  (0.164→0.074 median) while swapping spin changes little → the residual paddle-model
  deficit is tangential-velocity form error (median |Δv_t| = 0.68 m/s vs |Δv_n| = 0.02)
  plus racket-face-normal sensitivity (±5° tilt swings worst-case landings by ±0.3 m).
  Parameter refits cannot buy more here (landing-error sweep over (a_t, e-scale):
  best cell 0.163 m vs shipped 0.164 m — zero headroom); the ceiling is model FORM
  (dwell/normal estimation), consistent with H0−H1 in §5.1 and the noise-floor MC.

### 9.3 What was checked and NOT changed

- **Wobble δ**: the extractor's sphere-fit center offset is wrong by ~1.9 mm (35°
  direction error), verified by injection test and split-half stability. Effect is
  ~0.1 mm rms under the 9 mm noise floor (k_d bias ≈0, k_m < 3%) → documented in the
  README as a known systematic; not worth a full re-extraction on this rig.
- **Noise-floor Monte Carlo** (exact Stage-4 estimators on synthetic truth): the rig
  noise is NOT white — ~1.9 mm/axis white + a correlated AR(1) ρ≈0.94 component
  (5.2 mm marginal, ~60 ms correlation time; this is why short windows look clean but
  long arcs read 9 mm). Flight metric floor at 60 ms init: 37–60 mm median (white →
  colored) vs 76 observed → noise explains 24–63% of the variance; a genuine
  **flight model-form residual of ~45–66 mm @0.4 s remains**. Landing floor: 32–64 mm
  vs 164 observed → **noise explains only 4–15% — the landing miss is decisively
  model/racket-state error (~0.15 m median residual)**, robust to every noise model
  tested. Free deploy-side gains: 120 ms init window (flight 76→~57–68 mm), longer
  spin windows (+16 mm quadrature removed); deploy Kalman/observer tuning should
  model the colored noise term. Old-rig reference: both floors collapse to ~2 mm.

### 9.4 Improvement space without new data — final accounting

| Lever | Status | Gain |
| --- | --- | --- |
| Bounce t_c bias fix | **DONE** | e_n unbiased (0.9215); F3 clean; e>1 17%→4%; bounce-speed validation 5.96%→2.32% (FAIL→PASS) |
| Observed/terminal-window landing GT | **DONE** | honest metrics + n 48→84 |
| Paddle a_t via velocity channel | **DONE** | tangential gain trustworthy (0.58±0.02 stat) |
| Paddle e(u_n) exp form | **DONE** (F4) | landing-relevant e calibrated at optimum |
| Longer init windows for deploy prediction | free at deploy time | H2 shows the ceiling: mm-level at 100 ms lead |
| Paddle model FORM (tangential + face normal) | **the remaining wall** | would need cleaner racket tracking or dwell modeling; est. bulk of the H0 ~250 mm vs H1 67 mm gap |
| Magnus saturation / high spin | **not addressable** with this data | needs a dedicated high-spin capture |
