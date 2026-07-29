# Ball & Racket Physics — OptiTrack session fit (2026-07-30)

> Data: `mocap/data/*.c3d` — OptiTrack/Motive export, 360 Hz nominal, meters:
> **331 s** of amateur rally play (`Tui`, `zhengchang`, `zhengchang2`, `xuan`)
> plus **20.5 s of pure bounces** (`chuntan`). Competition table.
> Assets: `A` (ball, 8-marker RIGID constellation), `PPP1`/`PPP2` (rackets),
> `PPT` (table).
> Constants: `configs/ball_physics_optitrack_20260730.yaml`.
> Artifacts: `mocap/data/analysis/` (extracted npz, segments, fits, QA).
> Fitting code: `hope_training/ball_physics_fit/` — the **same** estimators as
> the 2026-07-03 venue fit; only the reader and the contact detector are new.

## 0. Headline

This session is best understood as an **independent replication attempt** of the
venue fit on a different mocap system. It replicates the two things it measures
best and disagrees on two contact numbers.

> **Correction, later the same day:** an earlier version of this report concluded
> the ball was fully coated and spin was unmeasurable. **Wrong — the ball is a
> rigid marker constellation and spin IS measurable** (§6). Everything has since
> been refitted with the spin channel; §§3–5 still carry the intermediate
> spin-blind numbers where they are needed to show what changed, always labelled
> as superseded. The table-e conflict was then adjudicated with the pure-bounce
> take (§4).

**Final table, after the spin channel was recovered and the table-e conflict
adjudicated.** (The intermediate spin-blind numbers are kept in the sections
below, labelled as superseded, because two of the corrections only make sense
against them.)

| constant | venue 07-03 | this session 07-30 | verdict |
| --- | --- | --- | --- |
| `k_d` (drag) | 0.1261 | **0.1253** | **CONFIRMED** (0.6%) |
| `k_m` (Magnus) | 0.00444 | **0.00404** (45 sidespin arcs) | **CONFIRMED** — venue value inside this fit's per-arc IQR [0.0026, 0.0055] |
| ω constant in flight | F7: yes | **reproduced** — half-arc ratio 0.90–1.16 | **CONFIRMED** |
| paddle `e_eff` | 0.654 | **0.646** (24 spin-complete strikes) | **CONFIRMED** (1.2%) |
| paddle `e` is speed-dependent | F4: kill constant-e | **reproduced** (rms 0.099 vs 0.123) | **CONFIRMED** |
| table `e` | 0.9215 | rally reconstruction 0.9443 → **proven biased +0.044**; pure-bounce t_c-free estimator **0.910**, CI [0.883, 0.931] | **VENUE VALUE SHIPS** |
| paddle `e(u_n)` decay `g2` | −0.0441 | **−0.0703** (was −0.0942 spin-blind) | gap HALVED by including spin; still open |
| table grip `a_t` | 0.369 | **0.275** (60 spin-complete bounces) | open; only `a_t + b_t·cosθ` identified |
| paddle `a_t` | 0.52 | **0.637** (24 strikes) | open; same collinearity |

Landing prediction, 24 spin-complete strike→landing pairs, both configs on the
**same events**:

| horizon | this session's config | venue config |
| --- | --- | --- |
| H0 (through the racket) | **207.8 mm** | 269.8 mm |
| H1 (measured out-state → landing) | 55.6 mm | 53.9 mm |
| H2 (~100 ms before landing) | 3.6 mm | 3.5 mm |

The spin-refitted contact model predicts strikes **23% better** than the venue
constants on this session's data, and ties on the pure-flight horizons. For
scale, the same config evaluated without using spin, over the larger 80-pair
set, reads H0 807.7 / H1 71.7 / H2 8.0 mm — but that set is also the
worse-tracked one, so that 4× is spin *and* subset quality, not spin alone.

### Sync status (decided 2026-07-30)

The capture is on the **competition table** (confirmed by the team), and the
table geometry drift seen between takes is **mocap instability, not the table
moving** — the table is a fixed 0.76 m unit. So the two sessions measured the
*same physical table* and the table-e gap in the table above is a **measurement
conflict, not a table difference**.

Supporting check: table e per take reads 0.9409 / 0.9468 / 0.9555 / 0.9441 with
bootstrap CI widths of 0.03–0.05, i.e. all four mutually consistent, and it does
**not** track mocap quality — the take with the worst corner-plane RMS (xuan,
2.03 mm) gives 0.9441, essentially the same as the best (tui, 0.47 mm → 0.9409).
Mocap drift is therefore not contaminating e.

**Decision: this session's constants are ACTIVE for the simulator, via the
config-path switch — no byte-pinned file was modified.**

```bash
export HOPE_BALL_PHYSICS_YAML=<repo>/configs/ball_physics_optitrack_20260730.yaml
```

That covers Isaac (`tasks/table_tennis/physics/params.py`), the tracking-task
virtual ball, `virtual_return_scorer.py` and `reference_oracle.py` — all four
resolve the env var first. For the MuJoCo C++ sim, pass
`.../cfg/model/a3_pingpong/ball_physics_optitrack_20260730.yaml` to
`BallPhysics::Init(m, d, config_path)`.

**What is deliberately NOT switched, and why.**
`configs/ball_physics_venue.yaml` and `hope_ws/.../hope_planner/constants.py`
are byte-pinned by the Stage-1 question-bank physics contract
(`scripts/rebind_stage1_question_bank_physics_contract.py`,
`EXPECTED_PHYSICS_FILES`). That rebind script "never regenerates or edits a
question" and only accepts AST-identical physics, so editing constants in place
would **invalidate the existing Stage-1 question bank** and break comparability
with every prior Stage-1 exam result. Consequence: the **deploy planner's
constant mirror still carries the venue numbers.** If deploy should move too,
that is a separate, deliberate operation requiring a bank regeneration.

Relative to the venue constants this changes `k_m` (0.00444 → 0.00404), the
paddle block (`e_eff` 0.654 → 0.646, the `e(u_n)` curve, `a_t` 0.52 → 0.637) and
the table grip (`a_t` 0.369 → 0.275). Table `e` deliberately stays at the venue
value, and `k_d` moves only 0.6%.

## 1. What the data is, and one thing that had to be fixed first

Independent geometric QA (all from `PPT`, nothing fitted):

| check | measured | reference | error |
| --- | --- | --- | --- |
| table length (marker span) | 2.7570 m | 2.740 m + ~2×9 mm marker standoff | ✓ |
| table width (marker span) | 1.5310 m | 1.525 m | +6 mm |
| 4 corners coplanar | 0.47 mm rms | — | ✓ |
| net posts, x in table frame | −0.0015 m | 0 (table centre) | 1.5 mm |
| net post above corner plane | 0.1521 m | ITTF net 0.1525 m | **0.4 mm** |

The net-height agreement to 0.4 mm is an independent confirmation of the
vertical scale, and the net-post x confirms the fitted table frame.

**Export blanks.** ~11.5% of frames are written **empty**: the ball, both
rackets *and* the bolted-down table markers vanish together (99.3%
co-occurrence), roughly every 11th frame with a drifting phase. The table is
static and permanently visible, so these are exporter blanks, not occlusion —
effective sample rate is ~317 Hz on an unchanged 360 Hz time grid. The
extractor repairs holes ≤ 3 frames by local quadratic fit through 6 real
samples per side and **marks** them (`ball_interpolated`,
`frame_blank_in_export`). Without the repair, tracked runs never reach the
15-frame minimum and arc extraction yields **zero** arcs. Measured effect on
restitution: bounces with a filled frame within ±4 of contact read e = 0.950
vs 0.944 for clean ones.

## 2. Pipeline: what was reused and what had to be new

Reused unchanged — these are the algorithms: `ballcore.fit_arcs_global` (RK4
shooting), `stage1_segments` (cross-gap arc pairing, two-sided meeting-point
contact times), `stage2_fits` (k_d → k_m → table e → tangential → paddle),
`predict_check`, `contact_model.predict_contact`.

Three things genuinely had to change:

1. **`optitrack/ot_extract_full.py`** (new). Neither existing reader fits this
   export: `extract_canonical.py` is mm-hardcoded, expects `b_PPP1_*` labels,
   loads whole files through `c3d.Reader.read_frames()` (hopeless on 2.3 GB /
   3255 columns), and — the real blocker — runs a **rigid-body Kabsch solve on
   the ball**, which on a coated ball emits fabricated quaternions and a
   fabricated centre offset. `optitrack/ot_extract.py` streams correctly but is
   ball-only: no table, no rackets. This session is the first with all four
   assets, so it fell between the two. Output is the canonical npz schema.

2. **`ballcore.detect_contacts_racket`** (new, venue path untouched). The
   existing detector infers a strike from pointwise `d(vel)/dt`; at 360 Hz with
   a ~6 mm ball-centre noise floor its 1 m/s threshold sits ~2.5σ out. It fired
   **1035–1860 times per take** against 41–95 real table bounces, shredding
   every flight into 0.09 s fragments and driving parabola-g to 3.5–8.6 m/s².
   With rackets tracked, a strike is instead *proposed* at a ball–racket
   distance minimum and *confirmed* by a two-sided velocity jump fitted on ~12
   real samples per side.

   | | median arc duration | median parabola-g | ballistic arcs |
   | --- | --- | --- | --- |
   | old detector | 0.094 s | 3.5–8.6 | 48 |
   | racket-aware | **0.24–0.33 s** | **9.4–10.2** | **274** |

3. **Rig-scaled quality gates.** The venue's 5/8/15 mm window-fit gates assume a
   ball centre from a 15-point *solved* rigid body. Here the centre is a
   per-frame sphere fit through wandering points, so window rms is ~11 mm and
   the venue gates reject **100%** of bounces. Gates are now `--rms-gate-*`
   parameters (venue defaults preserved) and every fit reports a
   strict-quartile sensitivity. For table e the gate does not carry the result:
   0.9443 over all 147 vs 0.9421 over the cleanest 37.

Take registry and stage role selection are now data-root driven
(`analysis/takes.json`), so the same code runs on either session.

## 3. Flight

**`k_d = 0.1253`** (100 longest ballistic arcs, joint RK4 shooting, g frozen,
rms 12.6 mm). Stability across 274 ballistic arcs: 0.1266 (all) / 0.1266 (mid
speed third) / 0.1261 (fast third) / 0.1253 (arcs ≥ 0.35 s); fitting g jointly
instead of freezing moves it < 0.2%. The slow third reads 0.1446 — drag is
weakly identified below ~3 m/s and that subset should not be used.

Venue: 0.1261. **Two different mocap systems agree to 0.6%.** This is the
strongest single result in the fit, and it is also indirect evidence that the
ball is the same coated 3.4 g ball (no session metadata records it).

**Gravity is frozen at 9.81, and that is measurement-backed.** Fitting `|g|`
freely returns:

| arc subset | median speed | fitted \|g\| |
| --- | --- | --- |
| slowest third (n=91) | 2.71 m/s | 9.70 |
| middle third (n=92) | 4.21 m/s | 9.90 |
| fastest third (n=91) | 5.89 m/s | **10.45** |

A clock or scale error is speed-*independent*. A monotone rise with ball speed
is the Magnus term being absorbed into gravity — fast shots carry topspin
(downward Magnus), slow pushes carry backspin. Since table geometry pins the
spatial scale to 0.3% and net height to 0.4 mm, 9.81 stands. **This is also the
only place spin shows up as a clean, real signal in this dataset.**

## 4. Contacts

### Table

`e = 0.9443`, 147 gated bounces, bootstrap CI95 [0.9379, 0.9538], v_n coverage
1.18–4.13 m/s. Speed law `e = 1.0058 − 0.0219·v_n`, slope CI [0.0100, 0.0368] —
sign matches the venue's optional form and the 8194-bounce literature
(−0.017…−0.021/m/s), magnitude ~2×.

**This conflicts with the venue's 0.9215 by 2.4%, outside both CIs, and implies
e = 0.953 at the ITTF 30 cm drop speed against the ITTF band 0.876–0.931.**

Since both sessions were on the **same competition table** (§0), this is a
measurement conflict — one of the two estimates carries a systematic. Ruled out
as the cause on this side: the quality gate (strict quartile 0.9421),
blank-frame interpolation (0.950 vs 0.944), spatial scale (§1), and mocap drift
(per-take e is flat against corner-plane RMS, §0). The one live suspicion
remaining on this side is a residual early-`t_c` bias: reconstructed contact
height sits +3.2 mm above geometric, and e rises with that offset across
terciles (0.938 / 0.944 / 0.966), worth roughly −0.01. On the venue side, note
their own history: they first measured 0.934, attributed +0.010 to a
contact-time bug and corrected to 0.9215 — a correction in the same direction
that could in principle have overshot, and their forensics estimator over all
218 bounces read 0.925 with CI95 up to 0.937.

**Adjudicator: a 30 cm drop test on the competition table.** Minutes of work,
model-free, and it settles the constant outright. Until then this session's
value is the one active in sim (§0) and the difference costs ≤1.4% of landing
error (§5).

### Racket

The normal channel is clean and the tangential channel is not:

| channel | residual |
| --- | --- |
| normal (restitution) | 0.81 m/s rms, median \|·\| 0.36 m/s = **7% of u_n** |
| tangential | 1.63 m/s rms — **2× worse** |

That ratio is the cost of unknown spin: R·\|ω\| at 50 rad/s is 1 m/s of
contact-point slip. So `e` is well determined and the tangential gain is not.

`e` per strike: median 0.6096, IQR [0.553, 0.697], u_n 2.2–8.8 m/s, 80 strikes.
Binned, against the venue's exponential form:

| u_n (m/s) | n | this session | venue exp |
| --- | --- | --- | --- |
| 2.0–4.0 | 19 | 0.697 | 0.695 |
| 4.0–5.5 | 31 | 0.625 | 0.633 |
| 5.5–7.0 | 26 | 0.597 | 0.586 |
| 7.0+ | 5 | 0.473 | 0.545 |

Robust (Huber) model comparison over 80 strikes: constant rms **0.1226**,
linear 0.0986, exponential 0.0994 — **the venue's F4 "kill constant-e" verdict
reproduces on an independent rig.** Fitted law
`e(u_n) = 0.9948·exp(−0.09418·u_n)`, CI95 g1 [0.863, 1.103], g2 [−0.1157,
−0.0670].

The venue's g2 = −0.0441 lies **outside** that CI: both rigs agree e falls with
impact speed, this one ~2× faster. The curves cross near u_n = 5 m/s (agreeing
to 0.7%) and diverge at the ends (u_n = 3: 0.750 vs 0.665; u_n = 7.5: 0.491 vs
0.545). Not a tracking artifact — the slope survives every quality subset
(clean racket extrapolation −0.1035, clean racket fit −0.0878, short gap
−0.0980) and both paddles (p1 −0.0806 n=27, p2 −0.1032 n=53), and
corr(e, u_n) = −0.60 dwarfs corr(e, any quality metric) = −0.07…−0.28. Most
likely a genuine rubber difference between sessions.

No paddle split: p1 0.6245 (n=27) vs p2 0.5983 (n=54) at matched u_n —
consistent with the venue's withdrawal of its per-paddle advice.

`a_t` / `b_t` are **unidentified** here, for two compounding reasons: no spin,
and collinearity at near-normal incidence (cos θ median 0.955) — fitting
(a_t, b_t) gives (0.21, 1.03), forcing b_t = 0 gives (1.10, 0), at the same rms.
Only `a_t + b_t·cos θ` is identified, exactly as at the venue.

## 5. Validation — landing prediction

`predict_check.py`, 80 strike→landing pairs (62 against an observed bounce):

| horizon | this session's YAML | venue YAML | what it exercises |
| --- | --- | --- | --- |
| H2 (~100 ms before landing) | **8.2 mm** | 8.6 mm | flight model only |
| H1 (measured out-state → landing) | **110 mm** | 108.7 mm | flight over a full arc |
| H0 (through the racket) | **901 mm** | 913 mm | contact + flight |

Two conclusions:

1. **The error ladder is entirely a spin story.** Over the last 100 ms spin is
   irrelevant and the model is good to 8 mm. Over a full arc, unknown outgoing
   spin costs 11 cm. Through the racket, unknown incoming *and* outgoing spin
   costs 90 cm.
2. **The two constant sets are indistinguishable** (≤ 1.4% on every horizon) on
   this session's own data. There is no measurable benefit to replacing the
   shipped constants, which is why §0 recommends not doing so.

## 6. Spin — RETRACTION: it IS measurable

> **This section previously reported that spin cannot be measured on this
> session. That was wrong, and it was the most consequential error in the
> report.** The ball is a rigid marker constellation; its rotation is
> recoverable. Everything below the retraction box is the corrected account.
> Sections 3–5 above were computed WITHOUT the spin channel and their
> spin-dependent parts (the H0/H1 landing errors in §5, the tangential blocks
> in §4) are therefore pessimistic pending a re-fit.

### 6.1 Why the first two tests were wrong

| test | reading | why it is invalid |
| --- | --- | --- |
| pairwise distances **by column index** | 6.2–6.9 mm | Motive relabels markers on a small spinning ball, so column *i* is not the same physical marker across frames |
| permutation-invariant **sorted-distance spectrum** | 2.2–2.8 mm vs a wandering Monte-Carlo null of 2.48 mm | the 28 distances span 11–40 mm, so adjacent sorted ranks differ by ~1 mm — comparable to the noise. Noise **swaps ranks**, and rank swapping inflates per-rank scatter regardless of rigidity. The test is biased toward "non-rigid" whenever the spectrum is dense relative to the noise |

The second one is the trap worth remembering: making a statistic
permutation-invariant by sorting does not make it robust, it just moves the
sensitivity into the ordering.

### 6.2 The valid test

Align **consecutive** frames: 2.8 ms apart the ball has barely rotated, so the
correspondence is unambiguous.

| take | consecutive-frame match rms |
| --- | --- |
| chuntan (pure bounce) | **0.91 mm** |
| Tui | 2.31 mm |
| zhengchang | 3.01 mm |
| zhengchang2 | 1.36 mm |
| xuan | **0.33 mm** |
| *control:* rigid + 1.5 mm marker noise | 3.13 mm |
| *control:* wandering points, redrawn each frame | **10.37 mm** |

All five takes sit far below the wandering null; chuntan and xuan sit below a
rigid control carrying 1.5 mm of noise, so per-marker noise is under ~0.7 mm.
Direct geometry confirms it: in the cleanest single frame all 8 markers lie on
one sphere of radius **18.6 mm with a radial spread of 0.77 mm**, spread over
the whole ball (angular separations 45.7°–152.5°).

Across all full-visibility frames the radial spread has median 2.56 mm against
that frame's 0.77 mm, i.e. **a fraction of frames carry ghost points** — which
is what collapsed the first template-averaging attempt (mean radius 15.9 mm and
max chord 31 mm on a constellation whose true max chord is 36 mm).

### 6.3 How orientation is recovered

`optitrack/ball_orientation.py::solve_orientation_chained`. Matching every frame
independently against a fixed template fails (5.0 mm pose rms, 17% of frames
solved) because the ball has rotated arbitrarily far from the seed frame and ICP
lands in a local minimum. Chaining consecutive clouds instead keeps every step
at ~1.5° . Absolute orientation is never needed — `spin_from_quats`
differentiates consecutive quaternions — so the chain is anchored at identity per
run and intra-run drift is irrelevant.

Two details that matter: partial visibility is the norm (all 8 markers co-occur
on ~11% of frames), so the observed subset must be centred against the **matched
template subset**, not the full template; and ghost points are rejected by a
free sphere fit before matching.

| take | frames solved | step rms | \|ω\| median | consecutive-ω correlation (x, y, z) |
| --- | --- | --- | --- | --- |
| chuntan | 56.4% | 0.70 mm | 1.6 rev/s | 0.41 / 0.26 / 0.31 |
| Tui | 43.2% | 1.29 mm | 5.1 rev/s | 0.54 / **0.82** / 0.64 |
| xuan | 37.7% | 0.66 mm | 1.7 rev/s | 0.58 / 0.77 / **0.81** |

The correlation column is the proof that this is physics and not fitted noise:
white noise would give ~0. And the ordering is right — the pure-bounce drop take
reads 1.6 rev/s against 5.1 rev/s for rally play.

**Aliasing limit ~22.9°/frame ≈ 22.9 rev/s at 360 Hz**, set by the smallest
marker separation (45.7° of arc). Faster spin relies on the tracking prediction
and is flagged per frame.

### 6.4 What this unlocks (not yet done)

`k_m` (Magnus), the table `a_t`/`b_t` grip block, the paddle tangential gain and
the whole spin-transfer model are all **fittable on this session** and are still
carrying inherited venue values in the shipped config. The §5 error ladder says
what is at stake: unmeasured spin was costing 110 mm over a full arc and 901 mm
through the racket.

### 6.5 Superseded: inferring spin from arc curvature

Before the orientation channel was found, spin was attacked indirectly
(`spin_recovery.py`) by fitting per-arc Magnus curvature `M = k_m·ω` and
calibrating `k_m` from the spin jump implied by the tangential velocity change at
a table bounce. That route genuinely does not work at this rig's ~10 mm arc-fit
noise floor — split-half repeatability 2.66 /s against a 0.56 /s signal, 31% of
`k_m` estimates negative, and a synthetic detection limit putting spin below
~10 rev/s out of reach. Those findings stand as a statement about **inferring
spin from curvature**, and they are why the aggregate-only Magnus evidence in §3
was all that seemed available at the time.

Spin was attacked indirectly anyway (`spin_recovery.py`): (A) per-arc Magnus
curvature `M = k_m·ω`, which needs no k_m, and (B) the spin jump implied by the
measured tangential velocity change at a table bounce, `|Δω| = |Δv_t|/(C·R)`,
which is in physical units — together these would have calibrated k_m
absolutely. All three diagnostics say the per-arc estimator is fitting noise:

- **split-half**: |M(first half) − M(second half)| median **2.66 /s** against a
  signal of 0.56 /s — repeatability 4.7× *worse* than the quantity measured;
- **bounce chain**: **31% of the k_m estimates come out negative**, IQR spans 0
  (k_m = 0.00158, CI95 [0.0004, 0.0032], against the venue's 0.00444);
- **synthetic detection limit** — inject a known Magnus vector into real arc
  geometries at the measured noise floor and re-fit (264 arcs):

  | injected \|M\| (1/s) | ≈ spin at k_m = 0.00444 | err / signal | per-axis corr (x, y, z) |
  | --- | --- | --- | --- |
  | 0.10 | 3.6 rev/s | 2.59 | 0.03 / 0.20 / −0.13 |
  | 0.30 | 10.8 rev/s | 1.16 | 0.08 / 0.47 / 0.40 |
  | 0.60 | 21.5 rev/s | 0.56 | 0.44 / **0.84** / 0.34 |
  | 1.00 | 35.8 rev/s | 0.37 | 0.43 / **0.87** / 0.76 |

So the limit is not "spin is invisible" but "**spin below ~10 rev/s is
invisible, and above ~20 rev/s it is recoverable only to ±40–56%**". The
best-recovered component is ω_y — the topspin/backspin axis for a ball
travelling along the table — which is exactly the component that shows up as
the apparent-g shift in §3. Note this synthetic uses *white* noise; the real
floor is correlated (the venue measured AR(1) with τ ≈ 60 ms), which is
substantially worse for curvature estimation, so the table above is
**optimistic** and the real-data split-half result is the binding one.

The cause is not camera precision but the coated ball plus correlated position
noise: a ~10 mm arc-fit floor over 0.25–0.66 s arcs leaves too little curvature
signal per event. The **aggregate** Magnus signal is real — it is what bends
apparent |g| from 9.70 to 10.45 across ~90-arc speed terciles (§3) — but per
event it is not recoverable.

Consequence: `k_m`, both `a_t` blocks and the entire spin-transfer model remain
**inherited from the venue fit**, and nothing here validates them.

## 7. What is still NOT measured, and the cheapest fixes

Ranked by value per unit effort. The first two need no new capture at all — they
are physical measurements on equipment already in the room, and both would let
every constant here be recomputed immediately.

1. **Weigh the ball.** ~10 seconds, and it is the single largest unbacked
   assumption in the file. `ball.mass = 0.0034 kg` is INHERITED from the venue
   session; nothing in this capture measured it. Both `k_d` and `k_m` are
   *acceleration* coefficients ∝ 1/m, so every aerodynamic number scales with
   it. This matters more than it did before the retraction in §6: the earlier
   assumption was that this ball was fully retro-reflective-coated (the venue's
   3.4 g), but it is actually a ball carrying **8 discrete markers**, which need
   not weigh the same. The 0.6% `k_d` agreement with the venue is reassuring —
   it implies similar m and C_d — but it is indirect. Record the ball type and
   mass in session metadata from now on.
2. **30 cm drop test on the competition table.** Minutes, model-free, and it
   settles `contact.table.e_eff` outright. Right now that key ships the VENUE
   value because this session's rally reconstruction was proven biased +0.044
   (§4) and the unbiased pure-bounce estimator has only n=20.
3. **Capture high spin at a higher frame rate, or change the marker layout.**
   The orientation solve aliases above **~22.9 rev/s** at 360 Hz, set by the
   smallest marker separation (45.7° of arc) — and the observed per-take maxima
   are 20.4–22.7 rev/s, i.e. sitting exactly at the ceiling, which is itself
   evidence that faster spin is being folded down. Competition serves run
   50–100+ rev/s and are **entirely outside coverage**. Either raise the rate or
   space the markers further apart (a larger minimum separation raises the
   limit proportionally).

Also worth recording next time, none of which was measured here:
`air.rho` (log temperature and pressure), the ball's `inertia_coeff` (2/3 is a
thin-shell assumption and it sets all spin transfer), and enough grazing
contacts to identify `mu_safety` — the friction cap never binds in this data, so
it is unidentified in both contact blocks.

Capture mechanics worth fixing:

4. **Check the exporter** — 11.5% of frames came out completely empty (§1).
5. **Capture high-u_n strikes (> 7 m/s)**: only 5 of 80 land there, and it is
   where the two rigs' `e(u_n)` curves diverge most.
6. **Get more spin-complete strikes.** Only 24 of 90 strikes have a measured
   incoming spin, because the racket occludes the ball right at contact — the
   whole paddle block rests on those 24 events. More camera coverage of the
   contact volume, or markers placed to survive racket occlusion, would widen it.
7. **A few deliberate double-bounce sequences in every session.** The
   pure-bounce take is what made the unbiased table-e estimator possible at all;
   rally takes yield zero usable bounce triples (§4).
