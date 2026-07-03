# Ball Physics v1 — Low-Speed Validation: Findings & Data-Processing Guide

> **SELF-CONTAINED HANDOFF DOCUMENT.** Written 2026-07-03 for an agent (or human) with **no prior
> context** — everything needed is in this file plus the referenced repo paths. Produced on the
> team RunPod by a 10-agent research workflow (Sony Ace paper deep-read, low-speed aerodynamics
> literature survey, v0-branch code audit, quantitative failure-mode analysis, mocap protocol
> design, reward-integration design; every load-bearing number adversarially re-verified by
> independent agents). Raw per-agent reports: `docs/research/ball_physics_2026-07-03/`.

---

## 0. Context — read this first

**Project**: HOPE — Agibot A3 humanoid table tennis. RL training in Isaac Lab
(`hope_training/whole_body_tracking`), sim2sim gate in MuJoCo, deploy via ONNX → C++ runner.
Mocap rig: ChingMu / "Avatar Pro" software, streamed over VRPN at a nominal 300 Hz.

**Goal of this track** (owner: yikang, per `docs/NOW.md`): build a validated ball physics model
(trajectory + spin) for the simulator, verified on **low-speed** strike data (roughly 2–8 m/s ball
speed, 0–60 rev/s spin — amateur play, NOT the 15–30 m/s / 100+ rev/s pro regime), then use the
model + predicted trajectories to add rewards that make the robot (a) land the ball on the table
and (b) produce spin.

**What already exists**:

| What | Where | State |
| --- | --- | --- |
| v0 physics implementation (flight, table/paddle contact, landing predictor, PACE-style rewards, torch + C++ mirror) | branch `origin/ball-physics-realistic`, under `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/physics/` + `configs/ball_physics.yaml` | Code sound, architecture reusable; **constants fit on a different rig/table/ball**; paddle block never fit on racket data at all |
| v0 fitting/analysis code | **Mac-local only**: `/Users/yyk956614/Desktop/Hope/Record` — `analysis/flight_model/simulator.py`, `analysis/contact_model/spin_equation.py`, `Record/qinghua/flights/PHYSICS_FIT.md`, `bounce_model/BOUNCE_MODEL.md` | NOT in the repo. The repo test `hope_training/whole_body_tracking/tests/test_ball_physics_vs_record.py` consumes it via `$RECORD_DIR` and **silently skips** when absent |
| NEW venue dataset (the input to this guide) | collected 2026-07-03 at the venue: Avatar Pro rigid bodies on **both the ball and the racket** (first time racket data exists) | To be processed — that is what Section 4 is for |

**The verdict in one paragraph** (details in Sections 1–3): the Sony Ace model family is the right
fit for this project and v0 already implements most of it. The validation question is NOT "does
Ace work at low speed" in general — it is three specific, falsifiable model-form divergences:
(**R1**) v0 uses a constant Magnus coefficient where Ace and CFD data say the Magnus force
**saturates** at high spin-ratio — and low-speed play lives exactly in the high-spin-ratio region
(predicted v0 landing error 6–41 cm there); (**R2**) v0's paddle restitution/tangential constants
were **never fit on real racket data** and published rubber measurements show strong velocity
dependence (±0.05 restitution error ≈ 15 cm landing shift); (**R3**) v0's table model is
grip-only with a deliberately disabled friction cap, unvalidated in the slip regime (shallow
incidence / high spin). Each has a kill/pass test in Section 4.

---

## 1. The model under test (v0) — exact equations and fitted values

All from `origin/ball-physics-realistic` (read with
`git show origin/ball-physics-realistic:<path>`). Ball: m = 2.7 g, R = 0.020 m, hollow-sphere
inertia I = c·mR² with c = 2/3.

**Flight** (`…/table_tennis/physics/flight.py:23-29`, RK4 integrator `rk4_step` lines 32-49):

```
a = g − k_d·|v|·v + k_m·(ω × v)          # ω = PHYSICAL spin, held constant in flight
```

**Contact — one impulse model for both table and paddle**
(`…/physics/spin_contact.py:37-81`; v_r = surface contact-point velocity, n = surface normal
auto-oriented against approach):

```
u    = v⁻ + ω⁻×(−R·n) − v_r              # contact-point relative velocity
u_n  = (u·n)·n ;  u_t = u − u_n ;  cosθ = |u_n| / √(|u_t|²+u_n²)
s    = clip( (a_t + b_t·cosθ)·|u_t| ,  0 ,  μ_safety·(1+e_eff)·|u_n| )
Δv_t = −s·û_t ;  Δv_n = −(1+e_eff)·u_n·n ;  Δω = −(1/(c·R))·(n × Δv_t)
```

**Fitted constants** (`configs/ball_physics.yaml`, provenance in its comments):

| Param | Value | Provenance / caveat |
| --- | --- | --- |
| k_d | 0.1222 1/m (C_d≈0.438) | HAND-PICKED between OptiTrack-markered 0.136 and C3D 0.107; rig/ball-specific, "re-fit per setup" |
| k_m | 0.0042 | OptiTrack recal 2026-06-30, **side-spin channel only** ("topspin channel is g-degenerate"), 95% CI [0.0035, 0.0049] = ±17% |
| g | 9.81 (frozen) | rigs read 9.76–9.82 incl. ~0.15° tilt — measurement artifact, always freeze g |
| table e_eff | 0.908 | 101-bounce median, narrow impact-speed band (~2.5–3.6 m/s) — velocity dependence UNTESTED |
| table a_t / b_t | 0.369 / 0.0 | grip regime only; a_t≈theoretical full-grip 0.4 — internally consistent |
| table μ_safety | 2.0 | **non-physical placeholder** — yaml's own words: "qinghua only sampled the grip regime; if shallow-angle SLIP matters, lower to a real friction coefficient" |
| paddle e_eff / a_t / b_t / μ | 0.4632 / 1.2048 / 0.3039 / 2.5 | **never fit on racket data** ("no racket hits in this dataset", `docs/ball_physics.md` on the branch); zero provenance comments — weakest block in the model |

Known v0 implementation gaps (fix before trusting sim rollouts; from the code audit):
1. Paddle contact passes racket **body linear velocity** as v_r — omits ω_racket×r (1–3 m/s at
   the face during a swing). Fit the paddle in a frame that includes it, or add the term.
2. Paddle hit detection = sphere of 0.075 m around the racket **body origin** — no face geometry.
3. Landing predictor ignores the net and second bounces (fine for rewards, not for validation).
4. Deploy-side `hope_ws/src/hope_planner/.../constants.py` still carries an OLD incompatible drag
   model (k = 0.5 **s/m** — different units!). Reconcile after v1 lands.

---

## 2. Sony Ace's model (Nature 652:886-891, Methods p.7-8) and where it diverges

Ace flight: same structure, but **c_d = 0.55 constant** and the Magnus coefficient is
spin-ratio-dependent:

```
c_M = 0.1·‖v‖/(r·‖ω‖) − 0.001
```

Algebraically this makes the Magnus **force** magnitude ≈ independent of spin magnitude
(lift coefficient C_L ≈ 0.26 saturated) — spin sets the force *direction*, not its size.
v0's constant k_m instead implies C_L = 0.752·SR, linear and unsaturated (SR = R·‖ω‖/‖v‖).

Ace table bounce: velocity-dependent restitution **ε(v_z) = 0.98 − 0.02·v_z⁻**, μ = 0.25, and an
explicit **sliding/rolling switch** (α from a ν_s test). Ace paddle: velocity-dependent
**ε = γ₁·e^{γ₂|v_z′|}** (fit on game data) + a small residual MLP (corrects velocity/spin error
by ~4% on average). Ace measures spin with event-camera gaze-control units watching the ball logo
(±4 rev/s, 400–700 Hz) — NOT mocap.

**Literature adjudication of the divergences** (full citations in the appendix):

- **Magnus saturation — Ace is right.** LES-CFD covering exactly our regime (Ito & Kamijima 2025,
  2.5–20 m/s × 15–90 rev/s): C_L plateaus at ≈0.31–0.42 once SR ≳ 0.75 while SR grows 6×; the
  effective ω×v-coefficient (v0's k_m) **falls ~5×** from SR 0.75 → 4.5. Low-speed play spans
  SR 0.5–3.8, squarely in saturation. Simulated landing gap, constant-k_m vs saturating law:
  **5.9 cm (2 m/s, 40 rev/s) → 16 cm (3 m/s, 60 rev/s) → 41 cm (5 m/s, 60 rev/s)**.
- **Rubber velocity dependence — real.** 8,194-bounce dataset across 10 rubbers (arXiv
  2604.11349): racket COR **falls ≈0.017–0.021 per m/s** of normal impact speed (Δ≈0.15 over
  2→12 m/s); tangential gain depends on both v_s and v_z. Independent confirmation: Rinaldi 2016.
- **Table restitution — safe-ish at low speed.** Ball shell buckling (sharp COR drop) starts at
  impact speeds ≳5 m/s; below that e is only weakly velocity-dependent. Table-bounce normal
  speeds in low-speed play are 1–4 m/s → v0's constant 0.908 MAY pass (test F3 decides).
- **Inverse Magnus — irrelevant here.** Needs Re ~1e5+; our 2–8 m/s span is Re 5.3e3–2.1e4.
  Constant C_d is defensible to ±10–15% in this band (no drag crisis until ~75+ m/s).
- **Cautionary mirror-image from Sony's own development** (secondary reporting): their
  low/mid-speed-fit drag overestimated at 19.6 m/s → simulated returns overshot the real table →
  they had to recalibrate per speed regime. Consequence for us: **record the (Re, SR) coverage of
  every fit and never silently extrapolate outside it.**

**Sensitivity ranking at low speed** (RK4 grid, 2–8 m/s × 10–60 rev/s; full table in
`docs/research/ball_physics_2026-07-03/failureModes.md`): paddle e_eff dominates
(±0.05 → ~15 cm landing), then Magnus model-form (above), then table e_eff (±0.05 → 3–10 cm
second-bounce), then tangential gain; k_d errors are ≤1.2 cm everywhere below 8 m/s. At 2–3 m/s
flight is nearly ballistic (total aero ≤ 20% of gravity) — **flight-parameter identification is
EASIEST at low speed; the contact models are where the risk lives.**

---

## 3. Falsification table — what the new data must decide

Priority-ordered. "Kill" = the v0 assumption is wrong, apply the listed fix.

| # | Assumption under test | Statistic to compute | KILL if | PASS if | Fix on kill |
| --- | --- | --- | --- | --- | --- |
| **F2** | Constant k_m (no Magnus saturation) | per-flight k̂_m = a_lat/(ω·v), sidespin channel, binned by SR ∈ [0.2, 3] | k̂_m at SR>1.5 bins < 80% of the SR<0.5 value | bin medians flat within ±15% | replace with Ace's c_M(SR) or C_L(SR)=0.752·SR/(1+0.752·SR/0.55) — a ~5-line change in `flight.py` (+ its C++ mirror + yaml keys) |
| **F4** | Constant paddle e_eff | e vs \|u_n\| from Set-D strikes (racket rigid body gives u_n per hit) | trend > ±0.05 across u_n 2–8 m/s | within ±0.025 of constant | e(u_n) linear or Ace exponential γ₁e^{γ₂\|u_n\|} |
| **F3** | Constant table e_eff | e = −v_n⁺/v_n⁻ vs v_n⁻ over 0.5–6 m/s (drops from several heights) | \|de/dv_n\| > 0.01 per m/s, or a break above ~5 m/s | flat within ±0.025 over v_n 1–6 m/s | Ace's ε(v_z) = a − b·v_z |
| **F5** | Grip-only tangential law (Coulomb cap disabled) | residuals Δω⁺, Δv_t⁺ vs \|u_t\|/\|u_n\| | residuals ≈0 below ratio 1 but growing beyond ratio ≈1.5 | RMS Δω⁺ < 3 rev/s in every ratio bin | fit real μ from shallow-incidence trials, set μ_safety to it (slip branch activates automatically via the existing clip) |
| F1 | Constant k_d | per-flight k̂_d binned by speed 2–8 m/s | monotonic trend; any bin off > ±20% | bins within ±10% of global | k_d(Re) lookup (unlikely needed) |
| F6 | No inverse Magnus | sign of lateral accel vs ω̂×v̂, worst cell = highest Re × lowest SR | any sign-reversed bin | all consistent | (expected pass) |
| F7 | No in-flight spin decay | ω(t) from ball rigid body over ≥0.4 s flights | decay > 10% per flight | < 5% per flight | exponential ω decay term (cheap) |
| F8 | Instantaneous paddle contact | outgoing-direction residual vs racket angular velocity | correlated, > 2° effect | uncorrelated, < 2° | fold into paddle refit (rotation during ~1–2 ms dwell) |

---

## 4. AGENT GUIDE — processing the venue dataset

> Written for an agent running **on the Mac**, with the new venue recordings and the existing
> `/Users/yyk956614/Desktop/Hope/Record` workspace (which contains the v0 fitting precedents:
> `analysis/flight_model/simulator.py` = RK4 oracle, `analysis/contact_model/spin_equation.py`
> = contact oracle, `Record/qinghua/flights/PHYSICS_FIT.md` + `bounce_model/BOUNCE_MODEL.md` =
> how the 2026-06-30 fits were done). Adapt paths to where the new data actually lives; the
> METHOD below is the contract. Work in a new folder, e.g. `Record/<venue>/2026-07-03/analysis/`.

### Stage 0 — Data QA (do NOT skip; the last dataset failed two of these)

1. **Sampling rate**: compute median and p90 inter-sample dt per stream. Nominal is 300 Hz
   (dt = 3.33 ms). The previous in-repo capture was actually **~44 Hz median with 42 ms p90
   gaps** despite the 300 Hz nominal. If the new data is not ≥250 Hz effective, flag it and
   check whether the raw `/vrpn_mocap/**/pose` topics were bagged (the relay's `/ball/point`
   output is position-only AND may drop); fitting still works at lower rates but spin-from-
   orientation does not.
2. **Units and frame**: the previous export was in **millimeters** with a z-offset inconsistent
   with the table frame. Establish: units (plot a known geometry — table length must be 2.740),
   axis convention, and where z=0 is (table surface vs floor; table top is 0.76 m above floor).
   Use the per-session static captures (table corners / net posts) to build the table frame:
   origin at table center, +Z up, +X along the table's long axis. Transform everything into it.
3. **Gravity sanity fit**: take 3+ clean no-spin free-flight arcs, fit a = g with drag off;
   require g = 9.81 ± 0.05 and tilt < 0.2°. If it fails, the frame/scale is wrong — stop and fix.
   (Precedent: rigs read 9.76–9.82 with ~0.15° tilt; always FREEZE g = 9.81 in later fits.)
4. **Time sync**: if racket and ball come from separate streams, verify with the
   racket-taps-static-ball sync event (if recorded) or the contact discontinuity itself.
5. **Ball mass**: get the recorded taped-ball mass m_taped (should be in session metadata;
   clean ball is 2.70 g). All fitted k values scale as 1/m: report both k_taped and
   k_clean = k_taped · (m_taped / 0.0027). If an A/B (minimal-tape vs full-tape) set exists,
   fit both to bound the residual surface-roughness effect on C_d.

### Stage 1 — Segmentation

1. Detect contacts as velocity discontinuities: sliding-window shooting fits (below) on either
   side of a candidate; a contact is where |Δv| between windows exceeds ~1 m/s in < 10 ms.
2. Cut each recording into **segments**: free-flight arcs, table bounces (pre-arc + post-arc),
   racket strikes (pre-arc + post-arc + racket track).
3. **Exclusion windows** around each contact: drop ball samples ±3 frames (10 ms) for table
   contacts, ±5 frames (17 ms) for racket contacts (marker-merge and blur risk).
4. Label each segment with its trial-set id from the session metadata:
   A = no-spin free flight, B = spin free flight, C1 = vertical drops, C2 = oblique no-spin
   bounces, C3 = spin bounces, D = racket strikes. If metadata is missing, classify by
   kinematics (drop = near-vertical, etc.) and say so in the report.

### Stage 2 — Fitting (ordered; each stage FREEZES everything before it)

**Method — RK4 shooting, never finite-difference acceleration.** Per segment, free parameters
(p₀, v₀, ω₀) + the shared physics constants; integrate with RK4 (h = 0.5–1 ms; reuse/port
`simulator.py`) and minimize Huber loss on positions. Reason: double-differencing 1 mm noise at
300 Hz gives σ_a ~ O(10²) m/s² while the drag+Magnus signal is 3–4 m/s² — invisible per-frame,
well-observable in a ≥0.3 s shooting fit (drag accel at 5 m/s ≈ 3.05 m/s²; Magnus at 30 rev/s,
5 m/s ≈ 3.96 m/s²).

**Spin**: primary path is **physics inversion from the position track** — fit the two ω
components ⊥ v per flight segment (lateral Magnus deflection ≈ ½·k_m·ω·v·t² ≈ 17 mm per
10 rad/s at v=4 m/s, t=0.4 s, vs ~1 mm mocap noise → resolution < 1 rev/s). The along-velocity
(rifle) spin component is unobservable in flight — accept that. Use mocap 6-DoF orientation for
spin ONLY below ~20–30 rev/s (at 300 Hz, 30 rev/s = 36°/frame — above that the rigid-body solve
aliases; check orientation streams for flips before trusting them). The **topspin channel is
g-degenerate**: always freeze g and prefer the sidespin (lateral) channel for k_m. If slow-mo
phone clips of the half-blackened ball exist, use them as ground-truth spot checks.

Order of fits:

1. **k_d ← Set A** (no-spin flights). Report per-speed-bin medians (feeds F1) and the global
   value with leave-one-trial-out scatter.
2. **k_m ← Set B, sidespin channel** (with k_d frozen). Bin k̂_m by SR (feeds F2 — this is the
   headline test). Cross-check: the previous rig's fit was 0.0042, CI [0.0035, 0.0049]. A venue
   value outside ~[0.003, 0.006] at LOW SR means a rig/frame problem, not physics.
3. **Table e_eff ← C1** (multi-height drops). Per-impact e = −v_n⁺/v_n⁻; scatter vs v_n⁻
   (feeds F3). Report the constant fit AND the linear fit e = a − b·v_n with CIs.
4. **Table a_t, b_t, μ ← C2 + C3** (with e frozen). C2's shallow-incidence (~20°) trials probe
   slip: fit μ from trials where the Coulomb cap binds; check residual pattern vs |u_t|/|u_n|
   (feeds F5). b_t stays 0 unless residuals show a cosθ trend.
5. **Paddle e_eff, a_t, b_t, μ ← Set D** (everything else frozen). Per strike: racket pose,
   velocity, and face normal from the racket rigid body — **bridge the ±30 ms occlusion window
   around contact by extrapolating a constant-acceleration (or cubic-spline) fit over
   [−120, −15] ms; never use post-contact frames near impact (blade vibration)**. Include the
   racket's ω×r contact-point correction in v_r. Incoming ball ω⁻ from the pre-contact flight
   fit. Report e vs |u_n| (feeds F4) plus the tangential fit. This is the first-ever real fit of
   the paddle block — expect it to move a lot from {0.463, 1.205, 0.304}.

### Stage 3 — Run the falsification table

Produce one plot + one verdict (KILL / PASS + confidence) per row of Section 3, in priority order
F2, F4, F3, F5, F1, F6, F7, F8. Where a test KILLS an assumption, fit the listed replacement form
and report both.

### Stage 4 — Held-out validation

Stratified 70/30 split **by trial** within every set. Fit on 70%, report on 30%:

1. Free flight: propagate from the first 60 ms of state → position error at +0.4 s:
   **median ≤ 25 mm, p95 ≤ 60 mm** (v0's precedent on its own rig was 22 mm).
2. Bounce: outgoing speed ≤ 5%, direction ≤ 3°, next-apex height ≤ 10 mm; ω⁺ ≤ 15% where
   observable.
3. **Strike → landing (the money metric)**: from measured pre-contact racket + ball state,
   predict the first-bounce landing point. **Median ≤ 0.10 m, p90 ≤ 0.25 m, on/off-table
   classification ≥ 90%**, and vertical error at the net plane ≤ 30 mm.
   Rationale for 0.10 m: mid-training policy landing dispersion is ≥0.3 m; model error ≤ σ/3
   preserves policy ranking by predicted land-rate, and 0.10 m ≈ 7% of the 1.37 m half-table.
4. If (3) fails while (1)+(2) pass → the deficit is the paddle model (expected — first real fit);
   iterate ONLY on Set D + the F4/F8 fixes rather than touching flight params.

### Stage 5 — Deliverables

1. **`ball_physics_venue.yaml`** — same schema as `configs/ball_physics.yaml` on the
   `ball-physics-realistic` branch (keep the comment style: every value gets provenance, CI, and
   the dataset's (speed, SR, v_n) coverage). If F2 killed constant k_m, add the new Magnus form's
   keys and note the required `flight.py`/C++ change.
2. **`FIT_REPORT.md`** — per-stage results, the 8 falsification verdicts with plots, held-out
   metrics vs the acceptance thresholds, and an explicit **validity envelope** (the (Re, SR, v_n)
   box the fits cover — nothing outside it is validated; remember Sony's high-speed extrapolation
   failure).
3. **Port the fitting code into the repo** (e.g. `hope_training/ball_physics_fit/`) so the team
   is no longer dependent on the Mac-local Record workspace; make
   `test_ball_physics_vs_record.py` fail loudly (not skip) when the oracle is missing.
4. Push the results branch; the RunPod side then wires the new yaml into Isaac/MuJoCo and starts
   the Tier-1 reward work (Section 5).

---

## 5. What the validated model unlocks (so the fitting effort has a target)

**Tier 1 — virtual-ball at-strike reward** (the immediate consumer, ~2–4 days of work on the
training side, no obs-contract change): in the existing racket-target task, at the exact-strike
frame, feed a sampled virtual incoming ball through the **paddle contact model** applied to the
achieved racket FK state, integrate the **flight model** (coarse RK4, h=10 ms — measured
~5.4 ms/step CUDA-graphed at 4096 envs ≈ +2–5% iteration time) to the table plane, and reward:
net-clearance kernel (w≈20) + landing-on-opponent-half kernel gated on net clearance (w≈30) +
landing-spin kernel à la Ace's w_s term (w≈5→10, gated on a legal landing). This directly
replaces the brittle "match a commanded racket velocity vector" objective with "produce a ball
that lands well" — velocity becomes instrumental. Model error budget for this to work is exactly
the Stage-4 acceptance threshold (≤0.10 m median).

**Tier 2** — real simulated ball in Isaac (the v0 env machinery), new obs contract (ball position
+ deploy-honest derived velocity), NEW task name. Gate: Tier-1-trained policy must transfer
zero-shot to the Tier-2 env before paying this cost. **Tier 3** — match play / self-play with
Ace-style skill conditioning (y_desired, [w_p, w_s]). Details + code pointers:
`docs/research/ball_physics_2026-07-03/rewardDesign.md`.

---

## Appendix — key sources

- Sony Ace: *Outplaying elite table tennis players with an autonomous robot*, Nature 652:886–891
  (2026). Local copy: `papers/s41586-026-10338-5.pdf` (Methods p.7-8 has all model equations).
- Ito & Kamijima, *Estimation of Aerodynamic Properties of a Spinning Table Tennis Ball Using CFD
  Analysis*, Trans. JSST 17(1):25–31, 2025 — C_D/C_L/C_M grids over exactly 2.5–20 m/s ×
  15–90 rev/s (the Magnus-saturation evidence).
- *Learning Racket-Ball Bounce Dynamics Across Diverse Rubbers*, arXiv 2604.11349 — 8,194 bounces,
  10 rubbers; racket COR velocity dependence ≈ −0.02 per m/s.
- Miyazaki et al., EJP 38:024001 (2017) — TT-ball lift crisis (high-Re only, not our regime).
- MPI gray-box, arXiv 2305.15189 — same ODE family with per-trajectory EKF coefficients (evidence
  that global constants underfit); DeepMind robotic TT, arXiv 2408.03906 — spin-sign-switched
  paddle contact params.
- Rémond/Rinaldi/Géminard PRE 106/107 — shell buckling above ~5 m/s; oblique bounce roll/slide
  regimes. Nakashima et al. 2010 — canonical rubber spin-transfer model.
- PACE, arXiv 2509.21690 §IV-B — the at-contact predicted-landing reward this feeds.
- Raw research reports (10 agents, incl. the three adversarial verifications):
  `docs/research/ball_physics_2026-07-03/`.
