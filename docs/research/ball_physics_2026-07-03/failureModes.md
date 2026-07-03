All computations complete. Composing the final analysis.

# Low-speed failure-mode analysis of the v0/Ace ball model family

## 0. Model under test (FACTS, from origin/ball-physics-realistic)

- Flight: `a = g − k_d|v|v + k_m(ω×v)`, ω **constant in flight** ("no aero spin decay") — `…/table_tennis/physics/flight.py:1-6,23-29`.
- Contact (table & paddle): `s(θ)=clip((a_t+b_t·cosθ)|u_t|, 0, μ(1+e)|u_n|)`, `dv_t=−s·û_t`, `dv_n=−(1+e)u_n·n`, `dω=−(1/(cR))(n×dv_t)` — `…/physics/spin_contact.py:9-21,58-80`.
- Constants (`configs/ball_physics.yaml`): k_d=0.1222 1/m (Cd≈0.438), k_m=0.0042 **with fitted 95% CI [0.0035,0.0049] = ±17%**, g=9.81; table e=0.908 (101-bounce median), a_t=0.369, b_t=0, μ_safety=2.0 with the yaml's own warning "**qinghua only sampled the grip regime**; if shallow-angle SLIP matters later, lower this to a real friction coefficient"; paddle e=0.4632, a_t=1.2048, b_t=0.3039, μ=2.5; ball m=2.7 g, R=0.020 m, c=2/3.
- Cross-check: k_d = ½ρC_dA/m = 0.5·1.20·0.438·1.2566e-3/0.0027 = **0.1223 1/m** ✓ (so yaml Cd≈0.438, mid Cd 0.4–0.5 range). Implied Magnus lift slope: C_L = (k_m·m/(½ρAR))·S = **0.752·S** where S = Rω/v (computed; i.e. constant k_m ≡ C_L linear in spin ratio with slope 0.75, no saturation).

## 1. Sensitivity grid (FACTS — RK4 h=5e-4, same scheme as flight.py; launch z=0.30 m above table, topspin, angle solved so nominal first bounce = 1.5 m; `*` = 1.5 m infeasible, max-range flight used)

Feasibility fact first: from 0.3 m height, a 1.5 m first-bounce flight requires **v₀ ≥ ~4.3 m/s** (max range: 0.60 m @2 m/s, 1.05 m @3 m/s, 1.46 m @4 m/s). Ballistic check: R_max=(v²/g)√(1+2gh/v²); v=2: (4/9.81)√(1+1.47)=0.64 m ✓. So "low-speed 1.5 m flights" only exist ≥~4.5 m/s; 2–3 m/s flights are intrinsically ≤1 m and lofted (θ 33–43°, T 0.34–0.47 s).

| v₀ m/s | f rev/s | L m | T s | ΔX k_d±10% | ΔX k_m±10% | ΔX ω±5rev/s | v_n⁻/|u_t| m/s | ΔX₂ e±0.05 | Δω⁺ a_t±10% | ΔX₂ a_t±10% | drag/g | Magnus/g | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 10 | 0.60* | 0.37 | 0.2 cm | 0.2 cm | 1.0 cm | 2.6/0.2 | 3.3 cm | 0.1 rev/s | 0.3 cm | 0.05 | 0.05 | 0.63 |
| 2 | 30 | 0.56* | 0.35 | 0.2 | 0.6 | 0.9 | 2.7/2.5 | 5.0 | 1.1 | 3.5 | 0.05 | 0.16 | 1.88 |
| 2 | 60 | 0.50* | 0.34 | 0.2 | 1.0 | 0.8 | 2.8/6.5 | 6.7 | 2.9 | 8.3 | 0.05 | 0.32 | 3.77 |
| 3 | 10 | 1.05* | 0.47 | 0.8 | 0.4 | 2.1 | 3.0/0.7 | 4.4 | 0.3 | 1.1 | 0.11 | 0.08 | 0.42 |
| 3 | 30 | 0.97* | 0.45 | 0.6 | 1.1 | 1.8 | 3.1/2.0 | 6.1 | 0.9 | 3.1 | 0.11 | 0.24 | 1.26 |
| 3 | 60 | 0.87* | 0.41 | 0.5 | 1.8 | 1.5 | 3.3/6.1 | 7.8 | 2.7 | 8.5 | 0.11 | 0.48 | 2.51 |
| 5 | 10 | 1.50 | 0.34 | 1.1 | 1.1 | 5.3 | 2.6/2.7 | 5.8 | 1.2 | 3.6 | 0.31 | 0.13 | 0.25 |
| 5 | 30 | 1.50 | 0.35 | 1.2 | 2.8 | 4.6 | 3.0/0.06 | 7.4 | 0.03 | 0.1 | 0.31 | 0.40 | 0.75 |
| 5 | 60 | 1.50 | 0.37 | 1.2 | 4.6 | 3.8 | 3.6/4.4 | 9.2 | 1.9 | 5.9 | 0.31 | 0.81 | 1.51 |
| 8 | 10 | 1.50 | 0.21 | 0.8 | 1.1 | 5.6 | 2.5/5.3 | 7.0 | 2.3 | 6.1 | 0.80 | 0.22 | 0.16 |
| 8 | 30 | 1.50 | 0.21 | 0.8 | 2.9 | 4.8 | 2.9/2.6 | 8.1 | 1.2 | 3.0 | 0.80 | 0.65 | 0.47 |
| 8 | 60 | 1.50 | 0.21 | 0.8 | 4.9 | 4.1 | 3.4/1.4 | 9.5 | 0.6 | 1.6 | 0.80 | 1.29 | 0.94 |

(ΔX = first-bounce shift; ΔX₂ = second-bounce shift after perturbing the table contact at the nominal impact state; Δω⁺ = post-bounce spin error.)

Supporting arithmetic:
- **k_d ±10%**: speed loss along flight is dv/dx=−k_d v ⇒ fraction 1−e^(−k_d L)=1−e^(−0.183)=17% over 1.5 m; a ±10% k_d error perturbs speed only ±1.7% ⇒ ≤1.2 cm everywhere. Negligible.
- **k_m ±10%**: Δa=0.1·k_m·ω·v; at 8 m/s, 60 rev/s: 0.1·0.0042·377·8=1.27 m/s² over T=0.21 s ⇒ ½·1.27·0.21²=2.8 cm vertical, amplified by flat descent (÷tan∼23°) → 4.9 cm ✓ table. **Note the yaml's own k_m CI is ±17%, i.e. ~±8 cm at 8 m/s/60 rev/s today.**
- **e ±0.05 (table)**: post-bounce hop R₂≈2v_x⁺·e·v_n/g; ΔR₂/R₂=Δe/e=5.5% ⇒ 3–10 cm; apex shift = e·v_n²·Δe/g = 0.908·9·0.05/9.81 = **4.2 cm at v_n=3** (apex 0.38 m).
- **a_t ±10%**: Δω⁺=0.1·a_t|u_t|/(cR)=2.77|u_t| rad/s = **0.44|u_t| rev/s**; ΔX₂≈Δv_t⁺·t_hop=0.0369|u_t|·(2e·v_n/g). Vanishes when contact rolls (|u_t|=|v_t−Rω|≈0, e.g. 5 m/s/30 rev/s row).

**Ranking (JUDGMENT):** For post-table-bounce prediction, **e_eff dominates at all speeds** (3–10 cm per ±0.05), with tangential-gain error second when |u_t| is large. Pure-flight parametric errors *shrink* at low speed (k_d ≤0.8 cm, k_m ≤1.8 cm at 2–3 m/s) and grow at high speed/spin (k_m ~5 cm). For the *deploy-relevant landing reward* (paddle hit → flight → first bounce), the paddle e=0.463 is the biggest lever: ±0.05 is ±11% relative; outgoing-speed error 0.05·|u_n|≈0.25 m/s at u_n=5, and dR/dv≈2R/v=0.6 s ⇒ **~15 cm landing shift** — larger than any flight parameter (arithmetic estimate, not simulated).

## 2. Regime shifts (FACTS)

- drag/gravity = k_d v²/g: 5% @2, 11% @3, 31% @5, 80% @8 m/s; equality at v=√(g/k_d)=**8.96 m/s** (= terminal velocity; matches known TT ~9 m/s).
- Magnus/gravity = k_m ω v/g = 1 on the hyperbola ω·v=g/k_m=**2336 rad·m/s²** (e.g. 60 rev/s ⇒ v=6.2 m/s). Grid values 0.05–1.29.
- Magnus/drag = k_m ω/(k_d v): equality at **ω[rev/s]=4.63·v[m/s]** — below that speed Magnus *out-ranks* drag. At 3 m/s/30 rev/s Magnus is 2.2× drag.
- So at 2–3 m/s, ≤30 rev/s: total aero ≤20% of g, parametric ±10% aero errors ≤2% of g ⇒ **flight is ballistic-ish and parameter errors shrink to ≲2 cm — GOOD** for low-speed validation. Magnus *matters most* at high ω·v (top-right of grid), where ±10% k_m ≈ 5 cm ≈ the ±5 rev/s spin-measurement effect.
- **The trap (JUDGMENT, quantified):** spin ratio S=Rω/v explodes at low speed: S=1.3–3.8 for 2–3 m/s at 30–60 rev/s. Constant k_m ≡ C_L=0.752·S with no saturation ⇒ implied C_L up to 2.8, aerodynamically implausible (sphere C_L data — from-memory literature — is quasi-linear only to S≈0.3–0.5, saturating ~0.4–0.6). Simulated gap linear-k_m vs a saturating law C_L=0.752S/(1+0.752S/0.55): **5.9 cm (2 m/s/40), 9.0 (2/60), 16.2 (3/60), 41 cm (5/60, S=1.5), 38 cm (8/60, S=0.94)**. This *model-form* risk dwarfs every parametric sensitivity precisely where parametric sensitivity is smallest. Whether the fitted k_m already averages-in saturation depends on the S-distribution of the qinghua fit flights — unknown from the yaml; **must be checked in the new dataset**.
- Re = vD/ν spans 5.3e3–2.1e4 across 2–8 m/s — far below the smooth-sphere drag crisis (~2–3e5) ⇒ constant Cd (hence k_d) is *safe* in this regime (JUDGMENT).

## 3. Contact physics at low incident speed (FACTS = arithmetic/code; priors marked JUDGMENT)

- **Contact time**: linear-shell estimate t_c=π√(m/k); t_c≈1 ms ⇒ k≈mπ²/t_c²≈27 kN/m (plausible for the shell). Hertz scaling t_c∝v^(−1/5): halving speed lengthens contact only ~15%. Ball travels v·t_c≈2–8 mm during contact — impulse (instantaneous) treatment is geometrically fine for the table. For the **paddle**, the racket moves 5–20 mm and its normal rotates ~1–3° (stroke ω~10–30 rad/s × 1–2 ms rubber dwell) during contact — second-order but measurable in fast strokes (JUDGMENT).
- **Velocity-dependent restitution — most suspect normal assumption.** Viscoelastic dissipation ⇒ e falls as v_n rises; celluloid shells additionally buckle above v_n≈5–6 m/s with a sharp e drop (JUDGMENT from-memory; buckling mostly a high-speed-play concern). e=0.908 is a *median over 101 bounces* with unreported v_n spread; my grid's lofted impacts all land in a narrow v_n=2.5–3.6 m/s band — so the current fit may look clean while hiding a slope. Signature: scatter e=−v_n⁺/v_n⁻ vs v_n⁻ shows a negative trend; a slope of just −0.01/(m/s) accumulates Δe=0.05 across a 5 m/s span = the full 3–10 cm sensitivity of §1.
- **Paddle e=0.463 constant — most suspect overall.** Sponge+topsheet is strain-stiffening and thickness-limited: e generically varies with |u_n| (JUDGMENT). Per §1 this parameter carries ~15 cm/0.05 of landing leverage; signature: e vs |u_n| from racket-rigid-body strike data.
- **Tangential law**: fitted table a_t=0.369 ≈ theoretical full-grip gain 1/(1+1/c)=0.4 (hollow-sphere effective tangential mass m/2.5) — internally consistent with "grippy, near no-slip" (yaml). Failure mode is the **unmodeled slip branch**: gross slip when required tangential impulse exceeds Coulomb, i.e. a_t|u_t| > μ(1+e)|u_n| ⇒ |u_t|/|u_n| > μ(1+e)/a_t = **1.55 at real μ≈0.3** — but μ_safety=2.0 pushes the cap to 3.8 so it *never binds* (yaml admits this). Grid rows already cross the real threshold: 2 m/s/60 rev/s (ratio 2.30), 3/60 (1.86), 8/10 (2.13). There the model over-transfers tangential impulse — over-rotates post-bounce spin by up to ~ (0.369−0.572·|u_n|/|u_t|)|u_t|/(cR): at 2/60, s_grip=2.41 vs s_Coulomb=1.62 m/s ⇒ **spin overprediction ≈ (2.41−1.62)/(cR·2π) ≈ 9 rev/s** and Δv_t⁺≈0.8 m/s. Signature: post-bounce spin/tangential-velocity residuals that grow monotonically with |u_t|/|u_n| beyond ~1.5, ≈0 below it.
- **cosθ term**: b_t=0 for the table (untested claim only inside grip regime); paddle b_t=0.304 fitted. If tangential restitution of rubber (stored tangential elastic energy, contact-point velocity reversal) varies with speed, constant (a_t,b_t) misfits as a residual trend vs |u_t| at fixed θ (JUDGMENT).
- **No spin decay in flight** (flight.py:4): air torque spins a TT ball down on a timescale of seconds; over 0.2–0.5 s flights that's a few % of ω ⇒ ≪1 cm via Magnus. Lowest-priority assumption (JUDGMENT), but the ball rigid body now makes ω(t) directly measurable — cheap to verify.

## 4. Required spin accuracy (FACTS from grid + arithmetic)

±5 rev/s propagated through Magnus to first bounce: **topspin range: 1.5–2.1 cm at 3 m/s (0.9–1.05 m flights), 4.1–5.6 cm at 8 m/s (1.5 m)**; **sidespin lateral: 3.4 cm at 3 m/s (0.97 m flight ⇒ 3.5 cm/m) vs 3.2 cm at 8 m/s (1.5 m ⇒ 2.1 cm/m)** — per meter of flight, low speed is *more* spin-sensitive (flat-trajectory scaling Δy≈½k_mΔω L²/v ∝ 1/v). Hand check @3 m/s: Δa=0.0042·31.4·3=0.40 m/s², ½·0.40·0.45²=4.0 cm ✓; @8 m/s: Δa=1.06 m/s², ½·1.06·0.21²=2.3 cm vertical → ÷tan(23°)≈5.4 cm range ✓. JUDGMENT: against a landing-reward shaping scale of ~10 cm (table half is 137×152.5 cm), **±5 rev/s is adequate (≤6 cm); ±2 rev/s makes spin measurement a negligible (<2.5 cm) error source**. Practical caveat: 60 rev/s at 300 Hz = 72°/frame — near marker-correspondence limits; the binding risk is losing the spin *track* at high rates, not precision at low rates.

## 5. Falsification table

| # | v0 assumption | Plot / statistic from the low-speed mocap dataset | Kill criterion | Pass threshold (ties to ≤5 cm landing budget) | Data you must deliberately collect |
|---|---|---|---|---|---|
| F1 | Constant k_d (Cd const) | Per-flight k̂_d = −(a_meas−g−Magnus)·v̂/|v|² binned by mean speed 2–8 m/s | Monotonic trend in bin medians; any bin off by >±20% | All speed-bin medians within ±10% of global k_d (±10% ⇒ ≤1.2 cm, §1); slope CI contains 0 | Ordinary flights across 2–8 m/s (already being collected). Expected PASS: Re 5e3–2e4 is subcritical |
| F2 | Constant k_m (Magnus linear in ω, no saturation) | Per-flight k̂_m = a_lat/(ω v) (side-spin channel only, per yaml) plotted vs S=Rω/v over S∈[0.2, 3] | k̂_m at S>1.5 bins < 80% of the S<0.5 value (saturation) — predicted consequence 6–41 cm landing gaps (§2) | Bin medians flat within ±15% across S; else replace with C_L(S) saturating law | **Low-v/high-ω sidespin throws: 2–3 m/s with 40–60 rev/s (S>1.5)** — the single highest-value experiment; also record the S-range of the original qinghua fit |
| F3 | Constant table e_eff | Scatter e=−v_n⁺/v_n⁻ vs v_n⁻, v_n∈0.5–8 m/s (drop tests + steep hits) | \|de/dv_n\| > 0.01 per m/s (⇒ Δe>0.05 over the play range = 3–10 cm, §1), or a break (buckling) above ~5 m/s | e flat within ±0.025 band over v_n 1–6 m/s | Drop tests from 0.05–3 m heights + hard vertical hits; current lofted-play data spans only v_n 2.5–3.6 m/s (§3) — insufficient by itself |
| F4 | Constant paddle e_eff | Same statistic vs \|u_n\| using racket rigid body (data-collection phase allows it) | Trend exceeding ±0.05 across u_n 2–8 m/s (0.05 ⇒ ~15 cm landing, §1) | e(u_n) within ±0.025 of constant; else table-lookup e(u_n) | Racket-clamped or hand-held strikes sweeping impact speed; log u_n per hit |
| F5 | Tangential law a_t+b_t cosθ with non-binding Coulomb cap (grip-only) | Residuals Δω⁺ and Δv_t⁺ (model−measured) vs \|u_t\|/\|u_n\| | Residuals ≈0 for ratio<1 but grow beyond ratio≈1.5 (slip onset μ(1+e)/a_t, §3) — predicted over-spin up to ~9 rev/s at ratio 2.3 | RMS Δω⁺ < 3 rev/s in *every* ratio bin incl. >1.5 (3 rev/s ⇒ <3.5 cm, §4); else fit real μ and lower mu_safety (yaml anticipates this) | Shallow grazing bounces and heavy-spin low-speed bounces (2–3 m/s, 60 rev/s naturally give ratio 1.9–2.3) |
| F6 | No inverse Magnus | Sign of lateral accel vs ω̂×v̂ for sidespin flights binned by (Re, S), esp. highest Re (8 m/s, Re 2.1e4) × lowest S (<0.3) | Any bin with reversed sign | All bins sign-consistent; \|a_lat\| within 2σ of k_m fit | Fast/low-spin sidespin flights. Expected PASS — inverse Magnus needs Re~1e5, 5× above our max (JUDGMENT) |
| F7 | No in-flight spin decay (flight.py:4) | ω(t) from ball rigid body over ≥0.4 s flights; fit exp decay rate λ | λ·T_flight > 0.1 (>10%/flight ⇒ Magnus landing error ~0.5 cm+, compounding over rallies) | \|Δω\|/ω < 5% per flight | Free: already implied by ball-rigid-body captures |
| F8 | Instantaneous (impulse) paddle contact | Outgoing-direction residual vs racket angular velocity at contact (proxy for normal rotation during ~1–2 ms dwell) | Residual slope consistent with ω_racket·t_c rotation (>2° effect) | No correlation, residual <2° | Fast-stroke strikes with racket rigid body at 300 Hz |

Priority order implied by the numbers (JUDGMENT): **F2 (Magnus saturation, up to 41 cm) > F4 (paddle e, ~15 cm/0.05) > F3 (table e, 3–10 cm/0.05) > F5 (slip branch, ~9 rev/s over-spin) > F1/F6/F7/F8 (expected passes / sub-cm)**. Scripts: /tmp/claude-0/-workspace-yikang/56ec1c84-c2df-4667-8842-59b9f726e183/scratchpad/sens.py (grid) and tasks/brix7u07c.output (follow-ups); model sources cited from `git show origin/ball-physics-realistic:` — repo untouched.