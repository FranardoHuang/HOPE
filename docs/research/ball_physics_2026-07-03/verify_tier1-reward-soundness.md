All verification done. Final verdict follows.

---

# VERDICT: Tier-1 "virtual ball at strike" — sub-claim by sub-claim

## (a) Required racket state at the exact-strike frame exists in hope_commands.py — **CONFIRMED** (with one small gap noted)

FACTS (file: `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py`, working tree):
- `_compute_racket_state` (def line 777): `racket_pos_w` (line 781 body-mode; ~790 wrist-mode `wpos + offset_w`), `racket_quat_w` (782 / 793), `racket_lin_vel_w` (783 / 792 incl. `ω×offset` correction), `racket_normal_w` (796–798, `matrix_from_quat(...)[:, :, mount_normal_axis] * mount_normal_sign`; cfg 1383–1384: axis=1, sign=+1, "confirmed in Step 11").
- Exact-strike mask: `exact_strike = |time_to_strike| <= 0.5*step_dt + 1e-6` at line 1031, inside `_update_metrics` (def 984). FK + timing refreshed at 991–992 *before* the mask, and the docstring (984–990) confirms rewards/obs read the same fresh buffers after `command_manager.compute()` — i.e., the exact hook point the design claims. `_compute_strike_timing` def 800, `strike_window` 837, `step_dt`=0.02 s (`tracking_env_cfg.py:312–316`: sim.dt 0.005 × decimation 4).
- Acceptance thresholds cited correctly: cfg lines 1392–1395 (window 0.1 s, pos 0.075 m, vel 0.5 m/s, normal 15°); composite pass 1071–1074. `_resample_command` def 705 (per-swing hook exists). Reward buffer-read pattern confirmed: `hope_rewards.py` `_cmd` (~line 28), `racket_position_tracking_exp` reads `cmd.racket_pos_w/racket_target_*/time_to_strike` directly.
- Contact-model inputs (v0 `predict_contact(v_minus, v_r, n, omega_minus, params)`, spin_contact.py): v_r ← `racket_lin_vel_w` ✓, n ← `racket_normal_w` ✓, v_minus/ω_minus ← sampled ✓, rollout start p₀ ← `racket_pos_w` ✓. All present.

GAP (minor, JUDGMENT): racket **angular velocity** is never stored as a buffer (body mode never reads `body_ang_vel_w`; wrist mode uses `wang` transiently at ~788–792). Irrelevant if the virtual contact point is the racket origin (design's construction), but blocks any future off-center-contact refinement without a one-line addition.

## (b) RK4 rollout for 4096 envs computationally negligible — **CONFIRMED for FLOPs; CONFIRMED-as-corrected for wall clock (only with the coarse + CUDA-graphed implementation; naive eager is NOT negligible)**

FACTS (honest FLOP count from `flight.py` code): `flight_accel` ≈ 31–40 FLOPs/env (norm ~9, drag ~6, cross 9, scales/adds ~10); `rk4_step` = 4 accels + state combine ≈ 160–250 FLOPs/env. One 100-step h=10 ms rollout, N=4096: ≈ **0.07–0.1 GFLOP**; ×24 calls/iter ≈ 2.4 GFLOP/iter ≈ **~20–25 µs of pure math** on an RTX 5090 (~105 TFLOPS fp32). Bandwidth floor ~0.1–0.2 ms. Negligible.

FACTS (my independent GPU microbench, this pod, GPU 2 at 31% background util, torch 2.8.0+cu128, N=4096 — reproduces the reward agent's F4 within noise):
| config | reward agent claimed | I measured |
|---|---|---|
| eager h=10 ms, 60 steps | ~27 ms | **27.7 ms** |
| eager h=10 ms, 100 steps | ~45 ms | **45.8 ms** |
| eager h=1 ms, 0.7 s | ~315 ms | **319.5 ms** |
| eager h=1 ms, 2 s (shipped `predict_landing` flight cost, excl. bisection) | 740–900 ms | **670.6 ms** (slightly better, same order) |
| CUDA-graph capture, h=10 ms, 100 steps | "expect 5–10× less" (estimate) | **5.39 ms measured** (8.5×; likely 2–4 ms on an idle GPU) |

So: 24 × 5.4 ms = **~130 ms/iter = +2.4–2.8%** on the 4.7–5.3 s/iter run — the design's "+2–5% final" is now a measurement, not an estimate. Conversely the shipped `predict_landing` (h=1 ms, 2 s, 24-iter bisection per crossing, landing.py:74–113) at 24 calls/iter would add ≥16 s/iter (+300%) — the design's own warning is correct and mandatory, not optional. Batch-size independence (launch-bound) confirmed by the 450× gap between eager wall clock and the bandwidth floor; gathering the ~30 striking envs into a small batch saves nothing, and `exact_strike.any()` is ~never False at 4096 envs (~30 strikes/step arithmetic checks out: 4096 envs / ~100–150-step swing period ≈ 27–41). Interp-instead-of-bisection error at h=10 ms ≈ ½·g·h² ≈ 0.5 mm ≪ σ=0.3 m kernel — safe.

## (c) Landing-only reward not gameable like the vel gate — **CORRECTED: it IS gameable; I find 4 exploits; proposed kernels close ~1.5 of them**

All formulas below from v0 `table_tennis/mdp/rewards.py` (landing def ~34–74, gate ~66–70; pass_net ~77–103) and `geometry.py` (net_x=1.37, table 2.74×1.525, net top 0.1525, P2 centre x=2.055), which the Tier-1 terms port.

1. **Dink exploit (partially closed).** Target kernel: a drop shot landing x≈1.45 (just past net) is 0.6 m from the P2-centre target → shaped = exp(−0.6²/0.3²) ≈ 0.018 → kernel ≈ 0 ✓ closed. BUT the **in_bounds_bonus is location-agnostic** (`reward = valid·shaped + bonus·opp`, rewards.py ~72–74): the dink still collects 30×1.0 (bonus) + full pass_net 20×(~1+0.5) = **~60 of the ~90-point virtual max** while a perfect targeted shot gets ~90. And single-frame evaluation at 50 Hz lets the policy *brake into the strike frame* (swing normally for the shaping terms, decelerate in the last 20 ms) — the dink costs almost nothing in the retained racket_position/imitation terms. **Fix needed**: make the bonus multiplicative with the kernel (or gate on landing depth x > net_x+0.3), and/or evaluate v_r as a 2–3-frame average.
2. **Zero-spin exploit (NOT closed at proposed weights).** virtual_spin max = 5→10 vs landing+net max ≈ 90: spin is 5–11% of the virtual pot, and brushing (tangential contact) trades directly against landing accuracy through the fitted a_t=1.205/b_t=0.304 paddle map. Ace's precedent (Methods, "Rewards") samples ws ∈ [−1,1] against wp ∈ [0,1] — i.e., spin weight **up to parity** with landing, plus a dedicated high-topspin event buffer. At 1:9 the policy will rationally hit flat. The ramp closes it only if it ramps toward ~parity (≥20–30), which the design doesn't commit to.
3. **Phantom-block / retreating-racket exploit (NOT closed).** The capture gate is position-only (<0.095 m). v0's real-hit detector additionally requires the ball to be **approaching** the paddle (`table_tennis_env.py:185–190`); the virtual path drops that check, and `orient_normal` (spin_contact.py, "flip n where (v_minus−v_r)·n > 0") makes **any** racket state produce a valid contact — including a stationary or retreating paddle, or the black face/edge-on orientation (racket_normal weight is cut to 0.5 in the design, so orientation is nearly free). With paddle e_eff=0.463 a passive wall-block of the sampled v_in can legally clear the net → the policy can stop swinging and farm ~full landing reward, which *re-creates* the degeneracy the design claims to kill (and hits the fitted model far outside its calibration regime — reward-hacking against model extrapolation error, bounded only by mu_safety=2.5). **Fix needed**: gate on minimum approach speed (v_r − v_in)·n̂ > v_min and/or a normal-alignment floor in the capture gate.
4. **Net-clear-without-landing exploit (NOT closed).** `pass_net_margin` gates only on `net_valid` (rewards.py ~100–103) — no landing requirement. Blasting long/out (or a moon-lob whose landing exceeds the 0.6–1.0 s coarse horizon → landing invalid) still collects up to 20×1.5 = 30. The v0 landing term's anti-net-farm gate (~66–70) protects landing-from-net-shots, but nothing protects net-reward-from-out-shots. At weight 20 vs 30 that's a full third of the pot for balls that never land. **Fix**: gate `clear_bonus` (or the whole term) on `landing_valid & in_bounds`, and log a horizon-exceeded rate (legit slow lobs: apex ~0.8 m ⇒ flight ~1.0 s, right at the horizon edge).

JUDGMENT on the comparison: the vel gate's failure is over-constraint (one admissible velocity vector), not reward hacking; the landing reward replaces it with a *hackable* objective unless exploits 1, 3, 4 are patched. The patches are all one-line gates, so the design survives — but the claim "would NOT be gameable" as stated is false.

## (d) No actor-obs change required — **CONFIRMED structurally, with two flagged caveats**

FACTS: New reward terms read command-term buffers via `_cmd(env, name)` (`hope_rewards.py` ~28–29), exactly like every existing racket term; the ObsGroup definitions are untouched. The 175-D deploy-parity actor group is `HOPEPolicyDeployParityCfg` (`config/agibot_a3/hope_env_cfg.py:226–244`) — no ball terms in it, nothing added. v0 precedent for keeping privileged ball info out of the actor exists (`table_tennis/mdp/observations.py:40–51`, landing marked critic-only).

CAVEATS (JUDGMENT, both real):
1. **The actor cannot observe v_in/ω_in at all** (they're not in ANY obs group in this design). The landing outcome is then partially random w.r.t. everything the actor sees → the policy can only learn the best response to the *mean* incoming ball; a broad venue-fitted v_in distribution converts directly into reward variance/gradient noise. Acceptable for Tier-1 shaping; but it caps how much this "subsumes" the vel gate (the policy cannot adapt the swing per-ball, which is the whole point of Tier ≥2).
2. **Critic tension**: putting v_in/ω_in (or predicted landing) into the *critic* group is contract-legal (critic never deploys) and would cut value variance — but it changes critic input width and breaks the planned warm-start from the E/A champion unless the critic is re-initialized. The design promises both warm-start and cites the v0 critic-obs pattern; it cannot have both without an explicit critic-resize step. Unstated in the design; must be decided before launch.

## Cross-checks on the reward agent's supporting facts
- v0 file citations (flight.py `flight_accel`/`rk4_step`; landing.py `predict_landing` signature, dt=1e-3/max_time=2.0/bisect 24, dual net+table bisection; spin_contact.py equation incl. `orient_normal`; rewards.py formulas incl. net-clearance gate; geometry constants): **all verified verbatim** via `git show origin/ball-physics-realistic:...`.
- Reward weights in `hope_env_cfg.py`: racket_position 4.0 / velocity 2.0 / normal 2.0 (lines ~123–137), racket_strike_success 5.0 (~268–271), racket_progress 10.0 (~274), upper-body imitation 1.0×4 (~277–284): **verified**.
- "F4 GPU microbench" numbers: **independently reproduced** (table above).
- UNVERIFIABLE without a run: the "~25–35 envs at exact_strike per step" figure (arithmetic is consistent but RSI phase staggering is an empirical property — settle with one logging-only run reading `exact_strike.sum()` per step), and the ±10 ms timing-jitter sensitivity claim (settle with the proposed `dbg_` sensitivity log).

**Bottom line**: (a) CONFIRMED, (b) CONFIRMED with the mandatory coarse+CUDA-graph implementation (+~2.5%/iter measured; shipped predictor would be +300%), (c) CORRECTED — landing-only reward is gameable via dink-bonus, zero-spin, phantom-block, and net-without-landing; requires 3 small gate patches (bonus×kernel or depth gate, approach-speed gate, landing-gated pass_net) plus a spin-weight ramp toward parity per Ace, (d) CONFIRMED for the 175-D actor contract, with an unresolved critic-obs-vs-warm-start decision.