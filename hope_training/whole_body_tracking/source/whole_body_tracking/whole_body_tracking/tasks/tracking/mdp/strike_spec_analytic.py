"""CLOSED-FORM strike-spec inverse — the "球 -> 题目" map as a formula you substitute into.

人话(owner 的原话):"应该统一把球->题目这个生成一大堆可直接带入变量的题库方便抽取……要覆盖
球速,三维位置,旋转对动作 match 从而对 task 的影响。有了公式应该就不会成为瓶颈了。"
这个模块就是那条公式。给一批球(速度 / 三维触球点 / 旋转)和一个瞄点,直接算出拍子该摆成什么
样——不迭代求解、不打滚动、不设收敛容差。解不出来的球有**具名**的拒绝理由;公式自己知道自己在
哪儿不成立,超出适用范围就明说并把那几行让回给真求解器,绝不外插硬答。

WHAT IT REPLACES.  ``strike_spec_torch.solve_strike_specs`` is a Levenberg-Marquardt search: a
mirror-law seed, a 5-column finite-difference Jacobian, a damped line search, and an acceptance
test ``resid < tol_m``.  Its cost is per-question data-dependent (170-420 ms across incoming-speed
deciles in the numpy oracle) and its failures are convergence failures, not physics — which is why
"which decile is unsolvable" flips end-for-end between the offline numpy path and the online torch
path on the SAME physics.  This module has no seed, no Jacobian, no line search and no tolerance:
a fixed op count for every question in the box.

THE CHAIN, CUT AT THE OUTGOING VELOCITY.

    (p, v-, w-, aim) --stage 1: flight inverse (T)--> v+ --stage 2: contact inverse (pin)--> (n, v_r)

Stage 1 is 3 equations in 3 unknowns once the flight is parameterised by FLIGHT TIME ``T``:
"be at (aim_x, aim_y, surface_z) at time T".  Stage 2 is exactly determined once a PIN is declared
for the two remaining free directions.  So the brief's "underdetermined by 3" becomes: 2 killed by
the pin, 1 kept EXPLICIT as ``T``.  ``T`` is the arc/effort knob.  It is not a hard-coded 0.45 s
seed constant hiding inside a solver any more.

STAGE 1 — FLIGHT INVERSE.  Quadratic drag does not integrate, so two changes of variable:
    write |v(t)| = vbar + delta(t), lam = k_d vbar, Om = k_m w+;  then
        vdot = M v + g_vec + f(t),   M = -lam I + [Om]_x,   f(t) = -k_d delta(t) v(t)
    is an EXACT rewrite.  M is decay-plus-skew, so exp(Mt) = e^{-lam t} Rodrigues(Om_hat, |Om| t)
    exactly and M^-1 has the closed skew-inverse form.  Position becomes AFFINE in v+:
        dp(T) = A(T) v+ + b(T) + C(T),   A = M^-1(e^{MT}-I),
                                         b = (M^-2(e^{MT}-I) - T M^-1) g_vec,
                                         C = int_0^T A(T-s) f(s) ds
        INVERSE:  v+ = A(T)^-1 [ dp_target - b(T) - C(T) ]          <- one 3x3 solve
    ``vbar`` is chosen self-consistently (= the closed-form time-average of |v|), which cancels the
    first-order defect; the remainder ``C`` is an m-node Gauss-Legendre integral on the previous
    Picard iterate.  That quadrature is the ONLY uncontrolled approximation in the whole chain.
    Dropping it costs 4.05 mm median / 57.8 mm max on the forward map — it is load-bearing, and it
    is a fixed node count, not an adaptive rule.

STAGE 2 — CONTACT INVERSE.  EXACT, closed form, no fitted numbers.  Under the friction cap being
inactive (checked per question, see ``cap_margin``), with ``a = v- - R (w- x n)`` the contact-point
material velocity and ``dv = v+ - v-``:
    dv = -(1+e) u_n n - a_t u_t      and      a - u_t = (a.n) n
    =>  dv + a_t a = alpha n   with   alpha = -(1+e) u_n + a_t (a.n)
    =>  (alpha I + [k]_x) n = c,     k = a_t R w-,     c = v+ - (1 - a_t) v-
    =>  alpha^4 + (|k|^2 - |c|^2) alpha^2 - (k.c)^2 = 0                 <- biquadratic
    =>  alpha^2 = 1/2 [ (|c|^2-|k|^2) + sqrt((|c|^2-|k|^2)^2 + 4(k.c)^2) ]      (the ONLY root: the
        other root of the biquadratic in alpha^2 is <= 0 because the product of the roots is
        -(k.c)^2 <= 0, so it is never a real alpha)
    =>  n = [alpha^2 c - alpha (k x c) + (k.c) k] / (alpha (alpha^2 + |k|^2))

    ALPHA'S SIGN IS NOT A CHOICE.  alpha = -(1+e) u_n + a_t (a.n).  On an APPROACHING contact
    (u_n < 0, which is the ``not_approaching`` predicate below) and with a.n = u_n + v_n:
        alpha = -(1+e) u_n + a_t u_n + a_t v_n = (a_t - 1 - e)(u_n) + a_t v_n.
    Since a_t = 0.52 < 1 + e (e >= 0.05 by the clamp), (a_t - 1 - e) < 0 and u_n < 0 make the first
    term strictly positive; on an ADVANCING racket (v_n > 0) the second is too.  So alpha > 0 is
    FORCED, and the positive square root is the physical branch by construction, not by convention.
    The two predicates that guarantee it are exactly ``not_approaching`` / ``not_advancing``, both
    reported.  (At w- = 0 the formula degenerates to n = normalize(v+ - (1-a_t) v-) — the CORRECTED
    mirror law.  The repo's LM seed uses normalize(v+ - v-); the missing a_t v- term is its ~22 deg
    face error and the reason the LM spends every iteration it has on face elevation.)

    Normal speed from the normal component beta_n := dv.n = -(1+e(|u_n|)) u_n, by a 4-step
    contraction (factor |g2| e |u_n| / (1+e) ~ 0.078 in the box, so 4 steps is < 1e-9), then
    v_n = a.n - u_n.  Outgoing spin w+ = w- + (a_t/(cR)) (n x u_t) closes the loop into stage 1's
    Magnus term — the contact MAKES spin (|w+| ~ 58 rad/s even from a spin-free ball), which is why
    the outer fixed point exists at all.  Measured contraction ~1/8 per round.

FACE SIGN.  The n this module returns is already in the orientation ``predict_paddle_contact``
would itself pick: u_n = (v- - v_r).n < 0 is exactly ``orient_normal``'s no-flip condition, and the
(w- x r).n term it drops is identically zero.  So +n and -n are not a hidden branch here — the
physics is invariant to the reported sign, and ``ref_normal`` sign-matching is applied afterwards
exactly as ``solve_strike_specs`` does it, for the clip-face convention only.

THE ANSWER SET IS A SPHERICAL CAP, NOT A POINT.  Two of the three free dimensions are a genuine
choice, so this module makes the choice a NAMED argument instead of an accident.  Eliminating u
from the contact equation gives, for a fixed v+:
        v_r(n) = a(n) + dv/a_t + beta (dv.n) n,      beta = 1/(1+e) - 1/a_t  < 0
As n sweeps the unit sphere, (dv.n) n sweeps a sphere through the origin, so v_r sweeps a SPHERE
    centre  m = v- + dv/a_t + (beta |dv| / 2) dvhat,      radius  rho = |beta| |dv| / 2
(exact at w- = 0 and fixed e; see ``answer_sphere`` for the measured error with spin).  The
friction cap restricts n to a cone of half-angle atan(mu) = 26.565 deg about dvhat, which by the
inscribed-angle relation is a spherical CAP of half-angle 2 atan(mu) = 53.13 deg about -dvhat.
``swing_gap`` is then the distance in m/s from the clip's own swing velocity to that cap: a
continuous, solver-independent difficulty label that cannot fail, and the thing that turns coverage
design into an inequality instead of a sampling study.

PINS (which point on the cap):
    ``"normal"``      v_r || n.  The incumbent convention — the LM seeds v_t at exactly 0 and its
                      w_speed regulariser was supposed to hold it there.  Exact (biquadratic).
                      DEFAULT, because it is the one that does not invalidate the shipped banks.
    ``"min_speed"``   the point of the cap nearest the origin — the least-effort objective the LM's
                      regulariser advertised and (measured: regulariser term 912x the landing term
                      at exit) never delivered.
    ``"clip_swing"``  the point of the cap nearest the clip's own swing velocity — maximises motion
                      match; needs ``v_clip``.
The last two are sphere projections followed by a fixed 3-round n-recovery contraction (spin and
e(u_n) make the sphere only approximate); ``pin_resid_mps`` reports how well the recovered n
reproduces the requested point, and rows that miss are flagged for fallback rather than shipped.

WHAT IS REFUSED, AND WHY IT IS VISIBLE.  Nothing here can fail to converge, so nothing is ever
rejected for a reason that is about the solver.  Everything rejected is rejected by a named
closed-form predicate (``REASONS``): budget / net / horizon / approach / advance / cap-active /
fit-envelope / non-finite.  Separately, ``needs_fallback`` marks rows where the CLOSED FORM's own
assumptions do not hold (cap active, pin residual too large, non-finite intermediate) — those defer
to the true solver and are counted.  ``generate``-style redrawing is deliberately NOT done here:
that is the caller's decision and it must be counted by the caller.

DEPENDENCIES.  torch only.  No repo imports on purpose, so the offline bank builder can load this
file by path (``gen_stage1_questions._load_mod``) exactly the way it already loads
``virtual_ball.py``, while the online producer imports it as a package submodule.  ``prm`` is
duck-typed: anything with the ``VirtualBallParams`` attribute names works.

NOT WIRED.  Nothing in this repo calls this module yet, and no default changes because it exists.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-12

#: Rejection reasons, index-aligned; ``reason = -1`` means accepted.  Same shape of vocabulary as
#: ``stroke_adapt_torch.REASONS`` so the two histograms read alike, but these are PHYSICAL
#: categories — there is no "resid_gt_tol" here because there is no residual to exceed.
REASONS = (
    "speed_over_budget",     # 0  |v_r| > speed_budget
    "net_not_cleared",       # 1  the closed-form arc is below the net top at the net plane
    "no_landing_in_horizon",  # 2  T >= the scorer's own rollout window
    "not_approaching",       # 3  u_n >= 0: the racket does not meet the ball (alpha proof fails)
    "not_advancing",         # 4  v_n <= 0: a block/absorb, not a stroke
    "cap_active",            # 5  friction cap binds — stage 2 is not exact here
    "fit_envelope",          # 6  |u_n| outside the venue restitution fit band (WARN by default)
    "nonfinite",             # 7  a non-finite intermediate (degenerate geometry)
    "pin_unrealised",        # 8  the recovered n does not reproduce the requested cap point
)
_R_SPEED, _R_NET, _R_HORIZON, _R_APPROACH, _R_ADVANCE, _R_CAP, _R_ENVELOPE, _R_NONFINITE, \
    _R_PIN = range(9)

#: Venue restitution e(u_n) = g1 exp(g2 |u_n|) was fitted on paddle u_n 1.4-7.2 m/s
#: (configs/ball_physics_venue.yaml VALIDITY ENVELOPE).  Outside it the closed form is faithful to
#: the model and the MODEL is extrapolating — a distinction the LM never surfaced at all.
FIT_UN_MIN = 1.4
FIT_UN_MAX = 7.2

#: Gauss-Legendre nodes/weights on [0, 1].  Fixed rules: the defect quadrature has a fixed node
#: count by design, never an adaptive one.
_GL = {
    2: ((0.211324865405187, 0.788675134594813), (0.5, 0.5)),
    3: ((0.112701665379258, 0.5, 0.887298334620742),
        (0.277777777777778, 0.444444444444444, 0.277777777777778)),
    4: ((0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797026),
        (0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727)),
    5: ((0.046910077030668, 0.230765344947158, 0.5, 0.769234655052842, 0.953089922969332),
        (0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683,
         0.118463442528095)),
}


# --------------------------------------------------------------------------------------------
# small linear-algebra pieces (all batched, all branch-free)
# --------------------------------------------------------------------------------------------
def _skew(v: torch.Tensor) -> torch.Tensor:
    """[v]_x for a batch of 3-vectors: (N,3) -> (N,3,3)."""
    n = v.shape[0]
    k = v.new_zeros(n, 3, 3)
    k[:, 0, 1] = -v[:, 2]
    k[:, 0, 2] = v[:, 1]
    k[:, 1, 0] = v[:, 2]
    k[:, 1, 2] = -v[:, 0]
    k[:, 2, 0] = -v[:, 1]
    k[:, 2, 1] = v[:, 0]
    return k


def _eye_like(v: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[0], 3, 3)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(_EPS)


def exp_decay_rotation(lam: torch.Tensor, om: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """exp(M t) for M = -lam I + [om]_x, in closed form.  lam (N,), om (N,3), t (N,) -> (N,3,3).

    Exact, not a series truncation: the isotropic part integrates to a scalar decay and the skew
    part to a rotation about om_hat by |om| t (Rodrigues).  The small-angle branches are the
    sinc/versine series so |om| -> 0 (no spin) is smooth rather than guarded by an epsilon.
    """
    ident = _eye_like(om)
    kmat = _skew(om)
    omg = torch.linalg.norm(om, dim=-1)
    th = omg * t
    omg_s = omg.clamp_min(1e-30)
    small = th.abs() <= 1e-6
    s1 = torch.where(small, t * (1.0 - (omg * t) ** 2 / 6.0), torch.sin(th) / omg_s)
    s2 = torch.where(small, 0.5 * t ** 2 * (1.0 - (omg * t) ** 2 / 12.0),
                     (1.0 - torch.cos(th)) / omg_s ** 2)
    rot = ident + s1[:, None, None] * kmat + s2[:, None, None] * (kmat @ kmat)
    return torch.exp(-lam * t)[:, None, None] * rot


def _minv(lam: torch.Tensor, om: torch.Tensor) -> torch.Tensor:
    """(-lam I + [om]_x)^-1 via the skew-inverse identity (a I + [k]_x)^-1."""
    ident = _eye_like(om)
    kmat = _skew(om)
    o2 = (om * om).sum(-1)
    den = (lam * (lam ** 2 + o2)).clamp_min(1e-30)
    num = ((lam ** 2)[:, None, None] * ident
           + lam[:, None, None] * kmat
           + om[:, :, None] * om[:, None, :])
    return -num / den[:, None, None]


def _linear_pieces(lam, om, t, g):
    """A(T) = M^-1(e^{MT}-I), b(T) = (M^-2(e^{MT}-I) - T M^-1) g_vec."""
    ident = _eye_like(om)
    emt = exp_decay_rotation(lam, om, t)
    mi = _minv(lam, om)
    a_mat = mi @ (emt - ident)
    g_vec = torch.zeros_like(om)
    g_vec[:, 2] = -g
    b_vec = ((mi @ a_mat) - t[:, None, None] * mi) @ g_vec[:, :, None]
    return a_mat, b_vec[:, :, 0], mi


def _interp_nodes(nodes_t, f_nodes, r):
    """Linear interpolation of f (N,m,3) sampled at ascending nodes_t (N,m) onto r (N,)."""
    m = nodes_t.shape[1]
    idx = torch.searchsorted(nodes_t.contiguous(), r[:, None].contiguous()).clamp(1, m - 1)[:, 0]
    t0 = torch.gather(nodes_t, 1, (idx - 1)[:, None])[:, 0]
    t1 = torch.gather(nodes_t, 1, idx[:, None])[:, 0]
    w = ((r - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0.0, 1.0)
    f0 = torch.gather(f_nodes, 1, (idx - 1)[:, None, None].expand(-1, 1, 3))[:, 0]
    f1 = torch.gather(f_nodes, 1, idx[:, None, None].expand(-1, 1, 3))[:, 0]
    return f0 + (f1 - f0) * w[:, None]


def _traj_at_nodes(nodes_t, v0, lam, om, g, f_nodes, node_x, node_w):
    """v(s) at the quadrature nodes under the linearised flow, with the previous defect folded in.

    The defect integral int_0^{s_j} e^{M(s_j-r)} f(r) dr uses the SAME Gauss rule rescaled to
    [0, s_j] with f interpolated from the outer nodes — a nested rule, still a fixed op count.
    """
    m = nodes_t.shape[1]
    ident = torch.eye(3, dtype=v0.dtype, device=v0.device)
    g_vec = torch.zeros_like(v0)
    g_vec[:, 2] = -g
    mi = _minv(lam, om)
    out = []
    for j in range(m):
        s = nodes_t[:, j]
        emt = exp_decay_rotation(lam, om, s)
        v = (emt @ v0[:, :, None])[:, :, 0] + (mi @ ((emt - ident) @ g_vec[:, :, None]))[:, :, 0]
        if f_nodes is not None:
            acc = torch.zeros_like(v)
            for lidx in range(len(node_x)):
                r = s * node_x[lidx]
                wgt = s * node_w[lidx]
                fr = _interp_nodes(nodes_t, f_nodes, r)
                ee = exp_decay_rotation(lam, om, (s - r).clamp_min(0.0))
                acc = acc + wgt[:, None] * (ee @ fr[:, :, None])[:, :, 0]
            v = v + acc
        out.append(v)
    del m
    return torch.stack(out, dim=1)


# --------------------------------------------------------------------------------------------
# stage 1 — flight inverse
# --------------------------------------------------------------------------------------------
def flight_inverse(dp_target, w_plus, t_flight, prm, n_nodes=4, n_picard=2):
    """Outgoing velocity that carries the ball from the contact point to ``dp_target`` in ``T``.

    ``dp_target`` (N,3) is (aim - p) with the z component (surface_z - p_z).  Returns
    ``(v_plus, info)``; ``info["vbar"]`` is the self-consistent mean speed the linearisation used.

    人话:飞行段没有初等原函数(二次阻力),但把速度写成"均值 + 偏差"之后,线性那一半是精确可解
    的(衰减 x 绕自旋轴旋转),只剩偏差项用固定几个高斯点补。节点数固定,不看数据、不设容差。
    """
    dt, dev = dp_target.dtype, dp_target.device
    n = dp_target.shape[0]
    om = prm.k_m * w_plus
    g = prm.g
    xs, ws = _GL[n_nodes]
    node_x = [torch.full((n,), float(x), dtype=dt, device=dev) for x in xs]
    node_w = [torch.full((n,), float(w), dtype=dt, device=dev) for w in ws]
    nodes_t = torch.stack([t_flight * x for x in node_x], dim=1)          # (N,m) ascending

    vbar = torch.linalg.norm(dp_target, dim=-1) / t_flight.clamp_min(1e-6)
    c_def = torch.zeros_like(dp_target)
    f_nodes = None
    v_plus = None
    for it in range(n_picard + 1):
        lam = prm.k_d * vbar
        a_mat, b_vec, _ = _linear_pieces(lam, om, t_flight, g)
        v_plus = torch.linalg.solve(a_mat, (dp_target - b_vec - c_def)[:, :, None])[:, :, 0]
        if it == n_picard:
            break
        vs = _traj_at_nodes(nodes_t, v_plus, lam, om, g, f_nodes, node_x, node_w)
        sp = torch.linalg.norm(vs, dim=-1)                                 # (N,m)
        wq = torch.stack(node_w, dim=1)
        vbar_new = (sp * wq).sum(1)                                        # (1/T) int |v| by GL
        f_nodes = -prm.k_d * (sp - vbar_new[:, None])[:, :, None] * vs
        lam_new = prm.k_d * vbar_new
        mi2 = _minv(lam_new, om)
        ident = torch.eye(3, dtype=dt, device=dev)
        c_def = torch.zeros_like(dp_target)
        for j in range(nodes_t.shape[1]):
            tau = t_flight - nodes_t[:, j]
            a_j = mi2 @ (exp_decay_rotation(lam_new, om, tau) - ident)
            c_def = c_def + (t_flight * node_w[j])[:, None] * (a_j @ f_nodes[:, j, :, None])[:, :, 0]
        vbar = vbar_new
    return v_plus, {"vbar": vbar}


# --------------------------------------------------------------------------------------------
# stage 2 — contact inverse
# --------------------------------------------------------------------------------------------
def _restitution_fixed_point(beta_n, prm, n_steps=4):
    """u_n from dv.n = -(1+e(|u_n|)) u_n.  Contraction factor ~0.078 in the box; 4 steps < 1e-9."""
    u_n = -beta_n / 1.6
    e = None
    for _ in range(n_steps):
        e = (prm.paddle_e_g1 * torch.exp(prm.paddle_e_g2 * u_n.abs())).clamp(0.05, 0.95)
        u_n = -beta_n / (1.0 + e)
    e = (prm.paddle_e_g1 * torch.exp(prm.paddle_e_g2 * u_n.abs())).clamp(0.05, 0.95)
    return u_n, e


def contact_inverse_normal_pin(v_plus, v_in, w_in, prm):
    """EXACT (n, v_r) under the pin ``v_r = v_n n`` (tangential racket velocity exactly zero).

    Returns ``(n, v_r, w_plus, diag)``.  ``diag`` carries ``u_n`` (signed), ``u_t`` (magnitude),
    ``e``, ``cap_margin`` = mu(1+e)|u_n| - a_t|u_t| (negative <=> the friction cap binds, i.e. the
    exactness assumption fails), and ``alpha`` (must be > 0; see the module docstring's proof).
    """
    a_t = prm.paddle_a_t
    rad = prm.ball_radius
    k = a_t * rad * w_in
    c = v_plus - (1.0 - a_t) * v_in
    k2 = (k * k).sum(-1)
    c2 = (c * c).sum(-1)
    kc = (k * c).sum(-1)
    disc = (c2 - k2) ** 2 + 4.0 * kc ** 2
    alpha2 = 0.5 * ((c2 - k2) + torch.sqrt(disc.clamp_min(0.0)))
    alpha = torch.sqrt(alpha2.clamp_min(1e-30))
    num = (alpha2[:, None] * c
           - alpha[:, None] * torch.cross(k, c, dim=-1)
           + kc[:, None] * k)
    n = _unit(num / (alpha * (alpha2 + k2)).clamp_min(1e-30)[:, None])
    n, v_r, w_plus, diag = close_from_normal(n, v_plus, v_in, w_in, prm)
    diag["alpha"] = alpha
    return n, v_r, w_plus, diag


def close_from_normal(n, v_plus, v_in, w_in, prm):
    """THE GENERAL CONTACT INVERSE: the racket velocity that turns (v-, w-) into v+ across face n.

    Exact for ANY n inside the friction cone — this is what makes the answer set 2-dimensional at
    fixed v+ and therefore what makes a PIN a real choice rather than a fiction.  Both components
    of the relative contact velocity are read off the REQUIRED velocity change, never off the
    ball's own velocity:
        u_n = -(dv.n) / (1 + e(|u_n|))        (the restitution fixed point)
        u_t = -(dv - (dv.n) n) / a_t          (the tangential impulse, cap inactive)
        v_r = a(n) - u_n n - u_t,   a(n) = v- - R (w- x n)
    (Reading u_t off ``a`` instead is only correct when the racket has no tangential velocity —
    i.e. exactly the ``pin="normal"`` case, where the biquadratic has already chosen the n that
    makes the two agree.  Using it for an off-normal pin silently answers a different question:
    the returned v_r then does NOT reproduce v+, and the landing moves.)
    """
    a_t, rad, c_in = prm.paddle_a_t, prm.ball_radius, prm.inertia_coeff
    dv = v_plus - v_in
    beta_n = (dv * n).sum(-1)
    u_n, e = _restitution_fixed_point(beta_n, prm)
    u_t_vec = -(dv - beta_n[:, None] * n) / a_t
    u_t = torch.linalg.norm(u_t_vec, dim=-1)
    a_vec = v_in - rad * torch.cross(w_in, n, dim=-1)
    v_r = a_vec - u_n[:, None] * n - u_t_vec
    v_n = (v_r * n).sum(-1)
    w_plus = w_in + (a_t / (c_in * rad)) * torch.cross(n, u_t_vec, dim=-1)
    cap_margin = prm.paddle_mu * (1.0 + e) * u_n.abs() - a_t * u_t
    return n, v_r, w_plus, {"u_n": u_n, "u_t": u_t, "e": e, "cap_margin": cap_margin,
                            "v_n": v_n, "dv": dv}


# --------------------------------------------------------------------------------------------
# the answer set: a spherical cap in racket-velocity space
# --------------------------------------------------------------------------------------------
def answer_sphere(v_in, w_in, v_plus, prm, e=None):
    """The set of racket velocities that produce ``v_plus`` from this ball, as a spherical cap.

    Eliminating u from the contact equation gives, for a fixed v+,
        v_r(n) = a(n) + dv/a_t + beta (dv.n) n,   beta = 1/(1+e) - 1/a_t < 0
    so as n sweeps the unit sphere v_r sweeps a sphere of
        centre m = v- + dv/a_t + (beta |dv| / 2) dvhat,   radius rho = |beta| |dv| / 2
    and the friction cap (angle(dv, n) <= atan(mu)) restricts it to a CAP of half-angle
    2 atan(mu) = 53.13 deg about ``-dvhat`` (inscribed-angle relation).

    EXACTNESS.  Exact at w- = 0 with e held at the value passed in.  With spin the a(n) term adds a
    per-n wobble of order R|w-| (2 cm x 50 rad/s = 1.0 m/s at the box edge), and e(u_n) varies ~9%
    over the box, so treat this as a geometric picture and a difficulty label — NOT as the answer.
    The answer comes from the pinned solve, which carries the spin exactly.  ``solve_analytic``
    reports the measured ``sphere_resid_mps`` = |v_r_pinned - projection| so the size of that
    approximation is never a guess.

    Returns a dict with ``centre`` (N,3), ``radius`` (N,), ``axis`` (N,3) (the cap axis, = -dvhat),
    ``cap_half_angle_rad`` (scalar float).
    """
    a_t = prm.paddle_a_t
    dv = v_plus - v_in
    dmag = torch.linalg.norm(dv, dim=-1)
    dhat = dv / dmag.clamp_min(_EPS)[:, None]
    if e is None:
        # e depends on u_n, which is not known before n is; the worst case dv.n = |dv| gives the
        # self-consistent upper-bound u_n, which is what the cap geometry wants anyway.
        _, e = _restitution_fixed_point(-dmag, prm)
    beta = 1.0 / (1.0 + e) - 1.0 / a_t
    centre = v_in + dv / a_t + (beta * dmag / 2.0)[:, None] * dhat
    radius = (beta.abs() * dmag / 2.0)
    return {"centre": centre, "radius": radius, "axis": -dhat,
            "cap_half_angle_rad": 2.0 * math.atan(prm.paddle_mu)}


def project_on_cap(target, sphere):
    """Nearest point of the spherical cap to ``target`` (N,3), and the signed gap in m/s.

    Returns ``(point, gap)``.  ``gap`` is the distance from ``target`` to the CAP (always >= 0 for
    the cap; it is 0 only on the cap itself).  Points inside the sphere but off the cap are handled
    by the rim branch, so this is the true cap distance, not the sphere distance.
    """
    centre, radius, axis = sphere["centre"], sphere["radius"], sphere["axis"]
    half = sphere["cap_half_angle_rad"]
    q = target - centre
    qn = torch.linalg.norm(q, dim=-1).clamp_min(_EPS)
    qh = q / qn[:, None]
    cos_pol = (qh * axis).sum(-1).clamp(-1.0, 1.0)
    in_cap = cos_pol >= math.cos(half)
    # spherical branch: radially onto the sphere
    p_sph = centre + radius[:, None] * qh
    # rim branch: nearest point of the rim circle (centre + radius(cos h axis + sin h e_perp))
    perp = qh - cos_pol[:, None] * axis
    perp = _unit(torch.where(torch.linalg.norm(perp, dim=-1, keepdim=True) > 1e-9, perp,
                             torch.roll(axis, 1, dims=-1) - (torch.roll(axis, 1, dims=-1) * axis)
                             .sum(-1, keepdim=True) * axis))
    p_rim = centre + radius[:, None] * (math.cos(half) * axis + math.sin(half) * perp)
    point = torch.where(in_cap[:, None], p_sph, p_rim)
    gap = torch.linalg.norm(target - point, dim=-1)
    return point, gap


def swing_gap(v_clip, v_in, w_in, v_plus, prm):
    """How far the clip's OWN swing velocity is from being a legal answer, in m/s.

    人话:一个连续的难度标签。0 表示这颗球用这条动作原样就能答;越大表示这条动作离"能答"越远。
    它不会失败、不需要求解器,一次减法就出来——所以覆盖设计从"抽样研究"变成"不等式"。

    Honest scope: built on ``answer_sphere``, so it inherits that function's spin/e approximation.
    It is a LABEL, not an acceptance test.
    """
    sph = answer_sphere(v_in, w_in, v_plus, prm)
    _, gap = project_on_cap(v_clip, sph)
    return gap


# --------------------------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def solve_analytic(
    p_contact: torch.Tensor,
    v_ball: torch.Tensor,
    w_ball: torch.Tensor,
    aim_xy: torch.Tensor,
    prm,
    surface_z: float,
    net_x: float,
    t_flight=0.66,
    pin: str = "normal",
    v_clip: torch.Tensor = None,
    ref_normal: torch.Tensor = None,
    speed_budget: float = 3.4,
    net_top_z: float = None,
    n_outer: int = 4,
    n_nodes: int = 4,
    n_picard: int = 2,
    horizon_s: float = 1.0,
    envelope_rejects: bool = False,
    pin_tol_mps: float = 0.02,
) -> dict:
    """THE FORMULA.  Substitute ball speed / 3D contact point / spin, get the demanded racket state.

    All tensors are ENV-LOCAL and batched ``(N, ...)``, the same convention as ``coarse_landing``
    and ``solve_strike_specs``:
      ``p_contact`` (N,3) the ball's arrival point, ``v_ball`` (N,3), ``w_ball`` (N,3) rad/s,
      ``aim_xy`` (N,2), ``surface_z`` the plane the ball CENTRE crosses (table surface + R),
      ``net_x`` the net plane.  ``t_flight`` is the declared free dof: a float or an (N,) tensor.

    Returns a dict:
      ``v_r`` (N,3), ``n`` (N,3) unit (sign-matched to ``ref_normal`` when given),
      ``v_plus`` (N,3), ``w_plus`` (N,3) the outgoing ball state the answer produces,
      ``ok`` (N,) bool, ``reason`` (N,) long index into ``REASONS`` (-1 where ok),
      ``needs_fallback`` (N,) bool — the closed form is outside its OWN validated envelope on this
        row and the caller should re-solve it with ``solve_strike_specs``; ``fallback_reason`` (N,),
      ``speed`` (N,), ``u_n`` (N,), ``e`` (N,), ``cap_margin`` (N,), ``net_z`` (N,) the closed-form
        arc height at the net plane, ``t_flight`` (N,), ``sphere_resid_mps`` (N,),
      ``pin_resid_mps`` (N,) — for off-normal pins, how far the realised answer is from the
        requested cap point.

    NO ROW IS EVER FABRICATED.  ``ok=False`` rows keep their numbers for diagnosis (they are a real
    closed-form answer to a question whose ANSWER is illegal, e.g. over the speed budget) but the
    caller must mask on ``ok``; ``needs_fallback`` rows are ones where the FORMULA is not entitled
    to an opinion.  Redrawing is not done here — see the module docstring.

    COMMON CASE IS ITERATION-FREE in the sense that matters: a fixed op count for every row, no
    data-dependent control flow, no convergence test, no early exit.  The ``n_outer`` rounds are a
    contracting fixed point (the Magnus term needs w+, and the CONTACT is what makes w+).

    ACCURACY vs COST, measured on 6000 rows of the ContinuousQuestionCfg box, replayed through
    ``coarse_landing`` (median landing error, mm):
        n_outer:            1        2        3        4        5
        nodes=3 picard=1  134.6     13.5      1.38     1.47     1.39     <- quadrature floor ~1.4 mm
        nodes=4 picard=2  135.7     12.5      1.22     0.13     0.06     <- outer loop is ~10x/round
    So the outer fixed point contracts about 10x per round and the QUADRATURE sets the floor.  The
    defaults (``n_outer=4, n_nodes=4, n_picard=2``) sit at 0.13 mm median / 0.96 mm max — 50x
    inside the 5 mm the offline bank tolerates.  ``n_outer=3, n_nodes=3, n_picard=1`` is the lean
    config at 1.4 mm and roughly a third of the cost; it is still better than the shipped banks'
    own 3.3 mm.  ``n_outer=1`` is NOT usable (Magnus is not a perturbation here).
    """
    n_rows = int(p_contact.shape[0])
    dev, dt = p_contact.device, p_contact.dtype
    if isinstance(t_flight, torch.Tensor):
        tf = t_flight.to(device=dev, dtype=dt).reshape(-1)
        if tf.shape[0] == 1:
            tf = tf.expand(n_rows)
    else:
        tf = torch.full((n_rows,), float(t_flight), device=dev, dtype=dt)
    if pin not in ("normal", "min_speed", "clip_swing"):
        raise ValueError("pin must be one of 'normal', 'min_speed', 'clip_swing'")
    if pin == "clip_swing" and v_clip is None:
        raise ValueError("pin='clip_swing' needs v_clip (N,3): the clip's own swing velocity")

    dp = torch.cat([aim_xy - p_contact[:, :2], (surface_z - p_contact[:, 2])[:, None]], dim=-1)

    w_plus = w_ball.clone()
    n = v_r = v_plus = None
    diag = {}
    pin_resid = torch.zeros(n_rows, device=dev, dtype=dt)
    sphere_resid = torch.zeros(n_rows, device=dev, dtype=dt)
    vbar = None
    for _round in range(int(n_outer)):
        v_plus, finfo = flight_inverse(dp, w_plus, tf, prm, n_nodes=n_nodes, n_picard=n_picard)
        vbar = finfo["vbar"]
        n, v_r, w_plus, diag = contact_inverse_normal_pin(v_plus, v_ball, w_ball, prm)
        if pin != "normal":
            target = torch.zeros_like(v_r) if pin == "min_speed" else v_clip
            n, v_r, w_plus, diag, pin_resid = _repin(
                n, v_r, v_plus, v_ball, w_ball, prm, target)
        sph = answer_sphere(v_ball, w_ball, v_plus, prm, e=diag["e"])
        _, sphere_resid = project_on_cap(v_r, sph)

    speed = torch.linalg.norm(v_r, dim=-1)
    net_z = _net_height(p_contact, v_plus, w_plus, prm, net_x, tf, vbar)

    # ---- named, closed-form predicates.  Order matters only for which single reason is reported.
    finite = (torch.isfinite(v_r).all(-1) & torch.isfinite(n).all(-1)
              & torch.isfinite(v_plus).all(-1) & torch.isfinite(speed))
    approach_ok = diag["u_n"] < 0.0
    advance_ok = diag["v_n"] > 0.0
    cap_ok = diag["cap_margin"] > 0.0
    speed_ok = speed <= float(speed_budget) + 1e-9
    horizon_ok = tf < float(horizon_s)
    if net_top_z is None:
        net_ok = torch.ones(n_rows, dtype=torch.bool, device=dev)
    else:
        net_ok = net_z > float(net_top_z)
    env_ok = (diag["u_n"].abs() >= FIT_UN_MIN) & (diag["u_n"].abs() <= FIT_UN_MAX)
    pin_ok = pin_resid <= float(pin_tol_mps)

    ok = finite & approach_ok & advance_ok & cap_ok & speed_ok & horizon_ok & net_ok & pin_ok
    if envelope_rejects:
        ok = ok & env_ok

    reason = torch.full((n_rows,), -1, dtype=torch.long, device=dev)

    def _mark(mask, code):
        return torch.where(mask & (reason < 0), torch.full_like(reason, code), reason)

    reason = _mark(~finite, _R_NONFINITE)
    reason = _mark(~approach_ok, _R_APPROACH)
    reason = _mark(~advance_ok, _R_ADVANCE)
    reason = _mark(~cap_ok, _R_CAP)
    reason = _mark(~pin_ok, _R_PIN)
    reason = _mark(~horizon_ok, _R_HORIZON)
    reason = _mark(~speed_ok, _R_SPEED)
    reason = _mark(~net_ok, _R_NET)
    if envelope_rejects:
        reason = _mark(~env_ok, _R_ENVELOPE)
    reason = torch.where(ok, torch.full_like(reason, -1), reason)

    # ---- fallback: where the CLOSED FORM's own assumptions do not hold.  These are NOT rejections
    # of the question — they are the formula declining to extrapolate.  Envelope excursions are a
    # PHYSICS-FIT warning: the true solver uses the same fit, so deferring would not help; they are
    # reported and (by default) not rejected, which is why they are not in needs_fallback.
    needs_fallback = (~finite) | (~cap_ok) | (~pin_ok)
    fb_reason = torch.full((n_rows,), -1, dtype=torch.long, device=dev)
    fb_reason = torch.where(~cap_ok, torch.full_like(fb_reason, _R_CAP), fb_reason)
    fb_reason = torch.where(~pin_ok, torch.full_like(fb_reason, _R_PIN), fb_reason)
    fb_reason = torch.where(~finite, torch.full_like(fb_reason, _R_NONFINITE), fb_reason)

    if ref_normal is not None:
        flip = torch.sum(n * ref_normal.to(device=dev, dtype=dt), dim=-1, keepdim=True) < 0.0
        n = torch.where(flip, -n, n)      # +-n span the same face; match the clip-face sign

    return {"v_r": v_r, "n": n, "v_plus": v_plus, "w_plus": w_plus,
            "ok": ok, "reason": reason,
            "needs_fallback": needs_fallback, "fallback_reason": fb_reason,
            "speed": speed, "u_n": diag["u_n"], "u_t": diag["u_t"], "e": diag["e"],
            "cap_margin": diag["cap_margin"], "net_z": net_z, "t_flight": tf,
            "fit_envelope_ok": env_ok,
            "pin_resid_mps": pin_resid, "sphere_resid_mps": sphere_resid}


def _repin(n0, v_r0, v_plus, v_in, w_in, prm, target, n_rounds=4):
    """Move the answer to the cap point nearest ``target``, then recover the face that realises it.

    The exact relation is  v_r(n) = S(n) + s(n)  with the SPIN-FREE sphere map
        S(n) = v- + dv/a_t + beta (dv.n) n            (this is ``answer_sphere``)
    and the spin wobble  s(n) = a(n) - v- = -R (w- x n),  |s| <= R|w-| (1.0 m/s at the box edge).
    So the projection is done on the SHIFTED target ``target - s(n)`` and the face is recovered from
        beta (dv.n) n = want - v- - dv/a_t,   beta (dv.n) < 0 inside the friction cone
        =>  n = -normalize(want - v- - dv/a_t)         <- the minus is the sign of beta, not a
                                                          convention: beta = 1/(1+e) - 1/a_t < 0
    ``s(n)`` and ``e(u_n)`` depend on n, so this is a FIXED ``n_rounds`` contraction, and
    ``pin_resid`` reports |v_r_realised - want| so nothing is asserted that was not realised.
    """
    a_t, rad = prm.paddle_a_t, prm.ball_radius
    dv = v_plus - v_in
    n = n0
    want = None
    for _ in range(int(n_rounds)):
        _, _, _, d = close_from_normal(n, v_plus, v_in, w_in, prm)
        sph = answer_sphere(v_in, w_in, v_plus, prm, e=d["e"])
        wobble = -rad * torch.cross(w_in, n, dim=-1)
        want, _ = project_on_cap(target - wobble, sph)
        n = -_unit(want - v_in - dv / a_t)
        want = want + wobble
    n, v_r, w_plus, diag = close_from_normal(n, v_plus, v_in, w_in, prm)
    resid = torch.linalg.norm(v_r - want, dim=-1)
    return n, v_r, w_plus, diag, resid


def _net_height(p_contact, v_plus, w_plus, prm, net_x, t_flight, vbar=None):
    """Closed-form ball height at the net plane, from the same linearised flow as stage 1.

    Solves for the time the closed-form x(t) reaches ``net_x`` by four Newton steps on the affine
    trajectory (x(t) is monotone while v_x > 0, which every legal return has), then evaluates z
    there.  Arcs that never reach the net plane inside T get ``-inf``, so the net predicate rejects
    them instead of silently passing.

    ACCURACY, MEASURED against the rollout's own ``net_z``: 4.2 mm median / 12.6 mm max when the
    stage-1 mean speed ``vbar`` is supplied (linearising about the LAUNCH speed instead costs
    20.4 mm median / 55.1 mm max).  The clearance VERDICT agreed with the rollout on 100.00 % of
    20 000 rows either way — but the net top is a hard 15.25 cm edge, so the better number is the
    one that gets used.  The defect quadrature is deliberately NOT carried here: this predicate
    needs millimetres, not the micrometres the landing point needs.
    """
    om = prm.k_m * w_plus
    dx = float(net_x) - p_contact[:, 0]
    lam = prm.k_d * (torch.linalg.norm(v_plus, dim=-1) if vbar is None else vbar)
    # bracket: fraction of T from the straight-line guess, then Newton on the affine trajectory
    frac = (dx / (v_plus[:, 0] * t_flight).clamp_min(1e-6)).clamp(0.0, 1.0)
    t = frac * t_flight
    for _ in range(4):
        pos, vel = _pos_vel_at(p_contact, v_plus, om, prm, t, lam)
        t = t - (pos[:, 0] - float(net_x)) / vel[:, 0].clamp_min(1e-6)
        t = t.clamp(torch.zeros_like(t), t_flight)
    pos, _ = _pos_vel_at(p_contact, v_plus, om, prm, t, lam)
    reached = (pos[:, 0] - float(net_x)).abs() < 1e-3
    neg_inf = torch.full_like(pos[:, 2], float("-inf"))
    return torch.where(reached, pos[:, 2], neg_inf)


def _pos_vel_at(p0, v0, om, prm, t, lam):
    """Position/velocity of the LINEARISED flow at time t, drag frozen at ``lam = k_d vbar``."""
    ident = _eye_like(v0)
    emt = exp_decay_rotation(lam, om, t)
    mi = _minv(lam, om)
    g_vec = torch.zeros_like(v0)
    g_vec[:, 2] = -prm.g
    vel = (emt @ v0[:, :, None])[:, :, 0] + (mi @ ((emt - ident) @ g_vec[:, :, None]))[:, :, 0]
    a_mat = mi @ (emt - ident)
    b_vec = ((mi @ a_mat) - t[:, None, None] * mi) @ g_vec[:, :, None]
    pos = p0 + (a_mat @ v0[:, :, None])[:, :, 0] + b_vec[:, :, 0]
    return pos, vel


# --------------------------------------------------------------------------------------------
# honest fallback to the true solver
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def solve_with_fallback(
    p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
    fallback_fn=None, **kwargs
) -> dict:
    """``solve_analytic``, with the rows it declines re-solved by the true solver and COUNTED.

    ``fallback_fn(idx)`` takes a LongTensor of row indices and returns a dict with at least
    ``v_r``/``n``/``ok`` for those rows.  When ``fallback_fn`` is None the module imports
    ``solve_strike_specs`` lazily (relative import inside the package, else by file path next to
    this module) and calls it with the matching arguments — so the default fallback is the
    incumbent LM on the same physics, not a second approximation.

    Adds to the returned dict: ``fallback_used`` (N,) bool, ``fallback_rate`` (float),
    ``fallback_solved`` (int).  Rows the fallback also could not answer keep ``ok=False``.
    """
    ref_normal = kwargs.get("ref_normal")
    speed_budget = float(kwargs.get("speed_budget", 3.4))
    out = solve_analytic(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x, **kwargs)
    need = out["needs_fallback"]
    out["fallback_used"] = need.clone()
    out["fallback_rate"] = float(need.float().mean()) if need.numel() else 0.0
    out["fallback_solved"] = 0
    idx = torch.nonzero(need, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return out
    if fallback_fn is None:
        fallback_fn = _default_fallback(
            p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x, ref_normal, speed_budget)
    sub = fallback_fn(idx)
    out["v_r"][idx] = sub["v_r"].to(out["v_r"].dtype)
    out["n"][idx] = sub["n"].to(out["n"].dtype)
    out["ok"][idx] = sub["ok"]
    out["fallback_solved"] = int(sub["ok"].sum())
    return out


def _default_fallback(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
                      ref_normal, speed_budget):
    """Lazily bind ``strike_spec_torch.solve_strike_specs``; importable both ways, like this file."""
    try:
        from .strike_spec_torch import solve_strike_specs      # package import (online path)
    except ImportError:                                        # loaded by file path (bank builder)
        import importlib.util
        import os
        import sys
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strike_spec_torch.py")
        spec = importlib.util.spec_from_file_location("_sst_for_analytic_fallback", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        solve_strike_specs = mod.solve_strike_specs

    def _fn(idx):
        return solve_strike_specs(
            p_contact[idx], v_ball[idx], w_ball[idx], aim_xy[idx], prm,
            surface_z=surface_z, net_x=net_x,
            ref_normal=(None if ref_normal is None else ref_normal[idx]),
            speed_budget=speed_budget,
        )
    return _fn


# --------------------------------------------------------------------------------------------
# the free dof, made explicit
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def best_t_min_speed(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
                     t_lo=0.50, t_hi=0.90, n_grid=5, **kwargs):
    """Pick T per row to minimise the demanded racket speed — a fixed grid plus one parabola step.

    人话:T(飞行时间)是唯一还剩下的自由数,它就是"弧线高低 / 出多少力"的旋钮。这个函数把 LM 的
    正则项**声称**要做、但从来没做到的那个目标(最省力的拍速)真的做了:五个点扫一遍,再用抛物线
    顶点收一次。固定次数,没有搜索。

    Returns ``(t_star, speed_star)``.  |v_r|(T) is empirically unimodal on this box (single interior
    minimum on 99.85% of questions in the Understand-phase measurement) — that is an OBSERVATION,
    not a theorem, so the grid is a scan and the parabola is a refinement, not a proof of optimality.
    """
    dev, dt = p_contact.device, p_contact.dtype
    ts = torch.linspace(float(t_lo), float(t_hi), int(n_grid), device=dev, dtype=dt)
    speeds = []
    for t in ts:
        o = solve_analytic(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
                           t_flight=float(t), **kwargs)
        # illegal or non-finite rows are pushed out of the argmin rather than dropped, so the
        # returned T is always a real grid point and the caller still sees ok/reason per row
        s = torch.where(o["ok"], o["speed"], torch.full_like(o["speed"], 1e6))
        speeds.append(torch.where(torch.isfinite(s), s, torch.full_like(s, 1e6)))
    sp = torch.stack(speeds, dim=1)                                     # (N, n_grid)
    j = sp.argmin(dim=1)
    jc = j.clamp(1, int(n_grid) - 2)
    y0 = torch.gather(sp, 1, (jc - 1)[:, None])[:, 0]
    y1 = torch.gather(sp, 1, jc[:, None])[:, 0]
    y2 = torch.gather(sp, 1, (jc + 1)[:, None])[:, 0]
    step = float(ts[1] - ts[0])
    den = (y0 - 2.0 * y1 + y2)
    shift = torch.where(den.abs() > 1e-9, 0.5 * (y0 - y2) / den.clamp_min(1e-9) * step,
                        torch.zeros_like(den))
    t_star = (ts[jc] + shift.clamp(-step, step)).clamp(float(t_lo), float(t_hi))
    # interior minima only: if the grid minimum sits on an endpoint, keep the endpoint
    t_star = torch.where(j == jc, t_star, ts[j])
    o = solve_analytic(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
                       t_flight=t_star, **kwargs)
    return t_star, o["speed"]


@torch.no_grad()
def feasible_t_scan(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
                    t_grid=(0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00), **kwargs):
    """Which T values on a grid are legal, per row.  Returns (feasible (N,K) bool, t_grid tensor).

    NOT a proof that the feasible set is an interval.  It is a scan, and it is named a scan: the
    Understand phase asserted an interval and did not prove one, so this function reports the grid
    it actually evaluated and leaves connectedness to whoever wants to prove it.
    """
    dev, dt = p_contact.device, p_contact.dtype
    ts = torch.as_tensor(list(t_grid), device=dev, dtype=dt)
    cols = []
    for t in ts:
        o = solve_analytic(p_contact, v_ball, w_ball, aim_xy, prm, surface_z, net_x,
                           t_flight=float(t), **kwargs)
        cols.append(o["ok"])
    return torch.stack(cols, dim=1), ts
