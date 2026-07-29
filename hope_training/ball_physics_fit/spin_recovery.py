"""
Spin recovery for a capture with NO ball orientation channel.

A fully mocap-coated ball has no rigid marker template, so its exported markers
wander on the surface: position is measurable, orientation is not. The venue
pipeline read spin from Kabsch quaternions; on this session that channel does
not exist at all, and `spin_from_quats` correctly returns NaN everywhere.

Spin is still OBSERVABLE, just indirectly, through two couplings:

  (A) IN FLIGHT -- Magnus bends the arc.  a = g - k_d|v|v + M x v  with
      M = k_m * omega  (units 1/s).  Fitting M per arc recovers k_m*omega
      WITHOUT needing k_m.  Only the component of M perpendicular to v does
      anything, so the v-parallel component is unobservable and is held near
      zero by a ridge term.  This is the same estimator that made the
      spin-blind gravity fit read |g| = 9.70 -> 10.45 m/s^2 as ball speed rose
      (2026-07-30 Stage 0): that speed-dependent excess IS the Magnus term.

  (B) ACROSS A TABLE BOUNCE -- the tangential impulse that changes v_t also
      changes omega, by a ratio fixed by the ball's inertia:
          dOmega = -(1/(C*R)) * n x dV_t ,  C = 2/5 -> here C_INERTIA = 2/3
      used by contact_model (thin spherical shell).  dV_t is measured well
      (it is a velocity difference, not a spin), so dOmega is KNOWN in physical
      units.  Comparing it with the Magnus jump (M_after - M_before) recovered
      from the arcs on each side gives

          k_m = |M_after - M_before| / |dOmega|

      i.e. an absolute calibration of the Magnus coefficient from geometry the
      capture measures directly.  That is what turns k_m*omega into omega.

Outputs <ANALYSIS>/fits/spin_recovery.json:
  - per-arc Magnus vector M and its half-arc split-half consistency
  - a synthetic recovery test (inject known M at the measured noise floor)
  - the bounce-chain estimate of k_m with a bootstrap CI
  - the resulting spin distribution in rev/s, by take and by stroke context

Usage: python spin_recovery.py [--kd 0.1253] [--min-dur 0.25]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

import paths

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ballcore import (TAKES, load_take, extract_arcs, arc_parabola, arc_is_ballistic,
                      R_BALL, G_NOM)

MAX_NFEV = 120          # enough to converge these 9-param blocks; 400 does not finish
C_INERTIA = 2.0 / 3.0          # matches contact_model.py (thin shell)
OUT = os.path.join(paths.ANALYSIS, "fits")


# ------------------------------------------------------------------ physics

def rk4_M(p0, v0, M, kd, g, dt, n):
    """Integrate a = g - kd|v|v + M x v. Returns positions (n+1, 3)."""
    P = np.empty((n + 1, 3))
    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    P[0] = p

    def acc(v):
        return g - kd * np.linalg.norm(v) * v + np.cross(M, v)

    for k in range(n):
        a1 = acc(v)
        v2 = v + 0.5 * dt * a1; a2 = acc(v2)
        v3 = v + 0.5 * dt * a2; a3 = acc(v3)
        v4 = v + dt * a3; a4 = acc(v4)
        p = p + (dt / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
        v = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        P[k + 1] = p
    return P


def rk4_M_batch(p0, v0, M, kd, g, dt, n):
    """Vectorised over arcs: p0,v0,M (N,3) -> positions (N, n+1, 3)."""
    N = p0.shape[0]
    P = np.empty((N, n + 1, 3))
    p, v = p0.copy(), v0.copy()
    P[:, 0] = p

    def acc(v):
        s = np.linalg.norm(v, axis=1, keepdims=True)
        return g - kd * s * v + np.cross(M, v)

    for k in range(n):
        a1 = acc(v)
        v2 = v + 0.5 * dt * a1; a2 = acc(v2)
        v3 = v + 0.5 * dt * a2; a3 = acc(v3)
        v4 = v + dt * a3; a4 = acc(v4)
        p = p + (dt / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
        v = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        P[:, k + 1] = p
    return P


def fit_M_batch(seqs, kd, ridge=2.0, g=None, verbose=True):
    """Fit (p0, v0, M) for MANY arcs at once.

    Each arc's residual block touches only its own 9 unknowns, so handing
    least_squares a block-diagonal jac_sparsity turns the numerical Jacobian
    from 9N residual evaluations into ~9, and the residual itself is one
    vectorised RK4 over the padded stack. Per-arc `lm` fitting the same set
    did not finish in 10 minutes; this returns in seconds.

    seqs: list of (t, pos). Returns list of dicts(M, p0, v0, rms).
    """
    from scipy.sparse import lil_matrix
    g = np.array([0.0, 0.0, -G_NOM]) if g is None else g
    N = len(seqs)
    lens = np.array([len(t) for t, _ in seqs])
    nmax = int(lens.max())
    dt = float(np.median(np.diff(seqs[0][0])))
    obs = np.zeros((N, nmax, 3))
    mask = np.zeros((N, nmax), bool)
    x0 = np.zeros(9 * N)
    for i, (t, pos) in enumerate(seqs):
        obs[i, :len(t)] = pos
        mask[i, :len(t)] = True
        x0[9 * i:9 * i + 3] = pos[0]
        x0[9 * i + 3:9 * i + 6] = np.polyfit(t - t[0], pos, 1)[0]

    nres = 3 * int(mask.sum()) + 3 * N
    flat = mask.ravel()

    def resid(x):
        X = x.reshape(N, 9)
        P = rk4_M_batch(X[:, :3], X[:, 3:6], X[:, 6:9], kd, g, dt, nmax - 1)
        d = ((P - obs).reshape(N * nmax, 3)[flat]).ravel()
        return np.concatenate([d, ridge * 1e-3 * X[:, 6:9].ravel()])

    S = lil_matrix((nres, 9 * N), dtype=np.int8)
    off = 0
    for i in range(N):
        k = 3 * lens[i]
        S[off:off + k, 9 * i:9 * i + 9] = 1
        off += k
    for i in range(N):
        S[off + 3 * i:off + 3 * i + 3, 9 * i + 6:9 * i + 9] = 1
    sol = least_squares(resid, x0, jac_sparsity=S.tocsr(), verbose=2 if verbose else 0,
                        x_scale="jac", max_nfev=MAX_NFEV)
    X = sol.x.reshape(N, 9)
    P = rk4_M_batch(X[:, :3], X[:, 3:6], X[:, 6:9], kd, g, dt, nmax - 1)
    out = []
    for i in range(N):
        r = P[i, :lens[i]] - obs[i, :lens[i]]
        out.append(dict(M=X[i, 6:9], p0=X[i, :3], v0=X[i, 3:6],
                        rms=float(np.sqrt((r ** 2).sum(1).mean()))))
    return out


def fit_M(t, pos, kd, ridge=2.0, g=None):
    """Fit (p0, v0, M) on one arc. Returns dict(M, v0, rms, cond_ratio).

    ridge penalises |M| weakly (1/s units) so the component parallel to v --
    which produces no force and is therefore unobservable -- relaxes to zero
    instead of running off to absorb noise.
    """
    g = np.array([0.0, 0.0, -G_NOM]) if g is None else g
    dt = float(np.median(np.diff(t)))
    n = len(t) - 1
    v0g = np.polyfit(t - t[0], pos, 1)[0]

    def resid(x):
        P = rk4_M(x[:3], x[3:6], x[6:9], kd, g, dt, n)
        return np.concatenate([(P - pos).ravel(), ridge * 1e-3 * x[6:9]])

    x0 = np.concatenate([pos[0], v0g, np.zeros(3)])
    sol = least_squares(resid, x0, method="lm", max_nfev=4000)
    r = sol.fun[:3 * len(t)].reshape(-1, 3)
    rms = float(np.sqrt((r ** 2).sum(1).mean()))
    return dict(M=sol.x[6:9], p0=sol.x[:3], v0=sol.x[3:6], rms=rms)


# ------------------------------------------------------------- arc handling

def collect_arcs(surface_z, kd, min_dur, max_rms=0.030):
    cands = []
    for name in TAKES:
        take = load_take(name)
        for ai, a in enumerate(extract_arcs(take, table_z=surface_z)):
            dur = a["t"][-1] - a["t"][0]
            if dur < min_dur or len(a["t"]) < 25:
                continue
            if not np.isfinite(a["pos"]).all():
                continue
            if arc_parabola(a)["rms"] > max_rms:
                continue
            cands.append(dict(take=name, arc=ai, t0=float(a["t"][0]),
                              t1=float(a["t"][-1]), dur=float(dur),
                              n=len(a["t"]), pre=a["pre_contact"],
                              post=a["post_contact"], t_arr=a["t"], pos=a["pos"]))
    print(f"  candidate arcs: {len(cands)}; batched Magnus fit...", flush=True)
    fits = fit_M_batch([(c["t_arr"], c["pos"]) for c in cands], kd)
    arcs = []
    for c, f in zip(cands, fits):
        if f["rms"] > max_rms:
            continue
        c.update(M=f["M"], v0=f["v0"], rms=f["rms"],
                 speed=float(np.linalg.norm(f["v0"])))
        arcs.append(c)
    return arcs


def split_half_consistency(arcs, kd):
    """Fit M on each half of every arc; the scatter between halves is the
    estimator's own repeatability, independent of any physics assumption."""
    pool = [a for a in arcs if a["n"] >= 60]
    if not pool:
        return []
    h1 = fit_M_batch([(a["t_arr"][:a["n"] // 2], a["pos"][:a["n"] // 2])
                      for a in pool], kd, verbose=False)
    h2 = fit_M_batch([(a["t_arr"][a["n"] // 2:], a["pos"][a["n"] // 2:])
                      for a in pool], kd, verbose=False)
    return [dict(take=a["take"], full=a["M"].tolist(),
                 h1=f1["M"].tolist(), h2=f2["M"].tolist(),
                 diff=float(np.linalg.norm(f1["M"] - f2["M"])))
            for a, f1, f2 in zip(pool, h1, h2)]


def synthetic_check(arcs, kd, noise_mm, n_trials=80, seed=0,
                    levels=(0.10, 0.30, 0.60, 1.00), km_ref=0.00444):
    """Detection-limit curve: inject KNOWN Magnus vectors of several magnitudes
    into real arc geometries at the measured noise floor, then re-fit.

    `levels` are |M| in 1/s. With the venue k_m = 0.00444 they correspond to
    roughly 3.6 / 10.8 / 21.5 / 35.8 rev/s -- i.e. from "barely spinning" to a
    solid amateur topspin. Reporting recovery as a function of level is what
    distinguishes "this rig cannot see spin at all" from "this rig cannot see
    SMALL spin", which are very different conclusions for the sim.
    """
    rng = np.random.default_rng(seed)
    g = np.array([0.0, 0.0, -G_NOM])
    pool = [a for a in arcs if a["n"] >= 60]
    if not pool:
        return []
    out = []
    for lvl in levels:
        seqs, truths = [], []
        for _ in range(n_trials):
            a = pool[rng.integers(0, len(pool))]
            dt = float(np.median(np.diff(a["t_arr"])))
            d = rng.normal(0, 1, 3)
            M_true = lvl * d / np.linalg.norm(d)
            P = rk4_M(a["pos"][0], a["v0"], M_true, kd, g, dt, a["n"] - 1)
            P = P + rng.normal(0, noise_mm * 1e-3, P.shape)
            seqs.append((a["t_arr"], P))
            truths.append(M_true)
        fits = fit_M_batch(seqs, kd, verbose=False)
        T = np.array(truths)
        F = np.array([f["M"] for f in fits])
        err = np.linalg.norm(F - T, axis=1)
        out.append(dict(
            level_1_s=lvl, rev_s_equiv=lvl / km_ref / (2 * np.pi), n=n_trials,
            err_median=float(np.median(err)),
            err_over_signal=float(np.median(err) / lvl),
            corr_per_axis=[float(np.corrcoef(T[:, i], F[:, i])[0, 1]) for i in range(3)],
            mag_true_median=float(np.median(np.linalg.norm(T, axis=1))),
            mag_fit_median=float(np.median(np.linalg.norm(F, axis=1)))))
    return out


# --------------------------------------------------- bounce chain -> k_m

def bounce_chain(arcs, bounces, max_dt=0.06):
    """Pair each table bounce with the arc ending just before and starting just
    after it, then compare the Magnus jump with the spin jump implied by the
    measured tangential velocity change.

    Contact model (contact_model.predict_contact, thin shell):
        dOmega = -(1/(C*R)) * (n x dV_t)
    so |dOmega| = |dV_t| / (C*R) for a horizontal table (n = z).
    """
    rows = []
    for b in bounces:
        pre = [a for a in arcs if a["take"] == b["take"]
               and 0 <= b["t_c"] - a["t1"] <= max_dt]
        post = [a for a in arcs if a["take"] == b["take"]
                and 0 <= a["t0"] - b["t_c"] <= max_dt]
        if not pre or not post:
            continue
        pre = min(pre, key=lambda a: b["t_c"] - a["t1"])
        post = min(post, key=lambda a: a["t0"] - b["t_c"])
        v_in = np.array(b["v_in"], float)
        v_out = np.array(b["v_out"], float)
        n = np.array([0.0, 0.0, 1.0])
        dV = v_out - v_in
        dV_t = dV - np.dot(dV, n) * n
        dOmega = -np.cross(n, dV_t) / (C_INERTIA * R_BALL)
        dM = post["M"] - pre["M"]
        if np.linalg.norm(dOmega) < 5.0:          # too small to calibrate on
            continue
        # projection of the measured Magnus jump onto the predicted spin jump
        u = dOmega / np.linalg.norm(dOmega)
        rows.append(dict(take=b["take"], t_c=b["t_c"],
                         dOmega=dOmega.tolist(), dOmega_mag=float(np.linalg.norm(dOmega)),
                         dM=dM.tolist(), dM_mag=float(np.linalg.norm(dM)),
                         dM_along=float(np.dot(dM, u)),
                         km_ratio=float(np.dot(dM, u) / np.linalg.norm(dOmega)),
                         vn_in=b["vn_in"], e_n=b["e_n"],
                         pre_dur=pre["dur"], post_dur=post["dur"],
                         pre_rms_mm=pre["rms"] * 1e3, post_rms_mm=post["rms"] * 1e3))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd", type=float, default=None,
                    help="drag coefficient (default: read stage2 fit)")
    ap.add_argument("--min-dur", type=float, default=0.25)
    ap.add_argument("--noise-mm", type=float, default=None,
                    help="position noise for the synthetic check (default: measured)")
    args = ap.parse_args()

    seg = os.path.join(paths.ANALYSIS, "segments")
    meta = json.load(open(os.path.join(seg, "meta.json")))
    bounces = json.load(open(os.path.join(seg, "bounces.json")))
    kd = args.kd
    if kd is None:
        s2 = json.load(open(os.path.join(OUT, "stage2_fits_all.json")))
        kd = s2["kd"]["kd"]
    print(f"kd = {kd:.4f}, surface_z = {meta['surface_z']*1e3:.1f} mm")

    arcs = collect_arcs(meta["surface_z"], kd, args.min_dur)
    print(f"arcs with a fitted Magnus vector: {len(arcs)}")
    noise_mm = args.noise_mm or float(np.median([a["rms"] for a in arcs]) * 1e3)

    halves = split_half_consistency(arcs, kd)
    synth = synthetic_check(arcs, kd, noise_mm)
    for row in synth:
        print(f"  synth |M|={row['level_1_s']:.2f} (~{row['rev_s_equiv']:.1f} rev/s): "
              f"err/signal={row['err_over_signal']:.2f} corr={[round(c,2) for c in row['corr_per_axis']]}",
              flush=True)
    chain = bounce_chain(arcs, bounces)
    print(f"split-half pairs {len(halves)}, synthetic trials {len(synth)}, "
          f"bounce-chain events {len(chain)}")

    km = None
    if len(chain) >= 10:
        ratios = np.array([r["km_ratio"] for r in chain])
        rng = np.random.default_rng(0)
        boots = [float(np.median(ratios[rng.integers(0, len(ratios), len(ratios))]))
                 for _ in range(2000)]
        km = dict(km_median=float(np.median(ratios)),
                  km_mean=float(np.mean(ratios)),
                  ci95=[float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5))],
                  n=len(ratios),
                  dOmega_median_rads=float(np.median([r["dOmega_mag"] for r in chain])))
        print(f"k_m from the bounce chain = {km['km_median']:.5f} "
              f"CI95 {km['ci95']}  (n={km['n']})")

    Mmag = np.array([float(np.linalg.norm(a["M"])) for a in arcs])
    spin = {}
    if km and km["km_median"] > 1e-6:
        rev = Mmag / km["km_median"] / (2 * np.pi)
        spin = dict(rev_s_median=float(np.median(rev)),
                    rev_s_p10=float(np.percentile(rev, 10)),
                    rev_s_p90=float(np.percentile(rev, 90)),
                    rev_s_max=float(rev.max()))
        print(f"spin: median {spin['rev_s_median']:.1f} rev/s, "
              f"p90 {spin['rev_s_p90']:.1f}, max {spin['rev_s_max']:.1f}")

    res = dict(
        kd=kd, surface_z=meta["surface_z"], min_dur=args.min_dur,
        n_arcs=len(arcs),
        arc_rms_median_mm=float(np.median([a["rms"] for a in arcs]) * 1e3),
        M_mag=dict(median=float(np.median(Mmag)),
                   p10=float(np.percentile(Mmag, 10)),
                   p90=float(np.percentile(Mmag, 90))),
        split_half=dict(n=len(halves),
                        diff_median=float(np.median([h["diff"] for h in halves]))
                        if halves else None,
                        rows=halves[:200]),
        synthetic=dict(noise_mm=noise_mm, levels=synth),
        bounce_chain=dict(n=len(chain), km=km, rows=chain),
        spin_rev_s=spin,
        per_arc=[dict(take=a["take"], arc=a["arc"], t0=a["t0"], dur=a["dur"],
                      speed=a["speed"], M=a["M"].tolist(),
                      M_mag=float(np.linalg.norm(a["M"])), rms_mm=a["rms"] * 1e3,
                      pre=a["pre"], post=a["post"]) for a in arcs],
    )
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "spin_recovery.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
