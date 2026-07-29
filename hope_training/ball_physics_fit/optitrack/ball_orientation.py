"""
Ball ORIENTATION (and therefore spin) from an OptiTrack marker-ball export.

Correction of record, 2026-07-30: the ball on this session IS a rigid marker
constellation and its rotation is recoverable. Two earlier tests said otherwise
and both were wrong for instructive reasons:

  1. Pairwise-distance scatter indexed BY COLUMN read 6.2-6.9 mm. Invalid:
     Motive relabels markers on a small spinning ball, so column i is not the
     same physical marker across frames.
  2. The permutation-invariant SORTED-DISTANCE SPECTRUM read 2.2-2.8 mm against
     a wandering-points Monte-Carlo null of 2.48 mm, i.e. "indistinguishable
     from random". Also invalid: the 28 pairwise distances of this constellation
     span 11-40 mm, so adjacent sorted ranks differ by ~1 mm - comparable to the
     noise. Noise therefore SWAPS RANKS, and rank swapping inflates the per-rank
     std regardless of rigidity. The spectrum test is biased toward "non-rigid"
     whenever the distance spectrum is dense relative to the noise.

The valid test is consecutive-frame alignment, because 2.8 ms apart the ball has
barely rotated so the correspondence is unambiguous. Measured medians:

     take          consecutive-frame match rms
     chuntan       0.91 mm       rigid control (sigma=1.5 mm):  3.13 mm
     Tui           2.31 mm       WANDERING null:               10.37 mm
     zhengchang    3.01 mm
     zhengchang2   1.36 mm
     xuan          0.33 mm

All five sit far below the wandering null, and chuntan/xuan sit below a rigid
control carrying 1.5 mm of noise - so per-marker noise is well under 1 mm.

Method here: SEQUENTIAL tracking. Correspondence is propagated frame to frame
(predicting the next rotation from the current angular velocity and picking the
assignment closest to that prediction), never solved independently per frame -
independent per-frame matching against a template is exactly what collapsed the
template to 15.4 mm mean radius on a 20 mm ball in the first attempt.

ALIASING LIMIT: the smallest marker separation is ~11.6 mm on a 20 mm sphere,
i.e. ~33.6 deg of arc, so a blind per-frame solve is ambiguous beyond ~17 deg of
rotation per frame = ~17 rev/s at 360 Hz. Sequential prediction pushes past that
but not indefinitely; `spin_alias_risk` flags frames where the step exceeded the
half-separation bound.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot

R_BALL = 0.020


def kabsch(A, B):
    """Rotation taking centred A onto centred B."""
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _assign(Pc, Tc_rot):
    """Hungarian nearest-neighbour assignment between observed and predicted."""
    cost = np.linalg.norm(Pc[:, None, :] - Tc_rot[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return ri, ci, float(cost[ri, ci].mean())


def fit_pose(P_vis, T, R_pred, refine=3):
    """Pose of an observed marker cloud given a predicted rotation.

    P_vis: RAW visible marker positions (k,3), k >= 3. T: full template (M,3),
    already centred on the ball centre.

    Partial visibility is the normal case here (all 8 markers are seen on only
    ~11% of frames), and it is why centring matters: the centroid of a VISIBLE
    SUBSET is not the ball centre, so the observed subset must be compared
    against the centroid of the MATCHING template subset, not the full template.
    Getting that wrong biases every partial frame. We therefore alternate
    assignment and Kabsch, re-centring both sides on the matched subset each
    round.

    Returns (R, ri, ci, rms, centre) where centre is the implied ball centre.
    """
    R = R_pred
    c_obs = P_vis.mean(0)
    ri = ci = None
    for _ in range(refine + 1):
        # predicted marker positions about the current centre estimate
        ri, ci, _ = _assign(P_vis - c_obs, T @ R.T)
        A = T[ci] - T[ci].mean(0)
        B = P_vis[ri] - P_vis[ri].mean(0)
        R = kabsch(A, B)
        # ball centre implied by this correspondence
        c_obs = P_vis[ri].mean(0) - (T[ci].mean(0) @ R.T)
    resid = (P_vis[ri] - c_obs) - T[ci] @ R.T
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    return R, ri, ci, rms, c_obs


def free_sphere(Q):
    """Algebraic sphere fit with free radius. Returns (centre, radius)."""
    A = np.column_stack([2 * Q, np.ones(len(Q))])
    b = (Q ** 2).sum(1)
    s, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = s[:3]
    return c, float(np.sqrt(max(s[3] + c @ c, 0.0)))


def build_template(P, ok, rate, n_avg=200, radial_gate=0.0025):
    """Rigid marker template, taken from the CLEANEST full-visibility frame.

    Averaging over many frames was the first attempt and it collapsed the
    template (mean radius 15.9 mm and max chord 31 mm on a constellation whose
    observed max chord is 40 mm): a fraction of frames carry GHOST points -
    across all 795 full-visibility frames of the bounce take the radial spread
    about the fitted sphere has median 2.56 mm, while the best single frame has
    0.77 mm - and a handful of bad correspondences drags the mean inward.

    So: score every full frame by how well its markers sit on one sphere, take
    the best as the template, then refine ONLY over frames that agree with it.
    """
    M = P.shape[1]
    full = np.where(ok.sum(1) == M)[0]
    if len(full) < 10:
        raise ValueError(f"only {len(full)} full-visibility frames")
    scores = []
    for k in full:
        c, r = free_sphere(P[k])
        scores.append(np.std(np.linalg.norm(P[k] - c, axis=1)))
    scores = np.array(scores)
    order = full[np.argsort(scores)]
    best = order[0]
    c, r = free_sphere(P[best])
    T = P[best] - c

    # refine on the cleanest frames only, rejecting any that will not align
    acc = np.zeros((M, 3)); cnt = np.zeros(M)
    R = np.eye(3)
    used = 0
    for k in order[:n_avg]:
        R, ri, ci, rms, cc = fit_pose(P[k], T, R)
        if rms > radial_gate:
            continue
        acc[ci] += (P[k][ri] - cc) @ R
        cnt[ci] += 1
        used += 1
    seen = cnt > 0
    if seen.all() and used >= 10:
        T = np.where(seen[:, None], acc / np.maximum(cnt, 1)[:, None], T)
        T -= T.mean(0)
    radii = np.linalg.norm(T, axis=1)
    return T, dict(seed_frame=int(best), n_full=len(full), n_used=used,
                   radius_mm=float(np.median(radii) * 1e3),
                   radius_spread_mm=float(np.ptp(radii) * 1e3),
                   seed_radial_std_mm=float(scores.min() * 1e3))


def clean_markers(P_vis, radius, tol=0.004):
    """Drop ghost points that do not sit on the ball surface.

    Uses a free sphere fit and keeps points within `tol` of the expected radius;
    falls back to all points when too few survive.
    """
    if len(P_vis) < 4:
        return np.ones(len(P_vis), bool)
    c, r = free_sphere(P_vis)
    d = np.linalg.norm(P_vis - c, axis=1)
    keep = np.abs(d - radius) <= tol
    return keep if keep.sum() >= 3 else np.ones(len(P_vis), bool)


def solve_orientation_chained(P, ok, rate, min_markers=4, rms_gate=0.0035,
                              max_gap_frames=3, radius=None):
    """Ball orientation by CHAINING consecutive frames.

    Matching every frame independently against a fixed template fails here:
    the ball has rotated arbitrarily far from whichever frame seeded the
    template, so ICP lands in a local minimum (measured: 5.0 mm pose rms and
    only 17% of frames solved). Matching frame k to frame k+1 instead keeps the
    rotation per step at ~1.5 deg, where the correspondence is unambiguous
    (measured: 0.91 mm match rms).

    Absolute orientation is not needed anywhere downstream - `spin_from_quats`
    differentiates consecutive quaternions - so the chain is anchored at
    identity at the start of each unbroken run and drift within a run is
    irrelevant to spin.

    Returns quat (nf,4 xyzw, NaN outside runs), per-step rms, per-step omega,
    run ids, and the aliasing bookkeeping.
    """
    nf, M, _ = P.shape
    if radius is None:
        full = np.where(ok.sum(1) == M)[0]
        if len(full):
            radius = float(np.median([free_sphere(P[k])[1] for k in full[:400]]))
        else:
            radius = R_BALL
    quat = np.full((nf, 4), np.nan)
    step_rms = np.full(nf, np.nan)
    omega = np.full((nf, 3), np.nan)
    run_id = np.full(nf, -1, np.int32)

    vis = [np.where(ok[k])[0] for k in range(nf)]
    R_cur, prev_k, rid = None, None, -1
    for k in range(nf):
        if len(vis[k]) < min_markers:
            continue
        Pk = P[k][vis[k]]
        keep = clean_markers(Pk, radius)
        Pk = Pk[keep]
        if len(Pk) < min_markers:
            continue
        if prev_k is None or (k - prev_k) > max_gap_frames or R_cur is None:
            rid += 1
            R_cur = np.eye(3)
            quat[k] = Rot.from_matrix(R_cur).as_quat()
            run_id[k] = rid
            prev_k, P_prev = k, Pk
            continue
        # rotation taking the PREVIOUS cloud onto this one
        dR, ri, ci, rms, _c = fit_pose(Pk, P_prev - P_prev.mean(0), np.eye(3))
        if rms > rms_gate:
            prev_k, P_prev, R_cur = None, None, None
            continue
        dt = (k - prev_k) / rate
        R_cur = dR @ R_cur
        quat[k] = Rot.from_matrix(R_cur).as_quat()
        step_rms[k] = rms
        omega[k] = Rot.from_matrix(dR).as_rotvec() / dt
        run_id[k] = rid
        prev_k, P_prev = k, Pk
    return dict(quat=quat, step_rms=step_rms, omega=omega, run_id=run_id,
                radius_m=radius)


def solve_orientation(P, ok, rate, template=None, min_markers=4,
                      rms_gate=0.006, max_gap_frames=12):
    """Per-frame ball orientation against a FIXED template (absolute pose).

    Kept for reference; `solve_orientation_chained` is what the extractor uses,
    because absolute pose is not needed and template matching is fragile once
    the ball has rotated far from the seed frame.
    """
    nf, M, _ = P.shape
    if template is None:
        template, seed = build_template(P, ok, rate)
    else:
        seed = None
    tpl_radius = float(np.median(np.linalg.norm(template, axis=1)))
    quat = np.full((nf, 4), np.nan)
    rms = np.full(nf, np.nan)
    alias = np.zeros(nf, bool)

    # half of the smallest inter-marker separation, as an angle: beyond this a
    # blind step is ambiguous
    d = np.linalg.norm(template[:, None] - template[None, :], axis=2)
    dmin = np.min(d[np.triu_indices(M, 1)])
    rad = float(np.median(np.linalg.norm(template, axis=1)))
    ang_limit = 2 * np.arcsin(min(dmin / (2 * rad), 1.0)) / 2

    R_prev, k_prev, w_prev = None, None, np.zeros(3)
    for k in range(nf):
        n = int(ok[k].sum())
        if n < min_markers:
            continue
        P_vis = P[k][ok[k]]
        keep = clean_markers(P_vis, tpl_radius)
        P_vis = P_vis[keep]
        if len(P_vis) < min_markers:
            continue
        if R_prev is not None and (k - k_prev) <= max_gap_frames:
            dt = (k - k_prev) / rate
            R_pred = Rot.from_rotvec(w_prev * dt).as_matrix() @ R_prev
        else:
            R_pred = np.eye(3) if R_prev is None else R_prev
        R, ri, ci, r, _c = fit_pose(P_vis, template, R_pred)
        if r > rms_gate:
            continue
        quat[k] = Rot.from_matrix(R).as_quat()
        rms[k] = r
        if R_prev is not None and (k - k_prev) <= max_gap_frames:
            dR = R @ R_prev.T
            rv = Rot.from_matrix(dR).as_rotvec()
            dt = (k - k_prev) / rate
            if np.linalg.norm(rv) > ang_limit:
                alias[k] = True
            w_prev = rv / dt
        R_prev, k_prev = R, k
    return dict(quat=quat, rms=rms, template=template, alias_risk=alias,
                ang_limit_deg=float(np.degrees(ang_limit)),
                alias_spin_limit_rev_s=float(ang_limit / (2 * np.pi) * rate),
                template_radius_mm=float(rad * 1e3), seed=seed)
