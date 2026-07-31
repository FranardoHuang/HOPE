"""Batched torch mirror of the planner's stroke SELECTOR + motion ADAPTER.

人话:规划器那边 (``hope_planner/stroke_select.py`` + ``stroke_adapt.py``) 是一颗球一颗球算的
numpy 版;这边是训练用的 batch 版,判据、公式、拒绝理由的字符串全都一模一样,读的也是同一份
动作模板文件。两边的一致性由 ``tests/test_stroke_select_torch.py`` 逐行对拍。

Frames: everything env-local, z above the FLOOR (``W_floor``) — the same convention as
``strike_spec_torch`` / ``coarse_landing`` / ``_vb_evaluate``. Contact-region x/y bands are
``B_yaw`` (base origin, yaw-only), so a ball position is converted with the base pose the caller
hands in. No fourth frame is invented.

The adapter's inverse solve is ``solve_strike_specs_fixed_dir``: the sibling of
``strike_spec_torch.solve_strike_specs`` whose ONLY structural difference is how the racket
velocity is built —

    free solve:   v_r = q[2] * n + q[3] * b1 + q[4] * b2     (face basis; a tilt DRAGS the velocity)
    fixed dir:    v_r = q[2] * d_hat_w                       (world direction, held fixed)

which is exactly what makes "the stroke's velocity direction is its identity" expressible.
"""

from __future__ import annotations

import math

import torch

from .strike_spec_torch import _face_from_angles, _seed
from .virtual_ball import VirtualBallParams, coarse_landing, predict_paddle_contact

_EPS = 1e-9

# Reject-reason codes, byte-identical to hope_planner.stroke_adapt / stroke_select so a training
# histogram and a deploy log can be compared directly.
REASONS = (
    "no_landing", "resid_gt_tol", "speed_over_cap", "speed_under_min",
    "dir_cone_exceeded", "net_not_cleared", "face_not_opponent_facing",
)
PREDICATES = (
    "P0_disabled", "P1_side", "P2_time_too_short", "P2_time_too_long",
    "P3_height", "P4_reach", "P5a_energy_insufficient", "P5b_energy_excess", "P6_runway",
)

TABLE_SURFACE_Z_W_FLOOR = 0.76
BALL_RADIUS_M = 0.02

# Private capability for the diagnostic ActionBall producer.  The ordinary
# public/formal solver keeps the checked ``torch.linalg.solve`` path and its
# early exit byte-for-byte below.  An exact identity (rather than a bool)
# prevents a typo-shaped truthy value from silently selecting the no-host-sync
# diagnostic path.
_DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY = object()


# ------------------------------------------------------------------ frames --- #
def base_yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,) yaw. Same expression as pp_policy.hpp:2362 / base_yaw_relative_y."""
    q = quat_wxyz
    return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def to_b_yaw(p_w: torch.Tensor, base_p_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    d = p_w - base_p_w
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], dim=-1)


def direction_world(v_hat_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """(K,3) B_yaw directions x (N,) yaw -> (N,K,3) world directions, R_z(yaw) . v_hat_b."""
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    vx, vy, vz = v_hat_b[None, :, 0], v_hat_b[None, :, 1], v_hat_b[None, :, 2]
    d = torch.stack([c * vx - s * vy, s * vx + c * vy, vz.expand_as(c * vx)], dim=-1)
    return d / (torch.linalg.norm(d, dim=-1, keepdim=True) + _EPS)


# ------------------------------------------------------------------ closing speed --- #
def closing_speed_demand(
    p_ball: torch.Tensor,
    v_ball: torch.Tensor,
    aim_xy: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    ball_radius: float = BALL_RADIUS_M,
    delta_t_flight: float = 0.45,
):
    """(v_n_req (N,), n_hat0 (N,3)) — the racket normal-channel speed this ball+aim demands.

    Exactly the shipped contact model: in ``predict_paddle_contact`` the lever arm is
    ``r = -R n_hat`` so ``(omega x r) . n_hat == 0`` and the normal channel is closed-form,

        v_out . n = -e (v_ball . n) + (1 + e) (v_r . n)
        =>  v_n_req = (b + e a) / (1 + e),  e = clamp(g1 exp(g2 |a - v_n_req|), 0.05, 0.95)

    with a = v_ball . n (negative for an incoming ball), b = v_out . n. Solved by the same 3-step
    fixed point the LM seeds with, so the selector's screen and the solver's seed are one number.
    """
    T = float(delta_t_flight)
    p_land = torch.cat([aim_xy, torch.full_like(aim_xy[:, :1], surface_z + ball_radius)], dim=-1)
    g = torch.zeros_like(p_ball)
    g[:, 2] = -prm.g
    v_out = (p_land - p_ball) / T - 0.5 * g * T
    dv = v_out - v_ball
    n0 = dv / (torch.linalg.norm(dv, dim=-1, keepdim=True) + _EPS)
    n0 = torch.where(n0[:, 0:1] < 0.0, -n0, n0)          # opponent-facing (+x)
    a = torch.sum(v_ball * n0, dim=-1)
    b = torch.sum(v_out * n0, dim=-1)
    e = torch.full_like(a, 0.5)
    v_n = (b + e * a) / (1.0 + e)
    for _ in range(3):
        u_n = torch.abs(a - v_n)
        e = (prm.paddle_e_g1 * torch.exp(prm.paddle_e_g2 * u_n)).clamp(0.05, 0.95)
        v_n = (b + e * a) / (1.0 + e)
    return v_n, n0


# ------------------------------------------------------------------ selector --- #
@torch.no_grad()
def select_stroke_batch(
    p_ball: torch.Tensor,          # (N,3) at-strike ball position, env-local W_floor
    v_ball: torch.Tensor,          # (N,3)
    t_avail: torch.Tensor,         # (N,)  latency-compensated time to strike, s
    base_pos: torch.Tensor,        # (N,3) env-local W_floor
    base_quat: torch.Tensor,       # (N,4) wxyz
    aim_xy: torch.Tensor,          # (N,2)
    protos,                        # StrokePrototypeTensors
    prm: VirtualBallParams,
    surface_z: float,
    split_y: float = 0.0,
    hysteresis_y: float = 0.04,
    wait_slack_s: float = 1.0,
    eps_energy: float = 0.10,
    ball_radius: float = BALL_RADIUS_M,
    delta_t_flight: float = 0.45,
):
    """Per-env stroke choice. Returns dict with ``clip_index`` (N,) long (-1 = none admissible),
    ``admissible`` (N,K) bool, ``reject_code`` (N,K) long (index into PREDICATES, -1 = admitted),
    ``v_n_req`` (N,), ``rank`` (N,K) long — the total order of §3.3 as a permutation of clip ids.

    Every predicate is an inequality on a MEASURED quantity; there is no weight to tune. The only
    human knob is one integer per stroke (``priority``).
    """
    N = p_ball.shape[0]
    K = len(protos)
    dev, dt = p_ball.device, p_ball.dtype
    yaw = base_yaw_of(base_quat)
    p_b = to_b_yaw(p_ball, base_pos, yaw)                       # (N,3)
    z_floor_off = TABLE_SURFACE_Z_W_FLOOR - float(surface_z)
    z_w = p_ball[:, 2] + z_floor_off                            # (N,)
    v_n_req, n0 = closing_speed_demand(
        p_ball, v_ball, aim_xy, prm, surface_z, ball_radius, delta_t_flight)
    d_w = direction_world(protos.v_hat_b.to(dev, dt), yaw)      # (N,K,3)
    proj = torch.sum(d_w * n0[:, None, :], dim=-1)              # (N,K)

    x_b = p_b[:, 0:1]
    y_b = p_b[:, 1:2]
    vn = v_n_req[:, None]
    ta = t_avail[:, None]
    is_fh = (protos.family_sign.to(dev, dt) > 0.0)[None, :]

    # Codes match PREDICATES order; the FIRST failing predicate is the reported reason.
    fails = [
        ~protos.enabled.to(dev)[None, :].expand(N, K),                       # P0
        torch.where(is_fh, ~(y_b < split_y + hysteresis_y),
                    ~(y_b > split_y - hysteresis_y)),                        # P1
        ta < protos.t_prepare_min.to(dev, dt)[None, :],                      # P2 short
        ta > protos.t_prepare_max.to(dev, dt)[None, :] + wait_slack_s,       # P2 long
        None,                                                                # P3, below
        None,                                                                # P4, below
        None, None, None,                                                    # P5a/P5b/P6, below
    ]
    z_lo = torch.maximum(
        protos.band_z_w.to(dev, dt)[:, 0] - protos.slack_z_w.to(dev, dt),
        torch.full((K,), TABLE_SURFACE_Z_W_FLOOR + ball_radius, device=dev, dtype=dt),
    )[None, :]
    z_hi = (protos.band_z_w.to(dev, dt)[:, 1] + protos.slack_z_w.to(dev, dt))[None, :]
    fails[4] = ~((z_w[:, None] >= z_lo) & (z_w[:, None] <= z_hi))
    sl = protos.slack_b_xy.to(dev, dt)[None, :]
    bx = protos.band_b_x.to(dev, dt)
    by = protos.band_b_y.to(dev, dt)
    fails[5] = ~(
        (x_b >= bx[None, :, 0] - sl) & (x_b <= bx[None, :, 1] + sl)
        & (y_b >= by[None, :, 0] - sl) & (y_b <= by[None, :, 1] + sl)
    )
    supply_max = protos.speed_max.to(dev, dt)[None, :] * proj.clamp(min=0.0)
    supply_min = protos.speed_min.to(dev, dt)[None, :] * proj
    fails[6] = vn > supply_max + eps_energy
    fails[7] = vn < supply_min - eps_energy
    fails[8] = vn / proj.clamp(min=1e-6) > protos.v_star_cap.to(dev, dt)[None, :] + eps_energy

    reject = torch.full((N, K), -1, dtype=torch.long, device=dev)
    for code in range(len(fails) - 1, -1, -1):
        reject = torch.where(fails[code], torch.full_like(reject, code), reject)
    adm = reject < 0

    # Total order: (priority, -time_margin, demand, clip_index). clip_index last makes it total.
    time_margin = (ta - protos.t_prepare_min.to(dev, dt)[None, :]) / \
        protos.t_prepare.to(dev, dt)[None, :].clamp(min=1e-9)
    demand = vn / (protos.speed_max.to(dev, dt)[None, :] * proj.clamp(min=1e-6)).clamp(min=1e-6)
    idx = torch.arange(K, device=dev, dtype=dt)[None, :].expand(N, K)
    # Pack the lexicographic key into one float: the fields are bounded and the packing weights are
    # 4 orders of magnitude apart, so ordering is preserved without a python sort.
    key = (protos.priority.to(dev, dt)[None, :] * 1e6
           - time_margin.clamp(-1e3, 1e3) * 1e3
           + demand.clamp(0.0, 1e3)
           + idx * 1e-4)
    key = torch.where(adm, key, torch.full_like(key, float("inf")))
    rank = torch.argsort(key, dim=-1)
    best = rank[:, 0]
    clip_index = torch.where(adm.any(dim=-1), best, torch.full_like(best, -1))
    return {
        "clip_index": clip_index, "admissible": adm, "reject_code": reject,
        "v_n_req": v_n_req, "n_hat0": n0, "d_hat_w": d_w, "proj": proj, "rank": rank,
    }


# ------------------------------------------------------------------ adapter --- #
def _forward_landing_fixed_dir(q, d_hat_w, p_strike, v_ball, w_ball, prm, surface_z, net_x,
                               h, n_steps):
    """q (N,3) = (theta, phi, s) -> (landing_xy, valid, v_r, n, net_z).

    Sibling of ``strike_spec_torch._forward_landing``; the ONE change is ``v_r = s * d_hat_w``.
    """
    n, _, _ = _face_from_angles(q[:, 0], q[:, 1])
    v_r = q[:, 2:3] * d_hat_w
    v_plus, w_plus = predict_paddle_contact(v_ball, v_r, n, w_ball, prm)
    land = coarse_landing(p_strike, v_plus, w_plus, prm, surface_z=surface_z, net_x=net_x,
                          h=h, n_steps=n_steps)
    return land["land_xy"], land["land_valid"], v_r, n, land["net_z"]


@torch.no_grad()
def solve_strike_specs_fixed_dir(
    p_strike: torch.Tensor,
    v_ball: torch.Tensor,
    w_ball: torch.Tensor,
    target_xy: torch.Tensor,
    d_hat_w: torch.Tensor,           # (N,3) unit; the stroke's identity, held FIXED
    speed_min: torch.Tensor,         # (N,)
    speed_max: torch.Tensor,         # (N,)
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_height: float = 0.1525,
    net_margin_m: float = 0.02,
    ball_radius: float = BALL_RADIUS_M,
    s0: torch.Tensor | None = None,
    n_iters: int = 12,
    tol_m: float = 0.02,
    w_speed: float = 0.03,
    delta_t_flight: float = 0.45,
    h: float = 0.01,
    n_steps: int = 100,
    _diagnostic_fixed_try_lm_authority=None,
) -> dict:
    """Batched fixed-direction inverse solve — the training-side motion adapter.

    THE FOUR CONSTRAINTS
      C1 direction: ``v_r = s * d_hat_w`` — exact by construction, not a variable.
      C2 contact:   ``p_contact = p_strike`` — the BALL's own point; returned verbatim, and on
                    failures too.
      C3 speed:     ``s in [speed_min, speed_max]``; out of range is ``ok=False``, NEVER a clamped
                    accepted command.
      C4 face:      ``n(theta, phi)`` both solved and emitted EXPLICITLY. A face implied by the
                    velocity degenerates into lofting the ball, and the landing reward — which
                    scores only WHERE the ball lands — cannot see that.

    Acceptance additionally requires NET CLEARANCE, reusing the trainer's own rule
    (``net_clear = net_valid & (net_z > net_top + ball_radius)``, hope_commands:4138) plus a margin.
    """
    if (
        _diagnostic_fixed_try_lm_authority is not None
        and _diagnostic_fixed_try_lm_authority
        is not _DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY
    ):
        raise RuntimeError(
            "diagnostic fixed-try LM solve requires the exact private authority"
        )
    diagnostic_fixed_try_lm = (
        _diagnostic_fixed_try_lm_authority
        is _DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY
    )
    N = p_strike.shape[0]
    dev, dt = p_strike.device, p_strike.dtype
    d_hat_w = d_hat_w / (torch.linalg.norm(d_hat_w, dim=-1, keepdim=True) + _EPS)

    def fwd(qq):
        return _forward_landing_fixed_dir(qq, d_hat_w, p_strike, v_ball, w_ball, prm,
                                          surface_z, net_x, h, n_steps)

    diagnostic_candidate_inputs = {}

    def fwd_diagnostic_candidates(qq):
        """Evaluate candidate-major ``(C,N,3)`` rows in one identical flat batch.

        The fixed-direction forward model has no cross-row reductions.  Flattening candidate
        rows therefore changes only launch grouping: every physical row still receives the same
        q/direction/ball/target values and executes the same fused rollout arithmetic.
        """

        candidate_count = int(qq.shape[0])

        def repeat_candidates(rows):
            return (
                rows.unsqueeze(0)
                .expand(candidate_count, *rows.shape)
                .reshape(candidate_count * N, *rows.shape[1:])
            )

        repeated_inputs = diagnostic_candidate_inputs.get(
            candidate_count
        )
        if repeated_inputs is None:
            repeated_inputs = (
                repeat_candidates(d_hat_w),
                repeat_candidates(p_strike),
                repeat_candidates(v_ball),
                repeat_candidates(w_ball),
            )
            diagnostic_candidate_inputs[candidate_count] = (
                repeated_inputs
            )
        flat = qq.reshape(candidate_count * N, 3)
        outputs = _forward_landing_fixed_dir(
            flat,
            *repeated_inputs,
            prm,
            surface_z,
            net_x,
            h,
            n_steps,
        )
        shaped = []
        for output in outputs:
            shaped.append(
                output.reshape(
                    candidate_count,
                    N,
                    *output.shape[1:],
                )
            )
        return tuple(shaped)

    seed5 = _seed(p_strike, v_ball, target_xy, prm, surface_z, delta_t_flight)
    if s0 is None:
        n0, _, _ = _face_from_angles(seed5[:, 0], seed5[:, 1])
        s0 = seed5[:, 2] / torch.sum(n0 * d_hat_w, dim=-1).clamp(min=0.05)
    q = torch.stack([seed5[:, 0], seed5[:, 1], s0.clamp(min=speed_min, max=speed_max)], dim=-1)

    def residual(land_xy_, q_):
        return torch.cat([land_xy_ - target_xy, w_speed * q_[:, 2:3]], dim=-1)      # (N,3)

    land_xy, valid, v_r, n, net_z = fwd(q)
    BIG = torch.tensor(1e6, device=dev, dtype=dt)
    r = residual(land_xy, q)
    cost = torch.where(valid, torch.sum(r * r, dim=-1), BIG)
    hstep = torch.tensor([3.5e-3, 3.5e-3, 0.02], device=dev, dtype=dt)
    lam = torch.full((N,), 1e-3, device=dev, dtype=dt)
    diagnostic_solve_ok = (
        torch.ones(N, dtype=torch.bool, device=dev)
        if diagnostic_fixed_try_lm
        else None
    )

    for _ in range(n_iters):
        if diagnostic_fixed_try_lm:
            # The three finite-difference columns are independent.  Stack them on a candidate
            # axis so the identical per-row forward model pays one launch group, not three.
            qj = q.unsqueeze(0).repeat(3, 1, 1)
            for j in range(3):
                qj[j, :, j] += hstep[j]
            lx_j, valid_j, _, _, _ = fwd_diagnostic_candidates(qj)
            columns = (
                torch.cat(
                    (
                        lx_j - target_xy.unsqueeze(0),
                        w_speed * qj[:, :, 2:3],
                    ),
                    dim=-1,
                )
                - r.unsqueeze(0)
            ) / hstep.view(3, 1, 1)
            columns = torch.where(
                valid_j.unsqueeze(-1),
                columns,
                torch.zeros_like(columns),
            )
            J = columns.permute(1, 2, 0).contiguous()
        else:
            J = torch.zeros(N, 3, 3, device=dev, dtype=dt)
            for j in range(3):
                qj = q.clone()
                qj[:, j] += hstep[j]
                lx_j, valid_j, _, _, _ = fwd(qj)
                col = (residual(lx_j, qj) - r) / hstep[j]
                J[:, :, j] = torch.where(
                    valid_j.unsqueeze(-1), col, torch.zeros_like(col)
                )
        JtJ = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)
        damp = torch.diag_embed(torch.diagonal(JtJ, dim1=1, dim2=2)) \
            + 1e-9 * torch.eye(3, device=dev, dtype=dt)
        accepted = torch.zeros(N, dtype=torch.bool, device=dev)
        if not diagnostic_fixed_try_lm:
            for _try in range(4):
                A = JtJ + lam.view(N, 1, 1) * damp
                try:
                    dq = torch.linalg.solve(A, -g.unsqueeze(-1)).squeeze(-1)
                except Exception:
                    dq = torch.linalg.lstsq(A, -g.unsqueeze(-1)).solution.squeeze(-1)
                q_new = q + torch.where(accepted.unsqueeze(-1), torch.zeros_like(dq), dq)
                # search box only: the ACCEPTED command is rejected outright if it needs more speed,
                # so this never produces a silently clamped output.
                q_new[:, 2] = q_new[:, 2].clamp(min=speed_min, max=speed_max)
                lx_new, valid_new, _, _, _ = fwd(q_new)
                r_new = residual(lx_new, q_new)
                cost_new = torch.where(valid_new, torch.sum(r_new * r_new, dim=-1), BIG)
                better = (cost_new < cost) & (~accepted)
                q = torch.where(better.unsqueeze(-1), q_new, q)
                r = torch.where(better.unsqueeze(-1), r_new, r)
                cost = torch.where(better, cost_new, cost)
                lam = torch.where(better, (lam * 0.3).clamp(min=1e-8), lam)
                lam = torch.where(~better & ~accepted, lam * 10.0, lam)
                accepted = accepted | better
                if bool(accepted.all()):
                    break
        else:
            # A row that rejects one damping try leaves q/r/cost unchanged and multiplies lambda
            # by exactly 10 before the next try.  Therefore all four candidates can be evaluated
            # together and the first improving one selected without changing the serial
            # first-better contract.  Build the lambda ladder through the same successive
            # multiplications (rather than ``lam * 10**k``) to preserve float rounding.
            lam_0 = lam
            lam_1 = lam_0 * 10.0
            lam_2 = lam_1 * 10.0
            lam_3 = lam_2 * 10.0
            lam_candidates = torch.stack(
                (lam_0, lam_1, lam_2, lam_3),
                dim=0,
            )
            A = (
                JtJ.unsqueeze(0)
                + lam_candidates.view(4, N, 1, 1)
                * damp.unsqueeze(0)
            )
            rhs = (
                -g.unsqueeze(0)
                .expand(4, N, 3)
                .reshape(4 * N, 3, 1)
            )
            dq_col, info = torch.linalg.solve_ex(
                A.reshape(4 * N, 3, 3),
                rhs,
                check_errors=False,
            )
            dq = dq_col.reshape(4, N, 3)
            info = info.reshape(4, N)
            q_new = q.unsqueeze(0) + dq
            # Search box only: the ACCEPTED command is rejected outright if it needs more speed,
            # so this never produces a silently clamped output.
            q_new[:, :, 2] = q_new[:, :, 2].clamp(
                min=speed_min.unsqueeze(0),
                max=speed_max.unsqueeze(0),
            )
            (
                lx_new,
                valid_new,
                v_r_new,
                n_new,
                net_z_new,
            ) = fwd_diagnostic_candidates(q_new)
            r_new = torch.cat(
                (
                    lx_new - target_xy.unsqueeze(0),
                    w_speed * q_new[:, :, 2:3],
                ),
                dim=-1,
            )
            cost_new = torch.where(
                valid_new,
                torch.sum(r_new * r_new, dim=-1),
                BIG,
            )
            better = cost_new < cost.unsqueeze(0)
            any_better = better.any(dim=0)
            first_better = torch.argmax(
                better.to(dtype=torch.long),
                dim=0,
            )
            row = torch.arange(N, device=dev)
            selected_q = q_new[first_better, row]
            selected_r = r_new[first_better, row]
            selected_cost = cost_new[first_better, row]
            selected_lam = lam_candidates[first_better, row]
            selected_land_xy = lx_new[first_better, row]
            selected_valid = valid_new[first_better, row]
            selected_v_r = v_r_new[first_better, row]
            selected_n = n_new[first_better, row]
            selected_net_z = net_z_new[first_better, row]

            # The serial loop only charges solve debt while a row remains active: candidates
            # after its first improvement are intentionally irrelevant.
            active_through = torch.where(
                any_better,
                first_better,
                torch.full_like(first_better, 3),
            )
            active_candidates = (
                torch.arange(4, device=dev).unsqueeze(1)
                <= active_through.unsqueeze(0)
            )
            solve_ok = (info == 0) & torch.isfinite(dq).all(dim=-1)
            diagnostic_solve_ok = diagnostic_solve_ok & (
                (~active_candidates) | solve_ok
            ).all(dim=0)

            q = torch.where(any_better.unsqueeze(-1), selected_q, q)
            r = torch.where(any_better.unsqueeze(-1), selected_r, r)
            cost = torch.where(any_better, selected_cost, cost)
            # Carry the exact forward outputs belonging to q.  They are the same tensors the
            # old final replay would recompute, so the diagnostic path need not pay one more
            # identical fixed-action rollout after the final LM iteration.
            land_xy = torch.where(
                any_better.unsqueeze(-1),
                selected_land_xy,
                land_xy,
            )
            valid = torch.where(any_better, selected_valid, valid)
            v_r = torch.where(
                any_better.unsqueeze(-1),
                selected_v_r,
                v_r,
            )
            n = torch.where(
                any_better.unsqueeze(-1),
                selected_n,
                n,
            )
            net_z = torch.where(any_better, selected_net_z, net_z)
            # Success applies ``* 0.3`` to the lambda of the first improving try.  Four
            # failures apply the fourth successive ``* 10`` exactly as the serial loop.
            failed_lam = lam_3 * 10.0
            accepted_lam = (selected_lam * 0.3).clamp(min=1e-8)
            lam = torch.where(any_better, accepted_lam, failed_lam)
            accepted = any_better

    if diagnostic_fixed_try_lm:
        torch._assert_async(
            diagnostic_solve_ok.all(),
            "diagnostic fixed-try LM solve produced non-finite output or nonzero solve_ex info",
        )

    if not diagnostic_fixed_try_lm:
        land_xy, valid, v_r, n, net_z = fwd(q)
    resid = torch.linalg.norm(land_xy - target_xy, dim=-1)
    s = q[:, 2]
    net_top = float(surface_z) + float(net_height)
    net_ok = torch.isfinite(net_z) & (net_z > net_top + ball_radius + net_margin_m)
    face_ok = n[:, 0] > 1e-6
    speed_ok = (s >= speed_min - 1e-9) & (s <= speed_max + 1e-9)
    ok = valid & (resid < tol_m) & speed_ok & net_ok & face_ok

    reason = torch.zeros(N, dtype=torch.long, device=dev)          # 0 = no_landing
    reason = torch.where(valid, torch.ones_like(reason), reason)   # 1 = resid_gt_tol
    reason = torch.where(valid & (s >= speed_max - 1e-6), torch.full_like(reason, 2), reason)
    reason = torch.where(valid & (s <= speed_min + 1e-6), torch.full_like(reason, 3), reason)
    reason = torch.where(valid & (resid < tol_m) & ~net_ok, torch.full_like(reason, 5), reason)
    reason = torch.where(valid & (resid < tol_m) & net_ok & ~face_ok,
                         torch.full_like(reason, 6), reason)
    reason = torch.where(ok, torch.full_like(reason, -1), reason)

    return {
        "p_contact": p_strike,                     # C2: the ball's own point, verbatim
        "v_r": v_r, "n": n, "speed": s,
        "landing_xy": land_xy, "resid_m": resid, "net_z": net_z,
        "clears_net": net_ok, "ok": ok, "reason": reason, "q": q,
    }


def dir_deviation_deg(v_r: torch.Tensor, d_hat_w: torch.Tensor) -> torch.Tensor:
    """C1 check. Zero by construction here (``v_r`` IS ``s * d_hat_w``); reported so a caller can
    assert it rather than trust it. arccos is ill-conditioned at 1, so expect ~1e-5 deg of noise."""
    v = v_r / (torch.linalg.norm(v_r, dim=-1, keepdim=True) + _EPS)
    return torch.rad2deg(torch.arccos(torch.sum(v * d_hat_w, dim=-1).clamp(-1.0, 1.0)))


def radians(deg: float) -> float:
    return math.radians(float(deg))
