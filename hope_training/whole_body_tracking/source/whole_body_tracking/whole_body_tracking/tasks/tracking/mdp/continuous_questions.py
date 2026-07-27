"""CONTINUOUS question generation — draw a ball, solve the target, batched.

人话(owner 裁定):离散题库是**考试**用的,训练必须连续采样。所以训练侧不再回放一张定长的题
表,而是从连续分布里画球(速度/旋转/落点/触球点都是区间,按 clip 各有自己的框),再用成熟的逆解
当场把"该把拍子摆成什么样"算出来。解不出来的球是一个**响亮且被计数**的事件——重新画一颗,不许
悄悄拿别的目标顶上。

Two paths, ONE physics:

* CONTINUOUS (training) — this module. Ball drawn from a continuous parameterisation; answer from
  ``solve_strike_specs`` / ``solve_strike_specs_fixed_dir`` on the trainer's own venue model.
* DISCRETE (exam, reproducibility) — the pre-solved npz bank, unchanged. It stays.

They are the same physics because they call the SAME solver on the SAME ``VirtualBallParams``.
``tests/test_continuous_vs_bank_parity.py`` pushes one BANK ROW's ball through the continuous
solver and asserts both answers land within the solver's own tolerance of each other and of the
aim, clear the net, and carry the SAME FACE SIGN — it deliberately does NOT assert the demanded
state agrees bitwise, because it does not: the 5-DoF solution manifold under a shared speed
regulariser is not pointwise unique (a face rotated a few degrees with a compensating velocity
lands in the same place). ``parity_report`` below is that assertion body; the boot gate in
``hope_commands`` imports the SAME function, so the property is checked on this machine at this
commit, not only in pytest.

FAILURE POLICY, structural not disciplinary. A drawn ball with no legal answer is never
substituted. ``generate`` returns an ``ok`` mask plus a per-reason histogram; the caller redraws
those rows (bounded rounds) and, if any row still has no answer after the budget, that is a loud
counted event — ``QuestionDrawResult.exhausted``. The unsolved rows come back **NaN-filled**
(``resid_m`` is kept for diagnosis) so that no caller, now or later, can install one by forgetting
to mask on ``ok``.

FRAME. Everything here is ENV-LOCAL — the same convention as ``coarse_landing`` / the trainer's own
scorer. ``generate`` takes no ``origins`` and returns no world coordinates ON PURPOSE: the caller
adds the env origin at install time, exactly like the bank seam does. (The old signature took
``origins``, shifted the contact point by it but not aim_x, and compared against env-local
surface/net scalars; both callers happened to pass zeros so it never showed. At origin=(3,0,0) it
converged confidently on a target 3 m short with the net behind the robot.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .strike_spec_torch import solve_strike_specs
from .stroke_adapt_torch import REASONS, direction_world, solve_strike_specs_fixed_dir
from .virtual_ball import VirtualBallParams, coarse_landing, predict_paddle_contact

_EPS = 1e-9

#: Reject-reason code indices into ``stroke_adapt_torch.REASONS`` used by the free-solve path.
_R_NO_LANDING = 0
_R_RESID = 1
_R_SPEED_OVER = 2
_R_NET = 5


@dataclass
class ContinuousQuestionCfg:
    """The continuous parameterisation. Every field is a RANGE, never a list of cases."""

    # incoming ball, env-local m/s; per-clip rows override the shared box when given
    vel_range: tuple = ((-4.5, -2.0), (-0.6, 0.6), (-1.0, 0.5))
    vel_range_per_clip: tuple | None = None
    spin_abs_max: float = 50.0
    spin_abs_max_per_clip: tuple | None = None
    # contact point (== the ball's arrival point), env-origin-relative m; per-clip rows required
    # for a multi-clip run (each stroke reaches a different region).
    pos_range_per_clip: tuple | None = None
    pos_range: tuple = ((0.50, 0.62), (-0.45, 0.45), (0.80, 1.20))
    # aim on the opponent half, env-local m (net at 1.87, far edge 3.24, |y| <= 0.7625)
    aim_x_range: tuple = (2.20, 2.90)
    aim_y_range: tuple = (-0.55, 0.55)
    # solver
    tol_m: float = 0.02
    n_iters: int = 12
    speed_budget: float = 3.4          # gate_speed_max - margin (pp_policy.hpp:234)
    max_redraw_rounds: int = 3
    fixed_direction: bool = False      # True -> the stroke adapter (direction = stroke identity)


@dataclass
class QuestionDrawResult:
    """One batch of solved questions plus the accounting that makes failures visible.

    Rows with ``ok=False`` are NaN in every geometric field — an unsolved ball is not installable
    by construction, not merely by convention. ``resid_m`` survives for diagnosis.
    """

    p_contact: torch.Tensor            # (N,3) the BALL's own arrival point, ENV-LOCAL
    v_racket: torch.Tensor             # (N,3)
    n_racket: torch.Tensor             # (N,3) unit, explicit face
    v_ball_in: torch.Tensor            # (N,3)
    w_ball_in: torch.Tensor            # (N,3)
    aim_xy: torch.Tensor               # (N,2) ENV-LOCAL
    ok: torch.Tensor                   # (N,) bool
    resid_m: torch.Tensor              # (N,)
    #: (N,3) the LAST ATTEMPTED incoming ball for EVERY row, solved or not. Accounting only — it
    #: is never an installable target (no answer travels with it), and it exists because a
    #: per-regime accept ledger cannot see which regime disappeared if the failed draws are NaN.
    attempted_v_ball_in: torch.Tensor = None
    rounds_used: int = 0
    exhausted: int = 0                 # rows still unsolved after max_redraw_rounds
    reason_counts: dict = field(default_factory=dict)


def _rows(table, clip_ids, device, dtype):
    t = torch.as_tensor(table, dtype=dtype, device=device)
    return t[clip_ids]


def _uniform_box(box_rows, gen, device, dtype):
    """box_rows (N,3,2) -> (N,3) uniform draw. Continuous by construction: no case list."""
    u = torch.rand(box_rows.shape[0], 3, device=device, dtype=dtype, generator=gen)
    return box_rows[:, :, 0] + (box_rows[:, :, 1] - box_rows[:, :, 0]) * u


@torch.no_grad()
def generate(
    clip_ids: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    cfg: ContinuousQuestionCfg,
    ref_normal: torch.Tensor | None = None,
    net_top_z: float | None = None,
    protos=None,
    base_quat=None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    h: float = 0.01,
    n_steps: int = 100,
) -> QuestionDrawResult:
    """Draw N balls continuously and solve each one's demanded racket state. ALL ENV-LOCAL.

    ``clip_ids`` (N,) selects the per-clip regime and supplies device. ``ref_normal`` (N,3) is the
    face the answer's sign is matched to — pass the RUNTIME raw +Y clip face; without it the
    solver's arbitrary "+x opponent-facing" seed convention wins and, on a backhand clip whose
    clip normal has x < 0, EVERY row comes back with the face opposite the clip face (measured:
    763/763 and 746/746 rows on the two shipped banks). That is the M3c 单翻病 by construction.
    ``net_top_z`` (env-local z of the net top, ball radius included) turns on the net-clearance
    rejection the bank generator has always applied; None = no net test (byte-identical to the
    historical free-solve behaviour). ``h``/``n_steps`` are the rollout parameters — pass the
    SCORER's own (``vb_rollout_h`` / ``vb_rollout_steps``) so the solved answer and the graded
    answer cannot drift apart by coincidence; the defaults only match today by luck.

    COST: one call is ``n_iters * (5 probes + up to 4 trial steps)`` batched rollouts of
    ``coarse_landing`` — kernel-launch bound, so the wall cost is per-CALL not per-env (measured:
    48 rows cost 63% of what 8192 rows cost). Measure with ``scripts/bench_continuous_questions.py``
    before raising ``n_iters``; the fallback when the budget is blown is to make the call RARER
    (buffer more rows per call), then to lower ``n_iters`` (accuracy degrades visibly through
    ``resid_m``), NEVER to substitute an unsolved target.
    """
    device = clip_ids.device
    N = int(clip_ids.shape[0])
    gen = generator

    pos_box = (_rows(cfg.pos_range_per_clip, clip_ids, device, dtype)
               if cfg.pos_range_per_clip is not None
               else torch.as_tensor(cfg.pos_range, dtype=dtype, device=device)[None].expand(N, 3, 2))
    vel_box = (_rows(cfg.vel_range_per_clip, clip_ids, device, dtype)
               if cfg.vel_range_per_clip is not None
               else torch.as_tensor(cfg.vel_range, dtype=dtype, device=device)[None].expand(N, 3, 2))
    spin_max = (_rows(cfg.spin_abs_max_per_clip, clip_ids, device, dtype)
                if cfg.spin_abs_max_per_clip is not None
                else torch.full((N,), float(cfg.spin_abs_max), device=device, dtype=dtype))
    if ref_normal is not None:
        ref_normal = ref_normal.to(device=device, dtype=dtype)
        if ref_normal.shape != (N, 3):
            raise ValueError(f"ref_normal must be (N,3) matching clip_ids, got {tuple(ref_normal.shape)}")

    nan = float("nan")
    p = torch.full((N, 3), nan, device=device, dtype=dtype)
    v_in = torch.full((N, 3), nan, device=device, dtype=dtype)
    w_in = torch.full((N, 3), nan, device=device, dtype=dtype)
    aim = torch.full((N, 2), nan, device=device, dtype=dtype)
    v_r = torch.full((N, 3), nan, device=device, dtype=dtype)
    n_r = torch.full((N, 3), nan, device=device, dtype=dtype)
    resid = torch.full((N,), float("inf"), device=device, dtype=dtype)
    v_att = torch.zeros(N, 3, device=device, dtype=dtype)
    ok = torch.zeros(N, dtype=torch.bool, device=device)
    counts: dict = {}
    rounds_used = 0

    todo = torch.arange(N, device=device)
    for _round in range(1, int(cfg.max_redraw_rounds) + 1):
        if todo.numel() == 0:
            break
        rounds_used = _round                      # assigned AFTER the break: no off-by-one
        m = todo.numel()
        p_m = _uniform_box(pos_box[todo], gen, device, dtype)
        v_m = _uniform_box(vel_box[todo], gen, device, dtype)
        s_m = spin_max[todo].unsqueeze(-1)
        w_m = (torch.rand(m, 3, device=device, dtype=dtype, generator=gen) * 2.0 - 1.0) * s_m
        ax = torch.rand(m, device=device, dtype=dtype, generator=gen)
        ay = torch.rand(m, device=device, dtype=dtype, generator=gen)
        aim_m = torch.stack([
            cfg.aim_x_range[0] + (cfg.aim_x_range[1] - cfg.aim_x_range[0]) * ax,
            cfg.aim_y_range[0] + (cfg.aim_y_range[1] - cfg.aim_y_range[0]) * ay,
        ], dim=-1)

        if cfg.fixed_direction:
            if protos is None or base_quat is None:
                raise ValueError(
                    "ContinuousQuestionCfg.fixed_direction=True needs the stroke prototypes and "
                    "the base orientation: the direction it holds fixed IS the stroke's identity"
                )
            from .stroke_adapt_torch import base_yaw_of
            yaw = base_yaw_of(base_quat[todo])
            d_all = direction_world(protos.v_hat_b.to(device, dtype), yaw)      # (m,K,3)
            d_m = d_all[torch.arange(m, device=device), clip_ids[todo]]
            out = solve_strike_specs_fixed_dir(
                p_m, v_m, w_m, aim_m, d_m,
                protos.speed_min.to(device, dtype)[clip_ids[todo]],
                protos.speed_max.to(device, dtype)[clip_ids[todo]],
                prm, surface_z=surface_z, net_x=net_x,
                n_iters=int(cfg.n_iters), tol_m=float(cfg.tol_m),
            )
            good = out["ok"]
            reasons = out["reason"]
        else:
            out = solve_strike_specs(
                p_m, v_m, w_m, aim_m, prm, surface_z=surface_z, net_x=net_x,
                ref_normal=(None if ref_normal is None else ref_normal[todo]),
                speed_budget=float(cfg.speed_budget), n_iters=int(cfg.n_iters),
                tol_m=float(cfg.tol_m), h=float(h), n_steps=int(n_steps),
            )
            good = out["ok"]
            # NET CLEARANCE. solve_strike_specs deliberately leaves the net out of ``ok`` (its
            # historical callers never tested it); the bank generator rejects and COUNTS it, so
            # the continuous path must too or it trains on returns that clip the net.
            if net_top_z is not None:
                net_ok = out["net_valid"] & (out["net_z"] > float(net_top_z))
                good = good & net_ok
            else:
                net_ok = torch.ones_like(good)
            # REAL reason codes. This used to hard-code code 1 for every failure, so a no-landing
            # and a speed-cap rejection both surfaced as "resid_gt_tol" and every histogram anyone
            # ever printed was fiction. Same expressions/order as solve_strike_specs_fixed_dir.
            valid = out["land_valid"]
            resid_ok = out["resid_m"] < float(cfg.tol_m)
            speed_ok = out["speed"] <= float(cfg.speed_budget) + 1e-9
            reasons = torch.full_like(good, _R_NO_LANDING, dtype=torch.long)
            reasons = torch.where(valid, torch.full_like(reasons, _R_RESID), reasons)
            reasons = torch.where(valid & resid_ok & ~speed_ok,
                                  torch.full_like(reasons, _R_SPEED_OVER), reasons)
            reasons = torch.where(valid & resid_ok & speed_ok & ~net_ok,
                                  torch.full_like(reasons, _R_NET), reasons)
            reasons = torch.where(good, torch.full_like(reasons, -1), reasons)

        v_att[todo] = v_m
        sel = todo[good]
        p[sel], v_in[sel], w_in[sel], aim[sel] = p_m[good], v_m[good], w_m[good], aim_m[good]
        v_r[sel], n_r[sel] = out["v_r"][good], out["n"][good]
        resid[sel] = out["resid_m"][good]
        ok[sel] = True
        bad = ~good
        if bool(bad.any()):
            # The unsolved rows keep their (finite) residual for diagnosis and STAY NaN everywhere
            # else — see the class docstring: not installable, structurally.
            resid[todo[bad]] = torch.nan_to_num(out["resid_m"][bad], nan=float("inf"))
            for code in reasons[bad].tolist():
                name = REASONS[code] if 0 <= code < len(REASONS) else "unsolved"
                counts[name] = counts.get(name, 0) + 1
        todo = todo[bad]

    return QuestionDrawResult(
        p_contact=p, v_racket=v_r, n_racket=n_r, v_ball_in=v_in, w_ball_in=w_in, aim_xy=aim,
        ok=ok, resid_m=resid, attempted_v_ball_in=v_att, rounds_used=rounds_used,
        exhausted=int((~ok).sum()),
        reason_counts=counts,
    )


@torch.no_grad()
def parity_report(
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    bank_vel: torch.Tensor,
    bank_normal: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    speed_budget: float = 4.0,
    tol_m: float = 0.005,
    n_iters: int = 12,
    h: float = 0.01,
    n_steps: int = 100,
    land_tol_m: float = 0.02,
    aim_tol_m: float = 0.006,
    speed_tol_mps: float = 0.10,
    min_coverage: float = 0.90,
) -> dict:
    """Push a BANK ROW's ball through the continuous solver and compare PHYSICAL CONSEQUENCES.

    人话:"两条路同物理"能断言的形式是**后果**(落点/过网/拍面在哪一侧/速度模长),不是 demanded
    state 逐位相等——5 自由度解流形在共享速度正则下不是逐点唯一的,拍面转 4 度、速度补一点,落点
    一样。写一句逐位相等的断言等于写一句在某个容差下碰巧能过的假话。

    Returns ``{"failures": [str, ...], "stats": {...}}``. Callers assert ``failures == []``. Used by
    BOTH ``tests/test_continuous_vs_bank_parity.py`` and the boot anchor gate, so the two can never
    drift apart.
    """
    dev, dt = p_contact.device, p_contact.dtype
    out = solve_strike_specs(
        p_contact, v_ball_in, w_ball_in, aim_xy, prm, surface_z=surface_z, net_x=net_x,
        ref_normal=ref_normal, speed_budget=float(speed_budget), n_iters=int(n_iters),
        tol_m=float(tol_m), h=float(h), n_steps=int(n_steps),
    )
    ok = out["ok"]
    n = int(p_contact.shape[0])
    fails: list = []
    cov = float(ok.float().mean()) if n else 0.0
    if cov < float(min_coverage):
        fails.append(
            f"A. coverage: the torch solver answered {int(ok.sum())}/{n} = {cov:.4f} of the rows "
            f"the numpy planner answered (< {min_coverage})"
        )
    stats = {"n": n, "coverage": cov}
    if int(ok.sum()) == 0:
        return {"failures": fails, "stats": stats}

    m = ok
    n_c, n_b = out["n"][m], bank_normal[m]
    ref = ref_normal[m]
    d_c = torch.sum(n_c * ref, dim=-1)
    d_b = torch.sum(n_b * ref, dim=-1)
    stats["min_dot_continuous_vs_clipface"] = float(d_c.min())
    stats["min_dot_bank_vs_clipface"] = float(d_b.min())
    if float(d_c.min()) <= 0.0:
        fails.append(
            f"B. face sign: {int((d_c <= 0).sum())}/{int(m.sum())} continuous answers point "
            f"OPPOSITE the clip face (min dot {float(d_c.min()):.4f}) — ref_normal sign alignment "
            f"is not reaching solve_strike_specs; this IS the M3c 单翻病"
        )
    if float(d_b.min()) <= 0.0:
        fails.append(
            f"B. face sign: {int((d_b <= 0).sum())}/{int(m.sum())} BANK answers point opposite the "
            f"clip face (min dot {float(d_b.min()):.4f}) — the bank itself is in the flipped "
            f"convention"
        )

    def _land(v_r, n_face):
        v_plus, w_plus = predict_paddle_contact(v_ball_in[m], v_r, n_face, w_ball_in[m], prm)
        return coarse_landing(p_contact[m], v_plus, w_plus, prm, surface_z=surface_z,
                              net_x=net_x, h=float(h), n_steps=int(n_steps))

    l_c, l_b = _land(out["v_r"][m], n_c), _land(bank_vel[m], n_b)
    for tag, land in (("continuous", l_c), ("bank", l_b)):
        if not bool(land["land_valid"].all()):
            fails.append(f"C. {tag}: {int((~land['land_valid']).sum())} answers never land")
        # D. net clearance — the acceptance the bank generator has always applied.
        clears = land["net_valid"] & (land["net_z"] > float(net_top_z))
        if not bool(clears.all()):
            fails.append(f"D. {tag}: {int((~clears).sum())} answers do not clear the net")
    both = l_c["land_valid"] & l_b["land_valid"]
    if bool(both.any()):
        d_land = torch.linalg.norm(l_c["land_xy"][both] - l_b["land_xy"][both], dim=-1)
        stats["land_delta_p50_m"] = float(d_land.median())
        stats["land_delta_max_m"] = float(d_land.max())
        if float(d_land.max()) > float(land_tol_m):
            fails.append(
                f"C. the two paths land {float(d_land.max()):.4f} m apart (> {land_tol_m}) for the "
                f"SAME ball — they are not the same physics"
            )
        for tag, land in (("continuous", l_c), ("bank", l_b)):
            d_aim = torch.linalg.norm(land["land_xy"][both] - aim_xy[m][both], dim=-1)
            stats[f"aim_err_max_m_{tag}"] = float(d_aim.max())
            if float(d_aim.max()) > float(aim_tol_m):
                fails.append(
                    f"C. {tag}: worst landing is {float(d_aim.max()):.4f} m from the aim point "
                    f"(> {aim_tol_m})"
                )
    # F. speed MAGNITUDE only. |v_r| is a physical consequence; the direction is not unique on the
    # solution manifold, so a bitwise v_r assertion would be a threshold-tuned lie.
    ds = (torch.linalg.norm(out["v_r"][m], dim=-1) - torch.linalg.norm(bank_vel[m], dim=-1)).abs()
    stats["speed_delta_max_mps"] = float(ds.max())
    if float(ds.max()) > float(speed_tol_mps):
        fails.append(f"F. demanded speed magnitudes differ by {float(ds.max()):.4f} m/s "
                     f"(> {speed_tol_mps})")
    ang = torch.rad2deg(torch.arccos(torch.sum(n_c * n_b, dim=-1).clamp(-1.0, 1.0)))
    stats["face_angle_max_deg"] = float(ang.max())      # reported, deliberately NOT asserted
    del dev, dt
    return {"failures": fails, "stats": stats}


@torch.no_grad()
def face_sign_negative_control(
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    speed_budget: float = 4.0,
    tol_m: float = 0.005,
    n_iters: int = 12,
) -> tuple[int, int]:
    """WITHOUT ``ref_normal``: how many solved rows come back on the WRONG side of the clip face.

    人话:这是"符号对齐还在不在"的反向证据。谁把 ref_normal 删掉,这个数会从"全翻"变成"不翻",
    测试当场红——不用起 env、不用等 3000 迭代。返回 (翻转行数, 解出行数)。
    """
    out = solve_strike_specs(
        p_contact, v_ball_in, w_ball_in, aim_xy, prm, surface_z=surface_z, net_x=net_x,
        ref_normal=None, speed_budget=float(speed_budget), n_iters=int(n_iters), tol_m=float(tol_m),
    )
    ok = out["ok"]
    if int(ok.sum()) == 0:
        return 0, 0
    d = torch.sum(out["n"][ok] * ref_normal[ok], dim=-1)
    return int((d < 0).sum()), int(ok.sum())
