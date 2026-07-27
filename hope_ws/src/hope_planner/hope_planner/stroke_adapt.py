"""Motion ADAPTER — fit the chosen stroke prototype to THIS ball.

人话:动作选好之后,还要把它"套"到这颗球上。四条硬约束(owner 原话):

1. 挥拍**方向**基本不变——方向就是这个动作的身份(拉就是拉、挡就是挡);
2. 触球**位置是球的**,不是动作模板的;
3. 拍**速大小可变,但有上限**;
4. **拍面朝向自由**——设成"能把球打回去"的那个角度。

Two things this module refuses to do, both because today's evidence says they break:

* **The face is emitted EXPLICITLY, never implied by the velocity.**  A published system omits an
  explicit face command and its learned solution degenerates into lofting the ball upward instead
  of returning it flat; our landing reward scores only WHERE the ball lands, not how it got there,
  so nothing downstream would catch it either.  ``StrokeCommand.n_racket`` is always solved and
  always published.
* **Faster is not better.**  Two of the five strokes are already at their optimum and get WORSE
  when sped up (one drops 0.97 -> 0.12 at 1.5x).  The speed is therefore a SOLVED quantity
  derived from this ball — a landing residual under 5 mm plus a least-effort regulariser — with
  the prototype's cap as a bound, never a target.

Out-of-range behaviour is uniform: return ``ok=False`` with a named reason and nothing else.
Never clamp the speed into range, never widen the cone, never move the contact off the ball,
never relax the landing tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .stroke_prototypes import StrokePrototype, StrokePrototypeSet, direction_world
from .stroke_select import (
    SelectorCfg,
    StrokeChoice,
    base_yaw_angle,
    closing_speed_demand,
    select_stroke,
)

#: Exact reason strings — they become metric names, so they are part of the interface.
REASON_NO_LANDING = "no_landing"
REASON_RESID = "resid_gt_tol"
REASON_SPEED_OVER = "speed_over_cap"
REASON_SPEED_UNDER = "speed_under_min"
REASON_DIR_CONE = "dir_cone_exceeded"
REASON_NET = "net_not_cleared"
REASON_FACE_BACKWARD = "face_not_opponent_facing"
REASON_FACE_ENVELOPE = "face_out_of_envelope"
ALL_REASONS = (
    REASON_NO_LANDING, REASON_RESID, REASON_SPEED_OVER, REASON_SPEED_UNDER,
    REASON_DIR_CONE, REASON_NET, REASON_FACE_BACKWARD, REASON_FACE_ENVELOPE,
)


@dataclass
class AdapterCfg:
    """The adapter's own constants.  ``surface_z`` declares the caller's z frame (see
    :class:`~hope_planner.stroke_select.SelectorCfg`); nothing else here is frame-dependent."""

    tol_m: float = 0.005              # StrikeSpecPlanner.TOL_M
    w_dir: float = 0.5
    max_iter: int = 30
    stage2_enable: bool = True
    net_margin_m: float = 0.02
    net_height_m: float = 0.1525      # TableParams.net_height, above the table SURFACE
    face_envelope_min_dot: float = -1.0   # -1.0 = no envelope pinned (the check is a no-op)
    face_envelope_ref_b: Optional[Sequence[float]] = None   # reference face in B_yaw
    use_fast_solver: bool = False


@dataclass(frozen=True)
class StrokeCommand:
    """What the robot is told to do.  All vectors in the CALLER's world frame."""

    p_contact: np.ndarray             # (3,) == the ball's own predicted contact position
    v_racket: np.ndarray              # (3,) = speed * d_hat_w
    n_racket: np.ndarray              # (3,) unit, PHYSICAL striking face, opponent-facing
    speed: float
    t_strike_s: float
    landing_xy: np.ndarray            # (2,)
    resid_m: float
    net_z_margin_m: float
    clears_net: bool
    dir_deviation_deg: float
    stage: int                        # 1 = direction pinned exactly, 2 = inside the cone
    ok: bool
    reason: str = ""
    motion_id: str = ""
    clip_index: int = -1
    v_plus: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    omega_plus: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    iterations: int = 0


def _fail(p_contact, reason, proto, stage, sol=None) -> StrokeCommand:
    """A refusal that still reports the four constrained quantities honestly.

    ``p_contact`` is the ball's position even on failure (constraint C2 is structural, not a
    solver outcome) — the regression test asserts exactly that.
    """
    return StrokeCommand(
        p_contact=np.asarray(p_contact, dtype=float).copy(),
        v_racket=np.asarray(sol["v_r"], dtype=float) if sol else np.full(3, np.nan),
        n_racket=np.asarray(sol["n"], dtype=float) if sol else np.full(3, np.nan),
        speed=float(sol["speed"]) if sol else float("nan"),
        t_strike_s=float("nan"),
        landing_xy=np.asarray(sol["landing_xy"], dtype=float) if sol else np.full(2, np.nan),
        resid_m=float(sol["resid_m"]) if sol else float("nan"),
        net_z_margin_m=float(sol["net_z_margin_m"]) if sol else float("nan"),
        clears_net=bool(sol["clears_net"]) if sol else False,
        dir_deviation_deg=float(sol["dir_deviation_deg"]) if sol else float("nan"),
        stage=stage,
        ok=False,
        reason=reason,
        motion_id=proto.motion_id,
        clip_index=proto.clip_index,
        iterations=int(sol["iterations"]) if sol else 0,
    )


def _accept(
    sol: dict,
    proto: StrokePrototype,
    d_hat_w: np.ndarray,
    p_ball: np.ndarray,
    cfg: AdapterCfg,
    sel: SelectorCfg,
    stage: int,
) -> Tuple[bool, str]:
    """Acceptance, in the order the reasons should be attributed.

    Includes the NET as an acceptance condition — that is a bug fix, not a nicety.
    ``RacketTargetPlanner.plan`` (racket_target_planner.py:253-258) returns ``valid=True``
    unconditionally even when all five of its flight-time retries fail to clear the net, and
    neither ``node.py:1572`` nor the flat wire carries a ``clears_net`` term, so a command that
    cannot clear the net is published as ``valid=1`` today.  Here it cannot be produced at all.
    """
    if not np.isfinite(sol["landing_xy"]).all():
        return False, REASON_NO_LANDING
    s = float(sol["speed"])
    if s > proto.speed_max_mps + 1e-9:
        return False, REASON_SPEED_OVER
    if s < proto.speed_min_mps - 1e-9:
        return False, REASON_SPEED_UNDER
    if sol["resid_m"] >= cfg.tol_m:
        # Attribute a missed landing to the SPEED BOUND when the search ended pressed against
        # one: the LM confines s to [speed_min, speed_max] while searching, so "could not reach
        # the aim, and s is sitting on the cap" means the stroke cannot supply this ball, not
        # that the solver wandered.  The command is refused either way — this only decides which
        # counter it lands in.
        if s >= proto.speed_max_mps - 1e-6:
            return False, REASON_SPEED_OVER
        if s <= proto.speed_min_mps + 1e-6:
            return False, REASON_SPEED_UNDER
        return False, REASON_RESID
    # Stage 1 pins the direction exactly (v_r IS s * d_hat), so the only slack here is arccos's
    # conditioning at 1: a 1e-16 dot-product error reads as ~1e-5 deg.
    tol = 1e-3 if stage == 1 else proto.v_dir_tol_deg + 1e-9
    if sol["dir_deviation_deg"] > tol:
        return False, REASON_DIR_CONE
    net_top = float(sel.surface_z) + float(cfg.net_height_m)
    z_at_net = net_top + float(sol["net_z_margin_m"]) if np.isfinite(sol["net_z_margin_m"]) \
        else float("-inf")
    if not (sol["clears_net"]
            and z_at_net > net_top + float(sel.ball_radius) + float(cfg.net_margin_m) - 1e-12):
        return False, REASON_NET
    n = np.asarray(sol["n"], dtype=float)
    if float(n[0]) <= 1e-6:
        return False, REASON_FACE_BACKWARD
    if cfg.face_envelope_ref_b is not None and cfg.face_envelope_min_dot > -1.0:
        ref = np.asarray(cfg.face_envelope_ref_b, dtype=float).reshape(3)
        ref = ref / (float(np.linalg.norm(ref)) + 1e-12)
        n_a = float(proto.face_sign) * n
        if float(np.dot(n_a, ref)) < cfg.face_envelope_min_dot:
            return False, REASON_FACE_ENVELOPE
    return True, ""


def fit_stroke_to_ball(
    p_ball_w,
    v_ball_w,
    omega_ball_w,
    aim_xy_w,
    proto: StrokePrototype,
    planner,
    base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    t_strike_s: float = float("nan"),
    cfg: Optional[AdapterCfg] = None,
    sel: Optional[SelectorCfg] = None,
) -> StrokeCommand:
    """Fit ``proto`` to this ball.  ``planner`` is a ``StrikeSpecPlanner`` (or the fast subclass).

    THE FOUR CONSTRAINTS AS EQUATIONS
    ---------------------------------
    C1  velocity direction preserved
            ``v_r = s * d_hat_w``,  ``d_hat_w = R_z(yaw_base) . proto.v_hat_b``
        The direction is not a variable at all in stage 1, so the deviation is 0 BY
        CONSTRUCTION; in stage 2 the hard bound is
            ``acos(v_hat_r . d_hat_w) <= proto.v_dir_tol_deg``  (default 10.0 deg)
        which the measured elevation gap between the loops (+44..+45 deg) and the
        blocks/press (-3..-1 deg) proves cannot convert one stroke into another.

    C2  contact position is the BALL's
            ``p_contact = p_ball_w``   — exact, not a variable, and true on failures too.

    C3  speed magnitude variable, with an upper bound
            ``s in [proto.speed_min_mps, proto.speed_max_mps]``,
            ``speed_max = min(nominal * retime_s_max, sqrt(2 a_max L_deep))``
        Both terms are MEASURED off the clip: the retime ceiling (fastest replay before a joint
        runs out of velocity/acceleration) and the runway ceiling (fastest the blade can be going
        at contact given its backswing travel and acceleration envelope).  There is no third
        term.  The deploy runner's ``gate_speed_max`` (pp_policy.hpp:234) is REPORTED as
        ``provenance.deploy_gate`` in the prototype file, never folded into this bound — an
        anonymous ``3.5 - 0.1`` used to sit here and it silently cut fh_loop/full
        (3.465 -> 3.400) for no stated reason (owner's ruling 2026-07-27).
        Hard: out of range is a refusal (``speed_over_cap`` / ``speed_under_min``), never a clamp.

    C4  blade orientation free
            ``n(theta, phi) = (cos th cos ph, cos th sin ph, sin th)``, both angles solved.
        Post-solve: ``|n| = 1``, ``n_x > 1e-6`` (opponent-facing), and — when an envelope is
        configured — ``dot(face_sign * n, ref) >= face_envelope_min_dot``.

    Stage 1 pins the direction exactly.  Only if it fails, and only when ``cfg.stage2_enable``,
    stage 2 re-solves with two extra unknowns rotating ``d_hat_w`` inside the cone.  If that fails
    too the result is ``ok=False`` with the stage-2 reason: nothing is relaxed further.
    """
    cfg = cfg or AdapterCfg()
    sel = sel or SelectorCfg()
    p_ball = np.asarray(p_ball_w, dtype=float).reshape(-1)[:3]
    v_ball = np.asarray(v_ball_w, dtype=float).reshape(-1)[:3]
    omega = np.zeros(3) if omega_ball_w is None else \
        np.asarray(omega_ball_w, dtype=float).reshape(-1)[:3]
    aim = np.asarray(aim_xy_w, dtype=float).reshape(-1)[:2]

    yaw = base_yaw_angle(base_quat_wxyz)
    d_hat_w = direction_world(proto, yaw)
    v_n_req, n0 = closing_speed_demand(p_ball, v_ball, aim, sel)
    s0 = v_n_req / max(float(np.dot(d_hat_w, n0)), 0.05)

    solve = (planner.solve_fast_fixed_direction
             if (cfg.use_fast_solver and hasattr(planner, "solve_fast_fixed_direction"))
             else planner.solve_fixed_direction)

    stages = [0.0]
    if cfg.stage2_enable and proto.v_dir_tol_deg > 0.0:
        stages.append(math.radians(proto.v_dir_tol_deg))

    last = None
    for stage_idx, cone in enumerate(stages, start=1):
        sol = solve(
            p_ball, v_ball, omega, aim, d_hat_w,
            proto.speed_min_mps, proto.speed_max_mps,
            dir_cone_rad=cone, s0=s0, max_iter=cfg.max_iter, tol_m=cfg.tol_m,
            w_dir=cfg.w_dir,
        )
        if sol is None:
            last = (REASON_NO_LANDING, None, stage_idx)
            continue
        ok, reason = _accept(sol, proto, d_hat_w, p_ball, cfg, sel, stage_idx)
        if ok:
            return StrokeCommand(
                p_contact=p_ball.copy(),                     # C2, verbatim
                v_racket=np.asarray(sol["v_r"], dtype=float),
                n_racket=np.asarray(sol["n"], dtype=float),
                speed=float(sol["speed"]),
                t_strike_s=float(t_strike_s),
                landing_xy=np.asarray(sol["landing_xy"], dtype=float),
                resid_m=float(sol["resid_m"]),
                net_z_margin_m=float(sol["net_z_margin_m"]),
                clears_net=bool(sol["clears_net"]),
                dir_deviation_deg=float(sol["dir_deviation_deg"]),
                stage=stage_idx,
                ok=True,
                motion_id=proto.motion_id,
                clip_index=proto.clip_index,
                v_plus=np.asarray(sol["v_plus"], dtype=float),
                omega_plus=np.asarray(sol["omega_plus"], dtype=float),
                iterations=int(sol["iterations"]),
            )
        last = (reason, sol, stage_idx)

    reason, sol, stage_idx = last
    return _fail(p_ball, reason, proto, stage_idx, sol)


def select_and_fit(
    p_ball_w,
    v_ball_w,
    omega_ball_w,
    t_avail_s: float,
    base_pos_w,
    base_quat_wxyz,
    aim_xy_w,
    protos: StrokePrototypeSet,
    planner,
    prev_choice: Optional[StrokeChoice] = None,
    sel: Optional[SelectorCfg] = None,
    cfg: Optional[AdapterCfg] = None,
    t_strike_s: float = float("nan"),
) -> Tuple[StrokeChoice, Optional[StrokeCommand]]:
    """The whole chain: ball -> planner task -> target motion.

    The selector produces a total order; this walks it, calling the adapter, and returns the first
    stroke that actually solves.  ``sel.max_solve_attempts`` bounds the walk (0 = unlimited, which
    is what training uses; the deploy path passes 2 for the latency budget).

    When nothing is admissible, or every admissible candidate fails to solve, the returned choice
    carries ``clip_index = -1`` and a reason.  There is no fallback stroke and no widened slack:
    the caller publishes ``valid = 0`` (deploy) or resamples the ball (training).
    """
    sel = sel or SelectorCfg()
    cfg = cfg or AdapterCfg()
    choice = select_stroke(
        p_ball_w, v_ball_w, omega_ball_w, t_avail_s, base_pos_w, base_quat_wxyz, aim_xy_w,
        protos, prev_choice=prev_choice, cfg=sel,
    )
    if not choice.ok:
        return choice, None

    budget = int(sel.max_solve_attempts) or len(choice.ranked_alternatives)
    reasons = dict(choice.reject_by_stroke)
    last_cmd = None
    for clip_index in choice.ranked_alternatives[:budget]:
        proto = protos[clip_index]
        cmd = fit_stroke_to_ball(
            p_ball_w, v_ball_w, omega_ball_w, aim_xy_w, proto, planner,
            base_quat_wxyz=base_quat_wxyz, t_strike_s=t_strike_s, cfg=cfg, sel=sel,
        )
        if cmd.ok:
            return (
                StrokeChoice(
                    clip_index=proto.clip_index,
                    motion_id=proto.motion_id,
                    family=proto.family_sign,
                    ranked_alternatives=choice.ranked_alternatives,
                    reject_by_stroke=reasons,
                    v_n_req_by_stroke=choice.v_n_req_by_stroke,
                ),
                cmd,
            )
        reasons[proto.motion_id] = "adapter_failed:" + cmd.reason
        last_cmd = cmd

    return (
        StrokeChoice(
            clip_index=-1,
            reject_reason="adapter_failed:" + (last_cmd.reason if last_cmd else "no_candidate"),
            ranked_alternatives=choice.ranked_alternatives,
            reject_by_stroke=reasons,
            v_n_req_by_stroke=choice.v_n_req_by_stroke,
        ),
        last_cmd,
    )
