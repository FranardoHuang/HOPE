"""Stroke SELECTOR — given the ball at the strike, which stroke do we play?

人话:球飞过来,先决定"打什么球":挡(自己几乎不出速度,但准备快,靠拍面角度借对方的力)还是
拉(自己出速度和上旋,但需要时间)。判据不是拍子速度也不是来球速度,而是两者之和——**合速度**
(closing speed):量到的相关系数 -0.61 说明拍速和来球速度大约 1:1 互相替代,所以一个动作的
"擅长区间"是一条 `拍速 + 来球速度 = 常数` 的线,而不是一个来球速度。把动作挥快一倍,它的
最佳来球速度就往下移一格——这就是 bh_loop_c 在自己速度下峰值 5.12 m/s(在训练球箱 2.0-4.6
之外,0.150 分)、挥 2x 后峰值移到 3.38(箱内,0.969 分)的全部机制。

Every admissibility condition below is an inequality on a MEASURED quantity — time available,
closing speed through the shipped contact model, contact height, reach, side — not a heuristic
score.  There is no weight to tune: the only human knob is one integer per stroke (``priority``).

Frames: everything in ONE caller frame, declared by ``SelectorCfg.surface_z`` (the table surface
height in the caller's own z).  The planner works in W_table (surface_z = 0.0); the trainer works
in W_floor (surface_z = 0.76).  The prototype's ``band_z_w`` is stored in W_floor, so the
comparison adds ``cfg.z_floor_offset`` = ``0.76 - surface_z``.  No fourth frame is invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .stroke_prototypes import StrokePrototype, StrokePrototypeSet, direction_world

# The physical table surface height above the FLOOR, and the ball radius: together they fix the
# lowest height at which a ball centre can ever exist over the table.  Same constants the virtual
# ball is integrated against (hope_commands.vb_table_surface_z / geometry.BALL_RADIUS).
TABLE_SURFACE_Z_W_FLOOR = 0.76
BALL_RADIUS_M = 0.02


@dataclass
class SelectorCfg:
    """Every threshold the selector uses.  All of them are physical, none is a weight."""

    # --- side (P1); mirrors SwingSideSelector / pp_policy.hpp:208-209 -------------------
    split_y: float = 0.0
    hysteresis_y: float = 0.04
    # --- time (P2) ---------------------------------------------------------------------
    wait_slack_s: float = 1.0
    # --- energy (P5) -------------------------------------------------------------------
    eps_energy: float = 0.10          # m/s
    delta_t_flight_s: float = 0.45    # the mirror-law screen's nominal flight time
    paddle_e_g1: float = 0.759        # e(u_n) = g1 exp(g2 |u_n|); configs/ball_physics_venue.yaml
    paddle_e_g2: float = -0.0441
    gravity: float = 9.81
    # --- frames ------------------------------------------------------------------------
    surface_z: float = 0.0            # table surface height in the CALLER's z frame
    ball_radius: float = BALL_RADIUS_M
    # --- misc ---------------------------------------------------------------------------
    max_solve_attempts: int = 0       # 0 = unlimited (training); the deploy path passes 2
    explore_prob: float = 0.0         # >0 only for a deliberate diversity arm (training)

    @property
    def z_floor_offset(self) -> float:
        """Add to a caller-frame z to get a height above the FLOOR (W_floor)."""
        return TABLE_SURFACE_Z_W_FLOOR - float(self.surface_z)

    @property
    def min_contact_z_w_floor(self) -> float:
        """No ball centre over the table can be lower than surface + one radius."""
        return TABLE_SURFACE_Z_W_FLOOR + float(self.ball_radius)


@dataclass(frozen=True)
class StrokeChoice:
    """What the selector decided, and why."""

    clip_index: int                   # -1 = nothing admissible
    motion_id: str = ""
    family: float = 0.0               # the wire's swing_sign: +1 forehand, -1 backhand
    ranked_alternatives: Tuple[int, ...] = ()
    reject_reason: str = ""
    #: per-stroke first failing predicate, e.g. {"bh_block": "P5a_energy_insufficient"}
    reject_by_stroke: Dict[str, str] = field(default_factory=dict)
    #: the closing-speed demand each stroke would have to supply, m/s (diagnostic + adapter seed)
    v_n_req_by_stroke: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.clip_index >= 0


def base_yaw_angle(base_quat_wxyz) -> float:
    """Yaw of a wxyz quaternion — the same expression as ``base_yaw_relative_y``
    (node_runtime_contract.py:871) and the C++ runner's ``tgt_b`` (pp_policy.hpp:2362)."""
    q = np.asarray(base_quat_wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-9:
        raise ValueError("base orientation quaternion is missing (norm ~ 0)")
    qw, qx, qy, qz = q / n
    return float(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def base_yaw_relative(p_w, base_pos_w, base_quat_wxyz) -> np.ndarray:
    """Point in the base's yaw-aligned frame B_yaw.

    ``p_b = R_z(-yaw) (p_w - p_base_w)``.  Its y component is exactly
    ``node_runtime_contract.base_yaw_relative_y`` (asserted in the tests), so the family this
    selector returns can never disagree with the deploy-side side rule.
    """
    d = np.asarray(p_w, dtype=float).reshape(-1)[:3] - \
        np.asarray(base_pos_w, dtype=float).reshape(-1)[:3]
    yaw = base_yaw_angle(base_quat_wxyz)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]], dtype=float)


def closing_speed_demand(
    p_ball_w,
    v_ball_w,
    aim_xy_w,
    cfg: SelectorCfg,
) -> Tuple[float, np.ndarray]:
    """(v_n_req, n_hat0) — the racket normal-channel speed this ball+aim DEMANDS.

    Exactly the shipped contact model, not a heuristic.  In ``predict_paddle_contact`` the lever
    arm is ``r = -R n_hat`` so ``(omega x r) . n_hat == 0`` identically and the normal channel is
    ``dv_n = -(1 + e) u_n n_hat`` with ``u_n = (v_ball - v_r) . n_hat``.  Therefore

        v_out . n_hat = -e (v_ball . n_hat) + (1 + e) (v_r . n_hat)

    and, writing ``a = v_ball . n_hat`` (negative for an incoming ball) and ``b = v_out . n_hat``,

        v_n_req = (b + e a) / (1 + e),     e = clamp(g1 exp(g2 |a - v_n_req|), 0.05, 0.95)

    solved by the same 3-step fixed point the solver seeds with.  This is the number that makes
    the block/loop trade-off fall out with no tuning: for a FAST ball ``a`` is large and negative,
    ``e a`` dominates and ``v_n_req`` is small — the block borrows the ball's energy; for a SLOW
    ball ``a ~ 0`` and ``v_n_req ~ b / (1 + e) > 0``, which a nearly-horizontal, ~1.7 m/s-capped
    block cannot supply at any aim.
    """
    p = np.asarray(p_ball_w, dtype=float)[:3]
    v = np.asarray(v_ball_w, dtype=float)[:3]
    aim = np.asarray(aim_xy_w, dtype=float)[:2]
    T = float(cfg.delta_t_flight_s)
    p_land = np.array([aim[0], aim[1], float(cfg.surface_z) + float(cfg.ball_radius)])
    g_vec = np.array([0.0, 0.0, -float(cfg.gravity)])
    v_out = (p_land - p) / T - 0.5 * g_vec * T

    dv = v_out - v
    nrm = float(np.linalg.norm(dv))
    n0 = dv / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
    if n0[0] < 0.0:
        n0 = -n0                                   # opponent-facing (+x) convention

    a = float(np.dot(v, n0))
    b = float(np.dot(v_out, n0))
    e = 0.5
    v_n_req = (b + e * a) / (1.0 + e)
    for _ in range(3):
        u_n = abs(a - v_n_req)
        e = float(np.clip(cfg.paddle_e_g1 * math.exp(cfg.paddle_e_g2 * u_n), 0.05, 0.95))
        v_n_req = (b + e * a) / (1.0 + e)
    return float(v_n_req), n0


def admissible(
    proto: StrokePrototype,
    *,
    p_ball_b: np.ndarray,
    z_w_floor: float,
    t_avail_s: float,
    v_n_req: float,
    proj: float,
    prev_family: Optional[float],
    cfg: SelectorCfg,
) -> Tuple[bool, str]:
    """Run P0..P6 in order; return (ok, first failing predicate name)."""
    # P0 — ENABLED.
    if not proto.enabled:
        return False, "P0_disabled"

    # P1 — SIDE.  Schmitt band; inside it BOTH families are admissible and the tie is broken
    # toward the previous choice, exactly like SwingSideSelector, so the deploy-side cross-check
    # in pp_reference_clock.hpp:74-99 can never reject us.
    y_b = float(p_ball_b[1])
    lo, hi = cfg.split_y - cfg.hysteresis_y, cfg.split_y + cfg.hysteresis_y
    if proto.family == "forehand":
        if not y_b < hi:
            return False, "P1_side"
    else:
        if not y_b > lo:
            return False, "P1_side"

    # P2 — TIME AVAILABLE.  "a loop needs time" as an inequality.
    if t_avail_s < proto.t_prepare_min_s:
        return False, "P2_time_too_short"
    if t_avail_s > proto.t_prepare_max_s + cfg.wait_slack_s:
        return False, "P2_time_too_long"

    # P3 — CONTACT HEIGHT.  The measured band, intersected with the physically legal floor: a ball
    # centre out over the table is never below surface + radius, so a band that dips under it
    # (the loops sweep down to 0.65-0.70 m during their rise) must not admit such a ball.
    z_lo = max(proto.band_z_w[0] - proto.slack_z_w_m, cfg.min_contact_z_w_floor)
    z_hi = proto.band_z_w[1] + proto.slack_z_w_m
    if not (z_lo <= z_w_floor <= z_hi):
        return False, "P3_height"

    # P4 — REACH.  The MEASURED sweep, not a shared half-width (blocks sweep millimetres, loops
    # sweep ~0.4 m; one shared box is 12x too wide for one and 4x too narrow for the other).
    x_b = float(p_ball_b[0])
    if not (proto.band_b_x[0] - proto.slack_b_xy_m <= x_b <= proto.band_b_x[1] + proto.slack_b_xy_m):
        return False, "P4_reach"
    if not (proto.band_b_y[0] - proto.slack_b_xy_m <= y_b <= proto.band_b_y[1] + proto.slack_b_xy_m):
        return False, "P4_reach"

    # P5 — ENERGY, in CLOSING SPEED.  P5a: the stroke can supply the demand at its speed cap.
    # P5b: it cannot avoid over-hitting even at its slowest.
    supply_max = proto.speed_max_mps * max(proj, 0.0)
    supply_min = proto.speed_min_mps * proj
    if v_n_req > supply_max + cfg.eps_energy:
        return False, "P5a_energy_insufficient"
    if v_n_req < supply_min - cfg.eps_energy:
        return False, "P5b_energy_excess"

    # P6 — RUNWAY.  Redundant with speed_max (which already folds v_star_cap in) so that a
    # runway-bound rejection is attributable rather than hidden inside P5a.
    if v_n_req / max(proj, 1e-6) > proto.v_star_cap_mps + cfg.eps_energy:
        return False, "P6_runway"
    return True, ""


def sort_key(
    proto: StrokePrototype, t_avail_s: float, v_n_req: float, proj: float,
) -> Tuple[float, float, float, int]:
    """The total order of §3.3.  ``clip_index`` last makes it total: no two prototypes tie."""
    time_margin = (t_avail_s - proto.t_prepare_min_s) / max(proto.t_prepare_s, 1e-9)
    demand = v_n_req / max(proto.speed_max_mps * max(proj, 1e-6), 1e-6)
    return (float(proto.priority), -float(time_margin), float(demand), int(proto.clip_index))


def select_stroke(
    p_ball_w,
    v_ball_w,
    omega_ball_w,
    t_avail_s: float,
    base_pos_w,
    base_quat_wxyz,
    aim_xy_w,
    protos: StrokePrototypeSet,
    prev_choice: Optional[StrokeChoice] = None,
    cfg: Optional[SelectorCfg] = None,
) -> StrokeChoice:
    """Pick the stroke to play.  Pure: no state except ``prev_choice``.

    ``omega_ball_w`` is accepted (and carried to the adapter) but does not enter the screen: the
    shipped contact model's normal channel is spin-independent by construction (``(omega x r) .
    n_hat == 0`` when ``r = -R n_hat``), so admitting on spin would be a heuristic, not physics.

    Returns a :class:`StrokeChoice`.  When nothing is admissible it returns ``clip_index = -1``
    with a populated ``reject_by_stroke`` — never an exception, never a fallback stroke, never a
    widened slack.  The caller publishes ``valid = 0`` (deploy) or resamples the ball (training).
    """
    cfg = cfg or SelectorCfg()
    p_ball_b = base_yaw_relative(p_ball_w, base_pos_w, base_quat_wxyz)
    z_w_floor = float(np.asarray(p_ball_w, dtype=float)[2]) + cfg.z_floor_offset
    yaw = base_yaw_angle(base_quat_wxyz)
    v_n_req, n0 = closing_speed_demand(p_ball_w, v_ball_w, aim_xy_w, cfg)
    prev_family = None if prev_choice is None or not prev_choice.ok else prev_choice.family

    reasons: Dict[str, str] = {}
    demands: Dict[str, float] = {}
    ranked = []
    for proto in protos:
        d_hat = direction_world(proto, yaw)
        proj = float(np.dot(d_hat, n0))
        demands[proto.motion_id] = v_n_req
        ok, why = admissible(
            proto, p_ball_b=p_ball_b, z_w_floor=z_w_floor, t_avail_s=float(t_avail_s),
            v_n_req=v_n_req, proj=proj, prev_family=prev_family, cfg=cfg,
        )
        if ok:
            ranked.append((sort_key(proto, float(t_avail_s), v_n_req, proj), proto))
        else:
            reasons[proto.motion_id] = why

    if not ranked:
        return StrokeChoice(
            clip_index=-1,
            reject_reason="no_stroke_admissible",
            reject_by_stroke=reasons,
            v_n_req_by_stroke=demands,
        )

    ranked.sort(key=lambda kv: kv[0])
    # Family hysteresis inside the +/-hysteresis_y band: when the leader and the runner-up differ
    # only in family and the ball sits inside the Schmitt band, keep the previous family.
    if prev_family is not None and len(ranked) > 1:
        y_b = float(p_ball_b[1])
        inside = abs(y_b - cfg.split_y) <= cfg.hysteresis_y
        if inside and ranked[0][1].family_sign != prev_family:
            for i, (_, cand) in enumerate(ranked):
                if cand.family_sign == prev_family and ranked[i][0][0] == ranked[0][0][0]:
                    ranked.insert(0, ranked.pop(i))
                    break

    best = ranked[0][1]
    return StrokeChoice(
        clip_index=best.clip_index,
        motion_id=best.motion_id,
        family=best.family_sign,
        ranked_alternatives=tuple(p.clip_index for _, p in ranked),
        reject_by_stroke=reasons,
        v_n_req_by_stroke=demands,
    )
