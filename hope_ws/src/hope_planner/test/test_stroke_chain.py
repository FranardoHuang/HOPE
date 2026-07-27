"""ball -> planner task -> target motion: prototypes, selector, adapter.

人话:这一套测试钉住三件事。
1. 动作模板文件是量出来的、能复现的、两边读同一份字节;
2. 选择器的判据是"合速度/时间/高度/够得着/哪一侧"这几条可测量的不等式——挡球答不了慢球、
   拉球没时间就不给选,都是这里写死的;
3. 适配器的四条约束(方向不变、触球点是球的、速度有上限、拍面自由)每一条都有对应断言,
   而且越界一律拒绝、绝不悄悄夹回范围。

Run: python -m pytest hope_ws/src/hope_planner/test/test_stroke_chain.py -q
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.node_runtime_contract import SwingSideSelector, base_yaw_relative_y
from hope_planner.strike_spec_fast import FastStrikeSpecPlanner
from hope_planner.strike_spec_planner import StrikeSpecPlanner
from hope_planner.stroke_adapt import (
    ALL_REASONS,
    REASON_NET,
    REASON_SPEED_OVER,
    AdapterCfg,
    fit_stroke_to_ball,
    select_and_fit,
)
from hope_planner.stroke_prototypes import (
    StrokePrototype,
    assert_strokes_are_distinguishable,
    direction_world,
    load_stroke_prototypes,
)
from hope_planner.stroke_select import (
    SelectorCfg,
    admissible,
    base_yaw_relative,
    closing_speed_demand,
    select_stroke,
    sort_key,
)

REPO = pathlib.Path(__file__).resolve().parents[4]
PROTO_PATH = REPO / "configs" / "stroke_prototypes_v1_20260727.json"

# The sim closed-loop convention: the table is centred on the robot (hope_planner.sim.yaml
# table_y_max 0.7825), and the planner's z = 0 is the TABLE SURFACE.
TABLE = TableParams(y_max=0.7825)
BASE_POS = np.array([-0.5, 0.0, 0.921 - 0.76])       # base_height_at_contact_m - table height
BASE_Q = np.array([1.0, 0.0, 0.0, 0.0])
AIM = np.array([2.05, 0.0])


@pytest.fixture(scope="module")
def protos():
    if not PROTO_PATH.exists():
        pytest.skip(f"{PROTO_PATH} not built; run scripts/build_stroke_prototypes.py")
    return load_stroke_prototypes(PROTO_PATH, scope="upper")


def _fast_config():
    """远粗近细 adaptive integration — the productionized setting (strike_spec_fast docstring).
    Keeps a 300-solve test suite inside seconds instead of minutes; the accuracy tolerances below
    are the ones benchmark_planner_latency already reports for it."""
    cfg = PlannerConfig()
    cfg.dt_integrate_coarse = 0.02
    return cfg


@pytest.fixture(scope="module")
def planner():
    return FastStrikeSpecPlanner(BallPhysics(), _fast_config(), TABLE)


def _sel():
    return SelectorCfg(surface_z=0.0)


def _acfg(**kw):
    """Adapter cfg paired with the coarse integrator.

    ``tol_m`` MUST match the integrator's own landing resolution: the 0.02 s adaptive cruise
    quantizes the landing point at ~1-2 cm (benchmark_planner_latency reports 18.6 mm for
    ``ss_fastnp_a20_warm``), so demanding 5 mm from it would fail every solve for a reason that
    has nothing to do with the stroke. 5 mm is the right tolerance for the 1 kHz scalar path.
    """
    kw.setdefault("use_fast_solver", True)
    kw.setdefault("tol_m", 0.02)
    return AdapterCfg(**kw)


def _contact_point(proto, z_frac=0.5):
    """A world (W_table) point inside this prototype's own measured contact region."""
    p = BASE_POS + np.array([proto.p_contact_b[0], proto.p_contact_b[1], 0.0])
    z_w_floor = proto.band_z_w[0] + z_frac * (proto.band_z_w[1] - proto.band_z_w[0])
    p[2] = max(z_w_floor, 0.78) - 0.76
    return p


# --------------------------------------------------------------------- prototypes --- #
def test_prototype_file_is_self_consistent(protos):
    """Loader verifies derived_sha256 over the records; a hand-edited field cannot load."""
    doc = json.loads(PROTO_PATH.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert len(protos) == 5
    assert protos.motion_ids == ("fh_loop", "bh_loop_c", "s0_highpress", "bh_block",
                                 "fh_block_syn")
    for i, p in enumerate(protos):
        assert p.clip_index == i
        assert p.scope == "upper"


def test_prototype_hand_edit_fails_closed(tmp_path, protos):
    doc = json.loads(PROTO_PATH.read_text(encoding="utf-8"))
    doc["scopes"]["upper"][0]["speed_max_mps"] = 99.0
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="derived_sha256"):
        load_stroke_prototypes(bad, scope="upper")


def test_prototype_sha_pin_fails_closed(protos):
    with pytest.raises(ValueError, match="does not match the pinned"):
        load_stroke_prototypes(PROTO_PATH, scope="upper", expected_sha256="0" * 64)


def test_prototype_clip_order_pin_fails_closed(protos):
    with pytest.raises(ValueError, match="clip order"):
        load_stroke_prototypes(PROTO_PATH, scope="upper",
                               expected_motion_ids=("bh_block", "fh_loop"))


def test_contact_frame_is_inside_the_compiled_window(protos):
    for p in protos:
        lo, hi = p.contact_window_frames
        assert lo <= p.contact_frame <= hi, f"{p.key}: {p.contact_frame} not in [{lo},{hi}]"
        assert 0.0 <= p.strike_phase <= 1.0


def test_speed_cap_is_the_min_of_the_two_measured_ceilings(protos):
    """The speed ceiling is TWO physical terms and nothing else.

    Both are measured off the clip: the retime ceiling (fastest replay before a joint runs out of
    velocity/acceleration) and the runway ceiling ``sqrt(2 a_max L_deep)``.  A third term — an
    unnamed ``3.5 - 0.1 = 3.4`` — used to sit inside this ``min``; it was removed on the owner's
    ruling 2026-07-27 because the 0.1 margin had no derivation and the literal silently cut
    fh_loop/full from 3.465 to 3.400.  This test is the guard against it coming back.
    """
    for p in protos:
        retime = p.speed_nominal_mps * p.retime_range[1]
        want = min(retime, p.v_star_cap_mps)
        assert math.isclose(p.speed_max_mps, want, rel_tol=1e-9), (
            f"{p.key}: speed_max {p.speed_max_mps} != min(retime {retime}, "
            f"v_star_cap {p.v_star_cap_mps}) — a third ceiling has been reintroduced"
        )


def test_deploy_gate_is_reported_never_applied(protos):
    """The on-robot runner's ``gate_speed_max`` is a NAMED receipt field, not a hidden clamp.

    pp_policy.hpp:234 sets ``gate_speed_max = 3.5`` and pp_policy.hpp:2459 rejects any commanded
    |v_racket| above it.  That is a property of the deploy runner, not of the stroke, so the build
    records it (and the per-stroke headroom) instead of shrinking the prototype.
    """
    doc = json.loads(PROTO_PATH.read_text(encoding="utf-8"))
    gate = doc["provenance"]["deploy_gate"]
    assert gate["applied"] is False
    assert gate["speed_max_mps"] == 3.5
    assert "pp_policy.hpp:234" in gate["source"]
    head = gate["headroom_mps"]
    assert set(head) == set(doc["scopes"]), "every scope needs a deploy-gate headroom row"
    for scope, rows in head.items():
        for rec in doc["scopes"][scope]:
            assert math.isclose(
                rows[rec["motion_id"]], gate["speed_max_mps"] - rec["speed_max_mps"], abs_tol=1e-6
            )
    assert gate["over_gate"] == [s + "/" + m for s, d in head.items() for m, h in d.items()
                                 if h < 0.0]


def test_direction_cone_cannot_convert_a_stroke(protos):
    assert_strokes_are_distinguishable(protos)
    loops = [p for p in protos if p.motion_id.endswith("loop") or p.motion_id.endswith("loop_c")]
    flats = [p for p in protos if p.motion_id in ("bh_block", "s0_highpress")]
    assert loops and flats
    assert min(p.elevation_deg for p in loops) - max(p.elevation_deg for p in flats) > 20.0


def test_prototype_velocity_points_forward(protos):
    for p in protos:
        assert p.v_hat_b[0] > 0.0
        assert abs(float(np.linalg.norm(p.v_hat_b)) - 1.0) < 1e-9


def test_prototype_rejects_a_backwards_stroke(protos):
    from dataclasses import replace
    with pytest.raises(ValueError, match="away from the opponent"):
        replace(protos[0], v_hat_b=np.array([-1.0, 0.0, 0.0]))


# ----------------------------------------------------------------------- frames --- #
def test_base_yaw_relative_matches_the_deploy_expression():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(-2, 2, 3)
        b = rng.uniform(-1, 1, 3)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert math.isclose(base_yaw_relative(p, b, q)[1], base_yaw_relative_y(p, b, q),
                            rel_tol=0, abs_tol=1e-12)


def test_z_frame_offset_is_declared_not_guessed():
    assert math.isclose(SelectorCfg(surface_z=0.0).z_floor_offset, 0.76)      # planner, W_table
    assert math.isclose(SelectorCfg(surface_z=0.76).z_floor_offset, 0.0)      # trainer, W_floor
    assert math.isclose(SelectorCfg().min_contact_z_w_floor, 0.78)


def test_direction_world_rotates_with_base_yaw(protos):
    p = protos.by_motion_id("fh_loop")
    d0 = direction_world(p, 0.0)
    assert np.allclose(d0, p.v_hat_b)
    d90 = direction_world(p, math.pi / 2)
    assert math.isclose(float(np.dot(d0[:2], d90[:2])), 0.0, abs_tol=1e-9)


# --------------------------------------------------------------------- selector --- #
def test_block_cannot_answer_a_slow_ball(protos):
    """The owner's physics, pinned: a block borrows the ball's energy and has almost none itself."""
    block = protos.by_motion_id("bh_block")
    cfg = _sel()
    p = _contact_point(block)
    p_b = base_yaw_relative(p, BASE_POS, BASE_Q)
    d_hat = direction_world(block, 0.0)
    for v_in in (0.5, 1.0, 1.5):
        v_n_req, n0 = closing_speed_demand(p, np.array([-v_in, 0.0, -0.3]), AIM, cfg)
        ok, why = admissible(block, p_ball_b=p_b, z_w_floor=p[2] + 0.76, t_avail_s=0.6,
                             v_n_req=v_n_req, proj=float(np.dot(d_hat, n0)),
                             prev_family=None, cfg=cfg)
        assert not ok and why == "P5a_energy_insufficient", (v_in, why)
    v_n_req, n0 = closing_speed_demand(p, np.array([-4.0, 0.0, -0.3]), AIM, cfg)
    ok, why = admissible(block, p_ball_b=p_b, z_w_floor=p[2] + 0.76, t_avail_s=0.6,
                         v_n_req=v_n_req, proj=float(np.dot(d_hat, n0)),
                         prev_family=None, cfg=cfg)
    assert ok, why


def test_closing_speed_demand_falls_as_the_ball_gets_faster(protos):
    """Blade speed and incoming speed are ~1:1 substitutes (measured corr -0.61): the SAME aim
    demands less of the blade as the ball arrives faster."""
    cfg = _sel()
    p = _contact_point(protos.by_motion_id("bh_block"))
    demands = [closing_speed_demand(p, np.array([-v, 0.0, -0.3]), AIM, cfg)[0]
               for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert all(b < a for a, b in zip(demands, demands[1:])), demands
    # ~1:1 substitution: 4 m/s more ball buys roughly 1.3 m/s less blade through the (1+e) split
    assert 0.9 < (demands[0] - demands[-1]) / 4.0 * 3.0 < 1.4


def test_loop_needs_time(protos):
    loop, block = protos.by_motion_id("fh_loop"), protos.by_motion_id("bh_block")
    assert loop.t_prepare_min_s > block.t_prepare_min_s
    cfg = _sel()
    for proto, t_avail, want in ((loop, 0.55, False), (loop, 1.20, True),
                                 (block, 0.55, True)):
        p = _contact_point(proto)
        p_b = base_yaw_relative(p, BASE_POS, BASE_Q)
        d_hat = direction_world(proto, 0.0)
        v_n_req, n0 = closing_speed_demand(p, np.array([-3.5, 0.0, -0.3]), AIM, cfg)
        ok, why = admissible(proto, p_ball_b=p_b, z_w_floor=p[2] + 0.76, t_avail_s=t_avail,
                             v_n_req=v_n_req, proj=float(np.dot(d_hat, n0)),
                             prev_family=None, cfg=cfg)
        if want:
            assert ok, (proto.motion_id, t_avail, why)
        else:
            assert why.startswith("P2_"), (proto.motion_id, t_avail, why)


def test_disabled_stroke_is_never_selected(protos):
    """fh_block_syn is bh_block at 42% speed with a +Y face flip (same source_sha256); it ships
    disabled until it is rebuilt from a real forehand-block source."""
    syn = protos.by_motion_id("fh_block_syn")
    assert not syn.enabled
    rng = np.random.default_rng(3)
    for _ in range(500):
        p = BASE_POS + rng.uniform([0.3, -0.6, 0.2], [1.0, 0.6, 0.8])
        ch = select_stroke(p, rng.uniform([-5, -1, -2], [-0.5, 1, 1]), np.zeros(3),
                           float(rng.uniform(0.3, 1.5)), BASE_POS, BASE_Q, AIM, protos,
                           cfg=_sel())
        assert ch.motion_id != "fh_block_syn"


def test_no_stroke_admissible_returns_invalid_not_an_exception(protos):
    ch = select_stroke(BASE_POS + np.array([-3.0, 0.0, 0.2]), np.array([-3.0, 0.0, 0.0]),
                       np.zeros(3), 0.6, BASE_POS, BASE_Q, AIM, protos, cfg=_sel())
    assert ch.clip_index == -1
    assert ch.reject_reason == "no_stroke_admissible"
    assert set(ch.reject_by_stroke) == set(protos.motion_ids)
    assert not ch.ok


def test_selector_is_deterministic_and_totally_ordered(protos):
    rng = np.random.default_rng(7)
    for _ in range(2000):
        p = BASE_POS + rng.uniform([0.3, -0.7, 0.1], [1.1, 0.7, 0.9])
        v = rng.uniform([-6, -1.5, -3], [-0.3, 1.5, 1.0])
        t = float(rng.uniform(0.2, 1.8))
        a = rng.uniform([1.9, -0.6], [3.1, 0.6])
        c1 = select_stroke(p, v, np.zeros(3), t, BASE_POS, BASE_Q, a, protos, cfg=_sel())
        c2 = select_stroke(p, v, np.zeros(3), t, BASE_POS, BASE_Q, a, protos, cfg=_sel())
        assert c1.clip_index == c2.clip_index
        assert c1.ranked_alternatives == c2.ranked_alternatives
        assert len(set(c1.ranked_alternatives)) == len(c1.ranked_alternatives)


def test_sort_key_never_ties(protos):
    keys = [sort_key(p, 1.0, 1.5, 0.8) for p in protos]
    assert len(set(keys)) == len(keys)


def test_family_matches_the_deploy_side_rule(protos):
    """resolve_planner_swing_sign (pp_reference_clock.hpp:74-99) rejects a family that disagrees
    with sign(target_y) outside the Schmitt band, so the selector must agree with
    SwingSideSelector wherever it commits."""
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(4000):
        p = BASE_POS + rng.uniform([0.4, -0.7, 0.25], [1.0, 0.7, 0.85])
        ch = select_stroke(p, rng.uniform([-5, -1, -2], [-1.0, 1, 0.5]), np.zeros(3),
                           float(rng.uniform(0.35, 1.4)), BASE_POS, BASE_Q, AIM, protos,
                           cfg=_sel())
        if not ch.ok:
            continue
        y_b = base_yaw_relative(p, BASE_POS, BASE_Q)[1]
        if abs(y_b) <= 0.04:          # inside the Schmitt band both families are legal
            continue
        want = SwingSideSelector().select(p, BASE_POS, BASE_Q)
        assert ch.family == want, (y_b, ch.motion_id, ch.family, want)
        checked += 1
    assert checked > 100


# ---------------------------------------------------------------------- adapter --- #
def _solved_cases(protos, planner, n=40, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        proto = protos[int(rng.integers(0, len(protos)))]
        p = _contact_point(proto, float(rng.uniform(0.2, 0.8)))
        v = np.array([-float(rng.uniform(2.0, 5.0)), float(rng.uniform(-0.6, 0.6)),
                      -float(rng.uniform(0.0, 1.2))])
        aim = np.array([float(rng.uniform(1.9, 2.4)), float(rng.uniform(-0.5, 0.5))])
        cmd = fit_stroke_to_ball(p, v, np.zeros(3), aim, proto, planner,
                                 base_quat_wxyz=BASE_Q, cfg=_acfg(), sel=_sel())
        out.append((proto, p, v, aim, cmd))
    return out


def test_c1_velocity_direction_is_preserved(protos, planner):
    """C1. Stage 1 is exact by construction (v_r IS s * d_hat); stage 2 is bounded by the cone."""
    hit = 0
    for proto, p, v, aim, cmd in _solved_cases(protos, planner):
        d_hat = direction_world(proto, 0.0)
        if not cmd.ok:
            continue
        hit += 1
        v_hat = cmd.v_racket / np.linalg.norm(cmd.v_racket)
        dev = math.degrees(math.acos(float(np.clip(np.dot(v_hat, d_hat), -1, 1))))
        if cmd.stage == 1:
            # EXACT by construction: v_r IS s * d_hat, so assert that identity directly. The
            # arccos form is only checked to 1e-3 deg because acos is ill-conditioned at 1
            # (a 1e-16 dot-product error is ~1e-5 deg of apparent angle).
            assert np.allclose(cmd.v_racket, cmd.speed * d_hat, rtol=0, atol=1e-9)
            assert dev < 1e-3, (proto.motion_id, dev)
        else:
            assert dev <= proto.v_dir_tol_deg + 1e-6
    assert hit > 10


def test_c2_contact_position_is_the_ball_even_on_failure(protos, planner):
    """C2. Structural: the adapter does not CHOOSE the point, it returns the ball's own."""
    for proto, p, v, aim, cmd in _solved_cases(protos, planner):
        assert np.array_equal(cmd.p_contact, p), (proto.motion_id, cmd.ok)


def test_c3_speed_stays_inside_the_stroke_cap(protos, planner):
    for proto, p, v, aim, cmd in _solved_cases(protos, planner):
        if cmd.ok:
            assert proto.speed_min_mps - 1e-9 <= cmd.speed <= proto.speed_max_mps + 1e-9


def test_c3_out_of_range_refuses_never_clamps(protos, planner):
    """A landing the stroke cannot reach comes back ok=False, not a clamped command."""
    proto = protos.by_motion_id("bh_block")
    p = _contact_point(proto)
    cmd = fit_stroke_to_ball(p, np.array([-0.5, 0.0, -0.2]), np.zeros(3),
                             np.array([2.6, 0.0]), proto, planner, base_quat_wxyz=BASE_Q,
                             cfg=_acfg(), sel=_sel())
    assert not cmd.ok
    assert cmd.reason == REASON_SPEED_OVER
    assert cmd.reason in ALL_REASONS
    assert np.array_equal(cmd.p_contact, p)


def test_c4_blade_orientation_is_free_but_legal(protos, planner):
    tilts = []
    for proto, p, v, aim, cmd in _solved_cases(protos, planner, n=60, seed=9):
        if not cmd.ok:
            continue
        assert abs(float(np.linalg.norm(cmd.n_racket)) - 1.0) < 1e-9
        assert cmd.n_racket[0] > 1e-6
        tilts.append(math.degrees(math.asin(float(np.clip(cmd.n_racket[2], -1, 1)))))
    assert len(tilts) > 10
    assert max(tilts) - min(tilts) > 20.0, tilts


def test_c4_face_envelope_is_enforced_when_configured(protos, planner):
    proto = protos.by_motion_id("bh_block")
    p = _contact_point(proto)
    v = np.array([-3.5, 0.0, -0.4])
    base = fit_stroke_to_ball(p, v, np.zeros(3), AIM, proto, planner, base_quat_wxyz=BASE_Q,
                              cfg=_acfg(), sel=_sel())
    assert base.ok
    tight = _acfg(face_envelope_min_dot=0.999,
                  face_envelope_ref_b=(-1.0, 0.0, 0.0))
    out = fit_stroke_to_ball(p, v, np.zeros(3), AIM, proto, planner, base_quat_wxyz=BASE_Q,
                             cfg=tight, sel=_sel())
    assert not out.ok and out.reason == "face_out_of_envelope"


def test_net_clearance_is_an_acceptance_condition(protos, planner):
    """Regression for racket_target_planner.py:253-258, which returns valid=True even when every
    net-clearance retry failed."""
    proto = protos.by_motion_id("bh_block")
    p = _contact_point(proto)
    # aim just past the net: any flight that lands there passes UNDER the net top
    cmd = fit_stroke_to_ball(p, np.array([-3.0, 0.0, -0.5]), np.zeros(3),
                             np.array([1.42, 0.0]), proto, planner, base_quat_wxyz=BASE_Q,
                             cfg=_acfg(), sel=_sel())
    assert not cmd.ok
    assert cmd.reason in (REASON_NET, REASON_SPEED_OVER, "speed_under_min"), cmd.reason
    for _, _, _, _, c in _solved_cases(protos, planner, n=40, seed=13):
        if c.ok:
            assert c.clears_net
            assert c.net_z_margin_m > 0.02 - 1e-12


def test_fixed_direction_matches_the_free_solve_when_the_direction_is_free(protos, planner):
    """Guards the new forward map: seed d_hat with the FREE solve's own v_r direction and the
    fixed-direction solver must land on the same aim."""
    proto = protos.by_motion_id("bh_block")
    p = _contact_point(proto)
    v = np.array([-3.5, 0.0, -0.4])
    free = planner.solve(p, v, np.zeros(3), AIM, racket_speed_budget=10.0,
                         with_sensitivities=False)
    assert free is not None
    d = free.v_r / np.linalg.norm(free.v_r)
    s = float(np.linalg.norm(free.v_r))
    out = planner.solve_fixed_direction(p, v, np.zeros(3), AIM, d, 0.2 * s, 3.0 * s)
    assert out is not None
    assert out["resid_m"] < 0.005
    assert out["dir_deviation_deg"] < 1e-9


def test_fast_and_scalar_fixed_direction_agree(protos):
    """The fast (batched-probe, adaptive-integration) twin must reproduce the scalar solve."""
    slow = StrikeSpecPlanner(BallPhysics(), PlannerConfig(), TABLE)
    fast = FastStrikeSpecPlanner(BallPhysics(), _fast_config(), TABLE)
    ps = load_stroke_prototypes(PROTO_PATH, scope="upper")
    proto = ps.by_motion_id("bh_block")
    p = _contact_point(proto)
    d = direction_world(proto, 0.0)
    for vx in (-2.5, -3.5, -4.5):
        a = slow.solve_fixed_direction(p, np.array([vx, 0.0, -0.4]), np.zeros(3), AIM, d,
                                       proto.speed_min_mps, proto.speed_max_mps)
        b = fast.solve_fast_fixed_direction(p, np.array([vx, 0.0, -0.4]), np.zeros(3), AIM, d,
                                            proto.speed_min_mps, proto.speed_max_mps)
        assert a is not None and b is not None
        assert abs(a["speed"] - b["speed"]) < 0.05, (vx, a["speed"], b["speed"])
        assert float(np.linalg.norm(a["landing_xy"] - b["landing_xy"])) < 0.03


def test_adapter_never_maximises_speed(protos, planner):
    """Faster is not better: two strokes are already at their optimum and get worse when sped up.
    The solved speed must FALL as the incoming ball supplies more of the closing speed."""
    proto = protos.by_motion_id("bh_block")
    p = _contact_point(proto)
    speeds = []
    for v_in in (3.5, 4.5, 5.5):
        cmd = fit_stroke_to_ball(p, np.array([-v_in, 0.0, -0.4]), np.zeros(3), AIM, proto,
                                 planner, base_quat_wxyz=BASE_Q, cfg=_acfg(), sel=_sel())
        assert cmd.ok, (v_in, cmd.reason)
        assert cmd.speed < proto.speed_max_mps - 1e-6, "solved speed pinned at the cap"
        speeds.append(cmd.speed)
    assert speeds[0] > speeds[1] > speeds[2], speeds


def test_commanded_contact_is_never_below_the_table(protos, planner):
    """THE regression for the failure that started this: a commanded contact point out over the
    table below surface + ball radius.  Structural — the adapter returns the ball's own point."""
    rng = np.random.default_rng(17)
    n_ok = 0
    for _ in range(300):
        proto = protos[int(rng.integers(0, len(protos)))]
        p = _contact_point(proto, float(rng.uniform(0.0, 1.0)))
        cmd = fit_stroke_to_ball(p, np.array([-float(rng.uniform(2, 5)), 0.0, -0.4]),
                                 np.zeros(3), AIM, proto, planner, base_quat_wxyz=BASE_Q,
                                 cfg=_acfg(), sel=_sel())
        assert np.array_equal(cmd.p_contact, p)
        if cmd.p_contact[0] + 0.5 > 0.5:      # W_table x=0 IS the near table edge
            assert cmd.p_contact[2] + 0.76 >= 0.78 - 1e-9
        n_ok += int(cmd.ok)
    assert n_ok > 50


def test_selector_p3_refuses_a_below_table_contact(protos):
    """The measured band of a rising loop dips to 0.70 m; P3 clamps it to the legal floor so a
    ball centre below surface+radius can never be admitted."""
    loop = protos.by_motion_id("fh_loop")
    assert loop.band_z_w[0] < 0.78            # the measurement really does dip under the table
    cfg = _sel()
    p = _contact_point(loop)
    p_b = base_yaw_relative(p, BASE_POS, BASE_Q)
    d_hat = direction_world(loop, 0.0)
    v_n_req, n0 = closing_speed_demand(p, np.array([-3.5, 0.0, -0.3]), AIM, cfg)
    ok, why = admissible(loop, p_ball_b=p_b, z_w_floor=0.70, t_avail_s=1.2, v_n_req=v_n_req,
                         proj=float(np.dot(d_hat, n0)), prev_family=None, cfg=cfg)
    assert not ok and why == "P3_height"


def test_select_and_fit_walks_the_order_and_reports_reasons(protos, planner):
    proto = protos.by_motion_id("bh_block")
    p = _contact_point(proto)
    choice, cmd = select_and_fit(p, np.array([-3.5, 0.0, -0.4]), np.zeros(3), 0.6,
                                 BASE_POS, BASE_Q, AIM, protos, planner, sel=_sel(),
                                 cfg=_acfg())
    assert choice.ok and cmd is not None and cmd.ok
    assert cmd.clip_index == choice.clip_index

    far = BASE_POS + np.array([-3.0, 0.0, 0.2])
    choice2, cmd2 = select_and_fit(far, np.array([-3.0, 0.0, 0.0]), np.zeros(3), 0.6,
                                   BASE_POS, BASE_Q, AIM, protos, planner, sel=_sel())
    assert choice2.clip_index == -1 and cmd2 is None
    assert choice2.reject_reason == "no_stroke_admissible"
