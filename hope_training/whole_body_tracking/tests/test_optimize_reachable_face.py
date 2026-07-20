"""optimize_reachable_face（轴 B 可达拍面优化器）unit tests —— numpy only,无 isaac/torch/mujoco。

人话:给定来球+触球点+拍速+目标落点,工具在 signed 击球面半球上找"合法回球最优拍面",
输出物理最优/可达最优/reward gap/n/-n 负控/landscape/分桶建议 manifest。这里验证:

* signed 单面硬门 n·v_hit > 0 严格无 abs(行为 + 源码守卫,SMASH Eq.11 对称形式禁止复制);
* n/-n 负控:翻面必被 signed 门拒 + reward 显著更差(管线内断言 + payload 落档);
* 已知玩具场景解析解:垂直来球+水平拍面,出球速度 = u + e(v+u)(venue 拟合 e(u_n) 口径),
  无切向/无旋转变化;且该玩具场景不过网 -> 硬门 fail-closed 拿 GATED 哨兵;
* 输入校验 fail-closed(非有限值/来球方向/触球点越网/落点出台);
* 可达圆锥参数校验 fail-closed;找不到合法拍面当场 SystemExit;
* landscape 全有限值、行数=粗网格、优化结果 ≥ 粗网格最优(细化只增不减);
* 可达最优在圆锥内、binding 圆锥时 reward gap > 0 且恒 ≥ 0;
* 分桶约定写死 train 0.80/1.00/1.20 · interp 0.90/1.10 · OOD 0.65/1.35,建议行 = 比例×最优角;
* manifest 一层内容寻址:sha256(canonical payload) 与文件内 sha 一致(CLI 端到端)。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_optimize_reachable_face.py -q
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import optimize_reachable_face as orf  # noqa: E402
import virtual_return_scorer as vrs  # noqa: E402

# 测试网格:粗 15° + 1 层细化(几秒级);语义与默认 6°/2 层一致,只是分辨率低。
COARSE, REFINE = 15.0, 1


@pytest.fixture(scope="module")
def geom():
    return orf.build_geometry()


@pytest.fixture(scope="module")
def params():
    return vrs.load_venue_params()


@pytest.fixture(scope="module")
def spec(geom):
    return vrs.VirtualReturnSpec(
        table_surface_z=geom.table_surface_z, net_x=geom.net_x, far_x=geom.far_x,
        half_width=geom.half_width, net_height=geom.net_height)


@pytest.fixture(scope="module")
def scenario(geom):
    """现实一题:venue 接触盒内触球,来球朝机器人,拍速朝对手,落点在对方台内。"""
    return orf.build_scenario(
        ball_vel=[-2.5, 0.3, -0.8], ball_spin=[0.0, 0.0, 0.0],
        contact_pos=[0.6, -0.1, 1.05], racket_vel=[4.0, 0.5, 1.5],
        target_landing_xy=[2.6, 0.3], geometry=geom)


@pytest.fixture(scope="module")
def payload_default(scenario, geom, params):
    """默认(拍速方向可达轴,35° 半角,非 binding)管线一跑,多测共用。"""
    return orf.run_pipeline(scenario, geometry=geom, params=params,
                            coarse_deg=COARSE, refine_levels=REFINE)


@pytest.fixture(scope="module")
def payload_binding(scenario, geom, params):
    """binding 圆锥(轴偏离物理最优,半角 15°):可达最优被压到锥边,gap > 0。"""
    return orf.run_pipeline(scenario, geometry=geom, params=params,
                            reach_axis=orf.face_from_angles(-15.0, -10.0),
                            reach_half_angle_deg=15.0,
                            coarse_deg=COARSE, refine_levels=REFINE)


# ---- signed 单面硬门 --------------------------------------------------------------------------
def test_signed_gate_strict_single_face():
    v = np.array([4.0, 0.5, 1.5])
    n = v / np.linalg.norm(v)
    ok, dot = orf.signed_face_gate(n, v)
    assert ok and dot > 0.0
    ok_flip, dot_flip = orf.signed_face_gate(-n, v)
    assert not ok_flip and dot_flip < 0.0
    # 严格单面:垂直(dot==0)也不过 —— 不是 |dot| 判据
    ok_perp, dot_perp = orf.signed_face_gate(np.array([0.0, 0.0, 1.0]),
                                             np.array([1.0, 0.0, 0.0]))
    assert not ok_perp and dot_perp == 0.0


def test_signed_gate_source_has_no_abs():
    """源码守卫:SMASH Eq.11 的 n/-n 对称形式(abs)禁止悄悄回流。"""
    src = inspect.getsource(orf.signed_face_gate)
    assert "abs(" not in src and "np.abs" not in src and "fabs" not in src


def test_flipped_optimum_is_gated(payload_default, scenario, params, geom, spec):
    """n* 合法,-n* 必须被 signed 门拒且拿 GATED 哨兵(负控的第一性行为)。"""
    n_star = np.asarray(payload_default["unconstrained_optimum"]["normal"], float)
    ev = orf.evaluate_face(n_star, scenario, params, geom, spec=spec)
    assert ev["legal"] and ev["reward_physics"] > 0.0
    flipped = orf.evaluate_face(-n_star / np.linalg.norm(n_star), scenario, params, geom,
                                spec=spec)
    assert not flipped["signed_face_ok"] and not flipped["legal"]
    assert flipped["reward_physics"] == orf.GATED_REWARD
    assert ev["reward_physics"] - flipped["reward_physics"] >= orf.NEG_CONTROL_MIN_GAP


def test_negative_control_recorded_in_payload(payload_default):
    for key in ("unconstrained_flipped", "reachable_flipped"):
        neg = payload_default["negative_control"][key]
        assert neg["signed_face_ok"] is False and neg["legal"] is False
        assert neg["reward_gap_vs_optimum"] >= orf.NEG_CONTROL_MIN_GAP
    assert payload_default["negative_control"]["min_gap_required"] == orf.NEG_CONTROL_MIN_GAP


# ---- 玩具场景解析解:垂直来球 + 水平拍面 ------------------------------------------------------
def test_vertical_ball_horizontal_face_analytic_contact(params, geom, spec):
    """v_in=(0,0,-v) 垂直下落,拍面 n=(0,0,1) 水平朝上,拍速 (0,0,u):
    法向相对速度 v+u,无切向 -> 出球 = (0, 0, u + e(v+u)),e = clip(g1·exp(g2(v+u)),0.05,0.95),
    旋转不变。经典拍面反弹闭式解,逐位对照 venue 冲量律输出。"""
    v_ball, u_racket = 3.0, 2.0
    toy = orf.Scenario(
        ball_vel=np.array([0.0, 0.0, -v_ball]), ball_spin=np.zeros(3),
        contact_pos=np.array([0.6, 0.0, 1.0]), racket_vel=np.array([0.0, 0.0, u_racket]),
        target_landing_xy=np.array([2.6, 0.0]))
    ev = orf.evaluate_face(np.array([0.0, 0.0, 1.0]), toy, params, geom, spec=spec)
    rel = v_ball + u_racket
    e = float(np.clip(params.paddle_e_g1 * math.exp(params.paddle_e_g2 * rel), 0.05, 0.95))
    expected = np.array([0.0, 0.0, u_racket + e * rel])
    assert np.allclose(ev["outgoing_vel"], expected, rtol=0.0, atol=1e-9)
    assert np.allclose(ev["outgoing_spin"], 0.0, atol=1e-12)
    assert ev["signed_face_ok"] and ev["contact_ok"]  # n·v_hit = u > 0,approach = u > 0.3


def test_vertical_toy_fails_net_gate(params, geom, spec):
    """垂直反弹不前进 -> 不过网,硬门 fail-closed:reward = GATED 哨兵(有限),legal False。"""
    toy = orf.Scenario(
        ball_vel=np.array([0.0, 0.0, -3.0]), ball_spin=np.zeros(3),
        contact_pos=np.array([0.6, 0.0, 1.0]), racket_vel=np.array([0.0, 0.0, 2.0]),
        target_landing_xy=np.array([2.6, 0.0]))
    ev = orf.evaluate_face(np.array([0.0, 0.0, 1.0]), toy, params, geom, spec=spec)
    assert not ev["net_crossed"] and not ev["legal"]
    assert ev["reward_physics"] == orf.GATED_REWARD and math.isfinite(ev["reward_physics"])


# ---- 输入/参数 fail-closed --------------------------------------------------------------------
def test_scenario_rejects_nonfinite(geom):
    with pytest.raises(ValueError, match="ball_vel"):
        orf.build_scenario(ball_vel=[np.nan, 0, -1], ball_spin=[0, 0, 0],
                           contact_pos=[0.6, 0, 1.0], racket_vel=[4, 0, 1],
                           target_landing_xy=[2.6, 0.0], geometry=geom)


def test_scenario_rejects_ball_not_approaching(geom):
    with pytest.raises(ValueError, match="approaching"):
        orf.build_scenario(ball_vel=[1.0, 0, -1], ball_spin=[0, 0, 0],
                           contact_pos=[0.6, 0, 1.0], racket_vel=[4, 0, 1],
                           target_landing_xy=[2.6, 0.0], geometry=geom)


def test_scenario_rejects_bad_contact_and_target(geom):
    with pytest.raises(ValueError, match="our side"):
        orf.build_scenario(ball_vel=[-2, 0, -1], ball_spin=[0, 0, 0],
                           contact_pos=[2.5, 0, 1.0], racket_vel=[4, 0, 1],
                           target_landing_xy=[2.6, 0.0], geometry=geom)
    with pytest.raises(ValueError, match="opponent half"):
        orf.build_scenario(ball_vel=[-2, 0, -1], ball_spin=[0, 0, 0],
                           contact_pos=[0.6, 0, 1.0], racket_vel=[4, 0, 1],
                           target_landing_xy=[1.0, 0.0], geometry=geom)  # 我方半区
    with pytest.raises(ValueError, match="opponent half"):
        orf.build_scenario(ball_vel=[-2, 0, -1], ball_spin=[0, 0, 0],
                           contact_pos=[0.6, 0, 1.0], racket_vel=[4, 0, 1],
                           target_landing_xy=[2.6, 1.2], geometry=geom)  # 出台宽


def test_reach_cone_parameter_validation():
    n = orf.face_from_angles(0.0, 0.0)
    with pytest.raises(ValueError, match="half angle"):
        orf.reach_cone_margin(n, [1, 0, 0], 0.0)
    with pytest.raises(ValueError, match="half angle"):
        orf.reach_cone_margin(n, [1, 0, 0], 120.0)
    with pytest.raises(ValueError, match="nonzero"):
        orf.reach_cone_margin(n, [0.0, 0.0, 0.0], 30.0)
    angle, margin, ok = orf.reach_cone_margin(n, [1.0, 0.0, 0.0], 30.0)
    assert angle == pytest.approx(0.0, abs=1e-9) and margin == pytest.approx(1.0) and ok


def test_no_legal_face_fails_closed(scenario, geom, params, spec):
    """拍速太小 -> 全候选被 phantom-block 门拒 -> 当场 SystemExit(不吐 manifest)。"""
    slow = orf.Scenario(ball_vel=scenario.ball_vel, ball_spin=scenario.ball_spin,
                        contact_pos=scenario.contact_pos,
                        racket_vel=np.array([0.05, 0.0, 0.0]),
                        target_landing_xy=scenario.target_landing_xy)
    with pytest.raises(SystemExit, match="no legal unconstrained"):
        orf.optimize_face(slow, params, geom, spec=spec, coarse_deg=45.0, refine_levels=0)


def test_unreachable_cone_fails_closed(scenario, geom, params, spec):
    """圆锥整个落在非法区(粗网格内无合法可达面)-> SystemExit,不静默降级。"""
    with pytest.raises(SystemExit, match="no legal reachable"):
        orf.optimize_face(scenario, params, geom, spec=spec, coarse_deg=COARSE,
                          refine_levels=0,
                          reach={"axis": list(orf.face_from_angles(-20.0, -12.0)),
                                 "half_angle_deg": 12.0})


# ---- landscape / 最优性 -----------------------------------------------------------------------
def test_landscape_finite_and_grid_shaped(payload_default):
    rows = payload_default["objective_landscape"]["rows"]
    n_yaw = len(np.arange(-180.0, 180.0, COARSE))
    n_pitch = len(np.arange(-85.0, 85.0 + 1e-9, COARSE))
    assert len(rows) == n_yaw * n_pitch
    for row in rows:
        assert len(row) == 4
        assert all(math.isfinite(float(v)) for v in row)  # GATED 哨兵也必须有限
        assert row[3] in (0, 1)
    assert any(row[3] == 1 for row in rows)  # 该场景存在合法回球


def test_optimum_dominates_coarse_landscape(payload_default):
    rows = payload_default["objective_landscape"]["rows"]
    coarse_best = max(row[2] for row in rows)
    assert payload_default["unconstrained_optimum"]["reward_physics"] >= coarse_best - 1e-12
    assert payload_default["unconstrained_optimum"]["legal"] is True


def test_reward_gap_nonnegative_and_zero_when_cone_loose(payload_default):
    """默认 35° 锥不 binding:可达最优 == 物理最优,gap == 0。"""
    assert payload_default["reward_gap_physics"] == pytest.approx(0.0, abs=1e-12)
    r = payload_default["reachable_optimum"]
    assert r["reach_ok"] and r["reach_angle_deg"] <= 35.0 + 1e-9


def test_binding_cone_positive_gap_and_in_cone(payload_binding):
    """binding 锥:可达最优被压进锥内(角 ≤ 半角),gap 严格 > 0。"""
    r = payload_binding["reachable_optimum"]
    assert r["reach_ok"] and r["reach_angle_deg"] <= 15.0 + 1e-6
    assert payload_binding["reward_gap_physics"] > 0.1
    u = payload_binding["unconstrained_optimum"]
    assert u["reward_physics"] >= r["reward_physics"]


# ---- 分桶约定 ---------------------------------------------------------------------------------
def test_bucket_convention_written_verbatim(payload_default):
    assert payload_default["bucket_convention"] == {
        "train": [0.80, 1.00, 1.20],
        "interpolation": [0.90, 1.10],
        "ood": [0.65, 1.35],
    }


def test_bucket_rows_scale_reachable_optimum(payload_default):
    r = payload_default["reachable_optimum"]
    for bucket, ratios in payload_default["bucket_convention"].items():
        rows = payload_default["bucket_suggestion"][bucket]
        assert [row["ratio"] for row in rows] == ratios
        for row in rows:
            assert row["signed_yaw_deg"] == pytest.approx(row["ratio"] * r["signed_yaw_deg"])
            assert row["signed_pitch_deg"] == pytest.approx(row["ratio"] * r["signed_pitch_deg"])
            assert isinstance(row["legal"], bool)
            assert math.isfinite(row["reward_physics"])
    # 比例 1.0(train 桶)就是可达最优本身,必须合法
    unit = [row for row in payload_default["bucket_suggestion"]["train"]
            if row["ratio"] == 1.00]
    assert len(unit) == 1 and unit[0]["legal"] is True


# ---- objective 无自指守卫 ---------------------------------------------------------------------
def test_reward_has_no_self_referential_normal_term():
    """reward 只吃回球结果量;不得出现"追踪自己选出的 target normal"(自指满分)。"""
    src = inspect.getsource(orf.reward_physics_terms)
    assert "normal" not in src and "target_normal" not in src
    sig = inspect.signature(orf.reward_physics_terms)
    assert set(sig.parameters) == {"net_margin", "land_err", "out_speed", "out_spin_mag"}


# ---- CLI 端到端 + 一层内容寻址 ----------------------------------------------------------------
def test_cli_end_to_end_manifest_content_addressed(tmp_path):
    out = tmp_path / "face_manifest.json"
    rc = orf.main([
        "--ball-vel=-2.5,0.3,-0.8", "--contact-pos", "0.6,-0.1,1.05",
        "--racket-vel", "4.0,0.5,1.5", "--target-landing", "2.6,0.3",
        "--coarse-deg", "18", "--refine-levels", "1", "--json", str(out),
    ])
    assert rc == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["kind"] == orf.MANIFEST_KIND
    assert manifest["schema_version"] == orf.SCHEMA_VERSION
    # 一层内容寻址:重算 canonical payload sha 必须一致(franco 拍板:单层 sha,不搞审批链)
    assert manifest["sha256"] == orf.canonical_sha256(manifest["payload"])
    payload = manifest["payload"]
    assert payload["unconstrained_optimum"]["legal"] is True
    assert payload["reachable_optimum"]["legal"] is True
    assert payload["reward_gap_physics"] >= 0.0
    assert payload["physics"]["venue_yaml_sha256"]
    assert payload["reachability"]["todo"].startswith("full")


def test_face_angle_roundtrip():
    for yaw, pitch in [(0.0, 0.0), (8.44, 3.89), (-15.0, -10.0), (170.0, 80.0),
                       (-179.0, -84.0)]:
        n = orf.face_from_angles(yaw, pitch)
        assert np.isfinite(n).all() and np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)
        yaw2, pitch2 = orf.angles_from_face(n)
        assert yaw2 == pytest.approx(yaw, abs=1e-9)
        assert pitch2 == pytest.approx(pitch, abs=1e-9)
    # signed:n 与 -n 的角度不同(半球身份不丢)
    n = orf.face_from_angles(8.44, 3.89)
    yaw_f, pitch_f = orf.angles_from_face(-n)
    assert abs(yaw_f - 8.44) > 90.0 and pitch_f == pytest.approx(-3.89, abs=1e-9)
