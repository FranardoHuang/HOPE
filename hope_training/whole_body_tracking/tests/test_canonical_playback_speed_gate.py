"""Unit tests for scripts/canonical_playback_speed_gate.py(慢放可行性门).

纯 CPU,NO mujoco/torch:逆动力学是**合成的解析模型**,所以每条判卷结论都可以手算核对。
合成模型直接照抄真实结构 tau(s) = c0(q) + s·L(q,qd) + s²·Q(q,qd,qdd):
  c0 = 重力(不随 s 缩放)、L = 关节阻尼、Q = 惯性 + 科氏。

覆盖:
  ① 二次律辨识精确复原 c0/L/Q,并被独立探针核过;
  ② 非二次项(接触力那一类)当场 fail loud,拒绝据此发证;
  ③ 静态重力超限 ⇒ **任何**慢放比都不可行(max_playback_ratio == 0),这是慢放救不了的那一项;
  ④ 阻尼项造成的**内部顶点**越界会被抓到(只查端点会漏);
  ⑤ 请求比越界 ⇒ verdict=infeasible + assert_playback_ratio_admissible 抛异常;
  ⑥ full/grounded scope 拒绝只凭关节力矩发证(IncompletePlaybackCertificate);
  ⑦ 交付计划:r<1 且已验 ⇒ runtime_playback(省下一次烤入);r>1 ⇒ 必须烤;无证书 ⇒ fail closed;
  ⑧ MUTATION CHECK:把门的两处关键逻辑各突变一次,断言测试确实抓得住。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_canonical_playback_speed_gate.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pbg = _load("canonical_playback_speed_gate")

FPS = 50.0
NV = 4
FRAMES = 6


# --------------------------------------------------------------------------------- #
# 合成逆动力学:结构与真实模型同形,系数任意可控                                       #
# --------------------------------------------------------------------------------- #
class SyntheticPlant:
    """tau = gravity(q) + damping*qd + inertia@qacc + coriolis(qd) (+ 可选非二次项)。"""

    def __init__(self, gravity, damping, inertia, coriolis=0.0, contact_gain=0.0):
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.damping = np.asarray(damping, dtype=np.float64)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.coriolis = float(coriolis)
        self.contact_gain = float(contact_gain)
        self.calls = 0

    def __call__(self, qpos, qvel, qacc):
        self.calls += 1
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        qacc = np.asarray(qacc, dtype=np.float64)
        tau = (
            self.gravity * np.cos(qpos)
            + self.damping * qvel
            + self.inertia @ qacc
            + self.coriolis * qvel * qvel
        )
        if self.contact_gain:
            # 接触力那一类:|qd| 是 s 的绝对值函数,不是二次多项式 —— 这正是 full scope
            # 上 mj_inverse 残差 1.96-13.3 N·m 的成因。
            tau = tau + self.contact_gain * np.sqrt(np.abs(qvel) + 1.0)
        return tau


def make_reference(frames=FRAMES, nv=NV, vel_scale=1.0, acc_scale=1.0):
    rng = np.random.RandomState(7)
    qpos = rng.uniform(-0.4, 0.4, size=(frames, nv))
    qvel = vel_scale * rng.uniform(-1.0, 1.0, size=(frames, nv))
    qacc = acc_scale * rng.uniform(-1.0, 1.0, size=(frames, nv))
    return qpos, qvel, qacc


def make_torque_inputs(plant, limit=200.0, support_mode="fixed_base", **kwargs):
    qpos, qvel, qacc = make_reference(**kwargs)
    return pbg.TorqueScreenInputs(
        qpos=qpos,
        qvel=qvel,
        qacc=qacc,
        inverse_dynamics=plant,
        effort_lower=np.full(NV, -float(limit)),
        effort_upper=np.full(NV, float(limit)),
        dof_names=["j%d" % index for index in range(NV)],
        support_mode=support_mode,
    )


def default_plant(**kwargs):
    inertia = np.eye(NV) * 3.0 + 0.2
    return SyntheticPlant(
        gravity=np.array([12.0, -8.0, 5.0, 3.0]),
        damping=np.array([2.0, 1.0, 0.5, 2.0]),
        inertia=inertia,
        **kwargs
    )


def make_kinematic(peak_vel=1.0, peak_acc=1.0, vel_cap=10.0, acc_cap=10.0):
    joint_vel = np.zeros((3, NV), dtype=np.float64)
    # j0 只贡献速度峰值(恒定 ⇒ 加速度 0);j1 只贡献加速度峰值(|Δqd|·fps = peak_acc)。
    joint_vel[:, 0] = peak_vel
    joint_vel[:, 1] = np.arange(3, dtype=np.float64) * (peak_acc / FPS)
    return pbg.KinematicScreenInputs(
        joint_vel=joint_vel,
        fps=FPS,
        velocity_limits=np.full(NV, vel_cap),
        acceleration_envelope=np.full(NV, acc_cap),
        joint_names=["j%d" % index for index in range(NV)],
        vel_limit_frac=1.0,
        budget_scale=1.0,
    )


# --------------------------------------------------------------------------------- #
# ① 二次律辨识                                                                       #
# --------------------------------------------------------------------------------- #
def test_identification_recovers_the_exact_quadratic_coefficients():
    plant = default_plant()
    inputs = make_torque_inputs(plant)
    law = pbg.identify_playback_quadratic(inputs)

    expected_c0 = plant.gravity * np.cos(inputs.qpos)
    expected_c1 = plant.damping * inputs.qvel
    expected_c2 = inputs.qacc @ plant.inertia.T
    assert np.allclose(law.c0, expected_c0, atol=1e-9)
    assert np.allclose(law.c1, expected_c1, atol=1e-9)
    assert np.allclose(law.c2, expected_c2, atol=1e-9)
    assert law.max_probe_residual < 1e-9


def test_coriolis_lands_in_the_quadratic_term_not_the_linear_one():
    """qd² 项随 s²,必须落在 c2 上;落错地方会让顶点位置算错。"""
    plant = default_plant(coriolis=1.5)
    inputs = make_torque_inputs(plant)
    law = pbg.identify_playback_quadratic(inputs)
    expected_c2 = inputs.qacc @ plant.inertia.T + 1.5 * inputs.qvel ** 2
    assert np.allclose(law.c2, expected_c2, atol=1e-9)
    assert np.allclose(law.c1, plant.damping * inputs.qvel, atol=1e-9)


# --------------------------------------------------------------------------------- #
# ② 非二次项 fail loud                                                               #
# --------------------------------------------------------------------------------- #
def test_contact_like_nonquadratic_term_is_refused_loudly():
    plant = default_plant(contact_gain=4.0)
    inputs = make_torque_inputs(plant)
    with pytest.raises(pbg.PlaybackLawViolation, match="not quadratic in the speed ratio"):
        pbg.identify_playback_quadratic(inputs)


def test_certify_propagates_the_law_violation_instead_of_certifying():
    plant = default_plant(contact_gain=4.0)
    with pytest.raises(pbg.PlaybackLawViolation):
        pbg.certify_playback_ratio(0.8, make_kinematic(), make_torque_inputs(plant))


# --------------------------------------------------------------------------------- #
# ③ 静态重力:慢放救不了的那一项                                                      #
# --------------------------------------------------------------------------------- #
def test_gravity_over_limit_makes_every_playback_ratio_infeasible():
    """c0 不随 s 缩放:重力自己超限 ⇒ max_playback_ratio == 0,再慢也没用。"""
    plant = default_plant()
    plant.gravity = np.array([300.0, 10.0, 10.0, 10.0])   # j0 静态就超 200
    inputs = make_torque_inputs(plant, limit=200.0)
    screen = pbg.torque_playback_screen(inputs, 0.05)     # 极慢的慢放
    assert screen["pass"] is False
    assert screen["static_gravity"]["pass"] is False
    assert screen["static_gravity"]["utilisation_max"] > 1.0
    assert screen["max_ratio"] == 0.0
    assert screen["offending"][0]["dof"] == "j0"


def test_gravity_inside_limit_keeps_the_slow_end_feasible():
    plant = default_plant()
    inputs = make_torque_inputs(plant, limit=200.0)
    screen = pbg.torque_playback_screen(inputs, 0.5)
    assert screen["pass"] is True
    assert screen["static_gravity"]["pass"] is True
    assert screen["max_ratio"] > 1.0


# --------------------------------------------------------------------------------- #
# ④ 内部顶点                                                                         #
# --------------------------------------------------------------------------------- #
def _vertex_case():
    """手造一条抛物线:两端都在界内,顶点在区间内部越界。

    f(s) = 90 + 260 s - 260 s²,limit = ±100。
    f(0)=90 ✓  f(1)=90 ✓  顶点 s*=0.5 -> f=155 ✗
    """
    c0 = np.array([[90.0]])
    c1 = np.array([[260.0]])
    c2 = np.array([[-260.0]])
    lower = np.array([-100.0])
    upper = np.array([100.0])
    return c0, c1, c2, lower, upper


def test_first_exit_ratio_matches_hand_computed_roots():
    """闭式首次出界比:三种情形都手算得出来,不靠搜索。"""
    lower, upper = np.array([-200.0]), np.array([200.0])
    # f(s) = 50 + 100s + 100s²  ->  100s² + 100s - 150 = 0  ->  s = 0.8228756555
    exact = (-100.0 + np.sqrt(100.0 ** 2 + 4.0 * 100.0 * 150.0)) / 200.0
    got = pbg.first_exit_ratio(
        np.array([[50.0]]), np.array([[100.0]]), np.array([[100.0]]), lower, upper)
    assert np.isclose(got[0, 0], exact)
    # c2 = 0 的退化(纯阻尼)情形:50 + 100s = 200 -> s = 1.5
    got_linear = pbg.first_exit_ratio(
        np.array([[50.0]]), np.array([[100.0]]), np.array([[0.0]]), lower, upper)
    assert np.isclose(got_linear[0, 0], 1.5)
    # 常数项自己就在界外 -> 0.0(慢放救不了)
    got_static = pbg.first_exit_ratio(
        np.array([[250.0]]), np.array([[100.0]]), np.array([[100.0]]), lower, upper)
    assert got_static[0, 0] == 0.0
    # 永不出界 -> inf
    got_never = pbg.first_exit_ratio(
        np.array([[1.0]]), np.array([[0.0]]), np.array([[0.0]]), lower, upper)
    assert np.isinf(got_never[0, 0])


def test_interior_vertex_violation_is_detected():
    c0, c1, c2, lower, upper = _vertex_case()
    minimum, maximum, _, arg_max = pbg.quadratic_interval_extrema(c0, c1, c2, 0.0, 1.0)
    assert np.isclose(arg_max[0, 0], 0.5)
    assert np.isclose(maximum[0, 0], 155.0)
    assert np.isclose(minimum[0, 0], 90.0)
    exit_ratio = pbg.first_exit_ratio(c0, c1, c2, lower, upper)
    assert 0.0 < exit_ratio[0, 0] < 0.5


def test_torque_screen_reports_the_interior_vertex_on_a_real_screen():
    """把顶点用例塞进整条筛子,确认 offending 标了 interior_vertex。"""

    class VertexPlant:
        def __call__(self, qpos, qvel, qacc):
            # tau(s) = 90 + 260 s - 260 s²,通过 qvel/qacc 的 s 幂次构造
            speed = qvel[0]          # = s * 1.0
            accel = qacc[0]          # = s^2 * 1.0
            return np.array([90.0 + 260.0 * speed - 260.0 * accel])

    inputs = pbg.TorqueScreenInputs(
        qpos=np.zeros((1, 1)),
        qvel=np.ones((1, 1)),
        qacc=np.ones((1, 1)),
        inverse_dynamics=VertexPlant(),
        effort_lower=np.array([-100.0]),
        effort_upper=np.array([100.0]),
        dof_names=["vertex_joint"],
    )
    screen = pbg.torque_playback_screen(inputs, 1.0)
    assert screen["pass"] is False
    assert screen["interior_vertex_violation_count"] == 1
    assert screen["offending"][0]["interior_vertex"] is True
    assert np.isclose(screen["offending"][0]["at_ratio"], 0.5)
    # 端点自己都在界内 —— 这就是"只查端点会漏"的实证
    endpoint_only = max(abs(90.0), abs(90.0 + 260.0 - 260.0))
    assert endpoint_only <= 100.0


# --------------------------------------------------------------------------------- #
# ⑤ 请求比越界 ⇒ 大声拒绝                                                            #
# --------------------------------------------------------------------------------- #
def test_requested_ratio_outside_envelope_is_refused():
    kinematic = make_kinematic(peak_vel=8.0, vel_cap=10.0)   # 运动学上限 1.25
    plant = default_plant()
    certificate = pbg.certify_playback_ratio(1.4, kinematic, make_torque_inputs(plant))
    assert certificate["verdict"] == "infeasible"
    assert certificate["admissible"] is False
    assert np.isclose(certificate["screens"]["kinematic_envelope"]["max_ratio"], 1.25)
    with pytest.raises(pbg.PlaybackRatioInfeasible):
        pbg.assert_playback_ratio_admissible(certificate)


def test_requested_ratio_inside_envelope_is_admissible():
    kinematic = make_kinematic(peak_vel=8.0, vel_cap=10.0)
    certificate = pbg.certify_playback_ratio(
        0.8, kinematic, make_torque_inputs(default_plant())
    )
    assert certificate["verdict"] == "feasible"
    assert certificate["admissible"] is True
    pbg.assert_playback_ratio_admissible(certificate)


def test_kinematic_only_certificate_is_never_shippable():
    certificate = pbg.certify_playback_ratio(0.8, make_kinematic(), None)
    assert certificate["verdict"] == "incomplete"
    assert certificate["admissible"] is False
    assert certificate["claim_scope"] == "kinematic_only_NOT_shippable"
    assert "torque_quadratic" in certificate["screens_not_run"]
    with pytest.raises(pbg.PlaybackRatioInfeasible):
        pbg.assert_playback_ratio_admissible(certificate)


def test_ceiling_is_the_minimum_over_screens():
    """运动学上限 1.25、力矩上限更高 ⇒ 证书取 1.25(取 min,不许挑好看的那个)。"""
    kinematic = make_kinematic(peak_vel=8.0, vel_cap=10.0)
    certificate = pbg.certify_playback_ratio(
        1.0, kinematic, make_torque_inputs(default_plant())
    )
    torque_ceiling = certificate["screens"]["torque_quadratic"]["max_ratio"]
    assert torque_ceiling > 1.25
    assert np.isclose(certificate["max_playback_ratio"], 1.25)


# --------------------------------------------------------------------------------- #
# ⑥ full / grounded scope 拒绝发证                                                   #
# --------------------------------------------------------------------------------- #
def test_grounded_scope_refuses_a_joint_torque_only_verdict():
    inputs = make_torque_inputs(default_plant(), support_mode="grounded")
    with pytest.raises(pbg.IncompletePlaybackCertificate) as excinfo:
        pbg.torque_playback_screen(inputs, 0.8)
    assert "ground_contact_support_polygon_screen" in excinfo.value.report["missing"]
    assert "STATIC BALANCE" in str(excinfo.value)


# --------------------------------------------------------------------------------- #
# ⑦ 离线交付计划                                                                     #
# --------------------------------------------------------------------------------- #
PINNED_LADDER = [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35]


def _feasible_certificate(cap=1.25):
    return {"verdict": "feasible", "max_playback_ratio": cap, "requested_ratio": 1.0}


def test_plan_saves_every_slowdown_bake_on_the_pinned_ladder():
    plan = pbg.plan_offline_speed_ladder(PINNED_LADDER, _feasible_certificate())
    delivery = {row["ratio"]: row["delivery"] for row in plan["ladder"]}
    assert delivery[0.65] == "runtime_playback"
    assert delivery[0.8] == "runtime_playback"
    assert delivery[0.9] == "runtime_playback"
    assert delivery[1.0] == "native_source"
    assert delivery[1.1] == "bake_required"
    assert delivery[1.2] == "bake_required"
    assert delivery[1.35] == "bake_required"
    # Franco 钉死的 7 档梯子 = 6 个非原生变体;门通过后只剩 3 个真要烤。
    assert plan["bakes_without_this_gate"] == 6
    assert plan["bakes_required"] == 3
    assert plan["bakes_saved"] == 3


def test_plan_without_certificate_fails_closed():
    plan = pbg.plan_offline_speed_ladder(PINNED_LADDER, None)
    assert plan["bakes_saved"] == 0
    assert plan["bakes_required"] == 6
    for row in plan["ladder"]:
        if row["ratio"] < 1.0:
            assert row["delivery"] == "bake_required_playback_unauthorised"


def test_plan_respects_the_certified_ceiling():
    """证书上限 0.85 ⇒ 0.9 这档没被授权,不能算进省下的烤入。"""
    plan = pbg.plan_offline_speed_ladder(PINNED_LADDER, _feasible_certificate(cap=0.85))
    delivery = {row["ratio"]: row["delivery"] for row in plan["ladder"]}
    assert delivery[0.8] == "runtime_playback"
    assert delivery[0.9] == "bake_required_playback_unauthorised"
    assert plan["bakes_saved"] == 2


def test_plan_never_lets_the_clock_go_above_native():
    """r>1 一律必须烤:round() 时钟在 s>1 会跳过资产帧,可能跳掉触球帧本身。"""
    plan = pbg.plan_offline_speed_ladder([1.05], _feasible_certificate(cap=3.0))
    assert plan["ladder"][0]["delivery"] == "bake_required"
    assert pbg.RUNTIME_PLAYBACK_MAX_RATIO == 1.0


def test_infeasible_certificate_does_not_authorise_playback():
    certificate = {"verdict": "infeasible", "max_playback_ratio": 0.0}
    plan = pbg.plan_offline_speed_ladder([0.8], certificate)
    assert plan["ladder"][0]["delivery"] == "bake_required_playback_unauthorised"
    assert plan["bakes_saved"] == 0


# --------------------------------------------------------------------------------- #
# ⑧ MUTATION CHECK                                                                   #
# --------------------------------------------------------------------------------- #
def test_mutation_endpoints_only_scan_is_caught():
    """突变体:把区间极值扫描改成"只查端点"(去掉内部顶点)。

    突变体会把顶点用例判成 PASS,真实实现判成 FAIL —— 说明 ④ 的断言确实盯着这行逻辑,
    不是碰巧通过的。
    """
    c0, c1, c2, lower, upper = _vertex_case()

    def mutant_extrema(c0, c1, c2, lo, hi):
        f_lo = c0 + c1 * lo + c2 * lo * lo
        f_hi = c0 + c1 * hi + c2 * hi * hi
        return np.minimum(f_lo, f_hi), np.maximum(f_lo, f_hi)

    mutant_min, mutant_max = mutant_extrema(c0, c1, c2, 0.0, 1.0)
    mutant_util = np.maximum(mutant_max / upper, mutant_min / lower)
    assert mutant_util.max() <= 1.0, "mutant must look clean — that is the whole point"

    real_min, real_max, _, _ = pbg.quadratic_interval_extrema(c0, c1, c2, 0.0, 1.0)
    real_util = np.maximum(real_max / upper, real_min / lower)
    assert real_util.max() > 1.0, "shipped gate must catch what the mutant misses"


def test_mutation_gravity_scaled_with_s_squared_is_caught():
    """突变体:假设整条 tau 都随 s² 缩放(即把重力也当成随 s² 缩)。

    这是最诱人的错误 —— "慢放当然更安全"。突变体在极慢比下把一条重力就超限的 clip
    判成可行;真实实现给 max_playback_ratio == 0。
    """
    plant = default_plant()
    plant.gravity = np.array([300.0, 10.0, 10.0, 10.0])
    inputs = make_torque_inputs(plant, limit=200.0)
    ratio = 0.05

    # 突变体:tau_mut(s) = s² · tau(1)
    native = np.array([
        plant(inputs.qpos[frame], inputs.qvel[frame], inputs.qacc[frame])
        for frame in range(inputs.qpos.shape[0])
    ])
    mutant_util = np.max(np.abs(native) * ratio * ratio / 200.0)
    assert mutant_util <= 1.0, "mutant must look clean at a very slow ratio"

    screen = pbg.torque_playback_screen(inputs, ratio)
    assert screen["pass"] is False
    assert screen["max_ratio"] == 0.0


def test_mutation_plan_treating_uncertified_as_covered_is_caught():
    """突变体:没有证书也把 r<1 当成慢放覆盖(乐观默认)。"""

    def mutant_plan(ladder, certificate):
        return sum(1 for r in ladder if r < 1.0)   # 无视证书

    assert mutant_plan(PINNED_LADDER, None) == 3
    real = pbg.plan_offline_speed_ladder(PINNED_LADDER, None)
    assert real["bakes_saved"] == 0


# --------------------------------------------------------------------------------- #
# CLI(plan 模式,纯 CPU)                                                            #
# --------------------------------------------------------------------------------- #
def test_cli_plan_mode_writes_a_plan(tmp_path):
    certificate_path = tmp_path / "cert.json"
    certificate_path.write_text(json.dumps(_feasible_certificate()), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    code = pbg.main([
        "--plan", "--certificate", str(certificate_path),
        "--ladder", "0.65,0.8,1.0,1.2", "--plan-out", str(plan_path),
    ])
    assert code == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["bakes_saved"] == 2
    assert plan["bakes_required"] == 1


def test_cli_refuses_to_overwrite_a_plan(tmp_path):
    certificate_path = tmp_path / "cert.json"
    certificate_path.write_text(json.dumps(_feasible_certificate()), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        pbg.main(["--plan", "--certificate", str(certificate_path),
                  "--plan-out", str(plan_path)])
    assert plan_path.read_text(encoding="utf-8") == "occupied"


def test_cli_requires_exactly_one_mode():
    with pytest.raises(SystemExit, match="exactly one"):
        pbg.main(["--certify", "--plan"])


# --------------------------------------------------------------------------------- #
# CLI(certify 模式,真的读一份 clip npz;host 没有 mujoco ⇒ 只跑运动学筛)             #
# --------------------------------------------------------------------------------- #
def _synthetic_clip(path, joint_vel_peak):
    v1 = _load("synthesize_timing")
    joints = len(v1.ISAAC_JOINT_NAMES)
    joint_vel = np.zeros((4, joints), dtype=np.float32)
    joint_vel[:, 24] = joint_vel_peak
    np.savez(path, fps=np.array([50], dtype=np.int64), joint_vel=joint_vel)
    return path


def test_cli_certify_on_a_real_clip_file_is_kinematic_only(tmp_path):
    """"actual clip" 路径真的通:读 npz + URDF 限位 + 实证包络,给出半证(不可发货)。"""
    clip = _synthetic_clip(tmp_path / "clip.npz", 0.5)
    budget = _synthetic_clip(tmp_path / "budget.npz", 20.0)
    certificate_path = tmp_path / "cert.json"
    code = pbg.main([
        "--certify", "--clip", str(clip), "--ratio", "0.8",
        "--budget-clips", str(budget), "--certificate", str(certificate_path),
    ])
    assert code == pbg.EXIT_INCOMPLETE                   # 半证不是通行证
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert certificate["verdict"] == "incomplete"        # 没有 --mjcf ⇒ 没跑力矩筛
    assert certificate["admissible"] is False
    assert certificate["claim_scope"] == "kinematic_only_NOT_shippable"
    assert certificate["clip"]["sha256"]
    assert certificate["screens"]["kinematic_envelope"]["ran"] is True


def test_cli_certify_reports_an_out_of_envelope_ratio_with_exit_code_three(tmp_path):
    v1 = _load("synthesize_timing")
    from audit_motion_npz import parse_urdf_limits

    urdf = _SCRIPTS.parents[2] / (
        "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf")
    cap = parse_urdf_limits(str(urdf))[v1.ISAAC_JOINT_NAMES[24]].velocity
    clip = _synthetic_clip(tmp_path / "clip.npz", 0.95 * cap * 0.85)  # r=1 已贴着上限
    budget = _synthetic_clip(tmp_path / "budget.npz", 20.0)
    certificate_path = tmp_path / "cert.json"
    code = pbg.main([
        "--certify", "--clip", str(clip), "--ratio", "1.3",
        "--budget-clips", str(budget), "--certificate", str(certificate_path),
    ])
    assert code == pbg.EXIT_INFEASIBLE
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert certificate["screens"]["kinematic_envelope"]["pass"] is False
    assert certificate["max_playback_ratio"] < 1.3
