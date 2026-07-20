"""Unit tests for the SIGNED stroke-scale upgrade of scripts/extend_stroke.py(轴 C).

人话:引拍长度双向缩放 —— --stroke-scale S,S<1 缩短、S>1 加深、S=1 恒等直拷。
历史缺陷(本文件的回归靶):负 --extend-frac 时 solve_amplitude 只在 [0, A_max] 单边
二分且方向固定为加深梯度,负目标落在括号外 ⇒ 二分静默收敛到 A≈0,输出一张几乎恒等的片。
现在缩短方向整体取 -w、幅度解算先网格找穿越括号再二分,并且 manifest 的比例对账
(|ratio-scale| ≤ 5%·scale)把"静默恒等片"直接判死。

Pure CPU, NO mujoco:与 test_extend_stroke.py 同一套解析 stub FK(平面二连杆)。

Covered:
  * CLI: --extend-frac/--stroke-scale 二选一 fail-closed;非法 scale(0/负/NaN)拒绝
  * scale=1 恒等:逐字节直拷 + manifest ratio=1 + bucket=train + sha256 双端一致
  * 缩短确实缩短:L 比例 < 1、逐 scale 单调、方向 = 加深梯度的反向
  * 触球锚:触球行/锁窗/ready/随挥逐位;|v*| 精确不变
  * 冻结:非选中关节 + 双腿逐位
  * 限位:grandfather 盒约束、缩短侧零余量关节 blocked、目标不可达 fail-loud
  * 分桶合同写死:train 0.80/1.00/1.20 · interpolation 0.90/1.10 · OOD 0.65/1.35
  * 速度元数据:v_max(L)=κ·√(v_start²+2·a_max·L) 数值 + 非法输入 fail-closed
  * manifest fail-closed:硬不变式为 False、比例对不上(静默恒等片形态)都拒绝出档

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_stroke_scale_signed.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "extend_stroke"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
es = sys.modules["extend_stroke"]
st = sys.modules["synthesize_timing"]

FPS = 50.0
J = 31
NB = 32
NAMES = es.ISAAC_JOINT_NAMES

C_WAIST_YAW = NAMES.index("waist_yaw_joint")
C_SH_PITCH = NAMES.index("right_shoulder_pitch_joint")
C_ELBOW = NAMES.index("right_elbow_joint")
C_WAIST_PITCH = NAMES.index("waist_pitch_joint")

L1, L2 = 0.30, 0.25


def stub_blade(q: np.ndarray, frames=None) -> np.ndarray:
    """Planar 2-link arm tilted by the waist pitch and swung about z by the waist yaw —
    the exact closed form of the legacy extend_stroke tests (keeps waist_pitch leverage
    nonzero so the pinned-joint test exercises the BLOCKED path)."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    yaw, a, b, pit = (q[:, C_WAIST_YAW], q[:, C_SH_PITCH], q[:, C_ELBOW], q[:, C_WAIST_PITCH])
    x = L1 * np.cos(a) + L2 * np.cos(a + b)
    z = L1 * np.sin(a) + L2 * np.sin(a + b)
    xp = x * np.cos(pit) + z * np.sin(pit)
    zp = -x * np.sin(pit) + z * np.cos(pit)
    return np.stack([xp * np.cos(yaw), xp * np.sin(yaw), zp], axis=1)


class LimitStub:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper
        self.effort, self.velocity = 100.0, 10.0


def limits(overrides: dict | None = None) -> dict:
    lim = {n: LimitStub(-3.0, 3.0) for n in NAMES}
    for n, (lo, hi) in (overrides or {}).items():
        lim[n] = LimitStub(lo, hi)
    return lim


def make_clip(T=60, contact=23, deep=13):
    """Backhand-shaped stub: the arm swings back to `deep` then forward through `contact`."""
    q = np.zeros((T, J), dtype=np.float32)
    s = np.arange(T, dtype=np.float64)
    q[:, C_SH_PITCH] = (-0.9 * np.sin(np.pi * np.clip(s / (2 * deep), 0, 1)) + 0.6 * s / T)
    q[:, C_ELBOW] = 0.55 + 0.35 * np.cos(np.pi * s / T)
    q[:, C_WAIST_YAW] = 0.25 * np.sin(2 * np.pi * s / (3 * T))
    # pinned flat (v5 反手 waist_pitch 形态:retarget 钳死,两个方向都零余量)
    q[:, C_WAIST_PITCH] = 0.30
    dq = np.gradient(q.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    bp = np.zeros((T, NB, 3), dtype=np.float32)
    bp[:, st.RACKET_BODY] = stub_blade(q.astype(np.float64)).astype(np.float32)
    bq = np.zeros((T, NB, 4), dtype=np.float32)
    bq[..., 0] = 1.0
    bl = np.gradient(bp.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    data = {"fps": np.array([int(FPS)], dtype=np.int64), "joint_pos": q, "joint_vel": dq,
            "body_pos_w": bp, "body_quat_w": bq, "body_lin_vel_w": bl,
            "body_ang_vel_w": np.zeros_like(bp)}
    return data, contact / (T - 1)


ARM = ["right_shoulder_pitch_joint", "right_elbow_joint", "waist_yaw_joint"]


def run(data, phase, joints=ARM, frac=-0.20, lim=None, **kw):
    return es.morph(data, phase, list(joints), frac, stub_blade, lim or limits(), **kw)


# ------------------------------------------------------------------ CLI fail-closed -- #
def test_cli_requires_exactly_one_of_frac_or_scale(tmp_path):
    base = ["--input", "x.npz", "--output", "y.npz", "--phase", "0.4",
            "--mjcf", "m.xml", "--body-order", "b.txt"]
    with pytest.raises(SystemExit, match="exactly one"):
        es.main(base)                                              # neither
    with pytest.raises(SystemExit, match="exactly one"):
        es.main(base + ["--extend-frac", "0.2", "--stroke-scale", "1.2"])   # both


@pytest.mark.parametrize("bad", ["0", "-0.5", "nan"])
def test_cli_rejects_nonpositive_or_nonfinite_scale(bad):
    base = ["--input", "x.npz", "--output", "y.npz", "--phase", "0.4",
            "--mjcf", "m.xml", "--body-order", "b.txt", "--stroke-scale", bad]
    with pytest.raises(SystemExit, match="finite and > 0"):
        es.main(base)


def test_morph_rejects_scale_at_or_below_zero():
    data, phase = make_clip()
    with pytest.raises(SystemExit, match="finite and > 0"):
        run(data, phase, frac=-1.0)          # scale = 0
    with pytest.raises(SystemExit, match="finite and > 0"):
        run(data, phase, frac=-1.5)          # scale < 0


# ------------------------------------------------------------------ scale=1 identity - #
def test_scale_one_is_bytewise_identity_with_manifest(tmp_path):
    data, phase = make_clip()
    src = tmp_path / "src.npz"
    out = tmp_path / "out_s100.npz"
    np.savez(src, **data)
    rc = es.main(["--input", str(src), "--output", str(out), "--phase", str(phase),
                  "--mjcf", "unused.xml", "--body-order", "unused.txt",
                  "--stroke-scale", "1.0"])
    assert rc == 0
    assert src.read_bytes() == out.read_bytes()                    # 逐字节恒等
    man = json.loads((tmp_path / "out_s100.npz.manifest.json").read_text())
    assert man["scale"]["requested"] == 1.0
    assert man["scale"]["ratio_measured_stored"] == 1.0
    assert man["bucket"] == "train"
    assert man["source"]["sha256"] == man["output"]["sha256"]      # 内容寻址:双端同 SHA
    assert len(man["source"]["sha256"]) == 64
    assert man["contact_invariance"]["contact_row_bitwise"] is True
    # 模式二推导口径:v_start=0 且 a_max=v*²/(2L) ⇒ v_max = κ·v*
    m2 = man["speed_modes"]["fixed_accel_utilization"]
    v_star = man["contact_invariance"]["v_star_src_mps"]
    assert m2["kappa"] == pytest.approx(0.8)
    assert m2["v_max_suggested_mps"] == pytest.approx(0.8 * v_star, rel=1e-3)
    # 模式一:v* 钉死,a_min = v*²/(2L)
    m1 = man["speed_modes"]["fixed_strike_velocity"]
    assert m1["a_min_mps2"] == pytest.approx(v_star ** 2 / (2 * m1["L_m"]), rel=1e-3)


# ------------------------------------------------------------------ shortening works - #
def test_shorten_actually_shortens_to_target():
    data, phase = make_clip()
    _, _, info, _ = run(data, phase, frac=-0.20)
    assert info["direction"] == "shorten"
    assert info["stroke_scale_out"] == pytest.approx(0.80, rel=2e-3)
    assert info["L_deep_out_fk"] < info["L_deep_src_fk"]


def test_scale_ratio_monotonic_across_the_seven_point_grid():
    """0.65 → 1.35 的实测 L/L0 必须严格单调升、逐点命中,且缩短侧 < 1 < 加深侧。

    缩短剂量按固定段(峰帧→触球)弧长比度量 —— 账本口径(argmax 最深帧)在引拍压平时
    身份跳变、L(A) 不连续,目标落进缺口会收敛到缺口边缘;固定段连续,逐点应精确命中。"""
    data, phase = make_clip()
    ratios = []
    for scale in (0.65, 0.80, 0.90, 1.10, 1.20, 1.35):
        _, _, info, _ = run(data, phase, frac=scale - 1.0)
        ratios.append(info["stroke_scale_out"])
        assert info["stroke_scale_out"] == pytest.approx(scale, rel=2e-3)
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    assert all(r < 1.0 for r in ratios[:3]) and all(r > 1.0 for r in ratios[3:])


def test_shorten_direction_is_negated_deepening_gradient():
    data, phase = make_clip()
    _, plan_s, info, _ = run(data, phase, frac=-0.25, refine_iters=0)
    # weights point OPPOSITE the deepening gradient, per joint and in aggregate
    assert float(np.dot(plan_s.weights, plan_s.grad)) < 0.0
    for wi, gi in zip(plan_s.weights, plan_s.grad):
        assert wi * gi <= 0.0
    assert max(abs(w) for w in plan_s.weights) == pytest.approx(1.0)


# ------------------------------------------------------------- contact anchor safety - #
def test_shorten_contact_row_lock_window_and_follow_through_bitwise():
    data, phase = make_clip()
    q_out, _, info, _ = run(data, phase, frac=-0.30)
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    c, s1 = info["contact_frame"], info["s1"]
    assert np.array_equal(q_out[c], q_src[c])              # contact anchor row
    assert np.array_equal(q_out[s1:], q_src[s1:])          # lock window (含随挥到片尾)
    assert np.array_equal(q_out[c:], q_src[c:])            # follow-through safe
    assert np.array_equal(q_out[0], q_src[0])              # ready pose held


def test_shorten_v_star_preserved_exactly():
    data, phase = make_clip()
    q_out, _, info, blade_src = run(data, phase, frac=-0.30)
    c = info["contact_frame"]
    v0 = st.clean_speed_at(blade_src, c, 1.0 / FPS)
    v1 = st.clean_speed_at(stub_blade(q_out), c, 1.0 / FPS)
    assert v1 == pytest.approx(v0, abs=1e-12)


def test_shorten_freezes_legs_and_non_selected_joints_bitwise():
    data, phase = make_clip()
    q_out, plan, info, _ = run(data, phase, frac=-0.25)
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    leg_cols = [NAMES.index(n) for n in es.LEG_JOINTS]
    assert not (set(int(c) for c in plan.cols) & set(leg_cols))    # legs never selected
    assert np.array_equal(q_out[:, leg_cols], q_src[:, leg_cols])  # …and bitwise frozen
    frozen = np.setdiff1d(np.arange(J), plan.cols)
    assert np.array_equal(q_out[:, frozen], q_src[:, frozen])
    assert info["frozen_joints_bitwise"]


# ----------------------------------------------------------------- limits fail-closed - #
def test_shorten_respects_grandfathered_limits():
    data, phase = make_clip()
    lo, hi = -0.35, 0.90
    q_out, _, _, _ = run(data, phase, frac=-0.20,
                         lim=limits({"waist_yaw_joint": (lo, hi)}))
    col = NAMES.index("waist_yaw_joint")
    src = np.asarray(data["joint_pos"], dtype=np.float64)[:, col]
    eff_lo, eff_hi = min(lo, src.min()), max(hi, src.max())
    assert q_out[:, col].min() >= eff_lo - 1e-7
    assert q_out[:, col].max() <= eff_hi + 1e-7


def test_shorten_blocked_joint_dropped_with_shortening_message(capsys):
    """缩短方向零余量的关节 = blocked,剔除 + WARN,消息用 shortening 措辞。"""
    data, phase = make_clip()
    v = float(np.asarray(data["joint_pos"], dtype=np.float64)[:, C_WAIST_PITCH].max())
    _, plan, info, _ = run(data, phase, joints=ARM + ["waist_pitch_joint"], frac=-0.20,
                           lim=limits({"waist_pitch_joint": (v, v)}), on_blocked="drop")
    assert "waist_pitch_joint" in info["joints_blocked"]
    assert C_WAIST_PITCH not in list(plan.cols)
    assert "ZERO shortening headroom" in capsys.readouterr().err
    assert info["stroke_scale_out"] == pytest.approx(0.80, rel=2e-3)


def test_shorten_unreachable_target_fails_loud():
    data, phase = make_clip()
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    tight = {n: (q[:, NAMES.index(n)].min() - 0.02, q[:, NAMES.index(n)].max() + 0.02)
             for n in ARM}
    with pytest.raises(SystemExit, match="unreachable"):
        run(data, phase, frac=-0.90, lim=limits(tight))    # scale 0.10 不可达


def test_deepen_regression_still_works_after_signing():
    data, phase = make_clip()
    _, plan, info, _ = run(data, phase, frac=0.20)
    assert info["direction"] == "deepen"
    assert info["stroke_scale_out"] == pytest.approx(1.20, rel=2e-3)
    assert float(np.dot(plan.weights, plan.grad)) > 0.0    # 加深方向不受签名改造影响


# ----------------------------------------------------------------- buckets + speeds --- #
def test_bucket_contract_is_hardcoded_and_complete():
    assert es.STROKE_SCALE_BUCKETS == {"train": (0.80, 1.00, 1.20),
                                       "interpolation": (0.90, 1.10),
                                       "OOD": (0.65, 1.35)}
    for s, b in ((0.80, "train"), (1.00, "train"), (1.20, "train"),
                 (0.90, "interpolation"), (1.10, "interpolation"),
                 (0.65, "OOD"), (1.35, "OOD")):
        assert es.bucket_of_scale(s) == b
    assert es.bucket_of_scale(0.77) == "unassigned"


def test_v_max_suggested_math_and_fail_closed():
    # κ·√(v0² + 2·a·L):0.8·√(1 + 2·10·0.5) = 0.8·√11
    assert es.v_max_suggested(0.8, 1.0, 10.0, 0.5) == pytest.approx(0.8 * np.sqrt(11.0))
    assert es.v_max_suggested(0.8, 0.0, 2.0, 1.0) == pytest.approx(1.6)
    with pytest.raises(SystemExit, match="finite"):
        es.v_max_suggested(0.8, 0.0, -1.0, 0.5)
    with pytest.raises(SystemExit, match="finite"):
        es.v_max_suggested(0.8, float("nan"), 1.0, 0.5)


def _manifest_kwargs(**over):
    kw = dict(scale=0.80, input_path="a.npz", output_path="b.npz",
              src_sha256="0" * 64, out_sha256="1" * 64, phase=0.39,
              contact_frame_idx=23, fps=50.0, frames=60,
              L_src_m=0.50, L_out_m=0.40, v_star_src_mps=2.2, v_star_out_mps=2.2,
              proofs=dict(contact_row_bitwise=True, lock_window_bitwise=True,
                          follow_through_bitwise=True, legs_bitwise=True,
                          ready_pose_bitwise=True, non_selected_joints_bitwise=True),
              kappa=0.8, v_start_mps=0.0, a_max_mps2=4.84, a_max_source="cli",
              ratio_fk=0.80)
    kw.update(over)
    return kw


def test_manifest_bakes_ratio_sha_and_bucket():
    man = es.build_scale_manifest(**_manifest_kwargs())
    assert man["scale"]["ratio_measured_stored"] == pytest.approx(0.8)
    assert man["bucket"] == "train"
    assert man["buckets_contract"] == {"train": [0.80, 1.00, 1.20],
                                       "interpolation": [0.90, 1.10],
                                       "OOD": [0.65, 1.35]}
    assert man["source"]["sha256"] == "0" * 64 and man["output"]["sha256"] == "1" * 64
    assert man["speed_modes"]["fixed_accel_utilization"]["v_max_suggested_mps"] > 0


def test_manifest_refuses_broken_invariant_and_silent_identity():
    bad = _manifest_kwargs()
    bad["proofs"] = dict(bad["proofs"], contact_row_bitwise=False)
    with pytest.raises(SystemExit, match="hard invariants violated"):
        es.build_scale_manifest(**bad)
    # 历史缺陷形态:请求 0.65 却实测 1.0(静默恒等片)→ 拒绝出档
    with pytest.raises(SystemExit, match="deviates"):
        es.build_scale_manifest(**_manifest_kwargs(scale=0.65, L_out_m=0.50, ratio_fk=1.0))
    with pytest.raises(SystemExit, match="missing hard invariants"):
        es.build_scale_manifest(**_manifest_kwargs(proofs={"contact_row_bitwise": True}))
    with pytest.raises(SystemExit, match="non-finite"):
        es.build_scale_manifest(**_manifest_kwargs(v_star_out_mps=float("nan")))
