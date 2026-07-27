"""拍面符号 / 瞄准角的"旋转等变"性质回归测试 —— 2026-07-27。

人话:一段挥拍录像,不管你把整个人绕世界 Z 轴转多少度,"用红面还是黑面击球"这件事都不该变;
而"这一拍瞄向哪里"应该跟着转多少就偏多少,不能多转也不能少转,更不能在 ±180 边界上跳一下。

WHY THIS FILE EXISTS(定这条性质的由头)
    2026-07-26 的动作库里,五段 clip 解出来的瞄准角 psi* 有一段(fh_loop)看着像整整偏了半圈:
        bh_block +38.4 / s0_highpress +48.3 / fh_block_syn +60.3 / bh_loop_c +74.5 / fh_loop -128.1
    当时的三个怀疑对象是:(1) atan2 分支/取模符号约定在负角那侧翻了;(2) 重定向把腕滚翻了 180;
    (3) mount_normal_sign 被重复施加。三个都用受控实验证伪了(见下面 test 里钉住的性质),
    真正的原因是这一批里 fh_loop 是**唯一的正手**,另外四段(含名字叫 fh_block_syn 但
    BINDINGS 里 clip_family=backhand 的那段)全是反手 —— 正反手本来就从身体两侧出拍。

    但排查过程中暴露的两类真缺陷是**代码**问题,这个文件就是拿来钉住它们的类别的:
      A. 求解器必须对世界 Z 旋转严格等变(不许有 wrap/分支伪影);
      B. 触球窗口必须和 clip 同源、且落在 clip 帧数以内(当时脚本里硬编码的窗口取自另一个
         build,对 T=88 的 clip 用了 (60, 92),既错帧段又越界)。

依赖:只用 numpy(suggest_face_sign 自称 numpy + PyYAML,不碰 isaac/torch/mujoco)。
注意 host 上 pytest 是 py3.8,不要用 zip(strict=) / math.ulp 之类的新接口。
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "hope_training", "whole_body_tracking", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

suggest_face_sign = pytest.importorskip(
    "suggest_face_sign", reason="需要 hope_training scripts 可导入(numpy-only 路径)"
)

N_BODIES = 32
WRIST_INDEX = 31  # analyze_strike_phase.RACKET_BODY,32 体 npz 的历史约定列
FPS = 50


def _rot_z(deg):
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_to_quat_wxyz(R):
    """旋转矩阵 -> wxyz 四元数(数值稳健分支,和 analyze_strike_phase.quat_to_rot 互逆)。"""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _axis_angle(axis, deg):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    th = np.deg2rad(float(deg))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _synth_stroke(T=41, contact=20, sweep_deg=110.0, roll_extra=None):
    """合成一段"挥拍":腕坐标系绕世界 Z 匀速扫过 sweep_deg,拍心沿一条向前上方的弧线走。

    这是一段**已知答案**的输入 —— 触球帧的拍面法向、拍速方向都是解析可算的,
    所以任何求解器在它上面的行为都可以被断言,而不是被观察。
    """
    P = np.zeros((T, N_BODIES, 3), dtype=np.float64)
    Q = np.zeros((T, N_BODIES, 4), dtype=np.float64)
    Q[:, :, 0] = 1.0
    t = np.arange(T, dtype=np.float64)
    # 腕姿态:基准姿态(拍面 +Y 朝世界 +X)再绕世界 Z 扫过 sweep_deg
    base = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    for k in range(T):
        frac = (k - contact) / float(max(T - 1, 1))
        R = _rot_z(sweep_deg * frac) @ base
        if roll_extra is not None:
            axis, deg = roll_extra
            R = R @ _axis_angle(axis, deg)
        Q[k, WRIST_INDEX] = _rot_to_quat_wxyz(R)
        # 拍腕位置:向前(+X)推进,同时抬高 —— 保证有限差分拿得到非退化速度
        P[k, WRIST_INDEX] = np.array([0.30 + 0.020 * k, -0.05 + 0.004 * k, 0.80 + 0.012 * k])
    return P, Q


def _write_npz(tmpdir, name, P, Q, with_stored_vel=True):
    # 夹具一律用 float64 存:这里要钉的是**代码的算术**(有没有 wrap/分支伪影),
    # 不是资产的存储精度。真实资产是 float32,round-trip 本身就有 ~1e-6 度的抖动,
    # 用 float32 存会把容差撑到看不出 quadrant 级别以下的问题。
    path = os.path.join(str(tmpdir), name)
    payload = dict(
        body_pos_w=P.astype(np.float64),
        body_quat_w=Q.astype(np.float64),
        fps=np.array([FPS], dtype=np.int64),
    )
    if with_stored_vel:
        v = np.zeros_like(P)
        v[1:-1] = (P[2:] - P[:-2]) * (FPS / 2.0)
        v[0], v[-1] = v[1], v[-2]
        payload["body_lin_vel_w"] = v.astype(np.float64)
        payload["body_ang_vel_w"] = np.zeros_like(P, dtype=np.float64)
    np.savez(path, **payload)
    return path


def _rotate_clip_world_z(P, Q, deg):
    """把整段 clip 绕世界 Z 转 deg —— 位置和姿态一起转,物理上是"同一个动作换个朝向"。"""
    R = _rot_z(deg)
    P2 = P @ R.T
    Q2 = Q.copy()
    for k in range(P.shape[0]):
        Rw = suggest_face_sign.quat_to_rot(Q[k, WRIST_INDEX][None, :])[0]
        Q2[k, WRIST_INDEX] = _rot_to_quat_wxyz(R @ Rw)
    return P2, Q2


# --------------------------------------------------------------------------------------
# A. 旋转等变 / 不变性 —— 钉住"atan2 分支、%/fmod 符号、wrap 区间"这一整类缺陷
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [-180.0, -147.0, -90.0, -30.0, 0.0, 30.0, 90.0, 147.0, 180.0])
def test_face_sign_is_invariant_under_world_z_rotation(tmpdir, delta):
    """击球面符号是**物理属性**,不是朝向属性:整段绕 Z 转任意角,sign(n·v) 必须一字不变。

    这是三个怀疑对象里第 (1) 个的判决性实验 —— 如果 atan2/取模/wrap 在负角那侧有分支伪影,
    delta=-147(fh_loop 源站位的机位角)这一格就会翻符号。
    """
    contact = 20
    P, Q = _synth_stroke(contact=contact)
    ref = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "ref.npz", P, Q), contact)
    P2, Q2 = _rotate_clip_world_z(P, Q, delta)
    got = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "rot.npz", P2, Q2), contact)

    assert got["suggested_sign"] == ref["suggested_sign"], (
        "绕世界 Z 转 %+.1f 度后击球面符号变了:%+.0f -> %+.0f。"
        "符号只能由「哪面超前」决定,不能由朝向决定。"
        % (delta, ref["suggested_sign"], got["suggested_sign"])
    )
    assert got["cos_clean"] == pytest.approx(ref["cos_clean"], abs=1e-9), (
        "cos(n, v) 不是旋转不变量了(delta=%+.1f):%.12f -> %.12f" % (delta, ref["cos_clean"], got["cos_clean"])
    )


def test_face_normal_azimuth_is_exactly_equivariant(tmpdir):
    """瞄准量必须**严格等变**:整段转 delta,拍面法向方位角就该正好挪 delta,不多不少。

    残差必须精确为 0(数值容差内),不能在 ±180 边界上多跳一圈 —— 这正是当初怀疑
    "psi* 差了整整 180"时要排除的伪影。
    """
    contact = 20
    P, Q = _synth_stroke(contact=contact)
    ref = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "ref.npz", P, Q), contact)
    az0 = np.rad2deg(np.arctan2(ref["normal_w"][1], ref["normal_w"][0]))
    for delta in (-180.0, -147.0, -90.0, -30.0, 30.0, 90.0, 147.0, 180.0):
        P2, Q2 = _rotate_clip_world_z(P, Q, delta)
        got = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "rot.npz", P2, Q2), contact)
        az1 = np.rad2deg(np.arctan2(got["normal_w"][1], got["normal_w"][0]))
        resid = (az1 - (az0 + delta) + 180.0) % 360.0 - 180.0
        assert abs(resid) < 1e-6, (
            "拍面方位角不等变:delta=%+.1f,期望 %+.4f,实得 %+.4f,残差 %+.4f 度。"
            % (delta, az0 + delta, az1, resid)
        )


# --------------------------------------------------------------------------------------
# B. 腕滚 180 的补偿恒等式 —— 解释"fh_block_syn 明明 sign=+1 却不偏"这件事
# --------------------------------------------------------------------------------------

def test_180_roll_about_forearm_flips_the_face_and_therefore_the_sign(tmpdir):
    """绕前臂轴(腕坐标系局部 X)滚 180:拍面法向精确反向,所以建议符号必须跟着翻。

    动作库里 fh_block_syn 就是这个情况 —— 它的 right_wrist_roll 在触球帧是 -149.7 度,
    另外四段是 +18.3 / +31.2 / +39.9 / +40.3。它被烘进了一个 ~180 的腕滚,又配了
    mount_normal_sign=+1(其余反手是 -1),两次翻转正好抵消,physical_B 是对的。
    这条测试把"两次翻转互相抵消"钉成显式恒等式,免得以后有人只改其中一边。
    """
    contact = 20
    P, Q = _synth_stroke(contact=contact)
    ref = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "ref.npz", P, Q), contact)

    P_r, Q_r = _synth_stroke(contact=contact, roll_extra=(np.array([1.0, 0.0, 0.0]), 180.0))
    got = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "roll.npz", P_r, Q_r), contact)

    n0 = np.asarray(ref["normal_w"], dtype=np.float64)
    n1 = np.asarray(got["normal_w"], dtype=np.float64)
    assert np.allclose(n1, -n0, atol=1e-9), "绕前臂轴滚 180 后拍面法向不是精确反向:%s vs %s" % (n1, n0)
    assert got["suggested_sign"] == -ref["suggested_sign"], (
        "拍面翻了但建议符号没翻 —— physical_B 会被算成朝后。%+.0f -> %+.0f"
        % (ref["suggested_sign"], got["suggested_sign"])
    )


def test_180_roll_about_the_face_axis_is_a_no_op_on_the_face(tmpdir):
    """绕拍面法向自身(局部 Y)滚 180:法向不动,符号也不该动。

    对照组 —— 证明上一条抓到的是"翻面",不是"只要动腕就翻符号"。
    """
    contact = 20
    P, Q = _synth_stroke(contact=contact)
    ref = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "ref.npz", P, Q), contact)
    P_r, Q_r = _synth_stroke(contact=contact, roll_extra=(np.array([0.0, 1.0, 0.0]), 180.0))
    got = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "faceroll.npz", P_r, Q_r), contact)

    assert np.allclose(np.asarray(got["normal_w"]), np.asarray(ref["normal_w"]), atol=1e-9)
    assert got["suggested_sign"] == ref["suggested_sign"]


# --------------------------------------------------------------------------------------
# C. 触球窗口必须和 clip 同源 —— 钉住"窗口取自另一个 build / 越界"这类缺陷
# --------------------------------------------------------------------------------------

def test_contact_frame_outside_the_clip_fails_loud(tmpdir):
    """越界的触球帧必须当场报错,不许静默截断。

    由头:瞄准脚本里硬编码 WIN["fh_loop"] = (60, 92),而那段 clip 只有 T=88 帧 ——
    窗口既取自另一个 build(upper_safe/p8b 的 [60,92],而 clip 是 upper_fast/p8f 的 [34,50]),
    又超出末帧 4 帧。任何按秒 x fps 折算出来的窗口都必须先过这道边界检查。
    """
    P, Q = _synth_stroke(T=41)
    path = _write_npz(tmpdir, "short.npz", P, Q)
    for bad in (41, 60, 92, -1):
        with pytest.raises(ValueError):
            suggest_face_sign.compute_face_sign(path, bad)


def test_ambiguous_face_is_reported_not_silently_signed(tmpdir):
    """拍面和拍速接近垂直时,sign(n·v) 没有方向意义,必须被标成 ambiguous 让人复核。

    fh_loop 在它自己申报的触球帧 41 上 B.x = -0.022(拍面几乎正好侧对球台),
    就是这种"贴边"体质 —— 工具必须把它报出来,而不是给个看着挺确定的 +1。
    """
    contact = 20
    # 让拍面法向与拍速接近正交:拍心几乎只沿 +X 走,而拍面朝 +Y
    T = 41
    P = np.zeros((T, N_BODIES, 3), dtype=np.float64)
    Q = np.zeros((T, N_BODIES, 4), dtype=np.float64)
    Q[:, :, 0] = 1.0
    base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    for k in range(T):
        Q[k, WRIST_INDEX] = _rot_to_quat_wxyz(base)
        P[k, WRIST_INDEX] = np.array([0.30 + 0.02 * k, 0.0, 0.90])
    res = suggest_face_sign.compute_face_sign(_write_npz(tmpdir, "amb.npz", P, Q), contact)
    assert res["ambiguous"] is True, (
        "拍面几乎垂直于拍速(|cos|=%.4f)却没被标 ambiguous" % abs(res["cos_clean"])
    )
