"""CPU-only tests for the measured-clip open-loop executability audit.

不需要 mujoco:只测纯数值部分和那道 fail-closed 锚点。真 clip 的数在 pod 上跑,不进单测。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "audit_measured_teacher_executability",
    _SCRIPTS / "audit_measured_teacher_executability.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["audit_measured_teacher_executability"] = _MOD
_SPEC.loader.exec_module(_MOD)

ExecutabilityAuditError = _MOD.ExecutabilityAuditError


class _StiffnessOnlyPlant:
    """`_verify_hold_anchor` 只读 kp,所以锚点可以脱离 mujoco 单独测。"""

    def __init__(self, kp):
        self.kp = np.asarray(kp, np.float64)


def _artifact(kp, birth, hold, archived):
    return {
        "physical_ready": {"joint_pos_rad": list(birth)},
        "hold_candidate": {
            "hold_qdes_joint_pos_rad": list(hold),
            "actuator_generalized_force_runtime_order_nm": list(archived),
        },
    }


def test_derivative_recovers_a_known_slope_in_every_mode():
    # 二次曲线的导数是解析已知的,任何一档微分都必须在内点上还原它。
    dt = 0.02
    steps = np.arange(40, dtype=np.float64)
    values = (3.0 * (steps * dt) ** 2 + 1.5 * (steps * dt))[:, None]
    expected = 6.0 * (steps * dt) + 1.5
    for mode in _MOD.DIFFERENTIATION_MODES:
        got = _MOD._derivative(values, mode, dt)[:, 0]
        # 端点由平移窗口给出,内点必须准;raw 中心差分对二次曲线本来就是精确的。
        assert np.allclose(got[8:-8], expected[8:-8], atol=1.0e-6), mode


def test_unknown_differentiation_mode_is_refused():
    with pytest.raises(ExecutabilityAuditError):
        _MOD._derivative(np.zeros((10, 1)), "boxcar9", 0.02)


def test_window_longer_than_the_clip_is_refused():
    with pytest.raises(ExecutabilityAuditError):
        _MOD._derivative(np.zeros((4, 1)), "sg13_3", 0.02)


def test_support_margin_is_positive_inside_and_negative_outside():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hull = _MOD._convex_hull_2d(square)
    assert hull.shape[0] == 4
    inside = _MOD._hull_margin(np.array([0.5, 0.5]), hull)
    outside = _MOD._hull_margin(np.array([1.25, 0.5]), hull)
    assert inside == pytest.approx(0.5, abs=1.0e-9)
    assert outside == pytest.approx(-0.25, abs=1.0e-9)


def test_degenerate_support_polygon_is_never_reported_as_balanced():
    # 两点共线 = 没有支撑面。绝不能因为"算不出多边形"就默认站得住。
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert _MOD._hull_margin(np.array([0.5, 0.0]), _MOD._convex_hull_2d(line)) == float(
        "-inf"
    )


# --------------------------------------------------------------------------
# 支撑多边形:带宽锚在哪只脚上,以及"退化"必须被抓出来而不是被读成"质心出界"
#
# 造两只 A3 尺寸量级的矩形鞋底(长 0.25 m、宽 0.10 m、站宽 0.30 m),每只脚
# 上下两层顶点,这样"带宽捞到几层"是可控的。所有测试都不碰 mujoco。
# --------------------------------------------------------------------------


def _rect_sole(*, cx, cy, z, length=0.25, width=0.10, thickness=0.04, grid=6):
    """一只矩形鞋底的顶点:底面在 ``z``,顶面在 ``z + thickness``。"""

    xs = np.linspace(cx - length / 2.0, cx + length / 2.0, grid)
    ys = np.linspace(cy - width / 2.0, cy + width / 2.0, grid)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    bottom = np.column_stack([flat, np.full(flat.shape[0], z)])
    top = np.column_stack([flat, np.full(flat.shape[0], z + thickness)])
    return np.vstack([bottom, top])


def _two_feet(*, left_z, right_z):
    return [
        _rect_sole(cx=0.0, cy=+0.15, z=left_z),
        _rect_sole(cx=0.0, cy=-0.15, z=right_z),
    ]


def test_both_feet_flat_gives_the_full_double_support_polygon():
    feet = _two_feet(left_z=-0.0005, right_z=-0.0005)
    report = _MOD.support_polygon(feet, np.array([0.0, 0.0]))
    assert report["status"] == "DOUBLE_SUPPORT"
    assert report["foot_on_floor"] == [True, True]
    # 质心在正中间,离最近的边(前后 0.125 m)有 0.125 m。
    assert report["margin_m"] == pytest.approx(0.125, abs=1.0e-9)
    assert report["hull_area_m2"] == pytest.approx(0.25 * 0.40, abs=1.0e-9)


def _ramped_sole(*, cx, cy, z, ramp=0.005, length=0.25, width=0.10, grid=9):
    """一只**倾斜**的鞋底:底面沿横向从 ``z`` 线性升到 ``z + ramp``。

    真鞋底就是这样 —— 脚一歪,底面顶点的高度就散开好几毫米。带宽锚在哪里,
    决定了这只脚能贡献多大一块支撑面;两只脚都平放的假鞋底测不出这个 bug。
    """

    xs = np.linspace(cx - length / 2.0, cx + length / 2.0, grid)
    ys = np.linspace(cy - width / 2.0, cy + width / 2.0, grid)
    grid_x, grid_y = np.meshgrid(xs, ys)
    fraction = (grid_y - ys.min()) / (ys.max() - ys.min())
    bottom = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), (z + ramp * fraction).ravel()]
    )
    top = bottom + np.array([0.0, 0.0, 0.04])
    return np.vstack([bottom, top])


def _legacy_global_band_polygon(feet, com_xy, band_m=_MOD.SUPPORT_BAND_M):
    """被换掉的那条规则,原样重写一份,专门用来证明"新旧确实不一样"。"""

    lowest = min(float(np.asarray(foot)[:, 2].min()) for foot in feet)
    support = np.vstack(
        [np.asarray(foot)[np.asarray(foot)[:, 2] <= lowest + band_m] for foot in feet]
    )
    hull = _MOD._convex_hull_2d(support[:, :2])
    return hull, _MOD._hull_margin(np.asarray(com_xy), hull)


def test_the_band_is_anchored_per_foot_not_to_the_lower_foot():
    """本轮修掉的 bug 的正面复现,用**倾斜**鞋底。

    两只脚都踩在地上(都在 LP 窗口 `±2 mm` 里),左脚比右脚高 `3 mm`,
    两只脚各自的底面又都沿横向散开 `5 mm`。
    旧规则把 `6 mm` 带锚在**两只脚合起来**的最低点上,较高那只脚于是只剩
    `3 mm` 的带可用,半边支撑面被砍掉;新规则每只脚锚自己,两只脚各自完整。
    """

    level = [
        _ramped_sole(cx=0.0, cy=+0.15, z=-0.0015),
        _ramped_sole(cx=0.0, cy=-0.15, z=-0.0015),
    ]
    offset = [
        _ramped_sole(cx=0.0, cy=+0.15, z=-0.0015 + 0.0030),
        _ramped_sole(cx=0.0, cy=-0.15, z=-0.0015),
    ]
    # 质心放在偏左的位置,让**左侧那条边**成为 binding 边 —— 否则前后两条边先卡住,
    # 支撑面横向缩水多少都看不出来,这条测试就白写了。
    com = np.array([0.0, 0.10])
    fixed_level = _MOD.support_polygon(level, com)
    fixed_offset = _MOD.support_polygon(offset, com)
    _legacy_level_hull, legacy_level = _legacy_global_band_polygon(level, com)
    _legacy_offset_hull, legacy_offset = _legacy_global_band_polygon(offset, com)

    # 新规则:抬高 3 mm 不改变任何一只脚自己的支撑面,面积与裕度都不动。
    assert fixed_offset["status"] == "DOUBLE_SUPPORT"
    assert fixed_offset["foot_on_floor"] == [True, True]
    assert fixed_offset["hull_area_m2"] == pytest.approx(
        fixed_level["hull_area_m2"], abs=1.0e-12
    )
    assert fixed_offset["margin_m"] == pytest.approx(
        fixed_level["margin_m"], abs=1.0e-12
    )
    # 旧规则:同一个 3 mm 让支撑面缩水,裕度跟着变 —— 变的是量具,不是机器人。
    assert legacy_offset < legacy_level - 1.0e-4
    assert fixed_offset["margin_m"] > legacy_offset + 1.0e-4


def test_mutation_lifting_one_foot_off_the_floor_removes_it_from_the_polygon():
    """把左脚抬到 LP 窗口之外,机器人别的地方一个数不动。

    旧规则会拿它最低那一层顶点继续建多边形(带宽 `6 mm` > 抬高量时),
    于是一只**悬空**的脚照样贡献支撑面。新规则必须把它踢出去,并把状态
    降成单脚支撑。
    """

    grounded = _MOD.support_polygon(
        _two_feet(left_z=-0.0005, right_z=-0.0005), np.zeros(2)
    )
    lifted = _MOD.support_polygon(
        _two_feet(left_z=0.0030, right_z=-0.0005), np.zeros(2)
    )
    assert grounded["status"] == "DOUBLE_SUPPORT"
    assert lifted["status"] == "SINGLE_FOOT_SUPPORT"
    assert lifted["foot_on_floor"] == [False, True]
    assert lifted["contributing_feet"] == [1]
    # 支撑面缩到只剩右脚,质心(在两脚正中)因此落到支撑面外 —— 这一条是真的失衡。
    assert lifted["margin_m"] < 0.0
    assert lifted["hull_area_m2"] < grounded["hull_area_m2"]


def test_mutation_both_feet_airborne_reports_no_polygon_not_a_negative_margin():
    """整条 clip 悬空 `1 cm` 的那个情形。

    旧规则照样吐一个数(而且是负的),于是"重定向没把脚放到地上"被读成
    "动捕对象站不稳"。新规则必须报 ``None`` 加一个具名状态。
    """

    report = _MOD.support_polygon(_two_feet(left_z=0.0109, right_z=0.0109), np.zeros(2))
    assert report["status"] == "NO_FOOT_ON_FLOOR"
    assert report["margin_m"] is None
    assert report["foot_on_floor"] == [False, False]


def test_mutation_a_polygon_collapsed_to_a_line_is_named_degenerate():
    """支撑点塌成一条线时,不出裕度数字,出 ``DEGENERATE_SUPPORT_POLYGON``。

    这里把两只脚都收成宽度 `0`(一条前后向的线),两条线又共线,
    于是整个多边形是一条线段 —— 任何"有符号裕度"都没有物理意义。
    """

    line_feet = [
        _rect_sole(cx=0.0, cy=0.0, z=-0.0005, width=0.0),
        _rect_sole(cx=0.0, cy=0.0, z=-0.0005, width=0.0),
    ]
    report = _MOD.support_polygon(line_feet, np.zeros(2))
    assert report["status"] == "DEGENERATE_SUPPORT_POLYGON"
    assert report["margin_m"] is None
    assert report["hull_minimum_width_m"] < _MOD.DEGENERATE_SUPPORT_WIDTH_M


def test_a_band_one_micrometre_too_coarse_still_fails_the_lifted_foot_check():
    """变异测试的"粗一档就过不了":把带宽调到 `1 m`(荒谬地宽),抬起来的脚
    仍然必须被踢出多边形 —— 因为"这只脚在不在地上"从此**不由带宽决定**。
    旧规则下这一条必然翻车。"""

    report = _MOD.support_polygon(
        _two_feet(left_z=0.0030, right_z=-0.0005), np.zeros(2), band_m=1.0
    )
    assert report["foot_on_floor"] == [False, True]
    assert report["status"] == "SINGLE_FOOT_SUPPORT"


def test_ground_geometry_contract_names_are_stable():
    """报告里那几个键是给外面读的,改名就等于悄悄换掉判据。"""

    report = _MOD.support_polygon(
        _two_feet(left_z=-0.0005, right_z=-0.0005), np.zeros(2)
    )
    for key in (
        "status",
        "margin_m",
        "foot_on_floor",
        "contributing_feet",
        "hull_area_m2",
        "hull_minimum_width_m",
        "foot_support_hull_min_width_m",
        "band_m",
        "band_rule",
    ):
        assert key in report


def test_support_polygon_refuses_anything_that_is_not_two_sole_arrays():
    with pytest.raises(ExecutabilityAuditError):
        _MOD.support_polygon([np.zeros((4, 3))], np.zeros(2))
    with pytest.raises(ExecutabilityAuditError):
        _MOD.support_polygon(
            [np.zeros((4, 2)), np.zeros((4, 2))], np.zeros(2)
        )


def test_hold_anchor_accepts_the_contract_it_was_calibrated_against():
    kp = np.array([100.0, 250.0, 20.0])
    birth = np.array([0.1, -0.2, 0.3])
    torque = np.array([5.0, -12.5, 1.0])
    hold = birth + torque / kp
    report = _MOD._verify_hold_anchor(
        _StiffnessOnlyPlant(kp), _artifact(kp, birth, hold, torque)
    )
    assert report["max_residual_nm"] < _MOD.HOLD_ANCHOR_TOLERANCE_NM


@pytest.mark.parametrize("perturbation", [1.0e-6, 1.0e-3, 1.0])
def test_hold_anchor_refuses_when_the_archived_torque_drifts(perturbation):
    # 变异测试:锚点必须真的会开火,否则它只是一句注释。
    kp = np.array([100.0, 250.0, 20.0])
    birth = np.array([0.1, -0.2, 0.3])
    torque = np.array([5.0, -12.5, 1.0])
    hold = birth + torque / kp
    drifted = torque.copy()
    drifted[1] += perturbation
    with pytest.raises(ExecutabilityAuditError, match="archived hold q_des"):
        _MOD._verify_hold_anchor(
            _StiffnessOnlyPlant(kp), _artifact(kp, birth, hold, drifted)
        )


_MESH = 7  # mujoco.mjtGeom.mjGEOM_MESH; 抄成常数,免得为了这几行去 import mujoco。


def _a3_like_ankle_geoms():
    """A3 每只脚两块 mesh:先视觉(不碰撞)后 collision。顺序和真模型一致。"""

    # geom:            0 floor   1 visual-L  2 coll-L   3 visual-R  4 coll-R  5 非 mesh
    return {
        "geom_bodyid": np.array([0, 26, 26, 32, 32, 26]),
        "geom_type": np.array([0, _MESH, _MESH, _MESH, _MESH, 6]),
        "geom_contype": np.array([0, 0, 1, 0, 1, 1]),
        "geom_conaffinity": np.array([7, 0, 7, 0, 7, 7]),
        "mesh_type": _MESH,
    }


def test_sole_selection_takes_the_collision_mesh_and_not_the_visual_one():
    # 视觉网格比 collision 网格低 1.12 mm,选错了整份离地读数就跟 LP 对不上。
    assert _MOD._collision_sole_geoms(body_ids=(26, 32), **_a3_like_ankle_geoms()) == [2, 4]


def test_sole_selection_refuses_when_only_visual_meshes_exist():
    # 变异测试:把 collision 属性抹掉,必须拒绝出数,而不是悄悄退回视觉网格。
    rows = _a3_like_ankle_geoms()
    rows["geom_contype"] = np.zeros_like(rows["geom_contype"])
    with pytest.raises(ExecutabilityAuditError, match="collidable ankle-roll sole meshes"):
        _MOD._collision_sole_geoms(body_ids=(26, 32), **rows)


def test_sole_selection_refuses_a_geom_that_collides_with_nothing():
    # conaffinity=0 的 mesh 谁也碰不到,拿它量"踩没踩到地"是自欺。
    rows = _a3_like_ankle_geoms()
    rows["geom_conaffinity"] = np.zeros_like(rows["geom_conaffinity"])
    with pytest.raises(ExecutabilityAuditError):
        _MOD._collision_sole_geoms(body_ids=(26, 32), **rows)


def test_sole_selection_ignores_non_mesh_collision_geoms_on_the_same_body():
    # 第 5 个 geom 挂在同一个 body 上、也会碰撞,但它不是 mesh:顶点枚举对它没有意义。
    assert 5 not in _MOD._collision_sole_geoms(body_ids=(26, 32), **_a3_like_ankle_geoms())


def test_hold_anchor_refuses_non_finite_archives():
    kp = np.array([100.0, 250.0, 20.0])
    birth = np.array([0.1, -0.2, 0.3])
    hold = birth.copy()
    with pytest.raises(ExecutabilityAuditError):
        _MOD._verify_hold_anchor(
            _StiffnessOnlyPlant(kp),
            _artifact(kp, birth, hold, [0.0, float("nan"), 0.0]),
        )
