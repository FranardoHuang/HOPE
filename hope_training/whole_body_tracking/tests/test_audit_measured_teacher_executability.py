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
