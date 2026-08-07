"""CPU-only tests for the measured-clip floor-grounding diagnostic.

不需要 mujoco:只测纯数值/纯契约的部分。真 clip 的接地解算在 pod 上跑,不进单测。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "ground_measured_clip_to_floor",
    _SCRIPTS / "ground_measured_clip_to_floor.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["ground_measured_clip_to_floor"] = _MOD
_SPEC.loader.exec_module(_MOD)

ClipGroundingError = _MOD.ClipGroundingError


# --------------------------------------------------------------------------
# 四元数归一化的不动点。这是本工具唯一一处"替共享模块挡浮点"的地方,必须有测试。
# --------------------------------------------------------------------------


def test_unit_quaternion_fixed_point_is_actually_a_fixed_point():
    # 拿一批不同量级/不同符号的四元数,逐个要求"再归一化一次逐位不变"。
    rng = np.random.default_rng(20260807)
    for _ in range(400):
        raw = rng.normal(size=4)
        if np.linalg.norm(raw) < 1.0e-6:
            continue
        fixed = _MOD._unit_quaternion_fixed_point(raw)
        again = fixed / float(np.linalg.norm(fixed))
        assert np.array_equal(again, fixed)
        # 方向没被改坏:与原四元数同向,夹角为零。
        unit = raw / np.linalg.norm(raw)
        assert float(np.dot(fixed, unit)) == pytest.approx(1.0, abs=1.0e-12)


def test_unit_quaternion_fixed_point_leaves_an_already_exact_unit_alone():
    exact = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(_MOD._unit_quaternion_fixed_point(exact), exact)


def test_unit_quaternion_fixed_point_refuses_a_degenerate_input():
    for bad in (np.zeros(4), np.array([np.nan, 0.0, 0.0, 0.0])):
        with pytest.raises(ClipGroundingError):
            _MOD._unit_quaternion_fixed_point(bad)


def test_one_plain_normalisation_is_not_always_enough():
    """本工具存在的理由:``q/‖q‖`` 在浮点上**不幂等**。

    这一条不断言"某个具体的四元数会翻车"(那依赖 numpy 版本),它断言的是
    **不动点迭代和单次归一化不是同一件事** —— 随机采样里一定找得到反例,
    找不到就说明这条护栏已经没有对象,应该有人来把它删掉而不是留着装样子。
    """

    rng = np.random.default_rng(11)
    differed = 0
    for _ in range(2000):
        raw = rng.normal(size=4)
        once = raw / np.linalg.norm(raw)
        twice = once / float(np.linalg.norm(once))
        if not np.array_equal(once, twice):
            differed += 1
    assert differed > 0


# --------------------------------------------------------------------------
# 帧选择
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("0", (0,)),
        ("all", (0, 1, 2, 3)),
        ("0,2", (0, 2)),
        ("1-3", (1, 2, 3)),
        ("3,1-2,3", (3, 1, 2)),
    ],
)
def test_parse_frames(spec, expected):
    assert _MOD._parse_frames(spec, 4) == expected


@pytest.mark.parametrize("spec", ["", "4", "-1", "3-1", "0,9"])
def test_parse_frames_refuses_out_of_range_or_empty(spec):
    with pytest.raises(ClipGroundingError):
        _MOD._parse_frames(spec, 4)


# --------------------------------------------------------------------------
# 收据口径
# --------------------------------------------------------------------------


def _receipt(**gates):
    table = {name: "PASS" for name in _MOD.GATE_NAMES}
    table.update(gates)
    table.setdefault("exact_model_identity", "PASS")
    table.setdefault("static_ground_dynamics", "PASS")
    return {"gates": table, "static_geometry": {"support": {"margin_m": 0.012}}}


def test_gate_table_reads_the_receipt_and_does_not_invent_a_second_verdict():
    table = _MOD._gate_table(_receipt(sole_floor="FAIL_CLOSED"))
    assert table["sole_floor"] == "FAIL_CLOSED"
    assert table["double_support"] == "PASS"
    assert set(_MOD.GATE_NAMES) <= set(table)


def test_margin_mm_passes_none_through_instead_of_printing_a_zero():
    assert _MOD._margin_mm(_receipt()) == pytest.approx(12.0)
    empty = _receipt()
    empty["static_geometry"]["support"]["margin_m"] = None
    assert _MOD._margin_mm(empty) is None


def test_leg_bodies_are_told_apart_from_the_bodies_the_racket_hangs_off():
    for name in (
        "left_hip_pitch_Link",
        "right_knee_Link",
        "left_ankle_roll_Link",
    ):
        assert _MOD._is_leg_body(name)
    for name in (
        "pelvis_link",
        "torso_Link",
        "right_wrist_yaw_Link",
        "right_shoulder_pitch_Link",
        "head_yaw_Link",
    ):
        assert not _MOD._is_leg_body(name)


# --------------------------------------------------------------------------
# 整鞋底 footprint 裕度(报告里给那道门配的对照读数)
# --------------------------------------------------------------------------


def _plate(cx, cy, z=0.0, half=0.05):
    xs = np.linspace(cx - half, cx + half, 4)
    ys = np.linspace(cy - half, cy + half, 4)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def test_footprint_margin_is_positive_inside_and_negative_outside():
    soles = [_plate(0.0, +0.15), _plate(0.0, -0.15)]
    inside = _MOD._footprint_margin(np.array([0.0, 0.0, 1.0]), soles)
    outside = _MOD._footprint_margin(np.array([0.30, 0.0, 1.0]), soles)
    assert inside == pytest.approx(0.05, abs=1.0e-9)
    assert outside == pytest.approx(-0.25, abs=1.0e-9)


def test_footprint_margin_is_none_when_there_is_no_two_dimensional_hull():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert _MOD._footprint_margin(np.array([0.5, 0.0, 1.0]), [line, line]) is None
