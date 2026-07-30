"""Per-env 零均值凹凸地垫的纯 numpy 性质(terrain_patch.py,2026-07-29 抬脚地形修复)。

Pinned here (host-only, no isaaclab):

* 零均值:authored ``[lo, hi]`` 居中成 ±(hi-lo)/2;样本落在对称整数级 {-K..K},均值≈0。
* 桌子一侧(x >= near edge)逐格恰好为 0——桌面 0.76 / 动作库 / 虚拟球标定不动。
* 机器人一侧真的有正负两种起伏(不是被削成"只凸不凹")。
* 5 mm 量化:所有值都是整数级;带宽窄到量化成死平垫时 fail-loud。
* 同 seed 逐 bit 复现,不同 seed 不同 pattern。
* extents:带桌子 = 近沿往后 3 m 起、桌长+0.5 m 余量止;没桌子 = 对称全粗糙垫。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_terrain_patch.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TP_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/terrain_patch.py"
)
_SPEC = importlib.util.spec_from_file_location("terrain_patch_under_test", TP_PATH)
TP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(TP)

_BAND = (0.0, 0.04)  # authored -> ±0.02 about z=0
_NEAR_X = 0.5


def _build(seed=7, band=_BAND, near_x=_NEAR_X):
    x_min, x_max, y_half = TP.patch_extents_m(near_x)
    hf = TP.build_patch_height_field(
        band, near_x, x_min, x_max, y_half, np.random.default_rng(seed)
    )
    return hf, (x_min, x_max, y_half)


def test_zero_mean_and_symmetric_band():
    hf, _ = _build()
    K = int(round(TP.zero_mean_half_band_m(_BAND) / TP.VERTICAL_SCALE_M))
    assert K == 4
    assert hf.min() >= -K and hf.max() <= K
    x_min, x_max, _ = TP.patch_extents_m(_NEAR_X)
    x_coords = x_min + np.arange(hf.shape[0]) * TP.HORIZONTAL_SCALE_M
    rough = hf[x_coords < _NEAR_X - 1e-9, :]
    # 对称整数级均匀采样,~1500+ 格的样本均值应该压在半带宽的 5% 以内
    mean_m = float(rough.mean()) * TP.VERTICAL_SCALE_M
    assert abs(mean_m) < 0.05 * TP.zero_mean_half_band_m(_BAND)
    # 真有正负两种起伏
    assert (rough > 0).any() and (rough < 0).any()


def test_table_side_is_exactly_flat_zero():
    hf, (x_min, _, _) = _build()
    x_coords = x_min + np.arange(hf.shape[0]) * TP.HORIZONTAL_SCALE_M
    flat = hf[x_coords >= _NEAR_X - 1e-9, :]
    assert flat.size > 0
    assert not flat.any()  # 逐格恰好 0
    rough = hf[x_coords < _NEAR_X - 1e-9, :]
    assert rough.any()


def test_quantization_levels_are_integers_of_5mm():
    hf, _ = _build()
    assert hf.dtype == np.int16
    K = int(round(TP.zero_mean_half_band_m(_BAND) / TP.VERTICAL_SCALE_M))
    assert set(np.unique(hf)).issubset(set(range(-K, K + 1)))


def test_seed_determinism():
    a, _ = _build(seed=123)
    b, _ = _build(seed=123)
    c, _ = _build(seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_extents_with_and_without_table():
    x_min, x_max, y_half = TP.patch_extents_m(_NEAR_X)
    # 近沿 0.5:垫子从 -2.5 起(机器人身后 3 m),到 0.5+2.74+0.5=3.74 止
    assert x_min == pytest.approx(_NEAR_X - TP.X_BACK_M)
    assert x_max == pytest.approx(_NEAR_X + 2.74 + TP.X_FORWARD_MARGIN_M)
    assert y_half == pytest.approx(TP.Y_HALF_M)
    # 没桌子(无 racket_target 的谱系):对称全粗糙垫
    assert TP.patch_extents_m(None) == (-TP.X_BACK_M, TP.X_BACK_M, TP.Y_HALF_M)
    hf = TP.build_patch_height_field(
        _BAND, None, -3.0, 3.0, 3.0, np.random.default_rng(0)
    )
    assert (hf > 0).any() and (hf < 0).any()


def test_built_row_span_covers_table_and_reaches_x_max():
    # 离散化后的实铺跨度必须盖过整张桌子的远沿(literal 2.74,不用 TP 常量自证),
    # 端点与 x_max 至多差一格,且平区一直铺到最后一行。
    hf, (x_min, x_max, _) = _build()
    realized_end = x_min + (hf.shape[0] - 1) * TP.HORIZONTAL_SCALE_M
    assert realized_end >= _NEAR_X + 2.74
    assert realized_end >= x_max - 1e-9  # ceil 语义:绝不短于声明的 extents
    assert realized_end - x_max < TP.HORIZONTAL_SCALE_M
    assert not hf[-1, :].any()


def test_dead_flat_band_is_refused():
    with pytest.raises(ValueError, match="hi - lo"):
        TP.build_patch_height_field(
            (0.02, 0.025), _NEAR_X, -2.5, 3.74, 3.0, np.random.default_rng(0)
        )


def test_nominal_1cm_band_is_accepted_despite_float_noise():
    # 0.03-0.02 = 0.009999999999999998:名义 1 cm 带不许被浮点噪声拒掉
    hf = TP.build_patch_height_field(
        (0.02, 0.03), _NEAR_X, -2.5, 3.74, 3.0, np.random.default_rng(0)
    )
    assert set(np.unique(hf)).issubset({-1, 0, 1})


def test_band_above_wall_correction_limit_is_refused():
    # 带宽 > 0.15 m:斜坡竖墙修正会把负高度顶到桌侧平区边界列
    assert TP.MAX_BAND_M == pytest.approx(0.15)
    with pytest.raises(ValueError, match="<= 0.15"):
        TP.build_patch_height_field(
            (0.0, 0.2), _NEAR_X, -2.5, 3.74, 3.0, np.random.default_rng(0)
        )


def test_non_multiple_band_is_refused_never_silently_rescaled():
    # [0.0, 0.015] -> ±7.5 mm 不是 5 mm 的整数级,拒绝而不是悄悄四舍五入
    with pytest.raises(ValueError, match="multiple"):
        TP.build_patch_height_field(
            (0.0, 0.015), _NEAR_X, -2.5, 3.74, 3.0, np.random.default_rng(0)
        )


def test_degenerate_extents_are_refused():
    with pytest.raises(ValueError, match="degenerate"):
        TP.build_patch_height_field(
            _BAND, None, 0.0, 0.0, 3.0, np.random.default_rng(0)
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
