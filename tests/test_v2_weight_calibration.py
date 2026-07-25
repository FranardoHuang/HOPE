"""v2 定权计算器(scripts/v2_weight_calibration.py)— host-only 单测。

钉死:①名义输入复现 redesign §3.5 的名义表(B≈771 @I=4.5、质量组 Σw≈138 及 60:45:35 分账);
②定序 模仿<击中<质量 恒成立;③fail-loud 面(缺键/非有限/出 range/窗盖满整拍);
④clamp 取"预算上限"与"实测 p95"的较小者。

Run:  python -m pytest tests/test_v2_weight_calibration.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from v2_weight_calibration import calibrate  # noqa: E402


def _nominal(**over):
    base = {
        "I_weight_sum": 4.5,
        "rho_I": 0.6,
        "rho_Q": 0.5,
        "p_capture_target": 0.7,
        "T_c_steps": 100,
        "window_steps": 13,
        "action_rate_sq_p95": 10.0,
        "action_acc_sq_p95": 40.0,
    }
    base.update(over)
    return base


def test_nominal_inputs_reproduce_the_blueprint_table():
    out = calibrate(_nominal())
    fr = out["frozen"]
    # B = m1·I·rho_I·T_c/p* = 2×4.5×0.6×100/0.7 ≈ 771.4(§3.5 脚注:850 是 I=5 历史口径)
    assert fr["strike_capture_bonus_weight"] == pytest.approx(771.4, abs=0.1)
    # 质量组 Σw = m2·m1·I·rho_I/(duty·rho_Q) = 1.5×2×2.7/(0.13×0.5) ≈ 124.6,按 60:45:35 分
    total = (
        fr["racket_position_weight"]
        + fr["racket_velocity_weight"]
        + fr["racket_normal_weight"]
    )
    assert total == pytest.approx(124.6, abs=0.3)
    # 输出按 1 位小数取整,比例断言给 1% 容差(53.4/40.1=1.3317 vs 4/3)
    assert fr["racket_position_weight"] / fr["racket_velocity_weight"] == pytest.approx(60 / 45, rel=1e-2)
    assert fr["racket_velocity_weight"] / fr["racket_normal_weight"] == pytest.approx(45 / 35, rel=1e-2)
    assert out["accounting"]["ordering_ok"] is True


def test_ordering_holds_across_measured_ranges():
    for rho_i in (0.3, 0.6, 0.9):
        for rho_q in (0.2, 0.5, 0.9):
            for p in (0.3, 0.7, 1.0):
                acc = calibrate(_nominal(rho_I=rho_i, rho_Q=rho_q, p_capture_target=p))["accounting"]
                assert acc["ordering_ok"], (rho_i, rho_q, p)
                assert acc["pse_mimic"] < acc["pse_hit_at_p_star"] < acc["pse_quality_at_target"]


def test_clamp_takes_min_of_budget_and_p95():
    # 早期收入地板 = 0.4×4.5 = 1.8;预算上限 clamp_rate = 1.8/0.2 = 9.0 < p95 10 → 取 9.0
    fr = calibrate(_nominal())["frozen"]
    assert fr["action_rate_value_clamp"] == pytest.approx(9.0, abs=0.01)
    # p95 更小时取 p95
    fr2 = calibrate(_nominal(action_rate_sq_p95=4.0))["frozen"]
    assert fr2["action_rate_value_clamp"] == pytest.approx(4.0, abs=0.01)


def test_fail_loud_surfaces():
    with pytest.raises(ValueError, match="missing key"):
        calibrate({k: v for k, v in _nominal().items() if k != "rho_I"})
    with pytest.raises(ValueError, match="finite number"):
        calibrate(_nominal(rho_I=float("nan")))
    with pytest.raises(ValueError, match="finite number"):
        calibrate(_nominal(rho_Q=True))
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        calibrate(_nominal(rho_I=0.0))
    with pytest.raises(ValueError, match="盖满整拍"):
        calibrate(_nominal(window_steps=100))
    with pytest.raises(ValueError, match="positive"):
        calibrate(_nominal(I_weight_sum=-1.0))


def test_one_shot_variance_stays_single_source():
    # B 是唯一 one-shot:质量组输出全是窗口 dense 权重,不该出现第二个 one-shot 字段
    fr = calibrate(_nominal())["frozen"]
    one_shots = [k for k in fr if "bonus" in k]
    assert one_shots == ["strike_capture_bonus_weight"]


def test_higher_measured_imitation_scales_everything_proportionally():
    lo = calibrate(_nominal(rho_I=0.5))["frozen"]
    hi = calibrate(_nominal(rho_I=0.75))["frozen"]
    ratio = hi["strike_capture_bonus_weight"] / lo["strike_capture_bonus_weight"]
    assert ratio == pytest.approx(1.5, rel=1e-3)
    ratio_q = hi["racket_position_weight"] / lo["racket_position_weight"]
    assert ratio_q == pytest.approx(1.5, rel=1e-3)


def test_calibrate_result_is_json_serializable_and_rounded():
    import json

    out = calibrate(_nominal())
    json.dumps(out)  # 不炸
    for value in out["frozen"].values():
        assert value == round(value, 2)
