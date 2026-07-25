"""v2 定权计算器(scripts/v2_weight_calibration.py)— host-only 单测。

钉死(v2.1,Franco 07-25:代理全删、分开学、上台扛大奖):①名义输入复现名义表
(质量组 Σw≈124.6 按 60:45:35;上台 w_land≈895.9/w_net≈134.4;被删代理冻结 0);
②定序 模仿<质量<上台 恒成立;③fail-loud 面;④clamp 取"预算上限"与"实测 p95"较小者。

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
        "p_legal_target": 0.7,
        "E_land_value_per_legal": 0.8,
    }
    base.update(over)
    return base


def test_nominal_inputs_reproduce_the_blueprint_table():
    out = calibrate(_nominal())
    fr = out["frozen"]
    # v2.2:landing 独扛,上台加码 1.5×。w_land = 1.5×8.1×100/(0.7×0.8) ≈ 2169.6;pass_net 冻结 0
    assert fr["virtual_landing_weight"] == pytest.approx(2169.6, abs=0.3)
    assert fr["virtual_pass_net_weight"] == 0.0
    # 被删代理冻结为 0
    assert fr["racket_strike_success_weight"] == 0.0
    assert fr["strike_capture_bonus_weight"] == 0.0
    assert fr["virtual_spin_weight"] == 0.0
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
                assert acc["pse_mimic"] < acc["pse_quality_at_target"] < acc["pse_table_at_target"]


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


def test_one_shot_prize_lives_only_in_the_table_group():
    # v2.1:唯一的每拍大额结算在上台组(物理组合),代理 one-shot 恒 0
    out = calibrate(_nominal())
    assert out["frozen"]["strike_capture_bonus_weight"] == 0.0
    assert out["accounting"]["per_swing_table_prize"] == pytest.approx(1735.7, abs=1.0)


def test_higher_measured_imitation_scales_everything_proportionally():
    lo = calibrate(_nominal(rho_I=0.5))["frozen"]
    hi = calibrate(_nominal(rho_I=0.75))["frozen"]
    ratio = hi["virtual_landing_weight"] / lo["virtual_landing_weight"]
    assert ratio == pytest.approx(1.5, rel=1e-3)
    ratio_q = hi["racket_position_weight"] / lo["racket_position_weight"]
    assert ratio_q == pytest.approx(1.5, rel=1e-3)


def test_calibrate_result_is_json_serializable_and_rounded():
    import json

    out = calibrate(_nominal())
    json.dumps(out)  # 不炸
    for value in out["frozen"].values():
        assert value == round(value, 2)
