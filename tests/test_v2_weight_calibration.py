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
        # 07-26 起名义夹具 = v4rg probe 实测值(k_eff 口径)
        "I_weight_sum": 4.5,
        "rho_I": 0.547,
        "k_eff_pos": 0.73,
        "k_eff_vel": 0.057,
        "k_eff_normal": 0.165,
        "T_c_steps": 46.3,
        "window_steps": 13,
        # 07-26 原夹具写的是 10.0 / 40.0,两个都 >= 各自的预算档(9.0 / 36.0),
        # 也就是说名义配方本身是不可行的 —— 见
        # test_a_clamp_at_or_below_the_operating_p95_is_refused。名义夹具改成可行值,
        # 让"取较小者"这条规则还有happy path可测。
        "action_rate_sq_p95": 4.0,
        "action_acc_sq_p95": 20.0,
        "p_legal_target": 0.7,
        "E_land_value_per_legal": 0.8,
        "motion_lineage": "v4rg_runtime_order_v3",
    }
    base.update(over)
    return base


def test_nominal_inputs_reproduce_the_blueprint_table():
    out = calibrate(_nominal())
    fr = out["frozen"]
    # 阶梯 1:3:7.5 锚实测:PSE_mimic=2.4615, q=7.3845, table=18.461
    # 质量 scale = 7.3845×46.3/(60×0.73+45×0.057+35×0.165=52.14) = 6.557 → 393.4/295.1/229.5
    assert fr["racket_position_weight"] == pytest.approx(393.4, abs=0.5)
    assert fr["racket_velocity_weight"] == pytest.approx(295.1, abs=0.5)
    assert fr["racket_normal_weight"] == pytest.approx(229.5, abs=0.5)
    # w_land = 18.461×46.3/(0.7×0.8) ≈ 1526.2(夹具 p_legal 0.7/E 0.8;正式表用实测 0.6/0.864)
    assert fr["virtual_landing_weight"] == pytest.approx(1526.2, abs=0.5)
    assert fr["virtual_pass_net_weight"] == 0.0
    # 被删代理冻结为 0
    assert fr["racket_strike_success_weight"] == 0.0
    assert fr["strike_capture_bonus_weight"] == 0.0
    assert fr["virtual_spin_weight"] == 0.0
    # 内部比例 60:45:35 保持
    assert fr["racket_position_weight"] / fr["racket_velocity_weight"] == pytest.approx(60 / 45, rel=1e-2)
    assert fr["racket_velocity_weight"] / fr["racket_normal_weight"] == pytest.approx(45 / 35, rel=1e-2)
    assert out["accounting"]["ordering_ok"] is True


def test_ordering_holds_across_measured_ranges():
    for rho_i in (0.3, 0.6, 0.9):
        for _ in (1,):
            for p in (0.3, 0.7, 1.0):
                acc = calibrate(_nominal(rho_I=rho_i, p_legal_target=p))["accounting"]
                assert acc["ordering_ok"], (rho_i, rho_q, p)
                assert acc["pse_mimic"] < acc["pse_quality_at_target"] < acc["pse_table_at_target"]


def test_clamp_takes_min_of_budget_and_p95():
    # 早期收入地板 = 0.4×4.5 = 1.8;预算上限 clamp_rate = 1.8/0.2 = 9.0 > p95 4 → 取 4.0
    fr = calibrate(_nominal())["frozen"]
    assert fr["action_rate_value_clamp"] == pytest.approx(4.0, abs=0.01)
    assert fr["action_acc_value_clamp"] == pytest.approx(20.0, abs=0.01)
    # p95 更靠近预算档但仍在其下 → 仍取 p95
    fr2 = calibrate(_nominal(action_rate_sq_p95=8.9))["frozen"]
    assert fr2["action_rate_value_clamp"] == pytest.approx(8.9, abs=0.01)


def test_a_clamp_at_or_below_the_operating_p95_is_refused():
    """该拦的:预算档掉到工作区 p95 之下 = 整条分布都在天花板上 = 死项。

    这条不是假想。2026-07-26 出表时 ``action_rate`` 走的就是这个分支:
    ``early_income/|w| = 1.8/0.2 = 9.0`` vs 名义 p95 ``10.0``,``min()`` 安静地返回 9.0,
    冻结成 ``value_clamp = 9.0`` 并一路进了 18 份内容寻址合同。实测工作区
    (build_1,唯一已知能打球的实现)是 ``||Δa||² = 63.1``(iter 4)到 ``10.8``(收敛),
    **全程在 9.0 之上**,于是 s15r1 两格五个 update 的 raw_sum 逐位等于 ``98304 × 9``。
    """

    # (甲)当年那份名义输入
    with pytest.raises(ValueError, match="infeasible"):
        calibrate(_nominal(action_rate_sq_p95=10.0))
    # (乙)build_1 实测的工作区,差得更远
    with pytest.raises(ValueError, match="infeasible"):
        calibrate(_nominal(action_rate_sq_p95=63.1))
    # (丙)相等也算不可行:p95 恰好等于预算档 = 至少 5% 的样本贴着天花板
    with pytest.raises(ValueError, match="infeasible"):
        calibrate(_nominal(action_rate_sq_p95=9.0))
    # (丁)二阶轴同规矩,预算档 1.8/0.05 = 36.0
    with pytest.raises(ValueError, match="infeasible"):
        calibrate(_nominal(action_acc_sq_p95=36.0))
    # 误拦的不许拦:比预算档低一点点就该照常出表
    assert calibrate(_nominal(action_acc_sq_p95=35.9))["frozen"][
        "action_acc_value_clamp"
    ] == pytest.approx(35.9, abs=0.01)


def test_refusal_names_the_weight_that_would_have_been_feasible():
    """拒收不能只说"不行":要给出这条预算规则在实测工作点上真正能给的剂量。"""

    with pytest.raises(ValueError) as excinfo:
        calibrate(_nominal(action_rate_sq_p95=63.1))
    message = str(excinfo.value)
    assert "Lower the weight" in message
    assert "0.02853" in message or "0.0285" in message  # 1.8 / 63.1


def test_p95_inputs_must_be_positive():
    with pytest.raises(ValueError, match="action_rate_sq_p95"):
        calibrate(_nominal(action_rate_sq_p95=0.0))
    with pytest.raises(ValueError, match="action_acc_sq_p95"):
        calibrate(_nominal(action_acc_sq_p95=-1.0))


def test_lineage_tag_is_mandatory_and_passes_through():
    out = calibrate(_nominal())
    assert out["motion_lineage"] == "v4rg_runtime_order_v3"
    with pytest.raises(ValueError, match="motion_lineage"):
        calibrate({k: v for k, v in _nominal().items() if k != "motion_lineage"})
    with pytest.raises(ValueError, match="motion_lineage"):
        calibrate(_nominal(motion_lineage="  "))


def test_fail_loud_surfaces():
    with pytest.raises(ValueError, match="missing key"):
        calibrate({k: v for k, v in _nominal().items() if k != "rho_I"})
    with pytest.raises(ValueError, match="finite number"):
        calibrate(_nominal(rho_I=float("nan")))
    with pytest.raises(ValueError, match="finite number"):
        calibrate(_nominal(k_eff_vel=True))
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        calibrate(_nominal(rho_I=0.0))
    with pytest.raises(ValueError, match="k_eff"):
        calibrate(_nominal(k_eff_pos=0.0))
    with pytest.raises(ValueError, match="盖满整拍"):
        calibrate(_nominal(window_steps=100))
    with pytest.raises(ValueError, match="positive"):
        calibrate(_nominal(I_weight_sum=-1.0))


def test_one_shot_prize_lives_only_in_the_table_group():
    # v2.1:唯一的每拍大额结算在上台组(物理组合),代理 one-shot 恒 0
    out = calibrate(_nominal())
    assert out["frozen"]["strike_capture_bonus_weight"] == 0.0
    assert out["accounting"]["per_swing_table_prize"] == pytest.approx(1526.2*0.8, abs=1.5)


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
