#!/usr/bin/env python3
"""v2 定权计算器:probe 实测量 -> 冻结权重表(reward_redesign_20260725 §2 公式的可执行版)。

人话:v2.1 的名义数值(质量核 60/45/35、上台 ~135/~900、action clamp@10)是拍脑袋假设推的,
排序对但绝对档位要贴实测。pod 上跑完 200-iter probe 后,把 tensorboard 读到的实测量填进
一个 JSON,喂给本脚本,它按定权公式重算并输出可直接进 prereg 的权重表 + 预算核对。
本脚本零依赖(纯 stdlib),host 上单测覆盖(tests/test_v2_weight_calibration.py)。

输入 JSON(全部为 probe 实测,单位在字段名里):
{
  "I_weight_sum": 4.5,            # 模仿组权重和(v2 = 4.5,无 anchor_pos;换谱系要改)
  "rho_I": 0.6,                   # 模仿核平均达成度(六项 value 均值;tensorboard 各项均值/权重/dt 反推)
  "rho_Q": 0.5,                   # 窗内质量核平均达成度(pos/vel/normal 三核窗内均值)
  "p_capture_target": 0.7,        # 目标命中率 p*(设计目标,不是当前值;当前值仅记录用)
  "T_c_steps": 100,               # 每拍周期步数(clip t_cycle / dt;probe 实测挥拍频率)
  "window_steps": 13,             # 击球窗步数(2*0.12/0.02 + 1)
  "action_rate_sq_p95": 10.0,     # ||Δa||² 的 p95(probe 若未记分布,用均值×3 保守替代并注明)
  "action_acc_sq_p95": 40.0,      # ||Δ²a||² 的 p95(二阶量纲大于一阶)
  "E_net_per_fire": 1.0,          # capture 触发步 pass_net 项原始期望值(核+0.5×合法)
  "E_land_per_fire": 1.4          # 同上 landing 项(核+合法奖金)
}

输出(v2.1):质量组 Σw 按 60:45:35 分账、上台组 w_net/w_land(落点扛"击中+打好"大奖,
PSE = 质量层 × 1.2)、被删代理冻结为 0、action clamp、罚项预算核对(早期分母=模仿收入)。

用法:
  python3 scripts/v2_weight_calibration.py measured.json          # 打印冻结表
  python3 scripts/v2_weight_calibration.py measured.json --json   # 机器可读输出
"""

from __future__ import annotations

import json
import math
import sys

# 设计常量(redesign §2/§3.5;改这里=改设计,须过 prereg)
M1 = 2.0            # 击中层 vs 模仿层边际
M2 = 1.5            # 质量层 vs 击中层边际
F_PENALTY = 0.15    # 罚项预算占早期收入的分数
RHO_I_MIN = 0.4     # 罚项预算用的早期模仿达成度下限
QUALITY_SPLIT = (60.0, 45.0, 35.0)   # pos:vel:normal 名义比例(W 偏位;只用比例,绝对值重算)
# v2.1(Franco 07-25 裁定):strike_success/capture_bonus 两个人造 AND 代理删除;
# 上台组(落点=物理正确的联合成绩单)扛"击中+打好"层,目标 PSE = 质量层 x 比例。
TABLE_OVER_QUALITY = 1.2             # 上台层 vs 质量层的 PSE 比例(上台是大奖)
NET_TO_LAND_WEIGHT_RATIO = 0.15      # w_net : w_land 固定比(过网是塑形,落点是主奖)

_REQUIRED = (
    "I_weight_sum", "rho_I", "rho_Q", "p_capture_target",
    "T_c_steps", "window_steps", "action_rate_sq_p95", "action_acc_sq_p95",
    # v2.1 上台定权输入:capture 触发那一步 net/landing 项的原始期望值(核+奖金,
    # probe 从 tensorboard 的 Episode_Reward/virtual_* 均值 ÷ weight ÷ dt ÷ 触发频率反推)
    "E_net_per_fire", "E_land_per_fire",
)


def calibrate(measured: dict) -> dict:
    """按定权公式把实测量换算成冻结权重表。全程 fail-loud:缺键/非有限/出range 直接炸。"""
    for key in _REQUIRED:
        if key not in measured:
            raise ValueError(f"measured JSON missing key: {key}")
        value = measured[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"measured[{key!r}] must be a finite number, got {value!r}")
    I = float(measured["I_weight_sum"])
    rho_i = float(measured["rho_I"])
    rho_q = float(measured["rho_Q"])
    p_star = float(measured["p_capture_target"])
    t_c = float(measured["T_c_steps"])
    w_steps = float(measured["window_steps"])
    if not 0.0 < rho_i <= 1.0 or not 0.0 < rho_q <= 1.0:
        raise ValueError("rho_I / rho_Q must be in (0, 1]")
    if not 0.0 < p_star <= 1.0:
        raise ValueError("p_capture_target must be in (0, 1]")
    if t_c <= 0 or w_steps <= 0 or I <= 0:
        raise ValueError("I_weight_sum / T_c_steps / window_steps must be positive")
    if w_steps >= t_c:
        raise ValueError("window_steps must be < T_c_steps (窗不能盖满整拍)")

    e_net = float(measured["E_net_per_fire"])
    e_land = float(measured["E_land_per_fire"])
    if e_net <= 0 or e_land <= 0:
        raise ValueError("E_net_per_fire / E_land_per_fire must be positive")
    # L1 模仿:每步等效收入(weight-units/步;×dt 才是真 reward,比例不变)
    pse_mimic = I * rho_i
    # L2' 质量(窗口 dense,分开学):Σw·duty·rho_Q = m2·m1·PSE_mimic
    pse_quality = M2 * M1 * pse_mimic
    duty = w_steps / t_c
    quality_sum = pse_quality / (duty * rho_q)
    split_total = sum(QUALITY_SPLIT)
    w_pos, w_vel, w_norm = (quality_sum * s / split_total for s in QUALITY_SPLIT)
    # L3' 上台(每拍一次性,物理组合的联合成绩单):
    # PSE_table = (w_net·E_net + w_land·E_land)·p*/T_c = TABLE_OVER_QUALITY × PSE_quality
    pse_table = TABLE_OVER_QUALITY * pse_quality
    w_land = pse_table * t_c / (p_star * (NET_TO_LAND_WEIGHT_RATIO * e_net + e_land))
    w_net = NET_TO_LAND_WEIGHT_RATIO * w_land
    # 罚项预算:P <= f × 早期模仿收入;clamp 使单帧最坏罚 <= 早期收入量级
    early_income = RHO_I_MIN * I
    budget = F_PENALTY * early_income
    # clamp:weight × clamp值 <= early_income(单帧梯度不反转)
    clamp_rate = early_income / 0.2          # action_rate weight -0.2
    clamp_acc = early_income / 0.05          # action_acc weight -0.05
    # p95 若低于 clamp,clamp 取 p95(不放松到没意义)
    clamp_rate = min(clamp_rate, float(measured["action_rate_sq_p95"]))
    clamp_acc = min(clamp_acc, float(measured["action_acc_sq_p95"]))

    return {
        "inputs": {k: float(measured[k]) for k in _REQUIRED},
        "constants": {"m1": M1, "m2": M2, "f_penalty": F_PENALTY, "rho_I_min": RHO_I_MIN},
        "frozen": {
            "racket_position_weight": round(w_pos, 1),
            "racket_velocity_weight": round(w_vel, 1),
            "racket_normal_weight": round(w_norm, 1),
            "virtual_pass_net_weight": round(w_net, 1),
            "virtual_landing_weight": round(w_land, 1),
            # v2.1 删除的代理与先验(冻结为 0,防旧值回流)
            "racket_strike_success_weight": 0.0,
            "strike_capture_bonus_weight": 0.0,
            "virtual_spin_weight": 0.0,
            "action_rate_value_clamp": round(clamp_rate, 2),
            "action_acc_value_clamp": round(clamp_acc, 2),
        },
        "accounting": {
            "pse_mimic": round(pse_mimic, 3),
            "pse_quality_at_target": round(pse_quality, 3),
            "pse_table_at_target": round(pse_table, 3),
            "per_swing_table_prize": round(w_net * e_net + w_land * e_land, 1),
            "penalty_budget_per_step": round(budget, 3),
            "early_income_floor": round(early_income, 3),
            "ordering_ok": pse_mimic < pse_quality < pse_table,
        },
    }


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[1]) as f:
        measured = json.load(f)
    table = calibrate(measured)
    if "--json" in argv:
        print(json.dumps(table, indent=2, ensure_ascii=False))
        return 0
    fr, acc = table["frozen"], table["accounting"]
    print("== v2 冻结权重表(probe 校准)==")
    for key, value in fr.items():
        print(f"  {key} = {value}")
    print("== 记账核对 ==")
    for key, value in acc.items():
        print(f"  {key} = {value}")
    if not acc["ordering_ok"]:
        print("!! 定序 模仿<质量<上台 不成立 — 检查输入或设计常量")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
