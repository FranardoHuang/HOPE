"""adaptive sigma 第三通道(normal/拍面)— host-only 守卫测试(2026-07-25 A1 JOB1)。

人话背景:P2.3 自适应 sigma 原来只收 racket_position/racket_velocity 两路(SMASH 是
pos/ori/vel 三路一起收的),pos 从 0.20 收到 0.075 时拍面奖励的相对权重被静默弱化最多
~7x——拍面通道"看起来在训练,实际在放水"。修复:adaptive_sigma_normal 旗标(默认关,
逐字节不变)让 racket_normal.std 与 racket_strike_success.std_normal 按 exact-strike
面角误差(弧度)的同一路衰减 EMA 锁步收紧,min=0.262(15° 验收线)/ max=0.52(~2x)。

本文件钉死的合同:
1. 更新式:sigma_normal = clamp(sigma_ema_scale * (nrm_err_sum/denom), min, max),
   与 pos/vel 同一驱动(_update_adaptive_sigma,从 _update_metrics 摘出以便直接驱动);
2. 锁步:racket_normal.std 与 racket_strike_success.std_normal 必须同值同拍更新;
3. 旗标关 = 逐字节不变:normal 相关的 std 不动、不注册 adaptive_sigma_normal 指标;
4. fail-loud 半配置:adaptive_sigma_normal=True 而 adaptive_sigma=False 构造期拒绝;
5. 驱动数据源:_update_metrics 用与 pos/vel 相同的 decay/exact-strike 掩码累加
   normal_err_rad(弧度,不是度)。
6. adaptive_sigma_monotonic 默认关;开启后三路只收紧不回宽,且 reward terms
   先整体解析再提交,缺任何一项都 fail-loud 且零部分写入。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_adaptive_sigma_normal.py -q
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from test_reward_flags_mdp import hope_commands_mod


# --------------------------------------------------------------------------------------------- #
# fakes(镜像 test_reward_flags_mdp 的 __new__ + SimpleNamespace 惯用法)
# --------------------------------------------------------------------------------------------- #
def _sigma_cfg(**overrides):
    values = dict(
        adaptive_sigma=True,
        adaptive_sigma_normal=True,
        adaptive_sigma_monotonic=False,
        sigma_update_every=1,
        sigma_ema_scale=1.0,
        sigma_pos_min=0.075,
        sigma_pos_max=0.20,
        sigma_vel_min=0.5,
        sigma_vel_max=1.0,
        sigma_normal_min=0.262,
        sigma_normal_max=0.52,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class _FakeRewardManager:
    """get_term_cfg 的最小仿体:未知项按 isaaclab 的行为抛 ValueError。"""

    def __init__(self, terms):
        self.terms = terms

    def get_term_cfg(self, name):
        if name not in self.terms:
            raise ValueError(f"unknown reward term {name}")
        return self.terms[name]


def _default_terms():
    return {
        "racket_position": types.SimpleNamespace(params={"std": 0.20}),
        "racket_velocity": types.SimpleNamespace(params={"std": 1.0}),
        "racket_normal": types.SimpleNamespace(params={"std": 0.262}),
        "racket_strike_success": types.SimpleNamespace(
            params={"std_pos": 0.075, "std_vel": 0.5, "std_normal": 0.262}
        ),
    }


def _lockstep_terms(pos=0.20, vel=1.0, normal=0.52):
    """Monotonic mode requires additive and multiplicative live widths to agree exactly."""
    return {
        "racket_position": types.SimpleNamespace(params={"std": pos}),
        "racket_velocity": types.SimpleNamespace(params={"std": vel}),
        "racket_normal": types.SimpleNamespace(params={"std": normal}),
        "racket_strike_success": types.SimpleNamespace(
            params={"std_pos": pos, "std_vel": vel, "std_normal": normal}
        ),
    }


def _sigma_cmd(cfg=None, terms=None, n=3):
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.cfg = cfg if cfg is not None else _sigma_cfg()
    rt.num_envs = n
    rt.device = "cpu"
    if terms is None:
        terms = _default_terms()
    rt._env = types.SimpleNamespace(
        common_step_counter=0, reward_manager=_FakeRewardManager(terms)
    )
    rt._exact_pos_err_sum = 0.0
    rt._exact_vel_err_sum = 0.0
    rt._exact_nrm_err_sum = 0.0
    rt._adaptive_sigma_pos = float(rt.cfg.sigma_pos_max)
    rt._adaptive_sigma_vel = float(rt.cfg.sigma_vel_max)
    rt._adaptive_sigma_normal = float(rt.cfg.sigma_normal_max)
    rt.metrics = {
        "adaptive_sigma_pos": torch.zeros(n),
        "adaptive_sigma_vel": torch.zeros(n),
    }
    if getattr(rt.cfg, "adaptive_sigma_normal", False):
        rt.metrics["adaptive_sigma_normal"] = torch.zeros(n)
    return rt, terms


# --------------------------------------------------------------------------------------------- #
# 1+2. 更新式 + 锁步
# --------------------------------------------------------------------------------------------- #
def test_normal_sigma_tracks_mean_exact_strike_face_error_and_stays_lockstep():
    rt, terms = _sigma_cmd()
    denom = 10.0
    rt._exact_pos_err_sum = 0.10 * denom
    rt._exact_vel_err_sum = 0.70 * denom
    rt._exact_nrm_err_sum = 0.35 * denom  # 弧度;0.262 < 0.35 < 0.52,不触 clamp

    rt._update_adaptive_sigma(enough=True, denom=denom)

    assert terms["racket_normal"].params["std"] == pytest.approx(0.35)
    # 锁步:成功乘子的 std_normal 必须同一个值(加法项/乘法项同宽度打分)
    assert terms["racket_strike_success"].params["std_normal"] == pytest.approx(0.35)
    assert rt._adaptive_sigma_normal == pytest.approx(0.35)
    assert torch.allclose(rt.metrics["adaptive_sigma_normal"], torch.full((3,), 0.35))
    # pos/vel 现役行为不受第三通道影响
    assert terms["racket_position"].params["std"] == pytest.approx(0.10)
    assert terms["racket_strike_success"].params["std_pos"] == pytest.approx(0.10)


def test_normal_sigma_clamps_to_acceptance_band():
    # 下夹:误差已经好于 15° 验收线,sigma 停在 0.262 不再收(kernel 不许窄过验收)
    rt, terms = _sigma_cmd()
    rt._exact_nrm_err_sum = 0.05 * 10.0
    rt._update_adaptive_sigma(enough=True, denom=10.0)
    assert terms["racket_normal"].params["std"] == pytest.approx(0.262)
    # 上夹:误差很大时 sigma 停在 0.52(起步宽度),不发散
    rt2, terms2 = _sigma_cmd()
    rt2._exact_nrm_err_sum = 2.0 * 10.0
    rt2._update_adaptive_sigma(enough=True, denom=10.0)
    assert terms2["racket_normal"].params["std"] == pytest.approx(0.52)


def test_sigma_ema_scale_applies_to_normal_channel_too():
    rt, terms = _sigma_cmd(cfg=_sigma_cfg(sigma_ema_scale=1.2))
    rt._exact_nrm_err_sum = 0.30 * 10.0
    rt._update_adaptive_sigma(enough=True, denom=10.0)
    assert terms["racket_normal"].params["std"] == pytest.approx(0.36)


def test_variant_task_without_normal_terms_is_a_noop_not_a_crash():
    # 只有 pos/vel 项的变体任务:racket_normal 缺席 -> get_term_cfg 抛 ValueError ->
    # 与 pos/vel 现役行为一致地按 no-op 吞掉(不 crash),内部 sigma 值仍推进。
    terms = {
        "racket_position": types.SimpleNamespace(params={"std": 0.20}),
        "racket_velocity": types.SimpleNamespace(params={"std": 1.0}),
        "racket_strike_success": types.SimpleNamespace(
            params={"std_pos": 0.075, "std_vel": 0.5, "std_normal": 0.262}
        ),
    }
    rt, _ = _sigma_cmd(terms=terms)
    rt._exact_nrm_err_sum = 0.35 * 10.0
    rt._update_adaptive_sigma(enough=True, denom=10.0)
    assert rt._adaptive_sigma_normal == pytest.approx(0.35)


def test_monotonic_mode_never_reopens_position_velocity_or_normal():
    rt, terms = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=_lockstep_terms(),
    )
    denom = 10.0
    rt._exact_pos_err_sum = 0.10 * denom
    rt._exact_vel_err_sum = 0.70 * denom
    rt._exact_nrm_err_sum = 0.35 * denom
    rt._update_adaptive_sigma(enough=True, denom=denom)
    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.10, 0.70, 0.35))

    # A worse later window produces max-clamped candidates, but PBHC-style monotonic mode
    # must keep the already-earned narrower kernels in all three channels.
    rt._exact_pos_err_sum = 2.0 * denom
    rt._exact_vel_err_sum = 3.0 * denom
    rt._exact_nrm_err_sum = 2.0 * denom
    rt._update_adaptive_sigma(enough=True, denom=denom)
    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.10, 0.70, 0.35))
    assert terms["racket_position"].params["std"] == pytest.approx(0.10)
    assert terms["racket_velocity"].params["std"] == pytest.approx(0.70)
    assert terms["racket_normal"].params["std"] == pytest.approx(0.35)
    assert terms["racket_strike_success"].params == pytest.approx(
        {"std_pos": 0.10, "std_vel": 0.70, "std_normal": 0.35}
    )


def test_monotonic_mode_honors_all_lower_and_upper_bounds():
    denom = 10.0
    lower, lower_terms = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=_lockstep_terms(),
    )
    lower._exact_pos_err_sum = 0.001 * denom
    lower._exact_vel_err_sum = 0.001 * denom
    lower._exact_nrm_err_sum = 0.001 * denom
    lower._update_adaptive_sigma(enough=True, denom=denom)
    assert (
        lower._adaptive_sigma_pos,
        lower._adaptive_sigma_vel,
        lower._adaptive_sigma_normal,
    ) == pytest.approx((0.075, 0.5, 0.262))
    assert lower_terms["racket_strike_success"].params == pytest.approx(
        {"std_pos": 0.075, "std_vel": 0.5, "std_normal": 0.262}
    )

    upper, upper_terms = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=_lockstep_terms(),
    )
    upper._exact_pos_err_sum = 2.0 * denom
    upper._exact_vel_err_sum = 3.0 * denom
    upper._exact_nrm_err_sum = 2.0 * denom
    upper._update_adaptive_sigma(enough=True, denom=denom)
    assert (
        upper._adaptive_sigma_pos,
        upper._adaptive_sigma_vel,
        upper._adaptive_sigma_normal,
    ) == pytest.approx((0.20, 1.0, 0.52))
    assert upper_terms["racket_strike_success"].params == pytest.approx(
        {"std_pos": 0.20, "std_vel": 1.0, "std_normal": 0.52}
    )


def test_legacy_mode_still_reopens_kernels_when_error_worsens():
    rt, _ = _sigma_cmd(cfg=_sigma_cfg(adaptive_sigma_monotonic=False))
    denom = 10.0
    rt._exact_pos_err_sum = 0.10 * denom
    rt._exact_vel_err_sum = 0.70 * denom
    rt._exact_nrm_err_sum = 0.35 * denom
    rt._update_adaptive_sigma(enough=True, denom=denom)

    rt._exact_pos_err_sum = 2.0 * denom
    rt._exact_vel_err_sum = 3.0 * denom
    rt._exact_nrm_err_sum = 2.0 * denom
    rt._update_adaptive_sigma(enough=True, denom=denom)
    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.20, 1.0, 0.52))


def test_monotonic_first_update_uses_live_narrow_widths_and_never_reopens_them():
    # The runtime telemetry cache still starts at cfg maxima, but the composed reward recipe can
    # already be at the precision floor.  A high-error first window must keep those live widths.
    terms = _lockstep_terms(pos=0.075, vel=0.5, normal=0.262)
    rt, _ = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=terms,
    )
    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.20, 1.0, 0.52))
    rt._exact_pos_err_sum = 2.0 * 10.0
    rt._exact_vel_err_sum = 3.0 * 10.0
    rt._exact_nrm_err_sum = 2.0 * 10.0

    rt._update_adaptive_sigma(enough=True, denom=10.0)

    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.075, 0.5, 0.262))
    assert terms["racket_strike_success"].params == pytest.approx(
        {"std_pos": 0.075, "std_vel": 0.5, "std_normal": 0.262}
    )


@pytest.mark.parametrize(
    "missing_term",
    ("racket_position", "racket_velocity", "racket_normal", "racket_strike_success"),
)
def test_monotonic_mode_missing_term_fails_before_any_reward_write(missing_term):
    terms = _lockstep_terms()
    terms.pop(missing_term)
    before = {name: dict(term.params) for name, term in terms.items()}
    rt, _ = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=terms,
    )
    rt._exact_pos_err_sum = 0.10 * 10.0
    rt._exact_vel_err_sum = 0.70 * 10.0
    rt._exact_nrm_err_sum = 0.35 * 10.0

    with pytest.raises(ValueError, match="complete atomic reward-term set"):
        rt._update_adaptive_sigma(enough=True, denom=10.0)

    assert {name: dict(term.params) for name, term in terms.items()} == before
    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.20, 1.0, 0.52))


def test_monotonic_mode_missing_success_parameter_is_atomic_too():
    terms = _lockstep_terms()
    terms["racket_strike_success"].params.pop("std_normal")
    before = {name: dict(term.params) for name, term in terms.items()}
    rt, _ = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=terms,
    )
    rt._exact_pos_err_sum = 0.10 * 10.0
    rt._exact_vel_err_sum = 0.70 * 10.0
    rt._exact_nrm_err_sum = 0.35 * 10.0

    with pytest.raises(ValueError, match="complete atomic reward-term set"):
        rt._update_adaptive_sigma(enough=True, denom=10.0)

    assert {name: dict(term.params) for name, term in terms.items()} == before


@pytest.mark.parametrize(
    ("additive_term", "success_key"),
    (
        ("racket_position", "std_pos"),
        ("racket_velocity", "std_vel"),
        ("racket_normal", "std_normal"),
    ),
)
def test_monotonic_mode_rejects_live_width_drift_before_any_write(
    additive_term, success_key
):
    terms = _lockstep_terms()
    terms[additive_term].params["std"] *= 0.5
    before = {name: dict(term.params) for name, term in terms.items()}
    rt, _ = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=terms,
    )
    rt._exact_pos_err_sum = 0.10 * 10.0
    rt._exact_vel_err_sum = 0.70 * 10.0
    rt._exact_nrm_err_sum = 0.35 * 10.0

    with pytest.raises(ValueError, match="exactly lockstep"):
        rt._update_adaptive_sigma(enough=True, denom=10.0)

    assert terms[additive_term].params["std"] != terms["racket_strike_success"].params[success_key]
    assert {name: dict(term.params) for name, term in terms.items()} == before


# --------------------------------------------------------------------------------------------- #
# 3. 旗标关 = 逐字节不变
# --------------------------------------------------------------------------------------------- #
def test_flag_off_leaves_normal_stds_and_metric_registry_untouched():
    rt, terms = _sigma_cmd(cfg=_sigma_cfg(adaptive_sigma_normal=False))
    rt._exact_pos_err_sum = 0.10 * 10.0
    rt._exact_vel_err_sum = 0.70 * 10.0
    rt._exact_nrm_err_sum = 0.35 * 10.0  # 累加是无条件的,但读取必须被旗标挡住

    rt._update_adaptive_sigma(enough=True, denom=10.0)

    # normal 两处 std 一动不动;pos/vel 照旧更新
    assert terms["racket_normal"].params["std"] == pytest.approx(0.262)
    assert terms["racket_strike_success"].params["std_normal"] == pytest.approx(0.262)
    assert rt._adaptive_sigma_normal == pytest.approx(0.52)  # 停在初始最大值
    assert terms["racket_position"].params["std"] == pytest.approx(0.10)
    # 指标键集合不变(默认跑的 wandb 键逐字节一致)
    assert "adaptive_sigma_normal" not in rt.metrics


def test_cfg_defaults_are_byte_identical_off():
    cfg_cls = hope_commands_mod.RacketTargetCommandCfg
    assert cfg_cls.adaptive_sigma_normal is False
    assert cfg_cls.adaptive_sigma_monotonic is False
    assert cfg_cls.sigma_normal_min == pytest.approx(0.262)  # 15° 验收线
    assert cfg_cls.sigma_normal_max == pytest.approx(0.52)   # ~2x 验收
    # 指标注册在 __init__ 里必须被旗标 gate 住
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand.__init__)
    assert 'if getattr(cfg, "adaptive_sigma_normal", False):' in src


# --------------------------------------------------------------------------------------------- #
# 4. fail-loud 半配置
# --------------------------------------------------------------------------------------------- #
def test_normal_without_adaptive_sigma_fails_loud():
    with pytest.raises(ValueError, match="requires adaptive_sigma=True"):
        hope_commands_mod._validate_adaptive_sigma_cfg(
            types.SimpleNamespace(adaptive_sigma=False, adaptive_sigma_normal=True)
        )
    # 合法组合全部放行
    for on, normal in ((False, False), (True, False), (True, True)):
        hope_commands_mod._validate_adaptive_sigma_cfg(
            types.SimpleNamespace(adaptive_sigma=on, adaptive_sigma_normal=normal)
        )
    # __init__ 必须在构造早期调用这个校验(旗标沉睡到课程期才炸是不允许的)
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand.__init__)
    assert "_validate_adaptive_sigma_cfg(cfg)" in src


def test_monotonic_without_adaptive_sigma_fails_loud():
    with pytest.raises(ValueError, match="adaptive_sigma_monotonic=True requires"):
        hope_commands_mod._validate_adaptive_sigma_cfg(
            types.SimpleNamespace(
                adaptive_sigma=False,
                adaptive_sigma_normal=False,
                adaptive_sigma_monotonic=True,
            )
        )
    hope_commands_mod._validate_adaptive_sigma_cfg(
        types.SimpleNamespace(
            adaptive_sigma=True,
            adaptive_sigma_normal=True,
            adaptive_sigma_monotonic=True,
        )
    )


# --------------------------------------------------------------------------------------------- #
# 5. 驱动数据源:与 pos/vel 同 decay/掩码的弧度 EMA
# --------------------------------------------------------------------------------------------- #
def test_driver_reuses_exact_strike_ema_pattern_in_radians():
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand._update_metrics)
    # 弧度先算,度数只是展示换算(第三通道驱动量纲 = 弧度,与 sigma_normal_min/max 一致)
    assert "normal_err_rad = torch.acos(cos_ang)" in src
    assert "normal_err_deg = normal_err_rad * (180.0 / math.pi)" in src
    # 与 pos/vel 完全平行的同 decay/同掩码累加式。现役路径把三个误差 sum
    # 放进同一次 batched host read,然后依次消费;两端都要钉住。
    assert "(normal_err_rad * exact_strike).sum()" in src
    assert "self._exact_nrm_err_sum = (" in src
    assert "decay * self._exact_nrm_err_sum + next(_exact_metric_values)" in src
    # pos/vel 的驱动累加式仍在(防止有人重构时顺手拆掉平行结构)
    assert "decay * self._exact_pos_err_sum + next(_exact_metric_values)" in src
    assert "decay * self._exact_vel_err_sum + next(_exact_metric_values)" in src


def test_update_metrics_delegates_to_extracted_sigma_updater():
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand._update_metrics)
    assert "self._update_adaptive_sigma(enough, denom)" in src
    # 摘出的方法保持原节拍闸门(adaptive_sigma & enough & sigma_update_every)
    upd = inspect.getsource(hope_commands_mod.RacketTargetCommand._update_adaptive_sigma)
    assert "self.cfg.adaptive_sigma" in upd
    assert "common_step_counter % int(self.cfg.sigma_update_every)" in upd


def test_not_enough_samples_keeps_all_sigmas_at_init():
    rt, terms = _sigma_cmd()
    rt._exact_nrm_err_sum = 0.35 * 10.0
    rt._update_adaptive_sigma(enough=False, denom=10.0)
    assert terms["racket_normal"].params["std"] == pytest.approx(0.262)
    assert rt._adaptive_sigma_normal == pytest.approx(0.52)
    # 指标仍每步广播当前值(reset 清零后要能恢复)
    assert torch.allclose(rt.metrics["adaptive_sigma_normal"], torch.full((3,), 0.52))


def test_monotonic_mode_not_enough_samples_is_a_noop():
    rt, terms = _sigma_cmd(
        cfg=_sigma_cfg(adaptive_sigma_monotonic=True),
        terms=_lockstep_terms(),
    )
    before = {name: dict(term.params) for name, term in terms.items()}
    rt._exact_pos_err_sum = 0.10 * 10.0
    rt._exact_vel_err_sum = 0.70 * 10.0
    rt._exact_nrm_err_sum = 0.35 * 10.0
    rt._update_adaptive_sigma(enough=False, denom=10.0)
    assert {name: dict(term.params) for name, term in terms.items()} == before
    assert (
        rt._adaptive_sigma_pos,
        rt._adaptive_sigma_vel,
        rt._adaptive_sigma_normal,
    ) == pytest.approx((0.20, 1.0, 0.52))
