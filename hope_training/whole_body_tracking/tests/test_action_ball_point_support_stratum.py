"""At level zero the support is one point, so the 1/3/1 schedule has nothing to say.

人话:``initial_center_single_question`` 的合同写着"32 条 arm 全是精确 0 时物理支撑集
就是 profile 中心那一个点"。生产方 ``_sampling_plan_for_request`` 照此对**每个**
proposal_index 都发 ``center``;但两个独立收据解码器
(``BaseBirthReceipt.from_identity_receipt`` / ``BallBaseSample.from_identity_receipt``)
用的是**另一条更粗的规则** —— 直接拿 ``mixture.stratum_for(index)`` 对答案。于是四格
`scale4096`(DR-L0 全关、`initial_center_single_question=true`)第一次存 checkpoint 时,
``assert_issued_birth`` 一路走到这里就炸:
``ValueError: birth sampling_stratum disagrees with mixture schedule``。

这是"期望值是第三份手抄"的老形状:验收方自己重推了一遍生产方的规则,推得比生产方粗。

这里钉死修复后的范围:

* 零 level + initial-center 的出生/挥拍收据**必须**能解码;
* 关掉那个开关的生产方在同样的零 level 上发的仍然是 ``interior``/``frontier``,
  判决**一个字没变**;
* 让路只认这张收据自己的数字能证明的那一档 —— 四件事缺一不可;
* 身份哈希仍然是兜底:把 stratum 改成"合法但不是它自己发的那个",照样死在
  ``birth_id does not match canonical identity``。
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


_BASE_PATH = Path(__file__).resolve().parent / "test_action_ball_sampling.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "action_ball_sampling_suite_for_point_support_test", _BASE_PATH
)
assert _BASE_SPEC is not None and _BASE_SPEC.loader is not None
BASE = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = BASE
_BASE_SPEC.loader.exec_module(BASE)

S = BASE.S
_profile = BASE._profile
_levels = BASE._levels
_birth = BASE._birth
_sample = BASE._sample

UID = 101
CYCLE = 5


def _mixture_sampler(*, initial_center: bool, seed=20260807):
    return S.ActionBallSampler(
        [_profile()],
        seed=seed,
        sampling_mixture=S.SamplingMixture(),
        contact_time_step_s=0.02,
        initial_center_single_question=initial_center,
    )


def _zero_levels():
    levels = _levels()
    assert all(getattr(levels, arm) == 0.0 for arm in S.ARM_KEYS)
    return levels


def _wide_levels():
    return _levels(
        position=0.5,
        speed=0.5,
        spin_magnitude=0.5,
        spin_direction=0.5,
        aim=0.5,
        base_spawn=0.5,
        base_travel=0.5,
        time_to_contact=0.5,
    )


def _scheduled(index):
    return S.SamplingMixture().stratum_for(index)


# --------------------------------------------------------------------------
# 1. The bug: the four-grid's own configuration could not decode its receipts.
# --------------------------------------------------------------------------


def test_level_zero_initial_center_births_and_samples_decode():
    sampler = _mixture_sampler(initial_center=True)
    levels = _zero_levels()

    strata = []
    for index in range(CYCLE):
        birth = _birth(sampler, epoch=index, levels=levels)
        sample = _sample(sampler, birth, epoch=index, levels=levels)
        strata.append(birth.sampling_stratum)

        assert (
            S.BaseBirthReceipt.from_identity_receipt(
                birth.to_state_dict()
            )
            == birth
        )
        assert (
            S.BallBaseSample.from_identity_receipt(
                {
                    "sample_id": sample.sample_id,
                    **sample.identity_payload(),
                }
            ).sample_id
            == sample.sample_id
        )

    # Every birth collapsed to the point plan, and the 1/3/1 schedule wanted
    # something else for four of the five slots -- that disagreement is the
    # whole reason the decoders used to refuse.
    assert strata == ["center"] * CYCLE
    assert [_scheduled(index) for index in range(CYCLE)] != strata


# --------------------------------------------------------------------------
# 2. Equal strength: the schedule still rules everywhere it means anything.
# --------------------------------------------------------------------------


def test_level_zero_without_initial_center_is_judged_exactly_as_before():
    """Same zero levels, flag off: the producer follows the schedule."""

    sampler = _mixture_sampler(initial_center=False)
    levels = _zero_levels()

    for index in range(CYCLE):
        birth = _birth(sampler, epoch=index, levels=levels)
        assert birth.sampling_stratum == _scheduled(index)
        assert (
            S.BaseBirthReceipt.from_identity_receipt(
                birth.to_state_dict()
            )
            == birth
        )


def test_a_widened_run_may_not_claim_the_collapse():
    """center at a non-center slot without the zero-level fact stays refused."""

    sampler = _mixture_sampler(initial_center=False)
    levels = _wide_levels()
    index = next(
        slot for slot in range(CYCLE) if _scheduled(slot) != "center"
    )
    for _ in range(index):
        _birth(sampler, levels=levels)
    birth = _birth(sampler, levels=levels)
    assert birth.birth_index == index
    assert birth.sampling_stratum != "center"

    forged = deepcopy(birth.to_state_dict())
    forged["sampling_stratum"] = "center"
    forged["frontier_arm"] = None
    forged["sampling_levels"] = S.DomainLevels().as_dict()

    with pytest.raises(
        ValueError,
        match="birth sampling_stratum disagrees with mixture schedule",
    ):
        S.BaseBirthReceipt.from_identity_receipt(forged)


def test_the_identity_hash_is_still_the_backstop_under_the_collapse():
    """Let the carve-out through and the canonical birth_id still catches it."""

    sampler = _mixture_sampler(initial_center=True)
    levels = _zero_levels()
    index = next(
        slot for slot in range(CYCLE) if _scheduled(slot) != "center"
    )
    for _ in range(index):
        _birth(sampler, levels=levels)
    birth = _birth(sampler, levels=levels)
    assert birth.sampling_stratum == "center"

    forged = deepcopy(birth.to_state_dict())
    forged["sampling_stratum"] = _scheduled(index)
    if forged["sampling_stratum"] == "frontier":
        forged["frontier_arm"] = "base_spawn_x_lower"

    with pytest.raises(
        ValueError,
        match="birth_id does not match canonical identity",
    ):
        S.BaseBirthReceipt.from_identity_receipt(forged)


# --------------------------------------------------------------------------
# 3. The carve-out is a four-way conjunction; drop any one and it is gone.
# --------------------------------------------------------------------------


def _collapse(**overrides):
    payload = {
        "domain_levels": S.DomainLevels(),
        "sampling_stratum": "center",
        "sampling_levels": S.DomainLevels(),
        "frontier_arm": None,
    }
    payload.update(overrides)
    return S._point_support_stratum_collapse(**payload)


def test_point_support_collapse_needs_every_one_of_its_four_facts():
    assert _collapse() is True
    assert _collapse(sampling_stratum="interior") is False
    assert _collapse(frontier_arm="base_spawn_x_lower") is False
    assert (
        _collapse(sampling_levels=_levels(base_spawn=1.0e-9)) is False
    )
    assert (
        _collapse(domain_levels=_levels(base_spawn=1.0e-9)) is False
    )
    # "Almost zero" is not zero: the contract says exact zero.
    assert (
        _collapse(domain_levels=_levels(speed=5.0e-324)) is False
    )
