"""Which per-event bookkeeping a sampler keeps, and what may be reconciled against it.

人话:诊断跑(A211/C211 那种 4096 格的)每发一个样本**故意**不写 ``sample -> birth``
的 assignment 行、也不推 ``assignment_head`` 哈希链,出生表只留"还活着的那些"。存
checkpoint 时 :meth:`ActionBallSampler.state_dict` 却拿这本故意空的账去核对一直在涨的
``sample_count``,于是每个诊断跑都在 update 0 之后存盘那一刻炸掉
(``sample authority ledger is inconsistent with retired/sample counts``)。

错的是**对账范围**,不是那个计数器 —— 它有真实的生产者(每个样本一次),而且这个模式
自己还维护着另一本能证伪它的账:每次出生恰好吃 ``DRAWS_PER_BIRTH`` 个随机数、每个样本
恰好吃 ``DRAWS_PER_SAMPLE`` 个,而诊断退休按定义不动 RNG 也不动 retired 前缀。所以
per-action 的 ``draw_count`` 仍然把两个计数器钉死,多一个少一个都要红。``load_state_dict``
本来就是拿这同一条格子去审精确档的状态包。

这里钉死修复后的范围:

* 诊断跑(live-births-only)干净状态**必须**能序列化;
* 但它的计数器仍然要对账 —— 换成对"每动作随机带"对账,多一条少一条都必须红;
* 精确跑那两条严格对账**一个字没动**,精确跑里造一次真实漂移必须仍然被拒;
* 状态包自陈 scope 并且这块牌子进签名;两种 scope 的状态包不能互相 resume,而且
  live-births-only 一旦发过样本就根本不能拿去做精确续跑(不可变 fixed-view 那种
  "只有出生、零样本"的仍然能完整复原)。

每个测试都写成"检查粗一个档次就通不过"的形状:把范围守卫去掉(``_diagnostic_fast_path``
翻回 False)、把替换后的随机带对账停掉、把 scope 牌子降级成"只要是个已知值就行"、把
"发过样本就不许续跑"那条去掉,都会让下面某一条变红。
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


# Reuse the profile/levels/birth/sample builders the sampler suite already
# owns.  A second hand-written fixture would make the expectations below a
# third copy of the contract instead of a reading of it.
_BASE_PATH = Path(__file__).resolve().parent / "test_action_ball_sampling.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "action_ball_sampling_suite_for_scope_test", _BASE_PATH
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
SCOPE_EXACT = S.SAMPLER_TRANSCRIPT_SCOPE_EXACT
SCOPE_DIAGNOSTIC = S.SAMPLER_TRANSCRIPT_SCOPE_DIAGNOSTIC


def _diagnostic_sampler(seed=20260807):
    return S.ActionBallSampler(
        [_profile()],
        seed=seed,
        diagnostic_unauthorized=True,
    )


def _exact_sampler(seed=20260807):
    return S.ActionBallSampler([_profile()], seed=seed)


def _live_diagnostic_run(env_count=6, generations=3):
    """Reproduce the shape a 4096-env diagnostic run has at checkpoint time.

    Every generation retires the previous births through the diagnostic seam
    and immediately issues a replacement birth + swing, so ``birth_count`` and
    ``sample_count`` keep climbing while the retained birth map stays exactly
    one row per environment and the assignment ledger stays empty.
    """

    sampler = _diagnostic_sampler()
    live = {}
    for env_id in range(env_count):
        birth = _birth(sampler, epoch=1)
        _sample(sampler, birth, epoch=1)
        live[env_id] = birth
    for generation in range(2, 2 + generations):
        # A partial, rotating subset -- the same asynchronous reset pattern the
        # runtime produces, not a whole-population barrier.
        env_ids = tuple(
            (generation * 3 + offset) % env_count
            for offset in range(env_count - 2)
        )
        sampler.forget_diagnostic_births(
            tuple(live[env_id] for env_id in env_ids)
        )
        for env_id in env_ids:
            birth = _birth(sampler, epoch=generation)
            _sample(sampler, birth, epoch=generation)
            live[env_id] = birth
    return sampler, live


def _exact_run(count=5):
    sampler = _exact_sampler()
    births = []
    for index in range(count):
        birth = _birth(sampler, epoch=index)
        _sample(sampler, birth, epoch=index)
        births.append(birth)
    return sampler, births


def _resign(state, **overrides):
    row = dict(state)
    row.update(overrides)
    payload = {key: row[key] for key in S._STATE_KEYS[:-1]}
    row["integrity_sha256"] = S._sha256_json(payload)
    return row


# --------------------------------------------------------------------------
# 1. The bug: a clean live-births-only run must be able to save.
# --------------------------------------------------------------------------


def test_live_births_only_run_can_serialize_and_says_so():
    sampler, live = _live_diagnostic_run()

    state = sampler.state_dict()

    assert state["transcript_scope"] == SCOPE_DIAGNOSTIC
    assert sampler.transcript_scope == SCOPE_DIAGNOSTIC
    # The assignment ledger is deliberately empty and the chain never moved.
    assert state["issued_sample_birth_indices"][str(UID)] == []
    assert (
        state["per_action"][str(UID)]["assignment_head_sha256"]
        == state["per_action"][str(UID)]["retired_assignment_head_sha256"]
    )
    # ...while the counters it could not be reconciled against kept climbing.
    assert state["per_action"][str(UID)]["sample_count"] > len(live)
    assert state["per_action"][str(UID)]["birth_count"] > len(live)
    # One retained birth row per live environment, none of them retired.
    assert len(state["issued_births"][str(UID)]) == len(live)
    assert state["per_action"][str(UID)]["retired_birth_count"] == 0
    assert state["per_action"][str(UID)]["retired_sample_count"] == 0
    assert {
        row["birth_id"] for row in state["issued_births"][str(UID)]
    } == {birth.birth_id for birth in live.values()}
    # Serialization is pure.
    assert sampler.state_dict() == state


def test_exact_run_is_unchanged_and_still_round_trips():
    sampler, _births = _exact_run()

    state = deepcopy(sampler.state_dict())

    assert state["transcript_scope"] == SCOPE_EXACT
    assert sampler.transcript_scope == SCOPE_EXACT
    assert state["issued_sample_birth_indices"][str(UID)] == [0, 1, 2, 3, 4]

    restored = _exact_sampler()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_exact_issued_birth_rows_stay_in_retired_to_highwater_order():
    """The payload still emits exactly ``range(retired_birth, birth_count)``."""

    sampler, _births = _exact_run(count=6)
    sampler.compact_retired_prefix(
        [
            S.SamplerRetirePrefixBarrier(
                action_uid=UID,
                retire_birth_through_inclusive=2,
                retire_sample_through_inclusive=2,
                expected_birth_highwater=sampler.birth_highwater_for(UID),
                expected_sample_highwater=sampler.sample_highwater_for(UID),
                expected_assignment_head_sha256=(
                    sampler.assignment_head_for(UID)
                ),
            )
        ]
    )

    state = sampler.state_dict()
    retired_births, _retired_samples = sampler.retired_prefix_for(UID)

    assert retired_births == 3
    assert [
        row["birth_index"] for row in state["issued_births"][str(UID)]
    ] == list(range(retired_births, sampler.birth_count_for(UID)))


# --------------------------------------------------------------------------
# 2. Mutation: take the range guard away and the diagnostic run goes red again.
# --------------------------------------------------------------------------


def test_removing_the_scope_guard_reds_the_diagnostic_run_again():
    """``_diagnostic_fast_path=False`` is exactly "delete the range guard"."""

    sampler, _live = _live_diagnostic_run()
    assert sampler.state_dict()["transcript_scope"] == SCOPE_DIAGNOSTIC

    sampler._diagnostic_fast_path = False

    with pytest.raises(
        RuntimeError,
        match="sample authority ledger is inconsistent with retired/sample",
    ):
        sampler.state_dict()


# --------------------------------------------------------------------------
# 3. Equal strength: real drift inside the diagnostic scope is still refused.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delta", (1, -1))
def test_live_births_only_sample_counter_drift_is_still_refused(delta):
    sampler, _live = _live_diagnostic_run()
    sampler.state_dict()

    sampler._sample_count_by_action[UID] += delta

    with pytest.raises(
        RuntimeError,
        match="sample authority ledger is inconsistent with the action draw",
    ):
        sampler.state_dict()


@pytest.mark.parametrize(
    "delta,expected",
    (
        (1, "sample authority ledger is inconsistent with the action draw"),
        # Dropping the counter puts the newest live birth past the high-water,
        # so the bounds witness catches it before the draw tape does.
        (-1, "references a birth index this sampler never issued"),
    ),
)
def test_live_births_only_birth_counter_drift_is_still_refused(
    delta, expected
):
    sampler, _live = _live_diagnostic_run()
    sampler.state_dict()

    sampler._birth_count_by_action[UID] += delta

    with pytest.raises(RuntimeError, match=expected):
        sampler.state_dict()


def test_live_births_only_refuses_a_birth_row_it_never_issued():
    sampler, live = _live_diagnostic_run()
    sampler.state_dict()

    forged_index = sampler.birth_count_for(UID)
    sampler._issued_births_by_action[UID][forged_index] = next(
        iter(live.values())
    )

    with pytest.raises(
        RuntimeError,
        match="references a birth index this sampler never issued",
    ):
        sampler.state_dict()


def test_live_births_only_refuses_an_assignment_row_it_claims_not_to_write():
    sampler, _live = _live_diagnostic_run()
    sampler.state_dict()

    sampler._issued_sample_birth_indices_by_action[UID].append(0)

    with pytest.raises(
        RuntimeError,
        match="retained a per-sample assignment row",
    ):
        sampler.state_dict()


def test_live_births_only_refuses_a_folded_compaction_segment():
    sampler, _live = _live_diagnostic_run()
    state = sampler.state_dict()
    donor, _births = _exact_run(count=4)
    donor.compact_retired_prefix(
        [
            S.SamplerRetirePrefixBarrier(
                action_uid=UID,
                retire_birth_through_inclusive=1,
                retire_sample_through_inclusive=1,
                expected_birth_highwater=donor.birth_highwater_for(UID),
                expected_sample_highwater=donor.sample_highwater_for(UID),
                expected_assignment_head_sha256=(
                    donor.assignment_head_for(UID)
                ),
            )
        ]
    )
    assert state["compaction_segments"][str(UID)] == []

    sampler._compaction_segments_by_action[UID] = list(
        donor._compaction_segments_by_action[UID]
    )

    with pytest.raises(
        RuntimeError,
        match="folded a compaction segment",
    ):
        sampler.state_dict()


@pytest.mark.parametrize(
    "attribute",
    (
        "_retired_birth_count_by_action",
        "_retired_sample_count_by_action",
    ),
)
def test_live_births_only_refuses_an_advanced_retired_prefix(attribute):
    sampler, _live = _live_diagnostic_run()
    sampler.state_dict()

    getattr(sampler, attribute)[UID] = 1

    with pytest.raises(
        RuntimeError,
        match="must not advance a retired birth/sample prefix",
    ):
        sampler.state_dict()


# --------------------------------------------------------------------------
# 4. The exact spelling's strict reconciliation did not move.
# --------------------------------------------------------------------------


def test_exact_run_still_refuses_a_dropped_assignment_row():
    sampler, _births = _exact_run()
    sampler.state_dict()

    sampler._issued_sample_birth_indices_by_action[UID].pop()

    with pytest.raises(
        RuntimeError,
        match="sample authority ledger is inconsistent with retired/sample",
    ):
        sampler.state_dict()


def test_exact_run_still_refuses_a_dropped_birth_row():
    sampler, _births = _exact_run()
    sampler.state_dict()

    del sampler._issued_births_by_action[UID][0]

    with pytest.raises(
        RuntimeError,
        match="retained birth transcript is inconsistent with retired/birth",
    ):
        sampler.state_dict()


def test_exact_run_still_refuses_a_rewritten_assignment_row():
    sampler, _births = _exact_run()
    sampler.state_dict()

    sampler._issued_sample_birth_indices_by_action[UID][-1] = 0

    with pytest.raises(
        RuntimeError,
        match="sample assignment append chain is inconsistent",
    ):
        sampler.state_dict()


def test_exact_run_still_refuses_a_sample_counter_drift():
    sampler, _births = _exact_run()
    sampler.state_dict()

    sampler._sample_count_by_action[UID] += 1

    with pytest.raises(
        RuntimeError,
        match="sample authority ledger is inconsistent with retired/sample",
    ):
        sampler.state_dict()


# --------------------------------------------------------------------------
# 5. Resume: the brand is signed and both directions fail closed.
# --------------------------------------------------------------------------


def test_transcript_scope_is_inside_the_state_signature():
    sampler, _live = _live_diagnostic_run()
    state = deepcopy(sampler.state_dict())

    state["transcript_scope"] = SCOPE_EXACT

    with pytest.raises(
        ValueError,
        match="sampler state integrity check failed",
    ):
        _exact_sampler().load_state_dict(state)


def test_diagnostic_state_cannot_resume_an_exact_sampler():
    sampler, _live = _live_diagnostic_run()
    state = deepcopy(sampler.state_dict())

    with pytest.raises(
        ValueError,
        match="sampler state transcript scope mismatch",
    ):
        _exact_sampler().load_state_dict(state)


def test_exact_state_cannot_resume_a_diagnostic_sampler():
    sampler, _births = _exact_run()
    state = deepcopy(sampler.state_dict())

    with pytest.raises(
        ValueError,
        match="sampler state transcript scope mismatch",
    ):
        _diagnostic_sampler().load_state_dict(state)


def test_a_known_but_wrong_scope_value_is_not_enough():
    """Downgrading the brand check to "is a known value" would pass this."""

    sampler, _births = _exact_run()
    state = _resign(
        deepcopy(sampler.state_dict()),
        transcript_scope=SCOPE_DIAGNOSTIC,
    )

    assert state["transcript_scope"] in S.SAMPLER_TRANSCRIPT_SCOPES
    with pytest.raises(
        ValueError,
        match="sampler state transcript scope mismatch",
    ):
        _exact_sampler().load_state_dict(state)


def test_an_unknown_scope_value_is_refused():
    sampler, _births = _exact_run()
    state = _resign(
        deepcopy(sampler.state_dict()),
        transcript_scope="whatever_the_writer_felt_like",
    )

    with pytest.raises(
        ValueError,
        match="unknown transcript scope",
    ):
        _exact_sampler().load_state_dict(state)


def test_a_state_without_the_brand_is_refused_by_name():
    sampler, _births = _exact_run()
    state = deepcopy(sampler.state_dict())
    del state["transcript_scope"]

    with pytest.raises(
        ValueError,
        match="predates the transcript-scope brand",
    ):
        _exact_sampler().load_state_dict(state)


def test_live_births_only_state_with_samples_cannot_be_resumed():
    sampler, _live = _live_diagnostic_run()
    state = deepcopy(sampler.state_dict())

    assert state["per_action"][str(UID)]["sample_count"] > 0
    with pytest.raises(
        ValueError,
        match="has no per-sample assignment transcript to resume from",
    ):
        _diagnostic_sampler().load_state_dict(state)


def test_births_only_diagnostic_state_still_resumes_exactly():
    """The immutable fixed-view tape reserves births and solves no swings."""

    sampler = _diagnostic_sampler()
    for index in range(4):
        _birth(sampler, epoch=index)
    state = deepcopy(sampler.state_dict())

    assert state["transcript_scope"] == SCOPE_DIAGNOSTIC
    assert state["per_action"][str(UID)]["sample_count"] == 0

    restored = _diagnostic_sampler()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.birth_count_for(UID) == 4
    # And the restored tape keeps issuing the same births.
    assert _birth(restored, epoch=4) == _birth(sampler, epoch=4)
