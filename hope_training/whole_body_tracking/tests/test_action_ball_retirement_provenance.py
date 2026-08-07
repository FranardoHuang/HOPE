"""谁在守"存档不许凭空编出一份退役来历"。

人话:``LazyActionTaskPool`` 的存档里有两样东西描述"这个 env 退到第几代了":
一本 ``retired_generations`` 台账,和一批 ``retired_births`` 记录。
``load_state_dict`` 会先比这两样对不对得上 —— 但它们**同源**,同一个文件里的
两处,一起改就一起过。真正的证人是下一步:每份 retired birth 都要拿去问
broker "这就是你当初记下的那一份收据吗"(``assert_consumed_birth``)。

2026-08-07 判决:同一个 broker 上还挂着一个从来没人调用过的
``assert_known_generation``,它只问"这个 env/代次你发过吗"。它比现役那道门
**粗一个档次**——代次是真的、内容被换过的伪造它会放行。所以删掉,不是接线。
本模块把这条判决钉住:

1. 该拦的仍拦:凭空编一个 broker 从没发过的代次的退役记录 -> 必须被拒,
   而且必须是被 ``assert_consumed_birth`` 拒的(对错误原文断言,不是"反正红了")。
2. 粗一档就过不了:把那道被删的门原样复刻在测试里当**变异体**,
   喂它一份"代次真、内容假"的收据 —— 变异体放行,现役那道门拒绝。
3. 现役那道门通过时,被删那道门必然也通过(所以删它没丢任何拦截面):
   broker 自己的 load 强制 ``consumed <= last``。
4. 它确实没了,也没人偷偷加回来。
"""

import importlib.util  # noqa: F401  (与同目录其它模块保持同一 bootstrap 形状)
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest


_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_action_ball_runtime as T  # noqa: E402

R = T.R


def _exact_pool(broker):
    pool = R.LazyActionTaskPool(
        T._bindings(1), T._pins(), "no_move", refill_size=2
    )
    pool.bind_solver(T.Solver())
    pool.bind_birth_authority(broker)
    return pool


def _retired_checkpoint():
    """env0 gen1: reserve -> consume -> request -> retire,正式(exact)存档。"""

    broker, _provider = T._broker(1)
    birth = T._reserve(broker, env_id=0, generation=1)
    T._consume(broker, birth)
    pool = _exact_pool(broker)
    pool.request(birth, swing_generation=0)
    pool.retire_birth(birth)
    state = deepcopy(pool.state_dict())
    assert state["pool_state_scope"] == R.POOL_STATE_SCOPE_EXACT
    assert state["retired_generations"] == [[0, 1]]
    return broker, state, birth


def _forge_unissued_retirement(state, birth, *, generation):
    """把那条退役记录整体改写成一个 broker 从没发过的代次。"""

    forged_birth = replace(birth, reset_generation=generation)
    forged = deepcopy(state)
    row = forged["actions"][0]["retired_births"][0]
    row["birth"] = forged_birth.to_dict()
    row["task_transcript_sha256"] = R.task_transcript_sha256(
        forged_birth.canonical_sha256, ()
    )
    forged["retired_generations"] = [[0, generation]]
    forged["integrity_sha256"] = T._integrity(forged)
    return forged, forged_birth


# --------------------------------------------------------------------------
# 1. 该拦的仍拦
# --------------------------------------------------------------------------


def test_honest_retired_checkpoint_round_trips():
    """先证明这条路本来是通的,免得下面的红色是"哪都跑不通"。"""

    broker, state, _birth = _retired_checkpoint()
    _exact_pool(broker).load_state_dict(deepcopy(state))


def test_checkpoint_cannot_invent_a_generation_the_broker_never_issued():
    """broker 只发过 env0 gen1;存档声称退到了 gen2。"""

    broker, state, birth = _retired_checkpoint()
    forged, _forged_birth = _forge_unissued_retirement(
        state, birth, generation=2
    )
    with pytest.raises(
        R.BirthProtocolError,
        match="birth is not the env's exact consumed generation",
    ):
        _exact_pool(broker).load_state_dict(forged)


def test_bumping_only_the_retired_ledger_is_caught_by_the_same_load():
    """只动台账、不动记录 —— 这一条是同源自证那一步抓的,记下它的原话。"""

    broker, state, _birth = _retired_checkpoint()
    forged = deepcopy(state)
    forged["retired_generations"] = [[0, 7]]
    forged["integrity_sha256"] = T._integrity(forged)
    with pytest.raises(
        R.ActionBallContractError,
        match="retired generation ledger differs from compact retired",
    ):
        _exact_pool(broker).load_state_dict(forged)


def test_the_refusal_really_comes_from_the_broker_witness():
    """变异:把 broker 那个证人拔掉,伪造必须换一层才被抓,老实存档仍要能载入。

    人话:这一条是防"其实是别的解码器顺手抓到的,和 broker 无关"。
    拔掉之后伪造仍然红,但红的原因换成了 solver 那本提案账 —— 说明
    ``assert_consumed_birth`` 确实是这条链上专门管来历的那一环,
    后面还压着第三个证人。
    """

    broker, state, birth = _retired_checkpoint()
    forged, _forged_birth = _forge_unissued_retirement(
        state, birth, generation=2
    )
    saved = R.ActionBirthBroker.assert_consumed_birth
    try:
        R.ActionBirthBroker.assert_consumed_birth = (
            lambda self, birth: None
        )
        with pytest.raises(
            ValueError,
            match="proposal sample was not assigned to exact birth/refill",
        ):
            _exact_pool(broker).load_state_dict(forged)
        # 控制组:变异体本身没有把老实存档也弄红。
        _exact_pool(broker).load_state_dict(deepcopy(state))
    finally:
        R.ActionBirthBroker.assert_consumed_birth = saved


# --------------------------------------------------------------------------
# 2. 粗一个档次的门就过不了
# --------------------------------------------------------------------------


def _deleted_coarse_gate(broker, *, env_id, reset_generation):
    """被删掉的 ``assert_known_generation`` 的原样复刻,作为变异体。

    原实现只有这一句:``if self._last_generation.get(env, 0) < generation``。
    """

    if broker._last_generation.get(env_id, 0) < reset_generation:
        raise R.BirthProtocolError(
            "generation is absent from the birth broker transcript"
        )


def test_the_deleted_gate_would_have_waved_a_content_swap_through():
    """代次是真的、内容被换过 —— 粗门放行,现役门拒绝。

    这是删掉它而不是接上它的全部理由:接上去只会在这条链上加一道
    比已有那道更弱的检查,让读代码的人以为来历多了一层保障。
    """

    broker, _provider = T._broker(3)
    real = T._reserve(broker, env_id=0, generation=1, slot=0)
    T._consume(broker, real)
    other = T._reserve(broker, env_id=1, generation=1, slot=1)
    T._consume(broker, other)
    # 一份"贴着 env0 gen1 标签、内容其实是另一格出生"的收据。
    swapped = replace(other, env_id=0)
    assert swapped.reset_generation == 1
    assert swapped != real

    # 变异体(被删的那道门):放行。
    _deleted_coarse_gate(broker, env_id=0, reset_generation=1)

    # 现役那道门:拒绝。
    with pytest.raises(
        R.BirthProtocolError,
        match="birth is not the env's exact consumed generation",
    ):
        broker.assert_consumed_birth(swapped)


# --------------------------------------------------------------------------
# 3. 删它没丢任何拦截面
# --------------------------------------------------------------------------


def test_consumed_witness_implies_the_deleted_gate_on_a_live_broker():
    """凡是现役门放行的 (env, 代次),粗门也一定放行;反过来不成立。"""

    broker, _provider = T._broker(3)
    receipts = []
    for env_id in (0, 1):
        for generation in (1, 2):
            birth = T._reserve(
                broker,
                env_id=env_id,
                generation=generation,
                slot=env_id,
            )
            T._consume(broker, birth)
            receipts.append(birth)
    # 第三格只预约、不消费:这正是粗门放行而现役门拒绝的那一格。
    pending = T._reserve(broker, env_id=2, generation=1, slot=2)

    strictly_weaker_somewhere = False
    for birth in receipts + [pending]:
        try:
            broker.assert_consumed_birth(birth)
            consumed_ok = True
        except R.BirthProtocolError:
            consumed_ok = False
        try:
            _deleted_coarse_gate(
                broker,
                env_id=birth.env_id,
                reset_generation=birth.reset_generation,
            )
            coarse_ok = True
        except R.BirthProtocolError:
            coarse_ok = False
        assert not consumed_ok or coarse_ok, (
            f"env{birth.env_id} gen{birth.reset_generation}: "
            "现役门放行但粗门拒绝 —— 删除的前提被推翻了"
        )
        strictly_weaker_somewhere |= coarse_ok and not consumed_ok
    assert strictly_weaker_somewhere, (
        "这组样本没覆盖到'粗门更松'的那一格,断言就没有判别力了"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda state: state.__setitem__(
                "consumed_generations", [[0, 5]]
            ),
            id="consumed-ahead-of-last",
        ),
        pytest.param(
            lambda state: state.__setitem__("last_generations", [[0, 1]]),
            id="last-behind-consumed",
        ),
        pytest.param(
            lambda state: state.__setitem__(
                "consumed_generations", [[0, 2], [3, 1]]
            ),
            id="consumed-env-with-no-last-row",
        ),
    ],
)
def test_broker_load_forbids_consumed_running_ahead_of_issued(mutate):
    """上一条的机制来源:broker 自己不许 consumed 跑到 last 前面。

    这是"现役门通过 => 粗门通过"在跨进程续跑下也成立的理由。
    """

    broker, _provider = T._broker(1)
    for generation in (1, 2):
        birth = T._reserve(broker, env_id=0, generation=generation)
        T._consume(broker, birth)
    state = deepcopy(broker.state_dict())

    forged = deepcopy(state)
    mutate(forged)
    forged["integrity_sha256"] = R._sha256_json(
        {k: v for k, v in forged.items() if k != "integrity_sha256"}
    )
    fresh, _ = T._broker(1)
    with pytest.raises(
        R.ActionBallContractError,
        match="consumed generation exceeds last generation",
    ):
        fresh.load_state_dict(forged)


# --------------------------------------------------------------------------
# 4. 它确实没了
# --------------------------------------------------------------------------


def test_the_coarse_gate_is_gone_and_the_owner_is_still_here():
    assert not hasattr(R.ActionBirthBroker, "assert_known_generation"), (
        "assert_known_generation 被加回来了。先读本模块:它比 "
        "assert_consumed_birth 粗一个档次,接上去是降级不是加固。"
    )
    assert callable(R.ActionBirthBroker.assert_consumed_birth)
