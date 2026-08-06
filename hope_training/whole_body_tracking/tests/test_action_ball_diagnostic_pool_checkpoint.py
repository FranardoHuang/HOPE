"""End-to-end: can a diagnostic (live-births-only) pool actually save?

人话:诊断跑(``diagnostic_unauthorized`` 且非 fixed-view)走的是一条"快路",
它**故意**不写逐提案的那几本账 —— 2-bit lifecycle、per-birth 的 proposed 计数、
sample_assignments、transcript 哈希链。那正是这条路存在的理由:那些 JSON/SHA/
dataclass 成本每次 reset 都要付,却没人消费。

代价是:``LazyActionTaskPool.state_dict()`` 原本逐条对的就是那几本账,于是
诊断跑**一次盘都存不下来** —— 发出第一条任务之后就必炸,连退休都还没发生。
A211/C211 的 scale4096 就是这么死在 update 0 的存盘上的。

本模块用**真的** ``LazyActionTaskPool`` + 真的 ``ActionBirthBroker`` 跑完整
旅程(发任务 -> 真 reset 退休 -> 再发任务 -> 存盘 -> 拒绝续跑),而不是把方法体
搬到裸类上跑。之前那一轮的验收正是缺了这一格:AST 壳测试全绿,存盘路依然是死的。

每条守卫都配一条变异测试,而且变异量刻意做成"粗一个档次的检查就抓不到":
用的是 ±1 的漂移,不是把账清空。
"""

import importlib.util
from pathlib import Path
import sys

import pytest


_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_action_ball_runtime as T  # noqa: E402

R = T.R


class _DelegatingSolver(T.Solver):
    """The stub a diagnostic pool is actually allowed to bind.

    人话:诊断池子不留逐出生 transcript,所以它的 solver 必须自陈
    "transcript 归池子管"。真实的 ``RacketTargetCommand`` 在
    ``diagnostic_unauthorized`` 下正是这么设的。
    """

    pool_owns_birth_task_transcripts = True


def _diagnostic_pool(env_count=2):
    """Real broker + real diagnostic pool, with `env_count` live births."""

    broker, _provider = T._broker(1, diagnostic_unauthorized=True)
    births = [
        T._reserve(broker, env_id=env_id) for env_id in range(env_count)
    ]
    for birth in births:
        T._commit(broker, birth)
    broker.consume_many_true_reset(
        tuple(T._claim(birth) for birth in births)
    )
    pool = R.LazyActionTaskPool(
        T._bindings(1),
        T._pins(),
        "no_move",
        refill_size=1,
        diagnostic_unauthorized=True,
    )
    solver = _DelegatingSolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)
    return pool, solver, broker, births


def _issue(pool, births):
    return pool.request_many(
        tuple(R.ActionTaskIssueRequest(birth, 0) for birth in births)
    )


def _uid(pool):
    return pool._bindings[0].action_uid


# --------------------------------------------------------------------------
# The bug this module exists for.
# --------------------------------------------------------------------------


def test_diagnostic_pool_saves_after_issuing_tasks():
    """This is the exact call that killed A211/C211 at update 0."""

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)

    state = pool.state_dict()

    assert state["pool_state_scope"] == R.POOL_STATE_SCOPE_DIAGNOSTIC
    assert state["schema_version"] == R.POOL_STATE_SCHEMA_VERSION


def test_diagnostic_pool_saves_across_a_true_reset():
    """A retirement used to desync the pool count from the solver's."""

    pool, _solver, broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    retired = births[0]
    pool.retire_birth(retired)
    replacement = T._reserve(
        broker, env_id=retired.env_id, generation=2
    )
    T._consume(broker, replacement)
    pool.request(replacement, swing_generation=0)

    state = pool.state_dict()
    assert state["pool_state_scope"] == R.POOL_STATE_SCOPE_DIAGNOSTIC


def test_diagnostic_save_is_repeatable_and_byte_identical():
    """Saving must not mutate the pool or the solver it cross-checks."""

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)

    first = pool.state_dict()
    second = pool.state_dict()

    assert first == second


def test_diagnostic_payload_omits_the_books_it_never_wrote():
    """A missing ledger must read as missing, not as a zero.

    人话:诊断跑的高水位可能已经几百,而 lifecycle 一行都没有。写一个
    "长度 0" 会把"这本账不存在"伪装成"这本账是空的"。宁可不写这几项。
    """

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)

    action = pool.state_dict()["actions"][0]
    assert "lifecycle_2bit_base64" not in action
    assert "lifecycle_sample_count" not in action
    assert "lifecycle_sha256" not in action
    for row in action["births"]:
        assert "proposed_count" not in row
        assert "sample_assignments" not in row
        assert "issued_task_transcript_sha256" not in row
    # ...but the books it *does* write are all there.
    assert action["ledger"]["admitted"] >= 1
    assert action["last_sample_index"] >= 0


# --------------------------------------------------------------------------
# Resume stays refused -- fail closed, with a reason a human can read.
# --------------------------------------------------------------------------


def test_diagnostic_checkpoint_refuses_to_restore_a_pool():
    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    state = pool.state_dict()

    fresh, _s, _b, _births = _diagnostic_pool()
    with pytest.raises(R.ActionBallContractError, match="launch fresh"):
        fresh.load_state_dict(state)


def test_exact_pool_refuses_a_diagnostic_checkpoint():
    """The missing half-ledger must not silently become zero."""

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    state = pool.state_dict()

    exact = R.LazyActionTaskPool(
        T._bindings(1), T._pins(), "no_move", refill_size=1
    )
    exact.bind_solver(T.Solver())
    with pytest.raises(
        R.ActionBallContractError, match="scope mismatch"
    ):
        exact.load_state_dict(state)


def test_diagnostic_pool_refuses_an_exact_checkpoint():
    exact_pool, _solver, _births = T._formal_pool_batch(2)
    exact_pool.request_many(
        tuple(R.ActionTaskIssueRequest(b, 0) for b in _births)
    )
    state = exact_pool.state_dict()
    assert state["pool_state_scope"] == R.POOL_STATE_SCOPE_EXACT

    diagnostic, _s, _b, _bi = _diagnostic_pool()
    with pytest.raises(
        R.ActionBallContractError, match="scope mismatch"
    ):
        diagnostic.load_state_dict(state)


def test_unknown_scope_brand_is_refused():
    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    state = pool.state_dict()
    state["pool_state_scope"] = "whatever_the_next_agent_invents"

    with pytest.raises(R.ActionBallContractError, match="unknown"):
        pool.load_state_dict(state)


# --------------------------------------------------------------------------
# The exact-scope path must be untouched.
# --------------------------------------------------------------------------


def test_exact_pool_still_carries_every_per_proposal_book():
    """Rescoping the diagnostic branch must not thin out the exact branch.

    人话:完整的 exact 存档/续跑回环由 ``test_action_ball_runtime.py`` 覆盖
    (那里会把 broker 也一起复原)。这里只钉住形状:正式跑的存档必须仍然
    带着诊断跑省掉的那几本账,而且牌子是 ``exact_per_birth``。
    """

    pool, _solver, births = T._formal_pool_batch(2)
    pool.request_many(
        tuple(R.ActionTaskIssueRequest(b, 0) for b in births)
    )
    state = pool.state_dict()

    assert state["pool_state_scope"] == R.POOL_STATE_SCOPE_EXACT
    action = state["actions"][0]
    assert "lifecycle_2bit_base64" in action
    assert "lifecycle_sample_count" in action
    assert "lifecycle_sha256" in action
    assert "proposed_count" in action["births"][0]
    assert "sample_assignments" in action["births"][0]
    assert "issued_task_transcript_sha256" in action["births"][0]
    assert pool.state_dict() == state


# --------------------------------------------------------------------------
# Pairing gate: a diagnostic pool may not bind a non-delegating solver.
# --------------------------------------------------------------------------


def test_diagnostic_pool_names_a_non_delegating_solver_at_save():
    """This misconfiguration used to surface as a bare KeyError."""

    broker, _provider = T._broker(1, diagnostic_unauthorized=True)
    births = [T._reserve(broker, env_id=env_id) for env_id in range(2)]
    for birth in births:
        T._commit(broker, birth)
    broker.consume_many_true_reset(
        tuple(T._claim(birth) for birth in births)
    )
    pool = R.LazyActionTaskPool(
        T._bindings(1),
        T._pins(),
        "no_move",
        refill_size=1,
        diagnostic_unauthorized=True,
    )
    pool.bind_solver(T.Solver())  # does NOT delegate
    pool.bind_birth_authority(broker)
    _issue(pool, births)

    with pytest.raises(
        R.ActionBallContractError,
        match="pool_owns_birth_task_transcripts",
    ):
        pool.state_dict()


def test_exact_pool_may_still_bind_a_delegating_solver():
    """The banded question bank does exactly this; it must keep working."""

    pool = R.LazyActionTaskPool(
        T._bindings(1), T._pins(), "no_move", refill_size=1
    )
    pool.bind_solver(_DelegatingSolver())


# --------------------------------------------------------------------------
# Mutation tests: prove each replacement guard really fires.
#
# 人话:每处漂移都只做 ±1。粗一个档次的检查(比如只查"非空"、只查不等号
# 方向、或者只在退休时才对账)全都抓不到这些。
# --------------------------------------------------------------------------


def test_mutation_proposed_count_drifting_by_one_is_caught():
    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    uid = _uid(pool)
    current = pool._ledger[uid]
    pool._ledger[uid] = R.PoolLedger(
        requests=current.requests,
        refill_calls=current.refill_calls,
        proposed=current.proposed + 1,
        admitted=current.admitted,
        issued=current.issued,
        discarded=current.discarded,
    )
    with pytest.raises(
        R.ActionBallContractError,
        match="cover sample indices",
    ):
        pool.state_dict()


def test_mutation_sample_highwater_drifting_by_one_is_caught():
    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    uid = _uid(pool)
    pool._last_sample_index[uid] += 1
    with pytest.raises(R.ActionBallContractError):
        pool.state_dict()


def test_mutation_admitted_decomposition_drifting_by_one_is_caught():
    """admitted must equal issued + discarded + still-pending, exactly.

    人话:这个变异刻意做成 ``PoolLedger`` 自己的构造校验挑不出毛病的样子
    (proposed >= admitted >= issued + discarded 全部满足),而且采样高水位
    也跟着搬了一格,所以只剩"收进来的任务去哪了"这一条等式被破坏。粗一档
    的检查(只查不等号方向)抓不到它。
    """

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    uid = _uid(pool)
    current = pool._ledger[uid]
    pool._ledger[uid] = R.PoolLedger(
        requests=current.requests,
        refill_calls=current.refill_calls,
        proposed=current.proposed + 1,
        admitted=current.admitted + 1,
        issued=current.issued,
        discarded=current.discarded,
    )
    pool._last_sample_index[uid] += 1
    with pytest.raises(
        R.ActionBallContractError,
        match="issued plus discarded plus still-pending",
    ):
        pool.state_dict()


def test_pool_ledger_is_the_sole_source_of_two_invariants():
    """Don't re-check what PoolLedger already refuses to construct.

    人话:``requests == issued`` 和 ``proposed >= admitted >= issued +
    discarded`` 由 ``PoolLedger.__post_init__`` 独家把守,所以存盘那层
    再查一遍永远不会开火。这条测试把"唯一真源在哪"钉住,免得以后有人
    在存盘处补一条永远绿的检查回来。
    """

    with pytest.raises(
        R.ActionBallContractError, match="requests == issued"
    ):
        R.PoolLedger(requests=1, proposed=1, admitted=1, issued=0)
    with pytest.raises(
        R.ActionBallContractError,
        match="proposed >= admitted >= issued",
    ):
        R.PoolLedger(requests=0, proposed=0, admitted=1, issued=0)


def test_mutation_solver_emitted_count_drifting_by_one_is_caught():
    """The cross-check against solver authority must survive rescoping.

    人话:这是上一轮真正想保住的那条 —— 池子和 solver 的累计发放数必须相等。
    换账本之后它仍然必须开火。
    """

    pool, solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    # 少一条就够,不需要把账清空。
    solver.emitted_tasks.pop()
    with pytest.raises(
        R.ActionBallContractError,
        match="admitted-task counts differ from solver authority",
    ):
        pool.state_dict()


def test_mutation_solver_emitted_count_drift_after_retire_is_caught():
    """The retire path is where the old, wrongly-scoped check misfired."""

    pool, solver, broker, births = _diagnostic_pool()
    _issue(pool, births)
    retired = births[0]
    pool.retire_birth(retired)
    replacement = T._reserve(
        broker, env_id=retired.env_id, generation=2
    )
    T._consume(broker, replacement)
    pool.request(replacement, swing_generation=0)
    pool.state_dict()

    solver.emitted_tasks.pop()
    with pytest.raises(
        R.ActionBallContractError,
        match="admitted-task counts differ from solver authority",
    ):
        pool.state_dict()


def test_mutation_writing_a_per_proposal_book_is_caught():
    """The rescoping is only sound while these tables stay empty."""

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    pool._task_lifecycle[_uid(pool)] = [0]
    with pytest.raises(
        R.ActionBallContractError,
        match="must not populate the per-proposal table",
    ):
        pool.state_dict()


def test_mutation_writing_a_retired_birth_record_is_caught():
    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    pool.state_dict()

    pool._retired_births[_uid(pool)] = {}
    with pytest.raises(
        R.ActionBallContractError,
        match="must not populate the per-proposal table",
    ):
        pool.state_dict()


def test_mutation_scope_brand_swap_breaks_integrity():
    """The brand is inside the integrity hash, not bolted on beside it."""

    pool, _solver, _broker, births = _diagnostic_pool()
    _issue(pool, births)
    state = pool.state_dict()
    state["pool_state_scope"] = R.POOL_STATE_SCOPE_EXACT

    exact = R.LazyActionTaskPool(
        T._bindings(1), T._pins(), "no_move", refill_size=1
    )
    exact.bind_solver(T.TranscriptProbeSolver())
    with pytest.raises(
        R.ActionBallContractError, match="integrity mismatch"
    ):
        exact.load_state_dict(state)
