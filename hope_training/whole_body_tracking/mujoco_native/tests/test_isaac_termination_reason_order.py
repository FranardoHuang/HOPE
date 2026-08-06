"""终止原因的**顺序**也是一份手抄件,而且指纹的覆盖面比它保护的语义面还小。

Isaac 评估终止项的顺序不是 ``hope_env_cfg.py`` 一个文件能决定的:那两个 HOPE 类
最终都派生自 ``tracking_env_cfg.TerminationsCfg``,而 ``configclass`` 是 dataclass
底子 —— 字段顺序先按父类声明序、再接子类新加的字段,子类**覆写**一条不会把它挪到
队尾。所以真实顺序是三个类、两个文件拼出来的:

    time_out, anchor_pos, anchor_ori, ee_body_pos      <- tracking_env_cfg.py
    base_fell_tilt, base_too_low, robot_hit_table      <- hope_env_cfg.py 父类
    joint_qdes_forbidden, joint_actual_forbidden       <- hope_env_cfg.py 子类

同一步里两条终止都成立时,排在前面的那条才是被记进收据的原因 —— 顺序就是"实验把锅
算在谁头上"。复刻侧把这份顺序抄成了四个元组,而 ``base_config`` 选择器只点了
``TerminationsCfg|time_out`` **一个名字**:往根类里加一条项、或者把 ``anchor_pos``
和 ``anchor_ori`` 换个位置,那个指纹一个 bit 都不会动。和 ``ee_body_pos`` 那个窟窿
同形,只是高了一层。

每条检查都配一个"粗一个档次就过不了"的变异:换位(集合与计数全不变)、把一条项从
父类搬进子类(名字、数量、集合统统不变,只有位置变了)、往根类里新加一条(旧选择器
的指纹原地不动,这里在测试里直接断言给你看)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
WBT_ROOT = REPO_ROOT / "hope_training/whole_body_tracking"
if str(WBT_ROOT) not in sys.path:
    sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import isaac_reference_envelope as envelope  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


ACTION_BALL = envelope.ACTION_BALL_TERMINATIONS_CLASS
DEPLOY_PARITY = envelope.DEPLOY_PARITY_TERMINATIONS_CLASS
BASE_CLASS = envelope.BASE_TERMINATIONS_CLASS

#: The narrow selector this round replaced, and the SHA it produced.  Kept here
#: as the *witness* that the old gate was structurally blind -- every mutation
#: below asserts this exact digest is still what the old selector computes.
OLD_NARROW_BASE_SELECTORS = (
    ("class_header", BASE_CLASS),
    ("class_assignments", f"{BASE_CLASS}|time_out"),
)
OLD_NARROW_BASE_SHA256 = (
    "aefdf83d0dbd39144da07cb4c7bcb2eee59c552174e00b8d28747cdde992e49c"
)

#: Two whole terms of the live grandparent, verbatim, so a test can swap them.
LIVE_ANCHOR_POS_BLOCK = """    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
"""
LIVE_ANCHOR_ORI_BLOCK = """    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
"""
LIVE_BASE_TOO_LOW_BLOCK = (
    "    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, "
    'params={"minimum_height": 0.5})\n'
)


def _clear() -> None:
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()
    envelope.clear_caches()


@pytest.fixture(autouse=True)
def _restore_caches():
    _clear()
    yield
    _clear()


def _mutate(tmp_path: Path, source: Path, old: str, new: str) -> Path:
    text = source.read_text("utf-8")
    assert text.count(old) == 1, f"mutation anchor is not unique: {old[:60]!r}"
    target = tmp_path / source.name
    target.write_text(text.replace(old, new), "utf-8")
    _clear()
    return target


def _order(base: Path | None = None, config: Path | None = None) -> tuple:
    return envelope.live_termination_reason_order(
        ACTION_BALL, config_path=config, base_config_path=base
    )


def _blockers(base: Path | None = None, config: Path | None = None) -> tuple:
    return envelope.live_termination_reason_order_blockers(
        ACTION_BALL,
        config_path=config,
        base_config_path=base,
        mirrored_active_order=vec_env.EXACT_ACTIVE_TERMINATION_REASON_ORDER,
        mirrored_hard_order=vec_env.EXACT_HARD_TERMINATION_REASON_ORDER,
        mirrored_partition={
            "phase_fidelity": vec_env.EXACT_PHASE_FIDELITY_REASON_ORDER,
            "base_and_joint": vec_env.EXACT_BASE_TERMINATION_REASON_ORDER,
            "table_guard": vec_env.EXACT_TABLE_GUARD_REASON_ORDER,
        },
    )


# ---------------------------------------------------------------------------
# What the live order actually is
# ---------------------------------------------------------------------------


def test_the_replica_reason_order_is_the_live_three_class_chain_order():
    assert _order() == vec_env.EXACT_ACTIVE_TERMINATION_REASON_ORDER
    assert _blockers() == ()

    # The chain really does leave hope_env_cfg.py -- if it did not, this whole
    # module would be testing a one-file question and could not see the hole.
    chain = envelope.live_chain_sources(ACTION_BALL)
    assert tuple(name for name, _ in chain) == (ACTION_BALL, DEPLOY_PARITY, BASE_CLASS)
    assert chain[0][1] == chain[1][1] != chain[2][1]
    assert chain[2][1].name == "tracking_env_cfg.py"


def test_the_head_of_the_order_is_supplied_by_the_out_of_file_grandparent():
    grandparent = envelope.live_declared_terms(BASE_CLASS)
    assert _order()[: len(grandparent)] == grandparent
    # ...and the only truncation term lives there too, so "hard" is the tail.
    assert envelope.live_timeout_term_names() == frozenset({"time_out"})
    assert vec_env.EXACT_HARD_TERMINATION_REASON_ORDER == tuple(
        term for term in _order() if term != "time_out"
    )


def test_an_override_keeps_the_slot_its_base_gave_it():
    """人话:子类覆写 ``ee_body_pos`` 没有把它挪到队尾,它还在父类给的第 4 格。

    这一条是 dataclass 字段顺序的核心,也是最容易在复刻里想当然写错的一步:
    "子类新加的排后面"是对的,"子类覆写的也排后面"是错的。
    """

    order = _order()
    assert "ee_body_pos" in envelope.live_declared_terms(ACTION_BALL)
    assert order.index("ee_body_pos") < order.index("base_fell_tilt")
    assert order.index("joint_qdes_forbidden") > order.index("robot_hit_table")


# ---------------------------------------------------------------------------
# Mutations: each one is invisible to a coarser check
# ---------------------------------------------------------------------------


def test_swapping_two_terms_in_the_grandparent_is_refused(tmp_path, monkeypatch):
    """换位:集合一样、数量一样、每条项的字节一样 —— 只有位置变了。"""

    mutated = _mutate(
        tmp_path,
        envelope.ISAAC_BASE_TERMINATION_CONFIG,
        LIVE_ANCHOR_POS_BLOCK + LIVE_ANCHOR_ORI_BLOCK,
        LIVE_ANCHOR_ORI_BLOCK + LIVE_ANCHOR_POS_BLOCK,
    )

    live = envelope.live_declared_terms(BASE_CLASS, base_config_path=mutated)
    original = envelope.live_declared_terms(BASE_CLASS)
    # A set check is blind.  A count check is blind.  Both asserted in-test, so
    # this is not a claim about the gate -- it is a fact about the mutation.
    assert set(live) == set(original) and len(live) == len(original)
    assert live != original

    blockers = _blockers(base=mutated)
    assert any("isaac_active_reason_order_differs" in item for item in blockers)

    # 旧选择器只点了 time_out,所以它的指纹**一个 bit 都没动**。
    assert (
        vec_env._semantic_ast_sha256(mutated, OLD_NARROW_BASE_SELECTORS)
        == OLD_NARROW_BASE_SHA256
    )
    # 新选择器点了四条,同一份改动它就开火了。
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_BASE_CONFIG", mutated)
    _clear()
    with pytest.raises(
        vec_env.VecEnvContractError, match="base_config semantic AST SHA-256 drifted"
    ):
        vec_env.phase_fidelity_sample_contract()


def test_moving_a_term_from_the_parent_into_the_subclass_is_refused(
    tmp_path, monkeypatch
):
    """把 ``base_too_low`` 从父类搬进子类:名字、数量、集合统统不变,只有位置变了。

    这是最锋利的一个变异 —— ``live_declared_term_blockers`` 比的是**每个类的集合**,
    它会因为两个类的集合都变了而开火;但如果只看"整条链一共声明了哪些项",那是完全
    一样的。顺序这道门看的正是那个只有位置在变的差别。
    """

    without = _mutate(
        tmp_path,
        envelope.ISAAC_TERMINATION_CONFIG,
        LIVE_BASE_TOO_LOW_BLOCK,
        "",
    )
    text = without.read_text("utf-8")
    anchor = "    joint_actual_forbidden = DoneTerm("
    assert text.count(anchor) == 1
    moved = tmp_path / "moved_hope_env_cfg.py"
    moved.write_text(
        text.replace(anchor, LIVE_BASE_TOO_LOW_BLOCK + anchor, 1), "utf-8"
    )
    _clear()

    live_order = _order(config=moved)
    assert set(live_order) == set(_order())
    assert len(live_order) == len(_order())
    assert live_order != _order()
    # It moved from slot 5 to behind the table guard and the first joint term.
    assert live_order.index("base_too_low") > live_order.index("robot_hit_table")
    assert _order().index("base_too_low") < _order().index("robot_hit_table")

    blockers = _blockers(config=moved)
    assert any("isaac_active_reason_order_differs" in item for item in blockers)
    assert any("isaac_hard_reason_order_differs" in item for item in blockers)
    # The bucket that claims it is no longer in the live order either.
    assert any("isaac_reason_bucket_out_of_live_order" in item for item in blockers)


def test_a_new_term_in_the_grandparent_lands_in_no_replica_bucket(
    tmp_path, monkeypatch
):
    """往**根类**里新加一条终止项 —— 旧选择器天生看不见这个名字。"""

    mutated = _mutate(
        tmp_path,
        envelope.ISAAC_BASE_TERMINATION_CONFIG,
        LIVE_ANCHOR_POS_BLOCK,
        LIVE_ANCHOR_POS_BLOCK
        + "    base_out_of_bounds = DoneTerm(func=mdp.bad_anchor_pos_z_only)\n",
    )

    # 旧的窄选择器:指纹原地不动。
    assert (
        vec_env._semantic_ast_sha256(mutated, OLD_NARROW_BASE_SELECTORS)
        == OLD_NARROW_BASE_SHA256
    )
    # 就算有人把新选择器的指纹也重钉了,这两道值检查照样红。
    blockers = _blockers(base=mutated)
    assert any(
        "live_terms_in_no_replica_bucket=['base_out_of_bounds']" in item
        for item in blockers
    )
    declared = envelope.live_declared_term_blockers(base_config_path=mutated)
    assert any("base_out_of_bounds" in item for item in declared)


def test_flipping_the_truncation_flag_moves_a_term_into_the_hard_order(
    tmp_path,
):
    """``time_out=True`` -> ``False``:名字、顺序、数量全不变,只是不再算截断。

    "哪几条是硬终止"以前是隐含在复刻的元组里的常识;现在它从活的构造调用里读。
    """

    mutated = _mutate(
        tmp_path,
        envelope.ISAAC_BASE_TERMINATION_CONFIG,
        "    time_out = DoneTerm(func=mdp.time_out, time_out=True)\n",
        "    time_out = DoneTerm(func=mdp.time_out, time_out=False)\n",
    )
    assert _order(base=mutated) == _order()  # order itself is untouched
    assert envelope.live_timeout_term_names(base_config_path=mutated) == frozenset()

    blockers = _blockers(base=mutated)
    assert any("isaac_hard_reason_order_differs" in item for item in blockers)
    assert any(
        "live_terms_in_no_replica_bucket=['time_out']" in item for item in blockers
    )


def test_a_base_this_reader_was_not_pointed_at_fails_closed(tmp_path):
    """把根类改名 = 这条链断了。断链必须报错,不能悄悄"到此为止"。

    这正是修复前的行为:出了文件的基类直接结束遍历,于是整个头部无人看管。
    """

    mutated = _mutate(
        tmp_path,
        envelope.ISAAC_TERMINATION_CONFIG,
        f"class {DEPLOY_PARITY}({BASE_CLASS}):",
        f"class {DEPLOY_PARITY}(SomeOtherTerminationsCfg):",
    )
    with pytest.raises(
        envelope.IsaacReferenceEnvelopeError, match="EXTERNAL_TERMINATION_BASES"
    ):
        envelope.live_class_chain(ACTION_BALL, mutated)
    blockers = _blockers(config=mutated)
    assert any("isaac_reason_order_unreadable" in item for item in blockers)


def test_unrelated_edits_to_the_base_config_do_not_fire(tmp_path, monkeypatch):
    """负对照:根类所在文件的别处怎么改,这道门都不该响。"""

    expected = vec_env.phase_fidelity_sample_contract()
    source = envelope.ISAAC_BASE_TERMINATION_CONFIG
    unrelated = tmp_path / source.name
    unrelated.write_text(
        source.read_text("utf-8")
        + "\n\ndef unrelated_reason_order_probe():\n    return 23\n",
        "utf-8",
    )
    _clear()
    assert _blockers(base=unrelated) == ()

    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_BASE_CONFIG", unrelated)
    _clear()
    assert vec_env.phase_fidelity_sample_contract() == expected


def test_the_receipt_says_which_chain_it_compared():
    contract = vec_env.phase_fidelity_sample_contract()
    assert contract["live_reason_order_class_chain"] == [
        ACTION_BALL,
        DEPLOY_PARITY,
        BASE_CLASS,
    ]
    assert "partition" in contract["live_reason_order_compared"]
    assert sorted(contract["live_declared_terms_compared"]) == sorted(
        envelope.DECLARED_TERMS
    )
