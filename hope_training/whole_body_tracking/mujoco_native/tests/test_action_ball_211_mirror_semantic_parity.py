"""The native 211 mirror must equal the LIVE Isaac leaf, not just its old SHA.

人话:``A211_SOURCE_SHA256`` 只证明源文件字节没动过。源文件一动,把这行 SHA 重钉成
新值是一行的事,而 mujoco_native 里那份手抄的身份串 / 有序布局到底跟没跟上,过去
没有任何一条测试在看 —— 5ed998f1 就是这么让 table 复刻停在原地两天的。

这个模块把手抄件逐符号跟活的叶子对一遍,并且每条检查都配一个"旧检查抓不到"的
变异用例:变异后的源文件总宽度仍是 211/319、名字集合一字不差、模块自带的
``assert`` 照样通过 —— 只有逐行有序比较才看得见。
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from hope_training.whole_body_tracking.mujoco_native import action_ball_211_abi as abi
from hope_training.whole_body_tracking.mujoco_native.scripts import (
    launch_mujoco_action_ball_211_diagnostic as shared,
)


REPO_ROOT = Path(abi.__file__).resolve().parents[3]
SOURCE_DIR = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking"
)
LIVE_SOURCES = {
    "A211": SOURCE_DIR / "action_ball_a211_trainability.py",
    "C211": SOURCE_DIR / "action_ball_c211_trainability.py",
}


def _mutated_copy(tmp_path: Path, label: str, old: str, new: str) -> Path:
    """Copy one live leaf with a single textual substitution applied."""

    source = LIVE_SOURCES[label]
    text = source.read_text("utf-8")
    assert text.count(old) == 1, f"mutation anchor is not unique: {old!r}"
    target = tmp_path / source.name
    target.write_text(text.replace(old, new), "utf-8")
    if label == "C211":
        # C211 loads its A sibling from its OWN directory, so the copy needs one.
        (tmp_path / LIVE_SOURCES["A211"].name).write_text(
            LIVE_SOURCES["A211"].read_text("utf-8"), "utf-8"
        )
    return target


# ---------------------------------------------------------------------------
# The parity itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_mirror_matches_the_live_isaac_leaf_symbol_by_symbol(label: str) -> None:
    profile = abi.PROFILES[label]
    module = abi.load_live_trainability_module(LIVE_SOURCES[label])

    for attribute, suffix in abi.MIRRORED_IDENTITY_SYMBOLS:
        assert getattr(profile, attribute) == getattr(module, f"{label}_{suffix}")

    live_actor = tuple(
        (str(name), int(width))
        for name, width in getattr(module, f"{label}_ACTOR_LAYOUT")
    )
    live_critic = tuple(
        (str(name), int(width))
        for name, width in getattr(module, f"{label}_CRITIC_LAYOUT")
    )
    assert profile.actor.layout == live_actor
    assert profile.critic.layout == live_critic
    assert profile.actor.width == getattr(module, f"{label}_ACTOR_WIDTH") == 211
    assert profile.critic.width == getattr(module, f"{label}_CRITIC_WIDTH") == 319

    assert abi.live_source_parity_blockers(profile, LIVE_SOURCES[label]) == ()


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_wait_mask_is_exactly_the_live_layout_task_tail_block(label: str) -> None:
    """The 13-D RESET_WAIT mask is derived from the live rows, not hardcoded."""

    profile = abi.PROFILES[label]
    module = abi.load_live_trainability_module(LIVE_SOURCES[label])
    for lane, suffix in ((profile.actor, "ACTOR"), (profile.critic, "CRITIC")):
        live_layout = tuple(
            (str(name), int(width))
            for name, width in getattr(module, f"{label}_{suffix}_LAYOUT")
        )
        tail = abi._wait_mask_tail_names(live_layout)
        masked = tuple(
            field.name for field in lane.fields if field.mask_when_task_invalid
        )
        assert tail == masked
        assert sum(dict(live_layout)[name] for name in tail) == 13


def test_live_sources_still_hash_to_their_pins() -> None:
    for label, path in LIVE_SOURCES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == abi.PROFILES[label].source_sha256, label


# ---------------------------------------------------------------------------
# Mutation tests: each one is invisible to width-only / name-set-only checks
# ---------------------------------------------------------------------------


def test_unmutated_copy_at_another_path_is_clean() -> None:
    """Non-vacuity floor: the loader itself must not be what fails below."""

    for label, path in LIVE_SOURCES.items():
        assert abi.live_source_parity_blockers(abi.PROFILES[label], path) == ()


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_same_width_row_swap_is_caught(tmp_path: Path, label: str) -> None:
    """Swap the two 1-D clocks: total width, name set and every row width equal.

    人话:把两个 1 维时钟对调。总宽度 211 没变、名字一个不少、每行宽度也都对得上,
    连源文件自己的 ``assert ... == 211`` 都照过。只有"逐行有序"才看得出来。
    """

    mutated = _mutated_copy(
        tmp_path,
        label,
        '    ("time_to_contact", 1),\n    ("time_to_teacher_start", 1),\n'
        '    ("task_valid", 1),\n)\n\n'
        f"{label}_CRITIC_LAYOUT",
        '    ("time_to_teacher_start", 1),\n    ("time_to_contact", 1),\n'
        '    ("task_valid", 1),\n)\n\n'
        f"{label}_CRITIC_LAYOUT",
    )
    module = abi.load_live_trainability_module(mutated)
    # The mutation is genuinely invisible to the coarse checks.
    assert getattr(module, f"{label}_ACTOR_WIDTH") == 211
    assert sorted(name for name, _ in getattr(module, f"{label}_ACTOR_LAYOUT")) == (
        sorted(name for name, _ in abi.PROFILES[label].actor.layout)
    )

    blockers = abi.live_source_parity_blockers(abi.PROFILES[label], mutated)
    assert any("actor_layout_differs" in blocker for blocker in blockers), blockers


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_dimension_moved_between_neighbour_rows_is_caught(
    tmp_path: Path, label: str
) -> None:
    """Move one dimension from ``joint_pos`` to ``joint_vel``.

    人话:把一维从 joint_pos 挪到 joint_vel。总宽度不变、行数不变、顺序不变、
    名字不变 —— 任何只看总宽或只看名字顺序的检查都会放行,读出来的却是错位的
    31/31 边界。
    """

    mutated = _mutated_copy(
        tmp_path,
        label,
        '    ("joint_pos", 31),\n    ("joint_vel", 31),\n'
        '    ("actions", 31),\n    ("racket_site_achieved_now_heading", 9),',
        '    ("joint_pos", 30),\n    ("joint_vel", 32),\n'
        '    ("actions", 31),\n    ("racket_site_achieved_now_heading", 9),',
    )
    module = abi.load_live_trainability_module(mutated)
    assert getattr(module, f"{label}_ACTOR_WIDTH") == 211
    assert tuple(
        name for name, _ in getattr(module, f"{label}_ACTOR_LAYOUT")
    ) == tuple(name for name, _ in abi.PROFILES[label].actor.layout)

    blockers = abi.live_source_parity_blockers(abi.PROFILES[label], mutated)
    assert any("actor_layout_differs" in blocker for blocker in blockers), blockers


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_task_row_leaving_the_masked_tail_is_caught(
    tmp_path: Path, label: str
) -> None:
    """Hoist ``desired_base_xy_world`` above the mimic anchor row.

    Total width, row count and name set are all unchanged, but the block that
    RESET_WAIT is allowed to hide shrinks from 6 rows to 5 while the mirror
    still hides 6 -- i.e. the mirror would blank a row the live leaf keeps
    visible during WAIT.
    """

    # Actor lane only, in two steps: drop the row out of the tail, then
    # re-insert it above the mimic anchor.  Same rows, same total width, but
    # the tail block the WAIT mask may hide is now one row shorter.
    mutated = _mutated_copy(
        tmp_path,
        label,
        '    ("desired_base_xy_world", 2),\n    ("time_to_contact", 1),\n'
        '    ("time_to_teacher_start", 1),\n    ("task_valid", 1),\n)\n\n'
        f"{label}_CRITIC_LAYOUT",
        '    ("time_to_contact", 1),\n'
        '    ("time_to_teacher_start", 1),\n    ("task_valid", 1),\n)\n\n'
        f"{label}_CRITIC_LAYOUT",
    )
    text = mutated.read_text("utf-8")
    anchor = '    ("racket_site_teacher_at_reference_hit_heading", 9),\n'
    assert text.count(anchor) == 2, "actor and critic each carry the anchor row"
    head, _, rest = text.partition(anchor)
    mutated.write_text(
        head + '    ("desired_base_xy_world", 2),\n' + anchor + rest, "utf-8"
    )

    module = abi.load_live_trainability_module(mutated)
    assert getattr(module, f"{label}_ACTOR_WIDTH") == 211
    assert sorted(name for name, _ in getattr(module, f"{label}_ACTOR_LAYOUT")) == (
        sorted(name for name, _ in abi.PROFILES[label].actor.layout)
    )

    blockers = abi.live_source_parity_blockers(abi.PROFILES[label], mutated)
    assert any("actor_wait_mask_differs" in blocker for blocker in blockers), blockers


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_normalizer_identity_bump_is_caught(tmp_path: Path, label: str) -> None:
    """A fresh-normalizer rename must not be re-pinnable without porting it."""

    lowered = label.lower()
    mutated = _mutated_copy(
        tmp_path,
        label,
        f'{label}_ACTOR_NORMALIZER_IDENTITY = "action_ball_{lowered}_actor_norm_v2"',
        f'{label}_ACTOR_NORMALIZER_IDENTITY = "action_ball_{lowered}_actor_norm_v3"',
    )
    blockers = abi.live_source_parity_blockers(abi.PROFILES[label], mutated)
    assert any(
        "actor_normalizer_identity_differs" in blocker for blocker in blockers
    ), blockers


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_absent_or_unloadable_live_leaf_fails_closed(
    tmp_path: Path, label: str
) -> None:
    missing = tmp_path / "not_here.py"
    blockers = abi.live_source_parity_blockers(abi.PROFILES[label], missing)
    assert blockers and "unloadable" in blockers[0]


# ---------------------------------------------------------------------------
# The regression this whole module exists for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_repinning_the_sha_alone_no_longer_lets_layout_drift_through(
    tmp_path: Path, label: str
) -> None:
    """Re-stamp the pin onto a drifted source: the launcher must STILL refuse.

    人话:这是 5ed998f1 的复现。当时的修法是"源文件动了 -> 把指纹重钉一下",
    镜像没跟。这条用例把重钉那一步替这个假想的粗心作者做完 —— SHA 门此刻完全
    通过 —— 然后要求语义门照样拦下来。
    """

    mutated = _mutated_copy(
        tmp_path,
        label,
        '    ("time_to_contact", 1),\n    ("time_to_teacher_start", 1),\n'
        '    ("task_valid", 1),\n)\n\n'
        f"{label}_CRITIC_LAYOUT",
        '    ("time_to_teacher_start", 1),\n    ("time_to_contact", 1),\n'
        '    ("task_valid", 1),\n)\n\n'
        f"{label}_CRITIC_LAYOUT",
    )
    repinned = replace(
        abi.PROFILES[label],
        source_sha256=hashlib.sha256(mutated.read_bytes()).hexdigest(),
    )
    # The SHA gate is now satisfied by construction.
    assert repinned.source_sha256 == hashlib.sha256(mutated.read_bytes()).hexdigest()
    assert abi.live_source_parity_blockers(repinned, mutated)


def test_source_lineage_self_declares_the_live_semantic_parity() -> None:
    """The receipt says which comparison actually ran, not just a digest."""

    for label in ("A211", "C211"):
        lineage = shared._source_lineage(abi.PROFILES[label])
        assert lineage["sha256"] == abi.PROFILES[label].source_sha256
        assert lineage["live_semantic_parity"] == (
            "exact_identities_ordered_layouts_widths_wait_mask"
        )
        assert lineage["live_semantic_parity_symbols_compared"] == "11"


@pytest.mark.parametrize("label", ("A211", "C211"))
def test_source_lineage_blocks_on_semantic_drift_even_when_both_shas_match(
    monkeypatch, label: str
) -> None:
    """Both digest gates pass; the launcher must still refuse to build an env."""

    monkeypatch.setattr(
        abi,
        "live_source_parity_blockers",
        lambda _profile, _path: (f"{label.lower()}_actor_layout_differs:synthetic",),
    )
    with pytest.raises(shared.LaunchBlocked, match="differ from the live Isaac"):
        shared._source_lineage(abi.PROFILES[label])
