"""通用护栏:新增一处"只有指纹罩着的手抄常量"这件事本身必须被测试发现。

前四次都是同一个形状:复刻侧的某个数/名单/顺序是从 Isaac 抄来的,唯一的保护是一枚
AST 指纹或 SHA 钉子,而**重钉一枚钉子是一行的事** —— 于是"字节没动"被当成了"抄对了"。
每次都是事后一处一处捞,因为从来没有一张"这个模块里到底有多少份手抄件"的清单。

这个模块测的是那张清单本身:

* 每个模块级常量都必须被显式分类。新加一个而不分类 —— 当场红。
* 自称"被活值罩着"的,测试会真的把常量的值跟**活值比对入口实际拿去比的那个值**对上;
  有人把比对改成拿另一个数去比(``5c4ced66`` 那个形状:比的是第三份手抄件),红。
* 自称"从活源读出来的",测试会检查它的赋值不是字面量;有人把活读"简化"回它当时返回的
  那个字面量(``5ed998f1`` 那个形状),红。
* 钉了文件摘要的,测试在 host 上重算一遍;编辑了文件忘了重钉,红 —— 不用烧一次 pod 时间。
* 真欠的债只能进 ``OPEN_MIRROR_DEBT``,并且强制写"真源在哪 / 怎么修 / 为什么这轮没修"。

护栏诚实的边界也在这里写清楚(见最后一个测试):它拦不住有人把一份新的手抄 Isaac 常量
硬标成 ``NOT_MIRRORED``。它保证的是**这件事必须有人动手写一行、署上一个理由档位**。
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WBT_ROOT = REPO_ROOT / "hope_training/whole_body_tracking"
if str(WBT_ROOT) not in sys.path:
    sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import mirrored_constant_registry as registry  # noqa: E402
from mujoco_native import table_termination  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


def _lane_copy(tmp_path: Path) -> Path:
    """A copy of every lane module, so a test can edit one for real."""

    target = tmp_path / "mujoco_native"
    target.mkdir()
    for path in registry.MODULE_DIR.glob("*.py"):
        shutil.copy2(path, target / path.name)
    return target


# ---------------------------------------------------------------------------
# The registry describes the lane it claims to describe
# ---------------------------------------------------------------------------


def test_every_module_level_constant_in_the_lane_is_classified():
    assert registry.registry_blockers() == ()


def test_the_registry_actually_covers_something():
    """负面对照的对照:如果这张表是空的,上面那条测试是永真的。"""

    receipt = registry.registry_receipt()
    assert receipt["modules_classified"] >= 15
    assert receipt["constants_classified"] >= 200
    counts = receipt["constants_by_reason"]
    # The strong classes must not be empty, or the guard is pure ceremony.
    assert counts[registry.LIVE_VALUE_COMPARED] >= 20
    assert counts[registry.PINNED_FILE_DIGEST] >= 4
    assert sum(counts.values()) == receipt["constants_classified"]


def test_the_registry_covers_every_file_in_the_lane():
    declared = set(registry.CLASSIFICATION) | set(registry.MODULES_WITHOUT_CONSTANTS)
    assert declared == set(registry.lane_modules())


# ---------------------------------------------------------------------------
# Mutations: the guard must actually fire
# ---------------------------------------------------------------------------


def test_a_brand_new_hand_copied_constant_is_refused(tmp_path, monkeypatch):
    """就是要求的那一条:新增一处只有指纹保护的手抄常量,测试必须红。"""

    lane = _lane_copy(tmp_path)
    target = lane / "vec_env.py"
    target.write_text(
        target.read_text("utf-8")
        + "\n\n# a hand copy someone pasted in from hope_env_cfg.py\n"
        "SOME_NEW_ISAAC_THRESHOLD_M = 0.42\n",
        "utf-8",
    )
    monkeypatch.setattr(registry, "MODULE_DIR", lane)

    blockers = registry.registry_blockers()
    assert any(
        "mirrored_constant_unclassified:vec_env.SOME_NEW_ISAAC_THRESHOLD_M" in item
        for item in blockers
    )


def test_a_brand_new_module_in_the_lane_is_refused(tmp_path, monkeypatch):
    lane = _lane_copy(tmp_path)
    (lane / "brand_new_mirror.py").write_text("ISAAC_SOMETHING = 1.5\n", "utf-8")
    monkeypatch.setattr(registry, "MODULE_DIR", lane)

    assert any(
        "mirror_module_unclassified:brand_new_mirror.py" in item
        for item in registry.registry_blockers()
    )


def test_a_lowercase_helper_does_not_need_classifying(tmp_path, monkeypatch):
    """负对照:普通小写模块级赋值不是常量,不该被这道门缠上。"""

    lane = _lane_copy(tmp_path)
    target = lane / "vec_env.py"
    target.write_text(
        target.read_text("utf-8") + "\n\n_unrelated_helper_cache = {}\n", "utf-8"
    )
    monkeypatch.setattr(registry, "MODULE_DIR", lane)
    assert registry.registry_blockers() == ()


def test_a_live_read_quietly_replaced_by_its_current_literal_is_refused(
    tmp_path, monkeypatch
):
    """``5ed998f1`` 的形状,在登记表这一层重放。

    有人觉得"每次 import 都去解析 Isaac 源文件太慢",把活读换成它当时返回的那个字面量。
    值在那一刻完全正确,指纹一个 bit 都不用动 —— 但复刻从此不再跟着上游走。
    """

    lane = _lane_copy(tmp_path)
    target = lane / "vec_env.py"
    text = target.read_text("utf-8")
    live_read = (
        "_ACTION_BALL_REFERENCE_ENVELOPE = isaac_reference_envelope"
        ".live_reference_envelope(\n"
        "    PHASE_TERMINATIONS_MIRRORED_ISAAC_CLASS\n"
        ")"
    )
    assert text.count(live_read) == 1
    frozen = (
        "_ACTION_BALL_REFERENCE_ENVELOPE = {\n"
        f"    'body_names': {vec_env.PHASE_EE_BODY_NAMES!r},\n"
        f"    'threshold_m': {vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M!r},\n"
        "}"
    )
    target.write_text(text.replace(live_read, frozen), "utf-8")
    monkeypatch.setattr(registry, "MODULE_DIR", lane)

    # The frozen literal is *correct today* -- assert that, so the test is about
    # the mechanism going away rather than about a wrong number.
    node = registry.module_level_constants("vec_env.py")[
        "_ACTION_BALL_REFERENCE_ENVELOPE"
    ]
    assert ast.literal_eval(node)["body_names"] == vec_env.PHASE_EE_BODY_NAMES

    assert any(
        "mirrored_constant_is_a_literal_after_all:vec_env."
        "_ACTION_BALL_REFERENCE_ENVELOPE" in item
        for item in registry.registry_blockers()
    )


def test_a_constant_that_stops_flowing_into_its_own_comparison_is_refused(
    monkeypatch,
):
    """``5c4ced66`` 的形状:门还在,但它比的已经不是这个常量了。

    这里把 ``phase_termination`` 那张表里 ``base_fell_tilt`` 那一条的镜像值换成另一个
    数,常量本身一个字节没动 —— 只有"到底拿谁去比"变了。登记表要能说出这句话。
    """

    real = vec_env.mirrored_isaac_termination_entries()

    def rewired():
        rows = []
        for entry in real:
            if entry[0] == "base_fell_tilt_limit_angle_rad":
                rows.append((*entry[:-1], 0.8))
            else:
                rows.append(entry)
        return tuple(rows)

    monkeypatch.setattr(
        vec_env, "mirrored_isaac_termination_entries", rewired
    )
    assert vec_env.BASE_FELL_TILT_LIMIT_ANGLE_RAD == 0.7  # untouched

    assert any(
        "mirrored_constant_not_the_compared_value:vec_env."
        "BASE_FELL_TILT_LIMIT_ANGLE_RAD" in item
        for item in registry.registry_blockers()
    )


def test_a_pinned_file_digest_that_stopped_matching_is_refused(
    tmp_path, monkeypatch
):
    """编辑了被钉的文件却忘了重钉 —— 以前只有 pod 上开起来才红。"""

    source = table_termination.CANONICAL_MUJOCO_IDENTITY_PY
    edited = tmp_path / source.name
    edited.write_text(source.read_text("utf-8") + "\n# an innocent comment\n", "utf-8")
    monkeypatch.setattr(table_termination, "CANONICAL_MUJOCO_IDENTITY_PY", edited)

    assert any(
        "mirrored_file_digest_differs:table_termination."
        "EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256" in item
        for item in registry.registry_blockers()
    )


def test_a_renamed_upstream_source_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vec_env,
        "TERMINATION_SOURCE_PHASE_GATE",
        tmp_path / "hope_commands_renamed_upstream.py",
    )
    assert any(
        "mirrored_source_path_absent:vec_env.TERMINATION_SOURCE_PHASE_GATE" in item
        for item in registry.registry_blockers()
    )


def test_debt_without_a_written_plan_is_refused(monkeypatch):
    trimmed = {
        key: value
        for key, value in registry.OPEN_MIRROR_DEBT.items()
        if key != "action_ball_c211_env.C211_ACTION_RATE_POST_DT_WEIGHT"
    }
    monkeypatch.setattr(registry, "OPEN_MIRROR_DEBT", trimmed)
    assert any(
        "mirrored_todo_without_a_written_plan:"
        "action_ball_c211_env.C211_ACTION_RATE_POST_DT_WEIGHT" in item
        for item in registry.registry_blockers()
    )


def test_debt_that_was_paid_but_left_behind_is_refused(monkeypatch):
    """反方向:债还了却忘了从清单里删,清单会开始骗人。"""

    stale = dict(registry.OPEN_MIRROR_DEBT)
    stale["vec_env.BASE_TOO_LOW_MINIMUM_HEIGHT_M"] = ("a", "b", "c")
    monkeypatch.setattr(registry, "OPEN_MIRROR_DEBT", stale)
    assert any(
        "mirror_debt_is_stale:vec_env.BASE_TOO_LOW_MINIMUM_HEIGHT_M" in item
        for item in registry.registry_blockers()
    )


def test_a_debt_note_that_says_nothing_is_refused(monkeypatch):
    hollow = dict(registry.OPEN_MIRROR_DEBT)
    hollow["action_ball_c211_env.C211_UPRIGHT_STD"] = ("", "", "")
    monkeypatch.setattr(registry, "OPEN_MIRROR_DEBT", hollow)
    assert any(
        "mirror_debt_incomplete:action_ball_c211_env.C211_UPRIGHT_STD" in item
        for item in registry.registry_blockers()
    )


def test_a_dead_live_value_provider_is_refused(monkeypatch):
    """接线断了(没有常量再引用某个 provider)也要说出来,不能静静烂着。"""

    providers = dict(registry.LIVE_VALUE_PROVIDERS)
    providers["never_referenced_provider"] = dict
    monkeypatch.setattr(registry, "LIVE_VALUE_PROVIDERS", providers)
    assert any(
        "mirrored_live_provider_unused:never_referenced_provider" in item
        for item in registry.registry_blockers()
    )


# ---------------------------------------------------------------------------
# What this guard does NOT promise
# ---------------------------------------------------------------------------


def test_the_guard_does_not_pretend_to_catch_a_dishonest_classification():
    """写清楚边界,免得下一个人以为这道门比它实际更强。

    把一份真手抄的 Isaac 常量标成 ``NOT_MIRRORED``,这道门是拦不住的 —— 它没有办法
    知道上游有没有同义的数。它能保证的是:这件事必须有人动手写一行、署上档位,而不是
    像前四次那样悄无声息地混进来。真正的语义防线是 ``LIVE_VALUE_COMPARED`` 那一档。
    """

    table = dict(registry.CLASSIFICATION["vec_env.py"])
    assert table["BASE_FELL_TILT_LIMIT_ANGLE_RAD"][0] == registry.LIVE_VALUE_COMPARED
    # A constant claiming NOT_MIRRORED carries no machine-checkable obligation.
    assert registry.NOT_MIRRORED not in registry.REASONS_REQUIRING_DETAIL
